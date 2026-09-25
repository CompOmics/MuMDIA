//! Reuse of a FASTA-built spectral library across runs (`predict_frag.library_cache`).
//!
//! A FASTA-mode `run` digests the FASTA, expands peptidoforms and predicts fragment
//! intensities and retention times on every invocation, and one `run` per file is the
//! documented way to search files separately, so each file paid the whole library build
//! again: about an hour of predict-frag on the 9.8M-peptidoform HYE library (perf survey
//! critic item 3). With `predict_frag.library_cache` set to a directory, the orchestrator
//! looks the library up there under a key of everything that determines it, publishes a
//! hit at the paths a build would have written, and stores a miss after building it.
//!
//! The key ([`LibraryCache::for_config`]) covers the FASTA's content hash, the `digest`,
//! `peptidoforms` and `predict_frag` sections (all but the cache directory itself), the
//! RNG seed that scrambles decoys, whether the iRT is a deferred DeepLC placeholder, the
//! installed MS2PIP, AlphaPeptDeep and DeepLC versions that the build would use, the
//! worker scripts' content, and the running executable's content, so a rebuilt engine or
//! an upgraded predictor never picks up an old library. When a version cannot be
//! determined the library is neither looked up nor stored, and the run builds as usual.
//!
//! The run stays in FASTA mode: every decision the orchestrator makes on "is this an
//! imported library" is the one a rebuild would make, and only the three build stages are
//! skipped. A hit is byte-identical to a rebuild when the build is deterministic, which
//! the native predictors are; a sidecar predictor's run-to-run variation is frozen at the
//! stored draw, as reusing any library would.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mumdia_core::config::{Config, FragPredictorKind, RtPredictorKind};
use mumdia_io::report::{ArtifactReport, Written};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Bumped when the layout of a cache entry or the key material changes, so an older
/// engine's entries are misses rather than misreadings.
const CACHE_FORMAT: u32 = 1;

/// The two library tables of a FASTA build, by their file names in the run directory.
const TABLES: [&str; 2] = [
    "fragment_library_precursors.parquet",
    "fragment_library_fragments.parquet",
];

/// How long a temporary store directory may go unwritten before the run that was
/// writing it counts as gone and the directory is removed. A live store writes its files
/// continuously, and a byte copy of the largest library takes minutes, not an hour.
const STALE_AFTER: std::time::Duration = std::time::Duration::from_secs(60 * 60);

/// One stored file in `entry.json`.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
struct StoredFile {
    bytes: u64,
    blake3: String,
}

/// `entry.json` of one cache entry: the key material and the size and content hash of
/// every stored file.
#[derive(Serialize, Deserialize)]
struct Entry {
    cache_format: u32,
    key: String,
    material: serde_json::Value,
    /// File name (a table or its `.report.json`) to its size and blake3.
    files: std::collections::BTreeMap<String, StoredFile>,
}

/// The four file names an entry holds: each table and its report.
fn entry_files() -> Vec<String> {
    TABLES
        .iter()
        .flat_map(|t| [t.to_string(), format!("{t}.report.json")])
        .collect()
}

/// A stored file's size and content hash.
fn stored_file(path: &Path) -> Result<StoredFile> {
    let p = path.to_string_lossy();
    Ok(StoredFile {
        bytes: std::fs::metadata(path)
            .with_context(|| format!("reading {p}"))?
            .len(),
        blake3: mumdia_io::hash::blake3_file(&p)?,
    })
}

/// A temporary directory this cache writes or sets aside: `<24 hex>.partial-...` (a store
/// in progress, or abandoned) or `<24 hex>.broken-...` (an unusable entry moved out of the
/// way). Anything else in the cache directory is not the engine's and is never touched.
fn leftover_kind(name: &str) -> Option<&'static str> {
    let (key, rest) = name.split_at_checked(24)?;
    if !key.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    if rest.starts_with(".partial-") {
        Some("partial")
    } else if rest.starts_with(".broken-") {
        Some("broken")
    } else {
        None
    }
}

/// Was `dir`, or any file directly inside it, written within [`STALE_AFTER`]? A
/// modification time in the future counts as recent (clock skew on a shared cache is not
/// evidence that nobody is writing, docs/31 F9).
fn written_recently(dir: &Path) -> bool {
    let recent = |p: &Path| match std::fs::metadata(p).and_then(|m| m.modified()) {
        Ok(t) => match t.elapsed() {
            Ok(age) => age < STALE_AFTER,
            Err(_) => true,
        },
        Err(_) => false,
    };
    recent(dir)
        || std::fs::read_dir(dir)
            .map(|rd| rd.flatten().any(|e| recent(&e.path())))
            .unwrap_or(false)
}

/// A name for a directory beside the entries that no other run picks: the key, `what`,
/// this process's id and the time in nanoseconds.
fn unique_sibling(dir: &Path, key: &str, what: &str) -> PathBuf {
    dir.join(format!(
        "{key}.{what}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

/// Remove the digest and peptidoform tables (and their reports) a previous build left in
/// `out_dir`, after a cache hit published a library they did not produce. A missing file
/// is the normal case; any other failure is a warning.
pub fn remove_build_intermediates(out_dir: &str) {
    for name in [
        "peptides.parquet",
        "peptides.parquet.report.json",
        "peptidoforms.parquet",
        "peptidoforms.parquet.report.json",
    ] {
        let path = Path::new(out_dir).join(name);
        match std::fs::remove_file(&path) {
            Ok(()) => info!(
                file = %path.display(),
                "library cache: removed an earlier build's table, which did not produce the \
                 reused library"
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => warn!(
                file = %path.display(),
                error = %e,
                "library cache: could not remove an earlier build's table beside the reused \
                 library"
            ),
        }
    }
}

/// One library's place in the cache.
pub struct LibraryCache {
    dir: PathBuf,
    key: String,
    material: serde_json::Value,
}

impl LibraryCache {
    /// The cache entry for a FASTA build under `cfg`, or `None` when no cache is
    /// configured or the key cannot be determined (the reason is logged).
    ///
    /// `engine` is `(version, git_sha)` from the manifest, recorded for a reader of
    /// `entry.json`; the executable's own hash is what makes the key safe for a dirty or
    /// unversioned build.
    pub fn for_config(
        cfg: &Config,
        fasta: &str,
        rt_placeholder: bool,
        engine: (&str, &str),
    ) -> Option<Self> {
        let dir = cfg.predict_frag.library_cache.as_deref()?;
        match key_material(cfg, fasta, rt_placeholder, engine) {
            Ok(material) => {
                let text = serde_json::to_string(&material).expect("key material serializes");
                let key = mumdia_io::hash::blake3_str(&text)[..24].to_string();
                Some(Self {
                    dir: PathBuf::from(dir),
                    key,
                    material,
                })
            }
            Err(e) => {
                warn!(
                    error = %format!("{e:#}"),
                    "library cache: cannot key this library, so it is built and not stored \
                     (predict_frag.library_cache)"
                );
                None
            }
        }
    }

    /// The entry's directory.
    fn entry_dir(&self) -> PathBuf {
        self.dir.join(&self.key)
    }

    /// The entry at `entry_dir`, if it is usable: an `entry.json` of this format and key
    /// that lists exactly the four files, each present at its stored size. With `deep`,
    /// every file's content hash is checked as well.
    fn validate(&self, entry_dir: &Path, deep: bool) -> Result<Entry> {
        let entry_path = entry_dir.join("entry.json");
        let entry: Entry = mumdia_io::json::read_json(&entry_path.to_string_lossy())?;
        anyhow::ensure!(
            entry.cache_format == CACHE_FORMAT && entry.key == self.key,
            "entry.json is for format {} key {}",
            entry.cache_format,
            entry.key
        );
        let want: std::collections::BTreeSet<String> = entry_files().into_iter().collect();
        anyhow::ensure!(
            entry
                .files
                .keys()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
                == want,
            "entry.json lists {:?}, not the two tables and their reports",
            entry.files.keys().collect::<Vec<_>>()
        );
        for (name, stored) in &entry.files {
            let path = entry_dir.join(name);
            let got = std::fs::metadata(&path)
                .with_context(|| format!("stored file {name}"))?
                .len();
            anyhow::ensure!(
                got == stored.bytes,
                "{name} is {got} bytes, stored at {}; it was changed after it was stored",
                stored.bytes
            );
            if deep {
                let h = mumdia_io::hash::blake3_file(&path.to_string_lossy())?;
                anyhow::ensure!(
                    h == stored.blake3,
                    "{name} no longer has the content it was stored with"
                );
            }
        }
        Ok(entry)
    }

    /// On a hit, publish a byte copy of the stored tables and their reports at `lib_p` and
    /// `lib_f` and return what their reports record. `None` is a miss: no entry, or one
    /// whose files are missing or not the size or content they were stored with, which is
    /// logged and then rebuilt over (and replaced by [`LibraryCache::store`]).
    ///
    /// A copy, not a hard link, so a tool that rewrites a run directory's library in place
    /// changes neither the cache nor another run. Each published copy is hashed against
    /// `entry.json`, which is one read of a file that is still in the page cache.
    pub fn restore(&self, lib_p: &str, lib_f: &str) -> Option<(Written, Written)> {
        let entry_dir = self.entry_dir();
        if !entry_dir.join("entry.json").is_file() {
            info!(
                key = %self.key,
                cache = %self.dir.display(),
                "library cache: no stored library for this FASTA and configuration; building it"
            );
            return None;
        }
        match self.try_restore(&entry_dir, lib_p, lib_f) {
            Ok(w) => {
                info!(
                    key = %self.key,
                    entry = %entry_dir.display(),
                    precursors = w.0.rows,
                    fragments = w.1.rows,
                    "library cache: reusing the stored library; digest, peptidoforms and \
                     predict-frag are skipped (predict_frag.library_cache)"
                );
                Some(w)
            }
            Err(e) => {
                warn!(
                    key = %self.key,
                    entry = %entry_dir.display(),
                    error = %format!("{e:#}"),
                    "library cache: the stored library is unusable; rebuilding it, and the \
                     rebuild replaces the entry"
                );
                None
            }
        }
    }

    fn try_restore(
        &self,
        entry_dir: &Path,
        lib_p: &str,
        lib_f: &str,
    ) -> Result<(Written, Written)> {
        let entry = self.validate(entry_dir, false)?;
        let mut written = Vec::with_capacity(2);
        for (table, out) in TABLES.iter().zip([lib_p, lib_f]) {
            let out_report = format!("{out}.report.json");
            for (name, dest) in [
                (table.to_string(), out.to_string()),
                (format!("{table}.report.json"), out_report.clone()),
            ] {
                let src = entry_dir.join(&name);
                mumdia_io::table::publish_byte_copy(&src.to_string_lossy(), &dest)?;
                let got = mumdia_io::hash::blake3_file(&dest)?;
                anyhow::ensure!(
                    got == entry.files[&name].blake3,
                    "{name} no longer has the content it was stored with"
                );
            }
            let report: ArtifactReport = mumdia_io::json::read_json(&out_report)?;
            written.push(report.written());
        }
        let frag = written.pop().expect("two tables");
        let prec = written.pop().expect("two tables");
        Ok((prec, frag))
    }

    /// Store a library just built at `lib_p` and `lib_f` (with their reports). Best
    /// effort: a failure is a warning, since the run's own library is complete either way.
    ///
    /// The files are byte-copied into a private temporary directory beside the entry,
    /// which is then renamed into place, so a reader sees a complete entry or none and the
    /// entry never shares a file with a run directory. When another run stored a usable
    /// copy of the same key first, this copy is discarded; an unusable entry is moved aside
    /// and replaced. Temporary directories that a killed run left behind are removed first.
    pub fn store(&self, lib_p: &str, lib_f: &str) {
        match self.try_store(lib_p, lib_f) {
            Ok(true) => info!(
                key = %self.key,
                entry = %self.entry_dir().display(),
                "library cache: stored this library for later runs (predict_frag.library_cache)"
            ),
            Ok(false) => info!(
                key = %self.key,
                "library cache: another run stored this library first; keeping that copy"
            ),
            Err(e) => warn!(
                key = %self.key,
                cache = %self.dir.display(),
                error = %format!("{e:#}"),
                "library cache: could not store this library; later runs will build it again"
            ),
        }
    }

    /// Remove the temporary directories of stores that died (unwritten for
    /// [`STALE_AFTER`]) and of unusable entries set aside by [`LibraryCache::store`].
    fn sweep_leftovers(&self) {
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return;
        };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let path = e.path();
            let Some(kind) = leftover_kind(&name) else {
                continue;
            };
            if !path.is_dir() || (kind == "partial" && written_recently(&path)) {
                continue;
            }
            match std::fs::remove_dir_all(&path) {
                Ok(()) => info!(
                    dir = %path.display(),
                    "library cache: removed an abandoned {kind} directory"
                ),
                Err(err) => warn!(
                    dir = %path.display(),
                    error = %err,
                    "library cache: could not remove an abandoned {kind} directory"
                ),
            }
        }
    }

    /// Move an unusable entry out of the way and remove it. Another run may have done
    /// so already, which is not an error.
    fn set_aside(&self, final_dir: &Path, why: &anyhow::Error) {
        warn!(
            key = %self.key,
            entry = %final_dir.display(),
            error = %format!("{why:#}"),
            "library cache: the stored library is unusable; replacing it with this build"
        );
        let aside = unique_sibling(&self.dir, &self.key, "broken");
        if std::fs::rename(final_dir, &aside).is_ok() {
            // A reader on a platform that keeps open files undeletable can make this fail;
            // the next store's sweep retries it.
            let _ = std::fs::remove_dir_all(&aside);
        }
    }

    fn try_store(&self, lib_p: &str, lib_f: &str) -> Result<bool> {
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating {}", self.dir.display()))?;
        self.sweep_leftovers();
        let final_dir = self.entry_dir();
        if final_dir.exists() {
            // The run that reaches here built its library for an hour, so checking every
            // stored file's content is cheap; a same-size change would otherwise never be
            // repaired.
            match self.validate(&final_dir, true) {
                Ok(_) => return Ok(false),
                Err(e) => self.set_aside(&final_dir, &e),
            }
        }
        let tmp = unique_sibling(&self.dir, &self.key, "partial");
        std::fs::create_dir_all(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        let result = (|| -> Result<bool> {
            let mut files = std::collections::BTreeMap::new();
            for (table, src) in TABLES.iter().zip([lib_p, lib_f]) {
                for (name, from) in [
                    (table.to_string(), src.to_string()),
                    (format!("{table}.report.json"), format!("{src}.report.json")),
                ] {
                    let to = tmp.join(&name);
                    mumdia_io::table::publish_byte_copy(&from, &to.to_string_lossy())?;
                    files.insert(name, stored_file(&to)?);
                }
            }
            let entry = Entry {
                cache_format: CACHE_FORMAT,
                key: self.key.clone(),
                material: self.material.clone(),
                files,
            };
            mumdia_io::json::write_json(&tmp.join("entry.json").to_string_lossy(), &entry)?;
            match std::fs::rename(&tmp, &final_dir) {
                Ok(()) => Ok(true),
                // Another run renamed its copy into place between the check and here.
                Err(_) if self.validate(&final_dir, false).is_ok() => Ok(false),
                Err(e) => Err(e).with_context(|| {
                    format!("renaming {} to {}", tmp.display(), final_dir.display())
                }),
            }
        })();
        if !matches!(result, Ok(true)) {
            let _ = std::fs::remove_dir_all(&tmp);
        }
        result
    }
}

/// Everything that determines a FASTA build's library, as JSON.
fn key_material(
    cfg: &Config,
    fasta: &str,
    rt_placeholder: bool,
    engine: (&str, &str),
) -> Result<serde_json::Value> {
    let pf = &cfg.predict_frag;
    let mut predict_frag = serde_json::to_value(pf)?;
    if let Some(o) = predict_frag.as_object_mut() {
        // Where the library is kept does not change it.
        o.remove("library_cache");
    }
    let version = |python: &Option<String>, module: &str, field: &str| -> Result<String> {
        let py = python
            .as_deref()
            .with_context(|| format!("predict_frag.{field} is not set"))?;
        let v = crate::sidecar::module_version(py, module)
            .with_context(|| format!("cannot read the installed {module} version through {py}"))?;
        Ok(format!("{module}-{v}"))
    };
    let mut workers: Vec<&str> = Vec::new();
    let fragment_model = match pf.predictor {
        FragPredictorKind::Native => "native".to_string(),
        FragPredictorKind::Ms2pip => {
            workers.push("ms2pip_worker.py");
            version(&pf.ms2pip_python, "ms2pip", "ms2pip_python")?
        }
        FragPredictorKind::Peptdeep => {
            workers.push("peptdeep_worker.py");
            version(&pf.peptdeep_python, "peptdeep", "peptdeep_python")?
        }
    };
    let rt_model = match pf.rt_predictor {
        RtPredictorKind::Native => "native".to_string(),
        RtPredictorKind::Deeplc if rt_placeholder => "deferred-placeholder".to_string(),
        RtPredictorKind::Deeplc => {
            workers.push("deeplc_worker.py");
            version(&pf.deeplc_python, "deeplc", "deeplc_python")?
        }
    };
    let mut worker_hashes = serde_json::Map::new();
    for w in workers {
        let path = crate::sidecar::resolve_script(&pf.sidecar_script_dir, w);
        let h = mumdia_io::hash::blake3_file(&path)
            .with_context(|| format!("hashing the worker script {path}"))?;
        worker_hashes.insert(w.to_string(), serde_json::Value::String(h));
    }
    let exe = std::env::current_exe().context("locating the running executable")?;
    let exe_hash = mumdia_io::hash::blake3_file(&exe.to_string_lossy())
        .with_context(|| format!("hashing the running executable {}", exe.display()))?;
    let fasta_hash = mumdia_io::hash::blake3_file(fasta)?;
    Ok(serde_json::json!({
        "cache_format": CACHE_FORMAT,
        "fasta_blake3": fasta_hash,
        "rng_seed": cfg.rng_seed,
        "digest": serde_json::to_value(&cfg.digest)?,
        "peptidoforms": serde_json::to_value(&cfg.peptidoforms)?,
        "predict_frag": predict_frag,
        "rt_placeholder": rt_placeholder,
        "models": { "fragment": fragment_model, "rt": rt_model },
        "workers": worker_hashes,
        "engine": { "version": engine.0, "git_sha": engine.1, "exe_blake3": exe_hash },
    }))
}

/// The `--lib-*` invocation that searches another file against a library this run just
/// built from the FASTA, logged when no cache is configured, or `None` when reusing it
/// that way would not reproduce this run.
///
/// Library-input mode decides the retention-time handling differently from FASTA mode
/// (`Config::deeplc_rt_source`, `RtImTrainConfig::repredicts_library_irt`), so the line
/// names the `rt_im_train.library_irt` value under which the two agree: `deeplc` when
/// this run's multi-head calibration is on DeepLC retention times, `library` otherwise
/// (no base-model re-prediction, the same head count). A library whose iRT is a deferred
/// DeepLC placeholder must not be reused as a library at all, and gets no line.
pub fn reuse_hint(cfg: &Config, lib_p: &str, lib_f: &str, rt_placeholder: bool) -> Option<String> {
    if rt_placeholder {
        return None;
    }
    let has_deeplc = cfg.predict_frag.deeplc_python.is_some();
    let heads = cfg
        .rt_im_train
        .multihead_heads(has_deeplc, cfg.deeplc_rt_source(false, has_deeplc));
    let library_irt = if heads > 0 && cfg.predict_frag.rt_predictor == RtPredictorKind::Deeplc {
        "deeplc"
    } else {
        "library"
    };
    Some(format!(
        "--lib-precursors {lib_p} --lib-fragments {lib_f} (in place of --fasta) with \
         rt_im_train.library_irt = \"{library_irt}\""
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia_libcache_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn fake_build(dir: &Path, tag: &str) -> (String, String) {
        std::fs::create_dir_all(dir).unwrap();
        let mut out = Vec::new();
        for (i, t) in TABLES.iter().enumerate() {
            let p = dir.join(t);
            // Published by rename, as every engine writer does.
            let tmp = dir.join(format!("{t}.tmp"));
            std::fs::write(&tmp, format!("{tag}-{i}").repeat(100)).unwrap();
            std::fs::rename(&tmp, &p).unwrap();
            let report = ArtifactReport {
                logical_name: t.to_string(),
                schema_name: t.to_string(),
                schema_version: 1,
                stage: "predict-frag".into(),
                rows: 10 + i as u64,
                content_hash: format!("hash-{tag}-{i}"),
                params: serde_json::Value::Null,
                stats: Default::default(),
                model_identity: None,
                elapsed_ms: 0,
            };
            report.write_for(&p.to_string_lossy()).unwrap();
            out.push(p.to_string_lossy().into_owned());
        }
        (out[0].clone(), out[1].clone())
    }

    fn cache_for(dir: &Path, fasta: &Path, cfg: &mut Config) -> LibraryCache {
        cfg.predict_frag.library_cache = Some(dir.to_string_lossy().into_owned());
        LibraryCache::for_config(cfg, &fasta.to_string_lossy(), false, ("0", "sha"))
            .expect("native predictors key without any interpreter")
    }

    #[test]
    fn a_stored_library_is_restored_and_a_changed_input_misses() {
        let root = tmp("roundtrip");
        let fasta = root.join("x.fasta");
        std::fs::write(&fasta, ">P1\nPEPTIDEK\n").unwrap();
        let mut cfg = Config::default();
        let cache = cache_for(&root.join("cache"), &fasta, &mut cfg);

        let run1 = root.join("run1");
        std::fs::create_dir_all(&run1).unwrap();
        let (p1, f1) = fake_build(&run1, "a");
        let run2 = root.join("run2");
        std::fs::create_dir_all(&run2).unwrap();
        let p2 = run2.join(TABLES[0]).to_string_lossy().into_owned();
        let f2 = run2.join(TABLES[1]).to_string_lossy().into_owned();
        assert!(cache.restore(&p2, &f2).is_none(), "empty cache is a miss");

        cache.store(&p1, &f1);
        let (wp, wf) = cache.restore(&p2, &f2).expect("stored library is a hit");
        assert_eq!((wp.rows, wf.rows), (10, 11));
        assert_eq!(wp.content_hash, "hash-a-0");
        assert_eq!(std::fs::read(&p2).unwrap(), std::fs::read(&p1).unwrap());
        assert_eq!(std::fs::read(&f2).unwrap(), std::fs::read(&f1).unwrap());
        // Storing again keeps the first copy.
        cache.store(&p1, &f1);

        // Rebuilding run1's output does not reach the stored copy (every writer renames).
        let (_p1b, _f1b) = fake_build(&run1, "b");
        let run3 = root.join("run3");
        std::fs::create_dir_all(&run3).unwrap();
        let p3 = run3.join(TABLES[0]).to_string_lossy().into_owned();
        let f3 = run3.join(TABLES[1]).to_string_lossy().into_owned();
        let (wp3, _) = cache.restore(&p3, &f3).expect("still a hit");
        assert_eq!(wp3.content_hash, "hash-a-0");

        // A different FASTA, seed or build setting is a different key.
        let base_key = cache.key.clone();
        std::fs::write(&fasta, ">P1\nPEPTIDER\n").unwrap();
        let other = cache_for(&root.join("cache"), &fasta, &mut cfg);
        assert_ne!(other.key, base_key);
        std::fs::write(&fasta, ">P1\nPEPTIDEK\n").unwrap();
        let mut seeded = cfg.clone();
        seeded.rng_seed = 7;
        assert_ne!(
            cache_for(&root.join("cache"), &fasta, &mut seeded).key,
            base_key
        );
        let mut digest = cfg.clone();
        digest.digest.missed_cleavages += 1;
        assert_ne!(
            cache_for(&root.join("cache"), &fasta, &mut digest).key,
            base_key
        );
        // The cache directory is not part of the key.
        assert_eq!(
            cache_for(&root.join("elsewhere"), &fasta, &mut cfg).key,
            base_key
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The two table paths of a fresh run directory `name` under `root`.
    fn run_dir(root: &Path, name: &str) -> (String, String) {
        let d = root.join(name);
        std::fs::create_dir_all(&d).unwrap();
        (
            d.join(TABLES[0]).to_string_lossy().into_owned(),
            d.join(TABLES[1]).to_string_lossy().into_owned(),
        )
    }

    #[test]
    fn a_changed_entry_is_a_miss_and_the_next_store_repairs_it() {
        let root = tmp("tamper");
        let fasta = root.join("x.fasta");
        std::fs::write(&fasta, ">P1\nPEPTIDEK\n").unwrap();
        let mut cfg = Config::default();
        let cache = cache_for(&root.join("cache"), &fasta, &mut cfg);
        let (p1, f1) = fake_build(&root.join("run1"), "a");
        cache.store(&p1, &f1);
        let stored = cache.entry_dir().join(TABLES[1]);
        let original = std::fs::read(&stored).unwrap();

        // A shorter file, then a same-size one with other bytes: both are misses, and each
        // time the store that follows the rebuild replaces the entry, so the run after it
        // hits again instead of rebuilding for ever.
        let mut same_size = original.clone();
        same_size[0] ^= 1;
        for (i, bad) in [b"short".to_vec(), same_size].into_iter().enumerate() {
            std::fs::write(&stored, &bad).unwrap();
            let (p, f) = run_dir(&root, &format!("miss{i}"));
            assert!(
                cache.restore(&p, &f).is_none(),
                "tampered entry {i} is a miss"
            );
            cache.store(&p1, &f1);
            assert_eq!(
                std::fs::read(&stored).unwrap(),
                original,
                "entry {i} repaired"
            );
            let (p, f) = run_dir(&root, &format!("hit{i}"));
            assert!(
                cache.restore(&p, &f).is_some(),
                "repaired entry {i} is a hit"
            );
        }

        // An entry directory without entry.json is replaced too.
        std::fs::remove_file(cache.entry_dir().join("entry.json")).unwrap();
        let (p, f) = run_dir(&root, "noentry");
        assert!(cache.restore(&p, &f).is_none());
        cache.store(&p1, &f1);
        assert!(cache.restore(&p, &f).is_some());

        // Neither a hit nor a store shares a file with a run directory: rewriting a run's
        // library IN PLACE (not by rename) reaches neither the cache nor another run.
        let (hp, hf) = run_dir(&root, "inplace");
        cache.restore(&hp, &hf).expect("hit");
        let (op, of) = run_dir(&root, "other");
        cache.restore(&op, &of).expect("hit");
        for path in [&hf, &f1] {
            let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
            std::io::Write::write_all(&mut file, b"XX").unwrap();
        }
        assert_eq!(std::fs::read(&stored).unwrap(), original);
        assert_eq!(std::fs::read(&of).unwrap(), original);
        let (p, f) = run_dir(&root, "after");
        assert!(cache.restore(&p, &f).is_some());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_store_removes_abandoned_temporary_directories_and_nothing_else() {
        let root = tmp("sweep");
        let fasta = root.join("x.fasta");
        std::fs::write(&fasta, ">P1\nPEPTIDEK\n").unwrap();
        let mut cfg = Config::default();
        let cache_dir = root.join("cache");
        let cache = cache_for(&cache_dir, &fasta, &mut cfg);
        let k = "0123456789abcdef01234567";
        let broken = cache_dir.join(format!("{k}.broken-1-2"));
        let fresh = cache_dir.join(format!("{k}.partial-1-2"));
        let foreign = cache_dir.join("notes.partial-1-2");
        for d in [&broken, &fresh, &foreign] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join(TABLES[0]), b"x").unwrap();
        }
        let (p1, f1) = fake_build(&root.join("run1"), "a");
        cache.store(&p1, &f1);
        assert!(!broken.exists(), "a set-aside entry is removed");
        assert!(
            fresh.exists(),
            "a store that is still writing is left alone"
        );
        assert!(
            foreign.exists(),
            "a directory that is not the cache's is never touched"
        );
        assert_eq!(leftover_kind(&format!("{k}.partial-9-9")), Some("partial"));
        assert_eq!(leftover_kind(&format!("{k}.broken-9-9")), Some("broken"));
        assert_eq!(leftover_kind(k), None);
        assert_eq!(leftover_kind("zz3456789abcdef01234567.partial-1"), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn a_partial_store_unwritten_for_an_hour_is_removed() {
        let root = tmp("stale");
        let fasta = root.join("x.fasta");
        std::fs::write(&fasta, ">P1\nPEPTIDEK\n").unwrap();
        let mut cfg = Config::default();
        let cache_dir = root.join("cache");
        let cache = cache_for(&cache_dir, &fasta, &mut cfg);
        let stale = cache_dir.join(format!("{}.partial-1-2", cache.key));
        std::fs::create_dir_all(&stale).unwrap();
        let file = stale.join(TABLES[0]);
        std::fs::write(&file, b"x").unwrap();
        let old = std::time::SystemTime::now() - 2 * STALE_AFTER;
        for p in [&file, &stale] {
            std::fs::File::open(p).unwrap().set_modified(old).unwrap();
        }
        let (p1, f1) = fake_build(&root.join("run1"), "a");
        cache.store(&p1, &f1);
        assert!(!stale.exists(), "an abandoned partial store is removed");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_sidecar_predictor_without_an_interpreter_is_not_cached() {
        let root = tmp("nokey");
        let fasta = root.join("x.fasta");
        std::fs::write(&fasta, ">P1\nPEPTIDEK\n").unwrap();
        let mut cfg = Config::default();
        cfg.predict_frag.library_cache = Some(root.join("cache").to_string_lossy().into_owned());
        cfg.predict_frag.predictor = FragPredictorKind::Ms2pip;
        cfg.predict_frag.ms2pip_python = None;
        assert!(
            LibraryCache::for_config(&cfg, &fasta.to_string_lossy(), false, ("0", "sha")).is_none()
        );
        cfg.predict_frag.library_cache = None;
        cfg.predict_frag.predictor = FragPredictorKind::Native;
        assert!(
            LibraryCache::for_config(&cfg, &fasta.to_string_lossy(), false, ("0", "sha")).is_none(),
            "no cache configured"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_reuse_hint_names_the_library_irt_that_reproduces_the_run() {
        let mut cfg = Config::default();
        // Native RT: library-input mode must keep the library's iRT.
        let h = reuse_hint(&cfg, "p.parquet", "f.parquet", false).unwrap();
        assert!(h.contains("--lib-precursors p.parquet --lib-fragments f.parquet"));
        assert!(h.contains("library_irt = \"library\""), "{h}");
        // DeepLC RT with the automatic multi-head step: it must stay on.
        cfg.predict_frag.rt_predictor = RtPredictorKind::Deeplc;
        cfg.predict_frag.deeplc_python = Some("python".into());
        assert!(reuse_hint(&cfg, "p", "f", false)
            .unwrap()
            .contains("library_irt = \"deeplc\""));
        // Multi-head off: no base-model re-prediction either.
        cfg.rt_im_train.multihead_calibration = Some(0);
        assert!(reuse_hint(&cfg, "p", "f", false)
            .unwrap()
            .contains("library_irt = \"library\""));
        // Placeholder iRT: never reusable as a library.
        assert!(reuse_hint(&cfg, "p", "f", true).is_none());
        // The equivalence the hint rests on.
        for (rt, py, mh) in [
            (RtPredictorKind::Native, None, None),
            (RtPredictorKind::Deeplc, Some("python"), None),
            (RtPredictorKind::Deeplc, Some("python"), Some(0)),
            (RtPredictorKind::Native, Some("python"), Some(4)),
            (RtPredictorKind::Deeplc, None, None),
        ] {
            let mut c = Config::default();
            c.predict_frag.rt_predictor = rt;
            c.predict_frag.deeplc_python = py.map(str::to_string);
            c.rt_im_train.multihead_calibration = mh;
            let has = py.is_some();
            let fasta_heads = c
                .rt_im_train
                .multihead_heads(has, c.deeplc_rt_source(false, has));
            let hint = reuse_hint(&c, "p", "f", false).unwrap();
            c.rt_im_train.library_irt = if hint.contains("\"deeplc\"") {
                mumdia_core::config::LibraryIrt::Deeplc
            } else {
                mumdia_core::config::LibraryIrt::Library
            };
            let lib_heads = c
                .rt_im_train
                .multihead_heads(has, c.deeplc_rt_source(true, has));
            assert_eq!(fasta_heads, lib_heads, "{rt:?} {py:?} {mh:?}");
            assert!(
                !c.rt_im_train.repredicts_library_irt(true, has),
                "{rt:?} {py:?} {mh:?}: library-input mode would re-predict"
            );
        }
    }
}
