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

/// `entry.json` of one cache entry: the key material and the size of every stored file.
#[derive(Serialize, Deserialize)]
struct Entry {
    cache_format: u32,
    key: String,
    material: serde_json::Value,
    /// File name (a table or its `.report.json`) to its size in bytes.
    files: std::collections::BTreeMap<String, u64>,
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

    /// On a hit, publish the stored tables and their reports at `lib_p` and `lib_f` (a
    /// hard link where the filesystem allows, a copy otherwise) and return what their
    /// reports record. `None` is a miss: no entry, or one whose files are missing or not
    /// the size they were stored at, which is logged and then rebuilt over.
    pub fn restore(&self, lib_p: &str, lib_f: &str) -> Option<(Written, Written)> {
        let entry_dir = self.entry_dir();
        let entry_path = entry_dir.join("entry.json");
        if !entry_path.is_file() {
            info!(
                key = %self.key,
                cache = %self.dir.display(),
                "library cache: no stored library for this FASTA and configuration; building it"
            );
            return None;
        }
        match self.try_restore(&entry_dir, &entry_path, lib_p, lib_f) {
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
                    "library cache: the stored library is unusable; rebuilding it"
                );
                None
            }
        }
    }

    fn try_restore(
        &self,
        entry_dir: &Path,
        entry_path: &Path,
        lib_p: &str,
        lib_f: &str,
    ) -> Result<(Written, Written)> {
        let entry: Entry = mumdia_io::json::read_json(&entry_path.to_string_lossy())?;
        anyhow::ensure!(
            entry.cache_format == CACHE_FORMAT && entry.key == self.key,
            "entry.json is for format {} key {}",
            entry.cache_format,
            entry.key
        );
        for (name, bytes) in &entry.files {
            let got = std::fs::metadata(entry_dir.join(name))
                .with_context(|| format!("stored file {name}"))?
                .len();
            anyhow::ensure!(
                got == *bytes,
                "{name} is {got} bytes, stored at {bytes}; it was changed after it was stored"
            );
        }
        let mut written = Vec::with_capacity(2);
        for (table, out) in TABLES.iter().zip([lib_p, lib_f]) {
            let src = entry_dir.join(table);
            let src_report = entry_dir.join(format!("{table}.report.json"));
            mumdia_io::table::publish_copy_of(&src.to_string_lossy(), out)?;
            let out_report = format!("{out}.report.json");
            mumdia_io::table::publish_copy_of(&src_report.to_string_lossy(), &out_report)?;
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
    /// The files go into a private temporary directory beside the entry, which is then
    /// renamed into place, so a reader sees a complete entry or none; when another run
    /// stored the same key first, this copy is discarded.
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

    fn try_store(&self, lib_p: &str, lib_f: &str) -> Result<bool> {
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating {}", self.dir.display()))?;
        let final_dir = self.entry_dir();
        if final_dir.join("entry.json").is_file() {
            return Ok(false);
        }
        let tmp = self.dir.join(format!(
            "{}.partial-{}-{}",
            self.key,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        let result = (|| -> Result<bool> {
            let mut files = std::collections::BTreeMap::new();
            for (table, src) in TABLES.iter().zip([lib_p, lib_f]) {
                for (name, from) in [
                    (table.to_string(), src.to_string()),
                    (format!("{table}.report.json"), format!("{src}.report.json")),
                ] {
                    let to = tmp.join(&name);
                    mumdia_io::table::publish_copy_of(&from, &to.to_string_lossy())?;
                    files.insert(name, std::fs::metadata(&to)?.len());
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
                Err(_) if final_dir.join("entry.json").is_file() => Ok(false),
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
        let mut out = Vec::new();
        for (i, t) in TABLES.iter().enumerate() {
            let p = dir.join(t);
            // Published by rename, as every engine writer does, so a hard link to an
            // earlier version keeps that version's bytes.
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

    #[test]
    fn a_stored_file_that_changed_size_is_a_miss() {
        let root = tmp("tamper");
        let fasta = root.join("x.fasta");
        std::fs::write(&fasta, ">P1\nPEPTIDEK\n").unwrap();
        let mut cfg = Config::default();
        let cache = cache_for(&root.join("cache"), &fasta, &mut cfg);
        let run1 = root.join("run1");
        std::fs::create_dir_all(&run1).unwrap();
        let (p1, f1) = fake_build(&run1, "a");
        cache.store(&p1, &f1);
        // Write a new, shorter file at the stored name (a new file, so run1's copy stays).
        let stored = cache.entry_dir().join(TABLES[1]);
        std::fs::remove_file(&stored).unwrap();
        std::fs::write(&stored, b"short").unwrap();
        let run2 = root.join("run2");
        std::fs::create_dir_all(&run2).unwrap();
        assert!(cache
            .restore(
                &run2.join(TABLES[0]).to_string_lossy(),
                &run2.join(TABLES[1]).to_string_lossy()
            )
            .is_none());
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
