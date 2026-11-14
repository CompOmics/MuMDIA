//! Output retention (`output.retain`, docs/01_overview_and_dataflow.md "Output retention").
//!
//! After a successful `run` or `run-experiment` the orchestrator removes the files its
//! retention tier does not keep. The rules are the ones `scripts/prune_outputs.py` applies to
//! a finished directory, and the two must stay in step: a file has a class
//! ([`Class`]), decided by its path alone and by whether the second pass applied, and a tier
//! keeps a fixed set of classes. A file the rules do not know is never removed.
//!
//! Every `.json` and `.tsv` file is a result: the reports, schemas and summaries of the
//! removed tables stay, so the provenance of what was computed survives what was deleted.

use anyhow::{Context, Result};
use mumdia_core::config::Retain;
use mumdia_core::manifest::{RemovedFile, Retention};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// What a file is, for retention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Kept by every tier.
    Results,
    /// Kept by `viewer` and `all`.
    Viewer,
    /// Kept by `all` only: superseded or working tables nothing reads after the run.
    Intermediate,
    /// Kept by `all` only: sidecar handoff files.
    Scratch,
    /// Not known to the rules; never removed.
    Unknown,
}

impl Class {
    pub fn as_str(self) -> &'static str {
        match self {
            Class::Results => "results",
            Class::Viewer => "viewer",
            Class::Intermediate => "intermediate",
            Class::Scratch => "scratch",
            Class::Unknown => "unknown",
        }
    }
}

/// Whether `tier` keeps a file of class `c`.
pub fn keeps(tier: Retain, c: Class) -> bool {
    match tier {
        Retain::All => true,
        Retain::Viewer => matches!(c, Class::Results | Class::Viewer | Class::Unknown),
        Retain::Results => matches!(c, Class::Results | Class::Unknown),
    }
}

/// The shape of an output directory.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    /// `run-experiment` (run directories under the root) rather than a single `run`.
    pub experiment: bool,
    /// The run directory names of an experiment.
    pub runs: BTreeSet<String>,
    /// The experiment's `experiment.second_pass` is not null.
    pub second_pass: bool,
}

const ROOT_RESULTS: &[&str] = &[
    "scored_combined.parquet",
    "scored_mbr.parquet",
    "mbr_transferred.parquet",
    "lfq_maxlfq.parquet",
    "lfq_maxlfq.parquet.peptide.parquet",
    "lfq_maxlfq.parquet.precursor.parquet",
    "quant_normalisation.parquet",
    "localisation_experiment.parquet",
    // The report's protein annotations (`protein_annotations`), read by `mumdia report`.
    "protein_annotations.parquet",
];
/// A FASTA-built library in the output directory: the searched library, which a viewer
/// shows predicted fragments from; the empirical library replaces it under a second pass.
const ROOT_LIBRARY: &[&str] = &[
    "fragment_library_precursors.parquet",
    "fragment_library_fragments.parquet",
];
const ROOT_INTERMEDIATE: &[&str] = &[
    "peptides.parquet",
    "peptidoforms.parquet",
    "fragment_library_precursors_deeplc.parquet",
];
const RUN_RESULTS: &[&str] = &[
    "scored.parquet",
    "psms_scored.parquet",
    "peptide_quant.parquet",
    "protein_group_quant.parquet",
    "fragment_quant.parquet",
    "quant_fragments.parquet",
    "localisation.parquet",
];
/// A run's evidence tables. Under a second pass these first-pass copies are not read.
pub const RUN_EVIDENCE: &[&str] = &[
    "run_windows.parquet",
    "chromatograms.parquet",
    // `extract.q1_traces`: the same traces on Q1 cycles (SCIEX ZT Scan DIA).
    "chromatograms_q1.parquet",
    "features.parquet",
    "psms_competed.parquet",
    "psms_extracted.parquet",
    "psms_extracted.parquet.peaks.parquet",
    "seed_psms.parquet",
];
const RUN_INTERMEDIATE: &[&str] = &[
    "fragment_library_precursors_multihead.parquet",
    "fragment_library_precursors_ft.parquet",
    "fragment_library_precursors_deeplc.parquet",
    "library_tag_prefiltered_precursors.parquet",
    "library_tag_prefiltered_fragments.parquet",
    "peptidoforms_tag_prefiltered.parquet",
    "candidate_audit.parquet",
    "run.pin",
];

/// `<name>.tmp-<pid>-<n>`: an unfinished atomic write.
fn is_tmp(name: &str) -> bool {
    let Some(i) = name.rfind(".tmp-") else {
        return false;
    };
    let rest = &name[i + 5..];
    let mut it = rest.splitn(2, '-');
    let (a, b) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
    !a.is_empty()
        && !b.is_empty()
        && a.bytes().all(|c| c.is_ascii_digit())
        && b.bytes().all(|c| c.is_ascii_digit())
}

/// The class of the file at `rel` (relative to the output directory, `/`-separated).
pub fn classify(rel: &str, layout: &Layout) -> Class {
    let parts: Vec<&str> = rel.split('/').filter(|s| !s.is_empty()).collect();
    let Some(&name) = parts.last() else {
        return Class::Unknown;
    };
    if is_tmp(name) {
        return Class::Unknown;
    }
    if parts[..parts.len() - 1].contains(&"sidecar_work") {
        return Class::Scratch;
    }
    if name.ends_with(".json") || name.ends_with(".tsv") {
        return Class::Results;
    }
    let second = layout.second_pass;
    if !layout.experiment {
        return run_class(&parts, false, true);
    }
    if parts.len() == 1 {
        return root_class(name, second);
    }
    match parts[0] {
        // The first-pass rescore: superseded by the second pass's, or copied up to the
        // experiment root when the second pass did not apply.
        "first_pass" => Class::Intermediate,
        "second_pass" => {
            if parts[1] == "library" {
                Class::Results
            } else if parts.len() >= 3 && layout.runs.contains(parts[1]) && second {
                Class::Viewer
            } else {
                Class::Unknown
            }
        }
        r if layout.runs.contains(r) => run_class(&parts[1..], second, false),
        _ => Class::Unknown,
    }
}

fn root_class(name: &str, second: bool) -> Class {
    if ROOT_RESULTS.contains(&name) {
        Class::Results
    } else if ROOT_LIBRARY.contains(&name) {
        if second {
            Class::Intermediate
        } else {
            Class::Viewer
        }
    } else if ROOT_INTERMEDIATE.contains(&name) {
        Class::Intermediate
    } else {
        Class::Unknown
    }
}

fn run_class(parts: &[&str], second: bool, root: bool) -> Class {
    let name = parts[parts.len() - 1];
    let evidence = if second {
        Class::Intermediate
    } else {
        Class::Viewer
    };
    if parts.len() == 1 {
        if root {
            let c = root_class(name, second);
            if c != Class::Unknown {
                return c;
            }
        }
        return if RUN_RESULTS.contains(&name) {
            Class::Results
        } else if RUN_EVIDENCE.contains(&name) {
            evidence
        } else if RUN_INTERMEDIATE.contains(&name) {
            Class::Intermediate
        } else {
            Class::Unknown
        };
    }
    match parts[0] {
        "spectra" => Class::Viewer,
        "rt_bootstrap" => Class::Intermediate,
        // Window-group bands (docs/33): each band's evidence tables and the overlap losers.
        "groups" => evidence,
        _ => Class::Unknown,
    }
}

/// Every regular file under `dir`, relative and `/`-separated, sorted. Symbolic links are
/// skipped (never followed or removed), and so is a `.d` directory.
fn walk(dir: &Path) -> Result<Vec<(String, std::fs::Metadata)>> {
    fn rec(base: &Path, dir: &Path, out: &mut Vec<(String, std::fs::Metadata)>) -> Result<()> {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("listing {}", dir.display()))?
            .collect::<std::io::Result<_>>()?;
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let path = e.path();
            let ft = e.file_type()?;
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                let n = e.file_name().to_string_lossy().to_ascii_lowercase();
                if !n.ends_with(".d") {
                    rec(base, &path, out)?;
                }
            } else if ft.is_file() {
                let rel = path
                    .strip_prefix(base)
                    .expect("under the base")
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                out.push((rel, e.metadata()?));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    rec(dir, dir, &mut out)?;
    Ok(out)
}

#[cfg(unix)]
fn inode(m: &std::fs::Metadata) -> Option<((u64, u64), u64)> {
    use std::os::unix::fs::MetadataExt;
    Some(((m.dev(), m.ino()), m.nlink()))
}
#[cfg(not(unix))]
fn inode(_m: &std::fs::Metadata) -> Option<((u64, u64), u64)> {
    None
}

/// The content hashes a manifest recorded, by path relative to `out_dir`. `records` are the
/// `(path, content_hash)` pairs of its artifacts, paths as the orchestrator wrote them
/// (`<out_dir>/<rel>`).
pub fn hashes_by_rel<'a>(
    out_dir: &str,
    records: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> BTreeMap<String, String> {
    let base = Path::new(out_dir);
    let mut m = BTreeMap::new();
    for (path, hash) in records {
        if let Ok(rel) = Path::new(path).strip_prefix(base) {
            let rel = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            m.insert(rel, hash.to_string());
        }
    }
    m
}

/// What `tier` removes from `out_dir`. `None` under `all`.
pub fn plan(
    out_dir: &Path,
    tier: Retain,
    layout: &Layout,
    hashes: &BTreeMap<String, String>,
) -> Result<Option<Retention>> {
    if tier == Retain::All {
        return Ok(None);
    }
    let files = walk(out_dir)?;
    let mut links_in_dir: BTreeMap<(u64, u64), u64> = BTreeMap::new();
    for (_, m) in &files {
        if let Some((ino, _)) = inode(m) {
            *links_in_dir.entry(ino).or_default() += 1;
        }
    }
    let removed: Vec<(String, Class, std::fs::Metadata)> = files
        .into_iter()
        .filter(|(rel, _)| rel != "pruned.json")
        .filter_map(|(rel, m)| {
            let c = classify(&rel, layout);
            (!keeps(tier, c)).then_some((rel, c, m))
        })
        .collect();
    let mut removed_links: BTreeMap<(u64, u64), u64> = BTreeMap::new();
    for (_, _, m) in &removed {
        if let Some((ino, _)) = inode(m) {
            *removed_links.entry(ino).or_default() += 1;
        }
    }
    let mut freed = 0u64;
    let mut seen = BTreeSet::new();
    for (_, _, m) in &removed {
        match inode(m) {
            None => freed += m.len(),
            Some((ino, nlink)) => {
                if removed_links[&ino] == links_in_dir[&ino]
                    && links_in_dir[&ino] == nlink
                    && seen.insert(ino)
                {
                    freed += m.len();
                }
            }
        }
    }
    let mut files: Vec<RemovedFile> = removed
        .iter()
        .map(|(rel, c, m)| RemovedFile {
            content_hash: hashes.get(rel).cloned(),
            path: rel.clone(),
            class: c.as_str().to_string(),
            bytes: m.len(),
        })
        .collect();
    let mut bytes_removed: u64 = files.iter().map(|f| f.bytes).sum();
    // What the run already removed while it ran ([`remove_early`]).
    for (f, f_freed) in take_early(out_dir) {
        bytes_removed += f.bytes;
        freed += f_freed;
        files.push(f);
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Some(Retention {
        tier: tier.as_str().to_string(),
        second_pass: layout.second_pass,
        bytes_removed,
        bytes_freed: freed,
        removed: files,
    }))
}

/// Files removed during a run, before the end ([`remove_early`]): absolute path, record and
/// the bytes the removal released.
static EARLY: std::sync::Mutex<Vec<(PathBuf, RemovedFile, u64)>> =
    std::sync::Mutex::new(Vec::new());

fn absolute(p: &Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

/// The early removals under `out_dir`, with their paths made relative to it.
fn take_early(out_dir: &Path) -> Vec<(RemovedFile, u64)> {
    let base = absolute(out_dir);
    let mut g = EARLY.lock().unwrap_or_else(|e| e.into_inner());
    let (mine, rest): (Vec<_>, Vec<_>) = g.drain(..).partition(|(p, _, _)| p.starts_with(&base));
    *g = rest;
    mine.into_iter()
        .map(|(p, mut f, freed)| {
            f.path = p
                .strip_prefix(&base)
                .expect("partitioned on it")
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            (f, freed)
        })
        .collect()
}

/// Remove `path` now, during the run, because the tier will not keep it and nothing later
/// in the run reads it (`output.retain` below `all`: the run's peak disk, not only its end
/// state). The removal is recorded and joins the manifest's `retention` block. A missing
/// file is skipped; a failure to remove is a warning, and the end of the run tries again.
pub fn remove_early(path: &str, class: Class, why: &str) {
    let p = Path::new(path);
    let Ok(m) = std::fs::symlink_metadata(p) else {
        return;
    };
    if !m.is_file() {
        return;
    }
    let freed = match inode(&m) {
        Some((_, nlink)) if nlink > 1 => 0,
        _ => m.len(),
    };
    match std::fs::remove_file(p) {
        Ok(()) => {
            tracing::debug!(path, bytes = m.len(), why, "output.retain: removed early");
            EARLY.lock().unwrap_or_else(|e| e.into_inner()).push((
                absolute(p),
                RemovedFile {
                    path: String::new(),
                    class: class.as_str().to_string(),
                    bytes: m.len(),
                    content_hash: None,
                },
                freed,
            ));
        }
        Err(e) => warn!(path, error = %e, "output.retain: could not remove early"),
    }
}

/// [`remove_early`] for every file of `dir` (not recursing) that `keep` rejects, by name.
pub fn remove_early_in(dir: &str, class: Class, why: &str, remove: impl Fn(&str) -> bool) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut names: Vec<String> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    for n in names {
        if remove(&n) {
            remove_early(&format!("{dir}/{n}"), class, why);
        }
    }
}

/// Remove the files of `r` from `out_dir`, then the directories that left empty. A file
/// already gone is not an error.
pub fn remove(out_dir: &Path, r: &Retention) -> Result<()> {
    let mut emptied: BTreeSet<PathBuf> = BTreeSet::new();
    for f in &r.removed {
        let p = out_dir.join(&f.path);
        match std::fs::remove_file(&p) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(e).with_context(|| format!("output.retain: removing {}", p.display()))
            }
        }
        let mut d = p.parent().map(Path::to_path_buf);
        while let Some(dir) = d {
            if dir == out_dir || !dir.starts_with(out_dir) {
                break;
            }
            d = dir.parent().map(Path::to_path_buf);
            emptied.insert(dir);
        }
    }
    let mut dirs: Vec<PathBuf> = emptied.into_iter().collect();
    dirs.sort_by_key(|d| std::cmp::Reverse(d.components().count()));
    for d in dirs {
        let empty = std::fs::read_dir(&d)
            .map(|mut it| it.next().is_none())
            .unwrap_or(false);
        if empty {
            if let Err(e) = std::fs::remove_dir(&d) {
                warn!(dir = %d.display(), error = %e, "output.retain: could not remove an emptied directory");
            }
        }
    }
    info!(
        tier = %r.tier,
        files = r.removed.len(),
        gb_freed = format!("{:.2}", r.bytes_freed as f64 / 1e9),
        second_pass = r.second_pass,
        "output.retain: removed what the tier does not keep (recorded under `retention` in the manifest)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp(second: bool) -> Layout {
        Layout {
            experiment: true,
            // An index-named run and one named after its input (`experiment.run_naming`).
            runs: ["r0".to_string(), "HYE_A01".to_string()].into(),
            second_pass: second,
        }
    }

    /// The table `scripts/prune_outputs.py` applies (tests/python/test_prune_outputs.py
    /// checks the same paths there).
    #[test]
    fn classes_follow_the_second_pass() {
        let l = exp(true);
        for (rel, c) in [
            ("experiment_manifest.json", Class::Results),
            ("scored_combined.parquet", Class::Results),
            ("lfq_maxlfq.parquet.peptide.parquet", Class::Results),
            ("peptides.tsv", Class::Results),
            ("first_pass/scored_combined.parquet", Class::Intermediate),
            (
                "first_pass/scored_combined.parquet.report.json",
                Class::Results,
            ),
            ("sidecar_work/x.npy", Class::Scratch),
            ("second_pass/library/lib_fragments.parquet", Class::Results),
            ("second_pass/r0/chromatograms.parquet", Class::Viewer),
            ("second_pass/r0/cal.json", Class::Results),
            ("r0/scored.parquet", Class::Results),
            ("r0/fragment_quant.parquet", Class::Results),
            ("r0/spectra/spectra_ms2.parquet", Class::Viewer),
            ("r0/chromatograms.parquet", Class::Intermediate),
            ("r0/psms_competed.parquet", Class::Intermediate),
            (
                "r0/psms_extracted.parquet.peaks.parquet",
                Class::Intermediate,
            ),
            ("r0/seed_psms.parquet.masscal.json", Class::Results),
            (
                "r0/fragment_library_precursors_multihead.parquet",
                Class::Intermediate,
            ),
            ("r0/rt_bootstrap/run_windows.parquet", Class::Intermediate),
            ("r0/rt_bootstrap/cal.json", Class::Results),
            ("r0/chromatograms.parquet.tmp-12-0", Class::Unknown),
            ("HYE_A01/scored.parquet", Class::Results),
            ("HYE_A01/chromatograms.parquet", Class::Intermediate),
            ("second_pass/HYE_A01/chromatograms.parquet", Class::Viewer),
            ("HYE_A02/chromatograms.parquet", Class::Unknown),
            ("r9/chromatograms.parquet", Class::Unknown),
            ("notes.txt", Class::Unknown),
        ] {
            assert_eq!(classify(rel, &l), c, "{rel}");
        }
        let l = exp(false);
        assert_eq!(classify("r0/chromatograms.parquet", &l), Class::Viewer);
        assert_eq!(
            classify("r0/groups/g00/features.parquet", &l),
            Class::Viewer
        );
        assert_eq!(
            classify("r0/fragment_library_precursors_multihead.parquet", &l),
            Class::Intermediate
        );
        let single = Layout::default();
        assert_eq!(classify("psms_scored.parquet", &single), Class::Results);
        assert_eq!(classify("features.parquet", &single), Class::Viewer);
        assert_eq!(
            classify("fragment_library_fragments.parquet", &single),
            Class::Viewer
        );
        assert_eq!(
            classify("peptidoforms.parquet", &single),
            Class::Intermediate
        );
        assert_eq!(
            classify("spectra/spectra_ms1.parquet", &single),
            Class::Viewer
        );
    }

    #[test]
    fn plan_and_remove() {
        let dir = std::env::temp_dir().join(format!("mumdia_retain_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let w = |rel: &str, n: usize| {
            let p = dir.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, vec![b'x'; n]).unwrap();
        };
        w("experiment_manifest.json", 3);
        w("r0/scored.parquet", 5);
        w("r0/features.parquet", 7);
        std::fs::hard_link(
            dir.join("r0/features.parquet"),
            dir.join("r0/psms_competed.parquet"),
        )
        .unwrap();
        w("r0/features.parquet.schema.json", 2);
        w("r0/rt_bootstrap/run_windows.parquet", 11);
        w("r0/spectra/spectra_ms2.parquet", 13);
        w("second_pass/r0/features.parquet", 17);
        w("notes.txt", 1);
        let l = Layout {
            experiment: true,
            runs: ["r0".to_string()].into(),
            second_pass: true,
        };
        let hashes = hashes_by_rel(
            &dir.to_string_lossy(),
            [(
                dir.join("r0")
                    .join("features.parquet")
                    .to_string_lossy()
                    .into_owned(),
                "abc".to_string(),
            )]
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str())),
        );
        assert!(plan(&dir, Retain::All, &l, &hashes).unwrap().is_none());
        let v = plan(&dir, Retain::Viewer, &l, &hashes).unwrap().unwrap();
        let paths: Vec<&str> = v.removed.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "r0/features.parquet",
                "r0/psms_competed.parquet",
                "r0/rt_bootstrap/run_windows.parquet"
            ]
        );
        assert_eq!(v.removed[0].content_hash.as_deref(), Some("abc"));
        assert_eq!(v.bytes_removed, 7 + 7 + 11);
        if cfg!(unix) {
            assert_eq!(v.bytes_freed, 7 + 11);
        }
        let r = plan(&dir, Retain::Results, &l, &hashes).unwrap().unwrap();
        assert_eq!(r.removed.len(), 5);
        remove(&dir, &r).unwrap();
        assert!(dir.join("r0/scored.parquet").exists());
        assert!(dir.join("r0/features.parquet.schema.json").exists());
        assert!(dir.join("notes.txt").exists());
        assert!(!dir.join("r0/rt_bootstrap").exists());
        assert!(!dir.join("second_pass/r0/features.parquet").exists());
        assert!(!dir.join("r0/spectra").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
