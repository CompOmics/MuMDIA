//! What this machine can search at once: its memory, the size of the chosen search space,
//! and the window-group (band) plan the Search tab proposes from the two.
//!
//! The band plan is a proposal, not a measurement. Its memory model is fitted to the
//! measured peaks of `docs/33_window_groups.md` (about 4 GB fixed per band plus about 2 GB
//! per million library precursors in the band: a 44.6M-precursor band of the 203M
//! immunopeptidomics library peaked at 96 GB, the unbanded 10.9M HYE library at 16.5 GiB),
//! which is conservative on tryptic data and about right on dense immunopeptidomics data.
//! Bands do not change what is identified once the calibration is pooled (docs/33 section
//! 8); they cost repeated fixed work, which is why the plan bands only when the unbanded
//! search would not fit.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Fixed memory of one band, GB: the run's decoded spectra, the band's fragment index and
/// the accumulator.
const BAND_FIXED_GB: f64 = 4.0;
/// Memory per million library precursors searched in one band, GB.
const GB_PER_MILLION_PRECURSORS: f64 = 2.0;
/// Beyond about 64 bands extra bands cost CPU and no memory worth having (docs/33 section 8),
/// and a band cannot be smaller than one isolation window.
const MAX_BANDS: usize = 64;
/// Distinct peptides tracked exactly before the estimate stops deduplicating.
const DEDUP_LIMIT: usize = 40_000_000;

/// The machine the search would run on.
#[derive(Clone, Debug, Serialize)]
pub struct Machine {
    pub total_memory_bytes: Option<u64>,
    pub threads: usize,
}

pub fn machine() -> Machine {
    Machine {
        total_memory_bytes: total_memory_bytes(),
        threads: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    }
}

/// Physical memory, or the cgroup limit when a container imposes a smaller one.
pub fn total_memory_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        }
        let mut s = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: `s` is a correctly sized MEMORYSTATUSEX with `dwLength` set, as the
        // function requires; it writes only into that struct.
        let ok = unsafe { GlobalMemoryStatusEx(&mut s) };
        if ok != 0 && s.total_phys > 0 {
            return Some(s.total_phys);
        }
        None
    }
    #[cfg(target_os = "linux")]
    {
        let total = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|m| parse_meminfo_total(&m));
        let limit = std::fs::read_to_string("/sys/fs/cgroup/memory.max")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
        match (total, limit) {
            (Some(t), Some(l)) => Some(t.min(l)),
            (t, _) => t,
        }
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_meminfo_total(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// One modification as the Search tab selects it.
#[derive(Clone, Debug, Deserialize)]
pub struct ModChoice {
    pub name: String,
    pub residue: char,
}

/// The search space the Search tab describes.
#[derive(Clone, Debug, Deserialize)]
pub struct SpaceRequest {
    /// `fasta` or `library`.
    pub mode: String,
    #[serde(default)]
    pub fasta: Option<String>,
    #[serde(default)]
    pub lib_precursors: Option<String>,
    #[serde(default = "one")]
    pub missed_cleavages: usize,
    #[serde(default = "seven")]
    pub min_len: usize,
    #[serde(default = "thirty")]
    pub max_len: usize,
    #[serde(default = "two")]
    pub min_charge: usize,
    #[serde(default = "three")]
    pub max_charge: usize,
    #[serde(default)]
    pub variable: Vec<ModChoice>,
    #[serde(default = "one")]
    pub max_variable_mods: usize,
    /// Fraction of the library a prescreen before prediction is expected to keep (1 when
    /// none is chosen), so the plan sizes the library that is actually searched.
    #[serde(default = "unity")]
    pub keep_fraction: f64,
    /// `--threads`, when the Search tab sets it.
    #[serde(default)]
    pub threads: Option<usize>,
}

fn one() -> usize {
    1
}
fn two() -> usize {
    2
}
fn three() -> usize {
    3
}
fn seven() -> usize {
    7
}
fn thirty() -> usize {
    30
}
fn unity() -> f64 {
    1.0
}

/// The size of the search space, and how it was obtained.
#[derive(Clone, Debug, Serialize)]
pub struct Space {
    /// Library precursors, targets and decoys, before any prescreen.
    pub precursors: Option<u64>,
    /// Distinct target peptide sequences (FASTA mode only).
    pub peptides: Option<u64>,
    /// True when the count was read from the library itself rather than estimated.
    pub exact: bool,
    /// Library precursors the search is expected to load after the prescreen.
    pub searched_precursors: Option<u64>,
    pub machine: Machine,
    pub plan: BandPlan,
    /// Why the estimate is missing, when it is.
    pub reason: Option<String>,
}

/// The proposed window-group plan.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct BandPlan {
    pub bands: usize,
    pub parallel: usize,
    pub budget_gb: f64,
    pub unbanded_gb: Option<f64>,
    pub band_gb: Option<f64>,
    pub note: String,
}

pub fn estimate(req: &SpaceRequest, engine: Option<&Path>) -> Space {
    let m = machine();
    let (precursors, peptides, exact, reason) = match req.mode.as_str() {
        "library" => match req.lib_precursors.as_deref().filter(|p| !p.is_empty()) {
            None => (
                None,
                None,
                false,
                Some("No precursor table selected.".to_string()),
            ),
            Some(p) => match engine.map(|e| library_rows(e, p)) {
                Some(Ok(n)) => (Some(n), None, true, None),
                Some(Err(e)) => (None, None, false, Some(e)),
                None => (None, None, false, Some("The engine was not found.".into())),
            },
        },
        _ => match req.fasta.as_deref().filter(|p| !p.is_empty()) {
            None => (None, None, false, Some("No FASTA selected.".to_string())),
            Some(f) => match fasta_space(Path::new(f), req) {
                Ok((pep, prec)) => (Some(prec), Some(pep), false, None),
                Err(e) => (None, None, false, Some(e)),
            },
        },
    };
    let keep = if req.keep_fraction.is_finite() {
        req.keep_fraction.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let searched = precursors.map(|p| (p as f64 * keep).round() as u64);
    let threads = req.threads.filter(|&t| t > 0).unwrap_or(m.threads);
    let plan = band_plan(searched, m.total_memory_bytes, threads);
    Space {
        precursors,
        peptides,
        exact,
        searched_precursors: searched,
        machine: m,
        plan,
        reason,
    }
}

/// Rows of a library precursor table, from the engine's footer-only `inspect`.
fn library_rows(engine: &Path, table: &str) -> Result<u64, String> {
    let mut cmd = crate::engine::command(engine);
    crate::components::stamp_env(&mut cmd);
    let out = cmd
        .args(["inspect", table])
        .output()
        .map_err(|e| format!("could not run the engine: {e}"))?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("rows: "))
        .and_then(|n| n.trim().parse().ok())
        .ok_or_else(|| "The engine could not read this precursor table.".to_string())
}

/// Distinct target peptides and library precursors (targets plus their decoys) a FASTA
/// gives under the Search tab's digest and modification settings.
///
/// Mirrors the native digest closely enough for sizing, not exactly: Trypsin/P (the engine
/// default), the configured missed cleavages and length range, N-terminal methionine
/// excision (the default), every charge in range, and one paired decoy per target
/// peptidoform. Each distinct peptide's modified forms are counted exactly: with `a_i`
/// variable alternatives at site `i`, the forms with at most `k` modified sites are the
/// elementary symmetric polynomials `e_0..e_k` of the `a_i`, summed.
pub fn fasta_space(path: &Path, req: &SpaceRequest) -> Result<(u64, u64), String> {
    let text = std::fs::read(path).map_err(|e| format!("could not read the FASTA: {e}"))?;
    let mut alternatives = [0u32; 256];
    for m in &req.variable {
        let r = m.residue.to_ascii_uppercase();
        if r.is_ascii_uppercase() {
            alternatives[r as usize] += 1;
        }
    }
    let charges = (req.max_charge.saturating_sub(req.min_charge) + 1) as f64;
    let mut seen: HashSet<u64> = HashSet::new();
    let mut dedup = true;
    let mut peptides = 0u64;
    let mut forms = 0f64;
    let mut count = |pep: &[u8]| {
        if dedup {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            pep.hash(&mut h);
            if !seen.insert(h.finish()) {
                return;
            }
            if seen.len() >= DEDUP_LIMIT {
                dedup = false;
            }
        }
        peptides += 1;
        forms += modified_forms(pep, &alternatives, req.max_variable_mods);
    };
    for seq in proteins(&text) {
        digest(&seq, req, &mut count);
    }
    let precursors = (forms * charges * 2.0).round() as u64;
    Ok((peptides, precursors))
}

fn proteins(text: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    for line in text.split(|&b| b == b'\n') {
        if line.first() == Some(&b'>') {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        cur.extend(
            line.iter()
                .filter(|b| b.is_ascii_alphabetic())
                .map(|b| b.to_ascii_uppercase()),
        );
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Tryptic (Trypsin/P) peptides of one protein, with N-terminal Met excision.
fn digest(seq: &[u8], req: &SpaceRequest, emit: &mut impl FnMut(&[u8])) {
    let mut sites: Vec<usize> = vec![0];
    for (i, &aa) in seq.iter().enumerate() {
        if (aa == b'K' || aa == b'R') && i + 1 < seq.len() {
            sites.push(i + 1);
        }
    }
    sites.push(seq.len());
    let mut starts: Vec<(usize, usize)> = (0..sites.len() - 1).map(|s| (s, sites[s])).collect();
    if seq.first() == Some(&b'M') && sites.len() > 1 && sites[1] > 1 {
        // The Met-excised forms of the N-terminal peptides start at residue 1.
        starts.push((0, 1));
    }
    for (s, from) in starts {
        for end in (s + 1)..sites.len().min(s + 2 + req.missed_cleavages) {
            let len = sites[end] - from;
            if len >= req.min_len && len <= req.max_len {
                emit(&seq[from..sites[end]]);
            }
        }
    }
}

fn modified_forms(pep: &[u8], alternatives: &[u32; 256], max_var: usize) -> f64 {
    let mut e = vec![0f64; max_var + 1];
    e[0] = 1.0;
    for &aa in pep {
        let a = alternatives[aa as usize] as f64;
        if a > 0.0 {
            for j in (1..=max_var).rev() {
                e[j] += e[j - 1] * a;
            }
        }
    }
    e.iter().sum()
}

/// The proposed plan for `precursors` library precursors on a machine with `memory` bytes.
pub fn band_plan(precursors: Option<u64>, memory: Option<u64>, threads: usize) -> BandPlan {
    let total_gb = memory.map(|b| b as f64 / 1e9);
    // Leave room for the operating system and the rest of the desktop: 60% of the memory,
    // and never more than all of it but 6 GB.
    let budget_gb = total_gb
        .map(|t| (0.6 * t).min(t - 6.0).max(1.0))
        .unwrap_or(0.0);
    let (Some(p), Some(total)) = (precursors, total_gb) else {
        return BandPlan {
            bands: 1,
            parallel: 1,
            budget_gb,
            unbanded_gb: None,
            band_gb: None,
            note: if memory.is_none() {
                "The memory of this machine could not be read, so no bands are proposed.".into()
            } else {
                "No search-space size yet, so no bands are proposed.".into()
            },
        };
    };
    let millions = p as f64 / 1e6;
    let unbanded = BAND_FIXED_GB + GB_PER_MILLION_PRECURSORS * millions;
    if unbanded <= budget_gb {
        return BandPlan {
            bands: 1,
            parallel: 1,
            budget_gb,
            unbanded_gb: Some(unbanded),
            band_gb: Some(unbanded),
            note: format!(
                "About {unbanded:.0} GB unbanded fits in the {budget_gb:.0} GB this {total:.0} GB \
                 machine can give a search, so it is not split."
            ),
        };
    }
    let room = budget_gb - BAND_FIXED_GB;
    let mut bands = if room >= GB_PER_MILLION_PRECURSORS {
        (GB_PER_MILLION_PRECURSORS * millions / room).ceil() as usize
    } else {
        millions.ceil() as usize
    };
    bands = bands.clamp(2, MAX_BANDS);
    let band_gb = BAND_FIXED_GB + GB_PER_MILLION_PRECURSORS * millions / bands as f64;
    // Bands at a time: as many as the budget holds, below the thread count (a band in
    // flight parks one worker, docs/33 section 8) and with at least four threads each.
    let by_threads = (threads / 4).max(1).min(threads.saturating_sub(1).max(1));
    let parallel = ((budget_gb / band_gb).floor() as usize).clamp(1, bands.min(by_threads));
    let mut note = format!(
        "About {unbanded:.0} GB unbanded exceeds the {budget_gb:.0} GB this {total:.0} GB machine \
         can give a search: {bands} bands of about {band_gb:.0} GB each, {parallel} at a time."
    );
    if band_gb > budget_gb {
        note.push_str(
            " Even the smallest useful band may not fit; consider a prescreen before prediction \
             or a smaller search space.",
        );
    }
    BandPlan {
        bands,
        parallel,
        budget_gb,
        unbanded_gb: Some(unbanded),
        band_gb: Some(band_gb),
        note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(variable: &[(&str, char)], max_var: usize) -> SpaceRequest {
        SpaceRequest {
            mode: "fasta".into(),
            fasta: None,
            lib_precursors: None,
            missed_cleavages: 1,
            min_len: 3,
            max_len: 30,
            min_charge: 2,
            max_charge: 3,
            variable: variable
                .iter()
                .map(|(n, r)| ModChoice {
                    name: n.to_string(),
                    residue: *r,
                })
                .collect(),
            max_variable_mods: max_var,
            keep_fraction: 1.0,
            threads: None,
        }
    }

    #[test]
    fn modified_forms_are_the_elementary_symmetric_sums() {
        let mut alt = [0u32; 256];
        alt[b'S' as usize] = 1;
        alt[b'M' as usize] = 2;
        // Sites S, M, S: a = (1, 2, 1). e0 = 1, e1 = 4, e2 = 1*2 + 1*1 + 2*1 = 5, e3 = 2.
        assert_eq!(modified_forms(b"ASMSK", &alt, 0), 1.0);
        assert_eq!(modified_forms(b"ASMSK", &alt, 1), 5.0);
        assert_eq!(modified_forms(b"ASMSK", &alt, 2), 10.0);
        assert_eq!(modified_forms(b"ASMSK", &alt, 3), 12.0);
    }

    #[test]
    fn the_digest_counts_missed_cleavages_lengths_and_met_excision() {
        let mut got: Vec<String> = Vec::new();
        let r = req(&[], 1);
        digest(b"MAKPEPRGGK", &r, &mut |p| {
            got.push(String::from_utf8(p.to_vec()).unwrap())
        });
        got.sort();
        // Trypsin/P cuts after K and R, before P as well: MAK | PEPR | GGK.
        assert_eq!(
            got,
            vec!["AK", "AKPEPR", "GGK", "MAK", "MAKPEPR", "PEPR", "PEPRGGK"]
                .into_iter()
                .filter(|p| p.len() >= 3)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_fasta_estimate_counts_charges_decoys_and_shared_peptides_once() {
        let dir = std::env::temp_dir().join(format!("mumdia-sizing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.fasta");
        // The same protein twice: its peptides count once.
        std::fs::write(&f, ">a\nGGGKSSSR\n>b\nGGGK\nSSSR\n").unwrap();
        let (pep, prec) = fasta_space(&f, &req(&[("Phospho", 'S')], 1)).unwrap();
        // GGGK, SSSR, GGGKSSSR; forms 1, 1 + 3, 1 + 3 = 9; x 2 charges x 2 labels.
        assert_eq!(pep, 3);
        assert_eq!(prec, 36);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_library_that_fits_is_not_split() {
        // 10.9M precursors (HYE) on a 128 GB machine.
        let p = band_plan(Some(10_900_000), Some(128_000_000_000), 32);
        assert_eq!((p.bands, p.parallel), (1, 1));
    }

    #[test]
    fn a_large_library_is_split_to_fit_the_budget() {
        // 203M precursors (immunopeptidomics) on a 128 GB desktop: 60% is 76.8 GB, so
        // bands of about 74 GB, ceil(406 / 72.8) = 6 of them, one at a time.
        let p = band_plan(Some(203_000_000), Some(128_000_000_000), 32);
        assert_eq!(p.bands, 6);
        assert_eq!(p.parallel, 1);
        assert!(p.band_gb.unwrap() <= p.budget_gb);
        // On a 1 TB host the same library needs fewer bands and runs several at once.
        let big = band_plan(Some(203_000_000), Some(1_000_000_000_000), 128);
        assert_eq!(big.bands, 1);
        let mid = band_plan(Some(203_000_000), Some(512_000_000_000), 128);
        assert_eq!(mid.bands, 2);
        assert_eq!(mid.parallel, 1);
    }

    #[test]
    fn the_bands_in_flight_always_fit_the_budget_and_the_threads() {
        for gb in [16u64, 32, 64, 128, 256, 512, 1024] {
            for m in [1u64, 5, 11, 40, 100, 203, 500, 2000] {
                for threads in [2usize, 8, 32, 128] {
                    let p = band_plan(Some(m * 1_000_000), Some(gb * 1_000_000_000), threads);
                    let band = p.band_gb.unwrap();
                    assert!(p.bands >= 1 && p.bands <= MAX_BANDS, "{p:?}");
                    assert!(p.parallel >= 1 && p.parallel <= p.bands, "{p:?}");
                    assert!(p.parallel < threads, "{p:?}");
                    if p.parallel > 1 {
                        assert!(p.parallel as f64 * band <= p.budget_gb, "{p:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn unknown_memory_proposes_no_bands() {
        let p = band_plan(Some(203_000_000), None, 32);
        assert_eq!(p.bands, 1);
    }

    #[test]
    fn this_machine_reports_its_memory() {
        let m = total_memory_bytes().expect("memory is readable on every supported OS");
        assert!(m > 1_000_000_000, "{m} bytes");
    }

    /// Against a real FASTA and the engine's own tables for it: set `MUMDIA_SIZING_FASTA`
    /// and run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn estimate_a_real_fasta() {
        let Ok(f) = std::env::var("MUMDIA_SIZING_FASTA") else {
            return;
        };
        let mut r = req(&[("Oxidation", 'M')], 1);
        r.missed_cleavages = 2;
        r.min_len = 5;
        r.max_len = 50;
        let t = std::time::Instant::now();
        let (pep, prec) = fasta_space(Path::new(&f), &r).unwrap();
        println!("peptides {pep} precursors {prec} in {:?}", t.elapsed());
    }

    #[test]
    fn meminfo_total_is_read_in_bytes() {
        assert_eq!(
            parse_meminfo_total("MemTotal:       16318408 kB\nMemFree: 1 kB\n"),
            Some(16_318_408 * 1024)
        );
    }
}
