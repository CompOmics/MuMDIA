//! Stage D `mumdia extract`: targeted 3D extraction (PLAN.md Stage D).
//!
//! Data-driven and peak-major: observed peaks probe the inverted fragment index,
//! and a candidate hypothesis is materialized only where fragment evidence
//! exists (a sparse accumulator keyed by `candidate_id`, entries created on first
//! collision). Work scales with peak-candidate collisions, not library size.
//! RT is applied as a per-candidate window post-filter (the documented fallback,
//! PLAN.md Stage D part 2); MVP is 3D so IM is absent.
//!
//! The cascade: (a) isolation-window candidate range + RT window membership,
//! (b) cheap matched-fragment presence gate, (c) matched-fragment count + a
//! consecutive-scan co-elution run, (d) apex detection. Exact intensity scores
//! are computed in the features stage from the emitted chromatograms.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{ExtractConfig, PeakClaim};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, Table};
use serde_json::json;
use tracing::info;

use mumdia_core::constants::{ppm_bounds, ISOTOPE_SPACING};

use crate::index::Library;
use crate::matchers::fragindex::FragIndex;
use crate::spectra::{load_ms1, load_ms2, Ms1Scan};
use mumdia_core::config::MatcherKind;
use mumdia_core::types::Ms2Scan;
use rayon::prelude::*;

/// Probe one (offset-corrected) query m/z against the selected matcher backend,
/// invoking `f(candidate_id, candidate_local_fragment_ordinal, predicted_intensity)`
/// for every verified match in the candidate window `[lo, hi)`. Bucketed resolves
/// the fragment ordinal via `Library::local_frag_index` (nearest stored m/z);
/// fragindex carries the true generating ordinal in `post_frag` (a semantic change
/// for candidates with fragments at sub-f32-identical m/z, per the plan). Both apply
/// the same tolerance; the fragindex index is already built at `frag_tol`.
#[inline]
fn probe_matched(
    fidx: Option<&FragIndex>,
    lib: &Library,
    frag_tol: f64,
    q_mz: f64,
    lo: u32,
    hi: u32,
    f: &mut dyn FnMut(u32, u16, f32),
) {
    match fidx {
        Some(idx) => idx.probe_peak(q_mz, lo, hi, |cid, _pmz, pint, frag| f(cid, frag, pint)),
        None => lib.page_search(q_mz, frag_tol, lo, hi, |cid, frag_mz, pi| {
            let frag = lib.local_frag_index(cid, frag_mz) as u16;
            f(cid, frag, pi);
        }),
    }
}

pub struct ExtractParams<'a> {
    pub ms2: &'a str,
    pub library_precursors: &'a str,
    pub library_fragments: &'a str,
    pub run_windows: &'a str,
    /// Optional MS1 spectra for precursor isotope-envelope features (PLAN.md
    /// Stage E). When absent, MS1 columns are null.
    pub ms1: Option<&'a str>,
    /// Optional per-run mass recalibration (search-seed `<seed>.masscal.json`):
    /// systematic fragment ppm offset + learned tolerance.
    pub mass_cal: Option<&'a str>,
    pub out_psms: &'a str,
    pub out_chrom: &'a str,
    pub cfg: &'a ExtractConfig,
    pub config_hash: &'a str,
}

/// One observed hit: scan RT, candidate-local fragment index, observed intensity
/// and observed m/z (for mass-accuracy features).
struct Hit {
    rt: f64,
    frag: u16,
    inten: f32,
    obs_mz: f64,
}

/// Index of the value in ascending `rts` nearest to `t` (binary search).
fn nearest_index(rts: &[f64], t: f64) -> usize {
    if rts.is_empty() {
        return 0;
    }
    let p = rts.partition_point(|&r| r < t);
    if p == 0 {
        0
    } else if p >= rts.len() {
        rts.len() - 1
    } else if (t - rts[p - 1]).abs() <= (rts[p] - t).abs() {
        p - 1
    } else {
        p
    }
}

/// Sum intensities of peaks within `tol_ppm` of `target` (m/z-sorted arrays).
fn sum_near(mz: &[f64], inten: &[f32], target: f64, tol_ppm: f64) -> f32 {
    if mz.is_empty() {
        return 0.0;
    }
    let (lo, hi) = ppm_bounds(target, tol_ppm);
    let s = mz.partition_point(|&m| m < lo);
    let mut acc = 0.0f32;
    let mut i = s;
    while i < mz.len() && mz[i] <= hi {
        acc += inten[i];
        i += 1;
    }
    acc
}

/// fragindex non-two-pass accumulation over isolation-window groups, in parallel.
/// Each scan belongs to exactly one window, so the groups are independent; the
/// per-candidate hit lists are merged by concatenating in (window-sorted) group
/// order. Bit-identical to serial accumulation: the per-candidate cascade rt-sorts
/// hits before the apex sum, and same-rt hits for a candidate all come from one
/// window, so the concatenation order does not affect the rt-sorted result. Only
/// the PeakClaim::None / Winner / Proportional (non-two-pass) strategies use this;
/// the co-elution two-pass path stays serial.
fn extract_accumulate_windows(
    idx: &FragIndex,
    scans: &[Ms2Scan],
    rt_lo: &[f64],
    rt_hi: &[f64],
    offset_factor: f64,
    cfg: &ExtractConfig,
) -> HashMap<u32, Vec<Hit>> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<(u64, u64), Vec<usize>> = BTreeMap::new();
    for (si, scan) in scans.iter().enumerate() {
        groups
            .entry((scan.window.lower_mz.to_bits(), scan.window.upper_mz.to_bits()))
            .or_default()
            .push(si);
    }
    let group_vec: Vec<Vec<usize>> = groups.into_values().collect();

    let partials: Vec<Vec<(u32, Vec<Hit>)>> = group_vec
        .par_iter()
        .map(|ids| {
            if ids.is_empty() {
                return Vec::new();
            }
            let w = &scans[ids[0]].window;
            let (lo, hi) = idx.candidate_range(w.lower_mz, w.upper_mz);
            if hi <= lo {
                return Vec::new();
            }
            let mut local: HashMap<u32, Vec<Hit>> = HashMap::new();
            let mut claimants: Vec<(u32, u16, f32)> = Vec::new();
            for &si in ids {
                let scan = &scans[si];
                let rt = scan.rt_seconds;
                for peak in &scan.peaks {
                    let inten = peak.intensity;
                    let q_mz = peak.mz / offset_factor;
                    let obs_mz = peak.mz;
                    claimants.clear();
                    idx.probe_peak(q_mz, lo, hi, |cid, _pmz, pint, frag| {
                        let c = cid as usize;
                        if rt < rt_lo[c] || rt > rt_hi[c] {
                            return;
                        }
                        claimants.push((cid, frag, pint));
                    });
                    if claimants.is_empty() {
                        continue;
                    }
                    match cfg.peak_claim {
                        PeakClaim::WinnerPredictedIntensity => {
                            let mut best = 0usize;
                            for i in 1..claimants.len() {
                                let a = claimants[i];
                                let b = claimants[best];
                                if a.2 > b.2 || (a.2 == b.2 && a.0 < b.0) {
                                    best = i;
                                }
                            }
                            let (cid, frag, _) = claimants[best];
                            local.entry(cid).or_default().push(Hit { rt, frag, inten, obs_mz });
                        }
                        PeakClaim::Proportional => {
                            let sump: f32 = claimants.iter().map(|c| c.2.max(0.0)).sum();
                            for &(cid, frag, pi) in &claimants {
                                let share = if sump > 0.0 {
                                    inten * (pi.max(0.0) / sump)
                                } else {
                                    inten / claimants.len() as f32
                                };
                                local.entry(cid).or_default().push(Hit { rt, frag, inten: share, obs_mz });
                            }
                        }
                        _ => {
                            for &(cid, frag, _) in &claimants {
                                local.entry(cid).or_default().push(Hit { rt, frag, inten, obs_mz });
                            }
                        }
                    }
                }
            }
            local.into_iter().collect()
        })
        .collect();

    let mut acc: HashMap<u32, Vec<Hit>> = HashMap::new();
    for part in partials {
        for (cid, hits) in part {
            acc.entry(cid).or_default().extend(hits);
        }
    }
    acc
}

pub fn run(p: ExtractParams) -> Result<(u64, u64)> {
    let t0 = Instant::now();
    let lib = Library::load(p.library_precursors, p.library_fragments, p.cfg.bucket_size)?;

    // run windows indexed by candidate_id
    let rw = Table::read(p.run_windows)?;
    let rw_cid = rw.u32("candidate_id")?;
    let rw_cal = rw.f64("rt_pred_cal")?;
    let rw_lo = rw.f64("rt_lo")?;
    let rw_hi = rw.f64("rt_hi")?;
    let ncand = lib.n_candidates();
    let mut rt_lo = vec![f64::NEG_INFINITY; ncand];
    let mut rt_hi = vec![f64::INFINITY; ncand];
    let mut rt_cal = vec![0.0f64; ncand];
    for i in 0..rw.nrows {
        let c = rw_cid[i] as usize;
        if c < ncand {
            rt_lo[c] = rw_lo[i];
            rt_hi[c] = rw_hi[i];
            rt_cal[c] = rw_cal[i];
        }
    }

    let scans = load_ms2(p.ms2)?;
    let ms1_scans: Vec<Ms1Scan> = match p.ms1 {
        Some(path) => load_ms1(path)?,
        None => Vec::new(),
    };
    let ms1_rts: Vec<f64> = ms1_scans.iter().map(|s| s.rt_seconds).collect();
    info!(
        candidates = ncand,
        scans = scans.len(),
        ms1 = ms1_scans.len(),
        "extract: loaded; probing peaks"
    );

    // Isolation-window -> sorted scan RTs, for zero-filled chromatogram grids.
    let windows: Vec<(f64, f64, Vec<f64>)> = if p.cfg.emit_window_grid {
        let mut tmp: HashMap<(u64, u64), Vec<f64>> = HashMap::new();
        for s in &scans {
            tmp.entry((s.window.lower_mz.to_bits(), s.window.upper_mz.to_bits()))
                .or_default()
                .push(s.rt_seconds);
        }
        let mut w: Vec<(f64, f64, Vec<f64>)> = tmp
            .into_iter()
            .map(|((lb, ub), mut v)| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (f64::from_bits(lb), f64::from_bits(ub), v)
            })
            .collect();
        w.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        w
    } else {
        Vec::new()
    };

    // Per-run mass recalibration (optional).
    let (frag_offset, frag_tol) = match p.mass_cal {
        Some(path) if std::path::Path::new(path).exists() => {
            let v: serde_json::Value = mumdia_io::json::read_json(path)?;
            let off = v.get("frag_ppm_offset").and_then(|x| x.as_f64()).unwrap_or(0.0);
            let tol = v
                .get("frag_tol_ppm")
                .and_then(|x| x.as_f64())
                .unwrap_or(p.cfg.frag_tol_ppm);
            info!(frag_ppm_offset = off, frag_tol_ppm = tol, "extract: using mass recalibration");
            (off, tol)
        }
        _ => (0.0, p.cfg.frag_tol_ppm),
    };
    let offset_factor = 1.0 + frag_offset * 1e-6;

    // fragindex backend, built once at the learned fragment tolerance when selected
    // (`MatcherKind::Fragindex`); otherwise the bucketed `Library::page_search` path
    // is used. `probe_matched` dispatches on this per peak.
    let fidx = matches!(p.cfg.matcher, MatcherKind::Fragindex)
        .then(|| FragIndex::build(&lib, frag_tol));

    // Peak-major accumulation.
    let mut acc: HashMap<u32, Vec<Hit>> = HashMap::new();
    // Reused per-peak buffer of (candidate_id, local_frag_index, predicted_intensity).
    let mut claimants: Vec<(u32, u16, f32)> = Vec::new();
    // Per-candidate (won, lost) peak intensity under the co-elution arbitration,
    // for the non-destructive `contested_frac` feature. Populated only on the
    // two-pass path.
    let mut contested: HashMap<u32, (f64, f64)> = HashMap::new();
    // The two co-elution strategies and the contested feature need a first pass to
    // build per-candidate elution profiles before shared peaks can be arbitrated.
    let two_pass = matches!(
        p.cfg.peak_claim,
        PeakClaim::CoelutionWinner
            | PeakClaim::CoelutionProportional
            | PeakClaim::CoelutionWinnerMargin
    ) || p.cfg.emit_contested_features;
    let claim_margin = p.cfg.peak_claim_margin as f32;

    if !two_pass {
        if let Some(idx) = fidx.as_ref() {
            // Parallel across isolation-window groups (bit-identical to serial: the
            // cascade rt-sorts each candidate's hits before summing).
            acc = extract_accumulate_windows(idx, &scans, &rt_lo, &rt_hi, offset_factor, p.cfg);
        } else {
        for scan in &scans {
            let (lo, hi) = lib.candidate_range(scan.window.lower_mz, scan.window.upper_mz);
            if hi <= lo {
                continue;
            }
            let rt = scan.rt_seconds;
            for peak in &scan.peaks {
                let inten = peak.intensity;
                let q_mz = peak.mz / offset_factor;
                // Collect every co-isolated, in-RT-window candidate matching this
                // peak, then apportion per the claim strategy. In wide DIA one peak
                // matches many candidates (~98% of fragments collide).
                claimants.clear();
                {
                    let mut push = |cid: u32, frag: u16, pi: f32| {
                        let c = cid as usize;
                        if rt < rt_lo[c] || rt > rt_hi[c] {
                            return;
                        }
                        claimants.push((cid, frag, pi));
                    };
                    probe_matched(fidx.as_ref(), &lib, frag_tol, q_mz, lo, hi, &mut push);
                }
                if claimants.is_empty() {
                    continue;
                }
                let obs_mz = peak.mz;
                match p.cfg.peak_claim {
                    PeakClaim::WinnerPredictedIntensity => {
                        let mut best = 0usize;
                        for i in 1..claimants.len() {
                            let a = claimants[i];
                            let b = claimants[best];
                            if a.2 > b.2 || (a.2 == b.2 && a.0 < b.0) {
                                best = i;
                            }
                        }
                        let (cid, frag, _) = claimants[best];
                        acc.entry(cid).or_default().push(Hit { rt, frag, inten, obs_mz });
                    }
                    PeakClaim::Proportional => {
                        let sump: f32 = claimants.iter().map(|c| c.2.max(0.0)).sum();
                        for &(cid, frag, pi) in &claimants {
                            let share = if sump > 0.0 {
                                inten * (pi.max(0.0) / sump)
                            } else {
                                inten / claimants.len() as f32
                            };
                            acc.entry(cid).or_default().push(Hit { rt, frag, inten: share, obs_mz });
                        }
                    }
                    // None (and the co-elution variants, which never reach here).
                    _ => {
                        for &(cid, frag, _) in &claimants {
                            acc.entry(cid).or_default().push(Hit { rt, frag, inten, obs_mz });
                        }
                    }
                }
            }
        }
        }
    } else {
        // PASS 1: base (None) accumulation to build honest elution profiles.
        for scan in &scans {
            let (lo, hi) = lib.candidate_range(scan.window.lower_mz, scan.window.upper_mz);
            if hi <= lo {
                continue;
            }
            let rt = scan.rt_seconds;
            for peak in &scan.peaks {
                let inten = peak.intensity;
                let q_mz = peak.mz / offset_factor;
                let obs_mz = peak.mz;
                claimants.clear();
                {
                    let mut push = |cid: u32, frag: u16, pi: f32| {
                        let c = cid as usize;
                        if rt < rt_lo[c] || rt > rt_hi[c] {
                            return;
                        }
                        claimants.push((cid, frag, pi));
                    };
                    probe_matched(fidx.as_ref(), &lib, frag_tol, q_mz, lo, hi, &mut push);
                }
                for &(cid, frag, _) in &claimants {
                    acc.entry(cid).or_default().push(Hit { rt, frag, inten, obs_mz });
                }
            }
        }
        // Per-candidate per-scan elution profile: summed matched intensity at each RT.
        let mut profile: HashMap<u32, HashMap<u64, f32>> = HashMap::new();
        for (cid, hits) in &acc {
            let m = profile.entry(*cid).or_default();
            for h in hits {
                *m.entry(h.rt.to_bits()).or_insert(0.0) += h.inten;
            }
        }
        let reassign = matches!(
            p.cfg.peak_claim,
            PeakClaim::CoelutionWinner
                | PeakClaim::CoelutionProportional
                | PeakClaim::CoelutionWinnerMargin
        );
        // PASS 2: arbitrate each shared peak by which claimant is most eluting at
        // this scan (profile height), and record won/lost intensity per candidate.
        let mut acc2: HashMap<u32, Vec<Hit>> = HashMap::new();
        for scan in &scans {
            let (lo, hi) = lib.candidate_range(scan.window.lower_mz, scan.window.upper_mz);
            if hi <= lo {
                continue;
            }
            let rt = scan.rt_seconds;
            let rtb = rt.to_bits();
            for peak in &scan.peaks {
                let inten = peak.intensity;
                let q_mz = peak.mz / offset_factor;
                let obs_mz = peak.mz;
                claimants.clear();
                {
                    let mut push = |cid: u32, frag: u16, pi: f32| {
                        let c = cid as usize;
                        if rt < rt_lo[c] || rt > rt_hi[c] {
                            return;
                        }
                        claimants.push((cid, frag, pi));
                    };
                    probe_matched(fidx.as_ref(), &lib, frag_tol, q_mz, lo, hi, &mut push);
                }
                if claimants.is_empty() {
                    continue;
                }
                let ph = |cid: u32| -> f32 {
                    profile.get(&cid).and_then(|m| m.get(&rtb)).copied().unwrap_or(0.0)
                };
                // winner: most eluting at this scan; ties -> higher predicted int -> lower cid.
                let mut best = 0usize;
                for i in 1..claimants.len() {
                    let (ci, _, pii) = claimants[i];
                    let (cb, _, pib) = claimants[best];
                    let (hi_, hb) = (ph(ci), ph(cb));
                    if hi_ > hb || (hi_ == hb && (pii > pib || (pii == pib && ci < cb))) {
                        best = i;
                    }
                }
                let win = claimants[best].0;
                let sum_ph: f32 = claimants.iter().map(|c| ph(c.0)).sum();
                // Margin gate: does the top eluter clearly dominate the runner-up?
                let top_ph = ph(win);
                let second_ph = claimants
                    .iter()
                    .filter(|c| c.0 != win)
                    .map(|c| ph(c.0))
                    .fold(0.0f32, f32::max);
                let dominant = top_ph > 0.0 && (second_ph <= 0.0 || top_ph >= claim_margin * second_ph);
                for &(cid, frag, _pi) in &claimants {
                    let e = contested.entry(cid).or_insert((0.0, 0.0));
                    if cid == win {
                        e.0 += inten as f64;
                    } else {
                        e.1 += inten as f64;
                    }
                    if reassign {
                        match p.cfg.peak_claim {
                            PeakClaim::CoelutionWinner => {
                                if cid == win {
                                    acc2.entry(cid).or_default().push(Hit { rt, frag, inten, obs_mz });
                                }
                            }
                            PeakClaim::CoelutionProportional => {
                                let share = if sum_ph > 0.0 {
                                    inten * (ph(cid) / sum_ph)
                                } else {
                                    inten / claimants.len() as f32
                                };
                                acc2.entry(cid).or_default().push(Hit { rt, frag, inten: share, obs_mz });
                            }
                            PeakClaim::CoelutionWinnerMargin => {
                                // Claim only when the top eluter dominates; else keep
                                // the peak shared (give every claimant the full peak).
                                if !dominant || cid == win {
                                    acc2.entry(cid).or_default().push(Hit { rt, frag, inten, obs_mz });
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        if reassign {
            acc = acc2;
        }
    }
    info!(materialized = acc.len(), "extract: candidates with evidence");

    // Cascade + apex per candidate.
    let scan_window = match p.cfg.scan_window_mode {
        mumdia_core::config::ScanWindowMode::Fixed => p.cfg.fixed_scan_window,
        // MVP: data-derived mode approximated by fixed default (v1 optimizer later)
        mumdia_core::config::ScanWindowMode::PeakWidthDerived => p.cfg.fixed_scan_window,
    }
    .max(1);

    // psms_extracted columns
    let (mut cid_c, mut apexrt_c, mut apexint_c) = (Vec::new(), Vec::new(), Vec::new());
    let (mut nmatch_c, mut corun_c, mut npred_c) = (Vec::new(), Vec::new(), Vec::new());
    let (mut calrt_c, mut mz_c, mut z_c) = (Vec::new(), Vec::new(), Vec::new());
    let (mut label_c, mut base_c, mut pform_c, mut prot_c, mut irt_c) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut apexim_c: Vec<Option<f64>> = Vec::new();
    // Fraction of this candidate's matched intensity that a co-eluting competitor
    // claims more strongly (co-elution arbitration); 0 when the two-pass path is off.
    let mut contested_c: Vec<f64> = Vec::new();
    // MS1 apex isotope intensities (null when no MS1 provided).
    let (mut ms1_m1, mut ms1_mono, mut ms1_i1, mut ms1_i2): (
        Vec<Option<f64>>,
        Vec<Option<f64>>,
        Vec<Option<f64>>,
        Vec<Option<f64>>,
    ) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());

    // chromatograms columns
    let (mut ch_cid, mut ch_name, mut ch_fmz, mut ch_pint) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut ch_obsmz, mut ch_rt, mut ch_int): (Vec<f64>, Vec<Vec<f32>>, Vec<Vec<f32>>) =
        (Vec::new(), Vec::new(), Vec::new());

    // Deterministic output order (a HashMap's iteration order is randomized,
    // and downstream floating-point sums in the rescorer are order-sensitive).
    let mut cand_ids: Vec<u32> = acc.keys().cloned().collect();
    cand_ids.sort_unstable();

    // Drain the accumulator into (cid, hits) in the deterministic sorted order,
    // then process candidates in parallel. Each candidate's work depends only on
    // its own hits plus read-only library/window/MS1 data, and every float
    // reduction (apex sum, wsum, chromatogram grids) is self-contained, so
    // collecting the results in cand_ids order and appending them serially below
    // yields output (PSM rows and chromatogram rows) byte-identical to the serial
    // loop. The optional Pearson gate now allocates its two scratch vectors per
    // candidate (was a hoisted reused buffer) because buffers cannot be shared
    // across parallel candidates.
    let cand_hits: Vec<(u32, Vec<Hit>)> =
        cand_ids.iter().map(|&cid| (cid, acc.remove(&cid).unwrap())).collect();

    struct CandOut {
        cid: u32,
        apex_rt: f64,
        apex_int: f32,
        n_match: i32,
        corun: i32,
        npred: i32,
        calrt: f64,
        mz: f64,
        contested: f64,
        z: i32,
        label: String,
        base: u32,
        pform: String,
        prot: String,
        irt: f32,
        ms1_m1: Option<f64>,
        ms1_mono: Option<f64>,
        ms1_i1: Option<f64>,
        ms1_i2: Option<f64>,
        /// (cid, frag_name, frag_mz, frag_obs_mz, predicted_intensity, rt, intensity)
        chrom: Vec<(u32, String, f64, f64, f32, Vec<f32>, Vec<f32>)>,
    }

    let results: Vec<Option<CandOut>> = cand_hits
        .into_par_iter()
        .map(|(cid, mut hits)| {
            // distinct matched fragments (tier b)
            let mut distinct: Vec<u16> = hits.iter().map(|h| h.frag).collect();
            distinct.sort_unstable();
            distinct.dedup();
            if distinct.len() < p.cfg.presence_min_matched.max(1) {
                return None;
            }

        // Group hits into scan groups by RT (dedupe same fragment in a scan by max).
        hits.sort_by(|a, b| a.rt.partial_cmp(&b.rt).unwrap());
        // scan groups: Vec<(rt, BTreeMap<frag,intensity>)>. A BTreeMap keeps the
        // per-scan fragment order fixed so the f32 apex sum is deterministic.
        let mut groups: Vec<(f64, BTreeMap<u16, f32>)> = Vec::new();
        for h in &hits {
            match groups.last_mut() {
                Some((rt, map)) if (*rt - h.rt).abs() < 1e-9 => {
                    let e = map.entry(h.frag).or_insert(0.0);
                    if h.inten > *e {
                        *e = h.inten;
                    }
                }
                _ => {
                    let mut m = BTreeMap::new();
                    m.insert(h.frag, h.inten);
                    groups.push((h.rt, m));
                }
            }
        }

        let (fmzs0, fints0, _) = lib.cand_frags(cid);

        // Apex: the scan group with the most distinct matched fragments, allowing
        // scans within `apex_count_tol` of that maximum (so a slightly-lower-count
        // but much more intense scan can still win), then the one maximizing the
        // summed intensity of its 3 most intense fragments. This is the diagnostic-
        // plot apex; robust to a bright single-fragment interferent that would win a
        // pure max-summed-intensity apex in chimeric DIA.
        // Distinct-fragment count per scan group, optionally smoothed by a centered
        // rolling SUM (`apex_count_window`). Low-intensity fragments flicker in/out
        // scan-to-scan; a single-scan count then spikes at noise scans and misplaces
        // the apex. The rolling sum makes the apex land in the region of *sustained*
        // fragment presence. It is deliberately a sum, not a mean: the window is
        // truncated at the profile edges, so interior positions accumulate more than
        // edge positions, which center-weights the apex toward the RT-window centre
        // (~= the predicted RT). That mild RT-prior steers off off-centre interfering
        // peaks; measured, sum beats mean by ~+300 IDs on the AIF file. Window 1
        // reproduces the exact per-scan-count behavior.
        let counts: Vec<usize> = groups.iter().map(|(_, m)| m.len()).collect();
        let w = p.cfg.apex_count_window.max(1);
        let r = w / 2;
        let smoothed: Vec<usize> = if w <= 1 {
            counts.clone()
        } else {
            (0..counts.len())
                .map(|i| {
                    let lo = i.saturating_sub(r);
                    let hi = (i + r).min(counts.len() - 1);
                    counts[lo..=hi].iter().sum()
                })
                .collect()
        };
        let maxc = smoothed.iter().copied().max().unwrap_or(0);
        let thresh = maxc.saturating_sub(p.cfg.apex_count_tol);
        // Optional Gaussian RT prior on the apex tiebreak: among count-qualified
        // scans, multiply the top-3 intensity by exp(-0.5*((rt - rt_pred_cal)/sigma)^2)
        // so a distant-from-prediction interferent inside a wide RT window cannot
        // define the apex. sigma = `apex_rt_prior_s`; 0 (or an unset rt_cal) disables it.
        let rt_prior_sigma = p.cfg.apex_rt_prior_s;
        let rt_cal_c = rt_cal[cid as usize];
        let use_prior = rt_prior_sigma > 0.0 && rt_cal_c > 0.0;
        // Signature-ion apex tiebreak: sum the OBSERVED intensity of the top-K
        // PREDICTED fragments (`apex_top_fragments`; 0 -> a default of 3) at each
        // qualifying scan, instead of the 3 brightest observed peaks. A bright
        // interferent on a non-signature ion can then no longer define the apex.
        let k_sig = if p.cfg.apex_top_fragments > 0 {
            p.cfg.apex_top_fragments
        } else {
            3
        };
        let sig: Vec<u16> = {
            let mut ord: Vec<usize> = (0..fints0.len()).collect();
            ord.sort_by(|&a, &b| fints0[b].partial_cmp(&fints0[a]).unwrap_or(std::cmp::Ordering::Equal));
            ord.into_iter().take(k_sig).map(|o| o as u16).collect()
        };
        let mut apex_rt = groups[0].0;
        let mut apex_sum = 0.0f32;
        let mut best_sig = f32::NEG_INFINITY;
        for (i, (rt, map)) in groups.iter().enumerate() {
            if map.is_empty() || smoothed[i] < thresh {
                continue;
            }
            let sig_sum: f32 = sig.iter().map(|&o| map.get(&o).copied().unwrap_or(0.0)).sum();
            let score = if use_prior {
                sig_sum * (-0.5 * ((*rt - rt_cal_c) / rt_prior_sigma).powi(2)).exp() as f32
            } else {
                sig_sum
            };
            if score > best_sig {
                best_sig = score;
                apex_rt = *rt;
                apex_sum = map.values().sum(); // report full apex intensity
            }
        }

        // Co-elution run: max consecutive scan groups with >= min_coelution frags.
        let mut best_run = 0usize;
        let mut cur = 0usize;
        for (_, map) in &groups {
            if map.len() >= p.cfg.presence_min_coelution.max(1) {
                cur += 1;
                best_run = best_run.max(cur);
            } else {
                cur = 0;
            }
        }

        // Acceptance (tier c): presence, consecutive-scan run, and matched
        // fraction of the predicted fragments (symmetric discriminator).
        let matched_fraction = distinct.len() as f64 / (fmzs0.len().max(1) as f64);
        if distinct.len() < p.cfg.presence_min_fragments.max(1)
            || best_run < scan_window
            || best_run < p.cfg.min_coelution_run
            || matched_fraction < p.cfg.min_matched_fraction
        {
            return None;
        }

        // Optional tier-d Pearson gate (kept for configurability; matched
        // fraction above is the primary symmetric discriminator).
        let apex_map = groups
            .iter()
            .find(|(rt, _)| (*rt - apex_rt).abs() < 1e-9)
            .map(|(_, m)| m);
        if p.cfg.min_frag_corr > 0.0 {
            if let Some(map) = apex_map {
                let obs: Vec<f64> = (0..fmzs0.len())
                    .map(|k| *map.get(&(k as u16)).unwrap_or(&0.0) as f64)
                    .collect();
                let pred: Vec<f64> = fints0.iter().map(|x| *x as f64).collect();
                if crate::stats::pearson(&obs, &pred) < p.cfg.min_frag_corr {
                    return None;
                }
            }
        }

        let c = &lib.cands[cid as usize];
        let contested_val = {
            let (w, l) = contested.get(&cid).copied().unwrap_or((0.0, 0.0));
            if w + l > 0.0 { l / (w + l) } else { 0.0 }
        };

        // MS1 apex isotope intensities: nearest MS1 scan to the apex RT.
        let (o_ms1_m1, o_ms1_mono, o_ms1_i1, o_ms1_i2) = if ms1_scans.is_empty() {
            (None, None, None, None)
        } else {
            let j = nearest_index(&ms1_rts, apex_rt);
            let s = &ms1_scans[j];
            let z = c.charge as f64;
            let sp = ISOTOPE_SPACING / z;
            let tol = p.cfg.prec_tol_ppm;
            (
                Some(sum_near(&s.mz, &s.intensity, c.precursor_mz - sp, tol) as f64),
                Some(sum_near(&s.mz, &s.intensity, c.precursor_mz, tol) as f64),
                Some(sum_near(&s.mz, &s.intensity, c.precursor_mz + sp, tol) as f64),
                Some(sum_near(&s.mz, &s.intensity, c.precursor_mz + 2.0 * sp, tol) as f64),
            )
        };

        // Per-fragment intensity-weighted observed m/z (for mass accuracy).
        let mut wsum: HashMap<u16, (f64, f64)> = HashMap::new(); // frag -> (sum w*mz, sum w)
        for h in &hits {
            let e = wsum.entry(h.frag).or_insert((0.0, 0.0));
            e.0 += h.obs_mz * h.inten as f64;
            e.1 += h.inten as f64;
        }

        let mut chrom_rows: Vec<(u32, String, f64, f64, f32, Vec<f32>, Vec<f32>)> = Vec::new();

        // Emit chromatograms. When emit_window_grid is set, each fragment is sampled
        // on the full isolation-window scan grid (all scans of the covering window(s)
        // within the RT window), with 0.0 where the fragment is absent, so the elution
        // profile drops to zero between peaks (correct boundary calling downstream).
        let (fmzs, fints, fnames) = lib.cand_frags(cid);
        let mut per_frag: HashMap<u16, Vec<(f64, f32)>> = HashMap::new();
        for (rt, map) in &groups {
            for (&frag, &inten) in map {
                per_frag.entry(frag).or_default().push((*rt, inten));
            }
        }
        let grid: Vec<f64> = if !windows.is_empty() {
            let pm = lib.cands[cid as usize].precursor_mz;
            let (lo, hi) = (rt_lo[cid as usize], rt_hi[cid as usize]);
            let mut g: Vec<f64> = Vec::new();
            for (wl, wu, rts) in &windows {
                if *wl <= pm && pm <= *wu {
                    let a = rts.partition_point(|&r| r < lo);
                    let b = rts.partition_point(|&r| r <= hi);
                    g.extend_from_slice(&rts[a..b]);
                }
            }
            g.sort_by(|a, b| a.partial_cmp(b).unwrap());
            g.dedup();
            g
        } else {
            Vec::new()
        };
        // Emit EVERY predicted transition, not just the observed ones. A predicted
        // fragment never matched anywhere gets a zero-intensity row (all-zero grid
        // trace, or an empty series without the grid) so the feature families see
        // the full predicted set: similarity/entropy/ion-series/coelution get the
        // correct denominator and a missing strong ion is penalized. obs m/z falls
        // back to the theoretical m/z, which is harmless because the mass-accuracy
        // features only count fragments with obs_apex > 0.
        for fi in 0..fmzs.len() {
            let frag = fi as u16;
            let obs_mz = wsum
                .get(&frag)
                .map(|(sm, sw)| if *sw > 0.0 { sm / sw } else { fmzs[fi] })
                .unwrap_or(fmzs[fi]);
            let (rts, ints): (Vec<f32>, Vec<f32>) = if !grid.is_empty() {
                let m: HashMap<u64, f32> = per_frag
                    .get(&frag)
                    .map(|v| v.iter().map(|(r, i)| (r.to_bits(), *i)).collect())
                    .unwrap_or_default();
                (
                    grid.iter().map(|r| *r as f32).collect(),
                    grid.iter().map(|r| *m.get(&r.to_bits()).unwrap_or(&0.0)).collect(),
                )
            } else {
                match per_frag.get(&frag) {
                    Some(v) => {
                        let mut s = v.clone();
                        s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        (
                            s.iter().map(|(r, _)| *r as f32).collect(),
                            s.iter().map(|(_, i)| *i).collect(),
                        )
                    }
                    None => (Vec::new(), Vec::new()),
                }
            };
            chrom_rows.push((cid, fnames[fi].clone(), fmzs[fi], obs_mz, fints[fi], rts, ints));
        }

        // MS1 isotope XICs (mono/+1/+2) sampled on the same scan grid as the
        // fragments, so the features stage can correlate the MS1 precursor
        // envelope against the MS2 fragments over the elution peak (DIA-NN
        // Ms1.Profile.Corr class). Grid mode only; nearest MS1 scan per grid RT.
        if !ms1_scans.is_empty() && !grid.is_empty() {
            let sp = ISOTOPE_SPACING / c.charge as f64;
            let tol = p.cfg.prec_tol_ppm;
            let grid_rt: Vec<f32> = grid.iter().map(|&r| r as f32).collect();
            for (nm, dmz) in [("ms1_mono", 0.0), ("ms1_iso1", sp), ("ms1_iso2", 2.0 * sp)] {
                let mz = c.precursor_mz + dmz;
                let ints: Vec<f32> = grid
                    .iter()
                    .map(|&r| {
                        let j = nearest_index(&ms1_rts, r);
                        sum_near(&ms1_scans[j].mz, &ms1_scans[j].intensity, mz, tol) as f32
                    })
                    .collect();
                chrom_rows.push((cid, nm.to_string(), mz, mz, 0.0, grid_rt.clone(), ints));
            }
        }

        Some(CandOut {
            cid,
            apex_rt,
            apex_int: apex_sum,
            n_match: distinct.len() as i32,
            corun: best_run as i32,
            npred: fmzs0.len() as i32,
            calrt: rt_cal[cid as usize],
            mz: c.precursor_mz,
            contested: contested_val,
            z: c.charge,
            label: if c.is_decoy { "decoy" } else { "target" }.to_string(),
            base: c.base_peptide_id,
            pform: c.peptidoform.clone(),
            prot: c.protein.clone(),
            irt: c.predicted_irt,
            ms1_m1: o_ms1_m1,
            ms1_mono: o_ms1_mono,
            ms1_i1: o_ms1_i1,
            ms1_i2: o_ms1_i2,
            chrom: chrom_rows,
        })
    })
    .collect();

    // Append results in the deterministic cand_ids order (parallel work above was
    // order-preserving via `collect`), reproducing the serial push order exactly.
    let mut n_accepted = 0u64;
    for r in results.into_iter().flatten() {
        n_accepted += 1;
        cid_c.push(r.cid);
        apexrt_c.push(r.apex_rt);
        apexim_c.push(None);
        apexint_c.push(r.apex_int);
        nmatch_c.push(r.n_match);
        corun_c.push(r.corun);
        npred_c.push(r.npred);
        calrt_c.push(r.calrt);
        mz_c.push(r.mz);
        contested_c.push(r.contested);
        z_c.push(r.z);
        label_c.push(r.label);
        base_c.push(r.base);
        pform_c.push(r.pform);
        prot_c.push(r.prot);
        irt_c.push(r.irt);
        ms1_m1.push(r.ms1_m1);
        ms1_mono.push(r.ms1_mono);
        ms1_i1.push(r.ms1_i1);
        ms1_i2.push(r.ms1_i2);
        for (cc, nm, fmz, omz, pint, rt, it) in r.chrom {
            ch_cid.push(cc);
            ch_name.push(nm);
            ch_fmz.push(fmz);
            ch_obsmz.push(omz);
            ch_pint.push(pint);
            ch_rt.push(rt);
            ch_int.push(it);
        }
    }

    let n_psms = write_table(
        p.out_psms,
        vec![
            Col::U32("candidate_id".into(), cid_c),
            Col::F64("apex_rt".into(), apexrt_c),
            Col::OptF64("apex_im".into(), apexim_c),
            Col::F32("apex_intensity".into(), apexint_c),
            Col::I32("n_matched_fragments".into(), nmatch_c),
            Col::I32("n_predicted_fragments".into(), npred_c),
            Col::I32("coelution_run".into(), corun_c),
            Col::F64("rt_pred_cal".into(), calrt_c),
            Col::F64("precursor_mz".into(), mz_c),
            Col::I32("charge".into(), z_c),
            Col::Str("label".into(), label_c),
            Col::U32("base_peptide_id".into(), base_c),
            Col::Str("peptidoform".into(), pform_c),
            Col::Str("protein".into(), prot_c),
            Col::F32("predicted_irt".into(), irt_c),
            Col::F64("contested_frac".into(), contested_c),
            Col::OptF64("ms1_isom1".into(), ms1_m1),
            Col::OptF64("ms1_mono".into(), ms1_mono),
            Col::OptF64("ms1_iso1".into(), ms1_i1),
            Col::OptF64("ms1_iso2".into(), ms1_i2),
        ],
    )?;

    let n_chrom = write_table(
        p.out_chrom,
        vec![
            Col::U32("candidate_id".into(), ch_cid),
            Col::Str("frag_name".into(), ch_name),
            Col::F64("frag_mz".into(), ch_fmz),
            Col::F64("frag_obs_mz".into(), ch_obsmz),
            Col::F32("predicted_intensity".into(), ch_pint),
            Col::ListF32("rt".into(), ch_rt),
            Col::ListF32("intensity".into(), ch_int),
        ],
    )?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("accepted".to_string(), json!(n_accepted));
    stats.insert("scan_window".to_string(), json!(scan_window));
    for (path, schema, rows) in [
        (p.out_psms, artifact::PSMS_EXTRACTED, n_psms),
        (p.out_chrom, artifact::CHROMATOGRAMS, n_chrom),
    ] {
        ArtifactReport {
            logical_name: schema.0.to_string(),
            schema_name: schema.0.to_string(),
            schema_version: schema.1,
            stage: "extract".to_string(),
            rows,
            content_hash: mumdia_io::hash::blake3_file(path)?,
            params: json!({
                "frag_tol_ppm": p.cfg.frag_tol_ppm,
                "presence_min_fragments": p.cfg.presence_min_fragments,
                "presence_min_coelution": p.cfg.presence_min_coelution,
                "scan_window": scan_window,
            }),
            stats: stats.clone(),
            model_identity: None,
            elapsed_ms: elapsed,
        }
        .write_for(path)?;
    }

    info!(
        accepted = n_accepted,
        chromatograms = n_chrom,
        elapsed_ms = elapsed,
        "extract: done"
    );
    Ok((n_psms, n_chrom))
}
