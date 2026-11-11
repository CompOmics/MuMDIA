//! Q1-deconvolved evidence of scanning-quadrupole spectra (`search_seed.q1_deconvolved`,
//! `extract.q1_deconvolved`).
//!
//! In a SCIEX ZT Scan DIA run the quadrupole sweeps the precursor range once a cycle in steps
//! far smaller than its window, so a fragment ion shows up at one m/z in a run of consecutive
//! steps of a cycle, centred where its precursor sits on the sweep. A peak can therefore be
//! given a Q1 position of its own, before any candidate is considered: the centroid of the
//! trace it belongs to along the Q1 axis. This module computes that position for every MS2
//! peak of a run ([`Q1Dec::build`]), and the search stages then count a peak for a candidate
//! only when its position lies within a tolerance of the candidate's expected Q1 centre, so a
//! fragment m/z that another precursor shares, transmitted around that precursor's Q1 position,
//! no longer counts for this one.
//!
//! **Traces.** Within one cycle, a peak of step `s` and a peak of step `s + 1` are linked when
//! each is the other's nearest within the fragment tolerance (mutual nearest neighbours, so a
//! trace is a chain and never branches). A chain is split where it falls below [`VALLEY`] of
//! the smaller of the maxima either side, so two precursors a few steps apart that share a
//! fragment m/z give two traces when the fragment's signal dips between them. Every peak of a
//! trace gets the trace's intensity-weighted mean step target; a peak linked to nothing is its
//! own trace, at its own step's target. Steps are kept apart throughout: nothing is summed
//! over a cycle (summing the steps of a cycle before deciding lost 13.7% of the
//! identifications, docs/09_extract.md, section 10).
//!
//! **Storage.** One signed byte per peak, the centroid minus the peak's own step target in
//! 1/[`UNITS_PER_STEP`] steps (+-15.9 steps), held beside the scans in memory by the stage
//! that reads it and never written: no spectra artifact changes, so with the keys off nothing
//! does.
//!
//! **The expected centre** of a candidate ([`expected_centre`]) is the mean m/z of its
//! precursor's averagine isotope envelope (the quadrupole transmits the whole envelope)
//! less the calibrated window offset (`rt_im_train.q1_calibrate`, in extract) or the nominal
//! zero offset (in the seed, which runs before the calibration). Extract's tolerance is
//! learned from the confident seed PSMs by the Q1 calibration (`q1_calibration.deconvolution`,
//! [`learn_tolerance`]); the seed's is [`SEED_TOL_STEPS`].
//!
//! Deterministic: links, splits and centroids run in step and m/z order; ties keep the lower
//! index.

use mumdia_core::config::Q1FragModel;
use mumdia_core::constants::{averagine_isotopes, ppm_bounds, ISOTOPE_SPACING, PROTON};
use mumdia_core::types::{Ms2Scan, Peak};
use rayon::prelude::*;
use serde_json::json;

use super::q1frag::{isotope_weights, offset_mz, Q1Grid, ISOTOPES};
use crate::calibrate::percentile;

/// Units of a stored centroid offset per step.
pub const UNITS_PER_STEP: f64 = 8.0;
/// A trace is split where it falls below this share of the smaller of the maxima either side.
pub const VALLEY: f64 = 0.5;
/// The seed's tolerance (steps) on a peak's Q1 centroid against the candidate's nominal
/// expected centre. The seed runs before the Q1 calibration, so it reads the nominal window
/// (no offset); the calibrated offset of the K562 ZT Scan runs stays within 0.4 steps of it
/// across the precursor range (docs/08, section 4h).
pub const SEED_TOL_STEPS: f64 = 2.0;
/// The fallback tolerance rule without decoys: median plus this many robust standard
/// deviations of the anchors' peaks' centroid offsets.
pub const TOL_SIGMAS: f64 = 3.0;
/// The smallest tolerance learned (steps).
pub const MIN_TOL_STEPS: f64 = 0.5;
/// Most anchors, and most decoys, the tolerance is learned from (every k-th above it).
pub const LEARN_ANCHORS: usize = 3_000;
/// Steps read beyond the calibrated window's reach on each side when the tolerance is
/// learned, so the traces of the anchors' covering steps are not cut short.
pub const LEARN_MARGIN_STEPS: f64 = 6.0;

const NONE: u32 = u32::MAX;
const MAD_TO_SIGMA: f64 = 1.4826;

/// For every peak of `a` (m/z ascending), the index of the nearest peak of `b` (m/z
/// ascending) within `tol_ppm`, [`NONE`] without one; the lower index on a tie. Linear.
fn nearest_all(a: &[Peak], b: &[Peak], tol_ppm: f64) -> Vec<u32> {
    let mut out = vec![NONE; a.len()];
    let mut j = 0usize;
    for (i, pa) in a.iter().enumerate() {
        let x = pa.mz as f64;
        let (lo, hi) = ppm_bounds(x, tol_ppm);
        while j < b.len() && (b[j].mz as f64) < lo {
            j += 1;
        }
        let (mut k, mut best, mut bd) = (j, NONE, f64::MAX);
        while k < b.len() && (b[k].mz as f64) <= hi {
            let d = (b[k].mz as f64 - x).abs();
            if d < bd {
                bd = d;
                best = k as u32;
            }
            k += 1;
        }
        out[i] = best;
    }
    out
}

/// The mutual-nearest links from the peaks of `a` to those of `b` (adjacent steps of one
/// cycle): per peak of `a`, its partner in `b` or [`NONE`].
fn mutual_links(a: &[Peak], b: &[Peak], tol_ppm: f64) -> Vec<u32> {
    let ab = nearest_all(a, b, tol_ppm);
    let ba = nearest_all(b, a, tol_ppm);
    ab.iter()
        .enumerate()
        .map(|(i, &k)| {
            if k != NONE && ba[k as usize] == i as u32 {
                k
            } else {
                NONE
            }
        })
        .collect()
}

/// The segments `[a, b)` of the intensities `v` of one trace, split after each local minimum
/// below [`VALLEY`] of the smaller of the maximum before it (since the last split) and the
/// maximum after it.
fn segments(v: &[f64]) -> Vec<(usize, usize)> {
    let n = v.len();
    let mut cuts = vec![0usize];
    for q in 1..n.saturating_sub(1) {
        if v[q] < v[q - 1] && v[q] <= v[q + 1] {
            let start = *cuts.last().expect("cuts start with 0");
            let left = v[start..q].iter().copied().fold(0.0f64, f64::max);
            let right = v[q + 1..].iter().copied().fold(0.0f64, f64::max);
            if v[q] < VALLEY * left.min(right) {
                cuts.push(q + 1);
            }
        }
    }
    cuts.push(n);
    cuts.windows(2).map(|w| (w[0], w[1])).collect()
}

/// A centroid offset `dx` (m/z) at step spacing `step` as stored.
#[inline]
fn encode(dx: f64, step: f64) -> i8 {
    let u = (dx / step * UNITS_PER_STEP).round();
    u.clamp(-127.0, 127.0) as i8
}

/// The centroid offsets of a run of consecutive steps of one cycle: `peaks[s]` the peaks of
/// step `s` (empty where the cycle has no scan of it), `target[s]` its target. Returns, per
/// step and peak, its trace's centroid minus the step's target, as stored.
pub fn annotate_run(peaks: &[&[Peak]], target: &[f64], step: f64, tol_ppm: f64) -> Vec<Vec<i8>> {
    let n = peaks.len();
    let next: Vec<Vec<u32>> = (0..n)
        .map(|s| {
            if s + 1 < n {
                mutual_links(peaks[s], peaks[s + 1], tol_ppm)
            } else {
                vec![NONE; peaks[s].len()]
            }
        })
        .collect();
    let mut has_prev: Vec<Vec<bool>> = peaks.iter().map(|p| vec![false; p.len()]).collect();
    for s in 0..n.saturating_sub(1) {
        for &k in &next[s] {
            if k != NONE {
                has_prev[s + 1][k as usize] = true;
            }
        }
    }
    let mut out: Vec<Vec<i8>> = peaks.iter().map(|p| vec![0i8; p.len()]).collect();
    let mut chain: Vec<(usize, usize)> = Vec::new();
    let mut v: Vec<f64> = Vec::new();
    for s in 0..n {
        let starts: Vec<usize> = (0..peaks[s].len()).filter(|&i| !has_prev[s][i]).collect();
        for i in starts {
            chain.clear();
            let (mut ss, mut ii) = (s, i);
            loop {
                chain.push((ss, ii));
                let k = next[ss][ii];
                if k == NONE {
                    break;
                }
                ss += 1;
                ii = k as usize;
            }
            if chain.len() == 1 {
                // Its own trace, at its own step: offset 0.
                continue;
            }
            v.clear();
            v.extend(chain.iter().map(|&(ss, ii)| peaks[ss][ii].intensity as f64));
            for (a, b) in segments(&v) {
                let (mut sw, mut st) = (0.0f64, 0.0f64);
                for q in a..b {
                    sw += v[q];
                    st += v[q] * target[chain[q].0];
                }
                let cen = if sw > 0.0 {
                    st / sw
                } else {
                    (a..b).map(|q| target[chain[q].0]).sum::<f64>() / (b - a) as f64
                };
                for &(ss, ii) in &chain[a..b] {
                    out[ss][ii] = encode(cen - target[ss], step);
                }
            }
        }
    }
    out
}

/// The step spacing of a sweep: the median spacing of its targets.
pub fn grid_step(target: &[f64]) -> f64 {
    let d: Vec<f64> = target.windows(2).map(|w| w[1] - w[0]).collect();
    if d.is_empty() {
        0.0
    } else {
        percentile(&d, 0.5)
    }
}

/// Every MS2 peak's Q1 centroid of one run's scans.
pub struct Q1Dec {
    /// By `scan_index`, per peak: its trace's centroid minus the scan's own step target, in
    /// 1/[`UNITS_PER_STEP`] steps; empty for a scan outside the sweep (and for the MS1 scan
    /// indices between the MS2 ones). Keyed by the scan's own index rather than its position
    /// in a stage's slice, so a lent slice and every probe site read it alike.
    off: Vec<Vec<i8>>,
    /// The sweep's step spacing (m/z).
    pub step_mz: f64,
}

impl Q1Dec {
    /// The centroids of every peak of `scans` on the sweep `grid` (its cycles in parallel),
    /// linking peaks within `tol_ppm`.
    pub fn build(scans: &[Ms2Scan], grid: &Q1Grid, tol_ppm: f64) -> Q1Dec {
        let step = grid_step(&grid.target);
        let per_cycle: Vec<Vec<(u32, Vec<i8>)>> = (0..grid.n_cycles)
            .into_par_iter()
            .map(|c| {
                let rows: Vec<Option<usize>> = (0..grid.n_steps)
                    .map(|s| grid.scan(c as u32, s as u32))
                    .collect();
                let peaks: Vec<&[Peak]> = rows
                    .iter()
                    .map(|r| r.map_or(&[][..], |si| &scans[si].peaks[..]))
                    .collect();
                let off = annotate_run(&peaks, &grid.target, step, tol_ppm);
                rows.iter()
                    .zip(off)
                    .filter_map(|(r, o)| r.map(|si| (si as u32, o)))
                    .collect()
            })
            .collect();
        let n = scans
            .iter()
            .map(|s| s.scan_index as usize + 1)
            .max()
            .unwrap_or(0);
        let mut off = vec![Vec::new(); n];
        for cyc in per_cycle {
            for (si, o) in cyc {
                off[scans[si as usize].scan_index as usize] = o;
            }
        }
        Q1Dec { off, step_mz: step }
    }

    /// The Q1 centroid (target units, m/z) of peak `j` of `scan`.
    #[inline]
    pub fn centroid(&self, scan: &Ms2Scan, j: usize) -> f64 {
        let t = scan.window.target_mz;
        match self
            .off
            .get(scan.scan_index as usize)
            .and_then(|v| v.get(j))
        {
            Some(&o) => t + o as f64 / UNITS_PER_STEP * self.step_mz,
            None => t,
        }
    }

    /// Bytes held (the offsets), for the log.
    pub fn bytes(&self) -> usize {
        self.off.iter().map(|v| v.len()).sum()
    }

    /// The share of peaks that are linked to a neighbouring step (a nonzero offset or a
    /// trace centred on their own step cannot be told apart here, so this counts nonzero
    /// offsets), for the log.
    pub fn moved_share(&self) -> f64 {
        let (mut n, mut k) = (0usize, 0usize);
        for v in &self.off {
            n += v.len();
            k += v.iter().filter(|&&o| o != 0).count();
        }
        if n == 0 {
            0.0
        } else {
            k as f64 / n as f64
        }
    }
}

/// The expected Q1 centre (target units, m/z) of a precursor of m/z `pm` and charge `z`: the
/// mean m/z of its averagine isotope envelope, less the window offset of the calibrated model
/// `m` at `rt` (the nominal 0 without one), as [`super::q1frag::CandQ1::new`] computes it.
pub fn expected_centre(m: Option<&Q1FragModel>, pm: f64, z: i32, rt: f64) -> f64 {
    let z = z.max(1);
    let sp = ISOTOPE_SPACING / z as f64;
    let pre = averagine_isotopes((pm - PROTON) * z as f64, ISOTOPES);
    let ipre: f64 = pre.iter().enumerate().map(|(i, w)| i as f64 * w).sum();
    pm + sp * ipre - m.map_or(0.0, |m| offset_mz(m, pm, None, rt))
}

/// The expected Q1 centre of fragment `fmz` (charge `fz`) of that precursor, as
/// [`super::q1frag::CandQ1::new`] computes it: the precursor isotopes that leave the fragment
/// monoisotopic weigh more (a large fragment carries most heavy atoms), so its centre lies
/// below the envelope's, by up to about a step for a doubly charged precursor of 2,000 Da.
pub fn fragment_centre(m: &Q1FragModel, pm: f64, z: i32, fmz: f64, fz: i32, rt: f64) -> f64 {
    let z = z.max(1);
    let sp = ISOTOPE_SPACING / z as f64;
    let w = isotope_weights(pm, z, fmz, fz.max(1));
    let ibar: f64 = w.iter().enumerate().map(|(k, x)| k as f64 * x).sum();
    pm + sp * ibar - offset_mz(m, pm, Some(fmz), rt)
}

/// Units of a stored fragment-centre shift per step.
pub const SHIFT_UNITS_PER_STEP: f64 = 16.0;

/// A fragment centre against its candidate's centre (m/z), as stored at step `step`.
pub fn encode_shift(dx: f64, step: f64) -> i8 {
    (dx / step * SHIFT_UNITS_PER_STEP)
        .round()
        .clamp(-127.0, 127.0) as i8
}

/// The admission of a peak for a candidate under the deconvolved evidence: its Q1 centroid
/// within `tol_mz` of the expected centre of the candidate's matched fragment (extract), or of
/// the candidate's envelope (the seed, which probes without fragment ordinals).
pub struct Q1Gate {
    pub dec: Q1Dec,
    /// Per candidate of the stage (local ids), its expected centre.
    pub centre: Vec<f32>,
    /// Per candidate, the index of its first library fragment in `frag_shift`, and per
    /// library fragment its expected centre against its candidate's, in
    /// 1/[`SHIFT_UNITS_PER_STEP`] steps; empty in the seed.
    pub frag_first: Vec<u32>,
    pub frag_shift: Vec<i8>,
    pub tol_mz: f64,
}

impl Q1Gate {
    /// Whether peak `j` of `scan` counts for fragment `frag` (the candidate's library
    /// ordinal) of candidate `c`.
    #[inline]
    pub fn admits_frag(&self, c: usize, frag: u16, scan: &Ms2Scan, j: usize) -> bool {
        let x = self.dec.centroid(scan, j);
        let shift = match self.frag_first.get(c) {
            Some(&f0) => self
                .frag_shift
                .get(f0 as usize + frag as usize)
                .map_or(0.0, |&v| v as f64 / SHIFT_UNITS_PER_STEP * self.dec.step_mz),
            None => 0.0,
        };
        (x - (self.centre[c] as f64 + shift)).abs() <= self.tol_mz
    }

    /// Whether peak `j` of `scan` counts for candidate `c`, against its envelope's centre.
    #[inline]
    pub fn admits(&self, c: usize, scan: &Ms2Scan, j: usize) -> bool {
        let x = self.dec.centroid(scan, j);
        (x - self.centre[c] as f64).abs() <= self.tol_mz
    }
}

/// The `q1d_*` columns of psms_extracted v8 (`features.q1_deconvolved`), in the order
/// [`row_evidence`] returns them; also the feature names.
pub const Q1D_COLUMNS: &[&str] = &[
    "q1d_n_frag",
    "q1d_n_incons",
    "q1d_share",
    "q1d_pred_share",
    "q1d_apex_share",
];

/// Cycles either side of the apex cycle [`row_evidence`] reads.
pub const EVIDENCE_CYCLES: u32 = 2;
/// Steps either side of the step nearest the precursor [`row_evidence`] considers (those
/// whose window as read covers the precursor are read).
const EVIDENCE_STEPS: i64 = 4;

/// The `q1d_*` values of a row without evidence (its apex not in the sweep).
pub fn empty_evidence() -> Vec<f64> {
    vec![0.0; Q1D_COLUMNS.len()]
}

/// The Q1-deconvolved evidence of a row (`features.q1_deconvolved`), read from the spectra
/// at its apex scan `apex_scan` (an index into `scans`): for each library fragment of local
/// candidate `c` (matched at `targets`, predicted `preds`), the nearest peak within `tol_ppm`
/// in every scan of the apex cycle and the [`EVIDENCE_CYCLES`] either side whose window as
/// read covers the precursor m/z `pm`; a matched peak is consistent when `gate` admits it for
/// that fragment. Returns [`Q1D_COLUMNS`]: the fragments with a consistent match, the
/// fragments matched only inconsistently, the consistent share of the matched intensity, the
/// predicted-intensity share of the fragments with a consistent match, and the consistent
/// share in the apex cycle alone.
#[allow(clippy::too_many_arguments)]
pub fn row_evidence(
    gate: &Q1Gate,
    grid: &Q1Grid,
    scans: &[Ms2Scan],
    c: usize,
    apex_scan: usize,
    pm: f64,
    targets: &[f64],
    preds: &[f64],
    tol_ppm: f64,
) -> Vec<f64> {
    let Some(&c0) = grid.scan_cycle.get(apex_scan) else {
        return empty_evidence();
    };
    if c0 == u32::MAX {
        return empty_evidence();
    }
    let nf = targets.len();
    let (mut cons, mut seen) = (vec![false; nf], vec![false; nf]);
    let (mut all_i, mut cons_i, mut apex_all, mut apex_cons) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mid = grid.nearest_step(pm) as i64;
    let first = c0.saturating_sub(EVIDENCE_CYCLES);
    let last = (c0 + EVIDENCE_CYCLES).min(grid.n_cycles.saturating_sub(1) as u32);
    for cy in first..=last {
        for s in (mid - EVIDENCE_STEPS)..=(mid + EVIDENCE_STEPS) {
            if s < 0 {
                continue;
            }
            let Some(si) = grid.scan(cy, s as u32) else {
                continue;
            };
            let scan = &scans[si];
            if !(scan.window.lower_mz <= pm && pm <= scan.window.upper_mz) {
                continue;
            }
            for f in 0..nf {
                let Some(j) = crate::spectra::nearest_peak(&scan.peaks, targets[f], tol_ppm) else {
                    continue;
                };
                let v = scan.peaks[j].intensity as f64;
                seen[f] = true;
                all_i += v;
                let ok = gate.admits_frag(c, f as u16, scan, j);
                if ok {
                    cons[f] = true;
                    cons_i += v;
                }
                if cy == c0 {
                    apex_all += v;
                    if ok {
                        apex_cons += v;
                    }
                }
            }
        }
    }
    let n_cons = cons.iter().filter(|&&x| x).count() as f64;
    let n_incons = (0..nf).filter(|&f| seen[f] && !cons[f]).count() as f64;
    let ptot: f64 = preds.iter().map(|p| p.max(0.0)).sum();
    let pcons: f64 = (0..nf)
        .filter(|&f| cons[f])
        .map(|f| preds[f].max(0.0))
        .sum();
    let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
    vec![
        n_cons,
        n_incons,
        ratio(cons_i, all_i),
        ratio(pcons, ptot),
        ratio(apex_cons, apex_all),
    ]
}

/// The tolerance grid of [`learn_tolerance`] (steps): 0.5 to 4 by 1/8.
fn tol_grid() -> Vec<f64> {
    (4..=32).map(|k| k as f64 / 8.0).collect()
}

/// The share of `abs` (sorted ascending) at or below `t`.
fn share_le(abs: &[f64], t: f64) -> f64 {
    if abs.is_empty() {
        0.0
    } else {
        abs.partition_point(|&x| x <= t) as f64 / abs.len() as f64
    }
}

/// The deconvolution tolerance (steps) from the centroid offsets (steps, against the expected
/// centres of the matched fragments) of the confident seed PSMs' peaks (`anchor`) and of
/// decoy seed PSMs' peaks (`decoy`, the null): the tolerance on a grid of 1/8 step from 0.5
/// to 4 where the share of anchor peaks kept most exceeds the share of decoy peaks kept (the
/// smaller on a tie), at least [`MIN_TOL_STEPS`]; without decoys, `|median| + TOL_SIGMAS`
/// robust standard deviations of the anchors'. Returns it and the `deconvolution` block of
/// `cal.json`.
pub fn learn_tolerance(
    anchor: &[f64],
    decoy: &[f64],
    n_anchors: usize,
    n_decoys: usize,
) -> (f64, serde_json::Value) {
    let med = if anchor.is_empty() {
        0.0
    } else {
        percentile(anchor, 0.5)
    };
    let dev: Vec<f64> = anchor.iter().map(|x| (x - med).abs()).collect();
    let sd = if dev.is_empty() {
        0.0
    } else {
        MAD_TO_SIGMA * percentile(&dev, 0.5)
    };
    let mut aa: Vec<f64> = anchor.iter().map(|x| x.abs()).collect();
    let mut da: Vec<f64> = decoy.iter().map(|x| x.abs()).collect();
    aa.sort_by(f64::total_cmp);
    da.sort_by(f64::total_cmp);
    let curve: Vec<(f64, f64, f64)> = tol_grid()
        .into_iter()
        .map(|t| (t, share_le(&aa, t), share_le(&da, t)))
        .collect();
    let (tol, rule) = if da.is_empty() || aa.is_empty() {
        (
            (med.abs() + TOL_SIGMAS * sd).max(MIN_TOL_STEPS),
            "robust_sd",
        )
    } else {
        let mut best = (MIN_TOL_STEPS, f64::NEG_INFINITY);
        for &(t, a, d) in &curve {
            if a - d > best.1 {
                best = (t, a - d);
            }
        }
        (best.0.max(MIN_TOL_STEPS), "anchor_minus_decoy")
    };
    let q = |v: &[f64], p: f64| if v.is_empty() { 0.0 } else { percentile(v, p) };
    let block = json!({
        "version": 1,
        "anchors": n_anchors,
        "decoys": n_decoys,
        "anchor_peaks": anchor.len(),
        "decoy_peaks": decoy.len(),
        "rule": rule,
        "valley": VALLEY,
        "tol_steps": tol,
        "anchor_offset_steps": {
            "median": med,
            "sd": sd,
            "p05": q(anchor, 0.05),
            "p25": q(anchor, 0.25),
            "p75": q(anchor, 0.75),
            "p95": q(anchor, 0.95),
        },
        "kept": curve
            .iter()
            .map(|&(t, a, d)| json!({"tol_steps": t, "anchor": a, "decoy": d}))
            .collect::<Vec<_>>(),
        "kept_at_tol": {"anchor": share_le(&aa, tol), "decoy": share_le(&da, tol)},
    });
    (tol, block)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pk(mz: f32, i: f32) -> Peak {
        Peak { mz, intensity: i }
    }

    #[test]
    fn a_trace_over_steps_gets_its_centroid_and_a_lone_peak_its_own_step() {
        let step = 0.76;
        let target: Vec<f64> = (0..9).map(|s| 500.0 + s as f64 * step).collect();
        // One fragment m/z transmitted over steps 2..=6 peaking at 4, a lone random match in
        // step 1, and another m/z in step 7 only.
        let prof = [0.0, 0.0, 10.0, 50.0, 100.0, 50.0, 10.0, 0.0, 0.0];
        let mut steps: Vec<Vec<Peak>> = vec![Vec::new(); 9];
        for (s, &v) in prof.iter().enumerate() {
            if v > 0.0 {
                steps[s].push(pk(300.0 + 0.0001 * s as f32, v as f32));
            }
        }
        steps[1].push(pk(412.3, 5.0));
        steps[7].push(pk(650.0, 20.0));
        for s in &mut steps {
            s.sort_by(|a, b| a.mz.total_cmp(&b.mz));
        }
        let refs: Vec<&[Peak]> = steps.iter().map(|v| &v[..]).collect();
        let off = annotate_run(&refs, &target, step, 20.0);
        for s in 2..=6 {
            let j = steps[s].iter().position(|p| p.mz < 301.0).unwrap();
            let cen = target[s] + off[s][j] as f64 / UNITS_PER_STEP * step;
            assert!(
                (cen - target[4]).abs() < 1e-9 + step / UNITS_PER_STEP,
                "step {s}: {cen}"
            );
        }
        let j = steps[1].iter().position(|p| p.mz > 400.0).unwrap();
        assert_eq!(off[1][j], 0, "a lone peak is at its own step");
        assert_eq!(off[7][0], 0);
    }

    #[test]
    fn two_precursors_sharing_a_fragment_mz_split_at_the_valley() {
        let step = 0.76;
        let target: Vec<f64> = (0..12).map(|s| 500.0 + s as f64 * step).collect();
        // Two bumps of one m/z: centred at steps 3 and 8, with a dip between.
        let prof = [
            0.0, 10.0, 60.0, 100.0, 60.0, 10.0, 5.0, 60.0, 90.0, 60.0, 10.0, 0.0,
        ];
        let steps: Vec<Vec<Peak>> = prof
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    vec![pk(450.0, v as f32)]
                } else {
                    Vec::new()
                }
            })
            .collect();
        let refs: Vec<&[Peak]> = steps.iter().map(|v| &v[..]).collect();
        let off = annotate_run(&refs, &target, step, 20.0);
        let cen = |s: usize| target[s] + off[s][0] as f64 / UNITS_PER_STEP * step;
        assert!((cen(3) - target[3]).abs() <= 0.25 * step, "{}", cen(3));
        assert!((cen(8) - target[8]).abs() <= 0.25 * step, "{}", cen(8));
        assert!((cen(2) - cen(4)).abs() < 1e-9);
        assert!((cen(7) - cen(9)).abs() < 1e-9);
        // Deterministic.
        assert_eq!(annotate_run(&refs, &target, step, 20.0), off);
    }

    #[test]
    fn links_are_mutual_and_within_the_tolerance() {
        let a = [pk(100.0, 1.0), pk(200.0, 1.0), pk(200.003, 1.0)];
        let b = [pk(100.0015, 1.0), pk(200.0021, 1.0)];
        let l = mutual_links(&a, &b, 20.0);
        // 100.0 -> 100.0015 (15 ppm); 200.003 is nearer 200.0021 than 200.0 is, so only it
        // links.
        assert_eq!(l, vec![0, NONE, 1]);
        assert_eq!(mutual_links(&a, &b, 10.0), vec![NONE, NONE, 1]);
    }

    #[test]
    fn the_expected_centre_is_the_envelope_mean_less_the_offset() {
        let c = expected_centre(None, 600.0, 2, 100.0);
        let pre = averagine_isotopes((600.0 - PROTON) * 2.0, ISOTOPES);
        let ipre: f64 = pre.iter().enumerate().map(|(i, w)| i as f64 * w).sum();
        assert!((c - (600.0 + ISOTOPE_SPACING / 2.0 * ipre)).abs() < 1e-12);
        // Anchors near 0 and decoys spread: the tolerance separates them.
        let anchor: Vec<f64> = (0..200).map(|i| (i as f64 / 200.0 - 0.5) * 1.6).collect();
        let decoy: Vec<f64> = (0..200).map(|i| (i as f64 / 200.0 - 0.5) * 12.0).collect();
        let (tol, block) = learn_tolerance(&anchor, &decoy, 20, 20);
        assert!((0.75..=1.0).contains(&tol), "{tol}");
        assert_eq!(block["rule"], "anchor_minus_decoy");
        assert_eq!(block["anchor_peaks"], 200);
        let (tol0, b0) = learn_tolerance(&anchor, &[], 20, 0);
        assert_eq!(b0["rule"], "robust_sd");
        assert!(tol0 >= MIN_TOL_STEPS);
        // A fragment's centre lies at or below its envelope's.
        let m = mumdia_core::config::Q1FragModel {
            step_mz: 0.76,
            half_width_mz: 1.9,
            edge_width_mz: 1.5,
            offset_terms: Vec::new(),
            tolerances: Default::default(),
        };
        let cf = fragment_centre(&m, 1000.0, 2, 1500.0, 1, 0.0);
        assert!(cf < expected_centre(Some(&m), 1000.0, 2, 0.0));
        assert!(cf > 1000.0);
    }
}
