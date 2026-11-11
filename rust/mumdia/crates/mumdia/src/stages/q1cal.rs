//! Q1 calibration of scanning-quadrupole spectra (`rt_im_train.q1_calibrate`).
//!
//! In a SCIEX ZT Scan DIA run every MS2 spectrum is one step of a quadrupole sweeping the
//! precursor range once a cycle, and a precursor is transmitted through the few consecutive
//! steps whose window covers it. Where in the sweep a precursor shows up says where the
//! quadrupole's window really is relative to the target the file records for each step, how
//! wide it is and how soft its edges are. This module measures that on the confident seed
//! PSMs, before extract, and writes it to `cal.json` (`q1_calibration`). Nothing reads it
//! back yet: it is the calibration the per-fragment Q1 decisions of a later change are to
//! be built on.
//!
//! For each anchor (a target seed PSM with `spectrum_q` under the anchor cut) and each of
//! its strongest library fragments, the matched intensity (the nearest peak within the
//! fragment tolerance of the offset-corrected fragment m/z, the seed's own rule) is read in
//! every step whose target lies within two half-widths of the precursor m/z, in the cycle
//! of the seed scan and the cycles either side, and summed per step over those cycles: the
//! fragment's Q1 profile. Its intensity-weighted centroid over the step targets is the
//! fragment's observed Q1 centre.
//!
//! The expected profile is the quadrupole's transmission summed over the precursor's
//! isotopes, each weighted by its Poisson-averagine abundance
//! (`mumdia_core::constants::averagine_isotopes`) times the chance that the fragment stays
//! monoisotopic when that isotope is transmitted, `(1 - f)^i` with `f` the fragment's share
//! of the precursor mass: a heavy isotope in the fragment moves it off the m/z matched
//! here. The transmission of a step with target `t` is a window centred at `t + delta`,
//! whole within `h - e/2` of the centre and falling linearly to nothing at `h + e/2` (a
//! box of `2h` convolved with a box of `e`, which is the window swept over one step when
//! `e` is the step). The nominal model is `delta = 0`, `h` the half-width as read
//! (`convert.isolation_half_width`) and `e` one step.
//!
//! The residual of a profile is its observed minus its expected centroid, in steps. A
//! window centred `delta` above its target shows a precursor `delta` early in the sweep, so
//! the offset is fitted to `delta - residual`: an additive model of robust LOESS terms
//! (backfitting, two rounds of two passes), always of the precursor m/z, and of the
//! fragment m/z (what a fragment's transit from the quadrupole to the detector depends on)
//! and the retention time, each kept when it narrows the profiles' residual spread by at
//! least 2%. `h` and `e` are fitted by least squares of the normalised profiles against
//! the model with a per-profile scale and floor (the random matches of a 20 ppm tolerance
//! in every step), on a grid; the offset is fitted under the nominal shape, the shape under
//! that offset and the offset again under the fitted shape. The nominal shape also scores
//! the three readings of the target, the window's centre (`delta = 0`), its lower edge
//! (`delta = +h`) and its upper edge (`delta = -h`), and `target_marks` names the one of
//! least squares.
//!
//! Measured on a ZenoTOF 8600 K562 run (25,305 anchors, 151,789 profiles, docs/08, section
//! 4h): the target is the window's centre (sum of squares 162 against 745 and 765 for the
//! edges); the residual median moved from -0.108 to +0.003 steps and the MAD from 0.188 to
//! 0.158 steps; the offset falls from +0.24 m/z at 395 m/z to -0.06 m/z at 900 m/z; and the
//! window is wider and softer than 3.8 m/z: whole within 0.57 m/z of its centre, nothing
//! beyond 4.56 m/z (h 2.57, e 4.00 m/z).
//!
//! Under `extract.q1_fragments` the calibration also learns the per-fragment decision
//! tolerances (`stages::q1frag::learn`) from the same anchors' RT x step patches of every
//! library fragment, read with the window it just fitted, and adds them to the block as
//! `fragment_decision`; extract decides with them.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use mumdia_core::config::interp_held;
use mumdia_core::constants::ISOTOPE_SPACING;
use mumdia_io::table::TableFile;
use rayon::prelude::*;
use serde_json::json;
use tracing::{info, warn};

use crate::calibrate::{percentile, Loess};
use crate::stages::extract::{Q1_MAX_STEP, Q1_MIN_OVERLAP, Q1_MIN_WINDOWS};
use crate::stages::q1frag;
#[cfg(test)]
use crate::stages::q1frag::transmission;
use crate::stages::q1frag::{profile_sse, step_key};

/// Strongest library fragments read per anchor.
const FRAGMENTS: usize = 6;
/// Cycles read either side of the seed scan's cycle.
const CYCLES_EACH_SIDE: u32 = 2;
/// Steps read around a precursor: targets within this many half-widths of its m/z.
const RANGE_HALF_WIDTHS: f64 = 2.0;
/// Fewest profiles a calibration is fitted from.
const MIN_PROFILES: usize = 200;
/// Most anchors read: the best by seed score above it.
const MAX_ANCHORS: usize = 50_000;
/// Most profiles the width and edge fit reads (every k-th above it).
const FIT_PROFILES: usize = 20_000;
/// Points of each offset grid.
const GRID: usize = 20;
/// LOESS span and robustness iterations of the offset fit.
const SPAN: f64 = 0.3;
const ROBUST_ITERS: usize = 2;
/// A further offset term (fragment m/z, retention time) is kept when it reduces the
/// profiles' residual spread by this share.
const TERM_GAIN: f64 = 0.02;
/// Most profiles each LOESS of the offset reads (every k-th above it).
const LOESS_PROFILES: usize = 20_000;
/// The robust standard deviation of a normal sample per unit of its median absolute
/// deviation.
const MAD_TO_SIGMA: f64 = 1.4826;
/// Most anchors the fragment-decision tolerances are learned from (every k-th above it).
const LEARN_ANCHORS: usize = 20_000;

/// The inputs of [`run`].
pub struct Q1CalParams<'a> {
    pub seed_psms: &'a str,
    /// `spectra_ms2.parquet` of the run (v3 records the sweep; v1 is read off its targets).
    pub ms2: &'a str,
    pub library_fragments: &'a str,
    /// The seed's `masscal.json` (fragment offset and tolerance), when it exists.
    pub mass_cal: Option<&'a str>,
    /// The anchor cut on `spectrum_q` (`rt_im_train.q_train`, relaxed by the anchor plan).
    pub q_max: f64,
    /// The fragment tolerance without a mass calibration (`extract.frag_tol_ppm`).
    pub frag_tol_fallback_ppm: f64,
    /// The run's `cal.json`, which gains `q1_calibration`.
    pub out_cal: &'a str,
    /// Also learn the per-fragment decision tolerances (`extract.q1_fragments`) and add
    /// them as `fragment_decision`.
    pub fragment_decision: bool,
}

/// The Q1 sweep of a run's MS2 spectra, in file (acquisition) order.
struct Sweep {
    scan_index: Vec<u32>,
    target: Vec<f64>,
    /// First row of every cycle, and the row count last.
    cycle_start: Vec<usize>,
    row_cycle: Vec<u32>,
    step_mz: f64,
    half_width: f64,
    /// Whether the cycles came from spectra_ms2 v3 rather than from the targets.
    recorded: bool,
    /// The steps: the distinct targets (1e-4 m/z) ascending, and every row's step (the rank
    /// of its target, which spectra_ms2 v3 records on a regular sweep).
    step_target: Vec<f64>,
    row_step: Vec<u32>,
}

impl Sweep {
    fn n_cycles(&self) -> usize {
        self.cycle_start.len().saturating_sub(1)
    }

    /// The rows of cycle `c` whose target lies within `r` of `pm` (targets ascend within a
    /// cycle).
    fn rows_near(&self, c: usize, pm: f64, r: f64) -> std::ops::Range<usize> {
        let (a, b) = (self.cycle_start[c], self.cycle_start[c + 1]);
        let t = &self.target[a..b];
        let lo = t.partition_point(|&x| x < pm - r);
        let hi = t.partition_point(|&x| x <= pm + r);
        a + lo..a + hi
    }
}

/// The sweep of `ms2`: the cycles spectra_ms2 v3 recorded, or, on v1, the same rule convert
/// records them with (a cycle starts where the target does not rise). Refused when the
/// windows are not a scanning quadrupole's, or the spectra carry 1/K0.
fn read_sweep(ms2: &str) -> Result<Sweep> {
    let t = TableFile::open(ms2).with_context(|| format!("q1-calibrate: reading {ms2}"))?;
    if t.has_column("im") {
        anyhow::bail!(
            "rt_im_train.q1_calibrate calibrates a scanning quadrupole, and {ms2} carries 1/K0 \
             (diaPASEF); unset it for these spectra"
        );
    }
    let scan_index = t.u32("scan_index")?;
    let target = t.f64("window_target")?;
    let lo = t.f64("window_lower")?;
    let hi = t.f64("window_upper")?;
    let recorded = t.has_column("q1_cycle");
    let row_cycle: Vec<u32> = if recorded {
        t.u32("q1_cycle")?
    } else {
        let mut c = 0u32;
        let mut out = Vec::with_capacity(target.len());
        for (i, &x) in target.iter().enumerate() {
            if i > 0 && x <= target[i - 1] {
                c += 1;
            }
            out.push(c);
        }
        out
    };
    let mut cycle_start = Vec::new();
    for (i, &c) in row_cycle.iter().enumerate() {
        if i == 0 || c != row_cycle[i - 1] {
            if i > 0 && c < row_cycle[i - 1] {
                anyhow::bail!("q1-calibrate: {ms2}: q1_cycle falls back at row {i}");
            }
            cycle_start.push(i);
        } else if target[i] <= target[i - 1] {
            anyhow::bail!(
                "q1-calibrate: {ms2}: the targets of cycle {c} do not rise at row {i}, so the \
                 rows are not one upward sweep"
            );
        }
    }
    cycle_start.push(row_cycle.len());
    let mut d: Vec<f64> = target.iter().map(|x| (x * 1e4).round() / 1e4).collect();
    d.sort_by(f64::total_cmp);
    d.dedup();
    let steps: Vec<f64> = d.windows(2).map(|w| w[1] - w[0]).collect();
    let step_mz = if steps.is_empty() {
        0.0
    } else {
        percentile(&steps, 0.5)
    };
    let widths: Vec<f64> = lo.iter().zip(&hi).map(|(a, b)| b - a).collect();
    let width = percentile(&widths, 0.5);
    if d.len() < Q1_MIN_WINDOWS
        || !(step_mz > 0.0 && step_mz < Q1_MAX_STEP)
        || width < Q1_MIN_OVERLAP * step_mz
    {
        anyhow::bail!(
            "rt_im_train.q1_calibrate calibrates a scanning quadrupole (SCIEX ZT Scan DIA read \
             with convert.isolation_half_width): at least {Q1_MIN_WINDOWS} overlapping \
             isolation windows whose targets step less than {Q1_MAX_STEP} m/z. {ms2} has {} \
             distinct targets, a median step of {step_mz:.4} m/z and a median width of \
             {width:.4} m/z; unset it for these spectra",
            d.len()
        );
    }
    let keys: Vec<i64> = d.iter().map(|&x| step_key(x)).collect();
    let row_step: Vec<u32> = target
        .iter()
        .map(|&x| keys.partition_point(|&k| k < step_key(x)) as u32)
        .collect();
    Ok(Sweep {
        scan_index,
        target,
        cycle_start,
        row_cycle,
        step_mz,
        half_width: 0.5 * width,
        recorded,
        step_target: d,
        row_step,
    })
}

/// One anchor: its candidate, precursor m/z, charge, retention time and seed scan's cycle.
#[derive(Clone, Copy, Debug)]
struct Anchor {
    cid: u32,
    pm: f64,
    z: i32,
    rt: f64,
    c0: u32,
}

/// The confident target seed PSMs whose seed scan is in the sweep, one per candidate, the
/// best `MAX_ANCHORS` by score, ordered by candidate.
fn read_anchors(seed: &str, sweep: &Sweep, q_max: f64) -> Result<Vec<Anchor>> {
    let t = TableFile::open(seed)?;
    let label = t.str("label")?;
    let q = t.f64("spectrum_q")?;
    let cid = t.u32("candidate_id")?;
    let pm = t.f64("precursor_mz")?;
    let z = t.i32("charge")?;
    let rt = t.f64("observed_rt")?;
    let scan = t.u32("scan_index")?;
    let score = t.f64("score")?;
    let sorted = sweep.scan_index.windows(2).all(|w| w[0] < w[1]);
    let row_of = |s: u32| -> Option<usize> {
        if sorted {
            sweep.scan_index.binary_search(&s).ok()
        } else {
            sweep.scan_index.iter().position(|&x| x == s)
        }
    };
    let mut best: BTreeMap<u32, (f64, Anchor)> = BTreeMap::new();
    for i in 0..t.nrows {
        if label[i] != "target" || q[i].is_nan() || q[i] >= q_max || !pm[i].is_finite() || z[i] <= 0
        {
            continue;
        }
        let Some(row) = row_of(scan[i]) else {
            continue;
        };
        let a = Anchor {
            cid: cid[i],
            pm: pm[i],
            z: z[i],
            rt: rt[i],
            c0: sweep.row_cycle[row],
        };
        match best.get(&cid[i]) {
            Some((s, _)) if *s >= score[i] => {}
            _ => {
                best.insert(cid[i], (score[i], a));
            }
        }
    }
    let mut v: Vec<(f64, Anchor)> = best.into_values().collect();
    if v.len() > MAX_ANCHORS {
        v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cid.cmp(&b.1.cid)));
        v.truncate(MAX_ANCHORS);
        v.sort_by_key(|a| a.1.cid);
    }
    Ok(v.into_iter().map(|(_, a)| a).collect())
}

/// A library fragment row of an anchor: the anchor's slot, m/z, predicted intensity and
/// charge.
type FragRow = (u32, f64, f64, i32);

/// A fragment match: anchor, fragment slot, spectrum row and intensity.
type Match = (u32, u8, u32, f64);

/// One library fragment of an anchor: its m/z and charge.
#[derive(Clone, Copy, Debug)]
struct Frag {
    mz: f64,
    z: i32,
}

/// The `FRAGMENTS` strongest library fragments of every anchor (by predicted intensity, the
/// earlier row on a tie), read one row group at a time.
fn read_fragments(path: &str, anchors: &[Anchor]) -> Result<Vec<Vec<Frag>>> {
    let max_cid = anchors.iter().map(|a| a.cid).max().unwrap_or(0) as usize;
    let mut slot = vec![u32::MAX; max_cid + 1];
    for (i, a) in anchors.iter().enumerate() {
        slot[a.cid as usize] = i as u32;
    }
    let t = TableFile::open(path).with_context(|| format!("q1-calibrate: reading {path}"))?;
    let has_z = t.has_column("frag_charge");
    let parts = t.row_group_parts()?;
    let found: Vec<Result<Vec<FragRow>>> = parts
        .par_iter()
        .map(|p| {
            let cid = p.u32("candidate_id")?;
            let mz = p.f64_widening("mz")?;
            let pi = p.f64_widening("predicted_intensity")?;
            let fz = if has_z {
                p.i32("frag_charge")?
            } else {
                vec![1; cid.len()]
            };
            Ok((0..cid.len())
                .filter_map(|i| {
                    let s = *slot.get(cid[i] as usize)?;
                    (s != u32::MAX).then_some((s, mz[i], pi[i], fz[i].max(1)))
                })
                .collect())
        })
        .collect();
    let mut per: Vec<Vec<(f64, Frag)>> = vec![Vec::new(); anchors.len()];
    for f in found {
        for (s, mz, pi, z) in f? {
            per[s as usize].push((pi, Frag { mz, z }));
        }
    }
    Ok(per
        .into_iter()
        .map(|mut v| {
            v.sort_by(|a, b| b.0.total_cmp(&a.0));
            v.truncate(FRAGMENTS);
            v.into_iter().map(|(_, f)| f).collect()
        })
        .collect())
}

/// The peak of `mz` (ascending) nearest `target` within `tol_ppm`, the lower on a tie
/// (`spectra::nearest_peak` on the artifact's own lists).
fn nearest(mz: &[f32], target: f64, tol_ppm: f64) -> Option<usize> {
    let (lo, hi) = mumdia_core::constants::ppm_bounds(target, tol_ppm);
    let mut j = mz.partition_point(|&m| (m as f64) < lo);
    let (mut best_d, mut best) = (f64::MAX, None);
    while j < mz.len() && (mz[j] as f64) <= hi {
        let d = (mz[j] as f64 - target).abs();
        if d < best_d {
            best_d = d;
            best = Some(j);
        }
        j += 1;
    }
    best
}

/// One fragment's Q1 profile: the anchor, the fragment's m/z and isotope weights and the
/// summed matched intensity at each of the anchor's step targets.
struct Profile {
    anchor: u32,
    fmz: f64,
    weights: Vec<f64>,
    y: Vec<f64>,
}

/// Read every anchor's fragment profiles from the MS2 spectra: per anchor, the steps
/// within `RANGE_HALF_WIDTHS` half-widths of its m/z in the cycles around its seed scan.
/// Returns the anchors' step targets and the profiles with signal in at least two steps.
fn read_profiles(
    ms2: &str,
    sweep: &Sweep,
    anchors: &[Anchor],
    frags: &[Vec<Frag>],
    factor: &dyn Fn(f64) -> f64,
    tol_ppm: f64,
) -> Result<(Vec<Vec<f64>>, Vec<Profile>)> {
    let r = RANGE_HALF_WIDTHS * sweep.half_width;
    let n_cycles = sweep.n_cycles();
    // The rows each anchor reads, and the anchor's step targets (keyed at 1e-4 m/z).
    let mut requests: Vec<(u32, u32)> = Vec::new(); // (row, anchor)
    let mut step_keys: Vec<Vec<(i64, f64)>> = vec![Vec::new(); anchors.len()];
    for (ai, a) in anchors.iter().enumerate() {
        if frags[ai].is_empty() {
            continue;
        }
        let lo = a.c0.saturating_sub(CYCLES_EACH_SIDE) as usize;
        let hi = (a.c0 + CYCLES_EACH_SIDE) as usize;
        let mut keys: BTreeMap<i64, f64> = BTreeMap::new();
        for c in lo..=hi.min(n_cycles.saturating_sub(1)) {
            for row in sweep.rows_near(c, a.pm, r) {
                requests.push((row as u32, ai as u32));
                let t = sweep.target[row];
                keys.entry((t * 1e4).round() as i64).or_insert(t);
            }
        }
        step_keys[ai] = keys.into_iter().collect();
    }
    requests.sort_unstable();
    // Fragment targets, offset-corrected once.
    let targets: Vec<Vec<f64>> = frags
        .iter()
        .map(|fs| fs.iter().map(|f| f.mz * factor(f.mz)).collect())
        .collect();
    let t = TableFile::open(ms2)?;
    let parts = t.row_group_parts()?;
    let mut first = Vec::with_capacity(parts.len());
    let mut acc = 0usize;
    for p in &parts {
        first.push(acc);
        acc += p.nrows;
    }
    // (anchor, fragment, row, intensity) of every match, per part in row order.
    let hits: Vec<Result<Vec<Match>>> = parts
        .par_iter()
        .zip(first.par_iter())
        .map(|(p, &f0)| {
            let a = requests.partition_point(|q| (q.0 as usize) < f0);
            let b = requests.partition_point(|q| (q.0 as usize) < f0 + p.nrows);
            let mut out = Vec::new();
            if a == b {
                return Ok(out);
            }
            let mut k = a;
            let mut row = f0;
            p.for_each_batch(Some(&["mz", "intensity"]), 1024, |batch| {
                let mza = crate::spectra::list_col(batch, "mz")?;
                let ina = crate::spectra::list_col(batch, "intensity")?;
                for i in 0..mza.len() {
                    while k < b && (requests[k].0 as usize) == row {
                        let ai = requests[k].1;
                        let (mz, inten) = (mza.row_slice(i, "mz")?, ina.row_slice(i, "intensity")?);
                        let n = mz.len().min(inten.len());
                        for (fi, &tg) in targets[ai as usize].iter().enumerate() {
                            if let Some(j) = nearest(&mz[..n], tg, tol_ppm) {
                                out.push((ai, fi as u8, row as u32, inten[j] as f64));
                            }
                        }
                        k += 1;
                    }
                    row += 1;
                }
                Ok(())
            })?;
            Ok(out)
        })
        .collect();
    let mut all: Vec<Match> = Vec::new();
    for h in hits {
        all.extend(h?);
    }
    all.sort_by_key(|x| (x.0, x.1, x.2));
    let step_t: Vec<Vec<f64>> = step_keys
        .iter()
        .map(|v| v.iter().map(|&(_, t)| t).collect())
        .collect();
    let mut profiles = Vec::new();
    let mut i = 0;
    while i < all.len() {
        let (ai, fi) = (all[i].0, all[i].1);
        let keys = &step_keys[ai as usize];
        let mut y = vec![0.0f64; keys.len()];
        while i < all.len() && all[i].0 == ai && all[i].1 == fi {
            let key = (sweep.target[all[i].2 as usize] * 1e4).round() as i64;
            if let Ok(s) = keys.binary_search_by_key(&key, |&(k, _)| k) {
                y[s] += all[i].3;
            }
            i += 1;
        }
        if y.iter().filter(|&&v| v > 0.0).count() < 2 {
            continue;
        }
        let a = &anchors[ai as usize];
        let f = frags[ai as usize][fi as usize];
        profiles.push(Profile {
            anchor: ai,
            fmz: f.mz,
            weights: isotope_weights(a, f),
            y,
        });
    }
    Ok((step_t, profiles))
}

/// The precursor isotopes' weights for a fragment ([`q1frag::isotope_weights`]).
fn isotope_weights(a: &Anchor, f: Frag) -> Vec<f64> {
    q1frag::isotope_weights(a.pm, a.z, f.mz, f.z)
}

/// The expected profile over the step targets `t` of a precursor of m/z `pm` and charge
/// `z` whose isotopes carry `w`, windows centred `delta` above their targets.
fn expected(t: &[f64], pm: f64, z: i32, w: &[f64], delta: f64, h: f64, e: f64) -> Vec<f64> {
    let sp = ISOTOPE_SPACING / z as f64;
    t.iter()
        .map(|&tk| {
            w.iter()
                .enumerate()
                .map(|(i, wi)| wi * q1frag::transmission(pm + i as f64 * sp - (tk + delta), h, e))
                .sum()
        })
        .collect()
}

/// The intensity-weighted centroid of `y` over `t`.
fn centroid(y: &[f64], t: &[f64]) -> Option<f64> {
    let s: f64 = y.iter().sum();
    (s > 0.0).then(|| y.iter().zip(t).map(|(a, b)| a * b).sum::<f64>() / s)
}

/// A variable the window-centre offset can depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Term {
    /// The precursor m/z (always in the model).
    PrecursorMz,
    /// The fragment m/z: what the fragment's transit from the quadrupole to the detector
    /// depends on.
    FragmentMz,
    /// The retention time.
    Rt,
}

impl Term {
    fn name(self) -> &'static str {
        match self {
            Term::PrecursorMz => "precursor_mz",
            Term::FragmentMz => "fragment_mz",
            Term::Rt => "rt",
        }
    }
}

/// The window-centre offset model (m/z): a sum of smooth terms, each a grid over its
/// variable, linearly interpolated and held at its ends. The first term carries the level;
/// the others are centred on the profiles.
#[derive(Clone, Debug, Default)]
struct Offset {
    terms: Vec<(Term, Vec<f64>, Vec<f64>)>,
}

impl Offset {
    fn at(&self, a: &Anchor, p: &Profile) -> f64 {
        self.terms
            .iter()
            .map(|(t, g, v)| interp_held(g, v, var(*t, a, p)))
            .sum()
    }
}

/// The value of `t` for profile `p` of anchor `a`.
fn var(t: Term, a: &Anchor, p: &Profile) -> f64 {
    match t {
        Term::PrecursorMz => a.pm,
        Term::FragmentMz => p.fmz,
        Term::Rt => a.rt,
    }
}

/// `f` sampled on `GRID` points over `[lo, hi]`.
fn sample(f: &Loess, lo: f64, hi: f64) -> (Vec<f64>, Vec<f64>) {
    let g: Vec<f64> = (0..GRID)
        .map(|k| lo + (hi - lo) * k as f64 / (GRID - 1) as f64)
        .collect();
    let v = g.iter().map(|&x| f.predict(x)).collect();
    (g, v)
}

/// Median, median absolute deviation, robust standard deviation (1.4826 MAD), quartiles,
/// 5th and 95th percentile and count of `v`.
fn dist(v: &[f64]) -> serde_json::Value {
    if v.is_empty() {
        return json!({"n": 0});
    }
    let med = percentile(v, 0.5);
    let dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
    json!({
        "n": v.len(),
        "median": med,
        "mad": percentile(&dev, 0.5),
        "sd": MAD_TO_SIGMA * percentile(&dev, 0.5),
        "p05": percentile(v, 0.05),
        "p25": percentile(v, 0.25),
        "p75": percentile(v, 0.75),
        "p95": percentile(v, 0.95),
    })
}

/// The robust standard deviation of `v`.
fn robust_sd(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let med = percentile(v, 0.5);
    let dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
    MAD_TO_SIGMA * percentile(&dev, 0.5)
}

/// The window model a residual is computed under: an offset (m/z) per profile, the
/// half-width and the edge width.
type OffsetFn<'a> = &'a (dyn Fn(&Anchor, &Profile) -> f64 + Sync);

/// Everything the fit reads: the anchors, their step targets and the profiles.
struct Data<'a> {
    anchors: &'a [Anchor],
    step_t: &'a [Vec<f64>],
    profiles: &'a [Profile],
}

impl Data<'_> {
    /// Each profile's residual (observed minus expected centroid, m/z) under the model.
    fn residuals(&self, off: OffsetFn, h: f64, e: f64) -> Vec<Option<f64>> {
        self.profiles
            .par_iter()
            .map(|p| {
                let a = &self.anchors[p.anchor as usize];
                let t = &self.step_t[p.anchor as usize];
                let ex = expected(t, a.pm, a.z, &p.weights, off(a, p), h, e);
                Some(centroid(&p.y, t)? - centroid(&ex, t)?)
            })
            .collect()
    }

    /// The summed profile SSE of the model over every `k`-th profile.
    fn sse(&self, off: OffsetFn, h: f64, e: f64, k: usize) -> f64 {
        let parts: Vec<f64> = self
            .profiles
            .par_iter()
            .step_by(k.max(1))
            .map(|p| {
                let a = &self.anchors[p.anchor as usize];
                let t = &self.step_t[p.anchor as usize];
                profile_sse(&p.y, &expected(t, a.pm, a.z, &p.weights, off(a, p), h, e))
            })
            .collect();
        parts.iter().sum()
    }

    /// The robust standard deviation of the residuals (m/z) under the model.
    fn spread(&self, off: &Offset, h: f64, e: f64) -> f64 {
        let r: Vec<f64> = self
            .residuals(&|a, p| off.at(a, p), h, e)
            .into_iter()
            .flatten()
            .collect();
        robust_sd(&r)
    }

    /// The offset with the terms `terms`, fitted by backfitting robust LOESS terms to what
    /// the windows must move by (`delta - residual`, since a window centred `delta` above
    /// its target shows a precursor `delta` early), from `start`, over two rounds of two
    /// passes. Each LOESS reads every `k`-th profile.
    fn fit_offset(&self, terms: &[Term], start: &Offset, h: f64, e: f64) -> Offset {
        let k = self.profiles.len().div_ceil(LOESS_PROFILES).max(1);
        let mut off = Offset {
            terms: terms
                .iter()
                .map(|&t| {
                    start.terms.iter().find(|x| x.0 == t).cloned().unwrap_or((
                        t,
                        Vec::new(),
                        Vec::new(),
                    ))
                })
                .collect(),
        };
        for _ in 0..2 {
            let r = self.residuals(&|a, p| off.at(a, p), h, e);
            // The subsample, in profile order: its variables and the target offset.
            let rows: Vec<(usize, f64)> = r
                .iter()
                .enumerate()
                .step_by(k)
                .filter_map(|(i, x)| x.map(|v| (i, v)))
                .collect();
            let at = |i: usize| -> (&Anchor, &Profile) {
                let p = &self.profiles[i];
                (&self.anchors[p.anchor as usize], p)
            };
            let target: Vec<f64> = rows
                .iter()
                .map(|&(i, v)| {
                    let (a, p) = at(i);
                    off.at(a, p) - v
                })
                .collect();
            for _ in 0..2 {
                for ti in 0..off.terms.len() {
                    let t = off.terms[ti].0;
                    let xs: Vec<f64> = rows
                        .iter()
                        .map(|&(i, _)| {
                            let (a, p) = at(i);
                            var(t, a, p)
                        })
                        .collect();
                    let ys: Vec<f64> = rows
                        .iter()
                        .zip(&target)
                        .map(|(&(i, _), &y)| {
                            let (a, p) = at(i);
                            let others: f64 = off
                                .terms
                                .iter()
                                .enumerate()
                                .filter(|(j, _)| *j != ti)
                                .map(|(_, (tt, g, v))| interp_held(g, v, var(*tt, a, p)))
                                .sum();
                            y - others
                        })
                        .collect();
                    let (g, mut v) = smooth(&xs, &ys);
                    if ti > 0 {
                        let mean = xs.iter().map(|&x| interp_held(&g, &v, x)).sum::<f64>()
                            / xs.len().max(1) as f64;
                        for x in v.iter_mut() {
                            *x -= mean;
                        }
                    }
                    off.terms[ti] = (t, g, v);
                }
            }
        }
        off
    }

    /// The precursor-m/z offset, then each further term kept when it narrows the residuals
    /// by at least `TERM_GAIN`: fragment m/z, then retention time.
    fn select_offset(&self, start: &Offset, h: f64, e: f64) -> (Offset, Vec<(Term, f64)>) {
        let mut terms = vec![Term::PrecursorMz];
        let mut off = self.fit_offset(&terms, start, h, e);
        let mut sd = self.spread(&off, h, e);
        let mut trail = vec![(Term::PrecursorMz, sd)];
        for t in [Term::FragmentMz, Term::Rt] {
            let mut with = terms.clone();
            with.push(t);
            let cand = self.fit_offset(&with, &off, h, e);
            let sd_c = self.spread(&cand, h, e);
            trail.push((t, sd_c));
            if sd_c < (1.0 - TERM_GAIN) * sd {
                terms = with;
                off = cand;
                sd = sd_c;
            }
        }
        (off, trail)
    }

    /// The half-width and edge width of least SSE under `off`: a grid of half-widths from
    /// half to twice the one as read and edges from 0 to six steps, refined around the best.
    fn fit_shape(&self, off: &Offset, hw: f64, step: f64) -> (f64, f64, f64) {
        let k = self.profiles.len().div_ceil(FIT_PROFILES);
        let o = |a: &Anchor, p: &Profile| off.at(a, p);
        let mut best = (f64::INFINITY, hw, step);
        for i in 0..=30 {
            let h = hw * (0.5 + 0.05 * i as f64);
            for j in 0..=24 {
                let e = step * 0.25 * j as f64;
                if e > 2.0 * h {
                    continue;
                }
                let v = self.sse(&o, h, e, k);
                if v < best.0 {
                    best = (v, h, e);
                }
            }
        }
        let (_, h0, e0) = best;
        for i in -5..=5 {
            for j in -5..=5 {
                let h = h0 + 0.01 * hw * i as f64;
                let e = (e0 + 0.05 * step * j as f64).max(0.0);
                if h <= 0.0 || e > 2.0 * h {
                    continue;
                }
                let v = self.sse(&o, h, e, k);
                if v < best.0 {
                    best = (v, h, e);
                }
            }
        }
        best
    }
}

/// A robust LOESS of `ys` on `xs`, sampled on a grid over their range.
fn smooth(xs: &[f64], ys: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let f = Loess::fit_robust(xs, ys, SPAN, 200, ROBUST_ITERS);
    let (lo, hi) = xs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    sample(&f, lo, hi)
}

/// The median residual (steps) of the profiles in each octile of `key`.
fn by_octile(
    d: &Data,
    r: &[Option<f64>],
    step: f64,
    key: impl Fn(&Anchor, &Profile) -> f64,
) -> serde_json::Value {
    let mut v: Vec<(f64, f64)> = d
        .profiles
        .iter()
        .zip(r)
        .filter_map(|(p, x)| x.map(|x| (key(&d.anchors[p.anchor as usize], p), x / step)))
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = v.len();
    (0..8)
        .map(|b| {
            let s = &v[b * n / 8..(b + 1) * n / 8];
            let r: Vec<f64> = s.iter().map(|x| x.1).collect();
            json!({
                "lo": s.first().map(|x| x.0),
                "hi": s.last().map(|x| x.0),
                "median_residual_steps": percentile(&r, 0.5),
            })
        })
        .collect::<Vec<_>>()
        .into()
}

/// Fit the calibration on the profiles; the `q1_calibration` block of `cal.json`.
///
/// The offset is fitted under the nominal shape, the shape under that offset, and the
/// offset again under the fitted shape. The three readings of the target are scored under
/// the nominal shape.
fn fit(sweep: &Sweep, d: &Data) -> serde_json::Value {
    let step = sweep.step_mz;
    let hw = sweep.half_width;
    let steps = |v: &[Option<f64>]| -> Vec<f64> { v.iter().flatten().map(|x| x / step).collect() };
    let zero = |_: &Anchor, _: &Profile| 0.0;
    let r_nom = d.residuals(&zero, hw, step);
    let (off1, _) = d.select_offset(&Offset::default(), hw, step);
    let (_, h, e) = d.fit_shape(&off1, hw, step);
    let (off, trail) = d.select_offset(&off1, h, e);
    let (sse_fit, h, e) = d.fit_shape(&off, hw, step);
    let offf = |a: &Anchor, p: &Profile| off.at(a, p);
    let r_cal = d.residuals(&offf, h, e);
    let k = d.profiles.len().div_ceil(FIT_PROFILES);
    let hyp = |delta: f64| {
        let r = d.residuals(&|_: &Anchor, _: &Profile| delta, hw, step);
        json!({
            "sse": d.sse(&|_: &Anchor, _: &Profile| delta, hw, step, k),
            "median_residual_steps": percentile(&steps(&r), 0.5),
        })
    };
    let hyps = [
        ("centre", hyp(0.0)),
        ("lower_edge", hyp(hw)),
        ("upper_edge", hyp(-hw)),
    ];
    let sse_of = |v: &serde_json::Value| v["sse"].as_f64().unwrap_or(f64::INFINITY);
    let marks = hyps
        .iter()
        .min_by(|a, b| sse_of(&a.1).total_cmp(&sse_of(&b.1)))
        .map(|x| x.0)
        .unwrap_or("centre");
    let per_profile: Vec<f64> = d
        .profiles
        .iter()
        .map(|p| off.at(&d.anchors[p.anchor as usize], p))
        .collect();
    let med_off = percentile(&per_profile, 0.5);
    let (r_nom_s, r_cal_s) = (steps(&r_nom), steps(&r_cal));
    let used: Vec<&str> = off.terms.iter().map(|t| t.0.name()).collect();
    info!(
        profiles = d.profiles.len(),
        anchors = d.anchors.len(),
        target_marks = marks,
        offset_median_mz = med_off,
        offset_median_steps = med_off / step,
        offset_terms = %used.join("+"),
        half_width_mz = h,
        edge_width_mz = e,
        nominal_median_steps = percentile(&r_nom_s, 0.5),
        nominal_sd_steps = robust_sd(&r_nom_s),
        calibrated_median_steps = percentile(&r_cal_s, 0.5),
        calibrated_sd_steps = robust_sd(&r_cal_s),
        "q1-calibrate: Q1 centres of the seed fragments against the isotope-envelope model"
    );
    let terms = serde_json::Map::from_iter(
        off.terms
            .iter()
            .map(|(t, g, v)| (t.name().to_string(), json!({"grid": g, "offset_mz": v}))),
    );
    let trail_json: Vec<serde_json::Value> = trail
        .iter()
        .map(|(t, sd)| json!({"term": t.name(), "residual_sd_steps": sd / step}))
        .collect();
    json!({
        "version": 1,
        "status": "fitted",
        "anchors": d.anchors.len(),
        "profiles": d.profiles.len(),
        "fragments_per_anchor": FRAGMENTS,
        "cycles_read": 2 * CYCLES_EACH_SIDE + 1,
        "range_half_widths": RANGE_HALF_WIDTHS,
        "sweep": {
            "recorded": sweep.recorded,
            "cycles": sweep.n_cycles(),
            "step_mz": step,
            "half_width_as_read_mz": hw,
        },
        "isotope_model": "poisson_averagine_fragment_conditional",
        "target_marks": marks,
        "hypotheses": serde_json::Map::from_iter(
            hyps.iter().map(|(n, v)| (n.to_string(), v.clone()))
        ),
        "offset": {
            "median_mz": med_off,
            "median_steps": med_off / step,
            "terms": terms,
            "selection": trail_json,
        },
        "half_width_mz": h,
        "edge_width_mz": e,
        "plateau_half_mz": h - 0.5 * e,
        "support_half_mz": h + 0.5 * e,
        "sse": {"fitted": sse_fit, "profiles_per_point": k},
        "residual_steps": {
            "nominal": dist(&r_nom_s),
            "calibrated": dist(&r_cal_s),
            "nominal_by_precursor_mz": by_octile(d, &r_nom, step, |a, _| a.pm),
            "nominal_by_fragment_mz": by_octile(d, &r_nom, step, |_, p| p.fmz),
            "nominal_by_rt": by_octile(d, &r_nom, step, |a, _| a.rt),
            "calibrated_by_fragment_mz": by_octile(d, &r_cal, step, |_, p| p.fmz),
        },
    })
}

/// One library fragment row of an anchor for the decision: its m/z, predicted intensity and
/// charge (from the name, as extract reads it).
type DecisionFrag = (f64, f64, i32);

/// Every library fragment of the anchors, in the file's order (the library's order, which
/// extract's decision lists keep); the charge from the fragment name (`y7^2`), 1 without
/// one.
fn read_all_fragments(path: &str, anchors: &[Anchor]) -> Result<Vec<Vec<DecisionFrag>>> {
    let max_cid = anchors.iter().map(|a| a.cid).max().unwrap_or(0) as usize;
    let mut slot = vec![u32::MAX; max_cid + 1];
    for (i, a) in anchors.iter().enumerate() {
        slot[a.cid as usize] = i as u32;
    }
    let t = TableFile::open(path).with_context(|| format!("q1-calibrate: reading {path}"))?;
    let has_name = t.has_column("name");
    let parts = t.row_group_parts()?;
    let found: Vec<Result<Vec<(u32, DecisionFrag)>>> = parts
        .par_iter()
        .map(|p| {
            let cid = p.u32("candidate_id")?;
            let mz = p.f64_widening("mz")?;
            let pi = p.f64_widening("predicted_intensity")?;
            let names = if has_name {
                p.str("name")?
            } else {
                vec![String::new(); cid.len()]
            };
            Ok((0..cid.len())
                .filter_map(|i| {
                    let s = *slot.get(cid[i] as usize)?;
                    (s != u32::MAX).then(|| (s, (mz[i], pi[i], q1frag::frag_charge_of(&names[i]))))
                })
                .collect())
        })
        .collect();
    let mut per: Vec<Vec<DecisionFrag>> = vec![Vec::new(); anchors.len()];
    for f in found {
        for (s, fr) in f? {
            per[s as usize].push(fr);
        }
    }
    Ok(per)
}

/// The `fragment_decision` block: the anchors' RT x step patches of every library fragment,
/// read with the window `block` describes, and the tolerances [`q1frag::learn`] finds on
/// them. At most [`LEARN_ANCHORS`] anchors (every k-th).
#[allow(clippy::too_many_arguments)]
fn learn_decision(
    ms2: &str,
    sweep: &Sweep,
    anchors: &[Anchor],
    library_fragments: &str,
    factor: &(dyn Fn(f64) -> f64 + Sync),
    tol_ppm: f64,
    block: &serde_json::Value,
) -> Result<serde_json::Value> {
    let model = q1frag::model_from_calibration(block, false)?;
    let k = anchors.len().div_ceil(LEARN_ANCHORS).max(1);
    let anchors: Vec<Anchor> = anchors.iter().step_by(k).copied().collect();
    let frags = read_all_fragments(library_fragments, &anchors)?;
    let n_cycles = sweep.n_cycles();
    let n_steps = sweep.step_target.len();
    // cycle x step -> row.
    let mut cell = vec![u32::MAX; n_cycles * n_steps];
    for (row, (&c, &st)) in sweep.row_cycle.iter().zip(&sweep.row_step).enumerate() {
        cell[c as usize * n_steps + st as usize] = row as u32;
    }
    let reach = q1frag::reach_mz(&model) + q1frag::MARGIN_STEPS * model.step_mz;
    // Each anchor's candidate, cycles and steps.
    struct Spec {
        cand: q1frag::CandQ1,
        cycles: Vec<u32>,
        s0: u32,
        ns: usize,
        start_ci: usize,
    }
    let specs: Vec<Spec> = anchors
        .iter()
        .zip(&frags)
        .map(|(a, fs)| {
            let mzs: Vec<f64> = fs.iter().map(|f| f.0).collect();
            let preds: Vec<f64> = fs.iter().map(|f| f.1).collect();
            let zs: Vec<i32> = fs.iter().map(|f| f.2).collect();
            let cand = q1frag::CandQ1::new(&model, a.pm, a.z, a.rt, &mzs, &zs, &preds, factor);
            let lo = a.c0.saturating_sub(q1frag::PATCH_CYCLES);
            let hi = (a.c0 + q1frag::PATCH_CYCLES).min(n_cycles.saturating_sub(1) as u32);
            let cycles: Vec<u32> = (lo..=hi).collect();
            let start_ci = (a.c0 - lo) as usize;
            let s0 = sweep
                .step_target
                .partition_point(|&t| t < cand.centre - reach) as u32;
            let s1 = sweep
                .step_target
                .partition_point(|&t| t <= cand.centre + reach);
            Spec {
                cand,
                cycles,
                s0,
                ns: s1.saturating_sub(s0 as usize),
                start_ci,
            }
        })
        .collect();
    // The rows each anchor reads.
    let mut requests: Vec<(u32, u32)> = Vec::new();
    for (ai, sp) in specs.iter().enumerate() {
        if sp.cand.frags.is_empty() {
            continue;
        }
        for &c in &sp.cycles {
            for si in 0..sp.ns {
                let row = cell[c as usize * n_steps + sp.s0 as usize + si];
                if row != u32::MAX {
                    requests.push((row, ai as u32));
                }
            }
        }
    }
    requests.sort_unstable();
    let targets: Vec<Vec<f64>> = specs
        .iter()
        .map(|sp| sp.cand.frags.iter().map(|f| f.target_mz).collect())
        .collect();
    let matches = gather(ms2, &requests, &targets, tol_ppm)?;
    let mut patches: Vec<q1frag::Patch> = specs
        .iter()
        .map(|sp| {
            let t = sweep.step_target[sp.s0 as usize..sp.s0 as usize + sp.ns].to_vec();
            q1frag::Patch::new(sp.cycles.clone(), sp.s0, t, sp.cand.frags.len())
        })
        .collect();
    for (ai, f, row, v) in matches {
        let sp = &specs[ai as usize];
        let (c, st) = (sweep.row_cycle[row as usize], sweep.row_step[row as usize]);
        let (ci, si) = ((c - sp.cycles[0]) as usize, (st - sp.s0) as usize);
        patches[ai as usize].set(f as usize, ci, si, v as f32);
    }
    let learn: Vec<q1frag::LearnAnchor> = specs
        .into_iter()
        .zip(patches)
        .filter(|(sp, _)| !sp.cand.frags.is_empty() && sp.ns > 0)
        .map(|(sp, patch)| q1frag::LearnAnchor {
            cand: sp.cand,
            patch,
            start_ci: sp.start_ci,
        })
        .collect();
    let (tol, out) = q1frag::learn(&learn, &model);
    info!(
        anchors = learn.len(),
        offset_tol_steps = tol.offset_steps,
        centre_tol_steps = tol.centre_steps,
        corr2d_min = tol.corr2d_min,
        belong_share = %out["under_tolerances"]["belong_share"],
        interfered_share = %out["under_tolerances"]["interfered_share"],
        "q1-calibrate: per-fragment decision tolerances learned from the seed"
    );
    Ok(out)
}

/// A fragment match of the decision's patches: anchor, fragment, spectrum row, intensity.
type PatchMatch = (u32, u16, u32, f64);

/// Every (anchor, fragment, row, intensity) match of `requests` (rows ascending, each with
/// its anchor) against the anchors' fragment `targets`: the nearest peak of the row within
/// `tol_ppm`. Sorted by anchor, fragment and row.
fn gather(
    ms2: &str,
    requests: &[(u32, u32)],
    targets: &[Vec<f64>],
    tol_ppm: f64,
) -> Result<Vec<PatchMatch>> {
    let t = TableFile::open(ms2)?;
    let parts = t.row_group_parts()?;
    let mut first = Vec::with_capacity(parts.len());
    let mut acc = 0usize;
    for p in &parts {
        first.push(acc);
        acc += p.nrows;
    }
    let hits: Vec<Result<Vec<PatchMatch>>> = parts
        .par_iter()
        .zip(first.par_iter())
        .map(|(p, &f0)| {
            let a = requests.partition_point(|q| (q.0 as usize) < f0);
            let b = requests.partition_point(|q| (q.0 as usize) < f0 + p.nrows);
            let mut out = Vec::new();
            if a == b {
                return Ok(out);
            }
            let mut k = a;
            let mut row = f0;
            p.for_each_batch(Some(&["mz", "intensity"]), 1024, |batch| {
                let mza = crate::spectra::list_col(batch, "mz")?;
                let ina = crate::spectra::list_col(batch, "intensity")?;
                for i in 0..mza.len() {
                    while k < b && (requests[k].0 as usize) == row {
                        let ai = requests[k].1;
                        let (mz, inten) = (mza.row_slice(i, "mz")?, ina.row_slice(i, "intensity")?);
                        let n = mz.len().min(inten.len());
                        for (fi, &tg) in targets[ai as usize].iter().enumerate() {
                            if let Some(j) = nearest(&mz[..n], tg, tol_ppm) {
                                out.push((ai, fi as u16, row as u32, inten[j] as f64));
                            }
                        }
                        k += 1;
                    }
                    row += 1;
                }
                Ok(())
            })?;
            Ok(out)
        })
        .collect();
    let mut all = Vec::new();
    for h in hits {
        all.extend(h?);
    }
    all.sort_by_key(|x| (x.0, x.1, x.2));
    Ok(all)
}

/// Run the Q1 calibration and add `q1_calibration` to `cal.json`. A run with too few
/// profiles gets `status: "too_few_profiles"` and a warning.
pub fn run(p: Q1CalParams) -> Result<()> {
    let t0 = std::time::Instant::now();
    let sweep = read_sweep(p.ms2)?;
    let anchors = read_anchors(p.seed_psms, &sweep, p.q_max)?;
    let frags = read_fragments(p.library_fragments, &anchors)?;
    let (mass_off, tol) = crate::masscal::read_mass_cal(p.mass_cal, p.frag_tol_fallback_ppm)?;
    let factor = |mz: f64| mass_off.factor_at(mz);
    let (step_t, profiles) = read_profiles(p.ms2, &sweep, &anchors, &frags, &factor, tol)?;
    let mut block = if profiles.len() < MIN_PROFILES {
        warn!(
            anchors = anchors.len(),
            profiles = profiles.len(),
            "q1-calibrate: too few seed fragment profiles for a Q1 calibration"
        );
        json!({
            "version": 1,
            "status": "too_few_profiles",
            "anchors": anchors.len(),
            "profiles": profiles.len(),
        })
    } else {
        fit(
            &sweep,
            &Data {
                anchors: &anchors,
                step_t: &step_t,
                profiles: &profiles,
            },
        )
    };
    if p.fragment_decision && block["status"] == "fitted" {
        let d = learn_decision(
            p.ms2,
            &sweep,
            &anchors,
            p.library_fragments,
            &factor,
            tol,
            &block,
        )?;
        if let Some(obj) = block.as_object_mut() {
            obj.insert("fragment_decision".into(), d);
        }
    }
    let mut cal: serde_json::Value = mumdia_io::json::read_json(p.out_cal)
        .with_context(|| format!("q1-calibrate: reading {}", p.out_cal))?;
    if let Some(obj) = cal.as_object_mut() {
        obj.insert("q1_calibration".into(), block);
    }
    mumdia_io::json::write_json(p.out_cal, &cal)?;
    info!(
        elapsed_ms = t0.elapsed().as_millis() as u64,
        frag_tol_ppm = tol,
        "q1-calibrate: done"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real-data check of a finished run: `MUMDIA_BENCH_Q1CAL=<run out dir>,<library
    /// fragments> cargo test --release -- --ignored q1cal_real --nocapture`. Reads the run's
    /// seed, spectra and mass calibration, writes the calibration into a copy of its
    /// `cal.json` in the temp directory and prints it.
    #[test]
    #[ignore]
    fn q1cal_real_run() {
        let Ok(v) = std::env::var("MUMDIA_BENCH_Q1CAL") else {
            return;
        };
        let (run, libf) = v
            .split_once(',')
            .expect("<run out dir>,<library fragments>");
        let cal =
            std::env::temp_dir().join(format!("mumdia_q1cal_real_{}.json", std::process::id()));
        let cal = cal.to_string_lossy().to_string();
        std::fs::copy(format!("{run}/cal.json"), &cal).unwrap();
        let seed = format!("{run}/seed_psms.parquet");
        run_q1(
            &seed,
            &format!("{run}/spectra/spectra_ms2.parquet"),
            libf,
            &cal,
        );
        let out: serde_json::Value = mumdia_io::json::read_json(&cal).unwrap();
        println!(
            "{}",
            serde_json::to_string_pretty(&out["q1_calibration"]).unwrap()
        );
        let _ = std::fs::remove_file(&cal);
    }

    fn run_q1(seed: &str, ms2: &str, libf: &str, cal: &str) {
        let mass_cal = format!("{seed}.masscal.json");
        run(Q1CalParams {
            seed_psms: seed,
            ms2,
            library_fragments: libf,
            mass_cal: Some(&mass_cal),
            q_max: 0.01,
            frag_tol_fallback_ppm: 20.0,
            out_cal: cal,
            fragment_decision: std::env::var("MUMDIA_BENCH_Q1CAL_DECISION").is_ok(),
        })
        .unwrap();
    }

    #[test]
    fn the_transmission_is_a_trapezoid() {
        assert_eq!(transmission(0.0, 1.9, 0.76), 1.0);
        assert!((transmission(1.9 - 0.38, 1.9, 0.76) - 1.0).abs() < 1e-12);
        assert!((transmission(1.9, 1.9, 0.76) - 0.5).abs() < 1e-12);
        assert_eq!(transmission(1.9 + 0.38, 1.9, 0.76), 0.0);
        assert_eq!(transmission(-3.0, 1.9, 0.76), 0.0);
        assert_eq!(transmission(1.8, 1.9, 0.0), 1.0);
        assert_eq!(transmission(2.0, 1.9, 0.0), 0.0);
    }

    /// The fragment-conditional weights move the envelope toward the monoisotope as the
    /// fragment takes more of the precursor, and a fragment of nothing keeps the averagine.
    #[test]
    fn heavier_fragments_weight_the_monoisotope_more() {
        let a = Anchor {
            cid: 0,
            pm: 1000.0,
            z: 2,
            rt: 0.0,
            c0: 0,
        };
        let light = isotope_weights(&a, Frag { mz: 147.11, z: 1 });
        let heavy = isotope_weights(&a, Frag { mz: 1500.0, z: 1 });
        let neutral = |mz: f64, z: i32| (mz - mumdia_core::constants::PROTON) * z as f64;
        let avg = mumdia_core::constants::averagine_isotopes(neutral(1000.0, 2), q1frag::ISOTOPES);
        assert!(heavy[0] > light[0] && light[0] > avg[0] - 1e-12);
        assert!((light.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let none = isotope_weights(
            &a,
            Frag {
                mz: mumdia_core::constants::PROTON,
                z: 1,
            },
        );
        for (x, y) in none.iter().zip(&avg) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    /// A window centred `delta` above its target shows a precursor `delta` early in the
    /// sweep: the centroid of the expected profile moves by `-delta`.
    #[test]
    fn an_offset_window_moves_the_centroid_the_other_way() {
        let t: Vec<f64> = (0..40).map(|k| 640.0 + 0.7614 * k as f64).collect();
        let w = vec![1.0];
        let base = centroid(&expected(&t, 655.0, 2, &w, 0.0, 1.9, 0.76), &t).unwrap();
        let moved = centroid(&expected(&t, 655.0, 2, &w, 0.3, 1.9, 0.76), &t).unwrap();
        assert!((base - 655.0).abs() < 0.05, "{base}");
        assert!((moved - base + 0.3).abs() < 0.05, "{moved} {base}");
    }

    #[test]
    fn the_profile_sse_fits_scale_and_floor() {
        let e = vec![0.0, 0.5, 1.0, 1.0, 0.5, 0.0];
        // y = 2 e + 1, normalised: an exact fit.
        let y: Vec<f64> = e.iter().map(|v| 2.0 * v + 1.0).collect();
        assert!(profile_sse(&y, &e).abs() < 1e-12);
        // A negative floor is refused: y = e - 0.2 clipped is not fitted exactly.
        let y2: Vec<f64> = e.iter().map(|v| (v - 0.2f64).max(0.0)).collect();
        assert!(profile_sse(&y2, &e) > 0.0);
    }

    /// The whole calibration on synthetic ZT Scan spectra: windows whose real centre sits
    /// 0.3 m/z above the recorded target, 1.9 m/z half-width as read. The fit finds the
    /// offset, calls the target the centre, and the calibrated residuals are centred.
    #[test]
    fn a_synthetic_sweep_is_calibrated() {
        use mumdia_io::table::{write_table, Col};
        let dir = std::env::temp_dir().join(format!("mumdia_q1cal_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = |n: &str| dir.join(n).to_string_lossy().to_string();
        let (step, hw, shift) = (0.7614f64, 1.9f64, 0.3f64);
        let n_steps = 120usize;
        let n_cycles = 12usize;
        let target = |k: usize| 400.0 + step * k as f64;
        // 300 precursors, charge 2, at m/z spread over the sweep, eluting at cycle 6; three
        // fragments each.
        let n_prec = 300usize;
        let prec_mz: Vec<f64> = (0..n_prec)
            .map(|i| 410.0 + (i as f64 * 0.2637) % 70.0)
            .collect();
        let frag_mz = |i: usize, f: usize| 200.0 + i as f64 * 1.7 + f as f64 * 0.31;
        let (mut si, mut rt, mut wt, mut wl, mut wu, mut mzs, mut ints) =
            (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        let mut row = 0u32;
        for c in 0..n_cycles {
            for k in 0..n_steps {
                let t = target(k);
                let mut peaks: Vec<(f32, f32)> = Vec::new();
                let elution = (-((c as f64 - 6.0) / 2.0).powi(2)).exp();
                for (i, &pm) in prec_mz.iter().enumerate() {
                    // The true transmission: centred at t + shift, isotopes of a 2+.
                    let a = Anchor {
                        cid: i as u32,
                        pm,
                        z: 2,
                        rt: 0.0,
                        c0: 0,
                    };
                    for f in 0..3 {
                        let fr = Frag {
                            mz: frag_mz(i, f),
                            z: 1,
                        };
                        let w = isotope_weights(&a, fr);
                        let tr = expected(&[t], pm, 2, &w, shift, hw, step)[0];
                        let v = 1000.0 * elution * tr;
                        if v > 1.0 {
                            peaks.push((fr.mz as f32, v as f32));
                        }
                    }
                }
                peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
                si.push(row);
                rt.push(c as f64 + 0.001 * k as f64);
                wt.push(t);
                wl.push(t - hw);
                wu.push(t + hw);
                mzs.push(peaks.iter().map(|p| p.0).collect::<Vec<f32>>());
                ints.push(peaks.iter().map(|p| p.1).collect::<Vec<f32>>());
                row += 1;
            }
        }
        write_table(
            &path("ms2.parquet"),
            vec![
                Col::U32("scan_index".into(), si),
                Col::F64("rt_seconds".into(), rt),
                Col::F64("window_target".into(), wt),
                Col::F64("window_lower".into(), wl),
                Col::F64("window_upper".into(), wu),
                Col::LargeListF32("mz".into(), mzs),
                Col::LargeListF32("intensity".into(), ints),
            ],
        )
        .unwrap();
        // Seed: each precursor at the step nearest it in cycle 6.
        let seed_scan: Vec<u32> = prec_mz
            .iter()
            .map(|&pm| {
                let k = (((pm - 400.0) / step).round() as usize).min(n_steps - 1);
                (6 * n_steps + k) as u32
            })
            .collect();
        write_table(
            &path("seed.parquet"),
            vec![
                Col::U32("candidate_id".into(), (0..n_prec as u32).collect()),
                Col::Str("label".into(), vec!["target".to_string(); n_prec]),
                Col::F64("spectrum_q".into(), vec![0.001; n_prec]),
                Col::F64("precursor_mz".into(), prec_mz.clone()),
                Col::I32("charge".into(), vec![2; n_prec]),
                Col::F64("observed_rt".into(), vec![6.0; n_prec]),
                Col::U32("scan_index".into(), seed_scan),
                Col::F64("score".into(), vec![1.0; n_prec]),
            ],
        )
        .unwrap();
        let mut fc = Vec::new();
        let mut fm = Vec::new();
        let mut fi = Vec::new();
        for i in 0..n_prec {
            for f in 0..3 {
                fc.push(i as u32);
                fm.push(frag_mz(i, f));
                fi.push(1.0f32 - 0.1 * f as f32);
            }
        }
        write_table(
            &path("frags.parquet"),
            vec![
                Col::U32("candidate_id".into(), fc),
                Col::F64("mz".into(), fm),
                Col::F32("predicted_intensity".into(), fi),
            ],
        )
        .unwrap();
        mumdia_io::json::write_json(&path("cal.json"), &json!({"method": "loess"})).unwrap();
        run(Q1CalParams {
            seed_psms: &path("seed.parquet"),
            ms2: &path("ms2.parquet"),
            library_fragments: &path("frags.parquet"),
            mass_cal: None,
            q_max: 0.01,
            frag_tol_fallback_ppm: 10.0,
            out_cal: &path("cal.json"),
            fragment_decision: true,
        })
        .unwrap();
        let cal: serde_json::Value = mumdia_io::json::read_json(&path("cal.json")).unwrap();
        assert_eq!(cal["method"], "loess", "the rest of cal.json is kept");
        let q = &cal["q1_calibration"];
        assert_eq!(q["status"], "fitted", "{q}");
        assert_eq!(q["target_marks"], "centre", "{q}");
        assert_eq!(q["sweep"]["recorded"], false);
        assert_eq!(q["sweep"]["cycles"], n_cycles);
        let off = q["offset"]["median_mz"].as_f64().unwrap();
        assert!((off - shift).abs() < 0.05, "offset {off}");
        let nominal = q["residual_steps"]["nominal"]["median"].as_f64().unwrap();
        assert!((nominal + shift / step).abs() < 0.1, "nominal {nominal}");
        let after = q["residual_steps"]["calibrated"]["median"]
            .as_f64()
            .unwrap();
        assert!(after.abs() < 0.05, "calibrated {after}");
        let h = q["half_width_mz"].as_f64().unwrap();
        let e = q["edge_width_mz"].as_f64().unwrap();
        assert!(
            (h - hw).abs() < 0.1 && (e - step).abs() < 0.2,
            "h {h} e {e}"
        );
        // The decision tolerances, learned on the same anchors under the fitted window: their
        // three fragments are all theirs, so nearly every present one belongs, and the
        // tolerances are a few tenths of a step.
        let fd = &q["fragment_decision"];
        let t = &fd["tolerances"];
        let off = t["offset_steps"].as_f64().unwrap();
        let cen = t["centre_steps"].as_f64().unwrap();
        assert!((0.25..1.0).contains(&off), "{fd}");
        assert!((0.25..1.0).contains(&cen), "{fd}");
        assert!(t["corr2d_min"].as_f64().unwrap() > 0.5, "{fd}");
        let b = fd["under_tolerances"]["belong_share"].as_f64().unwrap();
        assert!(b > 0.85, "{fd}");
        // Extract reads the same block back as its model.
        let m = q1frag::model_from_calibration(q, true).unwrap();
        assert_eq!(m.tolerances.offset_steps, off);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
