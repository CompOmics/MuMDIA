//! Per-fragment Q1 decisions of scanning-quadrupole spectra (`extract.q1_fragments`).
//!
//! In a SCIEX ZT Scan DIA run the quadrupole sweeps the precursor range once a cycle in steps
//! far smaller than its window, so the fragments of a precursor show up in a run of
//! consecutive steps of every cycle, centred where the calibrated window
//! (`rt_im_train.q1_calibrate`, [`crate::stages::q1cal`]) transmits the precursor's isotope
//! envelope. A fragment m/z that another precursor shares shows up centred where THAT
//! precursor is. This module reads, for one candidate, every library fragment's matched
//! intensity in each step of a patch around the precursor over the cycles around its apex,
//! keeps every step apart (summing the steps of a cycle lost 13.7% of the
//! identifications, docs/09_extract.md, section 10), and decides per fragment.
//!
//! **The patch.** Steps whose target lies within the calibrated window's reach (`h + e/2`)
//! of the precursor's expected Q1 centre, plus [`MARGIN_STEPS`] either side; the cycles
//! [`PATCH_CYCLES`] either side of the apex cycle (inside the candidate's RT window in
//! extract). A cell holds the intensity of the nearest peak within the fragment tolerance of
//! the offset-corrected fragment m/z in that cycle's scan of that step, the rule of the seed
//! and of the Q1 calibration.
//!
//! **The expected profile** of a fragment over the step targets is the transmission (a
//! trapezoid of half-width `h` and edge width `e`, centred `delta` above each target) summed
//! over the precursor's isotopes, weighted by the Poisson-averagine abundance times the
//! chance `(1 - f)^i` that the fragment stays monoisotopic (`isotope_weights`); its centre
//! in target units is `pm + sp * mean isotope - delta`. A shift `mu` (in steps) moves it.
//!
//! **The decision** ([`decide`]), in two rounds. A fragment is present when at least
//! [`MIN_CELLS`] of its cells within the window's reach of its expected centre are coherent:
//! they hold signal and so does a neighbouring step of the same cycle. The quadrupole
//! transmits a precursor through several consecutive steps, and a random match stands
//! alone: ZT Scan spectra hold about 1,550 peaks each, so a 20 ppm window finds a peak by
//! chance in roughly 2% of the cells, and a count of cells with signal would call nearly
//! every fragment present. The consensus elution over the round's set (each fragment
//! normalised to its own total) gives the 2D apex cycle, and the fragments' Q1 profiles are
//! their patches summed over the [`PROFILE_CYCLES`] cycles either side of it, every step
//! apart. One centre `mu` and one width (a scale of `h` and `e`, round 2 only) shared by the
//! set are fitted by least squares of the normalised profiles against the expected ones with
//! a scale (the fragment's height) and a floor each, on a grid of [`MU_STEP`] steps refined
//! by a parabola; each fragment's own centre comes from the same table at the shared width,
//! its residual is `1 - R^2` under the shared fit, and its 2D correlation is the Pearson
//! correlation over the whole patch of its cells with the elution of the set's other
//! fragments times its expected transmission at the shared centre. A present fragment
//! belongs when its own centre lies within the offset tolerance of the shared centre (the
//! shared centre clamped to within the centre tolerance of the expected one, so a candidate
//! whose fragments all sit elsewhere has none that belong) and its 2D correlation reaches the
//! correlation floor; otherwise it is interfered. Round 1 fits over every present fragment,
//! round 2 over the ones round 1 says belong (when at least two do).
//!
//! **The tolerances** are learned from the confident seed PSMs ([`learn`]): their fragments'
//! offsets from the shared centre, their shared centres and their 2D correlations, as the
//! median plus [`TOL_SIGMAS`] robust standard deviations (1.4826 MAD) and the
//! [`CORR_QUANTILE`] quantile of the correlation among the fragments within the offset
//! tolerance (with guards for a degenerate anchor set: [`MIN_OFFSET_TOL`],
//! [`MIN_CENTRE_TOL`], [`MAX_CORR_MIN`]); learned once without tolerances (one round over
//! every present fragment), then again from the two-round decisions under the first
//! tolerances.
//!
//! Deterministic: every sum runs in fragment, cycle and step order, ties keep the first.

use anyhow::{bail, Context, Result};
use mumdia_core::config::{interp_held, Q1FragModel, Q1FragTolerances, Q1OffsetTerm};
use mumdia_core::constants::{averagine_isotopes, ISOTOPE_SPACING, PROTON};
use mumdia_core::types::Ms2Scan;
use rayon::prelude::*;
use serde_json::json;

use crate::calibrate::percentile;

/// Cycles read either side of the apex cycle.
pub const PATCH_CYCLES: u32 = 3;
/// Cycles either side of the 2D apex the Q1 profiles sum.
pub const PROFILE_CYCLES: usize = 2;
/// Steps read beyond the calibrated window's reach, either side.
pub const MARGIN_STEPS: f64 = 2.0;
/// Fewest coherent cells (with signal, and signal in a neighbouring step of the same cycle)
/// within the window's reach a present fragment has.
pub const MIN_CELLS: usize = 3;
/// Isotopes of the precursor envelope (as `stages::q1cal`).
pub const ISOTOPES: usize = 6;
/// The centre grid: `-MU_MAX..=MU_MAX` steps by `MU_STEP`.
pub const MU_MAX: f64 = 6.0;
pub const MU_STEP: f64 = 0.25;
/// Grid points per step and on each side of 0.
const MU_Q: usize = 4;
const MU_K: usize = 24;
const MU_N: usize = 2 * MU_K + 1;
/// The width scales round 2 fits (of `h` and `e` together).
pub const WIDTHS: [f64; 3] = [0.85, 1.0, 1.2];
/// The tolerance rule: median plus this many robust standard deviations.
pub const TOL_SIGMAS: f64 = 3.0;
/// The correlation floor: this quantile of the 2D correlation of the anchors' fragments
/// within the offset tolerance.
pub const CORR_QUANTILE: f64 = 0.05;
/// The robust standard deviation of a normal sample per unit of its MAD.
const MAD_TO_SIGMA: f64 = 1.4826;
/// Guards of the learned tolerances against a degenerate anchor set (too few, or too clean
/// to have a spread): the offset and centre tolerances are at least half a step (the
/// calibrated profiles' residual SD on ZT Scan is 0.23 steps), and the correlation floor is
/// at most 0.8.
pub const MIN_OFFSET_TOL: f64 = 0.5;
pub const MIN_CENTRE_TOL: f64 = 0.5;
pub const MAX_CORR_MIN: f64 = 0.8;

/// The fragment classes of `q1f_frag_class`.
pub const ABSENT: u8 = 0;
pub const BELONGS: u8 = 1;
pub const INTERFERED: u8 = 2;

const _: () = assert!(MU_K == (MU_MAX as usize) * MU_Q && MU_Q == (1.0 / MU_STEP) as usize);

/// The share of a precursor ion at distance `x` from a window's centre that it transmits:
/// whole within `h - e/2`, nothing beyond `h + e/2`, linear in between.
pub(crate) fn transmission(x: f64, h: f64, e: f64) -> f64 {
    let ax = x.abs();
    if e <= 0.0 {
        return if ax <= h { 1.0 } else { 0.0 };
    }
    ((h + 0.5 * e - ax) / e).clamp(0.0, 1.0)
}

/// The neutral mass of an ion of m/z `mz` and charge `z`.
fn neutral(mz: f64, z: i32) -> f64 {
    (mz - PROTON) * z as f64
}

/// The precursor isotopes' weights for a fragment: Poisson-averagine abundance times the
/// chance `(1 - f)^i` that the fragment stays monoisotopic, `f` its share of the precursor
/// mass, normalised.
pub(crate) fn isotope_weights(pm: f64, z: i32, fmz: f64, fz: i32) -> Vec<f64> {
    let m = neutral(pm, z);
    let frac = (neutral(fmz, fz) / m).clamp(0.0, 1.0);
    let mut w = averagine_isotopes(m, ISOTOPES);
    let mut keep = 1.0;
    for x in w.iter_mut() {
        *x *= keep;
        keep *= 1.0 - frac;
    }
    let s: f64 = w.iter().sum();
    if s > 0.0 {
        for x in w.iter_mut() {
            *x /= s;
        }
    }
    w
}

/// Sum of squares of `y / sum(y)` against `s E + b` with the per-profile scale `s` and
/// floor `b >= 0` that fit best (a floor that would come out negative is 0).
pub(crate) fn profile_sse(y: &[f64], e: &[f64]) -> f64 {
    let tot: f64 = y.iter().sum();
    let n = y.len() as f64;
    let (mut see, mut se, mut sey, mut sy, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&yk, &ek) in y.iter().zip(e) {
        let yn = yk / tot;
        see += ek * ek;
        se += ek;
        sey += ek * yn;
        sy += yn;
        syy += yn * yn;
    }
    sse_of(n, see, se, sey, sy, syy)
}

/// [`profile_sse`] from its sums.
#[inline]
fn sse_of(n: f64, see: f64, se: f64, sey: f64, sy: f64, syy: f64) -> f64 {
    let det = see * n - se * se;
    let (s, b) = if det.abs() > 1e-12 {
        let s = (sey * n - se * sy) / det;
        let b = (see * sy - se * sey) / det;
        if s >= 0.0 && b >= 0.0 {
            (s, b)
        } else {
            (if see > 0.0 { sey / see } else { 0.0 }, 0.0)
        }
    } else {
        (if see > 0.0 { sey / see } else { 0.0 }, 0.0)
    };
    // sum (yn - s e - b)^2, expanded.
    syy + s * s * see + b * b * n - 2.0 * s * sey - 2.0 * b * sy + 2.0 * s * b * se
}

/// The window-centre offset (m/z) of the model at precursor m/z `pm`, fragment m/z `fmz`
/// and retention time `rt`; `fmz = None` leaves the fragment term out (its terms are
/// centred on the profiles, so that is the term's mean).
pub(crate) fn offset_mz(m: &Q1FragModel, pm: f64, fmz: Option<f64>, rt: f64) -> f64 {
    m.offset_terms
        .iter()
        .map(|t| match t.var.as_str() {
            "precursor_mz" => interp_held(&t.grid, &t.offset_mz, pm),
            "fragment_mz" => fmz.map_or(0.0, |f| interp_held(&t.grid, &t.offset_mz, f)),
            "rt" => interp_held(&t.grid, &t.offset_mz, rt),
            _ => 0.0,
        })
        .sum()
}

/// How far from a window's centre it still transmits anything (m/z): `h + e/2`.
pub(crate) fn reach_mz(m: &Q1FragModel) -> f64 {
    m.half_width_mz + 0.5 * m.edge_width_mz
}

/// The model of a `q1_calibration` block of `cal.json`. `with_tolerances` requires the
/// decision tolerances (`fragment_decision.tolerances`), which extract needs and the
/// calibration's own learning pass does not have yet.
pub fn model_from_calibration(
    block: &serde_json::Value,
    with_tolerances: bool,
) -> Result<Q1FragModel> {
    let status = block["status"].as_str().unwrap_or("missing");
    if status != "fitted" {
        bail!(
            "the Q1 calibration has status \"{status}\" (anchors {}, profiles {}), so there is \
             no calibrated Q1 window to decide with; unset extract.q1_fragments for this run",
            block["anchors"],
            block["profiles"]
        );
    }
    let num = |v: &serde_json::Value, what: &str| -> Result<f64> {
        v.as_f64()
            .filter(|x| x.is_finite())
            .with_context(|| format!("q1_calibration: {what} is not a finite number"))
    };
    let step_mz = num(&block["sweep"]["step_mz"], "sweep.step_mz")?;
    let half_width_mz = num(&block["half_width_mz"], "half_width_mz")?;
    let edge_width_mz = num(&block["edge_width_mz"], "edge_width_mz")?;
    if !(step_mz > 0.0 && half_width_mz > 0.0 && edge_width_mz >= 0.0) {
        bail!(
            "q1_calibration: step {step_mz}, half-width {half_width_mz} and edge \
             {edge_width_mz} m/z are not a window"
        );
    }
    let mut offset_terms = Vec::new();
    if let Some(terms) = block["offset"]["terms"].as_object() {
        // In the fit's order, the precursor term first.
        for var in ["precursor_mz", "fragment_mz", "rt"] {
            let Some(t) = terms.get(var) else { continue };
            let vec = |k: &str| -> Result<Vec<f64>> {
                t[k].as_array()
                    .with_context(|| format!("q1_calibration: offset term {var} has no {k}"))?
                    .iter()
                    .map(|x| num(x, k))
                    .collect()
            };
            let (grid, offset) = (vec("grid")?, vec("offset_mz")?);
            if grid.len() != offset.len() || grid.windows(2).any(|w| w[1] < w[0]) {
                bail!("q1_calibration: offset term {var} has a malformed grid");
            }
            offset_terms.push(Q1OffsetTerm {
                var: var.to_string(),
                grid,
                offset_mz: offset,
            });
        }
        if let Some(other) = terms
            .keys()
            .find(|k| !["precursor_mz", "fragment_mz", "rt"].contains(&k.as_str()))
        {
            bail!("q1_calibration: unknown offset term {other}");
        }
    }
    let tolerances = if with_tolerances {
        let t = &block["fragment_decision"]["tolerances"];
        if t.is_null() {
            bail!(
                "the Q1 calibration has no fragment decision tolerances: \
                 rt_im_train.q1_calibrate learns them only when extract.q1_fragments is on"
            );
        }
        Q1FragTolerances {
            offset_steps: num(&t["offset_steps"], "tolerances.offset_steps")?,
            centre_steps: num(&t["centre_steps"], "tolerances.centre_steps")?,
            corr2d_min: num(&t["corr2d_min"], "tolerances.corr2d_min")?,
        }
    } else {
        Q1FragTolerances::default()
    };
    Ok(Q1FragModel {
        step_mz,
        half_width_mz,
        edge_width_mz,
        offset_terms,
        tolerances,
    })
}

/// The decision model of a run: `q1_calibration` of its `cal.json`, with the tolerances.
pub fn read_model(cal_json: &str) -> Result<Q1FragModel> {
    let cal: serde_json::Value = mumdia_io::json::read_json(cal_json)
        .with_context(|| format!("extract.q1_fragments: reading {cal_json}"))?;
    let block = &cal["q1_calibration"];
    if block.is_null() {
        bail!(
            "extract.q1_fragments: {cal_json} has no q1_calibration; it is written by \
             rt_im_train.q1_calibrate"
        );
    }
    model_from_calibration(block, true).with_context(|| format!("extract.q1_fragments: {cal_json}"))
}

/// The charge of a fragment named like `y7` or `b3^2` (1 without a suffix).
pub(crate) fn frag_charge_of(name: &str) -> i32 {
    name.split_once('^')
        .and_then(|(_, c)| c.parse::<i32>().ok())
        .filter(|&c| c >= 1)
        .unwrap_or(1)
}

/// The steps of a sweep: its distinct window targets at 1e-4 m/z, ascending (a step is the
/// rank of its target, as spectra_ms2 v3 records it on a regular sweep).
pub(crate) fn step_key(target: f64) -> i64 {
    (target * 1e4).round() as i64
}

/// A run's Q1 sweep as a grid: cycle x step -> the stage's scan (spectra_ms2 v3).
pub struct Q1Grid {
    pub n_cycles: usize,
    pub n_steps: usize,
    /// Step targets (m/z), ascending.
    pub target: Vec<f64>,
    /// `cell[c * n_steps + s]`: the scan (index into the stage's slice), `u32::MAX` where the
    /// cycle has no scan of that step.
    cell: Vec<u32>,
    /// Per scan of the stage, its cycle (`u32::MAX` outside the sweep).
    pub scan_cycle: Vec<u32>,
}

impl Q1Grid {
    /// The grid of `scans` from the sweep spectra_ms2 v3 recorded (`q1_cycle`, `q1_step` by
    /// `scan_index`). Refused when a scan is not in the record, when a recorded step is not
    /// the rank of its scan's target (the sweep would not be one upward pass over the
    /// targets) or when one cycle holds a step twice.
    pub fn build(scans: &[Ms2Scan], sweep: &crate::spectra::Q1SweepCols) -> Result<Q1Grid> {
        let mut keys: Vec<i64> = scans.iter().map(|s| step_key(s.window.target_mz)).collect();
        keys.sort_unstable();
        keys.dedup();
        let target: Vec<f64> = keys.iter().map(|&k| k as f64 / 1e4).collect();
        let n_steps = keys.len();
        let mut rec: Vec<(u32, u32, u32)> = (0..sweep.scan_index.len())
            .map(|i| (sweep.scan_index[i], sweep.cycle[i], sweep.step[i]))
            .collect();
        rec.sort_unstable();
        let mut scan_cycle = vec![u32::MAX; scans.len()];
        let mut scan_step = vec![u32::MAX; scans.len()];
        let mut n_cycles = 0usize;
        for (si, s) in scans.iter().enumerate() {
            let j = rec.partition_point(|r| r.0 < s.scan_index);
            let Some(&(idx, c, st)) = rec.get(j).filter(|r| r.0 == s.scan_index) else {
                bail!(
                    "extract.q1_fragments: scan {} is not in the sweep spectra_ms2 recorded",
                    s.scan_index
                );
            };
            debug_assert_eq!(idx, s.scan_index);
            let rank = keys.partition_point(|&k| k < step_key(s.window.target_mz)) as u32;
            if st != rank {
                bail!(
                    "extract.q1_fragments: scan {} is step {st} of its cycle in spectra_ms2, and \
                     its target {} m/z is step {rank} of the sweep; the per-fragment decision \
                     reads one upward sweep over the targets",
                    s.scan_index,
                    s.window.target_mz
                );
            }
            scan_cycle[si] = c;
            scan_step[si] = st;
            n_cycles = n_cycles.max(c as usize + 1);
        }
        let mut cell = vec![u32::MAX; n_cycles * n_steps];
        for si in 0..scans.len() {
            let k = scan_cycle[si] as usize * n_steps + scan_step[si] as usize;
            if cell[k] != u32::MAX {
                bail!(
                    "extract.q1_fragments: cycle {} holds step {} twice (scans {} and {})",
                    scan_cycle[si],
                    scan_step[si],
                    scans[cell[k] as usize].scan_index,
                    scans[si].scan_index
                );
            }
            cell[k] = si as u32;
        }
        Ok(Q1Grid {
            n_cycles,
            n_steps,
            target,
            cell,
            scan_cycle,
        })
    }

    /// The scan of step `s` in cycle `c`.
    pub fn scan(&self, c: u32, s: u32) -> Option<usize> {
        if c as usize >= self.n_cycles || s as usize >= self.n_steps {
            return None;
        }
        let v = self.cell[c as usize * self.n_steps + s as usize];
        (v != u32::MAX).then_some(v as usize)
    }

    /// The steps whose target lies within `reach` of `x`, as `(first, count)`.
    pub fn steps_within(&self, x: f64, reach: f64) -> (u32, usize) {
        let a = self.target.partition_point(|&t| t < x - reach);
        let b = self.target.partition_point(|&t| t <= x + reach);
        (a as u32, b.saturating_sub(a))
    }

    /// The step whose target is nearest `x` (the lower on a tie).
    pub fn nearest_step(&self, x: f64) -> u32 {
        let j = self.target.partition_point(|&t| t < x);
        if j == 0 {
            return 0;
        }
        if j >= self.target.len() {
            return (self.target.len() - 1) as u32;
        }
        if (x - self.target[j - 1]) <= (self.target[j] - x) {
            (j - 1) as u32
        } else {
            j as u32
        }
    }
}

/// One library fragment as the decision reads it.
#[derive(Clone, Debug)]
pub struct FragQ1 {
    /// The m/z matched in the spectra (the library m/z times the mass-offset factor).
    pub target_mz: f64,
    /// Predicted intensity (negative values count as 0).
    pub pred: f64,
    /// The isotope weights and their mean index.
    w: Vec<f64>,
    ibar: f64,
    /// The expected Q1 centre, in target units (m/z of the step targets).
    pub centre: f64,
}

/// One candidate as the decision reads it.
#[derive(Clone, Debug)]
pub struct CandQ1 {
    /// Isotope spacing of the precursor (m/z).
    sp: f64,
    /// The precursor-level expected Q1 centre (target units), where the patch is centred.
    pub centre: f64,
    pub frags: Vec<FragQ1>,
}

impl CandQ1 {
    /// A candidate of precursor m/z `pm` and charge `z` with its apex at `rt`, and its
    /// library fragments (`frag_mz`, `frag_z`, `pred`); `factor(mz)` is the mass-offset
    /// factor the spectra are matched with.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: &Q1FragModel,
        pm: f64,
        z: i32,
        rt: f64,
        frag_mz: &[f64],
        frag_z: &[i32],
        pred: &[f64],
        factor: &dyn Fn(f64) -> f64,
    ) -> CandQ1 {
        let z = z.max(1);
        let sp = ISOTOPE_SPACING / z as f64;
        let pre = averagine_isotopes(neutral(pm, z), ISOTOPES);
        let ipre: f64 = pre.iter().enumerate().map(|(i, w)| i as f64 * w).sum();
        let centre = pm + sp * ipre - offset_mz(m, pm, None, rt);
        let frags = (0..frag_mz.len())
            .map(|i| {
                let w = isotope_weights(pm, z, frag_mz[i], frag_z[i].max(1));
                let ibar: f64 = w.iter().enumerate().map(|(k, x)| k as f64 * x).sum();
                FragQ1 {
                    target_mz: frag_mz[i] * factor(frag_mz[i]),
                    pred: pred[i].max(0.0),
                    centre: pm + sp * ibar - offset_mz(m, pm, Some(frag_mz[i]), rt),
                    w,
                    ibar,
                }
            })
            .collect();
        CandQ1 { sp, centre, frags }
    }

    /// The expected profile value of fragment `f` at target `t`, shifted by `mu` steps of
    /// `step` m/z, at width scale `omega`.
    fn expected_at(
        &self,
        m: &Q1FragModel,
        f: usize,
        t: f64,
        mu: f64,
        step: f64,
        omega: f64,
    ) -> f64 {
        let fr = &self.frags[f];
        let u = t - fr.centre - mu * step;
        let (h, e) = (omega * m.half_width_mz, omega * m.edge_width_mz);
        fr.w.iter()
            .enumerate()
            .map(|(i, wi)| wi * transmission(self.sp * (i as f64 - fr.ibar) - u, h, e))
            .sum()
    }
}

/// A candidate's RT x step patch: `y[(f * cycles + c) * ns + s]`.
#[derive(Clone, Debug)]
pub struct Patch {
    /// The cycles read, ascending.
    pub cycles: Vec<u32>,
    /// First step, and the targets of the steps read.
    pub s0: u32,
    pub t: Vec<f64>,
    pub nf: usize,
    y: Vec<f32>,
}

impl Patch {
    /// An empty patch of `nf` fragments.
    pub fn new(cycles: Vec<u32>, s0: u32, t: Vec<f64>, nf: usize) -> Patch {
        let n = nf * cycles.len() * t.len();
        Patch {
            cycles,
            s0,
            t,
            nf,
            y: vec![0.0; n],
        }
    }

    pub fn ns(&self) -> usize {
        self.t.len()
    }

    fn idx(&self, f: usize, ci: usize, si: usize) -> usize {
        (f * self.cycles.len() + ci) * self.t.len() + si
    }

    pub fn get(&self, f: usize, ci: usize, si: usize) -> f64 {
        self.y[self.idx(f, ci, si)] as f64
    }

    pub fn set(&mut self, f: usize, ci: usize, si: usize, v: f32) {
        let k = self.idx(f, ci, si);
        self.y[k] = v;
    }

    /// The step spacing of the patch (m/z): its targets' span over its steps, or `fallback`
    /// for a patch of one step.
    fn step(&self, fallback: f64) -> f64 {
        let n = self.t.len();
        if n >= 2 && self.t[n - 1] > self.t[0] {
            (self.t[n - 1] - self.t[0]) / (n - 1) as f64
        } else {
            fallback
        }
    }
}

/// The patch steps of `cand` on `grid`: within the window's reach plus [`MARGIN_STEPS`] of
/// its expected centre.
pub fn patch_steps(grid: &Q1Grid, m: &Q1FragModel, cand: &CandQ1) -> (u32, usize) {
    grid.steps_within(cand.centre, reach_mz(m) + MARGIN_STEPS * m.step_mz)
}

/// Read `cand`'s patch from the stage's scans over `cycles` and the steps `s0 .. s0 + ns`.
pub fn read_patch(
    grid: &Q1Grid,
    scans: &[Ms2Scan],
    cand: &CandQ1,
    cycles: Vec<u32>,
    s0: u32,
    ns: usize,
    tol_ppm: f64,
) -> Patch {
    let t = grid.target[s0 as usize..s0 as usize + ns].to_vec();
    let mut p = Patch::new(cycles, s0, t, cand.frags.len());
    for ci in 0..p.cycles.len() {
        let c = p.cycles[ci];
        for si in 0..ns {
            let Some(scan) = grid.scan(c, s0 + si as u32) else {
                continue;
            };
            let peaks = &scans[scan].peaks;
            for (f, fr) in cand.frags.iter().enumerate() {
                if let Some(j) = crate::spectra::nearest_peak(peaks, fr.target_mz, tol_ppm) {
                    p.set(f, ci, si, peaks[j].intensity);
                }
            }
        }
    }
    p
}

/// The decision of one candidate ([`decide`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Decision {
    /// Per library fragment: [`ABSENT`], [`BELONGS`] or [`INTERFERED`].
    pub class: Vec<u8>,
    /// Its own Q1 centre against its expected one (steps); NaN without a Q1 profile.
    pub offset: Vec<f64>,
    /// `1 - R^2` of its profile under the shared fit; NaN without a Q1 profile.
    pub resid: Vec<f64>,
    /// Its 2D correlation; NaN for an absent fragment.
    pub corr2d: Vec<f64>,
    /// Its observed intensity over the patch.
    pub tot: Vec<f64>,
    /// Whether it has a Q1 profile at the 2D apex.
    pub has_q: Vec<bool>,
    /// Whether it was in the final round's fitted set.
    pub in_set: Vec<bool>,
    /// The shared centre (steps, against the expected one) and width scale; 0 and 1 without
    /// a fit.
    pub centre: f64,
    pub width: f64,
    pub has_fit: bool,
    /// The 2D apex: an index into the patch's cycles.
    pub apex_ci: usize,
    /// The patch's step spacing (m/z).
    pub step: f64,
}

/// The per-candidate summary of a [`Decision`]: the `q1f_*` columns.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Summary {
    pub n_present: f64,
    pub n_belong: f64,
    pub n_interfered: f64,
    /// Predicted-intensity share of the fragments that belong.
    pub belong_share: f64,
    /// Observed-intensity share of the interfered fragments among the present ones.
    pub interfered_share: f64,
    /// Median and median absolute deviation of the fragments' own Q1 centres (steps).
    pub offset_median: f64,
    pub offset_mad: f64,
    /// Mean 2D correlation of the fragments that belong.
    pub corr2d: f64,
    /// Mean `1 - R^2` of the fitted set under the shared fit; 1 without a fit.
    pub fit_resid: f64,
    pub centre: f64,
    pub width: f64,
}

/// The median of `v` (the mean of the middle two of an even count); 0 when empty.
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Median absolute deviation of `v` around its median (unscaled).
fn mad(v: &[f64]) -> f64 {
    let m = median(v);
    let d: Vec<f64> = v.iter().map(|x| (x - m).abs()).collect();
    median(&d)
}

impl Decision {
    /// The summary, with each fragment's predicted intensity `pred`.
    pub fn summary(&self, cand: &CandQ1) -> Summary {
        let n = self.class.len();
        let count = |c: u8| self.class.iter().filter(|&&x| x == c).count() as f64;
        let ptot: f64 = cand.frags.iter().map(|f| f.pred).sum();
        let pb: f64 = (0..n)
            .filter(|&f| self.class[f] == BELONGS)
            .map(|f| cand.frags[f].pred)
            .sum();
        let otot: f64 = (0..n)
            .filter(|&f| self.class[f] != ABSENT)
            .map(|f| self.tot[f])
            .sum();
        let oi: f64 = (0..n)
            .filter(|&f| self.class[f] == INTERFERED)
            .map(|f| self.tot[f])
            .sum();
        let offs: Vec<f64> = (0..n)
            .filter(|&f| self.has_q[f])
            .map(|f| self.offset[f])
            .collect();
        let corr_b: Vec<f64> = (0..n)
            .filter(|&f| self.class[f] == BELONGS)
            .map(|f| self.corr2d[f])
            .collect();
        let res: Vec<f64> = (0..n)
            .filter(|&f| self.in_set[f] && self.has_q[f])
            .map(|f| self.resid[f])
            .collect();
        Summary {
            n_present: n as f64 - count(ABSENT),
            n_belong: count(BELONGS),
            n_interfered: count(INTERFERED),
            belong_share: if ptot > 0.0 { pb / ptot } else { 0.0 },
            interfered_share: if otot > 0.0 { oi / otot } else { 0.0 },
            offset_median: median(&offs),
            offset_mad: mad(&offs),
            corr2d: if corr_b.is_empty() {
                0.0
            } else {
                corr_b.iter().sum::<f64>() / corr_b.len() as f64
            },
            fit_resid: if res.is_empty() {
                1.0
            } else {
                res.iter().sum::<f64>() / res.len() as f64
            },
            centre: self.centre,
            width: self.width,
        }
    }

    /// A decision for `n` fragments without any signal read: every fragment absent.
    pub fn empty(n: usize) -> Decision {
        Decision {
            class: vec![ABSENT; n],
            offset: vec![f64::NAN; n],
            resid: vec![f64::NAN; n],
            corr2d: vec![f64::NAN; n],
            tot: vec![0.0; n],
            has_q: vec![false; n],
            in_set: vec![false; n],
            centre: 0.0,
            width: 1.0,
            has_fit: false,
            apex_ci: 0,
            step: 0.0,
        }
    }
}

/// The parabola through the grid values around the minimum at `k`, as a refined grid
/// position (in grid units, within half a point of `k`); `k` itself at the grid's ends or
/// on a flat stretch.
fn refine(v: &[f64], k: usize) -> f64 {
    if k == 0 || k + 1 >= v.len() {
        return k as f64;
    }
    let (a, b, c) = (v[k - 1], v[k], v[k + 1]);
    let den = a - 2.0 * b + c;
    if den.is_nan() || den <= 0.0 || !den.is_finite() {
        return k as f64;
    }
    k as f64 + (0.5 * (a - c) / den).clamp(-0.5, 0.5)
}

/// The grid value `k` of the centre grid, in steps.
#[inline]
fn mu_of(k: f64) -> f64 {
    -MU_MAX + k * MU_STEP
}

/// One round of [`decide`] over the fitted `set`.
struct Round {
    apex_ci: usize,
    has_q: Vec<bool>,
    mu_f: Vec<f64>,
    resid: Vec<f64>,
    corr2d: Vec<f64>,
    centre: f64,
    width: f64,
    has_fit: bool,
}

/// The decision of `cand` on its patch `p` under the model `m`: with tolerances, the two
/// rounds and the classes; without (the learning pass), one round over every present
/// fragment, which then all count as belonging. `start_ci` is the patch cycle of the
/// candidate's apex, the 2D apex when the fragments carry no signal to find one.
pub fn decide(
    p: &Patch,
    cand: &CandQ1,
    m: &Q1FragModel,
    tol: Option<&Q1FragTolerances>,
    start_ci: usize,
) -> Decision {
    let nf = p.nf;
    let nc = p.cycles.len();
    let ns = p.ns();
    if nf == 0 || nc == 0 || ns == 0 {
        return Decision::empty(nf);
    }
    let step = p.step(m.step_mz);
    // Per fragment: its observed total over the patch, and its coherent cells inside the
    // calibrated window's reach of its expected centre: cells with signal whose neighbouring
    // step in the same cycle has signal too. The quadrupole transmits a precursor through
    // several consecutive steps, and a random match stands alone (a 20 ppm window finds a
    // peak by chance in roughly 2% of the cells of ZT Scan spectra, so most fragments have a
    // few).
    let reach = reach_mz(m);
    let mut tot = vec![0.0f64; nf];
    let mut coherent = vec![0usize; nf];
    for f in 0..nf {
        let centre = cand.frags[f].centre;
        for ci in 0..nc {
            for si in 0..ns {
                let v = p.get(f, ci, si);
                if v <= 0.0 {
                    continue;
                }
                tot[f] += v;
                let near = (si > 0 && p.get(f, ci, si - 1) > 0.0)
                    || (si + 1 < ns && p.get(f, ci, si + 1) > 0.0);
                if near && (p.t[si] - centre).abs() <= reach {
                    coherent[f] += 1;
                }
            }
        }
    }
    let present: Vec<bool> = coherent.iter().map(|&c| c >= MIN_CELLS).collect();
    // Each fragment's elution, normalised to its own total over the patch.
    let elution: Vec<Vec<f64>> = (0..nf)
        .map(|f| {
            (0..nc)
                .map(|ci| {
                    if tot[f] > 0.0 {
                        (0..ns).map(|si| p.get(f, ci, si)).sum::<f64>() / tot[f]
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    // The expected profiles on the centre grid, per fragment and width, as a table over the
    // grid positions J = 4 si + 2K - k (targets spaced `step` apart from `p.t[0]`).
    let nj = MU_Q * (ns - 1) + 2 * MU_K + 1;
    let table = |f: usize, omega: f64| -> Vec<f64> {
        let fr = &cand.frags[f];
        let (h, e) = (omega * m.half_width_mz, omega * m.edge_width_mz);
        let base = p.t[0] - fr.centre;
        (0..nj)
            .map(|j| {
                let u = base + step * (j as f64 - MU_K as f64) / MU_Q as f64;
                fr.w.iter()
                    .enumerate()
                    .map(|(i, wi)| wi * transmission(cand.sp * (i as f64 - fr.ibar) - u, h, e))
                    .sum()
            })
            .collect()
    };
    // Tables computed once per (fragment, width) across both rounds.
    let mut tables: Vec<Vec<Option<Vec<f64>>>> = vec![vec![None; WIDTHS.len()]; nf];

    let round =
        |set: &[bool], widths: &[usize], tables: &mut Vec<Vec<Option<Vec<f64>>>>| -> Round {
            // The 2D apex cycle: the consensus elution of the set.
            let mut g = vec![0.0f64; nc];
            for f in 0..nf {
                if set[f] {
                    for ci in 0..nc {
                        g[ci] += elution[f][ci];
                    }
                }
            }
            let mut apex_ci = start_ci.min(nc - 1);
            let mut best = 0.0f64;
            for (ci, &v) in g.iter().enumerate() {
                if v > best {
                    best = v;
                    apex_ci = ci;
                }
            }
            let lo = apex_ci.saturating_sub(PROFILE_CYCLES);
            let hi = (apex_ci + PROFILE_CYCLES).min(nc - 1);
            // The Q1 profiles at the 2D apex.
            let q: Vec<Vec<f64>> = (0..nf)
                .map(|f| {
                    (0..ns)
                        .map(|si| (lo..=hi).map(|ci| p.get(f, ci, si)).sum())
                        .collect()
                })
                .collect();
            let has_q: Vec<bool> = (0..nf)
                .map(|f| present[f] && q[f].iter().filter(|&&v| v > 0.0).count() >= 2)
                .collect();
            // Per fragment and width: the profile SSE over the centre grid, and its SST.
            let mut sse: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); WIDTHS.len()]; nf];
            let mut sst = vec![0.0f64; nf];
            for f in 0..nf {
                if !has_q[f] {
                    continue;
                }
                let qt: f64 = q[f].iter().sum();
                let yn: Vec<f64> = q[f].iter().map(|v| v / qt).collect();
                let n = ns as f64;
                let sy: f64 = yn.iter().sum();
                let syy: f64 = yn.iter().map(|v| v * v).sum();
                sst[f] = syy - sy * sy / n;
                for &wi in widths {
                    let tab = tables[f][wi].get_or_insert_with(|| table(f, WIDTHS[wi]));
                    let row: Vec<f64> = (0..MU_N)
                        .map(|k| {
                            let (mut see, mut se, mut sey) = (0.0, 0.0, 0.0);
                            for (si, &y) in yn.iter().enumerate() {
                                let e = tab[MU_Q * si + 2 * MU_K - k];
                                see += e * e;
                                se += e;
                                sey += e * y;
                            }
                            sse_of(n, see, se, sey, sy, syy)
                        })
                        .collect();
                    sse[f][wi] = row;
                }
            }
            // The shared centre and width: least summed SSE over the set's fragments with a
            // profile; widths in order, grid points in order, the first minimum kept.
            let mut fit: Option<(usize, usize, f64)> = None;
            let mut totals: Vec<Vec<f64>> = vec![Vec::new(); WIDTHS.len()];
            for &wi in widths {
                let mut tsum = vec![0.0f64; MU_N];
                let mut any = false;
                for f in 0..nf {
                    if set[f] && has_q[f] {
                        any = true;
                        for (k, t) in tsum.iter_mut().enumerate() {
                            *t += sse[f][wi][k];
                        }
                    }
                }
                if !any {
                    continue;
                }
                for (k, &v) in tsum.iter().enumerate() {
                    if fit.is_none_or(|(_, _, b)| v < b) {
                        fit = Some((wi, k, v));
                    }
                }
                totals[wi] = tsum;
            }
            let one = WIDTHS
                .iter()
                .position(|&w| w == 1.0)
                .expect("width 1 in WIDTHS");
            let (wi, centre, has_fit) = match fit {
                Some((wi, k, _)) => (wi, mu_of(refine(&totals[wi], k)), true),
                None => (one, 0.0, false),
            };
            let omega = WIDTHS[wi];
            // Each fragment's own centre (at the shared width) and its residual under the fit.
            let mut mu_f = vec![f64::NAN; nf];
            let mut resid = vec![f64::NAN; nf];
            for f in 0..nf {
                if !has_q[f] {
                    continue;
                }
                // Every width the fit chose from was tabulated for every fragment with a profile.
                let row = &sse[f][wi];
                debug_assert_eq!(row.len(), MU_N);
                let mut kb = 0usize;
                for k in 1..MU_N {
                    if row[k] < row[kb] {
                        kb = k;
                    }
                }
                mu_f[f] = mu_of(refine(row, kb));
                let e: Vec<f64> =
                    p.t.iter()
                        .map(|&t| cand.expected_at(m, f, t, centre, step, omega))
                        .collect();
                let s = profile_sse(&q[f], &e);
                resid[f] = if sst[f] > 0.0 {
                    (s / sst[f]).clamp(0.0, 1.0)
                } else {
                    0.0
                };
            }
            // 2D correlation of every present fragment with the elution of the set's others
            // times its own expected transmission at the shared centre.
            let mut corr2d = vec![f64::NAN; nf];
            for f in 0..nf {
                if !present[f] {
                    continue;
                }
                let mut other: Vec<f64> = (0..nc)
                    .map(|ci| {
                        let own = if set[f] { elution[f][ci] } else { 0.0 };
                        (g[ci] - own).max(0.0)
                    })
                    .collect();
                if other.iter().all(|&v| v <= 0.0) {
                    // The set is this fragment alone: the other present fragments.
                    other = (0..nc)
                        .map(|ci| {
                            (0..nf)
                                .filter(|&o| o != f && present[o])
                                .map(|o| elution[o][ci])
                                .sum()
                        })
                        .collect();
                }
                let e: Vec<f64> =
                    p.t.iter()
                        .map(|&t| cand.expected_at(m, f, t, centre, step, omega))
                        .collect();
                let mut obs = Vec::with_capacity(nc * ns);
                let mut model = Vec::with_capacity(nc * ns);
                for (ci, &o) in other.iter().enumerate() {
                    for (si, &ev) in e.iter().enumerate() {
                        obs.push(p.get(f, ci, si));
                        model.push(o * ev);
                    }
                }
                corr2d[f] = crate::stats::pearson(&obs, &model);
            }
            Round {
                apex_ci,
                has_q,
                mu_f,
                resid,
                corr2d,
                centre,
                width: omega,
                has_fit,
            }
        };

    let classify = |r: &Round, tol: &Q1FragTolerances| -> Vec<u8> {
        let refc = r.centre.clamp(-tol.centre_steps, tol.centre_steps);
        (0..nf)
            .map(|f| {
                if !present[f] {
                    ABSENT
                } else if r.has_q[f]
                    && (r.mu_f[f] - refc).abs() <= tol.offset_steps
                    && r.corr2d[f] >= tol.corr2d_min
                {
                    BELONGS
                } else {
                    INTERFERED
                }
            })
            .collect()
    };

    let one = [WIDTHS
        .iter()
        .position(|&w| w == 1.0)
        .expect("width 1 in WIDTHS")];
    let all: Vec<usize> = (0..WIDTHS.len()).collect();
    let r1 = round(&present, &one, &mut tables);
    let (r, class, set) = match tol {
        None => {
            let class = present
                .iter()
                .map(|&x| if x { BELONGS } else { ABSENT })
                .collect();
            (r1, class, present.clone())
        }
        Some(t) => {
            let c1 = classify(&r1, t);
            let b: Vec<bool> = c1.iter().map(|&c| c == BELONGS).collect();
            let set2 = if b.iter().filter(|&&x| x).count() >= 2 {
                b
            } else {
                present.clone()
            };
            let r2 = round(&set2, &all, &mut tables);
            let c2 = classify(&r2, t);
            (r2, c2, set2)
        }
    };
    Decision {
        class,
        offset: r.mu_f,
        resid: r.resid,
        corr2d: r.corr2d,
        tot,
        has_q: r.has_q,
        in_set: set,
        centre: r.centre,
        width: r.width,
        has_fit: r.has_fit,
        apex_ci: r.apex_ci,
        step,
    }
}

/// One row's decision as psms_extracted v6 writes it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Q1fRow {
    pub summary: Summary,
    /// Per library fragment, in the library's order: the class code, its own centre
    /// (steps, against its expected one), its residual and its 2D correlation (NaN where
    /// [`Decision`] has none).
    pub class: Vec<f32>,
    pub offset: Vec<f32>,
    pub resid: Vec<f32>,
    pub corr2d: Vec<f32>,
    /// `retrace.q1_repick`: how far the apex moved (s); 0 unmoved.
    pub moved_s: f64,
    /// The 2D apex cycle minus the cycle of the apex scan.
    pub apex_shift_cycles: f64,
}

impl Q1fRow {
    /// The row of decision `d` on `patch`, whose apex scan is in cycle `apex_cycle`.
    pub fn of(d: &Decision, cand: &CandQ1, patch: &Patch, apex_cycle: u32) -> Q1fRow {
        let f32s = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let c2d = patch.cycles.get(d.apex_ci).copied().unwrap_or(apex_cycle);
        Q1fRow {
            summary: d.summary(cand),
            class: d.class.iter().map(|&c| c as f32).collect(),
            offset: f32s(&d.offset),
            resid: f32s(&d.resid),
            corr2d: f32s(&d.corr2d),
            moved_s: 0.0,
            apex_shift_cycles: c2d as f64 - apex_cycle as f64,
        }
    }

    /// The row of a candidate of `n` fragments whose apex is not in the sweep.
    pub fn empty(n: usize, cand: &CandQ1) -> Q1fRow {
        let d = Decision::empty(n);
        Q1fRow {
            summary: d.summary(cand),
            class: vec![ABSENT as f32; n],
            offset: vec![f32::NAN; n],
            resid: vec![f32::NAN; n],
            corr2d: vec![f32::NAN; n],
            moved_s: 0.0,
            apex_shift_cycles: 0.0,
        }
    }
}

/// What extract decides with: the run's grid and model, its scans and the fragment
/// tolerance (ppm).
pub struct DecideAt<'a> {
    pub grid: &'a Q1Grid,
    pub model: &'a Q1FragModel,
    pub scans: &'a [Ms2Scan],
    pub tol_ppm: f64,
}

impl DecideAt<'_> {
    /// The decision of `cand` with its apex at scan `apex_scan`: the patch over the cycles
    /// [`PATCH_CYCLES`] either side of the scan's cycle whose scan at the patch's middle step
    /// lies inside `[lo, hi]` (the candidate's RT window; the apex cycle always), and its
    /// [`decide`] under the model's tolerances. `None` when the scan is not in the sweep or
    /// the patch has no step.
    pub fn at(
        &self,
        cand: &CandQ1,
        apex_scan: usize,
        lo: f64,
        hi: f64,
    ) -> Option<(Patch, Decision)> {
        let g = self.grid;
        let c0 = *g.scan_cycle.get(apex_scan)?;
        if c0 == u32::MAX {
            return None;
        }
        let (s0, ns) = patch_steps(g, self.model, cand);
        if ns == 0 {
            return None;
        }
        let mid = g.nearest_step(cand.centre);
        let first = c0.saturating_sub(PATCH_CYCLES);
        let last = (c0 + PATCH_CYCLES).min(g.n_cycles.saturating_sub(1) as u32);
        let cycles: Vec<u32> = (first..=last)
            .filter(|&c| {
                c == c0
                    || g.scan(c, mid).is_some_and(|s| {
                        let r = self.scans[s].rt_seconds;
                        !(r < lo || r > hi)
                    })
            })
            .collect();
        let start_ci = cycles.iter().position(|&c| c == c0)?;
        let patch = read_patch(g, self.scans, cand, cycles, s0, ns, self.tol_ppm);
        let d = decide(
            &patch,
            cand,
            self.model,
            Some(&self.model.tolerances),
            start_ci,
        );
        Some((patch, d))
    }

    /// The scan of the 2D apex of `d` on `patch`: in the 2D apex cycle, the step nearest the
    /// fitted Q1 centre whose window as read covers the precursor m/z `pm` (searched outward
    /// up to three steps), which is a scan of the candidate's own extraction grid.
    pub fn apex_scan(&self, cand: &CandQ1, patch: &Patch, d: &Decision, pm: f64) -> Option<usize> {
        let g = self.grid;
        let c = *patch.cycles.get(d.apex_ci)?;
        let x = cand.centre + d.centre * d.step;
        let n = g.nearest_step(x) as i64;
        for k in [0i64, -1, 1, -2, 2, -3, 3] {
            let s = n + k;
            if s < 0 {
                continue;
            }
            if let Some(si) = g.scan(c, s as u32) {
                let w = &self.scans[si].window;
                if w.lower_mz <= pm && pm <= w.upper_mz {
                    return Some(si);
                }
            }
        }
        None
    }
}

/// The joint score `retrace.q1_repick` compares peaks by: the cosine of the square-rooted
/// intensities of the fragments that belong (summed over the profile cycles around the 2D
/// apex, every other fragment 0) against the square-rooted predictions, times `ln(1 + I)`
/// with `I` their prediction-weighted mean intensity.
pub fn repick_score(p: &Patch, d: &Decision, cand: &CandQ1) -> f64 {
    let nc = p.cycles.len();
    if nc == 0 {
        return 0.0;
    }
    let lo = d.apex_ci.saturating_sub(PROFILE_CYCLES);
    let hi = (d.apex_ci + PROFILE_CYCLES).min(nc - 1);
    let (mut dot, mut nq, mut np, mut wi, mut ws) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (f, fr) in cand.frags.iter().enumerate() {
        let qf: f64 = if d.class[f] == BELONGS {
            (lo..=hi)
                .map(|ci| (0..p.ns()).map(|si| p.get(f, ci, si)).sum::<f64>())
                .sum()
        } else {
            0.0
        };
        dot += fr.pred.sqrt() * qf.sqrt();
        nq += qf;
        np += fr.pred;
        wi += fr.pred * qf;
        ws += fr.pred;
    }
    if nq <= 0.0 || np <= 0.0 || ws <= 0.0 {
        return 0.0;
    }
    dot / (np.sqrt() * nq.sqrt()) * (1.0 + wi / ws).ln()
}

/// What [`learn`] collects from one pass over the anchors' decisions: each fragment's
/// offset from its shared centre (fragments with a Q1 profile), each anchor's shared centre
/// (anchors with a fit), and per present fragment with a profile its offset and 2D
/// correlation.
#[derive(Default)]
struct Pass {
    d: Vec<f64>,
    centre: Vec<f64>,
    corr: Vec<(f64, f64)>,
    present: usize,
    belong: usize,
    interfered: usize,
    fragments: usize,
}

fn collect(ds: &[Decision]) -> Pass {
    let mut p = Pass::default();
    for d in ds {
        if d.has_fit {
            p.centre.push(d.centre);
        }
        for f in 0..d.class.len() {
            p.fragments += 1;
            match d.class[f] {
                BELONGS => p.belong += 1,
                INTERFERED => p.interfered += 1,
                _ => {}
            }
            if d.class[f] != ABSENT {
                p.present += 1;
            }
            if d.has_q[f] {
                let x = d.offset[f] - d.centre;
                p.d.push(x);
                if d.corr2d[f].is_finite() {
                    p.corr.push((x, d.corr2d[f]));
                }
            }
        }
    }
    p
}

/// The robust standard deviation of `v` (1.4826 MAD).
fn robust_sd(v: &[f64]) -> f64 {
    MAD_TO_SIGMA * mad(v)
}

/// The tolerances of one pass ([`learn`]'s rule).
fn tolerances_of(p: &Pass) -> Q1FragTolerances {
    let (md, sd) = (median(&p.d), robust_sd(&p.d));
    let offset_steps = (md.abs() + TOL_SIGMAS * sd).max(MIN_OFFSET_TOL);
    let (mc, sc) = (median(&p.centre), robust_sd(&p.centre));
    let centre_steps = (mc.abs() + TOL_SIGMAS * sc).max(MIN_CENTRE_TOL);
    let within: Vec<f64> = p
        .corr
        .iter()
        .filter(|(x, _)| (x - md).abs() <= offset_steps)
        .map(|&(_, c)| c)
        .collect();
    let corr2d_min = if within.is_empty() {
        0.0
    } else {
        percentile(&within, CORR_QUANTILE).clamp(-1.0, MAX_CORR_MIN)
    };
    Q1FragTolerances {
        offset_steps,
        centre_steps,
        corr2d_min,
    }
}

/// Quantiles of `v` for the report.
fn dist(v: &[f64]) -> serde_json::Value {
    if v.is_empty() {
        return json!({"n": 0});
    }
    json!({
        "n": v.len(),
        "median": median(v),
        "sd": robust_sd(v),
        "p05": percentile(v, 0.05),
        "p25": percentile(v, 0.25),
        "p75": percentile(v, 0.75),
        "p95": percentile(v, 0.95),
    })
}

fn pass_json(p: &Pass) -> serde_json::Value {
    let corr: Vec<f64> = p.corr.iter().map(|&(_, c)| c).collect();
    let share = |n: usize| n as f64 / p.fragments.max(1) as f64;
    json!({
        "fragments": p.fragments,
        "present_share": share(p.present),
        "belong_share": share(p.belong),
        "interfered_share": share(p.interfered),
        "offset_from_shared_steps": dist(&p.d),
        "shared_centre_steps": dist(&p.centre),
        "corr2d": dist(&corr),
    })
}

/// One anchor of [`learn`]: its candidate, its patch and the patch cycle of its seed scan.
pub struct LearnAnchor {
    pub cand: CandQ1,
    pub patch: Patch,
    pub start_ci: usize,
}

/// Learn the decision tolerances from the confident seed PSMs' patches, and the
/// `fragment_decision` block of `cal.json`.
pub fn learn(anchors: &[LearnAnchor], m: &Q1FragModel) -> (Q1FragTolerances, serde_json::Value) {
    let run = |tol: Option<&Q1FragTolerances>| -> Vec<Decision> {
        anchors
            .par_iter()
            .map(|a| decide(&a.patch, &a.cand, m, tol, a.start_ci))
            .collect()
    };
    let p0 = collect(&run(None));
    let t0 = tolerances_of(&p0);
    let p1 = collect(&run(Some(&t0)));
    let t1 = tolerances_of(&p1);
    let p2 = collect(&run(Some(&t1)));
    let tj = |t: &Q1FragTolerances| {
        json!({
            "offset_steps": t.offset_steps,
            "centre_steps": t.centre_steps,
            "corr2d_min": t.corr2d_min,
        })
    };
    let block = json!({
        "version": 1,
        "anchors": anchors.len(),
        "patch_cycles": 2 * PATCH_CYCLES + 1,
        "profile_cycles": 2 * PROFILE_CYCLES + 1,
        "margin_steps": MARGIN_STEPS,
        "min_cells": MIN_CELLS,
        "rule": {
            "tol_sigmas": TOL_SIGMAS,
            "corr_quantile": CORR_QUANTILE,
        },
        "tolerances": tj(&t1),
        "first_tolerances": tj(&t0),
        "learning_pass": pass_json(&p0),
        "under_first_tolerances": pass_json(&p1),
        "under_tolerances": pass_json(&p2),
    });
    (t1, block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_core::types::{IsolationWindow, Peak};

    const STEP: f64 = 0.7614;

    fn model() -> Q1FragModel {
        Q1FragModel {
            step_mz: STEP,
            half_width_mz: 2.57,
            edge_width_mz: 4.0,
            offset_terms: vec![Q1OffsetTerm {
                var: "precursor_mz".into(),
                grid: vec![400.0, 900.0],
                offset_mz: vec![0.2, -0.05],
            }],
            tolerances: Q1FragTolerances {
                offset_steps: 0.8,
                centre_steps: 0.8,
                corr2d_min: 0.5,
            },
        }
    }

    fn cand(m: &Q1FragModel, pm: f64) -> CandQ1 {
        let fmz = [300.2, 401.3, 512.7, 620.1, 733.4, 845.0];
        let fz = [1; 6];
        let pred = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3];
        CandQ1::new(m, pm, 2, 600.0, &fmz, &fz, &pred, &|_| 1.0)
    }

    /// A patch of `cand` over targets `t` and seven cycles, the fragments in `own` at the
    /// candidate's expected profile with a Gaussian elution peaking at cycle 3, the ones in
    /// `shifted` (with their shift in m/z) at a precursor that much higher, a little floor.
    fn synthetic(
        m: &Q1FragModel,
        c: &CandQ1,
        t: &[f64],
        own: &[usize],
        shifted: &[(usize, f64)],
    ) -> Patch {
        let mut p = Patch::new((10..17).collect(), 0, t.to_vec(), c.frags.len());
        for ci in 0..7 {
            let el = (-0.5 * ((ci as f64 - 3.0) / 1.2).powi(2)).exp();
            for (si, &tt) in t.iter().enumerate() {
                for &f in own {
                    let v = 1000.0 * c.frags[f].pred * el * c.expected_at(m, f, tt, 0.0, STEP, 1.0);
                    p.set(f, ci, si, v as f32);
                }
                for &(f, d) in shifted {
                    let v = 800.0 * el * c.expected_at(m, f, tt, d / STEP, STEP, 1.0);
                    p.set(f, ci, si, v as f32);
                }
            }
        }
        p
    }

    fn targets_around(x: f64, n: i32) -> Vec<f64> {
        let base = (x / STEP).round() * STEP;
        (-n..=n).map(|k| base + k as f64 * STEP).collect()
    }

    #[test]
    fn transmission_is_the_trapezoid() {
        assert_eq!(transmission(0.0, 2.0, 1.0), 1.0);
        assert_eq!(transmission(1.5, 2.0, 1.0), 1.0);
        assert!((transmission(2.0, 2.0, 1.0) - 0.5).abs() < 1e-12);
        assert_eq!(transmission(2.6, 2.0, 1.0), 0.0);
        assert_eq!(transmission(1.9, 2.0, 0.0), 1.0);
        assert_eq!(transmission(2.1, 2.0, 0.0), 0.0);
    }

    #[test]
    fn the_fragment_conditional_weights_shift_the_centre_down() {
        let m = model();
        let c = cand(&m, 650.0);
        // A heavier fragment is likelier to carry the heavy isotope itself, so it stays
        // monoisotopic mostly when the precursor is light: its centre sits lower.
        assert!(c.frags[5].centre < c.frags[0].centre);
        assert!(c.centre > c.frags[5].centre);
        for f in &c.frags {
            assert!((f.w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn own_fragments_belong_and_a_neighbours_are_interfered() {
        let m = model();
        let c = cand(&m, 650.0);
        let t = targets_around(c.centre, 8);
        // Fragments 0-3 the candidate's, 4 a precursor 2.3 m/z (3 steps) higher, 5 absent.
        let p = synthetic(&m, &c, &t, &[0, 1, 2, 3], &[(4, 2.3)]);
        let d = decide(&p, &c, &m, Some(&m.tolerances), 3);
        assert_eq!(d.class, vec![1, 1, 1, 1, 2, 0], "{d:?}");
        assert!(d.centre.abs() < 0.1, "{}", d.centre);
        assert!((d.offset[4] - 3.0).abs() < 0.3, "{}", d.offset[4]);
        for f in 0..4 {
            assert!(d.offset[f].abs() < 0.1, "{f}: {}", d.offset[f]);
            assert!(d.corr2d[f] > 0.95, "{f}: {}", d.corr2d[f]);
            assert!(d.resid[f] < 0.05, "{f}: {}", d.resid[f]);
        }
        assert_eq!(d.apex_ci, 3);
        let s = d.summary(&c);
        assert_eq!((s.n_present, s.n_belong, s.n_interfered), (5.0, 4.0, 1.0));
        let pt: f64 = c.frags.iter().map(|f| f.pred).sum();
        assert!((s.belong_share - (1.0 + 0.8 + 0.6 + 0.5) / pt).abs() < 1e-12);
        assert!(s.interfered_share > 0.0 && s.interfered_share < 1.0);
        assert!(s.corr2d > 0.95 && s.fit_resid < 0.05);
    }

    #[test]
    fn a_shadow_whose_fragments_all_sit_elsewhere_has_none_that_belong() {
        let m = model();
        let c = cand(&m, 650.0);
        let t = targets_around(c.centre, 8);
        let shifted: Vec<(usize, f64)> = (0..6).map(|f| (f, 2.6)).collect();
        let p = synthetic(&m, &c, &t, &[], &shifted);
        let d = decide(&p, &c, &m, Some(&m.tolerances), 3);
        assert!(d.class.iter().all(|&x| x == INTERFERED), "{d:?}");
        // The shared centre found where they are.
        assert!((d.centre - 2.6 / STEP).abs() < 0.3, "{}", d.centre);
        let s = d.summary(&c);
        assert_eq!(s.n_belong, 0.0);
        assert_eq!(s.belong_share, 0.0);
        assert!((s.offset_median - 2.6 / STEP).abs() < 0.3);
    }

    #[test]
    fn a_fragment_off_in_rt_is_interfered_by_its_2d_correlation() {
        let m = model();
        let c = cand(&m, 650.0);
        let t = targets_around(c.centre, 8);
        let mut p = synthetic(&m, &c, &t, &[0, 1, 2, 3, 4], &[]);
        // Fragment 4: the right Q1 centre, eluting two cycles later than the others.
        for ci in 0..7 {
            let el = (-0.5 * ((ci as f64 - 6.0) / 0.6).powi(2)).exp();
            for (si, &tt) in t.iter().enumerate() {
                let v = 1000.0 * el * c.expected_at(&m, 4, tt, 0.0, STEP, 1.0);
                p.set(4, ci, si, v as f32);
            }
        }
        let d = decide(&p, &c, &m, Some(&m.tolerances), 3);
        assert_eq!(&d.class[..5], &[1, 1, 1, 1, 2], "{d:?}");
        assert!(d.corr2d[4] < 0.5, "{}", d.corr2d[4]);
    }

    #[test]
    fn without_tolerances_every_present_fragment_counts() {
        let m = model();
        let c = cand(&m, 650.0);
        let t = targets_around(c.centre, 8);
        let p = synthetic(&m, &c, &t, &[0, 1, 2], &[(3, 2.3)]);
        let d = decide(&p, &c, &m, None, 3);
        assert_eq!(d.class, vec![1, 1, 1, 1, 0, 0]);
        assert!(d.has_fit);
        // The deterministic decision: the same bits twice.
        let e = decide(&p, &c, &m, None, 3);
        assert_eq!(format!("{d:?}"), format!("{e:?}"));
    }

    #[test]
    fn an_empty_patch_is_all_absent() {
        let m = model();
        let c = cand(&m, 650.0);
        let t = targets_around(c.centre, 8);
        let p = Patch::new((0..7).collect(), 0, t, 6);
        let d = decide(&p, &c, &m, Some(&m.tolerances), 3);
        assert!(d.class.iter().all(|&x| x == ABSENT));
        assert!(!d.has_fit && d.centre == 0.0 && d.apex_ci == 3);
        let s = d.summary(&c);
        assert_eq!(s.n_present, 0.0);
        assert_eq!(s.fit_resid, 1.0);
    }

    /// Random matches stand alone in a cycle, so a fragment with only those is absent; the
    /// same number of cells in a run of neighbouring steps is present.
    #[test]
    fn isolated_random_matches_are_absent() {
        let m = model();
        let c = cand(&m, 650.0);
        let t = targets_around(c.centre, 8);
        let mut p = synthetic(&m, &c, &t, &[0, 1, 2, 3], &[]);
        // Fragment 4: six isolated hits, no two in neighbouring steps of one cycle.
        for (ci, si) in [(0, 2), (1, 6), (2, 10), (3, 4), (4, 8), (6, 12)] {
            p.set(4, ci, si, 500.0);
        }
        // Fragment 5: three neighbouring steps of one cycle near the centre.
        for si in 7..10 {
            p.set(5, 3, si, 50.0);
        }
        let d = decide(&p, &c, &m, Some(&m.tolerances), 3);
        assert_eq!(d.class[4], ABSENT, "{d:?}");
        assert_ne!(d.class[5], ABSENT, "{d:?}");
    }

    #[test]
    fn the_repick_score_prefers_the_peak_whose_fragments_belong() {
        let m = model();
        let c = cand(&m, 650.0);
        let t = targets_around(c.centre, 8);
        let good = synthetic(&m, &c, &t, &[0, 1, 2, 3, 4, 5], &[]);
        let bad = synthetic(&m, &c, &t, &[0], &[(1, 2.3), (2, 2.3), (3, 2.3), (4, 2.3)]);
        let dg = decide(&good, &c, &m, Some(&m.tolerances), 3);
        let db = decide(&bad, &c, &m, Some(&m.tolerances), 3);
        assert!(repick_score(&good, &dg, &c) > repick_score(&bad, &db, &c));
    }

    #[test]
    fn the_tolerances_are_learned_from_the_anchors() {
        let m = model();
        let anchors: Vec<LearnAnchor> = (0..60)
            .map(|i| {
                let pm = 450.0 + 7.3 * i as f64;
                let c = cand(&m, pm);
                let t = targets_around(c.centre, 8);
                // One fragment in six interfered by a neighbour 2 m/z away.
                let p = synthetic(&m, &c, &t, &[0, 1, 2, 3, 4], &[(5, 2.0)]);
                LearnAnchor {
                    cand: c,
                    patch: p,
                    start_ci: 3,
                }
            })
            .collect();
        let (tol, block) = learn(&anchors, &m);
        assert!(
            tol.offset_steps >= MIN_OFFSET_TOL && tol.offset_steps < 1.5,
            "{tol:?}"
        );
        assert!(
            tol.centre_steps >= MIN_CENTRE_TOL && tol.centre_steps < 1.5,
            "{tol:?}"
        );
        assert!(
            tol.corr2d_min > 0.5 && tol.corr2d_min <= MAX_CORR_MIN,
            "{tol:?}"
        );
        assert_eq!(block["anchors"], 60);
        let shares = &block["under_tolerances"];
        let b = shares["belong_share"].as_f64().unwrap();
        assert!((b - 5.0 / 6.0).abs() < 0.02, "{shares}");
        // Deterministic.
        let (tol2, block2) = learn(&anchors, &m);
        assert_eq!(tol, tol2);
        assert_eq!(block, block2);
    }

    #[test]
    fn the_model_is_read_from_the_calibration_block() {
        let block = json!({
            "status": "fitted",
            "sweep": {"step_mz": STEP},
            "half_width_mz": 2.57,
            "edge_width_mz": 4.0,
            "offset": {"terms": {
                "rt": {"grid": [100.0, 800.0], "offset_mz": [0.01, -0.01]},
                "precursor_mz": {"grid": [400.0, 900.0], "offset_mz": [0.2, -0.05]},
            }},
            "fragment_decision": {"tolerances": {
                "offset_steps": 0.7, "centre_steps": 0.6, "corr2d_min": 0.4,
            }},
        });
        let m = model_from_calibration(&block, true).unwrap();
        assert_eq!(m.offset_terms[0].var, "precursor_mz");
        assert_eq!(m.offset_terms[1].var, "rt");
        assert_eq!(m.tolerances.corr2d_min, 0.4);
        assert!((offset_mz(&m, 400.0, None, 100.0) - 0.21).abs() < 1e-12);
        assert!((offset_mz(&m, 900.0, Some(500.0), 800.0) + 0.06).abs() < 1e-12);
        let mut no_tol = block.clone();
        no_tol["fragment_decision"] = serde_json::Value::Null;
        assert!(model_from_calibration(&no_tol, true).is_err());
        assert!(model_from_calibration(&no_tol, false).is_ok());
        let mut failed = block.clone();
        failed["status"] = json!("too_few_profiles");
        assert!(model_from_calibration(&failed, false).is_err());
        let mut odd = block;
        odd["offset"]["terms"]["charge"] = json!({"grid": [1.0], "offset_mz": [0.0]});
        assert!(model_from_calibration(&odd, false).is_err());
    }

    #[test]
    fn fragment_charges_come_from_the_name() {
        assert_eq!(frag_charge_of("y7"), 1);
        assert_eq!(frag_charge_of("b3^2"), 2);
        assert_eq!(frag_charge_of("y12^3"), 3);
        assert_eq!(frag_charge_of("y5^x"), 1);
    }

    /// `cycles` sweeps of `steps` scans, the targets STEP apart; the scans in RT order.
    fn sweep(cycles: u32, steps: u32) -> (Vec<Ms2Scan>, crate::spectra::Q1SweepCols) {
        let mut scans = Vec::new();
        let mut cols = crate::spectra::Q1SweepCols {
            scan_index: Vec::new(),
            cycle: Vec::new(),
            step: Vec::new(),
        };
        for c in 0..cycles {
            for s in 0..steps {
                let t = 400.0 + STEP * s as f64;
                let idx = scans.len() as u32 * 2 + 7;
                scans.push(Ms2Scan {
                    scan_index: idx,
                    rt_seconds: c as f64 + 0.001 * s as f64,
                    window: IsolationWindow {
                        target_mz: t,
                        lower_mz: t - 1.9,
                        upper_mz: t + 1.9,
                        im_lower: None,
                        im_upper: None,
                    },
                    peaks: vec![Peak {
                        mz: 500.0,
                        intensity: (c * 100 + s) as f32,
                    }],
                    im: None,
                });
                cols.scan_index.push(idx);
                cols.cycle.push(c);
                cols.step.push(s);
            }
        }
        (scans, cols)
    }

    #[test]
    fn the_grid_maps_cycles_and_steps_to_scans() {
        let (scans, cols) = sweep(4, 20);
        let g = Q1Grid::build(&scans, &cols).unwrap();
        assert_eq!((g.n_cycles, g.n_steps), (4, 20));
        assert_eq!(g.scan(2, 5), Some(2 * 20 + 5));
        assert_eq!(g.scan(4, 0), None);
        assert_eq!(g.nearest_step(400.0 + STEP * 7.4), 7);
        assert_eq!(g.nearest_step(0.0), 0);
        assert_eq!(g.nearest_step(1e4), 19);
        let (s0, n) = g.steps_within(400.0 + STEP * 10.0, 2.0 * STEP + 1e-9);
        assert_eq!((s0, n), (8, 5));
        // The patch reads the nearest peak of each cell.
        let m = model();
        let c = CandQ1::new(&m, 405.0, 2, 1.0, &[500.0], &[1], &[1.0], &|_| 1.0);
        let p = read_patch(&g, &scans, &c, vec![1, 2], 3, 2, 10.0);
        assert_eq!(p.get(0, 1, 1), (200 + 4) as f64);
        // A recorded step that is not the target's rank is refused.
        let mut bad = cols;
        bad.step[3] = 4;
        assert!(Q1Grid::build(&scans, &bad).is_err());
    }
}
