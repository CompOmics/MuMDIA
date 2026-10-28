//! The profile-centred window (`quant.window_centre = "profile"`) and the pattern-fit
//! fragment rules (`quant.fragment_selection = "pattern_fit"` / `"pattern_profile"`),
//! docs/12_quant_lfq_align_mbr_report_audit.md, "Profile-centred window and pattern fit".
//!
//! Profile-centred window. Per candidate and run, the raw traces of the 6 fragments with
//! the highest library intensity (ties in row order; a fragment without a trace adds
//! nothing) are summed sample by sample around the sample `k` nearest the identification
//! apex. The window centre is `k + s`, where `s` in `0, -1, +1, -2, +2` (in that order)
//! maximises the sum over the 3-sample block centred on `k + s`; a later `s` must beat
//! the best so far by a relative 1e-9, so ties go to the smaller shift. Every fragment of
//! the candidate is then integrated over the fixed window around that one centre.
//!
//! Pattern fit. Per candidate, `p[f] = predicted_intensity^1.5` is the library pattern,
//! the same in every run. The fragment set `S` is the 10 fragments with the largest `p`
//! (`p > 0`) among those observed in at least one run of the experiment, the same set in
//! every run; `scale` is the sum of `p` over the 3 largest of them. Window samples `x[f, t]`
//! (the `2h + 1` samples around the centre, `h = fixed_scan_halfwidth`) are modelled as
//! `mu = p[f] g[t]`, a rank-1 fragment x time model with a free elution profile `g`, with
//! the noise variance `phi_cell mu + (0.5 mu)^2 + phi_cell^2`, `phi_cell = phi / dt`.
//! Seven passes of `g[t] = sum_f W x / sum_f W p` with Huber weights (1 standard deviation)
//! on the standardised excess `max(0, (x - mu) / sd)`, so only signal above the model is
//! down-weighted. `pattern_profile` reports `scale x` the trapezoid of `g`. `pattern_fit`
//! winsorises every cell at `mu + sd`, integrates the winsorised traces, and fits those
//! areas to the pattern the same way (`c = sum w a / sum w p`, variance
//! `phi mu + (0.5 mu)^2 + phi^2`, seven passes); it reports `c x scale`.
//!
//! `phi` is one dispersion per experiment ([`dispersion_label_free`]): per candidate in at
//! least 3 runs, each run's fragment areas are divided by the run's share of the
//! candidate's total area, and per fragment the variance over the runs of those
//! normalised areas, corrected for the degree of freedom the scale took, is divided by
//! their mean. `phi` is the median of that ratio over the cells whose mean lies between
//! the 20th and 60th percentile of the means. It reads no condition or species label.
//! `dt` is the median sample spacing within `2h` samples of every window centre.
//!
//! Arithmetic follows the prototype it was confirmed with: trapezoid weights in f64 from
//! the f32 retention times, sums in fragment order, numpy's linear quantile and median.
//! Determinism: every reduction runs in one thread over a fixed order, and every map is
//! ordered.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use mumdia_io::table::TableFile;

/// Fragments whose summed trace places the profile-centred window.
pub const PROFILE_FRAGMENTS: usize = 6;
/// The profile-centred window moves by at most this many samples.
pub const PROFILE_SEARCH: usize = 2;
/// Samples either side of the block centre that the block sums (a 3-sample block).
const PROFILE_BLOCK_HALF: usize = 1;
/// Samples either side of `k` the summed profile spans.
pub const PROFILE_SPAN: usize = PROFILE_SEARCH + PROFILE_BLOCK_HALF;
/// Relative margin a later shift must beat the best block by.
const PROFILE_TIE: f64 = 1e-9;

/// Fragments of the pattern set `S`.
pub const PATTERN_FRAGMENTS: usize = 10;
/// Fragments whose pattern values set the quantity's scale.
pub const PATTERN_SCALE_FRAGMENTS: usize = 3;
/// Exponent of the library intensity in the pattern.
pub const PATTERN_GAMMA: f64 = 1.5;
/// Relative pattern error in the noise variance.
pub const PATTERN_KAPPA: f64 = 0.5;
/// Huber threshold, in standard deviations of the excess.
pub const PATTERN_HUBER: f64 = 1.0;
/// Reweighting passes (each fit is evaluated once more after the last).
pub const PATTERN_ITERATIONS: usize = 6;
/// The cells whose mean lies between these quantiles of the means estimate `phi`.
const DISPERSION_QUANTILES: (f64, f64) = (0.2, 0.6);
/// A candidate needs this many runs, and this many fragments with a positive area sum, for
/// its cells to estimate `phi`.
pub const DISPERSION_MIN_RUNS: usize = 3;
const DISPERSION_MIN_FRAGMENTS: usize = 3;

/// What the cross-run step fits for the pattern rules: the experiment's dispersion, the
/// median sample spacing, and per candidate the fragments observed in at least one run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PatternFit {
    /// Area-level dispersion `phi` of `Var = phi x mean`.
    pub phi: f64,
    /// Median spacing (s) of the window samples, which turns `phi` into the sample-level
    /// `phi / dt`.
    pub dt: f64,
    /// Candidate -> names of its fragments observed (with a trace) in at least one run.
    pub observed: BTreeMap<u32, BTreeSet<String>>,
    /// Cells that estimated `phi`, and the cells eligible before the quantile cut.
    pub n_cells: usize,
    pub n_cells_eligible: usize,
}

impl PatternFit {
    /// The sample-level dispersion of the first level.
    pub fn phi_cell(&self) -> f64 {
        self.phi / self.dt
    }
}

/// numpy's `median`: the middle value, or the mean of the two middle values; NaN for none.
pub fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// numpy's `quantile(..., method = "linear")` of a sorted, non-empty slice.
pub fn quantile_sorted(v: &[f64], q: f64) -> f64 {
    let n = v.len();
    let virt = (n as f64 - 1.0) * q;
    if virt >= (n - 1) as f64 {
        return v[n - 1];
    }
    if virt < 0.0 {
        return v[0];
    }
    let prev = virt.floor();
    let gamma = virt - prev;
    let i = prev as usize;
    let (a, b) = (v[i], v[(i + 1).min(n - 1)]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// The slots of the `n` largest finite library intensities (descending, ties in slot
/// order), returned in slot order.
pub fn top_predicted_slots(preds: &[f32], n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..preds.len()).filter(|&i| preds[i].is_finite()).collect();
    order.sort_by(|&a, &b| preds[b].total_cmp(&preds[a]).then(a.cmp(&b)));
    order.truncate(n);
    order.sort_unstable();
    order
}

/// The shift of the profile-centred window from the summed profile `p`, where `p[j]` is
/// the summed intensity at sample `k + j - PROFILE_SPAN` (0 outside the trace).
pub fn profile_shift(p: &[f64; 2 * PROFILE_SPAN + 1]) -> isize {
    let mut best = -1.0f64;
    let mut shift = 0isize;
    for s in [0isize, -1, 1, -2, 2] {
        let c = (s + PROFILE_SPAN as isize) as usize;
        // numpy sums the 3-sample block as a + (b + c).
        let v = p[c - 1] + (p[c] + p[c + 1]);
        if v > best * (1.0 + PROFILE_TIE) {
            best = v;
            shift = s;
        }
    }
    shift
}

/// Trapezoid weights of window samples at retention times `rt` (NaN outside the trace):
/// half of each adjacent spacing, f64; a window with one sample weighs it 1.
pub fn trapezoid_weights(rt: &[f64]) -> Vec<f64> {
    let n = rt.len();
    let mut w = vec![0.0f64; n];
    let finite: Vec<bool> = rt.iter().map(|r| r.is_finite()).collect();
    for c in 0..n.saturating_sub(1) {
        let dt = if finite[c] && finite[c + 1] {
            rt[c + 1] - rt[c]
        } else {
            0.0
        };
        w[c] += 0.5 * dt;
        w[c + 1] += 0.5 * dt;
    }
    if finite.iter().filter(|&&f| f).count() == 1 {
        for (wi, &f) in w.iter_mut().zip(&finite) {
            *wi = if f { 1.0 } else { 0.0 };
        }
    }
    w
}

/// `sum_t x[t] w[t]`, in sample order.
pub fn weighted_area(x: &[f64], w: &[f64]) -> f64 {
    let mut a = 0.0f64;
    for (xi, wi) in x.iter().zip(w) {
        a += xi * wi;
    }
    a
}

/// The `k` slots with the largest `p` among `ok` ones (ties in slot order), as a mask.
fn top_mask(p: &[f64], ok: &[bool], k: usize) -> Vec<bool> {
    let mut order: Vec<usize> = (0..p.len()).filter(|&i| ok[i]).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    let mut m = vec![false; p.len()];
    for &i in order.iter().take(k) {
        m[i] = true;
    }
    m
}

/// Huber weight of the standardised excess of `x` over `mu`.
fn huber_excess(x: f64, mu: f64, var: f64, h: f64) -> f64 {
    let r = (x - mu) / var.sqrt();
    // numpy's maximum propagates a NaN, which then fails `r > h`.
    let r = if r.is_nan() { r } else { r.max(0.0) };
    if r > h {
        h / r.max(1e-300)
    } else {
        1.0
    }
}

fn noise_var(mu: f64, phi: f64, kappa: f64) -> f64 {
    let km = kappa * mu;
    phi * mu + km * km + phi * phi
}

/// One candidate in one run, ready for the pattern rules.
pub struct PatternInput<'a> {
    /// Window samples, row-major `[slot][t]`, `t` over the `2h + 1` window samples; 0 for a
    /// fragment without a trace and outside the trace.
    pub x: &'a [f64],
    /// Retention times of the window samples, NaN outside the trace.
    pub rt: &'a [f64],
    /// Library intensity per slot.
    pub pred: &'a [f32],
    /// Whether the slot's fragment was observed in at least one run of the experiment.
    pub eligible: &'a [bool],
}

/// A pattern rule's quantity, the fragments of `S`, and the status.
pub type PatternQuantity = (Option<f64>, usize, &'static str);

/// The quantity of one candidate in one run under `pattern_fit` (`profile = false`) or
/// `pattern_profile` (`profile = true`).
pub fn pattern_quantity(
    inp: &PatternInput,
    phi: f64,
    phi_cell: f64,
    profile: bool,
) -> PatternQuantity {
    let nt = inp.rt.len();
    let nf = inp.pred.len();
    debug_assert_eq!(inp.x.len(), nf * nt);
    let p: Vec<f64> = inp
        .pred
        .iter()
        .map(|&v| {
            let v = v as f64;
            let v = if v.is_nan() { 0.0 } else { v };
            v.powf(PATTERN_GAMMA)
        })
        .collect();
    let ok: Vec<bool> = (0..nf).map(|f| p[f] > 0.0 && inp.eligible[f]).collect();
    let sel = top_mask(&p, &ok, PATTERN_FRAGMENTS);
    let n_sel = sel.iter().filter(|&&s| s).count();
    if n_sel == 0 {
        return (None, 0, "no_pattern_fragments");
    }
    let mut scale = 0.0f64;
    for (f, &on) in top_mask(&p, &ok, PATTERN_SCALE_FRAGMENTS)
        .iter()
        .enumerate()
    {
        if on {
            scale += p[f];
        }
    }
    let pp: Vec<f64> = (0..nf).map(|f| if sel[f] { p[f] } else { 0.0 }).collect();
    let tm: Vec<bool> = inp.rt.iter().map(|r| r.is_finite()).collect();
    let tw = trapezoid_weights(inp.rt);
    // Level 1: the rank-1 cell model.
    let mut w = vec![0.0f64; nf * nt];
    for f in 0..nf {
        for t in 0..nt {
            if sel[f] && tm[t] {
                w[f * nt + t] = 1.0;
            }
        }
    }
    let mut g = vec![0.0f64; nt];
    for it in 0..=PATTERN_ITERATIONS {
        for t in 0..nt {
            let (mut den, mut num) = (0.0f64, 0.0f64);
            for f in 0..nf {
                den += w[f * nt + t] * pp[f];
                num += w[f * nt + t] * inp.x[f * nt + t];
            }
            g[t] = if den > 0.0 { num / den } else { 0.0 };
        }
        if it == PATTERN_ITERATIONS {
            break;
        }
        for f in 0..nf {
            for t in 0..nt {
                let i = f * nt + t;
                w[i] = if sel[f] && tm[t] {
                    let mu = pp[f] * g[t];
                    huber_excess(
                        inp.x[i],
                        mu,
                        noise_var(mu, phi_cell, PATTERN_KAPPA),
                        PATTERN_HUBER,
                    )
                } else {
                    0.0
                };
            }
        }
    }
    let finish = |v: f64| -> PatternQuantity {
        let q = v * scale;
        if q.is_finite() && q > 0.0 {
            (Some(q), n_sel, "quantified")
        } else {
            (None, n_sel, "nonpositive_pattern_fit")
        }
    };
    if profile {
        return finish(weighted_area(&g, &tw));
    }
    // Winsorise every cell of S at the model plus one standard deviation, and integrate.
    let mut a = vec![0.0f64; nf];
    let mut row = vec![0.0f64; nt];
    for f in 0..nf {
        for t in 0..nt {
            let xi = inp.x[f * nt + t];
            row[t] = if sel[f] && tm[t] {
                let mu = pp[f] * g[t];
                xi.min(mu + PATTERN_HUBER * noise_var(mu, phi_cell, PATTERN_KAPPA).sqrt())
            } else {
                xi
            };
        }
        a[f] = if sel[f] {
            weighted_area(&row, &tw)
        } else {
            0.0
        };
    }
    // Level 2: the areas against the pattern.
    let mut wf: Vec<f64> = sel.iter().map(|&s| if s { 1.0 } else { 0.0 }).collect();
    let mut c = f64::NAN;
    for it in 0..=PATTERN_ITERATIONS {
        let (mut sp, mut sa) = (0.0f64, 0.0f64);
        for f in 0..nf {
            sp += wf[f] * pp[f];
            sa += wf[f] * a[f];
        }
        c = if sp > 0.0 { sa / sp } else { f64::NAN };
        if it == PATTERN_ITERATIONS {
            break;
        }
        let cc = if c.is_nan() { 0.0 } else { c };
        for f in 0..nf {
            wf[f] = if sel[f] {
                let mu = cc * pp[f];
                huber_excess(a[f], mu, noise_var(mu, phi, PATTERN_KAPPA), PATTERN_HUBER)
            } else {
                0.0
            };
        }
    }
    finish(c)
}

/// One accepted candidate in one run, for [`dispersion_label_free`]: its fragment areas,
/// keyed by fragment, in slot order.
pub struct CandidateRun {
    pub cid: u32,
    pub areas: Vec<(u32, f64)>,
}

/// The label-free dispersion `phi` over every run (`runs[r]` the candidates of run `r`),
/// with the number of cells that estimated it and the number eligible. `None` when no
/// candidate has enough runs and fragments, or when `phi` is not positive (runs that do not
/// vary, where the noise model has no scale).
pub fn dispersion_label_free(runs: &[Vec<CandidateRun>]) -> Option<(f64, usize, usize)> {
    // Per candidate: (runs, sum of the run totals), in run order.
    let mut per_cand: BTreeMap<u32, (usize, f64)> = BTreeMap::new();
    for run in runs {
        for cr in run {
            let tot = cr.areas.iter().fold(0.0f64, |s, &(_, a)| s + a);
            let e = per_cand.entry(cr.cid).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += tot;
        }
    }
    // (candidate, fragment) -> (sum, sum of squares) of the run-normalised areas.
    let mut cells: BTreeMap<(u32, u32), (f64, f64)> = BTreeMap::new();
    for run in runs {
        for cr in run {
            let (nr, sum_tot) = per_cand[&cr.cid];
            let mt = sum_tot / nr.max(1) as f64;
            let tot = cr.areas.iter().fold(0.0f64, |s, &(_, a)| s + a);
            let f = if tot > 0.0 { mt / tot } else { 0.0 };
            for &(frag, a) in &cr.areas {
                let an = a * f;
                let e = cells.entry((cr.cid, frag)).or_insert((0.0, 0.0));
                e.0 += an;
                e.1 += an * an;
            }
        }
    }
    let mut npos: BTreeMap<u32, usize> = BTreeMap::new();
    for (&(c, _), &(s1, _)) in &cells {
        if s1 > 0.0 {
            *npos.entry(c).or_default() += 1;
        }
    }
    let mut mm: Vec<f64> = Vec::new();
    let mut vv: Vec<f64> = Vec::new();
    for (&(c, _), &(s1, s2)) in &cells {
        let nr = per_cand[&c].0;
        let np = npos.get(&c).copied().unwrap_or(0);
        if nr < DISPERSION_MIN_RUNS || s1 <= 0.0 || np < DISPERSION_MIN_FRAGMENTS {
            continue;
        }
        let nrf = nr as f64;
        let m = s1 / nrf;
        let v = (s2 / nrf - m * m) * (nrf / (nrf - 1.0).max(1.0));
        let npf = np as f64;
        let v = v * (npf / (npf - 1.0).max(1.0));
        mm.push(m);
        vv.push(v);
    }
    if mm.is_empty() {
        return None;
    }
    let mut sorted = mm.clone();
    sorted.sort_by(f64::total_cmp);
    let ql = quantile_sorted(&sorted, DISPERSION_QUANTILES.0);
    let qh = quantile_sorted(&sorted, DISPERSION_QUANTILES.1);
    let mut ratio: Vec<f64> = mm
        .iter()
        .zip(&vv)
        .filter(|(&m, _)| m >= ql && m <= qh)
        .map(|(&m, &v)| v / m)
        .collect();
    let n_used = ratio.len();
    let phi = median(&mut ratio);
    (phi.is_finite() && phi > 0.0).then_some((phi, n_used, mm.len()))
}

/// Fit the pattern rules over the first-pass `fragment_quant` tables of every run (the
/// cross-run export with the pattern columns, v6), in run order. `None` when `phi` cannot
/// be estimated (fewer than [`DISPERSION_MIN_RUNS`] runs, no candidate in that many, or runs
/// that do not vary).
pub fn fit_pattern(fragment_tables: &[String]) -> Result<Option<PatternFit>> {
    let mut names: BTreeMap<String, u32> = BTreeMap::new();
    let mut runs: Vec<Vec<CandidateRun>> = Vec::with_capacity(fragment_tables.len());
    let mut observed: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();
    let mut spacings: Vec<f64> = Vec::new();
    for path in fragment_tables {
        let t = TableFile::open(path)?;
        let need = |col: &str| -> anyhow::Error {
            anyhow::anyhow!(
                "the pattern fit reads `{col}` from {path}: pass the first-pass tables of the \
                 cross-run export under quant.fragment_selection = pattern_fit or \
                 pattern_profile (fragment_quant v6, quant --export-cross-run)"
            )
        };
        let cid = t.u32("candidate_id")?;
        let name = t.str("fragment_name")?;
        let area = t.f64("pattern_area").map_err(|_| need("pattern_area"))?;
        let obs = t.bool("observed").map_err(|_| need("observed"))?;
        let (off, vals) = t
            .list_f32_flat("window_rt")
            .map_err(|_| need("window_rt"))?;
        // One block per candidate: a candidate's rows are contiguous, and a repeated
        // candidate (a second scored row of it) is read once.
        let mut run: Vec<CandidateRun> = Vec::new();
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        let mut i = 0usize;
        while i < t.nrows {
            let c = cid[i];
            let mut j = i;
            while j < t.nrows && cid[j] == c {
                j += 1;
            }
            if seen.insert(c) {
                let mut areas: Vec<(u32, f64)> = Vec::with_capacity(j - i);
                let mut frags: BTreeSet<u32> = BTreeSet::new();
                for r in i..j {
                    let next = names.len() as u32;
                    let id = *names.entry(name[r].clone()).or_insert(next);
                    if !frags.insert(id) {
                        continue;
                    }
                    areas.push((id, area[r]));
                    if obs[r] {
                        observed.entry(c).or_default().insert(name[r].clone());
                    }
                }
                observed.entry(c).or_default();
                let rt = &vals[off[i]..off[i + 1]];
                for w in rt.windows(2) {
                    spacings.push(w[1] as f64 - w[0] as f64);
                }
                run.push(CandidateRun { cid: c, areas });
            }
            i = j;
        }
        runs.push(run);
    }
    let dt = median(&mut spacings);
    let Some((phi, n_cells, n_cells_eligible)) = dispersion_label_free(&runs) else {
        return Ok(None);
    };
    if !(dt.is_finite() && dt > 0.0) {
        return Ok(None);
    }
    Ok(Some(PatternFit {
        phi,
        dt,
        observed,
        n_cells,
        n_cells_eligible,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_median_and_quantile() {
        assert!(median(&mut []).is_nan());
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
        let v = [1.0, 2.0, 4.0, 8.0, 16.0];
        // numpy.quantile([1, 2, 4, 8, 16], [0, 0.2, 0.6, 0.95, 1])
        assert_eq!(quantile_sorted(&v, 0.0), 1.0);
        assert_eq!(quantile_sorted(&v, 0.2), 1.8);
        assert!((quantile_sorted(&v, 0.6) - 5.6).abs() < 1e-15);
        assert!((quantile_sorted(&v, 0.95) - 14.4).abs() < 1e-12);
        assert_eq!(quantile_sorted(&v, 1.0), 16.0);
        assert_eq!(quantile_sorted(&[7.0], 0.3), 7.0);
    }

    #[test]
    fn the_profile_shift_prefers_the_smaller_shift_on_a_tie() {
        // Flat: no shift.
        assert_eq!(profile_shift(&[1.0; 7]), 0);
        // All zero: no shift.
        assert_eq!(profile_shift(&[0.0; 7]), 0);
        // Peak one sample to the right of k.
        assert_eq!(profile_shift(&[0.0, 0.0, 1.0, 2.0, 5.0, 2.0, 0.0]), 1);
        // Peak two to the left.
        assert_eq!(profile_shift(&[3.0, 9.0, 3.0, 1.0, 0.0, 0.0, 0.0]), -2);
        // -2 and +2 tie: -2 is tried first and +2 does not beat it.
        assert_eq!(profile_shift(&[9.0, 9.0, 9.0, 0.0, 9.0, 9.0, 9.0]), -2);
        // A symmetric profile around k keeps k.
        assert_eq!(profile_shift(&[0.0, 1.0, 4.0, 1.0, 4.0, 1.0, 0.0]), 0);
        // A gain inside the 1e-9 margin does not move the window.
        let mut p = [1.0; 7];
        p[5] = 1.0 + 1e-12;
        assert_eq!(profile_shift(&p), 0);
    }

    #[test]
    fn top_predicted_slots_ranks_by_intensity_and_returns_slot_order() {
        let preds = [1.0, 5.0, f32::NAN, 5.0, 3.0, 0.5, 2.0, 4.0];
        assert_eq!(top_predicted_slots(&preds, 3), vec![1, 3, 7]);
        assert_eq!(top_predicted_slots(&preds, 6), vec![0, 1, 3, 4, 6, 7]);
        assert_eq!(top_predicted_slots(&preds[..2], 6), vec![0, 1]);
    }

    /// Runs that do not vary give no dispersion, and the rules fall back; one run that
    /// differs gives a positive one; two runs give none.
    #[test]
    fn the_dispersion_needs_three_runs_that_vary() {
        let cand = |scale: f64| CandidateRun {
            cid: 1,
            areas: vec![(0, 100.0 * scale), (1, 50.0), (2, 25.0), (3, 10.0)],
        };
        let same: Vec<Vec<CandidateRun>> = (0..3).map(|_| vec![cand(1.0)]).collect();
        assert!(dispersion_label_free(&same).is_none());
        let vary: Vec<Vec<CandidateRun>> = [1.0, 1.2, 0.9].iter().map(|&k| vec![cand(k)]).collect();
        let (phi, used, eligible) = dispersion_label_free(&vary).unwrap();
        assert!(
            phi > 0.0 && used >= 1 && eligible == 4,
            "{phi} {used} {eligible}"
        );
        assert!(dispersion_label_free(&vary[..2]).is_none());
    }

    #[test]
    fn trapezoid_weights_match_the_trapezoid_rule() {
        let w = trapezoid_weights(&[f64::NAN, 10.0, 11.0, 13.0, f64::NAN]);
        assert_eq!(w, vec![0.0, 0.5, 1.5, 1.0, 0.0]);
        let x = [0.0, 2.0, 4.0, 2.0, 0.0];
        // (2 + 4) / 2 * 1 + (4 + 2) / 2 * 2 = 9
        assert_eq!(weighted_area(&x, &w), 9.0);
        assert_eq!(
            trapezoid_weights(&[f64::NAN, 5.0, f64::NAN]),
            vec![0.0, 1.0, 0.0]
        );
    }
}
