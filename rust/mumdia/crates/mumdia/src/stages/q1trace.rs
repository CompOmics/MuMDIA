//! Cycle-level traces of scanning-quadrupole spectra (`extract.q1_traces`) and the separable
//! model of a candidate's RT x step patch (`features.q1_separable`), SCIEX ZT Scan DIA.
//!
//! **Cycle traces.** On a ZT Scan run a candidate's chromatogram grid holds every scan of the
//! isolation windows that cover its precursor: about five Q1 steps a cycle, 1.3 ms apart, then
//! a gap of about a second to the next cycle. Read as one series, the trace is a sawtooth.
//! Under `extract.q1_traces` extract writes every accepted candidate's traces once more with one
//! point per cycle (`chromatograms_q1.parquet`, beside the chromatograms, which stay as they
//! are): each fragment's samples of one cycle collapsed to the least-squares amplitude
//! `sum(w x) / sum(w^2)` under `w`, the fragment's expected Q1 profile at the step's target and
//! the precursor's calibrated expected centre (`CandQ1::expected`), the estimator of
//! `quant.q1_fragments` ([`super::q1quant::collapse_trace`]), at the retention time of the
//! cycle's representative step, the covering step whose target is nearest the expected centre
//! (a scan of the run). [`CycleSpans`] groups a grid into cycles and picks those steps;
//! [`ls_amplitude`] is the estimator. Read in place of the interleaved traces (the features, the
//! apex, the gate), the cycle traces measured fewer identifications (docs/37_ztscan.md, section
//! 15), so no stage reads them.
//!
//! **The separable model** ([`fit`]). On the per-fragment decision's patch (the seven cycles
//! around the apex, every step within the window's reach apart), the fragments of a precursor
//! are one elution profile times each fragment's own transmission:
//! `x[f, c, s] = A_f E_c T_f(s) + b_f`, with `T_f` the calibrated expected Q1 profile at the
//! decision's shared centre and width, `E` shared by the decision's fitted set (the fragments
//! that belong, or the present ones), `A_f >= 0` the fragment's amplitude and `b_f >= 0` a
//! floor (background). Fitted by alternating least squares from the set's consensus elution:
//! each fragment's `(A_f, b_f)` given `E`, then `E` given every `(A_f, b_f)`, [`SEP_ITERS`]
//! times, `E` scaled to a maximum of 1 after each round. Every library fragment then gets its
//! own `(A_f, b_f)` under the final `E`, its `R^2` and its interference share (the observed
//! intensity above its own component, over its observed intensity). `E` is the clean elution
//! profile, `A_f` an amplitude fitted against the other fragments' elution, and the residuals
//! what no single precursor explains.
//!
//! Deterministic: every sum runs in fragment, cycle and step order; ties keep the first.

use mumdia_core::config::Q1FragModel;

use super::q1frag::{CandQ1, Decision, Patch, ABSENT};

/// The cycles of a candidate's per-step grid (`extract.q1_traces`): consecutive grid samples
/// of one Q1 cycle form one span, and each span has a representative sample.
#[derive(Clone, Debug, PartialEq)]
pub struct CycleSpans {
    /// Per cycle, in grid order: the grid samples `start..end` it holds.
    pub spans: Vec<(usize, usize)>,
    /// Per cycle: the grid index of its representative sample, the one whose step target is
    /// nearest the expected Q1 centre (the first on a tie).
    pub rep: Vec<usize>,
}

impl CycleSpans {
    /// The cycles of a grid from each sample's `(cycle, step target)`, with the representative
    /// samples nearest `centre`. `None` for an empty grid, a sample outside the sweep
    /// (`u32::MAX`), or cycles that do not ascend along the grid (a cycle that comes back
    /// after another).
    pub fn of(place: &[(u32, f64)], centre: f64) -> Option<CycleSpans> {
        if place.is_empty() {
            return None;
        }
        let (mut spans, mut rep) = (Vec::new(), Vec::new());
        let mut last: Option<u32> = None;
        let mut i = 0usize;
        while i < place.len() {
            let c = place[i].0;
            if c == u32::MAX || last.is_some_and(|l| c <= l) {
                return None;
            }
            let a = i;
            let mut best = i;
            while i < place.len() && place[i].0 == c {
                if (place[i].1 - centre).abs() < (place[best].1 - centre).abs() {
                    best = i;
                }
                i += 1;
            }
            spans.push((a, i));
            rep.push(best);
            last = Some(c);
        }
        Some(CycleSpans { spans, rep })
    }

    /// Cycles.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

/// The least-squares amplitude of samples `x` under weights `w`, `sum(w x) / sum(w^2)`, summed
/// in order; 0 when every weight is 0.
pub fn ls_amplitude(xw: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (mut swx, mut sww) = (0.0f64, 0.0f64);
    for (x, w) in xw {
        swx += w * x;
        sww += w * w;
    }
    if sww > 0.0 {
        swx / sww
    } else {
        0.0
    }
}

/// Alternating least-squares rounds of [`fit`].
pub const SEP_ITERS: usize = 12;
/// The elution profile's share of its maximum that bounds the peak ([`SepFit::lo_ci`]), the
/// features stage's elution-peak fraction.
pub const SEP_BOUND_FRAC: f64 = 1.0 / 3.0;

/// The separable model of one candidate's patch ([`fit`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SepFit {
    /// Whether a shared profile was fitted (a set with signal); without, every value below is
    /// its empty form.
    pub has_fit: bool,
    /// The elution profile `E` over the patch's cycles, maximum 1.
    pub elution: Vec<f64>,
    /// Per library fragment: its amplitude `A_f` under `E` (intensity at `E = 1`, `T = 1`) and
    /// its floor `b_f`.
    pub amp: Vec<f64>,
    pub bg: Vec<f64>,
    /// Per library fragment: `1 - SSE / SST` of its cells under its own component (NaN for a
    /// fragment without varying signal), and the share of its observed intensity above its
    /// component (NaN without signal).
    pub r2: Vec<f64>,
    pub interf: Vec<f64>,
    /// Per library fragment: in the fitted set.
    pub fitted: Vec<bool>,
    /// `1 - SSE / SST` over the fitted set's cells together.
    pub r2_set: f64,
    /// The share of the present fragments' observed patch intensity the model's components
    /// explain (`sum min(x, A E T) / sum x`).
    pub explained: f64,
    /// The share of the present fragments' observed patch intensity the floors take.
    pub bg_share: f64,
    /// The apex of `E` in patch-cycle units, refined by a parabola.
    pub apex_ci: f64,
    /// The cycles `lo_ci..=hi_ci` around the apex where `E` stays at or above
    /// [`SEP_BOUND_FRAC`] of its maximum.
    pub lo_ci: usize,
    pub hi_ci: usize,
    /// Full width of `E` at half maximum, in cycles (linear interpolation; clipped at the
    /// patch's ends).
    pub fwhm: f64,
}

/// `(A, b)`, both at least 0, of the least-squares line `x = A m + b` from its sums over `n`
/// cells: `sm = sum m`, `smm = sum m^2`, `sx = sum x`, `smx = sum m x`. Without a floor when
/// the free solution is negative in either.
fn amp_floor(n: f64, sm: f64, smm: f64, sx: f64, smx: f64) -> (f64, f64) {
    let det = smm * n - sm * sm;
    if det.abs() > 1e-12 * (smm * n).max(1e-300) {
        let a = (smx * n - sm * sx) / det;
        let b = (smm * sx - sm * smx) / det;
        if a >= 0.0 && b >= 0.0 {
            return (a, b);
        }
    }
    let a = if smm > 0.0 { (smx / smm).max(0.0) } else { 0.0 };
    (a, 0.0)
}

/// The separable model of `cand` on its decision patch `p` under the decision `d` and the
/// model `m` (module docs). `T_f` is the expected profile at the decision's shared centre and
/// width. The fitted set is the decision's own (`Decision::in_set`) among the present
/// fragments, or every present fragment when that leaves fewer than two; no fit without a
/// fragment with signal.
pub fn fit(p: &Patch, d: &Decision, cand: &CandQ1, m: &Q1FragModel) -> SepFit {
    let (nf, nc, ns) = (p.nf, p.cycles.len(), p.ns());
    let empty = SepFit {
        has_fit: false,
        elution: vec![0.0; nc],
        amp: vec![0.0; nf],
        bg: vec![0.0; nf],
        r2: vec![f64::NAN; nf],
        interf: vec![f64::NAN; nf],
        fitted: vec![false; nf],
        r2_set: 0.0,
        explained: 0.0,
        bg_share: 0.0,
        apex_ci: 0.0,
        lo_ci: 0,
        hi_ci: 0,
        fwhm: 0.0,
    };
    if nf == 0 || nc == 0 || ns == 0 || d.class.len() != nf {
        return empty;
    }
    let step = if d.step > 0.0 { d.step } else { m.step_mz };
    let width = if d.width > 0.0 { d.width } else { 1.0 };
    // Each fragment's expected profile over the patch's steps at the shared centre and width.
    let t: Vec<Vec<f64>> = (0..nf)
        .map(|f| {
            p.t.iter()
                .map(|&tg| cand.expected_at(m, f, tg, d.centre, step, width))
                .collect()
        })
        .collect();
    let tot: Vec<f64> = (0..nf)
        .map(|f| {
            (0..nc)
                .map(|ci| (0..ns).map(|si| p.get(f, ci, si)).sum::<f64>())
                .sum()
        })
        .collect();
    let present: Vec<bool> = (0..nf).map(|f| d.class[f] != ABSENT).collect();
    let usable = |f: usize| present[f] && tot[f] > 0.0 && t[f].iter().any(|&v| v > 0.0);
    let mut set: Vec<bool> = (0..nf)
        .map(|f| usable(f) && d.in_set.get(f).copied().unwrap_or(false))
        .collect();
    if set.iter().filter(|&&x| x).count() < 2 {
        set = (0..nf).map(usable).collect();
    }
    if !set.iter().any(|&x| x) {
        return empty;
    }
    // The set's consensus elution, each fragment normalised to its own total.
    let mut e: Vec<f64> = (0..nc)
        .map(|ci| {
            (0..nf)
                .filter(|&f| set[f])
                .map(|f| (0..ns).map(|si| p.get(f, ci, si)).sum::<f64>() / tot[f])
                .sum()
        })
        .collect();
    let emax = e.iter().copied().fold(0.0f64, f64::max);
    if emax <= 0.0 {
        return empty;
    }
    e.iter_mut().for_each(|v| *v /= emax);
    let n_cells = (nc * ns) as f64;
    // `(A_f, b_f)` of fragment `f` under the profile `e`.
    let regress = |f: usize, e: &[f64]| -> (f64, f64) {
        let (mut sm, mut smm, mut sx, mut smx) = (0.0, 0.0, 0.0, 0.0);
        for (ci, &ec) in e.iter().enumerate() {
            for (si, &tv) in t[f].iter().enumerate() {
                let mv = ec * tv;
                let x = p.get(f, ci, si);
                sm += mv;
                smm += mv * mv;
                sx += x;
                smx += mv * x;
            }
        }
        amp_floor(n_cells, sm, smm, sx, smx)
    };
    let mut amp = vec![0.0f64; nf];
    let mut bg = vec![0.0f64; nf];
    for _ in 0..SEP_ITERS {
        for f in 0..nf {
            if set[f] {
                let (a, b) = regress(f, &e);
                amp[f] = a;
                bg[f] = b;
            }
        }
        // `E` given the components.
        let mut next = vec![0.0f64; nc];
        let mut moved = false;
        for (ci, nv) in next.iter_mut().enumerate() {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for f in 0..nf {
                if !set[f] || amp[f] <= 0.0 {
                    continue;
                }
                for (si, &tv) in t[f].iter().enumerate() {
                    let w = amp[f] * tv;
                    num += w * (p.get(f, ci, si) - bg[f]);
                    den += w * w;
                }
            }
            *nv = if den > 0.0 {
                (num / den).max(0.0)
            } else {
                e[ci]
            };
            moved |= nv.to_bits() != e[ci].to_bits();
        }
        let nmax = next.iter().copied().fold(0.0f64, f64::max);
        if nmax <= 0.0 {
            break;
        }
        next.iter_mut().for_each(|v| *v /= nmax);
        e = next;
        if !moved {
            break;
        }
    }
    // Every fragment's component under the final profile, and its residuals.
    let mut r2 = vec![f64::NAN; nf];
    let mut interf = vec![f64::NAN; nf];
    let (mut sse_set, mut sst_set) = (0.0f64, 0.0f64);
    let (mut expl, mut bgm, mut xtot) = (0.0f64, 0.0f64, 0.0f64);
    for f in 0..nf {
        let (a, b) = regress(f, &e);
        amp[f] = a;
        bg[f] = b;
        if tot[f] <= 0.0 {
            continue;
        }
        let mean = tot[f] / n_cells;
        let (mut sse, mut sst, mut above, mut ex) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (ci, &ec) in e.iter().enumerate() {
            for (si, &tv) in t[f].iter().enumerate() {
                let x = p.get(f, ci, si);
                let comp = a * ec * tv;
                let r = x - comp - b;
                sse += r * r;
                sst += (x - mean) * (x - mean);
                above += r.max(0.0);
                ex += x.min(comp);
            }
        }
        r2[f] = if sst > 0.0 { 1.0 - sse / sst } else { f64::NAN };
        interf[f] = (above / tot[f]).clamp(0.0, 1.0);
        if set[f] {
            sse_set += sse;
            sst_set += sst;
        }
        if present[f] {
            expl += ex;
            bgm += (b * n_cells).min(tot[f]);
            xtot += tot[f];
        }
    }
    // The apex of `E`, refined by a parabola through its neighbours, and its bounds.
    let mut ai = 0usize;
    for (ci, &v) in e.iter().enumerate() {
        if v > e[ai] {
            ai = ci;
        }
    }
    let apex_ci = if ai > 0 && ai + 1 < nc {
        let (l, c, r) = (e[ai - 1], e[ai], e[ai + 1]);
        let den = l - 2.0 * c + r;
        if den < 0.0 {
            ai as f64 + (0.5 * (l - r) / den).clamp(-0.5, 0.5)
        } else {
            ai as f64
        }
    } else {
        ai as f64
    };
    let (mut lo, mut hi) = (ai, ai);
    while lo > 0 && e[lo - 1] >= SEP_BOUND_FRAC {
        lo -= 1;
    }
    while hi + 1 < nc && e[hi + 1] >= SEP_BOUND_FRAC {
        hi += 1;
    }
    // Half maximum, interpolated between the last cycle at or above it and the next.
    let mut left = ai as f64;
    let mut k = ai;
    while k > 0 && e[k - 1] >= 0.5 {
        k -= 1;
    }
    left = if k > 0 {
        let (a, b) = (e[k - 1], e[k]);
        k as f64 - if b > a { (b - 0.5) / (b - a) } else { 0.0 }
    } else {
        left.min(0.0)
    };
    let mut right = ai as f64;
    let mut k = ai;
    while k + 1 < nc && e[k + 1] >= 0.5 {
        k += 1;
    }
    right = if k + 1 < nc {
        let (a, b) = (e[k], e[k + 1]);
        k as f64 + if a > b { (a - 0.5) / (a - b) } else { 0.0 }
    } else {
        right.max((nc - 1) as f64)
    };
    SepFit {
        has_fit: true,
        elution: e,
        amp,
        bg,
        r2,
        interf,
        fitted: set,
        r2_set: if sst_set > 0.0 {
            1.0 - sse_set / sst_set
        } else {
            0.0
        },
        explained: if xtot > 0.0 { expl / xtot } else { 0.0 },
        bg_share: if xtot > 0.0 { bgm / xtot } else { 0.0 },
        apex_ci,
        lo_ci: lo,
        hi_ci: hi,
        fwhm: (right - left).max(0.0),
    }
}

/// The feature columns of the separable model (`features.q1_separable`, psms_extracted v9).
pub const Q1S_COLUMNS: &[&str] = &[
    "q1s_r2",
    "q1s_r2_pred",
    "q1s_interf_pred",
    "q1s_explained",
    "q1s_bg_share",
    "q1s_amp_cos",
    "q1s_amp_corr",
    "q1s_n_fit",
    "q1s_fwhm",
    "q1s_apex_offset",
];

/// The [`Q1S_COLUMNS`] values of a row without a decision patch (no evidence).
pub fn empty_values() -> Vec<f64> {
    vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

/// The [`Q1S_COLUMNS`] values of a fit: the set's `R^2`; the predicted-intensity-weighted mean
/// `R^2` and interference share of the present fragments; the explained and floor shares; the
/// cosine of the square-rooted amplitudes with the square-rooted predictions and their Pearson
/// correlation, over every library fragment (an absent one at amplitude 0); the fitted
/// fragments; the profile's width; and the distance of its apex from the row's apex cycle
/// `apex_ci` (patch-cycle units). An empty fit reads as no evidence (0, the widths 0, the
/// offset the patch's half-length).
pub fn values(s: &SepFit, d: &Decision, cand: &CandQ1, row_apex_ci: usize) -> Vec<f64> {
    let nf = cand.frags.len();
    if !s.has_fit || s.amp.len() != nf {
        return vec![
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            s.elution.len() as f64 / 2.0,
        ];
    }
    let pred: Vec<f64> = cand.frags.iter().map(|f| f.pred.max(0.0)).collect();
    let (mut wr, mut wi, mut ws) = (0.0f64, 0.0f64, 0.0f64);
    for (f, &w) in pred.iter().enumerate() {
        if d.class.get(f).copied().unwrap_or(ABSENT) == ABSENT {
            continue;
        }
        if s.r2[f].is_finite() {
            wr += w * s.r2[f].max(-1.0);
        }
        if s.interf[f].is_finite() {
            wi += w * s.interf[f];
        }
        ws += w;
    }
    let amp: Vec<f64> = (0..nf)
        .map(|f| {
            if d.class.get(f).copied().unwrap_or(ABSENT) == ABSENT {
                0.0
            } else {
                s.amp[f].max(0.0)
            }
        })
        .collect();
    let sa: Vec<f64> = amp.iter().map(|v| v.sqrt()).collect();
    let sp: Vec<f64> = pred.iter().map(|v| v.sqrt()).collect();
    let cos = crate::stats::cosine(&sa, &sp);
    let corr = crate::stats::pearson(&amp, &pred);
    let fin = |v: f64| if v.is_finite() { v } else { 0.0 };
    vec![
        fin(s.r2_set),
        if ws > 0.0 { wr / ws } else { 0.0 },
        if ws > 0.0 { wi / ws } else { 1.0 },
        fin(s.explained),
        fin(s.bg_share),
        fin(cos),
        fin(corr),
        s.fitted.iter().filter(|&&x| x).count() as f64,
        s.fwhm,
        (s.apex_ci - row_apex_ci as f64).abs(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::q1frag::{self, BELONGS, INTERFERED};
    use mumdia_core::config::{Q1FragTolerances, Q1OffsetTerm};

    fn model() -> Q1FragModel {
        Q1FragModel {
            step_mz: 0.7614,
            half_width_mz: 2.57,
            edge_width_mz: 4.0,
            offset_terms: vec![Q1OffsetTerm {
                var: "precursor_mz".into(),
                grid: vec![400.0, 900.0],
                offset_mz: vec![0.0, 0.0],
            }],
            tolerances: Q1FragTolerances {
                offset_steps: 1.1,
                centre_steps: 0.7,
                corr2d_min: 0.5,
            },
        }
    }

    fn cand(m: &Q1FragModel, n: usize) -> CandQ1 {
        let mzs: Vec<f64> = (0..n).map(|i| 300.0 + 50.0 * i as f64).collect();
        let zs = vec![1; n];
        let preds: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
        CandQ1::new(m, 600.0, 2, 1000.0, &mzs, &zs, &preds, &|_| 1.0)
    }

    /// A patch whose fragment `f` is `amp[f] * g(c) * T_f(s)` (+ `bg`), `g` a Gaussian in
    /// cycles centred at `c0`, the steps 0.7614 apart around the candidate's centre.
    fn patch(m: &Q1FragModel, cand: &CandQ1, amp: &[f64], c0: f64, bg: f64) -> Patch {
        let ns = 17;
        let t: Vec<f64> = (0..ns)
            .map(|k| cand.centre + 0.7614 * (k as f64 - 8.0))
            .collect();
        let mut p = Patch::new((10..17).collect(), 0, t.clone(), amp.len());
        for (f, &a) in amp.iter().enumerate() {
            for ci in 0..7 {
                let g = (-0.5 * ((ci as f64 - c0) / 1.2).powi(2)).exp();
                for (si, &tg) in t.iter().enumerate() {
                    let v = a * g * cand.expected(m, f, tg, 0.0, 1.0) + bg;
                    p.set(f, ci, si, v as f32);
                }
            }
        }
        p
    }

    fn decision(nf: usize, class: u8) -> Decision {
        Decision {
            class: vec![class; nf],
            offset: vec![0.0; nf],
            resid: vec![0.0; nf],
            corr2d: vec![1.0; nf],
            tot: vec![1.0; nf],
            has_q: vec![true; nf],
            in_set: vec![true; nf],
            centre: 0.0,
            width: 1.0,
            has_fit: true,
            apex_ci: 3,
            step: 0.7614,
        }
    }

    #[test]
    fn cycles_are_spans_of_one_cycle_with_the_step_nearest_the_centre() {
        let place = [
            (5, 599.0),
            (5, 599.8),
            (5, 600.5),
            (6, 599.0),
            (6, 599.8),
            (6, 600.5),
            (7, 600.5),
        ];
        let c = CycleSpans::of(&place, 600.3).expect("cycles");
        assert_eq!(c.spans, vec![(0, 3), (3, 6), (6, 7)]);
        assert_eq!(c.rep, vec![2, 5, 6]);
        assert_eq!(c.len(), 3);
        // Outside the sweep, or a cycle that comes back: no cycles.
        assert!(CycleSpans::of(&[(5, 1.0), (u32::MAX, 1.0)], 1.0).is_none());
        assert!(CycleSpans::of(&[(5, 1.0), (6, 1.0), (5, 1.0)], 1.0).is_none());
        assert!(CycleSpans::of(&[], 1.0).is_none());
    }

    #[test]
    fn the_amplitude_is_the_least_squares_one() {
        // x = 3 w exactly.
        let x = [1.5, 3.0, 2.4];
        let w = [0.5, 1.0, 0.8];
        let a = ls_amplitude(x.iter().copied().zip(w.iter().copied()));
        assert!((a - 3.0).abs() < 1e-12);
        assert_eq!(ls_amplitude([(1.0, 0.0)].into_iter()), 0.0);
    }

    #[test]
    fn a_separable_patch_is_recovered() {
        let m = model();
        let c = cand(&m, 4);
        let amp = [100.0, 40.0, 250.0, 10.0];
        let p = patch(&m, &c, &amp, 3.0, 0.0);
        let d = decision(4, BELONGS);
        let s = fit(&p, &d, &c, &m);
        assert!(s.has_fit);
        // E peaks at the apex and is scaled to 1 there; the amplitudes are the planted ones.
        assert!((s.elution[3] - 1.0).abs() < 1e-6, "{:?}", s.elution);
        for (f, &want) in amp.iter().enumerate() {
            assert!((s.amp[f] - want).abs() / want < 1e-3, "{f}: {:?}", s.amp);
            assert!(s.r2[f] > 0.999, "{:?}", s.r2);
            assert!(s.interf[f] < 1e-3, "{:?}", s.interf);
        }
        assert!(s.r2_set > 0.999);
        assert!((s.apex_ci - 3.0).abs() < 1e-6);
        assert!(s.explained > 0.999);
        // Width at half maximum of a Gaussian of sigma 1.2 cycles: 2.83, sampled.
        assert!(s.fwhm > 2.0 && s.fwhm < 3.5, "{}", s.fwhm);
        assert!(s.lo_ci <= 2 && s.hi_ci >= 4);
        let v = values(&s, &d, &c, 3);
        assert_eq!(v.len(), Q1S_COLUMNS.len());
        assert!(v[0] > 0.999 && v[1] > 0.999 && v[2] < 1e-3);
        assert!(v[9] < 1e-6);
    }

    #[test]
    fn an_interferer_off_in_rt_is_read_as_interference() {
        let m = model();
        let c = cand(&m, 4);
        let mut p = patch(&m, &c, &[100.0, 100.0, 100.0, 100.0], 3.0, 0.0);
        // Fragment 3 also carries a stronger peak of another precursor two cycles later.
        for si in 0..17 {
            let tg = p.t[si];
            let v = p.get(3, 5, si) + 300.0 * c.expected(&m, 3, tg, 0.0, 1.0);
            p.set(3, 5, si, v as f32);
        }
        let mut d = decision(4, BELONGS);
        d.class[3] = INTERFERED;
        d.in_set[3] = false;
        let s = fit(&p, &d, &c, &m);
        assert!(s.has_fit);
        assert!(!s.fitted[3] && s.fitted[0]);
        assert!(s.interf[3] > 0.3, "{:?}", s.interf);
        assert!(s.interf[0] < 1e-3);
        assert!(s.r2[3] < s.r2[0]);
        // The profile is the set's: its apex stays at cycle 3.
        assert!((s.apex_ci - 3.0).abs() < 0.1, "{}", s.apex_ci);
    }

    #[test]
    fn a_floor_is_taken_as_background() {
        let m = model();
        let c = cand(&m, 3);
        let p = patch(&m, &c, &[100.0, 50.0, 80.0], 3.0, 5.0);
        let d = decision(3, BELONGS);
        let s = fit(&p, &d, &c, &m);
        for f in 0..3 {
            assert!((s.bg[f] - 5.0).abs() < 0.05, "{:?}", s.bg);
        }
        assert!(s.bg_share > 0.1);
    }

    #[test]
    fn no_signal_is_no_fit() {
        let m = model();
        let c = cand(&m, 3);
        let p = Patch::new((0..7).collect(), 0, vec![600.0; 17], 3);
        let d = decision(3, BELONGS);
        let s = fit(&p, &d, &c, &m);
        assert!(!s.has_fit);
        let v = values(&s, &d, &c, 3);
        assert_eq!(v.len(), Q1S_COLUMNS.len());
        assert_eq!(v[0], 0.0);
        // Deterministic.
        let p2 = patch(&m, &c, &[1.0, 2.0, 3.0], 2.0, 0.5);
        assert_eq!(fit(&p2, &d, &c, &m), fit(&p2, &d, &c, &m));
        let _ = q1frag::ABSENT;
    }
}
