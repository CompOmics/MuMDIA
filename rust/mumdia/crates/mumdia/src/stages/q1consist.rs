//! The Q1-consistency family of scanning-quadrupole spectra (`features.q1_consistency`).
//!
//! `extract.q1_fragments` decides per library fragment whether it belongs to the candidate's
//! precursor ([`crate::stages::q1frag`]): a fragment belongs when its own Q1 centre lies within
//! the offset tolerance of the centre SHARED by the candidate's fragments and its 2D
//! correlation reaches the floor. The shared centre is a least-squares fit over the fragments
//! that are present, so when interfering fragments dominate the fit it moves towards the
//! interferer, the 2D correlations of the candidate's own fragments fall, and none belongs.
//! The `q1f_*` summary then reads 0 for a precursor whose strongest fragments sit exactly
//! where its m/z says they should. On the ZT Scan K562 runs that is 60% of the precursors the
//! classifier scored just above 1% while their peak was the one another engine accepted.
//!
//! This family reads the same per-fragment decisions against the EXPECTED centre instead:
//! a fragment is consistent when it is present and its own Q1 centre (`q1f_frag_offset`,
//! steps from its expected centre) lies within the learned offset tolerance of 0, whatever
//! the shared fit did. It then scores the candidate's observed fragment intensities restricted
//! to the consistent fragments against the library: how much of the predicted and of the
//! observed intensity they carry, how well their pattern agrees with the prediction, and how
//! far from the expected centre the fragments sit, graded rather than cut.
//!
//! Inputs, all in library order: the decision lists of psms_extracted v6 (`q1f_frag_class`,
//! `q1f_frag_offset`, `q1f_frag_corr2d`), the predicted intensities and each fragment's
//! observed intensity summed over the candidate's elution peak (the features stage's own
//! window). Nothing reads a label.
//!
//! - `q1c_n`: consistent fragments.
//! - `q1c_n_corr`: consistent fragments whose 2D correlation also reaches the floor.
//! - `q1c_pred_share`, `q1c_obs_share`: their share of the predicted intensity and of the
//!   observed intensity of all library fragments.
//! - `q1c_entropy`, `q1c_cosine`: spectral-entropy similarity and cosine of the square-rooted
//!   observed intensities, the inconsistent fragments set to 0, against the square-rooted
//!   predictions of every library fragment (a missing strong fragment costs).
//! - `q1c_top3`: how many of the three most intense predicted fragments are consistent.
//! - `q1c_entropy_corr`: `q1c_entropy` over the consistent fragments that reach the 2D
//!   correlation floor.
//! - `q1c_soft_pred`, `q1c_soft_obs`, `q1c_soft_entropy`: the predicted and observed shares
//!   and the entropy similarity with each present fragment weighted by a Gaussian in its
//!   offset (SD a third of the tolerance) instead of the hard cut.
//! - `q1c_n_tight`, `q1c_n_loose`: present fragments within half and within twice the offset
//!   tolerance.
//! - `q1c_corr2d_mean`: the mean 2D correlation of the consistent fragments.
//! - `q1c_offset_wmean`: the predicted-intensity-weighted mean |offset| of the present
//!   fragments (capped at 6 steps; 6 without any); `q1c_offset_top3`: the mean |offset| of
//!   the three most intense predicted fragments (6 for one absent or without a profile).
//!
//! [`ALL`] also computes scores the classifier is not given ([`NAMES`] selects): the cosine
//! within the consistent fragments, their log intensity, whether the most intense predicted
//! fragment is consistent, and the entropy, cosine, cosine within, observed share and log
//! intensity over the fragments that belong (`q1f_frag_class` = 1). On the ZT Scan runs they
//! added nothing to the selected scores.
//!
//! Every value is finite; a candidate without decisions or without observed signal reads 0
//! (a list whose length differs from the library fragments reads as no decision).
//! Deterministic: fragments in library order, sums in that order, ties of the predicted
//! ranking to the earlier fragment.

use mumdia_core::config::Q1FragTolerances;

use super::features::entropy::spectral_entropy_similarity_sqrt;
use crate::stats::cosine;

/// The family's feature names, in column order: the scores the classifier is given, a subset
/// of [`ALL`] (every score [`compute`] returns), chosen on the ZT Scan K562 runs
/// (docs/37_ztscan.md, section 5).
pub const NAMES: &[&str] = &[
    "q1c_n",
    "q1c_n_corr",
    "q1c_pred_share",
    "q1c_obs_share",
    "q1c_entropy",
    "q1c_cosine",
    "q1c_top3",
    "q1c_entropy_corr",
    "q1c_soft_pred",
    "q1c_soft_obs",
    "q1c_soft_entropy",
    "q1c_n_tight",
    "q1c_n_loose",
    "q1c_corr2d_mean",
    "q1c_offset_wmean",
    "q1c_offset_top3",
];

/// Every score the family computes, in [`compute`]'s order.
pub const ALL: [&str; 24] = [
    "q1c_n",
    "q1c_n_corr",
    "q1c_pred_share",
    "q1c_obs_share",
    "q1c_entropy",
    "q1c_cosine",
    "q1c_cosine_within",
    "q1c_log_intensity",
    "q1c_top1",
    "q1c_top3",
    "q1c_entropy_corr",
    "q1c_b_entropy",
    "q1c_b_cosine",
    "q1c_b_cosine_within",
    "q1c_b_obs_share",
    "q1c_b_log_intensity",
    "q1c_soft_pred",
    "q1c_soft_obs",
    "q1c_soft_entropy",
    "q1c_n_tight",
    "q1c_n_loose",
    "q1c_corr2d_mean",
    "q1c_offset_wmean",
    "q1c_offset_top3",
];

/// The psms_extracted v6 lists the family reads, in the order [`Input`] takes them.
pub const LISTS: [&str; 3] = ["q1f_frag_class", "q1f_frag_offset", "q1f_frag_corr2d"];

/// The class code of a fragment that belongs (`q1frag::BELONGS`).
const BELONGS: f32 = 1.0;

/// The |offset| (steps) that stands for "no Q1 profile" and the cap of every offset read:
/// the decision's centre grid spans +-6 steps.
const OFFSET_CAP: f64 = 6.0;

/// One candidate's inputs, every slice in library order.
pub struct Input<'a> {
    /// `q1f_frag_class`: 0 absent, 1 belongs, 2 interfered.
    pub class: &'a [f32],
    /// `q1f_frag_offset`: the fragment's own Q1 centre against its expected one, steps;
    /// NaN without a Q1 profile.
    pub offset: &'a [f32],
    /// `q1f_frag_corr2d`: its 2D correlation; NaN for an absent fragment.
    pub corr2d: &'a [f32],
    /// Predicted intensities (negative values count as 0).
    pub pred: &'a [f64],
    /// Observed intensities, summed over the elution peak.
    pub obs: &'a [f64],
}

impl Input<'_> {
    /// Whether every list has one entry per library fragment; otherwise [`values`] reads the
    /// candidate as having no decision.
    pub fn lengths_agree(&self) -> bool {
        let n = self.pred.len();
        self.class.len() == n
            && self.offset.len() == n
            && self.corr2d.len() == n
            && self.obs.len() == n
    }
}

#[inline]
fn fin(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// The cosine of `sqrt(obs)` against `sqrt(pred)` over the fragments `keep` selects, or over
/// every fragment with the observed intensity of the others set to 0 (`within = false`).
fn sqrt_cosine(obs: &[f64], pred: &[f64], keep: &[bool], within: bool) -> f64 {
    let n = obs.len();
    let mut o = Vec::with_capacity(n);
    let mut l = Vec::with_capacity(n);
    for f in 0..n {
        if within && !keep[f] {
            continue;
        }
        let ob = if keep[f] { obs[f].max(0.0) } else { 0.0 };
        o.push(ob.sqrt());
        l.push(pred[f].max(0.0).sqrt());
    }
    if within && o.len() < 2 {
        return 0.0;
    }
    cosine(&o, &l)
}

/// The scores of one fragment set: entropy and cosine against every prediction, the cosine
/// within the set, its observed share and its log intensity.
fn set_scores(obs: &[f64], pred: &[f64], keep: &[bool], obs_tot: f64) -> [f64; 5] {
    let masked: Vec<f64> = obs
        .iter()
        .zip(keep)
        .map(|(o, k)| if *k { o.max(0.0) } else { 0.0 })
        .collect();
    let kept: f64 = masked.iter().sum();
    [
        spectral_entropy_similarity_sqrt(&masked, pred),
        sqrt_cosine(obs, pred, keep, false),
        sqrt_cosine(obs, pred, keep, true),
        if obs_tot > 0.0 { kept / obs_tot } else { 0.0 },
        (1.0 + kept).ln(),
    ]
}

/// Every score of [`ALL`] for one candidate; all 0 without a decision.
pub fn compute(inp: &Input, tol: &Q1FragTolerances) -> [f64; 24] {
    let n = inp.pred.len();
    let mut v = [0.0; 24];
    if n == 0 || !inp.lengths_agree() {
        return v;
    }
    let pred: Vec<f64> = inp.pred.iter().map(|p| p.max(0.0)).collect();
    let present: Vec<bool> = inp.class.iter().map(|c| *c > 0.0).collect();
    let has_q: Vec<bool> = inp.offset.iter().map(|o| o.is_finite()).collect();
    // |own centre - expected centre|, capped; a fragment without a profile reads as far as
    // the grid goes.
    let off: Vec<f64> = (0..n)
        .map(|f| {
            if has_q[f] {
                (inp.offset[f] as f64).abs().min(OFFSET_CAP)
            } else {
                OFFSET_CAP
            }
        })
        .collect();
    let within = |f: usize, t: f64| present[f] && has_q[f] && off[f] <= t;
    let consistent: Vec<bool> = (0..n).map(|f| within(f, tol.offset_steps)).collect();
    let with_corr: Vec<bool> = (0..n)
        .map(|f| {
            consistent[f] && inp.corr2d[f].is_finite() && inp.corr2d[f] as f64 >= tol.corr2d_min
        })
        .collect();
    let belongs: Vec<bool> = inp.class.iter().map(|c| *c == BELONGS).collect();
    let pred_tot: f64 = pred.iter().sum();
    let obs_tot: f64 = inp.obs.iter().map(|o| o.max(0.0)).sum();
    // The fragments in descending predicted intensity, ties to the earlier one.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| pred[b].total_cmp(&pred[a]).then(a.cmp(&b)));
    // A soft consistency weight: a Gaussian in the offset whose SD is a third of the
    // tolerance (the tolerance is the median plus three robust SDs of the seed anchors').
    let sig = (tol.offset_steps / 3.0).max(1e-6);
    let soft: Vec<f64> = (0..n)
        .map(|f| {
            if present[f] && has_q[f] {
                (-0.5 * (off[f] / sig).powi(2)).exp()
            } else {
                0.0
            }
        })
        .collect();

    let c = set_scores(inp.obs, &pred, &consistent, obs_tot);
    let b = set_scores(inp.obs, &pred, &belongs, obs_tot);
    let weighted = |w: &dyn Fn(usize) -> f64| -> Vec<f64> {
        (0..n).map(|f| inp.obs[f].max(0.0) * w(f)).collect()
    };
    let corr_obs = weighted(&|f| if with_corr[f] { 1.0 } else { 0.0 });
    let soft_obs = weighted(&|f| soft[f]);
    let share = |num: f64, den: f64| if den > 0.0 { num / den } else { 0.0 };
    let count = |s: &[bool]| s.iter().filter(|x| **x).count() as f64;
    let pred_c: f64 = (0..n).filter(|&f| consistent[f]).map(|f| pred[f]).sum();
    let pred_soft: f64 = (0..n).map(|f| pred[f] * soft[f]).sum();
    let corr_c: Vec<f64> = (0..n)
        .filter(|&f| consistent[f] && inp.corr2d[f].is_finite())
        .map(|f| inp.corr2d[f] as f64)
        .collect();
    let pred_present: f64 = (0..n).filter(|&f| present[f]).map(|f| pred[f]).sum();
    let off_w: f64 = (0..n)
        .filter(|&f| present[f])
        .map(|f| pred[f] * off[f])
        .sum();
    let top3_off: f64 = (0..3)
        .map(|k| match order.get(k) {
            Some(&f) if present[f] => off[f],
            _ => OFFSET_CAP,
        })
        .sum::<f64>()
        / 3.0;

    v[0] = count(&consistent);
    v[1] = count(&with_corr);
    v[2] = share(pred_c, pred_tot);
    v[3] = c[3];
    v[4] = c[0];
    v[5] = c[1];
    v[6] = c[2];
    v[7] = c[4];
    v[8] = if consistent[order[0]] { 1.0 } else { 0.0 };
    v[9] = order.iter().take(3).filter(|&&f| consistent[f]).count() as f64;
    v[10] = spectral_entropy_similarity_sqrt(&corr_obs, &pred);
    v[11] = b[0];
    v[12] = b[1];
    v[13] = b[2];
    v[14] = b[3];
    v[15] = b[4];
    v[16] = share(pred_soft, pred_tot);
    v[17] = share(soft_obs.iter().sum(), obs_tot);
    v[18] = spectral_entropy_similarity_sqrt(&soft_obs, &pred);
    v[19] = (0..n)
        .filter(|&f| within(f, tol.offset_steps / 2.0))
        .count() as f64;
    v[20] = (0..n)
        .filter(|&f| within(f, 2.0 * tol.offset_steps))
        .count() as f64;
    v[21] = if corr_c.is_empty() {
        0.0
    } else {
        corr_c.iter().sum::<f64>() / corr_c.len() as f64
    };
    v[22] = if pred_present > 0.0 {
        off_w / pred_present
    } else {
        OFFSET_CAP
    };
    v[23] = top3_off;
    v.iter_mut().for_each(|x| *x = fin(*x));
    v
}

/// The position in [`ALL`] of each of [`NAMES`], taken once.
fn selection() -> &'static [usize] {
    static SEL: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| {
        NAMES
            .iter()
            .map(|n| {
                ALL.iter()
                    .position(|a| a == n)
                    .expect("every family name is a computed score")
            })
            .collect()
    })
}

/// The family's values for one candidate ([`NAMES`] order); all 0 without a decision.
pub fn values(inp: &Input, tol: &Q1FragTolerances) -> Vec<f64> {
    let all = compute(inp, tol);
    selection().iter().map(|&k| all[k]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: Q1FragTolerances = Q1FragTolerances {
        offset_steps: 1.1,
        centre_steps: 0.7,
        corr2d_min: 0.55,
    };

    fn ix(name: &str) -> usize {
        ALL.iter().position(|n| *n == name).unwrap()
    }

    #[test]
    fn names_are_unique_computed_scores_and_every_value_is_written() {
        for list in [NAMES, &ALL[..]] {
            let mut s: Vec<&str> = list.to_vec();
            s.sort_unstable();
            s.dedup();
            assert_eq!(s.len(), list.len());
        }
        assert!(NAMES.iter().all(|n| ALL.contains(n)));
        let pred = [1.0, 0.5, 0.2];
        let obs = [100.0, 30.0, 10.0];
        let inp = Input {
            class: &[1.0, 2.0, 0.0],
            offset: &[0.1, 0.4, f32::NAN],
            corr2d: &[0.9, 0.2, f32::NAN],
            pred: &pred,
            obs: &obs,
        };
        let v = values(&inp, &TOL);
        assert_eq!(v.len(), NAMES.len());
        assert!(v.iter().all(|x| x.is_finite()));
        let all = compute(&inp, &TOL);
        for (k, n) in NAMES.iter().enumerate() {
            assert_eq!(v[k].to_bits(), all[ix(n)].to_bits(), "{n}");
        }
    }

    /// A precursor whose fragments sit at their expected centre while the shared fit was
    /// pulled away (none belongs) is consistent all the same; an interferer's fragments,
    /// three steps off, are not.
    #[test]
    fn consistency_is_against_the_expected_centre_not_the_shared_fit() {
        let pred = [1.0, 0.8, 0.5, 0.3];
        let obs = [50.0, 40.0, 900.0, 800.0];
        let v = compute(
            &Input {
                // every present fragment interfered under the shared fit
                class: &[2.0, 2.0, 2.0, 2.0],
                offset: &[0.2, -0.3, 3.1, 2.9],
                corr2d: &[0.7, 0.6, 0.9, 0.9],
                pred: &pred,
                obs: &obs,
            },
            &TOL,
        );
        assert_eq!(v[ix("q1c_n")], 2.0);
        assert_eq!(v[ix("q1c_n_corr")], 2.0);
        assert!((v[ix("q1c_pred_share")] - 1.8 / 2.6).abs() < 1e-12);
        assert!((v[ix("q1c_obs_share")] - 90.0 / 1790.0).abs() < 1e-12);
        assert_eq!(v[ix("q1c_top1")], 1.0);
        assert_eq!(v[ix("q1c_top3")], 2.0);
        // nothing belongs, so the belonging scores are 0
        assert_eq!(v[ix("q1c_b_obs_share")], 0.0);
        assert_eq!(v[ix("q1c_b_entropy")], 0.0);
        assert_eq!(v[ix("q1c_b_cosine")], 0.0);
        // the consistent pair matches its predictions closely
        assert!(v[ix("q1c_cosine_within")] > 0.99);
        // graded: within half the tolerance only the first two, within twice all but none
        // of the interferer's (3.1 and 2.9 > 2.2)
        assert_eq!(v[ix("q1c_n_tight")], 2.0);
        assert_eq!(v[ix("q1c_n_loose")], 2.0);
        assert!((v[ix("q1c_corr2d_mean")] - 0.65).abs() < 1e-6);
        // the interferer's fragments weigh nearly nothing softly
        assert!(v[ix("q1c_soft_obs")] < v[ix("q1c_obs_share")]);
        assert!(v[ix("q1c_soft_pred")] > 0.0 && v[ix("q1c_soft_pred")] < v[ix("q1c_pred_share")]);
        let wmean = (1.0 * 0.2 + 0.8 * 0.3 + 0.5 * 3.1 + 0.3 * 2.9) / 2.6;
        assert!((v[ix("q1c_offset_wmean")] - wmean).abs() < 1e-6);
        assert!((v[ix("q1c_offset_top3")] - (0.2 + 0.3 + 3.1) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn the_correlation_floor_only_narrows_the_corr_variant() {
        let pred = [1.0, 0.5];
        let obs = [10.0, 5.0];
        let v = compute(
            &Input {
                class: &[1.0, 1.0],
                offset: &[0.0, 0.5],
                corr2d: &[0.9, 0.1],
                pred: &pred,
                obs: &obs,
            },
            &TOL,
        );
        assert_eq!(v[ix("q1c_n")], 2.0);
        assert_eq!(v[ix("q1c_n_corr")], 1.0);
        assert!(v[ix("q1c_entropy_corr")] < v[ix("q1c_entropy")]);
    }

    /// Absent fragments and fragments without a Q1 profile read as far as the grid goes: a
    /// top-3 fragment missing costs the full cap, and a candidate with nothing present has the
    /// cap as its weighted offset.
    #[test]
    fn missing_profiles_read_as_the_offset_cap() {
        let pred = [1.0, 0.6, 0.4, 0.1];
        let v = compute(
            &Input {
                class: &[0.0, 2.0, 1.0, 1.0],
                offset: &[f32::NAN, f32::NAN, 0.5, 0.0],
                corr2d: &[f32::NAN, 0.3, 0.8, 0.9],
                pred: &pred,
                obs: &[0.0, 5.0, 4.0, 1.0],
            },
            &TOL,
        );
        assert!((v[ix("q1c_offset_top3")] - (6.0 + 6.0 + 0.5) / 3.0).abs() < 1e-9);
        assert!((v[ix("q1c_offset_wmean")] - (0.6 * 6.0 + 0.4 * 0.5) / 1.1).abs() < 1e-9);
        assert_eq!(v[ix("q1c_n")], 2.0);
        let none = compute(
            &Input {
                class: &[0.0, 0.0],
                offset: &[f32::NAN, f32::NAN],
                corr2d: &[f32::NAN, f32::NAN],
                pred: &[1.0, 0.5],
                obs: &[0.0, 0.0],
            },
            &TOL,
        );
        assert_eq!(none[ix("q1c_offset_wmean")], 6.0);
        assert_eq!(none[ix("q1c_offset_top3")], 6.0);
        assert_eq!(none[ix("q1c_n")], 0.0);
        assert!(none.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mismatched_or_empty_inputs_read_zero() {
        let pred = [1.0, 0.5];
        let obs = [10.0, 5.0];
        let short = values(
            &Input {
                class: &[1.0],
                offset: &[0.0],
                corr2d: &[0.9],
                pred: &pred,
                obs: &obs,
            },
            &TOL,
        );
        assert!(short.iter().all(|x| *x == 0.0));
        let none = values(
            &Input {
                class: &[],
                offset: &[],
                corr2d: &[],
                pred: &[],
                obs: &[],
            },
            &TOL,
        );
        assert!(none.iter().all(|x| *x == 0.0));
    }
}
