//! The temperature fit (`localise.temperature = "auto"`, docs/36_ptm_localisation.md
//! section 3).
//!
//! Without ground truth the decoy residue is the reference: a placement on it is wrong by
//! construction, so its wins, scaled by the ratio of candidate to decoy-residue positions,
//! estimate how many real calls are wrong. The model's own estimate is the sum of `1 - p` over
//! its real calls. The fit chooses the temperature at which the two agree among the calls at
//! or above [`CAL_THRESHOLD`], with a prior on the log temperature centred on `prior`:
//!
//! `objective(T) = (E_model - E_decoy)^2 / var + (ln T - ln prior)^2 / SIGMA^2`
//!
//! where `var` is the Poisson variance of the decoy estimate plus the binomial variance of the
//! model's. The temperature moves at all only when the two estimates disagree at `prior` by
//! more than [`GATE_Z`] standard errors: with no or one decoy-residue win in the population
//! the decoy estimate is near zero whatever the error rate, and a fit on it would sharpen the
//! posterior where the model is already optimistic (the three-run dilution series of the
//! synthetic phosphopeptides, docs/36 section 7).
//!
//! The population is the target peaks with a choice of site and a decoy-residue position
//! whose share of site-determining ions present is in the top [`SUBSET_SHARE`] of theirs. A
//! peak whose modification count or peptide is wrong has no right placement among the
//! hypotheses: its calls are as likely to land on the decoy residue as on any other position,
//! however confident, and no temperature makes the model say so. Such peaks see few of the
//! site-determining ions of any placement, so the share present selects the peaks with a
//! right answer. It is a property of the evidence, not of which placement won, so the decoy
//! residue stays exchangeable with the wrong real placements inside the population. On the
//! synthetic phosphopeptides (docs/36 section 7) two thirds of all peaks with a choice had no
//! synthesised placement and fitted on all of them the temperature ran to the top of the grid;
//! fitted on the top quarter it was 1.40, against 1.29 fitted on the synthesised peptides
//! alone.

use serde::Serialize;

use super::{model, status, PeakResult, RunAnalysis, Temperature};

/// Standard deviation of the prior on `ln T`: one standard deviation is a factor of 1.42.
/// With 0.5 the fit on the synthetic pure pool jumped between 1.1 and 2.3 as the population
/// cut moved by a few peaks, one decoy-residue win at a time; with 0.35 it stayed within 1.26
/// to 1.40 of the temperature fitted on the synthesised peptides alone (docs/36 section 7).
const SIGMA: f64 = 0.35;
/// The temperature grid: `ln T` from `ln 0.25` to `ln 8` in steps of `GRID_STEP`.
const LN_T_MIN: f64 = -1.386_294_361_119_890_6;
const LN_T_MAX: f64 = 2.079_441_541_679_835_7;
const GRID_STEP: f64 = 0.02;
/// The calls the model and the decoy residue are matched over: minimum site probability at
/// or above this, whatever `localise.threshold` reports at, so the temperature is a property
/// of the model and not of the reporting threshold.
pub(crate) const CAL_THRESHOLD: f64 = 0.75;
/// The share of eligible peaks, highest site-determining-ion presence first, the fit is made
/// on.
pub(crate) const SUBSET_SHARE: f64 = 0.25;
/// The disagreement, in standard errors, between the model's and the decoy residue's
/// estimates at the prior's centre below which the temperature stays there.
pub(crate) const GATE_Z: f64 = 2.0;

/// A grid point: objective, `ln T`, and (chi-square, real calls, decoy calls, model errors,
/// decoy-residue errors).
type GridPoint = (f64, f64, (f64, usize, usize, f64, f64));

/// What the fit found, for the report.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct Fit {
    pub temperature: f64,
    /// The centre of the prior.
    pub prior: f64,
    /// Eligible peaks (targets with a choice of site), and the ones the fit was made on.
    pub eligible: usize,
    pub peaks: usize,
    /// The smallest share of site-determining ions present in the population.
    pub sdi_share_cut: f64,
    /// At the fitted temperature: real and decoy-residue calls at or above the calibration
    /// threshold.
    pub real_calls: usize,
    pub decoy_calls: usize,
    /// The model's expected wrong real calls, and the decoy residue's estimate of them.
    pub model_errors: f64,
    pub decoy_errors: f64,
    /// The candidate-to-decoy position ratio of the population.
    pub ratio: f64,
    /// The disagreement of the two estimates at the prior's centre, in standard errors, and
    /// whether it was large enough to fit the temperature.
    pub z_at_prior: f64,
    pub fitted: bool,
}

/// One peak the fit may use.
pub(crate) struct CalPeak<'a> {
    pub ll: &'a [f64],
    pub hyps: &'a [Vec<usize>],
    pub on_decoy: &'a [bool],
    pub n_slots: usize,
    pub real_positions: usize,
    pub decoy_positions: usize,
    /// The share of its site-determining ions that are present.
    pub sdi_share: f64,
}

/// Whether a scored peak may enter the population: a choice of site, the population of the
/// false localisation rates (section 3). A peak without a decoy-residue position counts too:
/// its wrong calls can only land on real positions, which the position ratio accounts for.
fn eligible(res: &PeakResult) -> bool {
    res.status == status::SCORED && res.n_real >= 2
}

/// The eligible peaks of a run, before [`super::finish`].
pub(crate) fn cal_peaks<'a>(a: &'a RunAnalysis) -> Vec<CalPeak<'a>> {
    (0..a.peaks.len())
        .filter(|&pi| !a.rows[a.peaks[pi].rep].decoy && eligible(&a.results[pi]))
        .map(|pi| {
            let res = &a.results[pi];
            CalPeak {
                ll: &res.ll,
                hyps: &res.hyps,
                on_decoy: &res.on_decoy,
                n_slots: a.rows[a.peaks[pi].rep].split.n_slots(),
                real_positions: res.positions.len() - res.n_decoy_positions,
                decoy_positions: res.n_decoy_positions,
                sdi_share: f64::from(res.n_sdi_observed) / f64::from(res.n_sdi.max(1)),
            }
        })
        .collect()
}

/// Fit one temperature over the eligible peaks of `runs`.
pub(crate) fn fit_runs(runs: &[&RunAnalysis], prior: f64) -> Temperature {
    let peaks: Vec<CalPeak> = runs.iter().flat_map(|a| cal_peaks(a)).collect();
    let f = fit(&peaks, prior);
    Temperature {
        value: f.temperature,
        fit: Some(f),
    }
}

/// The calls at or above the calibration threshold at temperature `t`: real calls, decoy
/// calls, and the model's expected errors and their binomial variance among the real calls.
fn tally(peaks: &[&CalPeak], t: f64) -> (usize, usize, f64, f64) {
    let (mut real, mut decoy, mut err, mut var) = (0usize, 0usize, 0.0f64, 0.0f64);
    for p in peaks {
        let c = model::call(p.ll, p.hyps, p.n_slots, t);
        if c.min_site_prob < CAL_THRESHOLD {
            continue;
        }
        if p.on_decoy[c.best] {
            decoy += 1;
        } else {
            real += 1;
            let q = 1.0 - c.min_site_prob;
            err += q;
            var += q * (1.0 - q);
        }
    }
    (real, decoy, err, var)
}

/// Fit the temperature on the population of `peaks` (see the module documentation): the grid
/// point of least objective, ties to the one nearest the prior.
pub(crate) fn fit(peaks: &[CalPeak], prior: f64) -> Fit {
    // the population: the top SUBSET_SHARE by share of site-determining ions present
    let mut shares: Vec<f64> = peaks.iter().map(|p| p.sdi_share).collect();
    shares.sort_by(|a, b| b.total_cmp(a));
    let cut = if shares.is_empty() {
        0.0
    } else {
        let k = ((shares.len() as f64 * SUBSET_SHARE).ceil() as usize).clamp(1, shares.len());
        shares[k - 1]
    };
    let pop: Vec<&CalPeak> = peaks.iter().filter(|p| p.sdi_share >= cut).collect();
    let (rp, dp) = pop.iter().fold((0usize, 0usize), |acc, p| {
        (acc.0 + p.real_positions, acc.1 + p.decoy_positions)
    });
    let ratio = if dp > 0 {
        rp as f64 / dp as f64
    } else {
        f64::NAN
    };
    let at = |t: f64| {
        let (real, decoy, err, var) = tally(&pop, t);
        let e_decoy = decoy as f64 * ratio;
        let chi = if ratio.is_finite() {
            (err - e_decoy).powi(2) / (ratio * ratio * (decoy as f64 + 1.0) + var).max(1e-9)
        } else {
            0.0
        };
        (chi, real, decoy, err, e_decoy)
    };
    let ln_prior = prior.ln();
    let v0 = at(prior);
    let z_at_prior = v0.0.sqrt();
    if z_at_prior <= GATE_Z {
        return Fit {
            temperature: prior,
            prior,
            eligible: peaks.len(),
            peaks: pop.len(),
            sdi_share_cut: cut,
            real_calls: v0.1,
            decoy_calls: v0.2,
            model_errors: v0.3,
            decoy_errors: v0.4,
            ratio,
            z_at_prior,
            fitted: false,
        };
    }
    let mut best: Option<GridPoint> = None;
    let steps = ((LN_T_MAX - LN_T_MIN) / GRID_STEP).round() as usize;
    for k in 0..=steps {
        let ln_t = LN_T_MIN + k as f64 * GRID_STEP;
        let v = at(ln_t.exp());
        let obj = v.0 + (ln_t - ln_prior).powi(2) / (SIGMA * SIGMA);
        let better = match &best {
            None => true,
            Some((o, l, _)) => {
                obj < *o || (obj == *o && (ln_t - ln_prior).abs() < (l - ln_prior).abs())
            }
        };
        if better {
            best = Some((obj, ln_t, v));
        }
    }
    let (_, ln_t, v) = best.expect("the grid is not empty");
    Fit {
        temperature: ln_t.exp(),
        prior,
        eligible: peaks.len(),
        peaks: pop.len(),
        sdi_share_cut: cut,
        real_calls: v.1,
        decoy_calls: v.2,
        model_errors: v.3,
        decoy_errors: v.4,
        ratio,
        z_at_prior,
        fitted: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` peaks whose evidence favours the first of two real placements by `margin` nats,
    /// with a third, decoy, placement; `wrong` of them have the decoy on top.
    fn population(n: usize, margin: f64, wrong: usize) -> (Vec<Vec<f64>>, Vec<Vec<Vec<usize>>>) {
        let mut ll = Vec::new();
        let mut hyps = Vec::new();
        for i in 0..n {
            if i < wrong {
                ll.push(vec![0.0, -margin, margin]);
            } else {
                ll.push(vec![margin, 0.0, -margin]);
            }
            hyps.push(vec![vec![1], vec![3], vec![5]]);
        }
        (ll, hyps)
    }

    fn peaks<'a>(
        ll: &'a [Vec<f64>],
        hyps: &'a [Vec<Vec<usize>>],
        dec: &'a [bool],
        share: impl Fn(usize) -> f64,
    ) -> Vec<CalPeak<'a>> {
        ll.iter()
            .zip(hyps)
            .enumerate()
            .map(|(i, (l, h))| CalPeak {
                ll: l,
                hyps: h,
                on_decoy: dec,
                n_slots: 8,
                real_positions: 2,
                decoy_positions: 1,
                sdi_share: share(i),
            })
            .collect()
    }

    /// Without decoy wins the prior holds the temperature at or below its centre; with many
    /// decoy wins at a confident margin the fit flattens the posterior; a handful is noise.
    #[test]
    fn the_fit_follows_the_decoy_residue_and_the_prior() {
        let dec = [false, false, true];
        let flat = |_| 0.5;
        let (ll, hyps) = population(1600, 4.0, 0);
        let f = fit(&peaks(&ll, &hyps, &dec, flat), 1.0);
        assert!(f.temperature <= 1.0 + 1e-9, "{f:?}");
        assert_eq!(f.decoy_calls, 0);
        // the whole population ties on the share, so all of it is used
        assert_eq!(f.peaks, 1600);
        // 10% confidently wrong: the model at T = 1 claims about 2% errors
        let (ll, hyps) = population(1600, 4.0, 160);
        let f = fit(&peaks(&ll, &hyps, &dec, flat), 1.0);
        assert!(f.temperature > 1.2, "{f:?}");
        // a handful of decoy wins is noise: the gate keeps the temperature at 1
        let (ll, hyps) = population(1600, 4.0, 8);
        let f = fit(&peaks(&ll, &hyps, &dec, flat), 1.0);
        assert_eq!(f.temperature, 1.0, "{f:?}");
        assert!(!f.fitted && f.z_at_prior <= GATE_Z);
        let g = fit(&peaks(&ll, &hyps, &dec, flat), 1.0);
        assert_eq!(f.temperature, g.temperature);
        // so is a small population without a decoy win: it cannot show the model is
        // pessimistic
        let (ll, hyps) = population(100, 4.0, 0);
        let f = fit(&peaks(&ll, &hyps, &dec, flat), 1.3);
        assert_eq!(f.temperature, 1.3, "{f:?}");
        assert_eq!(f.decoy_calls, 0);
    }

    /// Peaks without a right answer (the decoy on top, little site-determining evidence) do
    /// not drive the fit: the population is the quarter with the most evidence.
    #[test]
    fn peaks_without_evidence_stay_out_of_the_population() {
        let dec = [false, false, true];
        // 400 informative peaks, all right; 1200 junk peaks, a quarter of them decoy wins
        let (ll, hyps) = population(1600, 4.0, 300);
        let share = |i: usize| if i < 1200 { 0.05 } else { 0.5 };
        // the junk is peaks 0..1200 (the first 300 of them decoy wins)
        let f = fit(&peaks(&ll, &hyps, &dec, share), 1.0);
        assert_eq!(f.peaks, 400);
        assert_eq!(f.sdi_share_cut, 0.5);
        assert_eq!(f.decoy_calls, 0);
        assert!(f.temperature <= 1.0 + 1e-9, "{f:?}");
        // all of them: the junk's decoy wins push the temperature up
        let g = fit(&peaks(&ll, &hyps, &dec, |_| 0.5), 1.0);
        assert!(g.temperature > 1.5, "{g:?}");
    }
}
