//! Stage B `mumdia rt-im-train`: per-run RT calibration and windows
//! (docs/08_rt_im_train.md). Calibrates the run-independent predicted iRT to
//! observed RT from confident seed PSMs, then sets a per-candidate RT window from
//! the residuals. MVP is 3D, so IM columns are null. The sidecars are not re-run
//! here.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{CalibrationMethod, RtImTrainConfig};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, TableFile};
use serde_json::json;
use tracing::{info, warn};

use crate::calibrate::{linear_fit, percentile, Loess};

pub struct RtImTrainParams<'a> {
    pub seed_psms: &'a str,
    pub library_precursors: &'a str,
    pub out_windows: &'a str,
    pub out_cal: &'a str,
    pub cfg: &'a RtImTrainConfig,
    pub config_hash: &'a str,
}

const INSUFFICIENT_ANCHORS_STATUS: &str = "insufficient_anchors_unbounded";

#[derive(Clone, Copy, Debug, PartialEq)]
enum WindowPlan {
    /// No trustworthy iRT -> run-RT mapping exists. Extract over the complete
    /// isolation-window RT range and mark calibrated RT as unavailable.
    Unbounded,
    /// A linear mapping is available, but too few anchors exist to estimate its
    /// residual distribution. Retain the configured broad fixed half-window.
    Fixed(f64),
    /// Enough anchors exist to derive the residual-percentile half-window.
    Calibrated,
}

fn window_plan(n_train: usize, min_anchors: usize, fallback_width: f64) -> WindowPlan {
    if n_train < 2 {
        WindowPlan::Unbounded
    } else if n_train < min_anchors {
        WindowPlan::Fixed(fallback_width)
    } else {
        WindowPlan::Calibrated
    }
}

/// Convert the best-per-base-peptide map into a fixed anchor order before any
/// floating-point fit or reduction. HashMap iteration order is randomized.
fn sorted_anchor_vectors(
    best_per_pep: HashMap<u32, (f64, f64, f64)>,
) -> (Vec<u32>, Vec<f64>, Vec<f64>) {
    let mut anchors: Vec<(u32, (f64, f64, f64))> = best_per_pep.into_iter().collect();
    anchors.sort_by_key(|(base_peptide_id, _)| *base_peptide_id);
    let ids = anchors.iter().map(|(id, _)| *id).collect();
    let train_irt = anchors.iter().map(|(_, values)| values.1).collect();
    let train_rt = anchors.iter().map(|(_, values)| values.2).collect();
    (ids, train_irt, train_rt)
}

/// Deterministic anchor-peptide holdout for window sizing. The rule
/// (`base_peptide_id % 1000 < round(frac*1000)`) is duplicated verbatim in
/// `deeplc_finetune.py` so the fine-tune reference excludes exactly these
/// peptides; keying on the base peptide keeps every charge/modform of one
/// peptide on the same side of the split (a row-wise split leaks).
fn is_holdout(base_peptide_id: u32, frac: f64) -> bool {
    base_peptide_id % 1000 < (frac * 1000.0).round() as u32
}

/// Minimum held-out anchors for a trustworthy p95; below this, sizing falls
/// back to in-sample with a warning rather than trusting a noisy tail estimate.
const MIN_HOLDOUT_ANCHORS: usize = 20;

/// Result of held-out window sizing, kept for `cal.json` so a run records both
/// which sizing produced `w_rt` and the held-out residual scale it came from.
struct HoldoutSizing {
    width: f64,
    n_sizing_train: usize,
    n_holdout: usize,
    /// Held-out residual at the configured `p_rt` percentile, pre-multiplier.
    resid_p_rt_s: f64,
    resid_abs_median_s: f64,
}

/// Materialize one candidate's RT metadata. `None` means calibration is
/// unavailable: NaN is an explicit internal sentinel for `rt_pred_cal`, while
/// infinite bounds make extraction recall-safe.
fn candidate_window(calibrated_rt: Option<f64>, width: Option<f64>) -> (f64, f64, f64) {
    match (calibrated_rt, width) {
        (Some(cal), Some(w)) => (cal, cal - w, cal + w),
        _ => (f64::NAN, f64::NEG_INFINITY, f64::INFINITY),
    }
}

pub fn run(p: RtImTrainParams) -> Result<u64> {
    let t0 = Instant::now();

    let holdout_frac = p.cfg.window_holdout_frac;
    if !(0.0..=0.9).contains(&holdout_frac) {
        anyhow::bail!(
            "rt_im_train.window_holdout_frac must be in [0.0, 0.9], got {holdout_frac}; \
             0.0 disables held-out window sizing"
        );
    }
    if holdout_frac > 0.0 && p.cfg.adaptive_rt_window {
        anyhow::bail!(
            "rt_im_train.window_holdout_frac and rt_im_train.adaptive_rt_window are mutually \
             exclusive: the adaptive per-bin percentiles are in-sample and would silently undo \
             the held-out sizing; disable one of them"
        );
    }

    // Library predicted iRT, keyed by candidate_id (single source of truth, so
    // a patched/updated library iRT is used for both training and application).
    let lib = TableFile::open(p.library_precursors)?;
    let lib_cid = lib.u32("candidate_id")?;
    let lib_irt = lib.f32("predicted_irt")?;
    let mut irt_by_cid: HashMap<u32, f64> = HashMap::with_capacity(lib.nrows);
    for i in 0..lib.nrows {
        irt_by_cid.insert(lib_cid[i], lib_irt[i] as f64);
    }

    // Training rows: confident seed PSMs, one apex (best score) per peptide;
    // predicted iRT is joined from the library by candidate_id.
    let seed = TableFile::open(p.seed_psms)?;
    let s_cid = seed.u32("candidate_id")?;
    let s_base = seed.u32("base_peptide_id")?;
    let s_q = seed.f64("spectrum_q")?;
    let s_score = seed.f64("score")?;
    let s_rt = seed.f64("observed_rt")?;
    let s_label = seed.str("label")?;

    let mut best_per_pep: HashMap<u32, (f64, f64, f64)> = HashMap::new(); // base -> (score, irt, rt)
    for i in 0..seed.nrows {
        if !s_q[i].is_finite()
            || !s_score[i].is_finite()
            || !s_rt[i].is_finite()
            || s_q[i] >= p.cfg.q_train
        {
            continue;
        }
        // Only target PSMs may anchor the RT calibration; a decoy anchor injects a
        // random iRT<->RT pair into the fit.
        if s_label[i] != "target" {
            continue;
        }
        let irt = match irt_by_cid.get(&s_cid[i]) {
            Some(v) if v.is_finite() => *v,
            None => continue,
            Some(_) => continue,
        };
        let e = best_per_pep
            .entry(s_base[i])
            .or_insert((f64::NEG_INFINITY, 0.0, 0.0));
        if s_score[i] > e.0 {
            *e = (s_score[i], irt, s_rt[i]);
        }
    }
    let (anchor_ids, train_irt, train_rt) = sorted_anchor_vectors(best_per_pep);
    let n_train = train_irt.len();
    info!(n_train, "rt-im-train: training points");

    // Fit calibration predicted_irt -> observed RT (seconds).
    // Zero or one point cannot define a useful mapping across the gradient.
    let calibration_available = n_train >= 2;
    let (slope, intercept) = if calibration_available {
        linear_fit(&train_irt, &train_rt)
    } else {
        (f64::NAN, f64::NAN)
    };
    let use_loess = calibration_available
        && matches!(p.cfg.calibration_method, CalibrationMethod::Loess)
        && n_train >= p.cfg.min_seed_for_calibration;
    let loess = if use_loess {
        Some(Loess::fit(&train_irt, &train_rt, p.cfg.loess_span, 200))
    } else {
        None
    };

    let predict = |irt: f64| -> f64 {
        if !calibration_available {
            return f64::NAN;
        }
        match &loess {
            Some(l) => l.predict(irt),
            None => slope * irt + intercept,
        }
    };

    // Residuals and RT window. Require enough anchors before trusting the
    // residual-percentile window: with only a handful of points a linear fit
    // passes ~exactly through them, so residuals ~0 and the window collapses to
    // the 1s floor (which then discards nearly every true co-elution). Below the
    // threshold, use the configured fixed fallback instead.
    let min_anchors = p.cfg.min_seed_for_calibration.max(2);
    let plan = window_plan(n_train, min_anchors, p.cfg.fallback_rt_window_s);

    // Held-out window sizing (`window_holdout_frac > 0`): fit the sizing curve on
    // the non-held-out anchors only and take the residual percentile of the
    // held-out anchors against it. In-sample residuals underestimate the tail an
    // unseen library peptide actually has, and they reward a memorizing RT model
    // with a narrower window; the held-out anchors never entered this fit nor
    // (when `finetune_deeplc` ran with the same fraction) the DeepLC fine-tune,
    // so their residuals are an honest error estimate. The final calibration
    // curve applied to the library still uses every anchor; only the width is
    // sized out-of-sample, which is slightly conservative (the all-anchor curve
    // is marginally better than the sizing curve).
    let holdout_sizing: Option<HoldoutSizing> = if holdout_frac > 0.0
        && plan == WindowPlan::Calibrated
    {
        let (mut tr_x, mut tr_y, mut ho_x, mut ho_y) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for k in 0..n_train {
            if is_holdout(anchor_ids[k], holdout_frac) {
                ho_x.push(train_irt[k]);
                ho_y.push(train_rt[k]);
            } else {
                tr_x.push(train_irt[k]);
                tr_y.push(train_rt[k]);
            }
        }
        if ho_x.len() < MIN_HOLDOUT_ANCHORS || tr_x.len() < min_anchors {
            warn!(
                n_holdout = ho_x.len(),
                n_sizing_train = tr_x.len(),
                min_holdout = MIN_HOLDOUT_ANCHORS,
                min_anchors,
                "rt-im-train: too few anchors on one side of the holdout split; \
                 falling back to in-sample window sizing"
            );
            None
        } else {
            // Same method selection as the main fit, refit on the sizing subset.
            let sizing_loess = use_loess.then(|| Loess::fit(&tr_x, &tr_y, p.cfg.loess_span, 200));
            let (s_slope, s_intercept) = if sizing_loess.is_none() {
                linear_fit(&tr_x, &tr_y)
            } else {
                (f64::NAN, f64::NAN)
            };
            let sizing_predict = |x: f64| -> f64 {
                match &sizing_loess {
                    Some(l) => l.predict(x),
                    None => s_slope * x + s_intercept,
                }
            };
            let resid: Vec<f64> = ho_x
                .iter()
                .zip(&ho_y)
                .map(|(x, y)| (y - sizing_predict(*x)).abs())
                .collect();
            let p_rt_s = percentile(&resid, p.cfg.p_rt);
            Some(HoldoutSizing {
                width: (p_rt_s * p.cfg.rt_window_multiplier).max(1.0),
                n_sizing_train: tr_x.len(),
                n_holdout: ho_x.len(),
                resid_p_rt_s: p_rt_s,
                resid_abs_median_s: percentile(&resid, 0.5),
            })
        }
    } else {
        None
    };

    let (w_rt, status): (Option<f64>, String) = match plan {
        WindowPlan::Unbounded => {
            warn!(
                n_train,
                min_anchors,
                "rt-im-train: fewer than two target anchors; using unbounded RT windows"
            );
            (None, INSUFFICIENT_ANCHORS_STATUS.to_string())
        }
        WindowPlan::Fixed(width) => {
            warn!(
                n_train,
                min_anchors, "rt-im-train: too few target anchors; using fallback fixed RT window"
            );
            (Some(width), "fallback_fixed".to_string())
        }
        WindowPlan::Calibrated => {
            let width = match &holdout_sizing {
                Some(h) => h.width,
                None => {
                    let resid: Vec<f64> = train_irt
                        .iter()
                        .zip(&train_rt)
                        .map(|(x, y)| (y - predict(*x)).abs())
                        .collect();
                    (percentile(&resid, p.cfg.p_rt) * p.cfg.rt_window_multiplier).max(1.0)
                }
            };
            let status = if use_loess { "loess" } else { "linear" };
            (Some(width), status.to_string())
        }
    };

    // Optional adaptive window: local residual-percentile half-width per
    // calibrated-RT bin, so well-calibrated regions get a tight window (less
    // interference) and poorly-calibrated regions a wider one (more recall).
    // `None` keeps the single global `w_rt`. Empty bins fall back to `w_rt`.
    let adaptive: Option<(f64, f64, Vec<f64>)> =
        if p.cfg.adaptive_rt_window && n_train >= min_anchors {
            let cals: Vec<f64> = train_irt.iter().map(|x| predict(*x)).collect();
            let resid: Vec<f64> = cals
                .iter()
                .zip(&train_rt)
                .map(|(c, y)| (y - c).abs())
                .collect();
            let rt_min = cals.iter().cloned().fold(f64::INFINITY, f64::min);
            let rt_max = cals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let nb = p.cfg.adaptive_rt_bins.max(1);
            if rt_max > rt_min {
                let span = rt_max - rt_min;
                let mut per_bin: Vec<Vec<f64>> = vec![Vec::new(); nb];
                for (c, r) in cals.iter().zip(&resid) {
                    let frac = ((c - rt_min) / span).clamp(0.0, 0.999_999);
                    per_bin[(frac * nb as f64) as usize].push(*r);
                }
                let lo_clamp = p.cfg.rt_window_min_s.max(0.0);
                let hi_clamp = p.cfg.fallback_rt_window_s.max(lo_clamp);
                let widths: Vec<f64> = per_bin
                    .iter()
                    .map(|rs| {
                        if rs.is_empty() {
                            w_rt.expect("adaptive RT windows require a calibrated global width")
                        } else {
                            (percentile(rs, p.cfg.p_rt) * p.cfg.rt_window_multiplier)
                                .clamp(lo_clamp, hi_clamp)
                        }
                    })
                    .collect();
                Some((rt_min, span, widths))
            } else {
                None
            }
        } else {
            None
        };

    // Apply to every library candidate.
    let cid = lib_cid;
    let irt = lib_irt;
    let n = lib.nrows;
    let (mut cid_c, mut cal_c, mut lo_c, mut hi_c) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut im_c, mut imlo_c, mut imhi_c) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for i in 0..n {
        let calibrated_rt = calibration_available.then(|| predict(irt[i] as f64));
        let width = calibrated_rt.map(|cal| match &adaptive {
            Some((rt_min, span, widths)) => {
                let nb = widths.len();
                let frac = ((cal - rt_min) / span).clamp(0.0, 0.999_999);
                widths[(frac * nb as f64) as usize]
            }
            None => w_rt.expect("available RT calibration requires a bounded window"),
        });
        let (cal, lo, hi) = candidate_window(calibrated_rt, width);
        cid_c.push(cid[i]);
        cal_c.push(cal);
        lo_c.push(lo);
        hi_c.push(hi);
        im_c.push(None);
        imlo_c.push(None);
        imhi_c.push(None);
    }

    let rows = write_table(
        p.out_windows,
        vec![
            Col::U32("candidate_id".into(), cid_c),
            Col::F64("rt_pred_cal".into(), cal_c),
            Col::F64("rt_lo".into(), lo_c),
            Col::F64("rt_hi".into(), hi_c),
            Col::OptF64("im_pred_cal".into(), im_c),
            Col::OptF64("im_lo".into(), imlo_c),
            Col::OptF64("im_hi".into(), imhi_c),
        ],
    )?;

    let method = if !calibration_available {
        "unavailable"
    } else if use_loess {
        "loess"
    } else {
        "linear"
    };
    let slope_report = calibration_available.then_some(slope);
    let intercept_report = calibration_available.then_some(intercept);

    // RT calibration-quality residuals over the training anchors (seconds):
    // signed median = residual bias, absolute median = typical accuracy, MAD =
    // spread. Diagnostic only (the RT window already derives from these
    // residuals); surfaced so a run's RT calibration can be judged good or biased.
    let (rt_residual_median_s, rt_residual_abs_median_s, rt_residual_mad_s) =
        if calibration_available {
            let signed: Vec<f64> = train_irt
                .iter()
                .zip(&train_rt)
                .map(|(x, y)| y - predict(*x))
                .collect();
            let med = percentile(&signed, 0.5);
            let absres: Vec<f64> = signed.iter().map(|r| r.abs()).collect();
            let mad: Vec<f64> = signed.iter().map(|r| (r - med).abs()).collect();
            (med, percentile(&absres, 0.5), percentile(&mad, 0.5))
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };

    // cal.json
    mumdia_io::json::write_json(
        p.out_cal,
        &json!({
            "method": method,
            "slope": slope_report,
            "intercept": intercept_report,
            "w_rt": w_rt,
            "p_rt": p.cfg.p_rt,
            "multiplier": p.cfg.rt_window_multiplier,
            "n_train": n_train,
            "calibration_status": status,
            // Window sizing provenance. "holdout" means w_rt came from held-out
            // residuals; "holdout_fallback_in_sample" means it was requested but
            // an anchor-count guard fell back; "in_sample" is the historical
            // behavior. The holdout_* fields are null unless sizing ran held-out.
            "w_rt_sizing": match (&holdout_sizing, holdout_frac > 0.0) {
                (Some(_), _) => "holdout",
                (None, true) => "holdout_fallback_in_sample",
                (None, false) => "in_sample",
            },
            "window_holdout_frac": holdout_frac,
            "n_sizing_train": holdout_sizing.as_ref().map(|h| h.n_sizing_train),
            "n_holdout": holdout_sizing.as_ref().map(|h| h.n_holdout),
            "holdout_resid_p_rt_s": holdout_sizing.as_ref().map(|h| h.resid_p_rt_s),
            "holdout_resid_abs_median_s": holdout_sizing.as_ref().map(|h| h.resid_abs_median_s),
            // Calibration-quality diagnostics (post-fit RT residuals, seconds).
            "rt_residual_median_s": rt_residual_median_s,
            "rt_residual_abs_median_s": rt_residual_abs_median_s,
            "rt_residual_mad_s": rt_residual_mad_s,
        }),
    )?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("n_train".to_string(), json!(n_train));
    stats.insert("w_rt".to_string(), json!(w_rt));
    stats.insert("calibration_status".to_string(), json!(status));
    ArtifactReport {
        logical_name: artifact::RUN_WINDOWS.0.to_string(),
        schema_name: artifact::RUN_WINDOWS.0.to_string(),
        schema_version: artifact::RUN_WINDOWS.1,
        stage: "rt-im-train".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out_windows)?,
        params: json!({"q_train": p.cfg.q_train, "p_rt": p.cfg.p_rt, "method": format!("{:?}", p.cfg.calibration_method)}),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    }
    .write_for(p.out_windows)?;

    info!(
        rows,
        w_rt = ?w_rt,
        status,
        elapsed_ms = elapsed,
        "rt-im-train: done"
    );
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_vectors_are_sorted_by_base_peptide_id() {
        let mut first = HashMap::new();
        first.insert(30, (9.0, 3.0, 300.0));
        first.insert(10, (7.0, 1.0, 100.0));
        first.insert(20, (8.0, 2.0, 200.0));

        let mut shuffled = HashMap::new();
        shuffled.insert(20, (8.0, 2.0, 200.0));
        shuffled.insert(30, (9.0, 3.0, 300.0));
        shuffled.insert(10, (7.0, 1.0, 100.0));

        let expected = (
            vec![10, 20, 30],
            vec![1.0, 2.0, 3.0],
            vec![100.0, 200.0, 300.0],
        );
        assert_eq!(sorted_anchor_vectors(first), expected);
        assert_eq!(sorted_anchor_vectors(shuffled), expected);
    }

    #[test]
    fn holdout_split_is_deterministic_and_fraction_shaped() {
        // frac 0.0 holds out nothing; frac 0.9 (the cap) holds out ids 0..899 mod 1000.
        assert!(!is_holdout(0, 0.0));
        assert!(!is_holdout(999, 0.0));
        // The documented rule, verbatim: base_peptide_id % 1000 < round(frac*1000).
        // deeplc_finetune.py pins these same four cases; keep them in sync.
        assert!(is_holdout(299, 0.3));
        assert!(!is_holdout(300, 0.3));
        assert!(is_holdout(1_000_299, 0.3));
        assert!(!is_holdout(1_000_300, 0.3));
        // Empirical fraction over a contiguous id range is exactly frac.
        let n_held = (0u32..1_000_000).filter(|id| is_holdout(*id, 0.3)).count();
        assert_eq!(n_held, 300_000);
    }

    #[test]
    fn sparse_anchor_policy_is_unbounded_only_below_two() {
        assert_eq!(window_plan(0, 50, 120.0), WindowPlan::Unbounded);
        assert_eq!(window_plan(1, 50, 120.0), WindowPlan::Unbounded);
        assert_eq!(window_plan(2, 50, 120.0), WindowPlan::Fixed(120.0));
        assert_eq!(window_plan(49, 50, 120.0), WindowPlan::Fixed(120.0));
        assert_eq!(window_plan(50, 50, 120.0), WindowPlan::Calibrated);
        assert_eq!(
            INSUFFICIENT_ANCHORS_STATUS,
            "insufficient_anchors_unbounded"
        );
    }

    #[test]
    fn unavailable_calibration_emits_unbounded_window_and_nan_prediction() {
        let (cal, lo, hi) = candidate_window(None, None);
        assert!(cal.is_nan());
        assert_eq!(lo, f64::NEG_INFINITY);
        assert_eq!(hi, f64::INFINITY);

        assert_eq!(
            candidate_window(Some(300.0), Some(120.0)),
            (300.0, 180.0, 420.0)
        );
    }
}
