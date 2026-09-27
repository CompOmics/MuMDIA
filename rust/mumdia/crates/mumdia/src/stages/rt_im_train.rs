//! Stage B `mumdia rt-im-train`: per-run RT calibration and windows
//! (docs/08_rt_im_train.md). Calibrates the run-independent predicted iRT to
//! observed RT from confident seed PSMs, then sets a per-candidate RT window from
//! the residuals. When the library carries `predicted_im` and the seed carries
//! `observed_im` (diaPASEF), the same anchors calibrate ion mobility: a per-charge
//! linear map in CCS space and a held-out residual window (`fit_im`). Otherwise the IM
//! columns are null. The sidecars are not re-run here.

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
    /// Take each anchor's predicted iRT from the seed table's own `predicted_irt` column
    /// instead of joining the library by `candidate_id`. For a pooled seed of a grouped
    /// run: its ids are library-wide while the band's table is local, and the pooling step
    /// has refreshed the column from the re-predicted band libraries, so the seed is the
    /// source of truth there. Off, anchors outside the library are dropped silently.
    pub anchor_irt_from_seed: bool,
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
    // `--out` must not be one of this stage's own inputs: every input is read
    // before the output is published, so writing over one replaces it and exits 0
    // (docs/31 F6). The shared guard existed and was wired into two stages.
    mumdia_io::refuse_output_over_input(
        p.out_windows,
        &[
            ("--seed-psms", p.seed_psms),
            ("--lib-precursors", p.library_precursors),
        ],
    )?;
    mumdia_io::refuse_output_over_input(
        p.out_cal,
        &[
            ("--seed-psms", p.seed_psms),
            ("--lib-precursors", p.library_precursors),
        ],
    )?;

    let im_holdout_frac = p.cfg.im_window_holdout_frac;
    if !(im_holdout_frac > 0.0 && im_holdout_frac <= 0.9) {
        anyhow::bail!(
            "rt_im_train.im_window_holdout_frac must be in (0.0, 0.9], got {im_holdout_frac}; \
             the IM window is always sized on held-out anchors"
        );
    }
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
    let lib_mz = lib.f64("precursor_mz")?;
    let lib_z = lib.i32("charge")?;
    let lib_irt = lib.f32("predicted_irt")?;
    // v1 libraries and imported ones without IM carry no `predicted_im`.
    let lib_im: Option<Vec<Option<f64>>> = if lib.has_column("predicted_im") {
        Some(lib.opt_f64("predicted_im")?)
    } else {
        None
    };
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
    let s_irt: Option<Vec<f32>> = if p.anchor_irt_from_seed {
        Some(seed.f32("predicted_irt")?)
    } else {
        None
    };

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
        let irt = match &s_irt {
            Some(col) => {
                let v = col[i] as f64;
                if !v.is_finite() {
                    continue;
                }
                v
            }
            None => match irt_by_cid.get(&s_cid[i]) {
                Some(v) if v.is_finite() => *v,
                None => continue,
                Some(_) => continue,
            },
        };
        let e = best_per_pep
            .entry(s_base[i])
            .or_insert((f64::NEG_INFINITY, 0.0, 0.0));
        if s_score[i] > e.0 {
            *e = (s_score[i], irt, s_rt[i]);
        }
    }
    let (anchor_ids, train_irt, train_rt) = sorted_anchor_vectors(best_per_pep);

    // IM anchors: the same confident targets, one per candidate (the seed table already
    // holds one row per candidate), because 1/K0 depends on charge where RT does not.
    let im_cal: Option<ImCal> = match &lib_im {
        Some(lib_im) if seed.has_column("observed_im") => {
            let s_obs_im = seed.opt_f64("observed_im")?;
            let s_mz = seed.f64("precursor_mz")?;
            let s_z = seed.i32("charge")?;
            let pred_by_cid: HashMap<u32, f64> = lib_cid
                .iter()
                .zip(lib_im)
                .filter_map(|(c, v)| v.map(|v| (*c, v)))
                .collect();
            let mut anchors: Vec<ImAnchor> = Vec::new();
            for i in 0..seed.nrows {
                if !s_q[i].is_finite() || s_q[i] >= p.cfg.q_train || s_label[i] != "target" {
                    continue;
                }
                let (Some(obs), Some(&pred)) = (s_obs_im[i], pred_by_cid.get(&s_cid[i])) else {
                    continue;
                };
                if obs.is_finite() && obs > 0.0 && pred.is_finite() && pred > 0.0 {
                    anchors.push(ImAnchor {
                        cid: s_cid[i],
                        base_peptide_id: s_base[i],
                        charge: s_z[i],
                        mz: s_mz[i],
                        pred_im: pred,
                        obs_im: obs,
                    });
                }
            }
            anchors.sort_by_key(|a| a.cid);
            let fit = fit_im(&anchors, p.cfg);
            // A 3D run carries both columns all-null; only a run with IM on both sides and
            // too few anchors is worth a warning.
            let has_im = !pred_by_cid.is_empty() && s_obs_im.iter().any(|v| v.is_some());
            if fit.is_none() && has_im {
                warn!(
                    n_anchors = anchors.len(),
                    min_anchors = p.cfg.min_seed_for_calibration,
                    "rt-im-train: too few IM anchors; IM calibration unavailable, IM windows null"
                );
            }
            fit
        }
        Some(lib_im) if lib_im.iter().any(|v| v.is_some()) => {
            warn!(
                "rt-im-train: the library has predicted_im but the seed table has no \
                 observed_im (a v1 seed); IM calibration unavailable, IM windows null"
            );
            None
        }
        _ => None,
    };
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
    let mut n_nonfinite_irt = 0u64;
    for i in 0..n {
        // A row whose library iRT is not finite has no calibrated RT, which is the
        // documented "search the whole gradient" sentinel rather than an arithmetic
        // accident. The parquet reader maps a null f32 to NaN, so one failed prediction
        // in an imported library reaches here (docs/31 F4).
        let usable_irt = (irt[i] as f64).is_finite();
        if !usable_irt {
            n_nonfinite_irt += 1;
        }
        let calibrated_rt = (calibration_available && usable_irt).then(|| predict(irt[i] as f64));
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
        let im_pred = match (&im_cal, &lib_im) {
            (Some(c), Some(v)) => v[i].map(|pred| c.predict(pred, lib_mz[i], lib_z[i])),
            _ => None,
        };
        let w_im = im_cal.as_ref().map(|c| c.w_im).unwrap_or(0.0);
        im_c.push(im_pred);
        imlo_c.push(im_pred.map(|v| v - w_im));
        imhi_c.push(im_pred.map(|v| v + w_im));
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
            // Ion-mobility calibration (null fields when unavailable). `w_im` is always
            // sized on held-out anchors unless `w_im_sizing` says otherwise; the in-sample
            // residual is a fit diagnostic only (docs/08 section 4).
            "im_method": if im_cal.is_some() { "per_charge_linear_ccs" } else { "unavailable" },
            "im_n_train": im_cal.as_ref().map(|c| c.n_train),
            "im_global": im_cal.as_ref().map(|c| json!({"a": c.coefs.0 .0, "b": c.coefs.0 .1})),
            "im_per_charge": im_cal.as_ref().map(|c| c.coefs.1.iter()
                .map(|(z, (a, b, n))| (z.to_string(), json!({"a": a, "b": b, "n": n})))
                .collect::<serde_json::Map<_, _>>()),
            "w_im": im_cal.as_ref().map(|c| c.w_im),
            "w_im_sizing": im_cal.as_ref().map(|c| c.sizing),
            "p_im": p.cfg.p_im,
            "im_window_holdout_frac": im_holdout_frac,
            "im_n_holdout": im_cal.as_ref().map(|c| c.n_holdout),
            "im_holdout_resid_abs_median": im_cal.as_ref().and_then(|c| c.holdout_abs_median),
            "im_holdout_resid_p_im": im_cal.as_ref().and_then(|c| c.holdout_p_im),
            "im_in_sample_resid_abs_median": im_cal.as_ref().map(|c| c.in_sample_abs_median),
        }),
    )?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("n_train".to_string(), json!(n_train));
    stats.insert("w_rt".to_string(), json!(w_rt));
    stats.insert("calibration_status".to_string(), json!(status));
    stats.insert(
        "im_n_train".to_string(),
        json!(im_cal.as_ref().map(|c| c.n_train)),
    );
    stats.insert("w_im".to_string(), json!(im_cal.as_ref().map(|c| c.w_im)));
    stats.insert(
        "candidates_without_finite_irt".to_string(),
        json!(n_nonfinite_irt),
    );
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

    if n_nonfinite_irt > 0 {
        tracing::warn!(
            candidates = n_nonfinite_irt,
            of = rows,
            "rt-im-train: these candidates have no finite library iRT, so they get the              unbounded RT window rather than a calibrated one; a null predicted_irt reads              as NaN (docs/31 F4)"
        );
    }
    info!(
        rows,
        w_rt = ?w_rt,
        status,
        without_finite_irt = n_nonfinite_irt,
        elapsed_ms = elapsed,
        "rt-im-train: done"
    );
    Ok(rows)
}

/// One ion-mobility calibration anchor: a confident target seed PSM with an observed and
/// a library-predicted 1/K0.
#[derive(Clone, Debug)]
pub(crate) struct ImAnchor {
    pub(crate) cid: u32,
    pub(crate) base_peptide_id: u32,
    pub(crate) charge: i32,
    pub(crate) mz: f64,
    pub(crate) pred_im: f64,
    pub(crate) obs_im: f64,
}

/// A fitted per-run IM calibration: `obs_ccs = a + b * pred_ccs`, per charge where that
/// charge has `im_min_anchors_per_charge` anchors, else the global fit.
struct ImCal {
    coefs: ImCoefs,
    w_im: f64,
    sizing: &'static str,
    n_train: usize,
    n_holdout: usize,
    holdout_abs_median: Option<f64>,
    holdout_p_im: Option<f64>,
    in_sample_abs_median: f64,
}

pub(crate) type ImCoefs = (
    (f64, f64),
    std::collections::BTreeMap<i32, (f64, f64, usize)>,
);

impl ImCal {
    fn predict(&self, pred_im: f64, mz: f64, charge: i32) -> f64 {
        im_predict(&self.coefs, pred_im, mz, charge)
    }
}

pub(crate) fn im_predict(coefs: &ImCoefs, pred_im: f64, mz: f64, charge: i32) -> f64 {
    use mumdia_core::constants::{ccs_to_im, im_to_ccs};
    let (a, b) = coefs.1.get(&charge).map(|c| (c.0, c.1)).unwrap_or(coefs.0);
    ccs_to_im(a + b * im_to_ccs(pred_im, mz, charge), mz, charge)
}

/// Least-squares `obs_ccs = a + b * pred_ccs`, globally and per charge. Anchors are in a
/// fixed order (sorted by candidate id), and the per-charge map is ordered, so the fit is
/// deterministic.
pub(crate) fn fit_im_coefs(anchors: &[&ImAnchor], min_per_charge: usize) -> ImCoefs {
    use mumdia_core::constants::im_to_ccs;
    let ccs = |a: &ImAnchor| {
        (
            im_to_ccs(a.pred_im, a.mz, a.charge),
            im_to_ccs(a.obs_im, a.mz, a.charge),
        )
    };
    let (x, y): (Vec<f64>, Vec<f64>) = anchors.iter().map(|a| ccs(a)).unzip();
    let (b, a) = linear_fit(&x, &y);
    let mut by_z: std::collections::BTreeMap<i32, (Vec<f64>, Vec<f64>)> = Default::default();
    for an in anchors {
        let (px, py) = ccs(an);
        let e = by_z.entry(an.charge).or_default();
        e.0.push(px);
        e.1.push(py);
    }
    let per_charge = by_z
        .into_iter()
        .filter(|(_, (xs, _))| xs.len() >= min_per_charge.max(2))
        .map(|(z, (xs, ys))| {
            let (bz, az) = linear_fit(&xs, &ys);
            (z, (az, bz, xs.len()))
        })
        .collect();
    ((a, b), per_charge)
}

/// Fit the IM calibration on every anchor and size `w_im` from anchors held out of a
/// refit (split by base peptide, as the RT holdout). `None` below
/// `min_seed_for_calibration` anchors: no IM window is safer than one from a handful of
/// points, and a null window means "do not gate".
fn fit_im(anchors: &[ImAnchor], cfg: &RtImTrainConfig) -> Option<ImCal> {
    if anchors.len() < cfg.min_seed_for_calibration.max(2) {
        return None;
    }
    let all: Vec<&ImAnchor> = anchors.iter().collect();
    let coefs = fit_im_coefs(&all, cfg.im_min_anchors_per_charge);
    let resid = |c: &ImCoefs, set: &[&ImAnchor]| -> Vec<f64> {
        set.iter()
            .map(|a| (a.obs_im - im_predict(c, a.pred_im, a.mz, a.charge)).abs())
            .collect()
    };
    let in_sample = resid(&coefs, &all);
    let (train, held): (Vec<&ImAnchor>, Vec<&ImAnchor>) = all
        .iter()
        .partition(|a| !is_holdout(a.base_peptide_id, cfg.im_window_holdout_frac));
    let (p_resid, sizing, holdout) = if held.len() >= MIN_HOLDOUT_ANCHORS
        && train.len() >= cfg.min_seed_for_calibration.max(2)
    {
        let r = resid(&fit_im_coefs(&train, cfg.im_min_anchors_per_charge), &held);
        let pr = percentile(&r, cfg.p_im);
        (pr, "holdout", Some((percentile(&r, 0.5), pr)))
    } else {
        warn!(
            n_holdout = held.len(),
            n_sizing_train = train.len(),
            "rt-im-train: too few IM anchors on one side of the holdout split; sizing w_im \
             in-sample"
        );
        (
            percentile(&in_sample, cfg.p_im),
            "holdout_fallback_in_sample",
            None,
        )
    };
    Some(ImCal {
        coefs,
        w_im: (p_resid * cfg.im_window_multiplier).max(cfg.im_window_min),
        sizing,
        n_train: anchors.len(),
        n_holdout: held.len(),
        holdout_abs_median: holdout.map(|h| h.0),
        holdout_p_im: holdout.map(|h| h.1),
        in_sample_abs_median: percentile(&in_sample, 0.5),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn im_calibration_recovers_a_per_charge_ccs_map_and_sizes_w_im_held_out() {
        use mumdia_core::constants::{ccs_to_im, im_to_ccs};
        // Charge 2: obs_ccs = 10 + 1.02 * pred_ccs; charge 3: obs_ccs = -5 + 0.98 * pred.
        // A deterministic +/-0.004 1/K0 jitter gives a known residual scale.
        let mut anchors = Vec::new();
        for i in 0..600u32 {
            let z = if i % 3 == 0 { 3 } else { 2 };
            let mz = 400.0 + (i as f64) * 0.9;
            let pred = 0.7 + (i as f64) * 0.0008;
            let (a, b) = if z == 2 { (10.0, 1.02) } else { (-5.0, 0.98) };
            let jitter = if i % 2 == 0 { 0.004 } else { -0.004 };
            let obs = ccs_to_im(a + b * im_to_ccs(pred, mz, z), mz, z) + jitter;
            anchors.push(ImAnchor {
                cid: i,
                base_peptide_id: i,
                charge: z,
                mz,
                pred_im: pred,
                obs_im: obs,
            });
        }
        let cfg = RtImTrainConfig::default();
        let cal = fit_im(&anchors, &cfg).expect("600 anchors calibrate");
        assert_eq!(cal.sizing, "holdout");
        assert_eq!(cal.n_holdout, 300, "ids 0..300 satisfy id % 1000 < 300");
        let (a2, b2, _) = cal.coefs.1[&2];
        assert!(
            (b2 - 1.02).abs() < 0.01 && (a2 - 10.0).abs() < 5.0,
            "{a2} {b2}"
        );
        let (_, b3, _) = cal.coefs.1[&3];
        assert!((b3 - 0.98).abs() < 0.01, "{b3}");
        // The residuals are the jitter, so the held-out p95 is ~0.004.
        let p = cal.holdout_p_im.unwrap();
        assert!((p - 0.004).abs() < 0.001, "{p}");
        assert_eq!(cal.w_im, p.max(cfg.im_window_min));
        // Too few anchors: no calibration rather than a window from a handful of points.
        assert!(fit_im(&anchors[..10], &cfg).is_none());
    }

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
