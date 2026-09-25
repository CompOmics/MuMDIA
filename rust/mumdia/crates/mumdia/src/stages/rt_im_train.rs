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

/// A fitted retention-time calibration: everything the anchors determine, which
/// [`apply`] then maps over a library table.
///
/// Under `groups.calibration = global` every band of a grouped run fits the same pooled
/// anchors with their iRT read from the seed (`anchor_irt_from_seed`), so the fit depends on
/// the seed and the configuration only, never on the band's library. `run_groups` therefore
/// fits once per run ([`fit_from_seed`]) and applies the one fit to every band, which is
/// what each band computed for itself before: one seed decode and one fit per band
/// (docs/33 section 4).
pub struct RtFit {
    n_train: usize,
    calibration_available: bool,
    slope: f64,
    intercept: f64,
    use_loess: bool,
    loess: Option<Loess>,
    w_rt: Option<f64>,
    status: String,
    holdout_frac: f64,
    holdout_sizing: Option<HoldoutSizing>,
    adaptive: Option<(f64, f64, Vec<f64>)>,
    /// Signed median, absolute median and MAD of the in-sample residuals (seconds).
    residuals: (f64, f64, f64),
}

impl RtFit {
    fn predict(&self, irt: f64) -> f64 {
        if !self.calibration_available {
            return f64::NAN;
        }
        match &self.loess {
            Some(l) => l.predict(irt),
            None => self.slope * irt + self.intercept,
        }
    }

    /// Anchors the fit was made on.
    pub fn n_train(&self) -> usize {
        self.n_train
    }
}

/// What [`apply`] writes: the windows of one library table and its `cal.json`.
pub struct ApplyParams<'a> {
    pub library_precursors: &'a str,
    pub out_windows: &'a str,
    pub out_cal: &'a str,
    pub cfg: &'a RtImTrainConfig,
    pub config_hash: &'a str,
}

fn check_cfg(cfg: &RtImTrainConfig) -> Result<()> {
    let holdout_frac = cfg.window_holdout_frac;
    if !(0.0..=0.9).contains(&holdout_frac) {
        anyhow::bail!(
            "rt_im_train.window_holdout_frac must be in [0.0, 0.9], got {holdout_frac}; \
             0.0 disables held-out window sizing"
        );
    }
    if holdout_frac > 0.0 && cfg.adaptive_rt_window {
        anyhow::bail!(
            "rt_im_train.window_holdout_frac and rt_im_train.adaptive_rt_window are mutually \
             exclusive: the adaptive per-bin percentiles are in-sample and would silently undo \
             the held-out sizing; disable one of them"
        );
    }
    Ok(())
}

pub fn run(p: RtImTrainParams) -> Result<u64> {
    let t0 = Instant::now();
    // `--out` must not be one of this stage's own inputs: every input is read
    // before the output is published, so writing over one replaces it and exits 0
    // (docs/31 F6). The shared guard existed and was wired into two stages.
    for out in [p.out_windows, p.out_cal] {
        mumdia_io::refuse_output_over_input(
            out,
            &[
                ("--seed-psms", p.seed_psms),
                ("--lib-precursors", p.library_precursors),
            ],
        )?;
    }
    check_cfg(p.cfg)?;

    // Library predicted iRT, keyed by candidate_id (single source of truth, so
    // a patched/updated library iRT is used for both training and application).
    let lib = TableFile::open(p.library_precursors)?;
    let lib_cid = lib.u32("candidate_id")?;
    let lib_irt = lib.f32("predicted_irt")?;
    // The join map is built only when the anchors take their iRT from the library; with
    // `anchor_irt_from_seed` nothing reads it.
    let irt_by_cid: Option<HashMap<u32, f64>> = (!p.anchor_irt_from_seed).then(|| {
        let mut m: HashMap<u32, f64> = HashMap::with_capacity(lib.nrows);
        for i in 0..lib.nrows {
            m.insert(lib_cid[i], lib_irt[i] as f64);
        }
        m
    });
    let fit = fit_anchors(p.seed_psms, p.cfg, irt_by_cid.as_ref())?;
    drop(irt_by_cid);
    write_windows(
        &fit,
        &ApplyParams {
            library_precursors: p.library_precursors,
            out_windows: p.out_windows,
            out_cal: p.out_cal,
            cfg: p.cfg,
            config_hash: p.config_hash,
        },
        lib_cid,
        lib_irt,
        t0,
    )
}

/// Fit the calibration on a seed table whose own `predicted_irt` column carries the anchors'
/// iRT (`RtImTrainParams::anchor_irt_from_seed`): the fit a grouped run's bands share under
/// `groups.calibration = global`. No library is read.
pub fn fit_from_seed(seed_psms: &str, cfg: &RtImTrainConfig) -> Result<RtFit> {
    check_cfg(cfg)?;
    fit_anchors(seed_psms, cfg, None)
}

/// Write the windows of `p.library_precursors` and its `cal.json` from a fit made by
/// [`fit_from_seed`]. With the fit [`run`] would have made and the same table, this writes
/// what [`run`] writes, byte for byte (the report's `elapsed_ms` aside).
pub fn apply(fit: &RtFit, p: &ApplyParams) -> Result<u64> {
    let t0 = Instant::now();
    for out in [p.out_windows, p.out_cal] {
        mumdia_io::refuse_output_over_input(out, &[("--lib-precursors", p.library_precursors)])?;
    }
    let lib = TableFile::open(p.library_precursors)?;
    let lib_cid = lib.u32("candidate_id")?;
    let lib_irt = lib.f32("predicted_irt")?;
    write_windows(fit, p, lib_cid, lib_irt, t0)
}

/// The anchors and the fit. `irt_by_cid` joins each anchor's iRT from the library; `None`
/// reads it from the seed's own `predicted_irt` column.
fn fit_anchors(
    seed_psms: &str,
    cfg: &RtImTrainConfig,
    irt_by_cid: Option<&HashMap<u32, f64>>,
) -> Result<RtFit> {
    let holdout_frac = cfg.window_holdout_frac;
    // Training rows: confident seed PSMs, one apex (best score) per peptide;
    // predicted iRT is joined from the library by candidate_id.
    let seed = TableFile::open(seed_psms)?;
    let s_cid = seed.u32("candidate_id")?;
    let s_base = seed.u32("base_peptide_id")?;
    let s_q = seed.f64("spectrum_q")?;
    let s_score = seed.f64("score")?;
    let s_rt = seed.f64("observed_rt")?;
    let s_label = seed.str("label")?;
    let s_irt: Option<Vec<f32>> = if irt_by_cid.is_none() {
        Some(seed.f32("predicted_irt")?)
    } else {
        None
    };

    let mut best_per_pep: HashMap<u32, (f64, f64, f64)> = HashMap::new(); // base -> (score, irt, rt)
    for i in 0..seed.nrows {
        if !s_q[i].is_finite()
            || !s_score[i].is_finite()
            || !s_rt[i].is_finite()
            || s_q[i] >= cfg.q_train
        {
            continue;
        }
        // Only target PSMs may anchor the RT calibration; a decoy anchor injects a
        // random iRT<->RT pair into the fit.
        if s_label[i] != "target" {
            continue;
        }
        let irt = match (&s_irt, irt_by_cid) {
            (Some(col), _) => {
                let v = col[i] as f64;
                if !v.is_finite() {
                    continue;
                }
                v
            }
            (None, Some(map)) => match map.get(&s_cid[i]) {
                Some(v) if v.is_finite() => *v,
                None => continue,
                Some(_) => continue,
            },
            (None, None) => unreachable!("the seed column is read when there is no join map"),
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
        && matches!(cfg.calibration_method, CalibrationMethod::Loess)
        && n_train >= cfg.min_seed_for_calibration;
    let loess = if use_loess {
        Some(Loess::fit(&train_irt, &train_rt, cfg.loess_span, 200))
    } else {
        None
    };

    let mut fit = RtFit {
        n_train,
        calibration_available,
        slope,
        intercept,
        use_loess,
        loess,
        w_rt: None,
        status: String::new(),
        holdout_frac,
        holdout_sizing: None,
        adaptive: None,
        residuals: (f64::NAN, f64::NAN, f64::NAN),
    };

    // Residuals and RT window. Require enough anchors before trusting the
    // residual-percentile window: with only a handful of points a linear fit
    // passes ~exactly through them, so residuals ~0 and the window collapses to
    // the 1s floor (which then discards nearly every true co-elution). Below the
    // threshold, use the configured fixed fallback instead.
    let min_anchors = cfg.min_seed_for_calibration.max(2);
    let plan = window_plan(n_train, min_anchors, cfg.fallback_rt_window_s);

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
    let holdout_sizing: Option<HoldoutSizing> =
        if holdout_frac > 0.0 && plan == WindowPlan::Calibrated {
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
                let sizing_loess = use_loess.then(|| Loess::fit(&tr_x, &tr_y, cfg.loess_span, 200));
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
                let p_rt_s = percentile(&resid, cfg.p_rt);
                Some(HoldoutSizing {
                    width: (p_rt_s * cfg.rt_window_multiplier).max(1.0),
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
                        .map(|(x, y)| (y - fit.predict(*x)).abs())
                        .collect();
                    (percentile(&resid, cfg.p_rt) * cfg.rt_window_multiplier).max(1.0)
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
    let adaptive: Option<(f64, f64, Vec<f64>)> = if cfg.adaptive_rt_window && n_train >= min_anchors
    {
        let cals: Vec<f64> = train_irt.iter().map(|x| fit.predict(*x)).collect();
        let resid: Vec<f64> = cals
            .iter()
            .zip(&train_rt)
            .map(|(c, y)| (y - c).abs())
            .collect();
        let rt_min = cals.iter().cloned().fold(f64::INFINITY, f64::min);
        let rt_max = cals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let nb = cfg.adaptive_rt_bins.max(1);
        if rt_max > rt_min {
            let span = rt_max - rt_min;
            let mut per_bin: Vec<Vec<f64>> = vec![Vec::new(); nb];
            for (c, r) in cals.iter().zip(&resid) {
                let frac = ((c - rt_min) / span).clamp(0.0, 0.999_999);
                per_bin[(frac * nb as f64) as usize].push(*r);
            }
            let lo_clamp = cfg.rt_window_min_s.max(0.0);
            let hi_clamp = cfg.fallback_rt_window_s.max(lo_clamp);
            let widths: Vec<f64> = per_bin
                .iter()
                .map(|rs| {
                    if rs.is_empty() {
                        w_rt.expect("adaptive RT windows require a calibrated global width")
                    } else {
                        (percentile(rs, cfg.p_rt) * cfg.rt_window_multiplier)
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

    // RT calibration-quality residuals over the training anchors (seconds):
    // signed median = residual bias, absolute median = typical accuracy, MAD =
    // spread. Diagnostic only (the RT window already derives from these
    // residuals); surfaced so a run's RT calibration can be judged good or biased.
    let residuals = if calibration_available {
        let signed: Vec<f64> = train_irt
            .iter()
            .zip(&train_rt)
            .map(|(x, y)| y - fit.predict(*x))
            .collect();
        let med = percentile(&signed, 0.5);
        let absres: Vec<f64> = signed.iter().map(|r| r.abs()).collect();
        let mad: Vec<f64> = signed.iter().map(|r| (r - med).abs()).collect();
        (med, percentile(&absres, 0.5), percentile(&mad, 0.5))
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };

    fit.w_rt = w_rt;
    fit.status = status;
    fit.holdout_sizing = holdout_sizing;
    fit.adaptive = adaptive;
    fit.residuals = residuals;
    Ok(fit)
}

/// Apply `fit` to every row of one library table and write the windows, the `cal.json` and
/// the windows' report.
fn write_windows(
    fit: &RtFit,
    p: &ApplyParams,
    lib_cid: Vec<u32>,
    lib_irt: Vec<f32>,
    t0: Instant,
) -> Result<u64> {
    // Apply to every library candidate.
    let cid = lib_cid;
    let irt = lib_irt;
    let n = cid.len();
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
        let calibrated_rt =
            (fit.calibration_available && usable_irt).then(|| fit.predict(irt[i] as f64));
        let width = calibrated_rt.map(|cal| match &fit.adaptive {
            Some((rt_min, span, widths)) => {
                let nb = widths.len();
                let frac = ((cal - rt_min) / span).clamp(0.0, 0.999_999);
                widths[(frac * nb as f64) as usize]
            }
            None => fit
                .w_rt
                .expect("available RT calibration requires a bounded window"),
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

    let method = if !fit.calibration_available {
        "unavailable"
    } else if fit.use_loess {
        "loess"
    } else {
        "linear"
    };
    let slope_report = fit.calibration_available.then_some(fit.slope);
    let intercept_report = fit.calibration_available.then_some(fit.intercept);
    let (rt_residual_median_s, rt_residual_abs_median_s, rt_residual_mad_s) = fit.residuals;
    let holdout_sizing = &fit.holdout_sizing;
    let holdout_frac = fit.holdout_frac;
    let w_rt = fit.w_rt;
    let status = fit.status.as_str();
    let n_train = fit.n_train;

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
            "w_rt_sizing": match (holdout_sizing, holdout_frac > 0.0) {
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
            "rt-im-train: these candidates have no finite library iRT, so they get the \
             unbounded RT window rather than a calibrated one; a null predicted_irt reads \
             as NaN (docs/31 F4)"
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

    /// A seed of `n` confident target anchors (plus a decoy and an unconfident row, which
    /// the fit must skip) and a library of `n + 5` rows, both with iRT columns.
    fn anchors_and_library(dir: &std::path::Path, n: usize) -> (String, String) {
        let seed = dir.join("seed.parquet").to_str().unwrap().to_string();
        let lib = dir.join("lib.parquet").to_str().unwrap().to_string();
        let m = n + 2;
        let irt: Vec<f32> = (0..m).map(|i| (i as f32) * 0.7 + 3.0).collect();
        let rt: Vec<f64> = (0..m)
            .map(|i| 100.0 + 12.0 * (i as f64 * 0.7 + 3.0) + ((i * 37 % 11) as f64 - 5.0) * 4.0)
            .collect();
        let mut label = vec!["target".to_string(); m];
        label[n] = "decoy".to_string();
        let mut q = vec![0.001; m];
        q[n + 1] = 0.5;
        write_table(
            &seed,
            vec![
                Col::U32("candidate_id".into(), (0..m as u32).collect()),
                Col::U32("base_peptide_id".into(), (0..m as u32).collect()),
                Col::F64("spectrum_q".into(), q),
                Col::F64("score".into(), (0..m).map(|i| 10.0 + i as f64).collect()),
                Col::F64("observed_rt".into(), rt),
                Col::Str("label".into(), label),
                Col::F32("predicted_irt".into(), irt),
            ],
        )
        .unwrap();
        write_table(
            &lib,
            vec![
                Col::U32("candidate_id".into(), (0..(n + 5) as u32).collect()),
                Col::F32(
                    "predicted_irt".into(),
                    (0..n + 5)
                        .map(|i| if i == 3 { f32::NAN } else { i as f32 * 0.9 })
                        .collect(),
                ),
            ],
        )
        .unwrap();
        (seed, lib)
    }

    #[test]
    fn one_fit_applied_writes_what_the_whole_stage_writes() {
        // The grouped path fits once per run and applies the fit to every band; each band
        // used to run the whole stage. Same windows bytes and the same cal.json, under the
        // default, the held-out sizing, the adaptive window and a linear fit, and for too
        // few anchors.
        let dir = std::env::temp_dir().join(format!("mumdia_rt_fit_apply_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let holdout = RtImTrainConfig {
            window_holdout_frac: 0.3,
            ..Default::default()
        };
        let adaptive = RtImTrainConfig {
            adaptive_rt_window: true,
            ..Default::default()
        };
        let linear = RtImTrainConfig {
            calibration_method: CalibrationMethod::Linear,
            ..Default::default()
        };
        for (tag, cfg, n) in [
            ("default", RtImTrainConfig::default(), 400usize),
            ("holdout", holdout, 400),
            ("adaptive", adaptive, 400),
            ("linear", linear, 400),
            ("few", RtImTrainConfig::default(), 20),
            ("one", RtImTrainConfig::default(), 1),
        ] {
            let sub = dir.join(tag);
            std::fs::create_dir_all(&sub).unwrap();
            let (seed, lib) = anchors_and_library(&sub, n);
            let out = |name: &str| sub.join(name).to_str().unwrap().to_string();
            run(RtImTrainParams {
                seed_psms: &seed,
                library_precursors: &lib,
                out_windows: &out("w_run.parquet"),
                out_cal: &out("cal_run.json"),
                cfg: &cfg,
                config_hash: "h",
                anchor_irt_from_seed: true,
            })
            .unwrap();
            let fit = fit_from_seed(&seed, &cfg).unwrap();
            apply(
                &fit,
                &ApplyParams {
                    library_precursors: &lib,
                    out_windows: &out("w_apply.parquet"),
                    out_cal: &out("cal_apply.json"),
                    cfg: &cfg,
                    config_hash: "h",
                },
            )
            .unwrap();
            assert_eq!(
                std::fs::read(out("w_run.parquet")).unwrap(),
                std::fs::read(out("w_apply.parquet")).unwrap(),
                "{tag}: the windows differ"
            );
            assert_eq!(
                std::fs::read(out("cal_run.json")).unwrap(),
                std::fs::read(out("cal_apply.json")).unwrap(),
                "{tag}: cal.json differs"
            );
            assert_eq!(
                fit.n_train(),
                n,
                "{tag}: the decoy and the unconfident row are not anchors"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
