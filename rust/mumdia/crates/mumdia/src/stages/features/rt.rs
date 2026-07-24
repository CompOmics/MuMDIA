//! Extended feature family: rt (retention-time agreement).
//!
//! Retention-time agreement between the observed apex RT and the calibrated
//! predicted RT, plus gradient- and peak-width-normalized variants and a
//! profile-derived apex-RT comparison. Contract: `NAMES` and `values(&Evidence)`
//! return the same number of items in the same order. Names are globally unique,
//! snake_case, and append-only (part of the frozen extended feature schema). See
//! the `Evidence` struct in `stages/features.rs` for available evidence and the
//! parent helpers `super::{mean, normalize_sum, best_xcorr, smooth3, peak_bounds}`.
use super::Evidence;

pub const NAMES: &[&str] = &[
    "rt_error_signed",
    "rt_error_abs",
    "rt_error_squared",
    "rt_error_signed_norm_gradient",
    "rt_error_abs_norm_gradient",
    "observed_rt_raw",
    "predicted_rt_raw",
    "observed_rt_fraction",
    "predicted_rt_fraction",
    "rt_error_over_peak_width",
    "rt_error_over_fwhm",
    "rt_diff_profile_apex",
    "predicted_rt_in_gradient",
];

/// Replace non-finite values with 0.0.
fn finite(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// RT width (seconds) of the peak spanned by `prof` at `frac` of apex height,
/// measured against `axis`. Returns 0.0 on any degenerate input.
fn rt_width(prof: &[f64], axis: &[f64], apex_idx: usize, frac: f64) -> f64 {
    let n = prof.len();
    if n < 3 || axis.len() != n {
        return 0.0;
    }
    let ai = apex_idx.min(n - 1);
    let (lo, hi) = super::peak_bounds(prof, ai, frac, 0);
    if lo >= n || hi >= n || hi < lo {
        return 0.0;
    }
    let w = axis[hi] - axis[lo];
    if w > 0.0 {
        finite(w)
    } else {
        0.0
    }
}

/// RT (seconds) at the argmax of the predicted-intensity-weighted reference
/// profile built over `axis_full`. Returns `None` if not computable.
fn profile_apex_rt_full(e: &Evidence) -> Option<f64> {
    let t = e.axis_full.len();
    if t == 0 || e.traces_full.is_empty() {
        return None;
    }
    let k = e.traces_full.len();
    // Weight each fragment trace by its predicted intensity; fall back to an
    // unweighted sum when no positive weight is available.
    let mut have_weight = false;
    let mut prof = vec![0.0f64; t];
    for (i, trace) in e.traces_full.iter().enumerate().take(k) {
        if trace.len() != t {
            continue;
        }
        let w = e.pred.get(i).copied().unwrap_or(0.0);
        if w > 0.0 {
            have_weight = true;
            for (j, value) in prof.iter_mut().enumerate().take(t) {
                *value += w * trace[j];
            }
        }
    }
    if !have_weight {
        // Unweighted fallback.
        prof.fill(0.0);
        for trace in e.traces_full.iter().take(k) {
            if trace.len() != t {
                continue;
            }
            for (j, value) in prof.iter_mut().enumerate().take(t) {
                *value += trace[j];
            }
        }
    }
    // argmax
    let mut best = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    let mut any = false;
    for (j, &value) in prof.iter().enumerate().take(t) {
        if value > best_v {
            best_v = value;
            best = j;
            any = true;
        }
    }
    if !any || best_v <= 0.0 {
        return None;
    }
    Some(e.axis_full[best])
}

pub fn values(e: &Evidence) -> Vec<f64> {
    let signed = e.apex_rt - e.rt_pred_cal;
    let abs = signed.abs();
    let sq = signed * signed;

    let grad = e.gradient;
    let (signed_ng, abs_ng, obs_frac, pred_frac) = if grad > 0.0 {
        (
            signed / grad,
            abs / grad,
            e.apex_rt / grad,
            e.rt_pred_cal / grad,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    // Peak-width normalized RT error: base width (near-baseline) and FWHM,
    // both from the reference elution profile over the peak-bounded axis.
    let base_width = rt_width(&e.ref_profile, &e.axis, e.apex_idx, 0.1);
    let fwhm = rt_width(&e.ref_profile, &e.axis, e.apex_idx, 0.5);
    let over_width = if base_width > 0.0 {
        abs / base_width
    } else {
        0.0
    };
    let over_fwhm = if fwhm > 0.0 { abs / fwhm } else { 0.0 };

    let profile_apex = match profile_apex_rt_full(e) {
        Some(rt) => (rt - e.rt_pred_cal).abs(),
        None => 0.0,
    };

    let in_gradient = if grad > 0.0 && e.rt_pred_cal >= 0.0 && e.rt_pred_cal <= grad {
        1.0
    } else {
        0.0
    };

    vec![
        finite(signed),
        finite(abs),
        finite(sq),
        finite(signed_ng),
        finite(abs_ng),
        finite(e.apex_rt),
        finite(e.rt_pred_cal),
        finite(obs_frac),
        finite(pred_frac),
        finite(over_width),
        finite(over_fwhm),
        finite(profile_apex),
        in_gradient,
    ]
}
