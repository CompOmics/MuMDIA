//! Extended feature family: chromatographic.
//!
//! Peak-shape quality of the reference elution profile R (predicted-intensity-
//! weighted sum of fragment XICs, peak-restricted unless the full window is
//! named) and of the individual fragment traces. Gaussian fits are by moment
//! matching (mu, sigma from intensity-weighted RT moments); the EMG comparison
//! uses a small grid over the exponential time constant with a least-squares
//! amplitude. Sources: OpenSWATH / AlphaDIA / MetaClean / DIA-NN / USP peak
//! descriptors.
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items
//! in the same order. Names are globally unique, snake_case, and append-only
//! (they are part of the frozen extended feature schema). See the `Evidence`
//! struct in `stages/features.rs` for the available per-PSM evidence, and use
//! the shared kernels `crate::stats::{pearson, cosine, spectral_angle}` plus the
//! parent helpers `super::{mean, normalize_sum, best_xcorr, smooth3}`.
use super::Evidence;
use crate::stats::cosine;

pub const NAMES: &[&str] = &[
    "gaussian_fit_r2",
    "gaussian_cosine",
    "emg_fit_improvement",
    "apex_prominence",
    "profile_peak_snr",
    "fwhm_seconds",
    "fwhm_to_window_ratio",
    "width_at_10pct",
    "width_ratio_10_50",
    "hwhm_asymmetry",
    "tailing_factor_usp",
    "asymmetry_factor_10pct",
    "apex_sharpness",
    "apex_curvature",
    "apex_to_boundary_ratio",
    "apex_dominance",
    "zigzag_index",
    "jaggedness",
    "roughness_2nd_deriv",
    "n_local_maxima",
    "modality",
    "rt_skewness",
    "rt_excess_kurtosis",
    "rt_std_seconds",
    "mean_mode_offset",
    "fraction_area_within_fwhm",
    "triangle_area_similarity",
    "baseline_fraction",
    "peak_completeness",
    "apex_centering_offset",
    "intensity_score",
    "total_xic_log",
    "frag_fwhm_cv",
    "frag_fwhm_mean",
    "frag_apex_rt_dispersion",
    "frag_apex_rt_dispersion_weighted",
    "frag_apex_offset_from_profile_mean",
    "frag_gaussianity_mean",
    "frag_gaussianity_weighted",
    "frag_zigzag_mean",
    "sumtrace_unweighted_gaussian_r2",
    "reference_profile_rt_entropy_peak",
    "reference_profile_rt_entropy_ratio",
];

const EPS: f64 = 1e-12;

pub fn values(e: &Evidence) -> Vec<f64> {
    let n_names = NAMES.len();
    let tp = e.axis.len();
    // Degenerate guard: no time axis -> all zeros.
    if tp == 0 {
        return vec![0.0; n_names];
    }
    let x = &e.axis; // RT (s) over the elution peak
    let k = e.traces.len();

    // Reference elution profile over the peak, rebuilt here with a fallback to
    // an unweighted fragment sum when no predicted intensities are available,
    // so the peak-shape features remain defined even without a library.
    let r = weighted_profile(&e.traces, &e.pred, tp);
    let a_peak = max_of(&r);
    let apex_r = argmax(&r);
    let apex_rt = x[apex_r];

    // Full-window reference profile for baseline / out-of-peak statistics.
    let tf = e.axis_full.len();
    let r_full = weighted_profile(&e.traces_full, &e.pred, tf);

    // Unweighted sum trace over the peak.
    let usum: Vec<f64> = (0..tp)
        .map(|t| {
            e.traces
                .iter()
                .map(|tr| tr.get(t).copied().unwrap_or(0.0))
                .sum::<f64>()
        })
        .collect();

    // --- Gaussian moment-match fit of R over the peak ---
    let (gfit, gr2) = gaussian_fit(x, &r);
    let gaussian_fit_r2 = gr2.clamp(0.0, 1.0);
    let gaussian_cosine = cosine(&r, &gfit);

    // --- EMG improvement over the Gaussian fit ---
    let emg_fit_improvement = emg_improvement(x, &r, &gfit);

    // --- apex prominence ---
    let apex_prominence = if a_peak.abs() > EPS {
        (a_peak - min_of(&r)) / (a_peak + EPS)
    } else {
        0.0
    };

    // --- profile peak S/N via MAD of R_full over out-of-peak scans ---
    let profile_peak_snr = peak_snr(a_peak, &e.axis_full, &r_full, x[0], x[tp - 1]);

    // --- width descriptors on R over the RT axis ---
    let (l50, r50, _) = crossings(x, &r, 0.5);
    let (l10, r10, _) = crossings(x, &r, 0.1);
    let (l05, r05, _) = crossings(x, &r, 0.05);
    let fwhm_seconds = (r50 - l50).max(0.0);
    let width_at_10pct = (r10 - l10).max(0.0);
    let width5 = (r05 - l05).max(0.0);

    let window_span = if tf >= 2 {
        (e.axis_full[tf - 1] - e.axis_full[0]).max(0.0)
    } else {
        (x[tp - 1] - x[0]).max(0.0)
    };
    let fwhm_to_window_ratio = if window_span > EPS {
        fwhm_seconds / window_span
    } else {
        0.0
    };
    let width_ratio_10_50 = if fwhm_seconds > EPS {
        width_at_10pct / fwhm_seconds
    } else {
        0.0
    };

    let left_hw = (apex_rt - l50).max(0.0);
    let right_hw = (r50 - apex_rt).max(0.0);
    let hwhm_asymmetry = if left_hw + right_hw > EPS {
        (right_hw - left_hw) / (right_hw + left_hw)
    } else {
        0.0
    };

    // USP tailing: W05 / (2 f), f = apex-to-left-5% distance.
    let f_left5 = (apex_rt - l05).max(0.0);
    let tailing_factor_usp = if f_left5 > EPS {
        width5 / (2.0 * f_left5)
    } else {
        0.0
    };

    // USP asymmetry at 10%: (right - apex) / (apex - left).
    let left10 = (apex_rt - l10).max(0.0);
    let right10 = (r10 - apex_rt).max(0.0);
    let asymmetry_factor_10pct = if left10 > EPS { right10 / left10 } else { 0.0 };

    // --- apex sharpness / curvature over FWHM boundary indices ---
    let (lb, ab, rb) = halfmax_bounds(&r);
    let apex_sharpness = {
        let mut terms = Vec::new();
        if ab > lb {
            terms.push((a_peak - r[lb]) / ((ab - lb) as f64));
        }
        if rb > ab {
            terms.push((a_peak - r[rb]) / ((rb - ab) as f64));
        }
        if !terms.is_empty() && a_peak > EPS {
            (terms.iter().sum::<f64>() / terms.len() as f64) / a_peak
        } else {
            0.0
        }
    };
    let apex_curvature = if a_peak > EPS && apex_r >= 1 && apex_r + 1 < tp {
        (2.0 * a_peak - r[apex_r - 1] - r[apex_r + 1]) / a_peak
    } else {
        0.0
    };

    let apex_to_boundary_ratio = {
        let edge = r[0].max(r[tp - 1]);
        if edge > EPS {
            // Bounded: a near-zero boundary otherwise sends this to ~1e16. An apex
            // more than 1000x the peak boundary is already fully isolated; the
            // distinction beyond that carries no information for the rescorer.
            (a_peak / edge).min(1000.0)
        } else {
            0.0
        }
    };

    let apex_dominance = {
        let mfull = super::mean(&r_full);
        if mfull > EPS {
            a_peak / mfull
        } else {
            0.0
        }
    };

    // --- roughness family ---
    let zigzag_index = zigzag(&r);
    let jaggedness = jaggedness(&r);
    let roughness_2nd_deriv = roughness2(&r);

    // --- multimodality on R_full ---
    let (n_local_maxima, modality) = maxima_stats(&r_full);

    // --- RT moments (baseline-subtracted) ---
    let (centroid, rt_std, rt_skew, rt_kurt) = moments(x, &r);
    let rt_skewness = rt_skew;
    let rt_excess_kurtosis = rt_kurt;
    let rt_std_seconds = rt_std;
    let mean_mode_offset = if fwhm_seconds > EPS {
        (centroid - apex_rt) / fwhm_seconds
    } else {
        0.0
    };

    // --- area descriptors ---
    let total_area = trapz(x, &r);
    let fraction_area_within_fwhm = if total_area > EPS {
        let a_in = trapz_between(x, &r, l50, r50);
        (a_in / total_area).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let triangle_area_similarity = {
        let tri = 0.5 * a_peak * width5;
        if tri > EPS {
            (total_area - tri).abs() / (tri + EPS)
        } else {
            0.0
        }
    };

    // --- baseline / completeness / centering ---
    let baseline_fraction = {
        if tf > 0 {
            let base = min_of(&r_full);
            let amax = max_of(&r_full);
            let thr = base + 0.1 * (amax - base);
            r_full.iter().filter(|&&v| v < thr).count() as f64 / tf as f64
        } else {
            0.0
        }
    };
    let peak_completeness = {
        let base = min_of(&r);
        let denom = a_peak - base;
        if denom > EPS {
            let left = (a_peak - r[0]) / denom;
            let right = (a_peak - r[tp - 1]) / denom;
            left.min(right).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let apex_centering_offset = if tf > 1 {
        let center = (tf as f64 - 1.0) / 2.0;
        (argmax(&r_full) as f64 - center).abs() / tf as f64
    } else {
        0.0
    };

    // --- integrated intensity ---
    let sum_peak: f64 = e.traces.iter().flatten().sum();
    let sum_full: f64 = e.traces_full.iter().flatten().sum();
    let intensity_score = if sum_full > EPS {
        (sum_peak / sum_full).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let total_xic_log = (1.0 + sum_peak.max(0.0)).ln();

    // --- per-fragment descriptors over matched fragments ---
    let mut frag_fwhm: Vec<f64> = Vec::new();
    let mut frag_apex_rt: Vec<f64> = Vec::new();
    let mut frag_gauss: Vec<f64> = Vec::new();
    let mut frag_zig: Vec<f64> = Vec::new();
    let mut frag_w: Vec<f64> = Vec::new();
    for fi in 0..k {
        let tr = &e.traces[fi];
        if tr.len() != tp || max_of(tr) <= 0.0 {
            continue;
        }
        let (fl, fr, fa) = crossings(x, tr, 0.5);
        frag_fwhm.push((fr - fl).max(0.0));
        frag_apex_rt.push(x[fa]);
        let (_, r2) = gaussian_fit(x, tr);
        frag_gauss.push(r2.clamp(0.0, 1.0));
        frag_zig.push(zigzag(tr));
        frag_w.push(e.pred.get(fi).copied().unwrap_or(0.0).max(0.0));
    }
    let frag_fwhm_mean = super::mean(&frag_fwhm);
    let frag_fwhm_cv = if frag_fwhm_mean > EPS {
        std_dev(&frag_fwhm) / frag_fwhm_mean
    } else {
        0.0
    };
    let frag_apex_rt_dispersion = std_dev(&frag_apex_rt);
    let frag_apex_rt_dispersion_weighted = weighted_std(&frag_apex_rt, &frag_w);
    let frag_apex_offset_from_profile_mean = if frag_apex_rt.is_empty() {
        0.0
    } else {
        frag_apex_rt
            .iter()
            .map(|&t| (t - apex_rt).abs())
            .sum::<f64>()
            / frag_apex_rt.len() as f64
    };
    let frag_gaussianity_mean = super::mean(&frag_gauss);
    let frag_gaussianity_weighted = weighted_mean(&frag_gauss, &frag_w);
    let frag_zigzag_mean = super::mean(&frag_zig);

    // --- unweighted sum-trace Gaussianity ---
    let (_, ur2) = gaussian_fit(x, &usum);
    let sumtrace_unweighted_gaussian_r2 = ur2.clamp(0.0, 1.0);

    // --- RT entropy of the reference profile ---
    let reference_profile_rt_entropy_peak = entropy(&r);
    let ent_full = entropy(&r_full);
    let reference_profile_rt_entropy_ratio = if ent_full > EPS {
        reference_profile_rt_entropy_peak / ent_full
    } else {
        0.0
    };

    let out = vec![
        gaussian_fit_r2,
        gaussian_cosine,
        emg_fit_improvement,
        apex_prominence,
        profile_peak_snr,
        fwhm_seconds,
        fwhm_to_window_ratio,
        width_at_10pct,
        width_ratio_10_50,
        hwhm_asymmetry,
        tailing_factor_usp,
        asymmetry_factor_10pct,
        apex_sharpness,
        apex_curvature,
        apex_to_boundary_ratio,
        apex_dominance,
        zigzag_index,
        jaggedness,
        roughness_2nd_deriv,
        n_local_maxima,
        modality,
        rt_skewness,
        rt_excess_kurtosis,
        rt_std_seconds,
        mean_mode_offset,
        fraction_area_within_fwhm,
        triangle_area_similarity,
        baseline_fraction,
        peak_completeness,
        apex_centering_offset,
        intensity_score,
        total_xic_log,
        frag_fwhm_cv,
        frag_fwhm_mean,
        frag_apex_rt_dispersion,
        frag_apex_rt_dispersion_weighted,
        frag_apex_offset_from_profile_mean,
        frag_gaussianity_mean,
        frag_gaussianity_weighted,
        frag_zigzag_mean,
        sumtrace_unweighted_gaussian_r2,
        reference_profile_rt_entropy_peak,
        reference_profile_rt_entropy_ratio,
    ];
    debug_assert_eq!(out.len(), n_names);
    // Final finiteness guard.
    out.into_iter()
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .collect()
}

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

/// Predicted-intensity-weighted sum of the fragment traces, falling back to an
/// unweighted sum when the total predicted weight is zero.
fn weighted_profile(traces: &[Vec<f64>], pred: &[f64], t: usize) -> Vec<f64> {
    let mut r = vec![0.0f64; t];
    if t == 0 {
        return r;
    }
    let wts: Vec<f64> = (0..traces.len())
        .map(|i| pred.get(i).copied().unwrap_or(0.0).max(0.0))
        .collect();
    let sumw: f64 = wts.iter().sum();
    for (i, tr) in traces.iter().enumerate() {
        let w = if sumw > 0.0 { wts[i] } else { 1.0 };
        if w == 0.0 {
            continue;
        }
        for k in 0..t.min(tr.len()) {
            r[k] += w * tr[k];
        }
    }
    r
}

fn argmax(v: &[f64]) -> usize {
    let mut bi = 0;
    let mut bv = f64::MIN;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

fn max_of(v: &[f64]) -> f64 {
    v.iter()
        .cloned()
        .fold(f64::MIN, f64::max)
        .clamp(0.0, f64::MAX)
}

fn min_of(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().cloned().fold(f64::MAX, f64::min)
    }
}

fn std_dev(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let m = v.iter().sum::<f64>() / n as f64;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
    if var > 0.0 {
        var.sqrt()
    } else {
        0.0
    }
}

fn weighted_mean(vals: &[f64], w: &[f64]) -> f64 {
    let sw: f64 = w.iter().take(vals.len()).sum();
    if sw <= 0.0 {
        return super::mean(vals);
    }
    vals.iter().zip(w).map(|(v, wi)| v * wi).sum::<f64>() / sw
}

fn weighted_std(vals: &[f64], w: &[f64]) -> f64 {
    let sw: f64 = w.iter().take(vals.len()).sum();
    if sw <= 0.0 {
        return std_dev(vals);
    }
    let m = vals.iter().zip(w).map(|(v, wi)| v * wi).sum::<f64>() / sw;
    let var = vals
        .iter()
        .zip(w)
        .map(|(v, wi)| wi * (v - m) * (v - m))
        .sum::<f64>()
        / sw;
    if var > 0.0 {
        var.sqrt()
    } else {
        0.0
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn trapz(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n - 1 {
        s += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]);
    }
    s.max(0.0)
}

/// Trapezoidal area of `y` over the RT points whose axis value lies within
/// `[lo, hi]`.
fn trapz_between(x: &[f64], y: &[f64], lo: f64, hi: f64) -> f64 {
    let n = x.len().min(y.len());
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..n {
        if x[i] >= lo && x[i] <= hi {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    trapz(&xs, &ys)
}

/// Shannon entropy of a non-negative profile normalized to sum 1 over RT.
fn entropy(y: &[f64]) -> f64 {
    let s: f64 = y.iter().filter(|v| **v > 0.0).sum();
    if s <= 0.0 {
        return 0.0;
    }
    let mut e = 0.0;
    for &v in y {
        if v > 0.0 {
            let p = v / s;
            e -= p * p.ln();
        }
    }
    if e.is_finite() {
        e.max(0.0)
    } else {
        0.0
    }
}

/// Coefficient of determination of `fit` against `y`.
fn r_squared(y: &[f64], fit: &[f64]) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    let ybar = y.iter().sum::<f64>() / n as f64;
    let sst: f64 = y.iter().map(|v| (v - ybar) * (v - ybar)).sum();
    if sst <= EPS {
        return 0.0;
    }
    let ssr: f64 = y.iter().zip(fit).map(|(a, b)| (a - b) * (a - b)).sum();
    let r = 1.0 - ssr / sst;
    if r.is_finite() {
        r
    } else {
        0.0
    }
}

fn rmse(y: &[f64], fit: &[f64]) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    let s: f64 = y.iter().zip(fit).map(|(a, b)| (a - b) * (a - b)).sum();
    (s / n as f64).sqrt()
}

/// Moment-matched Gaussian fit A*exp(-(x-mu)^2/2sigma^2). Returns (fitted, R^2).
fn gaussian_fit(x: &[f64], y: &[f64]) -> (Vec<f64>, f64) {
    let n = y.len();
    if n < 2 || x.len() < n {
        return (vec![0.0; n], 0.0);
    }
    let w: Vec<f64> = y.iter().map(|v| v.max(0.0)).collect();
    let sw: f64 = w.iter().sum();
    if sw <= EPS {
        return (vec![0.0; n], 0.0);
    }
    let mu = x.iter().zip(&w).map(|(xi, wi)| xi * wi).sum::<f64>() / sw;
    let var = x
        .iter()
        .zip(&w)
        .map(|(xi, wi)| wi * (xi - mu) * (xi - mu))
        .sum::<f64>()
        / sw;
    let span = x[n - 1] - x[0];
    let floor = if span > 0.0 {
        (span / n as f64).max(1e-6)
    } else {
        1e-6
    };
    let s = var.sqrt();
    let sigma = if s.is_finite() && s > floor { s } else { floor };
    let g: Vec<f64> = x
        .iter()
        .map(|xi| {
            let z = (xi - mu) / sigma;
            (-0.5 * z * z).exp()
        })
        .collect();
    let gg: f64 = g.iter().map(|v| v * v).sum();
    let a = if gg > EPS {
        y.iter().zip(&g).map(|(yi, gi)| yi * gi).sum::<f64>() / gg
    } else {
        0.0
    };
    let fit: Vec<f64> = g.iter().map(|gi| a * gi).collect();
    (fit.clone(), r_squared(y, &fit))
}

/// Complementary error function via Abramowitz-Stegun 7.1.26.
fn erfc(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-ax * ax).exp();
    let erf = sign * y;
    (1.0 - erf).clamp(0.0, 2.0)
}

/// Fraction improvement in RMSE from a 4-parameter EMG over the Gaussian fit,
/// clamped to [0,1]. The EMG time constant is grid-searched; the amplitude is
/// least-squares per grid point. Gaussian mu/sigma from moments of `y`.
fn emg_improvement(x: &[f64], y: &[f64], gfit: &[f64]) -> f64 {
    let n = y.len();
    if n < 3 {
        return 0.0;
    }
    let rmse_g = rmse(y, gfit);
    if rmse_g <= EPS {
        return 0.0;
    }
    let w: Vec<f64> = y.iter().map(|v| v.max(0.0)).collect();
    let sw: f64 = w.iter().sum();
    if sw <= EPS {
        return 0.0;
    }
    let mu = x.iter().zip(&w).map(|(xi, wi)| xi * wi).sum::<f64>() / sw;
    let var = x
        .iter()
        .zip(&w)
        .map(|(xi, wi)| wi * (xi - mu) * (xi - mu))
        .sum::<f64>()
        / sw;
    let span = x[n - 1] - x[0];
    let floor = if span > 0.0 {
        (span / n as f64).max(1e-6)
    } else {
        1e-6
    };
    let s = var.sqrt();
    let sigma = if s.is_finite() && s > floor { s } else { floor };

    let mut best = rmse_g;
    for &frac in &[0.1f64, 0.25, 0.5, 1.0, 2.0, 4.0] {
        let tau = frac * sigma;
        if tau <= EPS {
            continue;
        }
        let shape = emg_shape(x, mu, sigma, tau);
        let ss: f64 = shape.iter().map(|v| v * v).sum();
        if ss <= EPS {
            continue;
        }
        let a = y.iter().zip(&shape).map(|(yi, si)| yi * si).sum::<f64>() / ss;
        let fit: Vec<f64> = shape.iter().map(|si| a * si).collect();
        let e = rmse(y, &fit);
        if e.is_finite() && e < best {
            best = e;
        }
    }
    let imp = (rmse_g - best) / (rmse_g + EPS);
    imp.clamp(0.0, 1.0)
}

/// Unit-amplitude exponentially-modified Gaussian shape (numerically guarded).
fn emg_shape(x: &[f64], mu: f64, sigma: f64, tau: f64) -> Vec<f64> {
    let sqrt2 = std::f64::consts::SQRT_2;
    x.iter()
        .map(|&t| {
            let arg = (0.5 * (sigma / tau) * (sigma / tau) + (mu - t) / tau).clamp(-30.0, 30.0);
            let z = (mu - t) / (sqrt2 * sigma) + sigma / (sqrt2 * tau);
            let v = arg.exp() * erfc(z);
            if v.is_finite() {
                v.max(0.0)
            } else {
                0.0
            }
        })
        .collect()
}

/// Peak S/N: max peak profile / (1.4826 * MAD of the full profile over scans
/// outside the peak RT bounds). Falls back to unit noise when no out-of-peak
/// scans exist or the MAD is zero.
fn peak_snr(a_peak: f64, axis_full: &[f64], r_full: &[f64], lo: f64, hi: f64) -> f64 {
    let n = axis_full.len().min(r_full.len());
    let out: Vec<f64> = (0..n)
        .filter(|&i| axis_full[i] < lo || axis_full[i] > hi)
        .map(|i| r_full[i])
        .collect();
    if out.len() < 2 {
        return 0.0;
    }
    let med = median(out.clone());
    let mad = median(out.iter().map(|v| (v - med).abs()).collect());
    let noise = 1.4826 * mad;
    let denom = if noise > EPS { noise } else { 1.0 };
    // Bounded: a near-flat out-of-peak baseline gives noise ~ 1e-9 and otherwise
    // sends S/N to ~1e15. A ratio above 1000 already means "essentially noiseless".
    (a_peak / denom).clamp(0.0, 1000.0)
}

/// Fractional-height crossings on `y` over the RT axis `x` by linear
/// interpolation, walking outward from the apex. Returns (left_rt, right_rt,
/// apex_index).
fn crossings(x: &[f64], y: &[f64], frac: f64) -> (f64, f64, usize) {
    let n = y.len();
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    let apex = argmax(y);
    if n < 2 || y[apex] <= 0.0 {
        return (x[apex], x[apex], apex);
    }
    let thr = frac * y[apex];
    // left crossing
    let mut left = x[0];
    let mut i = apex;
    while i > 0 {
        if y[i - 1] < thr {
            let (y0, y1) = (y[i - 1], y[i]);
            let (x0, x1) = (x[i - 1], x[i]);
            left = if (y1 - y0).abs() > EPS {
                x0 + (thr - y0) / (y1 - y0) * (x1 - x0)
            } else {
                x0
            };
            break;
        }
        i -= 1;
    }
    // right crossing
    let mut right = x[n - 1];
    let mut j = apex;
    while j + 1 < n {
        if y[j + 1] < thr {
            let (y0, y1) = (y[j], y[j + 1]);
            let (x0, x1) = (x[j], x[j + 1]);
            right = if (y1 - y0).abs() > EPS {
                x0 + (thr - y0) / (y1 - y0) * (x1 - x0)
            } else {
                x1
            };
            break;
        }
        j += 1;
    }
    (left, right, apex)
}

/// Half-max boundary indices (inner points still >= 0.5*max) and apex index.
fn halfmax_bounds(y: &[f64]) -> (usize, usize, usize) {
    let n = y.len();
    if n == 0 {
        return (0, 0, 0);
    }
    let a = argmax(y);
    if n < 2 || y[a] <= 0.0 {
        return (0, a, n.saturating_sub(1));
    }
    let thr = 0.5 * y[a];
    let mut l = a;
    while l > 0 && y[l - 1] >= thr {
        l -= 1;
    }
    let mut r = a;
    while r + 1 < n && y[r + 1] >= thr {
        r += 1;
    }
    (l, a, r)
}

/// MetaClean zig-zag roughness index of a profile.
fn zigzag(y: &[f64]) -> f64 {
    let n = y.len();
    if n < 3 {
        return 0.0;
    }
    let a = max_of(y);
    if a <= EPS {
        return 0.0;
    }
    let mut s = 0.0;
    for t in 1..n - 1 {
        let d = 2.0 * y[t] - y[t - 1] - y[t + 1];
        s += d * d;
    }
    let v = s / (n as f64 * a * a);
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Jaggedness: sign changes in the first difference, minus one, per interior gap.
fn jaggedness(y: &[f64]) -> f64 {
    let n = y.len();
    if n < 3 {
        return 0.0;
    }
    let mut changes: i64 = 0;
    let mut prev = 0.0f64;
    let mut have = false;
    for i in 1..n {
        let d = y[i] - y[i - 1];
        if d == 0.0 {
            continue;
        }
        let s = d.signum();
        if have && s != prev {
            changes += 1;
        }
        prev = s;
        have = true;
    }
    let v = (changes - 1).max(0) as f64 / (n as f64 - 2.0);
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Second-derivative energy normalized by signal energy.
fn roughness2(y: &[f64]) -> f64 {
    let n = y.len();
    if n < 3 {
        return 0.0;
    }
    let mut num = 0.0;
    for t in 1..n - 1 {
        let d = y[t + 1] - 2.0 * y[t] + y[t - 1];
        num += d * d;
    }
    let den: f64 = y.iter().map(|v| v * v).sum();
    if den > EPS {
        let v = num / den;
        if v.is_finite() {
            v
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// Baseline-subtracted RT moments of a profile: (centroid, std, skewness,
/// excess kurtosis).
fn moments(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    let n = y.len();
    if n < 2 || x.len() < n {
        return (if n > 0 { x[0] } else { 0.0 }, 0.0, 0.0, 0.0);
    }
    let base = min_of(y);
    let w: Vec<f64> = y.iter().map(|v| (v - base).max(0.0)).collect();
    let sw: f64 = w.iter().sum();
    if sw <= EPS {
        return (x[argmax(y)], 0.0, 0.0, 0.0);
    }
    let c = x.iter().zip(&w).map(|(xi, wi)| xi * wi).sum::<f64>() / sw;
    let m2 = x
        .iter()
        .zip(&w)
        .map(|(xi, wi)| wi * (xi - c).powi(2))
        .sum::<f64>()
        / sw;
    let std = if m2 > 0.0 { m2.sqrt() } else { 0.0 };
    if std <= EPS {
        return (c, std, 0.0, 0.0);
    }
    let m3 = x
        .iter()
        .zip(&w)
        .map(|(xi, wi)| wi * (xi - c).powi(3))
        .sum::<f64>()
        / sw;
    let m4 = x
        .iter()
        .zip(&w)
        .map(|(xi, wi)| wi * (xi - c).powi(4))
        .sum::<f64>()
        / sw;
    let skew = m3 / std.powi(3);
    let kurt = m4 / std.powi(4) - 3.0;
    (
        c,
        std,
        if skew.is_finite() { skew } else { 0.0 },
        if kurt.is_finite() { kurt } else { 0.0 },
    )
}

/// Multimodality descriptors on the full profile: (count of qualifying local
/// maxima, modality valley depth between the two highest maxima).
fn maxima_stats(y: &[f64]) -> (f64, f64) {
    let n = y.len();
    if n < 3 {
        return (0.0, 0.0);
    }
    let amax = max_of(y);
    if amax <= EPS {
        return (0.0, 0.0);
    }
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..n - 1 {
        if y[i] > y[i - 1] && y[i] > y[i + 1] {
            peaks.push(i);
        }
    }
    // Count peaks above 0.1 relative height with prominence > 0.1*amax.
    let count = peaks
        .iter()
        .filter(|&&i| y[i] > 0.1 * amax && prominence(y, i) > 0.1 * amax)
        .count() as f64;

    // Modality: two highest maxima, valley between them.
    let modality = if peaks.len() < 2 {
        0.0
    } else {
        let mut sorted = peaks.clone();
        sorted.sort_by(|&a, &b| y[b].total_cmp(&y[a]));
        let (p1, p2) = (sorted[0], sorted[1]);
        let (lo, hi) = if p1 < p2 { (p1, p2) } else { (p2, p1) };
        let valley = y[lo..=hi].iter().cloned().fold(f64::MAX, f64::min);
        let minpk = y[p1].min(y[p2]);
        if minpk > EPS {
            ((minpk - valley) / minpk).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    (count, modality)
}

/// Topographic prominence of the local maximum at index `i`.
fn prominence(y: &[f64], i: usize) -> f64 {
    let n = y.len();
    let h = y[i];
    let mut lmin = h;
    let mut j = i;
    while j > 0 {
        j -= 1;
        if y[j] > h {
            break;
        }
        if y[j] < lmin {
            lmin = y[j];
        }
    }
    let mut rmin = h;
    let mut k = i;
    while k + 1 < n {
        k += 1;
        if y[k] > h {
            break;
        }
        if y[k] < rmin {
            rmin = y[k];
        }
    }
    (h - lmin.max(rmin)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(axis: Vec<f64>, traces: Vec<Vec<f64>>, pred: Vec<f64>) -> Evidence {
        let tp = axis.len();
        Evidence {
            axis: axis.clone(),
            traces: traces.clone(),
            axis_full: axis,
            traces_full: traces,
            pred,
            obs_apex: vec![],
            is_b: vec![],
            ordinal: vec![],
            frag_charge: vec![],
            frag_mz: vec![],
            frag_obs_mz: vec![],
            mass_err_ppm: vec![],
            apex_idx: tp / 2,
            ref_profile: vec![],
            apex_rt: 0.0,
            rt_pred_cal: 0.0,
            rt_err: 0.0,
            gradient: 1.0,
            precursor_mz: 0.0,
            charge: 2,
            seq_len: 8,
            n_matched: 0,
            n_predicted: 0,
            seed_score: 0.0,
            seed_identified: 0.0,
            apex_intensity: 0.0,
            ms1_mono: None,
            ms1_iso1: None,
            ms1_iso2: None,
            ms1_isom1: None,
            ms1_xic: vec![],
            ms1_precursor_features: false,
            deconv_explained: 0.0,
            deconv_active: 0.0,
            deconv_share: 0.0,
            deconv_max_collin: 0.0,
            deconv_shadow: 0.0,
        }
    }

    #[test]
    fn arity_matches_names() {
        let e = ev(vec![], vec![], vec![]);
        assert_eq!(values(&e).len(), NAMES.len());
    }

    #[test]
    fn empty_is_all_finite_zero() {
        let e = ev(vec![], vec![], vec![]);
        let v = values(&e);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn single_point_is_finite() {
        let e = ev(vec![10.0], vec![vec![5.0]], vec![1.0]);
        let v = values(&e);
        assert_eq!(v.len(), NAMES.len());
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn gaussian_peak_scores_well() {
        // A clean symmetric peak should have high Gaussian R^2 and low roughness.
        let axis: Vec<f64> = (0..21).map(|i| i as f64).collect();
        let trace: Vec<f64> = axis
            .iter()
            .map(|&t| (-(t - 10.0).powi(2) / (2.0 * 3.0 * 3.0)).exp())
            .collect();
        let e = ev(axis, vec![trace], vec![1.0]);
        let v = values(&e);
        let idx = |name: &str| NAMES.iter().position(|&n| n == name).unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
        assert!(
            v[idx("gaussian_fit_r2")] > 0.9,
            "r2={}",
            v[idx("gaussian_fit_r2")]
        );
        assert!(v[idx("gaussian_cosine")] > 0.9);
        assert!(v[idx("fwhm_seconds")] > 0.0);
        assert!(v[idx("jaggedness")] <= 1.0 && v[idx("jaggedness")] >= 0.0);
        // Nearly symmetric peak -> small HWHM asymmetry.
        assert!(v[idx("hwhm_asymmetry")].abs() < 0.2);
    }

    #[test]
    fn bimodal_detected() {
        let mut trace = vec![0.0; 21];
        for (i, value) in trace.iter_mut().enumerate().take(21) {
            let t = i as f64;
            *value = (-(t - 5.0).powi(2) / 2.0).exp() + (-(t - 15.0).powi(2) / 2.0).exp();
        }
        let axis: Vec<f64> = (0..21).map(|i| i as f64).collect();
        let e = ev(axis, vec![trace], vec![1.0]);
        let v = values(&e);
        let idx = |name: &str| NAMES.iter().position(|&n| n == name).unwrap();
        assert!(v[idx("n_local_maxima")] >= 2.0);
        assert!(v[idx("modality")] > 0.0);
    }
}
