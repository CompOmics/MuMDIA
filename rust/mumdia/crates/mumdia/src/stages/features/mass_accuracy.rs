//! Extended feature family: mass_accuracy.
//!
//! Fragment mass-error (ppm) distribution features. All ppm values are the
//! signed `mass_err_ppm` supplied per predicted fragment; only fragments
//! observed at the apex (`obs_apex[i] > 0.0`) are treated as matched and enter
//! the statistics. Dispersion, weighted, and trend variants are kept; the plan
//! also retains a signed mean and weighted mean variants (distinct
//! computations), so they are implemented.
//!
//! `precursor_mass_error_ppm` from the plan is SKIPPED: it needs a theoretical
//! precursor m/z derived from the peptidoform + charge via the mass model, which
//! is not present in `Evidence`.
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items in
//! the same order. Names are globally unique, snake_case, and append-only.
use super::Evidence;
use crate::stats::pearson;

pub const NAMES: &[&str] = &[
    "median_abs_frag_ppm",
    "signed_mean_frag_ppm",
    "ppm_std",
    "ppm_iqr",
    "ppm_range",
    "max_abs_frag_ppm",
    "intensity_weighted_abs_ppm",
    "intensity_weighted_signed_ppm",
    "intensity_weighted_ppm_std",
    "lib_weighted_abs_ppm",
    "frac_frag_within_half_tol",
    "high_ppm_intensity_frac",
    "ppm_intensity_anticorr",
    "mass_error_mz_trend",
    "mean_abs_mz_error_da",
    // positive mass-accuracy evidence (DIA-NN `Mass.Evidence` analogs)
    "mass_evidence_gauss",
    "mass_log_evidence",
];

/// Fragment mass tolerance used for the within-tolerance / off-mass features.
/// The config tolerance is not carried in `Evidence`; this MVP-conservative
/// constant matches the fixed fragment tolerance regime. Half of it gates the
/// "within half tolerance" / "high ppm" features.
const FRAG_TOL_PPM: f64 = 20.0;
const HALF_TOL_PPM: f64 = 0.5 * FRAG_TOL_PPM;

/// Replace non-finite with 0.0.
fn fin(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Median of a slice (0.0 if empty). Does not mutate the input.
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let m = if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    };
    fin(m)
}

/// Linear-interpolated quantile q in [0,1] over a pre-sorted slice.
fn quantile_sorted(s: &[f64], q: f64) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    if s.len() == 1 {
        return fin(s[0]);
    }
    let pos = q.clamp(0.0, 1.0) * (s.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    fin(s[lo] + (s[hi] - s[lo]) * frac)
}

/// Population standard deviation (0.0 if len < 2 or non-finite).
fn pop_std(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let m: f64 = v.iter().sum::<f64>() / n as f64;
    let var: f64 = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        0.0
    } else {
        fin(var.sqrt())
    }
}

/// Weighted mean of x with weights w (equal length). 0.0 if total weight <= 0.
fn weighted_mean(x: &[f64], w: &[f64]) -> f64 {
    let sw: f64 = w.iter().sum();
    if sw <= 0.0 {
        return 0.0;
    }
    let sx: f64 = x.iter().zip(w).map(|(a, b)| a * b).sum();
    fin(sx / sw)
}

pub fn values(e: &Evidence) -> Vec<f64> {
    let n = NAMES.len();
    let k = e.pred.len();

    // Matched-fragment aligned arrays. Matched = observed at apex.
    let mut ppm: Vec<f64> = Vec::new(); // signed ppm
    let mut absppm: Vec<f64> = Vec::new();
    let mut w_apex: Vec<f64> = Vec::new(); // apex intensity weights
    let mut w_lib: Vec<f64> = Vec::new(); // library (predicted) weights
    let mut theo_mz: Vec<f64> = Vec::new();
    let mut da_err: Vec<f64> = Vec::new(); // |obs_mz - theo_mz|

    let have = k == e.mass_err_ppm.len()
        && k == e.obs_apex.len()
        && k == e.pred.len()
        && k == e.frag_mz.len()
        && k == e.frag_obs_mz.len();

    if have {
        for i in 0..k {
            if e.obs_apex[i] > 0.0 {
                let p = e.mass_err_ppm[i];
                if !p.is_finite() {
                    continue;
                }
                ppm.push(p);
                absppm.push(p.abs());
                w_apex.push(e.obs_apex[i].max(0.0));
                w_lib.push(e.pred[i].max(0.0));
                theo_mz.push(e.frag_mz[i]);
                da_err.push((e.frag_obs_mz[i] - e.frag_mz[i]).abs());
            }
        }
    }

    if ppm.is_empty() {
        return vec![0.0; n];
    }

    // Sorted copy of signed ppm for quantiles.
    let mut sorted = ppm.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));

    // 1. median absolute ppm
    let median_abs = median(&absppm);

    // 2. signed mean ppm
    let signed_mean = fin(ppm.iter().sum::<f64>() / ppm.len() as f64);

    // 3. population std of signed ppm
    let ppm_std = pop_std(&ppm);

    // 4. IQR of signed ppm
    let q25 = quantile_sorted(&sorted, 0.25);
    let q75 = quantile_sorted(&sorted, 0.75);
    let ppm_iqr = fin(q75 - q25);

    // 5. range of signed ppm
    let ppm_range = fin(sorted[sorted.len() - 1] - sorted[0]);

    // 6. max absolute ppm
    let max_abs = fin(absppm
        .iter()
        .cloned()
        .fold(0.0_f64, |m, v| if v > m { v } else { m }));

    // 7. intensity-weighted absolute ppm
    let iw_abs = weighted_mean(&absppm, &w_apex);

    // 8. intensity-weighted signed ppm
    let iw_signed = weighted_mean(&ppm, &w_apex);

    // 9. intensity-weighted ppm std
    let iw_std = {
        let sw: f64 = w_apex.iter().sum();
        if sw <= 0.0 {
            0.0
        } else {
            let wm = iw_signed;
            let var: f64 = ppm
                .iter()
                .zip(&w_apex)
                .map(|(p, w)| w * (p - wm) * (p - wm))
                .sum::<f64>()
                / sw;
            if var <= 0.0 {
                0.0
            } else {
                fin(var.sqrt())
            }
        }
    };

    // 10. library-weighted absolute ppm
    let lib_abs = weighted_mean(&absppm, &w_lib);

    // 11. fraction of matched frags within half tolerance
    let within = absppm.iter().filter(|&&v| v < HALF_TOL_PPM).count();
    let frac_within = fin(within as f64 / absppm.len() as f64);

    // 12. off-mass observed-intensity fraction (|ppm| > half tol)
    let total_int: f64 = w_apex.iter().sum();
    let high_int: f64 = absppm
        .iter()
        .zip(&w_apex)
        .filter(|(&p, _)| p > HALF_TOL_PPM)
        .map(|(_, &a)| a)
        .sum();
    let high_ppm_frac = if total_int > 0.0 {
        fin(high_int / total_int)
    } else {
        0.0
    };

    // 13. pearson(|ppm|, apex intensity); real matches trend negative
    let ppm_int_anticorr = fin(pearson(&absppm, &w_apex));

    // 14. |pearson(theo m/z, signed ppm)|; nonzero slope = mixed origins
    let mz_trend = fin(pearson(&theo_mz, &ppm).abs());

    // 15. mean absolute m/z error in Da
    let mean_abs_da = fin(da_err.iter().sum::<f64>() / da_err.len() as f64);

    // 16-17. POSITIVE mass-accuracy evidence (all above measure error/badness). A
    // Gaussian kernel exp(-(ppm/sigma)^2/2) rewards fragments tightly on-mass; the
    // offset is already removed upstream so real matches concentrate near 0 ppm and
    // decoys spread across the tolerance. `mass_evidence_gauss` is the predicted-
    // weighted concentration in [0,1]; `mass_log_evidence` is the intensity-
    // accumulated, log-scaled (unbounded, granular) analog of DIA-NN `Mass.Evidence`.
    const SIGMA_PPM: f64 = 10.0;
    let (mut ev_num, mut ev_den, mut ev_acc) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..ppm.len() {
        let g = (-0.5 * (ppm[i] / SIGMA_PPM).powi(2)).exp();
        ev_num += w_lib[i] * g;
        ev_den += w_lib[i];
        ev_acc += w_apex[i] * g;
    }
    let mass_evidence_gauss = if ev_den > 0.0 {
        fin(ev_num / ev_den)
    } else {
        0.0
    };
    let mass_log_evidence = fin((1.0 + ev_acc).ln());

    vec![
        median_abs,
        signed_mean,
        ppm_std,
        ppm_iqr,
        ppm_range,
        max_abs,
        iw_abs,
        iw_signed,
        iw_std,
        lib_abs,
        frac_within,
        high_ppm_frac,
        ppm_int_anticorr,
        mz_trend,
        mean_abs_da,
        mass_evidence_gauss,
        mass_log_evidence,
    ]
}
