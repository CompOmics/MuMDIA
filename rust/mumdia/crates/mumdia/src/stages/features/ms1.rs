//! Extended feature family: ms1.
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items
//! in the same order. Names are globally unique, snake_case, and append-only
//! (they are part of the frozen extended feature schema). See the `Evidence`
//! struct in `stages/features.rs` for the available per-PSM evidence, and use
//! the shared kernels `crate::stats::{pearson, cosine, spectral_angle}` plus the
//! parent helpers `super::{mean, normalize_sum, best_xcorr, smooth3, peak_bounds}`.
//!
//! Precursor isotope-envelope agreement against a Poisson-averagine model
//! (lambda ~= 0.000594 * M), plus MS1/MS2 co-elution features that read the
//! (currently unpopulated) MS1 isotope XIC evidence and return 0.0 until it is
//! persisted. All values are guarded finite.
use super::Evidence;
use crate::stats::{cosine, pearson, spectral_angle};

pub const NAMES: &[&str] = &[
    // apex isotope-envelope agreement vs averagine
    "ms1_isotope_cosine_apex",
    "ms1_isotope_spectral_angle_apex",
    "ms1_isotope_chi2_apex",
    "ms1_isotope_manhattan_apex",
    // apex isotope ratios and deviations
    "iso_ratio_1_0",
    "iso_ratio_2_0",
    "iso_plus1_ratio_dev",
    "iso_plus2_ratio_dev",
    "iso_minus_one_fraction",
    "iso_overlap_flag",
    // apex intensity / presence
    "log_ms1_mono",
    "ms1_total_isotope_log",
    "has_ms1_signal",
    // apex isotope entropy
    "ms1_isotope_apex_entropy_3",
    "ms1_m1_entropy_contribution",
    // MS1 XIC co-elution / shape (need ms1_xic; 0.0 until persisted)
    "ms1_ms2_time_corr",
    "ms1_ms2_envelope_time_corr",
    "ms1_iso_coelution",
    "ms1_ms2_apex_rt_delta",
    "ms1_iso_ratio_stability",
    "ms1_mono_gaussianity",
    "ms1_ms2_fwhm_ratio",
    "ms1_isotope_corr_xic",
    "ms1_envelope_over_time_corr",
    "ms1_isotope_xic_shape_consistency",
];

const EPS: f64 = 1e-9;
const PROTON: f64 = 1.007_276_466_812;

/// Finite guard: replace NaN/Inf with 0.0.
#[inline]
fn fin(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// ln(1+x) with a guard for x <= -1 (returns 0.0).
#[inline]
fn ln1p(x: f64) -> f64 {
    let a = 1.0 + x;
    if a > 0.0 { fin(a.ln()) } else { 0.0 }
}

/// Neutral precursor mass from m/z and charge; 0.0 if charge <= 0.
fn neutral_mass(e: &Evidence) -> f64 {
    if e.charge <= 0 {
        return 0.0;
    }
    let z = e.charge as f64;
    fin(e.precursor_mz * z - z * PROTON)
}

/// Poisson-averagine isotope weights [T0,T1,T2] (unnormalized), lambda=0.000594*M.
fn averagine3(mass: f64) -> [f64; 3] {
    let lambda = (0.000_594 * mass).max(0.0);
    let e0 = (-lambda).exp();
    let t0 = fin(e0);
    let t1 = fin(lambda * e0);
    let t2 = fin(0.5 * lambda * lambda * e0);
    [t0, t1, t2]
}

/// Sum-normalize a fixed 3-vector (returns zeros if sum <= 0).
fn sumnorm3(v: [f64; 3]) -> [f64; 3] {
    let s = v[0] + v[1] + v[2];
    if s > 0.0 {
        [v[0] / s, v[1] / s, v[2] / s]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Shannon entropy (natural log) of a sum-normalized nonnegative slice.
fn entropy(v: &[f64]) -> f64 {
    let s: f64 = v.iter().filter(|&&x| x > 0.0).sum();
    if s <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for &x in v {
        if x > 0.0 {
            let p = x / s;
            h -= p * p.ln();
        }
    }
    fin(h)
}

/// Index of the maximum element; 0 if empty.
fn argmax(v: &[f64]) -> usize {
    let mut bi = 0usize;
    let mut bv = f64::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

/// FWHM in RT units of a trace over `axis`, via half-height peak bounds.
fn fwhm(axis: &[f64], trace: &[f64]) -> f64 {
    let n = trace.len().min(axis.len());
    if n < 3 {
        return 0.0;
    }
    let sm = super::smooth3(&trace[..n]);
    let ai = argmax(&sm);
    let (lo, hi) = super::peak_bounds(&sm, ai, 0.5, 0);
    if hi < axis.len() && lo < axis.len() && hi >= lo {
        fin(axis[hi] - axis[lo]).max(0.0)
    } else {
        0.0
    }
}

pub fn values(e: &Evidence) -> Vec<f64> {
    let i0 = e.ms1_mono.unwrap_or(0.0).max(0.0);
    let i1 = e.ms1_iso1.unwrap_or(0.0).max(0.0);
    let i2 = e.ms1_iso2.unwrap_or(0.0).max(0.0);
    let im1 = e.ms1_isom1.unwrap_or(0.0).max(0.0);

    let mass = neutral_mass(e);
    let t = averagine3(mass); // [T0,T1,T2]

    // --- apex isotope-envelope agreement vs averagine ---
    let obs3 = [i0, i1, i2];
    let iso_cos = fin(cosine(&obs3, &t));
    let iso_sa = fin(spectral_angle(&obs3, &t));

    let mn = sumnorm3(obs3);
    let tn = sumnorm3(t);
    // chi2 on sum-normalized envelopes
    let mut chi2 = 0.0;
    for k in 0..3 {
        let d = mn[k] - tn[k];
        chi2 += d * d / (tn[k] + EPS);
    }
    let iso_chi2 = fin(chi2);
    // manhattan agreement: 1 - 0.5 * L1
    let l1 = (mn[0] - tn[0]).abs() + (mn[1] - tn[1]).abs() + (mn[2] - tn[2]).abs();
    let iso_manh = fin(1.0 - 0.5 * l1);

    // --- apex isotope ratios and deviations ---
    let r10 = fin(i1 / (i0 + EPS));
    let r20 = fin(i2 / (i0 + EPS));
    let tr10 = t[1] / (t[0] + EPS);
    let tr20 = t[2] / (t[0] + EPS);
    let dev1 = fin((r10 - tr10).abs());
    let dev2 = fin((r20 - tr20).abs());
    let m1_frac = fin(im1 / (i0 + i1 + i2 + EPS));
    let overlap_flag = if im1 > 0.2 * i0 && i0 > 0.0 {
        1.0
    } else {
        0.0
    };

    // --- apex intensity / presence ---
    let log_mono = ln1p(i0);
    let total_log = ln1p(i0 + i1 + i2);
    let has_ms1 = if i0 > 0.0 { 1.0 } else { 0.0 };

    // --- apex isotope entropy ---
    let ent3 = entropy(&obs3);
    let ent4 = entropy(&[im1, i0, i1, i2]);
    let m1_ent_contrib = fin(ent4 - ent3);

    // --- MS1 XIC co-elution / shape features (0.0 until ms1_xic persisted) ---
    let (
        ms2_time_corr,
        env_time_corr,
        iso_coel,
        apex_rt_delta,
        ratio_stability,
        mono_gauss,
        fwhm_ratio,
        iso_corr_xic,
        env_over_time_corr,
        xic_shape_consistency,
    ) = xic_features(e);

    vec![
        iso_cos,
        iso_sa,
        iso_chi2,
        iso_manh,
        r10,
        r20,
        dev1,
        dev2,
        m1_frac,
        overlap_flag,
        log_mono,
        total_log,
        has_ms1,
        ent3,
        m1_ent_contrib,
        ms2_time_corr,
        env_time_corr,
        iso_coel,
        apex_rt_delta,
        ratio_stability,
        mono_gauss,
        fwhm_ratio,
        iso_corr_xic,
        env_over_time_corr,
        xic_shape_consistency,
    ]
}

/// Compute the ten MS1-XIC-dependent features. Returns all zeros when the MS1
/// isotope XIC evidence is absent (the current state) or too short.
fn xic_features(e: &Evidence) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let zeros = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if e.ms1_xic.len() < 3 {
        return zeros;
    }
    let mono = &e.ms1_xic[0];
    let p1 = &e.ms1_xic[1];
    let p2 = &e.ms1_xic[2];
    let r = &e.ref_profile;
    let axis = &e.axis;

    let t_peak = mono.len().min(p1.len()).min(p2.len());
    if t_peak < 2 {
        return zeros;
    }
    let mono = &mono[..t_peak];
    let p1s = &p1[..t_peak];
    let p2s = &p2[..t_peak];

    // envelope trace (mono + +1 + +2)
    let env: Vec<f64> = (0..t_peak).map(|i| mono[i] + p1s[i] + p2s[i]).collect();

    // ms1_ms2_time_corr: pearson(mono, R)
    let ms2_time_corr = fin(pearson(mono, r));
    // ms1_ms2_envelope_time_corr: pearson(envelope, R)
    let env_time_corr = fin(pearson(&env, r));

    // ms1_iso_coelution: mean pairwise pearson among mono/+1/+2
    let c01 = pearson(mono, p1s);
    let c02 = pearson(mono, p2s);
    let c12 = pearson(p1s, p2s);
    let iso_coel = fin((c01 + c02 + c12) / 3.0);

    // ms1_ms2_apex_rt_delta: |argmaxRT(mono) - argmaxRT(R)| / (base_width+eps)
    let apex_rt_delta = {
        let n = axis.len();
        if n >= 2 {
            let am_mono = argmax(mono).min(n - 1);
            let rn = r.len().min(n);
            let am_r = if rn >= 1 {
                argmax(&r[..rn]).min(n - 1)
            } else {
                0
            };
            let base_width = fwhm(axis, r);
            let d = (axis[am_mono] - axis[am_r]).abs();
            // Bounded to [0,1): d/(d+width) instead of d/width, so a degenerate
            // (~0) reference width can't send the feature to ~1e9.
            fin(d / (d + base_width + EPS))
        } else {
            0.0
        }
    };

    // ms1_iso_ratio_stability: mono-weighted std of (+1/mono) across scans with mono>0.1*max
    let ratio_stability = {
        let mmax = mono.iter().cloned().fold(0.0_f64, f64::max);
        if mmax > 0.0 {
            let thr = 0.1 * mmax;
            let (mut wsum, mut wr) = (0.0, 0.0);
            let mut rs: Vec<(f64, f64)> = Vec::new(); // (weight, ratio)
            for i in 0..t_peak {
                if mono[i] > thr {
                    let ratio = p1s[i] / (mono[i] + EPS);
                    rs.push((mono[i], ratio));
                    wsum += mono[i];
                    wr += mono[i] * ratio;
                }
            }
            if wsum > 0.0 && rs.len() >= 2 {
                let rbar = wr / wsum;
                let mut var = 0.0;
                for &(w, ratio) in &rs {
                    let d = ratio - rbar;
                    var += w * d * d;
                }
                var /= wsum;
                fin(var.max(0.0).sqrt())
            } else {
                0.0
            }
        } else {
            0.0
        }
    };

    // ms1_mono_gaussianity: R^2 (pearson^2) of mono XIC vs a moment-fit Gaussian
    let mono_gauss = {
        let wsum: f64 = mono.iter().sum();
        let na = axis.len().min(t_peak);
        if wsum > 0.0 && na >= 2 {
            // intensity-weighted mean and variance over axis
            let mut mu = 0.0;
            for i in 0..na {
                mu += axis[i] * mono[i];
            }
            mu /= wsum;
            let mut var = 0.0;
            for i in 0..na {
                let d = axis[i] - mu;
                var += mono[i] * d * d;
            }
            var /= wsum;
            if var > 0.0 {
                let g: Vec<f64> = (0..na)
                    .map(|i| {
                        let d = axis[i] - mu;
                        (-(d * d) / (2.0 * var)).exp()
                    })
                    .collect();
                let c = pearson(&mono[..na], &g);
                fin((c * c).clamp(0.0, 1.0))
            } else {
                0.0
            }
        } else {
            0.0
        }
    };

    // ms1_ms2_fwhm_ratio: FWHM(mono)/FWHM(R)
    let fwhm_ratio = {
        let fm = fwhm(axis, mono);
        let fr = fwhm(axis, r);
        fin(fm / (fr + EPS))
    };

    // averagine for XIC-integrated agreement
    let t = averagine3(neutral_mass(e));

    // ms1_isotope_corr_xic: pearson(window-integrated [A0,A1,A2], averagine)
    let iso_corr_xic = {
        let a0: f64 = mono.iter().sum();
        let a1: f64 = p1s.iter().sum();
        let a2: f64 = p2s.iter().sum();
        fin(pearson(&[a0, a1, a2], &t))
    };

    // ms1_envelope_over_time_corr: mono-weighted mean of per-scan pearson([mono,+1,+2], averagine)
    let env_over_time_corr = {
        let (mut wsum, mut wc) = (0.0, 0.0);
        for i in 0..t_peak {
            let w = mono[i];
            if w > 0.0 {
                let c = pearson(&[mono[i], p1s[i], p2s[i]], &t);
                wsum += w;
                wc += w * c;
            }
        }
        if wsum > 0.0 { fin(wc / wsum) } else { 0.0 }
    };

    // ms1_isotope_xic_shape_consistency: mean pairwise pearson among baseline-subtracted XICs
    let xic_shape_consistency = {
        let base = |v: &[f64]| -> Vec<f64> {
            let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let mn = if mn.is_finite() { mn } else { 0.0 };
            v.iter().map(|&x| (x - mn).max(0.0)).collect()
        };
        let b0 = base(mono);
        let b1 = base(p1s);
        let b2 = base(p2s);
        let s01 = pearson(&b0, &b1);
        let s02 = pearson(&b0, &b2);
        let s12 = pearson(&b1, &b2);
        fin((s01 + s02 + s12) / 3.0)
    };

    (
        ms2_time_corr,
        env_time_corr,
        iso_coel,
        apex_rt_delta,
        ratio_stability,
        mono_gauss,
        fwhm_ratio,
        iso_corr_xic,
        env_over_time_corr,
        xic_shape_consistency,
    )
}
