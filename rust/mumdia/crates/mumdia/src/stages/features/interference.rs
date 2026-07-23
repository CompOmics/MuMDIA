//! Extended feature family: interference.
//!
//! Detect co-isolated chimeric contamination via shared-profile residuals,
//! iterative interference removal (remove_ifs), tight/wide window contrast, and
//! rank decomposition of the fragment x time matrix. The central quantity is the
//! least-squares scale `r_f = <x_f, R> / <R, R>` that projects each fragment
//! trace onto the reference elution profile `R = e.ref_profile`.
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items in
//! the same order. Every value is finite (NaN/Inf replaced by 0.0) and all
//! degenerate cases (K = 0 fragments, T < 2 time points, empty windows) return
//! 0.0. Correlation/cosine are delegated to `crate::stats`; profile helpers to
//! the parent `features` module.
use super::Evidence;
use crate::stats::{cosine, pearson};

pub const NAMES: &[&str] = &[
    "explained_variance_ref",
    "profile_residual_fraction",
    "n_interfered_fragments",
    "corrected_vs_raw_cos",
    "corrected_vs_raw_ratio",
    "ifs_removed_count",
    "ifs_removed_intensity_frac",
    "ifs_corr_gain",
    "ifs_retained_frac",
    "matched_frac_after_ifs",
    "peak_to_full_area_ratio_profile",
    "peak_to_full_area_ratio_frag_mean",
    "peak_to_full_area_ratio_weighted",
    "out_of_peak_intensity_frac",
    "profile_corr_full_vs_peak_delta",
    "frac_frag_ref_corr_below_0_5",
    "explained_apex_intensity_frac",
    "apex_purity",
    "interference_apex_residual_fraction",
    "dominant_frag_ref_corr",
    "explained_variance_ratio",
    "second_component_fraction",
    "profile_second_peak_ratio",
    "n_competing_peaks_in_window",
    "matched_pred_intensity_fraction",
    "top_pred_frag_matched",
];

// Ref-corr thresholds used across features.
const COHERENT_THR: f64 = 0.7;
const IFS_MIN_CORR: f64 = 0.6;

pub fn values(e: &Evidence) -> Vec<f64> {
    let k = e.pred.len();
    let tp = e.ref_profile.len();
    let tf = e.axis_full.len();
    let r = &e.ref_profile;
    let ssr = ss(r);
    let r_apex = if e.apex_idx < r.len() {
        r[e.apex_idx]
    } else {
        0.0
    };

    // Per-fragment least-squares scale onto R.
    let mut rf = vec![0.0f64; k];
    if ssr > 0.0 {
        for (f, scale) in rf.iter_mut().enumerate().take(k) {
            if f < e.traces.len() {
                *scale = dot(&e.traces[f], r) / ssr;
            }
        }
    }

    // Reference profile over the full extraction window (pred-weighted sum).
    let mut rfull = vec![0.0f64; tf];
    for f in 0..k {
        if f < e.traces_full.len() {
            let x = &e.traces_full[f];
            let w = e.pred[f];
            let n = x.len().min(tf);
            for t in 0..n {
                rfull[t] += w * x[t];
            }
        }
    }

    // Matched fragment set (observed apex intensity present, trace available).
    let matched: Vec<usize> = (0..k)
        .filter(|&f| f < e.obs_apex.len() && e.obs_apex[f] > 0.0 && f < e.traces.len())
        .collect();

    // ---- 1. explained_variance_ref ----
    let mut num1 = 0.0;
    let mut den1 = 0.0;
    for (&scale, x) in rf.iter().zip(e.traces.iter()).take(k) {
        let n = x.len().min(tp);
        for t in 0..n {
            let resid = x[t] - scale * r[t];
            num1 += resid * resid;
            den1 += x[t] * x[t];
        }
    }
    let explained_variance_ref = if den1 > 0.0 { 1.0 - num1 / den1 } else { 0.0 };

    // ---- 2. profile_residual_fraction ----
    let mut acc2 = 0.0;
    let mut cnt2 = 0usize;
    for (&scale, x) in rf.iter().zip(e.traces.iter()).take(k) {
        let sx = ss(x);
        if sx > 0.0 {
            let n = x.len().min(tp);
            let mut resid = 0.0;
            for t in 0..n {
                let d = x[t] - scale * r[t];
                resid += d * d;
            }
            acc2 += resid / sx;
            cnt2 += 1;
        }
    }
    let profile_residual_fraction = if cnt2 > 0 { acc2 / cnt2 as f64 } else { 0.0 };

    // ---- 3. n_interfered_fragments ----
    let mut n_interfered = 0.0;
    for (f, &scale) in rf.iter().enumerate().take(k) {
        let a = if f < e.obs_apex.len() {
            e.obs_apex[f]
        } else {
            0.0
        };
        if a > 2.0 * scale * r_apex {
            n_interfered += 1.0;
        }
    }

    // ---- 4. corrected_vs_raw_cos ----
    let corrected_vs_raw_cos = cosine(&rf, &e.pred) - cosine(&e.obs_apex, &e.pred);

    // ---- 5. corrected_vs_raw_ratio ----
    let mut num5 = 0.0;
    let mut den5 = 0.0;
    for (f, &scale) in rf.iter().enumerate().take(k) {
        let a = if f < e.obs_apex.len() {
            e.obs_apex[f]
        } else {
            0.0
        };
        let cap = (1.5 * scale * r_apex).max(0.0);
        num5 += a.min(cap);
        den5 += a;
    }
    let corrected_vs_raw_ratio = if den5 > 0.0 { num5 / den5 } else { 0.0 };

    // ---- 6-10. remove_ifs iterative prune ----
    let mut s_set = matched.clone();
    let mut removed: Vec<usize> = Vec::new();
    loop {
        if s_set.len() <= 3 {
            break;
        }
        let mut min_corr = f64::INFINITY;
        let mut min_f = usize::MAX;
        for &f in &s_set {
            let lr = loo_ref(e, &s_set, f, tp);
            let c = pearson(&e.traces[f], &lr);
            if c < min_corr {
                min_corr = c;
                min_f = f;
            }
        }
        if min_f != usize::MAX && min_corr < IFS_MIN_CORR {
            s_set.retain(|&x| x != min_f);
            removed.push(min_f);
        } else {
            break;
        }
    }
    let ifs_removed_count = removed.len() as f64;

    let matched_int_sum: f64 = matched.iter().map(|&f| e.obs_apex[f]).sum();
    let removed_int_sum: f64 = removed.iter().map(|&f| e.obs_apex[f]).sum();
    let ifs_removed_intensity_frac = if matched_int_sum > 0.0 {
        removed_int_sum / matched_int_sum
    } else {
        0.0
    };

    let ifs_corr_gain = mean_loo(e, &s_set, tp) - mean_loo(e, &matched, tp);

    let ifs_retained_frac = if !matched.is_empty() {
        s_set.len() as f64 / matched.len() as f64
    } else {
        0.0
    };

    let n_pred = if e.n_predicted > 0 {
        e.n_predicted as f64
    } else {
        k as f64
    };
    let matched_frac_after_ifs = if n_pred > 0.0 {
        s_set.len() as f64 / n_pred
    } else {
        0.0
    };

    // ---- 11-14. peak vs full window area ratios ----
    let sum_peak_profile: f64 = r.iter().sum();
    let sum_full_profile: f64 = rfull.iter().sum();
    let peak_to_full_area_ratio_profile = if sum_full_profile > 0.0 {
        sum_peak_profile / sum_full_profile
    } else {
        0.0
    };
    let out_of_peak_intensity_frac = if sum_full_profile > 0.0 {
        1.0 - peak_to_full_area_ratio_profile
    } else {
        0.0
    };

    let mut acc_fm = 0.0;
    let mut cnt_fm = 0usize;
    let mut acc_w = 0.0;
    let mut wsum = 0.0;
    for f in 0..k {
        if f >= e.traces.len() || f >= e.traces_full.len() {
            continue;
        }
        let sp: f64 = e.traces[f].iter().sum();
        let sf: f64 = e.traces_full[f].iter().sum();
        if sf > 0.0 {
            let ratio = sp / sf;
            acc_fm += ratio;
            cnt_fm += 1;
            let w = e.pred[f];
            acc_w += w * ratio;
            wsum += w;
        }
    }
    let peak_to_full_area_ratio_frag_mean = if cnt_fm > 0 {
        acc_fm / cnt_fm as f64
    } else {
        0.0
    };
    let peak_to_full_area_ratio_weighted = if wsum > 0.0 { acc_w / wsum } else { 0.0 };

    // ---- 15. profile_corr_full_vs_peak_delta ----
    let mut sp_corr = 0.0;
    let mut sf_corr = 0.0;
    let mut c_corr = 0usize;
    for &f in &matched {
        sp_corr += pearson(&e.traces[f], r);
        let full_corr = if f < e.traces_full.len() {
            pearson(&e.traces_full[f], &rfull)
        } else {
            0.0
        };
        sf_corr += full_corr;
        c_corr += 1;
    }
    let profile_corr_full_vs_peak_delta = if c_corr > 0 {
        (sp_corr - sf_corr) / c_corr as f64
    } else {
        0.0
    };

    // ---- 16. frac_frag_ref_corr_below_0_5 ----
    let mut below = 0.0;
    for &f in &matched {
        if pearson(&e.traces[f], r) < 0.5 {
            below += 1.0;
        }
    }
    let frac_frag_ref_corr_below_0_5 = if !matched.is_empty() {
        below / matched.len() as f64
    } else {
        0.0
    };

    // ---- 17. explained_apex_intensity_frac ----
    let mut num17 = 0.0;
    let mut den17 = 0.0;
    for &f in &matched {
        let a = e.obs_apex[f];
        den17 += a;
        if pearson(&e.traces[f], r) >= COHERENT_THR {
            num17 += a;
        }
    }
    let explained_apex_intensity_frac = if den17 > 0.0 { num17 / den17 } else { 0.0 };

    // ---- 18. apex_purity ----
    let mut num18 = 0.0;
    let mut den18 = 0.0;
    for f in 0..k {
        if f >= e.traces.len() {
            continue;
        }
        let v = if e.apex_idx < e.traces[f].len() {
            e.traces[f][e.apex_idx]
        } else {
            0.0
        };
        den18 += v;
        if pearson(&e.traces[f], r) >= COHERENT_THR {
            num18 += v;
        }
    }
    let apex_purity = if den18 > 0.0 { num18 / den18 } else { 0.0 };

    // ---- 19. interference_apex_residual_fraction ----
    let mut num19 = 0.0;
    let mut den19 = 0.0;
    for (f, &scale) in rf.iter().enumerate().take(k) {
        if f >= e.traces.len() {
            continue;
        }
        let xa = if e.apex_idx < e.traces[f].len() {
            e.traces[f][e.apex_idx]
        } else {
            0.0
        };
        num19 += (xa - scale * r_apex).max(0.0);
        den19 += xa;
    }
    let interference_apex_residual_fraction = if den19 > 0.0 { num19 / den19 } else { 0.0 };

    // ---- 20. dominant_frag_ref_corr ----
    let mut best = 0usize;
    let mut best_pred = f64::NEG_INFINITY;
    for f in 0..k {
        if e.pred[f] > best_pred {
            best_pred = e.pred[f];
            best = f;
        }
    }
    let dominant_frag_ref_corr = if k > 0 && best < e.traces.len() {
        let all: Vec<usize> = (0..k).filter(|&g| g < e.traces.len()).collect();
        let lr = loo_ref(e, &all, best, tp);
        pearson(&e.traces[best], &lr)
    } else {
        0.0
    };

    // ---- 21-22. rank decomposition of matched fragment x time matrix ----
    let m = matched.len();
    let (mut explained_variance_ratio, mut second_component_fraction) = (0.0, 0.0);
    if m >= 1 {
        // Gram matrix G = X X^T over matched fragment peak traces (m x m).
        let mut g = vec![vec![0.0f64; m]; m];
        for i in 0..m {
            for j in i..m {
                let d = dot(&e.traces[matched[i]], &e.traces[matched[j]]);
                g[i][j] = d;
                g[j][i] = d;
            }
        }
        let mut trace = 0.0;
        for (i, row) in g.iter().enumerate().take(m) {
            trace += row[i];
        }
        if trace > 0.0 {
            let (lam1, v1) = power_top(&g);
            explained_variance_ratio = (lam1 / trace).clamp(0.0, 1.0);
            // Deflate and extract the second eigenvalue.
            for i in 0..m {
                for j in 0..m {
                    g[i][j] -= lam1 * v1[i] * v1[j];
                }
            }
            let (lam2, _) = power_top(&g);
            second_component_fraction = (lam2.max(0.0) / trace).clamp(0.0, 1.0);
        }
    }

    // ---- 23-24. competing peaks on the full-window reference profile ----
    let full_apex = if tf > 0 {
        let mut bi = 0usize;
        let mut bd = f64::INFINITY;
        for i in 0..tf {
            let d = (e.axis_full[i] - e.apex_rt).abs();
            if d < bd {
                bd = d;
                bi = i;
            }
        }
        bi
    } else {
        0
    };
    let max_peak = r.iter().cloned().fold(0.0f64, f64::max);
    let (lo, hi) = if rfull.len() >= 2 {
        super::peak_bounds(&rfull, full_apex, 0.5, 0)
    } else {
        (0, 0)
    };
    let mut second_peak = 0.0;
    let mut n_competing = 0.0;
    if rfull.len() >= 3 {
        let thr = 0.3 * max_peak;
        let w = 3isize;
        for i in 1..rfull.len() - 1 {
            let is_max = rfull[i] > rfull[i - 1] && rfull[i] >= rfull[i + 1];
            if !is_max {
                continue;
            }
            if (i < lo || i > hi) && rfull[i] > second_peak {
                second_peak = rfull[i];
            }
            if rfull[i] > thr && (i as isize - full_apex as isize).abs() >= w {
                n_competing += 1.0;
            }
        }
    }
    let profile_second_peak_ratio = if max_peak > 0.0 {
        second_peak / max_peak
    } else {
        0.0
    };
    let n_competing_peaks_in_window = n_competing;

    // ---- 25. matched_pred_intensity_fraction ----
    let tot_pred: f64 = e.pred.iter().sum();
    let seen_pred: f64 = matched.iter().map(|&f| e.pred[f]).sum();
    let matched_pred_intensity_fraction = if tot_pred > 0.0 {
        seen_pred / tot_pred
    } else {
        0.0
    };

    // ---- 26. top_pred_frag_matched ----
    let top_pred_frag_matched = if k > 0 && best < e.obs_apex.len() && e.obs_apex[best] > 0.0 {
        1.0
    } else {
        0.0
    };

    let out = vec![
        explained_variance_ref,
        profile_residual_fraction,
        n_interfered,
        corrected_vs_raw_cos,
        corrected_vs_raw_ratio,
        ifs_removed_count,
        ifs_removed_intensity_frac,
        ifs_corr_gain,
        ifs_retained_frac,
        matched_frac_after_ifs,
        peak_to_full_area_ratio_profile,
        peak_to_full_area_ratio_frag_mean,
        peak_to_full_area_ratio_weighted,
        out_of_peak_intensity_frac,
        profile_corr_full_vs_peak_delta,
        frac_frag_ref_corr_below_0_5,
        explained_apex_intensity_frac,
        apex_purity,
        interference_apex_residual_fraction,
        dominant_frag_ref_corr,
        explained_variance_ratio,
        second_component_fraction,
        profile_second_peak_ratio,
        n_competing_peaks_in_window,
        matched_pred_intensity_fraction,
        top_pred_frag_matched,
    ];
    debug_assert_eq!(out.len(), NAMES.len());
    out.into_iter().map(fin).collect()
}

// ---- private helpers ----

/// Finite guard: replace NaN/Inf with 0.0.
fn fin(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Dot product over the shared prefix length.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = 0.0;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

/// Sum of squares.
fn ss(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum()
}

/// Leave-one-out reference profile: predicted-intensity-weighted sum of the
/// peak traces of every fragment in `set` except `exclude`. Length `tp`.
fn loo_ref(e: &Evidence, set: &[usize], exclude: usize, tp: usize) -> Vec<f64> {
    let mut acc = vec![0.0f64; tp];
    for &g in set {
        if g == exclude || g >= e.traces.len() {
            continue;
        }
        let x = &e.traces[g];
        let w = e.pred[g];
        let n = x.len().min(tp);
        for t in 0..n {
            acc[t] += w * x[t];
        }
    }
    acc
}

/// Mean leave-one-out reference correlation over `set`.
fn mean_loo(e: &Evidence, set: &[usize], tp: usize) -> f64 {
    if set.len() < 2 {
        return 0.0;
    }
    let mut s = 0.0;
    let mut c = 0usize;
    for &f in set {
        if f >= e.traces.len() {
            continue;
        }
        let lr = loo_ref(e, set, f, tp);
        s += pearson(&e.traces[f], &lr);
        c += 1;
    }
    if c > 0 {
        s / c as f64
    } else {
        0.0
    }
}

/// Top eigenvalue and (normalized) eigenvector of a symmetric matrix via power
/// iteration. Returns (0.0, zeros) for an empty or degenerate matrix.
fn power_top(g: &[Vec<f64>]) -> (f64, Vec<f64>) {
    let m = g.len();
    if m == 0 {
        return (0.0, Vec::new());
    }
    let mut v = vec![1.0 / (m as f64).sqrt(); m];
    for _ in 0..100 {
        let mut nv = vec![0.0f64; m];
        for i in 0..m {
            let mut s = 0.0;
            for j in 0..m {
                s += g[i][j] * v[j];
            }
            nv[i] = s;
        }
        let norm = ss(&nv).sqrt();
        if norm <= 0.0 {
            return (0.0, vec![0.0; m]);
        }
        for x in nv.iter_mut() {
            *x /= norm;
        }
        v = nv;
    }
    // Rayleigh quotient: v is unit-norm, so lambda = v^T G v.
    let mut gv = vec![0.0f64; m];
    for i in 0..m {
        let mut s = 0.0;
        for j in 0..m {
            s += g[i][j] * v[j];
        }
        gv[i] = s;
    }
    (dot(&v, &gv), v)
}
