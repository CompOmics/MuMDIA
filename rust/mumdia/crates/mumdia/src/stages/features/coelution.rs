//! Extended feature family: coelution.
//!
//! Fragment co-elution discriminators built on the per-fragment intensity
//! traces and the predicted-intensity-weighted reference profile R. Three
//! groups: (1) per-fragment correlation against R (peak and full windows,
//! leave-one-out variants), (2) pairwise fragment-fragment Pearson statistics
//! and lag-optimized cross-correlation, (3) structured cross-correlations
//! (b vs y ions, charge-1 vs multiply-charged).
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items
//! in the same order. All values are guaranteed finite (NaN/Inf -> 0.0) and the
//! length is stable across empty/degenerate evidence. Correlation and cosine
//! use the shared kernels; cross-correlation uses `super::best_xcorr`.
//!
//! Note: the JSON feature `coelution_weighted_mean` is an exact alias of
//! `pairwise_coelution_weighted` (identical computation) and is emitted once
//! here under the `pairwise_coelution_weighted` name only.
use std::borrow::Cow;

use super::{best_xcorr_normed, xcorr_norm, Evidence};
use crate::stats::{pearson, pearson_pairs, pearson_vs, Centered};

pub const NAMES: &[&str] = &[
    "frag_ref_corr_mean",
    "frag_ref_corr_obsweighted",
    "frag_ref_corr_min",
    "frag_ref_corr_std",
    "frag_ref_corr_sq_mean",
    "frag_ref_corr_topk_weighted",
    "n_frag_ref_corr_above_0_9",
    "frac_frag_ref_corr_above_0_8",
    "frag_ref_corr_mean_full",
    "full_vs_peak_corr_gain",
    "pairwise_coelution_weighted",
    "pairwise_coelution_min",
    "pairwise_coelution_median",
    "pairwise_coelution_std",
    "pairwise_coelution_frac_negative",
    "pairwise_coelution_hi",
    "pairwise_coelution_lo",
    "coelution_hi_lo_contrast",
    "coelution_corr_entropy",
    "xcorr_shape_mean",
    "xcorr_shape_min",
    "xcorr_shape_std",
    "xcorr_lag_mean_abs",
    "xcorr_lag_std",
    "xcorr_lag_iqr",
    "xcorr_lag_frac_zero",
    "xcorr_lag_max_abs",
    "xcorr_lag_entropy",
    "ref_xcorr_lag_mean",
    "ref_xcorr_shape_mean",
    "observed_sum_vs_template_corr",
    "frag_loo_ref_corr_mean",
    "frag_loo_ref_corr_min",
    "frac_frags_apex_aligned",
    "top3_frag_ref_corr",
    "by_cross_coelution",
    "by_cross_lag_mean",
    "charge_cross_coelution",
];

const MAXLAG: i32 = 5;
const NBINS_CORR: usize = 10;

pub fn values(e: &Evidence) -> Vec<f64> {
    let k = e.pred.len();
    // Guards: traces array must be consistent with K; R over the peak axis.
    let traces = &e.traces;
    let r = &e.ref_profile;
    let has_traces = traces.len() == k && k > 0;

    // --- per-fragment ref correlation over the peak window ---
    // `r` centred once for every fragment (`pearson_vs` is `pearson` bit for bit).
    let r_c = Centered::new(r);
    let rc: Vec<f64> = if has_traces {
        (0..k).map(|i| pearson_vs(&traces[i], r, &r_c)).collect()
    } else {
        Vec::new()
    };

    // matched indices (fragment observed at apex)
    let matched: Vec<usize> = (0..k).filter(|&i| e.obs_apex[i] > 0.0).collect();

    // 1. frag_ref_corr_mean: mean over matched fragments
    let f_mean = mean_at(&rc, &matched);

    // 2. frag_ref_corr_obsweighted: sum a_f*rc / sum a_f
    let (mut wnum, mut wden) = (0.0f64, 0.0f64);
    for &i in &matched {
        let a = e.obs_apex[i];
        wnum += a * rc[i];
        wden += a;
    }
    let f_obsw = if wden > 0.0 { wnum / wden } else { 0.0 };

    // 3. frag_ref_corr_min (all fragments)
    let f_min = rc.iter().cloned().fold(f64::INFINITY, f64::min);
    let f_min = if rc.is_empty() { 0.0 } else { f_min };

    // 4. frag_ref_corr_std (population, all fragments)
    let f_std = std_pop(&rc);

    // 5. frag_ref_corr_sq_mean
    let f_sqmean = if rc.is_empty() {
        0.0
    } else {
        rc.iter().map(|v| v * v).sum::<f64>() / rc.len() as f64
    };

    // 6. frag_ref_corr_topk_weighted: lib-weighted mean rc over top-6 by pred
    let f_topk = weighted_mean_topk(&rc, &e.pred, 6);

    // 7. n_frag_ref_corr_above_0_9
    let n_above_09 = rc.iter().filter(|&&v| v >= 0.9).count() as f64;

    // 8. frac_frag_ref_corr_above_0_8
    let n_above_08 = rc.iter().filter(|&&v| v >= 0.8).count() as f64;
    let f_frac08 = if e.n_predicted > 0 {
        n_above_08 / e.n_predicted as f64
    } else {
        0.0
    };

    // --- full-window ref correlation ---
    let tf = &e.traces_full;
    let has_full = tf.len() == k && k > 0 && !e.axis_full.is_empty();
    // R over the full window. This used to be a second build of the profile
    // `interference` also builds inline, term for term the same sum; `build_evidence`
    // now builds it once (see `super::weighted_reference_full`). It is read only under
    // `has_full`, which requires `traces_full.len() == pred.len()` -- the condition that
    // made the two builds identical.
    //
    // The length is checked rather than assumed, for the reason `interference` gives:
    // `Evidence` is `pub` with `pub` fields, and a short profile would quietly shorten
    // every `pearson` below instead of failing. On `build_evidence`, the one constructor,
    // the check is a comparison and the rebuild never runs.
    let r_full_owned: Cow<[f64]> = if !has_full {
        Cow::Borrowed(&[][..])
    } else if e.ref_profile_full.len() == e.axis_full.len() {
        Cow::Borrowed(&e.ref_profile_full)
    } else {
        Cow::Owned(super::weighted_reference_full(
            &e.traces_full,
            &e.pred,
            e.axis_full.len(),
        ))
    };
    let r_full: &[f64] = &r_full_owned;
    let rc_full: Vec<f64> = if has_full {
        let r_full_c = Centered::new(r_full);
        (0..k)
            .map(|i| pearson_vs(&tf[i], r_full, &r_full_c))
            .collect()
    } else {
        Vec::new()
    };

    // 9. frag_ref_corr_mean_full (all fragments)
    let f_mean_full = mean_all(&rc_full);

    // 10. full_vs_peak_corr_gain: peak(all) - full(all)
    let f_gain = mean_all(&rc) - f_mean_full;

    // --- pairwise Pearson (symmetric matrix) + cross-correlation over pairs ---
    // Row-major flat k x k, one allocation instead of k. `corr[a * k + b]`.
    let mut corr = vec![0.0f64; k * k];
    let mut pair_p: Vec<f64> = Vec::new();
    let mut pair_w: Vec<f64> = Vec::new(); // l_a * l_b weights aligned with pair_p
    let mut xvals: Vec<f64> = Vec::new();
    let mut xlags: Vec<f64> = Vec::new();
    // b x y and charge-cross pair accumulators
    let mut by_sum = 0.0f64;
    let mut by_cnt = 0.0f64;
    let mut by_lag_sum = 0.0f64;
    let mut chg_sum = 0.0f64;
    let mut chg_cnt = 0.0f64;
    // Each trace's cross-correlation norm once (it was recomputed for every pair and for
    // the reference pass below); `xcorr_norm` is the expression `best_xcorr` evaluates.
    let norms: Vec<f64> = if has_traces {
        traces.iter().take(k).map(|t| xcorr_norm(t)).collect()
    } else {
        Vec::new()
    };
    // Every pair's Pearson from traces centred once, in the (a, b) order the loop reads.
    let mut pairs: Vec<f64> = Vec::new();
    if has_traces {
        pearson_pairs(&traces[..k], &mut pairs);
    }
    let mut pairs = pairs.into_iter();
    if has_traces {
        for a in 0..k {
            corr[a * k + a] = 1.0;
            for b in (a + 1)..k {
                let p = pairs.next().expect("one correlation per pair");
                corr[a * k + b] = p;
                corr[b * k + a] = p;
                pair_p.push(p);
                pair_w.push(e.pred[a] * e.pred[b]);
                let (lag, xv) =
                    best_xcorr_normed(&traces[a], &traces[b], MAXLAG, norms[a], norms[b]);
                xvals.push(xv);
                xlags.push(lag as f64);
                // b x y cross pairs
                if e.is_b[a] != e.is_b[b] {
                    by_sum += p;
                    by_cnt += 1.0;
                    by_lag_sum += (lag as f64).abs();
                }
                // charge cross: charge-1 vs charge>=2
                let ca = e.frag_charge[a];
                let cb = e.frag_charge[b];
                if (ca == 1 && cb >= 2) || (cb == 1 && ca >= 2) {
                    chg_sum += p;
                    chg_cnt += 1.0;
                }
            }
        }
    }

    // 11. pairwise_coelution_weighted (== coelution_weighted_mean, emitted once)
    let wsum: f64 = pair_w.iter().sum();
    let f_pw = if wsum > 0.0 {
        pair_p.iter().zip(&pair_w).map(|(p, w)| p * w).sum::<f64>() / wsum
    } else {
        0.0
    };

    // 12-15. pairwise min / median / std / frac negative
    let f_pmin = if pair_p.is_empty() {
        0.0
    } else {
        pair_p.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    let f_pmed = median(&pair_p);
    let f_pstd = std_pop(&pair_p);
    let f_pneg = if pair_p.is_empty() {
        0.0
    } else {
        pair_p.iter().filter(|&&v| v < 0.0).count() as f64 / pair_p.len() as f64
    };

    // 16-18. hi / lo / contrast: split fragments by pred (top vs bottom half)
    let order = sort_idx_desc(&e.pred);
    let half = k / 2;
    let hi_idx: Vec<usize> = order.iter().take(half).cloned().collect();
    let lo_idx: Vec<usize> = order.iter().skip(k - half).cloned().collect();
    let f_hi = subset_pair_mean(&corr, k, &hi_idx);
    let f_lo = subset_pair_mean(&corr, k, &lo_idx);
    let f_contrast = f_hi - f_lo;

    // 19. coelution_corr_entropy: histogram of pairwise pearson in [-1,1]
    let f_corr_ent = entropy_hist(&pair_p, -1.0, 1.0, NBINS_CORR);

    // 20-22. xcorr shape mean / min / std
    let f_xmean = mean_all(&xvals);
    let f_xmin = if xvals.is_empty() {
        0.0
    } else {
        xvals.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    let f_xstd = std_pop(&xvals);

    // 23-28. xcorr lag statistics
    let abs_lags: Vec<f64> = xlags.iter().map(|l| l.abs()).collect();
    let f_lag_mean_abs = mean_all(&abs_lags);
    let f_lag_std = std_pop(&xlags);
    let f_lag_iqr = iqr(&xlags);
    let f_lag_fzero = if xlags.is_empty() {
        0.0
    } else {
        xlags.iter().filter(|&&l| l == 0.0).count() as f64 / xlags.len() as f64
    };
    let f_lag_maxabs = abs_lags.iter().cloned().fold(0.0f64, f64::max);
    let f_lag_ent = entropy_hist(
        &xlags,
        -(MAXLAG as f64) - 0.5,
        MAXLAG as f64 + 0.5,
        (2 * MAXLAG + 1) as usize,
    );

    // 29-30. reference cross-correlation (each fragment vs R)
    let mut ref_lag_abs: Vec<f64> = Vec::new();
    let mut ref_xval: Vec<f64> = Vec::new();
    if has_traces {
        let norm_r = xcorr_norm(r);
        for (trace, &norm_t) in traces.iter().take(k).zip(&norms) {
            let (lag, xv) = best_xcorr_normed(trace, r, MAXLAG, norm_t, norm_r);
            ref_lag_abs.push((lag as f64).abs());
            ref_xval.push(xv);
        }
    }
    let f_ref_lag_mean = mean_all(&ref_lag_abs);
    let f_ref_xshape = mean_all(&ref_xval);

    // 31. observed_sum_vs_template_corr: pearson(S,R), S = unweighted sum over
    // matched fragments (fall back to all fragments if none matched).
    let sum_set: &[usize] = if matched.is_empty() {
        // build 0..k on the fly
        &[]
    } else {
        &matched
    };
    let s_trace: Vec<f64> = if has_traces {
        let t = r.len();
        let mut s = vec![0.0f64; t];
        if sum_set.is_empty() {
            for trace in traces.iter().take(k) {
                accumulate(&mut s, trace);
            }
        } else {
            for &i in sum_set {
                accumulate(&mut s, &traces[i]);
            }
        }
        s
    } else {
        Vec::new()
    };
    let f_sum_corr = if s_trace.is_empty() {
        0.0
    } else {
        pearson_vs(&s_trace, r, &r_c)
    };

    // 32-33. leave-one-out ref correlation: R^{-f} = R - pred[f]*traces[f]
    let mut loo: Vec<f64> = Vec::new();
    if has_traces {
        let t = r.len();
        for (i, trace) in traces.iter().enumerate().take(k) {
            let mut rm = r.clone();
            let lf = e.pred[i];
            for (idx, val) in rm.iter_mut().enumerate().take(t) {
                if idx < trace.len() {
                    *val -= lf * trace[idx];
                }
            }
            loo.push(pearson(trace, &rm));
        }
    }
    let f_loo_mean = mean_all(&loo);
    let f_loo_min = if loo.is_empty() {
        0.0
    } else {
        loo.iter().cloned().fold(f64::INFINITY, f64::min)
    };

    // 34. frac_frags_apex_aligned: frags whose argmax is within 1 scan of R's apex
    let f_apex_aligned = if has_traces && e.n_matched > 0 && !r.is_empty() {
        let r_ax = argmax(r);
        let aligned = matched
            .iter()
            .filter(|&&i| {
                let fa = argmax(&traces[i]);
                (fa as i64 - r_ax as i64).abs() <= 1
            })
            .count() as f64;
        aligned / e.n_matched as f64
    } else {
        0.0
    };

    // 35. top3_frag_ref_corr: mean rc over 3 largest-pred fragments (unweighted)
    let f_top3 = {
        let idx: Vec<usize> = order.iter().take(3).cloned().collect();
        mean_at(&rc, &idx)
    };

    // 36-37. b x y cross coelution and lag
    let f_by = if by_cnt > 0.0 { by_sum / by_cnt } else { 0.0 };
    let f_by_lag = if by_cnt > 0.0 {
        by_lag_sum / by_cnt
    } else {
        0.0
    };

    // 38. charge cross coelution
    let f_chg = if chg_cnt > 0.0 {
        chg_sum / chg_cnt
    } else {
        0.0
    };

    let out = vec![
        f_mean,
        f_obsw,
        f_min,
        f_std,
        f_sqmean,
        f_topk,
        n_above_09,
        f_frac08,
        f_mean_full,
        f_gain,
        f_pw,
        f_pmin,
        f_pmed,
        f_pstd,
        f_pneg,
        f_hi,
        f_lo,
        f_contrast,
        f_corr_ent,
        f_xmean,
        f_xmin,
        f_xstd,
        f_lag_mean_abs,
        f_lag_std,
        f_lag_iqr,
        f_lag_fzero,
        f_lag_maxabs,
        f_lag_ent,
        f_ref_lag_mean,
        f_ref_xshape,
        f_sum_corr,
        f_loo_mean,
        f_loo_min,
        f_apex_aligned,
        f_top3,
        f_by,
        f_by_lag,
        f_chg,
    ];
    out.into_iter().map(fin).collect()
}

// --- private helpers ---

#[inline]
fn fin(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

fn mean_all(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn mean_at(v: &[f64], idx: &[usize]) -> f64 {
    let mut s = 0.0;
    let mut n = 0.0;
    for &i in idx {
        if i < v.len() {
            s += v[i];
            n += 1.0;
        }
    }
    if n > 0.0 {
        s / n
    } else {
        0.0
    }
}

fn std_pop(v: &[f64]) -> f64 {
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

/// Median by selection (`super::median_select`), bit-identical to the sorted copy it
/// replaced; 0 for an empty `v`.
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    super::median_select(&mut v.to_vec())
}

/// Interquartile range by selection (`super::quantile_select`), bit-identical to the two
/// quantiles of the sorted copy it replaced; 0 below two values.
fn iqr(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let mut s = v.to_vec();
    let q75 = super::quantile_select(&mut s, 0.75);
    q75 - super::quantile_select(&mut s, 0.25)
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0usize;
    let mut bv = f64::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

fn accumulate(dst: &mut [f64], src: &[f64]) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] += src[i];
    }
}

/// Indices of `v` sorted by value descending (stable on ties by index).
fn sort_idx_desc(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx
}

/// Library-weighted mean of `rc` over the `k` fragments with the largest `pred`.
fn weighted_mean_topk(rc: &[f64], pred: &[f64], k: usize) -> f64 {
    if rc.is_empty() {
        return 0.0;
    }
    let order = sort_idx_desc(pred);
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for &i in order.iter().take(k) {
        if i < rc.len() {
            let w = pred[i];
            num += w * rc[i];
            den += w;
        }
    }
    if den > 0.0 {
        num / den
    } else {
        // fall back to unweighted mean over the same subset
        let idx: Vec<usize> = order.into_iter().take(k).collect();
        mean_at(rc, &idx)
    }
}

/// Mean pairwise Pearson over a subset of fragment indices, read from a precomputed
/// symmetric `k x k` matrix stored row-major in one buffer. Returns 0.0 when fewer than
/// two indices are given.
fn subset_pair_mean(corr: &[f64], k: usize, idx: &[usize]) -> f64 {
    let m = idx.len();
    if m < 2 {
        return 0.0;
    }
    let mut s = 0.0;
    let mut n = 0.0;
    for a in 0..m {
        for b in (a + 1)..m {
            let (ia, ib) = (idx[a], idx[b]);
            if ia < k && ib < k {
                s += corr[ia * k + ib];
                n += 1.0;
            }
        }
    }
    if n > 0.0 {
        s / n
    } else {
        0.0
    }
}

/// Shannon entropy (nats) of a histogram of `v` over `nbins` uniform bins in
/// [lo, hi]. Values outside the range are clamped to the edge bins.
fn entropy_hist(v: &[f64], lo: f64, hi: f64, nbins: usize) -> f64 {
    if v.is_empty() || nbins == 0 || hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
        return 0.0;
    }
    let mut counts = vec![0.0f64; nbins];
    let width = (hi - lo) / nbins as f64;
    if width.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return 0.0;
    }
    for &x in v {
        let mut bin = ((x - lo) / width).floor() as isize;
        if bin < 0 {
            bin = 0;
        }
        if bin as usize >= nbins {
            bin = nbins as isize - 1;
        }
        counts[bin as usize] += 1.0;
    }
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut ent = 0.0;
    for c in counts {
        if c > 0.0 {
            let p = c / total;
            ent -= p * p.ln();
        }
    }
    ent
}
