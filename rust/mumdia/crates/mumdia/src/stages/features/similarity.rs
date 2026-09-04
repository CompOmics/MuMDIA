//! Extended feature family: similarity.
//!
//! Observed-vs-library fragment-intensity agreement scored with many
//! complementary distance/correlation kernels. The observed vector `o` is the
//! per-fragment apex intensity (0 for absent); the library vector `l` is the
//! predicted intensity. `_matched` variants restrict to `o_i > 0`; `_area`
//! variants replace `o` with per-fragment peak-XIC trapezoid area. `on`/`ln`
//! denote sum-normalized `o`/`l`.
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items in
//! the same order. Every value is finite; degenerate cases return 0.0. Shared
//! kernels come from `crate::stats`; the parent helpers `mean`/`normalize_sum`
//! are reused rather than reimplemented.
use super::Evidence;
use super::{mean, normalize_sum};
use crate::stats::{cosine, pearson, spectral_angle};

pub const NAMES: &[&str] = &[
    "spectrum_cosine_matched",
    "spectrum_cosine_sqrt",
    "spectrum_cosine_log",
    "spectral_angle",
    "spectral_angle_sqrt",
    "spectral_angle_matched",
    "pearson_intensity_matched",
    "pearson_intensity_log",
    "spearman_intensity",
    "spearman_intensity_matched",
    "kendall_tau_intensity",
    "dot_product_raw",
    "dot_product_norm",
    "library_recall_intensity",
    "manhattan_sim",
    "manhattan_sqrt",
    "rmsd_norm",
    "mae_norm",
    "mse_log",
    "mae_weighted_pred",
    "abs_diff_q3",
    "max_positive_residual",
    "chebyshev_dist",
    "minkowski_p3",
    "bray_curtis",
    "bray_curtis_sqrt",
    "canberra",
    "canberra_matched",
    "wave_hedges",
    "chi_square_pearson",
    "chi_square_symmetric",
    "divergence_distance",
    "bhattacharyya_coef",
    "hellinger",
    "squared_chord",
    "harmonic_mean_sim",
    "jaccard_presence",
    "dice_presence",
    "intensity_weighted_pearson",
    "regression_slope",
    "gini_diff",
    "wasserstein_mz",
    "footrule_norm",
    "rank_overlap_top3",
    "top1_frag_match",
    "top1_predicted_observed",
    "frac_top3_predicted_observed",
    "count_strong_predicted_absent",
    "frac_predicted_absent",
    "cosine_area",
    "pearson_area",
    "spectral_angle_area",
    "cosine_fullwindow",
    "stein_scott_weighted_dot",
    // Unbounded / granular spectral-evidence scores. The normalized cosines above
    // saturate near 1 for any decent match (compressed dynamic range, ~25-45% of
    // values at the ceiling), so they cannot rank AMONG good PSMs. These three keep
    // resolution at the top end, matching DIA-NN's unbounded `Evidence`.
    "log_dot_product",
    "spectral_log_evidence",
    "scribe_score",
    // Summed-within-peak (area) twins of the three evidence scores above: each
    // fragment is summed over the RT-bounded scans before scoring, averaging out
    // per-scan noise (area beats apex empirically for cosine/angle/pearson).
    "log_dot_product_area",
    "spectral_log_evidence_area",
    "scribe_score_area",
    // Exclude low-ordinal ions (b1/b2/y1/y2 are frequently co-isolated-precursor
    // contamination): cosine over ordinal >= 3 fragments only.
    "cosine_high_ordinal",
    // Robust cosine trajectory: iteratively drop the fragment whose normalized
    // observed intensity deviates most from predicted (interference outlier) and
    // score after 1, 2, 3 removals. The trajectory discriminates a clean peptide
    // (barely improves) from a chimeric one (jumps when the interferent is dropped).
    "cosine_robust_trim1",
    "cosine_robust_trim2",
    "cosine_robust_trim3",
];

const EPS: f64 = 1e-9;

/// Replace non-finite with 0.0.
#[inline]
fn fin(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Elementwise sqrt of the max(0,·) values.
fn vsqrt(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x.max(0.0).sqrt()).collect()
}

/// Elementwise ln(1+max(0,·)).
fn vlog1p(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x.max(0.0).ln_1p()).collect()
}

/// Average-tie ascending ranks (1-based).
fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &p in &idx[i..=j] {
            r[p] = avg;
        }
        i = j + 1;
    }
    r
}

/// Kendall tau-b over paired samples.
fn kendall_tau_b(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let (mut c, mut d, mut tl, mut to) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        for j in (i + 1)..n {
            let da = a[i] - a[j];
            let db = b[i] - b[j];
            if da == 0.0 && db == 0.0 {
                continue;
            } else if da == 0.0 {
                tl += 1.0;
            } else if db == 0.0 {
                to += 1.0;
            } else if (da > 0.0) == (db > 0.0) {
                c += 1.0;
            } else {
                d += 1.0;
            }
        }
    }
    let denom = ((c + d + tl) * (c + d + to)).sqrt();
    if denom <= 0.0 {
        0.0
    } else {
        fin((c - d) / denom)
    }
}

/// Linear-interpolated quantile (q in [0,1]).
fn quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let pos = q.clamp(0.0, 1.0) * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if hi >= s.len() {
        return fin(s[s.len() - 1]);
    }
    fin(s[lo] + (s[hi] - s[lo]) * (pos - lo as f64))
}

/// Gini coefficient of non-negative values.
fn gini(v: &[f64]) -> f64 {
    let mut x: Vec<f64> = v.iter().filter(|&&a| a >= 0.0).copied().collect();
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = x.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    x.sort_by(|a, b| a.total_cmp(b));
    let mut acc = 0.0f64;
    for (i, &xi) in x.iter().enumerate() {
        acc += (i as f64 + 1.0) * xi;
    }
    let g = 2.0 * acc / (n as f64 * sum) - (n as f64 + 1.0) / n as f64;
    fin(g)
}

/// Weighted Pearson correlation with non-negative weights.
fn weighted_pearson(a: &[f64], b: &[f64], w: &[f64]) -> f64 {
    let n = a.len().min(b.len()).min(w.len());
    if n < 2 {
        return 0.0;
    }
    let sw: f64 = w[..n].iter().sum();
    if sw <= 0.0 {
        return 0.0;
    }
    let ma: f64 = (0..n).map(|i| w[i] * a[i]).sum::<f64>() / sw;
    let mb: f64 = (0..n).map(|i| w[i] * b[i]).sum::<f64>() / sw;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += w[i] * da * db;
        va += w[i] * da * da;
        vb += w[i] * db * db;
    }
    if va <= 0.0 || vb <= 0.0 {
        return 0.0;
    }
    fin(cov / (va.sqrt() * vb.sqrt()))
}

/// Trapezoid integral of y over x, clamped to be non-negative.
fn trapz(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mut s = 0.0f64;
    for t in 0..n - 1 {
        let dx = x[t + 1] - x[t];
        s += 0.5 * (y[t] + y[t + 1]) * dx;
    }
    fin(s).max(0.0)
}

/// Indices of the `k` largest strictly-positive entries, descending.
fn topk_pos(v: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).filter(|&i| v[i] > 0.0).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    idx.truncate(k);
    idx
}

/// Index of the maximum entry, or None if empty.
fn argmax(v: &[f64]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, &x) in v.iter().enumerate() {
        match best {
            None => best = Some(i),
            Some(b) if x > v[b] => best = Some(i),
            _ => {}
        }
    }
    best
}

pub fn values(e: &Evidence) -> Vec<f64> {
    // Aligned observed/library over the shorter of the two arrays.
    let n = e.pred.len().min(e.obs_apex.len());
    let l: Vec<f64> = e.pred[..n].iter().map(|&x| x.max(0.0)).collect();
    let o: Vec<f64> = e.obs_apex[..n].iter().map(|&x| x.max(0.0)).collect();

    // Matched-only pairs (o_i > 0).
    let mut om = Vec::new();
    let mut lm = Vec::new();
    for i in 0..n {
        if o[i] > 0.0 {
            om.push(o[i]);
            lm.push(l[i]);
        }
    }

    // Sum-normalized vectors.
    let on = normalize_sum(&o);
    let ln = normalize_sum(&l);

    // 1. spectrum_cosine_matched
    let f_cos_matched = fin(cosine(&om, &lm));
    // 2. spectrum_cosine_sqrt
    let f_cos_sqrt = fin(cosine(&vsqrt(&o), &vsqrt(&l)));
    // 3. spectrum_cosine_log
    let f_cos_log = fin(cosine(&vlog1p(&o), &vlog1p(&l)));
    // 4. spectral_angle
    let f_sa = fin(spectral_angle(&o, &l));
    // 5. spectral_angle_sqrt
    let f_sa_sqrt = fin(spectral_angle(&vsqrt(&o), &vsqrt(&l)));
    // 6. spectral_angle_matched
    let f_sa_matched = fin(spectral_angle(&om, &lm));
    // 7. pearson_intensity_matched
    let f_pear_matched = fin(pearson(&lm, &om));
    // 8. pearson_intensity_log
    let f_pear_log = fin(pearson(&vlog1p(&l), &vlog1p(&o)));
    // 9. spearman_intensity
    let f_spearman = fin(pearson(&ranks(&o), &ranks(&l)));
    // 10. spearman_intensity_matched
    let f_spearman_m = fin(pearson(&ranks(&om), &ranks(&lm)));
    // 11. kendall_tau_intensity
    let f_kendall = kendall_tau_b(&l, &o);
    // 12. dot_product_raw
    let f_dot_raw = fin((0..n).map(|i| l[i] * o[i]).sum());
    // 13. dot_product_norm
    let f_dot_norm = fin((0..on.len().min(ln.len())).map(|i| on[i] * ln[i]).sum());
    // 14. library_recall_intensity
    let tot_l: f64 = l.iter().sum();
    let seen_l: f64 = (0..n).filter(|&i| o[i] > 0.0).map(|i| l[i]).sum();
    let f_recall = if tot_l > 0.0 {
        fin(seen_l / tot_l)
    } else {
        0.0
    };

    // Per-fragment |on - ln| and signed residual on sum-normalized vectors.
    let m = on.len().min(ln.len());
    let abs_diff: Vec<f64> = (0..m).map(|i| (on[i] - ln[i]).abs()).collect();

    // 15. manhattan_sim = 1 - 0.5 * L1(on, ln)
    let l1: f64 = abs_diff.iter().sum();
    let f_manhattan = fin(1.0 - 0.5 * l1);
    // 16. manhattan_sqrt: L1 distance of sqrt-then-renormalized vectors
    let osn = normalize_sum(&vsqrt(&o));
    let lsn = normalize_sum(&vsqrt(&l));
    let f_manhattan_sqrt = {
        let mm = osn.len().min(lsn.len());
        fin((0..mm).map(|i| (osn[i] - lsn[i]).abs()).sum())
    };
    // 17. rmsd_norm
    let sq_diff: Vec<f64> = (0..m).map(|i| (on[i] - ln[i]).powi(2)).collect();
    let f_rmsd = fin(mean(&sq_diff).sqrt());
    // 18. mae_norm
    let f_mae = fin(mean(&abs_diff));
    // 19. mse_log
    let log_sq: Vec<f64> = (0..m)
        .map(|i| (((ln[i] + 1e-3).log2()) - ((on[i] + 1e-3).log2())).powi(2))
        .collect();
    let f_mse_log = fin(mean(&log_sq));
    // 20. mae_weighted_pred (weights = raw library l_i)
    let f_mae_w = {
        let sw: f64 = (0..m).map(|i| l[i]).sum();
        if sw > 0.0 {
            fin((0..m).map(|i| l[i] * abs_diff[i]).sum::<f64>() / sw)
        } else {
            0.0
        }
    };
    // 21. abs_diff_q3
    let f_q3 = quantile(&abs_diff, 0.75);
    // 22. max_positive_residual
    let f_max_res = fin((0..m).map(|i| on[i] - ln[i]).fold(0.0f64, f64::max));
    // 23. chebyshev_dist
    let f_cheby = fin(abs_diff.iter().cloned().fold(0.0f64, f64::max));
    // 24. minkowski_p3
    let f_mink = {
        let s: f64 = abs_diff.iter().map(|&d| d.powi(3)).sum();
        fin(s.max(0.0).cbrt())
    };
    // 25. bray_curtis (Ruzicka: sum min / sum max on sumnorm)
    let f_bray = {
        let mut smin = 0.0f64;
        let mut smax = 0.0f64;
        for i in 0..m {
            smin += on[i].min(ln[i]);
            smax += on[i].max(ln[i]);
        }
        if smax > 0.0 {
            fin(smin / smax)
        } else {
            0.0
        }
    };
    // 26. bray_curtis_sqrt (sqrt + max-normalized, min/max ratio)
    let f_bray_sqrt = {
        let os = vsqrt(&o);
        let ls = vsqrt(&l);
        let omax = os.iter().cloned().fold(0.0f64, f64::max);
        let lmax = ls.iter().cloned().fold(0.0f64, f64::max);
        if omax > 0.0 && lmax > 0.0 {
            let mut smin = 0.0f64;
            let mut smax = 0.0f64;
            for i in 0..os.len().min(ls.len()) {
                let a = os[i] / omax;
                let b = ls[i] / lmax;
                smin += a.min(b);
                smax += a.max(b);
            }
            if smax > 0.0 {
                fin(smin / smax)
            } else {
                0.0
            }
        } else {
            0.0
        }
    };
    // 27. canberra (all N, on+ln>0)
    let f_canberra = {
        let mut s = 0.0f64;
        for i in 0..m {
            let den = on[i] + ln[i];
            if den > 0.0 {
                s += (on[i] - ln[i]).abs() / den;
            }
        }
        fin(s)
    };
    // 28. canberra_matched (matched, / n_matched)
    let f_canberra_m = {
        let onm = normalize_sum(&om);
        let lnm = normalize_sum(&lm);
        let mm = onm.len().min(lnm.len());
        if mm > 0 {
            let mut s = 0.0f64;
            for i in 0..mm {
                let den = onm[i] + lnm[i];
                if den > 0.0 {
                    s += (onm[i] - lnm[i]).abs() / den;
                }
            }
            fin(s / mm as f64)
        } else {
            0.0
        }
    };
    // 29. wave_hedges (sum |on-ln|/max(on,ln))
    let f_wave = {
        let mut s = 0.0f64;
        for i in 0..m {
            let mx = on[i].max(ln[i]);
            if mx > 0.0 {
                s += (on[i] - ln[i]).abs() / mx;
            }
        }
        fin(s)
    };
    // 30. chi_square_pearson (predicted as expected)
    let f_chi_p = {
        let mut s = 0.0f64;
        for i in 0..m {
            s += (on[i] - ln[i]).powi(2) / (ln[i] + EPS);
        }
        fin(s)
    };
    // 31. chi_square_symmetric
    let f_chi_s = {
        let mut s = 0.0f64;
        for i in 0..m {
            s += (on[i] - ln[i]).powi(2) / (on[i] + ln[i] + EPS);
        }
        fin(s)
    };
    // 32. divergence_distance
    let f_div = {
        let mut s = 0.0f64;
        for i in 0..m {
            let den = on[i] + ln[i];
            if den > 0.0 {
                s += ((on[i] - ln[i]) / den).powi(2);
            }
        }
        fin(2.0 * s)
    };
    // 33. bhattacharyya_coef
    let f_bc = {
        let mut s = 0.0f64;
        for i in 0..m {
            s += (on[i] * ln[i]).max(0.0).sqrt();
        }
        fin(s)
    };
    // 34. hellinger
    let f_hellinger = fin((1.0 - f_bc).max(0.0).sqrt());
    // 35. squared_chord
    let f_sqchord = {
        let mut s = 0.0f64;
        for i in 0..m {
            s += (on[i].max(0.0).sqrt() - ln[i].max(0.0).sqrt()).powi(2);
        }
        fin(s)
    };
    // 36. harmonic_mean_sim
    let f_harm = {
        let mut s = 0.0f64;
        for i in 0..m {
            s += on[i] * ln[i] / (on[i] + ln[i] + EPS);
        }
        fin(2.0 * s)
    };

    // Presence sets.
    let max_o = o.iter().cloned().fold(0.0f64, f64::max);
    let set_pred: Vec<bool> = (0..n).map(|i| l[i] > 0.0).collect();
    let set_obs_thr: Vec<bool> = (0..n).map(|i| o[i] > 0.01 * max_o && o[i] > 0.0).collect();
    let set_obs: Vec<bool> = (0..n).map(|i| o[i] > 0.0).collect();
    // 37. jaccard_presence (predicted vs observed-over-1%-max)
    let f_jaccard = {
        let inter = (0..n).filter(|&i| set_pred[i] && set_obs_thr[i]).count();
        let uni = (0..n).filter(|&i| set_pred[i] || set_obs_thr[i]).count();
        if uni > 0 {
            fin(inter as f64 / uni as f64)
        } else {
            0.0
        }
    };
    // 38. dice_presence (predicted vs observed-present)
    let f_dice = {
        let inter = (0..n).filter(|&i| set_pred[i] && set_obs[i]).count();
        let a = set_pred.iter().filter(|&&b| b).count();
        let b = set_obs.iter().filter(|&&b| b).count();
        if a + b > 0 {
            fin(2.0 * inter as f64 / (a + b) as f64)
        } else {
            0.0
        }
    };
    // 39. intensity_weighted_pearson (weights = library l)
    let f_wpear = weighted_pearson(&l, &o, &l);
    // 40. regression_slope: cov(l,o)/var(l)
    let f_slope = {
        if n >= 2 {
            let ml = mean(&l);
            let mo = mean(&o);
            let mut cov = 0.0f64;
            let mut var = 0.0f64;
            for i in 0..n {
                let dl = l[i] - ml;
                cov += dl * (o[i] - mo);
                var += dl * dl;
            }
            if var > 0.0 {
                fin(cov / var)
            } else {
                0.0
            }
        } else {
            0.0
        }
    };
    // 41. gini_diff: Gini(o_matched) - Gini(l)
    let f_gini = fin(gini(&om) - gini(&l));
    // 42. wasserstein_mz: EMD over sumnorm CDFs ordered by theoretical m/z
    let f_wass = {
        let mz_n = n.min(e.frag_mz.len());
        if mz_n >= 2 {
            let mut idx: Vec<usize> = (0..mz_n).collect();
            idx.sort_by(|&a, &b| e.frag_mz[a].total_cmp(&e.frag_mz[b]));
            let mut cumo = 0.0f64;
            let mut cuml = 0.0f64;
            let mut w = 0.0f64;
            for j in 0..mz_n - 1 {
                cumo += on.get(idx[j]).copied().unwrap_or(0.0);
                cuml += ln.get(idx[j]).copied().unwrap_or(0.0);
                let dmz = (e.frag_mz[idx[j + 1]] - e.frag_mz[idx[j]]).abs();
                w += (cumo - cuml).abs() * dmz;
            }
            fin(w)
        } else {
            0.0
        }
    };
    // 43. footrule_norm: 1 - sum|rank_l - rank_o| / floor(N^2/2), descending ranks
    let f_footrule = {
        if n >= 2 {
            let neg_l: Vec<f64> = l.iter().map(|&x| -x).collect();
            let neg_o: Vec<f64> = o.iter().map(|&x| -x).collect();
            let rl = ranks(&neg_l);
            let ro = ranks(&neg_o);
            let disp: f64 = (0..n).map(|i| (rl[i] - ro[i]).abs()).sum();
            let denom = ((n * n) / 2) as f64;
            if denom > 0.0 {
                fin(1.0 - disp / denom)
            } else {
                0.0
            }
        } else {
            0.0
        }
    };
    // 44. rank_overlap_top3
    let f_rank3 = {
        let tl = topk_pos(&l, 3);
        let to = topk_pos(&o, 3);
        let inter = tl.iter().filter(|i| to.contains(i)).count();
        fin(inter as f64 / 3.0)
    };
    // 45. top1_frag_match
    let f_top1_match = match (argmax(&l), argmax(&o)) {
        (Some(a), Some(b)) if o[b] > 0.0 && l[a] > 0.0 && a == b => 1.0,
        _ => 0.0,
    };
    // 46. top1_predicted_observed
    let f_top1_pred_obs = match argmax(&l) {
        Some(a) if l[a] > 0.0 && o[a] > 0.0 => 1.0,
        _ => 0.0,
    };
    // 47. frac_top3_predicted_observed
    let f_top3_pred_obs = {
        let tl = topk_pos(&l, 3);
        if tl.is_empty() {
            0.0
        } else {
            let cnt = tl.iter().filter(|&&i| o[i] > 0.0).count();
            fin(cnt as f64 / 3.0)
        }
    };
    // 48. count_strong_predicted_absent
    let max_l = l.iter().cloned().fold(0.0f64, f64::max);
    let f_strong_absent = {
        if max_l > 0.0 {
            (0..n)
                .filter(|&i| l[i] >= 0.1 * max_l && o[i] == 0.0)
                .count() as f64
        } else {
            0.0
        }
    };
    // 49. frac_predicted_absent
    let f_frac_absent = {
        let np = e.n_predicted as f64;
        let nm = e.n_matched as f64;
        if np > 0.0 {
            fin((np - nm) / np)
        } else {
            0.0
        }
    };

    // Area vectors (peak window and full window) over all K predicted fragments.
    let kk = e.pred.len();
    let lk: Vec<f64> = e.pred.iter().map(|&x| x.max(0.0)).collect();
    let area: Vec<f64> = (0..kk)
        .map(|i| {
            if i < e.traces.len() {
                trapz(&e.axis, &e.traces[i])
            } else {
                0.0
            }
        })
        .collect();
    let area_full: Vec<f64> = (0..kk)
        .map(|i| {
            if i < e.traces_full.len() {
                trapz(&e.axis_full, &e.traces_full[i])
            } else {
                0.0
            }
        })
        .collect();
    // 50. cosine_area
    let f_cos_area = fin(cosine(&lk, &area));
    // 51. pearson_area
    let f_pear_area = fin(pearson(&lk, &area));
    // 52. spectral_angle_area
    let f_sa_area = fin(spectral_angle(&lk, &area));
    // 53. cosine_fullwindow
    let f_cos_full = fin(cosine(&lk, &area_full));
    // 54. stein_scott_weighted_dot
    let f_stein = {
        let mz_n = n.min(e.frag_mz.len());
        if mz_n >= 1 {
            let mut wo = Vec::with_capacity(mz_n);
            let mut wl = Vec::with_capacity(mz_n);
            for i in 0..mz_n {
                let mz3 = e.frag_mz[i].max(0.0).powi(3);
                wo.push(mz3 * o[i].max(0.0).powf(0.6));
                wl.push(mz3 * l[i].max(0.0).powf(0.6));
            }
            fin(cosine(&wo, &wl))
        } else {
            0.0
        }
    };

    // --- Unbounded / granular spectral evidence (address cosine saturation) ---
    // log_dot_product: log-scaled raw dot product. Grows with the number and
    // intensity of agreeing fragments and never hits a [0,1] ceiling.
    let dot_raw: f64 = (0..n).map(|i| o[i] * l[i]).sum();
    let f_log_dot = fin((1.0 + dot_raw).ln());
    // spectral_log_evidence: predicted-weighted accumulation of log observed
    // intensity over matched fragments (DIA-NN `Evidence` analog). `ln` is the
    // sum-normalized library vector, so each fragment's log-observed-intensity is
    // weighted by its predicted importance; unbounded, so it keeps ranking good PSMs.
    let f_log_evid = fin((0..n).map(|i| ln[i] * o[i].ln_1p()).sum());
    // scribe_score: -ln of the sum of squared differences between the sum-normalized
    // observed and predicted vectors (EncyclopeDIA Scribe). Diverges to large values
    // for near-perfect matches, i.e. high resolution exactly where cosine saturates.
    let f_scribe = fin(-((0..n).map(|i| (on[i] - ln[i]).powi(2)).sum::<f64>() + EPS).ln());
    // Summed-within-peak (area) twins: replace apex `o` with the per-fragment
    // peak-XIC area (`area`, aligned to `lk` over all predicted fragments).
    let la = normalize_sum(&lk);
    let aa = normalize_sum(&area);
    let dot_area: f64 = (0..kk).map(|i| area[i] * lk[i]).sum();
    let f_log_dot_area = fin((1.0 + dot_area).ln());
    let f_log_evid_area = fin((0..kk).map(|i| la[i] * area[i].ln_1p()).sum());
    let f_scribe_area = fin(-((0..kk).map(|i| (aa[i] - la[i]).powi(2)).sum::<f64>() + EPS).ln());

    // High-ordinal cosine: drop b1/b2/y1/y2 (ordinal <= 2), which are commonly
    // shared with a co-isolated precursor, then cosine over the rest.
    let f_cos_hi_ord = {
        let keep: Vec<usize> = (0..n)
            .filter(|&i| i < e.ordinal.len() && e.ordinal[i] >= 3)
            .collect();
        if keep.len() >= 2 {
            let oh: Vec<f64> = keep.iter().map(|&i| o[i]).collect();
            let lh: Vec<f64> = keep.iter().map(|&i| l[i]).collect();
            fin(cosine(&oh, &lh))
        } else {
            fin(cosine(&o, &l))
        }
    };
    // Robust trimmed-cosine trajectory: after removing the 1st/2nd/3rd worst-
    // residual fragment (renormalizing each round), record the cosine. Stops early
    // when <=3 fragments remain (later levels hold the last computed value).
    let (f_cos_trim1, f_cos_trim2, f_cos_trim3) = {
        let base = fin(cosine(&o, &l));
        let mut out = [base, base, base];
        let mut keep: Vec<usize> = (0..n).collect();
        for step in 0..3 {
            if keep.len() <= 3 {
                break;
            }
            let os: Vec<f64> = keep.iter().map(|&i| o[i]).collect();
            let ls: Vec<f64> = keep.iter().map(|&i| l[i]).collect();
            let osn = normalize_sum(&os);
            let lsn = normalize_sum(&ls);
            let mut worst = 0usize;
            let mut wd = -1.0f64;
            for j in 0..keep.len() {
                let d = (osn[j] - lsn[j]).abs();
                if d > wd {
                    wd = d;
                    worst = j;
                }
            }
            keep.remove(worst);
            let ot: Vec<f64> = keep.iter().map(|&i| o[i]).collect();
            let lt: Vec<f64> = keep.iter().map(|&i| l[i]).collect();
            // Do not let trimming collapse to an all-absent observed vector: the
            // trim ranks by residual and can remove every present fragment first,
            // leaving only predicted-but-absent ones, for which cosine is a
            // spurious 0.0 that reads as decoy-like for few-fragment peptides.
            // Stop and hold the last non-degenerate value instead.
            if ot.iter().map(|x| x.abs()).sum::<f64>() <= EPS {
                break;
            }
            let c = fin(cosine(&ot, &lt));
            for value in out.iter_mut().skip(step) {
                *value = c;
            }
        }
        (out[0], out[1], out[2])
    };

    vec![
        f_cos_matched,
        f_cos_sqrt,
        f_cos_log,
        f_sa,
        f_sa_sqrt,
        f_sa_matched,
        f_pear_matched,
        f_pear_log,
        f_spearman,
        f_spearman_m,
        f_kendall,
        f_dot_raw,
        f_dot_norm,
        f_recall,
        f_manhattan,
        f_manhattan_sqrt,
        f_rmsd,
        f_mae,
        f_mse_log,
        f_mae_w,
        f_q3,
        f_max_res,
        f_cheby,
        f_mink,
        f_bray,
        f_bray_sqrt,
        f_canberra,
        f_canberra_m,
        f_wave,
        f_chi_p,
        f_chi_s,
        f_div,
        f_bc,
        f_hellinger,
        f_sqchord,
        f_harm,
        f_jaccard,
        f_dice,
        f_wpear,
        f_slope,
        f_gini,
        f_wass,
        f_footrule,
        f_rank3,
        f_top1_match,
        f_top1_pred_obs,
        f_top3_pred_obs,
        f_strong_absent,
        f_frac_absent,
        f_cos_area,
        f_pear_area,
        f_sa_area,
        f_cos_full,
        f_stein,
        f_log_dot,
        f_log_evid,
        f_scribe,
        f_log_dot_area,
        f_log_evid_area,
        f_scribe_area,
        f_cos_hi_ord,
        f_cos_trim1,
        f_cos_trim2,
        f_cos_trim3,
    ]
}
