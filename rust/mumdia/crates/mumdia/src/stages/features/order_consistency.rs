//! Extended feature family: order_consistency.
//!
//! Prediction-free MS2-XIC quality. Every feature here is computed from the
//! observed per-fragment intensity traces alone (`Evidence.traces`,
//! `apex_idx`); none uses predicted intensities. They describe the "transpose"
//! view of the elution: at each scan the fragments are ranked by observed
//! intensity, and the features measure whether that ranking (and its magnitude
//! proportion) is preserved across the peak. A real peptidoform keeps a stable
//! fragment order over its elution; chimeric or noise matches jitter.
//!
//! This is orthogonal to the library-agreement families (similarity, coelution
//! vs the predicted-weighted reference), which all depend on predicted
//! intensities. The order-consistency features stay informative exactly where
//! prediction is wrong.
//!
//! Reference vector = the apex-scan fragment vector (fallback: the max-total
//! scan when the apex column is too sparse). Scans are weighted by their total
//! observed intensity so baseline/noise scans do not pollute the rank
//! statistics. All features are 0.0 for degenerate evidence (< 3 fragments
//! observed in the peak, or < 3 non-empty scans).
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items in
//! the same order; every value is finite (NaN/Inf -> 0.0); the length is stable.
use super::Evidence;
use crate::stats::{cosine, pearson};

pub const NAMES: &[&str] = &[
    "rank_corr_vs_apex_mean",
    "rank_corr_vs_apex_std",
    "rank_corr_adjacent_mean",
    "kendall_vs_apex_mean",
    "top1_frag_persistence",
    "top2_order_persistence",
    "argmax_frag_entropy",
    "self_cosine_vs_apex_mean",
];

const MIN_FRAGS: usize = 3;
const MIN_SCANS: usize = 3;

fn fin(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

/// Average (tie-corrected) ranks of `v`, ascending. Ties share the mean of the
/// positions they span, so Spearman via `pearson` on ranks is exact.
fn avg_rank(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j - 1) as f64 / 2.0;
        for k in i..j {
            r[idx[k]] = avg;
        }
        i = j;
    }
    r
}

/// Spearman rank correlation (Pearson on average ranks).
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    fin(pearson(&avg_rank(a), &avg_rank(b)))
}

/// Kendall tau over concordant/discordant pairs; tied pairs are ignored in both
/// numerator and denominator. 0.0 if no untied pair exists.
fn kendall(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let (mut c, mut d) = (0i64, 0i64);
    for i in 0..n {
        for j in (i + 1)..n {
            let sa = (a[i] - a[j]).partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal);
            let sb = (b[i] - b[j]).partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal);
            use std::cmp::Ordering::*;
            if sa == Equal || sb == Equal {
                continue;
            }
            if sa == sb {
                c += 1;
            } else {
                d += 1;
            }
        }
    }
    let t = c + d;
    if t == 0 { 0.0 } else { fin((c - d) as f64 / t as f64) }
}

/// Index of the maximum element (first on ties); None if all non-positive.
fn argmax_pos(v: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &x) in v.iter().enumerate() {
        if x > 0.0 && best.map_or(true, |(_, b)| x > b) {
            best = Some((i, x));
        }
    }
    best.map(|(i, _)| i)
}

pub fn values(e: &Evidence) -> Vec<f64> {
    let n = NAMES.len();
    let nfrag = e.traces.len();
    if nfrag < MIN_FRAGS {
        return vec![0.0; n];
    }
    let np = e.traces[0].len();
    if np == 0 || e.traces.iter().any(|t| t.len() != np) {
        return vec![0.0; n];
    }

    // Fragments observed anywhere in the peak window.
    let frag_ok: Vec<usize> = (0..nfrag)
        .filter(|&i| e.traces[i].iter().any(|&x| x > 0.0))
        .collect();
    if frag_ok.len() < MIN_FRAGS {
        return vec![0.0; n];
    }

    // Per-scan fragment vectors (restricted to observed fragments) and scan
    // weights (total observed intensity at the scan). Keep only non-empty scans.
    let mut cols: Vec<Vec<f64>> = Vec::new();
    let mut w: Vec<f64> = Vec::new();
    for k in 0..np {
        let col: Vec<f64> = frag_ok.iter().map(|&i| e.traces[i][k]).collect();
        let tot: f64 = col.iter().sum();
        if tot > 0.0 {
            cols.push(col);
            w.push(tot);
        }
    }
    if cols.len() < MIN_SCANS {
        return vec![0.0; n];
    }

    // Reference vector: apex-scan column if it has >= MIN_FRAGS non-zeros, else
    // the max-total scan. `apex_idx` indexes the full peak axis; map to the
    // filtered column if it survived, otherwise fall back.
    let apex_col: Vec<f64> = frag_ok.iter().map(|&i| e.traces[i][e.apex_idx]).collect();
    let apex_nz = apex_col.iter().filter(|&&x| x > 0.0).count();
    let refv: Vec<f64> = if e.apex_idx < np && apex_nz >= MIN_FRAGS {
        apex_col
    } else {
        // max-total scan among the kept columns
        let mut bi = 0;
        for i in 1..cols.len() {
            if w[i] > w[bi] {
                bi = i;
            }
        }
        cols[bi].clone()
    };
    let ref_argmax = argmax_pos(&refv);
    // ref top-2 order (indices of largest, second largest)
    let ref_top2: Option<(usize, usize)> = {
        let mut order: Vec<usize> = (0..refv.len()).collect();
        order.sort_by(|&i, &j| refv[j].partial_cmp(&refv[i]).unwrap_or(std::cmp::Ordering::Equal));
        if order.len() >= 2 && refv[order[0]] > 0.0 && refv[order[1]] > 0.0 {
            Some((order[0], order[1]))
        } else {
            None
        }
    };

    let wsum: f64 = w.iter().sum();

    // Per-scan statistics vs the reference vector, intensity-weighted.
    let mut sp: Vec<f64> = Vec::with_capacity(cols.len()); // spearman per scan
    let mut kd_acc = 0.0; // weighted kendall
    let mut cos_acc = 0.0; // weighted self-cosine
    let mut top1_acc = 0.0; // weighted top-1 persistence
    let mut top2_num = 0.0; // weighted top-2 order matches
    let mut top2_den = 0.0; // weighted scans where both top-2 frags present
    // argmax identity distribution (weighted) over fragment positions
    let mut argmax_w = vec![0.0f64; frag_ok.len()];

    for (col, &wk) in cols.iter().zip(&w) {
        sp.push(spearman(col, &refv));
        kd_acc += wk * kendall(col, &refv);
        cos_acc += wk * fin(cosine(col, &refv));
        if let (Some(a), Some(r)) = (argmax_pos(col), ref_argmax) {
            if a == r {
                top1_acc += wk;
            }
            argmax_w[a] += wk;
        }
        if let Some((hi, lo)) = ref_top2 {
            if col[hi] > 0.0 && col[lo] > 0.0 {
                top2_den += wk;
                if col[hi] > col[lo] {
                    top2_num += wk;
                }
            }
        }
    }

    // Weighted mean / std of the per-scan Spearman.
    let sp_mean = {
        let s: f64 = sp.iter().zip(&w).map(|(x, wk)| x * wk).sum();
        fin(s / wsum)
    };
    let sp_std = {
        let v: f64 = sp
            .iter()
            .zip(&w)
            .map(|(x, wk)| wk * (x - sp_mean) * (x - sp_mean))
            .sum::<f64>()
            / wsum;
        if v <= 0.0 { 0.0 } else { fin(v.sqrt()) }
    };

    // Adjacent-scan Spearman (consecutive kept columns), weighted by the mean of
    // the two scan weights.
    let adj_mean = {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..(cols.len() - 1) {
            let wk = 0.5 * (w[i] + w[i + 1]);
            num += wk * spearman(&cols[i], &cols[i + 1]);
            den += wk;
        }
        if den > 0.0 { fin(num / den) } else { 0.0 }
    };

    let kendall_mean = fin(kd_acc / wsum);
    let self_cos_mean = fin(cos_acc / wsum);
    let top1 = fin(top1_acc / wsum);
    let top2 = if top2_den > 0.0 { fin(top2_num / top2_den) } else { 0.0 };

    // Normalized Shannon entropy of the weighted argmax-fragment distribution.
    // 0 = one fragment dominates every scan (clean), 1 = argmax uniform (noisy).
    let argmax_entropy = {
        let tot: f64 = argmax_w.iter().sum();
        if tot <= 0.0 {
            0.0
        } else {
            let mut h = 0.0;
            let mut nz = 0usize;
            for &a in &argmax_w {
                if a > 0.0 {
                    let p = a / tot;
                    h -= p * p.ln();
                    nz += 1;
                }
            }
            if nz <= 1 {
                0.0
            } else {
                fin(h / (nz as f64).ln())
            }
        }
    };

    vec![
        sp_mean,
        sp_std,
        adj_mean,
        kendall_mean,
        top1,
        top2,
        argmax_entropy,
        self_cos_mean,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(traces: Vec<Vec<f64>>, apex_idx: usize) -> Evidence {
        let np = traces.first().map_or(0, |t| t.len());
        Evidence {
            axis: (0..np).map(|k| k as f64).collect(),
            traces,
            axis_full: vec![],
            traces_full: vec![],
            pred: vec![],
            obs_apex: vec![],
            is_b: vec![],
            ordinal: vec![],
            frag_charge: vec![],
            frag_mz: vec![],
            frag_obs_mz: vec![],
            mass_err_ppm: vec![],
            apex_idx,
            ref_profile: vec![],
            apex_rt: 0.0,
            rt_pred_cal: 0.0,
            rt_err: 0.0,
            gradient: 0.0,
            precursor_mz: 0.0,
            charge: 2,
            seq_len: 10,
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
        }
    }

    #[test]
    fn length_stable_on_degenerate() {
        assert_eq!(values(&ev(vec![], 0)).len(), NAMES.len());
        assert_eq!(values(&ev(vec![vec![1.0, 2.0]], 0)).len(), NAMES.len());
        assert!(values(&ev(vec![], 0)).iter().all(|x| *x == 0.0));
    }

    #[test]
    fn perfectly_consistent_order_scores_high() {
        // 3 fragments keeping the same intensity order across 4 scans, scaled by
        // a Gaussian-ish envelope: order preserved -> rank corr = 1, entropy = 0.
        let env = [0.4, 1.0, 0.8, 0.3];
        let base = [3.0, 2.0, 1.0];
        let traces: Vec<Vec<f64>> = base
            .iter()
            .map(|&b| env.iter().map(|&e| b * e).collect())
            .collect();
        let v = values(&ev(traces, 1));
        let idx = |name: &str| NAMES.iter().position(|&n| n == name).unwrap();
        assert!(v[idx("rank_corr_vs_apex_mean")] > 0.99, "rank {:?}", v);
        assert!(v[idx("top1_frag_persistence")] > 0.99);
        assert!(v[idx("argmax_frag_entropy")] < 1e-9);
        assert!(v[idx("self_cosine_vs_apex_mean")] > 0.99);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn shuffled_order_scores_low() {
        // Each scan a different argmax fragment -> unstable order: high entropy,
        // low top-1 persistence.
        let traces = vec![
            vec![3.0, 1.0, 1.0, 1.0],
            vec![1.0, 3.0, 1.0, 1.0],
            vec![1.0, 1.0, 3.0, 1.0],
            vec![1.0, 1.0, 1.0, 3.0],
        ];
        let v = values(&ev(traces, 1));
        let idx = |name: &str| NAMES.iter().position(|&n| n == name).unwrap();
        assert!(v[idx("argmax_frag_entropy")] > 0.9, "entropy {:?}", v);
        assert!(v[idx("top1_frag_persistence")] < 0.5);
    }
}
