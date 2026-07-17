//! Native semi-supervised rescorer (PLAN.md Stage F, MVP `native_tda`): a
//! Percolator/Mokapot-style linear model. Standardize features, then for each
//! cross-validation fold train a logistic regression on a positive set of
//! confident targets versus all decoys, iterating the positive-set selection.
//! Deterministic (weights start at zero, no RNG). Port features, not classifiers
//! (PLAN.md Section 8.5 item; the model is intentionally simple and swappable).

use crate::fdr::target_decoy_q;
use rayon::prelude::*;

/// Column mean/std over a SUBSET of rows (guarded; std < 1e-9 -> 1.0). Fitting the
/// scaler on the training fold only avoids leaking test-fold statistics into the
/// standardization.
fn fit_standardizer(x: &[Vec<f64>], idx: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let d = x.first().map(|r| r.len()).unwrap_or(0);
    let n = idx.len().max(1) as f64;
    let mut mean = vec![0.0; d];
    for &i in idx {
        for j in 0..d {
            mean[j] += x[i][j];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut std = vec![0.0; d];
    for &i in idx {
        for j in 0..d {
            let dd = x[i][j] - mean[j];
            std[j] += dd * dd;
        }
    }
    for s in &mut std {
        *s = (*s / n).sqrt();
        if *s < 1e-9 {
            *s = 1.0;
        }
    }
    (mean, std)
}

#[inline]
fn std_row(row: &[f64], mean: &[f64], std: &[f64]) -> Vec<f64> {
    (0..row.len()).map(|j| (row[j] - mean[j]) / std[j]).collect()
}

/// Logistic regression by full-batch gradient descent with L2. Weight[0] = bias.
fn logreg_fit(rows: &[&[f64]], y: &[f64], l2: f64, epochs: usize, lr: f64) -> Vec<f64> {
    let d = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut w = vec![0.0f64; d + 1];
    if rows.is_empty() {
        return w;
    }
    let n = rows.len() as f64;
    for _ in 0..epochs {
        let mut grad = vec![0.0f64; d + 1];
        for (r, &yi) in rows.iter().zip(y) {
            let mut z = w[0];
            for j in 0..d {
                z += w[j + 1] * r[j];
            }
            let p = 1.0 / (1.0 + (-z).exp());
            let err = p - yi;
            grad[0] += err;
            for j in 0..d {
                grad[j + 1] += err * r[j];
            }
        }
        w[0] -= lr * grad[0] / n;
        for j in 0..d {
            w[j + 1] -= lr * (grad[j + 1] / n + l2 * w[j + 1]);
        }
    }
    w
}

fn score_row(w: &[f64], r: &[f64]) -> f64 {
    let mut z = w[0];
    for j in 0..r.len() {
        z += w[j + 1] * r[j];
    }
    z
}

pub struct RescoreInput<'a> {
    pub features: &'a [Vec<f64>],
    pub is_decoy: &'a [bool],
    /// Cross-validation fold key. Use base_peptide_id so every charge/mod variant
    /// of a peptide lands in the same fold (no peptide leaks train<->test).
    pub fold_key: &'a [u32],
    pub init_score: &'a [f64],
    pub folds: usize,
    pub num_iter: usize,
    pub train_fdr: f64,
}

/// Run the semi-supervised rescorer, returning a discriminant score per PSM.
pub fn percolator_lite(inp: RescoreInput) -> Vec<f64> {
    let n = inp.features.len();
    if n == 0 {
        return Vec::new();
    }
    let folds = inp.folds.max(1);

    // Fold assignment by fold_key (base peptide): all charge/mod variants of a
    // peptide share a fold, so none leaks between train and test.
    let fold_of: Vec<usize> = inp.fold_key.iter().map(|c| (*c as usize) % folds).collect();

    // Folds are independent: each fits its own scaler + weights on its training
    // rows and scores only its own (disjoint) test set. Standardization is fit on
    // the TRAIN fold only (no test leakage), which is why each fold owns its scaler.
    let per_fold: Vec<Vec<(usize, f64)>> = (0..folds)
        .into_par_iter()
        .map(|test_fold| {
            let train_idx: Vec<usize> = (0..n).filter(|&i| fold_of[i] != test_fold).collect();
            let test_idx: Vec<usize> = (0..n).filter(|&i| fold_of[i] == test_fold).collect();
            if train_idx.is_empty() || test_idx.is_empty() {
                return Vec::new();
            }
            let (mean, std) = fit_standardizer(inp.features, &train_idx);
            // standardized train matrix (owned, so we can take &[f64] slices)
            let xtr: Vec<Vec<f64>> = train_idx
                .iter()
                .map(|&i| std_row(&inp.features[i], &mean, &std))
                .collect();
            let mut train_scores: Vec<f64> = train_idx.iter().map(|&i| inp.init_score[i]).collect();
            let mut w = vec![0.0; inp.features[0].len() + 1];
            let mut sd: Vec<(f64, bool)> = Vec::with_capacity(train_idx.len());
            for _ in 0..inp.num_iter.max(1) {
                sd.clear();
                sd.extend(
                    train_idx
                        .iter()
                        .enumerate()
                        .map(|(k, &i)| (train_scores[k], inp.is_decoy[i])),
                );
                let q = target_decoy_q(&sd);
                // positive set: confident targets; negatives: all decoys
                let mut rows: Vec<&[f64]> = Vec::new();
                let mut ys: Vec<f64> = Vec::new();
                let mut n_pos = 0;
                for (k, &i) in train_idx.iter().enumerate() {
                    if inp.is_decoy[i] {
                        rows.push(&xtr[k]);
                        ys.push(0.0);
                    } else if q[k] <= inp.train_fdr {
                        rows.push(&xtr[k]);
                        ys.push(1.0);
                        n_pos += 1;
                    }
                }
                // fallback: if too few confident targets, take the top-scoring half
                if n_pos < 10 {
                    let mut order: Vec<usize> = (0..train_idx.len()).collect();
                    order.sort_by(|&a, &b| train_scores[b].partial_cmp(&train_scores[a]).unwrap());
                    let take = (train_idx.len() / 2).max(1);
                    rows.clear();
                    ys.clear();
                    for (rank, &k) in order.iter().enumerate() {
                        let i = train_idx[k];
                        if inp.is_decoy[i] {
                            rows.push(&xtr[k]);
                            ys.push(0.0);
                        } else if rank < take {
                            rows.push(&xtr[k]);
                            ys.push(1.0);
                        }
                    }
                }
                w = logreg_fit(&rows, &ys, 1e-3, 200, 0.5);
                train_scores = (0..train_idx.len()).map(|k| score_row(&w, &xtr[k])).collect();
            }
            // score the held-out test fold with this fold's scaler + weights
            test_idx
                .iter()
                .map(|&i| (i, score_row(&w, &std_row(&inp.features[i], &mean, &std))))
                .collect()
        })
        .collect();

    let mut final_score = inp.init_score.to_vec();
    for fold_scores in per_fold {
        for (i, s) in fold_scores {
            final_score[i] = s;
        }
    }
    final_score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_targets_from_decoys() {
        // targets have high feature[0], decoys low, plus noise
        let mut features = Vec::new();
        let mut is_decoy = Vec::new();
        let mut cid = Vec::new();
        let mut init = Vec::new();
        for i in 0..200 {
            let decoy = i % 2 == 0;
            let base = if decoy { 0.0 } else { 3.0 };
            let noise = ((i * 7 % 5) as f64) * 0.1;
            features.push(vec![base + noise, noise]);
            is_decoy.push(decoy);
            cid.push(i as u32);
            init.push(base + noise); // init score already discriminates
        }
        let s = percolator_lite(RescoreInput {
            features: &features,
            is_decoy: &is_decoy,
            fold_key: &cid,
            init_score: &init,
            folds: 3,
            num_iter: 5,
            train_fdr: 0.05,
        });
        // mean target score should exceed mean decoy score
        let (mut ts, mut tn, mut ds, mut dn) = (0.0, 0, 0.0, 0);
        for i in 0..s.len() {
            if is_decoy[i] {
                ds += s[i];
                dn += 1;
            } else {
                ts += s[i];
                tn += 1;
            }
        }
        assert!(ts / tn as f64 > ds / dn as f64);
    }
}
