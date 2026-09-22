//! Native semi-supervised rescorer (docs/11_compete_rescore_fdr.md, MVP
//! `native_tda`): a Percolator/Mokapot-style linear model. Standardize features,
//! then for each cross-validation fold train a logistic regression on a positive
//! set of confident targets versus all decoys, iterating the positive-set
//! selection. Deterministic (weights start at zero, no RNG). Port features, not
//! classifiers; the model is intentionally simple and swappable.

use crate::fdr::target_decoy_q;
use rayon::prelude::*;

/// Column mean/std over a SUBSET of rows (guarded; std < 1e-9 -> 1.0). Fitting the
/// scaler on the training fold only avoids leaking test-fold statistics into the
/// standardization.
/// Row-major feature matrix: ONE flat allocation of `rows * n_features` values, instead of
/// a `Vec<f64>` per PSM.
///
/// The per-row form cost a 24-byte `Vec` header plus allocator overhead and a separate heap
/// block for every PSM, and building it required the whole table column-major first, so a
/// second full copy of the matrix was alive during the load. Rows are contiguous here, which
/// is also the access pattern of every consumer ([`FeatureMatrix::row`], the standardizer,
/// the sidecar writers).
pub struct FeatureMatrix {
    /// Feature values, stored f32 (4 B) and widened to f64 at every arithmetic use.
    /// A feature is a measurement, not an accumulator: f32 carries 7 significant digits,
    /// while the reductions that consume it (standardizer means/variances, the logistic
    /// gradient, the score) stay in f64 below. Halves the matrix, which is the single
    /// largest allocation of an experiment-wide rescore (33.6 GiB at 11.6M PSMs x 387
    /// features in f64; 16.8 GiB here).
    values: Vec<f32>,
    n_features: usize,
}

impl FeatureMatrix {
    /// Empty matrix of `n_features` columns, preallocated for `rows` rows.
    pub fn with_capacity(rows: usize, n_features: usize) -> FeatureMatrix {
        FeatureMatrix {
            values: Vec::with_capacity(rows.saturating_mul(n_features)),
            n_features,
        }
    }

    /// Append one value. Callers push exactly `n_features` values per row, in column order.
    /// The f64 the reader decoded is narrowed here, once.
    #[inline]
    pub fn push(&mut self, v: f64) {
        self.values.push(v as f32);
    }

    /// Append a whole row that a caller has already narrowed to f32.
    #[inline]
    pub fn push_row(&mut self, row: &[f32]) {
        self.values.extend_from_slice(row);
    }

    /// `(row, feature)` of the first non-finite value in flat row order, or `None`.
    ///
    /// Parallel because the serial form was `n_rows * n_features` `is_finite` checks on one
    /// thread over a matrix that is hundreds of gigabytes at experiment scale, for a
    /// predicate that is embarrassingly parallel. `find_first` is order-preserving: it
    /// returns the earliest matching row regardless of which worker found it, so the error
    /// a malformed table produces is the same row and the same message as before.
    pub fn find_non_finite(&self) -> Option<(usize, usize)> {
        if self.n_features == 0 {
            return None;
        }
        self.values
            .par_chunks_exact(self.n_features)
            .enumerate()
            .find_first(|(_, row)| row.iter().any(|v| !v.is_finite()))
            .map(|(row, values)| {
                let col = values
                    .iter()
                    .position(|v| !v.is_finite())
                    .expect("the row matched the predicate");
                (row, col)
            })
    }

    /// Fail loudly rather than silently mis-striding if a caller pushed a partial row.
    pub fn finish(self) -> anyhow::Result<FeatureMatrix> {
        if self.n_features > 0 && !self.values.len().is_multiple_of(self.n_features) {
            anyhow::bail!(
                "feature matrix has {} values, not a multiple of {} features",
                self.values.len(),
                self.n_features
            );
        }
        Ok(self)
    }

    /// Payload bytes of the value buffer, for the memory accounting.
    pub fn bytes(&self) -> usize {
        std::mem::size_of_val(self.values.as_slice())
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn rows(&self) -> usize {
        self.values.len().checked_div(self.n_features).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.rows() == 0
    }

    /// Row `i` as a contiguous slice of `n_features` values.
    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        let a = i * self.n_features;
        &self.values[a..a + self.n_features]
    }

    pub fn iter_rows(&self) -> impl Iterator<Item = &[f32]> {
        self.values.chunks_exact(self.n_features.max(1))
    }
}

fn fit_standardizer(x: &FeatureMatrix, idx: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let d = x.n_features();
    let n = idx.len().max(1) as f64;
    let mut mean = vec![0.0; d];
    for &i in idx {
        for (m, v) in mean.iter_mut().zip(x.row(i)) {
            *m += *v as f64;
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut std = vec![0.0; d];
    for &i in idx {
        for ((s, v), m) in std.iter_mut().zip(x.row(i)).zip(&mean) {
            let dd = *v as f64 - *m;
            *s += dd * dd;
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

/// Standardise one row into `out`. The subtraction and division are f64; the result is
/// kept f32 because `xtr` below holds one standardised copy of the training slice per
/// fold, and those copies are as large as the matrix itself.
#[inline]
fn std_row_into(row: &[f32], mean: &[f64], std: &[f64], out: &mut Vec<f32>) {
    out.extend((0..row.len()).map(|j| ((row[j] as f64 - mean[j]) / std[j]) as f32));
}

/// Score a row through the standardizer without materialising it.
///
/// The f32 narrowing is kept: `score_row(w, &std_row(..))` scored the standardised value
/// AFTER it had been rounded to f32, so computing the product from the f64 quotient
/// instead would change the last bits of every test-fold score.
#[inline]
fn score_std_row(w: &[f64], row: &[f32], mean: &[f64], std: &[f64]) -> f64 {
    let mut z = w[0];
    for j in 0..row.len() {
        let v = ((row[j] as f64 - mean[j]) / std[j]) as f32;
        z += w[j + 1] * v as f64;
    }
    z
}

/// Logistic regression by full-batch gradient descent with L2. Weight[0] = bias.
fn logreg_fit(rows: &[&[f32]], y: &[f64], l2: f64, epochs: usize, lr: f64) -> Vec<f64> {
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
                z += w[j + 1] * r[j] as f64;
            }
            let p = 1.0 / (1.0 + (-z).exp());
            let err = p - yi;
            grad[0] += err;
            for j in 0..d {
                grad[j + 1] += err * r[j] as f64;
            }
        }
        w[0] -= lr * grad[0] / n;
        for j in 0..d {
            w[j + 1] -= lr * (grad[j + 1] / n + l2 * w[j + 1]);
        }
    }
    w
}

fn score_row(w: &[f64], r: &[f32]) -> f64 {
    let mut z = w[0];
    for j in 0..r.len() {
        z += w[j + 1] * r[j] as f64;
    }
    z
}

pub struct RescoreInput<'a> {
    pub features: &'a FeatureMatrix,
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
    let n = inp.features.rows();
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
            // Standardized train matrix: ONE flat allocation of `train_rows * d`, row-major,
            // sliced with `chunks_exact(d)` below. It used to be a `Vec<Vec<f32>>`, which is
            // one heap block per training row -- the exact layout the comment on
            // `FeatureMatrix` records was removed from the matrix itself, surviving one
            // level down and multiplied by `folds`, since every fold is live at once inside
            // this parallel map. At experiment scale (11.6M PSMs, 3 folds) that is ~23M
            // live blocks here; now it is 3. Same values, same order, so the fit is
            // bit-identical.
            let d = inp.features.n_features();
            let mut xtr: Vec<f32> = Vec::with_capacity(train_idx.len().saturating_mul(d));
            for &i in &train_idx {
                std_row_into(inp.features.row(i), &mean, &std, &mut xtr);
            }
            let xrow = |k: usize| -> &[f32] { &xtr[k * d..(k + 1) * d] };
            let mut train_scores: Vec<f64> = train_idx.iter().map(|&i| inp.init_score[i]).collect();
            let mut w = vec![0.0; inp.features.n_features() + 1];
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
                let mut rows: Vec<&[f32]> = Vec::new();
                let mut ys: Vec<f64> = Vec::new();
                let mut n_pos = 0;
                for (k, &i) in train_idx.iter().enumerate() {
                    if inp.is_decoy[i] {
                        rows.push(xrow(k));
                        ys.push(0.0);
                    } else if q[k] <= inp.train_fdr {
                        rows.push(xrow(k));
                        ys.push(1.0);
                        n_pos += 1;
                    }
                }
                // fallback: if too few confident targets, take the top-scoring half
                if n_pos < 10 {
                    let mut order: Vec<usize> = (0..train_idx.len()).collect();
                    order.sort_by(|&a, &b| train_scores[b].total_cmp(&train_scores[a]));
                    let take = (train_idx.len() / 2).max(1);
                    rows.clear();
                    ys.clear();
                    for (rank, &k) in order.iter().enumerate() {
                        let i = train_idx[k];
                        if inp.is_decoy[i] {
                            rows.push(xrow(k));
                            ys.push(0.0);
                        } else if rank < take {
                            rows.push(xrow(k));
                            ys.push(1.0);
                        }
                    }
                }
                w = logreg_fit(&rows, &ys, 1e-3, 200, 0.5);
                train_scores = (0..train_idx.len())
                    .map(|k| score_row(&w, xrow(k)))
                    .collect();
            }
            // score the held-out test fold with this fold's scaler + weights
            test_idx
                .iter()
                .map(|&i| (i, score_std_row(&w, inp.features.row(i), &mean, &std)))
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

    /// The previous per-row standardiser: one `Vec<f32>` (one heap block) per row.
    fn std_row_ref(row: &[f32], mean: &[f64], std: &[f64]) -> Vec<f32> {
        (0..row.len())
            .map(|j| ((row[j] as f64 - mean[j]) / std[j]) as f32)
            .collect()
    }

    /// `percolator_lite` as it was before the flattening, transcribed: `xtr` as a
    /// `Vec<Vec<f32>>`, and the test fold scored through a materialised `std_row`. The
    /// claim the rewrite makes is bit-equality of the scores, so the old layout is kept
    /// here as the thing to compare against rather than asserting properties of the new
    /// one.
    fn percolator_lite_reference(inp: RescoreInput) -> Vec<f64> {
        let n = inp.features.rows();
        if n == 0 {
            return Vec::new();
        }
        let folds = inp.folds.max(1);
        let fold_of: Vec<usize> = inp.fold_key.iter().map(|c| (*c as usize) % folds).collect();
        let per_fold: Vec<Vec<(usize, f64)>> = (0..folds)
            .into_par_iter()
            .map(|test_fold| {
                let train_idx: Vec<usize> = (0..n).filter(|&i| fold_of[i] != test_fold).collect();
                let test_idx: Vec<usize> = (0..n).filter(|&i| fold_of[i] == test_fold).collect();
                if train_idx.is_empty() || test_idx.is_empty() {
                    return Vec::new();
                }
                let (mean, std) = fit_standardizer(inp.features, &train_idx);
                let xtr: Vec<Vec<f32>> = train_idx
                    .iter()
                    .map(|&i| std_row_ref(inp.features.row(i), &mean, &std))
                    .collect();
                let mut train_scores: Vec<f64> =
                    train_idx.iter().map(|&i| inp.init_score[i]).collect();
                let mut w = vec![0.0; inp.features.n_features() + 1];
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
                    let mut rows: Vec<&[f32]> = Vec::new();
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
                    if n_pos < 10 {
                        let mut order: Vec<usize> = (0..train_idx.len()).collect();
                        order.sort_by(|&a, &b| train_scores[b].total_cmp(&train_scores[a]));
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
                    train_scores = (0..train_idx.len())
                        .map(|k| score_row(&w, &xtr[k]))
                        .collect();
                }
                test_idx
                    .iter()
                    .map(|&i| {
                        (
                            i,
                            score_row(&w, &std_row_ref(inp.features.row(i), &mean, &std)),
                        )
                    })
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

    /// A synthetic population wide enough that the flattening matters and mixed enough
    /// that both the confident-target branch and the top-half fallback are exercised.
    fn crafted_population(n: usize, d: usize) -> (FeatureMatrix, Vec<bool>, Vec<u32>, Vec<f64>) {
        let mut features = FeatureMatrix::with_capacity(n, d);
        let (mut is_decoy, mut key, mut init) = (Vec::new(), Vec::new(), Vec::new());
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for i in 0..n {
            let decoy = i % 3 == 0;
            for j in 0..d {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let noise = (state >> 40) as f64 / 16_777_216.0;
                // Column 0 separates; the rest are noise of varying scale, including one
                // constant column (std guard) and one with a large offset.
                let v = match j {
                    0 => (if decoy { 0.0 } else { 2.5 }) + noise,
                    1 => 7.0,
                    2 => 1e6 * noise,
                    _ => noise - 0.5,
                };
                features.push(v);
            }
            is_decoy.push(decoy);
            key.push((i / 2) as u32);
            init.push(if decoy { -0.5 } else { 0.5 } + (i % 7) as f64 * 0.01);
        }
        (features.finish().unwrap(), is_decoy, key, init)
    }

    #[test]
    fn flat_training_matrix_scores_bit_identically_to_the_per_row_layout() {
        let (features, is_decoy, key, init) = crafted_population(600, 9);
        let mk = || RescoreInput {
            features: &features,
            is_decoy: &is_decoy,
            fold_key: &key,
            init_score: &init,
            folds: 3,
            num_iter: 4,
            train_fdr: 0.05,
        };
        let got = percolator_lite(mk());
        let want = percolator_lite_reference(mk());
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // And the fallback branch (too few confident targets to train on) as well.
        let (features, is_decoy, key, init) = crafted_population(40, 5);
        let mk = || RescoreInput {
            features: &features,
            is_decoy: &is_decoy,
            fold_key: &key,
            init_score: &init,
            folds: 2,
            num_iter: 3,
            train_fdr: 1e-9,
        };
        assert_eq!(
            percolator_lite(mk())
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            percolator_lite_reference(mk())
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn non_finite_scan_reports_the_first_row_in_flat_order() {
        let mut m = FeatureMatrix::with_capacity(4, 3);
        for v in [
            1.0,
            2.0,
            3.0, // row 0: clean
            4.0,
            5.0,
            6.0, // row 1: clean
            7.0,
            f64::NAN,
            f64::INFINITY, // row 2: feature 1 first
            f64::NEG_INFINITY,
            0.0,
            0.0, // row 3: also bad, must not win
        ] {
            m.push(v);
        }
        let m = m.finish().unwrap();
        assert_eq!(m.find_non_finite(), Some((2, 1)));
        // Same answer as the serial scan it replaces, on a clean matrix too.
        let serial = m
            .iter_rows()
            .enumerate()
            .find_map(|(r, row)| row.iter().position(|v| !v.is_finite()).map(|c| (r, c)));
        assert_eq!(m.find_non_finite(), serial);
        let mut clean = FeatureMatrix::with_capacity(2, 2);
        for v in [1.0, 2.0, 3.0, 4.0] {
            clean.push(v);
        }
        assert_eq!(clean.finish().unwrap().find_non_finite(), None);
        // An f64 that overflows f32 is non-finite once narrowed, and must be caught: the
        // scan reads the stored f32, not the value the reader decoded.
        let mut overflow = FeatureMatrix::with_capacity(1, 1);
        overflow.push(1e300);
        assert_eq!(overflow.finish().unwrap().find_non_finite(), Some((0, 0)));
    }

    #[test]
    fn separates_targets_from_decoys() {
        // targets have high feature[0], decoys low, plus noise
        let mut features = FeatureMatrix::with_capacity(200, 2);
        let mut is_decoy = Vec::new();
        let mut cid = Vec::new();
        let mut init = Vec::new();
        for i in 0..200 {
            let decoy = i % 2 == 0;
            let base = if decoy { 0.0 } else { 3.0 };
            let noise = ((i * 7 % 5) as f64) * 0.1;
            features.push(base + noise);
            features.push(noise);
            is_decoy.push(decoy);
            cid.push(i as u32);
            init.push(base + noise); // init score already discriminates
        }
        let features = features.finish().unwrap();
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
