//! Native semi-supervised rescorer (docs/11_compete_rescore_fdr.md, MVP
//! `native_tda`): a Percolator/Mokapot-style linear model. Standardize features,
//! then for each cross-validation fold train a logistic regression on a positive
//! set of confident targets versus all decoys, iterating the positive-set
//! selection. Deterministic (weights start at zero, no RNG). Port features, not
//! classifiers; the model is intentionally simple and swappable.

use crate::fdr::target_decoy_q_split;
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

fn fit_standardizer(x: &FeatureMatrix, idx: &[u32]) -> (Vec<f64>, Vec<f64>) {
    let d = x.n_features();
    let n = idx.len().max(1) as f64;
    let mut mean = vec![0.0; d];
    for &i in idx {
        for (m, v) in mean.iter_mut().zip(x.row(i as usize)) {
            *m += *v as f64;
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut std = vec![0.0; d];
    for &i in idx {
        for ((s, v), m) in std.iter_mut().zip(x.row(i as usize)).zip(&mean) {
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

/// Standardise one row into `out`, which must be exactly as wide as `row`. The
/// subtraction and division are f64; the result is kept f32 because `xtr` below holds one
/// standardised copy of the training slice, and that copy is as large as the matrix
/// itself.
///
/// Writing into a caller-provided slice rather than appending to a `Vec` is what lets the
/// whole standardised training matrix be filled by `par_chunks_mut`: rows are independent
/// (each value is a function of its own cell, the column mean and the column std), so the
/// parallel fill writes the same bits in the same places as the serial append did.
#[inline]
fn std_row_into(row: &[f32], mean: &[f64], std: &[f64], out: &mut [f32]) {
    for (j, o) in out.iter_mut().enumerate() {
        *o = ((row[j] as f64 - mean[j]) / std[j]) as f32;
    }
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

/// Columns per gradient band: 16 f32 = 64 B = exactly one cache line, so a band's pass
/// over the training slice uses every byte it fetches. Narrower bands would give more
/// parallel tasks but split a cache line across threads, so the same line would be
/// fetched once per band that touches it; wider bands read more than one line per row and
/// leave fewer tasks. At the shipped feature count (387) this is 25 bands.
const GRAD_BAND: usize = 16;

/// Logistic regression by full-batch gradient descent with L2. Weight[0] = bias.
///
/// Each epoch runs in two phases, and the split is what makes the epoch parallel without
/// moving a bit. The one-pass form it replaces was a single serial loop over training
/// rows, and it was serial all the way down: `z` is a left-fold into one f64 accumulator,
/// which rustc can neither reassociate into independent accumulators nor vectorise, so
/// every one of the `d` terms costs a full scalar add latency (~3 cycles on Zen 4) while
/// the independent multiplies hide behind it. At the pooled-experiment scale this kernel
/// runs at (~1.2M training rows x 387 features x `num_iter` 10 x `epochs` 200 = 2,000
/// full passes per fold) that chain alone is ~1.4e9 cycles per epoch, and the only source
/// of threads in the whole rescorer was `percolator_lite`'s parallel map over `folds`,
/// which defaults to 3. Three cores of a 64-core machine, for ~15 minutes per fold.
///
/// Phase 1 computes the per-row residual `err = sigmoid(z) - y`. Rows are independent, so
/// this parallelises over rows with no reordering at all: every row keeps its own strict
/// ascending-`j` fold for `z`, so `z`, `p` and `err` are bit-for-bit what they were.
///
/// Phase 2 folds the residuals into the gradient. `grad[0]` stays ONE serial left-fold
/// over the rows in ascending order, because that is the order it accumulated in before
/// and it is ~1 ms; it must not be parallelised. `grad[j+1]` is a separate left-fold per
/// column, and the columns are independent, so it parallelises over COLUMNS: each band
/// walks the rows in the same ascending order and adds the same terms to the same
/// accumulator in the same sequence. No sum is reordered anywhere in either phase.
///
/// The cost is one extra streaming pass over the training slice per epoch (phase 1 reads
/// it to produce `err`, phase 2 reads it again to fold), which is why the band width is a
/// cache line: the second pass then moves exactly its payload rather than 16x it.
///
/// Measured by `logreg_fit_epoch_cost` below on an i9-13900KS (8 P + 16 E cores,
/// dual-channel DDR5), release, 32 rayon threads, 387 features: 2.7-4.3x at 60,000 rows,
/// 3.2x at 300,000 rows (serial 104 ms/epoch, two-phase 32 ms/epoch), 4.4x at 16,000 rows
/// and 5.3x at 4,000. The gain FALLS as the slice grows because the two passes are then
/// DRAM streams: at 300,000 rows the two-phase epoch moves 2 x 464 MB in 32 ms, which is
/// ~29 GB/s and about all this desktop's two memory channels have. That is the honest
/// bound on the claim. Extrapolating the serial arm does corroborate the size of the
/// problem (104 ms/epoch at 300,000 x 387 scales to ~415 ms at a real ~1.2M-row fold, so
/// ~14 min for the 2,000 epochs of one fold), but a many-channel server should land far
/// above 3.2x and this fixture cannot show that. Halving the DRAM traffic by fusing the
/// two phases over L3-sized row blocks -- legal, since a per-column fold over consecutive
/// row blocks adds the same terms in the same order -- is the obvious next step and is
/// not done here.
fn logreg_fit(rows: &[&[f32]], y: &[f64], l2: f64, epochs: usize, lr: f64) -> Vec<f64> {
    let d = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut w = vec![0.0f64; d + 1];
    if rows.is_empty() {
        return w;
    }
    // `n` is the row count, as before; the `zip` below stopped at the shorter of the two
    // sequences, so the rows actually visited are the first `m`. Callers push `rows` and
    // `y` in lockstep, but keeping the truncation explicit means the two forms agree even
    // where they disagree with the caller.
    let n = rows.len() as f64;
    let m = rows.len().min(y.len());
    let rows = &rows[..m];
    let y = &y[..m];
    let mut errs = vec![0.0f64; m];
    let mut grad = vec![0.0f64; d];
    for _ in 0..epochs {
        // Phase 1: per-row residual, parallel over rows.
        let wv: &[f64] = &w;
        let (w0, wf) = (wv[0], &wv[1..]);
        rows.par_iter()
            .zip(y.par_iter())
            .zip(errs.par_iter_mut())
            .for_each(|((r, &yi), e)| {
                let r = &r[..d];
                let mut z = w0;
                for (wj, rj) in wf.iter().zip(r) {
                    z += *wj * *rj as f64;
                }
                let p = 1.0 / (1.0 + (-z).exp());
                *e = p - yi;
            });
        // Phase 2a: the bias gradient, serial and in row order. Do not parallelise.
        let mut grad0 = 0.0f64;
        for &e in &errs {
            grad0 += e;
        }
        // Phase 2b: one independent left-fold per feature column, parallel over bands.
        // The accumulators are a LOCAL array written back once, not the `grad` slice
        // itself: a band is 16 f64 = 128 B, so neighbouring bands share the cache lines at
        // their boundaries, and accumulating straight into `grad` made every one of the
        // `rows * 16` adds an invalidation of another thread's line. Measured on the
        // 60,000 x 387 fixture below, that false sharing alone was the difference between
        // 3.4x slower than the serial fit and faster than it.
        grad.par_chunks_mut(GRAD_BAND)
            .enumerate()
            .for_each(|(b, out)| {
                let (j0, j1) = (b * GRAD_BAND, b * GRAD_BAND + out.len());
                let mut local = [0.0f64; GRAD_BAND];
                let acc = &mut local[..out.len()];
                for (r, &e) in rows.iter().zip(&errs) {
                    for (a, v) in acc.iter_mut().zip(&r[j0..j1]) {
                        *a += e * *v as f64;
                    }
                }
                out.copy_from_slice(acc);
            });
        w[0] -= lr * grad0 / n;
        for j in 0..d {
            w[j + 1] -= lr * (grad[j] / n + l2 * w[j + 1]);
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
    // Every per-PSM bookkeeping vector below is u32-indexed; `fdr.rs` already asserts the
    // same bound for the q kernels this calls into.
    assert!(
        n <= u32::MAX as usize,
        "percolator_lite: {n} rows exceeds the u32 row index"
    );
    let folds = inp.folds.max(1);
    let d = inp.features.n_features();

    // Fold assignment by fold_key (base peptide): all charge/mod variants of a
    // peptide share a fold, so none leaks between train and test. The value is always
    // below `folds` and never above the u32 key it came from, so u32 stores it exactly and
    // the vector is 4 bytes per PSM rather than 8.
    let fold_of: Vec<u32> = inp
        .fold_key
        .iter()
        .map(|c| ((*c as usize) % folds) as u32)
        .collect();

    // Folds are independent: each fits its own scaler + weights on its training rows and
    // scores only its own (disjoint) test set. Standardization is fit on the TRAIN fold
    // only (no test leakage), which is why each fold owns its scaler.
    //
    // They run ONE AT A TIME. A parallel map over folds used to be the only source of
    // threads in this rescorer, and the price was that every fold's standardised training
    // copy was live at once: with the default 3 folds the peak was the matrix plus three
    // 2/3-sized copies of it, roughly (1 + folds) x the matrix (4.85 + 3 x 3.23 = 14.6 GB
    // on a six-run Astral pool of 3,133,636 PSMs x 387 features; 53.8 GB at 11.6M PSMs).
    // Sequentially it is the matrix plus one copy: 8.1 GB and 29.9 GB, a saving of 6.5 and
    // 23.9 GB. `train_idx`/`test_idx`, `mean`/`std`, `train_scores` and the `rows` pointer
    // vector become single-copy at the same time. What makes this affordable is that
    // `logreg_fit` and the per-row maps below are parallel in their own right, so one fold
    // already saturates the machine; before that change this would have been 3x the wall.
    // What it does cost is the serial remainder (the tied-block walk in `target_decoy_q`,
    // the positive-set build, the bias fold), which no longer overlaps across folds: a few
    // seconds per fold against a fit measured in minutes.
    let mut final_score = inp.init_score.to_vec();
    for test_fold in 0..folds {
        let mut train_idx: Vec<u32> = Vec::new();
        let mut test_idx: Vec<u32> = Vec::new();
        for (i, f) in fold_of.iter().enumerate() {
            if *f as usize == test_fold {
                test_idx.push(i as u32);
            } else {
                train_idx.push(i as u32);
            }
        }
        if train_idx.is_empty() || test_idx.is_empty() {
            continue;
        }
        let (mean, std) = fit_standardizer(inp.features, &train_idx);
        // Standardized train matrix: ONE flat allocation of `train_rows * d`, row-major.
        // It used to be a `Vec<Vec<f32>>`, which is one heap block per training row -- the
        // exact layout the comment on `FeatureMatrix` records was removed from the matrix
        // itself, surviving one level down. At experiment scale (11.6M PSMs) that was
        // ~7.7M live blocks per fold; now it is one. Same values, same order, so the fit
        // is bit-identical.
        let mut xtr: Vec<f32> = vec![0.0; train_idx.len().saturating_mul(d)];
        xtr.par_chunks_mut(d.max(1))
            .zip(train_idx.par_iter())
            .for_each(|(out, &i)| std_row_into(inp.features.row(i as usize), &mean, &std, out));
        let xrow = |k: usize| -> &[f32] { &xtr[k * d..(k + 1) * d] };
        let mut train_scores: Vec<f64> = train_idx
            .iter()
            .map(|&i| inp.init_score[i as usize])
            .collect();
        // The training fold's labels, gathered once. The q kernel used to be handed a
        // freshly zipped `Vec<(f64, bool)>` of the scores and these labels on every
        // iteration: a 16-byte-per-row buffer, allocated outside the loop and so resident
        // for the whole fit, whose only job was to pair two columns the fold already has.
        let train_decoy: Vec<bool> = train_idx
            .iter()
            .map(|&i| inp.is_decoy[i as usize])
            .collect();
        let mut w = vec![0.0; d + 1];
        for _ in 0..inp.num_iter.max(1) {
            let q = target_decoy_q_split(&train_scores, &train_decoy);
            // positive set: confident targets; negatives: all decoys
            let mut rows: Vec<&[f32]> = Vec::new();
            let mut ys: Vec<f64> = Vec::new();
            let mut n_pos = 0;
            for (k, &decoy) in train_decoy.iter().enumerate() {
                if decoy {
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
                let mut order: Vec<u32> = (0..train_idx.len() as u32).collect();
                order.sort_by(|&a, &b| {
                    train_scores[b as usize].total_cmp(&train_scores[a as usize])
                });
                let take = (train_idx.len() / 2).max(1);
                rows.clear();
                ys.clear();
                for (rank, &k) in order.iter().enumerate() {
                    if train_decoy[k as usize] {
                        rows.push(xrow(k as usize));
                        ys.push(0.0);
                    } else if rank < take {
                        rows.push(xrow(k as usize));
                        ys.push(1.0);
                    }
                }
            }
            w = logreg_fit(&rows, &ys, 1e-3, 200, 0.5);
            // One independent score per training row; the parallel map is an indexed
            // rayon collect, so the output order and every value are unchanged.
            train_scores = (0..train_idx.len())
                .into_par_iter()
                .map(|k| score_row(&w, xrow(k)))
                .collect();
        }
        // score the held-out test fold with this fold's scaler + weights
        let scored: Vec<f64> = test_idx
            .par_iter()
            .map(|&i| score_std_row(&w, inp.features.row(i as usize), &mean, &std))
            .collect();
        // A scatter onto this fold's own (disjoint) test rows, so doing it here rather
        // than after every fold has finished cannot move a value.
        for (&i, s) in test_idx.iter().zip(scored) {
            final_score[i as usize] = s;
        }
    }
    final_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdr::target_decoy_q;

    /// The previous standardiser, transcribed: `usize` row indices, which is what the
    /// per-fold bookkeeping used before it was narrowed to `u32`.
    fn fit_standardizer_reference(x: &FeatureMatrix, idx: &[usize]) -> (Vec<f64>, Vec<f64>) {
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

    /// The previous per-row standardiser: one `Vec<f32>` (one heap block) per row.
    fn std_row_ref(row: &[f32], mean: &[f64], std: &[f64]) -> Vec<f32> {
        (0..row.len())
            .map(|j| ((row[j] as f64 - mean[j]) / std[j]) as f32)
            .collect()
    }

    /// The previous `logreg_fit`, transcribed unchanged: one serial pass over the training
    /// rows per epoch, accumulating the residual and the whole gradient row by row. Kept
    /// as the thing the two-phase kernel is checked against, because the claim is equality
    /// of the fitted weights, not that the new arrangement is self-consistent.
    fn logreg_fit_reference(
        rows: &[&[f32]],
        y: &[f64],
        l2: f64,
        epochs: usize,
        lr: f64,
    ) -> Vec<f64> {
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

    /// A deterministic training slice: `rows` standardised-looking f32 values of `d`
    /// columns, alternating labels, values spread over several magnitudes so the f32
    /// narrowing and the f64 accumulation both have something to lose.
    fn training_slice(rows: usize, d: usize) -> (Vec<f32>, Vec<f64>) {
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut x = vec![0.0f32; rows * d];
        for (k, v) in x.iter_mut().enumerate() {
            let r = next();
            let mag = match k % 5 {
                0 => 1e-7,
                1 => 1.0,
                2 => 1e3,
                3 => 0.0, // exact zeros, so a band can be all-zero
                _ => 3.5,
            };
            *v = (((r >> 11) as f64 / 9_007_199_254_740_992.0) - 0.5) as f32 * mag as f32;
        }
        let y: Vec<f64> = (0..rows)
            .map(|i| if i % 3 == 0 { 0.0 } else { 1.0 })
            .collect();
        (x, y)
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
                let (mean, std) = fit_standardizer_reference(inp.features, &train_idx);
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
                    w = logreg_fit_reference(&rows, &ys, 1e-3, 200, 0.5);
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
    fn two_phase_gradient_fits_bit_identical_weights() {
        // Widths that straddle the 16-column band: under one band, exactly one, one over,
        // a whole number of bands, a ragged tail, and the shipped feature count.
        for &d in &[1usize, 5, 15, 16, 17, 32, 33, 387] {
            for &rows in &[1usize, 2, 7, 64, 301] {
                let (x, y) = training_slice(rows, d);
                let r: Vec<&[f32]> = x.chunks_exact(d).collect();
                let got = logreg_fit(&r, &y, 1e-3, 7, 0.5);
                let want = logreg_fit_reference(&r, &y, 1e-3, 7, 0.5);
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "d = {d}, rows = {rows}"
                );
            }
        }
        // Degenerate shapes the caller can still produce: no rows at all (an empty
        // positive set), a `y` shorter than `rows` (the `zip` truncation the one-pass form
        // had), and zero epochs.
        let (x, y) = training_slice(8, 4);
        let r: Vec<&[f32]> = x.chunks_exact(4).collect();
        let empty: [&[f32]; 0] = [];
        assert_eq!(
            logreg_fit(&empty, &y, 1e-3, 3, 0.5),
            logreg_fit_reference(&empty, &y, 1e-3, 3, 0.5)
        );
        assert_eq!(
            logreg_fit(&r, &y[..3], 1e-3, 5, 0.5)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            logreg_fit_reference(&r, &y[..3], 1e-3, 5, 0.5)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(logreg_fit(&r, &y, 1e-3, 0, 0.5), vec![0.0f64; 5]);
    }

    /// Microbenchmark, not an assertion about any machine: the one-pass serial fit against
    /// the two-phase one on a slice shaped like a training fold (387 features), small
    /// enough to run in a test. Both arms are handed the same prebuilt row pointers and
    /// each allocates its own per-epoch working set inside the timer, so neither is
    /// credited with work the other pays for. Run with
    /// `cargo test -p mumdia --release -- --ignored --nocapture logreg_fit_epoch_cost`.
    #[test]
    #[ignore = "microbenchmark; meaningful only in release"]
    fn logreg_fit_epoch_cost() {
        // Defaults are small enough to run anywhere; `MUMDIA_BENCH_ROWS`, `_D` and
        // `_EPOCHS` scale it towards a real training fold (~1.2M rows, 387 features) on a
        // machine with the memory for it.
        let env = |k: &str, dflt: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(dflt)
        };
        let (d, rows, epochs) = (
            env("MUMDIA_BENCH_D", 387),
            env("MUMDIA_BENCH_ROWS", 60_000),
            env("MUMDIA_BENCH_EPOCHS", 5),
        );
        let (x, y) = training_slice(rows, d);
        let r: Vec<&[f32]> = x.chunks_exact(d).collect();
        // One epoch each, untimed, so neither arm pays the first touch of `x`.
        assert_eq!(
            logreg_fit_reference(&r, &y, 1e-3, 1, 0.5),
            logreg_fit(&r, &y, 1e-3, 1, 0.5)
        );
        let t0 = std::time::Instant::now();
        let a = logreg_fit_reference(&r, &y, 1e-3, epochs, 0.5);
        let serial = t0.elapsed();
        let t1 = std::time::Instant::now();
        let b = logreg_fit(&r, &y, 1e-3, epochs, 0.5);
        let two_phase = t1.elapsed();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the benchmark arms must agree bit for bit"
        );
        println!(
            "logreg_fit {rows} rows x {d} features x {epochs} epochs, {} rayon threads:              serial {serial:?}, two-phase {two_phase:?} ({:.2}x)",
            rayon::current_num_threads(),
            serial.as_secs_f64() / two_phase.as_secs_f64()
        );
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
