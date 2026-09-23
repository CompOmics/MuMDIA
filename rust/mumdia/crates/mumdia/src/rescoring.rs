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

/// Columns per gradient band: 16 f32 = 64 B, the size of one cache line. That BOUNDS the
/// over-fetch of phase 2b at 2x; it does not remove it, and the earlier claim here that "a
/// band uses every byte it fetches" holds only where the row stride is a multiple of 64,
/// which at the shipped feature count it is not. The stride is 387 * 4 = 1548 B and
/// 1548 mod 64 = 12, so successive row starts walk the residues of gcd(1548, 64) = 4 and
/// land on a line boundary once every 16 rows; the other 15 straddle two lines. Expected
/// lines per band-row is 1/16 + 15/16 * 2 = 1.94, so a band moves about 1.94x its payload.
///
/// What the width actually trades is that over-fetch, roughly 1 + 16 / GRAD_BAND, against
/// the task count, ceil(d / GRAD_BAND). At d = 387: 8 gives 49 tasks at ~3x over-fetch, 16
/// gives 25 at ~2x, 32 gives 13 at ~1.5x, 64 gives 7 at ~1.25x -- and 25 tasks is already
/// fewer than this machine's 32 threads, so neither end of that trade is free.
///
/// Swept rather than argued (`logreg_fit_epoch_cost` below, 300,000 x 387 x 5 epochs, best
/// of 3 interleaved repeats, i9-13900KS, 32 rayon threads; the serial control arm read
/// 0.500-0.517 s in all four builds, so the four are comparable): two-phase 0.144 s at
/// GRAD_BAND 8, 0.101 s at 16, 0.123 s at 32, 0.160 s at 64. 16 is the measured optimum, so
/// the constant stands even though the reasoning that first justified it did not. The band
/// width cannot change any result: each column is a left-fold over rows in ascending order
/// whatever band it lands in.
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
/// The cost is more than the "one extra streaming pass over the training slice" an earlier
/// version of this comment claimed. Phase 1 reads the slice to produce `err` and phase 2b
/// reads it again to fold, which is the two passes; but phase 2b also re-reads the whole
/// `rows` pointer vector and the whole `errs` column ONCE PER BAND, so both are read
/// `ceil(d / GRAD_BAND)` = 25 times per epoch at the shipped feature count. At a real
/// 2.09M-row fold (the six-run Astral pool at 3 folds) that is 25 x 33.4 MB of fat
/// pointers plus 25 x 16.7 MB of residuals, 1.25 GB, on top of two 3.23 GB matrix passes:
/// 7.7 GB per epoch against the serial form's 3.28 GB, so 2.4x the DRAM traffic and not
/// the 2x this comment used to say. It is still a win because the serial form cannot use
/// more than one core of that bandwidth, but it is the reason the measured speedup is well
/// under the core count and falls as the slice grows.
///
/// Two ways out, neither taken here: fusing the two phases over L3-sized row blocks halves
/// the matrix traffic (legal, since a per-column fold over consecutive row blocks adds the
/// same terms in the same order), and indexing a flat training matrix instead of taking
/// `&[&[f32]]` would remove the pointer vector entirely -- `percolator_lite` already has
/// `xtr` flat and builds the pointers only to express the positive-set subset.
///
/// Measured by `logreg_fit_epoch_cost` below on an i9-13900KS (8 P + 16 E cores,
/// dual-channel DDR5), release, 32 rayon threads, 387 features, best of 3 interleaved
/// repeats: 6.1x at 4,000 rows, 8.5x at 16,000, 5.1x at 60,000, 5.1x at 300,000 (serial
/// 103 ms/epoch, two-phase 20 ms/epoch) and 4.9x at 1,000,000. An earlier single-shot run
/// of this fixture recorded 5.3x / 4.4x / 2.7-4.3x / 3.2x at the first four sizes; the
/// serial arm reproduces to 1% (104 against 103 ms/epoch), so it was the two-phase arm
/// that was mistimed, which is why every arm is now repeated and interleaved. The
/// 1,000,000-row point is the least trustworthy of the five (spread 1.8x across repeats
/// against 1.02-1.10x at the smaller sizes).
///
/// The gain does fall as the slice grows, and the ceiling is bandwidth: at 300,000 rows the
/// two-phase epoch moves 2 x 464 MB of matrix plus 25 x 7.2 MB of pointers and residuals in
/// 20 ms, 55 GB/s, and at 1,000,000 rows 3.70 GB in 69 ms, 53 GB/s -- the same ceiling
/// twice, which is about all this desktop's two channels have. Extrapolating the serial arm
/// corroborates the size of the problem: 103 ms/epoch at 300,000 x 387 scales to ~415 ms at
/// a real ~1.2M-row fold, so ~14 min for the 2,000 epochs of one fold. A many-channel
/// server should land above these ratios and this fixture cannot show that.
fn logreg_fit(rows: &[&[f32]], y: &[f64], l2: f64, epochs: usize, lr: f64) -> Vec<f64> {
    let d = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut w = vec![0.0f64; d + 1];
    if rows.is_empty() {
        return w;
    }
    // The one-pass form `zip`ped `rows` with `y` and so silently stopped at the shorter of
    // the two, while still dividing the gradient by `rows.len()`: a short `y` produced a
    // fitted model rather than an error. Every caller pushes the two in lockstep, so the
    // condition is a caller bug and not a shape to support; assert it rather than
    // entrenching the truncation. (Checked after the empty-`rows` early return above, which
    // both forms take before they look at `y` at all.)
    assert_eq!(
        rows.len(),
        y.len(),
        "logreg_fit: {} rows against {} labels",
        rows.len(),
        y.len()
    );
    let n = rows.len() as f64;
    let mut errs = vec![0.0f64; rows.len()];
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
    // The four per-PSM columns must all be `n` long. This used to be enforced by accident:
    // the fold loop was `(0..n).filter(|&i| fold_of[i] != test_fold)`, which panicked on a
    // short `fold_key`. Driving the loop off `fold_of` instead would have turned that into
    // something worse than a crash -- the trailing PSMs would be assigned to no fold, never
    // scored, and silently returned at their unrescored `init_score`, from where they enter
    // the q population as if they had been rescored. `is_decoy` and `init_score` are checked
    // for the same reason: both are indexed by matrix row below, and `final_score` is sized
    // from `init_score`. No caller is short today (`native_scores` builds all four from the
    // same table), but a sub-batched caller easily could be.
    assert_eq!(
        inp.fold_key.len(),
        n,
        "percolator_lite: {} fold keys against {n} matrix rows",
        inp.fold_key.len()
    );
    assert_eq!(
        inp.is_decoy.len(),
        n,
        "percolator_lite: {} decoy labels against {n} matrix rows",
        inp.is_decoy.len()
    );
    assert_eq!(
        inp.init_score.len(),
        n,
        "percolator_lite: {} initial scores against {n} matrix rows",
        inp.init_score.len()
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
    // copy was live at once. A training copy is `(folds - 1) / folds` of the matrix, so
    // `folds` of them is `folds - 1` matrices and the old peak was the matrix plus that:
    // `folds` x the matrix exactly, not the "(1 + folds) x" several places still say. With
    // the default 3 folds, 4.85 + 3 x 3.23 = 14.55 GB on a six-run Astral pool of 3,133,636
    // PSMs x 387 features (3.00x), and 53.8 GB at 11.6M PSMs.
    //
    // Sequentially it is the matrix plus ONE copy, `1 + (folds - 1) / folds` x the matrix:
    // 1.50x at 2 folds, 1.67x at 3, 1.80x at 5, approaching 2x and never above it. That is
    // 8.08 GB and 29.9 GB on the two pools above, a saving of 6.47 and 23.9 GB, and the
    // point is that the peak no longer grows with `folds` at all. Independently verified at
    // 300,000 x 387: folds=3 1,385 -> 767 MB, folds=5 2,308 -> 829 MB.
    // `train_idx`/`test_idx`, `mean`/`std`, `train_scores` and the `rows` pointer vector
    // become single-copy at the same time. What makes this affordable is that `logreg_fit`
    // and the per-row maps below are parallel in their own right, so one fold already
    // saturates the machine; before that change this would have been 3x the wall.
    //
    // The wall-clock trade is conditional in principle and the memory saving is not: this
    // arrangement is faster only where the per-fold speedup beats `folds`, because the old
    // one got exactly `folds`-way parallelism for free. In practice, on this fixture, it
    // wins everywhere measured. `percolator_lite_fold_cost` below, against a transcription
    // of the PARENT commit's production code, i9-13900KS, 32 rayon threads, 387 features,
    // 200,000 PSMs (training slices 206-310 MB, nothing cache-resident), best of 3
    // interleaved repeats:
    //
    //     folds   num_iter 2   num_iter 10 (the shipped `rescore.num_iter`)
    //     3       1.91x        2.03x
    //     5       1.31x        1.30x
    //
    // and on the same fixture at 30,000 PSMs, num_iter 10: 6.16x / 3.82x / 2.17x at 2 / 3 /
    // 5 folds (at num_iter 2: 5.82x / 3.46x / 1.89x). The implied per-fold speedup is
    // consistent across the two fold counts -- 107.4 s of parent against 3 x 17.6 s here at
    // 3 folds, 135.8 against 5 x 20.9 at 5, so about 6.1x and 6.5x -- and it beats both.
    //
    // Three earlier numbers are WITHDRAWN. "1.51x at 3 folds, 0.91x at 5" and "0.24x at 5
    // folds" on the 30,000-PSM fixture were single shots against a reference arm that was
    // NOT the parent: it reintroduced a `Vec<Vec<f32>>` the parent had already removed. That
    // arm is measurable here (`MUMDIA_BENCH_BASELINE=per_row`) and it is 1.03x the parent's
    // wall at 3 folds but 1.56x at 5 (211.6 s against 135.8), so it inflated exactly the
    // configuration where the change is weakest. It also perturbs the arm it is compared
    // against: at 5 folds it keeps ~800,000 live heap blocks, and the interleaved sequential
    // arm then varied 124-212 s against 104-105 s when the baseline is the parent. And
    // `MUMDIA_BENCH_ITERS` defaulted to 2 against a shipped `num_iter` of 10. The
    // L3-thrashing story told to explain the 0.24x was explaining a noise artifact:
    // repeated and interleaved, that exact configuration reads 1.89x. `num_iter` does move
    // the answer in the predicted direction, because the per-fold serial remainder is paid
    // once per fold whatever `num_iter` is and so costs relatively more when the fit is
    // short, but by 6% at 3 folds and by nothing at 5 -- not by a change of sign.
    //
    // This remains a microbenchmark of one kernel on one desktop: nothing here measures
    // `mumdia rescore` on a real competed table. A machine with more memory channels should
    // sit above these ratios, since the fitter is bandwidth-bound (see `logreg_fit`).
    //
    // What this also costs is the serial remainder (the tied-block walk in
    // `target_decoy_q`, the positive-set build, the bias fold, `fit_standardizer`), which
    // no longer overlaps across folds: a few seconds per fold against a fit measured in
    // minutes.
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

    /// `percolator_lite` two layouts back, before the training slice was flattened:
    /// `xtr` as a `Vec<Vec<f32>>`, and the test fold scored through a materialised
    /// `std_row`. Kept as the OLDEST form the current scores are still pinned against, so
    /// the equality claim reaches back past the parent as well.
    ///
    /// This is NOT the arm to time against. It is one heap block per training row, which
    /// the parent had already removed; see `percolator_lite_parent` for the baseline the
    /// benchmark uses.
    fn percolator_lite_per_row_reference(inp: RescoreInput) -> Vec<f64> {
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

    /// The parent's `std_row_into`: appends into a `Vec<f32>` rather than writing into a
    /// caller-provided slice, which is what made its `xtr` fill serial.
    fn std_row_into_parent(row: &[f32], mean: &[f64], std: &[f64], out: &mut Vec<f32>) {
        out.extend((0..row.len()).map(|j| ((row[j] as f64 - mean[j]) / std[j]) as f32));
    }

    /// The PARENT commit's production `percolator_lite`, transcribed line for line: the
    /// parallel map over folds, `usize` bookkeeping, a FLAT `xtr` filled serially by
    /// `std_row_into_parent`, the `Vec<(f64, bool)>` staging buffer allocated outside the
    /// `num_iter` loop and refilled inside it, the serial one-pass `logreg_fit_reference`,
    /// a serial `train_scores` rebuild, and the test fold scored by the allocation-free
    /// `score_std_row`.
    ///
    /// This is the baseline `percolator_lite_fold_cost` times, and getting it right is the
    /// whole point of the function. An earlier version of that benchmark timed the per-row
    /// reference above instead, which reintroduces a `Vec<Vec<f32>>` the parent had already
    /// removed. Measured at 200,000 x 387 x 10 iters, that arm costs 1.03x the parent's wall
    /// at 3 folds and 1.56x at 5, so it flattered the new arrangement most in the
    /// configuration where the new arrangement is weakest. Nothing here may be
    /// "modernised": if a line looks avoidable, that is the measurement it exists to make.
    fn percolator_lite_parent(inp: RescoreInput) -> Vec<f64> {
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
                let d = inp.features.n_features();
                let mut xtr: Vec<f32> = Vec::with_capacity(train_idx.len().saturating_mul(d));
                for &i in &train_idx {
                    std_row_into_parent(inp.features.row(i), &mean, &std, &mut xtr);
                }
                let xrow = |k: usize| -> &[f32] { &xtr[k * d..(k + 1) * d] };
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
                            rows.push(xrow(k));
                            ys.push(0.0);
                        } else if q[k] <= inp.train_fdr {
                            rows.push(xrow(k));
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
                                rows.push(xrow(k));
                                ys.push(0.0);
                            } else if rank < take {
                                rows.push(xrow(k));
                                ys.push(1.0);
                            }
                        }
                    }
                    w = logreg_fit_reference(&rows, &ys, 1e-3, 200, 0.5);
                    train_scores = (0..train_idx.len())
                        .map(|k| score_row(&w, xrow(k)))
                        .collect();
                }
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
    fn sequential_folds_score_bit_identically_to_both_older_layouts() {
        // Against the PARENT (the code actually replaced) and against the per-row layout
        // two steps back, because each pins a different claim: the parent pins this
        // series, the per-row one pins that the flattening before it still holds.
        let bits = |v: Vec<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
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
        let got = bits(percolator_lite(mk()));
        assert_eq!(
            got,
            bits(percolator_lite_parent(mk())),
            "against the parent"
        );
        assert_eq!(
            got,
            bits(percolator_lite_per_row_reference(mk())),
            "against the per-row layout"
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
        let got = bits(percolator_lite(mk()));
        assert_eq!(
            got,
            bits(percolator_lite_parent(mk())),
            "against the parent"
        );
        assert_eq!(
            got,
            bits(percolator_lite_per_row_reference(mk())),
            "against the per-row layout"
        );
    }

    #[test]
    #[should_panic(expected = "percolator_lite: 40 fold keys against 60 matrix rows")]
    fn a_short_fold_key_is_rejected_rather_than_silently_leaving_psms_unrescored() {
        // The parent panicked here (`(0..n).filter(|&i| fold_of[i] != test_fold)`). Driving
        // the fold loop off `fold_of` instead would return the trailing 20 PSMs at their
        // unrescored `init_score`, and nothing downstream can tell those from real scores.
        let (features, is_decoy, key, init) = crafted_population(60, 4);
        percolator_lite(RescoreInput {
            features: &features,
            is_decoy: &is_decoy,
            fold_key: &key[..40],
            init_score: &init,
            folds: 3,
            num_iter: 1,
            train_fdr: 0.05,
        });
    }

    #[test]
    #[should_panic(expected = "percolator_lite: 40 initial scores against 60 matrix rows")]
    fn a_short_init_score_is_rejected() {
        let (features, is_decoy, key, init) = crafted_population(60, 4);
        percolator_lite(RescoreInput {
            features: &features,
            is_decoy: &is_decoy,
            fold_key: &key,
            init_score: &init[..40],
            folds: 3,
            num_iter: 1,
            train_fdr: 0.05,
        });
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
        // positive set, which both forms return from before they look at `y`), and zero
        // epochs.
        let (x, y) = training_slice(8, 4);
        let r: Vec<&[f32]> = x.chunks_exact(4).collect();
        let empty: [&[f32]; 0] = [];
        assert_eq!(
            logreg_fit(&empty, &y, 1e-3, 3, 0.5),
            logreg_fit_reference(&empty, &y, 1e-3, 3, 0.5)
        );
        assert_eq!(logreg_fit(&r, &y, 1e-3, 0, 0.5), vec![0.0f64; 5]);
    }

    #[test]
    #[should_panic(expected = "logreg_fit: 8 rows against 3 labels")]
    fn mismatched_rows_and_labels_are_rejected_rather_than_truncated() {
        // The one-pass form `zip`ped and so fitted on the first 3 rows while dividing the
        // gradient by 8. Callers push the two in lockstep; a mismatch is a caller bug.
        let (x, y) = training_slice(8, 4);
        let r: Vec<&[f32]> = x.chunks_exact(4).collect();
        logreg_fit(&r, &y[..3], 1e-3, 5, 0.5);
    }

    /// Microbenchmark, not an assertion about any machine: the one-pass serial fit against
    /// the two-phase one on a slice shaped like a training fold (387 features), small
    /// enough to run in a test. Both arms are handed the same prebuilt row pointers and
    /// each allocates its own per-epoch working set inside the timer, so neither is
    /// credited with work the other pays for. Repeated `MUMDIA_BENCH_REPS` times with the
    /// arms interleaved and the order alternating, summarised on the minimum with the
    /// spread printed, for the same reason as `percolator_lite_fold_cost` below. Run with
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
        let (d, rows, epochs, reps) = (
            env("MUMDIA_BENCH_D", 387),
            env("MUMDIA_BENCH_ROWS", 60_000),
            env("MUMDIA_BENCH_EPOCHS", 5),
            env("MUMDIA_BENCH_REPS", 3).max(1),
        );
        let (x, y) = training_slice(rows, d);
        let r: Vec<&[f32]> = x.chunks_exact(d).collect();
        // One epoch each, untimed, so neither arm pays the first touch of `x`.
        assert_eq!(
            logreg_fit_reference(&r, &y, 1e-3, 1, 0.5),
            logreg_fit(&r, &y, 1e-3, 1, 0.5)
        );
        let bits = |v: Vec<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let time = |f: &dyn Fn() -> Vec<f64>| {
            let t = std::time::Instant::now();
            let v = f();
            (t.elapsed().as_secs_f64(), bits(v))
        };
        let old = || logreg_fit_reference(&r, &y, 1e-3, epochs, 0.5);
        let new = || logreg_fit(&r, &y, 1e-3, epochs, 0.5);
        let (mut olds, mut news) = (Vec::new(), Vec::new());
        for rep in 0..reps {
            let (a, b) = if rep % 2 == 0 {
                let a = time(&old);
                let b = time(&new);
                (a, b)
            } else {
                let b = time(&new);
                let a = time(&old);
                (a, b)
            };
            assert_eq!(a.1, b.1, "the benchmark arms must agree bit for bit");
            println!(
                "  rep {rep}: serial {:.3} s, two-phase {:.3} s ({:.2}x)",
                a.0,
                b.0,
                a.0 / b.0
            );
            olds.push(a.0);
            news.push(b.0);
        }
        let lo = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "logreg_fit {rows} rows x {d} features x {epochs} epochs, {} rayon threads, \
             {reps} reps: serial {:.3} s (spread {:.2}x), two-phase {:.3} s \
             (spread {:.2}x), best-of ratio {:.2}x",
            rayon::current_num_threads(),
            lo(&olds),
            hi(&olds) / lo(&olds),
            lo(&news),
            hi(&news) / lo(&news),
            lo(&olds) / lo(&news)
        );
    }

    /// Microbenchmark of the two commits TOGETHER, which is the only way to judge either:
    /// the old arm gets its threads from the parallel map over folds and fits each fold
    /// serially, the new one fits one fold at a time with a parallel fitter. Whether that
    /// is a win on wall clock depends entirely on whether the per-fold speedup beats
    /// `folds`; the memory saving (the matrix plus ONE standardised training copy instead
    /// of `folds` of them) does not depend on it. Both arms are handed the same
    /// `FeatureMatrix` and each allocates its own standardised copies, index vectors and
    /// per-epoch working set inside the timer.
    ///
    /// Three things about the harness, each of which was wrong in the first version of it
    /// and each of which changed the answer:
    ///
    /// - the baseline is `percolator_lite_parent`, the parent commit's production code. It
    ///   was `percolator_lite_per_row_reference`, which reintroduces a `Vec<Vec<f32>>` the
    ///   parent had already removed and is 1.4-1.7x slower than what is actually being
    ///   replaced;
    /// - `MUMDIA_BENCH_ITERS` defaults to the SHIPPED `rescore.num_iter` of 10, not to 2.
    ///   The per-fold serial remainder is paid once per fold whatever `num_iter` is, so
    ///   serialising folds hurts most when `num_iter` is small, and the sign of the answer
    ///   changes between 2 and 10;
    /// - every arm is repeated `MUMDIA_BENCH_REPS` times with the two arms interleaved and
    ///   the order alternating, and the summary is the MINIMUM over repeats with the full
    ///   spread printed. On this hybrid desktop two consecutive identical runs have
    ///   differed by 2.2x and arm order alone moved one arm by 1.7x; a single shot of each
    ///   arm is not a measurement.
    ///
    /// `MUMDIA_BENCH_ROWS`, `_D`, `_FOLDS`, `_ITERS` and `_REPS` scale it. Run with
    /// `cargo test -p mumdia --release -- --ignored --nocapture percolator_lite_fold_cost`.
    ///
    /// What it does NOT measure: the `rescore` stage. This is the fitter over a synthetic
    /// population, with none of the parquet load, the feature build, the handoff or the
    /// grouped q that surround it in a real run, so no share-of-stage-runtime claim can be
    /// made from it.
    #[test]
    #[ignore = "microbenchmark; meaningful only in release"]
    fn percolator_lite_fold_cost() {
        let env = |k: &str, dflt: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(dflt)
        };
        let (n, d, folds, num_iter, reps) = (
            env("MUMDIA_BENCH_ROWS", 30_000),
            env("MUMDIA_BENCH_D", 387),
            env("MUMDIA_BENCH_FOLDS", 3),
            env("MUMDIA_BENCH_ITERS", 10),
            env("MUMDIA_BENCH_REPS", 3).max(1),
        );
        let (features, is_decoy, key, init) = crafted_population(n, d);
        let mk = || RescoreInput {
            features: &features,
            is_decoy: &is_decoy,
            fold_key: &key,
            init_score: &init,
            folds,
            num_iter,
            train_fdr: 0.05,
        };
        let bits = |v: Vec<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let time = |f: &dyn Fn() -> Vec<f64>| {
            let t = std::time::Instant::now();
            let v = f();
            (t.elapsed().as_secs_f64(), bits(v))
        };
        // `MUMDIA_BENCH_BASELINE=per_row` times the OLD, WRONG baseline instead. It is here
        // only so the gap between the two baselines can be measured rather than asserted;
        // `parent` is the default and is the code this change actually replaced.
        let per_row = std::env::var("MUMDIA_BENCH_BASELINE").as_deref() == Ok("per_row");
        let base_name = if per_row { "per_row" } else { "parent" };
        let old = || {
            if per_row {
                percolator_lite_per_row_reference(mk())
            } else {
                percolator_lite_parent(mk())
            }
        };
        let new = || percolator_lite(mk());
        let (mut olds, mut news) = (Vec::new(), Vec::new());
        for rep in 0..reps {
            // Alternate which arm goes first, so a first-mover advantage (page faults,
            // turbo residency, the allocator's state) cannot accrue to one arm.
            let (a, b) = if rep % 2 == 0 {
                let a = time(&old);
                let b = time(&new);
                (a, b)
            } else {
                let b = time(&new);
                let a = time(&old);
                (a, b)
            };
            assert_eq!(a.1, b.1, "the benchmark arms must agree bit for bit");
            println!(
                "  rep {rep} ({}): {base_name} {:.3} s, sequential {:.3} s ({:.2}x)",
                if rep % 2 == 0 {
                    "baseline first"
                } else {
                    "new first"
                },
                a.0,
                b.0,
                a.0 / b.0
            );
            olds.push(a.0);
            news.push(b.0);
        }
        let lo = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "percolator_lite {n} PSMs x {d} features, {folds} folds x {num_iter} iters, \
             {} rayon threads, {reps} reps: parallel folds + serial fit [{base_name}] \
             {:.3} s (spread {:.2}x), sequential folds + parallel fit {:.3} s \
             (spread {:.2}x), best-of ratio {:.2}x",
            rayon::current_num_threads(),
            lo(&olds),
            hi(&olds) / lo(&olds),
            lo(&news),
            hi(&news) / lo(&news),
            lo(&olds) / lo(&news)
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
