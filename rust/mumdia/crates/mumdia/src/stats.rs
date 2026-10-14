//! Shared numeric kernels (docs/14_build_test_deploy_gotchas.md):
//! population-variance Pearson with a zero-variance guard returning 0, cosine
//! similarity, spectral angle, intensity-weighted median and mean absolute deviation.
//! One implementation, unit-tested.

/// Population Pearson correlation; returns 0 on zero variance (documented guard).
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let na = n as f64;
    let ma = a[..n].iter().sum::<f64>() / na;
    let mb = b[..n].iter().sum::<f64>() / na;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 0.0 || vb <= 0.0 {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// One vector centred once, for many Pearson correlations against partners of its own
/// length: its deviations from its mean and their sum of squares, computed with exactly
/// the operations [`pearson`] performs on it (the mean by the same `iter().sum::<f64>()`
/// divided by the same `n as f64`, the squares accumulated from `0.0` in ascending order).
/// That is what makes [`pearson_vs`] bit-identical to [`pearson`]: of the four reductions
/// in a Pearson correlation, this one's mean and variance are then done once instead of
/// once per partner.
pub struct Centered {
    d: Vec<f64>,
    va: f64,
}

/// Write the deviations of `a` from its mean into `out` (as long as `a`) and return their
/// sum of squares, with exactly the operations [`pearson`] performs on one argument.
fn centre_into(a: &[f64], out: &mut [f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let m = a.iter().sum::<f64>() / n as f64;
    let mut va = 0.0;
    for (dst, &x) in out.iter_mut().zip(a) {
        let dx = x - m;
        *dst = dx;
        va += dx * dx;
    }
    va
}

impl Centered {
    pub fn new(a: &[f64]) -> Centered {
        let mut d = vec![0.0f64; a.len()];
        let va = centre_into(a, &mut d);
        Centered { d, va }
    }

    pub fn len(&self) -> usize {
        self.d.len()
    }

    pub fn is_empty(&self) -> bool {
        self.d.is_empty()
    }
}

/// `pearson(a, r)` with `r` centred once: `rc` must be [`Centered::new`] of `r`. Bit for bit
/// equal to `pearson(a, r)`: the same deviations, the same products in the same operand
/// order, accumulated in the same order. A length mismatch, where `pearson` correlates the
/// shorter overlap with means of its own, is passed to `pearson` itself.
pub fn pearson_vs(a: &[f64], r: &[f64], rc: &Centered) -> f64 {
    let n = a.len();
    if n != r.len() || rc.len() != n {
        return pearson(a, r);
    }
    if n < 2 {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / n as f64;
    let (mut cov, mut va) = (0.0, 0.0);
    for (&x, &db) in a.iter().zip(&rc.d) {
        let da = x - ma;
        cov += da * db;
        va += da * da;
    }
    if va <= 0.0 || rc.va <= 0.0 {
        return 0.0;
    }
    cov / (va.sqrt() * rc.va.sqrt())
}

/// `pearson(rows[a], rows[b])` for every pair `a < b`, appended to `out` in `(a, b)`
/// lexicographic order, bit for bit.
///
/// A pair matrix recomputes both means and both variances in every one of its K(K-1)/2
/// correlations, although each is a property of one row. Here every row is centred once
/// (as [`Centered`] does) and each pair costs only its covariance, and the covariances of
/// one row against four partners run as four independent chains. Each chain still sums
/// its own products in ascending position from `0.0`, so every covariance, and the
/// `cov / (sqrt(va) * sqrt(vb))` built from it, is the one `pearson` computes. Rows of
/// unequal length, where `pearson` works on each pair's shorter overlap, go pair by pair
/// through `pearson` itself.
pub fn pearson_pairs<R: AsRef<[f64]>>(rows: &[R], out: &mut Vec<f64>) {
    let k = rows.len();
    if k < 2 {
        return;
    }
    let n = rows[0].as_ref().len();
    if rows.iter().any(|r| r.as_ref().len() != n) {
        for a in 0..k {
            for b in (a + 1)..k {
                out.push(pearson(rows[a].as_ref(), rows[b].as_ref()));
            }
        }
        return;
    }
    if n < 2 {
        out.extend(std::iter::repeat_n(0.0, k * (k - 1) / 2));
        return;
    }
    let mut d = vec![0.0f64; k * n];
    let mut va = vec![0.0f64; k];
    for (i, row) in rows.iter().enumerate() {
        va[i] = centre_into(row.as_ref(), &mut d[i * n..(i + 1) * n]);
    }
    let corr = |a: usize, b: usize, cov: f64| -> f64 {
        if va[a] <= 0.0 || va[b] <= 0.0 {
            0.0
        } else {
            cov / (va[a].sqrt() * va[b].sqrt())
        }
    };
    for a in 0..k {
        let da = &d[a * n..(a + 1) * n];
        let mut b = a + 1;
        while b + 4 <= k {
            let (d0, d1, d2, d3) = (
                &d[b * n..(b + 1) * n],
                &d[(b + 1) * n..(b + 2) * n],
                &d[(b + 2) * n..(b + 3) * n],
                &d[(b + 3) * n..(b + 4) * n],
            );
            let mut acc = [0.0f64; 4];
            for t in 0..n {
                let x = da[t];
                acc[0] += x * d0[t];
                acc[1] += x * d1[t];
                acc[2] += x * d2[t];
                acc[3] += x * d3[t];
            }
            for (j, &cov) in acc.iter().enumerate() {
                out.push(corr(a, b + j, cov));
            }
            b += 4;
        }
        while b < k {
            let db = &d[b * n..(b + 1) * n];
            let mut cov = 0.0;
            for (&x, &y) in da.iter().zip(db) {
                cov += x * y;
            }
            out.push(corr(a, b, cov));
            b += 1;
        }
    }
}

/// Cosine similarity; returns 0 if either vector is all-zero.
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Spectral angle in `[0, 1]` (1 = identical), derived from cosine.
pub fn spectral_angle(a: &[f64], b: &[f64]) -> f64 {
    let c = cosine(a, b).clamp(-1.0, 1.0);
    1.0 - 2.0 * c.acos() / std::f64::consts::PI
}

/// Weighted median of `(value, weight)` pairs: the smallest value whose cumulative weight
/// reaches half the total. `None` when empty or weightless. Sorts `pts` in place by value.
pub fn weighted_median(pts: &mut [(f64, f64)]) -> Option<f64> {
    let total: f64 = pts.iter().map(|p| p.1).sum();
    if pts.is_empty() || total.is_nan() || total <= 0.0 {
        return None;
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut acc = 0.0;
    for &(v, w) in pts.iter() {
        acc += w;
        if acc >= 0.5 * total {
            return Some(v);
        }
    }
    pts.last().map(|p| p.0)
}

/// Weighted mean absolute deviation of `(value, weight)` pairs from `center`:
/// `sum(w * |v - center|) / sum(w)`. Not a median absolute deviation. `None` when
/// weightless.
pub fn weighted_mean_abs_dev(pts: &[(f64, f64)], center: f64) -> Option<f64> {
    let total: f64 = pts.iter().map(|p| p.1).sum();
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    Some(
        pts.iter()
            .map(|&(v, w)| w * (v - center).abs())
            .sum::<f64>()
            / total,
    )
}

/// Weighted mean and population standard deviation of `(value, weight)` pairs:
/// `sum(w * v) / sum(w)` and `sqrt(sum(w * (v - mean)^2) / sum(w))`. `None` when the
/// weights sum to zero or less.
pub fn weighted_mean_sd(pts: &[(f64, f64)]) -> Option<(f64, f64)> {
    let sw: f64 = pts.iter().map(|p| p.1).sum();
    if sw <= 0.0 {
        return None;
    }
    let mean = pts.iter().map(|p| p.0 * p.1).sum::<f64>() / sw;
    let var = pts.iter().map(|p| p.1 * (p.0 - mean).powi(2)).sum::<f64>() / sw;
    Some((mean, var.sqrt()))
}

/// How [`loo_ref_corr`] forms the leave-one-out reference of member `f`: a profile built
/// from the members' weighted traces with `f`'s own contribution left out. The two forms
/// differ in floating point (a sum of the others is not bit-identical to a total minus one
/// term), and each caller keeps the form its values were defined with.
pub enum LooRef<'a> {
    /// `total[t] - w[f] * traces[f][t]` over the points the two share, `total` unchanged
    /// beyond them, where `total` is the weighted sum of every member, built by the caller:
    /// the co-elution family's `frag_loo_ref_corr_*` and retrace's `imc_ref_w`.
    Subtract(&'a [f64]),
    /// The weighted sum over `set` of every member but `f`, `len` points long, a member's
    /// trace cut at `len` and a member past the end of `traces` skipped: the interference
    /// family's prune, retained-set mean and dominant-fragment correlation.
    SumOthers { set: &'a [usize], len: usize },
}

/// Member `f`'s leave-one-out reference ([`LooRef`]), from the members' `traces` and their
/// weights `w` (the predicted intensities).
pub fn loo_reference<T: AsRef<[f64]>>(traces: &[T], w: &[f64], f: usize, r: LooRef) -> Vec<f64> {
    match r {
        LooRef::Subtract(total) => {
            let mut out = total.to_vec();
            let wf = w[f];
            for (v, x) in out.iter_mut().zip(traces[f].as_ref()) {
                *v -= wf * x;
            }
            out
        }
        LooRef::SumOthers { set, len } => {
            let mut acc = vec![0.0f64; len];
            for &g in set {
                if g == f || g >= traces.len() {
                    continue;
                }
                let wg = w[g];
                for (a, v) in acc.iter_mut().zip(traces[g].as_ref()) {
                    *a += wg * v;
                }
            }
            acc
        }
    }
}

/// The leave-one-out reference correlation of member `f`: the [`pearson`] correlation of
/// its trace with its leave-one-out reference ([`loo_reference`]). The one kernel every
/// leave-one-out reference correlation goes through: the co-elution and interference
/// families on elution traces, and retrace's `imc_ref_w` on 1/K0 profiles.
pub fn loo_ref_corr<T: AsRef<[f64]>>(traces: &[T], w: &[f64], f: usize, r: LooRef) -> f64 {
    pearson(traces[f].as_ref(), &loo_reference(traces, w, f, r))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel reproduces, bit for bit, the two private copies it replaced: the
    /// co-elution family's subtraction loop and the interference family's `loo_ref`. The
    /// 3D feature values therefore do not change. Traces of unequal length, members past
    /// the end and an excluded member are all exercised.
    #[test]
    fn the_loo_kernel_is_bit_identical_to_the_copies_it_replaced() {
        // The co-elution loop as it stood: R^{-f} = R - pred[f] * traces[f].
        fn coelution_old(traces: &[Vec<f64>], pred: &[f64], r: &[f64], k: usize) -> Vec<f64> {
            let mut loo = Vec::new();
            let t = r.len();
            for (i, trace) in traces.iter().enumerate().take(k) {
                let mut rm = r.to_vec();
                let lf = pred[i];
                for (idx, val) in rm.iter_mut().enumerate().take(t) {
                    if idx < trace.len() {
                        *val -= lf * trace[idx];
                    }
                }
                loo.push(pearson(trace, &rm));
            }
            loo
        }
        // The interference `loo_ref` as it stood.
        #[allow(clippy::needless_range_loop)]
        fn loo_ref_old(
            traces: &[Vec<f64>],
            pred: &[f64],
            set: &[usize],
            exclude: usize,
            tp: usize,
        ) -> Vec<f64> {
            let mut acc = vec![0.0f64; tp];
            for &g in set {
                if g == exclude || g >= traces.len() {
                    continue;
                }
                let x = &traces[g];
                let w = pred[g];
                let n = x.len().min(tp);
                for t in 0..n {
                    acc[t] += w * x[t];
                }
            }
            acc
        }
        // A deterministic pseudo-random stream.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for case in 0..200usize {
            let k = 1 + case % 9;
            let tp = 3 + case % 17;
            let traces: Vec<Vec<f64>> = (0..k)
                .map(|i| {
                    // Some traces shorter or longer than the reference.
                    let n = if i % 3 == 0 { tp } else { tp + i % 4 - 1 };
                    (0..n).map(|_| next() * 1e4).collect()
                })
                .collect();
            let pred: Vec<f64> = (0..k).map(|_| next()).collect();
            let r: Vec<f64> = (0..tp).map(|_| next() * 3e4).collect();
            let old = coelution_old(&traces, &pred, &r, k);
            for (f, want) in old.iter().enumerate() {
                let got = loo_ref_corr(&traces, &pred, f, LooRef::Subtract(&r));
                assert_eq!(got.to_bits(), want.to_bits(), "case {case}, fragment {f}");
            }
            // A set with members past the end, as the interference calls can pass.
            let set: Vec<usize> = (0..k + 2).filter(|g| g % 2 == 0 || *g >= k).collect();
            for &f in set.iter().filter(|&&f| f < k) {
                let sum = LooRef::SumOthers { set: &set, len: tp };
                let want = pearson(&traces[f], &loo_ref_old(&traces, &pred, &set, f, tp));
                assert_eq!(
                    loo_ref_corr(&traces, &pred, f, sum).to_bits(),
                    want.to_bits(),
                    "case {case}, member {f}"
                );
                let sum = LooRef::SumOthers { set: &set, len: tp };
                assert_eq!(
                    loo_reference(&traces, &pred, f, sum),
                    loo_ref_old(&traces, &pred, &set, f, tp)
                );
            }
        }
    }

    #[test]
    fn weighted_mean_sd_weighs_each_point() {
        assert_eq!(weighted_mean_sd(&[]), None);
        assert_eq!(weighted_mean_sd(&[(1.0, 0.0)]), None);
        let (m, sd) = weighted_mean_sd(&[(0.9, 1.0), (1.0, 1.0)]).unwrap();
        assert!((m - 0.95).abs() < 1e-12 && (sd - 0.05).abs() < 1e-12);
        // Weight 3 at 1.0 against 1 at 0.0: mean 0.75, variance 3/16.
        let (m, sd) = weighted_mean_sd(&[(1.0, 3.0), (0.0, 1.0)]).unwrap();
        assert!((m - 0.75).abs() < 1e-12);
        assert!((sd - (3.0f64 / 16.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn weighted_median_follows_the_intensity() {
        assert_eq!(weighted_median(&mut []), None);
        assert_eq!(weighted_median(&mut [(0.9, 0.0)]), None);
        // One intense peak outweighs two faint interferers on either side.
        assert_eq!(
            weighted_median(&mut [(1.10, 1.0), (0.85, 10.0), (0.70, 1.0)]),
            Some(0.85)
        );
        assert_eq!(weighted_median(&mut [(0.8, 1.0), (0.9, 1.0)]), Some(0.8));
    }

    #[test]
    fn weighted_mean_abs_dev_is_the_weighted_mean_distance() {
        assert_eq!(weighted_mean_abs_dev(&[], 1.0), None);
        assert_eq!(weighted_mean_abs_dev(&[(0.9, 0.0)], 1.0), None);
        // (1 * 0.5 + 3 * 0.1) / 4 = 0.2, where the weighted median distance would be 0.1.
        let d = weighted_mean_abs_dev(&[(0.5, 1.0), (1.1, 3.0)], 1.0).unwrap();
        assert!((d - 0.2).abs() < 1e-12, "{d}");
    }

    #[test]
    fn pearson_perfect_and_flat() {
        assert!((pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]) - 1.0).abs() < 1e-9);
        assert_eq!(pearson(&[1.0, 1.0, 1.0], &[2.0, 4.0, 6.0]), 0.0);
    }

    /// A deterministic stream for the equality tests below.
    fn stream(seed: u64) -> impl FnMut() -> f64 {
        let mut x = seed;
        move || {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (x >> 33) as f64 / (1u64 << 31) as f64
        }
    }

    /// Rows of every shape the correlation guards branch on: random, constant (zero
    /// variance), all-zero, a single spike, signed values, ties, signed zeros with
    /// subnormals, and a non-finite value.
    fn rows_of(n: usize, k: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut u = stream(seed);
        (0..k)
            .map(|r| {
                (0..n)
                    .map(|i| match r % 8 {
                        0 | 1 => u() * 1e3,
                        2 => 4.0,
                        3 => 0.0,
                        4 => {
                            if i == n / 2 {
                                2.5
                            } else {
                                0.0
                            }
                        }
                        5 => u() - 0.5,
                        6 => (u() * 4.0).floor(),
                        _ => {
                            if i % 3 == 0 {
                                -0.0
                            } else if i == 1 && seed.is_multiple_of(5) {
                                f64::NAN
                            } else {
                                f64::MIN_POSITIVE * u()
                            }
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn centred_pearson_matches_pearson_bit_for_bit() {
        for n in 0..40usize {
            for k in [0usize, 1, 2, 3, 5, 8, 12, 13] {
                let rows = rows_of(n, k, (n * 31 + k) as u64);
                let mut got = Vec::new();
                pearson_pairs(&rows, &mut got);
                let mut want = Vec::new();
                for a in 0..k {
                    for b in (a + 1)..k {
                        want.push(pearson(&rows[a], &rows[b]));
                    }
                }
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&want), "pairs, n {n}, k {k}");
                if let Some(r) = rows.first() {
                    let rc = Centered::new(r);
                    for row in &rows {
                        assert_eq!(
                            pearson_vs(row, r, &rc).to_bits(),
                            pearson(row, r).to_bits(),
                            "vs, n {n}, k {k}"
                        );
                    }
                }
            }
        }
        // Ragged rows take `pearson` pair by pair, and a reference of another length is
        // passed straight to it.
        let ragged = vec![
            vec![1.0, 2.0, 4.0],
            vec![3.0, 1.0],
            vec![0.5, 0.25, 0.0, 9.0],
        ];
        let mut got = Vec::new();
        pearson_pairs(&ragged, &mut got);
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            [(0, 1), (0, 2), (1, 2)]
                .iter()
                .map(|&(a, b)| pearson(&ragged[a], &ragged[b]).to_bits())
                .collect::<Vec<_>>()
        );
        let r = [1.0, 5.0];
        assert_eq!(
            pearson_vs(&ragged[0], &r, &Centered::new(&r)).to_bits(),
            pearson(&ragged[0], &r).to_bits()
        );
    }

    #[test]
    fn cosine_and_angle() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!((spectral_angle(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }
}
