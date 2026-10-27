//! Stable-set cross-run normalisation (`quant.normalisation = "stable_set"`,
//! docs/12_quant_lfq_align_mbr_report_audit.md, "Cross-run normalisation").
//!
//! The estimator, on `y`, the log2 quantity of an ion (one `(peptidoform, charge)`) in a run:
//!
//! 1. Deviations. The reference of an ion in run `j` is the median over the OTHER runs of
//!    `y - s_r(rt)`, with `s_r` the current shift of run `r` at the ion's apex in that run,
//!    when the ion is quantified in at least `max(2, ceil(R / 2)) - 1` other runs. The
//!    deviation is `y_j` minus that reference, so a run never enters its own reference.
//! 2. A first pass fits every run's shift on the deviations of every ion (step 4).
//! 3. Three passes then select a stable set first. Complete ions (quantified in every run)
//!    give the profile `D = (y - s) - mean over runs`. The most abundant half (by mean
//!    `y`), less the 5% with the largest `|D|`, gives the principal components of the run
//!    space; every composition effect of a design with `G` sample groups lies in a
//!    `G - 1`-dimensional subspace. Its dimension `k` is the number of eigenvalues above 3
//!    times their median (at least 1, at most `R - 2`), or `quant.normalisation_rank`. In
//!    that subspace a mean shift from the coordinate median (Gaussian kernel of bandwidth
//!    `sigma`, the per-element residual SD outside the subspace, cut at 4 bandwidths, 200
//!    steps) finds the largest population; the stable ions are the complete ions within
//!    `2 sigma` of it.
//! 4. Per run, on the fit ions: at least 200 or the run is not shifted. The global shift
//!    is the median deviation; the curve is the median deviation of each of
//!    `min(20, n / 200)` equal-count retention-time bins at the bins' median retention
//!    time, smoothed 1/4, 1/2, 1/4 (edge bins repeat) when there are 3 or more, and
//!    interpolated linearly, constant beyond the outer bins (fewer than 2 bins: the global
//!    shift). A row without a retention time takes the global shift.
//! 5. The fitted runs' global shifts are centred on their mean.
//!
//! It reads no condition or species label; the automatic `k` needs neither, a fixed `k`
//! is the number of sample groups less one. Arithmetic follows the prototype it was
//! confirmed with (numpy's median, linear quantile and `interp`); the principal axes come
//! from a Jacobi eigendecomposition of the cross-product matrix, which agrees with the
//! prototype's SVD to rounding. Determinism: ions are indexed in a fixed order, every sort
//! has explicit tie-breakers, every reduction runs in one thread.

use super::quant_pattern::{median, quantile_sorted};

/// Passes after the first, each with a fresh stable set.
pub const ITERATIONS: usize = 3;
/// Equal-count retention-time bins per run, at most.
pub const RT_BINS: usize = 20;
/// Fit ions a run, and each bin, needs.
pub const MIN_IONS: usize = 200;
/// The share of the complete ions, by abundance from the top, that set the subspace.
const PCA_FRACTION: f64 = 0.5;
/// The profiles above this quantile of their norm are left out of the subspace fit.
const PCA_TRIM_QUANTILE: f64 = 0.95;
/// An eigenvalue above this multiple of their median counts toward the automatic `k`.
const RANK_FACTOR: f64 = 3.0;
/// The stable set is within this many `sigma` of the largest population.
const STABLE_Z: f64 = 2.0;
const MEAN_SHIFT_MAX_ITER: usize = 200;
const MEAN_SHIFT_TOL: f64 = 1e-9;
/// The kernel is cut where the squared scaled distance reaches this.
const KERNEL_CUTOFF_SQ: f64 = 16.0;

/// One run's shift curve, in log2 units: `quantity_normalised = quantity / 2^shift`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curve {
    /// The bins' median retention times, ascending; empty for a single shift.
    pub knot_rt: Vec<f64>,
    /// The smoothed bin shifts at `knot_rt`.
    pub knot_shift: Vec<f64>,
    /// The bin medians before smoothing.
    pub knot_median: Vec<f64>,
    /// Per bin: its lowest and highest retention time and its ions.
    pub knot_lo: Vec<f64>,
    pub knot_hi: Vec<f64>,
    pub knot_n: Vec<usize>,
    /// The shift of a row without a retention time, and of every row without knots.
    pub global: f64,
    /// Fit ions of the last pass, and whether the run was fitted (at least [`MIN_IONS`]).
    pub n_fit: usize,
    pub fitted: bool,
}

impl Curve {
    /// The shift at retention time `rt`, as numpy's `interp` evaluates it.
    pub fn at(&self, rt: Option<f64>) -> f64 {
        let Some(t) = rt.filter(|t| t.is_finite()) else {
            return self.global;
        };
        if self.knot_rt.is_empty() {
            return self.global;
        }
        np_interp(t, &self.knot_rt, &self.knot_shift)
    }
}

/// numpy's `interp(x, xp, fp)` for finite `x` and non-empty ascending `xp`.
fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    if j == n - 1 || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let r = slope * (x - xp[j]) + fp[j];
    if !r.is_nan() {
        return r;
    }
    let r = slope * (x - xp[j + 1]) + fp[j + 1];
    if r.is_nan() && fp[j] == fp[j + 1] {
        fp[j]
    } else {
        r
    }
}

/// What one pass's stable-set selection found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StableInfo {
    pub k: usize,
    pub eigenvalues: Vec<f64>,
    pub sigma: f64,
    pub n_complete: usize,
    pub n_stable: usize,
}

/// The fit: one curve per run and the stable-set selection of each pass after the first.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StableFit {
    pub curves: Vec<Curve>,
    pub passes: Vec<StableInfo>,
}

/// The ions-by-runs layout the fit reads: `y[i * runs + j]` the log2 quantity of ion `i` in
/// run `j` (NaN when not quantified) and `rt` its apex retention time there (NaN when
/// unknown).
pub struct Matrix<'a> {
    pub y: &'a [f64],
    pub rt: &'a [f64],
    pub n_ions: usize,
    pub runs: usize,
}

/// The current shift of every (ion, run).
fn shift_matrix(m: &Matrix, curves: &[Curve]) -> Vec<f64> {
    let mut s = vec![0.0f64; m.y.len()];
    for i in 0..m.n_ions {
        for (j, c) in curves.iter().enumerate() {
            let rt = m.rt[i * m.runs + j];
            s[i * m.runs + j] = c.at(rt.is_finite().then_some(rt));
        }
    }
    s
}

/// Leave-one-out deviations and references.
fn deviations(m: &Matrix, s: &[f64], need: usize) -> (Vec<f64>, Vec<f64>) {
    let r = m.runs;
    let mut dev = vec![f64::NAN; m.y.len()];
    let mut ab = vec![f64::NAN; m.y.len()];
    let mut buf: Vec<f64> = Vec::with_capacity(r);
    for i in 0..m.n_ions {
        for j in 0..r {
            let y = m.y[i * r + j];
            if !y.is_finite() {
                continue;
            }
            buf.clear();
            for jj in 0..r {
                if jj != j {
                    let z = m.y[i * r + jj] - s[i * r + jj];
                    if z.is_finite() {
                        buf.push(z);
                    }
                }
            }
            if buf.is_empty() || buf.len() + 1 < need {
                continue;
            }
            let reference = median(&mut buf);
            dev[i * r + j] = y - reference;
            ab[i * r + j] = reference;
        }
    }
    (dev, ab)
}

/// Jacobi eigendecomposition of the symmetric `n x n` matrix `a` (row-major): eigenvalues
/// descending, with their unit eigenvectors as the columns of the returned row-major
/// matrix.
fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let total: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    for _ in 0..100 {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off.sqrt() <= 1e-15 * total.max(f64::MIN_POSITIVE) {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - sn * akq;
                    a[k * n + q] = sn * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - sn * aqk;
                    a[q * n + k] = sn * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - sn * vkq;
                    v[k * n + q] = sn * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| a[y * n + y].total_cmp(&a[x * n + x]).then(x.cmp(&y)));
    let vals: Vec<f64> = order.iter().map(|&i| a[i * n + i]).collect();
    let mut vecs = vec![0.0f64; n * n];
    for (c, &i) in order.iter().enumerate() {
        for k in 0..n {
            vecs[k * n + c] = v[k * n + i];
        }
    }
    (vals, vecs)
}

/// The stable set of one pass (a mask over ions), and what selected it.
fn stable_set(m: &Matrix, s: &[f64], rank: Option<usize>) -> (Vec<bool>, StableInfo) {
    let r = m.runs;
    let mut stable = vec![false; m.n_ions];
    let complete: Vec<usize> = (0..m.n_ions)
        .filter(|&i| (0..r).all(|j| m.y[i * r + j].is_finite()))
        .collect();
    let nc = complete.len();
    let mut info = StableInfo {
        n_complete: nc,
        ..StableInfo::default()
    };
    // Profiles: (y - s) less the ion's mean over runs.
    let mut d = vec![0.0f64; nc * r];
    let mut ab = vec![0.0f64; nc];
    for (ci, &i) in complete.iter().enumerate() {
        let mut zs = 0.0f64;
        let mut ys = 0.0f64;
        for j in 0..r {
            zs += m.y[i * r + j] - s[i * r + j];
            ys += m.y[i * r + j];
        }
        let zm = zs / r as f64;
        for j in 0..r {
            d[ci * r + j] = (m.y[i * r + j] - s[i * r + j]) - zm;
        }
        ab[ci] = ys / r as f64;
    }
    // The most abundant half (ties in ion order).
    let n_top = ((nc as f64 * PCA_FRACTION).ceil() as usize).min(nc);
    let mut order: Vec<usize> = (0..nc).collect();
    order.sort_by(|&a, &b| ab[b].total_cmp(&ab[a]).then(a.cmp(&b)));
    let top: Vec<usize> = order[..n_top].to_vec();
    let norm = |ci: usize| -> f64 {
        let mut q = 0.0f64;
        for j in 0..r {
            q += d[ci * r + j] * d[ci * r + j];
        }
        q.sqrt()
    };
    let nrm: Vec<f64> = top.iter().map(|&ci| norm(ci)).collect();
    if nrm.len() < 2 || r < 2 {
        return (stable, info);
    }
    let mut sorted = nrm.clone();
    sorted.sort_by(f64::total_cmp);
    let cut = quantile_sorted(&sorted, PCA_TRIM_QUANTILE);
    let fit: Vec<usize> = top
        .iter()
        .zip(&nrm)
        .filter(|(_, &q)| q <= cut)
        .map(|(&ci, _)| ci)
        .collect();
    let nf = fit.len();
    if nf < 2 {
        return (stable, info);
    }
    let mut mean = vec![0.0f64; r];
    for &ci in &fit {
        for j in 0..r {
            mean[j] += d[ci * r + j];
        }
    }
    for x in mean.iter_mut() {
        *x /= nf as f64;
    }
    let mut cross = vec![0.0f64; r * r];
    for &ci in &fit {
        for a in 0..r {
            let xa = d[ci * r + a] - mean[a];
            for b in a..r {
                cross[a * r + b] += xa * (d[ci * r + b] - mean[b]);
            }
        }
    }
    for a in 0..r {
        for b in 0..a {
            cross[a * r + b] = cross[b * r + a];
        }
    }
    let (vals, vecs) = symmetric_eigen(&cross, r);
    let lam: Vec<f64> = vals[..r - 1].iter().map(|v| v / nf as f64).collect();
    let k = match rank {
        Some(k) => k,
        None => {
            let mut l = lam.clone();
            let med = median(&mut l);
            let above = lam.iter().filter(|&&v| v > RANK_FACTOR * med).count();
            above.max(1).min(r.saturating_sub(2))
        }
    };
    info.k = k;
    info.eigenvalues = lam;
    // Coordinates in the subspace and the residual outside it.
    let mut t = vec![0.0f64; nc * k];
    let mut resid2 = vec![0.0f64; nc];
    for ci in 0..nc {
        for c in 0..k {
            let mut acc = 0.0f64;
            for j in 0..r {
                acc += d[ci * r + j] * vecs[j * r + c];
            }
            t[ci * k + c] = acc;
        }
        let mut q = 0.0f64;
        for j in 0..r {
            let mut proj = 0.0f64;
            for c in 0..k {
                proj += t[ci * k + c] * vecs[j * r + c];
            }
            let e = d[ci * r + j] - proj;
            q += e * e;
        }
        resid2[ci] = q;
    }
    let mut rt2: Vec<f64> = top.iter().map(|&ci| resid2[ci]).collect();
    let dof = (r as f64 - 1.0 - k as f64).max(1.0);
    let sigma = (median(&mut rt2) / dof).sqrt();
    info.sigma = sigma;
    // The largest population: a mean shift from the coordinate median of the top ions.
    let mut x: Vec<f64> = (0..k)
        .map(|c| {
            let mut col: Vec<f64> = top.iter().map(|&ci| t[ci * k + c]).collect();
            median(&mut col)
        })
        .collect();
    let h2 = sigma * sigma;
    if k > 0 {
        for _ in 0..MEAN_SHIFT_MAX_ITER {
            let mut sw = 0.0f64;
            let mut swx = vec![0.0f64; k];
            for &ci in &top {
                let mut dd = 0.0f64;
                for c in 0..k {
                    let e = t[ci * k + c] - x[c];
                    dd += e * e;
                }
                let dd = dd / h2;
                let w = if dd < KERNEL_CUTOFF_SQ {
                    (-0.5 * dd).exp()
                } else {
                    0.0
                };
                sw += w;
                for c in 0..k {
                    swx[c] += w * t[ci * k + c];
                }
            }
            let next: Vec<f64> = swx.iter().map(|v| v / sw).collect();
            let step = next
                .iter()
                .zip(&x)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, |m, v| if v > m || v.is_nan() { v } else { m });
            x = next;
            if step < MEAN_SHIFT_TOL {
                break;
            }
        }
    }
    let lim = STABLE_Z * sigma;
    for (ci, &i) in complete.iter().enumerate() {
        let mut q = 0.0f64;
        for c in 0..k {
            let e = t[ci * k + c] - x[c];
            q += e * e;
        }
        if q.sqrt() <= lim {
            stable[i] = true;
        }
    }
    info.n_stable = stable.iter().filter(|&&b| b).count();
    (stable, info)
}

/// One run's curve from its fit ions: `(deviation, retention time, reference)`.
fn fit_run(pts: &[(f64, f64, f64)]) -> Curve {
    let n_all = pts.len();
    if n_all < MIN_IONS {
        return Curve {
            n_fit: n_all,
            ..Curve::default()
        };
    }
    let mut all: Vec<f64> = pts.iter().map(|p| p.0).collect();
    let global = median(&mut all);
    let mut timed: Vec<(f64, f64, f64)> = pts.iter().copied().filter(|p| p.1.is_finite()).collect();
    timed.sort_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then(a.0.total_cmp(&b.0))
            .then(a.2.total_cmp(&b.2))
    });
    let n = timed.len();
    let bins = RT_BINS.min(n / MIN_IONS).max(1);
    let mut c = Curve {
        global,
        n_fit: n_all,
        fitted: true,
        ..Curve::default()
    };
    if bins < 2 {
        return c;
    }
    for b in 0..bins {
        let slice = &timed[b * n / bins..(b + 1) * n / bins];
        let mut rts: Vec<f64> = slice.iter().map(|p| p.1).collect();
        let mut devs: Vec<f64> = slice.iter().map(|p| p.0).collect();
        c.knot_lo.push(rts[0]);
        c.knot_hi.push(rts[rts.len() - 1]);
        c.knot_n.push(slice.len());
        c.knot_rt.push(median(&mut rts));
        c.knot_median.push(median(&mut devs));
    }
    let km = c.knot_median.clone();
    c.knot_shift = if bins >= 3 {
        (0..bins)
            .map(|b| {
                let prev = km[b.saturating_sub(1)];
                let next = km[(b + 1).min(bins - 1)];
                0.25 * prev + 0.5 * km[b] + 0.25 * next
            })
            .collect()
    } else {
        km
    };
    c
}

/// Fit every run's curve. Fewer than two runs give zero shifts. `rank` fixes the subspace
/// dimension (`quant.normalisation_rank`); it must leave a residual dimension
/// (`rank <= runs - 2`), which the caller checks.
pub fn fit(m: &Matrix, rank: Option<usize>) -> StableFit {
    let r = m.runs;
    let mut curves: Vec<Curve> = vec![Curve::default(); r];
    let mut passes: Vec<StableInfo> = Vec::new();
    if r < 2 {
        return StableFit { curves, passes };
    }
    let need = r.div_ceil(2).max(2);
    for pass in 0..=ITERATIONS {
        let s = shift_matrix(m, &curves);
        let (dev, ab) = deviations(m, &s, need);
        let stable = if pass > 0 {
            let (st, info) = stable_set(m, &s, rank);
            passes.push(info);
            Some(st)
        } else {
            None
        };
        let mut next: Vec<Curve> = (0..r)
            .map(|j| {
                let pts: Vec<(f64, f64, f64)> = (0..m.n_ions)
                    .filter(|&i| {
                        dev[i * r + j].is_finite() && stable.as_ref().is_none_or(|st| st[i])
                    })
                    .map(|i| (dev[i * r + j], m.rt[i * r + j], ab[i * r + j]))
                    .collect();
                fit_run(&pts)
            })
            .collect();
        let gl: Vec<f64> = next.iter().filter(|c| c.fitted).map(|c| c.global).collect();
        if !gl.is_empty() {
            let cc = gl.iter().sum::<f64>() / gl.len() as f64;
            for c in next.iter_mut().filter(|c| c.fitted) {
                c.global -= cc;
                for v in c.knot_shift.iter_mut() {
                    *v -= cc;
                }
                for v in c.knot_median.iter_mut() {
                    *v -= cc;
                }
            }
        }
        curves = next;
    }
    StableFit { curves, passes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interp_follows_numpy() {
        let xp = [1.0, 2.0, 4.0];
        let fp = [0.0, 1.0, -1.0];
        assert_eq!(np_interp(0.0, &xp, &fp), 0.0);
        assert_eq!(np_interp(1.0, &xp, &fp), 0.0);
        assert_eq!(np_interp(1.5, &xp, &fp), 0.5);
        assert_eq!(np_interp(2.0, &xp, &fp), 1.0);
        assert_eq!(np_interp(3.0, &xp, &fp), 0.0);
        assert_eq!(np_interp(4.0, &xp, &fp), -1.0);
        assert_eq!(np_interp(9.0, &xp, &fp), -1.0);
        assert_eq!(np_interp(5.0, &[5.0], &[2.0]), 2.0);
    }

    #[test]
    fn jacobi_recovers_a_known_spectrum() {
        // diag(3, 1) rotated by 30 degrees.
        let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let a = [
            3.0 * c * c + s * s,
            (3.0 - 1.0) * c * s,
            (3.0 - 1.0) * c * s,
            3.0 * s * s + c * c,
        ];
        let (vals, vecs) = symmetric_eigen(&a, 2);
        assert!((vals[0] - 3.0).abs() < 1e-14 && (vals[1] - 1.0).abs() < 1e-14);
        assert!((vecs[0].abs() - c).abs() < 1e-14 && (vecs[2].abs() - s).abs() < 1e-14);
    }
}
