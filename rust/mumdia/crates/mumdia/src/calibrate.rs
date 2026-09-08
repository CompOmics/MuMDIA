//! RT calibration models (docs/08_rt_im_train.md): a LOESS local-linear smoother
//! with a least-squares linear fallback. Maps predicted iRT (arbitrary units) to
//! observed RT (seconds) for this run.

/// Ordinary least-squares line y = slope*x + intercept.
pub fn linear_fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    if n < 2.0 {
        // Degenerate: constant map to the mean observed value.
        let mean = if ys.is_empty() {
            0.0
        } else {
            ys.iter().sum::<f64>() / ys.len() as f64
        };
        return (0.0, mean);
    }
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return (0.0, sy / n);
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    (slope, intercept)
}

/// A LOESS smoother evaluated on a precomputed grid for fast bulk application.
///
/// Outside the training range the smoother continues the local fit at the nearest
/// boundary: the value at the boundary grid point and the slope of the local line
/// fitted there. It used to switch to the global least-squares line the moment `x`
/// left the grid, and the two models need not agree at the boundary: on
/// `y = 200 + 10 x^2` over `x in [0, 9.9]` (span 0.3) the prediction jumped from
/// 193.4 at `x = 1e-6` to 38.3 at `x = 0`, and from 1173.5 to 1018.4 at the top,
/// discontinuities of about 155 s in retention-time units that misplaced
/// gradient-edge peptides relative to a narrow extraction window (docs/29 #10).
pub struct Loess {
    grid_x: Vec<f64>,
    grid_y: Vec<f64>,
    /// The global least-squares line: the fit for fewer than four points, and the
    /// value a degenerate local window (no spread) falls back to. Not the
    /// extrapolation model.
    slope: f64,
    intercept: f64,
    /// Extrapolation slope below the first and above the last grid point; the boundary
    /// grid VALUE plus this slope keeps `predict` continuous at both ends.
    ///
    /// The secant of the fitted curve over its end decile, not the pointwise local slope
    /// at the boundary. The pointwise slope comes from the one place where the tricubic
    /// window is one-sided and the anchors are sparsest, so it is the noisiest number the
    /// fit produces, unbounded, and free to be negative: on 55 anchors with 120 s of
    /// scatter it ranged 6.9 to 28.4 against a global slope near 24, and a peptidoform
    /// one iRT decade outside the anchors landed 1077 s from where the previous global
    /// line put it, against an `w_rt` near 180 s (docs/31 F7). A negative slope inverts
    /// the iRT-to-RT map, which the global line it replaced could not do. The secant
    /// averages the same curve over many grid points, and the clamp below keeps the
    /// result monotone and within a factor of the global least-squares slope.
    lo_slope: f64,
    hi_slope: f64,
}

impl Loess {
    /// Fit a LOESS model. `span` is the fraction of points in each local fit.
    /// `grid_n` is the number of grid points for later interpolation.
    pub fn fit(xs: &[f64], ys: &[f64], span: f64, grid_n: usize) -> Loess {
        let (slope, intercept) = linear_fit(xs, ys);
        // sort points by x
        let mut idx: Vec<usize> = (0..xs.len()).collect();
        idx.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
        let sx: Vec<f64> = idx.iter().map(|&i| xs[i]).collect();
        let sy: Vec<f64> = idx.iter().map(|&i| ys[i]).collect();
        let n = sx.len();
        if n < 4 {
            // too few points: grid is just the linear line
            let (lo, hi) = (
                sx.first().cloned().unwrap_or(0.0),
                sx.last().cloned().unwrap_or(1.0),
            );
            let grid_x: Vec<f64> = (0..grid_n.max(2))
                .map(|k| lo + (hi - lo) * k as f64 / (grid_n.max(2) - 1) as f64)
                .collect();
            let grid_y: Vec<f64> = grid_x.iter().map(|&x| slope * x + intercept).collect();
            return Loess {
                grid_x,
                grid_y,
                slope,
                intercept,
                lo_slope: slope,
                hi_slope: slope,
            };
        }
        let k = ((span * n as f64).ceil() as usize).clamp(3, n);
        let (lo, hi) = (sx[0], sx[n - 1]);
        let gn = grid_n.max(2);
        let mut grid_x = Vec::with_capacity(gn);
        let mut grid_y = Vec::with_capacity(gn);
        for g in 0..gn {
            let x = lo + (hi - lo) * g as f64 / (gn - 1) as f64;
            let (y, _) = local_linear(&sx, &sy, x, k, slope, intercept);
            grid_x.push(x);
            grid_y.push(y);
        }
        let (lo_slope, hi_slope) = boundary_slopes(&grid_x, &grid_y, slope);
        Loess {
            grid_x,
            grid_y,
            slope,
            intercept,
            lo_slope,
            hi_slope,
        }
    }

    /// Predict at x by linear interpolation over the grid. Outside the training range
    /// the local fit at the nearest boundary is continued (its value there and its
    /// slope), so the prediction is continuous across the boundary; see the type's
    /// documentation for what the previous global-line switch did.
    pub fn predict(&self, x: f64) -> f64 {
        // NaN in, NaN out. With x = NaN both boundary comparisons are false and
        // `partition_point` returns 0, so the interpolation below indexed `grid_x[-1]`:
        // a panic in debug and an out-of-bounds read in release, reachable from one
        // library row whose `predicted_irt` is null, because the parquet reader maps a
        // null f32 to NaN (docs/31 F4). Callers treat a NaN prediction as "no
        // calibration for this row", which is the documented recall-safe sentinel.
        if !x.is_finite() {
            return f64::NAN;
        }
        let g = &self.grid_x;
        if g.len() < 2 {
            return self.slope * x + self.intercept;
        }
        if x <= g[0] {
            return self.grid_y[0] + self.lo_slope * (x - g[0]);
        }
        let last = g.len() - 1;
        if x >= g[last] {
            return self.grid_y[last] + self.hi_slope * (x - g[last]);
        }
        let j = g.partition_point(|&gx| gx < x);
        let (x0, x1) = (g[j - 1], g[j]);
        let (y0, y1) = (self.grid_y[j - 1], self.grid_y[j]);
        if (x1 - x0).abs() < 1e-12 {
            return y0;
        }
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }
}

/// Extrapolation slopes for the two ends of a fitted grid.
///
/// Each is the secant of the fitted curve over its end decile, clamped to be
/// non-negative (an inverted iRT-to-RT map is never right) and, when the global
/// least-squares slope is positive, to at most four times it. Both guards exist because
/// this number multiplies a distance that has no bound: it is applied to queries outside
/// the anchor range entirely. See the field documentation on [`Loess`].
fn boundary_slopes(grid_x: &[f64], grid_y: &[f64], global_slope: f64) -> (f64, f64) {
    let n = grid_x.len();
    if n < 2 {
        return (global_slope, global_slope);
    }
    let step = (n / 10).max(1).min(n - 1);
    let secant = |a: usize, b: usize| {
        let dx = grid_x[b] - grid_x[a];
        if dx.abs() < 1e-12 {
            global_slope
        } else {
            (grid_y[b] - grid_y[a]) / dx
        }
    };
    let cap = if global_slope.is_finite() && global_slope > 0.0 {
        4.0 * global_slope
    } else {
        f64::INFINITY
    };
    let clamp = |v: f64| {
        if !v.is_finite() {
            return if global_slope.is_finite() {
                global_slope.max(0.0)
            } else {
                0.0
            };
        }
        v.clamp(0.0, cap)
    };
    (clamp(secant(0, step)), clamp(secant(n - 1 - step, n - 1)))
}

/// Weighted local linear regression at `x0` over the `k` nearest points
/// (tricubic weights): the fitted value at `x0` and the slope of the local line.
/// Falls back to the global line (value and slope) if degenerate.
fn local_linear(
    sx: &[f64],
    sy: &[f64],
    x0: f64,
    k: usize,
    slope: f64,
    intercept: f64,
) -> (f64, f64) {
    let n = sx.len();
    // find window of k nearest by walking from the insertion point
    let mut lo = sx.partition_point(|&x| x < x0);
    let mut hi = lo;
    let mut count = 0;
    while count < k && (lo > 0 || hi < n) {
        let take_left = if hi >= n {
            true
        } else if lo == 0 {
            false
        } else {
            (x0 - sx[lo - 1]) <= (sx[hi] - x0)
        };
        if take_left {
            lo -= 1;
        } else {
            hi += 1;
        }
        count += 1;
    }
    let dmax = ((x0 - sx[lo]).abs())
        .max((sx[hi - 1] - x0).abs())
        .max(1e-12);
    let (mut sw, mut swx, mut swy, mut swxx, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in lo..hi {
        let d = (sx[i] - x0).abs() / dmax;
        let w = if d < 1.0 {
            let t = 1.0 - d * d * d;
            t * t * t
        } else {
            0.0
        };
        sw += w;
        swx += w * sx[i];
        swy += w * sy[i];
        swxx += w * sx[i] * sx[i];
        swxy += w * sx[i] * sy[i];
    }
    let denom = sw * swxx - swx * swx;
    if sw < 1e-12 || denom.abs() < 1e-12 {
        return (slope * x0 + intercept, slope);
    }
    let b = (sw * swxy - swx * swy) / denom;
    let a = (swy - b * swx) / sw;
    (a + b * x0, b)
}

/// The p-th percentile (0..1) of the values (a copy is sorted).
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let rank = (p.clamp(0.0, 1.0) * (v.len() as f64 - 1.0)).round() as usize;
    v[rank]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_recovers_line() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 3.0 * x + 5.0).collect();
        let (s, b) = linear_fit(&xs, &ys);
        assert!((s - 3.0).abs() < 1e-6 && (b - 5.0).abs() < 1e-6);
    }

    #[test]
    fn loess_tracks_nonlinear() {
        let xs: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x * x).collect();
        let lo = Loess::fit(&xs, &ys, 0.3, 50);
        // near x=5, y should be ~25
        assert!((lo.predict(5.0) - 25.0).abs() < 2.0, "{}", lo.predict(5.0));
    }

    #[test]
    fn loess_is_continuous_at_both_training_boundaries() {
        // The docs/29 #10 reproduction: a curved map whose global line disagrees with
        // the local fit at both ends. The old predictor jumped by about 155 at each
        // boundary; the continued local fit must not.
        let xs: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 200.0 + 10.0 * x * x).collect();
        let lo = Loess::fit(&xs, &ys, 0.3, 50);
        let eps = 1e-6;
        let (a0, a1) = (lo.predict(0.0), lo.predict(eps));
        assert!((a0 - a1).abs() < 1e-3, "lower boundary jump: {a0} vs {a1}");
        let (b0, b1) = (lo.predict(9.9), lo.predict(9.9 - eps));
        assert!((b0 - b1).abs() < 1e-3, "upper boundary jump: {b0} vs {b1}");
        // Extrapolation follows the LOCAL slope, not the global line's (99 here): the
        // curve is nearly flat at the low end and steep (about 200) at the high end.
        let low_step = lo.predict(0.0) - lo.predict(-1.0);
        assert!(
            (0.0..50.0).contains(&low_step),
            "low-end extrapolation should be nearly flat, got a step of {low_step}"
        );
        let high_step = lo.predict(10.9) - lo.predict(9.9);
        assert!(
            high_step > 120.0,
            "high-end extrapolation should follow the steep local slope, got {high_step}"
        );
        // Far from the data the extrapolation is a line through the boundary value.
        assert!(
            (lo.predict(-2.0) - (lo.predict(0.0) - 2.0 * low_step)).abs() < 1e-9,
            "extrapolation must be linear in x"
        );
    }

    #[test]
    fn a_nan_query_returns_nan_instead_of_indexing_out_of_bounds() {
        // docs/31 F4: one library row with a null predicted_irt reaches this as NaN.
        let xs: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
        let lo = Loess::fit(&xs, &ys, 0.3, 20);
        assert!(lo.predict(f64::NAN).is_nan());
        assert!(lo.predict(f64::INFINITY).is_nan());
        assert!(lo.predict(f64::NEG_INFINITY).is_nan());
        // A degenerate fit takes the linear path and must not panic either.
        let tiny = Loess::fit(&[1.0, 2.0], &[1.0, 2.0], 0.5, 2);
        assert!(tiny.predict(f64::NAN).is_nan());
    }

    #[test]
    fn boundary_extrapolation_stays_monotone_and_bounded_under_noise() {
        // docs/31 F7: the pointwise boundary slope came from the sparsest, most one-sided
        // local window in the fit. On noisy anchors it was free to be negative (inverting
        // the iRT-to-RT map) or several times the global slope, and it multiplies a
        // distance that is unbounded because it applies outside the anchor range.
        // Deterministic pseudo-noise so the assertion is reproducible.
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        let mut noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 120.0
        };
        let xs: Vec<f64> = (0..55).map(|i| i as f64 * 2.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 500.0 + 24.0 * x + noise()).collect();
        let lo = Loess::fit(&xs, &ys, 0.3, 200);
        let (slope, _) = linear_fit(&xs, &ys);
        for s in [lo.lo_slope, lo.hi_slope] {
            assert!(s >= 0.0, "an inverted extrapolation slope: {s}");
            assert!(
                s <= 4.0 * slope,
                "an unbounded extrapolation slope: {s} vs {slope}"
            );
        }
        // Monotone outside the range, and continuous at both ends.
        let x0 = xs[0];
        let xn = xs[xs.len() - 1];
        assert!(lo.predict(x0 - 30.0) <= lo.predict(x0));
        assert!(lo.predict(xn + 30.0) >= lo.predict(xn));
        assert!((lo.predict(x0) - lo.predict(x0 + 1e-6)).abs() < 1.0);
        assert!((lo.predict(xn) - lo.predict(xn - 1e-6)).abs() < 1.0);
        // And the error one decade outside the anchors stays within a search window.
        let far = lo.predict(x0 - 30.0);
        let linear = slope * (x0 - 30.0) + lo.intercept;
        assert!(
            (far - linear).abs() < 600.0,
            "extrapolation {far} is implausibly far from the global line {linear}"
        );
    }

    #[test]
    fn percentile_basic() {
        let v: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        assert!((percentile(&v, 0.95) - 95.0).abs() < 1.0);
    }
}
