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
pub struct Loess {
    grid_x: Vec<f64>,
    grid_y: Vec<f64>,
    // linear fallback for x outside the training range
    slope: f64,
    intercept: f64,
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
            };
        }
        let k = ((span * n as f64).ceil() as usize).clamp(3, n);
        let (lo, hi) = (sx[0], sx[n - 1]);
        let gn = grid_n.max(2);
        let mut grid_x = Vec::with_capacity(gn);
        let mut grid_y = Vec::with_capacity(gn);
        for g in 0..gn {
            let x = lo + (hi - lo) * g as f64 / (gn - 1) as f64;
            grid_x.push(x);
            grid_y.push(local_linear(&sx, &sy, x, k, slope, intercept));
        }
        Loess {
            grid_x,
            grid_y,
            slope,
            intercept,
        }
    }

    /// Predict at x by linear interpolation over the grid; linear extrapolation
    /// outside the training range.
    pub fn predict(&self, x: f64) -> f64 {
        let g = &self.grid_x;
        if x <= g[0] || g.len() < 2 {
            return self.slope * x + self.intercept;
        }
        if x >= g[g.len() - 1] {
            return self.slope * x + self.intercept;
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

/// Weighted local linear regression at `x0` over the `k` nearest points
/// (tricubic weights). Falls back to the global line if degenerate.
fn local_linear(sx: &[f64], sy: &[f64], x0: f64, k: usize, slope: f64, intercept: f64) -> f64 {
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
        return slope * x0 + intercept;
    }
    let b = (sw * swxy - swx * swy) / denom;
    let a = (swy - b * swx) / sw;
    a + b * x0
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
    fn percentile_basic() {
        let v: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        assert!((percentile(&v, 0.95) - 95.0).abs() < 1.0);
    }
}
