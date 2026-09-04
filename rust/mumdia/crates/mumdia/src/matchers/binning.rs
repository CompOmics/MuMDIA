//! Log-space binning for the fragment index (docs/06_predict_frag_index_matchers.md).
//!
//! Bins are exactly one ppm-tolerance wide in log space, so two m/z values that
//! are within tolerance fall in bins that differ by at most 1. A query therefore
//! probes `bin-1, bin, bin+1` and verifies each posting with the exact
//! [`mumdia_core::constants::within_ppm`] predicate. All geometry is f64
//! regardless of how posting m/z is stored.

/// Precomputed log-space bin geometry for one tolerance and m/z range.
#[derive(Clone, Debug)]
pub struct LogBins {
    /// Multiplicative tolerance, `tol_ppm * 1e-6`.
    pub delta: f64,
    /// One-tolerance bin width in log space, `ln(1 + delta)`.
    pub w: f64,
    inv_w: f64,
    ln_min: f64,
    /// Number of bins (postings scatter into `0..n_bins`; `bin_start` has
    /// `n_bins + 1` entries).
    pub n_bins: usize,
}

impl LogBins {
    /// Build geometry from a tolerance and the m/z range `[mz_min, mz_max]`.
    /// `mz_min` must be > 0. The top is padded by one bin so `bin+1` never
    /// overflows the `bin_start` array.
    pub fn new(tol_ppm: f64, mz_min: f64, mz_max: f64) -> LogBins {
        assert!(mz_min > 0.0 && mz_max >= mz_min, "invalid m/z range");
        let delta = tol_ppm * 1e-6;
        let w = (1.0 + delta).ln();
        let inv_w = 1.0 / w;
        let ln_min = mz_min.ln();
        let span = mz_max.ln() - ln_min;
        // +2 pads the top so both bin(mz_max) and bin(mz_max)+1 are valid indices.
        let n_bins = (span * inv_w).floor() as usize + 2;
        LogBins {
            delta,
            w,
            inv_w,
            ln_min,
            n_bins,
        }
    }

    /// Bin index of an m/z value, clamped to `[0, n_bins - 1]`. Values at or below
    /// `mz_min` map to 0; values above `mz_max` are clamped to the top bin (the
    /// probe's `min(b+1, n_bins-1)` then yields an empty tail range).
    #[inline]
    pub fn bin(&self, mz: f64) -> usize {
        if mz <= 0.0 {
            return 0;
        }
        let y = (mz.ln() - self.ln_min) * self.inv_w;
        if y <= 0.0 {
            0
        } else {
            let b = y.floor() as usize;
            b.min(self.n_bins - 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LogBins;
    use mumdia_core::constants::within_ppm;

    #[test]
    fn within_tol_pairs_are_at_most_one_bin_apart() {
        let tol = 20.0;
        let bins = LogBins::new(tol, 200.0, 2000.0);
        // sample a grid; any within-tolerance pair must be <= 1 bin apart
        let mut mz = 200.0;
        while mz < 2000.0 {
            let partner = mz + tol * 1e-6 * mz * 0.99; // just inside tolerance
            assert!(within_ppm(mz, partner, tol));
            let db = (bins.bin(partner) as i64 - bins.bin(mz) as i64).abs();
            assert!(db <= 1, "bins differ by {db} at mz={mz}");
            mz *= 1.0007;
        }
    }

    #[test]
    fn out_of_range_clamps() {
        let bins = LogBins::new(20.0, 200.0, 2000.0);
        assert_eq!(bins.bin(0.0), 0);
        assert_eq!(bins.bin(-5.0), 0);
        assert_eq!(bins.bin(100.0), 0); // below min
        assert!(bins.bin(5000.0) < bins.n_bins); // above max, clamped
    }

    #[test]
    fn boundary_straddle_probe_covers_it() {
        // a pair straddling a bin boundary must be reachable by the +/-1 probe
        let tol = 20.0;
        let bins = LogBins::new(tol, 200.0, 2000.0);
        let a = 1000.0;
        let b = a + tol * 1e-6 * a * 0.5; // within tol
        let ba = bins.bin(a);
        let bb = bins.bin(b);
        assert!((bb as i64 - ba as i64).abs() <= 1);
    }
}
