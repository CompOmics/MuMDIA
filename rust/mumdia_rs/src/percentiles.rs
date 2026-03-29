/// Percentile and quantile computations matching MuMDIA's Numba implementations.

/// Compute the q-th percentile using linear interpolation (matches numpy/numba behavior).
/// q is given as a float between 0 and 100.
pub fn percentile_impl(data: &[f64], q: f64) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let pos = (q / 100.0) * (n as f64 - 1.0);
    let lower = pos as usize;
    let upper = if lower >= n - 1 { lower } else { lower + 1 };
    let weight = pos - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

/// Compute the q-th percentile on pre-sorted data (no sort needed).
pub fn percentile_sorted_impl(sorted_data: &[f64], q: f64) -> f64 {
    let n = sorted_data.len();
    if n == 0 {
        return 0.0;
    }
    let pos = (q / 100.0) * (n as f64 - 1.0);
    let lower = pos as usize;
    let upper = if lower >= n - 1 { lower } else { lower + 1 };
    let weight = pos - lower as f64;
    sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
}

/// Compute multiple percentiles on a 1D array. Returns one value per quantile.
pub fn compute_percentiles_impl(data: &[f64], qs: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![0.0; qs.len()];
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    qs.iter()
        .map(|&q| percentile_sorted_impl(&sorted, q))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let median = percentile_impl(&data, 50.0);
        assert!((median - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_percentile_empty() {
        let data: Vec<f64> = vec![];
        assert_eq!(percentile_impl(&data, 50.0), 0.0);
    }

    #[test]
    fn test_percentile_single() {
        let data = vec![42.0];
        assert!((percentile_impl(&data, 0.0) - 42.0).abs() < 1e-12);
        assert!((percentile_impl(&data, 50.0) - 42.0).abs() < 1e-12);
        assert!((percentile_impl(&data, 100.0) - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_percentiles() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let qs = vec![0.0, 25.0, 50.0, 75.0, 100.0];
        let result = compute_percentiles_impl(&data, &qs);
        assert!((result[0] - 1.0).abs() < 1e-12); // min
        assert!((result[2] - 3.0).abs() < 1e-12); // median
        assert!((result[4] - 5.0).abs() < 1e-12); // max
        // q25 <= q50 <= q75
        assert!(result[1] <= result[2]);
        assert!(result[2] <= result[3]);
    }
}
