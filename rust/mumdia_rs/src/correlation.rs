/// Pearson correlation functions matching MuMDIA's Numba implementations.

/// Compute Pearson correlation between two 1D arrays.
/// Returns 0.0 if either array has zero variance (safe variant).
pub fn pearson_1d_impl(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 || n != b.len() {
        return 0.0;
    }

    let mean_a: f64 = a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let std_a = (var_a / n as f64).sqrt();
    let std_b = (var_b / n as f64).sqrt();

    if std_a > 0.0 && std_b > 0.0 {
        cov / n as f64 / (std_a * std_b)
    } else {
        0.0
    }
}

/// Compute Pearson correlations between each row of a 2D matrix and a 1D prediction vector.
/// Returns one correlation per row. Matches compute_correlations() from features_fragment_intensity.py.
///
/// intensity_matrix: (num_psms, num_fragments) row-major
/// pred_frag_intens: (num_fragments,)
/// Returns: (num_psms,)
pub fn compute_correlations_impl(
    matrix_data: &[f64],
    num_rows: usize,
    num_cols: usize,
    preds: &[f64],
) -> Vec<f64> {
    let mut correlations = vec![0.0; num_rows];

    // Pre-compute prediction stats (same for all rows)
    let pred_mean: f64 = preds.iter().sum::<f64>() / num_cols as f64;
    let pred_var: f64 = preds.iter().map(|&v| (v - pred_mean).powi(2)).sum::<f64>();
    let pred_std = (pred_var / num_cols as f64).sqrt();

    if pred_std == 0.0 {
        return correlations; // all zeros if predictions are constant
    }

    for i in 0..num_rows {
        let row_start = i * num_cols;
        let row = &matrix_data[row_start..row_start + num_cols];

        let row_mean: f64 = row.iter().sum::<f64>() / num_cols as f64;
        let row_var: f64 = row.iter().map(|&v| (v - row_mean).powi(2)).sum::<f64>();
        let row_std = (row_var / num_cols as f64).sqrt();

        if row_std > 0.0 {
            let cov: f64 = row
                .iter()
                .zip(preds.iter())
                .map(|(&x, &y)| (x - row_mean) * (y - pred_mean))
                .sum::<f64>()
                / num_cols as f64;
            correlations[i] = cov / (row_std * pred_std);
        }
    }

    correlations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_1d_impl(&a, &b);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_pearson_anti_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let r = pearson_1d_impl(&a, &b);
        assert!((r - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_pearson_zero_variance() {
        let a = vec![3.0, 3.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(pearson_1d_impl(&a, &b), 0.0);
    }

    #[test]
    fn test_pearson_empty() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        assert_eq!(pearson_1d_impl(&a, &b), 0.0);
    }

    #[test]
    fn test_compute_correlations_perfect() {
        // 2 rows, 3 columns. Row 0 = 2x predictions, Row 1 = reversed.
        let preds = vec![1.0, 2.0, 3.0];
        let matrix = vec![
            2.0, 4.0, 6.0, // row 0: perfect correlation
            3.0, 2.0, 1.0, // row 1: anti-correlation
        ];
        let result = compute_correlations_impl(&matrix, 2, 3, &preds);
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_compute_correlations_zero_variance_row() {
        let preds = vec![1.0, 2.0, 3.0];
        let matrix = vec![
            5.0, 5.0, 5.0, // constant row
        ];
        let result = compute_correlations_impl(&matrix, 1, 3, &preds);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_compute_correlations_zero_variance_preds() {
        let preds = vec![3.0, 3.0, 3.0];
        let matrix = vec![1.0, 2.0, 3.0];
        let result = compute_correlations_impl(&matrix, 1, 3, &preds);
        assert_eq!(result[0], 0.0);
    }
}
