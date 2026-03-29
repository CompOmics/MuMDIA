/// Fragment correlation pipeline — ports the matrix operations from match_fragments().
/// Computes per-PSM correlations, PSM-pair and fragment-pair correlation matrices,
/// apex correlations, and MAE metrics.
use crate::correlation::{compute_correlations_impl, pearson_1d_impl};

/// Cosine similarity between two vectors. Returns 0.0 if either has zero norm.
pub fn cosine_similarity_impl(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute full pairwise correlation matrix for rows of a matrix.
/// Returns the upper triangle (excluding diagonal) as a sorted 1D array.
/// If `square_values` is true, returns R² (squared correlations).
pub fn corrcoef_upper_triangle(
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    square_values: bool,
) -> Vec<f64> {
    if n_rows <= 1 {
        return vec![];
    }

    // Pre-compute means and stds for each row
    let mut means = vec![0.0; n_rows];
    let mut stds = vec![0.0; n_rows];
    for i in 0..n_rows {
        let start = i * n_cols;
        let row = &data[start..start + n_cols];
        let mean: f64 = row.iter().sum::<f64>() / n_cols as f64;
        means[i] = mean;
        let var: f64 = row.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_cols as f64;
        stds[i] = var.sqrt();
    }

    // Compute upper triangle correlations
    let n_pairs = n_rows * (n_rows - 1) / 2;
    let mut result = Vec::with_capacity(n_pairs);

    for i in 0..n_rows {
        for j in (i + 1)..n_rows {
            if stds[i] > 0.0 && stds[j] > 0.0 {
                let row_i = &data[i * n_cols..(i + 1) * n_cols];
                let row_j = &data[j * n_cols..(j + 1) * n_cols];
                let cov: f64 = row_i
                    .iter()
                    .zip(row_j.iter())
                    .map(|(&a, &b)| (a - means[i]) * (b - means[j]))
                    .sum::<f64>()
                    / n_cols as f64;
                let mut r = cov / (stds[i] * stds[j]);
                if square_values {
                    r = r * r;
                }
                result.push(r);
            } else {
                result.push(0.0);
            }
        }
    }

    // Also include the symmetric lower triangle (matching np.corrcoef behavior
    // where we extract all off-diagonal elements)
    let mut full_offdiag = Vec::with_capacity(n_pairs * 2);
    for i in 0..n_rows {
        for j in 0..n_rows {
            if i == j {
                continue;
            }
            if stds[i] > 0.0 && stds[j] > 0.0 {
                let row_i = &data[i * n_cols..(i + 1) * n_cols];
                let row_j = &data[j * n_cols..(j + 1) * n_cols];
                let cov: f64 = row_i
                    .iter()
                    .zip(row_j.iter())
                    .map(|(&a, &b)| (a - means[i]) * (b - means[j]))
                    .sum::<f64>()
                    / n_cols as f64;
                let mut r = cov / (stds[i] * stds[j]);
                if square_values {
                    r = r * r;
                }
                full_offdiag.push(r);
            } else {
                full_offdiag.push(0.0);
            }
        }
    }

    full_offdiag.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    full_offdiag
}

/// Complete fragment correlation pipeline.
///
/// Takes an intensity matrix (n_psms x n_all_frags, already padded with zeros for
/// unmatched predictions and NOT yet row-normalized), prediction vectors, and
/// apex PSM index. Returns the 9 values that form CorrelationResults.
///
/// Returns: (correlations, correlation_counts, sum_pred_frag_intens,
///           corr_matrix_psm_ids, corr_matrix_frag_ids,
///           most_intens_cor, most_intens_cos,
///           mse_avg_pred_intens, mse_avg_pred_intens_total)
pub fn compute_fragment_correlations_impl(
    intensity_data: &[f64],
    n_psms: usize,
    n_frags: usize,
    matched_predictions: &[f64],
    non_matched_sum: f64,
    apex_psm_idx: usize,
) -> FragmentCorrelationResults {
    if n_psms == 0 || n_frags == 0 {
        return FragmentCorrelationResults::empty();
    }

    // Step 1: Row-wise normalization (each PSM row / its max)
    let mut normalized = intensity_data.to_vec();
    for i in 0..n_psms {
        let start = i * n_frags;
        let end = start + n_frags;
        let row = &normalized[start..end];
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val > 0.0 {
            for j in start..end {
                normalized[j] /= max_val;
            }
        }
    }

    // Step 2: Per-PSM correlations vs predictions
    let correlations =
        compute_correlations_impl(&normalized, n_psms, n_frags, matched_predictions);

    // Step 3: Count non-zero entries per PSM row
    let mut correlation_counts = vec![0.0; n_psms];
    for i in 0..n_psms {
        let start = i * n_frags;
        let count = normalized[start..start + n_frags]
            .iter()
            .filter(|&&v| v != 0.0)
            .count();
        correlation_counts[i] = count as f64;
    }

    // Step 4: Sum of matched predictions
    let sum_pred: f64 = matched_predictions.iter().sum();

    // Step 5: Apex correlations (Pearson + cosine for the apex PSM)
    let (most_intens_cor, most_intens_cos) = if apex_psm_idx < n_psms {
        let start = apex_psm_idx * n_frags;
        let apex_row = &normalized[start..start + n_frags];
        let cor = pearson_1d_impl(apex_row, matched_predictions);
        let cos = cosine_similarity_impl(apex_row, matched_predictions);
        (cor, cos)
    } else {
        (0.0, 0.0)
    };

    // Step 6: PSM-pair correlation matrix (squared = R², sorted)
    let corr_matrix_psm =
        corrcoef_upper_triangle(&normalized, n_psms, n_frags, true);

    // Step 7: Fragment-pair correlation matrix (raw r, sorted)
    // Transpose: columns become rows
    let mut transposed = vec![0.0; n_frags * n_psms];
    for i in 0..n_psms {
        for j in 0..n_frags {
            transposed[j * n_psms + i] = normalized[i * n_frags + j];
        }
    }
    let corr_matrix_frag =
        corrcoef_upper_triangle(&transposed, n_frags, n_psms, false);

    // Step 8: MAE computation
    let mut total_abs_error = 0.0;
    for i in 0..n_psms {
        let start = i * n_frags;
        for j in 0..n_frags {
            total_abs_error += (normalized[start + j] - matched_predictions[j]).abs();
        }
    }
    let mse_avg = total_abs_error / n_psms as f64;
    let mse_avg_total = (total_abs_error + non_matched_sum) / n_psms as f64;

    FragmentCorrelationResults {
        correlations,
        correlation_counts,
        sum_pred_frag_intens: sum_pred,
        corr_matrix_psm_ids: corr_matrix_psm,
        corr_matrix_frag_ids: corr_matrix_frag,
        most_intens_cor,
        most_intens_cos,
        mse_avg_pred_intens: mse_avg,
        mse_avg_pred_intens_total: mse_avg_total,
    }
}

pub struct FragmentCorrelationResults {
    pub correlations: Vec<f64>,
    pub correlation_counts: Vec<f64>,
    pub sum_pred_frag_intens: f64,
    pub corr_matrix_psm_ids: Vec<f64>,
    pub corr_matrix_frag_ids: Vec<f64>,
    pub most_intens_cor: f64,
    pub most_intens_cos: f64,
    pub mse_avg_pred_intens: f64,
    pub mse_avg_pred_intens_total: f64,
}

impl FragmentCorrelationResults {
    pub fn empty() -> Self {
        Self {
            correlations: vec![],
            correlation_counts: vec![],
            sum_pred_frag_intens: 0.0,
            corr_matrix_psm_ids: vec![],
            corr_matrix_frag_ids: vec![],
            most_intens_cor: 0.0,
            most_intens_cos: 0.0,
            mse_avg_pred_intens: 0.0,
            mse_avg_pred_intens_total: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let r = cosine_similarity_impl(&a, &a);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let r = cosine_similarity_impl(&a, &b);
        assert!(r.abs() < 1e-12);
    }

    #[test]
    fn test_cosine_similarity_zero() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity_impl(&a, &b), 0.0);
    }

    #[test]
    fn test_corrcoef_upper_triangle_perfect() {
        // 2 identical rows → correlation = 1.0 → R² = 1.0
        let data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let result = corrcoef_upper_triangle(&data, 2, 3, true);
        assert!(!result.is_empty());
        // Both off-diagonal elements should be 1.0 (R² of perfect correlation)
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_corrcoef_single_row() {
        let data = vec![1.0, 2.0, 3.0];
        let result = corrcoef_upper_triangle(&data, 1, 3, false);
        assert!(result.is_empty());
    }

    #[test]
    fn test_fragment_correlations_basic() {
        // 3 PSMs, 4 fragments (including 1 unmatched = zero)
        let intensity = vec![
            0.8, 0.6, 0.4, 0.0, // PSM 0 (apex)
            0.5, 0.3, 0.2, 0.0, // PSM 1
            0.9, 0.7, 0.5, 0.0, // PSM 2
        ];
        let predictions = vec![0.9, 0.7, 0.5, 0.1]; // includes unmatched
        let non_matched_sum = 0.1;

        let result = compute_fragment_correlations_impl(
            &intensity,
            3,
            4,
            &predictions,
            non_matched_sum,
            0, // apex is PSM 0
        );

        assert_eq!(result.correlations.len(), 3);
        assert_eq!(result.correlation_counts.len(), 3);
        assert!(!result.corr_matrix_psm_ids.is_empty());
        assert!(!result.corr_matrix_frag_ids.is_empty());
        // Correlations should be in [-1, 1]
        for &r in &result.correlations {
            assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&r));
        }
    }

    #[test]
    fn test_fragment_correlations_empty() {
        let result = compute_fragment_correlations_impl(&[], 0, 0, &[], 0.0, 0);
        assert!(result.correlations.is_empty());
    }
}
