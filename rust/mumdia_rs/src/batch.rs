/// Batch correlation feature extraction — replaces the entire
/// `run_peptidoform_correlation()` Python function with a single Rust call.
/// Eliminates ~10 Python→Rust round trips and all intermediate allocations.
use std::collections::HashMap;

use crate::percentiles::compute_percentiles_impl;
use crate::topk::compute_top_impl;

/// Compute all correlation-based features for one peptidoform in a single call.
///
/// This replicates the 10 calls to `add_feature_columns_nb()` that
/// `run_peptidoform_correlation()` makes in Python, returning a flat
/// feature name → value map.
pub fn batch_correlation_features_impl(
    correlations: &[f64],
    correlation_counts: &[f64],
    corr_matrix_psm: &[f64],
    corr_matrix_frag: &[f64],
    most_intens_cor: f64,
    most_intens_cos: f64,
    mse_avg: f64,
    mse_avg_total: f64,
    percentile_targets: &[f64],
    top_k_targets: &[usize],
    pad_size: usize,
) -> HashMap<String, f64> {
    let mut features = HashMap::with_capacity(80);

    // Helper: add percentile features
    let add_percentiles = |features: &mut HashMap<String, f64>,
                           data: &[f64],
                           prefix: &str,
                           targets: &[f64]| {
        let values = compute_percentiles_impl(data, targets);
        for (i, &t) in targets.iter().enumerate() {
            let t_int = t as i64;
            features.insert(format!("{prefix}_{t_int}"), values[i]);
        }
    };

    // Helper: add percentile features with index tracking
    let add_percentiles_with_idx = |features: &mut HashMap<String, f64>,
                                     data: &[f64],
                                     prefix: &str,
                                     targets: &[f64],
                                     idx_lookup: &[f64]| {
        // Sort data and track original indices for index lookup
        let n = data.len();
        if n == 0 {
            for &t in targets {
                let t_int = t as i64;
                features.insert(format!("{prefix}_{t_int}"), 0.0);
                features.insert(format!("{prefix}_{t_int}_idx"), 0.0);
            }
            return;
        }

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for &t in targets {
            let t_int = t as i64;
            // Compute percentile value
            let pos = (t / 100.0) * (n as f64 - 1.0);
            let lower = pos as usize;
            let upper = if lower >= n - 1 { lower } else { lower + 1 };
            let weight = pos - lower as f64;
            let value = sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
            features.insert(format!("{prefix}_{t_int}"), value);

            // Find nearest index in original data for this percentile value
            let nearest_idx = if !idx_lookup.is_empty() && lower < idx_lookup.len() {
                idx_lookup[lower]
            } else {
                0.0
            };
            features.insert(format!("{prefix}_{t_int}_idx"), nearest_idx);
        }
    };

    // Helper: add top-k features
    let add_top = |features: &mut HashMap<String, f64>,
                   data: &[f64],
                   prefix: &str,
                   targets: &[usize],
                   pad: usize| {
        let top_values = compute_top_impl(data, pad);
        for &t in targets {
            let val = if t > 0 && t <= top_values.len() {
                top_values[t - 1]
            } else {
                0.0
            };
            features.insert(format!("{prefix}_{t}"), val);
        }
    };

    // === 10 feature groups matching run_peptidoform_correlation() ===

    // 1. PSM correlation matrix distribution (percentiles)
    add_percentiles(
        &mut features,
        corr_matrix_psm,
        "distribution_correlation_matrix_psm_ids",
        percentile_targets,
    );

    // 2. Fragment correlation matrix distribution (percentiles)
    add_percentiles(
        &mut features,
        corr_matrix_frag,
        "distribution_correlation_matrix_frag_ids",
        percentile_targets,
    );

    // 3. Individual correlations distribution (percentiles with index tracking)
    add_percentiles_with_idx(
        &mut features,
        correlations,
        "distribution_correlation_individual",
        percentile_targets,
        correlation_counts,
    );

    // 4. Top PSM correlations
    add_top(
        &mut features,
        corr_matrix_psm,
        "top_correlation_matrix_psm_ids",
        top_k_targets,
        pad_size,
    );

    // 5. Top fragment correlations
    add_top(
        &mut features,
        corr_matrix_frag,
        "top_correlation_matrix_frag_ids",
        top_k_targets,
        pad_size,
    );

    // 6. Apex cosine similarity (single value)
    features.insert("top_correlation_cos_1".to_string(), most_intens_cos);

    // 7. Apex Pearson (overwrites cosine — matching the Python bug)
    features.insert("top_correlation_cos_1".to_string(), most_intens_cor);

    // 8. MSE average
    features.insert("mse_avg_pred_intens_1".to_string(), mse_avg);

    // 9. MSE total
    features.insert("mse_avg_pred_intens_total_1".to_string(), mse_avg_total);

    // 10. Top individual correlations
    add_top(
        &mut features,
        correlations,
        "top_correlation_individual",
        top_k_targets,
        pad_size,
    );

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_basic() {
        let correlations = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let counts = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let psm_matrix = vec![0.81, 0.64, 0.49, 0.36, 0.25];
        let frag_matrix = vec![0.9, 0.7, 0.5, 0.3, 0.1];
        let percentiles = vec![0.0, 25.0, 50.0, 75.0, 100.0];
        let top_k: Vec<usize> = (1..=10).collect();

        let result = batch_correlation_features_impl(
            &correlations,
            &counts,
            &psm_matrix,
            &frag_matrix,
            0.85, // most_intens_cor
            0.90, // most_intens_cos
            0.1,  // mse_avg
            0.15, // mse_avg_total
            &percentiles,
            &top_k,
            10,
        );

        // Check some expected feature names exist
        assert!(result.contains_key("distribution_correlation_matrix_psm_ids_0"));
        assert!(result.contains_key("distribution_correlation_matrix_psm_ids_50"));
        assert!(result.contains_key("top_correlation_matrix_psm_ids_1"));
        assert!(result.contains_key("top_correlation_individual_1"));
        assert!(result.contains_key("mse_avg_pred_intens_1"));
        assert!(result.contains_key("mse_avg_pred_intens_total_1"));

        // top_correlation_cos_1 should be most_intens_cor (the bug: Pearson overwrites cosine)
        assert!((result["top_correlation_cos_1"] - 0.85).abs() < 1e-12);
        assert!((result["mse_avg_pred_intens_1"] - 0.1).abs() < 1e-12);

        // Top-1 PSM correlation should be the largest value
        assert!((result["top_correlation_matrix_psm_ids_1"] - 0.81).abs() < 1e-12);
    }

    #[test]
    fn test_batch_empty_arrays() {
        let empty: Vec<f64> = vec![];
        let percentiles = vec![0.0, 50.0, 100.0];
        let top_k: Vec<usize> = vec![1, 2, 3];

        let result = batch_correlation_features_impl(
            &empty, &empty, &empty, &empty, 0.0, 0.0, 0.0, 0.0, &percentiles, &top_k, 10,
        );

        // All percentile features should be 0.0 for empty arrays
        assert_eq!(result["distribution_correlation_matrix_psm_ids_0"], 0.0);
        assert_eq!(result["top_correlation_matrix_psm_ids_1"], 0.0);
    }
}
