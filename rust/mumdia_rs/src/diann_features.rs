/// DIA-NN-style feature calculation in Rust.
/// Replaces the entire Python DIANNFeatureGenerator for the per-peptidoform case.
///
/// Core algorithm:
/// 1. Build pivot table (unique RTs × unique fragments → mean intensity)
/// 2. Find best fragment (highest total intensity)
/// 3. Smooth best fragment's elution profile (3-point moving average ≈ savgol w=3 p=1)
/// 4. Correlate all fragments against smoothed best → co-elution scores
/// 5. Extract top-N correlations, AUC, relative intensities, etc.
use std::collections::HashMap;

use crate::correlation::pearson_1d_impl;

/// 3-point moving average (equivalent to Savitzky-Golay with window=3, polyorder=1).
fn smooth_3pt(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n < 3 {
        return data.to_vec();
    }
    let mut result = vec![0.0; n];
    result[0] = data[0]; // edge: keep original
    for i in 1..n - 1 {
        result[i] = (data[i - 1] + data[i] + data[i + 1]) / 3.0;
    }
    result[n - 1] = data[n - 1]; // edge: keep original
    result
}

/// Build a pivot table from parallel arrays: rt[], fragment_id[], intensity[].
/// Returns (unique_rts sorted, fragment_names, matrix[rt_idx][frag_idx] = mean intensity).
/// Missing values are NaN.
fn build_pivot_table(
    rts: &[f64],
    frag_ids: &[u32], // fragment index into fragment_names
    intensities: &[f64],
    n_unique_frags: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    // Collect unique sorted RTs
    let mut unique_rts: Vec<f64> = Vec::new();
    let mut rt_to_idx: HashMap<u64, usize> = HashMap::new(); // f64 bits → index

    for &rt in rts {
        let key = rt.to_bits();
        if !rt_to_idx.contains_key(&key) {
            let idx = unique_rts.len();
            rt_to_idx.insert(key, idx);
            unique_rts.push(rt);
        }
    }

    // Sort RTs and remap indices
    let mut rt_order: Vec<usize> = (0..unique_rts.len()).collect();
    rt_order.sort_by(|&a, &b| unique_rts[a].partial_cmp(&unique_rts[b]).unwrap());
    let sorted_rts: Vec<f64> = rt_order.iter().map(|&i| unique_rts[i]).collect();
    let mut old_to_new = vec![0usize; unique_rts.len()];
    for (new_idx, &old_idx) in rt_order.iter().enumerate() {
        old_to_new[old_idx] = new_idx;
    }

    let n_rts = sorted_rts.len();

    // Accumulate: sum and count for mean
    let mut sums = vec![vec![0.0f64; n_unique_frags]; n_rts];
    let mut counts = vec![vec![0u32; n_unique_frags]; n_rts];

    for i in 0..rts.len() {
        let rt_key = rts[i].to_bits();
        let rt_idx = old_to_new[rt_to_idx[&rt_key]];
        let frag_idx = frag_ids[i] as usize;
        if frag_idx < n_unique_frags {
            sums[rt_idx][frag_idx] += intensities[i];
            counts[rt_idx][frag_idx] += 1;
        }
    }

    // Compute means (NaN for missing)
    let mut matrix = vec![vec![f64::NAN; n_unique_frags]; n_rts];
    for r in 0..n_rts {
        for f in 0..n_unique_frags {
            if counts[r][f] > 0 {
                matrix[r][f] = sums[r][f] / counts[r][f] as f64;
            }
        }
    }

    (sorted_rts, matrix)
}

/// Find the fragment with highest total intensity (sum across all RTs).
fn find_best_fragment(matrix: &[Vec<f64>], n_frags: usize) -> usize {
    let mut best_idx = 0;
    let mut best_sum = f64::NEG_INFINITY;
    for f in 0..n_frags {
        let sum: f64 = matrix
            .iter()
            .map(|row| if row[f].is_nan() { 0.0 } else { row[f] })
            .sum();
        if sum > best_sum {
            best_sum = sum;
            best_idx = f;
        }
    }
    best_idx
}

/// Find top-N fragments by total intensity, return their indices.
fn find_top_n_fragments(matrix: &[Vec<f64>], n_frags: usize, n: usize) -> Vec<usize> {
    let mut sums: Vec<(usize, f64)> = (0..n_frags)
        .map(|f| {
            let sum: f64 = matrix
                .iter()
                .map(|row| if row[f].is_nan() { 0.0 } else { row[f] })
                .sum();
            (f, sum)
        })
        .collect();
    sums.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sums.iter().take(n).map(|&(idx, _)| idx).collect()
}

/// Extract a column from the pivot matrix, replacing NaN with 0.
fn column_filled(matrix: &[Vec<f64>], col: usize) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| if row[col].is_nan() { 0.0 } else { row[col] })
        .collect()
}

/// Compute trapezoidal AUC for sorted (x, y) pairs.
fn trapezoid_auc(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 {
        return if y.is_empty() { 0.0 } else { y[0] };
    }
    let mut area = 0.0;
    for i in 1..x.len() {
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0;
    }
    area
}

/// Compute all DIA-NN-style features for one peptidoform.
///
/// Input: parallel arrays of (rt, fragment_id, intensity) for all fragment observations,
/// plus fragment name list, precursor info.
///
/// Returns HashMap of feature_name → value.
pub fn compute_diann_features_impl(
    rts: &[f64],
    frag_ids: &[u32],
    intensities: &[f64],
    fragment_names: &[String],
    precursor_mz: f64,
    precursor_charge: i32,
    peptide_length: usize,
    top_n: usize,          // default 6
    top_n_extended: usize,  // default 12
) -> HashMap<String, f64> {
    let mut features = HashMap::with_capacity(60);
    let n_frags = fragment_names.len();

    if rts.is_empty() || n_frags == 0 {
        // Return minimal features for empty data
        features.insert("diann_precursor_mz".into(), precursor_mz);
        features.insert("diann_precursor_charge".into(), precursor_charge as f64);
        features.insert("diann_precursor_length".into(), peptide_length as f64);
        return features;
    }

    // Step 1: Build pivot table
    let (sorted_rts, matrix) = build_pivot_table(rts, frag_ids, intensities, n_frags);
    let n_rts = sorted_rts.len();

    if n_rts < 2 {
        features.insert("diann_precursor_mz".into(), precursor_mz);
        features.insert("diann_precursor_charge".into(), precursor_charge as f64);
        features.insert("diann_precursor_length".into(), peptide_length as f64);
        return features;
    }

    // Step 2: Find best fragment and smooth its profile
    let best_frag = find_best_fragment(&matrix, n_frags);
    let best_trace = column_filled(&matrix, best_frag);
    let smoothed_best = smooth_3pt(&best_trace);

    // Step 3: Compute correlations for ALL fragments vs smoothed best
    let mut all_correlations: Vec<(usize, f64)> = Vec::with_capacity(n_frags);
    for f in 0..n_frags {
        let frag_trace = column_filled(&matrix, f);
        let r = pearson_1d_impl(&smoothed_best, &frag_trace);
        all_correlations.push((f, r));
    }

    // Step 4: Top-N fragments by intensity
    let top_n_frags = find_top_n_fragments(&matrix, n_frags, top_n);
    let top_n_ext_frags = find_top_n_fragments(&matrix, n_frags, top_n_extended);

    // === Feature Group 1: Pearson correlations top-N (extended) ===
    for i in 0..top_n_extended {
        let val = if i < top_n_ext_frags.len() {
            let frag_idx = top_n_ext_frags[i];
            all_correlations
                .iter()
                .find(|&&(idx, _)| idx == frag_idx)
                .map(|&(_, r)| if r.is_nan() { 0.0 } else { r })
                .unwrap_or(0.0)
        } else {
            f64::NAN
        };
        features.insert(format!("diann_pearson_correlations_top_12_{i}"), val);
    }

    // === Feature Group 1: Sum of correlations (top-N) ===
    let top_corr_sum: f64 = top_n_frags
        .iter()
        .map(|&idx| {
            all_correlations
                .iter()
                .find(|&&(i, _)| i == idx)
                .map(|&(_, r)| if r.is_nan() { 0.0 } else { r })
                .unwrap_or(0.0)
        })
        .sum();
    features.insert("diann_sum_correlations_mass_accuracy".into(), top_corr_sum);

    // === Feature Group 1: Remaining fragment correlations ===
    let top_set: std::collections::HashSet<usize> = top_n_frags.iter().copied().collect();
    let remaining_sum: f64 = all_correlations
        .iter()
        .filter(|&&(idx, _)| !top_set.contains(&idx))
        .map(|&(_, r)| if r.is_nan() { 0.0 } else { r })
        .sum();
    let remaining_count = n_frags.saturating_sub(top_n_frags.len());
    features.insert("diann_remaining_fragments_correlations".into(), remaining_sum);
    features.insert(
        "diann_remaining_fragments_mean".into(),
        if remaining_count > 0 {
            remaining_sum / remaining_count as f64
        } else {
            0.0
        },
    );

    // === Feature Group 1: Best b-fragments correlation ===
    let mut b_corrs: Vec<f64> = Vec::new();
    for (idx, &(frag_idx, r)) in all_correlations.iter().enumerate() {
        if idx < fragment_names.len() && fragment_names[frag_idx].starts_with('b') {
            b_corrs.push(if r.is_nan() { 0.0 } else { r });
        }
    }
    b_corrs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let best_b_sum: f64 = b_corrs.iter().take(3).sum();
    features.insert("diann_best_b_fragments_correlation".into(), best_b_sum);

    // === Feature Group 4: Weighted AUC ===
    let mut weighted_auc = 0.0;
    for &frag_idx in &top_n_frags {
        let frag_trace = column_filled(&matrix, frag_idx);
        let auc = trapezoid_auc(&sorted_rts, &frag_trace);
        let corr = all_correlations
            .iter()
            .find(|&&(i, _)| i == frag_idx)
            .map(|&(_, r)| if r.is_nan() { 0.0 } else { r.abs() })
            .unwrap_or(0.0);
        weighted_auc += auc * corr;
    }
    features.insert(
        "diann_weighted_auc".into(),
        if weighted_auc > 0.0 {
            weighted_auc.ln()
        } else {
            0.0
        },
    );

    // === Feature Group 5: Relative intensities top-6 ===
    let mut max_intensities: Vec<f64> = top_n_frags
        .iter()
        .map(|&idx| {
            matrix
                .iter()
                .map(|row| if row[idx].is_nan() { 0.0 } else { row[idx] })
                .fold(0.0f64, f64::max)
        })
        .collect();
    let global_max = max_intensities.iter().cloned().fold(0.0f64, f64::max);
    if global_max > 0.0 {
        for v in &mut max_intensities {
            *v /= global_max;
        }
    }
    for i in 0..top_n {
        let val = if i < max_intensities.len() {
            max_intensities[i]
        } else {
            f64::NAN
        };
        features.insert(format!("diann_relative_intensities_top_6_{i}"), val);
    }

    // === Feature Group 7: RT apex ===
    let apex_idx = best_trace
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    features.insert("diann_rt_apex".into(), sorted_rts[apex_idx]);

    // === Feature Group 8: Scanning window splits ===
    if n_rts >= 4 {
        let mid = n_rts / 2;
        let first_half: f64 = best_trace[..mid].iter().sum();
        let second_half: f64 = best_trace[mid..].iter().sum();
        let total = first_half + second_half;
        features.insert(
            "diann_scanning_window_splits".into(),
            if total > 0.0 {
                first_half / total
            } else {
                0.5
            },
        );
    } else {
        features.insert("diann_scanning_window_splits".into(), 0.5);
    }

    // === Feature Group 10: Precursor characteristics ===
    features.insert("diann_precursor_mz".into(), precursor_mz);
    features.insert("diann_precursor_charge".into(), precursor_charge as f64);
    features.insert("diann_precursor_length".into(), peptide_length as f64);

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smooth_3pt() {
        let data = vec![1.0, 3.0, 2.0, 4.0, 3.0];
        let smoothed = smooth_3pt(&data);
        assert_eq!(smoothed.len(), 5);
        assert_eq!(smoothed[0], 1.0); // edge
        assert!((smoothed[1] - 2.0).abs() < 1e-12); // (1+3+2)/3
        assert_eq!(smoothed[4], 3.0); // edge
    }

    #[test]
    fn test_smooth_short() {
        let data = vec![1.0, 2.0];
        let smoothed = smooth_3pt(&data);
        assert_eq!(smoothed, data);
    }

    #[test]
    fn test_build_pivot_table() {
        let rts = vec![1.0, 1.0, 2.0, 2.0, 3.0];
        let frag_ids = vec![0, 1, 0, 1, 0];
        let intensities = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (sorted_rts, matrix) = build_pivot_table(&rts, &frag_ids, &intensities, 2);
        assert_eq!(sorted_rts, vec![1.0, 2.0, 3.0]);
        assert_eq!(matrix.len(), 3);
        assert!((matrix[0][0] - 10.0).abs() < 1e-12);
        assert!((matrix[0][1] - 20.0).abs() < 1e-12);
        assert!((matrix[1][0] - 30.0).abs() < 1e-12);
        assert!(matrix[2][1].is_nan()); // frag 1 not observed at RT 3.0
    }

    #[test]
    fn test_compute_diann_features_basic() {
        let rts = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let frag_ids = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let intensities = vec![
            100.0, 50.0, 200.0, 100.0, 300.0, 150.0, 200.0, 100.0,
        ];
        let fragment_names = vec!["b3".to_string(), "y5".to_string()];

        let features = compute_diann_features_impl(
            &rts,
            &frag_ids,
            &intensities,
            &fragment_names,
            500.0, // precursor_mz
            2,     // charge
            10,    // peptide_length
            6,     // top_n
            12,    // top_n_extended
        );

        assert!(features.contains_key("diann_precursor_mz"));
        assert!((features["diann_precursor_mz"] - 500.0).abs() < 1e-12);
        assert!((features["diann_precursor_charge"] - 2.0).abs() < 1e-12);
        assert!(features.contains_key("diann_rt_apex"));
        assert!(features.contains_key("diann_weighted_auc"));
        assert!(features.contains_key("diann_pearson_correlations_top_12_0"));
        // Fragments are perfectly correlated (both increase then decrease)
        let corr = features["diann_pearson_correlations_top_12_0"];
        assert!(corr > 0.9, "Expected high correlation, got {corr}");
    }

    #[test]
    fn test_compute_diann_features_empty() {
        let features = compute_diann_features_impl(
            &[],
            &[],
            &[],
            &[],
            400.0,
            3,
            8,
            6,
            12,
        );
        assert!((features["diann_precursor_mz"] - 400.0).abs() < 1e-12);
        assert_eq!(features.len(), 3); // only precursor features
    }

    #[test]
    fn test_trapezoid_auc() {
        // Rectangle: y=1 from x=0 to x=2
        assert!((trapezoid_auc(&[0.0, 2.0], &[1.0, 1.0]) - 2.0).abs() < 1e-12);
        // Triangle: y=0 at x=0, y=2 at x=2
        assert!((trapezoid_auc(&[0.0, 2.0], &[0.0, 2.0]) - 2.0).abs() < 1e-12);
    }
}
