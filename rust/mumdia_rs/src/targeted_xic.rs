/// Targeted XIC (Extracted Ion Chromatogram) extraction from MS2 spectra.
///
/// For each peptidoform, extracts fragment ion intensities across ALL MS2 scans
/// in the elution window — not just the scans Sage assigned. This provides
/// complete elution profile data for co-elution scoring.
use std::collections::HashMap;

use crate::correlation::pearson_1d_impl;

/// Extract XIC features for a single peptidoform.
///
/// Searches all MS2 scans in the RT window for each target fragment m/z,
/// builds the XIC matrix, and computes co-elution features.
///
/// # Arguments
/// * `ms2_rts` - Sorted retention times for all MS2 scans
/// * `ms2_mz_offsets` - Start index in flat mz array for each scan
/// * `ms2_mz_lengths` - Number of peaks per scan
/// * `ms2_mz_flat` - All m/z values concatenated (sorted within each scan)
/// * `ms2_int_flat` - All intensity values concatenated
/// * `target_mzs` - Fragment m/z values to search for
/// * `target_predictions` - MS2PIP predicted intensities (for correlation)
/// * `rt_min`, `rt_max` - RT window bounds
/// * `ppm_tolerance` - Mass tolerance in ppm
pub fn extract_xic_features_impl(
    ms2_rts: &[f64],
    ms2_mz_offsets: &[u64],
    ms2_mz_lengths: &[u64],
    ms2_mz_flat: &[f64],
    ms2_int_flat: &[f64],
    target_mzs: &[f64],
    target_predictions: &[f64],
    rt_min: f64,
    rt_max: f64,
    ppm_tolerance: f64,
) -> HashMap<String, f64> {
    let n_frags = target_mzs.len();
    let n_scans = ms2_rts.len();

    if n_frags == 0 || n_scans == 0 {
        return HashMap::new();
    }

    // Find scan index range for [rt_min, rt_max] via binary search
    let scan_start = ms2_rts.partition_point(|&rt| rt < rt_min);
    let scan_end = ms2_rts.partition_point(|&rt| rt <= rt_max);
    let n_window_scans = scan_end - scan_start;

    if n_window_scans < 2 {
        let mut features = HashMap::new();
        features.insert("xic_coverage".into(), 0.0);
        features.insert("xic_total_scans".into(), n_window_scans as f64);
        return features;
    }

    // Build XIC matrix: (n_window_scans × n_frags), 0.0 for not detected
    let mut xic_matrix = vec![0.0f64; n_window_scans * n_frags];
    let mut scans_with_any_fragment = 0u32;

    for (scan_idx_in_window, scan_idx) in (scan_start..scan_end).enumerate() {
        let offset = ms2_mz_offsets[scan_idx] as usize;
        let length = ms2_mz_lengths[scan_idx] as usize;

        if length == 0 {
            continue;
        }

        let scan_mz = &ms2_mz_flat[offset..offset + length];
        let scan_int = &ms2_int_flat[offset..offset + length];
        let mut any_found = false;

        for (frag_idx, &target_mz) in target_mzs.iter().enumerate() {
            let tol = target_mz * ppm_tolerance * 1e-6;

            // Binary search for target m/z
            let pos = scan_mz.partition_point(|&mz| mz < target_mz - tol);

            // Check candidates around the insertion point
            let mut best_intensity = 0.0f64;
            for check_idx in pos.saturating_sub(1)..std::cmp::min(pos + 2, length) {
                let diff = (scan_mz[check_idx] - target_mz).abs();
                if diff <= tol && scan_int[check_idx] > best_intensity {
                    best_intensity = scan_int[check_idx];
                }
            }

            if best_intensity > 0.0 {
                xic_matrix[scan_idx_in_window * n_frags + frag_idx] = best_intensity;
                any_found = true;
            }
        }

        if any_found {
            scans_with_any_fragment += 1;
        }
    }

    // === Compute features from XIC matrix ===
    let mut features = HashMap::with_capacity(15);

    // Coverage features
    features.insert("xic_total_scans".into(), n_window_scans as f64);
    features.insert("xic_n_detected_scans".into(), scans_with_any_fragment as f64);
    features.insert(
        "xic_coverage".into(),
        scans_with_any_fragment as f64 / n_window_scans as f64,
    );

    // Extract column vectors (per-fragment XICs)
    let columns: Vec<Vec<f64>> = (0..n_frags)
        .map(|f| {
            (0..n_window_scans)
                .map(|s| xic_matrix[s * n_frags + f])
                .collect()
        })
        .collect();

    // Count fragments with non-zero signal
    let active_frags: Vec<usize> = columns
        .iter()
        .enumerate()
        .filter(|(_, col)| col.iter().any(|&v| v > 0.0))
        .map(|(i, _)| i)
        .collect();

    features.insert("xic_n_active_fragments".into(), active_frags.len() as f64);

    if active_frags.len() < 2 {
        features.insert("xic_best_coelution".into(), 0.0);
        features.insert("xic_mean_coelution".into(), 0.0);
        return features;
    }

    // Pairwise fragment XIC correlations (co-elution)
    let mut pairwise_corrs: Vec<f64> = Vec::new();
    for i in 0..active_frags.len() {
        for j in (i + 1)..active_frags.len() {
            let r = pearson_1d_impl(&columns[active_frags[i]], &columns[active_frags[j]]);
            if !r.is_nan() {
                pairwise_corrs.push(r);
            }
        }
    }

    if !pairwise_corrs.is_empty() {
        pairwise_corrs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        features.insert("xic_best_coelution".into(), pairwise_corrs[0]);
        features.insert(
            "xic_mean_coelution".into(),
            pairwise_corrs.iter().sum::<f64>() / pairwise_corrs.len() as f64,
        );
        let mid = pairwise_corrs.len() / 2;
        features.insert("xic_median_coelution".into(), pairwise_corrs[mid]);
        features.insert(
            "xic_n_coeluting".into(),
            pairwise_corrs.iter().filter(|&&r| r > 0.7).count() as f64,
        );
    } else {
        features.insert("xic_best_coelution".into(), 0.0);
        features.insert("xic_mean_coelution".into(), 0.0);
        features.insert("xic_median_coelution".into(), 0.0);
        features.insert("xic_n_coeluting".into(), 0.0);
    }

    // XIC vs prediction correlations
    // For each fragment: compute correlation of its XIC profile with prediction weight
    if target_predictions.len() == n_frags && !active_frags.is_empty() {
        // Build "predicted XIC" — the prediction value repeated for all scans
        // Correlation of actual XIC vs uniform doesn't make sense, so instead:
        // correlate the fragment's max XIC intensity vs predicted intensity
        let mut observed_maxes: Vec<f64> = Vec::new();
        let mut predicted_vals: Vec<f64> = Vec::new();

        for &f in &active_frags {
            let max_intensity = columns[f].iter().cloned().fold(0.0f64, f64::max);
            observed_maxes.push(max_intensity);
            predicted_vals.push(target_predictions[f]);
        }

        // Normalize both to [0, 1]
        let obs_max = observed_maxes.iter().cloned().fold(0.0f64, f64::max);
        let pred_max = predicted_vals.iter().cloned().fold(0.0f64, f64::max);
        if obs_max > 0.0 {
            for v in &mut observed_maxes {
                *v /= obs_max;
            }
        }
        if pred_max > 0.0 {
            for v in &mut predicted_vals {
                *v /= pred_max;
            }
        }

        let xic_vs_pred = pearson_1d_impl(&observed_maxes, &predicted_vals);
        features.insert(
            "xic_vs_prediction".into(),
            if xic_vs_pred.is_nan() { 0.0 } else { xic_vs_pred },
        );
    }

    // Best fragment: highest total XIC intensity
    let best_frag = active_frags
        .iter()
        .max_by(|&&a, &&b| {
            let sum_a: f64 = columns[a].iter().sum();
            let sum_b: f64 = columns[b].iter().sum();
            sum_a.partial_cmp(&sum_b).unwrap()
        })
        .copied()
        .unwrap_or(0);

    // Peak width (FWHM) of best fragment
    let best_xic = &columns[best_frag];
    let apex_val = best_xic.iter().cloned().fold(0.0f64, f64::max);
    if apex_val > 0.0 {
        let half_max = apex_val / 2.0;
        let above_half: usize = best_xic.iter().filter(|&&v| v >= half_max).count();
        // Approximate FWHM in scan units, convert to RT
        if n_window_scans > 1 {
            let rt_per_scan =
                (ms2_rts[scan_end - 1] - ms2_rts[scan_start]) / (n_window_scans - 1) as f64;
            features.insert("xic_peak_width".into(), above_half as f64 * rt_per_scan);
        }
        features.insert("xic_apex_intensity".into(), apex_val);
    }

    // Weighted AUC: sum of (XIC area × co-elution score) for active fragments
    let rts_window: Vec<f64> = ms2_rts[scan_start..scan_end].to_vec();
    let mut weighted_auc = 0.0;
    for &f in &active_frags {
        let xic = &columns[f];
        // Trapezoidal AUC
        let mut auc = 0.0;
        for i in 1..n_window_scans {
            auc += (rts_window[i] - rts_window[i - 1]) * (xic[i] + xic[i - 1]) / 2.0;
        }
        // Weight by mean co-elution with other fragments
        let mean_corr: f64 = active_frags
            .iter()
            .filter(|&&g| g != f)
            .map(|&g| {
                let r = pearson_1d_impl(&columns[f], &columns[g]);
                if r.is_nan() { 0.0 } else { r.abs() }
            })
            .sum::<f64>()
            / (active_frags.len() - 1).max(1) as f64;
        weighted_auc += auc * mean_corr;
    }
    features.insert(
        "xic_weighted_auc".into(),
        if weighted_auc > 0.0 {
            weighted_auc.ln()
        } else {
            0.0
        },
    );

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_ms2_data() -> (Vec<f64>, Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>) {
        // 5 scans at RT 1.0, 2.0, 3.0, 4.0, 5.0
        // Each scan has 3 peaks: 100.0, 200.0, 300.0 m/z
        let rts = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mz_offsets = vec![0u64, 3, 6, 9, 12];
        let mz_lengths = vec![3u64, 3, 3, 3, 3];
        let mz_flat = vec![
            100.0, 200.0, 300.0, // scan 0
            100.0, 200.0, 300.0, // scan 1
            100.0, 200.0, 300.0, // scan 2
            100.0, 200.0, 300.0, // scan 3
            100.0, 200.0, 300.0, // scan 4
        ];
        // Intensities: fragment at 100.0 has a peak at RT=3, fragment at 200.0 co-elutes
        let int_flat = vec![
            10.0, 5.0, 1.0, // scan 0
            50.0, 25.0, 1.0, // scan 1
            100.0, 50.0, 1.0, // scan 2 (apex)
            50.0, 25.0, 1.0, // scan 3
            10.0, 5.0, 1.0, // scan 4
        ];
        (rts, mz_offsets, mz_lengths, mz_flat, int_flat)
    }

    #[test]
    fn test_xic_basic() {
        let (rts, offsets, lengths, mz, ints) = make_simple_ms2_data();
        let target_mzs = vec![100.0, 200.0]; // two co-eluting fragments
        let predictions = vec![1.0, 0.5]; // predicted intensities

        let features = extract_xic_features_impl(
            &rts, &offsets, &lengths, &mz, &ints,
            &target_mzs, &predictions,
            0.5, 5.5, // RT window
            20.0,      // ppm (generous for test)
        );

        assert_eq!(features["xic_total_scans"], 5.0);
        assert_eq!(features["xic_n_detected_scans"], 5.0);
        assert_eq!(features["xic_coverage"], 1.0);
        // Fragments 100 and 200 are perfectly co-eluting (same profile shape)
        assert!(features["xic_best_coelution"] > 0.99);
    }

    #[test]
    fn test_xic_partial_coverage() {
        let (rts, offsets, lengths, mz, ints) = make_simple_ms2_data();
        let target_mzs = vec![100.0, 500.0]; // 500.0 doesn't exist in any scan

        let features = extract_xic_features_impl(
            &rts, &offsets, &lengths, &mz, &ints,
            &target_mzs, &[1.0, 0.5],
            0.5, 5.5, 20.0,
        );

        // Only fragment at 100.0 is detected, 500.0 is never found
        assert_eq!(features["xic_n_active_fragments"], 1.0);
    }

    #[test]
    fn test_xic_empty() {
        let features = extract_xic_features_impl(
            &[], &[], &[], &[], &[],
            &[100.0], &[1.0],
            0.0, 10.0, 20.0,
        );
        assert!(features.is_empty());
    }

    #[test]
    fn test_xic_rt_window() {
        let (rts, offsets, lengths, mz, ints) = make_simple_ms2_data();
        let target_mzs = vec![100.0, 200.0];

        // Narrow RT window: only scans 2-4
        let features = extract_xic_features_impl(
            &rts, &offsets, &lengths, &mz, &ints,
            &target_mzs, &[1.0, 0.5],
            1.5, 4.5, 20.0,
        );

        assert_eq!(features["xic_total_scans"], 3.0); // scans at RT 2, 3, 4
    }
}
