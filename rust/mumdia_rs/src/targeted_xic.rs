/// Targeted XIC (Extracted Ion Chromatogram) extraction from MS2 spectra.
///
/// For each peptidoform, extracts fragment ion intensities across ALL MS2 scans
/// in the elution window — not just the scans Sage assigned. This provides
/// complete elution profile data for co-elution scoring.
use std::collections::HashMap;

use crate::correlation::pearson_1d_impl;
use crate::mzml::MzMLData;
use rayon::prelude::*;

const MIN_CONSECUTIVE_FRAGMENT_SCANS: usize = 3;

#[derive(Clone)]
struct PreparedCandidate {
    idx: usize,
    precursor_mz: f64,
    rt_min: f64,
    rt_max: f64,
    target_mzs: Vec<f64>,
    target_predictions: Vec<f64>,
    b_count: f64,
    y_count: f64,
}

#[derive(Clone)]
struct IsolationWindowGroup {
    lower: f64,
    upper: f64,
    scan_indices: Vec<usize>,
}

fn flatten_ms2_data(
    mzml: &MzMLData,
) -> (
    Vec<f64>,
    Vec<u64>,
    Vec<u64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    let mut ms2_rts = Vec::with_capacity(mzml.ms2_spectra.len());
    let mut ms2_mz_offsets = Vec::with_capacity(mzml.ms2_spectra.len());
    let mut ms2_mz_lengths = Vec::with_capacity(mzml.ms2_spectra.len());
    let mut ms2_mz_flat = Vec::new();
    let mut ms2_int_flat = Vec::new();
    let mut ms2_iso_lower = Vec::with_capacity(mzml.ms2_spectra.len());
    let mut ms2_iso_upper = Vec::with_capacity(mzml.ms2_spectra.len());
    let mut offset = 0u64;

    for spectrum in &mzml.ms2_spectra {
        ms2_rts.push(spectrum.retention_time);
        ms2_mz_offsets.push(offset);
        ms2_mz_lengths.push(spectrum.mz.len() as u64);
        offset += spectrum.mz.len() as u64;
        ms2_mz_flat.extend_from_slice(&spectrum.mz);
        ms2_int_flat.extend_from_slice(&spectrum.intensity);
        let iso_target = spectrum.isolation_window_target.unwrap_or(0.0);
        let iso_lower = spectrum.isolation_window_lower.unwrap_or(0.0);
        let iso_upper = spectrum.isolation_window_upper.unwrap_or(0.0);
        ms2_iso_lower.push(iso_target - iso_lower);
        ms2_iso_upper.push(iso_target + iso_upper);
    }

    (
        ms2_rts,
        ms2_mz_offsets,
        ms2_mz_lengths,
        ms2_mz_flat,
        ms2_int_flat,
        ms2_iso_lower,
        ms2_iso_upper,
    )
}

fn isolation_window_key(lower: f64, upper: f64) -> (i64, i64) {
    (
        (lower * 1000.0).round() as i64,
        (upper * 1000.0).round() as i64,
    )
}

fn group_scans_by_isolation_window(
    ms2_iso_lower: &[f64],
    ms2_iso_upper: &[f64],
) -> Vec<IsolationWindowGroup> {
    let mut grouped: HashMap<(i64, i64), IsolationWindowGroup> = HashMap::new();
    for scan_idx in 0..ms2_iso_lower.len() {
        let lower = ms2_iso_lower[scan_idx];
        let upper = ms2_iso_upper[scan_idx];
        let key = isolation_window_key(lower, upper);
        grouped
            .entry(key)
            .and_modify(|group| group.scan_indices.push(scan_idx))
            .or_insert_with(|| IsolationWindowGroup {
                lower,
                upper,
                scan_indices: vec![scan_idx],
            });
    }

    let mut groups: Vec<IsolationWindowGroup> = grouped.into_values().collect();
    groups.sort_by(|a, b| {
        a.lower
            .partial_cmp(&b.lower)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    groups
}

fn longest_positive_run(values: &[f64]) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;

    for &value in values {
        if value > 0.0 {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }

    best
}

fn extract_xic_features_for_scan_subset(
    ms2_rts: &[f64],
    ms2_mz_offsets: &[u64],
    ms2_mz_lengths: &[u64],
    ms2_mz_flat: &[f64],
    ms2_int_flat: &[f64],
    scan_indices: &[usize],
    target_mzs: &[f64],
    target_predictions: &[f64],
    rt_min: f64,
    rt_max: f64,
    ppm_tolerance: f64,
) -> HashMap<String, f64> {
    let n_frags = target_mzs.len();
    if n_frags == 0 || scan_indices.is_empty() {
        return HashMap::new();
    }

    let matching_scans: Vec<usize> = scan_indices
        .iter()
        .copied()
        .filter(|&scan_idx| {
            let rt = ms2_rts[scan_idx];
            rt_min <= rt && rt <= rt_max
        })
        .collect();
    let n_matching_scans = matching_scans.len();

    let mut features = HashMap::with_capacity(24);
    features.insert("xic_total_scans".into(), n_matching_scans as f64);
    features.insert("xic_matching_window_scans".into(), n_matching_scans as f64);

    if n_matching_scans < 2 {
        features.insert("xic_coverage".into(), 0.0);
        features.insert("xic_n_detected_scans".into(), n_matching_scans as f64);
        return features;
    }

    let mut xic_matrix = vec![0.0f64; n_matching_scans * n_frags];
    let mut scans_with_any_fragment = 0u32;

    for (scan_idx_in_window, &scan_idx) in matching_scans.iter().enumerate() {
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
            let pos = scan_mz.partition_point(|&mz| mz < target_mz - tol);
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

    features.insert("xic_n_detected_scans".into(), scans_with_any_fragment as f64);
    features.insert(
        "xic_coverage".into(),
        scans_with_any_fragment as f64 / n_matching_scans as f64,
    );

    let columns: Vec<Vec<f64>> = (0..n_frags)
        .map(|f| {
            (0..n_matching_scans)
                .map(|s| xic_matrix[s * n_frags + f])
                .collect()
        })
        .collect();

    let max_consecutive_fragment_scans = columns
        .iter()
        .map(|column| longest_positive_run(column))
        .max()
        .unwrap_or(0);
    let fragments_with_min_consecutive = columns
        .iter()
        .filter(|column| longest_positive_run(column) >= MIN_CONSECUTIVE_FRAGMENT_SCANS)
        .count();

    features.insert(
        "xic_max_consecutive_fragment_scans".into(),
        max_consecutive_fragment_scans as f64,
    );
    features.insert(
        "xic_n_fragments_with_min_consecutive".into(),
        fragments_with_min_consecutive as f64,
    );
    features.insert(
        "xic_min_consecutive_scans_required".into(),
        MIN_CONSECUTIVE_FRAGMENT_SCANS as f64,
    );

    let active_frags: Vec<usize> = columns
        .iter()
        .enumerate()
        .filter(|(_, col)| col.iter().any(|&v| v > 0.0))
        .map(|(i, _)| i)
        .collect();
    features.insert("xic_n_active_fragments".into(), active_frags.len() as f64);

    {
        let frags_per_scan: Vec<f64> = (0..n_matching_scans)
            .map(|s| {
                (0..n_frags)
                    .filter(|&f| xic_matrix[s * n_frags + f] > 0.0)
                    .count() as f64
            })
            .collect();
        let mean_frags: f64 = frags_per_scan.iter().sum::<f64>() / n_matching_scans as f64;
        let max_frags = frags_per_scan.iter().cloned().fold(0.0f64, f64::max);
        features.insert("xic_mean_frags_per_scan".into(), mean_frags);
        features.insert("xic_max_frags_per_scan".into(), max_frags);
        let multi_frag_scans = frags_per_scan.iter().filter(|&&v| v >= 2.0).count();
        features.insert(
            "xic_multi_frag_coverage".into(),
            multi_frag_scans as f64 / n_matching_scans as f64,
        );
    }

    if active_frags.len() < 2 || scans_with_any_fragment < 3 {
        features.insert("xic_best_coelution".into(), 0.0);
        features.insert("xic_mean_coelution".into(), 0.0);
        features.insert("xic_median_coelution".into(), 0.0);
        features.insert("xic_n_coeluting".into(), 0.0);
        return features;
    }

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
        features.insert("xic_median_coelution".into(), pairwise_corrs[pairwise_corrs.len() / 2]);
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

    if target_predictions.len() == n_frags && !active_frags.is_empty() {
        let mut observed_maxes: Vec<f64> = Vec::new();
        let mut predicted_vals: Vec<f64> = Vec::new();
        for &f in &active_frags {
            observed_maxes.push(columns[f].iter().cloned().fold(0.0f64, f64::max));
            predicted_vals.push(target_predictions[f]);
        }
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
        features.insert("xic_vs_prediction".into(), if xic_vs_pred.is_nan() { 0.0 } else { xic_vs_pred });

        let apex_scan = (0..n_matching_scans)
            .max_by(|&a, &b| {
                let sum_a: f64 = (0..n_frags).map(|f| xic_matrix[a * n_frags + f]).sum();
                let sum_b: f64 = (0..n_frags).map(|f| xic_matrix[b * n_frags + f]).sum();
                sum_a.partial_cmp(&sum_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        let mut apex_observed: Vec<f64> = (0..n_frags)
            .map(|f| xic_matrix[apex_scan * n_frags + f])
            .collect();
        let mut apex_predicted: Vec<f64> = target_predictions.to_vec();
        let apex_obs_max = apex_observed.iter().cloned().fold(0.0f64, f64::max);
        let apex_pred_max = apex_predicted.iter().cloned().fold(0.0f64, f64::max);
        if apex_obs_max > 0.0 {
            for v in &mut apex_observed {
                *v /= apex_obs_max;
            }
        }
        if apex_pred_max > 0.0 {
            for v in &mut apex_predicted {
                *v /= apex_pred_max;
            }
        }
        let apex_corr = pearson_1d_impl(&apex_observed, &apex_predicted);
        features.insert("xic_apex_spectrum_corr".into(), if apex_corr.is_nan() { 0.0 } else { apex_corr });
    }

    let best_frag = active_frags
        .iter()
        .max_by(|&&a, &&b| {
            let sum_a: f64 = columns[a].iter().sum();
            let sum_b: f64 = columns[b].iter().sum();
            sum_a.partial_cmp(&sum_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
        .unwrap_or(0);
    let best_xic = &columns[best_frag];
    let apex_val = best_xic.iter().cloned().fold(0.0f64, f64::max);
    if apex_val > 0.0 {
        let apex_idx = best_xic
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        features.insert("xic_apex_rt".into(), ms2_rts[matching_scans[apex_idx]]);
    }

    let detected_scan_indices: Vec<usize> = (0..n_matching_scans)
        .filter(|&scan_idx| (0..n_frags).any(|frag_idx| xic_matrix[scan_idx * n_frags + frag_idx] > 0.0))
        .collect();
    if let Some(first_idx) = detected_scan_indices.first() {
        features.insert("xic_detected_rt_start".into(), ms2_rts[matching_scans[*first_idx]]);
    }
    if let Some(last_idx) = detected_scan_indices.last() {
        features.insert("xic_detected_rt_end".into(), ms2_rts[matching_scans[*last_idx]]);
    }
    if apex_val > 0.0 && n_matching_scans > 1 {
        let half_max = apex_val / 2.0;
        let above_half: usize = best_xic.iter().filter(|&&v| v >= half_max).count();
        let first_rt = ms2_rts[matching_scans[0]];
        let last_rt = ms2_rts[matching_scans[n_matching_scans - 1]];
        let rt_per_scan = (last_rt - first_rt) / (n_matching_scans - 1) as f64;
        features.insert("xic_peak_width".into(), above_half as f64 * rt_per_scan);
        features.insert("xic_apex_intensity".into(), apex_val);
    } else if apex_val > 0.0 {
        features.insert("xic_apex_intensity".into(), apex_val);
    }

    let matching_rts: Vec<f64> = matching_scans.iter().map(|&s| ms2_rts[s]).collect();
    let mut weighted_auc = 0.0;
    for &f in &active_frags {
        let xic = &columns[f];
        let mut auc = 0.0;
        for i in 1..n_matching_scans {
            auc += (matching_rts[i] - matching_rts[i - 1]) * (xic[i] + xic[i - 1]) / 2.0;
        }
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
    features.insert("xic_weighted_auc".into(), if weighted_auc > 0.0 { weighted_auc.ln() } else { 0.0 });

    features
}

/// DIA-NN's exact smoothing: 3-point weighted average.
/// dst[i] = 0.25 * src[i-1] + 0.5 * src[i] + 0.25 * src[i+1]
fn smooth_profile(data: &[f64], _window: usize) -> Vec<f64> {
    let n = data.len();
    if n <= 2 {
        return data.to_vec();
    }
    let mut out = vec![0.0; n];
    out[0] = data[0];
    out[n - 1] = data[n - 1];
    for i in 1..n - 1 {
        out[i] = 0.25 * data[i - 1] + 0.5 * data[i] + 0.25 * data[i + 1];
    }
    out
}

/// Extract XIC features for a single peptidoform.
///
/// Searches MS2 scans in the RT window whose isolation window covers the
/// precursor m/z, then extracts fragment ion intensities for co-elution scoring.
///
/// # Arguments
/// * `ms2_rts` - Sorted retention times for all MS2 scans
/// * `ms2_mz_offsets` - Start index in flat mz array for each scan
/// * `ms2_mz_lengths` - Number of peaks per scan
/// * `ms2_mz_flat` - All m/z values concatenated (sorted within each scan)
/// * `ms2_int_flat` - All intensity values concatenated
/// * `ms2_iso_lower` - Isolation window lower bound (target - lower_offset) per scan
/// * `ms2_iso_upper` - Isolation window upper bound (target + upper_offset) per scan
/// * `target_mzs` - Fragment m/z values to search for
/// * `target_predictions` - MS2PIP predicted intensities (for correlation)
/// * `precursor_mz` - Precursor m/z of this peptidoform (for isolation window filtering)
/// * `rt_min`, `rt_max` - RT window bounds
/// * `ppm_tolerance` - Mass tolerance in ppm
pub fn extract_xic_features_impl(
    ms2_rts: &[f64],
    ms2_mz_offsets: &[u64],
    ms2_mz_lengths: &[u64],
    ms2_mz_flat: &[f64],
    ms2_int_flat: &[f64],
    ms2_iso_lower: &[f64],
    ms2_iso_upper: &[f64],
    target_mzs: &[f64],
    target_predictions: &[f64],
    precursor_mz: f64,
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

    // Filter to scans whose isolation window covers the precursor m/z
    let matching_scans: Vec<usize> = (scan_start..scan_end)
        .filter(|&idx| ms2_iso_lower[idx] <= precursor_mz && precursor_mz <= ms2_iso_upper[idx])
        .collect();
    let n_matching_scans = matching_scans.len();

    let mut features = HashMap::with_capacity(20);
    features.insert("xic_total_scans".into(), (scan_end - scan_start) as f64);
    features.insert("xic_matching_window_scans".into(), n_matching_scans as f64);

    if n_matching_scans < 2 {
        features.insert("xic_coverage".into(), 0.0);
        features.insert("xic_n_detected_scans".into(), n_matching_scans as f64);
        return features;
    }

    // Build XIC matrix: (n_matching_scans × n_frags), 0.0 for not detected
    let mut xic_matrix = vec![0.0f64; n_matching_scans * n_frags];
    let mut scans_with_any_fragment = 0u32;

    for (scan_idx_in_window, &scan_idx) in matching_scans.iter().enumerate() {
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
    // Scans are already filtered to matching isolation windows.
    // Now compute coverage: how many of these matching scans had any fragment detected.

    features.insert("xic_n_detected_scans".into(), scans_with_any_fragment as f64);
    features.insert(
        "xic_coverage".into(),
        scans_with_any_fragment as f64 / n_matching_scans as f64,
    );

    // Extract per-fragment column vectors from the XIC matrix
    let columns: Vec<Vec<f64>> = (0..n_frags)
        .map(|f| {
            (0..n_matching_scans)
                .map(|s| xic_matrix[s * n_frags + f])
                .collect()
        })
        .collect();

    let max_consecutive_fragment_scans = columns
        .iter()
        .map(|column| longest_positive_run(column))
        .max()
        .unwrap_or(0);
    let fragments_with_min_consecutive = columns
        .iter()
        .filter(|column| longest_positive_run(column) >= MIN_CONSECUTIVE_FRAGMENT_SCANS)
        .count();

    features.insert(
        "xic_max_consecutive_fragment_scans".into(),
        max_consecutive_fragment_scans as f64,
    );
    features.insert(
        "xic_n_fragments_with_min_consecutive".into(),
        fragments_with_min_consecutive as f64,
    );
    features.insert(
        "xic_min_consecutive_scans_required".into(),
        MIN_CONSECUTIVE_FRAGMENT_SCANS as f64,
    );

    // Count fragments with non-zero signal
    let active_frags: Vec<usize> = columns
        .iter()
        .enumerate()
        .filter(|(_, col)| col.iter().any(|&v| v > 0.0))
        .map(|(i, _)| i)
        .collect();

    features.insert("xic_n_active_fragments".into(), active_frags.len() as f64);

    // Per-scan fragment count statistics
    {
        let frags_per_scan: Vec<f64> = (0..n_matching_scans)
            .map(|s| {
                (0..n_frags)
                    .filter(|&f| xic_matrix[s * n_frags + f] > 0.0)
                    .count() as f64
            })
            .collect();
        let mean_frags: f64 = frags_per_scan.iter().sum::<f64>() / n_matching_scans as f64;
        let max_frags = frags_per_scan.iter().cloned().fold(0.0f64, f64::max);
        features.insert("xic_mean_frags_per_scan".into(), mean_frags);
        features.insert("xic_max_frags_per_scan".into(), max_frags);
        let multi_frag_scans = frags_per_scan.iter().filter(|&&v| v >= 2.0).count();
        features.insert(
            "xic_multi_frag_coverage".into(),
            multi_frag_scans as f64 / n_matching_scans as f64,
        );
    }

    if active_frags.len() < 2 || scans_with_any_fragment < 3 {
        features.insert("xic_best_coelution".into(), 0.0);
        features.insert("xic_mean_coelution".into(), 0.0);
        features.insert("xic_median_coelution".into(), 0.0);
        features.insert("xic_n_coeluting".into(), 0.0);
        return features;
    }

    // Pairwise fragment XIC correlations (co-elution)
    let mut pairwise_corrs: Vec<f64> = Vec::new();
    for i in 0..active_frags.len() {
        for j in (i + 1)..active_frags.len() {
            let r = pearson_1d_impl(
                &columns[active_frags[i]],
                &columns[active_frags[j]],
            );
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

    // XIC vs prediction correlations (using ALL theoretical fragments, not just Sage-matched)
    if target_predictions.len() == n_frags && !active_frags.is_empty() {
        // 1. Max-intensity correlation: for each fragment, take its max XIC intensity
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

        // 2. Apex-scan correlation: at the scan with highest total intensity,
        //    correlate ALL fragment intensities against MS2PIP predictions.
        //    This uses all fragments (not just Sage-matched) for a full spectrum match.
        let apex_scan = (0..n_matching_scans)
            .max_by(|&a, &b| {
                let sum_a: f64 = (0..n_frags).map(|f| xic_matrix[a * n_frags + f]).sum();
                let sum_b: f64 = (0..n_frags).map(|f| xic_matrix[b * n_frags + f]).sum();
                sum_a.partial_cmp(&sum_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        // Extract observed spectrum at apex (all fragments, including zeros)
        let mut apex_observed: Vec<f64> = (0..n_frags)
            .map(|f| xic_matrix[apex_scan * n_frags + f])
            .collect();
        let mut apex_predicted: Vec<f64> = target_predictions.to_vec();

        // Normalize
        let apex_obs_max = apex_observed.iter().cloned().fold(0.0f64, f64::max);
        let apex_pred_max = apex_predicted.iter().cloned().fold(0.0f64, f64::max);
        if apex_obs_max > 0.0 {
            for v in &mut apex_observed {
                *v /= apex_obs_max;
            }
        }
        if apex_pred_max > 0.0 {
            for v in &mut apex_predicted {
                *v /= apex_pred_max;
            }
        }

        // Full spectrum correlation (all fragments including non-detected = 0)
        let apex_corr = pearson_1d_impl(&apex_observed, &apex_predicted);
        features.insert(
            "xic_apex_spectrum_corr".into(),
            if apex_corr.is_nan() { 0.0 } else { apex_corr },
        );

        // Also compute using only detected fragments at apex (non-zero pairs)
        let mut apex_obs_nz: Vec<f64> = Vec::new();
        let mut apex_pred_nz: Vec<f64> = Vec::new();
        for f in 0..n_frags {
            if apex_observed[f] > 0.0 {
                apex_obs_nz.push(apex_observed[f]);
                apex_pred_nz.push(apex_predicted[f]);
            }
        }
        if apex_obs_nz.len() >= 3 {
            let apex_corr_nz = pearson_1d_impl(&apex_obs_nz, &apex_pred_nz);
            features.insert(
                "xic_apex_spectrum_corr_detected".into(),
                if apex_corr_nz.is_nan() { 0.0 } else { apex_corr_nz },
            );
            features.insert("xic_apex_n_detected".into(), apex_obs_nz.len() as f64);
        }

        // 3. Mean spectrum correlation across all detected scans
        let mut scan_corrs: Vec<f64> = Vec::new();
        for s in 0..n_matching_scans {
            let mut obs: Vec<f64> = Vec::new();
            let mut pred: Vec<f64> = Vec::new();
            for f in 0..n_frags {
                let v = xic_matrix[s * n_frags + f];
                if v > 0.0 {
                    obs.push(v);
                    pred.push(target_predictions[f]);
                }
            }
            if obs.len() >= 3 {
                // Normalize obs
                let omax = obs.iter().cloned().fold(0.0f64, f64::max);
                if omax > 0.0 {
                    for v in &mut obs {
                        *v /= omax;
                    }
                }
                let r = pearson_1d_impl(&obs, &pred);
                if !r.is_nan() {
                    scan_corrs.push(r);
                }
            }
        }
        if !scan_corrs.is_empty() {
            scan_corrs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            features.insert("xic_best_spectrum_corr".into(), scan_corrs[0]);
            features.insert(
                "xic_mean_spectrum_corr".into(),
                scan_corrs.iter().sum::<f64>() / scan_corrs.len() as f64,
            );
            let mid = scan_corrs.len() / 2;
            features.insert("xic_median_spectrum_corr".into(), scan_corrs[mid]);
        }
    }

    // 4. DIA-NN-style per-fragment elution profile correlations with iterative pruning.
    //    Smooth XICs, then iteratively remove the worst-correlating fragment until
    //    only 3 remain. At each step, rebuild the reference from the top-3 most
    //    intense surviving fragments. Track the best 3 correlations seen.
    if active_frags.len() >= 2 && n_matching_scans >= 5 {
        let smooth_window = 3.min(n_matching_scans);
        let smoothed: Vec<Vec<f64>> = columns
            .iter()
            .map(|col| smooth_profile(col, smooth_window))
            .collect();

        // Rank fragments by total smoothed intensity
        let mut frag_sums: Vec<(usize, f64)> = active_frags
            .iter()
            .map(|&f| (f, smoothed[f].iter().sum::<f64>()))
            .collect();
        frag_sums.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // DIA-NN method: reference = smoothed XIC of the single best (most intense) fragment
        let best_frag_idx = frag_sums[0].0;
        let reference = &smoothed[best_frag_idx];

        // Correlate each fragment's RAW XIC against the smoothed best-fragment reference
        // DIA-NN does: corr(raw_fragment_xic, smoothed_best_fragment_xic)
        // Including the best fragment itself (raw vs smoothed ≈ high but not 1.0)
        let mut ref_corrs: Vec<f64> = Vec::new();
        for &(f, _) in &frag_sums {
            let r = pearson_1d_impl(&columns[f], reference);
            if !r.is_nan() {
                ref_corrs.push(r);
            }
        }

        if !ref_corrs.is_empty() {
            ref_corrs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            features.insert("xic_ref_corr_best".into(), ref_corrs[0]);
            features.insert(
                "xic_ref_corr_mean".into(),
                ref_corrs.iter().sum::<f64>() / ref_corrs.len() as f64,
            );
            let mid = ref_corrs.len() / 2;
            features.insert("xic_ref_corr_median".into(), ref_corrs[mid]);
            features.insert("xic_n_ref_corr_good".into(),
                ref_corrs.iter().filter(|&&r| r > 0.7).count() as f64);
            // Top-N individual correlations
            for (i, &r) in ref_corrs.iter().take(12).enumerate() {
                features.insert(format!("xic_ref_corr_{}", i), r);
            }
        }
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
        let apex_idx = best_xic
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        features.insert("xic_apex_rt".into(), ms2_rts[matching_scans[apex_idx]]);
    }

    let detected_scan_indices: Vec<usize> = (0..n_matching_scans)
        .filter(|&scan_idx| (0..n_frags).any(|frag_idx| xic_matrix[scan_idx * n_frags + frag_idx] > 0.0))
        .collect();
    if let Some(first_idx) = detected_scan_indices.first() {
        features.insert("xic_detected_rt_start".into(), ms2_rts[matching_scans[*first_idx]]);
    }
    if let Some(last_idx) = detected_scan_indices.last() {
        features.insert("xic_detected_rt_end".into(), ms2_rts[matching_scans[*last_idx]]);
    }

    if apex_val > 0.0 && n_matching_scans > 1 {
        let half_max = apex_val / 2.0;
        let above_half: usize = best_xic.iter().filter(|&&v| v >= half_max).count();
        let first_rt = ms2_rts[matching_scans[0]];
        let last_rt = ms2_rts[matching_scans[n_matching_scans - 1]];
        let rt_per_scan = (last_rt - first_rt) / (n_matching_scans - 1) as f64;
        features.insert("xic_peak_width".into(), above_half as f64 * rt_per_scan);
        features.insert("xic_apex_intensity".into(), apex_val);
    } else if apex_val > 0.0 {
        features.insert("xic_apex_intensity".into(), apex_val);
    }

    // Weighted AUC using matching scans
    let matching_rts: Vec<f64> = matching_scans.iter().map(|&s| ms2_rts[s]).collect();
    let mut weighted_auc = 0.0;
    for &f in &active_frags {
        let xic = &columns[f];
        // Trapezoidal AUC
        let mut auc = 0.0;
        for i in 1..n_matching_scans {
            auc += (matching_rts[i] - matching_rts[i - 1]) * (xic[i] + xic[i - 1]) / 2.0;
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


pub fn search_partition_chromatograms_impl(
    mzml: &MzMLData,
    peptides: &[String],
    charges: &[u8],
    precursor_mzs: &[f64],
    rt_mins: &[f64],
    rt_maxs: &[f64],
    predicted_fragment_mzs: &[f64],
    predicted_fragment_mz_offsets: &[u64],
    predicted_fragment_mz_lengths: &[u64],
    predicted_fragment_names: &[String],
    predicted_fragment_name_offsets: &[u64],
    predicted_fragment_name_lengths: &[u64],
    predicted_fragment_weights: &[f64],
    predicted_fragment_weight_offsets: &[u64],
    predicted_fragment_weight_lengths: &[u64],
    top_n: usize,
    ppm_tolerance: f64,
) -> Vec<HashMap<String, f64>> {
    let _ = peptides;

    let (
        ms2_rts,
        ms2_mz_offsets,
        ms2_mz_lengths,
        ms2_mz_flat,
        ms2_int_flat,
        ms2_iso_lower,
        ms2_iso_upper,
    ) = flatten_ms2_data(mzml);

    let prepared_candidates: Vec<PreparedCandidate> = (0..charges.len())
        .filter_map(|idx| {
            if charges[idx] == 0 {
                return None;
            }
            let precursor_mz = precursor_mzs[idx];
            if !precursor_mz.is_finite() || precursor_mz <= 0.0 {
                return None;
            }
            let mz_offset = predicted_fragment_mz_offsets[idx] as usize;
            let mz_length = predicted_fragment_mz_lengths[idx] as usize;
            let name_offset = predicted_fragment_name_offsets[idx] as usize;
            let name_length = predicted_fragment_name_lengths[idx] as usize;
            let weight_offset = predicted_fragment_weight_offsets[idx] as usize;
            let weight_length = predicted_fragment_weight_lengths[idx] as usize;
            if mz_length == 0 || name_length == 0 || weight_length == 0 {
                return None;
            }
            if mz_length != name_length || mz_length != weight_length {
                return None;
            }
            let mzs_slice = &predicted_fragment_mzs[mz_offset..mz_offset + mz_length];
            let names_slice = &predicted_fragment_names[name_offset..name_offset + name_length];
            let weights_slice =
                &predicted_fragment_weights[weight_offset..weight_offset + weight_length];
            let mut ranked: Vec<(f64, String, f64)> = mzs_slice
                .iter()
                .copied()
                .zip(names_slice.iter().cloned())
                .zip(weights_slice.iter().copied())
                .map(|((mz, name), weight)| (mz, name, weight))
                .collect();
            ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            ranked.truncate(top_n);

            let mut target_mzs = Vec::new();
            let mut target_predictions = Vec::new();
            let mut b_count = 0.0;
            let mut y_count = 0.0;
            for (mz, fragment_name, predicted_weight) in ranked {
                if fragment_name.starts_with('b') {
                    b_count += 1.0;
                } else if fragment_name.starts_with('y') {
                    y_count += 1.0;
                }
                target_mzs.push(mz);
                target_predictions.push(predicted_weight);
            }
            if target_mzs.is_empty() {
                return None;
            }
            Some(PreparedCandidate {
                idx,
                precursor_mz,
                rt_min: rt_mins[idx],
                rt_max: rt_maxs[idx],
                target_mzs,
                target_predictions,
                b_count,
                y_count,
            })
        })
        .collect();

    let window_groups = group_scans_by_isolation_window(&ms2_iso_lower, &ms2_iso_upper);
    eprintln!(
        "[mumdia_rs] Rust XIC search (indexed): {} isolation-window groups, {} prepared candidates, {} MS2 scans, {} rayon threads",
        window_groups.len(),
        prepared_candidates.len(),
        mzml.ms2_spectra.len(),
        rayon::current_num_threads()
    );

    window_groups
        .into_par_iter()
        .flat_map_iter(|window_group| {
            let candidates_for_window: Vec<&PreparedCandidate> = prepared_candidates
                .iter()
                .filter(|candidate| {
                    window_group.lower <= candidate.precursor_mz
                        && candidate.precursor_mz <= window_group.upper
                })
                .collect();
            if candidates_for_window.is_empty() {
                return Vec::new().into_iter();
            }

            eprintln!(
                "[mumdia_rs] Rust XIC window {:.4}-{:.4}: {} scans, {} candidates",
                window_group.lower,
                window_group.upper,
                window_group.scan_indices.len(),
                candidates_for_window.len()
            );

            #[derive(Copy, Clone)]
            struct FragEntry {
                cand_in_window: u32,
                fragment_idx: u16,
                target_mz: f64,
            }

            let n_cands = candidates_for_window.len();
            let max_frags = candidates_for_window
                .iter()
                .map(|c| c.target_mzs.len())
                .max()
                .unwrap_or(0);
            if max_frags == 0 {
                return Vec::new().into_iter();
            }

            let mut total_entries = 0usize;
            let mut min_mz = f64::INFINITY;
            let mut max_mz = 0.0f64;
            for c in &candidates_for_window {
                total_entries += c.target_mzs.len();
                for &m in &c.target_mzs {
                    if m < min_mz {
                        min_mz = m;
                    }
                    if m > max_mz {
                        max_mz = m;
                    }
                }
            }
            if total_entries == 0 || !min_mz.is_finite() {
                return Vec::new().into_iter();
            }

            let max_tol_da = max_mz * ppm_tolerance * 1e-6;
            let bucket_width = (max_tol_da * 2.0).max(1e-4);
            let inv_bucket_width = 1.0 / bucket_width;
            let n_buckets = ((max_mz - min_mz) * inv_bucket_width).ceil() as usize + 1;

            let mut counts = vec![0u32; n_buckets];
            for c in &candidates_for_window {
                for &m in &c.target_mzs {
                    let b = ((m - min_mz) * inv_bucket_width) as usize;
                    counts[b] += 1;
                }
            }
            let mut bucket_offsets = vec![0u32; n_buckets + 1];
            for i in 0..n_buckets {
                bucket_offsets[i + 1] = bucket_offsets[i] + counts[i];
            }
            let mut entries = vec![
                FragEntry {
                    cand_in_window: 0,
                    fragment_idx: 0,
                    target_mz: 0.0,
                };
                total_entries
            ];
            let mut cursors = bucket_offsets[..n_buckets].to_vec();
            for (ci, c) in candidates_for_window.iter().enumerate() {
                for (fi, &m) in c.target_mzs.iter().enumerate() {
                    let b = ((m - min_mz) * inv_bucket_width) as usize;
                    let pos = cursors[b] as usize;
                    entries[pos] = FragEntry {
                        cand_in_window: ci as u32,
                        fragment_idx: fi as u16,
                        target_mz: m,
                    };
                    cursors[b] += 1;
                }
            }
            for b in 0..n_buckets {
                let lo = bucket_offsets[b] as usize;
                let hi = bucket_offsets[b + 1] as usize;
                entries[lo..hi].sort_unstable_by(|a, b| {
                    a.target_mz.partial_cmp(&b.target_mz).unwrap()
                });
            }

            let n_window_scans = window_group.scan_indices.len();

            let mut cand_scan_rows: Vec<Vec<i32>> = Vec::with_capacity(n_cands);
            let mut cand_n_scans: Vec<usize> = Vec::with_capacity(n_cands);
            let mut cand_matrix_offsets: Vec<usize> = Vec::with_capacity(n_cands + 1);
            cand_matrix_offsets.push(0);
            let mut total_cells = 0usize;

            for c in &candidates_for_window {
                let n_frags = c.target_mzs.len();
                let mut row_map = vec![-1i32; n_window_scans];
                let mut row = 0i32;
                for (wi, &scan_idx) in window_group.scan_indices.iter().enumerate() {
                    let rt = ms2_rts[scan_idx];
                    if rt >= c.rt_min && rt <= c.rt_max {
                        row_map[wi] = row;
                        row += 1;
                    }
                }
                let n_scans_c = row as usize;
                cand_scan_rows.push(row_map);
                cand_n_scans.push(n_scans_c);
                total_cells += n_scans_c * n_frags;
                cand_matrix_offsets.push(total_cells);
            }

            let mut xic_flat = vec![0.0f64; total_cells];

            for (wi, &scan_idx) in window_group.scan_indices.iter().enumerate() {
                let offset = ms2_mz_offsets[scan_idx] as usize;
                let length = ms2_mz_lengths[scan_idx] as usize;
                if length == 0 {
                    continue;
                }
                let scan_mz = &ms2_mz_flat[offset..offset + length];
                let scan_int = &ms2_int_flat[offset..offset + length];

                for (pi, &peak_mz) in scan_mz.iter().enumerate() {
                    let intensity = scan_int[pi];
                    if intensity <= 0.0 {
                        continue;
                    }
                    let tol_da = peak_mz * ppm_tolerance * 1e-6;
                    let lo_mz = peak_mz - tol_da;
                    let hi_mz = peak_mz + tol_da;
                    let lo_f = (lo_mz - min_mz) * inv_bucket_width;
                    let hi_f = (hi_mz - min_mz) * inv_bucket_width;
                    if hi_f < 0.0 {
                        continue;
                    }
                    let b_lo = lo_f.max(0.0) as usize;
                    let b_hi = (hi_f as usize).min(n_buckets.saturating_sub(1));
                    if b_lo > b_hi {
                        continue;
                    }
                    for bi in b_lo..=b_hi {
                        let start = bucket_offsets[bi] as usize;
                        let end = bucket_offsets[bi + 1] as usize;
                        for entry in &entries[start..end] {
                            if entry.target_mz > hi_mz {
                                break;
                            }
                            if entry.target_mz >= lo_mz {
                                let ci = entry.cand_in_window as usize;
                                let row = cand_scan_rows[ci][wi];
                                if row < 0 {
                                    continue;
                                }
                                let n_frags = candidates_for_window[ci].target_mzs.len();
                                let base = cand_matrix_offsets[ci];
                                let cell =
                                    base + (row as usize) * n_frags + entry.fragment_idx as usize;
                                if intensity > xic_flat[cell] {
                                    xic_flat[cell] = intensity;
                                }
                            }
                        }
                    }
                }
            }

            let mut out: Vec<HashMap<String, f64>> = Vec::with_capacity(n_cands);

            for (ci, candidate) in candidates_for_window.iter().enumerate() {
                let n_frags = candidate.target_mzs.len();
                let n_scans_c = cand_n_scans[ci];
                let base = cand_matrix_offsets[ci];
                let xic = &xic_flat[base..base + n_scans_c * n_frags];

                let matching_scans_rts: Vec<f64> = window_group
                    .scan_indices
                    .iter()
                    .filter_map(|&s| {
                        let rt = ms2_rts[s];
                        if rt >= candidate.rt_min && rt <= candidate.rt_max {
                            Some(rt)
                        } else {
                            None
                        }
                    })
                    .collect();
                debug_assert_eq!(matching_scans_rts.len(), n_scans_c);

                let features = compute_features_from_xic(
                    xic,
                    n_scans_c,
                    n_frags,
                    &matching_scans_rts,
                    &candidate.target_predictions,
                );

                let n_detected = features.get("xic_n_detected_scans").copied().unwrap_or(0.0);
                let max_consec = features
                    .get("xic_max_consecutive_fragment_scans")
                    .copied()
                    .unwrap_or(0.0);
                if n_detected <= 0.0 || max_consec < MIN_CONSECUTIVE_FRAGMENT_SCANS as f64 {
                    continue;
                }

                let mut features = features;
                features.insert("candidate_idx".into(), candidate.idx as f64);
                features.insert("precursor_mz".into(), candidate.precursor_mz);
                features.insert("matched_top_fragments".into(), n_frags as f64);
                features.insert("matched_b_fragments".into(), candidate.b_count);
                features.insert("matched_y_fragments".into(), candidate.y_count);
                features.insert("isolation_lower".into(), window_group.lower);
                features.insert("isolation_upper".into(), window_group.upper);
                out.push(features);
            }

            out.into_iter()
        })
        .collect()
}

fn compute_features_from_xic(
    xic: &[f64],
    n_scans: usize,
    n_frags: usize,
    matching_rts: &[f64],
    target_predictions: &[f64],
) -> HashMap<String, f64> {
    let mut features: HashMap<String, f64> = HashMap::with_capacity(24);
    features.insert("xic_total_scans".into(), n_scans as f64);
    features.insert("xic_matching_window_scans".into(), n_scans as f64);

    if n_scans < 2 || n_frags == 0 {
        features.insert("xic_coverage".into(), 0.0);
        features.insert("xic_n_detected_scans".into(), n_scans as f64);
        return features;
    }

    let mut scans_with_any_fragment = 0u32;
    let mut multi_frag_scans = 0u32;
    let mut frag_count_sum = 0u64;
    let mut frag_count_max = 0u32;
    for s in 0..n_scans {
        let mut c = 0u32;
        for f in 0..n_frags {
            if xic[s * n_frags + f] > 0.0 {
                c += 1;
            }
        }
        if c > 0 {
            scans_with_any_fragment += 1;
        }
        if c >= 2 {
            multi_frag_scans += 1;
        }
        frag_count_sum += c as u64;
        if c > frag_count_max {
            frag_count_max = c;
        }
    }
    features.insert("xic_n_detected_scans".into(), scans_with_any_fragment as f64);
    features.insert(
        "xic_coverage".into(),
        scans_with_any_fragment as f64 / n_scans as f64,
    );

    struct ColStats {
        sum: f64,
        max: f64,
        any: bool,
        longest_run: usize,
    }
    let mut col_stats: Vec<ColStats> = Vec::with_capacity(n_frags);
    for f in 0..n_frags {
        let mut sum = 0.0;
        let mut mx = 0.0;
        let mut any = false;
        let mut best = 0usize;
        let mut cur = 0usize;
        for s in 0..n_scans {
            let v = xic[s * n_frags + f];
            if v > 0.0 {
                sum += v;
                any = true;
                if v > mx {
                    mx = v;
                }
                cur += 1;
                if cur > best {
                    best = cur;
                }
            } else {
                cur = 0;
            }
        }
        col_stats.push(ColStats {
            sum,
            max: mx,
            any,
            longest_run: best,
        });
    }

    let max_consecutive_fragment_scans =
        col_stats.iter().map(|c| c.longest_run).max().unwrap_or(0);
    let fragments_with_min_consecutive = col_stats
        .iter()
        .filter(|c| c.longest_run >= MIN_CONSECUTIVE_FRAGMENT_SCANS)
        .count();
    features.insert(
        "xic_max_consecutive_fragment_scans".into(),
        max_consecutive_fragment_scans as f64,
    );
    features.insert(
        "xic_n_fragments_with_min_consecutive".into(),
        fragments_with_min_consecutive as f64,
    );
    features.insert(
        "xic_min_consecutive_scans_required".into(),
        MIN_CONSECUTIVE_FRAGMENT_SCANS as f64,
    );

    let active_frags: Vec<usize> = (0..n_frags).filter(|&f| col_stats[f].any).collect();
    features.insert("xic_n_active_fragments".into(), active_frags.len() as f64);

    features.insert(
        "xic_mean_frags_per_scan".into(),
        frag_count_sum as f64 / n_scans as f64,
    );
    features.insert("xic_max_frags_per_scan".into(), frag_count_max as f64);
    features.insert(
        "xic_multi_frag_coverage".into(),
        multi_frag_scans as f64 / n_scans as f64,
    );

    if active_frags.len() < 2 || scans_with_any_fragment < 3 {
        features.insert("xic_best_coelution".into(), 0.0);
        features.insert("xic_mean_coelution".into(), 0.0);
        features.insert("xic_median_coelution".into(), 0.0);
        features.insert("xic_n_coeluting".into(), 0.0);
        return features;
    }

    let mut pairwise: Vec<f64> = Vec::new();
    for i in 0..active_frags.len() {
        for j in (i + 1)..active_frags.len() {
            let r = pearson_strided(xic, n_scans, n_frags, active_frags[i], active_frags[j]);
            if !r.is_nan() {
                pairwise.push(r);
            }
        }
    }
    if !pairwise.is_empty() {
        pairwise.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        features.insert("xic_best_coelution".into(), pairwise[0]);
        features.insert(
            "xic_mean_coelution".into(),
            pairwise.iter().sum::<f64>() / pairwise.len() as f64,
        );
        features.insert("xic_median_coelution".into(), pairwise[pairwise.len() / 2]);
        features.insert(
            "xic_n_coeluting".into(),
            pairwise.iter().filter(|&&r| r > 0.7).count() as f64,
        );
    } else {
        features.insert("xic_best_coelution".into(), 0.0);
        features.insert("xic_mean_coelution".into(), 0.0);
        features.insert("xic_median_coelution".into(), 0.0);
        features.insert("xic_n_coeluting".into(), 0.0);
    }

    if target_predictions.len() == n_frags && !active_frags.is_empty() {
        let mut obs_max: Vec<f64> = active_frags.iter().map(|&f| col_stats[f].max).collect();
        let mut pred_vals: Vec<f64> = active_frags.iter().map(|&f| target_predictions[f]).collect();
        let obs_peak = obs_max.iter().cloned().fold(0.0f64, f64::max);
        let pred_peak = pred_vals.iter().cloned().fold(0.0f64, f64::max);
        if obs_peak > 0.0 {
            for v in &mut obs_max {
                *v /= obs_peak;
            }
        }
        if pred_peak > 0.0 {
            for v in &mut pred_vals {
                *v /= pred_peak;
            }
        }
        let r = pearson_1d_impl(&obs_max, &pred_vals);
        features.insert("xic_vs_prediction".into(), if r.is_nan() { 0.0 } else { r });

        let mut apex_scan = 0usize;
        let mut apex_total = -1.0f64;
        for s in 0..n_scans {
            let mut t = 0.0;
            for f in 0..n_frags {
                t += xic[s * n_frags + f];
            }
            if t > apex_total {
                apex_total = t;
                apex_scan = s;
            }
        }
        let mut apex_obs: Vec<f64> = (0..n_frags).map(|f| xic[apex_scan * n_frags + f]).collect();
        let mut apex_pred: Vec<f64> = target_predictions.to_vec();
        let ao_peak = apex_obs.iter().cloned().fold(0.0f64, f64::max);
        let ap_peak = apex_pred.iter().cloned().fold(0.0f64, f64::max);
        if ao_peak > 0.0 {
            for v in &mut apex_obs {
                *v /= ao_peak;
            }
        }
        if ap_peak > 0.0 {
            for v in &mut apex_pred {
                *v /= ap_peak;
            }
        }
        let r2 = pearson_1d_impl(&apex_obs, &apex_pred);
        features.insert("xic_apex_spectrum_corr".into(), if r2.is_nan() { 0.0 } else { r2 });
    }

    let best_frag = active_frags
        .iter()
        .max_by(|&&a, &&b| {
            col_stats[a]
                .sum
                .partial_cmp(&col_stats[b].sum)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
        .unwrap_or(0);
    let apex_val = col_stats[best_frag].max;
    if apex_val > 0.0 {
        let mut apex_idx = 0usize;
        let mut best = 0.0f64;
        for s in 0..n_scans {
            let v = xic[s * n_frags + best_frag];
            if v > best {
                best = v;
                apex_idx = s;
            }
        }
        features.insert("xic_apex_rt".into(), matching_rts[apex_idx]);
    }

    let mut first_detected: Option<usize> = None;
    let mut last_detected: Option<usize> = None;
    for s in 0..n_scans {
        let any = (0..n_frags).any(|f| xic[s * n_frags + f] > 0.0);
        if any {
            if first_detected.is_none() {
                first_detected = Some(s);
            }
            last_detected = Some(s);
        }
    }
    if let Some(i) = first_detected {
        features.insert("xic_detected_rt_start".into(), matching_rts[i]);
    }
    if let Some(i) = last_detected {
        features.insert("xic_detected_rt_end".into(), matching_rts[i]);
    }

    if apex_val > 0.0 && n_scans > 1 {
        let half = apex_val / 2.0;
        let mut above = 0usize;
        for s in 0..n_scans {
            if xic[s * n_frags + best_frag] >= half {
                above += 1;
            }
        }
        let first_rt = matching_rts[0];
        let last_rt = matching_rts[n_scans - 1];
        let rt_per_scan = (last_rt - first_rt) / (n_scans - 1) as f64;
        features.insert("xic_peak_width".into(), above as f64 * rt_per_scan);
        features.insert("xic_apex_intensity".into(), apex_val);
    } else if apex_val > 0.0 {
        features.insert("xic_apex_intensity".into(), apex_val);
    }

    if !active_frags.is_empty() {
        let mut weighted_auc = 0.0f64;
        for &f in &active_frags {
            let mut auc = 0.0;
            for s in 1..n_scans {
                let a = xic[(s - 1) * n_frags + f];
                let b = xic[s * n_frags + f];
                auc += (matching_rts[s] - matching_rts[s - 1]) * (a + b) / 2.0;
            }
            let mut sum_abs = 0.0;
            let mut cnt = 0.0;
            for &g in &active_frags {
                if g == f {
                    continue;
                }
                let r = pearson_strided(xic, n_scans, n_frags, f, g);
                if !r.is_nan() {
                    sum_abs += r.abs();
                    cnt += 1.0;
                }
            }
            let weight = if cnt > 0.0 { sum_abs / cnt } else { 0.0 };
            weighted_auc += auc * weight;
        }
        features.insert(
            "xic_weighted_auc".into(),
            if weighted_auc > 0.0 { weighted_auc.ln() } else { 0.0 },
        );
    }

    features
}

fn pearson_strided(
    xic: &[f64],
    n_scans: usize,
    n_frags: usize,
    col_a: usize,
    col_b: usize,
) -> f64 {
    if n_scans < 2 {
        return f64::NAN;
    }
    let n = n_scans as f64;
    let mut sa = 0.0;
    let mut sb = 0.0;
    for s in 0..n_scans {
        sa += xic[s * n_frags + col_a];
        sb += xic[s * n_frags + col_b];
    }
    let ma = sa / n;
    let mb = sb / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for s in 0..n_scans {
        let a = xic[s * n_frags + col_a] - ma;
        let b = xic[s * n_frags + col_b] - mb;
        num += a * b;
        da += a * a;
        db += b * b;
    }
    let denom = (da * db).sqrt();
    if denom == 0.0 {
        f64::NAN
    } else {
        num / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_ms2_data() -> (Vec<f64>, Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // 5 scans at RT 1.0, 2.0, 3.0, 4.0, 5.0
        // Each scan has 3 peaks: 100.0, 200.0, 300.0 m/z
        // All scans have isolation window covering precursor at 500.0 m/z
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
        // All scans: isolation window 400-600 m/z
        let iso_lower = vec![400.0; 5];
        let iso_upper = vec![600.0; 5];
        (rts, mz_offsets, mz_lengths, mz_flat, int_flat, iso_lower, iso_upper)
    }

    #[test]
    fn test_xic_basic() {
        let (rts, offsets, lengths, mz, ints, iso_lo, iso_hi) = make_simple_ms2_data();
        let target_mzs = vec![100.0, 200.0]; // two co-eluting fragments
        let predictions = vec![1.0, 0.5]; // predicted intensities

        let features = extract_xic_features_impl(
            &rts, &offsets, &lengths, &mz, &ints,
            &iso_lo, &iso_hi,
            &target_mzs, &predictions,
            500.0, // precursor_mz (within 400-600 window)
            0.5, 5.5, // RT window
            20.0,      // ppm (generous for test)
        );

        assert_eq!(features["xic_matching_window_scans"], 5.0);
        assert_eq!(features["xic_n_detected_scans"], 5.0);
        assert_eq!(features["xic_coverage"], 1.0);
        // Fragments 100 and 200 are perfectly co-eluting (same profile shape)
        assert!(features["xic_best_coelution"] > 0.99);
    }

    #[test]
    fn test_xic_isolation_window_filtering() {
        let (rts, offsets, lengths, mz, ints, _, _) = make_simple_ms2_data();
        // Scans 0,1,2 have window 400-600; scans 3,4 have window 700-900
        let iso_lower = vec![400.0, 400.0, 400.0, 700.0, 700.0];
        let iso_upper = vec![600.0, 600.0, 600.0, 900.0, 900.0];
        let target_mzs = vec![100.0, 200.0];

        let features = extract_xic_features_impl(
            &rts, &offsets, &lengths, &mz, &ints,
            &iso_lower, &iso_upper,
            &target_mzs, &[1.0, 0.5],
            500.0, // precursor at 500 → only scans 0,1,2 match
            0.5, 5.5, 20.0,
        );

        // Only 3 scans have matching isolation window
        assert_eq!(features["xic_matching_window_scans"], 3.0);
        assert_eq!(features["xic_total_scans"], 5.0);
    }

    #[test]
    fn test_xic_partial_coverage() {
        let (rts, offsets, lengths, mz, ints, iso_lo, iso_hi) = make_simple_ms2_data();
        let target_mzs = vec![100.0, 500.0]; // 500.0 doesn't exist in any scan

        let features = extract_xic_features_impl(
            &rts, &offsets, &lengths, &mz, &ints,
            &iso_lo, &iso_hi,
            &target_mzs, &[1.0, 0.5],
            500.0, 0.5, 5.5, 20.0,
        );

        // Only fragment at 100.0 is detected, 500.0 is never found
        assert_eq!(features["xic_n_active_fragments"], 1.0);
    }

    #[test]
    fn test_xic_empty() {
        let features = extract_xic_features_impl(
            &[], &[], &[], &[], &[],
            &[], &[],
            &[100.0], &[1.0],
            500.0, 0.0, 10.0, 20.0,
        );
        assert!(features.is_empty());
    }

    #[test]
    fn test_xic_rt_window() {
        let (rts, offsets, lengths, mz, ints, iso_lo, iso_hi) = make_simple_ms2_data();
        let target_mzs = vec![100.0, 200.0];

        // Narrow RT window: only scans 2-4
        let features = extract_xic_features_impl(
            &rts, &offsets, &lengths, &mz, &ints,
            &iso_lo, &iso_hi,
            &target_mzs, &[1.0, 0.5],
            500.0, 1.5, 4.5, 20.0,
        );

        assert_eq!(features["xic_total_scans"], 3.0); // scans at RT 2, 3, 4
    }

    #[test]
    fn test_xic_tracks_consecutive_fragment_runs() {
        let rts = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let offsets = vec![0u64, 2, 4, 6, 8];
        let lengths = vec![2u64, 2, 2, 2, 2];
        let mz = vec![
            100.0, 200.0,
            100.0, 300.0,
            200.0, 300.0,
            100.0, 300.0,
            100.0, 200.0,
        ];
        let ints = vec![
            10.0, 5.0,
            12.0, 2.0,
            6.0, 2.0,
            9.0, 2.0,
            8.0, 4.0,
        ];
        let iso_lo = vec![400.0; 5];
        let iso_hi = vec![600.0; 5];

        let features = extract_xic_features_impl(
            &rts, &offsets, &lengths, &mz, &ints,
            &iso_lo, &iso_hi,
            &[100.0, 200.0], &[1.0, 0.5],
            500.0, 0.5, 5.5, 20.0,
        );

        assert_eq!(features["xic_max_consecutive_fragment_scans"], 2.0);
        assert_eq!(features["xic_n_fragments_with_min_consecutive"], 0.0);
        assert_eq!(features["xic_min_consecutive_scans_required"], 3.0);
    }
}
