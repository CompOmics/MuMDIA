use std::collections::HashMap;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

mod batch;
mod correlation;
mod diann_features;
mod fragment_correlations;
mod r#match;
pub mod mzml;
mod percentiles;
pub mod targeted_xic;
mod topk;

/// Compute the q-th percentile of a 1D array (q in 0..100).
/// Releases the GIL during computation for thread parallelism.
#[pyfunction]
fn percentile(py: Python<'_>, data: PyReadonlyArray1<f64>, q: f64) -> f64 {
    let data = data.as_slice().unwrap().to_vec();
    py.allow_threads(|| percentiles::percentile_impl(&data, q))
}

/// Compute multiple percentiles on a 1D array. Returns one value per quantile.
/// Releases the GIL during computation.
#[pyfunction]
fn compute_percentiles<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    qs: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let data_vec = data.as_slice().unwrap().to_vec();
    let qs_vec = qs.as_slice().unwrap().to_vec();
    let result = py.allow_threads(|| percentiles::compute_percentiles_impl(&data_vec, &qs_vec));
    PyArray1::from_vec(py, result)
}

/// Sort descending and return the first m values, zero-padded if needed.
/// Releases the GIL during computation.
#[pyfunction]
fn compute_top<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    m: usize,
) -> Bound<'py, PyArray1<f64>> {
    let data_vec = data.as_slice().unwrap().to_vec();
    let result = py.allow_threads(|| topk::compute_top_impl(&data_vec, m));
    PyArray1::from_vec(py, result)
}

/// Pearson correlation between two 1D arrays. Returns 0.0 for zero-variance inputs.
/// Releases the GIL during computation.
#[pyfunction]
fn pearson_1d(py: Python<'_>, a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> f64 {
    let a_vec = a.as_slice().unwrap().to_vec();
    let b_vec = b.as_slice().unwrap().to_vec();
    py.allow_threads(|| correlation::pearson_1d_impl(&a_vec, &b_vec))
}

/// Compute Pearson correlations between each row of a 2D matrix and a 1D prediction vector.
/// Returns one correlation per row. Releases the GIL during computation.
#[pyfunction]
fn compute_correlations<'py>(
    py: Python<'py>,
    intensity_matrix: PyReadonlyArray2<f64>,
    pred_frag_intens: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let shape = intensity_matrix.shape();
    let num_rows = shape[0];
    let num_cols = shape[1];

    let matrix_vec: Vec<f64> = intensity_matrix.as_array().iter().copied().collect();
    let preds_vec = pred_frag_intens.as_slice().unwrap().to_vec();

    let result = py.allow_threads(|| {
        correlation::compute_correlations_impl(&matrix_vec, num_rows, num_cols, &preds_vec)
    });
    PyArray1::from_vec(py, result)
}

/// Compute all correlation-based features for one peptidoform in a single Rust call.
/// Replaces the entire `run_peptidoform_correlation()` Python function.
#[pyfunction]
#[pyo3(signature = (correlations, correlation_counts, corr_matrix_psm, corr_matrix_frag, most_intens_cor, most_intens_cos, mse_avg, mse_avg_total, percentile_targets, top_k_targets, pad_size))]
fn batch_correlation_features(
    py: Python<'_>,
    correlations: PyReadonlyArray1<f64>,
    correlation_counts: PyReadonlyArray1<f64>,
    corr_matrix_psm: PyReadonlyArray1<f64>,
    corr_matrix_frag: PyReadonlyArray1<f64>,
    most_intens_cor: f64,
    most_intens_cos: f64,
    mse_avg: f64,
    mse_avg_total: f64,
    percentile_targets: Vec<f64>,
    top_k_targets: Vec<usize>,
    pad_size: usize,
) -> HashMap<String, f64> {
    let correlations = correlations.as_slice().unwrap();
    let counts = correlation_counts.as_slice().unwrap();
    let psm = corr_matrix_psm.as_slice().unwrap();
    let frag = corr_matrix_frag.as_slice().unwrap();

    py.allow_threads(|| {
        batch::batch_correlation_features_impl(
            correlations,
            counts,
            psm,
            frag,
            most_intens_cor,
            most_intens_cos,
            mse_avg,
            mse_avg_total,
            &percentile_targets,
            &top_k_targets,
            pad_size,
        )
    })
}

/// Compute cosine similarity between two 1D arrays. Returns 0.0 for zero-norm inputs.
#[pyfunction]
fn cosine_similarity(py: Python<'_>, a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> f64 {
    let a_vec = a.as_slice().unwrap().to_vec();
    let b_vec = b.as_slice().unwrap().to_vec();
    py.allow_threads(|| fragment_correlations::cosine_similarity_impl(&a_vec, &b_vec))
}

/// Complete fragment correlation pipeline.
/// Takes intensity matrix (n_psms x n_frags, NOT yet row-normalized),
/// prediction vector, non-matched prediction sum, and apex index.
/// Returns tuple of 9 values matching CorrelationResults.
#[pyfunction]
#[pyo3(signature = (intensity_data, n_psms, n_frags, matched_predictions, non_matched_sum, apex_psm_idx))]
fn compute_fragment_correlations<'py>(
    py: Python<'py>,
    intensity_data: PyReadonlyArray1<f64>,
    n_psms: usize,
    n_frags: usize,
    matched_predictions: PyReadonlyArray1<f64>,
    non_matched_sum: f64,
    apex_psm_idx: usize,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
) {
    let data = intensity_data.as_slice().unwrap().to_vec();
    let preds = matched_predictions.as_slice().unwrap().to_vec();

    let result = py.allow_threads(|| {
        fragment_correlations::compute_fragment_correlations_impl(
            &data,
            n_psms,
            n_frags,
            &preds,
            non_matched_sum,
            apex_psm_idx,
        )
    });

    (
        PyArray1::from_vec(py, result.correlations),
        PyArray1::from_vec(py, result.correlation_counts),
        result.sum_pred_frag_intens,
        PyArray1::from_vec(py, result.corr_matrix_psm_ids),
        PyArray1::from_vec(py, result.corr_matrix_frag_ids),
        result.most_intens_cor,
        result.most_intens_cos,
        result.mse_avg_pred_intens,
        result.mse_avg_pred_intens_total,
    )
}

/// Parse an mzML file and return (ms1_dict, ms2_to_ms1_map, ms2_dict).
/// Drop-in replacement for parser_mzml.get_ms1_mzml() — ~5-10x faster than PyOpenMS.
#[pyfunction]
fn parse_mzml_file(
    py: Python<'_>,
    file_path: &str,
) -> PyResult<(
    HashMap<String, HashMap<String, PyObject>>,
    HashMap<String, String>,
    HashMap<String, HashMap<String, PyObject>>,
)> {
    let data = py.allow_threads(|| mzml::parse_mzml(file_path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

    // Convert to Python dicts matching the PyOpenMS format
    let mut ms1_dict: HashMap<String, HashMap<String, PyObject>> = HashMap::new();
    for spec in &data.ms1_spectra {
        let mut entry: HashMap<String, PyObject> = HashMap::new();
        entry.insert(
            "mz".to_string(),
            PyArray1::from_vec(py, spec.mz.clone()).into_any().unbind(),
        );
        entry.insert(
            "intensity".to_string(),
            PyArray1::from_vec(py, spec.intensity.clone())
                .into_any()
                .unbind(),
        );
        entry.insert(
            "retention_time".to_string(),
            spec.retention_time.into_pyobject(py)?.unbind().into(),
        );
        ms1_dict.insert(spec.scan_id.clone(), entry);
    }

    let mut ms2_dict: HashMap<String, HashMap<String, PyObject>> = HashMap::new();
    for spec in &data.ms2_spectra {
        let mut entry: HashMap<String, PyObject> = HashMap::new();
        entry.insert(
            "mz".to_string(),
            PyArray1::from_vec(py, spec.mz.clone()).into_any().unbind(),
        );
        entry.insert(
            "intensity".to_string(),
            PyArray1::from_vec(py, spec.intensity.clone())
                .into_any()
                .unbind(),
        );
        entry.insert(
            "retention_time".to_string(),
            spec.retention_time.into_pyobject(py)?.unbind().into(),
        );
        // Isolation window for DIA MS2 scans
        if let Some(target) = spec.isolation_window_target {
            entry.insert(
                "isolation_window_target".to_string(),
                target.into_pyobject(py)?.unbind().into(),
            );
        }
        if let Some(lower) = spec.isolation_window_lower {
            entry.insert(
                "isolation_window_lower".to_string(),
                lower.into_pyobject(py)?.unbind().into(),
            );
        }
        if let Some(upper) = spec.isolation_window_upper {
            entry.insert(
                "isolation_window_upper".to_string(),
                upper.into_pyobject(py)?.unbind().into(),
            );
        }
        ms2_dict.insert(spec.scan_id.clone(), entry);
    }

    Ok((ms1_dict, data.ms2_to_ms1_map, ms2_dict))
}

/// Compute all DIA-NN-style features for one peptidoform in Rust.
/// Replaces the entire Python DIANNFeatureGenerator.calculate_all_features() call.
/// Compute all DIA-NN-style features for one peptidoform in Rust.
///
/// `na_strategy`: how to handle missing values in correlations.
///   - "overlap_only" (default): only correlate at RTs where both fragments are observed
///   - "fill_zero": fill NaN with 0 and use all RTs
#[pyfunction]
#[pyo3(signature = (rts, frag_ids, intensities, fragment_names, precursor_mz, precursor_charge, peptide_length, top_n=6, top_n_extended=12, na_strategy="overlap_only"))]
fn compute_diann_features(
    py: Python<'_>,
    rts: PyReadonlyArray1<f64>,
    frag_ids: PyReadonlyArray1<u32>,
    intensities: PyReadonlyArray1<f64>,
    fragment_names: Vec<String>,
    precursor_mz: f64,
    precursor_charge: i32,
    peptide_length: usize,
    top_n: usize,
    top_n_extended: usize,
    na_strategy: &str,
) -> HashMap<String, f64> {
    let rts_vec = rts.as_slice().unwrap().to_vec();
    let frag_ids_vec: Vec<u32> = frag_ids.as_slice().unwrap().to_vec();
    let intensities_vec = intensities.as_slice().unwrap().to_vec();
    let strategy = match na_strategy {
        "fill_zero" => diann_features::NaStrategy::FillZero,
        _ => diann_features::NaStrategy::OverlapOnly,
    };

    py.allow_threads(|| {
        diann_features::compute_diann_features_impl(
            &rts_vec,
            &frag_ids_vec,
            &intensities_vec,
            &fragment_names,
            precursor_mz,
            precursor_charge,
            peptide_length,
            top_n,
            top_n_extended,
            strategy,
        )
    })
}

/// Extract XIC features for a single peptidoform from MS2 scans matching its isolation window.
#[pyfunction]
#[pyo3(signature = (ms2_rts, ms2_mz_offsets, ms2_mz_lengths, ms2_mz_flat, ms2_int_flat, ms2_iso_lower, ms2_iso_upper, target_mzs, target_predictions, precursor_mz, rt_min, rt_max, ppm_tolerance=13.0))]
fn extract_xic_features(
    py: Python<'_>,
    ms2_rts: PyReadonlyArray1<f64>,
    ms2_mz_offsets: PyReadonlyArray1<u64>,
    ms2_mz_lengths: PyReadonlyArray1<u64>,
    ms2_mz_flat: PyReadonlyArray1<f64>,
    ms2_int_flat: PyReadonlyArray1<f64>,
    ms2_iso_lower: PyReadonlyArray1<f64>,
    ms2_iso_upper: PyReadonlyArray1<f64>,
    target_mzs: PyReadonlyArray1<f64>,
    target_predictions: PyReadonlyArray1<f64>,
    precursor_mz: f64,
    rt_min: f64,
    rt_max: f64,
    ppm_tolerance: f64,
) -> HashMap<String, f64> {
    let rts = ms2_rts.as_slice().unwrap().to_vec();
    let offsets = ms2_mz_offsets.as_slice().unwrap().to_vec();
    let lengths = ms2_mz_lengths.as_slice().unwrap().to_vec();
    let mz = ms2_mz_flat.as_slice().unwrap().to_vec();
    let ints = ms2_int_flat.as_slice().unwrap().to_vec();
    let iso_lo = ms2_iso_lower.as_slice().unwrap().to_vec();
    let iso_hi = ms2_iso_upper.as_slice().unwrap().to_vec();
    let targets = target_mzs.as_slice().unwrap().to_vec();
    let preds = target_predictions.as_slice().unwrap().to_vec();

    py.allow_threads(|| {
        targeted_xic::extract_xic_features_impl(
            &rts, &offsets, &lengths, &mz, &ints,
            &iso_lo, &iso_hi,
            &targets, &preds,
            precursor_mz, rt_min, rt_max,
            ppm_tolerance,
        )
    })
}

/// Batch XIC extraction: process ALL peptidoforms in a single Rust call.
/// Filters scans by isolation window per peptidoform for proper DIA handling.
#[pyfunction]
#[pyo3(signature = (ms2_rts, ms2_mz_offsets, ms2_mz_lengths, ms2_mz_flat, ms2_int_flat, ms2_iso_lower, ms2_iso_upper, all_target_mzs, all_target_mz_offsets, all_target_mz_lengths, all_predictions, all_pred_offsets, all_pred_lengths, all_precursor_mzs, all_rt_mins, all_rt_maxs, ppm_tolerance=13.0))]
fn batch_extract_xic_features(
    py: Python<'_>,
    ms2_rts: PyReadonlyArray1<f64>,
    ms2_mz_offsets: PyReadonlyArray1<u64>,
    ms2_mz_lengths: PyReadonlyArray1<u64>,
    ms2_mz_flat: PyReadonlyArray1<f64>,
    ms2_int_flat: PyReadonlyArray1<f64>,
    ms2_iso_lower: PyReadonlyArray1<f64>,
    ms2_iso_upper: PyReadonlyArray1<f64>,
    // Per-peptidoform targets: flat arrays with offsets
    all_target_mzs: PyReadonlyArray1<f64>,
    all_target_mz_offsets: PyReadonlyArray1<u64>,
    all_target_mz_lengths: PyReadonlyArray1<u64>,
    all_predictions: PyReadonlyArray1<f64>,
    all_pred_offsets: PyReadonlyArray1<u64>,
    all_pred_lengths: PyReadonlyArray1<u64>,
    all_precursor_mzs: PyReadonlyArray1<f64>,
    all_rt_mins: PyReadonlyArray1<f64>,
    all_rt_maxs: PyReadonlyArray1<f64>,
    ppm_tolerance: f64,
) -> Vec<HashMap<String, f64>> {
    let rts = ms2_rts.as_slice().unwrap();
    let offsets = ms2_mz_offsets.as_slice().unwrap();
    let lengths = ms2_mz_lengths.as_slice().unwrap();
    let mz = ms2_mz_flat.as_slice().unwrap();
    let ints = ms2_int_flat.as_slice().unwrap();
    let iso_lo = ms2_iso_lower.as_slice().unwrap();
    let iso_hi = ms2_iso_upper.as_slice().unwrap();
    let target_mzs = all_target_mzs.as_slice().unwrap();
    let target_offsets = all_target_mz_offsets.as_slice().unwrap();
    let target_lengths = all_target_mz_lengths.as_slice().unwrap();
    let predictions = all_predictions.as_slice().unwrap();
    let pred_offsets = all_pred_offsets.as_slice().unwrap();
    let pred_lengths = all_pred_lengths.as_slice().unwrap();
    let precursor_mzs = all_precursor_mzs.as_slice().unwrap();
    let rt_mins = all_rt_mins.as_slice().unwrap();
    let rt_maxs = all_rt_maxs.as_slice().unwrap();
    let n_peptidoforms = rt_mins.len();

    py.allow_threads(|| {
        (0..n_peptidoforms)
            .into_par_iter()
            .map(|i| {
                let t_off = target_offsets[i] as usize;
                let t_len = target_lengths[i] as usize;
                let p_off = pred_offsets[i] as usize;
                let p_len = pred_lengths[i] as usize;
                targeted_xic::extract_xic_features_impl(
                    rts,
                    offsets,
                    lengths,
                    mz,
                    ints,
                    iso_lo,
                    iso_hi,
                    &target_mzs[t_off..t_off + t_len],
                    &predictions[p_off..p_off + p_len],
                    precursor_mzs[i],
                    rt_mins[i],
                    rt_maxs[i],
                    ppm_tolerance,
                )
            })
            .collect()
    })
}

/// Search one RT-partition mzML for top predicted fragment chromatograms.
/// Parses mzML and performs targeted XIC extraction fully in Rust.
#[pyfunction]
#[pyo3(signature = (mzml_path, peptides, charges, precursor_mzs, rt_mins, rt_maxs, predicted_fragment_mzs, predicted_fragment_mz_offsets, predicted_fragment_mz_lengths, predicted_fragment_names, predicted_fragment_name_offsets, predicted_fragment_name_lengths, predicted_fragment_weights, predicted_fragment_weight_offsets, predicted_fragment_weight_lengths, top_n=3, ppm_tolerance=13.0))]
fn search_partition_chromatograms(
    py: Python<'_>,
    mzml_path: &str,
    peptides: Vec<String>,
    charges: PyReadonlyArray1<u64>,
    precursor_mzs: PyReadonlyArray1<f64>,
    rt_mins: PyReadonlyArray1<f64>,
    rt_maxs: PyReadonlyArray1<f64>,
    predicted_fragment_mzs: PyReadonlyArray1<f64>,
    predicted_fragment_mz_offsets: PyReadonlyArray1<u64>,
    predicted_fragment_mz_lengths: PyReadonlyArray1<u64>,
    predicted_fragment_names: Vec<String>,
    predicted_fragment_name_offsets: PyReadonlyArray1<u64>,
    predicted_fragment_name_lengths: PyReadonlyArray1<u64>,
    predicted_fragment_weights: PyReadonlyArray1<f64>,
    predicted_fragment_weight_offsets: PyReadonlyArray1<u64>,
    predicted_fragment_weight_lengths: PyReadonlyArray1<u64>,
    top_n: usize,
    ppm_tolerance: f64,
) -> PyResult<Vec<HashMap<String, f64>>> {
    let charges_slice = charges.as_slice()?;
    let precursor_mzs_slice = precursor_mzs.as_slice()?;
    let rt_mins_slice = rt_mins.as_slice()?;
    let rt_maxs_slice = rt_maxs.as_slice()?;
    let pred_mzs = predicted_fragment_mzs.as_slice()?;
    let pred_mz_offsets = predicted_fragment_mz_offsets.as_slice()?;
    let pred_mz_lengths = predicted_fragment_mz_lengths.as_slice()?;
    let pred_name_offsets = predicted_fragment_name_offsets.as_slice()?;
    let pred_name_lengths = predicted_fragment_name_lengths.as_slice()?;
    let pred_weights = predicted_fragment_weights.as_slice()?;
    let pred_weight_offsets = predicted_fragment_weight_offsets.as_slice()?;
    let pred_weight_lengths = predicted_fragment_weight_lengths.as_slice()?;

    let n_candidates = peptides.len();
    if charges_slice.len() != n_candidates
        || precursor_mzs_slice.len() != n_candidates
        || rt_mins_slice.len() != n_candidates
        || rt_maxs_slice.len() != n_candidates
        || pred_mz_offsets.len() != n_candidates
        || pred_mz_lengths.len() != n_candidates
        || pred_name_offsets.len() != n_candidates
        || pred_name_lengths.len() != n_candidates
        || pred_weight_offsets.len() != n_candidates
        || pred_weight_lengths.len() != n_candidates
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "candidate array lengths must match",
        ));
    }

    let charges_vec: Vec<u8> = charges_slice.iter().map(|&v| v as u8).collect();
    let mzml = py
        .allow_threads(|| mzml::parse_mzml(mzml_path))
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    Ok(py.allow_threads(|| {
        targeted_xic::search_partition_chromatograms_impl(
            &mzml,
            &peptides,
            &charges_vec,
            precursor_mzs_slice,
            rt_mins_slice,
            rt_maxs_slice,
            pred_mzs,
            pred_mz_offsets,
            pred_mz_lengths,
            &predicted_fragment_names,
            pred_name_offsets,
            pred_name_lengths,
            pred_weights,
            pred_weight_offsets,
            pred_weight_lengths,
            top_n,
            ppm_tolerance,
        )
    }))
}

/// Rust-accelerated numerical functions for MuMDIA proteomics pipeline.
#[pymodule]
fn mumdia_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(percentile, m)?)?;
    m.add_function(wrap_pyfunction!(compute_percentiles, m)?)?;
    m.add_function(wrap_pyfunction!(compute_top, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_1d, m)?)?;
    m.add_function(wrap_pyfunction!(compute_correlations, m)?)?;
    m.add_function(wrap_pyfunction!(batch_correlation_features, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fragment_correlations, m)?)?;
    m.add_function(wrap_pyfunction!(parse_mzml_file, m)?)?;
    m.add_function(wrap_pyfunction!(compute_diann_features, m)?)?;
    m.add_function(wrap_pyfunction!(extract_xic_features, m)?)?;
    m.add_function(wrap_pyfunction!(batch_extract_xic_features, m)?)?;
    m.add_function(wrap_pyfunction!(search_partition_chromatograms, m)?)?;
    m.add_function(wrap_pyfunction!(r#match::prefilter_window_candidates, m)?)?;
    Ok(())
}
