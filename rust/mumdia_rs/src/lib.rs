use std::collections::HashMap;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

mod batch;
mod correlation;
mod fragment_correlations;
mod mzml;
mod percentiles;
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
        ms2_dict.insert(spec.scan_id.clone(), entry);
    }

    Ok((ms1_dict, data.ms2_to_ms1_map, ms2_dict))
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
    Ok(())
}
