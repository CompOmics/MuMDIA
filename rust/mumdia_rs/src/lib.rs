use std::collections::HashMap;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

mod batch;
mod correlation;
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

/// Rust-accelerated numerical functions for MuMDIA proteomics pipeline.
#[pymodule]
fn mumdia_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(percentile, m)?)?;
    m.add_function(wrap_pyfunction!(compute_percentiles, m)?)?;
    m.add_function(wrap_pyfunction!(compute_top, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_1d, m)?)?;
    m.add_function(wrap_pyfunction!(compute_correlations, m)?)?;
    m.add_function(wrap_pyfunction!(batch_correlation_features, m)?)?;
    Ok(())
}
