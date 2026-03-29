use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

mod correlation;
mod percentiles;
mod topk;

/// Compute the q-th percentile of a 1D array (q in 0..100).
#[pyfunction]
fn percentile(data: PyReadonlyArray1<f64>, q: f64) -> f64 {
    percentiles::percentile_impl(data.as_slice().unwrap(), q)
}

/// Compute multiple percentiles on a 1D array. Returns one value per quantile.
#[pyfunction]
fn compute_percentiles<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    qs: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let result =
        percentiles::compute_percentiles_impl(data.as_slice().unwrap(), qs.as_slice().unwrap());
    PyArray1::from_vec(py, result)
}

/// Sort descending and return the first m values, zero-padded if needed.
#[pyfunction]
fn compute_top<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    m: usize,
) -> Bound<'py, PyArray1<f64>> {
    let result = topk::compute_top_impl(data.as_slice().unwrap(), m);
    PyArray1::from_vec(py, result)
}

/// Pearson correlation between two 1D arrays. Returns 0.0 for zero-variance inputs.
#[pyfunction]
fn pearson_1d(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> f64 {
    correlation::pearson_1d_impl(a.as_slice().unwrap(), b.as_slice().unwrap())
}

/// Compute Pearson correlations between each row of a 2D matrix and a 1D prediction vector.
/// Returns one correlation per row.
#[pyfunction]
fn compute_correlations<'py>(
    py: Python<'py>,
    intensity_matrix: PyReadonlyArray2<f64>,
    pred_frag_intens: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let shape = intensity_matrix.shape();
    let num_rows = shape[0];
    let num_cols = shape[1];

    // Get contiguous data (row-major)
    let matrix_vec: Vec<f64> = intensity_matrix
        .as_array()
        .iter()
        .copied()
        .collect();
    let preds = pred_frag_intens.as_slice().unwrap();

    let result = correlation::compute_correlations_impl(&matrix_vec, num_rows, num_cols, preds);
    PyArray1::from_vec(py, result)
}

/// Rust-accelerated numerical functions for MuMDIA proteomics pipeline.
#[pymodule]
fn mumdia_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(percentile, m)?)?;
    m.add_function(wrap_pyfunction!(compute_percentiles, m)?)?;
    m.add_function(wrap_pyfunction!(compute_top, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_1d, m)?)?;
    m.add_function(wrap_pyfunction!(compute_correlations, m)?)?;
    Ok(())
}
