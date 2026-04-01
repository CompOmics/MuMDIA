#!/usr/bin/env python
"""
MuMDIA: Multi-modal Data-Independent Acquisition proteomics analysis.

This module contains the core feature calculation and machine learning pipeline
for peptide-spectrum match scoring using retention time, fragment intensity,
and MS1 precursor features.
"""

import concurrent.futures
import logging
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

# Optional Rust backend: provides ~3x faster numerical functions and GIL release
try:
    import mumdia_rs

    _RUST_BACKEND = True
except ImportError:
    _RUST_BACKEND = False

# Optional numba: provide no-op decorator if unavailable
try:
    import numba as nb
except Exception:

    class _NB:
        def njit(self, *args, **kwargs):
            def deco(f):
                return f

            return deco

    nb = _NB()
import numpy as np
import polars as pl

# Defer keras/scikeras imports to runtime in create_model/run_mokapot

# Optional tqdm: fallback to identity iterator if unavailable
try:
    from tqdm import tqdm
except Exception:

    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


# Optional scipy: create a lazy placeholder that errors on use
try:
    from scipy import stats
except Exception:

    class _Stats:
        def __getattr__(self, name):
            raise ImportError("scipy is required for this functionality")

    stats = _Stats()

from data_structures import PickleConfig, SpectraData
from feature_generators.features_fragment_intensity import (
    get_features_fragment_intensity,
)
from feature_generators.features_general import add_count_and_filter_peptides
from feature_generators.features_retention_time import add_retention_time_features
from prediction_wrappers.wrapper_deeplc import get_predictions_retention_time_mainloop
from prediction_wrappers.wrapper_ms2pip import (
    get_predictions_fragment_intensity_main_loop,
)
from quantification.lfq import quantify_fragments
from utilities.logger import log_info
from utilities.plotting import plot_rt_margin_histogram, plot_XIC_with_margins

# Re-export for backward compatibility
__all__ = ["main", "PickleConfig", "SpectraData", "run_mokapot"]

# Set maximum threads for Polars to one to avoid oversubscription
os.environ["POLARS_MAX_THREADS"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


#############################################
# Numba-accelerated functions
#############################################
@nb.njit
def numba_percentile(data, q):
    """
    Compute the q-th percentile of a 1D array using a simple linear interpolation.
    q should be given as a float between 0 and 100.
    """
    n = data.shape[0]
    if n == 0:
        return 0.0
    sorted_data = np.sort(data)
    pos = (q / 100.0) * (n - 1)
    lower = int(pos)
    upper = lower if lower == n - 1 else lower + 1
    weight = pos - lower
    return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight


@nb.njit
def numba_percentile_sorted(sorted_data, q):
    """
    Compute the q-th percentile of a 1D array using a simple linear interpolation.
    q should be given as a float between 0 and 100.
    """
    n = sorted_data.shape[0]
    if n == 0:
        return 0.0
    pos = (q / 100.0) * (n - 1)
    lower = int(pos)
    upper = lower if lower == n - 1 else lower + 1
    weight = pos - lower
    return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight


@nb.njit
def numba_percentile_sorted_idx(sorted_data, q):
    """
    Compute the q-th percentile of a 1D array using a simple linear interpolation.
    q should be given as a float between 0 and 100.
    """
    n = sorted_data.shape[0]
    if n == 0:
        return 0.0, 0
    pos = (q / 100.0) * (n - 1)
    lower = int(pos)
    upper = lower if lower == n - 1 else lower + 1
    weight = pos - lower
    return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight, int(pos)


@nb.njit
def compute_percentiles_nb(data, qs):
    """
    Compute an array of percentiles given a 1D array and an array of q values.
    """
    m = qs.shape[0]
    result = np.empty(m, dtype=np.float64)
    data = np.sort(data)
    for i in range(m):
        result[i] = numba_percentile_sorted(data, qs[i])
    return result


@nb.njit
def compute_percentiles_nb_idx(data, qs, idx_lookup):
    """
    Compute an array of percentiles given a 1D array `data` and an array of q values `qs`,
    and use the provided `idx_lookup` array to retrieve index information.
    """
    m = qs.shape[0]
    result = np.empty(m, dtype=np.float64)
    computed_idx = np.empty(m, dtype=np.float64)
    for i in range(m):
        result[i], pos = numba_percentile_sorted_idx(data, qs[i])
        computed_idx[i] = idx_lookup[pos]
    return result, computed_idx


@nb.njit
def compute_top_nb(data, m):
    """
    Sort the array in descending order and return the first m values.
    If there are fewer than m elements, pad with zeros.
    """
    n = data.shape[0]
    sorted_data = np.sort(data)[::-1]
    result = np.empty(m, dtype=np.float64)
    for i in range(m):
        if i < n:
            result[i] = sorted_data[i]
        else:
            result[i] = 0.0
    return result


@nb.njit
def compute_top_nb_idx(data, m, idx_ret_list):
    """
    Sort the array in descending order and return the first m values.
    If there are fewer than m elements, pad with zeros.
    """
    n = data.shape[0]
    sorted_data = np.sort(data)[::-1]  # Descending sort
    result = np.empty(m, dtype=np.float64)
    result_idx = np.empty(m, dtype=np.float64)
    for i in range(m):
        if i < n:
            result[i] = sorted_data[i]
            # NOTE: This is a no-op (self-assignment). The idx_ret_list indices
            # are not reordered to match the sorted values. This means the index
            # tracking does not correspond to the actual top-k positions.
            idx_ret_list[i] = idx_ret_list[i]
        else:
            result[i] = 0.0
            idx_ret_list[i] = 0.0
    return result, idx_ret_list


@nb.njit
def corr_np_nb(data1, data2):
    """
    Compute the Pearson correlation coefficient between two 1D arrays.

    WARNING: No zero-variance guard — will produce NaN/inf if either array
    is constant. Use corr_np_nb_new() for a safe version.
    """
    n = data1.shape[0]
    sum1 = 0.0
    sum2 = 0.0
    for i in range(n):
        sum1 += data1[i]
        sum2 += data2[i]
    mean1 = sum1 / n
    mean2 = sum2 / n

    cov = 0.0
    var1 = 0.0
    var2 = 0.0
    for i in range(n):
        diff1 = data1[i] - mean1
        diff2 = data2[i] - mean2
        cov += diff1 * diff2
        var1 += diff1 * diff1
        var2 += diff2 * diff2
    std1 = (var1 / n) ** 0.5
    std2 = (var2 / n) ** 0.5
    # Division by zero if either std is 0 (constant array) — no guard here
    return cov / n / (std1 * std2)


@nb.njit
def pearson_np_nb(x, y):
    """
    Compute the Pearson correlation between a 2D array x and a 1D array y.
    Returns a 1D array with the correlation for each column of x.
    """
    m = x.shape[1]
    result = np.empty(m, dtype=np.float64)
    for i in range(m):
        result[i] = corr_np_nb(x[:, i], y)
    return result


#############################################
# End of Numba functions
#############################################


def create_model(meta=None):
    """
    Create and compile a Keras model for Mokapot PSM scoring.

    Args:
        meta: Metadata dict from scikit-keras containing n_features_in_,
            n_classes_, etc. Automatically passed by KerasClassifier at
            fit time with the actual feature count from training data.
    """
    try:
        from keras.layers import Dense
        from keras.models import Sequential
    except Exception as e:
        raise ImportError(
            f"Keras is required to build the model for mokapot integration ({e})."
        )

    # Extract feature count from scikit-keras metadata, or fall back to default
    n_features = meta["n_features_in_"] if meta else 69

    model = Sequential()
    model.add(Dense(100, input_dim=n_features, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))  # Binary output: target vs decoy
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def transform_bool(value: bool) -> int:
    """
    Return -1 if True, otherwise 1.
    """
    return -1 if value else 1


def run_mokapot(output_dir="results/") -> None:
    """
    Run the mokapot analysis on PSMs read from a PIN file.
    The results are saved to tab-delimited text files.
    """
    try:
        import mokapot
        from scikeras.wrappers import KerasClassifier
    except Exception as e:
        log_info(
            f"mokapot is not installed or failed to import ({e}). Skipping mokapot run."
        )
        return None
    psms = mokapot.read_pin(f"{output_dir}/outfile.pin")

    model = KerasClassifier(
        build_fn=create_model, epochs=100, batch_size=1000, verbose=10
    )
    results, models = mokapot.brew(psms, mokapot.Model(model), folds=3)  # psms)
    result_files = results.to_txt(dest_dir=output_dir)

    return result_files


def collapse_columns(
    df_psms_sub_peptidoform: pl.DataFrame,
    collapse_max_columns: List[str] = [],
    collapse_min_columns: List[str] = [],
    collapse_mean_columns: List[str] = [],
    collapse_sum_columns: List[str] = [],
    get_first_entry: List[str] = [],
):
    """
    Collapse multiple PSM rows for one peptidoform into a single feature row.

    Takes all PSMs for a peptidoform and produces one row by:
    - Taking the first row's values for metadata columns (get_first_entry)
    - Aggregating numeric columns via max/min/mean/sum, suffixed as e.g. "hyperscore_max"

    Returns a 1-row DataFrame with all collapsed columns concatenated horizontally.
    """
    # Take metadata from the first PSM row (arbitrary — these are identical per peptidoform)
    collapsed_columns = [df_psms_sub_peptidoform.select(get_first_entry).head(1)]
    operations = (
        ("max", collapse_max_columns),
        ("min", collapse_min_columns),
        ("mean", collapse_mean_columns),
        ("sum", collapse_sum_columns),
    )
    for op, collapse_list in operations:
        if collapse_list:
            # Apply aggregation across all rows, rename columns with suffix
            collapsed_columns.append(
                getattr(df_psms_sub_peptidoform[collapse_list], op)().rename(
                    {col: f"{col}_{op}" for col in collapse_list}
                )
            )
    # Horizontal concat: one metadata block + one block per aggregation type
    return pl.concat(collapsed_columns, how="horizontal")


def add_feature_columns_nb(data, feature_name, values, method, add_index, pad_size=10):
    """
    Compute a feature vector from the input data using Numba-accelerated routines.
    Returns a dictionary mapping column names to computed scalar values.
    """
    # logging.info(
    #     f"add_feature_columns_nb: feature_name={feature_name}, method={method}, values={values}, pad_size={pad_size}"
    # )
    data = np.asarray(data, dtype=np.float64)
    required_length = len(values)
    computed_idx = np.array([], dtype=np.float64)
    # logging.debug(f"Input data size: {data.size}")
    if data.size == 0:
        # logging.info("Input data is empty, returning zeros.")
        computed = np.zeros(required_length, dtype=np.float64)
        if len(add_index) > 0:
            computed_idx = np.zeros(required_length, dtype=np.float64)
    elif method == "percentile":
        qs = np.array(values, dtype=np.float64)
        # logging.info(f"Computing percentiles: qs={qs}")
        if len(add_index) > 0:
            computed, computed_idx = compute_percentiles_nb_idx(data, qs, add_index)
            # logging.debug(
            #     f"Percentile results: computed={computed}, computed_idx={computed_idx}"
            # )
        else:
            if _RUST_BACKEND:
                computed = mumdia_rs.compute_percentiles(data, qs)
            else:
                computed = compute_percentiles_nb(data, qs)
    elif method == "top":
        if len(add_index) > 0:
            computed, computed_idx = compute_top_nb_idx(
                data, required_length, add_index
            )
        else:
            if _RUST_BACKEND:
                computed = mumdia_rs.compute_top(data, required_length)
            else:
                computed = compute_top_nb(data, required_length)
    else:
        logging.error(f"Unknown method: {method}")
        raise ValueError(f"Unknown method: {method}")
    # Ensure computed is of the required length
    if computed.size < required_length:
        # logging.info(
        #     f"Padded computed array from size {computed.size} to {required_length}"
        # )
        padded = np.zeros(required_length, dtype=np.float64)
        padded[: computed.size] = computed
        computed = padded
        if len(add_index) > 0:
            padded_idx = np.zeros(required_length, dtype=np.float64)
            if computed_idx.size > 0:
                padded_idx[: computed_idx.size] = computed_idx
            computed_idx = padded_idx
    else:
        computed = computed[:required_length]
        if len(add_index) > 0 and computed_idx.size > 0:
            computed_idx = computed_idx[:required_length]

    if len(add_index) > 0:
        # logging.info(f"Returning feature dict with index columns for {feature_name}")
        return {
            **{f"{feature_name}_{v}": computed[i] for i, v in enumerate(values)},
            **{
                f"{feature_name}_{v}_idx": computed_idx[i] for i, v in enumerate(values)
            },
        }
    else:
        # logging.info(f"Returning feature dict for {feature_name}")
        return {f"{feature_name}_{v}": computed[i] for i, v in enumerate(values)}


def run_peptidoform_df(
    df_psms_sub_peptidoform: pl.DataFrame,
    collapse_max_columns: List[str] = [
        "fragment_ppm",
        "rank",
        "delta_next",
        "delta_rt_model",
        "matched_peaks",
        "longest_b",
        "longest_y",
        "matched_intensity_pct",
        "spectrum_q",
        "peptide_q",
        "rt_prediction_error_abs_relative",
        "precursor_ppm",
        "hyperscore",
        # "protein_q",
        "precursor_intensity_M",
        "precursor_intensity_M+1",
        "precursor_intensity_M-1",
    ],
    collapse_min_columns: List[str] = [
        "fragment_ppm",
        "rank",
        "delta_next",
        "delta_rt_model",
        "matched_peaks",
        "longest_b",
        "longest_y",
        "matched_intensity_pct",
        "fragment_intensity",
        "poisson",
        "spectrum_q",
        "peptide_q",
        "rt",
        "rt_predictions",
        "rt_prediction_error_abs",
        "rt_prediction_error_abs_relative",
        "precursor_ppm",
        "hyperscore",
        "delta_best",
        # "protein_q",
        "precursor_intensity_M",
        "precursor_intensity_M+1",
        "precursor_intensity_M-1",
    ],
    collapse_mean_columns: List[str] = [
        "spectrum_q",
        "peptide_q",
        # "protein_q",
        "precursor_intensity_M",
        "precursor_intensity_M+1",
        "precursor_intensity_M-1",
    ],
    collapse_sum_columns: List[str] = [
        "precursor_intensity_M",
        "precursor_intensity_M+1",
        "precursor_intensity_M-1",
    ],
    get_first_entry: List[str] = [
        "psm_id",
        "filename",
        "scannr",
        "peptide",
        "num_proteins",
        "proteins",
        "expmass",
        "calcmass",
        "is_decoy",
        "charge",
        "peptide_len",
        "missed_cleavages",
    ],
) -> pl.DataFrame:
    """
    Collapse all PSMs for one peptidoform into a single feature row for the PIN file.

    Takes a peptidoform-grouped sub-DataFrame containing multiple PSMs and:
    1. Collapses numeric Sage score columns via max/min/mean/sum aggregation
    2. Converts is_decoy (bool) to Label format (-1 for decoy, +1 for target)
    3. Creates SpecId as "psm_id|filename|scannr" (unique peptidoform identifier)

    The collapse_*_columns defaults define which Sage output columns get which
    aggregation. These are hardcoded here and NOT read from the config system.
    """
    df_psms_sub_peptidoform_collapsed = collapse_columns(
        df_psms_sub_peptidoform,
        collapse_max_columns=collapse_max_columns,
        collapse_min_columns=collapse_min_columns,
        collapse_mean_columns=collapse_mean_columns,
        collapse_sum_columns=collapse_sum_columns,
        get_first_entry=get_first_entry,
    )

    df_psms_sub_peptidoform_collapsed = df_psms_sub_peptidoform_collapsed.with_columns(
        pl.when(pl.col("is_decoy")).then(-1).otherwise(1).alias("is_decoy")
    )
    df_psms_sub_peptidoform_collapsed = df_psms_sub_peptidoform_collapsed.with_columns(
        pl.Series(
            "SpecId",
            df_psms_sub_peptidoform_collapsed["psm_id"]
            + "|"
            + df_psms_sub_peptidoform_collapsed["filename"]
            + "|"
            + df_psms_sub_peptidoform_collapsed["scannr"],
        )
    )

    return df_psms_sub_peptidoform_collapsed


def pearson_pvalue(r, n):
    """
    Compute the two-tailed p-value for a Pearson correlation coefficient
    given the sample size n.

    Parameters
    ----------
    r : float
        Pearson correlation coefficient.
    n : int
        Number of datapoints used in the correlation.

    Returns
    -------
    float
        Two-tailed p-value. Returns np.nan if n <= 2.
    """
    if n <= 2:
        return np.nan  # Not enough datapoints for a meaningful p-value.
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)
    return p_value


@nb.njit
def corr_np_nb_new(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient using Numba acceleration.

    Args:
        data1: First data array
        data2: Second data array

    Returns:
        Pearson correlation coefficient
    """
    n = data1.shape[0]
    if n == 0:
        return 0.0

    # Compute means
    sum1 = 0.0
    sum2 = 0.0
    for i in range(n):
        sum1 += data1[i]
        sum2 += data2[i]
    mean1 = sum1 / n
    mean2 = sum2 / n

    # Compute correlation
    cov = 0.0
    var1 = 0.0
    var2 = 0.0
    for i in range(n):
        diff1 = data1[i] - mean1
        diff2 = data2[i] - mean2
        cov += diff1 * diff2
        var1 += diff1 * diff1
        var2 += diff2 * diff2

    std1 = (var1 / n) ** 0.5
    std2 = (var2 / n) ** 0.5

    if std1 == 0.0 or std2 == 0.0:
        return 0.0

    return cov / n / (std1 * std2)


@nb.njit
def corr_np_with_n_new(data1, data2):
    """
    Compute Pearson correlation coefficient and return both the correlation
    and the number of datapoints used.

    Args:
        data1: First 1D array.
        data2: Second 1D array (same length as data1).

    Returns:
        Tuple of (correlation_coefficient, n) where n is the array length.
    """
    n = data1.shape[0]
    # Compute correlation as before
    sum1 = 0.0
    sum2 = 0.0
    for i in range(n):
        sum1 += data1[i]
        sum2 += data2[i]
    mean1 = sum1 / n
    mean2 = sum2 / n

    cov = 0.0
    var1 = 0.0
    var2 = 0.0
    for i in range(n):
        diff1 = data1[i] - mean1
        diff2 = data2[i] - mean2
        cov += diff1 * diff2
        var1 += diff1 * diff1
        var2 += diff2 * diff2
    std1 = (var1 / n) ** 0.5
    std2 = (var2 / n) ** 0.5

    # Return both the correlation and the count of datapoints
    return cov / n / (std1 * std2), n


def run_peptidoform_correlation(
    correlations_list,
    collect_distributions: List[int] = [
        0,
        25,
        50,
        75,
        100,
    ],  # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    collect_top: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # [1, 2, 3, 4, 5],
    pad_size=10,
):
    """
    Compute correlation-based features and return a one-row Polars DataFrame.
    """
    (
        correlations,
        correlation_result_counts,
        sum_pred_frag_intens,
        correlation_matrix_psm_ids,
        correlation_matrix_frag_ids,
        most_intens_cor,
        most_intens_cos,
        mse_avg_pred_intens,
        mse_avg_pred_intens_total,
    ) = correlations_list

    # Fast path: single Rust call replaces 10 Python→Rust round trips
    if _RUST_BACKEND:
        feature_dict = mumdia_rs.batch_correlation_features(
            np.asarray(correlations, dtype=np.float64),
            np.asarray(correlation_result_counts, dtype=np.float64),
            np.asarray(correlation_matrix_psm_ids, dtype=np.float64),
            np.asarray(correlation_matrix_frag_ids, dtype=np.float64),
            float(most_intens_cor),
            float(most_intens_cos),
            float(mse_avg_pred_intens),
            float(mse_avg_pred_intens_total),
            [float(x) for x in collect_distributions],
            [int(x) for x in collect_top],
            pad_size,
        )
        return pl.DataFrame(feature_dict)

    # Fallback: Python path with 10 separate calls
    feature_dict = {}
    params = [
        (
            correlation_matrix_psm_ids,
            "distribution_correlation_matrix_psm_ids",
            collect_distributions,
            "percentile",
            len(collect_distributions),
            [],
        ),
        (
            correlation_matrix_frag_ids,
            "distribution_correlation_matrix_frag_ids",
            collect_distributions,
            "percentile",
            len(collect_distributions),
            [],
        ),
        (
            correlations,
            "distribution_correlation_individual",
            collect_distributions,
            "percentile",
            len(collect_distributions),
            correlation_result_counts,
        ),
        (
            correlation_matrix_psm_ids,
            "top_correlation_matrix_psm_ids",
            collect_top,
            "top",
            pad_size,
            [],
        ),
        (
            correlation_matrix_frag_ids,
            "top_correlation_matrix_frag_ids",
            collect_top,
            "top",
            pad_size,
            [],
        ),
        ([most_intens_cos], "top_correlation_cos", [1], "top", pad_size, []),
        # BUG: Same feature name "top_correlation_cos" as above — Pearson overwrites cosine.
        # Should likely be "top_correlation_pearson" or "top_correlation_cor".
        ([most_intens_cor], "top_correlation_cos", [1], "top", pad_size, []),
        ([mse_avg_pred_intens], "mse_avg_pred_intens", [1], "top", pad_size, []),
        (
            [mse_avg_pred_intens_total],
            "mse_avg_pred_intens_total",
            [1],
            "top",
            pad_size,
            [],
        ),
        (correlations, "top_correlation_individual", collect_top, "top", pad_size, []),
    ]
    for data, feat_name, values, method, ps, add_index in params:
        feature_dict.update(
            add_feature_columns_nb(
                data, feat_name, values, method, add_index, pad_size=ps
            )
        )

    df = pl.DataFrame(feature_dict)
    return df


_diann_generator = None


def _get_diann_generator():
    """Return a shared DIANNFeatureGenerator, creating it once on first call."""
    global _diann_generator
    if _diann_generator is None:
        from feature_generators.diann_feature_generator import (
            DIANNFeatureGenerator,
            FeatureConfig,
        )

        _diann_generator = DIANNFeatureGenerator(FeatureConfig(n_jobs=1))
    return _diann_generator


def _prepare_diann_ms1(spectra_data):
    """Pre-convert ms1_dict to sorted numpy arrays for the DIA-NN generator."""
    gen = _get_diann_generator()
    if gen._ms1_prepared is None and spectra_data and spectra_data.ms1_dict:
        gen.prepare_ms1_dict(spectra_data.ms1_dict)


def run_peptidoform_diann(df_psms_sub, df_fragment_sub, spectra_data, ms2pip_preds):
    """
    Compute DIA-NN-style features for one peptidoform.

    Returns a 1-row Polars DataFrame with diann_* prefixed feature columns.
    Uses Rust implementation when available (mumdia_rs.compute_diann_features),
    falling back to the Python DIANNFeatureGenerator.
    """
    import re

    # === Fast path: Rust DIA-NN features ===
    if _RUST_BACKEND and "fragment_name" in df_fragment_sub.columns:
        try:
            # Extract precursor info from first PSM row
            first_row = df_psms_sub.row(0, named=True)
            peptide = first_row.get("peptide", "")
            calcmass = float(first_row.get("calcmass", 0.0))
            charge = int(first_row.get("charge", 2))
            precursor_mz = calcmass / charge + 1.007276466812
            peptide_length = len(re.sub(r"\[.*?\]", "", peptide))

            # Build fragment name → index mapping
            frag_name_col = df_fragment_sub["fragment_name"]
            unique_names = frag_name_col.unique().sort().to_list()
            name_to_idx = {name: i for i, name in enumerate(unique_names)}

            # Extract parallel arrays for Rust
            rts = df_fragment_sub["rt"].to_numpy().astype(np.float64)
            frag_ids = np.array(
                [name_to_idx[n] for n in frag_name_col.to_list()], dtype=np.uint32
            )
            intensities = (
                df_fragment_sub["fragment_intensity"].to_numpy().astype(np.float64)
            )

            features = mumdia_rs.compute_diann_features(
                rts,
                frag_ids,
                intensities,
                unique_names,
                precursor_mz,
                charge,
                peptide_length,
                na_strategy=_diann_na_strategy,
            )
            # Replace NaN with 0.0
            features = {k: (0.0 if v != v else v) for k, v in features.items()}
            return pl.DataFrame(features)
        except Exception:
            pass  # Fall through to Python path

    # === Python fallback: DIANNFeatureGenerator ===
    generator = _get_diann_generator()

    precursor_pd = df_psms_sub.head(1).to_pandas()
    fragments_pd = df_fragment_sub.to_pandas()

    if (
        "fragment_name" in fragments_pd.columns
        and "fragment_names" not in fragments_pd.columns
    ):
        fragments_pd = fragments_pd.rename(columns={"fragment_name": "fragment_names"})
    if (
        "fragment_ppm" in fragments_pd.columns
        and "ppm_error" not in fragments_pd.columns
    ):
        fragments_pd["ppm_error"] = fragments_pd["fragment_ppm"]
    if (
        "stripped_peptide" not in precursor_pd.columns
        and "peptide" in precursor_pd.columns
    ):
        precursor_pd["stripped_peptide"] = precursor_pd["peptide"].apply(
            lambda p: re.sub(r"\[.*?\]", "", p)
        )

    use_ms1 = (
        generator.config.enable_ms1_features and spectra_data and spectra_data.ms1_dict
    )
    try:
        features = generator.calculate_all_features(
            precursor=precursor_pd,
            fragments=fragments_pd,
            ms1_dict=spectra_data.ms1_dict if use_ms1 else None,
            ms2dict=spectra_data.ms2_dict if spectra_data else None,
            intensity_predictions=ms2pip_preds,
            parallel=False,
        )
    except Exception:
        return pl.DataFrame({"diann_failed": [1.0]})

    flat = {}
    for name, value in features.items():
        if isinstance(value, np.ndarray):
            for i, v in enumerate(value):
                try:
                    flat[f"diann_{name}_{i}"] = (
                        float(v) if not np.isnan(float(v)) else 0.0
                    )
                except (ValueError, TypeError):
                    flat[f"diann_{name}_{i}"] = 0.0
        else:
            try:
                flat[f"diann_{name}"] = (
                    float(value) if not np.isnan(float(value)) else 0.0
                )
            except (ValueError, TypeError):
                flat[f"diann_{name}"] = 0.0
    generator.clear_cache()
    return pl.DataFrame(flat)


def _run_diann_packed(packed_args):
    """Unpack args and run DIA-NN features. Module-level for ProcessPoolExecutor pickling."""
    return run_peptidoform_diann(*packed_args)


_use_diann_features = True  # Set from config in calculate_features()
_diann_na_strategy = "overlap_only"  # "overlap_only" or "fill_zero"


def process_peptidoform(args):
    """
    Process a single peptidoform group by computing its feature DataFrames and concatenating them.
    Computes: collapsed PSM features, correlation features, DIA-NN features, and XIC features.
    """
    (
        df_psms_sub_peptidoform,
        df_fragment_sub_peptidoform,
        correlations_list,
        spectra_data,
        ms2pip_preds,
        xic_features,
    ) = args
    dfs = [
        run_peptidoform_df(df_psms_sub_peptidoform),
        run_peptidoform_correlation(correlations_list),
    ]
    if _use_diann_features:
        dfs.append(
            run_peptidoform_diann(
                df_psms_sub_peptidoform,
                df_fragment_sub_peptidoform,
                spectra_data,
                ms2pip_preds,
            )
        )
    if xic_features:
        dfs.append(pl.DataFrame(xic_features))
    return pl.concat(dfs, how="horizontal")


# TODO move to feature generators
def find_mz_indices(spectrum, target_mz, ppm_tolerance=20):
    """
    Find indices in the sorted m/z array that are within a specified ppm tolerance of a target m/z value.

    Parameters
    ----------
    spectrum : dict
        Dictionary containing the spectrum data with keys 'mz', 'intensity', etc.
    target_mz : float
        The target m/z value to search for.
    ppm_tolerance : float, optional
        The tolerance in parts-per-million (default is 20 ppm).

    Returns
    -------
    indices : numpy.ndarray
        Array of indices in spectrum['mz'] that lie within the specified tolerance.
    """
    # Calculate the absolute tolerance
    tol = target_mz * ppm_tolerance * 1e-6

    # Define the lower and upper bounds of the m/z window
    lower_bound = target_mz - tol
    upper_bound = target_mz + tol

    # Use np.searchsorted to determine the range of indices
    mz_array = spectrum["mz"]
    lower_index = np.searchsorted(mz_array, lower_bound, side="left")
    upper_index = np.searchsorted(mz_array, upper_bound, side="right")

    # Return all indices within the window
    return np.arange(lower_index, upper_index)


def find_all_three_isotopic_peaks(
    spectrum,
    target_mz,
    charge,
    ppm_tolerance=20,
    isotope_mass_diff=1.0033548378,
    return_intensity=False,
):
    """
    Find indices for the target m/z value and its two neighboring isotopic peaks:
    M–1, M, and M+1. If return_intensity is True, return the intensity value (max intensity)
    corresponding to each peak instead of the indices.

    Parameters
    ----------
    spectrum : dict
        Dictionary containing the spectrum data with key 'mz' (a sorted NumPy array)
        and 'intensity' (a NumPy array of intensities).
    target_mz : float
        The target m/z value (typically corresponding to the monoisotopic peak).
    charge : int
        The charge state of the peptide.
    ppm_tolerance : float, optional
        Tolerance in parts-per-million for matching (default is 20 ppm).
    isotope_mass_diff : float, optional
        The nominal mass difference between isotopes (default is 1.0033548378 Da).
    return_intensity : bool, optional
        If True, returns the intensity value (maximum intensity among the matching peaks)
        instead of the indices.

    Returns
    -------
    dict
        A dictionary with keys 'M-1', 'M', and 'M+1'. Depending on return_intensity,
        each key maps either to a NumPy array of indices or to a single intensity value.
    """
    # Calculate the spacing for the given charge.
    spacing = isotope_mass_diff / charge

    # Determine indices for the main and neighboring peaks.
    main_indices = find_mz_indices(spectrum, target_mz, ppm_tolerance)
    lower_indices = find_mz_indices(spectrum, target_mz - spacing, ppm_tolerance)
    upper_indices = find_mz_indices(spectrum, target_mz + spacing, ppm_tolerance)

    if return_intensity:
        # Instead of indices, return the maximum intensity found within the tolerance window.
        intensity_M = (
            np.max(spectrum["intensity"][main_indices])
            if main_indices.size > 0
            else 0.0
        )
        intensity_M_minus = (
            np.max(spectrum["intensity"][lower_indices])
            if lower_indices.size > 0
            else 0.0
        )
        intensity_M_plus = (
            np.max(spectrum["intensity"][upper_indices])
            if upper_indices.size > 0
            else 0.0
        )
        return {"M-1": intensity_M_minus, "M": intensity_M, "M+1": intensity_M_plus}
    else:
        return {"M-1": lower_indices, "M": main_indices, "M+1": upper_indices}


def add_precursor_intensities_optimized_parallel(
    df_psms, ms1_dict, ms2_to_ms1_dict, max_workers=8
):
    """
    Add M-1, M, and M+1 precursor isotope peak intensities to the PSM DataFrame
    using parallel processing.

    For each unique (scannr, charge, calcmass) combination, looks up the
    preceding MS1 spectrum and extracts the maximum intensity within a 20 ppm
    tolerance window for each isotopic peak.

    Args:
        df_psms: Polars DataFrame with columns: scannr, charge, calcmass.
        ms1_dict: Dict mapping MS1 scan IDs to {mz, intensity, retention_time}.
        ms2_to_ms1_dict: Dict mapping MS2 scan IDs to preceding MS1 scan IDs.
        max_workers: Number of threads for ThreadPoolExecutor (default: 8).

    Returns:
        df_psms with added columns: M-1, M, M+1 (precursor isotope intensities).
    """
    # 1. Extract unique precursor combinations
    unique_precursors = df_psms.select(["scannr", "charge", "calcmass"]).unique()

    # 2. Define the function to compute intensities for a single row
    def compute_intensities(row):
        scannr, charge, calcmass = (
            row["scannr"],
            row["charge"],
            row["calcmass"],
        )  # NOTE: uses calcmass (theoretical) not expmass (observed). See also line 933.
        if scannr not in ms2_to_ms1_dict:
            return {"M-1": 0.0, "M": 0.0, "M+1": 0.0}
        # Look up the MS1 spectrum that immediately preceded this MS2 scan
        spectrum = ms1_dict.get(ms2_to_ms1_dict[scannr], {})
        if not spectrum:
            return {"M-1": 0.0, "M": 0.0, "M+1": 0.0}
        # Convert neutral mass to m/z: m/z = (mass / z) + proton_mass
        target_mz = (calcmass / charge) + 1.007276466812
        return find_all_three_isotopic_peaks(
            spectrum, target_mz, charge, return_intensity=True
        )

    # 3. Convert the unique precursors to a list of dictionaries for parallel processing.
    rows = unique_precursors.to_dicts()

    # 4. Use a thread pool to parallelize the intensity computations.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        intensities = list(executor.map(compute_intensities, rows))

    # 5. Convert the list of intensity dictionaries to a DataFrame and merge back.
    intensities_df = pl.DataFrame(intensities)
    unique_precursors = unique_precursors.hstack(intensities_df)

    # 6. Merge the computed intensities back into the original DataFrame.
    df_psms = df_psms.join(unique_precursors, on=["scannr", "charge", "calcmass"])
    return df_psms


def add_precursor_intensities(df_psms, ms1_dict, ms2_to_ms1_dict):
    """Efficiently add precursor intensity features using Polars vectorized operations."""

    def extract_intensities(scannr, charge, calcmass):
        if scannr not in ms2_to_ms1_dict:
            log_info(f"Missing scannr {scannr}")
            return {"M-1": 0.0, "M": 0.0, "M+1": 0.0}  # Default if missing

        spectrum = ms1_dict.get(ms2_to_ms1_dict[scannr], {})
        if not spectrum:
            log_info(f"Not a spectrum {scannr}")
            return {"M-1": 0.0, "M": 0.0, "M+1": 0.0}  # Default if spectrum missing
        target_mz = (calcmass / charge) + 1.007276466812
        return find_all_three_isotopic_peaks(
            spectrum, target_mz, charge, return_intensity=True
        )

    # Apply function using `.map_elements()`, storing result as a struct column
    df_psms = df_psms.with_columns(
        [
            pl.struct(["scannr", "charge", "calcmass"])
            .map_elements(
                lambda row: extract_intensities(
                    row["scannr"],
                    row["charge"],
                    row["calcmass"],  # Should this not be expmass?
                )
            )
            .alias("precursor_intensities")
        ]
    )

    # Extract individual intensity values by using the correct field names
    df_psms = df_psms.with_columns(
        [
            df_psms["precursor_intensities"]
            .struct.field("M-1")
            .alias("precursor_intensity_M-1"),
            df_psms["precursor_intensities"]
            .struct.field("M")
            .alias("precursor_intensity_M"),
            df_psms["precursor_intensities"]
            .struct.field("M+1")
            .alias("precursor_intensity_M+1"),
        ]
    ).drop(
        "precursor_intensities"
    )  # Drop struct column after extraction

    return df_psms


def calculate_rt_margins_intensity_based(
    df_fragments: pl.DataFrame, intensity_threshold: float, output_dir="xics"
) -> pl.DataFrame:
    """
    Calculate retention time margins based on a relative intensity threshold of the apex intensity fragment.
    The margins are determined by finding the retention times where the fragment intensity
    drops below the specified fraction of the apex intensity on both sides of the apex.
    If the intensity never drops below the threshold on one side, the margin is set to the
    first/last retention time where the most intense fragment was detected.
    The function also generates and saves a plot of the XIC with the calculated margins.

    Parameters
    ----------
    df_fragments : pl.DataFrame
        DataFrame containing fragment ion information for a single peptidoform.
    intensity_threshold : float
        Intensity threshold (as a fraction of apex intensity) to define retention time margins.
    output_dir : str
        Directory to save the XIC plots with margins.
    Returns
    -------
    left_bound : float
        Left retention time margin.
    right_bound : float
        Right retention time margin.
    apex_rt : float
        Retention time at apex intensity.
    """

    # Sort by rt
    df_sorted = df_fragments.sort("rt")
    # Find apex
    apex_idx = df_sorted["fragment_intensity"].arg_max()
    apex_rt = df_sorted["rt"][apex_idx]
    apex_intensity = df_sorted["fragment_intensity"][apex_idx]
    # Threshold value
    cutoff = intensity_threshold * apex_intensity
    apex_fragment_name = df_sorted["fragment_name"][apex_idx]

    # Left of apex
    left_df = df_sorted.filter(
        pl.col("fragment_name") == apex_fragment_name
    )  # only consider the apex fragment
    apex_idx_left = left_df["fragment_intensity"].arg_max()
    left_df = left_df[:apex_idx_left][::-1]  # reverse to go from apex down
    left_bound = apex_rt

    for rt, intensity in zip(left_df["rt"], left_df["fragment_intensity"]):
        if intensity < cutoff:
            left_bound = rt
            break

    # if the left bound is still the apex rt, set it to the first rt where fragment was detected
    if left_bound == apex_rt and len(left_df) > 0:
        left_bound = left_df["rt"][-1]

    # Right of apex
    right_df = df_sorted.filter(
        pl.col("fragment_name") == apex_fragment_name
    )  # only consider the apex fragment
    apex_idx_right = right_df["fragment_intensity"].arg_max()
    right_df = right_df[apex_idx_right + 1 :]
    right_bound = apex_rt
    for rt, intensity in zip(right_df["rt"], right_df["fragment_intensity"]):
        if intensity < cutoff:
            right_bound = rt
            break

    # if the right bound is still the apex rt, set it to the last rt where fragment was detected
    if right_bound == apex_rt and len(right_df) > 0:
        right_bound = right_df["rt"][-1]

    # plot XIC with the margins
    # plot_XIC_with_margins(df_sorted, output_dir=output_dir, adapted_interval=(left_bound, right_bound), apex_rt=apex_rt, cutoff=cutoff)

    return left_bound, right_bound, apex_rt


def calculate_min_max_margins(
    df_psms: pl.DataFrame,
    df_fragments: pl.DataFrame,
    top_n: int = 100,
    intensity_threshold: float = 0.01,
) -> dict:
    """
    Calculate the retention time distribution of the top N peptidoforms (with at least 6 PSMs, and then ranked by spectrum peptide q value)
    Min and max margins are defined as the 5th and 95th percentiles of the distribution of retention time margins
    across the top N peptidoforms.
    Returns a tuple with (min_diff, max_diff).

    Parameters
    ----------
    df_psms : pl.DataFrame
        DataFrame containing PSM information
    df_fragments : pl.DataFrame
        DataFrame containing fragment ion information
    top_n : int, optional
        Number of top peptidoforms to consider based on the lowest 'peptide_q' value (default is 100).
    intensity_threshold : float, optional
        Intensity threshold (as a fraction of apex intensity) to define retention time margins (default is 0.01).
    """

    # Step 1: Identify the 100 best scoring peptidoforms based on sage qvalue
    # group by peptide and charge to get unique peptidoforms, aggregate number of PSMs, keep min peptide_q

    df_top_peptidoforms = (
        df_psms.group_by(["peptide", "charge"])
        .agg([pl.count().alias("num_psms"), pl.min("peptide_q").alias("min_peptide_q")])
        .sort("min_peptide_q")
    )

    # filter for peptidoforms with at least 6 PSMs
    df_top_peptidoforms = df_top_peptidoforms.filter(pl.col("num_psms") >= 6)

    # get the top N peptidoforms
    df_top_peptidoforms = df_top_peptidoforms.head(top_n)

    # Step 2: Extract the retention times of the entire XICs from df_fragments of these peptidoforms
    df_fragments_top100 = df_fragments.filter(
        pl.col("peptide").is_in(df_top_peptidoforms["peptide"])
        & pl.col("charge").is_in(df_top_peptidoforms["charge"])
    )
    diffs = []

    for (peptidoform, charge), df_fragments_top100_sub in tqdm(
        df_fragments_top100.group_by(["peptide", "charge"])
    ):
        left_bound, right_bound, apex_rt = calculate_rt_margins_intensity_based(
            df_fragments_top100_sub,
            intensity_threshold,
            output_dir="debug/calibration_xics",
        )
        left_diff = apex_rt - left_bound
        right_diff = right_bound - apex_rt
        diffs.append(left_diff)
        diffs.append(right_diff)

    # remove 0 diffs (if the apex is at the start or end of the XIC)
    diffs = [d for d in diffs if d > 0]

    # Step 3: Calculate the min and max retention times across all these XICs
    if len(diffs) == 0:
        log_info("Could not calibrate retention time margins, using default values.")
        min_diff = 0.02
        max_diff = 0.2
    else:
        # get 5th and 95th percentiles
        min_diff = np.percentile(diffs, 5)
        max_diff = np.percentile(diffs, 95)
        log_info(f"Using min and max retention time margins: {min_diff}, {max_diff}")

    # plot histogram of diffs
    plot_rt_margin_histogram(
        diffs, output_dir="debug/calibration_xics", min_diff=min_diff, max_diff=max_diff
    )

    return min_diff, max_diff


def add_retention_time_margins(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    min_diff: float,
    max_diff: float,
    intensity_threshold: float,
) -> pl.DataFrame:
    """
    Add retention time margin features to the PSM DataFrame.
    """

    pept2lowermargins = {}
    pept2highermargins = {}

    log_info(
        "Calculating adapted retention time margins based on intensity for all peptides"
    )

    for (peptidoform, charge), df_fragments_sub in tqdm(
        df_fragment.group_by(["peptide", "charge"])
    ):
        # speed up: skip peptidoforms with only 1 PSM
        if df_fragments_sub["psm_id"].n_unique() < 2:
            pept2lowermargins[(peptidoform, charge)] = np.nan
            pept2highermargins[(peptidoform, charge)] = np.nan
            continue

        intensity_based_margins = calculate_rt_margins_intensity_based(
            df_fragments_sub, intensity_threshold, output_dir="xics"
        )
        left_bound, right_bound, apex_rt = intensity_based_margins

        # check if the intensity based margins are higher than max or lower than min
        left_diff = apex_rt - left_bound
        right_diff = right_bound - apex_rt

        if left_diff < min_diff:
            left_bound = apex_rt - min_diff
        if right_diff < min_diff:
            right_bound = apex_rt + min_diff
        if left_diff > max_diff:
            left_bound = apex_rt - max_diff
        if right_diff > max_diff:
            right_bound = apex_rt + max_diff

        pept2lowermargins[(peptidoform, charge)] = left_bound
        pept2highermargins[(peptidoform, charge)] = right_bound

    log_info("Adding retention time margin features to PSM DataFrame...")

    # add rt_lower_margin and rt_higher_margin to df_psms
    df_psms = df_psms.with_columns(
        [
            pl.struct(["peptide", "charge"])
            .map_elements(
                lambda row: pept2lowermargins.get(
                    (row["peptide"], row["charge"]), np.nan
                )
            )
            .alias("rt_lower_margin"),
            pl.struct(["peptide", "charge"])
            .map_elements(
                lambda row: pept2highermargins.get(
                    (row["peptide"], row["charge"]), np.nan
                )
            )
            .alias("rt_higher_margin"),
        ]
    )

    log_info("Adding retention time margin features to Fragment DataFrame...")

    # add rt_lower_margin and rt_higher_margin to df_fragment
    df_fragment = df_fragment.with_columns(
        [
            pl.struct(["peptide", "charge"])
            .map_elements(
                lambda row: pept2lowermargins.get(
                    (row["peptide"], row["charge"]), np.nan
                )
            )
            .alias("rt_lower_margin"),
            pl.struct(["peptide", "charge"])
            .map_elements(
                lambda row: pept2highermargins.get(
                    (row["peptide"], row["charge"]), np.nan
                )
            )
            .alias("rt_higher_margin"),
        ]
    )

    return df_psms, df_fragment


def add_retention_time_margins_loop(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    top_n: int = 10,
    intensity_threshold: float = 0.05,
) -> pl.DataFrame:
    """
    Add retention time margin features to the PSM DataFrame.
    """
    log_info("Calculating min max retention time margins based on intensity...")
    # Step 1: Calculate min and max retention time window based on top 100 peptidoforms
    min_diff, max_diff = calculate_min_max_margins(
        df_psms, df_fragment, top_n, intensity_threshold
    )

    # Step 2: Calculate adapted margins for each PSM based on the intensity of the most intense fragment
    # and use the retention time distribution as min and max
    log_info("Adding retention time margin features to PSM DataFrame...")
    df_psms, df_fragment = add_retention_time_margins(
        df_psms, df_fragment, min_diff, max_diff, intensity_threshold
    )

    return df_psms, df_fragment


def calculate_features(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    df_fragment_max: pl.DataFrame,  # Why not used?
    df_fragment_max_peptide: pl.DataFrame,
    *,  # Force keyword-only arguments
    filter_rel_rt_error: float = 0.1,
    min_occurrences: int = 5,
    filter_max_apex_rt: float = 0.75,
    config: dict = {},
    deeplc_model=None,
    pickle_config: Optional[PickleConfig] = None,
    parallel_workers: int = 24,  # Adjust to number of physical cores
    chunk_size: int = 500,  # Increase chunk size to reduce overhead
    spectra_data: Optional[SpectraData] = None,
) -> None:
    """
    Main feature computation pipeline — the workhorse of Stage 3.

    Orchestrates 9 steps to transform raw search results into a scored PIN file:
    1. DeepLC RT predictions (reuses transfer-learned model from Stage 2)
    2. RT error features + filtering (removes PSMs with >15% relative RT error)
    3. Regenerate apex DataFrames after RT filtering (data consistency)
    4. Peptide occurrence filtering (min_occurrences, default 5)
    5. MS2PIP fragment intensity predictions + RustyMS annotation → correlations
    6. Precursor M-1/M/M+1 isotope intensities from MS1 spectra
    7. Group PSMs by peptidoform, collapse via max/min/mean/sum
    8. Extract percentile + top-k features from correlation matrices (parallel)
    9. Write PIN file for Mokapot

    NOTE: filter_rel_rt_error and filter_max_apex_rt parameters are accepted
    but not used — hardcoded values are passed to the feature functions instead.
    The return type annotation says None but actually returns (df_fragment, df_psms).
    """
    # Handle pickle configuration
    if pickle_config is None:
        pickle_config = PickleConfig()

    # Handle spectra data
    if spectra_data is None:
        spectra_data = SpectraData()

    log_info("Obtaining retention time predictions for the main loop...")
    log_info(
        f"Reading the DeepLC pickle: {pickle_config.read_deeplc} and writing DeepLC pickle: {pickle_config.write_deeplc}"
    )

    df_psms.write_csv("debug/df_psms_before_rt.tsv", separator="\t")
    (
        _,
        _,
        df_psms,
    ) = get_predictions_retention_time_mainloop(  # Changed, since predictions_deeplc is just df_psms with RT predictions
        df_psms,
        pickle_config.write_deeplc,
        pickle_config.read_deeplc,
        deeplc_model,
        output_dir=config["mumdia"]["result_dir"],
    )

    # Step 2: Compute RT error features and filter out poor predictions.
    # predictions_deeplc=None because rt_predictions is already in df_psms from Step 1.
    log_info("Obtaining features retention time...")
    df_psms = add_retention_time_features(df_psms, filter_rel_rt_error=0.15)

    df_psms.write_csv("debug/df_psms_after_rt.csv", separator="\t")

    log_info("PSMs shape after RT filtering: {}".format(df_psms.shape))

    # CRITICAL FIX: Regenerate df_fragment_max_peptide after RT filtering
    # to ensure apex PSMs are consistent with filtered data
    log_info("Regenerating df_fragment_max_peptide after RT filtering...")

    # Filter df_fragment to only include PSMs that passed RT filtering
    df_fragment = df_fragment.filter(pl.col("psm_id").is_in(df_psms["psm_id"]))
    log_info(
        "df_fragment shape after filtering to match RT-filtered PSMs: {}".format(
            df_fragment.shape
        )
    )

    # Regenerate the maximum intensity fragment per PSM
    df_fragment_max = df_fragment.sort("fragment_intensity", descending=True).unique(
        subset="psm_id", keep="first", maintain_order=True
    )

    # Regenerate the apex PSM per peptide/charge combination from the filtered data
    df_fragment_max_peptide = (
        df_fragment_max.with_columns(
            [
                (pl.col("peptide") + "/" + pl.col("charge").cast(pl.Utf8)).alias(
                    "peptide_charge"
                )
            ]
        )
        .sort("fragment_intensity", descending=True)
        .unique(subset=["peptide", "charge"], keep="first")
    )

    log_info("Regenerated df_fragment_max_peptide:")
    log_info("  Shape: {}".format(df_fragment_max_peptide.shape))
    log_info("  Sample entries:")
    log_info(
        "Counting individual peptides per MS2 and filtering by minimum occurrences"
    )
    df_psms = add_count_and_filter_peptides(df_psms, min_occurrences)

    # Filter df_fragment to only include PSMs that passed all filtering
    df_fragment = df_fragment.filter(pl.col("psm_id").is_in(df_psms["psm_id"]))

    # Regenerate the maximum intensity fragment per PSM
    df_fragment_max = df_fragment.sort("fragment_intensity", descending=True).unique(
        subset="psm_id", keep="first", maintain_order=True
    )

    # Regenerate the apex PSM per peptide/charge combination from the fully filtered data
    df_fragment_max_peptide = (
        df_fragment_max.with_columns(
            [
                (pl.col("peptide") + "/" + pl.col("charge").cast(pl.Utf8)).alias(
                    "peptide_charge"
                )
            ]
        )
        .sort("fragment_intensity", descending=True)
        .unique(subset=["peptide", "charge"], keep="first")
    )

    log_info("Final df_fragment_max_peptide after all filtering:")
    log_info("  Shape: {}".format(df_fragment_max_peptide.shape))
    log_info("  This should now be consistent with all downstream processing")

    # Validation: Check that all peptides in df_fragment_max_peptide exist in the filtered data
    fragment_max_psm_ids = set(df_fragment_max_peptide["psm_id"].to_list())
    filtered_psm_ids = set(df_psms["psm_id"].to_list())
    missing_psms = fragment_max_psm_ids - filtered_psm_ids
    if missing_psms:
        log_info(
            "WARNING: {} PSMs in df_fragment_max_peptide are missing from filtered df_psms: {}".format(
                len(missing_psms), list(missing_psms)[:5]  # Show first 5
            )
        )
    else:
        log_info("VALIDATION PASSED: All apex PSMs exist in filtered data")

    # Step 4b: Calculate adaptive RT margins per peptidoform.
    # Calibrates min/max margins from top-scoring peptidoforms, then walks each
    # peptidoform's XIC outward from apex until intensity drops below threshold.
    # Adds rt_lower_margin and rt_higher_margin columns to both DataFrames.
    log_info("Calculating adaptive retention time margins...")

    # Construct fragment_name column if not present (Sage parquet doesn't include it)
    if "fragment_name" not in df_fragment.columns:
        df_fragment = df_fragment.with_columns(
            (
                pl.col("fragment_type")
                + pl.col("fragment_ordinals").cast(pl.Utf8)
                + "/"
                + pl.col("fragment_charge").cast(pl.Utf8)
            ).alias("fragment_name")
        )

    df_psms, df_fragment = add_retention_time_margins_loop(
        df_psms, df_fragment, top_n=100, intensity_threshold=0.05
    )

    log_info("Obtaining fragment intensity predictions for the main loop...")
    log_info(
        f"Reading the MS2PIP pickle: {pickle_config.read_ms2pip} and writing MS2PIP pickle: {pickle_config.write_ms2pip}"
    )

    df_fragment, ms2pip_predictions = get_predictions_fragment_intensity_main_loop(
        df_psms,
        df_fragment,
        read_ms2pip_pickle=pickle_config.read_ms2pip,
        write_ms2pip_pickle=pickle_config.write_ms2pip,
        output_dir=config["mumdia"]["result_dir"],
    )

    df_fragment.write_csv("debug/df_fragment_after_ms2pip.tsv", separator="\t")
    df_fragment_max_peptide.write_csv(
        "debug/df_fragment_max_peptide_after_ms2pip.tsv", separator="\t"
    )
    with open("debug/ms2pip_predictions.pkl", "wb") as f:
        pickle.dump(ms2pip_predictions, f)

    with open("debug/ms2dict.pkl", "wb") as f:
        pickle.dump(spectra_data.ms2_dict, f)

    df_psms.write_csv("debug/df_psms_after_ms2pip.tsv", separator="\t")

    fragment_dict, correlations_fragment_dict = get_features_fragment_intensity(
        ms2pip_predictions,
        df_fragment,
        df_fragment_max_peptide,
        read_correlation_pickles=pickle_config.read_correlation,
        write_correlation_pickles=pickle_config.write_correlation,
        ms2_dict=spectra_data.ms2_dict,
        output_dir=config["mumdia"]["result_dir"],
    )

    log_info("Step 5: obtain MS1 peak presence")

    df_psms = add_precursor_intensities(
        df_psms, spectra_data.ms1_dict, spectra_data.ms2_to_ms1_dict
    )

    log_info(f"Number of PSMs:{df_psms.shape[0]}")

    # Step 5b: Targeted XIC extraction from ALL MS2 scans
    xic_features_dict = {}
    if _RUST_BACKEND and spectra_data and spectra_data.ms2_dict:
        log_info("Step 5b: Extracting targeted XICs from all MS2 scans...")

        # Flatten MS2 data into sorted arrays for efficient Rust access
        ms2_items = sorted(spectra_data.ms2_dict.items(), key=lambda x: x[1]["retention_time"])
        ms2_rts_flat = np.array([v["retention_time"] for _, v in ms2_items], dtype=np.float64)

        mz_arrays = [np.asarray(v["mz"], dtype=np.float64) for _, v in ms2_items]
        int_arrays = [np.asarray(v["intensity"], dtype=np.float64) for _, v in ms2_items]

        offsets = np.zeros(len(mz_arrays), dtype=np.uint64)
        lengths = np.zeros(len(mz_arrays), dtype=np.uint64)
        offset = 0
        for i, mz_arr in enumerate(mz_arrays):
            offsets[i] = offset
            lengths[i] = len(mz_arr)
            offset += len(mz_arr)

        ms2_mz_flat = np.concatenate(mz_arrays) if mz_arrays else np.array([], dtype=np.float64)
        ms2_int_flat = np.concatenate(int_arrays) if int_arrays else np.array([], dtype=np.float64)

        log_info(f"  Flattened {len(ms2_items)} MS2 scans ({len(ms2_mz_flat)} total peaks)")

        # Build batch XIC targets from df_fragment (Sage output)
        if "fragment_mz_calculated" in df_fragment.columns:
            log_info("  Building XIC targets from fragment m/z values...")

            # Auto-detect RT margin (seconds vs minutes)
            rt_margin = 180.0 if ms2_rts_flat.max() > 200 else 3.0

            # Pre-group fragment data by peptide/charge for fast lookup
            frag_by_key = {}
            for (peptide, charge), df_sub in df_fragment.group_by(["peptide", "charge"]):
                key = f"{peptide}/{charge}"
                mzs = np.sort(df_sub["fragment_mz_calculated"].unique().to_numpy().astype(np.float64))
                frag_by_key[key] = mzs

            # Build flat arrays for batch Rust call
            xic_keys = []
            all_target_mzs_list = []
            all_preds_list = []
            all_rt_mins = []
            all_rt_maxs = []

            for k in correlations_fragment_dict.keys():
                if k not in frag_by_key:
                    continue
                frag_mzs = frag_by_key[k]
                if len(frag_mzs) == 0:
                    continue

                # RT window
                parts = k.rsplit("/", 1)
                if len(parts) != 2:
                    continue
                peptide, charge_str = parts
                pep_psms = df_psms.filter(
                    (pl.col("peptide") == peptide) & (pl.col("charge") == int(charge_str))
                )
                if len(pep_psms) == 0:
                    continue
                rt_center = float(pep_psms["rt"].median())

                # Predictions
                preds = ms2pip_predictions.get(k, {})
                pred_values = np.zeros(len(frag_mzs), dtype=np.float64)
                # Simple: just fill with available predictions (order may not match perfectly)
                pred_list = list(preds.values())
                for i in range(min(len(pred_list), len(pred_values))):
                    pred_values[i] = pred_list[i]

                xic_keys.append(k)
                all_target_mzs_list.append(frag_mzs)
                all_preds_list.append(pred_values)
                all_rt_mins.append(rt_center - rt_margin)
                all_rt_maxs.append(rt_center + rt_margin)

            log_info(f"  Prepared {len(xic_keys)} peptidoforms for batch XIC extraction")

            if xic_keys:
                # Flatten target arrays with offsets
                target_mz_flat = np.concatenate(all_target_mzs_list)
                target_offsets_arr = np.zeros(len(xic_keys), dtype=np.uint64)
                target_lengths_arr = np.zeros(len(xic_keys), dtype=np.uint64)
                pred_flat = np.concatenate(all_preds_list)
                pred_offsets_arr = np.zeros(len(xic_keys), dtype=np.uint64)
                pred_lengths_arr = np.zeros(len(xic_keys), dtype=np.uint64)

                t_offset = 0
                p_offset = 0
                for i in range(len(xic_keys)):
                    target_offsets_arr[i] = t_offset
                    target_lengths_arr[i] = len(all_target_mzs_list[i])
                    t_offset += len(all_target_mzs_list[i])
                    pred_offsets_arr[i] = p_offset
                    pred_lengths_arr[i] = len(all_preds_list[i])
                    p_offset += len(all_preds_list[i])

                # Single Rust call for ALL peptidoforms
                log_info("  Running batch XIC extraction in Rust...")
                results = mumdia_rs.batch_extract_xic_features(
                    ms2_rts_flat, offsets, lengths, ms2_mz_flat, ms2_int_flat,
                    target_mz_flat, target_offsets_arr, target_lengths_arr,
                    pred_flat, pred_offsets_arr, pred_lengths_arr,
                    np.array(all_rt_mins, dtype=np.float64),
                    np.array(all_rt_maxs, dtype=np.float64),
                    13.0,
                )
                for i, k in enumerate(xic_keys):
                    xic_features_dict[k] = results[i]

            log_info(f"  Extracted XICs for {len(xic_features_dict)} peptidoforms")
        else:
            log_info("  Skipping XIC: fragment_mz_calculated column not available")

    # Step 7: Group all PSMs by peptidoform (peptide/charge) and prepare for parallel processing.
    # Each peptidoform becomes one row in the final PIN file.
    log_info("Step 6: Grouping peptidoforms by peptide and charge")

    psm_dict = {}
    for (peptidoform, charge), df_sub_peptidoform in tqdm(
        df_psms.group_by(["peptide", "charge"])
    ):
        psm_dict[f"{peptidoform}/{charge}"] = df_sub_peptidoform

    log_info(f"Number of peptidoforms: {len(psm_dict)}")

    # Build argument tuples for parallel processing.
    # Each tuple: (psm_sub_df, fragment_sub_df, correlation_list, spectra_data, ms2pip_preds_for_peptidoform)
    # Only include peptidoforms that have both fragment and correlation data.
    peptidoform_args = [
        (
            psm_dict[k],
            fragment_dict[k],
            correlations_fragment_dict[k],
            spectra_data,
            ms2pip_predictions.get(k),
            xic_features_dict.get(k),
        )
        for k in psm_dict.keys()
        if k in correlations_fragment_dict
    ]

    # Save psm_dict to a pickle file for debugging or future use
    with open("debug/psm_dict.pkl", "wb") as f:
        pickle.dump(psm_dict, f)

    with open("debug/correlations_fragment_dict.pkl", "wb") as f:
        pickle.dump(correlations_fragment_dict, f)

    # Step 8: Process each peptidoform in parallel using ThreadPoolExecutor.
    # Each peptidoform is collapsed to one row (run_peptidoform_df) and gets
    # correlation features appended (run_peptidoform_correlation).
    # Chunking reduces ThreadPoolExecutor overhead for many small tasks.
    log_info("Step 7: Processing peptidoforms in parallel")

    # Set DIA-NN feature flags from config
    global _use_diann_features, _diann_na_strategy
    _use_diann_features = config.get("mumdia", {}).get("use_diann_features", True)
    _diann_na_strategy = config.get("mumdia", {}).get(
        "diann_na_strategy", "overlap_only"
    )
    if not _use_diann_features:
        log_info("  DIA-NN features DISABLED (use_diann_features=False)")
    else:
        log_info(f"  DIA-NN NA strategy: {_diann_na_strategy}")

    # Pre-convert MS1 data to sorted numpy arrays for fast DIA-NN elution profiles
    if _use_diann_features:
        _prepare_diann_ms1(spectra_data)

    # Sequential processing. Even with Rust GIL release, ThreadPoolExecutor is
    # slower because process_peptidoform() still does significant Python work
    # (Polars aggregation, DIA-NN pandas conversion, dict building) that holds
    # the GIL. Threading will only help once more of the pipeline is in Rust.
    pin_in = [
        process_peptidoform(args)
        for args in tqdm(peptidoform_args, desc="Processing peptidoforms")
    ]

    # Reset the shared DIA-NN generator after all processing is done
    global _diann_generator
    if _diann_generator is not None:
        _diann_generator.clear_cache()
    _diann_generator = None

    # Step 9: Build the PIN file for Mokapot.
    # Rename columns to Percolator/Mokapot PIN format, drop redundant scannr,
    # fill any NaN/null with 0.0 to avoid Mokapot errors.
    log_info("Step 8: Concatenating results")
    concatenated_df = (
        pl.concat(pin_in, how="diagonal")
        .rename(
            {
                "expmass": "ExpMass",
                "calcmass": "CalcMass",
                "psm_id": "ScanNr",  # Mokapot uses ScanNr for grouping
                "peptide": "Peptide",
                "proteins": "Proteins",
                "is_decoy": "Label",  # -1 = decoy, +1 = target
            }
        )
        .drop("scannr")  # Redundant after SpecId was created
        .fill_null(0.0)
        .fill_nan(0.0)
    )
    concatenated_df.write_csv(
        f"{config['mumdia']['result_dir']}/outfile.pin", separator="\t"
    )

    return df_fragment, df_psms


def main(
    df_fragment: Optional[pl.DataFrame] = None,
    df_psms: Optional[pl.DataFrame] = None,
    df_fragment_max: Optional[pl.DataFrame] = None,
    df_fragment_max_peptide: Optional[pl.DataFrame] = None,
    *,  # Force keyword-only arguments
    config: Dict[str, Any] = {},
    deeplc_model: Optional[Any] = None,
    pickle_config: Optional[PickleConfig] = None,
    spectra_data: Optional[SpectraData] = None,
) -> None:
    """
    Main MuMDIA workflow coordinator for feature calculation and PSM scoring.

    This function orchestrates the complete feature engineering pipeline,
    including retention time predictions, fragment intensity modeling,
    MS1 precursor analysis, and parallel peptidoform processing.

    Args:
        df_fragment: Fragment matches DataFrame from search engine
        df_psms: Peptide-spectrum matches DataFrame
        df_fragment_max: Maximum intensity fragments per PSM
        df_fragment_max_peptide: Maximum intensity fragments per peptide
        config: Configuration dictionary with workflow parameters
        deeplc_model: Optional pre-trained DeepLC model
        pickle_config: Configuration for caching predictions and features
        spectra_data: Container for MS1/MS2 spectral data
    """
    df_psms.write_csv("debug/df_psms_before_mumdia.tsv", separator="\t")
    df_psms = pl.DataFrame(df_psms)
    df_psms = df_psms.filter(~df_psms["peptide"].str.contains("U"))
    df_psms = df_psms.sort("rt")

    df_fragment, df_psms = calculate_features(
        df_psms,
        df_fragment,
        df_fragment_max,
        df_fragment_max_peptide,
        min_occurrences=config.get("mumdia", {}).get("min_occurrences", 5),
        pickle_config=pickle_config,
        deeplc_model=deeplc_model,
        config=config,
        spectra_data=spectra_data,
    )

    log_info("Done running MuMDIA...")
    mokapot_results = run_mokapot(output_dir=config["mumdia"]["result_dir"])

    df_fragment.write_csv("debug/df_fragment_before_quant.tsv", separator="\t")
    df_psms.write_csv("debug/df_psms_before_quant.tsv", separator="\t")

    # this file will later be used for quantification of proteins with directLFQ (combined with all runs)
    if (
        mokapot_results is not None
        and isinstance(mokapot_results, (list, tuple))
        and len(mokapot_results) > 1
    ):
        df_quant_fragment = quantify_fragments(
            df_fragment,
            mokapot_results[1],
            config=config,
            output_dir=config["mumdia"]["result_dir"],
        )
    else:
        logging.warning(
            "mokapot_results is None or does not have enough elements; skipping quantification step."
        )


if __name__ == "__main__":
    # In practice, load your input DataFrames (e.g., from parquet files) and then call main().
    # For demonstration, we call run_mokapot().
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "results/"
    run_mokapot(output_dir=output_dir)
