#!/usr/bin/env python
"""
MuMDIA: Multi-modal Data-Independent Acquisition proteomics analysis.

This module contains the core feature calculation and machine learning pipeline
for peptide-spectrum match scoring using retention time, fragment intensity,
and MS1 precursor features.
"""

import concurrent.futures
import json
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

RT_JOIN_DECIMALS = 5
PROTON_MASS = 1.007276466812


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
    df_fragments: pl.DataFrame,
    intensity_threshold: float,
    output_dir="xics",
    apex_rt: Optional[float] = None,
    preferred_fragments: Optional[List[str]] = None,
    all_rt_values: Optional[List[float]] = None,
) -> pl.DataFrame:
    """
    Calculate retention time margins based on a relative intensity threshold of an apex trace.
    The trace is built from preferred predicted fragments when available; otherwise the
    calculation falls back to all fragments for the peptidoform. The margins are determined
    by finding the retention times where the trace intensity drops below the specified
    fraction of the apex intensity on both sides of the apex.
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

    df_sorted = df_fragments.sort("rt")

    if preferred_fragments:
        df_trace_source = df_sorted.filter(
            pl.col("fragment_name").is_in(preferred_fragments)
        )
        if df_trace_source.is_empty():
            df_trace_source = df_sorted
    else:
        df_trace_source = df_sorted

    trace_df = (
        df_trace_source.group_by("rt")
        .agg(pl.sum("fragment_intensity").alias("trace_intensity"))
        .sort("rt")
    )

    if all_rt_values:
        rt_frame = pl.DataFrame(
            {
                "rt": sorted(
                    {
                        float(rt)
                        for rt in all_rt_values
                        if rt is not None and not np.isnan(float(rt))
                    }
                )
            }
        )
        if not rt_frame.is_empty():
            rt_frame = rt_frame.with_columns(
                pl.col("rt").round(RT_JOIN_DECIMALS).alias("_rt_key")
            )
            trace_df_for_join = (
                trace_df.with_columns(
                    pl.col("rt").round(RT_JOIN_DECIMALS).alias("_rt_key")
                )
                .group_by("_rt_key")
                .agg(pl.sum("trace_intensity").alias("trace_intensity"))
            )
            trace_df = (
                rt_frame.join(trace_df_for_join, on="_rt_key", how="left")
                .with_columns(pl.col("trace_intensity").fill_null(0.0))
                .drop("_rt_key")
                .sort("rt")
            )

    if preferred_fragments and not trace_df.is_empty():
        trace_intensity_preview = (
            trace_df["trace_intensity"].to_numpy().astype(np.float64)
        )
        if not np.any(trace_intensity_preview > 0.0):
            trace_df = (
                df_sorted.group_by("rt")
                .agg(pl.sum("fragment_intensity").alias("trace_intensity"))
                .sort("rt")
            )
            if all_rt_values:
                rt_frame = pl.DataFrame(
                    {
                        "rt": sorted(
                            {
                                float(rt)
                                for rt in all_rt_values
                                if rt is not None and not np.isnan(float(rt))
                            }
                        )
                    }
                )
                if not rt_frame.is_empty():
                    rt_frame = rt_frame.with_columns(
                        pl.col("rt").round(RT_JOIN_DECIMALS).alias("_rt_key")
                    )
                    trace_df_for_join = (
                        trace_df.with_columns(
                            pl.col("rt").round(RT_JOIN_DECIMALS).alias("_rt_key")
                        )
                        .group_by("_rt_key")
                        .agg(pl.sum("trace_intensity").alias("trace_intensity"))
                    )
                    trace_df = (
                        rt_frame.join(trace_df_for_join, on="_rt_key", how="left")
                        .with_columns(pl.col("trace_intensity").fill_null(0.0))
                        .drop("_rt_key")
                        .sort("rt")
                    )

    if trace_df.is_empty():
        return np.nan, np.nan, np.nan

    trace_rt = trace_df["rt"].to_numpy().astype(np.float64)
    trace_intensity = trace_df["trace_intensity"].to_numpy().astype(np.float64)
    positive_idx = np.where(trace_intensity > 0.0)[0]

    if apex_rt is None:
        if positive_idx.size > 0:
            apex_idx = int(positive_idx[np.argmax(trace_intensity[positive_idx])])
        else:
            apex_idx = int(np.argmax(trace_intensity))
        apex_rt = float(trace_rt[apex_idx])
    else:
        if positive_idx.size > 0:
            local = int(np.argmin(np.abs(trace_rt[positive_idx] - apex_rt)))
            apex_idx = int(positive_idx[local])
        else:
            apex_idx = int(np.argmin(np.abs(trace_rt - apex_rt)))
        apex_rt = float(trace_rt[apex_idx])

    apex_intensity = float(trace_intensity[apex_idx])
    if apex_intensity <= 0.0 and positive_idx.size > 0:
        apex_idx = int(positive_idx[np.argmax(trace_intensity[positive_idx])])
        apex_rt = float(trace_rt[apex_idx])
        apex_intensity = float(trace_intensity[apex_idx])

    if apex_intensity <= 0.0:
        return np.nan, np.nan, np.nan

    cutoff = intensity_threshold * apex_intensity

    left_df = trace_df[:apex_idx][::-1]
    left_bound = apex_rt
    for rt, intensity in zip(left_df["rt"], left_df["trace_intensity"]):
        if intensity < cutoff:
            left_bound = float(rt)
            break

    if left_bound == apex_rt and len(left_df) > 0:
        left_bound = float(left_df["rt"][-1])

    right_df = trace_df[apex_idx + 1 :]
    right_bound = apex_rt
    for rt, intensity in zip(right_df["rt"], right_df["trace_intensity"]):
        if intensity < cutoff:
            right_bound = float(rt)
            break

    if right_bound == apex_rt and len(right_df) > 0:
        right_bound = float(right_df["rt"][-1])

    # plot XIC with the margins
    # plot_XIC_with_margins(df_sorted, output_dir=output_dir, adapted_interval=(left_bound, right_bound), apex_rt=apex_rt, cutoff=cutoff)

    return float(left_bound), float(right_bound), float(apex_rt)


def _normalize_rt_values(rt_values: Optional[List[float]]) -> List[float]:
    """Return sorted unique finite RT values as Python floats."""
    if not rt_values:
        return []

    return sorted(
        {float(rt) for rt in rt_values if rt is not None and not np.isnan(float(rt))}
    )


def _extract_ms2_rt_values(
    ms2_dict: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[np.ndarray]:
    """Extract the global sorted MS2 RT grid from the spectra dictionary."""
    if not ms2_dict:
        return None

    rt_values = sorted(
        {
            float(values["retention_time"])
            for values in ms2_dict.values()
            if values.get("retention_time") is not None
            and not np.isnan(float(values["retention_time"]))
        }
    )
    if not rt_values:
        return None

    return np.asarray(rt_values, dtype=np.float64)


def _select_zero_fill_rt_values(
    peptidoform_rt_values: Optional[List[float]],
    global_ms2_rt_values: Optional[np.ndarray] = None,
) -> List[float]:
    """Select RT positions that should be explicitly zero-filled for one peptidoform."""
    local_rt_values = _normalize_rt_values(peptidoform_rt_values)
    if not local_rt_values:
        return []

    if global_ms2_rt_values is None or global_ms2_rt_values.size == 0:
        return local_rt_values

    lower_rt = local_rt_values[0]
    upper_rt = local_rt_values[-1]
    start_idx = int(np.searchsorted(global_ms2_rt_values, lower_rt, side="left"))
    end_idx = int(np.searchsorted(global_ms2_rt_values, upper_rt, side="right"))

    global_slice = global_ms2_rt_values[start_idx:end_idx]
    if global_slice.size == 0:
        return local_rt_values

    combined = np.unique(
        np.concatenate(
            [
                np.asarray(local_rt_values, dtype=np.float64),
                global_slice,
            ]
        )
    )
    return combined.astype(np.float64).tolist()


def _empty_predicted_fragment_apex_table(df_psms: pl.DataFrame) -> pl.DataFrame:
    """Return an empty predicted-apex table while preserving downstream schema."""
    result = df_psms.head(0)

    extra_columns = []
    if "annotated_fragment_count" not in result.columns:
        extra_columns.append(
            pl.lit(None).cast(pl.UInt32).alias("annotated_fragment_count")
        )
    if "max_annotated_fragment_intensity" not in result.columns:
        extra_columns.append(
            pl.lit(None).cast(pl.Float64).alias("max_annotated_fragment_intensity")
        )
    if "predicted_fragment_apex_score" not in result.columns:
        extra_columns.append(
            pl.lit(None).cast(pl.Float64).alias("predicted_fragment_apex_score")
        )
    if "matched_predicted_fragments" not in result.columns:
        extra_columns.append(
            pl.lit(None).cast(pl.UInt32).alias("matched_predicted_fragments")
        )
    if "max_matched_predicted_fragment_intensity" not in result.columns:
        extra_columns.append(
            pl.lit(None)
            .cast(pl.Float64)
            .alias("max_matched_predicted_fragment_intensity")
        )
    if "highest_predicted_fragment_observed_intensity" not in result.columns:
        extra_columns.append(
            pl.lit(None)
            .cast(pl.Float64)
            .alias("highest_predicted_fragment_observed_intensity")
        )
    if "predicted_rt_anchor" not in result.columns:
        extra_columns.append(pl.lit(None).cast(pl.Float64).alias("predicted_rt_anchor"))
    if "predicted_rt_distance" not in result.columns:
        extra_columns.append(
            pl.lit(None).cast(pl.Float64).alias("predicted_rt_distance")
        )
    if "min_annotated_count_required" not in result.columns:
        extra_columns.append(
            pl.lit(None).cast(pl.Int32).alias("min_annotated_count_required")
        )
    if "top_predicted_fragments" not in result.columns:
        extra_columns.append(
            pl.lit(None).cast(pl.Utf8).alias("top_predicted_fragments")
        )
    if "highest_predicted_fragment" not in result.columns:
        extra_columns.append(
            pl.lit(None).cast(pl.Utf8).alias("highest_predicted_fragment")
        )
    if "primary_window_id" not in result.columns:
        extra_columns.append(pl.lit(None).cast(pl.Utf8).alias("primary_window_id"))
    if "apex_window_id" not in result.columns:
        extra_columns.append(pl.lit(None).cast(pl.Utf8).alias("apex_window_id"))

    if extra_columns:
        result = result.with_columns(extra_columns)
    return result


def _clear_pin_output(output_dir: str) -> None:
    """Overwrite the PIN file with an empty file so stale results are not reused."""
    pin_path = os.path.join(output_dir, "outfile.pin")
    with open(pin_path, "w", encoding="utf-8"):
        pass


def _get_predicted_apex_lookup(
    df_fragment_max_peptide: Optional[pl.DataFrame],
    preferred_fragment_count: int = 2,
) -> Dict:
    """Build a lookup for predicted apex RT and preferred fragments per peptidoform."""
    if df_fragment_max_peptide is None or df_fragment_max_peptide.is_empty():
        return {}

    lookup = {}
    for row in df_fragment_max_peptide.to_dicts():
        fragments_raw = row.get("top_predicted_fragments", "")
        preferred_fragments = [f for f in str(fragments_raw).split(";") if f][
            :preferred_fragment_count
        ]
        lookup[(row["peptide"], row["charge"])] = {
            "apex_rt": float(row["rt"]) if row.get("rt") is not None else np.nan,
            "preferred_fragments": preferred_fragments,
            "primary_window_id": row.get("primary_window_id"),
            "apex_window_id": row.get("apex_window_id"),
        }
    return lookup


def _get_window_filtered_rt_values(
    df_psms: pl.DataFrame,
    peptide: str,
    charge: int,
    apex_info: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Return RT support for one peptidoform filtered to its selected DIA window."""
    df_sub = df_psms.filter(
        (pl.col("peptide") == peptide) & (pl.col("charge") == charge)
    )
    if df_sub.is_empty():
        return []
    primary_window_id = (
        None if apex_info is None else apex_info.get("primary_window_id")
    )
    df_sub = _filter_frame_to_window(df_sub, primary_window_id)
    return df_sub["rt"].drop_nulls().to_list() if "rt" in df_sub.columns else []


def calculate_min_max_margins(
    df_psms: pl.DataFrame,
    df_fragments: pl.DataFrame,
    df_fragment_max_peptide: Optional[pl.DataFrame] = None,
    top_n: int = 100,
    intensity_threshold: float = 0.01,
    preferred_fragment_count: int = 2,
    global_ms2_rt_values: Optional[np.ndarray] = None,
) -> tuple[float, float]:
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

    predicted_apex_lookup = _get_predicted_apex_lookup(
        df_fragment_max_peptide,
        preferred_fragment_count=preferred_fragment_count,
    )
    for (peptidoform, charge), df_fragments_top100_sub in tqdm(
        df_fragments_top100.group_by(["peptide", "charge"])
    ):
        apex_info = predicted_apex_lookup.get((peptidoform, charge), {})
        df_fragments_top100_sub = _filter_frame_to_window(
            df_fragments_top100_sub,
            apex_info.get("primary_window_id"),
        )
        left_bound, right_bound, apex_rt = calculate_rt_margins_intensity_based(
            df_fragments_top100_sub,
            intensity_threshold,
            output_dir="debug/calibration_xics",
            apex_rt=apex_info.get("apex_rt"),
            preferred_fragments=apex_info.get("preferred_fragments"),
            all_rt_values=_select_zero_fill_rt_values(
                _get_window_filtered_rt_values(
                    df_psms,
                    peptidoform,
                    charge,
                    apex_info,
                ),
                global_ms2_rt_values,
            ),
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
    if diffs:
        plot_rt_margin_histogram(
            diffs,
            output_dir="debug/calibration_xics",
            min_diff=min_diff,
            max_diff=max_diff,
        )

    return min_diff, max_diff


def add_retention_time_margins(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    min_diff: float,
    max_diff: float,
    intensity_threshold: float,
    df_fragment_max_peptide: Optional[pl.DataFrame] = None,
    preferred_fragment_count: int = 2,
    global_ms2_rt_values: Optional[np.ndarray] = None,
    margin_mode: str = "adaptive_per_peptidoform",
) -> pl.DataFrame:
    """
    Add retention time margin features to the PSM DataFrame.
    """

    pept2lowermargins = {}
    pept2highermargins = {}

    if margin_mode == "global_top_n":
        log_info(
            "Applying global RT margins derived from top-N calibration to all peptidoforms"
        )
    else:
        log_info(
            "Calculating adapted retention time margins based on intensity for all peptides"
        )

    predicted_apex_lookup = _get_predicted_apex_lookup(
        df_fragment_max_peptide,
        preferred_fragment_count=preferred_fragment_count,
    )
    for (peptidoform, charge), df_fragments_sub in tqdm(
        df_fragment.group_by(["peptide", "charge"])
    ):
        # speed up: skip peptidoforms with only 1 PSM
        if df_fragments_sub["psm_id"].n_unique() < 2:
            pept2lowermargins[(peptidoform, charge)] = np.nan
            pept2highermargins[(peptidoform, charge)] = np.nan
            continue

        apex_info = predicted_apex_lookup.get((peptidoform, charge), {})
        df_fragments_sub = _filter_frame_to_window(
            df_fragments_sub,
            apex_info.get("primary_window_id"),
        )
        if margin_mode == "global_top_n":
            apex_rt = apex_info.get("apex_rt")
            if apex_rt is None or np.isnan(apex_rt):
                rt_support = _get_window_filtered_rt_values(
                    df_psms,
                    peptidoform,
                    charge,
                    apex_info,
                )
                apex_rt = float(np.median(rt_support)) if rt_support else np.nan

            if np.isnan(apex_rt):
                pept2lowermargins[(peptidoform, charge)] = np.nan
                pept2highermargins[(peptidoform, charge)] = np.nan
                continue

            left_bound = float(apex_rt - max_diff)
            right_bound = float(apex_rt + max_diff)
        else:
            intensity_based_margins = calculate_rt_margins_intensity_based(
                df_fragments_sub,
                intensity_threshold,
                output_dir="xics",
                apex_rt=apex_info.get("apex_rt"),
                preferred_fragments=apex_info.get("preferred_fragments"),
                all_rt_values=_select_zero_fill_rt_values(
                    _get_window_filtered_rt_values(
                        df_psms,
                        peptidoform,
                        charge,
                        apex_info,
                    ),
                    global_ms2_rt_values,
                ),
            )
            left_bound, right_bound, apex_rt = intensity_based_margins

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
    df_fragment_max_peptide: Optional[pl.DataFrame] = None,
    top_n: int = 100,
    intensity_threshold: float = 0.05,
    preferred_fragment_count: int = 2,
    global_ms2_rt_values: Optional[np.ndarray] = None,
    margin_mode: str = "adaptive_per_peptidoform",
) -> pl.DataFrame:
    """
    Add retention time margin features to the PSM DataFrame.
    """
    if margin_mode not in {"adaptive_per_peptidoform", "global_top_n"}:
        log_info(
            f"Unknown rt_margin_mode '{margin_mode}', falling back to adaptive_per_peptidoform"
        )
        margin_mode = "adaptive_per_peptidoform"

    log_info("Calculating min max retention time margins based on intensity...")
    # Step 1: Calculate min and max retention time window based on top 100 peptidoforms
    min_diff, max_diff = calculate_min_max_margins(
        df_psms,
        df_fragment,
        df_fragment_max_peptide,
        top_n,
        intensity_threshold,
        preferred_fragment_count,
        global_ms2_rt_values,
    )

    log_info(
        f"RT margin mode: {margin_mode} (top_n={top_n}, min_diff={min_diff:.4f}, max_diff={max_diff:.4f})"
    )

    # Step 2: Calculate adapted margins for each PSM based on the intensity of the most intense fragment
    # and use the retention time distribution as min and max
    log_info("Adding retention time margin features to PSM DataFrame...")
    df_psms, df_fragment = add_retention_time_margins(
        df_psms,
        df_fragment,
        min_diff,
        max_diff,
        intensity_threshold,
        df_fragment_max_peptide,
        preferred_fragment_count,
        global_ms2_rt_values,
        margin_mode,
    )

    return df_psms, df_fragment


def dump_rt_margin_inputs(
    df_psms: pl.DataFrame,
    df_fragment_search: pl.DataFrame,
    df_fragment_for_margins: pl.DataFrame,
    df_fragment_max_peptide: Optional[pl.DataFrame],
    dump_dir: str,
    fragment_table_used_for_margin_calculation: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    global_ms2_rt_values: Optional[np.ndarray] = None,
) -> None:
    """Write the exact inputs used by the RT-margin step for later replay."""
    os.makedirs(dump_dir, exist_ok=True)

    df_psms.write_csv(os.path.join(dump_dir, "df_psms_pre_margin.tsv"), separator="\t")
    df_fragment_search.write_csv(
        os.path.join(dump_dir, "df_fragment_search_pre_margin.tsv"), separator="\t"
    )
    df_fragment_for_margins.write_csv(
        os.path.join(dump_dir, "df_fragment_reannotated_pre_margin.tsv"),
        separator="\t",
    )

    if df_fragment_max_peptide is not None and not df_fragment_max_peptide.is_empty():
        df_fragment_max_peptide.write_csv(
            os.path.join(dump_dir, "df_fragment_max_peptide_pre_margin.tsv"),
            separator="\t",
        )

    if global_ms2_rt_values is not None and global_ms2_rt_values.size > 0:
        pl.DataFrame({"rt": global_ms2_rt_values.tolist()}).write_csv(
            os.path.join(dump_dir, "global_ms2_rt_values.tsv"),
            separator="\t",
        )

    fragment_table_used = fragment_table_used_for_margin_calculation
    if fragment_table_used is None and metadata is not None:
        fragment_table_used = metadata.get("fragment_table_used_for_margin_calculation")
    if fragment_table_used is None:
        fragment_table_used = "df_fragment_reannotated_pre_margin.tsv"

    summary = {
        "df_psms_shape": list(df_psms.shape),
        "df_fragment_search_shape": list(df_fragment_search.shape),
        "df_fragment_reannotated_shape": list(df_fragment_for_margins.shape),
        "df_fragment_max_peptide_shape": (
            list(df_fragment_max_peptide.shape)
            if df_fragment_max_peptide is not None
            else None
        ),
        "fragment_table_used_for_margin_calculation": fragment_table_used,
        "global_ms2_rt_values_present": bool(
            global_ms2_rt_values is not None and global_ms2_rt_values.size > 0
        ),
    }
    if metadata:
        summary.update(metadata)

    with open(os.path.join(dump_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def dump_stage3_feature_inputs(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    feature_fragment_df: pl.DataFrame,
    df_fragment_max_peptide: pl.DataFrame,
    ms2pip_predictions: Dict[str, Dict[str, float]],
    theoretical_fragment_context: Dict[str, Dict[str, Any]],
    preannotated_fragment_dict: Dict[str, pl.DataFrame],
    config: Dict[str, Any],
    spectra_data: Optional[SpectraData],
    pickle_config: Optional[PickleConfig],
    dump_dir: str,
) -> None:
    """Persist the late Stage 3 inputs so feature generation can be replayed."""
    os.makedirs(dump_dir, exist_ok=True)

    df_psms.write_csv(
        os.path.join(dump_dir, "df_psms_pre_feature_calc.tsv"), separator="\t"
    )
    df_fragment.write_csv(
        os.path.join(dump_dir, "df_fragment_pre_feature_calc.tsv"), separator="\t"
    )
    feature_fragment_df.write_csv(
        os.path.join(dump_dir, "feature_fragment_df_pre_feature_calc.tsv"),
        separator="\t",
    )
    df_fragment_max_peptide.write_csv(
        os.path.join(dump_dir, "df_fragment_max_peptide_pre_feature_calc.tsv"),
        separator="\t",
    )

    with open(os.path.join(dump_dir, "ms2pip_predictions.pkl"), "wb") as handle:
        pickle.dump(ms2pip_predictions, handle)
    with open(
        os.path.join(dump_dir, "theoretical_fragment_context.pkl"), "wb"
    ) as handle:
        pickle.dump(theoretical_fragment_context, handle)
    with open(os.path.join(dump_dir, "preannotated_fragment_dict.pkl"), "wb") as handle:
        pickle.dump(preannotated_fragment_dict, handle)
    with open(os.path.join(dump_dir, "config.pkl"), "wb") as handle:
        pickle.dump(config, handle)

    ms1_dict = {} if spectra_data is None else spectra_data.ms1_dict
    ms2_to_ms1_dict = {} if spectra_data is None else spectra_data.ms2_to_ms1_dict
    ms2_dict = {} if spectra_data is None else spectra_data.ms2_dict
    with open(os.path.join(dump_dir, "ms1_dict.pkl"), "wb") as handle:
        pickle.dump(ms1_dict, handle)
    with open(os.path.join(dump_dir, "ms2_to_ms1_dict.pkl"), "wb") as handle:
        pickle.dump(ms2_to_ms1_dict, handle)
    with open(os.path.join(dump_dir, "ms2_dict.pkl"), "wb") as handle:
        pickle.dump(ms2_dict, handle)

    summary = {
        "df_psms_shape": list(df_psms.shape),
        "df_fragment_shape": list(df_fragment.shape),
        "feature_fragment_df_shape": list(feature_fragment_df.shape),
        "df_fragment_max_peptide_shape": list(df_fragment_max_peptide.shape),
        "ms2pip_prediction_count": len(ms2pip_predictions),
        "theoretical_fragment_context_count": len(theoretical_fragment_context),
        "preannotated_fragment_dict_count": len(preannotated_fragment_dict),
        "ms1_scan_count": len(ms1_dict),
        "ms2_scan_count": len(ms2_dict),
        "ms2_to_ms1_count": len(ms2_to_ms1_dict),
        "read_correlation_pickles": (
            False if pickle_config is None else bool(pickle_config.read_correlation)
        ),
        "write_correlation_pickles": (
            False if pickle_config is None else bool(pickle_config.write_correlation)
        ),
        "result_dir": config.get("mumdia", {}).get("result_dir"),
        "notes": "Inputs captured immediately before late Stage 3 feature generation.",
    }
    with open(os.path.join(dump_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def ensure_fragment_name_column(df_fragment: pl.DataFrame) -> pl.DataFrame:
    """Ensure the fragment table has a canonical `fragment_name` column."""
    if "fragment_name" in df_fragment.columns:
        return df_fragment

    return df_fragment.with_columns(
        (
            pl.col("fragment_type")
            + pl.col("fragment_ordinals").cast(pl.Utf8)
            + "/"
            + pl.col("fragment_charge").cast(pl.Utf8)
        ).alias("fragment_name")
    )


def _safe_float(value: Any) -> Optional[float]:
    """Convert numeric-like values to finite floats."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result):
        return None
    return result


def _window_id_from_bounds(
    iso_lower_mz: Optional[float], iso_upper_mz: Optional[float], decimals: int = 4
) -> Optional[str]:
    """Build a stable DIA window identifier from absolute isolation bounds."""
    if iso_lower_mz is None or iso_upper_mz is None:
        return None
    return f"{iso_lower_mz:.{decimals}f}|{iso_upper_mz:.{decimals}f}"


def _extract_scan_window_metadata(scan_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract absolute DIA window bounds and a stable window id for one MS2 scan."""
    iso_target = _safe_float(scan_entry.get("isolation_window_target"))
    iso_lower_offset = _safe_float(scan_entry.get("isolation_window_lower"))
    iso_upper_offset = _safe_float(scan_entry.get("isolation_window_upper"))

    iso_lower_mz = None
    iso_upper_mz = None
    if (
        iso_target is not None
        and iso_lower_offset is not None
        and iso_upper_offset is not None
    ):
        iso_lower_mz = iso_target - iso_lower_offset
        iso_upper_mz = iso_target + iso_upper_offset

    return {
        "isolation_window_target": iso_target,
        "isolation_window_lower": iso_lower_offset,
        "isolation_window_upper": iso_upper_offset,
        "iso_lower_mz": iso_lower_mz,
        "iso_upper_mz": iso_upper_mz,
        "window_id": _window_id_from_bounds(iso_lower_mz, iso_upper_mz),
    }


def _scan_entry_matches_precursor_mz(
    scan_entry: Dict[str, Any], precursor_mz: Optional[float]
) -> Optional[bool]:
    """Return whether a precursor m/z falls inside a scan's DIA window."""
    if precursor_mz is None:
        return None
    window_meta = _extract_scan_window_metadata(scan_entry)
    iso_lower_mz = window_meta.get("iso_lower_mz")
    iso_upper_mz = window_meta.get("iso_upper_mz")
    if iso_lower_mz is None or iso_upper_mz is None:
        return None
    return bool(iso_lower_mz <= precursor_mz <= iso_upper_mz)


def build_ms2_scan_window_metadata(ms2_dict: Dict[str, Dict[str, Any]]) -> pl.DataFrame:
    """Build a per-scan DIA isolation metadata table from the parsed mzML dict."""
    if not ms2_dict:
        return pl.DataFrame()

    rows = []
    for scan_id, values in ms2_dict.items():
        scan_meta = _extract_scan_window_metadata(values)
        rows.append(
            {
                "scan_id": str(scan_id),
                "scan_rt": _safe_float(values.get("retention_time")),
                **scan_meta,
            }
        )

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows)


def _precursor_window_compatibility_expr() -> pl.Expr:
    """Polars expression for precursor-vs-window compatibility."""
    return (
        pl.when(
            pl.col("precursor_mz").is_not_null()
            & pl.col("iso_lower_mz").is_not_null()
            & pl.col("iso_upper_mz").is_not_null()
        )
        .then(
            (pl.col("precursor_mz") >= pl.col("iso_lower_mz"))
            & (pl.col("precursor_mz") <= pl.col("iso_upper_mz"))
        )
        .otherwise(None)
        .alias("precursor_in_window")
    )


def attach_ms2_window_metadata(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    spectra_data: Optional[SpectraData],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Attach DIA isolation-window metadata to Stage 3 PSM and fragment tables."""
    if spectra_data is None or not getattr(spectra_data, "ms2_dict", None):
        return df_psms, df_fragment, pl.DataFrame()

    ms2_scan_metadata = build_ms2_scan_window_metadata(spectra_data.ms2_dict)
    spectra_data.ms2_scan_metadata = ms2_scan_metadata
    if ms2_scan_metadata.is_empty() or "scannr" not in df_psms.columns:
        return df_psms, df_fragment, ms2_scan_metadata

    psm_scan_join = ms2_scan_metadata.rename({"scan_id": "_scan_id"})
    df_psms = (
        df_psms.with_columns(pl.col("scannr").cast(pl.Utf8).alias("_scan_id"))
        .join(psm_scan_join, on="_scan_id", how="left")
        .drop("_scan_id")
        .with_columns(
            [
                (
                    pl.when(
                        pl.col("calcmass").is_not_null()
                        & pl.col("charge").is_not_null()
                    )
                    .then(
                        pl.col("calcmass") / pl.col("charge").cast(pl.Float64)
                        + PROTON_MASS
                    )
                    .otherwise(None)
                    .alias("precursor_mz")
                ),
            ]
        )
        .with_columns(_precursor_window_compatibility_expr())
    )

    fragment_join_cols = ["psm_id"] + [
        col
        for col in [
            "scannr",
            "scan_rt",
            "isolation_window_target",
            "isolation_window_lower",
            "isolation_window_upper",
            "iso_lower_mz",
            "iso_upper_mz",
            "window_id",
            "precursor_mz",
            "precursor_in_window",
        ]
        if col in df_psms.columns and col not in df_fragment.columns
    ]
    if fragment_join_cols:
        df_fragment = df_fragment.join(
            df_psms.select(fragment_join_cols).unique(
                subset=["psm_id"], maintain_order=True
            ),
            on="psm_id",
            how="left",
        )

    return df_psms, df_fragment, ms2_scan_metadata


def _filter_frame_to_window(
    df_sub: pl.DataFrame,
    primary_window_id: Optional[str] = None,
    *,
    require_precursor_match: bool = True,
) -> pl.DataFrame:
    """Restrict a peptidoform-level frame to the selected DIA window when possible."""
    if df_sub.is_empty():
        return df_sub

    if primary_window_id and "window_id" in df_sub.columns:
        filtered = df_sub.filter(pl.col("window_id") == primary_window_id)
        if require_precursor_match and "precursor_in_window" in filtered.columns:
            matched = filtered.filter(pl.col("precursor_in_window").fill_null(False))
            if not matched.is_empty():
                return matched
        if not filtered.is_empty():
            return filtered

    if require_precursor_match and "precursor_in_window" in df_sub.columns:
        matched = df_sub.filter(pl.col("precursor_in_window").fill_null(False))
        if not matched.is_empty():
            return matched

    return df_sub


def _select_primary_window_id(
    df_psms_sub: pl.DataFrame, anchor_rt: Optional[float] = None
) -> Optional[str]:
    """Pick the dominant DIA window for one peptidoform using count first, RT proximity second."""
    if df_psms_sub.is_empty() or "window_id" not in df_psms_sub.columns:
        return None

    candidate_df = _filter_frame_to_window(
        df_psms_sub, None, require_precursor_match=True
    )
    candidate_df = candidate_df.filter(pl.col("window_id").is_not_null())
    if candidate_df.is_empty():
        return None

    summary = candidate_df.group_by("window_id").agg(
        [
            pl.len().alias("window_scan_count"),
            pl.median("rt").alias("window_rt_median"),
        ]
    )
    if anchor_rt is None or np.isnan(float(anchor_rt)):
        anchor_rt = float(candidate_df["rt"].median())

    summary = summary.with_columns(
        (pl.col("window_rt_median") - float(anchor_rt))
        .abs()
        .alias("window_anchor_distance")
    ).sort(
        by=["window_scan_count", "window_anchor_distance", "window_id"],
        descending=[True, False, False],
    )
    if summary.is_empty():
        return None
    return str(summary["window_id"][0])


def build_theoretical_fragment_context(
    df_psms: pl.DataFrame,
    ms2pip_predictions: Dict[str, Dict[str, float]],
    max_fragment_charge_cap: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """Build a shared theoretical fragment context per peptidoform."""
    fragment_context: Dict[str, Dict[str, Any]] = {}

    try:
        from rustyms import CompoundPeptidoformIon, FragmentationModel
    except ImportError:
        CompoundPeptidoformIon = None
        FragmentationModel = None

    import re as _re

    for (peptide, charge), df_psms_sub in tqdm(
        df_psms.group_by(["peptide", "charge"]), desc="Building fragment context"
    ):
        key = f"{peptide}/{charge}"
        preds = ms2pip_predictions.get(key, {})
        charge_int = int(charge)

        if "rt_predictions" in df_psms_sub.columns:
            rt_predictions = df_psms_sub["rt_predictions"].drop_nulls()
            predicted_rt = (
                float(rt_predictions.median())
                if len(rt_predictions) > 0
                else float(df_psms_sub["rt"].median())
            )
        else:
            predicted_rt = float(df_psms_sub["rt"].median())

        calcmass_vals = df_psms_sub["calcmass"].drop_nulls()
        precursor_mz = None
        if len(calcmass_vals) > 0:
            precursor_mz = float(calcmass_vals[0]) / charge_int + 1.007276466812

        top_predicted_fragments = [
            fragment_name
            for fragment_name, _ in sorted(
                preds.items(), key=lambda item: item[1], reverse=True
            )
        ]

        named_frags = []
        if CompoundPeptidoformIon is not None and FragmentationModel is not None:
            try:
                pep_ion = CompoundPeptidoformIon(key)
                max_frag_charge = min(charge_int, max_fragment_charge_cap)
                theo_frags = pep_ion.generate_theoretical_fragments(
                    max_frag_charge, FragmentationModel.CidHcd
                )
                for frag in theo_frags:
                    frag_repr = repr(frag)
                    if (
                        ("ion='b" in frag_repr or "ion='y" in frag_repr)
                        and "H2O" not in frag_repr
                        and "NH3" not in frag_repr
                    ):
                        ion_m = _re.search(r"ion='([^']*)'", frag_repr)
                        ch_m = _re.search(r"charge=(\d+),", frag_repr)
                        if ion_m and ch_m:
                            frag_name = f"{ion_m.group(1)}/{ch_m.group(1)}"
                            frag_mz = frag.formula.monoisotopic_mass() / frag.charge
                            if 100 <= frag_mz <= 2500:
                                named_frags.append((frag_name, frag_mz))
            except Exception:
                named_frags = []

        named_frags.sort(key=lambda item: item[1])
        theoretical_mz_by_name = {name: mz for name, mz in named_frags}

        fragment_context[key] = {
            "predicted_rt": predicted_rt,
            "precursor_mz": precursor_mz,
            "predicted_intensities": preds,
            "top_predicted_fragments": top_predicted_fragments,
            "theoretical_fragments": named_frags,
            "theoretical_fragment_mz": theoretical_mz_by_name,
        }

    return fragment_context


def annotate_candidate_fragment_window(
    df_fragment: pl.DataFrame,
    fragment_context: Dict[str, Dict[str, Any]],
    top_n_predicted_fragments: int = 2,
) -> pl.DataFrame:
    """Annotate Sage fragment rows with shared theoretical/predicted context."""
    if df_fragment.is_empty():
        return df_fragment

    df_fragment = ensure_fragment_name_column(df_fragment)
    annotation_rows = []
    for key, context in fragment_context.items():
        peptide, charge_str = key.rsplit("/", 1)
        charge = int(charge_str)
        top_set = set(
            context.get("top_predicted_fragments", [])[:top_n_predicted_fragments]
        )
        pred_map = context.get("predicted_intensities", {})
        theo_map = context.get("theoretical_fragment_mz", {})
        annotation_rows.extend(
            {
                "peptide": peptide,
                "charge": charge,
                "fragment_name": fragment_name,
                "predicted_fragment_intensity": float(pred_map.get(fragment_name, 0.0)),
                "theoretical_fragment_mz": float(theo_map.get(fragment_name, np.nan)),
                "is_top_predicted_fragment": fragment_name in top_set,
                "is_theoretical_fragment": fragment_name in theo_map,
                "predicted_rt_anchor": float(context.get("predicted_rt", np.nan)),
            }
            for fragment_name in set(list(pred_map.keys()) + list(theo_map.keys()))
        )

    if len(annotation_rows) == 0:
        return df_fragment

    df_annotation = pl.DataFrame(annotation_rows)
    return df_fragment.join(
        df_annotation,
        on=["peptide", "charge", "fragment_name"],
        how="left",
    ).with_columns(
        [
            pl.col("predicted_fragment_intensity").fill_null(0.0),
            pl.col("is_top_predicted_fragment").fill_null(False),
            pl.col("is_theoretical_fragment").fill_null(False),
        ]
    )


def reannotate_candidate_ms2_spectra(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    fragment_context: Dict[str, Dict[str, Any]],
    ms2_dict: Dict[str, Dict[str, Any]],
    tolerance_ppm: float = 13.0,
) -> Dict[str, pl.DataFrame]:
    """Reannotate candidate-window MS2 spectra early and share the results downstream."""
    if df_psms.is_empty() or df_fragment.is_empty() or not ms2_dict:
        return {}

    try:
        from rustyms import (
            CompoundPeptidoformIon,
            FragmentationModel,
            MassMode,
            MatchingParameters,
            RawSpectrum,
        )
    except ImportError:
        return {}

    import re as _re

    ion_pattern = _re.compile(r"ion='([^']*)'")
    charge_pattern = _re.compile(r"charge=(\d+),")

    df_fragment = ensure_fragment_name_column(df_fragment)
    # Drop any duplicate scannr columns introduced by earlier joins (e.g. scannr_right)
    if "scannr_right" in df_fragment.columns:
        df_fragment = df_fragment.drop("scannr_right")
    psm_lookup = {}
    for row in df_psms.select(
        ["psm_id", "scannr", "peptide", "charge", "rt"]
    ).to_dicts():
        psm_lookup[int(row["psm_id"])] = row

    preannotated_fragment_dict: Dict[str, pl.DataFrame] = {}
    matching_parameters = MatchingParameters()
    matching_parameters.tolerance_ppm = tolerance_ppm
    annotation_cache: Dict[str, Optional[tuple[str, int]]] = {}
    fragment_groups = {
        f"{peptide}/{charge}": df_sub
        for (peptide, charge), df_sub in df_fragment.group_by(["peptide", "charge"])
    }

    for key, context in tqdm(
        fragment_context.items(), desc="Early candidate-window reannotation"
    ):
        peptide, charge_str = key.rsplit("/", 1)
        charge = int(charge_str)
        df_sub = fragment_groups.get(key)
        if df_sub is None:
            continue
        if df_sub.is_empty():
            continue

        if "rt" not in df_sub.columns:
            log_info(
                f"WARNING: 'rt' column missing from df_sub for key '{key}' "
                f"(columns: {df_sub.columns}); skipping."
            )
            continue

        rt_max_candidates = df_sub["rt"].drop_nulls()
        if len(rt_max_candidates) == 0:
            continue

        apex_rt = float(rt_max_candidates.median())
        precursor = key
        linear_peptide = CompoundPeptidoformIon(precursor)
        top_predicted_set = set(context.get("top_predicted_fragments", [])[:6])
        theoretical_fragment_mz = context.get("theoretical_fragment_mz", {})
        predicted_fragment_intensities = context.get("predicted_intensities", {})
        predicted_rt_anchor = float(context.get("predicted_rt", np.nan))
        precursor_mz = _safe_float(context.get("precursor_mz"))
        allow_offwindow_fallback = True
        if "precursor_in_window" in df_sub.columns:
            allow_offwindow_fallback = not bool(
                df_sub["precursor_in_window"].fill_null(False).any()
            )
        records = []
        unique_psm_rows = df_sub.select(["psm_id", "rt"]).unique(
            subset=["psm_id"], maintain_order=True
        )

        for row in unique_psm_rows.to_dicts():
            psm_id = int(row["psm_id"])
            psm_meta = psm_lookup.get(psm_id)
            if psm_meta is None:
                continue
            scannr = psm_meta["scannr"]
            if scannr not in ms2_dict:
                continue
            scan_entry = ms2_dict[scannr]
            precursor_matches_window = _scan_entry_matches_precursor_mz(
                scan_entry, precursor_mz
            )
            if precursor_matches_window is False and not allow_offwindow_fallback:
                continue
            scan_window_meta = _extract_scan_window_metadata(scan_entry)
            try:
                spectrum = RawSpectrum(
                    title=scannr,
                    num_scans=1,
                    rt=float(psm_meta["rt"]),
                    precursor_charge=charge,
                    precursor_mass=1.0,
                    mz_array=scan_entry["mz"],
                    intensity_array=scan_entry["intensity"],
                )
                annotated_spectrum = spectrum.annotate(
                    peptidoform=linear_peptide,
                    parameters=matching_parameters,
                    model=FragmentationModel.CidHcd,
                    mode=MassMode.Monoisotopic,
                )

                for annotated_peak in annotated_spectrum.spectrum:
                    if not annotated_peak.annotation:
                        continue
                    frag_repr = repr(annotated_peak.annotation[0])
                    parsed_annotation = annotation_cache.get(frag_repr)
                    if parsed_annotation is None and frag_repr not in annotation_cache:
                        ion_m = ion_pattern.search(frag_repr)
                        ch_m = charge_pattern.search(frag_repr)
                        if ion_m and ch_m:
                            parsed_annotation = (ion_m.group(1), int(ch_m.group(1)))
                        annotation_cache[frag_repr] = parsed_annotation
                    if parsed_annotation is None:
                        continue
                    ion_label, ion_charge = parsed_annotation
                    if not (ion_label.startswith("b") or ion_label.startswith("y")):
                        continue
                    if ion_charge != 1:
                        continue
                    fragment_name = f"{ion_label}/{ion_charge}"
                    records.append(
                        {
                            "psm_id": psm_id,
                            "fragment_type": ion_label[0],
                            "fragment_ordinals": ion_label[1:],
                            "fragment_charge": ion_charge,
                            "fragment_intensity": float(annotated_peak.intensity),
                            "fragment_mz": float(annotated_peak.experimental_mz),
                            "rt": float(psm_meta["rt"]),
                            "scannr": scannr,
                            "fragment_name": fragment_name,
                            "rt_max_peptide_sub": apex_rt,
                            "precursor": precursor,
                            "charge": charge,
                            "peptide": peptide,
                            "theoretical_fragment_mz": float(
                                theoretical_fragment_mz.get(fragment_name, np.nan)
                            ),
                            "predicted_fragment_intensity": float(
                                predicted_fragment_intensities.get(fragment_name, 0.0)
                            ),
                            "is_top_predicted_fragment": fragment_name
                            in top_predicted_set,
                            "predicted_rt_anchor": predicted_rt_anchor,
                            "precursor_mz": precursor_mz,
                            "precursor_in_window": precursor_matches_window,
                            **scan_window_meta,
                        }
                    )
            except Exception:
                continue

        if records:
            preannotated_fragment_dict[key] = (
                pl.DataFrame(records)
                .sort("fragment_intensity", descending=True)
                .unique(subset=["psm_id", "fragment_name"], keep="first")
            )

    return preannotated_fragment_dict


def filter_preannotated_fragment_dict(
    preannotated_fragment_dict: Dict[str, pl.DataFrame], valid_psm_ids: set[int]
) -> Dict[str, pl.DataFrame]:
    """Restrict shared reannotated fragment tables to the currently valid PSM set."""
    if not preannotated_fragment_dict or not valid_psm_ids:
        return {}

    valid_psm_id_series = pl.Series(
        "psm_id",
        list(valid_psm_ids),
        dtype=pl.Int64,
    )

    filtered = {}
    for key, df_sub in preannotated_fragment_dict.items():
        df_filtered = df_sub.filter(pl.col("psm_id").is_in(valid_psm_id_series))
        if not df_filtered.is_empty():
            filtered[key] = df_filtered
    return filtered


def enrich_preannotated_fragment_dict(
    preannotated_fragment_dict: Dict[str, pl.DataFrame],
    df_psms: pl.DataFrame,
    df_fragment_max_peptide: Optional[pl.DataFrame] = None,
) -> Dict[str, pl.DataFrame]:
    """Attach current apex and RT window metadata to shared reannotated fragment tables."""
    if not preannotated_fragment_dict:
        return preannotated_fragment_dict

    margin_lookup = {}
    if not df_psms.is_empty() and {
        "peptide",
        "charge",
        "rt_lower_margin",
        "rt_higher_margin",
    }.issubset(df_psms.columns):
        for row in (
            df_psms.select(
                [
                    "peptide",
                    "charge",
                    "rt_lower_margin",
                    "rt_higher_margin",
                ]
            )
            .unique(subset=["peptide", "charge"], maintain_order=True)
            .to_dicts()
        ):
            margin_lookup[(row["peptide"], row["charge"])] = (
                row.get("rt_lower_margin"),
                row.get("rt_higher_margin"),
            )

    apex_lookup = {}
    if (
        df_fragment_max_peptide is not None
        and not df_fragment_max_peptide.is_empty()
        and {"peptide", "charge", "rt"}.issubset(df_fragment_max_peptide.columns)
    ):
        select_cols = [
            col
            for col in [
                "peptide",
                "charge",
                "rt",
                "primary_window_id",
                "apex_window_id",
            ]
            if col in df_fragment_max_peptide.columns
        ]
        for row in df_fragment_max_peptide.select(select_cols).to_dicts():
            apex_lookup[(row["peptide"], row["charge"])] = row

    enriched = {}
    for key, df_sub in preannotated_fragment_dict.items():
        peptide, charge_str = key.rsplit("/", 1)
        charge = int(charge_str)
        apex_info = apex_lookup.get((peptide, charge), {})
        apex_rt = apex_info.get("rt")
        lower_margin, upper_margin = margin_lookup.get((peptide, charge), (None, None))

        cols = []
        if apex_rt is not None:
            cols.append(pl.lit(float(apex_rt)).alias("rt_max_peptide_sub"))
        primary_window_id = apex_info.get("primary_window_id")
        if primary_window_id is not None:
            cols.append(pl.lit(str(primary_window_id)).alias("primary_window_id"))
        apex_window_id = apex_info.get("apex_window_id")
        if apex_window_id is not None:
            cols.append(pl.lit(str(apex_window_id)).alias("apex_window_id"))
        if lower_margin is not None:
            cols.append(pl.lit(float(lower_margin)).alias("rt_lower_margin"))
        if upper_margin is not None:
            cols.append(pl.lit(float(upper_margin)).alias("rt_higher_margin"))

        enriched[key] = df_sub.with_columns(cols) if cols else df_sub

    return enriched


def flatten_preannotated_fragment_dict(
    preannotated_fragment_dict: Dict[str, pl.DataFrame],
    fallback_df_fragment: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Flatten shared reannotated fragment tables into one DataFrame for feature generation."""
    frames = [df for df in preannotated_fragment_dict.values() if not df.is_empty()]
    if frames:
        return pl.concat(frames, how="diagonal_relaxed")
    if fallback_df_fragment is not None:
        return fallback_df_fragment
    return pl.DataFrame()


def build_predicted_fragment_apex_table(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    ms2pip_predictions: Dict[str, Dict[str, float]],
    top_n_predicted_fragments: int = 6,
    min_fraction_of_max_count: float = 0.9,
) -> pl.DataFrame:
    """
    Select one apex PSM per peptidoform using fragment-rich RT regions first.

    Fully vectorised Polars implementation — no Python loop over peptidoforms.

    Candidate scans are first restricted to those whose annotated fragment count
    is within `min_fraction_of_max_count` of the peptidoform's maximum annotated
    fragment count. Within that retained region, the apex is chosen using the
    observed intensity of the highest-predicted fragment.
    """
    if df_psms.is_empty():
        return _empty_predicted_fragment_apex_table(df_psms)

    df_fragment = ensure_fragment_name_column(df_fragment)
    charge_dtype = df_psms["charge"].dtype

    # ── 1. Predicted RT per peptidoform ──────────────────────────────────
    rt_src = "rt_predictions" if "rt_predictions" in df_psms.columns else "rt"
    predicted_rt_df = df_psms.group_by(["peptide", "charge"]).agg(
        pl.col(rt_src).drop_nulls().median().alias("predicted_rt_anchor")
    )

    # ── 2. Primary window per peptidoform (vectorised _select_primary_window_id) ──
    has_window = (
        "window_id" in df_psms.columns and "precursor_in_window" in df_psms.columns
    )
    if has_window:
        win_candidates = df_psms.filter(
            pl.col("precursor_in_window").fill_null(False)
            & pl.col("window_id").is_not_null()
        )
        if not win_candidates.is_empty():
            primary_window_df = (
                win_candidates.group_by(["peptide", "charge", "window_id"])
                .agg(
                    [
                        pl.len().alias("_wsc"),
                        pl.col("rt").median().alias("_wrt"),
                    ]
                )
                .join(predicted_rt_df, on=["peptide", "charge"], how="left")
                .with_columns(
                    (
                        pl.col("_wrt")
                        - pl.col("predicted_rt_anchor").fill_null(pl.col("_wrt"))
                    )
                    .abs()
                    .alias("_wdist")
                )
                .sort(
                    ["peptide", "charge", "_wsc", "_wdist", "window_id"],
                    descending=[False, False, True, False, False],
                )
                .unique(subset=["peptide", "charge"], keep="first", maintain_order=True)
                .select(["peptide", "charge", "window_id"])
                .rename({"window_id": "primary_window_id"})
            )
        else:
            primary_window_df = pl.DataFrame(
                schema={
                    "peptide": pl.Utf8,
                    "charge": charge_dtype,
                    "primary_window_id": pl.Utf8,
                }
            )
    else:
        primary_window_df = pl.DataFrame(
            schema={
                "peptide": pl.Utf8,
                "charge": charge_dtype,
                "primary_window_id": pl.Utf8,
            }
        )

    # ── 3. Flatten ms2pip_predictions dict → DataFrame ───────────────────
    pred_rows: List[Dict[str, Any]] = []
    top_frags_rows: List[Dict[str, Any]] = []
    for precursor_key, preds in ms2pip_predictions.items():
        if not preds:
            continue
        sep_idx = precursor_key.rfind("/")
        if sep_idx < 0:
            continue
        pep = precursor_key[:sep_idx]
        try:
            chg = int(precursor_key[sep_idx + 1 :])
        except ValueError:
            continue
        sorted_frags = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        top_frags = [f for f, _ in sorted_frags[:top_n_predicted_fragments]]
        top_frags_rows.append(
            {
                "peptide": pep,
                "charge": chg,
                "top_predicted_fragments": ";".join(top_frags),
                "highest_predicted_fragment": top_frags[0] if top_frags else None,
            }
        )
        for rank, (frag_name, frag_intens) in enumerate(
            sorted_frags[:top_n_predicted_fragments]
        ):
            pred_rows.append(
                {
                    "peptide": pep,
                    "charge": chg,
                    "fragment_name": frag_name,
                    "predicted_fragment_intensity": float(frag_intens),
                    "is_highest_pred_frag": rank == 0,
                }
            )

    if pred_rows:
        df_preds = pl.DataFrame(pred_rows).with_columns(
            pl.col("charge").cast(charge_dtype)
        )
        df_top_frags = pl.DataFrame(top_frags_rows).with_columns(
            pl.col("charge").cast(charge_dtype)
        )
    else:
        df_preds = pl.DataFrame(
            schema={
                "peptide": pl.Utf8,
                "charge": charge_dtype,
                "fragment_name": pl.Utf8,
                "predicted_fragment_intensity": pl.Float64,
                "is_highest_pred_frag": pl.Boolean,
            }
        )
        df_top_frags = pl.DataFrame(
            schema={
                "peptide": pl.Utf8,
                "charge": charge_dtype,
                "top_predicted_fragments": pl.Utf8,
                "highest_predicted_fragment": pl.Utf8,
            }
        )

    # ── 4. Filter df_fragment to primary window ───────────────────────────
    if "peptide" in df_fragment.columns and "charge" in df_fragment.columns:
        df_frag_w_charge = df_fragment.with_columns(pl.col("charge").cast(charge_dtype))
    else:
        df_frag_w_charge = df_fragment

    if (
        has_window
        and not primary_window_df.is_empty()
        and "peptide" in df_frag_w_charge.columns
    ):
        df_frag_win = (
            df_frag_w_charge.join(
                primary_window_df, on=["peptide", "charge"], how="left"
            )
            .filter(
                pl.col("primary_window_id").is_null()
                | (pl.col("window_id") == pl.col("primary_window_id"))
            )
            .drop("primary_window_id")
        )
    else:
        df_frag_win = df_frag_w_charge

    # ── 5. Fragment counts per (peptide, charge, psm_id, rt) ──────────────
    if not df_frag_win.is_empty() and "peptide" in df_frag_win.columns:
        fragment_counts = df_frag_win.group_by(
            ["peptide", "charge", "psm_id", "rt"]
        ).agg(
            [
                pl.n_unique("fragment_name").alias("annotated_fragment_count"),
                pl.col("fragment_intensity")
                .max()
                .alias("max_annotated_fragment_intensity"),
            ]
        )
    else:
        fragment_counts = pl.DataFrame(
            schema={
                "peptide": pl.Utf8,
                "charge": charge_dtype,
                "psm_id": pl.Int64,
                "rt": pl.Float64,
                "annotated_fragment_count": pl.UInt32,
                "max_annotated_fragment_intensity": pl.Float64,
            }
        )

    # ── 6. Score top-predicted fragments ──────────────────────────────────
    if (
        not df_frag_win.is_empty()
        and not df_preds.is_empty()
        and "peptide" in df_frag_win.columns
    ):
        df_frag_scored = df_frag_win.join(
            df_preds.select(
                ["peptide", "charge", "fragment_name", "predicted_fragment_intensity"]
            ),
            on=["peptide", "charge", "fragment_name"],
            how="inner",
        ).with_columns(
            (
                pl.col("fragment_intensity") * pl.col("predicted_fragment_intensity")
            ).alias("_wpfi")
        )
        pred_scores = df_frag_scored.group_by(
            ["peptide", "charge", "psm_id", "rt"]
        ).agg(
            [
                pl.col("_wpfi").sum().alias("predicted_fragment_apex_score"),
                pl.n_unique("fragment_name").alias("matched_predicted_fragments"),
                pl.col("fragment_intensity")
                .max()
                .alias("max_matched_predicted_fragment_intensity"),
            ]
        )
        highest_pred_frags = df_preds.filter(pl.col("is_highest_pred_frag")).select(
            ["peptide", "charge", "fragment_name"]
        )
        top_frag_scores = (
            df_frag_win.join(
                highest_pred_frags,
                on=["peptide", "charge", "fragment_name"],
                how="inner",
            )
            .group_by(["peptide", "charge", "psm_id", "rt"])
            .agg(
                pl.col("fragment_intensity")
                .max()
                .alias("highest_predicted_fragment_observed_intensity")
            )
        )
    else:
        pred_scores = pl.DataFrame(
            schema={
                "peptide": pl.Utf8,
                "charge": charge_dtype,
                "psm_id": pl.Int64,
                "rt": pl.Float64,
                "predicted_fragment_apex_score": pl.Float64,
                "matched_predicted_fragments": pl.UInt32,
                "max_matched_predicted_fragment_intensity": pl.Float64,
            }
        )
        top_frag_scores = pl.DataFrame(
            schema={
                "peptide": pl.Utf8,
                "charge": charge_dtype,
                "psm_id": pl.Int64,
                "rt": pl.Float64,
                "highest_predicted_fragment_observed_intensity": pl.Float64,
            }
        )

    # ── 7. Build candidate PSM table (primary-window-filtered) ────────────
    keep_cols = [
        c
        for c in [
            "psm_id",
            "rt",
            "peptide",
            "charge",
            "window_id",
            "precursor_in_window",
        ]
        if c in df_psms.columns
    ]
    candidate_psms = df_psms.select(keep_cols).unique(
        subset=["psm_id"], maintain_order=True
    )

    if has_window and not primary_window_df.is_empty():
        candidate_withwin = candidate_psms.join(
            primary_window_df, on=["peptide", "charge"], how="left"
        )
        filtered_candidates = candidate_withwin.filter(
            pl.col("primary_window_id").is_null()
            | (
                (pl.col("window_id") == pl.col("primary_window_id"))
                & pl.col("precursor_in_window").fill_null(False)
            )
        )
        # Fall back to unfiltered for peptidoforms that lost all candidates
        covered_pf = filtered_candidates.select(["peptide", "charge"]).unique()
        missing_pf = (
            candidate_psms.select(["peptide", "charge"])
            .unique()
            .join(covered_pf, on=["peptide", "charge"], how="anti")
        )
        if not missing_pf.is_empty():
            fallback_cands = candidate_withwin.join(
                missing_pf, on=["peptide", "charge"], how="inner"
            )
            candidate_psms = pl.concat(
                [filtered_candidates, fallback_cands], how="diagonal_relaxed"
            )
        else:
            candidate_psms = filtered_candidates

    # ── 8. Join all scores and predicted RT ───────────────────────────────
    join_key = ["peptide", "charge", "psm_id", "rt"]
    # Cast charge in score tables to match candidate_psms
    for score_df_name, score_df in [
        ("fragment_counts", fragment_counts),
        ("pred_scores", pred_scores),
        ("top_frag_scores", top_frag_scores),
    ]:
        if "charge" in score_df.columns and score_df["charge"].dtype != charge_dtype:
            score_df = score_df.with_columns(pl.col("charge").cast(charge_dtype))
        if score_df_name == "fragment_counts":
            fragment_counts = score_df
        elif score_df_name == "pred_scores":
            pred_scores = score_df
        else:
            top_frag_scores = score_df

    candidate_psms = (
        candidate_psms.join(fragment_counts, on=join_key, how="left")
        .with_columns(
            [
                pl.col("annotated_fragment_count").fill_null(0),
                pl.col("max_annotated_fragment_intensity").fill_null(0.0),
            ]
        )
        .join(pred_scores, on=join_key, how="left")
        .with_columns(
            [
                pl.col("predicted_fragment_apex_score").fill_null(0.0),
                pl.col("matched_predicted_fragments").fill_null(0),
                pl.col("max_matched_predicted_fragment_intensity").fill_null(0.0),
            ]
        )
        .join(top_frag_scores, on=join_key, how="left")
        .with_columns(
            pl.col("highest_predicted_fragment_observed_intensity").fill_null(0.0)
        )
        .join(predicted_rt_df, on=["peptide", "charge"], how="left")
        .with_columns(
            (pl.col("rt") - pl.col("predicted_rt_anchor").fill_null(pl.col("rt")))
            .abs()
            .alias("predicted_rt_distance")
        )
    )

    if candidate_psms.is_empty():
        log_info(
            "Predicted-fragment apex selection produced no rows; falling back to empty apex table"
        )
        return _empty_predicted_fragment_apex_table(df_psms)

    # ── 9. Fragment-rich region per peptidoform + apex selection ──────────
    sort_cols = [
        "highest_predicted_fragment_observed_intensity",
        "annotated_fragment_count",
        "predicted_rt_distance",
        "predicted_fragment_apex_score",
        "matched_predicted_fragments",
        "max_matched_predicted_fragment_intensity",
    ]
    sort_desc = [True, True, False, True, True, True]

    candidate_psms = candidate_psms.with_columns(
        (
            (
                pl.col("annotated_fragment_count").max().over(["peptide", "charge"])
                * min_fraction_of_max_count
            )
            .ceil()
            .cast(pl.Int32)
        ).alias("min_annotated_count_required")
    )

    # sort + unique(keep="first") = "argmax per group" without a Python loop
    rich_apex = (
        candidate_psms.filter(
            pl.col("annotated_fragment_count") >= pl.col("min_annotated_count_required")
        )
        .sort(sort_cols, descending=sort_desc)
        .unique(subset=["peptide", "charge"], keep="first", maintain_order=True)
    )
    if rich_apex.is_empty():
        rich_apex = pl.DataFrame(schema=candidate_psms.schema)

    fallback_apex = candidate_psms.sort(sort_cols, descending=sort_desc).unique(
        subset=["peptide", "charge"], keep="first", maintain_order=True
    )
    covered_pf = rich_apex.select(["peptide", "charge"])
    uncovered_apex = fallback_apex.join(
        covered_pf, on=["peptide", "charge"], how="anti"
    )
    apex_candidates = pl.concat([rich_apex, uncovered_apex], how="diagonal_relaxed")

    # ── 10. Reconstruct full apex rows from df_psms ───────────────────────
    # apex_candidates carries psm_id + computed scoring columns.
    # Join to df_psms (dropping columns already in apex_candidates to avoid conflicts).
    shared_cols = set(apex_candidates.columns) & set(df_psms.columns) - {"psm_id"}
    df_psms_for_join = df_psms.drop([c for c in shared_cols if c != "psm_id"])

    result = apex_candidates.join(df_psms_for_join, on="psm_id", how="left")

    # Add annotation columns (top_predicted_fragments, highest_predicted_fragment)
    if not df_top_frags.is_empty():
        result = result.join(df_top_frags, on=["peptide", "charge"], how="left")
    else:
        result = result.with_columns(
            [
                pl.lit(None).cast(pl.Utf8).alias("top_predicted_fragments"),
                pl.lit(None).cast(pl.Utf8).alias("highest_predicted_fragment"),
            ]
        )

    # Ensure primary_window_id and apex_window_id columns exist
    if "primary_window_id" not in result.columns:
        result = result.with_columns(
            pl.lit(None).cast(pl.Utf8).alias("primary_window_id")
        )
    if "apex_window_id" not in result.columns:
        result = result.with_columns(
            (
                pl.col("window_id")
                if "window_id" in result.columns
                else pl.lit(None).cast(pl.Utf8)
            ).alias("apex_window_id")
        )

    return result


def _apply_candidate_rt_window(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    radius: float,
    *,
    lower_col: str = "rt_candidate_lower",
    upper_col: str = "rt_candidate_upper",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply a symmetric RT window around `rt_predictions` to PSMs and fragments."""
    df_psms_with_window = df_psms.with_columns(
        [
            (pl.col("rt_predictions") - radius).alias(lower_col),
            (pl.col("rt_predictions") + radius).alias(upper_col),
        ]
    )

    df_psms_windowed = df_psms_with_window.filter(
        (pl.col("rt") >= pl.col(lower_col)) & (pl.col("rt") <= pl.col(upper_col))
    )

    if "precursor_in_window" in df_psms_windowed.columns:
        df_psms_windowed = df_psms_windowed.filter(
            pl.col("precursor_in_window").is_null() | pl.col("precursor_in_window")
        )

    missing_peptidoforms = pl.DataFrame()
    fallback_psms = pl.DataFrame()
    if {"peptide", "charge"}.issubset(df_psms_with_window.columns):
        kept_peptidoforms = df_psms_windowed.select(["peptide", "charge"]).unique(
            maintain_order=True
        )
        missing_peptidoforms = (
            df_psms_with_window.select(["peptide", "charge"])
            .unique(maintain_order=True)
            .join(kept_peptidoforms, on=["peptide", "charge"], how="anti")
        )
        if not missing_peptidoforms.is_empty():
            fallback_psms = df_psms_with_window.join(
                missing_peptidoforms,
                on=["peptide", "charge"],
                how="inner",
            )
            df_psms_windowed = pl.concat(
                [df_psms_windowed, fallback_psms],
                how="diagonal_relaxed",
            )

    df_fragment_windowed = df_fragment
    if not df_fragment.is_empty():
        df_fragment_windowed = df_fragment.join(
            df_psms_windowed.select(["psm_id", lower_col, upper_col]),
            on="psm_id",
            how="inner",
        )
        if "rt" in df_fragment_windowed.columns:
            df_fragment_windowed = df_fragment_windowed.filter(
                (pl.col("rt") >= pl.col(lower_col))
                & (pl.col("rt") <= pl.col(upper_col))
            )
        if (
            not missing_peptidoforms.is_empty()
            and {"peptide", "charge"}.issubset(df_fragment.columns)
            and not fallback_psms.is_empty()
        ):
            fallback_fragment_rows = df_fragment.join(
                missing_peptidoforms,
                on=["peptide", "charge"],
                how="inner",
            ).join(
                fallback_psms.select(["psm_id", lower_col, upper_col]),
                on="psm_id",
                how="left",
            )
            df_fragment_windowed = pl.concat(
                [df_fragment_windowed, fallback_fragment_rows],
                how="diagonal_relaxed",
            )

    return df_psms_windowed, df_fragment_windowed


def filter_to_candidate_rt_window(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    *,
    quantile: float = 95.0,
    multiplier: float = 2.0,
    fallback_radius: float = 0.5,
):
    """
    Build and apply a candidate RT window around DeepLC-predicted RT.

    The window radius is estimated from the empirical RT prediction error
    distribution and then centered on `rt_predictions` for each PSM.
    """
    if df_psms.is_empty() or "rt_predictions" not in df_psms.columns:
        return df_psms, df_fragment, fallback_radius

    rt_errors = np.abs(
        df_psms["rt"].to_numpy().astype(np.float64)
        - df_psms["rt_predictions"].to_numpy().astype(np.float64)
    )
    rt_errors = rt_errors[np.isfinite(rt_errors)]

    if len(rt_errors) == 0:
        radius = fallback_radius
    else:
        radius = max(
            float(np.percentile(rt_errors, quantile)) * multiplier, fallback_radius
        )

    df_psms, df_fragment = _apply_candidate_rt_window(
        df_psms,
        df_fragment,
        radius,
        lower_col="rt_candidate_lower",
        upper_col="rt_candidate_upper",
    )

    return df_psms, df_fragment, radius


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
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Main feature computation pipeline — the workhorse of Stage 3.

    Orchestrates the Stage 3 transformation from raw search results to a PIN file:
    1. DeepLC RT predictions (reuses transfer-learned model from Stage 2)
    2. Early MS2PIP fragment intensity predictions for peptidoforms
    3. Predicted-fragment-driven apex selection per peptidoform
    4. Candidate RT window filtering around DeepLC-predicted RT
    5. RT error features + filtering
    6. Rebuild predicted apex table after filtering
    7. Peptide occurrence filtering (min_occurrences, default 5)
    8. Fragment correlations, precursor features, XIC features, and PIN writing

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
        n_epochs=config["mumdia"].get("deeplc_epochs_prediction", 50),
        min_peptidoform_occurrences=config["mumdia"].get(
            "deeplc_min_peptidoform_occurrences", 1
        ),
    )

    df_fragment = ensure_fragment_name_column(df_fragment)
    df_psms, df_fragment, ms2_scan_metadata = attach_ms2_window_metadata(
        df_psms,
        df_fragment,
        spectra_data,
    )
    if not ms2_scan_metadata.is_empty():
        log_info(
            "Attached DIA isolation-window metadata to Stage 3 PSM and fragment tables"
        )
        if "window_id" in df_psms.columns:
            log_info(
                "  PSMs with explicit window ids: {}".format(
                    int(df_psms["window_id"].drop_nulls().len())
                )
            )
        if "precursor_in_window" in df_psms.columns:
            precursor_match_count = int(
                df_psms.filter(pl.col("precursor_in_window").fill_null(False)).height
            )
            log_info(
                f"  PSMs already compatible with their DIA window: {precursor_match_count}"
            )

    log_info("Obtaining fragment intensity predictions early for apex selection...")
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

    log_info("Building shared theoretical fragment context...")
    theoretical_fragment_context = build_theoretical_fragment_context(
        df_psms,
        ms2pip_predictions,
        max_fragment_charge_cap=config["mumdia"].get(
            "max_fragment_charge_theoretical", 2
        ),
    )

    log_info("Annotating candidate Sage fragments with theoretical context...")
    df_fragment = annotate_candidate_fragment_window(
        df_fragment,
        theoretical_fragment_context,
        top_n_predicted_fragments=config["mumdia"].get(
            "predicted_apex_top_fragments", 2
        ),
    )

    log_info("Building predicted-fragment-driven apex table...")
    df_fragment_max_peptide = build_predicted_fragment_apex_table(
        df_psms,
        df_fragment,
        ms2pip_predictions,
        top_n_predicted_fragments=config["mumdia"].get(
            "predicted_apex_top_fragments", 2
        ),
        min_fraction_of_max_count=config["mumdia"].get(
            "predicted_apex_min_fraction_of_max_count", 0.9
        ),
    )
    if not df_fragment_max_peptide.is_empty():
        df_fragment_max_peptide.write_csv(
            "debug/df_fragment_predicted_apex_initial.tsv", separator="\t"
        )

    df_psms_before_candidate_window = df_psms.clone()
    df_fragment_before_candidate_window = df_fragment.clone()

    log_info("Filtering to candidate RT windows around predicted RT...")
    df_psms, df_fragment, candidate_rt_radius = filter_to_candidate_rt_window(
        df_psms,
        df_fragment,
        quantile=config["mumdia"].get("predicted_rt_window_quantile", 95.0),
        multiplier=config["mumdia"].get("predicted_rt_window_multiplier", 2.0),
        fallback_radius=config["mumdia"].get("predicted_rt_window_fallback", 0.5),
    )
    log_info(f"Candidate RT window radius: {candidate_rt_radius}")
    df_fragment.write_csv("debug/df_fragment_candidate_window.tsv", separator="\t")

    reannotation_rt_radius = max(
        candidate_rt_radius
        * config["mumdia"].get("predicted_rt_window_reannotation_multiplier", 1.15),
        candidate_rt_radius
        + config["mumdia"].get("predicted_rt_window_reannotation_min_extra", 0.0),
    )
    reannotation_rt_radius *= config["mumdia"].get(
        "predicted_rt_window_reannotation_range_scale", 2.0
    )

    if reannotation_rt_radius > candidate_rt_radius:
        df_psms_reannotation, df_fragment_reannotation = _apply_candidate_rt_window(
            df_psms_before_candidate_window,
            df_fragment_before_candidate_window,
            reannotation_rt_radius,
            lower_col="rt_reannotation_lower",
            upper_col="rt_reannotation_upper",
        )
        log_info(
            "Expanded RT window for early reannotation: "
            f"{reannotation_rt_radius} (base candidate radius {candidate_rt_radius})"
        )
    else:
        df_psms_reannotation = df_psms
        df_fragment_reannotation = df_fragment
        log_info("Early reannotation reuses the candidate RT window without expansion")

    df_fragment_reannotation.write_csv(
        "debug/df_fragment_candidate_window_reannotation.tsv", separator="\t"
    )

    log_info("Reannotating candidate-window MS2 spectra early...")
    preannotated_fragment_dict = reannotate_candidate_ms2_spectra(
        df_psms_reannotation,
        df_fragment_reannotation,
        theoretical_fragment_context,
        spectra_data.ms2_dict,
        tolerance_ppm=config["mumdia"].get("rustyms_annotation_tolerance_ppm", 13.0),
    )
    feature_fragment_df = flatten_preannotated_fragment_dict(
        preannotated_fragment_dict, df_fragment
    )
    df_fragment_max_peptide = build_predicted_fragment_apex_table(
        df_psms,
        feature_fragment_df,
        ms2pip_predictions,
        top_n_predicted_fragments=config["mumdia"].get(
            "predicted_apex_top_fragments", 2
        ),
        min_fraction_of_max_count=config["mumdia"].get(
            "predicted_apex_min_fraction_of_max_count", 0.9
        ),
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
    preannotated_fragment_dict = filter_preannotated_fragment_dict(
        preannotated_fragment_dict, set(df_psms["psm_id"].to_list())
    )
    feature_fragment_df = flatten_preannotated_fragment_dict(
        preannotated_fragment_dict, df_fragment
    )
    log_info(
        "df_fragment shape after filtering to match RT-filtered PSMs: {}".format(
            df_fragment.shape
        )
    )

    # Regenerate the maximum intensity fragment per PSM
    df_fragment_max = df_fragment.sort("fragment_intensity", descending=True).unique(
        subset="psm_id", keep="first", maintain_order=True
    )

    # Rebuild the predicted-fragment apex PSM per peptide/charge combination from the filtered data
    df_fragment_max_peptide = build_predicted_fragment_apex_table(
        df_psms,
        feature_fragment_df,
        ms2pip_predictions,
        top_n_predicted_fragments=config["mumdia"].get(
            "predicted_apex_top_fragments", 2
        ),
        min_fraction_of_max_count=config["mumdia"].get(
            "predicted_apex_min_fraction_of_max_count", 0.9
        ),
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
    preannotated_fragment_dict = filter_preannotated_fragment_dict(
        preannotated_fragment_dict, set(df_psms["psm_id"].to_list())
    )
    feature_fragment_df = flatten_preannotated_fragment_dict(
        preannotated_fragment_dict, df_fragment
    )

    # Regenerate the maximum intensity fragment per PSM
    df_fragment_max = df_fragment.sort("fragment_intensity", descending=True).unique(
        subset="psm_id", keep="first", maintain_order=True
    )

    # Regenerate the predicted-fragment apex PSM per peptide/charge combination from the fully filtered data
    df_fragment_max_peptide = build_predicted_fragment_apex_table(
        df_psms,
        feature_fragment_df,
        ms2pip_predictions,
        top_n_predicted_fragments=config["mumdia"].get(
            "predicted_apex_top_fragments", 2
        ),
        min_fraction_of_max_count=config["mumdia"].get(
            "predicted_apex_min_fraction_of_max_count", 0.9
        ),
    )

    log_info("Final df_fragment_max_peptide after all filtering:")
    log_info("  Shape: {}".format(df_fragment_max_peptide.shape))
    log_info("  This should now be consistent with all downstream processing")

    # Validation: Check that all peptides in df_fragment_max_peptide exist in the filtered data
    fragment_max_psm_ids = (
        set(df_fragment_max_peptide["psm_id"].to_list())
        if "psm_id" in df_fragment_max_peptide.columns
        else set()
    )
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

    if df_psms.is_empty():
        log_info(
            "No PSMs remain after Stage 3 filtering; skipping late feature generation."
        )
        _clear_pin_output(config["mumdia"]["result_dir"])
        return df_fragment, df_psms

    # Step 4b: Calculate adaptive RT margins per peptidoform.
    # Calibrates min/max margins from top-scoring peptidoforms, then walks each
    # peptidoform's XIC outward from apex until intensity drops below threshold.
    # Adds rt_lower_margin and rt_higher_margin columns to both DataFrames.
    log_info("Calculating adaptive retention time margins...")

    df_fragment = ensure_fragment_name_column(df_fragment)
    # Do not use the global all-MS2 RT grid by default here: it spans scans from
    # unrelated precursors/charge states within the same RT region and can make
    # peptidoform traces look artificially zig-zag. Keep zero-fill on the local
    # peptidoform RT support unless a more specific scan grid is available.
    global_ms2_rt_values = None

    rt_margin_top_n = int(config["mumdia"].get("rt_margin_top_n", 100))
    rt_margin_intensity_threshold = float(
        config["mumdia"].get("rt_margin_intensity_threshold", 0.05)
    )
    rt_margin_preferred_fragment_count = int(
        config["mumdia"].get("rt_margin_preferred_fragment_count", 2)
    )
    rt_margin_mode = str(
        config["mumdia"].get("rt_margin_mode", "adaptive_per_peptidoform")
    )

    rt_margin_dump_dir = os.path.join(
        config["mumdia"]["result_dir"], "rt_margin_input_dump"
    )
    dump_rt_margin_inputs(
        df_psms,
        df_fragment,
        feature_fragment_df,
        df_fragment_max_peptide,
        rt_margin_dump_dir,
        fragment_table_used_for_margin_calculation="df_fragment_search_pre_margin.tsv",
        metadata={
            "default_top_n": rt_margin_top_n,
            "default_intensity_threshold": rt_margin_intensity_threshold,
            "default_preferred_fragment_count": rt_margin_preferred_fragment_count,
            "rt_margin_mode": rt_margin_mode,
            "predicted_apex_top_fragments": config["mumdia"].get(
                "predicted_apex_top_fragments", 6
            ),
            "margin_replay_note": "Margin replay should use the recorded fragment table from this dump.",
            "notes": "These are the exact inputs used immediately before add_retention_time_margins_loop().",
        },
        global_ms2_rt_values=global_ms2_rt_values,
    )
    log_info(f"Wrote RT-margin replay dump to {rt_margin_dump_dir}")

    df_psms, df_fragment = add_retention_time_margins_loop(
        df_psms,
        feature_fragment_df,
        df_fragment_max_peptide,
        top_n=rt_margin_top_n,
        intensity_threshold=rt_margin_intensity_threshold,
        preferred_fragment_count=rt_margin_preferred_fragment_count,
        global_ms2_rt_values=global_ms2_rt_values,
        margin_mode=rt_margin_mode,
    )
    preannotated_fragment_dict = enrich_preannotated_fragment_dict(
        preannotated_fragment_dict,
        df_psms,
        df_fragment_max_peptide,
    )
    feature_fragment_df = flatten_preannotated_fragment_dict(
        preannotated_fragment_dict, df_fragment
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

    feature_calc_dump_dir = os.path.join(
        config["mumdia"]["result_dir"], "stage3_feature_input_dump"
    )
    dump_stage3_feature_inputs(
        df_psms,
        df_fragment,
        feature_fragment_df,
        df_fragment_max_peptide,
        ms2pip_predictions,
        theoretical_fragment_context,
        preannotated_fragment_dict,
        config,
        spectra_data,
        pickle_config,
        feature_calc_dump_dir,
    )
    log_info(f"Wrote Stage 3 feature dump to {feature_calc_dump_dir}")

    return run_late_stage3_feature_calculation(
        df_psms=df_psms,
        df_fragment=df_fragment,
        feature_fragment_df=feature_fragment_df,
        df_fragment_max_peptide=df_fragment_max_peptide,
        ms2pip_predictions=ms2pip_predictions,
        theoretical_fragment_context=theoretical_fragment_context,
        preannotated_fragment_dict=preannotated_fragment_dict,
        config=config,
        pickle_config=pickle_config,
        spectra_data=spectra_data,
    )


def run_late_stage3_feature_calculation(
    *,
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    feature_fragment_df: pl.DataFrame,
    df_fragment_max_peptide: pl.DataFrame,
    ms2pip_predictions: Dict[str, Dict[str, float]],
    theoretical_fragment_context: Dict[str, Dict[str, Any]],
    preannotated_fragment_dict: Optional[Dict[str, pl.DataFrame]],
    config: Dict[str, Any],
    pickle_config: Optional[PickleConfig],
    spectra_data: Optional[SpectraData],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the late Stage 3 feature generation from a persisted checkpoint."""
    if pickle_config is None:
        pickle_config = PickleConfig()
    if spectra_data is None:
        spectra_data = SpectraData()
    if preannotated_fragment_dict is None:
        preannotated_fragment_dict = {}

    fragment_dict, correlations_fragment_dict = get_features_fragment_intensity(
        ms2pip_predictions,
        feature_fragment_df,
        df_fragment_max_peptide,
        read_correlation_pickles=pickle_config.read_correlation,
        write_correlation_pickles=pickle_config.write_correlation,
        ms2_dict=spectra_data.ms2_dict,
        output_dir=config["mumdia"]["result_dir"],
        preannotated_fragment_dict=preannotated_fragment_dict,
    )

    log_info("Step 5: obtain MS1 peak presence")

    df_psms = add_precursor_intensities(
        df_psms, spectra_data.ms1_dict, spectra_data.ms2_to_ms1_dict
    )

    log_info(f"Number of PSMs:{df_psms.shape[0]}")

    primary_window_lookup: Dict[str, Optional[str]] = {}
    if (
        not df_fragment_max_peptide.is_empty()
        and "primary_window_id" in df_fragment_max_peptide.columns
        and {"peptide", "charge"}.issubset(df_fragment_max_peptide.columns)
    ):
        for _pw_row in df_fragment_max_peptide.select(
            ["peptide", "charge", "primary_window_id"]
        ).to_dicts():
            _pw_key = f"{_pw_row['peptide']}/{_pw_row['charge']}"
            primary_window_lookup[_pw_key] = _pw_row.get("primary_window_id")
        log_info(
            f"  Primary window lookup built for {len(primary_window_lookup)} peptidoforms"
        )

    xic_features_dict = {}
    if _RUST_BACKEND and spectra_data and spectra_data.ms2_dict:
        log_info("Step 5b: Extracting targeted XICs from all MS2 scans...")

        ms2_items = sorted(
            spectra_data.ms2_dict.items(), key=lambda x: x[1]["retention_time"]
        )
        ms2_rts_flat = np.array(
            [v["retention_time"] for _, v in ms2_items], dtype=np.float64
        )
        mz_arrays = [np.asarray(v["mz"], dtype=np.float64) for _, v in ms2_items]
        int_arrays = [
            np.asarray(v["intensity"], dtype=np.float64) for _, v in ms2_items
        ]

        ms2_iso_lower_flat = np.array(
            [
                v.get("isolation_window_target", 0.0)
                - v.get("isolation_window_lower", 0.0)
                for _, v in ms2_items
            ],
            dtype=np.float64,
        )
        ms2_iso_upper_flat = np.array(
            [
                v.get("isolation_window_target", 0.0)
                + v.get("isolation_window_upper", 0.0)
                for _, v in ms2_items
            ],
            dtype=np.float64,
        )
        has_isolation_windows = ms2_iso_upper_flat.max() > 0.0
        if has_isolation_windows:
            log_info(
                f"  Isolation windows available: {ms2_iso_lower_flat.min():.1f} - {ms2_iso_upper_flat.max():.1f} m/z"
            )
        else:
            log_info("  WARNING: No isolation window data — will search all scans")
            ms2_iso_lower_flat[:] = 0.0
            ms2_iso_upper_flat[:] = 1e6

        offsets = np.zeros(len(mz_arrays), dtype=np.uint64)
        lengths = np.zeros(len(mz_arrays), dtype=np.uint64)
        offset = 0
        for i, mz_arr in enumerate(mz_arrays):
            offsets[i] = offset
            lengths[i] = len(mz_arr)
            offset += len(mz_arr)

        ms2_mz_flat = (
            np.concatenate(mz_arrays) if mz_arrays else np.array([], dtype=np.float64)
        )
        ms2_int_flat = (
            np.concatenate(int_arrays) if int_arrays else np.array([], dtype=np.float64)
        )

        log_info(
            f"  Flattened {len(ms2_items)} MS2 scans ({len(ms2_mz_flat)} total peaks)"
        )

        if "fragment_mz_calculated" in df_fragment.columns:
            log_info("  Building XIC targets from fragment m/z values...")

            mzml_in_seconds = ms2_rts_flat.max() > 200
            psm_in_minutes = df_psms["rt"].max() < 200
            rt_margin = 30.0 if mzml_in_seconds else 0.5
            rt_conversion = 60.0 if (psm_in_minutes and mzml_in_seconds) else 1.0
            if rt_conversion != 1.0:
                log_info(
                    f"  RT unit conversion: PSM RT (minutes) × {rt_conversion} → mzML RT (seconds)"
                )
            log_info(f"  XIC RT window: ±{rt_margin:.0f}s (matches DIA-NN peak width)")

            frag_by_key = {}
            frag_names_key = {}
            rt_window_lookup = {}
            for (peptide, charge), df_sub in df_fragment.group_by(
                ["peptide", "charge"]
            ):
                key = f"{peptide}/{charge}"
                shared_df = preannotated_fragment_dict.get(key)
                if shared_df is not None and not shared_df.is_empty():
                    if {
                        "fragment_name",
                        "theoretical_fragment_mz",
                    }.issubset(shared_df.columns):
                        shared_named_frags = []
                        for row in (
                            shared_df.select(
                                [
                                    "fragment_name",
                                    "theoretical_fragment_mz",
                                    "predicted_fragment_intensity",
                                ]
                            )
                            .unique(subset=["fragment_name"], maintain_order=True)
                            .drop_nulls(subset=["theoretical_fragment_mz"])
                            .to_dicts()
                        ):
                            shared_named_frags.append(
                                (
                                    row["fragment_name"],
                                    float(row["theoretical_fragment_mz"]),
                                )
                            )
                        if shared_named_frags:
                            shared_named_frags.sort(key=lambda x: x[1])
                            frag_by_key[key] = np.array(
                                [mz for _, mz in shared_named_frags], dtype=np.float64
                            )
                            frag_names_key[key] = shared_named_frags

                    if {"rt_lower_margin", "rt_higher_margin"}.issubset(
                        shared_df.columns
                    ):
                        margin_rows = (
                            shared_df.select(["rt_lower_margin", "rt_higher_margin"])
                            .drop_nulls()
                            .head(1)
                        )
                        if not margin_rows.is_empty():
                            rt_window_lookup[key] = (
                                float(margin_rows["rt_lower_margin"][0]),
                                float(margin_rows["rt_higher_margin"][0]),
                            )

                    if key in frag_by_key:
                        continue

                context = theoretical_fragment_context.get(key, {})
                named_frags = context.get("theoretical_fragments", [])
                if named_frags:
                    frag_by_key[key] = np.array(
                        [mz for _, mz in named_frags], dtype=np.float64
                    )
                    frag_names_key[key] = named_frags
                    continue

                mzs = np.sort(
                    df_sub["fragment_mz_calculated"]
                    .unique()
                    .to_numpy()
                    .astype(np.float64)
                )
                frag_by_key[key] = mzs

            xic_keys = []
            all_target_mzs_list = []
            all_preds_list = []
            all_precursor_mzs = []
            all_rt_mins = []
            all_rt_maxs = []

            for k in correlations_fragment_dict.keys():
                if k not in frag_by_key:
                    continue
                frag_mzs = frag_by_key[k]
                if len(frag_mzs) == 0:
                    continue

                parts = k.rsplit("/", 1)
                if len(parts) != 2:
                    continue
                peptide, charge_str = parts
                charge_int = int(charge_str)
                pep_psms = df_psms.filter(
                    (pl.col("peptide") == peptide) & (pl.col("charge") == charge_int)
                )
                if len(pep_psms) == 0:
                    continue
                _xic_pw = primary_window_lookup.get(k)
                if _xic_pw is not None and "window_id" in pep_psms.columns:
                    _pep_win = pep_psms.filter(pl.col("window_id") == _xic_pw)
                    if "precursor_in_window" in _pep_win.columns:
                        _pep_win = _pep_win.filter(
                            pl.col("precursor_in_window").fill_null(False)
                        )
                    if not _pep_win.is_empty():
                        pep_psms = _pep_win
                rt_center_source = theoretical_fragment_context.get(k, {}).get(
                    "predicted_rt", float(pep_psms["rt"].median())
                )
                if not df_fragment_max_peptide.is_empty():
                    apex_rows = df_fragment_max_peptide.filter(
                        (pl.col("peptide") == peptide)
                        & (pl.col("charge") == charge_int)
                    )
                    if not apex_rows.is_empty():
                        rt_center_source = float(apex_rows["rt"][0])
                rt_center = float(rt_center_source) * rt_conversion

                prec_mz = theoretical_fragment_context.get(k, {}).get("precursor_mz")
                if prec_mz is None:
                    calcmass = float(pep_psms["calcmass"].first())
                    prec_mz = calcmass / charge_int + PROTON_MASS

                preds = ms2pip_predictions.get(k, {})
                if k in frag_names_key:
                    frag_with_preds = []
                    shared_df = preannotated_fragment_dict.get(k)
                    shared_pred_lookup = {}
                    if (
                        shared_df is not None
                        and not shared_df.is_empty()
                        and {"fragment_name", "predicted_fragment_intensity"}.issubset(
                            shared_df.columns
                        )
                    ):
                        shared_pred_lookup = {
                            row["fragment_name"]: float(
                                row["predicted_fragment_intensity"]
                            )
                            for row in shared_df.select(
                                ["fragment_name", "predicted_fragment_intensity"]
                            )
                            .unique(subset=["fragment_name"], maintain_order=True)
                            .to_dicts()
                        }
                    for fname, mz in frag_names_key[k]:
                        pred_val = shared_pred_lookup.get(fname, preds.get(fname, 0.0))
                        frag_with_preds.append((fname, mz, pred_val))

                    frag_with_preds.sort(key=lambda x: x[1])
                    frag_mzs = np.array(
                        [mz for _, mz, _ in frag_with_preds], dtype=np.float64
                    )
                    pred_values = np.array(
                        [p for _, _, p in frag_with_preds], dtype=np.float64
                    )
                else:
                    pred_values = np.zeros(len(frag_mzs), dtype=np.float64)
                    pred_list = list(preds.values())
                    for i in range(min(len(pred_list), len(pred_values))):
                        pred_values[i] = pred_list[i]

                xic_keys.append(k)
                all_target_mzs_list.append(frag_mzs)
                all_preds_list.append(pred_values)
                all_precursor_mzs.append(prec_mz)
                if k in rt_window_lookup:
                    rt_min, rt_max = rt_window_lookup[k]
                    all_rt_mins.append(rt_min * rt_conversion)
                    all_rt_maxs.append(rt_max * rt_conversion)
                else:
                    all_rt_mins.append(rt_center - rt_margin)
                    all_rt_maxs.append(rt_center + rt_margin)

            log_info(
                f"  Prepared {len(xic_keys)} peptidoforms for batch XIC extraction"
            )

            if xic_keys:
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

                log_info("  Running batch XIC extraction in Rust...")
                results = mumdia_rs.batch_extract_xic_features(
                    ms2_rts_flat,
                    offsets,
                    lengths,
                    ms2_mz_flat,
                    ms2_int_flat,
                    ms2_iso_lower_flat,
                    ms2_iso_upper_flat,
                    target_mz_flat,
                    target_offsets_arr,
                    target_lengths_arr,
                    pred_flat,
                    pred_offsets_arr,
                    pred_lengths_arr,
                    np.array(all_precursor_mzs, dtype=np.float64),
                    np.array(all_rt_mins, dtype=np.float64),
                    np.array(all_rt_maxs, dtype=np.float64),
                    13.0,
                )
                for i, k in enumerate(xic_keys):
                    xic_features_dict[k] = results[i]

            log_info(f"  Extracted XICs for {len(xic_features_dict)} peptidoforms")
        else:
            log_info("  Skipping XIC: fragment_mz_calculated column not available")

    log_info("Step 6: Grouping peptidoforms by peptide and charge")

    _has_window_cols = (
        "window_id" in df_psms.columns and "precursor_in_window" in df_psms.columns
    )
    psm_dict = {}
    n_win_filtered = 0
    for (peptidoform, charge), df_sub_peptidoform in tqdm(
        df_psms.group_by(["peptide", "charge"])
    ):
        key = f"{peptidoform}/{charge}"
        if _has_window_cols:
            pw_id = primary_window_lookup.get(key)
            if pw_id is not None:
                win_sub = df_sub_peptidoform.filter(
                    (pl.col("window_id") == pw_id)
                    & pl.col("precursor_in_window").fill_null(False)
                )
                if not win_sub.is_empty():
                    df_sub_peptidoform = win_sub
                    n_win_filtered += 1
        psm_dict[key] = df_sub_peptidoform
    if _has_window_cols:
        log_info(
            f"  PSM window filtering: {n_win_filtered}/{len(psm_dict)} peptidoforms restricted to their primary isolation window"
        )

    log_info(f"Number of peptidoforms: {len(psm_dict)}")

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

    if not peptidoform_args:
        log_info("No peptidoforms remain for PIN generation; skipping late Stage 3.")
        _clear_pin_output(config["mumdia"]["result_dir"])
        return df_fragment, df_psms

    with open("debug/psm_dict.pkl", "wb") as f:
        pickle.dump(psm_dict, f)

    with open("debug/correlations_fragment_dict.pkl", "wb") as f:
        pickle.dump(correlations_fragment_dict, f)

    log_info("Step 7: Processing peptidoforms in parallel")

    global _use_diann_features, _diann_na_strategy
    _use_diann_features = config.get("mumdia", {}).get("use_diann_features", True)
    _diann_na_strategy = config.get("mumdia", {}).get(
        "diann_na_strategy", "overlap_only"
    )
    if not _use_diann_features:
        log_info("  DIA-NN features DISABLED (use_diann_features=False)")
    else:
        log_info(f"  DIA-NN NA strategy: {_diann_na_strategy}")

    if _use_diann_features:
        _prepare_diann_ms1(spectra_data)

    pin_in = [
        process_peptidoform(args)
        for args in tqdm(peptidoform_args, desc="Processing peptidoforms")
    ]

    global _diann_generator
    if _diann_generator is not None:
        _diann_generator.clear_cache()
    _diann_generator = None

    log_info("Step 8: Concatenating results")
    concatenated_df = (
        pl.concat(pin_in, how="diagonal")
        .rename(
            {
                "expmass": "ExpMass",
                "calcmass": "CalcMass",
                "psm_id": "ScanNr",
                "peptide": "Peptide",
                "proteins": "Proteins",
                "is_decoy": "Label",
            }
        )
        .drop("scannr")
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

    pin_path = os.path.join(config["mumdia"]["result_dir"], "outfile.pin")
    if not os.path.exists(pin_path) or os.path.getsize(pin_path) == 0:
        log_info("No PIN rows were generated; skipping mokapot and quantification.")
        return

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
