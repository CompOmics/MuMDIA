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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import mokapot
import numba as nb
import numpy as np
import polars as pl
from keras.layers import Dense
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from tqdm import tqdm

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
from utilities.logger import log_info

# Re-export for backward compatibility
__all__ = ["main", "PickleConfig", "SpectraData", "run_mokapot"]

# Set maximum threads for Polars to one to avoid oversubscription
os.environ["POLARS_MAX_THREADS"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from scipy import stats

# Constant features for later concatenation.
last_features = ["proteins"]


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
    sorted_data = np.sort(data)[::-1]
    result = np.empty(m, dtype=np.float64)
    result_idx = np.empty(m, dtype=np.float64)
    for i in range(m):
        if i < n:
            result[i] = sorted_data[i]
            idx_ret_list[i] = idx_ret_list[i]
        else:
            result[i] = 0.0
            idx_ret_list[i] = 0.0
    return result, idx_ret_list


@nb.njit
def corr_np_nb(data1, data2):
    """
    Compute the Pearson correlation coefficient between two 1D arrays.
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


def create_model():
    """
    Create and compile a simple Keras model.
    """
    model = Sequential()
    model.add(Dense(100, input_dim=103, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
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
    psms = mokapot.read_pin("outfile.pin")
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
    Collapse columns using max, min, mean, and sum operations.
    """
    collapsed_columns = [df_psms_sub_peptidoform.select(get_first_entry).head(1)]
    operations = (
        ("max", collapse_max_columns),
        ("min", collapse_min_columns),
        ("mean", collapse_mean_columns),
        ("sum", collapse_sum_columns),
    )
    for op, collapse_list in operations:
        if collapse_list:
            collapsed_columns.append(
                getattr(df_psms_sub_peptidoform[collapse_list], op)().rename(
                    {col: f"{col}_{op}" for col in collapse_list}
                )
            )
    return pl.concat(collapsed_columns, how="horizontal")


def add_feature_columns_nb(data, feature_name, values, method, add_index, pad_size=10):
    """
    Compute a feature vector from the input data using Numba-accelerated routines.
    Returns a dictionary mapping column names to computed scalar values.
    """
    data = np.asarray(data, dtype=np.float64)
    required_length = len(values)
    if data.size == 0:
        computed = np.zeros(required_length, dtype=np.float64)
    elif method == "percentile":
        qs = np.array(values, dtype=np.float64)
        if len(add_index) > 0:
            computed, computed_idx = compute_percentiles_nb_idx(data, qs, add_index)
        else:
            computed = compute_percentiles_nb(data, qs)
    elif method == "top":
        if len(add_index) > 0:
            computed, computed_idx = compute_top_nb_idx(
                data, required_length, add_index
            )
        else:
            computed = compute_top_nb(data, required_length)
    else:
        raise ValueError(f"Unknown method: {method}")
    # Ensure computed is of the required length
    if computed.size < required_length:
        padded = np.zeros(required_length, dtype=np.float64)
        padded[: computed.size] = computed
        computed = padded
        if len(add_index) > 0:
            padded_idx = np.zeros(required_length, dtype=np.float64)
            padded_idx[: computed_idx.size] = computed_idx
            computed_idx = padded_idx
    else:
        computed = computed[:required_length]
        if len(add_index) > 0:
            computed_idx = computed_idx[:required_length]

    if len(add_index) > 0:
        return {
            **{f"{feature_name}_{v}": computed[i] for i, v in enumerate(values)},
            **{
                f"{feature_name}_{v}_idx": computed_idx[i] for i, v in enumerate(values)
            },
        }
    else:
        return {f"{feature_name}_{v}": computed[i] for i, v in enumerate(values)}


def run_peptidoform_df(
    df_psms_sub_peptidoform: pl.DataFrame,
    selected_features: List[str] = [],
    collect_distributions: List[int] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    collect_top: List[int] = [1, 2, 3, 4, 5],
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
    Process a peptidoform DataFrame to add calculated features.
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
        correlation_matrix_psm_ids_ignore_zeros,
        correlation_matrix_psm_ids_ignore_zeros_counts,
        correlation_matrix_psm_ids_missing,
        correlation_matrix_psm_ids_missing_zeros_counts,
        correlation_matrix_frag_ids_ignore_zeros,
        correlation_matrix_frag_ids_ignore_zeros_counts,
        correlation_matrix_frag_ids_missing,
        correlation_matrix_frag_ids_missing_zeros_counts,
        most_intens_cor,
        most_intens_cos,
        mse_avg_pred_intens,
        mse_avg_pred_intens_total,
    ) = correlations_list

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
        # Here, for percentiles and top values, we use the Numba-accelerated add_feature_columns_nb.
        feature_dict.update(
            add_feature_columns_nb(
                data, feat_name, values, method, add_index, pad_size=ps
            )
        )

    return pl.DataFrame(feature_dict)


def dataframe_to_dict_fragintensity(df_fragment: pl.DataFrame) -> dict:
    """
    Convert a DataFrame of fragment intensities into a dictionary keyed by psm_id.
    """
    fragment_dict = {}
    for psm_id, sub_df_fragment in df_fragment.group_by("psm_id"):
        fragment_dict[psm_id] = sub_df_fragment
    return fragment_dict


def process_peptidoform(args):
    """
    Process a single peptidoform group by computing its feature DataFrames and concatenating them.
    """
    df_psms_sub_peptidoform, df_fragment_sub_peptidoform, correlations_list = args
    df1 = run_peptidoform_df(df_psms_sub_peptidoform)
    df2 = run_peptidoform_correlation(correlations_list)
    return pl.concat([df1, df2], how="horizontal")


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
    # 1. Extract unique precursor combinations
    unique_precursors = df_psms.select(["scannr", "charge", "calcmass"]).unique()

    # 2. Define the function to compute intensities for a single row
    def compute_intensities(row):
        scannr, charge, calcmass = row["scannr"], row["charge"], row["calcmass"]
        if scannr not in ms2_to_ms1_dict:
            return {"M-1": 0.0, "M": 0.0, "M+1": 0.0}
        spectrum = ms1_dict.get(ms2_to_ms1_dict[scannr], {})
        if not spectrum:
            return {"M-1": 0.0, "M": 0.0, "M+1": 0.0}
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
                    row["scannr"], row["charge"], row["calcmass"]
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


def calculate_features(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    df_fragment_max: pl.DataFrame,
    df_fragment_max_peptide: pl.DataFrame,
    *,  # Force keyword-only arguments
    filter_rel_rt_error: float = 0.1,
    min_occurrences: int = 1,
    filter_max_apex_rt: float = 0.75,
    config: dict = {},
    deeplc_model=None,
    pickle_config: Optional[PickleConfig] = None,
    parallel_workers: int = 24,  # Adjust to number of physical cores
    chunk_size: int = 500,  # Increase chunk size to reduce overhead
    spectra_data: Optional[SpectraData] = None,
) -> None:
    """
    Process the PSM and fragment DataFrames, compute features, and save the output.
    This function uses parallel processing with task chunking.
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
    _, _, predictions_deeplc = get_predictions_retention_time_mainloop(
        df_psms, pickle_config.write_deeplc, pickle_config.read_deeplc, deeplc_model
    )

    log_info("Obtaining features retention time...")
    df_psms = add_retention_time_features(
        df_psms, predictions_deeplc, filter_rel_rt_error=0.15
    )

    log_info(
        "Counting individual peptides per MS2 and filtering by minimum occurrences"
    )
    df_psms = add_count_and_filter_peptides(df_psms, min_occurrences)

    log_info("Obtaining fragment intensity predictions for the main loop...")
    log_info(
        f"Reading the MS2PIP pickle: {pickle_config.read_ms2pip} and writing MS2PIP pickle: {pickle_config.write_ms2pip}"
    )

    df_fragment, ms2pip_predictions = get_predictions_fragment_intensity_main_loop(
        df_psms,
        df_fragment,
        read_ms2pip_pickle=pickle_config.read_ms2pip,
        write_ms2pip_pickle=pickle_config.write_ms2pip,
    )

    log_info("Obtaining features fragment intensity predictions...")
    fragment_dict, correlations_fragment_dict = get_features_fragment_intensity(
        ms2pip_predictions,
        df_fragment,
        df_fragment_max_peptide,
        read_correlation_pickles=pickle_config.read_correlation,
        write_correlation_pickles=pickle_config.write_correlation,
        ms2_dict=spectra_data.ms2_dict,
    )

    log_info("Step 5: obtain MS1 peak presence")

    df_psms = add_precursor_intensities(
        df_psms, spectra_data.ms1_dict, spectra_data.ms2_to_ms1_dict
    )

    log_info("Step 6: Grouping peptidoforms by peptide and charge")

    psm_dict = {}
    for (peptidoform, charge), df_sub_peptidoform in tqdm(
        df_psms.group_by(["peptide", "charge"])
    ):
        psm_dict[f"{peptidoform}/{charge}"] = df_sub_peptidoform

    # Pass data as-is (read-only) without deep copying.
    peptidoform_args = [
        (psm_dict[k], fragment_dict[k], correlations_fragment_dict[k])
        for k in psm_dict.keys()
        if k in correlations_fragment_dict
    ]

    log_info("Step 7: Processing peptidoforms in parallel (with chunking)")

    chunks = [
        peptidoform_args[i : i + chunk_size]
        for i in range(0, len(peptidoform_args), chunk_size)
    ]

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        chunk_results = list(
            tqdm(
                executor.map(
                    lambda chunk: [process_peptidoform(args) for args in chunk], chunks
                ),
                total=len(chunks),
                desc="Processing chunks",
            )
        )
    """
    chunk_results = []
    for chunk in tqdm(chunks):
        for args in chunk:
            chunk_results.append(process_peptidoform(args))
    """
    pin_in = [item for sublist in chunk_results for item in sublist]

    log_info("Step 8: Concatenating results")
    concatenated_df = (
        pl.concat(pin_in)
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
    concatenated_df.write_csv("outfile.pin", separator="\t")


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
    df_psms = pl.DataFrame(df_psms)
    df_psms = df_psms.filter(~df_psms["peptide"].str.contains("U"))
    df_psms = df_psms.sort("rt")

    calculate_features(
        df_psms,
        df_fragment,
        df_fragment_max,
        df_fragment_max_peptide,
        pickle_config=pickle_config,
        deeplc_model=deeplc_model,
        config=config,
        spectra_data=spectra_data,
    )

    log_info("Done running MuMDIA...")
    run_mokapot()


if __name__ == "__main__":
    # In practice, load your input DataFrames (e.g., from parquet files) and then call main().
    # For demonstration, we call run_mokapot().
    run_mokapot()
