#!/usr/bin/env python
import os
import logging
from tqdm import tqdm
import numpy as np
import polars as pl
import mokapot
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.neural_network import MLPClassifier
from typing import List
from sklearn.linear_model import Lasso
from prediction_wrappers.wrapper_deeplc import get_predictions_retention_time_mainloop
from feature_generators.features_retention_time import add_retention_time_features
from feature_generators.features_general import add_count_and_filter_peptides
from feature_generators.features_fragment_intensity import (
    get_features_fragment_intensity,
)
from prediction_wrappers.wrapper_ms2pip import (
    get_predictions_fragment_intensity_main_loop,
    get_predictions_fragment_intensity,
)
import xgboost as xgb
from mokapot.model import PercolatorModel
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from copy import deepcopy

# Import numba and set up Numba decorators
import numba as nb

# Set maximum threads for Polars to one to avoid oversubscription
os.environ["POLARS_MAX_THREADS"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# Assumes that log_info is defined in utilities.logger.
from utilities.logger import log_info

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
def compute_percentiles_nb(data, qs):
    """
    Compute an array of percentiles given a 1D array and an array of q values.
    """
    m = qs.shape[0]
    result = np.empty(m, dtype=np.float64)
    for i in range(m):
        result[i] = numba_percentile(data, qs[i])
    return result


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
    model.add(Dense(100, input_dim=124, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(10, activation="relu"))
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
    # model = KerasClassifier(
    #    build_fn=create_model, epochs=100, batch_size=1000, verbose=10
    # )
    results, models = mokapot.brew(psms)  # mokapot.Model(model), folds=3
    result_files = results.to_txt(dest_dir=output_dir)
    print(result_files)

    input()


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


def add_feature_columns_nb(data, feature_name, values, method, pad_size=10):
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
        computed = compute_percentiles_nb(data, qs)
    elif method == "top":
        computed = compute_top_nb(data, required_length)
    else:
        raise ValueError(f"Unknown method: {method}")
    # Ensure computed is of the required length
    if computed.size < required_length:
        padded = np.zeros(required_length, dtype=np.float64)
        padded[: computed.size] = computed
        computed = padded
    else:
        computed = computed[:required_length]
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
    ],
    collapse_mean_columns: List[str] = [
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
    ],
    collapse_sum_columns: List[str] = [
        "hyperscore",
        "delta_rt_model",
        "matched_peaks",
        "longest_b",
        "longest_y",
        "matched_intensity_pct",
        "fragment_intensity",
        "rt",
        "rt_predictions",
        "rt_prediction_error_abs",
        "rt_prediction_error_abs_relative",
        "precursor_ppm",
        "fragment_ppm",
        "delta_next",
        "delta_best",
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


def run_peptidoform_correlation(
    correlations_list,
    collect_distributions: List[int] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    collect_top: List[int] = [1, 2, 3, 4, 5],
    pad_size=10,
):
    """
    Compute correlation-based features and return a one-row Polars DataFrame.
    """
    correlations, correlation_matrix_psm_ids, correlation_matrix_frag_ids = (
        correlations_list
    )
    feature_dict = {}
    params = [
        (
            correlation_matrix_psm_ids,
            "distribution_correlation_matrix_psm_ids",
            collect_distributions,
            "percentile",
            len(collect_distributions),
        ),
        (
            correlation_matrix_frag_ids,
            "distribution_correlation_matrix_frag_ids",
            collect_distributions,
            "percentile",
            len(collect_distributions),
        ),
        (
            correlations,
            "distribution_correlation_individual",
            collect_distributions,
            "percentile",
            len(collect_distributions),
        ),
        (
            correlation_matrix_psm_ids,
            "top_correlation_matrix_psm_ids",
            collect_top,
            "top",
            pad_size,
        ),
        (
            correlation_matrix_frag_ids,
            "top_correlation_matrix_frag_ids",
            collect_top,
            "top",
            pad_size,
        ),
        (correlations, "top_correlation_individual", collect_top, "top", pad_size),
    ]
    for data, feat_name, values, method, ps in params:
        # Here, for percentiles and top values, we use the Numba-accelerated add_feature_columns_nb.
        feature_dict.update(
            add_feature_columns_nb(data, feat_name, values, method, pad_size=ps)
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


def calculate_features(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    df_fragment_max: pl.DataFrame,
    df_fragment_max_peptide: pl.DataFrame,
    filter_rel_rt_error: float = 0.1,
    min_occurrences: int = 10,
    filter_max_apex_rt: float = 0.75,
    config: dict = {},
    deeplc_model=None,
    write_deeplc_pickle: bool = False,
    write_ms2pip_pickle: bool = False,
    write_correlation_pickles: bool = False,
    read_deeplc_pickle: bool = False,
    read_ms2pip_pickle: bool = False,
    read_correlation_pickles: bool = False,
    parallel_workers: int = 24,  # Adjust to number of physical cores
    chunk_size: int = 500,  # Increase chunk size to reduce overhead
) -> None:
    """
    Process the PSM and fragment DataFrames, compute features, and save the output.
    This function uses parallel processing with task chunking.
    """
    log_info("Obtaining retention time predictions for the main loop...")
    _, _, predictions_deeplc = get_predictions_retention_time_mainloop(
        df_psms, write_deeplc_pickle, read_deeplc_pickle, deeplc_model
    )

    log_info("Obtaining features retention time...")
    df_psms = add_retention_time_features(
        df_psms, predictions_deeplc, filter_rel_rt_error=0.2
    )

    log_info(
        "Counting individual peptides per MS2 and filtering by minimum occurrences"
    )
    df_psms = add_count_and_filter_peptides(df_psms, min_occurrences)

    log_info("Obtaining fragment intensity predictions for the main loop...")
    df_fragment, ms2pip_predictions = get_predictions_fragment_intensity_main_loop(
        df_psms,
        df_fragment,
        read_ms2pip_pickle=read_ms2pip_pickle,
        write_ms2pip_pickle=write_ms2pip_pickle,
    )

    log_info("Obtaining features fragment intensity predictions...")
    fragment_dict, correlations_fragment_dict = get_features_fragment_intensity(
        ms2pip_predictions,
        df_fragment,
        df_fragment_max_peptide,
        read_correlation_pickles=read_correlation_pickles,
        write_correlation_pickles=write_correlation_pickles,
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
    df_fragment: pl.DataFrame = None,
    df_psms: pl.DataFrame = None,
    df_fragment_max: pl.DataFrame = None,
    df_fragment_max_peptide: pl.DataFrame = None,
    config: dict = {},
    deeplc_model=None,
    write_deeplc_pickle: bool = False,
    write_ms2pip_pickle: bool = False,
    write_correlation_pickles: bool = False,
    read_deeplc_pickle: bool = False,
    read_ms2pip_pickle: bool = False,
    read_correlation_pickles: bool = False,
) -> None:
    """
    Main function to run the complete analysis pipeline.
    """
    df_psms = pl.DataFrame(df_psms)
    df_psms = df_psms.filter(~df_psms["peptide"].str.contains("U"))
    df_psms = df_psms.sort("rt")

    calculate_features(
        df_psms,
        df_fragment,
        df_fragment_max,
        df_fragment_max_peptide,
        write_deeplc_pickle=write_deeplc_pickle,
        write_ms2pip_pickle=write_ms2pip_pickle,
        write_correlation_pickles=write_correlation_pickles,
        read_deeplc_pickle=read_deeplc_pickle,
        read_ms2pip_pickle=read_ms2pip_pickle,
        read_correlation_pickles=read_correlation_pickles,
        deeplc_model=deeplc_model,
        config=config,
    )

    log_info("Done running MuMDIA...")
    run_mokapot()


if __name__ == "__main__":
    # In practice, load your input DataFrames (e.g., from parquet files) and then call main().
    # For demonstration, we call run_mokapot().
    run_mokapot()
