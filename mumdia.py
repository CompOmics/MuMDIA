from tqdm import tqdm
import numpy as np
import polars as pl
from tqdm import tqdm
import mokapot
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import logging
import os
from sklearn.neural_network import MLPClassifier
from typing import Tuple, List
from sklearn.linear_model import Lasso
from prediction_wrappers.wrapper_deeplc import (
    get_predictions_retention_time_mainloop,
)
from feature_generators.features_retention_time import add_retention_time_features
from feature_generators.features_general import add_count_and_filter_peptides
from feature_generators.features_fragment_intensity import (
    get_features_fragment_intensity,
)
from prediction_wrappers.wrapper_ms2pip import (
    get_predictions_fragment_intensity_main_loop,
    get_predictions_fragment_intensity,
)

os.environ["POLARS_MAX_THREADS"] = "1"

# TODO make a logger module in a seperate file
from utilities.logger import log_info

from typing import Any
import xgboost as xgb
from mokapot.model import PercolatorModel

from prediction_wrappers.wrapper_ms2pip import (
    get_predictions_fragment_intensity_main_loop,
)
from prediction_wrappers.wrapper_deeplc import (
    get_predictions_retention_time_mainloop,
)


from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import cross_val_score

from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# standard_features = ["SpecId", "Label", "ExpMass", "CalcMass", "Peptide", "ScanNr"]
last_features = ["proteins"]


# Function to create the Keras model
def create_model():
    model = Sequential()
    model.add(Dense(100, input_dim=169, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def transform_bool(value: bool) -> int:
    """
    Transform a boolean value to an integer.

    Parameters:
    - value (bool): A boolean value.

    Returns:
    - int: Returns -1 if the value is True, else 1.
    """
    return -1 if value else 1


def run_mokapot() -> None:
    """
    Run the mokapot analysis on PSMs (Peptide-Spectrum Matches) read from a PIN file.
    The results are saved in tab-delimited text files.

    Side effects:
    - Reads PSMs from "outfile.pin".
    - Saves the analysis results in "mokapot.psms.txt" and "mokapot.peptides.txt".
    """
    # Read the PSMs from the PIN file:
    psms = mokapot.read_pin("outfile.pin")

    # model = MLPClassifier(hidden_layer_sizes=(10, 5, 5, 5), max_iter=50)
    # Wrap the model with KerasClassifier
    model = KerasClassifier(
        build_fn=create_model, epochs=100, batch_size=1000, verbose=10
    )

    results, models = mokapot.brew(psms, mokapot.Model(model), folds=5)
    # model = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1)
    # results, models = mokapot.brew(
    #    psms, mokapot.Model(model, max_iter=25), max_workers=20, folds=10
    # )
    # model = create_model()
    # Conduct the mokapot analysis:
    # results, models = mokapot.brew(psms, model, folds=3)

    # Save the results to two tab-delimited files
    # "mokapot.psms.txt" and "mokapot.peptides.txt"
    result_files = results.to_txt()


# https://stackoverflow.com/questions/73658716/how-to-calculate-correlation-between-1d-numpy-array-and-every-column-of-a-2d-num
def corr_np(data1, data2):
    mean1 = data1.mean()
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()
    corr = ((data1 * data2).mean() - mean1 * mean2) / (std1 * std2)
    return corr


def pearson_np(x, y):
    return np.array([corr_np(x[:, i], y) for i in range(x.shape[1])])


import numpy as np


def collapse_columns(
    df_psms_sub_peptidoform: pl.DataFrame,
    collapse_max_columns: list[str] = [],
    collapse_min_columns: list[str] = [],
    collapse_mean_columns: list[str] = [],
    collapse_sum_columns: list[str] = [],
    get_first_entry: list[str] = [],
):
    # Combine operations into a single loop to reduce redundancy
    collapsed_columns = [df_psms_sub_peptidoform.select(get_first_entry).head(1)]
    operations = (
        ("max", collapse_max_columns),
        ("min", collapse_min_columns),
        ("mean", collapse_mean_columns),
        ("sum", collapse_sum_columns),
    )
    for op, collapse_list in operations:
        collapsed_columns.append(
            getattr(df_psms_sub_peptidoform[collapse_list], op)().rename(
                {col: f"{col}_{op}" for col in collapse_list}
            )
        )

    # Use Polars concat for efficient horizontal concatenation
    return pl.concat(collapsed_columns, how="horizontal")


def add_feature_columns(data, feature_name, values, method, pad_size=10):
    if len(data) == 0:
        data = np.zeros(pad_size)
    elif method == "percentile":
        if type(data[0]) != np.float64:
            data = np.concatenate(data)
        data = np.percentile(data, values)
        data = np.pad(data, (0, pad_size), mode="constant", constant_values=0.0)
    elif method == "top":
        if type(data[0]) != np.float64:
            data = np.concatenate(data)
        data = np.sort(data, kind="heapsort", axis=None)[::-1]
        data = np.pad(data, (0, pad_size), mode="constant", constant_values=0.0)

    column_names = [f"{feature_name}_{v}" for v in values]
    return pl.DataFrame({column_names[i]: data[i] for i in range(len(column_names))})


def run_peptidoform(
    df_psms_sub_peptidoform: pl.DataFrame,
    correlations_list: list[float],
    selected_features: List[str] = [],
    collect_distributions: list[int] = [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
    ],
    collect_top: list[int] = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ],
    collapse_max_columns: list[str] = [
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
    collapse_min_columns: list[str] = [
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
    collapse_mean_columns: list[str] = [
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
    collapse_sum_columns: list[str] = [
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
    get_first_entry: list[str] = [
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
    pad_size=10,
) -> pl.DataFrame:
    """
    Process peptidoform data to enrich it with correlation and other calculated features.

    Parameters:
    - df_psms_sub_peptidoform (pl.DataFrame): DataFrame containing PSMs data.
    - df_fragment_sub_peptidoform (pl.DataFrame): DataFrame containing fragment data.
    - correlations_list (list[float]): List of correlation values.

    Returns:
    - pl.DataFrame: Enriched DataFrame with additional features.
    """

    # Perform bulk operations for max, min, mean, and sum
    distribution_features = [
        v for v in selected_features if v.startswith("distribution_")
    ]
    distribution_features_with_percentile = [
        dist_f + "_" + str(v)
        for dist_f in distribution_features
        for v in collect_distributions
    ]

    top_features = [v for v in selected_features if v.startswith("top_")]
    top_features_with_percentile = [
        dist_f + "_" + str(v) for dist_f in top_features for v in collect_top
    ]

    df_psms_sub_peptidoform_collapsed = collapse_columns(
        df_psms_sub_peptidoform,
        collapse_max_columns=collapse_max_columns,
        collapse_min_columns=collapse_min_columns,
        collapse_mean_columns=collapse_mean_columns,
        collapse_sum_columns=collapse_sum_columns,
        get_first_entry=get_first_entry,
    )

    # Update selected features
    selected_features = [
        v for v in list(df_psms_sub_peptidoform_collapsed.columns) if v != "proteins"
    ]

    correlations, correlation_matrix_psm_ids, correlation_matrix_frag_ids = (
        correlations_list
    )

    df_psms_sub_peptidoform_collapsed = df_psms_sub_peptidoform_collapsed.with_columns(
        pl.concat(
            [
                add_feature_columns(
                    correlation_matrix_psm_ids,
                    "distribution_correlation_matrix_psm_ids",
                    collect_distributions,
                    "percentile",
                    pad_size=len(collect_distributions),
                ),
                add_feature_columns(
                    correlation_matrix_frag_ids,
                    "distribution_correlation_matrix_frag_ids",
                    collect_distributions,
                    "percentile",
                    pad_size=len(collect_distributions),
                ),
                add_feature_columns(
                    correlations,
                    "distribution_correlation_individual",
                    collect_distributions,
                    "percentile",
                    pad_size=len(collect_distributions),
                ),
                add_feature_columns(
                    correlation_matrix_psm_ids,
                    "top_correlation_matrix_psm_ids",
                    collect_top,
                    "top",
                    pad_size,
                ),
                add_feature_columns(
                    correlation_matrix_frag_ids,
                    "top_correlation_matrix_frag_ids",
                    collect_top,
                    "top",
                    pad_size,
                ),
                add_feature_columns(
                    correlations,
                    "top_correlation_individual",
                    collect_top,
                    "top",
                    pad_size,
                ),
            ],
            how="horizontal",
        )
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

    columns_for_pin = ["SpecId"]
    columns_for_pin.extend(
        [
            v
            for v in selected_features
            if not v.startswith("distribution_") and not v.startswith("top_")
        ]
    )
    columns_for_pin.extend(distribution_features_with_percentile)
    columns_for_pin.extend(top_features_with_percentile)
    columns_for_pin.extend(last_features)

    # shape of elution
    # number of matches
    # time of profile
    # order conservation
    # number of peaks
    # number of peaks matched
    # apex squared error
    # MS1 peak present
    # MS1 correlation
    # MS1 correlation with MS2

    # TODO the renaming, dropping, filling, and cloning can be done later in the pipeline this is more efficient
    return df_psms_sub_peptidoform_collapsed[columns_for_pin]


def dataframe_to_dict_fragintensity(df_fragment: pl.DataFrame) -> dict:
    """
    Converts a DataFrame of fragment intensities into a dictionary.

    Parameters:
    - df_fragment (pl.DataFrame): A DataFrame containing fragment intensity data.

    Returns:
    - dict: A dictionary with PSM IDs as keys and corresponding DataFrames as values.
    """
    fragment_dict = {}
    for psm_id, sub_df_fragment in df_fragment.group_by("psm_id"):
        fragment_dict[psm_id] = sub_df_fragment

    return fragment_dict


def run_peptidoform_wrapper(
    args: Tuple[pl.DataFrame, pl.DataFrame, List[float]]
) -> pl.DataFrame:
    """
    Wrapper function for running the peptidoform processing.

    Parameters:
    - args (Tuple[pl.DataFrame, pl.DataFrame, List[float]]): A tuple containing the DataFrames and list of correlations.

    Returns:
    - pl.DataFrame: The processed DataFrame from run_peptidoform function.
    """
    return run_peptidoform(*args)


def process_peptidoforms(
    peptidoform_args: List[Tuple[pl.DataFrame, pl.DataFrame, List[float]]],
    max_workers: int = 2,
) -> List[pl.DataFrame]:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(run_peptidoform_wrapper, peptidoform_args)
        return list(results)


def calculate_features(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    df_fragment_max: pl.DataFrame,
    df_fragment_max_peptide: pl.DataFrame,
    filter_rel_rt_error: float = 0.1,
    min_occurrences: int = 0,
    filter_max_apex_rt: float = 0.75,
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
    Process multiple peptidoforms in parallel using ProcessPoolExecutor.

    Parameters:
    - peptidoform_args (List[Tuple[pl.DataFrame, pl.DataFrame, List[float]]]): A list of arguments to pass to the peptidoform processing function.
    - max_workers (int, optional): The maximum number of worker processes. Default is 2.

    Returns:
    - List[pl.DataFrame]: A list of DataFrames, each representing processed data for a peptidoform.
    - df_fragment (pl.DataFrame): DataFrame containing fragment data.
    - df_fragment_max (pl.DataFrame): DataFrame containing maximum fragment information.
    - df_fragment_max_peptide (pl.DataFrame): DataFrame containing maximum fragment information for each peptide.
    - filter_rel_rt_error (float, optional): The relative retention time error filter threshold. Default is 0.05.
    - min_occurrences (int, optional): The minimum number of occurrences for a peptide to be included. Default is 0.
    - filter_max_apex_rt (float, optional): The filter threshold for maximum apex retention time. Default is 5.0.

    Side effects:
    - Processes the DataFrames to calculate additional features and updates them in-place.
    - Writes the results to a file "outfile.pin".
    """
    log_info("Obtaining retention time predictions for the main loop...")
    dlc_calibration, dlc_transfer_learn, predictions_deeplc = (
        get_predictions_retention_time_mainloop(
            df_psms, write_deeplc_pickle, read_deeplc_pickle, deeplc_model
        )
    )

    log_info("Obtaining features retention time...")
    # TODO make it adjustable what features to include
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

    #########
    # Step 6: Go from peptidoform ID to subslice of the dataframe
    #########
    log_info("Step 6")

    psm_dict = {}
    for (peptidoform, charge), df_sub_peptidoform in tqdm(
        df_psms.group_by(["peptide", "charge"])
    ):
        psm_dict[f"{peptidoform}/{charge}"] = df_sub_peptidoform

    peptidoform_args = [
        (psm_dict[k], fragment_dict[k], correlations_fragment_dict[k])
        for k in psm_dict.keys()
        if k in correlations_fragment_dict.keys()
    ]

    pin_in = []

    # pin_in = process_peptidoforms(peptidoform_args)
    #########
    # Step 7: Calculate features for each peptidoform
    #########
    log_info("Step 7")
    for (
        df_psms_sub_peptidoform,
        df_fragment_sub_peptidoform,
        correlations_list,
    ) in tqdm(peptidoform_args):
        pin_in.append(
            run_peptidoform(
                df_psms_sub_peptidoform,
                correlations_list,
                config["mumdia"]["rescoring_features"],
            )
        )

    #########
    # Step 8: Concatenate the results
    #########
    log_info("Step 8")

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

    Parameters:
    - parquet_file_results (str): The file path for the parquet file containing results.
    - parquet_file_fragments (str): The file path for the parquet file containing fragment data.
    - q_value_filter (float, optional): The filter threshold for q-values. Default is 0.1.

    Side effects:
    - Reads data from the specified parquet files.
    - Performs data processing and feature calculation.
    - Outputs processed data to files and logs information about t he process.
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
    # main("results/results.sage.parquet", "results/matched_fragments.sage.parquet")
    # print("going to run mokapot")
    run_mokapot()
