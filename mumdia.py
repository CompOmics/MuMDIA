from deeplc_wrapper import get_predictions_retentiontime
from mzml_parser import get_spectra_mzml
from parquet_parser import parquet_reader
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import polars as pl
from tqdm import tqdm
from ms2pip_wrapper import get_predictions_fragment_intensity
from scipy.stats import pearsonr
import mokapot
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import logging
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from typing import Tuple, List
from sklearn.linear_model import Lasso

# TODO make a logger module in a seperate file
try:
    from run import log_info
except:
    import datetime

    start_time = datetime.datetime.now()
    from rich.logging import RichHandler
    from rich.console import Console

    console = Console()

    def log_info(message):
        current_time = datetime.datetime.now()
        elapsed = current_time - start_time
        # Add Rich markup for coloring and styling
        console.log(
            f"[green]{current_time:%Y-%m-%d %H:%M:%S}[/green] [bold blue]{message}[/bold blue] - Elapsed Time: [yellow]{elapsed}[/yellow]"
        )


from typing import Any
import xgboost as xgb
from mokapot.model import PercolatorModel
from deeplc_wrapper import predict_deeplc_pl

from keras.models import Sequential
from keras.layers import Dense

# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# standard_features = ["SpecId", "Label", "ExpMass", "CalcMass", "Peptide", "ScanNr"]
last_features = ["proteins"]


# Function to create the Keras model
def create_model():
    model = Sequential()
    model.add(Dense(20, input_dim=169, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(10, activation="relu"))
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

    model = MLPClassifier(hidden_layer_sizes=(10, 5, 5, 5), max_iter=50)
    # Wrap the model with KerasClassifier
    # model = KerasClassifier(
    #    build_fn=create_model, epochs=100, batch_size=10, verbose=10
    # )
    # results, models = mokapot.brew(psms, mokapot.Model(mlp), folds=5)
    # model = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1)
    # results, models = mokapot.brew(
    #    psms, mokapot.Model(model, max_iter=25), max_workers=20, folds=10
    # )

    # Conduct the mokapot analysis:
    results, models = mokapot.brew(psms, PercolatorModel(max_iter=50), folds=10)

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


def match_fragments(
    df_fragment_sub_peptidoform: pl.DataFrame, ms2pip_predictions: dict
) -> Tuple[float, np.ndarray, List[np.ndarray]]:
    """
    Match fragments and calculate their correlation.

    Parameters:
    - df_fragment_sub_peptidoform (pl.DataFrame): A DataFrame containing fragment data.
    - ms2pip_predictions (dict): A dictionary of MS2PIP predictions.

    Returns:
    - Tuple[float, np.ndarray, List[np.ndarray]]: A tuple containing the correlation result,
      a numpy array of PSM ID correlation matrix, and a list of numpy arrays of fragment ID correlation matrices.
    """
    intensity_matrix = df_fragment_sub_peptidoform.pivot(
        index="psm_id", columns="fragment_name", values="fragment_intensity"
    ).fill_null(0.0)

    if intensity_matrix.shape[0] > 1:
        correlation_matrix_frag_ids = intensity_matrix[:, 1:].corr()
    else:
        correlation_matrix_frag_ids = pl.DataFrame()

    intensity_matrix_transposed = intensity_matrix.transpose()
    correlation_matrix_psm_ids = np.square(
        intensity_matrix_transposed.corr().to_numpy().flatten()
    )
    correlation_matrix_psm_ids = np.sort(correlation_matrix_psm_ids)
    correlation_matrix_psm_ids = correlation_matrix_psm_ids[
        : -len(intensity_matrix_transposed)
    ]

    pred_frag_intens = np.array(
        [ms2pip_predictions.get(fid, 0.0) for fid in intensity_matrix.columns]
    )

    correlation_result = pearson_np(intensity_matrix_transposed, pred_frag_intens)

    """
    [
        (
            correlation_matrix_frag_ids[sorted_predicted_intens_names[i]].to_numpy()
            if sorted_predicted_intens_names[i] in correlation_matrix_frag_ids.columns
            else np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        for i in range(3)
    ]
    """

    return (
        correlation_result,
        correlation_matrix_psm_ids,
        correlation_matrix_frag_ids,
    )


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
        pl.Series(
            "is_decoy",
            df_psms_sub_peptidoform_collapsed["is_decoy"].apply(transform_bool),
            dtype=pl.Int32,
        )
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
    for psm_id, sub_df_fragment in df_fragment.groupby("psm_id"):
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


def deconv_filter_fragment_spectra(
    ms2pip_predictions: dict,
    df_fragment: pl.DataFrame,
    min_coefficient=0.0000000001,
    alpha_penalty=0.1,
):
    # Flatten the nested dictionary into a single mapping
    flat_ms2pip = {
        (peptide, fragment): value
        for peptide, fragments in ms2pip_predictions.items()
        for fragment, value in fragments.items()
    }

    df_fragment = df_fragment.with_columns(
        pl.concat_str([pl.col("peptide"), pl.col("charge")], separator="/").alias(
            "peptidoform"
        )
    )

    df_fragment = df_fragment.with_columns(
        pl.struct(["peptidoform", "fragment_name"])
        .apply(lambda x: flat_ms2pip.get((x["peptidoform"], x["fragment_name"]), None))
        .alias("predicted_intensity")
    )

    df_fragment = df_fragment.with_columns(pl.col("predicted_intensity").fill_null(0.0))

    selected_peptidoforms = []
    for group in df_fragment.groupby("scannr"):

        df_scan = group[1].sort("peak_identifier")

        exp_spectrum_intensity = (
            df_scan.unique(subset=["peak_identifier"], keep="first")[
                "fragment_intensity"
            ]
            .fill_null(0.0)
            .to_numpy()
        )
        exp_spectrum_intensity_normalized = exp_spectrum_intensity / np.sum(
            exp_spectrum_intensity
        )

        pivot_df = df_scan.pivot(
            values="predicted_intensity",
            index="peptidoform",
            columns="peak_identifier",
            aggregate_function="sum",
        )
        pivot_df_intensity = pivot_df[pivot_df.columns[1:]].fill_null(0.0).to_numpy().T

        lasso = Lasso(alpha=alpha_penalty, positive=True, max_iter=100000)
        lasso.fit(pivot_df_intensity, exp_spectrum_intensity_normalized)
        coefficients = lasso.coef_

        selected_peptidoforms_filter = pl.Series(coefficients > min_coefficient)

        selected_peptidoforms.extend(
            list(pivot_df[pivot_df.columns[0]].filter(selected_peptidoforms_filter))
        )
    return selected_peptidoforms


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
    if not read_deeplc_pickle:
        if deeplc_model is None:
            dlc_calibration, dlc_transfer_learn, predictions_deeplc = (
                get_predictions_retentiontime(df_psms)
            )
        else:
            predictions_deeplc = predict_deeplc_pl(df_psms, deeplc_model)

    if write_deeplc_pickle:
        if deeplc_model is None:
            with open("dlc_calibration.pkl", "wb") as f:
                pickle.dump(dlc_calibration, f)
            with open("dlc_transfer_learn.pkl", "wb") as f:
                pickle.dump(dlc_transfer_learn, f)
        with open("predictions_deeplc.pkl", "wb") as f:
            pickle.dump(predictions_deeplc, f)
    if read_deeplc_pickle:
        with open("dlc_calibration_first.pkl", "rb") as f:
            dlc_calibration = pickle.load(f)
        with open("dlc_transfer_learn_first.pkl", "rb") as f:
            dlc_transfer_learn = pickle.load(f)
        with open("predictions_deeplc.pkl", "rb") as f:
            predictions_deeplc = pickle.load(f)

    log_info("Obtained retention time predictions...")

    df_psms = df_psms.join(predictions_deeplc, on="peptide", how="left")
    max_rt = df_psms["rt"].max()

    df_psms = df_psms.with_columns(
        pl.Series(
            "rt_prediction_error_abs",
            abs(df_psms["rt"] - df_psms["rt_predictions"]),
        )
    )
    df_psms = df_psms.with_columns(
        pl.Series(
            "rt_prediction_error_abs_relative",
            abs(df_psms["rt"] - df_psms["rt_predictions"]) / max_rt,
        )
    )
    df_psms = df_psms.filter(
        df_psms["rt_prediction_error_abs_relative"] < filter_rel_rt_error
    )

    peptide_counts = df_psms.groupby("peptide").agg(pl.count().alias("count"))

    df_psms = (
        df_psms.join(peptide_counts, on="peptide")
        .filter(pl.col("count") >= min_occurrences)
        .drop("count")
    )

    if not read_ms2pip_pickle:
        ms2pip_predictions = get_predictions_fragment_intensity(df_psms)
    if write_ms2pip_pickle:
        with open("ms2pip_predictions.pkl", "wb") as f:
            pickle.dump(ms2pip_predictions, f)
    if read_ms2pip_pickle:
        with open("ms2pip_predictions.pkl", "rb") as f:
            ms2pip_predictions = pickle.load(f)

    df_fragment = df_fragment.filter(df_fragment["psm_id"].is_in(df_psms["psm_id"]))

    df_fragment = df_fragment.with_columns(
        pl.Series(
            "fragment_name",
            df_fragment["fragment_type"]
            + df_fragment["fragment_ordinals"]
            + "/"
            + df_fragment["fragment_charge"],
        )
    )

    psm_dict = {}
    fragment_dict = {}
    correlations_fragment_dict = {}

    log_info("Starting to process peptidoforms...")
    if not read_correlation_pickles:
        flat_ms2pip = {
            (peptide, fragment): value
            for peptide, fragments in ms2pip_predictions.items()
            for fragment, value in fragments.items()
        }

        df_fragment = df_fragment.with_columns(
            pl.concat_str([pl.col("peptide"), pl.col("charge")], separator="/").alias(
                "peptidoform"
            )
        )

        df_fragment = df_fragment.with_columns(
            pl.struct(["peptidoform", "fragment_name"])
            .apply(
                lambda x: flat_ms2pip.get((x["peptidoform"], x["fragment_name"]), None)
            )
            .alias("predicted_intensity")
        )

        df_fragment = df_fragment.with_columns(
            pl.col("predicted_intensity").fill_null(0.0)
        )
        aggregated_fragment_df = df_fragment.groupby("psm_id").agg(
            [
                pl.col("fragment_name").alias(
                    "fragment_names"
                ),  # Aggregates as a list by default
                pl.col("fragment_intensity").alias("observed_intensities"),
                pl.col("predicted_intensity").alias("predicted_intensities"),
            ]
        )

        # Step 2: Join the aggregated fragment data with the PSM dataframe
        psm_df_with_fragments = df_psms.join(
            aggregated_fragment_df, on="psm_id", how="left"
        )

        # Step 3: Convert lists to numpy arrays
        psm_df_with_fragments = psm_df_with_fragments.with_columns(
            [
                pl.col("fragment_names")
                .apply(lambda x: np.array(x) if x is not None else np.array([]))
                .alias("fragment_names"),
                pl.col("observed_intensities")
                .apply(lambda x: np.array(x) if x is not None else np.array([]))
                .alias("observed_intensities"),
                pl.col("predicted_intensities")
                .apply(lambda x: np.array(x) if x is not None else np.array([]))
                .alias("predicted_intensities"),
            ]
        )

        print(psm_df_with_fragments)
        input()
        print(df_psms.describe())
        print(df_psms.columns)
        print(df_psms)
        input()
        print(df_fragment.describe())
        print(df_fragment.columns)
        print(df_fragment)
        input()

        """
        # Create a column combining peptide and charge to avoid repetitive string operations
        df_fragment = df_fragment.with_columns(
            (pl.col("peptide") + "/" + pl.col("charge").cast(str)).alias(
                "peptide_charge"
            )
        )

        # Group by peptide and charge
        grouped_df = df_fragment.groupby("peptide_charge").agg(
            [
                pl.col("peptide").first().alias("peptide"),
                pl.col("charge").first().alias("charge"),
                pl.col("rt"),
            ]
        )

        # Filter the predictions and match fragments
        for group in tqdm(grouped_df.iter_rows(named=True)):
            peptide_charge = group["peptide_charge"]
            peptidoform = group["peptide"]
            charge = group["charge"]

            preds = ms2pip_predictions.get(peptide_charge)
            if not preds:
                continue

            df_fragment_max_peptide_sub = df_fragment_max_peptide.filter(
                pl.col("peptide") == peptidoform
            )

            # Use logical_and for filtering floating-point columns
            df_fragment_sub_peptidoform = df_fragment.filter(
                pl.col("peptide_charge") == peptide_charge
            ).filter(
                (pl.col("rt") - df_fragment_max_peptide_sub["rt"]).abs()
                < filter_max_apex_rt
            )

            if df_fragment_sub_peptidoform.height == 0:
                continue

            correlations, correlation_matrix_psm_ids, correlation_matrix_frag_ids = (
                match_fragments(df_fragment_sub_peptidoform, preds)
            )

            fragment_dict[peptide_charge] = df_fragment_sub_peptidoform
            correlations_fragment_dict[peptide_charge] = [
                correlations,
                correlation_matrix_psm_ids,
                correlation_matrix_frag_ids,
            ]

        # Partition the DataFrame by peptide and charge
        partitions = df_fragment.partition_by(["peptide", "charge"])

        for partition in tqdm(partitions):
            peptidoform = partition["peptide"][0]
            charge = partition["charge"][0]

            preds = ms2pip_predictions.get(f"{peptidoform}/{charge}")
            if not preds:
                continue

            df_fragment_max_peptide_sub = df_fragment_max_peptide.filter(
                df_fragment_max_peptide["peptide"] == peptidoform
            )

            # Filtering using vectorized operations
            df_fragment_sub_peptidoform = partition.filter(
                abs(partition["rt"] - df_fragment_max_peptide_sub["rt"])
                < filter_max_apex_rt
            )

            if df_fragment_sub_peptidoform.shape[0] == 0:
                continue

            correlations, correlation_matrix_psm_ids, correlation_matrix_frag_ids = (
                match_fragments(df_fragment_sub_peptidoform, preds)
            )
            fragment_dict[f"{peptidoform}/{charge}"] = df_fragment_sub_peptidoform
            correlations_fragment_dict[f"{peptidoform}/{charge}"] = [
                correlations,
                correlation_matrix_psm_ids,
                correlation_matrix_frag_ids,
            ]
        """
        """
        for (peptidoform, charge), df_fragment_sub_peptidoform in tqdm(
            df_fragment.groupby(["peptide", "charge"])
        ):
            preds = ms2pip_predictions.get(f"{peptidoform}/{charge}")
            if not preds:
                continue
            df_fragment_max_peptide_sub = df_fragment_max_peptide.filter(
                df_fragment_max_peptide["peptide"] == peptidoform
            )
            
            try:
                df_fragment_sub_peptidoform = df_fragment_sub_peptidoform.filter(
                    abs(
                        df_fragment_sub_peptidoform["rt"]
                        - df_fragment_max_peptide_sub["rt"]
                    )
                    < filter_max_apex_rt
                )
            except:
                continue

            if df_fragment_sub_peptidoform.shape[0] == 0:
                continue

            
            correlations, correlation_matrix_psm_ids, correlation_matrix_frag_ids = (
                match_fragments(df_fragment_sub_peptidoform, preds)
            )
            fragment_dict[f"{peptidoform}/{charge}"] = df_fragment_sub_peptidoform
            correlations_fragment_dict[f"{peptidoform}/{charge}"] = [
                correlations,
                correlation_matrix_psm_ids,
                correlation_matrix_frag_ids,
            
            ]
        """
        if write_correlation_pickles:
            with open("fragment_dict.pkl", "wb") as f:
                pickle.dump(fragment_dict, f)
            with open("correlations_fragment_dict.pkl", "wb") as f:
                pickle.dump(correlations_fragment_dict, f)
    if read_correlation_pickles:
        with open("fragment_dict.pkl", "rb") as f:
            fragment_dict = pickle.load(f)
        with open("correlations_fragment_dict.pkl", "rb") as f:
            correlations_fragment_dict = pickle.load(f)

    for (peptidoform, charge), df_sub_peptidoform in tqdm(
        df_psms.groupby(["peptide", "charge"])
    ):
        psm_dict[f"{peptidoform}/{charge}"] = df_sub_peptidoform

    peptidoform_args = [
        (psm_dict[k], fragment_dict[k], correlations_fragment_dict[k])
        for k in psm_dict.keys()
        if k in correlations_fragment_dict.keys()
    ]

    pin_in = []

    # pin_in = process_peptidoforms(peptidoform_args)

    for (
        df_psms_sub_peptidoform,
        df_fragment_sub_peptidoform,
        correlations_list,
    ) in tqdm(peptidoform_args):
        pin_in.append(
            run_peptidoform(
                df_psms_sub_peptidoform,
                # df_fragment_sub_peptidoform,
                correlations_list,
                config["mumdia"]["rescoring_features"],
            )
        )

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
    parquet_file_results: str = "",
    parquet_file_fragments: str = "",
    df_fragment: pl.DataFrame = None,
    df_psms: pl.DataFrame = None,
    df_fragment_max: pl.DataFrame = None,
    df_fragment_max_peptide: pl.DataFrame = None,
    config: dict = {},
    q_value_filter: float = 0.1,
    deeplc_model=None,
    write_deeplc_pickle: bool = False,
    write_ms2pip_pickle: bool = False,
    write_parquet_pickle: bool = False,
    write_correlation_pickles: bool = False,
    read_deeplc_pickle: bool = False,
    read_ms2pip_pickle: bool = False,
    read_parquet_pickle: bool = False,
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

    if (
        not read_parquet_pickle
        and len(parquet_file_results) > 0
        and len(parquet_file_fragments) > 0
    ):
        log_info("Reading parquet files...")
        df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide = parquet_reader(
            parquet_file_results=parquet_file_results,
            parquet_file_fragments=parquet_file_fragments,
            q_value_filter=q_value_filter,
        )

    if write_parquet_pickle:
        with open("df_fragment.pkl", "wb") as f:
            pickle.dump(df_fragment, f)
        with open("df_psms.pkl", "wb") as f:
            pickle.dump(df_psms, f)
        with open("df_fragment_max.pkl", "wb") as f:
            pickle.dump(df_fragment_max, f)
        with open("df_fragment_max_peptide.pkl", "wb") as f:
            pickle.dump(df_fragment_max_peptide, f)
    if read_parquet_pickle:
        with open("df_fragment.pkl", "rb") as f:
            df_fragment = pickle.load(f)
        with open("df_psms.pkl", "rb") as f:
            df_psms = pickle.load(f)
        with open("df_fragment_max.pkl", "rb") as f:
            df_fragment_max = pickle.load(f)
        with open("df_fragment_max_peptide.pkl", "rb") as f:
            df_fragment_max_peptide = pickle.load(f)

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
