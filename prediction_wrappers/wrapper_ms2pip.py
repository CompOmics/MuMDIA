"""
MS2PIP Fragment Intensity Prediction Wrapper for MuMDIA

This module provides interfaces to MS2PIP (MS2 Peak Intensity Prediction) for
predicting fragment ion intensities in tandem mass spectra. MS2PIP uses deep
learning models to predict the intensity of b and y fragment ions based on
peptide sequence and precursor charge state.

Key Features:
- Batch processing for efficient MS2PIP predictions
- PSM format conversion for MS2PIP compatibility  
- Fragment intensity prediction caching via pickle files
- Integration with Polars DataFrames for fast data processing
- Support for HCD fragmentation model (HCD2021)

The predictions are used as features in the MuMDIA machine learning pipeline
to improve peptide-spectrum match scoring and validation.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from ms2pip.core import predict_batch
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from tqdm import tqdm


def plot_performance(psm_list, preds, outfile="plot.png"):
    """
    Create a scatter plot comparing observed vs predicted retention times.

    Args:
        psm_list: List of PSM objects with retention_time attributes
        preds: Array of predicted retention times
        outfile: Output filename for the plot (default: "plot.png")
    """
    plt.scatter([v.retention_time for v in psm_list], preds, s=3, alpha=0.05)
    plt.xlabel("Observed retention time (min)")
    plt.ylabel("Predicted retention time (min)")
    plt.savefig(outfile)
    plt.close()


def batch_process_predict_batch(
    input_list, batch_size, process_function, n_processes=64
):
    """
    Process large PSM lists in batches for efficient MS2PIP prediction.

    This function splits the input into manageable batches to avoid memory
    issues with large datasets and enables parallel processing.

    Args:
        input_list: List of PSM objects for prediction
        batch_size: Number of PSMs to process per batch
        process_function: Function to apply to each batch (typically predict_batch)
        n_processes: Number of parallel processes to use (default: 64)

    Returns:
        Combined results from all batches
    """
    batches = [
        input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)
    ]

    results = []
    for batch in tqdm(batches):
        results.extend(process_function(batch, processes=n_processes, model="HCD2021"))

    return results


def get_predictions_fragment_intensity(df_psms):
    """
    Generate MS2PIP fragment intensity predictions for unique peptide-charge combinations.

    This function converts PSM data to MS2PIP format, runs predictions, and returns
    a dictionary mapping peptidoform strings to fragment ion intensity predictions.

    Args:
        df_psms: Polars DataFrame with columns ['peptide', 'charge', 'rt']

    Returns:
        Dictionary mapping peptidoform strings (peptide/charge) to fragment
        intensity dictionaries with keys like 'b1/1', 'y1/1', etc.
    """
    # Get unique peptide-charge combinations to avoid redundant predictions
    df_ms2pip = df_psms.unique(subset=["peptide", "charge"])

    # Convert to MS2PIP PSM format with peptidoform notation (peptide/charge)
    psm_list = [
        PSM(
            peptidoform=seq + "/" + str(charge),  # MS2PIP requires this format
            retention_time=tr,
            spectrum_id=idx,
            precursor_charge=charge,
        )
        for idx, (seq, tr, charge) in enumerate(
            zip(df_ms2pip["peptide"], df_ms2pip["rt"], df_ms2pip["charge"])
        )
    ]
    psm_list = PSMList(psm_list=psm_list)

    # Run MS2PIP predictions in batches for memory efficiency
    ms2pip_predictions = batch_process_predict_batch(psm_list, 500000, predict_batch)

    # Convert predictions to dictionary format for easy lookup
    ms2pip_predictions_dict = {}

    for pred in ms2pip_predictions:
        k = str(pred.psm.peptidoform)  # Use peptidoform as key
        try:
            # Convert b-ion predictions from log2 scale and create fragment keys
            ms2pip_predictions_dict[k] = dict(
                [
                    ("b%s/1" % (idx + 1), 2**v)  # Convert log2 back to linear scale
                    for idx, v in enumerate(pred.predicted_intensity["b"])
                ]
            )
        except:
            print(k)
        try:
            ms2pip_predictions_dict[k].update(
                dict(
                    [
                        ("y%s/1" % (idx + 1), 2**v)
                        for idx, v in enumerate(pred.predicted_intensity["y"])
                    ]
                )
            )
        except:
            print(k)

    return ms2pip_predictions_dict


def get_predictions_fragment_intensity_main_loop(
    df_psms: pl.DataFrame,
    df_fragment: pl.DataFrame,
    read_ms2pip_pickle: bool = False,
    write_ms2pip_pickle: bool = False,
) -> pl.DataFrame:
    """
    Get fragment intensity predictions using MS2PIP.
    Args:
        df_psms: PSM dataframe with the following columns:
            - peptide: Peptide sequence
            - spectrum_id: Spectrum ID
    """
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

    return df_fragment, ms2pip_predictions
