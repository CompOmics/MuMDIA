"""
DeepLC Retention Time Prediction Wrapper for MuMDIA

This module provides interfaces to DeepLC for retention time prediction in
liquid chromatography-mass spectrometry workflows. DeepLC uses deep learning
to predict peptide retention times based on sequence and chemical properties.

Key Features:
- Transfer learning for experiment-specific model adaptation
- Batch processing for efficient predictions
- Retention time bounds calculation for targeted searches
- Model training and validation with experimental data
- Integration with PSM utils for standardized data formats

The retention time predictions are used for:
1. Quality filtering of peptide-spectrum matches
2. Partitioning mzML files for targeted searches
3. Feature generation for machine learning scoring
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from deeplc import DeepLC
from matplotlib import pyplot as plt
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList


def plot_performance(
    psm_list: PSMList, preds: np.ndarray, outfile: str = "plot.png"
) -> None:
    """
    Create a scatter plot comparing observed vs predicted retention times.

    Args:
        psm_list: List of PSM objects with retention time information
        preds: Array of predicted retention times
        outfile: Output file path for the plot
    """
    plt.scatter([v.retention_time for v in psm_list], preds, s=3, alpha=0.05)
    plt.xlabel("Observed retention time (min)")
    plt.ylabel("Predicted retention time (min)")
    plt.savefig(outfile)
    plt.close()


def predict_deeplc_pl_old(psm_df_pl: pl.DataFrame, dlc_model: DeepLC) -> pl.DataFrame:
    """
    Legacy function: Generate DeepLC retention time predictions for all PSMs.

    Note: This function processes all PSMs individually. Use predict_deeplc_pl() for
    better performance with deduplicated peptides.

    Args:
        psm_df_pl: Polars DataFrame with 'peptide' and 'rt' columns
        dlc_model: Trained DeepLC model

    Returns:
        DataFrame with added 'rt_predictions' column
    """
    # rt_train get the psm_id and add instread, then merge with prev
    psm_list = [
        PSM(peptidoform=seq, retention_time=tr, spectrum_id=idx)
        for idx, (seq, tr) in enumerate(zip(psm_df_pl["peptide"], psm_df_pl["rt"]))
    ]

    psm_list = PSMList(psm_list=psm_list)

    psm_df_pl = psm_df_pl.with_columns(
        pl.Series("rt_predictions", dlc_model.make_preds(psm_list))
    )

    return psm_df_pl


def predict_deeplc_pl(psm_df_pl: pl.DataFrame, dlc_model: DeepLC) -> pl.DataFrame:
    """
    Generate DeepLC retention time predictions with peptide deduplication for efficiency.

    This function optimizes prediction by computing RT predictions only for unique peptides,
    then merging results back to the full PSM DataFrame.

    Args:
        psm_df_pl: Polars DataFrame containing PSM data with 'peptide' and 'rt' columns
        dlc_model: Trained DeepLC model for retention time prediction

    Returns:
        Original DataFrame with added 'rt_predictions' column containing predicted retention times
    """
    # Extract unique peptide entries (deduplicate by peptide sequence)
    unique_peptides_df = psm_df_pl.unique(subset="peptide")

    # Create a list of PSM objects for the unique peptides
    psm_list = [
        PSM(peptidoform=row["peptide"], retention_time=row["rt"], spectrum_id=idx)
        for idx, row in enumerate(unique_peptides_df.to_dicts())
    ]
    psm_list = PSMList(psm_list=psm_list)

    # Compute predictions for the unique peptides only
    predictions = dlc_model.make_preds(psm_list)
    unique_peptides_df = unique_peptides_df.with_columns(
        pl.Series("rt_predictions", predictions)
    )

    # Merge the unique predictions back to the original DataFrame based on peptide sequence
    psm_df_pl = psm_df_pl.join(
        unique_peptides_df.select(["peptide", "rt_predictions"]),
        on="peptide",
        how="left",
    )

    return psm_df_pl


def predict_deeplc(psms_list: List[Tuple], dlc_model: DeepLC) -> np.ndarray:
    """
    Generate retention time predictions for a list of peptide tuples.

    Args:
        psms_list: List of tuples containing peptide information, where:
                  - psms_list[i][-1] contains the peptide sequence
                  - psms_list[i][-2] contains the spectrum ID
        dlc_model: Trained DeepLC model

    Returns:
        Array of predicted retention times for each peptide
    """
    psm_list_calib = [
        PSM(peptidoform=seq, spectrum_id=idx)
        for seq, idx in zip(
            [psl[-1] for psl in psms_list], [psl[-2] for psl in psms_list]
        )
    ]
    psm_list_calib = PSMList(psm_list=psm_list_calib)

    return dlc_model.make_preds(psm_list_calib)


def retrain_deeplc(
    df_psms: pl.DataFrame,
    plot_perf: bool = True,
    outfile_calib: Union[str, Path] = "deeplc_calibration.png",
    outfile_transf_learn: Union[str, Path] = "deeplc_transfer_learn.png",
    percentile_exclude: float = 95,
    q_value_filter: float = 0.01,
) -> Tuple[DeepLC, DeepLC, float]:
    """
    Retrain DeepLC model with transfer learning and calculate retention time error bounds.

    This function performs a two-stage DeepLC training:
    1. Calibration on high-confidence PSMs
    2. Transfer learning on filtered data (excluding high-error predictions)

    Args:
        df_psms: Polars DataFrame with PSM data including 'spectrum_q', 'peptide', 'rt', 'fragment_intensity'
        plot_perf: Whether to generate performance plots
        outfile_calib: Output path for calibration performance plot
        outfile_transf_learn: Output path for transfer learning performance plot
        percentile_exclude: Percentile threshold for excluding high-error predictions (default: 95)
        q_value_filter: Q-value threshold for filtering high-confidence PSMs (default: 0.01)

    Returns:
        Tuple containing:
        - dlc_calibration: Initial calibrated DeepLC model
        - dlc_transfer_learn: Transfer learning DeepLC model
        - perc_95: 95th percentile of absolute RT prediction errors (doubled for window size)
    """
    print(df_psms)
    df_psms_filtered = df_psms.filter(df_psms["spectrum_q"] < q_value_filter)
    rt_train = (
        df_psms_filtered.sort("fragment_intensity")
        .unique(subset=["peptide"])
        .select(["peptide", "rt"])
    )

    psm_list_calib = [
        PSM(peptidoform=seq, retention_time=tr, spectrum_id=idx)
        for idx, (seq, tr) in enumerate(zip(rt_train["peptide"], rt_train["rt"]))
    ]
    psm_list_calib = PSMList(psm_list=psm_list_calib)

    dlc_calibration = DeepLC(
        batch_num=1024000, deeplc_retrain=False, pygam_calibration=False, n_jobs=64
    )

    dlc_calibration.calibrate_preds(psm_list_calib)

    # Perform calibration, make predictions and calculate metrics
    preds = dlc_calibration.make_preds(psm_list_calib, calibrate=True)

    errors = abs(np.array(preds) - np.array([v.retention_time for v in psm_list_calib]))
    selection = errors < np.percentile(errors, percentile_exclude)
    psm_list_calib_filtered_percentile = [
        psm for psm, incl in zip(psm_list_calib, selection) if incl
    ]
    psm_list_calib_filtered_percentile = PSMList(
        psm_list=psm_list_calib_filtered_percentile
    )

    # Make a DeepLC object with the models trained previously
    dlc_transfer_learn = DeepLC(
        batch_num=1024000, deeplc_retrain=True, n_epochs=75, n_jobs=64
    )

    # Perform calibration, make predictions and calculate metrics
    dlc_transfer_learn.calibrate_preds(psm_list_calib_filtered_percentile)
    preds_transflearn = dlc_transfer_learn.make_preds(
        psm_list_calib_filtered_percentile
    )

    if plot_perf:
        plot_performance(psm_list_calib, preds, outfile=outfile_calib)
        plot_performance(
            psm_list_calib_filtered_percentile,
            preds_transflearn,
            outfile=outfile_transf_learn,
        )

    return (
        dlc_calibration,
        dlc_transfer_learn,
        np.percentile(
            abs(
                np.array([v.retention_time for v in psm_list_calib_filtered_percentile])
                - preds_transflearn
            ),
            95,
        )
        * 2,
    )


def get_predictions_retentiontime(
    df_psms: pl.DataFrame,
    plot_perf: bool = True,
    outfile_calib: Union[str, Path] = "deeplc_calibration.png",
    outfile_transf_learn: Union[str, Path] = "deeplc_transfer_learn.png",
    percentile_exclude: float = 50,
    return_obj: bool = True,
    return_predictions: bool = True,
    q_value_filter: float = 0.01,
) -> Union[Tuple[DeepLC, DeepLC], Tuple[DeepLC, DeepLC, pl.DataFrame]]:
    """
    Complete DeepLC training and prediction pipeline.

    Performs calibration, transfer learning, and generates predictions for all peptides.
    This is the main function for retention time prediction in the initial workflow.

    Args:
        df_psms: Polars DataFrame with PSM data
        plot_perf: Whether to generate performance plots
        outfile_calib: Output path for calibration plot
        outfile_transf_learn: Output path for transfer learning plot
        percentile_exclude: Percentile for filtering training data (default: 50)
        return_obj: Whether to return trained model objects
        return_predictions: Whether to return prediction DataFrame
        q_value_filter: Q-value threshold for high-confidence PSMs

    Returns:
        If return_obj and return_predictions: (dlc_calibration, dlc_transfer_learn, predictions_df)
        If return_obj only: (dlc_calibration, dlc_transfer_learn)
    """
    df_psms_filtered = df_psms.filter(df_psms["spectrum_q"] < q_value_filter)

    rt_train = (
        df_psms_filtered.sort("fragment_intensity")
        .unique(subset=["peptide"])
        .select(["peptide", "rt"])
    )

    psm_list_calib = [
        PSM(peptidoform=seq, retention_time=tr, spectrum_id=idx)
        for idx, (seq, tr) in enumerate(zip(rt_train["peptide"], rt_train["rt"]))
    ]
    psm_list_calib = PSMList(psm_list=psm_list_calib)

    dlc_calibration = DeepLC(
        batch_num=1024000, deeplc_retrain=False, pygam_calibration=False, n_jobs=64
    )

    dlc_calibration.calibrate_preds(psm_list_calib)

    # Perform calibration, make predictions and calculate metrics
    preds = dlc_calibration.make_preds(psm_list_calib, calibrate=True)

    errors = abs(np.array(preds) - np.array([v.retention_time for v in psm_list_calib]))
    selection = errors < np.percentile(errors, percentile_exclude)
    psm_list_calib_filtered_percentile = [
        psm for psm, incl in zip(psm_list_calib, selection) if incl
    ]
    psm_list_calib_filtered_percentile = PSMList(
        psm_list=psm_list_calib_filtered_percentile
    )

    # Make a DeepLC object with the models trained previously
    dlc_transfer_learn = DeepLC(
        batch_num=1024000, deeplc_retrain=True, n_epochs=50, n_jobs=64
    )

    # Perform calibration, make predictions and calculate metrics
    dlc_transfer_learn.calibrate_preds(psm_list_calib_filtered_percentile)
    preds_transflearn = dlc_transfer_learn.make_preds(
        psm_list_calib_filtered_percentile
    )

    if plot_perf:
        plot_performance(psm_list_calib, preds, outfile=outfile_calib)
        plot_performance(
            psm_list_calib_filtered_percentile,
            preds_transflearn,
            outfile=outfile_transf_learn,
        )

    # TODO here I reuse code, but this should stand on its own
    rt_train = (
        df_psms.sort("fragment_intensity")
        .unique(subset=["peptide"])
        .select(["peptide", "rt"])
    )

    # rt_train get the psm_id and add instread, then merge with prev
    psm_list_calib = [
        PSM(peptidoform=seq, retention_time=tr, spectrum_id=idx)
        for idx, (seq, tr) in enumerate(zip(rt_train["peptide"], rt_train["rt"]))
    ]
    psm_list_calib = PSMList(psm_list=psm_list_calib)

    rt_train = rt_train.with_columns(
        pl.Series("rt_predictions", dlc_transfer_learn.make_preds(psm_list_calib))
    )

    if return_obj and not return_predictions:
        return dlc_calibration, dlc_transfer_learn
    if return_obj and return_predictions:
        return dlc_calibration, dlc_transfer_learn, rt_train


def get_predictions_retention_time_mainloop(
    df_psms: pl.DataFrame,
    write_deeplc_pickle: bool,
    read_deeplc_pickle: bool,
    deeplc_model: Optional[DeepLC] = None,
    output_dir: Union[str, Path] = "results",
) -> Tuple[Optional[DeepLC], Optional[DeepLC], pl.DataFrame]:
    """
    Main function for managing DeepLC predictions with caching support.

    This function handles the logic for training new models vs. using cached models,
    and manages pickle file I/O for caching trained models and predictions.

    Args:
        df_psms: Polars DataFrame containing PSM data
        write_deeplc_pickle: Whether to save models and predictions to pickle files
        read_deeplc_pickle: Whether to load models and predictions from pickle files
        deeplc_model: Optional pre-trained DeepLC model to use for predictions

    Returns:
        Tuple containing:
        - dlc_calibration: Calibrated DeepLC model (None if using pre-trained model)
        - dlc_transfer_learn: Transfer learning DeepLC model (None if using pre-trained model)
        - predictions_deeplc: DataFrame with retention time predictions
    """
    # If you need to write a pickle with predictions or if you are not writing or reading a pickle
    if write_deeplc_pickle or (not write_deeplc_pickle and not read_deeplc_pickle):
        if deeplc_model is None:  # When does this happen?
            (
                dlc_calibration,
                dlc_transfer_learn,
                predictions_deeplc,
            ) = get_predictions_retentiontime(df_psms)
        else:
            predictions_deeplc = predict_deeplc_pl(df_psms, deeplc_model)

    # If you need to write a pickle
    if write_deeplc_pickle:
        if deeplc_model is None:
            with open(f"{output_dir}/dlc_calibration.pkl", "wb") as f:
                pickle.dump(dlc_calibration, f)
            with open(f"{output_dir}/dlc_transfer_learn.pkl", "wb") as f:
                pickle.dump(dlc_transfer_learn, f)
        with open(f"{output_dir}/predictions_deeplc.pkl", "wb") as f:
            pickle.dump(predictions_deeplc, f)
    if read_deeplc_pickle:
        try:
            with open(f"{output_dir}/dlc_calibration_first.pkl", "rb") as f:
                dlc_calibration = pickle.load(f)
        except IOError:
            pass
        try:
            with open(f"{output_dir}/dlc_transfer_learn_first.pkl", "rb") as f:
                dlc_transfer_learn = pickle.load(f)
        except IOError:
            pass
        with open(f"{output_dir}/predictions_deeplc.pkl", "rb") as f:
            predictions_deeplc = pickle.load(
                f
            )  # FIXME: this gives a polars typeError, not sure why. Might be a polars version issue? or a pickle issue?

    if deeplc_model:
        return None, None, predictions_deeplc
    else:
        return dlc_calibration, dlc_transfer_learn, predictions_deeplc


def retrain_and_bounds(
    df_psms: pl.DataFrame,
    peptides: List[Tuple],
    result_dir: Union[str, Path] = "",
    coefficient_bounds: float = 1.0,
    correct_to_mzml_rt_constant: float = 60.0,
) -> Tuple[pd.DataFrame, DeepLC, DeepLC, float]:
    """
    Retrain DeepLC and calculate retention time bounds for windowed searches.

    This function combines DeepLC retraining with retention time bound calculation
    for creating time-based mzML partitions in the full search workflow.

    Args:
        df_psms: Polars DataFrame with PSM data for training
        peptides: List of peptide tuples from tryptic digestion
        result_dir: Directory for saving output plots and files
        coefficient_bounds: Multiplier for retention time bounds (default: 1.0)
        correct_to_mzml_rt_constant: Conversion factor for mzML time units (default: 60.0 seconds)

    Returns:
        Tuple containing:
        - peptide_df: Pandas DataFrame with peptides and RT predictions/bounds
        - dlc_calibration: Calibrated DeepLC model
        - dlc_transfer_learn: Transfer learning DeepLC model
        - perc_95: 95th percentile RT error for windowing
    """
    dlc_calibration, dlc_transfer_learn, perc_95 = retrain_deeplc(
        df_psms,
        outfile_calib=result_dir.joinpath("deeplc_calibration.png"),
        outfile_transf_learn=result_dir.joinpath("deeplc_transfer_learn.png"),
    )
    perc_95 = perc_95 * correct_to_mzml_rt_constant * coefficient_bounds
    predictions = predict_deeplc(peptides, dlc_transfer_learn)

    peptide_df = pd.DataFrame(
        peptides, columns=["protein", "start", "end", "id", "peptide"]
    )
    peptide_df["predictions"] = predictions
    peptide_df["predictions"] = peptide_df["predictions"] * correct_to_mzml_rt_constant
    peptide_df.to_csv("peptide_predictions.csv", index=False)
    peptide_df["predictions_lower"] = peptide_df["predictions"] - perc_95 / 2.0
    peptide_df["predictions_upper"] = peptide_df["predictions"] + perc_95 / 2.0

    return peptide_df, dlc_calibration, dlc_transfer_learn, perc_95
