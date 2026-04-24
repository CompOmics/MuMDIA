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

import os
import pickle
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList

from utilities.logger import log_info

try:
    from deeplc import DeepLC as _LegacyDeepLC

    _DEEPLC_V4 = False
except ImportError:
    _LegacyDeepLC = None
    from deeplc import (  # type: ignore[no-redef]
        calibrate as deeplc_calibrate,
        finetune as deeplc_finetune,
        predict as deeplc_predict,
        predict_and_calibrate as deeplc_predict_and_calibrate,
    )
    from deeplc.core import (
        DEFAULT_MODEL as _DEEPLC_V4_DEFAULT_MODEL,
        DEFAULT_MODEL_FALLBACK as _DEEPLC_V4_FALLBACK_MODEL,
        DEFAULT_MULTITASK_MODEL_PACKAGED as _DEEPLC_V4_PACKAGED_MULTITASK_MODEL,
    )

    _DEEPLC_V4 = True

if not _DEEPLC_V4:
    _DEEPLC_V4_DEFAULT_MODEL = None


def _register_v4_multitask_compat() -> None:
    if not _DEEPLC_V4 or "multitask_model" in sys.modules:
        return

    module_path = Path(__file__).resolve().parent.parent / "multitask_model.py"
    spec = importlib.util.spec_from_file_location("multitask_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load multitask compatibility module from {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules["multitask_model"] = module
    spec.loader.exec_module(module)


if _DEEPLC_V4:
    _register_v4_multitask_compat()
    if _DEEPLC_V4_PACKAGED_MULTITASK_MODEL.exists():
        _DEEPLC_V4_DEFAULT_MODEL = _DEEPLC_V4_PACKAGED_MULTITASK_MODEL


_DEEPLC_MODEL_LABEL = (
    "DeepLC v4 multitask"
    if _DEEPLC_V4
    and _DEEPLC_V4_DEFAULT_MODEL is not None
    and Path(_DEEPLC_V4_DEFAULT_MODEL).name == "multitask_model.pt"
    else (
        "DeepLC v4 fallback"
        if _DEEPLC_V4
        and _DEEPLC_V4_DEFAULT_MODEL is not None
        and Path(_DEEPLC_V4_DEFAULT_MODEL) == Path(_DEEPLC_V4_FALLBACK_MODEL)
        else ("DeepLC v4" if _DEEPLC_V4 else "DeepLC legacy")
    )
)

DeepLC = Any
DeepLCModel = Any

_DEEPLC_THREADS = min(64, os.cpu_count() or 1)
_DEEPLC_PREDICT_KWARGS = {
    "batch_size": 2048,
    "num_threads": _DEEPLC_THREADS,
    "show_progress": False,
}


def _clone_predict_kwargs(**overrides: Any) -> Dict[str, Any]:
    kwargs = dict(_DEEPLC_PREDICT_KWARGS)
    kwargs.update(overrides)
    return kwargs


def _make_v4_bundle(
    model: Any,
    calibration: Any,
    reference_psms: PSMList,
    predict_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "api": "deeplc_v4",
        "model": model,
        "calibration": calibration,
        "reference_psms": reference_psms,
        "predict_kwargs": dict(predict_kwargs or _DEEPLC_PREDICT_KWARGS),
    }


def _predict_with_model(
    psm_list: PSMList,
    dlc_model: DeepLCModel,
    calibrate: bool = True,
) -> np.ndarray:
    if _DEEPLC_V4:
        if isinstance(dlc_model, dict) and dlc_model.get("api") == "deeplc_v4":
            model = dlc_model.get("model")
            predict_kwargs = dlc_model.get("predict_kwargs") or _DEEPLC_PREDICT_KWARGS
            if calibrate:
                return np.asarray(
                    deeplc_predict_and_calibrate(
                        psm_list=psm_list,
                        psm_list_reference=dlc_model["reference_psms"],
                        model=model,
                        calibration=dlc_model["calibration"],
                        predict_kwargs=predict_kwargs,
                    )
                )
            return np.asarray(
                deeplc_predict(
                    psm_list=psm_list,
                    model=model,
                    predict_kwargs=predict_kwargs,
                )
            )

        return np.asarray(
            deeplc_predict(
                psm_list=psm_list,
                model=dlc_model,
                predict_kwargs=_DEEPLC_PREDICT_KWARGS,
            )
        )

    return np.asarray(dlc_model.make_preds(psm_list, calibrate=calibrate))


def _fit_baseline_calibration(
    psm_list_calib: PSMList,
) -> Tuple[DeepLCModel, np.ndarray]:
    if _DEEPLC_V4:
        predict_kwargs = _clone_predict_kwargs()
        calibration = deeplc_calibrate(
            psm_list_reference=psm_list_calib,
            model=_DEEPLC_V4_DEFAULT_MODEL,
            predict_kwargs=predict_kwargs,
        )
        bundle = _make_v4_bundle(
            model=_DEEPLC_V4_DEFAULT_MODEL,
            calibration=calibration,
            reference_psms=psm_list_calib,
            predict_kwargs=predict_kwargs,
        )
        preds = _predict_with_model(psm_list_calib, bundle, calibrate=True)
        return bundle, preds

    dlc_calibration = _LegacyDeepLC(
        batch_num=1024000, deeplc_retrain=False, pygam_calibration=False, n_jobs=64
    )
    dlc_calibration.calibrate_preds(psm_list_calib)
    preds = _predict_with_model(psm_list_calib, dlc_calibration, calibrate=True)
    return dlc_calibration, preds


def _fit_transfer_model(
    psm_list_calib_filtered: PSMList,
    n_epochs: int,
) -> Tuple[DeepLCModel, np.ndarray]:
    if _DEEPLC_V4:
        predict_kwargs = _clone_predict_kwargs()
        train_kwargs = {
            "epochs": n_epochs,
            "num_threads": _DEEPLC_THREADS,
            "batch_size": 1024,
            "show_progress": False,
        }
        finetuned_model = deeplc_finetune(
            psm_list_reference=psm_list_calib_filtered,
            model=_DEEPLC_V4_DEFAULT_MODEL,
            train_kwargs=train_kwargs,
        )
        calibration = deeplc_calibrate(
            psm_list_reference=psm_list_calib_filtered,
            model=finetuned_model,
            predict_kwargs=predict_kwargs,
        )
        bundle = _make_v4_bundle(
            model=finetuned_model,
            calibration=calibration,
            reference_psms=psm_list_calib_filtered,
            predict_kwargs=predict_kwargs,
        )
        preds = _predict_with_model(psm_list_calib_filtered, bundle, calibrate=True)
        return bundle, preds

    dlc_transfer_learn = _LegacyDeepLC(
        batch_num=1024000, deeplc_retrain=True, n_epochs=n_epochs, n_jobs=64
    )
    dlc_transfer_learn.calibrate_preds(psm_list_calib_filtered)
    preds = _predict_with_model(
        psm_list_calib_filtered, dlc_transfer_learn, calibrate=True
    )
    return dlc_transfer_learn, preds


def plot_performance(
    psm_list: PSMList,
    preds: np.ndarray,
    outfile: Union[str, Path] = "plot.png",
    model_label: Optional[str] = None,
) -> None:
    """
    Create a scatter plot comparing observed vs predicted retention times.

    Args:
        psm_list: List of PSM objects with retention time information
        preds: Array of predicted retention times
        outfile: Output file path for the plot
    """
    observed = np.asarray([v.retention_time for v in psm_list], dtype=float)
    predicted = np.asarray(preds, dtype=float)

    mae = float(np.mean(np.abs(observed - predicted)))
    pearson = float(np.corrcoef(observed, predicted)[0, 1])

    plt.figure(figsize=(7, 6))
    plt.scatter(observed, predicted, s=3, alpha=0.05)

    min_rt = float(min(observed.min(), predicted.min()))
    max_rt = float(max(observed.max(), predicted.max()))
    plt.plot(
        [min_rt, max_rt], [min_rt, max_rt], linestyle="--", linewidth=1, color="gray"
    )

    plt.xlabel("Observed retention time (min)")
    plt.ylabel("Predicted retention time (min)")
    annotation_lines = [f"MAE = {mae:.3f} min", f"Pearson r = {pearson:.3f}"]
    if model_label:
        annotation_lines.append(f"Model = {model_label}")

    plt.text(
        0.02,
        0.98,
        "\n".join(annotation_lines),
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def _select_apex_rt_training_rows(df_psms: pl.DataFrame) -> pl.DataFrame:
    """
    Select one training RT per peptide using the highest-fragment-intensity PSM.

    This approximates the chromatographic apex by taking the PSM whose matched
    fragment evidence is strongest for that peptide, then using its RT for DeepLC
    calibration and fine-tuning.
    """
    return (
        df_psms.sort("fragment_intensity", descending=True)
        .unique(subset=["peptide"], keep="first", maintain_order=True)
        .select(["peptide", "rt"])
    )


def _filter_deeplc_training_psms(
    df_psms: pl.DataFrame,
    q_value_filter: float,
    min_peptidoform_occurrences: int,
) -> pl.DataFrame:
    filtered = df_psms.filter(df_psms["spectrum_q"] < q_value_filter)

    if min_peptidoform_occurrences <= 1:
        return filtered

    occurrence_counts = filtered.group_by(["peptide", "charge"]).len()
    qualifying = occurrence_counts.filter(
        pl.col("len") >= min_peptidoform_occurrences
    ).select(["peptide", "charge"])

    return filtered.join(qualifying, on=["peptide", "charge"], how="inner")


def predict_deeplc_pl_old(
    psm_df_pl: pl.DataFrame, dlc_model: DeepLCModel
) -> pl.DataFrame:
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
        pl.Series("rt_predictions", _predict_with_model(psm_list, dlc_model))
    )

    return psm_df_pl


def predict_deeplc_pl(psm_df_pl: pl.DataFrame, dlc_model: DeepLCModel) -> pl.DataFrame:
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
    predictions = _predict_with_model(psm_list, dlc_model)
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


def predict_deeplc(psms_list: List[Tuple], dlc_model: DeepLCModel) -> np.ndarray:
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

    return _predict_with_model(psm_list_calib, dlc_model)


def retrain_deeplc(
    df_psms: pl.DataFrame,
    plot_perf: bool = True,
    outfile_calib: Union[str, Path] = "deeplc_calibration.png",
    outfile_transf_learn: Union[str, Path] = "deeplc_transfer_learn.png",
    percentile_exclude: float = 50,
    q_value_filter: float = 0.001,
    n_epochs: int = 75,
    min_peptidoform_occurrences: int = 1,
    calibration_only: bool = False,
) -> Tuple[DeepLCModel, DeepLCModel, float]:
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
        n_epochs: Number of DeepLC fine-tuning epochs for the transfer-learning stage
        min_peptidoform_occurrences: Minimum observations per peptide/charge required before a candidate can be used for DeepLC fine-tuning
        calibration_only: If True, skip transfer learning and reuse the calibrated model

    Returns:
        Tuple containing:
        - dlc_calibration: Initial calibrated DeepLC model
        - dlc_transfer_learn: Transfer learning DeepLC model
        - rt_split_window: RT split window width derived from the configured percentile (doubled for symmetric bounds)
    """
    df_psms_filtered = _filter_deeplc_training_psms(
        df_psms, q_value_filter, min_peptidoform_occurrences
    )
    rt_train = _select_apex_rt_training_rows(df_psms_filtered)

    psm_list_calib = [
        PSM(peptidoform=seq, retention_time=tr, spectrum_id=idx)
        for idx, (seq, tr) in enumerate(zip(rt_train["peptide"], rt_train["rt"]))
    ]
    psm_list_calib = PSMList(psm_list=psm_list_calib)

    dlc_calibration, preds = _fit_baseline_calibration(psm_list_calib)

    # Percentile-based filtering: remove the worst-predicted PSMs (top 5% by default)
    # before transfer learning. These outliers are likely misidentifications or
    # chromatographic anomalies that would degrade the transfer-learned model.
    errors = abs(np.array(preds) - np.array([v.retention_time for v in psm_list_calib]))
    selection = errors < np.percentile(errors, percentile_exclude)
    psm_list_calib_filtered_percentile = [
        psm for psm, incl in zip(psm_list_calib, selection) if incl
    ]
    psm_list_calib_filtered_percentile = PSMList(
        psm_list=psm_list_calib_filtered_percentile
    )

    if calibration_only:
        dlc_transfer_learn = dlc_calibration
        preds_transflearn = _predict_with_model(
            psm_list_calib_filtered_percentile,
            dlc_transfer_learn,
            calibrate=True,
        )
    else:
        dlc_transfer_learn, preds_transflearn = _fit_transfer_model(
            psm_list_calib_filtered_percentile,
            n_epochs=n_epochs,
        )

    if plot_perf:
        plot_performance(
            psm_list_calib,
            preds,
            outfile=outfile_calib,
            model_label=_DEEPLC_MODEL_LABEL,
        )
        plot_performance(
            psm_list_calib_filtered_percentile,
            preds_transflearn,
            outfile=outfile_transf_learn,
            model_label=_DEEPLC_MODEL_LABEL,
        )

    # Return the selected RT-error percentile as a full symmetric window width.
    # The value is doubled here because it will later be split into +/- halves
    # around the predicted RT.
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
    n_epochs: int = 50,
    min_peptidoform_occurrences: int = 1,
) -> Union[
    Tuple[DeepLCModel, DeepLCModel],
    Tuple[DeepLCModel, DeepLCModel, pl.DataFrame],
]:
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
        n_epochs: Number of DeepLC fine-tuning epochs for the transfer-learning stage
        min_peptidoform_occurrences: Minimum observations per peptide/charge required before a candidate can be used for DeepLC fine-tuning

    Returns:
        If return_obj and return_predictions: (dlc_calibration, dlc_transfer_learn, predictions_df)
        If return_obj only: (dlc_calibration, dlc_transfer_learn)
    """
    df_psms_filtered = _filter_deeplc_training_psms(
        df_psms, q_value_filter, min_peptidoform_occurrences
    )

    rt_train = _select_apex_rt_training_rows(df_psms_filtered)

    psm_list_calib = [
        PSM(peptidoform=seq, retention_time=tr, spectrum_id=idx)
        for idx, (seq, tr) in enumerate(zip(rt_train["peptide"], rt_train["rt"]))
    ]
    psm_list_calib = PSMList(psm_list=psm_list_calib)

    dlc_calibration, preds = _fit_baseline_calibration(psm_list_calib)

    # Use the 50th percentile (median) here -- more aggressive filtering than
    # retrain_deeplc's 95th percentile. This keeps only the better-predicted half
    # of PSMs for transfer learning, producing a tighter model at the cost of
    # discarding more training data. Suitable for the prediction pathway where
    # model accuracy matters more than the RT error bound estimate.
    errors = abs(np.array(preds) - np.array([v.retention_time for v in psm_list_calib]))
    selection = errors < np.percentile(errors, percentile_exclude)
    psm_list_calib_filtered_percentile = [
        psm for psm, incl in zip(psm_list_calib, selection) if incl
    ]
    psm_list_calib_filtered_percentile = PSMList(
        psm_list=psm_list_calib_filtered_percentile
    )

    dlc_transfer_learn, preds_transflearn = _fit_transfer_model(
        psm_list_calib_filtered_percentile,
        n_epochs=n_epochs,
    )

    if plot_perf:
        plot_performance(
            psm_list_calib,
            preds,
            outfile=outfile_calib,
            model_label=_DEEPLC_MODEL_LABEL,
        )
        plot_performance(
            psm_list_calib_filtered_percentile,
            preds_transflearn,
            outfile=outfile_transf_learn,
            model_label=_DEEPLC_MODEL_LABEL,
        )

    # TODO here I reuse code, but this should stand on its own
    rt_train = _select_apex_rt_training_rows(df_psms)

    # rt_train get the psm_id and add instread, then merge with prev
    psm_list_calib = [
        PSM(peptidoform=seq, retention_time=tr, spectrum_id=idx)
        for idx, (seq, tr) in enumerate(zip(rt_train["peptide"], rt_train["rt"]))
    ]
    psm_list_calib = PSMList(psm_list=psm_list_calib)

    rt_train = rt_train.with_columns(
        pl.Series(
            "rt_predictions", _predict_with_model(psm_list_calib, dlc_transfer_learn)
        )
    )

    if return_obj and not return_predictions:
        return dlc_calibration, dlc_transfer_learn
    if return_obj and return_predictions:
        return dlc_calibration, dlc_transfer_learn, rt_train


def get_predictions_retention_time_mainloop(
    df_psms: pl.DataFrame,
    write_deeplc_pickle: bool,
    read_deeplc_pickle: bool,
    deeplc_model: Optional[DeepLCModel] = None,
    output_dir: Union[str, Path] = "results",
    n_epochs: int = 50,
    min_peptidoform_occurrences: int = 1,
) -> Tuple[Optional[DeepLCModel], Optional[DeepLCModel], pl.DataFrame]:
    """
    Main function for managing DeepLC predictions with caching support.

    This function handles the logic for training new models vs. using cached models,
    and manages pickle file I/O for caching trained models and predictions.

    Args:
        df_psms: Polars DataFrame containing PSM data
        write_deeplc_pickle: Whether to save models and predictions to pickle files
        read_deeplc_pickle: Whether to load models and predictions from pickle files
        deeplc_model: Optional pre-trained DeepLC model to use for predictions
        n_epochs: Number of DeepLC fine-tuning epochs for the transfer-learning stage
        min_peptidoform_occurrences: Minimum observations per peptide/charge required before a candidate can be used for DeepLC fine-tuning

    Returns:
        Tuple containing:
        - dlc_calibration: Calibrated DeepLC model (None if using pre-trained model)
        - dlc_transfer_learn: Transfer learning DeepLC model (None if using pre-trained model)
        - predictions_deeplc: DataFrame with retention time predictions
    """
    # Three code paths:
    # 1. write_deeplc_pickle=True: train/predict now, then save to disk below.
    # 2. Both False: train/predict now, no caching -- normal non-cached run.
    # 3. read_deeplc_pickle=True: skip this block entirely, load from disk below.
    if write_deeplc_pickle or (not write_deeplc_pickle and not read_deeplc_pickle):
        if deeplc_model is None:  # When does this happen?
            # No pre-trained model supplied -- train from scratch via full pipeline
            (
                dlc_calibration,
                dlc_transfer_learn,
                predictions_deeplc,
            ) = get_predictions_retentiontime(
                df_psms,
                n_epochs=n_epochs,
                min_peptidoform_occurrences=min_peptidoform_occurrences,
            )
        else:
            # Pre-trained model supplied -- only run inference, skip training
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
        # NOTE: The pickle filenames use a "_first" suffix (e.g. dlc_calibration_first.pkl)
        # which does not match the filenames written above (dlc_calibration.pkl).
        # This means reading will only succeed if the files were saved under the
        # "_first" names by an earlier pipeline stage. See FIXME below.
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

    if deeplc_model is not None:
        return None, None, predictions_deeplc
    else:
        return dlc_calibration, dlc_transfer_learn, predictions_deeplc


def retrain_and_bounds(
    df_psms: pl.DataFrame,
    peptides: List[Tuple],
    result_dir: Union[str, Path] = "",
    coefficient_bounds: float = 1.0,
    correct_to_mzml_rt_constant: float = 60.0,
    percentile_exclude: float = 95.0,
    fixed_rt_window_seconds: Optional[float] = None,
    n_epochs: int = 75,
    min_peptidoform_occurrences: int = 1,
    calibration_only: bool = False,
) -> Tuple[pd.DataFrame, DeepLCModel, DeepLCModel, float]:
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
        percentile_exclude: RT-error percentile used for the split window estimate
        fixed_rt_window_seconds: Optional fixed split width in seconds overriding the percentile-derived width
        n_epochs: Number of DeepLC fine-tuning epochs for the transfer-learning stage
        min_peptidoform_occurrences: Minimum observations per peptide/charge required before a candidate can be used for DeepLC fine-tuning
        calibration_only: If True, skip transfer learning and reuse the calibrated model

    Returns:
        Tuple containing:
        - peptide_df: Pandas DataFrame with peptides and RT predictions/bounds
        - dlc_calibration: Calibrated DeepLC model
        - dlc_transfer_learn: Transfer learning DeepLC model
        - rt_split_window: RT split window width for partitioning
    """
    dlc_calibration, dlc_transfer_learn, rt_split_window = retrain_deeplc(
        df_psms,
        outfile_calib=result_dir.joinpath("deeplc_calibration.png"),
        outfile_transf_learn=result_dir.joinpath("deeplc_transfer_learn.png"),
        percentile_exclude=percentile_exclude,
        n_epochs=n_epochs,
        min_peptidoform_occurrences=min_peptidoform_occurrences,
        calibration_only=calibration_only,
    )
    # Convert the RT split window from DeepLC's native unit (minutes) to mzML time units
    # (seconds by default, factor=60). coefficient_bounds allows further scaling.
    rt_split_window = rt_split_window * correct_to_mzml_rt_constant * coefficient_bounds
    if fixed_rt_window_seconds is not None:
        rt_split_window = float(fixed_rt_window_seconds)
        log_info(f"RT window (fixed): {rt_split_window}")
    else:
        log_info(f"RT window (p{percentile_exclude:g}): {rt_split_window}")
    predictions = predict_deeplc(peptides, dlc_transfer_learn)

    peptide_df = pd.DataFrame(
        peptides, columns=["protein", "start", "end", "id", "peptide"]
    )
    # Also convert per-peptide predictions from minutes to mzML seconds
    peptide_df["predictions"] = predictions
    peptide_df["predictions"] = peptide_df["predictions"] * correct_to_mzml_rt_constant
    # peptide_df.to_csv("peptide_predictions.csv", index=False)
    # Build a symmetric RT window: rt_split_window is the full window width (already doubled
    # in retrain_deeplc), so divide by 2 to get the half-width for +/- bounds.
    peptide_df["predictions_lower"] = peptide_df["predictions"] - rt_split_window / 2.0
    peptide_df["predictions_upper"] = peptide_df["predictions"] + rt_split_window / 2.0

    return peptide_df, dlc_calibration, dlc_transfer_learn, rt_split_window
