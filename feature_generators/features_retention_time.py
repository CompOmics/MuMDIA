"""Retention time prediction error features using DeepLC predictions."""

from typing import Union

import polars as pl


def add_retention_time_features(
    df_psms: pl.DataFrame,
    predictions_deeplc: pl.DataFrame,
    filter_rel_rt_error: float = 0.2,
    rt_prediction_error_abs: bool = True,
    rt_prediction_error_abs_relative: bool = True,
) -> pl.DataFrame:
    """
    Add retention time prediction error features and filter by RT accuracy.

    This function joins DeepLC retention time predictions with PSMs, calculates
    absolute and relative prediction errors, and filters out PSMs with poor
    retention time predictions to improve overall data quality.

    Filter priority: relative error is checked first; absolute error is only
    used as a fallback when relative error is not computed. This means the
    ``filter_rel_rt_error`` threshold is interpreted differently depending on
    which mode is active: as a fraction of the experiment's total RT range
    when filtering on relative error, or as raw RT units (e.g. seconds/minutes)
    when falling back to absolute error.

    Args:
        df_psms: PSM DataFrame with 'peptide' and 'rt' columns
        predictions_deeplc: DataFrame with 'peptide' and 'rt_predictions' columns
        filter_rel_rt_error: Maximum relative RT error threshold for filtering (default: 0.2)
        rt_prediction_error_abs: Whether to calculate absolute RT error (default: True)
        rt_prediction_error_abs_relative: Whether to calculate relative RT error (default: True)

    Returns:
        PSM DataFrame with RT prediction features added and filtered by RT accuracy.
        Added columns: rt_predictions, rt_prediction_error_abs, rt_prediction_error_abs_relative
    """

    df_psms = df_psms.join(predictions_deeplc, on=["peptide"], how="left")
    # max_rt is the latest observed retention time in the experiment. Dividing
    # by max_rt converts absolute RT error into a fraction of the experiment's
    # total RT range, so the filter_rel_rt_error threshold (e.g. 0.2 = 20%)
    # is relative to how long the gradient run was.
    max_rt = df_psms["rt"].max()

    if rt_prediction_error_abs:
        df_psms = df_psms.with_columns(
            pl.Series(
                "rt_prediction_error_abs",
                abs(df_psms["rt"] - df_psms["rt_predictions"]),
            )
        )
    if rt_prediction_error_abs_relative:
        df_psms = df_psms.with_columns(
            pl.Series(
                "rt_prediction_error_abs_relative",
                abs(df_psms["rt"] - df_psms["rt_predictions"]) / max_rt,
            )
        )

    # Always assumes to filter on relative error first, if available
    if rt_prediction_error_abs_relative:
        df_psms = df_psms.filter(
            df_psms["rt_prediction_error_abs_relative"] < filter_rel_rt_error
        )
    elif rt_prediction_error_abs:
        df_psms = df_psms.filter(
            df_psms["rt_prediction_error_abs"] < filter_rel_rt_error
        )

    return df_psms
