import polars as pl


def add_retention_time_features(
    df_psms,
    predictions_deeplc,
    filter_rel_rt_error=0.2,
    rt_prediction_error_abs=True,
    rt_prediction_error_abs_relative=True,
):
    """
    Add retention time prediction error features to the PSM dataframe.
    Args:
        df_psms: PSM dataframe with the following columns:
            - peptide: Peptide sequence
            - rt: Retention time
        filter_rel_rt_error: Maximum relative retention time error to keep a PSM
    """

    df_psms = df_psms.join(predictions_deeplc, on="peptide", how="left")
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
