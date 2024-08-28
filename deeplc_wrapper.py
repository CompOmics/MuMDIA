from matplotlib import pyplot as plt
from deeplc import DeepLC

from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList

import numpy as np
import polars as pl
from operator import itemgetter


def plot_performance(psm_list, preds, outfile="plot.png"):
    plt.scatter([v.retention_time for v in psm_list], preds, s=3, alpha=0.05)
    plt.xlabel("Observed retention time (min)")
    plt.ylabel("Predicted retention time (min)")
    plt.savefig(outfile)
    plt.close()


def predict_deeplc_pl(psm_df_pl, dlc_model):
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


def predict_deeplc(psms_list, dlc_model):
    psm_list_calib = [
        PSM(peptidoform=seq, spectrum_id=idx)
        for seq, idx in zip(
            [psl[-1] for psl in psms_list], [psl[-2] for psl in psms_list]
        )
    ]
    psm_list_calib = PSMList(psm_list=psm_list_calib)

    return dlc_model.make_preds(psm_list_calib)


def retrain_deeplc(
    df_psms,
    plot_perf=True,
    outfile_calib="deeplc_calibration.png",
    outfile_transf_learn="deeplc_transfer_learn.png",
    percentile_exclude=95,
    q_value_filter=0.01,
):
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
        batch_num=1024000,
        deeplc_retrain=False,
        pygam_calibration=False,
        n_jobs=64
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
        batch_num=1024000,
        deeplc_retrain=True,
        n_epochs=50,
        n_jobs=64
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
    df_psms,
    plot_perf=True,
    outfile_calib="deeplc_calibration.png",
    outfile_transf_learn="deeplc_transfer_learn.png",
    percentile_exclude=50,
    return_obj=True,
    return_predictions=True,
    q_value_filter=0.01,
):
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
        batch_num=1024000,
        deeplc_retrain=False,
        pygam_calibration=False,
        n_jobs=64
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
        batch_num=1024000,
        deeplc_retrain=True,
        n_epochs=50,
        n_jobs=64
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
