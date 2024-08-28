import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

"""
import spectrum_utils.spectrum as sus
import spectrum_utils.plot as sup

from scripts.ms2pip_utils import (
    get_usi_spectrum,
    get_theoretical_spectrum,
    get_predicted_spectrum,
    get_intensity_array,
)
"""

from ms2pip.core import predict_batch
from matplotlib import pyplot as plt
from deeplc import DeepLC

from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList

import numpy as np
import polars as pl

from tqdm import tqdm


def plot_performance(psm_list, preds, outfile="plot.png"):
    plt.scatter([v.retention_time for v in psm_list], preds, s=3, alpha=0.05)
    plt.xlabel("Observed retention time (min)")
    plt.ylabel("Predicted retention time (min)")
    plt.savefig(outfile)
    plt.close()


def batch_process_predict_batch(
    input_list, batch_size, process_function, n_processes=64
):
    batches = [
        input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)
    ]

    results = []
    for batch in tqdm(batches):
        results.extend(process_function(batch, processes=n_processes, model="HCD2021"))

    return results


def get_predictions_fragment_intensity(df_psms):
    df_ms2pip = df_psms.unique(subset=["peptide", "charge"])

    psm_list = [
        PSM(
            peptidoform=seq + "/" + str(charge),
            retention_time=tr,
            spectrum_id=idx,
            precursor_charge=charge,
        )
        for idx, (seq, tr, charge) in enumerate(
            zip(df_ms2pip["peptide"], df_ms2pip["rt"], df_ms2pip["charge"])
        )
    ]
    psm_list = PSMList(psm_list=psm_list)

    ms2pip_predictions = batch_process_predict_batch(psm_list, 500000, predict_batch)

    ms2pip_predictions_dict = {}

    for pred in ms2pip_predictions:
        k = str(pred.psm.peptidoform)
        try:
            ms2pip_predictions_dict[k] = dict(
                [
                    ("b%s/1" % (idx + 1), 2**v)
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


"""
    dlc_calibration = DeepLC(
        batch_num=1024000,
        deeplc_retrain=False,
        pygam_calibration=False,
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
        n_epochs=10,
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

    rt_train = rt_train.with_columns(
        pl.Series("predictions", dlc_transfer_learn.make_preds(psm_list_calib))
    )

    if return_obj and not return_predictions:
        return dlc_calibration, dlc_transfer_learn
    if return_obj and return_predictions:
        return dlc_calibration, dlc_transfer_learn, rt_train
"""
