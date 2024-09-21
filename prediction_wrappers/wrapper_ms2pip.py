import matplotlib.pyplot as plt

from ms2pip.core import predict_batch
from matplotlib import pyplot as plt
from deeplc import DeepLC

from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList

import numpy as np
import polars as pl

from tqdm import tqdm
from operator import itemgetter
import pickle


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
