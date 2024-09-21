import pickle
from typing import List
import polars as pl
from numba import njit
import numpy as np
from tqdm import tqdm
from typing import Tuple
from MuMDIA.utilities.logger import log_info


@njit
def compute_correlations(intensity_matrix, pred_frag_intens):
    num_psms = intensity_matrix.shape[0]
    correlations = np.zeros(num_psms)
    for i in range(num_psms):
        x = intensity_matrix[i, :]
        y = pred_frag_intens
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x > 0 and std_y > 0:
            covariance = np.mean((x - mean_x) * (y - mean_y))
            correlations[i] = covariance / (std_x * std_y)
        else:
            correlations[i] = 0.0
    return correlations


def match_fragments(
    df_fragment_sub_peptidoform: pl.DataFrame, ms2pip_predictions: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match fragments and calculate their correlation.

    Parameters:
    - df_fragment_sub_peptidoform (pl.DataFrame): A DataFrame containing fragment data.
    - ms2pip_predictions (dict): A dictionary of MS2PIP predictions.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray]: Correlation result and correlation matrices.
    """
    # Pivot and convert to NumPy array
    intensity_matrix_df = df_fragment_sub_peptidoform.pivot(
        index="psm_id", columns="fragment_name", values="fragment_intensity"
    ).fill_null(0.0)

    intensity_matrix = intensity_matrix_df.to_numpy()
    fragment_names = intensity_matrix_df.columns

    # Prepare predicted fragment intensities
    pred_frag_intens = np.array(
        [ms2pip_predictions.get(fid, 0.0) for fid in fragment_names]
    )

    # Ensure data types are consistent
    intensity_matrix = intensity_matrix.astype(np.float32)
    pred_frag_intens = pred_frag_intens.astype(np.float32)

    # Compute correlations between observed and predicted intensities
    correlation_result = compute_correlations(intensity_matrix, pred_frag_intens)

    # Compute correlation matrix for PSM IDs
    if intensity_matrix.shape[0] > 1:
        correlation_matrix_psm_ids = np.corrcoef(intensity_matrix)
        # Remove diagonal elements and flatten
        correlation_matrix_psm_ids = correlation_matrix_psm_ids[
            ~np.eye(correlation_matrix_psm_ids.shape[0], dtype=bool)
        ]
        # Square and sort
        correlation_matrix_psm_ids = np.sort(correlation_matrix_psm_ids**2)
    else:
        correlation_matrix_psm_ids = np.array([])

    # Compute correlation matrix for fragment IDs
    if intensity_matrix.shape[1] > 1:
        correlation_matrix_frag_ids = np.corrcoef(intensity_matrix.T)
        # Remove diagonal elements and flatten
        correlation_matrix_frag_ids = correlation_matrix_frag_ids[
            ~np.eye(correlation_matrix_frag_ids.shape[0], dtype=bool)
        ]
        # Square and sort
        correlation_matrix_frag_ids = np.sort(correlation_matrix_frag_ids**2)
    else:
        correlation_matrix_frag_ids = np.array([])

    return (
        correlation_result,
        correlation_matrix_psm_ids,
        correlation_matrix_frag_ids,
    )


def get_features_fragment_intensity(
    ms2pip_predictions: dict,
    df_fragment: pl.DataFrame,
    df_fragment_max_peptide: pl.DataFrame,
    filter_max_apex_rt: float = 5.0,
    read_correlation_pickles: bool = False,
    write_correlation_pickles: bool = False,
):
    fragment_dict = {}
    correlations_fragment_dict = {}

    peptide_to_rt_max = dict(
        zip(
            df_fragment_max_peptide["peptide"].to_list(),
            df_fragment_max_peptide["rt"].to_list(),
        )
    )

    df_peptide_rt = pl.DataFrame(
        {
            "peptide": list(peptide_to_rt_max.keys()),
            "rt_max_peptide_sub": list(peptide_to_rt_max.values()),
        }
    )
    # Add rt_max_peptide_sub to df_fragment
    df_fragment = df_fragment.join(df_peptide_rt, on="peptide", how="left")
    df_fragment = df_fragment.filter(
        (pl.col("rt_max_peptide_sub").is_not_null())
        & (abs(pl.col("rt") - pl.col("rt_max_peptide_sub")) < filter_max_apex_rt)
    )

    # Ensure rt_max_peptide_sub is not null and apply the RT filter
    df_fragment = df_fragment.filter(
        (pl.col("rt_max_peptide_sub").is_not_null())
        & (abs(pl.col("rt") - pl.col("rt_max_peptide_sub")) < filter_max_apex_rt)
    )

    log_info("Starting to process peptidoforms...")

    if not read_correlation_pickles:
        for (peptidoform, charge), df_fragment_sub_peptidoform in tqdm(
            df_fragment.group_by(["peptide", "charge"])
        ):
            preds = ms2pip_predictions.get(f"{peptidoform}/{charge}")
            if not preds:
                continue

            if df_fragment_sub_peptidoform.shape[0] == 0:
                continue

            # Proceed with the rest of your code
            (
                correlations,
                correlation_matrix_psm_ids,
                correlation_matrix_frag_ids,
            ) = match_fragments(df_fragment_sub_peptidoform, preds)

            fragment_dict[f"{peptidoform}/{charge}"] = df_fragment_sub_peptidoform
            correlations_fragment_dict[f"{peptidoform}/{charge}"] = [
                correlations,
                correlation_matrix_psm_ids,
                correlation_matrix_frag_ids,
            ]

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

    return fragment_dict, correlations_fragment_dict
