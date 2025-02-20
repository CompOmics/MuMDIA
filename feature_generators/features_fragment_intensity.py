import pickle
from typing import List
import polars as pl
from numba import njit
import numpy as np
from tqdm import tqdm
from typing import Tuple
from utilities.logger import log_info


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


def corrcoef_ignore_both_missing(data):
    """
    Compute pairwise Pearson correlation coefficients between rows of the input
    matrix, ignoring observation positions where both corresponding values are zero.

    Parameters:
    data (np.ndarray): A 2D array where rows represent variables and columns represent observations.

    Returns:
    np.ndarray: A symmetric matrix of correlation coefficients.
    """
    n_rows = data.shape[0]
    corr_matrix = np.empty((n_rows, n_rows))

    for i in range(n_rows):
        for j in range(i, n_rows):
            # Create a mask that excludes positions where both values are zero.
            mask = ~((data[i, :] == 0) & (data[j, :] == 0))
            if np.sum(mask) > 1:
                # Compute the Pearson correlation coefficient using the valid entries.
                r = np.corrcoef(data[i, mask], data[j, mask])[0, 1]
            else:
                # Insufficient valid data points for correlation computation.
                r = np.nan
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    return corr_matrix


def corrcoef_ignore_both_missing_counts(data):
    """
    Compute pairwise Pearson correlation coefficients between rows of the input
    matrix, ignoring observation positions where both corresponding values are zero.
    Additionally, return a matrix that indicates the number of valid (i.e., used)
    data points for each correlation calculation.

    Parameters:
    data (np.ndarray): A 2D NumPy array where rows represent variables and columns represent observations.

    Returns:
    tuple: A tuple containing:
        - corr_matrix (np.ndarray): A symmetric matrix of correlation coefficients.
        - count_matrix (np.ndarray): A symmetric matrix with the count of valid observations for each pair.
    """
    n_rows = data.shape[0]
    corr_matrix = np.empty((n_rows, n_rows))
    count_matrix = np.empty((n_rows, n_rows), dtype=int)

    # Iterate over all pairs of rows (variables)
    for i in range(n_rows):
        for j in range(i, n_rows):
            # Create a mask that excludes positions where both values are zero
            mask = ~((data[i, :] == 0) & (data[j, :] == 0))
            # Count the number of observations used in the calculation
            count = np.sum(mask)
            count_matrix[i, j] = count
            count_matrix[j, i] = count

            if count > 1:
                # Compute Pearson correlation using only the valid data points
                r = np.corrcoef(data[i, mask], data[j, mask])[0, 1]
            else:
                # Not enough data to compute correlation
                r = np.nan
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    return corr_matrix, count_matrix


def corrcoef_ignore_zeros_counts(data):
    """
    Compute pairwise Pearson correlation coefficients between rows of the input
    matrix, ignoring observation positions where either corresponding value is zero.
    Additionally, return a matrix indicating the number of valid observations used
    for each correlation calculation.

    Parameters:
    data (np.ndarray): A 2D NumPy array where rows represent variables and columns
                       represent observations.

    Returns:
    tuple: A tuple containing:
        - corr_matrix (np.ndarray): A symmetric matrix of correlation coefficients.
        - count_matrix (np.ndarray): A symmetric matrix with the count of valid observations
                                     for each pair.
    """
    n_rows = data.shape[0]
    corr_matrix = np.empty((n_rows, n_rows))
    count_matrix = np.empty((n_rows, n_rows), dtype=int)

    # Iterate over all pairs of rows (variables)
    for i in range(n_rows):
        for j in range(i, n_rows):
            # Create a mask that selects positions where both values are nonzero
            mask = (data[i, :] != 0) & (data[j, :] != 0)
            count = np.sum(mask)
            count_matrix[i, j] = count
            count_matrix[j, i] = count

            if count > 1:
                # Compute Pearson correlation using only the valid (nonzero) observations.
                r = np.corrcoef(data[i, mask], data[j, mask])[0, 1]
            else:
                # Not enough data points to compute correlation reliably.
                r = np.nan
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    return corr_matrix, count_matrix


def corrcoef_ignore_zeros(data):
    """
    Compute the pairwise Pearson correlation coefficients between rows of the input
    matrix, ignoring any entries that are zero in either row.

    Parameters:
    data (np.ndarray): A 2D NumPy array where rows represent variables and columns represent observations.

    Returns:
    np.ndarray: A symmetric matrix of correlation coefficients.
    """
    n_rows = data.shape[0]
    corr_matrix = np.empty((n_rows, n_rows))
    # Iterate over pairs of rows
    for i in range(n_rows):
        for j in range(i, n_rows):
            # Create a mask that selects elements where both rows have nonzero values
            mask = (data[i, :] != 0) & (data[j, :] != 0)
            if np.sum(mask) > 1:
                # Compute Pearson correlation on the valid entries
                r = np.corrcoef(data[i, mask], data[j, mask])[0, 1]
            else:
                # Insufficient data points to compute correlation
                r = np.nan
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r
    return corr_matrix


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Parameters
    ----------
    vec1 : np.ndarray
        The first vector (e.g., experimental spectrum intensities).
    vec2 : np.ndarray
        The second vector (e.g., predicted spectrum intensities).

    Returns
    -------
    float
        The cosine similarity between vec1 and vec2. Returns 0.0 if either vector has zero norm.
    """
    # Calculate the dot product between the vectors.
    dot_product = np.dot(vec1, vec2)

    # Compute the L2 (Euclidean) norms of the vectors.
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Avoid division by zero by checking if either norm is zero.
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    # Return the cosine similarity.
    return dot_product / (norm_vec1 * norm_vec2)


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
    # TODO, correlations without the 0.0
    # TODO, skip those with only 1 spectrum
    # TODO, skip those with only 1 fragment
    # Pivot and convert to NumPy array

    intensity_matrix_df = df_fragment_sub_peptidoform.pivot(
        index="psm_id", columns="fragment_name", values="fragment_intensity"
    ).fill_null(0.0)

    max_intens_ms2pip = max(ms2pip_predictions.values())
    ms2pip_predictions = dict(
        [(k, v / max_intens_ms2pip) for k, v in ms2pip_predictions.items()]
    )

    most_abundant_frag_psm = df_fragment_sub_peptidoform.filter(
        df_fragment_sub_peptidoform["rt"]
        == df_fragment_sub_peptidoform["rt_max_peptide_sub"][0]
    )

    pred_frag_intens_individual = np.array(
        [
            ms2pip_predictions.get(fid, 0.0)
            for fid in most_abundant_frag_psm["fragment_name"]
        ]
    )

    most_intens_cor = np.corrcoef(
        pred_frag_intens_individual, most_abundant_frag_psm["fragment_intensity"]
    )[0][1]

    most_intens_cos = cosine_similarity(
        pred_frag_intens_individual, most_abundant_frag_psm["fragment_intensity"]
    )

    # first column is PMS ID, ignore that one, messes up calculation as it is numeric
    intensity_matrix = intensity_matrix_df[:, 1:].to_numpy()
    # Get fragment names, first column is PMS ID, ignore that one, messes up calculation as it is numeric
    fragment_names = intensity_matrix_df.columns[1:]

    # Prepare predicted fragment intensities
    pred_frag_intens = np.array(
        [ms2pip_predictions.get(fid, 0.0) for fid in fragment_names]
    )

    # Collect predictions for keys not listed in fragment_names.
    non_matched_predictions = np.array(
        [v for k, v in ms2pip_predictions.items() if k not in fragment_names]
    )

    sum_pred_frag_intens = np.array(
        sum([ms2pip_predictions.get(fid, 0.0) for fid in fragment_names])
    )

    # Ensure data types are consistent
    intensity_matrix = intensity_matrix.astype(np.float32)
    pred_frag_intens = pred_frag_intens.astype(np.float32)
    non_matched_predictions = non_matched_predictions.astype(np.float32)

    pred_frag_intens = np.concatenate((pred_frag_intens, non_matched_predictions))

    # Calculate the number of zeros to append
    pad_width = len(pred_frag_intens) - len(intensity_matrix[0])

    # Extend array 'a' by padding with zeros on the right side
    intensity_matrix = np.pad(
        intensity_matrix, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
    )

    intensity_matrix_normalized = intensity_matrix / intensity_matrix.max(
        axis=1, keepdims=True
    )

    # Compute correlations between observed and predicted intensities
    correlation_result = compute_correlations(
        intensity_matrix_normalized, pred_frag_intens
    )
    correlation_result_counts = (
        intensity_matrix_df.select(
            pl.fold(
                acc=pl.lit(0),
                exprs=[
                    (pl.col(c) != 0).cast(pl.Int64) for c in intensity_matrix_df.columns
                ],
                function=lambda acc, x: acc + x,
            ).alias("non_zero_count")
        )
        .to_numpy()
        .ravel()
    )

    mse_avg_pred_intens = (
        abs(intensity_matrix_normalized - pred_frag_intens).sum(axis=1)
    ).sum() / intensity_matrix_normalized.shape[0]
    mse_avg_pred_intens_total = (
        (abs(intensity_matrix_normalized - pred_frag_intens).sum(axis=1)).sum()
        + sum(non_matched_predictions)
    ) / intensity_matrix_normalized.shape[0]

    # Compute correlation matrix for PSM IDs
    if intensity_matrix_normalized.shape[0] > 1:
        correlation_matrix_psm_ids = np.corrcoef(intensity_matrix_normalized)

        """
        (
            correlation_matrix_psm_ids_ignore_zeros,
            correlation_matrix_psm_ids_ignore_zeros_counts,
        ) = corrcoef_ignore_zeros_counts(intensity_matrix)

        (
            correlation_matrix_psm_ids_missing,
            correlation_matrix_psm_ids_missing_zeros_counts,
        ) = corrcoef_ignore_both_missing_counts(intensity_matrix)
        """

        correlation_matrix_psm_ids_ignore_zeros = np.array([])
        correlation_matrix_psm_ids_ignore_zeros_counts = np.array([])
        correlation_matrix_psm_ids_missing = np.array([])
        correlation_matrix_psm_ids_missing_zeros_counts = np.array([])

        # Remove diagonal elements and flatten
        correlation_matrix_psm_ids = correlation_matrix_psm_ids[
            ~np.eye(correlation_matrix_psm_ids.shape[0], dtype=bool)
        ]
        # Square and sort
        correlation_matrix_psm_ids = np.sort(correlation_matrix_psm_ids**2)
    else:
        correlation_matrix_psm_ids = np.array([])
        correlation_matrix_psm_ids_ignore_zeros = np.array([])
        correlation_matrix_psm_ids_ignore_zeros_counts = np.array([])
        correlation_matrix_psm_ids_missing = np.array([])
        correlation_matrix_psm_ids_missing_zeros_counts = np.array([])

    # Compute correlation matrix for fragment IDs
    if intensity_matrix_normalized.shape[1] > 1:
        correlation_matrix_frag_ids = np.corrcoef(intensity_matrix_normalized.T)

        """
        (
            correlation_matrix_frag_ids_ignore_zeros,
            correlation_matrix_frag_ids_ignore_zeros_counts,
        ) = corrcoef_ignore_zeros_counts(intensity_matrix)

        (
            correlation_matrix_frag_ids_missing,
            correlation_matrix_frag_ids_missing_zeros_counts,
        ) = corrcoef_ignore_both_missing_counts(intensity_matrix)
        """
        correlation_matrix_frag_ids_ignore_zeros = np.array([])
        correlation_matrix_frag_ids_ignore_zeros_counts = np.array([])
        correlation_matrix_frag_ids_missing = np.array([])
        correlation_matrix_frag_ids_missing_zeros_counts = np.array([])

        # Remove diagonal elements and flatten
        correlation_matrix_frag_ids = correlation_matrix_frag_ids[
            ~np.eye(correlation_matrix_frag_ids.shape[0], dtype=bool)
        ]
        # Square and sort
        correlation_matrix_frag_ids = np.sort(correlation_matrix_frag_ids**2)
    else:
        correlation_matrix_frag_ids = np.array([])
        correlation_matrix_frag_ids_ignore_zeros = np.array([])
        correlation_matrix_frag_ids_ignore_zeros_counts = np.array([])
        correlation_matrix_frag_ids_missing = np.array([])
        correlation_matrix_frag_ids_missing_zeros_counts = np.array([])

    # Correlation result: 1D array of correlation values between predicted MS2PIP intensities and observed fragment intensities
    # Correlation matrix PSM IDs: 1D array of correlation values between PSMs
    # Correlation matrix fragment IDs: 1D array of correlation values between fragments

    # correlation_result, correlation_result_counts
    # correlation_matrix_psm_ids, correlation_matrix_psm_ids_counts
    # correlation_matrix_frag_ids, correlation_matrix_frag_ids_counts

    #        correlation_matrix_psm_ids_ignore_zeros,
    #        correlation_matrix_psm_ids_ignore_zeros_counts,
    #        correlation_matrix_frag_ids_ignore_zeros,
    #        correlation_matrix_frag_ids_ignore_zeros_counts,

    return (
        correlation_result,
        correlation_result_counts,
        sum_pred_frag_intens,
        correlation_matrix_psm_ids,
        correlation_matrix_frag_ids,
        correlation_matrix_psm_ids_ignore_zeros,
        correlation_matrix_psm_ids_ignore_zeros_counts,
        correlation_matrix_psm_ids_missing,
        correlation_matrix_psm_ids_missing_zeros_counts,
        correlation_matrix_frag_ids_ignore_zeros,
        correlation_matrix_frag_ids_ignore_zeros_counts,
        correlation_matrix_frag_ids_missing,
        correlation_matrix_frag_ids_missing_zeros_counts,
        most_intens_cor,
        most_intens_cos,
        mse_avg_pred_intens,
        mse_avg_pred_intens_total,
    )


def get_features_fragment_intensity(
    ms2pip_predictions: dict,
    df_fragment: pl.DataFrame,
    df_fragment_max_peptide: pl.DataFrame,
    filter_max_apex_rt: float = 3.0,
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

    log_info("Calculation of all correlation values...")

    if not read_correlation_pickles:
        for (peptidoform, charge), df_fragment_sub_peptidoform in tqdm(
            df_fragment.group_by(["peptide", "charge"])
        ):
            preds = ms2pip_predictions.get(f"{peptidoform}/{charge}")
            if not preds:
                log_info(f"No intensity prediction found for {peptidoform}/{charge}...")
                continue
            if df_fragment_sub_peptidoform.shape[0] == 0:
                log_info(f"No fragments found for {peptidoform}/{charge}...")
                continue

            (
                correlations,
                correlations_count,
                sum_pred_frag_intens,
                correlation_matrix_psm_ids,
                correlation_matrix_frag_ids,
                correlation_matrix_psm_ids_ignore_zeros,
                correlation_matrix_psm_ids_ignore_zeros_counts,
                correlation_matrix_psm_ids_missing,
                correlation_matrix_psm_ids_missing_zeros_counts,
                correlation_matrix_frag_ids_ignore_zeros,
                correlation_matrix_frag_ids_ignore_zeros_counts,
                correlation_matrix_frag_ids_missing,
                correlation_matrix_frag_ids_missing_zeros_counts,
                most_intens_cor,
                most_intens_cos,
                mse_avg_pred_intens,
                mse_avg_pred_intens_total,
            ) = match_fragments(df_fragment_sub_peptidoform, preds)

            fragment_dict[f"{peptidoform}/{charge}"] = df_fragment_sub_peptidoform
            correlations_fragment_dict[f"{peptidoform}/{charge}"] = [
                correlations,
                correlations_count,
                sum_pred_frag_intens,
                correlation_matrix_psm_ids,
                correlation_matrix_frag_ids,
                correlation_matrix_psm_ids_ignore_zeros,
                correlation_matrix_psm_ids_ignore_zeros_counts,
                correlation_matrix_psm_ids_missing,
                correlation_matrix_psm_ids_missing_zeros_counts,
                correlation_matrix_frag_ids_ignore_zeros,
                correlation_matrix_frag_ids_ignore_zeros_counts,
                correlation_matrix_frag_ids_missing,
                correlation_matrix_frag_ids_missing_zeros_counts,
                most_intens_cor,
                most_intens_cos,
                mse_avg_pred_intens,
                mse_avg_pred_intens_total,
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
