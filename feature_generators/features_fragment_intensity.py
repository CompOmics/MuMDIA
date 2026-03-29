"""
Fragment Intensity Feature Generation for MuMDIA

This module generates features based on fragment ion intensity correlations
between experimental and predicted spectra. These features are crucial for
assessing the quality of peptide-spectrum matches in data-independent acquisition.

Key Features:
- Pearson correlation calculation between observed and predicted fragment intensities
- Numba-optimized correlation computation for speed
- Integration with MS2PIP predictions and experimental fragment data
- Support for missing value handling in correlation calculations
- RustyMS integration for theoretical fragment generation

The correlation features help distinguish correct from incorrect identifications
by measuring how well the predicted fragment pattern matches the observed spectrum.
"""

import pickle
import re
from typing import List, Tuple

import numpy as np
import polars as pl
from numba import njit
from rustyms import (
    FragmentationModel,
    MassMode,
    RawSpectrum,
    CompoundPeptidoformIon,
    MatchingParameters,
)
from tqdm import tqdm

from data_structures import CorrelationResults, PickleConfig
from utilities.logger import log_info
from utilities.plotting import plot_XIC


@njit
def compute_correlations(intensity_matrix, pred_frag_intens):
    """
    Compute Pearson correlations between experimental and predicted intensities.

    This Numba-optimized function calculates correlation coefficients between
    each row of the intensity matrix (representing fragment intensities for
    different PSMs) and the predicted fragment intensities.

    Args:
        intensity_matrix: 2D array where each row contains fragment intensities for one PSM
        pred_frag_intens: 1D array of predicted fragment intensities

    Returns:
        Array of correlation coefficients, one per PSM
    """
    num_psms = intensity_matrix.shape[0]
    correlations = np.zeros(num_psms)

    for i in range(num_psms):
        x = intensity_matrix[i, :]  # Experimental intensities for this PSM
        y = pred_frag_intens  # Predicted intensities

        # Calculate means and standard deviations
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        std_x = np.std(x)
        std_y = np.std(y)

        # Only compute correlation if both arrays have non-zero variance
        if std_x > 0 and std_y > 0:
            covariance = np.mean((x - mean_x) * (y - mean_y))
            correlations[i] = covariance / (std_x * std_y)
        else:
            correlations[i] = 0.0  # No correlation possible with zero variance

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
    df_fragment_sub_peptidoform: pl.DataFrame, ms2pip_predictions: dict, ms2_dict: dict
) -> CorrelationResults:
    """
       Match observed fragment intensities to MS2PIP predictions using RustyMS annotation,
       then compute correlation-based features for scoring.

       Algorithm:
       1. For each unique PSM, annotate experimental spectrum with RustyMS (CID/HCD model,
          13 ppm tolerance, charge-1 b/y ions only).
       2. Build intensity matrix (PSMs × fragments), fill missing with 0.
       3. Normalize MS2PIP predictions to [0, 1] by dividing by max.
       4. Compute apex spectrum Pearson correlation and cosine similarity.
       5. Pad intensity matrix with zeros for non-matched predicted fragments.
       6. Normalize each PSM row by its max intensity.
       7. Compute per-PSM Pearson correlations vs predictions (Numba-accelerated).
       8. Compute PSM-to-PSM correlation matrix (squared, sorted) and fragment-to-fragment
          correlation matrix (sorted, not squared).
       9. Compute MAE between normalized intensities and predictions.

       Parameters:
       - df_fragment_sub_peptidoform (pl.DataFrame): Fragment data for one peptidoform.
         Required columns: psm_id, fragment_type, fragment_ordinals, fragment_charge,
         fragment_intensity, rt, scannr, fragment_name, rt_max_peptide_sub, precursor, charge.
       - ms2pip_predictions (dict): Fragment intensity predictions, e.g. {"b1/1": 0.5, "y3/1": 0.8}.
       - ms2_dict (dict): MS2 spectrum data, keyed by scan number: {scannr: {mz: array, intensity: array}}.

       Returns:
       - CorrelationResults: Dataclass with correlation arrays, matrices, and scalar metrics.

       Example of df_fragment_sub_peptidoform:

       ┌────────┬───────────────┬───────────────────┬─────────────────┬───┬───────────┬─────────────────────────────────┬───────────────┬────────────────────┐
       │ psm_id ┆ fragment_type ┆ fragment_ordinals ┆ fragment_charge ┆ … ┆ rt        ┆ scannr                          ┆ fragment_name ┆ rt_max_peptide_sub │
       │ ---    ┆ ---           ┆ ---               ┆ ---             ┆   ┆ ---       ┆ ---                             ┆ ---           ┆ ---                │
       │ i64    ┆ str           ┆ i32               ┆ i32             ┆   ┆ f32       ┆ str                             ┆ str           ┆ f64                │
       ╞════════╪═══════════════╪═══════════════════╪═════════════════╪═══╪═══════════╪═════════════════════════════════╪═══════════════╪════════════════════╡
       │ 659384 ┆ b             ┆ 10                ┆ 1               ┆ … ┆ 74.839798 ┆ controllerType=0 controllerNum… ┆ b10/1         ┆ 74.839798          │
       │ 659384 ┆ y             ┆ 13                ┆ 1               ┆ … ┆ 74.839798 ┆ controllerType=0 controllerNum… ┆ y13/1         ┆ 74.839798          │
       │ 659384 ┆ y             ┆ 11                ┆ 1               ┆ … ┆ 74.839798 ┆ controllerType=0 controllerNum… ┆ y11/1         ┆ 74.839798          │
       └────────┴───────────────┴───────────────────┴─────────────────┴───┴───────────┴─────────────────────────────────┴───────────────┴────────────────────┘

       ms2pip_predictions :

       {'b1/1': 0.0010326379173923124, 'b2/1': 0.006559936772673948, 'b3/1': 0.02018668526395722,
        'b4/1': 0.002342675702582015, 'b5/1': 0.006451339642362278, 'b6/1': 0.006209347132265164,
        'b7/1': 0.008126135332143791, 'b8/1': 0.006083811423429146, 'b9/1': 0.003027573342006256,
        'b10/1': 0.002845367012291415, 'b11/1': 0.0013209101948498476, 'b12/1': 0.0019497607452880947,
        'b13/1': 0.001245985002453295, 'b14/1': 0.0010717618591405257, 'b15/1': 0.0011213033307682174,
        'b16/1': 0.0011712898458443087, 'b17/1': 0.0010170142456872556, 'b18/1': 0.001000000146798956,
        'b19/1': 0.0010188020730757745, 'b20/1': 0.0010195337192217913, 'b21/1': 0.001016047289410171,
        'b22/1': 0.001000000146798956, 'b23/1': 0.0010061141816408883, 'y1/1': 0.004439031433375235,
        'y2/1': 0.0017560643548064148, 'y3/1': 0.002171987658281832, 'y4/1': 0.004375419808616321,
        'y5/1': 0.0057759590065042925, 'y6/1': 0.008868378887053153, 'y7/1': 0.014163818692869872,
        'y8/1': 0.011264669967818582, 'y9/1': 0.0025314100771837056, 'y10/1': 0.0030349131219689785,
        'y11/1': 0.003787855271323114, 'y12/1': 0.0016851449345072287, 'y13/1': 0.0014587592886677345,
        'y14/1': 0.001416223247958962, 'y15/1': 0.00130902513212185, 'y16/1': 0.001352687346781147,
        'y17/1': 0.001000000146798956, 'y18/1': 0.001000000146798956, 'y19/1': 0.0010095618558146676,
        'y20/1': 0.001000000146798956, 'y21/1': 0.001000000146798956, 'y22/1': 0.001000000146798956,
       'y23/1': 0.001000392879734152}

       Singular value of ms2_dict["mz"]:

       [390.25       390.51184082 390.75085449 390.90777588 391.24212646
    391.28445435 391.57687378 391.72180176 391.91046143 391.97335815
    392.22293091 392.24298096 392.47253418 394.25863647 396.23754883
    398.25378418 398.71051025 399.21282959 400.19152832 400.76434326
    401.21426392 403.24420166 403.25759888 404.40884399 406.24008179
    407.53829956 408.24691772 408.58117676 408.91616821 410.20162964
    410.25344849 411.16799927 411.22952271 411.72738647 411.87667847
    412.21066284 412.54425049 412.87768555 412.90170288 413.21505737
    413.23727417 413.55010986 414.26986694 415.18908691 416.21817017
    416.26379395 416.57556152 416.90911865 417.22177124 417.24380493
    418.70559692 419.20678711 419.31564331 419.47583008 420.76263428
    421.99017334 423.73931885 423.76361084 424.24124146 424.26806641
    424.81500244 425.71899414 426.55691528 427.20770264 428.20458984
    428.22930908 428.26220703 428.70422363 428.76654053 428.9546814
    429.08847046 429.21847534 429.50930786 429.72006226 429.77352905
    430.20709229 430.23226929 430.27636719 430.58743286 430.8843689
    431.21749878 431.24517822 431.55233765 432.2399292  432.53671265
    432.69567871 432.74530029 433.20684814 433.73657227 433.98843384
    434.22674561 437.2293396  437.25421143 437.44000244 437.98925781
    438.2388916  438.25531006 438.27059937 438.46032715 438.49029541
    438.58621216 438.74090576 438.77166748 438.8939209  439.21292114
    439.23062134 440.71240234 441.77398682 442.27764893 442.74542236
    443.20080566 443.45507812 443.77160645 444.23727417 444.47711182
    444.89755249 445.12005615 445.23196411 445.56689453 445.7578125
    445.89877319 446.12045288 446.25061035 446.72628784 446.75134277
    447.11849976 447.22943115 447.25802612 447.75949097 447.91125488
    448.21273804 448.24447632 448.26071167 448.60858154 448.73388672
    448.76104736 449.23370361 449.27978516 449.73730469 449.99536133
    450.28430176 450.90130615 451.2354126  452.24981689 452.56195068
    453.22839355 453.26531982 454.24697876 454.74679565 456.237854
    456.73892212 457.24081421 457.57757568 457.60522461 457.93960571
    458.2616272  458.76168823 459.23989868 459.2583313  459.74102783
    459.99395752 460.74700928 460.99584961 461.24926758 461.88278198
    462.14550781 462.91143799 464.2460022  464.27816772 464.49642944
    464.74731445 464.99707031 465.2472229  465.28622437 465.77398682
    466.27566528 466.77593994 467.27908325 467.73641968 468.26239014
    468.40808105 468.76608276 468.98751831 469.23846436 469.48928833
    469.74066162 469.76416016 470.60848999 470.94311523 471.27627563
    471.29534912 471.31011963 471.5743103  471.6121521  471.79397583
    471.90753174 472.24560547 472.94058228 473.26126099 473.56964111
    473.76260376 474.26486206 474.74606323 474.9152832  475.28955078
    475.58831787 475.75231934 475.76934814 475.92211914 476.2572937
    476.47277832 476.72891235 476.75006104 477.23010254 477.94888306
    478.26251221 478.28295898 478.61706543 478.79327393 478.95123291
    479.22183228 479.28448486 479.55606079 479.59158325 479.62072754
    479.92657471 480.25991821 480.29302979 480.48699951 480.7557373
    480.92285156 481.89813232 482.2300415  482.57147217 482.73052979
    482.90698242 483.24206543 483.74609375 483.80258179 484.25558472
    484.30392456 484.50259399 484.75341797 484.80584717 484.95263672
    485.0039978  485.2868042  485.62179565 485.94076538 487.23358154
    487.32406616 488.32778931 488.72241211 488.74191284 488.77575684
    489.25540161 489.27856445 489.58673096 490.22692871 490.25543213
    492.74371338 493.28793335 493.78866577 494.28985596 494.55801392
    495.49664307 495.57907104 495.74853516 495.91934204 496.95526123
    497.2902832  498.25192261 498.29510498 498.75308228 499.27142334
    499.76391602 500.7428894  501.24456787 501.78692627 502.26370239
    502.28582764 502.76504517 503.10742188 503.59143066 503.92553711
    504.25982666 504.29373169 504.59350586 504.61697388 504.77203369
    504.9281311  505.73080444 505.76141357 506.23873901 506.26229858
    506.58309937 506.76400757 506.91717529 506.99240112 507.25128174
    507.58786011 508.24996948 508.27038574 508.29034424 508.75094604
    510.59622192 510.74194336 511.24411011 511.27026367 511.57711792
    511.77005005 511.91131592 512.23925781 513.25476074 515.30462646
    515.80456543 515.92828369 516.79956055 517.28723145 518.29833984
    518.64349365 519.13952637 519.26123047 520.13848877 521.79827881
    521.96112061 522.25268555 522.29638672 522.6295166  523.27996826
    523.30413818 524.26934814 524.7935791  525.27124023 525.29449463
    525.76922607 525.79498291 526.27044678 526.32354736 526.7756958
    529.80871582 530.30877686 530.76348877 530.93597412 531.27032471
    531.51544189 531.5480957  531.6027832  531.77020264 531.93688965
    532.2723999  532.29364014 533.27270508 533.53033447 533.58587646
    533.60821533 533.77972412 533.9418335  534.03070068 534.25091553
    534.53436279 535.79638672 536.27215576 536.3125     536.7713623
    536.7767334  536.80279541 537.27832031 537.30358887 537.61022949
    537.80621338 537.94494629 538.28283691 538.30755615 538.80529785
    539.25543213 539.30358887 539.7567749  540.58673096 540.92205811
    541.25482178 541.50878906 542.00439453 543.29071045 546.28741455
    546.30895996 546.80969238 546.94976807 547.27966309 547.3215332
    547.61724854 547.93969727 548.27459717 549.2769165  550.26446533
    550.58746338 551.3203125  551.809021   552.01446533 552.26556396
    552.31256104 552.51593018 552.76629639 552.78826904 552.81158447
    552.95825195 553.0166626  553.26727295 553.315979   553.7689209
    554.29083252 554.7802124  554.80291748 555.28039551 555.61022949
    555.78320312 556.28045654 556.51708984 556.83837891 557.33758545
    558.01849365 558.27191162 558.31896973 558.36114502 558.5222168
    913.42755127 913.93615723 915.43811035 915.97247314 916.47363281
    927.98687744 928.38720703 928.88885498 929.38775635 929.88916016
    930.38708496 971.11413574 972.46832275 972.97241211 973.37127686
    994.95611572]
    """

    """
    Match fragments theoretical and experimental intensities.
    """

    if df_fragment_sub_peptidoform.is_empty():
        log_info("No fragments to match, returning empty results.")

    # Compile regex patterns for extracting ion and charge from annotation strings
    ion_pattern = r"ion='([^']*)'"
    charge_pattern = r"charge=(\d+),"

    fragment_records = []

    # # Plot XICs
    # plot_XIC(df_fragment_sub_peptidoform)

    # Get unique PSMs by sorting by fragment intensity and keeping the first occurrence per PSM
    unique_psm_id = df_fragment_sub_peptidoform.sort(
        "fragment_intensity", descending=True
    ).unique(
        subset=["psm_id"], keep="first"
    )  # TODO: is this the best approach to select the apex?

    unique_psm_id_dicts = unique_psm_id.to_dicts()

    # Iterate over each unique PSM to annotate and match fragments
    successful_psm_ids = []
    failed_psm_ids = []

    for row in unique_psm_id_dicts:
        psm_id = int(row["psm_id"])
        rt = float(row["rt"])
        scannr = row["scannr"]
        rt_max_peptide_sub = float(row["rt_max_peptide_sub"])
        precursor_charge = int(
            row["charge"]
        )  # This was fragment_charge before, but it is the precursor charge
        precursor = row[
            "precursor"
        ]  # TODO: check if its okay to do on precursor level. If we don't we have a problem with RT matching

        try:
            # Construct a RawSpectrum object for this PSM using the scan number and MS2 data
            spectrum = RawSpectrum(
                title=scannr,
                num_scans=1,
                rt=float(rt),
                precursor_charge=precursor_charge,
                precursor_mass=1.0,
                mz_array=ms2_dict[scannr]["mz"],
                intensity_array=ms2_dict[scannr]["intensity"],
            )

            # CompoundPeptidoformIon parses the "peptide/charge" string into a
            # structured peptidoform object that RustyMS can use for theoretical
            # fragment generation (handles modifications, charge state, etc.)
            linear_peptide = CompoundPeptidoformIon(precursor)

            # MatchingParameters controls how RustyMS matches experimental peaks
            # to theoretical fragments (here: 13 ppm mass tolerance window)
            matching_parameters = MatchingParameters()
            matching_parameters.tolerance_ppm = (
                13.0  # TODO: make this a parameter used by the config
            )

            # Annotate the spectrum with theoretical fragments using RustyMS
            annotated_spectrum = spectrum.annotate(
                peptidoform=linear_peptide,
                parameters=matching_parameters,
                model=FragmentationModel.CidHcd,
                mode=MassMode.Monoisotopic,
            )


            # Filter annotated peaks to keep only singly-charged b and y ions.
            # RustyMS annotations are accessed via repr() strings, so regex is
            # used to extract the ion type (e.g. "b3", "y7") from the annotation
            # representation. Only charge-1 fragments are kept to match MS2PIP's
            # prediction scope (which only predicts singly-charged b/y ions).
            matched_fragments = [
                annotated_peak
                for annotated_peak in annotated_spectrum.spectrum
                if annotated_peak.annotation
                and annotated_peak.annotation[0].charge == 1  # make configurable
                and (
                    re.search(ion_pattern, repr(annotated_peak.annotation[0]))
                    .group(1)
                    .startswith("b")
                    or re.search(ion_pattern, repr(annotated_peak.annotation[0]))
                    .group(1)
                    .startswith("y")
                )
            ]

            # # For each matched fragment, extract ion type, ordinal, charge, and intensity

            if len(matched_fragments) == 0:
                log_info(
                    "WARNING: No matched fragments found for PSM ID: {}, RT: {}".format(
                        psm_id, rt
                    )
                )
                failed_psm_ids.append(psm_id)
                continue

            for mf in matched_fragments:
                # Extract ion type (e.g. "b3", "y7") and charge from the RustyMS
                # annotation repr string using the precompiled regex patterns.
                # ion_label[0] gives the ion series letter, ion_label[1:] the ordinal.
                ion_label = re.search(ion_pattern, repr(mf.annotation[0])).group(1)
                ion_charge = re.search(charge_pattern, repr(mf.annotation[0])).group(1)

                fragment_records.append(
                    {
                        "psm_id": psm_id,
                        "fragment_type": ion_label[0],
                        "fragment_ordinals": ion_label[1:],
                        "fragment_charge": ion_charge,
                        "fragment_intensity": mf.intensity,
                        "fragment_mz": mf.experimental_mz,
                        "rt": rt,
                        "scannr": scannr,
                        "fragment_name": f"{ion_label}/{ion_charge}",
                        "rt_max_peptide_sub": rt_max_peptide_sub,
                    }
                )

            successful_psm_ids.append(psm_id)

        except Exception as e:
            log_info(
                "ERROR: Failed to process PSM ID: {}, RT: {}, Error: {}".format(
                    psm_id, rt, str(e)
                )
            )
            failed_psm_ids.append(psm_id)
            continue

    # If any fragment records were found, create a new DataFrame and ensure uniqueness per PSM/fragment
    if len(fragment_records) != 0:
        new_df_fragment_sub_peptidoform = (
            pl.DataFrame(fragment_records)
            .sort("fragment_intensity", descending=True)
            .unique(subset=["psm_id", "fragment_name"], keep="first")
        )

        # Replace the original DataFrame
        df_fragment_sub_peptidoform = new_df_fragment_sub_peptidoform
    else:
        log_info("ERROR: No fragment records were created! All PSMs failed processing.")
        # Keep the original DataFrame rather than creating an empty one
        log_info("Keeping original df_fragment_sub_peptidoform")

    # Pivot from long format (one row per PSM-fragment pair) to wide format
    # (one row per PSM, one column per fragment ion). This creates the intensity
    # matrix needed for correlation calculations. Missing fragments get filled
    # with 0.0, meaning a fragment not observed in a given PSM has zero intensity.
    intensity_matrix_df = df_fragment_sub_peptidoform.pivot(
        index="psm_id", columns="fragment_name", values="fragment_intensity"
    ).fill_null(0.0)

    """
    intensity_matrix_df

    ┌──────────┬─────────────┬─────────────┬──────────────┬────────────┬─────────────┬─────────────┬────────────┐
    │ psm_id   ┆ b4/1        ┆ b7/1        ┆ y6/1         ┆ y4/1       ┆ y2/1        ┆ b6/1        ┆ b10/1      │
    │ ---      ┆ ---         ┆ ---         ┆ ---          ┆ ---        ┆ ---         ┆ ---         ┆ ---        │
    │ f64      ┆ f32         ┆ f32         ┆ f32          ┆ f32        ┆ f32         ┆ f32         ┆ f32        │
    ╞══════════╪═════════════╪═════════════╪══════════════╪════════════╪═════════════╪═════════════╪════════════╡
    │ 813993.0 ┆ 4202.810059 ┆ 6978.210449 ┆ 17644.021484 ┆ 9831.50293 ┆ 9209.626953 ┆ 0.0         ┆ 0.0        │
    │ 866572.0 ┆ 0.0         ┆ 0.0         ┆ 0.0          ┆ 0.0        ┆ 2569.063721 ┆ 7737.804199 ┆ 410.700531 │
    └──────────┴─────────────┴─────────────┴──────────────┴────────────┴─────────────┴─────────────┴────────────┘
    """


    # Max-normalize MS2PIP predictions to [0, 1] range. This is necessary because
    # MS2PIP outputs raw predicted intensities on an arbitrary scale, while the
    # experimental intensities will also be max-normalized per PSM later (line ~722).
    # Both must be on the same scale for meaningful correlation and error metrics.
    max_intens_ms2pip = max(ms2pip_predictions.values())
    ms2pip_predictions = dict(
        [(k, v / max_intens_ms2pip) for k, v in ms2pip_predictions.items()]
    )

    """
    Get pearson and cosing similarity of spectrum with highest intensity
    """

    # Select the PSM(s) with RT equal to the maximum RT for this precursor (i.e., apex)
    target_rt = df_fragment_sub_peptidoform["rt_max_peptide_sub"][0]

    most_abundant_frag_psm = df_fragment_sub_peptidoform.filter(
        df_fragment_sub_peptidoform["rt"] == target_rt
    )

    # Build predicted intensity vector for the fragments present in the most abundant PSM
    pred_frag_intens_individual = np.array(
        [
            ms2pip_predictions.get(
                fid, 0.0
            )  # TODO: how do we handle fragments that MS2PIP cannot predict? Do we still use them? e.g. 'p' ions
            for fid in most_abundant_frag_psm["fragment_name"]
        ]
    )


    """
    Get pearson and cosine similarity of spectrum with highest intensity
    """
    # Compute Pearson correlation between predicted and observed intensities for the apex spectrum
    most_intens_cor = np.corrcoef(
        pred_frag_intens_individual, most_abundant_frag_psm["fragment_intensity"]
    )[0][1]


    # Compute cosine similarity between predicted and observed intensities for the apex spectrum
    most_intens_cos = cosine_similarity(
        pred_frag_intens_individual, most_abundant_frag_psm["fragment_intensity"]
    )


    """
    Get the intensity matrix of observations
    """
    # first column is PSM ID, ignore that one, messes up calculation as it is numeric
    intensity_matrix = intensity_matrix_df[:, 1:].to_numpy()
    # Get fragment names, first column is PMS ID, ignore that one, messes up calculation as it is numeric
    fragment_names = intensity_matrix_df.columns[1:]

    # Prepare predicted fragment intensities for all fragments in the matrix columns
    pred_frag_intens = np.array(
        [ms2pip_predictions.get(fid, 0.0) for fid in fragment_names]
    )


    # Collect predictions for keys not listed in fragment_names (i.e., fragments predicted but not observed)
    non_matched_predictions = np.array(
        [v for k, v in ms2pip_predictions.items() if k not in fragment_names]
    )

    # Sum of predicted intensities for matched fragments (for feature engineering)
    # TODO: is it a good idea to sum over PSMs? Or should we do it per PSM?
    sum_pred_frag_intens = np.array(
        sum([ms2pip_predictions.get(fid, 0.0) for fid in fragment_names])
    )


    # Ensure data types are consistent for downstream calculations
    intensity_matrix = intensity_matrix.astype(np.float32)
    pred_frag_intens = pred_frag_intens.astype(np.float32)
    non_matched_predictions = non_matched_predictions.astype(np.float32)

    # Concatenate predicted intensities for matched and non-matched fragments
    pred_frag_intens = np.concatenate((pred_frag_intens, non_matched_predictions))

    # Zero-pad the intensity matrix columns to match the full prediction vector length.
    # Fragments that MS2PIP predicted but were NOT observed in any PSM get zero observed
    # intensity. This ensures the prediction and observation vectors have equal length
    # for correlation/error calculations, and penalizes PSMs that are missing expected
    # fragments (a predicted-but-absent fragment contributes 0 observed vs nonzero predicted).
    pad_width = len(pred_frag_intens) - len(intensity_matrix[0])
    intensity_matrix = np.pad(
        intensity_matrix, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
    )

    # Normalize each PSM row independently by its own max intensity, scaling to [0, 1].
    # This removes absolute intensity differences between PSMs (e.g., due to varying
    # injection amounts or elution profiles) so that correlations reflect relative
    # fragment patterns rather than overall signal strength.
    intensity_matrix_normalized = intensity_matrix / intensity_matrix.max(
        axis=1, keepdims=True
    )

    # Compute correlations between observed and predicted intensities for each PSM
    correlation_result = compute_correlations(
        intensity_matrix_normalized, pred_frag_intens
    )

    # Count the number of nonzero entries per PSM (for feature engineering)
    # TODO: is there a relevance to this? Because the number of columns depends on the max number of fragments for the PSM with the most fragments
    correlation_result_counts = (
        intensity_matrix_df.select(
            pl.fold(  # fold is used to apply a function across multiple columns
                acc=pl.lit(0),  # Initialize accumulator to zero
                exprs=[
                    (pl.col(c) != 0).cast(pl.Int64)
                    for c in intensity_matrix_df.columns  # Convert non-zero entries to 1
                ],
                function=lambda acc, x: acc + x,  # Sum the non-zero counts
            ).alias(
                "non_zero_count"
            )  # Rename the result column
        )
        .to_numpy()  # Convert to NumPy array for consistency
        .ravel()  # Flatten the array to 1D
    )


    # Compute mean squared error between normalized observed and predicted intensities (per PSM, then averaged)
    mse_avg_pred_intens = (
        abs(intensity_matrix_normalized - pred_frag_intens).sum(axis=1)
    ).sum() / intensity_matrix_normalized.shape[0]

    # Compute total MSE including non-matched predictions
    mse_avg_pred_intens_total = (
        (abs(intensity_matrix_normalized - pred_frag_intens).sum(axis=1)).sum()
        + sum(non_matched_predictions)
    ) / intensity_matrix_normalized.shape[0]


    # Compute correlation matrix for PSM IDs (rows of intensity matrix)
    if intensity_matrix_normalized.shape[0] > 1:  # Ensure there are multiple PSMs
        correlation_matrix_psm_ids = np.corrcoef(
            intensity_matrix_normalized
        )  # Calculate correlation matrix for PSM IDs

        # Remove diagonal elements (self-correlation) and flatten to 1D
        correlation_matrix_psm_ids = correlation_matrix_psm_ids[
            ~np.eye(correlation_matrix_psm_ids.shape[0], dtype=bool)
        ]
        # Square correlations to get R² (coefficient of determination), which
        # represents the proportion of variance explained. Sorting produces a
        # ranked distribution of PSM-pair agreement for downstream feature extraction.
        # NOTE: this converts r to R², unlike the fragment correlation matrix below.
        correlation_matrix_psm_ids = np.sort(correlation_matrix_psm_ids**2)
    else:

        # If only one PSM, set all correlation matrices to empty
        correlation_matrix_psm_ids = np.array([])

    # Compute correlation matrix for fragment IDs (columns of intensity matrix)
    if intensity_matrix_normalized.shape[1] > 1:
        correlation_matrix_frag_ids = np.corrcoef(intensity_matrix_normalized.T)


        # Remove diagonal elements (self-correlation) and flatten to 1D
        correlation_matrix_frag_ids = correlation_matrix_frag_ids[
            ~np.eye(correlation_matrix_frag_ids.shape[0], dtype=bool)
        ]
        # Sort the fragment-pair correlations. NOTE: unlike the PSM correlation
        # matrix above, these are NOT squared (raw Pearson r, not R²). This
        # preserves the sign, allowing detection of anti-correlated fragment pairs.
        correlation_matrix_frag_ids = np.sort(correlation_matrix_frag_ids)
    else:

        # If only one fragment, set all correlation matrices to empty
        correlation_matrix_frag_ids = np.array([])



    return CorrelationResults(
        correlations=correlation_result,  # Pearson correlation between predicted and observed intensities
        correlations_count=correlation_result_counts,  # Count of non-zero fragments entries per PSM
        sum_pred_frag_intens=sum_pred_frag_intens,  # Sum of predicted fragment intensities for matched fragments
        correlation_matrix_psm_ids=correlation_matrix_psm_ids,  # Correlation matrix for PSMs, i.e. the correlation between fragments of different PSMs
        correlation_matrix_frag_ids=correlation_matrix_frag_ids,  # Correlation matrix for fragments, i.e. the correlation between fragments of every PSM
        most_intens_cor=most_intens_cor,  # Pearson correlation of the most intense PSM
        most_intens_cos=most_intens_cos,  # Cosine similarity of the most intense PSM
        mse_avg_pred_intens=mse_avg_pred_intens,  # Average MSE of predicted fragment intensities
        mse_avg_pred_intens_total=mse_avg_pred_intens_total,  # Total MSE of predicted fragment intensities including non-matched predictions
    )


def get_features_fragment_intensity(
    ms2pip_predictions: dict,
    df_fragment: pl.DataFrame,
    df_fragment_max_peptide: pl.DataFrame,
    filter_max_apex_rt: float = 3.0,
    read_correlation_pickles: bool = False,
    write_correlation_pickles: bool = False,
    ms2_dict: dict = {},
    output_dir: str = "results/",
):
    """
    Compute fragment intensity correlation features for all peptidoforms.

    For each unique peptide/charge combination, filters fragments to within
    ±filter_max_apex_rt seconds of the apex RT, then calls match_fragments()
    to compute Pearson correlations between observed and MS2PIP-predicted
    fragment intensities using RustyMS spectrum annotation.

    Args:
        ms2pip_predictions: Dict mapping "peptide/charge" to {fragment_key: intensity}.
        df_fragment: Fragment-level DataFrame with psm_id, peptide, charge, rt, etc.
        df_fragment_max_peptide: One row per peptide/charge — the apex (highest intensity) PSM.
        filter_max_apex_rt: Max RT deviation (seconds) from apex to keep fragments (default: 3.0).
        read_correlation_pickles: Load cached results from pickle files.
        write_correlation_pickles: Save computed results to pickle files.
        ms2_dict: Dict mapping scan numbers to {mz: array, intensity: array}.
        output_dir: Directory for pickle cache files.

    Returns:
        Tuple of (fragment_dict, correlations_fragment_dict) where:
        - fragment_dict: {"peptide/charge": filtered DataFrame} per peptidoform.
        - correlations_fragment_dict: {"peptide/charge": 9-element list of correlation arrays/scalars}.
    """
    if read_correlation_pickles and not write_correlation_pickles:
        try:
            with open(f"{output_dir}/fragment_dict.pkl", "rb") as f:
                fragment_dict = pickle.load(f)
            with open(f"{output_dir}/correlations_fragment_dict.pkl", "rb") as f:
                correlations_fragment_dict = pickle.load(f)
            log_info("Successfully loaded correlation data from pickle files")
            return fragment_dict, correlations_fragment_dict
        except FileNotFoundError:
            log_info("Pickle files not found, will compute correlations instead")
            read_correlation_pickles = False  # Fall back to computation

    fragment_dict = {}
    correlations_fragment_dict = {}

    df_fragment_max_peptide = df_fragment_max_peptide.with_columns(
        (pl.col("peptide") + "/" + pl.col("charge").cast(pl.Utf8)).alias("precursor")
    )


    precursor_to_rt_max = dict(
        zip(
            df_fragment_max_peptide["precursor"].to_list(),
            df_fragment_max_peptide["rt"].to_list(),
        )
    )


    df_precursor_rt = pl.DataFrame(
        {
            "precursor": list(precursor_to_rt_max.keys()),
            "rt_max_peptide_sub": list(precursor_to_rt_max.values()),
        }
    )

    df_fragment = df_fragment.with_columns(
        (pl.col("peptide") + "/" + pl.col("charge").cast(pl.Utf8)).alias("precursor")
    )

    df_fragment = df_fragment.join(df_precursor_rt, on="precursor", how="left")

    # Filter fragments to the retention time window around the apex.
    # If calibrated RT margins are available (rt_lower_margin / rt_higher_margin),
    # use them for per-peptidoform adaptive windows. Otherwise fall back to the
    # fixed ±filter_max_apex_rt seconds window.
    if "rt_lower_margin" in df_fragment.columns and "rt_higher_margin" in df_fragment.columns:
        df_fragment = df_fragment.filter(
            (pl.col("rt_max_peptide_sub").is_not_null())
            & (
                # Use calibrated margins where available, fall back to fixed window where NaN
                pl.when(pl.col("rt_lower_margin").is_not_null())
                .then(
                    (pl.col("rt") >= pl.col("rt_lower_margin"))
                    & (pl.col("rt") <= pl.col("rt_higher_margin"))
                )
                .otherwise(
                    abs(pl.col("rt") - pl.col("rt_max_peptide_sub")) < filter_max_apex_rt
                )
            )
        )
        log_info("Fragment filtering: using calibrated RT margins (with fixed fallback)")
    else:
        df_fragment = df_fragment.filter(
            (pl.col("rt_max_peptide_sub").is_not_null())
            & (abs(pl.col("rt") - pl.col("rt_max_peptide_sub")) < filter_max_apex_rt)
        )
        log_info(f"Fragment filtering: using fixed ±{filter_max_apex_rt}s window (no margins available)")

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

        results = match_fragments(df_fragment_sub_peptidoform, preds, ms2_dict)

        fragment_dict[f"{peptidoform}/{charge}"] = df_fragment_sub_peptidoform
        # Keep backward compatibility: convert dataclass back to list for downstream code
        correlations_fragment_dict[f"{peptidoform}/{charge}"] = [
            results.correlations,
            results.correlations_count,
            results.sum_pred_frag_intens,
            results.correlation_matrix_psm_ids,
            results.correlation_matrix_frag_ids,
            results.most_intens_cor,
            results.most_intens_cos,
            results.mse_avg_pred_intens,
            results.mse_avg_pred_intens_total,
        ]

    if write_correlation_pickles:
        with open(f"{output_dir}/fragment_dict.pkl", "wb") as f:
            pickle.dump(fragment_dict, f)
        with open(f"{output_dir}/correlations_fragment_dict.pkl", "wb") as f:
            pickle.dump(correlations_fragment_dict, f)

    return fragment_dict, correlations_fragment_dict
