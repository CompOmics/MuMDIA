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
from tqdm import tqdm

from utilities.logger import log_info
from data_structures import PickleConfig, CorrelationResults
from rustyms import (
    FragmentationModel,
    LinearPeptide,
    MassMode,
    RawSpectrum,
    CompoundPeptidoform,
)


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
        y = pred_frag_intens         # Predicted intensities
        
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
       Match fragments and calculate their correlation.

       Parameters:
       - df_fragment_sub_peptidoform (pl.DataFrame): A DataFrame containing fragment data.
       - ms2pip_predictions (dict): A dictionary of MS2PIP predictions.
       - ms2_dict (dict): Dictionary containing MS2 spectrum data.

       Returns:
       - CorrelationResults: A dataclass containing all correlation analysis results with descriptive field names.

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

    ion_pattern = r"ion='([^']*)'"
    charge_pattern = r"charge=(\d+),"

    fragment_records = []

    unique_psm_id = df_fragment_sub_peptidoform \
                .sort("fragment_intensity",descending=True) \
                .unique(subset=["psm_id"], keep="first")
    
    unique_psm_id_dicts = unique_psm_id.to_dicts()
    
    for row in unique_psm_id_dicts:
        psm_id = int(row["psm_id"])
        rt = float(row["rt"])
        scannr = row["scannr"]
        rt_max_peptide_sub = float(row["rt_max_peptide_sub"])
        precursor_charge = int(row["fragment_charge"])
        scannr = row["scannr"]
        peptide = row["peptide"]

        spectrum = RawSpectrum(
            title=row["scannr"],
            num_scans=1,
            rt=float(rt),
            precursor_charge=precursor_charge,
            precursor_mass=1.0,
            mz_array=ms2_dict[scannr]["mz"],
            intensity_array=ms2_dict[scannr]["intensity"],
        )

        linear_peptide = CompoundPeptidoform(
            peptide
        )

        annotated_spectrum = spectrum.annotate(
            peptide=linear_peptide,
            model=FragmentationModel.CidHcd,
            mode=MassMode.Monoisotopic,
        )

        matched_fragments = [
            annotated_peak
            for annotated_peak in annotated_spectrum.spectrum
            if annotated_peak.annotation
        ]
        
        for mf in matched_fragments:
            ion_label = re.search(ion_pattern, repr(mf.annotation[0])).group(1)
            ion_charge = re.search(charge_pattern, repr(mf.annotation[0])).group(1)

            fragment_records.append({
                "psm_id": psm_id,
                "fragment_type": ion_label[0],
                "fragment_ordinals": ion_label[1:],
                "fragment_charge": ion_charge,
                "fragment_intensity": mf.intensity,
                "rt": rt,
                "scannr": scannr,
                "fragment_name": f"{ion_label}/{ion_charge}",
                "rt_max_peptide_sub": rt_max_peptide_sub,
            })
    if len(fragment_records) != 0:
        df_fragment_sub_peptidoform = pl.DataFrame(fragment_records)\
                                        .sort("fragment_intensity",descending=True)\
                                        .unique(subset=["psm_id", "fragment_name"], keep="first")
    
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

    # Do a max normalization of the MS2PIP predictions by dividing by the maximum
    # predicted intensity
    max_intens_ms2pip = max(ms2pip_predictions.values())
    ms2pip_predictions = dict(
        [(k, v / max_intens_ms2pip) for k, v in ms2pip_predictions.items()]
    )

    """
    Get pearson and cosing similarity of spectrum with highest intensity
    """

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

    """
    Get pearson and cosing similarity of spectrum with highest intensity
    """

    """
    Get the intensity matrix of observations
    """

    # first column is PSM ID, ignore that one, messes up calculation as it is numeric
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

    return CorrelationResults(
        correlations=correlation_result,
        correlations_count=correlation_result_counts,
        sum_pred_frag_intens=sum_pred_frag_intens,
        correlation_matrix_psm_ids=correlation_matrix_psm_ids,
        correlation_matrix_frag_ids=correlation_matrix_frag_ids,
        correlation_matrix_psm_ids_ignore_zeros=correlation_matrix_psm_ids_ignore_zeros,
        correlation_matrix_psm_ids_ignore_zeros_counts=correlation_matrix_psm_ids_ignore_zeros_counts,
        correlation_matrix_psm_ids_missing=correlation_matrix_psm_ids_missing,
        correlation_matrix_psm_ids_missing_zeros_counts=correlation_matrix_psm_ids_missing_zeros_counts,
        correlation_matrix_frag_ids_ignore_zeros=correlation_matrix_frag_ids_ignore_zeros,
        correlation_matrix_frag_ids_ignore_zeros_counts=correlation_matrix_frag_ids_ignore_zeros_counts,
        correlation_matrix_frag_ids_missing=correlation_matrix_frag_ids_missing,
        correlation_matrix_frag_ids_missing_zeros_counts=correlation_matrix_frag_ids_missing_zeros_counts,
        most_intens_cor=most_intens_cor,
        most_intens_cos=most_intens_cos,
        mse_avg_pred_intens=mse_avg_pred_intens,
        mse_avg_pred_intens_total=mse_avg_pred_intens_total,
    )


def get_features_fragment_intensity(
    ms2pip_predictions: dict,
    df_fragment: pl.DataFrame,
    df_fragment_max_peptide: pl.DataFrame,
    filter_max_apex_rt: float = 3.0,
    read_correlation_pickles: bool = False,
    write_correlation_pickles: bool = False,
    ms2_dict: dict = {},
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

            results = match_fragments(df_fragment_sub_peptidoform, preds, ms2_dict)

            fragment_dict[f"{peptidoform}/{charge}"] = df_fragment_sub_peptidoform
            # Keep backward compatibility: convert dataclass back to list for downstream code
            correlations_fragment_dict[f"{peptidoform}/{charge}"] = [
                results.correlations,
                results.correlations_count,
                results.sum_pred_frag_intens,
                results.correlation_matrix_psm_ids,
                results.correlation_matrix_frag_ids,
                results.correlation_matrix_psm_ids_ignore_zeros,
                results.correlation_matrix_psm_ids_ignore_zeros_counts,
                results.correlation_matrix_psm_ids_missing,
                results.correlation_matrix_psm_ids_missing_zeros_counts,
                results.correlation_matrix_frag_ids_ignore_zeros,
                results.correlation_matrix_frag_ids_ignore_zeros_counts,
                results.correlation_matrix_frag_ids_missing,
                results.correlation_matrix_frag_ids_missing_zeros_counts,
                results.most_intens_cor,
                results.most_intens_cos,
                results.mse_avg_pred_intens,
                results.mse_avg_pred_intens_total,
            ]

        if write_correlation_pickles:
            with open("fragment_dict.pkl", "wb") as f:
                pickle.dump(fragment_dict, f)
            with open("correlations_fragment_dict.pkl", "wb") as f:
                pickle.dump(correlations_fragment_dict, f)
    if read_correlation_pickles:
        try:
            with open("fragment_dict.pkl", "rb") as f:
                fragment_dict = pickle.load(f)
            with open("correlations_fragment_dict.pkl", "rb") as f:
                correlations_fragment_dict = pickle.load(f)
            log_info("Successfully loaded correlation data from pickle files")
        except FileNotFoundError:
            log_info("Pickle files not found, will compute correlations instead")
            read_correlation_pickles = False  # Fall back to computation

    return fragment_dict, correlations_fragment_dict
 