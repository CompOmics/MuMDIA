from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
import random
import polars as pl


def generate_random_spectrum(num_peaks, mz_range, intensity_range):
    """
    Generate a random spectrum for testing purposes.
    
    Args:
        num_peaks: Number of peaks to generate
        mz_range: Tuple of (min_mz, max_mz) for m/z values
        intensity_range: Tuple of (min_intensity, max_intensity) for intensities
    
    Returns:
        List of tuples [(mz, intensity), ...]
    """
    spectrum = []
    for _ in range(num_peaks):
        mz = random.uniform(mz_range[0], mz_range[1])
        intensity = random.uniform(intensity_range[0], intensity_range[1])
        spectrum.append((mz, intensity))
    
    # Sort by m/z
    spectrum.sort(key=lambda x: x[0])
    return spectrum


def lasso_deconv():
    # Expected columns for PSM fragment data
    fragment_columns = [
        "psm_id",
        "fragment_type",
        "fragment_ordinals",
        "fragment_charge",
        "fragment_mz_experimental",
        "fragment_mz_calculated",
        "fragment_intensity",
        "peptide",
        "charge",
        "rt",
        "scannr",
        "peak_identifier",
    ]

    # Expected columns for PSM data
    psm_columns = [
        "psm_id",
        "filename",
        "scannr",
        "peptide",
        "stripped_peptide",
        "proteins",
        "num_proteins",
        "rank",
        "is_decoy",
        "expmass",
        "calcmass",
        "charge",
        "peptide_len",
        "missed_cleavages",
        "semi_enzymatic",
        "ms2_intensity",
        "isotope_error",
        "precursor_ppm",
        "fragment_ppm",
        "hyperscore",
        "delta_next",
        "delta_best",
        "rt",
        "aligned_rt",
        "predicted_rt",
        "delta_rt_model",
        "ion_mobility",
        "predicted_mobility",
        "delta_mobility",
        "matched_peaks",
        "longest_b",
        "longest_y",
        "longest_y_pct",
        "matched_intensity_pct",
        "scored_candidates",
        "poisson",
        "sage_discriminant_score",
        "posterior_error",
        "spectrum_q",
        "peptide_q",
        "protein_q",
        "reporter_ion_intensity",
        "fragment_intensity",
    ]

    # Parameters
    num_experimental_peaks = 50
    num_theoretical_spectra = 1500
    mz_range = (100, 600)
    intensity_range = (10, 100)

    # Generate experimental spectrum
    experimental_spectrum = generate_random_spectrum(
        num_experimental_peaks, mz_range, intensity_range
    )

    # Generate theoretical spectra with a random number of matched peaks
    theoretical_spectra = []
    for _ in range(num_theoretical_spectra):
        num_peaks = random.randint(5, num_experimental_peaks)
        theoretical_spectrum = generate_random_spectrum(
            num_peaks, mz_range, intensity_range
        )
        theoretical_spectra.append(theoretical_spectrum)

    # Convert spectra to numpy arrays for processing
    def spectrum_to_vector(spectrum, mz_values):
        mz_dict = dict(spectrum)
        return np.array([mz_dict.get(mz, 0) for mz in mz_values])

    # Get the set of all unique m/z values
    all_mz_values = sorted(set(mz for mz, _ in experimental_spectrum))

    # Convert experimental spectrum to vector
    exp_vector = spectrum_to_vector(experimental_spectrum, all_mz_values)

    # Convert theoretical spectra to matrix
    theoretical_matrix = np.array(
        [spectrum_to_vector(spec, all_mz_values) for spec in theoretical_spectra]
    ).T

    # Use Lasso to find the coefficients with L1 regularization
    lasso = Lasso(alpha=100.0, positive=True, max_iter=10000)
    lasso.fit(theoretical_matrix, exp_vector)
    coefficients = lasso.coef_

    # Print the results
    print("Coefficients:", coefficients)

    # Reconstruct the experimental spectrum from the theoretical spectra using the coefficients
    reconstructed_spectrum = np.dot(theoretical_matrix, coefficients)

    # Plot the experimental and reconstructed spectra for comparison
    plt.figure(figsize=(12, 6))
    plt.vlines(
        all_mz_values, 0, exp_vector, label="Experimental Spectrum", color="blue"
    )
    plt.vlines(
        all_mz_values,
        0,
        reconstructed_spectrum,
        label="Reconstructed Spectrum",
        color="red",
        linestyle="--",
    )
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title("Experimental vs Reconstructed Spectrum")
    plt.show()

    # Plot the stacked contribution of each theoretical spectrum
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(all_mz_values))
    colors = plt.cm.tab20(np.linspace(0, 1, num_theoretical_spectra))
    for j, vector in enumerate(theoretical_spectra):
        contribution = coefficients[j] * spectrum_to_vector(vector, all_mz_values)
        plt.bar(
            all_mz_values,
            contribution,
            bottom=bottom,
            color=colors[j],
            edgecolor="white",
            width=4,
            label=f"Theoretical Spectrum {j+1}",
        )
        bottom += contribution

    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    # plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title("Stacked Contributions of Theoretical Spectra")
    plt.show()


def deconv_filter_fragment_spectra(
    ms2pip_predictions: dict,
    df_fragment: pl.DataFrame,
    min_coefficient=0.0000000001,
    alpha_penalty=0.1,
):
    # Flatten the nested dictionary into a single mapping
    flat_ms2pip = {
        (peptide, fragment): value
        for peptide, fragments in ms2pip_predictions.items()
        for fragment, value in fragments.items()
    }

    df_fragment = df_fragment.with_columns(
        pl.concat_str([pl.col("peptide"), pl.col("charge")], separator="/").alias(
            "peptidoform"
        )
    )

    df_fragment = df_fragment.with_columns(
        pl.struct(["peptidoform", "fragment_name"])
        .apply(lambda x: flat_ms2pip.get((x["peptidoform"], x["fragment_name"]), None))
        .alias("predicted_intensity")
    )

    df_fragment = df_fragment.with_columns(pl.col("predicted_intensity").fill_null(0.0))

    selected_peptidoforms = []
    for group in df_fragment.group_by("scannr"):
        selected_peptidoforms = fit_lass_deconv(
            min_coefficient, alpha_penalty, selected_peptidoforms, group
        )
    return selected_peptidoforms


def fit_lass_deconv(min_coefficient, alpha_penalty, selected_peptidoforms, group):
    df_scan = group[1].sort("peak_identifier")

    exp_spectrum_intensity = (
        df_scan.unique(subset=["peak_identifier"], keep="first")["fragment_intensity"]
        .fill_null(0.0)
        .to_numpy()
    )
    exp_spectrum_intensity_normalized = exp_spectrum_intensity / np.sum(
        exp_spectrum_intensity
    )

    pivot_df = df_scan.pivot(
        values="predicted_intensity",
        index="peptidoform",
        columns="peak_identifier",
        aggregate_function="sum",
    )
    pivot_df_intensity = pivot_df[pivot_df.columns[1:]].fill_null(0.0).to_numpy().T

    lasso = Lasso(alpha=alpha_penalty, positive=True, max_iter=100000)
    lasso.fit(pivot_df_intensity, exp_spectrum_intensity_normalized)
    coefficients = lasso.coef_

    selected_peptidoforms_filter = pl.Series(coefficients > min_coefficient)

    selected_peptidoforms.extend(
        list(pivot_df[pivot_df.columns[0]].filter(selected_peptidoforms_filter))
    )

    return selected_peptidoforms
