"""Fragment-level label-free quantification using trapezoidal integration over RT."""

import logging
import os

import directlfq.config as lfqconfig
import directlfq.protein_intensity_estimation as lfqprot_estimation
import directlfq.utils as lfqutils
import polars as pl
from Bio import SeqIO
from pyteomics import proforma
from scipy import integrate
from tqdm import tqdm

from utilities.logger import log_info


def quantify_fragments(df_fragment, mokapot_psm_path, config, output_dir: str = None):
    """Quantify fragment ions based on their intensities over retention time.
    This function processes a DataFrame containing fragment ion data, filters it based on mokapot results,
    and quantifies the fragment ions using both peak intensity and integrated intensity methods.
    The results are saved to a CSV file in the specified output directory.

    TODO: These files should be combined for multiple runs and then used for directLFQ on fragment level.
    TODO: Compare which quantification method works best, remove the others.

    Args:
        df_fragment (pl.DataFrame): DataFrame containing fragment ion data with columns:
            - 'peptide': Peptide sequence with modifications.
            - 'stripped_peptide': Peptide sequence without modifications.
            - 'charge': Charge state of the peptide.
            - 'fragment_name': Name of the fragment ion (e.g., 'b3', 'y7').
            - 'proteins': Protein(s) the peptide maps to.
            - 'rt': Retention time of the fragment ion measurement.
            - 'fragment_intensity': Intensity of the fragment ion.
            - 'psm_id': Unique identifier for the PSM.
            - 'rt_lower_margin': Lower margin of the retention time window.
            - 'rt_higher_margin': Upper margin of the retention time window.
        mokapot_psm_path (str): Path to the mokapot PSM results TSV file (contains 'Peptide'
            and 'mokapot q-value' columns).
        config (dict): Configuration dictionary containing config['sage']['mzml_paths'] —
            list of mzML file paths (first element used for output naming).
        output_dir (str, optional): Directory to save the output CSV file. Defaults to None.
    Returns:
        pl.DataFrame: DataFrame with quantified fragment ion intensities.
    """
    # TODO: adapt this so it works for multiple runs
    # Extract the mzML filename (without extension) from the first path in the
    # Sage config. This basename is used as a column header in the output so
    # that multi-run results can later be merged on a per-run basis.
    mzml_filename = os.path.basename(config["sage"]["mzml_paths"][0]).split(".")[0]

    # Keep only fragments whose parent peptide passed Mokapot FDR control at
    # 1% q-value. Mokapot's "Peptide" column stores the modified sequence that
    # matches the "peptide" column in df_fragment.
    mokapot_peptides = pl.read_csv(mokapot_psm_path, separator="\t")
    mokapot_peptides = mokapot_peptides.filter(pl.col("mokapot q-value") < 0.01)
    df_fragment_mokapot_filtered = df_fragment.filter(
        pl.col("peptide").is_in(mokapot_peptides["Peptide"])
    )

    # Derive stripped_peptide (remove modifications like [Oxidation], [Carbamidomethyl], etc.)
    if "stripped_peptide" not in df_fragment_mokapot_filtered.columns:
        df_fragment_mokapot_filtered = df_fragment_mokapot_filtered.with_columns(
            pl.col("peptide").str.replace_all(r"\[.*?\]", "").alias("stripped_peptide")
        )

    # Join proteins from mokapot results if not already in df_fragment
    if "proteins" not in df_fragment_mokapot_filtered.columns:
        proteins_map = mokapot_peptides.select(["Peptide", "Proteins"]).unique(
            subset=["Peptide"]
        )
        df_fragment_mokapot_filtered = df_fragment_mokapot_filtered.join(
            proteins_map, left_on="peptide", right_on="Peptide", how="left"
        ).rename({"Proteins": "proteins"})

    results = []

    logging.info(f"Quantifying fragments")

    for (
        peptidoform,
        stripped_sequence,
        charge,
        fragment_name,
        proteins,
    ), df_fragment_mokapot_filtered_sub in tqdm(
        df_fragment_mokapot_filtered.group_by(
            ["peptide", "stripped_peptide", "charge", "fragment_name", "proteins"]
        )
    ):
        # Protein strings are in UniProt FASTA header format: "sp|P12345|PROT_HUMAN".
        # Split on '|' and take the third field (entry name) for each protein,
        # then rejoin with ';' for multi-protein groups.
        proteins = ";".join(
            proteinstring.split("|")[2] for proteinstring in proteins.split(";")
        )

        # Build a unique ion identifier that encodes the stripped sequence, full
        # modified peptidoform, charge state, and fragment name. This composite
        # key allows directLFQ to track individual fragment ions across runs.
        results.append(
            {
                "protein": proteins,
                "ion": "SEQ_"
                + stripped_sequence
                + "_MOD"
                + peptidoform
                + "_CHARGE_"
                + str(charge)
                + "_"
                + fragment_name,
                # mzml_filename + "_Intensity_peak": quantify_fragment_peak_intensity(df_fragment_mokapot_filtered_sub, margin=False),
                # mzml_filename + "_Intensity_peak_margin": quantify_fragment_peak_intensity(df_fragment_mokapot_filtered_sub, margin=True),
                mzml_filename
                + "_Intensity_integrated": quantify_fragment_integrated_intensity(
                    df_fragment_mokapot_filtered_sub, margin=False
                ),
                # mzml_filename + "_Intensity_integrated_margin": quantify_fragment_integrated_intensity(df_fragment_mokapot_filtered_sub, margin=True),
            }
        )

    df_quant_fragment = pl.DataFrame(results)
    df_quant_fragment.write_csv(
        os.path.join(output_dir, mzml_filename + "_fragment_level_intensities.csv")
    )

    return df_quant_fragment


def quantify_fragment_peak_intensity(df_fragment_ion_psms, margin: bool = False):
    """
    Return the maximum fragment intensity across all retention times.

    Args:
        df_fragment_ion_psms: Polars DataFrame with fragment_intensity column.
        margin: If True, filter to RT margin bounds first (requires rt_lower_margin/rt_higher_margin columns).

    Returns:
        Maximum fragment intensity value (float).
    """
    # return highest intensity of fragment over RT
    if margin:
        df_fragment_ion_psms = _filter_margin_rt(df_fragment_ion_psms)

    return df_fragment_ion_psms["fragment_intensity"].max()


def quantify_fragment_integrated_intensity(df_fragment_ion_psms, margin: bool = False):
    """
    Integrate fragment intensities over retention time using the trapezoidal rule.

    For a single time point, returns that intensity directly. For multiple time
    points, sorts by RT and computes the area under the curve via
    scipy.integrate.trapezoid.

    Args:
        df_fragment_ion_psms: Polars DataFrame with rt and fragment_intensity columns.
        margin: If True, filter to RT margin bounds first.

    Returns:
        Integrated intensity (float) or single-point intensity.
    """
    # integrate fragment intensities over RT
    if margin:
        df_fragment_ion_psms = _filter_margin_rt(df_fragment_ion_psms)

    # for fragments only measured at one time point, return that intensity
    if df_fragment_ion_psms.shape[0] == 1:
        return df_fragment_ion_psms["fragment_intensity"].item()

    # sort by RT to ensure correct integration
    df_fragment_ion_psms = df_fragment_ion_psms.sort("rt", descending=False)

    # approximate integration using trapezoidal rule
    aoc = integrate.trapezoid(
        y=df_fragment_ion_psms["fragment_intensity"].to_numpy(),
        x=df_fragment_ion_psms["rt"].to_numpy(),
    )
    return aoc


def _filter_margin_rt(df):
    """Filter DataFrame to rows within the calibrated RT margin bounds.
    Only applies filtering when multiple PSM IDs are present."""
    if df["psm_id"].n_unique() > 1:
        df = df.filter(
            (pl.col("rt") >= pl.col("rt_lower_margin"))
            & (pl.col("rt") <= pl.col("rt_higher_margin"))
        )
    return df


def quantify_proteins(df_fragment_quant_folder, output_dir: str = None):
    """Estimate protein intensities from fragment ion quantifications using directLFQ.
    This function is still TODO and not yet implemented.

    Args:
        df_fragment_quant_folder (str): Path to the folder containing fragment ion quantification CSV files.
        output_dir (str, optional): Directory to save the output CSV file. Defaults to None.
    Returns:
        directLFQ protein intensity DataFrame.

    """

    # copy pasterino from alphadia:
    # log_info.info("Performing label-free protein quantification using directLFQ")

    # combine all fragment quantification files in the folder
    # df_results_fragment = pd.concat(
    #    [
    #        pd.read_csv(os.path.join(df_fragment_quant_folder, f))
    #        for f in os.listdir(df_fragment_quant_folder)
    #        if f.endswith("_fragment_level_intensities.csv")
    #    ],
    #    ignore_index=True,
    # )

    # extract intensity columns
    # intensity_cols = [col for col in df_results_fragment.columns if col.endswith("_Intensity_integrated_margin")]
    # _intensity_df = df_results_fragment.select("Proteins", "ion", *intensity_cols)

    # lfqconfig.set_global_protein_and_ion_id(protein_id='Proteins', quant_id=mzml_filename + "_Intensity_integrated_margin")
    # lfqconfig.set_compile_normalized_ion_table(
    #    compile_normalized_ion_table=False
    # )  # save compute time by avoiding the creation of a normalized ion table
    # lfqconfig.check_wether_to_copy_numpy_arrays_derived_from_pandas()  # avoid read-only pandas bug on linux if applicable
    # lfqconfig.set_log_processed_proteins(
    #    log_processed_proteins=True
    # )  # here you can chose wether to log the processed proteins or not

    # _intensity_df.sort_values(by='_Intensity_integrated_margin', inplace=True, ignore_index=True)

    # lfq_df = lfqutils.index_and_log_transform_input_df(_intensity_df)
    # lfq_df = lfqutils.remove_allnan_rows_input_df(lfq_df)

    # protein_df, _ = lfqprot_estimation.estimate_protein_intensities(
    #    lfq_df,
    #    min_nonan=1,
    #    num_samples_quadratic=50,
    # )

    # protein_df.to_csv(os.path.join(output_dir, "protein_level_intensities_directlfq_out.csv"))

    # return protein_df
