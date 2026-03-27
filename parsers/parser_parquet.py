"""Sage parquet result parsing with modification normalization and DataFrame construction."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import polars as pl

from utilities.logger import log_info


def replace_mass_shift(
    peptide: str,
    replace_dict: Dict[str, str] = {
        "[+57.0215]": "[Carbamidomethyl]",
        "[+57.021465]": "[Carbamidomethyl]",
        "[+15.9949]": "[Oxidation]",
        "[-18.010565]": "[Glu->pyro-Glu]",
        "[-17.026548]": "[Gln->pyro-Glu]",
        "[+0.984016]": "[Deamidated]",
        "[+14.01565]": "[Methyl]",
        "[+27.994915]": "[Formyl]",
        "[+28.0313]": "[Dimethyl]",
        "[+79.96633]": "[Phospho]",
        "[+31.989829]": "[Dioxidation]",
        "[+31.989828]": "[Dioxidation]",
        "[+42.010565]": "[Acetyl]",
        "[+42.010567]": "[Acetyl]",
        "[+12.0000000]": "[Thiazolidine]",
        "[+12.000000]": "[Thiazolidine]",
        "[+12.00000]": "[Thiazolidine]",
        "[+12.0000]": "[Thiazolidine]",
        "[+12.000]": "[Thiazolidine]",
        "[+12.00]": "[Thiazolidine]",
        "[+12.0]": "[Thiazolidine]",
        "[-18.010565]": "[Glu->pyro-Glu]",
        "[-17.026549]": "[Gln->pyro-Glu]",
        "[-17.026549]": "[Gln->pyro-Glu]",
        "[+17.026549]": "[Ammonium]",
        "[+44.985078]": "[Nitro]",
        "[+44.985077]": "[Nitro]",
        "[+43.005814]": "[Carbamyl]",
        "[+114.042927]": "[GG]",
        "[+114.04293]": "[GG]",
        "[+114.03169]": "[Gluratylation]",
        "[+56.026215]": "[Delta:H(4)C(3)O(1)]",
        "[+71.03712]": "[Propionamide]",
    },
) -> str:
    """
    Replace mass shift annotations with standardized modification names.

    Converts numeric mass shift annotations (e.g., [+57.0215]) to readable
    modification names (e.g., [Carbamidomethyl]) for better interpretability.

    Args:
        peptide: Peptide sequence string with mass shift annotations
        replace_dict: Dictionary mapping mass shifts to modification names

    Returns:
        Peptide sequence with standardized modification names
    """
    for k, v in replace_dict.items():
        peptide = peptide.replace(k, v)
    return peptide


def parquet_reader(
    parquet_file_results: Union[str, Path] = "results.sage.parquet",
    parquet_file_fragments: Union[str, Path] = "matched_fragments.sage.parquet",
    q_value_filter: float = 1.0,
) -> Tuple[
    Optional[pl.DataFrame],
    Optional[pl.DataFrame],
    Optional[pl.DataFrame],
    Optional[pl.DataFrame],
]:
    """
    Load and process Sage search results from parquet files.

    This function loads PSM and fragment data from Sage parquet outputs, applies
    q-value filtering, processes modification annotations, and creates derived
    DataFrames for downstream analysis.

    Args:
        parquet_file_results: Path to Sage PSM results parquet file
        parquet_file_fragments: Path to Sage fragment matches parquet file
        q_value_filter: Q-value threshold for filtering PSMs (default: 1.0 = no filtering)

    Returns:
        Tuple containing:
        - df_fragment: All fragment matches with peptide info joined
        - df_psms: Filtered PSMs with fragment intensities added
        - df_fragment_max: Maximum intensity fragment per PSM
        - df_fragment_max_peptide: Maximum intensity fragment per unique peptide

        Returns (None, None, None, None) if no fragments pass the q-value filter
    """
    # Sage writes parquet files using the Arrow/Pandas convention, so we read
    # them with Pandas first (which handles Sage's schema directly) and then
    # convert to Polars DataFrames for the rest of the pipeline.
    df_fragment = pd.read_parquet(parquet_file_fragments)
    df_fragment.index = df_fragment["psm_id"]

    df_psms = pd.read_parquet(parquet_file_results)
    log_info("df_psms shape: {}".format(df_psms.shape))
    df_psms.drop_duplicates(
        subset=["scannr", "peptide", "charge"], inplace=True
    )  # Okay to add charge?
    log_info("df_psms shape after dropping duplicates: {}".format(df_psms.shape))

    df_psms = df_psms[df_psms["spectrum_q"] < q_value_filter]
    log_info("df_psms shape after filtering by q-value: {}".format(df_psms.shape))
    df_fragment = df_fragment[df_fragment.index.isin(df_psms["psm_id"])]

    # Convert from Pandas to Polars for efficient downstream processing
    df_fragment = pl.DataFrame(df_fragment)
    df_psms = pl.DataFrame(df_psms)

    if len(df_fragment["psm_id"]) == 0:
        log_info(
            "No fragments passed the q-value filter. Returning empty DataFrames."
        )  # TODO: shouuld throw error instead?
        return None, None, None, None

    # df_fragment_max.index = df_fragment_max["psm_id"]
    # df_psms.index = df_psms["psm_id"]
    # df_psms = pd.concat([df_psms, df_fragment_max["fragment_intensity"]], axis=1)

    # Normalize numeric mass-shift annotations in peptide sequences to named
    # modification strings. For example, [+57.0215] becomes [Carbamidomethyl].
    # This is needed because Sage outputs raw mass shifts, but downstream tools
    # (MS2PIP, DeepLC) expect named modification formats.
    df_psms = df_psms.with_columns(
        pl.col("peptide").map_elements(replace_mass_shift).alias("peptide")
    )
    log_info("df_psms shape after replacing mass shifts: {}".format(df_psms.shape))

    df_fragment = df_fragment.join(
        df_psms[["psm_id", "peptide", "charge", "rt"]], on="psm_id", how="left"
    )

    # Get the maximum fragment intensity per PSM
    df_fragment_max = df_fragment.sort("fragment_intensity", descending=True).unique(
        subset="psm_id", keep="first", maintain_order=True
    )

    # Build the "apex PSM per peptidoform" table: for each unique
    # (peptide, charge) pair, keep only the PSM with the highest fragment
    # intensity. This gives one representative spectrum per precursor, which
    # is used later for feature extraction and RT calibration.
    df_fragment_max_peptide = (
        df_fragment_max.with_columns(
            [
                # Add a combined identifier for grouping
                (pl.col("peptide") + "/" + pl.col("charge").cast(pl.Utf8)).alias(
                    "precursor"
                )
            ]
        )
        .sort("fragment_intensity", descending=True)
        .unique(subset=["peptide", "charge"], keep="first", maintain_order=True)
    )

    # df_psms = pd.concat([df_psms, df_fragment_max["fragment_intensity"]], axis=1)

    df_psms = df_psms.join(
        df_fragment_max[["psm_id", "fragment_intensity"]],
        on="psm_id",
        how="left",
    )
    log_info("df_psms shape after joining with fragment max: {}".format(df_psms.shape))

    return df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide
