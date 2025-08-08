"""
Sage Peptide Search Engine Wrapper for MuMDIA

This module provides Python interfaces for the Sage peptide search engine,
enabling both single-file searches and sophisticated retention time-partitioned
searches for improved speed and accuracy in proteomics workflows.

Key Features:
- Direct Sage execution with configuration management
- Retention time-based mzML partitioning for targeted searches
- Automatic PSM ID management across partitions
- Fragment intensity analysis and data consolidation
- Quality control through data integrity assertions

The retention window search strategy significantly reduces search space by
using retention time predictions to limit peptide candidates for each
time-partitioned mzML file, leading to faster and more accurate identifications.
"""

import copy
import json
import os
import pathlib
import subprocess
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from parsers.parser_parquet import parquet_reader
from sequence.fasta import write_to_fasta


def run_sage(
    config: Dict[str, Any], fasta_file: str, output_dir: Union[str, pathlib.Path]
) -> None:
    """
    Execute Sage peptide search engine via subprocess.

    This function writes the Sage configuration to a JSON file and executes
    the Sage binary with the specified parameters for peptide identification.

    Args:
        config: Dictionary containing Sage configuration parameters
        fasta_file: Path to the protein database FASTA file
        output_dir: Directory where Sage results will be written

    Returns:
        None (results written to parquet files in output_dir)
    """
    # Write the Sage configuration to a JSON file in the output directory
    # This allows Sage to read all parameters from a single configuration file
    json_path = pathlib.Path(output_dir).joinpath("sage_values.json")

    # Serialize the entire config dictionary to JSON format
    with open(json_path, "w") as file:
        json.dump(config, file, indent=4)

    # Log the exact command that will be executed for debugging purposes
    print(
        " ".join(
            map(
                str,
                [
                    "bin/sage",
                    json_path,
                    "-o",
                    output_dir,
                    "--annotate-matches",
                    "--parquet",
                    "--disable-telemetry-i-dont-want-to-improve-sage",
                ],
            )
        )
    )

    # Execute Sage peptide search engine with the configuration file
    # Key flags explained:
    # --annotate-matches: Include detailed fragment ion annotations in output
    # --parquet: Save results in Parquet format for efficient data processing
    # --disable-telemetry: Prevent Sage from sending usage statistics
    subprocess.run(
        [
            "bin/sage",  # Sage executable binary
            json_path,  # Configuration file path
            "-o",  # Output directory flag
            output_dir,  # Directory for results
            "--annotate-matches",  # Enable fragment ion annotations
            "--parquet",  # Use Parquet output format
            "--disable-telemetry-i-dont-want-to-improve-sage",  # Disable telemetry
        ]
    )


def retention_window_searches(
    mzml_dict: Dict[float, str],
    peptide_df: pd.DataFrame,
    config: Dict[str, Any],
    perc_95: float,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Perform Sage searches on retention time-partitioned mzML files.

    This function runs targeted searches on time-based mzML partitions using
    peptides predicted to elute in each time window, then combines results.

    Args:
        mzml_dict: Mapping of retention time upper bounds to mzML file paths
        peptide_df: DataFrame with peptides and RT prediction bounds
        config: Sage configuration dictionary
        perc_95: 95th percentile RT error for window overlap calculation

    Returns:
        Tuple containing combined results:
        - df_fragment: All fragment matches across time windows
        - df_psms: All PSMs across time windows
        - df_fragment_max: Maximum intensity fragments per PSM
        - df_fragment_max_peptide: Maximum intensity fragments per peptide
    """
    df_fragment_list = []
    df_psms_list = []
    psm_ident_start = 0  # Running counter to ensure unique PSM IDs across partitions

    # Process each retention time partition sequentially
    for upper_mzml_partition, mzml_path in mzml_dict.items():
        # Calculate peptide selection mask using retention time predictions
        # This selects peptides whose RT prediction intervals overlap with the current partition
        # The overlap calculation accounts for prediction uncertainty (perc_95)
        peptide_selection_mask = np.maximum(
            peptide_df["predictions_lower"], upper_mzml_partition - perc_95
        ) <= np.minimum(peptide_df["predictions_upper"], upper_mzml_partition)

        sub_peptide_df = peptide_df[peptide_selection_mask]

        # Skip partitions with no predicted peptides to avoid empty searches
        if len(sub_peptide_df.index) == 0:
            continue

        # Create a partition-specific FASTA file containing only predicted peptides
        # This targeted approach significantly reduces search space and improves speed
        fasta_file = os.path.join(os.path.dirname(mzml_path), "vectorized_output.fasta")
        write_to_fasta(sub_peptide_df, output_file=fasta_file)

        # Use a deep copy of the provided config to avoid mutating caller state
        sub_config = copy.deepcopy(config)

        sub_results = os.path.dirname(mzml_path)

        # Update configuration for this specific partition
        sub_config["sage"]["database"][
            "fasta"
        ] = fasta_file  # Set partition-specific FASTA
        sub_config["sage"]["mzml_paths"] = [mzml_path]  # Set partition-specific mzML

        # Execute Sage search on this retention time partition
        run_sage(sub_config["sage"], fasta_file, sub_results)

        # Parse Sage results from JSON metadata file
        result_file_json_path = pathlib.Path(sub_results).joinpath("results.json")
        try:
            with open(result_file_json_path, "r") as file:
                sage_result_json = json.load(file)
        except:
            # Handle cases where Sage failed to produce results for this partition
            print(result_file_json_path)
            continue

        # Load and process Sage output parquet files
        # q_value_filter=1.0 means no q-value filtering at this stage
        df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide = parquet_reader(
            parquet_file_results=sage_result_json["output_paths"][0],
            parquet_file_fragments=sage_result_json["output_paths"][1],
            q_value_filter=1.0,
        )

        # Skip partitions that produced no valid results
        if df_fragment is None:
            continue

        # Narrow types for static analysis
        assert isinstance(df_fragment, pl.DataFrame)
        assert isinstance(df_psms, pl.DataFrame)

        # Ensure unique PSM IDs across all partitions by adding running offset
        # This prevents ID conflicts when combining results from multiple partitions
        df_fragment = df_fragment.with_columns(
            (df_fragment["psm_id"] + psm_ident_start).alias("psm_id")
        )
        df_psms = df_psms.with_columns(
            (df_psms["psm_id"] + psm_ident_start).alias("psm_id")
        )

        # Accumulate results from this partition
        df_fragment_list.append(df_fragment)
        df_psms_list.append(df_psms)

        # Update PSM ID offset for next partition
        # Ensure numeric max for offset (handle empty/None safely)
        next_offset = psm_ident_start
        try:
            # Compute scalar max safely via an aggregation
            max_series = df_psms.select(pl.col("psm_id").max()).to_series()
            max_value = max_series[0] if len(max_series) > 0 else None
            if isinstance(max_value, (int, np.integer, float, np.floating)):
                next_offset = int(max_value) + 1
        except Exception:
            # Keep existing offset if anything unexpected happens
            pass
        psm_ident_start = next_offset

    # Combine results from all retention time partitions
    if len(df_fragment_list) == 0 or len(df_psms_list) == 0:
        # No results were produced; return empty DataFrames to keep pipeline running
        empty = pl.DataFrame()
        return empty, empty, empty, empty

    df_fragment = pl.concat(df_fragment_list)
    df_psms = pl.concat(df_psms_list)

    # Generate derived DataFrames for downstream analysis
    # df_fragment_max: Contains the highest intensity fragment for each PSM
    # This is used for feature generation and quality assessment
    df_fragment_max = df_fragment.sort("fragment_intensity", descending=True).unique(
        subset="psm_id", keep="first", maintain_order=True
    )

    # df_fragment_max_peptide: Contains the highest intensity fragment for each unique peptide
    # This helps identify the best representative fragment for each peptide sequence
    # Note: Could be extended to consider charge state as well
    df_fragment_max_peptide = df_fragment_max.unique(
        subset=["peptide"],
        keep="first",
    )

    # Data integrity checks to ensure consistent results across processing steps
    # These assertions help catch bugs in the partition processing logic
    assert len(set(df_fragment["peptide"])) == len(set(df_psms["peptide"])), (
        f"Mismatch in unique peptides: "
        f"{len(set(df_fragment['peptide']))} in df_fragment vs "
        f"{len(set(df_psms['peptide']))} in df_psms"
    )
    assert len(set(df_fragment_max["peptide"])) == len(set(df_psms["peptide"])), (
        f"Mismatch in unique peptides after max fragment selection: "
        f"{len(set(df_fragment_max['peptide']))} in df_fragment_max vs "
        f"{len(set(df_psms['peptide']))} in df_psms"
    )

    return (df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide)
