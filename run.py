#!/usr/bin/env python3
"""
MuMDIA

This is the main entry point for the MuMDIA proteomics analysis pipeline.
MuMDIA integrates multiple prediction tools and machine learning approaches
to improve peptide-spectrum match scoring in data-independent acquisition workflows.

Usage:
    python run.py --mzml_file data.mzML --fasta_file proteins.fasta --result_dir results/
    python run.py --config_file my_config.json
    python run.py --no-cache  # Force recomputation
"""

import os

os.environ["POLARS_MAX_THREADS"] = "1"

from pathlib import Path
from typing import cast
import argparse
import sys
from config import load_config_from_json
import polars as pl

import utilities.pickling as pickling
from data_structures import PickleConfig, SpectraData
from utilities.io_utils import remove_intermediate_files
from utilities.logger import log_info

import mumdia

from parsers.parser_mzml import get_ms1_mzml, split_mzml_by_retention_time
from parsers.parser_parquet import parquet_reader
from peptide_search.wrapper_sage import retention_window_searches, run_sage
from prediction_wrappers.wrapper_deeplc import retrain_and_bounds
from sequence.fasta import tryptic_digest_pyopenms


def run_initial_search(
    config_obj, result_dir, result_temp_results_initial_search, pickle_config
):
    """
    STAGE 1: Initial Search for Retention Time Model Training

    The MuMDIA pipeline uses a two-stage search strategy:
    1. Initial broad search: Used to train DeepLC retention time models
    2. Targeted search: Uses RT predictions to partition data for faster, more accurate searches

    Returns:
        Tuple of (df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide, dlc_transfer_learn)
    """
    # Get initial search config and mumdia settings
    initial_config = config_obj.get_initial_search_config()
    log_info(f"Initial search config: {initial_config}")
    mumdia_config = config_obj.get_mumdia_config()

    # Initialize variables to satisfy type checking and ensure defined in all branches
    df_fragment = pl.DataFrame()
    df_psms = pl.DataFrame()
    df_fragment_max = pl.DataFrame()
    df_fragment_max_peptide = pl.DataFrame()
    dlc_transfer_learn = None

    if not mumdia_config["read_initial_search_pickle"]:
        log_info("Running initial Sage search for RT model training...")
        # TODO: Earlier, implement a check whether the mzML file exists, because
        # otherwise Sage will still run on an non-existing file and later on an error
        # will be raised that is not very informative.
        run_sage(
            initial_config,
            config_obj.fasta_file,
            result_temp_results_initial_search,
        )

        df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide = parquet_reader(
            parquet_file_results=result_temp_results_initial_search.joinpath(
                "results.sage.parquet"
            ),
            parquet_file_fragments=result_temp_results_initial_search.joinpath(
                "matched_fragments.sage.parquet",
            ),
            q_value_filter=config_obj.fdr_init_search,
        )

        # Narrow types for static analysis
        assert isinstance(df_fragment, pl.DataFrame)
        assert isinstance(df_psms, pl.DataFrame)
        assert isinstance(df_fragment_max, pl.DataFrame)
        assert isinstance(df_fragment_max_peptide, pl.DataFrame)

    if mumdia_config["write_initial_search_pickle"]:
        # Create legacy config format for pickling compatibility
        legacy_config = config_obj.to_legacy_format()

        pickling.write_variables_to_pickles(
            df_fragment=cast(pl.DataFrame, df_fragment),
            df_psms=cast(pl.DataFrame, df_psms),
            df_fragment_max=cast(pl.DataFrame, df_fragment_max),
            df_fragment_max_peptide=cast(pl.DataFrame, df_fragment_max_peptide),
            config=legacy_config,
            dlc_transfer_learn=None,
            pickle_config=pickle_config,
            write_full_search_pickle=mumdia_config["write_full_search_pickle"],
            read_full_search_pickle=mumdia_config["read_full_search_pickle"],
            df_fragment_fname="df_fragment_initial_search.pkl",
            df_psms_fname="df_psms_initial_search.pkl",
            df_fragment_max_fname="df_fragment_max_initial_search.pkl",
            df_fragment_max_peptide_fname="df_fragment_max_peptide_initial_search.pkl",
            config_fname="config_initial_search.pkl",
            dlc_transfer_learn_fname="dlc_transfer_learn_initial_search.pkl",
            flags_fname="flags_initial_search.pkl",
            dir=result_dir,
            write_to_tsv=False,
        )

    if mumdia_config["read_initial_search_pickle"]:
        (
            df_fragment,
            df_psms,
            df_fragment_max,
            df_fragment_max_peptide,
            config,
            dlc_transfer_learn,
            flags,
        ) = pickling.read_variables_from_pickles(
            dir=result_dir,
            df_fragment_fname="df_fragment_initial_search.pkl",
            df_psms_fname="df_psms_initial_search.pkl",
            df_fragment_max_fname="df_fragment_max_initial_search.pkl",
            df_fragment_max_peptide_fname="df_fragment_max_peptide_initial_search.pkl",
            config_fname="config_initial_search.pkl",
            dlc_transfer_learn_fname="dlc_transfer_learn_initial_search.pkl",
            flags_fname="flags_initial_search.pkl",
        )

        # Update the config object with any flags that were saved
        # Note: In the new system, flags are handled through the config object
        # so we don't need to update args_dict like before

    # Ensure DataFrames are concrete types for downstream usage
    assert isinstance(df_psms, pl.DataFrame)
    assert isinstance(df_fragment, pl.DataFrame)
    assert isinstance(df_fragment_max, pl.DataFrame)
    assert isinstance(df_fragment_max_peptide, pl.DataFrame)

    log_info("Number of PSMs after initial search: {}".format(len(df_psms)))

    return (
        df_fragment,
        df_psms,
        df_fragment_max,
        df_fragment_max_peptide,
        dlc_transfer_learn,
    )


def run_targeted_search(
    config_obj,
    result_dir,
    pickle_config,
    df_fragment,
    df_psms,
    df_fragment_max,
    df_fragment_max_peptide,
    dlc_transfer_learn,
):
    """
    STAGE 2: Targeted Search with Retention Time Partitioning

    This stage uses the trained DeepLC model to predict retention times for all
    possible peptides, then partitions the mzML data by retention time for
    targeted searches that are both faster and more accurate.

    Args:
        config_obj: MuMDIAConfig object
        result_dir: Result directory path
        pickle_config: Pickle configuration
        df_fragment: Fragment DataFrame from initial search
        df_psms: PSMs DataFrame from initial search
        df_fragment_max: Fragment max DataFrame from initial search
        df_fragment_max_peptide: Fragment max peptide DataFrame from initial search
        dlc_transfer_learn: DeepLC transfer learning model

    Returns:
        Tuple of (df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide, dlc_transfer_learn)
    """
    # Get full search config and mumdia settings
    full_config = config_obj.get_full_search_config()
    mumdia_config = config_obj.get_mumdia_config()

    # Check if all required full search pickle files exist
    full_search_pickles = [
        "df_fragment.pkl",
        "df_psms.pkl",
        "df_fragment_max.pkl",
        "df_fragment_max_peptide.pkl",
        "config.pkl",
        "dlc_transfer_learn.pkl",
        "flags.pkl",
    ]

    full_search_pickles_exist = all(
        os.path.exists(result_dir.joinpath(pickle_file))
        for pickle_file in full_search_pickles
    )

    if mumdia_config["write_full_search_pickle"] or not full_search_pickles_exist:
        # --- Targeted search flow ---
        # 1. Tryptic digest: enumerate all possible peptides from the FASTA database.
        log_info("Generating peptide library and training DeepLC model...")
        peptides = tryptic_digest_pyopenms(config_obj.fasta_file)

        # 2. DeepLC training: use Stage-1 PSMs to train a retention-time model,
        #    then predict RT bounds for every tryptic peptide. perc_95 is the 95th
        #    percentile prediction error used as the RT tolerance window.
        # Narrow type for static analysis
        assert isinstance(df_psms, pl.DataFrame)
        peptide_df, dlc_calibration, dlc_transfer_learn, perc_95 = retrain_and_bounds(
            cast(pl.DataFrame, df_psms), peptides, result_dir=result_dir
        )

        # 3. mzML partitioning: split the original mzML into time slices whose
        #    width equals perc_95, so each slice covers one RT window.
        log_info("Partitioning mzML files by predicted retention time...")
        mzml_dict = split_mzml_by_retention_time(
            config_obj.mzml_file,  # use configured mzML
            time_interval=perc_95,
            dir_files=str(result_dir),
        )

        # Create legacy config format for retention window searches
        legacy_config = config_obj.to_legacy_format()

        # 4. Retention window searches: for each mzML partition, run Sage only
        #    against peptides predicted to elute in that window, then merge results.
        (
            df_fragment,
            df_psms,
            df_fragment_max,
            df_fragment_max_peptide,
        ) = retention_window_searches(mzml_dict, peptide_df, legacy_config, perc_95)

        # Sage's matched_fragments parquet does not include scannr (scan number);
        # it only lives in the PSM results table. Join it onto df_fragment here so
        # that downstream code can link fragments back to their source spectra.
        log_info("Adding the PSM identifier to fragments...")
        df_fragment = df_fragment.join(
            df_psms.select(["psm_id", "scannr"]), on="psm_id", how="left"
        )

        # Narrow types for static analysis
        assert isinstance(df_fragment, pl.DataFrame)
        assert isinstance(df_psms, pl.DataFrame)
        assert isinstance(df_fragment_max, pl.DataFrame)
        assert isinstance(df_fragment_max_peptide, pl.DataFrame)

        pickling.write_variables_to_pickles(
            df_fragment=cast(pl.DataFrame, df_fragment),
            df_psms=cast(pl.DataFrame, df_psms),
            df_fragment_max=cast(pl.DataFrame, df_fragment_max),
            df_fragment_max_peptide=cast(pl.DataFrame, df_fragment_max_peptide),
            config=legacy_config,
            dlc_transfer_learn=dlc_transfer_learn,
            pickle_config=pickle_config,
            write_full_search_pickle=mumdia_config["write_full_search_pickle"],
            read_full_search_pickle=mumdia_config["read_full_search_pickle"],
            dir=result_dir,
            write_to_tsv=True,
        )

    if mumdia_config["read_full_search_pickle"]:
        (
            df_fragment,
            df_psms,
            df_fragment_max,
            df_fragment_max_peptide,
            config,
            dlc_transfer_learn,
            flags,
        ) = pickling.read_variables_from_pickles(dir=result_dir)
        # Note: In the new system, flags are handled through the config object

    return (
        df_fragment,
        df_psms,
        df_fragment_max,
        df_fragment_max_peptide,
        dlc_transfer_learn,
    )


def main():
    """
    Main MuMDIA workflow orchestrator.

    This function coordinates the entire MuMDIA pipeline using the new simplified config system.
    """
    # Parse command line
    argumentsparser = argparse.ArgumentParser(description="Run MuMDIA workflow")
    argumentsparser.add_argument("config_file", help="Path to JSON configuration file")
    args = argumentsparser.parse_args()

    # Load configuration from JSON file
    try:
        config_obj = load_config_from_json(args.config_file)
        log_info(f"Loaded configuration from {args.config_file}")
    except Exception as e:
        log_info(f"Error loading configuration: {e}")
        sys.exit(1)

    log_info(f"Starting MuMDIA workflow with config file: {args.config_file}")

    # Create directories
    result_dir = Path(config_obj.result_dir)
    result_temp = result_dir / "temp"
    result_temp_results_initial_search = result_temp / "initial_search_results"

    # Create all necessary directories
    result_dir.mkdir(parents=True, exist_ok=True)
    result_temp.mkdir(parents=True, exist_ok=True)
    result_temp_results_initial_search.mkdir(parents=True, exist_ok=True)

    # Get mumdia configuration
    mumdia_config = config_obj.get_mumdia_config()

    # Configure pickle settings once for the entire workflow.
    # The mumdia_config dict uses keys like "write_deeplc_pickle" while the
    # PickleConfig dataclass uses shorter field names like "write_deeplc".
    # Each dict key is mapped to the corresponding dataclass field here.
    pickle_config = PickleConfig(
        write_deeplc=mumdia_config["write_deeplc_pickle"],
        write_ms2pip=mumdia_config["write_ms2pip_pickle"],
        write_correlation=mumdia_config["write_correlation_pickles"],
        read_deeplc=mumdia_config["read_deeplc_pickle"],
        read_ms2pip=mumdia_config["read_ms2pip_pickle"],
        read_correlation=mumdia_config["read_correlation_pickles"],
    )

    # Run initial search (Stage 1)
    (
        df_fragment,
        df_psms,
        df_fragment_max,
        df_fragment_max_peptide,
        dlc_transfer_learn,
    ) = run_initial_search(
        config_obj, result_dir, result_temp_results_initial_search, pickle_config
    )

    # Run targeted search (Stage 2)
    (
        df_fragment,
        df_psms,
        df_fragment_max,
        df_fragment_max_peptide,
        dlc_transfer_learn,
    ) = run_targeted_search(
        config_obj,
        result_dir,
        pickle_config,
        df_fragment,
        df_psms,
        df_fragment_max,
        df_fragment_max_peptide,
        dlc_transfer_learn,
    )

    # ============================================================================
    # STAGE 3: Feature Calculation and Machine Learning Pipeline
    # ============================================================================
    # Parse mzML to extract MS1 precursor information for additional features
    log_info("Parsing the mzML file for MS1 precursor information...")
    ms1_dict, ms2_to_ms1_dict, ms2_spectra = get_ms1_mzml(
        config_obj.mzml_file  # Using the mzml_file from the new config object
    )

    # Execute the main MuMDIA feature calculation and machine learning pipeline
    # This includes:
    # - Fragment intensity correlation features (MS2PIP predictions vs experimental)
    # - Retention time prediction error features (DeepLC predictions vs observed)
    # - MS1 precursor features (mass accuracy, intensity, charge state)
    # - Machine learning model training and PSM scoring
    log_info("Running MuMDIA feature calculation and machine learning pipeline...")

    # Configure spectra data
    spectra_data = SpectraData(
        ms1_dict=ms1_dict, ms2_to_ms1_dict=ms2_to_ms1_dict, ms2_dict=ms2_spectra
    )

    mumdia.main(
        df_fragment=df_fragment,
        df_psms=df_psms,
        df_fragment_max=df_fragment_max,
        df_fragment_max_peptide=df_fragment_max_peptide,
        config=config_obj.to_legacy_format(),  # Convert to legacy format for compatibility
        deeplc_model=dlc_transfer_learn,
        pickle_config=pickle_config,
        spectra_data=spectra_data,
    )

    # ============================================================================
    # STAGE 4: Optional Cleanup and Final Processing
    # ============================================================================
    # Clean up intermediate files if requested to save disk space
    if config_obj.remove_intermediate_files:
        log_info("Cleaning up intermediate files...")
        remove_intermediate_files(config_obj.result_dir)

    return config_obj.result_dir


if __name__ == "__main__":
    output_dir = main()
