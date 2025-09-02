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

import polars as pl

import utilities.pickling as pickling
from config import get_config  # Clean, simple config import
from data_structures import PickleConfig, SpectraData
from utilities.io_utils import remove_intermediate_files
from utilities.logger import log_info

def main() -> str:
    """
    Main MuMDIA workflow orchestrator.

    This function coordinates the entire MuMDIA pipeline using clean config system.
    """
    # Get configuration with clean, simple interface
    config_obj = get_config()
    log_info(f"Starting MuMDIA workflow with config file: {config_obj._config_file}")
    
    # Create directories
    result_dir = Path(config_obj.result_dir)
    result_temp = result_dir / "temp"
    result_temp_results_initial_search = result_temp / "initial_search_results"
    
    # Create all necessary directories
    result_dir.mkdir(parents=True, exist_ok=True)
    result_temp.mkdir(parents=True, exist_ok=True)
    result_temp_results_initial_search.mkdir(parents=True, exist_ok=True)
    
    # Get the mumdia configuration dictionary for backwards compatibility
    config = config_obj.to_legacy_format()

    # Lazy imports for heavy modules to avoid import errors during test collection
    from parsers.parser_mzml import get_ms1_mzml, split_mzml_by_retention_time
    from parsers.parser_parquet import parquet_reader
    from peptide_search.wrapper_sage import retention_window_searches, run_sage
    from prediction_wrappers.wrapper_deeplc import retrain_and_bounds
    from sequence.fasta import tryptic_digest_pyopenms

    args_dict = config["mumdia"]

    # Configure pickle settings once for the entire workflow
    pickle_config = PickleConfig(
        write_deeplc=args_dict["write_deeplc_pickle"],
        write_ms2pip=args_dict["write_ms2pip_pickle"],
        write_correlation=args_dict["write_correlation_pickles"],
        read_deeplc=args_dict["read_deeplc_pickle"],
        read_ms2pip=args_dict["read_ms2pip_pickle"],
        read_correlation=args_dict["read_correlation_pickles"],
    )

    # ============================================================================
    # STAGE 1: Initial Search for Retention Time Model Training
    # ============================================================================
    # The MuMDIA pipeline uses a two-stage search strategy:
    # 1. Initial broad search: Used to train DeepLC retention time models
    # 2. Targeted search: Uses RT predictions to partition data for faster, more accurate searches

    # Check if all required initial search pickle files exist
    initial_search_pickles = [
        "df_fragment_initial_search.pkl",
        "df_psms_initial_search.pkl",
        "df_fragment_max_initial_search.pkl",
        "df_fragment_max_peptide_initial_search.pkl",
        "config_initial_search.pkl",
        "dlc_transfer_learn_initial_search.pkl",
        "flags_initial_search.pkl",
    ]
    initial_search_pickles_exist = all(
        os.path.exists(result_dir.joinpath(pickle_file))
        for pickle_file in initial_search_pickles
    )

    # Initialize variables to satisfy type checking and ensure defined in all branches
    df_fragment = pl.DataFrame()
    df_psms = pl.DataFrame()
    df_fragment_max = pl.DataFrame()
    df_fragment_max_peptide = pl.DataFrame()
    dlc_transfer_learn = None

    if args_dict["write_initial_search_pickle"] or not initial_search_pickles_exist:
        log_info("Running initial Sage search for RT model training...")
        # TODO: Earlier, implement a check whether the mzML file exists, because otherwise Sage will still run on an non-existing file and later on an error will be raised that is not very informative.
        run_sage(
            config["sage_basic"],
            args_dict["fasta_file"],
            result_temp_results_initial_search,
        )

        df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide = parquet_reader(
            parquet_file_results=result_temp_results_initial_search / "results.sage.parquet",
            parquet_file_fragments=result_temp_results_initial_search / "matched_fragments.sage.parquet",
            q_value_filter=args_dict["fdr_init_search"],
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
            config=config,
            dlc_transfer_learn=None,
            pickle_config=pickle_config,
            write_full_search_pickle=args_dict["write_full_search_pickle"],
            read_full_search_pickle=args_dict["read_full_search_pickle"],
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

    if args_dict["read_initial_search_pickle"]:
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

        del flags["write_full_search_pickle"]
        del flags["read_full_search_pickle"]
        args_dict.update(flags)

    # Ensure DataFrames are concrete types for downstream usage
    assert isinstance(df_psms, pl.DataFrame)
    assert isinstance(df_fragment, pl.DataFrame)
    assert isinstance(df_fragment_max, pl.DataFrame)
    assert isinstance(df_fragment_max_peptide, pl.DataFrame)

    log_info("Number of PSMs after initial search: {}".format(len(df_psms)))

    # ============================================================================
    # STAGE 2: Targeted Search with Retention Time Partitioning
    # ============================================================================
    # This stage uses the trained DeepLC model to predict retention times for all
    # possible peptides, then partitions the mzML data by retention time for
    # targeted searches that are both faster and more accurate.

    # Check if all required initial search pickle files exist
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

    if args_dict["write_full_search_pickle"] or not full_search_pickles_exist:
        log_info("Generating peptide library and training DeepLC model...")
        peptides = tryptic_digest_pyopenms(config["sage"]["database"]["fasta"])

        # Train DeepLC retention time model and calculate prediction bounds
        # Narrow type for static analysis
        assert isinstance(df_psms, pl.DataFrame)
        peptide_df, dlc_calibration, dlc_transfer_learn, perc_95 = retrain_and_bounds(
            cast(pl.DataFrame, df_psms), peptides, result_dir=result_dir
        )

        log_info("Partitioning mzML files by predicted retention time...")
        mzml_dict = split_mzml_by_retention_time(
            config["sage_basic"]["mzml_paths"][0],  # use configured mzML
            time_interval=perc_95,
            dir_files=str(result_dir),
        )

        (
            df_fragment,
            df_psms,
            df_fragment_max,
            df_fragment_max_peptide,
        ) = retention_window_searches(mzml_dict, peptide_df, config, perc_95)

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
            config=config,
            dlc_transfer_learn=dlc_transfer_learn,
            pickle_config=pickle_config,
            write_full_search_pickle=args_dict["write_full_search_pickle"],
            read_full_search_pickle=args_dict["read_full_search_pickle"],
            dir=result_dir,
            write_to_tsv=True,
        )

    if args_dict["read_full_search_pickle"]:
        (
            df_fragment,
            df_psms,
            df_fragment_max,
            df_fragment_max_peptide,
            config,
            dlc_transfer_learn,
            flags,
        ) = pickling.read_variables_from_pickles(dir=result_dir)
        args_dict.update(flags)

    # ============================================================================
    # STAGE 3: Feature Calculation and Machine Learning Pipeline
    # ============================================================================
    # Parse mzML to extract MS1 precursor information for additional features
    log_info("Parsing the mzML file for MS1 precursor information...")
    ms1_dict, ms2_to_ms1_dict, ms2_spectra = get_ms1_mzml(
        config["sage_basic"]["mzml_paths"][0]  # TODO: should be for all mzml files
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

    # Import mumdia only when needed to avoid dependency issues
    import mumdia
    
    mumdia.main(
        df_fragment=df_fragment,
        df_psms=df_psms,
        df_fragment_max=df_fragment_max,
        df_fragment_max_peptide=df_fragment_max_peptide,
        config=config,
        deeplc_model=dlc_transfer_learn,
        pickle_config=pickle_config,
        spectra_data=spectra_data,
    )

    # ============================================================================
    # STAGE 4: Optional Cleanup and Final Processing
    # ============================================================================
    # Clean up intermediate files if requested to save disk space
    if args_dict["remove_intermediate_files"]:
        log_info("Cleaning up intermediate files...")
        remove_intermediate_files(args_dict["result_dir"])

    return config["mumdia"]["result_dir"]


if __name__ == "__main__":
    output_dir = main()  # For now output output_dir, should be handled differently
    # Run Mokapot for final statistical validation and FDR control
    try:
        from mumdia import run_mokapot

        run_mokapot(output_dir)
    except Exception as e:
        log_info(f"Skipping mokapot run: {e}")
