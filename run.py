#!/usr/bin/env python3
"""
MuMDIA (Multi-modal Data-Independant Acquisition) Main Workflow

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
import polars as pl
import argparse
import json
import sys
from typing import Tuple, cast

import mumdia
from data_structures import PickleConfig, SpectraData
from mumdia import run_mokapot
from parsers.parser_mzml import get_ms1_mzml, split_mzml_by_retention_time
from parsers.parser_parquet import parquet_reader
from peptide_search.wrapper_sage import retention_window_searches, run_sage
from prediction_wrappers.wrapper_deeplc import retrain_and_bounds
from sequence.fasta import tryptic_digest_pyopenms
from utilities.io_utils import create_dirs, remove_intermediate_files
from utilities.logger import log_info
from utilities.config_loader import merge_config_from_sources, write_updated_config
import utilities.pickling as pickling


def parse_arguments() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """
    Parse command line arguments for the MuMDIA workflow.

    Returns:
        Tuple containing:
        - parser: ArgumentParser object for checking explicitly provided arguments
        - args: Namespace object with parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--mzml_file",
        help="The location of the mzml file",
        default="mzml_files/LFQ_Orbitrap_AIF_Ecoli_01.mzML",
    )
    parser.add_argument(
        "--mzml_dir", help="The directory of the mzml file", default="mzml_files"
    )
    parser.add_argument(
        "--fasta_file",
        help="The location of the fasta file",
        default="fasta/unmodified_peptides.fasta",
    )
    parser.add_argument(
        "--result_dir", help="The location of the result directory", default="results"
    )
    parser.add_argument(
        "--config_file",
        help="The location of the config file",
        default="configs/config.json",
    )

    parser.add_argument(
        "--remove_intermediate_files",
        help="Remove intermediate results after completion",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--write_initial_search_pickle",
        help="Write initial search pickles",
        action="store_true",
        default=False,
    )

    # Default: read initial search pickles (can be disabled with --no-read_initial_search_pickle)
    parser.add_argument(
        "--read_initial_search_pickle",
        dest="read_initial_search_pickle",
        help="Read initial search pickles",
        action="store_true",
    )
    parser.add_argument(
        "--no-read_initial_search_pickle",
        dest="read_initial_search_pickle",
        help="Do not read initial search pickles",
        action="store_false",
    )
    parser.set_defaults(read_initial_search_pickle=True)

    parser.add_argument(
        "--write_deeplc_pickle",
        help="Write DeepLC pickles",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--write_ms2pip_pickle",
        help="Write MS2PIP pickles",
        action="store_true",
        default=False,
    )

    # Default: read DeepLC pickles (can be disabled with --no-read_deeplc_pickle)
    parser.add_argument(
        "--read_deeplc_pickle",
        dest="read_deeplc_pickle",
        help="Read DeepLC pickles",
        action="store_true",
    )
    parser.add_argument(
        "--no-read_deeplc_pickle",
        dest="read_deeplc_pickle",
        help="Do not read DeepLC pickles",
        action="store_false",
    )
    parser.set_defaults(read_deeplc_pickle=True)

    # Default: read MS2PIP pickles (can be disabled with --no-read_ms2pip_pickle)
    parser.add_argument(
        "--read_ms2pip_pickle",
        dest="read_ms2pip_pickle",
        help="Read MS2PIP pickles",
        action="store_true",
    )
    parser.add_argument(
        "--no-read_ms2pip_pickle",
        dest="read_ms2pip_pickle",
        help="Do not read MS2PIP pickles",
        action="store_false",
    )
    parser.set_defaults(read_ms2pip_pickle=True)

    parser.add_argument(
        "--write_correlation_pickles",
        help="Write correlation pickles",
        action="store_true",
        default=False,
    )

    # Default: read correlation pickles (can be disabled with --no-read_correlation_pickles)
    parser.add_argument(
        "--read_correlation_pickles",
        dest="read_correlation_pickles",
        help="Read correlation pickles",
        action="store_true",
    )
    parser.add_argument(
        "--no-read_correlation_pickles",
        dest="read_correlation_pickles",
        help="Do not read correlation pickles",
        action="store_false",
    )
    parser.set_defaults(read_correlation_pickles=True)

    # Default: use DeepLC transfer learning (can be disabled with --no-dlc_transfer_learn)
    parser.add_argument(
        "--dlc_transfer_learn",
        dest="dlc_transfer_learn",
        help="Use DeepLC transfer learning",
        action="store_true",
    )
    parser.add_argument(
        "--no-dlc_transfer_learn",
        dest="dlc_transfer_learn",
        help="Disable DeepLC transfer learning",
        action="store_false",
    )
    parser.set_defaults(dlc_transfer_learn=True)

    parser.add_argument(
        "--write_full_search_pickle",
        help="Write full search pickles",
        action="store_true",
        default=False,
    )

    # Default: read full search pickles (can be disabled with --no-read_full_search_pickle)
    parser.add_argument(
        "--read_full_search_pickle",
        dest="read_full_search_pickle",
        help="Read full search pickles",
        action="store_true",
    )
    parser.add_argument(
        "--no-read_full_search_pickle",
        dest="read_full_search_pickle",
        help="Do not read full search pickles",
        action="store_false",
    )
    parser.set_defaults(read_full_search_pickle=True)

    parser.add_argument(
        "--fdr_init_search",
        help="Q-value (FDR) threshold for initial search filtering",
        type=float,
        default=0.05,
    )

    # Additional possible configuration overrides from CLI
    parser.add_argument(
        "--sage_basic", help="Override sage basic settings in config", type=str
    )
    parser.add_argument(
        "--mumdia_fdr", help="Override mumdia FDR setting in config", type=float
    )

    return parser, parser.parse_args()


def was_arg_explicitly_provided(parser: argparse.ArgumentParser, arg_name: str) -> bool:
    """
    Check if an argument with destination `arg_name` was explicitly provided on the command line.

    Args:
        parser: ArgumentParser object containing argument definitions
        arg_name: Destination name of the argument to check

    Returns:
        True if the argument was explicitly provided, False otherwise
    """
    for action in parser._actions:
        if action.dest == arg_name:
            for option in action.option_strings:
                # If any of the option flags for this argument is present in sys.argv, consider it provided.
                if option in sys.argv:
                    return True
    return False


def modify_config(
    config_file: str,
    result_dir: str,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> str:
    """
    Load existing JSON (if any), merge with defaults + env + explicit CLI, and write to results.

    Returns path to updated config JSON.
    """
    # Load existing configuration if it exists
    existing_config = None
    if os.path.exists(config_file):
        with open(config_file, "r") as file:
            existing_config = json.load(file)
    else:
        log_info(
            f"Warning: Config file '{config_file}' not found. Using argparse defaults + env + CLI."
        )

    merged = merge_config_from_sources(existing_config, parser, args)
    return write_updated_config(merged, result_dir)


def main() -> str:
    """
    Main MuMDIA workflow orchestrator.

    This function coordinates the entire MuMDIA pipeline using argparse + JSON config.
    """
    log_info("Parsing command line arguments...")
    parser, args = parse_arguments()

    log_info("Creating the result directory...")
    result_dir, result_temp, result_temp_results_initial_search = create_dirs(args)

    log_info("Updating configuration if needed and saving to results folder...")
    new_config_file = modify_config(
        args.config_file, result_dir=args.result_dir, parser=parser, args=args
    )

    log_info("Reading the updated configuration JSON file...")
    with open(new_config_file, "r") as file:
        config = json.load(file)

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
            result_dir.joinpath(result_temp, result_temp_results_initial_search),
        )

        df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide = parquet_reader(
            parquet_file_results=result_dir.joinpath(
                result_temp, result_temp_results_initial_search, "results.sage.parquet"
            ),
            parquet_file_fragments=result_dir.joinpath(
                result_temp,
                result_temp_results_initial_search,
                "matched_fragments.sage.parquet",
            ),
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
    run_mokapot(output_dir)
