#!/usr/bin/env python3
"""
MuMDIA (Multi-modal Data-Independent Acquisition) Main Workflow

This is the main entry point for the MuMDIA proteomics analysis pipeline.
MuMDIA integrates multiple prediction tools and machine learning approaches
to improve peptide-spectrum match scoring in data-independent acquisition workflows.

The pipeline includes:
1. Peptide database generation and FASTA processing
2. Sage peptide search engine execution
3. DeepLC retention time prediction with transfer learning
4. MS2PIP fragment intensity prediction
5. Feature generation from multiple modalities
6. Machine learning-based PSM scoring with Mokapot
7. Quality control and result validation

Key Features:
- Retention time-partitioned searches for improved speed
- Multi-modal feature integration (RT, fragment intensity, precursor features)
- Configurable machine learning models (XGBoost, Neural Networks, Percolator)
- Comprehensive logging and intermediate result caching
- Parallel processing for computational efficiency

Usage:
    python run.py --mzml_file data.mzML --fasta_file proteins.fasta --result_dir results/
"""

import os

os.environ["POLARS_MAX_THREADS"] = "1"

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import polars as pl

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
from utilities.pickling import (
    read_variables_from_pickles,
    write_variables_to_pickles,
)


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
        help="Flag to indicate if intermediate results should be removed",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--write_initial_search_pickle",
        help="Flag to indicate if all result pickles should be written",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--read_initial_search_pickle",
        help="Flag to indicate if all result pickles should be read",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--write_deeplc_pickle",
        help="Flag to indicate if DeepLC pickles should be written",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--write_ms2pip_pickle",
        help="Flag to indicate if MS2PIP pickles should be written",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--read_deeplc_pickle",
        help="Flag to indicate if DeepLC pickles should be read",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--read_ms2pip_pickle",
        help="Flag to indicate if MS2PIP pickles should be read",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--write_correlation_pickles",
        help="Flag to indicate if correlation pickles should be written",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--read_correlation_pickles",
        help="Flag to indicate if correlation pickles should be read",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--dlc_transfer_learn",
        help="Flag to indicate if DeepLC should use transfer learning",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--write_full_search_pickle",
        help="Flag to indicate if the full search pickles should be written",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--read_full_search_pickle",
        help="Flag to indicate if the full search pickles should be read",
        type=bool,
        default=True,
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
    Update the configuration JSON file with command-line overrides if explicitly provided.

    This function loads an existing configuration and ensures that under the "mumdia" key,
    only those parameters that the user has explicitly specified on the command line will
    override the JSON config. Missing values are filled from argparse defaults.

    Args:
        config_file: Path to the original JSON configuration file
        result_dir: Path to the result directory for saving updated config
        parser: The ArgumentParser used to obtain default values and option strings
        args: The parsed command-line arguments

    Returns:
        Path to the updated configuration JSON file
    """
    # Load existing configuration if it exists
    if os.path.exists(config_file):
        with open(config_file, "r") as file:
            config = json.load(file)
    else:
        log_info(
            f"Warning: Config file '{config_file}' not found. Using argparse defaults."
        )
        config = {}

    # Ensure "mumdia" exists in the config
    if "mumdia" not in config:
        config["mumdia"] = {}

    # Obtain default values from the parser for all arguments that have a default
    default_args = {
        action.dest: action.default
        for action in parser._actions
        if action.default is not None
    }

    updated = False

    # Update only those values that were explicitly provided by the user.
    for key, value in vars(args).items():
        if was_arg_explicitly_provided(parser, key):
            # Only override if either the key is missing or the value differs.
            if key not in config["mumdia"] or config["mumdia"][key] != value:
                config["mumdia"][key] = value
                updated = True
        else:
            # If no value exists in the config, fill it with the argparse default.
            if key not in config["mumdia"]:
                config["mumdia"][key] = default_args.get(key, value)
                updated = True

    # Update mzML and FASTA paths in config if explicitly provided
    for section in ["sage_basic", "sage"]:
        if section not in config:
            config[section] = {}
        if was_arg_explicitly_provided(parser, "mzml_file"):
            config[section]["mzml_paths"] = [args.mzml_file]
        if was_arg_explicitly_provided(parser, "fasta_file"):
            config[section]["database"] = {"fasta": args.fasta_file}

    # Define new config path in the results folder
    new_config_path = os.path.join(result_dir, "updated_config.json")

    if updated:
        with open(new_config_path, "w") as file:
            json.dump(config, file, indent=4)
        log_info(f"Configuration updated and saved to {new_config_path}")
    else:
        log_info("No configuration changes were made, using existing values.")

    return new_config_path


def main() -> None:
    """
    Main MuMDIA workflow orchestrator.

    This function coordinates the entire MuMDIA pipeline:
    1. Parse command line arguments
    2. Create directory structure
    3. Update configuration with CLI overrides
    4. Run initial or full search workflow based on flags
    5. Perform feature calculation and machine learning
    6. Optional cleanup of intermediate files
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

    if args_dict["write_initial_search_pickle"] or not os.path.exists(
        result_dir.joinpath(
            result_temp, result_temp_results_initial_search, "results.sage.parquet"
        )
    ):
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

        write_variables_to_pickles(
            df_fragment=df_fragment,
            df_psms=df_psms,
            df_fragment_max=df_fragment_max,
            df_fragment_max_peptide=df_fragment_max_peptide,
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
        ) = read_variables_from_pickles(
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

    # ============================================================================
    # STAGE 2: Targeted Search with Retention Time Partitioning
    # ============================================================================
    # This stage uses the trained DeepLC model to predict retention times for all
    # possible peptides, then partitions the mzML data by retention time for
    # targeted searches that are both faster and more accurate.

    if args_dict["write_full_search_pickle"] or not os.path.exists(
        result_dir.joinpath("results.sage.parquet")
    ):
        log_info("Generating peptide library and training DeepLC model...")
        peptides = tryptic_digest_pyopenms(config["sage"]["database"]["fasta"])

        # Train DeepLC retention time model and calculate prediction bounds
        peptide_df, dlc_calibration, dlc_transfer_learn, perc_95 = retrain_and_bounds(
            df_psms, peptides, result_dir=result_dir
        )

        log_info("Partitioning mzML files by predicted retention time...")
        mzml_dict = split_mzml_by_retention_time(
            "mzml_files/LFQ_Orbitrap_AIF_Ecoli_01.mzML",
            time_interval=perc_95,
            dir_files="results/temp/",
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

        write_variables_to_pickles(
            df_fragment=df_fragment,
            df_psms=df_psms,
            df_fragment_max=df_fragment_max,
            df_fragment_max_peptide=df_fragment_max_peptide,
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
        ) = read_variables_from_pickles(dir=result_dir)
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


if __name__ == "__main__":
    main()
    # Run Mokapot for final statistical validation and FDR control
    run_mokapot()
