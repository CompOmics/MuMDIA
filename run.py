import os

os.environ["POLARS_MAX_THREADS"] = "1"

import argparse
import json

import mumdia
import pathlib
from pathlib import Path

from parsers.parser_mzml import split_mzml_by_retention_time, get_ms1_mzml
from parsers.parser_parquet import parquet_reader
from prediction_wrappers.wrapper_deeplc import (
    retrain_deeplc,
    predict_deeplc,
)
import pandas as pd
import polars as pl

import numpy as np
import pandas as pd

from sequence.fasta import write_to_fasta
from utilities.logger import log_info
from utilities.pickling import (
    write_variables_to_pickles,
    read_variables_from_pickles,
)
from peptide_search.wrapper_sage import run_sage
from sequence.fasta import tryptic_digest_pyopenms
from parsers.parser_mzml import split_mzml_by_retention_time
from parsers.parser_parquet import parquet_reader
from prediction_wrappers.wrapper_deeplc import retrain_deeplc
from prediction_wrappers.wrapper_deeplc import predict_deeplc

from utilities.io_utils import remove_intermediate_files
from utilities.io_utils import create_dirs

from peptide_search.wrapper_sage import run_sage
from peptide_search.wrapper_sage import retention_window_searches

from mumdia import run_mokapot

from prediction_wrappers.wrapper_deeplc import retrain_and_bounds


def parse_arguments():
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
        default="fasta/ecoli_22032024.fasta",
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
        default=True,
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
        default=True,
    )

    parser.add_argument(
        "--write_ms2pip_pickle",
        help="Flag to indicate if MS2PIP pickles should be written",
        type=bool,
        default=True,
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
        default=True,
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

    return parser.parse_args()


def modify_config(config_file, result_dir, **kwargs):
    """
    Modify the configuration file with command-line arguments and save it in the results folder.
    Only updates the config if arguments are provided.

    Args:
        config_file (str): Original config file path.
        result_dir (str): Path to the result directory.
        **kwargs: Command-line arguments as key-value pairs.
    Returns:
        str: Path to the new saved config file.
    """
    with open(config_file, "r") as file:
        config = json.load(file)

    updated = False
    for key, value in kwargs.items():
        if value is not None:  # Update only if a new value is provided
            keys = key.split(".")  # Allow nested keys
            sub_config = config
            for k in keys[:-1]:
                sub_config = sub_config.setdefault(k, {})
            if sub_config.get(keys[-1]) != value:
                sub_config[keys[-1]] = value
                updated = True

    # Define new config path in results folder
    new_config_path = os.path.join(result_dir, "updated_config.json")

    if updated:
        with open(new_config_path, "w") as file:
            json.dump(config, file, indent=4)
        log_info(f"Configuration updated and saved to {new_config_path}")
    else:
        log_info("No configuration changes made.")

    return new_config_path  # Return path to the new config file


def main():
    log_info("Parsing command line arguments...")
    args = parse_arguments()

    log_info("Creating the result directory...")
    result_dir, result_temp, result_temp_results_initial_search = create_dirs(args)

    log_info("Updating configuration if needed and saving to results folder...")
    new_config_file = modify_config(
        args.config_file,
        result_dir=args.result_dir,
    )

    log_info("Reading the updated configuration JSON file...")
    with open(new_config_file, "r") as file:
        config = json.load(file)

    if args.write_initial_search_pickle:
        run_sage(
            config["sage_basic"],
            args.fasta_file,
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
            q_value_filter=config["mumdia"]["fdr_init_search"],
        )

    if args.write_initial_search_pickle:
        write_variables_to_pickles(
            df_fragment=df_fragment,
            df_psms=df_psms,
            df_fragment_max=df_fragment_max,
            df_fragment_max_peptide=df_fragment_max_peptide,
            config=config,
            dlc_transfer_learn=args.dlc_transfer_learn,
            write_deeplc_pickle=args.write_deeplc_pickle,
            write_ms2pip_pickle=args.write_ms2pip_pickle,
            write_correlation_pickles=args.write_correlation_pickles,
            write_full_search_pickle=args.write_full_search_pickle,
            read_deeplc_pickle=args.read_deeplc_pickle,
            read_ms2pip_pickle=args.read_ms2pip_pickle,
            read_correlation_pickles=args.read_correlation_pickles,
            read_full_search_pickle=args.write_full_search_pickle,
            dir=result_dir,
        )

    if args.read_initial_search_pickle:
        (
            df_fragment,
            df_psms,
            df_fragment_max,
            df_fragment_max_peptide,
            config,
            dlc_transfer_learn,
            flags,
        ) = read_variables_from_pickles(dir=result_dir)
        args.update(flags)

    if args.write_full_search_pickle:
        peptides = tryptic_digest_pyopenms(args.fasta_file)

        peptide_df, dlc_calibration, dlc_transfer_learn, perc_95 = retrain_and_bounds(
            df_psms, peptides, result_dir=result_dir
        )

        mzml_dict = split_mzml_by_retention_time(
            "LFQ_Orbitrap_AIF_Ecoli_01.mzML",
            time_interval=perc_95,
            dir_files="results/temp/",
        )

        df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide = (
            retention_window_searches(mzml_dict, peptide_df, config, perc_95)
        )

        log_info("Adding the PSM identifier to fragments...")
        df_fragment = df_fragment.join(
            df_psms.select(["psm_id", "scannr"]), on="psm_id", how="left"
        )

    if args.write_full_search_pickle:
        write_variables_to_pickles(
            df_fragment=df_fragment,
            df_psms=df_psms,
            df_fragment_max=df_fragment_max,
            df_fragment_max_peptide=df_fragment_max_peptide,
            config=config,
            dlc_transfer_learn=args.dlc_transfer_learn,
            write_deeplc_pickle=args.write_deeplc_pickle,
            write_ms2pip_pickle=args.write_ms2pip_pickle,
            write_correlation_pickles=args.write_correlation_pickles,
            write_full_search_pickle=args.write_full_search_pickle,
            read_deeplc_pickle=args.read_deeplc_pickle,
            read_ms2pip_pickle=args.read_ms2pip_pickle,
            read_correlation_pickles=args.read_correlation_pickles,
            read_full_search_pickle=args.write_full_search_pickle,
            dir=result_dir,
        )

    if args.read_full_search_pickle:
        (
            df_fragment,
            df_psms,
            df_fragment_max,
            df_fragment_max_peptide,
            config,
            dlc_transfer_learn,
            flags,
        ) = read_variables_from_pickles(dir=result_dir)
        args.update(flags)

    mumdia.main(
        df_fragment=df_fragment,
        df_psms=df_psms,
        df_fragment_max=df_fragment_max,
        df_fragment_max_peptide=df_fragment_max_peptide,
        config=config,
        deeplc_model=dlc_transfer_learn,
        write_deeplc_pickle=args.write_deeplc_pickle,
        write_ms2pip_pickle=args.write_ms2pip_pickle,
        read_deeplc_pickle=args.read_deeplc_pickle,
        read_ms2pip_pickle=args.read_ms2pip_pickle,
    )

    if args.remove_intermediate_files:
        remove_intermediate_files(args.result_dir)


if __name__ == "__main__":
    main()
    run_mokapot()
