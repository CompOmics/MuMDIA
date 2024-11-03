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
    # Create an argument parser
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
        help="Flag to indicate if all result pickles (including deeplc models) should be written",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--read_initial_search_pickle",
        help="Flag to indicate if all result pickles (including deeplc models) should be read",
        type=bool,
        default=True,
    )

    # Parse the command line arguments
    return parser.parse_args()


def main():
    log_info("Parsing command line arguments...")
    # Parse the command line arguments
    args = parse_arguments()

    log_info("Creating the result directory...")
    # TODO overwrite configs supplied by the user
    # modify_config(args.key, args.value, config=args.config_file)

    result_dir, result_temp, result_temp_results_initial_search = create_dirs(args)

    log_info("Reading the configuration json file...")
    # Read the config file
    with open(args.config_file, "r") as file:
        config = json.load(file)

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
        q_value_filter=0.001,
    )

    if args.write_initial_search_pickle:
        write_variables_to_pickles(
            df_fragment=df_fragment,
            df_psms=df_psms,
            df_fragment_max=df_fragment_max,
            df_fragment_max_peptide=df_fragment_max_peptide,
            config=config,
            dlc_transfer_learn=True,
            write_deeplc_pickle=True,
            write_ms2pip_pickle=True,
            write_correlation_pickles=True,
            dir=result_dir,
        )

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

    log_info("Add the PSM identifier to fragments...")
    df_fragment = df_fragment.join(
        df_psms.select(["psm_id", "scannr"]), on="psm_id", how="left"
    )
    log_info("DONE - Add the PSM identifier to fragments...")

    if args.write_initial_search_pickle:
        write_variables_to_pickles(
            df_fragment=df_fragment,
            df_psms=df_psms,
            df_fragment_max=df_fragment_max,
            df_fragment_max_peptide=df_fragment_max_peptide,
            config=config,
            dlc_transfer_learn=dlc_transfer_learn,
            write_deeplc_pickle=True,
            write_ms2pip_pickle=True,
            write_correlation_pickles=True,
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

    mumdia.main(
        df_fragment=df_fragment,
        df_psms=df_psms,
        df_fragment_max=df_fragment_max,
        df_fragment_max_peptide=df_fragment_max_peptide,
        config=config,
        deeplc_model=dlc_transfer_learn,
        write_deeplc_pickle=True,
        write_ms2pip_pickle=True,
        read_deeplc_pickle=False,
        read_ms2pip_pickle=False,
    )

    # Remove intermediate files if specified
    if args.remove_intermediate_files:
        remove_intermediate_files(args.result_dir)


if __name__ == "__main__":
    main()
    run_mokapot()
