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

from utilities.pickling import (
    write_variables_to_pickles,
    read_variables_from_pickles,
)


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

from sequence.fasta import write_to_fasta

from utilities.io_utils import remove_intermediate_files
from utilities.io_utils import create_directory
from utilities.io_utils import create_dirs


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
        default="fasta/human.fasta",
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

    # Parse the command line arguments
    return parser.parse_args()


def main(skip_sage=False):
    log_info("Parsing command line arguments...")
    # Parse the command line arguments
    args = parse_arguments()

    # Call the modify_config function with the provided arguments
    # modify_config(args.key, args.value)

    log_info("Creating the result directory...")
    # Create the result directory
    create_directory(args.result_dir)

    log_info("Reading the configuration json file...")
    # Read the config file
    with open(args.config_file, "r") as file:
        config = json.load(file)

    log_info("Running SAGE...")
    # Call the run_sage function with the updated arguments
    if not skip_sage:
        run_sage(config, args.fasta_file, args.result_dir)

    log_info("Writing json")
    # Read the config file
    result_file_json_path = pathlib.Path(args.result_dir).joinpath("results.json")
    with open(result_file_json_path, "r") as file:
        sage_result_json = json.load(file)

    log_info("Running MuMDIA...")
    mumdia.main(
        sage_result_json["output_paths"][0],
        sage_result_json["output_paths"][1],
        q_value_filter=1.0,
        config=config,
    )

    mumdia.run_mokapot()

    # Remove intermediate files if specified
    if args.remove_intermediate_files:
        remove_intermediate_files(args.result_dir)


def retrain_and_bouds(df_psms, result_dir=""):
    dlc_calibration, dlc_transfer_learn, perc_95 = retrain_deeplc(
        df_psms,
        outfile_calib=result_dir.joinpath("deeplc_calibration.png"),
        outfile_transf_learn=result_dir.joinpath("deeplc_transfer_learn.png"),
    )
    perc_95 = perc_95 * 60.0 * 2.0
    predictions = predict_deeplc(peptides, dlc_transfer_learn)

    peptide_df = pd.DataFrame(
        peptides, columns=["protein", "start", "end", "id", "peptide"]
    )
    peptide_df["predictions"] = predictions
    peptide_df["predictions"] = peptide_df["predictions"] * 60.0
    peptide_df.to_csv("peptide_predictions.csv", index=False)
    peptide_df["predictions_lower"] = peptide_df["predictions"] - perc_95
    peptide_df["predictions_upper"] = peptide_df["predictions"] + perc_95

    return peptide_df, dlc_calibration, dlc_transfer_learn, perc_95


def retention_window_searches(mzml_dict, peptide_df, config, perc_95):
    df_fragment_list = []
    df_psms_list = []
    psm_ident_start = 0

    for upper_mzml_partition, mzml_path in mzml_dict.items():
        peptide_selection_mask = np.maximum(
            peptide_df["predictions_lower"], upper_mzml_partition - perc_95
        ) <= np.minimum(peptide_df["predictions_upper"], upper_mzml_partition)

        sub_peptide_df = peptide_df[peptide_selection_mask]

        # Check if any peptides fall in the range of the mzml, otherwise continue
        if len(sub_peptide_df.index) == 0:
            continue

        fasta_file = os.path.join(os.path.dirname(mzml_path), "vectorized_output.fasta")
        write_to_fasta(sub_peptide_df, output_file=fasta_file)

        with open("configs/config.json", "r") as file:
            config = json.load(file)

        sub_results = os.path.dirname(mzml_path)

        config["sage"]["database"]["fasta"] = fasta_file
        config["sage"]["mzml_paths"] = [mzml_path]

        # TODO change mzml in config!
        run_sage(config["sage"], fasta_file, sub_results)

        result_file_json_path = pathlib.Path(sub_results).joinpath("results.json")
        try:
            with open(result_file_json_path, "r") as file:
                sage_result_json = json.load(file)
        except:
            print(result_file_json_path)
            continue

        df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide = parquet_reader(
            parquet_file_results=sage_result_json["output_paths"][0],
            parquet_file_fragments=sage_result_json["output_paths"][1],
            q_value_filter=1.0,
        )

        if df_fragment is None:
            continue

        df_fragment = df_fragment.with_columns(
            (df_fragment["psm_id"] + psm_ident_start).alias("psm_id")
        )
        df_psms = df_psms.with_columns(
            (df_psms["psm_id"] + psm_ident_start).alias("psm_id")
        )

        df_fragment_list.append(df_fragment)
        df_psms_list.append(df_psms)

        psm_ident_start = df_psms["psm_id"].max() + 1

    df_fragment = pl.concat(df_fragment_list)
    df_psms = pl.concat(df_psms_list)

    if len(set(df_fragment["peptide"])) != len(set(df_psms["peptide"])):
        print(len(set(df_fragment["peptide"])), len(set(df_psms["peptide"])))

    df_fragment_max = df_fragment.sort("fragment_intensity", descending=True).unique(
        subset="psm_id", keep="first", maintain_order=True
    )

    if len(set(df_fragment_max["peptide"])) != len(set(df_psms["peptide"])):
        print(len(set(df_fragment_max["peptide"])), len(set(df_psms["peptide"])))

    # might also want to do on charge
    df_fragment_max_peptide = df_fragment_max.unique(
        subset=["peptide"],
        keep="first",
    )

    if len(set(df_fragment_max_peptide["peptide"])) != len(set(df_psms["peptide"])):
        print(
            len(set(df_fragment_max_peptide["peptide"])), len(set(df_psms["peptide"]))
        )

    return (df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide)


if __name__ == "__main__":
    write_initial_search_pickle = True
    read_initial_search_pickle = True
    # Path to your FASTA file
    fasta_file = "fasta/ecoli_22032024.fasta"

    # Perform the tryptic digest and retrieve the list of peptides
    peptides = tryptic_digest_pyopenms(fasta_file)

    args = parse_arguments()
    # TODO overwrite configs supplied by the user
    # modify_config(args.key, args.value, config=args.config_file)

    result_dir, result_temp, result_temp_results_initial_search = create_dirs(args)

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
        q_value_filter=0.01,
    )

    peptide_df, dlc_calibration, dlc_transfer_learn, perc_95 = retrain_and_bouds(
        df_psms, result_dir=result_dir
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

    if write_initial_search_pickle:
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
        )
    if read_initial_search_pickle:
        (
            df_fragment,
            df_psms,
            df_fragment_max,
            df_fragment_max_peptide,
            config,
            dlc_transfer_learn,
            flags,
        ) = read_variables_from_pickles()

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
