import os

os.environ["POLARS_MAX_THREADS"] = "1"

import argparse
import json

import subprocess
import subprocess
import mumdia
import pathlib
from pathlib import Path
import logging
from rich.logging import RichHandler
from rich.console import Console
import datetime
from pyteomics import parser, fasta
from parsers.parser_mzml import split_mzml_by_retention_time, get_ms1_mzml
from parsers.parser_parquet import parquet_reader
from MuMDIA.prediction_wrappers.wrapper_deeplc import (
    get_predictions_retentiontime,
    retrain_deeplc,
    predict_deeplc,
)
import pandas as pd
import polars as pl
import pickle
from matplotlib import pyplot as plt
from MuMDIA.utilities.logger import log_info

# Record the start time
start_time = datetime.datetime.now()

# Create a console for rich print
console = Console()

# Set up Rich logging configuration
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

import pyopenms as pms
import random

import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import random
from typing import Any, Dict
import pickle
import pandas as pd


def tryptic_digest_pyopenms(
    file_path,
    min_len=5,
    max_len=50,
    missed_cleavages=2,
    decoy_method="reverse",
    decoy_prefix="rev_",
    seq_types=["original", "decoy"],
):
    # Read the FASTA file
    fasta = pms.FASTAFile()
    entries = []
    fasta.load(file_path, entries)

    # Set up the enzyme digestion
    digestor = pms.ProteaseDigestion()
    digestor.setEnzyme("Trypsin")
    digestor.setMissedCleavages(missed_cleavages)

    peptides = []
    for entry in entries:
        # Process both original and decoy sequences
        for seq_type in seq_types:
            if seq_type == "original":
                protein_sequence = str(entry.sequence)
            else:
                if decoy_method == "reverse":
                    protein_sequence = str(entry.sequence)[::-1]
                elif decoy_method == "scramble":
                    seq_list = list(str(entry.sequence))
                    random.shuffle(seq_list)
                    protein_sequence = "".join(seq_list)
                else:
                    raise ValueError(
                        "Invalid decoy method. Choose 'reverse' or 'scramble'."
                    )

            protein_name = entry.identifier.split()[
                0
            ]  # Adjust based on your FASTA format

            # Perform the tryptic digest
            result = []
            digestor.digest(pms.AASequence.fromString(protein_sequence), result)

            for peptide in result:
                peptide_sequence = str(peptide.toString())
                len_pep_seq = len(peptide_sequence)
                start = protein_sequence.find(peptide_sequence)
                end = start + len_pep_seq
                if "X" in peptide_sequence:
                    continue
                if len_pep_seq >= min_len and len_pep_seq <= max_len:
                    if seq_type == "original":
                        peptides.append(
                            (
                                protein_name,
                                start,
                                end,
                                f"{protein_name}|{start}|{end}",
                                peptide_sequence,
                            )
                        )
                    else:
                        peptides.append(
                            (
                                f"{decoy_prefix}{protein_name}",
                                start,
                                end,
                                f"{decoy_prefix}{protein_name}|{start}|{end}",
                                peptide_sequence,
                            )
                        )

    return peptides


def write_to_fasta(df, output_file="vectorized_output.fasta"):
    # Combine 'id' and 'peptide' with a newline character
    fasta_series = ">" + df["id"] + "\n" + df["peptide"]

    # Join all rows with a newline character
    fasta_content = "\n".join(fasta_series)

    # Write the content to a file
    with open(output_file, "w") as fasta_file:
        fasta_file.write(fasta_content)


def run_sage(config, fasta_file, output_dir):
    # Get the "sage" values from the config
    # sage_values = config.get("sage", [])

    json_path = pathlib.Path(output_dir).joinpath("sage_values.json")

    # Write the sage values to a separate JSON file
    with open(json_path, "w") as file:
        json.dump(config, file, indent=4)

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
    # Call sage.exe with the path to the new JSON file as an argument
    #             "-f",
    #        fasta_file,
    subprocess.run(
        [
            "bin/sage",
            json_path,
            "-o",
            output_dir,
            "--annotate-matches",
            "--parquet",
            "--disable-telemetry-i-dont-want-to-improve-sage",
        ]
    )


def modify_config(key, value, config_file="config.json"):
    # Read the config.json file
    with open(config_file, "r") as file:
        config = json.load(file)

    # Modify the config based on the input
    config[key] = value

    # Write the modified config back to the file
    with open(config_file, "w") as file:
        json.dump(config, file, indent=4)

    log_info(f"Updated the config file with {key}: {value}")


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


def remove_intermediate_files(result_dir):
    log_info("Removing intermediate files...")
    # Remove the intermediate files
    intermediate_files = [
        "matched_fragments.sage.parquet",
        "results.sage.parquet",
        "sage_values.json",
    ]

    for file in intermediate_files:
        file_path = os.path.join(result_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        log_info("Directory already exists. Skipping creation")
        # Check for the presence of files and remove them if they exist

        # split up in cleanup function
        matched_fragments_file = os.path.join(
            directory_path, "matched_fragments.sage.parquet"
        )
        results_file = os.path.join(directory_path, "results.sage.parquet")

        if os.path.exists(matched_fragments_file):
            log_info("Removing existing parquet file: matched_fragments.sage.parquet")
            os.remove(matched_fragments_file)
        if os.path.exists(results_file):
            log_info("Removing existing parquet file: results.sage.parquet")
            os.remove(results_file)


def assign_identifiers(group):
    rounded_series = group["fragment_mz_experimental"].round(3)
    identifiers = rounded_series.rank(method="dense").cast(int)
    return group.with_columns(identifiers.alias("peak_identifier"))


def create_dirs(
    args,
    result_temp="temp",
    result_temp_results_initial_search="results_initial_search",
):
    result_dir = Path(args.result_dir)
    create_directory(result_dir.joinpath(result_temp))
    create_directory(
        result_dir.joinpath(result_temp, result_temp_results_initial_search)
    )
    return result_dir, result_temp, result_temp_results_initial_search


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


def write_variables_to_pickles(
    df_fragment,
    df_psms,
    df_fragment_max,
    df_fragment_max_peptide,
    config,
    dlc_transfer_learn,
    write_deeplc_pickle,
    write_ms2pip_pickle,
    write_correlation_pickles,
):
    with open("df_fragment.pkl", "wb") as f:
        pickle.dump(df_fragment, f)
    with open("df_psms.pkl", "wb") as f:
        pickle.dump(df_psms, f)
    with open("df_fragment_max.pkl", "wb") as f:
        pickle.dump(df_fragment_max, f)
    with open("df_fragment_max_peptide.pkl", "wb") as f:
        pickle.dump(df_fragment_max_peptide, f)
    with open("config.pkl", "wb") as f:
        pickle.dump(config, f)
    with open("dlc_transfer_learn.pkl", "wb") as f:
        pickle.dump(dlc_transfer_learn, f)
    # Also save the flags
    with open("flags.pkl", "wb") as f:
        pickle.dump(
            {
                "write_deeplc_pickle": write_deeplc_pickle,
                "write_ms2pip_pickle": write_ms2pip_pickle,
                "write_correlation_pickles": write_correlation_pickles,
                "read_deeplc_pickle": False,
                "read_ms2pip_pickle": False,
                "read_correlation_pickles": False,
            },
            f,
        )


def read_variables_from_pickles():
    with open("df_fragment.pkl", "rb") as f:
        df_fragment = pickle.load(f)
    with open("df_psms.pkl", "rb") as f:
        df_psms = pickle.load(f)
    with open("df_fragment_max.pkl", "rb") as f:
        df_fragment_max = pickle.load(f)
    with open("df_fragment_max_peptide.pkl", "rb") as f:
        df_fragment_max_peptide = pickle.load(f)
    with open("config.pkl", "rb") as f:
        config = pickle.load(f)
    with open("dlc_transfer_learn.pkl", "rb") as f:
        dlc_transfer_learn = pickle.load(f)
    with open("flags.pkl", "rb") as f:
        flags = pickle.load(f)
    return (
        df_fragment,
        df_psms,
        df_fragment_max,
        df_fragment_max_peptide,
        config,
        dlc_transfer_learn,
        flags,
    )


if __name__ == "__main__":
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

    # Now, write all variables to pickles before calling mumdia.main
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

    # Later, you can read the variables back
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
