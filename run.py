import argparse
import json
import os
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
from mzml_parser import split_mzml_by_retention_time, get_ms1_mzml
from parquet_parser import parquet_reader
from deeplc_wrapper import get_predictions_retentiontime, retrain_deeplc, predict_deeplc
import pandas as pd
import polars as pl
import pickle
from matplotlib import pyplot as plt

# os.environ["RAYON_NUM_THREADS"] = "64"
os.environ["POLARS_MAX_THREADS"] = "128"

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


def lasso_deconv():
    [
        "psm_id",
        "fragment_type",
        "fragment_ordinals",
        "fragment_charge",
        "fragment_mz_experimental",
        "fragment_mz_calculated",
        "fragment_intensity",
        "peptide",
        "charge",
        "rt",
        "scannr",
        "peak_identifier",
    ]

    [
        "psm_id",
        "filename",
        "scannr",
        "peptide",
        "stripped_peptide",
        "proteins",
        "num_proteins",
        "rank",
        "is_decoy",
        "expmass",
        "calcmass",
        "charge",
        "peptide_len",
        "missed_cleavages",
        "semi_enzymatic",
        "ms2_intensity",
        "isotope_error",
        "precursor_ppm",
        "fragment_ppm",
        "hyperscore",
        "delta_next",
        "delta_best",
        "rt",
        "aligned_rt",
        "predicted_rt",
        "delta_rt_model",
        "ion_mobility",
        "predicted_mobility",
        "delta_mobility",
        "matched_peaks",
        "longest_b",
        "longest_y",
        "longest_y_pct",
        "matched_intensity_pct",
        "scored_candidates",
        "poisson",
        "sage_discriminant_score",
        "posterior_error",
        "spectrum_q",
        "peptide_q",
        "protein_q",
        "reporter_ion_intensity",
        "fragment_intensity",
    ]

    # Parameters
    num_experimental_peaks = 50
    num_theoretical_spectra = 1500
    mz_range = (100, 600)
    intensity_range = (10, 100)

    # Generate experimental spectrum
    experimental_spectrum = generate_random_spectrum(
        num_experimental_peaks, mz_range, intensity_range
    )

    # Generate theoretical spectra with a random number of matched peaks
    theoretical_spectra = []
    for _ in range(num_theoretical_spectra):
        num_peaks = random.randint(5, num_experimental_peaks)
        theoretical_spectrum = generate_random_spectrum(
            num_peaks, mz_range, intensity_range
        )
        theoretical_spectra.append(theoretical_spectrum)

    # Convert spectra to numpy arrays for processing
    def spectrum_to_vector(spectrum, mz_values):
        mz_dict = dict(spectrum)
        return np.array([mz_dict.get(mz, 0) for mz in mz_values])

    # Get the set of all unique m/z values
    all_mz_values = sorted(set(mz for mz, _ in experimental_spectrum))

    # Convert experimental spectrum to vector
    exp_vector = spectrum_to_vector(experimental_spectrum, all_mz_values)

    # Convert theoretical spectra to matrix
    theoretical_matrix = np.array(
        [spectrum_to_vector(spec, all_mz_values) for spec in theoretical_spectra]
    ).T

    # Use Lasso to find the coefficients with L1 regularization
    lasso = Lasso(alpha=100.0, positive=True, max_iter=10000)
    lasso.fit(theoretical_matrix, exp_vector)
    coefficients = lasso.coef_

    # Print the results
    print("Coefficients:", coefficients)

    # Reconstruct the experimental spectrum from the theoretical spectra using the coefficients
    reconstructed_spectrum = np.dot(theoretical_matrix, coefficients)

    # Plot the experimental and reconstructed spectra for comparison
    plt.figure(figsize=(12, 6))
    plt.vlines(
        all_mz_values, 0, exp_vector, label="Experimental Spectrum", color="blue"
    )
    plt.vlines(
        all_mz_values,
        0,
        reconstructed_spectrum,
        label="Reconstructed Spectrum",
        color="red",
        linestyle="--",
    )
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title("Experimental vs Reconstructed Spectrum")
    plt.show()

    # Plot the stacked contribution of each theoretical spectrum
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(all_mz_values))
    colors = plt.cm.tab20(np.linspace(0, 1, num_theoretical_spectra))
    for j, vector in enumerate(theoretical_spectra):
        contribution = coefficients[j] * spectrum_to_vector(vector, all_mz_values)
        plt.bar(
            all_mz_values,
            contribution,
            bottom=bottom,
            color=colors[j],
            edgecolor="white",
            width=4,
            label=f"Theoretical Spectrum {j+1}",
        )
        bottom += contribution

    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    # plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title("Stacked Contributions of Theoretical Spectra")
    plt.show()


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


def log_info(message):
    current_time = datetime.datetime.now()
    elapsed = current_time - start_time
    # Add Rich markup for coloring and styling
    console.log(
        f"[green]{current_time:%Y-%m-%d %H:%M:%S}[/green] [bold blue]{message}[/bold blue] - Elapsed Time: [yellow]{elapsed}[/yellow]"
    )


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


def modify_config(key, value):
    # Read the config.json file
    with open("config.json", "r") as file:
        config = json.load(file)

    # Modify the config based on the input
    config[key] = value

    # Write the modified config back to the file
    with open("config.json", "w") as file:
        json.dump(config, file, indent=4)

    print("Config file updated successfully!")


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


if __name__ == "__main__":
    # Path to your FASTA file
    fasta_file = "fasta/ecoli_22032024.fasta"

    # Perform the tryptic digest and retrieve the list of peptides
    peptides = tryptic_digest_pyopenms(fasta_file)

    args = parse_arguments()
    result_dir = Path(args.result_dir)
    create_directory(result_dir.joinpath("temp"))
    create_directory(result_dir.joinpath("temp", "results_initial_search"))

    with open(args.config_file, "r") as file:
        config = json.load(file)
    """
    run_sage(
        config["sage_basic"],
        args.fasta_file,  # ecoli_22032024
        result_dir.joinpath("temp", "results_initial_search"),
    )

    df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide = parquet_reader(
        parquet_file_results=result_dir.joinpath(
            "temp", "results_initial_search", "results.sage.parquet"
        ),
        parquet_file_fragments=result_dir.joinpath(
            "temp", "results_initial_search", "matched_fragments.sage.parquet"
        ),
        q_value_filter=0.01,
    )

    ####################################
    "Seperate out the following !!!"
    ####################################

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

    ####################################
    "Seperate out the following !!!"
    ####################################

    ####################################
    "Seperate out the following !!!"
    ####################################

    mzml_dict = split_mzml_by_retention_time(
        "LFQ_Orbitrap_AIF_Ecoli_01.mzML",
        time_interval=perc_95,
        dir_files="results/temp/",
    )
    ####################################
    "Seperate out the following !!!"
    ####################################

    df_fragment_list = []
    df_psms_list = []
    df_fragment_max_list = []
    df_fragment_max_peptide_list = []

    psm_ident_start = 0

    for key, value in mzml_dict.items():
        bool_list = (
            (peptide_df["predictions_lower"] > (key - perc_95))
            & (peptide_df["predictions_lower"] < key)
        ) | (
            (peptide_df["predictions_upper"] < (key - perc_95))
            & (peptide_df["predictions_upper"] > key)
        )
        # TODO it needs to be introduced that the range is larger than the key - perc_95 and smaller than key + perc_95
        # | (
        #    (
        #        (peptide_df["predictions_lower"] < (key + perc_95))
        #        & (peptide_df["predictions_lower"] > key)
        #    )
        #    | (
        #        (peptide_df["predictions_upper"] > (key + perc_95))
        #        & (peptide_df["predictions_upper"] < key)
        #    )
        # )

        sub_peptide_df = peptide_df[bool_list]
        if len(sub_peptide_df.index) == 0:
            continue

        fasta_file = os.path.join(os.path.dirname(value), "vectorized_output.fasta")

        write_to_fasta(sub_peptide_df, output_file=fasta_file)

        with open("configs/config.json", "r") as file:
            config = json.load(file)

        sub_results = os.path.dirname(value)

        config["sage"]["database"]["fasta"] = fasta_file
        config["sage"]["mzml_paths"] = [value]

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
        df_fragment_max_list.append(df_fragment_max)
        df_fragment_max_peptide_list.append(df_fragment_max_peptide)

        psm_ident_start = df_psms["psm_id"].max() + 1

    df_fragment = pl.concat(df_fragment_list)
    df_psms = pl.concat(df_psms_list)

    if len(set(df_fragment["peptide"])) != len(set(df_psms["peptide"])):
        print(len(set(df_fragment["peptide"])), len(set(df_psms["peptide"])))

    # df_fragment_max = pl.concat(df_fragment_max_list)
    # df_fragment_max_peptide = pl.concat(df_fragment_max_peptide_list)

    # Will return random order of max fragment intensity
    df_fragment_max = df_fragment.sort("fragment_intensity", descending=True).unique(
        subset="psm_id", keep="first", maintain_order=True
    )

    if len(set(df_fragment_max["peptide"])) != len(set(df_psms["peptide"])):
        print(len(set(df_fragment_max["peptide"])), len(set(df_psms["peptide"])))

    # might also want to do on charge
    df_fragment_max_peptide = df_fragment_max.unique(
        subset=["peptide"],
        keep="first",  # , "charge"
    )

    if len(set(df_fragment_max_peptide["peptide"])) != len(set(df_psms["peptide"])):
        print(
            len(set(df_fragment_max_peptide["peptide"])), len(set(df_psms["peptide"]))
        )

    # print(df_fragment)
    # print(df_psms)
    # input()

    with open("df_fragment_first.pkl", "wb") as f:
        pickle.dump(df_fragment, f)
    with open("df_psms_first.pkl", "wb") as f:
        pickle.dump(df_psms, f)
    with open("df_fragment_max_first.pkl", "wb") as f:
        pickle.dump(df_fragment_max, f)
    with open("df_fragment_max_peptide_first.pkl", "wb") as f:
        pickle.dump(df_fragment_max_peptide, f)
    with open("dlc_calibration_first.pkl", "wb") as f:
        pickle.dump(dlc_calibration, f)
    with open("dlc_transfer_learn_first.pkl", "wb") as f:
        pickle.dump(dlc_transfer_learn, f)
    with open("perc_95_first.pkl", "wb") as f:
        pickle.dump(perc_95, f)
    """

    with open("configs/config.json", "r") as f:
        config = json.load(f)
    with open("df_fragment_first.pkl", "rb") as f:
        df_fragment = pickle.load(f)
    with open("df_psms_first.pkl", "rb") as f:
        df_psms = pickle.load(f)
    with open("df_fragment_max_first.pkl", "rb") as f:
        df_fragment_max = pickle.load(f)
    with open("df_fragment_max_peptide_first.pkl", "rb") as f:
        df_fragment_max_peptide = pickle.load(f)
    with open("dlc_calibration_first.pkl", "rb") as f:
        dlc_calibration = pickle.load(f)
    with open("dlc_transfer_learn_first.pkl", "rb") as f:
        dlc_transfer_learn = pickle.load(f)
    with open("perc_95_first.pkl", "rb") as f:
        perc_95 = pickle.load(f)

    log_info("Read the pickles...")

    df_fragment = df_fragment.join(
        df_psms.select(["psm_id", "scannr"]), on="psm_id", how="left"
    )

    log_info("Joined fragments with scannr...")

    # Group by 'scannr' and apply the function
    df_fragment = df_fragment.groupby("scannr").apply(assign_identifiers)

    log_info("Assigned identifiers to scannr...")
    """
    # Display the resulting DataFrame
    print(df_fragment)

    print(df_fragment.columns)
    input()
    print(df_psms.columns)
    input()
    """

    # Will return random order of max fragment intensity
    df_fragment_max = df_fragment.sort("fragment_intensity", descending=True).unique(
        subset="psm_id", keep="first", maintain_order=True
    )
    log_info("Fragment max...")

    # might also want to do on charge
    df_fragment_max_peptide = df_fragment_max.unique(
        subset=["peptide"],
        keep="first",  # , "charge"
    )
    log_info("Fragment max peptide...")

    mumdia.main(
        df_fragment=df_fragment,
        df_psms=df_psms,
        df_fragment_max=df_fragment_max,
        df_fragment_max_peptide=df_fragment_max_peptide,
        q_value_filter=1.0,
        config=config,
        deeplc_model=dlc_transfer_learn,
        # write_deeplc_pickle=True,
        # write_ms2pip_pickle=True,
        read_deeplc_pickle=True,
        read_ms2pip_pickle=True,
    )
