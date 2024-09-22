import json
import pathlib
import subprocess
from typing import Any, Dict
import numpy as np
import polars as pl
import os
from sequence.fasta import write_to_fasta
from parsers.parser_parquet import parquet_reader


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

    df_fragment_max = df_fragment.sort("fragment_intensity", descending=True).unique(
        subset="psm_id", keep="first", maintain_order=True
    )

    # might also want to do on charge
    df_fragment_max_peptide = df_fragment_max.unique(
        subset=["peptide"],
        keep="first",
    )

    # Assert and raise error if df_fragment and df_psms do not have the same number of unique peptides
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
