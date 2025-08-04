import os
from pathlib import Path

from utilities.logger import log_info


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


def assign_identifiers(group):
    rounded_series = group["fragment_mz_experimental"].round(3)
    identifiers = rounded_series.rank(method="dense").cast(int)
    return group.with_columns(identifiers.alias("peak_identifier"))
