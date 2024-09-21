import json
import pathlib
import subprocess
from typing import Any, Dict


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
