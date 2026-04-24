#!/usr/bin/env python3
"""Prepare and rerun Stage 2 in isolation using the custom Rust backend."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import polars as pl

from config import load_config_from_json
from parsers.parser_mzml import split_mzml_by_retention_time
from peptide_search.custom_engine import write_rust_stage2_partition_payload
from peptide_search.search_backend import run_targeted_search_backend
from prediction_wrappers.wrapper_deeplc import retrain_and_bounds
from run import _prepare_stage2_backend_context
from sequence.fasta import tryptic_digest_pyopenms
from utilities import pickling as pickling_utils
from utilities.logger import log_info

_INITIAL_PICKLE_NAMES = {
    "df_fragment_fname": "df_fragment_initial_search.pkl",
    "df_psms_fname": "df_psms_initial_search.pkl",
    "df_fragment_max_fname": "df_fragment_max_initial_search.pkl",
    "df_fragment_max_peptide_fname": "df_fragment_max_peptide_initial_search.pkl",
    "config_fname": "config_initial_search.pkl",
    "dlc_transfer_learn_fname": "dlc_transfer_learn_initial_search.pkl",
    "flags_fname": "flags_initial_search.pkl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Create a minimal Stage-2 bundle and per-partition Rust payloads.",
    )
    prepare.add_argument("--config", required=True, help="Path to MuMDIA config JSON.")
    prepare.add_argument(
        "--dump-dir",
        required=True,
        help="Directory where the Stage-2 bundle and Rust payloads will be written.",
    )

    run = subparsers.add_parser(
        "run",
        help="Run isolated Stage 2 from a previously prepared bundle.",
    )
    run.add_argument(
        "--dump-dir",
        required=True,
        help="Directory created by the prepare subcommand.",
    )
    run.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for isolated Stage-2 results. Defaults to <dump-dir>/stage2_output.",
    )

    return parser.parse_args()


def _read_initial_search_psms(result_dir: Path) -> pl.DataFrame:
    _, df_psms, _, _, _, _, _ = pickling_utils.read_variables_from_pickles(
        dir=result_dir,
        **_INITIAL_PICKLE_NAMES,
    )
    return df_psms


def _build_stage2_inputs(
    config_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, float, dict[float, str], dict[str, Any]]:
    config_obj = load_config_from_json(str(config_path))
    result_dir = Path(config_obj.result_dir)
    df_psms = _read_initial_search_psms(result_dir)

    log_info("Generating peptide library for isolated Stage 2...")
    peptides = tryptic_digest_pyopenms(config_obj.fasta_file)

    log_info("Training/refreshing DeepLC RT windows for isolated Stage 2...")
    peptide_df, _, _, rt_split_window = retrain_and_bounds(
        df_psms,
        peptides,
        result_dir=result_dir,
        coefficient_bounds=config_obj.rt_split_window_multiplier,
        percentile_exclude=config_obj.rt_split_percentile,
        fixed_rt_window_seconds=config_obj.rt_split_window_seconds,
        n_epochs=config_obj.deeplc_epochs_rt_window,
        min_peptidoform_occurrences=config_obj.deeplc_min_peptidoform_occurrences,
        calibration_only=config_obj.deeplc_use_calibration_only,
    )

    log_info("Splitting mzML for isolated Stage 2...")
    mzml_dict = split_mzml_by_retention_time(
        config_obj.mzml_file,
        time_interval=rt_split_window,
        dir_files=str(result_dir),
    )

    legacy_config = config_obj.to_legacy_format()
    mumdia_config = config_obj.get_mumdia_config()
    legacy_config["sage"]["custom_engine_max_candidates_per_spectrum"] = mumdia_config[
        "custom_engine_max_candidates_per_spectrum"
    ]

    backend_context = _prepare_stage2_backend_context(
        str(mumdia_config.get("targeted_search_engine", "sage")),
        peptide_df,
        legacy_config,
        mumdia_config,
        result_dir,
    )

    return legacy_config, peptide_df, rt_split_window, mzml_dict, backend_context


def _write_prepare_bundle(
    dump_dir: Path,
    config_path: Path,
    legacy_config: dict[str, Any],
    peptide_df: pd.DataFrame,
    rt_split_window: float,
    mzml_dict: dict[float, str],
    backend_context: dict[str, Any],
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "partitions").mkdir(parents=True, exist_ok=True)

    peptide_df.to_pickle(dump_dir / "peptide_df.pkl")
    with (dump_dir / "legacy_config.json").open("w", encoding="utf-8") as handle:
        json.dump(legacy_config, handle, indent=2)

    metadata = {
        "format_version": 1,
        "config_path": str(config_path),
        "rt_split_window": float(rt_split_window),
        "mzml_partitions": [
            {"upper_rt": float(upper_rt), "mzml_path": mzml_path}
            for upper_rt, mzml_path in mzml_dict.items()
        ],
    }

    ms2pip_predictions = backend_context.get("ms2pip_predictions")
    if ms2pip_predictions is not None:
        with (dump_dir / "stage2_ms2pip_predictions.pkl").open("wb") as handle:
            pickle.dump(ms2pip_predictions, handle)
        metadata["has_ms2pip_predictions"] = True
    else:
        metadata["has_ms2pip_predictions"] = False

    mumdia_config = legacy_config.get("mumdia", {})
    partition_files: list[dict[str, Any]] = []
    for partition_index, (upper_rt, mzml_path) in enumerate(mzml_dict.items()):
        peptide_selection_mask = np.maximum(
            peptide_df["predictions_lower"], upper_rt - rt_split_window
        ) <= np.minimum(peptide_df["predictions_upper"], upper_rt)
        sub_peptide_df = peptide_df[peptide_selection_mask].copy()
        if sub_peptide_df.empty:
            continue

        payload_path = dump_dir / "partitions" / f"partition_{partition_index:04d}.json"
        written_path = write_rust_stage2_partition_payload(
            payload_path,
            mzml_path,
            sub_peptide_df,
            legacy_config["sage"],
            mumdia_config,
            ms2pip_predictions=ms2pip_predictions,
        )
        partition_files.append(
            {
                "partition_index": partition_index,
                "upper_rt": float(upper_rt),
                "mzml_path": mzml_path,
                "payload_path": str(written_path) if written_path is not None else None,
                "peptide_rows": int(len(sub_peptide_df)),
            }
        )

    metadata["partition_files"] = partition_files
    with (dump_dir / "stage2_bundle.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    log_info(f"Wrote isolated Stage-2 bundle to {dump_dir}")


def prepare_bundle(config_path: Path, dump_dir: Path) -> None:
    legacy_config, peptide_df, rt_split_window, mzml_dict, backend_context = (
        _build_stage2_inputs(config_path)
    )
    _write_prepare_bundle(
        dump_dir,
        config_path,
        legacy_config,
        peptide_df,
        rt_split_window,
        mzml_dict,
        backend_context,
    )


def run_from_bundle(dump_dir: Path, output_dir: Path | None) -> None:
    with (dump_dir / "stage2_bundle.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (dump_dir / "legacy_config.json").open("r", encoding="utf-8") as handle:
        legacy_config = json.load(handle)

    peptide_df = pd.read_pickle(dump_dir / "peptide_df.pkl")
    mzml_dict = {
        float(entry["upper_rt"]): str(entry["mzml_path"])
        for entry in metadata["mzml_partitions"]
    }
    rt_split_window = float(metadata["rt_split_window"])

    backend_context: dict[str, Any] = {}
    ms2pip_path = dump_dir / "stage2_ms2pip_predictions.pkl"
    if ms2pip_path.exists():
        with ms2pip_path.open("rb") as handle:
            backend_context["ms2pip_predictions"] = pickle.load(handle)

    log_info("Running isolated Stage 2 from prepared bundle...")
    df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide = (
        run_targeted_search_backend(
            "custom",
            mzml_dict,
            peptide_df,
            legacy_config,
            rt_split_window,
            backend_context=backend_context,
        )
    )

    output_dir = output_dir or (dump_dir / "stage2_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_fragment.write_csv(output_dir / "df_fragment.tsv", separator="\t")
    df_psms.write_csv(output_dir / "df_psms.tsv", separator="\t")
    df_fragment_max.write_csv(output_dir / "df_fragment_max.tsv", separator="\t")
    df_fragment_max_peptide.write_csv(
        output_dir / "df_fragment_max_peptide.tsv", separator="\t"
    )

    summary = {
        "psm_rows": int(df_psms.height),
        "fragment_rows": int(df_fragment.height),
        "fragment_max_rows": int(df_fragment_max.height),
        "fragment_max_peptide_rows": int(df_fragment_max_peptide.height),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    log_info(f"Wrote isolated Stage-2 outputs to {output_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_bundle(Path(args.config).resolve(), Path(args.dump_dir).resolve())
        return

    if args.command == "run":
        output_dir = Path(args.output_dir).resolve() if args.output_dir else None
        run_from_bundle(Path(args.dump_dir).resolve(), output_dir)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
