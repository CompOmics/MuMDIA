#!/usr/bin/env python3
"""Replay the late Stage 3 feature calculation from a persisted dump."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Any

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_structures import PickleConfig, SpectraData
from mumdia import run_late_stage3_feature_calculation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-dir",
        required=True,
        help="Directory containing the late Stage 3 feature dump.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional override for config['mumdia']['result_dir'] during replay.",
    )
    parser.add_argument(
        "--read-correlation-pickles",
        action="store_true",
        help="Reuse existing correlation pickles during replay.",
    )
    parser.add_argument(
        "--write-correlation-pickles",
        action="store_true",
        help="Write correlation pickles during replay.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")
    return pl.read_csv(path, separator="\t")


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required pickle not found: {path}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def main() -> None:
    args = parse_args()
    dump_dir = Path(args.dump_dir)
    if not dump_dir.exists():
        raise FileNotFoundError(f"Dump directory not found: {dump_dir}")

    df_psms = load_table(dump_dir / "df_psms_pre_feature_calc.tsv")
    df_fragment = load_table(dump_dir / "df_fragment_pre_feature_calc.tsv")
    feature_fragment_df = load_table(
        dump_dir / "feature_fragment_df_pre_feature_calc.tsv"
    )
    df_fragment_max_peptide = load_table(
        dump_dir / "df_fragment_max_peptide_pre_feature_calc.tsv"
    )

    ms2pip_predictions = load_pickle(dump_dir / "ms2pip_predictions.pkl")
    theoretical_fragment_context = load_pickle(
        dump_dir / "theoretical_fragment_context.pkl"
    )
    preannotated_fragment_dict = load_pickle(
        dump_dir / "preannotated_fragment_dict.pkl"
    )
    config = load_pickle(dump_dir / "config.pkl")

    if args.output_dir:
        config = dict(config)
        config["mumdia"] = dict(config.get("mumdia", {}))
        config["mumdia"]["result_dir"] = args.output_dir

    spectra_data = SpectraData(
        ms1_dict=load_pickle(dump_dir / "ms1_dict.pkl"),
        ms2_to_ms1_dict=load_pickle(dump_dir / "ms2_to_ms1_dict.pkl"),
        ms2_dict=load_pickle(dump_dir / "ms2_dict.pkl"),
    )
    pickle_config = PickleConfig(
        read_correlation=args.read_correlation_pickles,
        write_correlation=args.write_correlation_pickles,
    )

    df_fragment_out, df_psms_out = run_late_stage3_feature_calculation(
        df_psms=df_psms,
        df_fragment=df_fragment,
        feature_fragment_df=feature_fragment_df,
        df_fragment_max_peptide=df_fragment_max_peptide,
        ms2pip_predictions=ms2pip_predictions,
        theoretical_fragment_context=theoretical_fragment_context,
        preannotated_fragment_dict=preannotated_fragment_dict,
        config=config,
        pickle_config=pickle_config,
        spectra_data=spectra_data,
    )

    summary = {
        "dump_dir": str(dump_dir),
        "output_dir": config.get("mumdia", {}).get("result_dir"),
        "df_fragment_shape": list(df_fragment_out.shape),
        "df_psms_shape": list(df_psms_out.shape),
    }
    (
        Path(config["mumdia"]["result_dir"]) / "stage3_feature_replay_summary.json"
    ).write_text(json.dumps(summary, indent=2))

    print(f"Loaded late Stage 3 feature dump from {dump_dir}")
    print(f"Wrote replay outputs to {config['mumdia']['result_dir']}")


if __name__ == "__main__":
    main()
