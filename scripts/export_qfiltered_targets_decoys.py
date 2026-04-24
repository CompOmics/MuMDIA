#!/usr/bin/env python3
"""Export target and decoy PSM candidates with an optional mokapot q-value filter."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-psms",
        default=(
            "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
            "high_scoring_decoys/decoys.mokapot.psms.txt"
        ),
        help="Mokapot target PSM export.",
    )
    parser.add_argument(
        "--decoy-psms",
        default=(
            "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
            "high_scoring_decoys/decoys.mokapot.decoy.psms.txt"
        ),
        help="Mokapot decoy PSM export.",
    )
    parser.add_argument(
        "--pin-file",
        default="results_ecoli_rerun_20260402_py312_rust_deeplc4_60s/outfile.pin",
        help="PIN file used to recover charge and SpecId.",
    )
    parser.add_argument(
        "--q-threshold",
        type=float,
        default=None,
        help="Optional mokapot q-value upper bound. Omit to keep all PSMs.",
    )
    parser.add_argument(
        "--output-path",
        default=(
            "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
            "high_scoring_decoys/targets_decoys_q0.10.tsv"
        ),
        help="Combined output TSV path.",
    )
    return parser.parse_args()


def load_with_charge(
    path: Path,
    pin_df: pd.DataFrame,
    label_group: str,
    q_threshold: float | None,
) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["mokapot q-value"] = pd.to_numeric(df["mokapot q-value"], errors="coerce")
    if q_threshold is not None:
        df = df[df["mokapot q-value"] <= q_threshold].copy()
    df = df.merge(
        pin_df[["ScanNr", "filename", "Peptide", "charge", "SpecId"]],
        on=["ScanNr", "filename", "Peptide"],
        how="left",
    )
    df = df.dropna(subset=["charge"]).copy()
    df["charge"] = pd.to_numeric(df["charge"], errors="coerce")
    df = df.dropna(subset=["charge"]).copy()
    df["charge"] = df["charge"].round().astype(int)
    df["label_group"] = label_group
    df["is_target"] = label_group == "target"
    return df


def main() -> None:
    args = parse_args()
    pin_df = pd.read_csv(
        args.pin_file,
        sep="\t",
        low_memory=False,
        usecols=["ScanNr", "filename", "Peptide", "charge", "SpecId"],
    )

    target_df = load_with_charge(
        Path(args.target_psms), pin_df, "target", args.q_threshold
    )
    decoy_df = load_with_charge(
        Path(args.decoy_psms), pin_df, "decoy", args.q_threshold
    )
    combined_df = pd.concat([target_df, decoy_df], ignore_index=True)
    combined_df = combined_df.sort_values(
        ["label_group", "mokapot q-value", "mokapot score"],
        ascending=[True, True, False],
        na_position="last",
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, sep="\t", index=False)

    if args.q_threshold is None:
        print(f"Targets (all q-values): {len(target_df)}")
        print(f"Decoys (all q-values): {len(decoy_df)}")
    else:
        print(f"Targets <= {args.q_threshold}: {len(target_df)}")
        print(f"Decoys <= {args.q_threshold}: {len(decoy_df)}")
    print(f"Combined TSV: {output_path}")


if __name__ == "__main__":
    main()