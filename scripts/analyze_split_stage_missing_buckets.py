#!/usr/bin/env python3
"""Bucket DIA-NN precursors by split-stage recovery status in MuMDIA."""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

PARTITION_RE = re.compile(r"part_([0-9.]+)_([0-9.]+)")
MOD_RE = re.compile(r"\[.*?\]")


def strip_modifications(peptide: str) -> str:
    return MOD_RE.sub("", str(peptide))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/config_ecoli_rerun_20260402_py312_rust_deeplc4_60s.json",
        help="Config JSON used for the current rerun.",
    )
    parser.add_argument(
        "--split-psms",
        default="debug/df_psms_after_retention_window_searches.tsv",
        help="Targeted split-stage PSM table.",
    )
    parser.add_argument(
        "--diann-report",
        default="diann_results/report_full.tsv",
        help="DIA-NN report TSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
            "split_stage_missing_buckets"
        ),
        help="Directory for bucket summaries.",
    )
    return parser.parse_args()


def load_partition_intervals(result_dir: Path, relevant_peptides: set[str]) -> dict[str, list[tuple[float, float]]]:
    seq_to_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    fasta_paths = glob.glob(str(result_dir / "temp" / "part_*" / "vectorized_output.fasta"))
    for fasta_path in fasta_paths:
        part_name = Path(fasta_path).parent.name
        match = PARTITION_RE.search(part_name)
        if not match:
            continue
        start = float(match.group(1))
        end = float(match.group(2))
        seen_here: set[str] = set()
        with open(fasta_path) as handle:
            for line in handle:
                line = line.strip()
                if (
                    line
                    and not line.startswith(">")
                    and line in relevant_peptides
                    and line not in seen_here
                ):
                    seq_to_intervals[line].append((start, end))
                    seen_here.add(line)
    return seq_to_intervals


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    result_dir = Path(config["result_dir"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(args.split_psms, sep="\t", low_memory=False)
    split_df = split_df[
        ~split_df["is_decoy"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    split_df["charge"] = pd.to_numeric(split_df["charge"], errors="coerce")
    split_df = split_df.dropna(subset=["stripped_peptide", "charge"]).copy()
    split_df["charge"] = split_df["charge"].astype(int)
    split_df["stripped_peptide"] = split_df["stripped_peptide"].map(strip_modifications)
    split_precursors = set(zip(split_df["stripped_peptide"], split_df["charge"]))

    diann_df = pd.read_csv(args.diann_report, sep="\t", low_memory=False)
    if "Decoy" in diann_df.columns:
        diann_df = diann_df[diann_df["Decoy"] == 0].copy()
    for column in ["Precursor.Charge", "RT", "Q.Value", "CScore", "Spectrum.Similarity"]:
        if column in diann_df.columns:
            diann_df[column] = pd.to_numeric(diann_df[column], errors="coerce")
    diann_df = diann_df.dropna(subset=["Stripped.Sequence", "Precursor.Charge", "RT"]).copy()
    diann_df["Precursor.Charge"] = diann_df["Precursor.Charge"].astype(int)
    diann_df["stripped_peptide"] = diann_df["Stripped.Sequence"].map(strip_modifications)
    rep_df = diann_df.sort_values("Q.Value").drop_duplicates(
        ["stripped_peptide", "Precursor.Charge"]
    )
    rep_df["rt_seconds"] = rep_df["RT"] * 60.0
    rep_df["modified"] = rep_df["Modified.Sequence"] != rep_df["Stripped.Sequence"]

    partition_intervals = load_partition_intervals(
        result_dir, set(rep_df["stripped_peptide"])
    )

    bucket_rows: list[dict[str, object]] = []
    for _, row in rep_df.iterrows():
        peptide = str(row["stripped_peptide"])
        charge = int(row["Precursor.Charge"])
        rt_seconds = float(row["rt_seconds"])
        key = (peptide, charge)
        intervals = partition_intervals.get(peptide, [])
        in_correct_partition = any(start <= rt_seconds <= end for start, end in intervals)
        nearest_distance = None
        if intervals:
            nearest_distance = min(
                0.0
                if start <= rt_seconds <= end
                else min(abs(rt_seconds - start), abs(rt_seconds - end))
                for start, end in intervals
            )

        if key in split_precursors:
            bucket = "recovered_by_split_stage"
        elif in_correct_partition:
            bucket = "missing_despite_correct_rt_partition"
        else:
            bucket = "missing_due_rt_partition_mismatch"

        bucket_rows.append(
            {
                "bucket": bucket,
                "stripped_peptide": peptide,
                "charge": charge,
                "Modified.Sequence": row.get("Modified.Sequence"),
                "Q.Value": row.get("Q.Value"),
                "CScore": row.get("CScore"),
                "Spectrum.Similarity": row.get("Spectrum.Similarity"),
                "RT": row.get("RT"),
                "Predicted.RT": row.get("Predicted.RT"),
                "rt_seconds": rt_seconds,
                "modified": bool(row["modified"]),
                "in_correct_partition": in_correct_partition,
                "nearest_partition_distance_sec": nearest_distance,
                "num_partitions_containing_peptide": len(intervals),
            }
        )

    bucket_df = pd.DataFrame(bucket_rows)
    bucket_df.to_csv(output_dir / "diann_precursor_buckets.tsv", sep="\t", index=False)

    summary_df = (
        bucket_df.groupby("bucket", dropna=False)
        .agg(
            n_precursors=("stripped_peptide", "size"),
            modified_count=("modified", "sum"),
            median_q=("Q.Value", "median"),
            median_cscore=("CScore", "median"),
            median_spectrum_similarity=("Spectrum.Similarity", "median"),
            median_nearest_partition_distance_sec=(
                "nearest_partition_distance_sec",
                "median",
            ),
        )
        .reset_index()
    )
    total = int(summary_df["n_precursors"].sum())
    summary_df["pct_of_diann_precursors"] = summary_df["n_precursors"] / total * 100.0
    summary_df["modified_pct"] = summary_df["modified_count"] / summary_df["n_precursors"] * 100.0
    summary_df = summary_df.sort_values("n_precursors", ascending=False)
    summary_df.to_csv(output_dir / "bucket_summary.tsv", sep="\t", index=False)

    examples_df = (
        bucket_df.sort_values(["bucket", "Q.Value"], na_position="last")
        .groupby("bucket", group_keys=False)
        .head(25)
    )
    examples_df.to_csv(output_dir / "bucket_examples_top25.tsv", sep="\t", index=False)

    print(summary_df.to_string(index=False))
    print(f"Wrote bucket details: {output_dir / 'diann_precursor_buckets.tsv'}")
    print(f"Wrote bucket summary: {output_dir / 'bucket_summary.tsv'}")
    print(f"Wrote examples: {output_dir / 'bucket_examples_top25.tsv'}")


if __name__ == "__main__":
    main()
