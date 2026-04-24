#!/usr/bin/env python3
"""Plot target and decoy counts above similarity cutoffs for one analysis table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT_TABLE = (
    "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
    "margin_summed_ms2pip_analysis_all_targets_decoys/"
    "margin_summed_ms2pip_correlations.tsv"
)
DEFAULT_OUTPUT_DIR = (
    "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
    "similarity_cutoff_comparison_all"
)
METRICS = ["cosine_similarity", "pearson_r", "spearman_r"]
METRIC_TITLES = {
    "cosine_similarity": "Cosine similarity",
    "pearson_r": "Pearson r",
    "spearman_r": "Spearman r",
}
COLOR_MAP = {"target": "#1f77b4", "decoy": "#d62728"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-table",
        default=DEFAULT_INPUT_TABLE,
        help="Correlation summary table containing candidate_group and similarity metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the plot and count summary TSV.",
    )
    parser.add_argument(
        "--label",
        default="all q-values",
        help="Label shown in the plot title.",
    )
    return parser.parse_args()


def build_counts(df: pd.DataFrame) -> pd.DataFrame:
    cutoffs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows: list[dict[str, object]] = []
    for candidate_group in ["target", "decoy"]:
        subset = df[df["candidate_group"] == candidate_group].copy()
        for metric in METRICS:
            values = pd.to_numeric(subset[metric], errors="coerce")
            for cutoff in cutoffs:
                rows.append(
                    {
                        "candidate_group": candidate_group,
                        "metric": metric,
                        "cutoff": cutoff,
                        "count_ge_cutoff": int((values >= cutoff).sum()),
                    }
                )
    return pd.DataFrame(rows)


def build_ratios(counts_df: pd.DataFrame) -> pd.DataFrame:
    pivot_df = (
        counts_df.pivot_table(
            index=["metric", "cutoff"],
            columns="candidate_group",
            values="count_ge_cutoff",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    for column in ["target", "decoy"]:
        if column not in pivot_df.columns:
            pivot_df[column] = 0
    pivot_df["target_to_decoy_ratio"] = pivot_df.apply(
        lambda row: (
            float(row["target"]) / float(row["decoy"])
            if float(row["decoy"]) > 0.0
            else float("nan")
        ),
        axis=1,
    )
    return pivot_df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.input_table, sep="\t", low_memory=False)
    counts_df = build_counts(summary_df)
    counts_path = output_dir / "target_decoy_counts_by_similarity_cutoff_all_qvalues.tsv"
    counts_df.to_csv(counts_path, sep="\t", index=False)
    ratio_df = build_ratios(counts_df)
    ratio_path = output_dir / "target_decoy_ratio_by_similarity_cutoff_all_qvalues.tsv"
    ratio_df.to_csv(ratio_path, sep="\t", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        metric_df = counts_df[counts_df["metric"] == metric].copy()
        for candidate_group in ["target", "decoy"]:
            sub = metric_df[
                metric_df["candidate_group"] == candidate_group
            ].sort_values("cutoff")
            ax.plot(
                sub["cutoff"],
                sub["count_ge_cutoff"],
                marker="o",
                linewidth=2,
                markersize=4,
                color=COLOR_MAP[candidate_group],
                label=candidate_group,
            )
        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("similarity cutoff")
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)
        ax.set_xlim(-0.02, 0.92)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        f"Target and decoy counts above similarity cutoffs: {args.label}",
        y=1.03,
    )

    plot_path = output_dir / "target_decoy_count_cutoff_comparison_all_qvalues.png"
    fig.savefig(str(plot_path), dpi=220, bbox_inches="tight")
    plt.close(fig)

    ratio_fig, ratio_axes = plt.subplots(
        1, 3, figsize=(15, 4.8), constrained_layout=True
    )
    for ax, metric in zip(ratio_axes, METRICS):
        sub = ratio_df[ratio_df["metric"] == metric].sort_values("cutoff")
        ax.plot(
            sub["cutoff"],
            sub["target_to_decoy_ratio"],
            marker="o",
            linewidth=2,
            markersize=4,
            color="#2ca02c",
        )
        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("similarity cutoff")
        ax.set_ylabel("target / decoy ratio")
        ax.grid(alpha=0.25)
        ax.set_xlim(-0.02, 0.92)

    ratio_fig.suptitle(
        f"Target-to-decoy ratio above similarity cutoffs: {args.label}",
        y=1.03,
    )
    ratio_plot_path = (
        output_dir / "target_decoy_ratio_cutoff_comparison_all_qvalues.png"
    )
    ratio_fig.savefig(str(ratio_plot_path), dpi=220, bbox_inches="tight")
    plt.close(ratio_fig)

    print(f"Wrote summary TSV: {counts_path}")
    print(f"Wrote comparison plot: {plot_path}")
    print(f"Wrote ratio TSV: {ratio_path}")
    print(f"Wrote ratio plot: {ratio_plot_path}")


if __name__ == "__main__":
    main()
