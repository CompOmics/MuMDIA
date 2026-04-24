#!/usr/bin/env python3
"""Plot target counts versus similarity cutoffs for multiple q-value thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q10-table",
        default=(
            "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
            "margin_summed_ms2pip_analysis_q10_targets_decoys/"
            "margin_summed_ms2pip_correlations.tsv"
        ),
        help="Correlation summary table for q<=10% targets/decoys.",
    )
    parser.add_argument(
        "--q25-table",
        default=(
            "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
            "margin_summed_ms2pip_analysis_q25_targets_decoys/"
            "margin_summed_ms2pip_correlations.tsv"
        ),
        help="Correlation summary table for q<=25% targets/decoys.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
            "similarity_cutoff_comparison"
        ),
        help="Directory for the comparison plot and summary TSV.",
    )
    return parser.parse_args()


def build_counts(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cutoffs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows: list[dict[str, object]] = []
    for candidate_group in ["target", "decoy"]:
        subset = df[df["candidate_group"] == candidate_group].copy()
        for metric in ["cosine_similarity", "pearson_r", "spearman_r"]:
            values = pd.to_numeric(subset[metric], errors="coerce")
            for cutoff in cutoffs:
                rows.append(
                    {
                        "q_label": label,
                        "candidate_group": candidate_group,
                        "metric": metric,
                        "cutoff": cutoff,
                        "count_ge_cutoff": int((values >= cutoff).sum()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q10_df = pd.read_csv(args.q10_table, sep="\t", low_memory=False)
    q25_df = pd.read_csv(args.q25_table, sep="\t", low_memory=False)

    counts_df = pd.concat(
        [build_counts(q10_df, "q<=10%"), build_counts(q25_df, "q<=25%")],
        ignore_index=True,
    )
    counts_path = output_dir / "target_decoy_counts_by_similarity_cutoff_q10_vs_q25.tsv"
    counts_df.to_csv(counts_path, sep="\t", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    metric_titles = {
        "cosine_similarity": "Cosine similarity",
        "pearson_r": "Pearson r",
        "spearman_r": "Spearman r",
    }
    color_map = {"q<=10%": "#1f77b4", "q<=25%": "#d62728"}
    row_groups = [("target", "Target count"), ("decoy", "Decoy count")]

    for row_idx, (candidate_group, y_label) in enumerate(row_groups):
        for col_idx, metric in enumerate(
            ["cosine_similarity", "pearson_r", "spearman_r"]
        ):
            ax = axes[row_idx, col_idx]
            metric_df = counts_df[
                (counts_df["metric"] == metric)
                & (counts_df["candidate_group"] == candidate_group)
            ].copy()
            for q_label in ["q<=10%", "q<=25%"]:
                sub = metric_df[metric_df["q_label"] == q_label].sort_values("cutoff")
                ax.plot(
                    sub["cutoff"],
                    sub["count_ge_cutoff"],
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    color=color_map[q_label],
                    label=q_label,
                )
            if row_idx == 0:
                ax.set_title(metric_titles[metric])
            ax.set_xlabel("similarity cutoff")
            ax.set_ylabel(y_label)
            ax.grid(alpha=0.25)
            ax.set_xlim(-0.02, 0.92)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        "Target and decoy counts above similarity cutoffs: q<=10% vs q<=25%",
        y=1.02,
    )

    plot_path = output_dir / "target_decoy_count_cutoff_comparison_q10_vs_q25.png"
    fig.savefig(str(plot_path), dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote summary TSV: {counts_path}")
    print(f"Wrote comparison plot: {plot_path}")


if __name__ == "__main__":
    main()