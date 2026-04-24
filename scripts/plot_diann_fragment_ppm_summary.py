#!/usr/bin/env python3
"""Plot DIA-NN fragment mass-accuracy summary from the available stats/log outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-tsv",
        default="diann_results/report_with_xic.stats.tsv",
        help="DIA-NN stats TSV containing median MS2 mass-accuracy fields.",
    )
    parser.add_argument(
        "--log-file",
        default="diann_results/report_with_xic.log.txt",
        help="DIA-NN log file used to recover the configured MS2 mass tolerance.",
    )
    parser.add_argument(
        "--output",
        default="diann_results/report_with_xic_ms2_ppm_summary.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def parse_log_metrics(log_path: Path) -> dict[str, float | None]:
    configured_ms2_ppm = None
    fixed_ms2_ppm = None

    if not log_path.exists():
        return {
            "configured_ms2_ppm": configured_ms2_ppm,
            "fixed_ms2_ppm": fixed_ms2_ppm,
        }

    text = log_path.read_text(encoding="utf-8", errors="replace")

    configured_match = re.search(r"--mass-acc\s+([0-9.]+)", text)
    if configured_match:
        configured_ms2_ppm = float(configured_match.group(1))

    fixed_match = re.search(
        r"Mass accuracy will be fixed to\s+([0-9.eE+-]+)\s+\(MS2\)", text
    )
    if fixed_match:
        fixed_ms2_ppm = float(fixed_match.group(1)) * 1e6

    return {
        "configured_ms2_ppm": configured_ms2_ppm,
        "fixed_ms2_ppm": fixed_ms2_ppm,
    }


def main() -> None:
    args = parse_args()
    stats_path = Path(args.stats_tsv)
    log_path = Path(args.log_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = pd.read_csv(stats_path, sep="\t")
    if stats.empty:
        raise ValueError(f"Stats file is empty: {stats_path}")

    row = stats.iloc[0]
    log_metrics = parse_log_metrics(log_path)

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    for label, key, color in [
        ("Configured MS2 tolerance", "configured_ms2_ppm", "#9ecae1"),
        ("Fixed MS2 tolerance", "fixed_ms2_ppm", "#6baed6"),
        ("Median MS2 error", "Median.Mass.Acc.MS2", "#3182bd"),
        ("Median corrected MS2 error", "Median.Mass.Acc.MS2.Corrected", "#08519c"),
    ]:
        if key in row.index:
            value = float(row[key])
        else:
            value = log_metrics.get(key)  # type: ignore[arg-type]
        if value is None:
            continue
        labels.append(label)
        values.append(float(value))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("ppm")
    ax.set_title("DIA-NN fragment mass accuracy (MS2)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    note = (
        "Available DIA-NN outputs here store summary MS2 mass accuracy, not per-fragment deltas.\n"
        "So this figure shows configured/fixed tolerances and the reported median MS2 ppm error."
    )
    fig.text(0.02, 0.01, note, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Wrote {output_path}")
    for label, value in zip(labels, values):
        print(f"{label}: {value:.6f} ppm")


if __name__ == "__main__":
    main()
