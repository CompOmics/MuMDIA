"""Extracted ion chromatogram (XIC) and RT margin visualization utilities."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import polars as pl
from matplotlib.lines import Line2D


def plot_XIC(df: pl.DataFrame, output_dir: str = "results"):
    """
    Plots fragment_intensity vs rt for each unique fragment_name.
    Colors by fragment_name, lines connect fragments, marker shape by psm_id.
    Adds two separate legends: one for fragment_name (colors), one for psm_id (shapes).
    Works with a Polars DataFrame.
    """
    # Convert to pandas first
    pdf = df.to_pandas()

    precursor = (pdf["peptide"] + "_" + pdf["charge"].astype(str)).unique()[0]

    # Validate required columns
    required_cols = {"fragment_intensity", "rt", "fragment_name", "psm_id"}
    missing = required_cols - set(pdf.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Unique colors for fragment_name
    fragment_names = pdf["fragment_name"].unique()
    colors = plt.cm.get_cmap("tab20", len(fragment_names))
    fragment_color_map = {frag: colors(i) for i, frag in enumerate(fragment_names)}

    # Unique marker styles for psm_id (repeat if there are many)
    marker_styles = ["o", "s", "D", "^", "v", "p", "*", "X", "H", "<", ">"]
    psm_ids = pdf["psm_id"].unique()
    psm_marker_map = {
        psm: marker_styles[i % len(marker_styles)] for i, psm in enumerate(psm_ids)
    }

    plt.figure(figsize=(10, 6))

    # Step 1: Plot continuous lines by fragment_name
    for frag in fragment_names:
        frag_df = pdf[pdf["fragment_name"] == frag].sort_values("rt")
        plt.plot(
            frag_df["rt"],
            frag_df["fragment_intensity"],
            color=fragment_color_map[frag],
            linestyle="-",
            linewidth=1,
        )

    # Step 2: Overlay markers by fragment_name + psm_id
    for frag in fragment_names:
        frag_df = pdf[pdf["fragment_name"] == frag]
        for psm in psm_ids:
            psm_df = frag_df[frag_df["psm_id"] == psm]
            if psm_df.empty:
                continue
            plt.scatter(
                psm_df["rt"],
                psm_df["fragment_intensity"],
                color=fragment_color_map[frag],
                marker=psm_marker_map[psm],
                edgecolors="black",
                linewidths=0.5,
            )

    plt.xlabel("Retention Time (RT)")
    plt.ylabel("Fragment Intensity")
    plt.title("Extracted Ion Chromatogram by Fragment")

    # --- Create two legends manually ---
    # Legend 1: fragment_name (color lines)
    frag_legend_elements = [
        Line2D([0], [0], color=fragment_color_map[frag], lw=2, label=frag)
        for frag in fragment_names
    ]
    legend1 = plt.legend(
        handles=frag_legend_elements,
        title="Fragment",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.gca().add_artist(legend1)  # add first legend manually

    # Legend 2: psm_id (marker shapes)
    psm_legend_elements = [
        Line2D(
            [0],
            [0],
            marker=psm_marker_map[psm],
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label=str(psm),
        )
        for psm in psm_ids
    ]
    plt.legend(
        handles=psm_legend_elements,
        title="PSM ID",
        bbox_to_anchor=(1.05, 0),
        loc="lower left",
    )

    plt.title(f"{precursor}")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{precursor}_XIC.svg")


def plot_XIC_with_margins(df: pl.DataFrame, output_dir: str = "debug/calibration_xics", adapted_interval=None, min_interval=None, max_interval=None, apex_rt=None, cutoff=None):
    """
    Plots fragment_intensity vs rt for each unique fragment_name.
    Colors by fragment_name, lines connect fragments, marker shape by psm_id.
    Adds two separate legends: one for fragment_name (colors), one for psm_id (shapes).
    Works with a Polars DataFrame.
    In addition to the normal plot_XIC, this function adds vertical lines for the provided intervals:
    - adapted_interval: tuple (left, right) for the adapted margins
    - min_interval: tuple (left, right) for the minimum RT interval
    - max_interval: tuple (left, right) for the maximum RT interval
    - apex_rt: float for the apex retention time
    """
    # Convert to pandas first
    pdf = df.to_pandas()

    precursor = (pdf["peptide"] + "_" + pdf["charge"].astype(str)).unique()[0]

    # Validate required columns
    required_cols = {"fragment_intensity", "rt", "fragment_name", "psm_id"}
    missing = required_cols - set(pdf.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Unique colors for fragment_name
    fragment_names = pdf["fragment_name"].unique()
    colors = plt.cm.get_cmap("tab20", len(fragment_names))
    fragment_color_map = {frag: colors(i) for i, frag in enumerate(fragment_names)}

    # Unique marker styles for psm_id (repeat if there are many)
    marker_styles = ["o", "s", "D", "^", "v", "p", "*", "X", "H", "<", ">"]
    psm_ids = pdf["psm_id"].unique()
    psm_marker_map = {
        psm: marker_styles[i % len(marker_styles)] for i, psm in enumerate(psm_ids)
    }

    plt.figure(figsize=(10, 6))

    # Step 1: Plot continuous lines by fragment_name
    for frag in fragment_names:
        frag_df = pdf[pdf["fragment_name"] == frag].sort_values("rt")
        plt.plot(
            frag_df["rt"],
            frag_df["fragment_intensity"],
            color=fragment_color_map[frag],
            linestyle="-",
            linewidth=1,
        )

    # Step 2: Overlay markers by fragment_name + psm_id
    for frag in fragment_names:
        frag_df = pdf[pdf["fragment_name"] == frag]
        for psm in psm_ids:
            psm_df = frag_df[frag_df["psm_id"] == psm]
            if psm_df.empty:
                continue
            plt.scatter(
                psm_df["rt"],
                psm_df["fragment_intensity"],
                color=fragment_color_map[frag],
                marker=psm_marker_map[psm],
                edgecolors="black",
                linewidths=0.5,
            )

    plt.xlabel("Retention Time (RT)")
    plt.ylabel("Fragment Intensity")
    plt.title("Extracted Ion Chromatogram by Fragment")

    # --- Create two legends manually ---
    # Legend 1: fragment_name (color lines)
    frag_legend_elements = [
        Line2D([0], [0], color=fragment_color_map[frag], lw=2, label=frag)
        for frag in fragment_names
    ]
    legend1 = plt.legend(
        handles=frag_legend_elements,
        title="Fragment",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.gca().add_artist(legend1)  # add first legend manually

    # Legend 2: psm_id (marker shapes)
    psm_legend_elements = [
        Line2D(
            [0],
            [0],
            marker=psm_marker_map[psm],
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label=str(psm),
        )
        for psm in psm_ids
    ]


    # add vertical lines for intervals
    if adapted_interval:
        plt.axvline(x=adapted_interval[0], color='gray', linestyle='--', label='Adapted Interval Start')
        plt.axvline(x=adapted_interval[1], color='gray', linestyle='--', label='Adapted Interval End')

    if min_interval:
        plt.axvline(x=min_interval[0], color='blue', linestyle='--', label='Min Interval Start')
        plt.axvline(x=min_interval[1], color='blue', linestyle='--', label='Min Interval End')

    if max_interval:
        plt.axvline(x=max_interval[0], color='red', linestyle='--', label='Max Interval Start')
        plt.axvline(x=max_interval[1], color='red', linestyle='--', label='Max Interval End')

    if apex_rt:
        plt.axvline(x=apex_rt, color='green', linestyle='-', label='Apex RT')

    if cutoff:
        plt.axhline(y=cutoff, color='purple', linestyle='-.', label='Cutoff Intensity')

    plt.legend(
        handles=psm_legend_elements,
        title="PSM ID",
        bbox_to_anchor=(1.05, 0),
        loc="lower left",
    )

    plt.title(f"{precursor}")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{precursor}_XIC.svg")

def plot_rt_margin_histogram(rt_margins, output_dir: str = "debug/calibration_xics", min_diff=None, max_diff=None):
    """
    Plots a histogram of the rt_margins from the PSM DataFrame.
    Expects rt_margins to be a list of tuples (left_margin, right_margin).
    """
    if not rt_margins:
        raise ValueError("rt_margins list is empty.")

    plt.figure(figsize=(10, 6))
    plt.hist(rt_margins, bins=100, alpha=0.5, label='Margins', color='orange')

    if min_diff is not None:
        plt.axvline(x=min_diff, color='red', linestyle='--', label='Min RT Margin')
    if max_diff is not None:
        plt.axvline(x=max_diff, color='green', linestyle='--', label='Max RT Margin')

    plt.xlabel('Retention Time Margin')
    plt.ylabel('Frequency')
    plt.title('Histogram of Retention Time Margins')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rt_margin_histogram.svg")