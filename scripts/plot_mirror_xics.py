#!/usr/bin/env python3
"""Generate mirror XIC plots for MuMDIA vs DIA-NN peptidoforms.

This script tries to be conservative about peptidoform matching:
- MuMDIA peptidoforms are taken from `merged_features.tsv`
- an exact DIA-NN `Precursor.Id` is reconstructed from the MuMDIA peptide string
- only exact precursor matches are treated as same-peptidoform comparisons

Outputs
-------
- candidate tables for all exact matches, suspicious cases, and top-overlap cases
- mirror plots in separate folders
- an index TSV linking each generated plot to summary metrics
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MS1_FEATURE_NAMES = {"ms1", "precursor", "ms1_precursor"}


def calculate_mumdia_rt_window(
    mu_df: pd.DataFrame,
    intensity_threshold: float = 0.05,
) -> tuple[float, float, float] | None:
    """Recompute MuMDIA-like RT bounds from the apex fragment trace."""
    if mu_df.empty:
        return None

    df_sorted = mu_df.sort_values("rt").reset_index(drop=True)
    apex_pos = int(df_sorted["fragment_intensity"].to_numpy().argmax())
    apex_row = df_sorted.iloc[apex_pos]
    apex_rt = float(apex_row["rt"])
    apex_intensity = float(apex_row["fragment_intensity"])
    if apex_intensity <= 0:
        return None

    cutoff = intensity_threshold * apex_intensity
    apex_feature = apex_row["feature"]
    apex_trace = (
        df_sorted[df_sorted["feature"] == apex_feature]
        .sort_values("rt")
        .reset_index(drop=True)
    )
    if apex_trace.empty:
        return None

    apex_trace_pos = int(apex_trace["fragment_intensity"].to_numpy().argmax())

    left_bound = apex_rt
    left_df = apex_trace.iloc[:apex_trace_pos].iloc[::-1]
    for _, row in left_df.iterrows():
        if float(row["fragment_intensity"]) < cutoff:
            left_bound = float(row["rt"])
            break
    if left_bound == apex_rt and not left_df.empty:
        left_bound = float(left_df.iloc[-1]["rt"])

    right_bound = apex_rt
    right_df = apex_trace.iloc[apex_trace_pos + 1 :]
    for _, row in right_df.iterrows():
        if float(row["fragment_intensity"]) < cutoff:
            right_bound = float(row["rt"])
            break
    if right_bound == apex_rt and not right_df.empty:
        right_bound = float(right_df.iloc[-1]["rt"])

    return left_bound, right_bound, apex_rt


def determine_plot_rt_limits(
    candidate: pd.Series,
    mu_df: pd.DataFrame,
    di_df: pd.DataFrame,
) -> tuple[float, float] | None:
    """Choose a tighter plotting window centered on MuMDIA's RT bounds."""
    if mu_df.empty and di_df.empty:
        return None

    combined_rt = pd.concat([mu_df["rt"], di_df["rt"]], ignore_index=True)
    combined_rt = combined_rt[np.isfinite(combined_rt)]
    if combined_rt.empty:
        return None

    global_min = float(combined_rt.min())
    global_max = float(combined_rt.max())

    bounds = calculate_mumdia_rt_window(mu_df)
    if bounds is not None:
        left_bound, right_bound, _ = bounds
        span = max(right_bound - left_bound, 0.02)
        padding = max(0.03, span * 0.20)
        x_min = left_bound - padding
        x_max = right_bound + padding
    else:
        center = float(candidate.get("rt_min", np.nan))
        if not np.isfinite(center):
            center = float(candidate.get("diann_rt", np.nan))
        if not np.isfinite(center):
            center = float((global_min + global_max) / 2.0)
        padding = 0.15
        x_min = center - padding
        x_max = center + padding

    for center in (candidate.get("rt_min", np.nan), candidate.get("diann_rt", np.nan)):
        if pd.notna(center):
            center = float(center)
            x_min = min(x_min, center - padding)
            x_max = max(x_max, center + padding)

    x_min = max(global_min, x_min)
    x_max = min(global_max, x_max)
    if x_max <= x_min:
        return None
    return x_min, x_max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-features",
        default="comparison_output_full/merged_features.tsv",
        help="MuMDIA-vs-DIA-NN merged feature table",
    )
    parser.add_argument(
        "--gap-peptides",
        default="comparison_output_full/gap_peptides.tsv",
        help="Gap/suspicious peptidoforms table",
    )
    parser.add_argument(
        "--df-fragment",
        default="results_full/df_fragment.tsv",
        help="MuMDIA fragment-level trace table",
    )
    parser.add_argument(
        "--diann-report",
        default="diann_results/report_with_xic.tsv",
        help="DIA-NN report with precursor IDs",
    )
    parser.add_argument(
        "--diann-xic",
        default="diann_results/report_with_xic_xic/LFQ_Orbitrap_AIF_Ecoli_01.xic.parquet",
        help="DIA-NN XIC parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_output_full/mirror_xics",
        help="Output directory",
    )
    parser.add_argument(
        "--top-overlap",
        type=int,
        default=200,
        help="Number of best non-suspicious exact matches to render",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=10,
        help="Maximum fragment traces to draw per plot",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500000,
        help="Chunk size for streaming MuMDIA df_fragment.tsv",
    )
    return parser.parse_args()


def strip_modifications(peptide: str) -> str:
    return re.sub(r"\[.*?\]", "", str(peptide))


def to_diann_precursor(peptide: str, charge: float | int) -> str:
    s = str(peptide)
    s = s.replace("[Carbamidomethyl]", "(UniMod:4)")
    s = s.replace("[Oxidation]", "(UniMod:35)")
    s = re.sub(r"\[(.*?)\]", lambda m: f"({m.group(1)})", s)
    return f"{s}{int(round(float(charge)))}"


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def is_ms1_feature(feature: str) -> bool:
    return str(feature).strip().lower() in MS1_FEATURE_NAMES


def canonicalize_feature_name(feature: str) -> str:
    """Return a normalized fragment label for cross-tool matching and coloring."""
    return str(feature).strip().lower()


def load_candidates(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = pd.read_csv(args.merged_features, sep="\t", low_memory=False)
    gap = pd.read_csv(args.gap_peptides, sep="\t", low_memory=False)
    diann_report = pd.read_csv(args.diann_report, sep="\t", low_memory=False)
    report_ids = set(diann_report["Precursor.Id"].dropna().astype(str))

    merged["target_precursor"] = [
        to_diann_precursor(peptide, charge)
        for peptide, charge in zip(merged["Peptide"], merged["charge"])
    ]
    merged["exact_match"] = merged["target_precursor"].isin(report_ids)
    merged["charge_int"] = merged["charge"].round().astype(int)
    merged["stripped_from_mumdia"] = merged["Peptide"].map(strip_modifications)
    exact = merged[merged["exact_match"]].copy()

    # Keep one best MuMDIA row per exact precursor.
    exact = exact.sort_values(
        by=["peptide_q_min", "diann_qvalue", "top_correlation_individual_1"],
        ascending=[True, True, False],
        na_position="last",
    ).drop_duplicates(subset=["target_precursor"], keep="first")

    gap_keys = set(
        zip(gap["stripped_peptide"].astype(str), gap["charge"].round().astype(int))
    )
    exact["gap_key"] = list(
        zip(exact["stripped_peptide"].astype(str), exact["charge_int"])
    )
    suspicious = exact[exact["gap_key"].isin(gap_keys)].copy()

    top_overlap = exact[
        ~exact["target_precursor"].isin(suspicious["target_precursor"])
    ].copy()
    top_overlap = top_overlap.sort_values(
        by=["peptide_q_min", "diann_qvalue", "top_correlation_individual_1"],
        ascending=[True, True, False],
        na_position="last",
    ).head(args.top_overlap)

    return exact, suspicious, top_overlap


def build_target_key_df(candidates: pd.DataFrame) -> pd.DataFrame:
    return (
        candidates[["Peptide", "charge_int"]]
        .drop_duplicates()
        .rename(columns={"Peptide": "peptide", "charge_int": "charge"})
    )


def stream_mumdia_fragments(
    fragment_path: Path, target_keys: pd.DataFrame, chunksize: int
) -> pd.DataFrame:
    usecols = [
        "fragment_type",
        "fragment_ordinals",
        "fragment_charge",
        "fragment_intensity",
        "peptide",
        "charge",
        "rt",
    ]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        fragment_path, sep="\t", usecols=usecols, chunksize=chunksize
    ):
        chunk["charge"] = chunk["charge"].round().astype(int)
        merged = chunk.merge(target_keys, on=["peptide", "charge"], how="inner")
        if not merged.empty:
            pieces.append(merged)
    if not pieces:
        return pd.DataFrame(columns=usecols + ["feature"])

    out = pd.concat(pieces, ignore_index=True)
    out["feature"] = (
        out["fragment_type"].astype(str)
        + out["fragment_ordinals"].astype(int).astype(str)
        + "^"
        + out["fragment_charge"].astype(int).astype(str)
    )
    return out


def pick_features(
    mu_df: pd.DataFrame, di_df: pd.DataFrame, top_features: int
) -> list[str]:
    mu_totals = (
        mu_df.groupby("feature_key")["fragment_intensity"]
        .sum()
        .sort_values(ascending=False)
    )
    di_totals = di_df.groupby("feature_key")["value"].sum().sort_values(ascending=False)

    selected: list[str] = []
    shared = [
        f for f in di_totals.index if f in mu_totals.index and not is_ms1_feature(f)
    ]
    for f in shared:
        if f not in selected:
            selected.append(f)
        if len(selected) >= min(top_features, max(4, len(shared))):
            break

    for series in (di_totals.index, mu_totals.index):
        for f in series:
            if is_ms1_feature(f):
                continue
            if f not in selected:
                selected.append(f)
            if len(selected) >= top_features:
                ms1_features = [
                    f
                    for f in di_totals.index
                    if f in mu_totals.index and is_ms1_feature(f)
                ]
                return selected + [f for f in ms1_features if f not in selected]
    ms1_features = [
        f for f in di_totals.index if f in mu_totals.index and is_ms1_feature(f)
    ]
    return selected + [f for f in ms1_features if f not in selected]


def add_relative_trace_column(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df[out_col] = []
        return df

    df[out_col] = df[value_col]
    ms1_mask = df[feature_col].map(is_ms1_feature)
    if not ms1_mask.any():
        return df

    ms1_max = float(df.loc[ms1_mask, value_col].max())
    if ms1_max > 0:
        df.loc[ms1_mask, out_col] = df.loc[ms1_mask, value_col] / ms1_max
    else:
        df.loc[ms1_mask, out_col] = 0.0
    return df


def plot_one(
    candidate: pd.Series,
    mu_all: pd.DataFrame,
    di_all: pd.DataFrame,
    out_path: Path,
    top_features: int,
) -> bool:
    peptide = candidate["Peptide"]
    charge = int(candidate["charge_int"])
    precursor = candidate["target_precursor"]

    mu_df = mu_all[(mu_all["peptide"] == peptide) & (mu_all["charge"] == charge)].copy()
    di_df = di_all[di_all["pr"] == precursor].copy()

    if mu_df.empty or di_df.empty:
        return False

    rt_limits = determine_plot_rt_limits(candidate, mu_df, di_df)
    if rt_limits is not None:
        x_min, x_max = rt_limits
        mu_df = mu_df[(mu_df["rt"] >= x_min) & (mu_df["rt"] <= x_max)].copy()
        di_df = di_df[(di_df["rt"] >= x_min) & (di_df["rt"] <= x_max)].copy()
        if mu_df.empty or di_df.empty:
            return False
    else:
        x_min = x_max = None

    chosen = pick_features(mu_df, di_df, top_features)
    if not chosen:
        return False

    mu_df = mu_df[mu_df["feature"].isin(chosen)].copy()
    di_df = di_df[di_df["feature"].isin(chosen)].copy()
    if mu_df.empty or di_df.empty:
        return False

    mu_max = float(mu_df["fragment_intensity"].max()) if not mu_df.empty else 1.0
    di_max = float(di_df["value"].max()) if not di_df.empty else 1.0
    mu_df["norm"] = mu_df["fragment_intensity"] / (mu_max if mu_max > 0 else 1.0)
    di_df["norm"] = di_df["value"] / (di_max if di_max > 0 else 1.0)
    mu_df = add_relative_trace_column(mu_df, "feature", "norm", "plot_norm")
    di_df = add_relative_trace_column(di_df, "feature", "norm", "plot_norm")
    mu_df["feature_key"] = mu_df["feature"].map(canonicalize_feature_name)
    di_df["feature_key"] = di_df["feature"].map(canonicalize_feature_name)

    shared_features = sorted(
        set(mu_df["feature_key"]).intersection(di_df["feature_key"])
    )
    colors = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(chosen), 1))
    ordered_feature_keys = []
    for feature in chosen:
        key = canonicalize_feature_name(feature)
        if key not in ordered_feature_keys:
            ordered_feature_keys.append(key)
    color_map = {
        feature_key: colors(i) for i, feature_key in enumerate(ordered_feature_keys)
    }

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.6], hspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    ax_meta = fig.add_subplot(gs[1, 0])

    # total traces
    di_total = di_df.groupby("rt", as_index=False)["norm"].sum().sort_values("rt")
    mu_total = mu_df.groupby("rt", as_index=False)["norm"].sum().sort_values("rt")
    di_apex_rt = np.nan
    mu_apex_rt = np.nan
    if not di_total.empty:
        di_total["norm"] = di_total["norm"] / max(di_total["norm"].max(), 1.0)
        di_apex_rt = float(di_total.loc[di_total["norm"].idxmax(), "rt"])
        ax.plot(
            di_total["rt"],
            di_total["norm"],
            color="black",
            linewidth=2.4,
            alpha=0.85,
            label="DIA-NN total",
            marker="o",
            markersize=2.6,
            markeredgewidth=0,
        )
    if not mu_total.empty:
        mu_total["norm"] = mu_total["norm"] / max(mu_total["norm"].max(), 1.0)
        mu_apex_rt = float(mu_total.loc[mu_total["norm"].idxmax(), "rt"])
        ax.plot(
            mu_total["rt"],
            -mu_total["norm"],
            color="dimgray",
            linewidth=2.4,
            alpha=0.85,
            label="MuMDIA total",
            marker="o",
            markersize=2.6,
            markeredgewidth=0,
        )

    for feature in chosen:
        feature_key = canonicalize_feature_name(feature)
        color = color_map[feature_key]
        di_sub = di_df[di_df["feature_key"] == feature_key].sort_values("rt")
        mu_sub = mu_df[mu_df["feature_key"] == feature_key].sort_values("rt")
        linewidth = 1.6 if is_ms1_feature(feature) else 1.2
        linestyle = ":" if is_ms1_feature(feature) else "-"
        mu_linestyle = (0, (4, 2)) if is_ms1_feature(feature) else "--"
        if not di_sub.empty:
            ax.plot(
                di_sub["rt"],
                di_sub["plot_norm"],
                color=color,
                alpha=0.85,
                linewidth=linewidth,
                linestyle=linestyle,
                marker="o",
                markersize=2.8,
                markeredgewidth=0,
            )
        if not mu_sub.empty:
            ax.plot(
                mu_sub["rt"],
                -mu_sub["plot_norm"],
                color=color,
                alpha=0.85,
                linewidth=linewidth,
                linestyle=mu_linestyle,
                marker="o",
                markersize=2.8,
                markeredgewidth=0,
            )

    di_rt = candidate.get("diann_rt", np.nan)
    mu_rt = candidate.get("rt_min", np.nan)
    top_ref_band = (0.15, 0.45)
    top_apex_band = (0.55, 1.05)
    bottom_ref_band = (-0.45, -0.15)
    bottom_apex_band = (-1.05, -0.55)
    if pd.notna(di_rt):
        ax.vlines(
            float(di_rt),
            ymin=top_ref_band[0],
            ymax=top_ref_band[1],
            color="black",
            linestyles=":",
            linewidth=1,
            alpha=0.8,
        )
    if pd.notna(mu_rt):
        ax.vlines(
            float(mu_rt),
            ymin=bottom_ref_band[0],
            ymax=bottom_ref_band[1],
            color="dimgray",
            linestyles=":",
            linewidth=1,
            alpha=0.8,
        )
    if pd.notna(di_apex_rt):
        ax.vlines(
            float(di_apex_rt),
            ymin=top_apex_band[0],
            ymax=top_apex_band[1],
            color="black",
            linestyles="-.",
            linewidth=1.3,
            alpha=0.85,
        )
    if pd.notna(mu_apex_rt):
        ax.vlines(
            float(mu_apex_rt),
            ymin=bottom_apex_band[0],
            ymax=bottom_apex_band[1],
            color="dimgray",
            linestyles="-.",
            linewidth=1.3,
            alpha=0.85,
        )
    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min, x_max)

    ax.axhline(0, color="lightgray", linewidth=1)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("RT (minutes)")
    ax.set_ylabel("Normalized intensity\n(+ DIA-NN / - MuMDIA)")
    ax.set_title(
        f"{precursor} | shared={len(shared_features)} | MuMDIA q={candidate.get('peptide_q_min', np.nan):.3g} | DIA-NN q={candidate.get('diann_qvalue', np.nan):.3g}"
    )

    legend_features = ", ".join(chosen[:8])
    if len(chosen) > 8:
        legend_features += ", ..."

    ax_meta.axis("off")
    meta_lines = [
        f"MuMDIA peptide: {peptide}",
        f"DIA-NN precursor: {precursor}",
        f"Apex RT (DIA-NN / MuMDIA): {di_apex_rt:.4f} / {mu_apex_rt:.4f}",
        f"Reference RT (DIA-NN / MuMDIA): {candidate.get('diann_rt', np.nan):.4f} / {candidate.get('rt_min', np.nan):.4f}",
        f"Shared plotted fragments: {len(shared_features)} / {len(chosen)} selected",
        f"MuMDIA top corr={candidate.get('top_correlation_individual_1', np.nan):.3f} | MuMDIA XIC best={candidate.get('xic_best_coelution', np.nan):.3f} | coverage={candidate.get('xic_coverage', np.nan):.3f}",
        f"DIA-NN best frag corr={candidate.get('diann_best_frag_corr', np.nan):.3f} | mean frag corr={candidate.get('diann_mean_frag_corr', np.nan):.3f} | n good={candidate.get('diann_n_good_frags', np.nan)}",
        f"Fragments: {legend_features}",
        "Display RT range is cropped to MuMDIA-style apex-fragment bounds with light padding.",
        "MS1 traces are rescaled relative to their own apex before plotting.",
        "Top RT markers = DIA-NN, bottom RT markers = MuMDIA.",
        "Dotted marker = reference RT, dash-dot marker = computed apex RT.",
        "Markers indicate individual sampled datapoints; lines connect them.",
        "Matching fragment labels use the same color in DIA-NN and MuMDIA.",
        "Line style: solid above zero = DIA-NN, dashed below zero = MuMDIA",
    ]
    ax_meta.text(
        0.01,
        0.95,
        "\n".join(meta_lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def render_subset(
    name: str,
    candidates: pd.DataFrame,
    mu_all: pd.DataFrame,
    di_all: pd.DataFrame,
    output_dir: Path,
    top_features: int,
) -> pd.DataFrame:
    subset_dir = output_dir / name
    subset_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in candidates.iterrows():
        filename = sanitize_filename(row["target_precursor"]) + ".png"
        out_path = subset_dir / filename
        ok = plot_one(row, mu_all, di_all, out_path, top_features=top_features)
        rows.append(
            {
                "subset": name,
                "Peptide": row["Peptide"],
                "charge": int(row["charge_int"]),
                "target_precursor": row["target_precursor"],
                "stripped_peptide": row["stripped_peptide"],
                "peptide_q_min": row.get("peptide_q_min", np.nan),
                "diann_qvalue": row.get("diann_qvalue", np.nan),
                "diann_best_frag_corr": row.get("diann_best_frag_corr", np.nan),
                "top_correlation_individual_1": row.get(
                    "top_correlation_individual_1", np.nan
                ),
                "xic_best_coelution": row.get("xic_best_coelution", np.nan),
                "plot_created": ok,
                "plot_path": str(out_path.relative_to(output_dir.parent)) if ok else "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exact, suspicious, top_overlap = load_candidates(args)
    exact.to_csv(output_dir / "mirror_candidates_exact.tsv", sep="\t", index=False)
    suspicious.to_csv(
        output_dir / "mirror_candidates_suspicious.tsv", sep="\t", index=False
    )
    top_overlap.to_csv(
        output_dir / "mirror_candidates_top_overlap.tsv", sep="\t", index=False
    )

    selected = pd.concat([suspicious, top_overlap], ignore_index=True)
    selected = selected.drop_duplicates(subset=["target_precursor"], keep="first")

    print(f"Exact same-peptidoform matches: {len(exact)}")
    print(f"Suspicious exact matches: {len(suspicious)}")
    print(f"Top-overlap exact matches selected: {len(top_overlap)}")
    print(f"Total plots requested: {len(selected)}")

    target_keys = build_target_key_df(selected)
    mu_all = stream_mumdia_fragments(
        Path(args.df_fragment), target_keys, args.chunksize
    )
    mu_all["feature_key"] = mu_all["feature"].map(canonicalize_feature_name)
    di_all = pd.read_parquet(args.diann_xic, columns=["pr", "feature", "rt", "value"])
    di_all = di_all[di_all["pr"].isin(selected["target_precursor"])].copy()
    di_all["feature_key"] = di_all["feature"].map(canonicalize_feature_name)

    index_frames = []
    if not suspicious.empty:
        index_frames.append(
            render_subset(
                "suspicious", suspicious, mu_all, di_all, output_dir, args.top_features
            )
        )
    if not top_overlap.empty:
        index_frames.append(
            render_subset(
                "top_overlap",
                top_overlap,
                mu_all,
                di_all,
                output_dir,
                args.top_features,
            )
        )

    if index_frames:
        index_df = pd.concat(index_frames, ignore_index=True)
    else:
        index_df = pd.DataFrame()
    index_df.to_csv(output_dir / "mirror_plot_index.tsv", sep="\t", index=False)
    print(f"Saved plot index to {output_dir / 'mirror_plot_index.tsv'}")


if __name__ == "__main__":
    main()
