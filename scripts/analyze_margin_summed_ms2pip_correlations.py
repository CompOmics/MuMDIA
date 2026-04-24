#!/usr/bin/env python3
"""Aggregate fragment intensities within RT margins and compare them to MS2PIP."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_FRAGMENT_TABLE = (
    "results_ecoli_rerun_20260402_py312_rust_deeplc4_60s/rt_margin_input_dump/"
    "replayed_margins_top2_ms2grid/df_fragment_with_replayed_margins.tsv"
)
DEFAULT_CONTEXT_PICKLE = (
    "results_ecoli_rerun_20260402_py312_rust_deeplc4_60s/stage3_feature_input_dump/"
    "theoretical_fragment_context.pkl"
)
DEFAULT_SUSPICIOUS_TSV = (
    "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
    "mirror_xics/mirror_candidates_suspicious.tsv"
)
DEFAULT_OUTPUT_DIR = (
    "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
    "margin_summed_ms2pip_analysis"
)

UNIMOD_REPLACEMENTS = {
    "[UniMod:4]": "[Carbamidomethyl]",
    "[UniMod:35]": "[Oxidation]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fragment-table",
        default=DEFAULT_FRAGMENT_TABLE,
        help="Fragment table containing RT margins and reannotated fragment names.",
    )
    parser.add_argument(
        "--context-pickle",
        default=DEFAULT_CONTEXT_PICKLE,
        help="Theoretical fragment context pickle from Stage 3 dump.",
    )
    parser.add_argument(
        "--suspicious-tsv",
        default=DEFAULT_SUSPICIOUS_TSV,
        help="Optional suspicious peptidoform TSV for group comparisons.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where summary tables and plots are written.",
    )
    parser.add_argument(
        "--min-fragments",
        type=int,
        default=3,
        help="Minimum number of non-zero comparison fragments required for correlations.",
    )
    parser.add_argument(
        "--top-fragment-count",
        type=int,
        default=12,
        help="Number of top summed fragments to store in the summary table.",
    )
    parser.add_argument(
        "--candidate-tsv",
        help=(
            "Optional TSV with Peptide+charge columns used to restrict analysis to a "
            "candidate set such as q<=10%% targets/decoys."
        ),
    )
    parser.add_argument(
        "--candidate-group-column",
        default="label_group",
        help=(
            "Column in --candidate-tsv used for group-wise summaries and plots. "
            "Ignored if the column is missing."
        ),
    )
    return parser.parse_args()


def normalize_peptide(peptide: str) -> str:
    text = str(peptide).strip()
    for src, dst in UNIMOD_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if np.isnan(parsed):
        return None
    return parsed


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    return pearson_corr(rankdata(x), rankdata(y))


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm == 0.0 or y_norm == 0.0:
        return float("nan")
    return float(np.dot(x, y) / (x_norm * y_norm))


def load_suspicious_peptidoforms(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if not {"Peptide", "charge"}.issubset(df.columns):
        return set()
    peptides = df["Peptide"].map(normalize_peptide)
    charges = pd.to_numeric(df["charge"], errors="coerce").dropna()
    out = set()
    for peptide, charge in zip(peptides, pd.to_numeric(df["charge"], errors="coerce")):
        if pd.isna(charge):
            continue
        out.add(f"{peptide}/{int(round(float(charge)))}")
    return out


def load_candidate_groups(
    path: Path | None,
    group_column: str,
) -> tuple[set[str] | None, dict[str, str]]:
    if path is None or not path.exists():
        return None, {}

    df = pd.read_csv(path, sep="\t", low_memory=False)
    if not {"Peptide", "charge"}.issubset(df.columns):
        raise KeyError(f"{path} must contain Peptide and charge columns")

    df = df.copy()
    df["Peptide"] = df["Peptide"].map(normalize_peptide)
    df["charge"] = pd.to_numeric(df["charge"], errors="coerce")
    df = df.dropna(subset=["charge"]).copy()
    df["charge"] = df["charge"].round().astype(int)
    df["peptidoform"] = [
        f"{peptide}/{charge}" for peptide, charge in zip(df["Peptide"], df["charge"])
    ]

    if group_column in df.columns:
        grouped = (
            df[["peptidoform", group_column]]
            .dropna(subset=[group_column])
            .drop_duplicates(subset=["peptidoform"], keep="first")
        )
        group_lookup = dict(
            zip(grouped["peptidoform"].astype(str), grouped[group_column].astype(str))
        )
    else:
        group_lookup = {}

    return set(df["peptidoform"].astype(str)), group_lookup


def prepare_fragment_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["peptide"] = df["peptide"].map(normalize_peptide)
    df["charge"] = pd.to_numeric(df["charge"], errors="coerce").round().astype("Int64")
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    df["fragment_intensity"] = pd.to_numeric(df["fragment_intensity"], errors="coerce")
    df["rt_lower_margin"] = pd.to_numeric(df["rt_lower_margin"], errors="coerce")
    df["rt_higher_margin"] = pd.to_numeric(df["rt_higher_margin"], errors="coerce")
    df["predicted_fragment_intensity"] = pd.to_numeric(
        df.get("predicted_fragment_intensity"), errors="coerce"
    )
    df = df.dropna(
        subset=[
            "peptide",
            "charge",
            "fragment_name",
            "rt",
            "fragment_intensity",
            "rt_lower_margin",
            "rt_higher_margin",
        ]
    ).copy()
    df["charge"] = df["charge"].astype(int)
    df["peptidoform"] = [
        f"{peptide}/{charge}" for peptide, charge in zip(df["peptide"], df["charge"])
    ]
    df = df[
        (df["fragment_intensity"] > 0)
        & (df["rt_higher_margin"] >= df["rt_lower_margin"])
    ].copy()
    return df


def aggregate_within_margin(
    df_sub: pd.DataFrame,
    predicted_intensities: dict[str, float],
    min_fragments: int,
    top_fragment_count: int,
    is_suspicious: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    margin_filtered = df_sub[
        (df_sub["rt"] >= df_sub["rt_lower_margin"])
        & (df_sub["rt"] <= df_sub["rt_higher_margin"])
    ].copy()

    if margin_filtered.empty:
        return {
            "margin_scan_count": 0,
            "observed_fragment_count": 0,
            "predicted_fragment_count": len(predicted_intensities),
            "nonzero_union_fragment_count": 0,
            "matched_nonzero_fragment_count": 0,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "cosine_similarity": np.nan,
            "observed_total_intensity": 0.0,
            "predicted_total_intensity": float(sum(predicted_intensities.values())),
            "is_suspicious": is_suspicious,
            "top_summed_fragments": "",
        }, pd.DataFrame()

    observed_sum = (
        margin_filtered.groupby("fragment_name", as_index=False)["fragment_intensity"]
        .sum()
        .rename(columns={"fragment_intensity": "observed_margin_sum"})
    )
    observed_map = dict(
        zip(observed_sum["fragment_name"], observed_sum["observed_margin_sum"])
    )

    all_fragment_names = sorted(set(predicted_intensities) | set(observed_map))
    comparison_df = pd.DataFrame(
        {
            "fragment_name": all_fragment_names,
            "predicted_intensity": [
                float(predicted_intensities.get(name, 0.0))
                for name in all_fragment_names
            ],
            "observed_margin_sum": [
                float(observed_map.get(name, 0.0)) for name in all_fragment_names
            ],
        }
    )
    comparison_df = comparison_df[
        (comparison_df["predicted_intensity"] > 0)
        | (comparison_df["observed_margin_sum"] > 0)
    ].copy()
    if comparison_df.empty:
        return {
            "margin_scan_count": (
                int(margin_filtered["scannr"].nunique())
                if "scannr" in margin_filtered.columns
                else 0
            ),
            "observed_fragment_count": 0,
            "predicted_fragment_count": len(predicted_intensities),
            "nonzero_union_fragment_count": 0,
            "matched_nonzero_fragment_count": 0,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "cosine_similarity": np.nan,
            "observed_total_intensity": 0.0,
            "predicted_total_intensity": float(sum(predicted_intensities.values())),
            "is_suspicious": is_suspicious,
            "top_summed_fragments": "",
        }, comparison_df

    comparison_df["observed_relative_intensity"] = (
        comparison_df["observed_margin_sum"]
        / comparison_df["observed_margin_sum"].max()
        * 100.0
        if float(comparison_df["observed_margin_sum"].max()) > 0.0
        else 0.0
    )
    comparison_df["predicted_relative_intensity"] = (
        comparison_df["predicted_intensity"]
        / comparison_df["predicted_intensity"].max()
        * 100.0
        if float(comparison_df["predicted_intensity"].max()) > 0.0
        else 0.0
    )

    x = comparison_df["predicted_intensity"].to_numpy(dtype=float)
    y = comparison_df["observed_margin_sum"].to_numpy(dtype=float)
    nonzero_union = (x > 0) | (y > 0)
    matched_nonzero = (x > 0) & (y > 0)
    metric_ready = int(nonzero_union.sum()) >= min_fragments

    top_fragments = ";".join(
        comparison_df.sort_values("observed_margin_sum", ascending=False)
        .head(top_fragment_count)["fragment_name"]
        .astype(str)
        .tolist()
    )

    summary = {
        "margin_scan_count": (
            int(margin_filtered["scannr"].nunique())
            if "scannr" in margin_filtered.columns
            else 0
        ),
        "observed_fragment_count": int(
            (comparison_df["observed_margin_sum"] > 0).sum()
        ),
        "predicted_fragment_count": int(
            (comparison_df["predicted_intensity"] > 0).sum()
        ),
        "nonzero_union_fragment_count": int(nonzero_union.sum()),
        "matched_nonzero_fragment_count": int(matched_nonzero.sum()),
        "pearson_r": pearson_corr(x, y) if metric_ready else np.nan,
        "spearman_r": spearman_corr(x, y) if metric_ready else np.nan,
        "cosine_similarity": cosine_similarity(x, y) if metric_ready else np.nan,
        "observed_total_intensity": float(comparison_df["observed_margin_sum"].sum()),
        "predicted_total_intensity": float(comparison_df["predicted_intensity"].sum()),
        "is_suspicious": is_suspicious,
        "top_summed_fragments": top_fragments,
    }
    return summary, comparison_df


def plot_histogram(summary_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    df = summary_df.dropna(subset=[metric]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    suspicious = df[df["is_suspicious"]]
    other = df[~df["is_suspicious"]]
    if not other.empty:
        ax.hist(
            other[metric], bins=40, alpha=0.6, color="#1f77b4", label="non-suspicious"
        )
    if not suspicious.empty:
        ax.hist(
            suspicious[metric], bins=40, alpha=0.6, color="#d62728", label="suspicious"
        )
    ax.set_xlabel(metric)
    ax.set_ylabel("peptidoforms")
    ax.set_title(f"Distribution of {metric} for margin-summed fragments")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_histogram_by_group(
    summary_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    group_column: str,
) -> None:
    df = summary_df.dropna(subset=[metric, group_column]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    for idx, group in enumerate(sorted(df[group_column].astype(str).unique())):
        sub = df[df[group_column].astype(str) == str(group)]
        if sub.empty:
            continue
        ax.hist(
            sub[metric],
            bins=40,
            alpha=0.55,
            color=palette[idx % len(palette)],
            label=str(group),
        )
    ax.set_xlabel(metric)
    ax.set_ylabel("peptidoforms")
    ax.set_title(f"Distribution of {metric} for margin-summed fragments")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_count_vs_correlation(summary_df: pd.DataFrame, output_path: Path) -> None:
    df = summary_df.dropna(subset=["pearson_r"]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    colors = np.where(df["is_suspicious"], "#d62728", "#1f77b4")
    ax.scatter(
        df["matched_nonzero_fragment_count"],
        df["pearson_r"],
        c=colors,
        alpha=0.65,
        s=30,
        edgecolors="none",
    )
    ax.set_xlabel("matched non-zero fragments")
    ax.set_ylabel("pearson_r")
    ax.set_title("Margin-summed fragment correlation vs matched fragment count")
    ax.grid(alpha=0.2)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_count_vs_correlation_by_group(
    summary_df: pd.DataFrame,
    output_path: Path,
    group_column: str,
) -> None:
    df = summary_df.dropna(subset=["pearson_r", group_column]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    for idx, group in enumerate(sorted(df[group_column].astype(str).unique())):
        sub = df[df[group_column].astype(str) == str(group)]
        if sub.empty:
            continue
        ax.scatter(
            sub["matched_nonzero_fragment_count"],
            sub["pearson_r"],
            c=palette[idx % len(palette)],
            alpha=0.6,
            s=30,
            edgecolors="none",
            label=str(group),
        )
    ax.set_xlabel("matched non-zero fragments")
    ax.set_ylabel("pearson_r")
    ax.set_title("Margin-summed fragment correlation vs matched fragment count")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_summary_json(summary_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_peptidoforms": int(len(summary_df)),
        "n_with_pearson": int(summary_df["pearson_r"].notna().sum()),
        "n_suspicious": int(summary_df["is_suspicious"].sum()),
    }
    for metric in ["pearson_r", "spearman_r", "cosine_similarity"]:
        series = summary_df[metric].dropna()
        if not series.empty:
            out[f"{metric}_median"] = float(series.median())
            out[f"{metric}_mean"] = float(series.mean())
    for label, mask in {
        "suspicious": summary_df["is_suspicious"],
        "non_suspicious": ~summary_df["is_suspicious"],
    }.items():
        subset = summary_df.loc[mask]
        out[label] = {
            "n": int(len(subset)),
            "n_with_pearson": int(subset["pearson_r"].notna().sum()),
            "pearson_median": (
                float(subset["pearson_r"].dropna().median())
                if subset["pearson_r"].notna().any()
                else None
            ),
            "spearman_median": (
                float(subset["spearman_r"].dropna().median())
                if subset["spearman_r"].notna().any()
                else None
            ),
            "cosine_median": (
                float(subset["cosine_similarity"].dropna().median())
                if subset["cosine_similarity"].notna().any()
                else None
            ),
            "matched_fragment_count_median": (
                float(subset["matched_nonzero_fragment_count"].median())
                if not subset.empty
                else None
            ),
        }
    return out


def build_group_summary_json(summary_df: pd.DataFrame, group_column: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if group_column not in summary_df.columns:
        return out

    for group in sorted(summary_df[group_column].dropna().astype(str).unique()):
        subset = summary_df[summary_df[group_column].astype(str) == str(group)]
        out[str(group)] = {
            "n": int(len(subset)),
            "n_with_pearson": int(subset["pearson_r"].notna().sum()),
            "pearson_median": float(subset["pearson_r"].dropna().median())
            if subset["pearson_r"].notna().any()
            else None,
            "spearman_median": float(subset["spearman_r"].dropna().median())
            if subset["spearman_r"].notna().any()
            else None,
            "cosine_median": float(subset["cosine_similarity"].dropna().median())
            if subset["cosine_similarity"].notna().any()
            else None,
            "matched_fragment_count_median": float(
                subset["matched_nonzero_fragment_count"].median()
            )
            if not subset.empty
            else None,
        }
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fragment_df = prepare_fragment_table(Path(args.fragment_table))
    theoretical_fragment_context = load_pickle(Path(args.context_pickle))
    suspicious_peptidoforms = load_suspicious_peptidoforms(Path(args.suspicious_tsv))
    candidate_peptidoforms, candidate_group_lookup = load_candidate_groups(
        Path(args.candidate_tsv) if args.candidate_tsv else None,
        args.candidate_group_column,
    )

    if candidate_peptidoforms is not None:
        fragment_df = fragment_df[
            fragment_df["peptidoform"].isin(candidate_peptidoforms)
        ].copy()

    summary_records: list[dict[str, Any]] = []
    fragment_records: list[pd.DataFrame] = []

    for peptidoform, df_sub in fragment_df.groupby("peptidoform", sort=True):
        context = theoretical_fragment_context.get(peptidoform, {})
        predicted_intensities = {
            str(k): float(v)
            for k, v in dict(context.get("predicted_intensities", {}) or {}).items()
            if safe_float(v) is not None and float(v) > 0
        }
        summary, comparison_df = aggregate_within_margin(
            df_sub=df_sub,
            predicted_intensities=predicted_intensities,
            min_fragments=args.min_fragments,
            top_fragment_count=args.top_fragment_count,
            is_suspicious=peptidoform in suspicious_peptidoforms,
        )
        peptide, charge = peptidoform.rsplit("/", 1)
        summary.update(
            {
                "peptidoform": peptidoform,
                "peptide": peptide,
                "charge": int(charge),
                "candidate_group": candidate_group_lookup.get(peptidoform),
            }
        )
        summary_records.append(summary)

        if not comparison_df.empty:
            fragment_records.append(
                comparison_df.assign(
                    peptidoform=peptidoform,
                    peptide=peptide,
                    charge=int(charge),
                    is_suspicious=peptidoform in suspicious_peptidoforms,
                    candidate_group=candidate_group_lookup.get(peptidoform),
                )
            )

    summary_df = pd.DataFrame(summary_records).sort_values(
        ["pearson_r", "spearman_r", "cosine_similarity"],
        ascending=[False, False, False],
        na_position="last",
    )
    summary_path = output_dir / "margin_summed_ms2pip_correlations.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    if fragment_records:
        fragment_long_df = pd.concat(fragment_records, ignore_index=True)
        fragment_long_df.to_csv(
            output_dir / "margin_summed_ms2pip_fragment_long.tsv",
            sep="\t",
            index=False,
        )

    plot_histogram(summary_df, "pearson_r", output_dir / "pearson_histogram.png")
    plot_histogram(summary_df, "cosine_similarity", output_dir / "cosine_histogram.png")
    plot_count_vs_correlation(
        summary_df, output_dir / "pearson_vs_matched_fragment_count.png"
    )
    if summary_df["candidate_group"].notna().any():
        plot_histogram_by_group(
            summary_df,
            "pearson_r",
            output_dir / "pearson_histogram_by_candidate_group.png",
            "candidate_group",
        )
        plot_histogram_by_group(
            summary_df,
            "cosine_similarity",
            output_dir / "cosine_histogram_by_candidate_group.png",
            "candidate_group",
        )
        plot_count_vs_correlation_by_group(
            summary_df,
            output_dir / "pearson_vs_matched_fragment_count_by_candidate_group.png",
            "candidate_group",
        )

    summary_json = build_summary_json(summary_df)
    summary_json["candidate_groups"] = build_group_summary_json(
        summary_df, "candidate_group"
    )
    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))

    print(f"Processed {len(summary_df)} peptidoforms")
    print(f"Wrote summary table to {summary_path}")
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
