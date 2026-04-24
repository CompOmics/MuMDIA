#!/usr/bin/env python3
"""Plot observed-vs-predicted MS2 spectra for suspicious peptidoforms."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SUSPICIOUS_TSV = (
    "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
    "mirror_xics/mirror_candidates_suspicious.tsv"
)
DEFAULT_DUMP_DIR = (
    "results_ecoli_rerun_20260402_py312_rust_deeplc4_60s/stage3_feature_input_dump"
)
DEFAULT_OUTPUT_DIR = (
    "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
    "suspicious_ms2_mirrors"
)

UNIMOD_REPLACEMENTS = {
    "[UniMod:4]": "[Carbamidomethyl]",
    "[UniMod:35]": "[Oxidation]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suspicious-tsv",
        default=DEFAULT_SUSPICIOUS_TSV,
        help="TSV containing suspicious peptidoforms.",
    )
    parser.add_argument(
        "--dump-dir",
        default=DEFAULT_DUMP_DIR,
        help="Stage 3 feature input dump directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots and the index TSV are written.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Optional maximum number of suspicious cases to render.",
    )
    parser.add_argument(
        "--peptidoform",
        action="append",
        dest="peptidoforms",
        help="Restrict plotting to one or more peptide/charge entries.",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=10,
        help="Number of fragment labels to annotate per panel.",
    )
    parser.add_argument(
        "--observed-min-rank-fraction",
        type=float,
        default=0.02,
        help="Only plot observed peaks above this fraction of the scan base peak.",
    )
    parser.add_argument(
        "--predicted-min-rank-fraction",
        type=float,
        default=0.02,
        help="Only plot predicted peaks above this fraction of the predicted base peak.",
    )
    parser.add_argument(
        "--mz-padding",
        type=float,
        default=25.0,
        help="Extra m/z padding applied around the combined observed/predicted range.",
    )
    return parser.parse_args()


def normalize_peptide(peptide: str) -> str:
    normalized = str(peptide).strip()
    for src, dst in UNIMOD_REPLACEMENTS.items():
        normalized = normalized.replace(src, dst)
    return normalized


def parse_peptidoform(value: str) -> tuple[str, int]:
    peptide, charge = str(value).strip().rsplit("/", 1)
    return normalize_peptide(peptide), int(charge)


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def safe_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(round(float(text)))
    except Exception:
        return None


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


def load_inputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    suspicious_df = pd.read_csv(args.suspicious_tsv, sep="\t", low_memory=False)
    suspicious_df["Peptide"] = suspicious_df["Peptide"].map(normalize_peptide)
    suspicious_df["charge"] = suspicious_df["charge"].astype(float).round().astype(int)
    suspicious_df["peptidoform"] = [
        f"{peptide}/{charge}"
        for peptide, charge in zip(suspicious_df["Peptide"], suspicious_df["charge"])
    ]

    df_psms = pd.read_csv(
        Path(args.dump_dir) / "df_psms_pre_feature_calc.tsv",
        sep="\t",
        low_memory=False,
    )
    df_psms["peptide"] = df_psms["peptide"].map(normalize_peptide)
    df_psms["charge"] = df_psms["charge"].astype(float).round().astype(int)
    df_psms["psm_id"] = df_psms["psm_id"].astype(int)
    df_psms["peptidoform"] = [
        f"{peptide}/{charge}"
        for peptide, charge in zip(df_psms["peptide"], df_psms["charge"])
    ]

    dump_dir = Path(args.dump_dir)
    ms2_dict = load_pickle(dump_dir / "ms2_dict.pkl")
    theoretical_fragment_context = load_pickle(
        dump_dir / "theoretical_fragment_context.pkl"
    )
    preannotated_fragment_dict = load_pickle(
        dump_dir / "preannotated_fragment_dict.pkl"
    )

    return (
        suspicious_df,
        df_psms,
        ms2_dict,
        theoretical_fragment_context,
        preannotated_fragment_dict,
    )


def filter_suspicious_cases(
    suspicious_df: pd.DataFrame, peptidoforms: list[str] | None, max_cases: int | None
) -> pd.DataFrame:
    filtered = suspicious_df.copy()
    if peptidoforms:
        wanted = {
            f"{peptide}/{charge}"
            for peptide, charge in map(parse_peptidoform, peptidoforms)
        }
        filtered = filtered[filtered["peptidoform"].isin(wanted)].copy()
    dedup_cols = ["peptidoform"]
    if "ScanNr" in filtered.columns:
        dedup_cols.append("ScanNr")
    filtered = filtered.drop_duplicates(  # type: ignore[call-overload]
        subset=dedup_cols, keep="first"
    )
    if max_cases is not None:
        filtered = filtered.head(max_cases).copy()
    return filtered


def select_representative_psm(
    suspicious_row: pd.Series,
    df_psms: pd.DataFrame,
) -> pd.Series | None:
    peptide = suspicious_row["Peptide"]
    charge = int(suspicious_row["charge"])
    peptidoform_df = df_psms[
        (df_psms["peptide"] == peptide) & (df_psms["charge"] == charge)
    ].copy()
    if peptidoform_df.empty:
        return None

    suspicious_psm_id = None
    for candidate_col in ["ScanNr", "scan_number", "psm_id"]:
        if candidate_col in suspicious_row.index:
            suspicious_psm_id = safe_int(suspicious_row[candidate_col])
            if suspicious_psm_id is not None:
                break

    if suspicious_psm_id is not None:
        direct = peptidoform_df[peptidoform_df["psm_id"] == suspicious_psm_id]
        if not direct.empty:
            return direct.iloc[0]

    if "filename" in suspicious_row.index and pd.notna(suspicious_row["filename"]):
        same_file = peptidoform_df[
            peptidoform_df["filename"].astype(str) == str(suspicious_row["filename"])
        ]
        if not same_file.empty:
            peptidoform_df = same_file.copy()

    sort_columns: list[str] = []
    ascending: list[bool] = []
    if "fragment_intensity" in peptidoform_df.columns:
        sort_columns.append("fragment_intensity")
        ascending.append(False)
    if "spectrum_q" in peptidoform_df.columns:
        sort_columns.append("spectrum_q")
        ascending.append(True)
    if "rt" in peptidoform_df.columns and "rt_min" in suspicious_row.index:
        target_rt = safe_float(suspicious_row.get("rt_min"))
        if target_rt is not None:
            peptidoform_df["__rt_distance"] = (
                peptidoform_df["rt"].astype(float) - target_rt
            ).abs()
            sort_columns.append("__rt_distance")
            ascending.append(True)

    if sort_columns:
        peptidoform_df = peptidoform_df.sort_values(
            by=sort_columns,
            ascending=ascending,
            na_position="last",
        )
    return peptidoform_df.iloc[0]


def get_preannotated_psm_frame(
    preannotated_fragment_dict: dict[str, Any],
    peptidoform: str,
    psm_id: int,
) -> pd.DataFrame:
    df_sub = preannotated_fragment_dict.get(peptidoform)
    if df_sub is None:
        return pd.DataFrame()
    if hasattr(df_sub, "to_pandas"):
        df_sub = df_sub.to_pandas()
    else:
        df_sub = pd.DataFrame(df_sub)
    if df_sub.empty or "psm_id" not in df_sub.columns:
        return pd.DataFrame()
    df_sub = df_sub[df_sub["psm_id"].astype(int) == int(psm_id)].copy()
    if df_sub.empty:
        return pd.DataFrame()
    if "fragment_name" in df_sub.columns:
        df_sub = df_sub.sort_values(  # type: ignore[call-overload]
            by=["fragment_intensity"], ascending=[False], na_position="last"
        ).drop_duplicates(subset=["fragment_name"], keep="first")
    return pd.DataFrame(df_sub)


def build_predicted_fragment_df(
    peptidoform: str,
    theoretical_fragment_context: dict[str, dict[str, Any]],
    preannotated_psm_df: pd.DataFrame,
) -> pd.DataFrame:
    context = theoretical_fragment_context.get(peptidoform, {})
    predicted_intensities = dict(context.get("predicted_intensities", {}) or {})
    theoretical_mz = dict(context.get("theoretical_fragment_mz", {}) or {})

    if not preannotated_psm_df.empty:
        for row in preannotated_psm_df.to_dict("records"):
            fragment_name = str(row.get("fragment_name", "")).strip()
            if not fragment_name:
                continue
            if fragment_name not in theoretical_mz:
                mz_value = safe_float(row.get("theoretical_fragment_mz"))
                if mz_value is not None:
                    theoretical_mz[fragment_name] = mz_value
            if fragment_name not in predicted_intensities:
                intensity_value = safe_float(row.get("predicted_fragment_intensity"))
                if intensity_value is not None:
                    predicted_intensities[fragment_name] = intensity_value

    records = []
    for fragment_name, intensity in predicted_intensities.items():
        mz_value = safe_float(theoretical_mz.get(fragment_name))
        intensity_value = safe_float(intensity)
        if mz_value is None or intensity_value is None or intensity_value <= 0:
            continue
        records.append(
            {
                "fragment_name": fragment_name,
                "mz": mz_value,
                "intensity": intensity_value,
            }
        )

    predicted_df = pd.DataFrame.from_records(records)
    if predicted_df.empty:
        return predicted_df
    predicted_df = predicted_df.sort_values("mz").reset_index(drop=True)
    predicted_df["normalized_intensity"] = (
        predicted_df["intensity"] / predicted_df["intensity"].max() * 100.0
    )

    observed_fragment_names = set(preannotated_psm_df.get("fragment_name", []))
    predicted_df["observed_in_reannotation"] = predicted_df["fragment_name"].isin(
        observed_fragment_names
    )
    return predicted_df


def build_observed_spectrum_df(
    scan_entry: dict[str, Any],
    observed_min_rank_fraction: float,
) -> pd.DataFrame:
    mz_values = np.asarray(scan_entry.get("mz", []), dtype=float)
    intensity_values = np.asarray(scan_entry.get("intensity", []), dtype=float)
    if mz_values.size == 0 or intensity_values.size == 0:
        return pd.DataFrame(columns=["mz", "intensity", "normalized_intensity"])

    observed_df = pd.DataFrame({"mz": mz_values, "intensity": intensity_values})
    observed_df = observed_df.replace([np.inf, -np.inf], np.nan).dropna()
    if observed_df.empty:
        return pd.DataFrame(observed_df)
    observed_df = observed_df[observed_df["intensity"] > 0].copy()
    if observed_df.empty:
        return pd.DataFrame(observed_df)
    observed_df["normalized_intensity"] = (
        observed_df["intensity"] / observed_df["intensity"].max() * 100.0
    )
    return pd.DataFrame(
        observed_df[
            observed_df["normalized_intensity"] >= observed_min_rank_fraction * 100.0
        ].copy()
    )


def normalize_preannotated_df(preannotated_psm_df: pd.DataFrame) -> pd.DataFrame:
    if preannotated_psm_df.empty:
        return preannotated_psm_df
    mz_column = None
    for candidate in ["fragment_mz", "fragment_mz_experimental"]:
        if candidate in preannotated_psm_df.columns:
            mz_column = candidate
            break
    if mz_column is None:
        return pd.DataFrame()

    df = preannotated_psm_df.copy()
    df["mz"] = pd.to_numeric(df[mz_column], errors="coerce")
    df["fragment_intensity"] = pd.to_numeric(df["fragment_intensity"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mz", "fragment_intensity"]
    )
    if df.empty:
        return pd.DataFrame(df)
    df = df[df["fragment_intensity"] > 0].copy()
    if df.empty:
        return pd.DataFrame(df)
    df["normalized_intensity"] = (
        df["fragment_intensity"] / df["fragment_intensity"].max() * 100.0
    )
    return pd.DataFrame(
        df.sort_values(  # type: ignore[call-overload]
            by=["fragment_intensity"], ascending=[False]
        ).reset_index(drop=True)
    )


def build_fragment_intensity_comparison_df(
    preannotated_psm_df: pd.DataFrame,
) -> pd.DataFrame:
    if preannotated_psm_df.empty:
        return pd.DataFrame()
    required_columns = {
        "fragment_name",
        "fragment_intensity",
        "predicted_fragment_intensity",
    }
    if not required_columns.issubset(preannotated_psm_df.columns):
        return pd.DataFrame()

    df = preannotated_psm_df.copy()
    df["fragment_intensity"] = pd.to_numeric(df["fragment_intensity"], errors="coerce")
    df["predicted_fragment_intensity"] = pd.to_numeric(
        df["predicted_fragment_intensity"], errors="coerce"
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["fragment_name", "fragment_intensity", "predicted_fragment_intensity"]
    )
    if df.empty:
        return pd.DataFrame(df)

    df = df[
        (df["fragment_intensity"] > 0) & (df["predicted_fragment_intensity"] > 0)
    ].copy()
    if df.empty:
        return pd.DataFrame(df)

    df = df.sort_values(  # type: ignore[call-overload]
        by=["fragment_intensity"], ascending=[False]
    ).drop_duplicates(subset=["fragment_name"], keep="first")
    df["observed_relative_intensity"] = (
        df["fragment_intensity"] / df["fragment_intensity"].max() * 100.0
    )
    df["predicted_relative_intensity"] = (
        df["predicted_fragment_intensity"]
        / df["predicted_fragment_intensity"].max()
        * 100.0
    )
    return pd.DataFrame(
        df.sort_values(  # type: ignore[call-overload]
            by=["predicted_relative_intensity"], ascending=[False]
        ).reset_index(drop=True)
    )


def annotate_top_peaks(
    ax: plt.Axes,
    df: pd.DataFrame,
    intensity_col: str,
    mz_col: str,
    label_col: str,
    top_n: int,
    text_offset: float,
) -> None:
    if df.empty or top_n <= 0:
        return
    top_df = df.sort_values(by=[intensity_col], ascending=[False]).head(top_n)
    for _, row in top_df.iterrows():
        label = str(row.get(label_col, "")).strip()
        if not label:
            continue
        ax.text(
            float(row[mz_col]),
            float(row[intensity_col]) + text_offset,
            label,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def choose_xlim(
    observed_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    mz_padding: float,
) -> tuple[float, float]:
    mz_values = []
    if not observed_df.empty:
        mz_values.extend(observed_df["mz"].tolist())
    if not predicted_df.empty:
        mz_values.extend(predicted_df["mz"].tolist())
    if not mz_values:
        return 100.0, 1500.0
    return max(0.0, min(mz_values) - mz_padding), max(mz_values) + mz_padding


def render_fragment_scatter_plot(
    suspicious_row: pd.Series,
    representative_psm: pd.Series,
    comparison_df: pd.DataFrame,
    output_path: Path,
    top_labels: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)

    x = comparison_df["predicted_relative_intensity"].astype(float)
    y = comparison_df["observed_relative_intensity"].astype(float)
    ax.scatter(
        x,
        y,
        s=45,
        color="#1f77b4",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    diagonal_limit = max(100.0, float(max(x.max(), y.max())) * 1.05)
    ax.plot([0, diagonal_limit], [0, diagonal_limit], linestyle="--", color="0.5")
    ax.set_xlim(0, diagonal_limit)
    ax.set_ylim(0, diagonal_limit)
    ax.set_xlabel("Predicted relative intensity")
    ax.set_ylabel("Observed relative intensity")
    ax.grid(alpha=0.25)

    for _, row in comparison_df.head(top_labels).iterrows():
        ax.text(
            float(row["predicted_relative_intensity"]) + 1.0,
            float(row["observed_relative_intensity"]) + 1.0,
            str(row["fragment_name"]),
            fontsize=8,
        )

    pearson_r = float(x.corr(y, method="pearson")) if len(comparison_df) > 1 else np.nan
    spearman_r = (
        float(x.corr(y, method="spearman")) if len(comparison_df) > 1 else np.nan
    )
    annotation_lines = [
        f"peptidoform={suspicious_row['peptidoform']}",
        f"psm_id={int(representative_psm['psm_id'])}",
        f"matched fragments={len(comparison_df)}",
        f"pearson_r={pearson_r:.3f}" if not np.isnan(pearson_r) else "pearson_r=NA",
        f"spearman_r={spearman_r:.3f}" if not np.isnan(spearman_r) else "spearman_r=NA",
    ]
    ax.set_title("Matched fragment intensities: predicted vs observed")
    ax.text(
        0.02,
        0.98,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "scatter_plot_path": str(output_path),
        "scatter_fragment_count": int(len(comparison_df)),
        "scatter_pearson_r": pearson_r,
        "scatter_spearman_r": spearman_r,
    }


def render_case_plot(
    suspicious_row: pd.Series,
    representative_psm: pd.Series,
    scan_entry: dict[str, Any],
    observed_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    output_path: Path,
    top_labels: int,
    mz_padding: float,
    predicted_min_rank_fraction: float,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not predicted_df.empty:
        predicted_df = pd.DataFrame(
            predicted_df[
                predicted_df["normalized_intensity"]
                >= predicted_min_rank_fraction * 100.0
            ].copy()
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.08},
    )
    ax_obs, ax_pred = axes

    if not observed_df.empty:
        ax_obs.vlines(
            observed_df["mz"],
            0,
            observed_df["normalized_intensity"],
            color="0.75",
            linewidth=0.8,
            alpha=0.8,
            label="all observed peaks",
        )
    if not matched_df.empty:
        ax_obs.vlines(
            matched_df["mz"],
            0,
            matched_df["normalized_intensity"],
            color="#1f77b4",
            linewidth=1.6,
            alpha=0.95,
            label="matched observed fragments",
        )
        annotate_top_peaks(
            ax_obs,
            matched_df,
            intensity_col="normalized_intensity",
            mz_col="mz",
            label_col="fragment_name",
            top_n=top_labels,
            text_offset=1.0,
        )

    predicted_observed = predicted_df[
        predicted_df.get("observed_in_reannotation", pd.Series(dtype=bool))
    ].copy()
    predicted_unobserved = predicted_df[
        ~predicted_df.get("observed_in_reannotation", pd.Series(dtype=bool))
    ].copy()
    if not predicted_unobserved.empty:
        ax_pred.vlines(
            predicted_unobserved["mz"],
            0,
            predicted_unobserved["normalized_intensity"],
            color="#ffbb78",
            linewidth=1.2,
            alpha=0.8,
            label="predicted only",
        )
    if not predicted_observed.empty:
        ax_pred.vlines(
            predicted_observed["mz"],
            0,
            predicted_observed["normalized_intensity"],
            color="#ff7f0e",
            linewidth=1.8,
            alpha=0.95,
            label="predicted + observed",
        )
    annotate_top_peaks(
        ax_pred,
        predicted_df,
        intensity_col="normalized_intensity",
        mz_col="mz",
        label_col="fragment_name",
        top_n=top_labels,
        text_offset=1.0,
    )

    x_min, x_max = choose_xlim(observed_df, predicted_df, mz_padding)
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper right", frameon=False)

    ax_obs.set_ylabel("Observed\nrelative intensity")
    ax_pred.set_ylabel("Predicted\nrelative intensity")
    ax_pred.set_xlabel("m/z")

    scan_rt = safe_float(representative_psm.get("rt"))
    predicted_rt = safe_float(representative_psm.get("rt_predictions"))
    suspicious_rt = safe_float(suspicious_row.get("rt_min"))
    subtitle = (
        f"peptidoform={suspicious_row['peptidoform']} | "
        f"psm_id={int(representative_psm['psm_id'])} | "
        f"scan={representative_psm['scannr']} | "
        f"rt={scan_rt if scan_rt is not None else 'NA'} | "
        f"rt_predictions={predicted_rt if predicted_rt is not None else 'NA'} | "
        f"suspicious_rt_min={suspicious_rt if suspicious_rt is not None else 'NA'}"
    )
    if "filename" in representative_psm.index:
        subtitle += f"\nfile={representative_psm['filename']}"

    fig.suptitle("Suspicious observed vs predicted MS2 spectrum", fontsize=14, y=0.98)
    fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=10)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "peptidoform": suspicious_row["peptidoform"],
        "Peptide": suspicious_row["Peptide"],
        "charge": int(suspicious_row["charge"]),
        "plot_path": str(output_path),
        "psm_id": int(representative_psm["psm_id"]),
        "scannr": str(representative_psm["scannr"]),
        "observed_peak_count": int(len(observed_df)),
        "matched_fragment_count": int(len(matched_df)),
        "predicted_fragment_count": int(len(predicted_df)),
        "top_predicted_observed_count": int(predicted_observed.shape[0]),
        "target_precursor": suspicious_row.get("target_precursor"),
        "gap_key": suspicious_row.get("gap_key"),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        suspicious_df,
        df_psms,
        ms2_dict,
        theoretical_fragment_context,
        preannotated_fragment_dict,
    ) = load_inputs(args)
    suspicious_df = filter_suspicious_cases(
        suspicious_df,
        peptidoforms=args.peptidoforms,
        max_cases=args.max_cases,
    )

    summary_records: list[dict[str, Any]] = []
    scatter_dir = output_dir / "intensity_scatter"
    for suspicious_row in suspicious_df.to_dict("records"):
        suspicious_series = pd.Series(suspicious_row)
        representative_psm = select_representative_psm(suspicious_series, df_psms)
        if representative_psm is None:
            summary_records.append(
                {
                    "peptidoform": suspicious_series["peptidoform"],
                    "status": "missing_psm",
                }
            )
            continue

        scannr = str(representative_psm["scannr"])
        scan_entry = ms2_dict.get(scannr)
        if scan_entry is None:
            summary_records.append(
                {
                    "peptidoform": suspicious_series["peptidoform"],
                    "psm_id": int(representative_psm["psm_id"]),
                    "scannr": scannr,
                    "status": "missing_ms2_scan",
                }
            )
            continue

        peptidoform = str(suspicious_series["peptidoform"])
        preannotated_psm_df = get_preannotated_psm_frame(
            preannotated_fragment_dict,
            peptidoform,
            int(representative_psm["psm_id"]),
        )
        matched_df = normalize_preannotated_df(preannotated_psm_df)
        comparison_df = build_fragment_intensity_comparison_df(preannotated_psm_df)
        predicted_df = build_predicted_fragment_df(
            peptidoform,
            theoretical_fragment_context,
            preannotated_psm_df,
        )
        observed_df = build_observed_spectrum_df(
            scan_entry,
            observed_min_rank_fraction=args.observed_min_rank_fraction,
        )

        if observed_df.empty and predicted_df.empty:
            summary_records.append(
                {
                    "peptidoform": peptidoform,
                    "psm_id": int(representative_psm["psm_id"]),
                    "scannr": scannr,
                    "status": "empty_observed_and_predicted",
                }
            )
            continue

        output_name = (
            f"{sanitize_filename(peptidoform)}"
            f"__psm_{int(representative_psm['psm_id'])}.png"
        )
        output_path = output_dir / output_name
        result = render_case_plot(
            suspicious_row=suspicious_series,
            representative_psm=representative_psm,
            scan_entry=scan_entry,
            observed_df=observed_df,
            matched_df=matched_df,
            predicted_df=predicted_df,
            output_path=output_path,
            top_labels=args.top_labels,
            mz_padding=args.mz_padding,
            predicted_min_rank_fraction=args.predicted_min_rank_fraction,
        )
        if not comparison_df.empty:
            scatter_name = (
                f"{sanitize_filename(peptidoform)}"
                f"__psm_{int(representative_psm['psm_id'])}__scatter.png"
            )
            scatter_result = render_fragment_scatter_plot(
                suspicious_row=suspicious_series,
                representative_psm=representative_psm,
                comparison_df=comparison_df,
                output_path=scatter_dir / scatter_name,
                top_labels=args.top_labels,
            )
            result.update(scatter_result)
        else:
            result.update(
                {
                    "scatter_plot_path": np.nan,
                    "scatter_fragment_count": 0,
                    "scatter_pearson_r": np.nan,
                    "scatter_spearman_r": np.nan,
                }
            )
        result["status"] = "ok"
        summary_records.append(result)

    summary_df = pd.DataFrame(summary_records)
    summary_path = output_dir / "suspicious_ms2_mirror_index.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    ok_count = (
        int((summary_df.get("status") == "ok").sum()) if not summary_df.empty else 0
    )
    print(f"Processed {len(summary_df)} suspicious cases")
    print(f"Successful plots: {ok_count}")
    print(f"Summary index: {summary_path}")


if __name__ == "__main__":
    main()
