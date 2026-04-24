"""
Apex RT comparison: MuMDIA vs DIA-NN, across ALL peptidoforms (including below FDR).

Usage:
    python scripts/analyze_apex_comparison.py \
        --mumdia-apex  debug/df_fragment_max_peptide_after_ms2pip.tsv \
        --mumdia-psms  debug/df_psms_after_ms2pip.tsv \
        --mokapot      runs/ecoli_splitmult1p5_calibonly/pipeline/xgboost.mokapot.psms.txt \
        --diann        diann_results/report_with_xic.tsv \
        --outdir       runs/ecoli_splitmult1p5_calibonly/comparison/apex_comparison

Outputs (all in --outdir):
    apex_comparison.tsv          – merged table with all RT & q-value columns
    scatter_apex_rt.png          – MuMDIA apex RT vs DIA-NN apex RT
    hist_delta_apex_rt.png       – histogram of (MuMDIA_rt – DIA-NN_rt) per FDR bucket
    scatter_rt_error_vs_delta.png – MuMDIA predicted-vs-obs RT error vs apex delta
    boxplot_delta_by_fdr.png     – box/violin of apex delta by FDR group
    summary.tsv                  – aggregate statistics per FDR bucket
"""

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strip_mods(peptide: str) -> str:
    """Remove bracketed modifications and keep only the amino acid sequence (no termini)."""
    # remove n-term dashes notation like n[...]-
    seq = re.sub(r"n\[[^\]]*\]-", "", peptide)
    seq = re.sub(r"\[[^\]]*\]", "", seq)
    # remove underscores (Percolator style _ wrapper)
    seq = seq.strip("_")
    return seq


def parse_precursor_id(pid: str) -> tuple[str, int] | tuple[None, None]:
    """
    Parse a DIA-NN Precursor.Id into (stripped_peptide, charge).
    Formats seen:
      AAAAEIAVK2       → seq=AAAAEIAVK, charge=2
      _AAAAEIAVK_2     → seq=AAAAEIAVK, charge=2
    """
    pid = str(pid).strip()
    if not pid:
        return None, None
    # trailing digit(s) = charge
    m = re.match(r"^(?:_)?([A-Z\[\]0-9+.\-nct]+?)(?:_)?(\d+)$", pid)
    if m:
        raw_seq, charge = m.group(1), int(m.group(2))
        return strip_mods(raw_seq), charge
    return None, None


def fdr_label(q: float) -> str:
    if pd.isna(q):
        return "no_q_value"
    if q <= 0.01:
        return "q≤1%"
    if q <= 0.05:
        return "q≤5%"
    if q <= 0.10:
        return "q≤10%"
    return "q>10%"


FDR_ORDER = ["q≤1%", "q≤5%", "q≤10%", "q>10%", "no_q_value"]
FDR_COLORS = {
    "q≤1%": "#2ca02c",
    "q≤5%": "#17becf",
    "q≤10%": "#ff7f0e",
    "q>10%": "#d62728",
    "no_q_value": "#9467bd",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_mumdia_apex(path: str) -> pd.DataFrame:
    """
    df_fragment_max_peptide_after_ms2pip.tsv
    One row per peptidoform (peptide/charge), rt = apex observed RT in seconds.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # stripped peptide key for matching
    df["stripped_peptide_key"] = df["peptide"].apply(strip_mods)
    # normalise charge
    df["charge"] = df["charge"].astype(int)
    return df


def load_mumdia_psms(path: str) -> pd.DataFrame:
    """
    df_psms_after_ms2pip.tsv — all PSMs (including below FDR).
    Used to pull rt_lower_margin / rt_higher_margin and to confirm apex.
    """
    cols_wanted = [
        "psm_id",
        "peptide",
        "charge",
        "rt",
        "scan_rt",
        "rt_candidate_lower",
        "rt_candidate_upper",
        "rt_lower_margin",
        "rt_higher_margin",
        "rt_predictions",
        "rt_prediction_error_abs",
        "rt_prediction_error_abs_relative",
        "spectrum_q",
        "peptide_q",
        "is_decoy",
        "fragment_intensity",
    ]
    df = pd.read_csv(
        path, sep="\t", low_memory=False, usecols=lambda c: c in cols_wanted
    )
    df["stripped_peptide_key"] = df["peptide"].apply(strip_mods)
    df["charge"] = df["charge"].astype(int)
    return df


def load_mokapot(path: str) -> pd.DataFrame:
    """
    xgboost.mokapot.psms.txt → one mokapot q-value per PSM (Peptide, SpecId).
    We aggregate to peptidoform by taking the minimum (best) q-value per peptide+charge.
    Note: Peptide column may be full modified sequence; SpecId = psm_id.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Extract charge from SpecId if Precursor.Charge not present
    # SpecId format: <int> psm_id
    # Peptide format: _SEQUENCEK_ (underscore-wrapped)
    df["stripped_peptide_key"] = df["Peptide"].apply(
        lambda p: strip_mods(str(p).strip("_"))
    )

    # charge from filename or ScanNr — look at columns
    # mokapot file has no explicit charge column; we join via SpecId → psm_id later
    df["psm_id"] = pd.to_numeric(df["SpecId"], errors="coerce")
    df["mokapot_q"] = pd.to_numeric(df["mokapot q-value"], errors="coerce")
    df["mokapot_score"] = pd.to_numeric(df["mokapot score"], errors="coerce")
    return df[["psm_id", "stripped_peptide_key", "mokapot_q", "mokapot_score"]]


def load_diann(path: str) -> pd.DataFrame:
    """
    report_with_xic.tsv from DIA-NN. Key columns: Precursor.Id, Q.Value, RT, Predicted.RT.
    Uses Stripped.Sequence + Precursor.Charge as the join key to avoid parsing issues
    with modifications like C(UniMod:4) in Precursor.Id.
    """
    cols_wanted = {
        "Precursor.Id",
        "Precursor.Charge",
        "Q.Value",
        "Global.Q.Value",
        "RT",
        "RT.Start",
        "RT.Stop",
        "Predicted.RT",
        "Stripped.Sequence",
        "Modified.Sequence",
        "Protein.Ids",
        "Precursor.Quantity",
        "Precursor.Normalised",
    }
    df = pd.read_csv(
        path, sep="\t", low_memory=False, usecols=lambda c: c in cols_wanted
    )

    # Prefer Stripped.Sequence directly; fall back to parsing Precursor.Id
    if "Stripped.Sequence" in df.columns:
        df["stripped_peptide_key"] = df["Stripped.Sequence"].astype(str).str.strip("_")
    else:
        df["stripped_peptide_key"] = df["Precursor.Id"].apply(
            lambda x: parse_precursor_id(str(x))[0]
        )

    if "Precursor.Charge" in df.columns:
        df["diann_charge"] = pd.to_numeric(
            df["Precursor.Charge"], errors="coerce"
        ).astype("Int64")
    else:
        df["diann_charge"] = df["Precursor.Id"].apply(
            lambda x: parse_precursor_id(str(x))[1]
        )
        df["diann_charge"] = pd.to_numeric(df["diann_charge"], errors="coerce").astype(
            "Int64"
        )

    df["diann_rt"] = pd.to_numeric(df["RT"], errors="coerce")
    df["diann_rt_start"] = pd.to_numeric(df.get("RT.Start", pd.NA), errors="coerce")
    df["diann_rt_stop"] = pd.to_numeric(df.get("RT.Stop", pd.NA), errors="coerce")
    df["diann_pred_rt"] = pd.to_numeric(df.get("Predicted.RT", pd.NA), errors="coerce")
    df["diann_q"] = pd.to_numeric(df["Q.Value"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------


def build_merged_table(
    df_apex: pd.DataFrame,
    df_psms: pd.DataFrame,
    df_mok: pd.DataFrame,
    df_diann: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge MuMDIA apex (one row per peptidoform) with mokapot q-values and DIA-NN.
    """
    # 1. Join mokapot scores onto MuMDIA apex via psm_id (deduplicate first)
    df_mok_by_psm = (
        df_mok.sort_values("mokapot_q").drop_duplicates("psm_id").set_index("psm_id")
    )
    apex_cols = df_apex.copy()
    apex_cols["mokapot_q"] = apex_cols["psm_id"].map(df_mok_by_psm["mokapot_q"])
    apex_cols["mokapot_score"] = apex_cols["psm_id"].map(df_mok_by_psm["mokapot_score"])

    # Also aggregate mokapot over all PSMs of this peptidoform (best = min q)
    # Join mokapot psm_ids → peptidoform key via df_psms
    psm_peptidoform = (
        df_psms[["psm_id", "stripped_peptide_key", "charge"]]
        .drop_duplicates("psm_id")
        .rename(columns={"stripped_peptide_key": "pep_key_psm"})
    )
    mok_with_pep = df_mok.merge(psm_peptidoform, on="psm_id", how="left")
    # fallback: use stripped_peptide_key already on df_mok (no charge info)
    valid = mok_with_pep.dropna(subset=["pep_key_psm"])
    if len(valid) == 0:
        mok_per_pep = pd.DataFrame(
            columns=[
                "stripped_peptide_key",
                "charge",
                "mokapot_q_best",
                "mokapot_score_best",
            ]
        )
    else:
        mok_per_pep = (
            valid.groupby(["pep_key_psm", "charge"])
            .agg(
                mokapot_q_best=("mokapot_q", "min"),
                mokapot_score_best=("mokapot_score", "max"),
            )
            .reset_index()
            .rename(columns={"pep_key_psm": "stripped_peptide_key"})
        )

    apex_cols = apex_cols.merge(
        mok_per_pep, on=["stripped_peptide_key", "charge"], how="left"
    )

    # 2. Attach RT margin info from df_psms (per-PSM margins at the apex psm_id)
    margin_cols = [
        "psm_id",
        "rt_candidate_lower",
        "rt_candidate_upper",
        "rt_lower_margin",
        "rt_higher_margin",
        "rt_prediction_error_abs",
        "rt_prediction_error_abs_relative",
    ]
    margin_cols = [c for c in margin_cols if c in df_psms.columns]
    df_margins = df_psms[margin_cols].drop_duplicates("psm_id")
    apex_cols = apex_cols.merge(
        df_margins, on="psm_id", how="left", suffixes=("", "_psm")
    )

    # 3. Merge DIA-NN on stripped_peptide_key + charge
    #    DIA-NN may have multiple rows (precursors) per peptidoform — keep best q
    diann_best = (
        df_diann.sort_values("diann_q")
        .groupby(["stripped_peptide_key", "diann_charge"])
        .first()
        .reset_index()
        .rename(columns={"diann_charge": "charge"})
    )

    merged = apex_cols.merge(
        diann_best[
            [
                "stripped_peptide_key",
                "charge",
                "diann_rt",
                "diann_rt_start",
                "diann_rt_stop",
                "diann_pred_rt",
                "diann_q",
            ]
        ],
        on=["stripped_peptide_key", "charge"],
        how="outer",
    )

    # MuMDIA apex rt is already in minutes (Sage outputs minutes)
    merged["mumdia_rt_min"] = merged["rt"]
    merged["diann_rt_min"] = merged["diann_rt"]  # DIA-NN RT also in minutes

    # delta_apex_rt: MuMDIA - DIA-NN (minutes)
    merged["delta_apex_rt_min"] = merged["mumdia_rt_min"] - merged["diann_rt_min"]
    merged["abs_delta_apex_rt_min"] = merged["delta_apex_rt_min"].abs()

    # 5. FDR group priority: mokapot_q_best → spectrum_q fallback
    merged["best_q"] = (
        merged["mokapot_q_best"]
        .astype("Float64")
        .combine_first(merged["spectrum_q"].astype("Float64"))
    )
    merged["fdr_group"] = merged["best_q"].apply(fdr_label)

    # Flag whether this peptidoform is present in each tool
    merged["in_diann"] = merged["diann_rt"].notna()
    merged["in_mumdia"] = merged["rt"].notna()

    return merged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def scatter_apex_rt(df: pd.DataFrame, outdir: str) -> None:
    """Residual plot: DIA-NN apex RT (x) vs MuMDIA − DIA-NN apex RT (y).
    All peptidoforms present in both tools; no q-value filtering."""
    sub = df.dropna(subset=["mumdia_rt_min", "diann_rt_min", "delta_apex_rt_min"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        sub["diann_rt_min"],
        sub["delta_apex_rt_min"],
        s=3,
        alpha=0.25,
        color="#1f77b4",
        rasterized=True,
    )
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(left=0)
    ax.set_xlabel("DIA-NN apex RT (min)")
    ax.set_ylabel("MuMDIA − DIA-NN apex RT (min)")
    ax.set_title(f"Apex RT residuals: MuMDIA vs DIA-NN  (n={len(sub):,})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scatter_apex_rt.png"), dpi=150)
    plt.close(fig)


def hist_delta_apex(df: pd.DataFrame, outdir: str) -> None:
    """Histogram of (MuMDIA - DIA-NN) apex RT per FDR group."""
    sub = df.dropna(subset=["delta_apex_rt_min"])

    fig, axes = plt.subplots(len(FDR_ORDER), 1, figsize=(7, 10), sharex=True)
    bins = np.linspace(-5, 5, 101)

    for ax, grp in zip(axes, FDR_ORDER):
        mask = sub["fdr_group"] == grp
        vals = sub.loc[mask, "delta_apex_rt_min"]
        if len(vals) == 0:
            ax.set_visible(False)
            continue
        ax.hist(vals, bins=bins, color=FDR_COLORS[grp], alpha=0.8, edgecolor="none")
        median = vals.median()
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.axvline(median, color="red", lw=1, ls="-", label=f"median={median:.2f}")
        ax.set_ylabel(f"{grp}\n(n={len(vals)})", fontsize=8)
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("MuMDIA apex RT − DIA-NN apex RT (min)")
    fig.suptitle("Apex RT difference by FDR group", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "hist_delta_apex_rt.png"), dpi=150)
    plt.close(fig)


def scatter_rt_error_vs_delta(df: pd.DataFrame, outdir: str) -> None:
    """Scatter: MuMDIA DeepLC prediction error vs apex RT delta."""
    col = "rt_prediction_error_abs_relative"
    if col not in df.columns:
        return
    sub = df.dropna(subset=[col, "delta_apex_rt_min", "in_diann"])
    sub = sub[sub["in_diann"]]
    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for grp in FDR_ORDER:
        mask = sub["fdr_group"] == grp
        if mask.sum() == 0:
            continue
        ax.scatter(
            sub.loc[mask, col],
            sub.loc[mask, "delta_apex_rt_min"],
            s=5,
            alpha=0.3,
            color=FDR_COLORS[grp],
            label=f"{grp} (n={mask.sum()})",
            rasterized=True,
        )

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("MuMDIA DeepLC relative RT error (abs)")
    ax.set_ylabel("MuMDIA − DIA-NN apex RT (min)")
    ax.set_title("DeepLC prediction error vs apex RT difference")
    ax.legend(markerscale=3, fontsize=7)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scatter_rt_error_vs_delta.png"), dpi=150)
    plt.close(fig)


def boxplot_delta_by_fdr(df: pd.DataFrame, outdir: str) -> None:
    """Violin/box plot: apex delta by FDR group."""
    sub = df.dropna(subset=["delta_apex_rt_min", "in_diann"])
    sub = sub[sub["in_diann"]]
    if len(sub) == 0:
        return

    groups = [grp for grp in FDR_ORDER if (sub["fdr_group"] == grp).sum() > 0]
    data = [
        sub.loc[sub["fdr_group"] == grp, "delta_apex_rt_min"].values for grp in groups
    ]
    colors = [FDR_COLORS[grp] for grp in groups]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{g}\n(n={len(d)})" for g, d in zip(groups, data)])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("MuMDIA − DIA-NN apex RT (min)")
    ax.set_title("Apex RT difference by MuMDIA FDR group")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "boxplot_delta_by_fdr.png"), dpi=150)
    plt.close(fig)


def scatter_diann_q_vs_delta(df: pd.DataFrame, outdir: str) -> None:
    """Scatter: DIA-NN q-value vs apex RT delta."""
    sub = df.dropna(subset=["diann_q", "delta_apex_rt_min"])
    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for grp in FDR_ORDER:
        mask = sub["fdr_group"] == grp
        if mask.sum() == 0:
            continue
        ax.scatter(
            sub.loc[mask, "diann_q"],
            sub.loc[mask, "delta_apex_rt_min"],
            s=5,
            alpha=0.3,
            color=FDR_COLORS[grp],
            label=f"{grp} (n={mask.sum()})",
            rasterized=True,
        )

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(0.01, color="grey", lw=0.7, ls=":", label="DIA-NN q=0.01")
    ax.set_xlabel("DIA-NN Q.Value")
    ax.set_ylabel("MuMDIA − DIA-NN apex RT (min)")
    ax.set_title("DIA-NN Q.Value vs apex RT difference (coloured by MuMDIA FDR)")
    ax.legend(markerscale=3, fontsize=7)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scatter_diann_q_vs_delta.png"), dpi=150)
    plt.close(fig)


def scatter_mumdia_vs_diann_q(df: pd.DataFrame, outdir: str) -> None:
    """Scatter: MuMDIA best q (mokapot or spectrum_q) vs DIA-NN Q.Value."""
    sub = df.dropna(subset=["best_q", "diann_q"])
    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        sub["diann_q"], sub["best_q"], s=4, alpha=0.2, color="#1f77b4", rasterized=True
    )
    # threshold lines
    for thresh, color in [(0.01, "green"), (0.05, "orange"), (0.10, "red")]:
        ax.axhline(
            thresh, color=color, lw=0.7, ls="--", alpha=0.8, label=f"MuMDIA q={thresh}"
        )
        ax.axvline(thresh, color=color, lw=0.7, ls=":", alpha=0.8)
    ax.set_xlabel("DIA-NN Q.Value")
    ax.set_ylabel("MuMDIA q-value (mokapot or sage spectrum_q)")
    ax.set_title("MuMDIA vs DIA-NN q-values per peptidoform")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scatter_mumdia_vs_diann_q.png"), dpi=150)
    plt.close(fig)


def scatter_apex_rt_windows(df: pd.DataFrame, outdir: str) -> None:
    """
    For peptidoforms where both are present, show MuMDIA RT margin window vs DIA-NN RT.
    Plots DIA-NN RT on x-axis, MuMDIA apex RT ± RT margins on y-axis as error bars.
    """
    needed = ["mumdia_rt_min", "diann_rt_min", "rt_lower_margin", "rt_higher_margin"]
    if not all(c in df.columns for c in needed):
        return
    sub = df.dropna(
        subset=["mumdia_rt_min", "diann_rt_min", "rt_lower_margin", "rt_higher_margin"]
    )
    if len(sub) == 0:
        return

    sub = sub.sort_values("mumdia_rt_min")
    # sample if too large
    if len(sub) > 2000:
        sub = sub.sample(2000, random_state=42).sort_values("mumdia_rt_min")

    fig, ax = plt.subplots(figsize=(7, 6))
    x = sub["diann_rt_min"].values
    y = sub["mumdia_rt_min"].values
    # margins are absolute RT positions; convert to deltas from apex
    yerr_low = (sub["mumdia_rt_min"] - sub["rt_lower_margin"]).clip(lower=0).values
    yerr_hi = (sub["rt_higher_margin"] - sub["mumdia_rt_min"]).clip(lower=0).values

    for grp in FDR_ORDER:
        mask = (sub["fdr_group"] == grp).values
        if mask.sum() == 0:
            continue
        ax.errorbar(
            x[mask],
            y[mask],
            yerr=[yerr_low[mask], yerr_hi[mask]],
            fmt="none",
            ecolor=FDR_COLORS[grp],
            alpha=0.25,
            linewidth=0.5,
        )
        ax.scatter(
            x[mask],
            y[mask],
            s=4,
            color=FDR_COLORS[grp],
            label=f"{grp} (n={mask.sum()})",
            zorder=3,
            rasterized=True,
        )

    lim = [0, max(x.max(), y.max()) * 1.02]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("DIA-NN apex RT (min)")
    ax.set_ylabel("MuMDIA apex RT ± margin (min)")
    ax.set_title("MuMDIA RT windows vs DIA-NN apex (sampled ≤2000)")
    ax.legend(markerscale=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scatter_apex_rt_windows.png"), dpi=150)
    plt.close(fig)


def write_summary(df: pd.DataFrame, outdir: str) -> None:
    rows = []
    for grp in FDR_ORDER:
        mask = df["fdr_group"] == grp
        sub = df[mask]
        in_diann = sub[sub["in_diann"]]
        delta = in_diann["delta_apex_rt_min"].dropna()
        rows.append(
            {
                "fdr_group": grp,
                "n_mumdia_peptidoforms": int(mask.sum()),
                "n_in_diann": len(in_diann),
                "pct_in_diann": 100 * len(in_diann) / len(sub) if len(sub) else np.nan,
                "delta_apex_rt_median_min": delta.median(),
                "delta_apex_rt_mean_min": delta.mean(),
                "delta_apex_rt_std_min": delta.std(),
                "delta_apex_rt_abs_median_min": delta.abs().median(),
                "pct_within_0.5min": (
                    100 * (delta.abs() < 0.5).sum() / len(delta)
                    if len(delta)
                    else np.nan
                ),
                "pct_within_1min": (
                    100 * (delta.abs() < 1.0).sum() / len(delta)
                    if len(delta)
                    else np.nan
                ),
                "pct_within_2min": (
                    100 * (delta.abs() < 2.0).sum() / len(delta)
                    if len(delta)
                    else np.nan
                ),
            }
        )
    # DIA-NN-only row
    diann_only = df[~df["in_mumdia"] & df["in_diann"]]
    rows.append(
        {
            "fdr_group": "diann_only",
            "n_mumdia_peptidoforms": 0,
            "n_in_diann": len(diann_only),
            "pct_in_diann": 100.0,
            "delta_apex_rt_median_min": np.nan,
            "delta_apex_rt_mean_min": np.nan,
            "delta_apex_rt_std_min": np.nan,
            "delta_apex_rt_abs_median_min": np.nan,
            "pct_within_0.5min": np.nan,
            "pct_within_1min": np.nan,
            "pct_within_2min": np.nan,
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        os.path.join(outdir, "summary.tsv"), sep="\t", index=False, float_format="%.4f"
    )
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mumdia-apex",
        default="runs/ecoli_splitmult1p5_calibonly/debug/df_fragment_max_peptide_after_ms2pip.tsv",
        help="df_fragment_max_peptide_after_ms2pip.tsv",
    )
    p.add_argument(
        "--mumdia-psms",
        default="runs/ecoli_splitmult1p5_calibonly/debug/df_psms_after_ms2pip.tsv",
        help="df_psms_after_ms2pip.tsv (with RT margin columns)",
    )
    p.add_argument(
        "--mokapot",
        default="runs/ecoli_splitmult1p5_calibonly/pipeline/xgboost.mokapot.psms.txt",
        help="Mokapot/XGBoost PSM-level q-values",
    )
    p.add_argument(
        "--diann", default="diann_results/report_with_xic.tsv", help="DIA-NN report TSV"
    )
    p.add_argument(
        "--outdir",
        default="runs/ecoli_splitmult1p5_calibonly/comparison/apex_comparison",
        help="Output directory",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading MuMDIA apex data...")
    df_apex = load_mumdia_apex(args.mumdia_apex)
    print(f"  {len(df_apex):,} peptidoforms in MuMDIA apex file")

    print("Loading MuMDIA PSM data (all, for margins)...")
    df_psms = load_mumdia_psms(args.mumdia_psms)
    print(f"  {len(df_psms):,} PSMs in MuMDIA PSM file")

    print("Loading mokapot scores...")
    df_mok = load_mokapot(args.mokapot)
    print(f"  {len(df_mok):,} PSMs with mokapot q-values")

    print("Loading DIA-NN report...")
    df_diann = load_diann(args.diann)
    print(f"  {len(df_diann):,} rows from DIA-NN")

    print("Building merged table...")
    merged = build_merged_table(df_apex, df_psms, df_mok, df_diann)
    print(f"  {len(merged):,} peptidoforms in merged table")
    print(f"  {merged['in_diann'].sum():,} present in DIA-NN")
    print(f"  {merged['in_mumdia'].sum():,} present in MuMDIA")
    in_both = (merged["in_diann"] & merged["in_mumdia"]).sum()
    print(f"  {in_both:,} present in both")

    out_tsv = os.path.join(args.outdir, "apex_comparison.tsv")
    merged.to_csv(out_tsv, sep="\t", index=False, float_format="%.6f")
    print(f"  Saved: {out_tsv}")

    print("Plotting...")
    scatter_apex_rt(merged, args.outdir)
    hist_delta_apex(merged, args.outdir)
    scatter_rt_error_vs_delta(merged, args.outdir)
    boxplot_delta_by_fdr(merged, args.outdir)
    scatter_diann_q_vs_delta(merged, args.outdir)
    scatter_mumdia_vs_diann_q(merged, args.outdir)
    scatter_apex_rt_windows(merged, args.outdir)

    print("\n=== Summary ===")
    write_summary(merged, args.outdir)

    print(f"\nAll outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
