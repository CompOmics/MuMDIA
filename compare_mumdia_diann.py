#!/usr/bin/env python3
"""
MuMDIA vs DIA-NN Comparison Script

Generates systematic comparison of identification, scoring, and intermediate
features between MuMDIA and DIA-NN on the same dataset.

Usage:
    python compare_mumdia_diann.py \
        --mumdia-dir test_data/results \
        --diann-report diann_results/report.tsv \
        --output-dir comparison_output
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Try to import matplotlib_venn, fall back to manual implementation
try:
    from matplotlib_venn import venn2

    HAS_VENN = True
except ImportError:
    HAS_VENN = False


def strip_modifications(peptide: str) -> str:
    """Remove modification annotations like [Carbamidomethyl] from peptide string."""
    return re.sub(r"\[.*?\]", "", peptide)


def extract_uniprot_accession(proteins: str) -> str:
    """Extract UniProt accession from MuMDIA protein string like sp|P0A7M9|RL31_ECOLI."""
    match = re.search(r"sp\|([A-Z0-9]+)\|", str(proteins))
    return match.group(1) if match else str(proteins).split("|")[0] if proteins else ""


# =============================================================================
# Data Loading
# =============================================================================


def load_mumdia_results(results_dir: str) -> dict:
    """Load MuMDIA results from a results directory."""
    data = {}

    # Mokapot peptide-level results
    pep_path = os.path.join(results_dir, "mokapot.peptides.txt")
    if os.path.exists(pep_path):
        df = pd.read_csv(pep_path, sep="\t")
        df["stripped_peptide"] = df["Peptide"].apply(strip_modifications)
        if "Proteins" in df.columns:
            df["uniprot_accession"] = df["Proteins"].apply(extract_uniprot_accession)
        data["peptides"] = df
        print(f"  MuMDIA peptides: {len(df)} total")

    # Mokapot PSM-level results
    psm_path = os.path.join(results_dir, "mokapot.psms.txt")
    if os.path.exists(psm_path):
        data["psms"] = pd.read_csv(psm_path, sep="\t")
        print(f"  MuMDIA PSMs: {len(data['psms'])} total")

    # PIN file (features)
    pin_path = os.path.join(results_dir, "outfile.pin")
    if os.path.exists(pin_path):
        data["pin"] = pd.read_csv(pin_path, sep="\t", low_memory=False)
        print(f"  MuMDIA PIN features: {data['pin'].shape[1]} columns")

    return data


def load_diann_results(report_path: str) -> dict:
    """Load DIA-NN results from report.tsv."""
    data = {}

    df = pd.read_csv(report_path, sep="\t", low_memory=False)
    # Filter to non-decoy if Decoy column exists
    if "Decoy" in df.columns:
        df = df[df["Decoy"] == 0]
    data["report"] = df
    print(f"  DIA-NN precursors: {len(df)} total")

    # Best peptide-level: group by stripped sequence, take best q-value
    best = (
        df.groupby("Stripped.Sequence")
        .agg(
            diann_qvalue=("Q.Value", "min"),
            diann_cscore=("CScore", "max"),
            diann_rt=("RT", "first"),
            diann_predicted_rt=(
                "Predicted.RT",
                "first",
            ),
            diann_quantity=("Precursor.Quantity", "sum"),
            diann_protein=("Protein.Ids", "first"),
        )
        .reset_index()
    )

    if "PEP" in df.columns:
        pep_agg = df.groupby("Stripped.Sequence")["PEP"].min().reset_index()
        best = best.merge(pep_agg, on="Stripped.Sequence", how="left")
        best.rename(columns={"PEP": "diann_pep"}, inplace=True)

    data["peptides"] = best
    print(f"  DIA-NN unique peptides: {len(best)}")

    return data


def build_matched_dataframe(mumdia: dict, diann: dict, fdr: float) -> pd.DataFrame:
    """Build outer-joined DataFrame matching peptides between tools."""
    m_pep = mumdia["peptides"].copy()
    d_pep = diann["peptides"].copy()

    # Filter by FDR
    m_pass = m_pep[m_pep["mokapot q-value"] <= fdr]
    d_pass = d_pep[d_pep["diann_qvalue"] <= fdr]

    matched = m_pass.merge(
        d_pass,
        left_on="stripped_peptide",
        right_on="Stripped.Sequence",
        how="outer",
        indicator=True,
    )

    matched["category"] = matched["_merge"].map(
        {"left_only": "MuMDIA-only", "right_only": "DIA-NN-only", "both": "Overlap"}
    )

    return matched


# =============================================================================
# Identification Comparison
# =============================================================================


def plot_peptide_venn(matched: pd.DataFrame, output_dir: str, fdr: float):
    """Plot Venn diagram of peptide overlap."""
    mumdia_only = (matched["category"] == "MuMDIA-only").sum()
    diann_only = (matched["category"] == "DIA-NN-only").sum()
    overlap = (matched["category"] == "Overlap").sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    if HAS_VENN:
        venn2(
            subsets=(mumdia_only, diann_only, overlap),
            set_labels=("MuMDIA", "DIA-NN"),
            ax=ax,
        )
    else:
        # Manual circles
        ax.text(0.3, 0.5, f"MuMDIA\nonly\n{mumdia_only}", ha="center", fontsize=14)
        ax.text(
            0.5, 0.5, f"Overlap\n{overlap}", ha="center", fontsize=14, weight="bold"
        )
        ax.text(0.7, 0.5, f"DIA-NN\nonly\n{diann_only}", ha="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_title(f"Peptide Identification Overlap at {fdr*100:.1f}% FDR")
    total = mumdia_only + diann_only + overlap
    jaccard = overlap / total if total > 0 else 0
    ax.text(
        0.5,
        0.02,
        f"Jaccard: {jaccard:.3f} | Total unique: {total}",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )
    fig.savefig(
        os.path.join(output_dir, "peptide_venn.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    return {"mumdia_only": mumdia_only, "diann_only": diann_only, "overlap": overlap}


def plot_cumulative_ids(mumdia: dict, diann: dict, output_dir: str):
    """Plot cumulative peptide identifications vs FDR threshold."""
    thresholds = np.logspace(-4, -0.5, 50)

    m_pep = mumdia["peptides"]
    d_pep = diann["peptides"]

    m_counts = [
        m_pep["stripped_peptide"][m_pep["mokapot q-value"] <= t].nunique()
        for t in thresholds
    ]
    d_counts = [
        d_pep["Stripped.Sequence"][d_pep["diann_qvalue"] <= t].nunique()
        for t in thresholds
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds * 100, m_counts, "b-", label="MuMDIA", linewidth=2)
    ax.plot(thresholds * 100, d_counts, "r-", label="DIA-NN", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("FDR threshold (%)")
    ax.set_ylabel("Unique peptides identified")
    ax.set_title("Cumulative Peptide Identifications vs FDR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="1% FDR")
    fig.savefig(
        os.path.join(output_dir, "cumulative_ids.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_overlap_vs_fdr(mumdia: dict, diann: dict, output_dir: str):
    """Plot overlap, MuMDIA-only, DIA-NN-only counts across FDR thresholds."""
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    m_pep = mumdia["peptides"]
    d_pep = diann["peptides"]

    overlap_counts, m_only_counts, d_only_counts = [], [], []
    for t in thresholds:
        m_set = set(m_pep.loc[m_pep["mokapot q-value"] <= t, "stripped_peptide"])
        d_set = set(d_pep.loc[d_pep["diann_qvalue"] <= t, "Stripped.Sequence"])
        overlap_counts.append(len(m_set & d_set))
        m_only_counts.append(len(m_set - d_set))
        d_only_counts.append(len(d_set - m_set))

    fig, ax = plt.subplots(figsize=(8, 6))
    x = [t * 100 for t in thresholds]
    ax.plot(x, overlap_counts, "g-o", label="Overlap", linewidth=2)
    ax.plot(x, m_only_counts, "b-s", label="MuMDIA-only", linewidth=2)
    ax.plot(x, d_only_counts, "r-^", label="DIA-NN-only", linewidth=2)
    ax.set_xlabel("FDR threshold (%)")
    ax.set_ylabel("Peptide count")
    ax.set_title("Peptide Overlap vs FDR Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(output_dir, "overlap_vs_fdr.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


# =============================================================================
# Scoring Comparison
# =============================================================================


def plot_qvalue_scatter(matched: pd.DataFrame, output_dir: str, fdr: float):
    """Scatter plot of q-values for peptides found by both tools."""
    both = matched[matched["category"] == "Overlap"].copy()
    if len(both) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        both["mokapot q-value"],
        both["diann_qvalue"],
        alpha=0.3,
        s=10,
        c="steelblue",
    )
    lims = [1e-7, max(fdr * 2, 0.05)]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x")
    ax.axhline(fdr, color="red", linestyle=":", alpha=0.5)
    ax.axvline(fdr, color="blue", linestyle=":", alpha=0.5)
    ax.set_xlabel("MuMDIA q-value (mokapot)")
    ax.set_ylabel("DIA-NN q-value")
    ax.set_title(f"Q-value Comparison (n={len(both)} shared peptides)")
    r, p = stats.spearmanr(
        both["mokapot q-value"].fillna(1), both["diann_qvalue"].fillna(1)
    )
    ax.text(0.05, 0.95, f"Spearman r = {r:.3f}", transform=ax.transAxes, fontsize=11)
    ax.legend()
    fig.savefig(
        os.path.join(output_dir, "qvalue_scatter.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_score_by_category(matched: pd.DataFrame, mumdia: dict, output_dir: str):
    """Box plots of MuMDIA mokapot scores by overlap category."""
    # Get mokapot scores for MuMDIA peptides
    if "psms" not in mumdia:
        return

    psms = mumdia["psms"].copy()
    psms["stripped_peptide"] = psms["Peptide"].apply(strip_modifications)

    # Best score per peptide
    best_scores = psms.groupby("stripped_peptide")["mokapot score"].max().reset_index()
    merged = matched.merge(best_scores, on="stripped_peptide", how="left")

    cats = ["MuMDIA-only", "Overlap"]
    data_by_cat = []
    labels = []
    for c in cats:
        vals = merged.loc[merged["category"] == c, "mokapot score"].dropna()
        if len(vals) > 0:
            data_by_cat.append(vals)
            labels.append(c)

    if not data_by_cat:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data_by_cat, labels=labels, patch_artist=True)
    colors = ["#4477AA", "#66CCEE", "#EE6677"]
    for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)
    ax.set_ylabel("MuMDIA mokapot score")
    ax.set_title("MuMDIA Score Distribution by Overlap Category")
    fig.savefig(
        os.path.join(output_dir, "score_by_category.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_qvalue_histograms(matched: pd.DataFrame, output_dir: str):
    """Overlaid histograms of q-values for shared peptides."""
    both = matched[matched["category"] == "Overlap"]
    if len(both) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.logspace(-6, 0, 50)
    ax.hist(
        both["mokapot q-value"].clip(lower=1e-7),
        bins=bins,
        alpha=0.6,
        label="MuMDIA",
        color="steelblue",
    )
    ax.hist(
        both["diann_qvalue"].clip(lower=1e-7),
        bins=bins,
        alpha=0.6,
        label="DIA-NN",
        color="coral",
    )
    ax.set_xscale("log")
    ax.set_xlabel("q-value")
    ax.set_ylabel("Count")
    ax.set_title("Q-value Distributions (shared peptides)")
    ax.legend()
    fig.savefig(
        os.path.join(output_dir, "qvalue_histograms.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


# =============================================================================
# Intermediate Step Comparison
# =============================================================================


def plot_rt_comparison(matched: pd.DataFrame, mumdia: dict, output_dir: str):
    """Side-by-side RT observed vs predicted for both tools."""
    both = matched[matched["category"] == "Overlap"].copy()

    # Get MuMDIA RT data from PIN file
    if "pin" not in mumdia or len(both) == 0:
        return

    pin = mumdia["pin"].copy()
    pin["stripped_peptide"] = pin["Peptide"].apply(strip_modifications)

    # Best RT per peptide from PIN
    rt_data = (
        pin.groupby("stripped_peptide")
        .agg(
            mumdia_rt=("rt_min", "first")
            if "rt_min" in pin.columns
            else ("rt_max", "first"),
            mumdia_rt_pred=(
                "rt_predictions_min"
                if "rt_predictions_min" in pin.columns
                else "rt_predictions_max",
                "first",
            ),
        )
        .reset_index()
    )

    merged = both.merge(rt_data, on="stripped_peptide", how="inner")
    if len(merged) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MuMDIA RT
    if "mumdia_rt" in merged.columns and "mumdia_rt_pred" in merged.columns:
        ax = axes[0]
        ax.scatter(merged["mumdia_rt"], merged["mumdia_rt_pred"], alpha=0.2, s=5)
        lims = [
            min(merged["mumdia_rt"].min(), merged["mumdia_rt_pred"].min()),
            max(merged["mumdia_rt"].max(), merged["mumdia_rt_pred"].max()),
        ]
        ax.plot(lims, lims, "r--", alpha=0.5)
        ax.set_xlabel("Observed RT")
        ax.set_ylabel("Predicted RT (DeepLC)")
        ax.set_title("MuMDIA: RT Prediction")

    # DIA-NN RT
    if "diann_rt" in merged.columns and "diann_predicted_rt" in merged.columns:
        ax = axes[1]
        valid = merged.dropna(subset=["diann_rt", "diann_predicted_rt"])
        ax.scatter(valid["diann_rt"], valid["diann_predicted_rt"], alpha=0.2, s=5)
        if len(valid) > 0:
            lims = [
                min(valid["diann_rt"].min(), valid["diann_predicted_rt"].min()),
                max(valid["diann_rt"].max(), valid["diann_predicted_rt"].max()),
            ]
            ax.plot(lims, lims, "r--", alpha=0.5)
        ax.set_xlabel("Observed RT")
        ax.set_ylabel("Predicted RT")
        ax.set_title("DIA-NN: RT Prediction")

    fig.suptitle("Retention Time Predictions (shared peptides)")
    fig.savefig(
        os.path.join(output_dir, "rt_comparison.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_feature_importance(mumdia: dict, output_dir: str):
    """Bar chart of top PIN features by target/decoy discrimination."""
    if "pin" not in mumdia:
        return

    pin = mumdia["pin"]
    if "Label" not in pin.columns:
        return

    # Get numeric feature columns (exclude metadata)
    exclude = {
        "ScanNr",
        "Label",
        "Peptide",
        "Proteins",
        "ExpMass",
        "CalcMass",
        "SpecId",
        "filename",
        "scannr",
    }
    feat_cols = [
        c
        for c in pin.columns
        if c not in exclude and pin[c].dtype in [np.float64, np.int64, np.float32]
    ]

    targets = pin[pin["Label"] == 1]
    decoys = pin[pin["Label"] == -1]

    # Compute discrimination: abs(mean_target - mean_decoy) / pooled_std
    scores = {}
    for col in feat_cols:
        t_vals = targets[col].dropna()
        d_vals = decoys[col].dropna()
        if len(t_vals) > 1 and len(d_vals) > 1:
            pooled_std = np.sqrt((t_vals.var() + d_vals.var()) / 2)
            if pooled_std > 0:
                scores[col] = abs(t_vals.mean() - d_vals.mean()) / pooled_std

    if not scores:
        return

    # Top 20
    sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
    names = [f[0] for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]

    # Color by source
    colors = []
    for n in names:
        if n.startswith("diann_"):
            colors.append("#EE6677")  # DIA-NN features
        elif "rt_prediction" in n or "rt_predictions" in n:
            colors.append("#44BB99")  # DeepLC
        elif "correlation" in n or "mse_avg" in n:
            colors.append("#EEDD88")  # MS2PIP/correlations
        else:
            colors.append("#4477AA")  # Sage

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Target/Decoy Discrimination (Cohen's d)")
    ax.set_title("Top 20 MuMDIA PIN Features by Discrimination")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#4477AA", label="Sage"),
        Patch(facecolor="#44BB99", label="DeepLC"),
        Patch(facecolor="#EEDD88", label="MS2PIP/Correlations"),
        Patch(facecolor="#EE6677", label="DIA-NN features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    fig.savefig(
        os.path.join(output_dir, "feature_importance.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


# =============================================================================
# Summary
# =============================================================================


def print_summary(mumdia: dict, diann: dict, matched: pd.DataFrame, fdr: float):
    """Print comparison summary."""
    m_pep = mumdia["peptides"]
    d_pep = diann["peptides"]

    m_count = (m_pep["mokapot q-value"] <= fdr).sum()
    d_count = (d_pep["diann_qvalue"] <= fdr).sum()
    overlap = (matched["category"] == "Overlap").sum()
    m_only = (matched["category"] == "MuMDIA-only").sum()
    d_only = (matched["category"] == "DIA-NN-only").sum()
    total = m_only + d_only + overlap
    jaccard = overlap / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  MuMDIA vs DIA-NN Comparison Summary ({fdr*100:.1f}% FDR)")
    print(f"{'='*60}")
    print(f"  MuMDIA peptides:     {m_count:>8}")
    print(f"  DIA-NN peptides:     {d_count:>8}")
    print(f"  Overlap:             {overlap:>8}")
    print(f"  MuMDIA-only:         {m_only:>8}")
    print(f"  DIA-NN-only:         {d_only:>8}")
    print(f"  Jaccard index:       {jaccard:>8.3f}")
    print(f"  Overlap/MuMDIA:      {overlap / m_count * 100 if m_count else 0:>7.1f}%")
    print(f"  Overlap/DIA-NN:      {overlap / d_count * 100 if d_count else 0:>7.1f}%")
    print(f"{'='*60}")


def save_matched_tsv(matched: pd.DataFrame, output_dir: str):
    """Save the matched peptide table for downstream analysis."""
    out_path = os.path.join(output_dir, "matched_peptides.tsv")
    matched.to_csv(out_path, sep="\t", index=False)
    print(f"  Saved matched peptide table to {out_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compare MuMDIA and DIA-NN results on the same dataset."
    )
    parser.add_argument("--mumdia-dir", required=True, help="MuMDIA results directory")
    parser.add_argument("--diann-report", required=True, help="DIA-NN report.tsv file")
    parser.add_argument(
        "--output-dir", default="comparison_output", help="Output directory for plots"
    )
    parser.add_argument(
        "--fdr", type=float, default=0.01, help="FDR threshold (default: 0.01)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading MuMDIA results...")
    mumdia = load_mumdia_results(args.mumdia_dir)

    print("Loading DIA-NN results...")
    diann = load_diann_results(args.diann_report)

    if "peptides" not in mumdia or "peptides" not in diann:
        print("ERROR: Could not load peptide data from both tools.")
        sys.exit(1)

    print(f"\nBuilding matched DataFrame at {args.fdr*100:.1f}% FDR...")
    matched = build_matched_dataframe(mumdia, diann, args.fdr)

    # === Summary ===
    print_summary(mumdia, diann, matched, args.fdr)
    save_matched_tsv(matched, args.output_dir)

    # === Identification plots ===
    print("\nGenerating identification comparison plots...")
    plot_peptide_venn(matched, args.output_dir, args.fdr)
    plot_cumulative_ids(mumdia, diann, args.output_dir)
    plot_overlap_vs_fdr(mumdia, diann, args.output_dir)

    # === Scoring plots ===
    print("Generating scoring comparison plots...")
    try:
        plot_qvalue_scatter(matched, args.output_dir, args.fdr)
    except Exception as e:
        print(f"  Warning: qvalue_scatter failed: {e}")
    try:
        plot_qvalue_histograms(matched, args.output_dir)
    except Exception as e:
        print(f"  Warning: qvalue_histograms failed: {e}")
    try:
        plot_score_by_category(matched, mumdia, args.output_dir)
    except Exception as e:
        print(f"  Warning: score_by_category failed: {e}")

    # === Intermediate plots ===
    print("Generating intermediate step comparison plots...")
    try:
        plot_rt_comparison(matched, mumdia, args.output_dir)
    except Exception as e:
        print(f"  Warning: rt_comparison failed: {e}")
    try:
        plot_feature_importance(mumdia, args.output_dir)
    except Exception as e:
        print(f"  Warning: feature_importance failed: {e}")

    print(f"\nAll plots saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
