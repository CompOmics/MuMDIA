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
# Feature-Level Comparison (per-peptidoform)
# =============================================================================


def parse_diann_fragment_correlations(report_df: pd.DataFrame) -> pd.DataFrame:
    """Parse DIA-NN Fragment.Correlations semicolon-delimited column into statistics."""
    rows = []
    for _, r in report_df.iterrows():
        corrs_str = str(r.get("Fragment.Correlations", ""))
        corrs = [float(x) for x in corrs_str.split(";") if x.strip() and x.strip() != "0"]

        row = {
            "precursor_id": r.get("Precursor.Id", ""),
            "stripped_peptide": r.get("Stripped.Sequence", ""),
            "charge": int(r.get("Precursor.Charge", 0)),
            "diann_qvalue": r.get("Q.Value", 1.0),
            "diann_cscore": r.get("CScore", 0.0),
            "diann_evidence": r.get("Evidence", 0.0),
            "diann_spectrum_similarity": r.get("Spectrum.Similarity", 0.0),
            "diann_ms1_corr": r.get("Ms1.Profile.Corr", 0.0),
            "diann_rt": r.get("RT", 0.0),
            "diann_predicted_rt": r.get("Predicted.RT", 0.0),
            "diann_quantity": r.get("Precursor.Quantity", 0.0),
            "diann_n_fragments": len(corrs),
            "diann_best_frag_corr": max(corrs) if corrs else 0.0,
            "diann_mean_frag_corr": np.mean(corrs) if corrs else 0.0,
            "diann_median_frag_corr": np.median(corrs) if corrs else 0.0,
            "diann_n_good_frags": sum(1 for c in corrs if c > 0.7),
        }

        # Parse Fragment.Quant.Raw
        quant_str = str(r.get("Fragment.Quant.Raw", ""))
        quants = [float(x) for x in quant_str.split(";") if x.strip()]
        nonzero_quants = [q for q in quants if q > 0]
        row["diann_n_quant_fragments"] = len(nonzero_quants)
        row["diann_total_quant"] = sum(nonzero_quants)
        row["diann_max_quant"] = max(nonzero_quants) if nonzero_quants else 0.0

        rows.append(row)
    return pd.DataFrame(rows)


def match_features(mumdia_pin: pd.DataFrame, diann_features: pd.DataFrame) -> pd.DataFrame:
    """Match MuMDIA PIN file features with DIA-NN features by peptide/charge."""
    # Build stripped peptide from MuMDIA PIN
    pin = mumdia_pin.copy()
    pin["stripped_peptide"] = pin["Peptide"].apply(strip_modifications)

    # Merge on peptide + charge
    merged = pin.merge(
        diann_features,
        on=["stripped_peptide", "charge"],
        how="inner",
        suffixes=("_mumdia", "_diann"),
    )
    return merged


def compare_fragment_correlations(merged: pd.DataFrame, output_dir: str):
    """Compare fragment correlation features between MuMDIA and DIA-NN."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. DIA-NN best frag corr vs MuMDIA top correlation individual
    ax = axes[0, 0]
    if "top_correlation_individual_1" in merged.columns:
        x = merged["diann_best_frag_corr"]
        y = merged["top_correlation_individual_1"]
        valid = x.notna() & y.notna() & (x > 0) & (y > 0)
        ax.scatter(x[valid], y[valid], alpha=0.1, s=2)
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
        r = x[valid].corr(y[valid])
        ax.set_xlabel("DIA-NN best fragment correlation")
        ax.set_ylabel("MuMDIA top_correlation_individual_1")
        ax.set_title(f"Best Fragment Correlation (r={r:.3f})")

    # 2. DIA-NN mean frag corr vs MuMDIA median correlation
    ax = axes[0, 1]
    if "distribution_correlation_individual_50" in merged.columns:
        x = merged["diann_mean_frag_corr"]
        y = merged["distribution_correlation_individual_50"]
        valid = x.notna() & y.notna()
        ax.scatter(x[valid], y[valid], alpha=0.1, s=2)
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
        r = x[valid].corr(y[valid])
        ax.set_xlabel("DIA-NN mean fragment correlation")
        ax.set_ylabel("MuMDIA median individual correlation")
        ax.set_title(f"Mean/Median Fragment Correlation (r={r:.3f})")

    # 3. DIA-NN CScore vs MuMDIA peptide_q_min
    ax = axes[0, 2]
    x = merged["diann_cscore"]
    y = -np.log10(merged["peptide_q_min"].clip(1e-10))
    valid = x.notna() & y.notna()
    ax.scatter(x[valid], y[valid], alpha=0.1, s=2)
    ax.set_xlabel("DIA-NN CScore")
    ax.set_ylabel("-log10(MuMDIA peptide_q_min)")
    ax.set_title("Score Comparison")

    # 4. DIA-NN n_good_frags vs MuMDIA matched_peaks_max
    ax = axes[1, 0]
    if "matched_peaks_max" in merged.columns:
        x = merged["diann_n_good_frags"]
        y = merged["matched_peaks_max"]
        valid = x.notna() & y.notna()
        ax.scatter(x[valid], y[valid], alpha=0.1, s=2)
        r = x[valid].corr(y[valid])
        ax.set_xlabel("DIA-NN fragments with corr>0.7")
        ax.set_ylabel("MuMDIA matched_peaks_max")
        ax.set_title(f"Fragment Count Comparison (r={r:.3f})")

    # 5. Histogram: DIA-NN fragment correlations vs MuMDIA diann_features
    ax = axes[1, 1]
    if "diann_pearson_correlations_top_12_0" in merged.columns:
        diann_corrs = merged["diann_best_frag_corr"].dropna()
        mumdia_corrs = merged["diann_pearson_correlations_top_12_0"].dropna()
        ax.hist(diann_corrs, bins=50, alpha=0.5, density=True, label="DIA-NN native")
        ax.hist(mumdia_corrs, bins=50, alpha=0.5, density=True, label="MuMDIA DIA-NN features")
        ax.legend()
        ax.set_xlabel("Best fragment correlation")
        ax.set_ylabel("Density")
        ax.set_title("Fragment Correlation Distribution")

    # 6. DIA-NN q-value vs MuMDIA q-value for overlapping peptides
    ax = axes[1, 2]
    x = -np.log10(merged["diann_qvalue"].clip(1e-10))
    y_col = "peptide_q_min" if "peptide_q_min" in merged.columns else None
    if y_col:
        y = -np.log10(merged[y_col].clip(1e-10))
        valid = x.notna() & y.notna()
        ax.scatter(x[valid], y[valid], alpha=0.1, s=2)
        ax.plot([0, 10], [0, 10], "r--", alpha=0.5)
        ax.set_xlabel("-log10(DIA-NN q-value)")
        ax.set_ylabel("-log10(MuMDIA peptide_q_min)")
        ax.set_title("Q-value Comparison (same peptides)")

    fig.suptitle("Feature-Level Comparison: MuMDIA vs DIA-NN", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_xic_features(merged: pd.DataFrame, output_dir: str):
    """Compare XIC extraction features with DIA-NN fragment correlations."""
    xic_cols = [c for c in merged.columns if c.startswith("xic_")]
    if not xic_cols:
        print("  No XIC features in PIN file, skipping XIC comparison")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. XIC best coelution vs DIA-NN best frag corr
    ax = axes[0, 0]
    if "xic_best_coelution" in merged.columns:
        x = merged["diann_best_frag_corr"]
        y = merged["xic_best_coelution"]
        valid = x.notna() & y.notna() & (x > 0)
        ax.scatter(x[valid], y[valid], alpha=0.1, s=2)
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
        r = x[valid].corr(y[valid])
        ax.set_xlabel("DIA-NN best fragment correlation")
        ax.set_ylabel("MuMDIA xic_best_coelution")
        ax.set_title(f"XIC Co-elution vs DIA-NN (r={r:.3f})")

    # 2. XIC coverage histogram by category
    ax = axes[0, 1]
    if "xic_coverage" in merged.columns:
        for cat, color in [("good", "green"), ("bad", "red")]:
            if cat == "good":
                mask = merged["diann_qvalue"] < 0.01
            else:
                mask = merged["diann_qvalue"] >= 0.01
            vals = merged.loc[mask, "xic_coverage"].dropna()
            ax.hist(vals, bins=50, alpha=0.5, label=f"DIA-NN q<0.01: {cat}", color=color, density=True)
        ax.set_xlabel("XIC coverage")
        ax.set_ylabel("Density")
        ax.set_title("XIC Coverage by DIA-NN Quality")
        ax.legend()

    # 3. XIC n_detected_scans vs DIA-NN fragment count
    ax = axes[0, 2]
    if "xic_n_detected_scans" in merged.columns:
        x = merged["diann_n_good_frags"]
        y = merged["xic_n_detected_scans"]
        valid = x.notna() & y.notna()
        ax.scatter(x[valid], y[valid], alpha=0.1, s=2)
        ax.set_xlabel("DIA-NN fragments with corr>0.7")
        ax.set_ylabel("MuMDIA XIC detected scans")
        ax.set_title("XIC Detection vs DIA-NN Fragments")

    # 4. XIC matching window scans
    ax = axes[1, 0]
    if "xic_matching_window_scans" in merged.columns:
        vals = merged["xic_matching_window_scans"].dropna()
        ax.hist(vals, bins=50, color="steelblue")
        ax.set_xlabel("Matching isolation window scans")
        ax.set_ylabel("Count")
        ax.set_title(f"XIC Window Scan Count (mean={vals.mean():.0f})")

    # 5. XIC vs prediction
    ax = axes[1, 1]
    if "xic_vs_prediction" in merged.columns:
        vals = merged["xic_vs_prediction"].dropna()
        nonzero = vals[vals != 0]
        ax.hist(nonzero, bins=50, color="coral")
        ax.set_xlabel("XIC vs MS2PIP prediction correlation")
        ax.set_ylabel("Count")
        ax.set_title(f"XIC vs Prediction (non-zero mean={nonzero.mean():.3f})")

    # 6. XIC n_active_fragments distribution
    ax = axes[1, 2]
    if "xic_n_active_fragments" in merged.columns:
        vals = merged["xic_n_active_fragments"].dropna()
        ax.hist(vals, bins=50, color="mediumpurple")
        ax.set_xlabel("Active fragments in XIC")
        ax.set_ylabel("Count")
        ax.set_title(f"XIC Active Fragments (mean={vals.mean():.1f})")

    fig.suptitle("XIC Feature Analysis: MuMDIA vs DIA-NN", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "xic_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_diann_native_vs_mumdia(merged: pd.DataFrame, output_dir: str):
    """Compare DIA-NN's native features with MuMDIA's DIA-NN-style features."""
    diann_mumdia_cols = [c for c in merged.columns if c.startswith("diann_pearson")]
    if not diann_mumdia_cols:
        print("  No MuMDIA DIA-NN features, skipping")
        return

    n_cols = min(12, len(diann_mumdia_cols))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes_flat = axes.flatten()

    for idx in range(n_cols):
        col = f"diann_pearson_correlations_top_12_{idx}"
        col_fz = f"diann_pearson_correlations_top_12_{idx}_fz"
        ax = axes_flat[idx]

        if col in merged.columns:
            vals_oo = merged[col].dropna()
            ax.hist(vals_oo, bins=50, alpha=0.5, label="overlap_only", density=True)
        if col_fz in merged.columns:
            vals_fz = merged[col_fz].dropna()
            ax.hist(vals_fz, bins=50, alpha=0.5, label="fill_zero", density=True)
        ax.set_title(f"Top {idx+1} frag corr", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle("MuMDIA's DIA-NN-style Fragment Correlations (overlap_only vs fill_zero)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "diann_features_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_feature_summary_table(merged: pd.DataFrame, output_dir: str):
    """Generate a comprehensive feature statistics table comparing both tools."""
    rows = []

    # MuMDIA features to summarize
    mumdia_features = {
        "peptide_q_min": "Sage peptide q-value (min)",
        "hyperscore_max": "Sage hyperscore (max)",
        "matched_peaks_max": "Matched peaks (max)",
        "matched_intensity_pct_max": "Matched intensity % (max)",
        "rt_prediction_error_abs_min": "RT prediction error (min)",
        "top_correlation_individual_1": "Top PSM-prediction corr",
        "distribution_correlation_individual_50": "Median PSM-prediction corr",
        "top_correlation_matrix_psm_ids_1": "Top PSM-PSM corr",
        "top_correlation_matrix_frag_ids_1": "Top frag-frag corr",
        "mse_avg_pred_intens_1": "MSE pred vs observed intensity",
        "xic_best_coelution": "XIC best co-elution",
        "xic_coverage": "XIC coverage",
        "xic_n_active_fragments": "XIC active fragments",
        "diann_pearson_correlations_top_12_0": "DIA-NN top-1 frag corr (MuMDIA)",
        "diann_weighted_auc": "DIA-NN weighted AUC (MuMDIA)",
    }

    diann_features = {
        "diann_qvalue": "DIA-NN q-value",
        "diann_cscore": "DIA-NN CScore",
        "diann_evidence": "DIA-NN Evidence",
        "diann_spectrum_similarity": "DIA-NN Spectrum Similarity",
        "diann_best_frag_corr": "DIA-NN best fragment corr",
        "diann_mean_frag_corr": "DIA-NN mean fragment corr",
        "diann_median_frag_corr": "DIA-NN median fragment corr",
        "diann_n_good_frags": "DIA-NN fragments corr>0.7",
        "diann_n_quant_fragments": "DIA-NN quantified fragments",
        "diann_ms1_corr": "DIA-NN MS1 profile correlation",
    }

    for col, label in {**mumdia_features, **diann_features}.items():
        if col not in merged.columns:
            continue
        vals = merged[col].dropna()
        source = "MuMDIA" if col in mumdia_features else "DIA-NN"
        rows.append({
            "Source": source,
            "Feature": label,
            "Column": col,
            "Mean": vals.mean(),
            "Median": vals.median(),
            "Std": vals.std(),
            "Min": vals.min(),
            "Max": vals.max(),
            "% Zero": (vals == 0).mean() * 100,
            "N": len(vals),
        })

    df_summary = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "feature_summary.tsv")
    df_summary.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    print(f"  Feature summary saved to {out_path}")
    print(f"\n  Feature Summary ({len(merged)} matched peptidoforms):")
    print(f"  {'Source':<10} {'Feature':<40} {'Mean':>10} {'Median':>10} {'% Zero':>8}")
    print(f"  {'-'*80}")
    for _, row in df_summary.iterrows():
        print(f"  {row['Source']:<10} {row['Feature']:<40} {row['Mean']:>10.4f} {row['Median']:>10.4f} {row['% Zero']:>7.1f}%")

    return df_summary


def diagnose_feature_gaps(merged: pd.DataFrame, output_dir: str):
    """Identify specific peptidoforms where MuMDIA features are much worse than DIA-NN."""
    if "diann_best_frag_corr" not in merged.columns:
        return

    # Peptides where DIA-NN has high correlation but MuMDIA has low
    diann_good = merged["diann_best_frag_corr"] > 0.8
    mumdia_bad_corr = merged.get("top_correlation_individual_1", pd.Series(dtype=float)) < 0.3
    mumdia_bad_xic = merged.get("xic_best_coelution", pd.Series(dtype=float)) < 0.1

    gap_peptides = merged[diann_good & (mumdia_bad_corr | mumdia_bad_xic)].copy()

    if len(gap_peptides) > 0:
        cols_to_save = [
            "stripped_peptide", "charge",
            "diann_qvalue", "diann_best_frag_corr", "diann_mean_frag_corr",
            "diann_n_good_frags", "diann_cscore",
        ]
        mumdia_cols = [
            "peptide_q_min", "top_correlation_individual_1",
            "matched_peaks_max", "xic_best_coelution", "xic_coverage",
            "xic_n_active_fragments", "xic_n_detected_scans",
        ]
        cols_to_save += [c for c in mumdia_cols if c in merged.columns]

        gap_peptides = gap_peptides[[c for c in cols_to_save if c in gap_peptides.columns]]
        gap_peptides = gap_peptides.sort_values("diann_best_frag_corr", ascending=False)

        out_path = os.path.join(output_dir, "gap_peptides.tsv")
        gap_peptides.head(500).to_csv(out_path, sep="\t", index=False, float_format="%.4f")
        print(f"  Found {len(gap_peptides)} peptides where DIA-NN >> MuMDIA")
        print(f"  Saved top 500 to {out_path}")
    else:
        print("  No significant feature gaps found")


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

    # === Feature-level comparison ===
    print("\nGenerating feature-level comparison...")
    if "pin" in mumdia:
        # Parse DIA-NN fragment-level features
        print("  Parsing DIA-NN fragment correlations...")
        diann_features = parse_diann_fragment_correlations(diann["report"])

        # Match features by peptide/charge
        print("  Matching features by peptide/charge...")
        merged_features = match_features(mumdia["pin"], diann_features)
        print(f"  Matched {len(merged_features)} peptidoforms between tools")

        if len(merged_features) > 0:
            # Feature summary table
            generate_feature_summary_table(merged_features, args.output_dir)

            # Fragment correlation comparison
            print("  Generating fragment correlation plots...")
            try:
                compare_fragment_correlations(merged_features, args.output_dir)
            except Exception as e:
                print(f"  Warning: fragment correlation comparison failed: {e}")

            # XIC comparison
            print("  Generating XIC comparison plots...")
            try:
                compare_xic_features(merged_features, args.output_dir)
            except Exception as e:
                print(f"  Warning: XIC comparison failed: {e}")

            # DIA-NN native vs MuMDIA features
            print("  Generating DIA-NN feature distribution plots...")
            try:
                compare_diann_native_vs_mumdia(merged_features, args.output_dir)
            except Exception as e:
                print(f"  Warning: DIA-NN feature comparison failed: {e}")

            # Diagnose gaps
            print("  Diagnosing feature gaps...")
            try:
                diagnose_feature_gaps(merged_features, args.output_dir)
            except Exception as e:
                print(f"  Warning: gap diagnosis failed: {e}")

            # Save full merged features
            out_path = os.path.join(args.output_dir, "merged_features.tsv")
            merged_features.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
            print(f"  Saved merged feature table to {out_path}")
    else:
        print("  No PIN file available, skipping feature comparison")

    print(f"\nAll plots saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
