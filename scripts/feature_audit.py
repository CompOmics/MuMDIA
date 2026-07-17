#!/usr/bin/env python
"""Feature audit for MuMDIA scored/competed feature tables.

Implements the sensitivity_plan backlog item P4.2 (spec
``sensitivity_plan/03_feature_evaluation.md`` section 5, evidence ladder
Level 1 to Level 3). The tool is non-invasive: it reads a competed features
Parquet plus the feature registry and writes an audit report. Nothing in the
engine or the input data is modified.

Evidence ladder covered here:

Level 1 (data quality), per feature:
  missing percentage, NaN/inf percentage, unique-value count, quantiles
  (5/25/50/75/95), constant flag, per-group distribution summaries for
  targets, decoys and entrapments (mean/median), correlation with
  prelim_score, correlation with peptide length and charge, and correlation
  with log apex intensity. Flags: constant, asymmetric missingness between
  targets and decoys, intensity-dominated, and target-vs-decoy separation
  that is much larger than target-vs-entrapment separation (a decoy
  construction artifact).

Level 2 (univariate utility), per feature:
  target-vs-decoy separation and target-vs-entrapment separation, both as a
  Mann-Whitney U rank AUC and its rank-biserial magnitude. Features that
  separate target from decoy strongly but target from entrapment weakly are
  flagged LEAKAGE_RISK.

Level 3 (redundancy):
  Spearman correlation matrix over the non-constant features, single-linkage
  clustering at ``|rho| >= threshold`` via union-find, with one representative
  reported per cluster.

Registry cross-check:
  every audited feature must appear in the registry; features that do not are
  reported as UNKNOWN, and per-family coverage is summarised.

Entrapment framing: this is a target-decoy plus entrapment experiment. The
"real" target group is the target-labelled rows whose protein matches the
real-species substring (default ``_ECOLI``). The entrapment group is the
target-labelled rows whose protein matches the entrapment substring (default
``_HUMAN``); these are foreign-proteome spike-ins that behave as a null. The
decoy group is the decoy-labelled rows. A useful feature separates real
targets from both decoys and entrapments; a feature that separates real
targets from decoys but not from entrapments is likely exploiting the decoy
construction scheme.

The tool is deterministic: sampling uses a fixed seed, and all iteration
orders are fixed.

Interpreter: ``C:/Users/robbi/anaconda3/envs/py312_mumdia/python.exe``.

Example:

    python feature_audit.py \
        --features C:/proteobench/out_ecoli/comp.parquet \
        --registry feature_registry.yaml \
        --max-rows 80000 --entrapment-substr _HUMAN
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy.stats import rankdata

# --- Meta columns that are never treated as scoring features. --------------
META_COLUMNS = [
    "candidate_id",
    "label",
    "base_peptide_id",
    "peptidoform",
    "protein",
    "apex_rt",
    "precursor_mz",
    "prelim_score",
    "charge",
    "q_value",
    "peptide_q_value",
    "pg_q_value",
    "global_q_value",
    "score",
]

# --- Flag thresholds (documented, deterministic). --------------------------
# A separation is "strong" when its rank-biserial magnitude reaches this.
SEP_STRONG = 0.15
# Leakage: target-vs-entrapment separation below this fraction of the
# target-vs-decoy separation, while the latter is strong.
LEAKAGE_RATIO = 0.5
# Intensity-dominated: absolute Spearman correlation with log apex intensity.
INTENSITY_CORR = 0.70
# Asymmetric missingness: target vs decoy missing-rate ratio, with a floor on
# the larger rate so that trivially tiny rates are not flagged.
ASYM_RATIO = 5.0
ASYM_MIN_RATE = 0.01
# Redundancy clustering default correlation threshold.
DEFAULT_REDUNDANCY_THRESHOLD = 0.90
# Default deterministic sampling seed.
DEFAULT_SEED = 1234

_BRACKET_RE = re.compile(r"\[[^\]]*\]")


# ---------------------------------------------------------------------------
# Registry parsing
# ---------------------------------------------------------------------------
def load_registry(path):
    """Load the feature registry YAML.

    Returns a dict name -> {family, level, direction, source_file, ...}. Only
    the active (non-collision) rows are usable as columns; rows carrying
    ``dropped_collision: true`` (keyed like ``name@family``) are kept so that
    coverage accounting is complete but are not expected as columns.
    """
    with open(path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    features = doc.get("features", {}) or {}
    reg = OrderedDict()
    for name, meta in features.items():
        meta = meta or {}
        reg[str(name)] = {
            "family": str(meta.get("family", "?")),
            "level": str(meta.get("level", "?")),
            "direction": str(meta.get("direction", "?")),
            "source_file": str(meta.get("source_file", "")),
            "dropped_collision": bool(meta.get("dropped_collision", False)),
        }
    return reg


# ---------------------------------------------------------------------------
# Data loading (bounded memory via deterministic row sampling)
# ---------------------------------------------------------------------------
def load_features(path, max_rows, seed):
    """Read the features Parquet, sampling rows deterministically if needed.

    Returns (dataframe, total_rows, sampled_flag).
    """
    table = pq.read_table(path)
    total_rows = table.num_rows
    sampled = False
    if max_rows and total_rows > max_rows:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(total_rows, size=max_rows, replace=False))
        table = table.take(pa.array(idx))
        sampled = True
    df = table.to_pandas()
    del table
    return df, total_rows, sampled


def peptide_length(pepform):
    """Stripped peptide length from a ProForma-lite peptidoform.

    Drops a ``DECOY_`` prefix, removes bracketed modifications with the regex
    ``\\[[^\\]]*\\]``, and counts uppercase residue letters.
    """
    if not isinstance(pepform, str) or not pepform:
        return np.nan
    seq = pepform
    if seq.startswith("DECOY_"):
        seq = seq[len("DECOY_"):]
    seq = _BRACKET_RE.sub("", seq)
    return float(sum(1 for c in seq if "A" <= c <= "Z"))


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def safe_spearman(x, y):
    """Spearman correlation over jointly finite pairs; NaN if undefined."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xr = rankdata(x[mask])
    yr = rankdata(y[mask])
    if xr.std() == 0.0 or yr.std() == 0.0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def auc_mwu(pos, neg):
    """Rank AUC = P(pos > neg) via the Mann-Whitney U statistic.

    Returns NaN when either group is empty. Ties contribute 0.5.
    """
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    n1 = len(pos)
    n2 = len(neg)
    if n1 == 0 or n2 == 0:
        return np.nan
    ranks = rankdata(np.concatenate([pos, neg]))
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n2))


def separation(auc):
    """Rank-biserial magnitude in [0, 1] from a rank AUC."""
    if auc is None or not np.isfinite(auc):
        return np.nan
    return abs(2.0 * auc - 1.0)


# ---------------------------------------------------------------------------
# Group masks
# ---------------------------------------------------------------------------
def substr_match(series, substrings):
    """Boolean mask: True where any of the substrings occurs in the string."""
    result = np.zeros(len(series), dtype=bool)
    for sub in substrings:
        if sub:
            result |= series.str.contains(re.escape(sub), regex=True).to_numpy()
    return result


# ---------------------------------------------------------------------------
# Redundancy clustering
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self, items):
        self.parent = {it: it for it in items}

    def find(self, a):
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression.
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic: smaller name becomes the root.
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def cluster_features(df, feature_cols, threshold):
    """Cluster features by single-linkage Spearman ``|rho| >= threshold``.

    Returns (clusters, corr_dataframe) where clusters is a dict
    root -> sorted member list, including singletons.
    """
    if not feature_cols:
        return {}, None
    sub = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    corr = sub.corr(method="spearman")
    cols = list(corr.columns)
    values = corr.to_numpy()
    uf = UnionFind(cols)
    n = len(cols)
    for i in range(n):
        row = values[i]
        for j in range(i + 1, n):
            r = row[j]
            if np.isfinite(r) and abs(r) >= threshold:
                uf.union(cols[i], cols[j])
    groups = defaultdict(list)
    for col in cols:
        groups[uf.find(col)].append(col)
    for root in groups:
        groups[root].sort()
    return dict(groups), corr


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def run_audit(args):
    registry = load_registry(args.registry)
    df, total_rows, sampled = load_features(args.features, args.max_rows, args.seed)
    n_used = len(df)

    if "label" not in df.columns:
        raise SystemExit("features table has no 'label' column")

    # Reference vectors for Level 1 correlations.
    label = df["label"].astype(str)
    protein = df["protein"].astype(str) if "protein" in df.columns else pd.Series([""] * n_used)
    prelim = df["prelim_score"].to_numpy(dtype=float) if "prelim_score" in df.columns else np.full(n_used, np.nan)
    charge_ref = df["charge"].to_numpy(dtype=float) if "charge" in df.columns else np.full(n_used, np.nan)
    pep_len = np.array([peptide_length(p) for p in df["peptidoform"]], dtype=float) if "peptidoform" in df.columns else np.full(n_used, np.nan)
    has_intensity_col = "log_apex_intensity" in df.columns
    log_int = df["log_apex_intensity"].to_numpy(dtype=float) if has_intensity_col else np.full(n_used, np.nan)

    real_subs = [s.strip() for s in args.real_substr.split(",") if s.strip()]
    entrap_subs = [s.strip() for s in args.entrapment_substr.split(",") if s.strip()]

    is_target = (label == "target").to_numpy()
    is_decoy = (label == "decoy").to_numpy()
    real_mask_prot = substr_match(protein, real_subs)
    entrap_mask_prot = substr_match(protein, entrap_subs)

    mask_target = is_target & real_mask_prot
    mask_decoy = is_decoy
    mask_entrap = is_target & entrap_mask_prot

    n_target = int(mask_target.sum())
    n_decoy = int(mask_decoy.sum())
    n_entrap = int(mask_entrap.sum())

    # Feature columns: numeric, not meta.
    meta_set = set(META_COLUMNS)
    feature_cols = []
    for col in df.columns:
        if col in meta_set:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    feature_cols.sort()

    rows = []
    unknown_features = []
    for col in feature_cols:
        arr = df[col].to_numpy(dtype=float)
        finite = np.isfinite(arr)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        finite_vals = arr[finite]
        uniq = int(np.unique(finite_vals).size) if finite_vals.size else 0
        constant = uniq <= 1

        if finite_vals.size:
            q05, q25, q50, q75, q95 = (
                float(np.percentile(finite_vals, p)) for p in (5, 25, 50, 75, 95)
            )
        else:
            q05 = q25 = q50 = q75 = q95 = np.nan

        def grp_stats(mask):
            vals = arr[mask & finite]
            if vals.size == 0:
                return np.nan, np.nan
            return float(vals.mean()), float(np.median(vals))

        mean_t, med_t = grp_stats(mask_target)
        mean_d, med_d = grp_stats(mask_decoy)
        mean_e, med_e = grp_stats(mask_entrap)

        corr_prelim = safe_spearman(arr, prelim)
        corr_len = safe_spearman(arr, pep_len)
        corr_charge = safe_spearman(arr, charge_ref)
        corr_int = safe_spearman(arr, log_int) if has_intensity_col else np.nan

        # Missingness by label group for the asymmetry flag.
        n_t_rows = int(is_target.sum())
        n_d_rows = int(is_decoy.sum())
        miss_rate_t = float(np.isnan(arr[is_target]).mean()) if n_t_rows else np.nan
        miss_rate_d = float(np.isnan(arr[is_decoy]).mean()) if n_d_rows else np.nan
        if np.isfinite(miss_rate_t) and np.isfinite(miss_rate_d):
            hi = max(miss_rate_t, miss_rate_d)
            lo = min(miss_rate_t, miss_rate_d)
            asym_ratio = (hi + 1e-12) / (lo + 1e-12)
        else:
            hi = np.nan
            asym_ratio = np.nan

        auc_td = auc_mwu(arr[mask_target], arr[mask_decoy])
        auc_te = auc_mwu(arr[mask_target], arr[mask_entrap])
        sep_td = separation(auc_td)
        sep_te = separation(auc_te)

        # Flags.
        flag_constant = constant
        flag_asym = bool(
            np.isfinite(asym_ratio)
            and hi > ASYM_MIN_RATE
            and asym_ratio > ASYM_RATIO
        )
        flag_intensity = bool(
            has_intensity_col
            and col != "log_apex_intensity"
            and np.isfinite(corr_int)
            and abs(corr_int) >= INTENSITY_CORR
        )
        flag_leakage = bool(
            np.isfinite(sep_td)
            and np.isfinite(sep_te)
            and sep_td >= SEP_STRONG
            and sep_te < LEAKAGE_RATIO * sep_td
        )

        reg = registry.get(col)
        in_registry = reg is not None
        if not in_registry:
            unknown_features.append(col)
        family = reg["family"] if in_registry else "UNKNOWN"
        level = reg["level"] if in_registry else "?"
        direction = reg["direction"] if in_registry else "?"

        rows.append(
            {
                "feature": col,
                "family": family,
                "level": level,
                "direction": direction,
                "in_registry": in_registry,
                "n_rows_used": n_used,
                "missing_pct": 100.0 * n_nan / n_used if n_used else np.nan,
                "inf_pct": 100.0 * n_inf / n_used if n_used else np.nan,
                "unique_count": uniq,
                "constant": constant,
                "q05": q05,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q95": q95,
                "n_target": n_target,
                "mean_target": mean_t,
                "median_target": med_t,
                "n_decoy": n_decoy,
                "mean_decoy": mean_d,
                "median_decoy": med_d,
                "n_entrap": n_entrap,
                "mean_entrap": mean_e,
                "median_entrap": med_e,
                "corr_prelim_score": corr_prelim,
                "corr_peptide_length": corr_len,
                "corr_charge": corr_charge,
                "corr_log_apex_intensity": corr_int,
                "missing_rate_target": miss_rate_t,
                "missing_rate_decoy": miss_rate_d,
                "asym_missing_ratio": asym_ratio,
                "auc_target_decoy": auc_td,
                "sep_target_decoy": sep_td,
                "auc_target_entrap": auc_te,
                "sep_target_entrap": sep_te,
                "sep_gap_decoy_minus_entrap": (sep_td - sep_te) if (np.isfinite(sep_td) and np.isfinite(sep_te)) else np.nan,
                "flag_constant": flag_constant,
                "flag_asymmetric_missing": flag_asym,
                "flag_intensity_dominated": flag_intensity,
                "flag_leakage": flag_leakage,
            }
        )

    audit = pd.DataFrame(rows)

    # Level 3 redundancy over non-constant features.
    nonconst = [r["feature"] for r in rows if not r["constant"]]
    clusters, _corr = cluster_features(df, nonconst, args.redundancy_threshold)

    # Representative per cluster: highest target-vs-entrapment separation,
    # then highest target-vs-decoy separation, then name.
    sep_te_map = {r["feature"]: (r["sep_target_entrap"] if np.isfinite(r["sep_target_entrap"]) else -1.0) for r in rows}
    sep_td_map = {r["feature"]: (r["sep_target_decoy"] if np.isfinite(r["sep_target_decoy"]) else -1.0) for r in rows}

    cluster_id_map = {}
    representative_map = {}
    multi_clusters = []
    cluster_counter = 0
    # Deterministic order: sort clusters by their sorted member list.
    for root, members in sorted(clusters.items(), key=lambda kv: kv[1]):
        if len(members) < 2:
            continue
        rep = sorted(
            members,
            key=lambda f: (-sep_te_map[f], -sep_td_map[f], f),
        )[0]
        cid = cluster_counter
        cluster_counter += 1
        for m in members:
            cluster_id_map[m] = cid
            representative_map[m] = (m == rep)
        multi_clusters.append(
            {
                "cluster_id": cid,
                "size": len(members),
                "representative": rep,
                "representative_sep_target_entrap": (None if sep_te_map[rep] < 0 else round(sep_te_map[rep], 6)),
                "members": members,
            }
        )

    n_singletons = sum(1 for m in clusters.values() if len(m) == 1)

    audit["cluster_id"] = audit["feature"].map(lambda f: cluster_id_map.get(f, -1))
    audit["is_cluster_representative"] = audit["feature"].map(lambda f: representative_map.get(f, False))

    return {
        "audit": audit,
        "rows": rows,
        "clusters": multi_clusters,
        "n_singletons": n_singletons,
        "unknown_features": unknown_features,
        "registry": registry,
        "feature_cols": feature_cols,
        "total_rows": total_rows,
        "n_used": n_used,
        "sampled": sampled,
        "n_target": n_target,
        "n_decoy": n_decoy,
        "n_entrap": n_entrap,
        "real_subs": real_subs,
        "entrap_subs": entrap_subs,
        "has_intensity_col": has_intensity_col,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_outputs(result, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    audit = result["audit"]
    rows = result["rows"]
    registry = result["registry"]

    csv_path = os.path.join(out_dir, "feature_audit.csv")
    audit.to_csv(csv_path, index=False, float_format="%.6g")

    clusters_path = os.path.join(out_dir, "redundancy_clusters.json")
    with open(clusters_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "spearman_abs_threshold": args.redundancy_threshold,
                "n_features_considered": int((~audit["constant"]).sum()),
                "n_clusters": len(result["clusters"]),
                "n_singletons": result["n_singletons"],
                "clusters": result["clusters"],
            },
            handle,
            indent=2,
        )

    # Warnings file.
    by_feature = {r["feature"]: r for r in rows}
    constant_feats = [r["feature"] for r in rows if r["flag_constant"]]
    asym_feats = [r["feature"] for r in rows if r["flag_asymmetric_missing"]]
    intensity_feats = [r["feature"] for r in rows if r["flag_intensity_dominated"]]
    leakage_feats = [r["feature"] for r in rows if r["flag_leakage"]]
    unknown = result["unknown_features"]

    # Registry features absent from the table (active rows only).
    table_names = set(result["feature_cols"])
    missing_from_table = [
        name
        for name, meta in registry.items()
        if not meta["dropped_collision"] and name not in table_names and name not in set(META_COLUMNS)
    ]

    warnings_path = os.path.join(out_dir, "warnings.txt")
    with open(warnings_path, "w", encoding="utf-8") as handle:
        handle.write("MuMDIA feature audit warnings\n")
        handle.write("=" * 60 + "\n\n")

        handle.write("CONSTANT features (%d)\n" % len(constant_feats))
        for f in constant_feats:
            handle.write("  %s\n" % f)
        handle.write("\n")

        handle.write("ASYMMETRIC MISSINGNESS target vs decoy > %gx (%d)\n" % (ASYM_RATIO, len(asym_feats)))
        for f in asym_feats:
            r = by_feature[f]
            handle.write(
                "  %s  target=%.4f decoy=%.4f ratio=%.1f\n"
                % (f, r["missing_rate_target"], r["missing_rate_decoy"], r["asym_missing_ratio"])
            )
        handle.write("\n")

        handle.write("INTENSITY-DOMINATED |corr(log_apex_intensity)| >= %g (%d)\n" % (INTENSITY_CORR, len(intensity_feats)))
        for f in intensity_feats:
            handle.write("  %s  corr=%.3f\n" % (f, by_feature[f]["corr_log_apex_intensity"]))
        handle.write("\n")

        handle.write(
            "LEAKAGE RISK sep(target,decoy) >= %g and sep(target,entrapment) < %g x sep(target,decoy) (%d)\n"
            % (SEP_STRONG, LEAKAGE_RATIO, len(leakage_feats))
        )
        for f in leakage_feats:
            r = by_feature[f]
            handle.write(
                "  %s  sep_td=%.3f sep_te=%.3f gap=%.3f\n"
                % (f, r["sep_target_decoy"], r["sep_target_entrap"], r["sep_gap_decoy_minus_entrap"])
            )
        handle.write("\n")

        handle.write("UNKNOWN features not in registry (%d)\n" % len(unknown))
        for f in unknown:
            handle.write("  %s\n" % f)
        handle.write("\n")

        handle.write("REGISTRY features absent from the table (%d)\n" % len(missing_from_table))
        for f in missing_from_table:
            handle.write("  %s\n" % f)
        handle.write("\n")

    # Summary file.
    summary_path = os.path.join(out_dir, "summary.txt")
    n_feat = len(result["feature_cols"])
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("MuMDIA feature audit summary\n")
        handle.write("=" * 60 + "\n\n")
        handle.write("Input features : %s\n" % os.path.abspath(args.features))
        handle.write("Registry       : %s\n" % os.path.abspath(args.registry))
        handle.write("Output dir     : %s\n" % os.path.abspath(out_dir))
        handle.write("\n")
        if result["sampled"]:
            handle.write(
                "Sampling       : %d of %d rows (seed=%d, deterministic)\n"
                % (result["n_used"], result["total_rows"], args.seed)
            )
        else:
            handle.write("Sampling       : none, all %d rows used\n" % result["total_rows"])
        handle.write("\n")
        handle.write("Groups\n")
        handle.write("  real targets (%s) : %d\n" % (",".join(result["real_subs"]), result["n_target"]))
        handle.write("  decoys                 : %d\n" % result["n_decoy"])
        handle.write("  entrapments (%s) : %d\n" % (",".join(result["entrap_subs"]), result["n_entrap"]))
        handle.write("  note: target-labelled rows matching neither the real nor the entrapment\n")
        handle.write("        substring fall outside both the target and entrapment groups.\n")
        if not result["has_intensity_col"]:
            handle.write("  note: no log_apex_intensity column found; intensity checks skipped.\n")
        handle.write("  note: per-run drift not computed (no run column in this input).\n")
        handle.write("\n")
        handle.write("Features audited : %d\n" % n_feat)
        handle.write("Constant         : %d\n" % len(constant_feats))
        handle.write("Asymmetric miss. : %d\n" % len(asym_feats))
        handle.write("Intensity-domin. : %d\n" % len(intensity_feats))
        handle.write("Leakage risk     : %d\n" % len(leakage_feats))
        handle.write("Unknown (registry): %d\n" % len(unknown))
        handle.write("Redundancy clusters (size>=2): %d\n" % len(result["clusters"]))
        handle.write("Redundancy singletons        : %d\n" % result["n_singletons"])
        handle.write("\n")

        # Family coverage over audited features.
        fam_counts = defaultdict(int)
        for r in rows:
            fam_counts[r["family"]] += 1
        handle.write("Family coverage (audited features)\n")
        for fam in sorted(fam_counts):
            handle.write("  %-32s %d\n" % (fam, fam_counts[fam]))
        handle.write("\n")

        # Top separations.
        def top_by(key, n=10):
            valid = [r for r in rows if np.isfinite(r[key])]
            return sorted(valid, key=lambda r: (-r[key], r["feature"]))[:n]

        handle.write("Top 10 target-vs-entrapment separation\n")
        for r in top_by("sep_target_entrap"):
            handle.write(
                "  %-40s sep_te=%.3f sep_td=%.3f family=%s\n"
                % (r["feature"], r["sep_target_entrap"], r["sep_target_decoy"], r["family"])
            )
        handle.write("\n")
        handle.write("Top 10 target-vs-decoy separation\n")
        for r in top_by("sep_target_decoy"):
            handle.write(
                "  %-40s sep_td=%.3f sep_te=%.3f family=%s\n"
                % (r["feature"], r["sep_target_decoy"], r["sep_target_entrap"], r["family"])
            )
        handle.write("\n")

    return {
        "csv_path": csv_path,
        "clusters_path": clusters_path,
        "warnings_path": warnings_path,
        "summary_path": summary_path,
        "constant_feats": constant_feats,
        "leakage_feats": leakage_feats,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit a MuMDIA competed features Parquet against the feature registry."
    )
    parser.add_argument("--features", required=True, help="competed features Parquet")
    parser.add_argument("--registry", required=True, help="feature_registry.yaml")
    parser.add_argument("--out", default=None, help="output directory (default <features_dir>/feature_audit/)")
    parser.add_argument("--max-rows", type=int, default=None, help="deterministic row sample cap")
    parser.add_argument("--entrapment-substr", default="_HUMAN", help="protein substring(s) for the entrapment null (comma-separated)")
    parser.add_argument("--real-substr", default="_ECOLI", help="protein substring(s) for the real target species (comma-separated)")
    parser.add_argument("--redundancy-threshold", type=float, default=DEFAULT_REDUNDANCY_THRESHOLD, help="abs Spearman rho for clustering")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sampling seed")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.out:
        out_dir = args.out
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.features)), "feature_audit")

    result = run_audit(args)
    paths = write_outputs(result, args, out_dir)

    rows = result["rows"]
    top_te = sorted(
        [r for r in rows if np.isfinite(r["sep_target_entrap"])],
        key=lambda r: (-r["sep_target_entrap"], r["feature"]),
    )[:3]

    print("feature_audit written to: %s" % os.path.abspath(out_dir))
    print("features audited        : %d" % len(result["feature_cols"]))
    print("constant                : %d" % len(paths["constant_feats"]))
    print("leakage-flagged         : %d" % len(paths["leakage_feats"]))
    print("redundancy clusters     : %d (size>=2)" % len(result["clusters"]))
    if result["sampled"]:
        print("sampling                : %d of %d rows (seed=%d)" % (result["n_used"], result["total_rows"], args.seed))
    print("top-3 target-vs-entrapment separation:")
    for r in top_te:
        print("  %-40s sep_te=%.3f (sep_td=%.3f)" % (r["feature"], r["sep_target_entrap"], r["sep_target_decoy"]))


if __name__ == "__main__":
    sys.exit(main())
