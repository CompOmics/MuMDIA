#!/usr/bin/env python
"""Grouped feature-family ablation for MuMDIA (spec 03 Sections 4-5, backlog P4.3).

Estimates how much each feature family contributes to sensitivity, with strict
leakage guards, on the comp.parquet feature table:

  * grouped cross-validation with the precursor (peptidoform + charge) held whole
    inside a single fold, so no row of a precursor trains a model that then scores
    another row of the same precursor;
  * imputation and standardization fit inside each training fold only;
  * target-decoy q-values computed from out-of-fold scores; the metric is the
    number of target rows passing an empirical FDP threshold;
  * two model families (L2 logistic regression, HistGradientBoosting).

Ablations reported per family: full-minus-family (does removing it drop targets?)
and minimal-baseline-plus-family (does adding it to a small baseline help?).

Reads Parquet only; deterministic (all hashing / seeds fixed).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")  # silence sklearn convergence chatter

SEED = 0

# Columns that must never enter the feature matrix (ids, labels, targets,
# leakage-prone scores). `charge` is treated as meta per the task spec.
META_COLUMNS = {
    "candidate_id", "label", "base_peptide_id", "peptidoform", "protein",
    "apex_rt", "precursor_mz", "prelim_score", "charge",
    "q_value", "peptide_q_value", "pg_q_value", "global_q_value",
    "score", "protein_group", "source",
}


# --------------------------------------------------------------------------- #
def load_registry(path):
    """Parse feature_registry.yaml (name -> family) without a YAML dependency.

    The file is a flat two-space-indented mapping under `features:`; each feature
    name is a quoted key at indent 2 and `family:` is a quoted value at indent 4.
    """
    fam = {}
    cur = None
    in_features = False
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.rstrip("\n") == "features:":
                in_features = True
                continue
            if not in_features:
                continue
            if line and not line[0].isspace() and line.strip():
                # a new top-level key ends the features block
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            if indent == 2 and stripped.endswith(":"):
                cur = stripped[:-1].strip().strip('"').strip("'")
            elif indent == 4 and stripped.startswith("family:") and cur is not None:
                val = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                fam[cur] = val
    return fam


def stable_fold(key, folds):
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h, 16) % folds


def target_ids_at_fdp(scores, is_target, fdp):
    """Count target rows passing `fdp` using target-decoy q-values.

    q at a score threshold = (n_decoys + 1) / max(n_targets, 1); ties share the
    group-end value; q-values are monotonised from the least confident end.
    """
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    t = is_target[order].astype(np.int64)
    d = 1 - t
    cum_t = np.cumsum(t)
    cum_d = np.cumsum(d)
    fdr = (cum_d + 1.0) / np.maximum(cum_t, 1)
    n = len(s)
    # tie handling: every row in an equal-score group gets the group-end fdr
    q = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        q[i:j + 1] = fdr[j]
        i = j + 1
    qv = np.minimum.accumulate(q[::-1])[::-1]
    return int(((qv <= fdp) & (t == 1)).sum())


# --------------------------------------------------------------------------- #
def build_model(kind):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    if kind == "logreg":
        # lbfgs must be allowed to converge: on the full ~355 collinear standardized
        # features it needs ~850 iterations, and an under-converged fit produces
        # degenerate scores (0 targets at 1% FDP). It stops early once tol is met, so
        # a high cap costs nothing on the small ablation subsets. liblinear coordinate
        # descent converges but is ~15x slower here.
        return LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
            tol=1e-3, random_state=SEED
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_depth=None,
            l2_regularization=1.0, random_state=SEED
        )
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="comp.parquet")
    ap.add_argument("--registry", required=True, help="feature_registry.yaml")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fdp", type=float, default=0.01)
    ap.add_argument("--model", choices=["logreg", "hgb", "both"], default="both")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    ap.add_argument("--clip-sd", type=float, default=8.0,
                    help="winsorize standardized features to +/- this many SD "
                         "(0 disables); tames outlier-driven logreg miscalibration")
    args = ap.parse_args()

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.features)),
                                       "feature_ablation")
    os.makedirs(out_dir, exist_ok=True)

    registry = load_registry(args.registry)

    # ---- load comp.parquet (optionally capped) --------------------------------
    schema = pq.read_schema(args.features)
    import pyarrow as pa
    numeric_cols = [
        n for n in schema.names
        if pa.types.is_floating(schema.field(n).type)
        or pa.types.is_integer(schema.field(n).type)
    ]
    feature_cols = [c for c in numeric_cols if c not in META_COLUMNS]
    read_cols = feature_cols + ["label", "peptidoform", "charge"]

    if args.max_rows and args.max_rows > 0:
        pf = pq.ParquetFile(args.features)
        batches = []
        got = 0
        for b in pf.iter_batches(batch_size=50000, columns=read_cols):
            batches.append(b)
            got += b.num_rows
            if got >= args.max_rows:
                break
        table = pa.Table.from_batches(batches).slice(0, args.max_rows)
    else:
        table = pq.read_table(args.features, columns=read_cols)

    n = table.num_rows
    label = np.array(table.column("label").to_pylist())
    is_target = (label == "target").astype(np.int64)
    peptidoform = table.column("peptidoform").to_pylist()
    charge = table.column("charge").to_numpy()

    # ---- leakage guard --------------------------------------------------------
    leaked = [c for c in feature_cols if c in META_COLUMNS]
    assert not leaked, f"leakage: meta columns in feature matrix: {leaked}"

    # ---- feature matrix -------------------------------------------------------
    X = table.select(feature_cols).to_pandas().to_numpy(dtype=np.float64)
    X[~np.isfinite(X)] = np.nan  # inf -> nan for the imputer

    # ---- family map -----------------------------------------------------------
    fam_of = {c: registry.get(c, "unknown") for c in feature_cols}
    families = {}
    for j, c in enumerate(feature_cols):
        families.setdefault(fam_of[c], []).append(j)
    fam_breakdown = {f: len(idx) for f, idx in sorted(families.items())}

    # ---- minimal baseline -----------------------------------------------------
    minimal_names = [c for c in feature_cols if registry.get(c) == "minimal"]
    if len(minimal_names) >= 5:
        baseline_names = minimal_names
    else:
        baseline_names = feature_cols[:5]  # fallback: first 5 available
    baseline_idx = [feature_cols.index(c) for c in baseline_names]

    print(f"[data] rows={n}  targets={int(is_target.sum())}  "
          f"decoys={int((1 - is_target).sum())}")
    print(f"[features] {len(feature_cols)} numeric feature columns "
          f"(of {len(schema.names)} total)")
    print(f"[families] {len(families)}: " +
          ", ".join(f"{f}={c}" for f, c in fam_breakdown.items()))
    print(f"[baseline] {len(baseline_idx)} features "
          f"({'minimal tier' if len(minimal_names) >= 5 else 'first-5 fallback'})")

    # ---- grouped folds --------------------------------------------------------
    folds_arr = np.array(
        [stable_fold(f"{peptidoform[i]}|{int(charge[i])}", args.folds)
         for i in range(n)],
        dtype=np.int64,
    )
    fold_sizes = [int((folds_arr == k).sum()) for k in range(args.folds)]
    print(f"[cv] {args.folds} grouped folds, sizes={fold_sizes}")

    # ---- configs to evaluate --------------------------------------------------
    all_idx = list(range(len(feature_cols)))
    configs = {"full": all_idx, "baseline": baseline_idx}
    for fam, idx in families.items():
        fam_set = set(idx)
        configs[f"minus::{fam}"] = [j for j in all_idx if j not in fam_set]
        configs[f"base+::{fam}"] = sorted(set(baseline_idx) | fam_set)

    models = ["logreg", "hgb"] if args.model == "both" else [args.model]

    # ---- run: per model, per fold fit shared imputer/scaler once --------------
    results = {}  # (model) -> {config -> ids}
    for mkind in models:
        print(f"\n[model] {mkind}")
        # precompute per-fold transformed full matrices
        fold_data = []
        for k in range(args.folds):
            te = folds_arr == k
            tr = ~te
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
            scl = StandardScaler()
            Xtr = scl.fit_transform(imp.fit_transform(X[tr]))
            Xte = scl.transform(imp.transform(X[te]))
            if args.clip_sd and args.clip_sd > 0:
                Xtr = np.clip(Xtr, -args.clip_sd, args.clip_sd)
                Xte = np.clip(Xte, -args.clip_sd, args.clip_sd)
            fold_data.append((tr, te, Xtr, Xte))

        config_scores = {c: np.full(n, -np.inf) for c in configs}
        for ci, (cname, idxs) in enumerate(configs.items()):
            idxs = np.array(idxs, dtype=int)
            for (tr, te, Xtr, Xte) in fold_data:
                model = build_model(mkind)
                model.fit(Xtr[:, idxs], is_target[tr])
                proba = model.predict_proba(Xte[:, idxs])[:, 1]
                config_scores[cname][te] = proba
        ids = {c: target_ids_at_fdp(config_scores[c], is_target, args.fdp)
               for c in configs}
        results[mkind] = ids
        print(f"  full={ids['full']}  baseline={ids['baseline']}")

    # ---- build table ----------------------------------------------------------
    rows = []
    for mkind in models:
        ids = results[mkind]
        ids_full = ids["full"]
        ids_base = ids["baseline"]
        tol_n = max(1, round(0.001 * ids_full))
        for fam in sorted(families):
            new_ids = ids[f"base+::{fam}"]
            minus_ids = ids[f"minus::{fam}"]
            rel_gain = (new_ids - ids_base) / ids_base if ids_base else 0.0
            delta_full = minus_ids - ids_full  # negative => removing hurts
            if delta_full < -tol_n:
                rec = "KEEP"
            elif delta_full > tol_n:
                rec = "HARMFUL"  # removing it improves the model
            elif rel_gain > 0.01:
                rec = "REDUNDANT_BUT_INFORMATIVE"  # helps alone, redundant in full
            else:
                rec = "REDUNDANT"
            rows.append({
                "feature_family": fam,
                "n_features": len(families[fam]),
                "baseline_identifications": ids_base,
                "new_identifications": new_ids,
                "relative_gain": round(rel_gain, 5),
                "full_identifications": ids_full,
                "minus_identifications": minus_ids,
                "delta_vs_full": delta_full,
                "model": mkind,
                "recommendation": rec,
            })

    # ---- write ----------------------------------------------------------------
    csv_path = os.path.join(out_dir, "feature_ablation.csv")
    json_path = os.path.join(out_dir, "feature_ablation.json")
    fieldnames = ["feature_family", "n_features", "baseline_identifications",
                  "new_identifications", "relative_gain", "full_identifications",
                  "minus_identifications", "delta_vs_full", "model", "recommendation"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {
        "params": {"folds": args.folds, "fdp": args.fdp, "model": args.model,
                   "max_rows": args.max_rows},
        "n_rows": n,
        "n_targets": int(is_target.sum()),
        "n_decoys": int((1 - is_target).sum()),
        "n_features": len(feature_cols),
        "family_breakdown": fam_breakdown,
        "baseline_n_features": len(baseline_idx),
        "full_identifications": {m: results[m]["full"] for m in models},
        "baseline_identifications": {m: results[m]["baseline"] for m in models},
        "table": rows,
    }
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- report ---------------------------------------------------------------
    print(f"\n=== Feature-family ablation (fdp={args.fdp}) ===")
    for mkind in models:
        mrows = [r for r in rows if r["model"] == mkind]
        by_delta = sorted(mrows, key=lambda r: r["delta_vs_full"])  # most negative first
        print(f"\n[{mkind}] full={results[mkind]['full']}  "
              f"baseline={results[mkind]['baseline']}")
        print("  most useful (largest target drop when removed):")
        for r in by_delta[:3]:
            print(f"    {r['feature_family']:<28} delta_vs_full={r['delta_vs_full']:+d}  "
                  f"rel_gain={r['relative_gain']:+.3f}  {r['recommendation']}")
        print("  least useful (removal helps or is neutral):")
        for r in by_delta[-3:][::-1]:
            print(f"    {r['feature_family']:<28} delta_vs_full={r['delta_vs_full']:+d}  "
                  f"rel_gain={r['relative_gain']:+.3f}  {r['recommendation']}")
    print(f"\n[written] {csv_path}")
    print(f"[written] {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
