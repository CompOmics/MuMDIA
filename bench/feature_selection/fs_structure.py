"""Structural and univariate analysis of one PIN: what the 387 features are, statistically.

Per feature: family, NaN and zero fractions, distinct values, dominance of the modal value,
target-vs-decoy AUC, and the worker's own univariate measure (targets at 1% FDR using that
feature alone, best sign). Between features: exact duplicates, |Pearson| >= 0.9999 affine
duplicates, Spearman clusters at several thresholds, and the PCA dimension of the
standardised matrix.

Usage: fs_structure.py PIN.parquet OUT_DIR TAG [--repo H:/.../MuMDIA_NG]
Writes OUT_DIR/TAG_features.csv, TAG_clusters.json, TAG_pairs.csv, TAG_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fs_lib  # noqa: E402


def rank_auc(x: np.ndarray, y: np.ndarray) -> float:
    """AUC of x as a score for y == 1 (targets), via the rank-sum statistic; ties averaged."""
    from scipy.stats import rankdata

    r = rankdata(x, method="average")
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pin")
    ap.add_argument("out_dir")
    ap.add_argument("tag")
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ap.add_argument("--corr-rows", type=int, default=300_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fam = fs_lib.family_map(args.repo)

    d = fs_lib.load_pin(args.pin)
    X, y, names = d["X"], d["y"], d["feat_cols"]
    n, nf = X.shape
    t0 = time.time()

    # ---- per-feature scalar statistics -------------------------------------------------
    rows = []
    for j, nm in enumerate(names):
        col = X[:, j]
        u, cnt = np.unique(col[:: max(1, n // 200_000)], return_counts=True)
        rows.append(
            dict(
                feature=nm,
                family=fam.get(nm, "?"),
                nan_frac=float(d["nan_frac"][j]),
                zero_frac=float((col == 0).mean()),
                distinct_sampled=int(len(u)),
                modal_frac=float(cnt.max() / cnt.sum()),
                mean=float(col.mean(dtype=np.float64)),
                std=float(col.std(dtype=np.float64)),
                min=float(col.min()),
                max=float(col.max()),
            )
        )
    df = pd.DataFrame(rows)
    df["constant"] = df["std"] == 0
    print(f"scalar stats {time.time() - t0:.0f}s", flush=True)

    # ---- univariate separation -----------------------------------------------------------
    t0 = time.time()
    rs = np.random.RandomState(0)
    sub = np.sort(rs.choice(n, size=min(n, args.corr_rows), replace=False))
    Xs, ys = X[sub], y[sub]
    df["auc_target"] = [rank_auc(Xs[:, j], ys) for j in range(nf)]
    # the worker's init-scan quantity, on the full pool, both signs
    tgt = y.astype(bool)
    n_pos, n_neg = [], []
    for j in range(nf):
        col = np.ascontiguousarray(X[:, j])
        n_pos.append(fs_lib.n_targets_at_col(col, tgt, 0.01))
        n_neg.append(fs_lib.n_targets_at_col(-col, tgt, 0.01))
    df["targets_1pct_pos"] = n_pos
    df["targets_1pct_neg"] = n_neg
    df["targets_1pct_best"] = df[["targets_1pct_pos", "targets_1pct_neg"]].max(axis=1)
    print(f"univariate {time.time() - t0:.0f}s", flush=True)

    # ---- redundancy ----------------------------------------------------------------------
    t0 = time.time()
    from scipy.stats import rankdata

    R = np.empty_like(Xs)
    for j in range(nf):
        R[:, j] = rankdata(Xs[:, j], method="average")
    R -= R.mean(axis=0)
    sd = R.std(axis=0)
    sd[sd == 0] = 1.0
    R /= sd
    rho = (R.T @ R) / len(sub)  # Spearman
    Zs = Xs - Xs.mean(axis=0)
    sdz = Zs.std(axis=0)
    sdz[sdz == 0] = 1.0
    Zs /= sdz
    pear = (Zs.T @ Zs) / len(sub)
    np.fill_diagonal(rho, 1.0)
    np.fill_diagonal(pear, 1.0)

    # exact duplicates on the full matrix (hash of bytes), affine duplicates from |Pearson|
    hashes = {}
    dup_of = [""] * nf
    for j in range(nf):
        h = hash(X[:, j].tobytes())
        if h in hashes and np.array_equal(X[:, hashes[h]], X[:, j]):
            dup_of[j] = names[hashes[h]]
        else:
            hashes.setdefault(h, j)
    df["exact_duplicate_of"] = dup_of
    pairs = []
    iu = np.triu_indices(nf, 1)
    for a, b in zip(*iu):
        p, s = pear[a, b], rho[a, b]
        if abs(p) >= 0.95 or abs(s) >= 0.95:
            pairs.append(dict(a=names[a], b=names[b], pearson=round(float(p), 5), spearman=round(float(s), 5)))
    pairs = pd.DataFrame(pairs).sort_values("spearman", key=lambda s: -s.abs()) if pairs else pd.DataFrame()
    df["affine_dup_of"] = ""
    for j in range(nf):
        if df.at[j, "constant"]:
            continue
        for i in range(j):
            if not df.at[i, "constant"] and abs(pear[i, j]) >= 0.9999:
                df.at[j, "affine_dup_of"] = names[i]
                break

    # hierarchical clusters on 1 - |Spearman|, average linkage
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    D = 1.0 - np.abs(rho)
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2, 0.0, 2.0)
    Z = linkage(squareform(D, checks=False), method="average")
    clusters = {}
    for thr in (0.02, 0.05, 0.10, 0.20, 0.30):
        lab = fcluster(Z, t=thr, criterion="distance")
        groups: dict[int, list[str]] = {}
        for j, l in enumerate(lab):
            groups.setdefault(int(l), []).append(names[j])
        clusters[f"abs_spearman_ge_{1 - thr:.2f}"] = dict(n_clusters=int(lab.max()), groups=sorted(groups.values(), key=len, reverse=True))
        df[f"cluster_{1 - thr:.2f}"] = lab
    print(f"redundancy {time.time() - t0:.0f}s", flush=True)

    # ---- PCA dimension -------------------------------------------------------------------
    t0 = time.time()
    keep = ~df["constant"].to_numpy()
    C = (Zs[:, keep].T @ Zs[:, keep]) / len(sub)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    ev = np.clip(ev, 0, None)
    cum = np.cumsum(ev) / ev.sum()
    pca = {f"components_for_{p}pct": int(np.searchsorted(cum, p / 100.0) + 1) for p in (80, 90, 95, 99, 99.9)}
    print(f"pca {time.time() - t0:.0f}s {pca}", flush=True)

    # ---- write ---------------------------------------------------------------------------
    df.to_csv(os.path.join(args.out_dir, f"{args.tag}_features.csv"), index=False)
    pairs.to_csv(os.path.join(args.out_dir, f"{args.tag}_pairs.csv"), index=False)
    with open(os.path.join(args.out_dir, f"{args.tag}_clusters.json"), "w") as fh:
        json.dump(clusters, fh, indent=1)
    summary = dict(
        pin=args.pin,
        rows=int(n),
        features=int(nf),
        targets=int(y.sum()),
        decoys=int(n - y.sum()),
        constant=int(df["constant"].sum()),
        near_constant_modal_ge_0_999=int((df["modal_frac"] >= 0.999).sum()),
        any_nan=int((df["nan_frac"] > 0).sum()),
        exact_duplicates=int((df["exact_duplicate_of"] != "").sum()),
        affine_duplicates=int((df["affine_dup_of"] != "").sum()),
        pairs_abs_corr_ge_0_95=int(len(pairs)),
        clusters={k: v["n_clusters"] for k, v in clusters.items()},
        pca=pca,
        per_family=df.groupby("family").size().to_dict(),
    )
    with open(os.path.join(args.out_dir, f"{args.tag}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
