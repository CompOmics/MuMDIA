"""Multivariate importance of the features, on the worker's own training population.

The rescoring worker trains on "targets at 1% FDR under the current model" against all
decoys, so that is the population an importance measure should use, not raw target vs
decoy (where ~97% of targets are wrong matches). Positives are the targets with out-of-fold
q <= 0.01 from a full-feature run saved by fs_objective.py (`--save-oof all`); negatives are
decoys.

Three views, all on standardised features:
  - L1 logistic path: at each C the number of non-zero weights and which features survive;
    the first C at which a feature is non-zero is its sparsity rank.
  - HistGradientBoosting: permutation importance on a held-out sample (drop in AUC), which
    handles redundancy honestly (a duplicated signal shares importance and both look small,
    which is the point: neither is needed alone) and gain-free (sklearn HGB has no gain
    attribute, so permutation is the single tree-based view).
  - single-feature-removed logistic AUC drop on the same set (cheap, linear, complementary).

Usage: fs_importance.py PIN.parquet OOF.npy OUT_DIR TAG
Writes OUT_DIR/TAG_importance.csv and TAG_l1_path.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fs_lib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pin")
    ap.add_argument("oof")
    ap.add_argument("out_dir")
    ap.add_argument("tag")
    ap.add_argument("--neg-cap", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = fs_lib.load_pin(args.pin)
    X, y, names = d["X"], d["y"], d["feat_cols"]
    oof = np.load(args.oof).astype(np.float64)
    q = fs_lib.tda_q(oof, y)
    pos = np.flatnonzero((q <= 0.01) & (y == 1))
    neg = np.flatnonzero(y == 0)
    rs = np.random.RandomState(args.seed)
    if len(neg) > args.neg_cap:
        neg = np.sort(rs.choice(neg, size=args.neg_cap, replace=False))
    idx = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    print(f"training population: {len(pos)} confident targets, {len(neg)} decoys", flush=True)

    Xs = fs_lib.standardise_inplace(np.ascontiguousarray(X[idx]).copy())
    del X
    perm = rs.permutation(len(idx))
    n_tr = int(0.7 * len(idx))
    tr, te = perm[:n_tr], perm[n_tr:]

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # ---- L1 path --------------------------------------------------------------------------
    t0 = time.time()
    Cs = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]
    first_c = np.full(len(names), np.inf)
    path_rows = []
    coef_at = {}
    for C in Cs:
        lr = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=300, tol=1e-3, class_weight="balanced")
        lr.fit(Xs[tr], lab[tr])
        w = lr.coef_.ravel()
        nz = np.flatnonzero(np.abs(w) > 1e-6)
        auc = roc_auc_score(lab[te], lr.decision_function(Xs[te]))
        n_pep = fs_lib.n_targets_at(lr.decision_function(Xs[te]), lab[te], 0.01)
        for j in nz:
            first_c[j] = min(first_c[j], C)
        coef_at[C] = w
        path_rows.append(dict(C=C, nonzero=int(len(nz)), auc_holdout=round(float(auc), 5), targets_1pct_holdout=n_pep))
        print(f"  L1 C={C}: {len(nz)} non-zero, AUC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(path_rows).to_csv(os.path.join(args.out_dir, f"{args.tag}_l1_path.csv"), index=False)

    # ---- gradient boosting + permutation importance ---------------------------------------
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True, random_state=args.seed)
    hgb.fit(Xs[tr], lab[tr])
    auc_hgb = roc_auc_score(lab[te], hgb.predict_proba(Xs[te])[:, 1])
    print(f"  HGB holdout AUC {auc_hgb:.4f}, {hgb.n_iter_} iters ({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time()
    te_s = te[: min(len(te), 150_000)]
    pi = permutation_importance(hgb, Xs[te_s], lab[te_s], scoring="roc_auc", n_repeats=3, random_state=args.seed, n_jobs=8)
    print(f"  permutation importance ({time.time() - t0:.0f}s)", flush=True)

    # ---- linear leave-one-out drop ----------------------------------------------------------
    t0 = time.time()
    lr_full = LogisticRegression(C=0.05, solver="lbfgs", max_iter=500, class_weight="balanced")
    lr_full.fit(Xs[tr], lab[tr])
    base_auc = roc_auc_score(lab[te], lr_full.decision_function(Xs[te]))
    w = lr_full.coef_.ravel()
    # zeroing a weight approximates refitting without it; cheap and monotone in |w| x spread
    loo = []
    for j in range(len(names)):
        w2 = w.copy()
        w2[j] = 0.0
        loo.append(base_auc - roc_auc_score(lab[te], Xs[te] @ w2 + lr_full.intercept_[0]))
    print(f"  linear LOO ({time.time() - t0:.0f}s), base AUC {base_auc:.4f}", flush=True)

    out = pd.DataFrame(
        dict(
            feature=names,
            l1_first_C=[None if np.isinf(c) else c for c in first_c],
            l1_coef_C0_01=coef_at[0.01],
            l1_coef_C0_1=coef_at[0.1],
            hgb_perm_auc_drop=pi.importances_mean,
            hgb_perm_auc_drop_std=pi.importances_std,
            linear_zero_weight_auc_drop=loo,
            linear_abs_weight=np.abs(w),
        )
    )
    out["hgb_perm_rank"] = out["hgb_perm_auc_drop"].rank(ascending=False, method="first").astype(int)
    out.to_csv(os.path.join(args.out_dir, f"{args.tag}_importance.csv"), index=False)
    print(out.sort_values("hgb_perm_auc_drop", ascending=False).head(25).to_string(), flush=True)


if __name__ == "__main__":
    main()
