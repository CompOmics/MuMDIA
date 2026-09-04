"""Build candidate feature subsets from the structural and importance tables.

Families of candidates, each a hypothesis about what can go:
  dedup            drop constants, exact and affine duplicates
  clust_<t>        one representative per Spearman cluster at |rho| >= t, the representative
                   being the member with the highest importance (permutation drop when given,
                   else the univariate targets@1%)
  top<k>_perm      the k features with the largest permutation importance
  top<k>_univ      the k features with the largest univariate targets@1%
  l1_C<c>          features with a non-zero L1 weight at that C
  no_<family>      everything except one family (ablation)
  only_<family>    one family alone (plus nothing else)

Usage: fs_subsets.py FEATURES.csv OUT.json [--importance IMPORTANCE.csv] [--clusters CLUSTERS.json]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features_csv")
    ap.add_argument("out_json")
    ap.add_argument("--importance", default=None)
    ap.add_argument("--clusters", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv).fillna({"exact_duplicate_of": "", "affine_dup_of": ""})
    names = df["feature"].tolist()
    score = df.set_index("feature")["targets_1pct_best"].astype(float)
    imp = None
    if args.importance:
        im = pd.read_csv(args.importance).set_index("feature")
        imp = im["hgb_perm_auc_drop"].astype(float)
        score = imp.reindex(names).fillna(0.0)

    dead = set(df.loc[df["constant"], "feature"]) | set(df.loc[df["exact_duplicate_of"] != "", "feature"]) | set(df.loc[df["affine_dup_of"] != "", "feature"])
    subsets: dict[str, list[str]] = {}
    subsets["dedup"] = [n for n in names if n not in dead]

    for col in [c for c in df.columns if c.startswith("cluster_")]:
        thr = col.split("_", 1)[1]
        reps = []
        for _, g in df[~df["feature"].isin(dead)].groupby(col):
            best = max(g["feature"], key=lambda n: (score.get(n, 0.0), -names.index(n)))
            reps.append(best)
        subsets[f"clust_{thr}"] = [n for n in names if n in set(reps)]

    ranked = [n for n in score.sort_values(ascending=False).index if n not in dead]
    for k in (10, 25, 50, 75, 100, 150, 200):
        subsets[f"top{k}_{'perm' if imp is not None else 'univ'}"] = [n for n in names if n in set(ranked[:k])]
    if imp is not None:
        univ = df[~df["feature"].isin(dead)].sort_values("targets_1pct_best", ascending=False)["feature"].tolist()
        for k in (25, 50, 100):
            subsets[f"top{k}_univ"] = [n for n in names if n in set(univ[:k])]
        for c in ("0.001", "0.005", "0.02"):
            col = f"l1_first_C"
            keep = im.index[(im[col].notna()) & (im[col] <= float(c))].tolist()
            subsets[f"l1_C{c}"] = [n for n in names if n in set(keep)]

    fams = df.set_index("feature")["family"]
    for fam in sorted(fams.unique()):
        members = set(fams.index[fams == fam])
        subsets[f"no_{fam}"] = [n for n in names if n not in members and n not in dead]
        subsets[f"only_{fam}"] = [n for n in names if n in members and n not in dead]

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(subsets, fh, indent=1)
    for k, v in subsets.items():
        print(f"{k:28s} {len(v):4d}")


if __name__ == "__main__":
    main()
