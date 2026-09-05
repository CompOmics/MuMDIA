"""Sweep training-set reduction against the rescoring objective.

The rescoring worker refits its MLP 30 times (3 folds x 10 iterations), 25 epochs each, on
every decoy in the fold plus the targets currently at 1% FDR: about 871k rows at a 19:1
negative ratio on the HYE benchmark, and 86% of the whole run's wall clock. Most of those
rows are trivially separable and contribute nothing after the first iteration. This sweeps
how far the training population can be thinned before identifications move, over three
axes:

  neg_ratio    cap decoys at k x the positives selected that iteration (0 = all)
  neg_select   which decoys survive the cap: random / margin (highest scoring) / hybrid
  train_sub    stratified thinning of what remains (fraction)
  warm_start   reuse the previous iteration's weights, warm_epochs instead of 25

Nothing here changes what is scored or how q-values are computed: selection, competition
and FDR still run over the full pool, so the decoy fraction at 1% is unaffected by
construction, and only sensitivity is at risk.

Usage: fs_train_sweep.py PIN.parquet OUT_DIR TAG [--features FILE] [--seeds 0,1]
                         [--rows N] [--grid quick|full|confirm]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fs_lib  # noqa: E402


def grid(name: str) -> list[tuple[str, dict]]:
    """(label, cfg overrides). The baseline is the shipped configuration."""
    out: list[tuple[str, dict]] = [("baseline", {})]
    if name in ("quick", "full"):
        for r in (10, 5, 3, 2, 1):
            out.append((f"neg{r}_random", dict(neg_ratio=r, neg_select="random")))
        for r in (5, 3, 1):
            out.append((f"neg{r}_margin", dict(neg_ratio=r, neg_select="margin")))
            out.append((f"neg{r}_hybrid", dict(neg_ratio=r, neg_select="hybrid")))
    if name == "full":
        for sub in (0.5, 0.25):
            out.append((f"sub{sub}", dict(train_sub=sub)))
            out.append((f"neg3_random_sub{sub}", dict(neg_ratio=3, neg_select="random", train_sub=sub)))
        out.append(("warm5", dict(warm_start=True, warm_epochs=5)))
        out.append(("warm5_neg3_random", dict(warm_start=True, warm_epochs=5, neg_ratio=3, neg_select="random")))
        out.append(("warm5_neg3_hybrid", dict(warm_start=True, warm_epochs=5, neg_ratio=3, neg_select="hybrid")))
        out.append(("warm5_neg1_hybrid", dict(warm_start=True, warm_epochs=5, neg_ratio=1, neg_select="hybrid")))
    if name == "candidates":
        # The recipes worth seeds: a self-limiting cap (neg5 never binds on a pool that is
        # already balanced), the same with warm start, and the aggressive variants that
        # won on HYE but cost a little on AIF.
        out += [
            ("neg5_hybrid", dict(neg_ratio=5, neg_select="hybrid")),
            ("warm5", dict(warm_start=True, warm_epochs=5)),
            ("warm5_neg5_hybrid", dict(warm_start=True, warm_epochs=5, neg_ratio=5, neg_select="hybrid")),
            ("warm5_neg3_hybrid", dict(warm_start=True, warm_epochs=5, neg_ratio=3, neg_select="hybrid")),
            ("neg1_margin", dict(neg_ratio=1, neg_select="margin")),
        ]
    if name == "nn":
        # The rescorer itself, which has never been swept with seeds. Everything is on top of
        # the fast recipe (3:1 hybrid cap, warm start 5), because that is what makes a 2x
        # bigger model or 2x more epochs affordable at all.
        R = dict(neg_ratio=3, neg_select="hybrid", warm_start=True, warm_epochs=5)
        out += [
            ("recipe", dict(R)),
            ("recipe_h256_128", dict(R, hidden=(256, 128))),
            ("recipe_h256_128_64", dict(R, hidden=(256, 128, 64))),
            ("recipe_h512_256", dict(R, hidden=(512, 256))),
            ("recipe_h64_32", dict(R, hidden=(64, 32))),
            ("recipe_drop0.1", dict(R, dropout=0.1)),
            ("recipe_drop0.5", dict(R, dropout=0.5)),
            ("recipe_ep50_w10", dict(R, epochs=50, warm_epochs=10)),
            ("recipe_ep12_w3", dict(R, epochs=12, warm_epochs=3)),
            ("recipe_lr3e-4", dict(R, lr=3e-4)),
            ("recipe_lr3e-3", dict(R, lr=3e-3)),
            ("recipe_b1024", dict(R, batch=1024)),
            ("recipe_b16k_lr2e-3", dict(R, batch=16384, lr=2e-3)),
            ("recipe_folds5", dict(R, folds=5)),
            ("recipe_seeds3", dict(R, seeds=3)),
            ("recipe_iters20", dict(R, iters=20)),
            ("recipe_iters6", dict(R, iters=6)),
            ("recipe_tfdr0.02", dict(R, train_fdr=0.02)),
            ("recipe_tfdr0.005", dict(R, train_fdr=0.005)),
            ("recipe_mfrac0.75", dict(R, margin_frac=0.75)),
            ("recipe_mfrac0.25", dict(R, margin_frac=0.25)),
            ("recipe_wd1e-3", dict(R, wd=1e-3)),
            ("recipe_wd0", dict(R, wd=0.0)),
        ]
    if name == "combo":
        # The few knobs that were neutral-to-positive on every pool in the `nn` grid, alone
        # and together, against the recipe and the shipped baseline, with more seeds.
        R = dict(neg_ratio=3, neg_select="hybrid", warm_start=True, warm_epochs=5)
        out += [
            ("recipe", dict(R)),
            ("recipe_folds5", dict(R, folds=5)),
            ("recipe_mfrac0.75", dict(R, margin_frac=0.75)),
            ("recipe_folds5_mfrac0.75", dict(R, folds=5, margin_frac=0.75)),
            ("recipe_folds5_seeds3", dict(R, folds=5, seeds=3)),
            ("recipe_folds5_mfrac0.75_seeds3", dict(R, folds=5, margin_frac=0.75, seeds=3)),
        ]
    if name == "confirm":
        out += [
            ("neg3_random", dict(neg_ratio=3, neg_select="random")),
            ("neg3_hybrid", dict(neg_ratio=3, neg_select="hybrid")),
            ("warm5", dict(warm_start=True, warm_epochs=5)),
            ("warm5_neg3_hybrid", dict(warm_start=True, warm_epochs=5, neg_ratio=3, neg_select="hybrid")),
        ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pin")
    ap.add_argument("out_dir")
    ap.add_argument("tag")
    ap.add_argument("--features", default=None, help="file with one feature name per line")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--entrapment", action="store_true",
                    help="classify spike-in targets from the Proteins column and report the empirical FDP")
    ap.add_argument("--grid", default="quick")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = fs_lib.load_pin(args.pin, entrapment=args.entrapment)
    if args.rows:
        d = fs_lib.subsample(d, args.rows, seed=0)
        print(f"subsampled to {len(d['y'])} rows", flush=True)
    cols = None
    if args.features:
        want = [ln.strip() for ln in open(args.features, encoding="utf-8") if ln.strip()]
        idx = {n: i for i, n in enumerate(d["feat_cols"])}
        missing = [w for w in want if w not in idx]
        if missing:
            sys.exit(f"{len(missing)} feature(s) not in the PIN, first: {missing[:3]}")
        cols = [idx[w] for w in want]
        print(f"feature subset: {len(cols)} of {len(d['feat_cols'])}", flush=True)

    variants = grid(args.grid)
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        variants = [v for v in variants if v[0] in keep]
    out_csv = os.path.join(args.out_dir, f"{args.tag}_train_sweep.csv")
    for label, over in variants:
        for sd in [int(x) for x in args.seeds.split(",")]:
            t0 = time.time()
            row, r = fs_lib.bench_subset(label, d, cols, cfg={**over, "seed_base": sd})
            row.update(
                dataset=args.tag,
                seed=sd,
                rows=int(len(d["y"])),
                n_features_used=len(cols) if cols else len(d["feat_cols"]),
                cfg=json.dumps(over, sort_keys=True),
                when=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            print(f"[{label} seed {sd}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
