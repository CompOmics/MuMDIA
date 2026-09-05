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
    ap.add_argument("--grid", default="quick")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = fs_lib.load_pin(args.pin)
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
