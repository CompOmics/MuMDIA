"""Judge feature subsets on the rescoring objective: peptides at 1% FDR, decoy fraction.

Runs the re-implemented `nn_torch` worker (fs_lib.run_rescoring, worker defaults) on each
subset and appends one row per run to OUT_DIR/TAG_objective.csv. Optionally saves the
out-of-fold scores of a run (`--save-oof NAME`) so a later step can define "confident
targets" without re-training.

Subsets come from a JSON file {name: [feature, ...]} plus the built-ins `all`, `minimal`,
`rich` (the engine's FeatureSet enums). `--rows N` subsamples the pool for screening;
omit it for the full pool.

Usage:
  fs_objective.py PIN.parquet OUT_DIR TAG [--subsets S.json] [--only a,b] [--seeds 0,1]
                  [--rows 400000] [--save-oof all]
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


def builtin_subsets(repo_root: str, names: list[str]) -> dict[str, list[str]]:
    fam = fs_lib.family_map(repo_root)
    return {
        "all": list(names),
        "minimal": [n for n in names if fam.get(n) == "minimal"],
        "rich": [n for n in names if fam.get(n) in ("minimal", "rich")],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pin")
    ap.add_argument("out_dir")
    ap.add_argument("tag")
    ap.add_argument("--subsets", default=None)
    ap.add_argument("--only", default=None, help="comma-separated subset names to run")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--save-oof", default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = fs_lib.load_pin(args.pin)
    if args.rows:
        d = fs_lib.subsample(d, args.rows, seed=0)
        print(f"subsampled to {len(d['y'])} rows", flush=True)
    names = d["feat_cols"]
    idx_of = {n: i for i, n in enumerate(names)}
    subsets = builtin_subsets(args.repo, names)
    if args.subsets:
        with open(args.subsets, encoding="utf-8") as fh:
            for k, v in json.load(fh).items():
                subsets[k] = [n for n in v if n in idx_of]
    only = [s.strip() for s in args.only.split(",")] if args.only else list(subsets)
    seeds = [int(s) for s in args.seeds.split(",")]
    out_csv = os.path.join(args.out_dir, f"{args.tag}_objective.csv")
    cfg = {}
    if args.iters:
        cfg["iters"] = args.iters

    for name in only:
        feats = subsets[name]
        cols = [idx_of[n] for n in feats]
        for sd in seeds:
            t0 = time.time()
            row, r = fs_lib.bench_subset(name, d, cols, cfg={**cfg, "seed_base": sd})
            row.update(dataset=args.tag, seed=sd, rows=int(len(d["y"])), when=time.strftime("%Y-%m-%d %H:%M:%S"))
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            if args.save_oof and name == args.save_oof:
                np.save(os.path.join(args.out_dir, f"{args.tag}_oof_{name}_seed{sd}.npy"), r["score"].astype(np.float32))
            print(f"[{name} seed {sd}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
