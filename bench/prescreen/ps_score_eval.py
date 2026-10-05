"""Reduction and retention of a prescreen scores table at several targets.

usage: ps_score_eval.py NAME SCORES_PARQUET INPUT_TABLE TIME_FILE REF_SCORED [REF_SCORED ...]

Uses the stage's rule (numpy "higher" quantile of the calibration half, keep score > cutoff)
on the `score` column, and matches the unfiltered run's accepted precursors (target,
precursor_q <= 0.01, seeds united) by (peptidoform, charge).
"""
import json
import math
import sys

import numpy as np
import pyarrow.parquet as pq

name, scores, inp, timef, *refs = sys.argv[1:]
cols = pq.read_schema(inp).names
idc = "candidate_id" if "candidate_id" in cols else "id"
t = pq.read_table(inp, columns=[idc, "peptidoform", "charge"]).to_pandas()
key = dict(zip(t[idc].astype(int), zip(t.peptidoform.str.removeprefix("DECOY_"), t.charge.astype(int))))
ref = set()
for r in refs:
    d = pq.read_table(r, columns=["label", "peptidoform", "charge", "precursor_q"]).to_pandas()
    d = d[(~d.label.astype(str).str.lower().str.startswith("decoy")) & (d.precursor_q <= 0.01)]
    ref |= set(zip(d.peptidoform.str.removeprefix("DECOY_"), d.charge.astype(int)))
sc = pq.read_table(scores, columns=["candidate_id", "score", "calibration", "label"]).to_pandas()
is_ref = sc.candidate_id.map(lambda c: key.get(int(c)) in ref).to_numpy()
dec = sc.label.astype(str).str.lower().str.startswith("decoy").to_numpy()
score = sc.score.to_numpy()
cal = np.sort(score[sc.calibration.to_numpy()])
wall = rss = None
for line in open(timef):
    if "Elapsed (wall" in line:
        wall = 0.0
        for x in line.rsplit(" ", 1)[1].strip().split(":"):
            wall = wall * 60 + float(x)
    if "Maximum resident" in line:
        rss = int(line.rsplit(" ", 1)[1]) / 1048576
out = {"dataset": name, "candidates": len(sc), "reference": int(is_ref.sum()), "wall_s": wall,
       "peak_gb": round(rss, 2) if rss else None, "targets": {}}
for q in [0.25, 0.40, 0.52, 0.54, 0.75]:
    cut = cal[min(len(cal) - 1, math.ceil((len(cal) - 1) * q))]
    keep = score > cut
    out["targets"][str(q)] = {
        "removed_pct": round(100 * (1 - keep.mean()), 2),
        "reference_kept_pct": round(100 * keep[is_ref].mean(), 3),
        "target_kept_pct": round(100 * keep[~dec].mean(), 2),
        "decoy_kept_pct": round(100 * keep[dec].mean(), 2),
    }
print(json.dumps(out))
