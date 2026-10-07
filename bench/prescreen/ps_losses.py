"""Split the prescreen's losses of OFF-accepted precursors into retrieval and scoring losses.

Reads the X_RET arm (tag retrieval, no rescue) scores table: `retrieved`, `base_score`, and the
calibration flag, recomputes the score-only cutoff, and classifies every precursor accepted in
any OFF seed (precursor_q <= 0.01, target).

usage: ps_losses.py ROOT [ARM] [TARGET]
"""
import glob
import math
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = sys.argv[1]
ARM = sys.argv[2] if len(sys.argv) > 2 else "X_RET"
Q = float(sys.argv[3]) if len(sys.argv) > 3 else 0.54
t = pq.read_table(f"{ROOT}/{ARM}/survivors.scores.parquet",
                  columns=["candidate_id", "retrieved", "base_score", "calibration", "in_scope"]).to_pandas()
cal = np.sort(t.loc[t.calibration, "base_score"].to_numpy())
cut = cal[min(len(cal) - 1, math.ceil((len(cal) - 1) * Q))]
ref = set()
for s in glob.glob(f"{ROOT}/OFF/s*/psms_scored.parquet"):
    d = pq.read_table(s, columns=["candidate_id", "label", "precursor_q"]).to_pandas()
    d = d[(~d.label.str.lower().str.startswith("decoy")) & (d.precursor_q <= 0.01)]
    ref |= set(d.candidate_id.astype(int))
r = t[t.candidate_id.isin(ref)]
nr = ~r.retrieved
lost_ret = int((nr & (r.base_score > cut)).sum())
lost_both = int((nr & (r.base_score <= cut)).sum())
lost_score = int((r.retrieved & (r.base_score <= cut)).sum())
n = len(r)
print(f"{ROOT}/{ARM}: score-only cutoff {cut:.4f} at target {Q}; OFF references {n}")
print(f"  not retrieved: {int(nr.sum())} ({100 * nr.mean():.2f}%); of these the score keeps {lost_ret} "
      f"(retrieval-only losses, {100 * lost_ret / n:.2f}%) and rejects {lost_both}")
print(f"  retrieved but scored at or below the cutoff: {lost_score} (scoring-only losses, {100 * lost_score / n:.2f}%)")
print(f"  library: not retrieved {int((~t.retrieved).sum())} of {len(t)} ({100 * (~t.retrieved).mean():.2f}%), "
      f"unretrieved the score would keep {int(((~t.retrieved) & (t.base_score > cut)).sum())}")
