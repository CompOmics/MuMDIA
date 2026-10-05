"""Derive another preset's survivors from a prescreen scores table, without rescoring.

The cutoff rule is the stage's own: numpy "higher" quantile of the calibration half's scores at
the target, keep in-scope candidates with score > cutoff, out-of-scope candidates always.

usage: ps_derive.py SCORES.parquet TARGET OUT_SURVIVORS.parquet
"""
import json
import math
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

src, q, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
t = pq.read_table(src, columns=["candidate_id", "label", "score", "in_scope", "calibration"])
score = t.column("score").to_numpy(zero_copy_only=False)
scope = t.column("in_scope").to_numpy(zero_copy_only=False)
cal = np.sort(score[t.column("calibration").to_numpy(zero_copy_only=False)])
cut = float(cal[min(len(cal) - 1, math.ceil((len(cal) - 1) * q))])
keep = (~scope) | (score > cut)
sub = t.filter(pa.array(keep)).select(["candidate_id", "label"])
pq.write_table(sub, out, compression="snappy")
stats = {"derived_from": src, "target": q, "cutoff": cut, "survivors": sub.num_rows,
         "screened": t.num_rows, "total_reduction": 1 - sub.num_rows / t.num_rows,
         "calibration_n": int(len(cal))}
json.dump({"stats": stats}, open(out + ".report.json", "w"), indent=1)
print(json.dumps(stats))
