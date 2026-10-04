#!/usr/bin/env python
"""Target-versus-decoy separation of the ion-mobility features (docs/TIMS_ROADMAP.md, P5, P7).

  tims_im_features.py --features out/features.parquet --scored out/psms_scored.parquet \
      [--q 0.01] [--json out/tims_im_features.json]

For each `features.im_features` / `features.im_shape_features` column (the rescore input), it prints the ROC AUC of
target against decoy (0.5 = no separation; below 0.5 means lower values are more
target-like) in three populations, joined on candidate_id:
  all         every target against every decoy
  accepted    targets at q_value <= q against every decoy (the separation that can help)
  null        targets against decoys among rows scoring below the median decoy score,
              where almost every target is false too. An AUC far from 0.5 here means the
              feature tells the label apart without telling right from wrong: leakage
plus the median of each feature per population.
"""
import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

IM_FEATURES = [
    "im_error",
    "im_error_abs",
    "im_frag_mad",
    "ms1_im_error_abs",
    "ms1_frag_im_diff",
    "has_ms1_im",
    "im_elution_error_abs",
    "im_elution_sd",
    "im_elution_frag_sd",
    # P7 peak shape (features.im_shape_features)
    "im_frag_width",
    "im_frag_width_mad",
    "im_frag_overlap",
    "ms1_im_width",
    "ms1_frag_overlap",
    "ms1_frag_width_logratio",
]


def auc(target: pd.Series, decoy: pd.Series) -> float:
    y = np.r_[np.ones(len(target)), np.zeros(len(decoy))]
    x = np.r_[target.to_numpy(), decoy.to_numpy()]
    if len(target) == 0 or len(decoy) == 0 or np.all(x == x[0]):
        return float("nan")
    return float(roc_auc_score(y, x))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--json")
    a = ap.parse_args()

    cols = [c for c in IM_FEATURES if c in pq.read_schema(a.features).names]
    if not cols:
        raise SystemExit(f"{a.features} has no IM feature column (features.im_features off?)")
    f = pq.read_table(a.features, columns=["candidate_id", *cols]).to_pandas()
    s = pq.read_table(a.scored, columns=["candidate_id", "label", "score", "q_value"]).to_pandas()
    d = s.merge(f.drop_duplicates("candidate_id"), on="candidate_id", how="inner")
    tgt, dec = d[d.label == "target"], d[d.label == "decoy"]
    acc = tgt[tgt.q_value <= a.q]
    cut = dec.score.median()
    low = d[d.score < cut]
    pops = {
        "all": (tgt, dec),
        "accepted": (acc, dec),
        "null": (low[low.label == "target"], low[low.label == "decoy"]),
    }
    out = {"n": {k: [len(t), len(v)] for k, (t, v) in pops.items()}, "features": {}}
    print(f"rows {len(d)}: targets {len(tgt)} (accepted {len(acc)}), decoys {len(dec)}; "
          f"null = score < {cut:.3f} ({len(pops['null'][0])} targets, {len(pops['null'][1])} decoys)")
    print(f"{'feature':22s} {'AUC all':>8s} {'accepted':>9s} {'null':>7s}   "
          f"{'med acc':>9s} {'med tgt':>9s} {'med dec':>9s}")
    for c in cols:
        r = {k: auc(t[c], v[c]) for k, (t, v) in pops.items()}
        r["median"] = {"accepted": float(acc[c].median()), "target": float(tgt[c].median()),
                       "decoy": float(dec[c].median())}
        out["features"][c] = r
        m = r["median"]
        print(f"{c:22s} {r['all']:8.3f} {r['accepted']:9.3f} {r['null']:7.3f}   "
              f"{m['accepted']:9.4f} {m['target']:9.4f} {m['decoy']:9.4f}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
