#!/usr/bin/env python
"""Which evidence makes a true precursor look like a decoy.

  fs_loss_evidence.py --loss loss.parquet --features out_run/features.parquet \
      --scored out_run/psms_scored.parquet [--source N] [--category right_peak_low_score] \
      [--out evidence.tsv] [--save-pops pops.parquet]

--loss is the per-precursor output of bench/tims/tims_loss.py: one row per reference precursor
with the best MuMDIA row's `candidate_id`, the loss `category` and the reference abundance
`ref_quantity`. The reference precursors are an independent identification of the same run,
so the ones MuMDIA rejects are mostly true precursors.

Populations, one feature row each at (candidate_id, selected_peak_rank) of psms_scored, joined
to features.parquet on (candidate_id, peak_rank):
  R   the loss keys of --category (default: right peak, low score), their best MuMDIA row
  A   the accepted loss keys
  Aq  A resampled to R's reference abundance (`ref_quantity`) distribution (abundance control)
  Dm  decoys resampled to R's rescore `score` distribution (the decoys R competes with)
  D   every decoy
Resampling: 20 quantile bins of R, as many draws per bin as R has, with replacement, seed 0.

Per feature (the `feature_columns` of features.parquet.schema.json, the rescore input): the
rank AUC (fs_lib.rank_auc) of R against A, Aq and Dm, of A against D (the target direction),
and the medians. AUC 0.5 = no separation, below 0.5 = R lower. "Missing evidence" is a feature
on which R is far from Aq but close to Dm; the table is sorted by
|AUC(R, Aq) - 0.5| - |AUC(R, Dm) - 0.5|.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fs_lib  # noqa: E402


def matched(pool: pd.DataFrame, ref: pd.Series, col: str, rng, bins: int = 20) -> pd.DataFrame:
    """Rows of `pool` drawn so that `pool[col]` follows the distribution of `ref`.

    `ref` is cut into `bins` quantile bins; each bin draws, with replacement, as many pool rows
    from the same `col` interval as `ref` has in it. A bin with no pool row draws nothing."""
    edges = np.unique(ref.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    rb = np.clip(np.searchsorted(edges, ref, side="right") - 1, 0, len(edges) - 2)
    pb = np.searchsorted(edges, pool[col], side="right") - 1
    pb[pool[col].to_numpy() == edges[-1]] = len(edges) - 2
    out = []
    for b, n in zip(*np.unique(rb, return_counts=True)):
        cand = np.flatnonzero(pb == b)
        if len(cand):
            out.append(pool.iloc[rng.choice(cand, n, replace=True)])
    return pd.concat(out) if out else pool.iloc[:0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--loss", required=True, help="bench/tims/tims_loss.py per-precursor parquet")
    ap.add_argument("--features", required=True, help="features.parquet (with its .schema.json)")
    ap.add_argument("--scored", required=True, help="psms_scored.parquet")
    ap.add_argument("--source", type=int, help="keep only this `source` of a pooled psms_scored")
    ap.add_argument("--category", default="right_peak_low_score")
    ap.add_argument("--label-col", default="category")
    ap.add_argument("--out", help="per-feature TSV")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--save-pops", help="parquet of (pop, candidate_id, peak_rank)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    feats = fs_lib.read_feature_columns(a.features)
    f = pq.read_table(a.features, columns=["candidate_id", "peak_rank", *feats]).to_pandas()
    s = pd.read_parquet(a.scored, columns=["candidate_id", "selected_peak_rank", "label", "score",
                                           "source"])
    if a.source is not None:
        s = s[s["source"] == a.source]
    s = s.drop(columns="source").merge(f, left_on=["candidate_id", "selected_peak_rank"],
                                       right_on=["candidate_id", "peak_rank"])
    lo = pd.read_parquet(a.loss, columns=["candidate_id", a.label_col, "ref_quantity"]).dropna()
    lo["candidate_id"] = lo["candidate_id"].astype(s["candidate_id"].dtype)
    t = s.merge(lo, on="candidate_id")
    R, A = t[t[a.label_col] == a.category], t[t[a.label_col] == "accepted"]
    if R.empty or A.empty:
        raise SystemExit(f"empty population: R {len(R)} ({a.category}), A {len(A)} (accepted)")
    D = s[s["label"] == "decoy"]
    Aq = matched(A, R["ref_quantity"], "ref_quantity", rng)
    Dm = matched(D, R["score"], "score", rng)
    print(f"R {len(R)}  A {len(A)}  Aq {len(Aq)}  Dm {len(Dm)}  D {len(D)}; median score "
          f"R {R.score.median():.3f} Dm {Dm.score.median():.3f} A {A.score.median():.3f}")

    if a.save_pops:
        pd.concat([p[["candidate_id", "peak_rank"]].assign(pop=k)
                   for k, p in dict(R=R, A=A, Aq=Aq, Dm=Dm).items()]).to_parquet(a.save_pops, index=False)

    rows = []
    for c in feats:
        v = {k: p[c].to_numpy(float) for k, p in dict(R=R, A=A, Aq=Aq, Dm=Dm, D=D).items()}
        rows.append({"feature": c,
                     "auc_R_A": fs_lib.rank_auc(v["R"], v["A"]),
                     "auc_R_Aq": fs_lib.rank_auc(v["R"], v["Aq"]),
                     "auc_R_Dm": fs_lib.rank_auc(v["R"], v["Dm"]),
                     "auc_A_D": fs_lib.rank_auc(v["A"], v["D"]),
                     **{f"med_{k}": float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
                        for k, x in v.items()}})
    o = pd.DataFrame(rows)
    # Missing evidence: R is as far from abundance-matched accepted targets as it is close to
    # score-matched decoys.
    o["missing"] = (o["auc_R_Aq"] - 0.5).abs() - (o["auc_R_Dm"] - 0.5).abs()
    o = o.sort_values(["missing", "feature"], ascending=[False, True], kind="stable")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(o.head(a.top).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if a.out:
        o.to_csv(a.out, sep="\t", index=False)


if __name__ == "__main__":
    assert fs_lib.rank_auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert fs_lib.rank_auc(np.array([1.0]), np.array([1.0])) == 0.5
    main()
