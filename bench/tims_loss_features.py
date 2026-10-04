#!/usr/bin/env python
"""Which evidence makes a true precursor look like a decoy (docs/TIMS_ROADMAP_bis.md, D2).

  tims_loss_features.py --loss loss.parquet --features features.parquet \
      --scored psms_scored.parquet [--category right_peak_low_score] [--out d2.tsv]

Populations, one feature row each at (candidate_id, selected_peak_rank):
  R   the D1 keys of --category (default: right peak, low score), their best MuMDIA row
  A   the D1 accepted keys
  Aq  A resampled to R's DIA-NN Precursor.Quantity distribution (abundance control)
  Dm  decoys resampled to R's rescore `score` distribution (the decoys R competes with)
  D   every decoy
Resampling: 20 quantile bins of R, as many draws per bin as R has, with replacement, seed 0.

Per feature (the 396 rescore inputs in features.parquet.schema.json): the ROC AUC of R against
A, Aq and Dm, of A against D (the target direction), and the medians. AUC 0.5 = no separation,
below 0.5 = R lower. "Missing evidence" is a feature on which R is far from Aq but close to Dm.
"""
import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) + P(x == y) / 2 (Mann-Whitney), NaNs dropped."""
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    r = rankdata(np.r_[x, y])
    return float((r[: len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def matched(pool: pd.DataFrame, ref: pd.Series, col: str, rng, bins: int = 20) -> pd.DataFrame:
    edges = np.unique(ref.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    rb = np.clip(np.searchsorted(edges, ref, side="right") - 1, 0, len(edges) - 2)
    pb = np.searchsorted(edges, pool[col], side="right") - 1
    pb[pool[col].to_numpy() == edges[-1]] = len(edges) - 2
    out = []
    for b, n in zip(*np.unique(rb, return_counts=True)):
        cand = np.flatnonzero(pb == b)
        if len(cand):
            out.append(pool.iloc[rng.choice(cand, n, replace=True)])
    return pd.concat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", required=True, help="tims_loss.py per-precursor parquet")
    ap.add_argument("--features", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--category", default="right_peak_low_score")
    ap.add_argument("--label-col", default="category")
    ap.add_argument("--out", help="per-feature TSV")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--save-pops", help="parquet of (pop, candidate_id, peak_rank) for D2b")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    feats = json.load(open(a.features + ".schema.json"))["feature_columns"]
    f = pq.read_table(a.features, columns=["candidate_id", "peak_rank", *feats]).to_pandas()
    s = pd.read_parquet(a.scored, columns=["candidate_id", "selected_peak_rank", "label", "score"])
    s = s.merge(f, left_on=["candidate_id", "selected_peak_rank"],
                right_on=["candidate_id", "peak_rank"])
    lo = pd.read_parquet(a.loss, columns=["candidate_id", a.label_col, "diann_quantity"]).dropna()
    lo["candidate_id"] = lo["candidate_id"].astype(s["candidate_id"].dtype)
    t = s.merge(lo, on="candidate_id")
    R, A = t[t[a.label_col] == a.category], t[t[a.label_col] == "accepted"]
    D = s[s["label"] == "decoy"]
    Aq = matched(A, R["diann_quantity"], "diann_quantity", rng)
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
                     "auc_R_A": auc(v["R"], v["A"]), "auc_R_Aq": auc(v["R"], v["Aq"]),
                     "auc_R_Dm": auc(v["R"], v["Dm"]), "auc_A_D": auc(v["A"], v["D"]),
                     **{f"med_{k}": float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
                        for k, x in v.items()}})
    o = pd.DataFrame(rows)
    # Missing evidence: R is as far from abundance-matched accepted targets as it is close to
    # score-matched decoys.
    o["missing"] = (o["auc_R_Aq"] - 0.5).abs() - (o["auc_R_Dm"] - 0.5).abs()
    o = o.sort_values("missing", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(o.head(a.top).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if a.out:
        o.to_csv(a.out, sep="\t", index=False)


if __name__ == "__main__":
    assert auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5
    main()
