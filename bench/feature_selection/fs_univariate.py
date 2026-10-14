#!/usr/bin/env python
"""Univariate separation of every feature column: rank AUC per feature and population.

  fs_univariate.py --features out_run/features.parquet [--scored out_run/psms_scored.parquet] \
      [--source N] [--q 0.01] [--q-col q_value] [--match REGEX] \
      [--label-table T.parquet --label-col COL --positive VALUE [--negative VALUE]] \
      [--top 40] [--out univariate.tsv] [--json univariate.json]

The feature columns are the `feature_columns` of `<features>.schema.json` (the rescore input
list), or, without that companion, every non-metadata column (fs_lib.read_feature_columns).
`--match` keeps the columns whose name matches a regular expression, e.g. `^im_|ms1_.*im`.

Rows. Without --scored, one row per features.parquet row (every candidate and every retained
peak). With --scored, one row per psms_scored row, joined to its feature row on
(candidate_id, selected_peak_rank) = (candidate_id, peak_rank), so each candidate is
described at the peak the rescorer selected. For an experiment-wide psms_scored, pass
--source with that run's features.parquet.

Populations, target-decoy mode (the default; the label is the `label` column):
  all         every target against every decoy
  accepted    targets at --q-col <= q against every decoy (needs --scored): the separation
              that can help the rescorer
  null        targets against decoys among rows scoring below the median decoy score (needs
              --scored), where almost every target is false too. An AUC far from 0.5 here
              means the feature tells the label apart without telling right from wrong:
              a leakage warning
Given-label mode (--label-table): one population, `all`, of rows whose --label-col equals
--positive against rows that equal --negative (or every other row when --negative is not
given). The label table needs `candidate_id` and is joined on (candidate_id, peak_rank) when
it has `peak_rank`, on (candidate_id, selected_peak_rank) when it has that instead, and on
candidate_id alone otherwise. The per-precursor output of bench/tims/tims_loss.py
(`category`, joined on candidate_id) is such a table.

Per feature it prints the rank AUC of each population (fs_lib.rank_auc: 0.5 = no separation,
below 0.5 = lower values are more positive-like, NaN-safe) and the medians of the positive
and negative rows of `all`, sorted by |AUC(all) - 0.5|. This is a univariate screen: it does
not say what a feature adds to the others (fs_importance.py, fs_subsets.py do that).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fs_lib  # noqa: E402


def load_rows(a) -> pd.DataFrame:
    """The row table with candidate_id, peak_rank and the label columns, before features."""
    if a.scored:
        cols = ["candidate_id", "selected_peak_rank", "label", "score", "source", a.q_col]
        s = pd.read_parquet(a.scored, columns=list(dict.fromkeys(cols)))
        if a.source is not None:
            s = s[s["source"] == a.source]
        return s.rename(columns={"selected_peak_rank": "peak_rank", a.q_col: "q"}).drop(columns="source")
    if a.source is not None:
        raise SystemExit("--source selects rows of a pooled psms_scored; pass --scored")
    return pd.read_parquet(a.features, columns=["candidate_id", "peak_rank", "label"])


def given_label(rows: pd.DataFrame, a) -> pd.DataFrame:
    """Attach `y` (True = positive) from --label-table; rows outside both classes are dropped."""
    names = pq.read_schema(a.label_table).names
    on = ["candidate_id"]
    rank_col = next((c for c in ("peak_rank", "selected_peak_rank") if c in names), None)
    cols = ["candidate_id", a.label_col] + ([rank_col] if rank_col else [])
    lt = pd.read_parquet(a.label_table, columns=cols).dropna(subset=["candidate_id"])
    lt["candidate_id"] = lt["candidate_id"].astype(rows["candidate_id"].dtype)
    if rank_col:
        lt = lt.rename(columns={rank_col: "peak_rank"})
        lt["peak_rank"] = lt["peak_rank"].astype(rows["peak_rank"].dtype)
        on.append("peak_rank")
    lt = lt.drop_duplicates(on)
    lab = lt[a.label_col].astype(str)
    pos = lab == a.positive
    neg = lab == a.negative if a.negative is not None else ~pos
    lt = lt[pos | neg].assign(y=pos[pos | neg])
    return rows.merge(lt[on + ["y"]], on=on, how="inner")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True, help="features.parquet (with its .schema.json)")
    ap.add_argument("--scored", help="psms_scored.parquet; adds the accepted and null populations")
    ap.add_argument("--source", type=int, help="keep only this `source` of a pooled psms_scored")
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--q-col", default="q_value", choices=["q_value", "run_psm_q"])
    ap.add_argument("--match", help="regular expression; keep the feature columns it matches")
    ap.add_argument("--label-table", help="parquet with candidate_id and --label-col")
    ap.add_argument("--label-col", default="category")
    ap.add_argument("--positive", help="--label-col value of the positive class")
    ap.add_argument("--negative", help="--label-col value of the negative class (default: the rest)")
    ap.add_argument("--top", type=int, default=40, help="rows printed (0 = all)")
    ap.add_argument("--out", help="per-feature TSV")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.label_table and a.positive is None:
        raise SystemExit("--label-table needs --positive")

    feats = fs_lib.read_feature_columns(a.features)
    if a.match:
        rx = re.compile(a.match)
        feats = [c for c in feats if rx.search(c)]
    if not feats:
        raise SystemExit(f"{a.features}: no feature column selected")

    rows = load_rows(a)
    if a.label_table:
        rows = given_label(rows, a)
    else:
        rows = rows.assign(y=rows["label"] == "target")
    f = pq.read_table(a.features, columns=["candidate_id", "peak_rank", *feats]).to_pandas()
    d = rows.merge(f.drop_duplicates(["candidate_id", "peak_rank"]),
                   on=["candidate_id", "peak_rank"], how="inner")
    P, N = d[d["y"]], d[~d["y"]]
    pops = {"all": (P, N)}
    if a.scored and not a.label_table:
        cut = N["score"].median()
        low = d[d["score"] < cut]
        pops["accepted"] = (P[P["q"] <= a.q], N)
        pops["null"] = (low[low["y"]], low[~low["y"]])
        print(f"null population: score < {cut:.4g} (median decoy score)")
    print(f"joined rows {len(d)} of {len(rows)}; " + ", ".join(
        f"{k}: {len(p)} positive / {len(n)} negative" for k, (p, n) in pops.items()))

    out = []
    for c in feats:
        r = {"feature": c}
        for k, (p, n) in pops.items():
            r[f"auc_{k}"] = fs_lib.rank_auc(p[c].to_numpy(float), n[c].to_numpy(float))
        for k, x in (("pos", P[c]), ("neg", N[c])):
            r[f"med_{k}"] = float(x.median()) if x.notna().any() else np.nan
        out.append(r)
    o = pd.DataFrame(out)
    o["sep_all"] = (o["auc_all"] - 0.5).abs()
    o = o.sort_values(["sep_all", "feature"], ascending=[False, True], kind="stable") \
        .drop(columns="sep_all").reset_index(drop=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    show = o if a.top <= 0 else o.head(a.top)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if a.out:
        o.to_csv(a.out, sep="\t", index=False)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"n": {k: [len(p), len(n)] for k, (p, n) in pops.items()},
                       "features": json.loads(o.to_json(orient="records"))}, fh, indent=1)


if __name__ == "__main__":
    assert fs_lib.rank_auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert fs_lib.rank_auc(np.array([1.0]), np.array([1.0])) == 0.5
    main()
