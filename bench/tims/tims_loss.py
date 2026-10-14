#!/usr/bin/env python
"""Precursor-level loss classifier: why MuMDIA misses a precursor that a reference accepts.

  tims_loss.py --reference reference_report.parquet --run-dir out_run \
      [--scored psms_scored.parquet] [--extracted psms_extracted.parquet] \
      [--lib fragment_library_precursors.parquet] [--windows run_windows.parquet] \
      [--peaks psms_extracted.parquet.peaks.parquet] [--source N] [--q-col q_value] \
      --out loss.parquet [--json loss.json]

The reference is an independent identification table of the same run; `tims_ref.py` lists
the column names. This script reads `Modified.Sequence`, `Stripped.Sequence`,
`Precursor.Charge`, `RT` (minutes), `Q.Value` and `Precursor.Quantity`, and `IM` and `Decoy`
when present.

Unit: a reference precursor at the threshold (Q.Value <= q, Decoy == 0) keyed on the I/L-merged
stripped sequence plus the charge; when several reference peptidoforms share a key, the lowest
Q.Value wins. Each key is paired with the best-scoring MuMDIA target row of that key in
psms_scored (highest `score`), at that row's `selected_peak_rank` in psms_extracted.
"Accepted" means that row's `--q-col` (default `q_value`, the pooled PSM q) <= q. "Right peak"
means its apex lies within --rt-tol seconds of the reference RT. RT and 1/K0 windows come from
run_windows for the row's candidate, or, when nothing was extracted, for the library candidate
of the key (the exact peptidoform when present).

Categories, in order of precedence:
  not_in_library          no library target of that key
  not_extracted_rt_out    no scored target row; reference RT outside the candidate's RT window
  not_extracted_im_out    ... RT inside, reference 1/K0 outside the 1/K0 window
  not_extracted_in_win    ... both inside: a gate or presence loss in extract
  accepted                best row passes q
  right_peak_low_score    rejected, apex within rt-tol of the reference RT
  wrong_peak_in_window    rejected, wrong apex, reference RT inside the RT window
  wrong_peak_out_window   rejected, wrong apex, reference RT outside the RT window
The `--out` table carries the best row's acceptance q as `q` and the reference values as
`ref_pf`, `ref_rt` (s), `ref_im`, `ref_q` and `ref_quantity`.
A key without a 1/K0 window (a run without ion mobility, or no ion-mobility calibration) or
without a reference 1/K0 counts as inside the 1/K0 window, so on such a run the
not_extracted_im_out category stays empty.

With --peaks (the `<psms_extracted>.peaks.parquet` that extract writes when
`extract.retain_top_peaks > 1`), each row also gets `oracle_rank`: the lowest peak_rank
whose apex lies within rt-tol of the reference RT (null if none).

For an experiment-wide psms_scored, pass `--source` (the run's index in the pooled table) with
that run's `--extracted`, `--windows` and `--peaks`, and `--q-col run_psm_q`, which is the
per-run PSM q. The grouped q columns of a pooled table are not a per-run unit.

This is a precursor-level view against the reference. The stage-by-stage ladder of every
library candidate (in library, extracted, competed, scored) comes from `mumdia audit`
(`candidate_audit.parquet`) and from each stage's `<artifact>.report.json`.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tims_ref import key, read_reference, reference_pf, strip  # noqa: E402

CATS = ["not_in_library", "not_extracted_rt_out", "not_extracted_im_out", "not_extracted_in_win",
        "accepted", "right_peak_low_score", "wrong_peak_in_window", "wrong_peak_out_window"]


def load_reference(path: str, q: float) -> pd.DataFrame:
    d = read_reference(path, ["Modified.Sequence", "Stripped.Sequence", "Precursor.Charge", "RT",
                              "Q.Value", "Precursor.Quantity"], optional=("IM", "Decoy"))
    d = d[(d["Q.Value"] <= q) & (d["Decoy"] == 0)].copy()
    d["key"] = key(d["Stripped.Sequence"], d["Precursor.Charge"])
    d = d.sort_values(["key", "Q.Value", "Modified.Sequence"]).drop_duplicates("key")
    d["ref_pf"] = d["Modified.Sequence"].map(reference_pf)
    d["ref_rt"] = d["RT"] * 60.0
    d["quintile"] = pd.qcut(d["Precursor.Quantity"].rank(method="first"), 5, labels=False) + 1
    return d.rename(columns={"IM": "ref_im", "Precursor.Quantity": "ref_quantity",
                             "Q.Value": "ref_q"})[
        ["key", "ref_pf", "ref_rt", "ref_im", "ref_q", "ref_quantity", "quintile"]]


def classify(d, scored, ext, lib, win, q, q_col, rt_tol, peaks=None, exact=False) -> pd.DataFrame:
    """exact=True pairs each key only with rows of the reference's own peptidoform."""
    t = scored[scored["label"] == "target"].copy()
    t["key"] = key(t["peptidoform"].map(strip), t["charge"])
    if exact:
        t = t[t["peptidoform"] == t["key"].map(d.set_index("key")["ref_pf"])]
    t = t.merge(ext, left_on=["candidate_id", "selected_peak_rank"],
                right_on=["candidate_id", "peak_rank"], how="left")
    t = t[t["key"].isin(d["key"])]
    best = t.sort_values(["key", "score", "candidate_id"], ascending=[True, False, True]) \
        .drop_duplicates("key")
    best = best[["key", "candidate_id", "peptidoform", "selected_peak_rank", "apex_rt", "apex_im",
                 "rt_pred_cal", "n_matched_fragments", "score", q_col, "precursor_q",
                 "peptide_q_value"]].rename(columns={"peptidoform": "mumdia_pf", q_col: "q"})

    # Exact-peptidoform join: the best target row of the reference peptidoform itself.
    ex = t.merge(d[["key", "ref_pf"]], left_on=["key", "peptidoform"], right_on=["key", "ref_pf"])
    ex = ex.sort_values(["key", "score", "candidate_id"], ascending=[True, False, True]) \
        .drop_duplicates("key")
    ex = ex[["key", q_col]].rename(columns={q_col: "exact_pf_q"})

    lt = lib[lib["label"] == "target"].copy()
    lt["key"] = key(lt["peptidoform"].map(strip), lt["charge"])
    lt = lt[lt["key"].isin(d["key"])].merge(d[["key", "ref_pf"]], on="key")
    lt["exact"] = lt["peptidoform"] == lt["ref_pf"]
    lt = lt.sort_values(["key", "exact", "candidate_id"], ascending=[True, False, True]) \
        .drop_duplicates("key")[["key", "candidate_id", "exact"]] \
        .rename(columns={"candidate_id": "lib_candidate_id", "exact": "exact_pf_in_library"})

    o = d.merge(lt, on="key", how="left").merge(best, on="key", how="left").merge(ex, on="key", how="left")
    o["window_candidate"] = o["candidate_id"].fillna(o["lib_candidate_id"])
    o = o.merge(win, left_on="window_candidate", right_on="candidate_id", how="left",
                suffixes=("", "_w"))
    o["exact_pf_match"] = o["mumdia_pf"] == o["ref_pf"]
    o["rt_err"] = o["apex_rt"] - o["ref_rt"]
    o["im_err"] = o["apex_im"] - o["ref_im"]
    o["rt_pred_err"] = o["rt_pred_cal_w"] - o["ref_rt"]
    rt_in = (o["ref_rt"] >= o["rt_lo"]) & (o["ref_rt"] <= o["rt_hi"])
    # No 1/K0 window or no reference 1/K0: nothing to be outside of.
    im_in = (o["im_lo"].isna() | o["ref_im"].isna()
             | ((o["ref_im"] >= o["im_lo"]) & (o["ref_im"] <= o["im_hi"])))
    o["ref_rt_in_window"], o["ref_im_in_window"] = rt_in, im_in
    right = o["rt_err"].abs() <= rt_tol
    conds = [o["lib_candidate_id"].isna(),
             o["candidate_id"].isna() & ~rt_in,
             o["candidate_id"].isna() & ~im_in,
             o["candidate_id"].isna(),
             o["q"] <= q,
             right, rt_in]
    o["category"] = np.select(conds, CATS[:7], CATS[7])
    if peaks is not None:
        p = peaks.merge(o[["candidate_id", "ref_rt"]].dropna(), on="candidate_id")
        p = p[(p["apex_rt"] - p["ref_rt"]).abs() <= rt_tol]
        o = o.merge(p.groupby("candidate_id")["peak_rank"].min().rename("oracle_rank"),
                    left_on="candidate_id", right_index=True, how="left")
    return o.drop(columns=["candidate_id_w"])


def summary(o: pd.DataFrame) -> pd.DataFrame:
    g = o.groupby("category")
    s = pd.DataFrame({
        "n": g.size(),
        "share": g.size() / len(o),
        "median_quintile": g["quintile"].median(),
        "median_abs_rt_err_s": g["rt_err"].apply(lambda x: x.abs().median()),
        "median_abs_im_err": g["im_err"].apply(lambda x: x.abs().median()),
        "im_err_gt_0.02": g["im_err"].apply(lambda x: (x.abs() > 0.02).mean() if x.notna().any() else np.nan),
        "p95_abs_rt_pred_err_s": g["rt_pred_err"].apply(lambda x: x.abs().quantile(0.95)),
        "median_n_matched": g["n_matched_fragments"].median(),
        "q_quartiles": g["q"].apply(lambda x: tuple(np.round(x.quantile([.25, .5, .75]), 3))
                                    if x.notna().any() else None),
        "exact_pf_match": g["exact_pf_match"].mean(),
    }).reindex(CATS).dropna(how="all", subset=["n"])
    s["n"] = s["n"].astype(int)
    return s


def main() -> None:
    warnings.simplefilter("ignore", RuntimeWarning)  # medians of the all-null not-extracted groups
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", required=True, help="reference precursor table (parquet)")
    ap.add_argument("--run-dir", required=True, help="single-run output directory")
    ap.add_argument("--scored", help="default <run-dir>/psms_scored.parquet")
    ap.add_argument("--extracted", help="default <run-dir>/psms_extracted.parquet")
    ap.add_argument("--lib", help="default <run-dir>/fragment_library_precursors.parquet")
    ap.add_argument("--windows", help="default <run-dir>/run_windows.parquet")
    ap.add_argument("--peaks", help="<psms_extracted>.peaks.parquet (extract.retain_top_peaks > 1)")
    ap.add_argument("--source", type=int, help="keep only this `source` of a pooled psms_scored")
    ap.add_argument("--q-col", default="q_value", choices=["q_value", "run_psm_q"],
                    help="psms_scored column that decides 'accepted' (default q_value)")
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--rt-tol", type=float, default=5.0, help="right-peak tolerance in seconds")
    ap.add_argument("--out", help="per-precursor parquet with the category label")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = a.run_dir.rstrip("/\\")
    scored = pd.read_parquet(a.scored or f"{r}/psms_scored.parquet",
                             columns=["candidate_id", "peptidoform", "charge", "label", "score",
                                      "source", a.q_col, "precursor_q", "peptide_q_value", "selected_peak_rank"])
    if a.source is not None:
        scored = scored[scored["source"] == a.source]
    ext = pd.read_parquet(a.extracted or f"{r}/psms_extracted.parquet",
                          columns=["candidate_id", "peak_rank", "apex_rt", "apex_im", "rt_pred_cal",
                                   "n_matched_fragments"])
    lib = pd.read_parquet(a.lib or f"{r}/fragment_library_precursors.parquet",
                          columns=["candidate_id", "peptidoform", "charge", "label"])
    win = pd.read_parquet(a.windows or f"{r}/run_windows.parquet",
                          columns=["candidate_id", "rt_pred_cal", "rt_lo", "rt_hi", "im_lo", "im_hi"])
    peaks = pd.read_parquet(a.peaks, columns=["candidate_id", "peak_rank", "apex_rt"]) if a.peaks else None
    d = load_reference(a.reference, a.q)
    o = classify(d, scored, ext, lib, win, a.q, a.q_col, a.rt_tol, peaks)
    o["category_exact_pf"] = o["key"].map(
        classify(d, scored, ext, lib, win, a.q, a.q_col, a.rt_tol, exact=True)
        .set_index("key")["category"])
    s = summary(o)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(f"reference precursors at q <= {a.q} (I/L stripped sequence, charge): {len(o)}; "
          f"MuMDIA acceptance on {a.q_col}")
    print(f"exact-peptidoform join: reference peptidoform in library {o['exact_pf_in_library'].mean():.3f}, "
          f"best row is that peptidoform {o['exact_pf_match'].mean():.3f}, "
          f"accepted on the exact peptidoform {(o['exact_pf_q'] <= a.q).sum()}")
    print(s.to_string(float_format=lambda x: f"{x:.4g}"))
    print("\naccepted share per reference Precursor.Quantity quintile (1 = faintest):")
    print((o["category"] == "accepted").groupby(o["quintile"]).mean().round(3).to_string())
    print("\ncategory (best row of the key) against category_exact_pf (the reference peptidoform only):")
    print(pd.crosstab(o["category"], o["category_exact_pf"]).reindex(index=CATS, columns=CATS)
          .dropna(how="all").dropna(axis=1, how="all").fillna(0).astype(int).to_string())
    if "oracle_rank" in o:
        print("\noracle_rank (reference apex among the top-K peaks) per category:")
        print(o.groupby("category")["oracle_rank"].value_counts(dropna=False).unstack(fill_value=0).to_string())
    if a.out:
        o.to_parquet(a.out, index=False)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"n": len(o), "q": a.q, "q_col": a.q_col,
                       "summary": json.loads(s.to_json(orient="index"))}, fh, indent=2)


if __name__ == "__main__":
    main()
