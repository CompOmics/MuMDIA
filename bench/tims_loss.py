#!/usr/bin/env python
"""Precursor-level loss classifier against DIA-NN (docs/TIMS_ROADMAP_bis.md, D1).

  tims_loss.py --diann output_diann/report.parquet --run-dir output_mumdia_p6_best \
      [--scored psms_scored.parquet] [--extracted psms_extracted.parquet] \
      [--peaks psms_extracted.parquet.peaks.parquet] --out loss.parquet [--json loss.json]

Unit: a DIA-NN 1% precursor (Q.Value <= q, Decoy == 0) keyed on the I/L-merged stripped
sequence plus the charge; when several DIA-NN modforms share a key, the lowest Q.Value wins.
Each key is paired with the best-scoring MuMDIA target row of that key in psms_scored
(highest `score`). "Accepted" means that row's `q_value` (pooled PSM q) <= q. "Right peak"
means its apex lies within --rt-tol seconds of DIA-NN's RT (minutes x 60). RT and IM windows
come from run_windows for the row's candidate, or, when nothing was extracted, for the
library candidate of the key (the exact peptidoform when present).

Categories, in order of precedence:
  not_in_library          no library target of that key
  not_extracted_rt_out    no scored target row; DIA-NN RT outside the candidate's RT window
  not_extracted_im_out    ... RT inside, DIA-NN IM outside the IM window
  not_extracted_in_win    ... both inside: a gate or presence loss in extract
  accepted                best row q_value <= q
  right_peak_low_score    rejected, apex within rt-tol of DIA-NN RT
  wrong_peak_in_window    rejected, wrong apex, DIA-NN RT inside the RT window
  wrong_peak_out_window   rejected, wrong apex, DIA-NN RT outside the RT window

With --peaks (the retain_top_peaks sidecar of the same extract), each row also gets
`oracle_rank`: the lowest peak_rank whose apex lies within rt-tol of DIA-NN RT (null if none).
"""
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd

_MOD = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_UNIMOD = {"UniMod:4": "Carbamidomethyl", "UniMod:35": "Oxidation", "UniMod:1": "Acetyl"}

CATS = ["not_in_library", "not_extracted_rt_out", "not_extracted_im_out", "not_extracted_in_win",
        "accepted", "right_peak_low_score", "wrong_peak_in_window", "wrong_peak_out_window"]


def strip(pf: str) -> str:
    return _MOD.sub("", pf.removeprefix("DECOY_")).replace("-", "").replace(".", "")


def diann_pf(ms: str) -> str:
    """DIA-NN Modified.Sequence to the MuMDIA ProForma-lite peptidoform."""
    ms = re.sub(r"^\(UniMod:1\)", "[Acetyl]-", ms)
    return re.sub(r"\((UniMod:\d+)\)", lambda m: f"[{_UNIMOD.get(m[1], m[1])}]", ms)


def key(seq: pd.Series, z: pd.Series) -> pd.Series:
    return seq.str.replace("I", "L") + "/" + z.astype(str)


def load_diann(path: str, q: float) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=["Modified.Sequence", "Stripped.Sequence", "Precursor.Charge",
                                       "RT", "IM", "Q.Value", "Decoy", "Precursor.Quantity"])
    d = d[(d["Q.Value"] <= q) & (d["Decoy"] == 0)].copy()
    d["key"] = key(d["Stripped.Sequence"], d["Precursor.Charge"])
    d = d.sort_values(["key", "Q.Value", "Modified.Sequence"]).drop_duplicates("key")
    d["diann_pf"] = d["Modified.Sequence"].map(diann_pf)
    d["diann_rt"] = d["RT"] * 60.0
    d["quintile"] = pd.qcut(d["Precursor.Quantity"].rank(method="first"), 5, labels=False) + 1
    return d.rename(columns={"IM": "diann_im", "Precursor.Quantity": "diann_quantity",
                             "Q.Value": "diann_q"})[
        ["key", "diann_pf", "diann_rt", "diann_im", "diann_q", "diann_quantity", "quintile"]]


def classify(d, scored, ext, lib, win, q, rt_tol, peaks=None, exact=False) -> pd.DataFrame:
    """exact=True pairs each key only with rows of DIA-NN's own peptidoform."""
    t = scored[scored["label"] == "target"].copy()
    t["key"] = key(t["peptidoform"].map(strip), t["charge"])
    if exact:
        t = t[t["peptidoform"] == t["key"].map(d.set_index("key")["diann_pf"])]
    t = t.merge(ext, left_on=["candidate_id", "selected_peak_rank"],
                right_on=["candidate_id", "peak_rank"], how="left")
    t = t[t["key"].isin(d["key"])]
    best = t.sort_values(["key", "score", "candidate_id"], ascending=[True, False, True]) \
        .drop_duplicates("key")
    best = best[["key", "candidate_id", "peptidoform", "selected_peak_rank", "apex_rt", "apex_im",
                 "rt_pred_cal", "n_matched_fragments", "score", "q_value", "precursor_q",
                 "peptide_q_value"]].rename(columns={"peptidoform": "mumdia_pf"})

    # Exact-peptidoform join: the best target row of the DIA-NN peptidoform itself.
    ex = t.merge(d[["key", "diann_pf"]], left_on=["key", "peptidoform"],
                 right_on=["key", "diann_pf"])
    ex = ex.sort_values(["key", "score"], ascending=[True, False]).drop_duplicates("key")
    ex = ex[["key", "q_value"]].rename(columns={"q_value": "exact_pf_q"})

    lt = lib[lib["label"] == "target"].copy()
    lt["key"] = key(lt["peptidoform"].map(strip), lt["charge"])
    lt = lt[lt["key"].isin(d["key"])].merge(d[["key", "diann_pf"]], on="key")
    lt["exact"] = lt["peptidoform"] == lt["diann_pf"]
    lt = lt.sort_values(["key", "exact", "candidate_id"], ascending=[True, False, True]) \
        .drop_duplicates("key")[["key", "candidate_id", "exact"]] \
        .rename(columns={"candidate_id": "lib_candidate_id", "exact": "exact_pf_in_library"})

    o = d.merge(lt, on="key", how="left").merge(best, on="key", how="left").merge(ex, on="key", how="left")
    o["window_candidate"] = o["candidate_id"].fillna(o["lib_candidate_id"])
    o = o.merge(win, left_on="window_candidate", right_on="candidate_id", how="left",
                suffixes=("", "_w"))
    o["exact_pf_match"] = o["mumdia_pf"] == o["diann_pf"]
    o["rt_err"] = o["apex_rt"] - o["diann_rt"]
    o["im_err"] = o["apex_im"] - o["diann_im"]
    o["rt_pred_err"] = o["rt_pred_cal_w"] - o["diann_rt"]
    rt_in = (o["diann_rt"] >= o["rt_lo"]) & (o["diann_rt"] <= o["rt_hi"])
    im_in = (o["diann_im"] >= o["im_lo"]) & (o["diann_im"] <= o["im_hi"])
    o["diann_rt_in_window"], o["diann_im_in_window"] = rt_in, im_in
    right = o["rt_err"].abs() <= rt_tol
    conds = [o["lib_candidate_id"].isna(),
             o["candidate_id"].isna() & ~rt_in,
             o["candidate_id"].isna() & ~im_in,
             o["candidate_id"].isna(),
             o["q_value"] <= q,
             right, rt_in]
    o["category"] = np.select(conds, CATS[:7], CATS[7])
    if peaks is not None:
        p = peaks.merge(o[["candidate_id", "diann_rt"]].dropna(), on="candidate_id")
        p = p[(p["apex_rt"] - p["diann_rt"]).abs() <= rt_tol]
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
        "q_quartiles": g["q_value"].apply(lambda x: tuple(np.round(x.quantile([.25, .5, .75]), 3))
                                          if x.notna().any() else None),
        "exact_pf_match": g["exact_pf_match"].mean(),
    }).reindex(CATS).dropna(how="all", subset=["n"])
    s["n"] = s["n"].astype(int)
    return s


def main() -> None:
    warnings.simplefilter("ignore", RuntimeWarning)  # medians of the all-null not-extracted groups
    ap = argparse.ArgumentParser()
    ap.add_argument("--diann", required=True)
    ap.add_argument("--run-dir", required=True, help="full run: library, run_windows")
    ap.add_argument("--scored", help="default <run-dir>/psms_scored.parquet")
    ap.add_argument("--extracted", help="default <run-dir>/psms_extracted.parquet")
    ap.add_argument("--lib", help="default <run-dir>/fragment_library_precursors.parquet")
    ap.add_argument("--peaks", help="<psms_extracted>.peaks.parquet (retain_top_peaks sidecar)")
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--rt-tol", type=float, default=5.0)
    ap.add_argument("--out", help="per-precursor parquet with the category label")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = a.run_dir.rstrip("/")
    scored = pd.read_parquet(a.scored or f"{r}/psms_scored.parquet",
                             columns=["candidate_id", "peptidoform", "charge", "label", "score",
                                      "q_value", "precursor_q", "peptide_q_value", "selected_peak_rank"])
    ext = pd.read_parquet(a.extracted or f"{r}/psms_extracted.parquet",
                          columns=["candidate_id", "peak_rank", "apex_rt", "apex_im", "rt_pred_cal",
                                   "n_matched_fragments"])
    lib = pd.read_parquet(a.lib or f"{r}/fragment_library_precursors.parquet",
                          columns=["candidate_id", "peptidoform", "charge", "label"])
    win = pd.read_parquet(f"{r}/run_windows.parquet",
                          columns=["candidate_id", "rt_pred_cal", "rt_lo", "rt_hi", "im_lo", "im_hi"])
    peaks = pd.read_parquet(a.peaks, columns=["candidate_id", "peak_rank", "apex_rt"]) if a.peaks else None
    d = load_diann(a.diann, a.q)
    o = classify(d, scored, ext, lib, win, a.q, a.rt_tol, peaks)
    o["category_exact_pf"] = o["key"].map(
        classify(d, scored, ext, lib, win, a.q, a.rt_tol, exact=True).set_index("key")["category"])
    s = summary(o)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(f"DIA-NN 1% precursors (I/L stripped sequence, charge): {len(o)}")
    print(f"exact-peptidoform join: DIA-NN peptidoform in library {o['exact_pf_in_library'].mean():.3f}, "
          f"best row is that peptidoform {o['exact_pf_match'].mean():.3f}, "
          f"accepted on the exact peptidoform {(o['exact_pf_q'] <= a.q).sum()}")
    print(s.to_string(float_format=lambda x: f"{x:.4g}"))
    print("\naccepted share per DIA-NN Precursor.Quantity quintile (1 = faintest):")
    print((o["category"] == "accepted").groupby(o["quintile"]).mean().round(3).to_string())
    print("\ncategory (best row of the key) against category_exact_pf (DIA-NN's peptidoform only):")
    print(pd.crosstab(o["category"], o["category_exact_pf"]).reindex(index=CATS, columns=CATS)
          .dropna(how="all").dropna(axis=1, how="all").fillna(0).astype(int).to_string())
    if "oracle_rank" in o:
        print("\noracle_rank (DIA-NN apex among the top-K peaks) per category:")
        print(o.groupby("category")["oracle_rank"].value_counts(dropna=False).unstack(fill_value=0).to_string())
    if a.out:
        o.to_parquet(a.out, index=False)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"n": len(o), "summary": json.loads(s.to_json(orient="index"))}, fh, indent=2)


if __name__ == "__main__":
    assert strip("DECOY_PEPM[Oxidation]C[Carbamidomethyl]K") == "PEPMCK"
    assert diann_pf("(UniMod:1)AM(UniMod:35)C(UniMod:4)K") == "[Acetyl]-AM[Oxidation]C[Carbamidomethyl]K"
    main()
