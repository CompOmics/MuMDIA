#!/usr/bin/env python
"""Ion-mobility accuracy of a MuMDIA library and its per-run calibration (G1 check),
measured on the confident precursors of a reference identification table.

  tims_im_check.py --reference reference_report.parquet \
      --lib out_run/fragment_library_precursors.parquet --windows out_run/run_windows.parquet \
      [--seed out_run/seed_psms.parquet] [--ms1 out_run/spectra_ms1.parquet] [--q 0.01]

The reference is an independent identification table of the same run; `tims_ref.py` lists
the column names. This script reads `Modified.Sequence`, `Precursor.Charge`, `Precursor.Mz`,
`RT` (minutes), `IM` and `Q.Value`, and `Predicted.IM` and `Decoy` when present.

For reference precursors at Q.Value <= q, joined to MuMDIA library targets on (ProForma-lite
peptidoform, charge), it reports the median and p95 |error| (V s cm^-2) against the
reference's observed `IM`, per charge and overall, for:
  raw          the library `predicted_im` (uncalibrated IM2Deep, or an imported value)
  calibrated   `run_windows.im_pred_cal` (rt-im-train's per-charge CCS fit)
  ref_pred     the reference's own `Predicted.IM`, when the table has it
  seed_obs     the seed's `observed_im` (fragment-based anchor 1/K0), when --seed is given
  ms1_obs      the MS1 1/K0 at the reference apex, when --ms1 is given (a spectra_ms1 v2
               table, which only the native diaPASEF reader writes): the most intense MS1
               peak within 10 ppm of Precursor.Mz in the MS1 spectrum nearest the reference RT
plus the fraction of those precursors whose reference IM lies inside [im_lo, im_hi], which is
the recall of the 1/K0 gate (`extract.im_gate`) at the current window.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tims_ref import read_reference, reference_pf  # noqa: E402


def err_stats(df: pd.DataFrame, col: str) -> dict:
    """Median / p95 |col - IM| per charge and overall, over rows where col is present."""
    out = {}
    d = df[df[col].notna() & df["IM"].notna()]
    for name, g in [("all", d)] + [(f"z{z}", g) for z, g in d.groupby("charge")]:
        e = (g[col] - g["IM"]).abs()
        out[name] = {"n": int(len(e)),
                     "median": round(float(e.median()), 5) if len(e) else None,
                     "p95": round(float(e.quantile(0.95)), 5) if len(e) else None}
    return out


def ms1_im(ms1_path: str, rt_s: np.ndarray, mz: np.ndarray, ppm: float = 10.0) -> np.ndarray:
    """1/K0 of the most intense MS1 peak within `ppm` of each mz, in the MS1 spectrum
    nearest each rt. NaN when nothing matches."""
    if "im" not in pq.read_schema(ms1_path).names:
        raise SystemExit(f"{ms1_path} has no `im` column (spectra v1: no ion mobility)")
    t = pq.read_table(ms1_path, columns=["rt_seconds", "mz", "intensity", "im"])
    rts = t.column("rt_seconds").to_numpy()
    order = np.argsort(rts, kind="stable")
    rts = rts[order]
    mzl, inl, iml = (t.column(c).combine_chunks() for c in ("mz", "intensity", "im"))
    out = np.full(len(rt_s), np.nan)
    k = np.clip(np.searchsorted(rts, rt_s), 1, len(rts) - 1)
    nearest = np.where(np.abs(rts[k - 1] - rt_s) <= np.abs(rts[k] - rt_s), k - 1, k)
    for spec in np.unique(nearest):
        row = int(order[spec])
        if not mzl[row].is_valid:
            continue
        smz = mzl[row].values.to_numpy()
        sin = inl[row].values.to_numpy()
        sim = iml[row].values.to_numpy()
        srt = np.argsort(smz, kind="stable")
        smz, sin, sim = smz[srt], sin[srt], sim[srt]
        for i in np.nonzero(nearest == spec)[0]:
            tol = mz[i] * ppm * 1e-6
            lo, hi = np.searchsorted(smz, [mz[i] - tol, mz[i] + tol], side="left")
            if hi > lo:
                out[i] = sim[lo + int(np.argmax(sin[lo:hi]))]
    return out


def join(ref: pd.DataFrame, lib: pd.DataFrame) -> pd.DataFrame:
    """Reference precursors joined to library targets on (ProForma-lite peptidoform, charge)."""
    d = ref.assign(peptidoform=ref["Modified.Sequence"].map(reference_pf),
                   charge=ref["Precursor.Charge"].astype(int))
    t = lib[lib["label"] == "target"]
    return d.merge(t, on=["peptidoform", "charge"], how="inner")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", required=True, help="reference precursor table (parquet)")
    ap.add_argument("--lib", required=True, help="fragment_library_precursors.parquet (v2, predicted_im)")
    ap.add_argument("--windows", required=True, help="run_windows.parquet")
    ap.add_argument("--seed", help="seed_psms.parquet (v2, observed_im)")
    ap.add_argument("--ms1", help="spectra_ms1.parquet (v2, per-peak im)")
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--seed-q", type=float, default=0.01,
                    help="seed spectrum_q of the confident targets (search_seed.fdr_seed)")
    ap.add_argument("--json")
    a = ap.parse_args()

    rf = read_reference(a.reference, ["Modified.Sequence", "Precursor.Charge", "Precursor.Mz", "RT",
                                      "IM", "Q.Value"], optional=("Predicted.IM", "Decoy"))
    rf = rf[(rf["Q.Value"] <= a.q) & (rf["Decoy"] == 0)]
    if "predicted_im" not in pq.read_schema(a.lib).names:
        raise SystemExit(f"{a.lib} has no predicted_im (library version 1: no ion mobility)")
    lib = pd.read_parquet(a.lib, columns=["candidate_id", "peptidoform", "charge", "label",
                                          "predicted_im"])
    win = pd.read_parquet(a.windows, columns=["candidate_id", "im_pred_cal", "im_lo", "im_hi"])
    df = join(rf, lib).merge(win, on="candidate_id", how="left")
    df = df.rename(columns={"predicted_im": "raw", "im_pred_cal": "calibrated",
                            "Predicted.IM": "ref_pred"})
    if df["ref_pred"].isna().all():
        df = df.drop(columns=["ref_pred"])
    if a.seed:
        if "observed_im" not in pq.read_schema(a.seed).names:
            raise SystemExit(f"{a.seed} has no observed_im (seed_psms version 1: no ion mobility)")
        # One row per candidate. observed_im is filled for the confident targets, and under
        # rt_im_train.im_window_sizing = seed_decoy_corrected (the default) also for weaker
        # targets and decoys up to the sizing cut; keep the confident targets only.
        seed = pd.read_parquet(a.seed, columns=["candidate_id", "observed_im", "label",
                                                "spectrum_q"])
        seed = seed[(seed["label"] == "target") & (seed["spectrum_q"] <= a.seed_q)]
        seed = seed[["candidate_id", "observed_im"]].dropna()
        df = df.merge(seed.rename(columns={"observed_im": "seed_obs"}), on="candidate_id",
                      how="left")
    if a.ms1:
        df["ms1_obs"] = ms1_im(a.ms1, df["RT"].to_numpy() * 60.0, df["Precursor.Mz"].to_numpy())

    cols = [c for c in ("raw", "calibrated", "ref_pred", "seed_obs", "ms1_obs") if c in df]
    res = {"reference_precursors": int(len(rf)), "joined": int(len(df)),
           "errors": {c: err_stats(df, c) for c in cols}}
    w = df[df["im_lo"].notna() & df["IM"].notna()]
    res["window_recall"] = {
        "n": int(len(w)),
        "inside": round(float(((w["IM"] >= w["im_lo"]) & (w["IM"] <= w["im_hi"])).mean()), 4)
        if len(w) else None,
        "median_half_width": round(float(((w["im_hi"] - w["im_lo"]) / 2).median()), 5)
        if len(w) else None,
    }
    if "ms1_obs" in df:
        res["errors"]["raw_vs_ms1_obs"] = err_stats(
            df.assign(IM=df["ms1_obs"]).dropna(subset=["IM"]), "raw")
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


def _selfcheck():
    rf = pd.DataFrame({"Modified.Sequence": ["AC(UniMod:4)K", "PEPK"],
                       "Precursor.Charge": [2, 3], "IM": [0.80, 0.90]})
    lib = pd.DataFrame({"candidate_id": [7, 8, 9], "peptidoform": ["AC[Carbamidomethyl]K",
                        "PEPK", "PEPK"], "charge": [2, 3, 2],
                        "label": ["target", "target", "target"], "predicted_im": [0.81, 0.95, 0.7]})
    j = join(rf, lib)
    assert j["candidate_id"].tolist() == [7, 8], j
    s = err_stats(j.rename(columns={"predicted_im": "raw"}), "raw")
    assert s["all"]["n"] == 2 and abs(s["z2"]["median"] - 0.01) < 1e-9, s


if __name__ == "__main__":
    _selfcheck()
    main()
