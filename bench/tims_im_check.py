#!/usr/bin/env python
"""Ion-mobility accuracy of a MuMDIA library and its per-run calibration, measured on
DIA-NN's confident precursors (docs/TIMS_ROADMAP.md, P2).

  tims_im_check.py --diann output_diann/report.parquet \
      --lib out/fragment_library_precursors.parquet --windows out/run_windows.parquet \
      [--seed out/seed_psms.parquet] [--ms1 conv/spectra_ms1.parquet] [--q 0.01]

For DIA-NN precursors at Q.Value <= q, joined to MuMDIA library targets on (ProForma,
charge), it reports median and p95 |error| (V s cm^-2) against DIA-NN's observed `IM`,
per charge and overall, for:
  raw          the library `predicted_im` (uncalibrated IM2Deep, or an imported value)
  calibrated   `run_windows.im_pred_cal` (rt-im-train's per-charge CCS fit)
  diann_pred   DIA-NN's own `Predicted.IM`, the reference point
  seed_obs     the seed's `observed_im` (fragment-based anchor IM), when --seed is given
  ms1_obs      the native reader's MS1 1/K0 at DIA-NN's apex, when --ms1 is given: the
               most intense MS1 peak within 10 ppm of Precursor.Mz in the MS1 spectrum
               nearest DIA-NN's RT
plus the fraction of those precursors whose DIA-NN IM lies inside [im_lo, im_hi], which is
the recall a P4 IM gate would have.
"""
import argparse
import json

import numpy as np
import pandas as pd

_UNIMOD = {"(UniMod:4)": "[Carbamidomethyl]", "(UniMod:35)": "[Oxidation]"}


def proforma(modseq: str) -> str:
    for k, v in _UNIMOD.items():
        modseq = modseq.replace(k, v)
    return modseq


def err_stats(df: pd.DataFrame, col: str) -> dict:
    """Median / p95 |col - IM| per charge and overall, over rows where col is present."""
    out = {}
    d = df[df[col].notna()]
    for name, g in [("all", d)] + [(f"z{z}", g) for z, g in d.groupby("charge")]:
        e = (g[col] - g["IM"]).abs()
        out[name] = {"n": int(len(e)),
                     "median": round(float(e.median()), 5) if len(e) else None,
                     "p95": round(float(e.quantile(0.95)), 5) if len(e) else None}
    return out


def ms1_im(ms1_path: str, rt_s: np.ndarray, mz: np.ndarray, ppm: float = 10.0) -> np.ndarray:
    """1/K0 of the most intense MS1 peak within `ppm` of each mz, in the MS1 spectrum
    nearest each rt. NaN when nothing matches."""
    import pyarrow.parquet as pq
    t = pq.read_table(ms1_path, columns=["rt_seconds", "mz", "intensity", "im"])
    rts = t.column("rt_seconds").to_numpy()
    order = np.argsort(rts)
    rts = rts[order]
    mzl, inl, iml = (t.column(c).combine_chunks() for c in ("mz", "intensity", "im"))
    out = np.full(len(rt_s), np.nan)
    k = np.clip(np.searchsorted(rts, rt_s), 1, len(rts) - 1)
    nearest = np.where(np.abs(rts[k - 1] - rt_s) <= np.abs(rts[k] - rt_s), k - 1, k)
    for spec in np.unique(nearest):
        row = int(order[spec])
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


def join(diann: pd.DataFrame, lib: pd.DataFrame) -> pd.DataFrame:
    """DIA-NN precursors joined to library targets on (ProForma, charge)."""
    d = diann.assign(peptidoform=diann["Modified.Sequence"].map(proforma),
                     charge=diann["Precursor.Charge"].astype(int))
    t = lib[lib["label"] == "target"]
    return d.merge(t, on=["peptidoform", "charge"], how="inner")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diann", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--seed")
    ap.add_argument("--ms1")
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--json")
    a = ap.parse_args()

    dn = pd.read_parquet(a.diann, columns=["Modified.Sequence", "Precursor.Charge",
                                           "Precursor.Mz", "RT", "IM", "Predicted.IM",
                                           "Q.Value", "Decoy"])
    dn = dn[(dn["Q.Value"] <= a.q) & (dn["Decoy"] == 0)]
    lib = pd.read_parquet(a.lib, columns=["candidate_id", "peptidoform", "charge", "label",
                                          "predicted_im"])
    win = pd.read_parquet(a.windows, columns=["candidate_id", "im_pred_cal", "im_lo", "im_hi"])
    df = join(dn, lib).merge(win, on="candidate_id", how="left")
    df = df.rename(columns={"predicted_im": "raw", "im_pred_cal": "calibrated",
                            "Predicted.IM": "diann_pred"})
    if a.seed:
        seed = pd.read_parquet(a.seed, columns=["candidate_id", "observed_im"])
        df = df.merge(seed.rename(columns={"observed_im": "seed_obs"}), on="candidate_id",
                      how="left")
    if a.ms1:
        df["ms1_obs"] = ms1_im(a.ms1, df["RT"].to_numpy() * 60.0, df["Precursor.Mz"].to_numpy())

    cols = [c for c in ("raw", "calibrated", "diann_pred", "seed_obs", "ms1_obs") if c in df]
    res = {"diann_precursors": int(len(dn)), "joined": int(len(df)),
           "errors": {c: err_stats(df, c) for c in cols}}
    w = df[df["im_lo"].notna()]
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
    dn = pd.DataFrame({"Modified.Sequence": ["AC(UniMod:4)K", "PEPK"],
                       "Precursor.Charge": [2, 3], "IM": [0.80, 0.90]})
    lib = pd.DataFrame({"candidate_id": [7, 8, 9], "peptidoform": ["AC[Carbamidomethyl]K",
                        "PEPK", "PEPK"], "charge": [2, 3, 2],
                        "label": ["target", "target", "target"], "predicted_im": [0.81, 0.95, 0.7]})
    j = join(dn, lib)
    assert j["candidate_id"].tolist() == [7, 8], j
    s = err_stats(j.rename(columns={"predicted_im": "raw"}), "raw")
    assert s["all"]["n"] == 2 and abs(s["z2"]["median"] - 0.01) < 1e-9, s


if __name__ == "__main__":
    _selfcheck()
    main()
