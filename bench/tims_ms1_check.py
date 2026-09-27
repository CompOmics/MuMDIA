#!/usr/bin/env python
"""Why MS1 evidence is missing for accepted precursors (docs/TIMS_ROADMAP_bis.md, D5).

  tims_ms1_check.py --loss loss.parquet --run-dir output_mumdia_p6_best [--category accepted]

For each D1 key of --category with a MuMDIA row: in the MS1 scan nearest its apex (and in the
+-2 neighbouring MS1 scans), the peak nearest the monoisotopic m/z (any 1/K0), and the nearest
peak within --im-tol of DIA-NN's 1/K0. Split by whether extract found an MS1 monoisotopic peak
(psms_extracted `ms1_mono` > 0). Reports the ppm error distribution, the share with a peak
inside --ppm (the extract tolerance, 20 ppm default) and inside 50 ppm, and intensities.
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def nearest(mz, im, inten, target, im_ref, im_tol):
    """(ppm of nearest peak at any IM, ppm/intensity of nearest peak within im_tol of im_ref)."""
    i = np.searchsorted(mz, target)
    lo, hi = max(i - 200, 0), min(i + 200, len(mz))
    if hi <= lo:
        return np.nan, np.nan, np.nan
    ppm = (mz[lo:hi] - target) / target * 1e6
    j = np.argmin(np.abs(ppm))
    near_im = np.abs(im[lo:hi] - im_ref) <= im_tol
    if near_im.any():
        k = np.flatnonzero(near_im)[np.argmin(np.abs(ppm[near_im]))]
        return ppm[j], ppm[k], inten[lo:hi][k]
    return ppm[j], np.nan, np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--category", default="accepted")
    ap.add_argument("--ppm", type=float, default=20.0)
    ap.add_argument("--im-tol", type=float, default=0.03)
    ap.add_argument("--out")
    a = ap.parse_args()
    r = a.run_dir.rstrip("/")
    lo = pd.read_parquet(a.loss)
    lo = lo[(lo.category == a.category) & lo.candidate_id.notna()].copy()
    lo["candidate_id"] = lo.candidate_id.astype("uint32")
    ex = pd.read_parquet(f"{r}/psms_extracted.parquet",
                         columns=["candidate_id", "peak_rank", "precursor_mz", "charge", "ms1_mono"])
    lo = lo.merge(ex, left_on=["candidate_id", "selected_peak_rank"],
                  right_on=["candidate_id", "peak_rank"], suffixes=("", "_ex"))
    lo["ms1_found"] = lo.ms1_mono.fillna(0) > 0
    ms1 = pq.read_table(f"{r}/spectra/spectra_ms1.parquet").to_pandas()
    rts = ms1.rt_seconds.to_numpy()
    # m/z-sorted arrays per MS1 scan (centroids are not guaranteed m/z-sorted across IM)
    scans = []
    for mz, it, im in zip(ms1.mz, ms1.intensity, ms1.im):
        o = np.argsort(mz, kind="stable")
        scans.append((np.asarray(mz)[o], np.asarray(im)[o], np.asarray(it)[o]))
    rows = []
    for x in lo.itertuples():
        s0 = int(np.argmin(np.abs(rts - x.apex_rt)))
        best = (np.nan, np.nan, np.nan)
        for s in range(max(s0 - 2, 0), min(s0 + 3, len(scans))):
            v = nearest(*scans[s], x.precursor_mz, x.diann_im, a.im_tol)
            if s == s0:
                first = v
            if np.isnan(best[1]) or (not np.isnan(v[1]) and abs(v[1]) < abs(best[1])):
                best = v
        rows.append((*first, best[1], best[2]))
    lo[["ppm_any", "ppm_im", "int_im", "ppm_im_pm2", "int_im_pm2"]] = rows
    for found, g in lo.groupby("ms1_found"):
        print(f"ms1_found={found}: n={len(g)}  DIA-NN quintile median {g.quintile.median():.0f}")
        for c in ["ppm_any", "ppm_im", "ppm_im_pm2"]:
            v = g[c].abs()
            print(f"  {c:11s} |ppm| median {v.median():7.1f}  <= {a.ppm:g} ppm {(v <= a.ppm).mean():.3f}"
                  f"  <= 50 ppm {(v <= 50).mean():.3f}  signed median {g[c].median():6.1f}")
        print(f"  intensity of the IM-matched peak, median {g.int_im.median():.0f}")
    if a.out:
        lo.to_parquet(a.out, index=False)


if __name__ == "__main__":
    t = nearest(np.array([100.0, 500.0, 500.01]), np.array([0.8, 0.9, 0.8]), np.array([1.0, 2, 3]),
                500.0, 0.8, 0.03)
    assert t[0] == 0 and abs(t[1] - 20.0) < 1e-6 and t[2] == 3
    main()
