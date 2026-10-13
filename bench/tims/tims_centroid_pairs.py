#!/usr/bin/env python
"""Centroid-pair histograms of native timsTOF spectra (G1 check of the diaPASEF reader).

Counts pairs of centroids at the same 1/K0 (within --im-tol) by their m/z distance in ppm,
in --n evenly spaced spectra per MS level. One ion cut into pieces shows up as pairs closer
than the instrument resolution (about 20-25 ppm FWHM on the data this was written for, so
the 4-10 ppm bin); two neighbouring ions that are kept apart show up at 10-25 ppm.

    python bench/tims/tims_centroid_pairs.py NAME:SPECTRA_DIR [NAME:SPECTRA_DIR ...]

SPECTRA_DIR holds spectra_ms1.parquet and spectra_ms2.parquet of version 2, which carry the
per-peak `im` list (written only by the native diaPASEF reader, docs/35_diapasef.md).

A second histogram counts pairs at the same m/z (within --mz-tol ppm) by their 1/K0
distance. One ion cut along mobility gives pairs much closer than a mobility peak width
(about 0.02 FWHM on the same data).
"""

import argparse

import numpy as np
import pyarrow.parquet as pq

BINS = [0.0, 4.0, 10.0, 25.0, 50.0]
IM_BINS = [0.0, 0.005, 0.01, 0.02, 0.05]


def pairs(mz, im, im_tol):
    """Pair counts per BINS interval for one m/z-sorted spectrum."""
    out = np.zeros(len(BINS) - 1, dtype=np.int64)
    for k in range(1, len(mz)):
        ppm = (mz[k:] - mz[:-k]) / mz[:-k] * 1e6
        near = ppm < BINS[-1]
        if not near.any():
            break
        ok = near & (np.abs(im[k:] - im[:-k]) <= im_tol)
        out += np.histogram(ppm[ok], bins=BINS)[0]
    return out


def im_pairs(mz, im, mz_tol):
    """Pair counts per IM_BINS interval (1/K0 distance) among peaks within mz_tol ppm."""
    out = np.zeros(len(IM_BINS) - 1, dtype=np.int64)
    for k in range(1, len(mz)):
        ppm = (mz[k:] - mz[:-k]) / mz[:-k] * 1e6
        near = ppm < mz_tol
        if not near.any():
            break
        out += np.histogram(np.abs(im[k:] - im[:-k])[near], bins=IM_BINS)[0]
    return out


def level(path, n, im_tol, mz_tol):
    """Total and median centroid count, and both pair histograms, over n spectra of one level."""
    f = pq.ParquetFile(path)
    if "im" not in f.schema_arrow.names:
        raise SystemExit(f"{path} has no `im` column (spectra v1: no ion mobility)")
    t = f.read(columns=["mz", "im"])
    rows = np.linspace(0, t.num_rows - 1, min(n, t.num_rows)).round().astype(int)
    mzc, imc = t.column("mz").combine_chunks(), t.column("im").combine_chunks()
    lens = np.diff(mzc.offsets.to_numpy())
    h = np.zeros(len(BINS) - 1, dtype=np.int64)
    hi = np.zeros(len(IM_BINS) - 1, dtype=np.int64)
    for r in rows:
        if not mzc[int(r)].is_valid:
            continue
        mz = np.asarray(mzc[int(r)].values, dtype=np.float64)
        im = np.asarray(imc[int(r)].values, dtype=np.float64)
        h += pairs(mz, im, im_tol)
        hi += im_pairs(mz, im, mz_tol)
    return int(lens.sum()), int(np.median(lens)), h, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="NAME:SPECTRA_DIR")
    ap.add_argument("--n", type=int, default=200, help="spectra per MS level (default 200)")
    ap.add_argument("--im-tol", type=float, default=0.005, help="same-1/K0 tolerance (default 0.005)")
    ap.add_argument("--mz-tol", type=float, default=3.0, help="same-m/z tolerance in ppm (default 3)")
    a = ap.parse_args()
    head = " ".join(f"{f'{lo:g}-{hi:g} ppm':>12}" for lo, hi in zip(BINS, BINS[1:]))
    ihead = " ".join(f"{f'dIM {lo:g}-{hi:g}':>13}" for lo, hi in zip(IM_BINS, IM_BINS[1:]))
    print(f"{'arm':<14}{'level':<6}{'centroids':>12}{'median':>8} {head} {ihead}")
    for arm in a.arms:
        name, d = arm.split(":", 1)
        for lv in ("ms1", "ms2"):
            tot, med, h, hi = level(f"{d}/spectra_{lv}.parquet", a.n, a.im_tol, a.mz_tol)
            print(f"{name:<14}{lv:<6}{tot:>12,}{med:>8,} " + " ".join(f"{x:>12,}" for x in h)
                  + " " + " ".join(f"{x:>13,}" for x in hi))


if __name__ == "__main__":
    # Self-check: a pair 5 ppm apart at one 1/K0, a third peak 20 ppm away at another 1/K0.
    m = np.array([500.0, 500.0025, 500.01])
    assert pairs(m, np.array([1.0, 1.0, 1.1]), 0.005).tolist() == [0, 1, 0, 0]
    # Same m/z at 1/K0 1.0, 1.003, 1.033: pairs 0.003, 0.030, 0.033 apart.
    assert im_pairs(np.array([500.0, 500.0, 500.0]), np.array([1.0, 1.003, 1.033]), 3.0).tolist() == [1, 0, 0, 2]
    main()
