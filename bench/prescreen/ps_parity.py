"""Parity of the Rust prescreen score with the tagbench prototype's rule.

Re-implements `fragment_rarity.histogram`/`probability` and the `lowmz_half` column of
`advanced_candidates.scores` in NumPy, line by line, with the MuMDIA eligibility (isolation
window `lower <= pmz < upper`, calibrated `rt_lo <= rt <= rt_hi`), and compares it with the
`base_score` column the stage wrote, for a seeded sample of candidates.

usage: ps_parity.py SPECTRA_MS2 LIB_PRECURSORS RUN_WINDOWS SCORES_PARQUET [N]
"""
import sys

import numpy as np
import pyarrow.parquet as pq

PROTON = 1.007276466812
WATER = 18.010564684
AA = {"G": 57.021463735, "A": 71.037113805, "S": 87.032028435, "P": 97.052763875,
      "V": 99.068413945, "T": 101.047678505, "C": 103.009184505, "L": 113.084064015,
      "I": 113.084064015, "N": 114.042927470, "D": 115.026943065, "Q": 128.058577540,
      "K": 128.094963050, "E": 129.042593135, "M": 131.040484645, "H": 137.058911875,
      "F": 147.068413945, "R": 156.101111050, "Y": 163.063328575, "W": 186.079312980}
MOD = {"Carbamidomethyl": 57.021463735, "Oxidation": 15.994914620}

ms2, libp, rwp, scp = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 3000


def masses(pf):
    pf = pf.removeprefix("DECOY_")
    out, i = [], 0
    while i < len(pf):
        r = pf[i]; i += 1; m = AA[r]
        if i < len(pf) and pf[i] == "[":
            j = pf.index("]", i); m += MOD[pf[i + 1:j]]; i = j + 1
        out.append(m)
    return np.array(out)


t = pq.read_table(ms2, columns=["rt_seconds", "window_lower", "window_upper", "mz", "intensity"])
rt = t.column("rt_seconds").to_numpy()
lo = t.column("window_lower").to_numpy(); hi = t.column("window_upper").to_numpy()
mz = [np.asarray(x, dtype=np.float32).astype(np.float64) for x in t.column("mz").to_pylist()]
iv = [np.asarray(x, dtype=np.float32) for x in t.column("intensity").to_pylist()]
for s in range(len(mz)):
    o = np.argsort(mz[s], kind="stable"); mz[s] = mz[s][o]; iv[s] = iv[s][o]
_, wid = np.unique(np.c_[lo, hi], axis=0, return_inverse=True)
wid = wid.ravel()
nbin = int(np.ceil(max(m.max() for m in mz if len(m)) * 100)) + 2
hist = {}
wn = np.bincount(wid)
for w in np.unique(wid):
    h = np.zeros(nbin, np.int64)
    for s in np.flatnonzero(wid == w):
        b = np.unique(np.rint(mz[s] * 100).astype(np.int64))
        h[b] += 1
    hist[w] = h


def probability(w, mass):
    b = int(np.rint(mass * 100)); c = 0
    for k in range(b - 1, b + 2):
        if 0 <= k < nbin:
            c += hist[w][k]
    return min(.99999, (c + .5) / (wn[w] + 1.))


def nearest(m, x, tol):
    j = np.searchsorted(m, x - tol, side="left"); best, err = -1, tol + 1.
    while j < len(m) and m[j] <= x + tol:
        d = abs(m[j] - x)
        if d < err:
            best, err = j, d
        j += 1
    return best


lib = pq.read_table(libp, columns=["candidate_id", "peptidoform", "charge", "precursor_mz"]).to_pandas()
rw = pq.read_table(rwp, columns=["candidate_id", "rt_lo", "rt_hi"]).to_pandas().set_index("candidate_id")
sc = pq.read_table(scp, columns=["candidate_id", "base_score"]).to_pandas().set_index("candidate_id")
groups = {}
for w in np.unique(wid):
    idx = np.flatnonzero(wid == w)
    idx = idx[np.argsort(rt[idx], kind="stable")]
    groups[w] = (lo[idx[0]], hi[idx[0]], rt[idx], idx)
rng = np.random.default_rng(20261004)
pick = lib.iloc[np.sort(rng.choice(len(lib), N, replace=False))]
errs, nz = [], 0
for _, c in pick.iterrows():
    seq = masses(c.peptidoform); L = len(seq); nser = 2 * min(2, int(c.charge))
    fr = np.empty((2, nser, L - 1))
    for o in range(2):
        bm = ym = 0.
        for k in range(L - 1):
            bm += seq[k] if o == 0 else seq[L - 1 - k]
            ym += seq[L - 1 - k] if o == 0 else seq[k]
            for z in range(nser // 2):
                fr[o, 2 * z, k] = bm / (z + 1) + PROTON
                fr[o, 2 * z + 1, k] = (ym + WATER) / (z + 1) + PROTON
    rlo, rhi = rw.loc[c.candidate_id, ["rt_lo", "rt_hi"]]
    if not (np.isfinite(rlo) and np.isfinite(rhi)):
        rlo, rhi = -np.inf, np.inf
    best = 0.
    elig = []
    for w, (gl, gh, grt, gidx) in groups.items():
        if gl <= c.precursor_mz < gh:
            a = np.searchsorted(grt, rlo, side="left"); b = np.searchsorted(grt, rhi, side="right")
            elig.extend(gidx[a:b])
    for s in elig:
        m = mz[s]; n = len(m)
        for o in range(2):
            seen = np.zeros(n, bool)
            for series in range(nser):
                for k in range(L - 1):
                    p = nearest(m, fr[o, series, k], .005)
                    if p >= 0 and iv[s][p] > 0:
                        seen[p] = True
            total = 0.
            for p in np.flatnonzero(seen):
                value = -np.log(probability(wid[s], m[p])) / np.sqrt(L)
                total += value * (.5 if m[p] < 300 else 1)
            best = max(best, total)
    got = sc.loc[c.candidate_id, "base_score"]
    errs.append(abs(got - best)); nz += best > 0
errs = np.array(errs)
print(f"candidates {len(errs)} (non-zero {nz}); max |rust - prototype| {errs.max():.3e}; "
      f"mean {errs.mean():.3e}; exact-equal {np.mean(errs == 0):.4f}")
