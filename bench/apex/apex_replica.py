"""Replicate extract's apex rule from chromatograms.parquet and try alternative tie-breaks.

Engine rule (extract.rs, apex_evidence_rank): per scan, n = distinct predicted fragments with
signal; smoothed = truncated centred rolling SUM of n over `w` scans; qualifying scans have
smoothed >= max(smoothed) - tol; among them maximise (n + sig/(sig+1)) * prior, where sig =
observed intensity of the top-3 predicted fragments and prior = exp(-0.5((rt-rt_cal)/sigma)^2);
the first scan wins a tie (strict >). Variants change only the score.
Usage: apex_replica.py RUN_DIR W SIGMA TOL
"""
import collections
import sys

import numpy as np
import pyarrow.parquet as pq

run, W, SIGMA, TOL = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])

sc = pq.read_table(f"{run}/psms_scored.parquet",
                   columns=["candidate_id", "label", "precursor_q", "apex_rt"]).to_pandas()
acc = sc[(sc.label == "target") & (sc.precursor_q <= 0.01)]
want = dict(zip(acc.candidate_id.astype(np.int64), acc.apex_rt))
rw = pq.read_table(f"{run}/run_windows.parquet", columns=["candidate_id", "rt_pred_cal"])
rcid = rw.column("candidate_id").to_numpy().astype(np.int64)
rcal = rw.column("rt_pred_cal").to_numpy()
rt_cal = dict(zip(rcid[np.isin(rcid, list(want))], rcal[np.isin(rcid, list(want))]))

pf = pq.ParquetFile(f"{run}/chromatograms.parquet")
cols = ["candidate_id", "frag_name", "predicted_intensity", "rt_axis", "intensity_trimmed",
        "trace_offset", "trace_len"]
per = collections.defaultdict(list)  # cid -> [(pred_int, full_trace)]
axes = {}
wanted = np.array(sorted(want), dtype=np.int64)
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg, columns=cols)
    cid = t.column("candidate_id").to_numpy().astype(np.int64)
    keep = np.isin(cid, wanted)
    if not keep.any():
        continue
    idx = np.nonzero(keep)[0]
    fname = t.column("frag_name").to_pylist()
    pint = t.column("predicted_intensity").to_numpy()
    ax_col, it_col = t.column("rt_axis"), t.column("intensity_trimmed")
    off, tl = t.column("trace_offset").to_numpy(), t.column("trace_len").to_numpy()
    last_axis = {}
    for i in range(idx.min(), idx.max() + 1):
        c = cid[i]
        if c not in want:
            continue
        if ax_col[i].is_valid:
            a = ax_col[i].values.to_numpy(zero_copy_only=False)
            if len(a):
                last_axis[c] = a
        if fname[i].startswith("ms1_") or c not in last_axis:
            continue
        axis = last_axis[c]
        full = np.zeros(len(axis), dtype=np.float64)
        if tl[i]:
            vals = it_col[i].values.to_numpy(zero_copy_only=False)
            full[int(off[i]):int(off[i]) + len(vals)] = vals
        axes[c] = axis
        per[c].append((float(pint[i]), full))


def pick(axis, M, sig, rtc, variant):
    n = (M > 0).sum(axis=0)
    if W <= 1:
        sm = n.astype(float)
    else:
        r = W // 2
        sm = np.array([n[max(0, i - r):min(len(n), i + r + 1)].sum() for i in range(len(n))], float)
    thresh = max(sm.max() - TOL, 0.0)
    q = (n > 0) & (sm >= thresh)
    prior = np.exp(-0.5 * ((axis - rtc) / SIGMA) ** 2) if (SIGMA > 0 and rtc > 0) else np.ones(len(axis))
    if variant in ("V3", "V4"):
        prior = np.ones(len(axis))
    if variant == "V5":
        # Region by sustained fragment count, apex by intensity: the qualifying scans
        # widened by the rolling half-width, then the brightest signature scan in it.
        r = W // 2
        region = np.zeros(len(n), bool)
        for i in np.nonzero(q)[0]:
            region[max(0, i - r):i + r + 1] = True
        region &= n > 0
        cand = np.nonzero(region)[0]
        return int(cand[np.argmax(sig[cand] * prior[cand])]) if len(cand) else -1
    if variant in ("V1", "V3"):
        tie = sig / (sig + 1.0)
    else:
        m = sig[q].max() if q.any() else 0.0
        tie = sig / m * 0.999 if m > 0 else np.zeros(len(sig))
    score = (n + tie).astype(np.float32) * prior.astype(np.float32)
    best, bi = -np.inf, -1
    for i in np.nonzero(q)[0]:
        if score[i] > best:
            best, bi = score[i], i
    return bi


stats = {v: collections.Counter() for v in ("V1", "V2", "V3", "V4", "V5")}
agree = collections.Counter()
N = 0
for c, apex in want.items():
    if c not in axes or not per[c]:
        continue
    axis = axes[c].astype(np.float64)
    rows = [p for p in per[c] if len(p[1]) == len(axis)]
    if not rows:
        continue
    M = np.array([r[1] for r in rows])
    order = np.argsort([-r[0] for r in rows], kind="stable")[:3]
    sig = M[order].sum(axis=0)
    total = M.sum(axis=0)
    eng = int(np.argmin(np.abs(axis - apex)))
    N += 1
    for v in stats:
        bi = pick(axis, M, sig, rt_cal.get(c, 0.0), v)
        if v == "V1":
            agree[bi == eng] += 1
        w0, w1 = max(0, bi - 3), min(len(total), bi + 4)
        stats[v][w0 + int(np.argmax(total[w0:w1])) - bi] += 1

print(f"n={N}; replica V1 reproduces the engine apex for {agree[True]} ({100 * agree[True] / N:.1f}%)")
for v, cnt in stats.items():
    line = "  ".join(f"{k:+d}:{100 * cnt[k] / N:5.1f}%" for k in range(-3, 4))
    print(f"{v}: offset of the summed-intensity maximum (+/-3 scans) from the chosen apex  {line}")
