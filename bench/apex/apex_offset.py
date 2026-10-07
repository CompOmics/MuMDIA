"""Where the chosen apex sits relative to the summed fragment trace, for accepted IDs.

For every accepted target precursor (precursor_q <= 0.01, its winning row), rebuild the
fragment traces from chromatograms.parquet (v1 or v2 layout, docs/15 "Layout v2"), sum the
MS2 fragment traces (MS1 pseudo-rows excluded) and report, in scans of that precursor's
window:
  local  = argmax of the summed trace within +/-3 scans of the chosen apex, minus the apex
  global = argmax of the summed trace over the whole extracted trace, minus the apex
  bounds = number of scans inside [elution_lo, elution_hi], and how many scans at more than
           half the apex height lie outside them on each side.
Usage: apex_offset.py RUN_DIR [OUT_TSV]
"""
import collections
import sys

import numpy as np
import pyarrow.parquet as pq

run = sys.argv[1]
out_tsv = sys.argv[2] if len(sys.argv) > 2 else None

sc = pq.read_table(f"{run}/psms_scored.parquet",
                   columns=["candidate_id", "label", "precursor_q", "peptidoform", "charge",
                            "apex_rt", "elution_lo", "elution_hi"]).to_pandas()
acc = sc[(sc.label == "target") & (sc.precursor_q <= 0.01)]
want = dict(zip(acc.candidate_id.astype(np.int64), zip(acc.apex_rt, acc.elution_lo, acc.elution_hi,
                                                         acc.peptidoform, acc.charge)))
print(f"accepted target precursors: {len(want)}", flush=True)

pf = pq.ParquetFile(f"{run}/chromatograms.parquet")
names = pf.schema_arrow.names
v2 = "rt_axis" in names
cols = ["candidate_id", "frag_name"] + (["rt_axis", "intensity_trimmed", "trace_offset", "trace_len"]
                                        if v2 else ["rt", "intensity"])
traces = collections.defaultdict(list)   # cid -> list of full fragment traces
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
    if v2:
        ax_col = t.column("rt_axis")
        it_col = t.column("intensity_trimmed")
        off = t.column("trace_offset").to_numpy()
        tl = t.column("trace_len").to_numpy()
        # The axis rule restarts per row group: a row with an empty rt_axis uses the
        # last axis its candidate wrote earlier in the group.
        last_axis = {}
        lo = idx.min()
        for i in range(lo, idx.max() + 1):
            if cid[i] not in want:
                continue
            a = ax_col[i].values.to_numpy(zero_copy_only=False) if ax_col[i].is_valid else None
            if a is not None and len(a):
                last_axis[cid[i]] = a
            if tl[i] == 0 or fname[i].startswith("ms1_"):
                continue
            axis = last_axis.get(cid[i])
            if axis is None:
                continue
            full = np.zeros(int(tl[i]), dtype=np.float64)
            vals = it_col[i].values.to_numpy(zero_copy_only=False)
            full[int(off[i]):int(off[i]) + len(vals)] = vals
            axes[cid[i]] = axis
            traces[cid[i]].append(full)
    else:
        rtc, inc = t.column("rt"), t.column("intensity")
        for i in idx:
            if fname[i].startswith("ms1_") or not rtc[i].is_valid:
                continue
            a = rtc[i].values.to_numpy(zero_copy_only=False)
            if not len(a):
                continue
            axes[cid[i]] = a
            traces[cid[i]].append(inc[i].values.to_numpy(zero_copy_only=False).astype(np.float64))

local_c, global_c = collections.Counter(), collections.Counter()
rows = []
left_miss = right_miss = 0
n = 0
for c, (apex, lo, hi, pform, z) in want.items():
    if c not in axes or not traces[c]:
        continue
    axis = axes[c].astype(np.float64)
    tr = [t for t in traces[c] if len(t) == len(axis)]
    if not tr:
        continue
    s = np.sum(tr, axis=0)
    a = int(np.argmin(np.abs(axis - apex)))
    w0, w1 = max(0, a - 3), min(len(s), a + 4)
    loc = w0 + int(np.argmax(s[w0:w1])) - a
    glo = int(np.argmax(s)) - a
    half = s[a] / 2.0
    inside = (axis >= lo) & (axis <= hi)
    # contiguous above-half run around the apex
    l = a
    while l - 1 >= 0 and s[l - 1] >= half:
        l -= 1
    r = a
    while r + 1 < len(s) and s[r + 1] >= half:
        r += 1
    lm = int(np.sum((axis[l:a] < lo)))
    rm = int(np.sum((axis[a + 1:r + 1] > hi)))
    left_miss += lm > 0
    right_miss += rm > 0
    local_c[loc] += 1
    global_c[max(-10, min(10, glo))] += 1
    n += 1
    rows.append((c, pform, z, apex, loc, glo, int(inside.sum()), lm, rm))

def show(name, cnt):
    print(f"{name} offset (scans, + = true maximum later than the chosen apex), n={n}:")
    for k in sorted(cnt):
        print(f"  {k:+3d}: {cnt[k]:7d}  {100 * cnt[k] / n:5.1f}%")

show("local (+/-3 scans)", local_c)
show("global (clipped at +/-10)", global_c)
print(f"IDs whose above-half-maximum region extends left of elution_lo: {left_miss} "
      f"({100 * left_miss / n:.1f}%), right of elution_hi: {right_miss} ({100 * right_miss / n:.1f}%)")
if out_tsv:
    with open(out_tsv, "w") as fh:
        fh.write("candidate_id\tpeptidoform\tcharge\tapex_rt\tlocal_offset\tglobal_offset\t"
                 "scans_in_bounds\tabove_half_left_of_bounds\tabove_half_right_of_bounds\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
