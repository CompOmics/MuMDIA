"""Retention of the RT-free tag prefilter against an unfiltered search.

usage: ps_tagpre_eval.py NAME SURVIVORS INPUT_TABLE TIME_FILE REF_SCORED [REF_SCORED ...]

Precursors are matched by (peptidoform, charge), so the input may be a library (candidate_id)
or a peptidoform table (id). REF_SCORED are psms_scored tables of the unfiltered run (several
NN seeds are united); a precursor is a reference when a target is accepted at precursor_q 0.01.
"""
import json
import sys

import pyarrow.parquet as pq

name, surv, inp, timef, *refs = sys.argv[1:]
cols = pq.read_schema(inp).names
idc = "candidate_id" if "candidate_id" in cols else "id"
t = pq.read_table(inp, columns=[idc, "peptidoform", "charge", "label"]).to_pandas()
t["key"] = list(zip(t.peptidoform.str.removeprefix("DECOY_"), t.charge.astype(int)))
kept_ids = set(pq.read_table(surv, columns=["candidate_id"]).column(0).to_pylist())
kept = t[t[idc].isin(kept_ids)]
ref = set()
for r in refs:
    d = pq.read_table(r, columns=["label", "peptidoform", "charge", "precursor_q"]).to_pandas()
    d = d[(~d.label.astype(str).str.lower().str.startswith("decoy")) & (d.precursor_q <= 0.01)]
    ref |= set(zip(d.peptidoform.str.removeprefix("DECOY_"), d.charge.astype(int)))
kept_keys = set(kept.key)
in_lib = ref & set(t.key)
st = json.load(open(surv + ".report.json"))["stats"]
wall = rss = None
for line in open(timef):
    if "Elapsed (wall" in line:
        wall = 0.0
        for x in line.rsplit(" ", 1)[1].strip().split(":"):
            wall = wall * 60 + float(x)
    if "Maximum resident" in line:
        rss = int(line.rsplit(" ", 1)[1]) / 1048576
dec = t.label.astype(str).str.lower().str.startswith("decoy")
kdec = kept.label.astype(str).str.lower().str.startswith("decoy")
out = {
    "dataset": name, "candidates": len(t), "kept": len(kept),
    "removed_pct": round(100 * (1 - len(kept) / len(t)), 2),
    "target_kept_pct": round(100 * (~kdec).sum() / max(1, (~dec).sum()), 2),
    "decoy_kept_pct": round(100 * kdec.sum() / max(1, dec.sum()), 2),
    "reference_precursors": len(in_lib),
    "reference_kept_pct": round(100 * len(in_lib & kept_keys) / max(1, len(in_lib)), 3),
    "reference_lost": len(in_lib - kept_keys),
    "tag_paths": st.get("tag_paths"), "not_expressible": st.get("tag_unsupported_candidates"),
    "wall_s": wall, "peak_gb": round(rss, 2) if rss else None,
}
print(json.dumps(out))
