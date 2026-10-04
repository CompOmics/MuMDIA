"""Summarise a ps_rtfree.sh campaign: identifications, FDP, stage times, peak memory, how much
went to prediction, and how many of OFF's accepted precursors each target would keep.

Precursors are matched across arms by (peptidoform, charge): the prescreened library is
renumbered, so candidate ids differ between OFF and PRE.

usage: ps_rtfree_agg.py ROOT INPUT_TABLE [--entrapment]
  INPUT_TABLE: what the screen read (FASTA mode: ROOT/PRE/peptidoforms.parquet; library mode:
  the imported lib precursors)
"""
import datetime as dt
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT, INPUT = sys.argv[1], sys.argv[2]
ENTRAP = "--entrapment" in sys.argv
ENT = {"marker": "ENTRAP_", "exclude": "REAL_", "contaminants": ["KRT", "K1C", "K2C", "ALBU", "TRYP"],
       "ratio": 0.560632}


def is_entrap(p):
    p = str(p)
    return ENT["marker"] in p and ENT["exclude"] not in p and not any(c in p for c in ENT["contaminants"])


def scored(path):
    st = json.load(open(path + ".report.json")).get("stats", {})
    t = pq.read_table(path, columns=["label", "peptidoform", "charge", "protein", "precursor_q",
                                     "peptide_q_value", "base_peptide_id"]).to_pandas()
    tgt = t[~t.label.astype(str).str.lower().str.startswith("decoy")]
    acc = tgt[tgt.precursor_q <= 0.01]
    out = {"peptides": st.get("target_peptides_at_1pct"), "precursors": st.get("target_precursors_at_1pct"),
           "accepted": set(zip(acc.peptidoform.str.removeprefix("DECOY_"), acc.charge.astype(int)))}
    if ENTRAP:
        pep = tgt.sort_values("peptide_q_value").drop_duplicates("base_peptide_id")
        pep = pep[pep.peptide_q_value <= 0.01]
        e = pep.protein.map(is_entrap)
        out["fdp"] = 100 * (ENT["ratio"] * int(e.sum()) + 1) / max(int((~e).sum()), 1)
    return out


def timing(name):
    out = {}
    for line in open(f"{ROOT}/time_{name}.txt", errors="replace"):
        if "Elapsed (wall" in line:
            s = 0.0
            for x in line.rsplit(" ", 1)[1].strip().split(":"):
                s = s * 60 + float(x)
            out["wall_s"] = s
        if "Maximum resident" in line:
            out["peak_gb"] = int(line.rsplit(" ", 1)[1]) / 1048576
    return out


def stages(log):
    """Seconds per stage from the run log's `run: stage start` lines."""
    pts = []
    for line in open(log, errors="replace"):
        m = re.match(r"(\S+Z)\s+INFO.*run: stage start.*stage=(\S+)", line)
        if m:
            pts.append((dt.datetime.fromisoformat(m.group(1).replace("Z", "+00:00")), m.group(2)))
        last = line
    m = re.match(r"(\S+Z)", last)
    end = dt.datetime.fromisoformat(m.group(1).replace("Z", "+00:00")) if m else pts[-1][0]
    out = {}
    for (t, s), nxt in zip(pts, pts[1:] + [(end, None)]):
        out[s] = out.get(s, 0) + (nxt[0] - t).total_seconds()
    return out


def mean_sd(v):
    v = [x for x in v if x is not None]
    m = sum(v) / len(v)
    return m, (math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0)


res = {}
for arm in ["OFF", "PRE"]:
    A = f"{ROOT}/{arm}"
    seeds = [scored(f"{A}/psms_scored.parquet")] + [scored(p) for p in sorted(glob.glob(f"{A}/s*/psms_scored.parquet"))]
    log = open(f"{ROOT}/{arm}.log", errors="replace").read()
    pred = re.search(r"kept=(\d+) of=(\d+).*passed to prediction", log)
    sub = re.search(r"library=(\d+) kept=(\d+).*replaces the imported", log)
    res[arm] = {"seeds": seeds, "time": timing(arm), "stages": stages(f"{ROOT}/{arm}.log"),
                "predicted": (int(pred.group(1)), int(pred.group(2))) if pred else None,
                "sublibrary": (int(sub.group(2)), int(sub.group(1))) if sub else None}

ref = set().union(*[s["accepted"] for s in res["OFF"]["seeds"]])
# Retention per target from the PRE screen's scores (same rule as the stage).
inp = pq.read_table(INPUT, columns=[c for c in ["candidate_id", "id", "peptidoform", "charge"]
                                    if c in pq.read_schema(INPUT).names]).to_pandas()
idc = "candidate_id" if "candidate_id" in inp else "id"
key = dict(zip(inp[idc].astype(int), zip(inp.peptidoform.str.removeprefix("DECOY_"), inp.charge.astype(int))))
sc = pq.read_table(f"{ROOT}/PRE/prescreen_survivors.scores.parquet",
                   columns=["candidate_id", "score", "in_scope", "calibration"]).to_pandas()
cal = np.sort(sc.score[sc.calibration].to_numpy())
ref_rows = sc[sc.candidate_id.map(lambda c: key.get(int(c)) in ref)]
retention = {}
for q in [0.52, 0.54, 0.75, 0.90]:
    cut = cal[min(len(cal) - 1, math.ceil((len(cal) - 1) * q))]
    retention[q] = {"cutoff": float(cut), "removed": float((sc.score <= cut).mean()),
                    "reference_kept": float((ref_rows.score > cut).mean()) if len(ref_rows) else None}

lines = [f"{ROOT}: OFF accepted precursors (seed union) {len(ref)}, matched in the screened table {len(ref_rows)}"]
off_pep = mean_sd([s["peptides"] for s in res["OFF"]["seeds"]])[0]
for arm, r in res.items():
    pep = mean_sd([s["peptides"] for s in r["seeds"]])
    prec = mean_sd([s["precursors"] for s in r["seeds"]])
    fdp = mean_sd([s["fdp"] for s in r["seeds"]]) if ENTRAP else None
    st = r["stages"]
    top = ", ".join(f"{k} {v / 60:.1f}" for k, v in sorted(st.items(), key=lambda x: -x[1])[:7])
    lines.append(
        f"{arm}: peptides {pep[0]:.1f} (sd {pep[1]:.1f}, {100 * (pep[0] / off_pep - 1):+.2f}%), precursors {prec[0]:.1f}"
        + (f", FDP {fdp[0]:.3f}%" if fdp else "")
        + f"; wall {r['time']['wall_s'] / 60:.1f} min, peak {r['time']['peak_gb']:.1f} GB"
        + (f"; predicted {r['predicted'][0]} of {r['predicted'][1]} peptidoforms" if r["predicted"] else "")
        + (f"; sub-library {r['sublibrary'][0]} of {r['sublibrary'][1]}" if r["sublibrary"] else "")
        + f"\n    stages (min): {top}")
for q, v in retention.items():
    lines.append(f"  target {q:.2f}: removed {100 * v['removed']:.2f}%, OFF precursors kept "
                 f"{100 * v['reference_kept']:.2f}%  (cutoff {v['cutoff']:.4f})")
txt = "\n".join(lines)
open(f"{ROOT}/summary.txt", "w").write(txt + "\n")
json.dump({"arms": {a: {k: v for k, v in r.items() if k != "seeds"} | {
    "peptides": [s["peptides"] for s in r["seeds"]], "precursors": [s["precursors"] for s in r["seeds"]],
    "fdp": [s.get("fdp") for s in r["seeds"]]} for a, r in res.items()},
    "reference_n": len(ref), "retention": retention}, open(f"{ROOT}/summary.json", "w"), indent=1, default=str)
print(txt)
