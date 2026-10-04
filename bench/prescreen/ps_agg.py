"""Aggregate one prescreen validation campaign (ps_val.sh) into summary.json and summary.txt.

usage: ps_agg.py ROOT [--entrapment]
"""
import glob
import json
import math
import os
import re
import sys

import pyarrow.parquet as pq

ROOT = sys.argv[1]
ENTRAP = "--entrapment" in sys.argv
ENT = {"marker": "ENTRAP_", "exclude": "REAL_",
       "contaminants": ["KRT", "K1C", "K2C", "ALBU", "TRYP"], "ratio": 0.560632}


def timing(name):
    p = f"{ROOT}/time_{name}.txt"
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p, errors="replace"):
        if "Elapsed (wall" in line:
            v = line.rsplit(" ", 1)[1].strip().split(":")
            s = 0.0
            for x in v:
                s = s * 60 + float(x)
            out["wall_s"] = s
        if "Maximum resident set size" in line:
            out["max_rss_gb"] = int(line.rsplit(" ", 1)[1]) / 1024 / 1024
    return out


def read(path, cols):
    have = set(pq.read_schema(path).names)
    return pq.read_table(path, columns=[c for c in cols if c in have]).to_pandas()


def is_entrap(prot):
    p = str(prot)
    if ENT["marker"] not in p or ENT["exclude"] in p:
        return False
    return not any(c in p for c in ENT["contaminants"])


def scored_stats(path):
    rep = json.load(open(path + ".report.json"))
    st = rep.get("stats", {})
    out = {"peptides": st.get("target_peptides_at_1pct"),
           "precursors": st.get("target_precursors_at_1pct"),
           "protein_groups": st.get("target_protein_groups_at_1pct")}
    t = read(path, ["candidate_id", "label", "peptidoform", "protein", "precursor_q",
                    "peptide_q_value", "base_peptide_id", "q_value"])
    dec = t["label"].astype(str).str.lower().str.startswith("decoy")
    tgt = t[~dec]
    acc = tgt[tgt["precursor_q"] <= 0.01]
    out["accepted_ids"] = set(int(x) for x in acc["candidate_id"])
    if ENTRAP:
        pep = tgt.sort_values("peptide_q_value").drop_duplicates("base_peptide_id")
        pep = pep[pep["peptide_q_value"] <= 0.01]
        ent = pep["protein"].map(is_entrap)
        n_real, n_ent = int((~ent).sum()), int(ent.sum())
        out["entrapment"] = {"real": n_real, "spike": n_ent,
                             "fdp_pct": 100 * (ENT["ratio"] * n_ent + 1) / max(n_real, 1)}
    return out


def mean_sd(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, sd


arms = [d for d in ["OFF", "PRESCAN", "CAP300", "SEN", "BAL", "STR", "AGG"]
        + sorted(os.path.basename(x) for x in glob.glob(f"{ROOT}/X_*"))
        + sorted(os.path.basename(x) for x in glob.glob(f"{ROOT}/S_*"))
        if os.path.isdir(f"{ROOT}/{d}")]
res = {}
for arm in arms:
    A = f"{ROOT}/{arm}"
    r = {"seeds": {}}
    for s in sorted(glob.glob(f"{A}/s*/psms_scored.parquet")):
        k = os.path.basename(os.path.dirname(s))
        try:
            r["seeds"][k] = scored_stats(s)
        except Exception as e:  # noqa: BLE001
            r["seeds"][k] = {"error": str(e)}
    surv = f"{A}/survivors.parquet"
    if os.path.exists(surv):
        r["survivors"] = set(int(x) for x in pq.read_table(surv, columns=["candidate_id"]).column(0).to_pylist())
        rep = json.load(open(surv + ".report.json"))
        r["screen_stats"] = rep.get("stats", {})
    sc = f"{A}/survivors.scores.parquet"
    if r.get("screen_stats", {}).get("sample_candidates") and os.path.exists(sc):
        r["universe"] = set(int(x) for x in pq.read_table(sc, columns=["candidate_id"]).column(0).to_pylist())
    ext = f"{A}/psms_extracted.parquet"
    if os.path.exists(ext):
        r["extracted_rows"] = pq.ParquetFile(ext).metadata.num_rows
    r["time"] = {k: timing(f"{arm}_{k}") for k in ["screen", "extract", "features", "compete"]}
    r["time"]["rescore"] = [timing(os.path.basename(x)[5:-4]) for x in sorted(glob.glob(f"{ROOT}/time_{arm}_rescore_s*.txt"))]
    res[arm] = r

base_rep = f"{ROOT}/BASE/psms_scored.parquet"
n_lib = None
if os.path.exists(f"{ROOT}/BASE/run_windows.parquet"):
    n_lib = pq.ParquetFile(f"{ROOT}/BASE/run_windows.parquet").metadata.num_rows

# Reference sets from OFF: accepted precursors (union over seeds), weak = the lowest quartile of
# OFF's extracted apex intensity among them, modified = carrying a variable modification.
off_ids = set()
for v in res.get("OFF", {}).get("seeds", {}).values():
    off_ids |= v.get("accepted_ids", set())
weak, modified = set(), set()
if off_ids and os.path.exists(f"{ROOT}/OFF/psms_extracted.parquet"):
    sch = pq.read_schema(f"{ROOT}/OFF/psms_extracted.parquet").names
    icol = next((c for c in ["apex_intensity", "sum_intensity", "intensity", "apex_sum", "area"] if c in sch), None)
    t = read(f"{ROOT}/OFF/psms_extracted.parquet", ["candidate_id", "peptidoform"] + ([icol] if icol else []))
    t = t[t["candidate_id"].isin(off_ids)]
    if icol:
        g = t.groupby("candidate_id")[icol].max()
        weak = set(int(x) for x in g[g <= g.quantile(0.25)].index)
    pf = t.drop_duplicates("candidate_id")
    modified = set(int(c) for c, p in zip(pf["candidate_id"], pf["peptidoform"])
                   if "[" in str(p).replace("C[Carbamidomethyl]", ""))

off_pep = mean_sd([v.get("peptides") for v in res.get("OFF", {}).get("seeds", {}).values()])
lines = []
summary = {"root": ROOT, "library_candidates": n_lib, "reference_n": len(off_ids),
           "weak_n": len(weak), "modified_n": len(modified), "arms": {}}
for arm, r in res.items():
    seeds = r["seeds"].values()
    pep = mean_sd([v.get("peptides") for v in seeds])
    prec = mean_sd([v.get("precursors") for v in seeds])
    fdp = mean_sd([v["entrapment"]["fdp_pct"] for v in seeds if "entrapment" in v])
    surv = r.get("survivors")
    st = r.get("screen_stats", {})
    row = {
        "seeds": len(r["seeds"]),
        "peptides_mean": pep[0], "peptides_sd": pep[1],
        "precursors_mean": prec[0], "precursors_sd": prec[1],
        "fdp_pct_mean": fdp[0], "fdp_pct_sd": fdp[1],
        "survivors": len(surv) if surv is not None else n_lib,
        "reduction": (1 - len(surv) / n_lib) if (surv is not None and n_lib) else 0.0,
        "reporting_reduction": st.get("reporting_reduction"),
        "cutoff": st.get("cutoff"), "calibration_n": st.get("calibration_n"),
        "target_retention_screen": st.get("target_retention"), "decoy_retention_screen": st.get("decoy_retention"),
        "extracted_rows": r.get("extracted_rows"),
        "time": r["time"],
    }
    uni = r.get("universe")
    if uni is not None:
        # Evaluation sample: denominators restricted to the sampled candidates.
        row["sampled"] = len(uni)
        row["reduction"] = st.get("reporting_reduction") or 0.0
        ref_u, weak_u, mod_u = off_ids & uni, weak & uni, modified & uni
        row["reference_n"], row["weak_n"], row["modified_n"] = len(ref_u), len(weak_u), len(mod_u)
        row["reference_retention"] = len(ref_u & surv) / len(ref_u) if ref_u else None
        row["weak_retention"] = len(weak_u & surv) / len(weak_u) if weak_u else None
        row["modified_retention"] = len(mod_u & surv) / len(mod_u) if mod_u else None
    elif surv is not None and off_ids:
        row["reference_retention"] = len(off_ids & surv) / len(off_ids)
        row["weak_retention"] = len(weak & surv) / len(weak) if weak else None
        row["modified_retention"] = len(modified & surv) / len(modified) if modified else None
    elif off_ids:
        row["reference_retention"] = row["weak_retention"] = row["modified_retention"] = 1.0
    if pep[0] and off_pep[0]:
        row["peptides_delta_pct"] = 100 * (pep[0] / off_pep[0] - 1)
    for k in ["retrieved", "not_retrieved", "retrieval_losses", "scoring_losses", "rescued_by_score",
              "retrieval_forms_examined", "retrieval_forms_skipped_by_family", "families", "tag_paths",
              "scope_reduction", "total_reduction", "in_scope", "bypass_reason", "extension_ms", "scoring_ms", "mass_hypotheses"]:
        if k in st:
            row[k] = st[k]
    summary["arms"][arm] = row
    t = r["time"]
    sw = (t.get("screen") or {}).get("wall_s")
    sm = (t.get("screen") or {}).get("max_rss_gb")
    ew = (t.get("extract") or {}).get("wall_s")
    em = (t.get("extract") or {}).get("max_rss_gb")
    fmt = lambda x, f: (f % x) if x is not None else "-"  # noqa: E731
    lines.append(
        f"{arm:8s} surv {row['survivors']!s:>10} red {fmt(100*row['reduction'],'%5.1f')}% "
        f"ref {fmt(100*row.get('reference_retention', 1.0) if row.get('reference_retention') is not None else None,'%6.2f')}% "
        f"weak {fmt(100*row['weak_retention'] if row.get('weak_retention') is not None else None,'%6.2f')}% "
        f"mod {fmt(100*row['modified_retention'] if row.get('modified_retention') is not None else None,'%6.2f')}% "
        f"pep {fmt(pep[0],'%8.1f')} (sd {fmt(pep[1],'%5.1f')}, {fmt(row.get('peptides_delta_pct'),'%+5.2f')}%) "
        f"prec {fmt(prec[0],'%8.1f')} FDP {fmt(fdp[0],'%5.3f')}% "
        f"cut {fmt(row['cutoff'],'%.4f')} screen {fmt(sw,'%6.1f')}s/{fmt(sm,'%5.1f')}GB "
        f"extract {fmt(ew,'%6.1f')}s/{fmt(em,'%5.1f')}GB rows {row['extracted_rows']}")
txt = (f"library candidates {n_lib}; OFF reference precursors (seed union) {len(off_ids)}, "
       f"weak {len(weak)}, modified {len(modified)}\n" + "\n".join(lines))
open(f"{ROOT}/summary.txt", "w").write(txt + "\n")
json.dump(summary, open(f"{ROOT}/summary.json", "w"), indent=1, default=str)
print(txt)
