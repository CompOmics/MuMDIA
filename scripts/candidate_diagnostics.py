#!/usr/bin/env python
"""Per-candidate diagnostic bundle export (sensitivity_plan backlog P7.2).

Implements the individual-candidate diagnostic packet described in
sensitivity_plan/02_sensitivity_diagnostic_plan.md section 9. For a selected
list of candidate_ids the script reads the run artifacts (non-invasively, never
writing back to them) and produces, per candidate:

  fragments.png            overlaid fragment chromatograms (MS2 left axis, MS1
                           right axis via twinx, apex and reference RT markers)
  predicted_vs_observed.png predicted vs observed-at-apex intensity per fragment
  candidate.json           metadata, per-fragment summary, and all comp features

A top-level index.txt lists the exported candidates and their key fields.

Design notes:
  - Deterministic: fixed Agg backend, sorted candidate order, sorted fragment
    order, no random state.
  - Bounded memory: chrom / comp / psms / scored are read with a pyarrow
    dataset filter (predicate pushdown) restricted to the requested
    candidate_ids, so the full tables are never materialized.
  - Robust to missing MS1 traces, missing comp, and missing scored inputs.

Interpreter: C:/Users/robbi/anaconda3/envs/py312_mumdia/python.exe
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")  # headless, deterministic rendering

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds


# --------------------------------------------------------------------------- #
# Fragment-name parsing and ordering
# --------------------------------------------------------------------------- #

_FRAG_RE = re.compile(r"^(?P<series>[a-zA-Z]+)(?P<ordinal>\d+)(?:\^(?P<charge>\d+))?$")


def is_ms1_name(name: str) -> bool:
    """Return True for MS1 precursor-isotope trace rows (ms1_mono, ms1_iso1, ...)."""
    return str(name).lower().startswith("ms1")


def frag_sort_key(name: str):
    """Deterministic, human-readable ordering key for fragment names.

    MS2 ions sort by series letter, ordinal, then fragment charge. Names that do
    not parse fall back to a lexical key placed after parsed ions.
    """
    m = _FRAG_RE.match(str(name))
    if not m:
        return (2, str(name), 0, 0)
    series = m.group("series").lower()
    ordinal = int(m.group("ordinal"))
    charge = int(m.group("charge")) if m.group("charge") else 1
    return (0, series, ordinal, charge)


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def load_filtered(path: str, cand_ids, columns=None):
    """Load only the rows for cand_ids from a parquet file (predicate pushdown).

    Returns a pandas DataFrame, or None if the path is falsy. Missing columns
    are tolerated: only columns present in the file schema are requested.
    """
    if not path:
        return None
    dataset = ds.dataset(path, format="parquet")
    if columns is not None:
        available = set(dataset.schema.names)
        columns = [c for c in columns if c in available]
        if "candidate_id" not in columns:
            columns = ["candidate_id"] + columns
    table = dataset.to_table(
        filter=ds.field("candidate_id").isin(list(cand_ids)),
        columns=columns,
    )
    return table.to_pandas()


def index_by_candidate(df):
    """Return {candidate_id: first-row dict} for quick per-candidate lookup."""
    out = {}
    if df is None:
        return out
    for row in df.to_dict("records"):
        cid = int(row["candidate_id"])
        if cid not in out:  # keep first occurrence for determinism
            out[cid] = row
    return out


def as_float_array(seq):
    """Coerce a nullable list column value to a 1-D float ndarray (empty if None)."""
    if seq is None:
        return np.empty(0, dtype=float)
    arr = np.asarray(list(seq), dtype=float)
    return arr


def value_at_rt(rt, intensity, target_rt):
    """Observed intensity at the scan nearest target_rt, or None if unavailable."""
    rt = as_float_array(rt)
    intensity = as_float_array(intensity)
    if rt.size == 0 or intensity.size == 0 or target_rt is None:
        return None
    n = min(rt.size, intensity.size)
    idx = int(np.argmin(np.abs(rt[:n] - float(target_rt))))
    return float(intensity[idx])


def count_nonzero(intensity):
    intensity = as_float_array(intensity)
    if intensity.size == 0:
        return 0
    return int(np.count_nonzero(intensity > 0.0))


def json_default(obj):
    """Make numpy scalars / arrays JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def clean_scalar(v):
    """Normalize a scalar for JSON: NaN/inf -> None, numpy -> python."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return v if np.isfinite(v) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# --------------------------------------------------------------------------- #
# Reference RT table
# --------------------------------------------------------------------------- #

def parse_ref_rt_file(path):
    """Parse a candidate_id,rt reference table. Returns {candidate_id: rt}.

    Accepts an optional header line and comma or whitespace separation. Lines
    that do not parse as (int, float) are skipped.
    """
    ref = {}
    if not path:
        return ref
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\t ]+", line)
            if len(parts) < 2:
                continue
            try:
                cid = int(float(parts[0]))
                rt = float(parts[1])
            except ValueError:
                continue  # header or malformed line
            ref[cid] = rt
    return ref


# --------------------------------------------------------------------------- #
# Candidate id selection
# --------------------------------------------------------------------------- #

def parse_candidate_list(arg_value, arg_file):
    ids = []
    if arg_value:
        for tok in re.split(r"[,\s]+", arg_value.strip()):
            if tok:
                ids.append(int(tok))
    if arg_file:
        with open(arg_file, "r", encoding="utf-8") as fh:
            for line in fh:
                for tok in re.split(r"[,\s]+", line.strip()):
                    if tok and not tok.startswith("#"):
                        ids.append(int(tok))
    # de-duplicate, preserve first-seen order, then sort for determinism
    seen = set()
    uniq = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return sorted(uniq)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_fragments(cand_id, frag_rows, ms1_rows, apex_rt, ref_rt,
                   ms1_scalars, title, out_path):
    """Overlaid fragment chromatograms with MS1 on a separate twin axis."""
    fig, ax = plt.subplots(figsize=(11, 6))

    cmap = plt.get_cmap("tab20")
    handles, labels = [], []

    # MS2 fragment traces on the left axis.
    for i, row in enumerate(frag_rows):
        rt = as_float_array(row["rt"])
        inten = as_float_array(row["intensity"])
        if rt.size == 0 or inten.size == 0:
            continue
        n = min(rt.size, inten.size)
        color = cmap(i % 20)
        (line,) = ax.plot(rt[:n], inten[:n], color=color, linewidth=1.2,
                          label=row["frag_name"])
        handles.append(line)
        labels.append(row["frag_name"])

    ax.set_xlabel("Retention time (s)")
    ax.set_ylabel("MS2 fragment intensity")
    ax.set_title(title)

    # MS1 traces on a separate twin axis (never normalized into MS2 scaling).
    ax2 = None
    if ms1_rows:
        ax2 = ax.twinx()
        ms1_cmap = plt.get_cmap("Dark2")
        for j, row in enumerate(ms1_rows):
            rt = as_float_array(row["rt"])
            inten = as_float_array(row["intensity"])
            if rt.size == 0 or inten.size == 0:
                continue
            n = min(rt.size, inten.size)
            color = ms1_cmap(j % 8)
            (line,) = ax2.plot(rt[:n], inten[:n], color=color, linewidth=1.4,
                              linestyle="--", label=row["frag_name"])
            handles.append(line)
            labels.append(row["frag_name"])
        ax2.set_ylabel("MS1 precursor-isotope intensity")
    elif ms1_scalars:
        # No MS1 traces; show available MS1 XIC scalars as points at the apex.
        ax2 = ax.twinx()
        ms1_cmap = plt.get_cmap("Dark2")
        for j, (name, val) in enumerate(sorted(ms1_scalars.items())):
            if val is None or apex_rt is None:
                continue
            color = ms1_cmap(j % 8)
            pt = ax2.scatter([apex_rt], [val], color=color, marker="D", s=40,
                             zorder=5, label=name + " (scalar)")
            handles.append(pt)
            labels.append(name + " (scalar)")
        ax2.set_ylabel("MS1 precursor-isotope intensity (scalar)")

    # Apex and reference RT markers.
    if apex_rt is not None:
        vl = ax.axvline(apex_rt, color="black", linewidth=1.4, linestyle="-",
                        label="apex_rt")
        handles.append(vl)
        labels.append("apex_rt")
    if ref_rt is not None:
        vr = ax.axvline(ref_rt, color="red", linewidth=1.4, linestyle=":",
                        label="reference_rt")
        handles.append(vr)
        labels.append("reference_rt")

    if handles:
        ax.legend(handles, labels, fontsize=7, ncol=2, loc="upper right",
                  framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_predicted_vs_observed(frag_summ, title, out_path):
    """Grouped bar chart: predicted intensity vs observed-at-apex per fragment."""
    names = [f["name"] for f in frag_summ]
    pred = np.array([f["predicted_intensity"] or 0.0 for f in frag_summ], float)
    obs = np.array(
        [(f["observed_apex_intensity"] or 0.0) for f in frag_summ], float
    )

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(names) + 2), 6))
    if names:
        x = np.arange(len(names))
        width = 0.4
        # Normalize each series to its own max so predicted (relative) and
        # observed (raw counts) shapes are comparable on one axis.
        pred_n = pred / pred.max() if pred.max() > 0 else pred
        obs_n = obs / obs.max() if obs.max() > 0 else obs
        ax.bar(x - width / 2, pred_n, width, label="predicted (norm)",
               color="#4C72B0")
        ax.bar(x + width / 2, obs_n, width, label="observed at apex (norm)",
               color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Relative intensity (per-series max = 1)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no MS2 fragments", ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-candidate bundle
# --------------------------------------------------------------------------- #

FEATURE_SKIP = {
    "candidate_id", "label", "base_peptide_id", "peptidoform", "protein",
    "apex_rt", "precursor_mz",
}


def build_bundle(cand_id, chrom_df, psms_row, comp_row, scored_row, ref_rt,
                 out_dir):
    """Write fragments.png, predicted_vs_observed.png and candidate.json.

    Returns a summary dict used for index.txt.
    """
    cdir = os.path.join(out_dir, str(cand_id))
    os.makedirs(cdir, exist_ok=True)

    # Split chrom rows into MS2 fragments and MS1 traces, sorted deterministically.
    rows = chrom_df.to_dict("records") if chrom_df is not None else []
    frag_rows = [r for r in rows if not is_ms1_name(r["frag_name"])]
    ms1_rows = [r for r in rows if is_ms1_name(r["frag_name"])]
    frag_rows.sort(key=lambda r: frag_sort_key(r["frag_name"]))
    ms1_rows.sort(key=lambda r: str(r["frag_name"]))

    # Metadata, tolerant of missing psms / scored rows.
    def g(row, key, default=None):
        if row is None or key not in row:
            return default
        return clean_scalar(row[key])

    peptidoform = g(psms_row, "peptidoform") or g(scored_row, "peptidoform") \
        or g(comp_row, "peptidoform")
    charge = g(psms_row, "charge")
    if charge is None:
        charge = g(scored_row, "charge")
    if charge is None:
        charge = g(comp_row, "charge")
    label = g(psms_row, "label") or g(scored_row, "label") or g(comp_row, "label")
    protein = g(psms_row, "protein") or g(scored_row, "protein") \
        or g(comp_row, "protein")
    apex_rt = g(psms_row, "apex_rt")
    if apex_rt is None:
        apex_rt = g(comp_row, "apex_rt")
    q_value = g(scored_row, "q_value")
    n_matched = g(psms_row, "n_matched_fragments")

    # MS1 XIC scalars from psms (fallback plotting + JSON record).
    ms1_scalars = {}
    for key in ("ms1_isom1", "ms1_mono", "ms1_iso1", "ms1_iso2"):
        val = g(psms_row, key)
        if val is not None:
            ms1_scalars[key] = val

    # Per-fragment summary (MS2 fragments only).
    frag_summ = []
    for r in frag_rows:
        frag_summ.append({
            "name": r["frag_name"],
            "frag_mz": clean_scalar(r.get("frag_mz")),
            "predicted_intensity": clean_scalar(r.get("predicted_intensity")),
            "observed_apex_intensity": value_at_rt(
                r.get("rt"), r.get("intensity"), apex_rt),
            "n_nonzero_points": count_nonzero(r.get("intensity")),
        })

    charge_str = "" if charge is None else "+%d" % int(charge)
    title = "%s%s  %s  (candidate %d)" % (
        peptidoform or "?", charge_str, label or "?", cand_id)

    plot_fragments(
        cand_id, frag_rows, ms1_rows, apex_rt, ref_rt, ms1_scalars, title,
        os.path.join(cdir, "fragments.png"),
    )
    plot_predicted_vs_observed(
        frag_summ, title, os.path.join(cdir, "predicted_vs_observed.png"),
    )

    # Feature values from comp (all columns except identity/context fields).
    features = {}
    if comp_row is not None:
        for k, v in comp_row.items():
            if k in FEATURE_SKIP:
                continue
            features[k] = clean_scalar(v)

    bundle = {
        "candidate_id": cand_id,
        "peptidoform": peptidoform,
        "charge": None if charge is None else int(charge),
        "label": label,
        "protein": protein,
        "apex_rt": apex_rt,
        "reference_rt": ref_rt,
        "q_value": q_value,
        "n_matched_fragments": None if n_matched is None else int(n_matched),
        "n_fragments_chrom": len(frag_rows),
        "n_ms1_traces": len(ms1_rows),
        "ms1_scalars": ms1_scalars,
        "fragments": frag_summ,
        "features": features,
    }
    with open(os.path.join(cdir, "candidate.json"), "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, default=json_default)

    return {
        "candidate_id": cand_id,
        "peptidoform": peptidoform or "?",
        "charge": "" if charge is None else int(charge),
        "label": label or "?",
        "q_value": q_value,
        "apex_rt": apex_rt,
        "n_matched_fragments": n_matched,
        "n_fragments": len(frag_rows),
        "protein": protein or "?",
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Export per-candidate diagnostic bundles (sensitivity_plan "
                    "P7.2, spec 02 section 9).",
    )
    ap.add_argument("--chrom", required=True, help="chrom.parquet path")
    ap.add_argument("--psms", required=True, help="psms.parquet path")
    ap.add_argument("--comp", default=None, help="comp.parquet path (optional)")
    ap.add_argument("--scored", default=None,
                    help="scored.parquet path (optional)")
    ap.add_argument("--candidates", default=None,
                    help="comma-separated candidate_ids")
    ap.add_argument("--candidates-file", default=None,
                    help="file with candidate_ids (comma/whitespace/newline)")
    ap.add_argument("--out", default="candidate_diag",
                    help="output directory (default: candidate_diag)")
    ap.add_argument("--ref-rt-file", default=None,
                    help="reference RT table: candidate_id,rt per line")
    args = ap.parse_args(argv)

    cand_ids = parse_candidate_list(args.candidates, args.candidates_file)
    if not cand_ids:
        ap.error("no candidate_ids given (use --candidates or --candidates-file)")

    os.makedirs(args.out, exist_ok=True)
    ref_rts = parse_ref_rt_file(args.ref_rt_file)

    # Bounded reads: only rows for the requested candidate_ids.
    chrom_df = load_filtered(args.chrom, cand_ids)
    psms_idx = index_by_candidate(load_filtered(args.psms, cand_ids))
    comp_idx = index_by_candidate(load_filtered(args.comp, cand_ids))
    scored_idx = index_by_candidate(load_filtered(args.scored, cand_ids))

    # Group chrom rows per candidate once.
    chrom_by_cand = {}
    if chrom_df is not None and len(chrom_df):
        for cid, grp in chrom_df.groupby("candidate_id"):
            chrom_by_cand[int(cid)] = grp

    summaries = []
    for cid in cand_ids:
        sub = chrom_by_cand.get(cid)
        if sub is None:
            print("warning: no chrom rows for candidate_id %d" % cid,
                  file=sys.stderr)
        summ = build_bundle(
            cid,
            sub,
            psms_idx.get(cid),
            comp_idx.get(cid),
            scored_idx.get(cid),
            ref_rts.get(cid),
            args.out,
        )
        summaries.append(summ)
        print("wrote %s" % os.path.join(args.out, str(cid)))

    # Top-level index.
    index_path = os.path.join(args.out, "index.txt")
    header = ["candidate_id", "peptidoform", "charge", "label", "q_value",
              "apex_rt", "n_matched_fragments", "n_fragments", "protein"]
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for s in summaries:
            qv = "" if s["q_value"] is None else "%.6g" % s["q_value"]
            ar = "" if s["apex_rt"] is None else "%.4f" % s["apex_rt"]
            nm = "" if s["n_matched_fragments"] is None else str(
                int(s["n_matched_fragments"]))
            fh.write("\t".join([
                str(s["candidate_id"]),
                str(s["peptidoform"]),
                str(s["charge"]),
                str(s["label"]),
                qv,
                ar,
                nm,
                str(s["n_fragments"]),
                str(s["protein"]),
            ]) + "\n")
    print("wrote %s" % index_path)


if __name__ == "__main__":
    main()
