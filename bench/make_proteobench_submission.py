#!/usr/bin/env python
"""Build a ProteoBench 'Custom' input TSV (quant LFQ ion, DIA HYE modules) from MuMDIA per-run
peptide_quant tables.

Usage:
  make_proteobench_submission.py --fasta ProteoBenchFASTA_MixedSpecies_HYE.fasta --out custom_input.tsv \
      LFQ_Orbitrap_AIF_Condition_A_Sample_Alpha_01=run/r0/peptide_quant.parquet ... (one RUN=parquet pair per run)

Format (ProteoBench parse_settings_custom.toml for the DIA ion modules): tab-separated, columns
  Sequence, Proteins, Charge, Modified sequence, <one intensity column per run name>
Rows are precursor ions = (Modified sequence, Charge). Missing intensities are left empty.
Proteins are expanded from MuMDIA's entry names (e.g. ADH1_YEAST) to every matching FASTA
identifier (sp|P00330|ADH1_YEAST;sp|Cont_P00330|ADH1_YEAST) so that ProteoBench's own
`Cont_` contaminant flag and `_HUMAN/_YEAST/_ECOLI` species flags apply unchanged.

With none of the options below, the output is the raw `peptide_quant.quantity` of every run,
unchanged.

--quantity selects the submitted value:
  raw      `peptide_quant.quantity` (default).
  maxlfq   the precursor table of the cross-run MaxLFQ output that `run-experiment` writes
           (`lfq_maxlfq.parquet.precursor.parquet`, one row per (peptidoform, charge, run)).
           For a precursor quantified from one feature, as here, its value is the raw quantity
           divided by one size factor per run, estimated over all ions and species together.
           Within a condition this acts like `aligned`. The factors also differ between
           conditions, and that part moves every ratio between two conditions by one amount:
           the species gaps stay, the human median moves.
           A value the raw table does not have stays empty.
  aligned  within-condition replicate alignment of the raw quantities. For every run the
           offset is the median over its ions of (log2 quantity minus the mean log2 quantity
           of the same ion over the runs of its condition); the run's values are divided by
           2^offset. The estimate never reads a species or protein label. It uses the
           condition of each run only to group replicates, and the replicates of one
           condition share one sample composition, so it cannot change composition: it removes
           loading or injection differences between replicates. The offsets of one condition
           average to about zero, so the ratio of an ion seen in every replicate barely moves;
           the ratio of an ion with a missing replicate loses the bias of the replicates it
           was seen in. It does not correct an offset between conditions.

--condition-gate keeps the values of an ion in a condition only if at least one run of that
condition both quantified the ion and identified it in the first pass (target,
`run_psm_q <= --gate-q`). The values of a condition that rests only on second-pass (match-
between-runs) evidence are left empty. This is rule F1 of `prov_filter.py`, read from the
current `run-experiment` layout. The evidence is a Boolean `first_pass_id` column when every
quant table carries one, else `<experiment>/first_pass/scored_combined.parquet`
(`--first-pass`), whose `source` is the experiment run index. The gate is an ion filter and
changes no value it keeps. It removes more finite ratios than ProteoBench's ion count shows,
because an ion that keeps values in only one condition is still counted; the summary line
prints both numbers.

maxlfq and the gate need the experiment run index of each input. It is read from
`experiment_manifest.json` (`--experiment-dir`, by default the directory two levels above the
first quant table) by matching each path against `experiment.peptide_quants`, with the parent
directory name `r<i>` as the fallback. aligned and the gate need the condition of each run:
`--conditions A,A,A,B,B,B`, or by default the `Condition_<X>` part of each run name.

The last stderr line is a summary: the ions written, the ProteoBench ion count (not a
contaminant, exactly one species flag of the module, values in at least --min-obs runs) and how
many of those have a finite ratio for each pair of conditions, in total and per species.
`--summary-json` writes the same numbers. `--species` must list the species flags of the module,
as its species_mapper does: the Low Input module maps only HUMAN and YEAST, so it needs
`--species HUMAN,YEAST` (with the default, ions carrying an E. coli flag are dropped or counted
as ambiguous, and the count no longer equals ProteoBench's).
"""
import argparse
import itertools
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

STRIP_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|[^A-Z]")
COND_RE = re.compile(r"Condition_([A-Za-z0-9]+)")
SPECIES = ("HUMAN", "YEAST", "ECOLI")
FIRST_PASS_COL = "first_pass_id"


def strip_sequence(pf: str) -> str:
    pf = pf.split("/")[0]
    pf = re.sub(r"^\[[^\]]*\]-", "", pf)          # N-terminal mod block
    pf = re.sub(r"-\[[^\]]*\]$", "", pf)          # C-terminal mod block
    return STRIP_RE.sub("", pf)


def fasta_entry_map(path: str) -> dict:
    """entry name -> list of full identifiers ('sp|ACC|ENTRY'), contaminant first."""
    m: dict = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            ident = line[1:].split()[0]
            parts = ident.split("|")
            entry = parts[2] if len(parts) >= 3 else ident
            m.setdefault(entry, []).append(ident)
    for v in m.values():
        v.sort(key=lambda s: ("Cont_" not in s, s))
    return m


def experiment_indices(paths, exp_dir):
    """Experiment run index of each quant table: manifest first, then the r<i> directory name."""
    man_path = os.path.join(exp_dir, "experiment_manifest.json")
    by_path = {}
    if os.path.exists(man_path):
        with open(man_path) as fh:
            man = json.load(fh)
        for i, p in enumerate(man.get("experiment", {}).get("peptide_quants", [])):
            by_path[os.path.realpath(p)] = i
    out = []
    for p in paths:
        i = by_path.get(os.path.realpath(p))
        if i is None:
            # the directory as given first, so that a staged tree of links r<i>/peptide_quant.parquet
            # to other tables works; then the resolved directory
            m = None
            for q in (os.path.abspath(p), os.path.realpath(p)):
                m = m or re.fullmatch(r"r(\d+)", os.path.basename(os.path.dirname(q)))
            if m is None:
                raise SystemExit(f"cannot find the experiment run index of {p}: it is not in "
                                 f"{man_path} and its directory is not named r<i>")
            i = int(m.group(1))
        out.append(i)
    if len(set(out)) != len(out):
        raise SystemExit(f"two inputs map to the same experiment run index: {out}")
    return out


def within_condition_offsets(long: pd.DataFrame) -> pd.Series:
    """Per-run log2 offset: median over the run's ions of log2 q minus the ion's condition mean."""
    lq = np.log2(long["quantity"])
    m = lq.groupby([long["key"], long["condition"]]).transform("mean")
    return (lq - m).groupby(long["run"]).median()


def species_flags(proteins: pd.Series, species=SPECIES):
    """ProteoBench rules on an expanded protein string: (species, scored)."""
    s = proteins.fillna("")
    cont = s.str.contains("Cont_", regex=False)
    flags = {sp: s.str.contains("_" + sp, regex=False) for sp in species}
    n = sum(f.astype(int) for f in flags.values())
    sp = pd.Series("", index=s.index, dtype=object)
    for k, f in flags.items():
        sp[f & (n == 1)] = k
    return sp, (~cont) & (n == 1)


def summarise(out: pd.DataFrame, run_cols, conds, min_obs: int, species=SPECIES) -> dict:
    sp, scored = species_flags(out["Proteins"], species)
    present = out[run_cols].notna()
    counted = scored & (present.sum(axis=1) >= min_obs)
    pairs, n_c = [], {}
    if conds is not None:
        labels = sorted(set(conds))
        n_c = {c: present[[r for r, k in zip(run_cols, conds) if k == c]].sum(axis=1) for c in labels}
        pairs = list(itertools.combinations(labels, 2))
    res = {"min_obs": min_obs, "ions_written": int(len(out)), "pb_ions": int(counted.sum())}
    for x, y in pairs:
        res[f"finite_{x}/{y}"] = int((counted & (n_c[x] > 0) & (n_c[y] > 0)).sum())
    res["species"] = {}
    for s in species:
        sel = counted & (sp == s)
        d = {"pb_ions": int(sel.sum())}
        for x, y in pairs:
            d[f"finite_{x}/{y}"] = int((sel & (n_c[x] > 0) & (n_c[y] > 0)).sum())
        res["species"][s] = d
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="RUN_NAME=peptide_quant.parquet pairs")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-fragments", type=int, default=1, help="minimum n_fragments_used per quantity")
    ap.add_argument("--quantity", choices=("raw", "maxlfq", "aligned"), default="raw",
                    help="submitted value: raw peptide_quant.quantity (default), the MaxLFQ "
                         "precursor table, or raw aligned within each condition")
    ap.add_argument("--condition-gate", action="store_true",
                    help="keep a condition's values of an ion only if one of its runs identified "
                         "the ion in the first pass")
    ap.add_argument("--gate-q", type=float, default=0.01,
                    help="first-pass run_psm_q threshold of the gate (default 0.01)")
    ap.add_argument("--experiment-dir", default=None,
                    help="run-experiment output directory (default: two levels above the first quant table)")
    ap.add_argument("--first-pass", default=None,
                    help="first-pass scored table (default <experiment>/first_pass/scored_combined.parquet)")
    ap.add_argument("--maxlfq", default=None,
                    help="MaxLFQ precursor table (default <experiment>/lfq_maxlfq.parquet.precursor.parquet)")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated condition of each run, in argument order "
                         "(default: Condition_<X> in the run name)")
    ap.add_argument("--min-obs", type=int, default=3,
                    help="minimum observed runs of an ion in the summary (ProteoBench default 3)")
    ap.add_argument("--summary-json", default=None, help="also write the summary as JSON")
    ap.add_argument("--species", default=",".join(SPECIES),
                    help="species flags of the module, for the summary only (default HUMAN,YEAST,ECOLI; "
                         "the Low Input module needs HUMAN,YEAST)")
    ap.add_argument("--quantity-column", default="quantity",
                    help="intensity column: quantity (default), or quantity_normalised of "
                         "peptide_quant_normalised tables (quant.normalisation)")
    a = ap.parse_args()

    specs = [s.split("=", 1) for s in a.runs]
    run_cols = [r for r, _ in specs]
    paths = [p for _, p in specs]

    conds = None
    if a.conditions:
        conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
        if len(conds) != len(run_cols):
            raise SystemExit(f"{len(run_cols)} runs but {len(conds)} conditions")
    else:
        found = [COND_RE.search(r) for r in run_cols]
        if all(found):
            conds = [m.group(1) for m in found]
    if conds is None and (a.quantity == "aligned" or a.condition_gate):
        raise SystemExit("--quantity aligned and --condition-gate need --conditions "
                         "(the run names carry no Condition_<X>)")

    exp_dir = a.experiment_dir or os.path.dirname(os.path.dirname(os.path.abspath(paths[0])))
    exp_idx = experiment_indices(paths, exp_dir) if (a.quantity == "maxlfq" or a.condition_gate) else None

    emap = fasta_entry_map(a.fasta)
    frames, stats, notes = [], [], []
    fp_col_everywhere = True
    for run, path in specs:
        t = pq.read_table(path).to_pandas()
        n0 = len(t)
        t = t[~t["peptidoform"].str.startswith("DECOY_")]
        if a.quantity_column != "quantity":
            t = t.drop(columns=["quantity"]).rename(columns={a.quantity_column: "quantity"})
        t = t[t["quantity"].notna() & (t["quantity"] > 0)]
        if "n_fragments_used" in t.columns:
            t = t[t["n_fragments_used"] >= a.min_fragments]
        agg = {"protein_group": ("protein_group", "first"), "quantity": ("quantity", "sum")}
        if FIRST_PASS_COL in t.columns:
            agg["fp"] = (FIRST_PASS_COL, "max")
        else:
            fp_col_everywhere = False
        # one value per precursor ion (the quant table is already unique per (peptidoform, charge))
        t = t.groupby(["peptidoform", "charge"], as_index=False).agg(**agg)
        stats.append((run, n0, len(t)))
        frames.append(t)

    if a.quantity != "raw" or a.condition_gate:
        long = pd.concat([f.assign(run=r) for f, r in zip(frames, run_cols)], ignore_index=True)
        long["key"] = long["peptidoform"] + "/" + long["charge"].astype(str)
        if conds is not None:
            long["condition"] = long["run"].map(dict(zip(run_cols, conds)))
        if exp_idx is not None:
            long["exp_index"] = long["run"].map(dict(zip(run_cols, exp_idx))).astype(np.int64)

        if a.quantity == "maxlfq":
            lpath = a.maxlfq or os.path.join(exp_dir, "lfq_maxlfq.parquet.precursor.parquet")
            lfq = pq.read_table(lpath, columns=["group", "charge", "run", "quantity"]).to_pandas()
            lfq = lfq.rename(columns={"group": "peptidoform", "run": "exp_index", "quantity": "lfq"})
            lfq["exp_index"] = lfq["exp_index"].astype(np.int64)
            n_before = len(long)
            long = long.merge(lfq, on=["peptidoform", "charge", "exp_index"], how="left")
            ok = long["lfq"].notna() & np.isfinite(long["lfq"]) & (long["lfq"] > 0)
            long = long[ok].copy()
            long["quantity"] = long["lfq"]
            long = long.drop(columns=["lfq"])
            notes.append(f"maxlfq: {lpath}; {n_before - len(long)} of {n_before} raw values have no MaxLFQ value")
        elif a.quantity == "aligned":
            off = within_condition_offsets(long)
            long["quantity"] = long["quantity"] / np.exp2(long["run"].map(off))
            notes.append("aligned: per-run log2 offsets " + "; ".join(f"{r} {off[r]:+.4f}" for r in run_cols))

        if a.condition_gate:
            if fp_col_everywhere:
                long["fp"] = long["fp"].astype(bool)
                src = f"column {FIRST_PASS_COL}"
            else:
                long = long.drop(columns=["fp"], errors="ignore")
                fpath = a.first_pass or os.path.join(exp_dir, "first_pass", "scored_combined.parquet")
                if not os.path.exists(fpath):
                    raise SystemExit(f"--condition-gate: no {FIRST_PASS_COL} column and no first-pass table {fpath}")
                fp = pq.read_table(fpath, columns=["peptidoform", "charge", "label", "source", "run_psm_q"]).to_pandas()
                fp = fp[(fp["label"].astype(str) == "target") & (fp["run_psm_q"] <= a.gate_q)]
                fp = fp[["peptidoform", "charge", "source"]].drop_duplicates()
                fp = fp.rename(columns={"source": "exp_index"}).assign(fp=True)
                fp["exp_index"] = fp["exp_index"].astype(np.int64)
                long = long.merge(fp, on=["peptidoform", "charge", "exp_index"], how="left")
                long["fp"] = long["fp"].fillna(False).astype(bool)
                src = f"{fpath}, run_psm_q <= {a.gate_q}"
            keep = long.groupby(["key", "condition"])["fp"].transform("any")
            notes.append(f"condition gate ({src}): {int((~keep).sum())} of {len(long)} values removed")
            long = long[keep]

        frames = [long[long["run"] == r][["peptidoform", "charge", "protein_group", "quantity"]].reset_index(drop=True)
                  for r in run_cols]

    wide = None
    for run, t in zip(run_cols, frames):
        t = t[["peptidoform", "charge", "protein_group", "quantity"]].rename(columns={"quantity": run})
        wide = t if wide is None else wide.merge(t, on=["peptidoform", "charge"], how="outer", suffixes=("", "_dup"))
        if "protein_group_dup" in wide.columns:
            wide["protein_group"] = wide["protein_group"].fillna(wide["protein_group_dup"])
            wide = wide.drop(columns=["protein_group_dup"])

    def expand(pg: str) -> str:
        out = []
        for name in str(pg).split(";"):
            name = name.strip()
            if not name:
                continue
            out.extend(emap.get(name, [name]))
        return ";".join(dict.fromkeys(out))

    wide["Proteins"] = wide["protein_group"].map(expand)
    wide["Sequence"] = wide["peptidoform"].map(strip_sequence)
    wide["Modified sequence"] = wide["peptidoform"]
    wide["Charge"] = wide["charge"].astype(int)
    out = wide[["Sequence", "Proteins", "Charge", "Modified sequence"] + run_cols]
    out.to_csv(a.out, sep="\t", index=False, float_format="%.6g")

    unmapped = sorted({n for pg in wide["protein_group"] for n in str(pg).split(";") if n and n not in emap})
    print(f"wrote {a.out}: {len(out)} precursor ions, {len(run_cols)} runs", file=sys.stderr)
    for run, n0, n1 in stats:
        print(f"  {run}: {n0} quant rows -> {n1} positive target ions", file=sys.stderr)
    for n in notes:
        print(f"  {n}", file=sys.stderr)
    cont = out["Proteins"].str.contains("Cont_").sum()
    print(f"  contaminant-flagged rows: {cont}; entry names not in FASTA: {len(unmapped)} {unmapped[:10]}", file=sys.stderr)
    for sp in ("_HUMAN", "_YEAST", "_ECOLI"):
        print(f"  {sp}: {out['Proteins'].str.contains(sp).sum()}", file=sys.stderr)

    species = tuple(x.strip().upper() for x in a.species.split(",") if x.strip())
    summ = summarise(out, run_cols, conds, a.min_obs, species)
    summ = {"quantity": a.quantity, "condition_gate": bool(a.condition_gate), **summ}
    parts = [f"quantity {a.quantity}", f"gate {'on' if a.condition_gate else 'off'}",
             f"ions written {summ['ions_written']}", f"ProteoBench ions (min_obs {a.min_obs}) {summ['pb_ions']}"]
    parts += [f"finite {k[7:]} {v}" for k, v in summ.items() if k.startswith("finite_")]
    for s in species:
        d = summ["species"][s]
        parts.append(f"{s} ions {d['pb_ions']}" + "".join(
            f", finite {k[7:]} {v}" for k, v in d.items() if k.startswith("finite_")))
    print("summary: " + "; ".join(parts), file=sys.stderr)
    if a.summary_json:
        with open(a.summary_json, "w") as fh:
            json.dump(summ, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
