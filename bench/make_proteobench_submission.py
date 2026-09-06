#!/usr/bin/env python
"""Build a ProteoBench 'Custom' input TSV (quant LFQ ion, DIA AIF HYE) from MuMDIA per-run
peptide_quant tables.

Usage:
  make_proteobench_submission.py --fasta ProteoBenchFASTA_MixedSpecies_HYE.fasta --out custom_input.tsv \
      LFQ_Orbitrap_AIF_Condition_A_Sample_Alpha_01=pepquant_A_01.parquet ... (six run=parquet pairs)

Format (ProteoBench parse_settings_custom.toml for the AIF module): tab-separated, columns
  Sequence, Proteins, Charge, Modified sequence, <one intensity column per run name>
Rows are precursor ions = (Modified sequence, Charge). Missing intensities are left empty.
Proteins are expanded from MuMDIA's entry names (e.g. ADH1_YEAST) to every matching FASTA
identifier (sp|P00330|ADH1_YEAST;sp|Cont_P00330|ADH1_YEAST) so that ProteoBench's own
`Cont_` contaminant flag and `_HUMAN/_YEAST/_ECOLI` species flags apply unchanged.
"""
import argparse
import re
import sys

import pandas as pd
import pyarrow.parquet as pq

STRIP_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|[^A-Z]")


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="RUN_NAME=peptide_quant.parquet pairs")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-fragments", type=int, default=1, help="minimum n_fragments_used per quantity")
    a = ap.parse_args()

    emap = fasta_entry_map(a.fasta)
    wide = None
    stats = []
    for spec in a.runs:
        run, path = spec.split("=", 1)
        t = pq.read_table(path).to_pandas()
        n0 = len(t)
        t = t[~t["peptidoform"].str.startswith("DECOY_")]
        t = t[t["quantity"].notna() & (t["quantity"] > 0)]
        if "n_fragments_used" in t.columns:
            t = t[t["n_fragments_used"] >= a.min_fragments]
        t = t[["peptidoform", "charge", "protein_group", "quantity"]].copy()
        # one value per precursor ion (the quant table is already unique per (peptidoform, charge))
        t = t.groupby(["peptidoform", "charge"], as_index=False).agg(
            protein_group=("protein_group", "first"), quantity=("quantity", "sum"))
        t = t.rename(columns={"quantity": run})
        stats.append((run, n0, len(t)))
        wide = t if wide is None else wide.merge(t, on=["peptidoform", "charge"], how="outer", suffixes=("", "_dup"))
        if "protein_group_dup" in wide.columns:
            wide["protein_group"] = wide["protein_group"].fillna(wide["protein_group_dup"])
            wide = wide.drop(columns=["protein_group_dup"])

    run_cols = [s.split("=", 1)[0] for s in a.runs]

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
    cont = out["Proteins"].str.contains("Cont_").sum()
    print(f"  contaminant-flagged rows: {cont}; entry names not in FASTA: {len(unmapped)} {unmapped[:10]}", file=sys.stderr)
    for sp in ("_HUMAN", "_YEAST", "_ECOLI"):
        print(f"  {sp}: {out['Proteins'].str.contains(sp).sum()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
