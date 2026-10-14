#!/usr/bin/env python
"""Compare a single-run MuMDIA result against a reference identification table at one q.

  tims_compare.py --reference reference_report.parquet \
      --mumdia out_run/psms_scored.parquet [--q 0.01] [--json out.json]

The reference is an independent identification table of the same run; `tims_ref.py` lists
the column names. This script reads `Precursor.Id`, `Stripped.Sequence`, `Q.Value`,
`Protein.Group` and `PG.Q.Value`.

Units, named because they differ between the two tables (CLAUDE.md, "Q-value columns"):
  reference  precursors      Precursor.Id with Q.Value <= q (run-level precursor q)
             peptides        distinct Stripped.Sequence among those precursors
             protein groups  distinct Protein.Group with PG.Q.Value <= q among those precursors
  MuMDIA     precursors      target (peptidoform, charge) rows with precursor_q <= q
             peptides        distinct stripped target sequences with peptide_q_value <= q
             protein groups  distinct target protein_group with pg_q_value <= q
The MuMDIA empirical decoy fraction is decoys / targets among peptide-level winners at q.
Overlap is on stripped sequences with I and L merged.

This compares end results only. Where the reference-only peptides were lost is answered per
precursor by `tims_loss.py`, and for every library candidate, stage by stage, by
`mumdia audit` (`candidate_audit.parquet`) and each stage's `<artifact>.report.json`.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tims_ref import il, read_reference, strip  # noqa: E402


def reference(path: str, q: float) -> dict:
    df = read_reference(path, ["Precursor.Id", "Stripped.Sequence", "Q.Value", "Protein.Group",
                               "PG.Q.Value"], optional=("Decoy",))
    pr = df[(df["Q.Value"] <= q) & (df["Decoy"] == 0)]
    return {
        "precursors": pr["Precursor.Id"].nunique(),
        "peptides": pr["Stripped.Sequence"].nunique(),
        "protein_groups": pr.loc[pr["PG.Q.Value"] <= q, "Protein.Group"].nunique(),
        "_peps": il(pr["Stripped.Sequence"].unique()),
    }


def mumdia(path: str, q: float) -> dict:
    df = pd.read_parquet(path, columns=["peptidoform", "charge", "label", "protein_group",
                                        "precursor_q", "peptide_q_value", "pg_q_value"])
    t = df[df["label"] == "target"]
    pep_win = df[df["peptide_q_value"] <= q]
    n_t = (pep_win["label"] == "target").sum()
    peps = t.loc[t["peptide_q_value"] <= q, "peptidoform"].map(strip)
    return {
        "precursors": len(t.loc[t["precursor_q"] <= q, ["peptidoform", "charge"]].drop_duplicates()),
        "peptides": peps.nunique(),
        "protein_groups": t.loc[t["pg_q_value"] <= q, "protein_group"].nunique(),
        "decoy_fraction_peptide": round(float((pep_win["label"] == "decoy").sum() / max(1, n_t)), 4),
        "_peps": il(peps.unique()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", required=True, help="reference precursor table (parquet)")
    ap.add_argument("--mumdia", required=True, help="single-run psms_scored.parquet")
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--json")
    a = ap.parse_args()
    r, m = reference(a.reference, a.q), mumdia(a.mumdia, a.q)
    rp, mp = r.pop("_peps"), m.pop("_peps")
    out = {
        "q": a.q, "reference": r, "mumdia": m,
        "mumdia_over_reference": {k: round(m[k] / r[k], 4)
                                  for k in ("precursors", "peptides", "protein_groups") if r[k]},
        "peptide_overlap_IL": {"both": len(rp & mp), "reference_only": len(rp - mp),
                               "mumdia_only": len(mp - rp)},
    }
    print(json.dumps(out, indent=2, default=int))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2, default=int)


if __name__ == "__main__":
    main()
