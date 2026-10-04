#!/usr/bin/env python
"""Compare a single-run MuMDIA result against a DIA-NN report at one q threshold.

  tims_compare.py --diann output_diann/report.parquet \
      --mumdia output_mumdia/psms_scored.parquet [--q 0.01] [--json out.json]

Units, named because they differ between engines (CLAUDE.md, "Q-value columns"):
  DIA-NN   precursors      Precursor.Id with Q.Value <= q (run-level precursor q)
           peptides        distinct Stripped.Sequence among those precursors
           protein groups  distinct Protein.Group with PG.Q.Value <= q among those precursors
  MuMDIA   precursors      target (peptidoform, charge) rows with precursor_q <= q
           peptides        distinct stripped target sequences with peptide_q_value <= q
           protein groups  distinct target protein_group with pg_q_value <= q
The MuMDIA empirical decoy fraction is decoys / targets among peptide-level winners at q.
Overlap is on stripped sequences with I and L merged.

With --extracted (and optionally --lib), the DIA-NN-only peptides are split into the loss
ladder of docs/TIMS_ROADMAP.md section 2: not in the library, in the library but never
accepted by extract (no target row in psms_extracted), and extracted but below q.
Extract acceptances are also counted (candidate rows at peak_rank 0).
"""
import argparse
import json
import re

import pandas as pd

_MOD = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def strip(pf: str) -> str:
    return _MOD.sub("", pf.removeprefix("DECOY_")).replace("-", "").replace(".", "")


def il(seqs) -> set:
    return {s.replace("I", "L") for s in seqs}


def diann(path: str, q: float) -> dict:
    df = pd.read_parquet(path, columns=["Precursor.Id", "Stripped.Sequence", "Q.Value",
                                        "Protein.Group", "PG.Q.Value"])
    pr = df[df["Q.Value"] <= q]
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


def ladder(missing: set, extracted: str, lib: str | None) -> dict:
    ex = pd.read_parquet(extracted, columns=["peptidoform", "label", "peak_rank"])
    t = ex[ex["label"] == "target"]
    ext = il(t["peptidoform"].map(strip).unique())
    out = {"extract_accepted_rank0": int((ex["peak_rank"] == 0).sum())}
    if lib:
        lb = pd.read_parquet(lib, columns=["peptidoform", "label"])
        inlib = il(lb.loc[lb["label"] == "target", "peptidoform"].map(strip).unique())
        out["not_in_library"] = len(missing - inlib)
        missing = missing & inlib
    out["never_extracted"] = len(missing - ext)
    out["extracted_below_q"] = len(missing & ext)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diann", required=True)
    ap.add_argument("--mumdia", required=True)
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--json")
    ap.add_argument("--extracted", help="psms_extracted.parquet, for the loss ladder")
    ap.add_argument("--lib", help="fragment_library_precursors.parquet, for 'not in library'")
    a = ap.parse_args()
    d, m = diann(a.diann, a.q), mumdia(a.mumdia, a.q)
    dp, mp = d.pop("_peps"), m.pop("_peps")
    out = {
        "q": a.q, "diann": d, "mumdia": m,
        "mumdia_over_diann": {k: round(m[k] / d[k], 4) for k in ("precursors", "peptides", "protein_groups") if d[k]},
        "peptide_overlap_IL": {"both": len(dp & mp), "diann_only": len(dp - mp), "mumdia_only": len(mp - dp)},
    }
    if a.extracted:
        out["loss_ladder"] = ladder(dp - mp, a.extracted, a.lib)
    print(json.dumps(out, indent=2, default=int))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2, default=int)


if __name__ == "__main__":
    assert strip("DECOY_PEPM[Oxidation]C[Carbamidomethyl]K") == "PEPMCK"
    main()
