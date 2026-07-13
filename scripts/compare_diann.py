"""Compare MuMDIA (Rust MVP) identifications against DIA-NN on the same run.

Run with the conda env that has pandas + pyarrow, e.g.:
  /c/Users/robbi/anaconda3/envs/py312_mumdia/python.exe scripts/compare_diann.py

Compares unique stripped peptide sequences at 1% FDR and their overlap.
"""
import re
import sys
import pandas as pd

OURS = "out_run/psms_scored.parquet"
DIANN = "out_diann/report.tsv"
Q = 0.01

def strip_mods(pf: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", pf)

def load_ours():
    df = pd.read_parquet(OURS)
    t = df[(df["label"] == "target") & (df["peptide_q_value"] <= Q)]
    peps = set(t["peptidoform"].map(strip_mods))
    return peps, len(t)

def load_diann():
    df = pd.read_csv(DIANN, sep="\t")
    # DIA-NN 1.9.x: precursor-level run q is 'Q.Value'; peptide sequence is
    # 'Stripped.Sequence'; precursor id is 'Precursor.Id'.
    qcol = "Q.Value" if "Q.Value" in df.columns else "Global.Q.Value"
    d = df[df[qcol] <= Q]
    peps = set(d["Stripped.Sequence"].unique())
    prec = d["Precursor.Id"].nunique() if "Precursor.Id" in d.columns else len(d)
    pg = None
    for c in ("Protein.Group", "Protein.Ids"):
        if c in d.columns:
            pg = d[c].nunique()
            break
    return peps, prec, pg

def main():
    ours, ours_psm = load_ours()
    diann, diann_prec, diann_pg = load_diann()
    inter = ours & diann
    union = ours | diann
    print("=" * 60)
    print(f"MuMDIA  target peptides @1% FDR : {len(ours)}   (PSMs {ours_psm})")
    print(f"DIA-NN  peptides @1% FDR        : {len(diann)}   (precursors {diann_prec}, protein groups {diann_pg})")
    print("-" * 60)
    print(f"overlap (shared peptides)       : {len(inter)}")
    print(f"MuMDIA-only                     : {len(ours - diann)}")
    print(f"DIA-NN-only                     : {len(diann - ours)}")
    print(f"Jaccard                         : {len(inter)/len(union):.3f}")
    if ours:
        print(f"fraction of MuMDIA IDs in DIA-NN: {len(inter)/len(ours):.3f}")
    print("=" * 60)

if __name__ == "__main__":
    sys.exit(main())
