"""Contaminant-aware entrapment true-FDR measurement.

The entrapment design spikes the human proteome into a pure-E. coli search and
treats every human ID as false. That over-counts: real handling contaminants
(keratins, albumin) and abundant proteins genuinely present are corroborated by
an independent engine (DIA-NN) on the same file. This script re-measures E. coli
at a true 1% FDR while EXCLUDING demonstrably-real human peptides from (a) the
training negatives and (b) the FDR false count, under three definitions:
  raw   - every human ID is false (original, pessimistic)
  crap  - exclude cRAP-style contaminant proteins (keratin/albumin/Hb/actin/...)
  diann - additionally exclude human sequences DIA-NN also identifies (present)

Usage: python entrapment_corrected.py <scored.parquet> [tag]
Operates POST-HOC on a shipped rescorer's scores (fixed), recomputing E. coli at
true 1% FDR under each accounting definition. The `raw` column reproduces the
shipped baseline; `crap`/`diann` show the count once contaminants are removed
from the false population. (An earlier retrain sweep confirmed excluding
contaminants from the TRAINING negatives is near-neutral, since they are a tiny
fraction of the ~40k human negatives; the accounting is the first-order fix.)
"""
import json
import re
import sys

import numpy as np
import pandas as pd

COMP = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else COMP

# cRAP-style contaminant entry-name tokens (UniProt _HUMAN naming in the protein col).
CRAP = re.compile(
    r"K[12][CH]\d|K22E|K1H\d|K2M\d|KRT\d|KRHB|ALBU|HB[ABDGE]\d?_|"
    r"ACT[BGSC]_|TBB\d|TBA\d|TRY[PB]|IGHG|IGKC|IGLC|CO\dA\d|CASA|CASB|CASK",
    re.I,
)


def strip(pf):
    return re.sub(r"\[[^\]]*\]", "", str(pf)).replace("DECOY_", "")


# DIA-NN-corroborated human sequences on the same raw file (independent-engine presence).
dn = pd.read_csv("out_diann/report_ent.tsv", sep="\t")
pc = "Protein.Names" if "Protein.Names" in dn.columns else "Protein.Ids"
dn = dn[dn["Q.Value"] <= 0.01]
dn_hum = set(
    dn[dn[pc].astype(str).str.contains("_HUMAN") & ~dn[pc].astype(str).str.contains("_ECOLI")][
        "Stripped.Sequence"
    ]
)

# library ratio N_ecoli_lib / N_human_lib (contaminant fraction of the 343k-peptide
# human library is negligible, so the ratio is held constant across definitions).
pep = pd.read_parquet("out_full/pep_ent.parquet", columns=["protein"])
ish = pep.protein.str.contains("_HUMAN") & ~pep.protein.str.contains("_ECOLI")
RATIO = int((~ish).sum()) / int(ish.sum())

c = pd.read_parquet(COMP)
c["strip"] = c.peptidoform.map(strip)
c["is_decoy"] = c.label.eq("decoy")
pcol = "protein_group" if "protein_group" in c.columns else "protein"
c["is_human"] = c[pcol].str.contains("_HUMAN") & ~c[pcol].str.contains("_ECOLI")
c["crap"] = c.is_human & c[pcol].map(lambda p: bool(CRAP.search(str(p))))
c["diann_corr"] = c.is_human & c.strip.isin(dn_hum)
SCORE = c.score.to_numpy(float)
dec = c.is_decoy.to_numpy()
hum = c.is_human.to_numpy()


def ent_q(sc, is_entrap, is_real, ratio):
    order = np.argsort(-sc, kind="stable")
    ne = nr = 0
    fdr = np.ones(len(sc))
    for rank, i in enumerate(order):
        if is_entrap[i]:
            ne += 1
        elif is_real[i]:
            nr += 1
        fdr[rank] = ratio * ne / max(1, nr)
    q = np.ones(len(sc))
    qmin = 1.0
    for rank in range(len(sc) - 1, -1, -1):
        qmin = min(qmin, fdr[rank])
        q[order[rank]] = qmin
    return q


def contam_mask(acct):
    if acct == "raw":
        return np.zeros(len(c), bool)
    if acct == "crap":
        return c.crap.to_numpy()
    return (c.crap | c.diann_corr).to_numpy()


def eco_at_true1(acct):
    contam = contam_mask(acct)
    is_entrap = hum & ~contam & ~dec     # valid false population (exclude human decoys)
    is_real = ~dec & ~hum                # E. coli targets
    q = ent_q(SCORE, is_entrap, is_real, RATIO)
    gate = q <= 0.01
    eco = c.loc[is_real & gate, "strip"].nunique()
    leak = c.loc[is_entrap & gate, "strip"].nunique()
    return eco, leak


print(f"=== {TAG} | ratio(eco/hum)={RATIO:.3f} | rows={len(c)} | "
      f"human PSMs={int(hum.sum())} cRAP={int(c.crap.sum())} DIANN-corr={int(c.diann_corr.sum())} ===")

# Framing A: hold the SHIPPED reported gate fixed (q_value<=0.01); the E. coli
# count is unchanged, but the TRUE FDR there drops once contaminants are not
# counted as false -> shows the shipped count is conservative.
if "q_value" in c.columns:
    gate = c.q_value.to_numpy() <= 0.01
    eco_fixed = c.loc[(~dec) & (~hum) & gate, "strip"].nunique()
    print(f"  [A] at shipped gate (q<=0.01): E.coli={eco_fixed}   true FDR by accounting:")
    for a in ("raw", "crap", "diann"):
        contam = contam_mask(a)
        false_h = c.loc[(hum & ~contam & ~dec) & gate, "strip"].nunique()
        print(f"        acct={a:6}: true FDR = {false_h*RATIO/max(1,eco_fixed)*100:4.2f}%  (false human seqs={false_h})")

# Framing B: re-threshold to a genuine true 1% under each accounting (score sweep)
# -> how many E. coli when the FDR budget is spent against the CORRECTED false set.
print("  [B] re-thresholded to true 1% FDR (score sweep):")
for a in ("raw", "crap", "diann"):
    eco, leak = eco_at_true1(a)
    print(f"        acct={a:6}: E.coli @ true 1% = {eco:5d}   (human leak: {leak})")
