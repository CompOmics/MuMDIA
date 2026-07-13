"""Entrapment-FDR analysis. The sample is pure E. coli; the library also contains
human (entrapment) peptides. Any human identification is false by construction,
so it measures the TRUE FDR independent of decoys and of DIA-NN.

FDR_true(ecoli) ~= n_human_ids * (n_ecoli_lib / n_human_lib) / n_ecoli_ids
(false IDs are assumed to hit E. coli and human in proportion to library size).

Usage: python analyze_entrapment.py <scored.parquet> [q]
"""
import sys
import re
import pandas as pd

scored = sys.argv[1]
Q = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

# library sizes (E. coli vs human-only target peptides)
pep = pd.read_parquet("out_full/pep_ent.parquet")
is_hum = pep.protein.str.contains("_HUMAN") & ~pep.protein.str.contains("_ECOLI")
n_hum_lib = int(is_hum.sum())
n_eco_lib = int((~is_hum).sum())
ratio = n_eco_lib / n_hum_lib

def strip(pf):
    return re.sub(r"\[[^\]]*\]", "", str(pf).replace("DECOY_", ""))

s = pd.read_parquet(scored)
t = s[(s.label == "target") & (s.q_value <= Q)].copy()
prot = "protein_group" if "protein_group" in t.columns else "protein"
t["human"] = t[prot].str.contains("_HUMAN") & ~t[prot].str.contains("_ECOLI")
t["strip"] = t.peptidoform.map(strip)

eco = t[~t.human]; hum = t[t.human]
n_eco = eco.strip.nunique(); n_hum = hum.strip.nunique()
fdr = n_hum * ratio / max(1, n_eco)

# Contaminant-aware correction: real handling contaminants (keratin/albumin/Hb/
# actin/Ig) inside the spike-in proteome are genuinely present, so counting them
# as false over-estimates the FDR. Report the corrected FDR excluding them.
import re as _re
_CRAP = _re.compile(r"K[12][CH]\d|K22E|K1H\d|K2M\d|KRT\d|KRHB|ALBU|HB[ABDGE]\d?_|"
                    r"ACT[BGSC]_|TBB\d|TBA\d|TRY[PB]|IGHG|IGKC|IGLC", _re.I)
hum_noncontam = hum[~hum[prot].map(lambda p: bool(_CRAP.search(str(p))))]
n_hum_nc = hum_noncontam.strip.nunique()
fdr_nc = n_hum_nc * ratio / max(1, n_eco)

# decoy TDA FDR for comparison
d = s[(s.label == "decoy") & (s.q_value <= Q)]
tda = len(d) / max(1, len(t))

# DIA-NN concordance of the E. coli IDs
dn = pd.read_csv("out_diann/report.tsv", sep="\t")
diann = set(dn[dn["Q.Value"] <= 0.01]["Stripped.Sequence"].unique())
ov = len(set(eco.strip) & diann)

print(f"library: ecoli_lib={n_eco_lib} human_lib={n_hum_lib} ratio(eco/hum)={ratio:.3f}")
print(f"@q<={Q}:  ecoli_ids={n_eco}  human(entrapment)_ids={n_hum}")
print(f"  ENTRAPMENT true FDR (ecoli) ~= {fdr*100:.1f}%   [reported TDA decoy FDR = {tda*100:.2f}%]")
print(f"  contaminant-corrected true FDR (cRAP excluded, false human seqs {n_hum}->{n_hum_nc}) ~= {fdr_nc*100:.1f}%")
print(f"  ecoli IDs in DIA-NN: {ov} ({ov/max(1,n_eco)*100:.1f}% concordance) | ecoli_ids as %of DIA-NN = {n_eco/len(diann)*100:.1f}%")
