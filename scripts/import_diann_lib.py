"""Import a DIA-NN predicted TARGET library (fragment-level parquet from
--gen-spec-lib --predictor) into the MuMDIA target-library schema, preserving the
real species-flagged protein identifiers (e.g. ALBU_HUMAN, ..._YEAST, ..._ECOLI)
needed for the ProteoBench species-ratio metric. Emits lib_precursors +
lib_fragments; run make_shift_decoys.py afterwards to add the decoy population.

Mapped modifications: Carbamidomethyl (UniMod:4), Oxidation (35), and the three
cysteine prenylations Farnesyl (44), GeranylGeranyl (48), Hydroxyfarnesyl (376).
Precursors carrying any other UniMod are dropped (their names are unmapped).

Usage: python import_diann_lib.py <diann_lib.parquet> <out_precursors.parquet> <out_fragments.parquet>
              [--charge-by-basic-residues]

--charge-by-basic-residues restricts the imported search space to charges a
peptide can physically carry: a precursor is kept only at charge
<= 1 (N-terminus) + (#R + #H + #K), and a b/y fragment only at charge
<= 1 (its N-terminal amine) + (basic residues within that fragment). Precursors
and fragments outside that range are dropped. Off by default (imports DIA-NN's
charges verbatim).
"""
import sys

import numpy as np
import pandas as pd
# The engine rejects `large_string` parquet columns ("column 'peptidoform' is not
# utf8"), and `to_parquet` picks the width itself: pandas 3.x chooses the large
# variant, so this helper silently emitted libraries the engine would not load.
from _lib_io import write_engine_parquet


# DIA-NN UniMod accession -> MuMDIA ProForma name. Carbamidomethyl/Oxidation are
# the standard pair; the three cysteine prenylations (Farnesyl 44, GeranylGeranyl
# 48, Hydroxyfarnesyl 376) enable a prenylation search. Each name must also exist
# in the Rust `unimod_mass` table. Replacement is substring-exact including the
# closing ")", so "(UniMod:4)" never matches inside "(UniMod:44)".
_UNIMOD_TO_PROFORMA = {
    "(UniMod:4)": "[Carbamidomethyl]",
    "(UniMod:35)": "[Oxidation]",
    "(UniMod:44)": "[Farnesyl]",
    "(UniMod:48)": "[GeranylGeranyl]",
    "(UniMod:376)": "[Hydroxyfarnesyl]",
}
# UniMod ids kept at import; any precursor carrying a mod outside this set is dropped.
_KEPT_UNIMOD_IDS = ("4", "35", "44", "48", "376")


def to_proforma(modseq):
    s = str(modseq)
    for unimod, name in _UNIMOD_TO_PROFORMA.items():
        s = s.replace(unimod, name)
    return s


def _fragment_basic_sites(seq_s, typ_s, k_s):
    """Basic-residue (R/H/K) count inside each b/y fragment's sub-sequence.

    A b-ion of ordinal k spans the first k residues of the stripped sequence; a
    y-ion of ordinal k spans the last k. Cumulative counts are cached per unique
    sequence so the whole fragment table is a single vectorized pass.
    """
    seqv = seq_s.to_numpy()
    typv = typ_s.to_numpy()
    kv = k_s.to_numpy()
    out = np.empty(len(seqv), dtype=np.int32)
    cache = {}
    for i in range(len(seqv)):
        s = seqv[i]
        cc = cache.get(s)
        if cc is None:
            arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
            is_basic = (
                (arr == ord("R")) | (arr == ord("H")) | (arr == ord("K"))
            ).astype(np.int32)
            cc = np.concatenate(([0], np.cumsum(is_basic)))  # cc[j] = basics in first j
            cache[s] = cc
        n = len(cc) - 1
        k = min(int(kv[i]), n)
        out[i] = cc[k] if typv[i] == "b" else cc[n] - cc[n - k]
    return out


def main():
    args = sys.argv[1:]
    charge_by_basic = "--charge-by-basic-residues" in args
    args = [a for a in args if not a.startswith("--")]
    inp, outp, outf = args[0:3]
    df = pd.read_parquet(inp)

    # Targets only (the re-exported speclib carries DIA-NN's own decoys).
    if "Decoy" in df.columns:
        df = df[df["Decoy"].astype(int) == 0]

    # b/y no-loss fragments only.
    lt = "Fragment.Loss.Type"
    if lt in df.columns:
        df = df[df[lt].astype(str).str.lower().isin(["noloss", "", "none", "nan"])]
    df = df[df["Fragment.Type"].astype(str).str.lower().isin(["b", "y"])]
    # drop precursors carrying any mod we do not map (keep _KEPT_UNIMOD_IDS only).
    _kept_alt = "|".join(f"{i}\\)" for i in _KEPT_UNIMOD_IDS)
    df = df[~df["Modified.Sequence"].astype(str).str.contains(
        rf"\(UniMod:(?!{_kept_alt})", regex=True)]

    # Composition-based charge restriction (opt-in). Done before candidate_id
    # assignment so dropped rows never receive an id and n_fragments stays exact.
    if charge_by_basic:
        n0_prec = df.drop_duplicates(["Modified.Sequence", "Precursor.Charge"]).shape[0]
        n0_frag = len(df)
        seq = df["Stripped.Sequence"].astype(str)
        n_basic = seq.str.count("[RHK]").astype(int)
        # precursor cap: 1 (N-terminus) + basic residues in the whole peptide.
        df = df[df["Precursor.Charge"].astype(int) <= 1 + n_basic]
        # fragment cap: 1 (fragment N-terminal amine) + basic residues in the ion.
        seq = df["Stripped.Sequence"].astype(str)
        typ = df["Fragment.Type"].astype(str).str.lower()
        k = df["Fragment.Series.Number"].astype(int)
        frag_basic = _fragment_basic_sites(seq, typ, k)
        df = df[df["Fragment.Charge"].astype(int) <= 1 + frag_basic]
        n1_prec = df.drop_duplicates(["Modified.Sequence", "Precursor.Charge"]).shape[0]
        print(
            f"charge-by-basic-residues: precursors {n0_prec} -> {n1_prec} "
            f"(dropped {n0_prec - n1_prec}), fragment rows {n0_frag} -> {len(df)} "
            f"(dropped {n0_frag - len(df)})"
        )

    df["peptidoform"] = df["Modified.Sequence"].map(to_proforma)
    df["key"] = df["peptidoform"] + "/" + df["Precursor.Charge"].astype(str)

    # Species-flagged protein string: prefer entry names (carry _HUMAN/_YEAST/_ECOLI),
    # fall back to accessions. Keep the ";"-joined multi-protein string intact so the
    # metric can drop multi-species precursors.
    prot_col = "Protein.Names" if "Protein.Names" in df.columns else "Protein.Ids"
    df["protein_str"] = df[prot_col].astype(str)

    # Sort precursors by m/z before assigning candidate_id, so the emitted library
    # is monotonic in precursor_mz (the fragment index's candidate_range assumes
    # this). The decoy builder re-sorts too, but this makes a direct target-only
    # import index-valid on its own. mergesort = stable for reproducibility.
    keys = df.drop_duplicates("key").sort_values("Precursor.Mz", kind="mergesort").reset_index(drop=True)
    keys["candidate_id"] = np.arange(len(keys), dtype=np.uint32)
    key2cand = dict(zip(keys["key"], keys["candidate_id"]))
    keys["base_peptide_id"] = pd.factorize(keys["Stripped.Sequence"])[0].astype(np.uint32)
    df["candidate_id"] = df["key"].map(key2cand).astype(np.uint32)

    nfrag = df.groupby("candidate_id").size()
    prec = pd.DataFrame({
        "candidate_id": keys["candidate_id"],
        "peptidoform_id": keys["candidate_id"].astype(np.uint32),
        "base_peptide_id": keys["base_peptide_id"],
        "peptidoform": keys["peptidoform"],
        "charge": keys["Precursor.Charge"].astype(np.int32),
        "precursor_mz": keys["Precursor.Mz"].astype(np.float64),
        "predicted_irt": keys["RT"].astype(np.float32),
        "label": "target",
        "protein": keys["protein_str"],
        "n_fragments": keys["candidate_id"].map(nfrag).fillna(0).astype(np.int32),
    })

    name = df["Fragment.Type"].astype(str) + df["Fragment.Series.Number"].astype(str)
    fc = df["Fragment.Charge"].astype(np.int32)
    name = np.where(fc > 1, name + "^" + fc.astype(str), name)
    # Fragment cardinality: how many distinct library precursors share each
    # fragment m/z (0.01 Da bin). A high value marks a non-unique, interference
    # prone ion; a low value marks a clean, quantification-friendly ion. Computed
    # once here at import time so downstream interference-aware feature/quant
    # selection reads a deterministic column instead of a runtime heuristic.
    mz_bin = (df["Product.Mz"] * 100.0).round().astype("int64")
    cardinality = df.groupby(mz_bin)["candidate_id"].transform("nunique").astype(np.int32)
    frag = pd.DataFrame({
        "candidate_id": df["candidate_id"],
        "mz": df["Product.Mz"].astype(np.float64),
        "predicted_intensity": df["Relative.Intensity"].astype(np.float32),
        "name": name,
        "ion_type": df["Fragment.Type"].astype(str).str.lower(),
        "ordinal": df["Fragment.Series.Number"].astype(np.int32),
        "frag_charge": fc,
        "cardinality": cardinality,
    }).sort_values("candidate_id").reset_index(drop=True)

    write_engine_parquet(prec, outp)
    write_engine_parquet(frag, outf)
    n_hum = prec.protein.str.contains("_HUMAN").sum()
    n_yea = prec.protein.str.contains("_YEAS").sum()
    n_eco = prec.protein.str.contains("_ECOLI").sum()
    print(f"target precursors {len(prec)} (human {n_hum}, yeast {n_yea}, ecoli {n_eco}), "
          f"fragments {len(frag)} -> {outp}, {outf}")


if __name__ == "__main__":
    main()
