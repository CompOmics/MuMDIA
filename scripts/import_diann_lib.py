"""Import a DIA-NN predicted TARGET library (fragment-level parquet from
--gen-spec-lib --predictor) into the MuMDIA target-library schema, preserving the
real species-flagged protein identifiers (e.g. ALBU_HUMAN, ..._YEAST, ..._ECOLI)
needed for the ProteoBench species-ratio metric. Emits lib_precursors +
lib_fragments; run make_shift_decoys.py afterwards to add the decoy population.

Usage: python import_diann_lib.py <diann_lib.parquet> <out_precursors.parquet> <out_fragments.parquet>
"""
import sys

import numpy as np
import pandas as pd


def to_proforma(modseq):
    return (
        str(modseq)
        .replace("(UniMod:4)", "[Carbamidomethyl]")
        .replace("(UniMod:35)", "[Oxidation]")
    )


def main():
    inp, outp, outf = sys.argv[1:4]
    df = pd.read_parquet(inp)

    # Targets only (the re-exported speclib carries DIA-NN's own decoys).
    if "Decoy" in df.columns:
        df = df[df["Decoy"].astype(int) == 0]

    # b/y no-loss fragments only.
    lt = "Fragment.Loss.Type"
    if lt in df.columns:
        df = df[df[lt].astype(str).str.lower().isin(["noloss", "", "none", "nan"])]
    df = df[df["Fragment.Type"].astype(str).str.lower().isin(["b", "y"])]
    # drop precursors carrying mods we do not map (only Carbamidomethyl/Oxidation).
    df = df[~df["Modified.Sequence"].astype(str).str.contains(r"\(UniMod:(?!4\)|35\))", regex=True)]

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

    prec.to_parquet(outp, index=False)
    frag.to_parquet(outf, index=False)
    n_hum = prec.protein.str.contains("_HUMAN").sum()
    n_yea = prec.protein.str.contains("_YEAS").sum()
    n_eco = prec.protein.str.contains("_ECOLI").sum()
    print(f"target precursors {len(prec)} (human {n_hum}, yeast {n_yea}, ecoli {n_eco}), "
          f"fragments {len(frag)} -> {outp}, {outf}")


if __name__ == "__main__":
    main()
