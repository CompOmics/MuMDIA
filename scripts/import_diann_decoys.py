"""Import a DIA-NN predicted decoy library (fragment-level parquet from
--gen-spec-lib --predictor) into the MuMDIA target-library schema. Emits clean
peptidoforms (no DECOY_ prefix yet) with DIA-NN-predicted fragments + iRT, ready for
per-run iRT fine-tuning; combine_mutdec.py then labels them decoy and prefixes.

Usage: python import_diann_decoys.py <diann_decoy_lib.parquet> <out_precursors.parquet> <out_fragments.parquet>
"""
import sys
import numpy as np
import pandas as pd

BASE_OFF = 100_000_000  # keep decoy ids from colliding with target ids


def to_proforma(modseq):
    return (str(modseq).replace("(UniMod:4)", "[Carbamidomethyl]")
            .replace("(UniMod:35)", "[Oxidation]"))


def main():
    inp, outp, outf = sys.argv[1:4]
    df = pd.read_parquet(inp)
    # keep b/y no-loss fragments
    lt = "Fragment.Loss.Type"
    if lt in df.columns:
        df = df[df[lt].astype(str).str.lower().isin(["noloss", "", "none", "nan"])]
    df = df[df["Fragment.Type"].astype(str).str.lower().isin(["b", "y"])]
    # drop precursors with unmapped mods
    df = df[~df["Modified.Sequence"].astype(str).str.contains(r"\(UniMod:(?!4\)|35\))", regex=True)]
    df["peptidoform"] = df["Modified.Sequence"].map(to_proforma)
    df["key"] = df["peptidoform"] + "/" + df["Precursor.Charge"].astype(str)

    keys = df.drop_duplicates("key").reset_index(drop=True)
    keys["candidate_id"] = np.arange(len(keys), dtype=np.uint32)
    key2cand = dict(zip(keys["key"], keys["candidate_id"]))
    # base peptide id per stripped sequence
    keys["base_peptide_id"] = (pd.factorize(keys["Stripped.Sequence"])[0] + BASE_OFF).astype(np.uint32)
    df["candidate_id"] = df["key"].map(key2cand).astype(np.uint32)

    nfrag = df.groupby("candidate_id").size()
    prec = pd.DataFrame({
        "candidate_id": keys["candidate_id"],
        "peptidoform_id": (keys["candidate_id"].astype(np.int64) + BASE_OFF).astype(np.uint32),
        "base_peptide_id": keys["base_peptide_id"],
        "peptidoform": keys["peptidoform"],
        "charge": keys["Precursor.Charge"].astype(np.int32),
        "precursor_mz": keys["Precursor.Mz"].astype(np.float64),
        "predicted_irt": keys["RT"].astype(np.float32),
        "label": "target",  # placeholder; combine_mutdec relabels to decoy
        "protein": "DECOY",
        "n_fragments": keys["candidate_id"].map(nfrag).fillna(0).astype(np.int32),
    })
    name = df["Fragment.Type"].astype(str) + df["Fragment.Series.Number"].astype(str)
    fc = df["Fragment.Charge"].astype(np.int32)
    name = np.where(fc > 1, name + "^" + fc.astype(str), name)
    frag = pd.DataFrame({
        "candidate_id": df["candidate_id"],
        "mz": df["Product.Mz"].astype(np.float64),
        "predicted_intensity": df["Relative.Intensity"].astype(np.float32),
        "name": name,
        "ion_type": df["Fragment.Type"].astype(str).str.lower(),
        "ordinal": df["Fragment.Series.Number"].astype(np.int32),
        "frag_charge": fc,
    }).sort_values("candidate_id").reset_index(drop=True)

    prec.to_parquet(outp, index=False)
    frag.to_parquet(outf, index=False)
    print(f"decoy precursors {len(prec)}, fragments {len(frag)} -> {outp}, {outf}")


if __name__ == "__main__":
    main()
