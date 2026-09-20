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

The library is read one parquet row group at a time, twice: a first pass collects the
precursor table (one row per peptidoform/charge), the fragment counts and the fragment
m/z cardinality, a second pass writes the fragment table batch by batch. Peak memory is
the precursor table plus one row group, not the whole fragment table: a DIA-NN
immunopeptidomics library of 142.7M precursors and 1.69 billion fragment rows (18.7 GB)
imports in tens of GB, where reading it whole into pandas needed about a terabyte.

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
import pyarrow as pa
import pyarrow.parquet as pq
# The engine rejects `large_string` parquet columns ("column 'peptidoform' is not
# utf8"), and `to_parquet` picks the width itself: pandas 3.x chooses the large
# variant, so this helper silently emitted libraries the engine would not load.
from _lib_io import narrow_table, sort_fragments_by_candidate, write_engine_parquet


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


# Columns read from the DIA-NN table. The optional ones are used when present.
_META_COLS = ["Modified.Sequence", "Stripped.Sequence", "Precursor.Charge", "Precursor.Mz", "RT"]
_FRAG_COLS = ["Product.Mz", "Relative.Intensity", "Fragment.Type", "Fragment.Series.Number",
              "Fragment.Charge"]
_OPTIONAL_COLS = ["Decoy", "Fragment.Loss.Type", "Protein.Names", "Protein.Ids"]
# 0.01 Da bins for the fragment cardinality; wide enough for any fragment m/z.
_MZ_BIN_MAX = 1_000_000


def _filter_rows(df, charge_by_basic):
    """The importer's row filter on one batch: targets, b/y no-loss fragments, mapped
    modifications only, and the optional composition-based charge cap. Returns the
    filtered frame and the (precursor, fragment) counts the charge cap removed."""
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
    dropped = (0, 0)
    # Composition-based charge restriction (opt-in). Done before candidate_id
    # assignment so dropped rows never receive an id and n_fragments stays exact.
    if charge_by_basic and len(df):
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
        dropped = (n0_prec - n1_prec, n0_frag - len(df))
    return df, dropped


def _keys(df):
    """The precursor key the whole import is joined on: ProForma peptidoform / charge."""
    return df["Modified.Sequence"].map(to_proforma) + "/" + df["Precursor.Charge"].astype(str)


def _mz_bins(mz):
    """0.01 Da bin of each fragment m/z, computed in float32 as the DIA-NN column is stored:
    the previous implementation multiplied the float32 pandas column by 100.0 and rounded in
    float32, and a bin that rounds differently in float64 would change `cardinality` for the
    same library, which is a feature downstream."""
    b = np.round(np.asarray(mz, dtype=np.float32) * np.float32(100.0)).astype(np.int64)
    return np.clip(b, 0, _MZ_BIN_MAX - 1)


def main():
    args = sys.argv[1:]
    charge_by_basic = "--charge-by-basic-residues" in args
    args = [a for a in args if not a.startswith("--")]
    inp, outp, outf = args[0:3]

    pf = pq.ParquetFile(inp)
    present = set(pf.schema_arrow.names)
    missing = [c for c in _META_COLS + _FRAG_COLS if c not in present]
    if missing:
        raise SystemExit(f"{inp}: not a DIA-NN fragment-level library, missing columns {missing}")
    optional = [c for c in _OPTIONAL_COLS if c in present]
    prot_col = "Protein.Names" if "Protein.Names" in present else "Protein.Ids"
    read_cols = _META_COLS + _FRAG_COLS + optional

    # ---- pass 1: precursor table, fragment counts, fragment m/z cardinality ----
    # A precursor's fragments are contiguous in a DIA-NN table, so each row group
    # contributes a handful of precursors that are new and at most one that continues
    # from the previous group; the frames are concatenated in file order and deduplicated
    # keeping the first occurrence, which is exactly what a whole-table
    # `drop_duplicates` would have kept.
    prec_parts, count_parts = [], []
    card = np.zeros(_MZ_BIN_MAX, dtype=np.int64)
    dropped_prec = dropped_frag = 0
    n_frag_total = 0
    # The last precursor of a row group can continue into the next one; its (bin) set is
    # carried over so a bin seen on both sides of the boundary is counted once, exactly as a
    # whole-table `nunique` would count it.
    carry_key, carry_bins = None, set()
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=read_cols).to_pandas()
        df, (dp, dfr) = _filter_rows(df, charge_by_basic)
        dropped_prec += dp
        dropped_frag += dfr
        if not len(df):
            continue
        n_frag_total += len(df)
        key = _keys(df)
        counts = key.value_counts(sort=False)
        count_parts.append(counts)
        first = ~key.duplicated()
        part = pd.DataFrame({
            "key": key[first].to_numpy(),
            "peptidoform": df.loc[first, "Modified.Sequence"].map(to_proforma).to_numpy(),
            "Stripped.Sequence": df.loc[first, "Stripped.Sequence"].astype(str).to_numpy(),
            "Precursor.Charge": df.loc[first, "Precursor.Charge"].astype(np.int32).to_numpy(),
            "Precursor.Mz": df.loc[first, "Precursor.Mz"].astype(np.float64).to_numpy(),
            "RT": df.loc[first, "RT"].astype(np.float32).to_numpy(),
            "protein_str": df.loc[first, prot_col].astype(str).to_numpy() if prot_col in df.columns
            else np.full(int(first.sum()), "", dtype=object),
        })
        prec_parts.append(part)
        # Fragment cardinality: how many distinct library precursors share each fragment
        # m/z (0.01 Da bin). A high value marks a non-unique, interference-prone ion; a
        # low value a clean, quantification-friendly one. Computed here at import time so
        # downstream selection reads a deterministic column. One count per (precursor,
        # bin) pair: deduplicated inside the batch, which is exact except for a precursor
        # split across two row groups with a fragment in the same bin on both sides.
        pairs = pd.DataFrame({"k": key.to_numpy(), "b": _mz_bins(df["Product.Mz"])}).drop_duplicates()
        if carry_key is not None:
            seen_before = (pairs["k"] == carry_key) & pairs["b"].isin(carry_bins)
            pairs = pairs[~seen_before]
        np.add.at(card, pairs["b"].to_numpy(), 1)
        last = key.iloc[-1]
        last_bins = set(pairs.loc[pairs["k"] == last, "b"].tolist())
        if last == carry_key:
            carry_bins |= last_bins
        else:
            carry_key, carry_bins = last, last_bins
    if charge_by_basic:
        print(
            f"charge-by-basic-residues: dropped {dropped_prec} precursor(s) and "
            f"{dropped_frag} fragment row(s)"
        )
    if not prec_parts:
        raise SystemExit(f"{inp}: no target b/y fragment rows survived the filters")

    keys = pd.concat(prec_parts, ignore_index=True)
    keys = keys[~keys["key"].duplicated()].reset_index(drop=True)
    nfrag = pd.concat(count_parts).groupby(level=0).sum()

    # DIA-NN leaves the protein empty for peptides it did not map to the FASTA, the
    # Biognosys iRT-kit standards above all (LGGNEQVTR, GTFIIDPGGVIR, ...). The engine
    # refuses a library with an empty required string, because an empty protein would
    # silently merge every such peptide into one anonymous protein group. Name that
    # group explicitly instead, so the peptides stay searchable and visibly unassigned.
    unassigned = keys["protein_str"].str.strip().isin(["", "nan", "None", "<NA>"])
    if unassigned.any():
        keys.loc[unassigned, "protein_str"] = "UNASSIGNED"
        print(f"protein: {int(unassigned.sum())} precursors had no protein in {prot_col}; "
              f"written as UNASSIGNED (typically the iRT-kit standards)")

    # Sort precursors by m/z before assigning candidate_id, so the emitted library
    # is monotonic in precursor_mz (the fragment index's candidate_range assumes
    # this). The decoy builder re-sorts too, but this makes a direct target-only
    # import index-valid on its own. mergesort = stable for reproducibility.
    keys = keys.sort_values("Precursor.Mz", kind="mergesort").reset_index(drop=True)
    keys["candidate_id"] = np.arange(len(keys), dtype=np.uint32)
    keys["base_peptide_id"] = pd.factorize(keys["Stripped.Sequence"])[0].astype(np.uint32)
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
        "n_fragments": keys["key"].map(nfrag).fillna(0).astype(np.int32),
    })
    write_engine_parquet(prec, outp)
    key_index = pd.Index(keys["key"])
    del prec_parts, count_parts, nfrag

    # ---- pass 2: the fragment table, one row group in, one row group out ----
    frag_schema = pa.schema([
        pa.field("candidate_id", pa.uint32(), False),
        pa.field("mz", pa.float64(), False),
        pa.field("predicted_intensity", pa.float32(), False),
        pa.field("name", pa.string(), False),
        pa.field("ion_type", pa.string(), False),
        pa.field("ordinal", pa.int32(), False),
        pa.field("frag_charge", pa.int32(), False),
        pa.field("cardinality", pa.int32(), False),
    ])
    n_written = 0
    writer = pq.ParquetWriter(str(outf), frag_schema, compression="snappy")
    try:
        for rg in range(pf.num_row_groups):
            df = pf.read_row_group(rg, columns=read_cols).to_pandas()
            df, _ = _filter_rows(df, charge_by_basic)
            if not len(df):
                continue
            cand = key_index.get_indexer(_keys(df))
            if (cand < 0).any():
                raise RuntimeError("pass 2 met a precursor key that pass 1 did not record")
            typ = df["Fragment.Type"].astype(str).str.lower()
            ordinal = df["Fragment.Series.Number"].astype(np.int32)
            fc = df["Fragment.Charge"].astype(np.int32)
            name = df["Fragment.Type"].astype(str) + ordinal.astype(str)
            name = np.where(fc > 1, name + "^" + fc.astype(str), name)
            bins = _mz_bins(df["Product.Mz"])
            table = pa.table({
                "candidate_id": pa.array(cand.astype(np.uint32), pa.uint32()),
                "mz": pa.array(df["Product.Mz"].astype(np.float64).to_numpy(), pa.float64()),
                "predicted_intensity": pa.array(
                    df["Relative.Intensity"].astype(np.float32).to_numpy(), pa.float32()),
                "name": pa.array(name.astype(str), pa.string()),
                "ion_type": pa.array(typ.to_numpy(), pa.string()),
                "ordinal": pa.array(ordinal.to_numpy(), pa.int32()),
                "frag_charge": pa.array(fc.to_numpy(), pa.int32()),
                "cardinality": pa.array(card[bins].astype(np.int32), pa.int32()),
            }, schema=frag_schema)
            writer.write_table(narrow_table(table))
            n_written += table.num_rows
    finally:
        writer.close()
    # The engine reads one candidate-id range of this table by row-group statistics, which
    # needs the rows in candidate order; the input order above is DIA-NN's.
    sort_fragments_by_candidate(outf)
    if n_written != n_frag_total:
        raise RuntimeError(f"fragment rows: pass 1 counted {n_frag_total}, pass 2 wrote {n_written}")

    n_hum = prec.protein.str.contains("_HUMAN").sum()
    n_yea = prec.protein.str.contains("_YEAS").sum()
    n_eco = prec.protein.str.contains("_ECOLI").sum()
    print(f"target precursors {len(prec)} (human {n_hum}, yeast {n_yea}, ecoli {n_eco}), "
          f"fragments {n_written} -> {outp}, {outf}")


if __name__ == "__main__":
    main()
