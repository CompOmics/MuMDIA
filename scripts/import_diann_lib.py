"""Import a DIA-NN predicted TARGET library (fragment-level parquet from
--gen-spec-lib --predictor) into the MuMDIA target-library schema, preserving the
real species-flagged protein identifiers (e.g. ALBU_HUMAN, ..._YEAST, ..._ECOLI)
needed for the ProteoBench species-ratio metric. Emits lib_precursors +
lib_fragments; run make_shift_decoys.py afterwards to add the decoy population.

Proteins: `Protein.Names` is used for a precursor when every `;`-separated token is a
UniProt entry name (one word with an underscore, `ALBU_HUMAN`), else `Protein.Ids`. DIA-NN
fills `Protein.Names` only for FASTA headers it parses as UniProt; for any other header,
such as `CON_sp|P02768|ALBU_HUMAN` contaminants or the `REAL_sp|` / `ENTRAP_sp|` proteins of
an entrapment FASTA, it writes the first description word (`Keratin,`) or nothing, which
would merge distinct proteins into one group or leave them UNASSIGNED. The import prints how
many protein entries fell back to their ids.

Protein annotations: `protein_annotations.parquet` is written beside the precursor table,
one row per distinct `protein` string: its accessions (`Protein.Ids`, with the accession
field of a `db|ACCESSION|ENTRY_NAME` id), its entry names (the entry names in
`Protein.Names`, then those of such ids) and its genes (`Genes`, without description words),
each `;`-separated in first-occurrence order. The engine joins it at report time to add
`protein_ids`, `protein_names` and `genes` to the TSVs; `protein` itself, the grouping key,
is unchanged. `--annotations <path>` writes it elsewhere, `--no-annotations` not at all
(the decoy prediction's own import). A library with neither `Protein.Ids` nor
`Protein.Names` gets none.

Kept modifications, by default: Carbamidomethyl (UniMod:4), Oxidation (35), and the
three cysteine prenylations Farnesyl (44), GeranylGeranyl (48), Hydroxyfarnesyl (376).
`--keep-unimod 4,35,21,...` names the kept accessions instead; each must be in the
modification catalogue (`_unimod.py`, the engine's `modifications.json`), which also
gives the ProForma name. Precursors carrying any other UniMod are dropped. A modification
DIA-NN writes before the first residue is N-terminal (`(UniMod:1)PEPK`, protein N-terminal
acetylation): it is kept as the ProForma group `[Acetyl]-PEPK` when its accession is kept
(`--keep-unimod 1,4,35`), and dropped with the precursor otherwise, as before. The import
prints how many N-terminally modified precursors it kept and dropped.

Ion mobility: the library's 1/K0 (`IM` in a 2.x parquet, `IonMobility` in older exports)
is written as `predicted_im` (float64, V s cm^-2; precursor table version 2), but only when
at least one precursor carries a finite, positive value. Missing, non-finite and
non-positive values are null. A library without such a value, such as one predicted for an
instrument without ion mobility, gets no column and is the version-1 table.

Usage: python import_diann_lib.py <diann_lib.parquet> <out_precursors.parquet> <out_fragments.parquet>
              [--charge-by-basic-residues] [--keep-unimod <id,id,...>]
              [--annotations <out_annotations.parquet> | --no-annotations]

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
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
# The engine rejects `large_string` parquet columns ("column 'peptidoform' is not
# utf8"), and `to_parquet` picks the width itself: pandas 3.x chooses the large
# variant, so this helper silently emitted libraries the engine would not load.
from _lib_io import (annotations_path, join_unique, narrow_table, sort_fragments_by_candidate,
                    uniprot_fields, write_engine_parquet, write_protein_annotations)
from _unimod import NAME_BY_UNIMOD


# DIA-NN UniMod accession -> MuMDIA ProForma name, for every catalogue entry, so each
# name exists in the engine's mass table by construction. Replacement is substring-exact
# including the closing ")", so "(UniMod:4)" never matches inside "(UniMod:44)".
_UNIMOD_TO_PROFORMA = {f"(UniMod:{i})": f"[{name}]" for i, name in NAME_BY_UNIMOD.items()}
# UniMod ids kept at import unless `--keep-unimod` names others: the standard pair and the
# three cysteine prenylations. Any precursor carrying a mod outside the kept set is dropped.
_DEFAULT_KEPT_UNIMOD_IDS = ("4", "35", "44", "48", "376")
_KEPT_UNIMOD_IDS = _DEFAULT_KEPT_UNIMOD_IDS


def to_proforma(modseq):
    """DIA-NN modified sequence -> ProForma with catalogue names. A leading `(UniMod:N)` is
    the N-terminal group and is written `[Name]-`, so it never reads as a modification of
    the first residue (or of no residue)."""
    s = str(modseq)
    lead = ""
    if s.startswith("("):
        close = s.find(")")
        name = _UNIMOD_TO_PROFORMA.get(s[:close + 1]) if close > 0 else None
        if name is not None:
            lead = name + "-"
            s = s[close + 1:]
    for unimod, name in _UNIMOD_TO_PROFORMA.items():
        s = s.replace(unimod, name)
    return lead + s


def parse_kept_ids(value):
    """`--keep-unimod 4,35,21` -> ("4", "35", "21"), refusing an accession the catalogue
    does not name: a kept id without a ProForma name would reach the engine as
    `(UniMod:N)` text, which it cannot parse."""
    ids = tuple(x.strip() for x in str(value).split(",") if x.strip())
    if not ids:
        raise SystemExit("--keep-unimod needs at least one UniMod accession")
    unknown = [i for i in ids if not i.isdigit() or int(i) not in NAME_BY_UNIMOD]
    if unknown:
        raise SystemExit(f"--keep-unimod: not in the modification catalogue: {unknown}")
    return ids


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
_OPTIONAL_COLS = ["Decoy", "Fragment.Loss.Type", "Protein.Names", "Protein.Ids", "IM",
                  "IonMobility"]
# Read in the first pass only, for the protein annotation table.
_ANNOTATION_COLS = ["Genes"]
# 0.01 Da bins for the fragment cardinality; wide enough for any fragment m/z.
_MZ_BIN_MAX = 1_000_000


def _n_term_precursors(df):
    """Distinct (modified sequence, charge) pairs of `df` with a modification before the
    first residue."""
    seq = df["Modified.Sequence"].astype(str)
    lead = df[seq.str.startswith("(")]
    return set(zip(lead["Modified.Sequence"].astype(str), lead["Precursor.Charge"].astype(int)))


def _filter_rows(df, charge_by_basic, n_term=None):
    """The importer's row filter on one batch: targets, b/y no-loss fragments, mapped
    modifications only, and the optional composition-based charge cap. Returns the
    filtered frame and the (precursor, fragment) counts the charge cap removed.

    `n_term`, a dict with sets `seen` and `kept`, collects the N-terminally modified
    precursors before and after the modification filter, for the importer's report."""
    # Targets only (the re-exported speclib carries DIA-NN's own decoys).
    if "Decoy" in df.columns:
        df = df[df["Decoy"].astype(int) == 0]
    # b/y no-loss fragments only.
    lt = "Fragment.Loss.Type"
    if lt in df.columns:
        df = df[df[lt].astype(str).str.lower().isin(["noloss", "", "none", "nan"])]
    df = df[df["Fragment.Type"].astype(str).str.lower().isin(["b", "y"])]
    # drop precursors carrying any mod we do not map (keep _KEPT_UNIMOD_IDS only). This
    # covers an N-terminal modification too: kept when its accession is kept, and then
    # written as the ProForma N-terminal group by `to_proforma`.
    if n_term is not None:
        n_term["seen"] |= _n_term_precursors(df)
    _kept_alt = "|".join(f"{i}\\)" for i in _KEPT_UNIMOD_IDS)
    df = df[~df["Modified.Sequence"].astype(str).str.contains(
        rf"\(UniMod:(?!{_kept_alt})", regex=True)]
    if n_term is not None:
        n_term["kept"] |= _n_term_precursors(df)
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


def _is_entry_name(token):
    """True for a UniProt-style entry name (`ALBU_HUMAN`, `K2C1_HUMAN`): one word with an
    underscore and nothing a description would carry. DIA-NN writes such a name in
    `Protein.Names` only for a FASTA header it parses as UniProt; for any other header
    (`CON_sp|...` contaminants, `REAL_sp|...` / `ENTRAP_sp|...` entrapment proteins) the
    cell is empty or the first word of the description (`Keratin,`, `Uncharacterized`)."""
    return bool(token) and "_" in token and not any(c in token for c in " ,|\t")


def _resolve_protein(ids, names):
    """The protein string of one precursor: `Protein.Names` when every `;`-separated token
    is an entry name, else `Protein.Ids` (the same per-protein list, also `;`-separated).

    A description word is not an identifier: written as the protein it merges every
    contaminant named `Keratin,` into one group, and an empty name makes the precursor
    UNASSIGNED although DIA-NN mapped it. The ids keep the proteins apart and still carry
    the species flag of a `db|accession|ENTRY_SPECIES` header."""
    names = "" if names is None else str(names).strip()
    if names and names not in ("nan", "None", "<NA>") and all(
            _is_entry_name(t.strip()) for t in names.split(";")):
        return names
    ids = "" if ids is None else str(ids).strip()
    return ids if ids not in ("nan", "None", "<NA>") else ""


def _resolve_names_only(names):
    """A library without `Protein.Ids`: the names as written (there is nothing to fall
    back to), empty when the cell is."""
    names = "" if names is None else str(names).strip()
    return names if names not in ("nan", "None", "<NA>") else ""


def _protein_strings(df, cache, stats):
    """`_resolve_protein` over the rows of `df`, once per distinct (ids, names) pair.

    `cache` maps a pair to its protein string across row groups; `stats["fallback"]`
    collects the pairs whose names were replaced by the ids, for the import report."""
    has_names = "Protein.Names" in df.columns
    has_ids = "Protein.Ids" in df.columns
    if not has_names and not has_ids:
        return np.full(len(df), "", dtype=object)
    ids = df["Protein.Ids"].astype(str).to_numpy() if has_ids else np.full(len(df), "", object)
    names = (df["Protein.Names"].astype(str).to_numpy() if has_names
             else np.full(len(df), "", object))
    out = np.empty(len(df), dtype=object)
    for i, pair in enumerate(zip(ids, names)):
        got = cache.get(pair)
        if got is None:
            got = _resolve_protein(*pair) if has_ids else _resolve_names_only(pair[1])
            cache[pair] = got
            if has_names and got != pair[1].strip() and got:
                stats["fallback"].add(pair)
        out[i] = got
    return out


def _clean(cell):
    """A protein cell as a string, empty for a missing value."""
    cell = "" if cell is None else str(cell).strip()
    return "" if cell in ("nan", "None", "<NA>") else cell


def _is_gene(token):
    """A gene symbol rather than the description word DIA-NN writes in `Genes` for a
    header it does not parse as UniProt (`Keratin,`)."""
    return bool(token) and not any(c in token for c in " ,|\t")


def _annotation_of(ids, names, genes):
    """(accessions, entry names, genes) of one precursor's protein cells, as token lists."""
    id_tokens = [t.strip() for t in _clean(ids).split(";") if t.strip()]
    parsed = [uniprot_fields(t) for t in id_tokens]
    accessions = [acc for acc, _ in parsed]
    entry_names = [t.strip() for t in _clean(names).split(";") if _is_entry_name(t.strip())]
    entry_names += [name for _, name in parsed if _is_entry_name(name)]
    gene_tokens = [t.strip() for t in _clean(genes).split(";") if _is_gene(t.strip())]
    return accessions, entry_names, gene_tokens


class _Annotations:
    """The protein annotation table, built from the distinct (ids, names, genes) cells of the
    imported precursors in file order: per `protein` string, the union of their tokens in
    first-occurrence order. Two cells meet under one string when the names merge isoforms
    (`ALBU_HUMAN` for `P02768` and for `P02768;P02768-2`)."""

    def __init__(self):
        self.seen = set()
        self.rows = {}

    def add(self, df, proteins):
        cols = [df[c].astype(str).to_numpy() if c in df.columns else np.full(len(df), "", object)
                for c in ("Protein.Ids", "Protein.Names", "Genes")]
        for prot, ids, names, genes in zip(proteins, *cols):
            if not prot or (prot, ids, names, genes) in self.seen:
                continue
            self.seen.add((prot, ids, names, genes))
            acc, nm, gn = _annotation_of(ids, names, genes)
            row = self.rows.setdefault(prot, ([], [], []))
            for into, new in zip(row, (acc, nm, gn)):
                into.extend(t for t in new if t not in into)

    def table(self):
        return {p: {"protein_ids": join_unique(a), "protein_names": join_unique(n),
                    "genes": join_unique(g)}
                for p, (a, n, g) in self.rows.items()}


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
    global _KEPT_UNIMOD_IDS
    args = sys.argv[1:]
    charge_by_basic = "--charge-by-basic-residues" in args
    no_annotations = "--no-annotations" in args
    out_annotations = None
    if "--annotations" in args:
        i = args.index("--annotations")
        if i + 1 >= len(args):
            raise SystemExit("--annotations needs a path")
        out_annotations = args[i + 1]
        del args[i:i + 2]
    if "--keep-unimod" in args:
        i = args.index("--keep-unimod")
        if i + 1 >= len(args):
            raise SystemExit("--keep-unimod needs a value, e.g. --keep-unimod 4,35,21")
        _KEPT_UNIMOD_IDS = parse_kept_ids(args[i + 1])
        del args[i:i + 2]
    args = [a for a in args if not a.startswith("--")]
    inp, outp, outf = args[0:3]
    if out_annotations is None and not no_annotations:
        out_annotations = annotations_path(outp)

    pf = pq.ParquetFile(inp)
    present = set(pf.schema_arrow.names)
    missing = [c for c in _META_COLS + _FRAG_COLS if c not in present]
    if missing:
        raise SystemExit(f"{inp}: not a DIA-NN fragment-level library, missing columns {missing}")
    optional = [c for c in _OPTIONAL_COLS if c in present]
    prot_cols = [c for c in ("Protein.Names", "Protein.Ids") if c in present]
    prot_cache, prot_stats = {}, {"fallback": set()}
    im_col = next((c for c in ("IM", "IonMobility") if c in present), None)
    read_cols = _META_COLS + _FRAG_COLS + optional
    annotate = out_annotations is not None and bool(prot_cols)
    read_cols_1 = read_cols + ([c for c in _ANNOTATION_COLS if c in present] if annotate else [])
    annotations = _Annotations()

    # ---- pass 1: precursor table, fragment counts, fragment m/z cardinality ----
    # A precursor's fragments are contiguous in a DIA-NN table, so each row group
    # contributes a handful of precursors that are new and at most one that continues
    # from the previous group; the frames are concatenated in file order and deduplicated
    # keeping the first occurrence, which is exactly what a whole-table
    # `drop_duplicates` would have kept.
    prec_parts, count_parts = [], []
    card = np.zeros(_MZ_BIN_MAX, dtype=np.int64)
    dropped_prec = dropped_frag = 0
    n_term = {"seen": set(), "kept": set()}
    n_frag_total = 0
    # The last precursor of a row group can continue into the next one; its (bin) set is
    # carried over so a bin seen on both sides of the boundary is counted once, exactly as a
    # whole-table `nunique` would count it.
    carry_key, carry_bins = None, set()
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=read_cols_1).to_pandas()
        df, (dp, dfr) = _filter_rows(df, charge_by_basic, n_term)
        dropped_prec += dp
        dropped_frag += dfr
        if not len(df):
            continue
        n_frag_total += len(df)
        key = _keys(df)
        counts = key.value_counts(sort=False)
        count_parts.append(counts)
        first = ~key.duplicated()
        proteins = _protein_strings(df.loc[first], prot_cache, prot_stats)
        if annotate:
            annotations.add(df.loc[first], proteins)
        part = pd.DataFrame({
            "key": key[first].to_numpy(),
            "peptidoform": df.loc[first, "Modified.Sequence"].map(to_proforma).to_numpy(),
            "Stripped.Sequence": df.loc[first, "Stripped.Sequence"].astype(str).to_numpy(),
            "Precursor.Charge": df.loc[first, "Precursor.Charge"].astype(np.int32).to_numpy(),
            "Precursor.Mz": df.loc[first, "Precursor.Mz"].astype(np.float64).to_numpy(),
            "RT": df.loc[first, "RT"].astype(np.float32).to_numpy(),
            "IM": df.loc[first, im_col].astype(np.float64).to_numpy() if im_col
            else np.full(int(first.sum()), np.nan),
            "protein_str": proteins,
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
    if n_term["seen"]:
        n_kept = len(n_term["kept"])
        n_drop = len(n_term["seen"]) - n_kept
        print(f"N-terminal modifications: kept {n_kept} and dropped {n_drop} target "
              f"precursor(s) with a modification before the first residue (kept only when "
              f"its UniMod accession is kept: --keep-unimod)")
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

    if prot_stats["fallback"]:
        fb = prot_stats["fallback"]
        n_rows = int(keys["protein_str"].isin({_resolve_protein(*p) for p in fb}).sum())
        example = sorted(fb)[0]
        print(f"protein: {len(fb)} protein entr(ies) without a UniProt entry name in "
              f"Protein.Names, e.g. {example[1]!r} for {example[0]!r}, are written as their "
              f"Protein.Ids ({n_rows} precursors)")
    # DIA-NN leaves the protein empty for peptides it did not map to the FASTA, the
    # Biognosys iRT-kit standards above all (LGGNEQVTR, GTFIIDPGGVIR, ...). The engine
    # refuses a library with an empty required string, because an empty protein would
    # silently merge every such peptide into one anonymous protein group. Name that
    # group explicitly instead, so the peptides stay searchable and visibly unassigned.
    unassigned = keys["protein_str"].str.strip().isin(["", "nan", "None", "<NA>"])
    if unassigned.any():
        keys.loc[unassigned, "protein_str"] = "UNASSIGNED"
        src = " or ".join(prot_cols) if prot_cols else "the library (no protein column)"
        print(f"protein: {int(unassigned.sum())} precursors had no protein in {src}; "
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
    # 1/K0 only from a library that has it: written as NaN, which arrow stores as null.
    im = keys["IM"].where(np.isfinite(keys["IM"]) & (keys["IM"] > 0))
    n_im = int(im.notna().sum())
    if n_im:
        prec["predicted_im"] = im.astype(np.float64)
        print(f"ion mobility: {n_im} of {len(prec)} precursors carry a library 1/K0 (from "
              f"{im_col}); {len(prec) - n_im} are null")
    else:
        print("ion mobility: " + (f"no finite positive value in {im_col}" if im_col
                                  else "no IM column in the library")
              + "; written without predicted_im")
    write_engine_parquet(prec, outp)
    if out_annotations is not None:
        if annotate:
            n_ann = write_protein_annotations(annotations.table(), out_annotations)
            print(f"protein annotations: {n_ann} protein entr(ies) with accessions, entry names "
                  f"and genes -> {out_annotations}")
        elif os.path.isfile(out_annotations):
            # Left by another library, it would annotate this one wrongly.
            os.remove(out_annotations)
            print(f"protein annotations: none (no Protein.Ids or Protein.Names); removed the "
                  f"stale {out_annotations}")
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
