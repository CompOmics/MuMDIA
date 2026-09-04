"""Contract tests for `scripts/import_diann_lib.py` (offline DIA-NN recipe).

Needs only pandas and pyarrow, so it always runs. The output of this script is
fed straight into `index.rs load()`, which enforces two hard preconditions and
bails naming the offending row: `candidate_id` must be the contiguous
row-aligned range `0..ncand` (`index.rs:112-125`) and precursors must ascend by
`precursor_mz` (`index.rs:215-231`). Because the importer is run once, offline,
by hand, a violation surfaces only when a search is launched hours later.

The modification map is the other risk. Replacement is substring-exact
including the closing parenthesis so that `(UniMod:4)` never matches inside
`(UniMod:44)`; a regression there would rewrite a Farnesyl site as
Carbamidomethyl and quietly search the wrong mass.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import (
    assert_engine_readable_parquet,
    assert_library_load_invariants,
    assert_string_columns_utf8,
    read_columns,
    run_worker_ok,
)

# (Modified.Sequence, charge, Precursor.Mz, RT, protein, decoy, loss_type, ion_types)
# Precursor m/z are deliberately out of order in the input, so a passing
# ordering assertion proves the importer sorts rather than inherits the order.
PRECURSORS = [
    ("PEPTIDEK", 2, 500.5, 20.0, "ALBU_HUMAN", 0, "noloss", ("y", "b", "y")),
    ("PEPTIDEK", 3, 334.0, 20.0, "ALBU_HUMAN", 0, "noloss", ("y", "b", "y")),
    ("SC(UniMod:4)ENICK", 2, 420.2, 30.0, "ENO1_YEAST", 0, "noloss", ("y", "b", "y")),
    ("M(UniMod:35)ELLOWK", 2, 480.1, 15.0, "GLYK_ECOLI", 0, "noloss", ("y", "b", "y")),
    ("K(UniMod:44)ITTENR", 2, 460.3, 25.0, "PRE1_HUMAN", 0, "noloss", ("y", "b", "y")),
    # dropped: UniMod:21 is not in the mapped set, so its ProForma name is unknown
    ("DROPME(UniMod:21)K", 2, 600.0, 40.0, "XXX_HUMAN", 0, "noloss", ("y", "b", "y")),
    # dropped: DIA-NN's own decoy half
    ("DIANNDECOYK", 2, 700.0, 50.0, "YYY_HUMAN", 1, "noloss", ("y", "b", "y")),
    # dropped: every fragment carries a neutral loss
    ("LOSSYPEPK", 2, 380.0, 12.0, "ZZZ_HUMAN", 0, "H3PO4", ("y", "b", "y")),
    # dropped: no b/y fragment at all
    ("AIONPEPK", 2, 390.0, 13.0, "WWW_HUMAN", 0, "noloss", ("a", "a", "a")),
]
FRAGMENT_SHAPE = ((3, 1), (2, 1), (5, 2))  # (Fragment.Series.Number, Fragment.Charge)

KEPT_KEYS = [
    ("M[Oxidation]ELLOWK", 2, 480.1),
    ("SC[Carbamidomethyl]ENICK", 2, 420.2),
    ("K[Farnesyl]ITTENR", 2, 460.3),
    ("PEPTIDEK", 3, 334.0),
    ("PEPTIDEK", 2, 500.5),
]


def _strip_unimod(modseq):
    out = []
    depth = 0
    for ch in modseq:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


@pytest.fixture(scope="module")
def diann_lib(tmp_path_factory):
    """Write a tiny fragment-level DIA-NN speclib in the columns the importer reads."""
    rows = {k: [] for k in (
        "Decoy", "Fragment.Loss.Type", "Fragment.Type", "Modified.Sequence",
        "Precursor.Charge", "Precursor.Mz", "RT", "Stripped.Sequence",
        "Product.Mz", "Relative.Intensity", "Fragment.Series.Number",
        "Fragment.Charge", "Protein.Names")}
    product = 200.0
    for modseq, z, mz, rt, prot, decoy, loss, ions in PRECURSORS:
        for ion, (series, fz) in zip(ions, FRAGMENT_SHAPE):
            product += 1.5
            rows["Decoy"].append(decoy)
            rows["Fragment.Loss.Type"].append(loss)
            rows["Fragment.Type"].append(ion)
            rows["Modified.Sequence"].append(modseq)
            rows["Precursor.Charge"].append(z)
            rows["Precursor.Mz"].append(mz)
            rows["RT"].append(rt)
            rows["Stripped.Sequence"].append(_strip_unimod(modseq))
            rows["Product.Mz"].append(product)
            rows["Relative.Intensity"].append(0.5)
            rows["Fragment.Series.Number"].append(series)
            rows["Fragment.Charge"].append(fz)
            rows["Protein.Names"].append(prot)
    path = tmp_path_factory.mktemp("diann") / "diann_lib.parquet"
    pq.write_table(
        pa.table({
            "Decoy": pa.array(rows["Decoy"], pa.int32()),
            "Fragment.Loss.Type": pa.array(rows["Fragment.Loss.Type"], pa.string()),
            "Fragment.Type": pa.array(rows["Fragment.Type"], pa.string()),
            "Modified.Sequence": pa.array(rows["Modified.Sequence"], pa.string()),
            "Precursor.Charge": pa.array(rows["Precursor.Charge"], pa.int32()),
            "Precursor.Mz": pa.array(rows["Precursor.Mz"], pa.float64()),
            "RT": pa.array(rows["RT"], pa.float64()),
            "Stripped.Sequence": pa.array(rows["Stripped.Sequence"], pa.string()),
            "Product.Mz": pa.array(rows["Product.Mz"], pa.float64()),
            "Relative.Intensity": pa.array(rows["Relative.Intensity"], pa.float64()),
            "Fragment.Series.Number": pa.array(rows["Fragment.Series.Number"], pa.int32()),
            "Fragment.Charge": pa.array(rows["Fragment.Charge"], pa.int32()),
            "Protein.Names": pa.array(rows["Protein.Names"], pa.string()),
        }),
        str(path), compression="snappy",
    )
    return path


@pytest.fixture(scope="module")
def imported(diann_lib, tmp_path_factory):
    d = tmp_path_factory.mktemp("imported")
    prec = d / "lib_precursors.parquet"
    frag = d / "lib_fragments.parquet"
    run_worker_ok("import_diann_lib.py", diann_lib, prec, frag)
    return {"prec": prec, "frag": frag}


def test_precursor_table_satisfies_the_index_preconditions(imported):
    """Contiguous `candidate_id` and ascending `precursor_mz`, or the index bails.

    Both are hard errors in `index.rs load()`, and the m/z ordering is load
    bearing beyond the check: the fragment index's `partition_point` search over
    `prec_mz` assumes it, so an unsorted import that slipped past would return
    the wrong candidate window for every spectrum.
    """
    assert_library_load_invariants(
        imported["prec"], imported["frag"], require_both_labels=False
    )
    cols = read_columns(imported["prec"])
    assert len(cols["candidate_id"]) == len(KEPT_KEYS)
    assert set(cols["label"]) == {"target"}, (
        "the importer must emit targets only; the decoy population comes from a "
        "decoy builder, which needs an unpolluted target half"
    )


def test_only_mapped_targets_survive_the_import(imported):
    """DIA-NN decoys, unmapped modifications and loss/non-by fragments are dropped.

    Keeping DIA-NN's own decoys would put two unrelated decoy populations in one
    library and break target/decoy exchangeability. Keeping an unmapped UniMod
    would emit a ProForma name the Rust `unimod_mass` table cannot resolve, and
    a neutral-loss or a-ion fragment would be matched at a mass the engine
    never predicts.
    """
    cols = read_columns(imported["prec"])
    got = sorted(zip(cols["peptidoform"],
                     (int(z) for z in cols["charge"]),
                     (round(float(m), 4) for m in cols["precursor_mz"])))
    assert got == sorted((p, z, round(m, 4)) for p, z, m in KEPT_KEYS)


def test_unimod_replacement_is_substring_exact(imported):
    """`(UniMod:4)` must not match inside `(UniMod:44)`.

    Replacement includes the closing parenthesis for exactly this reason. If it
    did not, the Farnesyl site would be rewritten as Carbamidomethyl and the
    search would look for a peptide 96 Da lighter than the one in the library,
    with nothing anywhere reporting a problem.
    """
    pforms = set(read_columns(imported["prec"])["peptidoform"])
    assert "K[Farnesyl]ITTENR" in pforms
    assert "K[Carbamidomethyl]4)ITTENR" not in pforms
    assert not any("UniMod" in p for p in pforms), (
        "a UniMod accession survived into the peptidoform string"
    )


def test_base_peptide_id_groups_charges_of_one_stripped_sequence(imported):
    """The two PEPTIDEK charges must share one `base_peptide_id`.

    Peptide-level q estimation performs picked target-decoy competition through
    `base_peptide_id`, and protein Top-N counts unique `base_peptide_id` values.
    If charges of one peptide got distinct ids, the same peptide would be
    counted twice at the peptide and protein level.
    """
    cols = read_columns(imported["prec"])
    by_pform = {}
    for pform, base in zip(cols["peptidoform"], cols["base_peptide_id"]):
        by_pform.setdefault(pform, set()).add(int(base))
    assert len(by_pform["PEPTIDEK"]) == 1
    assert len(set(int(b) for b in cols["base_peptide_id"])) == 4, (
        "expected four distinct stripped sequences among the imported precursors"
    )


def test_fragment_names_and_counts_match_the_precursor_table(imported):
    """`name`, `ion_type` and `n_fragments` must describe the emitted fragments.

    `n_fragments` is read as the per-precursor fragment count; if it disagrees
    with the fragment table the extraction stage sizes its evidence arrays from
    a number that does not exist. The `^<z>` suffix only for charge > 1 is what
    keeps a charge-2 ion distinguishable from its charge-1 sibling in
    quantification and in the fragment-consensus guard.
    """
    prec = read_columns(imported["prec"])
    frag = read_columns(imported["frag"])
    counts = {}
    for cid in frag["candidate_id"]:
        counts[int(cid)] = counts.get(int(cid), 0) + 1
    for cid, n in zip(prec["candidate_id"], prec["n_fragments"]):
        assert counts[int(cid)] == int(n)
    assert sorted(set(frag["name"])) == ["b2", "y3", "y5^2"]
    assert set(frag["ion_type"]) == {"b", "y"}


def test_protein_names_keep_their_species_flags(imported):
    """`_HUMAN` / `_YEAST` / `_ECOLI` entry names must survive the import.

    The ProteoBench species-ratio metric is computed from these strings. Falling
    back to bare accessions silently makes every benchmark ratio unassignable.
    """
    proteins = set(read_columns(imported["prec"])["protein"])
    assert any(p.endswith("_HUMAN") for p in proteins)
    assert any(p.endswith("_YEAST") for p in proteins)
    assert any(p.endswith("_ECOLI") for p in proteins)


def test_output_parquet_uses_only_a_codec_the_engine_can_decode(imported):
    """Snappy (or uncompressed), or the engine cannot open the library at all.

    The engine's parquet dependency is built with `features = ["arrow","snap"]`,
    so any other codec fails at read with "Disabled feature at compile time:
    zstd" and the library is unusable rather than merely slow.
    """
    for path in (imported["prec"], imported["frag"]):
        assert_engine_readable_parquet(path)


def test_output_parquet_string_columns_are_utf8(imported):
    """String columns must be arrow `utf8`, never `large_utf8`.

    `Table::str` downcasts to `StringArray` only, so a 64-bit-offset column
    makes the engine reject the library with "column 'peptidoform' is not
    utf8". The importer relies on the pandas default rather than casting to
    `pa.string()` explicitly, so the guarantee holds only for a pandas whose
    string dtype is the numpy object dtype.
    """
    for path in (imported["prec"], imported["frag"]):
        assert_string_columns_utf8(path)


def test_charge_by_basic_residues_drops_unsupportable_charges(diann_lib, tmp_path):
    """The opt-in charge restriction must drop rows before ids are assigned.

    PEPTIDEK carries one basic residue, so charge 3 exceeds
    `1 + (#R + #H + #K)`. The drop has to happen before `candidate_id` is
    assigned, otherwise the ids are no longer the contiguous range the index
    requires and `n_fragments` counts rows that were removed.
    """
    prec = tmp_path / "prec.parquet"
    frag = tmp_path / "frag.parquet"
    stdout, _ = run_worker_ok(
        "import_diann_lib.py", diann_lib, prec, frag, "--charge-by-basic-residues"
    )
    assert "charge-by-basic-residues" in stdout
    assert_library_load_invariants(prec, frag, require_both_labels=False)
    cols = read_columns(prec)
    kept = set(zip(cols["peptidoform"], (int(z) for z in cols["charge"])))
    assert ("PEPTIDEK", 3) not in kept
    assert ("PEPTIDEK", 2) in kept
    assert len(cols["candidate_id"]) == len(KEPT_KEYS) - 1
    mz = np.asarray(cols["precursor_mz"], dtype=float)
    assert np.all(np.diff(mz) >= 0.0)


def test_imported_library_feeds_the_documented_decoy_builder_step(imported, tmp_path):
    """The recipe must compose: importer output straight into a decoy builder.

    The importer emits targets only, so the library is unusable for FDR until a
    decoy builder runs on it (`docs/13_sidecars.md`, "DIA-NN recipe"). This is
    the composition an operator actually performs; a column the importer adds
    but the builder does not carry through would break it only at that second
    step, after the slow import has already succeeded.
    """
    prec = tmp_path / "lib_precursors.parquet"
    frag = tmp_path / "lib_fragments.parquet"
    run_worker_ok(
        "make_shift_decoys.py", imported["prec"], imported["frag"], prec, frag
    )
    p, f = assert_library_load_invariants(prec, frag, require_both_labels=True)
    assert p.num_rows == 2 * len(KEPT_KEYS)
    assert f.num_rows == 2 * pq.read_table(str(imported["frag"])).num_rows
    assert "cardinality" in read_columns(frag), (
        "the importer's cardinality column did not survive the decoy builder"
    )
    assert set(read_columns(prec)["label"]) == {"target", "decoy"}
