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
    # kept: a modification before the first residue is N-terminal, written as the
    # ProForma group `[Name]-` when its accession is kept
    ("(UniMod:4)NTERMCK", 2, 610.0, 41.0, "NTM_HUMAN", 0, "noloss", ("y", "b", "y")),
    # dropped by default, kept with --keep-unimod 1: protein N-terminal acetylation
    ("(UniMod:1)SACETYLK", 2, 620.0, 42.0, "NTA_HUMAN", 0, "noloss", ("y", "b", "y")),
    # dropped: DIA-NN's own decoy half
    ("DIANNDECOYK", 2, 700.0, 50.0, "YYY_HUMAN", 1, "noloss", ("y", "b", "y")),
    # dropped: every fragment carries a neutral loss
    ("LOSSYPEPK", 2, 380.0, 12.0, "ZZZ_HUMAN", 0, "H3PO4", ("y", "b", "y")),
    # dropped: no b/y fragment at all
    ("AIONPEPK", 2, 390.0, 13.0, "WWW_HUMAN", 0, "noloss", ("a", "a", "a")),
]
FRAGMENT_SHAPE = ((3, 1), (2, 1), (5, 2))  # (Fragment.Series.Number, Fragment.Charge)

KEPT_KEYS = [
    ("[Carbamidomethyl]-NTERMCK", 2, 610.0),
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
    assert len(set(int(b) for b in cols["base_peptide_id"])) == 5, (
        "expected five distinct stripped sequences among the imported precursors"
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


def test_an_empty_protein_is_written_as_unassigned(diann_lib, tmp_path):
    """DIA-NN leaves `Protein.Names` empty for peptides it did not map, the iRT-kit
    standards above all. The engine refuses a library with an empty required string
    (`index.rs load()`), so the importer must name that group rather than pass the
    empty cell through; otherwise every DIA-NN library with the standards in it fails
    to load."""
    src = pq.read_table(diann_lib)
    modseq = src.column("Modified.Sequence").to_pylist()
    charge = src.column("Precursor.Charge").to_pylist()
    prot = src.column("Protein.Names").to_pylist()
    blanked = [
        "" if (m == "PEPTIDEK" and z == 3) else p for m, z, p in zip(modseq, charge, prot)
    ]
    assert blanked != prot, "the fixture must contain the precursor being blanked"
    i = src.schema.get_field_index("Protein.Names")
    lib = tmp_path / "diann_lib_blank.parquet"
    pq.write_table(src.set_column(i, "Protein.Names", pa.array(blanked, pa.string())), str(lib),
                   compression="snappy")
    prec = tmp_path / "lib_precursors.parquet"
    frag = tmp_path / "lib_fragments.parquet"
    run_worker_ok("import_diann_lib.py", lib, prec, frag)
    cols = read_columns(prec)
    by_key = {(p, z): pr for p, z, pr in zip(cols["peptidoform"], cols["charge"], cols["protein"])}
    assert by_key[("PEPTIDEK", 3)] == "UNASSIGNED"
    assert by_key[("PEPTIDEK", 2)] == "ALBU_HUMAN", "only the blanked precursor changes"
    assert all(pr.strip() for pr in cols["protein"]), "no empty protein reaches the engine"


def _with_protein_columns(diann_lib, tmp_path, ids_by_key, names_by_key):
    """The fixture library with `Protein.Ids` added and `Protein.Names` replaced, both
    per (modified sequence, charge); precursors not named keep their fixture name as both."""
    src = pq.read_table(diann_lib)
    modseq = src.column("Modified.Sequence").to_pylist()
    charge = src.column("Precursor.Charge").to_pylist()
    prot = src.column("Protein.Names").to_pylist()
    ids = [ids_by_key.get((m, z), p) for m, z, p in zip(modseq, charge, prot)]
    names = [names_by_key.get((m, z), p) for m, z, p in zip(modseq, charge, prot)]
    i = src.schema.get_field_index("Protein.Names")
    table = src.set_column(i, "Protein.Names", pa.array(names, pa.string()))
    table = table.append_column("Protein.Ids", pa.array(ids, pa.string()))
    lib = tmp_path / "diann_lib_ids.parquet"
    pq.write_table(table, str(lib), compression="snappy")
    return lib


def test_a_description_word_in_protein_names_falls_back_to_the_ids(diann_lib, tmp_path):
    """DIA-NN writes a UniProt entry name in `Protein.Names` only for a header it parses
    as UniProt. For `CON_sp|...` contaminants it writes the first description word, so two
    different keratins both become `Keratin,` and merge into one protein group, and for the
    `REAL_sp|...` / `ENTRAP_sp|...` headers of an entrapment FASTA it writes nothing, which
    made a mapped precursor UNASSIGNED. Those rows must take `Protein.Ids`; rows whose names
    are entry names keep them, species flag and all."""
    lib = _with_protein_columns(
        diann_lib, tmp_path,
        ids_by_key={
            ("PEPTIDEK", 2): "P02768",
            ("PEPTIDEK", 3): "CON_sp|P02444|KRA33_SHEEP;CON_sp|P02445|KRA34_SHEEP",
            ("SC(UniMod:4)ENICK", 2): "CON_sp|P35908|K22E_HUMAN",
            ("M(UniMod:35)ELLOWK", 2): "REAL_sp|P0A6P9|ENO_ECOLI",
            ("K(UniMod:44)ITTENR", 2): "ENTRAP_sp|P0CG40|SP9_HUMAN",
            ("(UniMod:4)NTERMCK", 2): "CON_sp|P12081|HARS1_HUMAN;P49590",
        },
        names_by_key={
            ("PEPTIDEK", 2): "ALBU_HUMAN",
            ("PEPTIDEK", 3): "Keratin,",
            ("SC(UniMod:4)ENICK", 2): "Keratin,",
            ("M(UniMod:35)ELLOWK", 2): "Enolase",
            ("K(UniMod:44)ITTENR", 2): "",
            ("(UniMod:4)NTERMCK", 2): "Histidine--tRNA;SYHM_HUMAN",
        },
    )
    prec = tmp_path / "lib_precursors.parquet"
    frag = tmp_path / "lib_fragments.parquet"
    out, _ = run_worker_ok("import_diann_lib.py", lib, prec, frag)
    cols = read_columns(prec)
    by_key = {(p, int(z)): pr for p, z, pr in zip(cols["peptidoform"], cols["charge"],
                                                  cols["protein"])}
    assert by_key[("PEPTIDEK", 2)] == "ALBU_HUMAN", "a valid entry name is kept"
    assert by_key[("PEPTIDEK", 3)] == "CON_sp|P02444|KRA33_SHEEP;CON_sp|P02445|KRA34_SHEEP"
    assert by_key[("SC[Carbamidomethyl]ENICK", 2)] == "CON_sp|P35908|K22E_HUMAN", (
        "two different keratins must not share the group `Keratin,`"
    )
    assert by_key[("M[Oxidation]ELLOWK", 2)] == "REAL_sp|P0A6P9|ENO_ECOLI"
    assert by_key[("K[Farnesyl]ITTENR", 2)] == "ENTRAP_sp|P0CG40|SP9_HUMAN", (
        "an empty name with a mapped id is not UNASSIGNED"
    )
    assert by_key[("[Carbamidomethyl]-NTERMCK", 2)] == "CON_sp|P12081|HARS1_HUMAN;P49590", (
        "one description token among entry names takes the whole id list, keeping the "
        "per-protein alignment"
    )
    assert "UNASSIGNED" not in set(cols["protein"])
    assert not any(" " in p or "," in p for p in cols["protein"])
    assert "written as their Protein.Ids" in out


def test_empty_names_and_ids_are_unassigned(diann_lib, tmp_path):
    """With nothing in either column the precursor is UNASSIGNED, as before."""
    lib = _with_protein_columns(
        diann_lib, tmp_path,
        ids_by_key={("PEPTIDEK", 3): ""}, names_by_key={("PEPTIDEK", 3): ""},
    )
    prec = tmp_path / "lib_precursors.parquet"
    frag = tmp_path / "lib_fragments.parquet"
    run_worker_ok("import_diann_lib.py", lib, prec, frag)
    cols = read_columns(prec)
    by_key = {(p, int(z)): pr for p, z, pr in zip(cols["peptidoform"], cols["charge"],
                                                  cols["protein"])}
    assert by_key[("PEPTIDEK", 3)] == "UNASSIGNED"
    assert by_key[("PEPTIDEK", 2)] == "ALBU_HUMAN"


def test_output_parquet_uses_only_a_codec_the_engine_can_decode(imported):
    """Snappy, zstd or uncompressed, or the engine cannot open the library at all.

    The engine's parquet dependency is built with `features = ["arrow", "snap",
    "zstd"]`, so any other codec fails at read with "Disabled feature at compile
    time: <codec>" and the library is unusable rather than merely slow.
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


def test_keep_unimod_names_the_kept_accessions(diann_lib, tmp_path):
    """`--keep-unimod` keeps the named catalogue accessions with their ProForma names (the
    desktop passes the modifications the user selected); an N-terminal row goes with its
    accession."""
    prec = tmp_path / "p.parquet"
    frag = tmp_path / "f.parquet"
    out, _ = run_worker_ok("import_diann_lib.py", diann_lib, prec, frag,
                           "--keep-unimod", "35,44,21")
    assert_library_load_invariants(prec, frag, require_both_labels=False)
    got = set(read_columns(prec)["peptidoform"])
    assert "DROPME[Phospho]K" in got
    assert not any("NTERM" in g or "ACETYL" in g for g in got)
    assert not any("UniMod" in g for g in got)
    assert "kept 0 and dropped 2 target precursor(s)" in out, out


def test_an_n_terminal_acetyl_is_kept_as_the_proforma_group(diann_lib, tmp_path):
    """`(UniMod:1)SACETYLK` is protein N-terminal acetylation: kept as `[Acetyl]-SACETYLK`
    when accession 1 is kept, on the stripped sequence's own peptide id, and the import
    says how many N-terminal precursors it kept and dropped."""
    prec = tmp_path / "p.parquet"
    frag = tmp_path / "f.parquet"
    out, _ = run_worker_ok("import_diann_lib.py", diann_lib, prec, frag, "--keep-unimod", "1,4,35")
    assert_library_load_invariants(prec, frag, require_both_labels=False)
    cols = read_columns(prec)
    got = set(cols["peptidoform"])
    assert "[Acetyl]-SACETYLK" in got
    assert "[Carbamidomethyl]-NTERMCK" in got
    assert not any(g.startswith("[Acetyl]S") or "UniMod" in g for g in got)
    assert "kept 2 and dropped 0 target precursor(s)" in out, out


def test_by_default_an_n_terminal_acetyl_is_dropped_and_counted(diann_lib, tmp_path):
    prec = tmp_path / "p.parquet"
    frag = tmp_path / "f.parquet"
    out, _ = run_worker_ok("import_diann_lib.py", diann_lib, prec, frag)
    assert not any("ACETYL" in g for g in read_columns(prec)["peptidoform"])
    assert "kept 1 and dropped 1 target precursor(s)" in out, out


def test_keep_unimod_refuses_an_accession_outside_the_catalogue(diann_lib, tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "import_diann_lib.py"
    r = subprocess.run(
        [sys.executable, str(script), str(diann_lib), str(tmp_path / "p.parquet"),
         str(tmp_path / "f.parquet"), "--keep-unimod", "4,99999"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "not in the modification catalogue" in (r.stderr + r.stdout)


def _with_column(diann_lib, tmp_path, name, value_of):
    """The fixture library with an ion-mobility column `name`, valued per precursor."""
    src = pq.read_table(diann_lib)
    keys = zip(src.column("Modified.Sequence").to_pylist(),
               src.column("Precursor.Charge").to_pylist())
    values = [value_of(m, z) for m, z in keys]
    lib = tmp_path / f"diann_lib_{name}.parquet"
    pq.write_table(src.append_column(name, pa.array(values, pa.float64())), str(lib),
                   compression="snappy")
    prec = tmp_path / f"{name}_precursors.parquet"
    frag = tmp_path / f"{name}_fragments.parquet"
    stdout, _ = run_worker_ok("import_diann_lib.py", lib, prec, frag)
    return prec, frag, stdout


def test_a_library_without_ion_mobility_has_no_predicted_im_column(imported):
    """No IM column in the source: the version-1 table, with no always-null column."""
    assert "predicted_im" not in pq.read_schema(str(imported["prec"])).names


@pytest.mark.parametrize("name", ["IM", "IonMobility"])
def test_library_ion_mobility_is_written_as_predicted_im(diann_lib, tmp_path, name):
    """A finite positive 1/K0 is kept per precursor; anything else becomes null."""
    def value_of(modseq, charge):
        if modseq == "PEPTIDEK":
            return 0.80 + 0.1 * charge
        if modseq.startswith("SC"):
            return float("nan")
        if modseq.startswith("M"):
            return 0.0
        return 1.05

    prec, frag, stdout = _with_column(diann_lib, tmp_path, name, value_of)
    assert_library_load_invariants(prec, frag, require_both_labels=False,
                                   check_string_type=True)
    field = pq.read_schema(str(prec)).field("predicted_im")
    assert field.type == pa.float64() and field.nullable
    cols = pq.read_table(str(prec)).to_pydict()
    got = {(p, z): v for p, z, v in zip(cols["peptidoform"], cols["charge"], cols["predicted_im"])}
    assert got[("PEPTIDEK", 2)] == pytest.approx(1.0)
    assert got[("PEPTIDEK", 3)] == pytest.approx(1.1)
    assert got[("K[Farnesyl]ITTENR", 2)] == pytest.approx(1.05)
    assert got[("SC[Carbamidomethyl]ENICK", 2)] is None, "non-finite is null"
    assert got[("M[Oxidation]ELLOWK", 2)] is None, "non-positive is null"
    assert f"4 of 6 precursors carry a library 1/K0 (from {name}); 2 are null" in stdout


def test_an_all_zero_ion_mobility_column_is_not_written(diann_lib, tmp_path):
    """A library predicted without ion mobility (zeros) stays the version-1 table."""
    prec, _, stdout = _with_column(diann_lib, tmp_path, "IM", lambda m, z: 0.0)
    assert "predicted_im" not in pq.read_schema(str(prec)).names
    assert "no finite positive value in IM" in stdout


# ---------------------------------------------------------------- protein annotations


def _with_annotation_columns(diann_lib, tmp_path, cells_by_key):
    """The fixture library with `Protein.Ids`, `Protein.Names` and `Genes` set per (modified
    sequence, charge) from `cells_by_key` (ids, names, genes); precursors not named keep
    their fixture name as names and ids, and no gene."""
    src = pq.read_table(diann_lib)
    modseq = src.column("Modified.Sequence").to_pylist()
    charge = src.column("Precursor.Charge").to_pylist()
    prot = src.column("Protein.Names").to_pylist()
    cells = [cells_by_key.get((m, z), (p, p, "")) for m, z, p in zip(modseq, charge, prot)]
    i = src.schema.get_field_index("Protein.Names")
    table = src.set_column(i, "Protein.Names", pa.array([c[1] for c in cells], pa.string()))
    table = table.append_column("Protein.Ids", pa.array([c[0] for c in cells], pa.string()))
    table = table.append_column("Genes", pa.array([c[2] for c in cells], pa.string()))
    lib = tmp_path / "diann_lib_annotated.parquet"
    pq.write_table(table, str(lib), compression="snappy")
    return lib


ANNOTATED = {
    # one protein, aligned
    ("PEPTIDEK", 2): ("P02768", "ALBU_HUMAN", "ALB"),
    # a shared peptide: two proteins, one gene
    ("PEPTIDEK", 3): ("P02768;P02769", "ALBU_HUMAN;ALBU_BOVIN", "ALB"),
    # no gene
    ("SC(UniMod:4)ENICK", 2): ("P06733", "ENO1_YEAST", ""),
    # contaminants: a description word in Protein.Names and Genes, the ids carry the fields
    ("M(UniMod:35)ELLOWK", 2): ("CON_sp|P02444|KRA33_SHEEP;CON_sp|P02445|KRA34_SHEEP",
                                "Keratin,", "Keratin,"),
    # two precursors under one entry name with different isoform ids
    ("K(UniMod:44)ITTENR", 2): ("Q15363;Q15363-2", "TMED2_HUMAN", "TMED2"),
    ("(UniMod:4)NTERMCK", 2): ("Q15363", "TMED2_HUMAN", "TMED2"),
}


@pytest.fixture(scope="module")
def annotated(diann_lib, tmp_path_factory):
    d = tmp_path_factory.mktemp("annotated")
    lib = _with_annotation_columns(diann_lib, d, ANNOTATED)
    prec, frag = d / "lib_precursors.parquet", d / "lib_fragments.parquet"
    out, _ = run_worker_ok("import_diann_lib.py", lib, prec, frag)
    return {"dir": d, "lib": lib, "prec": prec, "frag": frag, "stdout": out}


def _annotation_rows(path):
    t = read_columns(path)
    return {k: (i, n, g) for k, i, n, g in zip(t["protein"], t["protein_ids"],
                                               t["protein_names"], t["genes"])}


def test_the_import_writes_accessions_names_and_genes_beside_the_precursors(annotated):
    """`protein_annotations.parquet` lands beside the precursor table, keyed by the
    library's `protein` string, with the accessions, entry names and genes of
    `Protein.Ids` / `Protein.Names` / `Genes`: aligned for one protein, `;`-separated for
    a shared peptide, empty where the library has no gene."""
    ann = annotated["dir"] / "protein_annotations.parquet"
    assert ann.is_file()
    assert "protein annotations:" in annotated["stdout"]
    rows = _annotation_rows(ann)
    assert rows["ALBU_HUMAN"] == ("P02768", "ALBU_HUMAN", "ALB")
    assert rows["ALBU_HUMAN;ALBU_BOVIN"] == ("P02768;P02769", "ALBU_HUMAN;ALBU_BOVIN", "ALB")
    assert rows["ENO1_YEAST"] == ("P06733", "ENO1_YEAST", ""), "a missing gene stays empty"


def test_every_library_protein_has_one_annotation_row_and_the_key_is_unchanged(annotated):
    """The key is the library's `protein` cell exactly, so every reported row joins; the
    rows are unique, ascend by key, and are utf8 without nulls. Decoys and UNASSIGNED are
    never annotated (the importer writes targets only)."""
    ann = annotated["dir"] / "protein_annotations.parquet"
    t = pq.read_table(str(ann))
    assert t.column_names == ["protein", "protein_ids", "protein_names", "genes",
                              "description"]
    assert all(pa.types.is_string(f.type) and not f.nullable for f in t.schema)
    keys = t.column("protein").to_pylist()
    assert keys == sorted(keys) and len(keys) == len(set(keys))
    lib_proteins = set(read_columns(annotated["prec"])["protein"])
    assert set(keys) == lib_proteins - {"UNASSIGNED"}
    assert not any(k.startswith("DECOY_") for k in keys)


def test_a_contaminant_id_gives_its_accessions_and_entry_names(annotated):
    """A `CON_sp|ACC|NAME` id has no UniProt entry name in `Protein.Names` (the description
    word `Keratin,`), so the protein key is the id list; the annotation still names the
    accessions and the entry names, and the description word is not a gene."""
    rows = _annotation_rows(annotated["dir"] / "protein_annotations.parquet")
    key = "CON_sp|P02444|KRA33_SHEEP;CON_sp|P02445|KRA34_SHEEP"
    assert rows[key] == ("P02444;P02445", "KRA33_SHEEP;KRA34_SHEEP", "")


def test_isoform_ids_under_one_entry_name_are_united_in_file_order(annotated):
    """Two precursors share the key `TMED2_HUMAN` with different id lists; the row has
    every accession once, in the order the library first gives them."""
    rows = _annotation_rows(annotated["dir"] / "protein_annotations.parquet")
    assert rows["TMED2_HUMAN"] == ("Q15363;Q15363-2", "TMED2_HUMAN", "TMED2")


def test_the_annotation_does_not_change_the_library_tables(annotated, tmp_path):
    """The side table is the only addition: the precursor and fragment tables are the same
    bytes with `--no-annotations`, which writes no table."""
    prec, frag = tmp_path / "lib_precursors.parquet", tmp_path / "lib_fragments.parquet"
    run_worker_ok("import_diann_lib.py", annotated["lib"], prec, frag, "--no-annotations")
    assert not (tmp_path / "protein_annotations.parquet").exists()
    assert prec.read_bytes() == annotated["prec"].read_bytes()
    assert frag.read_bytes() == annotated["frag"].read_bytes()


def test_annotations_path_option_and_a_names_only_library(imported, diann_lib, tmp_path):
    """`--annotations` writes the table where it is told; a library with `Protein.Names`
    only is annotated with its entry names and no accessions."""
    prec, frag = tmp_path / "p.parquet", tmp_path / "f.parquet"
    target = tmp_path / "elsewhere.parquet"
    run_worker_ok("import_diann_lib.py", diann_lib, prec, frag, "--annotations", target)
    assert target.is_file() and not (tmp_path / "protein_annotations.parquet").exists()
    rows = _annotation_rows(target)
    assert rows["ALBU_HUMAN"] == ("", "ALBU_HUMAN", "")
    assert (imported["prec"].parent / "protein_annotations.parquet").is_file()


def test_a_stale_table_is_removed_when_the_library_has_no_protein_columns(diann_lib, tmp_path):
    """A table left in the output directory by another library would annotate this one
    wrongly; with no protein columns to build one from, the import removes it."""
    src = pq.read_table(diann_lib)
    lib = tmp_path / "noprot.parquet"
    pq.write_table(src.drop(["Protein.Names"]), str(lib), compression="snappy")
    stale = tmp_path / "protein_annotations.parquet"
    stale.write_bytes(b"stale")
    run_worker_ok("import_diann_lib.py", lib, tmp_path / "p.parquet", tmp_path / "f.parquet")
    assert not stale.exists()


def test_the_decoy_builder_carries_the_annotations_to_its_output(annotated, tmp_path):
    """A documented recipe writes the decoy library elsewhere than the import; the builder
    copies the table beside its output, byte for byte, so the engine finds it next to the
    library it is given (the reverse builder: `test_decoy_builders.py`)."""
    out = tmp_path / "out"
    out.mkdir()
    prec, frag = out / "lib_precursors.parquet", out / "lib_fragments.parquet"
    run_worker_ok("make_shift_decoys.py", annotated["prec"], annotated["frag"], prec, frag)
    src = annotated["dir"] / "protein_annotations.parquet"
    assert (out / "protein_annotations.parquet").read_bytes() == src.read_bytes()
