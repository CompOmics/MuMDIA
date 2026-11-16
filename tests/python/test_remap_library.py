"""Contract tests for `scripts/remap_library.py` and `import_diann_lib.py --remap-fasta` (#319).

The remap replaces the `protein` column of a library with the proteins of another FASTA.
That column is the grouping key of every protein-level result, so the rules are checked
one by one: I/L equivalence, N-terminal Met excision, protein N-terminal groups only at a
protein start, shared peptides in FASTA order, decoys following their targets, unmapped
peptides dropped or kept, and a library the engine still loads afterwards.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pyarrow.parquet as pq
import pytest

from conftest import (
    SCRIPTS,
    assert_library_load_invariants,
    assert_string_columns_utf8,
    read_columns,
    run_worker,
    run_worker_ok,
)
from test_import_diann_lib import diann_lib  # noqa: F401  (module-scoped fixture)


def _load():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("mumdia_remap_library",
                                                  SCRIPTS / "remap_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# PEPTIDEK occurs in ONE as PEPTLDEK (I/L) and in TWO verbatim; KITTENR in ONE after its
# initiator Met; SCENICK in TWO; MELLOWK starts CON_P33333. NTERMCK occurs nowhere.
FASTA = """>sp|P11111|ONE_HUMAN Protein one OS=Homo sapiens GN=ONE PE=1
MKITTENRPEPTLDEKGG
>sp|P22222|TWO_YEAST Protein two OS=Saccharomyces GN=TWO PE=1
GKSCENICKPEPTIDEKAA
>CON_P33333 a contaminant
MELLOWKGG
"""


def _records(text):
    return [(r.split("\n", 1)[0].split()[0], r.split("\n", 1)[1].replace("\n", ""))
            for r in text.strip().lstrip(">").split("\n>")]


def test_sequences_map_up_to_i_l_in_fasta_order_once_per_protein():
    m = _load()
    recs = [("A", "PEPTLDEKPEPTIDEK"), ("B", "XXPEPTIDEK"), ("C", "MPEPTIDEK")]
    hits = m.map_sequences(["PEPTIDEK", "NOWHEREK"], recs)
    # One entry per protein even with two occurrences in A; C starts after its Met.
    assert hits == {"PEPTIDEK": [(0, True), (1, False), (2, True)]}
    assert m.map_sequences(["PEPTIDEK"], recs, met_excision=False)["PEPTIDEK"][2] == (2, False)
    # A sequence shorter than the others shrinks the window; nothing is missed.
    hits = m.map_sequences(["DEK", "PEPTIDEK"], recs)
    assert [r for r, _ in hits["DEK"]] == [0, 1, 2]


def test_a_protein_n_terminal_group_maps_only_where_the_peptide_starts_a_protein():
    m = _load()
    recs = [("INNER", "GGSACETYLKR"), ("NTERM", "MSACETYLKR"), ("PLAIN", "SACETYLK")]
    got, n = m.target_proteins(["[Acetyl]-SACETYLK", "SACETYLK", "[TMT6plex]-SACETYLK"], recs)
    assert got == ["NTERM;PLAIN", "INNER;NTERM;PLAIN", "INNER;NTERM;PLAIN"]
    assert n == 3
    # Without Met excision only position 0 starts a protein.
    got, _ = m.target_proteins(["[Acetyl]-SACETYLK"], recs, met_excision=False)
    assert got == ["PLAIN"]
    # No group named: the acetylated form maps like any other.
    got, _ = m.target_proteins(["[Acetyl]-SACETYLK"], recs, n_term_mods=())
    assert got == ["INNER;NTERM;PLAIN"]


def test_the_protein_key_options():
    m = _load()
    assert m.protein_key("sp|P11111|ONE_HUMAN", "header") == "sp|P11111|ONE_HUMAN"
    assert m.protein_key("sp|P11111|ONE_HUMAN", "accession") == "P11111"
    assert m.protein_key("sp|P11111|ONE_HUMAN", "entry_name") == "ONE_HUMAN"
    # A header without the fields keeps its first word.
    assert m.protein_key("CON_P33333", "entry_name") == "CON_P33333"
    assert _records(FASTA)[2] == ("CON_P33333", "MELLOWKGG")


@pytest.fixture()
def library(diann_lib, tmp_path):  # noqa: F811
    """The importer's fixture library with shift decoys, and the FASTA above."""
    tp, tf = tmp_path / "t_prec.parquet", tmp_path / "t_frag.parquet"
    run_worker_ok("import_diann_lib.py", diann_lib, tp, tf)
    prec, frag = tmp_path / "lib_precursors.parquet", tmp_path / "lib_fragments.parquet"
    run_worker_ok("make_shift_decoys.py", tp, tf, prec, frag)
    fasta = tmp_path / "new.fasta"
    fasta.write_text(FASTA)
    return {"prec": prec, "frag": frag, "fasta": fasta, "dir": tmp_path}


def _rows(path):
    c = read_columns(path)
    return [dict(zip(c, v)) for v in zip(*c.values())]


def test_a_library_is_remapped_and_its_decoys_follow(library):
    lib = library
    out_p, out_f = lib["dir"] / "r_prec.parquet", lib["dir"] / "r_frag.parquet"
    summary = lib["dir"] / "remap.json"
    run_worker_ok("remap_library.py", lib["prec"], lib["frag"], out_p, out_f,
                  "--fasta", lib["fasta"], "--summary", summary)
    p, f = assert_library_load_invariants(out_p, out_f, require_both_labels=True)
    assert_string_columns_utf8(out_p)
    rows = _rows(out_p)
    targets = {(r["peptidoform"], r["charge"]): r for r in rows if r["label"] == "target"}
    # NTERMCK occurs in no protein: dropped, with its decoy.
    assert sorted(k[0] for k in targets) == sorted(
        ["M[Oxidation]ELLOWK", "SC[Carbamidomethyl]ENICK", "K[Farnesyl]ITTENR",
         "PEPTIDEK", "PEPTIDEK"])
    assert targets[("PEPTIDEK", 2)]["protein"] == "sp|P11111|ONE_HUMAN;sp|P22222|TWO_YEAST"
    assert targets[("K[Farnesyl]ITTENR", 2)]["protein"] == "sp|P11111|ONE_HUMAN"
    assert targets[("M[Oxidation]ELLOWK", 2)]["protein"] == "CON_P33333"
    # Annotations from the new FASTA's headers.
    pep = targets[("PEPTIDEK", 2)]
    assert (pep["protein_ids"], pep["protein_names"], pep["genes"]) == (
        "P11111;P22222", "ONE_HUMAN;TWO_YEAST", "ONE;TWO")
    # Every decoy carries DECOY_ and its target's new protein, one decoy per target.
    decoys = [r for r in rows if r["label"] == "decoy"]
    assert len(decoys) == len(targets)
    tgt_by_pair = {(r["peptidoform_id"], r["charge"]): r for r in targets.values()}
    for d in decoys:
        assert d["protein"] == "DECOY_" + tgt_by_pair[(d["peptidoform_id"], d["charge"])]["protein"]
    # Fragments: the kept candidates' rows only, renumbered with their precursors.
    n_frag = read_columns(out_f)["candidate_id"]
    assert len(n_frag) == sum(r["n_fragments"] for r in rows)
    s = json.loads(summary.read_text())
    assert (s["targets_in"], s["targets_mapped"], s["targets_unmapped"]) == (6, 5, 1)
    assert s["rows_out"] == 10 and s["decoys_without_target"] == 0


def test_unmapped_keep_writes_unassigned(library):
    lib = library
    out_p, out_f = lib["dir"] / "k_prec.parquet", lib["dir"] / "k_frag.parquet"
    run_worker_ok("remap_library.py", lib["prec"], lib["frag"], out_p, out_f,
                  "--fasta", lib["fasta"], "--unmapped", "keep", "--protein-key", "entry_name")
    assert_library_load_invariants(out_p, out_f, require_both_labels=True)
    rows = _rows(out_p)
    assert len(rows) == 12
    nterm = [r for r in rows if r["peptidoform"].endswith("NTERMCK")]
    assert {r["protein"] for r in nterm} == {"UNASSIGNED", "DECOY_UNASSIGNED"}
    pep = next(r for r in rows if r["peptidoform"] == "PEPTIDEK")
    assert pep["protein"] == "ONE_HUMAN;TWO_YEAST"
    assert pep["genes"] == "ONE;TWO"


def test_the_remap_can_write_over_its_input(library):
    lib = library
    before = pq.read_metadata(str(lib["frag"])).num_rows
    run_worker_ok("remap_library.py", lib["prec"], lib["frag"], lib["prec"], lib["frag"],
                  "--fasta", lib["fasta"])
    p, f = assert_library_load_invariants(lib["prec"], lib["frag"], require_both_labels=True)
    assert p.num_rows == 10 and f.num_rows < before


def test_a_fasta_without_any_library_peptide_is_refused(library):
    lib = library
    empty = lib["dir"] / "none.fasta"
    empty.write_text(">X\nGGGGGGGGGG\n")
    rc, _, err = run_worker("remap_library.py", lib["prec"], lib["frag"],
                            lib["dir"] / "o.parquet", lib["dir"] / "of.parquet", "--fasta", empty)
    assert rc != 0 and "nothing to write" in err


def test_the_importer_remaps_its_targets(diann_lib, tmp_path):  # noqa: F811
    fasta = tmp_path / "new.fasta"
    fasta.write_text(FASTA)
    prec, frag = tmp_path / "p.parquet", tmp_path / "f.parquet"
    out, _ = run_worker_ok("import_diann_lib.py", diann_lib, prec, frag, "--remap-fasta", fasta,
                           "--remap-unmapped", "keep")
    assert "remap: 5 of 6 target precursors" in out
    p, _ = assert_library_load_invariants(prec, frag, require_both_labels=False)
    proteins = dict(zip(read_columns(prec)["peptidoform"], read_columns(prec)["protein"]))
    assert proteins["PEPTIDEK"] == "sp|P11111|ONE_HUMAN;sp|P22222|TWO_YEAST"
    assert proteins["[Carbamidomethyl]-NTERMCK"] == "UNASSIGNED"
    # The remapped targets feed the decoy builder as before.
    run_worker_ok("make_shift_decoys.py", prec, frag, tmp_path / "d.parquet",
                  tmp_path / "df.parquet")
    # --fasta and --remap-fasta name the proteins two ways; refused together.
    rc, _, err = run_worker("import_diann_lib.py", diann_lib, prec, frag, "--remap-fasta", fasta,
                            "--fasta", fasta)
    assert rc != 0 and "--fasta" in err
    rc, _, err = run_worker("import_diann_lib.py", diann_lib, prec, frag, "--remap-unmapped", "keep")
    assert rc != 0 and "--remap-fasta" in err


def test_a_decoy_without_a_peptidoform_pair_follows_its_base_peptide(tmp_path):
    """A FASTA-built library pairs a decoy with its target through `base_peptide_id` only;
    such a decoy takes the proteins of its base peptide's kept targets, and is dropped with
    them."""
    import pandas as pd

    m = _load()
    fasta = tmp_path / "f.fasta"
    fasta.write_text(">A\nGGPEPTIDEKGG\n>B\nPEPTLDEK\n")
    prec = pd.DataFrame({
        "candidate_id": [0, 1, 2, 3],
        "peptidoform_id": [0, 1, 10, 11],
        "base_peptide_id": [5, 6, 5, 6],
        "peptidoform": ["PEPTIDEK", "MISSINGK", "KEDITPEP", "KGNISSIM"],
        "charge": [2, 2, 2, 2],
        "label": ["target", "target", "decoy", "decoy"],
        "protein": ["old", "old2", "DECOY_old", "DECOY_old2"],
    })
    out, keep, s = m.remap_precursors(prec, str(fasta), log=lambda *_: None)
    assert keep.tolist() == [True, False, True, False]
    assert out["protein"].tolist() == ["A;B", "DECOY_A;B"]
    assert s["decoys_paired_by_base_peptide"] == 1 and s["decoys_without_target"] == 1


def test_the_cleavage_presets_are_the_engines():
    """The script carries a copy of `enzymes.json`'s presets (the scripts ship without the
    engine's sources); the copy and its rule semantics must be the engine's."""
    m = _load()
    path = SCRIPTS.parent / "rust" / "mumdia" / "crates" / "mumdia-core" / "src" / "enzymes.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    presets = {k: v["rule"] for k, v in spec["presets"].items() if v["rule"] is not None}
    assert m.ENZYME_RULES == presets
    for t in spec["test_vectors"]:
        assert m.CleavageRule(t["rule"]).sites(t["seq"]) == t["sites"], t


def test_only_a_specific_occurrence_counts_under_a_cleavage_rule():
    m = _load()
    recs = [("TRYPTIC", "AKPEPTIDEKAA"), ("INNER", "AAPEPTIDEKAA"), ("CTERM", "GRPEPTIDEK"),
            ("NOEND", "KPEPTIDEKP"), ("MET", "MPEPTIDEKR")]
    tp = m.cleavage_rule("trypsin_p")
    hits = m.map_sequences(["PEPTIDEK"], recs, rule=tp)
    # INNER is preceded by A; MET is the excised N-terminal form.
    assert [recs[r][0] for r, _ in hits["PEPTIDEK"]] == ["TRYPTIC", "CTERM", "NOEND", "MET"]
    hits = m.map_sequences(["PEPTIDEK"], recs, rule=m.cleavage_rule("trypsin"))
    # Trypsin does not cut before a P, so a peptide starting with P only starts a protein.
    assert [recs[r][0] for r, _ in hits["PEPTIDEK"]] == ["MET"]
    assert m.cleavage_rule("any") is None and m.cleavage_rule("non_specific") is None
    assert len(m.map_sequences(["PEPTIDEK"], recs)["PEPTIDEK"]) == 5
