"""N-terminal modifications (`[Acetyl]-PEPK`) through the library helpers.

The engine writes a terminal group as ProForma `[Name]-` before the first residue
(`peptidoforms.*_mods[].position`, docs/05). Every helper that reads a peptidoform as
text had been written for residue modifications only, and each one misread the group
silently rather than failing:

- `make_reverse_decoys.py` read the "A" of "Acetyl" as an alanine, so the decoy gained a
  residue, lost the acetyl and no longer had its target's precursor mass;
- `deeplc_finetune.py` and `mh_shard_predict.py` stripped `[Acetyl]` and kept the dash,
  so `-PEPK` was not a standard sequence and every acetylated row kept a retention time
  from another scale;
- `augment_library.py`, `predict_decoys.py` and `nn_rescore_worker.py` produced `-PEPK`
  as the stripped sequence, which matched no target and, for DIA-NN, was no sequence;
- `import_diann_lib.py` dropped every `(UniMod:1)` N-terminal precursor.

Each test below pins the corrected reading against an independent mass or string.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import (
    PROTON,
    RESIDUE_MASS,
    SCRIPTS,
    WATER,
    assert_library_load_invariants,
    read_columns,
    run_worker_ok,
)

ACETYL = 42.010565


def _load(name):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"terminal_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assigned(script, names):
    """Module-level constants and functions of `script`, without importing it (the DeepLC
    and torch workers import their frameworks at the top)."""
    tree = ast.parse((SCRIPTS / script).read_text(encoding="utf-8"))
    namespace = {"re": re}
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)) or (
                isinstance(node, ast.FunctionDef) and node.name in names):
            exec(compile(ast.Module(body=[node], type_ignores=[]), script, "exec"),  # noqa: S102
                 namespace)
    return namespace


# --------------------------------------------------------------- the reverse decoy builder

def _residue_mass(seq):
    return sum(RESIDUE_MASS[c] for c in seq)


def _b(seq, k, z, nterm=0.0):
    return (nterm + _residue_mass(seq[:k]) + z * PROTON) / z


def _y(seq, k, z):
    return (_residue_mass(seq[len(seq) - k:]) + WATER + z * PROTON) / z


# (peptidoform, stripped, N-terminal mass)
NTERM_TARGETS = [
    ("[Acetyl]-SAMPLEDK", "SAMPLEDK", ACETYL),
    ("SAMPLEDK", "SAMPLEDK", 0.0),
    ("[Acetyl]-MTEYKLVVVGAR", "MTEYKLVVVGAR", ACETYL),
    ("PEPTIDER", "PEPTIDER", 0.0),
    ("AGWNSEHK", "AGWNSEHK", 0.0),
]
NTERM_FRAGMENTS = (("b", 2, 1), ("b", 3, 1), ("y", 3, 1), ("y", 4, 2))


@pytest.fixture(scope="module")
def acetyl_reverse(tmp_path_factory):
    d = tmp_path_factory.mktemp("acetyl_reverse")
    rows = []
    for pform, seq, nt in NTERM_TARGETS:
        mz = (nt + _residue_mass(seq) + WATER + 2 * PROTON) / 2
        rows.append((pform, seq, nt, mz))
    rows.sort(key=lambda r: r[3])
    n = len(rows)
    bases = {}
    prec = pa.table({
        "candidate_id": pa.array(range(n), pa.uint32()),
        "peptidoform_id": pa.array(range(n), pa.uint32()),
        "base_peptide_id": pa.array([bases.setdefault(r[1], len(bases)) for r in rows],
                                    pa.uint32()),
        "peptidoform": pa.array([r[0] for r in rows], pa.string()),
        "charge": pa.array([2] * n, pa.int32()),
        "precursor_mz": pa.array([r[3] for r in rows], pa.float64()),
        "predicted_irt": pa.array([10.0 * i for i in range(n)], pa.float32()),
        "label": pa.array(["target"] * n, pa.string()),
        "protein": pa.array(["P1"] * n, pa.string()),
        "n_fragments": pa.array([len(NTERM_FRAGMENTS)] * n, pa.int32()),
    })
    cols = {k: [] for k in ("candidate_id", "mz", "predicted_intensity", "name", "ion_type",
                            "ordinal", "frag_charge", "cardinality")}
    for cid, (_, seq, nt, _) in enumerate(rows):
        for ion, k, z in NTERM_FRAGMENTS:
            cols["candidate_id"].append(cid)
            cols["mz"].append(_b(seq, k, z, nt) if ion == "b" else _y(seq, k, z))
            cols["predicted_intensity"].append(0.5)
            cols["name"].append(f"{ion}{k}" + (f"^{z}" if z > 1 else ""))
            cols["ion_type"].append(ion)
            cols["ordinal"].append(k)
            cols["frag_charge"].append(z)
            cols["cardinality"].append(1)
    frag = pa.table({
        "candidate_id": pa.array(cols["candidate_id"], pa.uint32()),
        "mz": pa.array(cols["mz"], pa.float64()),
        "predicted_intensity": pa.array(cols["predicted_intensity"], pa.float32()),
        "name": pa.array(cols["name"], pa.string()),
        "ion_type": pa.array(cols["ion_type"], pa.string()),
        "ordinal": pa.array(cols["ordinal"], pa.int32()),
        "frag_charge": pa.array(cols["frag_charge"], pa.int32()),
        "cardinality": pa.array(cols["cardinality"], pa.int32()),
    })
    tp, tf = d / "t_prec.parquet", d / "t_frag.parquet"
    pq.write_table(prec, str(tp), compression="snappy")
    pq.write_table(frag, str(tf), compression="snappy")
    op, of = d / "prec.parquet", d / "frag.parquet"
    stdout, _ = run_worker_ok("make_reverse_decoys.py", tp, tf, op, of)
    return op, of, stdout


def test_a_reverse_decoy_keeps_the_n_terminal_group_and_its_mass(acetyl_reverse):
    """The decoy of `[Acetyl]-SAMPLEDK` is `DECOY_[Acetyl]-` + the reversed sequence, at
    the target's precursor m/z, with every b ion 42.0106 Da above the plain b ion of the
    reversed sequence and every y ion without it."""
    prec_path, frag_path, stdout = acetyl_reverse
    assert "skipped-nonstd=0" in stdout, stdout
    assert_library_load_invariants(prec_path, frag_path)
    prec = read_columns(prec_path)
    frag = read_columns(frag_path)
    by_cid = {int(c): i for i, c in enumerate(prec["candidate_id"])}
    targets = {pf: i for i, (pf, lab) in enumerate(zip(prec["peptidoform"], prec["label"]))
               if lab == "target"}
    decoys = [i for i, lab in enumerate(prec["label"]) if lab == "decoy"]
    assert len(decoys) == len(targets) == len(NTERM_TARGETS)
    n_acetyl = sum(prec["peptidoform"][i].startswith("DECOY_[Acetyl]-") for i in decoys)
    assert n_acetyl == sum(pf.startswith("[Acetyl]-") for pf in targets) == 2
    checked = 0
    for i in decoys:
        pf = prec["peptidoform"][i][len("DECOY_"):]
        nt = ACETYL if pf.startswith("[Acetyl]-") else 0.0
        seq = pf[len("[Acetyl]-"):] if nt else pf
        assert re.fullmatch("[A-Z]+", seq), pf
        # The precursor m/z is the target's: same composition, same group.
        expect_mz = (nt + _residue_mass(seq) + WATER + 2 * PROTON) / 2
        assert abs(prec["precursor_mz"][i] - expect_mz) < 1e-6, pf
        cid = int(prec["candidate_id"][i])
        for k, c in enumerate(frag["candidate_id"]):
            if int(c) != cid:
                continue
            ion, o, z = frag["ion_type"][k], int(frag["ordinal"][k]), int(frag["frag_charge"][k])
            want = _b(seq, o, z, nt) if ion == "b" else _y(seq, o, z)
            assert 1e6 * abs(frag["mz"][k] - want) / want < 1.0, (pf, ion, o, z)
            checked += 1
    assert checked == len(decoys) * len(NTERM_FRAGMENTS)
    assert by_cid  # the ids are the row-aligned range checked above


def test_the_decoy_tokenizer_never_reads_a_terminal_group_as_residues():
    m = _load("make_reverse_decoys")
    assert m.nterm("[Acetyl]-SEK") == "Acetyl"
    assert m.nterm("DECOY_[Acetyl]-KES") == "Acetyl"
    assert m.parse("[Acetyl]-S[Phospho]EK") == [("S", "Phospho"), ("E", ""), ("K", "")]
    assert m.to_pform(m.parse("[Acetyl]-SEK"), "Acetyl") == "[Acetyl]-SEK"
    # A C-terminal group is not read as residues either: the precursor is refused.
    assert not m.valid(m.parse("PEPK-[Amidated]"))
    assert m.valid(m.parse("[Acetyl]-PEPK"), "Acetyl")
    assert not m.valid(m.parse("[NoSuchMod]-PEPK"), "NoSuchMod")


# --------------------------------------------------------------- DeepLC sequence filter

def test_deeplc_predicts_n_terminally_modified_rows():
    """`is_std` is the filter for what DeepLC predicts. `[Acetyl]-PEPK` must pass it, and
    the shard recipe must select the same rows (its constants are pinned to the worker's)."""
    w = _assigned("deeplc_finetune.py", {"STD", "MOD_RE", "strip_mods", "base_pf", "is_std"})
    for pf in ("[Acetyl]-PEPTIDEK", "DECOY_[Acetyl]-KEDITPEP", "[Acetyl]-M[Oxidation]PEPK",
               "PEPTIDEK", "PEPC[Carbamidomethyl]K"):
        assert w["is_std"](pf), pf
    assert not w["is_std"]("PEPUK")
    assert w["strip_mods"]("[Acetyl]-S[Phospho]EK") == "SEK"
    r = _assigned("mh_shard_predict.py", {"MOD_RE", "STD_RE"})
    assert r["MOD_RE"] == w["MOD_RE"]
    import pyarrow.compute as pc
    col = pa.array(["[Acetyl]-PEPK", "PEPK", "-PEPK", "[Acetyl]-PEPUK"])
    stripped = pc.replace_substring_regex(col, r["MOD_RE"], "")
    keep = pc.match_substring_regex(stripped, r["STD_RE"]).to_pylist()
    assert keep == [True, True, False, False]


# --------------------------------------------------------------- stripping elsewhere

@pytest.mark.parametrize("pform, want", [
    ("[Acetyl]-PEPTIDEK", "PEPTIDEK"),
    ("DECOY_[Acetyl]-KEDITPEP", "KEDITPEP"),
    ("[Acetyl]-S[Phospho]EK", "SEK"),
    ("PEPC[Carbamidomethyl]K", "PEPCK"),
])
def test_stripped_sequences_drop_the_terminal_dash(pform, want):
    assert _load("augment_library").stripped(pform) == want
    assert _load("predict_decoys").stripped(pform) == want
    strip_pep = _assigned("nn_rescore_worker.py", {"strip_pep"})["strip_pep"]
    assert strip_pep(pform.replace("DECOY_", "")) == want


def test_predict_decoys_writes_an_n_terminal_group_as_dia_nn_does():
    """DIA-NN writes an N-terminal modification before the first residue, without the
    ProForma dash; residue modifications keep their place."""
    m = _load("predict_decoys")
    assert m.to_diann("DECOY_[Acetyl]-KEDITPEP") == "_[UniMod:1]KEDITPEP_"
    assert m.to_diann("[Acetyl]-M[Oxidation]PEPK") == "_[UniMod:1]M[UniMod:35]PEPK_"
    assert m.to_diann("PEPC[Carbamidomethyl]K") == "_PEPC[UniMod:4]K_"


# --------------------------------------------------------------- the DIA-NN importer

def test_import_keeps_a_leading_unimod_as_the_proforma_n_terminal_group():
    m = _load("import_diann_lib")
    assert m.to_proforma("(UniMod:1)PEPTIDEK") == "[Acetyl]-PEPTIDEK"
    assert m.to_proforma("(UniMod:1)M(UniMod:35)PEPK") == "[Acetyl]-M[Oxidation]PEPK"
    assert m.to_proforma("PEPTIDEK(UniMod:1)") == "PEPTIDEK[Acetyl]"
    assert m.to_proforma("SC(UniMod:4)K") == "SC[Carbamidomethyl]K"


def test_the_engine_mass_of_a_terminal_group_is_the_helpers_mass():
    """The catalogue mass the helpers use for Acetyl is the one the engine applies."""
    assert abs(_load("make_reverse_decoys").mod_mass("Acetyl") - ACETYL) < 1e-5
    assert np.isclose(_load("make_reverse_decoys").frag_mz(
        [("S", ""), ("E", ""), ("K", "")], "b", 2, 1, "Acetyl"),
        _b("SEK", 2, 1, ACETYL), atol=1e-5)
