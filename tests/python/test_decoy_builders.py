"""Contract tests for `scripts/make_reverse_decoys.py` (offline DIA-NN recipe).

Need only pandas and pyarrow, so they always run. The script creates the decoy
population the whole FDR rests on (whose spectra `predict_decoys.py` then replaces),
and it is the step that makes an imported library index-valid: it re-sorts by
`precursor_mz` and reassigns contiguous `candidate_id`, which is what satisfies
`index.rs load()`.

Three properties matter beyond the index preconditions
(`CLAUDE.md`, "Coding conventions"):

* paired: every retained target has exactly one decoy, and a target whose decoy
  could not be built is removed with it. An unpaired target biases the null.
* collision free: no decoy stripped sequence may equal a real target's, or the
  decoy is scored against real evidence and the FDR is anti-conservative.
* exchangeable: the decoy keeps the target's precursor m/z and iRT, so it
  co-isolates and co-elutes and experiences the same interference.

The fragment m/z fixture is computed from the residue table in `conftest.py`,
transcribed independently of `make_reverse_decoys.py`, so its own calculator
check (abort above 5 ppm at p99) is a real cross-validation.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import (
    assert_library_load_invariants,
    assert_string_columns_utf8,
    parse_proforma,
    precursor_mz,
    read_columns,
    reverse_keep_cterm,
    run_worker,
    run_worker_ok,
    stripped,
    fragment_mz,
    library_im,
    SCRIPTS,
)

# (peptidoform, charge). PEPTIDEK appears twice so the builders see two charge
# rows of one base sequence, which may legitimately share a decoy sequence.
# AKAK reverses onto itself, and AGWK/WGAK reverse onto each other, so all three
# force the collision-resolution path. IEVNVELR reverses onto LEVNVEIR, which differs
# from it only by the isobaric I/L, so it collides too (found on an Astral HYE run).
TARGETS = [
    ("PEPTIDEK", 2),
    ("PEPTIDEK", 3),
    ("SAMPLER", 2),
    ("AKAK", 2),
    ("AGWK", 2),
    ("WGAK", 2),
    ("SC[Carbamidomethyl]ENICK", 2),
    ("IEVNVELR", 2),
]
COLLIDING = {"AKAK", "AGWK", "WGAK", "IEVNVELR"}
FRAGMENTS = (("y", 2, 1), ("b", 2, 1), ("y", 3, 2))


def _write_target_library(directory, mz_scale=1.0):
    """Write a target-only library in the `import_diann_lib` output schema."""
    rows = []
    for pform, charge in TARGETS:
        tokens = parse_proforma(pform)
        rows.append({
            "peptidoform": pform,
            "charge": charge,
            "tokens": tokens,
            "precursor_mz": precursor_mz(tokens, charge),
        })
    rows.sort(key=lambda r: r["precursor_mz"])
    bases = {}
    for i, r in enumerate(rows):
        r["candidate_id"] = i
        r["base_peptide_id"] = bases.setdefault(stripped(r["tokens"]), len(bases))

    prec = pa.table({
        "candidate_id": pa.array([r["candidate_id"] for r in rows], pa.uint32()),
        "peptidoform_id": pa.array([r["candidate_id"] for r in rows], pa.uint32()),
        "base_peptide_id": pa.array([r["base_peptide_id"] for r in rows], pa.uint32()),
        "peptidoform": pa.array([r["peptidoform"] for r in rows], pa.string()),
        "charge": pa.array([r["charge"] for r in rows], pa.int32()),
        "precursor_mz": pa.array([r["precursor_mz"] for r in rows], pa.float64()),
        "predicted_irt": pa.array(
            [100.0 + 10.0 * r["candidate_id"] for r in rows], pa.float32()),
        "label": pa.array(["target"] * len(rows), pa.string()),
        "protein": pa.array(
            ["PROT{}_ECOLI".format(r["base_peptide_id"]) for r in rows], pa.string()),
        "n_fragments": pa.array([len(FRAGMENTS)] * len(rows), pa.int32()),
        "predicted_im": pa.array([library_im(r["candidate_id"]) for r in rows], pa.float64()),
    })

    fcid, fmz, fint, fname, fion, ford, fz = [], [], [], [], [], [], []
    for r in rows:
        for ion, ordinal, charge in FRAGMENTS:
            fcid.append(r["candidate_id"])
            fmz.append(fragment_mz(r["tokens"], ion, ordinal, charge) * mz_scale)
            fint.append(0.5)
            fname.append("{}{}{}".format(
                ion, ordinal, "^{}".format(charge) if charge > 1 else ""))
            fion.append(ion)
            ford.append(ordinal)
            fz.append(charge)
    frag = pa.table({
        "candidate_id": pa.array(fcid, pa.uint32()),
        "mz": pa.array(fmz, pa.float64()),
        "predicted_intensity": pa.array(fint, pa.float32()),
        "name": pa.array(fname, pa.string()),
        "ion_type": pa.array(fion, pa.string()),
        "ordinal": pa.array(ford, pa.int32()),
        "frag_charge": pa.array(fz, pa.int32()),
        "cardinality": pa.array([1] * len(fcid), pa.int32()),
    })

    prec_path = directory / "target_precursors.parquet"
    frag_path = directory / "target_fragments.parquet"
    pq.write_table(prec, str(prec_path), compression="snappy")
    pq.write_table(frag, str(frag_path), compression="snappy")
    return {"prec": prec_path, "frag": frag_path, "rows": rows}


@pytest.fixture(scope="module")
def target_library(tmp_path_factory):
    return _write_target_library(tmp_path_factory.mktemp("target_lib"))


def _build(script, target_library, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    prec = out_dir / "lib_precursors.parquet"
    frag = out_dir / "lib_fragments.parquet"
    stdout, stderr = run_worker_ok(
        script, target_library["prec"], target_library["frag"], prec, frag
    )
    return {"prec": prec, "frag": frag, "stdout": stdout, "stderr": stderr}


@pytest.fixture(scope="module")
def reverse_library(target_library, tmp_path_factory):
    return _build("make_reverse_decoys.py", target_library,
                  tmp_path_factory.mktemp("reverse_out"))


def _split_by_label(prec_path, frag_path):
    prec = read_columns(prec_path)
    frag = read_columns(frag_path)
    idx_t = [i for i, lab in enumerate(prec["label"]) if lab == "target"]
    idx_d = [i for i, lab in enumerate(prec["label"]) if lab == "decoy"]
    return prec, frag, idx_t, idx_d


# ---------------------------------------------------------------------------
# shared invariants
# ---------------------------------------------------------------------------


def test_output_library_is_index_valid(reverse_library):
    """Contiguous ids, ascending `precursor_mz`, both labels present.

    The decoy build is the last step before a search, so a violation here is the
    one that actually reaches `index.rs load()` and aborts. The builder
    concatenates target and decoy, re-sorts by m/z and reassigns ids precisely to
    satisfy this; a reordering added afterwards would break it silently.
    """
    lib = reverse_library
    assert_library_load_invariants(lib["prec"], lib["frag"], require_both_labels=True)


def test_the_builder_says_its_decoys_must_be_predicted(reverse_library):
    """Its decoys copy their targets' intensities, which makes the reported FDR optimistic;
    the build says so, since the output passes every check a search makes."""
    assert "run predict_decoys.py build" in reverse_library["stderr"]


def test_every_target_has_exactly_one_decoy(target_library, reverse_library):
    """Target and decoy populations must be paired one to one.

    `q = (decoys + 1) / max(1, targets)` assumes the two populations are
    exchangeable. Retaining a target whose decoy could not be built, or emitting
    two decoys for one target, biases every q-value in the run, and nothing
    downstream can detect it.
    """
    lib = reverse_library
    prec, _, idx_t, idx_d = _split_by_label(lib["prec"], lib["frag"])
    assert len(idx_t) == len(TARGETS)
    assert len(idx_d) == len(idx_t)
    pid_t = [int(prec["peptidoform_id"][i]) for i in idx_t]
    pid_d = [int(prec["peptidoform_id"][i]) for i in idx_d]
    assert sorted(pid_t) == sorted(pid_d), (
        "decoys do not map one to one onto the targets by peptidoform_id"
    )
    assert len(set(pid_d)) == len(pid_d), "a target got more than one decoy"


def test_decoy_keeps_the_target_precursor_mz_and_irt(reverse_library):
    """The decoy must co-isolate and co-elute with its target.

    Precursor m/z and iRT are what put the decoy in the same isolation window at
    the same retention time, so it experiences the same chimeric interference.
    A decoy at a different m/z or RT is an easier problem than the target and
    the FDR it produces is optimistic. The library 1/K0 follows the iRT: the decoy
    takes its target's value, null included, so it shares the target's mobility
    window as well.
    """
    lib = reverse_library
    prec, _, idx_t, idx_d = _split_by_label(lib["prec"], lib["frag"])
    by_pid = {}
    for i in idx_t:
        by_pid[int(prec["peptidoform_id"][i])] = i
    for j in idx_d:
        i = by_pid[int(prec["peptidoform_id"][j])]
        assert float(prec["precursor_mz"][j]) == pytest.approx(
            float(prec["precursor_mz"][i]), abs=1e-9)
        assert float(prec["predicted_irt"][j]) == pytest.approx(
            float(prec["predicted_irt"][i]), rel=1e-6)
        assert int(prec["charge"][j]) == int(prec["charge"][i])
        assert prec["predicted_im"][j] == prec["predicted_im"][i]
    ims = list(prec["predicted_im"])
    assert any(v is None for v in ims) and any(v is not None for v in ims), (
        "the fixture must exercise both a null and a finite library 1/K0")


def test_builder_is_deterministic(target_library, tmp_path):
    """Two builds of the same library must be byte-for-byte equivalent.

    `make_reverse_decoys.py` resolves collisions with a Fisher-Yates seeded from
    a process-independent FNV-1a hash rather than Python's randomized builtin
    `hash`, so a rebuild does not need `PYTHONHASHSEED`. If that regressed, two
    library builds would carry different decoys and no result computed against
    one of them would be reproducible against the other.
    """
    a = _build("make_reverse_decoys.py", target_library, tmp_path / "a")
    b = _build("make_reverse_decoys.py", target_library, tmp_path / "b")
    for key in ("prec", "frag"):
        ca, cb = read_columns(a[key]), read_columns(b[key])
        assert list(ca) == list(cb)
        for col in ca:
            if isinstance(ca[col], list):
                assert ca[col] == cb[col], "{} differs between builds".format(col)
            else:
                assert np.allclose(np.asarray(ca[col], dtype=float),
                                   np.asarray(cb[col], dtype=float), atol=0, rtol=0,
                                   equal_nan=True)


def test_output_string_columns_are_utf8(reverse_library):
    """String columns must be arrow `utf8`, never `large_utf8`.

    The builder relies on the pandas default instead of casting to
    `pa.string()`, so on a pandas whose string dtype is arrow-backed the emitted
    library is rejected by the engine with "column 'peptidoform' is not utf8".
    """
    lib = reverse_library
    for path in (lib["prec"], lib["frag"]):
        assert_string_columns_utf8(path)


# ---------------------------------------------------------------------------
# make_reverse_decoys: reversal, collision resolution, m/z recomputation
# ---------------------------------------------------------------------------


def test_reverse_decoys_never_overlap_a_real_target_sequence(reverse_library):
    """`decoy_stripped` and `target_stripped` must be disjoint.

    A palindrome, or a peptide whose reversal equals another target, would
    otherwise produce a "decoy" that is a real peptide in the sample. It then
    collects real chromatographic evidence, the decoy score distribution shifts
    up, and every q-value in the run becomes anti-conservative.
    """
    prec, _, idx_t, idx_d = _split_by_label(reverse_library["prec"],
                                            reverse_library["frag"])
    # I and L are isobaric: a decoy equal to a target up to I/L is that target.
    il = lambda pf: stripped(parse_proforma(pf)).replace("I", "L")
    tgt = {il(prec["peptidoform"][i]) for i in idx_t}
    dec = {il(prec["peptidoform"][j]) for j in idx_d}
    assert tgt & dec == set()
    for j in idx_d:
        assert prec["peptidoform"][j].startswith("DECOY_"), (
            "the reverse builder must still mark the decoy label in the name"
        )


def test_reverse_keeps_the_cterm_residue_and_the_residue_composition(reverse_library):
    """Non-colliding peptides are reversed with the C-terminal residue fixed.

    Reversal preserves residue composition, which is what lets the decoy keep
    the target's precursor m/z; keeping the C-terminal residue keeps the decoy
    inside the enzyme's specificity, so it is not trivially separable from a
    real tryptic peptide by any feature.
    """
    prec, _, idx_t, idx_d = _split_by_label(reverse_library["prec"],
                                            reverse_library["frag"])
    tgt_by_pid = {int(prec["peptidoform_id"][i]): prec["peptidoform"][i]
                  for i in idx_t}
    resolved = 0
    for j in idx_d:
        pid = int(prec["peptidoform_id"][j])
        t_tokens = parse_proforma(tgt_by_pid[pid])
        d_tokens = parse_proforma(prec["peptidoform"][j])
        t_seq, d_seq = stripped(t_tokens), stripped(d_tokens)
        assert sorted(d_seq) == sorted(t_seq), "composition changed"
        assert d_seq[-1] == t_seq[-1], "the C-terminal residue moved"
        if t_seq in COLLIDING:
            resolved += 1
            assert d_seq != stripped(reverse_keep_cterm(t_tokens)), (
                "the colliding peptide {} kept its plain reversal".format(t_seq)
            )
        else:
            assert d_seq == stripped(reverse_keep_cterm(t_tokens))
    assert resolved == len(COLLIDING), (
        "expected {} collision-resolved peptides".format(len(COLLIDING))
    )


def test_distinct_target_sequences_do_not_share_a_decoy_sequence(reverse_library):
    """Two different base sequences must not collapse onto one decoy.

    Sharing a decoy shrinks the effective null relative to the target space, so
    the reported q-values are computed against fewer independent decoys than the
    target count implies. Charge rows of one base sequence may reuse it, which
    is why the check is per base sequence.
    """
    prec, _, idx_t, idx_d = _split_by_label(reverse_library["prec"],
                                            reverse_library["frag"])
    tgt_by_pid = {int(prec["peptidoform_id"][i]): prec["peptidoform"][i] for i in idx_t}
    pairs = {}
    for j in idx_d:
        pid = int(prec["peptidoform_id"][j])
        t_seq = stripped(parse_proforma(tgt_by_pid[pid]))
        d_seq = stripped(parse_proforma(prec["peptidoform"][j]))
        assert pairs.setdefault(t_seq, d_seq) == d_seq, (
            "one base sequence got two different decoy sequences"
        )
    assert len(set(pairs.values())) == len(pairs)


def test_reverse_recomputes_real_fragment_mz_for_the_decoy_sequence(reverse_library):
    """Decoy fragment m/z must be the real b/y of the decoy sequence.

    The decoy has another sequence than its target, so its fragments must be
    recomputed rather than copied. A copied m/z would describe the
    target's fragments under the decoy's sequence label, making the decoy score
    on real target evidence. Checked against the independent residue table in
    `conftest.py`.
    """
    prec, frag, idx_t, idx_d = _split_by_label(reverse_library["prec"],
                                              reverse_library["frag"])
    tokens_of_cid = {int(prec["candidate_id"][i]): parse_proforma(prec["peptidoform"][i])
                     for i in range(len(prec["candidate_id"]))}
    decoy_cids = {int(prec["candidate_id"][j]) for j in idx_d}
    checked = 0
    for k, cid in enumerate(frag["candidate_id"]):
        cid = int(cid)
        if cid not in decoy_cids:
            continue
        expect = fragment_mz(tokens_of_cid[cid], frag["ion_type"][k],
                             int(frag["ordinal"][k]), int(frag["frag_charge"][k]))
        got = float(frag["mz"][k])
        ppm = 1e6 * abs(got - expect) / expect
        assert ppm < 5.0, (
            "decoy fragment {} of candidate {} is {:.1f} ppm from the b/y m/z of "
            "the decoy sequence".format(frag["name"][k], cid, ppm)
        )
        checked += 1
    assert checked == len(TARGETS) * len(FRAGMENTS)


def test_reverse_aborts_when_the_library_mz_disagree_with_its_calculator(tmp_path):
    """A library whose fragment m/z fail the 5 ppm check must abort the build.

    The calculator check catches a mis-parsed peptidoform, an unmapped
    modification mass, or a library exported at the wrong charge convention. If
    the build continued, the decoys would sit at masses no real spectrum
    contains, the decoy score distribution would collapse, and the run would
    report near-zero FDR at every threshold.
    """
    lib = _write_target_library(tmp_path, mz_scale=1.0 + 100e-6)
    rc, out, err = run_worker(
        "make_reverse_decoys.py", lib["prec"], lib["frag"],
        tmp_path / "prec_out.parquet", tmp_path / "frag_out.parquet",
    )
    assert rc != 0
    assert "ABORT" in out + err
    assert not (tmp_path / "prec_out.parquet").exists()


ANNOTATION_COLUMNS = ["protein_ids", "protein_names", "genes"]


def _annotation(protein):
    """The fixture's annotation of a `PROT<n>_ECOLI` protein."""
    n = protein.replace("PROT", "").replace("_ECOLI", "")
    return (f"P{int(n):05d}", protein, f"gene{n}")


def _with_annotation_columns(lib, directory):
    """`lib` with the three annotation columns, written into `directory`."""
    t = pq.read_table(str(lib["prec"]))
    cells = [_annotation(p) for p in t.column("protein").to_pylist()]
    for j, c in enumerate(ANNOTATION_COLUMNS):
        t = t.append_column(c, pa.array([x[j] for x in cells], pa.string()))
    directory.mkdir(parents=True, exist_ok=True)
    prec = directory / "target_precursors.parquet"
    pq.write_table(t, str(prec), compression="snappy")
    return {**lib, "prec": prec}


def _assert_decoys_carry_their_targets_annotations(prec_path):
    """Every row has the annotation columns (utf8, no null); every target has its protein's
    annotation and every decoy exactly its target's, paired on (peptidoform_id, charge)."""
    t = pq.read_table(str(prec_path))
    for c in ANNOTATION_COLUMNS:
        f = t.schema.field(c)
        assert pa.types.is_string(f.type), f"{c} is {f.type}, the engine reads utf8 only"
        assert t.column(c).null_count == 0
    rows = t.to_pydict()
    by_pair = {}
    for i, lab in enumerate(rows["label"]):
        cell = tuple(rows[c][i] for c in ANNOTATION_COLUMNS)
        by_pair.setdefault((rows["peptidoform_id"][i], rows["charge"][i]), {})[lab] = cell
        if lab == "target":
            assert cell == _annotation(rows["protein"][i])
    assert by_pair and all(v["decoy"] == v["target"] for v in by_pair.values())


def test_the_annotation_columns_travel_with_every_target_onto_its_decoy(target_library, tmp_path):
    """The protein annotations are columns of the precursor table (#299): the builder keeps
    them on every target and copies them onto its decoy, so the output library carries them
    wherever it is copied, and writes no side table."""
    src = _with_annotation_columns(target_library, tmp_path / "in")
    out = _build("make_reverse_decoys.py", src, tmp_path / "out")
    _assert_decoys_carry_their_targets_annotations(out["prec"])
    assert_string_columns_utf8(out["prec"])
    assert not (out["prec"].parent / "protein_annotations.parquet").exists()
    assert "carried in the library's protein_ids" in out["stdout"]


def test_a_side_table_beside_an_older_library_becomes_its_columns(tmp_path):
    """A target library imported before the columns existed has its annotations in the side
    table beside it; the builder writes them into its output's columns, decoys taking their
    targets' values. With no side table, a stale one at the output belongs to another
    library and is removed, and the output has no annotation columns."""
    import sys

    sys.path.insert(0, str(SCRIPTS))
    from _lib_io import write_protein_annotations

    (tmp_path / "in").mkdir()
    src = _write_target_library(tmp_path / "in")
    proteins = sorted(set(read_columns(src["prec"])["protein"]))
    ann = tmp_path / "in" / "protein_annotations.parquet"
    write_protein_annotations(
        {p: dict(zip(ANNOTATION_COLUMNS, _annotation(p))) for p in proteins}, ann)
    out = _build("make_reverse_decoys.py", src, tmp_path / "out")
    _assert_decoys_carry_their_targets_annotations(out["prec"])
    stale = out["prec"].parent / "protein_annotations.parquet"
    assert not stale.exists(), "the output carries columns, not a side table"

    ann.unlink()
    stale.write_bytes(ann.parent.joinpath("target_precursors.parquet").read_bytes())
    out = _build("make_reverse_decoys.py", src, tmp_path / "out")
    assert not stale.exists()
    assert not set(ANNOTATION_COLUMNS) & set(pq.read_schema(str(out["prec"])).names)
