"""Contract tests for `scripts/make_shift_decoys.py` and
`scripts/make_reverse_decoys.py` (offline DIA-NN recipe).

Need only pandas and pyarrow, so they always run. These two scripts create the
null population the whole FDR rests on, and they are the step that makes an
imported library index-valid: both re-sort by `precursor_mz` and reassign
contiguous `candidate_id`, which is what satisfies `index.rs load()`.

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
)

DELTA_CH2 = 14.015650

# (peptidoform, charge). PEPTIDEK appears twice so the builders see two charge
# rows of one base sequence, which may legitimately share a decoy sequence.
# AKAK reverses onto itself, and AGWK/WGAK reverse onto each other, so all three
# force the collision-resolution path.
TARGETS = [
    ("PEPTIDEK", 2),
    ("PEPTIDEK", 3),
    ("SAMPLER", 2),
    ("AKAK", 2),
    ("AGWK", 2),
    ("WGAK", 2),
    ("SC[Carbamidomethyl]ENICK", 2),
]
COLLIDING = {"AKAK", "AGWK", "WGAK"}
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
    stdout, _ = run_worker_ok(
        script, target_library["prec"], target_library["frag"], prec, frag
    )
    return {"prec": prec, "frag": frag, "stdout": stdout}


@pytest.fixture(scope="module")
def shift_library(target_library, tmp_path_factory):
    return _build("make_shift_decoys.py", target_library,
                  tmp_path_factory.mktemp("shift_out"))


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


@pytest.mark.parametrize("builder", ["shift", "reverse"])
def test_output_library_is_index_valid(builder, shift_library, reverse_library):
    """Contiguous ids, ascending `precursor_mz`, both labels present.

    A decoy builder is the last step before a search, so a violation here is the
    one that actually reaches `index.rs load()` and aborts. Both builders
    concatenate target and decoy, re-sort by m/z and reassign ids precisely to
    satisfy this; a reordering added afterwards would break it silently.
    """
    lib = shift_library if builder == "shift" else reverse_library
    assert_library_load_invariants(lib["prec"], lib["frag"], require_both_labels=True)


@pytest.mark.parametrize("builder", ["shift", "reverse"])
def test_every_target_has_exactly_one_decoy(builder, target_library, shift_library,
                                            reverse_library):
    """Target and decoy populations must be paired one to one.

    `q = (decoys + 1) / max(1, targets)` assumes the two populations are
    exchangeable. Retaining a target whose decoy could not be built, or emitting
    two decoys for one target, biases every q-value in the run, and nothing
    downstream can detect it.
    """
    lib = shift_library if builder == "shift" else reverse_library
    prec, _, idx_t, idx_d = _split_by_label(lib["prec"], lib["frag"])
    assert len(idx_t) == len(TARGETS)
    assert len(idx_d) == len(idx_t)
    pid_t = [int(prec["peptidoform_id"][i]) for i in idx_t]
    pid_d = [int(prec["peptidoform_id"][i]) for i in idx_d]
    assert sorted(pid_t) == sorted(pid_d), (
        "decoys do not map one to one onto the targets by peptidoform_id"
    )
    assert len(set(pid_d)) == len(pid_d), "a target got more than one decoy"


@pytest.mark.parametrize("builder", ["shift", "reverse"])
def test_decoy_keeps_the_target_precursor_mz_and_irt(builder, shift_library,
                                                     reverse_library):
    """The decoy must co-isolate and co-elute with its target.

    Precursor m/z and iRT are what put the decoy in the same isolation window at
    the same retention time, so it experiences the same chimeric interference.
    A decoy at a different m/z or RT is an easier problem than the target and
    the FDR it produces is optimistic.
    """
    lib = shift_library if builder == "shift" else reverse_library
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


@pytest.mark.parametrize("script", ["make_shift_decoys.py", "make_reverse_decoys.py"])
def test_builder_is_deterministic(script, target_library, tmp_path):
    """Two builds of the same library must be byte-for-byte equivalent.

    `make_reverse_decoys.py` resolves collisions with a Fisher-Yates seeded from
    a process-independent FNV-1a hash rather than Python's randomized builtin
    `hash`, so a rebuild does not need `PYTHONHASHSEED`. If that regressed, two
    library builds would carry different decoys and no result computed against
    one of them would be reproducible against the other.
    """
    a = _build(script, target_library, tmp_path / "a")
    b = _build(script, target_library, tmp_path / "b")
    for key in ("prec", "frag"):
        ca, cb = read_columns(a[key]), read_columns(b[key])
        assert list(ca) == list(cb)
        for col in ca:
            if isinstance(ca[col], list):
                assert ca[col] == cb[col], "{} differs between builds".format(col)
            else:
                assert np.allclose(np.asarray(ca[col], dtype=float),
                                   np.asarray(cb[col], dtype=float), atol=0, rtol=0)


@pytest.mark.parametrize("builder", ["shift", "reverse"])
def test_output_string_columns_are_utf8(builder, shift_library, reverse_library):
    """String columns must be arrow `utf8`, never `large_utf8`.

    Both builders rely on the pandas default instead of casting to
    `pa.string()`, so on a pandas whose string dtype is arrow-backed the emitted
    library is rejected by the engine with "column 'peptidoform' is not utf8".
    """
    lib = shift_library if builder == "shift" else reverse_library
    for path in (lib["prec"], lib["frag"]):
        assert_string_columns_utf8(path)


# ---------------------------------------------------------------------------
# make_shift_decoys: the CH2 fragment shift
# ---------------------------------------------------------------------------


def test_shift_decoy_names_map_one_to_one_onto_the_targets(shift_library):
    """Every decoy peptidoform and protein is `DECOY_` + the target's.

    The `DECOY_` prefix is how the engine, `deeplc_finetune.py` and the report
    stage recognise a decoy. A decoy whose name does not map back to its target
    cannot be paired for picked target-decoy competition at the peptide level.
    """
    prec, _, idx_t, idx_d = _split_by_label(shift_library["prec"],
                                            shift_library["frag"])
    targets = {int(prec["peptidoform_id"][i]): (prec["peptidoform"][i],
                                                prec["protein"][i]) for i in idx_t}
    seen = set()
    for j in idx_d:
        pid = int(prec["peptidoform_id"][j])
        tp, tprot = targets[pid]
        assert prec["peptidoform"][j] == "DECOY_" + tp
        assert prec["protein"][j] == "DECOY_" + tprot
        seen.add(prec["peptidoform"][j])
    assert len(seen) == len(idx_d) - 1, (
        "expected exactly one duplicate decoy name, from the two charge rows of "
        "PEPTIDEK; the shift builder names decoys after the peptidoform only"
    )


def test_shift_moves_b_down_and_y_up_by_one_ch2_per_charge(shift_library):
    """b ions shift by `-DELTA/z` and y ions by `+DELTA/z`, net precursor zero.

    The net-zero precursor shift is the whole point: the decoy stays in the same
    isolation window while none of its fragments coincide with the target's. If
    the sign or the charge division were wrong, decoy fragments would land on
    real target fragment masses and the null would absorb real signal.
    """
    prec, frag, idx_t, idx_d = _split_by_label(shift_library["prec"],
                                              shift_library["frag"])
    pid_of_cid = {int(prec["candidate_id"][i]): int(prec["peptidoform_id"][i])
                  for i in range(len(prec["candidate_id"]))}
    label_of_cid = {int(prec["candidate_id"][i]): prec["label"][i]
                    for i in range(len(prec["candidate_id"]))}

    target_mz, decoy_mz = {}, {}
    for k, cid in enumerate(frag["candidate_id"]):
        cid = int(cid)
        key = (pid_of_cid[cid], frag["name"][k])
        (target_mz if label_of_cid[cid] == "target" else decoy_mz)[key] = (
            float(frag["mz"][k]), frag["ion_type"][k], int(frag["frag_charge"][k])
        )
    assert set(target_mz) == set(decoy_mz)
    assert len(decoy_mz) == len(TARGETS) * len(FRAGMENTS)
    for key, (tmz, ion, z) in target_mz.items():
        dmz = decoy_mz[key][0]
        expect = tmz + (-1.0 if ion == "b" else 1.0) * DELTA_CH2 / z
        assert dmz == pytest.approx(expect, abs=1e-6), (
            "fragment {} shifted by {} instead of {}".format(key, dmz - tmz, expect - tmz)
        )


def test_shift_copies_intensities_and_ion_annotation(shift_library):
    """Predicted intensities and ion annotation must be copied ion for ion.

    The decoy is meant to differ from its target only in fragment m/z. A decoy
    with different intensities is scored by a different feature distribution,
    so the target/decoy score comparison is no longer apples to apples.
    """
    prec, frag, _, _ = _split_by_label(shift_library["prec"], shift_library["frag"])
    label_of_cid = {int(prec["candidate_id"][i]): prec["label"][i]
                    for i in range(len(prec["candidate_id"]))}
    for k, cid in enumerate(frag["candidate_id"]):
        if label_of_cid[int(cid)] == "decoy":
            assert float(frag["predicted_intensity"][k]) == pytest.approx(0.5)
            assert frag["ion_type"][k] in ("b", "y")
            assert int(frag["frag_charge"][k]) in (1, 2)


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
    tgt = {stripped(parse_proforma(prec["peptidoform"][i])) for i in idx_t}
    dec = {stripped(parse_proforma(prec["peptidoform"][j])) for j in idx_d}
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

    Unlike the shift builder, this one changes the sequence, so the fragments
    must be recomputed rather than copied. A copied m/z would describe the
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
