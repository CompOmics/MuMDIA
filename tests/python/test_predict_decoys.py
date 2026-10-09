"""Contract tests for `scripts/predict_decoys.py` (decoys with DIA-NN-predicted spectra).

DIA-NN is not available in CI, so the prediction is stood in for by a table in the
`import_diann_lib.py` output schema, written here. What is checked is the part MuMDIA owns:
which decoys are sent for prediction, that shift decoys are refused, and that the merge
keeps the library paired, index-valid, and made of the target fragments plus the
predicted decoy fragments.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import assert_library_load_invariants, read_columns, run_worker, run_worker_ok
from test_decoy_builders import _write_target_library

PRED_INTENSITY = 0.9


@pytest.fixture(scope="module")
def libraries(tmp_path_factory):
    d = tmp_path_factory.mktemp("predict_decoys")
    tgt = _write_target_library(d)
    out = {}
    for script, name in (("make_reverse_decoys.py", "rev"), ("make_shift_decoys.py", "shift")):
        prec, frag = d / f"{name}_prec.parquet", d / f"{name}_frag.parquet"
        run_worker_ok(script, tgt["prec"], tgt["frag"], prec, frag)
        out[name] = (prec, frag)
    out["dir"] = d
    return out


def _fake_prediction(rev_prec, directory, skip=()):
    """A stand-in for DIA-NN's prediction of the decoy sequences: one row per decoy
    (DECOY_ stripped, as import_diann_lib.py would write it), four fragments each at a
    recognisable intensity. Decoys whose sequence is in `skip` are left unpredicted."""
    prec = read_columns(rev_prec)
    rows = [(pf[6:], z) for pf, z, lab in zip(prec["peptidoform"], prec["charge"], prec["label"])
            if lab == "decoy" and pf[6:] not in skip]
    pp = pa.table({
        "candidate_id": pa.array(range(len(rows)), pa.uint32()),
        "peptidoform": pa.array([r[0] for r in rows], pa.string()),
        "charge": pa.array([r[1] for r in rows], pa.int32()),
    })
    n = 4
    frag = pa.table({
        "candidate_id": pa.array(np.repeat(np.arange(len(rows)), n), pa.uint32()),
        "mz": pa.array(np.tile(np.arange(n) * 100.0 + 200.0, len(rows)), pa.float64()),
        "predicted_intensity": pa.array([PRED_INTENSITY] * (n * len(rows)), pa.float32()),
        "name": pa.array([f"y{k + 1}" for k in range(n)] * len(rows), pa.string()),
        "ion_type": pa.array(["y"] * (n * len(rows)), pa.string()),
        "ordinal": pa.array(list(range(1, n + 1)) * len(rows), pa.int32()),
        "frag_charge": pa.array([1] * (n * len(rows)), pa.int32()),
        "cardinality": pa.array([1] * (n * len(rows)), pa.int32()),
    })
    p, f = directory / "pred_prec.parquet", directory / "pred_frag.parquet"
    pq.write_table(pp, str(p))
    pq.write_table(frag, str(f))
    return p, f, n


def test_shift_decoys_are_refused(libraries, tmp_path):
    prec, frag = libraries["shift"]
    rc, _, err = run_worker("predict_decoys.py", "tsv", prec, frag, tmp_path / "x.tsv")
    assert rc != 0
    assert "own sequence" in err


def test_tsv_lists_every_decoy_once_in_dia_nn_notation(libraries, tmp_path):
    prec, frag = libraries["rev"]
    out = tmp_path / "decoys.tsv"
    run_worker_ok("predict_decoys.py", "tsv", prec, frag, out)
    lines = out.read_text().splitlines()
    header, body = lines[0].split("\t"), [line.split("\t") for line in lines[1:]]
    p = read_columns(prec)
    decoys = {(pf, z) for pf, z, lab in zip(p["peptidoform"], p["charge"], p["label"])
              if lab == "decoy"}
    assert len(body) == len(decoys)
    mods = [row[header.index("ModifiedPeptide")] for row in body]
    assert all(m.startswith("_") and m.endswith("_") and "DECOY_" not in m for m in mods)
    assert not any("[" in m and "UniMod:" not in m for m in mods)


def test_merge_keeps_targets_and_takes_the_predicted_decoy_fragments(libraries, tmp_path):
    prec, frag = libraries["rev"]
    p = read_columns(prec)
    decoy_seqs = sorted(pf[6:] for pf, lab in zip(p["peptidoform"], p["label"]) if lab == "decoy")
    skipped = decoy_seqs[0]
    pp, pf, n_pred = _fake_prediction(prec, tmp_path, skip={skipped})
    op, of = tmp_path / "lib_precursors.parquet", tmp_path / "lib_fragments.parquet"
    run_worker_ok("predict_decoys.py", "merge", prec, frag, pp, pf, op, of)

    out_prec, out_frag = assert_library_load_invariants(op, of, check_string_type=True)
    labels = out_prec.column("label").to_pylist()
    pforms = out_prec.column("peptidoform").to_pylist()
    pids = out_prec.column("peptidoform_id").to_pylist()
    # Paired: the unpredicted decoy is gone together with its target.
    assert labels.count("target") == labels.count("decoy")
    assert "DECOY_" + skipped not in pforms
    by_pid = {}
    for pid, lab in zip(pids, labels):
        by_pid.setdefault(pid, []).append(lab)
    assert all(sorted(v) == ["decoy", "target"] for v in by_pid.values())

    cid = np.asarray(out_frag.column("candidate_id").to_pylist())
    inten = np.asarray(out_frag.column("predicted_intensity").to_pylist())
    is_decoy = np.asarray([lab == "decoy" for lab in labels])
    assert np.all(inten[is_decoy[cid]] == np.float32(PRED_INTENSITY))
    assert np.all(inten[~is_decoy[cid]] == np.float32(0.5))
    counts = np.bincount(cid, minlength=len(labels))
    assert np.all(counts[is_decoy] == n_pred)
    assert counts.tolist() == out_prec.column("n_fragments").to_pylist()
    assert np.all(np.diff(cid) >= 0), "fragments are not sorted by candidate_id"
