"""Contract tests for `scripts/entrapment_worker.py` (Stage F, `entrapment`).

Skipped without scikit-learn. Selecting entrapment flips the engine's internal
`QMode` from `Decoy` to `Entrapment` (`rescore.rs:145`), so every q level (PSM,
per-run, peptide, protein group, precursor) is then computed against the
real-target-vs-spike-in null. The scores this worker returns are therefore the
input to the reported FDR, and the readback is validated with the same
exact/unique/finite rule as the other sidecars, keyed on `row_id`
(`rescore.rs:729-732`).
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import (
    assert_complete_finite_coverage,
    importorskip_any,
    read_columns,
    run_worker,
    run_worker_ok,
)

N_REAL = 120
N_ENTRAP = 120
N_DECOY = 60
N_ROWS = N_REAL + N_ENTRAP + N_DECOY
N_FEATURES = 4
N_GROUPS = 30


@pytest.fixture(scope="module")
def sklearn_available():
    importorskip_any("sklearn", "the entrapment rescorer needs a usable scikit-learn")


def _build_input(path, with_row_id=True, n_entrap=N_ENTRAP, seed=0):
    """Write `entrapment_in.parquet` in the layout `rescore.rs:693-714` writes.

    `candidate_id` deliberately collides (two rows share each value), the way it
    does in a competed multi-run pool where it is the library index. Only
    `row_id` is unique, so a worker keying its output on `candidate_id` would
    lose half the rows.
    """
    n_rows = N_REAL + n_entrap + N_DECOY
    rng = np.random.default_rng(seed)
    is_entrapment = np.zeros(n_rows, dtype=np.int32)
    is_entrapment[N_REAL:N_REAL + n_entrap] = 1
    is_decoy = np.zeros(n_rows, dtype=np.int32)
    is_decoy[N_REAL + n_entrap:] = 1

    feats = rng.normal(0.0, 1.0, size=(n_rows, N_FEATURES))
    feats[:N_REAL, 0] += 3.0          # real targets separate from the spike-in
    feats[N_REAL + n_entrap:, 0] -= 1.0  # decoys sit below both

    cols = {
        "candidate_id": pa.array(
            np.arange(n_rows, dtype=np.uint32) % np.uint32(max(1, n_rows // 2)),
            pa.uint32(),
        ),
        "base_peptide_id": pa.array(
            np.arange(n_rows, dtype=np.uint32) % np.uint32(N_GROUPS), pa.uint32()
        ),
        "is_entrapment": pa.array(is_entrapment, pa.int32()),
        "is_decoy": pa.array(is_decoy, pa.int32()),
    }
    if with_row_id:
        cols = {"row_id": pa.array(np.arange(n_rows, dtype=np.uint32), pa.uint32()),
                **cols}
    for j in range(N_FEATURES):
        cols["feat_{}".format(j)] = pa.array(feats[:, j], pa.float64())
    pq.write_table(pa.table(cols), str(path), compression="snappy")
    return {
        "path": path,
        "n_rows": n_rows,
        "is_entrapment": is_entrapment.astype(bool),
        "is_decoy": is_decoy.astype(bool),
        "candidate_id": [int(x) for x in cols["candidate_id"].to_pylist()],
    }


def test_scores_every_row_including_decoys(sklearn_available, tmp_path):
    """Every input row must get exactly one finite score, decoys included.

    Decoys are excluded from training (`entrapment_worker.py:80-83`) but must
    still be SCORED, by the final model fit on all non-decoy PSMs. A NaN or a
    missing decoy row makes the Rust readback bail, and if it did not, the
    decoy population behind the reported q-values would be silently truncated.
    """
    spec = _build_input(tmp_path / "entrapment_in.parquet")
    out = tmp_path / "entrapment_out.parquet"
    run_worker_ok("entrapment_worker.py", spec["path"], out, 3)

    assert_complete_finite_coverage(
        out, spec["n_rows"], key="row_id", sidecar="entrapment_worker"
    )
    cols = read_columns(out)
    order = np.argsort([int(x) for x in cols["row_id"]])
    scores = np.asarray(cols["score"], dtype=float)[order]
    assert np.isfinite(scores[spec["is_decoy"]]).all(), "a decoy row was left NaN"


def test_row_id_not_candidate_id_is_the_readback_key(sklearn_available, tmp_path):
    """`candidate_id` must be echoed per row, never used to deduplicate.

    `candidate_id` is the library index and repeats across competed runs, so
    keying the score map on it collides and later runs overwrite earlier ones.
    The separate `row_id` exists for exactly that reason
    (`docs/13_sidecars.md`, "Row index vs candidate_id").
    """
    spec = _build_input(tmp_path / "entrapment_in.parquet")
    out = tmp_path / "entrapment_out.parquet"
    run_worker_ok("entrapment_worker.py", spec["path"], out, 3)

    cols = read_columns(out)
    assert len(set(spec["candidate_id"])) < spec["n_rows"], (
        "the fixture no longer contains colliding candidate_id values, so this "
        "test cannot detect a candidate_id-keyed output"
    )
    row_id = [int(x) for x in cols["row_id"]]
    echoed = dict(zip(row_id, (int(x) for x in cols["candidate_id"])))
    assert len(row_id) == spec["n_rows"]
    for i, cid in enumerate(spec["candidate_id"]):
        assert echoed[i] == cid, "candidate_id was not echoed for row {}".format(i)


def test_out_of_fold_scores_separate_real_targets_from_spike_ins(
    sklearn_available, tmp_path
):
    """Real targets must outscore the spike-in negatives.

    Higher is better everywhere in the engine, and `entrapment_q` compares real
    targets against the spike-in null. An inverted or flat score would report
    the entrapment population as the confident one, i.e. an FDR computed the
    wrong way round with no error anywhere.
    """
    spec = _build_input(tmp_path / "entrapment_in.parquet")
    out = tmp_path / "entrapment_out.parquet"
    run_worker_ok("entrapment_worker.py", spec["path"], out, 3)

    cols = read_columns(out)
    order = np.argsort([int(x) for x in cols["row_id"]])
    scores = np.asarray(cols["score"], dtype=float)[order]
    real = ~(spec["is_entrapment"] | spec["is_decoy"])
    assert scores[real].mean() > scores[spec["is_entrapment"]].mean()


def test_missing_row_id_falls_back_to_positional_index(sklearn_available, tmp_path):
    """An input without `row_id` must still be scored positionally.

    The worker keeps that fallback (`entrapment_worker.py:75`) so an older
    feature table remains readable. If the fallback disappeared, the run would
    fail on a KeyError inside the sidecar rather than on anything the caller
    can act on.
    """
    spec = _build_input(tmp_path / "no_row_id.parquet", with_row_id=False)
    out = tmp_path / "entrapment_out.parquet"
    run_worker_ok("entrapment_worker.py", spec["path"], out, 3)
    assert_complete_finite_coverage(
        out, spec["n_rows"], key="row_id", sidecar="entrapment_worker"
    )


def test_single_class_input_fails_instead_of_scoring(sklearn_available, tmp_path):
    """With no spike-in rows the worker must exit nonzero.

    Training real-target-vs-spike-in is impossible without both classes, and a
    worker that returned constants would give the engine a complete, finite
    output that passes every coverage check while `QMode::Entrapment` computes
    q-values against an empty null. The Rust side guards this too
    (`rescore.rs:218-234`), so both layers must keep refusing.
    """
    spec = _build_input(tmp_path / "targets_only.parquet", n_entrap=0)
    out = tmp_path / "out.parquet"
    rc, _, err = run_worker("entrapment_worker.py", spec["path"], out, 3)
    assert rc != 0
    assert "entrapment" in err.lower()
    assert not out.exists()
