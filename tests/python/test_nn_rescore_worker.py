"""Contract tests for `scripts/nn_rescore_worker.py` (Stage F, `nn_torch`).

Skipped without torch. The contract under test is the one `CLAUDE.md` states
and `align_sidecar_scores` (`rescore.rs:1046-1082`) enforces: every input row
gets exactly one finite out-of-fold score, the output row count equals the
input row count, and no row is silently dropped. Under the production default
`rescore.strict = true` a violation aborts the run; with strict off the
requested classifier is silently replaced by `native_tda`, which is worse,
because the artifact then reports a model nobody asked for.

The PSM pool is tiny and the `MUMDIA_NN_*` knobs are turned down so the whole
file runs in seconds. Scores from a retrained MLP are only approximately
reproducible (`docs/13_sidecars.md`, determinism), so nothing here asserts an
exact score; the assertions are coverage, finiteness, and the direction of
separation.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    assert_complete_finite_coverage,
    importorskip_any,
    read_columns,
    run_worker,
    run_worker_ok,
    synthetic_psms,
    write_features_parquet,
    write_pin,
)

pytestmark = pytest.mark.usefixtures("scripts_dir")

# Small, deterministic, and far below the streaming threshold, so the backend
# is chosen by MUMDIA_NN_STREAM alone rather than by the size heuristic.
FAST_ENV = {
    "MUMDIA_NN_FOLDS": "2",
    "MUMDIA_NN_ITERS": "2",
    # ~10 minibatches per epoch on this pool. Far fewer gradient steps leave the
    # MLP at its random initialisation, where the sign of the separation is
    # arbitrary; the assertions below are about the contract, not about how few
    # steps the worker can survive.
    "MUMDIA_NN_EPOCHS": "40",
    "MUMDIA_NN_HIDDEN": "16,8",
    "MUMDIA_NN_BATCH": "64",
    "MUMDIA_NN_SEEDS": "1",
    "MUMDIA_NN_TRAIN_FDR": "0.05",
    "MUMDIA_NN_THREADS": "1",
    "MUMDIA_NN_DEVICE": "cpu",
    "MUMDIA_NN_FEATURES": "",
    "MUMDIA_NN_STREAM_GB": "64",
    "MUMDIA_NN_CHUNK": "128",
    "MUMDIA_NN_INIT_TOPK": "0",
    "PYTHONUTF8": "1",
}


@pytest.fixture(scope="module")
def torch_available():
    importorskip_any("torch", "nn_torch rescoring needs a usable torch")
    importorskip_any("pandas", "nn_rescore_worker reads the PIN with pandas")


@pytest.fixture(scope="module")
def psms():
    return synthetic_psms(n_targets=200, n_decoys=200, n_features=5, seed=0)


def _env(**overrides):
    env = dict(FAST_ENV)
    env.update({k: str(v) for k, v in overrides.items()})
    return env


def _assert_rescore_contract(out_path, psms, label):
    table = assert_complete_finite_coverage(out_path, psms["n"], sidecar=label)
    cols = read_columns(out_path)
    assert "q_value" in cols, "the sidecar must emit a q_value column"
    assert np.allclose(np.asarray(cols["q_value"], dtype=float), 0.0), (
        "q_value must be written as zeros; the Rust caller computes q itself "
        "(docs/13_sidecars.md), and a sidecar-computed q would be reported as "
        "the engine's own FDR"
    )
    # `candidate_id` echoes the SpecId tail, i.e. the flat row index, so score i
    # belongs to row i. If that mapping breaks, every PSM keeps another PSM's
    # score and the whole rescore is scrambled without any error.
    order = np.argsort([int(x) for x in cols["candidate_id"]])
    scores = np.asarray(cols["score"], dtype=float)[order]
    is_target = psms["labels"] == 1
    assert scores[is_target].mean() > scores[~is_target].mean(), (
        "targets do not outscore decoys, so the returned scores are not aligned "
        "to the input rows or carry no discrimination at all"
    )
    return table


def test_tsv_pin_scores_every_row_exactly_once(torch_available, psms, tmp_path):
    """Every PIN row must come back with exactly one finite score.

    A short, long, duplicated or non-finite output makes the Rust caller bail
    (`rescore.rs:1046-1082`). Under `rescore.strict = true`, the production
    default, that aborts the run after the whole search has already been paid
    for; under strict off it downgrades to `native_tda` and the artifact
    reports a classifier the user did not request.
    """
    pin = write_pin(tmp_path / "rescore.pin", psms)
    out = tmp_path / "rescore_sidecar_out.parquet"
    stdout, _ = run_worker_ok("nn_rescore_worker.py", pin, out, env=_env(MUMDIA_NN_STREAM=0))
    assert "format=tsv" in stdout
    assert "backend=in-memory" in stdout
    _assert_rescore_contract(out, psms, "nn_rescore_worker[tsv]")


def test_parquet_handoff_scores_every_row_exactly_once(torch_available, psms, tmp_path):
    """The `rescore.handoff = parquet` table must satisfy the same contract.

    Parquet is accepted by this worker only (`rescore.rs:943-959`) and is the
    path used at experiment scale, where a 30 GB text PIN also forced the
    streaming backend. Column names and semantics are identical either way, so
    a divergence here means the two handoffs disagree about which row is which.
    """
    features = write_features_parquet(tmp_path / "rescore.features.parquet", psms)
    out = tmp_path / "rescore_sidecar_out.parquet"
    stdout, _ = run_worker_ok(
        "nn_rescore_worker.py", features, out, env=_env(MUMDIA_NN_STREAM=0)
    )
    assert "format=parquet" in stdout
    _assert_rescore_contract(out, psms, "nn_rescore_worker[parquet]")


def test_streaming_memmap_backend_scores_every_row_and_cleans_up(
    torch_available, psms, tmp_path
):
    """The disk-backed backend must cover every row and delete its memmap.

    The streaming backend is what makes an experiment-wide rescore tractable,
    and it is selected by a size cliff rather than a preference, so it runs
    unattended. A leftover `<out>.feat.mm` is not cosmetic: an orphaned worker
    holding that file made every later rescore fail on a path it did not own
    (`nn_rescore_worker.py:424-425, 719-722`).
    """
    pin = write_pin(tmp_path / "rescore.pin", psms)
    out = tmp_path / "rescore_sidecar_out.parquet"
    stdout, _ = run_worker_ok("nn_rescore_worker.py", pin, out, env=_env(MUMDIA_NN_STREAM=1))
    assert "backend=stream(memmap)" in stdout
    _assert_rescore_contract(out, psms, "nn_rescore_worker[stream]")
    leftover = tmp_path / "rescore_sidecar_out.parquet.feat.mm"
    assert not leftover.exists(), "the streaming backend left its memmap behind"


def test_single_class_pin_fails_instead_of_scoring(torch_available, tmp_path):
    """A PIN with no decoys must exit nonzero rather than emit scores.

    Target-decoy competition is meaningless without both populations, and a
    worker that returned scores anyway would hand the engine a complete,
    finite, entirely uncalibrated output that passes every coverage check and
    produces a q-value column with no null behind it.
    """
    psms = synthetic_psms(n_targets=200, n_decoys=0, n_features=5, seed=1)
    pin = write_pin(tmp_path / "targets_only.pin", psms)
    out = tmp_path / "out.parquet"
    rc, _, err = run_worker(
        "nn_rescore_worker.py", pin, out, env=_env(MUMDIA_NN_STREAM=0)
    )
    assert rc != 0, "a single-class PIN was accepted"
    assert "targets and decoys" in err or "both targets and decoys" in err
    assert not out.exists(), "a failing worker still wrote an output table"


def test_unknown_feature_subset_fails_before_training(torch_available, psms, tmp_path):
    """`MUMDIA_NN_FEATURES` naming an absent column must abort immediately.

    The subset is applied before either backend reads the PIN, so a typo would
    otherwise silently rescore on fewer features than intended and quietly
    change the identification count of a benchmark arm.
    """
    pin = write_pin(tmp_path / "rescore.pin", psms)
    out = tmp_path / "out.parquet"
    rc, _, err = run_worker(
        "nn_rescore_worker.py", pin, out,
        env=_env(MUMDIA_NN_STREAM=0, MUMDIA_NN_FEATURES="feat_0,not_a_feature"),
    )
    assert rc != 0
    assert "MUMDIA_NN_FEATURES" in err


def _import_worker():
    """Import `nn_rescore_worker` as a module, for the helpers that need no torch.

    The worker imports torch lazily inside `main`, so the module itself loads with
    numpy and pyarrow alone. That keeps this test running in CI, where the rest of
    this file skips.
    """
    import importlib.util
    import pathlib

    importorskip_any("numpy", "folds_for returns a numpy array")
    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "nn_rescore_worker.py"
    spec = importlib.util.spec_from_file_location("nn_rescore_worker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_fold_keys_pair_a_target_with_its_decoy():
    """A target and its paired decoy must share a CV fold.

    `docs/11` states this worker uses "the same CV-fold scheme" as `percolator_lite`,
    which keys on `base_peptide_id`. It did not: the fold came from
    `md5(strip_pep(Peptide))`, and `strip_pep` leaves the `DECOY_` marker in place, so
    `DECOY_PEPTIDE` and `PEPTIDE` hashed apart. Stripping the marker would only have
    fixed a shift-decoy library; a reverse decoy's peptidoform is the reversed sequence,
    so no string derived from it can reach its target. The engine now writes
    `base_peptide_id` per row and names the file in `MUMDIA_NN_FOLD_KEYS`.
    """
    w = _import_worker()
    np = pytest.importorskip("numpy")

    # Rows 0 and 1 are a target and its REVERSE decoy: different sequences, one
    # base_peptide_id. Row 2 is an unrelated peptide.
    keys = np.array([7, 7, 9], dtype=np.uint32)
    peptides = ["-.PEPTIDEK.-", "-.DECOY_KEDITPEP.-", "-.SAMPLER.-"]

    folds = w.folds_for(peptides, keys, 3)
    assert folds[0] == folds[1], (
        "a target and its paired decoy must train in the same fold, else the model "
        "sees one of the pair while scoring the other"
    )
    assert len(folds) == 3
    assert folds.dtype == np.int16

    # The hashed fallback is what the fold-key file exists to replace: it splits the
    # pair. Asserted so the regression is visible if the file is ever dropped.
    hashed = w.folds_for(peptides, None, 3)
    assert hashed[0] != hashed[1]


def test_fold_keys_respect_the_streaming_row_offset():
    """The chunked backend passes a flat row offset; the keys must be sliced by it.

    Without the offset the streaming backend would fold every chunk as if it started
    at row 0, so the same PSM would land in a different fold depending on which
    backend the size heuristic chose.
    """
    w = _import_worker()
    np = pytest.importorskip("numpy")

    keys = np.arange(10, dtype=np.uint32)
    whole = w.folds_for([f"p{i}" for i in range(10)], keys, 3)
    chunk = w.folds_for([f"p{i}" for i in range(4, 7)], keys, 3, off=4)
    assert list(chunk) == list(whole[4:7])
