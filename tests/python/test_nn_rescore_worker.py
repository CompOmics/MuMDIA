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


def test_a_short_fold_key_file_is_refused_rather_than_leaving_rows_unfolded():
    """A fold-key companion shorter than the PIN must stop the run.

    Numpy slicing past the end returns a SHORT array instead of raising, and a short
    `fold` puts the tail rows in no fold at all: `np.where(fold == f)` cannot reach
    them, they are never scored, and they keep the zero initialiser, which the final
    rank-normalisation turns into a plausible tied mid-rank score. The Rust caller's
    completeness contract is satisfied by that, so `rescore.strict` does not catch it
    either (docs/31 F5).
    """
    w = _import_worker()
    np = pytest.importorskip("numpy")

    keys = np.arange(6, dtype=np.uint32)
    peptides = [f"p{i}" for i in range(10)]
    with pytest.raises(SystemExit) as exc:
        w.folds_for(peptides, keys, 3)
    assert "MUMDIA_NN_FOLD_KEYS" in str(exc.value)
    # A companion that covers the rows is unaffected, at an offset too.
    full = np.arange(10, dtype=np.uint32)
    assert len(w.folds_for(peptides, full, 3)) == 10
    assert len(w.folds_for(peptides[7:], full, 3, off=7)) == 3
    with pytest.raises(SystemExit):
        w.folds_for(peptides[7:], keys, 3, off=7)


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


# ------------------------------------------------- streaming threshold from free memory

def _load_nn_worker():
    import importlib.util

    from conftest import SCRIPTS

    spec = importlib.util.spec_from_file_location(
        "mumdia_nn_worker_threshold", SCRIPTS / "nn_rescore_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stream_threshold_is_sized_from_free_memory_and_never_below_the_old_default():
    """The fixed 4 GB was wrong at both ends; the replacement may only ever raise it.

    What prompted this: a 4.52 GiB feature matrix on a 96 GiB machine crossed the fixed
    threshold by 13% and took the disk-backed memmap, costing 166 minutes against roughly
    20 in memory. Sizing from free memory fixes that without making a small machine
    stream LESS eagerly than it does today, which is why the historical default is a
    floor rather than a starting point.
    """
    m = _load_nn_worker()

    # The helper reads the real machine, so the size-dependent behaviour is exercised
    # through the arithmetic it uses, which is deterministic.
    def derived(free_gb):
        return free_gb * m._FREE_MULTIPLIER

    # Greater than 1 on purpose: the memmap is a last resort, and between 1x and 2x free
    # memory the operating system's page file is the cheaper way to overflow.
    assert m._FREE_MULTIPLIER > 1, "the memmap must not be preferred over paging"
    assert derived(40) > 40, "a matrix larger than free memory still goes in memory"
    assert derived(90) > 4, "a 90 GiB-free workstation must not be held at 4 GB"
    assert derived(1) < 4, "a machine this small falls back to the floor"

    # The floor holds whatever the arithmetic says, on whatever machine this runs.
    gb, why = m.auto_stream_threshold_gb(4)
    assert gb >= 4
    assert isinstance(why, str) and why, "the threshold must explain where it came from"
    assert m.auto_stream_threshold_gb(99)[0] >= 99, "a higher default is never lowered"


def test_available_ram_is_optional_and_never_fatal():
    """It must degrade to the fixed default rather than raise, on any platform.

    `psutil` is deliberately not a dependency of the rescore environment, so this reads
    the platform directly; anywhere that fails, the caller keeps the old behaviour.
    """
    m = _load_nn_worker()
    free = m.available_ram_bytes()
    assert free is None or (isinstance(free, int) and free > 0)
    # Whatever it returns, the threshold is usable.
    gb, why = m.auto_stream_threshold_gb(4)
    assert gb >= 4 and isinstance(why, str)


# ------------------------------------------ default-path speed-ups are score-identical

# Every speed-up on the default path is meant to leave the output scores byte-identical,
# so each one keeps a switch back to the code it replaced. Running the worker once with
# all of them switched back and once with the defaults, on the same host and seed, must
# give the same bytes. A single-seed count is not a measurement (CLAUDE.md), so nothing
# weaker than exact equality would catch a change that only moves the arithmetic.
LEGACY_ENV = {
    # W6: score the training pool after the last round as well.
    "MUMDIA_NN_FINAL_POOL_SCORE": "1",
    # W1: gather scoring batches with a numpy fancy index.
    "MUMDIA_NN_GATHER": "numpy",
    # W5: scan the init features on one thread.
    "MUMDIA_NN_SCAN_THREADS": "1",
    # W2/W3: the serial load loop, without read-ahead or pre_buffer.
    "MUMDIA_NN_LOAD_THREADS": "0",
    # W4: the full stable sort for every positive selection and decoy order.
    "MUMDIA_NN_SELECT": "full",
}

IDENTITY_ENV = dict(
    FAST_ENV,
    MUMDIA_NN_FOLDS="3",
    MUMDIA_NN_ITERS="3",
    MUMDIA_NN_EPOCHS="8",
    MUMDIA_NN_HIDDEN="16,8",
    MUMDIA_NN_BATCH="128",
    # Run every round, so the last one (the one W6 skips scoring after) is reached.
    MUMDIA_NN_EARLY_STOP="0",
    MUMDIA_NN_NEG_RATIO="2",
    MUMDIA_NN_NEG_SELECT="hybrid",
    MUMDIA_NN_WARM_START="1",
    MUMDIA_NN_WARM_EPOCHS="2",
    MUMDIA_NN_INIT_SAMPLE="2000",
    MUMDIA_NN_CHUNK="700",
    MUMDIA_NN_THREADS="2",
    # One scan thread per 500 sample rows, so the default arm really runs the threaded init
    # scan (W5) on the 2,000-row sample; the production default of 20,000 would keep it
    # serial here.
    MUMDIA_NN_SCAN_ROWS_PER_THREAD="500",
)


def _identity_pool(tmp_path, n=6000, nf=13, row_group=1500, seed=3):
    """A parquet handoff with the awkward cells the load path must treat exactly as before.

    Several row groups (and a CHUNK smaller than one), NaN and infinite cells, a float64
    column with values beyond float32 range, a column with nulls, a constant column, -0.0
    cells, heavily tied columns and subnormal cells: float32 subnormals in an ordinary
    column, a float64 column with values that are normal in float64 but subnormal once
    narrowed to float32 plus float64 subnormals, and a column of zeros and float32
    subnormals only, whose float64 sums flush-to-zero decides. Returns (features path,
    fold-key path, n).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    labels = np.where(rng.random(n) < 0.5, 1, -1).astype(np.int32)
    true = (labels == 1) & (rng.random(n) < 0.4)
    x = rng.normal(0.0, 1.0, (n, nf))
    x[true] += np.linspace(2.5, 0.2, nf)
    x[:, 2] = np.round(x[:, 2], 1)
    x[rng.random(n) < 0.02, 3] = -0.0
    x[rng.random(n) < 0.01, 4] = np.nan
    x[rng.random(n) < 0.005, 5] = np.inf
    x[rng.random(n) < 0.005, 6] = -np.inf
    x[:, 7] = 1.5
    x[rng.random(n) < 0.01, 10] = 1e-40          # float32 subnormal
    x[:, 12] = 0.0
    x[true & (rng.random(n) < 0.5), 12] = 1e-40  # zeros and float32 subnormals only
    x[rng.random(n) < 0.01, 12] = -3e-42
    cols = {
        "SpecId": pa.array(["psm_%d" % i for i in range(n)], pa.string()),
        "Label": pa.array(labels, pa.int32()),
        "ScanNr": pa.array(np.arange(n, dtype=np.int32), pa.int32()),
        "ExpMass": pa.array(np.full(n, 500.0), pa.float64()),
        "CalcMass": pa.array(np.full(n, 500.0), pa.float64()),
    }
    for j in range(nf):
        if j == 8:
            v = x[:, j].copy()
            v[rng.random(n) < 0.005] = 1e300
            cols["f%02d" % j] = pa.array(v, pa.float64())
        elif j == 9:
            cols["f%02d" % j] = pa.array(
                x[:, j].astype(np.float32), pa.float32(), mask=rng.random(n) < 0.01
            )
        elif j == 11:
            v = x[:, j].copy()
            v[rng.random(n) < 0.01] = 1e-40      # normal float64, subnormal in float32
            v[rng.random(n) < 0.01] = -1e-310    # float64 subnormal
            cols["f%02d" % j] = pa.array(v, pa.float64())
        else:
            cols["f%02d" % j] = pa.array(x[:, j].astype(np.float32), pa.float32())
    cols["Peptide"] = pa.array(["-.PEP%dK.-" % (i // 2) for i in range(n)], pa.string())
    cols["Proteins"] = pa.array(["P%d" % (i % 7) for i in range(n)], pa.string())
    features = tmp_path / "identity.features.parquet"
    pq.write_table(pa.table(cols), str(features), row_group_size=row_group,
                   compression="snappy")
    keys = tmp_path / "identity.foldkeys.parquet"
    pq.write_table(
        pa.table({"fold_key": pa.array((np.arange(n) // 2).astype(np.uint32), pa.uint32())}),
        str(keys),
    )
    return features, keys, n


def _scores_by_row(out_path):
    cols = read_columns(out_path)
    cid = np.asarray(cols["candidate_id"], dtype=np.int64)
    return cid, np.asarray(cols["score"], dtype=np.float64)


def _identity_tsv(tmp_path, features):
    """The same pool as a tab-separated PIN (the `rescore.handoff = tsv` layout)."""
    import pyarrow.parquet as pq

    path = tmp_path / "identity.pin"
    pq.read_table(str(features)).to_pandas().to_csv(path, sep="\t", index=False)
    return path


def _identity_inputs(tmp_path, backend):
    """(PIN path, fold-key path, n, env) for one backend of the identity pool."""
    features, keys, n = _identity_pool(tmp_path)
    pin = _identity_tsv(tmp_path, features) if backend == "tsv" else features
    env = dict(IDENTITY_ENV, MUMDIA_NN_FOLD_KEYS=str(keys),
               MUMDIA_NN_STREAM="1" if backend == "stream" else "0")
    return pin, keys, n, env


def _assert_same_scores(ref, new, n, what):
    rc, rs = _scores_by_row(ref)
    nc, ns = _scores_by_row(new)
    assert len(rs) == len(ns) == n
    assert np.array_equal(rc, nc), "the row order of the output changed"
    assert rs.tobytes() == ns.tobytes(), (
        "%s changed the scores (%d of %d rows differ)"
        % (what, int(np.count_nonzero(rs != ns)), n)
    )


@pytest.mark.parametrize("backend", ["in-memory", "stream"])
def test_default_speedups_leave_scores_byte_identical(torch_available, tmp_path, backend):
    """The default path must score exactly as the code it replaced, on the same host."""
    pin, _keys, n, env = _identity_inputs(tmp_path, backend)
    ref = tmp_path / "legacy.parquet"
    new = tmp_path / "default.parquet"
    stdout_ref, _ = run_worker_ok("nn_rescore_worker.py", pin, ref, env=dict(env, **LEGACY_ENV))
    stdout, _ = run_worker_ok("nn_rescore_worker.py", pin, new, env=env)
    # The comparison is only as strong as the paths it runs: the default arm must have
    # threaded its init scan, and the legacy arm must not have.
    assert "2 scan thread(s)" in stdout, "the default arm did not run the threaded init scan"
    assert "2 scan thread(s)" not in stdout_ref
    _assert_same_scores(ref, new, n, "a default-path speed-up")


# The worker as it was before the default-path speed-ups (W1-W6) and the move of the
# training loop into `_build_trainer`: origin/main when they were written. The switches in
# LEGACY_ENV only reach code that kept a switch; comparing against this commit also covers
# the refactors that have none. When a later change moves the default scores on purpose,
# point this at the commit that made it.
REFERENCE_COMMIT = "6887c41b7ed04ace7eb1d744d750e83bb2c93e9e"


def _reference_worker(tmp_path):
    """Write the worker at REFERENCE_COMMIT to tmp_path, or skip when git cannot supply it."""
    import pathlib
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not available to extract the reference worker")
    root = pathlib.Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [git, "-C", str(root), "show", REFERENCE_COMMIT + ":scripts/nn_rescore_worker.py"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        pytest.skip("reference commit %s is not in this clone (a shallow checkout?)"
                    % REFERENCE_COMMIT[:7])
    path = tmp_path / "nn_rescore_worker_reference.py"
    path.write_bytes(proc.stdout)
    return path


@pytest.mark.parametrize("backend", ["in-memory", "stream", "tsv"])
def test_default_path_scores_as_the_reference_worker(torch_available, tmp_path, backend):
    """The default path must give the reference worker's score bytes, on the same host.

    A host-local comparison of two code versions rather than a committed hash, so it holds
    on any CPU generation. The default arm runs the threaded init scan
    (`MUMDIA_NN_SCAN_ROWS_PER_THREAD` in IDENTITY_ENV) and every other default speed-up.
    """
    reference = _reference_worker(tmp_path)
    pin, _keys, n, env = _identity_inputs(tmp_path, backend)
    ref = tmp_path / "reference.parquet"
    new = tmp_path / "default.parquet"
    run_worker_ok(str(reference), pin, ref, env=env)
    run_worker_ok("nn_rescore_worker.py", pin, new, env=env)
    _assert_same_scores(ref, new, n, "the current worker (against %s)" % REFERENCE_COMMIT[:7])


def test_threaded_init_scan_picks_the_serial_feature_and_count():
    """The threaded init scan must return the serial scan's (column, sign, count) exactly.

    Tied columns make the tie-breaking rule (lowest column, sign +1 first, strict `>`)
    decide the winner, which is the part a completion-order reduction would get wrong.
    """
    w = _import_worker()
    rng = np.random.default_rng(11)
    n, nf = 30000, 9
    tgt = rng.random(n) < 0.5
    x = rng.normal(size=(n, nf)).astype(np.float32)
    x[tgt, 1] += 2.0
    x[:, 4] = x[:, 1]           # an exact duplicate of the best column
    x[:, 6] = -x[:, 1]          # its mirror image, tying on sign -1
    x[:, 7] = np.round(x[:, 7], 1)
    for topk in (0, 500):
        serial = w.n_targets_at_many(x, tgt, 0.01, topk=topk, workers=1)
        for workers in (2, 3, 8):
            assert w.n_targets_at_many(x, tgt, 0.01, topk=topk, workers=workers) == serial
    assert serial[0] == 1 and serial[1] == 1, "the tie must go to the lowest column"
    assert w.scan_workers(16, 300000, 387) == 15
    assert w.scan_workers(16, 10000, 387) == 1
    assert w.scan_workers(16, 300000, 3) == 3
    assert w.scan_workers(16, 10000, 387, rows_per_thread=500) == 16
    # The memory cap: each task in flight holds ~64 B per sample row. It binds only when the
    # init sample escalates (the 8.07M-row immunopeptidomics ladder reaches 4.8M rows).
    assert w.scan_workers(16, 1_200_000, 347) == 13
    assert w.scan_workers(16, 4_800_000, 347) == 3
    assert w.scan_workers(16, 20_000_000, 347) == 1
    assert w.scan_workers(16, 4_800_000, 347, mem_bytes=0) == 1
    assert w.scan_workers(16, 4_800_000, 347, mem_bytes=8 << 30) == 16


def test_single_threaded_load_releases_its_moment_buffer(tmp_path):
    """With one fill thread the moment buffer lives on the main thread; it must not stay.

    Pool threads drop theirs when they exit. The main thread lives for the whole run, so a
    cached 32,768 x features float64 buffer (0.1 GB at 387 features) would stay resident
    through training.
    """
    w = _import_worker()
    import pyarrow.parquet as pq

    features, _keys, n = _identity_pool(tmp_path)
    names = [c for c in pq.read_schema(str(features)).names if c not in w.NON_FEATURE]
    out = np.empty((n, len(names)), np.float32)
    w.fill_parquet_matrix(str(features), names, n, 700, out, threads=1, read_ahead=False)
    assert getattr(w._TLS, "moments", None) is None, "the main thread kept its moment buffer"


@pytest.mark.parametrize("chunk", [250000, 700])
def test_parallel_parquet_load_reproduces_the_serial_matrix_and_moments(tmp_path, chunk):
    """The threaded, read-ahead load must give the serial loop's matrix, mean and std bytes.

    The float64 column sums depend on the order of their additions, so the parallel loader
    keeps the serial loop's sub-block partition and adds the partial sums in that order.
    The pool carries non-finite cells, a float64 column beyond float32 range, nulls, -0.0
    and a CHUNK smaller than a row group (700 rows against 1,500).
    """
    w = _import_worker()
    import pyarrow.parquet as pq

    features, _keys, n = _identity_pool(tmp_path)
    names = [c for c in pq.read_schema(str(features)).names if c not in w.NON_FEATURE]
    nf = len(names)

    ref = np.empty((n, nf), np.float32)
    r1, r2 = w._fill_parquet_matrix_legacy(str(features), names, n, chunk, ref)
    rmean, rstd = w.moments_to_mean_std(r1, r2, n)
    w.standardise_matrix(ref, rmean, rstd, chunk, threads=1)

    for threads, read_ahead, pre_buffer in [(1, False, False), (1, True, True),
                                            (3, True, True), (8, False, True)]:
        got = np.empty((n, nf), np.float32)
        g1, g2 = w.fill_parquet_matrix(str(features), names, n, chunk, got, threads=threads,
                                       read_ahead=read_ahead, pre_buffer=pre_buffer)
        assert g1.tobytes() == r1.tobytes() and g2.tobytes() == r2.tobytes(), (
            "column sums differ at threads=%d read_ahead=%s" % (threads, read_ahead))
        gmean, gstd = w.moments_to_mean_std(g1, g2, n)
        w.standardise_matrix(got, gmean, gstd, chunk, threads=threads, block=256)
        assert gmean.tobytes() == rmean.tobytes() and gstd.tobytes() == rstd.tobytes()
        assert got.tobytes() == ref.tobytes(), (
            "matrix differs at threads=%d read_ahead=%s" % (threads, read_ahead))
    assert np.isfinite(ref).all()


def test_moment_blocks_follow_the_serial_partition():
    w = _import_worker()
    assert w.moment_blocks(7000, 3000, sub=1000) == [
        (0, 1000), (1000, 2000), (2000, 3000),
        (3000, 4000), (4000, 5000), (5000, 6000),
        (6000, 7000),
    ]
    assert w.moment_blocks(131072, 250000) == [
        (0, 32768), (32768, 65536), (65536, 98304), (98304, 131072)
    ]
    assert w.moment_blocks(5, 2, sub=32768) == [(0, 2), (2, 4), (4, 5)]
    assert w.moment_blocks(0, 250000) == []


def _awkward_scores(rng, n, kind):
    """float32 scores with the values an exact order must get right."""
    if kind == "ties":
        s = np.round(rng.normal(size=n), 1)
    elif kind == "coarse":
        s = rng.integers(-3, 4, size=n).astype(np.float64)
    else:
        s = rng.normal(size=n)
    s = s.astype(np.float32)
    k = max(1, n // 50)
    s[rng.integers(0, n, k)] = np.float32(-0.0)
    s[rng.integers(0, n, k)] = np.float32(0.0)
    s[rng.integers(0, n, 3)] = np.float32(np.inf)
    s[rng.integers(0, n, 3)] = np.float32(-np.inf)
    s[rng.integers(0, n, 2)] = np.float32(1e-41)      # subnormal
    s[rng.integers(0, n, 2)] = np.float32(-1e-41)
    return s


def test_desc_order_is_the_stable_descending_argsort():
    """One uint64 key sort must reproduce `np.argsort(-s, kind="stable")` exactly."""
    w = _import_worker()
    rng = np.random.default_rng(5)
    for kind in ("ties", "coarse", "normal"):
        for n in (1, 2, 17, 5000):
            s = _awkward_scores(rng, n, kind) if n > 4 else rng.normal(size=n).astype(np.float32)
            want = np.argsort(-s, kind="stable")
            assert np.array_equal(w.desc_order(s), want), (kind, n)
            s2 = s.copy()
            s2[rng.integers(0, n, max(1, n // 100))] = np.float32(np.nan)
            s2[:1] = -np.float32(np.nan)                   # a NaN with its sign bit set
            assert np.array_equal(w.desc_order(s2), np.argsort(-s2, kind="stable")), (kind, n)
    assert w.desc_order(np.zeros(0, np.float32)).shape == (0,)
    f64 = rng.normal(size=100)
    assert np.array_equal(w.desc_order(f64), np.argsort(-f64, kind="stable"))


@pytest.mark.parametrize("kind", ["ties", "coarse", "normal"])
def test_windowed_positive_selection_equals_the_full_tda_q_selection(kind):
    """The certified window must select exactly `(tda_q <= thr) & target`, or decline.

    Randomised over pool size, target fraction, separation and threshold, with ties at
    the window edge, +-0.0, infinities and subnormals. A NaN anywhere must make it decline
    rather than guess where the sort puts NaN.
    """
    w = _import_worker()
    rng = np.random.default_rng({"ties": 1, "coarse": 2, "normal": 3}[kind])
    windowed = declined = 0
    for trial in range(60):
        n = int(rng.integers(50, 20000))
        tgt = rng.random(n) < rng.uniform(0.2, 0.8)
        s = _awkward_scores(rng, n, kind)
        s[tgt] += np.float32(rng.uniform(0.0, 3.0))
        thr = float(rng.choice([0.001, 0.01, 0.05, 0.2]))
        want = (w.tda_q(s, tgt.astype(np.float32)) <= thr) & tgt
        got = w.select_positives(s, tgt, thr)
        if got is None:
            declined += 1
        else:
            windowed += 1
            assert np.array_equal(got, want), (kind, trial, n, thr)
        assert w.n_targets_at_windowed(s, tgt.astype(np.float32), thr) == int(want.sum())
        s_nan = s.copy()
        s_nan[int(rng.integers(0, n))] = np.float32(np.nan)
        assert w.select_positives(s_nan, tgt, thr) is None
    assert windowed > 10, "the window was almost never certified; the test lost its point"


def test_thread_pools_compute_under_the_main_threads_flush_to_zero(torch_available, tmp_path):
    """Under the worker's default flush-to-zero, pool threads must match the serial code.

    FTZ/DAZ live in each thread's MXCSR, and a thread started on Windows does not inherit
    them, so every pool the worker starts runs `_fp_thread_init`. Checked where the state
    decides the bytes: the float64-to-float32 narrowing of a value that is subnormal in
    float32 (the fill), the float64 sums of a column of zeros and float32 subnormals (the
    moments), standardisation, and the init scan on a column whose only signal is
    subnormal, which DAZ turns into a tie. `desc_order` and the windowed selection run on
    the main thread and must still reproduce the stable argsort and `tda_q` under DAZ.

    Every input is built before flush-to-zero is switched on: under FTZ, `np.float32(1e-40)`
    is already 0.0, and the checks would pass on inputs without a single subnormal.
    """
    torch = pytest.importorskip("torch")
    w = _import_worker()
    import pyarrow.parquet as pq

    tiny = np.float32(1.2e-38)
    features, _keys, n = _identity_pool(tmp_path)
    names = [c for c in pq.read_schema(str(features)).names if c not in w.NON_FEATURE]
    nf = len(names)
    rng = np.random.default_rng(13)
    m = 40000
    tgt = rng.random(m) < 0.5
    x = rng.normal(size=(m, 3)).astype(np.float32)
    x[tgt, 0] += np.float32(1.0)
    x[:, 1] = np.float32(0.0)
    x[tgt & (rng.random(m) < 0.6), 1] = np.float32(1e-40)
    assert np.count_nonzero(x[:, 1]) > 1000
    scores = []
    for kind in ("ties", "coarse", "normal"):
        s = _awkward_scores(rng, 5000, kind)
        lab = rng.random(5000) < 0.5
        shifted = s.copy()
        shifted[lab] += np.float32(1.5)
        shifted[rng.integers(0, 5000, 20)] = np.float32(3e-41)
        assert np.any((s != 0) & (np.abs(s) < tiny))
        assert np.any((shifted != 0) & (np.abs(shifted) < tiny))
        scores.append((kind, s, lab, shifted))

    if not torch.set_flush_denormal(True):
        pytest.skip("this CPU cannot flush subnormals to zero")
    w._FLUSH["on"] = True
    try:
        init = w._fp_thread_init
        serial = w.n_targets_at_many(x, tgt, 0.01, workers=1)
        assert serial[0] == 0, "DAZ is not in effect: the subnormal column separated"
        assert w.n_targets_at_many(x, tgt, 0.01, workers=4, initializer=init) == serial

        ref = np.empty((n, nf), np.float32)
        r1, r2 = w._fill_parquet_matrix_legacy(str(features), names, n, 700, ref)
        j11 = names.index("f11")
        assert not np.any((ref[:, j11] != 0) & (np.abs(ref[:, j11]) < tiny)), (
            "flush-to-zero is not in effect on the main thread")
        got = np.empty((n, nf), np.float32)
        g1, g2 = w.fill_parquet_matrix(str(features), names, n, 700, got, threads=4,
                                       initializer=init)
        assert got.tobytes() == ref.tobytes(), "a fill thread narrowed without FTZ"
        assert g1.tobytes() == r1.tobytes() and g2.tobytes() == r2.tobytes(), (
            "a fill thread summed float32 subnormals without DAZ")
        mean, std = w.moments_to_mean_std(r1, r2, n)
        a, b = ref.copy(), ref.copy()
        w.standardise_matrix(a, mean, std, 700, threads=1)
        w.standardise_matrix(b, mean, std, 700, threads=4, initializer=init, block=256)
        assert a.tobytes() == b.tobytes()

        for kind, s, lab, shifted in scores:
            assert np.array_equal(w.desc_order(s), np.argsort(-s, kind="stable")), kind
            want = (w.tda_q(shifted, lab.astype(np.float32)) <= 0.05) & lab
            got_sel = w.select_positives(shifted, lab, 0.05)
            assert got_sel is None or np.array_equal(got_sel, want), kind
    finally:
        torch.set_flush_denormal(False)
        w._FLUSH["on"] = False


@pytest.mark.parametrize("backend", ["in-memory", "stream", "tsv"])
def test_parallel_folds_do_not_depend_on_the_process_count(torch_available, tmp_path, backend):
    """MUMDIA_NN_PARALLEL: the scores are a function of the tasks, not of the processes.

    The keyed epoch shuffle makes every (seed, fold) task self-contained, so one process
    running the six tasks in turn and three running them at once must return the same
    bytes, at the same per-process thread count. Every row is still scored exactly once,
    and neither the shared memmap nor the side arrays may be left behind. Each backend
    reaches the children by its own route: the parquet in-memory load writes the matrix
    into the memmap, the streaming backend hands over its own memmap, and the TSV load
    copies its standardised matrix into one.
    """
    pin, _keys, n, env = _identity_inputs(tmp_path, backend)
    env = dict(env, MUMDIA_NN_SEEDS="2", MUMDIA_NN_PARALLEL_THREADS="1")
    outs = {}
    for k in (1, 3):
        out = tmp_path / ("parallel_%d.parquet" % k)
        stdout, _ = run_worker_ok("nn_rescore_worker.py", pin, out,
                                  env=dict(env, MUMDIA_NN_PARALLEL=str(k)))
        assert "parallel training: %d process(es) x 1 torch thread(s) for 6 task(s)" % k in stdout
        assert ("backend=stream(memmap)" in stdout) == (backend == "stream")
        assert ("format=tsv" in stdout) == (backend == "tsv")
        assert_complete_finite_coverage(out, n, sidecar="nn_rescore_worker[parallel=%d]" % k)
        outs[k] = _scores_by_row(out)
    assert np.array_equal(outs[1][0], outs[3][0])
    assert outs[1][1].tobytes() == outs[3][1].tobytes(), (
        "the parallel scores depend on the process count")
    leftovers = [p.name for p in tmp_path.iterdir()
                 if p.name.endswith((".feat.mm", ".npy"))]
    assert not leftovers, "parallel training left files behind: %s" % leftovers


def test_failed_run_removes_its_memmap(torch_available, tmp_path):
    """A worker that fails must not leave its memmap behind (Windows keeps it mapped until
    the traceback is released, so the removal is retried after it is printed)."""
    features, keys, n = _identity_pool(tmp_path)
    env = dict(IDENTITY_ENV, MUMDIA_NN_FOLD_KEYS=str(keys), MUMDIA_NN_STREAM="1",
               MUMDIA_NN_TRAIN_FDR="0.0000001", MUMDIA_NN_INIT_FDR_MAX="0")
    out = tmp_path / "failing.parquet"
    rc, _, err = run_worker("nn_rescore_worker.py", features, out, env=env)
    assert rc != 0
    assert "selected no positive targets" in err
    assert not (tmp_path / "failing.parquet.feat.mm").exists()
    rc, _, err = run_worker("nn_rescore_worker.py", features, out,
                            env=dict(env, MUMDIA_NN_STREAM="0", MUMDIA_NN_PARALLEL="2"))
    assert rc != 0
    assert "selected no positive targets" in err
    # A child's log reaches the parent only inside the exception, and it is what explains
    # the failure: the rescans and the init feature the fold started from.
    assert "--- log of that task ---" in err
    assert "rescanning on" in err and "init=" in err
    assert not [p.name for p in tmp_path.iterdir() if p.name.startswith("failing.parquet.")]
