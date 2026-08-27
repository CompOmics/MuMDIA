"""Contract tests for `scripts/mokapot_worker.py` (Stage F, `mokapot`).

Skipped without mokapot. Two contracts:

1. Complete out-of-fold coverage. `_oof_scores` merges mokapot's held-out
   target and decoy confidence tables and raises unless they cover the PIN rows
   exactly once with finite scores (`mokapot_worker.py:136-163`). There is
   deliberately no in-sample or fold-averaging fallback, because averaging all
   fold models over all rows is not out-of-fold (two of three folds trained on
   each row) and makes nominal q-values anti-conservative.
2. Model selection. `MUMDIA_RESCORE_MODEL` chooses the classifier, and the
   worker's code default is the sklearn MLP even though the portable env sets
   `logreg`. A silently substituted model invalidates any A/B comparison and
   contradicts the rule that an explicitly requested external classifier must
   not become something else (`CLAUDE.md`, "FDR and sidecar rules").
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from conftest import (
    SCRIPTS,
    assert_complete_finite_coverage,
    importorskip_any,
    read_columns,
    run_worker,
    run_worker_ok,
    synthetic_psms,
    write_pin,
)

# mokapot's model is built with a hard-coded train_fdr of 0.01
# (`mokapot_worker.py:53`), so the pool has to be large enough for the
# semi-supervised step to find positives at 1%.
FAST_ENV = {
    "MUMDIA_RESCORE_MODEL": "logreg",
    "MUMDIA_BREW_ITERS": "2",
    "MUMDIA_MOKAPOT_WORKERS": "1",
    "MUMDIA_LR_MAX_ITER": "200",
    "PYTHONUTF8": "1",
}


@pytest.fixture(scope="module")
def mokapot_module():
    return importorskip_any("mokapot", "the mokapot rescorer needs a usable mokapot")


@pytest.fixture(scope="module")
def worker_module(mokapot_module):
    """Import `mokapot_worker.py` in-process to exercise `make_model` directly."""
    spec = importlib.util.spec_from_file_location(
        "mumdia_mokapot_worker", SCRIPTS / "mokapot_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def psms():
    return synthetic_psms(n_targets=600, n_decoys=600, n_features=5, seed=2)


def test_logreg_run_scores_every_pin_row_exactly_once(mokapot_module, psms, tmp_path):
    """Every PIN row must come back once, finite, from a held-out fold.

    The Rust caller validates exact, unique, finite coverage and bails
    otherwise (`rescore.rs:1046-1082`), so an incomplete confidence table costs
    the whole run under `rescore.strict = true`. The stdout line asserted here
    is the only place the out-of-fold branch announces itself, and there is no
    in-sample fallback behind it by design.
    """
    pin = write_pin(tmp_path / "rescore.pin", psms)
    out = tmp_path / "rescore_sidecar_out.parquet"
    stdout, _ = run_worker_ok("mokapot_worker.py", pin, out, env=FAST_ENV)
    assert "using complete out-of-fold confidence scores" in stdout
    assert "mokapot_worker[logreg]" in stdout, (
        "the worker did not report the requested model; a substituted "
        "classifier must never be silent"
    )
    assert_complete_finite_coverage(out, psms["n"], sidecar="mokapot_worker")

    cols = read_columns(out)
    assert np.allclose(np.asarray(cols["q_value"], dtype=float), 0.0), (
        "q_value must be written as zeros; the Rust caller computes q"
    )
    order = np.argsort([int(x) for x in cols["candidate_id"]])
    scores = np.asarray(cols["score"], dtype=float)[order]
    is_target = psms["labels"] == 1
    assert scores[is_target].mean() > scores[~is_target].mean(), (
        "targets do not outscore decoys, so the scores are not aligned to the "
        "input rows (candidate_id echoes the SpecId tail, i.e. the row index)"
    )


def test_rescore_model_env_selects_logistic_regression(worker_module, mokapot_module):
    """`MUMDIA_RESCORE_MODEL=logreg` must build a LogisticRegression.

    The worker's code default is the sklearn MLP, while the recommended
    portable environment (`env/mumdia-rescore.yml`) sets `logreg` and the Rust
    caller sets nothing. If this mapping breaks, a run configured for the fast
    convex path silently trains an MLP instead: different scores, different
    identification counts, and near-bit-exact reproducibility lost.
    """
    from sklearn.linear_model import LogisticRegression

    prior = os.environ.get("MUMDIA_RESCORE_MODEL")
    try:
        os.environ["MUMDIA_RESCORE_MODEL"] = "logreg"
        model = worker_module.make_model(mokapot_module)
        assert isinstance(model.estimator, LogisticRegression)
        for alias in ("logistic", "lr"):
            os.environ["MUMDIA_RESCORE_MODEL"] = alias
            assert isinstance(
                worker_module.make_model(mokapot_module).estimator, LogisticRegression
            ), "the documented alias {} no longer selects logreg".format(alias)
    finally:
        if prior is None:
            os.environ.pop("MUMDIA_RESCORE_MODEL", None)
        else:
            os.environ["MUMDIA_RESCORE_MODEL"] = prior


def test_default_and_percolator_models_are_unchanged(worker_module, mokapot_module):
    """The unset default is the sklearn MLP, and `percolator` means `model=None`.

    `model=None` is how the worker asks mokapot for its own grid-searched
    linear SVM. Confusing the two would make `classifier = mokapot` mean a
    different algorithm than every recorded benchmark used, while
    `psms_scored.parquet.report.json` still reports "mokapot".
    """
    from sklearn.neural_network import MLPClassifier

    prior = os.environ.get("MUMDIA_RESCORE_MODEL")
    try:
        os.environ.pop("MUMDIA_RESCORE_MODEL", None)
        assert isinstance(worker_module.make_model(mokapot_module).estimator,
                          MLPClassifier)
        for alias in ("percolator", "linear", "svm"):
            os.environ["MUMDIA_RESCORE_MODEL"] = alias
            assert worker_module.make_model(mokapot_module) is None
    finally:
        if prior is None:
            os.environ.pop("MUMDIA_RESCORE_MODEL", None)
        else:
            os.environ["MUMDIA_RESCORE_MODEL"] = prior


def test_unknown_model_name_fails_loudly(mokapot_module, psms, tmp_path):
    """An unrecognised `MUMDIA_RESCORE_MODEL` must abort, not fall back.

    A typo that quietly resolved to the default would run a different
    classifier from the one the operator asked for, and the artifact report
    would still name `mokapot`, so the substitution would never be noticed.
    """
    pin = write_pin(tmp_path / "rescore.pin", psms)
    out = tmp_path / "out.parquet"
    env = dict(FAST_ENV)
    env["MUMDIA_RESCORE_MODEL"] = "randomforest"
    rc, _, err = run_worker("mokapot_worker.py", pin, out, env=env)
    assert rc != 0
    assert "MUMDIA_RESCORE_MODEL" in err
    assert not out.exists()
