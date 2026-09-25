"""Thread policy and prediction plumbing of the two DeepLC workers.

The thread-cap helpers are pure stdlib, but both workers import DeepLC at module scope, so
they are lifted out of the source with `ast` and executed on their own. That keeps the
policy testable on an interpreter without DeepLC, which is every CI job.
"""

from __future__ import annotations

import ast
import os
import sys

import pytest

from conftest import SCRIPTS

DEEPLC_WORKERS = ["deeplc_worker.py", "deeplc_finetune.py"]
THREAD_HELPERS = [
    "_windows_physical_cores",
    "physical_cores",
    "deeplc_thread_cap",
    "capped_threads",
]


def _function_sources(script, names):
    text = (SCRIPTS / script).read_text(encoding="utf-8")
    tree = ast.parse(text)
    found = {
        node.name: ast.get_source_segment(text, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = [n for n in names if n not in found]
    assert not missing, "{} no longer defines {}".format(script, missing)
    return found


def _thread_helpers(script="deeplc_finetune.py"):
    """The four thread helpers of `script`, executed in a namespace of their own."""
    namespace = {"os": os, "sys": sys}
    for source in _function_sources(script, THREAD_HELPERS).values():
        exec(compile(source, str(SCRIPTS / script), "exec"), namespace)  # noqa: S102
    return namespace


def test_both_workers_carry_the_same_thread_cap_helpers():
    """The helpers are copied rather than shared, so they must not drift apart.

    `deeplc_worker.py` cannot import `deeplc_finetune.py`: that module's body pins the
    OpenMP pools and imports psm_utils, which the predict-frag worker does not need.
    """
    a = _function_sources("deeplc_worker.py", THREAD_HELPERS)
    b = _function_sources("deeplc_finetune.py", THREAD_HELPERS)
    for name in THREAD_HELPERS:
        assert a[name] == b[name], "{} differs between the two DeepLC workers".format(name)


def test_the_cap_is_a_ceiling_and_never_a_target():
    capped = _thread_helpers()["capped_threads"]
    assert capped(8, 0) == 8, "a cap of 0 means no cap"
    assert capped(128, 64) == 64
    assert capped(8, 64) == 8, "a request below the cap is taken as given"
    assert capped(64, 64) == 64
    assert capped(0, 64) == 1, "at least one thread"
    assert capped(-3, 0) == 1


def test_the_environment_override_sets_or_disables_the_cap(monkeypatch):
    helpers = _thread_helpers()
    monkeypatch.setenv("MUMDIA_DEEPLC_THREAD_CAP", "0")
    assert helpers["deeplc_thread_cap"]()[0] == 0
    monkeypatch.setenv("MUMDIA_DEEPLC_THREAD_CAP", "12")
    assert helpers["deeplc_thread_cap"]() == (12, "MUMDIA_DEEPLC_THREAD_CAP")
    auto = helpers["physical_cores"]()[0] or 0
    for raw in ("auto", "", "twelve"):
        monkeypatch.setenv("MUMDIA_DEEPLC_THREAD_CAP", raw)
        assert helpers["deeplc_thread_cap"]()[0] == auto, raw


def test_the_physical_core_count_is_plausible():
    """Between one and the logical CPU count, or unknown; never more cores than CPUs."""
    n, why = _thread_helpers()["physical_cores"]()
    assert isinstance(why, str) and why
    if n is None:
        pytest.skip("no physical core count on this platform: " + why)
    logical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    assert 1 <= n <= (logical or n), (n, logical, why)
