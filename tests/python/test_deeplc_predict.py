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


# ------------------------------------------------ the Arrow unique set and rewrite

ARROW_NAMES = ["STD", "MOD_RE", "strip_mods", "base_pf", "DECOY_PREFIX_RE", "STD_FULL_RE"]
ARROW_FUNCS = ["is_std", "library_bases", "unique_standard_bases", "rewrite_irt"]


def _arrow_helpers():
    """The constants and Arrow helpers of `deeplc_finetune.py`, without importing DeepLC."""
    import re

    np = pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")
    import pyarrow.compute as pc

    text = (SCRIPTS / "deeplc_finetune.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    namespace = {"re": re, "np": np, "pa": pa, "pc": pc}
    for node in tree.body:
        wanted = (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in ARROW_NAMES for t in node.targets)
        ) or (isinstance(node, ast.FunctionDef) and node.name in ARROW_FUNCS)
        if wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "deeplc_finetune.py", "exec"),
                 namespace)  # noqa: S102
    missing = [n for n in ARROW_NAMES + ARROW_FUNCS if n not in namespace]
    assert not missing, "deeplc_finetune.py no longer defines {}".format(missing)
    return namespace


def _library_peptidoforms(rng, n_base=400):
    """Targets, prefix decoys, interior markers, modifications and non-standard rows."""
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    out = []
    for i in range(n_base):
        s = "".join(rng.choice(aa, size=int(rng.integers(5, 15))))
        kind = i % 9
        if kind == 1:
            s = s[:2] + "M[Oxidation]" + s[2:]
        elif kind == 2:
            s = "[Acetyl]-" + s
        elif kind == 3:
            s = s[:3] + "U" + s[3:]
        elif kind == 4:
            s = s[:3] + "DECOY_" + s[3:]
        elif kind == 5:
            s = s + "C[Carbamidomethyl]"
        for _ in range(int(rng.integers(1, 4))):
            out.append(s)
            out.append("DECOY_" + s[::-1] if i % 2 else "DECOY_" + s)
    out += ["", "DECOY_", "DECOY_DECOY_PEPTIDEK", "pepTIDEK"]
    order = rng.permutation(len(out))
    return [out[i] for i in order]


@pytest.mark.parametrize("predict_limit", [0, 150])
def test_the_arrow_unique_set_and_rewrite_equal_the_per_row_loop(predict_limit):
    """Same unique order, same float32 bits and same counts as the dictionary version.

    The reference below is the loop the worker ran before: a Python set over every row, a
    dict of predictions, and one lookup per row. Values include NaN and infinity, which
    must be counted as "no finite prediction" and keep the imported value.
    """
    h = _arrow_helpers()
    np, pa = h["np"], h["pa"]
    rng = np.random.default_rng(3)
    pform = _library_peptidoforms(rng)
    cut = [0, 97, 400, 401, len(pform)]
    column = pa.chunked_array([pa.array(pform[a:b], pa.string()) for a, b in zip(cut, cut[1:])])
    orig = rng.uniform(0, 100, len(pform)).astype(np.float32)

    # reference (the previous implementation)
    ref_uniq, seen = [], set()
    for pf in pform:
        b = h["base_pf"](pf)
        if b not in seen and h["is_std"](pf):
            seen.add(b)
            ref_uniq.append(b)
    if predict_limit:
        ref_uniq = ref_uniq[:predict_limit]
    prediction = {}
    for k, b in enumerate(ref_uniq):
        prediction[b] = float("nan") if k % 17 == 3 else float("inf") if k % 29 == 5 else \
            float(rng.normal(50, 20))
    ref_new = np.empty(len(pform), dtype=np.float32)
    counts = {"none": 0, "nonfinite": 0, "ok": 0}
    for i, pf in enumerate(pform):
        v = prediction.get(h["base_pf"](pf))
        if v is None:
            ref_new[i] = orig[i]
            counts["none"] += 1
        elif not np.isfinite(v):
            ref_new[i] = orig[i]
            counts["nonfinite"] += 1
        else:
            ref_new[i] = v
            counts["ok"] += 1

    bases = h["library_bases"](column)
    uniq = h["unique_standard_bases"](bases)
    if predict_limit:
        uniq = uniq.slice(0, predict_limit)
    assert uniq.to_pylist() == ref_uniq, "unique set or its first-occurrence order differs"
    values = np.array([prediction[b] for b in uniq.to_pylist()], dtype=np.float64)
    new, summary = h["rewrite_irt"](bases, orig, uniq, values)
    assert new.dtype == np.float32
    assert np.array_equal(new.view(np.uint32), ref_new.view(np.uint32))
    assert summary == {
        "rows": len(pform),
        "repredicted": counts["ok"],
        "retained_imported": counts["none"] + counts["nonfinite"],
        "retained_non_standard": counts["none"],
        "retained_no_prediction": counts["nonfinite"],
    }
    assert counts["nonfinite"] > 0 and counts["none"] > 0, "the fixture lost a case"


def test_only_a_leading_decoy_marker_is_stripped():
    h = _arrow_helpers()
    pa = h["pa"]
    col = pa.chunked_array([pa.array(["DECOY_PEPTIDEK", "PEP_DECOY_TIDEK", "DECOY_DECOY_K"])])
    assert h["library_bases"](col).to_pylist() == ["PEPTIDEK", "PEP_DECOY_TIDEK", "DECOY_K"]
