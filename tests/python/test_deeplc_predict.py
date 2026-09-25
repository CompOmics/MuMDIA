"""Thread policy and prediction plumbing of the two DeepLC workers.

The thread-cap helpers are pure stdlib, but both workers import DeepLC at module scope, so
they are lifted out of the source with `ast` and executed on their own. That keeps the
policy testable on an interpreter without DeepLC, which is every CI job.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys

import pytest

from conftest import SCRIPTS, run_worker_ok

DEEPLC_WORKERS = ["deeplc_worker.py", "deeplc_finetune.py"]
THREAD_HELPERS = [
    "_windows_physical_cores",
    "_sysfs_physical_cores",
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
    shared = THREAD_HELPERS + ["load_base_model"]
    a = _function_sources("deeplc_worker.py", shared)
    b = _function_sources("deeplc_finetune.py", shared)
    for name in shared:
        assert a[name] == b[name], "{} differs between the two DeepLC workers".format(name)


def test_the_model_load_falls_back_on_any_private_api_change(monkeypatch, capsys):
    """A DeepLC that moves or re-signatures `_model_ops.load_model` gives None, not a crash.

    None makes every prediction call load its own model, which is what the workers did
    before they loaded it once. Faked with a stand-in `deeplc` package, so no DeepLC is
    needed.
    """
    import types

    source = _function_sources("deeplc_finetune.py", ["load_base_model"])["load_base_model"]
    namespace = {}
    exec(compile(source, "deeplc_finetune.py", "exec"), namespace)  # noqa: S102
    load = namespace["load_base_model"]

    def install(model_ops, core):
        pkg = types.ModuleType("deeplc")
        pkg.__path__ = []
        if model_ops is not None:
            pkg._model_ops = model_ops
            monkeypatch.setitem(sys.modules, "deeplc._model_ops", model_ops)
        else:
            monkeypatch.setitem(sys.modules, "deeplc._model_ops", None)
        monkeypatch.setitem(sys.modules, "deeplc", pkg)
        monkeypatch.setitem(sys.modules, "deeplc.core", core)

    core = types.SimpleNamespace(DEFAULT_MODEL="default.pt")
    ok = types.SimpleNamespace(load_model=lambda path: ("model", path))
    install(ok, core)
    assert load() == ("model", "default.pt")
    renamed = types.SimpleNamespace()  # AttributeError
    install(renamed, core)
    assert load() is None
    resigned = types.SimpleNamespace(load_model=lambda: "no argument")  # TypeError
    install(resigned, core)
    assert load() is None
    install(None, core)  # ImportError
    assert load() is None
    assert capsys.readouterr().out.count("every prediction call loads its own") == 3


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
    # `int(float("inf"))` raises OverflowError, not ValueError, and used to crash both
    # workers at start-up instead of falling back to auto.
    for raw in ("auto", "", "twelve", "nan", "inf", "-inf", "1e400"):
        monkeypatch.setenv("MUMDIA_DEEPLC_THREAD_CAP", raw)
        assert helpers["deeplc_thread_cap"]()[0] == auto, raw


def _fake_sysfs(root, cpus):
    """A sysfs `cpu` tree: `cpus` maps a CPU number to the topology files it has."""
    for cpu, files in cpus.items():
        topo = root / "cpu{}".format(cpu) / "topology"
        topo.mkdir(parents=True)
        for name, value in files.items():
            (topo / name).write_text(value + "\n", encoding="ascii")
    return str(root)


def test_the_sysfs_core_count_keys_on_the_sibling_list(tmp_path):
    """Hyperthreads of one core count once; cores of two ARM64 clusters count twice.

    The (physical_package_id, core_id) pair alone collapses a device-tree ARM64 SoC:
    `core_id` restarts in each cluster and the package id is -1 for every CPU.
    """
    count = _thread_helpers()["_sysfs_physical_cores"]
    # ARM64, two clusters of four: core_id 0-3 twice, one package, no SMT.
    arm = {c: {"core_cpus_list": str(c), "thread_siblings_list": str(c),
               "physical_package_id": "-1", "core_id": str(c % 4)} for c in range(8)}
    assert count(range(8), _fake_sysfs(tmp_path / "arm", arm)) == 8
    # x86 with SMT: CPUs 0-3 are cores 0-3, CPUs 4-7 their second hyperthreads.
    smt = {c: {"core_cpus_list": "{},{}".format(c % 4, c % 4 + 4),
               "physical_package_id": "0", "core_id": str(c % 4)} for c in range(8)}
    assert count(range(8), _fake_sysfs(tmp_path / "smt", smt)) == 4
    # An affinity mask of one hyperthread per core, and of both hyperthreads of one core.
    assert count([0, 1, 2, 3], _fake_sysfs(tmp_path / "smt_a", smt)) == 4
    assert count([0, 4], _fake_sysfs(tmp_path / "smt_b", smt)) == 1
    # A kernel before 5.5 has only thread_siblings_list.
    old = {c: {"thread_siblings_list": "{}-{}".format(c - c % 2, c - c % 2 + 1)}
           for c in range(6)}
    assert count(range(6), _fake_sysfs(tmp_path / "old", old)) == 3
    # Neither sibling list: the pair is the fallback.
    pairs = {c: {"physical_package_id": str(c // 4), "core_id": str(c % 2)} for c in range(8)}
    assert count(range(8), _fake_sysfs(tmp_path / "pairs", pairs)) == 4
    # No topology at all: unknown, and the caller falls back to the CPU count.
    assert count(range(4), str(tmp_path / "missing")) is None


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
ARROW_FUNCS = ["is_std", "library_bases", "unique_standard_bases", "rewrite_irt",
               "rewrite_positions", "union_positions"]


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


def test_the_band_union_rewrites_every_band_as_the_whole_library_would():
    """`--bands`: one union over the bands, and each band rewritten from its positions.

    Cut into contiguous bands in library order, the union is the whole library's unique
    set in the same order, so every band's rewritten column is exactly its slice of the
    whole library's, and the per-band counts add up to the whole library's counts. A band
    that repeats another band's rows (a window overlap) adds nothing to the union.
    """
    h = _arrow_helpers()
    np, pa = h["np"], h["pa"]
    rng = np.random.default_rng(5)
    pform = _library_peptidoforms(rng)
    orig = rng.uniform(0, 100, len(pform)).astype(np.float32)
    whole = pa.chunked_array([pa.array(pform, pa.string())])
    bases = h["library_bases"](whole)
    uniq = h["unique_standard_bases"](bases)
    values = rng.normal(50, 20, len(uniq))
    values[::13] = np.nan
    new_whole, s_whole = h["rewrite_irt"](bases, orig, uniq, values)

    cuts = [0, 120, 121, 500, len(pform)]
    band_bases = [h["library_bases"](pa.chunked_array([pa.array(pform[a:b], pa.string())]))
                  for a, b in zip(cuts, cuts[1:])]
    union, band_pos = h["union_positions"](band_bases)
    assert union.to_pylist() == uniq.to_pylist(), "the union is not the library's unique set"
    totals = {k: 0 for k in s_whole}
    for (a, b), pos in zip(zip(cuts, cuts[1:]), band_pos):
        new_b, s_b = h["rewrite_positions"](orig[a:b], pos, values, len(union))
        assert np.array_equal(new_b.view(np.uint32), new_whole[a:b].view(np.uint32))
        for k in totals:
            totals[k] += s_b[k]
    assert totals == s_whole

    # An overlap: the last band repeats the first 50 rows. The union does not grow, and the
    # repeated rows get the same values the first band gave them.
    overlap = band_bases + [h["library_bases"](pa.chunked_array([pa.array(pform[:50],
                                                                          pa.string())]))]
    union2, pos2 = h["union_positions"](overlap)
    assert union2.to_pylist() == uniq.to_pylist()
    new_o, _ = h["rewrite_positions"](orig[:50], pos2[-1], values, len(union2))
    assert np.array_equal(new_o.view(np.uint32), new_whole[:50].view(np.uint32))

    # A limit on the predicted set: positions past it count as not found, as they do for
    # a single table.
    lim = 40
    new_l, s_l = h["rewrite_irt"](bases, orig, uniq.slice(0, lim), values[:lim])
    new_p, s_p = h["rewrite_positions"](orig, np.concatenate(band_pos), values[:lim], lim)
    assert np.array_equal(new_l.view(np.uint32), new_p.view(np.uint32))
    assert s_l == s_p


def test_the_band_list_is_validated_before_anything_is_read(tmp_path):
    source = _module_source(["read_band_pairs"])
    ns = {}
    exec(compile(source, "deeplc_finetune.py", "exec"), ns)  # noqa: S102

    class Args:
        lib_in = lib_out = "-"
        multihead = 80
        no_finetune = False

    tsv = tmp_path / "bands.tsv"
    tsv.write_text("a.parquet\tb.parquet\n\nc.parquet\td.parquet\r\n", encoding="utf-8")
    Args.bands = str(tsv)
    assert ns["read_band_pairs"](Args) == [("a.parquet", "b.parquet"), ("c.parquet", "d.parquet")]
    for bad, why in [
        ("a.parquet\n", "expected"),
        ("a\tb\tc\n", "expected"),
        ("a\tb\nc\tb\n", "same output twice"),
        ("\n", "no band"),
    ]:
        tsv.write_text(bad, encoding="utf-8")
        with pytest.raises(SystemExit, match=why):
            ns["read_band_pairs"](Args)
    tsv.write_text("a\tb\n", encoding="utf-8")
    Args.lib_in = "x.parquet"
    with pytest.raises(SystemExit, match="positional"):
        ns["read_band_pairs"](Args)
    Args.lib_in, Args.multihead = "-", 0
    with pytest.raises(SystemExit, match="fine-tune"):
        ns["read_band_pairs"](Args)


def test_the_projection_cache_key_depends_on_the_strings_not_on_the_slicing():
    """The cache key hashes the sequences' bytes in order, whatever the Arrow layout."""
    import hashlib

    np = pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")
    source = _module_source(["_sequence_digest"])
    ns = {"np": np, "pa": pa}
    exec(compile(source, "deeplc_finetune.py", "exec"), ns)  # noqa: S102

    def digest(arr):
        h = hashlib.blake2b(digest_size=20)
        ns["_sequence_digest"](arr, h)
        return h.hexdigest()

    seqs = ["PEPTIDEK", "ACDK", "", "LLLLLLLLLLR"] * 300
    whole = pa.array(seqs, pa.large_string())
    padded = pa.array(["XX"] + seqs + ["YY"], pa.string()).slice(1, len(seqs))
    chunked = pa.chunked_array([pa.array(seqs[:7]), pa.array(seqs[7:])])
    assert digest(whole) == digest(padded) == digest(chunked)
    assert digest(whole) != digest(pa.array(seqs[::-1], pa.large_string()))
    assert digest(whole) != digest(pa.array(seqs[:-1], pa.large_string()))
    # Moving a character between neighbours keeps the concatenation and changes the key.
    moved = list(seqs)
    moved[0], moved[1] = "PEPTIDE", "KACDK"
    assert digest(whole) != digest(pa.array(moved, pa.large_string()))


def _assigned_constants(script, names):
    tree = ast.parse((SCRIPTS / script).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    out[target.id] = ast.literal_eval(node.value)
    return out


def test_the_recipe_selects_the_sequences_the_worker_predicts(tmp_path):
    """`mh_shard_predict.py uniq` keeps exactly the worker's unique standard set.

    The recipe copies the worker's residue and modification rules, because `uniq` runs
    without DeepLC; it used `+` where the worker uses `*`, so a row whose stripped base is
    empty was predicted by the worker and left on its imported iRT by the recipe.
    """
    import subprocess

    h = _arrow_helpers()
    np, pa = h["np"], h["pa"]
    import pyarrow.parquet as pq

    worker = _assigned_constants("deeplc_finetune.py", {"STD_FULL_RE", "MOD_RE"})
    recipe = _assigned_constants("mh_shard_predict.py", {"STD_RE", "MOD_RE"})
    assert recipe == {"STD_RE": worker["STD_FULL_RE"], "MOD_RE": worker["MOD_RE"]}

    pform = _library_peptidoforms(np.random.default_rng(5))
    assert "DECOY_" in pform and "" in pform, "the fixture lost the empty-base rows"
    lib = tmp_path / "lib.parquet"
    pq.write_table(pa.table({"peptidoform": pa.array(pform, pa.string())}), str(lib),
                   row_group_size=150)
    done = subprocess.run([sys.executable, str(SCRIPTS / "mh_shard_predict.py"), "uniq",
                           str(lib), str(tmp_path / "shard_"), "2"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    got = []
    for k in range(2):
        got += pq.read_table(str(tmp_path / "shard_{:02d}.parquet".format(k))).column(
            "seq").to_pylist()
    want = h["unique_standard_bases"](h["library_bases"](pa.chunked_array([pform])))
    assert sorted(got) == sorted(want.to_pylist())
    assert "" in got


def test_only_a_leading_decoy_marker_is_stripped():
    h = _arrow_helpers()
    pa = h["pa"]
    col = pa.chunked_array([pa.array(["DECOY_PEPTIDEK", "PEP_DECOY_TIDEK", "DECOY_DECOY_K"])])
    assert h["library_bases"](col).to_pylist() == ["PEPTIDEK", "PEP_DECOY_TIDEK", "DECOY_K"]


# ------------------------------------------------ sharded prediction (needs DeepLC)

SHARD_NAMES = ["SHARD_AUTO_THREADS"]
SHARD_FUNCS = ["shard_plan", "shard_bounds"]


def _shard_helpers():
    text = (SCRIPTS / "deeplc_finetune.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    namespace = {}
    for node in tree.body:
        wanted = (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in SHARD_NAMES for t in node.targets)
        ) or (isinstance(node, ast.FunctionDef) and node.name in SHARD_FUNCS)
        if wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "deeplc_finetune.py", "exec"),
                 namespace)  # noqa: S102
    missing = [n for n in SHARD_NAMES + SHARD_FUNCS if n not in namespace]
    assert not missing, "deeplc_finetune.py no longer defines {}".format(missing)
    return namespace


def test_the_shard_plan_is_a_function_of_the_request_the_budget_and_the_library():
    h = _shard_helpers()
    plan, bounds = h["shard_plan"], h["shard_bounds"]
    # One process by default, with the whole budget, whatever the library.
    assert plan(1, 64, 5_000_000, 100_000, False)[:2] == (1, 64)
    # K shards of budget / K threads.
    assert plan(8, 64, 5_000_000, 100_000, False)[:2] == (8, 8)
    assert plan(3, 64, 5_000_000, 100_000, False)[:2] == (3, 21)
    # Automatic: one shard per SHARD_AUTO_THREADS threads.
    auto = h["SHARD_AUTO_THREADS"]
    assert plan(0, 64, 5_000_000, 100_000, False)[:2] == (64 // auto, auto)
    assert plan(0, 4, 5_000_000, 100_000, False)[:2] == (1, 4)
    # Never more shards than threads, nor than whole chunks.
    assert plan(16, 4, 5_000_000, 100_000, False)[:2] == (4, 1)
    assert plan(8, 64, 250_000, 100_000, False)[:2] == (3, 8)
    # One chunk, or a GPU: one process with the whole budget.
    assert plan(8, 64, 90_000, 100_000, False)[:2] == (1, 64)
    assert plan(8, 64, 5_000_000, 100_000, True)[:2] == (1, 64)
    with pytest.raises(SystemExit):
        plan(-1, 64, 10, 10, False)
    # Slices start at multiples of the chunk and cover every row once, in order.
    for n, k, chunk in [(1_000, 3, 64), (4_805, 8, 100), (100, 2, 50), (7, 3, 1)]:
        b = bounds(n, k, chunk)
        assert b[0][0] == 0 and b[-1][1] == n
        assert all(a % chunk == 0 for a, _ in b)
        assert all(prev[1] == nxt[0] for prev, nxt in zip(b, b[1:]))
        assert len(b) <= k


def _deeplc_or_skip(minimum=(4, 4, 0)):
    """Skip unless DeepLC >= `minimum` is installed for this interpreter.

    Presence and version come from the package metadata: importing deeplc inside pytest,
    after numpy and pyarrow, is the import order that breaks torch on Windows.
    """
    if importlib.util.find_spec("deeplc") is None or importlib.util.find_spec("torch") is None:
        pytest.skip("DeepLC is not installed for this interpreter")
    import importlib.metadata as md

    raw = md.version("deeplc")
    parts = []
    for piece in raw.split(".")[:3]:
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits or 0))
    if tuple(parts + [0] * (3 - len(parts))) < minimum:
        pytest.skip("DeepLC {} is older than {}".format(raw, ".".join(map(str, minimum))))


def _write_shard_fixture(work, n_base=260):
    """A library of about 500 unique standard sequences and a seed with RT anchors."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(11)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    hyd = dict(zip(aa, rng.normal(0, 1, 20)))
    bases = ["".join(rng.choice(aa, size=int(rng.integers(7, 20)))) + "K" for _ in range(n_base)]
    pf = []
    for i, b in enumerate(bases):
        pf += [b, "DECOY_" + b[::-1]]
        if i % 5 == 0:
            pf += [b[:3] + "U" + b[3:]]          # non-standard: keeps its imported iRT
    n = len(pf)
    pq.write_table(pa.table({
        "candidate_id": pa.array(np.arange(n, dtype=np.uint32)),
        "peptidoform": pa.array(pf, pa.string()),
        "precursor_mz": pa.array(np.sort(rng.uniform(400, 1200, n))),
        "predicted_irt": pa.array(rng.uniform(0, 100, n).astype(np.float32)),
    }), str(work / "lib.parquet"))
    anchors = bases[:120]
    pq.write_table(pa.table({
        "peptidoform": anchors,
        "label": ["target"] * len(anchors),
        "spectrum_q": [0.001] * len(anchors),
        "observed_rt": [600.0 + 60.0 * sum(hyd[c] for c in b) + float(rng.normal(0, 20))
                        for b in anchors],
        "base_peptide_id": pa.array(range(len(anchors)), pa.uint32()),
    }), str(work / "seed.parquet"))


def _predicted_irt(path):
    import numpy as np
    import pyarrow.parquet as pq

    return np.asarray(pq.read_table(str(path)).column("predicted_irt"),
                      dtype=np.float32).view(np.uint32)


@pytest.mark.parametrize("mode", ["base", "multihead", "finetune"])
def test_sharded_prediction_writes_the_column_one_process_writes(tmp_path, mode):
    """K processes at T threads each write the `predicted_irt` one process at T writes.

    Equal threads per process is the condition: torch's CPU kernels round differently at
    different thread counts, so K=1 at T and K>1 at T/K are float-equivalent, not
    bit-identical. The same holds for the multi-head fit, which runs once in the parent
    on --threads: a fit at another thread count moves most rows in the last bits. The
    chunk is shrunk so a 500-sequence library spans several shards, and the GPU is hidden
    so the shards actually run (a GPU gets one process).

    The fine-tune case is the whole-module round trip: the parent `torch.save`s the
    fine-tuned model and every shard `torch.load`s it, which is the part a torch or DeepLC
    release is most likely to break. Both arms train on one CPU thread from seed 0, which
    reproduces the fine-tuned weights exactly, so the comparison is still bit for bit.
    """
    _deeplc_or_skip()
    _write_shard_fixture(tmp_path)
    lib, seed = tmp_path / "lib.parquet", tmp_path / "seed.parquet"
    env = {"MUMDIA_DEEPLC_THREAD_CAP": "0", "CUDA_VISIBLE_DEVICES": "-1"}
    mode_args = {
        "base": ["-", "--no-finetune"],
        "multihead": [str(seed), "--multihead", "80"],
        "finetune": [str(seed), "--epochs", "2", "--patience", "1", "--max-ref", "60",
                     "--seed", "0"],
    }[mode]
    outs = {}
    for k, threads in [(1, 1), (3, 3)]:
        out = tmp_path / "out_k{}.parquet".format(k)
        # The fit runs in the parent on --threads, so that stays at 1 in both arms; only
        # the prediction budget is split.
        run_worker_ok("deeplc_finetune.py", str(lib), mode_args[0], str(out), *mode_args[1:],
                      "--threads", "1", "--predict-threads", str(threads),
                      "--shards", str(k), "--predict-chunk", "64", env=env, timeout=1800)
        outs[k] = out
    one, three = _predicted_irt(outs[1]), _predicted_irt(outs[3])
    assert one.shape == three.shape
    assert (one == three).all(), "{} of {} rows differ".format(int((one != three).sum()),
                                                            len(one))
    s1 = json.loads((tmp_path / "out_k1.parquet.summary.json").read_text(encoding="utf-8"))
    s3 = json.loads((tmp_path / "out_k3.parquet.summary.json").read_text(encoding="utf-8"))
    for key in ("rows", "repredicted", "retained_imported", "retained_non_standard",
                "retained_no_prediction", "unique_predicted"):
        assert s1[key] == s3[key], key
    expected_model = {"base": "the DeepLC base model", "finetune": "the fine-tuned model",
                      "multihead": "the base model calibrated over 80 heads"}[mode]
    assert s1["model"] == s3["model"] == expected_model
    assert s1["shards"]["used"] == 1
    assert s3["shards"]["used"] == 3 and s3["shards"]["threads_per_shard"] == 1
    assert len(s3["shards"]["per_shard"]) == 3
    assert sum(p["rows"] for p in s3["shards"]["per_shard"]) == s3["unique_predicted"]
    for key in ("model_load", "predict", "featurisation", "forward", "rewrite", "write"):
        assert key in s1["timings_s"], key
    assert not list(tmp_path.glob("*.shards.*")), "the shard scratch directory was left behind"


def test_a_failed_shard_fails_the_stage_and_writes_nothing(tmp_path):
    """One shard exits non-zero: the stage fails, no library is written, nothing is left.

    The failure is injected from outside the worker: a `sitecustomize` on PYTHONPATH that
    ends the second shard at interpreter start-up, before it predicts anything.
    """
    _deeplc_or_skip()
    _write_shard_fixture(tmp_path)
    inject = tmp_path / "inject"
    inject.mkdir()
    (inject / "sitecustomize.py").write_text(
        "import os\n"
        "import sys\n"
        "if '--shard-worker' in sys.argv and 'shard_001' in sys.argv[-1]:\n"
        "    os._exit(7)\n",
        encoding="utf-8",
    )
    lib, out = tmp_path / "lib.parquet", tmp_path / "out.parquet"
    from conftest import run_worker

    path = os.pathsep.join(p for p in (str(inject), os.environ.get("PYTHONPATH", "")) if p)
    rc, stdout, stderr = run_worker(
        "deeplc_finetune.py", str(lib), "-", str(out), "--no-finetune", "--threads", "2",
        "--predict-threads", "2", "--shards", "2", "--predict-chunk", "64",
        env={"MUMDIA_DEEPLC_THREAD_CAP": "0", "CUDA_VISIBLE_DEVICES": "-1",
             "PYTHONPATH": path}, timeout=1800)
    assert rc != 0, stdout
    assert "prediction shard 2/2 exited with status 7" in (stdout + stderr)
    assert not out.exists()
    assert not list(tmp_path.glob("*.shards.*")), "the shard scratch directory was left behind"


@pytest.mark.parametrize("mode", ["base", "multihead"])
def test_bands_write_the_whole_library_column_band_by_band(tmp_path, mode):
    """`--bands` over contiguous slices writes what one call over the whole table writes.

    The fit runs once on the same anchors, the union is the whole table's unique set in
    the same order, so with the same threads and chunks every band's `predicted_irt` is
    bit for bit its slice of the single-table output. Each band gets its own summary.
    """
    _deeplc_or_skip()
    import numpy as np
    import pyarrow.parquet as pq

    _write_shard_fixture(tmp_path)
    lib, seed = tmp_path / "lib.parquet", tmp_path / "seed.parquet"
    env = {"MUMDIA_DEEPLC_THREAD_CAP": "0", "CUDA_VISIBLE_DEVICES": "-1"}
    mode_args = {"base": ["--no-finetune"], "multihead": ["--multihead", "80"]}[mode]
    seed_arg = "-" if mode == "base" else str(seed)
    common = ["--threads", "1", "--predict-threads", "1", "--predict-chunk", "64"]
    whole = tmp_path / "whole.parquet"
    run_worker_ok("deeplc_finetune.py", str(lib), seed_arg, str(whole), *mode_args, *common,
                  env=env, timeout=1800)
    table = pq.read_table(str(lib))
    cuts = [0, 150, 151, 400, table.num_rows]
    pairs = []
    for j, (a, b) in enumerate(zip(cuts, cuts[1:])):
        src = tmp_path / "band{}.parquet".format(j)
        pq.write_table(table.slice(a, b - a), str(src))
        pairs.append((src, tmp_path / "band{}_out.parquet".format(j)))
    tsv = tmp_path / "bands.tsv"
    tsv.write_text("".join("{}\t{}\n".format(a, b) for a, b in pairs), encoding="utf-8")
    run_worker_ok("deeplc_finetune.py", "-", seed_arg, "-", "--bands", str(tsv), *mode_args,
                  *common, env=env, timeout=1800)
    want = _predicted_irt(whole)
    got = np.concatenate([_predicted_irt(out) for _, out in pairs])
    assert (want == got).all(), "{} of {} rows differ".format(int((want != got).sum()),
                                                            len(want))
    whole_summary = json.loads((tmp_path / "whole.parquet.summary.json").read_text("utf-8"))
    total = {k: 0 for k in ("rows", "repredicted", "retained_imported")}
    for j, (_, out) in enumerate(pairs):
        s = json.loads((tmp_path / (out.name + ".summary.json")).read_text("utf-8"))
        assert s["bands"] == {"count": len(pairs), "index": j,
                              "union_unique": whole_summary["unique_predicted"]}
        assert s["model"] == whole_summary["model"]
        for k in total:
            total[k] += s[k]
        if mode == "multihead":
            assert s["multihead"]["heads"] == whole_summary["multihead"]["heads"]
    for k in total:
        assert total[k] == whole_summary[k], k


@pytest.mark.parametrize("mode", ["base", "multihead"])
def test_the_projection_cache_reproduces_the_prediction_and_is_read_back(tmp_path, mode):
    """`--projection-cache`: a miss writes the projection, a hit reads it, and the values
    match a plain prediction (bit for bit for the base model's default head, within float
    tolerance for the multi-head transform, which evaluates the heads in numpy)."""
    _deeplc_or_skip()
    import numpy as np

    _write_shard_fixture(tmp_path)
    lib, seed = tmp_path / "lib.parquet", tmp_path / "seed.parquet"
    env = {"MUMDIA_DEEPLC_THREAD_CAP": "0", "CUDA_VISIBLE_DEVICES": "-1"}
    mode_args = {"base": ["-", "--no-finetune"], "multihead": [str(seed), "--multihead", "80"]}[mode]
    common = ["--threads", "1", "--predict-threads", "1", "--predict-chunk", "64"]
    cache = tmp_path / "cache"
    outs = {}
    for arm, extra in [("plain", []), ("miss", ["--projection-cache", str(cache)]),
                       ("hit", ["--projection-cache", str(cache)])]:
        out = tmp_path / "{}.parquet".format(arm)
        run_worker_ok("deeplc_finetune.py", str(lib), mode_args[0], str(out), *mode_args[1:],
                      *common, *extra, env=env, timeout=1800)
        outs[arm] = out
    plain, miss, hit = (_predicted_irt(outs[a]) for a in ("plain", "miss", "hit"))
    assert (miss == hit).all(), "a cache hit changed the values"
    if mode == "base":
        assert (plain == miss).all(), "{} rows differ".format(int((plain != miss).sum()))
    else:
        # Float-equivalent: the heads are evaluated from the cached factors in numpy, and the
        # multi-head splines amplify last-bit differences for the few sequences outside the
        # anchors' range (measured here: 402 of 572 rows identical, 7 above 1e-3 s, the
        # largest 3.7 s on a 4-1602 s scale; a thread-count change does the same, docs/13).
        a = plain.view(np.float32).astype(np.float64)
        b = miss.view(np.float32).astype(np.float64)
        close = np.abs(a - b) <= 1e-3
        assert close.mean() >= 0.97, "{} of {} rows move by more than 1e-3".format(
            int((~close).sum()), len(close))
        assert np.median(np.abs(a - b)) == 0.0
    s_miss = json.loads((tmp_path / "miss.parquet.summary.json").read_text("utf-8"))
    s_hit = json.loads((tmp_path / "hit.parquet.summary.json").read_text("utf-8"))
    assert s_miss["projection_cache"]["used"] and not s_miss["projection_cache"]["hit"]
    assert s_hit["projection_cache"]["hit"]
    assert s_hit["projection_cache"]["key"] == s_miss["projection_cache"]["key"]
    entries = [p for p in cache.iterdir()]
    assert len(entries) == 1 and (entries[0] / "projections.npy").exists(), entries
    assert not [p for p in cache.iterdir() if ".tmp-" in p.name]


def test_a_projection_cache_miss_uses_the_whole_predict_thread_budget(tmp_path):
    """A miss computes the projection in this one process, so it gets `--predict-threads`,
    not the budget of one of the `--shards` it replaces (2 threads here, where a 2-shard plan
    gives each shard 1)."""
    _deeplc_or_skip()
    _write_shard_fixture(tmp_path)
    lib = tmp_path / "lib.parquet"
    out = tmp_path / "out.parquet"
    cache = tmp_path / "cache"
    env = {"MUMDIA_DEEPLC_THREAD_CAP": "0", "CUDA_VISIBLE_DEVICES": "-1"}
    run_worker_ok("deeplc_finetune.py", str(lib), "-", str(out), "--no-finetune",
                  "--threads", "2", "--predict-threads", "2", "--shards", "2",
                  "--predict-chunk", "64", "--projection-cache", str(cache),
                  env=env, timeout=1800)
    summary = json.loads((tmp_path / "out.parquet.summary.json").read_text("utf-8"))
    assert summary["projection_cache"]["used"] and not summary["projection_cache"]["hit"]
    assert summary["shards"]["used"] == 1 and summary["shards"]["plan"] == "projection cache"
    (entry,) = [p for p in cache.iterdir()]
    meta = json.loads((entry / "meta.json").read_text("utf-8"))
    assert meta["torch_threads"] == 2, meta


# ------------------------------------------------ shard clean-up (no DeepLC needed)


def _module_source(names):
    """The top-level statements of `deeplc_finetune.py` that define `names`, as source."""
    text = (SCRIPTS / "deeplc_finetune.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    parts = []
    for node in tree.body:
        defined = (
            [node.name] if isinstance(node, ast.FunctionDef)
            else [t.id for t in node.targets if isinstance(t, ast.Name)]
            if isinstance(node, ast.Assign) else []
        )
        if any(n in names for n in defined):
            # `get_source_segment` leaves out decorators; they start the segment here.
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            lines = text.splitlines()[start - 1:node.end_lineno]
            parts.append("\n".join(lines))
    assert len(parts) == len(names), "deeplc_finetune.py no longer defines {}".format(names)
    return "\n\n".join(parts)


def test_a_shard_ends_itself_when_its_parent_is_killed(tmp_path):
    """Hard-kill a parent that holds a shard's stdin pipe: the shard notices and exits.

    The kill is the one Python cannot intercept (SIGKILL, `TerminateProcess`), so the
    parent's own clean-up never runs and only the child's watchdog can end it. The child
    runs the worker's `_exit_when_the_parent_exits` with `os._exit` wrapped to leave a
    marker, because a process that is not ours to wait on has no other way to report.
    """
    import subprocess
    import time

    source = _module_source(["SHARD_ORPHANED_STATUS", "_exit_when_the_parent_exits"])
    marker = tmp_path / "exited"
    child = (
        "import os, sys, threading, time\n"
        "class _Os:\n"
        "    read = staticmethod(os.read)\n"
        "    @staticmethod\n"
        "    def _exit(status):\n"
        "        with open(sys.argv[1], 'w') as fh:\n"
        "            fh.write(str(status))\n"
        "        os._exit(status)\n"
        "ns = {'os': _Os, 'sys': sys}\n"
        "exec(compile(sys.argv[2], 'deeplc_finetune.py', 'exec'), ns)\n"
        "threading.Thread(target=ns['_exit_when_the_parent_exits'], daemon=True).start()\n"
        "time.sleep(120)\n"
    )
    parent = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2], sys.argv[3]],\n"
        "                     stdin=subprocess.PIPE)\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(120)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", parent, child, str(marker), source],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip().isdigit(), "the parent did not start its child"
        time.sleep(1.0)
        assert not marker.exists(), "the child exited while its parent was alive"
        proc.kill()
        proc.wait(timeout=30)
        deadline = time.monotonic() + 30
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert marker.exists(), "the child outlived its killed parent"
        ns = {}
        exec(compile(source, "deeplc_finetune.py", "exec"), ns)  # noqa: S102
        assert marker.read_text() == str(ns["SHARD_ORPHANED_STATUS"])
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()


def test_sigterm_during_sharding_unwinds_and_restores_the_handler():
    """SIGTERM inside `exit_on_termination_signals` is a SystemExit, so `finally` runs.

    Run in a subprocess: a handler that failed to install would let SIGTERM end pytest.
    An ignored signal (`nohup`) must stay ignored, and the default comes back afterwards.
    """
    import subprocess

    source = _module_source(["exit_on_termination_signals"])
    code = (
        "import contextlib, signal, sys\n"
        "ns = {'contextlib': contextlib, 'signal': signal}\n"
        "exec(compile(sys.argv[1], 'deeplc_finetune.py', 'exec'), ns)\n"
        "guard = ns['exit_on_termination_signals']\n"
        "cleaned = []\n"
        "try:\n"
        "    with guard():\n"
        "        try:\n"
        "            signal.raise_signal(signal.SIGTERM)\n"
        "            for _ in range(1000):\n"
        "                pass\n"
        "        finally:\n"
        "            cleaned.append(True)\n"
        "    print('no exit')\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 128 + signal.SIGTERM, exc.code\n"
        "assert cleaned == [True]\n"
        "assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "with guard():\n"
        "    assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN\n"
        "assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN\n"
        "print('OK')\n"
    )
    done = subprocess.run([sys.executable, "-c", code, source], capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0 and done.stdout.strip() == "OK", (done.stdout, done.stderr)
