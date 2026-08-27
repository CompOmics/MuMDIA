"""Import-order and structural contract tests for the predictor sidecars:
`deeplc_worker.py`, `deeplc_finetune.py`, `ms2pip_worker.py`.

The static assertions here need no ML dependency and always run. They exist
because the failure they guard keeps recurring and is invisible in review: in
both DeepLC workers `import deeplc` MUST execute before numpy and pyarrow at
module scope. DeepLC 4.x is torch-backed, and on Windows importing numpy (and
the pyarrow that follows) first aborts torch's DLL initialisation outright:

    OSError: [WinError 1114] A dynamic link library (DLL) initialization
    routine failed. Error loading "...\\torch\\lib\\c10.dll" or one of its
    dependencies.

`deeplc_worker.py` previously deferred `import deeplc` into `main()`, which put
it after the module-level numpy/pyarrow and reproduced the crash. The fault
stayed latent because imported-library mode skips predict-frag entirely, so only
a FASTA-mode library build reaches this worker. `mumdia doctor` cannot catch it
either: it probes with `importlib.util.find_spec`, which only asks whether a
module is importable. An import sorter or a routine tidy-up is enough to
reintroduce it, and the predictor sidecars have no strict gate and no fallback,
so the crash aborts the whole run.

`deeplc_finetune.py` has a second ordering rule of the same kind: the
`OMP/MKL/OPENBLAS` thread caps must be set before numpy and torch are imported.
Without them numpy's OpenBLAS (GNU OpenMP) and torch's Intel OpenMP each spawn a
full thread pool and oversubscribe the CPU during fine-tuning's sustained
backward pass, which crashed the development machine intermittently.
"""

from __future__ import annotations

import importlib.util

import pytest

from conftest import (
    SCRIPTS,
    first_line_matching,
    import_module_in_fresh_interpreter,
    importorskip_any,
)

DEEPLC_WORKERS = ["deeplc_worker.py", "deeplc_finetune.py"]


def _source(name):
    return (SCRIPTS / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("script", DEEPLC_WORKERS)
def test_deeplc_is_imported_before_numpy_and_pyarrow(script):
    """`import deeplc` must appear before numpy and pyarrow at module scope.

    If it does not, a FASTA-mode library build on Windows dies with
    `OSError: [WinError 1114] ... c10.dll` before predicting anything, and
    because the predictor sidecars abort the run rather than falling back, the
    whole search is lost. `mumdia doctor` reports green either way.
    """
    text = _source(script)
    deeplc_at = first_line_matching(text, r"import deeplc\b")
    assert deeplc_at is not None, "{} no longer imports deeplc".format(script)
    for module in ("numpy", "pyarrow", "torch"):
        at = first_line_matching(text, r"import {}\b".format(module))
        if at is None:
            continue
        assert deeplc_at < at, (
            "{}: `import deeplc` is on line {} but `import {}` is on line {}; "
            "deeplc must load torch first or torch's DLL init fails on Windows"
            .format(script, deeplc_at + 1, module, at + 1)
        )


@pytest.mark.parametrize("script", DEEPLC_WORKERS)
def test_deeplc_import_is_at_module_scope_not_inside_a_function(script):
    """The `import deeplc` line must be unindented.

    Deferring it into `main()` is exactly the regression that reintroduced the
    crash: the line still reads `import deeplc`, an ordering check on line
    numbers alone would still pass, but numpy and pyarrow have already been
    imported by then.
    """
    lines = _source(script).splitlines()
    module_level = [ln for ln in lines
                    if ln.startswith("import deeplc") or ln.startswith("from deeplc")]
    assert module_level, (
        "{}: deeplc is imported only inside a function; the module-level "
        "ordering guarantee is gone".format(script)
    )


def test_deeplc_finetune_caps_thread_pools_before_importing_numpy_or_torch():
    """The OpenMP thread caps must be set before numpy and torch are imported.

    numpy's OpenBLAS links GNU OpenMP while torch ships Intel OpenMP; they
    coexist only under `KMP_DUPLICATE_LIB_OK=TRUE`. Setting the caps after the
    import is a no-op, both runtimes then spin a full thread pool, and the
    sustained backward pass of a fine-tune crashes the machine intermittently,
    which looks like a hardware fault rather than an import-order bug.
    """
    text = _source("deeplc_finetune.py")
    caps = [
        r'os\.environ\["OMP_NUM_THREADS"\]',
        r'os\.environ\["OPENBLAS_NUM_THREADS"\]',
        r'os\.environ\["MKL_NUM_THREADS"\]',
        r'os\.environ\["KMP_DUPLICATE_LIB_OK"\]',
    ]
    numpy_at = first_line_matching(text, r"import numpy\b")
    torch_at = first_line_matching(text, r"import torch\b")
    assert numpy_at is not None and torch_at is not None
    for cap in caps:
        at = first_line_matching(text, r"\s*" + cap)
        assert at is not None, "the thread cap {} is gone".format(cap)
        assert at < numpy_at and at < torch_at, (
            "the thread cap on line {} runs after numpy/torch import and is "
            "therefore a no-op".format(at + 1)
        )


def test_ms2pip_worker_defers_its_heavy_imports_into_main():
    """`ms2pip` and `psm_utils` must be imported inside `main()`, not at module scope.

    They are imported inside `main` so the module itself loads on any
    interpreter, which is what lets `mumdia doctor` and any tooling inspect the
    worker without the MS2PIP environment. Hoisting them to module scope would
    turn a missing MS2PIP into an import error at a different point in the run
    than the interpreter probe reports.
    """
    text = _source("ms2pip_worker.py")
    for line in text.splitlines():
        assert not line.startswith("import ms2pip"), "ms2pip hoisted to module scope"
        assert not line.startswith("from ms2pip"), "ms2pip hoisted to module scope"
        assert not line.startswith("from psm_utils"), "psm_utils hoisted to module scope"
    assert "from ms2pip import predict_batch" in text
    assert "from psm_utils import PSM, PSMList" in text


@pytest.mark.parametrize(
    "script", ["ms2pip_worker.py", "deeplc_worker.py", "deeplc_finetune.py"]
)
def test_workers_guard_main_for_the_windows_spawn_start_method(script):
    """Every predictor worker needs an `if __name__ == "__main__"` guard.

    `ms2pip_worker.py` calls `predict_batch(..., processes=...)`, and on Windows
    multiprocessing uses `spawn`: each child re-imports the module, so without
    the guard the module body runs again in every child and the worker forks
    without bound instead of predicting.
    """
    assert '__name__ == "__main__"' in _source(script) or \
        "__name__ == '__main__'" in _source(script)


def test_ms2pip_worker_module_imports_without_ms2pip_installed():
    """The module must import with only numpy and pyarrow available.

    This is the flip side of the deferred import: a worker whose module body
    needed MS2PIP could not be loaded for inspection, and the failure would come
    from the module import rather than from the interpreter resolution that is
    supposed to report a missing package by name.
    """
    spec = importlib.util.spec_from_file_location(
        "mumdia_ms2pip_worker", SCRIPTS / "ms2pip_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


@pytest.mark.parametrize("script", DEEPLC_WORKERS)
def test_deeplc_worker_imports_in_a_fresh_interpreter(script):
    """With DeepLC present, a fresh-interpreter module import must succeed.

    This is the only test that exercises the real import order rather than the
    source text, and it has to run in a subprocess: inside pytest numpy and
    pyarrow are already loaded, which is the broken order, and importing a
    DeepLC worker in that state reproduces `WinError 1114` and would fail for
    the wrong reason. The subprocess is also how the engine runs the worker, so
    a green static check plus a red import here means the ordering rule moved.
    """
    # Gate on PRESENCE, not importability: importing deeplc inside pytest is
    # itself the broken order (numpy and pyarrow are already loaded), so an
    # importorskip here would skip on the very failure under test.
    for package in (["deeplc"] if script == "deeplc_worker.py"
                    else ["deeplc", "torch", "psm_utils"]):
        if importlib.util.find_spec(package) is None:
            pytest.skip("{} is not installed".format(package))

    rc, out, err = import_module_in_fresh_interpreter(script)
    if rc != 0:
        assert "WinError 1114" not in err and "c10.dll" not in err, (
            "{} reproduced the documented torch DLL-init failure: `import deeplc` "
            "no longer runs before numpy/pyarrow at module scope\n"
            "--- stderr ---\n{}".format(script, err)
        )
        tail = [ln for ln in err.strip().splitlines() if ln.strip()]
        pytest.skip(
            "{} is not importable in this environment for an unrelated reason: "
            "{}".format(script, tail[-1] if tail else "no stderr")
        )
    assert "MODULE_IMPORT_OK" in out, (
        "{} imported without confirming\n--- stdout ---\n{}".format(script, out)
    )


def test_ms2pip_worker_writes_the_documented_output_schema(tmp_path):
    """`id`, `ion_type`, `ordinal` (1-based), `intensity` (linear).

    The Rust side folds this into `HashMap<u32, HashMap<(ion, ordinal), f32>>`
    and looks up charge-1 fragments only (`predict_frag.rs:356-363`). A 0-based
    ordinal shifts every predicted intensity by one residue, and log2 intensities
    left unconverted would be compared against max-normalised native values on
    an entirely different scale.
    """
    importorskip_any("ms2pip", "the MS2PIP sidecar needs a usable ms2pip")
    importorskip_any("psm_utils", "the MS2PIP sidecar needs a usable psm_utils")
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from conftest import read_columns, run_worker_ok

    inp = tmp_path / "ms2pip_in.parquet"
    out = tmp_path / "ms2pip_out.parquet"
    pq.write_table(
        pa.table({
            "id": pa.array([0, 1], pa.uint32()),
            "peptidoform": pa.array(["PEPTIDEK", "SAMPLER"], pa.string()),
            "charge": pa.array([2, 2], pa.int32()),
        }),
        str(inp), compression="snappy",
    )
    run_worker_ok("ms2pip_worker.py", inp, out, "HCD")
    cols = read_columns(out)
    assert set(cols) == {"id", "ion_type", "ordinal", "intensity"}
    assert set(int(x) for x in cols["id"]) <= {0, 1}
    assert set(cols["ion_type"]) <= {"b", "y"}
    assert int(min(int(o) for o in cols["ordinal"])) == 1, "ordinals must be 1-based"
    intensity = np.asarray(cols["intensity"], dtype=float)
    assert np.isfinite(intensity).all()
    assert (intensity >= 0.0).all(), "log2 intensities were not converted to linear"
