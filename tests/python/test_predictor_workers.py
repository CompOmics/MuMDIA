"""Import-order and structural contract tests for the predictor sidecars:
`deeplc_worker.py`, `deeplc_finetune.py`, `ms2pip_worker.py`, `peptdeep_worker.py`.

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


def _load_ms2pip_worker():
    spec = importlib.util.spec_from_file_location(
        "mumdia_ms2pip_worker_rows", SCRIPTS / "ms2pip_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ms2pip_worker_flattens_predictions_like_the_per_fragment_loop():
    """`fragment_rows` must reproduce the loop it replaced, row for row.

    Per result: ions b then y, ordinals 1-based and ascending, the id repeated per
    fragment, `2**x - 0.001` clipped at 0 in float64 and stored as float32; a missing
    ion contributes nothing and a result without predictions is skipped rather than
    crashing the worker. `to_table` must emit `ion_type` as plain utf8, which is the
    only string encoding the engine reads.
    """
    import numpy as np
    import pyarrow as pa

    module = _load_ms2pip_worker()

    class Psm:
        def __init__(self, sid):
            self.spectrum_id = sid

    class Res:
        def __init__(self, sid, pred):
            self.psm = Psm(sid)
            self.predicted_intensity = pred

    results = [
        Res("7", {"b": np.array([-1.0, 0.0]), "y": np.array([-20.0])}),
        Res("9", {"b": None, "y": np.array([2.0])}),
        Res("11", None),
        Res("13", {"b": np.array([]), "y": None}),
    ]
    ids, ions, ords, chgs, ints = module.fragment_rows(results)
    assert ids.tolist() == [7, 7, 7, 9]
    assert ions.tolist() == [0, 0, 1, 1]
    assert ords.tolist() == [1, 2, 1, 1]
    assert chgs.tolist() == [1, 1, 1, 1]
    expected = np.clip(
        np.power(2.0, np.array([-1.0, 0.0, -20.0, 2.0])) - 0.001, 0.0, None
    ).astype(np.float32)
    assert ints.dtype == np.float32
    assert np.array_equal(ints, expected)
    assert ints[2] == 0.0, "2**-20 - 0.001 is negative and must clip to 0"

    tbl = module.to_table(ids, ions, ords, chgs, ints)
    assert tbl.schema.field("ion_type").type == pa.string()
    assert tbl.column("ion_type").to_pylist() == ["b", "b", "y", "y"]
    assert tbl.schema.field("id").type == pa.uint32()
    assert tbl.schema.field("frag_charge").type == pa.int32()
    assert tbl.schema.field("intensity").type == pa.float32()

    # A *ch2 model: the b2/y2 series follow b and y, carrying charge 2.
    ch2 = [Res("3", {"b": np.array([0.0]), "y": np.array([1.0]),
                     "b2": np.array([-2.0]), "y2": np.array([-1.0, -3.0])})]
    ids, ions, ords, chgs, ints = module.fragment_rows(ch2)
    assert ids.tolist() == [3, 3, 3, 3, 3]
    assert ions.tolist() == [0, 1, 0, 1, 1]
    assert ords.tolist() == [1, 1, 1, 1, 2]
    assert chgs.tolist() == [1, 1, 2, 2, 2]

    empty = module.fragment_rows([])
    assert all(a.size == 0 for a in empty)
    assert module.to_table(*empty).num_rows == 0


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
    """`id`, `ion_type`, `ordinal` (1-based), `frag_charge`, `intensity` (linear).

    The Rust side folds this into `HashMap<u32, HashMap<(ion, ordinal, charge), f32>>`
    (`sidecar::run_ms2pip`). A 0-based ordinal shifts every predicted intensity by one
    residue, and log2 intensities left unconverted would be compared against
    max-normalised native values on an entirely different scale. A single-charge model
    emits charge 1 throughout; a `*ch2` model adds the doubly charged series as charge 2,
    which is what lets `predict-frag` put every fragment on the model's scale.
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
    assert set(cols) == {"id", "ion_type", "ordinal", "frag_charge", "intensity"}
    assert set(int(x) for x in cols["id"]) <= {0, 1}
    assert set(cols["ion_type"]) <= {"b", "y"}
    assert int(min(int(o) for o in cols["ordinal"])) == 1, "ordinals must be 1-based"
    assert set(int(z) for z in cols["frag_charge"]) == {1}, "HCD predicts charge 1 only"
    intensity = np.asarray(cols["intensity"], dtype=float)
    assert np.isfinite(intensity).all()
    assert (intensity >= 0.0).all(), "log2 intensities were not converted to linear"

    # The doubly charged series of a *ch2 model arrive as charge 2, same columns.
    out2 = tmp_path / "ms2pip_out_ch2.parquet"
    run_worker_ok("ms2pip_worker.py", inp, out2, "HCDch2")
    cols2 = read_columns(out2)
    assert set(cols2) == set(cols)
    assert set(int(z) for z in cols2["frag_charge"]) == {1, 2}
    n1 = sum(1 for z in cols2["frag_charge"] if int(z) == 1)
    assert n1 == len(cols["frag_charge"]), "the charge-1 series must be as long as before"


# ---------------------------------------------------------------- peptdeep_worker

def _load_peptdeep_worker():
    spec = importlib.util.spec_from_file_location(
        "mumdia_peptdeep_worker", SCRIPTS / "peptdeep_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_peptdeep_worker_module_imports_without_peptdeep_installed():
    """Same deferred-import contract as the MS2PIP worker, for the same reason."""
    module = _load_peptdeep_worker()
    assert callable(module.main)
    text = _source("peptdeep_worker.py")
    for line in text.splitlines():
        assert not line.startswith("import peptdeep"), "peptdeep hoisted to module scope"
        assert not line.startswith("from peptdeep"), "peptdeep hoisted to module scope"
        assert not line.startswith("from alphabase"), "alphabase hoisted to module scope"


def test_peptdeep_worker_translates_proforma_to_alphabase():
    """`parse_peptidoform` must map MuMDIA's ProForma onto alphabase's convention.

    MuMDIA writes UniMod NAMES in brackets and alphabase spells a modification
    `Name@Residue` with 1-based sites, 0 for the N terminus and -1 for the C terminus,
    so the mapping is mechanical. Getting a site off by one would put the modification
    on the wrong residue and predict a plausible-looking wrong spectrum, which no
    later stage could detect.
    """
    module = _load_peptdeep_worker()
    p = module.parse_peptidoform

    assert p("PEPTIDEK") == ("PEPTIDEK", "", "")
    assert p("PEC[Carbamidomethyl]TIDE") == (
        "PECTIDE",
        "Carbamidomethyl@C",
        "3",
    ), "1-based site on the residue the bracket follows"
    assert p("[Acetyl]-PEPK") == ("PEPK", "Acetyl@Any_N-term", "0")
    assert p("PEPK-[Amidated]") == ("PEPK", "Amidated@Any_C-term", "-1")
    # Several modifications keep their order and their own sites.
    assert p("M[Oxidation]PEC[Carbamidomethyl]K") == (
        "MPECK",
        "Oxidation@M;Carbamidomethyl@C",
        "1;4",
    )


def test_peptdeep_worker_refuses_a_mass_delta_rather_than_guessing():
    """A bare delta names no modification, and the nearest UniMod is not necessarily
    the right one, so it is left unpredicted instead of silently substituted.

    The engine drops a candidate no predictor covered together with its target/decoy
    pair and fails the run above 2%, so refusing here is reported, not lost.
    """
    module = _load_peptdeep_worker()
    for bad in ["PEPT[+79.96633]IDEK", "PEC[57.021464]TIDE", "PEPT[-18.0106]IDEK"]:
        assert module.parse_peptidoform(bad) is None, bad
    # An unparseable string is refused rather than raising.
    assert module.parse_peptidoform("") is None
    assert module.parse_peptidoform("PEPT[Oxidation") is None


def test_peptdeep_worker_always_asks_for_both_fragment_charges():
    """`FRAG_TYPES` must keep the doubly charged series.

    The engine decides a model predicted charge 2 by finding ANY charge-2 key for that
    candidate; without one it fills those fragments from the native heuristic and
    max-normalises the two charge groups separately. That is the measured `HCD2021`
    failure: 78.6% of kept fragments heuristic and 0 confident PSMs at 1% on a real
    run. This is a contract of the worker, not a setting.
    """
    module = _load_peptdeep_worker()
    assert module.FRAG_TYPES == ["b_z1", "b_z2", "y_z1", "y_z2"]
    assert {c for c, _, _ in module.SERIES} == set(module.FRAG_TYPES)
    assert {z for _, _, z in module.SERIES} == {1, 2}
    assert {i for _, i, _ in module.SERIES} == {0, 1}


def test_peptdeep_worker_maps_fragment_rows_to_ordinals():
    """Row `i` of a precursor's slice is b(i+1) and y(nAA-1-i).

    Verified against AlphaPeptDeep's own `fragment_mz_df` for a 9-mer: row 0 carries
    b1 (72.0444) and y8, row 7 carries b8 and y1 (147.1128). An inverted y ordinal
    would pair every predicted intensity with the wrong fragment m/z, which the
    engine cannot notice: both series exist and both are plausible.
    """
    import pandas as pd

    import pyarrow as pa

    module = _load_peptdeep_worker()
    n_aa = 5
    width = n_aa - 1
    precursor_df = pd.DataFrame(
        {
            "mumdia_id": [7],
            "nAA": [n_aa],
            "frag_start_idx": [0],
            "frag_stop_idx": [width],
        }
    )
    intensity_df = pd.DataFrame(
        {
            "b_z1": [0.1, 0.2, 0.3, 0.4],
            "b_z2": [0.0, 0.0, 0.0, 0.0],
            "y_z1": [0.5, 0.6, 0.7, 0.8],
            "y_z2": [0.9, 0.0, 0.0, 0.0],
        }
    )
    ids, ions, ords, chgs, ints = module.fragment_rows(precursor_df, intensity_df)
    assert set(ids.tolist()) == {7}
    table = {}
    for ion, ordinal, charge, value in zip(ions, ords, chgs, ints):
        table[("by"[ion], int(ordinal), int(charge))] = round(float(value), 4)

    assert table[("b", 1, 1)] == 0.1 and table[("b", 4, 1)] == 0.4
    # y counts from the other end: row 0 is y(n-1) = y4, row 3 is y1.
    assert table[("y", 4, 1)] == 0.5 and table[("y", 1, 1)] == 0.8
    assert table[("y", 4, 2)] == 0.9
    # Zeros are kept: dropping a precursor's whole charge-2 series would make the
    # engine treat the model as charge-1-only.
    assert table[("b", 1, 2)] == 0.0
    assert len(table) == 4 * width

    arrow = module.to_table((ids, ions, ords, chgs, ints))
    assert arrow.schema.field("ion_type").type == pa.string(), "utf8, not dictionary"
    assert arrow.column_names == [
        "id",
        "ion_type",
        "ordinal",
        "frag_charge",
        "intensity",
    ], "the same five columns the MS2PIP worker writes"


def test_peptdeep_worker_end_to_end(tmp_path):
    """The real worker over the real model, when peptdeep is installed.

    Skipped without it. What this adds over the unit tests is the contract the engine
    actually reads: five columns, snappy, ids only from the request, both fragment
    charges present, intensities finite and in [0, 1].
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from conftest import read_columns, run_worker_ok

    importorskip_any("peptdeep")

    inp = tmp_path / "peptdeep_in.parquet"
    out = tmp_path / "peptdeep_out.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array([0, 1], pa.uint32()),
                "peptidoform": pa.array(
                    ["ACDEFGHIK", "PEC[Carbamidomethyl]TIDEK"], pa.string()
                ),
                "charge": pa.array([2, 3], pa.int32()),
            }
        ),
        inp,
        compression="snappy",
    )
    run_worker_ok("peptdeep_worker.py", inp, out, "generic", "30", "Lumos", "2")

    cols = read_columns(out)
    assert set(cols) == {"id", "ion_type", "ordinal", "frag_charge", "intensity"}
    assert set(int(x) for x in cols["id"]) <= {0, 1}
    assert set(cols["ion_type"]) <= {"b", "y"}
    assert min(int(o) for o in cols["ordinal"]) == 1, "ordinals must be 1-based"
    assert set(int(z) for z in cols["frag_charge"]) == {1, 2}, "both charges, always"
    intensity = np.asarray(cols["intensity"], dtype=float)
    assert np.isfinite(intensity).all()
    assert ((intensity >= 0.0) & (intensity <= 1.0)).all()
    assert intensity.max() > 0.0, "an all-zero library is not a prediction"


def test_peptdeep_worker_refuses_an_unknown_instrument(tmp_path):
    """AlphaPeptDeep maps an unknown instrument onto its default silently, which would
    predict a different spectrum than the configuration asked for and say nothing."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from conftest import run_worker

    importorskip_any("peptdeep")

    inp = tmp_path / "peptdeep_in.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array([0], pa.uint32()),
                "peptidoform": pa.array(["PEPTIDEK"], pa.string()),
                "charge": pa.array([2], pa.int32()),
            }
        ),
        inp,
        compression="snappy",
    )
    rc, _, err = run_worker(
        "peptdeep_worker.py",
        inp,
        tmp_path / "out.parquet",
        "generic",
        "30",
        "NoSuchInstrument",
        "1",
    )
    assert rc != 0
    assert "unknown instrument" in err.lower()


def test_peptdeep_worker_rejects_a_bad_device_request(tmp_path):
    """`MUMDIA_PEPTDEEP_DEVICE` follows `MUMDIA_NN_DEVICE`: auto|cuda|cpu, and `cuda`
    on a CPU-only torch errors rather than quietly measuring the wrong device.

    No dependency is needed to reach this, deliberately: the name is checked before
    the worker imports torch, peptdeep or alphabase, so a mistyped variable reports
    itself instead of surfacing as a missing module from a stack it never needed.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from conftest import run_worker

    inp = tmp_path / "peptdeep_in.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array([0], pa.uint32()),
                "peptidoform": pa.array(["PEPTIDEK"], pa.string()),
                "charge": pa.array([2], pa.int32()),
            }
        ),
        inp,
        compression="snappy",
    )
    rc, _, err = run_worker(
        "peptdeep_worker.py",
        inp,
        tmp_path / "out.parquet",
        env={"MUMDIA_PEPTDEEP_DEVICE": "tpu"},
    )
    assert rc != 0
    assert "must be auto, cuda or cpu" in err
