"""Shared fixtures and contract helpers for the Python sidecar tests.

The workers in `scripts/` are invoked by the Rust engine as subprocesses over a
positional-argument file contract (`docs/13_sidecars.md`, "Argv contract"): the
caller writes an input file, runs `python <worker> <argv...>`, and reads an
output Parquet keyed by `id` / `candidate_id` / `row_id`. The files on disk are
the entire interface, so these tests build tiny synthetic inputs, invoke the
worker exactly as the engine does, and assert the on-disk output contract.

Nothing here imports a worker's heavy dependency. Each test skips itself with
`pytest.importorskip` when the package it needs is absent, so the suite is
useful in a minimal environment and skips honestly rather than failing.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# tests/python/conftest.py -> tests/python -> tests -> repository root
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

# Monoisotopic residue masses, transcribed independently of
# `scripts/make_reverse_decoys.py`. A fragment m/z fixture built from this table
# is therefore a real cross-check of that worker's own calculator (which aborts
# above 5 ppm disagreement with the library) rather than a tautology.
RESIDUE_MASS = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047679,
    "C": 103.009185,
    "L": 113.084064,
    "I": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "Q": 128.058578,
    "K": 128.094963,
    "E": 129.042593,
    "M": 131.040485,
    "H": 137.058912,
    "F": 147.068414,
    "R": 156.101111,
    "Y": 163.063329,
    "W": 186.079313,
}
MOD_MASS = {"Carbamidomethyl": 57.021464, "Oxidation": 15.994915}
WATER = 18.010565
PROTON = 1.007276

AA20 = "ACDEFGHIKLMNPQRSTVWY"

# ---------------------------------------------------------------------------
# worker invocation
# ---------------------------------------------------------------------------


def run_worker(script, *args, env=None, cwd=None, timeout=900):
    """Run a sidecar the way the Rust caller does and return (rc, stdout, stderr).

    `script` is a file name under `scripts/` or an absolute path. `env` entries
    are overlaid on the inherited environment, matching how the engine injects
    `MUMDIA_*` knobs. The interpreter is always `sys.executable`, so the suite
    tests whatever Python is running it and never a hard-coded machine path.
    """
    path = Path(script)
    if not path.is_absolute():
        path = SCRIPTS / script
    cmd = [sys.executable, str(path), *[str(a) for a in args]]
    child_env = dict(os.environ)
    if env:
        child_env.update({k: str(v) for k, v in env.items()})
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        cwd=str(cwd) if cwd is not None else None,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_worker_ok(script, *args, env=None, cwd=None, timeout=900):
    """`run_worker` plus the exit-code assertion, with stderr in the message.

    A nonzero exit from a predictor or MBR worker aborts the whole engine run
    (`docs/13_sidecars.md`, "Failure behavior"), so a failing assertion here
    stands for an aborted production run; the captured stderr is the only
    diagnostic the engine itself would surface.
    """
    rc, out, err = run_worker(script, *args, env=env, cwd=cwd, timeout=timeout)
    assert rc == 0, (
        "{} exited {}\n--- argv ---\n{}\n--- stdout ---\n{}\n--- stderr ---\n{}".format(
            Path(script).name, rc, list(args), out, err
        )
    )
    return out, err


def importorskip_any(name, reason=None):
    """`pytest.importorskip` that also skips when the package is broken.

    `importorskip` skips only on `ImportError`. An optional dependency that is
    installed but unimportable for another reason (an incompatible pydantic, a
    stale tensorflow-backed DeepLC) would otherwise fail these tests for a
    reason that has nothing to do with the contract under test. `mumdia doctor`
    is the tool for environment health; the skip reason names the real error.
    """
    try:
        return __import__(name)
    except Exception as exc:  # noqa: BLE001 - any import failure means "cannot test"
        pytest.skip(
            "{}: {}{}".format(
                reason or "{} is not usable here".format(name),
                type(exc).__name__,
                ": {}".format(exc).splitlines()[0] if str(exc) else "",
            )
        )


def import_module_in_fresh_interpreter(script, timeout=300):
    """Import a worker at module scope in a brand-new interpreter.

    The import-order contract can only be tested this way. Inside pytest, numpy
    and pyarrow are already loaded by the time any test runs, so an in-process
    import of a DeepLC worker exercises the wrong order and, on Windows,
    reproduces the very `WinError 1114` the ordering exists to prevent. A fresh
    subprocess is also exactly how the engine runs the worker.
    """
    path = Path(script)
    if not path.is_absolute():
        path = SCRIPTS / script
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('w', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "assert callable(m.main)\n"
        "print('MODULE_IMPORT_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


@pytest.fixture(scope="session")
def scripts_dir():
    """Directory holding the engine-invoked workers."""
    assert SCRIPTS.is_dir(), "scripts/ not found (looked for {})".format(SCRIPTS)
    return SCRIPTS


@pytest.fixture(scope="session", name="run_worker")
def _run_worker_fixture():
    return run_worker


@pytest.fixture(scope="session", name="run_worker_ok")
def _run_worker_ok_fixture():
    return run_worker_ok


@pytest.fixture
def work(tmp_path):
    """Per-test scratch directory (nothing is written into the repository)."""
    return tmp_path


# ---------------------------------------------------------------------------
# parquet contract helpers
# ---------------------------------------------------------------------------


def parquet_codecs(path):
    """Every compression codec used by any column chunk in the file."""
    md = pq.ParquetFile(str(path)).metadata
    return {
        md.row_group(g).column(c).compression
        for g in range(md.num_row_groups)
        for c in range(md.row_group(g).num_columns)
    }


def assert_engine_readable_parquet(path):
    """SNAPPY (or uncompressed) only: the engine's parquet has no other codec.

    If this fails the engine rejects the file at read with
    `Parquet error: Disabled feature at compile time: zstd`, so a library or
    sidecar output that looks fine locally cannot be loaded at all
    (`CLAUDE.md`, "Sidecar and IO contracts").
    """
    codecs = parquet_codecs(path)
    assert codecs <= {"SNAPPY", "UNCOMPRESSED"}, (
        "{} uses codecs {}; the engine's parquet build compiles only snappy, so "
        "anything else fails at read".format(Path(path).name, sorted(codecs))
    )


def assert_string_columns_utf8(path):
    """String columns must be arrow `utf8`, never `large_utf8`.

    `Table::str` downcasts to `StringArray` only and rejects a 64-bit-offset
    column with `column '<name>' is not utf8`, so a helper written on a library
    that defaults to `large_utf8` (Polars) produces an unloadable library.
    """
    schema = pq.read_schema(str(path))
    offenders = [f.name for f in schema if pa.types.is_large_string(f.type)]
    assert not offenders, (
        "{} has large_utf8 string column(s) {}; the engine accepts arrow utf8 "
        "only".format(Path(path).name, offenders)
    )


# Why the string-type assertions above matter, and why they are plain assertions
# rather than expected failures.
#
# `DataFrame.to_parquet` chooses the string width itself, and pandas 3 adopted the
# arrow-backed `str` dtype, which round-trips as `large_string`. The engine accepts
# arrow `utf8` only, so on any modern pandas every library helper silently produced
# a file that fails at load with `column 'peptidoform' is not utf8`. The failure was
# version-dependent, which is worse than a plain bug: it worked for whoever wrote
# the script and broke for the next person.
#
# The helpers now write through `scripts/_lib_io.write_engine_parquet`, which
# narrows the large variants and pins snappy. These assertions are what keeps that
# true, so they must fail loudly if a helper goes back to `to_parquet`.


def read_columns(path, columns=None):
    """Read a parquet file into a dict of numpy arrays (strings as lists)."""
    table = pq.read_table(str(path), columns=columns)
    out = {}
    for name in table.column_names:
        col = table.column(name)
        if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
            out[name] = col.to_pylist()
        else:
            out[name] = np.asarray(col.to_pylist())
    return out


def assert_complete_finite_coverage(path, expected_rows, key="candidate_id", sidecar=""):
    """Mirror of `align_sidecar_scores` (`rescore.rs:1046-1082`).

    The Rust caller bails on a coverage mismatch, a duplicate row id, an
    out-of-range row id or a non-finite score. A sidecar that violates this
    kills the run under the production default `rescore.strict = true`, and
    silently replaces the requested classifier with `native_tda` when strict is
    off. Returns the output table.
    """
    table = pq.read_table(str(path))
    assert table.num_rows == expected_rows, (
        "{}: expected {} output rows, got {}".format(sidecar, expected_rows, table.num_rows)
    )
    ids = [int(x) for x in table.column(key).to_pylist()]
    scores = np.asarray(table.column("score").to_pylist(), dtype=np.float64)
    assert len(ids) == len(scores) == expected_rows
    assert sorted(ids) == list(range(expected_rows)), (
        "{}: {} is not an exact, duplicate-free cover of 0..{}".format(
            sidecar, key, expected_rows
        )
    )
    assert np.isfinite(scores).all(), "{}: returned non-finite score(s)".format(sidecar)
    return table


def assert_library_load_invariants(
    prec_path, frag_path, require_both_labels=True, check_string_type=False
):
    """The two hard preconditions of `index.rs load()`, plus the codec contract.

    `candidate_id` must be the contiguous row-aligned range `0..ncand`
    (`index.rs:112-125`) and precursors must ascend by `precursor_mz`
    (`index.rs:215-231`). Both are hard errors that abort the run, and a broken
    m/z ordering additionally makes the fragment index's `partition_point`
    search return the wrong candidate window.
    """
    prec = pq.read_table(str(prec_path))
    n = prec.num_rows
    cid = [int(x) for x in prec.column("candidate_id").to_pylist()]
    assert cid == list(range(n)), (
        "precursor candidate_id is not the contiguous range 0..ncand "
        "(index.rs:112-125 bails on the first offending row)"
    )
    mz = np.asarray(prec.column("precursor_mz").to_pylist(), dtype=np.float64)
    assert np.all(np.diff(mz) >= 0.0), (
        "precursors are not ascending by precursor_mz (index.rs:215-231)"
    )
    frag = pq.read_table(str(frag_path))
    fcid = np.asarray(frag.column("candidate_id").to_pylist(), dtype=np.int64)
    assert fcid.size == 0 or (fcid.min() >= 0 and fcid.max() < n), (
        "a fragment references a candidate_id past the precursor count "
        "(index.rs:133-139)"
    )
    labels = set(prec.column("label").to_pylist())
    if require_both_labels:
        assert labels == {"target", "decoy"}, (
            "library labels are {}; FDR needs both populations".format(sorted(labels))
        )
    for path in (prec_path, frag_path):
        assert_engine_readable_parquet(path)
        if check_string_type:
            assert_string_columns_utf8(path)
    return prec, frag


# ---------------------------------------------------------------------------
# synthetic PSMs: the PIN / parquet handoff for the rescore sidecars
# ---------------------------------------------------------------------------


def peptide(i):
    """Deterministic distinct peptide sequence for row `i` (standard residues)."""
    body = "".join(AA20[(i // (20 ** k)) % 20] for k in range(4))
    return body + "K"


def synthetic_psms(n_targets=600, n_decoys=600, n_features=5, seed=0):
    """Separable synthetic PSMs shared by the PIN and parquet handoff writers.

    Feature 0 carries the signal (targets shifted up); the rest are noise. The
    separation must be strong enough that every fold selects positives at the
    training FDR, otherwise `nn_rescore_worker` raises "selected no positive
    targets" and mokapot cannot fit its model, neither of which is the contract
    under test.
    """
    rng = np.random.default_rng(seed)
    n = n_targets + n_decoys
    labels = np.array([1] * n_targets + [-1] * n_decoys, dtype=np.int32)
    feats = rng.normal(0.0, 1.0, size=(n, n_features))
    feats[:n_targets, 0] += 4.0
    feats[:n_targets, 1] += 1.0
    pforms = []
    proteins = []
    for i in range(n):
        seq = peptide(i)
        if labels[i] == 1:
            pforms.append(seq)
            proteins.append("PROT{}_ECOLI".format(i % 37))
        else:
            pforms.append("DECOY_" + seq)
            proteins.append("DECOY_PROT{}_ECOLI".format(i % 37))
    mz = 400.0 + np.arange(n, dtype=np.float64) * 0.37
    return {
        "n": n,
        "n_targets": n_targets,
        "n_decoys": n_decoys,
        "labels": labels,
        "features": feats,
        "feature_names": ["feat_{}".format(j) for j in range(n_features)],
        "peptidoform": pforms,
        "protein": proteins,
        "precursor_mz": mz,
    }


def write_pin(path, psms):
    """Write the tab-separated PIN exactly as `rescore.rs:978-999` does.

    Fixed columns, `SpecId = psm_<i>` / `ScanNr = <i>` on the unique flat row
    index (never `candidate_id`, which repeats across runs), `ExpMass` and
    `CalcMass` both the precursor m/z at 5 decimals, features at 6 decimals,
    `Peptide` with Percolator flanking dots, one `Proteins` column.
    """
    names = psms["feature_names"]
    with open(str(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "SpecId\tLabel\tScanNr\tExpMass\tCalcMass\t"
            + "\t".join(names)
            + "\tPeptide\tProteins\n"
        )
        for i in range(psms["n"]):
            mz = psms["precursor_mz"][i]
            row = [
                "psm_{}".format(i),
                str(int(psms["labels"][i])),
                str(i),
                "{:.5f}".format(mz),
                "{:.5f}".format(mz),
            ]
            row += ["{:.6f}".format(v) for v in psms["features"][i]]
            row += ["-.{}.-".format(psms["peptidoform"][i]), psms["protein"][i]]
            fh.write("\t".join(row) + "\n")
    return path


def write_features_parquet(path, psms):
    """Write the `rescore.handoff = parquet` feature table (`rescore.rs:836-862`).

    Same column names and order as the PIN, `Label`/`ScanNr` int32, features
    float32; accepted by `nn_rescore_worker` only.
    """
    n = psms["n"]
    cols = {
        "SpecId": pa.array(["psm_{}".format(i) for i in range(n)], pa.string()),
        "Label": pa.array(psms["labels"].astype(np.int32), pa.int32()),
        "ScanNr": pa.array(np.arange(n, dtype=np.int32), pa.int32()),
        "ExpMass": pa.array(psms["precursor_mz"], pa.float64()),
        "CalcMass": pa.array(psms["precursor_mz"], pa.float64()),
    }
    for j, name in enumerate(psms["feature_names"]):
        cols[name] = pa.array(psms["features"][:, j].astype(np.float32), pa.float32())
    cols["Peptide"] = pa.array(
        ["-.{}.-".format(p) for p in psms["peptidoform"]], pa.string()
    )
    cols["Proteins"] = pa.array(psms["protein"], pa.string())
    pq.write_table(pa.table(cols), str(path), compression="snappy")
    return path


# ---------------------------------------------------------------------------
# synthetic scored / psms tables: the MBR contract
# ---------------------------------------------------------------------------


def write_scored_table(path, rows, extra_q=None):
    """Write an experiment-wide scored_combined table for `mbr_worker.py`.

    `rows` holds equal-length sequences for the columns MBR reads
    (`mbr_worker.py:81-82`): candidate_id, source, label, q_value, peptidoform,
    charge, protein_group. `extra_q` adds further PSM-level q columns
    (`run_psm_q`, `experiment_psm_q`) that the M5 augmentation must also lower.
    """
    cols = {
        "candidate_id": pa.array(
            np.asarray(rows["candidate_id"], dtype=np.uint32), pa.uint32()
        ),
        "source": pa.array(np.asarray(rows["source"], dtype=np.uint32), pa.uint32()),
        "label": pa.array(list(rows["label"]), pa.string()),
        "q_value": pa.array(np.asarray(rows["q_value"], dtype=np.float64), pa.float64()),
        "peptidoform": pa.array(list(rows["peptidoform"]), pa.string()),
        "charge": pa.array(np.asarray(rows["charge"], dtype=np.int32), pa.int32()),
        "protein_group": pa.array(list(rows["protein_group"]), pa.string()),
    }
    for name, values in (extra_q or {}).items():
        cols[name] = pa.array(np.asarray(values, dtype=np.float64), pa.float64())
    pq.write_table(pa.table(cols), str(path), compression="snappy")
    return path


def write_psms_table(path, candidate_ids, apex_rts):
    """Write a per-run psms.parquet: the two columns MBR reads (`mbr_worker.py:96`)."""
    pq.write_table(
        pa.table(
            {
                "candidate_id": pa.array(
                    np.asarray(candidate_ids, dtype=np.uint32), pa.uint32()
                ),
                "apex_rt": pa.array(
                    np.asarray(apex_rts, dtype=np.float64), pa.float64()
                ),
            }
        ),
        str(path),
        compression="snappy",
    )
    return path


# ---------------------------------------------------------------------------
# independent peptide mass arithmetic for the library-helper fixtures
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"([A-Z])(\[[^\]]*\])?")


def parse_proforma(peptidoform):
    """ProForma string -> [(residue, mod_name_or_empty), ...]."""
    text = peptidoform.replace("DECOY_", "")
    return [(res, mod[1:-1] if mod else "") for res, mod in _TOKEN.findall(text)]


def token_mass(token):
    return RESIDUE_MASS[token[0]] + (MOD_MASS[token[1]] if token[1] else 0.0)


def stripped(tokens):
    return "".join(res for res, _ in tokens)


def to_proforma(tokens):
    return "".join(res + ("[{}]".format(mod) if mod else "") for res, mod in tokens)


def reverse_keep_cterm(tokens):
    """The reversal `make_reverse_decoys.py` specifies: C-terminal residue fixed."""
    if len(tokens) < 2:
        return list(tokens)
    return tokens[:-1][::-1] + tokens[-1:]


def fragment_mz(tokens, ion_type, ordinal, charge):
    """Monoisotopic b/y fragment m/z from the independent residue table."""
    if ion_type == "b":
        neutral = sum(token_mass(t) for t in tokens[:ordinal])
    else:
        neutral = sum(token_mass(t) for t in tokens[len(tokens) - ordinal:]) + WATER
    return (neutral + charge * PROTON) / charge


def precursor_mz(tokens, charge):
    neutral = sum(token_mass(t) for t in tokens) + WATER
    return (neutral + charge * PROTON) / charge


# ---------------------------------------------------------------------------
# source-order helper (the DeepLC/torch import-order contract)
# ---------------------------------------------------------------------------


def first_line_matching(text, pattern):
    """Index of the first line matching `pattern`, or None."""
    rx = re.compile(pattern)
    for i, line in enumerate(text.splitlines()):
        if rx.match(line):
            return i
    return None
