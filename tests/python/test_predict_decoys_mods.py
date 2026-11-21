"""`scripts/predict_decoys.py` declares the library's modifications to DIA-NN.

DIA-NN knows a few modifications by itself (UniMod 4, 35 and 21 among them). For any other
one it must be told the mass in every run that reads a library carrying it. The decoys'
prediction was run without declarations, so a library built with methyl (UniMod:34) or
trimethyl (UniMod:37) variable modifications ended with "cannot generate a predicted library
without knowing the modification masses", a re-export of 0 precursors and an importer error
far from the cause. DIA-NN is not available in CI, so a stand-in executable records its
command lines and plays the two failure modes; the real-DIA-NN check is in the PR.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import SCRIPTS, run_worker


def _module():
    spec = importlib.util.spec_from_file_location("predict_decoys", SCRIPTS / "predict_decoys.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pd_mod = _module()

DECOYS = [
    "DECOY_KEDITH[Methyl]PEP",
    "DECOY_RELPK[Trimethyl]MAS",
    "DECOY_[Acetyl]-KEDITPEP",
    "DECOY_RELPKM[Oxidation]AS",
    "DECOY_KC[Carbamidomethyl]INEC[Carbamidomethyl]S",
]


def test_declarations_are_derived_from_the_decoys():
    assert pd_mod.diann_mods(DECOYS) == [
        "UniMod:1,42.010565",
        "UniMod:4,57.021464",
        "UniMod:34,14.015650",
        "UniMod:35,15.994915",
        "UniMod:37,42.046950",
    ]
    assert pd_mod.diann_mods(["DECOY_KEDITPEP"]) == []


def test_an_explicit_declaration_comes_first_and_wins_for_its_accession():
    got = pd_mod.diann_mods(DECOYS, ["UniMod:34,14.0157", "UniMod:999,1.5"])
    assert got[:2] == ["UniMod:34,14.0157", "UniMod:999,1.5"]
    assert sum(d.startswith("UniMod:34,") for d in got) == 1
    assert "UniMod:37,42.046950" in got


@pytest.mark.parametrize("bad", ["Methyl,14.01565", "UniMod:34", "UniMod:34,abc", "UniMod:34,-1"])
def test_a_malformed_declaration_is_refused(bad):
    with pytest.raises(SystemExit, match="expected UniMod:<id>,<mass>"):
        pd_mod.diann_mods(DECOYS, [bad])


def test_mod_args_declare_and_never_add_a_modification():
    args = pd_mod.mod_args(["UniMod:34,14.015650", "UniMod:37,42.046950"])
    assert args == ["--mod", "UniMod:34,14.015650", "--mod", "UniMod:37,42.046950"]


# DIA-NN 2.7's own lines, as it printed them on the reported library.
UNKNOWN = ("ERROR: below are the unknown modifications found in the library. While unlikely to "
           "have a signficant effect, these might negatively affect identification performance "
           "in certain scenarious and therefore must be declared using any of --mod, --fixed-mod "
           "or --var-mod: UniMod:34, UniMod:37")
NO_MASSES = ("ERROR: DIA-NN cannot generate a predicted library without knowing the modification "
             "masses, deep learning disabled")


def test_dia_nns_refusal_is_reported_with_the_accessions_it_named():
    out = f"{UNKNOWN}\n{NO_MASSES}\n{UNKNOWN} : 1\n"
    assert pd_mod.undeclared(out) == ["UniMod:34", "UniMod:37"]
    with pytest.raises(SystemExit, match=r"not given the mass of: UniMod:34, UniMod:37"):
        pd_mod.check_diann_output(out, [])
    pd_mod.check_diann_output(UNKNOWN, [])  # a warning alone is not a refusal


# ---------------------------------------------------------------------------
# the whole `build`, with a stand-in DIA-NN
# ---------------------------------------------------------------------------

FAKE = r'''
import json, os, sys
import pyarrow as pa, pyarrow.parquet as pq
argv = sys.argv[1:]
with open(os.environ["FAKE_DIANN_LOG"], "a") as f:
    f.write(json.dumps(argv) + "\n")
out = argv[argv.index("--out-lib") + 1]
if "--predictor" in argv:
    if os.environ["FAKE_DIANN_MODE"] == "nomass":
        print("ERROR: below are the unknown modifications found in the library. While unlikely "
              "to have a signficant effect, these might negatively affect identification "
              "performance in certain scenarious and therefore must be declared using any of "
              "--mod, --fixed-mod or --var-mod: UniMod:34, UniMod:37")
        print("ERROR: DIA-NN cannot generate a predicted library without knowing the "
              "modification masses, deep learning disabled")
    open(out + ".predicted.speclib", "wb").close()
else:
    print("0 precursors saved")
    pq.write_table(pa.table({"Precursor.Id": pa.array([], pa.string())}), out)
'''


@pytest.fixture()
def fake_diann(tmp_path):
    script = tmp_path / "fake_diann.py"
    script.write_text(FAKE)
    if os.name == "nt":
        exe = tmp_path / "diann.cmd"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        exe = tmp_path / "diann"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        exe.chmod(0o755)
    return exe, tmp_path / "calls.jsonl"


def _reverse_library(d):
    """A make_reverse_decoys.py-shaped library: targets and their reversed decoys, paired on
    (peptidoform_id, charge), carrying methyl, trimethyl and a protein N-terminal acetyl."""
    targets = ["PEPH[Methyl]TIDEK", "SAMK[Trimethyl]PLER", "[Acetyl]-PEPTIDEK"]
    decoys = DECOYS[:3]
    n = len(targets)
    prec = pa.table({
        "candidate_id": pa.array(range(2 * n), pa.uint32()),
        "peptidoform_id": pa.array(list(range(n)) * 2, pa.uint32()),
        "base_peptide_id": pa.array(list(range(n)) * 2, pa.uint32()),
        "peptidoform": pa.array(targets + decoys, pa.string()),
        "charge": pa.array([2] * (2 * n), pa.int32()),
        "precursor_mz": pa.array(np.linspace(400.0, 500.0, 2 * n), pa.float64()),
        "predicted_irt": pa.array([10.0] * (2 * n), pa.float32()),
        "label": pa.array(["target"] * n + ["decoy"] * n, pa.string()),
        "protein": pa.array(["P1"] * (2 * n), pa.string()),
    })
    frag = pa.table({
        "candidate_id": pa.array(range(2 * n), pa.uint32()),
        "mz": pa.array([300.0] * (2 * n), pa.float64()),
    })
    pq.write_table(prec, d / "rev_prec.parquet")
    pq.write_table(frag, d / "rev_frag.parquet")
    return d / "rev_prec.parquet", d / "rev_frag.parquet"


def _build(tmp_path, fake_diann, mode, *extra):
    exe, log = fake_diann
    prec, frag = _reverse_library(tmp_path)
    rc, out, err = run_worker(
        "predict_decoys.py", "build", prec, frag, tmp_path / "op.parquet", tmp_path / "of.parquet",
        "--diann", exe, "--threads", 2, "--work", tmp_path / "work", *extra,
        env={"FAKE_DIANN_LOG": log, "FAKE_DIANN_MODE": mode})
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return rc, out, err, calls


def _declared(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--mod"]


def test_both_dia_nn_runs_declare_the_modifications_and_an_empty_prediction_stops(
        tmp_path, fake_diann):
    rc, out, err, calls = _build(tmp_path, fake_diann, "empty", "--diann-mod=UniMod:999,1.5")
    want = ["UniMod:999,1.5", "UniMod:1,42.010565", "UniMod:34,14.015650", "UniMod:37,42.046950"]
    assert len(calls) == 2, (out, err)
    predict, reexport = calls
    assert "--predictor" in predict and "--predictor" not in reexport
    assert _declared(predict) == want
    assert _declared(reexport) == want
    assert "--var-mod" not in predict + reexport and "--fixed-mod" not in predict + reexport
    # 0 rows stop here, naming the cause, instead of in the importer.
    assert rc != 0
    assert "DIA-NN predicted no decoy spectra" in err
    assert "pred_precursors" not in out


def test_dia_nns_refusal_to_predict_stops_the_build_with_the_reason(tmp_path, fake_diann):
    rc, out, err, calls = _build(tmp_path, fake_diann, "nomass")
    assert rc != 0
    assert len(calls) == 1, "nothing after the refused prediction runs"
    assert "not given the mass of: UniMod:34, UniMod:37" in err
    assert "--diann-mod UniMod:<id>,<mass>" in err


def test_a_retry_does_not_reuse_a_failed_attempts_decoy_library(tmp_path, fake_diann):
    # A 0-row decoys.parquet left by an earlier attempt must not stand in for this run's
    # prediction: the re-export still runs, so the stale file was removed first.
    work = tmp_path / "work"
    work.mkdir()
    pq.write_table(pa.table({"Precursor.Id": pa.array(["stale"], pa.string())}),
                   work / "decoys.parquet")
    rc, out, err, calls = _build(tmp_path, fake_diann, "empty")
    assert len(calls) == 2, (out, err)
    assert "DIA-NN predicted no decoy spectra" in err
