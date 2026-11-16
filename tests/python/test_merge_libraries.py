"""Contract tests for `scripts/merge_libraries.py` (empirical + predicted library, #320).

DIA-NN is not available in CI, so the decoy prediction is stood in for as in
`test_predict_decoys.py`. What is checked: which library an entry takes its spectrum,
retention time and protein from, the retention-time calibration, the merged ids and
counts, a library the engine loads, and decoys trimmed to their target's fragment count.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import (
    SCRIPTS,
    assert_library_load_invariants,
    assert_string_columns_utf8,
    fragment_mz,
    parse_proforma,
    precursor_mz,
    read_columns,
    run_worker,
    run_worker_ok,
)
from test_decoy_builders import _write_target_library
from test_predict_decoys import _fake_prediction

EMPIRICAL = [("PEPTIDEK", 2), ("SAMPLER", 2), ("SC[Carbamidomethyl]ENICK", 2), ("IEVNVELR", 2),
             ("NEWPEPK", 2)]
EMP_FRAGMENTS = (("y", 2, 1), ("b", 2, 1))


def _load():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("mumdia_merge_libraries",
                                                  SCRIPTS / "merge_libraries.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_empirical(directory, predicted_rt):
    """An empirical library: four of the predicted library's entries and one of its own, two
    observed fragments each at 1.0 and 0.25 (the predicted ones are all 0.5), retention times
    in observed minutes, `rt = 0.5 * irt - 20` of the predicted entry, a decoy row that must be
    ignored, and a protein of its own."""
    rows = []
    for i, (pform, z) in enumerate(EMPIRICAL):
        tokens = parse_proforma(pform)
        rt = 0.5 * predicted_rt.get((pform, z), 300.0) - 20.0
        rows.append((pform, z, tokens, precursor_mz(tokens, z), rt))
    rows.append(("DECOY_KEDITPEP", 2, parse_proforma("KEDITPEP"),
                 precursor_mz(parse_proforma("KEDITPEP"), 2), 1.0))
    rows.sort(key=lambda r: r[3])
    prec = pa.table({
        "candidate_id": pa.array(range(len(rows)), pa.uint32()),
        "peptidoform_id": pa.array(range(len(rows)), pa.uint32()),
        "base_peptide_id": pa.array(range(len(rows)), pa.uint32()),
        "peptidoform": pa.array([r[0] for r in rows], pa.string()),
        "charge": pa.array([r[1] for r in rows], pa.int32()),
        "precursor_mz": pa.array([r[3] for r in rows], pa.float64()),
        "predicted_irt": pa.array([r[4] for r in rows], pa.float32()),
        "label": pa.array(["decoy" if r[0].startswith("DECOY_") else "target" for r in rows],
                          pa.string()),
        "protein": pa.array(["EMP_" + r[0] for r in rows], pa.string()),
        "n_fragments": pa.array([len(EMP_FRAGMENTS)] * len(rows), pa.int32()),
    })
    cols = {k: [] for k in ("candidate_id", "mz", "predicted_intensity", "name", "ion_type",
                            "ordinal", "frag_charge", "cardinality")}
    for cid, r in enumerate(rows):
        for (ion, ordinal, fz), inten in zip(EMP_FRAGMENTS, (1.0, 0.25)):
            cols["candidate_id"].append(cid)
            cols["mz"].append(fragment_mz(r[2], ion, ordinal, fz))
            cols["predicted_intensity"].append(inten)
            cols["name"].append(f"{ion}{ordinal}")
            cols["ion_type"].append(ion)
            cols["ordinal"].append(ordinal)
            cols["frag_charge"].append(fz)
            cols["cardinality"].append(9)
    frag = pa.table({
        "candidate_id": pa.array(cols["candidate_id"], pa.uint32()),
        "mz": pa.array(cols["mz"], pa.float64()),
        "predicted_intensity": pa.array(cols["predicted_intensity"], pa.float32()),
        "name": pa.array(cols["name"], pa.string()),
        "ion_type": pa.array(cols["ion_type"], pa.string()),
        "ordinal": pa.array(cols["ordinal"], pa.int32()),
        "frag_charge": pa.array(cols["frag_charge"], pa.int32()),
        "cardinality": pa.array(cols["cardinality"], pa.int32()),
    })
    p, f = directory / "emp_prec.parquet", directory / "emp_frag.parquet"
    pq.write_table(prec, str(p))
    pq.write_table(frag, str(f))
    return p, f


@pytest.fixture()
def libs(tmp_path):
    pred = _write_target_library(tmp_path)
    pc = read_columns(pred["prec"])
    rt = {(p, z): t for p, z, t in zip(pc["peptidoform"], pc["charge"], pc["predicted_irt"])}
    ep, ef = _write_empirical(tmp_path, rt)
    return {"pred": (pred["prec"], pred["frag"]), "emp": (ep, ef), "dir": tmp_path, "rt": rt}


def _merge(libs, *extra, name="m"):
    d = libs["dir"]
    out_p, out_f = d / f"{name}_prec.parquet", d / f"{name}_frag.parquet"
    summary = d / f"{name}.json"
    out, _ = run_worker_ok(
        "merge_libraries.py",
        "--empirical-precursors", libs["emp"][0], "--empirical-fragments", libs["emp"][1],
        "--predicted-precursors", libs["pred"][0], "--predicted-fragments", libs["pred"][1],
        "--out-precursors", out_p, "--out-fragments", out_f, "--decoys", "none",
        "--rt-min-shared", "3", "--summary", summary, *extra)
    return out_p, out_f, json.loads(summary.read_text()), out


def _by_key(prec_path, frag_path):
    p, f = read_columns(prec_path), read_columns(frag_path)
    frags = {}
    for cid, inten in zip(f["candidate_id"], f["predicted_intensity"]):
        frags.setdefault(cid, []).append(round(float(inten), 4))
    rows = {}
    for i, (pf, z) in enumerate(zip(p["peptidoform"], p["charge"])):
        row = {c: p[c][i] for c in p}
        row["intensities"] = sorted(frags[p["candidate_id"][i]])
        rows[(pf, z)] = row
    return rows


def test_monotone_curve_recovers_a_linear_map_and_extrapolates():
    m = _load()
    x = np.linspace(0, 100, 1000)
    y = 3.0 * x + 7.0
    kx, ky = m.monotone_curve(x, y)
    assert np.all(np.diff(ky) >= 0)
    np.testing.assert_allclose(m.apply_curve(kx, ky, [0.0, 50.0, 100.0, 150.0, -10.0]),
                               [7.0, 157.0, 307.0, 457.0, -23.0], atol=0.5)
    # A noisy, locally decreasing pair still gives a non-decreasing curve.
    rng = np.random.default_rng(0)
    kx, ky = m.monotone_curve(x, y + rng.normal(0, 40, x.size))
    assert np.all(np.diff(ky) >= 0)


def test_an_entry_in_both_takes_the_empirical_spectrum_and_the_predicted_protein(libs):
    out_p, out_f, s, _ = _merge(libs)
    p, f = assert_library_load_invariants(out_p, out_f, require_both_labels=False)
    assert_string_columns_utf8(out_p)
    rows = _by_key(out_p, out_f)
    # 8 predicted + 1 empirical-only; the empirical library's decoy is not read.
    assert len(rows) == 9 and s["shared"] == 4 and s["empirical_only_entries"] == 1
    assert s["targets_with_empirical_spectrum"] == 5
    shared = rows[("PEPTIDEK", 2)]
    assert shared["intensities"] == [0.25, 1.0]                # empirical spectrum
    assert shared["protein"].startswith("PROT") and shared["n_fragments"] == 2
    assert rows[("PEPTIDEK", 3)]["intensities"] == [1.0, 1.0, 1.0]   # predicted, max-normalised
    new = rows[("NEWPEPK", 2)]
    assert new["protein"] == "EMP_NEWPEPK" and new["intensities"] == [0.25, 1.0]
    # The calibration undoes rt = 0.5 * irt - 20 on the shared entries.
    for key in [("PEPTIDEK", 2), ("SAMPLER", 2), ("IEVNVELR", 2)]:
        assert abs(rows[key]["predicted_irt"] - libs["rt"][key]) < 1.0, key
    # One base peptide per stripped sequence across both halves.
    assert rows[("PEPTIDEK", 2)]["base_peptide_id"] == rows[("PEPTIDEK", 3)]["base_peptide_id"]
    assert len({r["base_peptide_id"] for r in rows.values()}) == 8
    # Cardinality is recomputed over the merged table (the empirical input said 9).
    assert max(read_columns(out_f)["cardinality"]) < 9


def test_predicted_precedence_and_dropping_empirical_only_entries(libs):
    out_p, out_f, s, _ = _merge(libs, "--precedence", "predicted", "--empirical-only", "drop",
                                "--rt", "keep", name="p")
    rows = _by_key(out_p, out_f)
    assert len(rows) == 8 and s["targets_with_empirical_spectrum"] == 0
    assert rows[("PEPTIDEK", 2)]["intensities"] == [1.0, 1.0, 1.0]
    out_p, out_f, s, _ = _merge(libs, "--precedence", "predicted", "--rt", "keep", name="q")
    rows = _by_key(out_p, out_f)
    # Kept, with its observed retention time unchanged under --rt keep.
    assert rows[("NEWPEPK", 2)]["predicted_irt"] == pytest.approx(0.5 * 300.0 - 20.0)


def test_calibration_needs_enough_shared_entries(libs):
    d = libs["dir"]
    rc, _, err = run_worker(
        "merge_libraries.py",
        "--empirical-precursors", libs["emp"][0], "--empirical-fragments", libs["emp"][1],
        "--predicted-precursors", libs["pred"][0], "--predicted-fragments", libs["pred"][1],
        "--out-precursors", d / "x.parquet", "--out-fragments", d / "y.parquet",
        "--decoys", "none")
    assert rc != 0 and "--rt-min-shared" in err
    rc, _, err = run_worker(
        "merge_libraries.py",
        "--empirical-precursors", libs["emp"][0], "--empirical-fragments", libs["emp"][1],
        "--predicted-precursors", libs["pred"][0], "--predicted-fragments", libs["pred"][1],
        "--out-precursors", d / "x.parquet", "--out-fragments", d / "y.parquet")
    assert rc != 0 and "--diann" in err


def test_the_merged_targets_take_predicted_decoys_trimmed_to_their_targets(libs):
    """The documented chain after the merge: reverse decoys, predicted decoy spectra (a
    stand-in for DIA-NN, four fragments each), then the trim to the target's count."""
    m = _load()
    d = libs["dir"]
    tp, tf, _, _ = _merge(libs)
    rp, rf = d / "rev_prec.parquet", d / "rev_frag.parquet"
    run_worker_ok("make_reverse_decoys.py", tp, tf, rp, rf)
    pp, pf, n_pred = _fake_prediction(rp, d)
    op, of = d / "out_prec.parquet", d / "out_frag.parquet"
    run_worker_ok("predict_decoys.py", "merge", rp, rf, pp, pf, op, of)
    removed = m.match_decoy_fragments(str(op), str(of))
    p, f = assert_library_load_invariants(op, of, require_both_labels=True)
    c = read_columns(op)
    by_pair = {}
    for lab, pfid, z, n in zip(c["label"], c["peptidoform_id"], c["charge"], c["n_fragments"]):
        by_pair.setdefault((pfid, z), {})[lab] = n
    assert all(v["target"] == v["decoy"] for v in by_pair.values()), by_pair
    assert removed == sum(n_pred - v["decoy"] for v in by_pair.values())
    counts = np.bincount(np.asarray(read_columns(of)["candidate_id"]), minlength=len(c["label"]))
    assert list(counts) == list(c["n_fragments"])
