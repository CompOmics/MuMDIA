"""bench/quant_condition_gate.py: the condition-level first-pass evidence gate.

A (precursor, condition) group keeps its values when at least one of its rows carries
`first_pass_id`; the flag comes from the engine column (peptide_quant v3) or is derived
from the first pass's scored table, and the two must agree on the same experiment.
"""
from __future__ import annotations

import importlib.util
import json
import math

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("quant_condition_gate",
                                              ROOT / "bench" / "quant_condition_gate.py")
qcg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qcg)


def table(pforms, charges, quantities, flags):
    return pd.DataFrame({"peptidoform": pforms, "charge": charges,
                         "quantity": quantities, "quant_status": ["quantified"] * len(pforms),
                         "first_pass_id": flags})


def test_gate_keeps_a_condition_with_one_first_pass_value_and_removes_the_rest():
    # P1 in A: flagged in a1 only, so both A values stay. P1 in B: no flag, both go.
    # P2: flagged only on a row without a quantity, which is no evidence by default.
    a1 = table(["P1", "P2"], [2, 2], [10.0, None], [True, True])
    a2 = table(["P1", "P2"], [2, 2], [12.0, 5.0], [False, False])
    b1 = table(["P1"], [2], [3.0], [False])
    b2 = table(["P1", "P1"], [2, 3], [4.0, 7.0], [False, True])
    out = qcg.gate([a1, a2, b1, b2], ["A", "A", "B", "B"])
    assert out[0].quantity.tolist()[0] == 10.0
    assert out[1].quantity.tolist()[0] == 12.0
    assert math.isnan(out[1].quantity.tolist()[1])
    assert out[1].quant_status.tolist()[1] == qcg.GATED_STATUS
    assert math.isnan(out[2].quantity.tolist()[0])
    # Charge is part of the key: P1/3 in B is flagged and kept, P1/2 in B is gated.
    assert math.isnan(out[3].quantity.tolist()[0])
    assert out[3].quantity.tolist()[1] == 7.0
    assert out[3].condition_first_pass.tolist() == [False, True]
    # The input tables are not modified.
    assert a2.quantity.tolist() == [12.0, 5.0]
    # With every flagged row as evidence, P2 in A is kept.
    anyrow = qcg.gate([a1, a2, b1, b2], ["A", "A", "B", "B"], evidence="any")
    assert anyrow[1].quantity.tolist()[1] == 5.0


def test_gate_refuses_mismatched_inputs():
    t = table(["P1"], [2], [1.0], [True])
    with pytest.raises(ValueError):
        qcg.gate([t], ["A", "B"])
    with pytest.raises(ValueError):
        qcg.gate([t], ["A"], evidence="some")


def _experiment(tmp_path, with_flag: bool):
    exp = tmp_path / "exp"
    runs = ["r0", "r1", "r2", "r3"]
    (exp / "first_pass").mkdir(parents=True)
    (exp / "experiment_manifest.json").write_text(json.dumps({"experiment": {"runs": runs}}))
    # First pass: P1/2 accepted in r0 and r3, P2/2 in r1 at q 0.02 only, a decoy in r2.
    fp = pa.table({
        "peptidoform": ["P1", "P1", "P2", "DECOY_P1"],
        "charge": pa.array([2, 2, 2, 2], pa.int32()),
        "label": ["target", "target", "target", "decoy"],
        "source": pa.array([0, 3, 1, 2], pa.uint32()),
        "run_psm_q": [0.001, 0.005, 0.02, 0.0],
    })
    pq.write_table(fp, exp / "first_pass" / "scored_combined.parquet")
    flags = {"r0": [True, False], "r1": [False, False], "r2": [False, False], "r3": [True, False]}
    for r in runs:
        (exp / r).mkdir()
        cols = {"candidate_id": pa.array([0, 1], pa.uint32()),
                "peptidoform": ["P1", "P2"], "charge": pa.array([2, 2], pa.int32()),
                "quantity": [100.0, 50.0], "quant_status": ["quantified", "quantified"]}
        if with_flag:
            cols["first_pass_id"] = flags[r]
        pq.write_table(pa.table(cols), exp / r / "peptide_quant.parquet")
    cmap = tmp_path / "conditions.tsv"
    cmap.write_text("run\tcondition\nr0\tA\nr1\tA\nr2\tB\nr3\tB\n")
    return exp, cmap


def test_main_engine_flag_and_derived_flag_agree(tmp_path):
    exp, cmap = _experiment(tmp_path, with_flag=True)
    out1 = tmp_path / "out_engine"
    assert qcg.main(["--experiment-dir", str(exp), "--conditions", str(cmap),
                     "--out-dir", str(out1), "--arm-layout"]) == 0
    out2 = tmp_path / "out_derived"
    assert qcg.main(["--experiment-dir", str(exp), "--conditions", str(cmap),
                     "--out-dir", str(out2), "--derive-first-pass"]) == 0
    for r in ["r0", "r1", "r2", "r3"]:
        g1 = pq.read_table(out1 / r / "peptide_quant.parquet").to_pandas()
        g2 = pq.read_table(out2 / r / "peptide_quant.parquet").to_pandas()
        assert g1.quantity.isna().tolist() == g2.quantity.isna().tolist()
        # P1 has first-pass evidence in both conditions; P2 in neither (q 0.02 > 0.01).
        assert g1.quantity.isna().tolist() == [False, True]
        t = pq.read_table(out1 / r / "peptide_quant.parquet")
        assert t.schema.field("peptidoform").type == pa.string()
        assert t.schema.field("quantity").type == pa.float64()
    assert (out1 / "pq_3.parquet").is_file()
    s = json.loads((out2 / "condition_gate_summary.json").read_text())
    assert all(r["engine_vs_derived_disagree"] == 0 for r in s["runs"])
    assert [r["gated_quantified_rows"] for r in s["runs"]] == [1, 1, 1, 1]


def test_main_without_the_column_needs_derive(tmp_path):
    exp, cmap = _experiment(tmp_path, with_flag=False)
    with pytest.raises(SystemExit):
        qcg.main(["--experiment-dir", str(exp), "--conditions", str(cmap),
                  "--out-dir", str(tmp_path / "o")])
    assert qcg.main(["--experiment-dir", str(exp), "--conditions", str(cmap),
                     "--out-dir", str(tmp_path / "o2"), "--derive-first-pass",
                     "--q", "0.05"]) == 0
    # At q 0.05 the r1 identification of P2 counts, so P2 is kept in A and gated in B.
    g1 = pq.read_table(tmp_path / "o2" / "r1" / "peptide_quant.parquet").to_pandas()
    g2 = pq.read_table(tmp_path / "o2" / "r2" / "peptide_quant.parquet").to_pandas()
    assert g1.quantity.isna().tolist() == [False, False]
    assert g2.quantity.isna().tolist() == [False, True]


def test_condition_map_must_match_the_runs(tmp_path):
    exp, _ = _experiment(tmp_path, with_flag=True)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"r0": "A", "r1": "A", "r2": "B"}))
    with pytest.raises(SystemExit):
        qcg.main(["--experiment-dir", str(exp), "--conditions", str(bad),
                  "--out-dir", str(tmp_path / "o")])
