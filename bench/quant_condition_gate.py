#!/usr/bin/env python
"""Condition-level first-pass evidence gate for the per-run quantities of a
`run-experiment` with a second pass (docs/12, "Second pass").

A (precursor, condition) group keeps its values only when at least one run of the
condition has a first-pass identification of the precursor. Every other value of the
group is a second-pass identification at the detection limit, and those groups carry
the most compressed ratios. It is rule F1 of `prov_filter.py`, on the directory layout of
`run-experiment` and on the engine's own flag. The gate changes no value: it removes values. A score
obtained with it is a different operating point (fewer quantified ions) and must be
compared with other results at a matched number of finite ratios.

The flag per value comes from the engine (`peptide_quant` v3 column `first_pass_id`,
written under `mbr.first_pass_flag`), or, with `--derive-first-pass`, from the
experiment's `first_pass/scored_combined.parquet`: a target row of the same run
(`source`) and the same `(peptidoform, charge)` at `run_psm_q <= --q`. The derived
flag joins on `(peptidoform, charge)`, the engine flag on the empirical library's
candidate ids; the two agree when no library has two candidates with the same
`(peptidoform, charge)`.

Usage:
  quant_condition_gate.py --experiment-dir EXP --conditions MAP --out-dir OUT
      [--quant-dir DIR [--quant-layout experiment|arm]] [--derive-first-pass]
      [--q 0.01] [--evidence quantified|any] [--arm-layout]

  MAP     a TSV with columns `run` and `condition` (one row per run, every run of the
          experiment), or a JSON object {run: condition}.
  EXP     the run-experiment output directory; run order (= `source`) is read from
          `experiment_manifest.json`.
  --quant-dir   read the per-run tables from another directory with the same runs, for
                example a quant arm on fixed identifications: `<DIR>/<run>/peptide_quant.parquet`
                (layout `experiment`, the default) or `<DIR>/pq_<i>.parquet` (layout `arm`).
  --evidence    `quantified` (default): only rows with a positive finite quantity count as
                evidence, as in the investigation's S7 gate; `any`: every flagged row counts.

Writes `OUT/<run>/peptide_quant.parquet` (the input table, the gated values set to null,
their `quant_status` set to `no_first_pass_in_condition`, and two Boolean columns
`first_pass_id` and `condition_first_pass`), with `--arm-layout` also `OUT/pq_<i>.parquet`,
and `OUT/condition_gate_summary.json`. The gated tables can be passed to
`make_proteobench_submission.py` unchanged; a null quantity is a missing value there.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FLAG = "first_pass_id"
GATED_STATUS = "no_first_pass_in_condition"


def read_conditions(path: str) -> dict[str, str]:
    """{run: condition} from a JSON object or a TSV with `run` and `condition`."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if not isinstance(m, dict):
            raise SystemExit(f"{path}: expected a JSON object {{run: condition}}")
        return {str(k): str(v) for k, v in m.items()}
    t = pd.read_csv(path, sep="\t", dtype=str)
    missing = {"run", "condition"} - set(t.columns)
    if missing:
        raise SystemExit(f"{path}: missing column(s) {sorted(missing)}")
    if t.run.duplicated().any():
        raise SystemExit(f"{path}: run listed twice: {sorted(t.run[t.run.duplicated()])}")
    return dict(zip(t.run, t.condition))


def experiment_runs(exp_dir: str) -> list[str]:
    """Run names in `source` order, from the experiment manifest."""
    with open(os.path.join(exp_dir, "experiment_manifest.json"), encoding="utf-8") as f:
        m = json.load(f)
    runs = m.get("experiment", {}).get("runs")
    if not runs:
        raise SystemExit(f"{exp_dir}: experiment_manifest.json lists no runs")
    return [str(r) for r in runs]


def derived_first_pass(scored: str, q: float) -> set[tuple[int, str, int]]:
    """(source, peptidoform, charge) of every first-pass target at run_psm_q <= q."""
    t = pq.read_table(scored, columns=["peptidoform", "charge", "label", "source", "run_psm_q"])
    d = t.to_pandas()
    d = d[(d.label.astype(str) == "target") & (d.run_psm_q <= q)]
    return set(zip(d.source.astype(int), d.peptidoform.astype(str), d.charge.astype(int)))


def quantified(q: pd.Series) -> pd.Series:
    """True where a quantity is a positive finite number."""
    v = pd.to_numeric(q, errors="coerce").astype("float64")
    return pd.Series(np.isfinite(v.to_numpy()) & (v.to_numpy() > 0), index=q.index)


def gate(tables: list[pd.DataFrame], conditions: list[str], evidence: str = "quantified"
         ) -> list[pd.DataFrame]:
    """Apply the gate to per-run tables (each with `peptidoform`, `charge`, `quantity`
    and a Boolean `first_pass_id`), in run order, with `conditions[i]` the condition of
    run i. Returns copies with `condition_first_pass` added and the gated quantities set
    to null (status `no_first_pass_in_condition`)."""
    if len(tables) != len(conditions):
        raise ValueError(f"{len(tables)} tables but {len(conditions)} conditions")
    if evidence not in ("quantified", "any"):
        raise ValueError(f"evidence must be 'quantified' or 'any', got {evidence!r}")
    evid: set[tuple[str, str, int]] = set()
    for t, c in zip(tables, conditions):
        ok = t[FLAG].astype(bool)
        if evidence == "quantified":
            ok &= quantified(t["quantity"])
        evid.update((c, p, int(z)) for p, z in zip(t.peptidoform[ok], t.charge[ok]))
    out = []
    for t, c in zip(tables, conditions):
        t = t.copy()
        keep = [(c, p, int(z)) in evid for p, z in zip(t.peptidoform, t.charge)]
        t["condition_first_pass"] = pd.Series(keep, index=t.index, dtype=bool)
        drop = ~t["condition_first_pass"]
        t["quantity"] = t["quantity"].astype("float64")
        t.loc[drop, "quantity"] = float("nan")
        if "quant_status" in t.columns:
            t.loc[drop, "quant_status"] = GATED_STATUS
        out.append(t)
    return out


def write_table(df: pd.DataFrame, path: str) -> None:
    """Snappy parquet, `utf8` strings, nullable float quantities (NaN -> null)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields, arrays = [], []
    for c in df.columns:
        s = df[c]
        if c == "quantity":
            arr = pa.array(s.astype("float64"), type=pa.float64(), from_pandas=True)
        elif s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            arr = pa.array(s.astype(str).tolist(), type=pa.string())
        else:
            arr = pa.array(s, from_pandas=True)
        fields.append(pa.field(c, arr.type))
        arrays.append(arr)
    pq.write_table(pa.Table.from_arrays(arrays, schema=pa.schema(fields)), path,
                   compression="snappy")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--quant-layout", choices=["experiment", "arm"], default="experiment")
    ap.add_argument("--derive-first-pass", action="store_true")
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--evidence", choices=["quantified", "any"], default="quantified")
    ap.add_argument("--arm-layout", action="store_true")
    a = ap.parse_args(argv)

    runs = experiment_runs(a.experiment_dir)
    cmap = read_conditions(a.conditions)
    unknown = sorted(set(cmap) - set(runs))
    missing = [r for r in runs if r not in cmap]
    if unknown or missing:
        raise SystemExit(f"condition map does not match the experiment's runs: "
                         f"not in the experiment {unknown}, without a condition {missing}")
    qdir = a.quant_dir or a.experiment_dir
    paths = [os.path.join(qdir, f"pq_{i}.parquet") if a.quant_layout == "arm"
             else os.path.join(qdir, r, "peptide_quant.parquet") for i, r in enumerate(runs)]
    tables = [pq.read_table(p).to_pandas() for p in paths]
    has_flag = [FLAG in t.columns for t in tables]
    if a.derive_first_pass:
        fp = derived_first_pass(os.path.join(a.experiment_dir, "first_pass",
                                             "scored_combined.parquet"), a.q)
        flag_source = f"derived from first_pass/scored_combined.parquet at run_psm_q <= {a.q}"
        for i, t in enumerate(tables):
            derived = [(i, p, int(z)) in fp for p, z in zip(t.peptidoform, t.charge)]
            if has_flag[i]:
                t["first_pass_id_engine"] = t[FLAG].astype(bool)
            t[FLAG] = pd.Series(derived, index=t.index, dtype=bool)
    else:
        if not all(has_flag):
            raise SystemExit(
                f"{[p for p, h in zip(paths, has_flag) if not h]}: no `{FLAG}` column; run "
                f"run-experiment with mbr.first_pass_flag = true, or pass --derive-first-pass")
        flag_source = "engine column first_pass_id (peptide_quant v3)"
    conds = [cmap[r] for r in runs]
    gated = gate(tables, conds, a.evidence)

    summary = {"flag_source": flag_source, "evidence": a.evidence, "runs": [],
               "conditions": sorted(set(conds))}
    for i, (r, c, t0, t) in enumerate(zip(runs, conds, tables, gated)):
        qd = quantified(t0["quantity"])
        row = {"run": r, "source": i, "condition": c, "rows": int(len(t)),
               "quantified_rows": int(qd.sum()),
               "first_pass_rows": int(t[FLAG].sum()),
               "kept_quantified_rows": int((qd & t.condition_first_pass).sum()),
               "gated_quantified_rows": int((qd & ~t.condition_first_pass).sum())}
        if "first_pass_id_engine" in t.columns:
            row["engine_vs_derived_disagree"] = int((t.first_pass_id_engine != t[FLAG]).sum())
        summary["runs"].append(row)
        write_table(t, os.path.join(a.out_dir, r, "peptide_quant.parquet"))
        if a.arm_layout:
            write_table(t, os.path.join(a.out_dir, f"pq_{i}.parquet"))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "condition_gate_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for row in summary["runs"]:
        print("\t".join(str(row[k]) for k in ("run", "condition", "quantified_rows",
                                               "first_pass_rows", "kept_quantified_rows",
                                               "gated_quantified_rows")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
