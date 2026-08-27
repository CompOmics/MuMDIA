#!/usr/bin/env python
"""Condition-evidence filters on quantification output: which run values may be
reported.

Why this exists. In a multi-run analysis with match-between-runs, an ion can be
quantified in a run where it was never identified directly. Those values are the
source of low-abundance ratio compression: an ion transferred only into the
condition where it is absent gets the noise floor, and the measured ratio collapses
toward one. Measured on the ProteoBench Astral set, the 300 worst E. coli ions had
64% of their condition-A values from transfers versus 12% for well-measured ions,
and removing the contaminated fragments did not fix them, because the remaining
fragments were inflated too.

Withholding a value is therefore not a cosmetic choice. Rule F1 below moved the
global median absolute epsilon from 0.195 to 0.176 and the species-equalised value
from 0.283 to 0.223 while giving up 6% of features.

The four rules, in decreasing strictness of what they keep:

    F1  report an ion's values in a condition only if at least one run of that
        condition identified it directly (pass-1 confident). This is the rule the
        chosen Astral submission used.
    F2  report only directly identified values; drop every transferred value.
    F3  report an ion at all only if it has at least two direct identifications
        overall, but keep its transferred values.
    F4  like F1, but require direct evidence in EVERY condition, else drop the ion
        entirely.

"Pass-1 confident" means `label == target` and `run_psm_q <= --q` in the scored
table of a run, taken from the analysis pass that did no transfer.

Usage:
    bench/prov_filter.py --quant-dir out/lowab_q --pass1-dir out/pool_nombr \\
        --out-dir out/lowab_q --runs A_REP1,A_REP2,A_REP3,B_REP1,B_REP2,B_REP3 \\
        --conditions A,A,A,B,B,B --rules F1

Writes `<out-dir>_<rule>/pepquant_<run>.parquet` for each requested rule, which is
the layout `bench/make_proteobench_submission.py` expects.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--quant-dir", required=True,
                    help="directory of pepquant_<run>.parquet to filter")
    ap.add_argument("--pass1-dir", required=True,
                    help="directory of scored_src<i>.parquet from the no-transfer pass")
    ap.add_argument("--out-dir", required=True,
                    help="output prefix; each rule writes to <out-dir>_<rule>")
    ap.add_argument("--runs", required=True,
                    help="comma-separated run names, in the same order as scored_src<i>")
    ap.add_argument("--conditions", required=True,
                    help="comma-separated condition label per run, same length as --runs")
    ap.add_argument("--rules", default="F1",
                    help="comma-separated subset of F1,F2,F3,F4 (default F1)")
    ap.add_argument("--q", type=float, default=0.01,
                    help="run_psm_q threshold for a direct identification")
    a = ap.parse_args()

    runs = [r for r in a.runs.split(",") if r]
    conds = [c for c in a.conditions.split(",") if c]
    if len(runs) != len(conds):
        print(f"error: {len(runs)} runs but {len(conds)} conditions", file=sys.stderr)
        return 2
    rules_wanted = [r.strip().upper() for r in a.rules.split(",") if r.strip()]
    by_cond: dict[str, list[str]] = {}
    for r, c in zip(runs, conds):
        by_cond.setdefault(c, []).append(r)

    # Direct identifications per run, from the pass that did no transfer.
    direct: dict[str, set[int]] = {}
    for i, r in enumerate(runs):
        path = os.path.join(a.pass1_dir, f"scored_src{i}.parquet")
        s = pq.read_table(path, columns=["candidate_id", "label", "run_psm_q"]).to_pandas()
        direct[r] = set(s[(s.label == "target") & (s.run_psm_q <= a.q)].candidate_id)
        print(f"{r}: {len(direct[r])} direct identifications at run_psm_q <= {a.q}")

    quant = {
        r: pq.read_table(os.path.join(a.quant_dir, f"pepquant_{r}.parquet")).to_pandas()
        for r in runs
    }
    ions = sorted(set().union(*[set(q.candidate_id) for q in quant.values()]))
    # Direct identification, and presence of a quantity, per (ion, run).
    is_direct = pd.DataFrame(
        {r: pd.Series([c in direct[r] for c in ions], index=ions) for r in runs}
    )
    # Build the presence matrix as bool directly. Reindexing a Series of True onto
    # the full ion index introduces NaN for absent ions, and filling that on an
    # object-dtype frame is deprecated in pandas and would start warning, then
    # change behaviour.
    present = {r: set(quant[r].candidate_id.values) for r in runs}
    has_value = pd.DataFrame(
        {r: pd.Series([c in present[r] for c in ions], index=ions, dtype=bool) for r in runs}
    )

    # Per condition: how many of its runs both identified the ion directly and
    # quantified it. A direct identification with no quantity is not evidence that
    # the reported number is supported, so both are required.
    cond_support = {
        c: (is_direct[rs] & has_value[rs]).sum(axis=1) for c, rs in by_cond.items()
    }
    total_support = sum(cond_support.values())

    cond_of = dict(zip(runs, conds))

    def keep(rule: str, run: str, ion: int) -> bool:
        if rule == "F1":
            return bool(cond_support[cond_of[run]][ion] > 0)
        if rule == "F2":
            return bool(is_direct.at[ion, run])
        if rule == "F3":
            return bool(total_support[ion] >= 2)
        if rule == "F4":
            return all(bool(s[ion] > 0) for s in cond_support.values())
        raise SystemExit(f"unknown rule {rule!r}; expected one of F1,F2,F3,F4")

    for rule in rules_wanted:
        out = f"{a.out_dir}_{rule}"
        os.makedirs(out, exist_ok=True)
        kept = {}
        for r in runs:
            p = quant[r]
            mask = np.array([keep(rule, r, c) for c in p.candidate_id.values], dtype=bool)
            filtered = p[mask]
            pq.write_table(
                pa.Table.from_pandas(filtered, preserve_index=False),
                os.path.join(out, f"pepquant_{r}.parquet"),
            )
            kept[r] = f"{len(filtered)}/{len(p)}"
        print(f"{rule} -> {out}: {kept}")
    print("PROV_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
