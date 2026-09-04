#!/usr/bin/env python3
"""Score one arm of the apex_evidence_rank quantification A/B on the HYE 3+3 set.

The comparison the A/B exists for: identification is flat under the promoted
`extract.apex_evidence_rank` default while the selected apex MOVES for about half of
identified peptides. Quantification integrates around that apex, so this measures
what the move does to known ratios.

Metrics, all computed the same way for both arms so the arms are comparable to each
other. They are NOT ProteoBench numbers: the module's own parser drops contaminants
by the `Cont_` accession flag, which MuMDIA's protein strings do not carry (they are
entry names), and it works on precursor ions from a Custom TSV. What is below is the
same quantity computed consistently, which is what an A/B needs.

- feature = (peptidoform, charge), the precursor-shaped row peptide_quant carries;
- intensity per run from `quantity`, `quant_status == "quantified"` only;
- species from the `_HUMAN` / `_YEAST` / `_ECOLI` suffixes in `protein_group`; a
  feature whose protein group spans more than one species is dropped, since its
  expected ratio is undefined;
- log2 ratio = median(log2 A intensities) - median(log2 B intensities);
- epsilon = observed log2 ratio - expected (HUMAN 0, YEAST +1, ECOLI -2), per the
  archived AIF module's `module_settings.toml` (A/B 1.0, 2.0, 0.25);
- CV = within-condition coefficient of variation on the raw intensities, taken per
  condition and averaged over the two, then reported as a median over features;
- completeness = features quantified in >= 3 runs and in all 6.

Usage:  python qab_score.py <arm_dir>            # e.g. qab/on
        python qab_score.py <arm_a> <arm_b>      # prints both plus the delta
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

import pyarrow.parquet as pq

RUNS_A = ["A01", "A02", "A03"]
RUNS_B = ["B01", "B02", "B03"]
EXPECTED = {"HUMAN": 0.0, "YEAST": 1.0, "ECOLI": -2.0}
SUFFIXES = {"_HUMAN": "HUMAN", "_YEAST": "YEAST", "_ECOLI": "ECOLI"}


def species_of(protein_group: str) -> str | None:
    """One species, or None when the group spans several or names none.

    A feature shared between species has no defined expected ratio, so including it
    would silently mix populations. The count of dropped features is reported.
    """
    found = set()
    for entry in protein_group.split(";"):
        for suffix, name in SUFFIXES.items():
            if entry.endswith(suffix):
                found.add(name)
    return found.pop() if len(found) == 1 else None


def load_arm(arm: Path) -> dict[tuple[str, int], dict[str, float]]:
    """(peptidoform, charge) -> {run: quantity} over the quantified rows."""
    table: dict[tuple[str, int], dict[str, float]] = {}
    proteins: dict[tuple[str, int], str] = {}
    for run in RUNS_A + RUNS_B:
        path = arm / run / "peptide_quant.parquet"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        t = pq.read_table(
            path,
            columns=["peptidoform", "charge", "protein_group", "quantity", "quant_status"],
        ).to_pydict()
        for pform, z, prot, q, status in zip(
            t["peptidoform"], t["charge"], t["protein_group"],
            t["quantity"], t["quant_status"],
        ):
            if status != "quantified" or q is None or not (q > 0) or not math.isfinite(q):
                continue
            key = (pform, z)
            table.setdefault(key, {})[run] = q
            proteins.setdefault(key, prot)
    for key, runs in table.items():
        runs["__protein__"] = proteins[key]  # type: ignore[assignment]
    return table


def cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = st.mean(values)
    return st.stdev(values) / m if m > 0 else None


def score(arm: Path) -> dict:
    table = load_arm(arm)
    per_species: dict[str, list[float]] = {k: [] for k in EXPECTED}
    eps_all: list[float] = []
    cvs: list[float] = []
    n_min3 = n_min6 = 0
    mixed = 0

    for key, runs in table.items():
        prot = runs["__protein__"]  # type: ignore[index]
        a = [runs[r] for r in RUNS_A if r in runs]
        b = [runs[r] for r in RUNS_B if r in runs]
        n_obs = len(a) + len(b)
        if n_obs >= 3:
            n_min3 += 1
        if n_obs == 6:
            n_min6 += 1

        for group in (a, b):
            c = cv(group)
            if c is not None:
                cvs.append(c)

        # A ratio needs both conditions.
        if not a or not b:
            continue
        sp = species_of(prot)
        if sp is None:
            mixed += 1
            continue
        ratio = st.median([math.log2(x) for x in a]) - st.median([math.log2(x) for x in b])
        e = ratio - EXPECTED[sp]
        per_species[sp].append(ratio)
        eps_all.append(abs(e))

    out = {
        "arm": arm.name,
        "features_quantified": len(table),
        "features_min3": n_min3,
        "features_min6": n_min6,
        "features_with_both_conditions": len(eps_all),
        "features_dropped_multispecies": mixed,
        "median_abs_epsilon_global": st.median(eps_all) if eps_all else None,
        "median_cv": st.median(cvs) if cvs else None,
        "species": {},
    }
    weighted = []
    for sp, ratios in per_species.items():
        if not ratios:
            continue
        med = st.median(ratios)
        abs_eps = st.median([abs(r - EXPECTED[sp]) for r in ratios])
        out["species"][sp] = {
            "n": len(ratios),
            "median_log2_ratio": med,
            "expected_log2": EXPECTED[sp],
            "median_abs_epsilon": abs_eps,
        }
        weighted.append(abs_eps)
    # Species-equal epsilon: the mean of the three per-species medians, so E. coli,
    # which is the smallest and worst population, is not drowned by human.
    out["median_abs_epsilon_species_equal"] = st.mean(weighted) if weighted else None
    return out


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    results = [score(Path(a)) for a in args]
    for r in results:
        print(json.dumps(r, indent=2))
    if len(results) == 2:
        x, y = results
        print("\ndelta (%s minus %s):" % (y["arm"], x["arm"]))
        for k in ("features_quantified", "features_min3", "features_min6",
                  "median_abs_epsilon_global", "median_abs_epsilon_species_equal",
                  "median_cv"):
            a, b = x.get(k), y.get(k)
            if a is None or b is None:
                continue
            print(f"  {k}: {a} -> {b}  ({b - a:+.4f})" if isinstance(a, float)
                  else f"  {k}: {a} -> {b}  ({b - a:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
