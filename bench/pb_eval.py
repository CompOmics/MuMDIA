#!/usr/bin/env python
"""Offline ProteoBench evaluation (quant LFQ ion, DIA AIF HYE) of a results file, using the
proteobench package's own parser and scoring code. No upload, no GitHub token.

  pb_eval.py custom_input.tsv --format Custom --module quant_lfq_DIA_ion_Astral \
      --out-prefix results/mumdia_astral

Needs the `proteobench` package installed; nothing is uploaded and no GitHub token
is used. The module id fixes the run names and the expected species ratios, so it
must match the data.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from proteobench.datapoint.quant_datapoint import QuantDatapointHYE
from proteobench.io.parsing.parse_ion import load_input_file
from proteobench.io.parsing.parse_settings import ParseSettingsBuilder
from proteobench.modules.constants import MODULE_SETTINGS_DIRS
from proteobench.score.quantscoresHYE import QuantScoresHYE



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file")
    ap.add_argument("--format", default="Custom")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--user-input", default=None, help="JSON file with the ProteoBench parameter fields")
    ap.add_argument(
        "--module",
        default="quant_lfq_DIA_ion_AIF",
        help="ProteoBench module id. As of proteobench 0.18.4 the DIA ion modules are "
             "quant_lfq_DIA_ion_AIF (archived: Met-oxidised peptides confound its ratios), "
             "_Astral, _diaPASEF, _ZenoTOF, _lowinput and _plasma. The module fixes the run "
             "names and the expected species ratios, so passing the wrong one silently "
             "scores against the wrong truth; an unknown id is rejected with the list.",
    )
    a = ap.parse_args()

    user_input = {
        "software_version": "0.1.0", "search_engine": "MuMDIA", "search_engine_version": "0.1.0",
        "ident_fdr_psm": 0.01, "ident_fdr_peptide": 0.01, "ident_fdr_protein": 0.01,
        "enable_match_between_runs": True, "precursor_mass_tolerance": "20 ppm",
        "fragment_mass_tolerance": "20 ppm", "enzyme": "Trypsin", "allowed_miscleavages": 1,
        "min_peptide_length": 7, "max_peptide_length": 30, "comments_for_plotting": "",
    }
    if a.user_input:
        user_input.update(json.load(open(a.user_input)))

    module_id = a.module
    if module_id not in MODULE_SETTINGS_DIRS:
        raise SystemExit(
            f"unknown module {module_id!r}; available: {sorted(MODULE_SETTINGS_DIRS)}"
        )
    print(f"module: {module_id}")
    df = load_input_file(a.input_file, a.format)
    ps = ParseSettingsBuilder(
        parse_settings_dir=MODULE_SETTINGS_DIRS[module_id], module_id=module_id
    ).build_parser(a.format)
    std, rep2raw = ps.convert_to_standard_format(df)
    qs = QuantScoresHYE("precursor ion", ps.species_expected_ratio(), ps.species_dict())
    inter = qs.generate_intermediate(std, rep2raw)
    dp = QuantDatapointHYE.generate_datapoint(inter, a.format, user_input, default_cutoff_min_feature=3)

    print(f"input rows: {len(df)}; standard-format rows: {len(std)}; precursor ions scored: {len(inter)}")
    print("species (unique-species ions):", inter["species"].value_counts().to_dict())
    med = inter.groupby("species")["log2_A_vs_B"].agg(["median", "mean", "count"]).round(3)
    med["expected"] = [round(float(np.log2(ps.species_expected_ratio()[s]["A_vs_B"])), 3) for s in med.index]
    print("log2(A/B) per species:\n", med.to_string())
    print("\nResults by min-observed cutoff (ProteoBench headline is cutoff 3):")
    rows = []
    for k, r in sorted(dp.results.items()):
        rows.append({"min_obs": k, **{kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}})
    res = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(res.to_string(index=False))
    if a.out_prefix:
        inter.to_csv(f"{a.out_prefix}_intermediate.tsv", sep="\t", index=False)
        res.to_csv(f"{a.out_prefix}_results.tsv", sep="\t", index=False)
        print(f"wrote {a.out_prefix}_intermediate.tsv and {a.out_prefix}_results.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
