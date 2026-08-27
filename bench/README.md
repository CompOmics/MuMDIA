# Benchmarks

Scripts and recorded results for the quantitative benchmarks MuMDIA is judged on,
so a number in the documentation can be traced to the commands that produced it.

These are not run by CI. They need real data, a DIA-NN licence for the imported
library, and hours of compute. The end-to-end test CI does run is
`ci/smoke.sh`, which checks that the pipeline works, not how well it performs.

## What is here

| file | purpose |
|---|---|
| `make_proteobench_submission.py` | Build a ProteoBench "Custom" input TSV from per-run `pepquant_<run>.parquet`. Expands MuMDIA's protein entry names back to full FASTA identifiers so ProteoBench's own `Cont_` contaminant flag and `_HUMAN`/`_YEAST`/`_ECOLI` species flags apply unchanged. |
| `pb_eval.py` | Score a submission offline with ProteoBench's own parser and scoring code. Nothing is uploaded and no token is used. `--module` selects the module, which fixes the run names and the expected ratios; an unknown id is rejected with the available list. |
| `prov_filter.py` | Condition-evidence filters (F1 to F4) deciding which run values may be reported. Withholding transferred values in a condition with no direct identification is what removes low-abundance ratio compression; see the docstring for the measurements. |

Requires `proteobench` (0.18.4 was used), `pyarrow`, `pandas`, `numpy`. That
version provides the DIA ion modules `quant_lfq_DIA_ion_AIF`, `_Astral`,
`_diaPASEF`, `_ZenoTOF`, `_lowinput` and `_plasma`.

`prov_filter.py` was checked against the pipeline it was extracted from: on the
six AIF runs it reproduces the recorded F1 counts exactly (69,190 / 71,432 /
66,213 / 60,891 / 63,053 / 60,296 kept).

## Reading a benchmark number

Two rules, both learned from getting them wrong:

1. **Always name the row unit and the q-value column.** `peptides.tsv` contains
   `(peptidoform, charge)` rows but is filtered on `peptide_q_value`, which is a
   base-peptide q, so its row count is not a precursor-q-controlled count. The
   units are tabulated in `README.md` and `docs/11_compete_rescore_fdr.md`.
2. **An identification count is not evidence about anything else.** More
   identifications at a stated threshold says nothing about whether the threshold
   is still calibrated or whether quantification improved. The three objectives are
   separate; `docs/20_sensitivity_and_quantification_playbook.md` is the policy.

ProteoBench's headline is `nr_feature` at `min_obs = 3`: ions quantified in at
least three of the six runs. `median_abs_epsilon_global` is the median absolute
deviation of the measured log2 ratio from the expected one, over all species
pooled; `..._eq_species` weights the three species equally, which matters because
the human ions dominate the pooled figure.

## Recorded results

Both taken with MuMDIA at the second-pass workflow described in
`docs/22_release_plan.md` WP7, which is prototype shell code on the benchmark
machine and not yet in the engine. DIA-NN 2.2.0 was run library-free with
`--reanalyse` on the same files. All figures are `min_obs = 3` ion-level.

### ProteoBench Astral, `LFQ_Astral_DIA_15min_50ng` (module `quant_lfq_DIA_ion_Astral`)

| | features | median abs eps, global | species-equalised | E. coli | CV median |
|---|---|---|---|---|---|
| MuMDIA | 100,528 | 0.176 | 0.223 | 0.301 | 0.105 |
| DIA-NN 2.2.0 | 115,045 | 0.203 | 0.257 | 0.355 | 0.141 |

Species medians of log2(A/B), MuMDIA: human -0.03, yeast +0.85, E. coli -1.75
(expected 0, +1, -2).

### ProteoBench AIF HYE, `LFQ_Orbitrap_AIF_Condition_{A,B}_Sample_Alpha_0{1,2,3}` (module `quant_lfq_DIA_ion_AIF`)

| | features | median abs eps, global | species-equalised | E. coli | CV median |
|---|---|---|---|---|---|
| MuMDIA | 70,689 | 0.154 | 0.204 | 0.283 | 0.314 |
| DIA-NN 2.2.0 | 89,800 | 0.234 | 0.287 | 0.318 | 0.182 |

This module is archived by ProteoBench because Met-oxidised peptides confound its
ratios. Excluding every oxidised precursor moves MuMDIA to 62,200 features and
0.133 global epsilon, and DIA-NN to 78,763 and 0.194: a similar improvement for
both, which is why the confound is the module's and not one tool's.

Read together: accuracy is competitive or better, completeness trails DIA-NN by
13% on Astral and 21% on AIF, and per-ion precision is comparable on Astral but
1.7 times worse on AIF, where the all-ion isolation window leaves interference the
current fragment selection does not remove.

## Reproducing

```
# 1. Per-run analysis and pooled rescore produce pepquant_<run>.parquet.
#    See docs/22_release_plan.md WP7 for the second-pass workflow this used.

# 2. Withhold values with no direct identification in their condition.
bench/prov_filter.py --quant-dir out/quant --pass1-dir out/pool_nombr \
    --out-dir out/quant --rules F1 \
    --runs A_REP1,A_REP2,A_REP3,B_REP1,B_REP2,B_REP3 \
    --conditions A,A,A,B,B,B

# 3. Build the submission TSV.
bench/make_proteobench_submission.py --fasta ProteoBenchFASTA_MixedSpecies_HYE.fasta \
    --out submission.tsv \
    LFQ_Astral_DIA_15min_50ng_Condition_A_REP1=out/quant_F1/pepquant_A_REP1.parquet \
    ...

# 4. Score it offline.
bench/pb_eval.py submission.tsv --format Custom \
    --module quant_lfq_DIA_ion_Astral --out-prefix results/mumdia
```

Uploading is a separate, manual step through the ProteoBench web form; these
scripts never submit anything.

## When adding a result here

State the module, the commit, the row unit, the q-value column, and the inputs.
The engine records the first two for you: `manifest.json` carries `git_sha`,
`commit_date` and the hash of every input, and `Manifest::provenance()` formats the
one-line stamp to quote.
