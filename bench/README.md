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

### Identification, `LFQ_Orbitrap_AIF_Ecoli_01.mzML` (2026-08-28, post-audit engine)

Measured after the `docs/25_release_readiness_review.md` fixes, to re-establish the
identification number under the three promoted defaults and the NnTorch fold-key
change. Engine at `9fc1c06` plus the two fixes this run itself produced.

Command: `mumdia run` with the AUGMENTED imported DIA-NN library
(`lib_precursors_aug.parquet`, 1,798,200 candidates), DeepLC 4.1.0 per-run
fine-tune, Extended features, `extract.gate_min_score = 0.2` set explicitly,
`nn_torch`, `--top-peaks-ms2 300`. 7 min 54 s wall on the development machine.

| row unit | q column | threshold | count |
|---|---|---|---|
| stripped peptides | `peptide_q_value` | 0.01 | **10,914** |
| stripped peptides | `run_psm_q` | 0.01 | 10,825 |
| (peptidoform, charge) | `peptide_q_value` | 0.01 | 10,914 |
| protein groups | `pg_q_value` | 0.01 | 1,634 |

Empirical decoy fraction at the 1% threshold: **0.0098**, on every q column. The
(peptidoform, charge) count equals the stripped-peptide count exactly -- 1.000
precursors per peptide -- because the default `compete.group_by = base_peptide`
collapses every charge and modification sibling before FDR. That is not a precursor
count; use `peptidoform_charge` if a precursor unit is wanted.

**Two corrections to the record this run forced.** The file is NOT all-ion: it has
152 contiguous 4 m/z isolation windows over 396.4-1004.7 m/z. And the 300-peak cap
is nearly inert on it -- 4.7% of MS2 spectra saturate 300, with p25 = 15, p50 = 44,
p75 = 135 peaks per spectrum -- against the 47.8% saturation `CLAUDE.md` records for
"the chimeric AIF benchmark run". Either the local file was re-converted since that
measurement or the characterisation was wrong; do not carry either figure forward
without re-deriving it. The narrow windows are also why the search takes ~70 s of the
wall time rather than the tens of minutes previously recorded.

#### A/B: `extract.apex_evidence_rank`, the promoted default

Both arms reuse ONE set of upstream artifacts from the run above -- same spectra,
same seed PSMs and mass calibration, same DeepLC-fine-tuned library, same RT windows
-- so the apex change is isolated from DeepLC draw variance, which alone has moved
held-out window sizing by 150-211 s across two draws of one arm. Both arms were
re-run rather than reusing the numbers above, because `nn_torch` is not
bit-deterministic.

| `apex_evidence_rank` | accepted by extract | stripped peptides, `peptide_q_value` <= 0.01 | decoy fraction |
|---|---|---|---|
| `true` (promoted default) | 45,338 | 10,914 | 0.0098 |
| `false` (previous default) | 51,111 | 10,921 | 0.0098 |

**Identification is flat**: a 7-peptide difference (0.06%), with 96.6% of the union
shared. The promotion neither gains nor costs sensitivity here, which is the evidence
it needed; it is not a sensitivity win, and it was not promoted as one.

**Quantification is not flat.** Of the 10,727 peptides both arms identify, **48.3%
have an apex more than 1 s apart** -- median 2.9 s, p90 9.9 s, max 104.8 s. Quant
integrates around that apex, so quantities from before this change are not comparable
with quantities after it.

Which apex is better, scored against `rt_pred_cal` from the shared window table
(independent of both arms): evidence rank is closer at every percentile -- median
27.67 s vs 29.69 s, beyond 30 s 47.06% vs 49.61%, beyond 60 s 20.40% vs 21.41%.
Consistent in direction, small in size, and a weak instrument: with `w_rt` at 187 s
the reference's own error dominates that ~28 s median, so it can show direction but
cannot resolve the apex improvement precisely.

### FDR validity: entrapment, E. coli + human 1:1 (2026-08-28)

The empirical-null experiment `docs/20` requires before promoting anything, run
specifically to test the three entrapment-path defects in `docs/25` section 3.

Spike-in built from `fasta/ecoli_22032024.fasta` plus a seeded 1:1-by-protein-count
sample of `fasta/human.fasta` (seed 20260828), real accessions prefixed `REAL_` and
spike-in accessions `ENTRAP_`. The prefixes are not cosmetic: the E. coli FASTA
already bundles 358 contaminants with a `Cont_` accession prefix, **150 of them
`_HUMAN`**, so using `_HUMAN` as the marker would have scored real contaminants as
entrapment negatives and inflated the measured FDR.

Declared before scoring, as the playbook requires: marker `ENTRAP_`, exclude `REAL_`
(protecting the 13,258 candidates shared between the two proteomes), contaminant
markers `KRT`, `K1C`, `K2C`, `ALBU`, `TRYP`.

`entrapment_ratio` **measured from the built library, not assumed**: 1,046,870 real
against 1,867,304 entrapment target candidates gives **0.560632**. A 1:1 protein
count does not give a 1:1 peptide count, because human proteins are longer -- which
is exactly what the ratio exists to correct.

Native predictors, native linear entrapment rescorer (`rescore.python` deliberately
unset, because that is the path the defect lived on), 5,828,348 candidates,
`gate_min_score = 0.6`, same converted spectra as the arm above.

| | count |
|---|---|
| accepted real targets, `peptide_q_value` <= 0.01 | 786 |
| accepted entrapment | 12 |
| accepted decoys, `peptide_q_value` <= 0.01 | **0** |
| accepted decoys, `q_value` <= 0.01 | 15 |

Empirical FDP = `(0.560632 x 12 + 1) / 786` = **0.0098** against a 1% threshold:
calibrated.

What each number tests:

- **0 decoys on `peptide_q_value` against 15 on `q_value`** is the signature of the
  grouped-q fix. Only the grouped path excludes decoys from the competition, so a
  decoy can no longer win a base-peptide group, take its q, and leave the real target
  at 1.0.
- **The score distribution** tests the training fix. Entrapment median -4.701 sits
  on top of the decoy median -4.693, both far below the real-target tail (real p99
  24.28, entrapment p99 -3.34), and the top 500 rows by score are 496 real / 2
  entrapment / 2 decoy. Entrapment behaves as a held-out negative class, which it
  could not do while being recruited as a positive training example.
- **The ratio** is accepted because it is positive; 0 or negative is now rejected at
  config validation, where it used to make every PSM pass at any threshold.

Not comparable with the identification arm above: different library (5.8M
native-predicted candidates against 1.8M DeepLC-fine-tuned), different rescorer, and
a strictly harder null. The number to read here is the FDP, not the 786.

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

## Resource profile of a reference run

Measured, not estimated: one AIF file from the HYE set on a 128-core shared Linux
server, imported DIA-NN library, per-run DeepLC fine-tune, Extended features,
`nn_torch` rescoring. Input 1.94 GB mzML, 232,976 MS2 spectra, 1,815,610 PSMs
after extraction. Per-stage figures come from each artifact's `report.json`
`elapsed_ms`, and the disk figures from the artifacts themselves.

| stage | wall clock | note |
|---|---|---|
| convert | 20 s | 232,976 MS2 spectra, uncapped |
| search-seed | 53 s | 270,879 seed PSMs |
| rt-im-train | 4 s | LOESS on the seed anchors |
| extract | 320 s | 27.1M chromatograms |
| features | 432 s | 387 features x 1.82M rows |
| compete | 43 s | |
| rescore | 4,100 s | `nn_torch`, single run, 32 threads under load |
| quant | 124 s | peptide, protein and fragment output |
| **total** | **about 85 min** | |

Rescoring is 80% of it. Two things follow. Threads matter more there than
anywhere else and not in the direction one expects: the NN worker measured faster
on 8 threads than on 32, so `--threads` is worth setting. And an experiment-wide
pooled rescore is the right shape for a multi-run analysis, because it replaces
one rescore per run with one for the batch, at a measured 0.834 ms per PSM.

Disk is the other surprise. The run wrote 13.1 GB of artifacts from a 1.94 GB
input, a factor of 6.8:

| artifact | size | rows |
|---|---|---|
| chromatograms | 4.53 GB | 27,109,906 |
| features | 3.78 GB | 1,815,610 |
| psms_competed | 3.78 GB | 1,815,610 |
| spectra_ms2 | 0.34 GB | 232,976 |
| run_windows | 0.32 GB | 10,881,402 |
| everything else | 0.36 GB | |

For RAM, the figure to size from is the pooled rescore's feature matrix:
`n_psms x n_features x 4` bytes. `MUMDIA_NN_STREAM_GB` (default 4) chooses between
holding it in memory and a much slower disk-backed memmap, and a matrix marginally
over the threshold silently takes the slow path, so set it deliberately for a large
pool.

## When adding a result here

State the module, the commit, the row unit, the q-value column, and the inputs.
The engine records the first two for you: `manifest.json` carries `git_sha`,
`commit_date` and the hash of every input, and `Manifest::provenance()` formats the
one-line stamp to quote.
