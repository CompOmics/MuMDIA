# Sensitivity and quantification playbook

> Operational decision guide. See docs/09, 11, 12, 18, and 19 for implementation
> details and the historical benchmark evidence.

## Purpose

Identification sensitivity, FDR validity, and quantification accuracy are
separate objectives. A larger identification count is not evidence of valid FDR
or accurate abundance estimates.

This document uses three status labels:

- **Validated reference**: measured on the named dataset with the stated
  acquisition, library, command, and result unit.
- **Recommended guardrail**: required for operational or statistical correctness;
  it is not itself a sensitivity claim.
- **Benchmark-gated**: plausible or useful experimentally, but not a production
  default until it passes the stated validation.

Always name the result unit:

| Question | Unit and q column |
|---|---|
| Base-peptide sensitivity | unique `base_peptide_id` at `peptide_q_value` |
| Single-run precursor list or matrix | `(peptidoform, charge)` at `precursor_q` |
| Per-run quant after pooled rescore | one `source` slice at `run_psm_q` |
| Protein identification | protein group at `pg_q_value` |
| Quantification accuracy | known-ratio error, CV, missingness, and dynamic range |

`peptides.tsv` is a hybrid: rows are `(peptidoform, charge)`, but selection and
the displayed q value use `peptide_q_value`. It is not a stripped-peptide table
or a precursor-q-controlled precursor table.

## Validated Orbitrap AIF sensitivity reference

**Validated reference.** The historical maximum-sensitivity result is scoped to
`LFQ_Orbitrap_AIF_Ecoli_01.mzML`, an imported DIA-NN E. coli library with paired
decoys, per-run DeepLC fine-tuning, Extended features, the default
`apex_pearson` gate at 0.2, and strict NnTorch rescoring.

Run from the repository root, using a fresh output directory:

```text
mumdia doctor --config config.local-diann-lib.json

mumdia run \
  --lib-precursors lib/lib_precursors.parquet \
  --lib-fragments lib/lib_fragments.parquet \
  --mzml mzml_files/LFQ_Orbitrap_AIF_Ecoli_01.mzML \
  --out-dir out_aif_nn \
  --config config.local-diann-lib.json \
  --top-peaks-ms2 300
```

The raw precursor table is intentional. Fine-tuning reads it and writes
`out_aif_nn/fragment_library_precursors_ft.parquet`; it does not modify the
input. `run` then uses and records the new table downstream.

The effective reference settings are:

| Setting | AIF reference |
|---|---|
| `features.set` | `extended` |
| `extract.gate_mode` | `apex_pearson` |
| `extract.min_frag_corr` | `0.2` |
| `extract.apex_count_window` | `5` |
| `extract.apex_rt_prior_s` | `120` seconds |
| `rt_im_train.finetune_deeplc` | `true` |
| `rt_im_train.rt_window_multiplier` | `1.5` |
| `rescore.classifier` | `nn_torch` |
| `rescore.strict` | `true` |
| conversion cap | CLI `--top-peaks-ms2 300` |

Both standalone `convert` and `run` default `--top-peaks-ms2` to `0`
(uncapped). The explicit 300 cap is therefore part of the benchmark. It is
independent of `search_seed.top_n_peaks=300`, which limits seed probing only and
does not truncate spectra used by extraction or quantification.

The historical result is about 10,300 `(peptidoform, charge)` rows in
`peptides.tsv` at `peptide_q_value <= 0.01`. It is a regression target, not a CI
assertion, a precursor-q count, or a universal DIA preset. The accepted-set decoy
fraction was approximately 0.98-0.99% at the 1% threshold; that is a
target-decoy sanity check, not an independent empirical-null validation.

**Recommended guardrails.**

- Keep `rescore.strict=true`; a requested sidecar must fail loudly rather than
  silently becoming native TDA.
- Verify `params.classifier` and `model_identity` in
  `psms_scored.parquet.report.json`. The report is the source of truth for the
  model that actually ran.
- Use a fresh output directory. `run` always recomputes, has no cache/resume, and
  does not remove stale optional files from a reused directory.
- Record the exact library and decoy provenance. Counts are uninterpretable
  without the search-space definition.
- Repeat stochastic runs when comparing small gains. NnTorch seeds NumPy/Torch
  but is not bit-deterministic; DeepLC fine-tuning is unseeded.

## Acquisition-specific choices

Do not generalize the AIF settings without a sweep on the new acquisition:

| Lever | Established on this AIF file | Outside this context |
|---|---|---|
| MS2 conversion cap | 300 slightly beat uncapped on chimeric AIF | Retune for narrow-window/SWATH/Astral data |
| Seed peak count | `search_seed.top_n_peaks=300` | Seed-only; not a conversion cap |
| Extraction threshold | 0.2 was best with NnTorch; native peaked nearer 0.6 | Tune gate and classifier together |
| Gate meaning | default is one-apex observed-vs-predicted intensity Pearson | Temporal co-elution requires `coelution`/`combined` |
| Apex window/prior | count window 5 and RT prior 120 s helped this AIF run | Validate correct-peak rate and RT residuals |
| RT fine-tune | major imported-library sensitivity lever | Requires enough confident seed anchors |
| `--max-spectra` | reads the file head | Externally prepare a mid-gradient slice; there is no offset |

MS1 isotope XICs are no longer a missing-evidence placeholder: normal
orchestrated extraction writes `ms1_mono`, `ms1_iso1`, and `ms1_iso2` traces when
MS1 scans and a usable grid are available. They remain absent for older
artifacts, standalone extraction without `--ms1`, or candidates without usable
MS1/grid evidence.

`extract.retain_top_peaks > 1` still writes only an unscored
`<psms>.peaks.parquet` research sidecar. The primary PSM table continues to carry
one selected peak, so increasing K does not currently increase model-visible
identifications.

## Competition and FDR

Stage `compete` keeps target/decoy label in its key. It resolves redundant
charge/modification/apex siblings separately within targets and within decoys; it
does not directly eliminate a target against its paired decoy. Later grouped-q
reductions in `rescore` select representatives by the requested biological unit
and then compare target and decoy populations to estimate q values.

The default `compete.group_by=precursor` collapses charge/modification siblings
within each label before rescoring. `peptidoform_charge` preserves precursor
forms and may improve precursor-matrix completeness, but it changes the
classifier/FDR population and is benchmark-gated.

A sensitivity setting may be promoted only when:

1. Strict mode completed with the requested classifier.
2. Labels, target-decoy pairing, and library search-space sizes were validated.
3. Every reported count names its row unit and q column.
4. The gain survives repeated stochastic runs and at least two relevant datasets
   or acquisition contexts.
5. Decoy behavior remains calibrated at the declared threshold.
6. An entrapment or other empirical-null experiment shows no material inflation.
7. The commit, command, resolved config, input/library hashes, model identity,
   seeds, target counts, decoy counts, and entrapment counts are retained.

For an entrapment experiment:

```text
empirical FDP = (entrapment_ratio * accepted_entrapments + 1)
                / max(1, accepted_real_targets)
```

Derive `entrapment_ratio` from real/foreign library search-space sizes, and
declare the protein marker, shared-peptide exclusions, and contaminant markers
before scoring. `classifier=entrapment` is an alternate rescoring/q-value path;
it is not an automatic read-only certification of an NnTorch accepted set.

## Quantification accuracy

**Recommended guardrail:** an accepted identification can be unquantifiable.
Missing evidence must remain missing, not become biological zero.

The current quant contract:

- carries `apex_rt` and elution bounds through competition/rescoring and anchors
  integration at the identified apex, with a legacy fallback for older artifacts;
- emits nullable `quantity`, `quant_status`, and the applied integration apex and
  bounds;
- sums only positive finite fragment areas;
- excludes MS1 precursor XIC pseudo-traces from the fragment Top-N sum;
- deduplicates sibling charge/modification rows by `base_peptide_id`, taking the
  maximum representative before protein Top-N rollup;
- ignores null, non-finite, and nonpositive inputs during LFQ combination.

Use Parquet for numerical work. TSV quantities are presentation-rounded to one
decimal and omit the quant status and integration diagnostics.

For a single-run precursor matrix:

```json
{ "quant": { "q_filter": "precursor_q" } }
```

For per-run quantification after experiment-wide rescoring:

```json
{ "quant": { "q_filter": "run_psm_q" } }
```

In the pooled case, first slice `psms_scored.parquet` by `source` and pair each
slice with that run's chromatograms. Quant has no source selector;
`q_filter=run_psm_q` selects a q column, not a run. The AIF sensitivity config's
`run_psm_q` setting is therefore not evidence that it is the optimal single-run
precursor filter.

`peak_window_mode=per_candidate` remains the default. Consensus half-widths are
estimated separately inside each quant invocation; they are not currently
shared across runs. Median-ratio normalization uses positive complete-case
features. If none exist, factors fall back to `1.0`; inspect the logged
`size_factors` before interpreting normalized ratios.

Report inclusion uses peptide q independently of `quant.q_filter`. A TSV report
row can therefore have blank quantity even when the identification is accepted.

Judge quant changes on known-ratio data using:

- bias or absolute log2-ratio error;
- within-condition CV and missingness;
- complete-case fraction, linearity, and dynamic range;
- blank/negative-control leakage;
- usable-fragment counts and integration apex/bound diagnostics;
- normalization factors.

Report performance both on a fixed common identification set and on all accepted
IDs so selection effects remain visible.

## Benchmark-gated work

| Candidate change | Evidence required before default promotion |
|---|---|
| Looser/alternative/soft extraction gate or MS1 rescue | multi-context sensitivity plus empirical-null FDR |
| Model-visible top-K peaks | stable `candidate_id + peak_rank`, competition/FDR, and known-ratio quant |
| `peptidoform_charge` competition | precursor-level FDR/entrapment plus completeness and ratio accuracy |
| Adaptive RT windows or tolerance optimization | multiple gradients/instruments with no FDR inflation |
| Shared consensus peak widths or ion sets | known-ratio bias, CV, and missingness across runs |
| Minimum clean-ion/interference rules | accuracy gain without unacceptable missingness |
| MBR transfer/requantification | transfer-decoy FDR and known-ratio validation |
| Acquisition peak caps | a separate cap sweep per acquisition class |

A one-file AIF count can nominate an experiment; it cannot establish a universal
default.

## Reproduction record

Retain this minimum record with every sensitivity or quantification benchmark:

```text
commit/build:
dataset and acquisition:
library provenance and decoy method:
exact command:
resolved config:
conversion cap:
actual classifier/model:
random seeds or ensemble:
row unit and q column:
targets/decoys at thresholds:
entrapment design and empirical FDP:
known-ratio error/CV/missingness:
normalization factors:
output directory:
```
