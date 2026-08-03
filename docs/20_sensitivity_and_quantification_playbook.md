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
(uncapped). The explicit 300 cap is therefore part of this benchmark and not a
default. It is independent of `search_seed.top_n_peaks=300`, which limits seed
probing only and does not truncate spectra used by extraction or quantification.
The cap is acquisition-specific and destructive; do not copy it to another
acquisition without the pre-flight below.

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
| MS2 conversion cap | 300 slightly beat uncapped on chimeric AIF | Must be re-measured per acquisition type; a 300 cap cost 60% of peptides on a 50-window run |
| Seed peak count | `search_seed.top_n_peaks=300` | Seed-only; not a conversion cap |
| Extraction threshold | 0.2 was best with NnTorch; native peaked nearer 0.6 | Tune gate and classifier together |
| Gate meaning | default is one-apex observed-vs-predicted intensity Pearson | Temporal co-elution requires `coelution`/`combined` |
| Apex window/prior | count window 5 and RT prior 120 s helped this AIF run | Validate correct-peak rate and RT residuals |
| RT fine-tune | major imported-library sensitivity lever | Requires enough confident seed anchors; per-file refit is not required (see below) |
| `--max-spectra` | reads the file head | Externally prepare a mid-gradient slice; there is no offset |

### MS2 conversion cap: re-measure per acquisition type

**Recommended guardrail.** `--top-peaks-ms2` is destructive at conversion time.
`peaks_of` sorts each MS2 spectrum by intensity, truncates to the top N, and
writes only the survivors into `spectra_ms2.parquet`
(`rust/mumdia/crates/mumdia/src/stages/convert.rs:76` through `:79`). Extraction
has no cap of its own and consumes whatever `convert` produced, so a cap chosen
on one acquisition silently deletes evidence on another. Peaks removed here
cannot be recovered by any downstream lever; re-conversion is required.

The failure is not a scoring failure. When most peaks are gone,
`extract.presence_min_fragments` (default `3`,
`rust/mumdia/crates/mumdia-core/src/config.rs:523` and `:690`) cannot be met and
real peptides never form a peak group. On a 50-window Orbitrap DIA run, a cap of
300 discarded 78.6% of all MS2 peaks and truncated 85.5% of spectra, and
`mumdia audit` restricted to peptides DIA-NN 2.2.0 confirms are present showed
49,105 of 78,782 (62.3%) stopped at `candidate_generated` with
`NO_PEAK_GROUP`, versus 5,380 lost to FDR and 355 lost to competition. Peptides.tsv rows at `peptide_q_value` <= 0.01
fell from 63,237 uncapped to 25,425 at cap 300 with the decoy fraction at 0.99%
in both arms, so this is lost sensitivity and not a changed threshold. The
canonical treatment of the flag, including the full peak census and the cap
dose-response, is docs/04_convert.md ("Choosing `--top-peaks-ms2`"); docs/18
finding A3 is the same result as a decision record.

**Pre-flight before trusting any cap.** Because `peaks_of` truncates to exactly
`top_n`, a truncated spectrum has exactly `top_n` peaks. The fraction of MS2
spectra sitting exactly at the cap is therefore a direct saturation estimate and
costs one read of the converted artifact:

```python
import polars as pl

CAP = 300
mz = pl.read_parquet("out_dir/spectra_ms2.parquet", columns=["mz"])
n = mz.select(pl.col("mz").list.len().alias("n"))["n"]
print("at cap:", float((n == CAP).mean()))
print("p25/p50/p95/max:", n.quantile(0.25), n.median(), n.quantile(0.95), n.max())
```

Interpretation:

- a small saturated fraction means the cap is mostly inactive and safe to keep
  (the AIF reference file saturates about 47.8% of spectra at 300);
- a large saturated fraction means the cap is the dominant filter on the run (the
  50-window run saturates 85.5%, and even its 25th percentile spectrum holds 572
  peaks);
- if the p25 count equals the cap, the cap is truncating essentially every
  spectrum and the number is meaningless as a distribution summary.

Run the same check on a short uncapped conversion (`convert --max-spectra N`
without `--top-peaks-ms2`) to see the true distribution before choosing a value.
Then confirm with an end-to-end cap sweep, since saturation
alone says how much is removed, not how much it costs. Default to `0` (uncapped)
whenever the sweep has not been run; the measured cost of uncapping on the
50-window run was 12.2x accepted candidates and 57.9 s to 91.9 s of extract wall
clock, which is small against a 60% peptide loss. When peak volume genuinely
must be bounded, take the cap from the knee of the sweep rather than from another
acquisition: on that run a cap of 1,400 recovered 83 of the 87 achievable
percentage points at about 77% of the peak volume.

### RT fine-tuning: once per library, not once per file

**Validated reference.** DeepLC fine-tuning of library iRT is a major
imported-library sensitivity lever and must happen. It does not have to happen
per file. Measured: a library whose `predicted_irt` was fine-tuned once and then
predicted across every peptidoform, run with `rt_im_train.finetune_deeplc =
false` so only the per-run LOESS calibration applies, gave median absolute RT
residual 6.06 s, MAD 6.11 s, slope 0.9907, intercept 16.4 s, against 6.14 s and
6.18 s with per-file fine-tuning. That is equal or marginally better, and it
removes about 36 minutes per file (2,166 s of a 5,127 s single-file run, covering
the fine-tune plus whole-library iRT prediction over roughly 5M peptidoforms).

This applies to a library fine-tuned once and re-predicted in full. It does not
license reusing a stale per-run `_ft` table produced on a different file, which
has previously underperformed a fresh per-run fit. Practical policy: for a
multi-file experiment against one fixed library, fine-tune the library once, pass
the fine-tuned library to every run, and set `finetune_deeplc = false`. For a
single file, or when the library is changing, keep the per-run fine-tune. Either
way, record which variant produced the library used, since the two are not
interchangeable in a benchmark comparison.

MS1 isotope XICs are no longer a missing-evidence placeholder: normal
orchestrated extraction writes `ms1_mono`, `ms1_iso1`, and `ms1_iso2` traces when
MS1 scans and a usable grid are available. They remain absent for older
artifacts, standalone extraction without `--ms1`, or candidates without usable
MS1/grid evidence.

`extract.retain_top_peaks > 1` still writes only an unscored
`<psms>.peaks.parquet` research sidecar. The primary PSM table continues to carry
one selected peak, so increasing K does not currently increase model-visible
identifications.

## Library completeness

**Recommended guardrail.** Sensitivity is bounded by the search space. A peptide
that is not in the library, or not enumerated by the digest, cannot be
identified, and no downstream lever recovers it. Check and record library
completeness before attributing a count gap to extraction or rescoring.

N-terminal methionine excision (native digest). `digest.n_term_met_excision`
(default `true`) makes the digest, for a peptide anchored at protein position 0
whose first residue is `M`, also emit the initiator-Met-removed form. The
initiator methionine is cleaved in vivo for most proteins, so this matches
DIA-NN `--met-excision`; omitting it makes the search database structurally miss
those peptides. Excision keys on protein position 0, not any interior or other
non-terminal `M`. Old configs still parse (the struct carries `#[serde(default,
deny_unknown_fields)]`). Met excision is standard-proteomics-correct, but it
changes native-digest output, so treating it as a trusted default remains
entrapment plus second-dataset gated per repository policy.

Imported-library augmentation. MuMDIA does not digest an imported library, so
the same gap is closed with `scripts/augment_library.py`. On the
`LFQ_Orbitrap_AIF_Ecoli_01` benchmark, every peptide DIA-NN reported at 1% FDR
but absent from the MuMDIA search database was an N-terminal Met-excision
peptide (about 209 not-in-database peptides, reduced to 0 after augmentation).
The script reuses the engine's own `digest`, `peptidoforms`, and `predict-frag`
stages so the added peptidoform strings are byte identical to what the search
consumes. It digests the FASTA (Met excision on), set-diffs against the imported
library's target base sequences, predicts native spectra and iRT for the missing
set, offsets `peptidoform_id`, `base_peptide_id`, and `candidate_id` to stay
disjoint, per-precursor max-normalizes the predicted intensities to the
base-peak convention, merges imported plus missing targets, hands off to
`make_shift_decoys.py` (default) or `make_reverse_decoys.py` for paired decoys,
then validates that `candidate_id` is contiguous, `precursor_mz` is ascending,
and both labels are present.

```text
python scripts/augment_library.py \
  --fasta fasta/ecoli.fasta \
  --imported-precursors lib/lib_precursors.parquet \
  --imported-fragments lib/lib_fragments.parquet \
  --out-precursors lib/lib_precursors_aug.parquet \
  --out-fragments lib/lib_fragments_aug.parquet \
  --mumdia-bin <path to mumdia release binary> \
  --config config.local-diann-lib.json \
  --work-dir <scratch>
```

The augmented `lib_precursors_aug.parquet` and `lib_fragments_aug.parquet` then
replace `--lib-precursors` and `--lib-fragments` in the `run` command. Pass the
augmented raw library, not a `_ft` variant, so per-run fine-tuning still runs:
the DeepLC fine-tune re-predicts iRT for the whole library, so imported and
augmented entries share one RT axis with no explicit reconciliation.
`--match-level base_sequence` (default) adds peptides whose stripped sequence is
absent; `peptidoform_charge` also adds absent modforms or charges of present
sequences, which changes the FDR population and is benchmark-gated.
`--decoy-strategy` must match the strategy that built the imported library's
decoys (`shift` by default); paired, collision-checked decoys are the property
of that downstream builder, not of the augment step itself. Using augmentation
as a routine step, rather than a diagnostic, remains benchmark and entrapment
gated.

## Where missing identifications are lost

**Findings with caveats.** All figures below are from a single file
(`LFQ_Orbitrap_AIF_Ecoli_01`), under decoy-based FDR only, not entrapment
validated, with nondeterministic NnTorch. They motivate work; they do not
license a default change.

Against DIA-NN 2.2.0 run library-free from the E. coli FASTA (matched search
space, 1% FDR), MuMDIA reached about 90-92% of DIA-NN peptides, about 89-91% of
precursors (peptidoform plus charge), and about 99-101% of protein groups.
DIA-NN ran its full library-free double pass; MuMDIA ran a single
library-based pass on a DIA-NN-derived library.

After Met-excision augmentation closed the database-completeness gap (209
not-in-database to 0), the DIA-NN peptides MuMDIA still misses are downstream and
faint, roughly 10x lower abundance than the shared set. The loss ranks:

| Stage where the missed peptide is lost | Share of remaining misses |
|---|---|
| Extraction presence/apex | about 49% |
| Rescore | about 26% |
| Seed | about 25% |

The extraction losses are mostly `presence_min_matched` and `min_coelution`, not
the `min_frag_corr` apex gate. Extraction presence/apex is therefore the largest
remaining lever, then rescore, then seed, all on faint signal.

Before reading a presence/apex loss on any other run as an extraction problem,
check the conversion cap. A cap that truncates most spectra produces exactly the
same signature (`NO_PEAK_GROUP` at `candidate_generated`) while the cause is the
converted artifact, not the extraction thresholds. On the 50-window run above,
62.3% of the confirmed-present peptides were lost this way at a cap of 300.

The rescorer is converged and feature-limited, not training-limited. An NnTorch
epoch and round sweep on the augmented pool showed that 10 epochs undertrains
(about -150 to -180 peptides), 25 epochs and 10 rounds is the knee (the default),
and 50 epochs and 20 rounds gave no real gain (+5, within noise). Removing the
extraction presence/apex filters wholesale was a wash (+96 net, roughly 1:1
churn), though decoy FDR stayed valid at 0.98% even under an 18x candidate flood.
The lever is better features or empirical-library spectra, not more epochs,
rounds, or open gates. Do not chase epoch or round counts.

For reference, the NnTorch training loop is seed x fold x round x epoch:

- folds: `rescore.folds`, overridable via `MUMDIA_NN_FOLDS`;
- rounds: `rescore.num_iter`, overridable via `MUMDIA_NN_ITERS`;
- epochs: `MUMDIA_NN_EPOCHS` (environment only, not engine-set; default `25`);
- seeds: `MUMDIA_NN_SEEDS` (default `1`).

## Competition and FDR

Stage `compete` keeps target/decoy label in its key. It resolves redundant
charge/modification/apex siblings separately within targets and within decoys; it
does not directly eliminate a target against its paired decoy. Later grouped-q
reductions in `rescore` select representatives by the requested biological unit
and then compare target and decoy populations to estimate q values.

The default `compete.group_by=precursor` is named misleadingly. It keys on
`(base_peptide_id, label_code, 0, peak_rank)`
(`rust/mumdia/crates/mumdia/src/stages/compete.rs:88`), and `base_peptide_id` is
the stripped sequence (`scripts/import_diann_lib.py:137` factorizes
`Stripped.Sequence`). Only the highest-`prelim` row of each group survives; the
rest are deleted before rescoring
(`rust/mumdia/crates/mumdia/src/stages/compete.rs:321` through `:344`). Every
charge and every modification variant of one peptide therefore collapses to a
single winner pre-FDR, and the default cannot produce a precursor-level count.

`peptidoform_charge` keys `(peptidoform_id, label_code, charge, peak_rank)`
(`rust/mumdia/crates/mumdia/src/stages/compete.rs:93` through `:98`) and is the
true precursor-level competition. Measured on a modification-rich imported
library, the default deleted 880,464 of 1,890,239 extracted candidates (46.6%),
while `peptidoform_charge` removed 0 rows and moved precursors per peptide from
1.000 to 1.174 (DIA-NN reports about 1.126 on comparable data). The peptide count
was unchanged, so the precursor-level key cost nothing there.

**Recommended guardrail for PTM work.** Under the default key, a modified form is
deleted whenever an unmodified or otherwise-modified sibling of the same stripped
sequence scores higher, which is the usual case. Set
`compete.group_by=peptidoform_charge` for any modification search, and for any
precursor-level count or precursor matrix. It also changes the classifier and FDR
population, so a base-peptide sensitivity benchmark should still report the
default key as its comparison baseline, and a change of default remains
benchmark-gated.

Pooling more runs into one rescore does not tighten q. The estimator is
`q = (n_decoys + 1) / max(1, n_targets)`
(`rust/mumdia/crates/mumdia/src/fdr.rs:38`), which is scale-invariant under
replicating the population. The only pool-size-dependent term is the `+1`
pseudocount, whose relative weight shrinks as the pool grows, so a larger pool is
if anything marginally looser. Never explain a per-run count difference by the
number of runs pooled; check the q column and the score distribution instead.

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
| `peptidoform_charge` competition as the global default | precursor-level FDR/entrapment plus completeness and ratio accuracy. Not gated for PTM searches or precursor-level counts, where it is required |
| Adaptive RT windows or tolerance optimization | multiple gradients/instruments with no FDR inflation |
| Shared consensus peak widths or ion sets | known-ratio bias, CV, and missingness across runs |
| Minimum clean-ion/interference rules | accuracy gain without unacceptable missingness |
| MBR transfer/requantification | transfer-decoy FDR and known-ratio validation |
| Any nonzero MS2 conversion cap | the saturation pre-flight plus an end-to-end cap sweep on that acquisition class; `0` is the default and the safe choice |
| N-terminal Met excision as a trusted digest default | entrapment plus a second dataset, though it is standard-proteomics-correct |
| Imported-library augmentation as a routine step | empirical-null FDR plus a second acquisition context |
| Wholesale relaxation of extract presence/apex filters | multi-context empirical-null FDR; the single-file result was a wash |

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
conversion cap and fraction of MS2 spectra saturated at it:
compete.group_by:
RT fine-tune variant (per run, once per library, or none):
actual classifier/model:
random seeds or ensemble:
row unit and q column:
targets/decoys at thresholds:
entrapment design and empirical FDP:
known-ratio error/CV/missingness:
normalization factors:
output directory:
```
