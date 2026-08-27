# compete, rescore, and FDR

> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

This subsystem is the tail of the identification chain (Stage F). It turns the
per-PSM feature table produced by the `features` stage into a scored,
FDR-controlled result set. It has three parts:

1. **compete** (`mumdia compete`): within each competition group, resolve
   redundant candidates before target-decoy counting, so several plausible forms
   of one identification cannot each be counted as a discovery. Default behaviour
   keeps only the best-scoring candidate per group, and the default group is the
   whole stripped peptide within one label, not one precursor (see "compete:
   grouping" below).
2. **rescore** (`mumdia rescore`): train a semi-supervised classifier over the
   competed PSMs of the whole experiment, produce a single discriminant `score`
   per PSM, and derive native target-decoy q-values at several aggregation
   levels (PSM, per-run PSM, precursor, peptide, protein group).
3. **fdr** (`crate::fdr`): the shared, stateless q-value kernels. Both
   `search-seed` and `rescore` call these; they are not a stage.

The design principle behind the sensitivity work is "preserve candidate evidence
until the workflow can make a well-calibrated decision". That is why every
non-default competition mode and the label rule keep decoys and redundant
variants alive so the rescorer and FDR have a valid null to work against.

## Files

| path | role |
|---|---|
| `rust/mumdia/crates/mumdia/src/stages/compete.rs` | `compete` stage: grouping, within-group competition resolution, competed table + optional audit |
| `rust/mumdia/crates/mumdia/src/stages/rescore.rs` | `rescore` stage: input concat, classifier dispatch, multi-context q-values, scored table |
| `rust/mumdia/crates/mumdia/src/rescoring.rs` | `percolator_lite`, the native semi-supervised linear rescorer (the `native_tda` path) |
| `rust/mumdia/crates/mumdia/src/fdr.rs` | q-value kernels: `target_decoy_q`, `entrapment_q`, `count_targets_at_q`, `validate_labels`, `ln_factorial` |
| `rust/mumdia/crates/mumdia-core/src/rejection.rs` | `RejectionReason` codes; compete's audit `rejection_reason` uses `code()` |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `CompeteConfig`, `CompetitionMode`, `CompeteGroupBy`, `RescoreConfig`, `RescorerKind` |
| `scripts/mokapot_worker.py` | Mokapot PIN sidecar (`RescorerKind::Mokapot`) |
| `scripts/nn_rescore_worker.py` | PyTorch MLP sidecar (`RescorerKind::NnTorch`); reads either the tab-separated PIN or the Parquet feature table |
| `scripts/entrapment_worker.py` | entrapment GBM sidecar (`RescorerKind::Entrapment`) |

## Inputs and outputs

### compete

Consumes the features artifact `psms_features` (path passed as `--features`),
plus its schema companion `<features>.schema.json` (read at compete.rs:46 via
`FeatureSchema::read`). Reads these columns (compete.rs:31-48): `candidate_id`
(u32), `label` (str), `base_peptide_id` (u32), `peptidoform` (str), `protein`
(str), `apex_rt`, `elution_lo`, `elution_hi` (f64), `precursor_mz` (f64),
`prelim_score` (f64), `peak_rank` (i32, defaulted to 0 when absent,
compete.rs:45), `charge`
(f64, only needed for `peptidoform_charge` grouping), and every feature column
named in the schema.

Produces `psms_competed` (schema version 3, `artifact::PSMS_COMPETED`,
schema.rs:20) at `--out`. Column schema includes `candidate_id`
(u32), `peak_rank` (i32), `label` (str), `base_peptide_id` (u32), `peptidoform`
(str), `protein`
(str), `apex_rt`, `elution_lo`, `elution_hi` (f64), `precursor_mz` (f64),
`prelim_score` (f64), then every
feature column (f64) carried through unchanged. It also writes
`<out>.schema.json` (compete.rs:188) so rescore recovers the exact feature list,
and `<out>.report.json`.

Optional sidecar `<out>.compete_audit.parquet` (only when
`compete.emit_competition_audit = true`, compete.rs:193-238): one row per removed
candidate with `candidate_id` (u32), `label` (str), `peptidoform` (str),
`winner_candidate_id` (u32), `loser_prelim` (f64), `winner_prelim` (f64),
`rejection_reason` (str). The reason string is `RejectionReason::code()`
(rejection.rs:50), which is `SCREAMING_SNAKE_CASE`, so the two values written here
are `OUTCOMPETED_BY_DECOY` and `OUTCOMPETED_BY_TARGET` (rejection.rs:64-65), not
lowercase. The reason is keyed on the **loser's own label** (`reason_of`,
compete.rs:194-200): a removed decoy is `OUTCOMPETED_BY_DECOY`, a removed target is
`OUTCOMPETED_BY_TARGET`. Because competition is within-label, the winner shares the
loser's label, so this equivalently describes the winner. `RejectionReason`
(rejection.rs:19-45) has 17 variants: 16 loss stages spanning the whole ladder
plus the `Reported` sentinel. compete only ever emits these two.

### rescore

Consumes one or more `psms_competed` tables (`--competed`, a `Vec<String>`).
The ordered feature schema from the first input is the expected contract, and
every later schema companion must match it exactly before tables are
concatenated. Reads `candidate_id`, `label`, `base_peptide_id`, `peptidoform`,
`protein`, `charge` (f64, cast to i32), `prelim_score`, `precursor_mz`,
`apex_rt`, `elution_lo`, `elution_hi`, `peak_rank` (i32, defaulted to 0 when
absent, rescore.rs:93), and the feature columns.

Four input conditions are hard errors before any scoring: fewer than 2
`rescore.folds` (rescore.rs:46-48), an empty `--competed` list (rescore.rs:43-45),
zero targets or zero decoys (rescore.rs:117-125), and any non-finite feature,
`prelim_score`, or `precursor_mz` (rescore.rs:126-143). A non-finite classifier
score is also rejected after scoring (rescore.rs:284-290).

Produces `psms_scored` (schema version 4, `artifact::PSMS_SCORED`, schema.rs:21)
at `--out`. Column schema is, in order (rescore.rs:508-541): `candidate_id` (u32),
`peptidoform` (str), `charge` (i32), `label` (str), `protein` (str),
`base_peptide_id` (u32), the carried identification `apex_rt`/`elution_lo`/
`elution_hi` (f64), `score` (f64, the classifier discriminant),
`q_value` (f64, pooled PSM q), `peptide_q_value` (f64), `protein_group` (str,
the protein-accession-set string, a duplicate of `protein`), `pg_q_value` (f64),
`global_q_value` (f64, byte-identical alias of `q_value`), `prelim_score` (f64),
`source` (u32, index into `--competed` identifying the run), `run_psm_q` (f64),
`experiment_psm_q` (f64, alias of `q_value`), `precursor_q` (f64), and
`selected_peak_rank` (i32, which chromatographic peak of the candidate the
rescorer kept; `0` = the up-front apex). Plus
`<out>.report.json` whose `params` records `classifier` (the path actually taken),
`classifier_requested`, `strict`, `folds`, `num_iter`, `train_fdr`,
`feature_schema_id`, `competed_inputs`, `config_hash` (rescore.rs:565-575),
`model_identity` (e.g. `native-percolator-lite-v1`, `mokapot-<estimator>`
(default `mokapot-nn`), `nn-torch-semisup-sidecar-v1`,
`entrapment-gbm-sidecar-v1`, `native-percolator-lite-entrapment-v1`,
rescore.rs:149/174/201/256/266/277), and `stats` records `psms`, `classifier`,
`target_psms_at_1pct`, `target_peptides_at_1pct`, `target_protein_groups_at_1pct`,
`target_precursors_at_1pct`, plus `entrapment_ratio` and
`entrapment_peptides_at_1pct` in entrapment mode (rescore.rs:544-557).

Sidecar working files are written under `--work-dir` (created with
`create_dir_all`, rescore.rs:753, 928). Filenames are per-invocation, tagged with
the output artifact's file stem plus the process id (rescore.rs:935-960), so two
concurrent rescores cannot clobber each other's handoff files. Two distinct file
contracts exist:

- **PIN sidecars** (`RescorerKind::Mokapot`, `RescorerKind::NnTorch`): input
  `rescore_<tag>.pin` (Percolator tab format) or, under
  `rescore.handoff = parquet` with the NnTorch worker,
  `rescore_<tag>.features.parquet`; output `rescore_<tag>_out.parquet`
  (rescore.rs:955-960). The PIN header is `SpecId  Label  ScanNr  ExpMass
  CalcMass  <features...>  Peptide  Proteins` (rescore.rs:975-977); each row is
  `psm_{i}`, `Label` = `-1` for a decoy else `+1` (rescore.rs:985), `ScanNr` = `i`
  (the flat row index), `ExpMass` = `CalcMass` = `precursor_mz` (5 decimals),
  the feature columns in schema order (6 decimals), `Peptide` = `-.<peptidoform>.-`,
  `Proteins` = `<protein>` (rescore.rs:984-993). The Parquet form carries the same
  logical columns under the same names, written in 250k-row batches with f32
  features (`write_features_parquet`, rescore.rs:833-906); f32 is not a precision
  loss relative to the PIN, which already wrote `{:.6}` and whose values the worker
  casts to f32 anyway. The worker echoes the SpecId tail
  as a `candidate_id` column (here the row index `i`) and emits a `score` column;
  scores are mapped back by row index through `align_sidecar_scores`, which
  requires every input row to be covered exactly once with a finite score and
  hard-errors on a missing, duplicate, out-of-range, or non-finite entry
  (rescore.rs:1032-1035, 1041-1077). There is no worst-score fallback for a missing
  row.
- **Entrapment GBM** (`RescorerKind::Entrapment` with `rescore.python` set): input
  `entrapment_in.parquet`, output `entrapment_out.parquet` (rescore.rs:754-755).
  The input parquet columns are `row_id` (u32, the flat row index used for
  score readback), `candidate_id` (u32), `base_peptide_id` (u32), `is_entrapment`
  (i32, 0/1), `is_decoy` (i32, 0/1), then the feature columns in schema order
  (f64) (rescore.rs:757-778). The worker is invoked as
  `python entrapment_worker.py <in> <out> <folds>` (rescore.rs:781-788, `folds`
  from `cfg.folds`); it reads back `row_id` (u32) + `score` (f64) and maps by
  `row_id` through the same `align_sidecar_scores` coverage contract, so an
  incomplete or duplicated response hard-errors (rescore.rs:793-796, 1041-1077).
  Both sidecars are run with `PYTHONUTF8=1` and a non-zero exit is a hard error
  (rescore.rs:787/789-791, 1007/1024-1026). The PIN sidecar is spawned behind a
  `ChildGuard` (rescore.rs:803-819) that kills the worker if the parent unwinds;
  without it a killed `mumdia` leaves the Python process holding its feature
  memmap and every later rescore fails on a file it cannot delete.

## How it works

### compete: grouping

`compete::run` (compete.rs:28) builds a competition group key per PSM
(compete.rs:71-101). The key is a fixed-size tuple `(u32, u8, i64, i32)` rather
than a freshly allocated String, chosen so grouping does not allocate per PSM:

- element 0 is `base_peptide_id` for `Precursor`/`Apex`, or a dense
  first-appearance `pform_id` for `PeptidoformCharge` (built at compete.rs:59-69);
- element 1 is the label code `0=target, 1=decoy, 2=other` (compete.rs:81-85);
- element 2 is a bucket: constant `0` for `Precursor` (compete.rs:88); the rounded
  apex-RT bucket `(apex_rt / apex_rt_tolerance_s).round()` for `Apex`
  (compete.rs:90); the rounded charge for `PeptidoformCharge` (compete.rs:96);
- element 3 is `peak_rank` (compete.rs:41-45, 86), so the top-K peak alternatives
  of one candidate compete only within their own rank. It is `0` for every row
  when top-K promotion is off, leaving the grouping unchanged.

**The label is part of the key on purpose.** A target is never competed against
its own decoy; competition only arbitrates redundant charge/mod variants within
the target population and, separately, within the decoy population. If a target
could evict its paired decoy, the decoy population would be depleted and the
target-decoy null would collapse, badly underestimating FDR. This is stated at
compete.rs:71-78 and re-stated on `CompetitionMode` in config.rs:865-870.

#### `group_by = precursor` does not group by precursor

The default variant is named `Precursor`, but element 0 of its key is
`base_peptide_id` (compete.rs:88), which is a **stripped-sequence** id in both
library paths: `pd.factorize(Stripped.Sequence)` on the imported path
(import_diann_lib.py:137), and the target peptide row id (shared by a decoy and
its target) on the native path (peptidoforms.rs:216-220). Every charge state and
every modification variant of one stripped peptide therefore falls in a single
group, and `WinnerTakeAll` keeps only the highest-`prelim_score` member and
deletes the rest (compete.rs:336-344, 349-358) before rescore and before any
target-decoy counting. The grouping is peptide-level, not precursor-level.

Two consequences follow.

1. Under this grouping at most one precursor form of a peptide survives compete
   per peak rank, so the downstream `precursor_q` column counts base peptides
   rather than precursors. A count reported from it is not a precursor count.
2. In a search with variable modifications, a modified form is deleted whenever
   its unmodified or otherwise-modified sibling scores higher, which is the
   common case. Measured on a modification-rich imported library, compete removed
   880,464 of 1,890,239 extracted candidates (46.6%). With
   `group_by = peptidoform_charge` compete removed 0 rows, the peptide count was
   unchanged, and precursors per peptide went from 1.000 to 1.174 (DIA-NN reports
   about 1.126 on comparable data).

In that single comparison the peptide count was unchanged, so the precursor-level
key cost nothing at the peptide level while recovering the sibling forms. Treat
`peptidoform_charge` as **required** for a PTM-oriented search, where the deleted
sibling is the analyte of interest. It stays benchmark-gated as a global default
because it changes the population the rescorer trains on and the population FDR is
estimated over, and that has not been validated across datasets.

`PeptidoformCharge` (config.rs:894-900) is the precursor-level grouping DIA-NN and
Spectronaut report at: sibling charges **and** sibling modforms of one peptide are
kept as separate groups rather than collapsed, because element 0 is the
peptidoform id and the charge sits in element 2. It requires the `charge` column
and bails if absent (compete.rs:52-55).

### compete: resolution

`resolve_competition` (compete.rs:321) is a pure, unit-tested function. It visits
group keys in **sorted order** (compete.rs:330-331) for determinism, and picks
the winner as the highest `prelim_score`, ties broken by smallest row index
(compete.rs:336-344). The per-mode behaviour (`CompetitionMode`, config.rs:873):

- `WinnerTakeAll` (default): keep only `win`; everything else is a removal pair
  `(loser, winner)` (compete.rs:349-358). This is the arm that deletes the
  charge/mod siblings described above; the deleted rows never reach rescore.
- `None`: keep all members, remove nothing (compete.rs:346-348). FDR handles the
  ambiguity downstream.
- `FeaturesOnly`: identical retained set to `None` (same match arm,
  compete.rs:346); the distinct name documents intent for the experiment matrix,
  the idea being that conflict/contested features carry the interference signal
  into rescoring instead of removal.
- `UniqueEvidence`: keep the winner; keep a loser only when its unique-fragment
  evidence `>= unique_evidence_min_fragments` (compete.rs:359-372). Evidence comes
  from `unique_evidence_with_source` (compete.rs:279-298): prefers an explicit
  `unique_fragment_count` column, else approximates it as
  `n_matched_fragments * (1 - contested_frac)` (the multiplier `(1 - contested)`
  clamped to [0,1]), else raw `n_matched_fragments`, else `None`. The contested
  fraction is resolved by `prefer_peak_contested_fraction` (compete.rs:301-306),
  which prefers the Extended-feature `peak_contested_frac` column and falls back to
  the legacy `contested_frac` spelling. Each column is read through `col_f64`
  (compete.rs:309-315), which accepts either an f64 or an i32 encoding, so an
  integer-typed fragment count is handled without a schema mismatch. If the mode is
  selected but no column is available it warns and falls back to winner-take-all
  (compete.rs:133-139, 366 `.unwrap_or(false)`).
- `MarginGated`: keep the winner; remove a loser only when
  `prelim[win] - prelim[m] >= margin`, otherwise keep it (compete.rs:373-385).
  Conservative removal for the low-FDR region.

The returned `keep` indices are sorted and deduped (compete.rs:388-389). Kept
rows are projected into the output columns (compete.rs:151-185), and the schema
is forwarded. The report records `group_by`, `mode`, `input_rows`, `kept`,
`removed` (compete.rs:241-255). `removed` is the number to read when checking
whether a grouping choice is silently discarding evidence: under
`group_by = precursor` on a modification-rich library it reached 46.6% of the
input rows, and under `group_by = peptidoform_charge` on the same input it was 0.

### rescore: input concat and classifier dispatch

`rescore::run` (rescore.rs:41) concatenates all `--competed` tables into flat
vectors (rescore.rs:78-111). It records a per-PSM `source` = the index of the
input table the PSM came from (rescore.rs:65-70, 108); for a single-run rescore
this is all-zero. `source` is why the PIN and entrapment sidecars key on a unique
flat row index rather than `candidate_id`: `candidate_id` is the library index and
repeats across runs, so an experiment-wide table would collide on it
(rescore.rs:65-70, 761, 872-875, 978-986).

**Pooling costs no per-run FDR, so batch only to fit memory.** Each input table
keeps its own `source` stamp, and `run_psm_q` re-runs the whole target-decoy
analysis within each source (rescore.rs:411-437), so a run pooled with others
still receives a genuine per-run q. Pooling does not tighten the pooled `q_value`
either, because the estimator is scale-invariant (see
[fdr: the kernels](#fdr-the-kernels)). What does scale is cost: the dense feature
matrix is `n_psms * n_features * 4` bytes, and pooled rescoring measured
0.834 ms/PSM on the streaming backend, i.e. linear in PSM count. Splitting a
large experiment into several `rescore --competed` batches is therefore
statistically free; choose the batch size from available RAM, not from a
statistical argument.

The NnTorch worker picks its backend from the handoff file size against
`MUMDIA_NN_STREAM_GB` (default 4, `nn_rescore_worker.py:299-300`). A matrix
marginally over that threshold falls silently to the disk-backed streaming memmap,
which re-reads the matrix every iteration and is much slower than the in-memory
path. `rescore.handoff = parquet` is the cheaper lever here: it keeps the same
values in a much smaller file, so a pool that would have streamed can stay in
memory.

Immediately after the concat loop, `crate::fdr::validate_labels` (rescore.rs:112)
rejects
any label that is not exactly `"target"` or `"decoy"`. `is_decoy` is derived at
rescore.rs:113; entrapment status is derived separately from the protein string by
`classify_entrapment` (rescore.rs:114, 641). Its exact rule (rescore.rs:652-666): a
decoy is neither entrapment nor real; a non-decoy is `is_entrapment` when its
protein **contains** `entrapment_marker` **and** (no `entrapment_exclude` set, or
the protein does not contain it) **and** the protein matches none of
`entrapment_contaminant_markers`; every other non-decoy is a real target. When
`entrapment_marker` is `None` nothing is entrapment and every non-decoy is real.
The contaminant carve-out exists so genuine contaminants living inside the spike-in
proteome (keratins, albumin) are not mislabeled as false negatives.

The classifier is dispatched on `RescoreConfig::classifier` (rescore.rs:152-283).
The stage tracks `classifier_used`, `model_identity`, and `qmode` so the report
reflects the path actually taken rather than the requested one (rescore.rs:148-150).

**`RescorerKind::NativeTda`** (config.rs:104, the default): calls `native_scores`
-> `percolator_lite` (rescore.rs:221, 616-632). `percolator_lite`
(rescoring.rs:100) is a Percolator/Mokapot-style linear model:

- Fold assignment is `fold_key % folds` where `fold_key = base_peptide_id`
  (rescoring.rs:109), so every charge/mod variant of a peptide lands in the same
  fold and no peptide leaks between train and test. "CV folds by candidate
  hashing" is the modulo-of-the-id scheme; the id used is the base-peptide id.
- Folds are processed in parallel (`rayon`, rescoring.rs:114-115) but each fold is
  independent and scores only its disjoint test set, so the result is
  order-independent and deterministic.
- Each fold fits its **own** standardizer on its training rows only
  (`fit_standardizer`, rescoring.rs:14-40, 122), which is the leak-free choice:
  test-fold statistics never enter standardization. std < 1e-9 is clamped to 1.0.
- Semi-supervised loop (`num_iter` iterations, rescoring.rs:131-176): compute
  target-decoy q on the current train scores, take confident targets
  (`q <= train_fdr`) as positives and all decoys as negatives, fit an L2 logistic
  regression (`logreg_fit`, rescoring.rs:50-77; full-batch gradient descent,
  `l2=1e-3`, `epochs=200`, `lr=0.5`, weight[0]=bias), and re-score the train fold.
  If fewer than 10 confident targets exist, it falls back to the top-scoring half
  as positives (rescoring.rs:155-171).
- Final score for each test row is `score_row(w, std_row(...))` written back by
  original index (rescoring.rs:178-190). Weights start at zero and there is no
  RNG, so the whole path is deterministic.
- **The result vector is seeded with `init_score` (the `prelim_score`)**
  (rescoring.rs:185). Only rows a fold actually scores are overwritten, so any row
  a fold cannot cover retains its prelim score. A fold whose training or test set
  is empty produces nothing and is skipped (rescoring.rs:119-121); this happens
  when a `fold_key` bucket is empty, e.g. very few peptides or all peptides landing
  in one fold. `folds` is clamped to at least 1 (rescoring.rs:105) and `num_iter`
  to at least 1 (rescoring.rs:131). The clamp on `folds` is defensive only: the
  stage rejects `rescore.folds < 2` before it ever calls `percolator_lite`
  (rescore.rs:46-48). `fit_standardizer` divides by `idx.len().max(1)`
  (rescoring.rs:16) so an empty index does not divide by zero.

**`RescorerKind::Mokapot`** (config.rs:106): runs `mokapot_worker.py` through
`run_pin_sidecar` (rescore.rs:158-186, 913). On success it uses the returned
scores; on failure it either hard-errors (if `rescore.strict`) or warns and falls
back to `native_scores` (rescore.rs:177-185). Requires `rescore.python`. Mokapot
reads the PIN through `mokapot.read_pin()` and cannot consume Parquet, so
`rescore.handoff = parquet` warns and writes the tab-separated PIN instead
(rescore.rs:946-954).

**`RescorerKind::NnTorch`** (config.rs:107-113): runs `nn_rescore_worker.py`
through the same `run_pin_sidecar` contract (rescore.rs:187-213). A nonlinear
PyTorch MLP with the same CV-fold + iterative positive-reselection scheme. The
initial feature/sign and every positive set are selected from that fold's
training rows only (`nn_rescore_worker.py:556-589`); empty, single-class, or
zero-positive folds hard-error (`:559-565`, `:602-605`), so held-out labels do not
influence OOF scoring. The
worker receives the NN hyperparameters through environment variables
`MUMDIA_NN_FOLDS`, `MUMDIA_NN_ITERS`, `MUMDIA_NN_TRAIN_FDR`, set from
`cfg.folds/num_iter/train_fdr` (rescore.rs:1012-1014), so the report reflects the
values actually used. Same strict/fallback logic as Mokapot.

> **NnTorch is seeded but not bit-deterministic.** The worker seeds numpy and
> torch per ensemble seed (`nn_rescore_worker.py:707-708`, plus
> `torch.manual_seed` at `:493`) but floating-point training and numerical
> kernels mean runs are only approximately reproducible. It also has a
> **scaler leak**: standardization statistics are computed over the **full**
> feature matrix, not per training fold, on all three backends. The in-memory
> PIN backend uses global median/IQR (`nn_rescore_worker.py:406-409`); the
> in-memory Parquet backend a global mean/std (`:387-391`); the streaming memmap
> backend accumulates a global mean/std in one pass (`:455-460`). This differs
> from the leak-free native `percolator_lite`, which fits the scaler on the train
> fold only. Both facts are properties of the sidecar, not the Rust dispatch.

**`RescorerKind::Percolator`**: **not wired.** Normal config loading rejects this
classifier before execution (`Config::validate`, config.rs:1389-1392); the
defensive rescore arm still errors under strict mode or falls back only for a
manually constructed compatibility config. `rescore.percolator_bin` is a dead
field until an adapter exists.

**`RescorerKind::Entrapment`** (config.rs:116-122): a decoy-independent path for
spike-in entrapment experiments. It requires `entrapment_marker` and at least one
matching PSM, else it warns/errors and falls back to native (rescore.rs:222-239).
With `rescore.python` set it runs `run_entrapment_gbm` (rescore.rs:241, 739): a
gradient-boosted sidecar trained out-of-fold by base peptide, positives = real
targets, negatives = spike-in targets (rescore.rs:739-797). Without python it
uses the native linear rescorer but with `is_entrapment` (not `is_decoy`) as the
negative label (rescore.rs:268, 279). Any of these sets `qmode = Entrapment`, so
q-values are computed by `entrapment_q` instead of `target_decoy_q`.

### rescore: the top-K per-candidate collapse

Between scoring and q-value computation the stage keeps only the best-scoring row
per `(source, candidate_id)` (rescore.rs:292-346), so when top-K peak promotion is
enabled the rescorer, not the up-front apex pick, selects which chromatographic
peak represents a candidate, and the terminal q-null holds exactly one row per
candidate. Without this, promoting K peaks would inflate the decoy null K-fold.
Ties break on lower `peak_rank`, then lower row index. Decoys collapse by the same
rule, so target/decoy exchangeability is preserved. At
`extract.promote_top_peaks = 1` every candidate has one peak, `best.len() == n`,
and the block is a no-op. The surviving row's rank is written out as
`selected_peak_rank`.

### rescore: the multi-context q columns

After scoring, the stage computes q-values at several aggregation levels, each an
**independent** target-decoy (or entrapment) analysis run on the appropriate
best-per-group reduction. This is deliberate: q-values at different levels are not
derived from one another, they are separately calibrated nulls.

| output column | grouping | how computed | file:line |
|---|---|---|---|
| `q_value` | none (pooled PSM) | `target_decoy_q` / `entrapment_q` over all PSMs | rescore.rs:349-364 |
| `experiment_psm_q` | none (pooled PSM) | clone of `q_value` | rescore.rs:407 |
| `global_q_value` | none (pooled PSM) | clone of `q_value`, backward-compat alias | rescore.rs:406 |
| `run_psm_q` | by `source` | independent TDA within each run, scattered back by row index | rescore.rs:411-437 |
| `precursor_q` | `(peptidoform, charge)` | best PSM per precursor, TDA over that set | rescore.rs:440-457 |
| `peptide_q_value` | `base_peptide_id` | best PSM per base peptide, TDA over that set | rescore.rs:370-378 |
| `pg_q_value` | protein-accession-set string | best PSM per protein group, TDA over that set | rescore.rs:383-400 |

`precursor_q` groups on `(peptidoform, charge)` at this stage, but what reaches
this stage depends on `compete.group_by`. Under the default
`group_by = precursor`, compete has already collapsed every charge and
modification variant of a stripped peptide to one row, so the precursor grouping
sees roughly one form per peptide and `precursor_q` reports a base-peptide
population. It is a precursor-level unit only under
`compete.group_by = peptidoform_charge`.

The per-level reduction is `grouped_q` (rescore.rs:673): for each key, keep the
best-scoring member `(score, is_decoy, is_entrapment, is_real, row)` with an exact
score tie resolved in favour of the active null (decoy or entrapment) so input row
order cannot make the accepted set anti-conservative (rescore.rs:683-706), run the
chosen q kernel over the one-row-per-group set (rescore.rs:707-719), then assign
the group q **only to the winning row** of each group and give every losing sibling
q = 1.0 (rescore.rs:721-728). A lower-scoring charge/mod variant (possibly itself a
false target) must not inherit the winner's low q. Group counts dedup by key on the
winner, so counts are unchanged, but per-PSM peptide/pg/precursor q no longer
propagate to losers.

`run_psm_q` groups rows by `source` in a `BTreeMap` for deterministic iteration
(rescore.rs:411-416) and runs a full independent TDA within each run, so a per-run
report gets a genuine per-run FDR instead of the pooled value. For a single-run
rescore (`source` all-zero) it equals `q_value`. After an experiment-wide rescore
it is the only correct per-file PSM unit: the grouped columns are assigned to one
winning row per experiment-wide group, so counting a single run on
`peptide_q_value`, `precursor_q`, or `pg_q_value` undercounts it by roughly a
factor of the number of pooled runs.

**Peptide q and precursor q are not interchangeable.** Both are best-per-group
TDA, but the peptide grouping (`base_peptide_id`) is coarser than the precursor
grouping (`peptidoform+charge`), so it has fewer groups. The one part of the
relation that follows from the estimator is the floor: the best group at a level
can reach no lower than `1 / n_target_groups` (fdr.rs:38), and peptide grouping has
no more target groups than precursor grouping, so the peptide-level floor is at
least the precursor-level floor. Beyond that floor the ordering is an empirical
property of the score distribution, not a guarantee, and has not been measured
here. Reporting both lets a consumer pick the FDR granularity that matches its
claim (a peptide-level ID list vs a precursor-level one). Under the default
`compete.group_by = precursor` the two levels see almost the same one-row-per-
peptide population, so the distinction only becomes real under
`compete.group_by = peptidoform_charge`.

Interned dense u32 ids are used for the protein and precursor groupings
(rescore.rs:383-391, 440-448) purely as a performance optimization: interning the
accession-set string and the `(peptidoform, charge)` tuple to first-seen integers
avoids hashing and cloning hundreds of thousands of strings inside `grouped_q`.
The mapping is bijective, so grouping and the resulting q-values are unchanged.

`is_reported` (rescore.rs:461-464) selects which rows count toward the 1%
summaries: real targets in entrapment mode (spike-in excluded), all non-decoys
otherwise. The report records `target_psms_at_1pct`, `target_peptides_at_1pct`,
`target_protein_groups_at_1pct`, `target_precursors_at_1pct`, and in entrapment
mode the `entrapment_ratio` and the entrapment-leak count `entrapment_peptides_at_1pct`
(spike-in peptides passing the 1% gate, a running FDR-validity check,
rescore.rs:498-506, 551-557).

### fdr: the kernels

`target_decoy_q` (fdr.rs:7) is the no-pi0 estimator
`q = (n_decoys + 1) / max(1, n_targets)`, monotonized:

1. Sort record indices by descending score (fdr.rs:12-18).
2. Walk in score order, processing **tied-score blocks together** so every PSM in
   a block gets the same FDR regardless of its arbitrary within-tie order
   (fdr.rs:27-43). This is a determinism requirement
   (`docs/14_build_test_deploy_gotchas.md`): a target/decoy interleave inside one
   tie block must not change the q.
3. FDR at rank = `(td + 1) / max(1, tt)` where `td`, `tt` are cumulative decoy and
   target counts at that score (fdr.rs:38). The `+1` is the conservative
   finite-sample pseudocount; the bare `n_decoys/n_targets` is optimistic in the
   low-count regime.
4. Monotonize from worst-scoring to best so q is non-increasing with score
   (fdr.rs:45-51): `q[i] = min(fdr at all ranks worse-or-equal)`.

The best target with perfect separation gets `q = 1/n_targets`, not 0
(test at fdr.rs:149-165).

**Pool size is not an FDR lever.** `(td + 1) / max(1, tt)` is a ratio of
cumulative counts (fdr.rs:38), so replicating the population k times maps it to
`td/tt + 1/(k*tt)`: every rank's estimate is unchanged apart from the `+1`
pseudocount, whose relative weight shrinks as the pool grows. Pooling more runs
into one rescore therefore cannot make q more stringent; at most it relaxes the
estimate by that vanishing pseudocount term. Do not explain a change in per-run
identification counts by the number of runs pooled. The real causes are the
classifier being trained on a different population and, if the wrong column was
read, the q unit (see `run_psm_q` above).

`entrapment_q` (fdr.rs:64) is the empirical-null analog:
`FDR(t) = (ratio * n_entrap(>=t) + 1) / max(1, n_real(>=t))`. `ratio` =
`N_real_lib / N_entrap_lib` corrects for unequal library sizes; the `+1` is the
same pseudocount. Rows that are neither entrapment nor real (decoys) are ranked
but enter no count (fdr.rs:87-98). Same tied-block walk and worst-to-best
monotonization as `target_decoy_q`. Unlike in-silico decoys, the entrapment
population experiences the same chimeric DIA interference as real targets, so the
estimate is not optimistic (fdr.rs:54-63). Uses a stable sort so ties keep input
order (fdr.rs:75-80).

`count_targets_at_q` (fdr.rs:115): count of non-decoy records with `q <= threshold`.

`validate_labels` (fdr.rs:127): hard-error on any label other than `"target"` or
`"decoy"`. An unknown or malformed label must not silently count as a target
because the target-decoy null depends on exact labeling. Entrapment status is
derived from the protein accession, not the label, which is why only two label
values are valid here.

`ln_factorial` (fdr.rs:137): `ln(n!)` via summed logs, used where matched-fragment
counts feed a hyperscore-style term; `n` is small so the naive loop is fine.

## Key types and functions

| name | file:line | what it does |
|---|---|---|
| `compete::run` | compete.rs:28 | full compete stage: group, resolve, write competed table + optional audit |
| `compete::resolve_competition` | compete.rs:321 | pure per-mode within-group resolution; returns kept indices + removal pairs |
| `compete::unique_evidence_with_source` | compete.rs:279 | derive per-candidate unique-fragment evidence for `UniqueEvidence` mode, plus the column it came from |
| `compete::prefer_peak_contested_fraction` | compete.rs:301 | choose the Extended `peak_contested_frac`, else the legacy `contested_frac` |
| `compete::col_f64` | compete.rs:309 | read a numeric column as f64, accepting an f64 or i32 encoding |
| `CompeteParams` | compete.rs:21 | `features`, `out`, `cfg`, `config_hash` |
| `rescore::run` | rescore.rs:41 | full rescore stage: concat, dispatch, multi-context q, scored table |
| `rescore::validate_feature_schema` | rescore.rs:596 | reject a concat whose feature companions differ in id or ordered columns |
| `rescore::native_scores` | rescore.rs:616 | thin wrapper calling `percolator_lite` with the config knobs |
| `rescore::classify_entrapment` | rescore.rs:641 | per-PSM `(is_entrapment, is_real_target)` from the protein string |
| `rescore::grouped_q` | rescore.rs:673 | best-per-group reduction + level q, winner-only assignment |
| `rescore::run_entrapment_gbm` | rescore.rs:739 | write parquet, run entrapment GBM worker, read scores by row_id |
| `rescore::run_pin_sidecar` | rescore.rs:913 | write PIN, run Mokapot/NnTorch worker, read scores by row index |
| `rescore::align_sidecar_scores` | rescore.rs:1041 | validate exact/unique/finite sidecar coverage, map scores by row id |
| `QMode` | rescore.rs:23 | which null q is computed against: `Decoy` or `Entrapment` |
| `RescoreParams` | rescore.rs:30 | `competed`, `out`, `work_dir`, `script_dir`, `cfg`, `config_hash` |
| `percolator_lite` | rescoring.rs:100 | native semi-supervised L2 logreg rescorer with per-fold scaler |
| `RescoreInput` | rescoring.rs:87 | features, is_decoy, fold_key, init_score, folds, num_iter, train_fdr |
| `fit_standardizer` | rescoring.rs:14 | per-fold mean/std over training rows only (leak-free) |
| `logreg_fit` | rescoring.rs:50 | full-batch GD logistic regression with L2, weight[0]=bias |
| `std_row` | rescoring.rs:43 | standardize one feature row with a fold's mean/std |
| `score_row` | rescoring.rs:79 | linear discriminant `w[0] + sum(w[j+1]*r[j])` |
| `RejectionReason` | rejection.rs:19 | loss-ladder reason codes; compete emits `OUTCOMPETED_BY_{TARGET,DECOY}` |
| `target_decoy_q` | fdr.rs:7 | `(D+1)/T` monotonized tied-block q from `(score, is_decoy)` |
| `entrapment_q` | fdr.rs:64 | `(ratio*E+1)/R` monotonized tied-block entrapment q |
| `count_targets_at_q` | fdr.rs:115 | count non-decoy records at or below a q threshold |
| `validate_labels` | fdr.rs:127 | label whitelist; rejects anything but target/decoy |
| `ln_factorial` | fdr.rs:137 | `ln(n!)` via summed logs |

## Configuration

The fields below are the live ones this subsystem reads. `percolator_bin` is the
one dead field kept in the struct (the `percolator` classifier is unwired).

### `compete` (`CompeteConfig`, config.rs:825-851)

| field | default | effect |
|---|---|---|
| `group_by` | `precursor` | despite the name, groups by stripped peptide: all charge **and** modification variants of one `base_peptide_id` share a group, separately within target and decoy labels; `apex` also buckets by rounded apex RT; `peptidoform_charge` keeps each peptidoform+charge separate and is required for a PTM search |
| `apex_rt_tolerance_s` | `5.0` | RT bucket width (s) for `group_by = apex` (compete.rs:90) |
| `mode` | `winner_take_all` | within-group resolution (`CompetitionMode`, config.rs:873): `winner_take_all`, `none`, `features_only`, `unique_evidence`, `margin_gated` |
| `margin` | `0.0` | score margin required to remove a loser under `margin_gated` (compete.rs:379) |
| `unique_evidence_min_fragments` | `2` | min unique-fragment count for a loser to survive under `unique_evidence` (compete.rs:361-366) |
| `emit_competition_audit` | `false` | write `<out>.compete_audit.parquet` (compete.rs:193) |

All non-default `mode` values are part of the sensitivity program and are
default-off; the production chain uses `winner_take_all` and is byte-identical
unless a knob is set.

### `rescore` (`RescoreConfig`, config.rs:1131-1170)

| field | default | effect |
|---|---|---|
| `classifier` | `native_tda` | which rescorer (`RescorerKind`, config.rs:100): `native_tda`, `mokapot`, `nn_torch`, `percolator` (unwired, rejected by `validate`), `entrapment` |
| `folds` | `3` | CV folds: native `percolator_lite`, PIN sidecars via `MUMDIA_NN_FOLDS` env (rescore.rs:1012), entrapment GBM via positional arg (rescore.rs:786). Values below 2 are a hard error (rescore.rs:46-48) |
| `train_fdr` | `0.01` | q threshold for the confident-positive set in the semi-supervised loop |
| `num_iter` | `10` | semi-supervised iterations for the native rescorer |
| `python` | `None` | interpreter for the Mokapot/NnTorch/entrapment sidecars; required for those paths |
| `percolator_bin` | `None` | dead field until the `percolator` path is wired (config.rs:1140) |
| `entrapment_marker` | `None` | protein substring marking spike-in negatives; required for `entrapment` |
| `entrapment_exclude` | `None` | substring that, if also present, keeps a PSM as a real target (shared peptides) |
| `entrapment_contaminant_markers` | `[]` | substrings marking genuine contaminants inside the spike-in proteome; matching PSMs stay real targets |
| `entrapment_ratio` | `1.0` | `N_real_lib / N_entrap_lib`, scales the entrapment FDR estimate |
| `strict` | `true` | production default: any sidecar failure / misconfiguration is a hard error; false explicitly enables compatibility fallback |
| `handoff` | `tsv` | how the feature matrix reaches a sidecar (`Handoff`, config.rs:1230): `tsv` writes the Percolator PIN, `parquet` writes an f32 Parquet feature table. `parquet` applies to `nn_torch` only; a mokapot run warns and falls back to `tsv` (rescore.rs:946-954) |

## Invariants, determinism, gotchas

- **`group_by = precursor` is peptide-level, not precursor-level** (compete.rs:88).
  Its key is `base_peptide_id`, the stripped-sequence id, so `winner_take_all`
  deletes every lower-scoring charge and modification sibling before rescore. On a
  modification-rich imported library that removed 46.6% of extracted candidates,
  and it makes the downstream `precursor_q` a base-peptide unit. Use
  `peptidoform_charge` for any search where a modified form must be reported
  independently.
- **Label stays in the competition key** (compete.rs:71-101, config.rs:865-870). A
  target never directly eliminates its own decoy in the `compete` stage.
  `rescore` still reduces target and decoy representatives by the requested
  biological unit and compares those populations when estimating q values.
  Removing the stage-level label partition would prematurely deplete the null.
- **compete is deterministic.** Groups are visited in sorted key order and the
  winner tie-breaks to the smallest row index (compete.rs:330-344). Kept indices
  are sorted+deduped (compete.rs:388-389). No floats are summed across an unordered
  map.
- **Tied-score blocks share one q** in both `target_decoy_q` and `entrapment_q`
  (fdr.rs:27-43, 86-104). Within-tie order is arbitrary and must not change the q.
- **`native_tda` is fully deterministic**: zero-initialized weights, no RNG,
  per-fold work is order-independent even under rayon (rescoring.rs:114-183).
- **A row a fold cannot score keeps its `prelim_score`.** `percolator_lite` seeds
  its output with `init_score` and only overwrites rows a fold actually scores
  (rescoring.rs:185-190); an empty train/test fold is skipped (rescoring.rs:119-121).
  So the discriminant `score` is a mix of learned scores and, for uncovered rows,
  the raw prelim. This is the intended safe fallback, not a bug.
- **`nn_torch` is seeded but not bit-deterministic** and has a **scaler leak**
  (standardization fit over the full matrix, not per fold, on all three backends;
  `nn_rescore_worker.py:387-391,406-409,455-460`). Do not treat its scores as
  reproducible; do not use it where byte-identity is required.
- **`percolator` is unwired.** Config loading rejects it (config.rs:1389-1392).
  The defensive fallback arm matters only to manually constructed compatibility
  configs; `percolator_bin` remains a dead field.
- **Only the best peak per `(source, candidate_id)` reaches the q kernels**
  (rescore.rs:292-346). The collapse is a no-op at `extract.promote_top_peaks = 1`;
  with promotion on it is what stops K peaks per candidate from K-inflating the
  decoy null. The surviving rank is reported as `selected_peak_rank`.
- **Sidecars key on the flat row index, not `candidate_id`** (rescore.rs:65-70,
  761, 872-875, 978-986). `candidate_id` is the library index and repeats across
  runs; keying on it collides in an experiment-wide (multi-file) rescore. A missing
  row in the sidecar output is a hard error, not a worst-score fallback: coverage
  must be exact/unique/finite (`align_sidecar_scores`, rescore.rs:1041-1077,
  missing-row bail at rescore.rs:1073-1075).
- **Pooling runs is statistically free; batch for RAM only.** `source` is stamped
  per input table and `run_psm_q` re-runs TDA within each source
  (rescore.rs:65-70, 108, 411-437), so a pooled rescore still gives every run its
  own FDR, and `(D+1)/T` is scale-invariant apart from the pseudocount
  (fdr.rs:38). Cost, not statistics, sets the batch size: `n_psms * n_features * 4`
  bytes of feature matrix and a measured 0.834 ms/PSM on the streaming backend.
- **`global_q_value` and `experiment_psm_q` are exact clones of `q_value`**
  (rescore.rs:406-407). `global_q_value` is kept only for backward-compat.
- **Losing siblings get q = 1.0** at the peptide/precursor/pg levels
  (rescore.rs:721-728); do not read a loser's level q as its FDR. After an
  experiment-wide rescore the groups are experiment-wide, so per-run counts must
  use `run_psm_q`, not a grouped column.
- **Multi-context q-values are independent per level**, each a separate TDA on its
  own best-per-group reduction. They are not derived from `q_value` by
  aggregation.
- **`validate_labels` runs before any counting** (rescore.rs:112). Any label other
  than target/decoy aborts the stage.
- **Schema companion is mandatory.** compete writes `<out>.schema.json`
  (compete.rs:188); rescore reads the first input's schema (rescore.rs:76) and
  validates every later companion against it for identity and ordered feature
  columns (`validate_feature_schema`, rescore.rs:79-80, 596-613). A divergent
  feature schema hard-errors rather than silently misreading columns.

## How to extend / modify

- **Add a competition mode**: extend `CompetitionMode` (config.rs:873), add a match
  arm in `resolve_competition` (compete.rs:345-386), and add a unit test alongside
  the existing ones (compete.rs:393-489). Keep the label in the key and keep group
  visitation in sorted order so determinism holds.
- **Add a grouping**: extend `CompeteGroupBy` (config.rs:894) and add a key arm in
  compete.rs:87-99. If it needs a new column, guard for its presence as
  `PeptidoformCharge` does (compete.rs:52-55). State in the doc comment which
  biological unit the key actually collapses; the `Precursor` variant name already
  misleads on that point.
- **Wire the `percolator` path**: replace the warn/bail at rescore.rs:214-220 with a
  PIN round-trip, and drop the `validate` rejection at config.rs:1389-1392. The PIN
  writer already exists (`run_pin_sidecar`,
  rescore.rs:913); percolator consumes the same PIN, so the work is invoking
  `percolator_bin` (config.rs:1140) and parsing its output back into a per-row score
  vector aligned to input order. Honor `strict`.
- **Add a rescorer sidecar** that follows the PIN contract: reuse `run_pin_sidecar`
  (rescore.rs:913), which writes SpecId `psm_i` / ScanNr `i`, ExpMass=CalcMass=mz,
  the feature columns in schema order, and `-.<peptidoform>.-` / `<protein>`
  (rescore.rs:974-993). The worker must echo the SpecId tail as `candidate_id` and
  emit a `score` column; scores are mapped back by that row index through
  `align_sidecar_scores` (rescore.rs:1032-1035). If the new worker can read
  Parquet, gate it in `use_pq` (rescore.rs:946-947) rather than assuming the PIN.
  NN hyperparameters are passed as `MUMDIA_NN_*` env vars (rescore.rs:1012-1014).
- **Add a q-value context**: add a grouping vector and a `grouped_q` call
  (mirror `precursor_q` at rescore.rs:440-457), then add the output column at
  rescore.rs:508-541 and a 1% count if desired. Reuse the interning pattern for
  string keys to avoid hashing large columns.
- **Change the FDR estimator**: `target_decoy_q` (fdr.rs:7) and `entrapment_q`
  (fdr.rs:64) are the only two kernels; both `search-seed` and `rescore` call them,
  so a change here is global. Preserve the tied-block walk and the worst-to-best
  monotonization or determinism breaks. The `+1` pseudocount is intentional; do not
  drop it to chase counts.
- **Fix the NnTorch scaler leak**: fit standardization per training fold in
  `nn_rescore_worker.py` (mirror `fit_standardizer` in rescoring.rs:14-40) instead
  of over the full matrix at `:387-391` (in-memory Parquet), `:406-409` (in-memory
  PIN), and `:455-460` (streaming memmap). All three backends need the change.
