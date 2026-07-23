# compete, rescore, and FDR

> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

This subsystem is the tail of the identification chain (PLAN.md Stage F). It
turns the per-PSM feature table produced by the `features` stage into a scored,
FDR-controlled result set. It has three parts:

1. **compete** (`mumdia compete`): within each competition group, resolve
   redundant candidates for the same elution peak before target-decoy counting,
   so multiple plausible candidates for one peak cannot each be counted as a
   discovery. Default behaviour keeps only the best-scoring candidate per group.
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
| `scripts/nn_rescore_worker.py` | PyTorch MLP PIN sidecar (`RescorerKind::NnTorch`) |
| `scripts/entrapment_worker.py` | entrapment GBM sidecar (`RescorerKind::Entrapment`) |

## Inputs and outputs

### compete

Consumes the features artifact `psms_features` (path passed as `--features`),
plus its schema companion `<features>.schema.json` (read at compete.rs:39 via
`FeatureSchema::read`). Reads these columns (compete.rs:31-45): `candidate_id`
(u32), `label` (str), `base_peptide_id` (u32), `peptidoform` (str), `protein`
(str), `apex_rt` (f64), `precursor_mz` (f64), `prelim_score` (f64), `charge`
(f64, only needed for `peptidoform_charge` grouping), and every feature column
named in the schema.

Produces `psms_competed` (schema version 1, `artifact::PSMS_COMPETED`,
schema.rs:20) at `--out`. Column schema (compete.rs:118-130): `candidate_id`
(u32), `label` (str), `base_peptide_id` (u32), `peptidoform` (str), `protein`
(str), `apex_rt` (f64), `precursor_mz` (f64), `prelim_score` (f64), then every
feature column (f64) carried through unchanged. It also writes
`<out>.schema.json` (compete.rs:133) so rescore recovers the exact feature list,
and `<out>.report.json`.

Optional sidecar `<out>.compete_audit.parquet` (only when
`compete.emit_competition_audit = true`, compete.rs:138-180): one row per removed
candidate with `candidate_id` (u32), `label` (str), `peptidoform` (str),
`winner_candidate_id` (u32), `loser_prelim` (f64), `winner_prelim` (f64),
`rejection_reason` (str). The reason string is `RejectionReason::code()`
(rejection.rs:50), which is `SCREAMING_SNAKE_CASE`, so the two values written here
are `OUTCOMPETED_BY_DECOY` and `OUTCOMPETED_BY_TARGET` (rejection.rs:64-65), not
lowercase. The reason is keyed on the **loser's own label** (`reason_of`,
compete.rs:139-145): a removed decoy is `OUTCOMPETED_BY_DECOY`, a removed target is
`OUTCOMPETED_BY_TARGET`. Because competition is within-label, the winner shares the
loser's label, so this equivalently describes the winner. `RejectionReason`
(rejection.rs:19-45) has 17 variants spanning the whole loss ladder; compete only
ever emits these two.

### rescore

Consumes one or more `psms_competed` tables (`--competed`, a `Vec<String>`,
main.rs:154). The feature list is read from the schema companion of the first
input only (rescore.rs:64); all inputs are assumed to share that schema. Reads
`candidate_id`, `label`, `base_peptide_id`, `peptidoform`, `protein`, `charge`
(f64, cast to i32), `prelim_score`, `precursor_mz`, and the feature columns
(rescore.rs:66-87).

Produces `psms_scored` (schema version 2, `artifact::PSMS_SCORED`, schema.rs:21)
at `--out`. Column schema (rescore.rs:320-347): `candidate_id` (u32),
`peptidoform` (str), `charge` (i32), `label` (str), `protein` (str),
`base_peptide_id` (u32), `score` (f64, the classifier discriminant),
`q_value` (f64, pooled PSM q), `peptide_q_value` (f64), `protein_group` (str,
the protein-accession-set string, a duplicate of `protein`), `pg_q_value` (f64),
`global_q_value` (f64, byte-identical alias of `q_value`), `prelim_score` (f64),
`source` (u32, index into `--competed` identifying the run), `run_psm_q` (f64),
`experiment_psm_q` (f64, alias of `q_value`), `precursor_q` (f64). Plus
`<out>.report.json` whose `params` records `classifier` (the path actually taken),
`folds`, `num_iter`, `train_fdr` (rescore.rs:368), `model_identity` (e.g.
`native-percolator-lite-v1`, `mokapot`, `nn-torch-semisup-sidecar-v1`,
`entrapment-gbm-sidecar-v1`, `native-percolator-lite-entrapment-v1`,
rescore.rs:98-99, 370), and `stats` records `psms`, `classifier`,
`target_psms_at_1pct`, `target_peptides_at_1pct`, `target_protein_groups_at_1pct`,
`target_precursors_at_1pct`, plus `entrapment_ratio` and
`entrapment_peptides_at_1pct` in entrapment mode (rescore.rs:350-360).

Sidecar working files are written under `--work-dir` (created with
`create_dir_all`, rescore.rs:517, 580). Two distinct file contracts exist:

- **PIN sidecars** (`RescorerKind::Mokapot`, `RescorerKind::NnTorch`): input
  `rescore.pin` (Percolator tab format), output `rescore_sidecar_out.parquet`
  (rescore.rs:581-582). The PIN header is `SpecId  Label  ScanNr  ExpMass
  CalcMass  <features...>  Peptide  Proteins` (rescore.rs:585-587); each row is
  `psm_{i}`, `Label` = `-1` for a decoy else `+1` (rescore.rs:595), `ScanNr` = `i`
  (the flat row index), `ExpMass` = `CalcMass` = `precursor_mz` (5 decimals),
  the feature columns in schema order (6 decimals), `Peptide` = `-.<peptidoform>.-`,
  `Proteins` = `<protein>` (rescore.rs:594-601). The worker echoes the SpecId tail
  as a `candidate_id` column (here the row index `i`) and emits a `score` column;
  scores are mapped back by row index and a missing row gets the worst score
  (`min - 1.0`, rescore.rs:625-630).
- **Entrapment GBM** (`RescorerKind::Entrapment` with `rescore.python` set): input
  `entrapment_in.parquet`, output `entrapment_out.parquet` (rescore.rs:518-519).
  The input parquet columns are `row_id` (u32, the flat row index used for
  score readback), `candidate_id` (u32), `base_peptide_id` (u32), `is_entrapment`
  (i32, 0/1), `is_decoy` (i32, 0/1), then the feature columns in schema order
  (f64) (rescore.rs:521-534). The worker is invoked as
  `python entrapment_worker.py <in> <out> <folds>` (rescore.rs:537-543, `folds`
  from `cfg.folds`); it reads back `row_id` (u32) + `score` (f64), maps by
  `row_id`, and a missing row gets `min - 1.0` (rescore.rs:549-555). Both sidecars
  are run with `PYTHONUTF8=1` and a non-zero exit is a hard error
  (rescore.rs:544-545, 618-619).

## How it works

### compete: grouping

`compete::run` (compete.rs:28) builds a competition group key per PSM
(compete.rs:72-93). The key is a fixed-size tuple `(u32, u8, i64)` rather than a
freshly allocated String, chosen so grouping does not allocate per PSM:

- element 0 is `base_peptide_id` for `Precursor`/`Apex`, or a dense
  first-appearance `pform_id` for `PeptidoformCharge` (built at compete.rs:52-62);
- element 1 is the label code `0=target, 1=decoy, 2=other` (compete.rs:74-78);
- element 2 is a bucket: constant `0` for `Precursor`; the rounded apex-RT bucket
  `(apex_rt / apex_rt_tolerance_s).round()` for `Apex` (compete.rs:82); the
  rounded charge for `PeptidoformCharge` (compete.rs:88).

**The label is part of the key on purpose.** A target is never competed against
its own decoy; competition only arbitrates redundant charge/mod variants within
the target population and, separately, within the decoy population. If a target
could evict its paired decoy, the decoy population would be depleted and the
target-decoy null would collapse, badly underestimating FDR. This is stated at
compete.rs:64-71 and re-stated on `CompetitionMode` in config.rs:705-710.

`PeptidoformCharge` (config.rs:734-741) is the precursor-level grouping DIA-NN and
Spectronaut report at: sibling charges of one peptide are kept as separate groups
rather than collapsed. It requires the `charge` column and bails if absent
(compete.rs:46-48).

### compete: resolution

`resolve_competition` (compete.rs:247) is a pure, unit-tested function. It visits
group keys in **sorted order** (compete.rs:256-257) for determinism, and picks
the winner as the highest `prelim_score`, ties broken by smallest row index
(compete.rs:262-265). The per-mode behaviour (`CompetitionMode`, config.rs:711):

- `WinnerTakeAll` (default): keep only `win`; everything else is a removal pair
  `(loser, winner)` (compete.rs:270-273).
- `None`: keep all members, remove nothing (compete.rs:267-269). FDR handles the
  ambiguity downstream.
- `FeaturesOnly`: identical retained set to `None` (same match arm,
  compete.rs:267); the distinct name documents intent for the experiment matrix,
  the idea being that conflict/contested features carry the interference signal
  into rescoring instead of removal.
- `UniqueEvidence`: keep the winner; keep a loser only when its unique-fragment
  evidence `>= unique_evidence_min_fragments` (compete.rs:274-287). Evidence comes
  from `unique_evidence` (compete.rs:218-232): prefers an explicit
  `unique_fragment_count` column, else approximates it as
  `n_matched_fragments * (1 - contested_frac)` clamped to [0,1], else raw
  `n_matched_fragments`, else `None`. Each of those columns is read through
  `col_f64` (compete.rs:235-241), which accepts either an f64 or an i32 encoding,
  so an integer-typed fragment count is handled without a schema mismatch. If the
  mode is selected but no column is available it warns and falls back to
  winner-take-all (compete.rs:100-105, 281 `.unwrap_or(false)`).
- `MarginGated`: keep the winner; remove a loser only when
  `prelim[win] - prelim[m] >= margin`, otherwise keep it (compete.rs:288-300).
  Conservative removal for the low-FDR region.

The returned `keep` indices are sorted and deduped (compete.rs:303-304). Kept
rows are projected into the output columns (compete.rs:117-131), and the schema
is forwarded. The report records `group_by`, `mode`, `input_rows`, `kept`,
`removed` (compete.rs:184-197).

### rescore: input concat and classifier dispatch

`rescore::run` (rescore.rs:41) concatenates all `--competed` tables into flat
vectors (rescore.rs:65-88). It records a per-PSM `source` = the index of the
input file the PSM came from (rescore.rs:61, 85); for a single-run rescore this
is all-zero. `source` is why the PIN and entrapment sidecars key on a unique flat
row index rather than `candidate_id`: `candidate_id` is the library index and
repeats across runs, so an experiment-wide table would collide on it
(rescore.rs:56-60, 522-525, 588-593).

Immediately after loading, `crate::fdr::validate_labels` (rescore.rs:89) rejects
any label that is not exactly `"target"` or `"decoy"`. `is_decoy` is derived at
rescore.rs:90; entrapment status is derived separately from the protein string by
`classify_entrapment` (rescore.rs:91, 411). Its exact rule (rescore.rs:422-436): a
decoy is neither entrapment nor real; a non-decoy is `is_entrapment` when its
protein **contains** `entrapment_marker` **and** (no `entrapment_exclude` set, or
the protein does not contain it) **and** the protein matches none of
`entrapment_contaminant_markers`; every other non-decoy is a real target. When
`entrapment_marker` is `None` nothing is entrapment and every non-decoy is real.
The contaminant carve-out exists so genuine contaminants living inside the spike-in
proteome (keratins, albumin) are not mislabeled as false negatives.

The classifier is dispatched on `RescoreConfig::classifier` (rescore.rs:104-192).
The stage tracks `classifier_used`, `model_identity`, and `qmode` so the report
reflects the path actually taken rather than the requested one (rescore.rs:97-99).

**`RescorerKind::NativeTda`** (config.rs:131, the default): calls `native_scores`
-> `percolator_lite` (rescore.rs:142, 386-402). `percolator_lite`
(rescoring.rs:98) is a Percolator/Mokapot-style linear model:

- Fold assignment is `fold_key % folds` where `fold_key = base_peptide_id`
  (rescoring.rs:107), so every charge/mod variant of a peptide lands in the same
  fold and no peptide leaks between train and test. "CV folds by candidate
  hashing" is the modulo-of-the-id scheme; the id used is the base-peptide id.
- Folds are processed in parallel (`rayon`, rescoring.rs:112-113) but each fold is
  independent and scores only its disjoint test set, so the result is
  order-independent and deterministic.
- Each fold fits its **own** standardizer on its training rows only
  (`fit_standardizer`, rescoring.rs:14-40, 120), which is the leak-free choice:
  test-fold statistics never enter standardization. std < 1e-9 is clamped to 1.0.
- Semi-supervised loop (`num_iter` iterations, rescoring.rs:129-172): compute
  target-decoy q on the current train scores, take confident targets
  (`q <= train_fdr`) as positives and all decoys as negatives, fit an L2 logistic
  regression (`logreg_fit`, rescoring.rs:48-75; full-batch gradient descent,
  `l2=1e-3`, `epochs=200`, `lr=0.5`, weight[0]=bias), and re-score the train fold.
  If fewer than 10 confident targets exist, it falls back to the top-scoring half
  as positives (rescoring.rs:152-169).
- Final score for each test row is `score_row(w, std_row(...))` written back by
  original index (rescoring.rs:173-186). Weights start at zero and there is no
  RNG, so the whole path is deterministic.
- **The result vector is seeded with `init_score` (the `prelim_score`)**
  (rescoring.rs:181). Only rows a fold actually scores are overwritten, so any row
  a fold cannot cover retains its prelim score. A fold whose training or test set
  is empty produces nothing and is skipped (rescoring.rs:117-119); this happens
  when a `fold_key` bucket is empty, e.g. very few peptides or all peptides landing
  in one fold. `folds` is clamped to at least 1 (rescoring.rs:103) and `num_iter`
  to at least 1 (rescoring.rs:129), so degenerate config still runs one fold and
  one iteration. `fit_standardizer` divides by `idx.len().max(1)` (rescoring.rs:15)
  so an empty index does not divide by zero.

**`RescorerKind::Mokapot`** (config.rs:132): runs `mokapot_worker.py` through
`run_pin_sidecar` (rescore.rs:105-119, 563). On success it uses the returned
scores; on failure it either hard-errors (if `rescore.strict`) or warns and falls
back to `native_scores` (rescore.rs:112-118). Requires `rescore.python`.

**`RescorerKind::NnTorch`** (config.rs:134-140): runs `nn_rescore_worker.py`
through the same `run_pin_sidecar` contract (rescore.rs:120-134). A nonlinear
PyTorch MLP with the same CV-fold + iterative positive-reselection scheme. The
worker receives the NN hyperparameters through environment variables
`MUMDIA_NN_FOLDS`, `MUMDIA_NN_ITERS`, `MUMDIA_NN_TRAIN_FDR`, set from
`cfg.folds/num_iter/train_fdr` (rescore.rs:610-616), so the report reflects the
values actually used. Same strict/fallback logic as Mokapot.

> **NnTorch is nondeterministic.** The worker seeds torch/numpy
> (`nn_rescore_worker.py:205,258-259`) but PyTorch training on floats plus
> nondeterministic kernels means runs are not byte-identical. It also has a
> **scaler leak**: standardization statistics are computed over the **full**
> feature matrix, not per training fold. In-memory backend uses global
> median/IQR (`nn_rescore_worker.py:134-137`); streaming backend accumulates a
> global mean/std in one pass (`:159-170`). This differs from the leak-free
> native `percolator_lite`, which fits the scaler on the train fold only. Both
> facts are properties of the sidecar, not the Rust dispatch.

**`RescorerKind::Percolator`** (config.rs:141-142): **not wired.** The arm
either hard-errors under `rescore.strict` or warns and uses `native_tda`
(rescore.rs:135-141). `percolator.exe` is installed on the dev machine but no
adapter exists; `rescore.percolator_bin` (config.rs:960) is a dead config field
until it is.

**`RescorerKind::Entrapment`** (config.rs:143-149): a decoy-independent path for
spike-in entrapment experiments. It requires `entrapment_marker` and at least one
matching PSM, else it warns/errors and falls back to native (rescore.rs:144-160).
With `rescore.python` set it runs `run_entrapment_gbm` (rescore.rs:162, 503): a
gradient-boosted sidecar trained out-of-fold by base peptide, positives = real
targets, negatives = spike-in targets (rescore.rs:497-556). Without python it
uses the native linear rescorer but with `is_entrapment` (not `is_decoy`) as the
negative label (rescore.rs:178, 189). Any of these sets `qmode = Entrapment`, so
q-values are computed by `entrapment_q` instead of `target_decoy_q`.

### rescore: the multi-context q columns

After scoring, the stage computes q-values at several aggregation levels, each an
**independent** target-decoy (or entrapment) analysis run on the appropriate
best-per-group reduction. This is deliberate: q-values at different levels are not
derived from one another, they are separately calibrated nulls.

| output column | grouping | how computed | file:line |
|---|---|---|---|
| `q_value` | none (pooled PSM) | `target_decoy_q` / `entrapment_q` over all PSMs | rescore.rs:196-204 |
| `experiment_psm_q` | none (pooled PSM) | clone of `q_value` | rescore.rs:231 |
| `global_q_value` | none (pooled PSM) | clone of `q_value`, backward-compat alias | rescore.rs:230 |
| `run_psm_q` | by `source` | independent TDA within each run, scattered back by row index | rescore.rs:235-259 |
| `precursor_q` | `(peptidoform, charge)` | best PSM per precursor, TDA over that set | rescore.rs:262-271 |
| `peptide_q_value` | `base_peptide_id` | best PSM per base peptide, TDA over that set | rescore.rs:210 |
| `pg_q_value` | protein-accession-set string | best PSM per protein group, TDA over that set | rescore.rs:215-224 |

The per-level reduction is `grouped_q` (rescore.rs:443): for each key, keep the
best-scoring member `(score, is_decoy, is_entrapment, is_real)`
(rescore.rs:453-461), run the chosen q kernel over the one-row-per-group set
(rescore.rs:463-474), then assign the group q **only to the winning row** of each
group and give every losing sibling q = 1.0 (rescore.rs:481-494). A lower-scoring
charge/mod variant (possibly itself a false target) must not inherit the winner's
low q. Group counts dedup by key on the winner, so counts are unchanged, but
per-PSM peptide/pg/precursor q no longer propagate to losers.

`run_psm_q` groups rows by `source` in a `BTreeMap` for deterministic iteration
(rescore.rs:236-239) and runs a full independent TDA within each run, so a per-run
report gets a genuine per-run FDR instead of the pooled value. For a single-run
rescore (`source` all-zero) it equals `q_value`.

**Why peptide q >= precursor q at the same threshold.** Both are best-per-group
TDA, but the peptide grouping (`base_peptide_id`) is coarser than the precursor
grouping (`peptidoform+charge`): several precursors collapse into one peptide. A
coarser grouping has fewer, higher-scoring representatives and a different
target/decoy balance among the survivors, so the monotonized q at a given level
is generally not lower than at the finer level. Reporting both lets a consumer
pick the FDR granularity that matches its claim (a peptide-level ID list vs a
precursor-level one).

Interned dense u32 ids are used for the protein and precursor groupings
(rescore.rs:215-223, 262-270) purely as a performance optimization: interning the
accession-set string and the `(peptidoform, charge)` tuple to first-seen integers
avoids hashing and cloning hundreds of thousands of strings inside `grouped_q`.
The mapping is bijective, so grouping and the resulting q-values are unchanged.

`is_reported` (rescore.rs:275-278) selects which rows count toward the 1%
summaries: real targets in entrapment mode (spike-in excluded), all non-decoys
otherwise. The report records `target_psms_at_1pct`, `target_peptides_at_1pct`,
`target_protein_groups_at_1pct`, `target_precursors_at_1pct`, and in entrapment
mode the `entrapment_ratio` and the entrapment-leak count `entrapment_peptides_at_1pct`
(spike-in peptides passing the 1% gate, a running FDR-validity check,
rescore.rs:308-318, 350-360).

### fdr: the kernels

`target_decoy_q` (fdr.rs:7) is the no-pi0 estimator
`q = (n_decoys + 1) / max(1, n_targets)`, monotonized:

1. Sort record indices by descending score (fdr.rs:12-18).
2. Walk in score order, processing **tied-score blocks together** so every PSM in
   a block gets the same FDR regardless of its arbitrary within-tie order
   (fdr.rs:27-43). This is a determinism requirement (PLAN.md Section 7): a
   target/decoy interleave inside one tie block must not change the q.
3. FDR at rank = `(td + 1) / max(1, tt)` where `td`, `tt` are cumulative decoy and
   target counts at that score (fdr.rs:38). The `+1` is the conservative
   finite-sample pseudocount; the bare `n_decoys/n_targets` is optimistic in the
   low-count regime.
4. Monotonize from worst-scoring to best so q is non-increasing with score
   (fdr.rs:45-51): `q[i] = min(fdr at all ranks worse-or-equal)`.

The best target with perfect separation gets `q = 1/n_targets`, not 0
(test at fdr.rs:144-157).

`entrapment_q` (fdr.rs:64) is the empirical-null analog:
`FDR(t) = (ratio * n_entrap(>=t) + 1) / max(1, n_real(>=t))`. `ratio` =
`N_real_lib / N_entrap_lib` corrects for unequal library sizes; the `+1` is the
same pseudocount. Rows that are neither entrapment nor real (decoys) are ranked
but enter no count (fdr.rs:87-92). Same tied-block walk and worst-to-best
monotonization as `target_decoy_q`. Unlike in-silico decoys, the entrapment
population experiences the same chimeric DIA interference as real targets, so the
estimate is not optimistic (fdr.rs:54-63). Uses a stable sort so ties keep input
order (fdr.rs:70-75).

`count_targets_at_q` (fdr.rs:110): count of non-decoy records with `q <= threshold`.

`validate_labels` (fdr.rs:122): hard-error on any label other than `"target"` or
`"decoy"`. An unknown or malformed label must not silently count as a target
because the target-decoy null depends on exact labeling. Entrapment status is
derived from the protein accession, not the label, which is why only two label
values are valid here.

`ln_factorial` (fdr.rs:132): `ln(n!)` via summed logs, used where matched-fragment
counts feed a hyperscore-style term; `n` is small so the naive loop is fine.

## Key types and functions

| name | file:line | what it does |
|---|---|---|
| `compete::run` | compete.rs:28 | full compete stage: group, resolve, write competed table + optional audit |
| `compete::resolve_competition` | compete.rs:247 | pure per-mode within-group resolution; returns kept indices + removal pairs |
| `compete::unique_evidence` | compete.rs:218 | derive per-candidate unique-fragment evidence for `UniqueEvidence` mode |
| `compete::col_f64` | compete.rs:235 | read a numeric column as f64, accepting an f64 or i32 encoding |
| `CompeteParams` | compete.rs:21 | `features`, `out`, `cfg`, `config_hash` |
| `rescore::run` | rescore.rs:41 | full rescore stage: concat, dispatch, multi-context q, scored table |
| `rescore::native_scores` | rescore.rs:386 | thin wrapper calling `percolator_lite` with the config knobs |
| `rescore::classify_entrapment` | rescore.rs:411 | per-PSM `(is_entrapment, is_real_target)` from the protein string |
| `rescore::grouped_q` | rescore.rs:443 | best-per-group reduction + level q, winner-only assignment |
| `rescore::run_pin_sidecar` | rescore.rs:563 | write PIN, run Mokapot/NnTorch worker, read scores by row index |
| `rescore::run_entrapment_gbm` | rescore.rs:503 | write parquet, run entrapment GBM worker, read scores by row_id |
| `QMode` | rescore.rs:23 | which null q is computed against: `Decoy` or `Entrapment` |
| `RescoreParams` | rescore.rs:30 | `competed`, `out`, `work_dir`, `script_dir`, `cfg`, `config_hash` |
| `percolator_lite` | rescoring.rs:98 | native semi-supervised L2 logreg rescorer with per-fold scaler |
| `RescoreInput` | rescoring.rs:85 | features, is_decoy, fold_key, init_score, folds, num_iter, train_fdr |
| `fit_standardizer` | rescoring.rs:14 | per-fold mean/std over training rows only (leak-free) |
| `logreg_fit` | rescoring.rs:48 | full-batch GD logistic regression with L2, weight[0]=bias |
| `std_row` | rescoring.rs:42 | standardize one feature row with a fold's mean/std |
| `score_row` | rescoring.rs:77 | linear discriminant `w[0] + sum(w[j+1]*r[j])` |
| `RejectionReason` | rejection.rs:19 | loss-ladder reason codes; compete emits `OUTCOMPETED_BY_{TARGET,DECOY}` |
| `target_decoy_q` | fdr.rs:7 | `(D+1)/T` monotonized tied-block q from `(score, is_decoy)` |
| `entrapment_q` | fdr.rs:64 | `(ratio*E+1)/R` monotonized tied-block entrapment q |
| `count_targets_at_q` | fdr.rs:110 | count non-decoy records at or below a q threshold |
| `validate_labels` | fdr.rs:122 | label whitelist; rejects anything but target/decoy |
| `ln_factorial` | fdr.rs:132 | `ln(n!)` via summed logs |

## Configuration

The config was recently pruned of dead fields; the fields below are the live ones
this subsystem reads.

### `compete` (`CompeteConfig`, config.rs:668-703)

| field | default | effect |
|---|---|---|
| `group_by` | `precursor` | competition grouping (`CompeteGroupBy`, config.rs:734): `precursor` groups a peptide's charge/mod variants and its decoy; `apex` also buckets by rounded apex RT; `peptidoform_charge` keeps each peptidoform+charge separate |
| `apex_rt_tolerance_s` | `5.0` | RT bucket width (s) for `group_by = apex` (compete.rs:82) |
| `mode` | `winner_take_all` | within-group resolution (`CompetitionMode`, config.rs:713): `winner_take_all`, `none`, `features_only`, `unique_evidence`, `margin_gated` |
| `margin` | `0.0` | score margin required to remove a loser under `margin_gated` (compete.rs:294) |
| `unique_evidence_min_fragments` | `2` | min unique-fragment count for a loser to survive under `unique_evidence` (compete.rs:276) |
| `emit_competition_audit` | `false` | write `<out>.compete_audit.parquet` (compete.rs:138) |

All non-default `mode` values are part of the sensitivity program and are
default-off; the production chain uses `winner_take_all` and is byte-identical
unless a knob is set.

### `rescore` (`RescoreConfig`, config.rs:953-1002)

| field | default | effect |
|---|---|---|
| `classifier` | `native_tda` | which rescorer (`RescorerKind`, config.rs:128): `native_tda`, `mokapot`, `nn_torch`, `percolator` (unwired), `entrapment` |
| `folds` | `3` | CV folds: native `percolator_lite`, PIN sidecars via `MUMDIA_NN_FOLDS` env (rescore.rs:614), entrapment GBM via positional arg (rescore.rs:541) |
| `train_fdr` | `0.01` | q threshold for the confident-positive set in the semi-supervised loop |
| `num_iter` | `10` | semi-supervised iterations for the native rescorer |
| `python` | `None` | interpreter for the Mokapot/NnTorch/entrapment sidecars; required for those paths |
| `percolator_bin` | `None` | dead field until the `percolator` path is wired |
| `entrapment_marker` | `None` | protein substring marking spike-in negatives; required for `entrapment` |
| `entrapment_exclude` | `None` | substring that, if also present, keeps a PSM as a real target (shared peptides) |
| `entrapment_contaminant_markers` | `[]` | substrings marking genuine contaminants inside the spike-in proteome; matching PSMs stay real targets |
| `entrapment_ratio` | `1.0` | `N_real_lib / N_entrap_lib`, scales the entrapment FDR estimate |
| `strict` | `false` | when true, any sidecar failure / misconfiguration is a hard error instead of a native fallback |

## Invariants, determinism, gotchas

- **Label stays in the competition key** (compete.rs:64-93, config.rs:705-710). A
  target never competes against its own decoy. Removing this depletes decoys and
  underestimates FDR.
- **compete is deterministic.** Groups are visited in sorted key order and the
  winner tie-breaks to the smallest row index (compete.rs:256-265). Kept indices
  are sorted+deduped (compete.rs:303-304). No floats are summed across an unordered
  map.
- **Tied-score blocks share one q** in both `target_decoy_q` and `entrapment_q`
  (fdr.rs:27-43, 82-99). Within-tie order is arbitrary and must not change the q.
- **`native_tda` is fully deterministic**: zero-initialized weights, no RNG,
  per-fold work is order-independent even under rayon (rescoring.rs:112-186).
- **A row a fold cannot score keeps its `prelim_score`.** `percolator_lite` seeds
  its output with `init_score` and only overwrites rows a fold actually scores
  (rescoring.rs:181-186); an empty train/test fold is skipped (rescoring.rs:117-119).
  So the discriminant `score` is a mix of learned scores and, for uncovered rows,
  the raw prelim. This is the intended safe fallback, not a bug.
- **`nn_torch` is nondeterministic** and has a **scaler leak** (standardization
  fit over the full matrix, not per fold; `nn_rescore_worker.py:134-137,159-170`).
  Do not treat its scores as reproducible; do not use it where byte-identity is
  required.
- **`percolator` is unwired** (rescore.rs:135-141). It silently degrades to
  `native_tda` unless `strict = true`. `percolator_bin` is a dead field.
- **Sidecars key on the flat row index, not `candidate_id`** (rescore.rs:56-60,
  522-525, 588-593). `candidate_id` is the library index and repeats across runs;
  keying on it collides in an experiment-wide (multi-file) rescore. A missing row
  in the sidecar output gets the worst score (rescore.rs:553-555, 629-630).
- **`global_q_value` and `experiment_psm_q` are exact clones of `q_value`**
  (rescore.rs:230-231). `global_q_value` is kept only for backward-compat.
- **Losing siblings get q = 1.0** at the peptide/precursor/pg levels
  (rescore.rs:481-494); do not read a loser's level q as its FDR.
- **Multi-context q-values are independent per level**, each a separate TDA on its
  own best-per-group reduction. They are not derived from `q_value` by
  aggregation.
- **`validate_labels` runs before any counting** (rescore.rs:89). Any label other
  than target/decoy aborts the stage.
- **Schema companion is mandatory.** compete writes `<out>.schema.json`
  (compete.rs:133); rescore reads only the first input's schema (rescore.rs:64) and
  assumes all inputs share it. Concatenating tables with divergent feature schemas
  will misread columns.

## How to extend / modify

- **Add a competition mode**: extend `CompetitionMode` (config.rs:713), add a match
  arm in `resolve_competition` (compete.rs:266-301), and add a unit test alongside
  the existing ones (compete.rs:308-390). Keep the label in the key and keep group
  visitation in sorted order so determinism holds.
- **Add a grouping**: extend `CompeteGroupBy` (config.rs:734) and add a key arm in
  compete.rs:79-91. If it needs a new column, guard for its presence as
  `PeptidoformCharge` does (compete.rs:46-48).
- **Wire the `percolator` path**: replace the warn/bail at rescore.rs:135-141 with a
  PIN round-trip. The PIN writer already exists (`run_pin_sidecar`,
  rescore.rs:563); percolator consumes the same PIN, so the work is invoking
  `percolator_bin` (config.rs:960) and parsing its output back into a per-row score
  vector aligned to input order. Honor `strict`.
- **Add a rescorer sidecar** that follows the PIN contract: reuse `run_pin_sidecar`
  (rescore.rs:563), which writes SpecId `psm_i` / ScanNr `i`, ExpMass=CalcMass=mz,
  the feature columns in schema order, and `-.<peptidoform>.-` / `<protein>`
  (rescore.rs:585-601). The worker must echo the SpecId tail as `candidate_id` and
  emit a `score` column; scores are mapped back by that row index (rescore.rs:625-630).
  NN hyperparameters are passed as `MUMDIA_NN_*` env vars (rescore.rs:614-616).
- **Add a q-value context**: add a grouping vector and a `grouped_q` call
  (mirror `precursor_q` at rescore.rs:262-271), then add the output column at
  rescore.rs:320-347 and a 1% count if desired. Reuse the interning pattern for
  string keys to avoid hashing large columns.
- **Change the FDR estimator**: `target_decoy_q` (fdr.rs:7) and `entrapment_q`
  (fdr.rs:64) are the only two kernels; both `search-seed` and `rescore` call them,
  so a change here is global. Preserve the tied-block walk and the worst-to-best
  monotonization or determinism breaks. The `+1` pseudocount is intentional; do not
  drop it to chase counts.
- **Fix the NnTorch scaler leak**: fit standardization per training fold in
  `nn_rescore_worker.py` (mirror `fit_standardizer` in rescoring.rs:14-40) instead
  of over the full matrix at `:134-137` / `:159-170`.
