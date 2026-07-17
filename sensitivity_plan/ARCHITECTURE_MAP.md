# MuMDIA Architecture Map

Stage-by-stage map of the real MuMDIA pipeline for the sensitivity-improvement
effort. It records where each stage lives, its key symbols with line numbers, its
input/output artifacts, the config knobs that govern it, its tests, and, most
importantly, every point at which a candidate precursor can be dropped. All paths
are repository-relative; all line numbers are 1-indexed against the working tree
on branch `feat/sensitivity-improvements`.

Sources: the architecture-mapping workflow journal (7 stage maps: extract,
features, compete, rescore/FDR, configuration, IO+candidate-identity+calibration,
scripts) and direct reads of the source. The companion feature registry is
`FEATURE_REGISTRY.md` / `feature_registry.yaml`.

## 0. Pipeline overview

Each stage is an independent `mumdia <cmd>` subcommand reading path-addressable
inputs and writing Parquet plus an `<artifact>.report.json` sidecar. The single-run
chain (orchestrated by `run`, `stages/run.rs`) is:

```
convert -> digest -> peptidoforms -> predict-frag -> search-seed -> rt-im-train
        -> extract -> features -> compete -> rescore -> report   (+ quant)
```

`candidate_id` is the dense identity key of the library and of every artifact from
extract onward. It is minted in `predict_frag.rs:163` as the `enumerate()` index
after sorting by precursor m/z. Pre-predict-frag stages (digest, peptidoforms)
have no `candidate_id`; they key on `base_peptide_id` / peptidoform id.
`Library::load` (`index.rs:78`) hard-asserts `candidate_id == 0..ncand`
contiguous and (`index.rs:135`) precursor-m/z ascending, so `candidate_id` is a
safe dense array index downstream.

## 1. Modules added by the sensitivity lead agent

Committed on `feat/sensitivity-improvements`: `2f46d6d` (rejection reasons, top-K
peak enumerator, config scaffolding), `eb9da89` (candidate `audit` stage +
subcommand), `de5ae2b` (competition modes wired into `compete`).

Status of each addition:
- `mumdia audit` subcommand: WIRED and verified on real data (P0.3/P0.4).
- `CompetitionMode` in `compete`: WIRED (`de5ae2b`); default `WinnerTakeAll`
  reproduces the legacy behaviour bit-for-bit; `none`/`features_only`/
  `unique_evidence`/`margin_gated` are selectable and unit tested.
- `enumerate_peaks` (`mumdia::peaks`): a tested pure helper, NOT yet called from
  `extract` (extract still emits one apex per candidate). This is the one
  remaining destructive-stage change; the exact hook site is in §4.
- `ExtractConfig.retain_top_peaks` / `emit_candidate_audit`: parsed and validated,
  NOT yet consumed by `extract` (the top-K wiring and the in-extract audit sidecar
  are the primary next step; see `NEXT_STEPS.md`).

### `mumdia_core::rejection` (`rust/mumdia/crates/mumdia-core/src/rejection.rs`)

- `RejectionReason` enum (`rejection.rs:19`): 16 loss categories plus a `Reported`
  sentinel, `#[serde(rename_all = "SCREAMING_SNAKE_CASE")]`. The spellings match
  spec 01 §4 exactly (`NO_PEAK_GROUP`, `OUTCOMPETED_BY_DECOY`, ...).
- `code()` (`rejection.rs:50`) stable string for Parquet/JSON;
  `stage_order()` (`rejection.rs:76`) the identification-loss ladder (0 = earliest
  stage, `Reported` = 255); `earliest()` (`rejection.rs:106`) keeps the earlier of
  two losses; `is_rejection()` (`rejection.rs:100`).
- This is the type behind the audit table's `rejection_reason`; a candidate's row
  records the earliest stage at which it was lost.

### `mumdia::peaks` (`rust/mumdia/crates/mumdia/src/peaks.rs`)

- `PeakGroup` struct (`peaks.rs:22`): `apex_idx`, `start_idx`, `end_idx`,
  `apex_intensity`, `area`, `rank`.
- `enumerate_peaks(profile, k, bound_fraction, min_prominence_frac)`
  (`peaks.rs:52`): pure, side-effect-free top-K local-maximum detector over a
  consensus elution profile. Walks fractional-height boundaries (matching
  `features.bound_peak_fraction`), drops maxima below a prominence floor,
  deduplicates maxima inside a stronger peak's envelope, returns peaks strongest-
  first by integrated `area`, deterministic (ties break by earlier `apex_idx`).
  `k = 1` reproduces the single-strongest-apex behaviour, so callers can adopt it
  incrementally. Fully unit-tested (`peaks.rs:154-268`), including the core
  "interference-dominant but true peak retained with top-K" case
  (`peaks.rs:190`).

### `stages::audit` (`rust/mumdia/crates/mumdia/src/stages/audit.rs`, `mumdia audit`)

- Wired as a subcommand (`main.rs:230`, `stages/mod.rs:5`). Non-destructive,
  post-hoc: reconstructs per-candidate stage flags and the earliest
  `RejectionReason` by tracking which `candidate_id`s survive across the artifact
  chain library -> psms(extract) -> competed(compete) -> scored(rescore), then
  writes `candidate_audit.parquet` and prints the identification-loss waterfall
  (`audit.rs:64`, reason assignment `audit.rs:133-155`, waterfall `audit.rs:199`).
- `load_extract_reasons` (`audit.rs:51`) reads an optional
  `<psms>.audit.parquet` sidecar to refine the coarse extract-stage bucket once an
  in-extract audit is emitted.

### New config fields (`rust/mumdia/crates/mumdia-core/src/config.rs`)

- `ExtractConfig.retain_top_peaks: usize` (`config.rs:528`, default 1): K
  chromatographic peak groups per candidate; 1 = legacy single apex. Validated
  `>= 1` (`config.rs:917`).
- `ExtractConfig.emit_candidate_audit: bool` (`config.rs:533`, default false):
  when true, extraction is to write `<out-psms>.audit.parquet` per-candidate
  survivor flags / earliest reason. Near-zero cost when false.
- `CompeteConfig.mode: CompetitionMode` (`config.rs:613`, default `WinnerTakeAll`),
  `margin: f64` (`config.rs:616`), `unique_evidence_min_fragments: usize`
  (`config.rs:620`), `emit_competition_audit: bool` (`config.rs:623`).
- `CompetitionMode` enum (`config.rs:646`): `WinnerTakeAll` (legacy) / `None` /
  `FeaturesOnly` / `UniqueEvidence` / `MarginGated`, with `from_token`
  (`config.rs:667`). Maps to spec 04 §6 strategies A/B/C/D. CONSUMED in
  `compete.rs` via the pure `resolve_competition()` (`de5ae2b`); `WinnerTakeAll`
  is bit-identical to the previous behaviour.

## 2. Stage-by-stage map

For each stage: main source, key symbols (line), input -> output artifacts, the
config knobs that govern it, and existing tests.

### convert (Stage 0)

- Source: `stages/convert.rs`. Entry `run` (`convert.rs:103`); `centroid`
  (`convert.rs:19`), `ConvertParams` (`convert.rs:86`), `ConvertOutputs`
  (`convert.rs:96`).
- IO: mzML -> `spectra_ms1.parquet`, `spectra_ms2.parquet`,
  `isolation_windows.parquet`, `ms2_to_ms1.parquet` (`convert.rs:172-175`).
  Assigns a monotonic `scan_index`, converts scan time to seconds, centroids
  profile spectra (local maxima), caps peaks top-N, synthesizes a full-range
  window for zero-bounded AIF scans.
- Knobs: `max_spectra` (`convert.rs:89`/`124`) caps spectra for fast iteration
  (run-level, not a candidate drop). No per-candidate drops.
- Tests: none at stage level (CLAUDE.md test gap).

### digest (Stage A)

- Source: `stages/digest.rs`. Entry `run` (`digest.rs:147`); `make_decoy`
  (`digest.rs` ~95-117).
- IO: FASTA -> `peptides.parquet` (keyed on `id`/`target_id`). Fully-tryptic
  in-silico digest, mints paired reverse/scramble decoys immediately, dedups
  targets by stripped sequence (insertion order preserved for determinism).
- Knobs: `digest.enzyme` (`config.rs:261`), `missed_cleavages` (271), `min_len`
  / `max_len` (272), `decoy.strategy` (244, `Reverse` default; `DiannShift`/`None`
  produce no decoy and `DiannShift` is rejected by validate).
- Tests: `trypsin_p_cleaves_after_kr`, `reverse_decoy_keeps_cterm`,
  `scramble_is_deterministic`.

### peptidoforms (Stage A2)

- Source: `stages/peptidoforms.rs`. Entry `run` (`peptidoforms.rs:68`);
  `proforma` (`peptidoforms.rs:22`).
- IO: `peptides.parquet` -> `peptidoforms.parquet` (keyed on peptidoform `id` +
  `base_peptide_id`). Expands stripped peptides into peptidoforms with fixed +
  variable mods and charges, emits ProForma-lite. Known limitation: a second mod
  at the same position is dropped; no terminal mods.
- Knobs: `peptidoforms.fixed_mods` (`config.rs:283`), `variable_mods` (284),
  `max_variable_mods` (285), `charge_min` / `charge_max` (286-287),
  `unknown_modification` (289, `Error` default).
- Tests: `proforma_places_mods`, `combos_bounded`.

### predict-frag (Stage C)

- Source: `stages/predict_frag.rs`. Entry `run` (`predict_frag.rs:50`);
  candidate_id mint after precursor-m/z sort (`predict_frag.rs:137,163`).
- IO: `peptidoforms.parquet` -> `fragment_library_precursors.parquet` +
  `fragment_library_fragments.parquet` (the library). Computes precursor and b/y
  fragment m/z (shared mass model), intensities (native or MS2PIP), iRT (native
  or DeepLC), top-N, and builds the contiguous `candidate_id` the inverted index
  needs. This is the birth of candidate identity.
- Knobs: `predict_frag.predictor` (`config.rs:334`), `rt_predictor` (335),
  `top_n_fragments` (341), `charge2_from_precursor_charge` (340),
  `ms2pip_model`/`ms2pip_python`/`deeplc_python` (342-344).
- Tests: none at stage level.

### search-seed (Stage S)

- Source: `stages/search_seed.rs`. Entry `run` (`search_seed.rs:45`); masscal.json
  writer (`search_seed.rs:186`).
- IO: library + `spectra_ms2.parquet` -> `seed_psms.parquet` (best-per-candidate)
  + `<seed>.masscal.json` `{frag_ppm_offset, frag_tol_ppm, n_dev}`. Native
  Sage-lite hyperscore for calibration only (not a library filter). Drives per-run
  mass recalibration; fallback `{0.0, cfg.fragment_tol_ppm}` when `n_dev < 20`.
- Knobs: `search_seed.fdr_seed` (`config.rs:368`), `fragment_tol_ppm` (370),
  `min_matched_peaks` (373/374), `report_psms` (371), `matcher` (381).
  `precursor_tol_ppm` (369) is a dead knob (warns).
- Tests: none at stage level.

### rt-im-train (Stage B)

- Source: `stages/rt_im_train.rs`. Entry `run` (`rt_im_train.rs:28`); run_windows
  writer (`rt_im_train.rs:135`), cal.json writer (`rt_im_train.rs:149`).
- IO: `seed_psms.parquet` + library -> `run_windows.parquet`
  (`candidate_id, rt_pred_cal, rt_lo, rt_hi, im_*` (IM null)) + `<cal>.cal.json`
  `{method, slope, intercept, w_rt, p_rt, multiplier, n_train, calibration_status}`.
  Fits predicted_irt -> observed RT (linear always, LOESS when configured), sets a
  single global RT window half-width `w_rt` applied uniformly to every candidate
  (no per-candidate uncertainty). Optional DeepLC multitask fine-tune first.
- Knobs: `rt_im_train.calibration_method` (`config.rs:401`, `None` rejected),
  `q_train` (402), `p_rt` / `rt_window_multiplier` (404-405),
  `min_seed_for_calibration` (406), `loess_span` (408), `fallback_rt_window_s`
  (410), `finetune_deeplc` (416). `tolerance_regime` (400) dead (warns).
- Tests: none at stage level.

### extract (Stage D) - the core stage

- Source: `stages/extract.rs`, `index.rs`, `matchers/fragindex.rs`. Entry `run`
  (`extract.rs:222`); `Hit` (`extract.rs:80`), accumulator `acc`
  (`extract.rs:302`), `extract_accumulate_windows` (`extract.rs:128`), per-candidate
  parallel map `cand_hits`/`into_par_iter` (`extract.rs:566`), `CandOut`
  (`extract.rs:569`), apex selection loop (`extract.rs:718`), smoothed rolling
  count (`extract.rs:681`), signature ions (`extract.rs:710`), co-elution run
  (`extract.rs:736`), acquisition-scan grid (`extract.rs:632`), chrom emission
  (`extract.rs:851`). Index: `FragIndex` (`fragindex.rs:24`), `probe_peak`
  (`fragindex.rs:152`), `Library::candidate_range` (`index.rs:233`), `cand_frags`
  (`index.rs:208`), `page_search` (`index.rs:242`).
- IO: library + `run_windows.parquet` + spectra + masscal -> `psms_extracted.parquet`
  (20 cols, one apex PSM per surviving candidate, write `extract.rs:960`) +
  `chromatograms.parquet` (7 cols, `LargeListF32` traces, keyed by candidate_id,
  write `extract.rs:986`). Peak-major over the SoA inverted index; a cheap-to-
  expensive cascade (distinct-fragment presence -> co-elution run -> matched
  fraction -> Pearson gate) accepts candidates. Emits exactly one apex per
  candidate (`extract.rs:718`); no peak dimension exists yet.
- Knobs (all `ExtractConfig`, `config.rs:436-533`): `frag_tol_ppm` (440),
  `prec_tol_ppm` (441), `presence_min_matched` (443), `presence_min_fragments`
  (445), `presence_min_coelution` (447), `min_frag_corr` (452/453),
  `min_matched_fraction` (458), `min_coelution_run` (515), `fixed_scan_window`
  (439), `apex_top_fragments` (465), `apex_rt_prior_s` (469), `apex_count_tol`
  (474), `apex_count_window` (484), `emit_window_grid` (489), `peak_claim` (498),
  `emit_contested_features` (503), `peak_claim_margin` (507), `matcher` (509),
  `ms1_rescue` (521), plus the new `retain_top_peaks` (528) and
  `emit_candidate_audit` (533). Dead: `k_select` (491), `max_fragment_charge`
  (495), `scan_scale` (438), `ScanWindowMode::PeakWidthDerived`.
- Tests: none at stage level (exercised only via full `run`).

### features (Stage E)

- Source: `stages/features.rs` + `stages/features/*.rs`. Entry `run`
  (`features.rs:491`); `FAMILIES` (49), `active_features` (201),
  `feature_schema_id` (220), `FeatureSchema` (226), `Evidence` (269),
  `build_evidence` (336), `fragment_features` (915), `peak_bounds` (853),
  `prelim_score` (722), per-PSM parallel block (608), cross-charge block (573),
  `write_pin` (1216). See `FEATURE_REGISTRY.md` for the 17-family breakdown.
- IO: `psms_extracted.parquet` + `chromatograms.parquet` (+ `seed_psms.parquet`)
  -> `features.parquet` + `<features>.schema.json` + `run.pin`. Drop-free: one
  output row per input PSM; missing chromatograms yield default/zero vectors
  (see drop table). Emits `prelim_score` (bookkeeping, not a feature column).
- Knobs (`FeaturesConfig`, `config.rs:560`): `set` (561, `FeatureSet`),
  `coelution_corr_threshold` (562), `bound_features` (567), `bound_peak_fraction`
  (571). `prec_tol_ppm` (563) is declared but unused here. Several family
  thresholds are hardcoded consts, not config (`FRAG_TOL_PPM`
  `mass_accuracy.rs:44`, `MAXLAG` `coelution.rs:62`, `MIN_FRAGS/MIN_SCANS`
  `order_consistency.rs:38`).
- Tests: `feature_sets_sized` (`features.rs:1263`, asserts tier sizes),
  `peptide_length_ignores_mods`, `xcorr_aligned_traces`, and per-family tests in
  `chromatographic.rs` and `order_consistency.rs`. Most family modules
  (similarity, entropy, coelution, interference, mass_accuracy, ion_series, ms1,
  rt, novel, nonzero, peak_scans) have no unit tests.

### compete (Stage F, part 1)

- Source: `stages/compete.rs`. Entry `run` (`compete.rs:28`); `CompeteParams`
  (`compete.rs:20`), `pform_id` (51), winner map (72), `label_code` (74), key
  match (79), winner selection (92-95), `keep` (101).
- IO: `features.parquet` -> `psms_competed.parquet` (survivors + carried feature
  schema). Selects one winner per competition group by `prelim_score`. The label
  is part of the group key (`compete.rs:74-78`), so a target never competes
  against its own decoy (the decoy null is preserved). Runs BEFORE rescore
  (`run.rs:243` between features `run.rs:231` and rescore `run.rs:252`); the
  winner is chosen on `prelim_score`, not on any rescorer output.
- Knobs (`CompeteConfig`, `config.rs:601`): `group_by` (607, `CompeteGroupBy`
  Precursor/Apex/PeptidoformCharge), `apex_rt_tolerance_s` (608), plus the new
  scaffolded `mode`/`margin`/`unique_evidence_min_fragments`/`emit_competition_audit`
  (613-623).
- Tests: `features_compete_rescore_run_on_crafted_input`
  (`tests/pipeline.rs:166`, default Precursor grouping only).

### rescore (Stage F, part 2) + target-decoy FDR

- Source: `stages/rescore.rs`, `rescoring.rs`, `fdr.rs`. Entry `run`
  (`rescore.rs:41`); `percolator_lite` (`rescoring.rs:98`), `logreg_fit`
  (`rescoring.rs:48`), `fit_standardizer` (`rescoring.rs:14`), `RescoreInput.fold_key`
  (`rescoring.rs:90`), `grouped_q` (`rescore.rs:367`), `classify_entrapment`
  (`rescore.rs:335`), `QMode` (`rescore.rs:22`), `run_mokapot` (`rescore.rs:485`),
  `run_entrapment_gbm` (`rescore.rs:427`). FDR: `target_decoy_q` (`fdr.rs:7`,
  numerator `n_decoys + 1`, tie-block collapsed, monotonized), `entrapment_q`
  (`fdr.rs:63`).
- IO: `psms_competed.parquet` (concatenated across inputs) -> `psms_scored.parquet`
  with ALL rows incl decoys (`rescore.rs:251`, no drop). Attaches PSM / peptide
  (`base_peptide_id`) / protein-group / global q-values. Native `percolator_lite`
  folds by `base_peptide_id % folds` (no peptide leaks across folds);
  standardizer fit on train rows only.
- Knobs (`RescoreConfig`, `config.rs:723`): `classifier` (724, `RescorerKind`
  NativeTda/Mokapot/Percolator/Entrapment), `folds` (725), `train_fdr` (726),
  `num_iter` (728), `python` (729), `percolator_bin` (730),
  `entrapment_marker`/`entrapment_exclude`/`entrapment_contaminant_markers`/
  `entrapment_ratio` (734-748), `strict` (754).
- Tests: `perfect_separation_q_is_conservative_plus_one`, `tied_scores_share_one_q`,
  `entrapment_q_ranks_real_above_spike_in`, `separates_targets_from_decoys`. Gap:
  no test covers `rescore::run` itself, the sidecar branches, or `QMode::Entrapment`
  end to end.

### report

- Source: `stages/report.rs`. Entry `run` (`report.rs:49`); peptide gate
  (`report.rs:92`), peptide dedup (97), protein gate (118), protein dedup (121).
- IO: `psms_scored.parquet` -> `peptides.tsv` + `proteins.tsv`. This is where FDR
  gating actually removes rows from output (targets only, q <= `q_threshold`).
- Knobs: `q_threshold` (`ReportParams`, `report.rs:19`; `run` uses 0.01).
- Tests: `strip_mods_and_decoy` (`report.rs:143`).

### quant (Stage G, beyond-MVP) and quant-lfq

- Source: `stages/quant.rs`. Entry `run` (`quant.rs:147`), `run_lfq_combine`
  (`quant.rs:413`). Trapezoid XIC integration + top-N sum + protein-group rollup +
  per-fragment export; MaxLFQ/directLFQ cross-run via `quant-lfq`.
- Knobs (`QuantConfig`, `config.rs:682`): `q_threshold` (682), `top_n_fragments`
  (684), `top_n_peptides` (686), `rollup` (688), `bound_peak` (690),
  `peak_fraction` (694), `peak_grace` (698), `peak_window_mode` (700),
  `reliable_q` (703).
- `align` (Stage D2, `align.rs:52`) and MBR (Stage D3) are beyond-MVP / stub and
  not in the `run` chain.

## 3. Where candidates are dropped

Every removal or decision point that can lose a candidate precursor, grouped by
stage, collected from all stage maps. `Reason` is the matching
`RejectionReason::code`. Stages before predict-frag have no `candidate_id`, so
their drops are audited at `base_peptide_id` / peptidoform-id level.

### digest (Stage A)

| file:line | condition | effect | reason |
|---|---|---|---|
| `digest.rs:72` | `missed = j-i-1 > cfg.missed_cleavages` -> break | peptide spanning too many missed cleavages never enumerated | `PEPTIDE_NOT_GENERATED` |
| `digest.rs:77` | `len < min_len` or `len > max_len` -> continue | peptide outside length bounds dropped | `PEPTIDE_NOT_GENERATED` |
| `digest.rs:81` | non-standard residue (B/J/O/U/X/Z) -> continue | ambiguous-residue peptide dropped | `PEPTIDE_NOT_GENERATED` |
| `digest.rs:95` | `make_decoy: n < 3` -> None | no decoy minted for a <3-residue target | `PEPTIDE_NOT_GENERATED` |
| `digest.rs:117` | `DecoyStrategy::DiannShift`/`None` -> None | no sequence-rewrite decoy (DiannShift unrealized; rejected by validate) | `PEPTIDE_NOT_GENERATED` |
| `digest.rs:162` | target already seen (dedup by stripped seq) | duplicate peptide merged; collapses protein multiplicity | `PEPTIDE_NOT_GENERATED` |

### peptidoforms (Stage A2)

| file:line | condition | effect | reason |
|---|---|---|---|
| `peptidoforms.rs:80` | `unimod_mass(mod).is_none()` -> `bail!` | unknown fixed/variable mod aborts the run (hard error) | `MODIFICATION_NOT_ALLOWED` |
| `peptidoforms.rs:22` | second mod at the same residue position | silently keeps only the first mod (documented limitation) | `MODIFICATION_NOT_ALLOWED` |
| `peptidoforms.rs:121` | `z` outside `[charge_min, charge_max]` | charge states outside the range never enumerated | `CHARGE_OUT_OF_RANGE` |

### predict-frag (Stage C)

| file:line | condition | effect | reason |
|---|---|---|---|
| `predict_frag.rs:73` | `parse_peptidoform(pform)` is Err | ProForma parse failure dropped before candidate_id | `MODIFICATION_NOT_ALLOWED` |
| `predict_frag.rs:83` | `frags.is_empty()` | peptidoform with no b/y fragments dropped | `NO_VALID_FRAGMENTS` |
| `predict_frag.rs:134` | `retain(!frags.is_empty())` after top-N | candidate left with zero fragments after top-N truncation removed | `NO_VALID_FRAGMENTS` |

### search-seed (Stage S)

| file:line | condition | effect | reason |
|---|---|---|---|
| `search_seed.rs:81` | seed matched count < `min_matched_peaks` | seed PSM not emitted (calibration only); does NOT remove the candidate, it just lacks a `seed_score`/`seed_identified` feature | `NO_FRAGMENT_TRACES` (soft) |

### extract (Stage D)

The three accumulation paths (parallel window, serial non-two-pass, two-pass)
repeat the same three peak-level gates. `hi <= lo` (empty candidate range),
the RT gate, and empty claimants are per-peak skips that only cause a candidate to
be dropped if all of its collisions are skipped (the implicit non-materialization
at `extract.rs:302/554`). The `return None` gates are the explicit per-candidate
drops.

| file:line | condition | effect | reason |
|---|---|---|---|
| `extract.rs:302` / `:554` | candidate never inserted into `acc` (no in-window, in-RT fragment collision anywhere) | silent, irreversible non-materialization; never appears in `psms_extracted`. The single largest invisible drop class | `NO_FRAGMENT_TRACES` |
| `extract.rs:154` / `:327` / `:390` / `:433` | `hi <= lo`: isolation-window group maps to an empty candidate range | window contributes no hits to any candidate | `WRONG_ISOLATION_WINDOW` |
| `extract.rs:169` / `:341` / `:402` / `:446` | `rt < rt_lo[c]` or `rt > rt_hi[c]` | peak collision discarded for the candidate (outside calibrated RT window) | `RT_PRUNED` |
| `extract.rs:174` / `:348` / `:453` | `claimants.is_empty()` | peak matched no in-range/in-RT candidate; peak-level skip | `NO_FRAGMENT_TRACES` |
| `extract.rs:600` | `distinct.len() < presence_min_matched.max(1)` | first explicit per-candidate drop (tier-b presence) | `NO_FRAGMENT_TRACES` / `NO_VALID_FRAGMENTS` |
| `extract.rs:750` | `distinct.len() < presence_min_fragments` (acceptance sub-cond 1) | too few distinct matched fragments | `NO_VALID_FRAGMENTS` |
| `extract.rs:751` | `best_run < scan_window` (= `fixed_scan_window.max(1)`) | co-elution run shorter than the consecutive-scan floor | `NO_PEAK_GROUP` |
| `extract.rs:752` | `best_run < min_coelution_run` (default 0 = off) | transient (likely interferent) co-elution | `NO_PEAK_GROUP` |
| `extract.rs:753` | `matched_fraction < min_matched_fraction` (default 0 = off) | too small a fraction of predicted fragments observed | `NO_VALID_FRAGMENTS` |
| `extract.rs:808` | `pearson(obs_apex, pred) < min_frag_corr` and not MS1-rescued | apex fragment pattern disagrees with the predicted spectrum | `PEAK_NOT_SELECTED` / `NO_VALID_FRAGMENTS` |
| `extract.rs:718` | single-argmax apex; only the best scan-group emitted | secondary co-eluting peaks of the same candidate are never emitted (top-1 only). No trigger today; becomes reachable with top-K | `PEAK_NOT_SELECTED` (latent) |

### features (Stage E) - not drops, but silent zeroing

The features stage drops nothing (one row per input PSM), but several branches
turn a candidate into a maximally decoy-like all-zero vector with no signal that
the zero is undefined rather than measured. These are the "undefined-zero"
ambiguities the sensitivity work targets.

| file:line | condition | effect | reason (undefined-zero) |
|---|---|---|---|
| `features.rs:611` | `chrom.get(cid)` None/empty | `FragFeatures::default()`: all legacy fragment/coelution/mass/interference features 0.0 | `NO_FRAGMENT_TRACES` |
| `features.rs:644` | Extended active and no chromatogram | entire extended battery zeroed for the candidate | `NO_FRAGMENT_TRACES` |
| `features.rs:748` | active column has no `fmap` entry | column filled with 0.0 (masks a missing/misnamed feature) | `REMOVED_DURING_REPORTING` |
| `features.rs:80` | extended name collides with reserved / repeats | FEATURE-column drop (not a candidate drop): `spectral_angle`, `rt_error_abs`, `novel::peptide_length`, `novel::seed_identified` removed from the schema | `REMOVED_DURING_REPORTING` |
| `features.rs:862` | `peak_bounds` apex at zero height | apex relocated to global max; changes peak-bounded feature values | `PEAK_NOT_SELECTED` |
| family early returns | `entropy.rs:124` (k==0), `chromatographic.rs:72` (empty axis), `mass_accuracy.rs:139` (no ppm), `order_consistency.rs:122` (<3 frags/scans), `ms1.rs:242` (`ms1_xic.len() < 3`) | family returns all-zero vector; the last is dead until extract persists `ms1_xic` | `NO_PEAK_GROUP` / `NO_VALID_FRAGMENTS` |

### compete (Stage F, part 1)

| file:line | condition | effect | reason |
|---|---|---|---|
| `compete.rs:95` | same-key group, target loser: `prelim[i] <= prelim[w]` (ties keep first-seen) | lower-prelim charge/mod/localization sibling of a target removed before FDR, no audit row | `OUTCOMPETED_BY_TARGET` |
| `compete.rs:95` | same-key group, decoy loser (label in key) | lower-prelim decoy sibling removed; thins the decoy null symmetrically (keeps FDR trustworthy) | `OUTCOMPETED_BY_DECOY` |
| `compete.rs:82` | Apex mode: two peaks of same base+label round to the same `apex_rt` bucket | a genuinely distinct RT peak of the same peptide is dropped if it shares the bucket | `PEAK_NOT_SELECTED` |
| `compete.rs:46` | `PeptidoformCharge` mode and `charge` column absent | whole stage `bail`s (config/precondition error, not a per-candidate drop) | run failure |

### rescore (Stage F, part 2)

| file:line | condition | effect | reason |
|---|---|---|---|
| `rescore.rs:251` | none | writes every PSM incl decoys; drops nothing (all real drops deferred to report) | `REMOVED_DURING_REPORTING` (deferred) |
| `rescore.rs:415` | `grouped_q`: PSM is not the max-score row of its peptide / protein-group key | losing sibling's group q hard-set to 1.0 (silent demotion; will fail the report gate) | `OUTCOMPETED_BY_TARGET` / `FAILED_PEPTIDE_FDR` |
| `rescoring.rs:117` | degenerate fold (empty train_idx/test_idx) | those PSMs silently keep `init_score` (prelim) instead of a rescored value; scoring degradation, not a drop | (none) |

### report

| file:line | condition | effect | reason |
|---|---|---|---|
| `report.rs:92` | `label != "target"` or `peptide_q_value > q_threshold` | row excluded from `peptides.tsv` (decoys; targets failing peptide FDR) | `FAILED_PEPTIDE_FDR` |
| `report.rs:97` | `(peptidoform, charge)` already emitted | duplicate precursor dropped (best-q kept via sort) | `REMOVED_DURING_REPORTING` |
| `report.rs:118` | `label != "target"` or PG empty or `pg_q_value > q_threshold` | row excluded from `proteins.tsv` (no dedicated protein-FDR reason code exists) | `FAILED_PEPTIDE_FDR` / `REMOVED_DURING_REPORTING` |
| `report.rs:121` | `protein_group` already emitted | duplicate protein group dropped (best-q kept) | `REMOVED_DURING_REPORTING` |

### configuration (run-level gates, not per-candidate)

`Config::validate` (`config.rs:825`) blocks the whole run rather than dropping a
candidate: `DiannShift` decoy (`config.rs:827`) and `CalibrationMethod::None`
(`config.rs:836`) are hard errors; an unknown `--profile` (`config.rs:888`) aborts.
Seven dead knobs only warn (`search_seed.precursor_tol_ppm`,
`rt_im_train.tolerance_regime`, `extract.k_select`, `extract.max_fragment_charge`,
`extract.scan_scale`, `digest.decoy.source`, `digest.decoy.ratio`) and have no
runtime effect. `extract.k_select` (default 50, unimplemented) is the natural home
for a candidate-count cap (`CANDIDATE_CAP_REACHED`), which is not realized today.

## 4. Hooks for the sensitivity work

Exact file:line sites for the six work items, from the stage maps.

### 4.1 Candidate audit (rejection reasons + per-candidate ledger)

- Already available: `mumdia audit` post-hoc reconstruction (`audit.rs:64`), the
  `RejectionReason` type (`rejection.rs:19`), the `emit_candidate_audit` flag
  (`config.rs:533`), and the sidecar reader `load_extract_reasons` (`audit.rs:51`)
  that consumes a future `<psms>.audit.parquet`.
- In-extract emission (to write that sidecar): the never-materialized cohort is
  `union(candidate_range over all windows)` minus `acc.keys()` (`extract.rs:302`/
  `:554`); separate `RT_PRUNED` from `NO_FRAGMENT_TRACES` by tallying the RT gates
  (`extract.rs:169`/`:341`/`:402`/`:446`); change the per-candidate parallel map
  (`extract.rs:593-921`) to return `Accepted|Rejected{reason, evidence}` instead of
  `Option<CandOut>`, carrying the failing gate (`extract.rs:600`, `:750-756`,
  `:808`) and its numeric evidence.
- Pre-candidate-id drops (no candidate_id yet): audit at peptidoform-id level in
  `predict_frag.rs:103` (RowOut match) and `:134` (retain).
- Compete losers: complement of `keep` at `compete.rs:101` (winner
  `and_modify`/`or_insert` at `:92-98`); `emit_competition_audit`
  (`config.rs:623`) is the flag.
- Rescore/report flags: per-candidate q outcomes at `rescore.rs:220-238`; the
  terminal reported/rejected state at `report.rs:91-107`.
- IO: `mumdia-io/table.rs` needs a null-aware `opt_i32` reader for a nullable
  reason column (`Col::OptI32` exists at `table.rs:33`; only `opt_f64`/`opt_*`
  readers check `is_null` today), or use `Col::Str`/`Col::Bool`.

### 4.2 Top-K peaks (one apex/candidate today)

- Primary hook: the single-argmax apex loop `extract.rs:715-733`. Replace with a
  local-maxima detector over the `score`/`smoothed` series; `peaks::enumerate_peaks`
  (`peaks.rs:52`) is the ready-made pure function.
- Emission: `CandOut` (`extract.rs:569`), single-row append (`extract.rs:926-958`),
  `psms_extracted` writer (`extract.rs:960`), chrom writer keyed by candidate_id
  (`extract.rs:986`, key `:989`). Add a `peak_index`/`peak_rank` column; candidate_id
  ceases to be unique.
- Downstream: `features.rs:524-543` chrom grouping must key by `(candidate_id,
  peak)`; compete group key (`compete.rs:72`) and rescore grouping (`rescore.rs:194`
  peptide, `:208` protein) must include `peak_rank` or explicitly collapse peaks.
- Config: `extract.retain_top_peaks` (`config.rs:528`).

### 4.3 Fragment claimant / conflict graph

- Per-peak conflict set: `claimants` buffer (`extract.rs:304`; per-path at
  `:337-347`/`:398-408`/`:443-452`). Two-pass arbitration loop `extract.rs:456-511`
  already picks a winner and computes shares; record edges `(scan_rt, peak_mz,
  frag, winner_cid, {loser_cids}, shares)` there.
- Contested scalar already flows: `contested` map (`extract.rs:308`, `:479-485`,
  `:814-817`) -> `contested_frac` column -> `features.rs:509` (read) / `:711`
  (`peak_contested_frac` push). Extend with `n_claimants`, `unique_fragment_count`,
  competitor score/margin the same way.
- Two-pass is active only when `peak_claim` is a `Coelution*` variant or
  `emit_contested_features = true` (`extract.rs:311`). Config surface:
  `PeakClaim` (`config.rs:187`), `emit_contested_features` (`config.rs:503`),
  `peak_claim_margin` (`config.rs:507`).
- Feature side: add `Evidence` fields (`features.rs:269`) + a new family module
  following the `NAMES` + `values(&Evidence)` contract and append to `FAMILIES`
  (`features.rs:49`); dedup/schema/PIN flow automatically.

### 4.4 Competition modes

- Config scaffolding present: `CompeteConfig.mode/margin/unique_evidence_min_fragments/
  emit_competition_audit` (`config.rs:613-623`), `CompetitionMode` enum
  (`config.rs:646`, `from_token` `:667`), `CompeteGroupBy` (`config.rs:681`).
- Consume in the winner loop `compete.rs:73-101`: `None`/`FeaturesOnly` set
  `keep = 0..nrows` (no removal); `MarginGated` replaces the strict `prelim[i] >
  prelim[*w]` (`compete.rs:95`) with a margin test; `UniqueEvidence` keeps a loser
  that carries enough independent fragment evidence (needs the claimant graph and
  a `unique_fragment_count` feature). `group_by` stays orthogonal (equivalence
  class); `mode` is the removal policy. Not yet wired.

### 4.5 Feature registry

- `FAMILIES` (`features.rs:49`), `active_features` (`:201`), `extended_names`
  (`:93`), `reserved_names` (`:67`), `feature_schema_id` (`:220`, blake3),
  `FeatureSchema` (`:226`) persisted as `<artifact>.schema.json` (`:755`) and read
  back (`FeatureSchema::read`, `:233`).
- The schema is carried compete (`compete.rs:38`/`:120`) -> rescore
  (`rescore.rs:64`) so the classifier never runs under a mismatched set. New
  columns require no rescore change: rescore consumes every schema column
  uniformly (`rescore.rs:75`, PIN at `:507`).

### 4.6 Calibration

- Mass: `masscal.json` writer (`search_seed.rs:186`), consumed at
  `extract.rs:280-293` (offset applied to every probe m/z; learned tol sets the
  index build tolerance). Uncertainty is only the tolerance width + `n_dev`.
- RT: `cal.json` writer (`rt_im_train.rs:149`), `run_windows` writer
  (`rt_im_train.rs:135`). `w_rt` is a single global scalar applied uniformly
  (`rt_im_train.rs:124-133`); there is no per-candidate RT uncertainty.
- Hook for per-candidate uncertainty (spec 03 §8.1 / 01 §3.5): capture the full
  residual distribution into `cal.json` (`rt_im_train.rs:104-114`) and
  `masscal.json` (`search_seed.rs:176-185`); add a `pred_sigma` field to
  `Evidence` (`features.rs:279`) fed by a new library/chromatogram column.
- Config: `CalibrationMethod` (`config.rs:97`), `RtImTrainConfig` (`config.rs:399`),
  `finetune_deeplc` (`config.rs:416`).

## 5. Differences between the code and the sensitivity_plan docs

- Competition is one stage, not the spec's taxonomy. Spec 04 §2 asks for seven
  separate competition stages with separate logs (chromatographic peak, duplicate
  precursor, peptide interference, peptidoform, localization, target-decoy,
  reporting dedup). Reality: `compete.rs` does ONLY within-label winner-take-all
  variant collapse on `prelim_score`, controlled by `CompeteGroupBy`. Peptide/
  protein grouping and duplicate removal are folded into `grouped_q`
  (`rescore.rs:367`) and the `report` gates, not separate stages.
- Target-decoy "competition" is not a competition here. Spec 01/04 list
  target-decoy competition as a distinct stage. In the code the label is part of
  the compete key (`compete.rs:74-78`) precisely so targets and decoys never
  compete head-to-head; the target-decoy relationship is realized as q-values in
  `fdr::target_decoy_q` (`fdr.rs:7`), and the target-decoy q-value is the trusted
  FDR estimate. The spec's `passed_target_decoy_competition` stage flag therefore
  has no producing stage; `audit.rs` maps it to `q_value <= threshold` instead
  (`audit.rs:146-155`).
- Competition happens before rescoring, not after. Spec 04 §11 and §10 recommend
  competition after initial rescoring. In the code `compete` runs before `rescore`
  (`run.rs:243` before `:252`) and picks the winner on `prelim_score`, a heuristic
  (`features.rs:722`). A candidate a trained classifier would rank higher can be
  eliminated before the classifier ever sees it. This is the single largest
  in-stage sensitivity risk and the reason the `CompetitionMode::None`/`FeaturesOnly`
  modes now exist (`de5ae2b`): selecting them preserves every candidate so the
  rescorer, not `prelim_score`, arbitrates. Moving competition to AFTER an initial
  rescoring pass (spec 04 §11) remains future work.
- One apex per candidate. Spec 01 §3.1 hypothesizes that a single apex is chosen
  too early; confirmed: `extract.rs:718` emits exactly one apex PSM per candidate,
  no peak dimension. `retain_top_peaks` (`config.rs:528`) and `peaks.rs` are the
  answer but are not yet wired into extract.
- Registry metadata is thinner in code than in spec. Spec 03 §2 wants
  `requires_calibration`, `uses_cross_run_information`, `missing_value_policy`,
  and `computational_cost` per feature; the code registry (`FAMILIES`) stores only
  ordered names + a `values` fn. Direction and level are documented in
  `FEATURE_REGISTRY.md`/`feature_registry.yaml`, not in the code. `missing_value_policy`
  is de facto "fill with 0.0" everywhere (see the features drop table), which the
  spec's leakage/undefined-zero concern flags.
- Calibration is a global pre-CV fit. Spec 03 §3 requires calibration to be fit
  within each training fold. The code fits standardization within-fold
  (`rescoring.rs:120`) but mass recalibration, RT calibration, and the DeepLC
  iRT fine-tune are one-shot whole-run fits whose derived features enter every
  fold (see `FEATURE_REGISTRY.md` §4). This is a documented leakage path, not yet
  addressed.
- "Candidate" scope. In the spec a candidate spans the whole search space; in the
  code `candidate_id` exists only from `predict_frag` onward (minted at
  `predict_frag.rs:163`). Digest/peptidoforms losses (`PEPTIDE_NOT_GENERATED`,
  `MODIFICATION_NOT_ALLOWED`, `CHARGE_OUT_OF_RANGE`) precede candidate identity and
  must be audited at `base_peptide_id`/peptidoform-id level, as `audit.rs` notes.
