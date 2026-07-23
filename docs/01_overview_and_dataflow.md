# Overview and end-to-end dataflow

> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

MuMDIA is a clean-room Rust reimplementation of a data-independent-acquisition
(DIA) proteomics search engine. It takes an mzML run plus a spectral library
source and produces peptide and protein-group identifications at target-decoy
FDR, with optional quantification. This document describes the pipeline as a
graph: every stage, the artifacts each stage reads and writes, the two library
sources (FASTA digest versus imported DIA-NN library) and how the orchestrator
branches between them, the single-run `run` orchestration and its
`manifest.json`, and the current best-performing workflow.

The design principle is that every stage is an independent command over
path-addressable inputs and outputs on disk (plan.md Section 3.5). No stage
shares in-memory state with another; the orchestrator only threads file paths.
This makes any stage runnable standalone on prior outputs, on a different
configuration, or on hand-crafted minimal files. `run` is a convenience that
chains the stages in order and records provenance; it computes nothing itself.

## Files

| Path | Role |
|---|---|
| `rust/mumdia/crates/mumdia/src/main.rs` | CLI: one `clap` subcommand per stage (`Cmd` enum, `main.rs:16`); each arm loads config, hashes it, and calls the stage `run`. |
| `rust/mumdia/crates/mumdia/src/stages/run.rs` | `run` orchestrator: preflight, library-source branch, per-run chain, manifest assembly (`run.rs:83`). |
| `rust/mumdia/crates/mumdia/src/stages/mod.rs` | Re-exports every stage module. |
| `rust/mumdia/crates/mumdia/src/stages/*.rs` | The stage implementations (`convert`, `digest`, `peptidoforms`, `predict_frag`, `search_seed`, `rt_im_train`, `extract`, `features`, `compete`, `rescore`, `quant`, `report`, `align`, `audit`). |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | Frozen `(logical name, schema version)` for every artifact (`artifact` module, `schema.rs:6`). |
| `rust/mumdia/crates/mumdia-core/src/manifest.rs` | `Manifest` and `ArtifactRecord` (`manifest.rs:10`, `manifest.rs:22`). |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | The single typed serde `Config` (`config.rs:1010`) with per-stage sections and `apply_profile` (`config.rs:1098`). |
| `rust/mumdia/crates/mumdia-io/src/table.rs` | The `Col`/`Table` typed Parquet layer every stage writes through (`table.rs:23`). |
| `rust/mumdia/crates/mumdia-io/src/lib.rs` | `record_artifact`, `inspect`, `init_logging`, blake3 hashing. |

## Inputs and outputs

The pipeline consumes an mzML file plus one of two library sources, and produces
a fixed set of Parquet artifacts plus JSON sidecars. Every stage also writes a
small `<artifact>.report.json` next to each Parquet output (row counts, key
distributions, parameters, timing; plan.md Section 3.5), which is not listed
below per artifact but always exists.

External inputs:
- `--mzml <file>` (mzML; `.raw`/`.d` must be converted upstream by msconvert).
- FASTA mode: `--fasta <file>` (digested into the library).
- Library-input mode: `--lib-precursors <parquet>` + `--lib-fragments <parquet>`
  (a prebuilt library, for example an imported DIA-NN speclib).

Parquet artifacts and their column schemas as written in the code:

`spectra_ms1.parquet` (`convert.rs:180`): `scan_index` u32, `rt_seconds` f64,
`mz` list<f32>, `intensity` list<f32>.

`spectra_ms2.parquet` (`convert.rs:207`): `scan_index` u32, `id` str,
`rt_seconds` f64, `window_id` u32, `window_target` f64, `window_lower` f64,
`window_upper` f64, `precursor_mz` opt<f64>, `precursor_charge` opt<i32>,
`mz` list<f32>, `intensity` list<f32>.

`isolation_windows.parquet` (`convert.rs:224`): `window_id` u32, `target` f64,
`lower` f64, `upper` f64.

`ms2_to_ms1.parquet` (`convert.rs:234`): `ms2_scan_index` u32,
`ms1_scan_index` i32 (the preceding MS1 scan; -1 if none).

`peptides.parquet` (digest; FASTA mode only): fully-tryptic stripped peptides
plus their paired decoys.

`peptidoforms.parquet` (peptidoforms; FASTA mode only): concrete peptidoforms
with fixed/variable mods and charges as ProForma-lite strings.

`fragment_library_precursors.parquet` (`predict_frag.rs:189`): `candidate_id`
u32, `peptidoform_id` u32, `base_peptide_id` u32, `peptidoform` str, `charge`
i32, `precursor_mz` f64, `predicted_irt` f32, `label` str
(`target`/`decoy`), `protein` str, `n_fragments` i32. `candidate_id` is a
contiguous 0-based index sorted by precursor m/z (the fragment-index build
precondition).

`fragment_library_fragments.parquet` (`predict_frag.rs:204`): `candidate_id`
u32, `mz` f64, `predicted_intensity` f32, `name` str, `ion_type` str, `ordinal`
i32, `frag_charge` i32.

`seed_psms.parquet` (`search_seed.rs:220`): `candidate_id` u32, `peptidoform`
str, `charge` i32, `precursor_mz` f64, `base_peptide_id` u32, `protein` str,
`label` str, `score` f64 (hyperscore), `spectrum_q` f64 (per-spectrum
target-decoy q), `observed_rt` f64, `predicted_irt` f32, `matched_peaks` i32,
`scan_index` u32. Sidecar `seed_psms.parquet.masscal.json` carries the per-run
mass recalibration consumed by extract.

`run_windows.parquet` (`rt_im_train.rs:185`): `candidate_id` u32, `rt_pred_cal`
f64, `rt_lo` f64, `rt_hi` f64, `im_pred_cal`/`im_lo`/`im_hi` opt<f64> (IM columns
null in 3D). Sidecar `cal.json` records the fitted RT calibration.

`psms_extracted.parquet` (`extract.rs:1359`): `candidate_id` u32, `apex_rt` f64,
`apex_im` opt<f64>, `apex_intensity` f32, `n_matched_fragments` i32,
`n_predicted_fragments` i32, `coelution_run` i32, `rt_pred_cal` f64,
`precursor_mz` f64, `charge` i32, `label` str, `base_peptide_id` u32,
`peptidoform` str, `protein` str, `predicted_irt` f32, `contested_frac` f64,
`ms1_isom1`/`ms1_mono`/`ms1_iso1`/`ms1_iso2` opt<f64>. Optional columns:
`contested_count_frac`, `apportioned_frac` when `emit_contested_features`
(`extract.rs:1383`); `gate_apex`, `gate_peak_spectral`, `gate_coelution`,
`gate_spectral_entropy` when `emit_gate_diagnostics` (`extract.rs:1389`).

`chromatograms.parquet` (`extract.rs:1399`): `candidate_id` u32, `frag_name`
str, `frag_mz` f64, `frag_obs_mz` f64, `predicted_intensity` f32, `rt`
largelist<f32>, `intensity` largelist<f32>. `LargeList` (64-bit offsets) is
required because total list-value count can exceed the 32-bit `ListArray` limit
when extraction accepts a very large candidate set.

`<psms>.peaks.parquet` (`extract.rs:1419`, written only when
`extract.retain_top_peaks > 1`): `candidate_id` u32, `peak_rank` i32, `apex_rt`
f64, `start_rt` f64, `end_rt` f64, `evidence_count` f64, `area` f64. This is a
diagnostic sidecar not yet scored downstream (see finding 8, Gotchas).

`features.parquet` (`features.rs:828`): bookkeeping columns `candidate_id`,
`label`, `base_peptide_id`, `peptidoform`, `protein`, `apex_rt`, `elution_lo`,
`elution_hi`, `precursor_mz`, `prelim_score`, followed by one f64 column per
active feature name (`features.rs:839`). Sidecar `features.parquet.schema.json`
records the ordered feature-column list and its hashed `schema_id`; `run.pin` is
the Percolator input written deterministically alongside.

`psms_competed.parquet` (`compete.rs:118`): the same bookkeeping columns plus
every feature column carried through unchanged (`compete.rs:128`), for surviving
rows only. Sidecar `psms_competed.parquet.schema.json` carries the schema
forward for rescore.

`psms_scored.parquet` (schema version 2, `rescore.rs:323`): `candidate_id` u32,
`peptidoform` str, `charge` i32, `label` str, `protein` str, `base_peptide_id`
u32, `score` f64, `q_value` f64, `peptide_q_value` f64, `protein_group` str,
`pg_q_value` f64, `global_q_value` f64, `prelim_score` f64, `source` u32,
`run_psm_q` f64, `experiment_psm_q` f64, `precursor_q` f64. The q-value columns
are independent per level, not a rollup (see Gotchas).

`peptide_quant.parquet` (`quant.rs:317`): `candidate_id`, `peptidoform`,
`charge`, `protein_group`, `quantity` f64, `n_fragments_used` i32.
`protein_group_quant.parquet` (`quant.rs:344`): `protein_group`, `quantity`,
`n_peptides` i32. Optional `fragment_quant.parquet` (`quant.rs:372`) and
`peak_bounds` (`quant.rs:278`) diagnostics. `quant-lfq` emits a
protein-by-run matrix (`quant.rs:479`): `protein_group`, `run` i32, `quantity`,
`n_features` i32.

`candidate_audit.parquet` (`audit.rs:180`, written when
`extract.emit_candidate_audit` or via `mumdia audit`): `run_id` str,
`precursor_id` u32, `modified_sequence` str, `charge` i32,
`target_decoy_label` str, `entrapment_label` bool, then the boolean stage-flags
`candidate_generated`, `traces_extracted`, `peak_generated`, `peak_selected`,
`variant_selected`, `target_decoy_winner`, `passed_precursor_fdr`,
`passed_peptide_fdr`, `reported`, and `rejection_reason` str (earliest loss
reason).

Human-readable outputs: `peptides.tsv`, `proteins.tsv` (report), and
`manifest.json` (orchestrator).

## How it works

### The pipeline as a graph

```
                          FASTA mode                          library-input mode
                       (--fasta present)                 (--lib-precursors + --lib-fragments)
                              |                                        |
                  digest  peptides.parquet                            |
                              |                                        |
             peptidoforms  peptidoforms.parquet                       |
                              |                                        |
        predict-frag  fragment_library_precursors.parquet   (supplied directly, digest/
                      fragment_library_fragments.parquet      peptidoforms/predict-frag skipped)
                              \________________________________________/
                                               |
                                       lib_p (precursors)
                                       lib_f (fragments)
                                               |
   --mzml ---> convert ---> spectra_ms1.parquet, spectra_ms2.parquet,
                            isolation_windows.parquet, ms2_to_ms1.parquet
                                               |
        spectra_ms2 + lib_p + lib_f --> search-seed --> seed_psms.parquet
                                                        seed_psms.parquet.masscal.json
                                               |
        [optional] lib_p + seed --> deeplc_finetune --> fragment_library_precursors_ft.parquet
                                    (rewrites predicted_irt; lib_p := *_ft)
                                               |
        seed + lib_p --> rt-im-train --> run_windows.parquet, cal.json
                                               |
   spectra_ms2 + lib_p + lib_f + run_windows + spectra_ms1 + masscal
                        --> extract --> psms_extracted.parquet, chromatograms.parquet
                                        [<psms>.peaks.parquet if retain_top_peaks>1]
                                               |
        psms_extracted + chromatograms + seed --> features --> features.parquet,
                                                               features.parquet.schema.json, run.pin
                                               |
        features --> compete --> psms_competed.parquet (+ .schema.json)
                                               |
        psms_competed --> rescore --> psms_scored.parquet
                                               |
        [optional] lib_p + psms_extracted + competed + scored --> audit --> candidate_audit.parquet
                                               |
        psms_scored + chromatograms --> quant --> peptide_quant.parquet,
                                                  protein_group_quant.parquet, fragment_quant.parquet
                                               |
        psms_scored + quant --> report --> peptides.tsv, proteins.tsv
                                               |
                                    all artifacts recorded --> manifest.json
```

### The two library sources and how run.rs branches

The spectral library is the experiment-level, run-independent artifact pair
`(lib_p, lib_f)`. The orchestrator produces it in one of two ways, decided by a
`match` on `(p.lib_precursors, p.lib_fragments)` at `run.rs:96`:

- Library-input mode (`(Some(lp), Some(lf))`, `run.rs:97`): the supplied library
  is consumed directly. `digest`, `peptidoforms`, and `predict-frag` are skipped
  and the FASTA is never read. The orchestrator only reads the two Parquet files
  to record their row counts in the manifest, then uses their paths downstream
  (`run.rs:105-109`). This is the highest-sensitivity path, used with an imported
  DIA-NN speclib that already carries fragment intensities and iRT.

- FASTA-digest mode (the `_ =>` arm, `run.rs:111`): the library is built from the
  FASTA. `digest::run` writes `peptides.parquet` (`run.rs:116`),
  `peptidoforms::run` writes `peptidoforms.parquet` (`run.rs:126`), and
  `predict_frag::run` writes both library Parquet files (`run.rs:136`). The
  returned `(lib_p, lib_f)` paths feed the rest of the chain. This path has zero
  external runtime dependencies (native predictors).

`preflight` (`run.rs:37`) validates the mode before any compute: exactly one of
the two source configurations must be present, and every required input path must
exist (`run.rs:42-62`). It also checks sidecar prerequisites: `finetune_deeplc`
requires `predict_frag.deeplc_python` (`run.rs:63`); `rescore.classifier=mokapot`
requires `rescore.python`; `rescore.classifier=entrapment` requires
`rescore.entrapment_marker` (`run.rs:69-79`).

### The per-run chain

After the library is resolved, the orchestrator runs the per-run stages in order,
recording each output in the manifest with `record_artifact` (`run.rs:83-343`):

1. `convert::run` (`run.rs:152`) reads the mzML through mzdata, centroids profile
   spectra, caps peaks, synthesizes a full-range window for AIF/all-ion scans,
   and writes the four spectra artifacts. It returns a `ConvertOutputs` struct of
   paths (`convert.rs:96`).

2. `search_seed::run` (`run.rs:171`) runs a native Sage-lite broad DIA search over
   `spectra_ms2` against the fragment index for calibration (not final ID). It
   writes `seed_psms.parquet` and the mass recalibration `masscal.json`. The seed
   is iRT-independent, so it is computed once here on the base library and reused.

3. Optional DeepLC multitask fine-tune (`run.rs:187`, gated on
   `rt_im_train.finetune_deeplc`): `sidecar::run_deeplc_finetune` adapts the RT
   model to this run's confident seed PSMs and rewrites `predicted_irt` into
   `fragment_library_precursors_ft.parquet`. The `lib_p` binding is then rebound
   to the fine-tuned file so rt-im-train and extract read it. `lib_f` is
   unchanged (fine-tune touches iRT only). This step is nondeterministic (no fixed
   torch/numpy seed).

4. `rt_im_train::run` (`run.rs:212`) maps the run-independent iRT onto this run's
   observed RT (LOESS or linear) and sets per-candidate RT windows from the
   residual percentile, writing `run_windows.parquet` and `cal.json`.

5. `extract::run` (`run.rs:224`) is the core stage: a peak-major cascade over the
   inverted fragment index. It reads `spectra_ms2`, both library files,
   `run_windows`, `spectra_ms1` (for MS1 isotope XICs), and the `masscal.json`; it
   writes `psms_extracted.parquet` and `chromatograms.parquet`. `run` passes
   `restrict_candidates: None` (no allowlist).

6. `features::run` (`run.rs:242`) computes the config-selected feature vector plus
   `prelim_score` per PSM and emits `features.parquet`, its schema sidecar, and
   `run.pin`. It also reads `seed` for search-engine corroboration features.

7. `compete::run` (`run.rs:254`) keeps the best candidate per competition group
   before FDR counting, writing `psms_competed.parquet` and carrying the feature
   schema forward.

8. `rescore::run` (`run.rs:263`) does semi-supervised rescoring and native
   target-decoy q-values at multiple contexts, writing `psms_scored.parquet`. In
   `run` it is passed a single competed table (`&[competed.clone()]`), so this is
   single-run rescoring; the standalone `rescore` command accepts several tables
   for experiment-wide scoring.

9. Optional `audit::run` (`run.rs:276`, gated on `extract.emit_candidate_audit`)
   reconstructs the per-candidate identification-loss ladder into
   `candidate_audit.parquet`. It is a cheap join over the artifact chain and runs
   no extraction.

10. `quant::run` (`run.rs:293`) integrates fragment chromatograms and rolls up to
    protein groups, writing `peptide_quant`, `protein_group_quant`, and
    `fragment_quant`.

11. `report::run` (`run.rs:309`) writes `peptides.tsv` and `proteins.tsv` from the
    scored table plus quant, thresholded at `quant.q_threshold`, and prints a
    stdout summary naming the actual rescorer used.

### The manifest

`Manifest::new` (`manifest.rs:34`) is seeded with the fully-resolved canonical
config JSON and its blake3 hash (`run.rs:87,91`). Every stage output is recorded
by `record_artifact` (`mumdia-io`), which builds an `ArtifactRecord`
(`manifest.rs:10`) holding the logical name, path, format, schema name and
version, row count, content hash, producing stage, and config hash. Model
identities (RT predictor, fragment predictor, rescorer, and the feature schema
id) are recorded at `run.rs:323-333`. The manifest is written last to
`<out_dir>/manifest.json` (`run.rs:334`) with `mumdia_io::json::write_json`. The
manifest is provenance recorded, not required: because inputs are
path-addressable, no stage depends on it to run (plan.md Section 3.5).

### Current best workflow

The validated best-performing configuration (plan.md preamble "Best workflow",
finding 1) is library-input mode with:
- an imported DIA-NN library (fragment intensities + iRT),
- per-run DeepLC multitask fine-tune of iRT (`rt_im_train.finetune_deeplc = true`
  + `predict_frag.deeplc_python`), which is essential in library mode because raw
  DIA-NN iRT is locally noisy (~110 s MAD) and the fine-tune tightens it to
  ~13-27 s (finding 5),
- the Extended feature set (`features.set = extended`, applied by `--profile dia`,
  `config.rs:1101`),
- the `nn_torch` PyTorch MLP rescorer (`rescore.classifier = nn_torch` +
  `rescore.python`), which is the dominant sensitivity lever and beats the native
  linear rescorer by ~8.5% (finding 1),
- a loose extraction gate (`extract.min_frag_corr ~= 0.2`, now the default), since
  a strong nonlinear rescorer prefers recall and absorbs the loose-gate flood
  (finding 2),
- the MS2 peak cap kept at 300 on AIF (finding 4).

On `LFQ_Orbitrap_AIF_Ecoli_01.mzML` this reaches ~10,300 peptides at 1% FDR. The
zero-dependency native FASTA-digest mode (~1,213 peptides) is the high-precision
fallback. This preset is invoked through `docker/config.diann-lib.json` plus
`--profile dia`.

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `Cmd` (enum) | `main.rs:16` | One clap subcommand per stage; `Run` at `main.rs:195`. |
| `main` | `main.rs:373` | Parses the CLI, loads config, dispatches to the stage `run`. |
| `load_config` | `main.rs:363` | Reads a config JSON or returns `Config::default()`. |
| `doctor` | `main.rs:311` | Probes configured sidecar interpreters for required packages. |
| `RunParams` | `run.rs:17` | The orchestrator's inputs (config, fasta, mzml, out_dir, lib paths, caps). |
| `preflight` | `run.rs:37` | Validates mode, input existence, and sidecar prerequisites before compute. |
| `stages::run::run` | `run.rs:83` | The orchestrator: library branch, per-run chain, manifest write. |
| `Manifest` / `ArtifactRecord` | `manifest.rs:22` / `manifest.rs:10` | Provenance record for a chained run. |
| `artifact` module | `schema.rs:6` | Frozen `(logical name, schema version)` constants. |
| `Config::apply_profile` | `config.rs:1098` | Applies the `dia` preset (Extended features, apex window 5, RT prior 120 s). |
| `Config::canonical_json` | `config.rs:1116` | Canonical JSON for hashing into the manifest. |
| `Col` (enum) | `table.rs:23` | Typed column for writing Parquet; `LargeListF32` for chromatograms. |
| `Table::read` + typed getters | `table.rs:204` | Reads a Parquet artifact and extracts typed columns. |

## Configuration

The orchestrator reads a single typed `Config` (`config.rs:1010`) with per-stage
sections. Every field has `#[serde(default)]` and the struct uses
`deny_unknown_fields`, so a partial JSON overrides only named fields and an
unknown key is a hard error. `--profile dia` is applied on top of the loaded
config (`main.rs:581`) before `run`. The fields this overview area touches:

| Field | Default | Effect |
|---|---|---|
| `rng_seed` | `0` | Seeds decoy generation and rescorer folds (determinism). |
| `digest.decoy.strategy` | `reverse` | Decoy scheme; `diann_shift` and `none` are rejected/no-op. |
| `peptidoforms.charge_min` / `charge_max` | `2` / `3` | Precursor charge range enumerated in FASTA mode. |
| `predict_frag.rt_predictor` | `native` | `native` or `deeplc` iRT source (recorded in manifest). |
| `predict_frag.predictor` | `native` | `native` or `ms2pip` fragment intensities. |
| `predict_frag.deeplc_python` | `None` | Interpreter for DeepLC; required if `finetune_deeplc`. |
| `predict_frag.sidecar_script_dir` | `"scripts"` | Where sidecar workers are resolved from. |
| `search_seed.top_n_peaks` | `300` | Seed-only MS2 peak cap (keep 300 on AIF, finding 4). |
| `rt_im_train.calibration_method` | `loess` | RT calibration; `none` is rejected by validation. |
| `rt_im_train.finetune_deeplc` | `false` | Enables the per-run DeepLC fine-tune of iRT (best workflow). |
| `rt_im_train.finetune_batch` | `0` | `0` auto-scales the fine-tune batch to seed size. |
| `extract.min_frag_corr` | `0.2` | Spectral-agreement gate threshold; loose default suits `nn_torch`. |
| `extract.gate_mode` | `apex_pearson` | Which spectral score the gate thresholds (`GateMode`, `config.rs:591`). |
| `extract.retain_top_peaks` | `1` | `>1` writes the `<psms>.peaks.parquet` sidecar (not yet scored). |
| `extract.emit_candidate_audit` | `false` | Emits `candidate_audit.parquet` in `run`. |
| `extract.emit_gate_diagnostics` | `false` | Adds the four `gate_*` columns to `psms_extracted`. |
| `features.set` | `minimal` | `minimal` (14) / `rich` (44) / `extended` (~381); `dia` profile sets extended. |
| `compete.mode` | `winner_take_all` | Within-group competition resolution (`CompetitionMode`). |
| `rescore.classifier` | `native_tda` | Rescorer; `nn_torch` is the best lever, needs `rescore.python`. |
| `rescore.python` | `None` | Interpreter for mokapot / nn_torch / entrapment sidecars. |
| `quant.q_threshold` | (see `QuantConfig`) | q-value threshold for the human-readable report. |

The config surface was recently pruned of dead or unwired fields (plan.md
"Config-surface note"): `threads`, `extract.{k_select, max_fragment_charge,
scan_scale, scan_window_mode}` + `ScanWindowMode`, `digest.decoy.{ratio, source}`
+ `DecoySource`, `search_seed.precursor_tol_ppm`, `rt_im_train.tolerance_regime`
+ `ToleranceRegime`, `FeatureSet::Custom`, `MatcherKind::Naive`, and
`CompetitionMode::from_token` were removed. Do not reintroduce them. Kept
deliberately as documented hooks: `DecoyStrategy::DiannShift` (deferred,
license-checked; rejected by validation), `CalibrationMethod::None` (rejected by
validation with a clear message), and the whole `mbr` section (planned feature).

## Invariants, determinism, gotchas

- Determinism is required (plan.md Section 7): seed the RNG, keep numeric
  summation order fixed. A HashMap f32 sum shifting the apex once broke
  reproducibility; use ordered maps or sorted iteration where floats are summed.
  The DeepLC fine-tune is the one deliberate exception (nondeterministic; no fixed
  torch seed), so a `run` with `finetune_deeplc = true` is not bit-reproducible.
- `candidate_id` must be a contiguous 0-based index sorted by precursor m/z; the
  fragment index build in extract depends on it. An imported library must be
  re-indexed by `make_reverse_decoys.py` to satisfy this.
- q-value columns in `psms_scored` are independent per level, not a rollup
  (finding 6): `q_value` (== `experiment_psm_q`), `run_psm_q` (per source),
  `precursor_q` (peptidoform+charge), `peptide_q_value` (stripped sequence),
  `pg_q_value`. Coarser grouping pools evidence, so peptide counts can exceed
  precursor counts. Report at the correct context: `precursor_q` for a precursor
  matrix, `run_psm_q` for cross-run library-mode quant. Do not threshold PSM q
  then deduplicate.
- Every default-off knob keeps the production chain byte-identical when unset;
  `retain_top_peaks=1`, `emit_candidate_audit=false`, `emit_gate_diagnostics=false`,
  and the other sensitivity-program toggles all default to the legacy path.
- The extraction gate is a training-pool and null-curation lever, not feature
  work (finding 3). Loosening it floods the pool with decoy-enriched junk that
  corrupts a linear rescorer; a nonlinear rescorer absorbs it. The gate optimum
  therefore inverts by rescorer.
- The selected apex is the strongest peak only ~48-52% of the time (finding 8);
  the `<psms>.peaks.parquet` top-K sidecar exists to expose alternative peaks but
  is not yet promoted through features/rescore.

## How to extend / modify

- To add a stage: implement it as `stages/<name>.rs` with a `run(Params)` that
  reads path-addressable inputs and writes Parquet through the `Col`/`Table`
  layer plus an `<artifact>.report.json`; add a `(logical name, version)` to
  `schema.rs`; add a `Cmd` arm in `main.rs`; and thread it into `run.rs` with a
  `record_artifact` call. Keep the stage runnable standalone.
- To add an artifact column: add the `Col::` in the writing stage, update the
  reading stage's typed getter, and bump the schema version in `schema.rs` if the
  change is not backward-compatible (`PSMS_SCORED` is already at version 2).
- To add a tuning profile: extend the `match` in `Config::apply_profile`
  (`config.rs:1098`); it should only set existing typed config fields, never add
  plumbing.
- To add a library source: extend the `match` on `(lib_precursors,
  lib_fragments)` in `run.rs:96` and the corresponding `preflight` branch
  (`run.rs:42`). The rest of the chain consumes `(lib_p, lib_f)` unchanged.
- To wire a new sidecar: follow the positional-CLI file contract
  (`sidecar::resolve_script` + a worker in `scripts/`), gate it behind a config
  field, and check its prerequisites in `preflight` and `doctor`.
- Stubs and unwired areas to be aware of: `mbr` is a partial sidecar, not the full
  experiment-wide flow; `align` is not in the `run` chain (needs >=2 runs); ion
  mobility / 4D is a data-model stub only (IM columns are always null in 3D);
  vendor readers other than mzML do not exist. Do not document these as complete.
