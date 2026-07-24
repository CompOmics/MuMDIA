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

The design principle is that every computational stage is an independent command
over path-addressable inputs and outputs on disk (the interstage contract,
docs/18 B1). No stage shares in-memory state with another. `run` threads those
file paths, invokes the optional DeepLC fine-tune between stages, writes the
human-readable report, and records provenance. Most stage commands can therefore
be run standalone on prior outputs, on a different configuration, or on
hand-crafted minimal files.

`run` recomputes every stage on every invocation. There is no artifact caching,
no skip-if-exists, and no resume: `run` calls each stage unconditionally and
overwrites its outputs even when identical artifacts already exist in `--out-dir`
(`run.rs:85-509`; no existence check precedes any stage call, and
`std::fs::create_dir_all` at `run.rs:90` does not clear or skip). To reuse a
prior artifact, invoke the downstream stage command standalone on it instead of
rerunning `run`. Use a fresh output directory for each orchestrated run. Reusing
one can leave a stale optional sidecar from an earlier configuration, and a
failed rerun can leave an old manifest beside partially replaced outputs.

The validated findings and the interstage, determinism, and sidecar contracts
cited throughout this document are consolidated, self-contained, in
`docs/18_findings_and_decisions.md` (sections A1-A7 for findings, B1-B3 for
contracts); the citations below point there. plan.md holds the deeper
algorithmic spec but is local-only and gitignored.

## Files

| Path | Role |
|---|---|
| `rust/mumdia/crates/mumdia/src/main.rs` | CLI: one `clap` subcommand per stage (`Cmd` enum, `main.rs:20`); each arm loads config, hashes it, and calls the stage `run`. |
| `rust/mumdia/crates/mumdia/src/stages/run.rs` | `run` orchestrator: preflight, library-source branch, per-run chain, manifest assembly (`run.rs:85`). |
| `rust/mumdia/crates/mumdia/src/stages/mod.rs` | Re-exports every stage module. |
| `rust/mumdia/crates/mumdia/src/stages/*.rs` | The stage implementations (`convert`, `digest`, `peptidoforms`, `predict_frag`, `search_seed`, `rt_im_train`, `extract`, `features`, `compete`, `rescore`, `quant`, `report`, `align`, `audit`). |
| `rust/mumdia/crates/mumdia/src/lib.rs` | Crate lib root (bin+lib split, `lib.rs:5-16`): re-exports the pipeline modules so integration tests can drive stages directly. Public modules: `stages`, `matchers` (fragment matchers, doc 06), `index` (`Library` + inverted index, doc 06), `predict` (predictor traits + native fallbacks, doc 06), `rescoring` (`percolator_lite` + native NN, doc 11), `calibrate` (LOESS/linear/percentile, doc 08), `peaks` (peak enumeration, doc 09), `spectra` (in-memory spectrum model + loaders), `sidecar` (Python worker dispatch, doc 13), `stats` (pearson/cosine/spectral_angle kernel), `fdr` (target-decoy + entrapment q, doc 11). |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | Frozen `(logical name, schema version)` for every artifact (`artifact` module, `schema.rs:6`). |
| `rust/mumdia/crates/mumdia-core/src/manifest.rs` | `Manifest` and `ArtifactRecord` (`manifest.rs:10`, `manifest.rs:22`). |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | The single typed serde `Config` (`config.rs:973`) with per-stage sections and `apply_profile` (`config.rs:1107`). |
| `rust/mumdia/crates/mumdia-io/src/table.rs` | The `Col`/`Table` typed Parquet layer every stage writes through (`table.rs:23`). |
| `rust/mumdia/crates/mumdia-io/src/lib.rs` | `record_artifact`, `inspect`, `init_logging`, blake3 hashing. |

### Crate responsibilities and predictor plumbing

Three crates, one direction of dependency (`mumdia` -> `mumdia-io` ->
`mumdia-core`):

```
mumdia-core   types, mass model, Config (per-stage sections + strategy enums),
              schema (frozen artifact names/versions), manifest, errors
mumdia-io     Col/Table over arrow+parquet (SNAPPY), blake3 hashing, JSON,
              inspect, per-artifact report.json
mumdia        bin + lib: fragment index, stages/, predictor + rescorer traits,
              sidecar dispatch; main.rs is a thin CLI over the lib
```

The two ML predictors are traits with a deterministic native fallback and an
optional Python sidecar, selected by config (the sidecar replaces the native
path without changing callers, `predict.rs:1-8`). The rescorer is not a trait: a
free function plus the `RescorerKind` enum.

```
RtPredictor        (trait, predict.rs:13)
   |-- NativeRt     (predict.rs:25, additive RT-coefficient model, deterministic)
   \-- DeepLC       (Python sidecar; predict_frag.rt_predictor = deeplc)

FragmentPredictor  (trait, predict.rs:19)
   |-- NativeFrag   (predict.rs:73, heuristic b/y intensities, deterministic)
   \-- MS2PIP       (Python sidecar; predict_frag.predictor = ms2pip)

Rescorer           (free fn + RescorerKind enum, no trait)
   native_tda (percolator_lite) | nn_torch | mokapot | entrapment
   (the last three are Python sidecars via rescore.python)
```

## Inputs and outputs

The pipeline consumes an mzML file plus one of two library sources, and produces
a fixed set of Parquet artifacts plus JSON sidecars. Most primary stage Parquets
receive a small `<artifact>.report.json` with row counts, parameters, hashes, and
timing. Coverage is deliberately partial: report TSVs, schema/PIN files, some
diagnostic/optional Parquets, and several Python-written outputs do not have one.
See docs/03 and docs/12 rather than assuming a report always exists.

External inputs:
- `--mzml <file>` (mzML; `.raw`/`.d` must be converted upstream by msconvert).
- FASTA mode: `--fasta <file>` (digested into the library).
- Library-input mode: `--lib-precursors <parquet>` + `--lib-fragments <parquet>`
  (a prebuilt library, for example an imported DIA-NN speclib).

Inputs on disk: a pre-built E. coli test library is already present under `lib/`:
`lib_precursors.parquet` + `lib_fragments.parquet` (an imported DIA-NN speclib
with fragment intensities and iRT) plus `lib_precursors_ft.parquet` (the DeepLC
fine-tuned iRT variant, the recommended `--lib-precursors` per docs/18 A4), and a
cached `seed_psms.parquet` (+ `.masscal.json`). See `docs/19_getting_started.md`
for the local sidecar environments and two copy-pasteable end-to-end runs (a
zero-dependency native FASTA run and the best-sensitivity library run).

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
diagnostic sidecar not yet scored downstream (see docs/18 A6, and Gotchas below).

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

`psms_scored.parquet` (schema version 3): `candidate_id` u32,
`peptidoform` str, `charge` i32, `label` str, `protein` str, `base_peptide_id`
u32, carried `apex_rt`/`elution_lo`/`elution_hi`, `score` f64, `q_value` f64,
`peptide_q_value` f64, `protein_group` str,
`pg_q_value` f64, `global_q_value` f64, `prelim_score` f64, `source` u32,
`run_psm_q` f64, `experiment_psm_q` f64, `precursor_q` f64. The q-value columns
are independent per level, not a rollup (see Gotchas).

`peptide_quant.parquet` (schema version 2): `candidate_id`, `base_peptide_id`,
`peptidoform`, `charge`, `protein_group`, nullable `quantity`, `quant_status`,
`n_fragments_used`, and nullable applied integration apex/bounds.
`protein_group_quant.parquet` (schema version 2): `protein_group`, nullable
`quantity`, `quant_status`, and `n_peptides` (unique positive base peptides).
Optional `fragment_quant.parquet` and
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

## CLI subcommands

`main.rs` is a thin `clap` layer: the `Cmd` enum (`main.rs:20`) defines one
subcommand per stage plus the `run` orchestrator, the experiment-level
`align`/`mbr`, and the utilities `inspect`/`report`/`doctor`. Each stage arm
loads the config, hashes its canonical JSON into a `config_hash`, and calls the
stage `run`; every stage command is runnable standalone on prior outputs. Flags
are `--kebab-case`; a `num_args = 1..` flag accepts several values.

Which subcommands read `--config` (verified against `main.rs`):

| Takes `--config` | No `--config` |
|---|---|
| `digest`, `peptidoforms`, `predict-frag`, `search-seed`, `rt-im-train`, `extract`, `features`, `compete`, `rescore`, `quant`, `run`, `align`, `mbr`, `doctor` (optional) | `convert`, `quant-lfq`, `inspect`, `audit`, `report` |

The five with no `--config` are fixed-behavior: `convert` always runs on
`Config::default()` (`main.rs:397`); `quant-lfq` takes typed string flags instead
(`main.rs:642-658`); `inspect`, `audit`, and `report` load no config at all
(`main.rs:712`, `main.rs:557`, `main.rs:715`). The full subcommand set, in source
order:

| `mumdia <cmd>` | Inputs (flags) | Outputs (flags / artifacts) |
|---|---|---|
| `convert` (`main.rs:23`) | `--mzml`, `--out-dir`; caps `--max-spectra`, `--top-peaks-ms2`, `--top-peaks-ms1` (each `0` = uncapped) | `spectra_ms1`, `spectra_ms2`, `isolation_windows`, `ms2_to_ms1` in `--out-dir` |
| `digest` (`main.rs:43`) | `--fasta`, `--config` | `--out` = `peptides.parquet` |
| `peptidoforms` (`main.rs:52`) | `--peptides`, `--config` | `--out` = `peptidoforms.parquet` |
| `predict-frag` (`main.rs:61`) | `--peptidoforms`, `--work-dir` (default `sidecar_work`), `--config` | `--out-precursors`, `--out-fragments` (the library pair) |
| `search-seed` (`main.rs:75`) | `--ms2`, `--library-precursors`, `--library-fragments`, `--config` | `--out` = `seed_psms.parquet` (+ `<out>.masscal.json`) |
| `rt-im-train` (`main.rs:88`) | `--seed-psms`, `--library-precursors`, `--config` | `--out-windows` = `run_windows.parquet`, `--out-cal` = `cal.json` |
| `extract` (`main.rs:101`) | `--ms2`, `--library-precursors`, `--library-fragments`, `--run-windows`, `--ms1` (opt), `--mass-cal` (opt), `--restrict-candidates` (opt allowlist), `--config` | `--out-psms` = `psms_extracted.parquet`, `--out-chrom` = `chromatograms.parquet` (+ `<psms>.peaks.parquet` when `retain_top_peaks>1`) |
| `features` (`main.rs:130`) | `--psms`, `--chromatograms`, `--seed` (opt), `--config` | `--out` = `features.parquet` (+ `.schema.json`), `--out-pin` = PIN |
| `compete` (`main.rs:146`) | `--features`, `--config` | `--out` = `psms_competed.parquet` (+ `.schema.json`) |
| `rescore` (`main.rs:155`) | `--competed` (1+ competed tables), `--config` | `--out` = `psms_scored.parquet` |
| `quant` (`main.rs:165`) | `--psms-scored`, `--chromatograms`, `--config` | `--out-peptide`, `--out-protein`, `--out-fragment` (opt), `--out-peak-bounds` (opt) |
| `quant-lfq` (`main.rs:184`) | `--inputs` (1+ per-run tables), `--method` (`maxlfq` default / `directlfq`), `--normalize` (`median_ratio` default / `median` / `none`) | `--out` = protein-by-run matrix; no `--config` |
| `run` (`main.rs:199`) | `--mzml`, `--out-dir`; `--fasta` xor (`--lib-precursors` + `--lib-fragments`); `--config`, `--profile`, `--max-spectra`, `--top-peaks-ms2` | the full chain + `manifest.json` |
| `align` (`main.rs:230`) | `--seeds` (1+ `seed_psms`, first = reference), `--config` | `--out` = `alignment.parquet` |
| `mbr` (`main.rs:240`) | `--scored` (experiment-wide `scored_combined`), `--psms` (1+ per-run in `source` order), `--frag` (0+ per-run `fragment_quant`), `--config` | `--out` = `transferred.parquet`, `--out-scored` (opt augmented scored) |
| `inspect` (`main.rs:261`) | positional `artifact` (any parquet) | schema + head + row count to stdout; no `--config` |
| `audit` (`main.rs:265`) | `--library-precursors`, `--psms`, `--competed`, `--scored`, `--q` (0.01), `--run-id` (`run`), `--entrapment-substr` (`""`) | `--out` = `candidate_audit.parquet`; no `--config` |
| `report` (`main.rs:292`) | `--scored`, `--peptide-quant` (opt), `--protein-quant` (opt), `--q` (0.01) | `peptides.tsv` + `proteins.tsv` in `--out-dir` |
| `doctor` (`main.rs:305`) | `--config` (opt) | probes the configured sidecar interpreters; nonzero exit if any FAIL |

Per-subcommand specifics that are easy to miss:

- `convert` is the only stage command with no `--config`; it always runs on
  `Config::default()` (`main.rs:397`). Because the three caps are not part of the
  config, `convert` folds them into the artifact `config_hash` (`main.rs:402`,
  comment.md A2/C4) so two different caps do not collide on an identical hash.
  `top_peaks_ms2` is an irreversible conversion-time cap that also affects
  extraction/features/quant; `search_seed.top_n_peaks` is the seed-only
  alternative.
- `quant-lfq`, `inspect`, `audit`, and `report` also take no `--config`.
  `quant-lfq` validates its two string flags: `--method` must be `maxlfq` or
  `directlfq` (`main.rs:649`) and `--normalize` is parsed by
  `NormalizeMethod::from_token` (`config.rs:754`), erroring on anything but
  `median_ratio`/`median`/`none` (`main.rs:652`).
- `search-seed` reads `extract.bucket_size` from the config (not a `search_seed`
  field) for its fragment-index bucketing (`main.rs:473`, `run.rs:225`).
- `rescore` standalone accepts several `--competed` tables for experiment-wide
  scoring and uses a fixed `sidecar_work` working directory (`main.rs:588`);
  inside `run` it is passed exactly one table and a per-out-dir work directory.
- `align` uses `rt_im_train.q_train` for its training set and a hardcoded 100-knot
  grid (`main.rs:666-667`); it needs >=2 seeds and is not part of the `run` chain.
- `mbr` (`main.rs:671`) is a wired standalone command, not part of `run`. It
  hard-errors when `mbr.strategy = none` (`main.rs:680`), when fewer than two
  `--psms` are supplied (`main.rs:685`), or when `mbr.python` is unset
  (`main.rs:688`), then runs the `mbr_worker.py` sidecar with the `mbr.*`
  thresholds (`main.rs:697`). It transfers identifications across a run set
  (Stage D3); see the extend section for its stub status inside `run`.
- `doctor` (`main.rs:313`) probes three interpreters and prints `[ ok ]` /
  `[FAIL]` / `[skip]` per line, exiting nonzero if any fail: `rescore.python`
  (packages depend on the classifier: `torch,numpy,pandas,pyarrow` for `nn_torch`,
  else `mokapot,sklearn,numpy,pandas,pyarrow`), `predict_frag.deeplc_python`
  (`deeplc,numpy,pandas`), and `predict_frag.ms2pip_python`
  (`ms2pip,numpy,pandas`).

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
                                    (new table with updated predicted_irt; lib_p := *_ft)
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
                              selected primary Parquets recorded --> manifest.json
```

### The two library sources and how run.rs branches

The spectral library is the experiment-level, run-independent artifact pair
`(lib_p, lib_f)`. The orchestrator produces it in one of two ways, decided by a
`match` on `(p.lib_precursors, p.lib_fragments)` at `run.rs:98`:

- Library-input mode (`(Some(lp), Some(lf))`, `run.rs:99`): the supplied library
  is consumed directly. `digest`, `peptidoforms`, and `predict-frag` are skipped
  and the FASTA is never read. The orchestrator only reads the two Parquet files
  to record their row counts in the manifest, then uses their paths downstream
  (`run.rs:107-125`). This is the highest-sensitivity path, used with an imported
  DIA-NN speclib that already carries fragment intensities and iRT.

- FASTA-digest mode (the `_ =>` arm, `run.rs:127`): the library is built from the
  FASTA. `digest::run` writes `peptides.parquet` (`run.rs:132`),
  `peptidoforms::run` writes `peptidoforms.parquet` (`run.rs:149`), and
  `predict_frag::run` writes both library Parquet files (`run.rs:166`). The
  returned `(lib_p, lib_f)` paths feed the rest of the chain. This path has zero
  external runtime dependencies (native predictors).

`preflight` (`run.rs:38`) validates the mode before any compute: exactly one of
the two source configurations must be present, and every required input path must
exist (`run.rs:42-61`). It also checks sidecar prerequisites: `finetune_deeplc`
requires `predict_frag.deeplc_python`; Mokapot and NnTorch require
`rescore.python`; entrapment requires `rescore.entrapment_marker`. `Config::validate`
rejects the same rescorer omissions at load time. `mumdia doctor` additionally
checks that the configured interpreters can import the required packages.

### The per-run chain

After the library is resolved, the orchestrator runs the per-run stages in order,
recording each output in the manifest with `record_artifact` (`run.rs:85-509`):

1. `convert::run` (`run.rs:196`) reads the mzML through mzdata, centroids profile
   spectra, caps peaks, synthesizes a full-range window for AIF/all-ion scans,
   and writes the four spectra artifacts. It returns a `ConvertOutputs` struct of
   paths (`convert.rs:96`). `run` forwards its `max_spectra` and `top_peaks_ms2`
   but forces `top_peaks_ms1 = 0` (`run.rs:201`), so `run` never caps MS1 peaks;
   only the standalone `convert` command can (`--top-peaks-ms1`).

2. `search_seed::run` (`run.rs:219`) runs a native Sage-lite broad DIA search over
   `spectra_ms2` against the fragment index for calibration (not final ID). It
   writes `seed_psms.parquet` and the mass recalibration `masscal.json`. The seed
   is iRT-independent, so it is computed once here on the base library and reused.
   It borrows `extract.bucket_size` for the fragment-index bucketing
   (`run.rs:225`), not a `search_seed` field.

3. Optional DeepLC multitask fine-tune (`run.rs:242`, gated on
   `rt_im_train.finetune_deeplc`): `sidecar::run_deeplc_finetune` adapts the RT
   model to this run's confident seed PSMs and writes a new
   `fragment_library_precursors_ft.parquet` whose `predicted_irt` values are
   updated; the input library is not modified. It is driven by the `rt_im_train`
   fields `finetune_epochs`, `finetune_patience`, `q_train` (the confident-seed
   cutoff), and `finetune_batch` (`run.rs:259-262`; `finetune_batch = 0`
   auto-scales the batch to the seed size). The `lib_p` binding is then rebound to
   the fine-tuned file so rt-im-train and extract read it. `lib_f` is unchanged
   (fine-tune touches iRT only). This step is nondeterministic (no fixed
   torch/numpy seed).

4. `rt_im_train::run` (`run.rs:284`) maps the run-independent iRT onto this run's
   observed RT (LOESS or linear) and sets per-candidate RT windows from the
   residual percentile, writing `run_windows.parquet` and `cal.json`.

5. `extract::run` (`run.rs:303`) is the core stage: a peak-major cascade over the
   inverted fragment index. It reads `spectra_ms2`, both library files,
   `run_windows`, `spectra_ms1` (for MS1 isotope XICs), and the `masscal.json`; it
   writes `psms_extracted.parquet` and `chromatograms.parquet`. `run` passes
   `restrict_candidates: None` (no allowlist).

6. `features::run` (`run.rs:335`) computes the config-selected feature vector plus
   `prelim_score` per PSM and emits `features.parquet`, its schema sidecar, and
   `run.pin`. It also reads `seed` for search-engine corroboration features.

7. `compete::run` (`run.rs:354`) keeps the best candidate per competition group
   before FDR counting, writing `psms_competed.parquet` and carrying the feature
   schema forward.

8. `rescore::run` (`run.rs:370`) does semi-supervised rescoring and native
   target-decoy q-values at multiple contexts, writing `psms_scored.parquet`. In
   `run` it is passed a single competed table (`std::slice::from_ref(&competed)`), so this is
   single-run rescoring; the standalone `rescore` command accepts several tables
   for experiment-wide scoring.

9. Optional `audit::run` (`run.rs:405`, gated on `extract.emit_candidate_audit`)
   reconstructs the per-candidate identification-loss ladder into
   `candidate_audit.parquet`. It is a cheap join over the artifact chain and runs
   no extraction. Inside `run` its parameters are fixed: `q_threshold = 0.01`,
   `run_id = out_dir`, and an empty `entrapment_substr` (`run.rs:413-415`); the
   standalone `audit` command exposes all three as flags.

10. `quant::run` (`run.rs:422`) integrates fragment chromatograms and rolls up to
    protein groups, writing `peptide_quant`, `protein_group_quant`, and
    `fragment_quant`.

11. `report::run` writes `peptides.tsv` and `proteins.tsv` from the scored table
    plus quant, thresholded at `quant.q_threshold`. `run` reads the rescore
    artifact report and uses its actual classifier in stdout and the manifest;
    `psms_scored.parquet.report.json` remains the authoritative record.

### The manifest

`Manifest` (`manifest.rs:22`) has five fields: `mumdia_version` (the crate
version, `env!("CARGO_PKG_VERSION")`, stamped by `Manifest::new`), `config_json`
(the fully-resolved canonical config), `config_hash` (its blake3 hash),
`model_identities` (a `BTreeMap<String, String>`), and `artifacts` (a
`BTreeMap<String, ArtifactRecord>`). `Manifest::new` (`manifest.rs:34`) seeds it
with the canonical config JSON and hash (`run.rs:89,93`). Selected primary
Parquet outputs are recorded by `record_artifact` (`mumdia-io`), which builds an
`ArtifactRecord`
(`manifest.rs:10`) with nine fields: `logical_name`, `path`, `format`,
`schema_name`, `schema_version`, `rows`, `content_hash`, `producing_stage`, and
`config_hash`. `Manifest::record` (`manifest.rs:44`) keys artifacts by
`logical_name`, so re-recording the same logical name overwrites and the map is
sorted, not insertion-ordered. The `producing_stage` is the stage name
(`digest`, `convert`, ...); in library-input mode the two library records carry
the synthetic stage `"library-input"` (`run.rs:114,122`). Four `model_identities`
keys are recorded: `rt_predictor`, `fragment_predictor`, `rescorer`, and
`feature_schema_id`. The rescorer identity comes from the rescore report, and the
fine-tuned precursor table is recorded when it is the table actually consumed
downstream. Calibration JSON, PIN/schema companions, report TSVs, and some
optional diagnostic sidecars are not manifest artifacts. The manifest is written last to
`<out_dir>/manifest.json` (`run.rs:501`) with `mumdia_io::json::write_json`. The
manifest is provenance recorded, not required: because inputs are
path-addressable, no stage depends on it to run (docs/18 B1).

### Current best workflow

The validated best-performing configuration (docs/18 A1 and its "current best
workflow") is library-input mode with:
- an imported DIA-NN library (fragment intensities + iRT),
- per-run DeepLC multitask fine-tune of iRT (`rt_im_train.finetune_deeplc = true`
  + `predict_frag.deeplc_python`), which is essential in library mode because raw
  DIA-NN iRT is locally noisy (~110 s MAD) and the fine-tune tightens it to
  ~13-27 s (docs/18 A4),
- the Extended feature set (`features.set = extended`, applied by `--profile dia`,
  `config.rs:1101`),
- the `nn_torch` PyTorch MLP rescorer (`rescore.classifier = nn_torch` +
  `rescore.python`, with `rescore.strict = true`), which is the dominant
  sensitivity lever and beats the native linear rescorer by ~8.5% (docs/18 A1),
- a loose default apex-intensity Pearson gate
  (`extract.gate_mode = apex_pearson`, `extract.min_frag_corr ~= 0.2`), since
  a strong nonlinear rescorer prefers recall and absorbs the loose-gate flood
  (docs/18 A1/A2),
- the conversion-time MS2 peak cap explicitly kept at 300 on AIF
  (`--top-peaks-ms2 300`; both conversion entry points otherwise default to
  uncapped). This is distinct from seed-only `search_seed.top_n_peaks`.

On `LFQ_Orbitrap_AIF_Ecoli_01.mzML` this historically reaches ~10,300
precursor-shaped `(peptidoform, charge)` report rows selected at
`peptide_q_value <= 0.01`; it is not a precursor-q count or a universal preset.
The zero-dependency native FASTA-digest mode (~1,213 rows under the same report
definition) is the high-precision fallback. See docs/20 for the exact command,
acceptance gates, and quantification-specific choices.

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `Cmd` (enum) | `main.rs:20` | One clap subcommand per stage; `Run` at `main.rs:199`. |
| `main` | `main.rs:386` | Parses the CLI, loads config, dispatches to the stage `run`. |
| `load_config` | `main.rs:376` | Reads a config JSON or returns `Config::default()`. |
| `doctor` | `main.rs:313` | Probes configured sidecar interpreters for required packages. |
| `RunParams` | `run.rs:18` | The orchestrator's inputs (config, fasta, mzml, out_dir, lib paths, caps). |
| `preflight` | `run.rs:38` | Validates mode, input existence, and sidecar prerequisites before compute. |
| `stages::run::run` | `run.rs:85` | The orchestrator: library branch, per-run chain, manifest write. |
| `Manifest` / `ArtifactRecord` | `manifest.rs:22` / `manifest.rs:10` | Provenance record for a chained run. |
| `artifact` module | `schema.rs:6` | Frozen `(logical name, schema version)` constants. |
| `Config::apply_profile` | `config.rs:1107` | Applies the `dia` preset (Extended features, apex window 5, RT prior 120 s). |
| `Config::canonical_json` | `config.rs:1116` | Canonical JSON for hashing into the manifest. |
| `NormalizeMethod::from_token` | `config.rs:754` | Parses `--normalize` for `quant-lfq` (`median_ratio`/`median`/`none`). |
| `Manifest::record` | `manifest.rs:44` | Inserts an `ArtifactRecord` keyed by `logical_name` (overwrites, sorted map). |
| `Col` (enum) | `table.rs:23` | Typed column for writing Parquet; `LargeListF32` for chromatograms. |
| `Table::read` + typed getters | `table.rs:204` | Reads a Parquet artifact and extracts typed columns. |

## Configuration

The orchestrator reads a single typed `Config` (`config.rs:973`) with per-stage
sections. Every field has `#[serde(default)]` and the struct uses
`deny_unknown_fields`, so a partial JSON overrides only named fields and an
unknown key is a hard error. `--profile dia` is applied on top of the loaded
config (`main.rs:607`) before `run`. The fields this overview area touches:

| Field | Default | Effect |
|---|---|---|
| `rng_seed` | `0` | Seeds decoy generation and rescorer folds (determinism). |
| `digest.decoy.strategy` | `reverse` | Decoy scheme; `diann_shift` and `none` are rejected/no-op. |
| `peptidoforms.charge_min` / `charge_max` | `2` / `3` | Precursor charge range enumerated in FASTA mode. |
| `predict_frag.rt_predictor` | `native` | `native` or `deeplc` iRT source (recorded in manifest). |
| `predict_frag.predictor` | `native` | `native` or `ms2pip` fragment intensities. |
| `predict_frag.deeplc_python` | `None` | Interpreter for DeepLC; required if `finetune_deeplc`; probed by `doctor`. |
| `predict_frag.ms2pip_python` | `None` | Interpreter for MS2PIP; probed by `doctor` (`main.rs:332`). |
| `predict_frag.sidecar_script_dir` | `"scripts"` | Where sidecar workers are resolved from (used by `rescore`/`mbr`). |
| `search_seed.top_n_peaks` | `300` | Seed-only MS2 peak cap (keep 300 on AIF, docs/18 A3). |
| `extract.bucket_size` | (see `ExtractConfig`) | Fragment-index bucket width; `search-seed` reads it too (`main.rs:473`). |
| `rt_im_train.calibration_method` | `loess` | RT calibration; `none` is rejected by validation. |
| `rt_im_train.q_train` | (see `RtImTrainConfig`) | Confident-seed q cutoff for the DeepLC fine-tune and the `align` reference set. |
| `rt_im_train.finetune_deeplc` | `false` | Enables the per-run DeepLC fine-tune of iRT (best workflow). |
| `rt_im_train.finetune_epochs` / `finetune_patience` | (see config) | DeepLC fine-tune schedule (`run.rs:259-260`). |
| `rt_im_train.finetune_batch` | `0` | `0` auto-scales the fine-tune batch to seed size. |
| `extract.min_frag_corr` | `0.2` | Spectral-agreement gate threshold; loose default suits `nn_torch`. |
| `extract.gate_mode` | `apex_pearson` | Which spectral score the gate thresholds (`GateMode`, `config.rs:591`). |
| `extract.retain_top_peaks` | `1` | `>1` writes the `<psms>.peaks.parquet` sidecar (not yet scored). |
| `extract.emit_candidate_audit` | `false` | Emits `candidate_audit.parquet` in `run`. |
| `extract.emit_gate_diagnostics` | `false` | Adds the four `gate_*` columns to `psms_extracted`. |
| `features.set` | `minimal` | `minimal` (14) / `rich` (44) / `extended` (381, per the `feature_sets_sized` test `features.rs:1590-1598`); `dia` profile sets extended. |
| `compete.mode` | `winner_take_all` | Within-group competition resolution (`CompetitionMode`). |
| `rescore.classifier` | `native_tda` | Rescorer; `nn_torch` is the best lever, needs `rescore.python`. |
| `rescore.python` | `None` | Interpreter for mokapot / nn_torch / entrapment sidecars. |
| `mbr.strategy` | `none` | MBR mode; `none` makes the standalone `mumdia mbr` command error out (`main.rs:680`). |
| `mbr.python` | `None` | `mbr_worker.py` interpreter; required when `strategy != none` (`main.rs:688`). |
| `quant.q_threshold` | `0.01` (`QuantConfig`) | q-value threshold for the human-readable report. |

The config surface was recently pruned of dead or unwired fields (see
`docs/02_config_and_data_model.md`): `threads`, `extract.{k_select, max_fragment_charge,
scan_scale, scan_window_mode}` + `ScanWindowMode`, `digest.decoy.{ratio, source}`
+ `DecoySource`, `search_seed.precursor_tol_ppm`, `rt_im_train.tolerance_regime`
+ `ToleranceRegime`, `FeatureSet::Custom`, `MatcherKind::Naive`, and
`CompetitionMode::from_token` were removed. Do not reintroduce them. Kept
deliberately as documented hooks: `DecoyStrategy::DiannShift` (deferred,
license-checked; rejected by validation), `CalibrationMethod::None` (rejected by
validation with a clear message), and the whole `mbr` section, which is wired to
the standalone `mumdia mbr` command (`main.rs:671`, calls `mbr_worker.py`) but is
not part of the `run` chain and needs >=2 runs.

## Invariants, determinism, gotchas

- Determinism is required (docs/18 B2, the determinism contract): seed the RNG,
  keep numeric summation order fixed. A HashMap f32 sum shifting the apex once broke
  reproducibility; use ordered maps or sorted iteration where floats are summed.
  The DeepLC fine-tune is the one deliberate exception (nondeterministic; no fixed
  torch seed), so a `run` with `finetune_deeplc = true` is not bit-reproducible.
- `candidate_id` must be a contiguous 0-based index sorted by precursor m/z; the
  fragment index build in extract depends on it. An imported library must be
  re-indexed by `make_reverse_decoys.py` to satisfy this.
- q-value columns in `psms_scored` are independent per level, not a rollup
  (docs/18 A5): `q_value` (== `experiment_psm_q`), `run_psm_q` (per source),
  `precursor_q` (peptidoform+charge), `peptide_q_value` (stripped sequence),
  `pg_q_value`. Coarser grouping pools evidence, so peptide counts can exceed
  precursor counts. Report at the correct context: `precursor_q` for a precursor
  matrix, `run_psm_q` for cross-run library-mode quant. Do not threshold PSM q
  then deduplicate.
- Every default-off knob keeps the production chain byte-identical when unset;
  `retain_top_peaks=1`, `emit_candidate_audit=false`, `emit_gate_diagnostics=false`,
  and the other sensitivity-program toggles all default to the legacy path.
- The extraction gate is a training-pool and null-curation lever, not feature
  work (docs/18 A2). Loosening it floods the pool with decoy-enriched junk that
  corrupts a linear rescorer; a nonlinear rescorer absorbs it. The gate optimum
  therefore inverts by rescorer.
- The selected apex is the strongest peak only ~48-52% of the time (docs/18 A6);
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
  change is not backward-compatible (`PSMS_SCORED` is already at version 3).
- To add a tuning profile: extend the `match` in `Config::apply_profile`
  (`config.rs:1107`); it should only set existing typed config fields, never add
  plumbing.
- To add a library source: extend the `match` on `(lib_precursors,
  lib_fragments)` in `run.rs:98` and the corresponding `preflight` branch
  (`run.rs:43`). The rest of the chain consumes `(lib_p, lib_f)` unchanged.
- To wire a new sidecar: follow the positional-CLI file contract
  (`sidecar::resolve_script` + a worker in `scripts/`), gate it behind a config
  field, and check its prerequisites in `preflight` and `doctor`.
- Stubs and unwired areas to be aware of: `mbr` has a wired standalone command
  (`mumdia mbr` -> `mbr_worker.py`) but is a partial sidecar (not the full
  experiment-wide flow) and is not in the `run` chain (needs >=2 runs); `align` is
  likewise standalone-only (needs >=2 runs); ion mobility / 4D is a data-model
  stub only (IM columns are always null in 3D); vendor readers other than mzML do
  not exist. Do not document these as complete.
