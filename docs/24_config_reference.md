# MuMDIA configuration reference

GENERATED FILE. Do not edit. `ci/gen_config_reference.py` parses
`rust/mumdia/crates/mumdia-core/src/config.rs` and the environment-variable
reads in the crates and the sidecar scripts, so every field, type, default,
and description below is the one the code actually uses. Change the field or
its doc comment in the Rust source and regenerate. An edit made to this file
is lost on the next run.

```text
python ci/gen_config_reference.py            # regenerate
python ci/gen_config_reference.py --check    # fail if this file is stale
```

For the command-line interface that loads these files, read
`docs/23_cli_reference.md`. For how the config is loaded, validated, and
hashed into the run manifest, read `docs/02_config_and_data_model.md`.

## How to read this document

The configuration is one JSON object with a per-stage section. Every field
carries `#[serde(default)]` and the top-level object is
`deny_unknown_fields`, so a config may omit any field but may not contain a
key the engine does not know: a typo is a hard parse error, not a silently
ignored line. `--config` therefore always describes a complete
configuration, with the defaults in this document filling the rest.

Columns:

- **Default** is the value from the `impl Default` block, rendered as the
  JSON a config file would carry. A field whose default the parser could not
  resolve is marked `unresolved` and listed at the end of this document,
  never omitted.
- **Gated** is non-empty when the field's own doc comment marks it as not
  part of the shipped, validated chain. It repeats the phrase that matched:
  `benchmark-gated` and `gated` mean the change needs entrapment plus a
  second acquisition before it becomes a default (CLAUDE.md, "Changes that
  remain benchmark-gated"); `diagnostic` means the field only adds a sidecar
  artifact or extra columns; `not yet wired` means no code reads the field
  yet. Treat a non-empty cell as: do not enable this because it sounds
  useful.
- **Description** is the field's Rust doc comment, unwrapped to one
  paragraph. Nothing is paraphrased.

Enum-valued fields show their default as the serde spelling; the accepted
values of every enum are in "Enumerations" at the end. An empty description
means the field carries no doc comment in the source, not that it is
undocumented on purpose; those fields are counted under "Coverage".

## Sections

| Section | Struct | Fields | Stage document |
|---|---|---|---|
| [(top level)](#top-level) | `Config` | 16 | [docs/02_config_and_data_model.md](02_config_and_data_model.md) |
| [`convert`](#convert) | `ConvertConfig` | 4 |  |
| [`prescan`](#prescan) | `PrescanConfig` | 7 | [docs/21_prescan.md](21_prescan.md) |
| [`digest`](#digest) | `DigestConfig` | 6 | [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md) |
| [`digest.decoy`](#digestdecoy) | `DecoyConfig` | 1 | [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md) |
| [`peptidoforms`](#peptidoforms) | `PeptidoformsConfig` | 7 | [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md) |
| [`predict_frag`](#predict_frag) | `PredictFragConfig` | 14 | [docs/06_predict_frag_index_matchers.md](06_predict_frag_index_matchers.md) |
| [`search_seed`](#search_seed) | `SearchSeedConfig` | 8 | [docs/07_search_seed.md](07_search_seed.md) |
| [`rt_im_train`](#rt_im_train) | `RtImTrainConfig` | 19 | [docs/08_rt_im_train.md](08_rt_im_train.md) |
| [`extract`](#extract) | `ExtractConfig` | 37 | [docs/09_extract.md](09_extract.md) |
| [`extract.claim_cues`](#extractclaim_cues) | `ClaimCues` | 7 | [docs/09_extract.md](09_extract.md) |
| [`features`](#features) | `FeaturesConfig` | 11 | [docs/10_features.md](10_features.md) |
| [`compete`](#compete) | `CompeteConfig` | 6 | [docs/11_compete_rescore_fdr.md](11_compete_rescore_fdr.md) |
| [`rescore`](#rescore) | `RescoreConfig` | 22 | [docs/11_compete_rescore_fdr.md](11_compete_rescore_fdr.md) |
| [`quant`](#quant) | `QuantConfig` | 17 | [docs/12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md) |
| [`mbr`](#mbr) | `MbrConfig` | 9 | [docs/12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md) |
| [`experiment`](#experiment) | `ExperimentConfig` | 3 | [docs/01_overview_and_dataflow.md](01_overview_and_dataflow.md) |
| [`groups`](#groups) | `GroupsConfig` | 8 |  |
| [`peptidoforms.fixed_mods[] / peptidoforms.variable_mods[]`](#peptidoformsfixed_mods--peptidoformsvariable_mods) | `ResidueMod` | 2 | [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md) |

## (top level)

`Config` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/02_config_and_data_model.md](02_config_and_data_model.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `convert` | `ConvertConfig` | the `ConvertConfig` section's own defaults |  |  |
| `prescan` | `PrescanConfig` | the `PrescanConfig` section's own defaults |  |  |
| `rng_seed` | `u64` | `0` |  |  |
| `digest` | `DigestConfig` | the `DigestConfig` section's own defaults |  |  |
| `peptidoforms` | `PeptidoformsConfig` | the `PeptidoformsConfig` section's own defaults |  |  |
| `predict_frag` | `PredictFragConfig` | the `PredictFragConfig` section's own defaults |  |  |
| `search_seed` | `SearchSeedConfig` | the `SearchSeedConfig` section's own defaults |  |  |
| `rt_im_train` | `RtImTrainConfig` | the `RtImTrainConfig` section's own defaults |  |  |
| `extract` | `ExtractConfig` | the `ExtractConfig` section's own defaults |  |  |
| `features` | `FeaturesConfig` | the `FeaturesConfig` section's own defaults |  |  |
| `compete` | `CompeteConfig` | the `CompeteConfig` section's own defaults |  |  |
| `rescore` | `RescoreConfig` | the `RescoreConfig` section's own defaults |  |  |
| `quant` | `QuantConfig` | the `QuantConfig` section's own defaults |  |  |
| `mbr` | `MbrConfig` | the `MbrConfig` section's own defaults |  |  |
| `experiment` | `ExperimentConfig` | the `ExperimentConfig` section's own defaults |  |  |
| `groups` | `GroupsConfig` | the `GroupsConfig` section's own defaults |  |  |

## convert

`ConvertConfig` (rust/mumdia/crates/mumdia-core/src/config.rs).

Vendor-format conversion, read by every subcommand that takes a spectra path (`raw.rs`; docs/04_convert.md, "Vendor formats"). The engine itself reads mzML only, deliberately: `mzdata` is pinned to its pure-Rust `mzml` + `miniz_oxide` features so the build needs no C or .NET toolchain, and its vendor readers would reintroduce both. A vendor file is therefore converted to mzML first by an external converter run as a child process, in the same way the Python sidecars are: ThermoRawFileParser for Thermo `.raw`, ProteoWizard `msconvert` for everything else and as the Thermo fallback.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `thermo_raw_parser` | `String` | `"auto"` |  | Path to the ThermoRawFileParser executable, or `"auto"` to search. `"auto"` looks at `MUMDIA_THERMO_PARSER`, then beside the engine binary, then on `PATH`. Empty means the same as `"auto"`; a real path is used verbatim and its absence is an error rather than a silent fallback, because a fallback would convert with a different program than the one asked for and vendor conversion is not reproducible across converters. |
| `msconvert` | `String` | `"auto"` |  | Path to ProteoWizard `msconvert`, or `"auto"` to search. Used for every vendor format except Thermo, which prefers ThermoRawFileParser: Bruker `.d`, SCIEX `.wiff`, Agilent `.d` and Waters `.raw`. It is also the Thermo fallback when no ThermoRawFileParser is found. `"auto"` searches `MUMDIA_MSCONVERT`, beside the engine binary, the version-stamped ProteoWizard directories under Program Files on Windows (newest first), then `PATH`. MuMDIA never ships or downloads ProteoWizard. Its vendor readers bundle the instrument vendors' own libraries under the vendors' licence terms, which the user accepts when obtaining it, and automating that acceptance is not MuMDIA's to do. |
| `msconvert_args` | `Vec<String>` | `[]` |  | Extra arguments appended to every `msconvert` invocation. An escape hatch, not a tuning surface. The per-vendor defaults already request indexed 64-bit zlib mzML, vendor peak picking where it exists, and `--combineIonMobilitySpectra` for Bruker. Use this for something the defaults cannot express, such as an `--filter` that trims an acquisition. Arguments are passed through verbatim and are not validated. |
| `reuse_converted` | `bool` | `true` |  | Reuse an mzML that already sits beside the `.raw` and is newer than it. On by default: conversion is minutes per file and its output is deterministic given the same converter, so re-running a search should not pay for it twice. Turn it off when the neighbouring mzML may have come from a different converter or a different `.raw` of the same name. |

## prescan

`PrescanConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/21_prescan.md](21_prescan.md).

Sequence-tag prescan (`mumdia prescan`). Prunes modification-bearing candidates that have no anchored tag support in a given run, before the per-run library is assembled. The screen is deliberately blind to target/decoy label: tags are emitted in both orientations and a reverse decoy preserves composition and precursor m/z, so a decoy survives exactly when its target does. That keeps exchangeability, and therefore downstream FDR, intact.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `tol_da` | `f64` | `0.005` |  | Peak-delta match tolerance in Da. Permissive on purpose: a false tag only fails to prune, while a missed tag discards a real candidate with no way to recover it downstream. |
| `rt_slack_s` | `f64` | `150.0` |  | Widen each candidate's RT window by this many seconds before binning. The window comes from a calibration fitted on a different run, and `cal.json` residuals are in-sample and roughly 3x optimistic, so size this from out-of-sample RT error, not from the reported fit. |
| `rt_bin_s` | `f64` | `25.0` |  | RT bin width for the observed-tag index. |
| `top_peaks` | `usize` | `150` |  | Most intense peaks per MS2 used to build tags (0 = all). This bounds the O(peaks^2) delta search and is NOT destructive: it only affects tag construction, never the spectra artifact that extraction later reads. |
| `mods` | `Vec<String>` | `["C:Carbamidomethyl", "M:Oxidation"]` |  | Residue:UniModName entries that may appear in a screened peptidoform, e.g. `C:Carbamidomethyl`. A peptidoform carrying anything outside this set plus `anchor_mods` is dropped rather than screened on a partially understood sequence. |
| `anchor_mods` | `Vec<String>` | `[]` |  | Residue:UniModName entries the screen anchors ON. Only trimers covering one of these positions count as evidence, so backbone signal cannot keep a modified hypothesis alive. |
| `anchor_all` | `bool` | `false` |  | Screen EVERY candidate on every trimer of its sequence, modified or not, instead of only the modification-bearing candidates on their anchored trimers. This turns the prescan from a modform pruner into a per-run library pruner for a search space that is large on its own, such as a predicted immunopeptidomics library of 10^8 precursors, where a run supports only a small fraction of the enumeration. The screen stays label-blind (both orientations of every trimer; a reverse decoy's tag set is its target's), so it remains a compute reduction and never a discriminator. `anchor_mods` may be empty when this is set. Default off: the anchored screen is the measured one. |

## digest

`DigestConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `enzyme` | `Enzyme` | `trypsin_p` |  |  |
| `missed_cleavages` | `u32` | `2` |  |  |
| `min_len` | `usize` | `5` |  |  |
| `max_len` | `usize` | `50` |  |  |
| `decoy` | `DecoyConfig` | the `DecoyConfig` section's own defaults |  |  |
| `n_term_met_excision` | `bool` | `true` |  | N-terminal methionine excision: when a protein begins with `M`, also emit the initiator-Met-removed form of its N-terminal peptides. The initiator methionine is cleaved in vivo for most proteins, so search engines (including DIA-NN via `--met-excision`) enumerate both forms. Omitting it makes the search database structurally miss those excised peptides. |

## digest.decoy

`DecoyConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `strategy` | `DecoyStrategy` | `reverse` |  |  |

## peptidoforms

`PeptidoformsConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `fixed_mods` | `Vec<ResidueMod>` | `[{"residue": "C", "name": "Carbamidomethyl"}]` |  | UniMod names applied to every matching residue (residue -> mod name). |
| `variable_mods` | `Vec<ResidueMod>` | `[{"residue": "M", "name": "Oxidation"}]` |  |  |
| `max_variable_mods` | `usize` | `1` |  |  |
| `charge_min` | `i32` | `2` |  |  |
| `charge_max` | `i32` | `3` |  |  |
| `charge_by_basic_residues` | `bool` | `false` | benchmark-gated | Composition-based precursor charge range. When true, ignore `charge_min`/`charge_max` and emit every charge from 1 up to `1 (N-terminus) + (#R + #H + #K)`, the proton-carrying capacity of the peptide. Peptides therefore never receive a charge state they cannot physically hold, and each peptide's range depends on its own basic-residue count. Default false (fixed `charge_min..=charge_max` for every peptide). Pairs with `predict_frag.charge_by_basic_residues` for fragments. Changing the enumerated charge states changes the search/training/FDR population, so this remains benchmark-gated. |
| `unknown_modification` | `UnknownModPolicy` | `error` |  | `error` (default) or `skip` for unknown modifications. |

## predict_frag

`PredictFragConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/06_predict_frag_index_matchers.md](06_predict_frag_index_matchers.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `predictor` | `FragPredictorKind` | `native` |  |  |
| `rt_predictor` | `RtPredictorKind` | `native` |  |  |
| `charge2_from_precursor_charge` | `i32` | `2` |  | Fragment charges rule: charge 1 always; charge 2 added for precursor charge >= this threshold (docs/18_findings_and_decisions.md). Default 2: DIA-NN uses doubly-charged fragments for ~16% of charge-2 precursors' transitions, so blocking them (the old default of 3) discarded real signal. |
| `charge_by_basic_residues` | `bool` | `false` | benchmark-gated | Composition-based fragment charge cap. When true, a b/y fragment is kept at charge z only if `z <= 1 (its N-terminal amine) + (#R + #H + #K within that fragment)`, and never above the precursor charge. This supersedes the `charge2_from_precursor_charge` rule when set. Default false. Pairs with `peptidoforms.charge_by_basic_residues` for precursors; benchmark-gated because it changes the scored transition set. |
| `top_n_fragments` | `usize` | `6` |  |  |
| `ms2pip_model` | `String` | `"HCDch2"` |  | MS2PIP model name, as `ms2pip.predict_batch` takes it (`HCD`, `HCD2021`, `HCDch2`, `CID`, `CIDch2`, `TTOF5600`, `timsTOF2024`, ...). Single-charge models predict the b/y series at charge 1; charge-2 fragments then take the native heuristic, normalised per charge group. The `*ch2` models also predict the doubly charged series, and every fragment then carries a model intensity on one scale. Default `HCDch2` since 2026-09-07, on correctness grounds: with `HCD2021` (6 fragments, charge-2 fragments from precursor charge 2) the heuristic charge-2 values, each group normalised to 1.0, filled 78.6% of the fragment slots on the HYE FASTA library and the seed search found 0 confident PSMs at 1% on a real run (41.6% decoys among the top 1,000 seed scores). `HCDch2` with 12 fragments gave 19,308 confident seeds on the same run against 21,856 with the DIA-NN library. |
| `ms2pip_python` | `Option<String>` | `null` |  | Python executable for the MS2PIP sidecar (env with ms2pip + pyarrow). |
| `peptdeep_model` | `String` | `"generic"` |  | AlphaPeptDeep MS2 model, as `peptdeep.pretrained_models.ModelManager` names it: `generic`, `phospho`, `digly` or `HLA`. Default `generic`. The specialised models are trained on their own enrichment chemistry and are not a better `generic`. |
| `peptdeep_nce` | `f64` | `30.0` |  | Normalised collision energy the AlphaPeptDeep MS2 model is conditioned on. There is no neutral value. The model was trained across a range of energies and predicts a different spectrum at each, so this is an instrument setting that has to match the data, not a tuning knob: b/y ratios move with it. Default 30.0, AlphaPeptDeep's own. Read the value the acquisition used; where the vendor reports a stepped or absolute energy, convert it before writing it here. |
| `peptdeep_instrument` | `String` | `"Lumos"` |  | Instrument the AlphaPeptDeep MS2 model is conditioned on, as its own vocabulary spells it (`Lumos`, `QE`, `QEHFX`, `Exploris`, `Fusion`, `Eclipse`, `timsTOF`, `SciexTOF`, ...). Unknown names fall back to the model's default instrument inside AlphaPeptDeep rather than failing, so the worker checks the name against the installed vocabulary and refuses one it does not recognise. |
| `peptdeep_python` | `Option<String>` | `null` |  | Python executable for the AlphaPeptDeep sidecar (env with peptdeep + pyarrow). Its own environment rather than a shared one, for the same reason MS2PIP has one: AlphaPeptDeep pins the alphabase/alpharaw stack alongside torch, and a resolver conflict with DeepLC would otherwise take out the retention-time model too. |
| `deeplc_python` | `Option<String>` | `null` |  | Python executable for the DeepLC sidecar (env with deeplc + pyarrow). |
| `sidecar_script_dir` | `String` | `"scripts"` |  | Directory holding the sidecar worker scripts. |
| `defer_deeplc_to_multihead` | `bool` | `false` |  | Skip DeepLC in a FASTA library build (`rt_predictor = deeplc`) when the multi-head calibration will re-predict every row anyway. Default `false`. With `rt_predictor = deeplc` the automatic multi-head calibration runs, and it rewrites the `predicted_irt` of every standard-residue row against the run's anchors (the only rows it keeps are non-standard ones, and a FASTA digest emits none), so the library's own DeepLC pass is work whose only output is overwritten: about 19 minutes on the 9.8M-peptidoform HYE FASTA library. Nothing reads the library's iRT before the calibration: the seed is iRT-independent and only passes the column through. Set, the orchestrators (`run`, `run-experiment`, and their grouped path) write the native model's iRT as a placeholder, the library's model identity says so, and the run fails if the calibration's summary reports any row it did not re-predict (`retained_imported > 0`), because such a row would keep the placeholder. Ignored where the multi-head calibration does not run, and by the standalone `predict-frag`. Final outputs are then byte-identical to the default's; the intermediate library table, the seed's pass-through iRT column and the predict-frag report differ. Opt-in because the library table is no longer a DeepLC library, which matters to anyone who reuses it as `--lib-precursors` elsewhere. Validate by comparing `psms_scored.parquet` against a default FASTA run (on CPU, where DeepLC's base prediction is deterministic). |

## search_seed

`SearchSeedConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/07_search_seed.md](07_search_seed.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `fdr_seed` | `f64` | `0.01` |  |  |
| `fragment_tol_ppm` | `f64` | `20.0` |  |  |
| `report_psms` | `usize` | `5` |  | Max reported PSMs per spectrum (wide-window DIA, docs/07_search_seed.md). |
| `min_matched_peaks` | `usize` | `4` |  | Minimum matched fragments for a seed PSM. |
| `top_n_peaks` | `usize` | `300` |  | If > 0, probe only the `top_n_peaks` most intense peaks per MS2 scan (0 = all peaks). The seed only produces calibration anchors (RT/mass/IM), which come from abundant peptides, so this cuts the dominant per-peak index probing cost without discarding peaks from the downstream extraction artifact. Default 300; set to 0 to probe every converted peak. |
| `matcher` | `MatcherKind` | `fragindex` |  | Fragment-matcher backend (docs/06_predict_frag_index_matchers.md). Default `Fragindex`. |
| `two_pass_mass_cal` | `bool` | `false` |  | Robust two-pass fragment mass calibration (sensitivity_plan P3.1). After the first median-offset + tolerance fit, re-fit on only the deviations inside the first-pass tolerance window (rejecting outliers), giving a tighter, more robust offset + local uncertainty. Falls back to the single-pass result when too few in-window calibrants remain. Default false (single pass unchanged). |
| `mass_cal_loess` | `bool` | `false` | benchmark-gated | m/z-dependent fragment mass calibration. When true, fit a LOESS of the calibrant ppm deviation versus fragment m/z and emit a sampled correction grid to `<seed>.masscal.json`; extract then applies an m/z-interpolated offset per peak instead of the single scalar `frag_ppm_offset`. This removes any m/z-correlated curvature the flat offset leaves. Default false (scalar offset unchanged), opt-in and benchmark-gated. |

## rt_im_train

`RtImTrainConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/08_rt_im_train.md](08_rt_im_train.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `calibration_method` | `CalibrationMethod` | `loess` |  |  |
| `q_train` | `f64` | `0.01` |  |  |
| `p_rt` | `f64` | `0.95` |  | Percentile of \|obs - calibrated_pred\| residuals for the RT window. |
| `rt_window_multiplier` | `f64` | `1.0` |  |  |
| `min_seed_for_calibration` | `usize` | `50` |  |  |
| `loess_span` | `f64` | `0.3` |  | LOESS span (fraction of points in each local fit). |
| `fallback_rt_window_s` | `f64` | `120.0` |  | Fallback fixed RT window in seconds when calibration cannot be fit. |
| `finetune_deeplc` | `bool` | `false` |  | Fine-tune the DeepLC multitask model on this run's confident seed PSMs and rewrite the library's `predicted_irt` before RT calibration. Requires `predict_frag.deeplc_python` (the DeepLC interpreter). Off by default; the main use is library-input mode, where the base iRT comes from the imported library rather than a DeepLC prediction. |
| `multihead_calibration` | `Option<usize>` | `null` |  | Calibrate the DeepLC base model against this run's confident seed PSMs across this many of its best-correlating LC-setup heads, instead of fine-tuning. `null` (the default) is AUTOMATIC: `DEFAULT_MULTIHEAD_HEADS` heads when a DeepLC interpreter is available and `finetune_deeplc` is off, and nothing otherwise. `0` turns it off explicitly; any other number asks for exactly that many heads and is then a hard requirement, so a run without a `predict_frag.deeplc_python` fails rather than quietly doing without. Automatic rather than a plain number because a default that a native, Python-free run cannot satisfy would turn "no interpreter" from a supported configuration into a startup error, and because `finetune_deeplc` occupies the same slot: a configuration that asks for the fine-tune keeps it instead of being refused for a conflict it never wrote. `deeplc.predict` returns ONE of the model's 6,543 heads, the setup named by its `DEFAULT_TASK_NAME`, on that setup's gradient. The per-run LOESS then maps that column onto observed RT, and a smooth increasing curve can stretch and bend the axis but cannot reorder two peptides: whatever elution order that one setup produces survives into the calibrated result. Different chromatography reorders peptides, which is what the multitask model exists to represent, so the ordering is the part the curve cannot repair. `MultiHeadRidgeCalibration` ranks every head against this run's anchors, spline-calibrates the best ones and ridge-combines them. It never fits more head weights than half the reference, so a small anchor set degrades to fewer heads rather than overfitting. Measured on two acquisitions, six pooled runs each, at an unchanged empirical decoy fraction of 0.0100 in every arm (`docs/08_rt_im_train.md` section 4d): AIF +4.8% peptides, Astral +14.3%, protein groups +2.4% and +7.1%. The mechanism is ordering, not thresholds -- fewer candidates reach rescore and more of them are real -- and the cost is 1.4x to 1.7x wall clock, because the calibration is fitted against each run's own anchors and so cannot be shared across an experiment. |
| `finetune_epochs` | `usize` | `25` |  | DeepLC fine-tune training epochs (passed to `deeplc_finetune.py --epochs`). Early stopping with `finetune_patience` usually halts before this cap, so it is an upper bound rather than a fixed count. Only used when `finetune_deeplc`. |
| `finetune_patience` | `usize` | `10` |  | DeepLC fine-tune early-stopping patience (`--patience`): epochs without validation-loss improvement before stopping. Only used when `finetune_deeplc`. |
| `finetune_batch` | `usize` | `0` |  | DeepLC fine-tune batch size (`--batch`). 0 (default) auto-scales to the confident seed size so each epoch has >= ~30 gradient steps; a fixed large batch underfits small seeds (a ~4k-peptide reference at batch 512 is ~8 steps/epoch and never converges). Only used when `finetune_deeplc`. |
| `adaptive_rt_window` | `bool` | `false` |  | Adaptive RT window (sensitivity_plan spec 03 §3.5, backlog P3.2/P3.3): instead of one global residual-percentile half-width for every candidate, bin the calibration anchors by calibrated RT and give each candidate the LOCAL residual percentile of its RT region, clamped to `[rt_window_min_s, fallback_rt_window_s]` and scaled by `rt_window_multiplier`. A fixed window is simultaneously too wide for well-calibrated regions and too narrow for poorly-calibrated ones; this tightens clean regions (less interference) and widens noisy ones (more recall). Empty/sparse bins fall back to the global width. Default false. |
| `adaptive_rt_bins` | `usize` | `12` |  | Number of equal-width calibrated-RT bins for the adaptive window. |
| `rt_window_min_s` | `f64` | `1.0` |  | Lower clamp (seconds) for any RT half-window (the existing 1 s floor). |
| `window_holdout_frac` | `f64` | `0.0` | benchmark-gated, do not default | Size `w_rt` from HELD-OUT residuals instead of in-sample ones. A fraction of anchor peptides (`base_peptide_id % 1000 < round(frac*1000)`, so the split is deterministic and shared with `deeplc_finetune.py`) is excluded from the sizing fit and, when `finetune_deeplc` runs, from the fine-tune reference; `w_rt` is then the residual percentile of those held-out anchors against the fit they never entered. The final calibration curve still uses every anchor. In-sample sizing underestimates the tail and rewards a memorizing RT model with a window it does not deserve (measured: it inverted the 4.0.0a2/4.1.0 ranking); held-out sizing measured +0.9% peptides with DeepLC 4.1.0 and -1.5% with 4.0.0a2 on the AIF benchmark, both at 0.98% decoy, so enable it only with a generalizing RT model. 0.0 (default) keeps in-sample sizing. Mutually exclusive with `adaptive_rt_window`. Benchmark-gated; do not default on. |
| `library_irt` | `LibraryIrt` | `auto` |  | Where an imported library's `predicted_irt` comes from. `auto` (the default) re-predicts every peptidoform with the DeepLC base model when `predict_frag.deeplc_python` is configured and keeps the imported values, with a warning, when it is not; `deeplc` requires the interpreter; `library` keeps the imported values. Ignored under `finetune_deeplc` (the fine-tune re-predicts every peptidoform itself) and in FASTA mode (predict-frag already produces DeepLC predictions). Measured on the AIF benchmark with calibration only and native_tda: 10,416 peptides at 1% from DeepLC 4.1.1 base predictions against 10,015 from the DIA-NN library iRT and 10,181 from a per-run fine-tune, with `w_rt` 343 s against 632 s and 472 s (docs/08 section 4c). `run-experiment` predicts once per experiment. |
| `deeplc_predict_shards` | `usize` | `1` |  | Worker processes for the whole-library DeepLC prediction: the multi-head calibration, the base-model re-prediction under `library_irt`, and the prediction after `finetune_deeplc` (`deeplc_finetune.py --shards`). The calibration or the fine-tuned model is fitted once and handed to every process, and each process predicts a contiguous slice of the unique sequences cut at a multiple of the 100,000-sequence prediction call, so it makes the calls one process would have made. The thread budget (the engine's thread count after the DeepLC thread cap) is divided evenly, so `K` processes get `budget / K` torch threads each. `1` (the default) is one process, the behaviour before this setting existed; `0` is automatic, one process per 8 threads of the budget. A GPU always gets one process. Whether sharding pays is not established. docs/32 attributes the per-process rate (about 6,000 sequences per second) to featurisation, which is single-threaded Python. On the one CPU measured so far (an i9 desktop, docs/08, "Sharded whole-library prediction") the forward pass dominated at 8 threads or fewer and scaled with threads inside one process, so four processes of two threads were no faster than one of eight. Sharding is expected to help only where one process stops scaling with threads, as the multi-head step did on doxy (10:41 at 96 threads, 18:09 at 128); the survey's arithmetic for HYE at 8 to 12 shards is 2.5 to 4.5 minutes, unmeasured. With `K` processes at the same threads each as one process the `predicted_irt` column is bit-identical (`tests/python/test_deeplc_predict.py`). At the same engine thread count the fit is the same, but each process predicts on `budget / K` threads instead of `budget`, and torch's CPU kernels round differently at a different thread count: most rows move in the last bits, and under the multi-head calibration a few sequences at the edge of the reference range move by up to about two minutes (129 s measured, docs/13, "DeepLC thread cap"). A sharded run is therefore float-equivalent to an unsharded one, not bit-identical. Each process is its own Python process with torch and DeepLC loaded (0.57 GB resident after the model load on the desktop measured, of which the model is about 35 MB), and this step can hold the process-tree peak. Validate on two acquisitions (peptides at 1% inside the seed spread, `docs/08_rt_im_train.md` section 4d) before defaulting it on. |
| `deeplc_projection_cache` | `Option<String>` | `null` |  | Directory for DeepLC's run-independent trunk projection (`deeplc_finetune.py --projection-cache`). `null` (the default) is off. Calibrated RT is `ridge(spline_h(head_h(proj(trunk(x)))))` over the selected heads, and only the head selection, the splines and the ridge depend on a run. The projection, 64 float32 per sequence, depends on the sequence and the model alone, yet every multi-head calibration and base-model re-prediction recomputed it, which is essentially the whole of the step (10:41 of the HYE multi-head step at 96 threads). Set, the first call over a sequence list writes `<dir>/<key>/projections.npy` (the key covers the DeepLC version, the model file and the exact list; 256 B per sequence, about 1.26 GB for HYE's 4.91M) and later calls over the same list read it and evaluate only the heads they need: `rt_library_scope = per_run`, every rerun of an experiment, and the bands of `groups.rt_adaptation = once_per_run` across runs. A miss computes the projection in one process on the whole prediction-thread budget, the threads a one-process prediction gets (`deeplc_predict_shards` does not split it). Base model only: a fine-tune has no factored head and ignores it. Needs DeepLC 4.5.0 or newer, which added the factored prediction matrix it reads. On DeepLC 4.4.x (the engine's floor) the worker warns, records why in the summary, writes nothing and predicts exactly as without it. Float-equivalent, not bit-identical: the heads are evaluated in numpy from the cached factors instead of in torch. Measured with DeepLC 4.5.0 on CPU: the base-model re-prediction bit-identical on every row of the smoke library (3,820 rows) but not in general (on the 572-row test fixture 109 rows moved, by at most 1.5e-5 s); the multi-head calibration bit-identical on 3,782 of those rows and within 7.6e-6 s on the rest, and on a synthetic 572-row library with sequences outside the anchors' range 402 rows identical and 7 above 1e-3 s, the largest 3.7 s, which is the spline edge amplification a thread-count change shows too (docs/13). A hit took 0.12 s against 7.0 s for the prediction. Validate at scale as a DeepLC version change: per-row max \|delta predicted_irt\|, the selected heads, and peptides at 1% inside the seed spread on two acquisitions. |

## extract

`ExtractConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/09_extract.md](09_extract.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `windows_in_flight` | `Option<usize>` | `null` |  | Isolation windows probed per batch before the candidates no later window can touch are scored and written. `None` (the default) uses the rayon thread count capped at 16. The hit accumulator holds the windows in flight, so this sets the stage's peak almost linearly: on the HYE benchmark at 32 threads, 32 in flight is 24.65 GiB, 16 is 16.57 GiB and 8 is 12.31 GiB, with identical output (docs/27 section 3.10). It is a memory knob only: each window is probed in parallel over sub-ranges of its candidates, so a small batch still uses every thread. Set 8 or 4 on a memory-bound machine. Not a sensitivity knob. It does move the chromatogram table's parquet row group boundaries, which follow the flush batches, so two runs at different settings produce files that differ byte for byte while holding the same rows in the same order with the same values (measured on one band: 29,028,466 rows, 10.4 billion trace elements, every per-column sum equal). Compare values, not bytes, across settings. |
| `fixed_scan_window` | `usize` | `3` |  |  |
| `frag_tol_ppm` | `f64` | `20.0` |  |  |
| `prec_tol_ppm` | `f64` | `20.0` |  |  |
| `presence_min_matched` | `usize` | `3` |  | tier-(b) minimum matched fragment count. |
| `presence_min_fragments` | `usize` | `3` |  | minimum distinct fragments for acceptance. |
| `presence_min_coelution` | `usize` | `2` |  | minimum simultaneously-present fragments over the consecutive-scan run. |
| `gate_min_score` | `f64` | `0.2` |  | tier-(d) spectral-agreement gate: reject a candidate whose observed fragment intensities agree with the predicted pattern below this score. Renamed from `min_frag_corr`, which was accurate for none of the four `gate_mode` values: under the default `apex_pearson` it is an intensity correlation at ONE apex scan rather than a chromatographic co-elution correlation, and under `spectral_entropy` it is not a correlation at all. The old name is not accepted (`deny_unknown_fields`), so an old config fails loudly with the offending key named rather than silently reverting to a default. Applied symmetrically to targets and decoys, but that alone does not prove null exchangeability in chimeric DIA; validate every threshold with an independent entrapment. 0 disables. |
| `min_matched_fraction` | `f64` | `0.0` |  | tier-(c) minimum fraction of the candidate's predicted fragments that must be observed. With enough predicted fragments (top_n>=~10) this is a strong, symmetric discriminator: real peptides match a large fraction, chimeric false matches and decoys match a small fraction alike, so the target-decoy null stays valid. |
| `apex_top_fragments` | `usize` | `0` |  | Shape-aware apex selection: choose the apex scan group by the summed observed intensity of only the top-K predicted (signature) fragments, rather than all matched fragments. In chimeric DIA a bright co-eluting interferent contributing to arbitrary channels wins a max-over-all-fragments apex; restricting to the peptide's strongest predicted ions locks onto its true elution instead. 0 selects the implementation default of the top 3 predicted fragments. |
| `apex_rt_prior_s` | `f64` | `0.0` |  | Optional Gaussian RT prior on apex selection: weight each scan group by exp(-0.5*((rt - rt_cal)/sigma)^2) with sigma = this value in seconds, so a distant interferent inside a wide RT window cannot define the apex. 0 = off. |
| `apex_count_tol` | `usize` | `1` |  | Fragment-count apex: pick the scan with the most distinct matched fragments, allowing scans within `apex_count_tol` of that maximum (so a slightly-lower- count but much more intense scan can still win), then the max summed-top-3 intensity among them. Supersedes the summed-intensity apex when set. |
| `apex_count_window` | `usize` | `1` |  | Rolling-window width (in scan groups, centered, odd) for the distinct- fragment count that drives apex selection. Low-intensity fragments flicker in and out scan-to-scan; a single-scan count then spikes at noise scans and misplaces the apex. This sums the per-scan distinct-fragment count over a centered window so the apex lands in the region of *sustained* fragment presence, not an isolated flicker. A sum (not a mean) is used deliberately: edge truncation makes interior positions accumulate more, center-weighting the apex toward the RT-window centre (~= predicted RT) as a mild RT-prior; measured to beat a mean by ~+300 IDs on AIF. 1 = no smoothing (per-scan). |
| `apex_gaussian_sigma_scans` | `f64` | `0.0` | benchmark-gated | Gaussian matched-filter smoothing of the per-scan fragment-count series before apex selection, as a sigma in scan units. 0.0 (default) keeps the `apex_count_window` rolling-sum smoother unchanged. When > 0, the count series is convolved with a Gaussian kernel (radius = 3*sigma) instead, which localizes the apex more robustly than a uniform window against scan-to-scan flicker. Opt-in and benchmark-gated: it changes apex selection and therefore identifications. |
| `emit_window_grid` | `bool` | `true` |  | Emit per-fragment chromatograms on the FULL isolation-window scan grid with 0.0 where a fragment is absent (aggregating scans of the same isolation window), so the elution profile drops to zero between peaks and the features-stage boundary calling is not misled by interpolated gaps. |
| `chromatogram_schema` | `u32` | `1` |  | On-disk layout of `chromatograms.parquet` (docs/15_data_dictionary.md). `1`, the default, stores every row's retention-time axis and its whole trace, zero-filled over the candidate's window in window-grid mode. `2` stores the axis once per candidate per parquet row group (`rt_axis`) and each trace from its first to its last nonzero value (`intensity_trimmed`), with two extra columns (`trace_offset`, `trace_len`) that rebuild it. Every reader (features, quant, the pool) accepts both layouts and rebuilds the same rows bit for bit, so every table downstream of extract is byte-identical; only the chromatogram table changes (smaller, with a different content hash). Opt-in because a reader outside the engine, or an engine binary from before v2, reads only `rt` and `intensity` and stops at their absence from a v2 table; `mumdia::chromatograms::rewrite` converts a table between the layouts. The pool splices band tables of one layout only, so all bands of a grouped run share it. |
| `bucket_size` | `usize` | `8192` |  | m/z bucket size (power of two). |
| `peak_claim` | `PeakClaim` | `none` |  | How a shared observed peak's intensity is apportioned among co-isolated, co-eluting candidates that all match it (see `PeakClaim`). |
| `claim_cues` | `ClaimCues` | the `ClaimCues` section's own defaults |  | Composable claim-weight cues for `PeakClaim::CoelutionMultiCue` (modular fragment-competition framework). All default off (weight 1.0). |
| `emit_demix_features` | `bool` | `false` |  | Spectrum-centric NNLS demixing (D2, fragment-competition report). When true, at each accepted candidate's apex scan, assemble the co-isolated candidate x fragment design matrix, solve non-negative least squares (deterministic ridge-regularized), and emit non-destructive demix features (deconv_explained_frac, deconv_active, deconv_share) so the rescorer sees each candidate's interference-corrected abundance. Default false; changes no extracted intensity. |
| `demix_lambda` | `f64` | `1.0` |  | Ridge for the demix NNLS passive solve (keeps it PD/deterministic under the ~98% wide-window column collinearity). Default 1.0. |
| `demix_max_candidates` | `usize` | `64` |  | Cap on the number of co-isolated candidates (design-matrix columns) in a single demix solve, to bound compute on crowded windows. Default 64. |
| `demix_scan_stride` | `usize` | `1` |  | Scan stride for the DESTRUCTIVE `CoelutionDemix` redistribution: solve the per-scan NNLS every Nth scan and reuse the resulting candidate abundances to apportion the intervening scans (a re-solve is forced whenever a new candidate enters the co-isolated set, so accuracy is preserved where the population changes). This is the practicality lever - a full per-scan solve over the ~465k scans of a wide-window run is impractical. 1 (default) solves at every scan. Only affects `CoelutionDemix`; the non-destructive demix FEATURES are unaffected. |
| `emit_contested_features` | `bool` | `false` |  | Emit a non-destructive `contested_frac` per PSM: the fraction of a candidate's matched intensity that a co-eluting competitor claims more strongly (by the two-pass elution-profile arbitration). Does not alter the extracted intensities; feeds a rescorer feature. Forces the two-pass path. |
| `peak_claim_margin` | `f64` | `2.0` |  | Dominance factor for `CoelutionWinnerMargin`: a shared peak is claimed winner-take-all only if the top eluter's profile height is at least this multiple of the runner-up's; otherwise the peak stays shared. |
| `matcher` | `MatcherKind` | `fragindex` |  | Fragment-matcher backend (docs/06_predict_frag_index_matchers.md). Default `Fragindex`. |
| `min_coelution_run` | `usize` | `0` |  | Minimum-PSMs-per-peptide evidence filter: reject a candidate whose fragments co-elute over fewer than this many consecutive scan groups (`coelution_run`). A single/double-scan spike is a transient (likely-interferent) match; a real peptide persists across its elution. 0 disables (the `scan_window` floor still applies). This is the DIA analog of a "seen in >= N PSMs" requirement. |
| `ms1_rescue` | `bool` | `false` |  | Rescue a candidate that fails the single-scan fragment-Pearson gate when it has adequate matched fragments AND MS1 isotope-pattern support (mono + a plausible +1/mono ratio). Off by default: it relaxes acceptance, so enable it only with target-decoy/entrapment FDR validation. MS1 evidence is now computed before the gate so this can take effect. |
| `retain_top_peaks` | `usize` | `1` | diagnostic, not currently | Number of chromatographic peak hypotheses to enumerate per candidate. `K>1` writes up to K local maxima to the diagnostic `<out-psms>.peaks.parquet` sidecar. The primary PSM still contains only the selected apex, so these extra hypotheses are not currently rescored or used to improve identifications. K=1 preserves the single-apex behaviour. |
| `promote_top_peaks` | `usize` | `1` | gated | Number of chromatographic peaks PROMOTED to real feature/rescore rows per candidate (AlphaDIA plan #7, top-K). `1` (default) emits only the selected apex, so the pipeline is byte-identical. `>1` additionally emits the next strongest non-overlapping `enumerate_peaks` groups (each a full re-sliced PSM record carrying `peak_rank`), so the rescorer can pick the correct-but-not-apex peak; the selected apex stays `peak_rank = 0`. Must be `<= retain_top_peaks`. Behaviour-changing and benchmark/entrapment-gated: it changes the extracted row population, and compete/rescore must collapse per candidate so the decoy null is not K-inflated. |
| `alt_peak_min_area_frac` | `f64` | `0.10` |  | Minimum integrated area of a promoted alternate peak (rank >= 1) as a fraction of the rank-0 peak's area. Suppresses noise-level alternates. Only used when `promote_top_peaks > 1`. |
| `alt_peak_min_separation_s` | `f64` | `5.0` |  | Minimum apex-RT separation (seconds) between a promoted alternate peak and the rank-0 apex, so a near-duplicate of the selected peak is not re-emitted. Only used when `promote_top_peaks > 1`. |
| `emit_candidate_audit` | `bool` | `false` | diagnostic | Diagnostic candidate-audit: when true, extraction records, for every probed candidate, either the survivor stage-flags or the earliest `RejectionReason`, and writes `<out-psms>.audit.parquet` (spec 01 §4 / P0.3). Near-zero cost when false (no per-candidate audit allocation). Default false (production). |
| `apex_evidence_rank` | `bool` | `true` |  | Evidence-count apex selection: choose the apex scan by the NUMBER of distinct co-eluting predicted fragments present (breadth of evidence), using observed signature-ion intensity only as a sub-integer tiebreak. In wide-window DIA a single fragment m/z channel is chimeric, so the tallest scan is often a co-isolated interferent; the scan where the most of the peptide's own predicted transitions co-elute is a more reliable apex. Default `true`, on correctness grounds rather than a count: `false` keeps the legacy signature-intensity apex, whose score is 0.0 at every qualifying scan when none of the top-K predicted fragments is observed, so the strict `>` never replaces the first candidate and the apex silently becomes the LOWEST-RT qualifying scan. The rolling distinct-fragment count (`apex_count_window`) still gates which scans qualify in both modes. |
| `emit_gate_diagnostics` | `bool` | `false` | diagnostic | Emit the four gate-diagnostic scores (`gate_apex`, `gate_peak_spectral`, `gate_coelution`, `gate_spectral_entropy`) as extra `psms.parquet` columns, for the offline gate-metric comparison. Default `false` (diagnostic sidecar, like `emit_candidate_audit`): when off, neither the columns nor the extra per-candidate score computation happen, so the default chain is byte-identical. |
| `gate_mode` | `GateMode` | `apex_pearson` |  | Which spectral-agreement score the `gate_min_score` gate thresholds (sensitivity program). The legacy gate uses a single apex-scan intensity Pearson, which one chimeric scan can dominate. See `GateMode`. |
| `gate_coelution_min` | `f64` | `0.5` |  | Second threshold for `GateMode::Combined`: the co-elution score must exceed this while the peak-integrated spectral score exceeds `gate_min_score`. Requiring BOTH is more specific (rejects interferents that pass one axis). |

## extract.claim_cues

`ClaimCues` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/09_extract.md](09_extract.md).

Composable per-claimant weight cues for `PeakClaim::CoelutionMultiCue` (the modular fragment-competition framework). Each cue is label-blind (reads only observed/predicted m/z + intensity, RT, MS1) so target/decoy exchangeability is preserved, and each defaults OFF (weight 1.0) so the composite weight reduces to the plain elution-profile height. Enable cues incrementally and validate as non-destructive features before any destructive/default use.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `mz_close` | `bool` | `false` |  | Sub-tolerance m/z proximity (S3): weight a claimant by `exp(-(ppm_err/sigma)^2)`, where `ppm_err` is the signed ppm offset of the observed peak from this claimant's predicted fragment m/z. Two collided fragments share a peak only because both fall within `frag_tol`, but the observed peak sits at the true owner's m/z; the sub-tolerance offset is a novel apportionment weight (engines use ppm only as a binary gate). |
| `mz_close_sigma_ppm` | `f64` | `5.0` |  | Gaussian sigma (ppm) for the `mz_close` cue. Default 5 ppm. |
| `rt_prior` | `bool` | `false` |  | DeepLC retention-time prior (S3): weight a claimant by `exp(-(rt - rt_pred)^2 / 2 tau^2)`, where `rt_pred` is the candidate's calibrated predicted RT. A co-isolated interferent whose predicted RT is far from the current scan gets a low weight even if it briefly co-elutes, so a shared peak is apportioned toward the candidate the RT model actually places there. No-op where the predicted RT is unset (0). |
| `rt_prior_tau_s` | `f64` | `30.0` |  | Gaussian sigma (seconds) for the `rt_prior` cue. Default 30 s. |
| `ms1_support` | `bool` | `false` |  | MS1 precursor-envelope support (S4, cross-dimension): weight a claimant by whether its own precursor isotope envelope (mono + a plausible +1/mono ratio) is actually present in the nearest MS1 scan. A shift/reverse decoy has a well-defined precursor m/z but no real co-eluting MS1 precursor, so its support is noise, starving its MS2 claim via an orthogonal dimension that is nearly impossible to fake. No-op when no MS1 is provided. Down-weights (never zeroes) so a genuinely MS1-poor real peptide is not eliminated. |
| `reassign` | `bool` | `false` | gated | DESTRUCTIVE redistribution for `CoelutionMultiCue`. When true, the cue-weighted arbitration rewrites the extracted peak intensities (winner-take-all on the composite weight), instead of only emitting the apportioned/contested features. The competed evidence then feeds EVERY downstream feature (co-elution, spectral, mass-accuracy, ...), so this is the impactful form. Off by default; changes the search/FDR evidence, so it is entrapment-gated per CLAUDE.md. |
| `apportion_em_iters` | `u32` | `0` |  | Uniqueness-seeded EM apportionment (S2): number of fixed-point iterations that re-seed each candidate's per-scan elution profile from its APPORTIONED (not full) intensity before the final arbitration. The plain profile is built from full intensities, so a borrowing candidate's profile is inflated by the very peaks it borrows; re-seeding from the cue-weighted share removes that feedback, while uncontested (single-claimant) peaks contribute full intensity every iteration as an immovable anchor. 0 (default) disables EM (single-pass profile). Deterministic (fixed N); applies under `CoelutionMultiCue`. |

## features

`FeaturesConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/10_features.md](10_features.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `set` | `FeatureSet` | `minimal` |  |  |
| `emit_pin` | `bool` | `false` |  | Write the Percolator-style `.pin` text file requested by `--out-pin`. No MuMDIA stage consumes it (`rescore` builds its own PIN for the sidecars); it exists for external tooling. At 1.5M rows x 387 features it is a ~5.4 GB text write. Default false: nothing in MuMDIA reads it, which makes the write pure cost unless an external tool wants the file. Set true to get the artifact back. |
| `coelution_corr_threshold` | `f64` | `0.9` |  |  |
| `prec_tol_ppm` | `f64` | `20.0` |  |  |
| `bound_features` | `bool` | `true` |  | Restrict trace-based features (co-elution, profile, xcorr, interference, base width) to the elution peak around the apex rather than the whole extracted RT window, so they are not diluted over large RT stretches. |
| `bound_peak_fraction` | `f64` | `1.0 / 3.0 (0.333333)` | diagnostic | Peak-boundary threshold as a fraction of apex height (DIA-NN-style: descend to peak*fraction, or stop earlier at a valley below it). 1/3 matched DIA-NN's RT bounds best in the diagnostic-plot benchmark. |
| `bound_peak_grace` | `usize` | `0` |  | Grace when walking the elution-peak boundary: number of consecutive sub-threshold scans to BRIDGE before stopping. 0 (default) stops at the first scan below `bound_peak_fraction` (brittle on jagged/gappy profiles); 1 bridges a single-scan dip (DIA sampling gap / noise), giving steadier boundaries. |
| `bound_from_confident` | `bool` | `true` |  | Elution-peak boundary source. When true (default) a single set of left/right half-widths (seconds) is learned once from the confident seed PSMs (`spectrum_q <= 0.01`, target-only, the same set that anchors RT calibration / DeepLC fine-tune) and applied to EVERY candidate around its own apex. This removes per-candidate boundary manipulation so a decoy is scored over a real- peptide-width window centred on its apex. When false, each candidate detects its own peak boundary from its top-3-predicted-fragment profile (per-candidate, but noisy/manipulable for chimeric decoys; the legacy behaviour). If the seed yields < 20 confident anchors the stage logs a warning and falls back to per-candidate detection for that run. |
| `bound_confident_pct` | `f64` | `50.0` |  | Percentile (0-100) of the confident-set half-widths taken as the global left/ right elution half-width when `bound_from_confident` is true. 50 = median (typical real peak width); higher percentiles widen the shared window. |
| `ms1_precursor_features` | `bool` | `false` | benchmark-gated | Emit the MS1 apex-isotope precursor feature `ms1_isotope_height_corr` (Pearson of the observed apex isotope heights `[i0,i1,i2]` against the Poisson-averagine model). Default false (the feature is present in the battery but returns 0.0, so the vector length is unchanged in effect). It overlaps the existing `ms1_isotope_cosine_apex`, so it is opt-in and benchmark-gated rather than default-on (AlphaDIA-plan item 12). |
| `chrom_loaders` | `usize` | `3` |  | Chromatogram decode threads in the main feature pass. The pass decodes the chromatogram table one chunk at a time while the features of the chunk before are computed; with one loader the whole decode ran on a single core, which bound the stage whenever decoding a chunk took longer than computing one (measured on an 8-12-mer immunopeptidomics run before the decode overlapped the computation: 3.7 of a 4-minute stage were the load). Each loader reads its own chunk from that chunk's row span, and the computation takes the chunks in table order, so the chunks, every feature value and the features table bytes are the same at every setting; only the time and the memory move. The pass holds up to `chrom_loaders + 1` decoded chunks (0.92 GiB of traces each at the HYE benchmark shape, docs/27 section 3.4), where one loader held two. The value is an upper bound: a pass never runs more loaders than `--threads` (the engine's thread pool) or than it has chunks, so `--threads 1` decodes on one loader as before. Loaders beyond each pass's first come from a process-wide pool of four, so concurrent bands or runs (`groups.parallel`, `experiment.parallel_runs`) share that pool instead of multiplying it. Default 3; `1` restores the single loader and `0` is read as `1`. A memory knob and a speed knob, not a sensitivity knob. |

## compete

`CompeteConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/11_compete_rescore_fdr.md](11_compete_rescore_fdr.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `group_by` | `CompeteGroupBy` | `peptidoform_charge` |  | Competition grouping: `precursor` collapses charge/modification siblings separately within each target/decoy label; targets and decoys therefore do not compete directly. `apex` also groups by rounded apex RT; `peptidoform_charge` keeps each peptidoform+charge as its own group (precursor-level, as DIA-NN/Spectronaut report), so sibling charges of one peptide are not collapsed. |
| `apex_rt_tolerance_s` | `f64` | `5.0` |  |  |
| `mode` | `CompetitionMode` | `winner_take_all` |  | How within-group competition resolves (sensitivity program, spec 04 §6 / P2.4). `winner_take_all` = legacy (keep only the top `prelim_score` per group). The other modes preserve more candidate evidence for the rescorer/ FDR to arbitrate. Default `winner_take_all` (unchanged behaviour). |
| `margin` | `f64` | `0.0` | gated | Score margin (in `prelim_score` units) required to remove a loser under `margin_gated`. A loser closer than this to the winner is kept. |
| `unique_evidence_min_fragments` | `usize` | `2` |  | Minimum distinct unique-fragment count a loser must have to survive under `unique_evidence` (needs the `unique_fragment_count` feature; falls back to winner-take-all when the column is absent). |
| `emit_competition_audit` | `bool` | `false` | diagnostic | Diagnostic: when true, write `<out>.compete_audit.parquet` recording every removed candidate with its group, winner, scores, and removal reason. |

## rescore

`RescoreConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/11_compete_rescore_fdr.md](11_compete_rescore_fdr.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `classifier` | `RescorerKind` | `native_tda` |  |  |
| `folds` | `usize` | `3` |  |  |
| `train_fdr` | `f64` | `0.01` |  |  |
| `num_iter` | `usize` | `10` |  | number of semi-supervised iterations for the native rescorer. |
| `max_feature_matrix_gib` | `f64` | `0.0` |  | Refuse a rescore whose in-memory feature matrix would exceed this many GiB. 0 (default) means no ceiling, which is the previous behaviour. The matrix is a flat f32 `FeatureMatrix`: four bytes per value. Nothing in the workspace estimates or checks available memory (there is deliberately no `sysinfo` dependency), so an experiment-wide rescore over enough runs was simply killed by the OS after however long it took to get there. `native_tda` fits its folds one at a time and holds one standardised training copy, `(folds - 1) / folds` of the matrix, so its peak is `1 + (folds - 1) / folds` times this figure: 1.67x at the default 3 folds, 1.80x at 5, approaching 2x and never above it. It does not grow with `folds`. Setting a ceiling converts that into an error at startup, naming the estimate and the two ways out. It is not a batching implementation: sub-batching changes which PSMs share a pooled `q_value`, so it is the operator's decision, not a silent one. |
| `python` | `Option<String>` | `null` |  |  |
| `percolator_bin` | `Option<String>` | `null` |  | Path to an external `percolator` executable. Parsed and never read: no stage launches percolator, and `RescorerKind` has no variant that would. It is the only silently inert config field in the tree, since the three MBR ones warn (see `validate`). Kept rather than deleted because the external-percolator path is still intended; `validate` now warns when it is set. |
| `entrapment_marker` | `Option<String>` | `null` |  | Protein-accession substring marking spike-in (entrapment) negatives, e.g. "_HUMAN". Required when `classifier = entrapment`; PSMs whose protein contains it are the empirical false population. |
| `entrapment_exclude` | `Option<String>` | `null` |  | If a protein also contains this substring it is NOT counted as entrapment (the sample's own species, e.g. "_ECOLI"): shared peptides then count as real targets. `None` = the marker alone decides. |
| `entrapment_contaminant_markers` | `Vec<String>` | `[]` |  | Protein substrings marking genuine contaminants inside the spike-in proteome (e.g. "KRT", "ALBU", keratin/albumin entry-name tokens). A PSM matching `entrapment_marker` but also one of these is treated as a REAL target, not an entrapment negative: such peptides are truly present (handling contaminants) so using them as negatives mislabels real signal and inflates the estimated FDR. Empty = every spike-in hit is a negative. |
| `entrapment_ratio` | `f64` | `1.0` |  | N_real_lib / N_entrap_lib. Scales the entrapment FDR estimate so it is unbiased when the spike-in library differs in size from the real one. |
| `strict` | `bool` | `true` |  | When true, any sidecar/classifier failure or misconfiguration (Mokapot or entrapment sidecar error, unwired percolator, entrapment mode with no entrapment PSMs) is a hard error instead of a silent fall back to the native rescorer. Default true so a named scientific workflow cannot silently execute a different model; set false only for explicit legacy compatibility. |
| `handoff` | `Handoff` | `parquet` |  | How the feature matrix reaches a sidecar rescorer. See `Handoff`. Defaults to `parquet`, which applies to nn_torch only; mokapot and entrapment sidecars always receive the tab-separated PIN. |
| `features` | `Option<Vec<String>>` | `null` |  | Restrict the classifier's input to these feature columns, by name. Absent (the default) falls through to `feature_preset`. The restriction is a projection, not a reordering: the columns keep the order of the feature schema, only those named are read out of the competed table, and the matrix, the sidecar handoff and the training all shrink with the list. Feature selection is a memory and I/O lever, not a speed one (docs/28 section 7), and any list must clear the sensitivity gate before it becomes a default. Mutually exclusive with `RescoreConfig::features_file`. Every name must exist in the competed table's schema; a missing one is an error, never a silent drop. |
| `features_file` | `Option<String>` | `null` |  | The same restriction, read from a file with one feature name per line (blank lines and `#` comments ignored), which is how a 100+ name list stays readable. |
| `feature_preset` | `FeaturePreset` | `all` |  | Named feature list used when neither `features` nor `features_file` is set. `all` is every feature the competed table carries. `compact` is the 114-name list of docs/28 section 12 (`bench/feature_selection/fs_union75_dedup.txt`, embedded in the binary), which with the hard-negative training recipe reproduced the full Extended set within seed noise on three pools (HYE A01 +1.2%, AIF -0.2%, entrapment +4.9%, spike-in FDP unchanged) at 3.4x less rescore memory. Preset names the table lacks are skipped with a log line rather than an error, so a preset tolerates a smaller `features.set`; the intersection must not be empty. Explicit lists stay strict. Default `all`: the projection is a memory lever (3.4x smaller rescore matrix), not a sensitivity one, and it cost 1.2% on the held-out HYE B01 pool under the default training (+0.2% / -0.1% / +1.5% on A01 / AIF / entrapment), so it is the option for pooled rescoring on small machines (docs/28 section 21), not the default. |
| `train_neg_ratio` | `f64` | `2.0` |  | Cap the decoys the sidecar TRAINS on at this multiple of the targets it selected that iteration; 0 trains on every decoy, which is about 19:1 on a DIA pool and is where the rescore spends its time. This thins gradient steps only. Selection, scoring, target-decoy competition and q-values still run over the full pool, so the cap cannot loosen the q threshold; what it can move is the learned boundary. Default 2 since 2026-09-16, measured against the previous 3 with three seeds on two pools and on the entrapment pool: Astral six-run pool 116,711 against 116,309 peptides (+0.35%), HYE B01 63,096 against 63,004 (+0.15%), AIF spike-in library +0.56% real peptides at an empirical FDP of 1.025% against 1.044%, at 16% less rescore wall. `1` with `margin` selection is still the recipe that loses 10% on the entrapment pool (docs/28). |
| `train_neg_select` | `NegSelect` | `hybrid` |  | Which decoys survive `RescoreConfig::train_neg_ratio`. See `NegSelect`. |
| `train_subsample` | `f64` | `0.0` |  | Stratified thinning of whatever survived the cap: a fraction in (0, 1], or a row cap when > 1. Positives and negatives are thinned by the same factor, so the class balance is unchanged. 0 (the default) keeps every row. |
| `train_warm_epochs` | `usize` | `5` |  | Reuse the previous iteration's weights and optimiser state, running this many epochs from the second self-training iteration on instead of a full fresh fit. 0 (the default) refits from scratch every iteration, which is 25 epochs x 10 iterations x 3 folds of the whole training set. |
| `train_margin_frac` | `f64` | `0.5` |  | Under `train_neg_select = hybrid`, the share of the negative budget taken from the margin (highest-scoring decoys); the rest is sampled at random. Default 0.5. |
| `seeds` | `usize` | `1` |  | Independent self-training passes whose out-of-fold scores are rank-averaged. 1 (the default) is a single pass. 3 was the one knob positive on every pool of the seeded sweep (docs/28 section 17), at three times the training cost. |

## quant

`QuantConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `q_threshold` | `f64` | `0.01` |  | Peptide-level q-value cutoff for inclusion. |
| `top_n_fragments` | `usize` | `3` |  | Number of top fragments summed per peptidoform. |
| `top_n_peptides` | `usize` | `3` |  | Number of top peptides summed per protein group (TopNSum). |
| `rollup` | `RollupMethod` | `top_n_sum` |  |  |
| `bound_peak` | `bool` | `true` |  | Integrate each fragment only over the detected elution-peak window rather than the whole chromatogram. The window is found from the summed XIC apex. |
| `peak_fraction` | `f64` | `1.0 / 6.0 (0.166667)` |  | Descent threshold for the peak-window walk: stop where the summed XIC drops below `peak_fraction` * apex height (1/6 expanded from the 1/3 feature bound). |
| `peak_grace` | `usize` | `1` |  | Zig-zag grace: bridge up to this many consecutive sub-threshold scans during the peak-window walk; the boundary triggers on `peak_grace + 1` consecutive sub-threshold scans (1 = stop on 2 consecutive misses). |
| `peak_window_mode` | `PeakWindowMode` | `per_candidate` |  | Per-candidate window vs a consensus width derived from confident peptides. |
| `reliable_q` | `f64` | `0.001` |  | Peptide q-value cutoff defining the "confident" set that calibrates the consensus half-widths (Consensus mode only). Tighter than `q_threshold`. |
| `q_filter` | `QuantQColumn` | `peptide_q` |  | Which q-value column to filter candidates on (`peptide_q` default; `precursor_q` is single-run only; use `run_psm_q` for per-run slices of an experiment-wide rescore). See `QuantQColumn`. |
| `interference_envelope` | `bool` | `false` | benchmark-gated | Apply an apex-outward interference-correction envelope to each fragment trace before integrating its area, stripping co-eluting interference in the peak wings. Off by default (identity on a clean peak). Opt-in and benchmark-gated: it changes reported quantities. |
| `fragment_selection` | `FragmentSelection` | `observed_area` | benchmark-gated | Which fragments enter the top-N sum. `observed_area` (default, legacy) ranks by the integrated area itself, which preferentially selects interfered fragments (their areas are inflated) and so varies run to run. `predicted` ranks by the library (predicted or empirical) fragment intensity, a per-precursor constant, so every run sums the same fragments. Astral HYE 2026-08-26: CV 0.163 -> 0.112 on 6/6 ions at top-3. Benchmark-gated. |
| `fixed_scan_halfwidth` | `usize` | `0` |  | When > 0, integrate each fragment over the `2k+1` scans centred on the identification apex instead of the descent-walk window (`bound_peak` window ignored; falls back to it when the apex is unknown). A fixed narrow window is far less sensitive to interference in the peak wings than the walked bounds. 0 (default) = off. |
| `baseline_subtract` | `bool` | `false` | benchmark-gated | Subtract a per-fragment local background before integrating (fixed-scan window only). The background is the `baseline_quantile` quantile of the intensities in the two flanks (`baseline_flank_scans` samples on each side of the integration window); window intensities are clipped at zero after subtraction. Targets the additive floor that compresses ratios in the low-abundance condition. Off by default; benchmark-gated. |
| `baseline_flank_scans` | `usize` | `12` |  | Flank length (samples per side) used to estimate the background. |
| `baseline_quantile` | `f64` | `0.25` |  | Quantile of the flank intensities taken as the background level. |
| `fixed_window_s` | `f64` | `0.0` |  | When > 0, integrate each fragment over the samples within `fixed_window_s` seconds of the identification apex (instrument-independent alternative to `fixed_scan_halfwidth`, which it overrides). 0 (default) = off. |

## mbr

`MbrConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `strategy` | `MbrStrategy` | `none` |  |  |
| `q_anchor` | `f64` | `0.01` |  | q-value for a precursor to become a cross-run anchor (validated at 0.01). |
| `min_anchor_runs` | `usize` | `2` |  | Minimum number of OTHER runs a precursor must be confident in to transfer. |
| `q_transfer` | `f64` | `0.01` |  | Accept threshold for a transferred identification's transfer q-value. |
| `rt_window_s` | `f64` | `20.0` | benchmark-gated, not yet wired | NOT YET WIRED. Transfer RT half-window (seconds) around the cross-run-predicted RT. The M2 leave-target-out residual was ~17 s at p95, ~15x tighter than the search window, which is where this default comes from -- but no code reads this field yet, so setting it has no effect. Kept as the recorded design value for the MBR transfer tier; `validate()` warns if it is changed from the default. See CLAUDE.md, "MBR transfer/re-extraction remains benchmark-gated". |
| `decoy_transfer` | `DecoyTransfer` | `permuted_rt` | not yet wired | NOT YET WIRED. Which decoy-transfer null would estimate the false-transfer rate (M4). No code reads this field yet; `validate()` warns if it is changed. |
| `consensus_corr_min` | `f64` | `0.0` |  | Minimum correlation of the observed fragment pattern to the empirical consensus for a transfer to be accepted (interference guard; 0 disables). |
| `requant_all` | `bool` | `false` | not yet wired | NOT YET WIRED. Would requantify already-identified precursors too (fill the matrix), not only transferred ones, under `strategy = Full`. No code reads this field yet; `validate()` warns if it is changed. |
| `python` | `Option<String>` | `null` |  | Python interpreter for the `mbr_worker.py` sidecar (pandas/pyarrow/numpy; e.g. the `py312_mumdia` env). Required when `strategy != None`. |

## experiment

`ExperimentConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/01_overview_and_dataflow.md](01_overview_and_dataflow.md).

Options for the experiment-wide orchestrator (`mumdia run-experiment`).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `parallel_runs` | `usize` | `1` |  | How many per-run search chains to execute concurrently. 1 (default) is strictly sequential, i.e. the historical behaviour. Runs are independent, so raising this scales nearly linearly in wall time, but EACH concurrent run holds its own extraction working set (tens of GB on a large library), so the practical ceiling is memory, not cores. Raise it deliberately after checking peak RSS for a single run; 2-4 is a reasonable start on a large-memory machine. Results are unaffected: chunks are processed in index order and completion order never reaches the output. |
| `rt_library_scope` | `RtLibraryScope` | `first_run_only` |  | How often the library's retention times are adapted to a run: once on the first run and reused (`first_run_only`, the default) or separately for every run (`per_run`). Governs whichever adaptation is active -- `rt_im_train.finetune_deeplc` or `rt_im_train.multihead_calibration` -- because they are the same shape of work: one full re-prediction of the library against that run's confident seed PSMs, which on a 9.4M-row library is the most expensive step in the experiment. Each run then fits its own LOESS on top of whichever library it was given, and that per-run fit is what absorbs chromatographic drift. Accepts the old name `finetune_scope`, which is what it was called when only the fine-tune could be shared. `first_run_only` assumes the runs share an elution ORDER, which replicate injections on one LC method do. A per-run LOESS can stretch and bend the axis but cannot reorder two peptides, so a batch that genuinely reorders -- different gradients, different columns, a method change part-way -- wants `per_run`, and so does a long batch where drift accumulates (see the measured cost above). |
| `overlap_front_threads` | `usize` | `0` |  | Threads given to converting and seeding runs 2..N while run 1 adapts the library's retention times, under `rt_library_scope = first_run_only`. `0` (the default) runs them after run 1, as before. Runs 2..N convert their spectra and seed on the base library (the seed is iRT-independent), so nothing of theirs waits for run 1's adapted library until rt-im-train. Set to `N`, those fronts run on a pool of `N` threads while run 1's DeepLC sidecar gets the remaining `threads - N` (disjoint budgets), and every run's rest follows once run 1 has finished. On the six-file HYE Astral experiment a front is convert 1.9-2.5 min plus seed 0.4 min per file, against a first-run multi-head step of 11.7 min, so up to about 12 minutes of fronts fit behind it. Ungrouped runs only: a grouped run seeds per band inside its band loop, and its adaptation sits between those band seeds and its extract. The fronts' outputs are byte-identical; run 1's DeepLC predicts on `threads - N` torch threads instead of `threads`, which moves the adapted library in the last bits unless the DeepLC thread cap binds both counts to the same number (on an SMT host with `N` below the logical-minus-physical core count it does). Float-equivalent, hence opt-in. The fronts keep the ungrouped experiment's phase order: every conversion first, then one seed library and fragment index for all of their seeds (run 1 seeds on its own load before the overlap starts). That library is held beside the DeepLC worker while the fronts seed, which is where the experiment's peak can sit. Not measured at scale. |

## groups

`GroupsConfig` (rust/mumdia/crates/mumdia-core/src/config.rs).

Searching a run one isolation-window group at a time. A group of isolation windows can only select precursors whose m/z lies in the group's band, so its seed, calibration, extract, features and compete need only that band of the library (`Library::load_with_fragment_offset`): the library, the hit accumulator and the accepted rows are all one band's worth instead of the whole run's, which is what bounds the memory of a search against a library of 10^8 precursors. Only rescore, quant and report see everything, after the group artifacts are pooled with library-wide ids. The groups run one after another in this process; `docs/33_window_groups.md` has the layout and the measurements.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `window_groups` | `usize` | `1` |  | Number of window groups. `1` (the default) is the ordinary single-library search. Groups are contiguous bands of isolation windows balanced by the number of library precursors they select, read from the precursor table's row-group statistics. |
| `calibration` | `GroupCalibration` | `global` |  | Anchors for the RT calibration of each group; see `GroupCalibration`. |
| `parallel` | `usize` | `1` |  | Bands searched at the same time inside one run. `1` (the default) is one band at a time, which is what bounds the memory: each band in flight holds its own extraction working set, so the peak is this many bands' worth. Raise it to fill a large machine, after checking one band's peak RSS: on a 203M-precursor library at 63 bands the largest band took 39 GB and the median far less. Results do not depend on it; bands are independent and their artifacts are pooled in band order either way. The bands go through a bounded queue: this many workers each take the next band as soon as their current one is done, most expensive first (estimated precursors times MS2 peaks of the band's windows for the seed and extract, accepted rows for features and compete), rather than in fixed chunks that waited for their slowest band. It must stay below the thread count: a band in flight parks one worker on its accumulation channel, so as many bands as there are threads leaves nothing to do the probing and the run deadlocks. A larger value is clamped to `threads - 1` with a warning rather than hanging. |
| `rt_adaptation` | `GroupRtAdaptation` | `per_band` |  | How often the library's retention times are adapted under `calibration = global`: `per_band` (the default) runs one DeepLC sidecar per band, `once_per_run` one per run over the union of the bands. `per_group` calibration always adapts per band. Each per-band sidecar starts an interpreter, imports torch and DeepLC, reads the pooled seed, refits the same heads on the same anchors (head 2503 in every band of the HYE sweep) and predicts every sequence of its band, so a sequence whose charge states fall in two bands is predicted twice (10.9M HYE rows are 4.91M unique sequences). On HYE Astral the multi-head step took about 13 min unbanded and 19-24 min at 2-16 bands. `once_per_run` fits once, predicts the union once and writes each band's table under the name a per-band run gives it (`groups/gNN/lib_precursors_multihead.parquet` or `lib_precursors_deeplc.parquet`), so the shared-band reuse of later runs and the seed refresh are unchanged. Under `run-experiment` with the multi-head calibration off, the library is re-predicted once for the experiment, and the bands then keep those values instead of each band of each run re-predicting them. Float-equivalent, not bit-identical: a sequence is predicted in different company, and torch's CPU kernels round by batch. On a synthetic library, one call over contiguous bands writes exactly the whole-library column band by band (`tests/python/test_deeplc_predict.py`). Validate on two acquisitions (peptides at 1% inside the seed spread, the per-band max \|delta predicted_irt\| and the selected heads) before defaulting it on. |
| `balance` | `GroupBalance` | `precursors` |  | What the band plan balances: `precursors` (the default), the estimated library precursors per band, or `cost`, per window the precursors it selects times the MS2 peaks of its scans. Band cost follows spectral density more than precursor count: on the immunopeptidomics search two bands of 2.98M and 3.03M precursors took 42 s and 460 s, and at 81-94 bands the slowest windows (418-460 s) set the floor of the run. The queue already starts the most expensive bands first whatever this says; `cost` also moves the cuts, so that no band is several times the work of the others. Output-changing, hence opt-in: the cuts decide which candidates sit at a band edge, which the overlap deduplication and the edge candidates' neighbours depend on, and the pooled row order the classifier sees. Validate like a band-count change (docs/33 section 8): peptides at 1% inside the seed spread against `precursors`, and the per-band wall times, on two acquisitions. Note that the MS2 is decoded before the plan under either setting. |
| `delete_band_intermediates` | `bool` | `false` |  | Delete each band's `psms_extracted.parquet` and `features.parquet` (with their reports and schema companions, and `run.pin` where one was written) once the pool is written. Default `false`. Disk only: no stage reads them after pooling. The features are carried by the competed table and the extracted table's one reader, the candidate audit, reads the pooled copy. On the immunopeptidomics runs the band features alone were 55 GB per run, in an experiment that wrote about 2.7 TB of artifacts. The manifest keeps their records, and the band directories can no longer be re-featured; the chromatograms and competed tables that `mumdia pool --groups-dir` re-pools from are kept. |
| `pool_competed` | `bool` | `true` |  | Write the run's pooled `psms_competed.parquet`. Default `true`. With `false`, rescore reads the bands' own competed tables in band order, each with the run's `source`, and the pooled copy is not written: one full write and read of the run's widest artifact less (about 83 GB per run on the immunopeptidomics experiment). This happens only where it cannot change a result: the bands' library row spans must be disjoint, so no candidate was searched in two bands and there is no overlap duplicate to drop, and neither the candidate audit (`extract.emit_candidate_audit`) nor match-between-runs (`mbr.strategy`), which read the pooled table, may be on. Otherwise the table is pooled as with `true`, and the log says why. `psms_scored.parquet` is byte-identical either way; what changes is the artifact set. The run's manifest then has no pooled `psms_competed` record, the scored table's report lists the band tables under `competed_inputs` with their `competed_sources`, and a later standalone `mumdia rescore` or audit needs the table rebuilt first with `mumdia pool --groups-dir`, which the band tables allow. Validate on a grouped run by comparing `psms_scored.parquet` byte for byte against a run with the default. |
| `pool_chromatograms` | `bool` | `true` |  | Write the run's pooled `chromatograms.parquet`. Default `true`. With `false`, quant reads the bands' own chromatogram tables in band order, dropping from each the candidates the pool's overlap dedup gave to another band, and the pooled copy is not written: one full splice write, hash and read of the run's largest artifact less (about 68 GB per run on the immunopeptidomics experiment). The pool writes those loser sets to `groups/overlap_losers.parquet` (`band`, `candidate_id`, with the band tables it belongs to named in its footer), so a later `mumdia quant --chromatograms <band tables> --overlap-losers <that file>` reads the same rows, and refuses band tables that are not the named ones in their order. Unlike `pool_competed` this holds for overlapping bands as well, including an overlap candidate compete deleted in one band (the losers are found in the chromatogram tables themselves), and nothing but quant reads the pooled table (the candidate audit and match-between-runs do not). The quant tables are byte-identical either way; what changes is the artifact set. There is no pooled `chromatograms.parquet` (one an earlier run left in the directory is removed) and no manifest record for it, `overlap_losers.parquet` is written and recorded instead, and the quant report's `chromatograms` lists the band tables with `chromatogram_dropped_candidates`. The band chromatogram tables are then the run's only chromatograms, so the band directories are no longer disposable once the run is accepted: deleting them loses re-quantification. `mumdia pool --groups-dir` rebuilds the pooled table from them. Validate on a grouped run by comparing `peptide_quant.parquet`, `protein_group_quant.parquet` and `fragment_quant.parquet` byte for byte against a run with the default. |

## peptidoforms.fixed_mods[] / peptidoforms.variable_mods[]

`ResidueMod` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md).

Element type of `peptidoforms.fixed_mods` and `peptidoforms.variable_mods`. Each element is a JSON object with these keys; the list default is on the owning field.

`ResidueMod` has no `impl Default` block, so an element must set every key.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `residue` | `char` | none: must be set |  | Target residue; `*` for any / terminal handled separately in MVP. |
| `name` | `String` | none: must be set |  | UniMod name. |

## Named profiles

`mumdia run --profile NAME` applies a named override set on top of
`--config` and the defaults, from `Config::apply_profile`. The overrides:

| Profile | Overrides |
|---|---|
| `dia` | `features.set` = `extended`; `extract.apex_count_window` = `5`; `extract.apex_rt_prior_s` = `120.0` |

A profile is applied after the config file is parsed, so it wins over a
value the config file set for the same field.

## Enumerations

Accepted values for every enum-typed field above, with the serde spelling a
config file must use. The default variant is marked. Sorted by type name.

### `CalibrationMethod`

(rust/mumdia/crates/mumdia-core/src/config.rs)

| Value | Default | Description |
|---|---|---|
| `loess` | yes |  |
| `linear` |  |  |
| `none` |  |  |

### `CompeteGroupBy`

(rust/mumdia/crates/mumdia-core/src/config.rs)

| Value | Default | Description |
|---|---|---|
| `base_peptide` |  | One winner per stripped base peptide, per label: every charge state AND every modification variant of one peptide collapses to a single winner before FDR (`compete.rs` keys the group on `base_peptide_id`, which comes from the stripped sequence). The default until 2026-09-06 and renamed from `precursor`, which it is not; the old name is not accepted, so an old config fails loudly rather than silently changing the competition unit. Opt in to it for a peptide-level population; never use it for a PTM search, where it deletes the modified form whenever an unmodified sibling scores higher. |
| `apex` |  |  |
| `peptidoform_charge` |  | Precursor-level, the default: every distinct peptidoform + charge is its own group, so sibling charge states and modforms of one peptide are kept and compete only against their own alternative peaks; the label stays in the key so a target never competes against its own decoy. This is the unit DIA-NN and Spectronaut report at, and the key every benchmark of docs/28 ran under: entrapment (spike-in FDP 0.48-0.64%, flat), HYE and AIF. Measured against `base_peptide` on a modification-rich library it removed 0 instead of 46.6% of the extracted candidates at an unchanged peptide count, with 1.174 precursors per peptide (DIA-NN about 1.126). |

### `CompetitionMode`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Within-group competition resolution (spec 04 §6). Only `WinnerTakeAll` removes candidates unconditionally; the others preserve candidates the rescorer can still discriminate, which is the sensitivity program's central principle ("preserve candidate evidence until the workflow can make a calibrated decision"). Target/decoy labels remain part of the competition key in every mode, so a target never competes against its own decoy (the null is preserved).

| Value | Default | Description |
|---|---|---|
| `winner_take_all` | yes | Legacy: keep only the highest `prelim_score` candidate per group. |
| `none` |  | Keep every candidate (no within-group removal); FDR handles ambiguity. |
| `features_only` |  | Keep every candidate; conflict/contested features (added upstream) carry the interference signal into rescoring. Same retained set as `None`; the name documents intent for the experiment matrix. |
| `unique_evidence` |  | Keep a loser when it has enough independent evidence (`unique_fragment_count >= unique_evidence_min_fragments`); otherwise remove it (winner-take-all fallback). |
| `margin_gated` |  | Remove a loser only when `winner_score - loser_score >= margin`; otherwise keep it. Conservative removal for the low-FDR region. |

### `DecoyStrategy`

(rust/mumdia/crates/mumdia-core/src/config.rs)

| Value | Default | Description |
|---|---|---|
| `reverse` | yes | Reverse the sequence keeping the C-terminal residue fixed. Documented, clean-room default for MVP (docs/14_build_test_deploy_gotchas.md). No borrowed map. |
| `scramble` |  | Deterministic seeded shuffle of the interior residues. |
| `diann_shift` |  | DIA-NN terminal-residue fragment m/z shift. Deferred: license-checked addition (docs/14_build_test_deploy_gotchas.md), not part of MVP. |
| `none` |  |  |

### `DecoyTransfer`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Decoy-transfer null for the MBR false-transfer FDR (M4). `ReverseSequence` transfers reverse/scramble decoys at the same expected RT; `PermutedRt` transfers real precursors to a decoupled (wrong) expected RT; `Both` combines them. The prototype's shuffled-RT null gave a ~0.6% in-window false rate vs 66.6% true (113x separation), so the transfer q-value is well-calibrated.

| Value | Default | Description |
|---|---|---|
| `permuted_rt` | yes |  |
| `reverse_sequence` |  |  |
| `both` |  |  |

### `Enzyme`

(rust/mumdia/crates/mumdia-core/src/config.rs)

| Value | Default | Description |
|---|---|---|
| `trypsin_p` | yes | Trypsin/P: cut after K or R (including before P). |
| `trypsin` |  | Classic trypsin: cut after K or R but not before P. |

### `FeaturePreset`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Named feature list for `RescoreConfig::feature_preset`.

| Value | Default | Description |
|---|---|---|
| `all` | yes | Every feature column of the competed table. |
| `compact` |  | The 114-feature list of docs/28 section 12, embedded in the engine. |

### `FeatureSet`

(rust/mumdia/crates/mumdia-core/src/config.rs)

| Value | Default | Description |
|---|---|---|
| `minimal` | yes | MVP feature set (docs/10_features.md). |
| `rich` |  |  |
| `extended` |  | Minimal + Rich + the extended battery (DIA-NN / OpenSWATH / AlphaDIA / MS2Rescore / OktoberFest analogs + novel families) from the per-family modules in `stages/features/`. Superset, opt-in; the classifier picks the signal it can use (esp. under the nonlinear `Entrapment` rescorer). |

### `FragPredictorKind`

(rust/mumdia/crates/mumdia-core/src/config.rs)

| Value | Default | Description |
|---|---|---|
| `native` | yes | Native heuristic intensity model (no Python). MVP default. |
| `ms2pip` |  | MS2PIP Python sidecar (docs/13_sidecars.md). |
| `peptdeep` |  | AlphaPeptDeep Python sidecar (`peptdeep_worker.py`, docs/13_sidecars.md). A transformer intensity model conditioned on collision energy and instrument, which MS2PIP is not. It supplies intensities for the same `(ion_type, ordinal, charge)` triples the engine already enumerates, so it changes the numbers on the fragments rather than which fragments exist: the engine generates b and y only (`mumdia_core::mass::IonType`), and AlphaPeptDeep's a/c/x/z and neutral-loss series have nowhere to go until that changes. Benchmark-gated. It is opt-in until there is entrapment plus a second acquisition, and a seed-PSM count alone does not promote it. |

### `FragmentSelection`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Fragment ranking for the quant top-N sum. See `QuantConfig::fragment_selection`.

| Value | Default | Description |
|---|---|---|
| `observed_area` | yes | Rank fragments by their own integrated area (legacy). |
| `predicted` |  | Rank fragments by library intensity (`predicted_intensity` in the chromatogram table). |

### `GateMode`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Spectral-agreement score the extraction acceptance gate (`gate_min_score`) thresholds. All are computed at the gate from data already in hand.

| Value | Default | Description |
|---|---|---|
| `apex_pearson` | yes | Legacy: Pearson of observed-vs-predicted fragment intensities at the single apex scan. One chimeric scan can dominate it. |
| `peak_spectral` |  | Pearson of the PEAK-INTEGRATED observed spectrum (each fragment summed over the elution-peak scans) vs predicted intensities. Averages out a single interfered scan; the standard library-dot-product measure. |
| `spectral_entropy` |  | Li spectral-entropy similarity of the sqrt-transformed apex-scan observed vs predicted intensities (`spectral_entropy_similarity_sqrt`). The full-feature gate search (all ~379 features, target-vs-decoy) found this the single best gate discriminator: AUC 0.826 / matched-pool recall 69.8%, versus apex Pearson's 0.781 / 64.5%. Same inputs as `ApexPearson`, better separation. |
| `coelution` |  | Predicted-intensity-weighted mean CO-ELUTION correlation of each matched fragment's XIC to the signature reference over the elution peak (temporal agreement, orthogonal to intensity agreement). |
| `combined` |  | Require BOTH: peak-integrated spectral Pearson >= `gate_min_score` AND the co-elution score >= `gate_coelution_min`. More specific (an interferent passing one axis is still rejected), for a cleaner FDR pool. |

### `GroupBalance`

(rust/mumdia/crates/mumdia-core/src/config.rs)

What a grouped run's band plan balances.

| Value | Default | Description |
|---|---|---|
| `precursors` | yes | The estimated library precursors each band selects. The behaviour before this setting existed. |
| `cost` |  | The estimated search cost: per window, the precursors it selects times the MS2 peaks its scans carry. |

### `GroupCalibration`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Which anchors the retention-time calibration of a window group is fitted on.

| Value | Default | Description |
|---|---|---|
| `global` | yes | The confident seed PSMs of every group, pooled (q re-estimated on the union), so each group's LOESS and multi-head fit see the whole run's anchors. The default: a group holds a fraction of the anchors, and the fit quality is what sets the RT window that the extract of every group then pays for. |
| `per_group` |  | Each group calibrates on its own seeds only. Cheaper by one pooling pass and fully independent per group; kept for the comparison, not as a recommendation. |

### `GroupRtAdaptation`

(rust/mumdia/crates/mumdia-core/src/config.rs)

How often a grouped run adapts the library's retention times (the multi-head calibration or the base-model re-prediction) under `groups.calibration = global`.

| Value | Default | Description |
|---|---|---|
| `per_band` | yes | One DeepLC sidecar per band, each fitting the pooled anchors and predicting its own band. The behaviour before this setting existed. |
| `once_per_run` |  | One sidecar per run over the union of the bands: the calibration is fitted once, each unique sequence is predicted once, and every band's table is written under the name a per-band run gives it. A library the caller already re-predicted with the base model (`run-experiment` with the multi-head calibration off) is not re-predicted per band again. |

### `Handoff`

(rust/mumdia/crates/mumdia-core/src/config.rs)

How the feature matrix crosses the Rust -> Python boundary for a sidecar rescorer.

| Value | Default | Description |
|---|---|---|
| `tsv` |  | Tab-separated PIN. Percolator's format, and what `mokapot.read_pin` requires, so it is what a mokapot or entrapment sidecar receives whatever this is set to. |
| `parquet` | yes | Parquet feature table with f32 features, and the default since 2026-09-05. The TSV path makes the worker parse every column into a float64 pandas frame before it builds its float32 matrix, so the text file, the frame and the matrix are alive together. Measured on the HYE competed table (2,603,894 PSMs x 387 features, one self-training iteration, 32 threads), parquet against tsv: rescore peak 29.96 -> 8.95 GB, wall 8:35 -> 6:33, sidecar file 9.53 -> 3.28 GB, the worker's read and standardise phase 111.7 -> 16.9 s, and 47,752 against 47,762 peptides at 1% with the decoy fraction 1.00% either way (docs/28 section 11). An earlier 8,858,206-PSM experiment-wide rescore went from 671.6 min to 12 min, because there the 30.18 GB TSV crossed the worker's streaming threshold and every iteration re-read a 12.77 GB memmap. Features are f32 because the TSV was already lossy (`{:.6}`) and the worker casts to f32 regardless; the two paths therefore feed marginally different values into a chaotic self-training loop, which is where that 10-peptide difference comes from. nn_torch only: `mokapot_worker.py` calls `mokapot.read_pin()` and cannot read Parquet, so a mokapot run falls back to `Tsv` with a warning instead of failing. |
| `raw` |  | Opt-in: the features as one row-major little-endian f32 `.npy` matrix, beside a small parquet of the metadata columns and a `<name>.raw.json` description (feature names, per-feature min/max, the parquet handoff's row-group size) that the worker is given. The engine streams each decoded batch straight into the file with no transpose and no parquet encode, and the worker copies the matrix into its own with no decode and no column-to-row transpose: on the 258.75M-row immunopeptidomics pool the parquet path spent an estimated 26 min of serial engine CPU encoding and the worker a strided fill of every column. The file is the raw size, 4 bytes a value, about 11% more than the snappy parquet there, so it pays where the codec, not the disk, is the limit (a RAM-backed `MUMDIA_SIDECAR_DIR`, an SSD). Features that compress well make the gap much larger: docs/13 has a table where the raw write was the slower one, so measure on the data first. Scores are byte-identical to `Parquet`: the worker fills the same matrix and sums the float64 moments over the same partition (the description carries the row-group size), and drops the same constant columns. Validate a new host by rescoring one pool with each handoff, same seed and threads, and comparing `psms_scored.parquet` byte for byte (`tests/python/test_nn_rescore_worker.py` does it on a fixture). nn_torch only; a mokapot run falls back to `Tsv` with a warning. |

### `LibraryIrt`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Source of `predicted_irt` for an imported library; see `RtImTrainConfig::library_irt`.

| Value | Default | Description |
|---|---|---|
| `auto` | yes |  |
| `library` |  |  |
| `deeplc` |  |  |

### `MatcherKind`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Fragment-matcher backend for search-seed and extract (docs/06_predict_frag_index_matchers.md). Default `Fragindex` (log-bin CSR matcher): on narrow-window DIA it is ~1.95x faster in search-seed and ~1.26x in extract with essentially unchanged IDs (HYE B_01: peptides -0.1%); `Bucketed` is the previous `Library::page_search` path (retained for A/B and for the AIF full-range-window case, where the predicate difference shifts IDs more).

| Value | Default | Description |
|---|---|---|
| `bucketed` |  |  |
| `fragindex` | yes |  |

### `MbrStrategy`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Match-between-runs strategy (Stage D3, docs/12_quant_lfq_align_mbr_report_audit.md). Default `None` reproduces the current chain byte-for-byte. ONLY `None` VS NOT-`None` IS IMPLEMENTED. The three non-`None` variants are described below as the intended staging, but no code distinguishes them: every test in the tree is `strategy != None`, so selecting `RtTransfer` or `Full` today behaves exactly like `EmpiricalLibrary`. They are kept as the recorded design ladder rather than deleted because the MBR tier is planned and benchmark-gated (CLAUDE.md); `validate()` warns when a non-`None` variant is selected so a config cannot quietly expect more than it gets. Intended staging: `EmpiricalLibrary` builds the consensus anchor library only; `RtTransfer` adds cross-run expected-RT transfer extraction; `Full` adds requantification. All require >= 2 runs and a decoy-transfer FDR (see the plan).

| Value | Default | Description |
|---|---|---|
| `none` | yes | No match-between-runs (default). |
| `empirical_library` |  | Build the cross-run consensus anchor library (M1) only; no transfer. |
| `rt_transfer` |  | EmpiricalLibrary + cross-run expected-RT transfer extraction (M2/M3). |
| `full` |  | RtTransfer + requantification of accepted transfers (M5). |

### `NegSelect`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Which decoys survive the training-set negative cap.

| Value | Default | Description |
|---|---|---|
| `random` | yes | A uniform random sample of the fold's decoys. The population the model sees keeps the shape of the real decoy distribution, only thinner. |
| `margin` |  | The highest-scoring decoys under the current model: the part of the decoy distribution that still competes with accepted targets, and the only part the decision boundary depends on. Trains on hard negatives only, so the model never sees the easy bulk it must also keep rejecting. |
| `hybrid` |  | Half the budget from the margin, half sampled at random from the rest, so the boundary is informed by the hard cases without losing the shape of the bulk. |

### `PeakClaim`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Fragment-peak apportionment when one observed MS2 peak matches the fragments of several co-isolated, co-eluting candidates (near-universal in wide-window DIA: ~98% of fragment m/z collide within tolerance). Decides how the peak's intensity is shared, to stop a chimeric candidate borrowing a real peptide's peak wholesale.

| Value | Default | Description |
|---|---|---|
| `none` | yes | Every matching candidate gets the full peak intensity (legacy default). |
| `winner_predicted_intensity` |  | Winner-take-all: only the candidate with the highest predicted intensity for its matching fragment gets the peak; the rest get nothing. |
| `proportional` |  | Soft apportionment: split the peak intensity across claimants in proportion to their predicted intensity for the matching fragment. |
| `coelution_winner` |  | Presence-aware winner-take-all (two-pass): a first pass builds each candidate's per-scan elution profile (summed matched intensity); the peak then goes to the claimant most eluting at that scan (highest profile height, i.e. best corroborated by its OTHER fragments), not the one that merely predicts the brightest ion there. |
| `coelution_proportional` |  | Presence-aware soft apportionment (two-pass): split the peak across claimants in proportion to their per-scan elution-profile height. |
| `coelution_winner_margin` |  | Margin-gated co-elution winner (two-pass): winner-take-all ONLY when the top eluter's profile height dominates the runner-up by `peak_claim_margin` (else the peak stays shared among all claimants, as in `None`). Avoids stripping real peptides at ambiguous peaks where no candidate clearly owns the elution. |
| `coelution_multi_cue` |  | Multi-cue co-elution winner (two-pass, modular fragment-competition framework). The per-claimant competition weight is the elution profile height multiplied by the composable cues enabled in `ClaimCues` (sub-tolerance m/z proximity, RT prior, isotope coherence, MS1 precursor support, ...), each defaulting to 1.0 so this reduces to `CoelutionWinner` when no cue is enabled. Winner-take-all on the composite weight when `reassign` is set. |
| `coelution_demix` |  | Spectrum-centric demix redistribution (two-pass, destructive). At each scan the co-isolated candidate x fragment design matrix is assembled and solved by non-negative least squares; each shared peak's intensity is then split among its claimants in proportion to `beta_c * D[peak,c]` (the joint deconvolution) instead of stripped winner-take-all. The smooth, principled destructive mode - the CHIMERYS coefficient split, made chromatographic and clean-room. Always redistributes; the demix FEATURES are the separate `emit_demix_features` path. Deterministic (sorted candidate columns, ridge NNLS). |
| `coelution_shadow` |  | Shadow-subtraction redistribution (two-pass, destructive, no solver). At each scan, each co-eluter's abundance is estimated from the channels it ALONE claims (its unique ions, `a_p = median y/D` over those); every candidate then keeps, at each of its channels, `max(0, y - sum_{p != c} a_p * D[peak,p])` - its intensity minus the interferers' estimated contributions. Unlike winner-take-all, several real co-eluters can both retain signal at a shared peak; unlike the NNLS demix it needs no solve, so it is cheap. A candidate with no unique ion cannot be estimated and contributes no subtraction. The gentle destructive mode. Deterministic; default off. |

### `PeakWindowMode`

(rust/mumdia/crates/mumdia-core/src/config.rs)

How the elution-peak integration window is chosen per candidate in quant.

| Value | Default | Description |
|---|---|---|
| `per_candidate` | yes | Each candidate's window comes from its own summed-XIC descent walk. Exact per peak but sensitive to interference (stretched) and sparse peaks (collapsed). |
| `consensus` |  | Consensus window: the median left and right half-widths of confident peptides (a near-constant instrument/gradient property) applied around each candidate's apex. Robust to a single window being distorted. The widths are estimated per quant invocation, not shared automatically across runs. |

### `QuantQColumn`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Which q-value column quant filters candidates on. Peptide- or precursor-level q is appropriate for a single-run rescore. Under experiment-wide rescoring, those grouped q-values are pooled and carried only on the best PSM across all runs, so filtering per-run slices on them creates disjoint quant sets. `RunPsmQ` is the run-local FDR gate for that cross-run workflow; `PsmQ` keeps the pooled per-PSM gate available when that is explicitly intended.

| Value | Default | Description |
|---|---|---|
| `peptide_q` | yes | Filter on `peptide_q_value` (per-run peptide FDR). Default. |
| `precursor_q` |  | Filter on `precursor_q`. This is valid only for a single-run rescore: experiment-wide rescoring currently computes precursor q-values over the pooled experiment and assigns each precursor's grouped q-value to its best PSM, so it is not a per-run cross-run-quant gate. |
| `psm_q` |  | Filter on the per-PSM `q_value`. In an experiment-wide rescore this is a pooled-experiment PSM q-value, not a run-local FDR estimate. |
| `run_psm_q` |  | Filter on `run_psm_q` (per-run PSM FDR). The correct choice for cross-run quant off an experiment-wide rescore: each run's PSMs are FDR-controlled within their own run, so quant keeps the right per-run precursors without the external `split_scored.py` peptide-q overwrite. |

### `RescorerKind`

(rust/mumdia/crates/mumdia-core/src/config.rs)

| Value | Default | Description |
|---|---|---|
| `native_tda` | yes | Native semi-supervised linear rescorer + target-decoy q-values. MVP default (always available). |
| `mokapot` |  | Mokapot Python sidecar (docs/13_sidecars.md). |
| `nn_torch` |  | PyTorch semi-supervised MLP sidecar (`nn_rescore_worker.py`): a nonlinear Percolator/mokapot-style rescorer (CV folds + iterative positive re-selection). On the E.coli benchmark it beats the linear mokapot model on the same PIN, and — being robust to an unfiltered pool — gains further when the extraction gate is opened. Same positional-CLI PIN contract as Mokapot; requires `rescore.python` to point at an interpreter with torch. |
| `percolator` |  | External percolator.exe over the PIN file. |
| `entrapment` |  | Spike-in (entrapment) negative rescorer: treat foreign-proteome PSMs (identified by `entrapment_marker`) as real negatives, train a nonlinear GBM sidecar (out-of-fold by base peptide) or a native linear fallback, and report entrapment-calibrated q-values. The chimeric false matches that in-silico decoys under-model appear as real negatives here, so it closes the FDR-validity gap the decoy schemes cannot. |

### `RollupMethod`

(rust/mumdia/crates/mumdia-core/src/config.rs)

| Value | Default | Description |
|---|---|---|
| `top_n_sum` | yes | Sum of the top-N most abundant peptides (single-run default). |
| `sum` |  | Sum of all group peptides. |

### `RtLibraryScope`

(rust/mumdia/crates/mumdia-core/src/config.rs)

How many DeepLC fine-tunes an experiment pays for.

| Value | Default | Description |
|---|---|---|
| `first_run_only` | yes | Adapt the library's retention times once, on the FIRST run's confident seeds, and reuse that library for every run. Each run still fits its OWN retention-time calibration (LOESS by default) on top of it. "Adapt" is whichever of the two mechanisms is active: the DeepLC fine-tune, or multi-head calibration. They occupy the same slot, cost the same kind of time -- one full re-prediction of the library per run -- and are amortised the same way. MEASURED COST (6-run ProteoBench HYE AIF set, 2026-07-28). Reuse is NOT free: the run that owned the fine-tune reached a median \|RT residual\| of 15.2 s, while the five reusing runs reached 20.3, 20.5, 20.9, 24.9 and 25.4 s -- +7.2 s, +47% on average -- and their calibrated RT windows widened from 145 s to 179-227 s. The degradation is MONOTONIC in acquisition order, i.e. real chromatographic drift that a single fine-tune cannot track; per-run LOESS corrects the slope (0.96-0.99) but not the scatter. Wider windows also cost compute downstream: extract roughly doubled (126 s -> 203-242 s) and features up to tripled (116 s -> 215-388 s), which claws back part of the saving. It is still the default because the fine-tune dominates a large experiment: one 36.5 min fine-tune instead of N. On an 80-run batch that is ~48 h saved against ~6.5 h of extra extract/features. But on a long batch the drift keeps growing, so prefer `PerRun` when the extra hours are affordable, and treat periodic re-fine-tuning (not yet implemented) as the better answer for very large batches. |
| `per_run` |  | Fine-tune separately for every run. Adapts the model weights to each run's own chromatography instead of only calibrating a shared model, which measurably tightens retention time: see the numbers on `FirstRunOnly`. Costs one full DeepLC fine-tune per run (36.5 min on the HYE library: 5.7 min training plus 30.8 min predicting 4.9M peptidoforms). |

### `RtPredictorKind`

(rust/mumdia/crates/mumdia-core/src/config.rs)

| Value | Default | Description |
|---|---|---|
| `native` | yes | Native additive retention-coefficient model (no Python). MVP default so the engine runs with zero external runtime dependencies. |
| `deeplc` |  | DeepLC Python sidecar (docs/13_sidecars.md). |

### `UnknownModPolicy`

(rust/mumdia/crates/mumdia-core/src/config.rs)

| Value | Default | Description |
|---|---|---|
| `error` | yes |  |
| `skip` |  |  |

1 enum(s) are declared in `rust/mumdia/crates/mumdia-core/src/config.rs` but are not reachable from `Config`, so they are not config values: `NormalizeMethod`. They are CLI-only or helper types.

## Environment variables

Collected by scanning `std::env::var`/`var_os` across
`rust/mumdia/crates/**/*.rs` and `os.environ`/`os.getenv` across
`scripts/*.py`. These are not config keys: nothing validates them, a typo is
silently ignored, and none of them appears in the run manifest. Prefer a
config field or a CLI flag where one exists, and treat this table as the
record of what the code will read if the variable happens to be set.

`Side` says which process reads the variable. **engine** is the Rust binary;
**sidecar** is a Python worker, which the engine launches as a child process
and which therefore inherits the engine's environment. A variable read on
both sides is marked **both**.

`Default in code` is the fallback the reading code supplies when the variable
is unset. Two workers can disagree, in which case every distinct fallback is
listed with the file it is in. A default marked `computed:` has no literal
fallback in the code: the reading function works out the behaviour, and the
text describes it (`COMPUTED_ENV_DEFAULTS` in `ci/gen_config_reference.py`).

`Read at` and `Site` name the file and the enclosing function as
`path::function`: `Type::method` for a Rust method, `Trait::method` for a
default trait method, inline modules and nested Rust functions joined with
`::`, `Class.method` in Python, `<module>` outside any
function. No line number is cited, so the tables change only when a read
moves to another function.

13 of these are also SET by the engine before the worker starts, so the worker's own fallback applies only when the engine did not set it: `MUMDIA_NN_FOLDS`, `MUMDIA_NN_FOLD_KEYS`, `MUMDIA_NN_ITERS`, `MUMDIA_NN_MARGIN_FRAC`, `MUMDIA_NN_NEG_RATIO`, `MUMDIA_NN_NEG_SELECT`, `MUMDIA_NN_SEEDS`, `MUMDIA_NN_THREADS`, `MUMDIA_NN_TRAIN_FDR`, `MUMDIA_NN_TRAIN_SUB`, `MUMDIA_NN_WARM_EPOCHS`, `MUMDIA_NN_WARM_START`, `OMP_NUM_THREADS`. See the next table.

| Variable | Side | Default in code | Read at |
|---|---|---|---|
| `CONDA_PREFIX` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `DEEPLC_FT_THREADS` | sidecar | `"8"` | `scripts/deeplc_finetune.py::<module>` |
| `MUMDIA_BREW_ITERS` | sidecar | `"20"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_CHROM_ROW_GROUP_ROWS` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/chromatograms.rs::row_group_rows` |
| `MUMDIA_CONVERT_THREADS` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/stages/convert.rs::convert_threads` |
| `MUMDIA_DEEPLC_RAW_OUTPUT` | sidecar | `""` | `scripts/deeplc_finetune.py::quiet_deeplc_progress`, `scripts/deeplc_worker.py::quiet_deeplc_progress` |
| `MUMDIA_DEEPLC_THREAD_CAP` | sidecar | `"auto"` | `scripts/deeplc_finetune.py::deeplc_thread_cap`, `scripts/deeplc_worker.py::deeplc_thread_cap` |
| `MUMDIA_ENTRAPMENT_MODEL` | sidecar | `"gbm"` | `scripts/entrapment_worker.py::_new_model` |
| `MUMDIA_KEEP_HANDOFF` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::keep_handoff` |
| `MUMDIA_LR_C` | sidecar | `"1.0"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_LR_MAX_ITER` | sidecar | `"1000"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_MOKAPOT_WORKERS` | sidecar | `"3"` | `scripts/mokapot_worker.py::main` |
| `MUMDIA_MSCONVERT` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/raw.rs::locate_msconvert` |
| `MUMDIA_NN_ALPHA` | sidecar | `"1e-4"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_NN_BATCH` | sidecar | `4096` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_CHUNK` | sidecar | `250000` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_CLAMP_TINY` | sidecar | `1e-20` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_DEBUG_DENORMALS` | sidecar | `0` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_DEVICE` | sidecar | `"auto"` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_DROPOUT` | sidecar | `0.3` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_DROP_CONSTANT` | sidecar | `1` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_EARLY_STOP` | sidecar | `1` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_EARLY_STOP_TOL` | sidecar | `0.01` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_EPOCHS` | sidecar | `25` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_FEATURES` | sidecar | `""` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_FINAL_POOL_SCORE` | sidecar | `0` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_FLUSH_DENORMAL` | sidecar | `1` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_FOLDS` | sidecar | `3` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_FOLD_KEYS` | sidecar | `""` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_GATHER` | sidecar | `"torch"` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_HIDDEN` | sidecar | `"128,64"` in nn_rescore_worker.py; `"128,64,64,32"` in mokapot_worker.py | `scripts/mokapot_worker.py::make_model`, `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_INIT_FDR_MAX` | sidecar | `0.05` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_INIT_SAMPLE` | sidecar | `300000` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_INIT_TOPK` | sidecar | `0` | `scripts/nn_rescore_worker.py::_build_trainer.run_fold` |
| `MUMDIA_NN_ITERS` | sidecar | `5` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_LOAD_THREADS` | sidecar | `"auto"` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_LR` | sidecar | `1e-3` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_MARGIN_FRAC` | sidecar | `0.5` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_MAX_ITER` | sidecar | `"200"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_NN_NEG_RATIO` | sidecar | `0.0` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_NEG_SELECT` | sidecar | `"random"` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_PARALLEL` | sidecar | `0` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_PARALLEL_THREADS` | sidecar | `""` | `scripts/nn_rescore_worker.py::_train_in_processes` |
| `MUMDIA_NN_PREGATHER_GB` | sidecar | `8` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_PRE_BUFFER` | sidecar | `1` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_READ_AHEAD` | sidecar | `1` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_SCAN_MEM_GB` | sidecar | `1.0` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_SCAN_ROWS_PER_THREAD` | sidecar | `20000` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_SCAN_THREADS` | sidecar | `"auto"` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_SEED` | sidecar | `0` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_SEEDS` | sidecar | `1` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_SELECT` | sidecar | `"window"` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_SOLVER` | sidecar | `"adam"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_NN_STREAM` | sidecar | `"auto"` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_STREAM_GB` | sidecar | `4` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_THREADS` | both | `16` | `rust/mumdia/crates/mumdia/src/main.rs::apply_threads`, `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_THREAD_CAP` | sidecar | `"auto"` | `scripts/nn_rescore_worker.py::torch_thread_cap` |
| `MUMDIA_NN_TRAIN_FDR` | sidecar | `0.01` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_TRAIN_SUB` | sidecar | `0.0` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_WARM_EPOCHS` | sidecar | `0` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_WARM_START` | sidecar | `0` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_NN_WD` | sidecar | `1e-4` | `scripts/nn_rescore_worker.py::main` |
| `MUMDIA_PARQUET_COMPRESSION` | engine | computed: snappy. `zstd`, or `uncompressed` / `none`, changes the codec (`table.rs` `codec`) | `rust/mumdia/crates/mumdia-io/src/table.rs::codec` |
| `MUMDIA_PARQUET_DECODE_THREADS` | engine | computed: automatic column groups, up to the codec pool's threads; one reader for a coalesced scan or inside a rayon pool. `k` asks for k groups, `1` is one reader (`table.rs` `automatic_decode_groups`) | `rust/mumdia/crates/mumdia-io/src/table.rs::TableFile::scan_spec` |
| `MUMDIA_PARQUET_PLAN` | engine | computed: on (capped writers plan their float encodings). `0` / `off` / `false` / `no` restores the unplanned layout (`table.rs` `plan_enabled`) | `rust/mumdia/crates/mumdia-io/src/table.rs::plan_enabled` |
| `MUMDIA_PARQUET_THREADS` | engine | computed: min(`--threads`, 8), or min(cores, 8) without `--threads`. `0` or `1` is serial (`codec.rs` `codec_threads`) | `rust/mumdia/crates/mumdia-io/src/codec.rs::codec_threads` |
| `MUMDIA_PEPTDEEP_DEVICE` | sidecar | `"auto"` | `scripts/peptdeep_worker.py::main` |
| `MUMDIA_PREDICT_FRAG_CONCURRENT` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/stages/predict_frag.rs::assign_predictions` |
| `MUMDIA_PYTHON` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `MUMDIA_PYTHON_DEEPLC` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `MUMDIA_PYTHON_MBR` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `MUMDIA_PYTHON_MS2PIP` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `MUMDIA_PYTHON_PEPTDEEP` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `MUMDIA_PYTHON_RESCORE` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `MUMDIA_QUANT_SELECTIVE_READ` | engine | computed: on (quant skips, unread, the chromatogram data pages that hold no kept row). `0` reads every page of every row group it opens; the outputs are the same (`stages/quant.rs` `selective_read_enabled`) | `rust/mumdia/crates/mumdia/src/stages/quant.rs::selective_read_enabled` |
| `MUMDIA_RESCORE_MODEL` | both | `"nn"` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_hashed`, `scripts/mokapot_worker.py::main`, `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_SCRIPTS` | sidecar | `os.path.dirname(os.path.abspath(__file__` | `scripts/mh_shard_predict.py::<module>` |
| `MUMDIA_SIDECAR_DIR` | engine | computed: `<out-dir>/sidecar_work` under `run` and `run-experiment`; `sidecar_work` in the current directory, or `--work-dir`, for `mumdia rescore`. A path moves the rescore sidecar files there (`stages/rescore.rs` `sidecar_work_dir`) | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::sidecar_work_dir` |
| `MUMDIA_SIDECAR_SPACE_CHECK` | engine | computed: on (before the handoff is written, a rescore sidecar run is refused when its work directory has less room than a PIN or raw handoff cannot be smaller than, and warned about below the files' usual size). `0` / `off` / `false` / `no` skips the check (`stages/rescore.rs` `check_sidecar_space`) | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::check_sidecar_space` |
| `MUMDIA_THERMO_PARSER` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/raw.rs::locate_parser` |
| `MUMDIA_WIDE_SCAN` | engine | computed: `plain`: the plain reader with its parallel decode for rescore's feature stream and compete's pass-through copy. `rowgroup` decodes the feature stream one row group a batch; `coalesced` reads each row group's projected column chunks in one sequential read (`stages/mod.rs` `WideScan`) | `rust/mumdia/crates/mumdia/src/stages/mod.rs::WideScan::from_env` |
| `MUMDIA_XGB_DEPTH` | sidecar | `"6"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_XGB_JOBS` | sidecar | `"0"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_XGB_LR` | sidecar | `"0.1"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_XGB_TREES` | sidecar | `"200"` | `scripts/mokapot_worker.py::make_model` |
| `OMP_NUM_THREADS` | both | `16` | `rust/mumdia/crates/mumdia/src/main.rs::apply_threads`, `scripts/nn_rescore_worker.py::main` |
| `PATH` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/raw.rs::locate` |
| `ProgramFiles` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/raw.rs::locate_msconvert` |
| `ProgramFiles(x86)` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/raw.rs::locate_msconvert` |
| `VIRTUAL_ENV` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |

90 variables are read: 28 engine-side, 65 sidecar-side, 3 on both sides.

### Variables the code sets

A variable set here overrides whatever the caller exported, so exporting one
of these has no effect on the process listed. The engine's `--threads` is the
one exception noted in its own help text: it sets `MUMDIA_NN_THREADS` and
`OMP_NUM_THREADS` for the sidecars only if they are not already set.

| Variable | Set by | Value | Site |
|---|---|---|---|
| `KMP_DUPLICATE_LIB_OK` | sidecar | `"TRUE"` | `scripts/deeplc_finetune.py::<module>` |
| `MKL_NUM_THREADS` | sidecar | `"1"` | `scripts/deeplc_finetune.py::<module>` |
| `MUMDIA_NN_FOLDS` | engine | `p.cfg.folds.to_string()` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_pin_sidecar` |
| `MUMDIA_NN_FOLD_KEYS` | engine | `foldkeys` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_pin_sidecar` |
| `MUMDIA_NN_ITERS` | engine | `p.cfg.num_iter.to_string()` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_pin_sidecar` |
| `MUMDIA_NN_MARGIN_FRAC` | engine | `p.cfg.train_margin_frac.to_string()` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_pin_sidecar` |
| `MUMDIA_NN_NEG_RATIO` | engine | `p.cfg.train_neg_ratio.to_string()` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_pin_sidecar` |
| `MUMDIA_NN_NEG_SELECT` | engine | `match p.cfg.train_neg_select { mumdia_core::config::NegSelect::Random => "random", mumdia_core::config::NegSelect::Margin => "margin", mumdia_core::config::NegSelect::Hybrid => "hybrid", }` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_pin_sidecar` |
| `MUMDIA_NN_SEEDS` | engine | `p.cfg.seeds.max(1).to_string()` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_pin_sidecar` |
| `MUMDIA_NN_THREADS` | engine | `n.to_string()` | `rust/mumdia/crates/mumdia/src/main.rs::apply_threads` |
| `MUMDIA_NN_TRAIN_FDR` | engine | `p.cfg.train_fdr.to_string()` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_pin_sidecar` |
| `MUMDIA_NN_TRAIN_SUB` | engine | `p.cfg.train_subsample.to_string()` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_pin_sidecar` |
| `MUMDIA_NN_WARM_EPOCHS` | engine | `p.cfg.train_warm_epochs.to_string()` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_pin_sidecar` |
| `MUMDIA_NN_WARM_START` | engine | `if p.cfg.train_warm_epochs > 0 { "1" } else { "0" }` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_pin_sidecar` |
| `NUMEXPR_NUM_THREADS` | sidecar | `"1"` | `scripts/deeplc_finetune.py::<module>` |
| `OMP_NUM_THREADS` | both | `"1"` in deeplc_finetune.py; `n.to_string()` in main.rs | `rust/mumdia/crates/mumdia/src/main.rs::apply_threads`, `scripts/deeplc_finetune.py::<module>` |
| `OPENBLAS_NUM_THREADS` | sidecar | `"1"` | `scripts/deeplc_finetune.py::<module>` |
| `PYTHONIOENCODING` | engine | `"utf-8"` | `rust/mumdia/crates/mumdia/src/sidecar.rs::run_worker` |
| `PYTHONUTF8` | engine | `"1"` | `rust/mumdia/crates/mumdia/src/sidecar.rs::run_worker`, `rust/mumdia/crates/mumdia/src/stages/rescore.rs::free_bytes`, `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_entrapment_gbm` |

## Unresolved by the generator

Listed rather than omitted, so a parsing gap is visible in the document
instead of looking like an absent field.

Every field whose struct has an `impl Default` resolved from the source.

2 field(s) have no default because the owning struct has no `impl Default`. That is the source's intent, not a parsing gap: a list element must carry every key.

- `peptidoforms.fixed_mods[].name` (`String`)
- `peptidoforms.fixed_mods[].residue` (`char`)

5 environment read(s) whose name is not a literal. Reads with the same function, access and argument share one entry, which gives their number when there is more than one:

- `rust/mumdia/crates/mumdia/src/stages/extract.rs::PsmRows::push: env read via closure of `&mut self``
- `rust/mumdia/crates/mumdia/src/stages/extract.rs::PsmStream::push: env read via closure of `&mut self``
- `rust/mumdia/crates/mumdia/src/stages/extract.rs::flush_below: env read via closure of `chunk.slices_mut()``
- `rust/mumdia/crates/mumdia/src/stages/extract.rs::flush_below: env read via closure of `runs[r].span_mut(m)``
- `rust/mumdia/crates/mumdia/src/stages/rescore.rs::check_sidecar_space: env read of `k``

## Coverage

19 structs and 204 fields emitted from `rust/mumdia/crates/mumdia-core/src/config.rs`, plus 27 enumerations, 1 named profile(s), 90 environment variables read and 19 set.

20 field(s) carry a gating marker in their doc comment. 48 field(s) carry no doc comment at all, so their description is empty above. 0 default(s) could not be resolved and 2 have none by design.
