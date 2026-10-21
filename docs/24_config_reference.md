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
| [(top level)](#top-level) | `Config` | 18 | [docs/02_config_and_data_model.md](02_config_and_data_model.md) |
| [`convert`](#convert) | `ConvertConfig` | 13 |  |
| [`prescan`](#prescan) | `PrescanConfig` | 7 | [docs/21_prescan.md](21_prescan.md) |
| [`prescreen`](#prescreen) | `PrescreenConfig` | 31 |  |
| [`prescreen.tags`](#prescreentags) | `PrescreenTagsConfig` | 7 |  |
| [`prescreen.mass_hypotheses`](#prescreenmass_hypotheses) | `PrescreenMassConfig` | 7 |  |
| [`prescreen.trace`](#prescreentrace) | `PrescreenTraceConfig` | 8 |  |
| [`digest`](#digest) | `DigestConfig` | 7 | [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md) |
| [`digest.decoy`](#digestdecoy) | `DecoyConfig` | 1 | [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md) |
| [`peptidoforms`](#peptidoforms) | `PeptidoformsConfig` | 7 | [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md) |
| [`predict_frag`](#predict_frag) | `PredictFragConfig` | 17 | [docs/06_predict_frag_index_matchers.md](06_predict_frag_index_matchers.md) |
| [`search_seed`](#search_seed) | `SearchSeedConfig` | 11 | [docs/07_search_seed.md](07_search_seed.md) |
| [`rt_im_train`](#rt_im_train) | `RtImTrainConfig` | 31 | [docs/08_rt_im_train.md](08_rt_im_train.md) |
| [`extract`](#extract) | `ExtractConfig` | 45 | [docs/09_extract.md](09_extract.md) |
| [`extract.claim_cues`](#extractclaim_cues) | `ClaimCues` | 7 | [docs/09_extract.md](09_extract.md) |
| [`features`](#features) | `FeaturesConfig` | 12 | [docs/10_features.md](10_features.md) |
| [`compete`](#compete) | `CompeteConfig` | 6 | [docs/11_compete_rescore_fdr.md](11_compete_rescore_fdr.md) |
| [`rescore`](#rescore) | `RescoreConfig` | 31 | [docs/11_compete_rescore_fdr.md](11_compete_rescore_fdr.md) |
| [`quant`](#quant) | `QuantConfig` | 19 | [docs/12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md) |
| [`mbr`](#mbr) | `MbrConfig` | 9 | [docs/12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md) |
| [`experiment`](#experiment) | `ExperimentConfig` | 3 | [docs/01_overview_and_dataflow.md](01_overview_and_dataflow.md) |
| [`groups`](#groups) | `GroupsConfig` | 8 |  |
| [`retrace`](#retrace) | `RetraceConfig` | 5 | [docs/09_extract.md](09_extract.md) |
| [`peptidoforms.fixed_mods[] / peptidoforms.variable_mods[]`](#peptidoformsfixed_mods--peptidoformsvariable_mods) | `ResidueMod` | 2 | [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md) |

## (top level)

`Config` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/02_config_and_data_model.md](02_config_and_data_model.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `convert` | `ConvertConfig` | the `ConvertConfig` section's own defaults |  |  |
| `prescan` | `PrescanConfig` | the `PrescanConfig` section's own defaults |  |  |
| `prescreen` | `PrescreenConfig` | the `PrescreenConfig` section's own defaults |  |  |
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
| `retrace` | `RetraceConfig` | the `RetraceConfig` section's own defaults |  | Raw-event trace rebuild for diaPASEF runs (`RetraceConfig`). Omitted from the canonical configuration while every key is at its default. |

## convert

`ConvertConfig` (rust/mumdia/crates/mumdia-core/src/config.rs).

Vendor-format conversion, read by every subcommand that takes a spectra path (`raw.rs`; docs/04_convert.md, "Vendor formats"). The engine itself reads mzML only, deliberately: `mzdata` is pinned to its pure-Rust `mzml` + `miniz_oxide` features so the build needs no C or .NET toolchain, and its vendor readers would reintroduce both. A vendor file is therefore converted to mzML first by an external converter run as a child process, in the same way the Python sidecars are: ThermoRawFileParser for Thermo `.raw`, ProteoWizard `msconvert` for everything else and as the Thermo fallback.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `thermo_raw_parser` | `String` | `"auto"` |  | Path to the ThermoRawFileParser executable, or `"auto"` to search. `"auto"` looks at `MUMDIA_THERMO_PARSER`, then beside the engine binary, then on `PATH`. Empty means the same as `"auto"`; a real path is used verbatim and its absence is an error rather than a silent fallback, because a fallback would convert with a different program than the one asked for and vendor conversion is not reproducible across converters. |
| `msconvert` | `String` | `"auto"` |  | Path to ProteoWizard `msconvert`, or `"auto"` to search. Used for every vendor format except Thermo, which prefers ThermoRawFileParser: Bruker `.d`, SCIEX `.wiff`, Agilent `.d` and Waters `.raw`. It is also the Thermo fallback when no ThermoRawFileParser is found. `"auto"` searches `MUMDIA_MSCONVERT`, beside the engine binary, the version-stamped ProteoWizard directories under Program Files on Windows (newest first), then `PATH`. MuMDIA never ships or downloads ProteoWizard. Its vendor readers bundle the instrument vendors' own libraries under the vendors' licence terms, which the user accepts when obtaining it, and automating that acceptance is not MuMDIA's to do. |
| `msconvert_args` | `Vec<String>` | `[]` |  | Extra arguments appended to every `msconvert` invocation. An escape hatch, not a tuning surface. The per-vendor defaults already request indexed 64-bit zlib mzML, vendor peak picking where it exists, and `--combineIonMobilitySpectra` for Bruker. Use this for something the defaults cannot express, such as an `--filter` that trims an acquisition. Arguments are passed through verbatim and are not validated. |
| `reuse_converted` | `bool` | `true` |  | Reuse an mzML that already sits beside the `.raw` and is newer than it. On by default: conversion is minutes per file and its output is deterministic given the same converter, so re-running a search should not pay for it twice. Turn it off when the neighbouring mzML may have come from a different converter or a different `.raw` of the same name. |
| `parallel_conversions` | `usize` | `4` |  | How many vendor files `run` and `run-experiment` convert to mzML at once. Default 4. Every vendor input is converted before the first run starts, and the conversions used to run one after another, so an experiment of N `.raw` files paid N converter runs of several minutes each on the critical path. Each conversion is a separate child process writing its own destination under its own lock, and its output does not depend on what else runs beside it, so running them concurrently changes the wall time only. The bound is there because a converter reads a multi-GB file and writes a larger one: more at once than the disk can feed is slower, not faster. `1` converts one at a time, as before. Only the first conversion of an input pays this at all; `reuse_converted` skips the rest. Not measured at scale. |
| `tdf_mz_ppm` | `f64` | `10.0` |  | Native diaPASEF reader (`stages::convert::tdf`): m/z merge tolerance in ppm between neighbouring TOF points of one centroid (single linkage). Default 10. Read only for a timsTOF `.d` whose frames are diaPASEF, which `convert` reads natively; every other input ignores the `tdf_*` keys. Each `tdf_*` key is omitted from the canonical configuration at its default, so it changes no config hash. |
| `tdf_im_gap_scans` | `u32` | `30` |  | Native diaPASEF reader: a gap of more than this many TIMS scans without signal splits one m/z trace into separate centroids. Default 30. Provisional: it rests on a peak census of one run and the seed plateau, not on an end-to-end sweep. |
| `tdf_min_points` | `u32` | `2` |  | Native diaPASEF reader: a centroid built from fewer raw TOF x scan points than this is dropped. Default 2. Provisional, as `tdf_im_gap_scans`. |
| `tdf_mz_valley` | `f64` | `0.0` | benchmark-gated | Native diaPASEF reader: split each m/z x mobility cluster at the valleys of its m/z profile. A local minimum of the smoothed profile is a cut when it is below this fraction of the smaller of the two humps beside it, so one ion stays one centroid and two neighbouring ions become two. Must be in [0, 1); 0 (default) is off and leaves the centroids unchanged. Benchmark-gated (docs/35_diapasef.md). |
| `tdf_mz_smooth_ppm` | `f64` | `8.0` |  | Native diaPASEF reader: half-width in ppm of the triangular smoothing applied to the m/z profile before `tdf_mz_valley` looks for valleys; 0 is no smoothing. Default 8. Unused while `tdf_mz_valley` is 0. |
| `tdf_im_valley` | `f64` | `0.0` | benchmark-gated | Native diaPASEF reader: split each cluster (after the m/z valley split, when that is on) at the valleys of its mobility profile, the summed intensity per TIMS scan after a triangular smoothing of half-width `tdf_im_smooth_scans`. Separates ions of one m/z whose mobility profiles touch without a `tdf_im_gap_scans` gap. Must be in [0, 1); 0 (default) is off and leaves the centroids unchanged. Benchmark-gated. |
| `tdf_im_smooth_scans` | `f64` | `8.0` |  | Native diaPASEF reader: half-width in TIMS scans of the mobility profile smoothing used by `tdf_im_valley`; 0 is no smoothing. Default 8. Unused while `tdf_im_valley` is 0. |
| `tdf_ms1_denoise` | `bool` | `true` |  | Native diaPASEF reader: remove MS1 noise before centroiding, with the method and default MS1 settings of dnoise (Garrett, Diedrich and Yates, bioRxiv 2026, doi:10.64898/2026.08.27.747603): keep only MS1 points on runs of at least 5 occupied mobility scans (TOF neighbourhood +/- 3, gaps of up to 2 scans bridged, two passes), drop features no diaPASEF window reaches (windows padded by 3 Th and 0.015 1/K0), and drop the weak m/z halo of bright ions. Retrace applies the same filter to the raw MS1 frames it holds, so both read the same MS1 events. MS/MS spectra are never filtered. Default true since 2026-10-17; only the native diaPASEF reader reads it. On the six ProteoBench HYE diaPASEF runs it kept 8.7% of the MS1 points with identical identifications (98,524 peptides at 1%, 89,973 ProteoBench ions at k = 3); on the 15 min E. coli run 14,121 against 14,044 peptides at 1% and an entrapment FDP of 0.30-0.38% against 0.39-0.46% (three NN seeds each; docs/35_diapasef.md section 16). `false` restores the unfiltered MS1. |

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

## prescreen

`PrescreenConfig` (rust/mumdia/crates/mumdia-core/src/config.rs).

Fragment-rarity candidate prescreen (`mumdia prescreen`, and `run`/`run-experiment` when `enabled`). Each candidate is scored on the spectra of its own isolation window inside its calibrated RT bounds: its b/y fragments at charges `1..=min(max_frag_charge, z)`, in both the forward and the fully reversed sequence, are matched to the observed peaks within `frag_tol_da`; each matched peak counts once, weighted by `-ln p`, where `p` is the share of that window's spectra with a peak within +/-0.01 Da of it (0.01 Da bins, spectrum-only, no identifications), and by `low_mz_weight` below `low_mz_threshold`; the sum is divided by the square root of the peptide length; the candidate's score is the maximum over its eligible spectra. A cutoff is the `target` quantile ("higher" method) of the scores of a seeded calibration half of the in-scope candidates, labels never used; a candidate is kept when its score exceeds it. Targets and decoys are scored by the identical rule, each on its own sequence, m/z and RT window. Default off.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `false` |  | Run the prescreen inside `run` / `run-experiment`, between rt-im-train and extract, and extract only its survivors. Default false. |
| `preset` | `PrescreenPreset` | `balanced` |  | Calibration preset; `custom` takes `target`. |
| `target` | `f64` | `0.54` |  | Calibration rejection target in (0, 1) when `preset = "custom"`. |
| `seed` | `u64` | `20261003` |  | Seed of the calibration/reporting split. |
| `min_calibration` | `usize` | `100` |  | Fewer in-scope calibration candidates than this bypasses the filter (everything kept), with the reason in the report. |
| `scope` | `PrescreenScope` | `all` |  |  |
| `scope_mods` | `Vec<String>` | `[]` |  | `RESIDUE:UniModName` entries defining `scope = "modified"`, e.g. `M:Oxidation`. |
| `scope_match` | `PrescreenScopeMatch` | `any` |  |  |
| `top_peaks` | `usize` | `0` |  | Most intense peaks per MS2 used for both the rarity histogram and the matching; 0 (the default) uses every positive peak. 300 is the capped comparison setting. |
| `frag_tol_da` | `f64` | `0.005` |  | Candidate fragment matching tolerance, absolute m/z (Da). |
| `max_frag_charge` | `i32` | `2` |  | Highest fragment charge; each candidate uses `1..=min(this, precursor charge)`. |
| `low_mz_threshold` | `f64` | `300.0` |  | Matched peaks below this m/z are weighted by `low_mz_weight`. |
| `low_mz_weight` | `f64` | `0.5` |  |  |
| `both_orientations` | `bool` | `true` |  | Also score the fully reversed sequence and keep the larger score, as the measured prototype did. The same rule applies to targets and decoys. |
| `rt_slack_s` | `f64` | `0.0` |  | Widen each candidate's calibrated RT bounds by this many seconds on both sides. |
| `crowding_exponent` | `f64` | `0.0` |  | Crowding adjustment: divide each spectrum's score by (its peak count / the window's mean peak count)^exponent, the ratio clamped to [0.25, 4]; 0 (default) = off, 0.25 = the prototype's phase-11 setting. |
| `repeat_bonus` | `f64` | `0.0` |  | Repeat bonus: `score / q95 + w * repeat / q95`, where `repeat` counts each matched fragment as strongly as its weaker appearance in a neighbouring scan (two eligible scans, 8 s); 0 (default) = off, 0.1 = the prototype's phase-11 setting. |
| `write_scores` | `bool` | `false` |  | Write the per-candidate scores beside the survivors (`<out>.scores.parquet`). |
| `retrieval` | `PrescreenRetrieval` | `all` |  | Candidate retrieval before scoring. `all` (default) scores every candidate. |
| `rescue` | `bool` | `true` |  | With `retrieval = "tags"`, score candidates without tag evidence anyway (the permissive rescue route) instead of dropping them. Default true. |
| `delayed_modforms` | `bool` | `false` |  | With `retrieval = "tags"`, retrieve backbone families first and examine forms only in retrieved families (delayed modification enumeration). Same retrieved set. |
| `tags` | `PrescreenTagsConfig` | the `PrescreenTagsConfig` section's own defaults |  |  |
| `complement_bonus` | `f64` | `0.0` |  | Weight of the complementary-ion component (prototype 0.25 measured); 0 = off. |
| `tag_bonus` | `f64` | `0.0` |  | Weight of the tag-evidence component (spectrum rarity x path fit, each distinct trimer once); 0 = off. |
| `fasta_bonus` | `f64` | `0.0` |  | Weight of the FASTA/library-side tag information, kept separate from the spectrum side; 0 = off. |
| `flank_bonus` | `f64` | `0.0` |  | Weight of positioned flank support (an observed tag whose ladder endpoints sit at the candidate's b/y masses); 0 = off. |
| `mass_hypotheses` | `PrescreenMassConfig` | the `PrescreenMassConfig` section's own defaults |  |  |
| `trace` | `PrescreenTraceConfig` | the `PrescreenTraceConfig` section's own defaults |  |  |
| `localization` | `PrescreenLocalization` | `own` |  |  |
| `tag_prefilter` | `bool` | `false` |  | Database-free tag prefilter before any prediction, without retention times. Tags are discovered from the spectra alone (`prescreen.tags`); a candidate is kept when one of its trimers, in its own residue states and at a fragment charge it can carry, was observed in its isolation window anywhere in the run, or when no trimer of it can be expressed in the tag alphabet. No fragment score is computed. In FASTA mode only the kept peptidoforms go to MS2PIP and DeepLC; with an imported library the kept candidates become a sub-library before the seed search and the multi-head calibration. Everything after it uses the predicted retention times as usual, including the fragment-rarity score when `enabled`. Single-file `run` only for now. Default false. |
| `score_before_prediction` | `bool` | `false` |  | The fragment-rarity score before any prediction and without retention times: every candidate is scored over the whole gradient of its isolation window, with the preset / target and `crowding_exponent` of this block, and only the candidates kept go to prediction (FASTA mode) or into the sub-library (imported library), as for `tag_prefilter`; with both set a candidate must pass both. Uses no predicted fragment intensity and no retention time. Measured on the 8-12-mer immunopeptidomics library with `crowding_exponent = 0.25`: 54% removed, 99.0% of an unfiltered search's accepted precursors kept (Astral HYE: 52% removed, 95.2% kept, so data-dependent). Single-file `run` only. Default false. |

## prescreen.tags

`PrescreenTagsConfig` (rust/mumdia/crates/mumdia-core/src/config.rs).

Database-free tag discovery used by retrieval and the optional evidence components.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `edge_tol_da` | `f64` | `0.005` |  | Neutral residue-mass tolerance of a tag edge (Da); `edge_tol_da / z` in m/z. |
| `max_charge` | `i32` | `2` |  | Highest fragment charge of a tag edge. |
| `gap_edges` | `bool` | `false` |  | Also allow two-residue gap steps; every compatible residue pair is kept as an alternative and gapped three-peak ladders are keyed separately from four-peak ones. |
| `rms_sigma_da` | `f64` | `0.003` |  | Sigma of the whole-path soft weight `exp(-0.5 (rms / sigma)^2)` (Da). |
| `extra_mods` | `Vec<String>` | `[]` |  | Variable states added to the chemistry configuration's alphabet, `RESIDUE:Name`. |
| `positioned` | `bool` | `false` |  | Positioned retrieval: a candidate is retrieved only by a tag whose ladder starts at one of its own b or y fragment masses (+/- 0.01 Da), not by the same trimer anywhere in the spectrum. Much more specific in dense spectra. |
| `positioned_residues` | `usize` | `3` |  | Residue steps of a positioned ladder: 3 (four consecutive fragments, the default) or 2 (three consecutive fragments, more permissive). |

## prescreen.mass_hypotheses

`PrescreenMassConfig` (rust/mumdia/crates/mumdia-core/src/config.rs).

Blind neutral-mass hypotheses from tag paths and complementary peaks, with optional MS1.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `false` |  | Infer hypotheses and write `<out>.mass_hypotheses.parquet`. |
| `sample_spectra` | `usize` | `512` |  | Spectra sampled for inference (seeded); 0 = every spectrum (expensive). |
| `ms1` | `bool` | `false` |  | Look for the monoisotopic and +1 isotope peaks in MS1 scans within `ms1_rt_s`. |
| `ms1_rt_s` | `f64` | `2.0` |  |  |
| `ms1_ppm` | `f64` | `10.0` |  | MS1 tolerance: max(0.005 Da, this many ppm). |
| `require_ms1` | `bool` | `false` |  | Drop hypotheses without an MS1 link. A hard gate; off by default. |
| `bonus` | `f64` | `0.0` |  | Weight of the candidate mass-support component in the combined score; 0 = report only. |

## prescreen.trace

`PrescreenTraceConfig` (rust/mumdia/crates/mumdia-core/src/config.rs).

Fragment traces over consecutive same-window scans.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `false` |  | Compute the trace component (reported in the scores table). |
| `scans` | `usize` | `7` |  | Consecutive same-window scans per pool. |
| `cluster_da` | `f64` | `0.01` |  | Maximum m/z range (max - min) of one merged cluster. |
| `min_detections` | `usize` | `2` |  | Scans a merged cluster must be detected in. |
| `pooling` | `PrescreenPooling` | `merged` |  |  |
| `score` | `PrescreenTraceScore` | `tag_coherence` |  |  |
| `min_quality` | `f64` | `0.6` |  | Trace cosine counted as compatible by `coherent_fragments`. |
| `bonus` | `f64` | `0.0` |  | Weight of the trace component in the combined score; 0 = report only. |

## digest

`DigestConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `enzyme` | `Enzyme` | `trypsin_p` |  |  |
| `cleavage_rule` | `Option<String>` | `null` |  | The cleavage rule of `enzyme = "custom"`: residue pairs around the cut separated by commas, `*` for any residue, a leading `!` for a pair never cut (the syntax of DIA-NN `--cut`), e.g. `K*,R*,D*,!*P`. Only with `enzyme = "custom"`. Omitted from the canonical configuration when unset. |
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
| `im_predictor` | `ImPredictorKind` | `none` |  | Ion-mobility predictor for the library's `predicted_im` (1/K0, V s cm^-2) in a FASTA build. `none` (the default): the library has no `predicted_im` column. `im2deep`: IM2Deep predicts one 1/K0 per (peptidoform, charge), and the run fails without an IM2Deep interpreter (`im2deep_python`). `auto`: `im2deep` when the run's spectra carry 1/K0 (a diaPASEF `.d`) and an IM2Deep interpreter resolves, otherwise `none` with a log line; a standalone `predict-frag` has no run, so there it is `none`. Decoys built by the native digest get their own prediction, as they get their own iRT. A candidate IM2Deep returns no value for is dropped with its pair, under the same 2% ceiling as a missing iRT or intensity. Library-input mode ignores the key: an imported library carries its own 1/K0. Omitted from the canonical configuration at its default. |
| `im2deep_python` | `Option<String>` | `null` |  | Python executable for the IM2Deep sidecar (env with im2deep >= 2.0.0, psm_utils, torch and pyarrow), or `"auto"` to discover one (`MUMDIA_PYTHON_IM2DEEP`). Used only when `im_predictor` is `im2deep` or `auto`. Omitted from the canonical configuration when unset. |
| `sidecar_script_dir` | `String` | `"scripts"` |  | Directory holding the sidecar worker scripts. |
| `defer_deeplc_to_multihead` | `bool` | `true` |  | Skip DeepLC in a FASTA library build (`rt_predictor = deeplc`) when the multi-head calibration will re-predict every row anyway. Default `true` since 2026-09-27 (previously `false`). With `rt_predictor = deeplc` the automatic multi-head calibration runs, and it rewrites the `predicted_irt` of every standard-residue row against the run's anchors (the only rows it keeps are non-standard ones, and a FASTA digest emits none), so the library's own DeepLC pass is work whose only output is overwritten: about 19 minutes on the 9.8M-peptidoform HYE FASTA library. Nothing reads the library's iRT before the calibration: the seed is iRT-independent and only passes the column through. Set, the orchestrators (`run`, `run-experiment`, and their grouped path) write the native model's iRT as a placeholder, the library's model identity says so, and the run fails if the calibration's summary reports any row it did not re-predict (`retained_imported > 0`), because such a row would keep the placeholder. Ignored where the multi-head calibration does not run, and by the standalone `predict-frag`. Final outputs are then byte-identical to a run with `false`; the intermediate library table, the seed's pass-through iRT column and the predict-frag report differ. The library table is then not a DeepLC library, which matters to anyone who reuses it as `--lib-precursors` elsewhere: set `false` for a build meant to be reused that way. Measured before the default changed, on the E. coli entries of the ProteoBench HYE FASTA against `LFQ_Orbitrap_AIF_Ecoli_01` (MS2PIP `HCDch2`, DeepLC 4.4.0, CPU): `psms_scored.parquet`, the three quant tables and both TSVs byte-identical to a run with `false`, and the run 14:56 -> 12:29. |
| `library_cache` | `Option<String>` | `"auto"` |  | Where `run` and `run-experiment` keep the library they build from a FASTA, to reuse it on a later run. `"auto"` (the default since 2026-09-28): `libraries/` under the engine's cache root, which is `MUMDIA_CACHE_DIR` when it is set and otherwise the per-user cache directory (`$XDG_CACHE_HOME/mumdia` or `~/.cache/mumdia` on Linux, `~/Library/Caches/mumdia` on macOS, `%LOCALAPPDATA%\mumdia\cache` on Windows; `MUMDIA_CACHE_DIR=off` turns `"auto"` off). A path: that directory, as given. `null` or `"off"`: every FASTA run builds its library, the behaviour before 2026-09-28. A FASTA-mode run digests, expands peptidoforms and predicts the library on every invocation, and one `run` per file is the way to search files separately, so each file paid the whole build again (about an hour of predict-frag on the 9.8M-peptidoform HYE library). Set, the orchestrator looks the library up under a key of everything that determines it: the FASTA's content hash, the `digest`, `peptidoforms` and `predict_frag` sections (all but this field), `rng_seed`, whether the iRT is a deferred DeepLC placeholder, the installed MS2PIP, AlphaPeptDeep, DeepLC or IM2Deep version the build uses, the worker scripts' content and the running executable's content. A hit is copied to the paths a build writes (a byte copy, never a hard link, and checked against the blake3 recorded when it was stored) and digest, peptidoforms and predict-frag are skipped; a miss is built and then stored. When a predictor version cannot be read, nothing is looked up or stored. The run stays in FASTA mode, so every other decision is the one a rebuild makes. A hit is byte-identical to a rebuild when the build is deterministic, which the native predictors are; with a sidecar predictor it reuses the stored prediction. The manifest records the two library tables with the stage `library-cache`, and the run directory then holds no `peptides.parquet` or `peptidoforms.parquet` (an earlier build's are removed). Validate by running one FASTA search twice with the same cache directory and comparing `peptides.tsv` and `proteins.tsv` (the smoke test does this). A stored entry whose files changed is rebuilt and replaced, and temporary directories left by a killed store are removed after an hour. The cache is bounded: after a run stores into it, the least recently used entries of this cache and of `rt_im_train.deeplc_projection_cache` are removed until the two together are within `MUMDIA_CACHE_MAX_GB` (default 100 GiB; an entry used within the last hour is kept). The key covers the running executable, so every engine build stores its own entries and the bound is what removes the old ones. `mumdia doctor` reports the directory and its size. |

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
| `two_pass_mass_cal` | `bool` | `true` |  | Robust two-pass fragment mass calibration (sensitivity_plan P3.1). After the first median-offset + tolerance fit, re-fit on only the deviations inside the first-pass tolerance window (rejecting outliers), giving a tighter, more robust offset + local uncertainty. Falls back to the single-pass result when too few in-window calibrants remain. Default true since 2026-10-11. Against the 0.6.0 defaults, mean of three NN seeds: Astral HYE +1.8% peptides, the ProteoBench Astral entrapment module +1.1% precursors, an Astral HeLa entrapment run +2.0% human precursors at a measured FDP of 1%, Orbitrap AIF HYE +0.5%. It costs an Astral single cell 2.9% of its precursors, and the measured FDP at a reported 1% rises slightly in every entrapment test (AIF 0.99% -> 1.02%, ProteoBench paired 0.86% -> 0.88%, HeLa 0.89% -> 0.97%). Set false for single-cell input or where that margin matters. |
| `mass_cal_loess` | `bool` | `false` | benchmark-gated | m/z-dependent fragment mass calibration. When true, fit a LOESS of the calibrant ppm deviation versus fragment m/z and emit a sampled correction grid to `<seed>.masscal.json`; extract then applies an m/z-interpolated offset per peak instead of the single scalar `frag_ppm_offset`. This removes any m/z-correlated curvature the flat offset leaves. Default false (scalar offset unchanged), opt-in and benchmark-gated. |
| `frag_tol_floor_ppm` | `f64` | `5.0` |  | Floor (ppm) on the fragment tolerance learned from the seed's calibrants, which `extract` then searches with. Default 5, the value that was hard-coded. The learned tolerance is `1.5 x p95(\|deviation - median\|)`; on an instrument whose fragment mass error is well under a ppm (Astral) the floor, not the data, sets it. |
| `count_fragments_once` | `bool` | `false` | benchmark-gated | Count each predicted fragment of a candidate at most once per spectrum, with the intensity of its most intense matching peak, instead of once per matching peak. On diaPASEF slot spectra the mobility pieces of one fragment ion sit at one m/z and would otherwise count as several fragments in the hyperscore's `ln(matched!)` (docs/07_search_seed.md). Needs `matcher = fragindex`, whose seed index then also holds each posting's fragment ordinal (2 bytes per posting). Default false: the seed is unchanged. Benchmark-gated (docs/35_diapasef.md). Omitted from the canonical configuration at its default. |
| `frag_tol_mad_k` | `f64` | `0.0` | benchmark-gated | Size the learned fragment tolerance from the robust spread of the calibrant deviations, `frag_tol_mad_k x 1.4826 x MAD`, instead of `1.5 x p95`, where MAD is the median of `\|deviation - median\|`. Still floored at `frag_tol_floor_ppm`. Under `two_pass_mass_cal` the second pass refits only the offset on the deviations inside the first-pass window and keeps the first-pass width: a MAD re-estimated on inliers truncated at that width is narrower than the spread it was cut from. A p95 is set by the tail of the deviations, so a 10% shoulder of random matches can widen it several-fold, where the MAD follows the core. Default 0: the p95 estimator, bit-identical to before the key existed. Benchmark-gated (docs/07_search_seed.md). Omitted from the canonical configuration at its default. |

## rt_im_train

`RtImTrainConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/08_rt_im_train.md](08_rt_im_train.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `calibration_method` | `CalibrationMethod` | `loess` |  |  |
| `q_train` | `f64` | `0.01` |  |  |
| `p_rt` | `f64` | `0.95` |  | Percentile of \|obs - calibrated_pred\| residuals for the RT window. |
| `rt_window_multiplier` | `f64` | `1.0` |  |  |
| `min_seed_for_calibration` | `usize` | `50` |  |  |
| `anchor_q_ladder` | `Vec<f64>` | `[0.02, 0.05]` |  | Relaxed `spectrum_q` cuts tried, in order, when fewer than `min_seed_for_calibration` target anchors pass `q_train`. Rungs at or below `q_train` are ignored; empty keeps the strict cut. When even the last rung is short, the multi-head calibration is skipped (the library keeps base-model retention times) instead of failing on an empty reference. |
| `loess_span` | `f64` | `0.3` |  | LOESS span (fraction of points in each local fit). |
| `loess_robust_iters` | `usize` | `0` |  | Robustness iterations of the RT LOESS (bisquare reweighting on six median absolute residuals). 0 (default) is the plain fit; 2 keeps a handful of false seed anchors from pulling the curve (single-cell runs: residuals of +14 to +82 s moved the calibration 2-3 s across the mid-gradient). |
| `anchor_x_filter` | `bool` | `false` |  | Drop the anchors whose predicted iRT lies outside `[Q1 - 3 IQR, Q3 + 3 IQR]` of the anchors' predicted iRT before anything is fitted to them: the linear and LOESS fits, the window sizing (in-sample, held-out and local) and the composition correction all see the same filtered anchors. A few anchors with an absurd predicted iRT stretch the LOESS grid over the whole x range, so the gradient falls into one or two grid cells; `loess_robust_iters` reweights residuals and does not stop that. `cal.json` records `n_rt_outliers_removed` when anchors were removed. Default false: every anchor is fitted, as before. Omitted from the canonical configuration at its default. |
| `composition_correction` | `bool` | `true` |  | Correct each candidate's calibrated RT for the composition-dependent residual that the predicted RT leaves (a per-run Huber regression of the anchors' residuals on charge 3, charge >= 4, length, internal K/R, His and Met-ox): on the single-cell Astral runs found charge-4 precursors elute 4.6-8.1 s before their calibrated prediction and 8-14% of them fell outside the window. The window width is unchanged; only its centre moves. Decoys share their target's composition, so their windows move with it. On by default since 2026-10-08. |
| `fallback_rt_window_s` | `f64` | `120.0` |  | Fallback fixed RT window in seconds when calibration cannot be fit. |
| `finetune_deeplc` | `bool` | `false` |  | Fine-tune the DeepLC multitask model on this run's confident seed PSMs and rewrite the library's `predicted_irt` before RT calibration. Requires `predict_frag.deeplc_python` (the DeepLC interpreter). Off by default; the main use is library-input mode, where the base iRT comes from the imported library rather than a DeepLC prediction. |
| `multihead_calibration` | `Option<usize>` | `null` |  | Calibrate the DeepLC base model against this run's confident seed PSMs across this many of its best-correlating LC-setup heads, instead of fine-tuning. `null` (the default) is AUTOMATIC: `DEFAULT_MULTIHEAD_HEADS` heads when a DeepLC interpreter is available and `finetune_deeplc` is off, and nothing otherwise. `0` turns it off explicitly; any other number asks for exactly that many heads and is then a hard requirement, so a run without a `predict_frag.deeplc_python` fails rather than quietly doing without. Automatic rather than a plain number because a default that a native, Python-free run cannot satisfy would turn "no interpreter" from a supported configuration into a startup error, and because `finetune_deeplc` occupies the same slot: a configuration that asks for the fine-tune keeps it instead of being refused for a conflict it never wrote. `deeplc.predict` returns ONE of the model's 6,543 heads, the setup named by its `DEFAULT_TASK_NAME`, on that setup's gradient. The per-run LOESS then maps that column onto observed RT, and a smooth increasing curve can stretch and bend the axis but cannot reorder two peptides: whatever elution order that one setup produces survives into the calibrated result. Different chromatography reorders peptides, which is what the multitask model exists to represent, so the ordering is the part the curve cannot repair. `MultiHeadRidgeCalibration` ranks every head against this run's anchors, spline-calibrates the best ones and ridge-combines them. It never fits more head weights than half the reference, so a small anchor set degrades to fewer heads rather than overfitting. Measured on two acquisitions, six pooled runs each, at an unchanged empirical decoy fraction of 0.0100 in every arm (`docs/08_rt_im_train.md` section 4d): AIF +4.8% peptides, Astral +14.3%, protein groups +2.4% and +7.1%. The mechanism is ordering, not thresholds -- fewer candidates reach rescore and more of them are real -- and the cost is 1.4x to 1.7x wall clock, because the calibration is fitted against each run's own anchors and so cannot be shared across an experiment. |
| `multihead_anchors` | `MultiheadAnchors` | `auto` |  | Which anchors the multi-head calibration of the run that adapts the library is fitted on. `seed` is the run's confident seed PSMs at `q_train`. `search` adds a bootstrap pass on that run: the seed-anchored library is searched (rt-im-train, extract, features, compete and a single-run rescore, in `<run>/rt_bootstrap/`), the targets at `precursor_q <= q_train` of that search become the anchors (observed RT = `apex_rt`), the multi-head calibration is refitted on them, and the run is searched again on the refitted library. `auto` (the default since 2026-10-12) chooses per adapting run: `search` when its seed search found fewer than `multihead_search_below` anchors (distinct target base peptides at `q_train`, the count `rt_im_train::plan_anchors` makes) or had to relax `q_train` through `anchor_q_ladder` to find enough, and on every run whose spectra carry 1/K0 (diaPASEF, whatever its anchor count), else `seed`. Under `run-experiment` with `experiment.rt_library_scope = first_run_only` only the first run adapts the library, so only it is ever searched twice; under `per_run` every run decides for itself. The per-run LOESS still fits on the seed anchors. Measured through the engine (2026-10-12, three NN seeds per arm, entrapment FDP unchanged or lower in every set): the gain falls as the seed anchors rise. Single-cell Astral HeLa, 1,249 seed anchors: +5.1% real precursors at entrapment FDP 1%; a 216-cell single-cell plate (100-250 seed anchors per cell): +7.8% / +5.5% precursors; Orbitrap AIF E. coli entrapment, 5,025: +0.9%; ProteoBench Astral entrapment, 13,786: +0.46%. On diaPASEF the gain does not fall with the anchor count (2026-10-14): the six 60 min HYE runs of the ProteoBench diaPASEF module, 11,144 seed anchors, +5.7% peptides and +6.5% ProteoBench ions (one seed); the 15 min E. coli run, 4,639 seed anchors, +0.6% peptides over three seeds with the entrapment FDP 0.43% against 0.46%; hence the 1/K0 rule. The cost is a fixed extract-to-rescore pass and a second library prediction on the adapting run, which reads the DeepLC projection cache the first one wrote: +63 s on a single-cell run, +4 to 5 min (+51% wall) on a three-run bulk Astral experiment with a 17M-row library. `auto` buys the large gain where seed anchors are scarce and leaves bulk runs at their previous speed (`docs/08_rt_im_train.md` section 4f). Fewer search anchors than `min_seed_for_calibration` (distinct target base peptides) or a failure anywhere in the bootstrap pass keeps the seed-anchored library, with a warning. Where no multi-head calibration runs (no DeepLC interpreter, a native retention-time source, `finetune_deeplc`, `multihead_calibration = 0`) or for a grouped run (`groups.window_groups > 1`), `auto` is `seed`; an explicit `search` is refused there by validation, or inert with a warning under the automatic head count. The default, `auto`, is omitted from the canonical configuration, so the config hash of every configuration that does not set the field is unchanged. |
| `multihead_search_below` | `usize` | `2500` |  | Under `multihead_anchors = auto`: an adapting run whose seed search found fewer target anchors than this (distinct target base peptides at `q_train`) uses search anchors. Chosen between the measured points: 1,249 seed anchors gained +5.1%, 5,025 gained +0.9% (`docs/08_rt_im_train.md` section 4f). `0` makes `auto` behave as `seed` except on a run that had to relax `q_train`. Ignored under `seed` and `search`. Omitted from the canonical configuration at its default. |
| `finetune_epochs` | `usize` | `25` |  | DeepLC fine-tune training epochs (passed to `deeplc_finetune.py --epochs`). Early stopping with `finetune_patience` usually halts before this cap, so it is an upper bound rather than a fixed count. Only used when `finetune_deeplc`. |
| `finetune_patience` | `usize` | `10` |  | DeepLC fine-tune early-stopping patience (`--patience`): epochs without validation-loss improvement before stopping. Only used when `finetune_deeplc`. |
| `finetune_batch` | `usize` | `0` |  | DeepLC fine-tune batch size (`--batch`). 0 (default) auto-scales to the confident seed size so each epoch has >= ~30 gradient steps; a fixed large batch underfits small seeds (a ~4k-peptide reference at batch 512 is ~8 steps/epoch and never converges). Only used when `finetune_deeplc`. |
| `adaptive_rt_window` | `bool` | `false` |  | Adaptive RT window (sensitivity_plan spec 03 §3.5, backlog P3.2/P3.3): instead of one global residual-percentile half-width for every candidate, bin the calibration anchors by calibrated RT and give each candidate the LOCAL residual percentile of its RT region, clamped to `[rt_window_min_s, fallback_rt_window_s]` and scaled by `rt_window_multiplier`. A fixed window is simultaneously too wide for well-calibrated regions and too narrow for poorly-calibrated ones; this tightens clean regions (less interference) and widens noisy ones (more recall). Empty/sparse bins fall back to the global width. Default false. |
| `adaptive_rt_bins` | `usize` | `12` |  | Number of equal-width calibrated-RT bins for the adaptive window. |
| `local_window_anchors` | `usize` | `100` |  | Widen-only local RT window: for every calibrated RT, the `local_window_anchors` anchors nearest in calibrated RT give a signed-residual quantile on each side (`(1 - p_rt) / 2` early, `(1 + p_rt) / 2` late) times `local_window_multiplier`, and each side of the window becomes the larger of that and the global `w_rt`, capped at `fallback_rt_window_s`. Where the predicted retention time loses the elution order (typically the gradient ends) the anchors' residuals are one-sided and large, and the window widens on that side only; elsewhere it stays the global window. Default 100 since 2026-10-08; 0 is off. Mutually exclusive with `adaptive_rt_window`. |
| `local_window_multiplier` | `f64` | `1.0` |  | Multiplier on the local one-sided quantiles of `local_window_anchors`. |
| `rt_window_min_s` | `f64` | `1.0` |  | Lower clamp (seconds) for any RT half-window (the existing 1 s floor). |
| `window_holdout_frac` | `f64` | `0.0` | benchmark-gated, do not default | Size `w_rt` from HELD-OUT residuals instead of in-sample ones. A fraction of anchor peptides (`base_peptide_id % 1000 < round(frac*1000)`, so the split is deterministic and shared with `deeplc_finetune.py`) is excluded from the sizing fit and, when `finetune_deeplc` runs, from the fine-tune reference; `w_rt` is then the residual percentile of those held-out anchors against the fit they never entered. The final calibration curve still uses every anchor. In-sample sizing underestimates the tail and rewards a memorizing RT model with a window it does not deserve (measured: it inverted the 4.0.0a2/4.1.0 ranking); held-out sizing measured +0.9% peptides with DeepLC 4.1.0 and -1.5% with 4.0.0a2 on the AIF benchmark, both at 0.98% decoy, so enable it only with a generalizing RT model. 0.0 (default) keeps in-sample sizing. Mutually exclusive with `adaptive_rt_window`. Benchmark-gated; do not default on. |
| `library_irt` | `LibraryIrt` | `auto` |  | Where an imported library's `predicted_irt` comes from. `auto` (the default) re-predicts every peptidoform with the DeepLC base model when `predict_frag.deeplc_python` is configured and keeps the imported values, with a warning, when it is not; `deeplc` requires the interpreter; `library` keeps the imported values. Ignored under `finetune_deeplc` (the fine-tune re-predicts every peptidoform itself) and in FASTA mode (predict-frag already produces DeepLC predictions). Measured on the AIF benchmark with calibration only and native_tda: 10,416 peptides at 1% from DeepLC 4.1.1 base predictions against 10,015 from the DIA-NN library iRT and 10,181 from a per-run fine-tune, with `w_rt` 343 s against 632 s and 472 s (docs/08 section 4c). `run-experiment` predicts once per experiment. |
| `im_min_anchors_per_charge` | `usize` | `50` |  | Ion-mobility calibration (docs/08_rt_im_train.md, "Ion-mobility calibration"): a charge with fewer anchors than this uses the global CCS fit rather than its own. The calibration runs only when the seed carries `observed_im` (spectra with 1/K0) and the library `predicted_im`. Default 50. Omitted from the canonical configuration at its default. |
| `p_im` | `f64` | `0.95` |  | Percentile of the held-out \|observed - calibrated\| 1/K0 residuals that sizes the IM half-window `w_im`, as `p_rt` sizes `w_rt`. Default 0.95. Omitted from the canonical configuration at its default. |
| `im_window_multiplier` | `f64` | `1.0` |  | Multiplier on that percentile. Default 1.0. Omitted from the canonical configuration at its default. |
| `im_window_min` | `f64` | `0.005` |  | Lower clamp of the IM half-window, in V s cm^-2. Default 0.005. Omitted from the canonical configuration at its default. |
| `deeplc_predict_shards` | `usize` | `0` |  | Worker processes for the whole-library DeepLC prediction: the multi-head calibration, the base-model re-prediction under `library_irt`, and the prediction after `finetune_deeplc` (`deeplc_finetune.py --shards`). The calibration or the fine-tuned model is fitted once and handed to every process, and each process predicts a contiguous slice of the unique sequences cut at a multiple of the 100,000-sequence prediction call, so it makes the calls one process would have made. The thread budget (the engine's thread count after the DeepLC thread cap) is divided evenly, so `K` processes get `budget / K` torch threads each. `0`, the default since 2026-09-27, is automatic: one process per 8 threads of the budget, so a budget of 8 or fewer threads (a desktop under the DeepLC thread cap) is one process, as before. `1`, the previous default, is always one process. A GPU always gets one process. Measured before the default changed (EPYC 9354, 128 threads, a budget of 64 capped threads, so 8 processes of 8): a six-run Astral experiment 26:00 -> 18:02, a five-run Orbitrap AIF experiment 29:47 -> 21:04 and an entrapment run 6:27 -> 3:32. The adapted library moved in 413 and 414 of 10,881,402 rows by at most 0.05 and 0.42 s (298 of 5,828,348 by at most 1.8 s on the entrapment library), with the same 80 heads, and the peptides at 1% moved by -0.09%, -0.04% and -0.05% over 10 seeds each, inside the seed spread, at an unchanged entrapment FDP (0.988 -> 0.984%). On the one desktop measured before (an i9, docs/08, "Sharded whole-library prediction") the forward pass dominated at 8 threads or fewer and scaled with threads inside one process, so four processes of two threads were no faster than one of eight: there the automatic count is one process. With `K` processes at the same threads each as one process the `predicted_irt` column is bit-identical (`tests/python/test_deeplc_predict.py`). At the same engine thread count the fit is the same, but each process predicts on `budget / K` threads instead of `budget`, and torch's CPU kernels round differently at a different thread count: most rows move in the last bits, and under the multi-head calibration a few sequences at the edge of the reference range move by up to about two minutes (129 s measured, docs/13, "DeepLC thread cap"). A sharded run is therefore float-equivalent to an unsharded one, not bit-identical. Each process is its own Python process with torch and DeepLC loaded (0.57 GB resident after the model load on the desktop measured, of which the model is about 35 MB), and this step can hold the process-tree peak. |
| `deeplc_projection_cache` | `Option<String>` | `"auto"` |  | Directory for DeepLC's run-independent trunk projection (`deeplc_finetune.py --projection-cache`). `"auto"` (the default since 2026-09-28): `deeplc_projections/` under the engine's cache root (`MUMDIA_CACHE_DIR`, else the per-user cache directory, as for `predict_frag.library_cache`). A path: that directory. `null` or `"off"`: off, the behaviour before 2026-09-28. The two caches share the `MUMDIA_CACHE_MAX_GB` bound (default 100 GiB, least recently used entries removed first). Calibrated RT is `ridge(spline_h(head_h(proj(trunk(x)))))` over the selected heads, and only the head selection, the splines and the ridge depend on a run. The projection, 64 float32 per sequence, depends on the sequence and the model alone, yet every multi-head calibration and base-model re-prediction recomputed it, which is essentially the whole of the step (10:41 of the HYE multi-head step at 96 threads). Set, the first call over a sequence list writes `<dir>/<key>/projections.npy` (the key covers the DeepLC version, the model file and the exact list; 256 B per sequence, about 1.26 GB for HYE's 4.91M) and later calls over the same list read it and evaluate only the heads they need: `rt_library_scope = per_run`, every rerun of an experiment, and the bands of `groups.rt_adaptation = once_per_run` across runs. A miss computes the projection in the prediction's own shard plan (`deeplc_predict_shards`), each shard filling its rows of the entry's file, and the heads are evaluated from the projection in the same plan on a miss and a hit alike. Measured on doxy (128 threads, whole runs, DeepLC 4.5.0): an Orbitrap AIF entrapment run 3:15 without the cache, 3:16 on a miss, 0:56 on a hit; an Astral run on the HYE library 5:14, 5:42 and 2:33. In one process the miss had taken 6:28 and 11:29. Base model only: a fine-tune has no factored head and ignores it. Needs DeepLC 4.5.0 or newer, which added the factored prediction matrix it reads, and 4.5.0 is the engine's floor since 2026-09-28. A later release that moves that private API makes the worker record why in the summary, write nothing and predict exactly as without it; it warns only when the directory was named explicitly, since `"auto"` asks for the cache wherever it is available. Float-equivalent, not bit-identical: the heads are evaluated in numpy from the cached factors instead of in torch. Measured with DeepLC 4.5.0 on CPU: the base-model re-prediction bit-identical on every row of the smoke library (3,820 rows) but not in general (on the 572-row test fixture 109 rows moved, by at most 1.5e-5 s); the multi-head calibration bit-identical on 3,782 of those rows and within 7.6e-6 s on the rest, and on a synthetic 572-row library with sequences outside the anchors' range 402 rows identical and 7 above 1e-3 s, the largest 3.7 s, which is the spline edge amplification a thread-count change shows too (docs/13). A hit took 0.12 s against 7.0 s for the prediction. Validate at scale as a DeepLC version change: per-row max \|delta predicted_irt\|, the selected heads, and peptides at 1% inside the seed spread on two acquisitions. That gate ran before it became the default (DeepLC 4.5.0, multi-head calibration, 10 NN seeds a run): an Orbitrap AIF entrapment run gave the same peptides at 1% seed for seed with and without the cache (mean 10,131.5, FDP 0.984% either way), and an Astral run -0.016% (81,167.0 -> 81,154.0, Welch t -0.18). Hits there took the entrapment run from 8:04 to 1:40 and the Astral run from 12:46 to 5:12. |

## extract

`ExtractConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/09_extract.md](09_extract.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `windows_in_flight` | `Option<usize>` | `null` |  | Isolation windows probed per batch before the candidates no later window can touch are scored and written. `None` (the default) uses the rayon thread count capped at 16, and at 4 on spectra that carry 1/K0 (diaPASEF). The hit accumulator holds the windows in flight, so this sets the stage's peak almost linearly: on the HYE benchmark at 32 threads, 32 in flight is 24.65 GiB, 16 is 16.57 GiB and 8 is 12.31 GiB, with identical output (docs/27 section 3.10). A diaPASEF run has few, dense window groups, and one 60 min HYE run at 20 threads took 45.1 GiB at 16, 31.9 at 8 and 21.4 at 4, fastest at 4, with identical output (docs/35_diapasef.md section 16). It is a memory knob only: each window is probed in parallel over sub-ranges of its candidates, so a small batch still uses every thread. Set 8 or 4 on a memory-bound machine. Not a sensitivity knob. It does move the chromatogram table's parquet row group boundaries, which follow the flush batches, so two runs at different settings produce files that differ byte for byte while holding the same rows in the same order with the same values (measured on one band: 29,028,466 rows, 10.4 billion trace elements, every per-column sum equal). Compare values, not bytes, across settings. |
| `fixed_scan_window` | `usize` | `2` |  |  |
| `frag_tol_ppm` | `f64` | `20.0` |  |  |
| `prec_tol_ppm` | `f64` | `3.0` |  |  |
| `presence_min_matched` | `usize` | `3` |  | tier-(b) minimum matched fragment count. |
| `presence_min_fragments` | `usize` | `3` |  | minimum distinct fragments for acceptance. |
| `presence_min_coelution` | `usize` | `2` |  | minimum simultaneously-present fragments over the consecutive-scan run. |
| `gate_min_score` | `f64` | `0.6` |  | tier-(d) spectral-agreement gate: reject a candidate whose observed fragment intensities agree with the predicted pattern below this score. Renamed from `min_frag_corr`, which was accurate for none of the four `gate_mode` values: under the default `apex_pearson` it is an intensity correlation at ONE apex scan rather than a chromatographic co-elution correlation, and under `spectral_entropy` it is not a correlation at all. The old name is not accepted (`deny_unknown_fields`), so an old config fails loudly with the offending key named rather than silently reverting to a default. Applied symmetrically to targets and decoys, but that alone does not prove null exchangeability in chimeric DIA; validate every threshold with an independent entrapment. 0 disables. |
| `min_matched_fraction` | `f64` | `0.0` |  | tier-(c) minimum fraction of the candidate's predicted fragments that must be observed. With enough predicted fragments (top_n>=~10) this is a strong, symmetric discriminator: real peptides match a large fraction, chimeric false matches and decoys match a small fraction alike, so the target-decoy null stays valid. |
| `apex_top_fragments` | `usize` | `0` |  | Shape-aware apex selection: choose the apex scan group by the summed observed intensity of only the top-K predicted (signature) fragments, rather than all matched fragments. In chimeric DIA a bright co-eluting interferent contributing to arbitrary channels wins a max-over-all-fragments apex; restricting to the peptide's strongest predicted ions locks onto its true elution instead. 0 selects the implementation default of the top 3 predicted fragments. |
| `apex_rt_prior_s` | `f64` | `0.0` |  | Optional Gaussian RT prior on apex selection: weight each scan group by exp(-0.5*((rt - rt_cal)/sigma)^2) with sigma = this value in seconds, so a distant interferent inside a wide RT window cannot define the apex. 0 = off. |
| `apex_count_tol` | `usize` | `1` |  | Fragment-count apex: pick the scan with the most distinct matched fragments, allowing scans within `apex_count_tol` of that maximum (so a slightly-lower- count but much more intense scan can still win), then the max summed-top-3 intensity among them. Supersedes the summed-intensity apex when set. |
| `apex_count_window` | `usize` | `0` |  | Rolling-window width (in scan groups, centered, odd) for the distinct- fragment count that drives apex selection. Low-intensity fragments flicker in and out scan-to-scan; a single-scan count then spikes at noise scans and misplaces the apex. This sums the per-scan distinct-fragment count over a centered window so the apex lands in the region of *sustained* fragment presence, not an isolated flicker. A sum (not a mean) is used deliberately: edge truncation makes interior positions accumulate more, center-weighting the apex toward the RT-window centre (~= predicted RT) as a mild RT-prior; measured to beat a mean by ~+300 IDs on AIF. 1 = no smoothing (per-scan). `"auto"` (also written `0`, the default since 2026-10-11) chooses by the instrument `convert` recorded for the run (`crate::instrument` in the engine): 1 on an Orbitrap Astral, 5 on every other instrument and when none is recorded. Measured one setting at a time against the 0.6.0 defaults, window 1 against 5, mean of three NN seeds: Astral HYE +1.3% peptides, the ProteoBench Astral entrapment module +1.0% precursors (paired FDP 0.84% against 0.86%), an Astral single cell +2.4% precursors and an Astral HeLa entrapment run +3.6% human precursors at a measured FDP of 1% (FDP at the reported 1%: 0.71% against 0.89%); on the Orbitrap (one seed), AIF HYE -0.9% and the AIF entrapment run -0.8%. Astral scans are fast enough that a fragment rarely flickers out, so smoothing over five of them blurs the apex instead of steadying it. A number is used as given. |
| `apex_gaussian_sigma_scans` | `f64` | `0.0` | benchmark-gated | Gaussian matched-filter smoothing of the per-scan fragment-count series before apex selection, as a sigma in scan units. 0.0 (default) keeps the `apex_count_window` rolling-sum smoother unchanged. When > 0, the count series is convolved with a Gaussian kernel (radius = 3*sigma) instead, which localizes the apex more robustly than a uniform window against scan-to-scan flicker. Opt-in and benchmark-gated: it changes apex selection and therefore identifications. |
| `emit_window_grid` | `bool` | `true` |  | Emit per-fragment chromatograms on the FULL isolation-window scan grid with 0.0 where a fragment is absent (aggregating scans of the same isolation window), so the elution profile drops to zero between peaks and the features-stage boundary calling is not misled by interpolated gaps. |
| `chromatogram_schema` | `u32` | `2` |  | On-disk layout of `chromatograms.parquet` (docs/15_data_dictionary.md). `2`, the default since 2026-09-27, stores the axis once per candidate per parquet row group (`rt_axis`) and each trace from its first to its last nonzero value (`intensity_trimmed`), with two extra columns (`trace_offset`, `trace_len`) that rebuild it. `1`, the previous default, stores every row's retention-time axis and its whole trace, zero-filled over the candidate's window in window-grid mode. Every reader (features, quant, the pool) accepts both layouts and rebuilds the same rows bit for bit, so every table downstream of extract is byte-identical; only the chromatogram table changes (smaller, with a different content hash). A reader outside the engine, or an engine binary from before v2, reads only `rt` and `intensity` and stops at their absence from a v2 table: set `1` for such a reader, or convert a table with `mumdia::chromatograms::rewrite`. The pool splices band tables of one layout only, so all bands of a grouped run share it. Measured before the default changed: on a six-run Astral experiment and a five-run Orbitrap AIF experiment (imported HYE library, multi-head calibration, `nn_torch`) every output after extract was byte-identical to v1, and the chromatogram tables went from 4.75 to 3.95 GB and from 21.7 to 15.4 GB. |
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
| `ms1_scan_halfwidth` | `usize` | `1` |  | MS1 scans either side of the nearest one read for a precursor isotope: the apex isotope intensities and every point of the MS1 grid traces take the maximum over the nearest MS1 scan +- this many (default 1 since 2026-10-08). 0 reads the nearest scan only. At low ion counts a mono peak present in both neighbouring MS1 scans is often missing from the one nearest the apex, which then reads as no MS1 signal at all. |
| `emit_rt_window` | `bool` | `true` |  | Write each candidate's RT window (`rt_lo`, `rt_hi`) into `psms_extracted`, so the features stage can scale the RT error by the window it was searched in (the `rtw_*` family). On by default since 2026-10-08; false keeps the previous schema. |
| `ms1_calibrate` | `bool` | `true` |  | Learn the MS1 precursor mass offset from the confident seed anchors before extract and centre every MS1 isotope read on it (`stages::ms1cal`). On by default since 2026-10-08. |
| `ms1_ppm_offset` | `f64` | `0.0` |  | MS1 precursor mass offset in ppm (observed - theoretical) applied to every MS1 isotope read. Set by `ms1_calibrate` in `run` / `run-experiment`; 0 by default. |
| `retain_top_peaks` | `usize` | `1` | diagnostic, not currently | Number of chromatographic peak hypotheses to enumerate per candidate. `K>1` writes up to K local maxima to the diagnostic `<out-psms>.peaks.parquet` sidecar. The primary PSM still contains only the selected apex, so these extra hypotheses are not currently rescored or used to improve identifications. K=1 preserves the single-apex behaviour. |
| `promote_top_peaks` | `usize` | `1` | gated | Number of chromatographic peaks PROMOTED to real feature/rescore rows per candidate (AlphaDIA plan #7, top-K). `1` (default) emits only the selected apex, so the pipeline is byte-identical. `>1` additionally emits the next strongest non-overlapping `enumerate_peaks` groups (each a full re-sliced PSM record carrying `peak_rank`), so the rescorer can pick the correct-but-not-apex peak; the selected apex stays `peak_rank = 0`. Must be `<= retain_top_peaks`. Behaviour-changing and benchmark/entrapment-gated: it changes the extracted row population, and compete/rescore must collapse per candidate so the decoy null is not K-inflated. |
| `alt_peak_min_area_frac` | `f64` | `0.10` |  | Minimum integrated area of a promoted alternate peak (rank >= 1) as a fraction of the rank-0 peak's area. Suppresses noise-level alternates. Only used when `promote_top_peaks > 1`. |
| `alt_peak_min_separation_s` | `f64` | `5.0` |  | Minimum apex-RT separation (seconds) between a promoted alternate peak and the rank-0 apex, so a near-duplicate of the selected peak is not re-emitted. Only used when `promote_top_peaks > 1`. |
| `emit_candidate_audit` | `bool` | `false` | diagnostic | Diagnostic candidate-audit: when true, extraction records, for every probed candidate, either the survivor stage-flags or the earliest `RejectionReason`, and writes `<out-psms>.audit.parquet` (spec 01 §4 / P0.3). Near-zero cost when false (no per-candidate audit allocation). Default false (production). |
| `apex_evidence_rank` | `bool` | `true` |  | Evidence-count apex selection: choose the apex scan by the NUMBER of distinct co-eluting predicted fragments present (breadth of evidence), using observed signature-ion intensity only as a sub-integer tiebreak. In wide-window DIA a single fragment m/z channel is chimeric, so the tallest scan is often a co-isolated interferent; the scan where the most of the peptide's own predicted transitions co-elute is a more reliable apex. Default `true`, on correctness grounds rather than a count: `false` keeps the legacy signature-intensity apex, whose score is 0.0 at every qualifying scan when none of the top-K predicted fragments is observed, so the strict `>` never replaces the first candidate and the apex silently becomes the LOWEST-RT qualifying scan. The rolling distinct-fragment count (`apex_count_window`) still gates which scans qualify in both modes. |
| `apex_refine_intensity` | `bool` | `false` | benchmark-gated | Choose the apex by intensity INSIDE the region the fragment count selects. The qualifying scans (rolling distinct-fragment count within `apex_count_tol` of its maximum) are widened by the rolling half-width (`apex_count_window / 2`), and the apex is the scan in that region with the largest observed intensity of the top-K predicted fragments (`apex_top_fragments`, 3 by default), times the RT prior when it is on. The count still decides WHERE the peptide elutes, so a bright interferent outside the sustained-evidence region cannot take the apex; inside it, intensity decides WHICH scan. Without it the count also picks the scan, and on a tailing peak the rolling count is largest after the intensity maximum: measured on Astral REP1 (`apex_count_window` 5), the summed fragment intensity peaked one scan BEFORE the chosen apex for 41% of the accepted precursors and one scan after for 6%; recomputed with this rule, 85% at the apex, 6% and 5% either side. Default false; benchmark-gated. |
| `emit_gate_diagnostics` | `bool` | `false` | diagnostic | Emit the five gate-diagnostic scores (`gate_apex`, `gate_peak_spectral`, `gate_coelution`, `gate_spectral_entropy`, `gate_window_mean`) as extra `psms.parquet` columns, for the offline gate-metric comparison. Default `false` (diagnostic sidecar, like `emit_candidate_audit`): when off, neither the columns nor the extra per-candidate score computation happen, so the default chain is byte-identical. |
| `gate_mode` | `GateMode` | `window_mean` |  | Which spectral-agreement score the `gate_min_score` gate thresholds (sensitivity program). The legacy gate uses a single apex-scan intensity Pearson, which one chimeric scan can dominate. See `GateMode`. |
| `gate_coelution_min` | `f64` | `0.5` |  | Second threshold for `GateMode::Combined`: the co-elution score must exceed this while the peak-integrated spectral score exceeds `gate_min_score`. Requiring BOTH is more specific (rejects interferents that pass one axis). |
| `gate_rt_weight` | `f64` | `0.2` |  | Weight of the RT error in `GateMode::WindowMean`: the score is the mean of the summed-window entropy, coverage and cosine minus this times the apex RT error over the candidate's own window half-width on that side (clipped to 1). |
| `gate_support_min` | `Option<f64>` | `null` |  | A second acceptance threshold under `GateMode::WindowMean`, the diaPASEF gate: a candidate must pass `gate_min_score` and this, without the MS1 rescue. On spectra that carry 1/K0 the score is the window-mean score plus `0.023 ln(1 + apex intensity)` and `0.009 ln(1 + MS1 monoisotopic intensity at the apex)`, minus `1.23` times the apex 1/K0 error against the calibrated prediction, `7.44` times the 1/K0 spread of the matched fragments at the apex (`apex_im_mad`) and `0.18` times the MS1-to-fragment 1/K0 difference: weights fitted on the gate-time quantities of two HYE diaPASEF runs, on targets only. Without 1/K0 it is the window-mean score plus `0.05 ln(1 + apex intensity)` and `0.02 ln(1 + MS1 mono)`. The MS1 term counts only where the apex reads the mono, +1 and +2 isotopes, as the `log_mono_ms1` feature does. 0 is off. Unset (the default) is 0.845 on spectra that carry 1/K0 and off otherwise. On the six ProteoBench HYE diaPASEF runs, 0.845 keeps 41% of the candidates the window-mean gate at 0.6 passes, for -1.6% peptides at 1% and -36% wall time (26.8 against 41.6 min); on six 5-min HYE diaPASEF runs -1.75% peptides at -48% wall time; on the E. coli diaPASEF entrapment run -1.6% peptides at an entrapment FDP of 0.32-0.35% against 0.30-0.38% (three NN seeds). A window-mean threshold of 0.75 keeps 45% and loses more identifications. Without 1/K0 the gate is not fitted for a default; 1.215 kept 42% on the same runs (docs/35_diapasef.md section 16). |
| `im_gate` | `ImGate` | `off` | benchmark-gated | The 1/K0 gate of a run whose spectra carry ion mobility (a diaPASEF `.d` read by the native reader): a fragment peak whose 1/K0 lies outside the candidate's calibrated 1/K0 window (`run_windows.im_lo`/`im_hi`, from `rt-im-train`) is not matched to that candidate, as a peak outside its RT window is not. See `ImGate`. Default `off`. Benchmark-gated (docs/35_diapasef.md). Omitted from the canonical configuration at its default. |

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
| `bound_from_confident` | `bool` | `false` |  | Elution-peak boundary source. When false (default since 2026-10-04) each candidate detects its own peak boundary from its top-3-predicted-fragment profile. When true, a single set of left/right half-widths (seconds) is learned once from the confident seed PSMs (`spectrum_q <= 0.01`, target-only, the set that anchors RT calibration) and applied to EVERY candidate around its own apex, so a peptide whose peak is wider than the median is cut short. The shared width was introduced to stop a chimeric decoy from fitting its own boundaries; the entrapment measurement below shows no FDR cost from per-candidate bounds. Measured 2026-10-03 (doxy, 5 NN seeds per arm, benchmark config): per-candidate bounds gave Astral REP1 on the HYE library +2.20% peptides at 1% (Welch t +15.7) and the Orbitrap AIF entrapment run +0.17% (t +0.9) at an unchanged entrapment FDP (0.989% -> 1.000%, 2 SE 0.039); the share of IDs whose above-half-maximum peak extends past the bounds fell from 20% / 26% (left / right) to 2% / 6% on Astral. If the seed yields < 20 confident anchors when true, the stage logs a warning and falls back to per-candidate detection for that run. |
| `bound_confident_pct` | `f64` | `50.0` |  | Percentile (0-100) of the confident-set half-widths taken as the global left/ right elution half-width when `bound_from_confident` is true. 50 = median (typical real peak width); higher percentiles widen the shared window. |
| `ms1_precursor_features` | `bool` | `false` | benchmark-gated | Emit the MS1 apex-isotope precursor feature `ms1_isotope_height_corr` (Pearson of the observed apex isotope heights `[i0,i1,i2]` against the Poisson-averagine model). Default false (the feature is present in the battery but returns 0.0, so the vector length is unchanged in effect). It overlaps the existing `ms1_isotope_cosine_apex`, so it is opt-in and benchmark-gated rather than default-on (AlphaDIA-plan item 12). |
| `chrom_loaders` | `usize` | `3` |  | Chromatogram decode threads in the main feature pass. The pass decodes the chromatogram table one chunk at a time while the features of the chunk before are computed; with one loader the whole decode ran on a single core, which bound the stage whenever decoding a chunk took longer than computing one (measured on an 8-12-mer immunopeptidomics run before the decode overlapped the computation: 3.7 of a 4-minute stage were the load). Each loader reads its own chunk from that chunk's row span, and the computation takes the chunks in table order, so the chunks, every feature value and the features table bytes are the same at every setting; only the time and the memory move. The pass holds up to `chrom_loaders + 1` decoded chunks (0.92 GiB of traces each at the HYE benchmark shape, docs/27 section 3.4), where one loader held two. The value is an upper bound: a pass never runs more loaders than `--threads` (the engine's thread pool) or than it has chunks, so `--threads 1` decodes on one loader as before. Loaders beyond each pass's first come from a process-wide pool of four, so concurrent bands or runs (`groups.parallel`, `experiment.parallel_runs`) share that pool instead of multiplying it. Under `experiment.parallel_runs = "auto"` the pool grows to `chrom_loaders - 1` per concurrent run while the per-run chains execute, because the number of chains was sized on the measured peak of a run that had all of its loaders; with the pool of four, three of five concurrent passes on the six HYE diaPASEF runs decoded on one loader and took 280-295 s against 95-104 s. Default 3; `1` restores the single loader and `0` is read as `1`. A memory knob and a speed knob, not a sensitivity knob. |
| `im_features` | `ImFeatures` | `off` | benchmark-gated | The ion-mobility family (`im.rs`, docs/10_features.md): the apex 1/K0 against the calibrated prediction, the fragment 1/K0 dispersion at the apex, the MS1 precursor 1/K0, and the 1/K0 of the fragment traces over the elution peak. Part of the `extended` set only, appended after the other families. See `ImFeatures`. On a run whose spectra carry 1/K0 it also makes extract write the per-point 1/K0 list of each chromatogram row (chromatograms v3/v4). Default `off`. Benchmark-gated (docs/35_diapasef.md). Omitted from the canonical configuration at its default. |

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
| `prefilter_keep_decoys` | `f64` | `0.35` |  | Send only the strongest competed PSMs to the classifier: those a quick cross-validated linear model scores above the decoy that completes this fraction of the decoys (default 0.35); 0 keeps every PSM. The model is fitted per experiment on `prefilter_features` (a default list when unset): for each fold a logistic regression of the targets at `train_fdr` against the decoys of a 400,000-row sample of the other folds, refitted three times, and every PSM is scored by the model of the fold it is held out of, on the rescore's own fold keys. Targets and decoys above the cut are all kept, so the share of rows kept follows the pool: 36% on a six-run diaPASEF pool, 49% on a six-run Astral pool at 0.35. The classifier, its q values and the scored table see only the kept rows; a dropped PSM is not identified. When the model cannot be fitted (too few confident targets, a missing or non-finite column) or the pool has fewer decoys than `min_train_decoys` (100 at least) every PSM goes to the classifier with a warning, and a pool smaller than `prefilter_min_psms` is not cut. Must be in [0, 1). The default since 2026-10-17: at 0.35 within 0.11% of the peptides on diaPASEF, Astral and Orbitrap pools, an unchanged species-entrapment FDP and carbamidomethyl null, for 41-45% of the rescore wall time and 54-60% of its peak memory (docs/35_diapasef.md section 16). |
| `prefilter_min_psms` | `usize` | `1_000_000` |  | The fewest competed PSMs a pool needs before `prefilter_keep_decoys` cuts it (default 1,000,000); a smaller pool goes to the classifier whole. The prefilter is a speed and memory lever for large pools: on the 27,376-PSM Orbitrap AIF entrapment run it saved 2 s of a 10 s rescore and cost 1.3% of the real precursors at an unchanged FDP, while every pool measured from 1.8 million PSMs up lost nothing (docs/35_diapasef.md section 16). 0 cuts every pool. |
| `prefilter_features` | `Option<Vec<String>>` | `null` |  | The feature columns the prefilter model reads; unset is a default list of agreement, co-elution, intensity, matched-count, retention-time, MS1 and peak-shape features. Names the competed table does not carry are skipped with a warning. |
| `python` | `Option<String>` | `null` |  |  |
| `percolator_bin` | `Option<String>` | `null` |  | Path to an external `percolator` executable. Parsed and never read: no stage launches percolator, and `RescorerKind` has no variant that would. It is the only silently inert config field in the tree, since the three MBR ones warn (see `validate`). Kept rather than deleted because the external-percolator path is still intended; `validate` now warns when it is set. |
| `entrapment_marker` | `Option<String>` | `null` |  | Protein-accession substring marking spike-in (entrapment) negatives, e.g. "_HUMAN". Required when `classifier = entrapment`; PSMs whose protein contains it are the empirical false population. |
| `entrapment_exclude` | `Option<String>` | `null` |  | If a protein also contains this substring it is NOT counted as entrapment (the sample's own species, e.g. "_ECOLI"): shared peptides then count as real targets. `None` = the marker alone decides. |
| `entrapment_contaminant_markers` | `Vec<String>` | `[]` |  | Protein substrings marking genuine contaminants inside the spike-in proteome (e.g. "KRT", "ALBU", keratin/albumin entry-name tokens). A PSM matching `entrapment_marker` but also one of these is treated as a REAL target, not an entrapment negative: such peptides are truly present (handling contaminants) so using them as negatives mislabels real signal and inflates the estimated FDR. Empty = every spike-in hit is a negative. |
| `entrapment_ratio` | `f64` | `1.0` |  | N_real_lib / N_entrap_lib. Scales the entrapment FDR estimate so it is unbiased when the spike-in library differs in size from the real one. |
| `strict` | `bool` | `true` |  | When true, any sidecar/classifier failure or misconfiguration (Mokapot or entrapment sidecar error, unwired percolator, entrapment mode with no entrapment PSMs) is a hard error instead of a silent fall back to the native rescorer. Default true so a named scientific workflow cannot silently execute a different model; set false only for explicit legacy compatibility. |
| `handoff` | `Handoff` | `raw` |  | How the feature matrix reaches a sidecar rescorer. See `Handoff`. Defaults to `raw`, which applies to nn_torch only; mokapot and entrapment sidecars always receive the tab-separated PIN. |
| `features` | `Option<Vec<String>>` | `null` |  | Restrict the classifier's input to these feature columns, by name. Absent (the default) falls through to `feature_preset`. The restriction is a projection, not a reordering: the columns keep the order of the feature schema, only those named are read out of the competed table, and the matrix, the sidecar handoff and the training all shrink with the list. Feature selection is a memory and I/O lever, not a speed one (docs/28 section 7), and any list must clear the sensitivity gate before it becomes a default. Mutually exclusive with `RescoreConfig::features_file`. Every name must exist in the competed table's schema; a missing one is an error, never a silent drop. |
| `features_file` | `Option<String>` | `null` |  | The same restriction, read from a file with one feature name per line (blank lines and `#` comments ignored), which is how a 100+ name list stays readable. |
| `feature_preset` | `FeaturePreset` | `all` |  | Named feature list used when neither `features` nor `features_file` is set. `all` is every feature the competed table carries. `compact` is the 114-name list of docs/28 section 12 (`bench/feature_selection/fs_union75_dedup.txt`, embedded in the binary), which with the hard-negative training recipe reproduced the full Extended set within seed noise on three pools (HYE A01 +1.2%, AIF -0.2%, entrapment +4.9%, spike-in FDP unchanged) at 3.4x less rescore memory. Preset names the table lacks are skipped with a log line rather than an error, so a preset tolerates a smaller `features.set`; the intersection must not be empty. Explicit lists stay strict. Default `all`: the projection is a memory lever (3.4x smaller rescore matrix), not a sensitivity one, and it cost 1.2% on the held-out HYE B01 pool under the default training (+0.2% / -0.1% / +1.5% on A01 / AIF / entrapment), so it is the option for pooled rescoring on small machines (docs/28 section 21), not the default. |
| `train_neg_ratio` | `f64` | `2.0` |  | Cap the decoys the sidecar TRAINS on at this multiple of the targets it selected that iteration; 0 trains on every decoy, which is about 19:1 on a DIA pool and is where the rescore spends its time. This thins gradient steps only. Selection, scoring, target-decoy competition and q-values still run over the full pool, so the cap cannot loosen the q threshold; what it can move is the learned boundary. Default 2 since 2026-09-16, measured against the previous 3 with three seeds on two pools and on the entrapment pool: Astral six-run pool 116,711 against 116,309 peptides (+0.35%), HYE B01 63,096 against 63,004 (+0.15%), AIF spike-in library +0.56% real peptides at an empirical FDP of 1.025% against 1.044%, at 16% less rescore wall. `1` with `margin` selection is still the recipe that loses 10% on the entrapment pool (docs/28). |
| `train_neg_select` | `NegSelect` | `hybrid` |  | Which decoys survive `RescoreConfig::train_neg_ratio`. See `NegSelect`. |
| `train_subsample` | `f64` | `0.0` |  | Stratified thinning of whatever survived the cap: a fraction in (0, 1], or a row cap when > 1. Positives and negatives are thinned by the same factor, so the class balance is unchanged. 0 (the default) keeps every row. |
| `train_warm_epochs` | `usize` | `5` |  | Reuse the previous iteration's weights and optimiser state, running this many epochs from the second self-training iteration on instead of a full fresh fit. 0 (the default) refits from scratch every iteration, which is 25 epochs x 10 iterations x 3 folds of the whole training set. |
| `train_margin_frac` | `f64` | `0.5` |  | Under `train_neg_select = hybrid`, the share of the negative budget taken from the margin (highest-scoring decoys); the rest is sampled at random. Default 0.5. |
| `seeds` | `usize` | `1` |  | Independent self-training passes whose out-of-fold scores are rank-averaged. 1 (the default) is a single pass. 3 was the one knob positive on every pool of the seeded sweep (docs/28 section 17), at three times the training cost. |
| `shadow_min_shared` | `usize` | `3` |  | Shadow demotion: demote (below every other score) a candidate of either label when at least this many of its library fragments lie within `shadow_ppm` of a fragment of a higher-scoring, first-pass-accepted target in the same run and isolation window with its apex within `shadow_rt_s`; the candidate's own base peptide never lends. Default 3 since 2026-10-08; 0 is off. Without the inputs below it is skipped with a warning. Needs the library fragments and the runs' isolation windows, which `run` and `run-experiment` pass and `rescore` takes as `--lib-fragments` and `--isolation-windows`. |
| `shadow_ppm` | `f64` | `10.0` |  | Fragment m/z tolerance (ppm) of the shadow test. |
| `shadow_rt_s` | `f64` | `3.0` |  | Apex RT distance (seconds) within which a lender counts. |
| `shadow_region` | `usize` | `8` |  | Candidates examined per run: the top `shadow_region x accepted` rows by first-pass score. |
| `shadow_im_tolerance` | `f64` | `0.0` | benchmark-gated | The 1/K0 term of the shadow test on a run whose spectra carry ion mobility: a lender counts only when its apex 1/K0 is within this distance (V s cm^-2) of the candidate's own. Two precursors co-isolated in one diaPASEF slot but separated in mobility do not share fragment peaks, so a shared fragment m/z is not borrowed evidence there. The apex 1/K0 is joined from each run's `psms_extracted` on `(candidate_id, selected_peak_rank)`; a row without one (3D data, or a lender or candidate whose apex 1/K0 is null) keeps the m/z-and-RT test alone. Default 0: off, bit-identical to before the key existed. Benchmark-gated (docs/35_diapasef.md). Omitted from the canonical configuration at its default. |
| `min_train_decoys` | `usize` | `100` |  | Fewest decoys the classifier is trained with. Below this the learned classifier is not trained (a handful of decoys lets it overfit, one decoy then outranks every target and nothing passes 1%): the candidates are ranked by their untrained `prelim_score` and q-values come from plain target-decoy competition, with a warning and `classifier = "prelim_score_few_decoys"` in the scored table's report. A strict gate on a very small or very clean run can leave this few. 0 disables it. |

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
| `fragment_selection` | `FragmentSelection` | `observed_area` | benchmark-gated | Which fragments enter the top-N sum. `observed_area` (default, legacy) ranks by the integrated area itself, which preferentially selects interfered fragments (their areas are inflated) and so varies run to run. `predicted` ranks by the library (predicted or empirical) fragment intensity, a per-precursor constant, so every run sums the same fragments. Astral HYE 2026-08-26: CV 0.163 -> 0.112 on 6/6 ions at top-3. Benchmark-gated. `cross_run` weights each fragment by its consistency across the runs of an experiment (`FragmentSelection::CrossRun`); benchmark-gated as well. |
| `fixed_scan_halfwidth` | `usize` | `3` |  | When > 0, integrate each fragment over the `2k+1` scans centred on the identification apex instead of the descent-walk window (`bound_peak` window ignored; falls back to it when the apex is unknown). A fixed narrow window is far less sensitive to interference in the peak wings than the walked bounds. Default 3 (seven samples) since 2026-10-04; 0 restores the walked window. Measured on the six-file Astral HYE experiment (one seed, 63,141 precursors quantified in at least 2 runs per condition): median \|log2 ratio error\| 0.266 -> 0.163, species-equal 0.322 -> 0.201, median CV 0.19 -> 0.10, the identifications unchanged. The width is in samples of the precursor's window, so it scales with the cycle time; check it on an acquisition with much wider or narrower peaks. |
| `baseline_subtract` | `bool` | `false` | benchmark-gated | Subtract a per-fragment local background before integrating (fixed-scan window only). The background is the `baseline_quantile` quantile of the intensities in the two flanks (`baseline_flank_scans` samples on each side of the integration window); window intensities are clipped at zero after subtraction. Targets the additive floor that compresses ratios in the low-abundance condition. Off by default; benchmark-gated. |
| `baseline_flank_scans` | `usize` | `12` |  | Flank length (samples per side) used to estimate the background. |
| `baseline_quantile` | `f64` | `0.25` |  | Quantile of the flank intensities taken as the background level. |
| `fixed_window_s` | `f64` | `0.0` |  | When > 0, integrate each fragment over the samples within `fixed_window_s` seconds of the identification apex (instrument-independent alternative to `fixed_scan_halfwidth`, which it overrides). 0 (default) = off. |
| `baseline_source` | `BaselineSource` | `run` | benchmark-gated | Where the flank background of `baseline_subtract` comes from. `run` (default): the `baseline_quantile` quantile of this run's flanks. `experiment`: one level per (candidate, fragment), the mean over every run of the raw flank mean, which the cross-run step of `run-experiment` fits and subtracts in place of the per-run quantile (docs/12, "Cross-run fragment selection and background"). Needs `baseline_subtract = true`. Without the cross-run step (`run`, or `quant` without `--cross-run-from`) the per-run quantile applies. Benchmark-gated. Omitted from the canonical configuration at its default. |
| `fixed_scan_halfwidth_scale` | `f64` | `0.0` | benchmark-gated | When > 0, the cross-run step integrates each candidate over its own scan halfwidth `h = clamp(round(fixed_scan_halfwidth_scale x HWHM), 3, 7)` in every run, in place of `fixed_scan_halfwidth`. HWHM is the half width at half maximum, in scans, of the candidate's summed fragment trace in its brightest run, measured in the first pass; the experiment background is then taken beyond that window. Needs the scan form of the fixed window (`fixed_scan_halfwidth > 0`, `fixed_window_s = 0`) and `baseline_source = experiment`. 0 (default) = off. Benchmark-gated. Omitted from the canonical configuration at its default. |

## mbr

`MbrConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `strategy` | `MbrStrategy` | `second_pass` |  |  |
| `q_anchor` | `f64` | `0.01` |  | The first-pass cut (validated at 0.01). Under `second_pass`, the experiment-wide `precursor_q` at which a target enters the empirical library, and the `run_psm_q` at which one run's apex counts toward its consensus retention time. Under the transfer variants, the q-value for a precursor to become a cross-run anchor. |
| `min_anchor_runs` | `usize` | `2` |  | Transfer variants only: minimum number of OTHER runs a precursor must be confident in to transfer. |
| `q_transfer` | `f64` | `0.01` |  | Transfer variants only: accept threshold for a transfer's q-value. |
| `rt_window_s` | `f64` | `20.0` | benchmark-gated, not yet wired | NOT YET WIRED. Transfer RT half-window (seconds) around the cross-run-predicted RT. The M2 leave-target-out residual was ~17 s at p95, ~15x tighter than the search window, which is where this default comes from -- but no code reads this field yet, so setting it has no effect. Kept as the recorded design value for the MBR transfer tier; `validate()` warns if it is changed from the default. See CLAUDE.md, "MBR transfer/re-extraction remains benchmark-gated". |
| `decoy_transfer` | `DecoyTransfer` | `permuted_rt` | not yet wired | NOT YET WIRED. Which decoy-transfer null would estimate the false-transfer rate (M4). No code reads this field yet; `validate()` warns if it is changed. |
| `consensus_corr_min` | `f64` | `0.0` |  | Minimum correlation of the observed fragment pattern to the empirical consensus for a transfer to be accepted (interference guard; 0 disables). |
| `requant_all` | `bool` | `false` | not yet wired | NOT YET WIRED. Would requantify already-identified precursors too (fill the matrix), not only transferred ones, under `strategy = Full`. No code reads this field yet; `validate()` warns if it is changed. |
| `python` | `Option<String>` | `null` |  | Python interpreter for the `mbr_worker.py` sidecar (pandas/pyarrow/numpy; e.g. the `py312_mumdia` env). Required by the transfer variants only; `second_pass` and `none` run no sidecar. |

## experiment

`ExperimentConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/01_overview_and_dataflow.md](01_overview_and_dataflow.md).

Options for the experiment-wide orchestrator (`mumdia run-experiment`).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `parallel_runs` | `usize` | `0` |  | How many per-run search chains to execute concurrently, or `"auto"`. `"auto"` (also written `0`) is the default since 2026-09-27; `1`, the previous default, is strictly sequential. Runs are independent, so concurrency scales nearly linearly in wall time, but EACH concurrent run holds its own extraction working set (tens of GB on a large library), so the practical ceiling is memory, not cores, and `"auto"` measures it. Results are unaffected: chunks are processed in index order and completion order never reaches the output. An explicit number runs the chains in chunks of that size on the engine's one thread pool, as it always has. `"auto"` sizes the concurrency from the thread budget: at most one run per 16 threads of `--threads`, and never more than the runs left. Each concurrent chain then runs in a thread pool of its own, `threads / runs` wide, so extract's fan-out and the feature loaders are sized to that run's share instead of each to the whole pool, and a run starts as soon as a slot is free rather than at a chunk boundary. On Linux the conversions, the seeds and the chains are each bounded by memory: the first of them runs alone, from a reset high-water mark (`/proc/self/clear_refs`), and the rest run at most `0.7 x (available + resident) / peak` at once, where the peak is the process's `VmHWM` after that step and `available` is `MemAvailable`, lowered to the headroom of the process's memory cgroup (a container's or a batch job's limit) when that is smaller. When the first run adapts the library with a search of itself (`rt_im_train.multihead_anchors`, the bootstrap pass), only that adaptation runs alone: its peak sizes the chains, and the first run's own chain then runs among them. The peak counts this process only, not its child processes, so chains that each run their own DeepLC adaptation (`rt_library_scope = per_run`) run one at a time under `"auto"`; give a number to run such chains concurrently. Elsewhere (any platform but Linux) there is no memory reading, so `"auto"` runs the chains one at a time and the log says why; give a number to run them concurrently there. Because of the reset, a lifetime peak read from the process's resource usage (`/usr/bin/time -v`) covers only the time since the last measured step; measure an `"auto"` experiment's memory with a sampling profiler (`bench/mem_profile.py`). The rows are the same as at `1`; a stage inside a narrower pool can lay out its intermediate files differently, as a different `--threads` does. Measured before the default changed, together with `extract.chromatogram_schema = 2` and `overlap_front_threads = "auto"`: on a six-run Astral experiment and a five-run Orbitrap AIF experiment (EPYC 9354, 128 threads) every final output was byte-identical to the sequential run's (24 and 21 tables), and the experiments went from 32:49 to 26:00 and from 44:37 to 29:47. |
| `rt_library_scope` | `RtLibraryScope` | `first_run_only` |  | How often the library's retention times are adapted to a run: once on the first run and reused (`first_run_only`, the default) or separately for every run (`per_run`). Governs whichever adaptation is active -- `rt_im_train.finetune_deeplc` or `rt_im_train.multihead_calibration` -- because they are the same shape of work: one full re-prediction of the library against that run's confident seed PSMs, which on a 9.4M-row library is the most expensive step in the experiment. Each run then fits its own LOESS on top of whichever library it was given, and that per-run fit is what absorbs chromatographic drift. Accepts the old name `finetune_scope`, which is what it was called when only the fine-tune could be shared. `first_run_only` assumes the runs share an elution ORDER, which replicate injections on one LC method do. A per-run LOESS can stretch and bend the axis but cannot reorder two peptides, so a batch that genuinely reorders -- different gradients, different columns, a method change part-way -- wants `per_run`, and so does a long batch where drift accumulates (see the measured cost above). |
| `overlap_front_threads` | `Option<usize>` | `null` |  | Threads given to converting and seeding runs 2..N while run 1 adapts the library's retention times, under `rt_library_scope = first_run_only`, or `"auto"` (also written `null`), the default since 2026-09-27. `0`, the previous default, runs them after run 1. `"auto"` gives the fronts the threads of the budget beyond the physical cores the process may run on, which is what the DeepLC thread cap leaves idle: run 1's DeepLC then gets exactly the physical cores, the thread count it would have had without the overlap, so the adapted library is bit-identical. Where the physical cores cannot be read (any platform but Linux), where the budget does not exceed them, or where `MUMDIA_DEEPLC_THREAD_CAP` sets the cap by hand, `"auto"` is `0`. Runs 2..N convert their spectra and seed on the base library (the seed is iRT-independent), so nothing of theirs waits for run 1's adapted library until rt-im-train. Set to `N`, those fronts run on a pool of `N` threads while run 1's DeepLC sidecar gets the remaining `threads - N` (disjoint budgets), and every run's rest follows once run 1 has finished. On the six-file HYE Astral experiment a front is convert 1.9-2.5 min plus seed 0.4 min per file, against a first-run multi-head step of 11.7 min, so up to about 12 minutes of fronts fit behind it. Ungrouped runs only: a grouped run seeds per band inside its band loop, and its adaptation sits between those band seeds and its extract. The fronts' outputs are byte-identical; run 1's DeepLC predicts on `threads - N` torch threads instead of `threads`, which moves the adapted library in the last bits unless the DeepLC thread cap binds both counts to the same number (on an SMT host with `N` below the logical-minus-physical core count it does, and `"auto"` chooses exactly that). An explicit count beyond it is float-equivalent, not bit-identical. The fronts keep the ungrouped experiment's phase order: every conversion first, then one seed library and fragment index for all of their seeds (run 1 seeds on its own load before the overlap starts). That library is held beside the DeepLC worker while the fronts seed, which is where the experiment's peak can sit. Measured with `parallel_runs`: see there. |

## groups

`GroupsConfig` (rust/mumdia/crates/mumdia-core/src/config.rs).

Searching a run one isolation-window group at a time. A group of isolation windows can only select precursors whose m/z lies in the group's band, so its seed, calibration, extract, features and compete need only that band of the library (`Library::load_with_fragment_offset`): the library, the hit accumulator and the accepted rows are all one band's worth instead of the whole run's, which is what bounds the memory of a search against a library of 10^8 precursors. Only rescore, quant and report see everything, after the group artifacts are pooled with library-wide ids. The groups run one after another in this process; `docs/33_window_groups.md` has the layout and the measurements.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `window_groups` | `usize` | `1` |  | Number of window groups. `1` (the default) is the ordinary single-library search. Groups are contiguous bands of isolation windows balanced by the number of library precursors they select, read from the precursor table's row-group statistics. |
| `calibration` | `GroupCalibration` | `global` |  | Anchors for the RT calibration of each group; see `GroupCalibration`. |
| `parallel` | `usize` | `1` |  | Bands searched at the same time inside one run. `1` (the default) is one band at a time, which is what bounds the memory: each band in flight holds its own extraction working set, so the peak is this many bands' worth. Raise it to fill a large machine, after checking one band's peak RSS: on a 203M-precursor library at 63 bands the largest band took 39 GB and the median far less. Results do not depend on it; bands are independent and their artifacts are pooled in band order either way. The bands go through a bounded queue: this many workers each take the next band as soon as their current one is done, most expensive first (estimated precursors times MS2 peaks of the band's windows for the seed and extract, accepted rows for features and compete), rather than in fixed chunks that waited for their slowest band. It must stay below the thread count: a band in flight parks one worker on its accumulation channel, so as many bands as there are threads leaves nothing to do the probing and the run deadlocks. A larger value is clamped to `threads - 1` with a warning rather than hanging. |
| `rt_adaptation` | `GroupRtAdaptation` | `once_per_run` |  | How often the library's retention times are adapted under `calibration = global`: `once_per_run` (the default since 2026-09-27) runs one DeepLC sidecar per run over the union of the bands, `per_band` (the previous default) one per band. `per_group` calibration always adapts per band. Each per-band sidecar starts an interpreter, imports torch and DeepLC, reads the pooled seed, refits the same heads on the same anchors (head 2503 in every band of the HYE sweep) and predicts every sequence of its band, so a sequence whose charge states fall in two bands is predicted twice (10.9M HYE rows are 4.91M unique sequences). On HYE Astral the multi-head step took about 13 min unbanded and 19-24 min at 2-16 bands. `once_per_run` fits once, predicts the union once and writes each band's table under the name a per-band run gives it (`groups/gNN/lib_precursors_multihead.parquet` or `lib_precursors_deeplc.parquet`), so the shared-band reuse of later runs and the seed refresh are unchanged. Under `run-experiment` with the multi-head calibration off, the library is re-predicted once for the experiment, and the bands then keep those values instead of each band of each run re-predicting them. Float-equivalent, not bit-identical: a sequence is predicted in different company, and torch's CPU kernels round by batch. On a synthetic library, one call over contiguous bands writes exactly the whole-library column band by band (`tests/python/test_deeplc_predict.py`). Measured before the default changed (EPYC 9354, 128 threads): a six-run Astral experiment at 16 bands 36:06 -> 25:39 and an eight-band entrapment run 5:44 -> 3:45; 0.8% of the band rows moved, by at most 0.07 s (0.6%, at most 2.9 s, on the entrapment library), with the same heads; and the peptides at 1% +0.35% (3 seeds) and +0.24% (10 seeds, the entrapment FDP +0.022 pp against two standard errors of 0.027). |
| `balance` | `GroupBalance` | `precursors` |  | What the band plan balances: `precursors` (the default), the estimated library precursors per band, or `cost`, per window the precursors it selects times the MS2 peaks of its scans. Band cost follows spectral density more than precursor count: on the immunopeptidomics search two bands of 2.98M and 3.03M precursors took 42 s and 460 s, and at 81-94 bands the slowest windows (418-460 s) set the floor of the run. The queue already starts the most expensive bands first whatever this says; `cost` also moves the cuts, so that no band is several times the work of the others. Output-changing, hence opt-in: the cuts decide which candidates sit at a band edge, which the overlap deduplication and the edge candidates' neighbours depend on, and the pooled row order the classifier sees. Validate like a band-count change (docs/33 section 8): peptides at 1% inside the seed spread against `precursors`, and the per-band wall times, on two acquisitions. Note that the MS2 is decoded before the plan under either setting. |
| `delete_band_intermediates` | `bool` | `true` |  | Delete each band's `psms_extracted.parquet` and `features.parquet` (with their reports and schema companions, and `run.pin` where one was written) once the pool is written. Default `true` since 2026-09-27 (previously `false`). Disk only: no stage reads them after pooling. The features are carried by the competed table and the extracted table's one reader, the candidate audit, reads the pooled copy. On the immunopeptidomics runs the band features alone were 55 GB per run, in an experiment that wrote about 2.7 TB of artifacts. The manifest keeps their records, and the band directories can no longer be re-featured (set `false` to keep them for that); the chromatograms and competed tables that quant, rescore and `mumdia pool --groups-dir` read are kept. Measured before the default changed, on an eight-band entrapment run: `psms_scored.parquet`, the three quant tables and both TSVs byte-identical. |
| `pool_competed` | `bool` | `false` |  | Write the run's pooled `psms_competed.parquet`. Default `false` since 2026-09-26 (previously `true`). With `false`, rescore reads the bands' own competed tables in band order, each with the run's `source`, and the pooled copy is not written: one full write and read of the run's widest artifact less (17.5 GB on one immunopeptidomics run of 63 bands, about 83 GB per run before the feature columns were narrowed to f32). This happens only where it cannot change a result: the bands' library row spans must be disjoint, so no candidate was searched in two bands and there is no overlap duplicate to drop, and neither the candidate audit (`extract.emit_candidate_audit`) nor match-between-runs (`mbr.strategy`), which read the pooled table, may be on. Otherwise the table is pooled as with `true`, and the log says why. `psms_scored.parquet` is byte-identical either way; what changes is the artifact set. The run's manifest then has no pooled `psms_competed` record, the scored table's report lists the band tables under `competed_inputs` with their `competed_sources`, and a later standalone `mumdia rescore` or audit needs the table rebuilt first with `mumdia pool --groups-dir`, which the band tables allow. `true` writes the pooled table as before. The default changed on the byte-identity, measured with `pool_chromatograms` off alongside (`docs/33_window_groups.md` section 5): on one immunopeptidomics run of 63 bands, `psms_scored.parquet`, the three quant tables and both TSVs were identical to the pooled run's, the run directory went from 152 to 80 GB, and the pool stage from 66 s (30 min on a disk that sustained 22 MB/s) to nothing; on a six-run HYE experiment of 16 bands a run, all 36 outputs the two arms share were identical. |
| `pool_chromatograms` | `bool` | `false` |  | Write the run's pooled `chromatograms.parquet`. Default `false` since 2026-09-26 (previously `true`; see `pool_competed` for the measurement). With `false`, quant reads the bands' own chromatogram tables in band order, dropping from each the candidates the pool's overlap dedup gave to another band, and the pooled copy is not written: one full splice write, hash and read of the run's largest artifact less (59.5 GB on the immunopeptidomics run above). The pool writes those loser sets to `groups/overlap_losers.parquet` (`band`, `candidate_id`, with the band tables it belongs to named in its footer), so a later `mumdia quant --chromatograms <band tables> --overlap-losers <that file>` reads the same rows, and refuses band tables that are not the named ones in their order. Unlike `pool_competed` this holds for overlapping bands as well, including an overlap candidate compete deleted in one band (the losers are found in the chromatogram tables themselves), and nothing but quant reads the pooled table (the candidate audit and match-between-runs do not). The quant tables are byte-identical either way; what changes is the artifact set. There is no pooled `chromatograms.parquet` (one an earlier run left in the directory is removed) and no manifest record for it, `overlap_losers.parquet` is written and recorded instead, and the quant report's `chromatograms` lists the band tables with `chromatogram_dropped_candidates`. The band chromatogram tables are then the run's only chromatograms, so the band directories are no longer disposable once the run is accepted: deleting them loses re-quantification. `mumdia pool --groups-dir` rebuilds the pooled table from them. `true` writes the pooled table as before. |

## retrace

`RetraceConfig` (rust/mumdia/crates/mumdia-core/src/config.rs). stage document: [docs/09_extract.md](09_extract.md).

Chromatogram traces rebuilt from the raw timsTOF events (the `retrace` stage, between extract and features; docs/35_diapasef.md section 6, docs/09 section 9). Each fragment trace point becomes the sum of every raw event in the grid point's frame and in the quad slots whose isolation window holds the precursor, within the learned fragment tolerance and inside `apex_im +/- im_half_width`, instead of the intensity of one centroid. The MS1 isotope traces are rebuilt the same way from the MS1 frames extract samples. diaPASEF only: a run whose spectra were not read natively from a `.d` is refused. Every key is omitted from the canonical configuration at its default, and the whole section with them, so a configuration that does not set it keeps its config hash.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `enabled` | `bool` | `false` | benchmark-gated | Run the stage inside `run` and `run-experiment` (and the second pass), after extract. Extract then writes its centroid traces to `chromatograms.centroid.parquet`, which is deleted once retrace has written `chromatograms.parquet`. Not in the RT bootstrap pass, which only collects anchors. Default false. Benchmark-gated (docs/35_diapasef.md). |
| `repick` | `bool` | `false` | benchmark-gated | Re-choose each candidate's apex retention time and 1/K0 centre on the raw events before the traces are rebuilt, among extract's apex and the peaks of the `extract.retain_top_peaks` sidecar. Needs `retain_top_peaks >= 2` and `promote_top_peaks = 1`, and is refused with `extract.apex_refine_intensity`. It overrides the apex that `apex_evidence_rank` chose, by design. Default false. Benchmark-gated. |
| `im_half_width` | `f64` | `0.015` |  | Half-width in 1/K0 of the band around the candidate's `apex_im` (`im_pred_cal` when that is null) inside which raw events are summed, for the fragment and the MS1 traces alike. Measured flat from 0.012 to 0.020 on one diaPASEF run. Default 0.015. |
| `ms1` | `bool` | `true` |  | Also rebuild the MS1 isotope traces (`ms1_mono`, `ms1_iso1`, `ms1_iso2`) and their apex scalars. False copies the centroid MS1 rows and keeps extract's MS1 scalars. Default true; kept as a switch until an engine ablation of the raw MS1 band confirms the prototype's +1.6%. |
| `apex_features` | `bool` | `false` | benchmark-gated | Write `imc_ref_w` into `psms_extracted`, the agreement of the fragments' raw 1/K0 profiles at the apex, which the ion-mobility feature family then carries as its tenth feature. Needs `enabled` and a `features.im_features` other than `off`. Default false. Benchmark-gated: alone it measured +1.3% peptides, and the added peptides carried about 2.7% FDP. |

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
| `dia` | `features.set` = `extended`; `extract.apex_count_window` = `0`; `extract.apex_rt_prior_s` = `120.0` |

A profile is applied after the config file is parsed, so it wins over a
value the config file set for the same field.

## Enumerations

Accepted values for every enum-typed field above, with the serde spelling a
config file must use. The default variant is marked. Sorted by type name.

### `BaselineSource`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Source of the flank background level. See `QuantConfig::baseline_source`.

| Value | Default | Description |
|---|---|---|
| `run` | yes | The `baseline_quantile` quantile of this run's flanks. |
| `experiment` |  | The mean flank level over every run of the experiment (cross-run step). |

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
| `trypsin_p` | yes | Trypsin/P: cut after K or R (including before P). Rule `K*,R*`. |
| `trypsin` |  | Classic trypsin: cut after K or R but not before P. Rule `K*,R*,!*P`. |
| `lys_c` |  | Lys-C: cut after K. Rule `K*`. |
| `lys_n` |  | Lys-N: cut before K. Rule `*K`. |
| `arg_c` |  | Arg-C: cut after R, not before P. Rule `R*,!*P`. |
| `asp_n` |  | Asp-N: cut before D. Rule `*D`. |
| `glu_c` |  | Glu-C: cut after E (bicarbonate buffer). Rule `E*`; `E*,D*` as a custom rule. |
| `chymotrypsin` |  | Chymotrypsin, high specificity: cut after F, Y or W, not before P. Rule `F*,Y*,W*,!*P`. |
| `non_specific` |  | Non-specific: every peptide bond (rule `**`). Peptides are enumerated by length (`digest.min_len..=digest.max_len`), so `digest.missed_cleavages` does not apply, and N-terminal Met excision adds nothing (every start position is enumerated). For immunopeptidomics and peptidomics; the search space grows with the length range, about one peptide per residue and length. |
| `custom` |  | The rule in `digest.cleavage_rule` (residue pairs around the cut, `*` any, `!` never; the syntax of DIA-NN `--cut`), e.g. `K*,R*,D*,!*P`. |

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
| `cross_run` |  | Weight every fragment by its consistency across the runs of an experiment, fitted by the cross-run step of `run-experiment`: the quantity is `sum(weight x area)` with the same weights in every run (docs/12, "Cross-run fragment selection and background"). Where no weight was fitted (single run, `quant` without `--cross-run-from`, a candidate with no positive weighted area), `predicted` applies, with `top_n_fragments`. |

### `GateMode`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Spectral-agreement score the extraction acceptance gate (`gate_min_score`) thresholds. All are computed at the gate from data already in hand.

| Value | Default | Description |
|---|---|---|
| `apex_pearson` |  | Legacy: Pearson of observed-vs-predicted fragment intensities at the single apex scan. One chimeric scan can dominate it. |
| `peak_spectral` |  | Pearson of the PEAK-INTEGRATED observed spectrum (each fragment summed over the elution-peak scans) vs predicted intensities. Averages out a single interfered scan; the standard library-dot-product measure. |
| `spectral_entropy` |  | Li spectral-entropy similarity of the sqrt-transformed apex-scan observed vs predicted intensities (`spectral_entropy_similarity_sqrt`). The full-feature gate search (all ~379 features, target-vs-decoy) found this the single best gate discriminator: AUC 0.826 / matched-pool recall 69.8%, versus apex Pearson's 0.781 / 64.5%. Same inputs as `ApexPearson`, better separation. |
| `coelution` |  | Predicted-intensity-weighted mean CO-ELUTION correlation of each matched fragment's XIC to the signature reference over the elution peak (temporal agreement, orthogonal to intensity agreement). |
| `combined` |  | Require BOTH: peak-integrated spectral Pearson >= `gate_min_score` AND the co-elution score >= `gate_coelution_min`. More specific (an interferent passing one axis is still rejected), for a cleaner FDR pool. |
| `window_entropy` |  | Spectral-entropy similarity (sqrt) of the spectrum summed over the apex +-2 scans of the isolation window's acquisition grid, against the library. At near-single-ion intensities a fragment is present in one scan and absent in the next, so any single-scan score is noise; summed over five scans it is not. On the Astral HYE and Q Exactive AIF HYE runs a threshold of 0.6 kept 42% / 35% of the candidates with 99.99% / 99.90% of the gate-0 identifications, where the apex Pearson at 0.2 kept 32% and lost about 10%. |
| `window_mean` | yes | Mean of three agreement scores of the spectrum summed over the apex +-2 scans of the acquisition grid (sqrt entropy similarity, the share of predicted intensity observed, sqrt cosine), minus `gate_rt_weight` times the apex RT error over the window half-width on its side. On the Astral and Q Exactive HYE runs the top 10% of candidates by this score held 99.4% / 99.6% of the gate-0 identifications, and the top 15% held 99.9%. The default since 2026-10-08, with `gate_min_score` 0.6. |

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
| `per_band` |  | One DeepLC sidecar per band, each fitting the pooled anchors and predicting its own band. The behaviour before this setting existed, and the default until 2026-09-27. |
| `once_per_run` | yes | One sidecar per run over the union of the bands: the calibration is fitted once, each unique sequence is predicted once, and every band's table is written under the name a per-band run gives it. A library the caller already re-predicted with the base model (`run-experiment` with the multi-head calibration off) is not re-predicted per band again. The default since 2026-09-27. |

### `Handoff`

(rust/mumdia/crates/mumdia-core/src/config.rs)

How the feature matrix crosses the Rust -> Python boundary for a sidecar rescorer.

| Value | Default | Description |
|---|---|---|
| `tsv` |  | Tab-separated PIN. Percolator's format, and what `mokapot.read_pin` requires, so it is what a mokapot or entrapment sidecar receives whatever this is set to. |
| `parquet` |  | Parquet feature table with f32 features: the default from 2026-09-05 until `Raw` replaced it on 2026-09-28, with identical scores. The TSV path makes the worker parse every column into a float64 pandas frame before it builds its float32 matrix, so the text file, the frame and the matrix are alive together. Measured on the HYE competed table (2,603,894 PSMs x 387 features, one self-training iteration, 32 threads), parquet against tsv: rescore peak 29.96 -> 8.95 GB, wall 8:35 -> 6:33, sidecar file 9.53 -> 3.28 GB, the worker's read and standardise phase 111.7 -> 16.9 s, and 47,752 against 47,762 peptides at 1% with the decoy fraction 1.00% either way (docs/28 section 11). An earlier 8,858,206-PSM experiment-wide rescore went from 671.6 min to 12 min, because there the 30.18 GB TSV crossed the worker's streaming threshold and every iteration re-read a 12.77 GB memmap. Features are f32 because the TSV was already lossy (`{:.6}`) and the worker casts to f32 regardless; the two paths therefore feed marginally different values into a chaotic self-training loop, which is where that 10-peptide difference comes from. nn_torch only: `mokapot_worker.py` calls `mokapot.read_pin()` and cannot read Parquet, so a mokapot run falls back to `Tsv` with a warning instead of failing. Choose it over `Raw` where the sidecar directory is short of room: a parquet file has no size floor, so the space check only warns for it, while a raw matrix is exactly 4 bytes a value and is refused below that. |
| `raw` | yes | The default since 2026-09-28: the features as one row-major little-endian f32 `.npy` matrix, beside a small parquet of the metadata columns and a `<name>.raw.json` description (feature names, per-feature min/max, the parquet handoff's row-group size) that the worker is given. The engine streams each decoded batch straight into the file with no transpose and no parquet encode, and the worker copies the matrix into its own with no decode and no column-to-row transpose: on the 258.75M-row immunopeptidomics pool the parquet path spent an estimated 26 min of serial engine CPU encoding and the worker a strided fill of every column. The file is the raw size, 4 bytes a value, about 11% more than the snappy parquet there, so it pays where the codec, not the disk, is the limit (a RAM-backed `MUMDIA_SIDECAR_DIR`, an SSD). Features that compress well make the file gap much larger, and docs/13 records a small table where the raw write was the slower one; on the benchmark pools of the 2026-09-27 campaign the two walls were within a minute of each other, with byte-identical scored tables. Scores are byte-identical to `Parquet`: the worker fills the same matrix and sums the float64 moments over the same partition (the description carries the row-group size), and drops the same constant columns. Validate a new host by rescoring one pool with each handoff, same seed and threads, and comparing `psms_scored.parquet` byte for byte (`tests/python/test_nn_rescore_worker.py` does it on a fixture). nn_torch only; a mokapot run falls back to `Tsv` with a warning. The sidecar space check refuses the run when the work directory cannot hold the matrix (exactly 4 bytes a value, docs/13); set `Parquet` where the room is short, or point `MUMDIA_SIDECAR_DIR` elsewhere. |

### `ImFeatures`

(rust/mumdia/crates/mumdia-core/src/config.rs)

`features.im_features`: the ion-mobility feature family.

| Value | Default | Description |
|---|---|---|
| `off` | yes | No ion-mobility features. The feature schema is the 3D one. |
| `on` |  | The ion-mobility family is part of the `extended` set. On a run without 1/K0 its columns are 0. |
| `auto` |  | `on` when the spectra carry 1/K0 (`mumdia::instrument`), `off` otherwise. |

### `ImGate`

(rust/mumdia/crates/mumdia-core/src/config.rs)

`extract.im_gate`: whether fragment peaks are matched only inside a candidate's 1/K0 window.

| Value | Default | Description |
|---|---|---|
| `off` | yes | No 1/K0 gate: every fragment peak inside the RT window is matched, as on 3D data. |
| `fragments` |  | A fragment peak outside the candidate's 1/K0 window is not matched. The MS1 lookups are not gated. A run without per-peak 1/K0, or whose `run_windows` has no 1/K0 window, is extracted without the gate and says so in the log. |
| `auto` |  | `fragments` when the run's spectra carry 1/K0 (`mumdia::instrument`) and its `run_windows` carries 1/K0 windows, `off` otherwise. |

### `ImPredictorKind`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Where a FASTA-built library's `predicted_im` (1/K0) comes from; see `PredictFragConfig::im_predictor`.

| Value | Default | Description |
|---|---|---|
| `none` | yes | No ion-mobility prediction. The library has no `predicted_im` column and is the version-1 table. |
| `im2deep` |  | IM2Deep Python sidecar (`im2deep_worker.py`, docs/13_sidecars.md): one CCS per (peptidoform, charge) from the single-conformer model, converted to 1/K0 at the precursor m/z. A hard requirement: the run fails without an IM2Deep interpreter. |
| `auto` |  | `im2deep` when the run's spectra carry 1/K0 (a diaPASEF `.d`, `mumdia::instrument`) AND an IM2Deep interpreter resolves (`predict_frag.im2deep_python`, its environment variable or discovery), otherwise `none`, with a log line. |

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

Match-between-runs strategy (docs/12_quant_lfq_align_mbr_report_audit.md, "Second pass"). Read by `run-experiment` only; a single-run `run` has no other run to match. `SecondPass` (the default since 2026-10-13) searches every run a second time on an empirical library: the precursors the pooled first pass accepted experiment-wide (`precursor_q <= mbr.q_anchor`), each with its paired library decoy, both given the precursor's cross-run consensus retention time (`stages::second_pass`). The second pass is an ordinary search of that library, with its own seed, retention-time calibration, extraction and one pooled target-decoy rescore, and it is what the experiment reports. Measured against the first pass on four pools (docs/12): \| pool \| per-run IDs at 1% \| null \| \|---\|---\|---\| \| ProteoBench Astral entrapment, 3 runs x 3 NN seeds \| +8.0% \| entrapment FDP 0.95% -> 0.70% \| \| 22 HeLa runs, human + yeast + E. coli library \| +16.9% \| species FDP 1.42% -> 0.80% \| \| 216 single cells + 24 blank wells \| +54% to +57% \| decoys 0.97% -> 0.98%, 0 IDs in blanks \| \| 22-run HYE mixture \| proteins 4,366 -> 5,298 \| E. coli log2 A/B -1.71 -> -1.48 \| The last row is the known gap: quantities of the precursors the second pass adds in runs where they are weak compress the ratios (CV 0.255 -> 0.245, median \|error\| 0.238 -> 0.252). That is a quantification limitation, not a false identification. `EmpiricalLibrary`, `RtTransfer` and `Full` are the earlier rescuable-tier transfer (`scripts/mbr_worker.py`), which needs `mbr.python`. No code distinguishes the three: each accepts an already extracted, sub-threshold candidate on retention-time agreement alone. On the same pools its transfers measured 2.3-2.7% entrapment FDP (ProteoBench) and about 4x the false-ID rate of the direct identifications (single cells), so it stays opt-in and `validate()` warns when it is selected.

| Value | Default | Description |
|---|---|---|
| `none` |  | No match-between-runs: the first pass is the result. |
| `second_pass` | yes | Search every run again on the empirical library of the first pass (default). |
| `empirical_library` |  | Rescuable-tier transfer; behaves as `RtTransfer`. |
| `rt_transfer` |  | Rescuable-tier transfer on cross-run expected retention time (`mbr_worker.py`). |
| `full` |  | Rescuable-tier transfer; behaves as `RtTransfer`. |

### `MultiheadAnchors`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Anchors of the multi-head calibration; see `RtImTrainConfig::multihead_anchors`.

| Value | Default | Description |
|---|---|---|
| `auto` | yes | `search` when the adapting run's seed search is short of anchors (`RtImTrainConfig::multihead_search_below`) or its spectra carry 1/K0, else `seed`. |
| `seed` |  | The run's confident seed PSMs (`spectrum_q <= q_train`). |
| `search` |  | The confident targets of a bootstrap search of the run on the seed-anchored library (`precursor_q <= q_train`, observed RT = `apex_rt`). |

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

### `PrescreenLocalization`

(rust/mumdia/crates/mumdia-core/src/config.rs)

How modification siblings share evidence (`prescreen.localization`).

| Value | Default | Description |
|---|---|---|
| `own` | yes | Every form keeps its own score. |
| `family_support` |  | Family support: every form of one backbone with the same modification composition, charge and label takes the family's best score, so compatible localizations and ties pass or fail together. Support is kept separate from localization; no site is chosen. |
| `best_site` |  | Keep only forms whose own score equals the family best (ties kept). Measured to lose modified references in the prototype; available for comparison only. |

### `PrescreenPooling`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Fragment-trace pooling mode (`prescreen.trace.pooling`).

| Value | Default | Description |
|---|---|---|
| `merged` | yes | Bounded mass clusters over the pooled scans (max - min <= `cluster_da`), kept when seen in at least `min_detections` scans. |
| `unmerged` |  | Every pooled peak as its own feature, for comparison with merging. |

### `PrescreenPreset`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Calibration presets of the fragment-rarity prescreen (`prescreen.preset`). Each names a calibration rejection target, the share of the calibration half's scores at or below the cutoff. Measured on six Astral HYE runs (exploratory screening on sampled candidates, tagbench PHASE10): reduction / reference retention / weak-reference retention 50.29% / 99.70% / 99.24% (sensitive), 52.28% / 99.66% / 99.20% (balanced), 74.62% / 96.94% / 94.22% (stringent), 90.30% / 90.58% / 82.67% (aggressive). A target does not guarantee the same reduction on another run or library; the report records the measured one.

| Value | Default | Description |
|---|---|---|
| `sensitive` |  | Target 0.52. |
| `balanced` | yes | Target 0.54. |
| `stringent` |  | Target 0.75. |
| `aggressive` |  | Target 0.90. |
| `custom` |  | Use `prescreen.target`. |

### `PrescreenRetrieval`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Which candidates the prescreen scores (`prescreen.retrieval`).

| Value | Default | Description |
|---|---|---|
| `all` | yes | Score every candidate (the measured recommended mode). |
| `tags` |  | Retrieve candidates through database-free tags first: a candidate is retrieved when one of its trimers, in its own residue states, was observed in an eligible spectrum. Candidates without such evidence take the rescue route when `rescue` is on. |

### `PrescreenScope`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Which candidates the prescreen filters (`prescreen.scope`). Candidates outside the scope pass through unscored, and the cutoff is calibrated within the scope only.

| Value | Default | Description |
|---|---|---|
| `all` | yes | Every candidate form. |
| `modified` |  | Only forms carrying the modifications in `prescreen.scope_mods` (`scope_match` says whether any or all of them). |

### `PrescreenScopeMatch`

(rust/mumdia/crates/mumdia-core/src/config.rs)

How several `prescreen.scope_mods` combine.

| Value | Default | Description |
|---|---|---|
| `any` | yes | A form is in scope when it carries at least one of the listed modifications. |
| `all` |  | A form is in scope only when it carries every listed modification. |

### `PrescreenTraceScore`

(rust/mumdia/crates/mumdia-core/src/config.rs)

Which trace evidence the trace component scores (`prescreen.trace.score`).

| Value | Default | Description |
|---|---|---|
| `tag_coherence` | yes | Tag coherence: for each candidate trimer, the best (minimum pairwise trace cosine of the four path features x whole-path fit) in a pool; summed over distinct trimers. |
| `coherent_fragments` |  | Disconnected ladders: candidate fragments matched to pooled features, counted when their trace is compatible (cosine >= `min_quality`) with the most intense matched feature. |

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
| `HOME` | engine | computed: set by the operating system; on Unix and macOS the per-user cache directory under it holds the engine's caches when MUMDIA_CACHE_DIR is unset (`cache.rs` `root`) | `rust/mumdia/crates/mumdia/src/cache.rs::root` |
| `LOCALAPPDATA` | engine | computed: set by Windows; `%LOCALAPPDATA%\mumdia\cache` holds the engine's caches when MUMDIA_CACHE_DIR is unset (`cache.rs` `root`) | `rust/mumdia/crates/mumdia/src/cache.rs::root` |
| `MUMDIA_BREW_ITERS` | sidecar | `"20"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_CACHE_DIR` | engine | computed: the per-user cache directory: `$XDG_CACHE_HOME/mumdia` or `~/.cache/mumdia` (Linux and other Unix), `~/Library/Caches/mumdia` (macOS), `%LOCALAPPDATA%\mumdia\cache` (Windows). A path moves the `"auto"` caches there; `off` / `0` / `false` / `none` turns them off (`cache.rs` `root`) | `rust/mumdia/crates/mumdia/src/cache.rs::root` |
| `MUMDIA_CACHE_MAX_GB` | engine | computed: 100 (GiB): the library and DeepLC projection caches together, least recently used entries removed first, none used within the last hour. `0` or `unlimited` is no bound (`cache.rs` `budget`) | `rust/mumdia/crates/mumdia/src/cache.rs::budget` |
| `MUMDIA_CACHE_MIN_FREE_GB` | engine | computed: 10 (GiB): the free space a library-cache store must leave on the cache's disk, measured before the copy; `0` is no margin (`library_cache.rs` `min_free_after_store`) | `rust/mumdia/crates/mumdia/src/library_cache.rs::min_free_after_store` |
| `MUMDIA_CHROM_ROW_GROUP_ROWS` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/chromatograms.rs::row_group_rows` |
| `MUMDIA_CONVERT_THREADS` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/stages/convert.rs::convert_threads` |
| `MUMDIA_DEEPLC_RAW_OUTPUT` | sidecar | `""` | `scripts/deeplc_finetune.py::quiet_deeplc_progress`, `scripts/deeplc_worker.py::quiet_deeplc_progress` |
| `MUMDIA_DEEPLC_THREAD_CAP` | both | `"auto"` | `rust/mumdia/crates/mumdia/src/sched.rs::auto_overlap_threads`, `scripts/deeplc_finetune.py::main`, `scripts/deeplc_worker.py::main` |
| `MUMDIA_ENTRAPMENT_MODEL` | sidecar | `"gbm"` | `scripts/entrapment_worker.py::_new_model` |
| `MUMDIA_IM2DEEP_THREAD_CAP` | sidecar | `"auto"` | `scripts/im2deep_worker.py::main` |
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
| `MUMDIA_NN_PARALLEL` | sidecar | `"auto"` | `scripts/nn_rescore_worker.py::main` |
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
| `MUMDIA_PYTHON_IM2DEEP` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `MUMDIA_PYTHON_MBR` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `MUMDIA_PYTHON_MS2PIP` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `MUMDIA_PYTHON_PEPTDEEP` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `MUMDIA_PYTHON_RESCORE` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `MUMDIA_QUANT_SELECTIVE_READ` | engine | computed: on (quant skips, unread, the chromatogram data pages that hold no kept row). `0` reads every page of every row group it opens; the outputs are the same (`stages/quant.rs` `selective_read_enabled`) | `rust/mumdia/crates/mumdia/src/stages/quant.rs::selective_read_enabled` |
| `MUMDIA_REPICK_DUMP` | engine | computed: off. Diagnostic: a path makes `retrace.repick` write every peak it scored there (candidate, order, apex RT, score); the pick is the same either way (`stages/retrace.rs` `repick`) | `rust/mumdia/crates/mumdia/src/stages/retrace.rs::run_hashed` |
| `MUMDIA_RESCORE_MODEL` | both | `"nn"` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::run_hashed`, `scripts/mokapot_worker.py::main`, `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_RTW_FEATURES` | engine | computed: on when unset; `0` leaves the window-scaled RT error family out (`features.rs` `env_families`) | `rust/mumdia/crates/mumdia/src/stages/features.rs::env_families` |
| `MUMDIA_SCRIPTS` | sidecar | `os.path.dirname(os.path.abspath(__file__` | `scripts/mh_shard_predict.py::<module>` |
| `MUMDIA_SIDECAR_DIR` | engine | computed: `<out-dir>/sidecar_work` under `run` and `run-experiment`; `sidecar_work` in the current directory, or `--work-dir`, for `mumdia rescore`. A path moves the rescore sidecar files there (`stages/rescore.rs` `sidecar_work_dir`) | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::sidecar_work_dir` |
| `MUMDIA_SIDECAR_SPACE_CHECK` | engine | computed: on (before the handoff is written, a rescore sidecar run is refused when its work directory has less room than a PIN or raw handoff cannot be smaller than, and warned about below the files' usual size). `0` / `off` / `false` / `no` skips the check (`stages/rescore.rs` `check_sidecar_space`) | `rust/mumdia/crates/mumdia/src/stages/rescore.rs::check_sidecar_space` |
| `MUMDIA_THERMO_PARSER` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/raw.rs::locate_parser` |
| `MUMDIA_WIDE_SCAN` | engine | computed: `plain`: the plain reader with its parallel decode for rescore's feature stream and compete's pass-through copy. `rowgroup` decodes the feature stream one row group a batch; `coalesced` reads each row group's projected column chunks in one sequential read (`stages/mod.rs` `WideScan`) | `rust/mumdia/crates/mumdia/src/stages/mod.rs::WideScan::from_env` |
| `MUMDIA_WINDOW_FEATURES` | engine | computed: on when unset; `0` leaves the fixed-window feature family out (`features.rs` `env_families`) | `rust/mumdia/crates/mumdia/src/stages/features.rs::env_families` |
| `MUMDIA_XGB_DEPTH` | sidecar | `"6"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_XGB_JOBS` | sidecar | `"0"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_XGB_LR` | sidecar | `"0.1"` | `scripts/mokapot_worker.py::make_model` |
| `MUMDIA_XGB_TREES` | sidecar | `"200"` | `scripts/mokapot_worker.py::make_model` |
| `OMP_NUM_THREADS` | both | `16` | `rust/mumdia/crates/mumdia/src/main.rs::apply_threads`, `scripts/nn_rescore_worker.py::main` |
| `PATH` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/raw.rs::locate` |
| `ProgramFiles` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/raw.rs::locate_msconvert` |
| `ProgramFiles(x86)` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/raw.rs::locate_msconvert` |
| `VIRTUAL_ENV` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs::candidates` |
| `XDG_CACHE_HOME` | engine | computed: set by the desktop environment, often unset; an absolute value puts the engine's caches in `$XDG_CACHE_HOME/mumdia` on Linux when MUMDIA_CACHE_DIR is unset (`cache.rs` `root`) | `rust/mumdia/crates/mumdia/src/cache.rs::root` |

101 variables are read: 39 engine-side, 66 sidecar-side, 4 on both sides.

### Variables the code sets

A variable set here overrides whatever the caller exported, so exporting one
of these has no effect on the process listed. The engine's `--threads` is the
one exception noted in its own help text: it sets `MUMDIA_NN_THREADS` and
`OMP_NUM_THREADS` for the sidecars only if they are not already set.

| Variable | Set by | Value | Site |
|---|---|---|---|
| `KMP_DUPLICATE_LIB_OK` | sidecar | `"TRUE"` | `scripts/deeplc_finetune.py::<module>` |
| `MKL_NUM_THREADS` | sidecar | `"1"` | `scripts/deeplc_finetune.py::<module>` |
| `MUMDIA_FREE_SPACE_PATH` | engine | `&probe` | `rust/mumdia/crates/mumdia/src/cache.rs::free_space` |
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

8 environment read(s) whose name is not a literal. Reads with the same function, access and argument share one entry, which gives their number when there is more than one:

- `rust/mumdia/crates/mumdia/src/stages/extract.rs::PsmRows::push: env read via closure of `&mut self``
- `rust/mumdia/crates/mumdia/src/stages/extract.rs::PsmStream::push: env read via closure of `&mut self``
- `rust/mumdia/crates/mumdia/src/stages/extract.rs::flush_below: env read via closure of `chunk.slices_mut()``
- `rust/mumdia/crates/mumdia/src/stages/extract.rs::flush_below: env read via closure of `runs[r].span_mut(m)``
- `rust/mumdia/crates/mumdia/src/stages/prescreen/tags.rs::discover: env read via closure of `alpha.key(alpha.canonical(a, b, c), z, false)``
- `rust/mumdia/crates/mumdia/src/stages/prescreen/tags.rs::discover: env read via closure of `alpha.key(alpha.canonical(la as u16, p.1, p.2), z, true)``
- `rust/mumdia/crates/mumdia/src/stages/prescreen/tags.rs::discover: env read via closure of `alpha.key(alpha.canonical(p.1, p.2, lb as u16), z, true)``
- `rust/mumdia/crates/mumdia/src/stages/rescore.rs::check_sidecar_space: env read of `k``

## Coverage

24 structs and 312 fields emitted from `rust/mumdia/crates/mumdia-core/src/config.rs`, plus 39 enumerations, 1 named profile(s), 101 environment variables read and 20 set.

33 field(s) carry a gating marker in their doc comment. 59 field(s) carry no doc comment at all, so their description is empty above. 0 default(s) could not be resolved and 2 have none by design.
