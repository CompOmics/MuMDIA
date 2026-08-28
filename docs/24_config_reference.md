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
| [(top level)](#top-level) | `Config` | 14 | [docs/02_config_and_data_model.md](02_config_and_data_model.md) |
| [`prescan`](#prescan) | `PrescanConfig` | 6 | [docs/21_prescan.md](21_prescan.md) |
| [`digest`](#digest) | `DigestConfig` | 6 | [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md) |
| [`digest.decoy`](#digestdecoy) | `DecoyConfig` | 1 | [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md) |
| [`peptidoforms`](#peptidoforms) | `PeptidoformsConfig` | 7 | [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md) |
| [`predict_frag`](#predict_frag) | `PredictFragConfig` | 9 | [docs/06_predict_frag_index_matchers.md](06_predict_frag_index_matchers.md) |
| [`search_seed`](#search_seed) | `SearchSeedConfig` | 8 | [docs/07_search_seed.md](07_search_seed.md) |
| [`rt_im_train`](#rt_im_train) | `RtImTrainConfig` | 15 | [docs/08_rt_im_train.md](08_rt_im_train.md) |
| [`extract`](#extract) | `ExtractConfig` | 35 | [docs/09_extract.md](09_extract.md) |
| [`extract.claim_cues`](#extractclaim_cues) | `ClaimCues` | 7 | [docs/09_extract.md](09_extract.md) |
| [`features`](#features) | `FeaturesConfig` | 10 | [docs/10_features.md](10_features.md) |
| [`compete`](#compete) | `CompeteConfig` | 6 | [docs/11_compete_rescore_fdr.md](11_compete_rescore_fdr.md) |
| [`rescore`](#rescore) | `RescoreConfig` | 13 | [docs/11_compete_rescore_fdr.md](11_compete_rescore_fdr.md) |
| [`quant`](#quant) | `QuantConfig` | 17 | [docs/12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md) |
| [`mbr`](#mbr) | `MbrConfig` | 9 | [docs/12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md) |
| [`experiment`](#experiment) | `ExperimentConfig` | 2 | [docs/01_overview_and_dataflow.md](01_overview_and_dataflow.md) |
| [`peptidoforms.fixed_mods[] / peptidoforms.variable_mods[]`](#peptidoformsfixed_mods--peptidoformsvariable_mods) | `ResidueMod` | 2 | [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md) |

## (top level)

`Config` (rust/mumdia/crates/mumdia-core/src/config.rs:1457). stage document: [docs/02_config_and_data_model.md](02_config_and_data_model.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
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

## prescan

`PrescanConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:281). stage document: [docs/21_prescan.md](21_prescan.md).

Sequence-tag prescan (`mumdia prescan`). Prunes modification-bearing candidates that have no anchored tag support in a given run, before the per-run library is assembled. The screen is deliberately blind to target/decoy label: tags are emitted in both orientations and a reverse decoy preserves composition and precursor m/z, so a decoy survives exactly when its target does. That keeps exchangeability, and therefore downstream FDR, intact.

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `tol_da` | `f64` | `0.005` |  | Peak-delta match tolerance in Da. Permissive on purpose: a false tag only fails to prune, while a missed tag discards a real candidate with no way to recover it downstream. |
| `rt_slack_s` | `f64` | `150.0` |  | Widen each candidate's RT window by this many seconds before binning. The window comes from a calibration fitted on a different run, and `cal.json` residuals are in-sample and roughly 3x optimistic, so size this from out-of-sample RT error, not from the reported fit. |
| `rt_bin_s` | `f64` | `25.0` |  | RT bin width for the observed-tag index. |
| `top_peaks` | `usize` | `150` |  | Most intense peaks per MS2 used to build tags (0 = all). This bounds the O(peaks^2) delta search and is NOT destructive: it only affects tag construction, never the spectra artifact that extraction later reads. |
| `mods` | `Vec<String>` | `["C:Carbamidomethyl", "M:Oxidation"]` |  | Residue:UniModName entries that may appear in a screened peptidoform, e.g. `C:Carbamidomethyl`. A peptidoform carrying anything outside this set plus `anchor_mods` is dropped rather than screened on a partially understood sequence. |
| `anchor_mods` | `Vec<String>` | `[]` |  | Residue:UniModName entries the screen anchors ON. Only trimers covering one of these positions count as evidence, so backbone signal cannot keep a modified hypothesis alive. |

## digest

`DigestConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:318). stage document: [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `enzyme` | `Enzyme` | `trypsin_p` |  |  |
| `missed_cleavages` | `u32` | `2` |  |  |
| `min_len` | `usize` | `5` |  |  |
| `max_len` | `usize` | `50` |  |  |
| `decoy` | `DecoyConfig` | the `DecoyConfig` section's own defaults |  |  |
| `n_term_met_excision` | `bool` | `true` |  | N-terminal methionine excision: when a protein begins with `M`, also emit the initiator-Met-removed form of its N-terminal peptides. The initiator methionine is cleaved in vivo for most proteins, so search engines (including DIA-NN via `--met-excision`) enumerate both forms. Omitting it makes the search database structurally miss those excised peptides. |

## digest.decoy

`DecoyConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:264). stage document: [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `strategy` | `DecoyStrategy` | `reverse` |  |  |

## peptidoforms

`PeptidoformsConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:346). stage document: [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md).

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

`PredictFragConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:405). stage document: [docs/06_predict_frag_index_matchers.md](06_predict_frag_index_matchers.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `predictor` | `FragPredictorKind` | `native` |  |  |
| `rt_predictor` | `RtPredictorKind` | `native` |  |  |
| `charge2_from_precursor_charge` | `i32` | `2` |  | Fragment charges rule: charge 1 always; charge 2 added for precursor charge >= this threshold (docs/18_findings_and_decisions.md). Default 2: DIA-NN uses doubly-charged fragments for ~16% of charge-2 precursors' transitions, so blocking them (the old default of 3) discarded real signal. |
| `charge_by_basic_residues` | `bool` | `false` | benchmark-gated | Composition-based fragment charge cap. When true, a b/y fragment is kept at charge z only if `z <= 1 (its N-terminal amine) + (#R + #H + #K within that fragment)`, and never above the precursor charge. This supersedes the `charge2_from_precursor_charge` rule when set. Default false. Pairs with `peptidoforms.charge_by_basic_residues` for precursors; benchmark-gated because it changes the scored transition set. |
| `top_n_fragments` | `usize` | `6` |  |  |
| `ms2pip_model` | `String` | `"HCD"` |  |  |
| `ms2pip_python` | `Option<String>` | `null` |  | Python executable for the MS2PIP sidecar (env with ms2pip + pyarrow). |
| `deeplc_python` | `Option<String>` | `null` |  | Python executable for the DeepLC sidecar (env with deeplc + pyarrow). |
| `sidecar_script_dir` | `String` | `"scripts"` |  | Directory holding the sidecar worker scripts. |

## search_seed

`SearchSeedConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:448). stage document: [docs/07_search_seed.md](07_search_seed.md).

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

`RtImTrainConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:495). stage document: [docs/08_rt_im_train.md](08_rt_im_train.md).

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
| `finetune_epochs` | `usize` | `25` |  | DeepLC fine-tune training epochs (passed to `deeplc_finetune.py --epochs`). Early stopping with `finetune_patience` usually halts before this cap, so it is an upper bound rather than a fixed count. Only used when `finetune_deeplc`. |
| `finetune_patience` | `usize` | `10` |  | DeepLC fine-tune early-stopping patience (`--patience`): epochs without validation-loss improvement before stopping. Only used when `finetune_deeplc`. |
| `finetune_batch` | `usize` | `0` |  | DeepLC fine-tune batch size (`--batch`). 0 (default) auto-scales to the confident seed size so each epoch has >= ~30 gradient steps; a fixed large batch underfits small seeds (a ~4k-peptide reference at batch 512 is ~8 steps/epoch and never converges). Only used when `finetune_deeplc`. |
| `adaptive_rt_window` | `bool` | `false` |  | Adaptive RT window (sensitivity_plan spec 03 §3.5, backlog P3.2/P3.3): instead of one global residual-percentile half-width for every candidate, bin the calibration anchors by calibrated RT and give each candidate the LOCAL residual percentile of its RT region, clamped to `[rt_window_min_s, fallback_rt_window_s]` and scaled by `rt_window_multiplier`. A fixed window is simultaneously too wide for well-calibrated regions and too narrow for poorly-calibrated ones; this tightens clean regions (less interference) and widens noisy ones (more recall). Empty/sparse bins fall back to the global width. Default false. |
| `adaptive_rt_bins` | `usize` | `12` |  | Number of equal-width calibrated-RT bins for the adaptive window. |
| `rt_window_min_s` | `f64` | `1.0` |  | Lower clamp (seconds) for any RT half-window (the existing 1 s floor). |
| `window_holdout_frac` | `f64` | `0.0` | benchmark-gated, do not default | Size `w_rt` from HELD-OUT residuals instead of in-sample ones. A fraction of anchor peptides (`base_peptide_id % 1000 < round(frac*1000)`, so the split is deterministic and shared with `deeplc_finetune.py`) is excluded from the sizing fit and, when `finetune_deeplc` runs, from the fine-tune reference; `w_rt` is then the residual percentile of those held-out anchors against the fit they never entered. The final calibration curve still uses every anchor. In-sample sizing underestimates the tail and rewards a memorizing RT model with a window it does not deserve (measured: it inverted the 4.0.0a2/4.1.0 ranking); held-out sizing measured +0.9% peptides with DeepLC 4.1.0 and -1.5% with 4.0.0a2 on the AIF benchmark, both at 0.98% decoy, so enable it only with a generalizing RT model. 0.0 (default) keeps in-sample sizing. Mutually exclusive with `adaptive_rt_window`. Benchmark-gated; do not default on. |

## extract

`ExtractConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:576). stage document: [docs/09_extract.md](09_extract.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
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

`ClaimCues` (rust/mumdia/crates/mumdia-core/src/config.rs:193). stage document: [docs/09_extract.md](09_extract.md).

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

`FeaturesConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:864). stage document: [docs/10_features.md](10_features.md).

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

## compete

`CompeteConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:934). stage document: [docs/11_compete_rescore_fdr.md](11_compete_rescore_fdr.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `group_by` | `CompeteGroupBy` | `base_peptide` |  | Competition grouping: `precursor` collapses charge/modification siblings separately within each target/decoy label; targets and decoys therefore do not compete directly. `apex` also groups by rounded apex RT; `peptidoform_charge` keeps each peptidoform+charge as its own group (precursor-level, as DIA-NN/Spectronaut report), so sibling charges of one peptide are not collapsed. |
| `apex_rt_tolerance_s` | `f64` | `5.0` |  |  |
| `mode` | `CompetitionMode` | `winner_take_all` |  | How within-group competition resolves (sensitivity program, spec 04 §6 / P2.4). `winner_take_all` = legacy (keep only the top `prelim_score` per group). The other modes preserve more candidate evidence for the rescorer/ FDR to arbitrate. Default `winner_take_all` (unchanged behaviour). |
| `margin` | `f64` | `0.0` | gated | Score margin (in `prelim_score` units) required to remove a loser under `margin_gated`. A loser closer than this to the winner is kept. |
| `unique_evidence_min_fragments` | `usize` | `2` |  | Minimum distinct unique-fragment count a loser must have to survive under `unique_evidence` (needs the `unique_fragment_count` feature; falls back to winner-take-all when the column is absent). |
| `emit_competition_audit` | `bool` | `false` | diagnostic | Diagnostic: when true, write `<out>.compete_audit.parquet` recording every removed candidate with its group, winner, scores, and removal reason. |

## rescore

`RescoreConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:1292). stage document: [docs/11_compete_rescore_fdr.md](11_compete_rescore_fdr.md).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `classifier` | `RescorerKind` | `native_tda` |  |  |
| `folds` | `usize` | `3` |  |  |
| `train_fdr` | `f64` | `0.01` |  |  |
| `num_iter` | `usize` | `10` |  | number of semi-supervised iterations for the native rescorer. |
| `max_feature_matrix_gib` | `f64` | `0.0` |  | Refuse a rescore whose in-memory feature matrix would exceed this many GiB. 0 (default) means no ceiling, which is the previous behaviour. The matrix is `Vec<Vec<f64>>`: eight bytes per value, plus a heap allocation and a 24-byte spine entry per PSM. Nothing in the workspace estimates or checks available memory (there is deliberately no `sysinfo` dependency), so an experiment-wide rescore over enough runs was simply killed by the OS after however long it took to get there. `native_tda` additionally runs all folds in parallel, each holding an owned standardised copy of its training slice, so the true peak is roughly `(1 + folds) x` this figure. Setting a ceiling converts that into an error at startup, naming the estimate and the two ways out. It is not a batching implementation: sub-batching changes which PSMs share a pooled `q_value`, so it is the operator's decision, not a silent one. |
| `python` | `Option<String>` | `null` |  |  |
| `percolator_bin` | `Option<String>` | `null` |  | Path to an external `percolator` executable. Parsed and never read: no stage launches percolator, and `RescorerKind` has no variant that would. It is the only silently inert config field in the tree, since the three MBR ones warn (see `validate`). Kept rather than deleted because the external-percolator path is still intended; `validate` now warns when it is set. |
| `entrapment_marker` | `Option<String>` | `null` |  | Protein-accession substring marking spike-in (entrapment) negatives, e.g. "_HUMAN". Required when `classifier = entrapment`; PSMs whose protein contains it are the empirical false population. |
| `entrapment_exclude` | `Option<String>` | `null` |  | If a protein also contains this substring it is NOT counted as entrapment (the sample's own species, e.g. "_ECOLI"): shared peptides then count as real targets. `None` = the marker alone decides. |
| `entrapment_contaminant_markers` | `Vec<String>` | `[]` |  | Protein substrings marking genuine contaminants inside the spike-in proteome (e.g. "KRT", "ALBU", keratin/albumin entry-name tokens). A PSM matching `entrapment_marker` but also one of these is treated as a REAL target, not an entrapment negative: such peptides are truly present (handling contaminants) so using them as negatives mislabels real signal and inflates the estimated FDR. Empty = every spike-in hit is a negative. |
| `entrapment_ratio` | `f64` | `1.0` |  | N_real_lib / N_entrap_lib. Scales the entrapment FDR estimate so it is unbiased when the spike-in library differs in size from the real one. |
| `strict` | `bool` | `true` |  | When true, any sidecar/classifier failure or misconfiguration (Mokapot or entrapment sidecar error, unwired percolator, entrapment mode with no entrapment PSMs) is a hard error instead of a silent fall back to the native rescorer. Default true so a named scientific workflow cannot silently execute a different model; set false only for explicit legacy compatibility. |
| `handoff` | `Handoff` | `tsv` |  | How the feature matrix reaches a sidecar rescorer. See `Handoff`. `parquet` is dramatically faster on large pools but applies to nn_torch only. |

## quant

`QuantConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:1106). stage document: [docs/12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md).

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

`MbrConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:1245). stage document: [docs/12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md).

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

`ExperimentConfig` (rust/mumdia/crates/mumdia-core/src/config.rs:1430). stage document: [docs/01_overview_and_dataflow.md](01_overview_and_dataflow.md).

Options for the experiment-wide orchestrator (`mumdia run-experiment`).

| Field | Type | Default | Gated | Description |
|---|---|---|---|---|
| `parallel_runs` | `usize` | `1` |  | How many per-run search chains to execute concurrently. 1 (default) is strictly sequential, i.e. the historical behaviour. Runs are independent, so raising this scales nearly linearly in wall time, but EACH concurrent run holds its own extraction working set (tens of GB on a large library), so the practical ceiling is memory, not cores. Raise it deliberately after checking peak RSS for a single run; 2-4 is a reasonable start on a large-memory machine. Results are unaffected: chunks are processed in index order and completion order never reaches the output. |
| `finetune_scope` | `FinetuneScope` | `first_run_only` |  | Whether the DeepLC fine-tune runs once for the experiment or once per run. Only consulted when `rt_im_train.finetune_deeplc` is set. |

## peptidoforms.fixed_mods[] / peptidoforms.variable_mods[]

`ResidueMod` (rust/mumdia/crates/mumdia-core/src/config.rs:388). stage document: [docs/05_digest_peptidoforms.md](05_digest_peptidoforms.md).

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

(rust/mumdia/crates/mumdia-core/src/config.rs:57)

| Value | Default | Description |
|---|---|---|
| `loess` | yes |  |
| `linear` |  |  |
| `none` |  |  |

### `CompeteGroupBy`

(rust/mumdia/crates/mumdia-core/src/config.rs:1001)

| Value | Default | Description |
|---|---|---|
| `base_peptide` |  | One winner per stripped base peptide, per label. Renamed from `precursor`, which it is not: `compete.rs` keys the group on `base_peptide_id`, which comes from the stripped sequence, so every charge state AND every modification variant of one peptide collapses to a single winner before FDR. Use `peptidoform_charge` for a genuine precursor unit, and note that it is REQUIRED for a PTM search. The old name is not accepted, so an old config fails loudly rather than silently changing the competition unit. |
| `apex` |  |  |
| `peptidoform_charge` |  | Precursor-level: separate every distinct peptidoform+charge. Recovers sibling charges the peptide-level `Precursor` grouping collapses; the label stays in the key so a target never competes against its own decoy. |

### `CompetitionMode`

(rust/mumdia/crates/mumdia-core/src/config.rs:980)

Within-group competition resolution (spec 04 §6). Only `WinnerTakeAll` removes candidates unconditionally; the others preserve candidates the rescorer can still discriminate, which is the sensitivity program's central principle ("preserve candidate evidence until the workflow can make a calibrated decision"). Target/decoy labels remain part of the competition key in every mode, so a target never competes against its own decoy (the null is preserved).

| Value | Default | Description |
|---|---|---|
| `winner_take_all` | yes | Legacy: keep only the highest `prelim_score` candidate per group. |
| `none` |  | Keep every candidate (no within-group removal); FDR handles ambiguity. |
| `features_only` |  | Keep every candidate; conflict/contested features (added upstream) carry the interference signal into rescoring. Same retained set as `None`; the name documents intent for the experiment matrix. |
| `unique_evidence` |  | Keep a loser when it has enough independent evidence (`unique_fragment_count >= unique_evidence_min_fragments`); otherwise remove it (winner-take-all fallback). |
| `margin_gated` |  | Remove a loser only when `winner_score - loser_score >= margin`; otherwise keep it. Conservative removal for the low-FDR region. |

### `DecoyStrategy`

(rust/mumdia/crates/mumdia-core/src/config.rs:17)

| Value | Default | Description |
|---|---|---|
| `reverse` | yes | Reverse the sequence keeping the C-terminal residue fixed. Documented, clean-room default for MVP (docs/14_build_test_deploy_gotchas.md). No borrowed map. |
| `scramble` |  | Deterministic seeded shuffle of the interior residues. |
| `diann_shift` |  | DIA-NN terminal-residue fragment m/z shift. Deferred: license-checked addition (docs/14_build_test_deploy_gotchas.md), not part of MVP. |
| `none` |  |  |

### `DecoyTransfer`

(rust/mumdia/crates/mumdia-core/src/config.rs:1236)

Decoy-transfer null for the MBR false-transfer FDR (M4). `ReverseSequence` transfers reverse/scramble decoys at the same expected RT; `PermutedRt` transfers real precursors to a decoupled (wrong) expected RT; `Both` combines them. The prototype's shuffled-RT null gave a ~0.6% in-window false rate vs 66.6% true (113x separation), so the transfer q-value is well-calibrated.

| Value | Default | Description |
|---|---|---|
| `permuted_rt` | yes |  |
| `reverse_sequence` |  |  |
| `both` |  |  |

### `Enzyme`

(rust/mumdia/crates/mumdia-core/src/config.rs:47)

| Value | Default | Description |
|---|---|---|
| `trypsin_p` | yes | Trypsin/P: cut after K or R (including before P). |
| `trypsin` |  | Classic trypsin: cut after K or R but not before P. |

### `FeatureSet`

(rust/mumdia/crates/mumdia-core/src/config.rs:66)

| Value | Default | Description |
|---|---|---|
| `minimal` | yes | MVP feature set (docs/10_features.md). |
| `rich` |  |  |
| `extended` |  | Minimal + Rich + the extended battery (DIA-NN / OpenSWATH / AlphaDIA / MS2Rescore / OktoberFest analogs + novel families) from the per-family modules in `stages/features/`. Superset, opt-in; the classifier picks the signal it can use (esp. under the nonlinear `Entrapment` rescorer). |

### `FinetuneScope`

(rust/mumdia/crates/mumdia-core/src/config.rs:1378)

How many DeepLC fine-tunes an experiment pays for.

| Value | Default | Description |
|---|---|---|
| `first_run_only` | yes | Fine-tune DeepLC once, on the FIRST run's confident seeds, and reuse that library for every run. Each run still fits its OWN retention-time calibration (LOESS by default) on top of it. MEASURED COST (6-run ProteoBench HYE AIF set, 2026-07-28). Reuse is NOT free: the run that owned the fine-tune reached a median \|RT residual\| of 15.2 s, while the five reusing runs reached 20.3, 20.5, 20.9, 24.9 and 25.4 s -- +7.2 s, +47% on average -- and their calibrated RT windows widened from 145 s to 179-227 s. The degradation is MONOTONIC in acquisition order, i.e. real chromatographic drift that a single fine-tune cannot track; per-run LOESS corrects the slope (0.96-0.99) but not the scatter. Wider windows also cost compute downstream: extract roughly doubled (126 s -> 203-242 s) and features up to tripled (116 s -> 215-388 s), which claws back part of the saving. It is still the default because the fine-tune dominates a large experiment: one 36.5 min fine-tune instead of N. On an 80-run batch that is ~48 h saved against ~6.5 h of extra extract/features. But on a long batch the drift keeps growing, so prefer `PerRun` when the extra hours are affordable, and treat periodic re-fine-tuning (not yet implemented) as the better answer for very large batches. |
| `per_run` |  | Fine-tune separately for every run. Adapts the model weights to each run's own chromatography instead of only calibrating a shared model, which measurably tightens retention time: see the numbers on `FirstRunOnly`. Costs one full DeepLC fine-tune per run (36.5 min on the HYE library: 5.7 min training plus 30.8 min predicting 4.9M peptidoforms). |

### `FragPredictorKind`

(rust/mumdia/crates/mumdia-core/src/config.rs:91)

| Value | Default | Description |
|---|---|---|
| `native` | yes | Native heuristic intensity model (no Python). MVP default. |
| `ms2pip` |  | MS2PIP Python sidecar (docs/13_sidecars.md). |

### `FragmentSelection`

(rust/mumdia/crates/mumdia-core/src/config.rs:1171)

Fragment ranking for the quant top-N sum. See `QuantConfig::fragment_selection`.

| Value | Default | Description |
|---|---|---|
| `observed_area` | yes | Rank fragments by their own integrated area (legacy). |
| `predicted` |  | Rank fragments by library intensity (`predicted_intensity` in the chromatogram table). |

### `GateMode`

(rust/mumdia/crates/mumdia-core/src/config.rs:837)

Spectral-agreement score the extraction acceptance gate (`gate_min_score`) thresholds. All are computed at the gate from data already in hand.

| Value | Default | Description |
|---|---|---|
| `apex_pearson` | yes | Legacy: Pearson of observed-vs-predicted fragment intensities at the single apex scan. One chimeric scan can dominate it. |
| `peak_spectral` |  | Pearson of the PEAK-INTEGRATED observed spectrum (each fragment summed over the elution-peak scans) vs predicted intensities. Averages out a single interfered scan; the standard library-dot-product measure. |
| `spectral_entropy` |  | Li spectral-entropy similarity of the sqrt-transformed apex-scan observed vs predicted intensities (`spectral_entropy_similarity_sqrt`). The full-feature gate search (all ~379 features, target-vs-decoy) found this the single best gate discriminator: AUC 0.826 / matched-pool recall 69.8%, versus apex Pearson's 0.781 / 64.5%. Same inputs as `ApexPearson`, better separation. |
| `coelution` |  | Predicted-intensity-weighted mean CO-ELUTION correlation of each matched fragment's XIC to the signature reference over the elution peak (temporal agreement, orthogonal to intensity agreement). |
| `combined` |  | Require BOTH: peak-integrated spectral Pearson >= `gate_min_score` AND the co-elution score >= `gate_coelution_min`. More specific (an interferent passing one axis is still rejected), for a cleaner FDR pool. |

### `Handoff`

(rust/mumdia/crates/mumdia-core/src/config.rs:1411)

How the feature matrix crosses the Rust -> Python boundary for a sidecar rescorer.

| Value | Default | Description |
|---|---|---|
| `tsv` | yes | Tab-separated PIN. Percolator's format, and what `mokapot.read_pin` requires. |
| `parquet` |  | Parquet feature table with f32 features. Measured on an 8,858,206-PSM experiment-wide rescore: the 30.18 GB TSV exceeded the worker's streaming threshold, so every iteration re-read a 12.77 GB memmap; Parquet kept the matrix in memory and the rescore went from 671.6 min to 12 min with the decoy fraction unchanged at 0.988%. Features are f32 because the TSV was already lossy (`{:.6}`) and the worker casts to f32 regardless. nn_torch only: `mokapot_worker.py` calls `mokapot.read_pin()` and cannot read Parquet, so a mokapot run falls back to `Tsv` with a warning instead of failing. |

### `MatcherKind`

(rust/mumdia/crates/mumdia-core/src/config.rs:39)

Fragment-matcher backend for search-seed and extract (docs/06_predict_frag_index_matchers.md). Default `Fragindex` (log-bin CSR matcher): on narrow-window DIA it is ~1.95x faster in search-seed and ~1.26x in extract with essentially unchanged IDs (HYE B_01: peptides -0.1%); `Bucketed` is the previous `Library::page_search` path (retained for A/B and for the AIF full-range-window case, where the predicate difference shifts IDs more).

| Value | Default | Description |
|---|---|---|
| `bucketed` |  |  |
| `fragindex` | yes |  |

### `MbrStrategy`

(rust/mumdia/crates/mumdia-core/src/config.rs:1217)

Match-between-runs strategy (Stage D3, docs/12_quant_lfq_align_mbr_report_audit.md). Default `None` reproduces the current chain byte-for-byte. ONLY `None` VS NOT-`None` IS IMPLEMENTED. The three non-`None` variants are described below as the intended staging, but no code distinguishes them: every test in the tree is `strategy != None`, so selecting `RtTransfer` or `Full` today behaves exactly like `EmpiricalLibrary`. They are kept as the recorded design ladder rather than deleted because the MBR tier is planned and benchmark-gated (CLAUDE.md); `validate()` warns when a non-`None` variant is selected so a config cannot quietly expect more than it gets. Intended staging: `EmpiricalLibrary` builds the consensus anchor library only; `RtTransfer` adds cross-run expected-RT transfer extraction; `Full` adds requantification. All require >= 2 runs and a decoy-transfer FDR (see the plan).

| Value | Default | Description |
|---|---|---|
| `none` | yes | No match-between-runs (default). |
| `empirical_library` |  | Build the cross-run consensus anchor library (M1) only; no transfer. |
| `rt_transfer` |  | EmpiricalLibrary + cross-run expected-RT transfer extraction (M2/M3). |
| `full` |  | RtTransfer + requantification of accepted transfers (M5). |

### `PeakClaim`

(rust/mumdia/crates/mumdia-core/src/config.rs:133)

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

(rust/mumdia/crates/mumdia-core/src/config.rs:1030)

How the elution-peak integration window is chosen per candidate in quant.

| Value | Default | Description |
|---|---|---|
| `per_candidate` | yes | Each candidate's window comes from its own summed-XIC descent walk. Exact per peak but sensitive to interference (stretched) and sparse peaks (collapsed). |
| `consensus` |  | Consensus window: the median left and right half-widths of confident peptides (a near-constant instrument/gradient property) applied around each candidate's apex. Robust to a single window being distorted. The widths are estimated per quant invocation, not shared automatically across runs. |

### `QuantQColumn`

(rust/mumdia/crates/mumdia-core/src/config.rs:1085)

Which q-value column quant filters candidates on. Peptide- or precursor-level q is appropriate for a single-run rescore. Under experiment-wide rescoring, those grouped q-values are pooled and carried only on the best PSM across all runs, so filtering per-run slices on them creates disjoint quant sets. `RunPsmQ` is the run-local FDR gate for that cross-run workflow; `PsmQ` keeps the pooled per-PSM gate available when that is explicitly intended.

| Value | Default | Description |
|---|---|---|
| `peptide_q` | yes | Filter on `peptide_q_value` (per-run peptide FDR). Default. |
| `precursor_q` |  | Filter on `precursor_q`. This is valid only for a single-run rescore: experiment-wide rescoring currently computes precursor q-values over the pooled experiment and assigns each precursor's grouped q-value to its best PSM, so it is not a per-run cross-run-quant gate. |
| `psm_q` |  | Filter on the per-PSM `q_value`. In an experiment-wide rescore this is a pooled-experiment PSM q-value, not a run-local FDR estimate. |
| `run_psm_q` |  | Filter on `run_psm_q` (per-run PSM FDR). The correct choice for cross-run quant off an experiment-wide rescore: each run's PSMs are FDR-controlled within their own run, so quant keeps the right per-run precursors without the external `split_scored.py` peptide-q overwrite. |

### `RescorerKind`

(rust/mumdia/crates/mumdia-core/src/config.rs:101)

| Value | Default | Description |
|---|---|---|
| `native_tda` | yes | Native semi-supervised linear rescorer + target-decoy q-values. MVP default (always available). |
| `mokapot` |  | Mokapot Python sidecar (docs/13_sidecars.md). |
| `nn_torch` |  | PyTorch semi-supervised MLP sidecar (`nn_rescore_worker.py`): a nonlinear Percolator/mokapot-style rescorer (CV folds + iterative positive re-selection). On the E.coli benchmark it beats the linear mokapot model on the same PIN, and — being robust to an unfiltered pool — gains further when the extraction gate is opened. Same positional-CLI PIN contract as Mokapot; requires `rescore.python` to point at an interpreter with torch. |
| `percolator` |  | External percolator.exe over the PIN file. |
| `entrapment` |  | Spike-in (entrapment) negative rescorer: treat foreign-proteome PSMs (identified by `entrapment_marker`) as real negatives, train a nonlinear GBM sidecar (out-of-fold by base peptide) or a native linear fallback, and report entrapment-calibrated q-values. The chimeric false matches that in-silico decoys under-model appear as real negatives here, so it closes the FDR-validity gap the decoy schemes cannot. |

### `RollupMethod`

(rust/mumdia/crates/mumdia-core/src/config.rs:1019)

| Value | Default | Description |
|---|---|---|
| `top_n_sum` | yes | Sum of the top-N most abundant peptides (single-run default). |
| `sum` |  | Sum of all group peptides. |

### `RtPredictorKind`

(rust/mumdia/crates/mumdia-core/src/config.rs:80)

| Value | Default | Description |
|---|---|---|
| `native` | yes | Native additive retention-coefficient model (no Python). MVP default so the engine runs with zero external runtime dependencies. |
| `deeplc` |  | DeepLC Python sidecar (docs/13_sidecars.md). |

### `UnknownModPolicy`

(rust/mumdia/crates/mumdia-core/src/config.rs:397)

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
listed with the file it is in.

6 of these are also SET by the engine before the worker starts, so the worker's own fallback applies only when the engine did not set it: `MUMDIA_NN_FOLDS`, `MUMDIA_NN_FOLD_KEYS`, `MUMDIA_NN_ITERS`, `MUMDIA_NN_THREADS`, `MUMDIA_NN_TRAIN_FDR`, `OMP_NUM_THREADS`. See the next table.

| Variable | Side | Default in code | Read at |
|---|---|---|---|
| `CONDA_PREFIX` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs:187` |
| `DEEPLC_FT_THREADS` | sidecar | `"8"` | `scripts/deeplc_finetune.py:23` |
| `MUMDIA_BREW_ITERS` | sidecar | `"20"` | `scripts/mokapot_worker.py:38` |
| `MUMDIA_ENTRAPMENT_MODEL` | sidecar | `"gbm"` | `scripts/entrapment_worker.py:34` |
| `MUMDIA_LR_C` | sidecar | `"1.0"` | `scripts/mokapot_worker.py:48` |
| `MUMDIA_LR_MAX_ITER` | sidecar | `"1000"` | `scripts/mokapot_worker.py:49` |
| `MUMDIA_MOKAPOT_WORKERS` | sidecar | `"3"` | `scripts/mokapot_worker.py:120` |
| `MUMDIA_NN_ALPHA` | sidecar | `"1e-4"` | `scripts/mokapot_worker.py:90` |
| `MUMDIA_NN_BATCH` | sidecar | `4096` | `scripts/nn_rescore_worker.py:268` |
| `MUMDIA_NN_CHUNK` | sidecar | `250000` | `scripts/nn_rescore_worker.py:271` |
| `MUMDIA_NN_DEVICE` | sidecar | `"auto"` | `scripts/nn_rescore_worker.py:279` |
| `MUMDIA_NN_DROPOUT` | sidecar | `0.3` | `scripts/nn_rescore_worker.py:265` |
| `MUMDIA_NN_EARLY_STOP` | sidecar | `1` | `scripts/nn_rescore_worker.py:272` |
| `MUMDIA_NN_EARLY_STOP_TOL` | sidecar | `0.01` | `scripts/nn_rescore_worker.py:273` |
| `MUMDIA_NN_EPOCHS` | sidecar | `25` | `scripts/nn_rescore_worker.py:263` |
| `MUMDIA_NN_FEATURES` | sidecar | `""` | `scripts/nn_rescore_worker.py:355` |
| `MUMDIA_NN_FOLDS` | sidecar | `3` | `scripts/nn_rescore_worker.py:257` |
| `MUMDIA_NN_FOLD_KEYS` | sidecar | `""` | `scripts/nn_rescore_worker.py:394` |
| `MUMDIA_NN_HIDDEN` | sidecar | `"128,64"` in nn_rescore_worker.py; `"128,64,64,32"` in mokapot_worker.py | `scripts/mokapot_worker.py:83`, `scripts/nn_rescore_worker.py:264` |
| `MUMDIA_NN_INIT_SAMPLE` | sidecar | `300000` | `scripts/nn_rescore_worker.py:524` |
| `MUMDIA_NN_INIT_TOPK` | sidecar | `0` | `scripts/nn_rescore_worker.py:646` |
| `MUMDIA_NN_ITERS` | sidecar | `5` | `scripts/nn_rescore_worker.py:262` |
| `MUMDIA_NN_LR` | sidecar | `1e-3` | `scripts/nn_rescore_worker.py:266` |
| `MUMDIA_NN_MAX_ITER` | sidecar | `"200"` | `scripts/mokapot_worker.py:91` |
| `MUMDIA_NN_NEG_RATIO` | sidecar | `0.0` | `scripts/nn_rescore_worker.py:261` |
| `MUMDIA_NN_PREGATHER_GB` | sidecar | `8` | `scripts/nn_rescore_worker.py:274` |
| `MUMDIA_NN_SEEDS` | sidecar | `1` | `scripts/nn_rescore_worker.py:270` |
| `MUMDIA_NN_SOLVER` | sidecar | `"adam"` | `scripts/mokapot_worker.py:89` |
| `MUMDIA_NN_STREAM` | sidecar | `"auto"` | `scripts/nn_rescore_worker.py:316` |
| `MUMDIA_NN_STREAM_GB` | sidecar | `4` | `scripts/nn_rescore_worker.py:325` |
| `MUMDIA_NN_THREADS` | both | `16` | `rust/mumdia/crates/mumdia/src/main.rs:85`, `scripts/nn_rescore_worker.py:300`, `scripts/nn_rescore_worker.py:301` |
| `MUMDIA_NN_TRAIN_FDR` | sidecar | `0.01` | `scripts/nn_rescore_worker.py:269` |
| `MUMDIA_NN_TRAIN_SUB` | sidecar | `0.0` | `scripts/nn_rescore_worker.py:258` |
| `MUMDIA_NN_WARM_EPOCHS` | sidecar | `0` | `scripts/nn_rescore_worker.py:260` |
| `MUMDIA_NN_WARM_START` | sidecar | `0` | `scripts/nn_rescore_worker.py:259` |
| `MUMDIA_NN_WD` | sidecar | `1e-4` | `scripts/nn_rescore_worker.py:267` |
| `MUMDIA_PYTHON` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs:178` |
| `MUMDIA_PYTHON_DEEPLC` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs:177` |
| `MUMDIA_PYTHON_MBR` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs:177` |
| `MUMDIA_PYTHON_MS2PIP` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs:177` |
| `MUMDIA_PYTHON_RESCORE` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs:177` |
| `MUMDIA_RESCORE_MODEL` | both | `"nn"` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs:236`, `scripts/mokapot_worker.py:181`, `scripts/mokapot_worker.py:37` |
| `MUMDIA_XGB_DEPTH` | sidecar | `"6"` | `scripts/mokapot_worker.py:63` |
| `MUMDIA_XGB_JOBS` | sidecar | `"0"` | `scripts/mokapot_worker.py:68` |
| `MUMDIA_XGB_LR` | sidecar | `"0.1"` | `scripts/mokapot_worker.py:64` |
| `MUMDIA_XGB_TREES` | sidecar | `"200"` | `scripts/mokapot_worker.py:62` |
| `OMP_NUM_THREADS` | both | `16` | `rust/mumdia/crates/mumdia/src/main.rs:85`, `scripts/nn_rescore_worker.py:303`, `scripts/nn_rescore_worker.py:304` |
| `VIRTUAL_ENV` | engine | none (unset means off) | `rust/mumdia/crates/mumdia/src/python.rs:187` |

48 variables are read: 10 engine-side, 41 sidecar-side, 3 on both sides.

### Variables the code sets

A variable set here overrides whatever the caller exported, so exporting one
of these has no effect on the process listed. The engine's `--threads` is the
one exception noted in its own help text: it sets `MUMDIA_NN_THREADS` and
`OMP_NUM_THREADS` for the sidecars only if they are not already set.

| Variable | Set by | Value | Site |
|---|---|---|---|
| `KMP_DUPLICATE_LIB_OK` | sidecar | `"TRUE"` | `scripts/deeplc_finetune.py:24` |
| `MKL_NUM_THREADS` | sidecar | `"1"` | `scripts/deeplc_finetune.py:27` |
| `MUMDIA_NN_FOLDS` | engine | `p.cfg.folds.to_string()` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs:1126` |
| `MUMDIA_NN_FOLD_KEYS` | engine | `&foldkeys` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs:1129` |
| `MUMDIA_NN_ITERS` | engine | `p.cfg.num_iter.to_string()` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs:1127` |
| `MUMDIA_NN_THREADS` | engine | `n.to_string()` | `rust/mumdia/crates/mumdia/src/main.rs:86` |
| `MUMDIA_NN_TRAIN_FDR` | engine | `p.cfg.train_fdr.to_string()` | `rust/mumdia/crates/mumdia/src/stages/rescore.rs:1128` |
| `NUMEXPR_NUM_THREADS` | sidecar | `"1"` | `scripts/deeplc_finetune.py:28` |
| `OMP_NUM_THREADS` | both | `"1"` in deeplc_finetune.py; `n.to_string()` in main.rs | `rust/mumdia/crates/mumdia/src/main.rs:86`, `scripts/deeplc_finetune.py:25` |
| `OPENBLAS_NUM_THREADS` | sidecar | `"1"` | `scripts/deeplc_finetune.py:26` |
| `PYTHONIOENCODING` | engine | `"utf-8"` | `rust/mumdia/crates/mumdia/src/sidecar.rs:249` |
| `PYTHONUTF8` | engine | `"1"` | `rust/mumdia/crates/mumdia/src/sidecar.rs:249`, `rust/mumdia/crates/mumdia/src/stages/rescore.rs:1120`, `rust/mumdia/crates/mumdia/src/stages/rescore.rs:881` |

## Unresolved by the generator

Listed rather than omitted, so a parsing gap is visible in the document
instead of looking like an absent field.

Every field whose struct has an `impl Default` resolved from the source.

2 field(s) have no default because the owning struct has no `impl Default`. That is the source's intent, not a parsing gap: a list element must carry every key.

- `peptidoforms.fixed_mods[].name` (`String`)
- `peptidoforms.fixed_mods[].residue` (`char`)

## Coverage

17 structs and 167 fields emitted from `rust/mumdia/crates/mumdia-core/src/config.rs`, plus 21 enumerations, 1 named profile(s), 48 environment variables read and 12 set.

20 field(s) carry a gating marker in their doc comment. 47 field(s) carry no doc comment at all, so their description is empty above. 0 default(s) could not be resolved and 2 have none by design.
