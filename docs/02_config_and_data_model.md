# Config, mass model, constants, schema, manifest
> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

This document covers the `mumdia-core` crate: the shared vocabulary that every
stage depends on but that contains no stage logic itself. Four concerns live
here.

1. **Typed configuration** (`config.rs`): one serde structure with per-stage
   sections and one strategy enum per algorithmic choice point. `Config` parses
   from JSON, rejects unknown keys, and validates on load so a misconfiguration
   fails at startup rather than producing a bogus run.
2. **Mass model** (`mass.rs` + `constants.rs`): the single source of residue
   masses, physical constants, m/z conversion, ppm predicates, ProForma-lite
   peptidoform parsing, and b/y fragment generation. No other crate defines a
   mass constant or a ppm predicate.
3. **Frozen artifact schema ids** (`schema.rs`): `(logical name, version)` pairs
   stamped into every Parquet artifact and its `report.json` so a stage can
   detect a schema mismatch.
4. **Run manifest** (`manifest.rs`): per-artifact provenance (content hash,
   producing stage, config hash) written once by the `run` orchestrator.

The crate is declared in `lib.rs:6-13` (`config`, `constants`, `error`,
`manifest`, `mass`, `rejection`, `schema`, `types`) and exposes
`version()` from `CARGO_PKG_VERSION` (`lib.rs:15-17`).

## Files

| Path | Role |
|---|---|
| `rust/mumdia/crates/mumdia-core/src/config.rs` | All config structs + every strategy enum; `Config::from_json`, `validate`, `apply_profile`, `canonical_json` (1160 lines) |
| `rust/mumdia/crates/mumdia-core/src/constants.rs` | Physical constants (`PROTON`, `WATER`, `AMMONIA`, `ISOTOPE_SPACING`), `residue_mass`, `mass_to_mz`, ppm predicates |
| `rust/mumdia/crates/mumdia-core/src/mass.rs` | `unimod_mass`, `IonType`, `Fragment`, `ParsedPeptidoform`, `parse_peptidoform`, b/y fragment generation |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | Frozen `(name, version)` ids for every artifact; `PSMS_SCORED` is v2, all others v1 |
| `rust/mumdia/crates/mumdia-core/src/manifest.rs` | `Manifest` + `ArtifactRecord` (provenance) |
| `rust/mumdia/crates/mumdia-core/src/types.rs` | `Peak`, `IsolationWindow`, `Label`, `Ms2Scan` |
| `rust/mumdia/crates/mumdia-core/src/rejection.rs` | `RejectionReason` ladder for the candidate-audit table |
| `rust/mumdia/crates/mumdia-core/src/error.rs` | `MassError`, `ConfigError` (thiserror) |
| `rust/mumdia/crates/mumdia-core/src/lib.rs` | Module wiring + `version()` |
| `rust/mumdia/crates/mumdia-io/src/lib.rs` | `record_artifact` (builds `ArtifactRecord` by hashing the file), `inspect` |
| `rust/mumdia/crates/mumdia-io/src/hash.rs` | `blake3_file`, `blake3_str` (content + config hashing) |
| `rust/mumdia/crates/mumdia-io/src/report.rs` | `ArtifactReport` written as `<artifact>.report.json` |

## Inputs and outputs

`mumdia-core` produces no artifacts of its own. It defines the types the other
crates read and write. Two things it owns appear on disk:

**`manifest.json`** (one per `run`, written at `run.rs:334-335`). Serialized
`Manifest`; fields (`manifest.rs:22-31`):

| Field | Type | Meaning |
|---|---|---|
| `mumdia_version` | String | `CARGO_PKG_VERSION` at build time (`manifest.rs:37`) |
| `config_json` | String | Fully-resolved config, from `Config::canonical_json()` |
| `config_hash` | String | `blake3_str(canonical_json)` (`run.rs:87`) |
| `model_identities` | BTreeMap<String,String> | `rt_predictor`, `fragment_predictor`, `rescorer`, `feature_schema_id` (`run.rs:323-332`) |
| `artifacts` | BTreeMap<String,ArtifactRecord> | one entry per produced artifact, keyed by logical name |

**`ArtifactRecord`** (`manifest.rs:9-20`), one per artifact:
`logical_name`, `path`, `format` (always `"parquet"`), `schema_name`,
`schema_version`, `rows`, `content_hash` (blake3 of the file bytes),
`producing_stage`, `config_hash`. Built by `record_artifact`
(`mumdia-io/src/lib.rs:20-39`), which hashes the written Parquet file with
`blake3_file` (`hash.rs:8-19`, streamed in 64 KiB chunks).

**`ArtifactReport`** / `<artifact>.report.json` (`report.rs:11-24`) is written
alongside each artifact by its producing stage, not by core: `logical_name`,
`schema_name`, `schema_version`, `stage`, `rows`, `content_hash`, `params`
(resolved parameters), `stats` (key metrics), `model_identity`, `elapsed_ms`.
`write_for` appends `.report.json` to the artifact path (`report.rs:26-31`).

### Artifact schema ids (`schema.rs:6-24`)

Every id is a `(&str, u32)` constant. A stage stamps `.0` and `.1` into its
`ArtifactReport` and `ArtifactRecord`.

| Constant | Logical name | Version |
|---|---|---|
| `SPECTRA_MS1` | `spectra_ms1` | 1 |
| `SPECTRA_MS2` | `spectra_ms2` | 1 |
| `ISOLATION_WINDOWS` | `isolation_windows` | 1 |
| `MS2_TO_MS1` | `ms2_to_ms1` | 1 |
| `PEPTIDES` | `peptides` | 1 |
| `PEPTIDOFORMS` | `peptidoforms` | 1 |
| `FRAGMENT_LIBRARY_PRECURSORS` | `fragment_library_precursors` | 1 |
| `FRAGMENT_LIBRARY_FRAGMENTS` | `fragment_library_fragments` | 1 |
| `SEED_PSMS` | `seed_psms` | 1 |
| `RUN_WINDOWS` | `run_windows` | 1 |
| `PSMS_EXTRACTED` | `psms_extracted` | 1 |
| `CHROMATOGRAMS` | `chromatograms` | 1 |
| `FEATURES` | `features` | 1 |
| `PSMS_COMPETED` | `psms_competed` | 1 |
| `PSMS_SCORED` | `psms_scored` | **2** |
| `PEPTIDE_QUANT` | `peptide_quant` | 1 |
| `PROTEIN_GROUP_QUANT` | `protein_group_quant` | 1 |

`PSMS_SCORED` is the only artifact bumped past v1. Its v2 column layout is
written by the rescore stage (`rescore.rs:320-347`) and is the schema a
downstream reader must expect:

| Column | Type | Meaning |
|---|---|---|
| `candidate_id` | U32 | library candidate id |
| `peptidoform` | Str | ProForma-lite string |
| `charge` | I32 | precursor charge |
| `label` | Str | `target` / `decoy` |
| `protein` | Str | protein accession |
| `base_peptide_id` | U32 | interned stripped-peptide id (peptide-level q grouping) |
| `score` | F64 | rescorer score |
| `q_value` | F64 | per-PSM q (pooled) |
| `peptide_q_value` | F64 | peptide-level q (global per base peptide) |
| `protein_group` | Str | protein-group key |
| `pg_q_value` | F64 | protein-group q |
| `global_q_value` | F64 | experiment-wide PSM q (== `q_value` for a single-run rescore) |
| `prelim_score` | F64 | pre-rescore feature-stage score |
| `source` | U32 | run index (0 for single-run; index into `--competed` for experiment-wide) |
| `run_psm_q` | F64 | per-run PSM FDR |
| `experiment_psm_q` | F64 | pooled PSM FDR |
| `precursor_q` | F64 | per (peptidoform+charge) FDR |

The three trailing multi-context q columns (`run_psm_q`, `experiment_psm_q`,
`precursor_q`, `rescore.rs:340-345`) are the reason for the v2 bump; a v1 reader
that assumed the old column set would misread them. `quant.q_filter`
(see [QuantConfig](#quantconfig-quantrs)) selects which of these q columns the
quant stage filters on.

## How it works

### Config load path

`load_config` (`main.rs:363-371`) reads the `--config` file to a string and
calls `Config::from_json` (`config.rs:1046-1051`), which does
`serde_json::from_str` and then `validate()`. With no `--config` it returns
`Config::default()` (`config.rs:1025-1042`), which is fully populated from the
per-field defaults. `--profile <name>` then calls `apply_profile`
(`main.rs:582`, `config.rs:1098-1112`) on top of the loaded config.

`deny_unknown_fields` on every section (`config.rs:207`, `218`, `239`, ...) means
a typo like `{"digest":{"min_len":7,"bogus":1}}` is a parse error, verified by
`unknown_key_rejected` (`config.rs:1136-1139`). `#[serde(default)]` on every
section and field means a partial JSON overlays defaults: `{"digest":{"min_len":7}}`
keeps `max_len=50`, `charge_max=3`, `top_n_peaks=300` (`config.rs:1142-1149`).
The `t::<T>()` helper (`config.rs:199-204`) is a terse `T::default()` used inside
the per-struct `Default` impls.

### `validate()` hard-error checks (`config.rs:1056-1091`)

Four combinations are rejected at load so they never silently produce a wrong
result. Defaults always pass.

1. `digest.decoy.strategy == DiannShift` -> `Invalid`: the engine digest
   produces zero decoys under it, giving an invalid target-decoy FDR
   (`config.rs:1058-1066`).
2. `rt_im_train.calibration_method == None` -> `Invalid`: `None` would silently
   fall through to the linear fit, so it is rejected and the user must pick
   `linear` or `loess` (`config.rs:1067-1073`).
3. `extract.retain_top_peaks == 0` -> `Invalid`: must be `>= 1` (1 = legacy
   single apex) (`config.rs:1074-1080`).
4. `extract.min_frag_corr` not finite or outside `[0, 1]` -> `Invalid`
   (`config.rs:1081-1089`). 0 disables the gate. Verified by
   `explicit_uncapped_seed_and_invalid_gate_are_distinguished`
   (`config.rs:1152-1158`).

`canonical_json` (`config.rs:1116-1118`) serializes the fully-resolved config;
`run` hashes it with `blake3_str` into `config_hash` (`run.rs:87`) and stores the
JSON verbatim in the manifest. There is no separate pretty vs canonical form; it
is a plain `serde_json::to_string`.

### `apply_profile` (`config.rs:1098-1112`)

Only `dia` is defined. It sets `features.set = Extended`,
`extract.apex_count_window = 5`, `extract.apex_rt_prior_s = 120.0`. Any other name
is an `Invalid` error. All other extraction defaults stay at their conservative
baselines; the profile is a convenience shortcut, not a full preset file.

### Mass model math (`mass.rs`, `constants.rs`)

**Neutral mass** (`mass.rs:66-73`): `WATER + n_term_mod + c_term_mod` plus, per
residue, `residue_mass(r) + mod_delta`. `residue_mass` (`constants.rs:26-51`) is
the 20-standard-amino-acid table; `B, J, O, U, X, Z` return `None` (ambiguous /
non-standard) and cause `MassError::AmbiguousResidue` at parse time. Leucine and
isoleucine share `113.084064015` (`constants.rs:34-35`).

**m/z conversion** (`constants.rs:59-62`):
`mass_to_mz(m, z) = (m + z * PROTON) / z`. `precursor_mz` (`mass.rs:76-78`) is
this over the neutral mass.

**Fragment generation** (`mass.rs:82-127`): a forward prefix scan builds b ions
(`b2..b(n-1)`, dropping `b1`) and a reverse suffix scan builds y ions
(`y3..y(n-1)`, dropping `y1` and `y2`) because b1/y1/y2 are low-information
(`mass.rs:93`, `mass.rs:113`). Each fragment m/z is
`(residue_sum_incl_terminus + z * PROTON) / z` (`mass.rs:97`, `mass.rs:116`).
`Fragment` (`mass.rs:45-53`) carries `ion_type`, `ordinal`, `charge`, `mz`, and a
stable `name` (`b3`, or `y5^2` for charge 2, `frag_name` at `mass.rs:130-136`).

**ppm predicates.** Three exist and they differ by which mass normalizes the
difference; a maintainer must not treat them as interchangeable.

- `ppm_diff(observed, theoretical)` (`constants.rs:66-68`):
  `1e6 * (observed - theoretical) / theoretical`. **Signed**, normalized by the
  **theoretical** mass. Used for reporting a mass error and for
  `ppm_match(obs, theo, tol) = |ppm_diff| <= tol` (`constants.rs:72-74`).
- `ppm_bounds(mz, tol)` (`constants.rs:78-81`): returns `(mz - d, mz + d)` with
  `d = mz * tol * 1e-6`. **Symmetric window centered on the query** `mz`,
  normalized by the query itself.
- `within_ppm(a, b, tol)` (`constants.rs:92-96`): `lo = min(a,b)`, `hi = max(a,b)`,
  true iff `hi - lo <= tol * 1e-6 * lo`. **Min-relative**, symmetric in argument
  order. This is the canonical fragment-index predicate (fragindex_spec 2.1):
  it is algebraically `hi/lo <= 1 + delta` and `ln(hi) - ln(lo) <= ln(1 + delta)`,
  the last form being what makes log-space binning exact and is proven exact
  against the log-bin +/-1 probe. `within_ppm_three_forms_agree`
  (`constants.rs:102-122`) checks the three forms round identically away from the
  edge; `within_ppm_edges` (`constants.rs:124-132`) checks edge inclusivity and
  argument-order symmetry.

The three normalize by theoretical (`ppm_diff`), by query center (`ppm_bounds`),
and by the smaller mass (`within_ppm`) respectively, so they disagree at the
tolerance edge. Index probing must use `within_ppm`; do not substitute a
`ppm_bounds` window there.

**UniMod subset** (`mass.rs:13-26`): `Carbamidomethyl` 57.021463735,
`Oxidation` 15.994914620, `Acetyl` 42.010564684, `Phospho` 79.966331090,
`Deamidated` 0.984016106, `Methyl` 14.015650064, `Dimethyl` 28.031300128,
`Carbamyl` 43.005813726. An unknown name is `MassError::UnknownModification`
(`mass.rs:219`), never a silent zero.

**ProForma-lite parse** (`parse_peptidoform`, `mass.rs:143-203`): optional
N-terminal `[Mod]-`, residues each optionally followed by `[Mod]`, optional
trailing `-[Mod]`. A `[Mod]` is a UniMod name or a signed float such as
`[+15.9949]` (`parse_bracket`, `mass.rs:207-223`, tries `unimod_mass` first then
`f64` parse). Non-alphabetic characters and ambiguous residues are errors. A
second mod at the same residue position accumulates (`+=`, `mass.rs:189`); the
data model has one `mods[i]` slot per residue plus separate terminal deltas
(`ParsedPeptidoform`, `mass.rs:56-62`).

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `Config` | config.rs:1008-1042 | Top-level config; 11 stage sections + `mbr` + `rng_seed` |
| `Config::from_json` | config.rs:1046-1051 | Parse (deny unknown) then `validate` |
| `Config::validate` | config.rs:1056-1091 | Four hard-error checks |
| `Config::apply_profile` | config.rs:1098-1112 | `dia` preset only |
| `Config::canonical_json` | config.rs:1116-1118 | Serialize for manifest hashing |
| `residue_mass` | constants.rs:26-51 | 20-AA monoisotopic table; `None` for ambiguous |
| `mass_to_mz` | constants.rs:59-62 | Neutral mass -> m/z |
| `ppm_diff` / `ppm_match` | constants.rs:66-74 | Theoretical-relative signed ppm + tolerance test |
| `ppm_bounds` | constants.rs:78-81 | Query-centered symmetric m/z window |
| `within_ppm` | constants.rs:92-96 | Min-relative canonical index predicate |
| `parse_peptidoform` | mass.rs:143-203 | ProForma-lite parser |
| `ParsedPeptidoform::fragments` | mass.rs:82-127 | b/y ions, drops b1/y1/y2 |
| `unimod_mass` | mass.rs:13-26 | 8-name UniMod subset |
| `Manifest` / `ArtifactRecord` | manifest.rs:9-51 | Run provenance |
| `record_artifact` | mumdia-io/src/lib.rs:20-39 | Build `ArtifactRecord`, hash file |
| `RejectionReason` | rejection.rs:19-113 | Ordered identification-loss ladder |
| `Label` | types.rs:33-51 | Target/Decoy; `.pin()` = +1/-1 |

## Configuration

Every section is `#[serde(default, deny_unknown_fields)]`. Below, each field is
listed with its default and effect. Fields marked **stub** or **default-off** are
called out explicitly.

### Strategy enums (live variants)

| Enum | file:line | Variants (default in bold) | Notes |
|---|---|---|---|
| `DecoyStrategy` | config.rs:17-27 | **`reverse`**, `scramble`, `diann_shift`, `none` | `diann_shift` rejected by validate; `none` produces no decoys (invalid FDR) |
| `MatcherKind` | config.rs:42-45 | `bucketed`, **`fragindex`** | fragment-matcher backend for search-seed + extract |
| `Enzyme` | config.rs:54-59 | **`trypsin_p`**, `trypsin` | cut after K/R, with/without before-P |
| `CalibrationMethod` | config.rs:68-72 | **`loess`**, `linear`, `none` | `none` rejected by validate (falls through to linear) |
| `FeatureSet` | config.rs:81-90 | **`minimal`** (14), `rich` (44), `extended` (~383) | superset battery in `stages/features/` |
| `RtPredictorKind` | config.rs:99-105 | **`native`**, `deeplc` | DeepLC is a Python sidecar |
| `FragPredictorKind` | config.rs:113-119 | **`native`**, `ms2pip` | MS2PIP is a Python sidecar |
| `RescorerKind` | config.rs:128-150 | **`native_tda`**, `mokapot`, `nn_torch`, `percolator`, `entrapment` | see RescoreConfig |
| `PeakClaim` | config.rs:164-188 | **`none`**, `winner_predicted_intensity`, `proportional`, `coelution_winner`, `coelution_proportional`, `coelution_winner_margin` | shared-peak apportionment |
| `GateMode` | config.rs:591-614 | **`apex_pearson`**, `peak_spectral`, `spectral_entropy`, `coelution`, `combined` | which spectral score `min_frag_corr` thresholds |
| `UnknownModPolicy` | config.rs:280-283 | **`error`**, `skip` | unknown-mod behavior |
| `CompetitionMode` | config.rs:713-730 | **`winner_take_all`**, `none`, `features_only`, `unique_evidence`, `margin_gated` | within-group resolution |
| `CompeteGroupBy` | config.rs:734-741 | **`precursor`**, `apex`, `peptidoform_charge` | competition grouping key |
| `RollupMethod` | config.rs:745-750 | **`top_n_sum`**, `sum` | protein rollup |
| `PeakWindowMode` | config.rs:760-771 | **`per_candidate`**, `consensus` | quant integration window |
| `NormalizeMethod` | config.rs:779-792 | `none`, **`median_ratio`**, `median` | cross-run LFQ normalization; `from_token` at 796-803 |
| `QuantQColumn` | config.rs:816-828 | **`peptide_q`**, `psm_q`, `run_psm_q` | which q column quant filters on |
| `MbrStrategy` | config.rs:883-893 | **`none`**, `empirical_library`, `rt_transfer`, `full` | **stub**: only `none` runs; the rest are config hooks (Stage D3 not wired) |
| `DecoyTransfer` | config.rs:900-907 | **`permuted_rt`**, `reverse_sequence`, `both` | MBR false-transfer null (**stub**, MBR-only) |

### `DigestConfig` (config.rs:217-236)

| Field | Default | Effect |
|---|---|---|
| `enzyme` | `trypsin_p` | cleavage rule |
| `missed_cleavages` | 2 | max missed cleavages |
| `min_len` | 5 | min peptide length |
| `max_len` | 50 | max peptide length |
| `decoy.strategy` | `reverse` | decoy scheme (`DecoyConfig`, config.rs:206-215) |

### `PeptidoformsConfig` (config.rs:238-267)

| Field | Default | Effect |
|---|---|---|
| `fixed_mods` | `[{C, Carbamidomethyl}]` | applied to every matching residue |
| `variable_mods` | `[{M, Oxidation}]` | optionally applied |
| `max_variable_mods` | 1 | max simultaneous variable mods |
| `charge_min` | 2 | lowest precursor charge |
| `charge_max` | 3 | highest precursor charge |
| `unknown_modification` | `error` | `error` or `skip` |

`ResidueMod` (config.rs:269-276) is `{residue: char, name: String}` where `name`
is a UniMod name. `deny_unknown_fields` applies but it has no `#[serde(default)]`,
so both keys are required inside a `ResidueMod` entry.

### `PredictFragConfig` (config.rs:290-322)

| Field | Default | Effect |
|---|---|---|
| `predictor` | `native` | fragment-intensity source (native heuristic or MS2PIP) |
| `rt_predictor` | `native` | iRT source (native or DeepLC) |
| `charge2_from_precursor_charge` | 2 | add charge-2 fragments for precursor charge >= this |
| `top_n_fragments` | 6 | fragments kept per candidate |
| `ms2pip_model` | `"HCD"` | MS2PIP model name |
| `ms2pip_python` | `None` | interpreter for MS2PIP sidecar |
| `deeplc_python` | `None` | interpreter for DeepLC sidecar |
| `sidecar_script_dir` | `"scripts"` | directory holding worker scripts |

### `SearchSeedConfig` (config.rs:324-360)

| Field | Default | Effect |
|---|---|---|
| `fdr_seed` | 0.01 | seed FDR cutoff for calibration anchors |
| `fragment_tol_ppm` | 20.0 | fragment match tolerance |
| `report_psms` | 5 | max PSMs reported per spectrum |
| `min_matched_peaks` | 4 | min matched fragments per seed PSM |
| `top_n_peaks` | 300 | probe only the N most intense MS2 peaks (0 = all) |
| `matcher` | `fragindex` | matcher backend |
| `two_pass_mass_cal` | `false` | **default-off** robust two-pass mass calibration (P3.1) |

### `RtImTrainConfig` (config.rs:362-427)

| Field | Default | Effect |
|---|---|---|
| `calibration_method` | `loess` | RT calibration (`none` rejected by validate) |
| `q_train` | 0.01 | q cutoff for calibration anchors |
| `p_rt` | 0.95 | residual percentile for the RT window |
| `rt_window_multiplier` | 1.0 | scales the RT half-window |
| `min_seed_for_calibration` | 50 | min anchors before calibrating |
| `loess_span` | 0.3 | LOESS local-fit fraction |
| `fallback_rt_window_s` | 120.0 | fixed window when calibration cannot fit |
| `finetune_deeplc` | `false` | **default-off** DeepLC multitask fine-tune (nondeterministic; needs `deeplc_python`) |
| `finetune_epochs` | 25 | fine-tune epoch cap (early stopping usually halts earlier) |
| `finetune_patience` | 10 | early-stopping patience |
| `finetune_batch` | 0 | 0 = auto-scale batch to seed size |
| `adaptive_rt_window` | `false` | **default-off** per-region residual window (P3.2/P3.3) |
| `adaptive_rt_bins` | 12 | RT bins for the adaptive window |
| `rt_window_min_s` | 1.0 | lower clamp on any RT half-window |

### `ExtractConfig` (config.rs:429-585)

The largest section; the core extraction stage. Cascade thresholds and apex
selection dominate.

| Field | Default | Effect |
|---|---|---|
| `fixed_scan_window` | 3 | scan-window floor around the apex |
| `frag_tol_ppm` | 20.0 | fragment tolerance |
| `prec_tol_ppm` | 20.0 | precursor tolerance |
| `presence_min_matched` | 3 | tier-(b) min matched fragment count |
| `presence_min_fragments` | 3 | min distinct fragments for acceptance |
| `presence_min_coelution` | 2 | min simultaneously-present fragments over the run |
| `min_frag_corr` | 0.2 | tier-(d) spectral-agreement gate (0 disables; must be in [0,1]) |
| `min_matched_fraction` | 0.0 | tier-(c) min fraction of predicted fragments observed |
| `apex_top_fragments` | 0 | signature-fragment apex (0 = all matched; superseded by `apex_count_tol`) |
| `apex_rt_prior_s` | 0.0 | Gaussian RT prior sigma on apex (0 = off) |
| `apex_count_tol` | 1 | fragment-count apex slack |
| `apex_count_window` | 1 | rolling distinct-fragment count width (1 = none; profile `dia` sets 5) |
| `emit_window_grid` | `true` | zero-filled window-grid chromatograms |
| `bucket_size` | 8192 | m/z bucket size (power of two) |
| `peak_claim` | `none` | shared-peak apportionment mode |
| `emit_contested_features` | `false` | **default-off** `contested_frac` feature (forces two-pass) |
| `peak_claim_margin` | 2.0 | dominance factor for `coelution_winner_margin` |
| `matcher` | `fragindex` | matcher backend |
| `min_coelution_run` | 0 | **default-off** min consecutive co-elution scans |
| `ms1_rescue` | `false` | **default-off** MS1-isotope rescue of gate failures |
| `retain_top_peaks` | 1 | K peak groups per candidate (1 = legacy; must be >= 1) |
| `emit_candidate_audit` | `false` | **default-off** writes `<psms>.audit.parquet` (P0.3) |
| `apex_evidence_rank` | `false` | **default-off** evidence-count apex |
| `emit_gate_diagnostics` | `false` | **default-off** four gate-score columns |
| `gate_mode` | `apex_pearson` | which spectral score `min_frag_corr` thresholds |
| `gate_coelution_min` | 0.5 | second threshold for `gate_mode = combined` |

The `min_frag_corr` default was relaxed from a historical 0.5 to 0.2
(config.rs:557-562). Every default-off knob is explicitly documented as
requiring entrapment/target-decoy FDR validation before use.

### `FeaturesConfig` (config.rs:616-664)

| Field | Default | Effect |
|---|---|---|
| `set` | `minimal` | feature set (minimal / rich / extended) |
| `coelution_corr_threshold` | 0.9 | co-elution correlation cutoff |
| `prec_tol_ppm` | 20.0 | precursor tolerance for MS1 features |
| `bound_features` | `true` | restrict trace features to the elution peak |
| `bound_peak_fraction` | 1/3 | peak-boundary descent fraction of apex height |
| `bound_peak_grace` | 0 | consecutive sub-threshold scans to bridge |
| `bound_from_confident` | `true` | learn one global peak width from confident seed PSMs |
| `bound_confident_pct` | 50.0 | percentile of confident half-widths as the shared width |

### `CompeteConfig` (config.rs:666-703)

| Field | Default | Effect |
|---|---|---|
| `group_by` | `precursor` | competition grouping key |
| `apex_rt_tolerance_s` | 5.0 | RT bucket for `apex` grouping |
| `mode` | `winner_take_all` | within-group resolution |
| `margin` | 0.0 | score margin for `margin_gated` |
| `unique_evidence_min_fragments` | 2 | min unique fragments for `unique_evidence` |
| `emit_competition_audit` | `false` | **default-off** writes `<out>.compete_audit.parquet` |

### `QuantConfig` (config.rs:830-874)

| Field | Default | Effect |
|---|---|---|
| `q_threshold` | 0.01 | peptide-q cutoff for inclusion |
| `top_n_fragments` | 3 | fragments summed per peptidoform |
| `top_n_peptides` | 3 | peptides summed per protein group (TopNSum) |
| `rollup` | `top_n_sum` | protein rollup method |
| `bound_peak` | `true` | integrate only over the detected peak window |
| `peak_fraction` | 1/6 | descent threshold for the peak-window walk |
| `peak_grace` | 1 | zig-zag grace (bridge N sub-threshold scans) |
| `peak_window_mode` | `per_candidate` | per-candidate vs consensus window |
| `reliable_q` | 0.001 | confident-set q for the consensus width |
| `q_filter` | `peptide_q` | which q column to filter on (`QuantQColumn`) |

### `RescoreConfig` (config.rs:951-1002)

| Field | Default | Effect |
|---|---|---|
| `classifier` | `native_tda` | rescorer backend |
| `folds` | 3 | CV folds |
| `train_fdr` | 0.01 | positive-selection FDR |
| `num_iter` | 10 | semi-supervised iterations (native) |
| `python` | `None` | interpreter for a Python rescorer (mokapot/nn_torch/entrapment) |
| `percolator_bin` | `None` | percolator.exe path |
| `entrapment_marker` | `None` | protein substring marking spike-in negatives (required for `entrapment`) |
| `entrapment_exclude` | `None` | substring that de-marks (own-species) |
| `entrapment_contaminant_markers` | `[]` | substrings that keep a spike-in hit as a real target |
| `entrapment_ratio` | 1.0 | N_real_lib / N_entrap_lib scaling |
| `strict` | `false` | **default-off** turn any rescorer failure into a hard error (recommended for production) |

### `MbrConfig` (config.rs:909-949) — stub

Config hooks only; the MBR stage (D3) is not wired into the run chain. Fields:
`strategy` (`none` default), `q_anchor` 0.01, `min_anchor_runs` 2,
`q_transfer` 0.01, `rt_window_s` 20.0, `decoy_transfer` `permuted_rt`,
`consensus_corr_min` 0.0, `requant_all` `false`, `python` `None`. `mbr` is the
one top-level section with an explicit extra `#[serde(default)]` attribute
(config.rs:1022-1023). With `strategy = none` the chain is byte-identical to no
MBR.

### Top-level `Config` (config.rs:1008-1024)

`rng_seed` (default 0) plus the eleven stage sections and `mbr`. `rng_seed`
seeds every RNG (decoy scramble, CV fold assignment) for determinism.

### Removed / not present

The recent cleanup deleted several dead enums that older docs still mention
(`DecoySource`, `ToleranceRegime`, `ScanWindowMode::PeakWidthDerived`,
`FeatureSet::Custom`). They are **not** in the current `config.rs`. Do not
re-add or document them; if you see them referenced elsewhere, that reference is
stale.

## Invariants, determinism, gotchas

- **Single source of masses.** No crate outside `mumdia-core` defines a residue
  mass, `PROTON`, `WATER`, `AMMONIA`, or `ISOTOPE_SPACING`
  (`constants.rs:10-22`). `ISOTOPE_SPACING = 1.003354835` is the 13C-12C mass
  difference used for MS1 envelope spacing. `PROTON = 1.007276466812` is the
  physically correct proton mass, deliberately not DIA-NN's H-atom value
  1.007825035 (`constants.rs:8-10`). All constants are own-derived from
  CODATA/AME; nothing is copied from another engine (clean-room boundary).
- **Three ppm predicates are not interchangeable.** `within_ppm` (min-relative)
  is the only one the fragment index may use; `ppm_bounds` (query-centered) and
  `ppm_diff`/`ppm_match` (theoretical-relative) disagree at the tolerance edge.
  Substituting one for another shifts which fragments match at the boundary.
- **Validation is fail-loud, not fail-quiet.** `DiannShift` decoys and
  `CalibrationMethod::None` are rejected rather than silently degraded
  (`config.rs:1058-1073`). `DecoyStrategy::None` is *not* rejected but produces no
  decoys and therefore an invalid target-decoy FDR; treat it as a diagnostic-only
  setting.
- **`deny_unknown_fields` everywhere.** A misspelled key fails the whole load, so
  a config file cannot silently ignore a setting the author intended.
- **Determinism.** `rng_seed` seeds all randomness. `canonical_json` is a stable
  `serde_json::to_string` (field order fixed by struct declaration order), so the
  same config hashes to the same `config_hash` across runs. `manifest.artifacts`
  and `model_identities` are `BTreeMap`s (`manifest.rs:28-30`), so manifest key
  order is deterministic. Note `finetune_deeplc` fine-tuning is itself
  nondeterministic (no torch seed) and breaks byte-identical reproducibility when
  enabled.
- **Fragment generation drops b1/y1/y2** unconditionally (`mass.rs:93`,
  `mass.rs:113`); a peptide shorter than 2 residues yields no fragments
  (`mass.rs:85-87`).
- **`ParsedPeptidoform::neutral_mass` uses `.expect`** on residue masses
  (`mass.rs:70`), safe only because `parse_peptidoform` already rejected
  ambiguous residues. Constructing a `ParsedPeptidoform` by hand with a
  non-standard residue byte would panic.
- **Second mod at the same position accumulates** (`mass.rs:189`), which is a
  behavior difference from engines that drop it; terminal mods are separate
  fields, not part of `mods[i]`.
- **`PSMS_SCORED` is v2, all other artifacts v1.** A reader that hardcodes v1 for
  the scored table will misread the three multi-context q columns.
- **`content_hash` is the file's blake3, not the logical content.** Any byte
  change (compression, column order) changes the hash; it is a change detector,
  not a canonical-content identity.

## How to extend / modify

- **Add a config field.** Add it to the section struct, give it a value in that
  struct's `Default` impl (every field must have one; `#[serde(default)]` is at
  the struct level), and document the effect inline. Do not hardcode a choice the
  config could express (project convention). If the field can be set to a value
  that would silently corrupt results, add a `validate()` check
  (`config.rs:1056-1091`) that rejects it with `ConfigError::Invalid`.
- **Add a strategy variant.** Extend the enum, keep `#[serde(rename_all =
  "snake_case")]`, and (if it is not implementable yet) reject it in `validate`
  the way `DiannShift` and `CalibrationMethod::None` are, rather than letting it
  fall through. State default-off status in the doc comment.
- **Add a UniMod modification.** Add a `name => mass` arm to `unimod_mass`
  (`mass.rs:13-26`). Use the PSI-MS/UniMod monoisotopic delta so the Python
  sidecar adapters map the name. No table is copied from another tool.
- **Add an artifact / bump a schema.** Add or bump the constant in
  `schema.rs:6-24`. Bump the version whenever the column set changes
  (as `PSMS_SCORED` went to 2). Update the producing stage's `ArtifactReport`
  (`schema_name`, `schema_version`) and its `record_artifact` call so the manifest
  and the sidecar report agree.
- **Add a manifest field.** Extend `Manifest` or `ArtifactRecord`
  (`manifest.rs`); both are plain serde structs. Keep new maps as `BTreeMap` for
  deterministic key order.
- **Never edit `plan.md`** (gitignored spec, project rule). Keep validated
  numbers consistent across `README`, `COMPARISON.md`, and `CLAUDE.md`.
