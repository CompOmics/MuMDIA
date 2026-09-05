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
   recorded in each artifact's `report.json` and in the `manifest.json`
   `ArtifactRecord`. They are provenance only: no stage reads them back, and
   `Table::read` (`table.rs:317-319`, delegating to `read_inner`,
   `table.rs:332-371`) does no version check, so a schema id is never used to
   detect a mismatch or invalidate a downstream artifact.
4. **Run manifest** (`manifest.rs`): per-artifact provenance (content hash,
   producing stage, config hash) written once by the `run` orchestrator.

The crate is declared in `lib.rs:6-13` (`config`, `constants`, `error`,
`manifest`, `mass`, `rejection`, `schema`, `types`) and exposes
`version()` from `CARGO_PKG_VERSION` (`lib.rs:15-17`).

## Files

| Path | Role |
|---|---|
| `rust/mumdia/crates/mumdia-core/src/config.rs` | All config structs + every strategy enum; `Config::from_json`, `validate`, `apply_profile`, `canonical_json` (1585 lines) |
| `rust/mumdia/crates/mumdia-core/src/constants.rs` | Physical constants (`PROTON`, `WATER`, `AMMONIA`, `ISOTOPE_SPACING`), `residue_mass`, `mass_to_mz`, ppm predicates |
| `rust/mumdia/crates/mumdia-core/src/mass.rs` | `unimod_mass`, `IonType`, `Fragment`, `ParsedPeptidoform`, `parse_peptidoform`, b/y fragment generation |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | Frozen `(name, version)` ids for every artifact; `PSMS_SCORED` is v4, `PSMS_COMPETED` is v3, `PSMS_EXTRACTED`/`PEPTIDE_QUANT`/`PROTEIN_GROUP_QUANT` are v2, all others v1 |
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

**`manifest.json`** (one per `run`, written at `run.rs:523-524`). Serialized
`Manifest`; fields (`manifest.rs:22-31`):

| Field | Type | Meaning |
|---|---|---|
| `mumdia_version` | String | `CARGO_PKG_VERSION` at build time (`manifest.rs:36`) |
| `config_json` | String | Fully-resolved config, from `Config::canonical_json()` |
| `config_hash` | String | `blake3_str(canonical_json)` (`run.rs:89`) |
| `model_identities` | BTreeMap<String,String> | `rt_predictor`, `fragment_predictor`, `rescorer`, `feature_schema_id` (`run.rs:512-521`) |
| `artifacts` | BTreeMap<String,ArtifactRecord> | one entry per produced artifact, keyed by logical name |

**`ArtifactRecord`** (`manifest.rs:9-20`), one per artifact:
`logical_name`, `path`, `format` (always `"parquet"`), `schema_name`,
`schema_version`, `rows`, `content_hash` (blake3 of the file bytes),
`producing_stage`, `config_hash`. Built by `record_artifact`
(`mumdia-io/src/lib.rs:20-39`), which hashes the written Parquet file with
`blake3_file` (`hash.rs:8-20`, streamed in 64 KiB chunks).

**`ArtifactReport`** / `<artifact>.report.json` (`report.rs:11-24`) is written
alongside each artifact by its producing stage, not by core: `logical_name`,
`schema_name`, `schema_version`, `stage`, `rows`, `content_hash`, `params`
(resolved parameters), `stats` (key metrics), `model_identity`, `elapsed_ms`.
`write_for` appends `.report.json` to the artifact path (`report.rs:26-31`).

### Artifact schema ids (`schema.rs:7-24`)

Every id is a `(&str, u32)` constant. A stage stamps `.0` and `.1` into its
`ArtifactReport` and `ArtifactRecord`. Both are provenance records only; nothing
reads them back to validate a downstream read (`Table::read`, `table.rs:317-319`,
does no version check, and `write_table`, `table.rs:166-211`, does not stamp the
id into the Parquet file itself).

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
| `PSMS_EXTRACTED` | `psms_extracted` | **2** |
| `CHROMATOGRAMS` | `chromatograms` | 1 |
| `FEATURES` | `features` | 1 |
| `PSMS_COMPETED` | `psms_competed` | **3** |
| `PSMS_SCORED` | `psms_scored` | **4** |
| `PEPTIDE_QUANT` | `peptide_quant` | **2** |
| `PROTEIN_GROUP_QUANT` | `protein_group_quant` | **2** |
| `FRAGMENT_QUANT` | `fragment_quant` | 1 |

`PSMS_SCORED` v3 added the identification apex/bounds so quant can integrate the
selected peak; v4 added `selected_peak_rank` for top-K peak promotion. The
`psms_extracted` and `psms_competed` bumps carry `peak_rank` through the same
chain. The scored column layout is written by the rescore stage
(`rescore.rs:508-540`) and is the schema a downstream reader must expect:

| Column | Type | Meaning |
|---|---|---|
| `candidate_id` | U32 | library candidate id |
| `peptidoform` | Str | ProForma-lite string |
| `charge` | I32 | precursor charge |
| `label` | Str | `target` / `decoy` |
| `protein` | Str | protein accession |
| `base_peptide_id` | U32 | interned stripped-peptide id; the peptide-level q grouping and also the default `compete.group_by = precursor` key |
| `apex_rt` | F64 | identification apex used by downstream quant |
| `elution_lo` / `elution_hi` | F64 | identified elution bounds carried through |
| `score` | F64 | rescorer score |
| `q_value` | F64 | per-PSM q (pooled) |
| `peptide_q_value` | F64 | peptide-level q (global per base peptide) |
| `protein_group` | Str | protein-group key |
| `pg_q_value` | F64 | protein-group q |
| `global_q_value` | F64 | byte-identical clone of `q_value` (rescore.rs:406), kept for backward compatibility |
| `prelim_score` | F64 | pre-rescore feature-stage score |
| `source` | U32 | index of the `--competed` table the PSM came from, stamped during input concat (rescore.rs:65-70, 108); 0 throughout a single-run rescore |
| `run_psm_q` | F64 | per-run PSM FDR: an independent target-decoy analysis within each `source` (rescore.rs:411-437) |
| `experiment_psm_q` | F64 | pooled PSM FDR (clone of `q_value`, rescore.rs:407) |
| `precursor_q` | F64 | per (peptidoform+charge) FDR over the rows that survived compete |
| `selected_peak_rank` | I32 | which enumerated chromatographic peak won: 0 = the up-front apex, >0 = a promoted alternate (`extract.promote_top_peaks`) |

The multi-context q columns, the carried peak coordinates, and
`selected_peak_rank` are the reasons for the schema bumps; an older reader that
assumes the prior column set will misread them. `quant.q_filter`
(see the `QuantConfig` section below) selects which of these q columns the
quant stage filters on.

Three properties of these columns are easy to misread.

- `source` makes pooling safe: because `run_psm_q` re-runs the whole target-decoy
  analysis within each source, a run pooled with others still receives its own
  per-run FDR. The grouped columns (`peptide_q_value`, `precursor_q`,
  `pg_q_value`) are assigned only to each group's winning row and the groups are
  experiment-wide, so a per-run count must use `run_psm_q`.
- Pooling more runs does not tighten `q_value`. The shared kernel in
  `crates/mumdia/src/fdr.rs:38` is `(n_decoys + 1) / max(1, n_targets)`, which is
  invariant under replicating the population apart from the `+1` pseudocount. That
  term is the only pool-size dependence, and its relative weight shrinks as the
  pool grows, so a larger pool is if anything marginally LOOSER, never
  statistically stronger. Batch a large experiment to fit memory; the split does
  not change the FDR each run receives, and a per-run count difference must not be
  attributed to pool size.
- `precursor_q` is a precursor-level unit only when compete was run with
  `group_by = peptidoform_charge`. Under the default `group_by = precursor` the
  charge and modification siblings were already collapsed by stripped peptide, so
  the column counts base peptides. See docs/11_compete_rescore_fdr.md.

## How it works

### Config load path

`load_config` (`main.rs:416-424`) reads the `--config` file to a string and
calls `Config::from_json` (`config.rs:1315-1320`), which does
`serde_json::from_str` and then `validate()`. With no `--config` it returns
`Config::default()` (`config.rs:1293-1311`), which is fully populated from the
per-field defaults. `--profile <name>` then calls `apply_profile`
(`main.rs:647` for `run`, `main.rs:674` for `run-experiment`;
`config.rs:1446-1460`) on top of the loaded config.

`deny_unknown_fields` on every section (`config.rs:262`, `273`, `301`, ...) means
a typo like `{"digest":{"min_len":7,"bogus":1}}` is a parse error, verified by
`unknown_key_rejected` (`config.rs:1542-1546`). `#[serde(default)]` on every
section and field means a partial JSON overlays defaults: `{"digest":{"min_len":7}}`
keeps `max_len=50`, `charge_max=3`, `top_n_peaks=300` (`config.rs:1548-1556`).
The `t::<T>()` helper (`config.rs:254-259`) is a terse `T::default()` used inside
the per-struct `Default` impls. `shipped_configs_parse` (`config.rs:1473-1497`)
additionally parses every tracked config file, because a documentation key such
as `_comment` in a shipped config is a hard parse error under
`deny_unknown_fields`.

### `validate()` hard-error checks (`config.rs:1325-1439`)

Known invalid combinations are rejected at load so they never silently produce a
wrong result. This is targeted validation, not proof that every scientifically
poor setting is detectable. Defaults always pass.

1. `digest.decoy.strategy == DiannShift` -> `Invalid`: the engine digest
   produces zero decoys under it, giving an invalid target-decoy FDR
   (`config.rs:1327-1335`).
2. `rt_im_train.calibration_method == None` -> `Invalid`: `None` would silently
   fall through to the linear fit, so it is rejected and the user must pick
   `linear` or `loess` (`config.rs:1336-1342`).
3. `extract.retain_top_peaks == 0` -> `Invalid`: must be `>= 1` (1 = legacy
   single apex) (`config.rs:1343-1349`).
4. `extract.min_frag_corr` not finite or outside `[0, 1]` -> `Invalid`
   (`config.rs:1350-1358`). 0 disables the gate. Verified by
   `explicit_uncapped_seed_and_invalid_gate_are_distinguished`
   (`config.rs:1577-1584`).
5. `rescore.folds < 2` -> `Invalid`: every PSM needs an out-of-fold score
   (`config.rs:1359-1365`).
6. `rescore.num_iter == 0` -> `Invalid`: iterative model training needs at least
   one iteration (`config.rs:1366-1370`).
7. `rescore.train_fdr` non-finite or outside `(0, 1]` -> `Invalid`
   (`config.rs:1371-1378`).
8. Mokapot or NnTorch without `rescore.python` -> `Invalid`
   (`config.rs:1379-1388`).
9. `classifier = percolator` -> `Invalid`: the adapter is not wired
   (`config.rs:1389-1395`).
10. Entrapment without `rescore.entrapment_marker` -> `Invalid`
    (`config.rs:1396-1404`).

`validate()` also emits two non-fatal warnings (`config.rs:1405-1437`), because
erroring would break configs that already parse: one when `mbr.rt_window_s`,
`mbr.decoy_transfer`, or `mbr.requant_all` is moved off its default (no stage
reads those fields), and one when `mbr.strategy` is any non-`none` variant (only
`none` vs not-`none` is distinguished in code).

`canonical_json` (`config.rs:1464-1466`) serializes the fully-resolved config;
`run` hashes it with `blake3_str` into `config_hash` (`run.rs:89`) and stores the
JSON verbatim in the manifest. There is no separate pretty vs canonical form; it
is a plain `serde_json::to_string`.

### `apply_profile` (`config.rs:1446-1460`)

Only `dia` is defined. It sets `features.set = Extended`,
`extract.apex_count_window = 5`, `extract.apex_rt_prior_s = 120.0`. Any other name
is an `Invalid` error. All other extraction defaults stay at their conservative
baselines; the profile is a convenience shortcut, not a full preset file.

### Mass model math (`mass.rs`, `constants.rs`)

**Neutral mass** (`mass.rs:84-91`): `WATER + n_term_mod + c_term_mod` plus, per
residue, `residue_mass(r) + mod_delta`. `residue_mass` (`constants.rs:26-51`) is
the 20-standard-amino-acid table; `B, J, O, U, X, Z` return `None` (ambiguous /
non-standard) and cause `MassError::AmbiguousResidue` at parse time. Leucine and
isoleucine share `113.084064015` (`constants.rs:35-36`). `is_standard_residue`
(`constants.rs:54-56`) is the boolean form (`residue_mass(aa).is_some()`), used by
the digest to reject peptides containing a non-standard residue byte.

**m/z conversion** (`constants.rs:59-62`):
`mass_to_mz(m, z) = (m + z * PROTON) / z`. `precursor_mz` (`mass.rs:94-96`) is
this over the neutral mass.

**Fragment generation** (`mass.rs:100-143`): a forward prefix scan builds b ions
(`b2..b(n-1)`, dropping `b1`) and a reverse suffix scan builds y ions
(`y3..y(n-1)`, dropping `y1` and `y2`) because b1/y1/y2 are low-information
(`mass.rs:111`, `mass.rs:129`). Each fragment m/z is
`(residue_sum_incl_terminus + z * PROTON) / z` (`mass.rs:115`, `mass.rs:133`).
`Fragment` (`mass.rs:52-57`) stores only `ion_type`, `ordinal`, `charge`, `mz`.
The stable text name (`b3`, or `y5^2` for charge 2) is derived on demand by the
`name()` method (`mass.rs:68-70`, over `frag_name` at `mass.rs:174-180`) rather
than stored, because generation runs over every theoretical ion of every
peptidoform and the top-N truncation then discards most of them; only callers
that keep a row materialize the string. `IonType` (`mass.rs:36-48`) is the
`B`/`Y` enum; `IonType::symbol()` returns the lowercase `'b'`/`'y'` used in the
fragment name. These two series are the only ones the MVP scores (see
docs/18_findings_and_decisions.md).

**Basic-residue counts** (`mass.rs:148-153`, `mass.rs:160-171`).
`basic_residue_count()` counts Arg/His/Lys over the whole peptide and
`fragment_basic_sites(ion, ordinal)` counts them within a b/y fragment's
sub-sequence (a b-ion of ordinal `k` spans the first `k` residues, a y-ion the
last `k`). These back the composition-based charge caps
`peptidoforms.charge_by_basic_residues` and
`predict_frag.charge_by_basic_residues`.

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
  (`constants.rs:102-126`) checks the three forms round identically away from the
  edge; `within_ppm_edges` (`constants.rs:128-136`) checks edge inclusivity and
  argument-order symmetry.

The three normalize by theoretical (`ppm_diff`), by query center (`ppm_bounds`),
and by the smaller mass (`within_ppm`) respectively, so they disagree at the
tolerance edge. Index probing must use `within_ppm`; do not substitute a
`ppm_bounds` window there.

**UniMod subset** (`mass.rs:13-32`), eleven names: `Carbamidomethyl`
57.021463735, `Oxidation` 15.994914620, `Acetyl` 42.010564684, `Phospho`
79.966331090, `Deamidated` 0.984016106, `Methyl` 14.015650064, `Dimethyl`
28.031300128, `Carbamyl` 43.005813726, plus the three cysteine prenylation
deltas (UniMod 44/48/376) `Farnesyl` 204.1878011, `GeranylGeranyl` 272.2504012,
`Hydroxyfarnesyl` 220.1827157. An unknown name is
`MassError::UnknownModification` (`mass.rs:265`), never a silent zero.

**ProForma-lite parse** (`parse_peptidoform`, `mass.rs:187-249`): optional
N-terminal `[Mod]-`, residues each optionally followed by `[Mod]`, optional
trailing `-[Mod]`. A `[Mod]` is a UniMod name or a signed float such as
`[+15.9949]` (`parse_bracket`, `mass.rs:253-269`, tries `unimod_mass` first then
`f64` parse). Non-alphabetic characters and ambiguous residues are errors. A
second mod at the same residue position accumulates (`+=`, `mass.rs:235`); the
data model has one `mods[i]` slot per residue plus separate terminal deltas
(`ParsedPeptidoform`, `mass.rs:75-80`).

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `Config` | config.rs:1276 | Top-level config; 10 stage sections + `mbr` + `experiment` + `rng_seed` |
| `Config::from_json` | config.rs:1315-1320 | Parse (deny unknown) then `validate` |
| `Config::validate` | config.rs:1325-1439 | Ten hard-error checks + two MBR warnings |
| `Config::apply_profile` | config.rs:1446-1460 | `dia` preset only |
| `Config::canonical_json` | config.rs:1464-1466 | Serialize for manifest hashing |
| `residue_mass` | constants.rs:26-51 | 20-AA monoisotopic table; `None` for ambiguous |
| `is_standard_residue` | constants.rs:54-56 | `residue_mass(aa).is_some()` boolean form |
| `mass_to_mz` | constants.rs:59-62 | Neutral mass -> m/z |
| `ppm_diff` / `ppm_match` | constants.rs:66-74 | Theoretical-relative signed ppm + tolerance test |
| `ppm_bounds` | constants.rs:78-81 | Query-centered symmetric m/z window |
| `within_ppm` | constants.rs:92-96 | Min-relative canonical index predicate |
| `parse_peptidoform` | mass.rs:187-249 | ProForma-lite parser |
| `ParsedPeptidoform::fragments` | mass.rs:100-143 | b/y ions, drops b1/y1/y2 |
| `ParsedPeptidoform::basic_residue_count` | mass.rs:148-153 | Arg/His/Lys count for the charge cap |
| `ParsedPeptidoform::fragment_basic_sites` | mass.rs:160-171 | Arg/His/Lys inside a b/y fragment |
| `IonType` | mass.rs:36-48 | `B`/`Y` enum; `symbol()` -> `'b'`/`'y'` |
| `unimod_mass` | mass.rs:13-32 | 11-name UniMod subset |
| `Manifest` / `ArtifactRecord` | manifest.rs:9-51 | Run provenance; `new`/`record`/`get` (manifest.rs:33-51) |
| `record_artifact` | mumdia-io/src/lib.rs:20-39 | Build `ArtifactRecord`, hash file |
| `RejectionReason` | rejection.rs:19-113 | Ordered identification-loss ladder |
| `Label` | types.rs:33-51 | Target/Decoy; `.pin()` = +1/-1, `.is_decoy()` |

### Core data types (`types.rs`)

Ion mobility is `Option`/nullable throughout so one model serves 3D and 4D runs;
the MVP is 3D so every IM field is `None` (`types.rs:1-3`).

| Type | file:line | Fields / behavior |
|---|---|---|
| `Peak` | types.rs:9-13 | `mz` f64, `intensity` f32, `ion_mobility` `Option<f32>` (`None` for Orbitrap DIA) |
| `IsolationWindow` | types.rs:17-30 | `target_mz`/`lower_mz`/`upper_mz` f64, `im_lower`/`im_upper` `Option<f32>`; `covers(mz)` is inclusive m/z containment (`types.rs:25-30`) |
| `Label` | types.rs:33-51 | `Target`/`Decoy` (serde snake_case); `pin()` -> +1/-1 (Percolator), `is_decoy()` |
| `Ms2Scan` | types.rs:54-62 | `scan_index` u32, `id` String, `rt_seconds` f64, `window`, m/z-sorted `peaks` `Vec<Peak>` |

### Error types (`error.rs`)

Both enums derive `thiserror::Error`; the `#[error(...)]` string is the Display
message. Misconfiguration and bad input fail loudly (see
docs/18_findings_and_decisions.md).

| Variant | file:line | Raised when |
|---|---|---|
| `MassError::Parse(String)` | error.rs:8-9 | peptidoform parse failure: stray `-`, non-alphabetic char, unclosed `[`, or no residues (mass.rs:216, 221, 241, 258) |
| `MassError::AmbiguousResidue(char)` | error.rs:10-11 | a residue with no monoisotopic mass (B/J/O/U/X/Z) at parse (mass.rs:227) |
| `MassError::UnknownModification(String)` | error.rs:12-13 | a `[...]` mod that is neither a UniMod name nor a parseable signed float (mass.rs:265) |
| `ConfigError::Parse(String)` | error.rs:18-19 | `serde_json` failure inside `Config::from_json` (config.rs:1317) |
| `ConfigError::Invalid(String)` | error.rs:20-21 | a `validate()` rejection or unknown `--profile` (config.rs:1328-1404, 1454) |

### Candidate-audit reason ladder (`rejection.rs`)

`RejectionReason` is the ordered identification-loss ladder written to the
candidate-audit table (`emit_candidate_audit` / `mumdia audit`). Each candidate's
row records the EARLIEST stage at which it was lost, so the aggregate answers
"where was each DIA-NN-only precursor first lost?" without conflating later
stages. The serialized spelling is `SCREAMING_SNAKE_CASE` and equals
`code()`; `code()` (rejection.rs:50-71) is the stable string written to
Parquet/JSON without a serde round-trip. `stage_order()` (rejection.rs:76-97) is
the ladder position (0 = earliest); `earliest(a, b)` (rejection.rs:106-112) keeps
the smaller `stage_order` and `Reported` never overrides a real rejection;
`is_rejection()` (rejection.rs:100-102) is true for any non-`Reported` variant.

| `stage_order` | Variant / `code()` | Stage lost at |
|---|---|---|
| 0 | `PeptideNotGenerated` / `PEPTIDE_NOT_GENERATED` | search space (A) |
| 1 | `ModificationNotAllowed` / `MODIFICATION_NOT_ALLOWED` | search space (A) |
| 2 | `ChargeOutOfRange` / `CHARGE_OUT_OF_RANGE` | search space (A) |
| 3 | `PrecursorMzOutOfRange` / `PRECURSOR_MZ_OUT_OF_RANGE` | search space (A) |
| 4 | `NoValidFragments` / `NO_VALID_FRAGMENTS` | search space (A) |
| 5 | `WrongIsolationWindow` / `WRONG_ISOLATION_WINDOW` | search space (A) |
| 6 | `RtPruned` / `RT_PRUNED` | candidate generation (B) |
| 7 | `CandidateCapReached` / `CANDIDATE_CAP_REACHED` | candidate generation (B) |
| 8 | `NoFragmentTraces` / `NO_FRAGMENT_TRACES` | extraction (C, D) |
| 9 | `NoPeakGroup` / `NO_PEAK_GROUP` | extraction (C, D) |
| 10 | `PeakNotSelected` / `PEAK_NOT_SELECTED` | peak/peptide ranking (E) |
| 11 | `OutcompetedByTarget` / `OUTCOMPETED_BY_TARGET` | competition (G) |
| 12 | `OutcompetedByDecoy` / `OUTCOMPETED_BY_DECOY` | competition (G) |
| 13 | `FailedPrecursorFdr` / `FAILED_PRECURSOR_FDR` | FDR + reporting (H) |
| 14 | `FailedPeptideFdr` / `FAILED_PEPTIDE_FDR` | FDR + reporting (H) |
| 15 | `RemovedDuringReporting` / `REMOVED_DURING_REPORTING` | FDR + reporting (H) |
| 255 | `Reported` / `REPORTED` | sentinel: reached the final report (not a loss) |

## Configuration

Every section is `#[serde(default, deny_unknown_fields)]`. Below, each field is
listed with its default and effect. Fields marked **default-off**, **inert**, or
**behaviour-changing** are called out explicitly.

### Strategy enums (live variants)

| Enum | file:line | Variants (default in bold) | Notes |
|---|---|---|---|
| `DecoyStrategy` | config.rs:17-28 | **`reverse`**, `scramble`, `diann_shift`, `none` | `diann_shift` rejected by validate; `none` produces no decoys (invalid FDR) |
| `MatcherKind` | config.rs:38-42 | `bucketed`, **`fragindex`** | fragment-matcher backend for search-seed + extract |
| `Enzyme` | config.rs:46-52 | **`trypsin_p`**, `trypsin` | cut after K/R, with/without before-P |
| `CalibrationMethod` | config.rs:56-61 | **`loess`**, `linear`, `none` | `none` rejected by validate (falls through to linear) |
| `FeatureSet` | config.rs:65-75 | **`minimal`** (14), `rich` (44), `extended` (387) | superset battery in `stages/features/`; the `feature_sets_sized` test (`features.rs:1724-1738`) asserts 14, 14+30, and `14 + 30 + extended_names().len() + 6`, which currently resolves to 387 (337 deduplicated family names) |
| `RtPredictorKind` | config.rs:79-86 | **`native`**, `deeplc` | DeepLC is a Python sidecar |
| `FragPredictorKind` | config.rs:90-96 | **`native`**, `ms2pip` | MS2PIP is a Python sidecar |
| `RescorerKind` | config.rs:100-123 | **`native_tda`**, `mokapot`, `nn_torch`, `percolator`, `entrapment` | see RescoreConfig |
| `PeakClaim` | config.rs:132-182 | **`none`**, `winner_predicted_intensity`, `proportional`, `coelution_winner`, `coelution_proportional`, `coelution_winner_margin`, `coelution_multi_cue`, `coelution_demix`, `coelution_shadow` | shared-peak apportionment; the last three are the fragment-competition framework (see below) |
| `ClaimCues` (struct, not an enum) | config.rs:192-248 | `mz_close`, `rt_prior`, `ms1_support`, `reassign`, `apportion_em_iters` + their sigmas | composable per-claimant weight cues for `coelution_multi_cue`; every cue defaults off (weight 1.0) |
| `GateMode` | config.rs:735-758 | **`apex_pearson`**, `peak_spectral`, `spectral_entropy`, `coelution`, `combined` | which spectral score `min_frag_corr` thresholds |
| `UnknownModPolicy` | config.rs:353-357 | **`error`**, `skip` | unknown-mod behavior |
| `CompetitionMode` | config.rs:873-890 | **`winner_take_all`**, `none`, `features_only`, `unique_evidence`, `margin_gated` | within-group resolution |
| `CompeteGroupBy` | config.rs:894-901 | **`precursor`**, `apex`, `peptidoform_charge` | competition grouping key; `precursor` is a misnomer for stripped-peptide grouping (see below) |
| `RollupMethod` | config.rs:905-911 | **`top_n_sum`**, `sum` | protein rollup |
| `PeakWindowMode` | config.rs:916-928 | **`per_candidate`**, `consensus` | quant integration window |
| `NormalizeMethod` | config.rs:936-949 | `none`, **`median_ratio`**, `median` | cross-run LFQ normalization; `from_token` at 953-960 |
| `QuantQColumn` | config.rs:971-988 | **`peptide_q`**, `precursor_q`, `psm_q`, `run_psm_q` | which q column quant filters on |
| `MbrStrategy` | config.rs:1058-1068 | **`none`**, `empirical_library`, `rt_transfer`, `full` | only `none` vs not-`none` is distinguished in code; the three non-`none` variants behave identically and `validate()` warns |
| `DecoyTransfer` | config.rs:1077-1082 | **`permuted_rt`**, `reverse_sequence`, `both` | MBR false-transfer null; **not read by any stage** (`validate()` warns if changed) |
| `FinetuneScope` | config.rs:1197-1225 | **`first_run_only`**, `per_run` | `run-experiment` only: how many DeepLC fine-tunes an experiment pays for |
| `Handoff` | config.rs:1230-1244 | **`tsv`**, `parquet` | how the feature matrix crosses into a sidecar rescorer; `parquet` is nn_torch only (mokapot falls back to `tsv` with a warning) |

#### Variant semantics (behaviorally-rich enums)

The table names every variant; the ones below carry distinct behavior a
maintainer must not confuse. Line refs are `config.rs`.

**`MatcherKind`** (36-42). `Fragindex` (default) is the log-bin CSR matcher: on
narrow-window DIA it is ~1.95x faster in search-seed and ~1.26x in extract with
essentially unchanged IDs (peptides -0.1% on one benchmark run). `Bucketed` is the previous
`Library::page_search` path, retained for A/B comparison and for the AIF
full-range-window case, where the min-relative vs query-relative predicate
difference shifts IDs more.

**`RescorerKind`** (100-123). `NativeTda` (default): native semi-supervised
linear rescorer + target-decoy q, always available. `Mokapot` (105-106): mokapot
Python sidecar over the PIN. `NnTorch` (107-113): PyTorch semi-supervised MLP
sidecar (`nn_rescore_worker.py`), a nonlinear Percolator/mokapot-style rescorer
with CV folds + iterative positive re-selection; on the E.coli benchmark it beats
linear mokapot on the same PIN and gains further when the extraction gate is
opened; requires `rescore.python` to point at a torch interpreter. `Percolator`
(114-115): external `percolator.exe` over the PIN. `Entrapment` (116-122): treats
foreign-proteome PSMs (marked by `entrapment_marker`) as real negatives, trains a
nonlinear GBM sidecar (out-of-fold by base peptide) or a native linear fallback,
and reports entrapment-calibrated q-values; the chimeric false matches that
in-silico decoys under-model appear as real negatives here.

**`PeakClaim`** (132-182) apportions one observed MS2 peak that matches the
fragments of several co-isolated, co-eluting candidates (near-universal in
wide-window DIA: ~98% of fragment m/z collide within tolerance), to stop a
chimeric candidate borrowing a real peptide's peak wholesale. `None` (default,
legacy): every claimant gets the full peak intensity. `WinnerPredictedIntensity`:
winner-take-all by highest predicted intensity for the matching fragment.
`Proportional`: split by predicted intensity. `CoelutionWinner` (two-pass): a
first pass builds each candidate's per-scan summed-matched-intensity elution
profile, then the peak goes to the claimant most eluting at that scan (best
corroborated by its OTHER fragments), not the one predicting the brightest ion.
`CoelutionProportional` (two-pass): split by per-scan elution-profile height.
`CoelutionWinnerMargin` (two-pass): winner-take-all only when the top eluter's
profile height dominates the runner-up by `peak_claim_margin`, else the peak
stays shared (as `None`).

The last three variants are the modular fragment-competition framework and are
all default-off and benchmark/entrapment-gated, because they change the evidence
every downstream feature sees. `CoelutionMultiCue` multiplies the elution-profile
height by the cues enabled in `ClaimCues` (`config.rs:192-248`): `mz_close`
weights by the sub-tolerance ppm offset of the observed peak from each claimant's
predicted m/z, `rt_prior` by a Gaussian on the calibrated predicted RT,
`ms1_support` by whether the claimant's own precursor envelope is present in the
nearest MS1 scan, and `apportion_em_iters` re-seeds the profiles from the
apportioned rather than full intensities. Each cue defaults to weight 1.0, so
with none enabled the variant reduces to `CoelutionWinner`; it only rewrites
intensities when `claim_cues.reassign` is set. `CoelutionDemix` (destructive)
solves a per-scan non-negative least squares over the co-isolated
candidate-by-fragment design matrix and splits each shared peak by the
deconvolution coefficients; `extract.demix_scan_stride` is the practicality lever
that lets it re-solve only every Nth scan. `CoelutionShadow` (destructive, no
solver) estimates each co-eluter's abundance from the channels it alone claims
and lets every candidate keep its intensity minus the interferers' estimated
contributions, so unlike winner-take-all several real co-eluters can both retain
signal.

**`GateMode`** (735-758) selects which spectral-agreement score `min_frag_corr`
thresholds, all computed at the gate from data in hand. `ApexPearson` (default,
legacy): Pearson of observed-vs-predicted fragment intensities at the single apex
scan (one chimeric scan can dominate). `PeakSpectral`: Pearson of the
peak-integrated observed spectrum (each fragment summed over the elution-peak
scans) vs predicted; the standard library-dot-product measure. `SpectralEntropy`:
Li spectral-entropy similarity of the sqrt-transformed apex-scan intensities
(`spectral_entropy_similarity_sqrt`); the full-feature gate search found it the
single best gate discriminator (AUC 0.826 / matched-pool recall 69.8% vs apex
Pearson 0.781 / 64.5%). `Coelution`: predicted-intensity-weighted mean co-elution
correlation of each matched fragment's XIC to the signature reference over the
peak (temporal agreement, orthogonal to intensity). `Combined`: require BOTH
peak-integrated Pearson >= `min_frag_corr` AND co-elution >= `gate_coelution_min`.

**`CompetitionMode`** (873-890). `WinnerTakeAll` (default, legacy): keep only the
top `prelim_score` per group. `None`: keep every candidate (FDR handles
ambiguity). `FeaturesOnly`: same retained set as `None`, with conflict/contested
features carrying the interference signal into rescoring (the name documents
intent for the experiment matrix). `UniqueEvidence`: keep a loser only when
`unique_fragment_count >= unique_evidence_min_fragments`, else winner-take-all
fallback (also falls back to WTA when the feature column is absent).
`MarginGated`: remove a loser only when `winner_score - loser_score >= margin`,
else keep it (conservative removal for the low-FDR region). Target/decoy labels
stay in the competition key in every mode, so a target never competes against its
own decoy.

**`MbrStrategy`** (1043-1068) and **`DecoyTransfer`** (1070-1082) are the recorded
MBR design ladder, not four (respectively three) distinct behaviors. Only
`strategy == None` vs `!= None` is ever tested in the tree, so `RtTransfer` and
`Full` behave exactly like `EmpiricalLibrary` today, and `validate()` warns when
any non-`None` variant is selected. `DecoyTransfer` is read by no stage at all
and `validate()` warns when it is moved off its default. The intended staging is:
`None` (default) reproduces the chain byte-for-byte; `EmpiricalLibrary` builds
the cross-run consensus anchor library only (M1); `RtTransfer` adds cross-run
expected-RT transfer extraction (M2/M3); `Full` adds requantification (M5). The
intended null (M4) is `PermutedRt` (default), transferring real precursors to a
decoupled wrong expected RT, `ReverseSequence`, transferring reverse/scramble
decoys at the same expected RT, or `Both`. All require >= 2 runs.

**`CompeteGroupBy`** (894-901). `Precursor` (default) does **not** group by
precursor. Its full key is `(base_peptide_id, label_code, 0, peak_rank)`
(compete.rs:88), where `base_peptide_id` is the stripped-sequence id
(`pd.factorize(Stripped.Sequence)` on the imported path,
import_diann_lib.py:137; the target peptide row id on the native path,
peptidoforms.rs:216-220), so every charge state and every modification variant of
one stripped peptide shares a group and `winner_take_all` deletes all but the
best-scoring member before rescore. `Apex` replaces the constant `0` with a
rounded apex-RT bucket (`apex_rt_tolerance_s`, compete.rs:89-92).
`PeptidoformCharge` keys `(pform_id, label_code, charge, peak_rank)`
(compete.rs:93-98), so each distinct peptidoform+charge is its own group
(precursor-level, as DIA-NN/Spectronaut report), recovering the sibling forms the
stripped-peptide grouping collapses. `peak_rank` is in the key in every mode, so a
promoted alternate peak competes only against other candidates at the same rank.

The difference is large in a modification-rich search. Measured on a
modification-expanded imported library, the default grouping removed 880,464 of
1,890,239 extracted candidates (46.6%), while `peptidoform_charge` removed 0 rows,
left the peptide count unchanged, and raised precursors per peptide from 1.000 to
1.174 (DIA-NN reports about 1.126 on comparable data). Under the default grouping
a modified form is deleted whenever an unmodified or otherwise-modified sibling
scores higher, so `peptidoform_charge` is required for a PTM-oriented search; it
stays benchmark-gated as a global default because it changes the rescorer training
and FDR population.

The label stays in the key in every mode: a
target never directly competes against its paired decoy in the `compete` stage.
This does not remove target-decoy comparison from FDR: `rescore` later selects
best representatives at each q-value unit and estimates the target-decoy null.

**`PeakWindowMode`** (913-928). `PerCandidate` (default): each candidate's quant
window is anchored at the identified apex when available, with a summed-XIC
fallback for older scored artifacts; its descent bounds can still be stretched by
interference or collapse on sparse peaks. `Consensus`: the median left/right
half-widths of confident peptides applied around each candidate's identified
apex. Consensus widths are estimated independently inside each quant invocation,
not shared across runs.

**`QuantQColumn`** selects which q column quant filters on. `PeptideQ` (default)
uses `peptide_q_value`. `PrecursorQ` uses `precursor_q` and is valid only for a
single-run rescore; after pooled rescoring that grouped q is experiment-wide. It
is also a precursor-level filter only when compete ran with
`group_by = peptidoform_charge`; under the default grouping the sibling forms were
already removed, so it selects base peptides.
`PsmQ` uses pooled `q_value`. `RunPsmQ` uses per-source `run_psm_q` and is the
run-local choice after an experiment-wide rescore. Quant has no source selector:
slice the scored table by `source` before pairing it with each run's
chromatograms. Changing the q column does not perform that slice.

**`RollupMethod`** (903-911): `TopNSum` (default) sums the top-N most abundant
peptides per protein group (`top_n_peptides`); `Sum` sums all group peptides.
**`NormalizeMethod`** (930-960, `quant-lfq` CLI token, not a config field):
`MedianRatio` (default) is a DESeq-style median-of-ratios size factor over
complete-case features, robust to a minority of changing features (does not flatten
a spike-in design's real fold changes); `Median` aligns each run's median intensity
(simpler, less robust to composition shifts); `None` uses raw areas.

**`FinetuneScope`** (1194-1225), consulted only by `run-experiment` and only when
`rt_im_train.finetune_deeplc` is set. `FirstRunOnly` (default) fine-tunes DeepLC
once on the first run's confident seeds and reuses that library everywhere; each
run still fits its own RT calibration on top. Reuse is not free: on a 6-run
benchmark set the owning run reached a median absolute RT residual of 15.2 s
while the five reusing runs reached 20.3 to 25.4 s, their calibrated RT windows
widened from 145 s to 179-227 s, and the degradation was monotonic in
acquisition order, which is chromatographic drift a single fine-tune cannot
track. Wider windows also cost compute downstream (extract roughly doubled,
features up to tripled). It is still the default because one fine-tune instead of
N dominates a large experiment. `PerRun` fine-tunes for every run and measurably
tightens RT, at one full fine-tune per run.

The cost of reuse is a drift effect, not a fixed penalty. Separately measured on a
single file, an already-once-fine-tuned library plus the per-run LOESS gave a
median absolute residual of 6.06 s (MAD 6.11 s, slope 0.9907) against 6.14 s
(MAD 6.18 s) with a fresh per-file fine-tune, that is equal or marginally better,
while removing about 36 min of wall clock (the fine-tune plus a whole-library iRT
prediction over millions of peptidoforms). So reuse is cheap where the
chromatography matches the fine-tune and expensive where it has drifted; prefer
`PerRun` on long batches, and treat periodic re-fine-tuning (not implemented) as
the better answer for very large ones.

**`Handoff`** (1227-1244) selects how the feature matrix crosses into a sidecar
rescorer. `Tsv` (default) writes a Percolator-format PIN, which is what
`mokapot.read_pin` requires. `Parquet` writes an f32 feature table instead and
applies to `nn_torch` only; `mokapot_worker.py` cannot read it, so a mokapot run
falls back to `Tsv` with a warning rather than failing. On a measured
8,858,206-PSM experiment-wide rescore the 30.18 GB TSV exceeded the worker's
streaming threshold, so every iteration re-read a 12.77 GB memmap; Parquet kept
the matrix in memory and the rescore went from 671.6 min to 12 min with the decoy
fraction unchanged at 0.988%. The f32 width is not a new loss: the TSV was
already written at `{:.6}` and the worker casts to f32 regardless.

### `DigestConfig` (config.rs:272-298)

| Field | Default | Effect |
|---|---|---|
| `enzyme` | `trypsin_p` | cleavage rule |
| `missed_cleavages` | 2 | max missed cleavages |
| `min_len` | 5 | min peptide length |
| `max_len` | 50 | max peptide length |
| `decoy.strategy` | `reverse` | decoy scheme (`DecoyConfig`, config.rs:261-270) |
| `n_term_met_excision` | `true` (bool) | when a protein begins with `M`, also emit the initiator-Met-removed form of its N-terminal peptides, re-checked against `min_len`/`max_len` and the standard-residue rule (matches DIA-NN `--met-excision`); keyed on protein position 0, not any leading `M`. Omitting it makes the search database structurally miss these excised peptides |

### `PeptidoformsConfig` (config.rs:300-340)

| Field | Default | Effect |
|---|---|---|
| `fixed_mods` | `[{C, Carbamidomethyl}]` | applied to every matching residue |
| `variable_mods` | `[{M, Oxidation}]` | optionally applied |
| `max_variable_mods` | 1 | max simultaneous variable mods |
| `charge_min` | 2 | lowest precursor charge |
| `charge_max` | 3 | highest precursor charge |
| `charge_by_basic_residues` | `false` | when true, ignore `charge_min`/`charge_max` and emit charges `1..=1+(#R+#H+#K)` per peptide (proton capacity: N-terminus plus one per basic residue) |
| `unknown_modification` | `error` | `error` or `skip` |

`charge_by_basic_residues` restricts the enumerated precursor charge states to
what a peptide can physically hold (one proton on the N-terminus plus one per Arg,
His, Lys). It replaces the fixed range rather than clamping within it, so a peptide
with no basic residue is emitted only at charge 1, and a peptide that cannot reach
`charge_min` is not dropped by `charge_min` but simply enumerated from 1. Pairs with
`predict_frag.charge_by_basic_residues` (fragments). It changes the search /
training / FDR population, so it is benchmark-gated and defaults off. On the
imported path, `import_diann_lib.py --charge-by-basic-residues` applies the same
rule (on the reference DIA-NN E. coli library it dropped 7.9% of precursors, almost
all charge-3 with a single basic residue).

`ResidueMod` (config.rs:342-349) is `{residue: char, name: String}` where `name`
is a UniMod name. The doc comment reserves `residue: '*'` for "any" and notes
terminal mods are handled separately in the MVP (config.rs:345).
`deny_unknown_fields` applies but the struct has no `#[serde(default)]` (unlike
every other section), so both keys are required inside a `ResidueMod` entry.

### `PredictFragConfig` (config.rs:359-399)

| Field | Default | Effect |
|---|---|---|
| `predictor` | `native` | fragment-intensity source (native heuristic or MS2PIP) |
| `rt_predictor` | `native` | iRT source (native or DeepLC) |
| `charge2_from_precursor_charge` | 2 | add charge-2 fragments for precursor charge >= this |
| `charge_by_basic_residues` | `false` | when true, keep a b/y fragment at charge z only if `z <= 1+(basic residues within the fragment)` and `z <= precursor charge`; supersedes `charge2_from_precursor_charge` |
| `top_n_fragments` | 6 | fragments kept per candidate |
| `ms2pip_model` | `"HCD"` | MS2PIP model name |
| `ms2pip_python` | `None` | interpreter for MS2PIP sidecar |
| `deeplc_python` | `None` | interpreter for DeepLC sidecar |
| `sidecar_script_dir` | `"scripts"` | directory holding worker scripts |

### `SearchSeedConfig` (config.rs:401-445)

| Field | Default | Effect |
|---|---|---|
| `fdr_seed` | 0.01 | seed FDR cutoff for calibration anchors |
| `fragment_tol_ppm` | 20.0 | fragment match tolerance |
| `report_psms` | 5 | max PSMs reported per spectrum |
| `min_matched_peaks` | 4 | min matched fragments per seed PSM |
| `top_n_peaks` | 300 | probe only the N most intense MS2 peaks (0 = all) |
| `matcher` | `fragindex` | matcher backend |
| `two_pass_mass_cal` | `false` | **default-off** robust two-pass mass calibration (P3.1) |
| `mass_cal_loess` | `false` | **default-off** m/z-dependent fragment mass calibration: fit a LOESS of calibrant ppm deviation against fragment m/z, write a sampled grid to `<seed>.masscal.json`, and let extract interpolate a per-peak offset instead of the scalar `frag_ppm_offset` |

`search_seed.top_n_peaks` bounds only the seed stage's per-peak index probing. It
is **not** the `convert`/`run` flag `--top-peaks-ms2`, which truncates the stored
spectra artifact itself and so removes those peaks from extraction permanently.
Extract applies no peak cap of its own and simply consumes whatever `convert`
wrote, while the seed cap never touches that artifact. The two are separate
mechanisms but not fully independent, because the seed selects its peaks from what
`convert` already wrote: a conversion cap below `top_n_peaks` also shrinks the
seed's input. Above it they do not interact, which is what was measured on a
50-window Orbitrap DIA run, where changing `--top-peaks-ms2` between 300 and
uncapped left the seed output identical (80,474 PSMs, 14,877 confident) while the
cap cost about 60% of the final peptides (a 2.5x difference). Read
docs/04_convert.md before setting the destructive one.

### `RtImTrainConfig` (config.rs:447-512)

| Field | Default | Effect |
|---|---|---|
| `calibration_method` | `loess` | RT calibration (`none` rejected by validate) |
| `q_train` | 0.01 | q cutoff for calibration anchors |
| `p_rt` | 0.95 | residual percentile for the RT window, taken over the same anchors the calibration was fit on |
| `rt_window_multiplier` | 1.0 | scales the RT half-window |
| `min_seed_for_calibration` | 50 | min anchors before calibrating |
| `loess_span` | 0.3 | LOESS local-fit fraction |
| `fallback_rt_window_s` | 120.0 | fixed window when calibration cannot fit |
| `finetune_deeplc` | `false` | **default-off** DeepLC fine-tune (nondeterministic; needs `deeplc_python`); the default RT path is prediction plus per-run LOESS calibration |
| `library_irt` | `auto` | library-input mode only: `auto` re-predicts the imported iRT with the DeepLC base model when `deeplc_python` is set (else keeps it, with a warning), `deeplc` requires the interpreter, `library` keeps the imported values. Ignored under `finetune_deeplc`. `run-experiment` predicts once per experiment (docs/08 section 4c) |
| `finetune_epochs` | 25 | fine-tune epoch cap (early stopping usually halts earlier) |
| `finetune_patience` | 10 | early-stopping patience |
| `finetune_batch` | 0 | 0 = auto-scale batch to seed size |
| `adaptive_rt_window` | `false` | **default-off** per-region residual window (P3.2/P3.3) |
| `adaptive_rt_bins` | 12 | RT bins for the adaptive window |
| `rt_window_min_s` | 1.0 | lower clamp on any RT half-window |

`p_rt` and the residual statistics reported in `cal.json` are IN-SAMPLE: the
loess is fit on the confident anchors (`rt_im_train.rs:136-140`) and `w_rt` is
then the residual percentile over those same anchors
(`rt_im_train.rs:176-184`). On a
measured run the reported median absolute residual was 6.14 s while the
out-of-sample median against an independent engine's retention times was 17.6 s
(p90 146.3 s), roughly 3x worse. Treat the reported value as a fit diagnostic and
size any external RT tolerance from out-of-sample numbers. See docs/08_rt_im_train.md.

### `ExtractConfig` (config.rs:514-729)

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
| `apex_top_fragments` | 0 | signature-fragment apex: sums the observed intensity of the top-K predicted fragments per scan; `0` falls back to a default of 3 (`extract.rs:1857-1861`), not all-matched |
| `apex_rt_prior_s` | 0.0 | Gaussian RT prior sigma on apex (0 = off) |
| `apex_count_tol` | 1 | fragment-count apex slack |
| `apex_count_window` | 1 | rolling distinct-fragment count width (1 = none; profile `dia` sets 5) |
| `apex_gaussian_sigma_scans` | 0.0 | Gaussian apex smoother sigma in scans (0 = rolling-sum unchanged; opt-in, benchmark-gated) |
| `emit_window_grid` | `true` | zero-filled window-grid chromatograms |
| `bucket_size` | 8192 | m/z bucket size (power of two) |
| `peak_claim` | `none` | shared-peak apportionment mode |
| `claim_cues` | all off | per-claimant weight cues for `peak_claim = coelution_multi_cue` (`ClaimCues`, config.rs:192-248); every cue is weight 1.0 by default, and `claim_cues.reassign` is what makes the arbitration destructive |
| `emit_demix_features` | `false` | **default-off** non-destructive NNLS demix features (`deconv_explained_frac`, `deconv_active`, `deconv_share`); changes no extracted intensity |
| `demix_lambda` | 1.0 | ridge for the demix NNLS solve (keeps it positive-definite and deterministic under wide-window collinearity) |
| `demix_max_candidates` | 64 | cap on design-matrix columns per demix solve, to bound cost on crowded windows |
| `demix_scan_stride` | 1 | scan stride for the destructive `coelution_demix` redistribution; a re-solve is forced whenever a new candidate enters the co-isolated set. Does not affect the demix features |
| `emit_contested_features` | `false` | **default-off** `contested_frac` feature (forces two-pass) |
| `peak_claim_margin` | 2.0 | dominance factor for `coelution_winner_margin` |
| `matcher` | `fragindex` | matcher backend |
| `min_coelution_run` | 0 | **default-off** min consecutive co-elution scans |
| `ms1_rescue` | `false` | **default-off** MS1-isotope rescue of gate failures |
| `retain_top_peaks` | 1 | K peak groups ENUMERATED per candidate; `K>1` only writes the diagnostic `<out-psms>.peaks.parquet` sidecar (1 = legacy; must be >= 1) |
| `promote_top_peaks` | 1 | K peak groups PROMOTED to real feature/rescore rows, each carrying `peak_rank` (selected apex stays rank 0). Must be `<= retain_top_peaks`. **Behaviour-changing and entrapment-gated**: it changes the extracted row population, so compete/rescore must collapse per candidate or the decoy null is K-inflated |
| `alt_peak_min_area_frac` | 0.10 | min integrated area of a promoted alternate as a fraction of the rank-0 peak; only used when `promote_top_peaks > 1` |
| `alt_peak_min_separation_s` | 5.0 | min apex-RT separation of a promoted alternate from the rank-0 apex; only used when `promote_top_peaks > 1` |
| `emit_candidate_audit` | `false` | **default-off**; in `run` gates the separate `audit` stage that writes `candidate_audit.parquet` (`run.rs:428-429`); no stage writes `<psms>.audit.parquet`, although `audit` will read it if present (`audit.rs:52`), so the field's own doc comment overstates what extraction does; no-op for standalone `mumdia extract` (P0.3) |
| `apex_evidence_rank` | `false` | **default-off** evidence-count apex |
| `emit_gate_diagnostics` | `false` | **default-off** four gate-score columns |
| `gate_mode` | `apex_pearson` | which spectral score `min_frag_corr` thresholds |
| `gate_coelution_min` | 0.5 | second threshold for `gate_mode = combined` |

The `min_frag_corr` default was relaxed from a historical 0.5 to 0.2
(config.rs:692-697). Every default-off knob is explicitly documented as
requiring entrapment/target-decoy FDR validation before use.

### `FeaturesConfig` (config.rs:760-823)

| Field | Default | Effect |
|---|---|---|
| `set` | `minimal` | feature set (minimal / rich / extended) |
| `emit_pin` | `true` | write the Percolator-style `.pin` text file requested by `--out-pin`. No MuMDIA stage consumes it (`rescore` builds its own PIN), and at 1.5M rows by 387 features it is a ~5.4 GB text write, so set it false when nothing external needs it |
| `coelution_corr_threshold` | 0.9 | co-elution correlation cutoff |
| `prec_tol_ppm` | 20.0 | precursor tolerance for MS1 features |
| `bound_features` | `true` | restrict trace features to the elution peak |
| `bound_peak_fraction` | 1/3 | peak-boundary descent fraction of apex height |
| `bound_peak_grace` | 0 | consecutive sub-threshold scans to bridge |
| `bound_from_confident` | `true` | learn one global peak width from confident seed PSMs |
| `bound_confident_pct` | 50.0 | percentile of confident half-widths as the shared width |
| `ms1_precursor_features` | `false` | **default-off** MS1 apex-isotope feature `ms1_isotope_height_corr`; it overlaps the existing `ms1_isotope_cosine_apex`, so it is opt-in. The name stays in the battery either way and returns 0.0 when off, so the vector length does not change |

### `CompeteConfig` (config.rs:825-851)

| Field | Default | Effect |
|---|---|---|
| `group_by` | `precursor` | competition grouping key; `precursor` groups by stripped peptide (all charge and modification siblings collapse), `peptidoform_charge` is the real precursor-level key and is required for a PTM search |
| `apex_rt_tolerance_s` | 5.0 | RT bucket for `apex` grouping |
| `mode` | `winner_take_all` | within-group resolution |
| `margin` | 0.0 | score margin for `margin_gated` |
| `unique_evidence_min_fragments` | 2 | min unique fragments for `unique_evidence` |
| `emit_competition_audit` | `false` | **default-off** writes `<out>.compete_audit.parquet` |

### `QuantConfig` (config.rs:990-1041)

| Field | Default | Effect |
|---|---|---|
| `q_threshold` | 0.01 | cutoff applied to the q column selected by `q_filter` |
| `top_n_fragments` | 3 | fragments summed per peptidoform |
| `top_n_peptides` | 3 | peptides summed per protein group (TopNSum) |
| `rollup` | `top_n_sum` | protein rollup method |
| `bound_peak` | `true` | integrate only over the detected peak window |
| `peak_fraction` | 1/6 | descent threshold for the peak-window walk |
| `peak_grace` | 1 | zig-zag grace (bridge N sub-threshold scans) |
| `peak_window_mode` | `per_candidate` | per-candidate vs consensus window |
| `reliable_q` | 0.001 | confident-set q for the consensus width |
| `q_filter` | `peptide_q` | `peptide_q`, `precursor_q` (single-run only), `psm_q`, or `run_psm_q` |
| `interference_envelope` | `false` | apex-outward interference envelope on fragment traces before integration (opt-in, benchmark-gated) |

### `RescoreConfig` (config.rs:1131-1188)

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
| `strict` | `true` | fail on a rescorer sidecar failure or unsupported classifier; set false only for explicit compatibility fallback |
| `handoff` | `parquet` | how the feature matrix reaches a sidecar rescorer (`Handoff`); mokapot/entrapment fall back to TSV automatically (docs/28 section 11) |
| `features` / `features_file` | `None` | explicit feature projection by name (inline list or one-name-per-line file); strict: a missing name is an error |
| `feature_preset` | `all` | named list used when no explicit list is set: `all` every column, `compact` the embedded 114-feature list of docs/28 section 12 (3.4x smaller rescore matrix; the option for pooled rescoring on small machines, -1.2% on the held-out HYE B01 pool). Preset names the table lacks are skipped with a log line |
| `train_neg_ratio` | 3.0 | cap on decoys per positive in each training fold (0 = every decoy) |
| `train_neg_select` | `hybrid` | which decoys survive the cap: `random`, `margin` (highest-scoring), `hybrid` (`train_margin_frac` from the margin, rest random) |
| `train_margin_frac` | 0.5 | margin share under `hybrid` |
| `train_subsample` | 0.0 | random fraction of training rows kept after the cap (0 = all) |
| `train_warm_epochs` | 5 | epochs per self-training iteration when reusing the previous iteration's weights (0 = cold refit, 25 epochs, every iteration) |
| `seeds` | 1 | independent self-training passes rank-averaged out of fold; 3 is the sensitivity recipe of docs/28 section 15 |

### `MbrConfig` (config.rs:1084-1129), partly inert

The MBR stage (D3) is not wired into the run chain. Fields: `strategy` (`none`
default), `q_anchor` 0.01, `min_anchor_runs` 2, `q_transfer` 0.01, `rt_window_s`
20.0, `decoy_transfer` `permuted_rt`, `consensus_corr_min` 0.0, `requant_all`
`false`, `python` `None`. Of these, `rt_window_s`, `decoy_transfer`, and
`requant_all` are read by no stage at all and `validate()` warns when any is
moved off its default, because a config that sets them looks configured but
changes nothing. With `strategy = none` the chain is byte-identical to no MBR.

### `ExperimentConfig` (config.rs:1247-1272)

Options for `mumdia run-experiment` only. `parallel_runs` (default 1) is how many
per-run search chains execute concurrently; 1 is the historical strictly
sequential behavior. Runs are independent so raising it scales nearly linearly in
wall time, but each concurrent run holds its own extraction working set, so the
practical ceiling is memory rather than cores. Results are unaffected: chunks are
processed in index order and completion order never reaches the output.
`finetune_scope` (default `first_run_only`) is consulted only when
`rt_im_train.finetune_deeplc` is set; see `FinetuneScope` above.

### Top-level `Config` (config.rs:1274-1311)

`rng_seed` (default 0) plus the ten stage sections, `mbr`, and `experiment`.
`rng_seed` seeds every RNG (decoy scramble, CV fold assignment) for determinism.
`mbr` and `experiment` are the two sections carrying an extra explicit
`#[serde(default)]` attribute on the field as well as on the struct
(config.rs:1288-1291), so a config predating either section still parses.

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
- **Validation is targeted and fail-loud for known invalid states.** Invalid
  decoy/calibration settings, impossible rescorer coverage, missing required
  sidecar settings, and invalid numeric bounds are rejected. It is not a general
  scientific validator. `DecoyStrategy::None` is still accepted but produces no
  decoys and therefore no valid target-decoy FDR; treat it as diagnostic-only.
- **`deny_unknown_fields` everywhere.** A misspelled key fails the whole load, so
  a config file cannot silently ignore a setting the author intended.
- **An enum variant name is not a specification of what it keys.**
  `CompeteGroupBy::Precursor` (config.rs:894-901) groups by stripped peptide, not
  by precursor, and the downstream `precursor_q` column inherits that unit. When
  adding or reading a grouping variant, check the key the stage actually builds
  (compete.rs:87-99) rather than the variant name.
- **Determinism.** `rng_seed` seeds all randomness inside the engine.
  `canonical_json` is a stable `serde_json::to_string` (field order fixed by
  struct declaration order), so the same config hashes to the same `config_hash`
  across runs. `manifest.artifacts` and `model_identities` are `BTreeMap`s
  (`manifest.rs:28-30`), so manifest key order is deterministic. Sidecars are the
  exception: `deeplc_finetune.py` sets no numpy or torch seed, so
  `finetune_deeplc` is not reproducible byte-for-byte, and the `nn_torch` worker
  seeds numpy and torch but its training kernels are not guaranteed
  bit-deterministic either.
- **Fragment generation drops b1/y1/y2** unconditionally (`mass.rs:111`,
  `mass.rs:129`); a peptide shorter than 2 residues yields no fragments
  (`mass.rs:103-105`).
- **`ParsedPeptidoform::neutral_mass` uses `.expect`** on residue masses
  (`mass.rs:88`), safe only because `parse_peptidoform` already rejected
  ambiguous residues. Constructing a `ParsedPeptidoform` by hand with a
  non-standard residue byte would panic.
- **Second mod at the same position accumulates** (`mass.rs:235`), which is a
  behavior difference from engines that drop it; terminal mods are separate
  fields, not part of `mods[i]`.
- **Correctness changes have bumped five schemas.** `PSMS_EXTRACTED`,
  `PEPTIDE_QUANT`, and `PROTEIN_GROUP_QUANT` are v2; `PSMS_COMPETED` is v3;
  `PSMS_SCORED` is v4. Other registry entries remain v1. Readers must honor the
  registry rather than assume one version globally.
- **`content_hash` is the file's blake3, not the logical content.** Any byte
  change (compression, column order) changes the hash; it is a change detector,
  not a canonical-content identity.

## How to extend / modify

- **Add a config field.** Add it to the section struct, give it a value in that
  struct's `Default` impl (every field must have one; `#[serde(default)]` is at
  the struct level), and document the effect inline. Do not hardcode a choice the
  config could express (project convention). If the field can be set to a value
  that would silently corrupt results, add a `validate()` check
  (`config.rs:1325-1439`) that rejects it with `ConfigError::Invalid`. If the
  field is accepted but not yet read by any stage, add a `validate()` warning
  instead of leaving it silently inert, as the MBR fields do
  (`config.rs:1405-1437`).
- **Add a strategy variant.** Extend the enum, keep `#[serde(rename_all =
  "snake_case")]`, and (if it is not implementable yet) reject it in `validate`
  the way `DiannShift` and `CalibrationMethod::None` are, rather than letting it
  fall through. State default-off status in the doc comment.
- **Add a UniMod modification.** Add a `name => mass` arm to `unimod_mass`
  (`mass.rs:13-32`). Use the PSI-MS/UniMod monoisotopic delta so the Python
  sidecar adapters map the name. No table is copied from another tool.
- **Add an artifact / bump a schema.** Add or bump the constant in
  `schema.rs:7-24`. Bump the version whenever the column set changes
  (as `PSMS_SCORED` went to 4). Update the producing stage's `ArtifactReport`
  (`schema_name`, `schema_version`) and its `record_artifact` call so the manifest
  and the sidecar report agree.
- **Add a manifest field.** Extend `Manifest` or `ArtifactRecord`
  (`manifest.rs`); both are plain serde structs. Keep new maps as `BTreeMap` for
  deterministic key order.
- **`plan.md` is untracked design history, not the contract.** It is excluded by
  the root-Markdown ignore rule, so it is not the place to record current
  behavior: where it disagrees with the code, the tests and this tracked `docs/`
  guide are authoritative. Keep validated numbers consistent across `README`,
  `COMPARISON.md`, and `CLAUDE.md`.
