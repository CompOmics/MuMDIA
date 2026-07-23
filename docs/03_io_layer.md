# IO layer: Col/Table, Parquet, report.json, hashing, inspect
> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

The `mumdia-io` crate is the on-disk contract layer for the whole engine. Every
stage reads its path-addressable inputs and writes its outputs through this
crate, so no stage hand-rolls Arrow `RecordBatch`es or touches the Parquet
reader/writer directly. The crate provides five things:

1. A small typed column model (`Col`) and a read-back table (`Table`) over
   Arrow + Parquet, so a stage declares its schema as a `Vec<Col>` and reads it
   back by column name with typed getters (`table.rs`).
2. Parquet write (`write_table`) and read (`Table::read`), SNAPPY-compressed,
   the open self-describing interstage format (plan.md Section 3.3).
3. The per-artifact `<artifact>.report.json` sidecar (`ArtifactReport` in
   `report.rs`) so a stage can be evaluated (row counts, resolved params, key
   distributions, model identity, timing) without loading the full table.
4. blake3 content hashing of files and strings (`hash.rs`), which feeds the
   `content_hash` on every artifact record and the `config_hash` used to
   invalidate downstream artifacts.
5. The `mumdia inspect <artifact>` implementation (`inspect` in `lib.rs`):
   schema + head sample + row count for any Parquet file.

This crate reads no `Config` fields of its own. It is a mechanism layer; the
concrete per-artifact column schemas live in each stage's `write_table` call,
and the frozen schema identifiers live in `mumdia-core` (`schema.rs`). The only
"configuration" here is compile-time (SNAPPY compression, a 64 KiB hash buffer,
the schema-id tuples).

## Files

| path | role |
|---|---|
| `rust/mumdia/crates/mumdia-io/src/lib.rs` | crate root: `init_logging`, `record_artifact`, `inspect`; re-exports the modules |
| `rust/mumdia/crates/mumdia-io/src/table.rs` | `Col` enum (write side), `write_table`, `Table` (read side) and the typed getters |
| `rust/mumdia/crates/mumdia-io/src/report.rs` | `ArtifactReport` struct + `write_for` (the `.report.json` sidecar) |
| `rust/mumdia/crates/mumdia-io/src/hash.rs` | `blake3_file`, `blake3_str` |
| `rust/mumdia/crates/mumdia-io/src/json.rs` | `write_json`, `read_json` (pretty JSON via serde) |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | frozen `(logical name, schema version)` tuples for every artifact |
| `rust/mumdia/crates/mumdia-core/src/manifest.rs` | `ArtifactRecord` / `Manifest` (populated from `record_artifact`) |

## Inputs and outputs

The IO layer is schema-agnostic: it does not itself consume or produce a fixed
set of named artifacts. It is the read/write mechanism that every stage uses.
What is fixed here is the artifact **identity registry** and the shape of the
two JSON sidecars.

### Artifact schema registry (`mumdia-core/src/schema.rs:7-23`)

Each artifact carries a logical schema name and version so a stage can validate
its inputs and a model is never applied under a mismatched schema. They are
`pub const` tuples in the `artifact` submodule, referenced as
`mumdia_core::schema::artifact::PEPTIDES` and so on. The tuples are
`(name, version)`:

| logical name | version | producing stage |
|---|---|---|
| `spectra_ms1` | 1 | convert |
| `spectra_ms2` | 1 | convert |
| `isolation_windows` | 1 | convert |
| `ms2_to_ms1` | 1 | convert |
| `peptides` | 1 | digest |
| `peptidoforms` | 1 | peptidoforms |
| `fragment_library_precursors` | 1 | predict-frag / library-input |
| `fragment_library_fragments` | 1 | predict-frag / library-input |
| `seed_psms` | 1 | search-seed |
| `run_windows` | 1 | rt-im-train |
| `psms_extracted` | 1 | extract |
| `chromatograms` | 1 | extract |
| `features` | 1 | features |
| `psms_competed` | 1 | compete |
| `psms_scored` | 2 | rescore |
| `peptide_quant` | 1 | quant |
| `protein_group_quant` | 1 | quant |

Note only `psms_scored` is at version 2; everything else is version 1. The
version is not consumed for a hard gate in this crate; it is recorded in the
report and the manifest so a downstream reader can detect a mismatch.

### Example concrete column schemas

The IO crate stores no schema definitions; a stage's `write_table(path, vec![
Col::… ])` is the schema. Two examples read from the actual code:

`isolation_windows` (`stages/convert.rs:223-228`), one row per distinct window:
`window_index: u32`, `target: f64`, `lower: f64`, `upper: f64`.
`ms2_to_ms1` (`stages/convert.rs:231-237`): `ms2_scan_index: u32`,
`ms1_scan_index: i32`.

`peptides` (`stages/digest.rs:216-225`): `id: u32`, `peptide: utf8`,
`protein: utf8`, `start: i32`, `end: i32`, `label: utf8`, `target_id: i32`,
`decoy_strategy: utf8`.

To see any artifact's real schema, run `mumdia inspect <artifact.parquet>` (see
below) rather than trusting a doc; the schema is authoritative on disk.

### JSON sidecars produced

- `<artifact>.report.json` (per artifact, written by every stage via
  `ArtifactReport::write_for`, `report.rs:28`). Fields below.
- `manifest.json` (written once by the `run` orchestrator from the collected
  `ArtifactRecord`s, `stages/run.rs`). This crate supplies the per-record
  builder `record_artifact` (`lib.rs:20`); it does not write the manifest file.

## How it works

### Write side: `Col` -> Arrow -> Parquet

`Col` (`table.rs:23-42`) is an enum with one variant per supported column type.
Each variant carries `(String name, Vec<values>)`. The scalar variants are
`I64`, `I32`, `U32`, `F64`, `F32`, `Bool`, `Str`; the nullable variants are
`OptF64`, `OptF32`, `OptI32`, `OptStr` (each a `Vec<Option<T>>`); the list
variants are `ListF32`, `ListF64`, and `LargeListF32`. `LargeListF32`
(`table.rs:41`) is encoded as an Arrow `LargeList` with 64-bit offsets, needed
when the total list-value count across all rows can exceed the ~2.1 billion
limit of a 32-bit `ListArray` offset buffer (for example per-fragment
chromatograms when extraction accepts a very large candidate set). The `Opt*`
variants exist to back conditional and ion-mobility columns under the plan.md
Section 2 missing-value policy (`table.rs:5-6`); ion-mobility columns are written
null throughout the 3D MVP.

`Col` has four private helpers used by the writer (all non-`pub`):
- `name()` (`table.rs:45`): the column name, via a match over every variant.
- `len()` (`table.rs:64`): the row count of the inner `Vec`.
- `field()` (`table.rs:83`): the Arrow `Field`. Scalar variants are
  `nullable = false`; the `Opt*` and all list variants are `nullable = true`.
  List inner items are declared `Field::new("item", Float32/64, true)`.
- `into_array()` (`table.rs:107`): the consuming conversion into an `ArrayRef`.
  It **moves** the `Vec` into the Arrow array instead of cloning, so the column
  data is copied only once during a write. Scalar `Vec<T>` and `Vec<Option<T>>`
  go straight through `PrimitiveArray::from`; list variants are built with a
  `ListBuilder`/`LargeListBuilder`, appending each row's slice and then
  `append(true)` (a present, possibly empty, list).

`write_table(path, cols)` (`table.rs:151`) is the single write entry point and
returns the row count as `u64`:
1. Reject an empty column set (`table.rs:152`).
2. Reject duplicate column names via a `HashSet` (`table.rs:157-162`). Arrow
   allows duplicate names but readers resolve a name to the first match, which
   would silently hide the second column, so this is a hard error.
3. Take `nrows` from column 0 and require every column to match
   (`table.rs:163-173`); a mismatch is a hard error naming the offending column.
4. Build the `Schema` from `field()` over all columns (`table.rs:174-175`),
   then consume the columns into `ArrayRef`s (`table.rs:178`). The `fields`
   vector is captured before the consume so the schema still has everything.
5. `RecordBatch::try_new` (`table.rs:179`), create parent dirs
   (`create_dir_all(...).ok()`, best-effort, `table.rs:182-184`), create the
   file, build `WriterProperties` with `Compression::SNAPPY` (`table.rs:186`),
   and write a single batch through `ArrowWriter`, then `close()`
   (`table.rs:189-191`). One `write_table` call produces exactly one row group /
   one logical batch.

### Read side: Parquet -> `Table` -> typed `Vec`

`Table` (`table.rs:197-201`) holds the `Arc<Schema>`, the `Vec<RecordBatch>`,
and `nrows`. `Table::read(path)` (`table.rs:204`) opens the file, builds a
`ParquetRecordBatchReaderBuilder`, captures the schema, then iterates the reader
collecting every batch and summing `num_rows()`. It errors with context
`"opening {path}"` if the file cannot be opened and `"reading parquet {path}"` if
the builder cannot parse the Parquet footer. The whole file is materialized
into memory; there is no streaming or predicate pushdown.

Column access is by name. `idx(name)` (`table.rs:228`) resolves a name to a
column index via `schema.index_of`, returning a descriptive error listing all
column names if the name is absent. The typed getters each downcast every
batch's column to the concrete Arrow array type and concatenate across batches
into one `Vec`. They error if the downcast fails, so the type is checked at read
time. The message wording is per getter: `"column '<name>' is not
f64|f32|i64|i32|u32|bool"` for the scalar getters (and `opt_f64` reuses the f64
message), `"column '<name>' is not utf8"` for `str` (note: `utf8`, not `str`),
and for `list_f32` either `"column '<name>' is not a list"` when the column is
neither a `List` nor a `LargeList` (`table.rs:419`) or `"list '<name>' inner is
not f32"` when the inner element array is not f32 (`table.rs:396`).

The getters and their exact null behaviour:

- `f64` (`table.rs:234`), `f32` (`table.rs:254`): fast path when
  `null_count() == 0` uses `extend_from_slice(a.values())`; otherwise iterate
  and map a null to `f64::NAN` / `f32::NAN`. Nulls become NaN.
- `i64` (`table.rs:274`), `i32` (`table.rs:294`), `u32` (`table.rs:314`): fast
  path on no nulls; otherwise iterate pushing `a.value(k)` **without checking
  `is_null`**. A null therefore comes through as the underlying buffer value
  (typically 0), not as a sentinel. See the gotcha below.
- `bool` (`table.rs:334`): always iterates `a.value(k)`; never checks null.
- `str` (`table.rs:350`): iterates; a null maps to an empty `String`.
- `opt_f64` (`table.rs:370`): the only null-preserving getter. Returns
  `Vec<Option<f64>>`, mapping a null to `None`.
- `list_f32` (`table.rs:389`): reads an f32 list column and accepts **both**
  `List` (32-bit offsets) and `LargeList` (64-bit offsets) encodings, so a
  chromatogram artifact written by `Col::ListF32` or `Col::LargeListF32` reads
  back through the same call. An outer null list becomes an empty `Vec`; a
  present list is materialized via `f.values().to_vec()`.

`column_names()` (`table.rs:224`) returns the schema field names in order.

### Hashing (`hash.rs`)

`blake3_file(path)` (`hash.rs:8`) streams the file in 64 KiB chunks
(`[0u8; 1 << 16]`) through a `blake3::Hasher` and returns the hex digest. This
is the artifact `content_hash`. It is fallible: it returns `Result` and errors
with context `"hashing {path}"` if the file cannot be opened or a read fails.
`blake3_str(s)` (`hash.rs:23`) is a one-shot hex digest of a string, used for the
`config_hash`; it is infallible and returns a plain `String` rather than a
`Result`. The engine derives the
config hash from `Config::canonical_json()` (`config.rs:1116`, a plain
`serde_json::to_string`), for example at `main.rs:404`. The `convert` command is
a deliberate exception: because `--max-spectra`, `--top-peaks-ms2`, and
`--top-peaks-ms1` are CLI caps that change the spectra output but are not part
of `Config`, they are folded into the hash with a unit-separator (`\u{1f}`)
alongside the canonical config JSON so two different caps do not collapse to the
same `config_hash` (`main.rs:389-392`).

### JSON (`json.rs`)

Both JSON sidecars (`<artifact>.report.json` and `manifest.json`) and every JSON
scalar the engine persists go through this module.
`write_json<T: Serialize>(path, value)` (`json.rs:7`) best-effort-creates the
parent directory (`create_dir_all(...).ok()`, `json.rs:8-9`, same best-effort
policy as `write_table`), serializes with `serde_json::to_string_pretty`
(pretty-printed, human-diffable), and writes the file, erroring with context
`"writing json {path}"` on an I/O failure.
`read_json<T: DeserializeOwned>(path)` (`json.rs:16`) is the counterpart used to
load configs and JSON sidecars back; it errors with context `"reading json
{path}"` if the file cannot be read and `"parsing json {path}"` if
deserialization fails. Serialized key order follows the type's serde field order,
which is why `ArtifactReport.stats` uses a `BTreeMap` (below) to keep key order
deterministic.

### Report sidecar (`report.rs`)

`ArtifactReport` (`report.rs:11-24`) is the per-artifact JSON summary; it derives
`Clone`, `Debug`, `Serialize`, `Deserialize` (`report.rs:10`). A stage
constructs it and calls `write_for(artifact_path)` (`report.rs:28`), which
appends `.report.json` to the artifact path and writes it via
`json::write_json` (pretty-printed). Fields:

| field | type | meaning |
|---|---|---|
| `logical_name` | String | logical artifact name (usually the schema name) |
| `schema_name` | String | schema name from the `schema.rs` tuple |
| `schema_version` | u32 | schema version from the tuple |
| `stage` | String | producing stage, e.g. `"digest"` |
| `rows` | u64 | row count returned by `write_table` |
| `content_hash` | String | `blake3_file` of the artifact just written |
| `params` | `serde_json::Value` | the resolved parameters the stage actually used |
| `stats` | `BTreeMap<String, Value>` | summary key distributions / metrics (ordered) |
| `model_identity` | `Option<String>` | sidecar / predictor identity, else `None` |
| `elapsed_ms` | u128 | wall-clock ms for the stage |

`stats` is a `BTreeMap` (not a `HashMap`) so the JSON key order is
deterministic. Concrete example: `digest` records `params` with enzyme, missed
cleavages, length bounds, decoy strategy, and rng seed, and `stats` with
`n_targets` / `n_decoys` (`stages/digest.rs:229-249`). `convert` has a
stage-local helper `write_reports` (`stages/convert.rs:272-294`) that writes one
report per artifact with a shared `params` and empty `stats`; note this
`write_reports` lives in `convert.rs`, it is not part of the `mumdia-io` public
API. Every other stage builds its `ArtifactReport` directly (see the
`write_for` call sites in `align`, `compete`, `extract`, `search_seed`,
`predict_frag`, `peptidoforms`, `rt_im_train`, `rescore`, `features`, `quant`).

### `record_artifact` and the manifest

`record_artifact(logical_name, schema, path, rows, stage, config_hash)`
(`lib.rs:20`) builds an `ArtifactRecord` (`manifest.rs:10-20`, in
`mumdia_core::manifest`; derives `Clone`/`Debug`/`Serialize`/`Deserialize`) for
the run manifest. The record has nine fields: `logical_name`, `path`, and `rows`
are copied straight from the arguments; `format` is hard-coded to `"parquet"`
(`lib.rs:31`); `schema_name`/`schema_version` come from the `schema` tuple;
`content_hash` is `blake3_file(path)` of the file just written; `producing_stage`
comes from the `stage` argument (the struct field and the argument are named
differently); and `config_hash` is the argument. The
`run` orchestrator calls it once per artifact and records each into the
`Manifest` (`stages/run.rs`, many call sites around `record_artifact(...)`),
which is then serialized to `manifest.json`. Standalone single-stage invocations
write only the `.report.json` sidecar, not a manifest.

### `inspect` (`lib.rs:43`)

`inspect(path)` reads the whole table via `Table::read`, then builds a string:
`artifact: <path>`, `rows: <n>`, a `schema:` block listing each field as
`  <name>: <DataType>` with a ` (nullable)` suffix when the field is nullable,
and a `head:` block. The head is the first up-to-10 rows sliced from the
**first batch only** (`first.slice(0, min(num_rows, 10))`, `lib.rs:57-59`),
formatted with `arrow::util::pretty::pretty_format_batches`. If pretty-print
fails, the head is silently omitted (`Err(_) => {}`, `lib.rs:66`). The CLI
command `Cmd::Inspect { artifact }` (`main.rs:257`) simply prints the returned
string (`main.rs:658-659`).

### Logging

`init_logging()` (`lib.rs:13`) initializes `tracing_subscriber` once, honouring
`RUST_LOG` and defaulting to `info`, with `with_target(false)`. It uses
`try_init` so a second call is a no-op rather than a panic.

## Key types and functions

| name | file:line | what it does |
|---|---|---|
| `Col` (enum) | `table.rs:23` | typed write-side column: scalar, `Opt*`, and list variants |
| `Col::field` | `table.rs:83` | Arrow `Field`; scalars non-nullable, `Opt*`/lists nullable |
| `Col::into_array` | `table.rs:107` | consuming move of the `Vec` into an `ArrayRef` (copy once) |
| `write_table` | `table.rs:151` | validate + write one SNAPPY Parquet batch; returns row count |
| `Table` (struct) | `table.rs:197` | read-back table: schema, batches, nrows |
| `Table::read` | `table.rs:204` | read a Parquet file fully into memory |
| `Table::column_names` | `table.rs:224` | schema field names, in order |
| `Table::f64` / `f32` | `table.rs:234` / `254` | float getters; null -> NaN |
| `Table::i64`/`i32`/`u32` | `table.rs:274`/`294`/`314` | integer getters; null NOT checked (-> buffer value) |
| `Table::bool` | `table.rs:334` | bool getter; null NOT checked |
| `Table::str` | `table.rs:350` | string getter; null -> `""` |
| `Table::opt_f64` | `table.rs:370` | only null-preserving getter; -> `Vec<Option<f64>>` |
| `Table::list_f32` | `table.rs:389` | f32 list getter; reads `List` and `LargeList`; null row -> empty `Vec` |
| `ArtifactReport` | `report.rs:11` | per-artifact JSON summary struct |
| `ArtifactReport::write_for` | `report.rs:28` | write `<artifact>.report.json` |
| `blake3_file` | `hash.rs:8` | streamed blake3 hex digest of a file (`content_hash`) |
| `blake3_str` | `hash.rs:23` | one-shot blake3 hex digest of a string (`config_hash`) |
| `write_json` / `read_json` | `json.rs:7` / `16` | pretty serde JSON write / read, creating parent dirs on write |
| `record_artifact` | `lib.rs:20` | build an `ArtifactRecord` (format hard-coded `"parquet"`) |
| `inspect` | `lib.rs:43` | schema + head(<=10, first batch) + row count as a string |
| `init_logging` | `lib.rs:13` | `tracing` init honouring `RUST_LOG`, default `info` |

## Configuration

This subsystem reads no `Config` fields. It has no config surface of its own, so
the recent pruning of dead config fields in `mumdia-core::config` did not touch
it. Its behaviour is fixed at compile time:

- Compression is `SNAPPY`, hard-coded in `write_table` (`table.rs:186-188`); it
  is not configurable and there is no other codec path.
- The hash read buffer is 64 KiB (`hash.rs:11`).
- Schema identifiers are the constants in `mumdia-core/src/schema.rs:7-23`; a
  new artifact requires a new tuple there, not a config change.
- Dependency features are pinned in the workspace `Cargo.toml`: `arrow` v59 with
  `["prettyprint"]` (needed by `inspect`), `parquet` v59 with
  `default-features = false, features = ["arrow", "snap"]` (pure-Rust snap, no
  cmake/C toolchain), `blake3` v1. Do not re-enable Parquet default features;
  they pull in a C compression backend that breaks the pure-Rust build
  constraint (see CLAUDE.md environment gotchas).

## Invariants, determinism, gotchas

- **Determinism.** `write_table` writes columns in the order given and a single
  batch, so byte layout is stable for stable input. `ArtifactReport.stats` is a
  `BTreeMap`, so JSON key order is fixed. `config_hash` comes from
  `Config::canonical_json` (serde field order) so it is reproducible. blake3 is
  deterministic. All of this is required by plan.md Section 7.
- **Integer/bool null gotcha.** `i64`/`i32`/`u32`/`bool` getters do not check
  `is_null`; in the presence of nulls they return the raw buffer value (usually
  0 / false), silently. This is safe today only because the MVP integer/bool
  columns are written non-nullable (the `I*`/`Bool` variants set
  `nullable = false`). If you write a nullable integer column via `OptI32`,
  reading it back with `i32()` will erase the null distinction. Only `opt_f64`
  is null-preserving on the read side. This is a known gap (CLAUDE.md
  "Correctness": add null-aware getters).
- **Read-side coverage asymmetry.** The write side has `OptF32`, `OptI32`,
  `OptStr`, `ListF64` variants with no matching null-aware / typed reader.
  `OptF32` reads back via `f32()` (null -> NaN, acceptable), `OptStr` via
  `str()` (null -> `""`), `OptI32` via `i32()` (null erased), and there is no
  `list_f64` reader at all. Add the corresponding getter before relying on a
  round-trip of those variants.
- **Inner-list nulls are not represented.** List item fields are declared
  nullable, but the builders only ever `append(true)` present lists and never
  append inner-element nulls; `list_f32` reads inner values with
  `values().to_vec()` and cannot surface an inner null. Only the outer
  list-level null (empty `Vec`) is modeled.
- **Duplicate column names are rejected** at write time (`table.rs:157-162`)
  because Arrow readers resolve to the first match and would hide the rest.
- **Length equality is enforced**: all columns must have identical length or
  `write_table` errors (`table.rs:163-173`).
- **Everything is loaded into memory.** `Table::read` collects all batches and
  `inspect` reads the full table just to print 10 rows; there is no streaming
  path. For very large artifacts this is a real memory cost. `inspect`'s head is
  taken from the first batch only, so a file with tiny leading batches shows few
  rows even when later batches are large.
- **`inspect` swallows pretty-print errors** (`lib.rs:66`): a formatting failure
  omits the head silently rather than erroring, so absence of a head block is
  not proof of an empty table.
- **`create_dir_all` on write is best-effort** (`.ok()`, `table.rs:183`); a real
  permission failure surfaces later at `File::create`, not at the mkdir.
- **`record_artifact` hard-codes `format = "parquet"`** (`lib.rs:31`); it is not
  suitable for a non-Parquet artifact without a change there.
- **LargeList transparency.** `list_f32` intentionally reads both `List` and
  `LargeList`, so an artifact whose encoding differs between two builds (32- vs
  64-bit offsets) still reads identically; do not assume a fixed offset width
  when consuming chromatograms.
- **Test coverage is one round-trip.** The crate's only unit test is
  `roundtrip_mixed_columns` (`table.rs:430`), which writes then reads back
  `U32`/`F64`/`Str`/`OptF64`/`ListF32`/`LargeListF32`, asserting among other
  things that a `LargeListF32` column cross-reads through `list_f32`. `hash.rs`,
  `json.rs`, `report.rs`, and `lib.rs` (`inspect`, `record_artifact`,
  `init_logging`) have no unit tests, and the integer/bool null gotcha and the
  `Opt*`/`ListF64` reader gaps above are consequently not exercised by the suite.

## How to extend / modify

- **Add a new column type.** Add a variant to `Col` (`table.rs:23`) and extend
  the four matches: `name`, `len`, `field`, `into_array`. Decide nullability in
  `field`. Then add the matching read-side getter on `Table` following the
  downcast + concatenate pattern of the existing getters, and be explicit about
  null handling (prefer an `opt_*` return over a silent sentinel for anything
  that can be null in practice).
- **Add a null-aware integer/string getter.** This is the standing correctness
  item. Mirror `opt_f64` (`table.rs:370`): iterate `is_null(k)` and return
  `Vec<Option<T>>`. Do not change the existing non-optional getters' signatures;
  add new ones so current call sites are unaffected.
- **Register a new artifact.** Add a `(name, version)` tuple to
  `mumdia-core/src/schema.rs` and pass it to `write_table` (schema = the
  `Vec<Col>` you write) plus the `ArtifactReport`/`record_artifact` calls. Bump
  the version only on a breaking column-schema change (as was done for
  `psms_scored` -> 2), and document the change so readers can detect a mismatch.
- **Change compression.** It is a one-line change in `write_table`
  (`table.rs:186`); keep it to a codec available under the pinned pure-Rust
  Parquet features, and re-measure round-trip and file size.
- **Extend the report.** Add a field to `ArtifactReport` (`report.rs:11`). Keep
  `stats` a `BTreeMap` for deterministic key order. Since it is
  `Serialize`/`Deserialize`, adding a field is backward compatible only if it is
  `Option` or has a serde default; otherwise old `.report.json` files fail to
  deserialize.
- **Reuse the inspector.** Any tool that needs a schema/head view should call
  `mumdia_io::inspect` rather than re-opening Parquet, so the output format stays
  consistent with the `mumdia inspect` CLI command.
