# Build, test, determinism, clean-room, deployment, gotchas
> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

This document is the developer handoff reference for everything around the engine
code rather than the algorithms themselves: how the Rust workspace builds and why
it is configured the way it is, what the test suite does and does not cover, the
determinism contract the engine must satisfy, the clean-room licensing boundary,
and the three GitHub Actions workflows plus the Docker image that ship the
project. It is descriptive of the current tree, not aspirational; where something
is a stub, unwired, or default-off it is marked as such.

The cargo workspace is `rust/mumdia/`. All engine code lives in three crates
(`mumdia-core`, `mumdia-io`, `mumdia`). The Python sidecars live in `scripts/`,
their non-CI checks in `ci/`, the conda specs in `env/`, the example
configurations in `configs/`, the container definition in `Dockerfile` +
`docker/`, and the CI/release/Docker automation plus Dependabot in
`.github/`.

## Files

| Path | Role |
|---|---|
| `rust/mumdia/Cargo.toml` | Workspace manifest: members, shared dep versions, release profile, the arrow-ipc opt-level override |
| `rust/mumdia/rust-toolchain.toml` | Pins toolchain channel `1.96.1` + `rustfmt`/`clippy`; rustup auto-installs it on first `cargo` call in this dir |
| `rust/mumdia/.cargo/config.toml` | Machine-specific, gitignored: redirects `target-dir` off the OneDrive tree |
| `rust/mumdia/.cargo/config.toml.example` | Committed template of the above; a fresh clone with no local copy builds into `./target` |
| `rust/mumdia/crates/mumdia/Cargo.toml` | The bin+lib crate `mumdia`; `[[bin]]` name `mumdia` -> `src/main.rs`; all deps come from `workspace.dependencies` |
| `rust/mumdia/crates/mumdia-core/Cargo.toml` | Core crate; depends only on `serde`/`serde_json`/`thiserror` (no arrow/parquet, so no I/O layer) |
| `rust/mumdia/crates/mumdia-io/Cargo.toml` | I/O crate; adds `arrow`/`parquet`/`blake3` over `mumdia-core` |
| `rust/mumdia/crates/mumdia/tests/pipeline.rs` | The only integration test file: extract -> features -> compete -> rescore on crafted Parquet |
| `rust/mumdia/crates/mumdia-core/build.rs` | Stamps the git commit and build date into the crate so `manifest.json` can record them |
| `.github/workflows/ci.yml` | Seven jobs: `lint` (fmt + clippy `-D warnings` + rustdoc), `audit` (`cargo audit`/`cargo deny`), `build-test` matrix on ubuntu/macos/windows, `smoke` (end-to-end `run` and `run-experiment` on a generated fixture, ubuntu + windows), `sidecar-imports` (a real conda env per sidecar, matrixed, plus `pip-audit`), `smoke-cross-platform` (asserts the two platforms produced byte-identical `peptides.tsv` and `proteins.tsv`), `sidecars` (compileall + JSON/YAML parse + doc-reference check + generated-document freshness); on push-to-`main`, every PR, weekly, and on demand |
| `.github/workflows/release.yml` | Dormant until a `v*` tag; `validate-tag` gates on tag-equals-workspace-version, ancestry from `main` and a green `ci.yml` for that exact SHA, then builds four target binaries, smoke-tests each, unpacks each archive into a clean directory and runs that archive's own `ci/smoke.sh`, and attaches archives + `.sha256` to the Release. `workflow_dispatch` rehearses everything except the upload |
| `.github/workflows/docker.yml` | Builds the image into the local daemon, smoke-tests it, then pushes to GHCR only on a `v*` tag; build-and-smoke-only on `workflow_dispatch` |
| `.github/dependabot.yml` | Monthly grouped Cargo + GitHub Actions + Docker base-image updates; `arrow*`/`parquet*` grouped apart because they carry the on-disk contract. No `pip` entry: the Python pins live in the pip sections of the `env/` conda specifications, which Dependabot cannot parse |
| `ci/check_doc_refs.py` | Fails when a tracked file cites a Markdown document the repository does not ship |
| `ci/smoke.sh` | End-to-end smoke test: builds the fixture, runs `convert` and `run`, then `ci/check_smoke.py`; runnable locally as well as in CI |
| `ci/make_fixture_mzml.py` | Generates the fixture mzML from `test_data/fixture.fasta` and the library the engine builds from it, so the planted peaks cannot disagree with the mass model |
| `ci/check_smoke.py` | Asserts the smoke run's artifacts, manifest completeness, and schema versions |
| `ci/gen_cli_reference.py`, `ci/gen_config_reference.py` | Generate the CLI and config reference documents from `--help` and the `config.rs` doc comments |
| `ci/gen_third_party_licenses.py` | Generates `THIRD_PARTY_LICENSES.md` from `Cargo.lock` plus the crates' own notice files; `--check` fails when stale |
| `ci/gen_sbom.py` | Generates `sbom.cdx.json` (CycloneDX 1.5) from `cargo metadata --locked`, components plus dependency graph; `--check` fails when stale |
| `test_data/fixture.fasta` | The only committed input datum: a few proteins the fixture generator digests. No mzML or Parquet is committed |
| `Dockerfile` | Two-stage image: Rust build stage + micromamba runtime with two sidecar envs |
| `docker/config.dia.json` | Baked FASTA-digest config (MS2PIP + DeepLC + strict mokapot wired to in-image envs) |
| `docker/config.diann-lib.json` | Baked library-input config (DeepLC fine-tune + strict NnTorch through the torch-capable DeepLC env) |
| `env/docker-rescore.yml` | Conda spec for the in-image `rescore` env (`python=3.11`, `mokapot==0.10.0` + `ms2pip==4.0.0.dev9`) |
| `env/docker-deeplc.yml` | Conda spec for the in-image `deeplc` env (`python=3.11`, `torch==2.12.1+cpu` + `deeplc==4.1.1` from PyPI) |
| `env/mumdia-rescore.yml` | Minimal host env for the default mokapot rescorer only (`python=3.12`, no torch/DeepLC/MS2PIP) |
| `env/mumdia-deeplc.yml` | Host env for the DeepLC sidecars (`python=3.11`, `torch==2.12.1+cpu`, `deeplc==4.1.1`, `psm-utils`, `pyarrow`) |
| `configs/examples/*.json` | Portable example configs (`native`, `fasta-sidecars`, `diann-library`), all using `"auto"` interpreters |
| `rust/mumdia/crates/mumdia/src/sidecar.rs` | Sidecar subprocess clients + `resolve_script` path resolution |
| `rust/mumdia/crates/mumdia/src/main.rs` | Thin CLI + global flags (`--threads`, `--log-level`, `-v`, `-q`); `doctor` reports whether the config can run (`main.rs:438`) |
| `rust/mumdia/crates/mumdia/src/python.rs` | Sidecar interpreter roles and `"auto"` discovery (`python.rs:35` for `Role`, `python.rs:245` for `discover`) |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `Config::validate` (`config.rs:1434`) + `apply_profile` (`config.rs:1587`) |
| `rust/mumdia/crates/mumdia-core/src/constants.rs` | Physical constants; the clean-room provenance note (`constants.rs:1-6`) |

## Inputs and outputs

The build/test/deploy area does not itself produce the engine's Parquet artifacts
(the stages do). The load-bearing "artifacts" here are the compiled binary, the
container image, release archives, and the small Parquet fixtures that the
integration test crafts by hand. The fixture schemas are worth documenting because
they are the minimal input contract that any new stage-level test must satisfy;
they are built in `tests/pipeline.rs`.

`lib_prec.parquet` (crafted at `tests/pipeline.rs:31`): `candidate_id:u32`,
`peptidoform_id:u32`, `base_peptide_id:u32`, `peptidoform:str`, `charge:i32`,
`precursor_mz:f64`, `predicted_irt:f32`, `label:str` (`target`/`decoy`),
`protein:str`, `n_fragments:i32`.

`lib_frag.parquet` (`tests/pipeline.rs:50`): `candidate_id:u32`, `mz:f64`,
`predicted_intensity:f32`, `name:str`, `ion_type:str`, `ordinal:i32`,
`frag_charge:i32`.

`ms2.parquet` (`tests/pipeline.rs:129`): `scan_index:u32`, `id:str`,
`rt_seconds:f64`, `window_id:u32`, `window_target:f64`, `window_lower:f64`,
`window_upper:f64`, `precursor_mz:OptF64`, `precursor_charge:OptI32`,
`mz:ListF32`, `intensity:ListF32`.

`windows.parquet` (`tests/pipeline.rs:151`): `candidate_id:u32`,
`rt_pred_cal:f64`, `rt_lo:f64`, `rt_hi:f64`, `im_pred_cal:OptF64`, `im_lo:OptF64`,
`im_hi:OptF64` (the IM columns are always null in the 3D MVP).

The build stage of the Docker image produces `/build/release/mumdia`
(`Dockerfile:30`, target dir forced by `ENV CARGO_TARGET_DIR=/build` at
`Dockerfile:27`); the release workflow produces `mumdia-<tag>-<target>.{tar.gz,zip}`
archives (`release.yml:67-103`). The archive is a working installation, not just
an executable: it stages the binary, `README.md`, `LICENSE`, `CHANGELOG.md`,
`docs/`, `scripts/`, `env/`, and `configs/` when the tag carries it
(`release.yml:74-84`), because the ML predictors and rescorers are Python
sidecars the engine launches by path and a bare binary cannot run them.
Non-Windows targets ship a `.tar.gz` (`tar czf`, `release.yml:90`); the Windows
target ships a `.zip` (`7z a`, `release.yml:87`). Each archive gets a sha256
sidecar written with whichever of `sha256sum`/`shasum` the runner has
(`release.yml:95-99`), and all three patterns are uploaded
(`release.yml:105-112`). Both `docker/*.json` configs are copied into the image
as `/opt/mumdia/config.dia.json` and `/opt/mumdia/config.diann-lib.json`. Input
mzML, FASTA, and library Parquets remain user data and must be mounted.

## How it works

**Build configuration.** The workspace declares three members
(`Cargo.toml:3`) sharing one `[workspace.package]` (`Cargo.toml:5-23`): version
`0.1.0`, `edition = "2021"`, `rust-version = "1.96"`, license `Apache-2.0`, plus
the crates.io metadata (description, repository, homepage, keywords, categories)
and `publish = false`. The declared MSRV deliberately tracks
`rust-toolchain.toml` (channel `1.96.1`), which is the only Rust version CI
builds and tests; the edition-2024 transitive dependencies need at least 1.85, so
a lower floor may well work, but it is not verified and is therefore not claimed
(`Cargo.toml:7-11`). Lower it only together with an MSRV job that proves it.

Every dependency is version-anchored once in `[workspace.dependencies]`
(`Cargo.toml:25-43`) and each crate re-exports it with `x.workspace = true` (for
example `crates/mumdia/Cargo.toml:20-34`). The workspace uses the v2 feature
resolver (`resolver = "2"`, `Cargo.toml:2`) and lists three members
(`Cargo.toml:3`). Each crate declares only the subset it needs: `mumdia-core`
depends on `serde`/`serde_json`/`thiserror` only (no arrow/parquet, so the core
types stay I/O-free); `mumdia-io` adds `arrow`/`parquet`/`blake3` over
`mumdia-core`; the `mumdia` bin+lib crate pulls the full set including `clap`,
`rayon`, `mzdata`, and `tracing`. The engine needs **no cmake and no system C libraries**, which is what the feature
selection buys: `parquet` uses `default-features = false, features =
["arrow","snap"]` to drop the C zlib-ng backend that needs cmake and to use the
pure-Rust SNAPPY codec (`Cargo.toml:36-37`); `mzdata` uses `["mzml",
"miniz_oxide"]` for a pure-Rust zlib backend (`Cargo.toml:38-39`); `arrow` adds
`["prettyprint"]` for `inspect` (`Cargo.toml:35`).

It is **not** a pure-Rust build, which this section previously claimed. `cc` is a
build dependency of exactly two crates, and `cargo tree -i cc` shows both:
`libmimalloc-sys` compiles the vendored mimalloc C allocator that the binary
installs globally (`Cargo.toml:40-43`, `main.rs:12-13`), and `blake3` compiles its
SIMD paths. So a C compiler is required, though nothing external to the dependency
tree is: no cmake, no system zlib, no vendored library to install first. Every
supported platform's default toolchain provides the compiler, which is why this went
unnoticed.

The dropped codecs are visible at the read side, not only at build time. SNAPPY
is the only codec compiled in, so any Parquet the engine is asked to read must be
SNAPPY (or uncompressed); a zstd file fails with `Parquet error: Disabled feature
at compile time: zstd`. `mumdia-io` always writes SNAPPY
(`mumdia-io/src/table.rs:205`), so this only bites on tables produced outside the
engine. The same applies to string columns: `Table::str` downcasts to arrow
`StringArray` and rejects anything else with `column '<name>' is not utf8`
(`mumdia-io/src/table.rs:503-511`), so a `large_utf8` column is refused. pandas
and pyarrow default to SNAPPY plus `utf8`; Polars defaults to zstd plus
`large_utf8` and produces a file the engine cannot load. See
`docs/13_sidecars.md` for the sidecar-side statement of the same contract.

The release profile is `opt-level = 3` with `lto = "thin"`, `codegen-units = 1`
and `strip = "symbols"` (`Cargo.toml:45-52`; the strip was added because release
archives were shipping debug symbols that are most of the binary size for no user
benefit, and a backtrace still names the functions), with one exception:
`arrow-ipc` is pinned to `opt-level = 1` (`Cargo.toml:58-59`). Its generated
FlatBuffer code reliably crashes rustc codegen at opt-level 3 on the Windows
toolchain (`STATUS_ILLEGAL_INSTRUCTION`); MuMDIA never uses Arrow IPC, so lowering
just that one crate's optimization avoids the crash with no runtime cost.

**Target directory off OneDrive.** `.cargo/config.toml` sets
`build.target-dir = "C:/Users/robbi/mumdia_build"` (`config.toml:5`). Building
under the OneDrive-synced tree causes sync churn to lock and corrupt incremental
build artifacts, surfacing as nondeterministic `STATUS_ACCESS_VIOLATION` compiler
crashes. This file is machine-specific and gitignored; the committed
`.cargo/config.toml.example` documents the same fix and notes the alternative of
setting the `CARGO_TARGET_DIR` env var. A fresh clone with no local
`config.toml` builds into the default `./target` and works anywhere. In the
Docker build the redirect is overridden by `ENV CARGO_TARGET_DIR=/build`
(`Dockerfile:27`) so the copied source cannot pick up a stray machine path.

**Test suite.** `cargo test --workspace` runs 178 `#[test]` functions across the
workspace. 176 are inline `#[cfg(test)]` unit tests co-located with their code; 2
are the integration tests in `tests/pipeline.rs`. The end-to-end coverage is
separate and does not run under `cargo test`: `ci/smoke.sh` drives the built
binary over a generated fixture. The full validation gate per `CLAUDE.md` is
`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and
`cargo test --workspace` (plus `cargo build --release --locked`). CI now enforces
all of it and more, plus the smoke run and the non-Rust checks (see the CI
workflow below), so `fmt` and `clippy -D warnings` are no longer a local-only
responsibility. Run them locally anyway, because they are cheap and the CI
feedback loop is not.
Coverage by area (counts are `#[test]` occurrences per file, recounted with
`grep -c "#\[test\]"`):

- `mumdia-core` (29): `config.rs` (10), `mass.rs` (8), `rejection.rs` (5),
  `manifest.rs` (4), `constants.rs` (2). Config tests cover the quant
  fixed-window fields and their validation (`config.rs:1615`), that every shipped
  `configs/examples/*.json` and `docker/*.json` parses under
  `deny_unknown_fields` (`config.rs:1651`), the
  experiment fine-tune scope and parallel-run defaults (`config.rs:1679,1698`),
  default round-trip (`config.rs:1711`), unknown-key rejection
  (`config.rs:1722`), partial override keeping defaults (`config.rs:1728`),
  `quant.q_filter` parse/serialize (`config.rs:1738`), the
  invalid-rescore-contract branches (`config.rs:1748`), and the
  uncapped-seed-versus-invalid-gate distinction (`config.rs:1757`).
- `mumdia-io` (2): `table.rs` (2), covering a mixed-column Parquet round-trip and
  `read_cols` agreeing with a full `read` on the selected subset.
- `mumdia` (147): `quant.rs` (22), `peaks.rs` (9), `compete.rs` (8),
  `peptidoforms.rs` (7), `digest.rs` (7), `python.rs` (7), `fragindex.rs` (7),
  `prescan.rs` (6), `extract.rs` (6), `solve.rs` (6), `chromatographic.rs` (5),
  `quant_lfq.rs` (5), `main.rs` (5), `rt_im_train.rs` (4), `rescore.rs` (4),
  `report.rs` (4), `mass_uncertainty.rs` (4), `apex_dispersion.rs` (4),
  `features.rs` (4), `fdr.rs` (4), `order_consistency.rs` (3), `binning.rs` (3),
  `calibrate.rs` (3), `stats.rs` (2), `audit.rs` (2), `search_seed.rs` (1),
  `predict_frag.rs` (1), `rescoring.rs` (1), `index.rs` (1), plus the 2 in
  `tests/pipeline.rs`. (`rescoring.rs` is the native-rescorer scoring kernel;
  `stages/rescore.rs` is the rescore-stage orchestration and coverage checks.)

The integration test `extract_recovers_planted_target_and_is_deterministic`
(`tests/pipeline.rs:189`) plants a target with three fragments across five scans
and a decoy with no matching peaks, runs `extract` twice, and asserts the target
(candidate 0) is accepted, the decoy (1) is not, the two runs are byte-equal on
`candidate_id` and `apex_rt`, and the apex lands on the most-intense scan
(rt 140). `features_compete_rescore_run_on_crafted_input`
(`tests/pipeline.rs:218`) drives the rest of the chain on the same fixtures and
checks the PIN header and that a `q_value` column is produced. Both run the
**native** paths only (`Config::default()`), and `tmp()` (`tests/pipeline.rs:10`)
gives every crafted file a per-process, atomic-counter-unique path because cargo
runs tests concurrently in one process.

**What each module's tests assert (unit coverage detail).** This enumerates the
substance behind the counts above, so a reader knows what "green" verifies at the
function level.

_`mumdia-core`:_

- `mass.rs`: PEPTIDE neutral monoisotopic mass; Carbamidomethyl by-name vs
  by-mass agree; N-term/C-term mods; low-information fragments dropped; the basic
  residue count counts only R/H/K; per-fragment basic sites by slice;
  unknown-mod error; ambiguous-residue error.
- `rejection.rs`: rejection `code()` strings match the spec; serde round-trip uses
  the spec spelling; `earliest` keeps the smaller (earlier) stage; `is_rejection`
  flags only losses; stage order is a monotone ladder.
- `config.rs`: default round-trips (`min_len=5`, `FeatureSet::Minimal`,
  `top_n_peaks=300`, `rescore.strict=true`); unknown key rejected; partial override
  keeps defaults; the `quant.q_filter` enum parses and re-serializes to its snake
  token; the invalid-rescore contracts are rejected (`folds<2`, `num_iter=0`,
  `nn_torch` without `python`, `percolator`, `entrapment` without a marker); an
  explicit uncapped seed (`search_seed.top_n_peaks=0`, allowed) is distinguished
  from an invalid gate (`extract.gate_min_score` outside `[0,1]`, rejected); the
  six quant fixed-window fields round-trip and their invalid values are rejected;
  every configuration the repository ships (`configs/examples/*.json`,
  `docker/*.json`) loads under `deny_unknown_fields`, which is the real guard
  behind the CI JSON-parse step; and the experiment defaults are sequential runs
  with a first-run-only fine-tune scope.
- `constants.rs`: the three `within_ppm` forms agree; the inclusive edge case.

_`mumdia-io`:_ `table.rs` round-trips a mixed-column `Table` through Parquet and
checks that `read_cols` on a column subset returns exactly what a full `read`
returns for those columns (the only two I/O-layer unit tests).

_`mumdia`:_

- `peaks.rs`: empty/zero profile yields no peaks; a single triangular peak; `K=1`
  keeps the strongest by area; a true peak is retained under a dominant
  interferent with top-K; two maxima ranked by area; a peak truncated at the left
  edge; a prominence filter suppresses noise flicker; overlapping maxima collapse
  to the stronger; ties break by earlier apex (determinism).
- `quant.rs`: trapezoid area; window clip to range; the peak window bounds the
  summed XIC and rejects a lone interferent; grace bridges a single dip; median of
  odd/even/empty; median-ratio recovers a global scale not real changes; `None`
  leaves the matrix unnormalized; the peak-window apex prefers co-elution over a
  lone interferent; the identification apex anchors the quant window against a
  brighter off-apex peak; the fragment summary distinguishes missing, all-zero, and
  positive traces (absent evidence is nullable, not an abundance of zero); protein
  rollup uses one maximum per `base_peptide_id`; a quant run preserves
  unquantifiable IDs and honours the applied-window contract; the peak window never
  collapses to a single sample; the base sequence strips mods and a `DECOY_`
  prefix; a centred envelope clips wing interference; the fixed window in scan and
  second forms selects the apex sub-window; `flank_baseline` uses the flank
  quantile; `select_fragment_areas` ranks by predicted (library) intensity;
  fixed-window plus predicted selection uses the library fragments at the apex;
  adding a `predicted_intensity` column does not move a default quantity
  (the bit-identity guard for the new options being off); an empty chromatogram
  table preserves every identification instead of dropping it; and predicted
  selection without that column is a clear error rather than a wrong number.
- `compete.rs`: `winner_take_all` keeps only the winner; `none`/`features_only`
  keep all; `margin_gated` keeps close losers and removes distant ones;
  `unique_evidence` keeps losers with enough unique evidence and falls back to
  winner-take-all without evidence data; `unique_evidence` prefers the Extended
  `peak_contested` fraction when it is present; winner-take-all is deterministic
  across groups; the winner tie-breaks to the smallest index.
- `matchers/fragindex.rs`: the window probe and the callback probe agree;
  fragindex-vs-naive equivalence; two fragments hitting
  one peak count both; the epoch counter resets with no carry across scans;
  tolerance edges (inside matches, outside does not); the precursor-window gate
  excludes out-of-range; the +/-1 bin probe finds within-tolerance pairs across
  bin boundaries.
- `matchers/binning.rs`: within-tolerance pairs are at most one bin apart;
  out-of-range clamps; a boundary-straddling pair is covered by the probe.
- `quant_lfq.rs`: a single sample is the column sum; recovers a known profile
  (complete and with missing values); total intensity preserved; a disconnected
  sample falls back to its own sum.
- `stages/extract.rs`: scalar and grid interpolation agree; co-eluting fragments
  score high; a non-co-eluting interferent drops the score; too few scans does not
  reject; peak-spectral is high when the integrated pattern matches and recovers a
  fragment absent at the apex scan.
- `stages/features/chromatographic.rs`: arity matches the names list; empty input
  is all-finite-zero; a single point is finite; a Gaussian peak scores well; a
  bimodal trace is detected.
- `stages/features/apex_dispersion.rs`: names match the values length and are
  finite; co-eluting fragments have low dispersion; scattered fragments have high
  dispersion; truncation is flagged when the apex sits at an edge.
- `stages/features/mass_uncertainty.rs`: names match values length and are finite;
  only observed fragments count toward the mass error; breadth of the top
  predicted ions; concentration is high when one fragment dominates.
- `stages/features/order_consistency.rs`: length stable on a degenerate input; a
  perfectly consistent intensity order scores high; a shuffled order scores low.
- `stages/features.rs`: `peptide_length` ignores mods; the feature sets are sized
  (Minimal = 14, Rich = 44, Extended = 381, checked against the family battery);
  an unavailable (NaN/inf) RT calibration contributes zero error rather than
  leaking into the matrix; cross-correlation of aligned traces.
- `fdr.rs`: perfect separation gives the conservative `(n_decoys+1)/n_targets` q;
  tied scores share one q; the entrapment q ranks real targets above spike-ins and
  shares one q across ties.
- `calibrate.rs`: linear fit recovers a line; LOESS tracks a nonlinear curve;
  percentile basics.
- `stages/digest.rs`: Trypsin/P cleaves after K/R; Met excision emits both
  N-terminal forms and only at protein position 0; the reverse decoy keeps the
  C-terminal residue; the scramble is deterministic; `collision_safe_decoy` avoids
  collisions with targets and with other decoys; an impossible low-complexity
  sequence drops the target/decoy pair rather than emitting a colliding decoy.
- `stages/peptidoforms.rs`: ProForma places mods at the right positions; the
  variable-mod combination count is bounded; the default rules and form order are
  preserved; same-site alternatives never stack and are deterministic; the `skip`
  unknown-modification policy removes the offending form; fixed-mod stacking and
  fixed/variable overlap are rejected; invalid charges and wildcard modifications
  are rejected.
- `stages/audit.rs`: the waterfall assigns the earliest loss per candidate; the
  entrapment label comes from a protein-name substring.
- `stages/report.rs`: mods and a `DECOY_` prefix are stripped; the report filters
  on label and threshold and never writes a decoy; nothing passing still writes a
  header rather than an empty file; quantities join and unquantified rows stay
  empty.
- `manifest.rs` (`mumdia-core`): `new` stamps the build identity (commit and build
  date from `build.rs`); inputs and artifacts are keyed and ordered; an older
  manifest written before the provenance fields still parses; and the whole
  manifest round-trips through JSON.
- `stats.rs`: Pearson (perfect and flat) and cosine/angle.
- `main.rs`: the global flags parse on either side of the subcommand;
  `log_filter` maps `-v`/`-vv`/`-q`/`--log-level` onto tracing levels;
  `--threads 0` is rejected; the conversion cap defaults to uncapped and an
  explicit cap is preserved (CLI parsing).
- `python.rs`: each `Role` reports its config field and env var; `required_by`
  follows the configuration, so a native config resolves nothing; an explicit path
  is honoured and a missing one is named; a discovery failure names every place it
  looked; `"auto"` is case-insensitive; and `resolve_script_dir` prefers a
  directory that actually holds the workers.
- `solve.rs`: the identity problem recovers `b`; a negative solution is clamped to
  zero; two-column apportionment; collinear columns stay deterministic under the
  ridge term; the zero problem is zero; and precomputed normal equations are
  bit-identical to a per-move rebuild (the determinism guard for the solver).
- `stages/prescan.rs`: tokenisation merges I and L and rejects an unknown
  modification; anchored trimers cover only the anchor and are reversible; a
  modified residue mass is backbone plus delta; the spectrum-trimer scan finds a
  planted ladder; a bad modification spec is an error.
- `stages/predict_frag.rs`: distinct precursors are counted per m/z bin.
- `index.rs`: page search finds only in-window and in-tolerance hits.
- `search_seed.rs`: zero selects all and the seed cap keeps only the top-intensity
  peaks.
- `rescoring.rs`: the native rescorer separates targets from decoys.
- `stages/rescore.rs`: a one-sided population (all targets or all decoys) is
  refused with both counts named, rather than producing a meaningless q; the
  competed feature schema must match on id and ordered columns; sidecar scores must
  give exact, unique, finite coverage of every row; an exact picked-peptide score
  tie is won by the decoy (conservative FDR).
- `stages/rt_im_train.rs`: anchor vectors are sorted by `base_peptide_id`
  (deterministic fit input); the held-out split is deterministic and
  fraction-shaped (`window_holdout_frac`); the sparse-anchor policy leaves the
  window unbounded
  only below two anchors; an unavailable calibration emits an unbounded window and
  a NaN prediction rather than a spurious value.

**Sidecar invocation contract.** `sidecar.rs` is a set of thin subprocess
clients over a positional-CLI file contract: write an input Parquet, run
`python <script> <args...>`, read an output Parquet keyed by id
(`sidecar.rs:1-4`). `resolve_script` (`sidecar.rs:20`) finds a worker by trying
the configured dir relative to CWD, then relative to the binary's own directory,
then `<exe_dir>/scripts`, so a deployed binary locates its workers regardless of
CWD. `python::resolve_script_dir` (`python.rs:372`) makes the same decision one
level up, before any worker runs: it accepts the configured directory only if it
actually holds a worker file, and otherwise tries the config file's own directory
and the executable's directory, which is the layout of the release archive.
`run_worker` (`sidecar.rs:233`) sets `PYTHONUTF8=1` and
`PYTHONIOENCODING=utf-8` only when its `utf8` argument is set (the DeepLC calls)
because Keras crashes on the Windows cp1252 console otherwise
(`sidecar.rs:240`). Worker output is inherited rather than captured, on purpose:
the NN rescorer prints per-iteration progress across hours and a user watching a
run needs to see it live. The cost is that on failure the traceback has already
scrolled past, so the error itself names the interpreter, the script, and the full
argument list, and points at `mumdia doctor` (`sidecar.rs:263-270`). Do not
capture the output to improve the message; extend the message instead.

**Doctor.** `mumdia doctor` (`main.rs:438`) answers whether a configuration can
actually run, and is the one command to run before a long job. It reports four
things in order.

1. The worker scripts. `predict_frag.sidecar_script_dir` is resolved through
   `python::resolve_script_dir` (`python.rs:372`), and every worker file each
   needed role runs must exist. This check is skipped entirely when no role is
   required, so a native configuration is not failed over a directory it never
   opens. The previous version failed exactly that case, and a missing script
   directory was the most common misconfiguration.
2. One line per interpreter role, for all four roles (`Role::Rescore`,
   `DeepLc`, `Ms2pip`, `Mbr`; `python.rs:35`). `mbr.python` was previously never
   probed. A role that is neither required by the config nor explicitly named
   prints `[skip]` and is not probed at all. An `"auto"` field is resolved by
   `python::discover` (`python.rs:245`) exactly as a run would resolve it, and the
   line states the provenance (configured, which environment variable, the
   activated environment, or `PATH`).
3. The versions of the packages whose version changes results (`deeplc`, `torch`,
   `mokapot`, `ms2pip`, `numpy`), and a warning when DeepLC is below the 4.1.1
   floor (`main.rs:566`, `version_below` at `main.rs:611`, which reads a
   pre-release such as `4.0.0a2` as 4.0.0 so it stays below the floor).
4. A verdict. `doctor` exits non-zero if any required role is unusable
   (`main.rs:599-603`); a configured-but-unneeded role that cannot import prints
   `warn` rather than `FAIL`.

The required-module list per role is what the workers actually import, not what
their dependency trees imply (`python.rs:71-85`): `nn_torch` needs
`torch,numpy,pandas,pyarrow`, mokapot and entrapment need
`mokapot,sklearn,numpy,pandas,pyarrow`, DeepLC needs
`deeplc,numpy,pandas,pyarrow,torch,psm_utils` (longer than `deeplc_worker.py`'s
own imports because the same interpreter runs `deeplc_finetune.py`), and MBR needs
`numpy,pyarrow`. The probe still runs `importlib.util.find_spec` in the candidate
interpreter (`python.rs:196-200`), so it answers importability only and cannot see
an import-order fault of the kind described under **Build gotchas** below. The
container CI job covers that case for the image by importing DeepLC for real in
the worker's order.

**Docker image.** Two-stage build. Stage 1 (`Dockerfile:25-30`) builds the
release binary on `rust:1.96-bookworm` with `cargo build --release --locked
--bin mumdia`. Stage 2 (`Dockerfile:33`) is
`mambaorg/micromamba:1.5.10-bookworm-slim`; it switches to `USER root`
(`Dockerfile:34`) and sets `MAMBA_ROOT_PREFIX=/opt/conda` (`Dockerfile:35`), which
is why the baked configs point at `/opt/conda/envs/<env>/bin/python`. It installs
`build-essential` for any sdist-only pip dependency (`Dockerfile:40-41`), then
creates two conda envs from `env/docker-rescore.yml` and `env/docker-deeplc.yml`
and runs `micromamba clean -a -y` (`Dockerfile:44-48`). `git` is no longer
installed: DeepLC is pinned to a PyPI version rather than a repository commit, so
nothing in the build clones anything. It copies the binary to
`/usr/local/bin/mumdia`, `scripts/` to `/opt/mumdia/scripts`, and both Docker
configs to `/opt/mumdia/config.dia.json` and `/opt/mumdia/config.diann-lib.json`
(`Dockerfile:51-54`), sets `MUMDIA_RESCORE_MODEL=logreg` (`Dockerfile:57`),
declares the standard OCI labels (title, description, source, licenses, vendor;
`Dockerfile:61-65`), sets the working directory to `/data` (`Dockerfile:67`, which
is the bind-mount point in the usage example), drops back to the base image's
unprivileged user (`USER $MAMBA_USER`, `Dockerfile:78`), and sets
`ENTRYPOINT ["mumdia"]` with `CMD ["--help"]` (`Dockerfile:79-80`).

Running unprivileged is deliberate and has a documented consequence: the container
user's uid does not match the host user's, so a bind mount the engine must write
to needs the host uid and gid passed with docker's `--user` flag. Without it even
the mount point fails with `mkdir: cannot create directory '/data': Permission
denied` (`Dockerfile:12-15`, measured 2026-08-27). Nothing inside the image needs
to be writable at run time, because everything MuMDIA writes lands under
`--out-dir`.

The two conda envs both pin `python=3.11` on purpose: mokapot and MS2PIP pull
`pandas<2`, which has no cp312 wheel and would force a fragile source build
(`docker-rescore.yml:8-9`), and DeepLC 4.1.1 itself requires Python >= 3.11
(`docker-deeplc.yml:11`). The `rescore` env anchors only the two tools
(`mokapot==0.10.0`, `ms2pip==4.0.0.dev9`) plus `numpy<2`/`pyarrow`/`scikit-learn`,
leaving their scientific-Python graph to pip (`docker-rescore.yml:16-21`). The
`deeplc` env installs `torch==2.12.1+cpu` from the PyTorch CPU index-url plus
`deeplc==4.1.1` and `pyarrow` (`docker-deeplc.yml:18-22`); it no longer caps
`numpy<2`, which 4.1.1 does not require, and the multitask model weight ships
inside the DeepLC package, so nothing is downloaded at run time.

**CI workflow.** `ci.yml` triggers on push to `main` and on every pull request
(`ci.yml:3-6`), with a `concurrency` group per ref and `cancel-in-progress: true`
(`ci.yml:8-10`) so a newer push supersedes an in-flight run on the same ref. It
runs five jobs, and together they are the full gate `CLAUDE.md` states plus the
end-to-end and non-Rust checks:

- `lint` (`ci.yml:16-43`), one `ubuntu-latest` runner under
  `working-directory: rust/mumdia`: `cargo fmt --check` (`ci.yml:37-38`) and
  `cargo clippy --workspace --all-targets --locked -- -D warnings`
  (`ci.yml:42-43`). `--all-targets` is deliberate, so dead code reachable only
  from a test module still fails. Formatting and clippy were previously a local
  responsibility and `main` could carry a tree that failed them; when this job was
  added it immediately caught two unformatted hunks and two dead functions on a
  branch treated as ready.
- `build-test` (`ci.yml:45-74`): `cargo build --release --locked` then
  `cargo test --workspace --locked` on `ubuntu-latest`, `macos-latest` and
  `windows-latest` with `fail-fast: false`. `--workspace` is explicit because the
  release binary is only one of the three members. `--locked` forces the committed
  `Cargo.lock`, so lockfile drift fails CI.
- `smoke` (`ci.yml:82-116`), on `ubuntu-latest` and `windows-latest` with Python
  3.12 plus `pyarrow`: build `--bin mumdia`, then `ci/smoke.sh` under bash on both
  platforms (`ci.yml:114-116`). This is the only end-to-end coverage there is. The
  Rust integration test builds its inputs in process and starts at `extract`, so
  mzML parsing, the `digest -> peptidoforms -> predict-frag` library build, the
  `run` orchestrator, `manifest.json`, RT calibration on real anchors, and `quant`
  and `report` writing files were all untested before it. The fixture is
  generated rather than committed: `ci/make_fixture_mzml.py` builds the mzML from
  `test_data/fixture.fasta` and from the library the engine itself derives from
  it, so the planted peaks cannot disagree with the engine's own mass model, and
  the repository still contains no mzML or Parquet. It uses
  `configs/examples/native.json`, so it needs no sidecar and no network.
- `sidecars` (`ci.yml:118-164`), one `ubuntu-latest` runner with Python 3.12:
  `python -m compileall -q scripts ci` (`ci.yml:132-133`), a JSON parse of every
  tracked `*.json` from `git ls-files` (`ci.yml:139-146`), a YAML parse of every
  tracked `env/*.yml` and `docker/*.yml` (`ci.yml:150-158`), and
  `python ci/check_doc_refs.py` (`ci.yml:163-164`), which fails when a tracked
  file cites a Markdown document the repository does not ship.

All three cargo jobs cache `~/.cargo/registry`, `~/.cargo/git` and
`rust/mumdia/target` keyed on `Cargo.lock` with a per-OS restore-key prefix
(`ci.yml:27-35`, `ci.yml:58-66`, `ci.yml:99-107`); the lint job uses a distinct
key prefix so its artifacts do not collide with the matrix builds, and `smoke`
shares the `build-test` key deliberately, since it builds the same binary.

What CI still does not do: run the Python sidecars. `compileall` is a syntax
floor, not validation. Real DeepLC, mokapot, `nn_torch`, entrapment and MBR
behaviour is exercised nowhere in CI except inside the Docker job's `doctor` and
import checks, and the smoke job is deliberately native-only. `pytest` coverage
of the worker contracts is still outstanding
(`docs/22_release_plan.md`, WP6). The JSON parse is likewise a syntax floor; the
real config check is the Rust test `shipped_configs_parse` (`config.rs:1651`),
which loads every shipped config under `deny_unknown_fields`.

**Release workflow.** `release.yml` fires only on `v*` tags (`release.yml:13-15`)
and declares `permissions: contents: write` so it can attach archives
(`release.yml:17-18`). It builds four targets (`release.yml:26-39`):
`x86_64-unknown-linux-musl` on `ubuntu-latest` (with a `musl-tools` install step,
`release.yml:46-48`), `aarch64-apple-darwin` on `macos-latest`,
`x86_64-apple-darwin` on `macos-13`, and `x86_64-pc-windows-msvc` on
`windows-latest`, each with `fail-fast: false`. Every target then:

1. smoke-tests its own binary on its own architecture (`release.yml:58-65`):
   `--version`, `--help` and `doctor` must all succeed. A binary whose `--help` or
   `doctor` is broken fails the release instead of shipping;
2. stages a working installation rather than a bare executable
   (`release.yml:67-84`): the binary, `README.md`, `LICENSE`, `CHANGELOG.md`,
   `docs/`, `scripts/`, `env/`, and `configs/` when the tag carries it. The reason
   is stated in the workflow header (`release.yml:8-12`): the ML predictors and
   rescorers are Python sidecars the engine launches by path, so a binary alone
   cannot fine-tune retention times or rescore with mokapot or the NN;
3. archives it (`7z a` to `.zip` on Windows, `tar czf` to `.tar.gz` elsewhere,
   `release.yml:86-92`) and writes a sha256 sidecar using whichever of
   `sha256sum` or `shasum -a 256` the runner provides (`release.yml:95-99`);
4. prints the archive tree and the checksum into the job log
   (`release.yml:100-103`), then uploads `*.tar.gz`, `*.zip` and `*.sha256` to the
   Release with `generate_release_notes: true` (`release.yml:105-112`).

**Docker workflow.** `docker.yml` declares
`permissions: contents: read, packages: write` (`docker.yml:19-21`), runs on `v*`
tags and on manual `workflow_dispatch` (`docker.yml:14-17`), and builds
`linux/amd64` only (`docker.yml:57`), so the published image is amd64. The order
matters: it builds once with `load: true, push: false` into the local daemon
(`docker.yml:52-63`) so the smoke test runs against exactly the image that will be
pushed, and only then pushes. The GHCR login (`docker.yml:31-37`) and the push
step (`docker.yml:100-110`) are both gated on
`startsWith(github.ref, 'refs/tags/v')`, so a manual dispatch validates the image
without publishing; a tag pushes `ghcr.io/compomics/mumdia:<tag>` plus `:latest`
plus `:sha` (`docker.yml:39-47`). Buildx uses the GitHub Actions layer cache
(`cache-from: type=gha`, `cache-to: type=gha,mode=max`).

The smoke test (`docker.yml:65-98`) is the point of this workflow outside a
release, because the image is the only distribution where the sidecars are
guaranteed present and nothing else in CI installs DeepLC, mokapot or MS2PIP. It
asserts, in order: `--version` and `--help` succeed; the runtime uid is not 0, so
a regression back to root fails the build; `doctor` passes on both baked configs;
the `deeplc` environment imports `deeplc, numpy, pyarrow, torch, psm_utils` for
real and in the worker's own order, which is the only check in the repository that
can catch the torch-DLL-after-numpy failure, because a module-presence probe
cannot see it; the `rescore` environment imports
`mokapot, sklearn, numpy, pandas, pyarrow`; and the documented invocation writes
into a bind mount as the host user, passing the host uid and gid through docker's
`--user` flag. That last assertion exists because if it breaks the image is
unusable for its primary purpose regardless of what the other checks say.

**Dependabot.** `.github/dependabot.yml` keeps Cargo (`/rust/mumdia`), GitHub
Actions (`/`) and the Docker base images (`/`) current, monthly and grouped, with
`arrow*`/`parquet*` in their own group because they carry the on-disk contract and
an unreviewed bump is a data-format change rather than a routine upgrade. Every bump
still has to pass the full gate, and a dependency change must commit the updated
`Cargo.lock` in the same pull request because both CI and the Docker build use
`--locked`.

**The Python side has no Dependabot coverage, by construction.** The sidecar pins
live in the `pip:` sections of the conda specifications under `env/`, and
Dependabot's pip ecosystem reads `requirements.txt`, `pyproject.toml`, `Pipfile` or
`poetry.lock`. A mirror requirements file added to satisfy the scanner would be a
second list that nothing installs and that drifts from the one that does. What
covers the gap instead is `pip-audit` in the `sidecar-imports` job, which audits the
RESOLVED environment (so transitive packages no specification names are included).
It is strict on the weekly scheduled run and advisory on a pull request, because an
advisory lands independently of any pull request and a strict gate would fail
changes that did not cause it. The same job uploads `pip freeze --all` per
environment as a 90-day artifact, which is the only record of what a given build
actually installed. The permanent fix is `scripts/pyproject.toml` (docs/22, WP4).

**Base images are pinned by digest.** A tag is mutable, so `rust:1.96-bookworm` is
repointed by its publisher and a rebuild of the same commit can produce a different
image. Both `FROM` lines carry `@sha256:...` with the tag kept for readability, and
the Dependabot `docker` entry is what stops a digest pin from quietly becoming an
unpatched base.

**Two generated inventories, for two readers.** `THIRD_PARTY_LICENSES.md` is the
notice document: 173 crates with SPDX expressions, the licence texts, and the actual
per-crate copyright lines and NOTICE files, which is what MIT, BSD and Apache-2.0
section 4(d) require to travel with a distributed binary. `sbom.cdx.json` is the
machine inventory: the same 173 components as CycloneDX 1.5 with purls and the
dependency graph, which is what a vulnerability scanner or an institutional software
inventory consumes. Both are generated from the same lockfile, both are checked for
staleness in CI, and both ship in the release archive and the image. The SBOM carries
no timestamp or serial number on purpose: either would change on every regeneration
and defeat the staleness check.

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `Config::validate` | `config.rs:1434` | Rejects footgun combinations at load and warns on inert ones; defaults always pass |
| `Config::apply_profile` | `config.rs:1587` | Applies the `dia` preset (Extended features, apex window 5, RT prior 120 s) |
| `Config::canonical_json` | `config.rs:1605` | Canonical config JSON for the manifest hash |
| `doctor` | `main.rs:438` | Reports whether the config can run (scripts, interpreters, versions); exits non-zero if a required role is unusable |
| `version_below` | `main.rs:611` | Dotted-version comparison for the DeepLC floor; reads `4.0.0a2` as 4.0.0 |
| `Cli::log_filter` | `main.rs:55` | Maps `--log-level`/`-v`/`-vv`/`-q` onto a tracing filter, or `None` to leave `RUST_LOG` in charge |
| `apply_threads` | `main.rs:74` | Builds the global rayon pool from `--threads` and exports `MUMDIA_NN_THREADS`/`OMP_NUM_THREADS` when unset |
| `python::Role` | `python.rs:35` | The four sidecar interpreter slots, each with its config field, env var, workers and required modules |
| `python::discover` | `python.rs:245` | `"auto"` interpreter discovery: role env var, `MUMDIA_PYTHON`, `CONDA_PREFIX`/`VIRTUAL_ENV`, `PATH`; accepts only if the modules import |
| `python::resolve` | `python.rs:280` | Resolves every needed role before the config hash is taken, so the manifest records the interpreter that ran |
| `python::resolve_script_dir` | `python.rs:372` | Picks the worker directory: configured, then the config file's directory, then the exe's directory |
| `sidecar::resolve_script` | `sidecar.rs:20` | CWD/exe-dir/`scripts` path resolution for a single worker |
| `sidecar::run_worker` | `sidecar.rs:222` | Spawns `python <script> <args>`; forces UTF-8 I/O only when its `utf8` arg is set (DeepLC calls) |
| `sidecar::run_deeplc_finetune` | `sidecar.rs:111` | Positional-CLI DeepLC multitask fine-tune; a known nondeterministic path (no torch/numpy seed) |
| `make_decoy` | `digest.rs:119` | Reverse or seeded-scramble decoy generation |
| `collision_safe_decoy` | `digest.rs:155` | Retries scrambles until the decoy collides with no target/other decoy; drops the pair if none exists |
| `splitmix64` | `digest.rs:177` | Deterministic PRNG for the scramble Fisher-Yates shuffle |
| `fnv1a` | `digest.rs:185` | Per-peptide seed hash so the scramble is peptide-stable |
| `tmp` | `tests/pipeline.rs:10` | Per-process unique temp path for concurrent tests |

## Configuration

This area reads very little config directly; it mostly enforces config validity
and wires interpreters. The config surface was recently pruned of dead fields, so
do not reintroduce removed knobs. The fields relevant here:

- `Config::validate` (`config.rs:1434`) rejects twelve combinations, all of which
  the committed defaults pass. Six are strategy, gate or quantification footguns:
  `digest.decoy.strategy = diann_shift` (threaded but realized nowhere; would
  yield zero decoys and an invalid FDR, `config.rs:1436`);
  `rt_im_train.calibration_method = none` (silently falls through to the linear
  fit, `config.rs:1445`); `extract.retain_top_peaks = 0` (must be >= 1, where 1 is
  the legacy single-apex behaviour, `config.rs:1452`); `extract.gate_min_score`
  outside `[0,1]` or non-finite (0 disables the gate, `config.rs:1459`);
  `quant.fixed_window_s` non-finite or negative (`config.rs:1468`); and
  `quant.baseline_quantile` non-finite or outside its range
  (`config.rs:1475`). Six guard the rescore contract: `rescore.folds < 2` (every
  PSM needs an out-of-fold score, `config.rs:1482`); `rescore.num_iter = 0`
  (`config.rs:1489`); `rescore.train_fdr` non-finite or outside `(0,1]`
  (`config.rs:1494`); `mokapot`/`nn_torch` without `rescore.python`
  (`config.rs:1502`); `percolator` (declared but not wired, `config.rs:1512`); and
  `entrapment` without `rescore.entrapment_marker` (`config.rs:1519`).
- `validate` also warns without failing in four cases (`config.rs:1528-1578`),
  which is how the inert and mutually-exclusive fields announce themselves rather
  than silently doing nothing: a non-default `mbr.rt_window_s`,
  `mbr.decoy_transfer` or `mbr.requant_all` (all three marked NOT YET WIRED at
  `config.rs:1207,1210,1217`); a non-`none` `mbr.strategy`, whose tiers are not
  distinguished; `quant.fixed_window_s` and `quant.fixed_scan_halfwidth` both set,
  where the seconds form wins; and `quant.baseline_subtract` without a fixed
  window to subtract inside.
- `apply_profile("dia")` (`config.rs:1587`) sets `features.set = Extended`,
  `extract.apex_count_window = 5`, `extract.apex_rt_prior_s = 120.0`. Any other
  name is an error.
- Deserialization uses `#[serde(deny_unknown_fields)]` throughout `config.rs` (20
  occurrences), so an unknown or misspelled key is a hard load error rather than a
  silently-ignored no-op (tested at `config.rs:1722`). This is also what makes
  `shipped_configs_parse` (`config.rs:1651`) a real check on every config the
  repository ships, and why the CI JSON-parse step is only a syntax floor beneath
  it.
- Sidecar interpreters are not read from the config verbatim. `python::resolve`
  (`python.rs:280`) rewrites each needed role's field to a concrete path before
  the config hash is taken, so `manifest.json` records the interpreter that
  actually ran rather than the word `auto`. That makes the config hash
  machine-specific for an `"auto"` config, which is correct: two runs whose
  rescorer came from different environments are not the same configuration.
- The Docker default rescorer model is selected by the env var
  `MUMDIA_RESCORE_MODEL=logreg` baked at `Dockerfile:57`; the NN worker also reads
  `MUMDIA_NN_SEEDS` (`scripts/nn_rescore_worker.py:42`).
- Both `docker/*.json` configs open with the same DIA preset written out
  explicitly (`features.set = extended`, `extract.apex_count_window = 5`,
  `extract.apex_rt_prior_s = 120.0`), matching `apply_profile("dia")`.
- `docker/config.dia.json` (FASTA-digest path) sets
  `predict_frag.predictor = ms2pip` and `predict_frag.rt_predictor = deeplc`, and
  points sidecars at fixed in-image interpreters:
  `/opt/conda/envs/rescore/bin/python` for MS2PIP + mokapot and
  `/opt/conda/envs/deeplc/bin/python` for DeepLC, with
  `sidecar_script_dir = /opt/mumdia/scripts` (`docker/config.dia.json:4-15`); the
  rescorer is `mokapot`. Absolute paths rather than `"auto"` is deliberate here:
  the interpreters in the image are at fixed locations, and an explicit path is
  never second-guessed.
- `docker/config.diann-lib.json` (library-input path) does NOT set
  `predictor`/`rt_predictor` (fragment intensities + iRT come from the imported
  library), configures only `deeplc_python` + `sidecar_script_dir` for the
  optional iRT fine-tune, and additionally sets `rt_im_train.finetune_deeplc =
  true` and `rt_im_train.rt_window_multiplier = 1.5`. It selects `nn_torch` using
  the torch-capable `/opt/conda/envs/deeplc/bin/python` and sets
  `rescore.strict = true`.
- `configs/examples/{native,fasta-sidecars,diann-library}.json` are the portable
  counterparts of the same three workflows, all using `"auto"` and a relative
  `sidecar_script_dir`. `native.json` names no interpreter at all and needs no
  Python. `configs/README.md` documents the resolution order for users.
- The pip pins in the two in-image envs are exact and reproducibility-load-bearing:
  `rescore` = `mokapot==0.10.0` + `ms2pip==4.0.0.dev9` + `numpy<2` (rest via pip,
  `docker-rescore.yml:16-21`); `deeplc` = `torch==2.12.1+cpu` + `deeplc==4.1.1`
  from PyPI + `pyarrow` (`docker-deeplc.yml:18-22`), with no `numpy<2` cap, which
  4.1.1 does not require. The two host envs differ:
  `env/mumdia-deeplc.yml` mirrors the image's DeepLC pins and adds `psm-utils`
  explicitly because `deeplc_finetune.py` imports it directly, while
  `env/mumdia-rescore.yml` is a deliberately different, minimal pin set
  (`python=3.12` with conda `numpy`/`pandas`/`pyarrow`/`scikit-learn` and pip
  `mokapot`, unpinned, and no `pandas<2` constraint because it installs neither
  MS2PIP nor DeepLC). `env/mumdia-rescore.yml` therefore has no torch and cannot
  serve the `nn_torch` rescorer; that role needs the DeepLC environment or another
  torch-capable interpreter.

## Invariants, determinism, gotchas

**Determinism.** The native engine must be byte-reproducible across runs.
Concretely: no `rand` crate is used anywhere; the only randomness is the decoy
scramble, which is a hand-rolled `splitmix64` (`digest.rs:177`) seeded per peptide
by `seed ^ fnv1a(peptide)` (`digest.rs:135`) so it does not depend on iteration
order. Float summation order is fixed: a HashMap-order f32 sum once shifted an
apex and broke reproducibility, so ordered maps / sorted iteration are used where
floats are summed (for example the digest stats use a `BTreeMap`,
`digest.rs:301`). The integration test asserts this directly by comparing
`apex_rt` across two extract runs (`tests/pipeline.rs:211`).

**Known nondeterministic paths (opt-in only).** Two paths are explicitly
nondeterministic and are never on the default path. The DeepLC multitask
fine-tune (`scripts/deeplc_finetune.py`, invoked by `sidecar::run_deeplc_finetune`
at `sidecar.rs:111`) sets no torch/numpy seed. The PyTorch NN rescorer
(`scripts/nn_rescore_worker.py`) is only approximately reproducible; its own
header says so (`nn_rescore_worker.py:29-30`) and offers `MUMDIA_NN_SEEDS>1` to
ensemble seeds and average out-of-fold scores as mitigation. Enabling either
breaks byte-reproducibility of the run; that is a known and accepted trade for the
identification gain.

**Clean-room boundary.** No coefficient vector, intensity model, or constant table
is copied from another proteomics engine. Physical constants are public-domain
facts derived from CODATA / AME atomic masses, with the provenance stated in the
file header (`constants.rs:1-6`); the proton mass is deliberately the physically
correct value, not DIA-NN's H-atom value (`constants.rs:8-10`). Decoys use a
documented reverse/scramble scheme (`digest.rs:117-147`), not an imported scheme.
DIA-NN itself is never vendored
(1.8.1+ / 2.x are proprietary); the user runs DIA-NN under their own license and
imports the predicted library via `scripts/import_diann_lib.py` +
`scripts/make_reverse_decoys.py`.

**Build gotchas (do not "fix" these back).** (1) Keep `target-dir` off any
cloud-synced tree; the symptom of getting this wrong is intermittent compiler
access-violation crashes, not a clean error. (2) Keep the pure-Rust dep features
(`parquet` no-default + `snap`, `mzdata` `miniz_oxide`); adding a C-backed codec
reintroduces a cmake/C toolchain requirement, and note the read-side consequence
above (SNAPPY-only input). (3) Keep `arrow-ipc` at
`opt-level = 1`; raising it crashes rustc codegen on Windows. (4) Keep the
toolchain at the pinned `1.96.1`, and keep the declared `rust-version` tracking
it. The edition-2024 dependencies need at least 1.85, but 1.96.1 is the only
version CI builds and tests, so a lower floor is unverified; lowering it needs an
MSRV job that proves it. (5) `cargo test` runs tests concurrently in one process, so any new
test that writes files must use a unique path per call (see `tmp`), not a shared
fixed name. (6) Do not sort the imports in `scripts/deeplc_worker.py` or
`scripts/deeplc_finetune.py`. `import deeplc` must run before numpy and pyarrow:
DeepLC 4.x is torch-backed, and on Windows importing numpy first aborts torch's
DLL initialisation with `OSError: [WinError 1114] ... Error loading
"...\torch\lib\c10.dll"`. Neither the Rust test suite nor `mumdia doctor` can
catch this, and imported-library mode never reaches `deeplc_worker.py` at all, so
the only thing that exercises it is a FASTA-mode library build.

**Test-coverage gaps (what green does NOT prove).** Both the `cargo test` suite
and the smoke job exercise native paths only. Inside `cargo test` there is still
no stage-level test for `convert`, `search-seed`, `predict-frag`, the `run`
orchestrator, `manifest.json`, `inspect`, or the library-input path; `ci/smoke.sh`
now covers all of those except the library-input path, but it runs outside
`cargo test`, so a green `cargo test --workspace` alone still proves none of them.
The real sidecar strategies (MS2PIP, DeepLC, DeepLC fine-tune, mokapot, the
PyTorch NN, entrapment, MBR) never run in the test suite or in CI; only the native
fallbacks are covered. (Percolator is not a gap but a rejection: `validate`
refuses it, `config.rs:1512`.) This gap has already shipped one real defect: a
module-level import reordering in `scripts/deeplc_worker.py` made every
FASTA-mode DeepLC prediction abort on Windows, and a green workspace suite plus a
green `mumdia doctor` both reported no problem. Treat any sidecar change as
validated only by running it. MS1 extraction and mass-calibration paths are
exercised only in full runs, not unit tests. There is no multi-run coverage (align, MBR,
quant-lfq cross-run beyond the single-file unit tests) and no entrapment-rescorer
coverage. `--locked` in CI means a stale `Cargo.lock` fails the build, so update
the lockfile in the same commit as a dependency bump.

What CI added in the release work narrows the gap without closing it: the `smoke`
job runs the whole native chain on a generated fixture, `compileall` catches a
syntax error in a worker, the JSON and YAML parses catch a malformed config or env
spec, `check_doc_refs.py` catches a citation to a document a clone does not
receive, and the Docker job's real DeepLC import catches the import-order fault.
None of that runs a Python worker on data. The outstanding item is `pytest`
coverage of the worker contracts (`docs/22_release_plan.md`, WP6).

**Docker gotchas.** The two conda envs are pinned to Python 3.11 on purpose (see
above); bumping to 3.12 reintroduces a source build of `pandas<2`.
`build-essential` in the runtime stage is required, not incidental
(`Dockerfile:40-41`); `git` was removed with the DeepLC PyPI pin and must not come
back with a git-installed dependency. The image must keep running as a non-root
user: the Docker smoke test fails the build if the runtime uid is 0
(`docker.yml:72-75`), and the consequence for users is that a writable bind mount
needs the host uid and gid passed through docker's `--user` flag. A
`workflow_dispatch` of `docker.yml` builds and smoke-tests but does not push; only
a `v*` tag publishes. The published image is `linux/amd64` only
(`docker.yml:57`), so on arm64 hosts (Apple Silicon) it runs under emulation. Both
standard configs are inside the image. A custom config can still be mounted, and
the mzML/library/FASTA inputs always need a data mount.

## How to extend / modify

- **Add a dependency:** add it once to `[workspace.dependencies]`
  (`Cargo.toml:25`), reference it with `x.workspace = true` in the crate manifest,
  prefer pure-Rust features, and commit the updated `Cargo.lock` in the same
  change so `--locked` CI stays green.
- **Add a validation gate:** extend `Config::validate` (`config.rs:1434`) with a
  branch that only fires on a non-default value, and add a rejection test next to
  the existing ones (`config.rs:1748`). Do not make defaults fail validation. If
  the field is accepted but not yet read by any stage, warn instead of rejecting
  and say so in the doc comment, as the inert MBR fields do
  (`config.rs:1528-1578`).
- **Add a stage-level test:** craft fixtures with `mumdia_io::table::write_table`
  following the schemas above, use `tmp()` (`tests/pipeline.rs:10`) for unique
  paths, and assert both correctness and cross-run equality to lock in
  determinism. Prefer covering one of the untested stages listed above.
- **Add a Python sidecar:** put the worker in `scripts/`, add its client to
  `sidecar.rs` over the positional-CLI Parquet contract, add it to an existing
  `python::Role` or add a new one (`python.rs:35`) listing every package the worker
  actually imports (not what its dependency tree implies) and every worker file
  `doctor` should look for, pin its packages in a new `env/*.yml`, and add it to
  the Docker envs if it should ship in the image. Write its output Parquet with
  SNAPPY and arrow `utf8` columns. If it imports a torch-backed package, import
  that package before numpy. If it trains a model, either seed it for determinism
  or document it as nondeterministic like the DeepLC/NN paths.
- **Cut a release:** push a `v*` tag. That triggers both `release.yml` (four
  per-target archives plus checksums on the GitHub Release, each binary
  smoke-tested first) and `docker.yml` (build, smoke test, then GHCR push of
  `<tag>` + `latest` + `sha`). To validate the image without publishing, run
  `docker.yml` via `workflow_dispatch`. To add a release target, extend the
  `release.yml` matrix (`release.yml:26-39`) and add any required cross-compile
  tooling step; note that the per-target smoke test runs the binary on its own
  runner, so a cross-compiled target with no matching runner cannot be
  smoke-tested there. Neither workflow has ever executed on a GitHub runner: no
  `v*` tag has been pushed, so the checks are verified locally and on a Linux host
  but the YAML that drives them is not (`docs/22_release_plan.md`, section 2b).
- **Change the Docker default rescorer:** edit the baked `docker/config.dia.json`
  (`rescore.classifier`) and/or the `MUMDIA_RESCORE_MODEL` env in `Dockerfile:57`.

## `set -e` plus `cmd && echo` in a command substitution exits silently

An orchestration script that collects a run list like this looks harmless and is not:

```bash
set -euo pipefail
RUNS=($(for d in "$OUT"/p*/; do
          r=$(basename "$d")
          [ -f "$d/psms_competed.parquet" ] && echo "$r"     # WRONG under set -e
        done | sort))
```

The test returns false for any run that has not completed. If the LAST directory in the glob is
one of those, the subshell's final command fails, the command substitution returns non-zero, and
`set -e` terminates the script. Nothing is written to stderr, so a redirected log contains only
`nohup: ignoring input` and the exit status is 1 with no diagnosis. Observed in production after a
batch left five of eighty-three runs incomplete, the last of them being the final glob entry.

Use an explicit conditional, which never leaves a failing command last:

```bash
          if [ -f "$d/psms_competed.parquet" ]; then echo "$r"; fi
```

`bash -x <script>` is the fastest way to find this class of failure: the trace stops exactly where
the shell gives up, whereas the log shows nothing at all.
