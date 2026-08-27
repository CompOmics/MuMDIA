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
the conda specs in `env/`, the container definition in `Dockerfile` +
`docker/`, and the CI/release/Docker automation in `.github/workflows/`.

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
| `.github/workflows/ci.yml` | Build + test matrix on ubuntu/macos/windows, `--locked`; on push-to-`main` + every PR |
| `.github/workflows/release.yml` | Dormant until a `v*` tag; builds per-platform binaries and attaches archives to the Release |
| `.github/workflows/docker.yml` | Builds the image; pushes to GHCR only on a `v*` tag, build-only on `workflow_dispatch` |
| `Dockerfile` | Two-stage image: Rust build stage + micromamba runtime with two sidecar envs |
| `docker/config.dia.json` | Baked FASTA-digest config (MS2PIP + DeepLC + strict mokapot wired to in-image envs) |
| `docker/config.diann-lib.json` | Baked library-input config (DeepLC fine-tune + strict NnTorch through the torch-capable DeepLC env) |
| `env/docker-rescore.yml` | Conda spec for the in-image `rescore` env (`python=3.11`, `mokapot==0.10.0` + `ms2pip==4.0.0.dev9`) |
| `env/docker-deeplc.yml` | Conda spec for the in-image `deeplc` env (`python=3.11`, `torch==2.12.1+cpu` + DeepLC 4.0 multitask pinned git commit) |
| `env/mumdia-rescore.yml` | Minimal host env for the default mokapot rescorer only (`python=3.12`, no torch/DeepLC/MS2PIP) |
| `rust/mumdia/crates/mumdia/src/sidecar.rs` | Sidecar subprocess clients + `resolve_script` path resolution |
| `rust/mumdia/crates/mumdia/src/main.rs` | Thin CLI; `doctor` subcommand probes configured sidecar interpreters (`main.rs:346`) |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `Config::validate` (`config.rs:1019`) + `apply_profile` (`config.rs:1107`) |
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
(`Dockerfile:21`, target dir forced by `ENV CARGO_TARGET_DIR=/build` at
`Dockerfile:18`); the release workflow produces `mumdia-<tag>-<target>.{tar.gz,zip}`
archives (`release.yml:50-60`) containing the binary plus `README.md` and
`LICENSE`. Non-Windows targets ship a `.tar.gz` (`tar czf`, `release.yml:59`);
the Windows target ships a `.zip` (`7z a`, `release.yml:57`). Both
`docker/*.json` configs are copied into the image as
`/opt/mumdia/config.dia.json` and `/opt/mumdia/config.diann-lib.json`. Input
mzML, FASTA, and library Parquets remain user data and must be mounted.

## How it works

**Build configuration.** The workspace declares three members
(`Cargo.toml:3`) sharing one `[workspace.package]` (`Cargo.toml:5-10`): version
`0.1.0`, `edition = "2021"`, `rust-version = "1.85"`, license `Apache-2.0`. The
crates are edition 2021, but several transitive dependencies are edition 2024,
which is why the minimum compiler is 1.85; `rust-toolchain.toml` pins a
known-good stable `1.96.1` above that floor so every machine and CI runner uses
the same rustc.

Every dependency is version-anchored once in `[workspace.dependencies]`
(`Cargo.toml:12-26`) and each crate re-exports it with `x.workspace = true` (for
example `crates/mumdia/Cargo.toml:11-25`). The workspace uses the v2 feature
resolver (`resolver = "2"`, `Cargo.toml:2`) and lists three members
(`Cargo.toml:3`). Each crate declares only the subset it needs: `mumdia-core`
depends on `serde`/`serde_json`/`thiserror` only (no arrow/parquet, so the core
types stay I/O-free); `mumdia-io` adds `arrow`/`parquet`/`blake3` over
`mumdia-core`; the `mumdia` bin+lib crate pulls the full set including `clap`,
`rayon`, `mzdata`, and `tracing`. The engine is a **pure-Rust build with
no C toolchain**: `parquet` uses `default-features = false, features =
["arrow","snap"]` to drop the C zlib-ng backend that needs cmake and to use the
pure-Rust SNAPPY codec (`Cargo.toml:23-24`); `mzdata` uses `["mzml",
"miniz_oxide"]` for a pure-Rust zlib backend (`Cargo.toml:25-26`); `arrow` adds
`["prettyprint"]` for `inspect` (`Cargo.toml:22`).

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

The release profile is `opt-level = 3` (`Cargo.toml:28-29`) with one exception:
`arrow-ipc` is pinned to `opt-level = 1` (`Cargo.toml:34-35`). Its generated
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
(`Dockerfile:18`) so the copied source cannot pick up a stray machine path.

**Test suite.** `cargo test --workspace` runs 126 `#[test]` functions across the
workspace. 124 are inline `#[cfg(test)]` unit tests co-located with their code; 2
are the integration tests in `tests/pipeline.rs`. The full local validation gate
per `CLAUDE.md` is `cargo fmt --check`, `cargo clippy --workspace --all-targets --
-D warnings`, and `cargo test --workspace` (plus `cargo build --release
--locked`); note that CI itself runs only build + test (see the CI workflow
below), so `fmt`/`clippy -D warnings` are a local/pre-commit responsibility.
Coverage by area (test counts are `#[test]` occurrences per file):

- `mumdia-core`: `mass.rs` (6), `rejection.rs` (5), `config.rs` (6),
  `constants.rs` (2). Config tests cover default round-trip
  (`config.rs:1134`), unknown-key rejection (`config.rs:1145`), partial override
  keeping defaults (`config.rs:1151`), `quant.q_filter` parse/serialize
  (`config.rs:1161`), the invalid-rescore-contract branches (`config.rs:1171`),
  and the uncapped-seed-versus-invalid-gate distinction (`config.rs:1180`).
- `mumdia-io`: `table.rs` (1).
- `mumdia`: `quant.rs` (13), `peaks.rs` (9), `compete.rs` (8), `peptidoforms.rs`
  (7), `fragindex.rs` (6), `quant_lfq.rs` (5), `extract.rs` (5),
  `chromatographic.rs` (5), `digest.rs` (5), `fdr.rs` (4), `features.rs` (4),
  `apex_dispersion.rs` (4), `mass_uncertainty.rs` (4), `calibrate.rs` (3),
  `binning.rs` (3), `rescore.rs` (3), `rt_im_train.rs` (3), `order_consistency.rs`
  (3), `stats.rs` (2), `main.rs` (2), `audit.rs` (2), `index.rs` (1),
  `rescoring.rs` (1), `search_seed.rs` (1), `report.rs` (1). (`rescoring.rs` is
  the native-rescorer scoring kernel; `stages/rescore.rs` is the rescore-stage
  orchestration and coverage checks.)

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
  by-mass agree; N-term/C-term mods; low-information fragments dropped;
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
  from an invalid gate (`extract.min_frag_corr` outside `[0,1]`, rejected).
- `constants.rs`: the three `within_ppm` forms agree; the inclusive edge case.

_`mumdia-io`:_ `table.rs` round-trips a mixed-column `Table` through Parquet
(the only I/O-layer unit test).

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
  collapses to a single sample.
- `compete.rs`: `winner_take_all` keeps only the winner; `none`/`features_only`
  keep all; `margin_gated` keeps close losers and removes distant ones;
  `unique_evidence` keeps losers with enough unique evidence and falls back to
  winner-take-all without evidence data; `unique_evidence` prefers the Extended
  `peak_contested` fraction when it is present; winner-take-all is deterministic
  across groups; the winner tie-breaks to the smallest index.
- `matchers/fragindex.rs`: fragindex-vs-naive equivalence; two fragments hitting
  one peak count both; the epoch counter resets with no carry across scans;
  tolerance edges (inside matches, outside does not); the precursor-window gate
  excludes out-of-range; the +/-1 bin probe finds within-tolerance pairs across
  bin boundaries.
- `matchers/binning.rs`: within-tolerance pairs are at most one bin apart;
  out-of-range clamps; a boundary-straddling pair is covered by the probe.
- `quant_lfq.rs`: a single sample is the column sum; recovers a known profile
  (complete and with missing values); total intensity preserved; a disconnected
  sample falls back to its own sum.
- `stages/extract.rs`: co-eluting fragments score high; a non-co-eluting
  interferent drops the score; too few scans does not reject; peak-spectral is
  high when the integrated pattern matches and recovers a fragment absent at the
  apex scan.
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
- `stages/digest.rs`: Trypsin/P cleaves after K/R; the reverse decoy keeps the
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
- `stats.rs`: Pearson (perfect and flat) and cosine/angle.
- `main.rs`: the conversion cap defaults to uncapped and an explicit cap is
  preserved (CLI parsing).
- `index.rs`: page search finds only in-window and in-tolerance hits.
- `search_seed.rs`: zero selects all and the seed cap keeps only the top-intensity
  peaks.
- `rescoring.rs`: the native rescorer separates targets from decoys.
- `stages/rescore.rs`: the competed feature schema must match on id and ordered
  columns; sidecar scores must give exact, unique, finite coverage of every row;
  an exact picked-peptide score tie is won by the decoy (conservative FDR).
- `stages/rt_im_train.rs`: anchor vectors are sorted by `base_peptide_id`
  (deterministic fit input); the sparse-anchor policy leaves the window unbounded
  only below two anchors; an unavailable calibration emits an unbounded window and
  a NaN prediction rather than a spurious value.
- `report.rs`: mod-stripping and decoy handling for the TSV report.

**Sidecar invocation contract.** `sidecar.rs` is a set of thin subprocess
clients over a positional-CLI file contract: write an input Parquet, run
`python <script> <args...>`, read an output Parquet keyed by id
(`sidecar.rs:1-4`). `resolve_script` (`sidecar.rs:20`) finds a worker by trying
the configured dir relative to CWD, then relative to the binary's own directory,
then `<exe_dir>/scripts`, so a deployed binary locates its workers regardless of
CWD. `run_worker` (`sidecar.rs:217`) sets `PYTHONUTF8=1` and
`PYTHONIOENCODING=utf-8` only when its `utf8` argument is set (the DeepLC calls)
because Keras crashes on the Windows cp1252 console otherwise
(`sidecar.rs:223-224`).

**Doctor.** `mumdia doctor` (`main.rs:346`) probes three interpreters
(`main.rs:358-377`) with an inline `importlib.util.find_spec` check and reports
`[ ok ]` / `[FAIL]` / `[skip]` per interpreter; `[skip]` means the interpreter is
not configured, so the native path is used (`main.rs:381`). The rescore
interpreter's required packages depend on the selected classifier: `NnTorch`
needs `torch,numpy,pandas,pyarrow`; mokapot and entrapment (the `_` arm) need
`mokapot,sklearn,numpy,pandas,pyarrow` (`main.rs:351-357`). The other two are
fixed: `predict_frag.deeplc_python` needs
`deeplc,numpy,pandas,pyarrow,torch,psm_utils` (`main.rs:368-371`) and
`predict_frag.ms2pip_python` needs `ms2pip,numpy,pandas` (`main.rs:372-376`). The
DeepLC list is longer than that script's own imports on purpose: the same
interpreter also runs `deeplc_finetune.py`, which imports pyarrow, torch and
psm_utils. The probe asserts what the scripts import rather than what the
dependency tree implies, so a green `doctor` no longer precedes a crash at the
fine-tune step. It exits non-zero if any configured env is unusable
(`main.rs:409-411`). `find_spec` only answers whether a module is importable, so
`doctor` cannot detect an import-order fault of the kind described under
**Build gotchas** below.

**Docker image.** Two-stage build. Stage 1 (`Dockerfile:15-21`) builds the
release binary on `rust:1.96-bookworm` with `cargo build --release --locked
--bin mumdia`. Stage 2 (`Dockerfile:24`) is `mambaorg/micromamba:1.5.10-bookworm-slim`;
it switches to `USER root` (`Dockerfile:25`) and sets
`MAMBA_ROOT_PREFIX=/opt/conda` (`Dockerfile:26`), which is why the baked configs
point at `/opt/conda/envs/<env>/bin/python`. It installs `git` +
`build-essential` (git for the pinned DeepLC commit, build-essential for any
sdist-only pip dep), then creates two conda envs from `env/docker-rescore.yml` and
`env/docker-deeplc.yml` and runs `micromamba clean -a -y` (`Dockerfile:30-38`). It
copies the binary to `/usr/local/bin/mumdia`, `scripts/` to `/opt/mumdia/scripts`,
and both Docker configs to `/opt/mumdia/config.dia.json` and
`/opt/mumdia/config.diann-lib.json`, sets `MUMDIA_RESCORE_MODEL=logreg`
(`Dockerfile:47`), sets the container working directory to `/data`
(`WORKDIR /data`, `Dockerfile:49`, which is the bind-mount point in the usage
example), and sets `ENTRYPOINT ["mumdia"]` with `CMD ["--help"]`
(`Dockerfile:50-51`). The two conda envs both pin `python=3.11` on purpose:
mokapot/MS2PIP/DeepLC pull `pandas<2`, which has no cp312 wheel and would force a
fragile source build (`docker-rescore.yml:8-9`, `docker-deeplc.yml:8-9`). The
`rescore` env anchors only the two tools (`mokapot==0.10.0`,
`ms2pip==4.0.0.dev9`) plus `numpy<2`/`pyarrow`/`scikit-learn`, leaving their
scientific-Python graph to pip (`docker-rescore.yml:16-21`). The `deeplc` env
installs `torch==2.12.1+cpu` from the PyTorch CPU index-url, `numpy<2`/`pyarrow`,
and DeepLC from a pinned git commit (`5c6a94e3...`); the multitask model weight
ships inside the DeepLC package, so no separate download is needed
(`docker-deeplc.yml:4-5`, `16-21`).

**CI / release / docker workflows.** `ci.yml` triggers on push to `main` and on
every pull request (`ci.yml:3-6`), with a `concurrency` group per ref and
`cancel-in-progress: true` (`ci.yml:8-10`) so a newer push supersedes an in-flight
run on the same ref. It runs a `build + test` matrix on `ubuntu-latest`,
`macos-latest`, `windows-latest` with `fail-fast: false` (`ci.yml:16-19`), all
under `working-directory: rust/mumdia` (`ci.yml:20-22`). It caches the cargo
registry, git, and `rust/mumdia/target` keyed on `Cargo.lock`, with a
`${{ runner.os }}-cargo-` prefix restore-key fallback (`ci.yml:28-36`), then
`cargo build --release --locked` and `cargo test --locked` (`ci.yml:38-42`).
`--locked` forces the committed `Cargo.lock`, so a lockfile drift fails CI.
`release.yml` (`release.yml:8-9`) fires only on `v*` tags, declares
`permissions: contents: write` so it can attach archives (`release.yml:11-12`),
and builds three targets: `x86_64-unknown-linux-musl` (with `musl-tools`),
`aarch64-apple-darwin`, `x86_64-pc-windows-msvc` (`release.yml:20-30`), packaging
each with the binary + README + LICENSE and uploading to the Release with
`generate_release_notes: true` (`release.yml:47-68`). `docker.yml` declares
`permissions: packages: write` (`docker.yml:11-13`), builds `linux/amd64` only
(`docker.yml:46`, so the published image is amd64), and uses the GitHub Actions
buildx layer cache (`cache-from/cache-to: type=gha`, `docker.yml:50-51`). It runs
on `v*` tags and on manual `workflow_dispatch`, but the GHCR login and `push:` are
both gated on `startsWith(github.ref, 'refs/tags/v')` (`docker.yml:24`,
`docker.yml:47`), so a manual dispatch validates the build without publishing; a
tag pushes `ghcr.io/compomics/mumdia:<tag>` + `:latest` + `:sha`
(`docker.yml:36-39`).

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `Config::validate` | `config.rs:1019` | Rejects footgun combinations at load; defaults always pass |
| `Config::apply_profile` | `config.rs:1107` | Applies the `dia` preset (Extended features, apex window 5, RT prior 120 s) |
| `Config::canonical_json` | `config.rs:1125` | Canonical config JSON for the manifest hash |
| `doctor` | `main.rs:346` | Probes configured sidecar interpreters; exits non-zero if any is unusable |
| `sidecar::resolve_script` | `sidecar.rs:20` | CWD/exe-dir/`scripts` path resolution for workers |
| `sidecar::run_worker` | `sidecar.rs:217` | Spawns `python <script> <args>`; forces UTF-8 I/O only when its `utf8` arg is set (DeepLC calls) |
| `sidecar::run_deeplc_finetune` | `sidecar.rs:111` | Positional-CLI DeepLC multitask fine-tune; a known nondeterministic path (no torch/numpy seed) |
| `make_decoy` | `digest.rs:101` | Reverse or seeded-scramble decoy generation |
| `collision_safe_decoy` | `digest.rs:137` | Retries scrambles until the decoy collides with no target/other decoy; drops the pair if none exists |
| `splitmix64` | `digest.rs:159` | Deterministic PRNG for the scramble Fisher-Yates shuffle |
| `fnv1a` | `digest.rs:167` | Per-peptide seed hash so the scramble is peptide-stable |
| `tmp` | `tests/pipeline.rs:10` | Per-process unique temp path for concurrent tests |

## Configuration

This area reads very little config directly; it mostly enforces config validity
and wires interpreters. The config surface was recently pruned of dead fields, so
do not reintroduce removed knobs. The fields relevant here:

- `Config::validate` (`config.rs:1026`) rejects ten combinations, all of which
  the committed defaults pass. Four are strategy/gate footguns:
  `digest.decoy.strategy = diann_shift` (threaded but realized nowhere; would
  yield zero decoys and an invalid FDR, `config.rs:1021`);
  `rt_im_train.calibration_method = none` (silently falls through to the linear
  fit, `config.rs:1030`); `extract.retain_top_peaks = 0` (must be >= 1, where 1 is
  the legacy single-apex behaviour, `config.rs:1037`); and `extract.min_frag_corr`
  outside `[0,1]` or non-finite (0 disables the gate, `config.rs:1044`). Six guard
  the rescore contract: `rescore.folds < 2` (every PSM needs an out-of-fold score,
  `config.rs:1053`); `rescore.num_iter = 0` (`config.rs:1060`); `rescore.train_fdr`
  non-finite or outside `(0,1]` (`config.rs:1065`); `mokapot`/`nn_torch` without
  `rescore.python` (`config.rs:1073`); `percolator` (declared but not wired,
  `config.rs:1083`); and `entrapment` without `rescore.entrapment_marker`
  (`config.rs:1090`).
- `apply_profile("dia")` (`config.rs:1107`) sets `features.set = Extended`,
  `extract.apex_count_window = 5`, `extract.apex_rt_prior_s = 120.0`. Any other
  name is an error.
- Deserialization uses `#[serde(deny_unknown_fields)]` throughout `config.rs` (15
  occurrences), so an unknown or misspelled key is a hard load error rather than a
  silently-ignored no-op (tested at `config.rs:1145`).
- The Docker default rescorer model is selected by the env var
  `MUMDIA_RESCORE_MODEL=logreg` baked at `Dockerfile:47`; the NN worker also reads
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
  rescorer is `mokapot`.
- `docker/config.diann-lib.json` (library-input path) does NOT set
  `predictor`/`rt_predictor` (fragment intensities + iRT come from the imported
  library), configures only `deeplc_python` + `sidecar_script_dir` for the
  optional iRT fine-tune, and additionally sets `rt_im_train.finetune_deeplc =
  true` and `rt_im_train.rt_window_multiplier = 1.5`. It selects `nn_torch` using
  the torch-capable `/opt/conda/envs/deeplc/bin/python` and sets
  `rescore.strict = true`.
- The pip pins in the two in-image envs are exact and reproducibility-load-bearing:
  `rescore` = `mokapot==0.10.0` + `ms2pip==4.0.0.dev9` + `numpy<2` (rest via pip);
  `deeplc` = `torch==2.12.1+cpu` + DeepLC pinned git commit `5c6a94e3...` + `numpy<2`.
  The host-only `env/mumdia-rescore.yml` is a different pin set: `python=3.12`
  with conda `numpy`/`pandas`/`pyarrow`/`scikit-learn` and pip `mokapot`
  (unpinned, and no `pandas<2` constraint because it installs neither MS2PIP nor
  DeepLC).

## Invariants, determinism, gotchas

**Determinism.** The native engine must be byte-reproducible across runs.
Concretely: no `rand` crate is used anywhere; the only randomness is the decoy
scramble, which is a hand-rolled `splitmix64` (`digest.rs:159`) seeded per peptide
by `seed ^ fnv1a(peptide)` (`digest.rs:117`) so it does not depend on iteration
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
documented reverse/scramble scheme (`digest.rs:99-129`), not an imported scheme.
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
toolchain at or above 1.85; the edition-2024 dependencies will not compile on
older rustc. (5) `cargo test` runs tests concurrently in one process, so any new
test that writes files must use a unique path per call (see `tmp`), not a shared
fixed name. (6) Do not sort the imports in `scripts/deeplc_worker.py` or
`scripts/deeplc_finetune.py`. `import deeplc` must run before numpy and pyarrow:
DeepLC 4.x is torch-backed, and on Windows importing numpy first aborts torch's
DLL initialisation with `OSError: [WinError 1114] ... Error loading
"...\torch\lib\c10.dll"`. Neither the Rust test suite nor `mumdia doctor` can
catch this, and imported-library mode never reaches `deeplc_worker.py` at all, so
the only thing that exercises it is a FASTA-mode library build.

**Test-coverage gaps (what green does NOT prove).** The suite exercises native
paths only. There is no stage-level test for `convert`, `search-seed`,
`predict-frag`, the `run` orchestrator, `manifest.json`, `inspect`, `report`, or
the library-input path. The real sidecar strategies (MS2PIP, DeepLC, DeepLC
fine-tune, mokapot, percolator) never run in the test suite; only the native
fallbacks are covered. This gap has already shipped one real defect: a
module-level import reordering in `scripts/deeplc_worker.py` made every
FASTA-mode DeepLC prediction abort on Windows, and a green workspace suite plus a
green `mumdia doctor` both reported no problem. Treat any sidecar change as
validated only by running it. MS1 extraction and mass-calibration paths are
exercised only in full runs, not unit tests. There is no multi-run coverage (align, MBR,
quant-lfq cross-run beyond the single-file unit tests) and no entrapment-rescorer
coverage. `--locked` in CI means a stale `Cargo.lock` fails the build, so update
the lockfile in the same commit as a dependency bump.

**Docker gotchas.** The two conda envs are pinned to Python 3.11 on purpose (see
above); bumping to 3.12 reintroduces a source build of `pandas<2`. `git` and
`build-essential` in the runtime stage are required, not incidental
(`Dockerfile:29`). A `workflow_dispatch` of `docker.yml` builds but does not push;
only a `v*` tag publishes. The published image is `linux/amd64` only
(`docker.yml:46`), so on arm64 hosts (Apple Silicon) it runs under emulation.
Both standard configs are inside the image. A custom config can still be mounted,
and the mzML/library/FASTA inputs always need a data mount.

## How to extend / modify

- **Add a dependency:** add it once to `[workspace.dependencies]`
  (`Cargo.toml:12`), reference it with `x.workspace = true` in the crate manifest,
  prefer pure-Rust features, and commit the updated `Cargo.lock` in the same
  change so `--locked` CI stays green.
- **Add a validation gate:** extend `Config::validate` (`config.rs:1019`) with a
  branch that only fires on a non-default value, and add a rejection test next to
  the existing ones (`config.rs:1171`). Do not make defaults fail validation.
- **Add a stage-level test:** craft fixtures with `mumdia_io::table::write_table`
  following the schemas above, use `tmp()` (`tests/pipeline.rs:10`) for unique
  paths, and assert both correctness and cross-run equality to lock in
  determinism. Prefer covering one of the untested stages listed above.
- **Add a Python sidecar:** put the worker in `scripts/`, add its client to
  `sidecar.rs` over the positional-CLI Parquet contract, add a `doctor` check row
  (`main.rs:358`) listing every package the worker actually imports (not what its
  dependency tree implies), pin its packages in a new `env/*.yml`, and add it to
  the Docker envs if it should ship in the image. Write its output Parquet with
  SNAPPY and arrow `utf8` columns. If it imports a torch-backed package, import
  that package before numpy. If it trains a model, either seed it for determinism
  or document it as nondeterministic like the DeepLC/NN paths.
- **Cut a release:** push a `v*` tag. That triggers both `release.yml` (per-platform
  archives on the GitHub Release) and `docker.yml` (GHCR push of `<tag>` + `latest`).
  To validate the image without publishing, run `docker.yml` via
  `workflow_dispatch`. To add a release target, extend the `release.yml` matrix
  (`release.yml:20-30`) and add any required cross-compile tooling step.
- **Change the Docker default rescorer:** edit the baked `docker/config.dia.json`
  (`rescore.classifier`) and/or the `MUMDIA_RESCORE_MODEL` env in `Dockerfile:47`.

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
