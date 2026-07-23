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
| `rust/mumdia/crates/mumdia/Cargo.toml` | The bin+lib crate `mumdia`; all deps come from `workspace.dependencies` |
| `rust/mumdia/crates/mumdia/tests/pipeline.rs` | The only integration test file: extract -> features -> compete -> rescore on crafted Parquet |
| `.github/workflows/ci.yml` | Build + test matrix on ubuntu/macos/windows, `--locked` |
| `.github/workflows/release.yml` | Dormant until a `v*` tag; builds per-platform binaries and attaches archives to the Release |
| `.github/workflows/docker.yml` | Builds the image; pushes to GHCR only on a `v*` tag, build-only on `workflow_dispatch` |
| `Dockerfile` | Two-stage image: Rust build stage + micromamba runtime with two sidecar envs |
| `docker/config.dia.json` | Baked FASTA-digest config (MS2PIP + DeepLC + mokapot wired to in-image envs) |
| `docker/config.diann-lib.json` | Baked library-input config (DeepLC fine-tune + mokapot) |
| `env/docker-rescore.yml` | Conda spec for the in-image `rescore` env (mokapot + MS2PIP) |
| `env/docker-deeplc.yml` | Conda spec for the in-image `deeplc` env (DeepLC 4.0 multitask + CPU torch) |
| `env/mumdia-rescore.yml` | Minimal host env for the default mokapot rescorer only (no torch/DeepLC/MS2PIP) |
| `rust/mumdia/crates/mumdia/src/sidecar.rs` | Sidecar subprocess clients + `resolve_script` path resolution |
| `rust/mumdia/crates/mumdia/src/main.rs` | Thin CLI; `doctor` subcommand probes configured sidecar interpreters (`main.rs:311`) |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `Config::validate` (`config.rs:1056`) + `apply_profile` (`config.rs:1098`) |
| `rust/mumdia/crates/mumdia-core/src/constants.rs` | Physical constants; the clean-room provenance note (`constants.rs:1-6`) |

## Inputs and outputs

The build/test/deploy area does not itself produce the engine's Parquet artifacts
(the stages do). The load-bearing "artifacts" here are the compiled binary, the
container image, release archives, and the small Parquet fixtures that the
integration test crafts by hand. The fixture schemas are worth documenting because
they are the minimal input contract that any new stage-level test must satisfy;
they are built in `tests/pipeline.rs`.

`lib_prec.parquet` (crafted at `tests/pipeline.rs:28`): `candidate_id:u32`,
`peptidoform_id:u32`, `base_peptide_id:u32`, `peptidoform:str`, `charge:i32`,
`precursor_mz:f64`, `predicted_irt:f32`, `label:str` (`target`/`decoy`),
`protein:str`, `n_fragments:i32`.

`lib_frag.parquet` (`tests/pipeline.rs:44`): `candidate_id:u32`, `mz:f64`,
`predicted_intensity:f32`, `name:str`, `ion_type:str`, `ordinal:i32`,
`frag_charge:i32`.

`ms2.parquet` (`tests/pipeline.rs:62`): `scan_index:u32`, `id:str`,
`rt_seconds:f64`, `window_id:u32`, `window_target:f64`, `window_lower:f64`,
`window_upper:f64`, `precursor_mz:OptF64`, `precursor_charge:OptI32`,
`mz:ListF32`, `intensity:ListF32`.

`windows.parquet` (`tests/pipeline.rs:104`): `candidate_id:u32`,
`rt_pred_cal:f64`, `rt_lo:f64`, `rt_hi:f64`, `im_pred_cal:OptF64`, `im_lo:OptF64`,
`im_hi:OptF64` (the IM columns are always null in the 3D MVP).

The build stage of the Docker image produces `/build/release/mumdia`
(`Dockerfile:20`, target dir forced by `ENV CARGO_TARGET_DIR=/build` at
`Dockerfile:17`); the release workflow produces `mumdia-<tag>-<target>.{tar.gz,zip}`
archives (`release.yml:50-60`) containing the binary plus `README.md` and
`LICENSE`.

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
example `crates/mumdia/Cargo.toml:11-25`). The engine is a **pure-Rust build with
no C toolchain**: `parquet` uses `default-features = false, features =
["arrow","snap"]` to drop the C zlib-ng backend that needs cmake and to use the
pure-Rust SNAPPY codec (`Cargo.toml:23-24`); `mzdata` uses `["mzml",
"miniz_oxide"]` for a pure-Rust zlib backend (`Cargo.toml:25-26`); `arrow` adds
`["prettyprint"]` for `inspect` (`Cargo.toml:22`).

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
(`Dockerfile:17`) so the copied source cannot pick up a stray machine path.

**Test suite.** `cargo test` runs 105 `#[test]` functions across the workspace.
103 are inline `#[cfg(test)]` unit tests co-located with their code; 2 are the
integration tests in `tests/pipeline.rs`. Coverage by area (test counts are
`#[test]` occurrences per file):

- `mumdia-core`: `mass.rs` (6), `rejection.rs` (5), `config.rs` (4),
  `constants.rs` (2). Config tests cover default round-trip
  (`config.rs:1126`), unknown-key rejection (`config.rs:1136`), partial override
  keeping defaults (`config.rs:1142`), and the gate-validation branches
  (`config.rs:1152`).
- `mumdia-io`: `table.rs` (1).
- `mumdia`: `peaks.rs` (9), `quant.rs` (9), `compete.rs` (7), `fragindex.rs` (6),
  `quant_lfq.rs` (5), `extract.rs` (5), `chromatographic.rs` (5), `fdr.rs` (4),
  `apex_dispersion.rs` (4), `mass_uncertainty.rs` (4), `calibrate.rs` (3),
  `binning.rs` (3), `features.rs` (3), `digest.rs` (3), `order_consistency.rs`
  (3), `stats.rs` (2), `main.rs` (2), `audit.rs` (2), `index.rs` (1),
  `rescoring.rs` (1), `search_seed.rs` (1), `report.rs` (1), `peptidoforms.rs`
  (2).

The integration test `extract_recovers_planted_target_and_is_deterministic`
(`tests/pipeline.rs:144`) plants a target with three fragments across five scans
and a decoy with no matching peaks, runs `extract` twice, and asserts the target
(candidate 0) is accepted, the decoy (1) is not, the two runs are byte-equal on
`candidate_id` and `apex_rt`, and the apex lands on the most-intense scan
(rt 140). `features_compete_rescore_run_on_crafted_input`
(`tests/pipeline.rs:167`) drives the rest of the chain on the same fixtures and
checks the PIN header and that a `q_value` column is produced. Both run the
**native** paths only (`Config::default()`), and `tmp()` (`tests/pipeline.rs:10`)
gives every crafted file a per-process, atomic-counter-unique path because cargo
runs tests concurrently in one process.

**Sidecar invocation contract.** `sidecar.rs` is a set of thin subprocess
clients over a positional-CLI file contract: write an input Parquet, run
`python <script> <args...>`, read an output Parquet keyed by id
(`sidecar.rs:1-4`). `resolve_script` (`sidecar.rs:18`) finds a worker by trying
the configured dir relative to CWD, then relative to the binary's own directory,
then `<exe_dir>/scripts`, so a deployed binary locates its workers regardless of
CWD. `run_worker` (`sidecar.rs:174`) sets `PYTHONUTF8=1` and
`PYTHONIOENCODING=utf-8` for the DeepLC calls because Keras crashes on the
Windows cp1252 console otherwise (`sidecar.rs:180-182`).

**Doctor.** `mumdia doctor` (`main.rs:311`) probes each configured sidecar
interpreter with an inline `importlib.util.find_spec` check and reports
`[ ok ]` / `[FAIL]` / `[skip]` per interpreter. Required packages depend on the
selected rescorer: `NnTorch` needs `torch,numpy,pandas,pyarrow`; mokapot and
entrapment need `mokapot,sklearn,numpy,pandas,pyarrow` (`main.rs:316-319`). It
exits non-zero if any configured env is unusable (`main.rs:356`).

**Docker image.** Two-stage build. Stage 1 (`Dockerfile:15-20`) builds the
release binary on `rust:1.96-bookworm` with `cargo build --release --locked
--bin mumdia`. Stage 2 (`Dockerfile:23`) is `mambaorg/micromamba:1.5.10-bookworm-slim`;
it installs `git` + `build-essential` (git for the pinned DeepLC commit,
build-essential for any sdist-only pip dep), then creates two conda envs from
`env/docker-rescore.yml` and `env/docker-deeplc.yml` and runs `micromamba clean
-a -y` (`Dockerfile:29-37`). It copies the binary to `/usr/local/bin/mumdia`,
`scripts/` to `/opt/mumdia/scripts`, and `docker/config.dia.json` to
`/opt/mumdia/config.dia.json` (`Dockerfile:40-42`), sets
`MUMDIA_RESCORE_MODEL=logreg` (`Dockerfile:45`), and sets `ENTRYPOINT ["mumdia"]`
with `CMD ["--help"]` (`Dockerfile:48-49`). The two conda envs both pin
`python=3.11` on purpose: mokapot/MS2PIP/DeepLC pull `pandas<2`, which has no
cp312 wheel and would force a fragile source build (`docker-rescore.yml:8-9`,
`docker-deeplc.yml:8-9`). Torch is CPU-only (`docker-deeplc.yml:17-18`) and
DeepLC is a pinned git commit (`docker-deeplc.yml:21`).

**CI / release / docker workflows.** `ci.yml` runs a `build + test` matrix on
`ubuntu-latest`, `macos-latest`, `windows-latest` with `fail-fast: false`
(`ci.yml:16-19`), all under `working-directory: rust/mumdia`
(`ci.yml:20-22`). It caches the cargo registry, git, and `rust/mumdia/target`
keyed on `Cargo.lock` (`ci.yml:28-36`), then `cargo build --release --locked`
and `cargo test --locked` (`ci.yml:38-42`). `--locked` forces the committed
`Cargo.lock`, so a lockfile drift fails CI. `release.yml` (`release.yml:8-9`)
fires only on `v*` tags and builds three targets:
`x86_64-unknown-linux-musl` (with `musl-tools`), `aarch64-apple-darwin`,
`x86_64-pc-windows-msvc` (`release.yml:20-30`), packaging each with the binary +
README + LICENSE and uploading to the Release (`release.yml:47-68`).
`docker.yml` builds the image on `v*` tags and on manual `workflow_dispatch`, but
`push:` is gated on `startsWith(github.ref, 'refs/tags/v')` (`docker.yml:47`), so
a manual dispatch validates the build without publishing; a tag pushes
`ghcr.io/compomics/mumdia:<tag>` + `:latest` + `:sha` (`docker.yml:36-38`).

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `Config::validate` | `config.rs:1056` | Rejects footgun combinations at load; defaults always pass |
| `Config::apply_profile` | `config.rs:1098` | Applies the `dia` preset (Extended features, apex window 5, RT prior 120 s) |
| `Config::canonical_json` | `config.rs:1116` | Canonical config JSON for the manifest hash |
| `doctor` | `main.rs:311` | Probes configured sidecar interpreters; exits non-zero if any is unusable |
| `sidecar::resolve_script` | `sidecar.rs:18` | CWD/exe-dir/`scripts` path resolution for workers |
| `sidecar::run_worker` | `sidecar.rs:174` | Spawns `python <script> <args>`; forces UTF-8 I/O for DeepLC |
| `make_decoy` | `digest.rs:92` | Reverse or seeded-scramble decoy generation |
| `splitmix64` | `digest.rs:122` | Deterministic PRNG for the scramble Fisher-Yates shuffle |
| `fnv1a` | `digest.rs:130` | Per-peptide seed hash so the scramble is peptide-stable |
| `tmp` | `tests/pipeline.rs:10` | Per-process unique temp path for concurrent tests |

## Configuration

This area reads very little config directly; it mostly enforces config validity
and wires interpreters. The config surface was recently pruned of dead fields, so
do not reintroduce removed knobs. The fields relevant here:

- `Config::validate` (`config.rs:1056`) rejects four combinations, all of which
  the committed defaults pass: `digest.decoy.strategy = diann_shift`
  (threaded but realized nowhere; would yield zero decoys and an invalid FDR,
  `config.rs:1058`); `rt_im_train.calibration_method = none` (silently falls
  through to the linear fit, `config.rs:1067`); `extract.retain_top_peaks = 0`
  (must be >= 1, where 1 is the legacy single-apex behaviour, `config.rs:1074`);
  and `extract.min_frag_corr` outside `[0,1]` or non-finite (0 disables the gate,
  `config.rs:1081`).
- `apply_profile("dia")` (`config.rs:1100`) sets `features.set = Extended`,
  `extract.apex_count_window = 5`, `extract.apex_rt_prior_s = 120.0`. Any other
  name is an error.
- Deserialization uses `#[serde(deny_unknown_fields)]` throughout `config.rs` (17
  occurrences), so an unknown or misspelled key is a hard load error rather than a
  silently-ignored no-op (tested at `config.rs:1136`).
- The Docker default rescorer model is selected by the env var
  `MUMDIA_RESCORE_MODEL=logreg` baked at `Dockerfile:45`; the NN worker also reads
  `MUMDIA_NN_SEEDS` (`scripts/nn_rescore_worker.py:42`).
- The baked configs point sidecars at fixed in-image interpreters:
  `/opt/conda/envs/rescore/bin/python` for MS2PIP + mokapot and
  `/opt/conda/envs/deeplc/bin/python` for DeepLC (`docker/config.dia.json:6-13`).
  `docker/config.diann-lib.json` additionally sets `rt_im_train.finetune_deeplc =
  true` and `rt_window_multiplier = 1.5`.

## Invariants, determinism, gotchas

**Determinism (PLAN.md Section 7).** The native engine must be byte-reproducible
across runs. Concretely: no `rand` crate is used anywhere; the only randomness is
the decoy scramble, which is a hand-rolled `splitmix64` (`digest.rs:122`) seeded
per peptide by `seed ^ fnv1a(peptide)` (`digest.rs:108`) so it does not depend on
iteration order. Float summation order is fixed: a HashMap-order f32 sum once
shifted an apex and broke reproducibility, so ordered maps / sorted iteration are
used where floats are summed (for example the digest stats use a `BTreeMap`,
`digest.rs:229`). The integration test asserts this directly by comparing
`apex_rt` across two extract runs (`tests/pipeline.rs:159-160`).

**Known nondeterministic paths (opt-in only).** Two paths are explicitly
nondeterministic and are never on the default path. The DeepLC multitask
fine-tune (`scripts/deeplc_finetune.py`, invoked by `sidecar::run_deeplc_finetune`
at `sidecar.rs:106`) sets no torch/numpy seed. The PyTorch NN rescorer
(`scripts/nn_rescore_worker.py`) is only approximately reproducible; its own
header says so (`nn_rescore_worker.py:29-30`) and offers `MUMDIA_NN_SEEDS>1` to
ensemble seeds and average out-of-fold scores as mitigation. Enabling either
breaks byte-reproducibility of the run; that is a known and accepted trade for the
identification gain.

**Clean-room boundary (PLAN.md Section 11).** No coefficient vector, intensity
model, or constant table is copied from another proteomics engine. Physical
constants are public-domain facts derived from CODATA / AME atomic masses, with
the provenance stated in the file header (`constants.rs:1-6`); the proton mass is
deliberately the physically correct value, not DIA-NN's H-atom value
(`constants.rs:8-10`). Decoys use a documented reverse/scramble scheme
(`digest.rs:90-120`), not an imported scheme. DIA-NN itself is never vendored
(1.8.1+ / 2.x are proprietary); the user runs DIA-NN under their own license and
imports the predicted library via `scripts/import_diann_lib.py` +
`scripts/make_reverse_decoys.py`.

**Build gotchas (do not "fix" these back).** (1) Keep `target-dir` off any
cloud-synced tree; the symptom of getting this wrong is intermittent compiler
access-violation crashes, not a clean error. (2) Keep the pure-Rust dep features
(`parquet` no-default + `snap`, `mzdata` `miniz_oxide`); adding a C-backed codec
reintroduces a cmake/C toolchain requirement. (3) Keep `arrow-ipc` at
`opt-level = 1`; raising it crashes rustc codegen on Windows. (4) Keep the
toolchain at or above 1.85; the edition-2024 dependencies will not compile on
older rustc. (5) `cargo test` runs tests concurrently in one process, so any new
test that writes files must use a unique path per call (see `tmp`), not a shared
fixed name.

**Test-coverage gaps (what green does NOT prove).** The suite exercises native
paths only. There is no stage-level test for `convert`, `search-seed`,
`predict-frag`, the `run` orchestrator, `manifest.json`, `inspect`, `report`, or
the library-input path. The real sidecar strategies (MS2PIP, DeepLC, DeepLC
fine-tune, mokapot, percolator) never run in the test suite; only the native
fallbacks are covered. MS1 extraction and mass-calibration paths are exercised
only in full runs, not unit tests. There is no multi-run coverage (align, MBR,
quant-lfq cross-run beyond the single-file unit tests) and no entrapment-rescorer
coverage. `--locked` in CI means a stale `Cargo.lock` fails the build, so update
the lockfile in the same commit as a dependency bump.

**Docker gotchas.** The two conda envs are pinned to Python 3.11 on purpose (see
above); bumping to 3.12 reintroduces a source build of `pandas<2`. `git` and
`build-essential` in the runtime stage are required, not incidental
(`Dockerfile:29`). A `workflow_dispatch` of `docker.yml` builds but does not push;
only a `v*` tag publishes.

## How to extend / modify

- **Add a dependency:** add it once to `[workspace.dependencies]`
  (`Cargo.toml:12`), reference it with `x.workspace = true` in the crate manifest,
  prefer pure-Rust features, and commit the updated `Cargo.lock` in the same
  change so `--locked` CI stays green.
- **Add a validation gate:** extend `Config::validate` (`config.rs:1056`) with a
  branch that only fires on a non-default value, and add a rejection test next to
  the existing ones (`config.rs:1152`). Do not make defaults fail validation.
- **Add a stage-level test:** craft fixtures with `mumdia_io::table::write_table`
  following the schemas above, use `tmp()` (`tests/pipeline.rs:10`) for unique
  paths, and assert both correctness and cross-run equality to lock in
  determinism. Prefer covering one of the untested stages listed above.
- **Add a Python sidecar:** put the worker in `scripts/`, add its client to
  `sidecar.rs` over the positional-CLI Parquet contract, add a `doctor` check row
  (`main.rs:320`), pin its packages in a new `env/*.yml`, and add it to the
  Docker envs if it should ship in the image. If it trains a model, either seed it
  for determinism or document it as nondeterministic like the DeepLC/NN paths.
- **Cut a release:** push a `v*` tag. That triggers both `release.yml` (per-platform
  archives on the GitHub Release) and `docker.yml` (GHCR push of `<tag>` + `latest`).
  To validate the image without publishing, run `docker.yml` via
  `workflow_dispatch`. To add a release target, extend the `release.yml` matrix
  (`release.yml:20-30`) and add any required cross-compile tooling step.
- **Change the Docker default rescorer:** edit the baked `docker/config.dia.json`
  (`rescore.classifier`) and/or the `MUMDIA_RESCORE_MODEL` env in `Dockerfile:45`.
