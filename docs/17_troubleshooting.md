# Troubleshooting index: symptom -> cause -> fix
> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

A single lookup for the failure modes that are easy to hit but hard to diagnose,
gathered from docs 01-14 and `CLAUDE.md`. Many of these fail quietly (a silent
fallback, a nondeterministic wobble, a void result on a truncated file) rather
than with a clean error, so the table maps the observed symptom to the underlying
cause and the concrete fix. Code behaviour is cited to `file:line`; the referenced
source is the source of truth if it has moved.

## Lookup table

| Symptom | Cause | Fix |
|---|---|---|
| `cargo build`/`cargo test` crashes intermittently with `STATUS_ACCESS_VIOLATION` or `STATUS_ILLEGAL_INSTRUCTION`, not a clean error | Build target dir sits under the OneDrive-synced tree; sync churn locks and corrupts incremental artifacts | Redirect `build.target-dir` off OneDrive (or set `CARGO_TARGET_DIR`); see `rust/mumdia/.cargo/config.toml.example` |
| rustc crashes with `STATUS_ILLEGAL_INSTRUCTION` while compiling `arrow-ipc` at release opt-level 3 | `arrow-ipc`'s generated FlatBuffer code crashes rustc codegen at opt-level 3 on the Windows toolchain | Keep the `opt-level = 1` pin for `arrow-ipc` (`Cargo.toml:34-35`); do not remove it |
| Sidecar run fails with "worker not found" / a spawn error naming a script path | `predict_frag.sidecar_script_dir` does not resolve to an existing dir from any of the three candidates | Run from the project root, or set `sidecar_script_dir` to a native absolute path (`c:/...`, not a git-bash `/c/...` path) |
| `rescore.classifier = nn_torch` (or `mokapot`) produces `native_tda` scores | A compatibility config explicitly set `rescore.strict = false`, then worker resolution/execution failed and fallback ran | Keep the default `strict = true`; inspect `psms_scored.parquet.report.json` for the classifier actually used |
| `run` aborts with "MS2PIP returned no predictions" | MS2PIP returned an empty map; `predict-frag` has no whole-run native fallback | Fix the MS2PIP env (`mumdia doctor`), or set `predict_frag.predictor = native` |
| `run` aborts during the DeepLC predict / fine-tune / MBR step | `predict-frag`, the fine-tune, and MBR propagate worker errors with `?`; only `rescore` falls back | Fix the sidecar env, or disable that path (`rt_predictor = native`, `finetune_deeplc = false`, `mbr.strategy = none`) |
| A short `--max-spectra` run finds almost nothing | The flag reads the file head and the early gradient is void | Use the full file or externally prepare a mid-gradient mzML slice; `--max-spectra` has no offset |
| A reused `--out-dir` contains contradictory or stale sidecars | `run` recomputes named outputs but does not clear the directory; optional files can remain and a failed rerun can leave an old manifest | Use a fresh output directory for every `run`; reuse artifacts only through explicit standalone stage commands |
| Config load fails with an unknown-field parse error | `#[serde(deny_unknown_fields)]` rejects any unknown or misspelled key (including a removed knob) | Fix or remove the key; check it against the current `config.rs`, not an old config |
| Two runs of the same command give slightly different identification counts | A nondeterministic opt-in path is on: DeepLC fine-tune (no seed) or the PyTorch NN rescorer (approximately reproducible) | Accept the trade for the ID gain, or leave those paths off; set `MUMDIA_NN_SEEDS > 1` to average out NN variance |
| Peptidoforms silently get iRT 0.0 | DeepLC returned no iRT for some peptidoforms; they are anchored at 0.0 with a warning | Check the DeepLC env and the `n_irt_missing` warning; unmatched peptidoforms are a known foot-gun |

## Build failures

**rustc access violation / illegal instruction under OneDrive.** Building inside
the OneDrive-synced project tree causes the sync client to lock and corrupt
incremental build artifacts mid-compile, which surfaces as intermittent
`STATUS_ACCESS_VIOLATION` or `STATUS_ILLEGAL_INSTRUCTION` compiler crashes rather
than a clean error. The fix is to keep the build target dir off any cloud-synced
tree. On this machine `rust/mumdia/.cargo/config.toml` sets `build.target-dir` to
`C:/Users/robbi/mumdia_build`; that file is machine-specific and gitignored, and
the committed `rust/mumdia/.cargo/config.toml.example` documents the same fix plus
the `CARGO_TARGET_DIR` env-var alternative. A fresh clone with no local
`config.toml` builds into `./target` and works anywhere off a synced folder. Do
not "fix" the redirect back.

**arrow-ipc opt-level pin.** The release profile is `opt-level = 3` with one
exception: `arrow-ipc` is pinned to `opt-level = 1` (`Cargo.toml:34-35`). Its
generated FlatBuffer code reliably crashes rustc codegen at opt-level 3 on the
Windows toolchain (`STATUS_ILLEGAL_INSTRUCTION`). MuMDIA never uses Arrow IPC, so
lowering just that one crate's optimization avoids the crash at no runtime cost.
Do not raise it. Related: keep the pure-Rust dep features (`parquet` no-default +
`snap`, `mzdata` `miniz_oxide`, `Cargo.toml:23-26`); a C-backed codec reintroduces
a cmake/C toolchain requirement.

## Sidecar resolution and silent fallback

**"Worker not found" and the git-bash path trap.** `resolve_script`
(`sidecar.rs:20-38`) locates a worker by trying, in order: the configured dir
relative to the CWD (`sidecar.rs:21-22`), then the same dir relative to the
binary's own directory, then `<exe_dir>/scripts`. If none exist it returns the
CWD-relative path so the eventual spawn error names it (`sidecar.rs:37`). Because
the build target dir is redirected off OneDrive while `scripts/` lives under the
project, `<exe_dir>/scripts` does not exist next to the binary on this machine, so
resolution succeeds only via the CWD-relative branch. The specific trap: if
`predict_frag.sidecar_script_dir` is a git-bash-style POSIX path such as
`/c/Users/...`, `std::path::Path::exists` on Windows cannot stat it and returns
false, so resolution falls through to the non-existent path and the worker spawn
fails. Use a native Windows path (`c:/Users/...`) or a plain relative dir run from
the project root, or an absolute in-container path such as `/opt/mumdia/scripts`
in the Docker configs.

**A broken rescorer masked by an explicitly enabled native fallback.** `rescore`
is the only stage that can fall back to `native_tda` when its sidecar fails. With
`rescore.strict = false` (not the default), a failed or misconfigured
mokapot/nn_torch sidecar is logged
as a warning and the run continues on native scores (`rescore.rs:172-180` for
mokapot, `rescore.rs:199-207` for nn_torch). This means a worker that never
resolved (see the path trap above) or crashed produces a completed run whose
scores are silently native, not the classifier the config named. Set
`rescore.strict = true` so any sidecar failure is a hard error
(`rescore.rs:173`, `rescore.rs:200`); this is the production default. Confirm
`params.classifier` and `model_identity` in
`psms_scored.parquet.report.json` match what was intended.

## No fallback outside rescore

Only `rescore` has a native fallback. The predictor, fine-tune, and MBR stages
propagate worker errors with `?` and abort the whole run:

- **MS2PIP** hard-bails with `bail!("MS2PIP returned no predictions")` when the
  worker returns an empty map (`predict_frag.rs:342-344`); any other MS2PIP error
  propagates through the `?` at `predict_frag.rs:341`. There is no whole-run
  native fallback (a single candidate that MS2PIP returned nothing for does fall
  back per-candidate at `predict_frag.rs:387-389`, but an empty map aborts).
- **DeepLC predict** propagates worker errors with the `?` at
  `predict_frag.rs:291`. It does not bail on partial output: peptidoforms with no
  returned iRT are anchored at `0.0` with an `n_irt_missing` warning
  (`predict_frag.rs:293-308`), which is a known foot-gun rather than a stop.
- **DeepLC fine-tune** propagates errors with the `?` at `run.rs:253-263`, so a
  fine-tune failure aborts `run` with no fallback.
- **MBR** likewise propagates its worker error and is not on the `run` chain.

If any of these fails, fix the sidecar env (`mumdia doctor` probes the configured
interpreters) or switch that path back to native (`predict_frag.predictor =
native`, `predict_frag.rt_predictor = native`, `rt_im_train.finetune_deeplc =
false`, `mbr.strategy = none`).

## Truncated-run void

`--max-spectra N` on `convert` or `run` reads only the first N spectra across all
MS levels (`convert.rs:121`; the flag is documented at `docs/04_convert.md:245`).
The early chromatographic gradient is void, so a few thousand leading spectra
carry almost no identifiable peptides and a short cap will look like a broken
engine. For meaningful identifications use the full file, or externally create a
mid-gradient mzML slice and run that file. `--max-spectra` itself cannot seek to
an offset. This is expected behaviour, not a bug.

## Config load errors

Every config struct in `config.rs` carries `#[serde(deny_unknown_fields)]` (15
occurrences), and `Config::from_json` parses with unknown-key rejection before
validating (`config.rs:1009`). An unknown or misspelled key is therefore a hard
load error, not a silently-ignored no-op (tested at `config.rs:1146`). This bites
most often after the config surface is pruned: a knob that was removed, or renamed,
now fails the load. Check the offending key against the current `config.rs`, not
against an older config file. Separately, `Config::validate` (`config.rs:1019`)
rejects a few footgun combinations that would otherwise produce invalid results,
for example `digest.decoy.strategy = diann_shift` (`config.rs:1021`, zero decoys
and an invalid FDR) and `rt_im_train.calibration_method = none`
(`config.rs:1030`, silently falls through to the linear fit); the committed
defaults always pass.

## Run-to-run nondeterminism

The native engine is byte-reproducible; two opt-in paths are not, and neither is
on the default path:

- **DeepLC multitask fine-tune** (`scripts/deeplc_finetune.py`, invoked by
  `sidecar::run_deeplc_finetune` at `sidecar.rs:111`) sets no torch or numpy seed,
  so the rewritten iRT and every downstream artifact vary run to run.
- **PyTorch NN rescorer** (`scripts/nn_rescore_worker.py`) seeds torch and numpy
  per seed and uses a content-hash fold split, but training is only approximately
  reproducible; its own header says so (`nn_rescore_worker.py:29-30`). The
  `adam` solver plus BLAS threading drifts the scores slightly.

Enabling either breaks byte-reproducibility; that is a known and accepted trade
for the identification gain. To damp the NN variance, set `MUMDIA_NN_SEEDS > 1` to
ensemble seeds and average the out-of-fold scores
(`nn_rescore_worker.py:29-30`). If bit-exact reproducibility is required, leave
`finetune_deeplc = false` and use `native_tda` or the near-bit-exact mokapot
`logreg` path.
