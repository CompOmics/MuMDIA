# Changelog

All notable changes to MuMDIA are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`0.1.0` will be the first tagged release of the Rust engine. The superseded
Python implementation remains available at the tag `legacy-python-v1`.

Two things are versioned independently of this file and matter when reading old
results: the per-artifact Parquet schema versions in
`rust/mumdia/crates/mumdia-core/src/schema.rs`, and the feature-set identity
`classifier_feature_schema_id`, which is a hash of the active feature list rather
than a number. Both are recorded in every run's `manifest.json`.

## [Unreleased]

### Added

- `quant.fragment_selection = predicted` ranks a precursor's fragments for the
  top-N sum by their library intensity instead of by their own integrated area.
  Ranking by observed area preferentially selects interfered fragments, because
  interference inflates the very quantity the ranking rewards, and the selected
  set then varies between runs.
- `quant.fixed_scan_halfwidth` and `quant.fixed_window_s` integrate a fixed
  window centred on the identification apex instead of the descent-walk bounds.
  The seconds form is instrument-independent and overrides the scan form. On the
  ProteoBench Astral HYE set these two options together moved median absolute
  epsilon from 0.273 to 0.195 and CV from 0.175 to 0.107.
- `quant.baseline_subtract`, with `baseline_flank_scans` and
  `baseline_quantile`, subtracts a flank-quantile background inside the fixed
  window.
- `prescan` stage: a native per-run sequence-tag prescan that prunes
  modification-bearing candidate hypotheses with no anchored tag support, 11.6
  times faster than the previous Python screen. Only modform hypotheses are ever
  pruned.
- `rt_im_train.window_holdout_frac` sizes the RT window from held-out anchors
  rather than from the anchors the calibration was fitted on. Benchmark-gated and
  off by default: it gained 1.1% of peptides with DeepLC 4.1.0 but lost 1.5% with
  the overfitting 4.0.0a2 model, so it interacts with RT-model quality.
- `env/mumdia-deeplc.yml`: a portable conda spec for the DeepLC sidecars. They
  previously had no committed local environment, so running them meant
  reconstructing a developer machine by hand.
- Sidecar interpreter discovery. A `python` field may be `"auto"` or absent, and
  the engine finds an interpreter from `MUMDIA_PYTHON_<ROLE>`, `MUMDIA_PYTHON`,
  `CONDA_PREFIX`, `VIRTUAL_ENV`, or `PATH`, accepting a candidate only after it
  imports what that role's workers import. A role is resolved only if the
  configuration uses it, so a default native run still needs no Python at all.
  Explicit paths behave exactly as before.
- Global CLI flags, accepted before or after the subcommand: `--threads N`
  bounds the engine's rayon pool and is forwarded to the sidecars as
  `MUMDIA_NN_THREADS` and `OMP_NUM_THREADS` when those are unset;
  `--log-level`, `-v`/`-vv` and `-q` set verbosity. Previously the only control
  was `RUST_LOG`, which is not discoverable from `--help`, and thread count could
  not be bounded at all: the engine never read `RAYON_NUM_THREADS`, so a run took
  every core on a shared machine.
- `configs/examples/{native,fasta-sidecars,diann-library}.json`, portable
  starting points that use `"auto"`, with `configs/README.md` explaining the
  resolution order and the environment specs. These replace the only tracked
  config, which named one developer's interpreters and OneDrive path and was the
  config the documentation told everyone to run.
- `ci/check_doc_refs.py`, run in CI: fails when a tracked file cites a Markdown
  document the repository does not ship.
- `CONTRIBUTING.md`, `SECURITY.md`, this changelog, and
  `docs/22_release_plan.md`.

All new quantification options default to off, so an existing configuration
produces bit-identical results.

### Changed

- CI now enforces the full stated gate: `cargo fmt --check` and
  `cargo clippy --workspace --all-targets -- -D warnings` in addition to the
  build and tests, plus `python -m compileall` over the sidecars, a JSON parse of
  every tracked configuration, a YAML parse of the environment specs, and the
  documentation-reference check. Formatting and clippy were previously a local
  responsibility, so `main` could carry a tree that failed them.
- The Docker DeepLC environment pins `deeplc==4.1.1` from PyPI instead of a git
  commit on the 4.0 multitask branch, and no longer caps `numpy<2`, which 4.1.1
  does not require. 4.1.1 is a floor, not merely the current release: the 4.0.0a2
  multitask preview overfits per-run fine-tuning badly enough to invert RT-model
  rankings. Verified by building the image and importing the workers' graph in
  the worker's own order: DeepLC 4.1.1, torch 2.12.1+cpu, numpy 2.4.6 in the
  `deeplc` environment and mokapot 0.10.0 in `rescore`, with `mumdia doctor`
  passing on both baked configurations.
- The image no longer runs as root after setup. It needs
  `--user "$(id -u):$(id -g)"` to write into a bind mount, which the documented
  invocation now passes and the Docker workflow now asserts.
- Under a fixed integration window, the reported `integration_lo_rt` and
  `integration_hi_rt` are now the retention-time extent actually integrated
  rather than the walked bounds that were ignored. Measured on the AIF benchmark
  run behind the ProteoBench submission: 72,168 quantified precursors, every
  `quantity`, `n_fragments_used`, `quant_status` and `integration_apex_rt`
  bit-identical, and the reported window corrected from a 29.1 s median (the
  descent walk) to 34.9 s (the fixed window that produced the numbers).
- `mumdia doctor` reports whether the configuration can actually run: the
  interpreter each role resolves to and how it was found, the versions of the
  packages whose version changes results, whether the worker scripts are where
  the engine will look, and a warning when DeepLC is older than 4.1.1. It now
  covers `mbr.python` and the script directory, neither of which it checked
  before, and it no longer fails a native configuration over a worker directory
  that configuration never opens.
- `predict_frag.sidecar_script_dir` is resolved against the config file's own
  directory and against the executable's directory, not only the current working
  directory. The same config invoked from elsewhere used to silently change which
  worker scripts ran.
- `run-experiment` warns when it overrides the configured `quant.q_filter` to
  gate per-run quantification on the pooled q value, instead of doing it
  silently.
- Dependabot keeps Cargo and GitHub Actions dependencies current, with
  `arrow`/`parquet` grouped separately because they carry the on-disk contract.

### Fixed

- MBR transfers are now quantified. The augmented scored table lowered only
  `q_value`, while quantification gates on `quant.q_filter`, which the experiment
  path sets to `run_psm_q`. An accepted transfer therefore kept a sub-threshold
  `run_psm_q` and was dropped: 34,280 of 34,664 transfers on a six-run HYE
  experiment, so match-between-runs appeared to run and changed almost nothing.
- `quant` reads `predicted_intensity` as an optional chromatogram column.
  Requiring it made every chromatogram artifact written before that column
  existed unquantifiable; the `predicted` ranking, its only consumer, now fails
  with an actionable message instead.
- `scripts/deeplc_worker.py` imports `deeplc` before `numpy` and `pyarrow`. The
  wrong order aborts torch DLL initialization on Windows with
  `OSError: [WinError 1114] ... c10.dll`. The failure was latent because
  imported-library mode skips the stage that reaches it.
- `.gitignore` covers `rust/mumdia/target` (the root-anchored `/target` never
  matched it), the experiment configurations that carry machine-specific
  interpreter paths, and the local benchmark data directories. It no longer
  matches `docs/22_release_plan.md`, which an unanchored `*_plan.md` rule had
  silently excluded from version control.
- Source comments no longer cite untracked local design notes. About 130
  references pointed at documents a clone does not receive; they now point at the
  tracked `docs/` guide.

### Known limitations

- The Python sidecars are not exercised by CI, and there is no checked-in fixture
  or end-to-end smoke test. A passing test run is not sidecar validation.
- A configuration names Python interpreters by absolute path, so it is not
  portable between machines without editing.
- `run` processes a single mzML file. `run-experiment` does not call the report
  stage, so it writes no `peptides.tsv` or `proteins.tsv`.
- `mbr.strategy` distinguishes only none from not-none; `rt_window_s`,
  `decoy_transfer` and `requant_all` are accepted but not wired.
- `extract.retain_top_peaks > 1` writes diagnostic peak alternatives that do not
  reach features or rescoring.
- No ion mobility support, mzML input only, and no wildcard or terminal variable
  modifications.
