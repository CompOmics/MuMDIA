# Changelog

All notable changes to MuMDIA are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`0.1.0` is the first tagged release of the Rust engine. The superseded
Python implementation remains available at the tag `legacy-python-v1`.

Two things are versioned independently of this file and matter when reading old
results: the per-artifact Parquet schema versions in
`rust/mumdia/crates/mumdia-core/src/schema.rs`, and the feature-set identity
`classifier_feature_schema_id`, which is a hash of the active feature list rather
than a number. Both are recorded in every run's `manifest.json`.

## [Unreleased]

## [0.1.0] - 2026-09-06

### Pre-release audit (2026-08-28)

A six-way audit of the tree ([`docs/25_release_readiness_review.md`](docs/25_release_readiness_review.md))
found three release blockers and about thirty further defects. All three blockers
and most of the rest are fixed; that document's status section records what is
deliberately still open. The entries below fold into the sections that follow.

**Second audit (2026-08-28, external).** A second review of the same tree raised
four further blockers, two of which were defects in the first audit's own output: a
generated document quoting a value from a test as though the engine set it
(`ci/gen_config_reference.py` did not skip `#[cfg(test)]`), and a `build.rs` that
guessed at the git directory and stamped a stale commit into every manifest. Both
fixed, the second verified across a commit. The other two were release mechanics and
are covered under Added below: a release archive that could not run the verification
its own documentation prescribes, and a `v*` tag that could publish any commit with
no CI behind it.

**Two changes require re-measurement rather than only review.**

- Three defaults changed on correctness grounds, not from a count:
  `extract.apex_evidence_rank` to `true` (the legacy apex silently selected the
  lowest-RT qualifying scan when none of the top-K predicted fragments was
  observed anywhere) and `features.emit_pin` to `false` (no stage reads the file;
  it is a ~5.4 GB write per run). `extract.gate_min_score` was also briefly changed
  to `0.6` and then measured back to `0.2`: 0.6 costs 4.4% of peptides for
  `native_tda` and 4.7% for `nn_torch` at an unchanged decoy fraction, because the
  gate sweep that motivated it predates the current defaults and its optimum has
  moved to the loose end for both rescorers.
- The `nn_torch` CV fold is keyed on `base_peptide_id` supplied by the engine
  rather than on a hash of the peptidoform, so a target and its paired decoy now
  share a fold as `percolator_lite` always did and as `docs/11` always claimed.

Both move `nn_torch` counts, and the entrapment fixes below invalidate any
entrapment measurement previously taken through the native rescorer.

**Breaking: interface renames with no compatibility aliases.** A tag freezes
these, so they are free now and a major bump later. CLI:
`--library-precursors`/`--library-fragments` to `--lib-`, `--out-chrom` to
`--out-chromatograms`, `--scored` to `--psms-scored`, `--out-scored` to
`--out-psms-scored`, `--psms` to `--psms-extracted`, and `--seed`/`--seeds` to
`--seed-psms`. Config: `extract.min_frag_corr` to `extract.gate_min_score` (it is
not a correlation under any `gate_mode`), and `compete.group_by = "precursor"` to
`"base_peptide"` (it keys on the stripped sequence). An old name now fails with
the offending key and the valid alternatives listed. Local configs need:

```bash
sed -i 's/"min_frag_corr"/"gate_min_score"/; s/"group_by": *"precursor"/"group_by": "base_peptide"/' config.*.json
```

### Added

- **Several files pool by default.** `run` given more than one `--mzml` dispatches to
  `run-experiment`: files provided together are rescored together (one pooled FDR),
  quantified per run and aligned across runs (MaxLFQ). Searching files separately is
  the opt-in, one `run` per file.
- **Experiment-wide report.** `run-experiment` writes `peptides.tsv` and `proteins.tsv`
  at the experiment root, selected on the experiment-wide `peptide_q_value` and
  `pg_q_value`, with an `n_runs` column (per-run acceptances on `run_psm_q`) and one
  `quantity_<run>` and `lfq_<run>` column per run. `mumdia report --experiment-dir`
  rewrites the pair at another threshold. No per-run TSVs are written, because the
  grouped q columns go to each group's experiment-wide winner only.
- **Vendor formats.** A vendor file given as `--mzml` is converted to mzML first
  (`convert`, `run`, `run-experiment`, `peak-census`): Thermo `.raw` through
  ThermoRawFileParser (or msconvert), Bruker and Agilent `.d`, SCIEX `.wiff` and
  Waters `.raw` through ProteoWizard `msconvert`. Converters are located
  (`convert.thermo_raw_parser`, `convert.msconvert`, both `auto`, or
  `MUMDIA_THERMO_PARSER` / `MUMDIA_MSCONVERT`), never shipped. The mzML is written
  beside the input (or into the output directory when that is not writable) through
  a `<name>.partial.mzML` temporary and reused on later runs when newer than its
  source (`convert.reuse_converted`). `mumdia doctor` reports both converters and does
  not fail for their absence. Only Thermo is exercised end to end (a 3.7 GB Astral
  run, 6:40 through ThermoRawFileParser 2.0.0); the four msconvert formats are wired
  and unverified, and ion mobility is discarded.
- **Library retention time from the DeepLC base model.** `rt_im_train.library_irt`
  (`auto`, `library`, `deeplc`) re-predicts an imported library's iRT with the DeepLC
  base model once per experiment when a DeepLC interpreter is configured, because the
  imported DIA-NN iRT is the worst RT source measured: AIF 10,416 peptides against
  10,015 raw; HYE B01 58,842 against 56,556 raw over three NN seeds. The optional
  fine-tune (`finetune_deeplc`) remains available and is still +2.4% on HYE.
- **DeepLC 4.1.1 is a floor.** `mumdia doctor`, the sidecar launch
  (`sidecar::require_deeplc_version`) and both worker scripts refuse an older DeepLC
  (`mumdia_core::constants::MIN_DEEPLC_VERSION`), because the default
  prediction-plus-calibration workflow is only sound on a base model that does not
  memorise its anchors.
- **Rescore handoff and training recipe.** `rescore.handoff = parquet` replaces the
  TSV handoff to the Python worker (rescore peak 29.96 to 8.95 GB, wall 8:35 to 6:33
  on the HYE competed table, identical identifications; mokapot still receives a PIN).
  The worker trains on the targets at 1% plus a capped, hybrid-selected decoy sample
  with warm refits (`train_neg_ratio 3`, `train_neg_select hybrid`,
  `train_warm_epochs 5`): HYE A01 +1.0%, HYE B01 +2.2%, AIF -0.1%, entrapment +3.3% at
  an unchanged spike-in FDP, at 9 to 19x less training time. `rescore.features` /
  `features_file` project the classifier's input columns; `feature_preset = compact`
  (114 features) is an opt-in memory lever, not a sensitivity one;
  `max_feature_matrix_gib` turns an oversized matrix into an error at startup;
  `MUMDIA_NN_SEED` sets the worker's base seed.
- **Memory footprint.** Streaming Parquet readers, incremental extract output flushed
  as isolation windows close (`extract.windows_in_flight`, auto, capped at 16), f32
  bulk arrays and a chunked features stage. HYE B01 single run: 231 GiB and 1:07:30
  before, 16.5 GiB and 17:52 after (compact preset; about 20:40 with every feature).
  Six pooled HYE runs rescored in 18 minutes at 15.9 GB against 4:34:42 at 40 GB.
- **N-terminal methionine excision** in the native digest
  (`digest.n_term_met_excision`, default on, matching DIA-NN `--met-excision`); old
  configurations still parse. `scripts/augment_library.py` uses the same digest to add
  the tryptic peptides an imported library is missing.
- **Imported libraries with empty protein cells load.** An empty `protein` (DIA-NN
  writes the iRT-kit standards that way) is grouped as `UNASSIGNED` with a warning
  that counts the rows; `scripts/import_diann_lib.py` writes the same group. An empty
  `peptidoform` is still an error.
- **Desktop application** (`desktop/`, "MuMDIA Console"): a Windows `.msi` and a Linux
  `.AppImage` built by the release workflow, bundling the engine and `uv`. It creates
  its own Python environment (no conda), installs ThermoRawFileParser on request,
  locates msconvert and DIA-NN, and rescores all files provided together by default.
  Its backend is unit-tested and both bundles were built and inspected; nobody has yet
  clicked through the interface end to end.

- The release archive verifies itself. It now carries `ci/smoke.sh`, its two helper
  scripts and `test_data/fixture.fasta`, and `release.yml` unpacks every archive it
  builds into a clean directory and runs that archive's own smoke test, on every
  target. `docs/19` told the reader to run exactly this while the archive shipped
  neither `ci/` nor `test_data/`; testing the artifact rather than the tree it came
  from is also the only check that can catch a packaging mistake, and it gives macOS
  its first end-to-end coverage.
- `validate-tag`, a release job every build depends on: the tag must equal the
  workspace version, the tagged commit must be an ancestor of `main`, and `ci.yml`
  must have a successful run for that exact SHA. A tag push does not trigger
  `ci.yml`, so the only checks behind a release were previously `--version`,
  `--help` and `doctor`.
- `run-experiment` coverage: `ci/smoke.sh` now runs the multi-run orchestrator over
  two copies of the fixture and asserts the pooled rescore, the by-source split, the
  per-run quantification and the cross-run LFQ. The multi-run path had no test of
  any kind, and its split had a silent data-loss case (see Fixed).
- The experiment manifest records one artifact per output it writes, each with a
  content hash, row count and schema version. It previously listed output paths and
  nothing else, so two experiment results could not be compared. New artifact
  identity `lfq_maxlfq` for the cross-run table.
- `sbom.cdx.json`: a CycloneDX 1.5 software bill of materials generated from
  `cargo metadata --locked` by `ci/gen_sbom.py`, covering all 173 components with
  purls and the full dependency graph. Shipped in the release archive and at
  `/opt/mumdia/sbom.cdx.json` in the image, and checked for staleness in CI.
  `THIRD_PARTY_LICENSES.md` is a notice document for a human reader; this is the
  machine inventory a vulnerability scanner or a software inventory consumes.
- `pip-audit` over both resolved sidecar environments, strict on the weekly
  scheduled run and advisory on pull requests, plus `pip freeze --all` uploaded per
  environment as a 90-day artifact. This is what covers the Python dependency
  surface, which has no Dependabot support: the pins live in the `pip:` sections of
  the `env/` conda specifications, which Dependabot cannot parse, and a mirror
  requirements file would be a second list that nothing installs. Reasoning in
  `docs/14`.
- Release platforms are now Linux (musl), Windows and Apple silicon. The Intel Mac
  target was removed: it required GitHub's `macos-13` label, which no longer receives
  a runner (measured 2026-08-28: queued over two hours with none assigned, in two
  separate rehearsals, while every other target finished in about three minutes), so
  a real tag would have hung until GitHub's 24-hour queue timeout and then failed (a
  rehearsal job was observed reporting exactly `24h0m0s`; that is the limit on waiting
  for a runner, not the six-hour limit on a running job). Cross-compiling it on
  the Apple silicon runner was rejected because the result cannot be executed there,
  and publishing the one archive nobody ran is what the verification step above exists
  to prevent. Intel Mac users build from source or use the container image.
- Docker base images pinned by digest, with a Dependabot `docker` entry to keep the
  pins current. A tag is mutable, so a rebuild of the same commit could previously
  produce a different image.
- `--min-assertions` on `ci/check_smoke.py`: the smoke run fails if fewer assertions
  execute than the count quoted in the documentation. The documented count was 112
  while 117 ran, and a guard block that stops executing fails no assertion, so it
  reads as a pass.
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
- End-to-end smoke test, run in CI on Linux and Windows: `ci/smoke.sh` builds a
  synthetic library from `test_data/fixture.fasta`, generates a matching mzML from
  the engine's own library so the planted peaks cannot disagree with the mass
  model, runs the single-run pipeline twice, runs `run-experiment` over two copies
  of the fixture, and asserts 136 things. It covers mzML parsing, the library
  build, the `run` orchestrator and its manifest, retention-time calibration, the
  report writers, and the multi-run path (pooled rescore, by-source split, per-run
  quant, cross-run LFQ), none of which had any test. `--min-assertions` fails the
  run if fewer assertions execute than the count quoted here, so a guard block
  that stops running cannot pass silently.
- A further CI job asserts the two platforms produced byte-identical
  `peptides.tsv` and `proteins.tsv`. The native pipeline turns out to be
  byte-reproducible across operating systems, not only across runs.
- `tests/python`: 71 tests over the Python worker contracts, run in CI. Tests
  needing torch, mokapot, deeplc or ms2pip skip rather than fail.
- `docs/23_cli_reference.md` and `docs/24_config_reference.md`, generated from
  `--help` and from `config.rs` by `ci/gen_*_reference.py` and checked for
  freshness in CI, so a new flag or field lands with its documentation. The second
  includes the environment-variable table that existed nowhere: 47 variables read
  across engine and sidecars, plus the 11 the code sets.
- `bench/`: the portable part of the ProteoBench scoring path, the two recorded
  results with their row units and q columns, and the measured resource profile of
  a reference run (85 minutes and 13.1 GB of artifacts from a 1.94 GB mzML,
  rescoring 80% of it).
- `ci/check_doc_refs.py`, run in CI: fails when a tracked file cites a Markdown
  document the repository does not ship.
- `CONTRIBUTING.md`, `SECURITY.md`, this changelog, and
  `docs/22_release_plan.md`.

All new quantification options default to off, so an existing configuration
produces bit-identical results.

### Changed

- **Competition key: `compete.group_by = peptidoform_charge`** (keys
  `(pform_id, label, charge, peak_rank)`). Sibling charge states and modforms of one
  peptide are separate precursors that compete only against their own alternative
  peaks, the unit DIA-NN reports at and the key every benchmark in
  `docs/28_feature_selection_analysis.md` ran under (entrapment FDP flat at
  0.48-0.64%). The previous default `base_peptide` (renamed from `precursor`, which it
  was not) deleted every charge and modification variant of a peptide but the highest
  `prelim_score` before rescore: 23% of the extracted candidates on HYE B01, 46.6% on a
  modification-rich library, at an unchanged peptide count. It stays available as an
  explicit peptide-level population and must not be used for a PTM search.

- CPU PyTorch in the three DeepLC-bearing environment sets (`env/docker-deeplc.yml`,
  `env/mumdia-deeplc.yml`, `env/console-requirements.txt`) moves from `2.12.1+cpu` to
  `2.14.0+cpu`, the first version whose metadata allows a `setuptools` without
  PYSEC-2026-3447 (`>=77.0.3` instead of `<82`). Resolved on 2026-09-06 with
  `deeplc==4.1.1`: numpy 2.4.6, pandas 2.3.3, psm-utils 1.5.5, scikit-learn 1.9.0,
  setuptools 84.0.0, i.e. the same scientific stack as before with only torch
  changed. DeepLC 4.1.1 declares `torch<3,>=2.6.0`. Neural-network training is not
  bit-deterministic across torch versions, so expect seed-level, not result-level,
  differences in `nn_torch` rescoring.
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

- The sidecar environment specifications pin `setuptools>=83`. The CI audit of the
  resolved DeepLC environment found `setuptools 81.0.0` (PYSEC-2026-3447,
  CVE-2026-59890, fixed in 83.0.0) and failed the main branch after the merge of #54.
  The conda-level pin alone did not hold: `torch 2.12.1` declares `setuptools<82`, so
  pip downgraded the conda-installed 84.0.0 to 81.0.0 underneath it. torch is now
  2.14.0 (see Changed) and the floor is repeated in the pip sections and in the
  desktop requirement set. The audit step is advisory on pushes to main as well as
  on pull requests, as its comment already intended; the weekly scheduled run and a
  manual dispatch stay strict.
- A single malformed retention time in an mzML aborted the whole run. `convert`
  validated peak m/z and intensity but not the scan start time, so one `NaN` value
  passed unchecked into the spectra artifact and then panicked inside extract with
  `called `Option::unwrap()` on a `None` value`, naming neither the file, nor the
  scan, nor the value. Reproduced by editing one value in the fixture mzML. Such
  spectra are now dropped with a count and the first offending scan id, which loses
  nothing (a spectrum with no retention time cannot be placed in a chromatogram) and
  leaves identifications unchanged; `ci/smoke.sh` asserts all three.
- Every float ordering in the workspace now uses `total_cmp` rather than
  `partial_cmp(..).unwrap()` (25 sites) or `partial_cmp(..).unwrap_or(Equal)` (36
  sites). The first panics on NaN; the second is worse, because `Equal`-on-NaN is an
  intransitive comparator and `sort_by` has detected that and panicked since Rust
  1.81, so it converted a deterministic failure into an intermittent one. `total_cmp`
  agrees with both on every finite value: the fixture's `peptides.tsv` and
  `proteins.tsv` hashes are byte-identical across the change. One of the rewritten
  comparators picks the competition winner, where treating every NaN as equal made
  the surviving row depend on iteration order.
- `compete` panicked instead of erroring when a `.schema.json` companion named a
  feature column the parquet does not have, which a stale companion beside a
  rewritten table produces. It now names the column and the file and says to delete
  the companion.
- `scripts/make_reverse_decoys.py` silently assigned 0 Da to any modification outside
  its eight-name table, so those decoys got fragment m/z for the wrong molecule and
  could never match. A decoy that cannot match does not compete, which makes the
  target-decoy null optimistic for exactly the peptides carrying that modification,
  and nothing in the output distinguished such a decoy from a good one. The sampled
  calculator check could not catch it: it compares 500 precursors at the 99th
  percentile. Unknown modifications now raise, `valid()` rejects the peptidoform so
  no decoy is written for it, and the script reports the names and counts. This
  matches the engine's own parser, which has always returned
  `MassError::UnknownModification`.
- `run-experiment` dropped PSMs silently when splitting the pooled scored table by
  run. `split_by_source` filtered on `source == i` for each output table and returned
  `Ok` regardless, so any row whose `source` had no output table went nowhere: every
  per-run quantity and the cross-run LFQ were then computed from a smaller population
  with no error and no warning. It now counts the rows it placed and refuses if that
  is not all of them.
- The library helpers write parquet the engine can read on any pandas.
  `DataFrame.to_parquet` chooses the arrow string width itself, and pandas 3
  writes `large_string`, which the engine rejects at load with
  `column 'peptidoform' is not utf8`. Every library built by
  `import_diann_lib.py`, `make_shift_decoys.py`, `make_reverse_decoys.py` or
  `augment_library.py` on a current pandas was therefore unreadable, breaking the
  imported-library path. They now write through `scripts/_lib_io.py`. Found by the
  new sidecar contract tests.
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

- The sidecar contract tests cover the workers' file contracts, not the science:
  the tests that need torch, mokapot, DeepLC or MS2PIP skip on a runner without
  them, so CI does not validate rescoring or retention-time prediction behaviour.
- The end-to-end smoke test runs on Linux and Windows, not macOS, and uses the
  native predictors only. A separate job imports DeepLC, mokapot and MS2PIP in
  real conda environments on any pull request touching `scripts/`, `env/`,
  `tests/python/` or the Dockerfile, but no CI job runs the sidecar path end to
  end on data.
- Under an experiment-wide rescore no per-run TSV report is written; per-run counts
  come from the split scored tables on `run_psm_q`, and the experiment-wide
  `peptides.tsv` / `proteins.tsv` are the reports.
- `mbr.strategy` distinguishes only none from not-none; `rt_window_s`,
  `decoy_transfer` and `requant_all` are accepted but not wired.
- `extract.retain_top_peaks > 1` writes diagnostic peak alternatives that do not
  reach features or rescoring.
- No ion mobility support: a Bruker diaPASEF `.d` is converted to 3D spectra and
  searched with more interference than a 4D engine would see. Vendor formats other
  than Thermo `.raw` are converted through msconvert but have not been exercised on
  real files. No wildcard or terminal variable modifications.
- The desktop application has not been clicked through end to end; its backend is
  unit-tested and both bundles were built and inspected.
