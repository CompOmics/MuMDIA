# 22. Release plan

Plan for the first public releases of MuMDIA. Written 2026-08-27 from an inventory
of the tree at `b39769e` plus the uncommitted quant/MBR patch. Every gap below
cites the file that shows it. Effort figures are working days for one developer
with agent support and are estimates, not commitments.

The plan proposes two releases:

- `v0.1.0`: the engine as it exists today, made installable, portable, tested,
  and documented. Single-run `run` and pooled multi-run `run-experiment` without
  the second-pass MBR. Nothing scientifically new is promoted to a default.
- `v0.2.0`: the validated multi-run workflow (second pass against an empirical
  library, fragment-consensus guard, condition-evidence reporting rule, and the
  quant fixed-window defaults) moved from the doxy shell prototypes into the
  engine, gated by entrapment and two acquisitions as `docs/20` requires.

Splitting keeps the "hard to run" problems from blocking on the largest piece of
new engineering.

## 1. Where the tree stands

### 1.1 Repository and build state

- Branch `feat/sensitivity-improvements` is 11 commits ahead of `main`; the three
  commits on `main` not on the branch are merges of this branch's own PRs (#44
  to #46). A dry-run merge has zero conflicts.
- Uncommitted patch (216 insertions, 11 deletions) in
  `rust/mumdia/crates/mumdia-core/src/config.rs`,
  `rust/mumdia/crates/mumdia/src/stages/quant.rs`, `scripts/mbr_worker.py`. It
  adds `quant.fragment_selection`, `fixed_scan_halfwidth`, `fixed_window_s`,
  `baseline_subtract`, `baseline_flank_scans`, `baseline_quantile`, and the
  MBR worker fix that lowers `run_psm_q` and `experiment_psm_q` for accepted
  transfers. Measured 2026-08-27 against the stated gates:
  - `cargo fmt --check` fails (two hunks, `quant.rs:474`, `quant.rs:698`);
  - `cargo clippy -D warnings` fails: `trapezoid_scans` (`quant.rs:85`) and
    `trapezoid_scans_opts` (`quant.rs:110`) are never called;
  - `cargo test`: 121 pass, 1 fails.
    `quant_run_preserves_unquantifiable_ids_and_applied_window_contract` panics
    because `quant.rs:476` now requires a `predicted_intensity` chromatogram
    column. Pre-patch `chromatograms.parquet` artifacts and the test fixture lack
    it. This is a compatibility regression, not a test bug;
  - no test references any of the six new fields.
- CI (`.github/workflows/ci.yml`) runs only `cargo build --release --locked` and
  `cargo test --locked` on three OSes. It does not run fmt, clippy,
  `python -m compileall`, or a config-parse check, although `CLAUDE.md` names
  all four as the gate. `release.yml` and `docker.yml` trigger on `v*` tags; no
  such tag has ever been pushed, so neither workflow has ever run. The only tag
  is `legacy-python-v1`.
- Repository is public (`github.com/CompOmics/MuMDIA`), Apache-2.0, no releases,
  no open issues or PRs.

### 1.2 Why it is hard to run today

1. Python interpreters are absolute paths in the config
   (`predict_frag.deeplc_python`, `predict_frag.ms2pip_python`,
   `rescore.python`, `mbr.python`; `config.rs:422-441,1215,1241`). There is no
   PATH lookup and no discovery. The only tracked config,
   `config.local-diann-lib.json:13-18`, carries `C:/Users/robbi/...` and
   `H:/OneDrive - UGent/...` paths and is the config `README.md` and `CLAUDE.md`
   tell users to run.
2. `predict_frag.sidecar_script_dir` defaults to `"scripts"` relative to the
   current working directory (`config.rs:439`, resolution in
   `crates/mumdia/src/sidecar.rs:20-38`). Running from any other directory
   silently changes which scripts are found.
3. The strongest measured workflow (pass 2, guard, provenance rule, fixed-window
   quant) exists only as shell and pandas prototypes on doxy
   (`~/astral/pass2_pipeline_w.sh`, `pass2_guard_w.sh`, `prov_filter.py`,
   `~/hye/pass2_aif.sh`; local copies under `C:/Users/robbi/mumdia_bench/`).
   No Rust code builds an empirical library or drives `extract
   --restrict-candidates --run-windows --library-fragments` for a second pass
   (`run.rs:340` and `run_experiment.rs:221` pass `restrict_candidates: None`).
4. About 40 `MUMDIA_*` environment variables tune the Python workers
   (`scripts/nn_rescore_worker.py`, `mokapot_worker.py`,
   `entrapment_worker.py`, `deeplc_finetune.py`). None is exposed in `Config`
   or on the CLI; they are listed only in `docs/13`. The Rust side reads no
   thread setting at all (`RAYON_NUM_THREADS` is rayon's own default; there is
   no `--threads`).
5. No `pyproject.toml` or `requirements.txt` for the sidecars. `env/` has three
   conda specs; `env/mumdia-rescore.yml` is unpinned, the Docker specs pin
   `mokapot==0.10.0`, `torch==2.12.1+cpu`, and a DeepLC git commit. No spec
   covers `xgboost` or the four library helpers.
6. `mumdia doctor` (`main.rs:364-432`) probes three interpreters by
   `importlib.util.find_spec`. It does not probe `mbr.python`, does not check
   `sidecar_script_dir` or the worker files, does not check versions (DeepLC
   4.0.0a2 vs 4.1.0, `numpy<2`), does not import `deeplc` in the order the
   worker needs (the `WinError 1114` failure), and does not check input files
   or the library contract (`index.rs:112-125,215-231`).
7. No global CLI flags. `--config` is absent on `convert`, `quant-lfq`,
   `inspect`, `audit`, `report` (`main.rs:26-360`). Verbosity is `RUST_LOG`
   only (`mumdia-io/src/lib.rs:13-17`). No progress output during multi-hour
   stages; each stage logs once at completion.
8. `run` accepts one mzML (`run.rs:23`). `run-experiment` never calls `report`,
   writes no `peptides.tsv`/`proteins.tsv`, passes `out_fragment: None`
   (`run_experiment.rs:510`) so `fragment_quant.parquet` is missing and the
   fragment-consensus guard cannot run, and writes a thin
   `experiment_manifest.json` without artifact hashes, engine version, or model
   identities (`run_experiment.rs:526-536`).
9. MBR tiers are inert: `mbr.strategy` only distinguishes none from not-none
   (`config.rs:1534-1540`); `rt_window_s`, `decoy_transfer`, `requant_all` are
   marked NOT YET WIRED (`config.rs:1196-1212`). The `mbr` stage is inline in
   `main.rs:801-841`.
10. Worktree clutter: 19 root `config*.json` files are untracked and not
    ignored (`.gitignore` covers `cfg_*.json` and `config.strasbourg*`, not
    `config_*` or `config.aif-*`/`config.hye-*`); `rust/mumdia/target/` is not
    ignored (`/target` is root-anchored); `lib/`, `raw_files/`, `val2/`,
    `val9/`, `mbr_k3/`, `mbr_work/`, `alphadia/` (about 100 GB in total with the
    ignored `out_*` dirs) sit untracked in the tree.
11. `plan.md` is untracked (ignored by the root `/*.md` rule) but cited as
    authoritative in more than 30 tracked source files (`manifest.rs:1`,
    `error.rs:1`, `hash.rs:1`, `tests/pipeline.rs:1`, `docs/14:437,466`). A
    public clone has dangling references.
12. No fixture data is checked in (122 tracked files, none `.mzML`, `.parquet`
    or `.tsv`); `.gitignore:3` whitelists `test_data/**/*.parquet` but no
    `test_data/` exists. The only smoke procedure (`docs/19:275-320`) needs an
    untracked mid-gradient mzML slice and untracked `lib/`, and its regression
    targets are marked "not re-verified".

### 1.3 What is in good shape

- 152 unit tests plus 2 integration tests (`crates/mumdia/tests/pipeline.rs`)
  that build inputs in-process and assert byte-identical apex RT across runs.
- Native engine is deterministic by construction (no `rand`, ordered
  reductions; `docs/14:437-446`).
- Versioned artifact schemas (`mumdia-core/src/schema.rs:6-26`, 19 tuples),
  `report.json` per artifact, blake3 content hashes, full `manifest.json` for
  `run` (`manifest.rs:9-31`).
- `anyhow` error handling throughout; zero `process::exit`; no
  `todo!`/`unimplemented!`.
- Developer documentation `docs/01` to `docs/21` is thorough and code-cited;
  `docs/15` documents every Parquet column; `docs/17` covers quiet failure
  modes.
- Release and Docker workflows exist and only need to be exercised.
- Pure-Rust build (no C toolchain), `Cargo.lock` tracked, `--locked` builds.

## 2. Decisions (WP0, settled 2026-08-27)

| # | Decision | Settled as |
|---|---|---|
| 1 | Scope of the first release | Two releases as described above. `v0.1.0` = today's engine made installable. `v0.2.0` = second-pass workflow. |
| 2 | Version number | `0.1.0` (matches `Cargo.toml`; the Python predecessor is tagged `legacy-python-v1`). |
| 3 | Supported platforms | Linux, Windows and macOS. Binaries for `x86_64-unknown-linux-musl`, `x86_64-pc-windows-msvc`, `aarch64-apple-darwin` and `x86_64-apple-darwin`, each smoke-tested on its own architecture. Docker `linux/amd64`. GPU optional, never required. |
| 4 | Primary install path | Docker as the reference environment; binary plus pinned conda env as the native path. Both smoke-tested in CI. |
| 5 | Recommended profile in README | Imported DIA-NN library (highest measured sensitivity) with the FASTA-native path documented as the licence-free alternative. |
| 6 | Defaults promoted in `v0.1.0` | None. Ship the new quant options documented and off. Promote in `v0.2.0` after entrapment (see WP7). |
| 7 | `plan.md` | **Stays untracked.** Reversed from the original recommendation to track it, on inspection of its contents: section 8 is a comparative dossier that quotes proprietary constants and internal line numbers from a closed-source engine (for example the DIA-NN terminal-residue decoy mutation map, verbatim from `diann.cpp`), and section 11 itself warns that reuse needs licence clearance. Publishing it would contradict the clean-room boundary the README claims. The 130 dangling citations were redirected to the tracked `docs/` guide instead, and `ci/check_doc_refs.py` now prevents new ones. |
| 8 | Citation | Add `CITATION.cff`; connect the repository to Zenodo so `v0.1.0` gets a DOI. **Blocked on the author list**, which is not derivable from the repository and must not be guessed. |
| 9 | Python floor | Python >= 3.11, **DeepLC 4.1.1** (Robbin's preference and now the pinned version; 4.0.0a2 overfits, `docs/08` section 4b), mokapot 0.10, torch >= 2.6 CPU. Verified: DeepLC 4.1.1 resolves with torch 2.12.1+cpu, numpy 2.4.6 and pandas 2.3.3 on Python 3.11, both in the image and from `env/mumdia-deeplc.yml`. |
| 10 | Environment variables | Keep the ~10 NN knobs that matter as config fields passed to the worker; leave the rest as documented env vars. |

## 2b. Progress (2026-08-27)

WP1, WP2 and most of WP4 are done; the full gate is green on the branch. What
landed, and what each item was verified against:

| commit | change | verification |
|---|---|---|
| `83ba81d` | quant: library-ranked fragments, fixed integration window, optional `predicted_intensity`, honest reported bounds | 6 new unit tests; `quantity` bit-identical to the submission output on 72,168 real precursors, reported window corrected 29.1 s -> 34.9 s |
| `9d584ba` | MBR: lower every PSM q column for an accepted transfer | the 34,280-of-34,664 dropped-transfer measurement |
| `000f22d` | ignore the build dir, experiment configs and local data | `git check-ignore`; `docs/22` itself was being silently ignored |
| `16cfbd7` | DeepLC 4.1.1 pin and `env/mumdia-deeplc.yml` | conda resolved the spec and the worker's import graph loads on Linux |
| `46ed1aa` | CONTRIBUTING, SECURITY, CHANGELOG | - |
| `8b70855` | CI runs fmt, clippy, compileall, config and env parsing, and the doc-reference check; Dependabot | the gate caught two unformatted hunks, two dead functions and one failing test on a branch treated as ready |
| `0b4133f` | Cargo metadata, MSRV aligned to the tested toolchain, stripped binaries | `cargo metadata`; `Cargo.lock` unchanged |
| `a3ce684`, `140c79b` | release archive is a working installation with checksums; image smoke-tested and non-root | built on a Linux host: 4.62 GB, uid 57439, `doctor` passes on both configs, bind mount writable with `--user` and refused without it |
| `ec89265` | 130 citations redirected off untracked design notes | `ci/check_doc_refs.py`: 305 references, all resolvable |
| `ed87157` | TSV report columns documented, schema-version drift fixed | read against `report.rs` and `schema.rs` |

Not done, in the order the sequencing below puts them: WP3 (portability and
usability) entirely; WP4's Python packaging (`scripts/pyproject.toml`, pinning
`env/mumdia-rescore.yml`); WP5 (README rewrite, generated CLI and config
references, resource table); WP6 (fixture, end-to-end smoke test, Python tests,
`unwrap` audit, seeding DeepLC, provenance in the manifest, benchmark suite);
`CITATION.cff` pending an author list; and the branch is not merged or pushed.

Two consequences of not having pushed: the new CI workflow has never executed on
a GitHub runner, and neither has the Docker workflow. The equivalent checks were
run locally and on a Linux host, so the gate itself is verified, but the YAML
that drives it is not.

## 3. Work packages

### WP1. Stabilize the tree (2 to 3 days)

Goal: `main` is green on the full gate and contains all validated code.

- [x] Fix the quant patch: read `predicted_intensity` optionally
      (`Table::read_cols` with a fallback; require it only when
      `fragment_selection = predicted`); delete or use `trapezoid_scans` and
      `trapezoid_scans_opts`; run `cargo fmt`. Also corrected the reported
      integration bounds, which described the walked window the fixed-window path
      never integrated.
- [x] Add tests for `select_fragment_areas`, `trapezoid_fixed_opts` (scan and
      second forms), `flank_baseline`, and a `QuantConfig` round-trip of the six
      new fields. Add one test for the legacy path being bit-identical with the
      new fields at defaults. 159 tests now pass, up from 152 with one failing.
- [x] Commit the validated code (`83ba81d`, `9d584ba`).
- [ ] Merge the branch into `main`; delete stale branches. Not started: the
      branch is 20 commits ahead and has not been pushed.
- [x] `.gitignore`: `**/target/`, the experiment configs, and the local data
      directories. Also re-included the standard root files, which `/*.md` had
      excluded, and root-anchored the note rules, one of which was silently
      ignoring this document.
- [ ] Move the two or three configs worth keeping into `configs/examples/` with
      placeholders instead of machine paths (see WP3).
- [x] Remove `config.local-diann-lib.json` from tracking after WP3 provides a
      portable replacement; update the references in `CLAUDE.md`, `docs/19`,
      `docs/20`. Replaced by `configs/examples/{native,fasta-sidecars,diann-library}.json`,
      all using `"auto"`; the old file stays on disk but ignored. `README.md`
      never named it.
- [x] Decide `plan.md` (WP0 #7) and act on it: it stays untracked, and the 130
      citations that pointed at it and at other local notes now point at the
      tracked guide (`ec89265`).

Acceptance: `cargo fmt --check`, `cargo clippy --workspace --all-targets -D
warnings`, `cargo test --workspace`, `python -m compileall -q scripts` all
pass on `main`; `git status` clean on a fresh clone plus the documented local
files.

### WP2. CI and quality gates (1 to 2 days)

- [x] `ci.yml`: added `cargo fmt --check`, `cargo clippy ... -D warnings`,
      `python -m compileall`, a JSON parse over every tracked config, a YAML
      parse over the env specs, and the documentation-reference check. Point
      `shipped_configs_parse` at `configs/examples/` once WP3 creates it.
- [ ] Add a `doctor` unit test and a CLI `--help` snapshot test so subcommand
      renames are caught.
- [ ] Add the fixture smoke job from WP6 once the fixture exists.
- [x] `dependabot.yml` for Cargo and GitHub Actions, with `arrow`/`parquet`
      grouped apart because they carry the on-disk contract. `cargo audit` not
      added.
- [ ] Branch protection on `main`: CI required.

Acceptance: a PR that breaks any stated gate is red.

### WP3. Portability and usability (4 to 6 days)

Goal: one config runs unchanged on Windows, Linux, macOS, and in Docker.

- [x] Interpreter discovery: `"auto"` or absent resolves through
      `MUMDIA_PYTHON_<ROLE>`, `MUMDIA_PYTHON`, `CONDA_PREFIX`, `VIRTUAL_ENV`,
      then `python3`/`python` on PATH, accepting a candidate only after it
      imports the role's modules. Explicit paths keep working. The resolved path
      lands in `manifest.json` (via the resolved config) rather than in each
      stage's `report.json`.
- [x] `sidecar_script_dir`: resolved against the config file's directory and the
      executable's directory as well as the CWD; the release archive and the
      image both satisfy it.
- [ ] `mumdia init-config --profile fasta|diann-lib --out config.json`: writes
      a commented config with discovered interpreters and no machine-specific
      strings.
- [x] `mumdia doctor`: probes `mbr.python`, checks `sidecar_script_dir` and each
      worker file, reports package versions, and warns below the DeepLC 4.1.1
      floor.
- [ ] `doctor` remainder: probe `percolator_bin`; execute each worker with a
      `--selftest` so the DeepLC import order is exercised (a module-presence
      probe cannot see it; the container CI job covers it for the image); and
      with `--mzml`/`--lib-precursors`/`--fasta` check readability and the
      library contract (`candidate_id` contiguous, `precursor_mz` sorted,
      snappy + utf8).
- [x] Global CLI flags: `--threads N` (rayon global pool, forwarded to workers as
      `MUMDIA_NN_THREADS`/`OMP_NUM_THREADS`), `--log-level`, `-v`/`-vv`, `-q`,
      all accepted on either side of the subcommand.
- [ ] `--config` on every subcommand that reads config (still absent on
      `convert`, `quant-lfq`, `inspect`, `audit`, `report`).
- [ ] Progress: periodic `info!` lines with counts and elapsed time inside the
      long loops (extract, features, rescore worker relay), plus a per-stage
      wall-clock summary at the end of `run` and `run-experiment`.
- [ ] Config surface: add `Config::validate()` warnings for fields that are set
      but not read in the selected mode; group experimental fields in the
      generated reference (WP5) rather than by moving them in the schema.
- [ ] Move the NN knobs that matter (`threads`, `device`, `stream_gb`,
      `seeds`, `epochs`, `iters`, `batch`, `lr`, `hidden`, `early_stop`) into a
      `rescore.nn` config struct passed as worker arguments; keep env vars as
      overrides.
- [ ] Sidecar failures: on non-zero worker exit, surface the worker's last 30
      stderr lines in the error; check this exists in `sidecar.rs` and add a
      test.

Acceptance: the same `configs/examples/diann-lib.json` runs `mumdia doctor`
green and completes the fixture run on the three OSes and in Docker.

### WP4. Packaging (3 to 5 days)

- [x] Cargo metadata: description, repository, homepage, keywords, categories,
      `publish = false`, `rust-version` aligned to the tested toolchain, and
      `strip = "symbols"`. `readme` omitted, since nothing is published to
      crates.io.
- [x] Release archive layout: binary, `scripts/`, `env/`, `docs/`, `README.md`,
      `LICENSE`, `CHANGELOG.md`, `configs/` when present, plus sha256 checksums
      and a per-target binary smoke test. `aarch64-unknown-linux-gnu` not added;
      `x86_64-apple-darwin` was.
- [ ] Python sidecars: `scripts/pyproject.toml` for a `mumdia-sidecars`
      package with extras `rescore`, `deeplc`, `ms2pip`, `mbr`, `helpers`;
      pinned lower bounds; console entry points for the helpers. Pin the conda
      specs (`deeplc==4.1.0`, torch CPU, `mokapot==0.10.0`, `psm_utils`,
      `pyarrow`, `numpy<2` where required); add `xgboost` as optional.
- [x] Docker: non-root user, OCI labels, `git` dropped from the image, and the
      image built and checked in CI before any push. Verified on a Linux host
      rather than assumed. The fixture run cannot be added until WP6 creates a
      fixture.
- [ ] Windows: verify the release binary runs without the OneDrive target-dir
      redirect and that `mimalloc` and `PYTHONUTF8` behave from a plain shell.

Acceptance: downloading the release archive on a clean machine, creating the
conda env from the shipped spec, and running `mumdia doctor` then the fixture
succeeds without editing any path.

### WP5. User documentation (4 to 6 days)

- [ ] `README.md` rewrite for users: what it does, install (release archive,
      Docker, from source), quickstart on public data, the two library sources,
      output files with `peptides.tsv`/`proteins.tsv` column tables, FDR unit
      table (`q_value`, `run_psm_q`, `precursor_q`, `peptide_q_value`,
      `pg_q_value`), hardware sizing, benchmark table with the q unit named,
      status of experimental features, citation, license. Badges for CI,
      release, GHCR.
- [ ] `docs/23_cli_reference.md` generated from `--help` (a small script or
      `clap_mangen`/`clap-markdown`) and `docs/24_config_reference.md`
      generated from the `config.rs` doc comments and defaults, both regenerated
      by CI so they cannot drift. Include the environment-variable table now
      scattered across `docs/13`.
- [ ] `docs/19_getting_started.md`: rewrite around the public fixture and the
      release archive; remove developer-machine context.
- [ ] `docs/15`: add the TSV columns (currently excluded, `docs/15:816-826`).
- [ ] `docs/14`: refresh the test counts (states 126, actual 152) and describe
      the new CI gates.
- [ ] Reference-run resource table: wall clock, peak RSS, disk per stage for one
      AIF file and for the six-file HYE experiment, from the `elapsed_ms` values
      already in `report.json`.
- [ ] `CHANGELOG.md` (Keep a Changelog format), `CITATION.cff`,
      `CONTRIBUTING.md` (build, gates, clean-room rule, how to add a stage),
      `SECURITY.md`, issue and PR templates.

Acceptance: a colleague who has never run MuMDIA completes the README
quickstart on their machine using only the docs. Record who did it and when in
`docs/18`.

### WP6. Validation assets (5 to 8 days, partly parallel with WP3 to WP5)

- [ ] Fixture: a small public DIA slice (mid-gradient, a few hundred MS2
      spectra from the PRIDE AIF E. coli file or the ProteoBench Astral raws)
      plus a matching library subset of a few thousand precursors, under
      `test_data/` with a licence note. If larger than a few MB, fetch in CI
      from a fixed URL with a checksum.
- [ ] End-to-end smoke test in CI: `mumdia run` on the fixture in FASTA-native
      mode (no sidecars) with count bands; a second job with the conda env for
      the sidecar path. Assert `manifest.json` completeness and the documented
      artifact schema versions.
- [ ] Python tests (`pytest`) for each worker on a synthetic 200-row PIN or
      table: contract coverage (every input row scored once, finite scores),
      the M5 q-lowering in `mbr_worker.py`, the DeepLC import order.
- [ ] Robustness audit of production `unwrap` sites (`quant.rs` 21,
      `audit.rs` 15, `prescan.rs` 12) and tests for missing optional columns,
      empty runs, runs with no decoys, and a failing sidecar.
- [ ] Determinism: seed `deeplc_finetune.py` (numpy and torch, record the seed
      in `report.json`); document remaining non-determinism in `docs/14`.
- [ ] Provenance: manifest gains git SHA and build date (`build.rs` or
      `vergen`), hashes of the input mzML, FASTA, and library, and the CLI
      arguments; `experiment_manifest.json` gains the same fields as
      `manifest.json`.
- [ ] Benchmark suite under `bench/`: scripts that reproduce the AIF E. coli
      count, the HYE AIF ProteoBench scores, and the Astral ProteoBench scores
      against the recorded DIA-NN references, using the ProteoBench offline
      scorer. Results and the exact commit go into `docs/18`.

Acceptance: CI runs the fixture on every PR; the benchmark suite reproduces
the numbers in `docs/18` on the reference machine at the tagged commit.

### WP7. Productize the validated multi-run workflow (`v0.2.0`, 15 to 20 days)

This is the port of the doxy prototypes (`pass2_pipeline_w.sh`,
`pass2_guard_w.sh`, `prov_filter.py`, `pass2_aif.sh`) into the engine. The
measured effect is recorded in `docs/18` and the memory files: Astral min-3
features 92,172 to 100,528 with median |eps| 0.302 to 0.176 and CV 0.192 to
0.105; AIF |eps| 0.210 to 0.154 at unchanged completeness.

- [ ] New stage `empirical-library`: inputs `scored_combined.parquet`, per-run
      `fragment_quant.parquet`, the library tables; outputs
      `lib_empirical_fragments.parquet` (targets with experiment PSM q <= 0.01,
      cross-run consensus intensities, paired shift decoys carrying the target
      pattern) and per-run `windows_<run>.parquet` (own apex where quantified,
      otherwise cross-run median apex plus run offset; half-width from the
      cross-run RT residual p95 times two, measured 6 s on Astral and 30 s on
      AIF).
- [ ] `run-experiment` phase 2: per-run `extract --restrict-candidates
      --run-windows --library-fragments --mass-cal`, `features`, `compete`,
      pooled `rescore`, split by `source`, per-run `quant --out-fragment`.
- [ ] Fragment-consensus guard as a Rust step on the second-pass scored tables:
      rows accepted in pass 2 but not first-pass confident in that run keep
      their q only if the cosine to the consensus of anchor runs is >= 0.8 with
      >= 1 anchor (defaults measured on Astral; make both configurable).
- [ ] Condition-evidence reporting rule (F1): report a precursor's values in a
      condition only if at least one run of that condition is first-pass
      confident. Needs a run-to-condition mapping (`--conditions A,A,A,B,B,B`
      or an experiment design file).
- [ ] `mbr.strategy` semantics become `none | rt_transfer | second_pass`; wire
      or delete `rt_window_s`, `decoy_transfer`, `requant_all`; move the `mbr`
      stage out of `main.rs` into `stages/mbr.rs`.
- [ ] `run-experiment` output: per-run `peptides.tsv` and `proteins.tsv`, an
      experiment-wide precursor-by-run matrix, and an optional
      `mumdia export --format proteobench` for benchmarking.
- [ ] Quant defaults promotion: `fragment_selection = predicted`,
      `top_n_fragments = 12`, `fixed_window_s` derived per run as 1.5 times the
      median first-pass peak half-width (5 s on Astral, 20 s on AIF). Gate:
      entrapment on both acquisitions with an unchanged empirical decoy
      fraction, then a `docs/18` decision record. Both acquisitions are already
      measured; entrapment is not.
- [ ] `compete.group_by = peptidoform_charge` as default for the imported
      library profile, same gate.

Acceptance: `mumdia run-experiment` on the six Astral files reproduces the
`SUBMISSION_astral_v2` numbers within run-to-run variance without any shell
script; `docs/20` promotion gates satisfied and recorded.

### WP8. Release mechanics (1 to 2 days per release)

- [ ] Freeze: `main` green, `CHANGELOG.md` section written, version bumped in
      `Cargo.toml` and `pyproject.toml`.
- [ ] Tag `v0.1.0`; confirm `release.yml` produces the three archives and
      `docker.yml` pushes `ghcr.io/compomics/mumdia:v0.1.0`.
- [ ] Download each archive on a clean machine or CI runner, run `doctor` and
      the fixture. Pull the image and do the same.
- [ ] Publish the GitHub release with notes: what is supported, what is
      experimental, known limitations (mzML only, no ion mobility, DIA-NN
      library needs a DIA-NN licence), benchmark table with q units.
- [ ] Zenodo DOI; update `CITATION.cff`.
- [ ] Post-release: bug-fix cadence, issues as the roadmap tracker, and the
      backlog in `docs/18` (top-K peak promotion, ion mobility, trace demixing
      for AIF interference).

## 4. Sequencing and effort

| Order | Package | Days | Depends on |
|---|---|---|---|
| 1 | WP0 decisions | 0.5 | |
| 2 | WP1 stabilize | 2 to 3 | WP0 |
| 3 | WP2 CI gates | 1 to 2 | WP1 |
| 4 | WP3 portability and usability | 4 to 6 | WP1 |
| 5 | WP6 fixture, smoke, tests, provenance | 5 to 8 | WP1; benchmark suite can start earlier |
| 6 | WP4 packaging | 3 to 5 | WP3 |
| 7 | WP5 user docs | 4 to 6 | WP3, WP4 |
| 8 | WP8 release `v0.1.0` | 1 to 2 | all above |
| 9 | WP7 second-pass workflow | 15 to 20 | `v0.1.0` |
| 10 | WP8 release `v0.2.0` | 1 to 2 | WP7 |

`v0.1.0`: about 20 to 30 working days. `v0.2.0`: another 16 to 22. WP3, WP5,
and WP6 can proceed in parallel once WP1 is merged.

## 5. Definition of done for `v0.1.0`

- `main` passes fmt, clippy `-D warnings`, tests, `compileall`, config parse,
  and the fixture smoke test on Linux, Windows, and macOS.
- No machine-specific path in any tracked file. Configs resolve interpreters
  and scripts portably.
- Release archives contain binary, scripts, env specs, example configs, docs,
  and checksums. The Docker image is built and smoke-tested by CI and published.
- `mumdia doctor` catches the misconfigurations listed in `docs/17`.
- `manifest.json` records engine version, git SHA, config hash, input hashes,
  model identities, and artifact hashes; `experiment_manifest.json` records the
  same.
- README quickstart completed by a second person on a clean machine.
- `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md` present.
- Benchmark table in README names the row unit and q-value unit for each
  number and cites the commit that produced it.

## 6. Out of scope for both releases

Ion mobility and 4D data (`docs/08:30`), vendor formats other than mzML
(`docs/01:642`), wildcard or terminal variable modifications
(`docs/05:200`), percolator (`config.rs:1493-1499`), top-K peak promotion into
rescoring (`CLAUDE.md`), and trace-level demixing for AIF interference. Each is
listed as roadmap in the release notes, not implied by the docs as available.
