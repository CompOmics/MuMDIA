# Contributing to MuMDIA

MuMDIA is a Rust DIA proteomics search engine with a small set of Python sidecar
workers. This file covers the build, the checks a change has to pass, and the
project-specific rules that are easy to break without noticing.

Start with [`docs/README.md`](docs/README.md), which routes to the code-grounded
subsystem reference. [`CLAUDE.md`](CLAUDE.md) is the condensed version of the same
material and states the invariants a change must not violate.

## Build and check

The workspace lives in `rust/mumdia`. The toolchain is pinned by
`rust/mumdia/rust-toolchain.toml`, so rustup installs the right version on the
first `cargo` invocation. No cmake and no system C libraries are needed, but a C
COMPILER is: `libmimalloc-sys` compiles the vendored mimalloc allocator and
`blake3` compiles its SIMD paths, both through the `cc` build dependency. Every
supported platform's default toolchain provides one.

```bash
cd rust/mumdia
cargo fmt --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo build --release --locked
```

For Python changes:

```bash
python -m compileall -q scripts ci
python -m pytest tests/python -q -rs
python ci/check_doc_refs.py
```

And whenever a config field, a CLI flag or a dependency changes, the generated
documents have to be regenerated in the same commit. CI fails on a stale one,
because a reference nobody regenerates is worse than none: it reads as current.

```bash
python ci/gen_cli_reference.py            # docs/23, from the binary's --help
python ci/gen_config_reference.py         # docs/24, from config.rs
python ci/gen_third_party_licenses.py     # THIRD_PARTY_LICENSES.md, from Cargo.lock
```

CI runs all of the above on Linux, and the build and tests also on macOS and
Windows. A pull request that fails any of them will not be merged.

If you develop on a synced filesystem (OneDrive, Dropbox), redirect the Cargo
target directory off it first. Copy `rust/mumdia/.cargo/config.toml.example` to
`rust/mumdia/.cargo/config.toml` and set a local path. Syncing incremental build
artifacts causes file-locking corruption and is slow.

## What the test suite does and does not cover

`cargo test --workspace` exercises the native engine: config parsing and
validation, the IO layer, the fragment index, extraction, features, competition,
FDR, quantification, and a small end-to-end extract-to-rescore integration test
that builds its inputs in process.

It does **not** exercise the Python sidecars on data. `tests/python` covers their
file contracts, but every test needing torch, mokapot, DeepLC or MS2PIP skips on a
runner without them, so a passing suite is not evidence that a sidecar's science
works. A separate CI job does import DeepLC, mokapot and MS2PIP in real conda
environments, and does so on any pull request touching `scripts/`, `env/`,
`tests/python/` or the Dockerfile -- which catches an import-order or dependency
break, not a behaviour change. Say so explicitly when you report results: if you
changed a sidecar and only ran the Rust suite, the sidecar is untested.

There **is** an end-to-end smoke test, on Linux and Windows:

```bash
bash ci/smoke.sh
```

It generates its own mzML fixture rather than committing one, and generates it
from the library the engine has just built, so the planted peaks cannot disagree
with the mass model. 117 assertions cover mzML parsing, the library build, the
`run` orchestrator, the manifest, retention-time calibration, FDR and recovery,
quantification, and byte-identical output between two runs -- compared across
every artifact via the manifest's per-artifact content hashes, not just the two
rounded TSVs.

What it still cannot see is documented in
[`docs/25_release_readiness_review.md`](docs/25_release_readiness_review.md)
section 8: it runs FASTA mode only, so the imported-library production path is
uncovered; its noise is synthetic, so interference and chimeric spectra are not
represented; and its 3,820 candidates put the FDR pseudocount in charge, so it
says nothing about calibration at scale. Validate data-path changes on a real run
and report the numbers.

## Rules that are easy to break

These come from measured failures, not style preference. Each is stated with its
consequence in [`CLAUDE.md`](CLAUDE.md) and, in more detail, in the `docs/` guide.

- **Determinism.** Never let a float reduction depend on `HashMap` iteration
  order. Use an ordered map or sort explicit keys. A single unordered `f32` sum
  once moved a chromatographic apex.
- **No label leakage.** Target/decoy labels and grouping keys must not become
  predictive features.
- **Paired decoys.** Keep target and decoy populations exchangeable and
  collision-free, and keep the shared `base_peptide_id` pairing intact:
  peptide-level q estimation performs picked target-decoy competition through it.
- **Artifact schemas are versioned.** A column change requires a version bump in
  `mumdia-core/src/schema.rs` and compatibility behavior where older artifacts can
  reasonably still be read. Treat an existing artifact on disk as an input you do
  not control.
- **Clean-room boundary.** Do not copy constants, mutation maps, coefficient
  vectors, or code out of DIA-NN or any other closed implementation. Reference
  behavior may inform a design; it may not be transcribed. Some local design notes
  quote third-party source and are deliberately untracked, which is why
  `ci/check_doc_refs.py` refuses citations to documents the repository does not
  ship.
- **Parquet written outside `mumdia-io`** must be snappy-compressed with Arrow
  `utf8` string columns. Polars defaults (zstd, `large_utf8`) are rejected by the
  reader.
- **Sidecar output must cover every input row exactly once with finite scores.**
  An in-sample fallback is not an acceptable substitute for out-of-fold scores.
- **Do not present an identification count as evidence of anything else.** More
  identifications at a stated q threshold says nothing about whether the threshold
  is still calibrated or whether quantification improved. The three objectives are
  separate; see [`docs/20_sensitivity_and_quantification_playbook.md`](docs/20_sensitivity_and_quantification_playbook.md).

## Changing a default

A new default that affects results needs more than a better number on one file:

1. an empirical null (entrapment, or exchangeable paired decoys) showing the q
   threshold stays calibrated;
2. at least two datasets in different acquisition contexts;
3. a decision record in `docs/18_findings_and_decisions.md` naming the row unit
   and the q-value column the counts were selected with.

Acquisition-specific values (peak caps, RT windows, integration windows) must not
become global defaults. `--top-peaks-ms2 300` is the standing example: it is
right for one chimeric AIF run and discards 78.6% of MS2 peaks on a 50-window
Orbitrap DIA run, costing 60% of the peptides.

## Reporting benchmark numbers

Always name the row unit and the q-value column. `q_value` and
`experiment_psm_q` are pooled PSM level, `run_psm_q` is within-run PSM,
`peptide_q_value` is base peptide, `precursor_q` is peptidoform plus charge but
only under `compete.group_by = peptidoform_charge`, and `pg_q_value` is the
protein group. The grouped columns are written only to each group's single winning
row, so a per-run count taken from them after an experiment-wide rescore is
diluted by roughly one over the number of runs and is meaningless. Include the
commit the numbers came from.

## Commits and pull requests

- Conventional Commit subjects (`feat(quant): ...`, `fix(mbr): ...`,
  `docs: ...`, `chore(env): ...`). Keep the subject under about 72 characters.
- Explain **why** in the body, with the measurement if the change is motivated by
  one. A commit that changes results should say what it was measured against.
- One logical change per commit. A feature, its tests, and the doc update that
  keeps `docs/` truthful belong together; unrelated cleanup does not.
- Update the `docs/` page that describes the behavior you changed. The guide is
  code-grounded with `file:line` citations, so a change that leaves it stale is
  incomplete.

## Reporting a bug

Include the MuMDIA version (`mumdia --version`), the platform, the config, the
`manifest.json` from the output directory if the run produced one, and the stage
that failed with its log line. `mumdia doctor --config <your config>` output is
useful for anything involving a Python sidecar.

For a suspected security issue, see [`SECURITY.md`](SECURITY.md) instead of
opening an issue.
