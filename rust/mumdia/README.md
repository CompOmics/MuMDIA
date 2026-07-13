# MuMDIA (Rust) — MVP

A clean-room Rust reimplementation of MuMDIA, a data-independent-acquisition
(DIA) proteomics search engine, built to the specification in `../../PLAN.md`.
This is the **MVP milestone** (PLAN.md Section 10): mzML only, 3D (no ion
mobility), one decoy strategy, fixed tolerances, b/y singly-charged fragments,
the `minimal` feature set, per-run extraction, and experiment-wide rescoring
with native target-decoy q-values.

## Pipeline

One binary, one subcommand per stage. Each stage is independently runnable on
path-addressable inputs and writes Parquet plus a `<artifact>.report.json`
(PLAN.md Section 3.5).

```
convert       mzML                     -> spectra_ms1/ms2/isolation_windows/ms2_to_ms1.parquet
digest        FASTA                    -> peptides.parquet            (fully tryptic + decoys)
peptidoforms  peptides                 -> peptidoforms.parquet        (fixed+variable mods, charges)
predict-frag  peptidoforms             -> fragment_library_*.parquet  (b/y m/z + intensity + iRT)
search-seed   spectra_ms2 + library    -> seed_psms.parquet           (native hyperscore seed)
rt-im-train   seed_psms + library      -> run_windows.parquet, cal.json (LOESS RT calibration)
extract       spectra_ms2 + lib + win  -> psms_extracted, chromatograms (peak-major cascade)
features      psms + chromatograms     -> features.parquet, run.pin   (minimal feature set)
compete       features                 -> psms_competed.parquet       (best per group, pre-FDR)
rescore       psms_competed            -> psms_scored.parquet          (native TDA q-values)
run           config + FASTA + mzML    -> all of the above + manifest.json
inspect       <artifact.parquet>       -> schema + head + row count
```

## Build

```
cd rust/mumdia
cargo build --release        # binary at $CARGO_TARGET_DIR/release/mumdia
cargo test                   # 15 unit + 2 integration tests
```

Notes:
- The build is pure Rust (no cmake/C toolchain): Parquet uses SNAPPY, mzdata uses
  the `miniz_oxide` zlib backend, and `arrow-ipc` is pinned to a lighter
  opt-level to avoid a Windows codegen crash (see `Cargo.toml`).
- `.cargo/config.toml` points the target directory off any OneDrive-synced path.

## Run

Full pipeline on the E. coli test data:

```
mumdia run \
  --fasta ../../fasta/ecoli_22032024.fasta \
  --mzml  ../../mzml_files/LFQ_Orbitrap_AIF_Ecoli_01.mzML \
  --out-dir out_run
```

Any stage standalone, e.g. extraction on prior outputs:

```
mumdia extract \
  --ms2 out/spectra/spectra_ms2.parquet \
  --library-precursors out/fragment_library_precursors.parquet \
  --library-fragments  out/fragment_library_fragments.parquet \
  --run-windows out/run_windows.parquet \
  --out-psms out/psms_extracted.parquet \
  --out-chrom out/chromatograms.parquet
mumdia inspect out/psms_extracted.parquet
```

## Validated result (E. coli, full LFQ_Orbitrap_AIF file)

- Library: 342,395 target peptides + decoys -> 2,040,864 candidates, 12.2M fragments.
- Spectra: 3,106 MS1 + 465,806 MS2 scans; 152 overlapping ~4 Da DIA windows.
- Seed: 1,307 confident peptides at 1% FDR.
- End to end: **1,217 target peptides at 1% FDR** after extraction + native rescoring.

The pipeline is deterministic (seeded decoys, zero-initialized logistic
rescorer, single-threaded numerics) and every stage runs standalone.

## Architecture

- `mumdia-core`: shared types, typed config with per-stage overrides and
  unknown-key rejection, the run manifest, a ProForma/UniMod mass model, and
  physical constants.
- `mumdia-io`: a typed column/table layer over Arrow + Parquet, blake3 content
  hashing, JSON, the `inspect` command, and per-artifact reports.
- `mumdia` (bin + lib): the fragment index, the predictor/rescorer traits, and
  the stages.

## Choice points (PLAN.md Section 9)

Every algorithmic choice is a typed config field backed by a strategy. MVP ships
MVP-conservative defaults: `decoy.strategy = reverse`, fixed tolerances,
`features.set = minimal`, native RT and fragment-intensity predictors, and the
native target-decoy rescorer. Later tiers (learned tolerances, MS2PIP/DeepLC
sidecars, Mokapot, IM, MBR, quantification) are strategies switched on, not new
plumbing.

## Documented MVP deviations from PLAN.md

- **Seed search** is a native Sage-lite hyperscore over the shared fragment
  index (behind the file contract), not the Sage binary. The published
  `sage-core` crate is an unrelated project, and a native seed avoids a heavy
  git dependency; a Sage adapter is the intended v1 default.
- **ML predictors** default to native fallbacks (a linear retention model and a
  heuristic fragment-intensity model) so the engine runs with zero external
  runtime dependencies. MS2PIP/DeepLC/Mokapot sidecars are opt-in strategies
  over the same file contract. Identification counts will rise substantially
  once the real predictors are wired (v1).
- **RT is a per-candidate window post-filter** at probe time rather than a
  pre-partitioned RT index (the fallback the plan explicitly allows), keeping
  the fragment index run-independent.
- **Tables** use `arrow` + `parquet` directly rather than Polars; the on-disk
  Parquet is identical and open.
- No Python MuMDIA pipeline or Sage binary is present in this environment, so
  golden-file parity is replaced by a crafted-input standalone test, a
  determinism test, and the reported E. coli identification count.
```
