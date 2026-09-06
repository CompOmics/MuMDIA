# Getting started: local setup and a worked end-to-end run
> Part of the MuMDIA developer documentation (see docs/README.md).

This document is the practical on-ramp for running MuMDIA on this machine. It
covers the local Python sidecar environments, converting vendor files to mzML,
the pre-built E. coli test library in `lib/`, two copy-pasteable end-to-end runs
(a zero-dependency native run and the best-sensitivity library run), and a smoke
check. It complements the algorithmic docs (04-12) with the concrete inputs and
commands that already exist on disk.

Commands below call `mumdia`, meaning whichever binary you built or unpacked:
`rust/mumdia/target/release/mumdia` from a source build, `/usr/local/bin/mumdia`
in the container, or the executable from a release archive. On a machine whose
Cargo target directory is redirected, which is the case on a synced filesystem
(`rust/mumdia/.cargo/config.toml`, `docs/14_build_test_deploy_gotchas.md`), the
binary is under that path instead.

Identification counts quoted here are taken from previously documented runs
(`CLAUDE.md`, project memory). They are stated as regression targets and were
not re-verified while writing this document.

## 0. A run that works before you have any data

The fastest way to confirm a build is sound, and the only path here that needs
nothing but the repository:

```
ci/smoke.sh
```

It builds a small synthetic library from `test_data/fixture.fasta`, generates a
matching mzML, runs the full single-run pipeline over it twice, runs `run-experiment` over two
copies of it, and asserts 136 things about the result, including that the two
single runs are byte-identical and that the by-run split of the pooled rescore
accounts for every row. It needs a
built `mumdia` binary (found automatically, or set `MUMDIA_BIN`) and a Python with
pyarrow. No sidecar, no network, no data file.

It runs from a release archive too, not only from a checkout: the archive carries
`ci/smoke.sh`, its two helper scripts and `test_data/fixture.fasta`, and
`ci/smoke.sh` finds the `mumdia` binary sitting beside them. `release.yml` runs
exactly this from a freshly unpacked archive on every target before publishing,
so the procedure in this section is tested on the artifact you download rather than
on the tree it was built from.

The fixture is generated rather than shipped, and generated from the engine's own
library: `ci/make_fixture_mzml.py` reads the precursor and fragment tables
`mumdia predict-frag` has just produced and plants exactly those m/z values, so it
cannot disagree with the mass model. What it exercises, and the Rust suite does
not, is mzML parsing, the library build, the `run` orchestrator and its manifest,
retention-time calibration on real anchors, and `quant` and `report` writing files.

Measured on it: 99.3% of planted peptides recovered, zero decoys at 1% peptide q,
LOESS calibration fitted on 110 anchors with a 1.29 s in-sample residual median.
Those are properties of a 3,820-candidate synthetic library, not a performance
claim; `bench/README.md` has the real numbers.

This is a functional check. It says the pipeline works, not how well it performs.

## 1. Local Python sidecar environments

The engine runs end to end with zero external dependencies using the native
predictors and the native `percolator_lite` rescorer. The real ML predictors and
rescorers are opt-in Python sidecars invoked over a positional-CLI file contract
(input Parquet/PIN as argv, output Parquet; see `docs/13_sidecars.md`). Each
sidecar is a separate conda environment, wired into a run purely by pointing a
config field at that environment's `python.exe`. The interpreter path is never
hardcoded in Rust; it comes from the config
(`predict_frag.ms2pip_python`, `predict_frag.deeplc_python`, `rescore.python`,
defined at `rust/mumdia/crates/mumdia-core/src/config.rs:263`, `:265`, `:921`).

Two environment specifications ship with the repository, and between them they
cover every sidecar this document uses:

| Spec | Creates | Covers |
|---|---|---|
| `env/mumdia-rescore.yml` | `mumdia-rescore` (python 3.12) | the mokapot rescore path; no torch, no DeepLC, no MS2PIP |
| `env/mumdia-deeplc.yml` | `mumdia-deeplc` (python 3.11) | `deeplc_worker.py` and `deeplc_finetune.py`, with DeepLC 4.1.1 and CPU torch. The same interpreter satisfies the `nn_torch` rescorer, which needs torch, numpy, pandas and pyarrow |

```
conda env create -f env/mumdia-deeplc.yml
```

DeepLC must be 4.1.1 or newer. The 4.0.0a2 multitask preview overfits per-run
fine-tuning badly enough to invert retention-time model rankings, so an older
version changes results and not only speed; `mumdia doctor` fails on one, and the
engine refuses to launch the DeepLC workers with it. MS2PIP is pinned only in the Docker specification
(`env/docker-rescore.yml`), because the native fragment predictor is the default
and MS2PIP is opt-in.

You usually do not have to name the interpreters at all. A field set to `"auto"`,
which is what `configs/examples/*.json` use, resolves through
`MUMDIA_PYTHON_<ROLE>`, `MUMDIA_PYTHON`, `CONDA_PREFIX`, `VIRTUAL_ENV`, then
`python3` and `python` on `PATH`, accepting a candidate only after it imports what
that role's workers import, so activating the environment is normally enough. Name
an absolute path when a machine has several candidates and you want to pin one; an
explicit path is never second-guessed.

Notes:

- The `nn_torch` rescorer requires `rescore.python` to point at an interpreter
  with torch (validated at `config.rs:1073-1082`, and re-checked before compute in
  `rust/mumdia/crates/mumdia/src/stages/run.rs:68-75`); `py312_mumdia` satisfies this.
- Use `deeplc_mt`, not the older `deeplc_multitask`, for the fine-tune: prediction
  crashes in `deeplc_multitask` on this machine (`CLAUDE.md`, ML predictors
  section). `deeplc_finetune.py` pins OpenMP/BLAS threads to avoid an
  oversubscription crash specific to this env
  (`scripts/deeplc_finetune.py:22-28`).
- The `ms2rescore` environment named in `CLAUDE.md` is not present here. mokapot
  and MS2PIP are importable in `py311_workshop` and in `deeplc_mt` if you want the
  mokapot rescorer or the MS2PIP predictor. The best-workflow config in Section 4
  uses `nn_torch` plus the imported library's own fragment intensities, so this
  environment is not required for it.

### Locating and creating environments

List what exists and confirm an interpreter's contents:

```
conda env list
conda run -n mumdia-deeplc python -c "import deeplc, torch; print(deeplc.__version__, torch.__version__)"
```

`mumdia doctor` is the check that matters, because it resolves the interpreters
exactly as a run does and then reports what it found: the path, which rule
resolved it, the versions of the packages whose version changes results, and
whether the worker scripts are where the engine will look.

```
mumdia doctor --config configs/examples/diann-library.json
```

### Environment to config-field mapping

| Config field | Value on this machine | Effect |
|---|---|---|
| `rescore.python` | `py312_mumdia` python | interpreter for the `nn_torch` sidecar |
| `rescore.classifier` | `nn_torch` | selects the PyTorch semi-supervised rescorer |
| `predict_frag.deeplc_python` | `deeplc_mt` python | interpreter for the DeepLC fine-tune / prediction |
| `rt_im_train.finetune_deeplc` | `true` | enables the per-run DeepLC fine-tune (requires `deeplc_python`, enforced at `rust/mumdia/crates/mumdia/src/stages/run.rs:62-67`) |
| `predict_frag.ms2pip_python` | unset | would point at an env with MS2PIP; not used by the library run |

## 2. Converting vendor files to centroided mzML

MuMDIA reads only mzML. `convert` (Stage 0) is the sole point that touches a
vendor format, and it goes through the `mzdata` crate built with the `mzml`
feature only (`docs/04_convert.md`; `CLAUDE.md` build notes). Thermo `.raw`,
Bruker `.d`/TDF, and SCIEX `.wiff` are not read natively (they are on the
beyond-MVP roadmap, `CLAUDE.md`). Convert them first with ProteoWizard
`msconvert`.

Produce centroided mzML using the vendor peak-picking filter:

```
msconvert sample.raw --mzML --64 --zlib --filter "peakPicking vendor msLevel=1-"
```

- `peakPicking vendor msLevel=1-` applies the instrument vendor's centroiding to
  all MS levels. Vendor peak-picking is higher quality than a generic algorithm.
- For Bruker `.d`, point `msconvert` at the `.d` directory; vendor peak-picking
  requires the Bruker vendor libraries in the ProteoWizard build.
- MuMDIA also centroids profile spectra itself if it receives profile data (local
  maxima plus parabolic apex, `rust/mumdia/crates/mumdia/src/stages/convert.rs`),
  and synthesizes a full-range isolation window for zero-bounded AIF/all-ion
  scans, so a profile mzML still runs. Centroiding at conversion is preferred
  because the vendor algorithm is better and the files are smaller.

## 3. The pre-built E. coli test library (`lib/`)

`lib/` holds a ready-to-use imported library for the E. coli AIF test file, so
you can run the library-input path without regenerating anything. All files are
gitignored (large binaries kept on disk).

| File | Rows | Role |
|---|---|---|
| `lib/lib_precursors.parquet` | 1,691,048 | Precursor library, `predicted_irt` = raw DIA-NN iRT |
| `lib/lib_precursors_ft.parquet` | 1,691,048 | Prior DeepLC-fine-tuned output; useful for direct stages or runs with fine-tuning disabled |
| `lib/lib_fragments.parquet` | 19,998,666 | b/y fragment m/z + predicted intensities, keyed by `candidate_id` |
| `lib/seed_psms.parquet` | 144,821 | Saved search-seed PSMs from the run that built the library |
| `lib/seed_psms.parquet.masscal.json` | 1 | Per-run mass-recalibration sidecar for that seed |

The precursor and fragment schemas match the library-input contract consumed by
the fragment index (`rust/mumdia/crates/mumdia/src/index.rs:55-71`):
`candidate_id, peptidoform_id, base_peptide_id, peptidoform, charge,
precursor_mz, predicted_irt, label, protein, n_fragments` for precursors, and
`candidate_id, mz, predicted_intensity, name, ion_type, ordinal, frag_charge`
for fragments. `candidate_id` is contiguous and precursors are sorted by
`precursor_mz`, the preconditions the index build checks on load.

### Provenance

The library was built with the license-clean DIA-NN recipe (`CLAUDE.md`, DIA-NN
library recipe): a DIA-NN predicted E. coli library was imported with
`scripts/import_diann_lib.py`, then reverse decoys were added and the table
re-sorted and re-indexed with `scripts/make_reverse_decoys.py`. Each target
carries a paired `DECOY_`-prefixed reverse decoy, and contaminant entries are
kept (visible as `Cont_` proteins). MuMDIA does not contain or redistribute
DIA-NN; the user runs DIA-NN under their own academic license to predict the
source library.

### Raw vs `_ft`: which precursor file to use (easy to get wrong)

Both precursor files have identical schemas and row counts and differ only in the
`predicted_irt` column:

- `lib_precursors.parquet` (raw) carries the DIA-NN iRT scale (small values,
  roughly -50 to +150; e.g. the first target `YGC[Carbamidomethyl]AE` has
  `predicted_irt = -22.9`).
- `lib_precursors_ft.parquet` (`_ft`) has that column overwritten by a DeepLC
  fine-tune, on a different, much larger scale (the same peptide reads
  `predicted_irt = 1413.6`).

You can tell them apart at a glance with `mumdia inspect lib/lib_precursors.parquet`
vs `..._ft.parquet` and comparing the `predicted_irt` magnitudes. Raw DIA-NN iRT
is locally noisy, so if fine-tuning is disabled and you consume a precursor
library directly, the `_ft` file is the one to use: `rt-im-train` reads
`predicted_irt` as-is to fit the RT
calibration and set per-candidate windows
(`rust/mumdia/crates/mumdia/src/stages/rt_im_train.rs:65-83`), and raw iRT
gives poor windows.

Interaction with the per-run fine-tune (the best-workflow config sets
`rt_im_train.finetune_deeplc = true`): when the fine-tune is on, `run` fine-tunes
DeepLC on this run's confident seed PSMs and writes a new output table with
replaced `predicted_irt` for every standard peptidoform before RT calibration
(`rust/mumdia/crates/mumdia/src/stages/run.rs:242-280`;
`scripts/deeplc_finetune.py:156-159`). The input file's `predicted_irt` is then
only a fallback for non-standard peptidoforms (`deeplc_finetune.py:95`, `:156`),
so raw and `_ft` may converge when the fine-tune is on. For a clean
reproduction, pass the raw table when `finetune_deeplc = true` and let `run`
create and record its own `_ft` output. The pre-existing `_ft` distinction matters
when `finetune_deeplc = false`.

### The seed and mass-cal sidecars

`lib/seed_psms.parquet` is the search-seed artifact saved from the full run that
built this library (144,821 candidate PSMs: `peptidoform, charge, label, score,
spectrum_q, observed_rt, predicted_irt, matched_peaks, scan_index`, ...). The
`run` orchestrator recomputes its own seed each time and does not read this file
(`run.rs` computes search-seed internally), so it is provided for standalone
stages (`rt-im-train`, `align`, `mbr`, `audit`) and for inspection.
`lib/seed_psms.parquet.masscal.json` is the per-run fragment mass recalibration
written by `search-seed` (`{"frag_ppm_offset": 0.79, "frag_tol_ppm": 19.2,
"n_dev": 65678}`); it likewise belongs to that prior run.

## 4. Worked end-to-end runs

Run from the repository root. Unlike section 0, these need data you supply:
`fasta/ecoli_22032024.fasta` and `mzml_files/LFQ_Orbitrap_AIF_Ecoli_01.mzML`, a
3.4 GB AIF file from ProteomeXchange. Both directories are gitignored, so a fresh
clone has neither.

### 4a. Zero-dependency native FASTA run

No Python, no sidecars. `--profile dia` selects the Extended feature set,
rolling-window apex counting (window 5), and the RT prior
(`config.rs:1109-1113`). Native predictors and the native rescorer are used
throughout.

```
mumdia run \
  --fasta   fasta/ecoli_22032024.fasta \
  --mzml    mzml_files/LFQ_Orbitrap_AIF_Ecoli_01.mzML \
  --out-dir out_native \
  --profile dia \
  --top-peaks-ms2 300
```

Expected result (documented, not re-verified here): approximately 1,213 target
peptides (precursor rows in `out_native/peptides.tsv`) at 1% FDR. This path is
byte-reproducible: two runs on the same input produce an identical
`peptides.tsv`, and so do runs on different operating systems, which `ci/smoke.sh`
and the `smoke-cross-platform` CI job assert.

`--top-peaks-ms2 300` belongs to this one chimeric AIF acquisition and must not be
carried elsewhere: on a 50-window Orbitrap DIA run the same cap cost 60% of the
peptides (`docs/04_convert.md`).

### 4b. Best-workflow library run

Library-input mode plus the local sidecar config. `run` skips
digest/peptidoforms/predict-frag and consumes the prebuilt library
(`main.rs:208-215`). The config `configs/examples/diann-library.json` turns on:
Extended features, `gate_min_score = 0.2`, rolling-window apex (5) and RT prior
(120 s), the per-run DeepLC fine-tune, the `nn_torch` rescorer with
`rescore.strict = true`, an RT-window multiplier of 1.5, and
`quant.q_filter = run_psm_q`. Both interpreters are `"auto"`, so they are
discovered from `MUMDIA_PYTHON_*`, an activated environment, or `PATH`, and a
candidate is accepted only if it can import what the worker imports; name an
absolute path instead if you prefer to pin one. Use a new output directory:
`run` has no cache/resume and does not clear stale optional files from a reused
directory.

```
mumdia run \
  --lib-precursors lib/lib_precursors.parquet \
  --lib-fragments  lib/lib_fragments.parquet \
  --mzml mzml_files/LFQ_Orbitrap_AIF_Ecoli_01.mzML \
  --out-dir out_lib \
  --config configs/examples/diann-library.json \
  --top-peaks-ms2 300
```

The raw precursor table is intentional: enabled fine-tuning writes a new
`fragment_library_precursors_ft.parquet` in the output directory and rebinds
downstream stages to it without modifying the input. The explicit conversion cap
is also essential to this reproduction. Both `run` and standalone `convert`
otherwise default to uncapped MS2 spectra; `search_seed.top_n_peaks=300` is a
separate seed-only probe cap. Do not carry `--top-peaks-ms2 300` to your own
data. It is specific to this chimeric AIF file, and on a 50-window Orbitrap DIA
run the same value discarded 78.6% of MS2 peaks and cost 60% of the peptides.
See docs/04_convert.md ("Choosing `--top-peaks-ms2`") before setting any cap.

Expected result (documented, not re-verified here): approximately 10,300
precursor-shaped report rows at peptide q <= 1% with the `nn_torch` rescorer.
Runtime is roughly 10 minutes
on this machine (the fine-tune dominates; without it the chain is under 3
minutes). This run is nondeterministic: `deeplc_finetune.py` sets no torch/numpy
seed, so the fine-tuned iRT and the final count vary slightly between runs
(`CLAUDE.md`, ML predictors section). NnTorch itself seeds NumPy/Torch but is not
guaranteed bit-deterministic.

The `run` stdout ends with a one-line summary, for example:

```
MuMDIA: <N> precursor rows, <M> protein groups at peptide/PG q <= 0.01 (rescorer used: nn_torch)
```

`N` is the number of precursor rows written to `peptides.tsv` (the report unit is
peptidoform + charge, filtered to targets with peptide q <= the threshold;
`rust/mumdia/crates/mumdia/src/stages/report.rs:100-122`, threshold from
`quant.q_threshold`, default 0.01, `config.rs:824`; summary printed at
`run.rs:469-472`).

`peptides.tsv` is therefore not a stripped-peptide count and is not controlled by
`precursor_q`. Confirm the actual classifier and model identity in
`out_lib/psms_scored.parquet.report.json`; strict mode should make any sidecar
failure fatal, and the report is the source of truth.

## 5. Acceptance and smoke check

### Fast smoke checks

Start with `mumdia doctor --config configs/examples/diann-library.json`; it validates
imports without launching a full run.

Do not use `--max-spectra 20000` with the fine-tuning recipe as a sidecar smoke
test. `--max-spectra` reads the file head, and this run's early gradient contains
no confident fine-tune anchors at that size, so the test can fail before it
meaningfully exercises DeepLC or NnTorch. The flag cannot select a mid-gradient
offset.

For a pipeline smoke test, externally prepare a centroided mid-gradient mzML
slice containing real peptide signal and run the same Section 4b command against
that file, still with `--top-peaks-ms2 300` and a fresh output directory. A
plumbing-only alternative is a scratch config with
`rt_im_train.finetune_deeplc=false`; do not compare its identification count to
the validated full-run target.

Verify the requested sidecars through their logs and
`psms_scored.parquet.report.json`, then check that `peptides.tsv`,
`proteins.tsv`, `manifest.json`, and the primary Parquets exist.

### Full-run regression guard

Treat the Section 4 counts as regression targets on
`LFQ_Orbitrap_AIF_Ecoli_01.mzML`:

- The fixture smoke test (`ci/smoke.sh`), which needs no external data: 136
  assertions. Run this first; it fails faster and more specifically than any count
  below.
- Native FASTA run (`--profile dia`): approximately 1,213 precursor-shaped
  report rows passing peptide q <= 1%, deterministic (exact match expected).
- Best-workflow library run (`configs/examples/diann-library.json`, strict `nn_torch`,
  explicit `--top-peaks-ms2 300`): approximately 10,300 rows under the same
  report definition. Treat this as a band because DeepLC fine-tuning, although
  seeded from `rng_seed`, is not bit-for-bit reproducible
  and NnTorch is seeded but not bit-deterministic.

Count the report rows, which are one per `(peptidoform, charge)` but selected by
`peptide_q_value`:

```
wc -l out_lib/peptides.tsv   # subtract 1 for the header
```

A native count far below 1,213 or a library count far below 10,300 indicates a
regression (or, for the library run, a missing or misconfigured sidecar). These
numbers are documented from prior runs and were not re-verified in this document.
The fixture smoke test is the part that IS verified on every commit, and it is the
one to trust when the two disagree about whether the engine works.

Every run records what produced it: `manifest.json` carries the engine version, the
short commit with a `-dirty` marker when the worktree was dirty, the commit date,
the full command line, and a blake3 hash of every input. Quote that stamp with any
number you report.
