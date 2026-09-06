# Troubleshooting index: symptom -> cause -> fix
> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

A single lookup for the failure modes that are easy to hit but hard to diagnose,
gathered from docs 01-14, `CLAUDE.md`, and measured runs. Many of these fail
quietly (a silent fallback, a nondeterministic wobble, a void result on a
truncated file, a count that is wrong in the right units) rather than with a
clean error, so the table maps the observed symptom to the underlying cause and
the concrete fix. Code behaviour is cited to `file:line`; the referenced source
is the source of truth if it has moved.

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
| Peptide count is a fraction of expectation while the empirical decoy fraction still sits at the target | `--top-peaks-ms2` truncated the MS2 peaks at convert time and baked the truncation into the spectra artifact (`convert.rs:76-79`) | Reconvert uncapped or with a much larger cap; the right cap is acquisition-specific, not a universal preset |
| `mumdia audit` attributes most missed peptides to `NO_PEAK_GROUP` at `candidate_generated` | The same peak truncation leaves too few surviving fragments to satisfy `extract.presence_min_fragments` (`extract.rs:1926-1931`) | Raise or remove `--top-peaks-ms2` and reconvert; do not lower `presence_min_fragments` to compensate |
| Every modified form of a peptide is missing from the output; `precursor_q` reports exactly 1.000 precursors per peptide | `compete.group_by = base_peptide` keys the stripped sequence, not the precursor (`compete.rs:88`), and deletes all but the top-scoring sibling before rescore (`compete.rs:319-340`) | Set `compete.group_by = peptidoform_charge` (`compete.rs:93-98`); required for any PTM search |
| `cal.json` reports a small `rt_residual_abs_median_s` but RT windows still miss peptides | The residual is measured on the same anchors the calibration was fitted to (`rt_im_train.rs:137`, `rt_im_train.rs:177-185`), so it is in-sample | Treat it as a fit diagnostic; size external RT tolerances from an out-of-sample comparison |
| RT windows in a PTM search behave as if the modification were absent | The imported library gave every modform of a stripped peptide the same `predicted_irt` | Check per-stripped-peptide variance of `predicted_irt` in the library; re-predict iRT per peptidoform |
| A single-file run spends over half an hour before extract starts | `rt_im_train.finetune_deeplc = true` fine-tunes and then re-predicts iRT over the whole library on every file | Fine-tune once into the library's `predicted_irt`, then run with `finetune_deeplc = false` and the per-run LOESS |
| `OSError: [WinError 1114] ... Error loading "...\torch\lib\c10.dll"` from a DeepLC worker | numpy/pyarrow were imported before `deeplc`; DeepLC 4.x is torch-backed and the wrong order aborts torch DLL init | Keep `import deeplc` first (`scripts/deeplc_worker.py:26`); the ordering is load-bearing, do not sort those imports |
| `mumdia doctor` is green, then the run crashes at the DeepLC fine-tune on a missing import | doctor probed only part of what `deeplc_finetune.py` imports | Current doctor probes `deeplc,numpy,pandas,pyarrow,torch,psm_utils` (`main.rs:370`); re-run it after any env change |
| `Parquet error: Disabled feature at compile time: zstd` | A parquet written outside `mumdia-io` used zstd; the parquet crate is built without that codec | Write SNAPPY (`table.rs:205`); polars defaults to zstd, so pass the compression explicitly |
| `column '<name>' is not utf8` (for example `column 'peptidoform' is not utf8`) | The string column is arrow `large_utf8`, the polars default | Cast string columns to arrow `utf8` before writing (`table.rs:511`) |
| `library precursor row N has candidate_id M but candidate_id must be the contiguous range 0..ncand` | An externally built library has non-row-aligned candidate ids | Reindex so `candidate_id` is `0..ncand` in precursor row order (`index.rs:112-125`) |
| `library precursors must be ascending by precursor_mz` | The library was not sorted by `precursor_mz`, which the fragment index's candidate-window search assumes | Sort precursors by m/z before assigning `candidate_id` (`index.rs:215-231`); fragments need valid ids but no sorted order |
| An `nn_torch` rescore takes far longer than its measured per-PSM rate predicts | The feature matrix marginally exceeded `MUMDIA_NN_STREAM_GB` (default 4) and fell to the disk-backed streaming memmap | Compute `n_psms x n_features x 4 bytes` up front and raise `MUMDIA_NN_STREAM_GB` if RAM allows (`nn_rescore_worker.py:299`) |
| `peptides.tsv` / `proteins.tsv` only at the root of a `run-experiment` output tree, none per run | The experiment-wide report is the only valid TSV unit under a pooled rescore: the grouped q columns are assigned to each group's experiment-wide winner | Read the root pair (one quantity column per run, `n_runs`); for per-run counts use the split `scored.parquet` on `run_psm_q`; `mumdia report --experiment-dir` rewrites the pair at another threshold |
| Per-run counts on `peptide_q_value` / `precursor_q` / `pg_q_value` are roughly `1/n_runs` of expectation after an experiment-wide rescore | Those grouped q columns are assigned only to each group's single winning row (`rescore.rs:721-728`) and the grouping spans the whole experiment | Use `run_psm_q` as the per-file unit after an experiment-wide rescore |
| Per-run quant gated on a different column than `quant.q_filter` names | `run-experiment` overrides `q_filter` and gates on the pooled `q_value` (`run_experiment.rs:490-498`) | Expected; the override is now logged as a warning. Run `quant` standalone to choose the column |
| Pooling more runs into one rescore did not tighten q | `q = (decoys + 1) / max(1, targets)` (`fdr.rs:38`) is scale-invariant under replicating the population; the `+1` makes a larger pool marginally looser | Do not attribute per-run count changes to pool size; sub-batch to fit RAM instead |

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

**DeepLC import order and the torch DLL crash.** `scripts/deeplc_worker.py` must
`import deeplc` before numpy and pyarrow (`scripts/deeplc_worker.py:26`). DeepLC
4.x is torch-backed, and on Windows importing numpy (and the pyarrow that follows
it) first aborts torch's DLL initialisation outright:

```text
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

`deeplc_finetune.py` already ordered its imports this way; the prediction worker
deferred `import deeplc` into `main()`, which put it after the module-level numpy
and pyarrow and reproduced the crash. The ordering is load-bearing, not
stylistic, so do not sort those imports. The bug stayed latent for a long time
because imported-library mode skips `predict-frag` entirely: only a FASTA-mode
library build reaches the native RT-prediction path.

**A green doctor that still crashes at fine-tune.** `mumdia doctor` probes the
DeepLC interpreter for `deeplc,numpy,pandas,pyarrow,torch,psm_utils`
(`main.rs:370`), because `deeplc_finetune.py` imports pyarrow, torch and
psm_utils on top of deeplc itself. A narrower probe let a green doctor precede a
crash at the fine-tune step, which on an experiment-wide batch is discovered long
after the run was launched. In practice DeepLC 4.x pulls torch and psm-utils
itself, so the extra names usually catch a missing pyarrow; the check asserts
what the scripts import rather than what the dependency tree implies. Re-run
doctor after any environment change instead of trusting an earlier green result.

## Sidecar interpreter not found, or the wrong one used

**Symptom.** `... is required by this configuration but no usable interpreter was
found`, or a run that reaches rescore and fails on a missing module.

**Cause and fix.** The role's field is absent or `"auto"` and nothing on the
machine could import what the worker imports. The message lists every place that
was searched and every module that was required. In order of directness: name the
interpreter in the config; export the role's variable
(`MUMDIA_PYTHON_RESCORE`, `MUMDIA_PYTHON_DEEPLC`, `MUMDIA_PYTHON_MS2PIP`,
`MUMDIA_PYTHON_MBR`); or activate an environment so `CONDA_PREFIX` or
`VIRTUAL_ENV` points at it. `env/mumdia-rescore.yml` and `env/mumdia-deeplc.yml`
build suitable environments.

**Symptom.** Discovery picks an interpreter you did not expect.

**Cause and fix.** An activated environment (`CONDA_PREFIX`) outranks `PATH`, and
a role-specific variable outranks both. Run
`mumdia doctor --config <your config>`: it reports the resolved path, the
provenance (which rule matched), and the versions found. Pin the path in the
config when the machine has several candidate environments.

**Symptom.** `doctor` is green but a DeepLC run's retention-time windows look
wrong.

**Cause and fix.** Check the DeepLC version `doctor` prints. Below 4.1.1 it warns,
because the 4.0.0a2 multitask preview overfits per-run fine-tuning badly enough to
invert RT-model rankings (see the retention-time section below). That is a
different result, not a slower one.

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

## Peak truncation: `--top-peaks-ms2`

`peaks_of` (`convert.rs:76-79`) keeps the `top_n` most intense peaks of each MS2
spectrum and writes the truncated spectrum into the spectra artifact, so the
truncation is permanent for every downstream stage. `extract` applies no cap of
its own and consumes whatever `convert` wrote. `search_seed.top_n_peaks`
(`config.rs:410-415`) is a separate, non-destructive limit that only bounds seed
index-probing cost. The two are distinct mechanisms but not fully independent:
the seed selects its peaks from whatever `convert` already wrote, so a conversion
cap below `top_n_peaks` also shrinks the seed's input. Above it they do not
interact, which is why moving the conversion cap from 300 to uncapped left seed
output identical (80,474 PSMs, 14,877 confident) while the end-to-end result
changed substantially. Both `convert` and `run` default `--top-peaks-ms2` to `0`
(uncapped).

The right cap is acquisition-specific, and `300` is aggressive on anything but
the chimeric AIF benchmark run it was originally tuned on. On one 50-window
Orbitrap DIA run, `--top-peaks-ms2 300` discarded 78.6% of all MS2 peaks and
truncated 85.5% of spectra, because even the 25th-percentile spectrum holds 572
peaks. On the AIF run only 47.8% of spectra saturate the same cap.

End-to-end on that 50-window file, with only this flag changed:

| Arm | `peptides.tsv` rows @ `peptide_q_value` <= 0.01 | Protein groups | Empirical decoy fraction |
|---|---|---|---|
| `--top-peaks-ms2 300` | 25,425 | 4,554 | 0.99% |
| uncapped | 63,237 | 7,336 | 0.99% |

The decoy fraction is identical in both arms, so this is a sensitivity
difference, not a loosened threshold.

The mechanism is peak-group formation, not scoring. With most peaks gone,
`extract.presence_min_fragments` (default 3) cannot be met and the candidate
returns no peak group (`extract.rs:1926-1931`). `mumdia audit` on the capped arm,
restricted to peptides the reference search confirms are present, put
49,105 of 78,782 (62.3%) at `candidate_generated` with `NO_PEAK_GROUP`; only
5,380 were lost to FDR and 355 to competition. A counterfactual replay against
the uncapped artifact recovered 41,948 of those 49,105 (85.4%). Do not lower
`presence_min_fragments` to compensate: that trades a real fragment requirement
for a truncation artifact.

Fix: reconvert without the cap, or with a cap taken from this run's own peak
census. Recovery is graded rather than binary, so a peak-volume budget can still
be spent deliberately; `docs/04_convert.md` ("Choosing `--top-peaks-ms2`") holds
the peak census, the cap dose-response, and the extraction cost of uncapping, and
`docs/20_sensitivity_and_quantification_playbook.md` holds the pre-flight
saturation check to run before trusting any cap.

## Competition deletes modforms before rescore

**`compete.group_by = base_peptide` is a misnomer.** The enum variant is named
`Precursor`, but the group key built at `compete.rs:88` is
`(base_peptide_id, label_code, 0, peak_rank)`, and `base_peptide_id` is the
stripped-sequence identity (`import_diann_lib.py:137` factorises
`Stripped.Sequence`). `resolve_competition` then keeps only the highest
`prelim_score` member of each group and deletes the rest (`compete.rs:319-340`),
before rescore and before any FDR estimate. Every charge and every modification
variant of one stripped peptide therefore collapses to a single surviving row.

On a modification-rich imported library this deleted 880,464 of 1,890,239
extracted candidates (46.6%). With `compete.group_by = peptidoform_charge`, which
keys `(peptidoform_id, label, charge, peak_rank)` (`compete.rs:93-98`), compete
removed zero rows and precursors per peptide moved from 1.000 to 1.174, against
about 1.126 reported by DIA-NN on comparable data. The peptide count was
unchanged, so the precursor-level key costs nothing at the peptide level.

The observable symptom in a PTM search is that the modified form is simply absent
from the output whenever its unmodified or alkylated sibling scored higher, which
is the usual case. For modification work treat `peptidoform_charge` as required
rather than as a benchmark-gated option. Note also that a `precursor_q` produced
under the default key is a base-peptide count, not a precursor count.

## Retention time: in-sample residuals, fine-tune scope, modform iRT

**`cal.json` residuals are in-sample and optimistic.** `rt-im-train` fits the
LOESS on the seed anchors (`rt_im_train.rs:137`) and then derives the RT window
half-width from a percentile of the residuals of those same anchors
(`rt_im_train.rs:177-185`). The reported `rt_residual_abs_median_s` is therefore
a fit diagnostic, not a prediction-error estimate. Measured on one run, a
reported 6.14 s became p50 17.6 s and p90 146.3 s when the same calibration was
scored out-of-sample against an external search's retention times, roughly 3x
worse at the median and much worse in the tail. Size any externally imposed RT
tolerance from out-of-sample numbers, and do not read a small `cal.json` residual
as evidence that the windows are wide enough.

**Per-file DeepLC fine-tuning is not required.** `rt_im_train.finetune_deeplc =
true` fine-tunes on the run's own seeds and then re-predicts iRT across the whole
library, which on a large library cost 2,166 s of a 5,127 s single-file run
(about 36 minutes). A library whose `predicted_irt` was fine-tuned once and
predicted over every peptidoform, combined with the per-run LOESS calibration
only (`finetune_deeplc = false`), gave median absolute RT residual 6.06 s, MAD
6.11 s, slope 0.9907, intercept 16.4 s, against 6.14 s and 6.18 s with per-file
fine-tuning. Equal or marginally better, at no per-file cost. Fine-tuning is
still the right call when a library's iRT has never been calibrated for the
instrument or gradient at all; it is the repeated re-fitting on every file that
is wasted.

**An imported library may give every modform the same iRT.** Measured on a
modification-expanded imported library, 79.7% of stripped-peptide groups had an
identical raw `predicted_irt` across all their modforms: the modified forms
inherited the unmodified form's retention time and the modification was never
modelled. Spearman correlation between the raw iRT and a properly predicted iRT
was 0.9876 for unmodified peptides but 0.4980 for modified ones. Before trusting
RT windows in a PTM search, check the per-stripped-peptide variance of
`predicted_irt` in the library; if it is zero for most groups, re-predict iRT per
peptidoform.

## External Parquet and library preconditions

Anything written outside `mumdia-io` and then read by the engine has to match the
reader exactly. Two defaults of common Python writers break it:

- **Compression.** `mumdia-io` writes and expects SNAPPY (`table.rs:205`). The
  parquet crate is built with no default features plus `snap`
  (`Cargo.toml:23-26`), so a zstd file fails with
  `Parquet error: Disabled feature at compile time: zstd`. Polars defaults to
  zstd; pass the compression explicitly.
- **String type.** String columns must be arrow `utf8`. A `large_utf8` column
  fails with `column '<name>' is not utf8` (`table.rs:511`). Polars produces
  `large_utf8` by default; cast before writing.

Two structural preconditions on a library are checked explicitly and fail loudly
rather than corrupting results:

- `candidate_id` must be the contiguous, row-aligned range `0..ncand` in the
  precursor table's row order (`index.rs:112-125`). Violating it would misgroup
  fragments.
- Precursors must be ascending by `precursor_mz` (`index.rs:215-231`), because
  the fragment index's `partition_point` candidate-window search assumes it. An
  unsorted import would otherwise silently return wrong candidate windows.

Fragments are grouped by a counting sort over `candidate_id`, so they need valid
ids but not a sorted order. The decoy-builder scripts satisfy both preconditions;
a target-only import fed straight to the engine must satisfy them itself, which
is why `import_diann_lib.py` sorts by `Precursor.Mz` before assigning ids.

## Rescore scale and backend selection

**`nn_torch` silently taking the slow path.** The NN worker picks its backend
from the size of the feature matrix: in memory at or below
`MUMDIA_NN_STREAM_GB` (default 4), a disk-backed streaming memmap above it
(`nn_rescore_worker.py:299`). The threshold is a cliff, not a taper. A 4.31 GB
matrix against the 4.00 GB default fell to the streaming backend and ran much
slower for an 8% overshoot. The matrix is `n_psms x n_features x 4 bytes`, so
compute it before launching and either raise `MUMDIA_NN_STREAM_GB` if RAM allows
or choose the streaming path deliberately.

**Sub-batching a pooled rescore is statistically free.** The streaming backend
measured 0.834 ms/PSM and scales linearly, so a pooled rescore is bounded by RAM
and storage rather than by statistics. `rescore --competed` accepts several
competed tables, stamps `source` with the index of the input table each PSM came
from (`rescore.rs:65-70`, `rescore.rs:108`), and computes a per-source
`run_psm_q` alongside the pooled `q_value` (`rescore.rs:403-408`). Pooling
therefore never costs per-run FDR, and splitting a large experiment into batches
that fit memory does not change what any single run's q means. Batch to fit RAM,
not to chase a q target.

## run-experiment: reporting and q units

**No report stage.** `run-experiment` never invokes report, so there is no
`peptides.tsv` and no `proteins.tsv` anywhere in its output tree. Per-run counts
must come from the split scored tables or from `mumdia report` invoked manually
against them. Their absence is expected, not a failed run.

**Grouped q columns are per-winner and experiment-wide.** `peptide_q_value`,
`precursor_q` and `pg_q_value` are assigned only to each group's single winning
row; every losing sibling keeps 1.0 (`rescore.rs:721-728`). That is correct for
an experiment-wide count, but the grouping spans the whole experiment, so
counting one run's rows on those columns yields roughly `1/n_runs` of the
expected number and is meaningless as a per-file figure. The correct per-file
unit after an experiment-wide rescore is `run_psm_q`.

**Per-run quant ignores `quant.q_filter`.** `run-experiment` overrides the
configured filter and gates each run's quant on the pooled `q_value`
(`run_experiment.rs:490-498`). This is deliberate, and changing `q_filter` does
not select a source in any case, but it was previously silent; it now logs a
warning naming both the configured and the used column. To gate on a different
column, run `quant` standalone against the split scored table.

**Pooling more runs does not tighten q.** `q = (decoys + 1) / max(1, targets)`
(`fdr.rs:38`) is scale-invariant under replicating the population: doubling both
counts leaves the ratio unchanged. The only pool-size term is the `+1`
pseudocount, which makes a larger pool marginally looser, not tighter. Per-run
count differences between a small and a large pool come from the model the
rescorer learned on the pooled data, not from any effect of pool size on the q
estimate.

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
