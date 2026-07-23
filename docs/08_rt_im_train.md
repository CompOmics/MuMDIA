# rt-im-train (Stage B): RT calibration + DeepLC fine-tune

> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

Stage B (`mumdia rt-im-train`, PLAN.md Stage B) turns the run-independent
predicted iRT carried on each library candidate into a per-run predicted
retention time in seconds, and derives a per-candidate RT acceptance window that
the extractor uses to bound its scan search. It is the bridge between the
library (built once, run-independent) and this run's chromatography.

The stage does two things:

1. Fit a calibration map `predicted_irt -> observed_rt` from the confident seed
   PSMs of this run (linear least squares always; a LOESS local-linear smoother
   when configured and enough anchors are present).
2. Set an RT half-window from the residual distribution of that fit
   (residual percentile times a multiplier), and emit `rt_lo`/`rt_hi` around the
   calibrated RT of every library candidate.

An optional pre-step, wired into the `run` orchestrator rather than into this
stage, fine-tunes the DeepLC multitask model on the same confident seed PSMs and
rewrites the library's `predicted_irt` before Stage B reads it. The fine-tune is
a Python sidecar and is nondeterministic. It is default-off.

Ion mobility (IM) is stubbed. The MVP is 3D, so the IM columns are always
written null; there is no IM calibration or IM window.

## Files

| path | role |
|---|---|
| `rust/mumdia/crates/mumdia/src/stages/rt_im_train.rs` | The stage. Joins iRT to seed PSMs, fits calibration, computes windows, writes `run_windows.parquet` + `cal.json`. |
| `rust/mumdia/crates/mumdia/src/calibrate.rs` | Calibration math: `linear_fit`, the `Loess` local-linear smoother, and `percentile`. Shared, no external deps. |
| `rust/mumdia/crates/mumdia/src/stages/run.rs` | Orchestrator. Runs the optional DeepLC fine-tune between `search-seed` and this stage, then calls `rt_im_train::run` (see `run.rs:187-219`). |
| `rust/mumdia/crates/mumdia/src/sidecar.rs` | `run_deeplc_finetune` (sidecar.rs:106-129): the file-contract client that invokes the fine-tune worker. |
| `scripts/deeplc_finetune.py` | The fine-tune worker: transfer-learns DeepLC 4.0 on the seed and rewrites `predicted_irt` in the library parquet. |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `RtImTrainConfig` (config.rs:362-427), `CalibrationMethod` enum (config.rs:66-77), and the load-time validation that rejects `calibration_method=none` (config.rs:1067-1073). |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | `artifact::RUN_WINDOWS = ("run_windows", 1)` (schema.rs:16). |

## Inputs and outputs

### Consumed

**`fragment_library_precursors.parquet`** (the run-independent library; in
library-input mode this is the imported speclib, and when `finetune_deeplc` is
set it is the `_ft` rewrite). Columns read (rt_im_train.rs:34-35):

| column | type | use |
|---|---|---|
| `candidate_id` | u32 | join key; also the output row identity |
| `predicted_irt` | f32 | the value calibrated to observed RT |

The library is the single source of truth for iRT (rt_im_train.rs:31-32): both the
training anchors and the applied calibration read `predicted_irt` from this same
table, so a patched or fine-tuned library iRT is used consistently for fit and
apply.

**`seed_psms.parquet`** (from `search-seed`). Columns read (rt_im_train.rs:44-49):

| column | type | use |
|---|---|---|
| `candidate_id` | u32 | join to library `predicted_irt` |
| `base_peptide_id` | u32 | best-per-peptide grouping key |
| `spectrum_q` | f64 | confidence gate (`< q_train`) |
| `score` | f64 | picks the best PSM within a `base_peptide_id` |
| `observed_rt` | f64 | the calibration target (seconds) |
| `label` | str | only rows equal to `"target"` anchor the fit |

### Produced

**`run_windows.parquet`** (`artifact::RUN_WINDOWS`, version 1). One row per
library candidate. Columns (rt_im_train.rs:182-193):

| column | type | meaning |
|---|---|---|
| `candidate_id` | u32 | library candidate identity |
| `rt_pred_cal` | f64 | calibrated predicted RT (seconds) |
| `rt_lo` | f64 | `rt_pred_cal - width` (window lower bound, seconds) |
| `rt_hi` | f64 | `rt_pred_cal + width` (window upper bound, seconds) |
| `im_pred_cal` | f64, nullable | always `None` (3D MVP) |
| `im_lo` | f64, nullable | always `None` |
| `im_hi` | f64, nullable | always `None` |

**`cal.json`** (side artifact, written with `mumdia_io::json::write_json`,
rt_im_train.rs:196-208). Fields:

| field | meaning |
|---|---|
| `method` | `"loess"` if the LOESS path was used, else `"linear"` |
| `slope`, `intercept` | the linear fit coefficients (always computed, even under LOESS) |
| `w_rt` | the global RT half-window (seconds) |
| `p_rt` | the residual percentile used |
| `multiplier` | `rt_window_multiplier` |
| `n_train` | number of calibration anchors |
| `calibration_status` | `"loess"`, `"linear"`, or `"fallback_fixed"` |

**`<run_windows>.report.json`** (`ArtifactReport`, rt_im_train.rs:215-227) records
rows, content hash, params (`q_train`, `p_rt`, `method`), and stats (`n_train`,
`w_rt`, `calibration_status`).

## How it works

The stage entry point is `run(p: RtImTrainParams)` (rt_im_train.rs:28).

### 1. Join iRT to candidates

Read the library precursors and build `irt_by_cid: HashMap<u32, f64>` mapping
`candidate_id -> predicted_irt` (rt_im_train.rs:33-39). This is the only source of
iRT; both training and application use it.

### 2. Select calibration anchors

Read the seed PSMs and build `best_per_pep: HashMap<u32, (score, irt, rt)>`,
keyed by `base_peptide_id` (rt_im_train.rs:51-69). A seed row enters the pool only
if:

- `spectrum_q < q_train` (rt_im_train.rs:53), and
- `label == "target"` (rt_im_train.rs:58); a decoy anchor would inject a random
  iRT/RT pair into the fit, and
- its `candidate_id` resolves to a library `predicted_irt` (rt_im_train.rs:61).

Within a `base_peptide_id`, the highest-`score` PSM wins (rt_im_train.rs:66-67), so
each confident peptide contributes exactly one `(predicted_irt, observed_rt)`
anchor. The anchors are then split into parallel vectors `train_irt`/`train_rt`
with `n_train = train_irt.len()` (rt_im_train.rs:70-72).

### 3. Fit the calibration map

A linear fit is always computed: `let (slope, intercept) = linear_fit(&train_irt,
&train_rt)` (rt_im_train.rs:76). LOESS is used only when the method is
`CalibrationMethod::Loess` and there are at least `min_seed_for_calibration`
anchors (rt_im_train.rs:77-78); it is fit with `Loess::fit(&train_irt, &train_rt,
loess_span, 200)` (200 grid points, rt_im_train.rs:80).

`predict` is a closure (rt_im_train.rs:85-90): `loess.predict(irt)` when a LOESS
model exists, else `slope * irt + intercept`. The linear coefficients are
therefore both the primary map (Linear method) and the extrapolation fallback
(LOESS outside its training range; see below).

The math:

- **`linear_fit`** (calibrate.rs:6-28) is ordinary least squares. With
  `n = xs.len()`: `slope = (n*Sxy - Sx*Sy) / (n*Sxx - Sx*Sx)` and
  `intercept = (Sy - slope*Sx) / n`. Degenerate guards: fewer than 2 points
  returns `(0, mean(ys))` (a constant map, calibrate.rs:8-16); a near-zero
  denominator (all `x` equal) returns `(0, Sy/n)` (calibrate.rs:22-24).
- **`Loess::fit`** (calibrate.rs:42-83) first computes the global linear fit as
  the extrapolation fallback (calibrate.rs:43), sorts the points by `x`
  (calibrate.rs:45-48), and if fewer than 4 points are present just fills the grid
  from the linear line (calibrate.rs:50-66). Otherwise the local window size is
  `k = clamp(ceil(span * n), 3, n)` (calibrate.rs:67) and it evaluates a
  local-linear regression on a uniform grid of `grid_n` points spanning
  `[min(x), max(x)]` (calibrate.rs:69-76).
- **`local_linear`** (calibrate.rs:107-151) selects the `k` nearest anchors by
  walking outward from the insertion point (calibrate.rs:110-127), assigns each a
  tricubic weight `w = (1 - d^3)^3` with `d = |x_i - x0| / dmax` (calibrate.rs:
  131-137), and solves a weighted least-squares line, returning the fitted value
  at `x0`. If the weighted system is degenerate it falls back to the global line
  (calibrate.rs:145-147).
- **`Loess::predict`** (calibrate.rs:87-102) uses the linear fallback for `x` at or
  outside the grid ends (calibrate.rs:89-94) and otherwise linearly interpolates
  between the two bracketing grid nodes found by `partition_point`
  (calibrate.rs:95-101). Interpolating a precomputed grid makes bulk application
  over the whole library cheap.

### 4. Derive the RT window

`min_anchors = min_seed_for_calibration.max(2)` (rt_im_train.rs:97). When
`n_train >= min_anchors`, the absolute residuals `|observed_rt - predict(irt)|`
are formed (rt_im_train.rs:99-103), and the global half-window is `w_rt =
percentile(resid, p_rt) * rt_window_multiplier`, floored at 1.0 second
(rt_im_train.rs:104-105). The status is `"loess"` or `"linear"`. When
`n_train < min_anchors` the stage warns and uses the fixed fallback
`fallback_rt_window_s` with status `"fallback_fixed"` (rt_im_train.rs:108-114).

The anchor-count gate exists for a concrete failure mode documented in the code
(rt_im_train.rs:92-96): with only a handful of anchors a linear fit passes almost
exactly through them, so residuals are near zero, the percentile window collapses
to the 1-second floor, and that floor then discards nearly every true co-elution
downstream. The fixed fallback avoids that collapse.

### 5. Optional adaptive per-region window (default off)

When `adaptive_rt_window` is set (and `n_train >= min_anchors`), the global
`w_rt` is replaced by a per-bin half-width (rt_im_train.rs:120-153). Anchors are
binned into `adaptive_rt_bins` equal-width bins over the calibrated-RT range; each
bin's half-width is its local residual percentile times the multiplier, clamped to
`[rt_window_min_s, fallback_rt_window_s]` (rt_im_train.rs:134-146). Empty bins fall
back to the global `w_rt` (rt_im_train.rs:139-140). Each candidate then takes the
width of the bin its calibrated RT lands in (rt_im_train.rs:165-171). The rationale
(config.rs:393-402): a single fixed window is simultaneously too wide for
well-calibrated regions and too narrow for poorly-calibrated ones. This knob is
part of the sensitivity program and has not passed the entrapment gate, so it stays
default-off.

### 6. Apply to every candidate and write

For each library candidate (rt_im_train.rs:163-180): `cal = predict(irt)`, the width
is either the adaptive per-bin value or the global `w_rt`, and the row is
`(candidate_id, cal, cal - width, cal + width)` with the three IM columns pushed as
`None`. The table is written (rt_im_train.rs:182-193), `cal.json` is written
(rt_im_train.rs:196-208), and the artifact report is emitted (rt_im_train.rs:215-227).

### DeepLC multitask fine-tune (orchestrator pre-step, default off)

This is not part of `rt_im_train::run`; it runs in the `run` orchestrator between
`search-seed` and Stage B, guarded by `cfg.rt_im_train.finetune_deeplc`
(run.rs:187-208). Preflight (`run.rs:63-68`) rejects `finetune_deeplc` unless
`predict_frag.deeplc_python` names a Python interpreter with DeepLC 4.0 multitask.

When enabled, the orchestrator resolves `deeplc_finetune.py`
(`sidecar::resolve_script`), writes a fine-tuned library
`fragment_library_precursors_ft.parquet`, and rebinds `lib_p` to that file so both
Stage B and `extract` read the fine-tuned iRT (run.rs:187-208). The fine-tune is
driven through `run_deeplc_finetune` (sidecar.rs:106-129), whose positional CLI
contract is `deeplc_finetune.py <lib_in> <seed> <lib_out> --epochs E --patience P
--q-train Q --batch B`. The worker runs with `PYTHONUTF8=1` because DeepLC/Keras
crash on the Windows cp1252 console (sidecar.rs:173-182).

Inside the worker (`scripts/deeplc_finetune.py`): the reference set is the
confident target seed PSMs (`label == "target"`, `spectrum_q <= q_train`, standard
residues only), one observed RT per peptidoform (deeplc_finetune.py:97-109). The
batch size auto-scales when `--batch 0`: `min(512, max(16, n_ref // 30))`, so each
epoch runs at least about 30 gradient steps; a fixed large batch underfits small
references (deeplc_finetune.py:111-117). `deeplc.finetune` transfer-learns the
model (deeplc_finetune.py:128), and predictions are made on the DECOY_-stripped
underlying sequence so decoys land on the same iRT scale as targets
(deeplc_finetune.py:44, 135-154). The rewritten column overwrites `predicted_irt`
in place in the output parquet (deeplc_finetune.py:156-159); peptidoforms with no
prediction keep their original iRT (deeplc_finetune.py:156).

The fine-tune sets no torch/numpy seed, so it is nondeterministic across runs
(CLAUDE.md notes this). Its main use is library-input mode, where the base iRT is
the imported DIA-NN library value rather than a native/DeepLC prediction, and a
per-run fine-tune materially tightens the RT window.

## Key types and functions

| name | file:line | what it does |
|---|---|---|
| `RtImTrainParams` | rt_im_train.rs:19-26 | Input struct: seed PSMs path, library precursors path, output windows + cal paths, config, config hash. |
| `rt_im_train::run` | rt_im_train.rs:28-231 | The stage: join iRT, select anchors, fit, window, apply, write. |
| `linear_fit` | calibrate.rs:6-28 | OLS `y = slope*x + intercept` with degenerate-case guards. |
| `Loess` | calibrate.rs:31-37 | Grid-based local-linear smoother; carries a linear fallback for extrapolation. |
| `Loess::fit` | calibrate.rs:42-83 | Sorts anchors, builds a `grid_n`-point local-linear grid, `k = clamp(ceil(span*n),3,n)`. |
| `Loess::predict` | calibrate.rs:87-102 | Grid interpolation inside range, linear extrapolation outside. |
| `local_linear` | calibrate.rs:107-151 | Tricubic-weighted local least squares at one point. |
| `percentile` | calibrate.rs:154-162 | Nearest-rank percentile: sorts a copy, `rank = round(p*(len-1))`. Not interpolated. |
| `CalibrationMethod` | config.rs:66-77 | Enum `{ Loess, Linear, None }`; default `Loess`. `None` is rejected at load. |
| `RtImTrainConfig` | config.rs:362-427 | All Stage B config fields (below). |
| `run_deeplc_finetune` | sidecar.rs:106-129 | Sidecar client: `deeplc_finetune.py <lib_in> <seed> <lib_out> [flags]`. |

## Configuration

All fields live under `rt_im_train` in the config (`RtImTrainConfig`,
config.rs:362-427). The struct is `#[serde(default, deny_unknown_fields)]`, so any
unknown key is a hard load error and every field has the default below. The config
surface was pruned: there are no IM calibration fields (IM is a stub), and
`CalibrationMethod::None` is a foot-gun rejected at load (config.rs:1067-1073) even
though the enum variant still exists.

| field | default | effect |
|---|---|---|
| `calibration_method` | `loess` | `loess`, `linear`, or `none`. `loess` uses the smoother when anchors suffice, else falls back to linear. `none` is rejected by `Config::validate` (config.rs:1067-1073) because the code path silently degrades to linear. |
| `q_train` | `0.01` | Max `spectrum_q` for a seed PSM to become a calibration anchor (rt_im_train.rs:53) and to enter the DeepLC fine-tune reference (`--q-train`). |
| `p_rt` | `0.95` | Residual percentile for the RT half-window (rt_im_train.rs:104). 0.95 keeps ~95% of anchor residuals inside the window. |
| `rt_window_multiplier` | `1.0` | Scales the residual-percentile window (rt_im_train.rs:104). Larger widens the window: higher recall, more interference. |
| `min_seed_for_calibration` | `50` | Minimum anchors to (a) use LOESS instead of linear (rt_im_train.rs:78) and (b) trust the residual window instead of the fixed fallback (via `min_anchors = max(this, 2)`, rt_im_train.rs:97). |
| `loess_span` | `0.3` | LOESS local-fit fraction; passed as `span` to `Loess::fit` (rt_im_train.rs:80). Fraction of anchors in each local window. |
| `fallback_rt_window_s` | `120.0` | Fixed half-window (seconds) when anchors are too few (rt_im_train.rs:113); also the upper clamp for adaptive bin widths (rt_im_train.rs:135). |
| `finetune_deeplc` | `false` | Enable the DeepLC multitask fine-tune pre-step in `run` (run.rs:187). Requires `predict_frag.deeplc_python` (preflight, run.rs:63-68). |
| `finetune_epochs` | `25` | Fine-tune epoch cap (`--epochs`); early stopping usually halts sooner. Used only when `finetune_deeplc`. |
| `finetune_patience` | `10` | Early-stopping patience (`--patience`): epochs without val-loss improvement before stopping. |
| `finetune_batch` | `0` | Fine-tune batch size (`--batch`). `0` auto-scales to the seed size so each epoch has ~30+ steps (config.rs:388-392); a fixed large batch underfits small seeds. |
| `adaptive_rt_window` | `false` | Per-region window instead of one global width (rt_im_train.rs:120-153). Sensitivity-program knob, not yet default-on. |
| `adaptive_rt_bins` | `12` | Number of equal-width calibrated-RT bins for the adaptive window (rt_im_train.rs:126). |
| `rt_window_min_s` | `1.0` | Lower clamp (seconds) for any adaptive half-window (rt_im_train.rs:134); mirrors the 1s floor on the global window. |

## Invariants, determinism, gotchas

- **IM is a stub.** `im_pred_cal`, `im_lo`, `im_hi` are always `None`
  (rt_im_train.rs:177-179). There is no IM calibration, no IM window, and no IM
  config field. Any 4D/diaPASEF work must add both the model and the columns.
- **Only targets anchor the fit.** Decoy seed PSMs are excluded
  (rt_im_train.rs:58); admitting them would inject random iRT/RT pairs. Keep this
  filter if you refactor anchor selection.
- **Library is the single iRT source.** Training and application both read
  `predicted_irt` from the same library table (rt_im_train.rs:31-39, 156-157). When
  the fine-tune rewrites the library, the orchestrator rebinds `lib_p` to the `_ft`
  file (run.rs:205) so both this stage and `extract` see the updated iRT. Do not
  reintroduce a second iRT source.
- **The 1-second floor.** The global window is floored at 1.0s (rt_im_train.rs:105).
  With too few anchors the fit is near-exact, residuals collapse, and this floor
  would discard true co-elutions, which is exactly why the `min_anchors` fixed
  fallback exists (rt_im_train.rs:92-114). Do not remove the fallback branch.
- **`percentile` is nearest-rank, not interpolated** (calibrate.rs:154-162). It
  sorts a copy each call, so it is order-independent, but repeated calls on the
  same data re-sort. This is fine at Stage B sizes.
- **Determinism caveat: HashMap iteration order.** `best_per_pep` is a `HashMap`
  and `train_irt`/`train_rt` are built from `.values()` (rt_im_train.rs:70-71),
  whose order is randomized per process. `linear_fit` sums over that order
  (calibrate.rs:17-20), and floating-point addition is not associative, so `slope`,
  `intercept`, `w_rt`, and the calibrated RTs can differ in the last few ULPs
  between runs. The effect is tiny but real. `Loess::fit` re-sorts by `x`
  internally (calibrate.rs:45-48), so its grid is order-stable, yet it too reuses the
  order-dependent linear fallback. If byte-identical calibration is required, sort
  the anchors by `candidate_id` (or `base_peptide_id`) before summing. CLAUDE.md's
  determinism rule (ordered iteration where floats are summed) applies here.
- **The fine-tune is explicitly nondeterministic.** `deeplc_finetune.py` sets no
  torch/numpy seed (CLAUDE.md), so `finetune_deeplc = true` makes `predicted_irt`,
  and therefore the whole run, non-reproducible. Treat it as an accuracy lever, not
  a deterministic default.
- **`slope`/`intercept` are always emitted in `cal.json` even under LOESS**
  (rt_im_train.rs:76, 197-208). They are the LOESS extrapolation fallback, not dead
  values; do not assume they were unused when `method == "loess"`.
- **`CalibrationMethod::None` still exists but is rejected** at config load
  (config.rs:1067-1073). The stage would otherwise fall through to the linear path
  because `use_loess` matches only `Loess` (rt_im_train.rs:77). This fallthrough is
  a known correctness wart (see CLAUDE.md "Correctness").
- **Report schema version is 1** (`artifact::RUN_WINDOWS`, schema.rs:16). Bump it if
  the column set changes.

## How to extend / modify

- **Add IM (4D).** Populate `im_pred_cal`/`im_lo`/`im_hi` (currently `None` at
  rt_im_train.rs:177-179) from an IM calibration analogous to the RT path (an
  IM2Deep-style model), and add IM config fields to `RtImTrainConfig`. The output
  columns already exist and are nullable, so downstream reads survive the
  transition. `extract` and the IM feature families must then consume them.
- **Change the calibration model.** Extend `CalibrationMethod` (config.rs:66-77) and
  branch in the `predict` closure setup (rt_im_train.rs:77-90). Keep `linear_fit` as
  the universal fallback so a degenerate anchor set never panics. New models belong
  in `calibrate.rs` next to `Loess`, with unit tests like `loess_tracks_nonlinear`
  (calibrate.rs:176-183).
- **Tune the window policy.** The residual-percentile logic is localized at
  rt_im_train.rs:97-153. New window strategies (for example a symmetric-vs-asymmetric
  window, or a learned per-charge width) should be config-gated and default-off, then
  validated against the entrapment holdout before becoming a default, per the
  sensitivity program (`sensitivity_plan/NEXT_STEPS.md`).
- **Make calibration byte-deterministic.** Sort the anchors deterministically (by
  `candidate_id`) before building `train_irt`/`train_rt` at rt_im_train.rs:70-71, so
  `linear_fit`'s summation order is fixed.
- **Swap the fine-tune worker.** The contract is purely the CLI and the parquet
  columns (`sidecar.rs:106-129`, `scripts/deeplc_finetune.py`). A replacement worker
  need only accept `<lib_in> <seed> <lib_out>` plus the four flags and rewrite the
  `predicted_irt` column. Preserve the DECOY_-stripping behavior
  (deeplc_finetune.py:44, 135-156) so decoys stay on the target iRT scale, and set a
  seed if you want the fine-tune to be reproducible.
