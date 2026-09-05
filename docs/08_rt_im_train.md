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
   PSMs of this run (a linear least-squares fit whenever at least two anchors
   exist; a LOESS local-linear smoother when configured and enough anchors are
   present). With fewer than two target anchors no trustworthy mapping exists,
   so the calibrated RT is marked unavailable and the windows are left unbounded.
2. Set an RT half-window from the residual distribution of that fit
   (residual percentile times a multiplier), and emit `rt_lo`/`rt_hi` around the
   calibrated RT of every library candidate.

An optional pre-step, wired into the `run` orchestrator rather than into this
stage, fine-tunes the DeepLC multitask model on the same confident seed PSMs and
writes a new precursor-library table with updated `predicted_irt` before Stage B
reads it. The input library is unchanged. The fine-tune is a Python sidecar and
is nondeterministic. It is default-off.

Ion mobility (IM) is stubbed. The MVP is 3D, so the IM columns are always
written null; there is no IM calibration or IM window.

## Files

| path | role |
|---|---|
| `rust/mumdia/crates/mumdia/src/stages/rt_im_train.rs` | The stage. Joins iRT to seed PSMs, fits calibration, computes windows, writes `run_windows.parquet` + `cal.json`. |
| `rust/mumdia/crates/mumdia/src/calibrate.rs` | Calibration math: `linear_fit`, the `Loess` local-linear smoother, and `percentile`. Shared, no external deps. |
| `rust/mumdia/crates/mumdia/src/stages/run.rs` | Orchestrator. Runs the optional DeepLC fine-tune between `search-seed` and this stage, then calls `rt_im_train::run` (see `run.rs:265-314`). |
| `rust/mumdia/crates/mumdia/src/sidecar.rs` | `run_deeplc_finetune` (sidecar.rs:110-155): the file-contract client that invokes the fine-tune worker. |
| `scripts/deeplc_finetune.py` | The fine-tune worker: transfer-learns DeepLC (4.1.1 or newer; older versions are refused) on the seed and writes a new library parquet with replaced `predicted_irt`. |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `RtImTrainConfig` (config.rs:447-512), `CalibrationMethod` enum (config.rs:56-61), and the load-time validation that rejects `calibration_method=none` (config.rs:1336-1342). |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | `artifact::RUN_WINDOWS = ("run_windows", 1)` (schema.rs:16). |

## Inputs and outputs

### Consumed

**`fragment_library_precursors.parquet`** (the run-independent library; in
library-input mode this is the imported speclib, and when `finetune_deeplc` is
set it is the `_ft` rewrite). Columns read (rt_im_train.rs:78-79):

| column | type | use |
|---|---|---|
| `candidate_id` | u32 | join key; also the output row identity |
| `predicted_irt` | f32 | the value calibrated to observed RT |

The library is the single source of truth for iRT (rt_im_train.rs:75-83): both the
training anchors and the applied calibration read `predicted_irt` from this same
table, so a patched or fine-tuned library iRT is used consistently for fit and
apply.

**`seed_psms.parquet`** (from `search-seed`). Columns read (rt_im_train.rs:88-93):

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
library candidate. Columns (rt_im_train.rs:266-277):

| column | type | meaning |
|---|---|---|
| `candidate_id` | u32 | library candidate identity |
| `rt_pred_cal` | f64 | calibrated predicted RT (seconds); `NaN` when calibration is unavailable (fewer than two anchors) |
| `rt_lo` | f64 | `rt_pred_cal - width` (window lower bound, seconds); negative infinity when calibration is unavailable |
| `rt_hi` | f64 | `rt_pred_cal + width` (window upper bound, seconds); positive infinity when calibration is unavailable |
| `im_pred_cal` | f64, nullable | always `None` (3D MVP) |
| `im_lo` | f64, nullable | always `None` |
| `im_hi` | f64, nullable | always `None` |

The unbounded row `(NaN, -inf, +inf)` is materialized by `candidate_window`
(rt_im_train.rs:65-70) whenever no calibrated RT or window width is available; the
infinite bounds make the downstream extractor scan the full isolation-window RT
range (recall-safe) rather than a numeric window.

**`cal.json`** (side artifact, written with `mumdia_io::json::write_json`,
rt_im_train.rs:309-325). Fields:

| field | meaning |
|---|---|
| `method` | `"loess"` if the LOESS path was used, `"linear"` if the linear map was used, or `"unavailable"` when fewer than two anchors exist (rt_im_train.rs:279-285) |
| `slope`, `intercept` | the linear fit coefficients, computed only when at least two anchors exist and therefore serialized as `null` when calibration is unavailable (rt_im_train.rs:286-287) |
| `w_rt` | the global RT half-window (seconds), `null` when the windows are unbounded |
| `p_rt` | the residual percentile used |
| `multiplier` | `rt_window_multiplier` |
| `n_train` | number of calibration anchors |
| `calibration_status` | `"loess"`, `"linear"`, `"fallback_fixed"`, or `"insufficient_anchors_unbounded"` |
| `rt_residual_median_s` | signed median residual `observed_rt - predict(irt)` over the training anchors (residual bias, seconds); computed as `NaN` when calibration is unavailable (rt_im_train.rs:293-306), which serde_json writes to the file as `null` |
| `rt_residual_abs_median_s` | median absolute residual over the same anchors (seconds) |
| `rt_residual_mad_s` | median absolute deviation of the signed residuals about their median (seconds) |

Every "median" among the three is the nearest-rank `percentile(.., 0.5)`
(calibrate.rs:156-164), not an interpolated median.

The three residual fields are in-sample. They are evaluated with `predict` over
exactly the `train_irt`/`train_rt` anchors that trained the fit
(rt_im_train.rs:293-306), and under LOESS those are the same points the smoother
was fit through (rt_im_train.rs:137). They are a fit diagnostic, not an error
estimate. Measured on one run, a reported `rt_residual_abs_median_s` of 6.14 s
corresponded to p50 17.6 s and p90 146.3 s when the same calibrated RTs were
scored against DIA-NN retention times for that run, roughly three times worse at
the median. Size any external RT tolerance from held-out numbers, not from this
field.

**`<run_windows>.report.json`** (`ArtifactReport`, rt_im_train.rs:332-344) records
`logical_name`/`schema_name`/`schema_version` (all from `artifact::RUN_WINDOWS`),
`stage = "rt-im-train"`, rows, the blake3 content hash of `run_windows.parquet`,
params (`q_train`, `p_rt`, `method` as the `Debug` form of `calibration_method`),
stats (`n_train`, `w_rt`, `calibration_status`), and `elapsed_ms`. `model_identity`
is `None` (this stage applies no serialized model, rt_im_train.rs:341). Params
`stats` is a `BTreeMap`, so its key order is stable.

## How it works

The stage entry point is `run(p: RtImTrainParams)` (rt_im_train.rs:72). Standalone
it is invoked as `mumdia rt-im-train --seed-psms <p> --library-precursors <p>
--out-windows <p> --out-cal <p> [--config <p>]` (the `RtImTrain` subcommand,
main.rs:94-105, dispatched at main.rs:517-534); the `run` orchestrator calls the
same `run` function directly (run.rs:307-314). Both call sites build the config
hash and pass it as `config_hash`, but the stage never reads it (see gotchas).

### 1. Join iRT to candidates

Read the library precursors and build `irt_by_cid: HashMap<u32, f64>` mapping
`candidate_id -> predicted_irt` (rt_im_train.rs:80-83). This is the only source of
iRT; both training and application use it.

### 2. Select calibration anchors

Read the seed PSMs and build `best_per_pep: HashMap<u32, (score, irt, rt)>`,
keyed by `base_peptide_id` (rt_im_train.rs:95-120). A seed row enters the pool only
if:

- `spectrum_q`, `score`, and `observed_rt` are all finite and
  `spectrum_q < q_train` (rt_im_train.rs:97-103); a non-finite field or a row at
  or above the threshold is skipped, and
- `label == "target"` (rt_im_train.rs:106); a decoy anchor would inject a random
  iRT/RT pair into the fit, and
- its `candidate_id` resolves to a finite library `predicted_irt`
  (rt_im_train.rs:109-113); a missing or non-finite library iRT is skipped.

Within a `base_peptide_id`, the highest-`score` PSM wins (rt_im_train.rs:117-118),
so each confident peptide contributes exactly one `(predicted_irt, observed_rt)`
anchor. `sorted_anchor_vectors` (rt_im_train.rs:54-60) then sorts the anchors by
`base_peptide_id` and splits them into the parallel vectors `train_irt`/`train_rt`
(rt_im_train.rs:121), with `n_train = train_irt.len()` (rt_im_train.rs:122). Sorting
before any float reduction makes the fit order-stable (see the determinism note in
the gotchas).

### 3. Fit the calibration map

Calibration is attempted only when `calibration_available = n_train >= 2`
(rt_im_train.rs:127): zero or one point cannot define a useful mapping across the
gradient. When available, the linear fit is computed: `let (slope, intercept) =
linear_fit(&train_irt, &train_rt)` (rt_im_train.rs:128-132); otherwise `slope` and
`intercept` are both `NaN`. LOESS is used only when calibration is available, the
method is `CalibrationMethod::Loess`, and there are at least
`min_seed_for_calibration` anchors (rt_im_train.rs:133-135); it is fit with
`Loess::fit(&train_irt, &train_rt, loess_span, 200)` (200 grid points,
rt_im_train.rs:136-140).

`predict` is a closure (rt_im_train.rs:142-150): it returns `NaN` when calibration
is unavailable, otherwise `loess.predict(irt)` when a LOESS model exists, else
`slope * irt + intercept`. The linear coefficients are therefore both the primary
map (Linear method) and the extrapolation fallback (LOESS outside its training
range; see below).

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
- **`local_linear`** (calibrate.rs:107-153) selects the `k` nearest anchors by
  walking outward from the insertion point (calibrate.rs:110-127), assigns each a
  tricubic weight `w = (1 - d^3)^3` with `d = |x_i - x0| / dmax` (calibrate.rs:
  133-139), and solves a weighted least-squares line, returning the fitted value
  at `x0`. `dmax` is the larger of the two window-edge distances, floored at
  `1e-12` to avoid a zero divide (calibrate.rs:128-130), and the weight is exactly 0
  when `d >= 1.0` (calibrate.rs:134-139), so anchors past the window edge do not
  contribute. If the weighted system is degenerate, meaning `sw < 1e-12` (all
  weights vanished) or `sw*swxx - swx^2` near zero (calibrate.rs:146-149), it
  falls back to the global line.
- **`Loess::predict`** (calibrate.rs:87-102) uses the linear fallback for `x` at or
  outside the grid ends, and also when the grid has fewer than 2 nodes
  (calibrate.rs:89-94), and otherwise linearly interpolates between the two
  bracketing grid nodes found by `partition_point` (calibrate.rs:95-101). When the
  two bracketing nodes are within `1e-12` in `x` it returns the lower node's `y`
  rather than interpolating, guarding against a zero divide (calibrate.rs:98-99).
  Interpolating a precomputed grid makes bulk application over the whole library
  cheap.

### 4. Derive the RT window

`min_anchors = min_seed_for_calibration.max(2)` (rt_im_train.rs:157). The half-window
is then chosen by `window_plan(n_train, min_anchors, fallback_rt_window_s)`
(rt_im_train.rs:42-50, called at 158-159), which returns one of three `WindowPlan`
variants (rt_im_train.rs:30-40):

- **`Unbounded`** (`n_train < 2`): no trustworthy iRT to RT mapping exists. The
  stage warns, sets `w_rt = None`, and uses status
  `"insufficient_anchors_unbounded"` (rt_im_train.rs:160-167). Every candidate then
  gets the unbounded window `(NaN, -inf, +inf)` so extraction scans the full
  isolation-window RT range.
- **`Fixed(fallback_rt_window_s)`** (`2 <= n_train < min_anchors`): a linear map
  exists but there are too few anchors to estimate its residual distribution. The
  stage warns and retains the configured broad fixed half-window with status
  `"fallback_fixed"` (rt_im_train.rs:168-175).
- **`Calibrated`** (`n_train >= min_anchors`): the absolute residuals
  `|observed_rt - predict(irt)|` are formed (rt_im_train.rs:177-181), and the global
  half-window is `w_rt = percentile(resid, p_rt) * rt_window_multiplier`, floored at
  1.0 second (rt_im_train.rs:182). The status is `"loess"` or `"linear"`
  (rt_im_train.rs:183).

The anchor-count gate exists for a concrete failure mode documented in the code
(rt_im_train.rs:152-156): with only a handful of anchors a linear fit passes almost
exactly through them, so residuals are near zero, the percentile window collapses
to the 1-second floor, and that floor then discards nearly every true co-elution
downstream. The fixed fallback avoids that collapse; the unbounded case below two
anchors avoids fitting a mapping from a single point at all.

The same in-sample property holds at full anchor counts, less severely. The
residuals are formed on the anchors the fit consumed, and under LOESS on the
points the smoother interpolated, so `w_rt` is systematically narrower than the
RT error on library candidates that were not anchors. `p_rt = 0.95` and
`rt_window_multiplier` absorb part of that gap. Do not read `w_rt` as a measured
95th percentile of prediction error over the whole library.
`window_holdout_frac` (next subsection) replaces this in-sample estimate with a
measured one.

### 4b. Held-out window sizing (`window_holdout_frac`, default off)

In-sample sizing has a second defect beyond optimism: it inverts under model
capacity. A high-capacity RT model that memorizes its anchors gets small
in-sample residuals and therefore a narrow window, exactly when its error on
unseen library peptides is largest. Measured on the AIF benchmark with two
DeepLC versions, in-sample residuals ranked them backwards relative to held-out
residuals (in-sample abs-median 15.9 s vs 24.9 s; held-out median 195.1 s vs
46.4 s for the same two models).

`window_holdout_frac = f` (0 < f <= 0.9) sizes the window out-of-sample:

- anchor peptides with `base_peptide_id % 1000 < round(f*1000)` are held out
  (`is_holdout`, rt_im_train.rs). Keying on the base peptide keeps all
  charge/modform rows of one peptide on the same side; the modulo rule is
  deterministic and duplicated verbatim in `deeplc_finetune.py` so the
  orchestrated fine-tune excludes exactly the same peptides from its reference
  (`--window-holdout-frac`, passed by `run`/`run-experiment`). Without that
  exclusion, adapter memorization leaks into the "held-out" residuals.
- the sizing fit (same method selection as the main fit) is trained on the
  non-held-out anchors only, and `w_rt = percentile(|holdout residuals|, p_rt)
  * rt_window_multiplier`.
- the calibration curve applied to the library still uses every anchor; only
  the width is sized held-out, which is marginally conservative.
- guards: fewer than 20 held-out anchors (`MIN_HOLDOUT_ANCHORS`) or fewer than
  `min_anchors` sizing-train anchors falls back to in-sample sizing with a
  warning; `cal.json.w_rt_sizing` records `"holdout"`,
  `"holdout_fallback_in_sample"`, or `"in_sample"`, with `n_sizing_train`,
  `n_holdout`, `holdout_resid_p_rt_s` (pre-multiplier), and
  `holdout_resid_abs_median_s` alongside.
- mutually exclusive with `adaptive_rt_window` (hard error): the adaptive
  per-bin percentiles are in-sample and would silently undo the held-out
  sizing.
- standalone `rt-im-train` warns when the flag is set without
  `finetune_deeplc`: the holdout is honest against this stage's fit, but a
  library iRT produced by a fine-tune that saw these anchors still leaks.

Measured (AIF benchmark, `LFQ_Orbitrap_AIF_Ecoli_01`, frac 0.3): with DeepLC
4.1.0 the window moved 141.5 -> 204.9 s (held-out p95 136.6 s x 1.5) and
peptides at 1% moved 10,703 -> 10,822 (+1.1%) at an unchanged 0.98% decoy
fraction; with the overfitting 4.0.0a2 model the honest window is ~950 s and
costs 1.5% of peptides, so the option only pays with a generalizing RT model.
It stays benchmark-gated and off by default (see CLAUDE.md).

### 4c. Library iRT source: DeepLC 4.1.1 base predictions against the imported iRT and the fine-tune (2026-09-05)

`rt_im_train.library_irt` (default `auto`) decides what the calibration in sections 1-4
calibrates in library-input mode. Three sources were run on the AIF benchmark
(`LFQ_Orbitrap_AIF_Ecoli_01`, imported DIA-NN library, calibration only, `native_tda` so the
arms differ in RT alone, one seed each):

| library iRT | peptides at 1% | `w_rt` | in-sample residual median | PSMs extracted | run wall |
|---|---|---|---|---|---|
| DIA-NN library (raw) | 10,015 | 632 s | 117.9 s | 104,025 | 2:11 |
| per-run DeepLC fine-tune (`finetune_deeplc`) | 10,181 | 472 s | 61.4 s | 89,531 | - |
| **DeepLC 4.1.1 base model (`library_irt = deeplc`)** | **10,416** | **343 s** | 65.5 s | 72,453 | 1:19 |

The base-model prediction beats the fine-tune by 2.3% and the imported iRT by 4.0%, with the
narrowest window and the fewest candidates to extract. The in-sample residual median does not
rank the arms (61.4 s for the fine-tune against 65.5 s for the base model): the fine-tune
adapts to its own anchors, which the in-sample number rewards and the held-out window does
not (section 4b). The DIA-NN iRT is the worst source, so "calibration only" on a raw imported
library is not the default this option describes: `auto` re-predicts whenever a DeepLC
interpreter is configured, and warns when it keeps the imported values because none is.

The prediction is run-independent, so `run-experiment` computes it once for the whole
experiment (`fragment_library_precursors_deeplc.parquet` in the experiment directory) and
`run` once per invocation. Cost: 845k target peptidoforms in about 8 minutes on 32 CPU threads
(DeepLC 4.1.1, torch CPU). A re-predicted table is a plain precursor table; pass it as
`--lib-precursors` with `library_irt = library` to skip the step on later invocations. The
worker is `deeplc_finetune.py --no-finetune` (docs/13), so every row is predicted on its
`DECOY_`-stripped sequence (a shift decoy therefore shares its target's value, a reversed
decoy gets its own) and rows with non-standard residues keep the imported value, exactly as
under a fine-tune. Both label populations must move to the new scale together: a table whose
targets carry DeepLC values and whose decoys keep the imported iRT is not exchangeable and
its decoy-based q is invalid. This is the reason DeepLC 4.1.1 is the engine's floor: the default path calibrates
base-model predictions and 4.0.0a2's base model memorised anchors.

HYE B01 (`LFQ_Orbitrap_AIF_Condition_B_Sample_Alpha_01`, 10.9M-row imported library, fast
rescore recipe with `nn_torch`, `window_holdout_frac 0.3`, one seed each) with the same three
sources; the base-model arm ran through the engine's own `library_irt = auto` path:

| library iRT | peptides at 1% | `w_rt` | in-sample residual median | held-out p95 | candidates extracted | run wall | peak |
|---|---|---|---|---|---|---|---|
| DIA-NN library (raw) | 55,090 | 691 s | 130.0 s | 461 s | 2,744,896 | 30:28 | 24.6 GB |
| library fine-tuned once on A_01, reused | 59,124 | 345 s | 35.8 s | 230 s | 2,603,894 | 17:52 | 16.5 GiB |
| **DeepLC 4.1.1 base model (`auto`)** | 58,813 | 414 s | 78.3 s | 276 s | 2,497,844 | 46:36 (27 min is the one-off prediction) | 18.1 GB |

On this acquisition the base model is +6.8% over the imported iRT and 0.5% under the
once-fine-tuned library, inside the pool's 0.9% seed band; on AIF it was 2.3% over the
fine-tune. The in-sample residual again exaggerates the fine-tune's advantage (35.8 s against
78.3 s for a 0.5% difference in peptides): held-out p95 (230 against 276 s) is the honest
comparison. The wide raw windows also cost compute: extract ran 10.6 minutes at 24.6 GB
against 5.9 minutes at 18.1 GB. The one-off prediction of the HYE library (4,910,158 unique
stripped sequences, targets and reversed decoys) took 27 minutes on 64 CPU threads; under
`run-experiment` that is paid once for the six runs, and a saved re-predicted table with
`library_irt = library` skips it entirely.

The first HYE attempt at this arm was made with a scratch script that paired decoys as
`DECOY_` + target sequence. The HYE library's decoys are reversed sequences, so all 5.44M
of them kept the imported iRT while the targets moved to DeepLC's scale, and the arm gave
57,501 under an invalid, non-exchangeable decoy population. The engine path predicts every
stripped sequence and does not have this failure mode; the AIF library's decoys are shift
decoys (99.96% paired), so its arm was unaffected.

### 5. Optional adaptive per-region window (default off)

When `adaptive_rt_window` is set (and `n_train >= min_anchors`, which guarantees the
`Calibrated` plan and therefore a non-null `w_rt`), the global `w_rt` is replaced by
a per-bin half-width (rt_im_train.rs:192-229). Anchors are binned into
`nb = adaptive_rt_bins.max(1)` equal-width bins over the calibrated-RT range
(rt_im_train.rs:202), with the bin index computed from
`frac = ((cal - rt_min)/span).clamp(0.0, 0.999_999)` so the maximum RT lands in
the last bin rather than out of range (rt_im_train.rs:207). Each bin's
half-width is its local residual percentile times the multiplier, clamped to
`[lo_clamp, hi_clamp]` where `lo_clamp = rt_window_min_s.max(0.0)` and
`hi_clamp = fallback_rt_window_s.max(lo_clamp)` (rt_im_train.rs:210-211). Empty
bins fall back to the global `w_rt` (rt_im_train.rs:215-216). Each candidate then
takes the width of the bin its calibrated RT lands in (rt_im_train.rs:248-253).
Degenerate case: when all calibrated anchor RTs are equal (`rt_max <= rt_min`) the
adaptive block yields `None` and the stage silently uses the global `w_rt`
(rt_im_train.rs:203, 224-226). The rationale (config.rs:478-487): a single fixed
window is simultaneously too wide for well-calibrated regions and too narrow for
poorly-calibrated ones. This knob is part of the sensitivity program and has not
passed the entrapment gate, so it stays default-off.

### 6. Apply to every candidate and write

For each library candidate (rt_im_train.rs:246-264): `calibrated_rt =
calibration_available.then(|| predict(irt))` (rt_im_train.rs:247), the width is
either the adaptive per-bin value or the global `w_rt` (rt_im_train.rs:248-255), and
`candidate_window(calibrated_rt, width)` (rt_im_train.rs:256) produces the row
`(cal, cal - width, cal + width)`, or the unbounded `(NaN, -inf, +inf)` when either
value is absent. The three IM columns are pushed as `None` (rt_im_train.rs:261-263).
The table is written (rt_im_train.rs:266-277), `cal.json` is written
(rt_im_train.rs:309-325), and the artifact report is emitted (rt_im_train.rs:332-344).

### DeepLC multitask fine-tune (orchestrator pre-step, default off)

This is not part of `rt_im_train::run`; it runs in the `run` orchestrator between
`search-seed` and Stage B, guarded by `cfg.rt_im_train.finetune_deeplc`
(run.rs:265-303). The same hook, when the fine-tune is off, runs the base-model
re-prediction of section 4c (`sidecar::run_deeplc_repredict`, `library_irt = auto|deeplc`);
`run-experiment` hoists that re-prediction above its per-run fan-out because it does not
depend on the run. Preflight (`run.rs:62-67`) rejects `finetune_deeplc` unless
`predict_frag.deeplc_python` is set; it only checks that the field is present, not
that the interpreter can import DeepLC. The import probe lives in `mumdia doctor`,
which checks `deeplc,numpy,pandas,pyarrow,torch,psm_utils` on that interpreter
(main.rs:368-371) because `deeplc_finetune.py` imports pyarrow, torch and
`psm_utils` as well as `deeplc`. The seed PSMs are searched once on the base
library before the fine-tune and are reused as-is (the search-seed hyperscore does
not depend on iRT), so the fine-tune and Stage B both consume that same seed table;
only the library's `predicted_irt` changes in the newly written `_ft` table between
them.

When enabled, the orchestrator resolves `deeplc_finetune.py`
(`sidecar::resolve_script`, run.rs:271-274), writes a fine-tuned library
`fragment_library_precursors_ft.parquet` (run.rs:275), and rebinds `lib_p` to that
file so both Stage B and `extract` read the fine-tuned iRT (run.rs:265-303). The
manifest entry for `fragment_library_precursors` is replaced by the `_ft` table
(run.rs:291-299), so provenance points at the artifact actually consumed
downstream. The fine-tune is driven through `run_deeplc_finetune` (sidecar.rs:110-155), whose
positional CLI contract is `deeplc_finetune.py <lib_in> <seed> <lib_out> --epochs E
--patience P --q-train Q --batch B`. The worker runs with both `PYTHONUTF8=1` and
`PYTHONIOENCODING=utf-8` set (the `utf8` flag on `run_worker`, sidecar.rs:217-225)
because DeepLC/Keras crash on the Windows cp1252 console. `run_worker` spawns
`python script arg...` and returns an error if the process exits non-zero
(sidecar.rs:226-232); there is no JSON request file, only argv plus the parquet
column contract.

Inside the worker (`scripts/deeplc_finetune.py`): the reference set is the
confident target seed PSMs (`label == "target"`, `spectrum_q <= q_train`, standard
residues only via `is_std`, deeplc_finetune.py:117-121). Note the gate is `<=`
q_train here, whereas the Rust anchor selection uses strict `<` q_train
(rt_im_train.rs:100), so the two reference sets can differ by the boundary rows.
The worker keys the reference dict by `peptidoform` string (`ref[pf] = observed_rt`,
deeplc_finetune.py:117-121), so it joins seed to library by peptidoform, not by
`candidate_id` as the Rust stage does, and it keeps one RT per peptidoform by
last-write-wins in iteration order rather than best-score. The batch size
auto-scales when `--batch 0`: `min(512, max(16, n_ref // 30))`, so each epoch runs
at least about 30 gradient steps; a fixed large batch underfits small references
(deeplc_finetune.py:132-135). `deeplc.finetune(ref_psms, train_kwargs=...)`
transfer-learns the model (deeplc_finetune.py:137-146); predictions come from
`deeplc.predict(batch, model=ft_model)` over the unique standard peptidoforms in
chunks of 100_000 (deeplc_finetune.py:169-184). DeepLC's multitask models return a 2D
array (one column per task head); `agg` reduces it to one iRT by averaging across
heads (`a.mean(axis=1)`, deeplc_finetune.py:58-60, called at 175). Prediction is on
the DECOY_-stripped underlying sequence so decoys land on the same iRT scale as
targets (deeplc_finetune.py:45, 153-161). The rewritten column overwrites
`predicted_irt` in place in the output parquet (deeplc_finetune.py:189-192);
peptidoforms with no prediction, including non-standard ones, keep their original
iRT (`preds.get(base_pf(pf), orig[i])`, deeplc_finetune.py:189).

The worker also carries a documented OpenMP crash fix: numpy's OpenBLAS (GNU
OpenMP) and torch's Intel OpenMP coexist only under `KMP_DUPLICATE_LIB_OK=TRUE`,
and each spawns a full thread pool that oversubscribes the CPU during the
sustained backward pass and crashes the machine, so the worker sets the OMP/BLAS
thread caps to 1 before importing numpy and torch (deeplc_finetune.py:6-13, 22-28)
and bounds torch's own pool to `DEEPLC_FT_THREADS` (default 8) after import
(deeplc_finetune.py:100-105). `deeplc` is imported before numpy for OpenMP load
order (deeplc_finetune.py:33). Beyond the four flags the engine passes, the worker
accepts standalone-only flags the engine never sets, so they take their defaults:
`--device {cpu,cuda}` (cuda sidesteps the CPU OpenMP crash entirely), `--threads`,
`--predict-threads` (0 = reuse `--threads`; the prediction phase is forward-only
and usually tolerates more threads than the fine-tune backward pass), `--max-ref`,
`--predict-limit`, and `--skip-predict` (deeplc_finetune.py:68-79, 89-94).

The fine-tune sets no torch/numpy seed, so it is nondeterministic across runs
(CLAUDE.md notes this). Its main use is library-input mode, where the base iRT is
the imported library value rather than a native or DeepLC prediction and is often
too coarse for a tight RT window.

The fine-tune does not have to run per file. Measured on one run: a library whose
`predicted_irt` was fine-tuned once and predicted over every peptidoform, then
calibrated by this stage's per-run LOESS with `finetune_deeplc = false`, gave
median |RT residual| 6.06 s, MAD 6.11 s, slope 0.9907, intercept 16.4 s, against
6.14 s and 6.18 s for the same run with the per-file fine-tune enabled. Equal or
marginally better. The per-file fine-tune cost 2,166 s of a 5,127 s single-file
run (transfer learning plus whole-library iRT prediction over roughly 5M
peptidoforms), so moving it once per library removed about 42% of that run's wall
clock. Note the scope of the measurement: it compared RT residuals, both of them
in-sample per the caveat above, on a single run. It did not compare identification
counts, and it is not a general result across instruments or gradients. The likely
split of work is that the fine-tune places the library on a sane iRT scale while
the per-run LOESS (rt_im_train.rs:136-140) absorbs this run's chromatography.
Leave `finetune_deeplc = false` when the library iRT was already fine-tuned, and
never pass an already fine-tuned `_ft` library while also enabling the fine-tune,
which would fine-tune a fine-tuned model. Note that CLAUDE.md's validated
sensitivity workflow still specifies per-run fine-tuning, and that benchmark has
not been rerun against a once-fine-tuned library. Until it is, treat the choice as
open: per-run fine-tuning is the benchmarked default, once-per-library is the
cheaper option with equal RT residuals on the one run measured here.

## Key types and functions

| name | file:line | what it does |
|---|---|---|
| `RtImTrainParams` | rt_im_train.rs:19-26 | Input struct: seed PSMs path, library precursors path, output windows + cal paths, config, config hash. `config_hash` is a field but `run` never reads it (dead param; see gotchas). |
| `WindowPlan` | rt_im_train.rs:30-40 | Enum `{ Unbounded, Fixed(f64), Calibrated }`: the three window regimes by anchor count. |
| `window_plan` | rt_im_train.rs:42-50 | Maps `(n_train, min_anchors, fallback_width)` to a `WindowPlan`. |
| `sorted_anchor_vectors` | rt_im_train.rs:54-60 | Sorts the best-per-peptide map by `base_peptide_id` into `(train_irt, train_rt)` so the fit is order-stable. |
| `candidate_window` | rt_im_train.rs:65-70 | Builds `(cal, lo, hi)`; returns `(NaN, -inf, +inf)` when calibrated RT or width is absent. |
| `rt_im_train::run` | rt_im_train.rs:72-354 | The stage: join iRT, select anchors, fit, window, apply, write. |
| `linear_fit` | calibrate.rs:6-28 | OLS `y = slope*x + intercept` with degenerate-case guards. |
| `Loess` | calibrate.rs:31-37 | Grid-based local-linear smoother; carries a linear fallback for extrapolation. |
| `Loess::fit` | calibrate.rs:42-83 | Sorts anchors, builds a `grid_n`-point local-linear grid, `k = clamp(ceil(span*n),3,n)`. |
| `Loess::predict` | calibrate.rs:87-102 | Grid interpolation inside range, linear extrapolation outside. |
| `local_linear` | calibrate.rs:107-153 | Tricubic-weighted local least squares at one point. |
| `percentile` | calibrate.rs:156-164 | Nearest-rank percentile: sorts a copy, `rank = round(p.clamp(0,1)*(len-1))`. Not interpolated. Empty input returns 0.0. |
| `CalibrationMethod` | config.rs:56-61 | Enum `{ Loess, Linear, None }`; default `Loess`. `None` is rejected at load. |
| `RtImTrainConfig` | config.rs:447-512 | All Stage B config fields (below). |
| `run_deeplc_finetune` | sidecar.rs:110-155 | Sidecar client: `deeplc_finetune.py <lib_in> <seed> <lib_out> [flags]`. |

## Configuration

All fields live under `rt_im_train` in the config (`RtImTrainConfig`,
config.rs:447-512). The struct is `#[serde(default, deny_unknown_fields)]`, so any
unknown key is a hard load error and every field has the default below. The config
surface was pruned: there are no IM calibration fields (IM is a stub), and
`CalibrationMethod::None` is a foot-gun rejected at load (config.rs:1336-1342) even
though the enum variant still exists.

| field | default | effect |
|---|---|---|
| `calibration_method` | `loess` | `loess`, `linear`, or `none`. `loess` uses the smoother when anchors suffice, else falls back to linear. `none` is rejected by `Config::validate` (config.rs:1336-1342) because the code path silently degrades to linear. |
| `q_train` | `0.01` | Max `spectrum_q` for a seed PSM to become a calibration anchor (rt_im_train.rs:100) and to enter the DeepLC fine-tune reference (`--q-train`). |
| `p_rt` | `0.95` | Residual percentile for the RT half-window (rt_im_train.rs:182). 0.95 keeps ~95% of anchor residuals inside the window. |
| `rt_window_multiplier` | `1.0` | Scales the residual-percentile window (rt_im_train.rs:182). Larger widens the window: higher recall, more interference. |
| `min_seed_for_calibration` | `50` | Minimum anchors to (a) use LOESS instead of linear (rt_im_train.rs:135) and (b) trust the residual window instead of the fixed fallback (via `min_anchors = max(this, 2)`, rt_im_train.rs:157). |
| `loess_span` | `0.3` | LOESS local-fit fraction; passed as `span` to `Loess::fit` (rt_im_train.rs:137). Fraction of anchors in each local window. |
| `fallback_rt_window_s` | `120.0` | Fixed half-window (seconds) when anchors are too few (`WindowPlan::Fixed`, rt_im_train.rs:168-175); also the upper clamp for adaptive bin widths (rt_im_train.rs:211). |
| `finetune_deeplc` | `false` | Enable the DeepLC multitask fine-tune pre-step in `run` (run.rs:265). Requires `predict_frag.deeplc_python` (preflight, run.rs:62-67). On one measured run it was not needed per file once the library iRT had been fine-tuned once: the per-run LOESS reached the same residuals for much less wall clock (see the fine-tune section for the scope of that measurement). |
| `finetune_epochs` | `25` | Fine-tune epoch cap (`--epochs`); early stopping usually halts sooner. Used only when `finetune_deeplc`. |
| `finetune_patience` | `10` | Early-stopping patience (`--patience`): epochs without val-loss improvement before stopping. |
| `finetune_batch` | `0` | Fine-tune batch size (`--batch`). `0` auto-scales to the seed size so each epoch has ~30+ steps (config.rs:473-477); a fixed large batch underfits small seeds. |
| `adaptive_rt_window` | `false` | Per-region window instead of one global width (rt_im_train.rs:192-229). Sensitivity-program knob, not yet default-on. |
| `adaptive_rt_bins` | `12` | Number of equal-width calibrated-RT bins for the adaptive window (rt_im_train.rs:202). |
| `rt_window_min_s` | `1.0` | Lower clamp (seconds) for any adaptive half-window (rt_im_train.rs:210); mirrors the 1s floor on the global window. |
| `library_irt` | `auto` | Library-input mode only. `auto` re-predicts the imported `predicted_irt` with the DeepLC base model when `predict_frag.deeplc_python` is set and keeps it, with a warning, when not; `deeplc` requires the interpreter (preflight); `library` keeps the imported values. Ignored under `finetune_deeplc` and in FASTA mode. Section 4c has the measurement. |
| `window_holdout_frac` | `0.0` | Size `w_rt` from held-out anchor residuals instead of in-sample ones (section 4b). `base_peptide_id % 1000 < round(frac*1000)` selects the holdout; the same rule excludes those peptides from the orchestrated DeepLC fine-tune. Range `[0.0, 0.9]`, `0.0` = off; mutually exclusive with `adaptive_rt_window`. Benchmark-gated. |

## Invariants, determinism, gotchas

- **IM is a stub.** `im_pred_cal`, `im_lo`, `im_hi` are always `None`
  (rt_im_train.rs:261-263). There is no IM calibration, no IM window, and no IM
  config field. Any 4D/diaPASEF work must add both the model and the columns.
- **Only targets anchor the fit.** Decoy seed PSMs are excluded
  (rt_im_train.rs:106); admitting them would inject random iRT/RT pairs. Keep this
  filter if you refactor anchor selection.
- **Library is the single iRT source.** Training and application both read
  `predicted_irt` from the same library table (rt_im_train.rs:75-83, 232-233). When
  the fine-tune writes its new table, the orchestrator rebinds `lib_p` to the
  `_ft` file so both this stage and `extract` see the updated iRT. Do not
  reintroduce a second iRT source or imply that the original file was mutated.
- **An imported library may give every modform the same iRT.** The stage trusts
  `predicted_irt` per `candidate_id` (rt_im_train.rs:80-83) and has no way to tell
  an iRT that was modelled for a modified form from one inherited unchanged from
  its unmodified sibling. Measured on one modification-expanded imported library,
  79.7% of stripped-peptide groups carried an identical raw `predicted_irt`
  across all of their modforms, meaning the modification was never modelled;
  Spearman correlation against a proper per-modform prediction was 0.9876 for
  unmodified peptides but only 0.4980 for modified ones. The failure is expected
  to be quiet rather than loud: the unmodified majority dominates the anchors, so
  the fit still converges and the in-sample residuals still look healthy, while a
  modified candidate is centred on its unmodified sibling's RT and is mispositioned
  by however much the modification actually shifts elution. Check the
  within-`base_peptide_id` variance of `predicted_irt` before trusting RT windows
  in a PTM search, and re-predict the expanded peptidoform set with DeepLC if the
  variance is zero.
- **The reported RT residuals are in-sample.** `rt_residual_median_s`,
  `rt_residual_abs_median_s`, and `rt_residual_mad_s` (rt_im_train.rs:293-306) are
  evaluated with `predict` on the anchors that trained the fit, and under LOESS on
  the very points the smoother was fit through (rt_im_train.rs:137). On one
  measured run a reported 6.14 s absolute median was p50 17.6 s and p90 146.3 s
  against DIA-NN retention times for the same run. Quote the field as a fit
  diagnostic, never as the run's RT accuracy, and do not size an external
  tolerance from it.
- **Fewer than two anchors leaves the windows unbounded.** With `n_train < 2` no
  linear map is fit (`slope`/`intercept` are `NaN`), `w_rt` is `None`, and every
  candidate gets `(rt_pred_cal = NaN, rt_lo = -inf, rt_hi = +inf)` with status
  `"insufficient_anchors_unbounded"` (rt_im_train.rs:42-50, 160-167). Downstream this
  is recall-safe: the extractor scans the full isolation-window RT range rather than
  a numeric window. Do not treat `NaN`/infinite rows as a bug. The feature stage
  neutralizes the `NaN` `rt_pred_cal` sentinel explicitly: `calibrated_rt_error`
  (features.rs:337-344) returns a 0 RT error when either `apex_rt` or `rt_pred_cal`
  is non-finite, so the sentinel never contaminates the feature matrix or the
  preliminary competition score.
- **The 1-second floor.** The `Calibrated` global window is floored at 1.0s
  (rt_im_train.rs:182). With too few anchors the fit is near-exact, residuals
  collapse, and this floor would discard true co-elutions, which is exactly why the
  `Fixed` fallback exists for `2 <= n_train < min_anchors` (rt_im_train.rs:152-186).
  Do not remove the fallback branch.
- **`percentile` is nearest-rank, not interpolated** (calibrate.rs:156-164). It
  sorts a copy each call, so it is order-independent, but repeated calls on the
  same data re-sort. This is fine at Stage B sizes.
- **Calibration is order-deterministic (fixed).** `best_per_pep` is still a
  `HashMap`, but `sorted_anchor_vectors` (rt_im_train.rs:54-60, called at 121) sorts
  the anchors by `base_peptide_id` before building `train_irt`/`train_rt`, so
  `linear_fit`'s summation order (calibrate.rs:17-20) is fixed across processes and
  `slope`/`intercept`/`w_rt`/the calibrated RTs are reproducible. `Loess::fit`
  additionally re-sorts by `x` internally (calibrate.rs:45-48). This satisfies
  CLAUDE.md's determinism rule (ordered iteration where floats are summed); do not
  reintroduce a `.values()`-order reduction. The test
  `anchor_vectors_are_sorted_by_base_peptide_id` (rt_im_train.rs:360-375) guards it.
- **The fine-tune is explicitly nondeterministic.** `deeplc_finetune.py` sets no
  torch/numpy seed (CLAUDE.md), so `finetune_deeplc = true` makes `predicted_irt`,
  and therefore the whole run, non-reproducible. Treat it as an accuracy lever, not
  a deterministic default.
- **`slope`/`intercept` are emitted in `cal.json` whenever calibration is available,
  including under LOESS** (rt_im_train.rs:128, 286-287, 313-314). They are the LOESS
  extrapolation fallback, not dead values; do not assume they were unused when
  `method == "loess"`. They are serialized as `null` only when calibration is
  unavailable (`n_train < 2`), since the fit is not computed in that case.
- **`CalibrationMethod::None` still exists but is rejected** at config load
  (config.rs:1336-1342). The stage would otherwise fall through to the linear path
  because `use_loess` matches only `Loess` (rt_im_train.rs:133-135). This fallthrough
  is a known correctness wart (see CLAUDE.md "Correctness").
- **Report schema version is 1** (`artifact::RUN_WINDOWS`, schema.rs:16). Bump it if
  the column set changes.
- **`config_hash` is a dead param in this stage.** Both call sites build the blake3
  config hash and pass it as `RtImTrainParams.config_hash` (run.rs:313,
  main.rs:532), but `run` never reads it (unlike stages that stamp it into their
  report). Do not rely on the report to carry the config hash for Stage B.
- **Anchor gate is `<`, fine-tune gate is `<=`.** The Rust anchor selection keeps
  seed rows with `spectrum_q < q_train` (rt_im_train.rs:100), while the DeepLC
  fine-tune worker keeps `spectrum_q <= q_train` (deeplc_finetune.py:120). The two
  confident-seed reference sets can therefore differ by the boundary rows. Keep this
  in mind when reasoning about why the fine-tune reference count and the calibration
  anchor count are not identical.
- **Fine-tune joins by peptidoform, Stage B joins by `candidate_id`.** The stage
  maps seed to library iRT through `candidate_id` (rt_im_train.rs:109-113); the worker
  maps seed observed RT to library peptidoforms through the `peptidoform` string
  (deeplc_finetune.py:117-121, 189). A library whose peptidoform strings do not match
  the seed's would silently fine-tune on nothing.
- **Test coverage.** `calibrate.rs` has three unit tests: `linear_recovers_line`
  (calibrate.rs:170-176), `loess_tracks_nonlinear` (calibrate.rs:178-185), and
  `percentile_basic` (calibrate.rs:187-191). The stage `rt_im_train.rs` now has three
  unit tests covering the helper functions: `anchor_vectors_are_sorted_by_base_peptide_id`
  (rt_im_train.rs:360-375), `sparse_anchor_policy_is_unbounded_only_below_two`
  (rt_im_train.rs:377-388), and `unavailable_calibration_emits_unbounded_window_and_nan_prediction`
  (rt_im_train.rs:390-401). The full `run` body (anchor selection over a real seed
  table, the LOESS fit path, and the adaptive window) is still exercised only in full
  runs.

## How to extend / modify

- **Add IM (4D).** Populate `im_pred_cal`/`im_lo`/`im_hi` (currently `None` at
  rt_im_train.rs:261-263) from an IM calibration analogous to the RT path (an
  IM2Deep-style model), and add IM config fields to `RtImTrainConfig`. The output
  columns already exist and are nullable, so downstream reads survive the
  transition. `extract` and the IM feature families must then consume them.
- **Change the calibration model.** Extend `CalibrationMethod` (config.rs:56-61) and
  branch in the `predict` closure setup (rt_im_train.rs:133-150). Keep `linear_fit` as
  the universal fallback so a degenerate anchor set never panics. New models belong
  in `calibrate.rs` next to `Loess`, with unit tests like `loess_tracks_nonlinear`
  (calibrate.rs:178-185).
- **Tune the window policy.** The window-derivation logic is localized at
  rt_im_train.rs:152-229 (the `window_plan` match plus the optional adaptive block).
  New window strategies (for example a symmetric-vs-asymmetric window, or a learned
  per-charge width) should be config-gated and default-off, then validated against the
  entrapment holdout before becoming a default, per the sensitivity program
  (`docs/20_sensitivity_and_quantification_playbook.md`).
- **Swap the fine-tune worker.** The contract is purely the CLI and the parquet
  columns (`sidecar.rs:107-155`, `scripts/deeplc_finetune.py`). A replacement worker
  need only accept `<lib_in> <seed> <lib_out>` plus the four flags and rewrite the
  `predicted_irt` column. Preserve the DECOY_-stripping behavior
  (deeplc_finetune.py:45, 153-161, 189) so decoys stay on the target iRT scale, and set a
  seed if you want the fine-tune to be reproducible. The worker also runs
  standalone, which is how a library is fine-tuned once and then reused across
  files with `finetune_deeplc = false`; the engine needs only the rewritten
  precursor parquet.
