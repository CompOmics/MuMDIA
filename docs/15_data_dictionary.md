# 15. Parquet data dictionary

Consolidated column reference for every artifact the MuMDIA engine writes. Every
row here was taken from the `write_table`/`Col` construction in the stage source,
not from prose. Citations are `file:line`.

## Type legend

Columns are built from the `Col` enum in
`rust/mumdia/crates/mumdia-io/src/table.rs`. Each variant maps to a fixed Arrow
`DataType` and nullability in `Col::field` (`table.rs:87-100`):

| `Col` variant | Arrow type | nullable |
|---|---|---|
| `I64` | Int64 | no |
| `I32` | Int32 | no |
| `U32` | UInt32 | no |
| `F64` | Float64 | no |
| `F32` | Float32 | no |
| `Bool` | Boolean | no |
| `Str` | Utf8 | no |
| `OptF64` | Float64 | yes |
| `OptF32` | Float32 | yes |
| `OptI32` | Int32 | yes |
| `OptStr` | Utf8 | yes |
| `ListF32` | List\<Float32\> | yes (list + item) |
| `ListF64` | List\<Float64\> | yes (list + item) |
| `LargeListF32` | LargeList\<Float32\> | yes (list + item) |

All non-`Opt`, non-`List` columns are written as Arrow non-nullable
(`nullable = false`). Below, "nullable" means the Arrow field nullability, not
whether a value is semantically optional. Note that `mumdia-io` integer and bool
getters do not check `is_null` (safe only because the non-nullable columns are
never null in practice).

Non-Parquet sidecars written alongside artifacts (JSON manifests, the feature
schema companion, the PIN, the human-readable TSV reports) are documented in
their own sections at the end.

---

## Stage 0: `convert` (`stages/convert.rs`)

### spectra_ms1 (`convert.rs:171-179`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `scan_index` | UInt32 | no | - | monotonic 0-based scan index assigned in read order |
| `rt_seconds` | Float64 | no | s | scan start time, converted from mzdata minutes |
| `mz` | List\<Float32\> | yes | m/z | centroided, m/z-sorted peak m/z values |
| `intensity` | List\<Float32\> | yes | counts | peak intensities aligned to `mz` |

v2 (written only by the native diaPASEF reader, docs/35_diapasef.md) appends:

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `im` | LargeList\<Float32\> | yes | V s cm^-2 | per-peak intensity-weighted 1/K0, aligned to `mz` |

### spectra_ms2 (`convert.rs:198-213`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `scan_index` | UInt32 | no | - | monotonic scan index (shared axis with MS1) |
| `id` | Utf8 | no | - | vendor spectrum id string |
| `rt_seconds` | Float64 | no | s | scan start time |
| `window_id` | UInt32 | no | - | dense id of the distinct isolation window (`IsolationWindow::key`: the m/z bound bits, and on v2 the 1/K0 bound bits), matches `isolation_windows.window_id` |
| `window_target` | Float64 | no | m/z | isolation-window target m/z (0.0 for AIF/all-ion) |
| `window_lower` | Float64 | no | m/z | isolation-window lower bound (0.0 for AIF full-range fallback) |
| `window_upper` | Float64 | no | m/z | isolation-window upper bound (1.0e6 for AIF full-range fallback) |
| `precursor_mz` | Float64 | yes | m/z | selected precursor m/z, null when absent |
| `precursor_charge` | Int32 | yes | - | precursor charge, null when absent |
| `mz` | List\<Float32\> | yes | m/z | centroided fragment m/z values |
| `intensity` | List\<Float32\> | yes | counts | fragment intensities aligned to `mz` |

v2 (native diaPASEF only) adds `window_im_lower` and `window_im_upper` after
`precursor_charge`, and `im` after `intensity`:

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `window_im_lower` | Float32 | yes | V s cm^-2 | the slot's lower 1/K0 bound |
| `window_im_upper` | Float32 | yes | V s cm^-2 | the slot's upper 1/K0 bound |
| `im` | LargeList\<Float32\> | yes | V s cm^-2 | per-peak intensity-weighted 1/K0, aligned to `mz` |

### isolation_windows (`convert.rs:215-226`)

The distinct-window column is `window_id` (verified: `convert.rs:218`).

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `window_id` | UInt32 | no | - | dense window id (0-based first-seen order) |
| `target` | Float64 | no | m/z | window target m/z |
| `lower` | Float64 | no | m/z | window lower bound |
| `upper` | Float64 | no | m/z | window upper bound |

v2 (native diaPASEF only) adds `im_lower` and `im_upper` (Float32, nullable, V s cm^-2):
one row per diaPASEF slot. Provenance only; no stage reads them.

### ms2_to_ms1 (`convert.rs:228-234`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `ms2_scan_index` | UInt32 | no | - | MS2 `scan_index` |
| `ms1_scan_index` | Int32 | no | - | `scan_index` of the most recent preceding MS1 scan, or -1 if none |

---

## Stage A: `digest` (`stages/digest.rs`)

### peptides (`digest.rs:286-298`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `id` | UInt32 | no | - | peptide row id (targets and decoys interleaved) |
| `peptide` | Utf8 | no | - | stripped amino-acid sequence |
| `protein` | Utf8 | no | - | `;`-joined protein accessions; decoys carry a `DECOY_` prefix |
| `start` | Int32 | no | - | 0-based start offset in the source protein |
| `end` | Int32 | no | - | 0-based end offset (exclusive) |
| `label` | Utf8 | no | - | `target` or `decoy` |
| `target_id` | Int32 | no | - | for a decoy, the `id` of its paired target; -1 for a target |
| `decoy_strategy` | Utf8 | no | - | lowercased decoy strategy name (e.g. `reverse`, `scramble`) |

---

## Stage A2: `peptidoforms` (`stages/peptidoforms.rs`)

### peptidoforms (`peptidoforms.rs:253-265`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `id` | UInt32 | no | - | peptidoform row id (unique per mod/charge expansion) |
| `peptide_id` | UInt32 | no | - | source `peptides.id` |
| `base_peptide_id` | UInt32 | no | - | target `id` for grouping (a decoy points at its target); competition/FDR key |
| `peptide` | Utf8 | no | - | stripped sequence (carried from `peptides`) |
| `peptidoform` | Utf8 | no | - | ProForma-lite string with UniMod mod names in brackets |
| `charge` | Int32 | no | - | precursor charge state |
| `label` | Utf8 | no | - | `target` or `decoy` |
| `protein` | Utf8 | no | - | protein accessions (carried from `peptides`) |

---

## Stage C: `predict-frag` (`stages/predict_frag.rs`)

### fragment_library precursors (`predict_frag.rs:197-211`)

Schema `fragment_library_precursors` version 1, or version 2 when the library has an
ion-mobility source (`artifact::fragment_library_precursors(has_im)`). Version 2 is version
1 plus `predicted_im`; the column is never written always-null.

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | contiguous id, sorted by ascending `precursor_mz` (the fragment-index build key) |
| `peptidoform_id` | UInt32 | no | - | source `peptidoforms.id` |
| `base_peptide_id` | UInt32 | no | - | grouping key (target/decoy family) |
| `peptidoform` | Utf8 | no | - | ProForma-lite peptidoform |
| `charge` | Int32 | no | - | precursor charge |
| `precursor_mz` | Float64 | no | m/z | precursor m/z from the shared mass model |
| `predicted_irt` | Float32 | no | iRT units | predicted indexed retention time (native or DeepLC) |
| `label` | Utf8 | no | - | `target` or `decoy` |
| `protein` | Utf8 | no | - | protein accessions |
| `n_fragments` | Int32 | no | - | number of fragments kept for this candidate after top-N |
| `predicted_im` | Float64 | yes | V s cm^-2 (1/K0) | version 2 only. Library reduced ion mobility: IM2Deep's single-conformer CCS converted at `precursor_mz` (predict-frag under `predict_frag.im_predictor`; never null there, a miss is dropped with its pair), or the imported library's `IM` / `IonMobility` (null where the source value is missing, non-finite or not positive). A script-built decoy carries its target's value, null included |

### fragment_library fragments (`predict_frag.rs:212-223`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | precursor `candidate_id` this fragment belongs to |
| `mz` | Float64 | no | m/z | theoretical fragment m/z |
| `predicted_intensity` | Float32 | no | relative | predicted fragment intensity (native heuristic or MS2PIP) |
| `name` | Utf8 | no | - | fragment name, e.g. `b3`, `y7`, `b3^2` |
| `ion_type` | Utf8 | no | - | ion-type symbol (`b`/`y`) |
| `ordinal` | Int32 | no | - | ion ordinal (residue position) |
| `frag_charge` | Int32 | no | - | fragment charge state |
| `cardinality` | Int32 | no | count | distinct library precursors sharing this fragment m/z (0.01 Da bin); low = clean/quantification-friendly ion, high = interference-prone. Precomputed at library build |

---

## Stage S: `search-seed` (`stages/search_seed.rs`)

### seed_psms (`search_seed.rs:233-250`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id |
| `peptidoform` | Utf8 | no | - | ProForma-lite peptidoform |
| `charge` | Int32 | no | - | precursor charge |
| `precursor_mz` | Float64 | no | m/z | precursor m/z |
| `base_peptide_id` | UInt32 | no | - | grouping key |
| `protein` | Utf8 | no | - | protein accessions |
| `label` | Utf8 | no | - | `target` or `decoy` |
| `score` | Float64 | no | - | best Sage-lite hyperscore for this candidate |
| `spectrum_q` | Float64 | no | - | target-decoy q-value (best per candidate) |
| `observed_rt` | Float64 | no | s | RT of the best-scoring scan |
| `predicted_irt` | Float32 | no | iRT units | library predicted iRT (carried) |
| `matched_peaks` | Int32 | no | - | matched fragment count at the best scan |
| `scan_index` | UInt32 | no | - | `scan_index` of the best-scoring MS2 scan |
| `observed_im` | Float64 | yes | 1/K0 (V s cm^-2) | v2 only: the intensity-weighted median 1/K0 of the matched fragment peaks of a confident target (`q <= fdr_seed`) in its best scan, each peak being the calibrant's nearest peak within `fragment_tol_ppm`. Under `rt_im_train.im_window_sizing = seed_decoy_corrected` (the default) every other PSM with `spectrum_q <= im_window_sizing_q`, targets and decoys, carries one by the same rule, for the window sizing only; null for every other row |

Version 1 has the thirteen columns above `observed_im`. Version 2 (`artifact::seed_psms(true)`)
adds `observed_im` and is written only when the searched spectra carry per-peak 1/K0 (spectra
v2, the native diaPASEF reader); its report then also carries the stat `with_observed_im`, the
number of rows with a value. A 3D seed is version 1, byte for byte. Every manifest site records
the version read from the table's footer (`search_seed::seed_schema`). The report's `params`
gain `count_fragments_once: true` only when that key is on.

### `<seed>.masscal.json` (`search_seed.rs:217-227`)

Per-run fragment mass recalibration sidecar consumed by `extract`.

| field | JSON type | units | meaning |
|---|---|---|---|
| `frag_ppm_offset` | number | ppm | median systematic fragment mass offset (0.0 when < 20 calibrants) |
| `frag_tol_ppm` | number | ppm | learned fragment tolerance (95th percentile of centered deviations x1.5, floor 5.0); falls back to the configured tolerance when too few calibrants |
| `frag_ppm_sigma` | number | ppm | local mass-uncertainty estimate (equal to `frag_tol_ppm`) |
| `n_dev` | integer | - | number of matched-fragment ppm deviations collected |
| `cal_passes` | integer | - | calibration passes run (0 = fallback, 1 = single, 2 = robust two-pass) |

### rt_bootstrap_anchors (`rt_bootstrap.rs`, `write_search_anchors`)

Written only when the bootstrap runs (`rt_im_train.multihead_anchors = search`, or the
default `auto` on a run short of seed anchors), to
`<run>/rt_bootstrap/search_anchors.parquet`: the confident targets of the bootstrap
search that the multi-head calibration is refitted on (docs/08 section 4f). Seed-shaped,
so `deeplc_finetune.py` reads it as it reads `seed_psms`. One row per candidate, ascending
`candidate_id`.

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id |
| `peptidoform` | Utf8 | no | - | ProForma-lite peptidoform |
| `charge` | Int32 | no | - | precursor charge |
| `base_peptide_id` | UInt32 | no | - | grouping key (also the `window_holdout_frac` split key) |
| `label` | Utf8 | no | - | always `target` |
| `score` | Float64 | no | - | the bootstrap rescorer's score of the kept row |
| `spectrum_q` | Float64 | no | - | the bootstrap's `precursor_q` (`<= rt_im_train.q_train`) |
| `observed_rt` | Float64 | no | s | the bootstrap's `apex_rt` |

---

## Stage B: `rt-im-train` (`stages/rt_im_train.rs`)

### run_windows (`rt_im_train.rs:266-277`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id (one row per library candidate) |
| `rt_pred_cal` | Float64 | no | s | predicted iRT calibrated to this run's observed RT |
| `rt_lo` | Float64 | no | s | lower RT window bound (`rt_pred_cal - width`) |
| `rt_hi` | Float64 | no | s | upper RT window bound (`rt_pred_cal + width`) |
| `im_pred_cal` | Float64 | yes | 1/K0 | the library `predicted_im` through this run's ion-mobility calibration (docs/08, "Ion-mobility calibration"); null on a 3D run, without a calibration, and for a row without a finite, positive `predicted_im` |
| `im_lo` | Float64 | yes | 1/K0 | `im_pred_cal - w_im`; null where `im_pred_cal` is |
| `im_hi` | Float64 | yes | 1/K0 | `im_pred_cal + w_im`; null where `im_pred_cal` is |

The table stays version 1 on every run: the three IM columns were always part of it. They are
filled only when the seed carries `observed_im` and the library `predicted_im`. Extract reads
them through its one `run_windows` reader (`extract::read_run_windows`) for the 1/K0 gate and
for `psms_extracted`'s `im_pred_cal` (docs/09 section 8). A NaN `im_lo` or `im_hi` is refused
with the row named, as a NaN RT bound is.

### `<run_windows>` cal.json (`rt_im_train.rs:290-302`)

RT-calibration sidecar (written to `out_cal`, not part of the Parquet contract).

| field | JSON type | meaning |
|---|---|---|
| `method` | string | `loess`, `linear`, or `unavailable` (no calibration fit ran) |
| `slope` | number or null | linear fit slope (iRT -> s); null when calibration is unavailable |
| `intercept` | number or null | linear fit intercept (s); null when calibration is unavailable |
| `w_rt` | number | global RT half-window width (s) |
| `p_rt` | number | residual percentile used for the window |
| `multiplier` | number | RT window multiplier |
| `n_train` | integer | number of confident anchors used |
| `calibration_status` | string | `loess`, `linear`, or `fallback_fixed` |
| `n_rt_outliers_removed` | integer | only under `rt_im_train.anchor_x_filter` and only when above 0: the anchors outside `[Q1 - 3 IQR, Q3 + 3 IQR]` of their predicted iRT, dropped before every fit; `n_train` counts the rest |

On a run whose seed carries `observed_im` (spectra with 1/K0), and only there, `cal.json` also
holds the ion-mobility block below, and the `run_windows` report's stats gain `im_n_train` and
`w_im`. A 3D `cal.json` has none of these keys.

| field | JSON type | meaning |
|---|---|---|
| `im_method` | string | `per_charge_linear_ccs`, or `unavailable` (the library has no `predicted_im`, or too few anchors) |
| `im_n_train` | integer or null | ion-mobility anchors: confident targets at the anchor cut with a finite, positive observed and predicted 1/K0, one per candidate |
| `im_global` | object or null | `{a, b}` of the global fit `obs_ccs = a + b * pred_ccs` |
| `im_per_charge` | object or null | per charge with at least `im_min_anchors_per_charge` anchors: `{a, b, n}`; other charges use `im_global` |
| `w_im` | number or null | IM half-window (1/K0) |
| `w_im_sizing` | string or null | `holdout`, or `holdout_fallback_in_sample` when one side of the split had too few anchors, or `seed_decoy_corrected` when the decoy-corrected seed estimate widened `w_im` |
| `p_im` | number | residual percentile that sizes `w_im` |
| `im_window_multiplier` | number | multiplier on that percentile and on the decoy-corrected seed estimate |
| `im_window_min` | number | lower clamp of `w_im` |
| `im_window_sizing` | object | under `rt_im_train.im_window_sizing = seed_decoy_corrected` (the default) only: `{mode, q_max, seed_targets, seed_decoys, decoy_corrected_p_im, applied}`, the seed PSMs up to `q_max` that entered the estimate, the decoy-corrected `p_im` residual (null when targets minus decoys were fewer than 100) and whether it widened `w_im` |
| `im_holdout_frac` | number | fraction of anchor base peptides held out (the constant 0.3) |
| `im_window_floor` | object | second pass of match-between-runs only, on a run whose first pass had a finite `w_im`: `{floor, configured, first_pass_w_im, source}`, the `im_window_min` applied (the larger of the configured value and the first pass's `w_im`) and the first-pass `cal.json` it came from |
| `im_n_holdout` | integer or null | held-out anchors |
| `im_holdout_resid_abs_median` | number or null | median held-out absolute residual (1/K0) |
| `im_holdout_resid_p_im` | number or null | held-out residual at `p_im`, before the multiplier |
| `im_in_sample_resid_abs_median` | number or null | median in-sample absolute residual (a fit diagnostic, not an error estimate) |

---

## Stage D: `extract` (`stages/extract.rs`)

### psms_extracted (`extract.rs:1474-1495`, base columns)

One row per accepted candidate (the selected apex PSM).

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id |
| `apex_rt` | Float64 | no | s | selected apex retention time |
| `apex_im` | Float64 | yes | 1/K0 (V s cm^-2) | v2: always null. v4: the intensity-weighted median 1/K0 of the candidate's matched fragment peaks in its apex scan (each library fragment's nearest peak within the fragment tolerance, inside the 1/K0 window under `extract.im_gate`); null where the apex scan has none |
| `apex_intensity` | Float32 | no | counts | summed observed intensity of all matched fragments at the apex scan |
| `n_matched_fragments` | Int32 | no | - | count of distinct matched predicted fragments |
| `n_predicted_fragments` | Int32 | no | - | number of predicted fragments in the library |
| `coelution_run` | Int32 | no | scans | longest consecutive-scan co-elution run |
| `rt_pred_cal` | Float64 | no | s | calibrated predicted RT (from `run_windows`) |
| `precursor_mz` | Float64 | no | m/z | precursor m/z |
| `charge` | Int32 | no | - | precursor charge |
| `label` | Utf8 | no | - | `target` or `decoy` |
| `base_peptide_id` | UInt32 | no | - | grouping key |
| `peptidoform` | Utf8 | no | - | ProForma-lite peptidoform |
| `protein` | Utf8 | no | - | protein accessions |
| `predicted_irt` | Float32 | no | iRT units | library predicted iRT |
| `contested_frac` | Float64 | no | fraction | fraction of contested intensity lost to better co-eluters (0 when the two-pass path is off) |
| `ms1_isom1` | Float64 | yes | counts | MS1 intensity at mono minus one isotope spacing; null when no MS1 |
| `ms1_mono` | Float64 | yes | counts | MS1 monoisotopic intensity near the apex; null when no MS1 |
| `ms1_iso1` | Float64 | yes | counts | MS1 +1 isotope intensity; null when no MS1 |
| `ms1_iso2` | Float64 | yes | counts | MS1 +2 isotope intensity; null when no MS1 |

Version 2 (`artifact::PSMS_EXTRACTED`) is the table above, written by every run whose
spectra carry no 1/K0. Version 4 (`artifact::psms_extracted(true)`) is written only when the
extracted spectra carry per-peak 1/K0 (spectra v2, the native diaPASEF reader): `apex_im` is
filled and three columns follow it, before `apex_intensity`. The number and the meaning are
the IM branch's v4; its v3 (`apex_im` filled alone) and v5 (1/K0 peak-shape columns) are never
written. Readers select columns by name; `extract::psms_schema` reads the version from the
footer. Every row, the selected apex and every promoted alternative, carries the values of its
own apex.

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `apex_im_mad` | Float64 | yes | 1/K0 | v4 only: the same fragment peaks' intensity-weighted mean absolute deviation from `apex_im` (not a median absolute deviation) |
| `ms1_apex_im` | Float64 | yes | 1/K0 | v4 only: in the MS1 scan nearest the apex, the 1/K0 of the peak within `extract.prec_tol_ppm` of the offset-corrected precursor m/z that lies nearest `apex_im` (the most intense in-tolerance peak without an `apex_im`); not gated; null without MS1 or without such a peak |
| `im_pred_cal` | Float64 | yes | 1/K0 | v4 only: the candidate's `run_windows.im_pred_cal`; null without an ion-mobility calibration |

**Retrace patch** (`retrace.enabled`, docs/09 section 9). On a diaPASEF run whose
chromatograms retrace rebuilt from the raw events, `psms_extracted` is patched once, in
place, and keeps version 4: per rank-0 row, `apex_rt` and `apex_im` under `retrace.repick`
(the re-picked apex and its band centre); `apex_intensity` and `n_matched_fragments` read
from the rebuilt fragment traces at the apex; `ms1_mono`, `ms1_iso1` and `ms1_iso2` read
from the rebuilt MS1 traces at the apex, null only where extract had none and the raw trace
is 0 there. `ms1_isom1`, `apex_im_mad` and `ms1_apex_im` keep extract's values. A row whose
candidate has no rebuilt rows keeps all of extract's values. The table's report records
the patch under `params.retrace` (`patched_columns` and the row counts). One optional
column, read by name and written only under `retrace.apex_features` (as `rt_lo`/`rt_hi`
are only under `extract.emit_rt_window`):

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `imc_ref_w` | Float64 | no | correlation | retrace's apex 1/K0 agreement: over the rebuilt fragment rows with signal, the predicted-intensity-weighted mean leave-one-out reference correlation of each fragment's raw 1/K0 profile (`apex_im +/- 0.04`, 40 bins, MS2 frames within 1.5 s of the apex); 0.0 with no such fragment or no rebuilt rows |

Conditional columns, appended only when `extract.emit_contested_features`
(`extract.rs:1498-1501`):

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `contested_count_frac` | Float64 | no | fraction | fraction of contested fragment-peak instances lost |
| `apportioned_frac` | Float64 | no | fraction | fraction of contested intensity kept under proportional apportionment |

Conditional columns, appended only when `extract.emit_gate_diagnostics`
(`extract.rs:1504-1509`). All four gate scores are computed together only when
this flag is set (`extract.rs:1224-1229`); when it is off they are not computed
and the columns are not written, so the default schema stays byte-identical:

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `gate_apex` | Float32 | no | correlation | single-apex-scan intensity Pearson |
| `gate_peak_spectral` | Float32 | no | correlation | peak-integrated spectral Pearson |
| `gate_coelution` | Float32 | no | correlation | temporal co-elution score |
| `gate_spectral_entropy` | Float32 | no | similarity | apex spectral-entropy similarity (sqrt) |

### chromatograms (`extract.rs:1512-1526`)

One row per predicted transition per accepted candidate, plus MS1 isotope XIC
pseudo-rows (`frag_name` = `ms1_mono`/`ms1_iso1`/`ms1_iso2`) when MS1 + window
grid are available.

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id |
| `frag_name` | Utf8 | no | - | fragment name (or `ms1_mono`/`ms1_iso1`/`ms1_iso2`) |
| `frag_mz` | Float64 | no | m/z | theoretical fragment m/z (precursor m/z for MS1 rows) |
| `frag_obs_mz` | Float64 | no | m/z | intensity-weighted observed m/z; falls back to `frag_mz` |
| `predicted_intensity` | Float32 | no | relative | library predicted intensity (0.0 for MS1 rows) |
| `rt` | LargeList\<Float32\> | yes | s | XIC retention times (empty for a never-observed transition) |
| `intensity` | LargeList\<Float32\> | yes | counts | XIC intensities aligned to `rt` |

`rt`/`intensity` use `LargeList` (64-bit offsets) because the total list-value
count can exceed the 32-bit `ListArray` offset ceiling on large candidate sets.

Each trace covers the candidate's RT window `[rt_lo, rt_hi]`, with one exception: the second
pass of match-between-runs on a run with ion mobility records the traces `params.trace_pad_s`
seconds further on each side (the extract report of both tables, carried by retrace), so a
fixed integration window and its baseline flanks fit around an apex near the window's edge
(docs/12, "Second pass"). The schema is unchanged. The features stage reads the parameter and
cuts every trace back to the row's window (`psms_extracted` `rt_lo`, `rt_hi`) before computing
anything; quantification reads the whole trace.

The table above is schema version 1, the default before 2026-09-27. Version 2, the
default since, is the layout below. Versions 3 and 4 add the per-point 1/K0 of a 4D run to
versions 1 and 2 ("1/K0 lists", below).

#### Layout v2 (`extract.chromatogram_schema = 2`)

Schema version 2 holds the same rows in the same order and is written only under
`extract.chromatogram_schema = 2`. It replaces the two list columns `rt` and
`intensity` with `rt_axis` and `intensity_trimmed`, and adds two columns. The other
five columns are the v1 ones. Every reader in the engine (features, quant, the pool)
accepts both layouts and tells them apart by these columns
(`mumdia::chromatograms::Layout::of`). A table with some but not all four v2 columns,
or with a v2 column beside `rt` or `intensity`, is refused as neither layout.

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `rt_axis` | LargeList\<Float32\> | yes | s | XIC retention times, written only where no earlier row of the candidate in the row group supplies them (see below) |
| `intensity_trimmed` | LargeList\<Float32\> | yes | counts | XIC intensities from the first to the last value that is not `+0.0` |
| `trace_offset` | UInt32 | no | points | position in the full trace of the first stored `intensity_trimmed` value |
| `trace_len` | UInt32 | no | points | length of the full trace; 0 for a never-observed transition |

The lists are renamed, not reused, so that a reader that knows only v1 fails on a v2
table instead of misreading it. An engine binary from before v2, or a script that
reads `rt` and `intensity`, stops at the missing column. With the v1 names it would
take an empty axis beside a trimmed trace for an observed row and compute from it
without an error.

`intensity_trimmed` holds the full trace from its first to its last value that is not
`+0.0`. The comparison is on the bit pattern, so a `-0.0` or a NaN is stored as a
value. Every other point of the full trace is `+0.0`, and a trace that is `+0.0`
throughout stores no value. `rt_axis` holds a row's axis only where no earlier row can
supply it: an observed row writes its axis unless the last row that wrote an axis in
the same parquet row group belongs to the same candidate and wrote the same bit
patterns. Then it writes an empty `rt_axis` and uses that axis. In window-grid mode
this is one axis per candidate per row group.

The rule restarts at every row group, so each row group can be decoded on its own. A
reader that starts inside a row group and inside a candidate must first read the rows
from the row group's first row. To rebuild a v1 row, take the row's own `rt_axis`, or
else the last `rt_axis` its candidate carried in the row group, and `trace_len` zeros
with the stored `intensity_trimmed` copied in at `trace_offset`.
`mumdia::chromatograms::Decoder` does exactly this, and
`mumdia::chromatograms::rewrite` converts a table between the two layouts; convert a
v2 table to v1 before handing it to a tool outside the engine. A pooled table has the
bands' layout, and the pool refuses bands of different layouts.

Measured on three real runs by the ignored test `v2_on_a_real_artifact`
(`chromatograms.rs`), each artifact rewritten in both layouts by today's writer
(65,536-row groups, snappy, `rt` PLAIN):

| run | rows | v1 | v2 | `rt` -> `rt_axis` values | `intensity` -> `intensity_trimmed` values |
|---|---|---|---|---|---|
| AIF E. coli | 671,362 | 82.2 MB | 58.6 MB (-28.7%) | 62.5 M -> 5.79 M | 62.5 M -> 26.6 M |
| AIF entrapment | 348,138 | 297.7 MB | 148.1 MB (-50.3%) | 287.9 M -> 34.9 M | 287.9 M -> 179.9 M |
| Astral HYE (15-min gradient) | 9,388,687 | 610.2 MB | 515.6 MB (-15.5%) | 348.5 M -> 26.9 M | 348.5 M -> 196.7 M |

On the same three, the features stage (Extended set, confident bounds) and quant wrote
byte-identical tables from both layouts. The saving grows with trace length: the
Astral run's traces are short (about 37 points a row), so the zero margins and the
repeated axis are a smaller part of it. A dictionary-encoded axis was 2.7% larger on
the AIF run and 3.2% and 0.3% smaller on the other two, so the axis column (`rt`,
`rt_axis`) stays PLAIN in both layouts. These sizes were measured with the v2 lists
still named `rt` and `intensity`; the rename changes two names in the footer and no
page.

#### 1/K0 lists (versions 3 and 4)

On spectra that carry per-peak 1/K0, and only while a consumer of the list is on (the
ion-mobility feature family, `features.im_features`, or retrace, `retrace.enabled`), each
row also carries its per-point 1/K0. Version 3 is version 1 plus `im`, version 4 is version 2 plus `im_trimmed`
(`artifact::chromatograms(schema, has_im)`), so `extract.chromatogram_schema = 1` keeps
working on 4D data. Otherwise the table is version 1 or 2, byte for byte.

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `im` | LargeList\<Float32\> | yes | 1/K0 | v3: one value per `intensity` value: the 1/K0 of the peak the trace holds at that point, `+0.0` where the trace is `+0.0`; empty for a never-observed transition; `+0.0` at every point of an MS1 row |
| `im_trimmed` | LargeList\<Float32\> | yes | 1/K0 | v4: `im` cut to the run `intensity_trimmed` keeps (the same `trace_offset`, the same length); the decoder rebuilds the full list with `+0.0` margins |

The values have a resolution of 1e-4 V s cm^-2 (the extracted hit stores them as a u16). The
list adds about 38% to the table. `mumdia::chromatograms::recorded_schema` reads a table's
version from its columns, `rewrite` carries the list between layouts, and the pool refuses
bands whose columns differ. The extract report of a 4D run carries `params.reader =
"timsrust"` and `params.im_gate` (the gate it applied) in both tables' reports, so a stage
that reads only these artifacts resolves the TIMS predicate from them.

#### Rebuilt by retrace

Under `retrace.enabled` extract writes its table to `chromatograms.centroid.parquet`, and
retrace writes `chromatograms.parquet` with the same rows, the same layout and the same
version: the intensities of the fragment and MS1 rows are sums of raw timsTOF events
(docs/09 section 9), and a rebuilt fragment point's 1/K0 is the intensity-weighted 1/K0 of
its events (`+0.0` where they sum to nothing). The centroid table is deleted after
retrace succeeds. The report (`stage = "retrace"`) carries extract's parameters, the reader,
and retrace's own under `params.retrace`; its `stats` carry the rebuilt and copied row
counts, the frames decoded, the raw events summed and the median MS1 mass residual
(`ms1_mass_residual_ppm_median`).

### `<psms>.peaks.parquet`, top-K peak retention (`extract.rs:1532-1543`)

Written next to the PSM table only when `extract.retain_top_peaks > 1`; one row
per (candidate, peak).

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id |
| `peak_rank` | Int32 | no | - | peak rank by co-eluting fragment breadth-area |
| `apex_rt` | Float64 | no | s | peak apex RT |
| `start_rt` | Float64 | no | s | peak start RT |
| `end_rt` | Float64 | no | s | peak end RT |
| `evidence_count` | Float64 | no | - | distinct co-eluting fragment count at the apex |
| `area` | Float64 | no | - | breadth-profile area of the peak |

---

## Stage E: `features` (`stages/features.rs`)

### features.parquet (`features.rs:913-932`)

Bookkeeping columns followed by the active feature columns. Schema version 2
(`artifact::FEATURES`): every feature column is `Float32`, holding the value the
stage computed in f64 narrowed by `v as f32`, except the five
`F64_FEATURE_COLUMNS` of `stages/features.rs` (`charge`, `n_matched_fragments`,
`unique_fragment_count`, `peak_contested_frac`, `contested_frac`), which stay
`Float64` because compete or rescore reads them as f64 before narrowing. Every
classifier narrows every feature to f32 with the same `as f32` (the native
`FeatureMatrix` is flat f32 and the sidecar handoff is Float32), so the stored f32
is the value the classifier used before, and every scored output is unchanged. The
file is about half the bytes of version 1.

Version 1 stored every feature column as `Float64`. The engine reads both
(compete, rescore, and `TableFile::f64_widening` for a single column); a v1 table
given to compete is rewritten into the `psms_competed` v4 layout. An external reader
sees `float32` columns in a v2 file; widen with `astype("float64")` where a tool
needs it, which adds no precision the engine ever used. Band tables written before
and after the change cannot be pooled into one table, because the splice requires
identical schemas; re-run the bands with one binary. The PIN (`features.emit_pin`)
is written from the f64 values before they are narrowed, so its text is the same
under both versions.

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id |
| `label` | Utf8 | no | - | `target` or `decoy` |
| `base_peptide_id` | UInt32 | no | - | grouping key |
| `peptidoform` | Utf8 | no | - | ProForma-lite peptidoform |
| `protein` | Utf8 | no | - | protein accessions |
| `apex_rt` | Float64 | no | s | selected apex RT |
| `elution_lo` | Float64 | no | s | detected elution-peak lower RT bound |
| `elution_hi` | Float64 | no | s | detected elution-peak upper RT bound |
| `precursor_mz` | Float64 | no | m/z | precursor m/z |
| `prelim_score` | Float64 | no | - | preliminary composite score (feature-derived; rescorer warm-start) |
| ...feature columns... | Float32 (Float64 for `F64_FEATURE_COLUMNS`) | no | varies | one column per name in `active_features(set)` (see below) |

The active feature list depends on `features.set`
(`features.rs:210-231`): Minimal (14), Rich (14 + 30 = 44), or Extended
(14 + 30 + extended battery + 6 psms-derived; **381** columns total, the value of
`active_features(FeatureSet::Extended).len()` confirmed by the runtime
`features=381` log. The `feature_sets_sized` test (`features.rs:1590-1605`)
asserts the structural formula `14 + 30 + extended_names().len() + 6` and name
uniqueness, not a literal count).

**Minimal feature columns** (`features.rs:158-173`):
`rt_error_abs`, `rt_error_rel`, `n_matched_fragments`, `coelution_run`,
`log_apex_intensity`, `frag_corr`, `frag_cosine`, `spectral_angle`,
`coelution_mean`, `coelution_best`, `n_coelution_above`, `charge`,
`peptide_length`, `n_proteins`.

**Rich-extra feature columns** (added for Rich and Extended,
`features.rs:176-207`):
`library_norm_manhattan`, `library_rmsd`, `xcorr_coelution`, `xcorr_shape`,
`sum_b_intensity`, `sum_y_intensity`, `diff_by_intensity`, `n_b_ions`,
`n_y_ions`, `weighted_mass_error`, `mean_mass_error`, `isotope_corr`,
`ms1_isom1_ratio`, `log_mono_ms1`, `has_ms1`, `log_sn`, `n_observations`,
`base_width_rt`, `seed_score`, `seed_identified`, `matched_fraction`,
`profile_cos`, `ref_corr`, `best_ref_corr`, `low_frag_coel`, `evidence`,
`contrast_min`, `resid_corr`, `coel_clean`, `shadow_frac`.

**psms-derived Extended extras** (added only for Extended,
`features.rs:220-228`):
`peak_contested_frac`, `peak_contested_count_frac`, `peak_apportioned_frac`,
`n_charge_states`, `charge_multi_flag`, `cross_charge_intensity_log`.

**Extended battery families.** The 15 family modules under
`stages/features/` are concatenated in the registry order at
`features.rs:52-68`. Each family exposes a `NAMES: &[&str]` array; the registry
drops any name colliding with a Minimal/Rich name and any cross-family duplicate,
keeping the first appearance (`features.rs:84-99`). The raw per-family names are
listed below (raw total 335; the deduplicated survivors are the Extended battery
columns).

- **similarity** (`similarity.rs:18-96`, 64): `spectrum_cosine_matched`,
  `spectrum_cosine_sqrt`, `spectrum_cosine_log`, `spectral_angle`,
  `spectral_angle_sqrt`, `spectral_angle_matched`, `pearson_intensity_matched`,
  `pearson_intensity_log`, `spearman_intensity`, `spearman_intensity_matched`,
  `kendall_tau_intensity`, `dot_product_raw`, `dot_product_norm`,
  `library_recall_intensity`, `manhattan_sim`, `manhattan_sqrt`, `rmsd_norm`,
  `mae_norm`, `mse_log`, `mae_weighted_pred`, `abs_diff_q3`,
  `max_positive_residual`, `chebyshev_dist`, `minkowski_p3`, `bray_curtis`,
  `bray_curtis_sqrt`, `canberra`, `canberra_matched`, `wave_hedges`,
  `chi_square_pearson`, `chi_square_symmetric`, `divergence_distance`,
  `bhattacharyya_coef`, `hellinger`, `squared_chord`, `harmonic_mean_sim`,
  `jaccard_presence`, `dice_presence`, `intensity_weighted_pearson`,
  `regression_slope`, `gini_diff`, `wasserstein_mz`, `footrule_norm`,
  `rank_overlap_top3`, `top1_frag_match`, `top1_predicted_observed`,
  `frac_top3_predicted_observed`, `count_strong_predicted_absent`,
  `frac_predicted_absent`, `cosine_area`, `pearson_area`, `spectral_angle_area`,
  `cosine_fullwindow`, `stein_scott_weighted_dot`, `log_dot_product`,
  `spectral_log_evidence`, `scribe_score`, `log_dot_product_area`,
  `spectral_log_evidence_area`, `scribe_score_area`, `cosine_high_ordinal`,
  `cosine_robust_trim1`, `cosine_robust_trim2`, `cosine_robust_trim3`.
- **entropy** (`entropy.rs:13-32`, 18): `spectral_entropy_similarity`,
  `weighted_spectral_entropy_similarity`, `spectral_entropy_similarity_sqrt`,
  `spectral_entropy_similarity_topk`, `spectral_entropy_similarity_area`,
  `jensen_shannon_divergence`, `jeffreys_divergence`, `kl_obs_pred`,
  `kl_pred_obs`, `cross_entropy_obs_pred`, `obs_spectrum_entropy`,
  `pred_spectrum_entropy`, `entropy_diff`, `entropy_ratio`,
  `obs_normalized_entropy`, `normalized_entropy_diff`,
  `residual_spectrum_entropy`, `entropy_weight_obs`.
- **coelution** (`coelution.rs:21-60`, 38): `frag_ref_corr_mean`,
  `frag_ref_corr_obsweighted`, `frag_ref_corr_min`, `frag_ref_corr_std`,
  `frag_ref_corr_sq_mean`, `frag_ref_corr_topk_weighted`,
  `n_frag_ref_corr_above_0_9`, `frac_frag_ref_corr_above_0_8`,
  `frag_ref_corr_mean_full`, `full_vs_peak_corr_gain`,
  `pairwise_coelution_weighted`, `pairwise_coelution_min`,
  `pairwise_coelution_median`, `pairwise_coelution_std`,
  `pairwise_coelution_frac_negative`, `pairwise_coelution_hi`,
  `pairwise_coelution_lo`, `coelution_hi_lo_contrast`, `coelution_corr_entropy`,
  `xcorr_shape_mean`, `xcorr_shape_min`, `xcorr_shape_std`, `xcorr_lag_mean_abs`,
  `xcorr_lag_std`, `xcorr_lag_iqr`, `xcorr_lag_frac_zero`, `xcorr_lag_max_abs`,
  `xcorr_lag_entropy`, `ref_xcorr_lag_mean`, `ref_xcorr_shape_mean`,
  `observed_sum_vs_template_corr`, `frag_loo_ref_corr_mean`,
  `frag_loo_ref_corr_min`, `frac_frags_apex_aligned`, `top3_frag_ref_corr`,
  `by_cross_coelution`, `by_cross_lag_mean`, `charge_cross_coelution`.
- **interference** (`interference.rs:17-44`, 26): `explained_variance_ref`,
  `profile_residual_fraction`, `n_interfered_fragments`, `corrected_vs_raw_cos`,
  `corrected_vs_raw_ratio`, `ifs_removed_count`, `ifs_removed_intensity_frac`,
  `ifs_corr_gain`, `ifs_retained_frac`, `matched_frac_after_ifs`,
  `peak_to_full_area_ratio_profile`, `peak_to_full_area_ratio_frag_mean`,
  `peak_to_full_area_ratio_weighted`, `out_of_peak_intensity_frac`,
  `profile_corr_full_vs_peak_delta`, `frac_frag_ref_corr_below_0_5`,
  `explained_apex_intensity_frac`, `apex_purity`,
  `interference_apex_residual_fraction`, `dominant_frag_ref_corr`,
  `explained_variance_ratio`, `second_component_fraction`,
  `profile_second_peak_ratio`, `n_competing_peaks_in_window`,
  `matched_pred_intensity_fraction`, `top_pred_frag_matched`.
- **chromatographic** (`chromatographic.rs:20-64`, 43): `gaussian_fit_r2`,
  `gaussian_cosine`, `emg_fit_improvement`, `apex_prominence`,
  `profile_peak_snr`, `fwhm_seconds`, `fwhm_to_window_ratio`, `width_at_10pct`,
  `width_ratio_10_50`, `hwhm_asymmetry`, `tailing_factor_usp`,
  `asymmetry_factor_10pct`, `apex_sharpness`, `apex_curvature`,
  `apex_to_boundary_ratio`, `apex_dominance`, `zigzag_index`, `jaggedness`,
  `roughness_2nd_deriv`, `n_local_maxima`, `modality`, `rt_skewness`,
  `rt_excess_kurtosis`, `rt_std_seconds`, `mean_mode_offset`,
  `fraction_area_within_fwhm`, `triangle_area_similarity`, `baseline_fraction`,
  `peak_completeness`, `apex_centering_offset`, `intensity_score`,
  `total_xic_log`, `frag_fwhm_cv`, `frag_fwhm_mean`, `frag_apex_rt_dispersion`,
  `frag_apex_rt_dispersion_weighted`, `frag_apex_offset_from_profile_mean`,
  `frag_gaussianity_mean`, `frag_gaussianity_weighted`, `frag_zigzag_mean`,
  `sumtrace_unweighted_gaussian_r2`, `reference_profile_rt_entropy_peak`,
  `reference_profile_rt_entropy_ratio`.
- **mass_accuracy** (`mass_accuracy.rs:19-38`, 17): `median_abs_frag_ppm`,
  `signed_mean_frag_ppm`, `ppm_std`, `ppm_iqr`, `ppm_range`, `max_abs_frag_ppm`,
  `intensity_weighted_abs_ppm`, `intensity_weighted_signed_ppm`,
  `intensity_weighted_ppm_std`, `lib_weighted_abs_ppm`,
  `frac_frag_within_half_tol`, `high_ppm_intensity_frac`, `ppm_intensity_anticorr`,
  `mass_error_mz_trend`, `mean_abs_mz_error_da`, `mass_evidence_gauss`,
  `mass_log_evidence`.
- **ion_series** (`ion_series.rs:19-54`, 34): `n_matched_b`, `n_matched_y`,
  `frac_matched_b`, `frac_matched_y`, `by_count_balance`, `by_intensity_ratio`,
  `by_ratio_agreement`, `by_ratio_consistency`, `longest_b_run`,
  `longest_y_run`, `longest_run_max`, `longest_run_frac_length`,
  `series_coverage_b`, `series_coverage_y`, `sequence_coverage`,
  `series_gap_fraction`, `by_complement_count`, `by_complement_mz_consistency`,
  `by_complement_coelution`, `ordinal_intensity_concordance_y`,
  `ordinal_intensity_concordance_b`, `series_coelution_y`, `series_coelution_b`,
  `spectral_angle_b`, `spectral_angle_y`, `pearson_b`, `pearson_y`,
  `cosine_charge1`, `cosine_charge2`, `charge_corr_balance`,
  `mean_matched_ordinal_norm`, `by_ion_contiguous_intensity`,
  `by_ion_contiguous_lib_frac`, `both_series_present`.
- **ms1** (`ms1.rs:18-49`, 25): `ms1_isotope_cosine_apex`,
  `ms1_isotope_spectral_angle_apex`, `ms1_isotope_chi2_apex`,
  `ms1_isotope_manhattan_apex`, `iso_ratio_1_0`, `iso_ratio_2_0`,
  `iso_plus1_ratio_dev`, `iso_plus2_ratio_dev`, `iso_minus_one_fraction`,
  `iso_overlap_flag`, `log_ms1_mono`, `ms1_total_isotope_log`, `has_ms1_signal`,
  `ms1_isotope_apex_entropy_3`, `ms1_m1_entropy_contribution`,
  `ms1_ms2_time_corr`, `ms1_ms2_envelope_time_corr`, `ms1_iso_coelution`,
  `ms1_ms2_apex_rt_delta`, `ms1_iso_ratio_stability`, `ms1_mono_gaussianity`,
  `ms1_ms2_fwhm_ratio`, `ms1_isotope_corr_xic`, `ms1_envelope_over_time_corr`,
  `ms1_isotope_xic_shape_consistency`.
- **rt** (`rt.rs:12-26`, 13): `rt_error_signed`, `rt_error_abs`,
  `rt_error_squared`, `rt_error_signed_norm_gradient`,
  `rt_error_abs_norm_gradient`, `observed_rt_raw`, `predicted_rt_raw`,
  `observed_rt_fraction`, `predicted_rt_fraction`, `rt_error_over_peak_width`,
  `rt_error_over_fwhm`, `rt_diff_profile_apex`, `predicted_rt_in_gradient`.
- **novel** (`novel.rs:16-29`, 12): `log_seed_hyperscore`,
  `seed_hyperscore_per_matched`, `seed_identified`, `peptide_length`,
  `precursor_charge`, `charge_is_2`, `charge_is_3`, `charge_is_4plus`,
  `precursor_mass`, `log_total_matched_intensity`, `n_matched_frags`,
  `n_predicted_frags`.
- **nonzero** (`nonzero.rs:17-30`, 12): `frag_corr_peakmax`,
  `frag_cosine_peakmax`, `spectral_angle_peakmax`, `frag_corr_matched_nz`,
  `frag_cosine_matched_nz`, `peakmax_apex_gain`, `n_frag_present_inpeak`,
  `frac_frag_present_inpeak`, `coelution_mean_bothpos`, `coelution_mean_summpos`,
  `ref_corr_nz`, `profile_cos_nz`.
- **order_consistency** (`order_consistency.rs:27-36`, 8):
  `rank_corr_vs_apex_mean`, `rank_corr_vs_apex_std`, `rank_corr_adjacent_mean`,
  `kendall_vs_apex_mean`, `top1_frag_persistence`, `top2_order_persistence`,
  `argmax_frag_entropy`, `self_cosine_vs_apex_mean`.
- **peak_scans** (`peak_scans.rs:16`, 2): `n_peak_scans`,
  `peak_window_degenerate`.
- **apex_dispersion** (`apex_dispersion.rs:15-29`, 13): `frag_apex_rt_std`,
  `frag_apex_rt_mad`, `frag_apex_max_dev`, `frag_apex_mean_dev`,
  `frag_apex_agree_frac`, `precursor_frag_apex_delta`, `peak_symmetry`,
  `peak_tailing`, `peak_n_local_maxima`, `peak_shoulder_score`,
  `peak_fwhm_scans`, `peak_truncation`, `apex_frac_of_window`.
- **mass_uncertainty** (`mass_uncertainty.rs:18-29`, 10): `frag_mass_err_median`,
  `frag_mass_err_abs_median`, `frag_mass_err_std`, `frag_mass_err_iqr`,
  `frag_mass_err_max_abs`, `frag_mass_err_range`, `effective_frag_count`,
  `evidence_concentration`, `frac_top3_pred_observed`, `frac_top5_pred_observed`.

### `<features>.schema.json` (`features.rs:936-942`)

Feature schema companion carried forward to compete/rescore so the classifier
input is reproducible.

| field | JSON type | meaning |
|---|---|---|
| `feature_columns` | array of strings | ordered active feature list |
| `schema_id` | string | blake3 hash of the joined column list |

### PIN (`features.rs:1537-1573`)

Percolator input, tab-separated text (written to `out_pin`). Header:
`SpecId  Label  ScanNr  ExpMass  CalcMass  <feature cols...>  Peptide  Proteins`.

| field | meaning |
|---|---|
| `SpecId` | `cand_<candidate_id>` |
| `Label` | 1 for target, -1 for decoy |
| `ScanNr` | `candidate_id` |
| `ExpMass` | precursor m/z (5 decimals) |
| `CalcMass` | precursor m/z (5 decimals; same value) |
| feature columns | active feature values (6 decimals), one per feature |
| `Peptide` | `-.<peptidoform>.-` |
| `Proteins` | protein-accession string |

---

## Stage F: `compete` (`stages/compete.rs`)

### psms_competed (`compete.rs:121-151`)

Bookkeeping columns plus the feature columns carried from the schema. Schema
version 4 (`artifact::PSMS_COMPETED`) stores each feature column in the width
`features.parquet` v2 does (`feature_storage_type`: `Float32`, or `Float64` for
`F64_FEATURE_COLUMNS`); version 3 stored every feature column as `Float64`.
Rescore reads both and scores them identically. Compete given a v1 features table
narrows its Float32 features with `as f32` into the v4 layout. A `<out>.schema.json`
companion is copied forward (`compete.rs:153`).

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id |
| `label` | Utf8 | no | - | `target` or `decoy` |
| `base_peptide_id` | UInt32 | no | - | grouping key |
| `peptidoform` | Utf8 | no | - | ProForma-lite peptidoform |
| `protein` | Utf8 | no | - | protein accessions |
| `apex_rt` | Float64 | no | s | apex RT |
| `elution_lo` | Float64 | no | s | identified elution lower bound |
| `elution_hi` | Float64 | no | s | identified elution upper bound |
| `precursor_mz` | Float64 | no | m/z | precursor m/z |
| `prelim_score` | Float64 | no | - | preliminary composite score |
| ...feature columns... | Float32 (Float64 for `F64_FEATURE_COLUMNS`) | no | varies | each `feature_columns` entry from the schema |

### `<out>.compete_audit.parquet` (`compete.rs:167-202`)

Conditional, written only when `compete.emit_competition_audit`; one row per
candidate removed by within-group competition.

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | removed candidate id |
| `label` | Utf8 | no | - | `target` or `decoy` (same label as its winner) |
| `peptidoform` | Utf8 | no | - | removed peptidoform |
| `winner_candidate_id` | UInt32 | no | - | candidate id that outcompeted it |
| `loser_prelim` | Float64 | no | - | loser preliminary score |
| `winner_prelim` | Float64 | no | - | winner preliminary score |
| `rejection_reason` | Utf8 | no | - | `OUTCOMPETED_BY_TARGET` or `OUTCOMPETED_BY_DECOY` |

---

## Stage F: `rescore` (`stages/rescore.rs`)

### psms_scored (schema v3, `rescore.rs:508-541`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id |
| `peptidoform` | Utf8 | no | - | ProForma-lite peptidoform |
| `charge` | Int32 | no | - | precursor charge |
| `label` | Utf8 | no | - | `target` or `decoy` |
| `protein` | Utf8 | no | - | protein accessions |
| `base_peptide_id` | UInt32 | no | - | grouping key |
| `apex_rt` | Float64 | no | s | identification apex carried through for quant |
| `elution_lo` | Float64 | no | s | identified elution lower bound |
| `elution_hi` | Float64 | no | s | identified elution upper bound |
| `score` | Float64 | no | - | rescorer discriminant score (higher is better) |
| `q_value` | Float64 | no | - | PSM-level q-value (pooled target-decoy or entrapment null) |
| `peptide_q_value` | Float64 | no | - | peptide-level q (best PSM per base peptide); losers get 1.0 |
| `protein_group` | Utf8 | no | - | protein-accession-set string (decoys carry `DECOY_`) |
| `pg_q_value` | Float64 | no | - | protein-group-level q (best PSM per protein group); losers get 1.0 |
| `global_q_value` | Float64 | no | - | byte-identical alias of pooled `q_value` (backward-compat) |
| `prelim_score` | Float64 | no | - | preliminary composite score carried forward |
| `source` | UInt32 | no | - | index into `--competed` inputs (run identity); all-zero single-run |
| `run_psm_q` | Float64 | no | - | per-run PSM q (TDA within each `source`) |
| `experiment_psm_q` | Float64 | no | - | pooled PSM q (equals `q_value`) |
| `precursor_q` | Float64 | no | - | per-precursor q (best PSM per peptidoform + charge); losers get 1.0 |

Which null the q-values use depends on the classifier: target-decoy for
`NativeTda`/`Mokapot`/`NnTorch`/`Percolator`, entrapment for the `Entrapment`
rescorer. The q-value branch is at `rescore.rs:289-297`; the `qmode` selector
defaults to target-decoy (`rescore.rs:145`) and switches to entrapment only
inside the `Entrapment` classifier arm (`rescore.rs:252-273`).

#### q-value column units

The table carries seven q columns and they are not interchangeable. A reported
count is only interpretable when both the row unit and the q column used to
select it are named.

| column | q computed over | populated on | produced at |
|---|---|---|---|
| `q_value` | every PSM row, pooled across all `source` values | every row | `rescore.rs:522` |
| `experiment_psm_q` | same population as `q_value` | every row | `rescore.rs:535` |
| `global_q_value` | same population as `q_value` (backward-compat alias) | every row | `rescore.rs:526` |
| `run_psm_q` | PSM rows within one `source`, target-decoy re-run per run | every row | `rescore.rs:411`, written at `rescore.rs:534` |
| `precursor_q` | best PSM per `(peptidoform, charge)` | that group's winning row only; losers 1.0 | `rescore.rs:449`, written at `rescore.rs:536` |
| `peptide_q_value` | best PSM per `base_peptide_id` (stripped sequence) | that group's winning row only; losers 1.0 | `rescore.rs:370`, written at `rescore.rs:523` |
| `pg_q_value` | best PSM per `protein_group` | that group's winning row only; losers 1.0 | `rescore.rs:392`, written at `rescore.rs:525` |

The three grouped columns are sparse by construction. `grouped_q`
(`rescore.rs:673`) picks the best-scoring row of each group, computes the q over
that reduced set, and then scatters the value back to the winning row alone,
leaving 1.0 on every losing sibling (`rescore.rs:721-728`). This is deliberate: a
lower-scoring charge or modification variant, which may itself be a false target,
must not inherit the winner's q. Deduplicated peptide, precursor, and
protein-group counts are unaffected because they select the winner anyway.

One further caveat applies to `precursor_q` specifically. It groups on
`(peptidoform, charge)` over the rows that reach rescore, and under the default
`compete.group_by = base_peptide` competition has already collapsed the charge and
modification siblings by stripped peptide, so the surviving population holds
roughly one form per peptide and the column then reports a base-peptide count.
It is a genuine precursor unit only under `compete.group_by = peptidoform_charge`
(docs/11_compete_rescore_fdr.md).

The consequence for multi-run work is that grouping in an experiment-wide rescore
is also experiment-wide, so each group has exactly one populated row across the
whole experiment. Splitting the pooled table by `source` and counting a single
run's rows on a grouped column keeps only about `1/n_runs` of the rows it should
and is not a per-run number. Use `run_psm_q` for a per-file PSM count or a
per-file quant gate; use the grouped columns only when rescore itself was
single-run. `quant.q_filter` selects among `peptide_q_value`, `precursor_q`,
pooled `q_value`, and `run_psm_q`, and does not select a run
(docs/12_quant_lfq_align_mbr_report_audit.md).

---

## Stage G: `quant` (`stages/quant.rs`)

### peptide_quant (schema v2, `quant.rs:487-502`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id |
| `base_peptide_id` | UInt32 | no | - | unique base-peptide rollup key |
| `peptidoform` | Utf8 | no | - | ProForma-lite peptidoform |
| `charge` | Int32 | no | - | precursor charge |
| `protein_group` | Utf8 | no | - | protein-accession-set string |
| `quantity` | Float64 | yes | intensity x s | sum of top-N positive finite fragment areas; null when not quantifiable |
| `quant_status` | Utf8 | no | - | quantified or explicit missing-quantity reason |
| `n_fragments_used` | Int32 | no | - | number of positive finite fragments summed |
| `integration_apex_rt` | Float64 | yes | s | apex actually used for integration |
| `integration_lo_rt` | Float64 | yes | s | integration lower bound actually used |
| `integration_hi_rt` | Float64 | yes | s | integration upper bound actually used |
| `first_pass_id` | Boolean | no | - | v3 only: the precursor had a first-pass identification in this run (`run_psm_q <= mbr.q_anchor`); written by `run-experiment` after a second pass under `mbr.first_pass_flag` |

### protein_group_quant (schema v2, `quant.rs:519-527`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `protein_group` | Utf8 | no | - | protein-accession-set string |
| `quantity` | Float64 | yes | intensity x s | rollup over unique positive base-peptide representatives |
| `quant_status` | Utf8 | no | - | quantified or `no_quantifiable_peptide` |
| `n_peptides` | Int32 | no | - | unique positive base peptides before Top-N truncation |

### fragment_quant (optional, `quant.rs:558-568`)

Written only when `out_fragment` is set (ion-level directLFQ input).

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id |
| `peptidoform` | Utf8 | no | - | ProForma-lite peptidoform |
| `charge` | Int32 | no | - | precursor charge |
| `protein_group` | Utf8 | no | - | protein-accession-set string |
| `fragment_name` | Utf8 | no | - | fragment name |
| `quantity` | Float64 | no | intensity x s | per-fragment trapezoid area over the peak window |

Schema v1 holds only these columns and only the positive finite areas. Under the cross-run
export (`quant --export-cross-run`, or the first pass of the cross-run step of
`run-experiment`; docs/12_quant_lfq_align_mbr_report_audit.md, "Cross-run fragment selection
and background") the table is v5: the zero-area fragments are kept, with `quantity` 0, and
these columns follow. The number is above the IM branch's v2 to v4, whose meanings were not
documented. Every value is NaN for a candidate integrated without a fixed window.

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `apex_corr` | Float64 | no | - | population Pearson correlation of the fragment's fixed-window samples (after the per-run baseline, before the envelope) with the sum of the candidate's other fragments; 0 for a flat window |
| `flank_mean` | Float64 | no | counts | mean of the `baseline_flank_scans` raw samples either side of the fixed window; NaN without a flank sample |
| `flank_mean_h3` .. `flank_mean_h7` | Float64 | no | counts | the same mean beyond a window of +/-3 to +/-7 scans around the apex |
| `peak_hwhm` | Float64 | no | scans | the candidate's half width at half maximum in this run: of its summed fragment trace within +/-7 scans of the apex, each fragment less its `flank_mean`; one value per candidate, repeated on its rows |

Under a pattern rule (`quant.fragment_selection = pattern_fit` or `pattern_profile`) the
export is v6: v5 plus these columns, for the pattern fit of the cross-run step
(docs/12_quant_lfq_align_mbr_report_audit.md, "Profile-centred window and pattern fit").

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `pattern_area` | Float64 | no | intensity x s | the fragment's trapezoid area over the window samples around the candidate's window centre, in f64; 0 without a trace |
| `observed` | Boolean | no | - | the fragment has a trace in this run |
| `window_rt` | List<Float32> | no | s | the candidate's retention times within twice the halfwidth of the window centre, on the candidate's first row; empty on its other rows |

### peak-bounds diagnostic (optional, `quant.rs:421-429`)

Written only when `out_peak_bounds` is set and `bound_peak` is on.

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id |
| `lo_rt` | Float64 | no | s | integration window lower RT |
| `hi_rt` | Float64 | no | s | integration window upper RT |
| `width_s` | Float64 | no | s | `hi_rt - lo_rt` |

### quant-lfq cross-run output (`quant.rs:720-728`, `run_lfq_combine` at `quant.rs:657`)

Long-form MaxLFQ (peptide-level) or directLFQ (ion-level) matrix.

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `protein_group` | Utf8 | no | - | protein-accession-set string |
| `run` | Int32 | no | - | input run index (0-based) |
| `quantity` | Float64 | no | intensity | LFQ profile value for that protein in that run |
| `n_features` | Int32 | no | - | number of features contributing to the protein |

---

## Stage D2: `align` (`stages/align.rs`)

Experiment-level cross-run stage that prepares MBR. It puts every run on a common
RT coordinate: the first input seed is the reference, and for each run it fits a
monotone LOESS mapping from the run's observed RT to the reference RT over
confidently shared peptides, then emits that mapping on a `grid_n`-point grid.
The reference run (or a run with fewer than four shared peptides) uses the
identity mapping. This artifact is written with an inline `alignment` (v1)
`ArtifactReport` (`align.rs:141-153`); it is not registered in the frozen
`schema.rs` artifact table.

### alignment (`align.rs:130-138`)

One row per (run, grid point).

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `run_id` | UInt32 | no | - | 0-based input run index (position in the seed list); run 0 is the reference |
| `source_rt` | Float64 | no | s | grid point in this run's own observed RT coordinate |
| `reference_rt` | Float64 | no | s | RT mapped onto the reference coordinate; identity for the reference run or when fewer than four peptides are shared with the reference |
| `residual_spread` | Float64 | no | s | 95th-percentile absolute LOESS residual on shared peptides for this run; 0.0 for the reference or an unfit run; bounds how tight an MBR window can be |

---

## Match-between-runs second pass (`stages/second_pass.rs`, `run-experiment`)

The default `mbr.strategy = second_pass` adds these to an experiment directory
(docs/12, "Second pass"):

| path | content |
|---|---|
| `first_pass/scored_combined.parquet` | the pooled first pass, `psms_scored` schema |
| `second_pass/library/lib_precursors.parquet` | `fragment_library_precursors`: the first library's rows of every experiment-wide accepted target and its paired decoy, ids contiguous from 0, `predicted_irt` = cross-run consensus (s, reference run's scale); on an experiment with ion mobility and a library with `predicted_im`, `predicted_im` = cross-run consensus apex 1/K0 (reference run's scale), the decoy taking its target's value in both columns |
| `second_pass/library/lib_fragments.parquet` | `fragment_library_fragments` of those rows, ids remapped |
| `second_pass/library/entries.parquet` | `candidate_id` (u32) of the first library's rows kept |
| `second_pass/library/empirical_library.json` | counts and alignment summary (`entries`, `targets`, `decoys`, `reference_source`, `runs_aligned`, `runs_unaligned`, `entries_from_winner`, `entries_without_decoy`, `median_observations`, `q`); `im` and `diagnostics` below, each only on an experiment with ion mobility |
| `second_pass/<run>/` | the run's seed, windows, calibration, extraction, features and competed table on the empirical library |

The root `scored_combined.parquet` is then the second pass's, its `candidate_id` in the
empirical library's numbering.

`empirical_library.json`, `im` (present when some run's first-pass `psms_extracted` carries an
apex 1/K0):

| field | JSON type | meaning |
|---|---|---|
| `reference_source` | integer or null | the run every run's 1/K0 was moved onto: the retention-time reference |
| `runs_aligned` | integer | runs moved by their median offset, the reference included |
| `runs_unaligned` | integer | runs with apex 1/K0 values sharing fewer than 200 precursors with the reference; they keep their own scale |
| `offsets` | object | per `source`, the median of (reference - run) apex 1/K0 over the shared precursors; 0 for the reference |
| `entries_from_runs` | integer | entries whose 1/K0 is the median over their confident runs |
| `entries_from_winner` | integer | entries with no confident run carrying a 1/K0, which take their winning row's |
| `entries_without_im` | integer | entries with no finite 1/K0, which keep the library's `predicted_im` |

`empirical_library.json`, `diagnostics` (written after the pooled second-pass rescore; logged on
every experiment, written on one with ion mobility). Rows are scored target or decoy rows at
`run_psm_q <= q`; a decoy belongs to the entry it is paired with, and an entry's anchor runs are
the runs where the first pass accepted it:

| field | JSON type | meaning |
|---|---|---|
| `q` | number | `mbr.q_anchor` |
| `runs` | array | per run: `source`, `first_pass_accepted` and `second_pass_accepted` (targets), `new` (accepted by the second pass only), `dropped_in_library` (accepted by the first pass only, among the entries), `dropped_outside_library` (accepted by the first pass in this run but not experiment-wide, so never searched again) |
| `targets_anchor`, `decoys_anchor` | integer | accepted second-pass targets and decoys in their entry's anchor runs |
| `targets_non_anchor`, `decoys_non_anchor` | integer | the same in the other runs |
| `decoy_fraction_anchor`, `decoy_fraction_non_anchor` | number or null | decoys over targets of each group; null without targets |

## `audit` (`stages/audit.rs`)

### candidate_audit.parquet (`audit.rs:180-200`)

One row per library candidate (the full search space), with per-stage survival
flags and the earliest rejection reason.

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `run_id` | Utf8 | no | - | run identifier stamped on every row |
| `precursor_id` | UInt32 | no | - | library `candidate_id` |
| `modified_sequence` | Utf8 | no | - | ProForma-lite peptidoform |
| `charge` | Int32 | no | - | precursor charge |
| `target_decoy_label` | Utf8 | no | - | `target` or `decoy` |
| `entrapment_label` | Boolean | no | - | protein contains the entrapment substring |
| `candidate_generated` | Boolean | no | - | in the search space (always true) |
| `traces_extracted` | Boolean | no | - | produced fragment traces in extract |
| `peak_generated` | Boolean | no | - | an accepted peak exists (equals `traces_extracted` at artifact resolution) |
| `peak_selected` | Boolean | no | - | selected as the reported peak (same resolution) |
| `variant_selected` | Boolean | no | - | survived within-group competition |
| `target_decoy_winner` | Boolean | no | - | present in the scored table |
| `passed_precursor_fdr` | Boolean | no | - | q-value <= threshold |
| `passed_peptide_fdr` | Boolean | no | - | peptide-q and precursor gate both pass |
| `reported` | Boolean | no | - | passed the precursor FDR gate |
| `rejection_reason` | Utf8 | no | - | earliest-loss reason code (e.g. `REPORTED`, `NO_PEAK_GROUP`, `OUTCOMPETED_BY_TARGET`, `FAILED_PRECURSOR_FDR`) |

### `<audit>.metrics.json` (`audit.rs:206-216`)

| field | JSON type | meaning |
|---|---|---|
| `run_id` | string | run identifier |
| `q_threshold` | number | FDR threshold used |
| `search_space` | integer | total library candidates |
| `extracted` | integer | candidates with accepted peaks |
| `competed` | integer | competition survivors |
| `reported` | integer | candidates reported at threshold |
| `trace_recall` | number | extracted / search_space |
| `waterfall` | object | map of rejection-reason code to count |

---

## Cross-stage JSON sidecars

### `<artifact>.report.json` (`mumdia-io/src/report.rs:10-24`)

Written next to most primary stage Parquets, but not every Parquet or stage
output. Optional diagnostics, schema/PIN files, TSV reports, and some
experiment-level/Python-written outputs have partial or no coverage.

| field | JSON type | meaning |
|---|---|---|
| `logical_name` | string | artifact logical name (e.g. `spectra_ms2`) |
| `schema_name` | string | frozen schema name |
| `schema_version` | integer | schema version (from `mumdia-core/src/schema.rs`) |
| `stage` | string | producing stage (e.g. `convert`, `extract`) |
| `rows` | integer | row count |
| `content_hash` | string | blake3 hash of the artifact file |
| `params` | object | resolved parameters the stage actually used |
| `stats` | object | summary key distributions / metrics |
| `model_identity` | string or null | predictor/rescorer model identity where applicable |
| `elapsed_ms` | integer | stage wall-clock time (ms) |

### manifest.json (`mumdia-core/src/manifest.rs:9-31`)

Written by the `run` orchestrator (`stages/run.rs:500-501`). Records per-artifact
provenance for reproducibility. Top-level `Manifest`:

| field | JSON type | meaning |
|---|---|---|
| `mumdia_version` | string | crate version |
| `config_json` | string | fully-resolved config JSON |
| `config_hash` | string | hash of the resolved config |
| `model_identities` | object | map of role (`rt_predictor`, `fragment_predictor`, `rescorer`, `feature_schema_id`) to identity string |
| `artifacts` | object | map of logical name to `ArtifactRecord` |

Each `ArtifactRecord`:

| field | JSON type | meaning |
|---|---|---|
| `logical_name` | string | artifact logical name |
| `path` | string | output path |
| `format` | string | file format (Parquet) |
| `schema_name` | string | schema name |
| `schema_version` | integer | schema version |
| `rows` | integer | row count |
| `content_hash` | string | blake3 hash |
| `producing_stage` | string | stage that wrote it |
| `config_hash` | string | config hash at write time |

---

## TSV reports (`peptides.tsv`, `proteins.tsv`)

Not Parquet, but these are the files most users actually read, so they are
documented here rather than left to the source. Written by `report`
(`stages/report.rs`): `report::run` inside `run` and the `mumdia report` handler,
and `report::run_experiment` inside `run-experiment` (also `mumdia report
--experiment-dir`), which writes one experiment-wide pair at the experiment root
with an `n_runs` column and one quantity column per run and no per-run TSVs; take
per-run counts from the per-run split `scored.parquet` tables on `run_psm_q`.

**Acceptance rule, both writers.** A row is written when it is a target and either
its grouped q is at or under the threshold or it is flagged `is_transferred` by
`mumdia mbr`. The two columns `is_transferred` and `transfer_q` export that basis
(docs/29 #19): a transferred row keeps its grouped q, which is usually 1.0, and a
tighter `--q` does not revoke a transfer that already passed `mbr.q_transfer`. A
transfer is a peptide-level acceptance, not protein-group confidence; a protein group
admitted through a transferred row carries the flag so the reader can tell.

Values are formatted for reading, not for analysis: `q_value` is printed to six
decimals, `score` to four, and `quantity` to one
(`report.rs:39-45,110-120,144`). Use `peptide_quant.parquet` and
`protein_group_quant.parquet` for anything quantitative.

### `peptides.tsv` (`report.rs:95-122`)

One row per confident precursor, deduplicated on `(peptidoform, charge)` keeping
the best q (`report.rs:90,105-108`). Targets only, filtered on
`peptide_q_value <= --q` (default 0.01, `report.rs:101`).

| column | source | notes |
|---|---|---|
| `precursor` | `peptidoform` | ProForma peptidoform, modifications included |
| `stripped_sequence` | `strip(peptidoform)` | residues only; drops a `DECOY_` prefix and every bracketed or parenthesised modification block (`report.rs:24-37`) |
| `charge` | `charge` | precursor charge |
| `protein` | `protein` | the PSM's protein string, not the inferred group |
| `q_value` | `peptide_q_value` | **base-peptide** q, six decimals |
| `score` | `score` | rescorer score, four decimals |
| `quantity` | `peptide_quant.parquet` joined on `(peptidoform, charge)` | empty when the precursor was not quantifiable or no `--peptide-quant` was passed (`report.rs:39-45,61-72`) |
| `is_transferred` | `is_transferred` (`mumdia mbr`), `false` without it | `true` when the row was admitted as a match-between-runs transfer rather than on its q |
| `transfer_q` | `transfer_q` (`mumdia mbr`) | the transfer q the row was accepted at, six decimals; empty on every other row |

The row unit and the filter column deliberately disagree, and this is the single
most common misreading of MuMDIA output: rows are precursors
`(peptidoform, charge)` while the threshold is applied to `peptide_q_value`,
which is a base-peptide q. A row count is therefore **not** a
precursor-q-controlled count. The stage logs both numbers, `precursors` and
`stripped_sequences` (`report.rs:149-154`); quote whichever you mean and name it.

### `proteins.tsv` (`report.rs:133-146`)

One row per confident protein group, deduplicated keeping the best q. Targets
with a non-empty group, filtered on `pg_q_value <= --q` (`report.rs:137`).

| column | source | notes |
|---|---|---|
| `protein_group` | `protein_group` | accession set as grouped by rescore |
| `q_value` | `pg_q_value` | protein-group q, six decimals |
| `quantity` | `protein_group_quant.parquet` joined on `protein_group` | empty when not quantifiable or no `--protein-quant` was passed |
| `is_transferred` | `is_transferred` of the admitting row | `true` when the group entered through a transferred peptide row; a transfer is not protein-group confidence |
| `transfer_q` | `transfer_q` of the admitting row | empty unless `is_transferred` |

Both grouped q columns (`peptide_q_value`, `pg_q_value`) are written only to each
group's single winning row, so under an experiment-wide rescore the grouping is
experiment-wide and a per-run count taken from them is diluted by roughly one over
the number of runs. See `docs/11_compete_rescore_fdr.md`.

## Not covered here

Sidecar worker I/O in the working directory (the mokapot/NN PIN, the entrapment
GBM input and output Parquet, the MS2PIP and DeepLC exchange files) are transient
contract files, not engine artifacts.

## Schema-version registry

`mumdia-core/src/schema.rs` freezes the `(logical name, version)` pairs. As of
`schema.rs:7-25`: `psms_scored` and `psms_competed` are v4, and `features`,
`psms_extracted`, `peptide_quant` and `protein_group_quant` v2; every other
registered artifact is v1. Artifacts with a 1/K0 variant take their version from a selector
(`artifact::X(has_im)`) and record v1 or main's version on every 3D run: `spectra_ms1`,
`spectra_ms2`, `isolation_windows`, `seed_psms` and, with an ion-mobility source,
`fragment_library_precursors` are v2 on 4D data, `psms_extracted` v4 and `chromatograms` v3
or v4 (docs/35_diapasef.md). Each artifact stamps its version into its own `report.json`
(`mumdia-io/src/report.rs:14`) and into `manifest.json`
(`mumdia-core/src/manifest.rs:15`), so an artifact on disk states its own version
rather than inheriting the engine's.
