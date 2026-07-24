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

### spectra_ms2 (`convert.rs:198-213`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `scan_index` | UInt32 | no | - | monotonic scan index (shared axis with MS1) |
| `id` | Utf8 | no | - | vendor spectrum id string |
| `rt_seconds` | Float64 | no | s | scan start time |
| `window_id` | UInt32 | no | - | dense id of the distinct isolation window (`(lower,upper)` bits), matches `isolation_windows.window_id` |
| `window_target` | Float64 | no | m/z | isolation-window target m/z (0.0 for AIF/all-ion) |
| `window_lower` | Float64 | no | m/z | isolation-window lower bound (0.0 for AIF full-range fallback) |
| `window_upper` | Float64 | no | m/z | isolation-window upper bound (1.0e6 for AIF full-range fallback) |
| `precursor_mz` | Float64 | yes | m/z | selected precursor m/z, null when absent |
| `precursor_charge` | Int32 | yes | - | precursor charge, null when absent |
| `mz` | List\<Float32\> | yes | m/z | centroided fragment m/z values |
| `intensity` | List\<Float32\> | yes | counts | fragment intensities aligned to `mz` |

### isolation_windows (`convert.rs:215-226`)

The distinct-window column is `window_id` (verified: `convert.rs:218`).

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `window_id` | UInt32 | no | - | dense window id (0-based first-seen order) |
| `target` | Float64 | no | m/z | window target m/z |
| `lower` | Float64 | no | m/z | window lower bound |
| `upper` | Float64 | no | m/z | window upper bound |

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

### `<seed>.masscal.json` (`search_seed.rs:217-227`)

Per-run fragment mass recalibration sidecar consumed by `extract`.

| field | JSON type | units | meaning |
|---|---|---|---|
| `frag_ppm_offset` | number | ppm | median systematic fragment mass offset (0.0 when < 20 calibrants) |
| `frag_tol_ppm` | number | ppm | learned fragment tolerance (95th percentile of centered deviations x1.5, floor 5.0); falls back to the configured tolerance when too few calibrants |
| `frag_ppm_sigma` | number | ppm | local mass-uncertainty estimate (equal to `frag_tol_ppm`) |
| `n_dev` | integer | - | number of matched-fragment ppm deviations collected |
| `cal_passes` | integer | - | calibration passes run (0 = fallback, 1 = single, 2 = robust two-pass) |

---

## Stage B: `rt-im-train` (`stages/rt_im_train.rs`)

### run_windows (`rt_im_train.rs:266-277`)

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id (one row per library candidate) |
| `rt_pred_cal` | Float64 | no | s | predicted iRT calibrated to this run's observed RT |
| `rt_lo` | Float64 | no | s | lower RT window bound (`rt_pred_cal - width`) |
| `rt_hi` | Float64 | no | s | upper RT window bound (`rt_pred_cal + width`) |
| `im_pred_cal` | Float64 | yes | 1/K0 | calibrated ion mobility; always null in the 3D MVP |
| `im_lo` | Float64 | yes | 1/K0 | IM window lower bound; always null |
| `im_hi` | Float64 | yes | 1/K0 | IM window upper bound; always null |

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

---

## Stage D: `extract` (`stages/extract.rs`)

### psms_extracted (`extract.rs:1474-1495`, base columns)

One row per accepted candidate (the selected apex PSM).

| column | Arrow type | nullable | units | meaning |
|---|---|---|---|---|
| `candidate_id` | UInt32 | no | - | library candidate id |
| `apex_rt` | Float64 | no | s | selected apex retention time |
| `apex_im` | Float64 | yes | 1/K0 | apex ion mobility; always null in the 3D MVP |
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

Bookkeeping columns followed by the active feature columns. Every feature column
is `Float64`.

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
| ...feature columns... | Float64 | no | varies | one column per name in `active_features(set)` (see below) |

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

Bookkeeping columns plus the feature columns carried from the schema (each
`Float64`). A `<out>.schema.json` companion is copied forward
(`compete.rs:153`).

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
| ...feature columns... | Float64 | no | varies | each `feature_columns` entry from the schema |

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

### psms_scored (schema v3, `rescore.rs:447-476`)

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
| `pg_q_value` | Float64 | no | - | protein-group-level q |
| `global_q_value` | Float64 | no | - | byte-identical alias of pooled `q_value` (backward-compat) |
| `prelim_score` | Float64 | no | - | preliminary composite score carried forward |
| `source` | UInt32 | no | - | index into `--competed` inputs (run identity); all-zero single-run |
| `run_psm_q` | Float64 | no | - | per-run PSM q (TDA within each `source`) |
| `experiment_psm_q` | Float64 | no | - | pooled PSM q (equals `q_value`) |
| `precursor_q` | Float64 | no | - | per-precursor q (grouped on peptidoform + charge) |

Which null the q-values use depends on the classifier: target-decoy for
`NativeTda`/`Mokapot`/`NnTorch`/`Percolator`, entrapment for the `Entrapment`
rescorer. The q-value branch is at `rescore.rs:289-297`; the `qmode` selector
defaults to target-decoy (`rescore.rs:145`) and switches to entrapment only
inside the `Entrapment` classifier arm (`rescore.rs:252-273`).

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

## Not covered here

`report` writes `peptides.tsv` and `proteins.tsv` (`stages/run.rs:459-461`), which
are human-readable TSV, not Parquet, and are outside this Parquet data dictionary.
Sidecar worker I/O in the working directory (the Mokapot/NN PIN, the entrapment
GBM in/out Parquet, MS2PIP/DeepLC exchange files) are transient contract files,
not engine artifacts.

## Schema-version registry

`mumdia-core/src/schema.rs` freezes the `(logical name, version)` pairs.
`psms_competed`, `peptide_quant`, and `protein_group_quant` are v2;
`psms_scored` is v3; the remaining registered artifacts are v1.
