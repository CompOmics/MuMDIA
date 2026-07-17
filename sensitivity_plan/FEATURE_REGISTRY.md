# MuMDIA Feature Registry

Companion to `ARCHITECTURE_MAP.md`. This registry documents every scoring feature
MuMDIA can compute for a PSM, its family, its level, its expected monotone
direction, and its source file. It exists to satisfy the sensitivity program's
feature-registry requirement (spec `03_feature_evaluation.md` §2): before any
feature is added, removed, or ablated, the current set must be enumerated with
enough metadata to run the leakage checks (§3), the evidence ladder (§5), and the
family ablations (§4).

A machine-readable copy of this table is `feature_registry.yaml` at the repository
root (one entry per feature: `family`, `level`, `direction`, `source_file`).

> Update: the Extended battery now has **383** features (was 356 at first
> writing; the registry documents 383 including the Minimal/Rich tiers). Two
> families were added this session (append-only, so the Extended-count test is
> dynamic and stays green):
> - **apex_dispersion** (13, `stages/features/apex_dispersion.rs`, P5.3): per-
>   fragment apex-RT scatter (std/mad/max/mean deviation, agreement fraction),
>   precursor-fragment apex delta, and consensus peak shape (symmetry, tailing,
>   local maxima, shoulder, FWHM, truncation, apex position in window).
> - **mass_uncertainty** (10, `stages/features/mass_uncertainty.rs`, P5.1/P5.2):
>   fragment mass-error distribution (median/abs-median/std/IQR/max/range),
>   effective fragment count, evidence concentration, and fraction of top-3/top-5
>   predicted ions observed (breadth of the strong ions).
>
> These are computed from the per-PSM `Evidence` (peak-bounded traces + mass
> errors) and are interference-resistant (shape and breadth, not absolute
> intensity). They are registered in `feature_registry.yaml`; the per-family
> tables below predate them.

## 1. Feature-set tiers

The active feature set is selected by `features.set` (`FeatureSet` enum,
`rust/mumdia/crates/mumdia-core/src/config.rs:110`). Tiers are cumulative:

| Tier | `FeatureSet` | Count | Definition |
|---|---|---:|---|
| Minimal | `Minimal` (default) | 14 | RT error, matched fragments, co-elution, apex intensity, library agreement, metadata (`MINIMAL_FEATURES`, `features.rs:149`). |
| Rich | `Rich` / `Custom` | 44 | Minimal + 30 (`RICH_EXTRA`, `features.rs:167`). `Custom` is aliased to `Rich`. |
| Extended | `Extended` | 356 | Rich + the 12-family Evidence battery (deduplicated) + 4 psms-derived / cross-candidate extras. Enabled by `--profile dia`. |

The tier sizes are asserted by the `feature_sets_sized` test
(`features.rs:1263`): `Minimal = 14`, `Rich = 44`, and
`Extended = 14 + 30 + extended_names().len() + 4`.

This registry documents 360 feature definitions across 17 families. Four of them
(`spectral_angle`, `rt_error_abs`, `peptide_length`, `seed_identified` in their
extended-family form) collide with reserved Minimal/Rich names and are dropped
from the scored schema (see §3 of the mechanism below), so the Extended tier
scores 356 distinct columns.

## 2. Registry mechanism (how the schema is built and frozen)

All feature logic lives in `rust/mumdia/crates/mumdia/src/stages/features.rs` and
the per-family modules under `stages/features/`.

- `FAMILIES` (`features.rs:49`) is the ordered, append-only const array of the 12
  extended-battery families: `[(NAMES, values_fn)]`. Family order defines the
  frozen extended-schema column order. New families are appended here.
- Each family module exposes the fixed contract `pub const NAMES: &[&str]` and
  `pub fn values(&Evidence) -> Vec<f64>` of matching length. `Evidence`
  (`features.rs:269`) is the per-PSM input struct handed to every family.
- `extended_names()` (`features.rs:93`) deduplicates the family names: it drops
  names colliding with the reserved Minimal/Rich set (`reserved_names`,
  `features.rs:67`) and cross-family repeats (keep-first). The four dropped
  collisions above are removed here; their computed values are discarded.
- `active_features(set)` (`features.rs:201`) assembles the ordered active column
  list for the configured tier.
- `feature_schema_id(cols)` (`features.rs:220`) is the hashed feature-schema
  mechanism: blake3 of the comma-joined ordered active column names. Any change
  to the set (add/drop/reorder) changes the id.
- `FeatureSchema` (`features.rs:226`) is the companion record
  `{feature_columns, schema_id}`, written to `<artifact>.schema.json`
  (`features.rs:755`) and read back in compete and rescore
  (`FeatureSchema::read`, `features.rs:233`) so the classifier never runs under a
  mismatched feature set.

`level` values: `fragment` (per-fragment spectral/co-elution kernels),
`peak` (peak-bounded chromatographic quantities), `precursor` (MS1 / charge),
`candidate` (identity/metadata), `run` (run-context). `direction` is the expected
monotone sign toward a correct target: `higher_better`, `lower_better`,
`neutral`, or `?` when undetermined.

## 3. Feature tables by family

Grouped by family in schema (first-appearance) order. Notes are abbreviated from
the inventory; source file is the basename under `stages/features/` (or
`features.rs` for the Minimal/Rich/extra rows). Rows marked "DROPPED from schema"
are computed but removed by the dedup at `features.rs:93` because they collide
with a reserved name.

### minimal (14)

Minimal tier base features (RT error, matched fragments, co-elution, apex intensity, library agreement, metadata). Always active.

| name | level | direction | source file | note |
|---|---|---|---|---|
| `rt_error_abs` | candidate | lower_better | `features.rs` | \|apex_rt - rt_pred_cal\|; pushed at line 663 |
| `rt_error_rel` | candidate | lower_better | `features.rs` | rt_err/gradient; line 664 |
| `n_matched_fragments` | candidate | higher_better | `features.rs` | line 665 |
| `coelution_run` | peak | higher_better | `features.rs` | from psms; line 666 |
| `log_apex_intensity` | peak | higher_better | `features.rs` | line 667 |
| `frag_corr` | fragment | higher_better | `features.rs` | pearson(obs_apex,pred); CONTESTED/interference-relevant; line 668 |
| `frag_cosine` | fragment | higher_better | `features.rs` | line 669 |
| `spectral_angle` | fragment | higher_better | `features.rs` | normalized similarity in [0,1]; line 670; RESERVED name (shadows similarity::spectral_a... |
| `coelution_mean` | fragment | higher_better | `features.rs` | mean pairwise trace pearson; line 671 |
| `coelution_best` | fragment | higher_better | `features.rs` | line 672 |
| `n_coelution_above` | fragment | higher_better | `features.rs` | count pairwise corr>=coelution_corr_threshold; line 673 |
| `charge` | precursor | neutral | `features.rs` | line 674 |
| `peptide_length` | candidate | neutral | `features.rs` | line 675; RESERVED (shadows novel::peptide_length) |
| `n_proteins` | candidate | lower_better | `features.rs` | protein-group multiplicity; line 676 |

### rich (30)

Rich tier additions (library agreement, xcorr, ion-series sums, mass error, MS1 isotope, S/N, seed corroboration, DIA-NN profile/interference proxies).

| name | level | direction | source file | note |
|---|---|---|---|---|
| `library_norm_manhattan` | fragment | lower_better | `features.rs` | line 677 |
| `library_rmsd` | fragment | lower_better | `features.rs` | line 678 |
| `xcorr_coelution` | fragment | lower_better | `features.rs` | mean abs xcorr lag; line 679 |
| `xcorr_shape` | fragment | higher_better | `features.rs` | line 680 |
| `sum_b_intensity` | fragment | higher_better | `features.rs` | line 681 |
| `sum_y_intensity` | fragment | higher_better | `features.rs` | line 682 |
| `diff_by_intensity` | fragment | neutral | `features.rs` | sum_b-sum_y; line 683 |
| `n_b_ions` | fragment | higher_better | `features.rs` | line 684 |
| `n_y_ions` | fragment | higher_better | `features.rs` | line 685 |
| `weighted_mass_error` | fragment | lower_better | `features.rs` | intensity-weighted mean \|ppm\|; line 686 |
| `mean_mass_error` | fragment | lower_better | `features.rs` | line 687 |
| `isotope_corr` | precursor | higher_better | `features.rs` | MS1 averagine corr; line 688 |
| `ms1_isom1_ratio` | precursor | lower_better | `features.rs` | iso -1/mono contamination; line 689 |
| `log_mono_ms1` | precursor | higher_better | `features.rs` | line 690 |
| `has_ms1` | precursor | higher_better | `features.rs` | line 691 |
| `log_sn` | peak | higher_better | `features.rs` | apex vs median trace; line 692 |
| `n_observations` | peak | neutral | `features.rs` | scans in peak; line 693 |
| `base_width_rt` | peak | neutral | `features.rs` | line 694 |
| `seed_score` | candidate | higher_better | `features.rs` | from seed_psms map; line 695 |
| `seed_identified` | candidate | higher_better | `features.rs` | seed spectrum_q<=0.01 flag; line 696; RESERVED (shadows novel::seed_identified) |
| `matched_fraction` | fragment | higher_better | `features.rs` | n_matched/n_pred; line 700 |
| `profile_cos` | fragment | higher_better | `features.rs` | DIA-NN pCos; line 702 |
| `ref_corr` | fragment | higher_better | `features.rs` | DIA-NN pTimeCorr; line 703 |
| `best_ref_corr` | fragment | higher_better | `features.rs` | line 704 |
| `low_frag_coel` | fragment | higher_better | `features.rs` | pResCorr proxy; line 705 |
| `evidence` | fragment | higher_better | `features.rs` | summed frag-vs-ref corr (DIA-NN Evidence); CONTESTED-relevant; line 706 |
| `contrast_min` | fragment | higher_better | `features.rs` | INTERFERENCE: min frag-vs-others corr (low=interfered); line 707 |
| `resid_corr` | fragment | lower_better | `features.rs` | INTERFERENCE: mean residual pairwise corr (high=shared interferent); line 708 |
| `coel_clean` | fragment | higher_better | `features.rs` | INTERFERENCE: co-elution after 1.5*r*ref capping; line 709 |
| `shadow_frac` | fragment | lower_better | `features.rs` | INTERFERENCE: fraction of intensity above cap; line 710 |

### extended-extra (psms-derived) (1)

Extended extra carried from the extract psms table (contested fraction), not an Evidence family.

| name | level | direction | source file | note |
|---|---|---|---|---|
| `peak_contested_frac` | candidate | lower_better | `features.rs` | EXISTING CONTESTED FEATURE: reads extract `contested_frac` column (line 509), pushed li... |

### extended-extra (cross-candidate) (3)

Extended extra aggregated across candidates of one peptidoform (cross-charge corroboration); the only cross-candidate path in the feature stage.

| name | level | direction | source file | note |
|---|---|---|---|---|
| `n_charge_states` | precursor | higher_better | `features.rs` | cross-candidate aggregate over peptidoform; line 579/729 |
| `charge_multi_flag` | precursor | higher_better | `features.rs` | >=2 charge states; line 580/730 |
| `cross_charge_intensity_log` | precursor | higher_better | `features.rs` | ln(1+summed apex of other charge states); unbounded; line 583/731 |

### similarity (64)

FAMILIES[0] observed-vs-library spectral agreement kernels (cosine/pearson/spearman/kendall/distances/presence/area/unbounded evidence).

| name | level | direction | source file | note |
|---|---|---|---|---|
| `spectrum_cosine_matched` | fragment | higher_better | `similarity.rs` |  |
| `spectrum_cosine_sqrt` | fragment | higher_better | `similarity.rs` |  |
| `spectrum_cosine_log` | fragment | higher_better | `similarity.rs` |  |
| `spectral_angle` | fragment | higher_better | `similarity.rs` | DROPPED from schema: collides with reserved minimal::spectral_angle |
| `spectral_angle_sqrt` | fragment | higher_better | `similarity.rs` |  |
| `spectral_angle_matched` | fragment | higher_better | `similarity.rs` |  |
| `pearson_intensity_matched` | fragment | higher_better | `similarity.rs` |  |
| `pearson_intensity_log` | fragment | higher_better | `similarity.rs` |  |
| `spearman_intensity` | fragment | higher_better | `similarity.rs` |  |
| `spearman_intensity_matched` | fragment | higher_better | `similarity.rs` |  |
| `kendall_tau_intensity` | fragment | higher_better | `similarity.rs` |  |
| `dot_product_raw` | fragment | higher_better | `similarity.rs` | unbounded |
| `dot_product_norm` | fragment | higher_better | `similarity.rs` |  |
| `library_recall_intensity` | fragment | higher_better | `similarity.rs` | predicted intensity fraction observed |
| `manhattan_sim` | fragment | higher_better | `similarity.rs` | 1-0.5*L1 |
| `manhattan_sqrt` | fragment | lower_better | `similarity.rs` | raw L1 of sqrt-renormalized |
| `rmsd_norm` | fragment | lower_better | `similarity.rs` |  |
| `mae_norm` | fragment | lower_better | `similarity.rs` |  |
| `mse_log` | fragment | lower_better | `similarity.rs` |  |
| `mae_weighted_pred` | fragment | lower_better | `similarity.rs` |  |
| `abs_diff_q3` | fragment | lower_better | `similarity.rs` |  |
| `max_positive_residual` | fragment | lower_better | `similarity.rs` |  |
| `chebyshev_dist` | fragment | lower_better | `similarity.rs` |  |
| `minkowski_p3` | fragment | lower_better | `similarity.rs` |  |
| `bray_curtis` | fragment | higher_better | `similarity.rs` | Ruzicka min/max |
| `bray_curtis_sqrt` | fragment | higher_better | `similarity.rs` |  |
| `canberra` | fragment | lower_better | `similarity.rs` |  |
| `canberra_matched` | fragment | lower_better | `similarity.rs` |  |
| `wave_hedges` | fragment | lower_better | `similarity.rs` |  |
| `chi_square_pearson` | fragment | lower_better | `similarity.rs` |  |
| `chi_square_symmetric` | fragment | lower_better | `similarity.rs` |  |
| `divergence_distance` | fragment | lower_better | `similarity.rs` |  |
| `bhattacharyya_coef` | fragment | higher_better | `similarity.rs` |  |
| `hellinger` | fragment | lower_better | `similarity.rs` |  |
| `squared_chord` | fragment | lower_better | `similarity.rs` |  |
| `harmonic_mean_sim` | fragment | higher_better | `similarity.rs` |  |
| `jaccard_presence` | fragment | higher_better | `similarity.rs` | predicted vs observed>1%max presence |
| `dice_presence` | fragment | higher_better | `similarity.rs` |  |
| `intensity_weighted_pearson` | fragment | higher_better | `similarity.rs` |  |
| `regression_slope` | fragment | neutral | `similarity.rs` | cov(l,o)/var(l), ~1 ideal |
| `gini_diff` | fragment | neutral | `similarity.rs` | Gini(obs_matched)-Gini(lib) |
| `wasserstein_mz` | fragment | lower_better | `similarity.rs` | EMD over m/z-ordered CDFs |
| `footrule_norm` | fragment | higher_better | `similarity.rs` |  |
| `rank_overlap_top3` | fragment | higher_better | `similarity.rs` |  |
| `top1_frag_match` | fragment | higher_better | `similarity.rs` | argmax(l)==argmax(o) flag |
| `top1_predicted_observed` | fragment | higher_better | `similarity.rs` |  |
| `frac_top3_predicted_observed` | fragment | higher_better | `similarity.rs` |  |
| `count_strong_predicted_absent` | fragment | lower_better | `similarity.rs` |  |
| `frac_predicted_absent` | fragment | lower_better | `similarity.rs` | (n_pred-n_matched)/n_pred |
| `cosine_area` | peak | higher_better | `similarity.rs` | peak-XIC trapezoid area vs lib |
| `pearson_area` | peak | higher_better | `similarity.rs` |  |
| `spectral_angle_area` | peak | higher_better | `similarity.rs` |  |
| `cosine_fullwindow` | peak | higher_better | `similarity.rs` | full-window area vs lib |
| `stein_scott_weighted_dot` | fragment | higher_better | `similarity.rs` | mz^3 * intensity^0.6 weighted cosine |
| `log_dot_product` | fragment | higher_better | `similarity.rs` | UNBOUNDED evidence (anti-saturation) |
| `spectral_log_evidence` | fragment | higher_better | `similarity.rs` | UNBOUNDED DIA-NN Evidence analog |
| `scribe_score` | fragment | higher_better | `similarity.rs` | EncyclopeDIA Scribe, UNBOUNDED |
| `log_dot_product_area` | peak | higher_better | `similarity.rs` | area twin |
| `spectral_log_evidence_area` | peak | higher_better | `similarity.rs` |  |
| `scribe_score_area` | peak | higher_better | `similarity.rs` |  |
| `cosine_high_ordinal` | fragment | higher_better | `similarity.rs` | ordinal>=3 only (drop co-isolation-prone b1/b2/y1/y2) |
| `cosine_robust_trim1` | fragment | higher_better | `similarity.rs` | cosine after dropping worst-residual fragment; interference trajectory |
| `cosine_robust_trim2` | fragment | higher_better | `similarity.rs` |  |
| `cosine_robust_trim3` | fragment | higher_better | `similarity.rs` |  |

### entropy (18)

FAMILIES[1] spectral-entropy / divergence features (Li entropy sim, JSD/KL/cross-entropy, obs/pred entropy).

| name | level | direction | source file | note |
|---|---|---|---|---|
| `spectral_entropy_similarity` | fragment | higher_better | `entropy.rs` | Li entropy sim |
| `weighted_spectral_entropy_similarity` | fragment | higher_better | `entropy.rs` |  |
| `spectral_entropy_similarity_sqrt` | fragment | higher_better | `entropy.rs` |  |
| `spectral_entropy_similarity_topk` | fragment | higher_better | `entropy.rs` | top-6 by predicted |
| `spectral_entropy_similarity_area` | peak | higher_better | `entropy.rs` |  |
| `jensen_shannon_divergence` | fragment | lower_better | `entropy.rs` |  |
| `jeffreys_divergence` | fragment | lower_better | `entropy.rs` | symmetric KL |
| `kl_obs_pred` | fragment | lower_better | `entropy.rs` |  |
| `kl_pred_obs` | fragment | lower_better | `entropy.rs` |  |
| `cross_entropy_obs_pred` | fragment | lower_better | `entropy.rs` |  |
| `obs_spectrum_entropy` | fragment | neutral | `entropy.rs` |  |
| `pred_spectrum_entropy` | fragment | neutral | `entropy.rs` |  |
| `entropy_diff` | fragment | neutral | `entropy.rs` |  |
| `entropy_ratio` | fragment | neutral | `entropy.rs` |  |
| `obs_normalized_entropy` | fragment | neutral | `entropy.rs` | Pielou evenness |
| `normalized_entropy_diff` | fragment | neutral | `entropy.rs` |  |
| `residual_spectrum_entropy` | fragment | lower_better | `entropy.rs` |  |
| `entropy_weight_obs` | fragment | neutral | `entropy.rs` | Li exponent |

### coelution (38)

FAMILIES[2] fragment-vs-reference and pairwise co-elution + cross-correlation-lag stats + b/y and charge cross co-elution.

| name | level | direction | source file | note |
|---|---|---|---|---|
| `frag_ref_corr_mean` | fragment | higher_better | `coelution.rs` |  |
| `frag_ref_corr_obsweighted` | fragment | higher_better | `coelution.rs` |  |
| `frag_ref_corr_min` | fragment | higher_better | `coelution.rs` |  |
| `frag_ref_corr_std` | fragment | lower_better | `coelution.rs` |  |
| `frag_ref_corr_sq_mean` | fragment | higher_better | `coelution.rs` |  |
| `frag_ref_corr_topk_weighted` | fragment | higher_better | `coelution.rs` |  |
| `n_frag_ref_corr_above_0_9` | fragment | higher_better | `coelution.rs` |  |
| `frac_frag_ref_corr_above_0_8` | fragment | higher_better | `coelution.rs` |  |
| `frag_ref_corr_mean_full` | fragment | higher_better | `coelution.rs` | full window |
| `full_vs_peak_corr_gain` | fragment | neutral | `coelution.rs` |  |
| `pairwise_coelution_weighted` | fragment | higher_better | `coelution.rs` | == coelution_weighted_mean alias (emitted once) |
| `pairwise_coelution_min` | fragment | higher_better | `coelution.rs` |  |
| `pairwise_coelution_median` | fragment | higher_better | `coelution.rs` |  |
| `pairwise_coelution_std` | fragment | lower_better | `coelution.rs` |  |
| `pairwise_coelution_frac_negative` | fragment | lower_better | `coelution.rs` | INTERFERENCE indicator |
| `pairwise_coelution_hi` | fragment | higher_better | `coelution.rs` | top-half-pred fragments |
| `pairwise_coelution_lo` | fragment | higher_better | `coelution.rs` |  |
| `coelution_hi_lo_contrast` | fragment | neutral | `coelution.rs` |  |
| `coelution_corr_entropy` | fragment | lower_better | `coelution.rs` |  |
| `xcorr_shape_mean` | fragment | higher_better | `coelution.rs` |  |
| `xcorr_shape_min` | fragment | higher_better | `coelution.rs` |  |
| `xcorr_shape_std` | fragment | lower_better | `coelution.rs` |  |
| `xcorr_lag_mean_abs` | fragment | lower_better | `coelution.rs` |  |
| `xcorr_lag_std` | fragment | lower_better | `coelution.rs` |  |
| `xcorr_lag_iqr` | fragment | lower_better | `coelution.rs` |  |
| `xcorr_lag_frac_zero` | fragment | higher_better | `coelution.rs` |  |
| `xcorr_lag_max_abs` | fragment | lower_better | `coelution.rs` |  |
| `xcorr_lag_entropy` | fragment | lower_better | `coelution.rs` |  |
| `ref_xcorr_lag_mean` | fragment | lower_better | `coelution.rs` | each frag vs reference |
| `ref_xcorr_shape_mean` | fragment | higher_better | `coelution.rs` |  |
| `observed_sum_vs_template_corr` | fragment | higher_better | `coelution.rs` |  |
| `frag_loo_ref_corr_mean` | fragment | higher_better | `coelution.rs` | leave-one-out reference; INTERFERENCE-relevant |
| `frag_loo_ref_corr_min` | fragment | higher_better | `coelution.rs` |  |
| `frac_frags_apex_aligned` | peak | higher_better | `coelution.rs` | apex-dispersion-relevant |
| `top3_frag_ref_corr` | fragment | higher_better | `coelution.rs` |  |
| `by_cross_coelution` | fragment | higher_better | `coelution.rs` | b-vs-y cross pairs |
| `by_cross_lag_mean` | fragment | lower_better | `coelution.rs` |  |
| `charge_cross_coelution` | fragment | higher_better | `coelution.rs` | charge-1 vs charge>=2 pairs |

### interference (26)

FAMILIES[3] contested/interference: explained variance, residual fraction, iterative prune, area ratios, apex purity, PCA rank, competing-peak counts.

| name | level | direction | source file | note |
|---|---|---|---|---|
| `explained_variance_ref` | fragment | higher_better | `interference.rs` | CONTESTED family |
| `profile_residual_fraction` | fragment | lower_better | `interference.rs` |  |
| `n_interfered_fragments` | fragment | lower_better | `interference.rs` |  |
| `corrected_vs_raw_cos` | fragment | neutral | `interference.rs` |  |
| `corrected_vs_raw_ratio` | fragment | higher_better | `interference.rs` |  |
| `ifs_removed_count` | fragment | lower_better | `interference.rs` | remove_ifs iterative prune |
| `ifs_removed_intensity_frac` | fragment | lower_better | `interference.rs` |  |
| `ifs_corr_gain` | fragment | neutral | `interference.rs` | large gain = was interfered |
| `ifs_retained_frac` | fragment | higher_better | `interference.rs` |  |
| `matched_frac_after_ifs` | fragment | higher_better | `interference.rs` |  |
| `peak_to_full_area_ratio_profile` | peak | higher_better | `interference.rs` |  |
| `peak_to_full_area_ratio_frag_mean` | peak | higher_better | `interference.rs` |  |
| `peak_to_full_area_ratio_weighted` | peak | higher_better | `interference.rs` |  |
| `out_of_peak_intensity_frac` | peak | lower_better | `interference.rs` |  |
| `profile_corr_full_vs_peak_delta` | fragment | neutral | `interference.rs` |  |
| `frac_frag_ref_corr_below_0_5` | fragment | lower_better | `interference.rs` |  |
| `explained_apex_intensity_frac` | peak | higher_better | `interference.rs` |  |
| `apex_purity` | peak | higher_better | `interference.rs` | CONTESTED: fraction of apex intensity from coherent fragments |
| `interference_apex_residual_fraction` | peak | lower_better | `interference.rs` |  |
| `dominant_frag_ref_corr` | fragment | higher_better | `interference.rs` |  |
| `explained_variance_ratio` | peak | higher_better | `interference.rs` | top eigenvalue / trace of Gram (power_top) |
| `second_component_fraction` | peak | lower_better | `interference.rs` | CONTESTED: second eigenvalue fraction = 2-component (chimera) evidence |
| `profile_second_peak_ratio` | peak | lower_better | `interference.rs` |  |
| `n_competing_peaks_in_window` | peak | lower_better | `interference.rs` | CONTESTED: competing chromatographic peaks in window |
| `matched_pred_intensity_fraction` | fragment | higher_better | `interference.rs` |  |
| `top_pred_frag_matched` | fragment | higher_better | `interference.rs` |  |

### chromatographic (43)

FAMILIES[4] peak shape: Gaussian/EMG fits, FWHM/asymmetry/tailing, roughness/zigzag, RT moments, apex dispersion, per-fragment gaussianity.

| name | level | direction | source file | note |
|---|---|---|---|---|
| `gaussian_fit_r2` | peak | higher_better | `chromatographic.rs` |  |
| `gaussian_cosine` | peak | higher_better | `chromatographic.rs` |  |
| `emg_fit_improvement` | peak | neutral | `chromatographic.rs` | tailing indicator |
| `apex_prominence` | peak | higher_better | `chromatographic.rs` |  |
| `profile_peak_snr` | peak | higher_better | `chromatographic.rs` |  |
| `fwhm_seconds` | peak | neutral | `chromatographic.rs` |  |
| `fwhm_to_window_ratio` | peak | neutral | `chromatographic.rs` |  |
| `width_at_10pct` | peak | neutral | `chromatographic.rs` |  |
| `width_ratio_10_50` | peak | neutral | `chromatographic.rs` |  |
| `hwhm_asymmetry` | peak | lower_better | `chromatographic.rs` | abs toward 0 |
| `tailing_factor_usp` | peak | neutral | `chromatographic.rs` | ~1 ideal |
| `asymmetry_factor_10pct` | peak | neutral | `chromatographic.rs` | ~1 ideal |
| `apex_sharpness` | peak | higher_better | `chromatographic.rs` |  |
| `apex_curvature` | peak | higher_better | `chromatographic.rs` |  |
| `apex_to_boundary_ratio` | peak | higher_better | `chromatographic.rs` |  |
| `apex_dominance` | peak | higher_better | `chromatographic.rs` |  |
| `zigzag_index` | peak | lower_better | `chromatographic.rs` |  |
| `jaggedness` | peak | lower_better | `chromatographic.rs` |  |
| `roughness_2nd_deriv` | peak | lower_better | `chromatographic.rs` |  |
| `n_local_maxima` | peak | lower_better | `chromatographic.rs` |  |
| `modality` | peak | lower_better | `chromatographic.rs` | multimodality valley depth |
| `rt_skewness` | peak | neutral | `chromatographic.rs` |  |
| `rt_excess_kurtosis` | peak | neutral | `chromatographic.rs` |  |
| `rt_std_seconds` | peak | neutral | `chromatographic.rs` |  |
| `mean_mode_offset` | peak | lower_better | `chromatographic.rs` |  |
| `fraction_area_within_fwhm` | peak | higher_better | `chromatographic.rs` |  |
| `triangle_area_similarity` | peak | lower_better | `chromatographic.rs` |  |
| `baseline_fraction` | peak | neutral | `chromatographic.rs` |  |
| `peak_completeness` | peak | higher_better | `chromatographic.rs` | window-edge/degeneracy indicator |
| `apex_centering_offset` | peak | lower_better | `chromatographic.rs` | apex distance from window center |
| `intensity_score` | peak | higher_better | `chromatographic.rs` |  |
| `total_xic_log` | peak | higher_better | `chromatographic.rs` |  |
| `frag_fwhm_cv` | fragment | lower_better | `chromatographic.rs` |  |
| `frag_fwhm_mean` | fragment | neutral | `chromatographic.rs` |  |
| `frag_apex_rt_dispersion` | peak | lower_better | `chromatographic.rs` | APEX DISPERSION (existing); std of per-fragment apex RTs |
| `frag_apex_rt_dispersion_weighted` | peak | lower_better | `chromatographic.rs` | APEX DISPERSION (existing), pred-weighted |
| `frag_apex_offset_from_profile_mean` | peak | lower_better | `chromatographic.rs` | APEX DISPERSION (existing) |
| `frag_gaussianity_mean` | fragment | higher_better | `chromatographic.rs` |  |
| `frag_gaussianity_weighted` | fragment | higher_better | `chromatographic.rs` |  |
| `frag_zigzag_mean` | fragment | lower_better | `chromatographic.rs` |  |
| `sumtrace_unweighted_gaussian_r2` | peak | higher_better | `chromatographic.rs` |  |
| `reference_profile_rt_entropy_peak` | peak | lower_better | `chromatographic.rs` |  |
| `reference_profile_rt_entropy_ratio` | peak | neutral | `chromatographic.rs` |  |

### mass_accuracy (17)

FAMILIES[5] fragment ppm-error distribution + positive mass-evidence.

| name | level | direction | source file | note |
|---|---|---|---|---|
| `median_abs_frag_ppm` | fragment | lower_better | `mass_accuracy.rs` |  |
| `signed_mean_frag_ppm` | fragment | neutral | `mass_accuracy.rs` | ~0 ideal |
| `ppm_std` | fragment | lower_better | `mass_accuracy.rs` |  |
| `ppm_iqr` | fragment | lower_better | `mass_accuracy.rs` |  |
| `ppm_range` | fragment | lower_better | `mass_accuracy.rs` |  |
| `max_abs_frag_ppm` | fragment | lower_better | `mass_accuracy.rs` |  |
| `intensity_weighted_abs_ppm` | fragment | lower_better | `mass_accuracy.rs` |  |
| `intensity_weighted_signed_ppm` | fragment | neutral | `mass_accuracy.rs` |  |
| `intensity_weighted_ppm_std` | fragment | lower_better | `mass_accuracy.rs` |  |
| `lib_weighted_abs_ppm` | fragment | lower_better | `mass_accuracy.rs` |  |
| `frac_frag_within_half_tol` | fragment | higher_better | `mass_accuracy.rs` | uses hardcoded HALF_TOL_PPM=10 |
| `high_ppm_intensity_frac` | fragment | lower_better | `mass_accuracy.rs` |  |
| `ppm_intensity_anticorr` | fragment | lower_better | `mass_accuracy.rs` | pearson(\|ppm\|,intensity); real trends negative |
| `mass_error_mz_trend` | fragment | lower_better | `mass_accuracy.rs` |  |
| `mean_abs_mz_error_da` | fragment | lower_better | `mass_accuracy.rs` |  |
| `mass_evidence_gauss` | fragment | higher_better | `mass_accuracy.rs` | positive evidence; SIGMA_PPM=10 hardcoded |
| `mass_log_evidence` | fragment | higher_better | `mass_accuracy.rs` | UNBOUNDED DIA-NN Mass.Evidence analog |

### ion_series (34)

FAMILIES[6] b/y series coverage, ladder runs, complementarity, per-series similarity/co-elution, charge-resolved cosine.

| name | level | direction | source file | note |
|---|---|---|---|---|
| `n_matched_b` | fragment | higher_better | `ion_series.rs` |  |
| `n_matched_y` | fragment | higher_better | `ion_series.rs` |  |
| `frac_matched_b` | fragment | higher_better | `ion_series.rs` |  |
| `frac_matched_y` | fragment | higher_better | `ion_series.rs` |  |
| `by_count_balance` | fragment | higher_better | `ion_series.rs` |  |
| `by_intensity_ratio` | fragment | neutral | `ion_series.rs` |  |
| `by_ratio_agreement` | fragment | lower_better | `ion_series.rs` | tanh log-odds discrepancy |
| `by_ratio_consistency` | fragment | lower_better | `ion_series.rs` |  |
| `longest_b_run` | fragment | higher_better | `ion_series.rs` |  |
| `longest_y_run` | fragment | higher_better | `ion_series.rs` |  |
| `longest_run_max` | fragment | higher_better | `ion_series.rs` |  |
| `longest_run_frac_length` | fragment | higher_better | `ion_series.rs` |  |
| `series_coverage_b` | fragment | higher_better | `ion_series.rs` |  |
| `series_coverage_y` | fragment | higher_better | `ion_series.rs` |  |
| `sequence_coverage` | candidate | higher_better | `ion_series.rs` |  |
| `series_gap_fraction` | fragment | lower_better | `ion_series.rs` |  |
| `by_complement_count` | fragment | higher_better | `ion_series.rs` |  |
| `by_complement_mz_consistency` | fragment | lower_better | `ion_series.rs` | ppm dev of b+y from M+2H |
| `by_complement_coelution` | fragment | higher_better | `ion_series.rs` |  |
| `ordinal_intensity_concordance_y` | fragment | higher_better | `ion_series.rs` |  |
| `ordinal_intensity_concordance_b` | fragment | higher_better | `ion_series.rs` |  |
| `series_coelution_y` | fragment | higher_better | `ion_series.rs` |  |
| `series_coelution_b` | fragment | higher_better | `ion_series.rs` |  |
| `spectral_angle_b` | fragment | higher_better | `ion_series.rs` |  |
| `spectral_angle_y` | fragment | higher_better | `ion_series.rs` |  |
| `pearson_b` | fragment | higher_better | `ion_series.rs` |  |
| `pearson_y` | fragment | higher_better | `ion_series.rs` |  |
| `cosine_charge1` | fragment | higher_better | `ion_series.rs` |  |
| `cosine_charge2` | fragment | higher_better | `ion_series.rs` |  |
| `charge_corr_balance` | fragment | higher_better | `ion_series.rs` |  |
| `mean_matched_ordinal_norm` | fragment | neutral | `ion_series.rs` |  |
| `by_ion_contiguous_intensity` | fragment | higher_better | `ion_series.rs` |  |
| `by_ion_contiguous_lib_frac` | fragment | higher_better | `ion_series.rs` |  |
| `both_series_present` | fragment | higher_better | `ion_series.rs` |  |

### ms1 (25)

FAMILIES[7] MS1 isotope-envelope agreement + MS1/MS2 XIC co-elution/shape (10 XIC features are 0.0 until extract persists ms1_xic).

| name | level | direction | source file | note |
|---|---|---|---|---|
| `ms1_isotope_cosine_apex` | precursor | higher_better | `ms1.rs` |  |
| `ms1_isotope_spectral_angle_apex` | precursor | higher_better | `ms1.rs` |  |
| `ms1_isotope_chi2_apex` | precursor | lower_better | `ms1.rs` |  |
| `ms1_isotope_manhattan_apex` | precursor | higher_better | `ms1.rs` |  |
| `iso_ratio_1_0` | precursor | neutral | `ms1.rs` |  |
| `iso_ratio_2_0` | precursor | neutral | `ms1.rs` |  |
| `iso_plus1_ratio_dev` | precursor | lower_better | `ms1.rs` |  |
| `iso_plus2_ratio_dev` | precursor | lower_better | `ms1.rs` |  |
| `iso_minus_one_fraction` | precursor | lower_better | `ms1.rs` | co-isolation contamination |
| `iso_overlap_flag` | precursor | lower_better | `ms1.rs` |  |
| `log_ms1_mono` | precursor | higher_better | `ms1.rs` |  |
| `ms1_total_isotope_log` | precursor | higher_better | `ms1.rs` |  |
| `has_ms1_signal` | precursor | higher_better | `ms1.rs` |  |
| `ms1_isotope_apex_entropy_3` | precursor | neutral | `ms1.rs` |  |
| `ms1_m1_entropy_contribution` | precursor | neutral | `ms1.rs` |  |
| `ms1_ms2_time_corr` | precursor | higher_better | `ms1.rs` | 0.0 until ms1_xic persisted by extract |
| `ms1_ms2_envelope_time_corr` | precursor | higher_better | `ms1.rs` | 0.0 until ms1_xic persisted |
| `ms1_iso_coelution` | precursor | higher_better | `ms1.rs` | 0.0 until ms1_xic persisted |
| `ms1_ms2_apex_rt_delta` | precursor | lower_better | `ms1.rs` | 0.0 until ms1_xic persisted; was the mis-scaled/unbounded feature, now bounded d/(d+width) |
| `ms1_iso_ratio_stability` | precursor | lower_better | `ms1.rs` | 0.0 until ms1_xic persisted |
| `ms1_mono_gaussianity` | precursor | higher_better | `ms1.rs` | 0.0 until ms1_xic persisted |
| `ms1_ms2_fwhm_ratio` | precursor | neutral | `ms1.rs` | ~1 ideal; 0.0 until ms1_xic persisted |
| `ms1_isotope_corr_xic` | precursor | higher_better | `ms1.rs` | 0.0 until ms1_xic persisted |
| `ms1_envelope_over_time_corr` | precursor | higher_better | `ms1.rs` | 0.0 until ms1_xic persisted |
| `ms1_isotope_xic_shape_consistency` | precursor | higher_better | `ms1.rs` | 0.0 until ms1_xic persisted |

### rt (13)

FAMILIES[8] RT-agreement (signed/abs/squared, gradient- and peak-width-normalized, profile-apex delta).

| name | level | direction | source file | note |
|---|---|---|---|---|
| `rt_error_signed` | candidate | neutral | `rt.rs` | ~0 ideal |
| `rt_error_abs` | candidate | lower_better | `rt.rs` | DROPPED from schema: collides with reserved minimal::rt_error_abs |
| `rt_error_squared` | candidate | lower_better | `rt.rs` |  |
| `rt_error_signed_norm_gradient` | run | neutral | `rt.rs` |  |
| `rt_error_abs_norm_gradient` | run | lower_better | `rt.rs` |  |
| `observed_rt_raw` | candidate | neutral | `rt.rs` |  |
| `predicted_rt_raw` | candidate | neutral | `rt.rs` |  |
| `observed_rt_fraction` | run | neutral | `rt.rs` |  |
| `predicted_rt_fraction` | run | neutral | `rt.rs` |  |
| `rt_error_over_peak_width` | peak | lower_better | `rt.rs` |  |
| `rt_error_over_fwhm` | peak | lower_better | `rt.rs` |  |
| `rt_diff_profile_apex` | peak | lower_better | `rt.rs` |  |
| `predicted_rt_in_gradient` | run | higher_better | `rt.rs` | flag |

### novel (12)

FAMILIES[9] seed corroboration + precursor/charge metadata + count/intensity summaries.

| name | level | direction | source file | note |
|---|---|---|---|---|
| `log_seed_hyperscore` | candidate | higher_better | `novel.rs` |  |
| `seed_hyperscore_per_matched` | candidate | higher_better | `novel.rs` |  |
| `seed_identified` | candidate | higher_better | `novel.rs` | DROPPED from schema: collides with reserved rich::seed_identified |
| `peptide_length` | candidate | neutral | `novel.rs` | DROPPED from schema: collides with reserved minimal::peptide_length |
| `precursor_charge` | precursor | neutral | `novel.rs` |  |
| `charge_is_2` | precursor | neutral | `novel.rs` | flag |
| `charge_is_3` | precursor | neutral | `novel.rs` | flag |
| `charge_is_4plus` | precursor | neutral | `novel.rs` | flag |
| `precursor_mass` | precursor | neutral | `novel.rs` |  |
| `log_total_matched_intensity` | fragment | higher_better | `novel.rs` |  |
| `n_matched_frags` | fragment | higher_better | `novel.rs` |  |
| `n_predicted_frags` | fragment | neutral | `novel.rs` |  |

### nonzero (12)

FAMILIES[10] zero-ignoring variants (per-fragment peak-max spectral agreement, both-positive co-elution) to fix apex-scan-alignment zeros.

| name | level | direction | source file | note |
|---|---|---|---|---|
| `frag_corr_peakmax` | fragment | higher_better | `nonzero.rs` | peak-max obs (fixes apex-scan-alignment zeros) |
| `frag_cosine_peakmax` | fragment | higher_better | `nonzero.rs` |  |
| `spectral_angle_peakmax` | fragment | higher_better | `nonzero.rs` |  |
| `frag_corr_matched_nz` | fragment | higher_better | `nonzero.rs` |  |
| `frag_cosine_matched_nz` | fragment | higher_better | `nonzero.rs` |  |
| `peakmax_apex_gain` | fragment | lower_better | `nonzero.rs` | how much apex scan understates peak; high=off-apex/chimera |
| `n_frag_present_inpeak` | fragment | higher_better | `nonzero.rs` |  |
| `frac_frag_present_inpeak` | fragment | higher_better | `nonzero.rs` |  |
| `coelution_mean_bothpos` | fragment | higher_better | `nonzero.rs` |  |
| `coelution_mean_summpos` | fragment | higher_better | `nonzero.rs` |  |
| `ref_corr_nz` | fragment | higher_better | `nonzero.rs` |  |
| `profile_cos_nz` | fragment | higher_better | `nonzero.rs` |  |

### order_consistency (8)

FAMILIES[11] prediction-free MS2-XIC rank-stability (per-scan Spearman/Kendall vs apex, top1/2 persistence, argmax entropy).

| name | level | direction | source file | note |
|---|---|---|---|---|
| `rank_corr_vs_apex_mean` | peak | higher_better | `order_consistency.rs` | prediction-free MS2-XIC rank stability |
| `rank_corr_vs_apex_std` | peak | lower_better | `order_consistency.rs` |  |
| `rank_corr_adjacent_mean` | peak | higher_better | `order_consistency.rs` |  |
| `kendall_vs_apex_mean` | peak | higher_better | `order_consistency.rs` |  |
| `top1_frag_persistence` | peak | higher_better | `order_consistency.rs` |  |
| `top2_order_persistence` | peak | higher_better | `order_consistency.rs` |  |
| `argmax_frag_entropy` | peak | lower_better | `order_consistency.rs` | 0=one fragment dominates (clean), 1=noisy |
| `self_cosine_vs_apex_mean` | peak | higher_better | `order_consistency.rs` |  |

### peak_scans (2)

FAMILIES[12] label-blind window-degeneracy indicators (n_peak_scans, peak_window_degenerate); the observability-model reference family.

| name | level | direction | source file | note |
|---|---|---|---|---|
| `n_peak_scans` | peak | higher_better | `peak_scans.rs` | OBSERVABILITY: label-blind measured-scan count; disambiguates undefined vs genuine zero |
| `peak_window_degenerate` | peak | lower_better | `peak_scans.rs` | OBSERVABILITY: flag when <3 non-empty scans (window-based families collapse to 0) |
## 4. Leakage audit (spec 03 §3)

Spec 03 §3 lists inputs that must never be used as normal scoring features, and
requires imputation/scaling/calibration/scoring to be fit within each training
fold. The rescore stage
(`rust/mumdia/crates/mumdia/src/stages/rescore.rs`) consumes every column in
`FeatureSchema.feature_columns` uniformly (`rescore.rs:64,75`) with no
label-leakage screen, so leakage prevention currently depends entirely on which
columns the features stage emits, not on a guard in the classifier.

| Forbidden input (spec 03 §3) | Used as a feature today? | Assessment |
|---|---|---|
| Final q-value | No | `q_value` / `peptide_q_value` / `pg_q_value` are computed after scoring in rescore and written to `psms_scored`; they are not in the feature list. |
| Final posterior error probability | No | Not computed as a feature. |
| Final target-decoy winner | No | Competition winner is not fed back; `label` is used only for training targets/decoys, not as a feature column. |
| Candidate rank generated by the same model | No (with caveat) | `prelim_score` seeds the native semi-supervised loop as `init_score` (`rescore.rs:321`) but is bookkeeping, not a feature column. In the native path it is used only within-fold via `target_decoy_q` on train rows, so it is not a direct leak. It is not an output-model rank fed back to itself. |
| DIA-NN score or identification status | No | The imported DIA-NN library contributes predicted fragment intensities and iRT only; DIA-NN scores/IDs are not ingested as features. |
| Protein evidence computed after precursor scoring | No | `n_proteins` is protein-group multiplicity from the library/digest, available before scoring; `pg_q` is computed after and is not a feature. |
| Cross-run evidence using the held-out run | No | Single-run pipeline. `cross_charge_intensity_log` / `n_charge_states` aggregate charge siblings within the same run, not across runs. No cross-run feature exists. |
| Calibration values fit using the held-out candidate | RISK | RT features (`rt` family, `rt_error_*`) use per-run LOESS/linear RT calibration fit on all confident seed PSMs before cross-validation; mass-accuracy features use the per-run `masscal.json` offset/tolerance; the optional DeepLC iRT fine-tune is likewise global. The native rescorer folds by `base_peptide_id`, so a held-out peptide's own confident seed PSM can be inside the calibration anchor set. This is a genuine, documented pre-CV global-fit leakage path (fdr-rescore map, leakage risk #4). |
| Features computed globally before cross-validation | RISK | Same root cause: search-seed mass recalibration, rt-im-train RT calibration, and DeepLC fine-tune are one-shot whole-run fits, and the derived RT/mass/iRT features carry that global fit into every fold. Standardization itself is fit within-fold in the native path (`fit_standardizer` on train rows only, `rescoring.rs:120`), so the leak is in the upstream calibration, not the scaler. |

Two further leakage risks recorded in the rescore/fdr stage map, not tied to a
single feature:

1. Mokapot fold split is not peptide-grouped. `run_mokapot` writes the PIN keyed
   on a flat row index (`rescore.rs:515-522`) and mokapot's internal brew CV can
   place charge/mod variants of one peptide in different folds. The native
   `percolator_lite` path avoids this by folding on `base_peptide_id`
   (`rescoring.rs:107`); the mokapot sidecar carries a peptide-level leakage risk
   the native path does not.
2. No label-leakage screen exists on the feature columns
   (`rescore.rs:64,75`). Any future feature that encodes decoy/reverse status
   (for example a sequence-derived quantity that differs systematically for the
   documented reverse/scramble decoy scheme) would leak the label directly. This
   is also the caution behind spec 04 §9 (competition features must be computed
   symmetrically for targets and decoys). New contested/claimant features must be
   audited for target/decoy symmetry before they are added.

## 5. Gaps vs spec 03 §8 (candidate feature families to add)

Status of each requested family against the current registry. "Exists" means a
populated, scored column; "present-but-dead" means the column is emitted but
returns a constant because an upstream input is not persisted; "partial" means
some members exist and others do not; "missing" means no analog is scored.

| Spec 03 §8 family | Status | Evidence in the current registry / gap |
|---|---|---|
| 8.1 Uncertainty-normalized residuals | Partial | Signed/abs/squared and gradient- and peak-width-normalized RT residuals exist (`rt` family). They are NOT divided by a per-candidate RT uncertainty: `w_rt` is a single global scalar in `cal.json`, and `Evidence` carries only `e.pred` (a point prediction), no per-fragment `pred_sigma`. Mass errors (`mass_accuracy`) are not normalized by a predicted per-fragment mass uncertainty. Mobility residuals are missing (IM null, 3D MVP). |
| 8.2 Fragment evidence distributions | Mostly exists | `coelution` (38) covers median/min/IQR co-elution and `n_coelution_above` (fraction above threshold); `mass_accuracy` covers median ppm + dispersion; `similarity` covers `frac_top3_predicted_observed`, `library_recall_intensity` (explained predicted intensity), `gini_diff` (evidence concentration); `interference` covers explained observed intensity. Effective-fragment-count as a named feature is the main missing member. |
| 8.3 Apex dispersion | Exists | `chromatographic` has `frag_apex_rt_dispersion`, `frag_apex_rt_dispersion_weighted`, `frag_apex_offset_from_profile_mean`; `coelution` has `frac_frags_apex_aligned`. Precursor-to-fragment apex deviation exists in `ms1` (and the mis-scaled `ms1_ms2_apex_rt_delta` in `novel`, flagged for a fix in CLAUDE.md). |
| 8.4 Peak-shape evidence | Mostly exists | `chromatographic` (43) covers Gaussian/EMG fits, FWHM, asymmetry, tailing, roughness/zigzag, and bimodal detection (a local-maxima proxy). Missing: an explicit peak-truncation indicator, boundary-agreement, and a distinct shoulder score. Consensus profile and fraction-explained-by-consensus are computed internally (`ref_profile` / `interference`) but not all exported as named peak-shape features. |
| 8.5 Interference and contested evidence | Partial | Per-candidate proxies exist: `interference` (26) with contrast/residual/apex-purity/competing-peak counts/PCA rank, plus `peak_contested_frac` (contested intensity fraction from extract two-pass arbitration) and `resid_corr` (residual similarity). Missing the cross-candidate claimant-graph members: number of claimants per fragment, unique-fragment count, unique-fragment intensity fraction, shared-fragment count, shared-trace correlation, local/isolation-window candidate density, and correlation with competing candidate profiles. These require the fragment claimant/conflict graph (spec 04 §5), which extract computes internally (the `contested` map) but does not persist as edges. The scaffolded `CompetitionMode::UniqueEvidence` needs a `unique_fragment_count` feature that does not exist yet. This is the largest feature gap. |
| 8.6 Candidate ambiguity | Missing | No margin/rank-gap features are scored today. `compete` drops losers without recording margins; the new `emit_competition_audit` scaffolding would expose group size, winner/loser scores, and margins, but no ambiguity features (margin-from-best-alternative-peak/peptide/decoy, candidates-within-threshold, score entropy, near-isobaric count, alternative-localization count) are emitted. `n_charge_states` / `charge_multi_flag` are cross-charge corroboration, not ambiguity margins. Note spec 03 §8.6 forbids feeding the final-model margin back into the same model, so these must come from an earlier-stage score. |
| 8.7 MS1 and isotope evidence | Partial | Apex-level isotope agreement exists: `isotope_corr` (averagine correlation), `ms1_isom1_ratio` (incorrect-monoisotope / iso-1 contamination), `log_mono_ms1`, `has_ms1`, and the `ms1` family's isotope-envelope kernels. Present-but-dead: the 10 MS1/MS2 XIC co-elution/shape features (`ms1.rs`, `xic_features`) return 0.0 for every PSM until extract persists `ms1_xic`, so isotope co-elution and precursor-fragment apex agreement over the XIC are not yet live. |
| 8.8 Modification-aware evidence | Missing | No modification-count/class, prediction-training-coverage indicator, modification-specific RT/spectral residual percentile, localization ambiguity, site-determining-ion count/intensity, or alternative-peptidoform count. `novel` carries only `peptide_length` and charge metadata. The peptidoforms stage also drops a second mod at the same position and does no localization scoring, so the upstream evidence is not produced. |
| 8.9 Run context | Partial | Scans-across-peak exists (`n_observations`, `peak_scans::n_peak_scans`, `base_width_rt`). Missing as features: isolation-window width, cycle time, local signal density, local candidate density, run-level calibration quality (`w_rt` / `masscal` live in `cal.json` / `masscal.json` but are not exposed as feature columns), and run-level prediction-residual statistics. Run-context features must be normalized by run when added. |

### Summary

Well covered: fragment evidence distributions (8.2), apex dispersion (8.3),
peak shape (8.4), and apex-level isotope evidence (part of 8.7).

Largest gaps, in priority order matching spec 04 and the sensitivity backlog:

1. Cross-candidate contested/claimant-graph features (8.5) - blocked on
   persisting the extract `contested` conflict edges; unlocks
   `CompetitionMode::UniqueEvidence` and margin/ambiguity work.
2. Candidate-ambiguity margins (8.6) - blocked on the top-K peak dimension and
   the competition-audit emission.
3. Uncertainty-normalized residuals (8.1) - blocked on persisting a per-candidate
   RT/mass/iRT uncertainty (`pred_sigma`) rather than the global `w_rt` scalar.
4. Modification-aware evidence (8.8) - blocked on localization scoring in the
   peptidoforms stage.
5. Live MS1 XIC isotope co-elution (8.7) - blocked on extract persisting
   `ms1_xic`.
6. Run-context features (8.9) - straightforward to add from `cal.json` /
   `masscal.json` and the acquisition grid.
