# features (Stage E): the feature battery + PIN

> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

Stage E (`mumdia features`) turns one extracted PSM (apex-level identification
plus its per-fragment chromatograms) into a fixed, named, versioned feature
vector for the semi-supervised rescorer. It computes a config-selected feature
set, a scalar `prelim_score` used by `compete`, a Percolator PIN, and a
`blake3` schema id that pins the ordered column list so the classifier is never
trained or applied under a mismatched feature layout.

The stage is pure per-PSM: each row's features are a function of that row's own
inputs only, with two cross-row exceptions computed up front (charge-state
corroboration grouped by peptidoform, and a global elution half-width learned
from the confident seed set). This makes the heavy per-PSM work embarrassingly
parallel (`rayon`) while keeping byte-identical output to a serial run.

Everything is driven by `FeaturesConfig` (`mumdia-core/src/config.rs:618`). The
three feature sets are `Minimal` (14), `Rich` (44), and `Extended` (381). The
default is `Minimal`; the tuned `--profile dia` preset and the DIA-NN-library
recipe use `Extended`.

## Files

| path | role |
|---|---|
| `rust/mumdia/crates/mumdia/src/stages/features.rs` | stage entry point, `run`, `Evidence`, `build_evidence`, `fragment_features`, boundary detection, `prelim_score`, PIN, schema hash, the Minimal/Rich column lists, and the Extended family registry |
| `rust/mumdia/crates/mumdia/src/stages/features/similarity.rs` | Extended family: observed-vs-library intensity agreement kernels (64 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/entropy.rs` | Extended family: spectral-entropy / information-divergence (18 names); also exports the gate kernel `spectral_entropy_similarity_sqrt` |
| `rust/mumdia/crates/mumdia/src/stages/features/coelution.rs` | Extended family: fragment-vs-reference and pairwise co-elution + cross-correlation (38 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/interference.rs` | Extended family: co-isolation / chimera detection, interference removal, rank decomposition (26 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/chromatographic.rs` | Extended family: peak-shape descriptors of the reference profile and fragments (43 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/mass_accuracy.rs` | Extended family: fragment ppm distribution + positive mass evidence (17 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/ion_series.rs` | Extended family: b/y series coverage, runs, complementarity, per-series similarity (34 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/ms1.rs` | Extended family: precursor isotope-envelope agreement + MS1/MS2 XIC co-elution (25 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/rt.rs` | Extended family: RT-agreement variants (13 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/novel.rs` | Extended family: seed corroboration + precursor/charge metadata (12 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/nonzero.rs` | Extended family: zero-ignoring apex/co-elution variants (12 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/order_consistency.rs` | Extended family: prediction-free MS2-XIC rank stability (8 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/peak_scans.rs` | Extended family: peak-scan count / window-degeneracy indicator (2 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/apex_dispersion.rs` | Extended family: fragment apex dispersion + consensus peak shape (13 names) |
| `rust/mumdia/crates/mumdia/src/stages/features/mass_uncertainty.rs` | Extended family: fragment mass-error dispersion + evidence breadth (10 names) |
| `rust/mumdia/crates/mumdia/src/stats.rs` | shared `pearson`/`cosine`/`spectral_angle` kernels used by every family |
| `rust/mumdia/crates/mumdia-core/src/config.rs:618` | `FeaturesConfig` |

## Inputs and outputs

`FeaturesParams` (`features.rs:503`) names the paths: `psms` (extracted PSMs),
`chromatograms`, optional `seed` (seed PSMs), `out` (features Parquet),
`out_pin` (PIN), `cfg`, `config_hash`.

### Consumed: psms_extracted (`features.rs:516`-`540`)

Columns read (getter -> column): `candidate_id` (u32), `apex_rt` (f64),
`apex_intensity` (f32), `n_matched_fragments` (i32), `n_predicted_fragments`
(i32, optional; defaults to 6 when absent), `coelution_run` (i32),
`rt_pred_cal` (f64), `charge` (i32), `label` (str), `base_peptide_id` (u32),
`peptidoform` (str), `protein` (str), `precursor_mz` (f64). Optional soft-
competition columns default to 0.0 when absent: `contested_frac`,
`contested_count_frac`, `apportioned_frac`. Optional MS1 apex isotope columns
(`opt_f64`, default `None`): `ms1_isom1`, `ms1_mono`, `ms1_iso1`, `ms1_iso2`.

### Consumed: chromatograms (`features.rs:543`-`570`)

`candidate_id` (u32), `frag_name` (str), `frag_mz` (f64), `frag_obs_mz` (f64,
optional; falls back to `frag_mz`), `predicted_intensity` (f32), `rt`
(list<f32>), `intensity` (list<f32>). Rows whose `frag_name` starts with
`ms1_` are routed to a separate `ms1x` map and fed to the MS1 XIC evidence;
all others are the fragment chromatograms.

### Consumed: seed PSMs (optional, `features.rs:576`-`600`)

`candidate_id` (u32), `score` (f64), `spectrum_q` (f64), `label` (str). Builds
the `candidate_id -> seed score` and `-> identified flag` maps, and the
confident-target set (`spectrum_q <= 0.01` and `label == "target"`) used for
`bound_from_confident`.

### Produced: features Parquet (`features.rs:827`-`843`)

Bookkeeping columns, in order: `candidate_id` (u32), `label` (str),
`base_peptide_id` (u32), `peptidoform` (str), `protein` (str), `apex_rt` (f64),
`elution_lo` (f64), `elution_hi` (f64), `precursor_mz` (f64), `prelim_score`
(f64). Then one `F64` column per name in `active_features(cfg.set)`, in order.
`elution_lo`/`elution_hi` are the RT bounds the stage actually used (emitted so
downstream and plotting read them rather than re-derive).

### Produced: companion + PIN + report

- `<out>.schema.json` (`features.rs:846`): `FeatureSchema { feature_columns,
  schema_id }`, read back by `FeatureSchema::read` (`features.rs:241`).
- `<out_pin>`: Percolator PIN, header
  `SpecId Label ScanNr ExpMass CalcMass <features...> Peptide Proteins`
  (`write_pin`, `features.rs:1388`).
- `<out>.report.json` (`ArtifactReport`, `features.rs:866`): logical name
  `features`, schema version 1, `stats` carrying `feature_schema_id`,
  `n_features`, and `set`; `params` records `set` and
  `coelution_corr_threshold` (`features.rs:873`); `content_hash` is the blake3 of
  the features Parquet; `model_identity` is `None`.

## How it works

Control flow of `run` (`features.rs:514`):

1. Read the PSM table and pull the scalar columns (`features.rs:516`-`540`).
2. Read chromatograms and group `ChromRow`s by `candidate_id` into `chrom`,
   splitting off `ms1_*` rows into `ms1x` (`features.rs:543`-`570`). A
   `ChromRow` (`features.rs:264`) holds `frag_name`, `frag_mz`, `frag_obs_mz`,
   `pred_int`, and the `rt`/`inten` vectors.
3. Build the seed maps and confident-target set from the optional seed table
   (`features.rs:576`-`600`).
4. If `bound_from_confident` is set, learn a global elution half-width
   (see "bound_from_confident" below), producing `Option<(L, R)>` in seconds
   (`features.rs:606`-`650`).
5. Compute the gradient as `max(apex_rt).max(1.0)` (`features.rs:652`); this is
   the run length used to normalize RT errors.
6. Compute the three cross-charge corroboration columns by grouping rows by
   `peptidoform` (`features.rs:662`-`674`).
7. In parallel over rows (`features.rs:697`-`742`), compute the two expensive
   per-PSM pieces: `fragment_features` (the Minimal/Rich fragment battery) and,
   when Extended, `build_evidence` + `extended_values`. Results collect into a
   `Vec<PerPsm>` indexed by row, preserving order.
8. Serially assemble `fmap` (name -> per-row value vector), `prelim`, and the
   elution bounds (`features.rs:744`-`819`). The serial loop reads `per[i]` and
   pushes each named value with the `push` closure (`features.rs:678`).
9. Insert the three cross-charge columns (`features.rs:822`-`824`).
10. Build the output columns (bookkeeping + `active_features`), write Parquet,
    write the schema JSON, build the feature matrix, write the PIN, and emit the
    report (`features.rs:826`-`878`).

### Fragment features (`fragment_features`, `features.rs:1078`)

For each fragment it takes the observed intensity at the scan nearest the apex
RT (`obs`), the predicted intensity (`pred`), and `|ppm|`. It then computes:
`frag_corr`=`pearson(obs,pred)`, `frag_cosine`=`cosine(obs,pred)`,
`spectral_angle`; sum-normalized L1 (`norm_manhattan`) and `rmsd`; intensity-
weighted and unweighted mean `|ppm|`; b/y intensity sums and counts.

It then aligns all fragment traces on the union RT axis (`axis_full`), restricts
to the elution peak (`lo_i..=hi_i`) so co-elution/profile features are not
diluted over the whole `+/- w_rt` window, and computes pairwise Pearson
statistics (`coelution_mean`, `coelution_best`, `n_coelution_above`), lag-
optimized cross-correlation (`xcorr_coelution` mean absolute lag,
`xcorr_shape`), the DIA-NN-style profile block (`profile_cos` = elution^2-
weighted spectral cosine, `ref_corr` = mean fragment-vs-reference Pearson,
`best_ref_corr`, `low_frag_coel`), and the interference-correction block
(`evidence` = summed fragment-vs-reference correlations, `contrast_min`,
`resid_corr`, `coel_clean`, `shadow_frac` from the `1.5*r*ref` cap). Finally
`log_sn` from apex vs median trace point. `elution_lo`/`elution_hi`/
`base_width_rt`/`n_observations` come from the peak-bounded axis.

### Elution-boundary detection

`peak_bounds` (`features.rs:946`) descends from the apex-nearest scan while the
smoothed profile stays `>= frac * apex_height`, bridging up to `grace`
consecutive sub-threshold scans. If the supplied apex sits at zero height it is
relocated to the global maximum first (`features.rs:955`), so a zero-height apex
does not collapse the window. The reference profile for boundary finding is the
`smooth3` (`features.rs:925`) of the summed top-3-predicted-intensity fragment
XICs.

### prelim_score (`features.rs:815`)

```
prelim = n_matched * (0.5 + max(0, frag_corr))
       + max(0, coelution_mean)
       + 0.1 * ln(1 + apex_intensity)
       - rt_err / gradient
```

A cheap heuristic (not a trained score) that rewards matched-fragment count
scaled by spectral correlation, co-elution, and log intensity, penalized by
gradient-normalized RT error. `compete` uses it as the within-group ranking key.

### Directly-computed Minimal/Rich scalars (serial loop, `features.rs:744`-`818`)

Several Minimal/Rich columns are assembled inline in the serial loop rather than
inside `fragment_features`:

- `rt_error_abs` = `|apex_rt - rt_pred_cal|`; `rt_error_rel` = that over
  `gradient` (`features.rs:754`-`755`).
- `log_apex_intensity` = `ln(1 + apex_intensity)` (`features.rs:758`).
- `n_matched_fragments`, `coelution_run`, `charge` pass through from the PSM
  columns; `peptide_length` calls `peptide_length(peptidoform)`.
- `n_proteins` = `protein.matches(';').count() + 1` (`features.rs:767`): the
  `protein` string is semicolon-delimited, so this counts group membership.
- `diff_by_intensity` = `sum_b_intensity - sum_y_intensity` (`features.rs:774`).
- `matched_fraction` = `n_matched / max(1, n_predicted)` (`features.rs:788`-
  `792`).

### MS1 isotope features (Rich set, `isotope_features`, `features.rs:1367`)

The four Rich MS1 columns (`isotope_corr`, `ms1_isom1_ratio`, `log_mono_ms1`,
`has_ms1`) come from `isotope_features` (`features.rs:1367`) over the apex
isotope intensities carried on the PSM rows. It fits a Poisson-averagine envelope
`[1, lambda, lambda^2/2]` with `lambda = 0.00052 * neutral_mass`
(`features.rs:1377`), the neutral mass being
`precursor_mz * charge - charge * PROTON` (`features.rs:750`). This `0.00052`
averagine differs from the `0.000594` used by the Extended `ms1` family
(`ms1.rs:81`); the two MS1 code paths are independent and both are kept.
`isotope_corr` = `pearson([mono, +1, +2], theo)`, `ms1_isom1_ratio` =
`isom1 / (mono + 1)`, `log_mono_ms1` = `ln(1 + mono)`, `has_ms1` = 1.0 when
`mono`/`+1`/`+2` are all present, else all four return 0.0.

### The Evidence struct (`features.rs:278`)

`build_evidence` (`features.rs:345`) constructs the per-PSM `Evidence` handed to
every Extended family. It mirrors the alignment and peak-bounding of
`fragment_features` so families see the same elution peak. Fields:

- Time series: `axis` (RT seconds over the detected elution peak), `traces`
  (per-fragment intensity over `axis`, zero-filled, fragment order), `axis_full`
  / `traces_full` (whole extracted window), `apex_idx` (index of apex in
  `axis`), `ref_profile` (predicted-intensity-weighted sum of the peak traces).
- Fragment-indexed arrays (shared order): `pred` (library intensity),
  `obs_apex` (intensity at the apex scan; `> 0` defines "matched"), `is_b`,
  `ordinal`, `frag_charge`, `frag_mz` (theoretical), `frag_obs_mz` (intensity-
  weighted observed), `mass_err_ppm` (signed ppm).
- `apex_rt` is set inside `build_evidence` itself (`features.rs:483`) from its
  `apex_rt` argument, not by the caller.
- Scalars filled by the caller after build (`features.rs:718`-`732`):
  `rt_pred_cal`, `rt_err`, `gradient`, `precursor_mz`, `charge`, `seq_len`,
  `n_matched`, `n_predicted`, `seed_score`, `seed_identified`, `apex_intensity`
  (plus the MS1 apex isotopes below, `features.rs:729`-`732`). All start at a
  zero/`None` default set by `build_evidence` (`features.rs:483`-`499`).
- MS1: `ms1_mono`/`ms1_iso1`/`ms1_iso2`/`ms1_isom1` (apex isotope intensities,
  `None` when no MS1) and `ms1_xic` (the `[mono,+1,+2]` XICs resampled onto
  `axis`; empty unless the extract stage persisted `ms1_*` chromatogram rows).

`parse_ion` (`features.rs:331`) parses `b3`, `y7`, `b3^2` into
`(is_b, ordinal, charge)`.

### The Extended battery

`FAMILIES` (`features.rs:52`) is an ordered array of `(NAMES, values)` pairs;
the order is part of the frozen schema and is append-only. Each family exposes
`NAMES: &[&str]` and `values(&Evidence) -> Vec<f64>` of identical length and
matching order. `extended_values` (`features.rs:134`) calls each family and
applies the precomputed dedup plan (`extended_value_plan`, `features.rs:108`),
keeping names and values in lockstep, and coerces any non-finite value to 0.0.

Deduplication (`extended_name_refs`, `features.rs:80`): a name that already
appears in `MINIMAL_FEATURES` or `RICH_EXTRA` (`reserved_names`,
`features.rs:72`), or that repeats across families, is kept only on first
appearance. In the current tree four Extended names are dropped as reserved
collisions: `spectral_angle` (similarity vs Minimal), `rt_error_abs` (rt vs
Minimal), `peptide_length` and `seed_identified` (novel vs Minimal/Rich). So the
335 raw family names reduce to 331 unique Extended names.

## The three feature sets

`active_features(set)` (`features.rs:206`) returns the ordered active column
list:

- `Minimal` (14, `MINIMAL_FEATURES` `features.rs:154`): `rt_error_abs`,
  `rt_error_rel`, `n_matched_fragments`, `coelution_run`, `log_apex_intensity`,
  `frag_corr`, `frag_cosine`, `spectral_angle`, `coelution_mean`,
  `coelution_best`, `n_coelution_above`, `charge`, `peptide_length`,
  `n_proteins`.
- `Rich` (44 = Minimal + 30, `RICH_EXTRA` `features.rs:172`): adds
  `library_norm_manhattan`, `library_rmsd`, `xcorr_coelution`, `xcorr_shape`,
  `sum_b_intensity`, `sum_y_intensity`, `diff_by_intensity`, `n_b_ions`,
  `n_y_ions`, `weighted_mass_error`, `mean_mass_error`, `isotope_corr`,
  `ms1_isom1_ratio`, `log_mono_ms1`, `has_ms1`, `log_sn`, `n_observations`,
  `base_width_rt`, `seed_score`, `seed_identified`, `matched_fraction`,
  `profile_cos`, `ref_corr`, `best_ref_corr`, `low_frag_coel`, `evidence`,
  `contrast_min`, `resid_corr`, `coel_clean`, `shadow_frac`.
- `Extended` (381 = Minimal + Rich + 331 family names + 6 psms-derived): appends
  `extended_names()` then six columns computed outside the family registry.
  Three co-elution peak-contest metrics (`peak_contested_frac`,
  `peak_contested_count_frac`, `peak_apportioned_frac`) come from the PSM
  columns (default 0.0 when absent) and separate peak-borrowing decoys from
  genuine IDs. Three charge-corroboration columns (`n_charge_states`,
  `charge_multi_flag`, `cross_charge_intensity_log`) aggregate across the charge
  states of one peptidoform, an axis invisible to the per-PSM Evidence families.

The size invariant is asserted in `feature_sets_sized` (`features.rs:1434`):
`Extended.len() == 14 + 30 + extended_names().len() + 6`, and all Extended names
are unique. `FeatureSet` (`config.rs:81`) has exactly `Minimal`, `Rich`,
`Extended`; the earlier `Custom` variant is gone (do not document it).

## Extended family reference

Each family below lists its name count and what it measures. All values are
finite (NaN/Inf coerced to 0.0), and every family returns a stable-length vector
even on degenerate evidence.

### similarity (64, `similarity.rs`)

Observed-vs-library fragment-intensity agreement under many kernels. `o` is the
per-fragment apex intensity, `l` the predicted intensity; `_matched` restricts
to `o_i > 0`, `_area` replaces `o` with the per-fragment peak-XIC trapezoid
area, `on`/`ln` are sum-normalized. Names: `spectrum_cosine_matched`,
`spectrum_cosine_sqrt`, `spectrum_cosine_log`, `spectral_angle` (dropped as
reserved), `spectral_angle_sqrt`, `spectral_angle_matched`,
`pearson_intensity_matched`, `pearson_intensity_log`, `spearman_intensity`,
`spearman_intensity_matched`, `kendall_tau_intensity`, `dot_product_raw`,
`dot_product_norm`, `library_recall_intensity`, `manhattan_sim`,
`manhattan_sqrt`, `rmsd_norm`, `mae_norm`, `mse_log`, `mae_weighted_pred`,
`abs_diff_q3`, `max_positive_residual`, `chebyshev_dist`, `minkowski_p3`,
`bray_curtis`, `bray_curtis_sqrt`, `canberra`, `canberra_matched`,
`wave_hedges`, `chi_square_pearson`, `chi_square_symmetric`,
`divergence_distance`, `bhattacharyya_coef`, `hellinger`, `squared_chord`,
`harmonic_mean_sim`, `jaccard_presence`, `dice_presence`,
`intensity_weighted_pearson`, `regression_slope`, `gini_diff`, `wasserstein_mz`,
`footrule_norm`, `rank_overlap_top3`, `top1_frag_match`,
`top1_predicted_observed`, `frac_top3_predicted_observed`,
`count_strong_predicted_absent`, `frac_predicted_absent`, `cosine_area`,
`pearson_area`, `spectral_angle_area`, `cosine_fullwindow`,
`stein_scott_weighted_dot`, and the unbounded/granular scores that address
cosine saturation (`log_dot_product`, `spectral_log_evidence`, `scribe_score`,
plus their `_area` twins), `cosine_high_ordinal` (drop b1/b2/y1/y2), and the
robust trimmed-cosine trajectory (`cosine_robust_trim1/2/3`). Local helpers
include tie-corrected `ranks`, `kendall_tau_b`, interpolated `quantile`, `gini`,
`weighted_pearson`, and `trapz`.

### entropy (18, `entropy.rs`)

Li spectral-entropy similarity and information divergences between sum-
normalized `o` and `l`. `entropy_sim` (`entropy.rs:91`) is
`1 - (2 H(m) - H(o) - H(l)) / ln 4` with `m = (o+l)/2`, clamped to [0,1].
Names: `spectral_entropy_similarity`, `weighted_spectral_entropy_similarity`
(Li per-spectrum weighting), `spectral_entropy_similarity_sqrt`,
`spectral_entropy_similarity_topk` (top-6 by predicted intensity),
`spectral_entropy_similarity_area`, `jensen_shannon_divergence`,
`jeffreys_divergence`, `kl_obs_pred`, `kl_pred_obs`, `cross_entropy_obs_pred`,
`obs_spectrum_entropy`, `pred_spectrum_entropy`, `entropy_diff`,
`entropy_ratio`, `obs_normalized_entropy` (Pielou evenness),
`normalized_entropy_diff`, `residual_spectrum_entropy`, `entropy_weight_obs`.
The public `spectral_entropy_similarity_sqrt(obs, pred)` (`entropy.rs:82`) is
reused by the extraction gate `GateMode::SpectralEntropy` so the kernel is not
duplicated.

### coelution (38, `coelution.rs`)

Fragment co-elution against the predicted-weighted reference profile R and
pairwise between fragments. Three groups: per-fragment vs R (peak/full windows,
leave-one-out), pairwise Pearson statistics and lag cross-correlation
(`best_xcorr`, `MAXLAG=5`), and structured cross-correlations (b vs y ions,
charge-1 vs multiply-charged). Names include `frag_ref_corr_mean`,
`frag_ref_corr_obsweighted`, `frag_ref_corr_min`, `frag_ref_corr_std`,
`frag_ref_corr_sq_mean`, `frag_ref_corr_topk_weighted`,
`n_frag_ref_corr_above_0_9`, `frac_frag_ref_corr_above_0_8`,
`frag_ref_corr_mean_full`, `full_vs_peak_corr_gain`,
`pairwise_coelution_weighted`, `pairwise_coelution_min/median/std/frac_negative`,
`pairwise_coelution_hi/lo`, `coelution_hi_lo_contrast`,
`coelution_corr_entropy` (Shannon entropy of a 10-bin `NBINS_CORR`
pairwise-Pearson histogram over `[-1, 1]`), the `xcorr_shape_*` and
`xcorr_lag_*` statistics
(`_mean/_min/_std/_mean_abs/_iqr/_frac_zero/_max_abs/_entropy`),
`ref_xcorr_lag_mean`, `ref_xcorr_shape_mean`, `observed_sum_vs_template_corr`,
`frag_loo_ref_corr_mean/_min`, `frac_frags_apex_aligned`, `top3_frag_ref_corr`,
`by_cross_coelution`, `by_cross_lag_mean`, `charge_cross_coelution`. Note the
JSON `coelution_weighted_mean` is an exact alias of `pairwise_coelution_weighted`
and is emitted once under the latter.

### interference (26, `interference.rs`)

Co-isolation/chimera detection via the least-squares scale
`r_f = <x_f, R> / <R, R>` projecting each fragment onto R. Two gating constants:
`IFS_MIN_CORR = 0.6` (`interference.rs:48`) prunes the least-coherent matched
fragment in the iterative `remove_ifs` loop until its leave-one-out ref-corr
stays above it or only 3 fragments remain, and `COHERENT_THR = 0.7`
(`interference.rs:47`) gates `explained_apex_intensity_frac` and `apex_purity`
(only fragments with ref-corr at or above it count as explained);
`n_interfered_fragments` flags a fragment whose apex intensity exceeds
`2 * r_f * R_apex`. Names: `explained_variance_ref`,
`profile_residual_fraction`, `n_interfered_fragments`,
`corrected_vs_raw_cos`, `corrected_vs_raw_ratio`, the iterative interference-
removal block (`ifs_removed_count`, `ifs_removed_intensity_frac`,
`ifs_corr_gain`, `ifs_retained_frac`, `matched_frac_after_ifs`), the peak-vs-
full area ratios (`peak_to_full_area_ratio_profile/_frag_mean/_weighted`,
`out_of_peak_intensity_frac`), `profile_corr_full_vs_peak_delta`,
`frac_frag_ref_corr_below_0_5`, `explained_apex_intensity_frac`, `apex_purity`,
`interference_apex_residual_fraction`, `dominant_frag_ref_corr`, the rank
decomposition of the matched fragment x time Gram matrix by power iteration
(`explained_variance_ratio`, `second_component_fraction`), competing-peak
descriptors on the full-window profile (`profile_second_peak_ratio`,
`n_competing_peaks_in_window`), `matched_pred_intensity_fraction`, and
`top_pred_frag_matched`.

### chromatographic (43, `chromatographic.rs`)

Peak-shape quality of R (Gaussian moment-match fit, EMG grid-fit comparison via
`erfc`) and of individual fragments. Names: `gaussian_fit_r2`,
`gaussian_cosine`, `emg_fit_improvement`, `apex_prominence`, `profile_peak_snr`
(MAD-based over out-of-peak scans), width descriptors (`fwhm_seconds`,
`fwhm_to_window_ratio`, `width_at_10pct`, `width_ratio_10_50`),
asymmetry/tailing (`hwhm_asymmetry`, `tailing_factor_usp`,
`asymmetry_factor_10pct`), apex shape (`apex_sharpness`, `apex_curvature`,
`apex_to_boundary_ratio`, `apex_dominance`), roughness (`zigzag_index`,
`jaggedness`, `roughness_2nd_deriv`), multimodality (`n_local_maxima`,
`modality`), RT moments (`rt_skewness`, `rt_excess_kurtosis`, `rt_std_seconds`,
`mean_mode_offset`), area descriptors (`fraction_area_within_fwhm`,
`triangle_area_similarity`), `baseline_fraction`, `peak_completeness`,
`apex_centering_offset`, `intensity_score`, `total_xic_log`, per-fragment
descriptors (`frag_fwhm_cv/_mean`, `frag_apex_rt_dispersion/_weighted`,
`frag_apex_offset_from_profile_mean`, `frag_gaussianity_mean/_weighted`,
`frag_zigzag_mean`), `sumtrace_unweighted_gaussian_r2`, and
`reference_profile_rt_entropy_peak/_ratio`. Has unit tests
(`chromatographic.rs:876`).

### mass_accuracy (17, `mass_accuracy.rs`)

Fragment ppm-error distribution over matched fragments (uses a fixed
`FRAG_TOL_PPM = 20.0`; the config tolerance is not carried in Evidence). Names:
`median_abs_frag_ppm`, `signed_mean_frag_ppm`, `ppm_std`, `ppm_iqr`,
`ppm_range`, `max_abs_frag_ppm`, `intensity_weighted_abs_ppm`,
`intensity_weighted_signed_ppm`, `intensity_weighted_ppm_std`,
`lib_weighted_abs_ppm`, `frac_frag_within_half_tol`, `high_ppm_intensity_frac`,
`ppm_intensity_anticorr`, `mass_error_mz_trend`, `mean_abs_mz_error_da`, and the
positive DIA-NN-style evidence `mass_evidence_gauss` (predicted-weighted
Gaussian concentration, sigma 10 ppm) and `mass_log_evidence`.
`precursor_mass_error_ppm` (in the plan.md feature spec, local only) is
deliberately skipped (no theoretical precursor m/z in Evidence).

### ion_series (34, `ion_series.rs`)

b/y series coverage, ladder contiguity, complementarity, per-series similarity.
Names: `n_matched_b/_y`, `frac_matched_b/_y`, `by_count_balance`,
`by_intensity_ratio`, `by_ratio_agreement` (tanh-squashed log-odds),
`by_ratio_consistency`, `longest_b_run/_y_run`, `longest_run_max`,
`longest_run_frac_length`, `series_coverage_b/_y`, `sequence_coverage`
(cleavage-site union), `series_gap_fraction`, `by_complement_count`,
`by_complement_mz_consistency` (b+y obs m/z vs M + 2 proton),
`by_complement_coelution`, `ordinal_intensity_concordance_y/_b`,
`series_coelution_y/_b`, `spectral_angle_b/_y`, `pearson_b/_y`,
`cosine_charge1/_charge2`, `charge_corr_balance`, `mean_matched_ordinal_norm`,
`by_ion_contiguous_intensity`, `by_ion_contiguous_lib_frac`,
`both_series_present`. Uses `PROTON` from `mumdia-core::constants`.

### ms1 (25, `ms1.rs`)

Precursor isotope-envelope agreement against a Poisson-averagine model
(`lambda = 0.000594 * M`) plus MS1/MS2 XIC co-elution. Apex names:
`ms1_isotope_cosine_apex`, `ms1_isotope_spectral_angle_apex`,
`ms1_isotope_chi2_apex`, `ms1_isotope_manhattan_apex`, `iso_ratio_1_0`,
`iso_ratio_2_0`, `iso_plus1_ratio_dev`, `iso_plus2_ratio_dev`,
`iso_minus_one_fraction`, `iso_overlap_flag`, `log_ms1_mono`,
`ms1_total_isotope_log`, `has_ms1_signal`, `ms1_isotope_apex_entropy_3`,
`ms1_m1_entropy_contribution`. The XIC block (`ms1_ms2_time_corr`,
`ms1_ms2_envelope_time_corr`, `ms1_iso_coelution`, `ms1_ms2_apex_rt_delta`,
`ms1_iso_ratio_stability`, `ms1_mono_gaussianity`, `ms1_ms2_fwhm_ratio`,
`ms1_isotope_corr_xic`, `ms1_envelope_over_time_corr`,
`ms1_isotope_xic_shape_consistency`) reads `Evidence.ms1_xic` and returns 0.0
until the extract stage persists MS1 isotope XICs (evidence gap; currently
unpopulated in the default chain).

### rt (13, `rt.rs`)

RT agreement between the observed apex and the calibrated predicted RT. Names:
`rt_error_signed`, `rt_error_abs` (dropped as reserved), `rt_error_squared`,
`rt_error_signed_norm_gradient`, `rt_error_abs_norm_gradient`, `observed_rt_raw`,
`predicted_rt_raw`, `observed_rt_fraction`, `predicted_rt_fraction`,
`rt_error_over_peak_width` (base width at 10%), `rt_error_over_fwhm`,
`rt_diff_profile_apex` (vs full-window profile argmax RT),
`predicted_rt_in_gradient`.

### novel (12, `novel.rs`)

Seed corroboration and precursor/charge metadata. Names:
`log_seed_hyperscore`, `seed_hyperscore_per_matched`, `seed_identified` (dropped
as reserved), `peptide_length` (dropped as reserved), `precursor_charge`,
`charge_is_2/_is_3/_is_4plus`, `precursor_mass`, `log_total_matched_intensity`,
`n_matched_frags`, `n_predicted_frags`. Sequence-dependent features (missed
cleavages, C-terminal residue, modification count) are skipped because Evidence
carries only `seq_len`.

### nonzero (12, `nonzero.rs`)

Zero-ignoring variants of the apex spectral and co-elution features. The apex
scan samples one grid point, so a fragment peaking one scan off reads 0 there
(~8% of fragments); these recompute over the per-fragment peak-max and over
present-only scans. Names: `frag_corr_peakmax`, `frag_cosine_peakmax`,
`spectral_angle_peakmax`, `frag_corr_matched_nz`, `frag_cosine_matched_nz`,
`peakmax_apex_gain`, `n_frag_present_inpeak`, `frac_frag_present_inpeak`,
`coelution_mean_bothpos`, `coelution_mean_summpos`, `ref_corr_nz`,
`profile_cos_nz`.

### order_consistency (8, `order_consistency.rs`)

Prediction-free MS2-XIC rank stability: at each scan the fragments are ranked by
observed intensity, and the features measure whether that ranking persists
across the peak (orthogonal to library-agreement families). Names:
`rank_corr_vs_apex_mean`, `rank_corr_vs_apex_std`, `rank_corr_adjacent_mean`,
`kendall_vs_apex_mean`, `top1_frag_persistence`, `top2_order_persistence`,
`argmax_frag_entropy`, `self_cosine_vs_apex_mean`. Degenerate below 3 fragments
or 3 non-empty scans. Has unit tests (`order_consistency.rs:271`).

### peak_scans (2, `peak_scans.rs`)

Label-blind window-degeneracy indicator, emitted for every PSM so the rescorer
can tell an undefined zero from a measured zero when the window-based families
collapse. Names: `n_peak_scans`, `peak_window_degenerate` (1 when fewer than 3
non-empty scans, mirroring `order_consistency::MIN_SCANS`).

### apex_dispersion (13, `apex_dispersion.rs`)

Fragment apex dispersion and consensus peak shape, intensity-independent
(breadth of co-elution rather than height). Names: `frag_apex_rt_std`,
`frag_apex_rt_mad`, `frag_apex_max_dev`, `frag_apex_mean_dev`,
`frag_apex_agree_frac`, `precursor_frag_apex_delta`, `peak_symmetry`,
`peak_tailing`, `peak_n_local_maxima`, `peak_shoulder_score`, `peak_fwhm_scans`,
`peak_truncation`, `apex_frac_of_window`. `precursor_frag_apex_delta` reads the
mono MS1 XIC (`ms1_xic[0]`) and is 0.0 until the extract stage persists MS1
isotope XICs (the same evidence gap as the `ms1` XIC block). Has unit tests
(`apex_dispersion.rs:226`).

### mass_uncertainty (10, `mass_uncertainty.rs`)

Fragment mass-error distribution over matched fragments plus evidence-breadth.
Names: `frag_mass_err_median`, `frag_mass_err_abs_median`, `frag_mass_err_std`,
`frag_mass_err_iqr`, `frag_mass_err_max_abs`, `frag_mass_err_range`,
`effective_frag_count` (inverse participation ratio), `evidence_concentration`
(fraction in the strongest fragment), `frac_top3_pred_observed`,
`frac_top5_pred_observed`. Has unit tests (`mass_uncertainty.rs:133`).

## The Percolator PIN (`write_pin`, `features.rs:1388`)

Streamed row-by-row through a `BufWriter` (not materialized as one String).
Header: `SpecId\tLabel\tScanNr\tExpMass\tCalcMass\t<features joined by tab>\t
Peptide\tProteins`. Per row: `SpecId = cand_<candidate_id>`,
`Label = -1` when `label == "decoy"` else `1`, `ScanNr = candidate_id`,
`ExpMass = CalcMass = precursor_mz` (`{:.5}`), each feature at `{:.6}`, then
`Peptide = -.<peptidoform>.-` and `Proteins = <protein>`. The feature matrix
passed to the PIN is built column-parallel from `fmap` in `active_features`
order (`features.rs:856`), so the PIN and the Parquet share the same ordered
feature list.

## The feature-schema hash (`feature_schema_id`, `features.rs:229`)

`blake3_str(cols.join(","))` of the ordered active column list. Written to
`<out>.schema.json` and recorded in the report `stats`. It is a content hash of
the exact ordered names, so any addition, removal, or reordering changes the id
and a classifier trained under one schema is never silently applied under
another. Because the family registry order is frozen and append-only, appending
a new family or feature at the end changes the id predictably while leaving all
prior positions stable.

## bound_from_confident (elution-boundary calibration, `features.rs:606`)

When `bound_from_confident` is true (the default), the stage learns one pair of
elution half-widths `(L, R)` in seconds from the confident-target seed set
(`spectrum_q <= 0.01`, `label == "target"`; the same anchor set used for RT
calibration and DeepLC fine-tune). For each confident candidate it detects the
per-candidate peak with `elution_peak_rt_bounds` (`features.rs:1038`, which
returns `None` for a candidate with fewer than 3 distinct scans, so it does not
contribute an anchor) and records `apex - lo` and `hi - apex`. If at least 20
anchors resolve, it takes the `bound_confident_pct` percentile of the left and
right half-widths (median by default) and returns `Some((L, R))`; every
candidate is then bounded on `[apex - L, apex + R]` via `global_bound_indices`
(`features.rs:1011`), which falls back to the single apex-nearest scan
`(ai, ai)` when the mapped window collapses between grid points (sparse grid or
a half-width below one cycle). This
removes per-candidate boundary manipulation, so a chimeric decoy is scored over
a real-peptide-width window centred on its apex rather than one it can widen or
narrow. With fewer than 20 anchors the stage logs a warning and falls back to
per-candidate boundary detection for that run. When the flag is false, every
candidate detects its own peak boundary from its top-3-predicted-fragment
profile (the legacy path). The same `global_bounds` argument threads into both
`fragment_features` and `build_evidence`, so Minimal/Rich and Extended see the
identical window.

## The shared stats kernel (`stats.rs`)

One implementation of `pearson`, `cosine`, `spectral_angle`, used by
`fragment_features` and every family (do not reimplement). `pearson`
(`stats.rs:6`) is population Pearson with a zero-variance guard returning 0.0
for `n < 2` or zero variance. `cosine` (`stats.rs:29`) returns 0.0 if either
vector is all-zero. `spectral_angle` (`stats.rs:44`) is
`1 - 2 * acos(clamp(cosine, -1, 1)) / pi`, in [0,1] with 1 = identical. Families
add local specializations that do not belong in the shared kernel (weighted
Pearson, Spearman via `pearson` on average ranks, Kendall tau, windowed cross-
correlation via `super::best_xcorr`, `features.rs:1337`, which returns the best
normalized correlation and its integer lag over `[-maxlag, maxlag]` as
`(lag_of_max, value.max(0.0))`), but the base Pearson/cosine call the shared
functions.

## Configuration

`FeaturesConfig` (`config.rs:618`) is `#[serde(default, deny_unknown_fields)]`,
so every field defaults independently and an unknown config key is a hard load
error. The `set` field's default is `t()` (`config.rs:654`), a generic
`Default`-forwarding helper, so it resolves to `FeatureSet::default()` =
`Minimal`. The config was pruned of dead fields (the `FeatureSet::Custom`
variant no longer exists).

| field | default | effect |
|---|---|---|
| `set` | `Minimal` | which set `active_features` returns (Minimal 14 / Rich 44 / Extended 381) |
| `coelution_corr_threshold` | 0.9 | threshold for `n_coelution_above` (count of pairwise fragment correlations at or above it) |
| `prec_tol_ppm` | 20.0 | precursor tolerance carried for feature bookkeeping |
| `bound_features` | true | restrict trace-based features to the elution peak instead of the whole extracted window; **gates only the Minimal/Rich `fragment_features` path** (Extended `build_evidence` always peak-bounds, see gotchas) |
| `bound_peak_fraction` | 1/3 | peak-boundary threshold as a fraction of apex height (DIA-NN-style; matched DIA-NN RT bounds best) |
| `bound_peak_grace` | 0 | consecutive sub-threshold scans to bridge before stopping (0 = stop at first miss; 1 bridges a single-scan dip) |
| `bound_from_confident` | true | learn one global left/right half-width from the confident seed set and apply it to every candidate; false = per-candidate detection |
| `bound_confident_pct` | 50.0 | percentile of the confident-set half-widths taken as the global half-width (50 = median) |

Note that the fragment tolerance used inside `mass_accuracy` is a hardcoded
`FRAG_TOL_PPM = 20.0` (`mass_accuracy.rs:44`), not `prec_tol_ppm`; Evidence does
not carry the configured fragment tolerance.

## Invariants, determinism, gotchas

- Every family must return exactly `NAMES.len()` values in `NAMES` order; a
  `debug_assert_eq!` in `extended_values` (`features.rs:139`) and in most
  families catches a mismatch in debug builds. Non-finite values are coerced to
  0.0 at family boundaries and again in `extended_values`.
- The `FAMILIES` registry order and each family's `NAMES` order are the frozen
  schema; they are append-only. Reordering or renaming changes `schema_id` and
  invalidates any trained classifier.
- Deduplication is stable and precomputed once (`extended_value_plan`,
  `features.rs:108`), reproducing the same survivors and order as
  `extended_name_refs`, so names and values stay in lockstep across runs.
- Determinism: the parallel per-PSM pass collects into a `Vec` indexed by row,
  so the serial assembly is byte-identical to a serial run regardless of thread
  count. RT-axis alignment maps intensities keyed by `f32::to_bits`
  (`features.rs:389`), which is exact-equality safe because the same `rt`
  values are reused, not recomputed.
- "Matched" throughout the families means `obs_apex[i] > 0.0` (observed at the
  apex scan), which differs subtly from "present in the peak" used by the
  `nonzero` family (per-fragment peak-max `> 0`).
- The apex scan samples a single grid point; a fragment peaking one scan off
  reads 0.0 at the apex. The `nonzero` family exists specifically to give the
  classifier zero-tolerant variants alongside the originals.
- `peptide_length` (`features.rs:246`) strips a leading `DECOY_` prefix before
  counting residues, and ignores bracketed modifications, so the decoy marker is
  not a length-based target/decoy label leak (tested at `features.rs:1425`).
- MS1 XIC features return 0.0 unless the extract stage wrote `ms1_*`
  chromatogram rows; `ms1_xic` is otherwise empty. This is a known evidence gap,
  not a bug.
- `bound_features` gates only the Minimal/Rich `fragment_features` path
  (`features.rs:1154`): when false, that path scores over the whole extracted
  window. The Extended `build_evidence` (`features.rs:345`) takes no such flag
  and always peak-bounds `axis`/`traces` while still retaining
  `axis_full`/`traces_full`, so Extended families read whichever window they
  name regardless of `bound_features`, and `global_bounds` from
  `bound_from_confident` always applies to them.
- The peptidoform grouping for cross-charge features uses the ProForma string,
  which is charge-independent and keeps `DECOY_` peptidoforms grouped among
  themselves, so it is not a target/decoy label leak (`features.rs:662`).

## How to extend / modify

- To add a new Extended family: create a module under `stages/features/`
  exposing `pub const NAMES: &[&str]` and `pub fn values(&Evidence) -> Vec<f64>`
  of matching length and order, `mod`-declare it (`features.rs:32`), and append
  `(name, values)` to `FAMILIES` (`features.rs:52`). Appending at the end keeps
  all prior schema positions stable. Reuse `crate::stats` and the parent helpers
  (`mean`, `normalize_sum`, `best_xcorr`, `smooth3`, `peak_bounds`) rather than
  reimplementing kernels. Add an arity unit test (`values(&e).len() ==
  NAMES.len()`) and a degenerate-evidence finiteness test.
- To add a Minimal/Rich feature: append the name to `MINIMAL_FEATURES` or
  `RICH_EXTRA` and push its value in the serial loop (`features.rs:754`); make
  sure no Extended family already uses the name, or it will be dropped as a
  reserved collision.
- New names must be globally unique across Minimal, Rich, and every family; a
  collision is silently dropped by the dedup filter, so run
  `feature_sets_sized` (`features.rs:1434`) after any change to confirm the
  size and uniqueness invariants.
- Do not reach into vendor formats or duplicate the mass model / stats kernel;
  Evidence is the sole per-PSM interface for families, and any new scalar a
  family needs must be added to `Evidence` (`features.rs:278`) and filled by the
  caller (`features.rs:718`).
- Prefer adding a config field (backed by a default) over hardcoding a
  threshold, consistent with the project convention; the current hardcoded
  fragment tolerance in `mass_accuracy` is a documented exception awaiting a
  tolerance carried on `Evidence`.
