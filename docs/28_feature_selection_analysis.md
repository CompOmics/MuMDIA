# 28. Feature selection for rescoring: analysis

Status: analysis only. No engine code changes. Scripts in `bench/feature_selection/`,
outputs under `C:/Users/robbi/mumdia_bench/fs/` (off OneDrive; large). Measured
2026-09-04.

> 2026-09-05: sections 10-19 add the implementation and its measurements. Headline: the
> sidecar handoff is now parquet by default (rescore peak 29.96 -> 8.95 GB); a 114-feature
> list with hard-negative training took the full-scale HYE rescore from 33.04 GiB / 53:07 to
> **5.49 GB / 3:19** at -0.66% peptides; the six-run pooled rescore runs in **18 minutes at
> 15.9 GB** at parity with August (72,344 vs 72,044 peptides); and a complete single run is
> **17:52 at 16.5 GiB** (was 67:30 at 86.6 GiB the day before). A seeded sweep of the
> rescorer's hyperparameters (section 17) and of the pipeline's extraction/RT factors
> (section 18) found the shipped values at a local optimum on every pool. Sections 1-9 are
> the original analysis and stand unchanged. Section 21 records what shipped as default.

## 0. Summary

- Of the 387 Extended features, 43 carry no information the classifier can see: 10 are
  constant under the default configuration, 20 are bit-identical to another column, 13 are
  affine images of another column. The live set is 344, and it has an effective dimension
  of about 120 (PCA: 123 components for 95% of the variance; 239 Spearman clusters at
  |rho| >= 0.90).
- On the full pools, two seeds each, the subsets that hold on both A01 (where they were
  built) and B01 (held out) are: the 344 live features (+0.1% / -0.1%), the 150 most
  important by permutation (+0.1% / +0.1%), and the unions of permutation top-75/100 with
  the L1 set at 138, 122 and 114 features (0.0 to -0.3% on both). Below that the loss is
  real and consistent on the held-out set: 100 features -0.9%, 86 features -1.6%. The
  engine's `rich` set (44) costs 7.5-8.3% and `minimal` (14) costs 22-28%. Cluster
  representatives transfer worst (239 at -1.2%, 189 at -1.4% held out). The decoy fraction
  at 1% is 1.00% in every one of the 130 runs, as target-decoy competition guarantees.
- The real worker confirms it: `nn_rescore_worker.py` on the B01 PIN with
  `MUMDIA_NN_FEATURES` = the 122-feature union gives 58,962 peptides and 66,174 target PSMs
  at 1% against 58,978 and 66,081 for the engine run's own 387-feature pass on the same PIN
  (section 6.4).
- No feature family is indispensable: removing any one family alone costs at most 2.1%
  (coelution) and most cost nothing measurable, because every family is covered by others.
  No family is sufficient alone (each alone loses 14% to 100%).
- Feature selection buys memory and I/O, not training time. The Rust matrix, the PIN and the
  worker's matrices scale with the feature count (a 122-feature set is 3.2x smaller than
  387), but the MLP's CPU training time per row is flat in the feature count: 4.1-4.6 us per
  row per epoch at 32 threads for every width from 387 down to 100, measured uncontended on
  doxy, and the real worker trained for 2,879 s on 122 features against 2,753 s on 387.
  Per-batch overhead dominates, not the first-layer GEMM. Training speed has to come from
  the knobs already measured in August (warm start, batch size, GPU), not from fewer
  features.
- The largest memory lever in rescoring is not feature selection but the handoff format:
  `rescore.handoff = tsv` (the default) makes the worker parse a 9.5 GB text file through
  `pandas.read_csv` into a float64 frame before it builds its float32 matrix, and on that
  path a feature subset does not reduce the parse. `Handoff::Parquet` already exists and
  reads only the requested columns; with it, a subset shrinks every Python allocation.
- Recommended next step, if pursued: the 114-feature deduplicated union set (section 6.3;
  its 122-feature parent is the one validated through the real worker), together with
  `rescore.handoff = parquet`, gated on the AIF benchmark and an entrapment run before it
  touches any default (CLAUDE.md, "Validate new sensitivity defaults"). Expected gain on
  the HYE run: rescore stage peak from 33 GiB to roughly 10 GiB and the PIN from 9.5 GB to
  3 GB (or a 1.2 GB parquet); expected time gain: a few percent.

## 1. Question and why it matters

The Extended feature set is 387 columns. Rescore is the tallest stage of a single run
(33.04 GiB, docs/27 section 0) and 79% of its wall clock (3,187 of 4,050 s on the HYE
benchmark), and every byte of it scales with the feature count `D`:

| quantity | at D = 387, N = 2,603,894 PSMs | scaling |
|---|---|---|
| Rust `FeatureMatrix` (f32) | 3.75 GiB (measured, `mem: rescore feature matrix`) | N x D x 4 |
| PIN TSV handed to the sidecar | 9.54 GB, 3,663 B/row, ~9.3 B per value | N x D x ~9.3 |
| worker `pandas.read_csv` frame (TSV handoff) | ~8 GB of float64 feature columns plus object columns | N x D x 8 |
| worker float32 matrices `X` and `Xs` (TSV in-memory path holds both while standardising) | 3.75 GiB each | N x D x 4 |
| per-fold training gather | 2/3 x N x D x 4 = 2.5 GiB | N x D |
| rescore tree peak (Rust + worker) | 33.04 GiB | ~ N x D |

So a feature count of 120 would, to first order, cut the stage peak, the PIN and the
features table by 3.2x. Whether it also cuts time, and what it costs in identifications,
are the two measured questions below. The objective is the quantity the engine optimises:
stripped peptides at 1% FDR, with the decoy fraction as the validity check.

## 2. Data and protocol

Two HYE runs from the current engine with the imported HYE library, `nn_torch` strict,
Extended features, `compete.group_by = peptidoform_charge`:

| tag | file | competed PSMs | targets | decoys |
|---|---|---|---|---|
| A01 | `LFQ_Orbitrap_AIF_Condition_A_Sample_Alpha_01` (August run, `~/hye/out_hye41/A_01/run.pin`) | 1,815,610 | 973,483 | 842,127 |
| B01 | `LFQ_Orbitrap_AIF_Condition_B_Sample_Alpha_01` (2026-09-04 profile run, `out_inst/run.pin`) | 2,603,894 | 1,364,317 | 1,239,577 |

The PINs are converted to float32 parquet (`~/hye/pin2parquet.py`). Everything below is
computed from the PIN alone; the features stage is not re-run. The features-stage PIN
(`run.pin`, SpecId `cand_<id>`) and the rescore-stage PIN the worker consumed
(`sidecar_work/*.pin`, SpecId `psm_<row>`) were verified row-aligned on B01: identical
labels, masses and peptides in the same order.

The objective is measured with `bench/feature_selection/fs_lib.py::run_rescoring`, a
re-implementation of `scripts/nn_rescore_worker.py` under the configuration the engine
ships: md5(stripped peptide) folds (3), per-fold init-feature scan over both signs,
Percolator-style self-training on targets at 1% vs all decoys for `num_iter = 10`
iterations (the `RescoreConfig` default, passed as `MUMDIA_NN_ITERS`; the worker's own
docstring default of 5 is never what the engine runs), a fresh MLP 128-64 with dropout 0.3
per iteration, churn early stop at 1% (never triggered here, as in the real run), mean/std
standardisation clipped at +-8, rank averaging over seeds. A feature subset is evaluated by
running that loop on the subset's columns only, so the init scan, the self-training
trajectory and the model all see the reduced set.

Faithfulness checks: on A01 the harness gives 59,046 peptides at 1% (seed 0), the number
the August notebook baseline recorded for the real worker on this PIN. On B01 the engine
run's own worker output, scored with the same picked-peptide metric, gives 58,978 peptides
and 66,081 target PSMs (the engine reported 66,081); the harness gives 59,246 and 59,773
for two seeds.

Runs were on an RTX 4090, several at a time; wall times are indicative only. The noise
floor is two seeds of the full set: 59,046 vs 59,611 on A01 (0.95% apart), 59,246 vs
59,773 on B01 (0.9%). On the 400k-row screening subsample three seeds of the full set
spanned 0.64%. Differences inside those bands are not differences.

## 3. What the 387 features are, structurally

Both datasets agree on every count below to within one feature, and the per-feature
univariate strength correlates at Spearman 0.971 across them: the structure is a property
of the feature definitions, not of one run.

### 3.1 Dead by construction: 43 columns

Ten columns are constant, for configuration reasons rather than data:

| feature | family | why |
|---|---|---|
| has_ms1 | rich | always 1 (MS1 present on every row) |
| ms1_isotope_height_corr | ms1 | `ms1_precursor_features` off (default) |
| deconv_explained_frac | demix | `emit_demix_features` off (default) |
| deconv_active | demix | `emit_demix_features` off (default) |
| deconv_share | demix | `emit_demix_features` off (default) |
| deconv_max_collinearity | demix | `emit_demix_features` off (default) |
| shadow_kept_frac | demix | `emit_demix_features` off (default) |
| peak_contested_frac | psm_extra | `emit_contested_features` off (default) |
| peak_contested_count_frac | psm_extra | `emit_contested_features` off (default) |
| peak_apportioned_frac | psm_extra | `emit_contested_features` off (default) |

Twenty columns are bit-identical to another column: the same quantity computed twice
under two names, once by a legacy set and once by an extended family.

| feature | family | identical to |
|---|---|---|
| rmsd_norm | similarity | library_rmsd |
| bhattacharyya_coef | similarity | spectrum_cosine_sqrt |
| cosine_high_ordinal | similarity | frag_cosine |
| xcorr_shape_mean | coelution | xcorr_shape |
| xcorr_lag_mean_abs | coelution | xcorr_coelution |
| apex_purity | interference | explained_apex_intensity_frac |
| matched_pred_intensity_fraction | interference | library_recall_intensity |
| top_pred_frag_matched | interference | top1_predicted_observed |
| intensity_weighted_abs_ppm | mass_accuracy | weighted_mass_error |
| log_ms1_mono | ms1 | log_mono_ms1 |
| ms1_isotope_xic_shape_consistency | ms1 | ms1_iso_coelution |
| rt_error_abs_norm_gradient | rt | rt_error_rel |
| precursor_charge | novel | charge |
| n_matched_frags | novel | n_matched_fragments |
| profile_cos_nz | nonzero | profile_cos |
| frag_mass_err_abs_median | mass_uncertainty | median_abs_frag_ppm |
| frag_mass_err_iqr | mass_uncertainty | ppm_iqr |
| frag_mass_err_max_abs | mass_uncertainty | max_abs_frag_ppm |
| frag_mass_err_range | mass_uncertainty | ppm_range |
| frac_top3_pred_observed | mass_uncertainty | frac_top3_predicted_observed |

Thirteen more are affine images of another column (|Pearson| >= 0.9999): a rescaling, a
sign flip, or a division by a per-run constant such as the gradient length. A standardised
linear layer cannot tell them from their source.

| feature | family | affine image of | also in B01 |
|---|---|---|---|
| rt_error_rel | minimal | rt_error_abs | yes |
| manhattan_sim | similarity | library_norm_manhattan | yes |
| squared_chord | similarity | spectrum_cosine_sqrt | yes |
| harmonic_mean_sim | similarity | chi_square_symmetric | yes |
| frac_predicted_absent | similarity | matched_fraction | no (0.9998) |
| jensen_shannon_divergence | entropy | spectral_entropy_similarity | yes |
| out_of_peak_intensity_frac | interference | peak_to_full_area_ratio_profile | yes |
| fwhm_to_window_ratio | chromatographic | fwhm_seconds | yes |
| rt_error_signed_norm_gradient | rt | rt_error_signed | yes |
| observed_rt_fraction | rt | observed_rt_raw | yes |
| predicted_rt_fraction | rt | predicted_rt_raw | yes |
| rt_diff_profile_apex | rt | apex_centering_offset | yes |
| log_total_matched_intensity | novel | log_apex_intensity | yes |

387 - 43 = 344 live features. Two of those are near-constant (`predicted_rt_in_gradient`,
`peak_window_degenerate`: modal value on 99.99% of rows).

### 3.2 Redundancy among the live 344

Spearman correlation on a 300k-row sample, average-linkage clusters at a |rho| threshold:

| \|rho\| >= | clusters (A01) | clusters (B01) |
|---|---|---|
| 0.98 | 302 | 304 |
| 0.95 | 274 | 286 |
| 0.90 | 239 | 254 |
| 0.80 | 189 | 191 |
| 0.70 | 154 | 162 |

615 pairs (A01; 570 on B01) have |correlation| >= 0.95. The largest cluster at 0.90 has 23
members and is one idea, intensity-profile similarity, computed 23 ways
(`spectrum_cosine_sqrt`, `spectral_angle_sqrt`, `library_norm_manhattan`, `bray_curtis`,
`hellinger`, `spectral_entropy_similarity` and its four variants, `jeffreys_divergence`,
`kl_pred_obs`, ...). Other 7-member clusters: the log-intensity family
(`log_apex_intensity`, `dot_product_raw`, `log_dot_product`, `total_xic_log`, ...), the
xcorr-shape family, the b-ion count family, the ppm-spread family.

PCA of the standardised live matrix: 48 components carry 80% of the variance, 86 carry
90%, 123 carry 95%, 194 carry 99%.

### 3.3 Univariate strength

The worker's own criterion (targets at 1% FDR using one feature alone, better sign) on the
full A01 pool: 277 of 387 features give zero targets at 1% on their own; the best give
41,116 (`spectral_entropy_similarity_topk`), 39,022 (`library_rmsd`), 38,915
(`minkowski_p3`), 38,807 (`bray_curtis_sqrt`). Every one of the top 30 is a spectrum
similarity or entropy measure. That is what the init scan picks (it chose
`spectral_entropy_similarity_topk` or `bray_curtis_sqrt` in every fold of the real run),
and it says nothing about what the multivariate model needs: `rt_error_abs`, the strongest
feature multivariately (3.4), gives zero targets alone. Subsets chosen by univariate
strength are the worst in section 5 (-13% to -19% at 25-100 features): they select 100
copies of the same signal.

### 3.4 Multivariate importance, and why it is a weak guide here

Importance is measured on the population the worker trains on: targets with out-of-fold
q <= 0.01 under the full model (68,054 on A01) against decoys (400k sampled). That problem
is nearly separable: an L1 logistic regression reaches holdout AUC 0.9996 with 31 features
and 0.9999 with 82, and gradient boosting reaches 0.9999 with all of them. Every importance
measure therefore works on a saturated objective; the values are tiny (the largest
permutation drop is 1.6e-4 AUC) and the rankings disagree: Spearman correlations between
permutation importance, linear leave-one-out drop, linear weight, L1 entry order and
univariate strength are between 0.01 and 0.51. Rankings are usable to build candidate
subsets and to pick a representative inside a cluster; they are not evidence that a subset
is sufficient. Only sections 5 and 6 are.

What the rankings agree on:

- `rt_error_abs` is the strongest single feature multivariately by a factor of 6 and is
  invisible univariately.
- Permutation importance mass is concentrated: the top 10 live features carry 68% of the
  total positive drop, the top 50 carry 91%, the top 100 carry 97.5%. 131 of the 344 live
  features have a permutation drop <= 0 on this saturated problem.
- 36 features enter the L1 path at C <= 0.001, 78 at C <= 0.005, 202 at 0.02; 12 never
  enter even at C = 0.5.

Per family (A01, live features only):

| family | features | live | best univariate | univariate zero | perm share | in perm top 100 | in L1 at C=0.005 |
|---|---|---|---|---|---|---|---|
| minimal | 14 | 13 | 27,207 | 10 | 0.414 | 7 | 4 |
| entropy | 18 | 17 | 41,116 | 7 | 0.122 | 5 | 3 |
| similarity | 63 | 56 | 39,022 | 24 | 0.111 | 19 | 17 |
| interference | 26 | 22 | 22,870 | 21 | 0.071 | 8 | 5 |
| rich | 30 | 29 | 39,022 | 18 | 0.054 | 9 | 1 |
| chromatographic | 43 | 42 | 0 | 43 | 0.045 | 8 | 5 |
| nonzero | 12 | 11 | 27,530 | 5 | 0.041 | 4 | 2 |
| coelution | 38 | 36 | 28,485 | 16 | 0.036 | 7 | 11 |
| ion_series | 34 | 34 | 25,464 | 27 | 0.026 | 7 | 7 |
| mass_accuracy | 17 | 16 | 0 | 17 | 0.021 | 4 | 2 |
| ms1 | 26 | 23 | 0 | 26 | 0.021 | 7 | 10 |
| mass_uncertainty | 10 | 5 | 22,523 | 8 | 0.012 | 3 | 2 |
| rt | 12 | 7 | 0 | 12 | 0.007 | 3 | 1 |
| novel | 10 | 7 | 24,026 | 9 | 0.007 | 2 | 3 |
| psm_extra | 6 | 3 | 0 | 6 | 0.005 | 2 | 2 |
| apex_dispersion | 13 | 13 | 0 | 13 | 0.004 | 3 | 3 |
| order_consistency | 8 | 8 | 0 | 8 | 0.002 | 1 | 0 |
| demix | 5 | 0 | 0 | 5 | 0 | 0 | 0 |
| peak_scans | 2 | 2 | 0 | 2 | 0 | 0 | 0 |

Top 25 by permutation importance (A01):

| feature | family | perm drop x 1e5 | L1 first C | univariate targets |
|---|---|---|---|---|
| rt_error_abs | minimal | 15.7 | 0.0005 | 0 |
| spectral_entropy_similarity_topk | entropy | 2.4 | 0.0005 | 41,116 |
| spectral_entropy_similarity_area | entropy | 2.1 | 0.0005 | 32,023 |
| peak_to_full_area_ratio_weighted | interference | 2.0 | 0.0005 | 0 |
| n_frag_present_inpeak | nonzero | 1.2 | 0.0005 | 26,055 |
| regression_slope | similarity | 1.0 | 0.5 | 0 |
| canberra_matched | similarity | 0.9 | 0.0005 | 0 |
| n_y_ions | rich | 0.8 | 0.001 | 0 |
| gaussian_cosine | chromatographic | 0.8 | 0.0005 | 0 |
| charge | minimal | 0.6 | 0.001 | 0 |
| profile_peak_snr | chromatographic | 0.5 | 0.002 | 0 |
| mae_weighted_pred | similarity | 0.5 | 0.0005 | 25,001 |
| mass_evidence_gauss | mass_accuracy | 0.4 | 0.0005 | 0 |
| diff_by_intensity | rich | 0.4 | 0.5 | 0 |
| frac_top5_pred_observed | mass_uncertainty | 0.4 | 0.0005 | 0 |
| seed_score | rich | 0.4 | 0.02 | 24,026 |
| pairwise_coelution_hi | coelution | 0.4 | 0.005 | 28,485 |
| peak_to_full_area_ratio_frag_mean | interference | 0.3 | 0.005 | 0 |
| series_gap_fraction | ion_series | 0.3 | 0.002 | 0 |
| wave_hedges | similarity | 0.3 | 0.01 | 23,164 |
| pairwise_coelution_weighted | coelution | 0.3 | 0.0005 | 14,045 |
| by_ion_contiguous_intensity | ion_series | 0.3 | 0.01 | 0 |
| lib_weighted_abs_ppm | mass_accuracy | 0.3 | 0.01 | 0 |
| ms1_isotope_corr_xic | ms1 | 0.3 | 0.0005 | 0 |
| coelution_corr_entropy | coelution | 0.2 | 0.0005 | 3,841 |

## 4. Candidate subsets

`bench/feature_selection/fs_subsets.py` builds, from the tables above:

- `dedup` (344): the live set;
- `clust_t` (302, 274, 239, 189, 154): one representative per Spearman cluster at |rho| >=
  0.98, 0.95, 0.90, 0.80, 0.70, the representative being the member with the largest
  permutation importance;
- `topK_perm` (10 to 200) by permutation importance; `topK_univ` (25, 50, 100) by
  univariate strength;
- `l1_C` (42, 86, 224): the non-zero set of the L1 path at C = 0.001, 0.005, 0.02;
- `no_<family>` and `only_<family>`: every family removed alone, every family alone;
- the engine's own `minimal` (14) and `rich` (44);
- second round, from the first round's results: unions of the permutation top-75/top-100
  with the L1 C=0.005 set (122, 138), intersections of cluster representatives with the
  permutation top-200 (160, 131), all six "cheap" families removed together (292), and the
  122-feature union with its 8 remaining duplicates removed (114).

## 5. Screening on a 400k-row subsample of A01

One seed per subset, 10 iterations; the full set was run with three seeds (mean 14,442
peptides, spread 0.64%) and the subsets marked * with three seeds. Relative to that mean:

| subset | D | peptides at 1% | rel. | note |
|---|---|---|---|---|
| all * | 387 | 14,442 | 0 | |
| dedup * | 344 | 14,339 | -0.7% | |
| clust_0.98 | 302 | 14,337 | -0.7% | |
| no_cheap_families * | 292 | 14,322 | -0.8% | |
| clust_0.95 | 274 | 14,287 | -1.1% | |
| clust_0.90 * | 239 | 14,316 | -0.9% | |
| l1_C0.02 | 224 | 14,217 | -1.6% | |
| top200_perm | 200 | 14,346 | -0.7% | |
| clust_0.80 * | 189 | 14,330 | -0.8% | |
| clust090_x_top200 * | 160 | 14,309 | -0.9% | |
| clust_0.70 | 154 | 14,283 | -1.1% | |
| top150_perm | 150 | 14,219 | -1.5% | |
| union_top100_l1005 * | 138 | 14,383 | -0.4% | |
| clust080_x_top200 * | 131 | 14,331 | -0.8% | |
| union_top75_l1005 * | 122 | 14,432 | -0.1% | best compact set |
| top100_perm * | 100 | 14,242 | -1.4% | |
| l1_C0.005 * | 86 | 14,312 | -0.9% | |
| top75_perm | 75 | 14,285 | -1.1% | |
| top50_perm | 50 | 13,974 | -3.2% | real loss begins |
| l1_C0.001 | 42 | 13,942 | -3.5% | |
| top25_perm | 25 | 13,631 | -5.6% | |
| top10_perm | 10 | 12,794 | -11.4% | |
| top100_univ | 100 | 12,603 | -12.7% | univariate ranking is the wrong criterion |
| top50_univ | 50 | 12,039 | -16.6% | |
| top25_univ | 25 | 11,717 | -18.9% | |

Every subset of 75 or more features that was chosen multivariately sits between -1.6% and
-0.1%, i.e. at or just below the noise floor; real losses start below 75. The consistent
-0.7% to -0.9% of the large subsets (dedup, cluster representatives) across three seeds each
is at the edge of the noise band and may be a small real effect: duplicated inputs act as an
implicit re-weighting of the first layer at initialisation. It does not reproduce on the
full pool (section 6, dedup +0.1%).

Family ablations, one seed each, relative to the same mean:

| removed family | D | rel. | removed family | D | rel. |
|---|---|---|---|---|---|
| rich | 315 | +0.7% | mass_accuracy | 328 | -0.9% |
| rt | 337 | +0.1% | ms1 | 321 | -1.0% |
| peak_scans | 342 | -0.1% | novel | 337 | -1.1% |
| mass_uncertainty | 339 | -0.4% | interference | 322 | -1.2% |
| order_consistency | 336 | -0.4% | entropy | 327 | -1.3% |
| ion_series | 310 | -0.5% | apex_dispersion | 331 | -1.4% |
| similarity | 288 | -0.5% | chromatographic | 302 | -1.9% |
| minimal | 331 | -0.6% | coelution | 308 | -2.1% |
| psm_extra | 341 | -0.6% | demix | 344 | -0.9% (= dedup) |
| nonzero | 333 | -0.8% | | | |

No family is indispensable: removing the 63-member similarity family costs 0.5%, removing
the 14 minimal features that carry 41% of the permutation importance costs 0.6%, because
in each case the others cover it. The two elution-shape families, chromatographic and
coelution, are the most expensive to lose and are the least redundant with the rest. Every
family alone is far worse than the full set (best: rich alone -13.8%, similarity alone
-17.3%; mass_accuracy, ms1, order_consistency, peak_scans, rt, psm_extra, apex_dispersion
alone give zero peptides at 1%, because without a spectrum-similarity or RT-error feature
the init scan has nothing to bootstrap from).

## 6. Confirmation on the full pools

### 6.1 A01, full pool (1,815,610 PSMs), 10 iterations, two seeds each

| subset | D | peptides at 1% (mean of 2) | min / max | PSMs at 1% | decoy frac | rel. |
|---|---|---|---|---|---|---|
| all | 387 | 59,328 | 59,046 / 59,611 | 70,490 | 0.0100 | 0 |
| dedup | 344 | 59,370 | 59,253 / 59,488 | 70,534 | 0.0100 | +0.1% |
| top150_perm | 150 | 59,398 | 59,375 / 59,421 | 70,632 | 0.0100 | +0.1% |
| clust_0.90 | 239 | 59,122 | 59,003 / 59,241 | 70,250 | 0.0100 | -0.4% |
| top100_perm | 100 | 59,113 | 59,050 / 59,177 | 70,016 | 0.0100 | -0.4% |
| union_top100_l1005 | 138 | 59,332 | 59,322 / 59,342 | 70,438 | 0.0100 | 0.0% |
| union_top75_l1005 | 122 | 59,230 | 59,057 / 59,403 | 70,313 | 0.0100 | -0.2% |
| union75_dedup | 114 | 59,159 | 59,087 / 59,231 | 70,193 | 0.0100 | -0.3% |
| l1_C0.005 | 86 | 58,890 | 58,862 / 58,918 | 70,022 | 0.0100 | -0.7% |
| clust_0.80 | 189 | 58,817 | 58,787 / 58,847 | 69,935 | 0.0100 | -0.9% |
| rich | 44 | 54,889 | 54,865 / 54,913 | 64,802 | 0.0100 | -7.5% |
| minimal | 14 | 46,039 | 46,019 / 46,059 | 54,019 | 0.0100 | -22.4% |

Every multivariately chosen subset of 100 or more features is inside the seed band
(59,046 to 59,611); the three union sets (138, 122, 114) sit at 0.0%, -0.2% and -0.3%.
The 86-feature L1 set and, oddly, the 189 cluster representatives at 0.80 are the first to
show a consistent loss, both seeds below every seed of the full set. The two large sets
(dedup, top150) are marginally above the full set, so the -0.7% they showed on the 400k
subsample was subsample noise, not a redundancy effect.

### 6.2 B01, full pool (2,603,894 PSMs), 10 iterations, two seeds each

B01 is the held-out set: every subset was built from A01 tables only.

| subset | D | peptides at 1% (mean of 2) | min / max | PSMs at 1% | rel. |
|---|---|---|---|---|---|
| all | 387 | 59,510 | 59,246 / 59,773 | 66,769 | 0 |
| dedup | 344 | 59,470 | 59,430 / 59,509 | 66,725 | -0.1% |
| top150_perm | 150 | 59,547 | 59,471 / 59,622 | 66,895 | +0.1% |
| clust_0.90 | 239 | 58,790 | 58,616 / 58,964 | 65,921 | -1.2% |
| clust_0.80 | 189 | 58,707 | 58,697 / 58,716 | 65,750 | -1.4% |
| union_top100_l1005 | 138 | 59,447 | 59,318 / 59,575 | 66,762 | -0.1% |
| union_top75_l1005 | 122 | 59,399 | 59,211 / 59,586 | 66,600 | -0.2% |
| union75_dedup | 114 | 59,306 | 59,170 / 59,441 | 66,438 | -0.3% |
| top100_perm | 100 | 58,990 | 58,827 / 59,152 | 66,210 | -0.9% |
| l1_C0.005 | 86 | 58,552 | 58,461 / 58,643 | 65,661 | -1.6% |
| rich | 44 | 54,566 | 54,536 / 54,595 | 60,827 | -8.3% |
| minimal | 14 | 42,672 | 42,351 / 42,992 | 47,481 | -28.3% |

Held out, the picture sharpens. Dropping the 43 dead columns costs nothing (-0.1%), and
the 150 most important features by permutation still match the full set (+0.1%, both seeds
inside the band). Below that the losses that were at the noise edge on A01 become
consistent: 100 features -0.9%, 86 features -1.6%, with both seeds below every seed of the
full set. The cluster-representative sets transfer worst of all: 239 representatives at
|rho| >= 0.90 lose 1.2% here against 0.4% on A01, i.e. choosing one member per correlation
cluster by A01 importance picks representatives that are not the best member on B01,
whereas an importance ranking degrades gracefully. The three union sets transfer: 138,
122 and 114 features at -0.1%, -0.2% and -0.3%, each with at least one seed inside the
full set's band and the other within 0.15% of it. Decoy fraction 1.00% throughout.

### 6.3 The 114-feature deduplicated union

The best compact set from screening is the union of the permutation top 75 with the L1
C = 0.005 set (122 features; 39 in both, 36 only in the first, 47 only in the second),
minus the 8 members that section 3.1 shows to be duplicates of other members
(`rt_error_rel`, `rt_error_abs_norm_gradient`, `precursor_charge`,
`log_total_matched_intensity`, `fwhm_to_window_ratio`, `predicted_rt_fraction`,
`frac_top3_pred_observed`, `ms1_isotope_xic_shape_consistency`). It draws on 16 of the 19
families (not demix, order_consistency, peak_scans):

- minimal (6): charge, coelution_best, coelution_run, log_apex_intensity, peptide_length, rt_error_abs
- rich (7): diff_by_intensity, evidence, ms1_isom1_ratio, n_y_ions, seed_score, sum_y_intensity, xcorr_shape
- similarity (24): abs_diff_q3, bray_curtis, bray_curtis_sqrt, canberra_matched, chi_square_pearson, cosine_fullwindow, dot_product_norm, dot_product_raw, footrule_norm, frac_top3_predicted_observed, intensity_weighted_pearson, log_dot_product, mae_norm, mae_weighted_pred, max_positive_residual, regression_slope, scribe_score_area, spectral_angle_area, spectral_log_evidence, spectral_log_evidence_area, spectrum_cosine_log, stein_scott_weighted_dot, wasserstein_mz, wave_hedges
- entropy (6): cross_entropy_obs_pred, obs_spectrum_entropy, pred_spectrum_entropy, spectral_entropy_similarity_area, spectral_entropy_similarity_topk, weighted_spectral_entropy_similarity
- coelution (13): coelution_corr_entropy, coelution_hi_lo_contrast, frac_frag_ref_corr_above_0_8, frag_loo_ref_corr_min, frag_ref_corr_obsweighted, frag_ref_corr_std, full_vs_peak_corr_gain, observed_sum_vs_template_corr, pairwise_coelution_hi, pairwise_coelution_weighted, ref_xcorr_lag_mean, top3_frag_ref_corr, xcorr_lag_frac_zero
- interference (8): corrected_vs_raw_cos, explained_apex_intensity_frac, explained_variance_ref, ifs_removed_count, ifs_removed_intensity_frac, peak_to_full_area_ratio_frag_mean, peak_to_full_area_ratio_weighted, second_component_fraction
- chromatographic (7): frag_apex_rt_dispersion, frag_zigzag_mean, fwhm_seconds, gaussian_cosine, profile_peak_snr, roughness_2nd_deriv, width_at_10pct
- ion_series (10): by_complement_coelution, by_intensity_ratio, by_ion_contiguous_intensity, cosine_charge2, frac_matched_b, mean_matched_ordinal_norm, ordinal_intensity_concordance_y, series_gap_fraction, spectral_angle_b, spectral_angle_y
- ms1 (12): has_ms1_signal, iso_minus_one_fraction, iso_overlap_flag, ms1_iso_coelution, ms1_iso_ratio_stability, ms1_isotope_apex_entropy_3, ms1_isotope_corr_xic, ms1_isotope_cosine_apex, ms1_isotope_spectral_angle_apex, ms1_ms2_apex_rt_delta, ms1_ms2_fwhm_ratio, ms1_ms2_time_corr
- rt (3): observed_rt_raw, predicted_rt_raw, rt_error_over_peak_width
- novel (4): charge_is_4plus, log_seed_hyperscore, n_predicted_frags, precursor_mass
- nonzero (4): frac_frag_present_inpeak, frag_corr_matched_nz, frag_cosine_matched_nz, n_frag_present_inpeak
- mass_accuracy (3): high_ppm_intensity_frac, lib_weighted_abs_ppm, mass_evidence_gauss
- mass_uncertainty (2): effective_frag_count, frac_top5_pred_observed
- apex_dispersion (3): frag_apex_rt_std, peak_fwhm_scans, peak_shoulder_score
- psm_extra (2): cross_charge_intensity_log, n_charge_states

The list is `C:/Users/robbi/mumdia_bench/fs/fs_union75_dedup.txt`, one name per line, in
PIN column order, i.e. directly usable as `MUMDIA_NN_FEATURES`.

### 6.4 The real worker on B01 with the 122-feature union

`scripts/nn_rescore_worker.py` itself, on `out_inst/run.pin` (TSV, in-memory backend, 32
threads, `MUMDIA_NN_FOLDS=3 MUMDIA_NN_ITERS=10 MUMDIA_NN_TRAIN_FDR=0.01` as `rescore.rs`
passes them) with `MUMDIA_NN_FEATURES` naming the 122-feature union, against the engine
run's own worker output on the same PIN, both scored with the same metric:

| | all 387 (engine run) | 122-feature union | change |
|---|---|---|---|
| target PSMs at 1% | 66,081 | 66,174 | +0.1% |
| decoys at 1% / decoy fraction | 659 / 1.00% | 660 / 1.00% | |
| peptides at 1% | 58,978 | 58,962 | -0.03% |
| worker wall (measured phases) | 3,062.8 s | 3,009.1 s | -1.8% |
| of which train | 2,752.5 s | 2,878.8 s | +4.6% |
| of which PIN read + standardise | 192.6 s | 80.3 s | -58% |
| of which init feature scan | 54.2 s | 17.3 s | -68% |
| worker max RSS | about 29 GiB (33.04 GiB stage peak minus the Rust share) | 24.0 GiB | about -17% |

The identifications are the same to three decimals. The time is the same: training, 96%
of the worker, did not get faster with a third of the columns (section 7), and the only
phases that did are the ones that move the data. The memory dropped by a sixth, not by two
thirds, because on the TSV in-memory path the worker still parses all 394 columns into a
float64 frame before it selects (section 8).

## 7. Time: feature count does not set the training cost

`~/hye/memprof/nn_cpu_scaling.py` trains the worker's exact MLP (128-64, BatchNorm,
dropout 0.3, Adam, batch 4096, BCE) for one epoch over 400,000 rows of the B01 PIN at
several widths, CPU only, on an otherwise idle doxy:

| D | 32 threads, us per row per epoch | 16 threads |
|---|---|---|
| 387 | 4.11 | 4.55 |
| 344 | 4.65 | 4.24 |
| 300 | 4.22 | 4.37 |
| 240 | 4.55 | 4.26 |
| 200 | 4.08 | 4.42 |
| 150 | 4.32 | 4.33 |
| 120 | 4.20 | 3.70 |
| 100 | 4.25 | 3.84 |
| 75 | | 3.96 |
| 50 | | 3.91 |
| 25 | | 3.79 |

Flat, with run-to-run noise larger than any trend. At batch 4096 the per-batch fixed cost
(kernel launches, thread synchronisation on small GEMMs, BatchNorm, the optimiser step)
dominates, and the first layer's D x 128 GEMM, which is 86% of the arithmetic, is not
what the CPU spends its time on. A 122-feature set will therefore not make the 53-minute
rescore of the HYE run faster by any useful amount; the August measurements
(`rescoring-speed-bench`: warm start 2.4x, batch 16384 2.3x, GPU 17x) are where the time
is. What selection removes from the time budget is the PIN write and parse (3% of the
worker's wall clock at full width) and the standardisation pass.

The real worker agrees (section 6.4): 2,879 s of training on 122 features against 2,753 s
on 387, on the same machine, the same PIN and the same iteration count. The same flatness
holds on the GPU in the harness (train time 97 s at 387 vs 80 s at 14 features on A01, 5
iterations), for the same reason.

## 8. Where a selection could be applied

Three places, increasing in scope and in what they save:

1. Worker only. `MUMDIA_NN_FEATURES=<file with one name per line>` exists in
   `scripts/nn_rescore_worker.py`. On the parquet handoff and on the streaming TSV path the
   listed columns are the only ones parsed, standardised and moved. On the default in-memory
   TSV path the whole PIN is still read with `pandas.read_csv` (all 394 columns, float64),
   and only the float32 matrices and the training shrink; the largest Python allocation
   does not. No engine change, no schema change.
2. Rescore projection. Read only the selected columns from the competed table into
   `FeatureMatrix` and write a narrower PIN. Removes the Rust matrix and the PIN size too.
   Needs a config field naming the subset, and `psms_scored.parquet.report.json` must record
   the classifier's actual feature list (it is the source of truth for what was used).
3. Features stage set. A new `FeatureSet` variant computing only the selected features:
   `features.parquet` (5.5 GB here), compete, rescore and the PIN all shrink, and families
   that are dropped entirely stop being computed. A schema change (`feature_schema_id`) and
   the largest surface; the least attractive, because feature computation is now 7 minutes
   and 3 GiB (docs/27) and the 114-feature set still touches 16 of 19 families.

Independent of feature selection, and larger: `rescore.handoff = parquet` already exists
(`Handoff::Parquet`, documented from an 8.9M-PSM experiment-wide rescore: 671.6 min to 12
min because the TSV exceeded the worker's streaming threshold). It replaces the pandas
float64 parse with a columnar read of only the requested columns, so it is both the biggest
single memory saving in this stage and the thing that makes option 1 save memory at all.
It is nn_torch only; mokapot falls back to TSV.

None of these change FDR mechanics: target-decoy competition and q-values are computed
from scores exactly as now, so the decoy fraction at 1% stays 1% by construction (it did
in all 100+ runs here). What a reduced set changes is sensitivity, which section 6
measures, and the risk of the self-training loop overfitting, which fewer features can
only reduce.

## 10. What was implemented (2026-09-05)

Three levers, all default-off until section 15 says otherwise, so an unconfigured run behaves
exactly as before:

| lever | where | what it does |
|---|---|---|
| `rescore.handoff = parquet` | already existed, `rescore.rs` | the sidecar reads a columnar feature table instead of a 9.5 GB text PIN |
| `rescore.features` / `features_file` | new, `rescore.rs::resolve_feature_subset` | projects the classifier's input columns in schema order; the matrix, the handoff and the training all shrink with the list |
| `rescore.train_neg_ratio` / `train_neg_select` / `train_subsample` / `train_warm_epochs` | new, plumbed to `MUMDIA_NN_*` | cap and choose the decoys the sidecar trains on, and warm-start the model between self-training iterations |

`train_neg_select` adds two strategies to the worker's existing random cap:

- **margin**: keep the highest-scoring decoys under the current model, i.e. the only part of
  the decoy distribution still competing with accepted targets;
- **hybrid**: half the budget from the margin, half sampled at random from the rest, so the
  boundary is informed by hard cases without losing the shape of the bulk.

None of this touches FDR. Selection, scoring, target-decoy competition and q-values still run
over the full pool with every decoy; only the rows the optimiser sees change. The decoy
fraction at 1% was 0.94-1.00% in all 150+ runs below, as that design guarantees.

## 11. The sidecar handoff is the largest memory lever

Same competed table (2,603,894 PSMs x 387 features), one self-training iteration per arm so the
comparison is the data path rather than the training, 32 threads, sequential:

| | TSV (shipped default) | Parquet |
|---|---|---|
| rescore peak RSS (Rust + worker) | 29.96 GB | **8.95 GB** |
| wall | 8:34.9 | **6:33.2** |
| sidecar file on disk | 9.53 GB | 3.28 GB |
| worker PIN read + standardise | 111.7 s (26.4% of the worker) | 16.9 s (4.8%) |
| peptides at 1% | 47,762 | 47,752 |
| decoy fraction | 1.00% | 1.00% |

A 70% memory cut for a config field that already existed. The mechanism is in section 8: on the
TSV path the worker parses all 394 columns into a float64 pandas frame before it builds its
float32 matrix, so the text file, the frame and the matrix are alive together. Parquet reads
only the requested columns, already typed.

The 10-peptide difference (0.02%) is the NN's own nondeterminism, not the data path: the TSV is
lossy at `{:.6}` while parquet carries f32, so the two arms see marginally different inputs and
the self-training trajectory diverges. Score correlation between the arms is 0.63 while the
accepted sets differ by 0.02%, which is the usual picture for this model.

## 12. Training-set reduction: it pays in proportion to the imbalance

The worker refits 30 times (3 folds x 10 iterations, 25 epochs) on the targets currently at 1%
plus **every** decoy in the fold. How lopsided that is varies enormously by dataset, and that
single number predicts everything below:

| pool | PSMs | targets passing at 1% | decoys per positive in training |
|---|---|---|---|
| HYE A01 | 1,815,610 | 7.2% | ~12-18 : 1 |
| AIF E. coli | 89,531 | 22.7% | ~3 : 1 |
| entrapment (E. coli + 1:1 human) | 129,692 | 3.1% | ~22 : 1 |

Peptides at 1%, two seeds per cell, against each pool's own baseline:

| recipe | HYE A01 | AIF | entrapment | training speed-up (HYE / entrap) |
|---|---|---|---|---|
| baseline (all decoys, fresh fit) | 59,615 | 10,523 | 2,831 | 1.0x |
| neg5 hybrid | **+0.61%** | 0.00% (never binds) | -0.81% | 2.2x / 6.0x |
| warm5 | -0.61% | -0.18% | +2.16% | 3.4x / 3.0x |
| warm5 + neg5 hybrid | +0.41% | 0.00% | +3.99% | 7.0x / 12x |
| **warm5 + neg3 hybrid** | **+0.97%** | -0.18% | **+4.20%** | **8.9x / 16.4x** |
| neg1 margin | +1.27% | -0.17% | **-10.35%** | 5.2x / 20x |

Three findings:

1. **Which decoys you keep matters more than how many.** At a matched budget, margin and hybrid
   selection beat random by 0.5-2% on HYE, and land above the full-decoy baseline; random
   thinning degrades monotonically (-0.3% at 3:1, -1.4% at 1:1). The easy decoy bulk is not
   informative after the first iteration, but the model still needs *some* of it: pure margin
   at 1:1 is the worst recipe tested.
2. **A cap is self-limiting, a quota is not.** `neg5` never binds on AIF (identical rows per
   fit, identical output) and binds hard on HYE. That is the property to ship: it does nothing
   where there is nothing to gain.
3. **Too aggressive fails, and only the third dataset showed it.** `neg1_margin` was the best
   recipe on HYE (+2.67% at seed 0) and the fastest anywhere, and it loses 10.3% on the
   entrapment pool. One benchmark would have shipped it.

Seed discipline matters here: on HYE, seed 0 of the baseline scored 59,046 while seeds 1 and 2
scored 59,611 and 59,619. Every headline number above is a two-seed mean against a two-seed
baseline; the single-seed grid that first suggested +1.6 to +2.7% overstated the gains.

## 13. The empirical null: entrapment

Target-decoy q cannot detect a model that has learned to separate targets from *decoys* rather
than true from false, which is exactly the risk when training only on hard decoys. The E. coli
AIF file searched against an E. coli + 1:1 human entrapment library gives that null: the human
peptides are absent from the vial, so any that pass at 1% are false by construction. 41,041 of
67,304 target rows are spike-ins.

FDP is the engine's own estimator (`fdr.rs::entrapment_q`, `(ratio * entrapment + 1) / real`,
ratio 0.560632), at the peptide level, two seeds:

| recipe | peptides | decoy fraction | spike-in peptides | FDP |
|---|---|---|---|---|
| baseline | 2,831 | 0.97% | 26.5 | 0.57% |
| neg5 hybrid | 2,808 | 0.95% | 22.0 | 0.48% |
| warm5 | 2,892 | 0.96% | 26.5 | 0.55% |
| warm5 + neg5 hybrid | 2,944 | 0.96% | 30.5 | 0.62% |
| warm5 + neg3 hybrid | 2,950 | 0.95% | 28.5 | 0.58% |
| neg1 margin | 2,538 | 0.95% | 24.0 | 0.57% |

FDP moves between 0.48% and 0.62% around a 0.57% baseline, on counts of 22-30 spike-in peptides
where Poisson noise alone is +-0.1 percentage points. There is no evidence that hard-negative
training buys identifications by exploiting decoy structure: the extra peptides are as real as
the baseline's.

## 14. Full scale, through the engine

The same HYE competed table the audit profiled (2,603,894 PSMs x 387 features), rescored by
`mumdia rescore` on 32 threads, one arm per configuration, sequential:

| arm | peptides at 1% | PSMs at 1% | peak RSS | wall |
|---|---|---|---|---|
| shipped: TSV, 387 features, every decoy, fresh fits | 59,515 | 66,081 | 33.04 GiB | 53:07 |
| `full`: parquet + 114 features + 5:1 hybrid cap | 59,285 (-0.39%) | 65,824 | **5.51 GB** | **13:06** |
| `fast`: + warm start 5, 3:1 hybrid cap | 59,124 (-0.66%) | 65,672 | **5.49 GB** | **3:19** |

**6x less memory and 16x less time, at -0.66% peptides**, which is inside this pool's 0.9%
seed band. The sidecar handoff shrank from 9.53 GB of text to 1.1 GB of parquet.

Both arms recorded their provenance: `psms_scored.parquet.report.json` carries
`n_features_used: 114`, `n_features_available: 387`, the `feature_selection_id`, the full
feature list, and the training knobs actually used.

Note that rescore is no longer the stage that sets the run's peak. With extract at 28.13 GiB
(docs/27 section 3.10) and features at 3.03 GiB, a single run's ceiling moves back to extract.

## 15. Recommended configuration

Shipped as the default (the second and third items since the end of 2026-09-05; section 21 has
the decision record):

- **`rescore.handoff = parquet`.** 70% less memory, 24% less wall, identical identifications,
  no configuration required, and mokapot/entrapment sidecars keep receiving the TSV they need.
- **The training recipe: `train_neg_ratio: 3`, `train_neg_select: hybrid`,
  `train_warm_epochs: 5`.** Against the previous training (every decoy, cold refits), with
  seeds: HYE A01 +0.97%, HYE B01 +2.19%, AIF -0.09%, entrapment +3.33% at an unchanged
  spike-in FDP, for 9-19x less training time. `train_neg_ratio: 0, train_neg_select: random,
  train_warm_epochs: 0` restore the previous training exactly.
- **Every feature (`feature_preset: all`).**

Opt-in, each for a stated reason:

**Compact features** (`"feature_preset": "compact"`, the embedded 114-name list of section 12;
`features_file` still accepts any list). A memory lever: the rescore matrix is 3.4x smaller
(full-scale HYE rescore 5.49 GB / 3:19 against 13.5 GB / 6:20 with every feature; six pooled
HYE runs 15.9 GB where every feature would need roughly 30 GB). Under the default training it measured +0.2% (A01), -1.2% (B01, the pool
it was never fitted on), -0.1% (AIF), +1.5% (entrapment), so it is for pooled rescoring on
machines where the matrix does not fit.

**Sensitivity** (about 5x the default's rescore wall; +0.4 / +0.4 / +0.6 pp over the default
on HYE A01 / AIF / entrapment, spike-in FDP unchanged; +0.2% on HYE B01 through the engine at
18:38 against 3:31):

```json
"rescore": {
  "folds": 5,
  "train_margin_frac": 0.75,
  "seeds": 3
}
```

Evidence per pool, peptides at 1% against that pool's own 387-feature all-decoy baseline, two
seeds each unless marked:

| pool | 114 features alone | training recipe alone | both |
|---|---|---|---|
| HYE A01 | -0.3% | +0.97% | +1.16% |
| HYE B01 | -0.3% | +2.19% | +0.98% |
| AIF E. coli | +0.04% | -0.09% | -0.18% |
| entrapment | -2.14% | +3.33% | +4.87% (FDP 0.57% vs 0.56%) |

The conservative variant, for a caller who wants the memory but not a changed training
trajectory, is the compact preset plus `train_neg_ratio: 5, train_neg_select: hybrid`: the 5:1
cap never binds on a balanced pool (identical rows and identical output on AIF) and still gives
2.2x on HYE, for the `full` arm's 13:06 and 5.51 GB.

What NOT to do:

- `train_neg_select: margin` at `train_neg_ratio: 1`. Fastest of everything tested (20x) and
  +1.27% on HYE, and it loses **10.35%** on the entrapment pool. Hard negatives sharpen the
  boundary; they cannot be the only thing the model sees, and how much bulk a pool can spare
  depends on how imbalanced it is.
- `train_warm_epochs` on its own. -0.61% on HYE and -0.18% on AIF with seeds; it only pays in
  combination with a cap (+0.97% at 3:1 hybrid).
- Reading a single-seed sweep as a result. Seed 0 of the HYE baseline scored 59,046 against
  59,611 and 59,619 for seeds 1 and 2, which turned a +0.6% recipe into an apparent +1.6% one.

## 16. Caveats specific to the 2026-09-05 work

- The 114-feature list was selected on a DIA-NN-library search and transfers well to another
  such search (AIF, +0.04%) but costs 2.1% on the FASTA-built entrapment library, whose feature
  distributions differ. A production selection should be re-derived per library type, or the
  list should be treated as tuned for imported-library workflows.
- The entrapment FDP rests on 22-30 spike-in peptides per arm. It is sensitive enough to exclude
  a gross artifact (and did exclude `neg1_margin` on sensitivity grounds), but it cannot resolve
  FDP differences below about 0.1 percentage points.
- Every training-reduction number is from the harness except the two full-scale arms; the
  harness reproduces the real worker to within 0.03% (section 6.4) but is not the worker.
- Nothing here was measured on a third instrument type. Both HYE and AIF are Orbitrap AIF
  acquisitions, and the entrapment pool is the AIF file again with a different library.

## 17. The rescorer itself, swept with seeds (2026-09-05)

Training at 3-20 s per fit instead of 165 s made a systematic sweep of the classifier
affordable for the first time. Every arm is the fast recipe (3:1 hybrid cap, warm start 5)
with one knob changed, two seeds per arm (three on the combination grid), on three pools;
`recipe` is the reference each pool is read against, and `baseline` is the configuration
shipped before any of this work. Peptides at 1%. The B01 column (seeds 1 and 2, baseline
59,213) landed later: its baseline training took 2,924-3,065 s against 156 s for the recipe,
the recipe is +2.2% there, and the B01 ranking agrees with A01 (iterations 20, folds 5 and
seeds 3 above the recipe; iterations 6 below it) while still disagreeing with AIF and the
entrapment pool on iterations 20, which is why none of them moved.

| arm | HYE A01 | HYE B01 | AIF | entrapment (FDP) |
|---|---|---|---|---|
| baseline (all decoys, fresh fits, 128-64) | 59,615 | 59,213 | 10,512 | 2,837 (0.49%) |
| **recipe** | **60,192 (+0.97%)** | **60,510 (+2.19%)** | 10,502 (-0.09%) | 2,932 (+3.33%, 0.60%) |
| hidden 256-128 | +0.81% | +2.35% | -0.15% | +2.50% |
| hidden 256-128-64 | +0.66% | +1.81% | -0.42% | +4.34% |
| hidden 512-256 | +0.53% | +2.21% | -0.53% | +0.09% |
| hidden 64-32 | +0.50% | +1.06% | +0.01% | -1.67% |
| dropout 0.1 | +0.41% | +1.90% | 0.00% | +1.99% |
| dropout 0.5 | +0.26% | +1.55% | -0.11% | +3.05% |
| epochs 50 / warm 10 | +0.42% | +2.55% | -0.63% | +2.03% |
| epochs 12 / warm 3 | -0.16% | +1.18% | -0.08% | -0.23% |
| lr 3e-4 | -0.02% | +1.35% | -0.03% | **-7.10%** |
| lr 3e-3 | +0.45% | +1.70% | -0.20% | +2.86% |
| batch 1024 | +0.90% | +2.35% | -0.17% | +1.50% |
| batch 16384, lr 2e-3 | +0.28% | +1.14% | -0.54% | +1.15% |
| folds 5 | +0.55% | +2.72% | +0.04% | +3.33% |
| seeds 3 (rank ensemble) | **+1.27%** | +2.68% | +0.10% | +3.24% |
| iterations 20 | **+1.42%** | **+3.12%** | -0.09% | +1.64% |
| iterations 6 | -1.62% | **-0.54%** | +0.02% | **-4.65%** |
| train FDR 0.02 | +0.80% | +2.58% | -0.65% | +3.95% |
| train FDR 0.005 | +0.04% | +1.68% | -0.01% | **-5.09%** |
| margin fraction 0.75 | +0.63% | +2.19% | -0.08% | +3.70% |
| margin fraction 0.25 | +0.77% | +2.36% | -0.10% | +2.94% |
| weight decay 1e-3 | -0.31% | +1.09% | -0.20% | +3.84% |
| weight decay 0 | +0.42% | +1.83% | 0.00% | +3.40% |

Read across the pools, nothing beats the recipe consistently:

- **Capacity is not the lever.** Wider and deeper networks are monotonically worse on A01
  and AIF (512-256: -0.5% on both); the shipped 128-64 is at the optimum. 64-32 is as good
  as 128-64 on AIF at half the cost and loses 1.7% on the entrapment pool and 0.5% on A01.
- **The training schedule is at its optimum too.** 25 epochs, lr 1e-3, dropout 0.3, batch
  4096, weight decay 1e-4 are each the best or within noise of the best of their column;
  every departure that helps one pool costs another (lr 3e-4 loses 7.1% on entrapment;
  batch 16k loses 0.5% on AIF; 50 epochs loses 0.6% on AIF).
- **Fewer iterations is not free, more is not consistently better.** 6 iterations loses
  1.6% (A01) and 4.7% (entrapment); 20 gains 0.45 pp on A01, nothing on AIF and loses 1.7 pp
  on entrapment against the recipe. The 10 the engine ships is right.
- **The train-FDR threshold is asymmetric and pool-dependent**: 0.005 loses 5.1% on
  entrapment (too few positives to learn from), 0.02 gains there and loses 0.65% on AIF.
  0.01 stays.
- **Two knobs are neutral-to-positive everywhere**, and small: a 3-seed rank ensemble
  (+0.3 pp on A01, +0.2 pp on AIF, -0.1 pp on entrapment, at 3x the training) and 5 folds
  (-0.4 pp, +0.1 pp, 0.0 pp, at 1.7x). Combined with a 0.75 margin fraction, three seeds
  each on AIF: folds 5 + margin 0.75 + seeds 3 is +0.22% against the 387-feature baseline
  where the recipe alone is -0.23%; the same combination on the entrapment pool is in
  section 17.1.

The spike-in FDP across the whole grid on the entrapment pool is 0.49-0.64% against a 0.49%
baseline (the recipe's 0.60% is 29 spike-in peptides against 23; across the four seeds run
on this pool the baseline averages 0.53% and the recipe 0.59%, a difference well inside the
Poisson noise of counts that size). No arm learns decoy structure.

### 17.1 Combination grid, three seeds

Entrapment pool, seeds 3-5, against the 387-feature all-decoy baseline of those seeds
(2,804 peptides, FDP 0.56%):

| arm | peptides | vs baseline | spike-in peptides | FDP | training |
|---|---|---|---|---|---|
| recipe | 2,940 | +4.87% | 30.7 | 0.62% | 8.1 s |
| recipe + folds 5 | 2,908 | +3.72% | 26.0 | 0.54% | 16.0 s |
| recipe + margin 0.75 | 2,910 | +3.80% | 30.0 | 0.62% | 8.2 s |
| recipe + folds 5 + margin 0.75 | 2,924 | +4.28% | 26.3 | 0.54% | 15.1 s |
| recipe + folds 5 + seeds 3 | 2,929 | +4.46% | 26.0 | 0.53% | 43.2 s |
| recipe + folds 5 + margin 0.75 + seeds 3 | 2,958 | +5.49% | 28.0 | 0.57% | 44.9 s |

AIF, the same seeds (baseline 10,523): recipe -0.23%, folds 5 +0.01%, margin 0.75 -0.19%,
folds 5 + margin 0.75 +0.14%, folds 5 + seeds 3 +0.16%, all three +0.22%.

HYE A01, seeds 3-5 (baseline 59,421):

| arm | peptides | vs baseline | training |
|---|---|---|---|
| recipe | 60,109 | +1.16% | 18.0 s |
| recipe + folds 5 | 60,136 | +1.20% | 35.7 s |
| recipe + margin 0.75 | 60,017 | +1.00% | 18.0 s |
| recipe + folds 5 + margin 0.75 | 60,193 | +1.30% | 35.6 s |
| recipe + folds 5 + seeds 3 | 60,317 | +1.51% | 107.4 s |
| **recipe + folds 5 + margin 0.75 + seeds 3** | **60,352** | **+1.57%** | 107.2 s |

The full combination is the best arm on all three pools: +1.57% (A01), +0.22% (AIF), +5.49%
(entrapment, FDP 0.57% against the baseline's 0.56%), against the shipped baseline, and its
training (107 s on A01) is still 1.5x faster than the baseline's 164 s. That makes two
recipes worth naming rather than one; section 15 has both.

Conclusion: individually, no knob beats the recipe consistently; together, 5 folds, a 0.75
margin fraction and a 3-seed rank ensemble do, on every pool, for about 6x the recipe's
training time, which is still less than the shipped baseline's. Both recipes are in section
15; `rescore.seeds` and `rescore.train_margin_frac` were added so the second one is
configurable rather than an environment recipe.

## 18. Pipeline factors, end to end (2026-09-05)

With rescoring at ~3 minutes, a complete HYE run costs about 16 minutes, so the extraction
and RT parameters that docs/20 lists as benchmark-gated could be tested on the engine's own
objective rather than on proxies. One factor at a time from the current defaults, fast
rescore recipe throughout, HYE B01, one run each (this pool's seed band is 0.9%):

| arm | peptides at 1% | extracted rows | whole-run wall | whole-run peak |
|---|---|---|---|---|
| **current** (`gate_min_score` 0.2, then named `min_frag_corr`; `rt_window_multiplier` 1.5, `apex_count_window` 5) | **59,223** | 2,546,844 | 16:32 | 28.1 GB |
| `gate_min_score` 0.1 | 59,245 (+0.04%) | 3,124,230 (+23%) | 19:32 | 24.7 GB |
| `gate_min_score` 0.3 | 58,874 (-0.6%) | 2,027,300 | 13:42 | 28.0 GB |
| `rt_window_multiplier` 1.25 | 58,747 (-0.8%) | 2,441,118 | 14:26 | 23.9 GB |
| `rt_window_multiplier` 2.0 | 58,921 (-0.5%) | 2,695,313 | 20:20 | 34.4 GB |
| `window_holdout_frac` 0.3 | 59,124 (-0.2%) | 2,603,894 | 18:00 | 30.0 GB |
| `apex_count_window` 3 | 58,355 (-1.5%) | 2,658,530 | 16:20 | 28.0 GB |
| `apex_count_window` 7 | 57,805 (-2.4%) | 2,430,212 | 15:54 | 28.1 GB |

Every single-factor move is neutral or worse. The current defaults are a local optimum on
this acquisition; the only "gain" (0.04% at `gate_min_score` 0.1) costs 23% more rows through
every later stage and three minutes of wall for nothing. Held-out window sizing, which
docs/08 measured at +1.1% on AIF with a per-run DeepLC fine-tune, is neutral here with a
pre-fine-tuned library and the LOESS calibration, consistent with the mechanism it addresses
(an in-sample residual percentile) not being the limiting factor on this pool. The whole-run
peaks in this table are from the build before the extract merge and `windows_in_flight` cap
(section 19 has the current build).

## 19. The whole run on the current build, and the six-run experiment

One complete `mumdia run` on the HYE B01 file with everything in this document and docs/27
applied (streamed features, window-closing extract with incremental merge and 16 windows in
flight, parquet handoff) plus the fast rescore recipe (114 features, 3:1 hybrid cap, warm
start 5), 32 threads on doxy, per-stage peak of the process tree:

| stage | wall | peak RSS |
|---|---|---|
| convert | 13 s | 0.20 GB |
| search-seed | 19 s | 6.79 GB |
| rt-im-train | 2 s | 6.61 GB |
| extract | 5:23 | **16.49 GiB** |
| features | 6:20 | 3.11 GiB |
| compete | 33 s | 4.43 GB |
| rescore | 3:31 | 5.57 GB |
| quant | 1:29 | 2.50 GB |
| report | 1 s | 2.61 GB |
| **whole run** | **17:52** | **16.47 GiB** |

59,124 peptides, 65,672 target PSMs and 8,974 protein groups at 1%. On 2026-09-04 morning
the same run took 67:30 at 86.6 GiB (231 GiB before the first memory commit) for 59,515
peptides; the identifications are inside the pool's 0.9% seed band and the whole thing now
fits the 32 GB machine docs/27 set as its target with room to spare. Extract is the tallest
stage and features the longest; `extract.windows_in_flight: 8` takes the peak to about 12.3
GiB for another 30 s.

The six-run experiment (`run-experiment`: six per-run chains through compete, one pooled
rescore, per-run quant, LFQ combine) follows from the same numbers. The pooled rescore over
all 11,637,874 competed PSMs with the fast recipe was measured directly: **18:05 and 15.9 GB**
for 72,344 peptides and 390,333 target PSMs at 1%, against the August pooled result of 72,044
and 391,087 that took a multi-hour TSV-path rescore. With the per-run chain at ~11.4 min and
~16.5 GiB (the pre-rescore stages of the table above):

| `experiment.parallel_runs` | wall | peak |
|---|---|---|
| 1 | ~1 h 35 min | ~17 GB |
| 3 | ~50 min | ~50 GB |
| 6 | ~40 min | ~100 GB |

The per-run figure is from one file (the others are 1.8-2.1M competed rows, so +-10%).

The sensitivity recipe (section 15) through the engine on the same competed table: rescore
**18:38 at 5.57 GB for 59,242 peptides**, +0.2% over the fast recipe's 59,124 at 5.3x its
rescore wall (3 seeds x 5 folds x 10 iterations), so the whole run would be about 33 minutes.
That is an option for a final pass, not a default.

The like-for-like six-run pooled baseline landed: the same 11,637,874 competed PSMs under
the pre-recipe default (parquet handoff, all 387 features, every decoy, cold refits, one seed)
took **4:34:42 at 40.1 GB** for 71,926 peptides and 390,266 target PSMs at 1%, of which 97%
was training (15,675 s). The fast recipe's 18:05 at 15.9 GB for 72,344 peptides is therefore
15x faster, 2.5x smaller and +0.6% on the pooled experiment, consistent with the per-run
pools of section 17.

## 21. Shipped defaults (2026-09-05)

What became default, and what did not, from everything above:

- **Rescore: the training recipe.** `train_neg_ratio = 3`, `train_neg_select = hybrid`,
  `train_warm_epochs = 5`, on top of `handoff = parquet`, with every feature. Measured with
  seeds against the previous defaults (every decoy, cold refits):

  | pool | previous default | training recipe | training time |
  |---|---|---|---|
  | HYE A01 (2.4M PSMs) | 59,615 | 60,192 (+0.97%) | 164 s -> 18 s |
  | HYE B01 (2.6M PSMs) | 59,213 | 60,510 (+2.19%) | 2,994 s -> 156 s |
  | AIF (chimeric, 97k PSMs) | 10,512 | 10,502 (-0.09%) | flat |
  | entrapment (FASTA-built, spike-in FDP) | 2,837 (0.49%) | 2,932 (+3.33%, 0.60%) | 3x faster |

  (The A01 and AIF training times are the harness's; B01's are the real worker's.) The
  recipe is positive or flat on every pool including the empirical null, and the FDP is
  unchanged. `train_neg_ratio = 0, train_neg_select = random, train_warm_epochs = 0` restore
  the previous input exactly.
- **Not the compact feature preset.** With the training recipe in place, projecting to the
  114-feature list measured +0.2% (A01), **-1.2% (B01, seeds 59,754 / 59,829 against
  60,342 / 60,678)**, -0.1% (AIF) and +1.5% (entrapment). B01 is the one pool the list was
  never fitted on and its loss is four times the seed spread, so the projection is not free.
  What it buys is memory and handoff I/O: the rescore matrix is 3.4x smaller, and the
  full-scale HYE rescore stage measured 5.49 GB / 3:19 with it against **13.5 GB / 6:20**
  with every feature under the same training (process-tree peaks; 8.8 GB for the largest
  single process; 59,293 peptides, one seed). Six pooled HYE runs: 15.9 GB with it, roughly
  30 GB estimated with every feature. `rescore.feature_preset = compact` is
  therefore the documented option for pooled rescoring on machines where the matrix does not
  fit, not the default. The single-run peak is set by extract (16.5 GiB) either way.
- **Not the sensitivity recipe** (`folds 5, train_margin_frac 0.75, seeds 3`): +0.2 to +0.6
  pp over the fast recipe, at 5.3x the rescore wall through the engine (section 19). Documented
  in section 15 as the option for a final pass.
- **Retention time: calibration of DeepLC base-model predictions, DeepLC >= 4.1.1 required.**
  `rt_im_train.library_irt = auto` re-predicts an imported library's iRT with the base model
  when a DeepLC interpreter is configured (docs/08 section 4c: AIF 10,416 peptides against
  10,015 raw and 10,181 fine-tuned; HYE B01 over NN seeds 1-3 58,842 against 56,556 raw and
  60,278 with a once-fine-tuned library; `w_rt` 343 s against 632 s and 472 s on AIF), once
  per experiment. The once-per-library fine-tune remains the recommended extra step on a
  large reference (+2.4% on HYE, -2.3% on AIF). `finetune_deeplc` stays off; the fine-tune remains available. The floor is
  enforced by `doctor`, by the sidecar launch and by both worker scripts, because the default
  path is only sound on a base model that does not memorise its anchors.
- **Features stage: no cut.** The stage's cost is not the feature arithmetic. On the HYE B01
  competed table (2.6M candidates, 32 threads) the Extended set (387 features) took 380 s at
  3.11 GiB, `rich` (44) 255 s at 2.64 GB and `minimal` (14) 239 s at 2.59 GB, so removing
  96% of the features saves 37% of the wall and no memory; about 230 s is chromatogram decode,
  RT-axis and peak-shape work that every set pays. `rich` and `minimal` also cost 7.5-8.3%
  and 22-28% of the peptides (section 6). The stage keeps computing Extended and the
  projection happens at rescore, where it is free. The 43 dead columns (section 3) could be
  dropped from Extended for 11% less PIN I/O at zero sensitivity cost, but that is a schema
  version bump for about 15 s and is not done here.
- **Pooled experiment:** six HYE runs rescored together, previous default 4:34:42 at 40.1 GB
  for 71,926 peptides; fast recipe 18:05 at 15.9 GB for 72,344 (section 19).
- **Unchanged:** the rescorer's hyperparameters (section 17), the extraction and RT window
  defaults (section 18), `extract.windows_in_flight` (auto), and every benchmark-gated item
  in CLAUDE.md.

Reference point on the 2026-09-05 build, HYE B01, 32 threads: a complete `mumdia run` with
the compact preset is 17:52 at 16.5 GiB (extract is the tallest stage at 16.5 GiB, features
the longest at 6:20); under the shipped defaults (every feature) the rescore stage is 6:20
instead of 3:31, so about 20:40 at the same 16.5 GiB peak. The six-run pooled rescore with the
compact preset is 18 minutes at 15.9 GB for 72,344 peptides.

Shipped example configuration end to end (2026-09-06, merged tree, doxy, 32 threads):
`configs/examples/diann-library.json` with interpreters discovered through
`MUMDIA_PYTHON_DEEPLC` / `MUMDIA_PYTHON_RESCORE`, the raw imported HYE library (51 empty
`protein` cells filled, see the import script), HYE B01, everything else default:

| stage | wall | peak tree RSS |
|---|---|---|
| convert | 13 s | 0.2 GB |
| search-seed | 22 s | 6.8 GB |
| deeplc-repredict (4.91M unique sequences, once per experiment) | 25:19 | 10.8 GB |
| rt-im-train | 2 s | (w_rt 397 s) |
| extract | 4:41 | **17.0 GiB** |
| features | 4:13 | 3.0 GiB |
| compete (default key, 23% of candidates removed) | 20 s | 3.3 GB |
| rescore (387 features, training recipe) | 4:39 | 9.4 GB |
| quant + report | 1:18 | 2.6 GB |
| **whole run** | **41:08** | **17.9 GB** |

58,974 stripped peptides and 58,294 target PSMs at 1%, 9,082 protein groups at 1%; manifest
`rt_predictor = deeplc-base-model`, rescore `feature_preset all, train_neg_ratio 3, hybrid,
warm 5`. Without the one-off re-prediction the run is about 16 minutes. For scale: the same
file gave 56,556 peptides from the imported iRT and 60,278 from a once-fine-tuned library
(section 4c of docs/08, seeds 1-3, fast-recipe configuration).

## 20. Caveats of the original (2026-09-04) analysis

- Two HYE runs on one instrument type. The playbook rule (validate on at least two
  acquisition contexts plus an empirical null) means the AIF E. coli benchmark and an
  entrapment run are required before any subset becomes a default. The July AIF PIN under
  `out_aif_nn/` has the older 159-column schema and could not be reused here.
- The constant columns are constant because of the default configuration. A search with
  `emit_demix_features`, `emit_contested_features` or `ms1_precursor_features` turned on
  needs those columns back; a selection must be expressed per configuration, or keep the
  gated families whenever their gate is on.
- The re-implemented worker matches the real one (section 2) but uses mean/std
  standardisation where the in-memory TSV backend uses median/IQR. Section 6.4 is the
  check through the real worker.
- Subset construction used A01 only; B01 is a held-out test of those subsets, not a second
  training set. Subsets built on B01 would differ in detail (the univariate rankings
  correlate at 0.97, not 1.0).
- The screening subsample has 400k rows; its peptide counts are comparable to one another,
  not to full-pool counts.
- Nothing here is a statement about the Minimal or Rich sets as *classifiers*: they lose
  7.5% and 22% because they omit information the Extended families carry, not because the
  Extended set is 387 wide. The finding is that ~120 of the 387 carry that information.
