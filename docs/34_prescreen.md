# 34. Prescreen: fragment-rarity candidate filtering

`mumdia prescreen` (`stages/prescreen/`) scores every library candidate on the spectra of its own
isolation window inside its calibrated retention-time bounds and keeps the candidates whose
score exceeds a label-blind calibration quantile. Its purpose is a compute reduction: about half
of the candidates removed before extraction while nearly every identifiable candidate is kept.
It is off by default (`prescreen.enabled = false`).

It is a port of the `tagbench` prototype (`advanced_candidates.scores`, column `lowmz_half`, and
`advanced_filter.export`). The Rust score equals a line-by-line NumPy port of the prototype rule
exactly (maximum absolute difference 0.0 on 1,500 sampled candidates of the AIF entrapment run,
631 of them non-zero; `bench/prescreen/ps_parity.py`).

## 1. The score

For each eligible spectrum and each orientation of the residue-mass array:

- eligible spectrum: its isolation window holds the precursor m/z (`lower <= m < upper`) and its
  retention time lies inside the candidate's `run_windows` bounds (inclusive; unbounded or
  missing bounds mean the whole gradient, as everywhere else in MuMDIA);
- fragments: b and y at `k = 1..L-1` (b1 and y1 included) and charges `1..=min(2, z)`;
  `b = prefix / z + H+`, `y = (suffix + H2O) / z + H+`;
- matching: the nearest peak within 0.005 Da (first on a tie, both edges inclusive); it counts
  only if its intensity is positive, and each peak counts once however many fragments land on it;
- weight: `-ln p`, with `p = min(0.99999, (n_{b-1} + n_b + n_{b+1} + 0.5) / (n_window + 1))`,
  where `n_b` counts the window's spectra holding a peak in bin `b = rint(100 m/z)` (ties to even,
  as numpy). The histogram reads spectra only; no library, label or identification enters it;
- low-m/z evidence below m/z 300 has weight 0.5; there is no intensity threshold;
- the sum is divided by `sqrt(L)`; the candidate's score is the maximum over eligible spectra and
  over the forward and fully reversed orientation.

With both orientations the score is invariant under reversal of the residue array, so a fully
reversed decoy scores exactly as its target. Targets and decoys are scored by the same rule, each
on its own residues, m/z and RT window. The survivor target and decoy retention measured on both
validation runs are equal within 1 percentage point (section 5).

Peaks are stored as f32 in `spectra_ms2.parquet`. Every comparison is done in f64 on the widened
stored value, so a peak whose f32 value is 500.00500488 is outside `500.0 + 0.005`, as the f64
reference has it, although an f32 comparison would accept it (test
`f32_peaks_are_compared_in_f64_at_the_tolerance_boundary`).

## 2. Calibration

The in-scope candidates are split into a calibration and a reporting half by a seeded SplitMix64
hash of the candidate id (`prescreen.seed`); the split is independent of labels, row order and
thread count. The cutoff is numpy's `quantile(calibration_scores, target, method="higher")`,
that is `sorted[ceil((n - 1) q)]`, and a candidate is kept when its score is strictly greater.
The report gives the reduction on the reporting half, which did not set the cutoff. The cutoff is
recalibrated on every run, for every score definition and scope.

| Preset | Target | Prototype reduction | Reference retention | Weak-reference retention |
|---|---:|---:|---:|---:|
| `sensitive` | 0.52 | 50.29% | 99.70% | 99.24% |
| `balanced` (default) | 0.54 | 52.28% | 99.66% | 99.20% |
| `stringent` | 0.75 | 74.62% | 96.94% | 94.22% |
| `aggressive` | 0.90 | 90.30% | 90.58% | 82.67% |
| `custom` | `prescreen.target` | | | |

The prototype columns are exploratory six-run screening measurements on 1,500 reference and 4,000
sampled background precursors per run. A target does not guarantee the same reduction on another
run; section 5 gives the production measurements.

## 3. Scope

`scope = "modified"` with `scope_mods` (`RESIDUE:Name`, e.g. `M:Oxidation`; `n`/`c` for the
termini) filters only the forms carrying the listed modifications (`scope_match = "any"`) or all
of them (`"all"`). Matching is by residue and mass delta, so named, `UNIMOD:` and numeric spellings
agree. Candidates outside the scope pass through unscored, the cutoff is calibrated within the
scope, and the report gives both denominators: `scope_reduction` (within the class) and
`total_reduction` (over the library). With fewer than `min_calibration` (100) calibration
candidates the filter is bypassed, every candidate kept and the reason written to the report
(`bypass_reason`). Peptidoforms the mass model cannot parse also pass through, counted as
`unparsed_passed_through`.

The modifications defining filter eligibility are separate from the ones allowed in tags: an
oxidation-scoped filter never requires an oxidation-carrying tag.

## 4. Optional components

All are off by default. The prototype measured none of them as an improvement on the score of
section 1 near 50% reduction, so they are for experiments.

| Setting | Component |
|---|---|
| `retrieval = "tags"`, `rescue` | Database-free tag discovery (`tags.rs`) then retrieval: a candidate is retrieved when one of its trimers, in its own residue states and at a fragment charge it can carry, was observed in an eligible spectrum. `rescue = true` (default) scores the unretrieved candidates anyway; `false` drops them. The report separates `retrieval_losses` (unretrieved candidates the score would have kept) from `scoring_losses`. |
| `delayed_modforms` | Retrieve backbone families on backbone states first and examine forms only in retrieved families. Same retrieved set as form-by-form retrieval (tested); `retrieval_forms_skipped_by_family` counts the saving. `retrieval::forms_with_tag` is the delayed generator, equal as a set to enumeration-first generation filtered on the tag (tested with two modification types). |
| `tags.*` | Edge tolerance 0.005 Da neutral (0.005/z m/z), fragment charges 1-2, whole-path RMS sigma 0.003 Da, optional `gap_edges` (two-residue steps, every compatible pair kept, three-peak ladders keyed separately from four-peak ones), `extra_mods`. The alphabet comes from `peptidoforms.fixed_mods`/`variable_mods`: I and L merged, a fixed modification replaces its residue's state, a variable one adds a state; positions the alphabet cannot express break the trimers covering them and are never substituted. There is no peak, degree or path-count cap. |
| `complement_bonus` | Complementary-ion support of observed tag paths: complements of the four path peaks for the candidate's neutral mass, on distinct peaks off the path, never reused; 2/3/4 complements give 1/3, 2/3, 1 times the path fit, times spectrum tag information, divided by `sqrt(L)`. |
| `tag_bonus`, `fasta_bonus` | Spectrum-side tag information times path fit, and the library-side information from the backbone/position index (`retrieval::BackboneIndex`), kept as separate columns. Each distinct trimer counts once. |
| `flank_bonus` | Positioned ladders whose endpoints sit at the candidate's own b or y masses. |
| `mass_hypotheses.*` | Blind neutral-mass hypotheses from tag paths and two or more complementary peaks, per 0.01 Da bin, from a seeded sample of spectra, written to `<out>.mass_hypotheses.parquet` before any candidate is consulted; optional MS1 mono/+1 isotope links within 2 s and max(0.005 Da, 10 ppm); `require_ms1` is a hard gate and off. |
| `trace.*` | Seven-scan same-window pools, bounded 0.01 Da clusters (max - min), intensity-weighted centroids, summed intensity, detection counts and unit per-scan profiles; `merged` (two or more detections) or `unmerged`; `tag_coherence` (minimum pairwise trace cosine of the four path features times fit) or `coherent_fragments` (matched fragments whose trace is compatible with the most intense matched feature, so ladders need not be connected). The single-scan score is unaffected. |
| `localization` | `own` (default), `family_support` (every localization sibling, same residues, charge, label and modification composition, takes the family best; ties and alternatives pass together) or `best_site` (keeps only the best site, ties kept; measured to lose modified references in the prototype). |

With any component weight positive, the combined score is `base / q95(base) + sum_k w_k c_k /
q95(c_k)` over the calibration half, the prototype's scaling, and the cutoff is calibrated on it.
`write_scores` writes every candidate's base score, combined score, retrieval flag and component
values to `<out>.scores.parquet`.

The complement and flank components re-walk the tag paths per candidate and spectrum, which the
prototype afforded on 5,500 candidates per run but which does not scale to a 10.9M-candidate
library. `mumdia prescreen --sample-candidates N` scores a seeded, label-blind sample of about N
candidates for such measurements; its survivors cover the sample only and are not an extract
allowlist.

## 5. Validation

RESULTS

## 6. Use

Inside a run: `"prescreen": {"enabled": true}` (`configs/examples/prescreen-balanced.json`). The
stage runs after `rt-im-train`, on the library extract will read, and extract searches only its
survivors. Not yet supported with `groups.window_groups > 1` (validation rejects the combination).

Standalone, for a run directory written by `mumdia run`:

```text
mumdia prescreen --ms2 RUN/spectra/spectra_ms2.parquet \
  --lib-precursors RUN/fragment_library_precursors_multihead.parquet \
  --run-windows RUN/run_windows.parquet --out survivors.parquet \
  --preset balanced --write-scores
mumdia extract ... --restrict-candidates survivors.parquet
```

`--preset`, `--target` and `--top-peaks` override the configuration. `configs/examples/
prescreen-oxidation.json` is the oxidation-only filter; `prescreen-components.json` enables tag
retrieval with rescue, the 0.25 complementary bonus and family support.

The validation campaign is reproducible with `bench/prescreen/ps_val.sh` (baseline run, then
paired arms that share the baseline's spectra, run windows, adapted library and mass
calibration), `bench/prescreen/ps_ext.sh` (component arms), `bench/prescreen/ps_agg.py`
(summary) and `bench/prescreen/ps_plot.py` (figures).
