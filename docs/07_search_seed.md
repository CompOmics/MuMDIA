# search-seed (Stage S): broad search + mass recalibration

> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

Stage S is a native broad, DIA-aware search whose product is **calibration
anchors, not final identifications**. It scores library candidates against every
MS2 scan with a Sage-lite hyperscore, keeps the single best-scoring spectrum per
candidate, assigns target-decoy q-values, and from the confident subset derives a
per-run fragment mass recalibration (systematic ppm offset plus a learned
tolerance). Downstream stages consume its two outputs:

- `rt-im-train` reads `seed_psms.parquet` (the confident target PSMs give the
  observed-RT-vs-predicted-iRT anchors for LOESS/linear calibration and the RT
  window widths).
- `extract` reads `<seed>.masscal.json` (the offset is applied to the observed
  peak m/z to align it with the predicted-fragment frame, the learned tolerance
  replaces `extract.frag_tol_ppm`).

The stage sits behind a file contract (`search_seed.rs:2-4`), so a real Sage /
sage-core adapter can replace the native scorer later without touching the
consumers, as long as it writes the same `seed_psms` schema and `masscal.json`.
Library-level decoys are the single source of truth: there is no engine-side decoy
generation here, so target-decoy counting is never mixed-method
(`search_seed.rs:6-7`). Because the seed is broad and best-per-candidate rather
than best-per-spectrum, it is deliberately high-recall and low-precision at this
point; precision is recovered downstream by RT/mass calibration, feature
rescoring, and FDR control.

The stage is iRT-independent: `predicted_irt` is only copied through to the
output, never used in scoring. That is why the `run` orchestrator computes the
seed once on the base library and reuses it both before and after the optional
DeepLC fine-tune (`run.rs:237-280`).

## Files

| path | role |
|---|---|
| `rust/mumdia/crates/mumdia/src/stages/search_seed.rs` | the stage: scoring loop, best-per-candidate merge, mass recalibration, output writers |
| `rust/mumdia/crates/mumdia/src/fdr.rs` | `target_decoy_q`, `count_targets_at_q`, `ln_factorial` (shared with `rescore`) |
| `rust/mumdia/crates/mumdia/src/matchers/fragindex.rs` | `FragIndex` (CSR inverted fragment index) + `SeedScratch` epoch accumulator |
| `rust/mumdia/crates/mumdia/src/index.rs` | `Library::load`, `candidate_range`, `page_search` (bucketed backend), `cand_frags` |
| `rust/mumdia/crates/mumdia/src/calibrate.rs` | `percentile` used by the tolerance fit |
| `rust/mumdia/crates/mumdia/src/spectra.rs` | `load_ms2` (reads the converted MS2 Parquet into `Ms2Scan`, with the per-peak 1/K0 of spectra v2), `nearest_peak` (the one nearest-peak rule of the calibrants and the 1/K0 points) |
| `rust/mumdia/crates/mumdia/src/stats.rs` | `weighted_median` (the observed 1/K0) |
| `rust/mumdia/crates/mumdia-core/src/constants.rs` | `ppm_bounds`, `ppm_diff`, `within_ppm` (the shared ppm math) |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `SearchSeedConfig` (`:286`), `MatcherKind` (`:38`) |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | `artifact::SEED_PSMS = ("seed_psms", 1)`, `SEED_PSMS_V2` and the selector `artifact::seed_psms(has_im)` |
| `rust/mumdia/crates/mumdia/src/main.rs` | CLI `Cmd::SearchSeed` (`:75`, dispatch `:458`) |
| `rust/mumdia/crates/mumdia/src/stages/run.rs` | orchestrator wiring (`:219`), masscal handoff to extract (`:309`) |

## Inputs and outputs

### Consumed

- **MS2 spectra** (`--ms2`, a `convert` output Parquet), loaded by `load_ms2`
  into `Vec<Ms2Scan>`. Each `Ms2Scan` carries `scan_index`, `rt_seconds`,
  an `IsolationWindow { target_mz, lower_mz, upper_mz, im_lower, im_upper }`, and
  `peaks: Vec<Peak { mz: f32, intensity: f32 }>`. The peak m/z is the artifact's
  own f32 width; this stage widens it with `as f64` at each comparison, which is
  exact, exactly as it already did for the library's f32 fragment m/z.
- **Library** (`--lib-precursors` + `--lib-fragments`), loaded by
  `Library::load_with` (`index.rs`). Provides the per-candidate columns
  (`peptidoform(cid)`, `charge`, `prec_mz`, `base_peptide_id`, `protein(cid)`,
  `is_decoy`, `predicted_irt: f32`, and the CSR `frag_offsets`; `cand(cid)` reads
  them back as one borrowed record), the flat fragment arrays (`frag_mz`,
  `frag_int`, `frag_name_id`), and the bucketed inverted index for `page_search`.
- **Config**: `cfg.search_seed` plus `cfg.extract.bucket_size` (the bucketed
  Library index bucket width; `run.rs:225`, `main.rs:473`).

### Produced

**`seed_psms.parquet`** (schema id `seed_psms` v1, `schema.rs:15`), one row per
candidate that, in at least one scan, both cleared `min_matched_peaks` and ranked
within that scan's top `report_psms` (the per-scan hyperscore sort is truncated to
`report_psms` *before* the best-per-candidate fold, `search_seed.rs:89`/`:369`, so
a candidate that always ranks below `report_psms` gets no row even if it cleared
`min_matched_peaks`). Rows are sorted by `candidate_id`. Written at
`search_seed.rs:233-250`. The `ArtifactReport` records `logical_name` =
`schema_name` = `"seed_psms"`, `schema_version = 1`, and `stage = "search-seed"`
(`search_seed.rs:257-260`):

| column | type | meaning |
|---|---|---|
| `candidate_id` | u32 | dense library candidate id (== library row index) |
| `peptidoform` | str | ProForma-lite peptidoform string |
| `charge` | i32 | precursor charge |
| `precursor_mz` | f64 | library precursor m/z |
| `base_peptide_id` | u32 | stripped-peptide id (for peptide-level rollup) |
| `protein` | str | protein accession |
| `label` | str | `"target"` or `"decoy"` (from `Candidate::is_decoy`) |
| `score` | f64 | best hyperscore over all scans |
| `spectrum_q` | f64 | best-per-candidate target-decoy q-value |
| `observed_rt` | f64 | RT (seconds) of the best-scoring scan |
| `predicted_irt` | f32 | library iRT, copied through (unused in scoring) |
| `matched_peaks` | i32 | matched-fragment count at the best scan |
| `scan_index` | u32 | `scan_index` of the best-scoring scan |
| `observed_im` | f64, nullable | v2 only: observed 1/K0 of a confident target and, under `rt_im_train.im_window_sizing = seed_decoy_corrected` (the default), of every other PSM up to `im_window_sizing_q` (step 6) |

The seed is version 2 (`artifact::seed_psms(true)`), with the `observed_im` column and the
report stat `with_observed_im`, only when the searched scans carry per-peak 1/K0 (spectra v2,
the native diaPASEF reader). Otherwise it is version 1 with exactly the columns above it, byte
for byte. The orchestrators record the version read from the table's footer
(`search_seed::seed_schema`), not a constant.

**`<out>.masscal.json`** (fitted and serialised by `masscal::MassCal`, `masscal.rs`):

| key | type | meaning |
|---|---|---|
| `frag_ppm_offset` | f64 | median signed ppm of `observed_peak` relative to `predicted_fragment` (`ppm_diff(peak_mz, fmz)`), i.e. the systematic offset |
| `frag_tol_ppm` | f64 | learned tolerance: `max(frag_tol_floor_ppm, 1.5 * P95(|dev - offset|))`, or `max(frag_tol_floor_ppm, frag_tol_mad_k * 1.4826 * MAD)` under `search_seed.frag_tol_mad_k` |
| `frag_ppm_sigma` | f64 | duplicate of `frag_tol_ppm` (the learned tolerance is the local mass-uncertainty estimate) |
| `n_dev` | usize | number of fragment-to-nearest-peak deviations collected (`devs.len()`) |
| `cal_passes` | int | 0 (fallback, too few devs), 1 (single pass), or 2 (robust second pass) |

Only `frag_ppm_offset` and `frag_tol_ppm` are consumed downstream (by `extract`,
`extract.rs:678-698`). `frag_ppm_sigma`, `n_dev`, and `cal_passes` are written but
read by no consumer (grep-confirmed); they are diagnostic / audit fields. The
deviations are not the postings matched during scoring: for every confident target
PSM a fresh nearest-peak search over the candidate's full library fragment list is
run at its best scan (see step 6), so `n_dev` counts fragment-to-peak pairs, not
matched hyperscore postings.

**`<out>.report.json`** (`ArtifactReport`, `search_seed.rs:256-274`): rows,
`content_hash` (blake3 of the output Parquet, `search_seed.rs:262`), `params`
(`fragment_tol_ppm`, `report_psms`, `min_matched_peaks`, `top_n_peaks`,
`fdr_seed`, and `count_fragments_once: true` only when that key is on; note `matcher`
and `two_pass_mass_cal` are *not* recorded in `params`),
a `stats` map (`psms`, and `targets_at_q<fdr_seed>` whose key is the float-
formatted threshold, e.g. `targets_at_q0.01`), `model_identity =
"native-seed-hyperscore-v1"`, and `elapsed_ms`.

## How it works

Entry point: `search_seed::run(SearchSeedParams)` (`search_seed.rs:175`), a wrapper
over `run_hashed` (`search_seed.rs:181`), itself a wrapper over `run_returning_scans`
(`search_seed.rs:189`).

**1. Load** the library and MS2 scans (`load_ms2`), then log candidate and scan
counts. On the fragindex matcher the library is the m/z-only one
(`Library::load_mz_only`) and its index the m/z-only index
(`FragIndex::build_mz_only`): the seed reads neither predicted intensities nor
fragment names (the hyperscore is a match count plus observed intensity, and the mass
recalibration walks `frag_mz`), so neither the library nor the index holds those two
columns, 12 bytes per fragment at the build peak (AIF, 20.6M fragments: seed peak
1,269-1,297 to 1,023-1,052 MB). They are still decoded, batch by batch, and checked
exactly as extract's full load checks them (presence and type, NULLs, non-finite
intensities, more distinct fragment names than the u16 id holds), then discarded. A
library that extract would refuse is therefore refused here, with the same message,
before any DeepLC step runs on the seed's output. The seed's accumulator probes through `FragIndex::probe_peak_cand`, which visits the same
postings in the same order as `probe_peak`; the payload entry points refuse an
m/z-only index. The bucketed matcher loads the full payload, as `page_search` needs it. On the fragindex matcher the MS2 decode runs
concurrently with the library load and the index build (`rayon::join`); they are
independent, and the scans are resident during the index build either way. The
bucketed matcher keeps them in sequence, because its library load builds a sorted
copy of every fragment and holding the scans through that transient would raise the
peak. `run_returning_scans` hands the decoded scans back to the caller: the
ungrouped `run` lends them to extract when no DeepLC step runs between seed and
extract and `extract.matcher` is `fragindex`, so a run decodes its MS2 once. With a
DeepLC step it does not, because the scans would then sit in the parent across the
sidecar's peak; with the bucketed extract matcher it does not either, because they
would sit through that matcher's sorted library copy. Only the MS2 is lent: extract
decodes the MS1 itself, concurrently with its library load. `load_ms2` sorts the
returned `Vec<Ms2Scan>` by
`rt_seconds` ascending (`spectra.rs:95`); this RT ordering is what makes the
within-group strictly-greater update deterministic (earliest-RT wins a tie). It
does **not** re-sort each scan's peaks: peak m/z order is inherited from `convert`
(see the mass-cal invariant below).

**2. Build the matcher.** When `cfg.matcher == Fragindex` (the default), a
`FragIndex` is built once over the whole library at `cfg.fragment_tol_ppm`
(`search_seed.rs:56-57`; `FragIndex::build`, `fragindex.rs:46`). This is a
log-space-binned CSR inverted fragment index: postings are scattered in
candidate-id order so `post_cand` is ascending within every bin, which lets
`probe_peak` narrow to the precursor-window candidate sub-range by binary search
(`fragindex.rs:152-182`). The bucketed backend uses the Library's own inverted
index via `page_search` and needs no separate build.

**3. Best-per-candidate accumulation.** Two paths produce the same
`HashMap<u32, Best>` where `Best { score, rt, matched, scan_index }`
(`search_seed.rs:37-43`):

- *Fragindex path* (`seed_fragindex_windows`, `search_seed.rs:313-410`): scans
  are grouped by isolation window, keyed on
  `(lower_mz.to_bits(), upper_mz.to_bits())` in a `BTreeMap` for deterministic
  group order (`:321-330`). Each scan belongs to exactly one window, so groups
  are independent parallel units; `rayon` `par_iter().map_init(SeedScratch::new)`
  processes them (`:334-390`). Per group, `candidate_range` is computed once
  (`:343`). Per scan, `select_peaks` picks the probe set, `SeedScratch::accumulate`
  probes each peak and fuses `(count, obs_sum)` per touched candidate
  (`fragindex.rs:214-237`), the candidates with `count >= min_matched_peaks` (the
  scratch's qualified list, collected as they reach the count) are scored by
  `hyperscore`, sorted by score desc then candidate-id asc, truncated to
  `report_psms`, and folded into a group-local best with a strictly-greater update
  (`:357-385`). Since 2026-09-25 the parallel unit is a contiguous RT chunk of a
  served window group rather than the whole group (`seed_chunk_plan`): groups whose
  candidate range is empty get no task, and the served groups are split when their scans
  would give fewer than about four tasks per thread, never below 32 scans per chunk. A
  banded search serves only the one to three windows over its band, so its whole probe
  phase ran on that many threads before. A group's chunks merge back in chunk order with
  the same strictly-greater update, so an earlier chunk keeps an exact tie and the group's
  best is the one the scan-by-scan walk gives
  (`chunk_tests::a_chunked_seed_is_the_one_task_per_window_seed_exactly`). An ungrouped
  run with many windows mostly keeps one task per group.
- *Count each fragment once* (`count_fragments_once`, default false, fragindex only). The
  default accumulation counts every matched posting, so two peaks inside the tolerance of
  one predicted fragment count as two matched fragments in `ln(matched!)`. On a diaPASEF
  slot spectrum the mobility pieces of one fragment ion sit at one m/z and do exactly
  that. With the key on, the seed index also carries each posting's fragment ordinal
  (`FragIndex::build_mz_only_with_ordinals`, 2 bytes per posting more; the lent
  `SeedLibrary` records it in its key), and `SeedScratch::accumulate_count_once` collects
  the scan's `(candidate, fragment, intensity)` hits, sorts them by `(candidate, fragment)`
  and counts each pair once at its most intense peak, adding the maxima in that order.
  The default `accumulate` and `probe_peak_cand` are untouched. With no fragment matched
  twice the counts and the qualified set are the default's
  (`count_once_is_the_plain_accumulation_when_no_fragment_repeats`). Config validation
  refuses the key with `matcher = bucketed`. Benchmark-gated (docs/35_diapasef.md, arm
  T13). On the IM branch the mobility valley split alone cut the seed's confident anchors
  from 5,200 to 3,331, and counting once brought them to 5,534, +3.8% peptides with the
  split (three seeds). The diaPASEF recipe (`configs/examples/diapasef.json`) sets it
  with both valley fractions.
- *Bucketed path* (serial, `search_seed.rs:66-107`): per scan, `candidate_range` on
  the Library, `select_peaks`, then `page_search` for each probed peak accumulates
  `(count, obs_sum)` into a `HashMap`. Same `min_matched_peaks` filter, hyperscore,
  sort, `report_psms` truncation, and best-per-candidate update.

Both paths skip a scan whose candidate range is empty (`hi <= lo`,
`search_seed.rs:69-71` / `:344-346`), so out-of-library-range windows cost nothing.
Both accumulate `obs_sum` as the summed **observed** peak intensity of matched
postings; the predicted fragment intensity is deliberately discarded in the seed
(`fragindex.rs:185-190`, and the `_pi`/`_mz` discards at `search_seed.rs:77`).

The two backends do **not** use the same tolerance-edge predicate, so their matched
sets (and therefore ID counts) can differ slightly on the same data. The fragindex
`probe_peak` verifies with `within_ppm` (min-relative, symmetric about the smaller
mass, f64; `constants.rs:92`), while the bucketed `page_search` matches inside
`ppm_bounds(peak, tol)` (query/observed-relative, f32-truncated bounds;
`index.rs:258-282`). The mass-cal deviation collection uses a third convention,
`ppm_diff` (theoretical/predicted-relative; `constants.rs:66`). Do not assume the
three are interchangeable at the edge; `config.rs:30-35` records that the
predicate difference shifts IDs more on the AIF full-range-window case.

**Hyperscore** (`search_seed.rs:413-415`):

```
hyperscore = ln(matched!) + ln(1 + sum_obs)
```

`ln(matched!)` is `ln_factorial(matched)` = the summed logs of `2..=matched`
(`fdr.rs:137-143`), rewarding fragment breadth; `ln(1 + sum_obs)` adds a bounded
intensity term. This is the Sage-style form and is intentionally simple because
the seed is a calibration pass, not the scored identification.

**4. Cross-group merge** (fragindex path, `search_seed.rs:392-409`): a total-order
merge over the per-group partials keeps, per candidate, `max score`, ties broken by
`earliest rt`, then `min scan_index`. This is order-independent, so it is
bit-identical to the serial global best regardless of thread or group scheduling.

**5. q-values.** Rows are collected and sorted by `candidate_id`
(`search_seed.rs:111-112`), a `(score, is_decoy)` vector is built, and
`target_decoy_q` (`fdr.rs:7-52`) computes per-candidate q. That routine sorts by
score descending, walks tied-score blocks together so every PSM in a block gets
the same q, uses the conservative numerator `q = (n_decoys + 1) / max(1, n_targets)`,
and monotonizes from worst to best score so q is non-increasing. The output column
is `spectrum_q`. `count_targets_at_q(q, is_decoy, fdr_seed)` (`fdr.rs:115`) reports
the confident target count into `stats` (`search_seed.rs:148`).

**6. Fragment mass recalibration** (`search_seed.rs:376-468`). A
`scan_index -> &Ms2Scan` map is built (`:381-384`). For every **confident target**
PSM (`!is_decoy && q <= fdr_seed`, `:400-410`), each library fragment m/z of that
candidate (`lib.cand_frag_mz(cid)`, `:417`; `index.rs:1566`) is searched against the
best scan's peaks. `cand_frag_mz` is the only fragment accessor the m/z-only seed
library serves: `cand_frags` hands out intensities and names too, and panics on a
library that does not hold them. The search window is `max(50 ppm,
fragment_tol_ppm)` (`ppm_bounds(fmz, collect_ppm)`, `:425-429`; `partition_point`
finds the low edge, then a linear scan to the high edge).
This `partition_point` + linear scan **assumes `scan.peaks` is m/z-ascending**;
`load_ms2` does not re-sort peaks by m/z (`spectra.rs:20-97`), so the assumption
rests on `convert` writing peaks in m/z order. The nearest peak by absolute m/z
distance (`spectra::nearest_peak`; a tie keeps the first, lower-m/z peak, because only a
strictly smaller distance replaces the best) contributes one signed ppm deviation
(`ppm_diff(peak_mz, fmz)`, `constants.rs:66`) to `devs` (`:426-449`). Every library fragment of the candidate
is probed, not only those that matched during scoring, so a fragment that found no
scoring posting can still supply a calibrant. The 50 ppm floor keeps the net wider
than a narrow `fragment_tol_ppm`, so a systematic offset larger than the tolerance can
still be measured; on a wide-tolerance search the net is the tolerance itself, so the
percentile that sets the learned tolerance is not truncated by the net.

**Observed 1/K0** (`observed_im`, spectra with per-peak 1/K0 only). The same walk gives each
confident target its observed 1/K0: every nearest peak whose deviation lies within
`fragment_tol_ppm` (not the wider collection window, so random matches stay out) contributes
its `(1/K0, intensity)`, and `observed_im` is the intensity-weighted median of those points
(`stats::weighted_median`: the smallest value whose cumulative weight reaches half the
total). Fragments inherit the precursor's mobility in diaPASEF, because the TIMS separation
precedes fragmentation, so the median over the matched fragments is the precursor's 1/K0
at its best scan, and an intense fragment outweighs a faint interferer at another 1/K0. The
1/K0 point of a fragment is the peak whose deviation calibrates the mass, so a 3D run's
calibrants are unchanged by the shared helper. Under `rt_im_train.im_window_sizing =
seed_decoy_corrected`, the default (`SearchSeedParams::observed_im_q`), a separate pass after
the calibrant walk gives every other PSM with `spectrum_q` at most `im_window_sizing_q`
(default 0.1), targets and decoys, an `observed_im` by the same rule; rt-im-train uses those
rows only to size the 1/K0 window, and takes its anchors from the confident targets alone.
Under `anchors`, decoys and unconfident targets get null. A row without a matched peak in
tolerance gets null either way. rt-im-train calibrates the library's `predicted_im` against
these values (docs/08_rt_im_train.md, "Ion-mobility calibration").
The grouped seed pool does not carry the column yet: a grouped run is refused on diaPASEF
input (docs/35_diapasef.md section 3).

The one estimator is `masscal::fit_floored(devs, floor, mad_k)`. It takes the median
of the deviations as the offset (`sorted[len/2]`, the upper-middle element for
even-length inputs, not the two-element average) and sets the tolerance to
`max(floor, 1.5 * P95(|dev - offset|))` using `calibrate::percentile(.., 0.95)`
(nearest-rank on `round(p*(n-1))`). `floor` is `search_seed.frag_tol_floor_ppm` (default
5) in every pass. Control flow (`MassCal::fit_from`):

- `devs.len() < 20`: fallback, `(0.0, fragment_tol_ppm, cal_passes = 0)` (no
  offset, tolerance unchanged).
- otherwise single pass: `(o1, t1, 1)`.
- if `two_pass_mass_cal`: keep only deviations inside `|dev - o1| <= t1` and, when
  `>= 20` survive, re-fit to `(o2, t2, 2)`; otherwise keep the single-pass result.
  The second pass rejects random-match outliers so they cannot bias the median.

`search_seed.frag_tol_mad_k` (default 0, off) replaces the P95 width with the robust
sigma of the deviations: `mad_k * 1.4826 * MAD`, MAD being the median of
`|dev - offset|`, still floored at `frag_tol_floor_ppm`. A P95 is set by the tail of the
deviations, so a shoulder of random matches (10% of the deviations in
`the_mad_estimator_ignores_a_shoulder_that_sets_the_p95`) can widen it several-fold,
where the MAD follows the core. Under `two_pass_mass_cal` its second pass refits the
offset only, on the deviations inside the first-pass window, and keeps the first-pass
width: a MAD re-estimated on inliers truncated at that width is narrower than the
spread they were cut from, so every pass would tighten it for no reason in the data
(`the_mad_second_pass_refits_the_offset_and_keeps_the_width`). At 0 the result is the
P95 estimator's, bit for bit. On the IM branch MAD with k = 4 gave +5.7% over the P95
arm on diaPASEF E. coli (three seeds, single pass); HYE was not isolated, and every IM
number is single-pass. On main with two-pass mass calibration (diaPASEF E. coli, one seed)
k = 4 learned 10.0 ppm against 11.7 ppm for P95 and gave 11,037 peptides against 11,072, no
gain. The key is benchmark-gated (docs/35_diapasef.md, arm T3), and
whether one estimator can serve every acquisition is open; if one ties or wins
everywhere, the other is retired.

A negative result, recorded so it is not tried again: do not size the fragment tolerance
from accepted identifications. The IM branch's variant C re-estimated the tolerance from
the deviations of the identified precursors and set it to 7.99 ppm, which cost 2.7% of
the identifications; a fixed 8 ppm cost 3.7%. The accepted set is already filtered by the
tolerance it would size, so its spread understates the run's.

The estimator is `masscal::MassCal::fit_from` and the JSON body its `to_json`, both in
`masscal.rs`, because a grouped search fits the very same estimator a second time: each
band's seed would otherwise learn a tolerance from its own ~2,000 deviations and
`seed-pool` would average the bands' scalars, which measured 35% wider and cost 3.4% of the
peptides (`docs/33_window_groups.md` section 4a). Under `groups.window_groups > 1` the stage
is additionally asked (`SearchSeedParams::emit_calibrants`) to write
`<out>.masscal.parquet`: `candidate_id` (library-wide), `scan_index`, `frag_mz`, `ppm`, one
row per deviation, for every target PSM of the band and not only its own confident ones,
so the pooled q rather than the band's own q chooses the calibrants (a fixed prefix of the
band's 2,000 best targets was short at 2 and 4 bands; `docs/33_window_groups.md` section
4a). The report's stats then carry `calibrant_deviations`, `calibrant_bytes` (16 B each) and
`calibrant_target_rows`; the count is bounded by the band's scans times `report_psms` times
the fragments per candidate, not by the library. An ungrouped run writes no sidecar, adds
no stats and is byte-identical.

**7. Write** `seed_psms.parquet` (`write_table`, `:233`) and the `ArtifactReport`
(`:256-274`), then log `psms`, `confident`, `elapsed_ms`.

### How the outputs are consumed

`run` writes the seed to `seed_psms.parquet` (`run.rs:218-235`) and passes
`mass_cal: Some("<seed>.masscal.json")` to `extract` (`run.rs:309`). In
`extract` (`extract.rs:678-698`), the JSON is read: `frag_ppm_offset` becomes
`offset_factor = 1.0 + frag_ppm_offset * 1e-6` (`extract.rs:698`), which divides
each observed peak m/z before matching (`q_mz = peak.mz / offset_factor`, e.g.
`extract.rs:303`/`:740`) to bring observed peaks into the predicted-fragment
frame, and `frag_tol_ppm` replaces `extract.frag_tol_ppm` as the matching
tolerance (falling back to the config value if the file is absent,
`extract.rs:696`).

## Key types and functions

| name | file:line | what it does |
|---|---|---|
| `run` | `search_seed.rs:175` | stage entry point; orchestrates load, score, q, masscal, write |
| `run_hashed` | `search_seed.rs:181` | `run`, returning the row count and the report content hash (`mumdia_io::report::Written`) |
| `run_returning_scans` | `search_seed.rs:189` | `run_hashed`, also handing back the MS2 scans it decoded itself (`None` when they were lent), for a caller that lends them on to extract |
| `SearchSeedParams` | `search_seed.rs:27` | input struct (`ms2`, `library_precursors`, `library_fragments`, `out` output-path prefix, `cfg`, `bucket_size`, `config_hash`, `fragment_offset`, `ms2_scans`, `emit_calibrants`, `library`) |
| `SeedLibrary` / `SeedLibrary::load` | `search_seed.rs:92` / `:120` | a seed's library and index, loaded once and lent to every seed of `run-experiment`; records how it was built and is refused by a seed with other settings |
| `Best` | `search_seed.rs:168` | per-candidate best `{ score, rt, matched, scan_index }` |
| `seed_fragindex_windows` | `search_seed.rs:695` | parallel per-window fragindex scoring + deterministic merge |
| `select_peaks` | `search_seed.rs:563` | top-N-by-intensity peak selection, re-sorted to index order |
| `hyperscore` | `search_seed.rs:850` | `ln(matched!) + ln(1 + sum_obs)` |
| `Library::load_mz_only` | `index.rs:1133` | the seed's library: precursor columns and fragment m/z; the payload is decoded and checked as the full load checks it, then discarded |
| `FragIndex::build` / `build_mz_only` | `fragindex.rs:78` / `:99` | build the CSR inverted index at a fixed tolerance; `build_mz_only` without the intensity and fragment-ordinal payload (`has_payload`, `:105`) |
| `FragIndex::probe_peak` / `probe_peak_cand` | `fragindex.rs:396` / `:425` | matched postings for one peak in a candidate range; `probe_peak_cand` reports the candidate only and is what the seed probes through (it also serves an m/z-only index) |
| `SeedScratch::with_min_count` / `accumulate` / `qualified` | `fragindex.rs:667` / `:691` / `:741` | epoch-stamped `(count, obs_sum)` accumulator, one 16-byte slot per candidate; `qualified` lists the candidates that reached `min_matched_peaks`, in first-reached order |
| `SeedScratch::accumulate_count_once` / `FragIndex::probe_peak_cand_frag` / `build_mz_only_with_ordinals` | `fragindex.rs` | `count_fragments_once`: each (candidate, fragment) counted once per scan at its most intense peak, through an m/z-only index that also holds the fragment ordinals |
| `spectra::nearest_peak` | `spectra.rs` | nearest peak within a ppm window, ties to the first; serves the calibrant deviations and the 1/K0 points |
| `stats::weighted_median` | `stats.rs` | the observed 1/K0 of a confident target |
| `seed_schema` | `search_seed.rs` | the `seed_psms` version a table on disk records (v2 when it has `observed_im`) |
| `Library::candidate_range` | `index.rs:1612` | half-open candidate id range for an isolation window |
| `Library::page_search` | `index.rs:1621` | bucketed inverted-index probe (bucketed backend) |
| `Library::cand_frag_mz` | `index.rs:1566` | fragment m/z slice for a candidate; the one fragment accessor an m/z-only library serves |
| `Library::cand_frags` | `index.rs:1549` | `(m/z, intensity, name id)` slices for a candidate; panics on a library without the payload |
| `target_decoy_q` | `fdr.rs:51` | tied-block, monotonized `(n_decoys+1)/max(1,n_targets)` q |
| `count_targets_at_q` | `fdr.rs:238` | target count at or below a q threshold |
| `ln_factorial` | `fdr.rs:260` | `ln(n!)` via summed logs |
| `ppm_diff` / `ppm_bounds` | `constants.rs:97` / `:109` | signed ppm (theoretical-relative) and ppm window bounds (query-relative) |
| `within_ppm` | `constants.rs:130` | min-relative tolerance predicate used by the fragindex match (differs at the edge from the two above) |
| `load_ms2` | `spectra.rs:86` | reads `spectra_ms2.parquet` into `Vec<Ms2Scan>`, RT-sorted, decoding row-contiguous parts in parallel |
| `percentile` | `calibrate.rs:250` | nearest-rank percentile (used for the tolerance) |

## Configuration

`SearchSeedConfig` (`config.rs:286-320`, `#[serde(default, deny_unknown_fields)]`).
The struct was pruned to the fields actually read here; unknown keys are rejected
on load.

| field | default | effect |
|---|---|---|
| `fdr_seed` | `0.01` | q threshold defining "confident" for the calibration subset and the reported `targets_at_q` stat |
| `fragment_tol_ppm` | `20.0` | matching tolerance for scoring; also the masscal fallback tolerance when too few deviations |
| `report_psms` | `5` | max candidates kept per spectrum before the best-per-candidate fold (wide-window DIA) |
| `min_matched_peaks` | `4` | minimum matched fragments for a candidate to score in a scan |
| `top_n_peaks` | `300` | probe only the N most intense peaks per scan (`0` = all); seed-only, does not shrink the extract artifact |
| `matcher` | `Fragindex` | backend; `MatcherKind::Bucketed` (`config.rs:38`) uses the serial `page_search` path |
| `two_pass_mass_cal` | `true` | robust second-pass mass fit on the in-window inliers (sensitivity_plan P3.1); default true since 2026-10-11 |
| `count_fragments_once` | `false` | count each predicted fragment once per spectrum at its most intense matching peak (step 3); fragindex only; omitted from the canonical configuration at the default |
| `frag_tol_floor_ppm` | `5.0` | floor of the learned tolerance, in both passes |
| `frag_tol_mad_k` | `0.0` | `> 0`: the learned tolerance is `frag_tol_mad_k * 1.4826 * MAD` instead of `1.5 * P95`, and the second pass refits the offset only (step 6); omitted from the canonical configuration at the default; benchmark-gated |

`bucket_size` for the bucketed Library index is taken from `cfg.extract.bucket_size`,
not from `SearchSeedConfig` (`run.rs:225`). The 50 ppm mass-calibration search
window, the `1.5 * P95` scale, the `1.4826` MAD-to-sigma factor and the `>= 20`
deviation threshold are constants (`search_seed.rs`, `masscal.rs`), not config fields;
the floor is `frag_tol_floor_ppm`.

The standalone subcommand `Cmd::SearchSeed` (`main.rs:75-86`, dispatch `:458-476`)
takes `--ms2`, `--lib-precursors`, `--lib-fragments`, `--out`, and
`--config` (all `String`; `--config` optional). There is no `--bucket-size` flag;
`bucket_size` is read from the resolved config's `extract.bucket_size`
(`main.rs:473`).

## Invariants, determinism, gotchas

- **Determinism** (`docs/14_build_test_deploy_gotchas.md`): the fragindex parallel
  path is bit-identical to the *serial fragindex* best-per-candidate (this is a
  parallel-vs-serial claim about the same backend, not a fragindex-vs-bucketed
  claim; the two backends use different edge predicates, see the accumulation
  section). Groups run in `BTreeMap` key order, within a group scans run in
  RT-ascending order (guaranteed by `load_ms2`'s `sort_by rt_seconds`,
  `spectra.rs:95`) with a strictly-greater update, and the cross-group merge is a
  total order (`max score`, tie earliest RT, tie min scan_index;
  `search_seed.rs:392-409`). `select_peaks` re-sorts the top-N back to
  index-ascending (`:299`) so the `obs_sum` float reduction is summed in a fixed
  order. The `HashMap` is only ever reduced through this total order, never
  iterated for a float sum.
- **Best-per-candidate, not best-per-spectrum.** One output row per candidate that,
  in at least one scan, cleared `min_matched_peaks` **and** ranked within that
  scan's top `report_psms` (the per-scan sort is truncated before the fold). A
  candidate below `report_psms` in every scan it appears in is dropped. Ties in the
  update use strictly-greater (`score > entry.score`, `:97` and `:377`), so the
  earliest-RT scan wins a tie.
- **Library decoys only.** `label` comes straight from `Candidate::is_decoy`
  (`:138`); the stage never mints decoys. The target-decoy null therefore depends on
  the library carrying paired decoys (see the DIA-NN library recipe / `digest`).
- **obs_sum is observed intensity.** The seed drops predicted fragment intensity on
  purpose (`fragindex.rs:185-190`); do not "fix" the discarded `_pi`/`_mz`.
- **Mass-cal fallback is an identity.** With `< 20` deviations the offset is `0.0`
  and the tolerance is left at `fragment_tol_ppm` (`cal_passes = 0`); extract then
  applies no shift. `frag_ppm_sigma` is an exact copy of `frag_tol_ppm`.
- **The 50 ppm mass-cal window is fixed** and independent of `fragment_tol_ppm`; it
  is intentionally wide so a systematic error larger than the scoring tolerance is
  still observable. Do not tie it to `fragment_tol_ppm`.
- **Mass-cal assumes m/z-sorted peaks.** The deviation search uses
  `scan.peaks.partition_point(|pk| pk.mz < lo)` + a linear scan to the high edge
  (`search_seed.rs:168-178`), which is only correct if `scan.peaks` is m/z-ascending.
  `load_ms2` sorts scans by RT but never re-sorts peaks (`spectra.rs:20-97`), so the
  invariant is inherited from `convert`. The fragindex scoring path does not need
  it (it bins each peak independently); only mass-cal relies on it.
- **`config_hash` is carried but unused** inside `run` (part of `SearchSeedParams`
  for call-site uniformity, `search_seed.rs:34`). In the standalone CLI it is
  `blake3_str(cfg.canonical_json())` (`main.rs:466`); in the `run` orchestrator it
  is the shared chain hash `ch` (`run.rs:226`). Either way the report's
  `content_hash` is the blake3 of the output Parquet, not this value.
- **`top_n_peaks` is seed-only.** It caps the probe set to reduce index probing on
  the dominant cost, but abundant peptides supply the calibration anchors anyway;
  the downstream `extract` stage still sees all converted peaks.
- **`MatcherKind::Bucketed` stays serial** (`search_seed.rs:66-107`); only the
  fragindex path is parallelized.
- **iRT is inert here.** `predicted_irt` is only copied to the output, so the seed
  can be computed once and reused across the DeepLC fine-tune boundary
  (`run.rs:237-280`).
- **Test coverage.** Only `select_peaks` has a unit test here
  (`zero_selects_all_and_seed_cap_keeps_only_top_intensity_peaks`,
  `search_seed.rs:417-450`), asserting `top_n = 0` returns all indices and
  `top_n = 300` over 305 peaks keeps only the top-intensity `5..305` re-sorted to
  index order. `target_decoy_q` (tied-block, `+1` conservatism) and `entrapment_q`
  are tested in `fdr.rs` (`:145-207`); `ln_factorial` has no direct test. The
  fragindex match, epoch reset, precursor gate, and the naive-equivalence gate are
  tested in `fragindex.rs` (`:280-451`).
  There is no stage-level test for `search_seed::run`, the masscal output, or
  bucketed-vs-fragindex equivalence at the stage level (see docs/14_build_test_deploy_gotchas.md).

## How to extend / modify

- **Swap in a real Sage / sage-core scorer.** Replace the accumulation in `run`
  behind the existing file contract. Keep the `seed_psms` schema and
  `masscal.json` keys unchanged so `rt-im-train` and `extract` need no edits. Bump
  `model_identity` (`search_seed.rs:271`) so provenance in `report.json` is honest.
- **Add a `seed_psms` column.** Bump the schema version at `schema.rs:15`
  (`("seed_psms", 1)` -> `2`), push the new `Col` in the `write_table` call
  (`search_seed.rs:233-250`), and update any consumer that reads by column name.
- **Make the mass-cal window / thresholds configurable.** The 50 ppm search
  window, the `>= 20` minimum, the `1.5x` scale, and the `5.0 ppm` floor are
  literals in `run`; promote them to `SearchSeedConfig` fields (with conservative
  defaults) if you need to tune them, per the "every algorithmic choice is a typed
  config field" convention.
- **Add a matcher backend.** Add a `MatcherKind` variant (`config.rs:38`) and a
  branch in the `if let Some(idx) = fidx` dispatch (`search_seed.rs:63-108`); mirror
  the deterministic merge if the new path is parallel.
- **Charge-2 / m/z-binned tolerance.** The current fit produces one global offset
  and tolerance. To make them charge- or m/z-dependent, partition `devs` before
  `masscal::fit` and emit per-bin entries in `masscal.json`, then teach `extract` to
  pick the matching bin.
- **Change the score.** `hyperscore` (`search_seed.rs:413`) is a free function; keep
  it monotone in matched-fragment count and observed intensity so the target-decoy q
  ordering stays meaningful, and keep the summation order fixed for determinism.
