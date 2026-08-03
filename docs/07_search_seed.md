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
| `rust/mumdia/crates/mumdia/src/spectra.rs` | `load_ms2` (reads the converted MS2 Parquet into `Ms2Scan`) |
| `rust/mumdia/crates/mumdia-core/src/constants.rs` | `ppm_bounds`, `ppm_diff`, `within_ppm` (the shared ppm math) |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `SearchSeedConfig` (`:286`), `MatcherKind` (`:38`) |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | `artifact::SEED_PSMS = ("seed_psms", 1)` (`:15`) |
| `rust/mumdia/crates/mumdia/src/main.rs` | CLI `Cmd::SearchSeed` (`:75`, dispatch `:458`) |
| `rust/mumdia/crates/mumdia/src/stages/run.rs` | orchestrator wiring (`:219`), masscal handoff to extract (`:309`) |

## Inputs and outputs

### Consumed

- **MS2 spectra** (`--ms2`, a `convert` output Parquet), loaded by `load_ms2`
  into `Vec<Ms2Scan>`. Each `Ms2Scan` carries `scan_index`, `id`, `rt_seconds`,
  an `IsolationWindow { target_mz, lower_mz, upper_mz, im_lower, im_upper }`, and
  `peaks: Vec<Peak { mz: f64, intensity: f32, ion_mobility }>`.
- **Library** (`--library-precursors` + `--library-fragments`), loaded by
  `Library::load` (`index.rs:54`). Provides `cands: Vec<Candidate>`
  (`candidate_id`, `peptidoform`, `charge`, `precursor_mz`, `base_peptide_id`,
  `protein`, `is_decoy`, `predicted_irt: f32`, `frag_start`, `n_frag`), the flat
  fragment arrays (`frag_mz`, `frag_int`, `frag_name`), and the bucketed inverted
  index for `page_search`.
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

**`<out>.masscal.json`** (written at `search_seed.rs:217-227`):

| key | type | meaning |
|---|---|---|
| `frag_ppm_offset` | f64 | median signed ppm of `observed_peak` relative to `predicted_fragment` (`ppm_diff(peak_mz, fmz)`), i.e. the systematic offset |
| `frag_tol_ppm` | f64 | learned tolerance: `max(5.0, 1.5 * P95(|dev - offset|))` |
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
`fdr_seed`; note `matcher` and `two_pass_mass_cal` are *not* recorded in `params`),
a `stats` map (`psms`, and `targets_at_q<fdr_seed>` whose key is the float-
formatted threshold, e.g. `targets_at_q0.01`), `model_identity =
"native-seed-hyperscore-v1"`, and `elapsed_ms`.

## How it works

Entry point: `search_seed::run(SearchSeedParams)` (`search_seed.rs:45-283`).

**1. Load** the library (`Library::load`, `index.rs:54`) and MS2 scans
(`load_ms2`, `spectra.rs:20`), then log candidate and scan counts
(`search_seed.rs:47-53`). `load_ms2` sorts the returned `Vec<Ms2Scan>` by
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
  (`fragindex.rs:214-237`), candidates with `count >= min_matched_peaks` are
  scored by `hyperscore`, sorted by score desc then candidate-id asc, truncated to
  `report_psms`, and folded into a group-local best with a strictly-greater update
  (`:357-385`).
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

**6. Fragment mass recalibration** (`search_seed.rs:150-231`). A
`scan_index -> &Ms2Scan` map is built (`:155-158`). For every **confident target**
PSM (`!is_decoy && q <= fdr_seed`, `:161`), each library fragment m/z of that
candidate (`lib.cand_frags(cid)`, `index.rs:209`) is searched against the best
scan's peaks inside a **hardcoded 50 ppm window** (`ppm_bounds(fmz, 50.0)`,
`:167`; `partition_point` finds the low edge, then a linear scan to the high edge).
This `partition_point` + linear scan **assumes `scan.peaks` is m/z-ascending**;
`load_ms2` does not re-sort peaks by m/z (`spectra.rs:20-97`), so the assumption
rests on `convert` writing peaks in m/z order. The nearest peak by absolute m/z
distance contributes one signed ppm deviation (`ppm_diff(peak_mz, fmz)`,
`constants.rs:66`) to `devs` (`:160-184`). Every library fragment of the candidate
is probed, not only those that matched during scoring, so a fragment that found no
scoring posting can still supply a calibrant. The 50 ppm net is deliberately wider
than `fragment_tol_ppm` so a systematic offset larger than the tolerance can still
be measured.

The fit closure (`:187-194`) takes deviations, sorts them, takes the median as the
offset (`sorted[len/2]`, the upper-middle element for even-length inputs, not the
two-element average), then sets `tol = max(5.0, 1.5 * P95(|dev - offset|))` using
`calibrate::percentile(.., 0.95)` (nearest-rank on `round(p*(n-1))`,
`calibrate.rs:156-164`). Control flow (`:195-216`):

- `devs.len() < 20`: fallback, `(0.0, fragment_tol_ppm, cal_passes = 0)` (no
  offset, tolerance unchanged).
- otherwise single pass: `(o1, t1, 1)`.
- if `two_pass_mass_cal`: keep only deviations inside `|dev - o1| <= t1` and, when
  `>= 20` survive, re-fit to `(o2, t2, 2)`; otherwise keep the single-pass result.
  The second pass rejects random-match outliers so they cannot bias the median.

The result is written to `<out>.masscal.json` (`:217-227`).

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
| `run` | `search_seed.rs:45` | stage entry point; orchestrates load, score, q, masscal, write |
| `SearchSeedParams` | `search_seed.rs:27` | input struct (`ms2`, `library_precursors`, `library_fragments`, `out` output-path prefix, `cfg`, `bucket_size`, `config_hash`) |
| `Best` | `search_seed.rs:37` | per-candidate best `{ score, rt, matched, scan_index }` |
| `seed_fragindex_windows` | `search_seed.rs:313` | parallel per-window fragindex scoring + deterministic merge |
| `select_peaks` | `search_seed.rs:288` | top-N-by-intensity peak selection, re-sorted to index order |
| `hyperscore` | `search_seed.rs:413` | `ln(matched!) + ln(1 + sum_obs)` |
| `FragIndex::build` | `fragindex.rs:46` | build CSR inverted index at a fixed tolerance |
| `FragIndex::probe_peak` | `fragindex.rs:152` | matched postings for one peak in a candidate range |
| `SeedScratch::accumulate` | `fragindex.rs:214` | epoch-stamped fused `(count, obs_sum)` accumulation |
| `Library::candidate_range` | `index.rs:238` | half-open candidate id range for an isolation window |
| `Library::page_search` | `index.rs:247` | bucketed inverted-index probe (bucketed backend) |
| `Library::cand_frags` | `index.rs:209` | `(m/z, intensity, name)` slices for a candidate |
| `target_decoy_q` | `fdr.rs:7` | tied-block, monotonized `(n_decoys+1)/max(1,n_targets)` q |
| `count_targets_at_q` | `fdr.rs:115` | target count at or below a q threshold |
| `ln_factorial` | `fdr.rs:137` | `ln(n!)` via summed logs |
| `ppm_diff` / `ppm_bounds` | `constants.rs:66` / `:78` | signed ppm (theoretical-relative) and ppm window bounds (query-relative) |
| `within_ppm` | `constants.rs:92` | min-relative tolerance predicate used by the fragindex match (differs at the edge from the two above) |
| `load_ms2` | `spectra.rs:20` | reads `spectra_ms2.parquet` into `Vec<Ms2Scan>`, RT-sorted |
| `percentile` | `calibrate.rs:156` | nearest-rank percentile (used for the tolerance) |

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
| `two_pass_mass_cal` | `false` | robust second-pass mass fit on the in-window inliers (sensitivity_plan P3.1) |

`bucket_size` for the bucketed Library index is taken from `cfg.extract.bucket_size`,
not from `SearchSeedConfig` (`run.rs:225`). The 50 ppm mass-calibration search
window, the `min 5.0 ppm` tolerance floor, the `1.5 * P95` scale, and the `>= 20`
deviation threshold are hardcoded in `search_seed.rs` (`:167`, `:192`, `:195`),
not config fields.

The standalone subcommand `Cmd::SearchSeed` (`main.rs:75-86`, dispatch `:458-476`)
takes `--ms2`, `--library-precursors`, `--library-fragments`, `--out`, and
`--config` (all `String`; `--config` optional). There is no `--bucket-size` flag;
`bucket_size` is read from the resolved config's `extract.bucket_size`
(`main.rs:473`).

## Invariants, determinism, gotchas

- **Determinism** (PLAN.md Section 7): the fragindex parallel path is bit-identical
  to the *serial fragindex* best-per-candidate (this is a parallel-vs-serial claim
  about the same backend, not a fragindex-vs-bucketed claim; the two backends use
  different edge predicates, see the accumulation section). Groups run in `BTreeMap`
  key order, within a group scans run in RT-ascending order (guaranteed by
  `load_ms2`'s `sort_by rt_seconds`, `spectra.rs:95`) with a strictly-greater
  update, and the cross-group merge is a total order (`max score`, tie earliest RT,
  tie min scan_index; `search_seed.rs:392-409`). `select_peaks` re-sorts the top-N
  back to index-ascending (`:299`) so the `obs_sum` float reduction is summed in a
  fixed order. The `HashMap` is only ever reduced through this total order, never
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
  and tolerance. To make them charge- or m/z-dependent, partition `devs` before the
  `fit` closure and emit per-bin entries in `masscal.json`, then teach `extract` to
  pick the matching bin.
- **Change the score.** `hyperscore` (`search_seed.rs:413`) is a free function; keep
  it monotone in matched-fragment count and observed intensity so the target-decoy q
  ordering stays meaningful, and keep the summation order fixed for determinism.
