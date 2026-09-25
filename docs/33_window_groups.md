# 33. Searching a run one isolation-window group at a time

`groups.window_groups` (default `1`) makes `mumdia run` search a run as a sequence of
independent, smaller searches, one per group of isolation windows, and pool the results
before rescoring. It exists for one reason: on a large library (the 8-12-mer
immunopeptidomics library has 142.7 million precursors and 1.69 billion fragments) the
monolithic search holds the whole library, the whole hit accumulator and every accepted row
at once, and that is what pushed a single-file search to 335 GB and the serial extract of
the same file past a 471 GB commit limit. A DIA isolation window can only select precursors
whose m/z lies inside it, so a group of windows and the library band under it form a closed
search that needs nothing outside the band. Searching group by group bounds the memory to
one band's worth by construction, without changing what any stage computes.

This page is the code-grounded description: how the groups are planned, what each band's
stages see, what the two pooling stages do, the two calibration modes, the output layout,
and what has been measured. `groups.rs`, `stages/run_groups.rs`, `stages/seed_pool.rs` and
`stages/pool.rs` are the sources; `index.rs` (`Library::load_with_fragment_offset`) and
`mumdia-io/src/table.rs` (`TableFile::open_rows`, `row_group_stats`) are the loads it rests
on.

## 1. Configuration

```json
"groups": {
  "window_groups": 8,
  "calibration": "global"
}
```

| Field | Default | Meaning |
|---|---|---|
| `window_groups` | `1` | Number of groups. `1` is the ordinary single-library search. Clamped to the number of distinct isolation windows; groups whose band selects no precursor are merged into a neighbour, so the number actually searched can be smaller (the log and `groups/plan.json` say how many). |
| `calibration` | `global` | Whose seed anchors calibrate each group's retention time and mass: `global` pools every group's seeds first (section 4); `per_group` uses the group's own. |
| `parallel` | `1` | Bands in flight at a time, through a bounded queue that starts the most expensive band first (section 8, "Scheduling the bands"). Clamped below the thread count. |
| `rt_adaptation` | `per_band` | `once_per_run` adapts the library's retention times in one DeepLC worker per run over the union of the bands, under `global` calibration (section 4b). Float-equivalent, opt-in. |
| `balance` | `precursors` | `cost` balances the cuts on precursors times MS2 peaks per window instead (section 2). Output-changing, opt-in. |
| `delete_band_intermediates` | `false` | Delete each band's `psms_extracted` and `features` tables once the pool is written (section 6). Disk only. |
| `pool_competed` | `true` | `false` leaves the competed rows per band and has rescore read the band tables with a table-to-source map, where that cannot change a result (section 5). `psms_scored.parquet` is byte-identical; the pooled competed table is not written. Opt-in. |
| `pool_chromatograms` | `true` | `false` leaves the chromatograms per band and has quant read the band tables with the overlap losers the pool persists to `groups/overlap_losers.parquet` (section 5). The quant tables are byte-identical; the pooled chromatogram table is not written, and the band directories become the run's only chromatograms (section 6). Opt-in. |

`run-experiment` (and `run` with several `--mzml`, which dispatches to it) searches every run
grouped, with one experiment-wide rescore over the pooled competed tables and the
per-source `run_psm_q` (`docs/11_compete_rescore_fdr.md`). Under
`experiment.rt_library_scope = first_run_only` the runs after the first reuse the first
run's adapted bands (section 3).

## 2. Planning the groups

The run's isolation windows come from `spectra/isolation_windows.parquet` (`lower`,
`upper`, written by convert). The library's precursor table is m/z-sorted with
row-aligned `candidate_id` (a hard invariant, `index.rs`), so its parquet row groups carry
usable footer statistics: rows, minimum and maximum `precursor_mz`. `groups::plan` reads
those statistics (`TableFile::row_group_stats`, no table read), estimates for every window
how many precursors it selects by pro-rating each row group over the window's range, and
cuts the windows, in ascending m/z, into `window_groups` contiguous groups whose estimated
precursor counts are as equal as a greedy cut makes them. Balancing by precursors rather
than by windows matters: on the immunopeptidomics library the per-window precursor counts
differed by 3x across the m/z range, and equal window counts would have made the memory
peak the largest band's, not the average's.

`groups.balance = cost` (default `precursors`) balances the cuts on an estimate of the search
cost instead: per window, the precursors it selects times the MS2 peaks its scans carry
(`groups::window_costs`; the run's MS2 is decoded before the plan for this, and for the
dispatch order, section 8). Band cost follows spectral density more than precursor count:
on the immunopeptidomics search two bands of 2.98M and 3.03M precursors took 42 s and 460 s,
and at 81-94 bands the slowest single windows (418-460 s) set the floor of the run. The
bands' `est_precursors` and the merge of empty bands are unchanged, and `plan.json` records
`"balance": "Cost"`. The setting moves the cuts, and with them which candidates sit at a band
edge and the order of the pooled rows the classifier sees, so it is output-changing and
opt-in; validate it as a band-count change (section 8): peptides at 1% inside the seed spread
against `precursors`, and the per-band wall times, on two acquisitions. Not measured.

A band is the union of its windows' m/z ranges. Two consequences are written to
`groups/plan.json`:

- `est_unselectable`: precursors outside every window (charge states above the top window,
  typically). They belong to no band and are never loaded, which is the right outcome for
  something the run cannot measure. The ungrouped search loads them and extracts nothing.
- `est_duplicated`: precursors in the overlap of two adjacent windows across a cut. They are
  searched in both bands; the pool keeps one row per candidate (section 5). Overlapping
  window schemes therefore cost the overlap twice and nothing else.

## 3. One band, one library

A band is the precursor rows `[first, first + n)` of the m/z-sorted library whose m/z lies in
the band's range (`Library::precursor_row_span`, from the row-group statistics and one decode
of `precursor_mz` over the boundary groups). The band's first library row is its offset;
local id plus offset is the library-wide id, and every stage below carries it in the manifest
as `name[gNN]`.

The stages then run unchanged on that band, as if it were the library:

- `search-seed` loads the band straight from the library by row span
  (`Library::load_row_span_with`: local ids `0..n`, the fragments whose ids lie in the band's
  range read from the shared, library-wide fragment table). When that table is sorted by
  `candidate_id` at row-group granularity (`scripts/sort_fragments.py`; every library writer
  ends with the sort) the read is selective; an unsorted table still loads through a filtered
  scan of the whole table, with a warning, and costs the scan per band.
- the RT model (multi-head calibration, the optional fine-tune, or the base-model
  re-prediction, whichever the configuration resolves to; `docs/08_rt_im_train.md`) runs
  on the band table and writes the band's re-predicted table. The DeepLC sidecars rewrite a
  precursor FILE, so a band that an RT model adapts is first written out as a precursor table
  of its own, `groups/gNN/lib_precursors.parquet`, with band-local ids `0..n`
  (`groups::write_band_slice`, which reads only the row groups that cover the span through
  `TableFile::open_rows`).
- `rt-im-train` writes the band's `run_windows.parquet` and `cal.json`.
- `extract`, `features` and `compete` write the band's `psms_extracted`, `chromatograms`,
  `features` and `psms_competed`. The id columns are library-wide on the way out
  (local id plus `Library::global_offset`), so a band's table means the same thing as the
  run's and pooling is a concatenation.

`rt-im-train` and `extract` read the adapted band file where an RT model ran, and the band by
row span from the library where none did (`rt_model` `library`, or the bands keeping a
re-prediction the caller made, section 4b). A span load is the same library, value for value,
as a band file of the same rows (`band_slice_file_loads_with_the_fragment_offset_and_matches_the_range_load`
in `index.rs`), and every artifact of a grouped fixture run is byte-identical either way.
Until 2026-09-25 every band was written out before its seed whether or not anything rewrote
it: on a 203.5M-precursor library searched without an RT model, that was the whole precursor
table decoded and re-encoded once per run, for a file only the seed, `rt-im-train` and
`extract` read.

Nothing outside the band is resident during any of these, except the run's spectra, which
are decoded once for all the bands (next section).

Under `experiment.rt_library_scope = first_run_only` the runs after the first reuse the first
run's adapted bands, and they need no slice at all: the seed reads the band by span, and the
adapted table replaces it everywhere else. Where every run adapts its own bands
(`rt_library_scope = per_run`), a run after the first takes the first grouped run's slices
(`GroupRun::slices_from`) when the two `groups/plan.json` list the same bands: a slice is a
deterministic function of the library and the row span, so rewriting it would produce the
same bytes. On a 203M-precursor library that is the whole precursor table not written, per
run after the first.

### The run's spectra, decoded once per phase

Every band searches the whole run and differs only in its slice of the library, so the
scans are the same bytes for all of them. `run_groups` therefore decodes them once for all
the bands and lends the buffers: MS2 for the seeding phase
(`search_seed::SearchSeedParams::ms2_scans`), MS2 and MS1 for the extraction phase
(`extract::ExtractParams::scans`, which carries both). `None` there means "open the file
yourself", which is what a standalone `mumdia search-seed` or `mumdia extract` and an
ungrouped `run` still do. An EMPTY lent slice also means that: the stage decodes the named
path rather than believing that the run has no such spectra, so lending an empty MS1 while
still passing `--ms1` cannot silently drop every MS1 feature.

Once per phase and not once per run. Nothing between the seeding and the extraction phase
reads the spectra, and what does sit there is the per-band retention-time phase, one DeepLC
sidecar process per band, sequentially, 63 of them on the 63-band run. Holding a ~1 GB
decoded MS2 buffer across that is the resident set the banding exists to bound, so the
extraction phase pays for a second decode of the same artifact instead. That is two decodes
per run against the `2m` this replaces, and `run.rs` declines to share for the same reason
in the ungrouped single-run case. Both buffers are dropped at the end of the extraction
phase, before `pool::run` reads and rewrites the run's largest artifacts.

What makes the sharing safe is NOT that nothing corrects the observed m/z. The per-run mass
recalibration corrects exactly that -- `peak.mz / mass_off.factor_at(peak.mz)` -- and under
`groups.calibration = per_group` each band applies its own factor to the same buffer. It is
safe because that corrected value is computed into a local (`q_mz`) at every call site and
never written back; because the stages take a shared slice they cannot write through;
because `load_ms2` and `load_ms1` sort by retention time before returning, so no stage
re-sorts; because `select_peaks` returns peak indices instead of truncating `scan.peaks`;
and because the destructive peak-claim strategies rewrite the band's own `Hit` intensities,
not the spectra.

Two of those are one edit away from being false, and both edits are the obvious
optimisation of their call site. Hoisting `q_mz` out of the per-peak loop by writing it
back into the scan would bake band g00's factor into the peaks band g01 reads. Truncating
`scan.peaks` in `select_peaks` instead of building an index vector per scan per band would
hand the following band, and extract, capped spectra; a 300-peak cap costs 60% of the
peptides on a 50-window Orbitrap DIA run (`docs/04_convert.md`). The borrow checker refuses
both today, because the scans arrive as `&[Ms2Scan]`; `run_groups::scan_fingerprint` covers
what it does not (interior mutability, `unsafe`, and the second decode differing from the
first) with a `debug_assert` that costs nothing in a release build. The fingerprint mixes
every field the stages read, so it followed `Ms2Scan` when the `id` was deleted and when
`Peak.mz` was narrowed to `f32`; it is an internal digest compared between two decodes of
one artifact, never a stored value, so its numeric value is free to change.

What it removes, for `m` bands with `p` of them in flight:

| | decodes per run | MS2 resident | MS1 resident |
|---|---|---|---|
| before | `2m` MS2, `m` MS1 | `p` copies while seeding, `p` while extracting | `p` copies while extracting |
| after | 2 MS2, 1 MS1 | 1 copy while seeding, 1 while extracting | 1 copy while extracting |

Neither buffer is resident during the retention-time phase, or during pooling, in either
column. The saving scales with `groups.parallel` and nothing is traded for it: at
`parallel: 8` the extraction phase holds 1 GB of MS2 instead of 8, at `parallel: 48` (a
100-band run on a smaller library) 1 GB instead of 48.

Measured on the fixture at `window_groups: 3, parallel: 2`, the load-ms2 and load-ms1 stage
lines go from 6 MS2 decodes and 3 MS1 decodes to 2 and 1, and the run's artifacts are
byte-identical to the pre-change binary's under both `calibration: global` and
`calibration: per_group`, which is the mode in which the bands apply different mass offsets
to the one shared buffer.

At production scale one run's MS2 was about 1 GB decoded (301,127 scans on the immuno data,
293,271 on the Astral data), and 63 bands were 126 decodes of it. Since `Peak` was narrowed
to two `f32` and `Ms2Scan.id` deleted it is about a third of that: measured on
`LFQ_Orbitrap_AIF_Ecoli_01` (465,806 scans, 41,293,465 MS2 points, the engine's own
`mem: ms2 scans` report), 0.968 GiB before and 0.342 GiB after.

It also removes allocations, which is the failure the banding is up against: the engine
dies at the kernel's per-process mapping limit (1,048,576; the live grouped run peaked at
556,573 mappings and 244 GB). A decoded MS2 scan used to be two heap blocks, its
`Vec<Peak>` and its id `String`, so one copy of a 301,127-scan run was about 602,000 blocks
and eight concurrent copies about 4.8 million. Deleting the id halves that to one block per
scan. How many of those become distinct mappings depends on the allocator's size classes
and is not measured here; the block count is exact.

`ci/smoke.sh` runs a grouped arm (`window_groups: 3, parallel: 2, calibration: per_group`)
so that this path, and `run_groups` generally, has regression cover: without it every
assertion in the suite is about the ungrouped `scans: None` arm, which is a no-op.

## 4. The seed pool and the two calibration modes

Each band's seed has two problems as a calibration anchor set. Its `spectrum_q` was
estimated on the band's PSMs alone, and the q floor is `1/T` (`docs/11_compete_rescore_fdr.md`), so a
small band cannot reach 1% at all: on the CI fixture, 179 seed PSMs give every one of them
q = 0.0056 as one pool and none of them q < 0.01 as three bands of 73, 67 and 39. And its
ids are local.

`seed-pool` (`stages/seed_pool.rs`) reads every band's seed, maps the ids to library-wide
ones, keeps the higher-scoring row where two bands searched the same candidate, re-estimates
`spectrum_q` over the union with the seed's own target-decoy kernel (`fdr::target_decoy_q`),
and writes the run-level `seed_psms.parquet`. Beside it goes
`seed_psms.parquet.masscal.json`, the run's fragment mass calibration, fitted once over the
bands' calibrant deviations (next subsection). It also writes each band's view of the
pooled seed, `groups/gNN/seed_psms_pooled.parquet`: the band's own rows, local ids,
pooled q.

### 4a. The mass calibration is fitted once, not averaged

The tolerance `search-seed` learns is `1.5 * p95(|dev - median|)` over the ppm deviations of
the matched fragments of its confident target PSMs (`docs/07_search_seed.md`). Combining the
bands' fitted SCALARS is not that estimator, and it is systematically wider. Measured on the
six-file HYE Astral benchmark, the same library and the same retention-time model, 100 bands
against none:

| | unbanded | 100 bands, scalars combined |
|---|---|---|
| `frag_ppm_offset` | -1.8486 | -1.8834 |
| `frag_ppm_sigma` | 8.452 | 11.400 |
| `n_dev` | 181,196 | 200,257 |
| `ppm_residual_mad` | 0.907 | 1.035 |
| candidates accepted by extract | 4,986,153 | 6,609,984 |
| peptides at 1% | 113,860 | 110,006 |
| precursors at 1% | 126,436 | 121,966 |

The offsets agree and the tolerance is 35% wider, extract then accepts 33% more candidates
and the run returns 3.4% fewer peptides. Causal rather than correlated: one band extracted
twice, identical in every input but the calibration file it was handed, accepted 12,414
candidates at 11.40 ppm and 8,791 at 8.45; and an unbanded control on the same adapted
library returned the unbanded arm's 113,860 peptides exactly.

Two things compound. A p95 estimated on one band's ~2,000 deviations has a heavier tail than
the p95 of the union, and a mean of those p95s does not recover it. And each band selects
its calibrants on its OWN `spectrum_q`, which is not the pooled one: on that benchmark it is
looser (106,088 band-confident PSMs against 97,584 pooled).

So the bands write their deviations and the pool fits them, exactly as the retention-time
calibration already uses the pooled anchors under `groups.calibration = global`:

- `search-seed`, asked for it (`SearchSeedParams::emit_calibrants`, set only by the grouped
  path), writes `<seed>.masscal.parquet` beside the masscal: `candidate_id` (LIBRARY-WIDE,
  the band's local id plus its fragment offset), `scan_index`, `frag_mz` and `ppm`, one row
  per matched fragment of every target PSM of the band. 16 B per deviation; the size is
  bounded below.
- `seed-pool` reads them, keeps the deviations whose PSM the POOLED q accepts at
  `search_seed.fdr_seed`, and fits `masscal::MassCal::fit_from` -- the same function
  `search-seed` calls -- once. A deviation is kept only from the band whose row the pool
  kept for that candidate, and only for that row's scan. A precursor in the overlap of two
  windows across a cut is loaded by both bands, and each band serves both windows for it,
  so both bands usually hold the same PSM with the same deviations; the band index is what
  makes it contribute one PSM's fragments, as in an ungrouped seed. Keyed on the scan
  alone, as it was until 2026-09-25, those deviations were counted once per band.
  `two_pass_mass_cal` and the `mass_cal_loess` grid work on this path, because both read the
  deviations rather than a scalar.
- The band's own `<seed>.masscal.json` is unchanged: still fitted on that band's confident
  targets alone, and still what `groups.calibration = per_group` extracts with.

The band's q is not the pooled q in either direction, so the sidecar carries more than the
band's own selection: EVERY target PSM of the band is offered whatever its own q says, and
the pooled q decides. That matters where the band q is STRICTER, which is the `1/T` case
above: on the CI fixture at three bands, every band's own q rejects every one of its
targets, so a strictly q-selected sidecar would be empty in all three. With the offer, the
three bands contribute 437, 401 and 230 deviations, the pool selects all 1,068 of them and
fits `frag_tol_ppm` 5.0 at offset 0.0 -- the same calibration, to the digit, that the
ungrouped search of the same spectra produces. Before this the same run calibrated nothing
and extracted at the configured 20 ppm.

Until 2026-09-25 the offer was a fixed prefix, each band's 2,000 best targets. No rule a
band can apply to its own data gives a superset of what the pooled q accepts, because the
pooled threshold moves with the other bands: a band of clean, high-scoring targets lowers
it, and the pool then accepts targets of a noisier band at scores where that band's own q is
well above the threshold. The prefix was sized from the 100-band benchmark (about 1,000
pooled-accepted targets per band) and was exact from 8 bands up, and short below it. On the
six-file HYE Astral benchmark, 2026-09-24:

| bands | calibrant deviations | `frag_tol_ppm` |
|---|---|---|
| unbanded | 181,196 | 8.45 |
| 2, 2,000-target prefix | 167,418 | 7.76 |
| 4, 2,000-target prefix | 177,380 | 8.24 |
| 8 and more, 2,000-target prefix | 181,196 | 8.45 |

The missing deviations belonged to the lower-scoring accepted targets, which are the
noisier ones, so the short fits came out narrower rather than wider. With every target
offered, and each accepted candidate taken from one band, the pooled fit is the unbanded
fit at any band count and with overlapping windows; `ci/smoke.sh` checks this at two bands
on the fixture, and `tests/pipeline.rs`
(`a_two_band_pooled_mass_calibration_equals_the_unbanded_fit`) on a two-band library built
so that one band's own q rejects hundreds of targets the pooled q accepts, which the old
prefix failed by 2,000 deviations of 30,000. Its second arm makes the two windows overlap
across the cut (400-501 and 500-600, 13 shared targets): keyed on the scan alone the pool
counted their deviations twice, 30,052 against the unbanded 30,000.

The sidecar stays 16 B per deviation, now for every target of the band, and its size
follows the band's scans rather than its library. The seed keeps one row per candidate and
each MS2 scan contributes at most `search_seed.report_psms` candidates (5 by default), and
a row adds at most one deviation per library fragment of its candidate. A band therefore
holds at most `served scans x report_psms x fragments per candidate` deviations whatever its
precursor count: for a run of 100,000 MS2 scans, 5 rows per scan and 12 fragments per
candidate, 6M deviations and 96 MB over all its bands together, which is a bound and not a
measurement. The band's `search-seed` holds them in memory until the write, which moves the
columns rather than copying them. `seed-pool` decodes one band's sidecar at a time. Each
band's seed report records `calibrant_deviations`, `calibrant_bytes` and
`calibrant_target_rows`, and the seed-pool log line records `band_deviations`,
`band_deviation_bytes` and `largest_band_bytes`. On the CI fixture at two bands that is 616
and 452 deviations (9,856 and 7,232 B, from 103 and 76 target rows), all 1,068 of them
accepted. Not yet measured at scale: see section 10.

`masscal.json` gains `masscal_source`, which reads `pooled_deviations` or `band_scalars`. A
band directory seeded before the sidecar existed has none, and the pool then combines the
scalars as it always did, naming the missing groups in a warning; if `mass_cal_loess` is set
it warns separately that the grid cannot be recovered from scalars. A band with no
calibrants wrote the configured tolerance in place of a learned one, and when the pooled fit
has fewer than `masscal::MIN_CALIBRANTS` deviations the pool keeps that tolerance rather
than fitting a percentile of a handful of points, with a warning.

`groups.calibration` decides which anchors each band's calibration sees:

| | `global` (default) | `per_group` |
|---|---|---|
| RT model (multi-head / fine-tune) | the pooled seed, joined by peptidoform | the band's seed |
| `rt-im-train` anchors | the pooled seed, iRT read from the seed row (`anchor_irt_from_seed`), one fit shared by every band | the band's seed, iRT joined from the band table |
| `extract` mass calibration | the pooled `masscal.json` | the band's |
| `features` seed corroboration and confident elution boundary | the band's view of the pooled seed | the band's seed |

Under `global`, after a per-band re-prediction of the library iRT, the pooled anchors carry
the library's iRT as it was when seeded; `seed_pool::refresh_irt` then copies each anchor's
new value from its band's table into `seed_psms_calibrated.parquet`, which is what
`rt-im-train` reads. The RT fit is then identical across bands (same anchors, same model),
and only the windows differ, because the precursors do. So it is fitted once:
`rt_im_train::fit_from_seed` reads the pooled (or refreshed) seed and fits the curve, the
window width and the optional held-out and adaptive sizing, and every band only applies that
fit to its own table (`rt_im_train::apply`), writing its `run_windows.parquet` and a
`cal.json` identical to the one it wrote when it fitted for itself
(`one_fit_applied_writes_what_the_whole_stage_writes`). Before, each band decoded the pooled
seed and refitted the same curve, one seed decode and one fit per band on the band loop's
critical path; the log now shows one `stage=rt-fit` line per run. `per_group` fits per band,
as before.

Refitting per band is sound because the multi-head ridge and the base-model re-prediction
are deterministic in their anchors: the same pooled anchors give the same head selection,
the same ridge and the same predictions, so every band carries one model. What it costs is
one DeepLC worker per band, which section 4b below removes. The optional DeepLC fine-tune is not deterministic and
would be trained once per band, so a grouped run refuses `rt_im_train.finetune_deeplc`;
fine-tune the library once beforehand (`docs/08_rt_im_train.md`, once per library) and
search that table.

`per_group` is kept for the comparison the design asked for, not as a recommendation: it is
one pooling pass cheaper and every group is independent, but each group fits on a fraction
of the anchors, and the fit quality sets the RT window that the extract of every group then
pays for. On the CI fixture it cannot fit at all (no band has a confident anchor, for the
`1/T` reason above); on a real run every band has thousands.

### 4b. One retention-time adaptation per run (`groups.rt_adaptation`)

The per-band refit above is sound but not cheap. Each band's DeepLC sidecar starts an
interpreter, imports torch and DeepLC, reads the pooled seed, refits the same heads on the
same anchors (head 2503 in every band of the HYE sweep), and predicts every sequence of its
own band. The charge states of one peptidoform sit at `(M + z * 1.007) / z`, which differ by
a factor of at least 1.33, so they usually fall in different bands and each of those bands
predicts the sequence again: the 10.9M HYE precursor rows are 4.91M unique sequences
unbanded. Measured on HYE Astral (2026-09-24), the multi-head step took about 13 min
unbanded and 19-24 min at 2-16 bands.

`groups.rt_adaptation = once_per_run` (default `per_band`, the behaviour before the setting
existed) runs one worker per run under `calibration = global`:
`deeplc_finetune.py - <seed> - --bands <tsv>`, where `groups/rt_bands.tsv` lists every
band's table and output. The worker fits the multi-head calibration once, predicts the union
of the bands' unique sequences once, in the order the bands list them, and rewrites each band
with the rule a single table gets, writing `groups/gNN/lib_precursors_multihead.parquet` (or
`_deeplc` for the base-model re-prediction) and its `.summary.json` under the names a
per-band run uses. The shared-band reuse of later runs, `seed_pool::refresh_irt` and the
manifest records are therefore unchanged. `per_group` keeps one sidecar per band, fitted on
the band's own anchors.

The setting also covers a re-prediction the caller has already made. With the multi-head
calibration off and `library_irt` resolving to DeepLC, `run-experiment` re-predicts the
imported library once for the whole experiment, and each band of each run then re-predicted
its slice of that table again, because `experiment.rt_library_scope` shares only an
adaptation (a fine-tune or the multi-head calibration). Under `once_per_run` the bands keep
the experiment-level values (`GroupRun::library_irt_repredicted`); under `per_band` they
re-predict as before.

Output effect: float-equivalent to `per_band`, not bit-identical. A sequence is predicted
once, in different company from its per-band prediction, and torch's CPU kernels round by
batch. Measured on the smoke fixture with DeepLC 4.5.0 on CPU (3,820 precursors, 4 threads):

| | `per_band` | `once_per_run` |
|---|---|---|
| 3 bands, multi-head 80: sequences predicted | 2,954 (1,174 + 1,310 + 470) | 1,910 |
| same, wall of the run | 25 s | 12 s |
| same, selected heads | identical | identical |
| same, largest per-row change of `predicted_irt` | | 0.87 s |
| same, PSMs at `q_value` 1% / peptides at `peptide_q_value` 5% | 154 / 152 | 156 / 152 |
| `run-experiment`, 2 runs of 2 bands, multi-head off: wall | 39 s | 9 s |
| same, bands re-predicted | 4 | 0 |
| same, largest change of a band value against the experiment-level one | 6.1e-5 s | 0 |

On a synthetic library one `--bands` call over contiguous bands writes exactly the
whole-library column band by band, under both the multi-head calibration and the base model
(`tests/python/test_deeplc_predict.py`,
`test_bands_write_the_whole_library_column_band_by_band`), so the union call reproduces what
an unbanded run predicts. Before defaulting it on: the per-band max |delta `predicted_irt`|
and the selected heads against `per_band`, and peptides at 1% on a banded HYE arm inside the
seed spread, on two acquisitions.

## 5. The artifact pool

`pool` (`stages/pool.rs`) appends the band tables, batch by batch, into the standard
`chromatograms.parquet` and `psms_competed.parquet`. The bands already wrote library-wide
ids, so this is a concatenation rather than a rewrite. Where two bands hold the same
candidate (window overlap across a cut), the competed row with the higher `prelim_score`
wins and the loser's rows are dropped from every pooled table, so the pooled tables hold
each candidate once and rescore's competition is unchanged. One batch is resident at a time,
and the writer's row groups are capped, so the pool costs one read and one write of the
group artifacts and no more memory than any single stage.

Two of the four tables an ungrouped run writes are not pooled, because nothing reads them:

- `features.parquet`: the competed table carries the feature columns, and no stage opens a
  run-level feature table. On a real run it was 55 GB of writes per run.
- `psms_extracted.parquet`: its one consumer is the candidate audit, which is off by
  default (`extract.emit_candidate_audit`). With the audit on it is pooled as before; with
  it off the band tables stay where they are and the run says so in the log. This is the
  run's second-widest artifact, so pooling it cost a full read and a full write for a file
  nothing opened.

Within a band, the competed table is normally a hard link to the band's feature table: under
the shipped grouping compete removes no row, so it publishes the features file's own bytes
instead of rewriting them (docs/11 "compete: how the competed table is published"). The two
names then cost the disk once, and deleting one band file does not free the space while the
other name exists. The pool splices from the competed name as before, so the pooled
`psms_competed.parquet` inherits the bands' 65,536-row feature row groups where it used to
inherit the 131,072-row groups of the rewrite. Its values and row order are unchanged, and so
is everything rescore reads from it, but its bytes and its manifest and report `content_hash`
differ from a grouped run made before 2026-09-25 whenever a band table holds more than one
row group.

Each pooled table is hashed once, for the manifest record and the report beside it
together. At experiment scale those tables are tens of GB, and hashing reads all of it.
The band artifacts are hashed once as well, by the stage that writes them, for its own
report. A band's manifest record reuses that hash (`run_hashed`, docs/03 "Each artifact is
hashed once"), and under `run-experiment`, which keeps no per-run manifest, no band record
is built at all. Before 2026-09-25 every band closure re-hashed its seed, windows,
extracted, chromatogram, feature and competed tables for a record that `run-experiment`
then dropped.

The pooling itself is a byte copy. A band's rows are already in the order the pooled table
wants and already encoded, so `pool` splices each band's parquet row groups into the output
without decoding them (`mumdia_io::table::SpliceWriter`, the column-chunk append the parquet
writer exposes for concatenation). Only the row groups that hold a candidate the overlap
dedup drops are decoded, filtered and re-encoded, and window overlap puts those at the two
ends of a band. Measured on the production seven-file experiment, both arms on the same
63 bands of one run (537,047,953 chromatogram rows and 35,844,209 competed rows, 136 GB):

| pooling | wall | peak RSS |
|---|---|---|
| decode and re-encode | 3.5 MB/s, 42.7 GB in 2 h 50 min (killed) | 4.3 GB |
| splice the row groups | 3 min 12 s for all 136 GB | 2.6 GB |

The disk reads 221 MB/s (`dd`, direct), so the old path was two orders of magnitude off the
hardware and single-threaded: 110% CPU throughout. The spliced output holds the same rows in
the same order with the same values; its row groups are the bands' own, so it is not
byte-identical to a re-encoded pool.

Under `extract.chromatogram_schema = 2` the band chromatogram tables are v2
(docs/15_data_dictionary.md, "Layout v2"), and every band of a run has the same layout,
since the splice refuses a band whose columns differ. A v2 row group decodes on its own,
so splicing whole groups keeps the pooled table readable. The groups the pool decodes
and filters are written back as one group each, whatever their size: the filter drops
whole candidates, which keeps every surviving candidate's axis row, while a split would
leave the rows after it without one.

`mumdia pool --groups-dir <run>/groups` runs the same stage standalone, which is how those
numbers were taken, and is what to reach for when a grouped search finished but the run did
not: the band directories hold everything, and pooling them is a copy. The feature and competed schema companions
(`<table>.schema.json`, the classifier's column list) are copied from the first band; every
band wrote the same one. Each pooled table gets a `.report.json` whose stage is `pool` and
whose stats record the number of groups and the overlap duplicates removed.

The overlap dedup itself is skipped when the bands' library row spans are disjoint. A band
is a row span of the m/z-sorted precursor table and a candidate id is the band-local id
plus the band's first row, so bands whose spans do not overlap cannot share a candidate.
`run_groups` knows the spans from the plan and tells the pool (`bands_disjoint`), which then
does not decode `candidate_id` and `prelim_score` of every band's competed table and build
a map over every candidate of the run to find no duplicate. The test is on the row spans,
not on the band m/z bounds: two bands that touch at one m/z value both hold a precursor at
exactly that value, and the spans say so. The standalone `mumdia pool` has no plan to ask
and looks as before. The pooled tables are byte-identical either way
(`disjoint_bands_skip_the_dedup_and_pool_the_same_bytes`).

With `groups.pool_competed = false` the competed rows are not pooled at all. Rescore
already accepts several competed tables; it reads the bands' tables in band order with a
table-to-source map (`RescoreParams::sources`, every band of run i stamped `source` i), so
its input rows are exactly the rows the pooled table would hold, in its order, and
`psms_scored.parquet` is byte-identical to the pooled run's
(`band_tables_with_a_source_map_score_the_pooled_tables_bytes`). That saves one full write
and read of the run's widest artifact, about 83 GB per run on the immunopeptidomics
experiment (about 581 GB for its seven runs). It is done only where it cannot change a
result or starve a reader: the bands' row spans must be disjoint, since otherwise the
overlap losers have to be dropped, and neither the candidate audit nor match-between-runs
may be on, since both read the pooled table. Otherwise the table is pooled and the log
says why. What changes is the artifact set: there is no pooled `psms_competed.parquet`
(one an earlier run left in the directory is removed) and no manifest record for it, and
the scored table's report lists the band tables under `competed_inputs` with their
`competed_sources`. A later standalone `mumdia rescore` or `mumdia audit` needs the table,
which `mumdia pool --groups-dir` rebuilds from the bands. To validate on a data set, run
the same grouped configuration with the default and with `false` and compare
`psms_scored.parquet` byte for byte.

With `groups.pool_chromatograms = false` the chromatograms are not pooled either. Quant is
the table's one reader (the candidate audit and match-between-runs do not open it), and it
reads a run's chromatogram tables as a list: the bands' own tables in band order, each with
the candidates it does not contribute (`quant::ChromTable`). Those are the overlap losers
of the chromatogram tables, the same sets the pooled table is spliced without, and here the
pool writes them to `groups/overlap_losers.parquet` (`band`, `candidate_id`; no rows for
disjoint bands), so unlike `pool_competed` this also holds for overlapping bands. The losers
are found in the chromatogram tables themselves, not only in the competed ones
(`pool::table_losers`): a band's chromatogram table holds every candidate extract accepted,
including ones compete then deleted there, so under `compete.group_by = base_peptide` or
`apex` a candidate can be in two bands' chromatogram tables and one competed table. It is
kept from the band whose competed row won it (from the first band that holds it when no
band kept it) and dropped from the others, in the pooled table as in the band tables.
Before this, the pooled table held both bands' rows of such a candidate and quant summed
them; the default `peptidoform_charge` gives each candidate a group of its own and deletes
none, so it was not affected. The loser file's footer names the band tables it belongs to,
in order (directory and file name, row count, and the content hash of each table's
report), and a standalone quant refuses a list of band tables that differs in count, order
or identity. Quant reads the tables in order, drops each one's losers, and concatenates the per-table stores with the
same `ChromStore::append` that joins row groups, which rebuilds a file seam exactly as it
rebuilds a row-group seam: its store is the one the pooled table gives, and so are the quant
tables, byte for byte (`quant_from_the_band_tables_writes_the_pooled_runs_bytes`, with two
bands overlapping, two of the overlap candidates deleted by compete in one band each;
`band_tables_read_in_order_rebuild_the_pooled_tables_store`, with a band whose groups are
all pruned). A candidate with rows in two of the tables, which is what the band tables of
overlapping bands read without their losers give, is refused rather than quantified from
both; so is a candidate whose rows continue from the end of one table into the next, which
a real band table never gives, although the store rebuilds that seam. That saves the
splice write, the hash and one read of the run's largest artifact, about 68 GB per run on
the immunopeptidomics experiment. The artifact set changes: no pooled `chromatograms.parquet`
(one an earlier run left is removed, as a loser table an earlier run left is under the
default) and no manifest record for it, an `overlap_losers` record instead, and the quant report's `chromatograms` lists the band tables with
`chromatogram_dropped_candidates`. A later standalone quant is
`mumdia quant --chromatograms groups/g00/chromatograms.parquet ... --overlap-losers
groups/overlap_losers.parquet`, with the band tables in band order, or `mumdia pool
--groups-dir` rebuilds the pooled table. To validate on a data set, run the same grouped
configuration with the default and with `false` and compare `peptide_quant.parquet`,
`protein_group_quant.parquet` and `fragment_quant.parquet` byte for byte; the smoke does this
on its three-band fixture. Done on the AIF benchmark in four bands (augmented DIA-NN
library, `native_tda`, 1,050,807 chromatogram rows, disjoint bands): the three quant tables,
`psms_scored.parquet` and both TSVs were byte-identical, the pooled table the run no longer
wrote was 237 MB, and `mumdia quant` over the four band tables with the loser file wrote
the same quant tables again. Overlapping bands are covered by the tests above only
(`a_candidate_compete_deleted_in_one_band_is_pooled_from_one_band` for the loser sets).

From here on the run is an ordinary run: rescore, audit, quant and report read the pooled
tables (or the band tables above), and `psms_scored.parquet.report.json` names the
classifier as always. The manifest's
RT model identity says `(per window group)` after the model that ran.

## 6. Output layout

```text
out/
  spectra/                           convert, as always
  groups/plan.json                   bands, windows, estimates, calibration mode
  groups/gNN/lib_precursors.parquet  the band (local ids), only where an RT model rewrites it
  groups/gNN/seed_psms.parquet       band seed (+ .masscal.json)
  groups/gNN/seed_psms_pooled.parquet  the band's view of the pooled seed
  groups/gNN/lib_precursors_<model>.parquet  re-predicted band, when an RT model ran
  groups/gNN/run_windows.parquet     the band's RT windows
  groups/gNN/cal.json                the band's RT calibration
  groups/gNN/{psms_extracted,chromatograms,features,psms_competed}.parquet
  seed_psms.parquet                  pooled seed, library-wide ids (+ .masscal.json)
  seed_psms_calibrated.parquet       pooled seed with refreshed iRT (global, after re-prediction)
  cal.json                           run-level RT calibration record (section 7)
  {chromatograms,psms_competed}.parquet  pooled (+ .report.json); psms_competed not
                                     under groups.pool_competed = false, chromatograms not
                                     under groups.pool_chromatograms = false (section 5)
  groups/overlap_losers.parquet      the pool's overlap losers per band, written under
                                     groups.pool_chromatograms = false
  psms_extracted.parquet             pooled only under extract.emit_candidate_audit
  psms_scored.parquet, quant, peptides.tsv, proteins.tsv, manifest.json  as always
```

By default the band directories are diagnostics and reproducibility material, not inputs to
any later stage; delete them once the run is accepted if space matters. Under
`groups.pool_chromatograms = false` they are not: the band chromatogram tables and
`groups/overlap_losers.parquet` are then the run's only chromatograms, and deleting them
loses re-quantification and re-pooling. `groups.delete_band_intermediates`
(default `false`) does part of that automatically: once the pool is written, each band's
`psms_extracted.parquet` and `features.parquet` (with their reports, schema companions and
any `run.pin`) are deleted, which on the immunopeptidomics experiment was most of the band
directories' volume. The chromatograms and competed tables stay, because `mumdia pool
--groups-dir` re-pools from them; the manifest keeps the deleted tables' records. Disk only:
every output is unchanged. There is no run-level
`run_windows.parquet`: the windows are per band, and nothing after compete reads them.

## 7. The run-level `cal.json`

`groups::summarise_cal` writes it from the bands' own so that tooling written for an
ungrouped run (the CI smoke checker, notebooks) reads it unchanged. Under `global` the
first band's record is the run's, because the fit is one; under `per_group` the record sums
`n_train`, takes the median `w_rt`, weights the residual summaries by anchors and reports
`calibration_status = mixed` when the bands disagree. Either way it adds `window_groups`,
`groups_calibration` and a `groups` list with each band's `n_train`, `w_rt`, status and
median absolute residual. As everywhere, the in-sample residuals are fit diagnostics, not
error estimates (`docs/08_rt_im_train.md` section 4).

## 8. Measured

CI fixture (`ci/smoke.sh`: 160 planted peptides, 8 windows, native Python-free
configuration, `native_tda`), 2026-09-20, same binary:

| run | groups searched | stripped peptides at 1% | smoke assertions |
|---|---|---|---|
| ungrouped | 1 | 150 | 168 of 168 |
| `window_groups: 3`, global | 3 | 151 | 168 of 168 |
| `window_groups: 8`, global | 5 (three merged: empty bands) | 151 | 168 of 168 |
| `window_groups: 3`, per_group | 3 | 147 | RT checks fail: no band has a confident anchor (`1/T`) |

The fixture cannot measure memory or time; the real-scale measurement below does.

Ungrouped and three-group global share 149 stripped peptides; the two that differ are
sibling-modform ties on synthetic spectra, and the grouped run's extract tolerance was the
configured 20 ppm rather than the ungrouped run's learned 5 ppm, for the same `1/T` reason
(no band reached a confident seed, so no band calibrated). A repeated grouped run is
byte-identical in every artifact.

### Real scale: AT10234AUH against the 8-12-mer library (2026-09-20/21)

Measured with a zero-code harness on basilisk (EPYC 7H12, 2 TB) before this PR existed:
`scripts`-level slicing of the library into 8 groups of 14-15 windows in m/z order (equal
window counts, NOT precursor-balanced: 0 / 22.6M / 43.9M / 44.6M / 27.9M / 24.9M / 22.8M /
16.8M precursors; the empty low-m/z group skipped), one `mumdia run` per group at 32 threads,
4 groups at a time, then one pooled `rescore --competed` over the groups' competed tables.
Reference: the monolithic calibrated run of the same file (`run812_ms1`, 128 threads):
22,850,003 accepted PSMs, 11,271 precursors and 10,213 peptides at 1%, 6 h 24 min, 173 GB
engine RSS.

**Global calibration** (each group extracted with the monolithic run's RT windows, mass
calibration and MS1, which is what `groups.calibration = global` does in the engine):

| group | precursors | extract wall (32 thr) | extract RSS | accepted |
|---|---|---|---|---|
| g01 | 22.6M | 19.6 min | 68 GB | 2,600,526 |
| g02 | 43.9M | 41.1 min | 94 GB | 5,845,126 |
| g03 | 44.6M | 42.6 min | 96 GB | 5,517,121 |
| g04 | 27.9M | 35.6 min | 80 GB | 4,647,444 |
| g05 | 24.9M | 24.5 min | 63 GB | 3,195,039 |
| g06 | 22.8M | 9.8 min | 42 GB | 913,969 |
| g07 | 16.8M | 3.0 min | 22 GB | 130,778 |

The seven extractions reproduce the monolithic one row for row: the same 22,850,003
candidates in the same order with identical apex retention times (checked over the union of
the group tables with ids mapped back). The pooled rescore over the seven competed tables
(31.7 min at 40.6 GB) gave 11,271 precursors at 1%, the monolithic count exactly, and
10,346 peptides at 1% against 10,213 (+1.3%; 9,627 shared, 586 only monolithic, 719 only
grouped): the same PSMs and features in a different row order, so the classifier's folds and
initial ranking differ within the seed spread. Wall clock for extract + features + compete +
rescore: 107 min for the grouped arm (4 x 32 threads) against 329 min for the monolithic
stages (128 threads); extract CPU summed over the groups 11.7 CPU-hours. The partition adds
no measurable cost, and the independent 32-thread processes scale where the one 128-thread
extract did not.

**Per-group calibration** (each group seeded and calibrated on its own): three of seven
groups (g01, g02, g07) had no confident seed on their own q scale, the `1/T` floor at real
scale, and ran with an unbounded RT window at the configured 20 ppm instead of the learned
9.5 ppm. g01 accepted 6,831,580 PSMs from 11% of the library and took 9.6 h at 560 GB, g07
2:43 h at 159 GB for the smallest band, and g02 aborted after 6 h at 664 GB with
`memory allocation ... failed` at the host's commit limit. The four calibrated groups had
`w_rt` between 130 and 404 s (monolithic 286 s) and ran in 26-76 min at 36-79 GB. The pooled
rescore over the six surviving groups gave 8,845 peptides at 1%, with g02's band missing
entirely. This settles the default: `global`.

What the numbers say about the group count: memory per group follows the accepted rows
of the band (about 16-17 GB per million accepted PSMs here, plus a fixed part), not the
precursor count, so precursor-balanced bands and more of them are the lever for a 100 GB
desktop (8 equal-window groups peaked at 96 GB; a 12-group precursor-balanced plan is the
next measurement), together with streaming extract's accepted rows to disk during the band.
The pooled rescore (40.6 GB here for 22.85M PSMs) is the stage grouping does not bound; the
compact feature preset is its lever.

### How many groups, and what it costs

Measured 2026-09-21 on four idle 128-core hosts, one group count each, the same file and
the same run-wide calibration, six threads per band and no per-band fragment table
(`extract --fragment-offset` against the shared sorted table). Bands are precursor-balanced.

| asked for | bands | precursors per band | largest band | extract CPU | cores per band | modelled wall |
|---|---|---|---|---|---|---|
| 48 | 48 | 4.24M | 44.3 GB | 25,901 s | 2.7 | 13.2 min |
| 64 | 63 | 3.23M | 39.0 GB | 20,821 s | 2.7 | 8.1 min |
| 96 | 81 | 2.51M | 31.2 GB | 42,174 s | 2.3 | 14.9 min |
| 192 | 94 | 2.16M | 23.0 GB | 39,624 s | 2.1 | 14.9 min |

All four arms accepted **22,850,003 PSMs, the monolithic run's count exactly**, so the
number of bands changes neither what is searched nor what is extracted. The wall is
modelled as the summed band time over the number of bands run at once (12, 16, 21, 21),
because the arms were interrupted and resumed; the CPU totals are measured.

Pooling each arm's competed tables and rescoring (28 to 29 minutes at 40.6-40.8 GB in
every arm, since the pool is the same 22.85M PSMs whatever the banding):

| arm | peptides at 1% | precursors | protein groups |
|---|---|---|---|
| monolithic | 10,213 | 11,271 | 4,777 |
| 8 bands (the earlier arm) | 10,346 | 11,271 | 4,761 |
| 48 bands | 10,559 | 11,579 | 4,899 |
| 63 bands | 10,608 | 11,594 | 4,863 |
| 81 bands | 10,382 | 11,404 | 4,841 |
| 94 bands | 10,631 | 11,643 | 4,893 |

The five grouped arms span 10,346 to 10,631 peptides with no trend in the band count, which
is the classifier reshuffling on a differently ordered input: the PSMs and features are
identical, only the row order into rescore differs, and CLAUDE.md puts that at up to about
1% on a pool of this size. Read this as "grouping costs no identifications", not as a gain
over the monolithic run; a gain would need seeds, which none of these arms used.

Three things this says:

- **A band cannot be smaller than one isolation window.** The run has 114 windows, so 96
  groups became 81 bands and 192 became 94. Beyond one window per band, the only way to
  divide further is to split a window's candidates by m/z, which extract already does
  across threads (`accumulate_groups`) but not across processes.
- **Memory falls sublinearly.** Halving the precursors per band from 4.2M to 2.2M takes the
  largest band from 44 to 23 GB, not to 22, because each band pays a fixed cost: the run's
  spectra (1 GB here), the band's fragment index, and the accumulator, which is sized by the
  windows in flight rather than by the band.
- **Cost rises past 64 bands.** The 81- and 94-band arms spent about twice the CPU of the
  63-band arm. Each extra band repeats the fixed load, and the per-band single-threaded
  load phase becomes a larger share, visible as cores per band falling from 2.7 to 2.1.
  Those two arms also ran 21 bands at once against 12 and 16, so part of the penalty is
  host contention rather than band size.

On this data the useful range is therefore 48 to 64 bands: about 40 GB per band, which is
what a 100 GB desktop can run two of at a time, or one with room to spare.

### Band size, not band count: the HYE Astral measurement (2026-09-22)

The band counts above were derived on a 203M-precursor library, where a band of a 63-band
plan holds 3.2M precursors. Repeating the exercise on a 10.9M-precursor library says the
useful range is a property of the BAND, not of the plan, and that a small library should not
be banded at all.

Six 15-minute Astral files, imported HYE library, the same adapted retention times in both
arms, one arm unbanded and one cut into 100 bands of three isolation windows each
(about 126,000 precursors per band):

| | unbanded | 100 bands |
|---|---|---|
| precursors at 1% | 126,436 | 121,966 |
| peptides at 1% | 113,860 | 110,006 |
| protein groups at 1% | 12,166 | 12,029 |
| extract, summed over six files | 4.6 CPU-min | 153 CPU-min |
| search-seed, summed | 1.8 CPU-min | 215 CPU-min |
| engine peak resident | 11.8 GB | 182 GB |
| peak mappings | 456 | 11,441 |

Two things to take from it.

**The fixed cost per band is the run's spectra, and it does not shrink with the band.** Each
band decodes its own copy: 3.84 GiB of MS2 scans to search a 0.016 GiB slice of library, a
ratio of 240 to 1. That is why the seed costs 215 CPU-minutes here against 1.8, and why 48
bands in flight hold 182 GB. It is also why raising `groups.parallel` stops helping: at 100
bands the machine was at load 35 of 128, waiting on decodes rather than searching.

**The identification loss was a defect, not a property of banding.** A grouped run fitted
its fragment mass calibration per band and combined the bands' scalars, giving 11.400 ppm
against the 8.452 a single fit gives on the same data: the tolerance is
`1.5 x p95(|dev - median|)`, and a 95th percentile over one band's ~2,000 deviations has a
heavier tail than the same percentile over the union. Each band also selected its calibrants
on its own q, whose 1/T floor is looser (106,088 confident seed PSMs against 97,584). The
wider tolerance admitted 33% more candidates, and the extra noise cost 3.4% of the peptides.
Proved rather than inferred: one band extracted twice, identical in every input except which
calibration file it was handed, accepted 12,414 candidates at 11.400 ppm and 8,791 at 8.452.
A control arm, unbanded on the same adapted library, reproduced the unbanded arm's 113,860
peptides exactly, which rules out the seeding difference between the arms.

Fitting the calibration once on the bands' pooled deviations settles it. The banded arm
repeated on that build reproduces the unbanded calibration to eight significant figures and
selects the same calibrants:

| | unbanded | banded, per-band scalars | banded, pooled deviations |
|---|---|---|---|
| offset | -1.8486016959 | -1.8834 | -1.8486016989 |
| tolerance | 8.452381550 | 11.400 | 8.452381790 |
| calibrant deviations | 181,196 | 200,257 | 181,196 |
| residual MAD | 0.90689065 | 1.035 | 0.90689063 |

The residual difference is the sidecar storing deviations as f32. Downstream, the banded run
then extracts the SAME candidate set as the unbanded one, to the row: 4,986,153 accepted and
74,115,941 chromatogram rows in both, and 4,986,153 scored rows against the unbanded run's
4,986,153.

| at 1% | unbanded | 100 bands, per-band scalars | 100 bands, pooled deviations |
|---|---|---|---|
| precursors | 126,436 | 121,966 | 125,983 |
| peptides | 113,860 | 110,006 | 113,789 |
| protein groups | 12,166 | 12,029 | 12,221 |

That is 98% of the lost peptides recovered, and what remains is inside the single-seed
spread this pool shows (about 0.4%, docs/28): the protein groups come back slightly above
the unbanded arm, which is the same noise in the other direction. Banding is
identification-neutral on this data once the calibration is fitted once, and what it costs
is the fixed per-band work above.

### Scheduling the bands: a bounded queue, longest first

`groups.parallel` bands are in flight at a time in each of the three band phases (seed,
retention-time windows and extract, features and compete). They go through a bounded work
queue (`groups::run_bounded`): that many workers each take the next band as soon as their
current one is done. Until 2026-09-25 the phases ran in fixed chunks of `parallel` bands
with a barrier after each chunk, so a chunk waited for its slowest band while the other
slots sat idle. The bound on the resident set is the same, since no more than `parallel`
bands are ever in flight, and results are still merged in band order, so the artifacts and
the manifest do not depend on the schedule.

The queue starts the most expensive bands first. Band cost follows spectral density rather
than the precursor count the plan balances: on the immunopeptidomics search two bands of
2.98M and 3.03M precursors took 42 s and 460 s (2026-09-21). The seed and extract phases
estimate a band's cost as the sum over its windows of the window's estimated precursors
times the MS2 peaks of its scans (`groups::window_costs`, both known before the seed); the
features phase uses the rows the band's extract accepted. Starting the long bands first
keeps one of them from arriving last and running alone. The gain is not measured; it is
zero at the default `parallel: 1`, and it applies only where the band phases are CPU-bound.

Inside a band, the seed's parallel unit is a contiguous RT chunk of a window group rather
than the whole group (`docs/07_search_seed.md`, step 3). A band serves only the one to three
isolation windows over its m/z range (114 windows over 63 bands on the immunopeptidomics
plan, three per band on the 100-band HYE plan), so with one task per window its probe phase
ran on one to three threads whatever `--threads` was. The chunked seed is bit-identical to
the per-window one; the gain is not measured. The interval from a band's
`search-seed: loaded` to its `search-seed: mass recalibration` log line is the index build
plus the probe, which is what the change shortens.

### `groups.parallel` and the thread count

A band in flight occupies one rayon worker, which then blocks on its own extraction's
accumulation channel while the probing tasks run on the other workers. So `groups.parallel`
must stay below `--threads`: with as many bands as threads every worker parks and the run
makes no progress at all (reproduced on the fixture at `parallel = 2, --threads 2`: the
process sat at 0.1 s of CPU indefinitely, with no error and no output). The orchestrator
clamps the value to `threads - 1` and warns. The production runs are far from the bound (8
bands on 128 threads), which is why this was not visible before.

## 9. The confident elution bounds are pooled too

`features.bound_from_confident` (default on) fits one pair of elution half-widths from the
run's confident seed anchors and gives every candidate the window `[apex - L, apex + R]`.
It needs at least 20 anchors with a resolvable peak; below that it warns and falls back to
per-candidate boundary detection.

Twenty is a floor on the RUN's anchors, and a band holds a slice of the m/z range and
therefore a slice of them. Measured on the seven-file immunopeptidomics search
(`window_groups = 63`, 203M-precursor library), AT10273AUH had **735 confident anchors in
the pooled seed and 0 or 1 in each band's chromatogram table**, so every band logged

```text
features: bound_from_confident set but < 20 confident anchors; falling back to
per-candidate boundary
```

while the same run searched unbanded fitted a global window. That is a banded/unbanded
divergence of the same kind as the per-band mass calibration of section 7, in the stage
after it.

The fix is the same move: the samples are pooled, not the fitted scalars. The band loop is
now two phases. Phase 1 runs `rt-im-train` and `extract` for every band and calls
`features::confident_bound_samples`, which is the pass `features` would have run
internally -- same confident set, same row-group pruning, same detector -- returning the
per-anchor half-widths instead of a percentile of them. The run absorbs every band's
samples, fits `features::bounds_from_samples` once, and phase 2 runs `features` and
`compete` for every band through `features::run_with_bounds` on that one pair. A run whose
pooled set is genuinely under 20 anchors still falls back, now once rather than band by
band.

Two things come free with the split. The shared MS2 and MS1 buffers are dropped at the end
of phase 1 rather than after competition, and phase 2 skips the extra streaming pass over
the chromatogram table that each band used to make.

Sizing the cost of the defect: rescoring one banded competed table (AT10273AUH,
29,429,386 rows) under the first pass's own recipe gave 9,148 peptides against the
unbanded first pass's 10,342, with the rescore recipe held fixed -- 11.5% of peptides,
10.0% of precursors and 6.1% of protein groups, from a band search that pushed 88% MORE
candidates into the scored table. How much of that this fix returns is not yet measured.

## 10. What is not there yet

- Bands run in one process, `groups.parallel` at a time through the bounded queue. Running
  them as child processes is still open and needs nothing in the artifacts: the band
  directories are already independent, and the pool reads whatever is there.
- Choosing `window_groups` from a memory budget rather than by hand. The plan's
  `est_precursors` per band and the measured bytes per precursor of extract are what it
  would use.
- Measurements at scale of the 2026-09-25 changes: the band queue and the chunked seed
  (per-band wall times on the immunopeptidomics or HYE banded arms), `rt_adaptation =
  once_per_run` and `balance = cost` (peptides at 1% inside the seed spread on two
  acquisitions), and the projection cache (`rt_im_train.deeplc_projection_cache`) under
  `rt_library_scope = per_run`. The fixture runs pin their output effect, not their gain.
- The calibrant sidecar's size and the grouped seed phase's peak at scale, now that every
  target is offered (section 4a): the per-band `calibrant_bytes` and the seed and seed-pool
  peaks on HYE at 2 bands and on the 63-band immunopeptidomics plan. The bound in section 4a
  says they follow the run's scans and not its library; if a band's sidecar turns out to
  matter beside the seed's own tables, the seed can stream it to the parquet in row groups
  and `seed-pool` can filter it row group by row group.
- `experiment.overlap_front_threads` overlaps the convert and seed of runs 2..N with the
  first run's adaptation only for ungrouped runs; a grouped run seeds inside its band loop,
  so overlapping it would need the band loop split between runs.
- Entrapment validation. The grouped search computes the same scores from the same
  evidence, and the fixture shows identical identifications within tie noise, but the
  policy in `docs/20` asks for an empirical null on two acquisitions before any default
  changes, and `window_groups` stays `1` until then.
