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

`run-experiment` (and `run` with several `--mzml`, which dispatches to it) refuses a
grouped configuration rather than searching each run against the whole library while the
key says otherwise. Run each file with `mumdia run --mzml <one file>` and pool the competed
tables with `mumdia rescore --competed a b c`, which stamps `source` and computes the
per-source `run_psm_q` (`docs/11_compete_rescore_fdr.md`).

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

A band is the union of its windows' m/z ranges. Two consequences are written to
`groups/plan.json`:

- `est_unselectable`: precursors outside every window (charge states above the top window,
  typically). They belong to no band and are never loaded, which is the right outcome for
  something the run cannot measure. The ungrouped search loads them and extracts nothing.
- `est_duplicated`: precursors in the overlap of two adjacent windows across a cut. They are
  searched in both bands; the pool keeps one row per candidate (section 5). Overlapping
  window schemes therefore cost the overlap twice and nothing else.

## 3. One band, one library

For each group the orchestrator writes the band as a precursor table of its own,
`groups/gNN/lib_precursors.parquet`, with band-local ids `0..n`
(`groups::write_band_slice`, which reads only the row groups that cover the span through
`TableFile::open_rows`). The band's first library row is its offset; local id plus offset is
the library-wide id, and every stage below carries it in the manifest as `name[gNN]`.

The stages then run unchanged on that table, as if it were the library:

- `search-seed` loads the band with `Library::load_with_fragment_offset`: the band file plus
  the fragments whose ids lie in the band's range, read from the shared, library-wide
  fragment table. When that table is sorted by `candidate_id` at row-group granularity
  (`scripts/sort_fragments.py`; every library writer ends with the sort) the read is
  selective; an unsorted table still loads through a filtered scan of the whole table, with a
  warning, and costs the scan per band.
- the RT model (multi-head calibration, the optional fine-tune, or the base-model
  re-prediction, whichever the configuration resolves to; `docs/08_rt_im_train.md`) runs
  on the band table and writes the band's re-predicted table. The DeepLC sidecars rewrite a
  precursor file, which is why the band is a file rather than an in-memory slice.
- `rt-im-train` writes the band's `run_windows.parquet` and `cal.json`.
- `extract`, `features` and `compete` write the band's `psms_extracted`, `chromatograms`,
  `features` and `psms_competed`. The id columns are library-wide on the way out
  (local id plus `Library::global_offset`), so a band's table means the same thing as the
  run's and pooling is a concatenation.

Nothing outside the band is resident during any of these. The converted spectra are read by
each stage as they always are.

Under `experiment.rt_library_scope = first_run_only` the runs after the first reuse the first
run's adapted bands, and then they reuse its band slices too: a slice is a deterministic
function of the library and the row span, so rewriting it would produce the same bytes. Only
the seed reads it, and the adapted table replaces it everywhere else. On a 203M-precursor
library that is the whole precursor table not written, per run after the first.

## 4. The seed pool and the two calibration modes

Each band's seed has two problems as a calibration anchor set. Its `spectrum_q` was
estimated on the band's PSMs alone, and the q floor is `1/T` (`docs/11_compete_rescore_fdr.md`), so a
small band cannot reach 1% at all: on the CI fixture, 179 seed PSMs give every one of them
q = 0.0056 as one pool and none of them q < 0.01 as three bands of 73, 67 and 39. And its
ids are local.

`seed-pool` (`stages/seed_pool.rs`) reads every band's seed, maps the ids to library-wide
ones, keeps the higher-scoring row where two bands searched the same candidate, re-estimates
`spectrum_q` over the union with the seed's own target-decoy kernel (`fdr::target_decoy_q`),
and writes the run-level `seed_psms.parquet`. Beside it goes `seed_psms.parquet.masscal.json`:
the bands' scalar ppm offsets and learned tolerances combined by calibrant count (`n_dev`).
The optional m/z-dependent grids are not combined, because they need the per-fragment
deviations the seed does not keep; extract then applies the scalar offset. A band with no
calibrants wrote the configured tolerance in place of a learned one (search-seed's failure
branch), and when no band calibrated the pool keeps that tolerance rather than averaging
nothing into a zero, with a warning. It also writes each band's view of the pooled seed,
`groups/gNN/seed_psms_pooled.parquet`: the band's own rows, local ids, pooled q.

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
and only the windows differ, because the precursors do.

Refitting per band is sound because the multi-head ridge and the base-model re-prediction
are deterministic in their anchors: the same pooled anchors give the same head selection,
the same ridge and the same predictions, so every band carries one model, at the cost of
one anchor pass per band (seconds). The optional DeepLC fine-tune is not deterministic and
would be trained once per band, so a grouped run refuses `rt_im_train.finetune_deeplc`;
fine-tune the library once beforehand (`docs/08_rt_im_train.md`, once per library) and
search that table.

`per_group` is kept for the comparison the design asked for, not as a recommendation: it is
one pooling pass cheaper and every group is independent, but each group fits on a fraction
of the anchors, and the fit quality sets the RT window that the extract of every group then
pays for. On the CI fixture it cannot fit at all (no band has a confident anchor, for the
`1/T` reason above); on a real run every band has thousands.

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

Each pooled table is hashed once, for the manifest record and the report beside it
together. At experiment scale those tables are tens of GB, and hashing reads all of it.

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

`mumdia pool --groups-dir <run>/groups` runs the same stage standalone, which is how those
numbers were taken, and is what to reach for when a grouped search finished but the run did
not: the band directories hold everything, and pooling them is a copy. The feature and competed schema companions
(`<table>.schema.json`, the classifier's column list) are copied from the first band; every
band wrote the same one. Each pooled table gets a `.report.json` whose stage is `pool` and
whose stats record the number of groups and the overlap duplicates removed.

From here on the run is an ordinary run: rescore, audit, quant and report read the pooled
tables, and `psms_scored.parquet.report.json` names the classifier as always. The manifest's
RT model identity says `(per window group)` after the model that ran.

## 6. Output layout

```text
out/
  spectra/                           convert, as always
  groups/plan.json                   bands, windows, estimates, calibration mode
  groups/gNN/lib_precursors.parquet  the band (local ids)
  groups/gNN/seed_psms.parquet       band seed (+ .masscal.json)
  groups/gNN/seed_psms_pooled.parquet  the band's view of the pooled seed
  groups/gNN/lib_precursors_<model>.parquet  re-predicted band, when an RT model ran
  groups/gNN/run_windows.parquet     the band's RT windows
  groups/gNN/cal.json                the band's RT calibration
  groups/gNN/{psms_extracted,chromatograms,features,psms_competed}.parquet
  seed_psms.parquet                  pooled seed, library-wide ids (+ .masscal.json)
  seed_psms_calibrated.parquet       pooled seed with refreshed iRT (global, after re-prediction)
  cal.json                           run-level RT calibration record (section 7)
  {chromatograms,psms_competed}.parquet  pooled (+ .report.json)
  psms_extracted.parquet             pooled only under extract.emit_candidate_audit
  psms_scored.parquet, quant, peptides.tsv, proteins.tsv, manifest.json  as always
```

The band directories are diagnostics and reproducibility material, not inputs to any later
stage; delete them once the run is accepted if space matters. There is no run-level
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

### `groups.parallel` and the thread count

A band in flight occupies one rayon worker, which then blocks on its own extraction's
accumulation channel while the probing tasks run on the other workers. So `groups.parallel`
must stay below `--threads`: with as many bands as threads every worker parks and the run
makes no progress at all (reproduced on the fixture at `parallel = 2, --threads 2`: the
process sat at 0.1 s of CPU indefinitely, with no error and no output). The orchestrator
clamps the value to `threads - 1` and warns. The production runs are far from the bound (8
bands on 128 threads), which is why this was not visible before.

## 9. What is not there yet

- Groups run one after another in one process. Running them as child processes in parallel
  is the next step and needs nothing in the artifacts: the band directories are already
  independent, and the pool reads whatever is there.
- Choosing `window_groups` from a memory budget rather than by hand. The plan's
  `est_precursors` per band and the measured bytes per precursor of extract are what it
  would use.
- `run-experiment`: per-run grouped search with the pooled `rescore --competed` and the
  per-source `run_psm_q` it already provides.
- Entrapment validation. The grouped search computes the same scores from the same
  evidence, and the fixture shows identical identifications within tie noise, but the
  policy in `docs/20` asks for an empirical null on two acquisitions before any default
  changes, and `window_groups` stays `1` until then.
