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
  `features` and `psms_competed` with local ids.

Nothing outside the band is resident during any of these. The converted spectra are read by
each stage as they always are.

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

`pool` (`stages/pool.rs`) rewrites the four band tables with library-wide ids and appends
them, batch by batch, into the standard `psms_extracted.parquet`, `chromatograms.parquet`,
`features.parquet` and `psms_competed.parquet`. Where two bands hold the same candidate
(window overlap across a cut), the competed row with the higher `prelim_score` wins and the
loser's rows are dropped from all four tables, so the pooled tables hold each candidate
once and rescore's competition is unchanged. One batch is resident at a time, and the writer's
row groups are capped, so the pool costs one read and one write of the group artifacts and
no more memory than any single stage. The feature and competed schema companions
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
  {psms_extracted,chromatograms,features,psms_competed}.parquet  pooled (+ .report.json)
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

Ungrouped and three-group global share 149 stripped peptides; the two that differ are
sibling-modform ties on synthetic spectra, and the grouped run's extract tolerance was the
configured 20 ppm rather than the ungrouped run's learned 5 ppm, for the same `1/T` reason
(no band reached a confident seed, so no band calibrated). A repeated grouped run is
byte-identical in every artifact.

The real-scale measurement is the one that decides the default group count and the
calibration mode: the immunopeptidomics file AT10234AUH against the 8-12-mer library,
eight groups, per-group and global arms, against the monolithic run (10,213 peptides,
6 h 36 min, 335 GB). It is running as this page is written and will be recorded here with
the memory and time per group.

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
