# extract (Stage D): the core targeted extraction
> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

`extract` is the central stage of the pipeline. It takes the run-independent
spectral library (precursor and b/y fragment m/z, predicted intensities, iRT),
the per-candidate RT windows learned in `rt-im-train`, and the converted MS2
(optionally MS1) scans, and produces one apex-level PSM per surviving candidate
plus per-fragment chromatograms. Exact intensity-based scores are not computed
here; they are computed downstream in `features` from the chromatograms this
stage emits. `extract` is therefore an evidence-gathering and coarse-acceptance
stage, not a scoring stage.

The design is **data-driven and peak-major** (see the module doc comment,
`extract.rs:1`). Observed peaks probe an inverted fragment index, and a candidate
hypothesis is materialized only where fragment evidence actually collides with an
observed peak. Work scales with peak-candidate collisions, not with library size.
In wide-window DIA roughly 98% of fragment m/z collide within tolerance
(`extract.rs:1479`), so one observed peak typically matches many co-isolated
candidates; the peak-claim strategies decide how that shared intensity is
apportioned.

RT is applied as a per-candidate window post-filter (the documented Stage D part
2 fallback). On spectra that carry per-peak 1/K0 (a diaPASEF `.d` read by the native
reader) the same post-filter can take the candidate's 1/K0 window, and the stage
writes 1/K0 evidence; section 8 ("Ion mobility") describes both. On a 3D run, `apex_im`
is written null and every artifact is what it always was.

## Files

| Path | Role |
|---|---|
| `rust/mumdia/crates/mumdia/src/stages/extract.rs` | The stage: accumulation, cascade, apex, chromatogram/MS1 emission, schema writing |
| `rust/mumdia/crates/mumdia/src/peaks.rs` | Pure top-K chromatographic peak enumerator (`enumerate_peaks`) for the `retain_top_peaks` path |
| `rust/mumdia/crates/mumdia/src/index.rs` | `Library` (SoA library + bucketed `page_search`, `candidate_range`, `cand_frags`) |
| `rust/mumdia/crates/mumdia/src/matchers/fragindex.rs` | `FragIndex` backend (`build`, `probe_peak`, `candidate_range`), the default matcher |
| `rust/mumdia/crates/mumdia/src/spectra.rs` | `load_ms2` / `load_ms1` / `Ms1Scan` scan loaders |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `ExtractConfig`, `GateMode`, `PeakClaim`, `ClaimCues`, `MatcherKind` |
| `rust/mumdia/crates/mumdia-core/src/constants.rs` | `ISOTOPE_SPACING`, `ppm_bounds` |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | `artifact::PSMS_EXTRACTED`, `artifact::CHROMATOGRAMS` schema ids |

Entry point: `stages::extract::run(ExtractParams)` (`extract.rs:1300`), wired from
the CLI in `main.rs:535` (`Cmd::Extract`) and from the orchestrator in
`stages/run.rs:326`.

## Inputs and outputs

### Inputs (`ExtractParams`, `extract.rs:83`)

- `ms2` (Parquet): converted MS2 scans, loaded via `load_ms2`. On the fragindex
  matcher the MS2 and MS1 decodes run concurrently with the library load and the
  index build; the mass calibration JSON is read first because its learned tolerance
  is the one the index is built at, and its log lines and errors keep their old place
  after the spectra. A caller may lend the decodes (`ExtractParams::scans`): a
  grouped search lends both to every band, and the ungrouped `run` lends the seed's
  MS2 alone (`SharedScans::ms1 = None`, so extract still decodes the MS1 concurrently
  with its library load, and its errors keep their place after the library's). `run`
  lends only when no DeepLC step ran in between and `extract.matcher` is `fragindex`:
  the bucketed matcher's library load builds a sorted copy of every fragment, and scans
  held from the seed would sit through that transient.
  Each `Ms2Scan` carries `rt_seconds`, an isolation `window` (`lower_mz`,
  `upper_mz`), and centroided `peaks` (`mz`, `intensity`). This stage applies no
  peak cap of its own: the MS2 peak budget is fixed at conversion time by
  `--top-peaks-ms2` (`convert.rs:76-79`) and everything below operates on
  whatever survived it. `search_seed.top_n_peaks` (`config.rs:410-415`) is a
  separate, non-destructive seed-probing limit and does not change the evidence
  available here.
- `library_precursors` + `library_fragments` (Parquet): loaded once into
  `Library::load_with` (`extract.rs:1306`) at `cfg.bucket_size`. `load_with`
  skips building the bucketed `page_search` index when the fragindex backend is
  selected (`index.rs:73`).
- `run_windows` (Parquet): per-candidate RT windows, columns `candidate_id`,
  `rt_pred_cal`, `rt_lo`, `rt_hi`, read by `read_run_windows` and scattered into the
  dense `rt_lo`/`rt_hi`/`rt_cal` arrays indexed by `candidate_id`. The decoded
  columns (28 bytes per row) are dropped as soon as the scatter is done, so only the
  24-byte-per-candidate dense copy lives through the extraction. Candidates with no
  window row keep `[-inf, +inf]` and `rt_cal = NaN`, the same "calibration
  unavailable" sentinel `rt-im-train` writes; the NaN disables the Gaussian RT prior
  for those candidates (the prior requires `rt_cal > 0`). A NaN `rt_lo` or `rt_hi` is
  rejected with the row named, because a NaN bound would match every scan. An
  orchestrator hands the same three arrays over in memory
  (`ExtractParams::rt_windows`, from `rt_im_train::run_in_memory`); extract uses
  them only when they were fitted on its own `library_precursors` path, written to
  its own `run_windows` path, and cover exactly its library's candidates
  (`RtWindows::mismatch`), and reads the file otherwise, so the arrays and every
  output are the same either way (docs/08, "The in-memory handoff to extract").
- `ms1` (optional Parquet): MS1 scans via `load_ms1` (`extract.rs:1350`). When
  absent, all MS1 columns are null.
- `mass_cal` (optional JSON, the seed's `<seed>.masscal.json`): reads
  `frag_ppm_offset` and `frag_tol_ppm` (`extract.rs:1394`), and optionally an
  m/z-dependent offset grid. Missing file = offset 0 and `cfg.frag_tol_ppm`.
- `restrict_candidates` (optional prior `psms.parquet`): a `candidate_id`
  allowlist (`extract.rs:1316`) for the "gate first, then compete" workflow.

### Output: `psms_extracted` (`out_psms`, schema `psms_extracted` v2, v4 on 4D data)

Column order and types from `extract.rs:2480`:

| Column | Type | Meaning |
|---|---|---|
| `candidate_id` | U32 | Library candidate id (sorted ascending in the file) |
| `peak_rank` | I32 | Chromatographic peak rank; 0 = selected apex. Ranks >= 1 exist only when `promote_top_peaks > 1` (`extract.rs:2205`) |
| `apex_rt` | F64 | Selected apex RT (seconds) |
| `apex_im` | OptF64 | Null on a 3D run (v2). On spectra with 1/K0 (v4): the intensity-weighted median 1/K0 of the candidate's matched fragment peaks in its apex scan |
| `apex_intensity` | F32 | Full summed intensity of the apex scan group |
| `n_matched_fragments` | I32 | Distinct matched predicted fragments |
| `n_predicted_fragments` | I32 | Predicted fragment count for the candidate |
| `coelution_run` | I32 | Longest consecutive-scan co-elution run |
| `rt_pred_cal` | F64 | Calibrated predicted RT for this candidate |
| `precursor_mz` | F64 | Candidate precursor m/z |
| `charge` | I32 | Precursor charge |
| `label` | Str | `"target"` or `"decoy"` |
| `base_peptide_id` | U32 | Stripped-peptide id (for competition grouping) |
| `peptidoform` | Str | ProForma-lite string |
| `protein` | Str | Protein accession |
| `predicted_irt` | F32 | Library iRT |
| `contested_frac` | F64 | Intensity fraction lost to a better co-eluter (0 when two-pass off) |
| `ms1_isom1` | OptF64 | MS1 intensity at (mono - spacing/z) |
| `ms1_mono` | OptF64 | MS1 monoisotopic intensity |
| `ms1_iso1` | OptF64 | MS1 +1 isotope intensity |
| `ms1_iso2` | OptF64 | MS1 +2 isotope intensity |

The rows are written as they are produced (`PsmStream`). Each time 65,536 rows have
accumulated (`WRITE_TABLE_CHUNK_ROWS`) they go to the table's `TableWriter`, which is
exactly the chunk sequence `write_table` cut the whole table into, so the file is
byte-identical to the one written at the end of the stage before 2026-09-25 while only
one chunk of rows is resident (at most about 6 GB less on an unbanded
immunopeptidomics run). `the_streamed_psms_table_is_the_write_table_file` checks the
bytes at 0, 1, 65,536, 65,537 and 131,075 rows, with every optional column group.
The one exception is `emit_demix_features`: its five columns are patched in after the
candidate loop, so the rows are kept whole and written at the end through
`write_table_hashed`, which produces the same chunks. Either way the table, like the
chromatograms, is hashed as it is written, so the report's content hash needs no
read-back (`docs/03_io_layer.md`, "Hash on write"). The stage logs `extract:
psms_extracted writer` with the row count, whether the table was streamed and the
writer's busy time.

A chunk write that fails inside the candidate loop stops the loop, and the stage
then publishes neither table: the chromatogram writer is dropped unclosed
(`close_chromatograms`), so its temporary file is removed and a chromatograms table
already at the path stays as it was, as the `psms_extracted` one does. Publishing
the chromatograms there would put a truncated table beside an older
`psms_extracted`. Before the rows were streamed, the psms table was written after
the chromatograms were complete, so this case could not arise.

On spectra that carry 1/K0 the table is v4 (`artifact::psms_extracted(has_im)`):
`apex_im` is filled and three nullable Float64 columns follow it, before
`apex_intensity`: `apex_im_mad` (the fragments' intensity-weighted mean absolute
deviation from `apex_im`), `ms1_apex_im` (the 1/K0 of the MS1 precursor peak nearest
`apex_im`) and `im_pred_cal` (the candidate's calibrated 1/K0 from `run_windows`,
null without an ion-mobility calibration). Section 8 defines them. A 3D table keeps
v2's exact column set. `extract::psms_schema(path)` reads the version back from the
table's footer, which is what the orchestrators record in the manifest.

Conditional columns (default-off, added only when the knob is set so the
production schema stays byte-identical):
- `emit_contested_features` -> `contested_count_frac` F64, `apportioned_frac` F64 (`extract.rs:2505`).
- `emit_gate_diagnostics` -> `gate_apex`, `gate_peak_spectral`, `gate_coelution`, `gate_spectral_entropy` (all F32, `extract.rs:2511`).
- `emit_demix_features` -> `deconv_explained_frac`, `deconv_active`,
  `deconv_share`, `deconv_max_collinearity`, `shadow_kept_frac` (all F32,
  `extract.rs:2517`), from the spectrum-centric NNLS demix at the apex scan.
- `q1_profile` -> v5: `q1_corr`, `q1_offset`, `q1_frag_sd`, `q1_bin_frac` (F64, section 10).
- `q1_fragments` -> v6: the `q1f_*` summary and the `q1f_frag_*` per-fragment lists of the
  Q1 decision (section 12).
- `features.ms1_envelope` -> v7: the MS1 isotope-envelope evidence, 16 F64 columns
  `ms1e_*` (section 13).

The three soft-competition columns are all derived from the per-candidate
`Contested` accumulator (`extract.rs:130`), whose fields are `won`/`lost`
(summed observed intensity of shared peaks this candidate won or lost as
most-eluting claimant), `n_won`/`n_lost` (the corresponding peak-instance
counts), and `apportioned` (the co-elution-weighted proportional share the
candidate would keep under `CoelutionProportional`). The columns are (all 0 when
the two-pass path did not run):
- `contested_frac` = `lost / (won + lost)` (`extract.rs:2048`): fraction of
  contested *intensity* lost to a better co-eluter.
- `contested_count_frac` = `n_lost / (n_won + n_lost)` (`extract.rs:2056`):
  fraction of contested fragment-*peaks* lost.
- `apportioned_frac` = `apportioned / (won + lost)` (`extract.rs:2064`): fraction
  of contested intensity the candidate retains under proportional apportionment
  (1 = keeps all; ~0 = a peak-borrower stripped by its co-eluting competitors).

### Output: `chromatograms` (`out_chrom`, schema `chromatograms` v1 or v2; v3 or v4 with 1/K0)

Column order from `extract.rs:2528`:

| Column | Type | Meaning |
|---|---|---|
| `candidate_id` | U32 | Candidate id |
| `frag_name` | Str | Fragment name (`b3`, `y7`, or `ms1_mono`/`ms1_iso1`/`ms1_iso2`) |
| `frag_mz` | F64 | Theoretical fragment m/z (or precursor isotope m/z for MS1 rows) |
| `frag_obs_mz` | F64 | Intensity-weighted observed m/z (falls back to theoretical) |
| `predicted_intensity` | F32 | Library predicted intensity (0.0 for MS1 rows) |
| `rt` | LargeListF32 | Per-scan RT axis of the trace |
| `intensity` | LargeListF32 | Per-scan intensity, 0-filled on the window grid |

One row is emitted for **every** predicted transition (`extract.rs:2095`), so
`features` sees the full predicted set and can penalize a missing strong ion. A
never-observed fragment carries an **empty** trace (not a grid-length zero
vector) to avoid bloating the list column (`extract.rs:2098`). `rt`/`intensity`
are `LargeList` (64-bit offsets) because the total list-value count can exceed
the ~2.1B 32-bit `ListArray` offset ceiling when gates are opened wide
(`extract.rs:2534`). `frag_name` is written as a string but the library stores
fragment names as interned u16 dictionary ids, resolved back through
`Library::frag_name_str` (`extract.rs:2137`).

The table is written in 65,536-row groups (`CHROM_ROW_GROUP_ROWS`) with the
`rt` column PLAIN (`TableWriter::with_plain_column("rt")`): every fragment row
of a candidate carries the same axis, and snappy shortens those repeated PLAIN
runs far better than a dictionary's bit-packed indices. On the AIF
chromatograms that is 12.0% smaller than the planned dictionary (160.9 against
182.8 MB) with identical values. The other float columns keep the writer's
planned encodings (docs/03_io_layer.md, "Float encodings planned from the first
rows").

`extract.chromatogram_schema = 2` (the default since 2026-09-27; 1 before) writes the
same rows in the v2
layout (docs/15_data_dictionary.md, "Layout v2"): each candidate's axis once per
row group (`rt_axis`), each intensity trace from its first to its last nonzero
value (`intensity_trimmed`), and `trace_offset` / `trace_len` to rebuild it. Every
row passes through one `chromatograms::Encoder` in table order. The encoder counts
rows to know where each row group starts, so it and the writer take the row-group
size from one place (`chromatograms::row_group_rows`). The report and the run
manifest record schema version 2. Every table downstream of extract (features,
competed, scored, quant) is byte-identical to the v1 run's; only the chromatogram
table and its content hash change: on a six-run Astral and a five-run Orbitrap AIF
experiment the chromatogram tables went from 4.75 to 3.95 GB and from 21.7 to 15.4 GB
with every later table identical. A reader outside the engine, or an engine binary from
before v2, reads `rt` and `intensity`, which a v2 table does not have: set
`chromatogram_schema = 1` for it. Such a reader stops at the missing column rather than
misreading the table, and `mumdia::chromatograms::rewrite` converts the table back
to v1 for it.

On spectra that carry 1/K0, and only while a consumer of the list is on
(`ExtractParams::chrom_im`: the ion-mobility feature family, `features.im_features`;
retrace, section 9), every row also carries its per-point 1/K0: `im`, parallel to
`intensity`, in v1 (schema v3), and `im_trimmed`, cut to the run `intensity_trimmed`
keeps with the same `trace_offset`, in v2 (schema v4). The value at a point is the 1/K0
of the peak whose intensity the trace holds there (`Hit::im`, 1e-4 resolution), and
`+0.0` wherever the trace is `+0.0`, so the v2 trim loses nothing and both layouts
decode to the same list (`Decoder::row_im`). A never-observed fragment has an empty list
beside its empty trace; the MS1 rows hold `+0.0` at every point. The numbering keeps
`chromatogram_schema = 1` working on 4D data. `chromatograms::recorded_schema(path)`
reads the version back from the table (its layout and whether it has the list), and
every recording site uses it. The list adds about 38% to the table, so it is written
only for a consumer. The pooled-band splice copies whatever columns the bands wrote
and refuses bands whose schemas differ, and `chromatograms::rewrite` carries the list
in either direction.

To validate v2 on a new dataset, run the same configuration twice, once with
`extract.chromatogram_schema = 2`, and compare `features.parquet`,
`psms_competed.parquet`, `psms_scored.parquet` and the three quant tables with `cmp`.
`ci/smoke.sh` (arm 4e) does this on the fixture, ungrouped and grouped, and once more
with `MUMDIA_CHROM_ROW_GROUP_ROWS=1`. That environment variable is a test knob: it
sets the rows per chromatogram row group (default 65,536), which moves the seams of the
table and nothing downstream, so at 1 every row is its own group.

### Output: top-K peaks sidecar (`<out_psms>.peaks.parquet`)

Written only when `retain_top_peaks > 1` produced rows (`extract.rs:2544`), one
row per (candidate, peak). Columns: `candidate_id` U32, `peak_rank` I32,
`apex_rt` F64, `start_rt` F64, `end_rt` F64, `evidence_count` F64, `area` F64.
**These peaks are not scored** and do not affect FDR; they are candidate peaks
for an offline peak-selection model (see below). This sidecar is distinct from
`promote_top_peaks`, which instead emits extra `psms_extracted` rows carrying
`peak_rank >= 1`.

Each artifact (`psms_extracted` and `chromatograms`, but **not** the top-K peaks
sidecar) also gets an `<artifact>.report.json` (`extract.rs:2569`) recording the
row count, blake3 content hash, stage name, schema name+version, and
`elapsed_ms`. The `params` object (`extract.rs:2577`) carries `frag_tol_ppm`
(nominal), `effective_frag_tol_ppm` (post mass-cal), `frag_ppm_offset`,
`presence_min_fragments`, `presence_min_coelution`, `gate_min_score`, `gate_mode`,
`gate_coelution_min`, and `scan_window`. The `stats` map (`extract.rs:2562`)
carries `accepted` (the accepted-candidate count) and `scan_window`.
`model_identity` is `None` (no model in this stage).

## How it works

The control flow of `run` (`extract.rs:1300`) is: load library and windows, load
scans, build the matcher, accumulate peak-candidate hits (peak-major), then per
candidate run the acceptance cascade and apex selection, emit chromatograms and
MS1 XICs, and write the tables.

### The footprint guard

Before anything is probed, `run_hashed` estimates what the hit store will hold at once
(`window_loads`, `Footprint`, `check_footprint`) and refuses an extraction that cannot fit,
instead of growing until the operating system kills it. The estimate is the stage's largest
structure, the fragment-peak matches (16-byte `Hit`s, plus the 4-byte candidate id a probing
task keeps beside each until it groups them: 20 bytes a match):

- per isolation window group (`window_groups`), the sum over the candidates the window
  reaches (every k-th past 65,536, scaled by k) of the candidate's fragment count times the
  peaks of the window's scans inside its RT window (widened by the trace pad) times the
  chance that one peak falls within the fragment tolerance of one fragment;
- that chance takes the window's peaks as spread evenly over the log m/z span of its scans
  (a sample of 64): `2 * tol_ppm * 1e-6 / ln(hi / lo)` per peak. Under the 1/K0 gate only
  the share of the slot's 1/K0 range the candidate's window overlaps counts, and an
  allowlist drops the candidates it excludes;
- held at once: the largest batch of `windows_in_flight` consecutive windows on the streamed
  path, every window on the paths that accumulate the whole run (a two-pass peak claim, the
  contested features, the bucketed matcher).

The ceiling is `extract.max_footprint_gib`: unset (`auto`), the machine's physical memory
(the cgroup limit when a container sets a smaller one, read by `memlog::total_memory`),
looked up only once the estimate passes 4 GiB; a number is a ceiling of your own; `0` turns
the check off. Under `groups.parallel` each band in flight gets its share. The refusal names
the estimate, the ceiling and, when one window holds more than 100,000 MS2 spectra or the
run has a single window, the all-ion layout and the convert key that undoes the case it was
written for. The key is not part of the canonical configuration.

Every run logs the estimate (`extract: estimated hit store`, `resident_gib`, `run_gib`,
`busiest_window_scans`) and, on the streamed path, the matches the tasks actually returned
(`extract: task-local fragment indexes`, `hits`, `largest_batch_hits`), so the two can be
compared. It is an order of magnitude, not a prediction of the peak RSS: real peaks crowd
where fragments crowd, the batch total overstates what the sub-range flush keeps at once,
and the library, the spectra and the chromatogram buffers come on top. Measured, estimate
against the matches collected (largest batch, streamed path):

| layout | windows | in flight | estimated | collected | ratio |
|---|---|---|---|---|---|
| Q Exactive HF-X DIA (E. coli entrapment run) | 152 | 16 | 7.46 M | 8.40 M | 0.89 |
| Orbitrap Astral DIA (HYE, 15 min) | 250 | 16 | 17.7 M | 20.6 M | 0.86 |
| timsTOF Ultra 2 diaPASEF, native reader (HYE, 15 min) | 24 | 4 | 220 M | 462 M | 0.48 |
| Astral immunopeptidomics, 63 isolation-window groups (largest band) | 2 | 1 | 153 M | 155 M | 0.99 |
| ZenoTOF 8600 ZT Scan DIA, `isolation_half_width = 1.9` | 663 | 16 | 112 M | 166 M | 0.67 |
| the same run read as one all-ion window | 1 | 1 | 388,000 M (7,226 GiB) | killed past 521 GB | |

Over whole runs the estimate is 0.43 to 0.95 of the matches collected. The largest
estimate of the five ordinary layouts is 4.1 GiB (diaPASEF), far below the memory of any
machine that searches them; the all-ion layout is refused everywhere. A sample of peaks
against a sample of fragments, counted, is what gets within a factor of two: peaks spread
evenly over the m/z span of a scan read 3.4 to 8.5 times low, and two log m/z histograms
at 0.5% still 3.1 times, because library fragments repeat exact values (the y1 ion of
every tryptic peptide, the b2 ion of each residue pair) and spectra carry peaks there.
The estimate took 0.2 s at the start of the ZT Scan extract (663 windows, 5.9 million
candidates) and 0.04 s on an immunopeptidomics band.

The case it exists for is an all-ion layout: spectra whose isolation window is the whole
m/z range, which every spectrum then shares, so that every spectrum reaches every candidate
and no candidate can be written before the end of the run. A SCIEX ZT Scan DIA run read
without isolation offsets was such a layout (one window, 2.15 million spectra) and grew past
521 GB in 3.5 minutes; docs/04_convert.md, "SCIEX ZT Scan DIA".

### 1. Matcher and mass recalibration

The fragment tolerance and a systematic ppm offset come from `mass_cal`
(`extract.rs:1394`). The offset is applied as a **divisor**, not a subtraction:
`MassOffset::factor_at(mz)` returns `1 + ppm(mz)*1e-6` (`extract.rs:640`) and
every observed peak m/z is corrected by `q_mz = peak.mz / factor_at(peak.mz)`
before probing (`extract.rs:705`). The offset may be a single scalar ppm or, when
the seed fitted one, an m/z-dependent grid that is linearly interpolated and
clamped at the ends (`extract.rs:632`), so the correction is not necessarily
constant across the m/z range.

#### The per-peak setup: a per-scan scratch, and the per-window buffer that was not

`q_mz` is then binned, which is a `ln()`. `accumulate_groups` fills a task-local
`Vec<(f64, u32)>` with one scan's `(q_mz, bin)` and then probes from it, rather than
computing both inside the probe per peak. The scratch is a few KB, refilled per scan
and reused, and it makes exactly the same number of `ln()` calls the probe made. It
still pays, because a separate pass takes the `ln()` off the dependency chain that
the bin-cache load and then the posting loads hang from. Measured on
`rust/mumdia/crates/mumdia/tests/bench_fragindex.rs`, which is `#[ignore]`d and has
to be asked for:

- `bench_wall`, extract's own task shape over rayon, 32 threads, min of 9, two
  passes: -11.7 / -6.4% on the default shape (16 windows in flight, 4 tasks each),
  -15.1 / -17.9% with smaller windows (1-2 tasks each), -17.4 / -18.6% with one wide
  all-ion window in the batch.
- `bench_probe`, one thread, m/z-sorted peaks: -9.1 / -12.7 / -6.7% for a narrow /
  medium / wide candidate window.

The scratch carries `q_mz` and not just the bin, so `factor_at` still runs once per
peak. That matters under `search_seed.mass_cal_loess`, where the offset is a ~74
point grid and `factor_at` is a binary search plus an interpolation: 8.3 ns/peak
against 0.725 for the scalar default. A bin-only scratch, which recomputes `q_mz` in
the peak loop, turned that arm from -1.7% into +5.6%.

**A shared per-window buffer was tried twice and reverted twice.** The m/z half went
first (8 B/peak, +200 MB of peak RSS on the AIF fixture, 317 MB with the whole run in
one batch, no measurable time). The `ln()` bin half was then hoisted into a flat
`u32` buffer per window, shared by that window's sub-range tasks, and reverted in
turn: the buffer has to be filled before `rayon::scope` starts, so it turns per-task
work that was SPREAD ACROSS THE POOL into serial work on the calling thread, and the
default shape has nowhere near enough tasks per window to pay that back.
`tasks_per_window` is `(2 * threads).div_ceil(groups.len())`, which is 4 at 32
threads and the default 16 windows in flight, so the fill costs one pass per peak to
save three thirty-seconds of one; break-even needs `tasks_per_window >= threads`,
i.e. a batch of one or two windows. Measured, same harness:

The two numbers in each cell are two independent passes of min-of-9, on a machine
that was quieter for the first; read the direction and the sign, which are the same
in both, rather than the third digit.

| `bench_wall`, min of 9, 32 threads | in probe | per-scan scratch | shared buffer, serial fill | shared buffer, filled on the pool |
|---|---|---|---|---|
| default: 16 windows, 4 tasks each | 51.3 / 62.8 ms | -11.7 / -6.4% | **+8.5 / +22.1%** | -16.6 / -8.5% |
| 16 windows, 1-2 tasks each | 19.9 / 43.9 ms | -15.1 / -17.9% | **+38.9 / +30.0%** | -10.3 / -15.3% |
| one wide window, 49 tasks | 167.9 / 189.3 ms | -17.4 / -18.6% | -23.7 / -19.0% | -26.2 / -22.1% |

Filling it on the pool instead removes the regression, and it is still not worth it:
against the plain scratch it is -5.6 / -2.3% on the default shape and -10.7 / -4.3%
on the all-ion shape, and it LOSES 5.7 / 3.2% on the second row, for 4 bytes per peak
of every window in flight. The in-flight fraction is `windows_in_flight / n_windows`
-- 16 of 152 on the AIF benchmark, about 17 MB there, but the whole run's peaks on an
acquisition with 16 windows or fewer, and one copy per band under `groups.parallel`,
in extract, the tallest stage in the pipeline.

End to end, `mumdia extract` on the real AIF run (152 windows, 1.69M candidates,
465,806 MS2 scans, 32 threads, i9-13900KS, three binaries interleaved over 24 / 21 /
24 reps), timing the accumulation phase between the `loaded; probing peaks` and
`candidates with evidence` log lines so the 310 MB spectra read is not in the number:

| | min | mean of 3 fastest | mean of 5 fastest | median |
|---|---|---|---|---|
| in probe (before either change) | 3,274 ms | 3,360 ms | 3,419 ms | 4,233 ms |
| shared buffer, serial fill | **+14.0%** | **+13.2%** | **+14.0%** | **+9.3%** |
| per-scan scratch (shipped) | -0.3% | -1.9% | -2.3% | -4.5% |

The machine is noisy (reps spread 2.5x, and it is a hybrid CPU), so read the sign and
the ordering, not the third digit. The buffer's regression is the same size the model
predicts: one serial pass over the in-flight peaks, 39.6M x 5.4 ns = 430 ms, against a
3.3 s accumulation phase. The scratch's gain is smaller end to end than the -6 to -18%
`bench_wall` measures on the probe loop, because the accumulation phase also runs the
RT filter, the claim arbitration, the hit store and the counting sort, none of which
this touches.

All 69 runs wrote a byte-identical `psms_extracted.parquet` and
`chromatograms.parquet` (41,677 accepted in every one), and peak RSS did not separate
the arms (2,266-2,499 MB across all three).
`candidate_range_split_reproduces_the_unsplit_accumulation` runs the fixture at two,
eight and sixteen threads and compares the accumulation hit for hit, and `ci/smoke.sh`
wrote 127 byte-identical artifacts against a binary built from the parent commit (the
five that differ are logs and a work-directory path inside `planted.json`).

Two matcher backends dispatch through `Prober::probe` (`extract.rs:57`):
- `MatcherKind::Fragindex` (default): probes bins `bin-1 ..= bin+1` of the
  whole-library log-space geometry at the learned tolerance, verifies each posting
  with the exact f64 ppm predicate, and carries the **true generating fragment
  ordinal** in `post_frag`. On the default streamed path each probing task builds a
  `LocalIndex` over its own candidate sub-range and the global `FragIndex` is not
  built; the two-pass arbitration and `emit_demix_features`, which probe arbitrary
  candidate windows, build the global index as before (docs/06, "Task-local
  indexes"). The postings a task sees, and their order, are the narrowed global
  index's, so the outputs are the same either way
  (`candidate_range_split_reproduces_the_unsplit_accumulation` compares the two
  probes callback for callback on every task shape of its fixture and the two
  accumulations hit for hit).
- Bucketed fallback (`MatcherKind::Bucketed`): `Library::page_search`
  (`index.rs:350`) resolves the fragment ordinal by nearest stored m/z via
  `Library::local_frag_index` (`index.rs:325`). This is a semantic difference for
  fragments at sub-f32-identical m/z (`extract.rs:48`). The bucketed index is
  only built when this backend is selected, so choosing it is the only reason to
  pay for it (`extract.rs:1302`).

`Library::candidate_range` / `FragIndex::candidate_range` (`index.rs:341`,
`fragindex.rs:154`) give the half-open candidate-id range `[lo, hi)` whose
precursor m/z falls in an isolation window, exploiting that the library is sorted
by precursor m/z. This is what makes a per-window probe cheap.

### 2. Peak-major accumulation

The accumulator `acc: HashMap<u32, Vec<Hit>>` (`extract.rs:1435`) maps each
candidate to the observed hits it collected. A `Hit` is `{scan, frag, inten,
obs_mz}`, 16 bytes: `scan` is the index of the scan in the stage's `scans` slice,
whose `rt_seconds` is the hit's RT (`hit_rt`), and `obs_mz` is the peak's own f32
m/z. Until 2026-09-25 it was 24 bytes, carrying a copy of the scan RT as an f64 and
the f32 m/z widened to an f64; both are recovered exactly, so no value moved. Code
that depends on the order of hits sorts and groups them on the looked-up RT, never on
the scan index, and the two-pass elution profile still keys on the RT's bits, so two
scans that share an RT behave as they always did. Entries are created lazily on the
first collision.

There are three accumulation paths:

- **Streamed, single-pass** (`accumulate_groups`): the default, used with the
  fragindex matcher whenever the path is not two-pass, with or without a `restrict`
  allowlist. Scans are grouped by isolation window, a batch of
  `extract.windows_in_flight` windows is probed at a time, and each window is cut
  on a shared grid of candidate sub-ranges; one task probes one (sub-range, window)
  pair through its own `LocalIndex`. A sub-range is flushed to the per-candidate
  pass as soon as every window of the batch has reported, so the accumulator holds
  the windows in flight rather than the run. A flush hands the pass at most
  `CAND_CHUNK` candidates. When every candidate of a flush sits in one window's
  store as its only segment, which is the common case (a sub-range reached by one
  window, no leftovers from the batch before), the flush lends slices of that store
  directly (`single_run_span`, `flush_below`); otherwise the candidates are
  gathered and concatenated window by window into a reusable buffer first. The
  zero-copy flush hands over exactly the batches the gather would build, because
  each flush becomes one chromatogram chunk and the parquet page framing follows
  the chunk sizes (`the_zero_copy_flush_hands_over_the_batches_the_gather_builds`).
  Its effect on the AIF run (1.8M candidates, standalone extract, commit against
  its parent, interleaved on an i9-13900KS) is inside run-to-run noise: +0.4% wall
  at 32 threads (11 pairs), +1.9% and -2.7% at one thread (12 and 5 pairs), peak
  working set within 10 MiB. The copy it removes is at most about 1% of extract by
  the survey's estimate, so AIF cannot resolve it; it was not measured on a larger
  run.
- **Serial single-pass**: the fallback of the bucketed matcher. It honors every
  non-co-elution `peak_claim` strategy.
- **Two-pass co-elution** (`extract_twopass_windows`, `extract.rs:787`): used
  when `peak_claim` is one of the `Coelution*` variants **or**
  `emit_contested_features` is set (`extract.rs:1443`). Like the single-pass
  parallel path it partitions scans by isolation window and fans out over the
  ~150 windows with rayon (`extract.rs:782`), so it stays parallel even with a
  `restrict` allowlist. Pass 1 builds each candidate's per-scan elution profile
  (summed matched intensity, `extract.rs:872`). Pass 2 arbitrates each shared
  peak to the claimant with the highest competition weight at that scan
  (`extract.rs:1200`), tracks won/lost/apportioned contested intensity in
  `Contested` (`extract.rs:130`), and reassigns intensity per the co-elution
  claim variant (`extract.rs:1236`). The `reassign` flag (`extract.rs:1557`) is
  true for `CoelutionWinner`, `CoelutionProportional`, `CoelutionWinnerMargin`,
  `CoelutionDemix`, and `CoelutionShadow`, and for `CoelutionMultiCue` **only**
  when `claim_cues.reassign` is also set: `CoelutionMultiCue` ships
  non-destructive, feeding the contested/apportioned features only. When the
  two-pass path is triggered by `emit_contested_features` alone (a non-co-elution
  `peak_claim`), pass 2 still computes the contested statistics but the returned
  accumulation is the base full-intensity `acc1`, not the reassigned `acc2`
  (`extract.rs:1274`). So `emit_contested_features` adds the soft-competition
  features **without** altering any extracted intensity. `restrict` is honored
  inside both passes via the push closure (`extract.rs:853`, `extract.rs:907`).

Per-peak claim strategies (`PeakClaim`, `config.rs:132`; applied in the serial
single-pass loop at `extract.rs:1500` and mirrored in the parallel path at
`extract.rs:718`):
- `None`: every matching candidate gets the full peak intensity (legacy default).
- `WinnerPredictedIntensity`: only the claimant with the highest predicted
  intensity keeps the peak; ties break by lowest `candidate_id` for determinism
  (`extract.rs:1501`).
- `Proportional`: split the peak by predicted-intensity share (`extract.rs:1518`).
- `CoelutionWinner` / `CoelutionProportional` / `CoelutionWinnerMargin`: two-pass
  variants keyed on elution-profile height, arbitrated in `extract.rs:1236`.
  `CoelutionWinnerMargin` only strips a peak from the runner-up when the top
  eluter dominates by `peak_claim_margin` (`extract.rs:1218`); otherwise the peak
  stays shared, avoiding stripping real peptides at ambiguous peaks.
- `CoelutionMultiCue`: the per-claimant weight is the elution-profile height
  multiplied by the composable cues enabled in `ClaimCues` (`config.rs:192`):
  sub-tolerance m/z proximity, RT prior, MS1 precursor support, and so on
  (`claim_cue_multiplier`, `extract.rs:163`). Every cue defaults to 1.0, so with
  no cue enabled the arbitration reduces bit-identically to `CoelutionWinner`
  (`extract.rs:1184`).
- `CoelutionDemix`: at each scan (subject to `demix_scan_stride`) the co-isolated
  candidate-by-fragment design matrix is solved by ridge NNLS
  (`demix_solve_scan`, `extract.rs:279`) and each shared peak is split in
  proportion to the joint deconvolution rather than stripped winner-take-all.
  Always destructive.
- `CoelutionShadow`: solver-free shadow subtraction; each claimant's peak
  intensity is reduced by the portion other co-eluters explain.

These last three, and the `emit_demix_features` columns they share machinery
with, are recent and unvalidated. Treat them as research paths, not tuning knobs.

### 3. Per-candidate cascade (parallel over candidates)

`acc` is drained into `(cid, hits)` in sorted `candidate_id` order (`cand_hits`
at `extract.rs:1651`) for determinism, then each candidate is processed in
parallel via `into_par_iter().map(...)` returning `Vec<CandOut>`
(`extract.rs:1721`). Each candidate's work depends only on its own hits plus
read-only library/window/MS1 data, so `collect()` in the sorted order reproduces
the serial push order byte-for-byte (`extract.rs:1645`). The map returns a `Vec`
rather than an `Option` because `promote_top_peaks > 1` can emit more than one row
per candidate.

The cheap-to-expensive acceptance cascade, in order:

1. **Distinct-fragment presence (tier b)**: distinct matched fragments must be at
   least `presence_min_matched` (`extract.rs:1727`), else the candidate is dropped
   before any grouping work. `presence_min_matched`, `presence_min_fragments`, and
   `presence_min_coelution` are each floored at 1 via `.max(1)` (`extract.rs:1727`,
   `extract.rs:1926`, `extract.rs:1915`), so a configured 0 still requires at
   least one fragment.
2. **Scan grouping**: hits are rt-sorted (on the looked-up scan RT) and grouped
   into scan groups, deduping the same fragment within one scan by max. The groups
   are one dense `ScanGroups` per candidate: the group RTs, a `groups x width` f32
   value array and a presence bitmask of the same shape, `width` being one past the
   largest fragment ordinal among the candidate's hits. Until 2026-09-25 each group
   was a `BTreeMap<frag, intensity>`; the dense form answers the same questions
   (count, fragments ascending, lookup, the values summed in ordinal order through
   the same `Iterator::sum`) with the same values, so the f32 apex sum is
   deterministic and unchanged, and it allocates three buffers per candidate
   instead of a tree node per occupied group
   (`dense_groups_answer_what_the_trees_answered`). Because `width` is one past the
   largest observed ordinal rather than the number of observed fragments, the
   buffers hold `groups x (max observed ordinal + 1)` f32 values plus the presence
   words, at most `groups x n_predicted_fragments` values since ordinals are
   candidate-local. That can exceed the traces the candidate emits: a candidate
   that observed only ordinal 11 holds 12 values per group against one trace, and
   in sparse (non-grid) mode a trace covers only the groups where its fragment
   occurs. Measured on the AIF run against the tree form (standalone extract,
   interleaved, i9-13900KS), the change is inside run-to-run noise: -1.5% wall at
   32 threads (11 pairs, CPU time -0.3%), -1.5% and -3.7% at one thread (12 and 5
   pairs), peak working set within 10 MiB. The survey estimated at most 1-2% of
   extract; the saving grows with the scan groups per candidate, which AIF
   exercises little, and it was not measured on a larger run.
   Presence is a bit rather than a sentinel value, and a fragment first seen by a
   later hit of the same group starts from 0.0 before the max, exactly as the tree's
   `entry(..).or_insert(0.0)` did.
3. **Acquisition-scan grid projection** (`extract.rs:1782`): when
   `emit_window_grid` is on, the sparse groups are projected onto the full set of
   covering-window scans inside the RT window, so missing acquisition scans count
   as 0 and break a co-elution run rather than being invisible.
4. **Apex selection** (see below).
5. **Co-elution run**: the longest run of consecutive scan groups with at least
   `presence_min_coelution` fragments present (`extract.rs:1911`).
6. **Acceptance (tier c)** (`extract.rs:1926`): reject unless distinct fragments
   >= `presence_min_fragments`, `best_run >= scan_window` (the
   `fixed_scan_window` floor), `best_run >= min_coelution_run`, and
   `matched_fraction >= min_matched_fraction`. `matched_fraction` is
   `distinct / n_predicted` (`extract.rs:1925`); it is the primary symmetric
   discriminator (real peptides match a large fraction, chimeric and decoy matches
   a small fraction alike, keeping the target-decoy null valid).
7. **MS1 isotope evidence** (`extract.rs:1948`) computed *before* the Pearson gate
   so it can rescue a candidate. `ms1_support` requires a present mono and a +1/mono
   ratio in `[0.1, 1.5]` (`extract.rs:1960`).
8. **Pearson gate (tier d, optional)** (`extract.rs:2004`): only when
   `gate_min_score > 0.0`. It thresholds the score of the **active** `GateMode`
   and only the active score is computed (the closures at `extract.rs:1989` are
   lazy). If `ms1_rescue` is set, a gate failure is overridden when the candidate
   has MS1 support and enough matched fragments (`extract.rs:2020`).

The two presence thresholds (`presence_min_matched` at tier b,
`presence_min_fragments` at tier c) are the main coupling between this stage and
`convert`. Because the `--top-peaks-ms2` truncation is baked into the spectra
artifact and extract adds no cap of its own, a cap that removes most MS2 peaks
removes the fragment evidence these thresholds count, and the candidate is
rejected before any gate or score is reached. Measured on a 50-window Orbitrap
DIA run, `--top-peaks-ms2 300` discarded 78.6% of MS2 peaks; a `mumdia audit`
ladder restricted to peptides that an external library-free search confirms are
present in the run showed 49,105 of 78,782 (62.3%) stopping at
`candidate_generated` with rejection code `DID_NOT_SURVIVE_EXTRACTION` (`rejection.rs:62`).
Only 5,380 were lost to FDR and 355 to competition. Replaying those candidates
against the uncapped artifact recovered 41,948 of the 49,105 (85.4%), which is
what identifies the cap as the cause.

`DID_NOT_SURVIVE_EXTRACTION` (named `NO_PEAK_GROUP` until docs/29 #16) does not mean
"never assembled `presence_min_fragments` distinct fragments", although this document
previously said so. `audit.rs` reads a
per-candidate audit table that `extract` does not write -- `emit_candidate_audit`
is unwired -- so the reason map is always empty and the `_ => DidNotSurviveExtraction`
catch-all absorbs presence failures, matched-fraction failures and every
extraction-gate rejection alike. Read it as "did not survive extract". The
attribution to the peak cap above stands on the replay experiment, not on the
label.

When `DID_NOT_SURVIVE_EXTRACTION` dominates an audit ladder, check the conversion cap before
tuning the presence thresholds or gates here; see docs/04_convert.md for the
peak census and the cap dose-response.

### 4. Apex selection (`extract.rs:1793`)

The apex is chosen among scan groups whose smoothed distinct-fragment count
qualifies, then by a per-scan score:

- **Rolling-window count** (`apex_count_window`): a centered rolling **sum** of
  the per-scan distinct-fragment count (`extract.rs:1809`). It is deliberately a
  sum, not a mean: edge truncation makes interior positions accumulate more,
  center-weighting the apex toward the RT-window centre (a mild RT prior). Window
  1 reproduces exact per-scan counts. The default `"auto"` (stored as 0) is resolved
  at the start of the stage from the instrument `convert` recorded in
  `<spectra_ms2>.report.json` (`params.instrument`, `instrument.rs`): 1 on an Orbitrap
  Astral, 5 on any other instrument and on spectra converted before the field existed.
  The resolved window and the instrument are in the `psms_extracted` report. Measured one
  setting at a time against the 0.6.0 defaults (2026-10-11), window 1 against 5, mean of
  three NN seeds: Astral HYE +1.3% peptides, the ProteoBench Astral entrapment module
  +1.0% precursors (paired FDP 0.86% -> 0.84%), an Astral single cell +2.4%, an Astral
  HeLa entrapment run +3.6% human precursors at a measured FDP of 1% (FDP at the reported
  1%: 0.89% -> 0.71%); Orbitrap (one seed) AIF HYE -0.9% and the AIF entrapment run -0.8%.
  The new engine resolving "auto" was byte-identical to the previous engine given the
  window explicitly, on Astral (1) and Q Exactive (5). Only scans with smoothed count `>= maxc -
  apex_count_tol` qualify (`extract.rs:1848`). An opt-in Gaussian smoother over
  the same profile is available via `apex_gaussian_sigma_scans` (default 0.0 =
  off, `extract.rs:1813`).
- **Signature ions** (`apex_top_fragments`, default 3 via `k_sig`,
  `extract.rs:1860`): the top-K predicted fragments, so a bright interferent on a
  non-signature ion cannot define the apex.
- **RT prior** (`apex_rt_prior_s`): when > 0 and `rt_cal > 0`
  (`extract.rs:1855`), each qualifying scan's score is multiplied by
  `exp(-0.5*((rt - rt_cal)/sigma)^2)` (`extract.rs:1886`).
- **Scoring mode** (`apex_evidence_rank`, `extract.rs:1890`): when true, the score
  is `n_frag + sig_sum/(sig_sum+1)` times the prior, so the number of distinct
  co-eluting predicted fragments dominates and observed signature intensity only
  breaks sub-integer ties. This is interference-resistant in wide-window DIA
  because intensity is chimeric. It is the default; when false, the legacy
  `sig_sum * prior` is used (`extract.rs:1902`), bit-identical to the pre-feature
  behaviour. In practice the tie-break carries no intensity information:
  `sig_sum` is an observed intensity in the thousands, so `sig_sum/(sig_sum+1)` is
  about 0.9999 at every scan, and among scans of equal count the RT prior decides.
- **Intensity inside the count region** (`apex_refine_intensity`, default false,
  benchmark-gated): the qualifying scans, widened by the rolling half-width
  (`apex_count_window / 2`), form the region where the peptide elutes; the apex is
  the scan in that region with the largest signature intensity times the prior. The
  count still decides where the peak is, intensity which scan is its top. Without
  it the count also picks the scan, and on a tailing peak the rolling count is
  largest after the intensity maximum. Measured on Astral REP1 with
  `apex_count_window` 5 (88,092 accepted precursors, `bench/apex/apex_offset.py`): the summed fragment intensity
  peaked one scan BEFORE the chosen apex for 41.1%, at it for 40.7% and one scan after
  for 5.6%. A replica of the rule (`bench/apex/apex_replica.py`, reproducing the engine's apex
  for 88,089 of 88,092) showed that neither the tie-break nor the prior causes it
  (removing either moves the 41% to 40%), and that this rule puts 85% at the
  intensity maximum with the rest symmetric (5.8% before, 5.1% after). Window 1 (the
  engine default) has a smaller, opposite-signed spread (15% before, 11% after).

`apex_intensity` reported is the full summed intensity of the winning scan group
(`extract.rs:1907`), not the signature-only score.

### 5. Gate scoring internals

Spectral-agreement scores share a helper `peak_window` (`extract.rs:523`) that
finds the contiguous elution-peak scan range `[lo, hi]` around the signature-ion
apex (scans above 10% of the reference apex height), returning `None` when there
are fewer than 3 scans or no reference signal. This restriction matters: over the
full wide extraction window the traces are mostly zeros and any correlation is
noise, so the co-elution and spectral gates are only meaningful across the
elution peak itself (`extract.rs:520`). `GateMode` scores:
- `ApexPearson`: Pearson of observed-vs-predicted intensities at the single apex
  scan (`extract.rs:1989`); one chimeric scan can dominate it.
- `PeakSpectral`: `peak_spectral_score` (`extract.rs:564`) correlates the
  peak-summed observed spectrum (each fragment integrated over the peak scans)
  with predicted intensities, averaging out a single interfered scan.
- `SpectralEntropy`: Li spectral-entropy similarity of the sqrt-transformed apex
  spectrum via the shared `features::entropy` kernel (`extract.rs:1995`).
- `Coelution`: `coelution_gate_score` (`extract.rs:590`), the
  predicted-intensity-weighted mean Pearson of each matched fragment's XIC to the
  signature-ion reference profile, restricted to the elution peak. Orthogonal to
  intensity agreement (temporal, not shape).
- `Combined`: requires `peak_spectral >= gate_min_score` **and** `coelution >=
  gate_coelution_min` (`extract.rs:2016`).

All four diagnostic scores are computed for every accepted candidate only when
`emit_gate_diagnostics` is set (`extract.rs:2035`); otherwise they are zeroed and
not written, so the default chain pays no extra cost.

### 6. Chromatograms and MS1 XICs

Per-fragment observed m/z is an intensity-weighted mean per fragment
(`extract.rs:2073`) for mass-accuracy features. Every predicted transition emits a
chromatogram row (`extract.rs:2095`); observed fragments carry the grid-sampled
(or rt-sorted) trace, absent ones an empty trace. When MS1 is present and grid
mode is on, three MS1 isotope XICs (`ms1_mono`, `ms1_iso1`, `ms1_iso2`) are
sampled on the same scan grid via nearest-MS1-scan lookup (`extract.rs:2151`),
using `ISOTOPE_SPACING / charge` (`constants.rs:22`, value `1.003354835`) for the
isotope offsets and `sum_near` (`extract.rs:495`) to integrate within
`prec_tol_ppm`. The apex MS1 isotope intensities (`ms1_isom1`/mono/`iso1`/`iso2`)
are the per-PSM columns, taken from the nearest MS1 scan to the apex RT
(`extract.rs:1957`).

The chromatogram rows of each candidate chunk are handed to one writer thread
through a two-slot channel, so encoding overlaps extraction. The stage logs one
`extract: chromatogram writer` line with `chunks`, `send_blocked_ms` (time the
extraction side waited for a free channel slot, the column build excluded) and
`writer_busy_ms` (time the writer thread spent in `write_cols`, plus the final
`close`, which runs on the extraction side once the loop has ended). A
`send_blocked_ms` near zero says the serial encoder does not bound the stage; one
that approaches the stage's accumulation time says it does, and that a parallel
column encoder (survey item R2) would pay off here. The line is log output only.

On a 4D run the chromatogram writer encodes whole row groups as jobs on the parquet
codec pool while the next fills (`TableWriter::with_row_groups_in_flight`, up to
`chromatograms::ROW_GROUPS_IN_FLIGHT` = 16 and never more than the pool has threads;
docs/03_io_layer.md, "Parallel column codec"). The bytes are the serial writer's. The
one writer thread was the whole of a HYE diaPASEF extract after its first 20 s. A 3D
table is written as before; enabling it there waits for a wall-time and RSS measurement
on an Astral, an Orbitrap and a banded run.

### 7. Top-K peak enumeration (`retain_top_peaks`)

Two independent knobs consume the enumerator, and they must not be confused:
`retain_top_peaks` writes the **unscored** `.peaks.parquet` sidecar, whereas
`promote_top_peaks` emits additional `psms_extracted` rows with `peak_rank >= 1`
that do flow downstream. Both default to 1 (off).

When `retain_top_peaks > 1` and the candidate has groups (`extract.rs:2173`), the
per-scan distinct-fragment count profile (`count_prof`, `extract.rs:2176`) is
passed to `crate::peaks::enumerate_peaks` (`peaks.rs:52`) with `k =
retain_top_peaks`, `bound_fraction = 1/3` and `min_prominence_frac = 0.1`
(`extract.rs:2177`). `enumerate_peaks` is a pure, side-effect-free function
(`peaks.rs:1`):
1. An empty profile, `k == 0`, or an all-non-positive profile returns an empty
   vector (`peaks.rs:58`, `peaks.rs:63`).
2. **Local maxima** (`peaks.rs:71`): index `i` qualifies when its value is `>
   0` and `>= prom_floor` (the prominence floor `min_prominence_frac *
   global_max`, `peaks.rs:66`, tested at `peaks.rs:74`), is **strictly** greater than its left neighbour
   (or is the left edge), and is `>=` its right neighbour (or is the right edge).
   The strict-left / non-strict-right rule makes a flat plateau register once, at
   its left edge; edges count as maxima against their single neighbour.
3. **Boundary walk** (`peaks.rs:87`): from each apex, walk left and right,
   stopping when the profile drops below `bound_fraction * apex` **or** turns back
   upward (a valley); `area` is the integrated profile over `[start_idx,
   end_idx]`.
4. **Dedup + rank** (`peaks.rs:123`): peaks are sorted strongest-first by `area`,
   then by `apex_intensity`, then by earliest `apex_idx` (`peaks.rs:125`); a
   maximum whose apex falls inside an already-kept peak's `[start_idx, end_idx]`
   envelope is dropped (`peaks.rs:140`); the first `k` survivors are kept and
   assigned `rank` 0..k-1 (`peaks.rs:148`).

`enumerate_peaks(.., k=1, ..)` returns the single global-argmax peak, so callers
can adopt it incrementally. The retained peaks rank by **co-eluting fragment
breadth (count), not intensity**, per finding A6 in
docs/18_findings_and_decisions.md (intensity is chimeric in DIA). Back in
`extract`, each returned `PeakGroup` is mapped to a sidecar row
(`extract.rs:2179`): `apex_rt`/`start_rt`/`end_rt` are the RTs of the
apex/start/end scan groups, `evidence_count` is the distinct-fragment count at
the apex group (`groups[pk.apex_idx].1.len()`), and `area` is `pk.area`. At the
default `promote_top_peaks = 1` the PSM table still reports only the single
selected apex, so FDR is unaffected and the sidecar peaks are unscored candidate
peaks for an offline peak-selection model (`extract.rs:1700`).

`promote_top_peaks > 1` (`extract.rs:2238`) is the separate, opt-in path that does
change FDR: it re-enumerates the same count profile (`extract.rs:2252`), filters
alternates by `alt_peak_min_area_frac` and `alt_peak_min_separation_s` plus the
`presence_min_matched` floor (`extract.rs:2292`), and pushes up to
`promote_top_peaks - 1` extra `CandOut` rows with `peak_rank >= 1`
(`extract.rs:2297`). Because these become real scored rows, enabling it changes
the training/FDR population and needs entrapment validation, not a count check.

### 8. Ion mobility (4D data)

Everything here applies only when the decoded spectra carry per-peak 1/K0
(`has_im`: the scans' `im` lists, which `load_ms2`/`load_ms1` decode only from spectra
v2, the native diaPASEF reader's). A 3D run takes none of these paths, and its
artifacts and reports are byte-identical to before (`extract_writes_the_1k0_evidence_only_for_spectra_with_ion_mobility`
in `tests/pipeline.rs`, and the smoke fixture).

**One window type and one reader.** `rt_im_train::RtWindows` carries, besides the RT
arrays, `im_lo`/`im_hi` (f32, `-inf`/`+inf` for a candidate without a window) and
`im_cal` (`Option<f64>` per candidate), all `None` when no row has a 1/K0 window, which
is every 3D run. `read_run_windows` is the one reader of `run_windows` and fills them
from `im_lo`, `im_hi` and `im_pred_cal`, refusing a NaN 1/K0 bound with the row named,
as it refuses a NaN RT bound. `rt-im-train` fills the same fields while it writes the
table (`RtWindowsBuilder`, only when the fit has an ion-mobility calibration, and
withholding the in-memory form on a NaN 1/K0 bound), so the in-memory handover of
`run`, `run-experiment` and the RT bootstrap is the file, 1/K0 included
(`im_windows_are_the_calibration_applied_to_every_row`,
`the_windows_builder_scatters_the_im_columns_as_extract_reads_them`).

**The window test.** Every guard site asks `RtWindows::admits(c, rt, im)`, through a
`Guard` that knows whether the 1/K0 gate is on: the streamed probe (`ProbeTask`), the
serial path, both passes and the EM, demix and shadow arms of the two-pass
arbitration, and the per-scan demix of `emit_demix_features`. With `im` `None` (gate
off, or a 3D scan) it is `!(rt < rt_lo || rt > rt_hi)`, exactly the test it replaces,
including the local asymmetric RT windows `rt-im-train` writes.

**The 1/K0 gate** (`extract.im_gate`, default `off`). `fragments`: a fragment peak whose
1/K0 lies outside the candidate's window `[im_lo, im_hi]` is not matched to it, as a
peak outside its RT window is not. `auto`: `fragments` when the spectra and the windows
both carry 1/K0, `off` otherwise. An explicit `fragments` without either runs ungated
with a warning. The window is the calibration's `w_im` (docs/08 section 7), and the
gate's optimum follows it: on the IM branch an `im_window_multiplier` of 0.5 lost 11%.
The MS1 reads are not gated: the IM branch's `fragments_ms1` variant measured -0.6%,
within noise, and is not ported. `claim_cues.ms1_support` and `ms1_rescue` therefore
stay ungated on 4D data as well (both off by default). The `window_mean` gate's summed
spectrum is built from the hit groups, which the 1/K0 gate filters, so it reads gated
evidence with no further change; its scan grid stays m/z-keyed. The isolated gate
effect was measured only on E. coli (+5.7%, three seeds); promoting the default to
`auto` needs a paired gate-on/gate-off entrapment arm and HYE diaPASEF seeds. On main the
gate and the ion-mobility feature family together gave +5.0% on the same run (one seed,
docs/35 section 11). The diaPASEF recipe (`configs/examples/diapasef.json`) sets
`im_gate = fragments`.

**Evidence** (psms_extracted v4), computed per accepted row at the row's own apex, so a
promoted alternative peak carries its own (`im_at`). The apex scan is the one the demix
resolves (`demix_apex_scan`), which the `rt_scan` lookup map is built for whenever
`emit_demix_features` is on or the spectra carry 1/K0 (lookup only, never iterated).
`apex_im` is the seed's `observed_im` rule at the extraction apex: each library
fragment takes its nearest peak within the fragment tolerance of its offset-corrected
m/z (`spectra::nearest_peak`, the rule of the seed's calibrants), a peak outside the
1/K0 window is skipped under the gate, and `apex_im` is the intensity-weighted median of
their 1/K0 (`stats::weighted_median`), `apex_im_mad` their weighted mean absolute
deviation from it (`stats::weighted_mean_abs_dev`). `ms1_apex_im`: in the MS1 scan
nearest the apex, the peak within `prec_tol_ppm` of the precursor m/z, shifted by the
learned `ms1_ppm_offset` as every MS1 read is, whose 1/K0 lies nearest `apex_im`, or the
most intense one without an `apex_im` (`spectra::ms1_peak_near`). Nearest rather than
most intense, because at low load the most intense in-tolerance MS1 peak is often
another ion in another mobility band. It is not gated. `im_pred_cal` is copied from the
windows. Each hit carries its peak's 1/K0 (x 1e4 as a u16 in the two padding bytes of
`Hit`, which stays 16 bytes), which is what the chromatograms' per-point list is built
from.

**MS1 mass calibration** (`ms1cal::estimate`). On 3D data the offset is the median ppm
error of the most intense MS1 peak within 15 ppm of each anchor. On 4D data an anchor
with an `observed_im` takes, over the same three scans, the in-tolerance peak whose
1/K0 lies nearest it, by the same `ms1_peak_near` rule; an anchor without one keeps the
most-intense rule.

### 9. Raw-event trace rebuild (`retrace`, diaPASEF, default off)

`retrace.enabled` adds the `retrace` stage between extract and features
(`stages/retrace.rs`). It rewrites `chromatograms.parquet` from the raw timsTOF events of
the `.d`, with the same rows in the same order, in the same layout and schema version as
extract's table (v3 or v4, the per-point 1/K0 list included), so features, quant and the
pooled readers read it unchanged. It then patches `psms_extracted` once, in place. Off,
the stage does not run and every artifact is byte-identical
(`the_chain_without_retrace_is_byte_identical_and_refuses_retrace_on_3d_input` in
`tests/pipeline.rs`). docs/35_diapasef.md section 6 has the measurements.

Why: an extract trace point is the intensity of one convert centroid, and convert's
centroiding loses weak signal (single-linkage m/z chaining, `convert.tdf_min_points`, the
scan gap). The MS1 isotope traces are also ungated in 1/K0, and on the E. coli diaPASEF run
about two thirds of their signal came from other ions at the same m/z.

What a rebuilt point is, per chromatogram row:

- fragment rows (`predicted_intensity > 0`): the sum of every raw event in the grid
  point's frame (the MS2 frame within 1 ms of the point's RT), in the quad slots whose
  isolation window holds the precursor m/z, with TOF inside the learned fragment tolerance
  of the calibrated fragment m/z, and 1/K0 inside `apex_im +/- retrace.im_half_width`
  (`im_pred_cal` when `apex_im` is null). The m/z calibration is extract's, read through
  the one reader `masscal::read_mass_cal` and applied with `MassOffset::factor_at`. The
  point's 1/K0 is the intensity-weighted 1/K0 of the events, and `+0.0` where they sum to
  nothing, so both layouts decode to the same list. A point counts any event with an
  intensity above 0 (the IM branch's `min_events` floor is not kept: at its default of 1
  it was a no-op, and 2 or 3 lost identifications);
- MS1 rows (`ms1_mono`, `ms1_iso1`, `ms1_iso2`, when `retrace.ms1`): the same sum over the
  MS1 frames extract's MS1 read takes at the grid RT (`ms1_neighbours`: the nearest frame
  and its `extract.ms1_scan_halfwidth` neighbours, the largest sum kept), within
  `extract.prec_tol_ppm` of the row's own m/z and inside the same 1/K0 band. The row's m/z
  already carries extract's learned MS1 offset (`(precursor_mz + dmz) * (1 +
  ms1_ppm_offset * 1e-6)`), so retrace never applies the offset again. It checks it
  instead: the report records the median ppm residual of the raw `ms1_mono` events against
  the row m/z (`stats.ms1_mass_residual_ppm_median`), and the stage warns when it exceeds
  half the precursor tolerance. A half-width of 0 is the IM branch's nearest-frame rule;
  the prototype's +1.6% was measured with it, at 20 ppm, before `ms1_scan_halfwidth`,
  `prec_tol_ppm = 3` and `ms1_calibrate` existed (arm T10 measures the difference);
- every other row and column (`frag_obs_mz` included) is copied.

The TOF -> m/z converter and the 1/K0 calibration are convert's (`convert::RawTdf`:
`tof_of`, `mz_of`, `im_of_scan`), so raw events, centroids and `apex_im` share one scale.
The rule reads no label.

**Apex re-pick** (`retrace.repick`, default off). Before any trace is built, retrace scores
extract's apex and every `<psms_extracted>.peaks.parquet` peak of the candidate
(`extract.retain_top_peaks`, 5 measured) on the raw events. For each peak it takes the MS2
frames within 1.5 s of that peak's apex and bins each fragment's events at 0.002 over the
`run_windows` 1/K0 window. For each +/-0.015 band (15 bins) it computes the cosine of the
square-rooted band sums against the square-rooted predictions, and the
prediction-weighted band intensity `I`. The band with the best `cos x sqrt(I / I_max)`
scores the peak `cos x ln(1 + I)`, and the peak with the highest score wins; ties go to
extract's apex, then to the lower rank. A candidate without signal keeps extract's apex.
The winner's apex RT and band centre (the prediction-weighted 1/K0 centroid of the profile
inside the band) replace `apex_rt` and `apex_im`, and the traces and the apex scalars are
built around them. The re-pick reads no retention time. It overrides the apex that
`apex_evidence_rank` chose, by design. It needs `retain_top_peaks >= 2` and
`promote_top_peaks = 1`, and is refused with `apex_refine_intensity` (`Config::validate`).
`apex_im_mad` and `ms1_apex_im` stay those of extract's apex. Diagnostic:
`MUMDIA_REPICK_DUMP=<path>` writes every scored peak (candidate, order with 0 for
extract's apex, apex RT, score); the pick is the same either way.

**The psms_extracted patch.** Retrace patches `psms_extracted` once, per rank-0 row, and
records what it did under `params.retrace` in the table's report (`patched_columns` and
the row counts). The schema version stays 4.

- `apex_rt` and `apex_im` of a re-picked candidate.
- `apex_intensity` (the sum over the fragment rows of the trace point nearest the apex)
  and `n_matched_fragments` (fragment rows with any signal), whenever a fragment row of the
  candidate was rebuilt. Raw traces beside centroid scalars would leave the table
  inconsistent (the scalars alone measured +0.6% on the IM branch).
- `ms1_mono`, `ms1_iso1` and `ms1_iso2` at the point nearest the apex, each when its row
  was rebuilt. A refreshed value stays null only where extract had none and the raw trace
  has no signal at the apex either.
- `ms1_isom1` is not refreshed. There is no `-1` isotope trace to rebuild it from, so it
  stays extract's centroid value, summed over the MS1 scans without a 1/K0 band
  (IM-blind).
- `imc_ref_w` only under `retrace.apex_features` (a new column, read by name, 0.0 without
  a record). Over the rebuilt fragment rows, a raw 1/K0 profile per fragment in
  `apex_im +/- 0.04` (40 bins), summed over the MS2 frames within 1.5 s of the apex, in the
  quad slots holding the precursor and inside the fragment's TOF window. The score is the
  predicted-intensity-weighted mean leave-one-out reference correlation of each profile
  with signal (`stats::loo_ref_corr`, the kernel the co-elution and interference families
  use). The ion-mobility feature family carries it as its tenth feature (docs/10).

A rank-0 row whose candidate has no rebuilt rows keeps extract's values, with a warning.
The other readers of `psms_extracted` see the patched values: the shadow demotion's 1/K0
term (`extract::apex_im_by_peak`) reads the re-picked `apex_im`, and quant integrates
around the `apex_rt` the scored rows carry from features.

**Where it runs.** `retrace::run_if_enabled`, inside the extract-to-compete core
(`chain::extract_to_compete`), so in `run`, every run of `run-experiment`, and the second
pass. The core takes a flag (`ChainParams::retrace`): the RT bootstrap pass passes it off,
because that pass only collects anchors. Extract then writes its centroid traces, with the
per-point 1/K0 list, to `chromatograms.centroid.parquet`; retrace writes
`chromatograms.parquet`, and the centroid table (12-17 GB on a HYE diaPASEF run) is deleted
afterwards through the helper that also deletes a grouped run's band tables
(`stages::delete_intermediate`). The manifest records each table once, as the later stages
read it. The `.d` path comes from the spectra report (`params.mzml` with
`reader = timsrust`); a run whose spectra were read from mzML, including a timsTOF mzML
from msconvert, is refused with the reason, as are `groups.window_groups > 1` and
`extract.promote_top_peaks > 1` (`Config::validate`). The standalone `mumdia retrace`
takes the same inputs by path (docs/23), and `mumdia extract` writes the 1/K0 list for it
under `retrace.enabled`.

**How it runs.** The frame index is read without decoding, and every frame inside the
extracted candidates' RT windows (plus 5 s) is decoded once and held as a `TofFrame`
(`convert::tims_frames`): its events sorted by TOF and packed into one `u64` each, with a
table of the first event of every 64-index TOF bucket, so a TOF window is one short binary
search (`TofFrame::first_at`, pinned to the whole-frame `partition_point` by a unit test).
The chromatograms are read by the table's actual row groups, so a v2 table is always
decoded from a row-group start, on one reader plus up to seven leased from a process-wide
pool (`sched::DecoderLease`, which also bounds the features stage's decoders). They are
summed in parallel per candidate, with the frames and slot scans planned once per
candidate grid, and written through `chromatograms::writer` with whole row groups encoded
on the codec pool (`TableWriter::with_row_groups_in_flight`,
`chromatograms::ROW_GROUPS_IN_FLIGHT`). The writer publishes only after the last row group,
so an error leaves no partial table. Peak memory is the held frames. The report's `stats`
carry the rebuilt and copied row counts, the frames decoded, the readers, the raw events
summed, the fragment grid points without a frame, the MS1 residual and the time of each
phase.

### 10. Scanning-quadrupole data (SCIEX ZT Scan DIA, default off)

A ZT Scan DIA run read with `convert.isolation_half_width` (docs/04_convert.md, "SCIEX ZT
Scan DIA") has one isolation window per Q1 bin: 663 windows of 3.8 m/z whose targets step
0.7614 m/z, each sampled once a 1 s cycle, the bins 1.345 ms apart. A precursor falls in
about five of them, so its candidate's chromatogram grid (the scans of every covering
window) interleaves five bins a cycle, each at its own transmission. Three keys model the
sweep instead. All three are off by default and omitted from the canonical configuration,
so a 3D or 4D search without them is byte-identical; each is refused on any other layout.

- The layout (`Q1Layout`): the distinct windows in `WindowKey` order, and a cycle per sweep,
  read off the scans in acquisition order (a new cycle where a bin's target falls back by
  more than half a step, or where a window repeats). It must have at least 50 windows
  stepping less than 1.5 m/z and at least 1.5 steps wide, the thresholds of convert's
  scanning-quadrupole diagnosis; anything else, and spectra with 1/K0, is refused with the
  layout named. The extract log and reports record the windows, cycles and step.
- `extract.q1_cycles` sums a candidate's matches in the bins of one cycle into one point
  (`q1_merge`): every hit moves to the scan of the candidate's anchor bin (the covering
  window whose target is nearest the precursor m/z) in its own cycle, one fragment's
  intensities at one point are summed in hit order (in f64, so deterministically), its
  observed m/z is their intensity-weighted mean, and the chromatogram grid is the anchor
  bin's scans. Hits in a cycle where the anchor bin has no scan (only at the ends of a
  regular sweep) are dropped. Every score, gate and trace after it reads the merged hits.
- `extract.q1_profile` writes each accepted row's Q1-profile evidence (psms_extracted v5,
  four Float64 columns appended last, `q1_profile_of`). Over the unmerged hits of the five
  cycles around the apex cycle, the matched intensity in each covering window is compared
  with the share of the precursor that window transmits: a window of half-width `h` swept
  over one step `s` during its bin passes a precursor at distance `d` from its target
  whole while `d <= h - s / 2`, not at all beyond `h + s / 2`, and linearly between.
  `q1_corr` is their Pearson correlation; `q1_offset` the intensity-weighted centroid of
  the window targets minus the expected one, in half-widths; `q1_frag_sd` the
  intensity-weighted spread of each fragment's own centroid around the observed one, in
  half-widths; `q1_bin_frac` the share of covering windows with signal. A fragment shared
  with a precursor at another m/z sits in the bins that transmit that one, which pulls
  its centroid away. Promoted alternative peaks carry their own.
- `extract.q1_gate_max_offset` rejects a row whose `|q1_offset|` passes it; it needs
  `q1_profile`. `features.q1_features` appends the four columns, under their own names,
  after every other column of the `extended` set (`features::Q1_FEATURES`); it needs
  `extract.q1_profile`, and a table without the columns is an error.

Measured on the ZenoTOF 8600 K562 REP01 run (`isolation_half_width = 1.9`, imported
library with predicted decoys, the shipped `nn_torch` recipe; precursors at
`precursor_q`, peptides at `peptide_q_value`, protein groups at `pg_q_value`, 1%; seed 0
unless noted, seeds 1-3 rescoring the same competed table):

| arm | precursors | peptides | protein groups | precursors vs base |
|---|---|---|---|---|
| base (no Q1 key), seed 0 | 97,401 | 78,612 | 7,742 | |
| base, mean of seeds 0-3 (precursor range) | 97,317 (96,873-97,541) | 78,627 | 7,721 | |
| `q1_profile` + `features.q1_features`, seed 0 | 97,805 | 78,951 | 7,660 | +0.4% |
| the same, mean of seeds 0-3 (precursor range) | 97,724 (97,203-98,037) | 78,920 | 7,708 | +0.4% |
| `q1_cycles` | 84,043 | 68,675 | 7,187 | -13.7% |
| `q1_cycles`, `apex_count_window = 1` | 86,340 | 70,442 | 7,346 | -11.4% |
| `q1_cycles` + `q1_profile` + `features.q1_features` | 85,898 | 69,832 | 7,263 | -11.8% |
| three runs as one experiment, base | 110,433 | 88,958 | 8,981 | |
| three runs as one experiment, `q1_profile` + `features.q1_features` | 110,860 | 89,381 | 9,012 | +0.4% |

| arm (entrapment library, mean of seeds 0-3 and range) | human precursors | entrapment precursors | combined FDP |
|---|---|---|---|
| base | 95,115 (95,050-95,191) | 376 (365-391) | 1.39% (1.35-1.44%) |
| `q1_profile` + `features.q1_features` | 95,223 (95,002-95,438) | 359 (345-369) | 1.32% (1.27-1.36%) |

The Q1 profile separates (profile arm, seed 0): its correlation has a median of 0.68 for
the targets accepted at 1% against 0.13 for all extracted targets and 0.07 for the decoys,
and the fragments' centroid spread 0.08 against 0.16 half-widths for both. It adds little,
most likely because the interleaved grid already carries the same evidence: every fragment
of a precursor rises and falls with the same transmission across the bins of a cycle,
which the trace correlations read, and an interfering fragment does not. `q1_bin_frac` is
1 for every row (some peak matches in every covering bin), so the rescorer drops it as a
constant column. Summing the bins removes the modulation and leaves a fifth of the
points; with the shipped extraction settings 28% more candidates pass extract and 12-14%
fewer precursors are identified. The gate is no use at any threshold measured: the
centroid offset of 90% of the accepted targets lies within -0.11 to +0.11 half-widths,
and `|q1_offset| > 0.3` removes 4.0% of the extracted targets and 4.1% of the decoys
alike.

All three keys stay off and benchmark-gated: a second ZT Scan acquisition could not be
converted with its isolation targets for this measurement (docs/04_convert.md, "SCIEX ZT
Scan DIA"), and CLAUDE.md asks for one before any default changes.

The Q1 calibration that the per-fragment decisions of a later change are to read
(`rt_im_train.q1_calibrate`, docs/08_rt_im_train.md, section 4h) found the window centred
on its target and wider and softer than its nominal 3.8 m/z: whole within 0.57 m/z of
its centre and nothing beyond 4.56 m/z. The trapezoid of `q1_profile` above is the nominal
one (one step of edge, the half-width as read).

### 11. MS1 mass calibration: the surface and the tolerance (default off)

Before extract, `run` and `run-experiment` learn the MS1 offset from the confident seed
anchors (`stages::ms1cal`, `extract.ms1_calibrate = true`, the default since 2026-10-08):
the median signed ppm error of the most intense MS1 peak within 15 ppm of each anchor's
precursor m/z, in the MS1 scan nearest its observed RT and one either side (section 8 for
the 4D rule), and every MS1 read of extract is centred on it, `prec_tol_ppm` (3 ppm) wide.
Two opt-in keys extend that:

- **`extract.ms1_calibrate = "surface"`** fits the anchors' errors as an additive smooth
  function of precursor m/z and retention time (`ms1cal::fit_surface`): the anchors whose
  error lies within four robust SD of the median (a peak another ion put in the 15 ppm
  window is not an MS1 error), at most 40,000 of them (every k-th in seed order), a robust
  LOESS of m/z and one of retention time (span 0.3, two bisquare iterations), backfitted
  over three rounds, the retention-time term centred on the anchors, each sampled on a
  20-point grid over the anchors' range and held flat beyond it (`Ms1Surface`). Every MS1
  read of extract then takes the offset at its own m/z and retention time
  (`ExtractConfig::ms1_shift`): the apex isotope intensities at the apex, each point of the
  three MS1 grid traces at its own retention time (the row's m/z is the one at the apex),
  and `ms1_apex_im`. The MS1 support of `claim_cues` reads at the library m/z, as it does
  under the one offset. With fewer than 200 inlier anchors the one offset is used, with a
  warning. `true` and `false` keep the spelling and the configuration hash they had.
- **`extract.prec_tol_auto`** sets `prec_tol_ppm` to 3 robust SD (`ms1cal::PREC_TOL_AUTO_K`)
  of the anchors' residuals after the calibration in use (the surface's, or the one
  offset's), the SD being 1.4826 MAD refined once over the residuals within four of them,
  clamped to 1-15 ppm. It needs `ms1_calibrate` on; with too few anchors the configured
  tolerance stays. On an Orbitrap the rule lands near the 3 ppm default: on the Q Exactive
  AIF E. coli entrapment run (5,303 anchors) the residuals' robust SD was 0.93 ppm after the
  one offset and 0.80 ppm after the surface, and the tolerance 2.6 and 2.2 ppm.

Either key records `ms1_calibration` (`ExtractConfig::ms1_cal`: mode, the surface, the
anchors, the residual median and robust SD as read, after the offset and after the
surface, and the tolerance), `ms1_ppm_offset` and `prec_tol_ppm` in both extract reports;
a report without them is unchanged, and so is every read without them (`ms1_shift` is the
expression every read used before). The standalone `mumdia extract` does not calibrate
MS1, under either key, as it never did.

**ZenoTOF 8600, ZT Scan REP01** (24,124 of 25,305 confident seed anchors with an MS1 peak
within 15 ppm; seed decoys 62% of the time, at a robust SD of 11.2 ppm, so a random peak in
the window is common):

| | median, ppm | robust SD, ppm | anchors within 3 ppm of the median |
|---|---|---|---|
| as read | +2.84 | 2.67 | 66% (decoys 19%) |
| after the one offset | 0 | 2.67 | |
| after the surface | 0 | 2.61 | |

The error drifts: by retention time from +3.4 ppm at the start of the gradient to +2.3 ppm
at its end (the surface's retention-time term spans -1.27 to +1.12 ppm over the anchors'
range), by precursor m/z from +3.4 ppm at 400 m/z to +2.6 ppm at 900 m/z (m/z term 2.79 to
3.71 ppm). The surface takes the octile medians of the residuals from +0.56/-0.50 ppm (m/z
/ retention time, extremes) to within 0.16 ppm, but the spread is mostly not drift: it
depends on intensity (robust SD 3.5 ppm for the weakest sixth of the anchors, 1.6 ppm for
the strongest) and stays at 2.6 ppm. So 3 ppm is too narrow for this instrument: it holds
about two thirds of the anchors' MS1 peaks, where the same window on an Orbitrap holds
nearly all. `prec_tol_auto` sets 7.17 ppm after the surface (robust SD 2.39 ppm after the
refinement) and 7.38 ppm after the one offset. What that does to identifications is in the
table of docs/04_convert.md, "SCIEX ZT Scan DIA".

### 12. Per-fragment Q1 decisions (scanning quadrupole, automatic for ZT Scan files)

On a ZT Scan DIA run every Q1 step is an isolation window of its own (section 10), so a
fragment m/z that a candidate's library predicts is seen in the run of steps around the
candidate's precursor, and so is the same m/z of any other precursor within the window's
reach. Where in the sweep a fragment's signal is centred says which precursor it came from.
`extract.q1_fragments` (`stages::q1frag`, psms_extracted v6) decides that for every library
fragment of every accepted row, after the gate, from the spectra directly. It needs the
window that `rt_im_train.q1_calibrate` fits (docs/08_rt_im_train.md, section 4h) and the
decision tolerances the same calibration learns from the confident seed PSMs under this key,
which `run` and `run-experiment` read from the run's `cal.json` (`mumdia extract` takes it as
`--q1-cal`), and the sweep spectra_ms2 v3 records (`q1_cycle`, `q1_step`, docs/04): a
cycle x step grid of the stage's scans (`Q1Grid`; a recorded step that is not the rank of
its target, or a step twice in a cycle, is refused). Refused with `extract.q1_cycles`, which
sums the steps this keeps apart, on spectra with 1/K0, and, through `q1_calibrate`, with
`groups.window_groups > 1`.

**The patch.** For a row with its apex at a scan of cycle `c0`: the cycles `c0 - 3 ..= c0 + 3`
whose scan at the patch's middle step lies inside the candidate's RT window (the apex cycle
always), and the steps whose target lies within the window's reach (`h + e/2`, 4.56 m/z on
ZT Scan) plus two steps of the precursor's expected Q1 centre (the isotope envelope's centre
minus the fitted window offset): 17 steps on ZT Scan, 119 cells a fragment. Each cell is the
intensity of the nearest peak within the fragment tolerance of the mass-corrected library
fragment m/z in that scan, the rule of the seed and of the calibration. Every step stays a
cell of its own: the steps of a cycle are never summed (summing them lost 13.7% of the
identifications in section 10).

**The decision** (`q1frag::decide`). The expected profile of a fragment is the calibrated
trapezoid summed over the precursor's isotopes, weighted by abundance times the chance the
fragment stays monoisotopic (its centre therefore depends on the fragment). Then, in two
rounds:

- *Present*: at least three coherent cells within the window's reach of the fragment's
  expected centre, cells with signal whose neighbouring step in the same cycle has signal too.
  The spectra are dense (about 1,550 peaks a spectrum, so a 20 ppm window at 600 m/z holds a
  peak in roughly 2% of the cells by chance), so a fragment of a precursor that is not there
  still has a few isolated matches in its 119 cells, and a count of cells with signal would
  call nearly every fragment present; a precursor is transmitted through several consecutive
  steps. On REP01 the coherence rule calls 2.7% of the extracted rows' fragments absent
  (0.1% of the confident seed anchors'), a count of two cells with signal 1.6%.
- *2D apex*: the cycle where the round's fragments, each normalised to its own total over the
  patch, sum highest. The Q1 profile of a fragment is its patch summed over the five cycles
  around it, every step apart.
- *Shared centre*: one centre (a shift of the expected profile, in steps, on a 0.25-step grid
  from -6 to +6 refined by a parabola) and, in round 2, one width (0.85, 1 or 1.2 times the
  calibrated `h` and `e`) shared by the round's fragments with a Q1 profile, by least squares
  of their normalised profiles against the expected ones with a scale (the fragment's height)
  and a floor each.
- *Per fragment*: its own centre (the same fit, alone, at the shared width), its residual
  (`1 - R^2` of its profile under the shared fit) and its 2D correlation: the Pearson
  correlation over the 119 cells of its patch with the elution of the round's other fragments
  times its own expected transmission at the shared centre.
- *Class*: `belongs` when its own centre lies within the offset tolerance of the shared
  centre, the shared centre clamped to within the centre tolerance of the expected one (so a
  candidate whose fragments all sit at another precursor has none that belong), and its 2D
  correlation reaches the floor; `interfered` when present otherwise; `absent` when not
  present.

Round 1 fits over every present fragment; round 2 repeats the fit, the 2D apex and the
correlations over the fragments round 1 says belong (when at least two do), which is the
consensus iterated once. Deterministic: every sum runs in fragment, cycle and step order,
and ties keep the first.

**The tolerances** are learned per run (docs/08 section 4h): from the confident seed
anchors' own patches, the median plus three robust SD of their fragments' offsets from the
shared centre (offset tolerance) and of the shared centres from the expected one (centre
tolerance), and the 5th percentile of the 2D correlation of the fragments within the offset
tolerance (the floor).

**What it writes** (psms_extracted v6, appended last; the report's `params.q1_fragments`
records the window, the tolerances and the patch shape, its `stats.q1_fragments` the
fragments of each class):

| column | type | meaning |
|---|---|---|
| `q1f_n_present`, `q1f_n_belong`, `q1f_n_interfered` | F64 | fragments of each kind |
| `q1f_belong_share` | F64 | predicted-intensity share of the fragments that belong |
| `q1f_interfered_share` | F64 | observed-intensity share of the interfered fragments among the present ones |
| `q1f_offset_median`, `q1f_offset_mad` | F64 | median and MAD of the fragments' own Q1 centres against their expected ones (steps) |
| `q1f_corr2d` | F64 | mean 2D correlation of the fragments that belong |
| `q1f_fit_resid` | F64 | mean `1 - R^2` of the shared fit over its fragments; 1 without a fit |
| `q1f_centre`, `q1f_width` | F64 | the shared centre (steps against the expected one) and width scale |
| `q1f_apex_shift_cycles` | F64 | the 2D apex cycle minus the apex scan's cycle |
| `q1f_apex_moved_s` | F64 | only under `retrace.q1_repick`: how far the apex moved (s) |
| `q1f_frag_class` | List F32 | per library fragment, in the library's order: 0 absent, 1 belongs, 2 interfered |
| `q1f_frag_offset`, `q1f_frag_resid`, `q1f_frag_corr2d` | List F32 | per library fragment: its own centre (steps), residual and 2D correlation; NaN where there is none |

The per-fragment lists sit in `psms_extracted` itself rather than in a sidecar table: a
decision belongs to one row (one apex, one `peak_rank`), every consumer joins rows of that
table on `(candidate_id, peak_rank)` already (the shadow test's 1/K0 term does), and a table
of its own would be one more file each orchestrator, pool and retention tier has to carry.
Readers select columns by name, so every other stage reads v6 as v2, and `features` reads the
summary under `features.q1_fragment_features` (docs/10_features.md).

**`retrace.q1_repick`** (automatic for ZT Scan files; needs `q1_fragments`) is the scanning-quadrupole
analogue of `retrace.repick` (section 9). ZT Scan has no raw events beyond the centroids
extract reads, so it runs inside extract and needs neither `retrace.enabled` nor a `.d`. Its
candidates are extract's apex and the next peaks of the count profile whose envelope does not
hold it (`enumerate_peaks`, up to `retrace.q1_repick_peaks`, default five, as
`retain_top_peaks` enumerates them; 0 compares none and only refines the apex to its own
peak's 2D apex, docs/37_ztscan.md, section 8, has what that saves and costs). Each is
decided as above and scored like `retrace.repick`'s joint score: the cosine of the
square-rooted intensities of the fragments that belong (summed over the five profile cycles,
every other fragment 0) against the square-rooted predictions, times `ln(1 + I)`, `I` their
prediction-weighted mean intensity. The apex moves to the best peak's 2D apex (its cycle, at
the step nearest the fitted Q1 centre whose window as read covers the precursor, a scan of the
candidate's own grid) when at least three fragments belong there; the apex intensity, the MS1
values and the decision are read again at the new apex. Refused with
`extract.promote_top_peaks > 1` (it moves the rank-0 apex), `extract.apex_refine_intensity`
and `retrace.repick` (each re-chooses the apex).

`rescore.shadow_q1` reads the lists (docs/11_compete_rescore_fdr.md, "shadow demotion and its
Q1 term").

**Measured** on the ZenoTOF 8600 K562 REP01 run (the original mzML read with
`isolation_half_width = 1.9`, imported library with predicted decoys, the shipped `nn_torch`
recipe, 64 threads; seeds 1-3 rescore the run's competed table with `MUMDIA_NN_SEED`;
precursors at `precursor_q` 1%; the human-library arms on one host, the entrapment arms
(human + yeast + E. coli library, r = 0.397, combined FDP = N_E (1 + 1/r) / (N_H + N_E)) on
another, every arm of a library with one binary on one host). Every arm sets
`rt_im_train.q1_calibrate`, which alone changes no table (section 10). "Matched FDP" is the
human precursors where the arm's combined FDP reaches the base arm's mean at 1% (1.386%),
interpolated along the `precursor_q` thresholds.

| arm | precursors, mean of seeds 0-3 (range) | vs base | entrapment: human precursors at 1%, mean | combined FDP at 1%, mean (range) | human precursors at matched FDP, mean (range) | vs base |
|---|---|---|---|---|---|---|
| base (no key) | 97,317 (96,873-97,541) | | 95,115 | 1.39% (1.35-1.44%) | 95,135 (94,877-95,342) | |
| `q1_fragments` + `features.q1_fragment_features` | 98,728 (98,352-99,086) | +1.45% | 95,829 | 1.31% (1.28-1.36%) | 96,337 (95,885-96,496) | +1.26% |
| the same + `retrace.q1_repick` | 100,194 (100,067-100,329) | +2.96% | 97,254 | 1.29% (1.24-1.31%) | 97,915 (97,705-98,222) | +2.92% |
| the same + `rescore.shadow_q1` | 98,705 (98,356-99,046) | +1.43% | 95,996 | 1.29% (1.25-1.33%) | 96,624 (96,037-96,946) | +1.57% |
| all four | 100,257 (100,132-100,520) | +3.02% | 97,272 | 1.22% (1.20-1.24%) | 98,155 (97,933-98,392) | +3.17% |

- **The decision and its features** (+1.3 to +1.5%): the accepted targets of the
  `q1_fragments` arm have 5.35 of their 12 library fragments that belong on average (75.6%
  at least three), 6.5 interfered and 0.2 absent; the decoys 0.09 that belong (0.8% at least
  three). In the learning pass 79% of the confident seed anchors' fragments belong. Learned
  on REP01: offset tolerance 1.15 steps, centre tolerance 0.69 steps, 2D-correlation floor
  0.556 (the entrapment run: 1.10, 0.68, 0.578).
- **The re-pick** is the largest single lever (+1.5 pp over the decision alone, at a lower
  FDP): it moved 128,123 apexes (3.1% of the rows), 60,164 of them of targets accepted at 1%
  (60% of them), by a median of exactly one cycle (1.0 s); 1,518 accepted rows moved more than
  5 s, to another peak. So extract's count apex (the rolling count over the interleaved
  steps) sits a cycle off the cycle where the fragments that belong peak, on most of the
  identified precursors.
- **The shadow test's Q1 term** changes the count by nothing measurable (-0.02% and +0.06%
  precursors against the arms without it; +0.3 and +0.25 pp at matched FDP, inside the seed
  ranges). It demotes five times as many rows of the first pass (14,880 targets and 14,168
  decoys on REP01 under all four keys, against 2,644 and 2,448 for the window test), label
  for label, and lowers the entrapment precursors at 1% (339 against 357 with the re-pick
  alone), so the reported 1% moves closer to the measured FDP.
- **The reported 1%**: 1.39% measured FDP for the base, 1.22% with all four keys.
- **Three runs as one experiment** (REP01-03, second pass on, one seed, one host): base
  110,433 precursors, 88,958 peptides, 8,981 protein groups (per run on `run_psm_q`
  106,485 / 106,561 / 106,583; first pass 98,441 / 98,435 / 98,265); all four keys 113,013
  (+2.3%), 90,739 (+2.0%), 9,027 (+0.5%) (per run 109,028 / 109,185 / 109,093, +2.4%; first
  pass 101,104 / 101,212 / 100,449, +2.6%).
- **Cost** (shared hosts; walls indicative): extract 341 s on the human-library host
  against 267-305 s for two base runs with the decision, 463-470 s with the re-pick, which
  decides up to six times per candidate; on the entrapment host 548-551 s against 528-538 s,
  and 758-766 s (where that time goes, and what bounding the alternatives saves:
  docs/37_ztscan.md, section 8). `q1_calibrate` with the learning pass took 66 s (the
  learning itself 4 s);
  the peak memory stays at 35-41 GB (base 37 GB). In the three-run experiment, whose three
  extracts ran at the same time on a loaded host, extract took 1,372-1,447 s per run against
  390-545 s, and the whole run 3,941 s against 2,401 s.

**Label blindness.** Every decision reads the spectra and the library only. Among the rows
whose untrained `prelim_score` is below the decoys' median (where targets are
overwhelmingly false), decoys and targets carry the same distributions: the two-sample
Kolmogorov-Smirnov statistic of each `q1f_*` feature is 0.004-0.005 (all four keys, REP01;
1.03 M decoys against 0.99 M targets), as for features the vector already has (`frag_corr`
0.004, `coelution_mean` 0.003, `rt_error_abs` 0.006), while the accepted targets differ
(`q1f_n_belong` median 5 against 0, `q1f_corr2d` 0.82 against 0, `q1f_fit_resid` 0.11
against 0.77).

All four keys stay off and benchmark-gated: one acquisition, and CLAUDE.md asks for a second
one (docs/04_convert.md, "SCIEX ZT Scan DIA") before any default changes.

The decisions are also read by quantification: `quant.q1_fragments` collapses every trace
to one value per cycle under the calibrated transmission and leaves the fragments that do not
belong out of the quantity (docs/12, "Quantification from the fragments that belong").

### 13. MS1 isotope-envelope evidence (`features.ms1_envelope`, default off)

Under `features.ms1_envelope` (`ExtractConfig::ms1_envelope`, which `run`, `run-experiment`
and `mumdia extract` set from that key, never read from a configuration file), every row,
rank 0 and promoted alike, reads the precursor's MS1 envelope at its own apex
(`stages/ms1env.rs`, `read`): in the MS1 scan nearest the apex and two either side, the peak
nearest each of M-1, M0 .. M+4 and the half-spacing positions above M0 and M+1, at the m/z the
MS1 calibration reads each scan at (`ExtractConfig::ms1_shift` at the scan's retention time:
the surface under `extract.ms1_calibrate = "surface"`, else the one offset), within twice
`extract.prec_tol_ppm` and at least 10 ppm (`ms1env::search_tol`). It writes psms_extracted
v7, the table of v2, v5 or v6 plus, appended last (F64):

| column | meaning |
|---|---|
| `ms1e_i_m1`, `ms1e_i_0` .. `ms1e_i_4` | summed intensity over the scans of the nearest peak of M-1, M0 .. M+4 |
| `ms1e_ppm_m1`, `ms1e_ppm_0` .. `ms1e_ppm_4` | their intensity-weighted ppm error against the calibrated m/z; NaN where none was found |
| `ms1e_i_half` | summed intensity at the half-spacing positions above M0 and M+1 |
| `ms1e_scans` | MS1 scans read (5, fewer at the run's ends) |
| `ms1e_tol_ppm` | the search tolerance |
| `ms1e_q1_agree` | under `extract.q1_fragments`, for a row whose decision has a fit: the fitted Q1 centre (`q1f_centre`, steps) less the centre the observed envelope predicts, `mu + sp (ibar_averagine - ibar_observed) / step`; NaN otherwise |

The features stage derives the nine `ms1e_*` features from it (docs/10_features.md,
"ms1_envelope"). Refused on spectra that carry 1/K0 (extract) and with
`groups.window_groups > 1` (validation). The report records `params.ms1_envelope` (tolerance,
scans, isotopes). Off, nothing is read and the table, its report and every later artifact
keep their bytes.

### 14. The unfragmented precursor, multi-scale Q1 scores and Q1-deconvolved evidence (scanning quadrupole)

Three additions read the sweep further. Each has its own key, omitted from the canonical
configuration unless it is on; with all of them off nothing below is computed and every
artifact keeps its bytes. The three feature families are automatic for ZT Scan files (docs/37_ztscan.md,
"Automatic for ZT Scan files"); the hard counts (`extract.q1_deconvolved`,
`search_seed.q1_deconvolved`) stay opt-in. What they measured is in docs/37_ztscan.md, section
10.

**The unfragmented precursor** (`features.q1_precursor`, `stages/q1score.rs`). Part of a
precursor survives the collision cell, and in a ZT Scan MS2 spectrum it shows up at its own m/z
in the steps whose window transmits it. Each isotope (M0, M+1, M+2) is one ion, so its Q1
profile is the calibrated window centred on that isotope's own m/z, not the envelope a fragment
inherits. Under the key (`ExtractConfig::q1_precursor`, set by `run`, `run-experiment` and
`mumdia extract` from it; it needs `extract.q1_fragments`) every row's precursor isotopes are
read as fragments of their own (`CandQ1::precursor`: each at `pm + k sp` times the mass-offset
factor, its expected centre `pm + k sp - delta`, its weight the Poisson-averagine abundance) on
the cells of the decision's patch (section 12), and `q1score::precursor_scores` writes:

| column | meaning |
|---|---|
| `q1p_n_iso` | isotopes present by the decision's coherent-cell rule (0 to 3) |
| `q1p_share` | the precursor's share of the precursor plus all library fragment intensity at the 2D apex (the five profile cycles, every patch step) |
| `q1p_offset` | one shift of the present isotopes' Q1 profiles from where the calibrated window puts them, steps, fitted on the decision's centre grid at the calibrated width (0 without a fit) |
| `q1p_resid` | their mean `1 - R^2` under that shift (1 without a fit) |
| `q1p_frag_corr_mean`, `q1p_frag_corr_max` | mean and largest Pearson correlation of the present fragments' Q1 profiles with the precursor's (its isotopes summed); 0 without precursor signal |
| `q1p_n_belong` | fragments whose own Q1 centre lies within the decision's offset tolerance of the precursor's shift and whose 2D correlation reaches the floor: the belonging count referenced to the precursor instead of the fragments' shared centre (0 without a fit) |
| `q1p_centre_agree` | the distance between the precursor's shift and the fragments' shared centre, steps (6 without both fits) |

**Multi-scale Q1 scores** (`features.q1_multiscale`, `q1score::multiscale`, needs
`extract.q1_fragments`). Over the 3, 7 and 11 steps centred on the step nearest the fragments'
fitted Q1 centre (`q1score::SCALES`), with the fragments' Q1 profiles of the decision at its 2D
apex, for each scale `K`: `q1mK_corr_best`, the mean correlation of every present fragment's
profile with the most intense fragment's (over the window); `q1mK_shape`, the `1 - R^2` (with a
scale and a floor) of the present fragments' consensus profile, each normalised to its own
total, against their mean expected transmission at the fitted centre and width;
`q1mK_centroid`, the most intense fragment's intensity-weighted Q1 centroid against its expected
centre, steps; and with `features.q1_precursor` on as well `q1mK_corr_prec`, the mean
correlation of the fragments' profiles with the precursor's. The narrow scale reads the
transmission's flat top, the wide one its edges; on three steps the centroid is the middle
step's target to within half a step. None duplicates a `q1f_*` value: those are fits of every
fragment over the whole patch, these are model-free statistics of one window.

**Q1-deconvolved evidence** (`stages/q1dec.rs`). Every MS2 peak is given a Q1 position of its
own, before any candidate is considered: within one cycle, a peak of step `s` and a peak of step
`s + 1` are linked when each is the other's nearest within the fragment tolerance (mutual
nearest neighbours, so a trace never branches), a chain is split after each local minimum
below half the smaller of the maxima either side (`q1dec::VALLEY`), and every peak of a segment
gets the segment's intensity-weighted mean step target; a peak linked to nothing is at its own
step. `Q1Dec::build` does that for the whole run in parallel over cycles at the stage's load
(about 6 s and one signed byte a peak, 3.3 GB for the 3.34 billion peaks of a K562 run, held
beside the scans and never written, so no spectra artifact changes). A peak is consistent with
a fragment of a candidate when its centroid lies within the tolerance of the fragment's
expected Q1 centre (`q1dec::fragment_centre`, the decision's: the precursor isotopes that keep
the fragment monoisotopic weigh more, so a large fragment's centre lies up to about a step below
the envelope's), stored per library fragment as its shift from the candidate's centre in
1/16 step. The tolerance is learned by the Q1 calibration (docs/08_rt_im_train.md, section 4h,
`deconvolution`) and handed to extract with the model (`ExtractConfig::q1_dec_tol_steps`;
`mumdia extract --q1-cal`). Two consumers:

- `extract.q1_deconvolved` counts only the consistent peaks: every probe site asks
  `Guard::q1_admits` beside the RT window, so the hits, the presence rules, the apex, the gate
  and the traces see only them (`search_seed.q1_deconvolved` does the same in the seed against
  the precursor's nominal envelope centre within `q1dec::SEED_TOL_STEPS`, docs/07). Needs
  `rt_im_train.q1_calibrate`; refused with `extract.q1_cycles`, with window groups and on
  spectra with 1/K0.
- `features.q1_deconvolved` leaves the extraction as it is and reads, at each row's apex scan,
  for every library fragment the nearest peak in every scan of the apex cycle and the two either
  side whose window covers the precursor (`q1dec::row_evidence`): `q1d_n_frag` (fragments with
  a consistent match), `q1d_n_incons` (fragments matched only inconsistently), `q1d_share`
  (the consistent share of the matched intensity), `q1d_pred_share` (the predicted-intensity
  share of the fragments with a consistent match) and `q1d_apex_share` (the consistent share in
  the apex cycle alone). Needs `extract.q1_fragments`.

All three families write psms_extracted v8 (`artifact::psms_extracted_q1s`): the table of v6
or v7 plus, appended last, the `q1p_*`, then the `q1m*`, then the `q1d_*` columns (F64), each
only under its key; the features stage reads them by name (docs/10_features.md). The extract
report records `params.q1_fragments.q1_scores` (which families, the isotopes, the scales) and
`params.q1_deconvolved` (gate or features, the tolerance, the step, the valley share, the
cycles read), each only when on. Every value is read from the spectra and the library; nothing
reads a label (docs/37, section 10, has the decoy against low-target comparison).

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `run` | `extract.rs:1300` | Stage entry point; orchestrates load, accumulate, cascade, write |
| `ExtractParams` | `extract.rs:83` | Input path bundle + config + config hash |
| `Hit` / `hit_rt` | `extract.rs` | One observed hit (16 bytes): `scan`, `frag`, `inten`, `obs_mz`; `hit_rt` looks its RT up in `scans` |
| `Contested` | `extract.rs:130` | Per-candidate two-pass contested-peak stats: `won`/`lost` intensity, `n_won`/`n_lost` peak counts, `apportioned` share |
| `CandOut` | `extract.rs:1656` | Per-candidate parallel-map result (PSM row + chrom rows + peaks); one candidate may yield several |
| `Prober` / `Prober::probe` | `extract.rs:38`, `extract.rs:57` | Dispatch a peak probe to fragindex or bucketed backend |
| `MassOffset::factor_at` / `read_mass_cal` | `masscal.rs` | Scalar or m/z-interpolated ppm divisor for observed m/z; the one reader of `masscal.json`, shared with retrace |
| `ms1_neighbours` | `extract.rs` | The MS1 scans an MS1 read takes the maximum over (`ms1_scan_halfwidth`); shared with retrace |
| `claim_cue_multiplier` | `extract.rs:163` | Composite per-claimant cue weight for `CoelutionMultiCue` |
| `demix_apex_scan` / `demix_solve_scan` | `extract.rs:259`, `extract.rs:279` | Spectrum-centric ridge-NNLS demix at one scan |
| `demix_features_for` | `extract.rs:414` | Per-candidate demix features from a solved scan |
| `extract_accumulate_windows` | `extract.rs:662` | Parallel single-pass per-window accumulation |
| `extract_twopass_windows` | `extract.rs:787` | Parallel two-pass co-elution arbitration + contested stats |
| `peak_window` | `extract.rs:523` | Elution-peak scan range around the signature apex (10% height) |
| `peak_spectral_score` | `extract.rs:564` | Peak-integrated observed-vs-predicted Pearson |
| `coelution_gate_score` | `extract.rs:590` | Weighted mean XIC-to-reference co-elution Pearson |
| `nearest_index` | `extract.rs:139` | Binary search for nearest RT in a sorted array |
| `sum_near` | `extract.rs:495` | Sum intensities within a ppm window (m/z-sorted) |
| `enumerate_peaks` | `peaks.rs:52` | Pure top-K peak-group enumerator |
| `PeakGroup` | `peaks.rs:22` | One enumerated peak (apex/start/end idx, apex intensity, area, rank) |
| `Library::load_with` | `index.rs:73` | Load the SoA library, optionally skipping the bucketed index |
| `Library::candidate_range` | `index.rs:341` | Candidate-id range for an isolation window |
| `Library::cand_frags` | `index.rs:312` | Fragment m/z, predicted intensity, and interned name-id slices |
| `Library::page_search` | `index.rs:350` | Bucketed-backend fragment probe |
| `FragIndex::probe_peak` | `fragindex.rs:169` | Verified postings for one observed peak in a candidate range |
| `FragIndex::probe_peak_win_binned` / `bin_of` | `fragindex.rs:225` / `:197` | The same probe with the bin handed in, which is how the per-scan setup scratch probes |

## Configuration

All fields live in `ExtractConfig` (`config.rs:516`); defaults in
`config.rs:683`. The config was pruned of dead fields in the "clean-slate
declutter" commit: `extract.scan_window_mode` (and the `ScanWindowMode` enum),
`extract.scan_scale`, `extract.k_select`, `extract.max_fragment_charge`, and
`extract.tolerance_regime` were removed. Do not reintroduce them.

| Field | Default | Effect |
|---|---|---|
| `max_footprint_gib` | unset (`auto`) | Refuse an extraction whose estimated hit store passes this many GiB; `auto` is the machine's memory, `0` is off. See "The footprint guard" |
| `fixed_scan_window` | 2 (3 before 2026-10-08) | Minimum co-elution run length (`scan_window` floor); `.max(1)` at `extract.rs:1591` |
| `frag_tol_ppm` | 20.0 | Fragment match tolerance (overridden by `mass_cal`) |
| `prec_tol_ppm` | 3.0 (20.0 before 2026-10-08) | MS1 isotope integration tolerance, centred on the offset `ms1_calibrate` learns from the seed |
| `ms1_calibrate` | `true` | `true`: one MS1 offset learned from the seed; `"surface"`: an offset per m/z and retention time; `false`: none (section 11) |
| `prec_tol_auto` | false | Set `prec_tol_ppm` to three robust SD of the calibrated MS1 residuals (section 11) |
| `presence_min_matched` | 3 | Tier-b: minimum distinct matched fragments (`extract.rs:1727`) |
| `presence_min_fragments` | 3 | Acceptance: minimum distinct fragments (`extract.rs:1926`) |
| `presence_min_coelution` | 2 | Min simultaneously-present fragments to extend a run (`extract.rs:1915`) |
| `gate_min_score` | **0.6** | Threshold of the active `gate_mode`; 0 disables the gate. Default since 2026-10-08 with `gate_mode = window_mean` (window-summed agreement minus an RT term; see CLAUDE.md, "Extraction gate and retention time"). Under the previous `apex_pearson` default it was 0.2, measured better than an apex-Pearson 0.6 (docs/18). Renamed from `min_frag_corr` |
| `min_matched_fraction` | 0.0 | Acceptance: min matched/predicted fraction (default off) |
| `apex_top_fragments` | 0 | Signature-ion count for apex; 0 -> default 3 (`extract.rs:1860`). Config marks it superseded by `apex_count_tol`, kept for compat (`config.rs:545`) |
| `apex_rt_prior_s` | 0.0 | Gaussian RT-prior sigma on apex tiebreak; 0 = off |
| `apex_count_tol` | 1 | Count slack for qualifying apex scans |
| `apex_count_window` | `"auto"` | Rolling-sum width for the count profile; 1 = no smoothing. `"auto"`: 1 on an Orbitrap Astral, 5 elsewhere. Window 5 cut AIF apex misassignment (median \|dRT\| 131s -> 9s) |
| `apex_gaussian_sigma_scans` | 0.0 | Opt-in Gaussian apex smoother; 0 = off (`config.rs:572`) |
| `apex_evidence_rank` | **true** | Breadth-of-evidence apex. The legacy signature-intensity apex silently falls back to the lowest-RT qualifying scan when none of the top-K predicted fragments is observed |
| `emit_window_grid` | true | Zero-filled full-window-grid chromatograms |
| `bucket_size` | 8192 | m/z bucket size (power of two) |
| `peak_claim` | `None` | Shared-peak apportionment strategy (`PeakClaim`, `config.rs:132`) |
| `claim_cues` | all off | Composable cue weights for `CoelutionMultiCue` (`ClaimCues`, `config.rs:192`); `claim_cues.reassign` is what makes MultiCue destructive |
| `peak_claim_margin` | 2.0 | Dominance factor for `CoelutionWinnerMargin` |
| `emit_demix_features` | false | Adds the five `deconv_*`/`shadow_*` columns; forces the apex demix solve (`config.rs:592`) |
| `demix_lambda` | 1.0 | Ridge term of the NNLS demix (`config.rs:595`) |
| `demix_max_candidates` | 64 | Column cap of the per-scan demix design matrix (`config.rs:598`) |
| `demix_scan_stride` | 1 | Solve the demix every Nth scan (`config.rs:606`) |
| `emit_contested_features` | false | Adds `contested_count_frac`/`apportioned_frac`; forces the two-pass path |
| `matcher` | `Fragindex` | Fragment-matcher backend |
| `min_coelution_run` | 0 | Extra co-elution-run floor (0 = off; `scan_window` still applies) |
| `ms1_rescue` | false | Rescue Pearson-gate failures with MS1 isotope support |
| `retain_top_peaks` | 1 | K>1 writes the `.peaks.parquet` sidecar (unscored) |
| `promote_top_peaks` | 1 | K>1 emits extra `psms_extracted` rows with `peak_rank >= 1`; **does** change FDR (`config.rs:645`) |
| `alt_peak_min_area_frac` | 0.10 | Alternate peak must reach 10% of the rank-0 area (`config.rs:649`) |
| `alt_peak_min_separation_s` | 5.0 | Alternate apex must sit >= 5 s from the rank-0 apex (`config.rs:653`) |
| `emit_candidate_audit` | false | Candidate-audit sidecar (diagnostic) |
| `emit_gate_diagnostics` | false | Adds the four `gate_*` diagnostic columns |
| `gate_mode` | `ApexPearson` | Which score `gate_min_score` thresholds (`GateMode`, `config.rs:735`) |
| `gate_coelution_min` | 0.5 | Second threshold for `GateMode::Combined` |
| `q1_cycles`, `q1_profile`, `q1_gate_max_offset` | off | The scanning-quadrupole sweep model (section 10) |
| `q1_fragments` | false | The per-fragment Q1 decision of every accepted row (psms_extracted v6); needs `rt_im_train.q1_calibrate` and spectra_ms2 v3 (section 12) |
| `q1_deconvolved` | false | Count only the peaks whose Q1 centroid sits where the matched fragment is expected, within the tolerance the Q1 calibration learns; needs `rt_im_train.q1_calibrate` and spectra_ms2 v3 (section 14) |
| `im_gate` | `off` | The 1/K0 gate on fragment peaks (`off`, `fragments`, `auto`; section 8). Inert on 3D data. Omitted from the canonical configuration at its default |

Note: `emit_candidate_audit` is a declared knob but the candidate-audit write is
not present in this `extract.rs`; the audit ladder is produced by the separate
`mumdia audit` command. Treat the in-stage audit as unwired here.

### Default-off sensitivity knobs (index)

Every knob below is default-off, so the production PSM schema and per-candidate
compute stay byte-identical unless the knob is set. None has an end-to-end
identification-gain measurement yet; each must pass the entrapment-holdout gate
on >=2 datasets before being enabled by default (see the benchmark-gated list in
CLAUDE.md and docs/20_sensitivity_and_quantification_playbook.md). Two of the
knobs live in other stages' configs (`search_seed`, `rt_im_train`) but are listed
here for one index.

| knob | extra columns / artifact | validation status |
|---|---|---|
| `extract.retain_top_peaks` (`config.rs:635`, default 1) | `<out_psms>.peaks.parquet` sidecar, unscored, written only when K>1 (`extract.rs:2544`); no PSM columns | ID loop not closed (sidecar peaks are unscored); gate pending |
| `extract.promote_top_peaks` (`config.rs:645`, default 1) | extra `psms_extracted` rows with `peak_rank >= 1` (`extract.rs:2297`); no new columns | **not schema-neutral in rows**: changes the scored population and therefore FDR; on diaPASEF (IM branch, seed 0) K = 2 lost 5.7% and K = 3 5.9%, and `retrace.repick` is the diaPASEF path (docs/35 section 12); gate pending |
| `extract.emit_candidate_audit` (`config.rs:658`) | none in `extract.rs` (unused there); in `run` it gates the `audit` stage -> `candidate_audit.parquet` (`run.rs:428`) | diagnostic; no ID effect |
| `extract.emit_gate_diagnostics` (`config.rs:673`) | 4 F32 cols `gate_apex`, `gate_peak_spectral`, `gate_coelution`, `gate_spectral_entropy` (`extract.rs:2511`) | diagnostic; no ID effect |
| `extract.apex_evidence_rank` (`config.rs:667`) | none; changes the apex-selection score (`extract.rs:1890`) | **now the default**, promoted on correctness grounds rather than a count (docs/25 section 2): the legacy score cannot distinguish "no signature fragment observed anywhere" from "the earliest qualifying scan is the apex" |
| `extract.apex_gaussian_sigma_scans` (`config.rs:572`) | none; smooths the apex count profile | not measured; gate pending |
| `search_seed.two_pass_mass_cal` | none; refits the `<seed>.masscal.json` offset + tolerance | **default since 2026-10-11** (three NN seeds): Astral HYE +1.8%, ProteoBench Astral entrapment +1.1%, Astral HeLa entrapment +2.0% at a measured FDP of 1%, AIF HYE +0.5%; costs an Astral single cell 2.9% and raises the measured FDP at a reported 1% slightly (AIF entrapment 0.99% -> 1.02%, ProteoBench paired 0.86% -> 0.88%) |
| `rt_im_train.adaptive_rt_window` (`config.rs:487`) | none; per-region RT half-window widths in `run_windows` | not measured; gate pending |
| `extract.emit_contested_features` (`config.rs:611`) | 2 F64 cols `contested_count_frac`, `apportioned_frac` (`extract.rs:2505`); forces the two-pass path (`extract.rs:1443`) | not measured; gate pending |
| `extract.emit_demix_features` (`config.rs:592`) | 5 F32 cols `deconv_explained_frac`, `deconv_active`, `deconv_share`, `deconv_max_collinearity`, `shadow_kept_frac` (`extract.rs:2517`) | not measured; gate pending |
| `extract.peak_claim` = `CoelutionMultiCue` / `CoelutionDemix` / `CoelutionShadow` (`config.rs:132`) | none; MultiCue is non-destructive unless `claim_cues.reassign`, the other two always rewrite extracted intensity | not measured; gate pending |
| `retrace.enabled` (default false, diaPASEF only) | none; rewrites the chromatogram values from the raw `.d` and patches the `psms_extracted` apex scalars (section 9) | IM branch: +5.9% peptides E. coli (3 seeds), entrapment at an unchanged FDP, HYE +15.0% (seed 0); main: +5.6% E. coli (one seed, docs/35 section 11); gate pending on main (docs/35 section 6) |
| `retrace.repick` (default false) | none; patches `apex_rt` / `apex_im` | IM branch: +7.4% peptides E. coli (3 seeds), entrapment +5.4% real peptides at an FDP of 0.37-0.47%, HYE +8.4% (seed 0); main: +8.0% E. coli (one seed); gate pending on main |
| `extract.im_gate` (default `off`, diaPASEF only) | none; fragment peaks outside the 1/K0 window are not matched (section 8) | IM branch: +5.7% E. coli (3 seeds); main, with `features.im_features`: +5.0% (one seed); gate: paired entrapment and HYE seeds (docs/35 arm T12) |
| `retrace.apex_features` (default false) | 1 F64 col `imc_ref_w` in `psms_extracted`; the tenth ion-mobility feature | IM branch: +1.3% alone, about 2.7% FDP on the added peptides; gate pending |
| `extract.q1_cycles` (default false, scanning quadrupole only) | none; a candidate's Q1 bins of one cycle become one chromatogram point (section 10) | ZT Scan REP01: -13.7% precursors (one seed); keep off |
| `extract.q1_profile` + `features.q1_features` (default false) | psms_extracted v5: 4 F64 cols `q1_corr`, `q1_offset`, `q1_frag_sd`, `q1_bin_frac`; four features | ZT Scan REP01: +0.4% precursors (mean of four NN seeds, inside the seed range), +0.4% in a three-run experiment, +0.1% human precursors on an entrapment library at a combined FDP of 1.32% against 1.39%; one acquisition |
| `extract.q1_gate_max_offset` (unset) | none; rejects on `q1_offset` | not useful at any threshold measured (section 10) |
| `rt_im_train.q1_calibrate` (automatic for ZT Scan files, scanning quadrupole only) | none; a `q1_calibration` block in `cal.json` (docs/08 section 4h), with `fragment_decision` under `extract.q1_fragments` | diagnostic: every other artifact unchanged (scored table byte-identical on ZT Scan REP01) |
| `extract.q1_fragments` + `features.q1_fragment_features` (automatic for ZT Scan files, scanning quadrupole only) | psms_extracted v6: the `q1f_*` summary (F64) and four `q1f_frag_*` per-fragment lists; eight features (section 12) | ZT Scan REP01: +1.45% precursors (four NN seeds, beyond the seed range), +1.26% human precursors at a matched entrapment FDP, combined FDP at 1% 1.31% against 1.39%; one acquisition |
| `retrace.q1_repick` (automatic for ZT Scan files, needs `extract.q1_fragments`) | `q1f_apex_moved_s`; moves the rank-0 apex to the 2D apex of the fragments that belong (section 12) | ZT Scan REP01, on top of the decision: +2.96% precursors, +2.92% human precursors at a matched FDP (four NN seeds); 60% of the accepted targets' apexes move, by one cycle; extract +40-75% |
| `features.ms1_envelope` (default false) | psms_extracted v7: 16 F64 cols `ms1e_*`; nine features (section 13) | ZT Scan REP01, four NN seeds: human precursors at a matched entrapment FDP +0.01%, precursors on the human library +0.05%, both inside the seed range; label-blind (decoy against low-target KS at most 0.005); keep off |
| `features.q1_precursor`, `features.q1_multiscale`, `features.q1_deconvolved` (automatic for ZT Scan files, need `extract.q1_fragments`) | psms_extracted v8: 8 `q1p_*`, 9 or 12 `q1m*`, 5 `q1d_*` F64 cols; as many features (section 14) | ZT Scan REP01, four NN seeds, the three on top of the recipe: +0.76% precursors, +1.38% human precursors at a matched entrapment FDP, beyond the seed spread on both libraries; each alone inside it except `q1_deconvolved`; in `configs/examples/ztscan.json`; one acquisition |
| `extract.q1_deconvolved`, `search_seed.q1_deconvolved` (default false) | none; only Q1-consistent peaks are counted (section 14) | ZT Scan REP01: -25.6% precursors (extract, learned 1.0-step tolerance), -8.4% at 2 steps, -2.5% (seed), -38% (both); keep off |
| `rescore.shadow_q1` (automatic for ZT Scan files, needs `extract.q1_fragments`) | none; the shadow test co-isolates by m/z and counts shared fragments by their Q1 centre (docs/11) | ZT Scan REP01: no count change (inside the seed range), fewer entrapment precursors at 1%; with the other three keys +3.02% precursors, +3.17% at a matched FDP, measured FDP at 1% 1.22% against 1.39%, +2.3% in a three-run experiment |
| `extract.ms1_calibrate = "surface"` (default `true`) | none; MS1 reads at a per-m/z, per-retention-time offset (section 11) | ZT Scan REP01: +0.2% precursors, +0.2% human precursors on an entrapment library at a combined FDP of 1.39% (one seed); with `prec_tol_auto` and `q1_calibrate` +0.46% (four NN seeds), +0.19% at 1.40% against 1.39%; one acquisition |
| `extract.prec_tol_auto` (default false) | none; sets `prec_tol_ppm` from the calibrated MS1 residuals (section 11) | ZT Scan REP01: 7.1-7.4 ppm, +0.45% precursors, -0.1% human precursors on the entrapment library at 1.37% (one seed); one acquisition |
| `compete.mode` (`CompetitionMode`, `config.rs:840`, default `winner_take_all`) | retains more rows in `competed`; optional `<out>.compete_audit.parquet` via `emit_competition_audit` (`config.rs:850`) | not measured; gate pending |

## Invariants, determinism, gotchas

- **Determinism**: output is emitted in ascending `candidate_id` order
  (`extract.rs:1651`); the parallel per-candidate map preserves that order via
  `collect()`. Per-scan fragment values are read in ascending ordinal order
  (`ScanGroups`, which replaced a `BTreeMap` per scan group) so f32 apex sums have a
  fixed addition order. The parallel window accumulation is documented
  as bit-identical to the serial loop (`extract.rs:1456`). A
  HashMap f32 sum shifting the apex once broke reproducibility; keep ordered maps
  and sorted iteration wherever floats are summed.
- **Default-off contract**: `retain_top_peaks=1`, `promote_top_peaks=1`,
  `apex_evidence_rank=false`, `emit_contested_features=false`,
  `emit_demix_features=false`, `emit_gate_diagnostics=false`, `peak_claim=None`
  make the schema and per-candidate compute byte-identical to the production chain.
  Every sensitivity knob added here must keep that property. Note that
  `promote_top_peaks` is schema-neutral but **not** row-neutral, so it is the one
  knob in this list that alters the FDR population when enabled.
- **The gate optimum depends on the rescorer**, not the gate in isolation. The
  full-feature search found `spectral_entropy_similarity_sqrt` the single best
  target/decoy discriminator (AUC 0.826), yet gating on it *regressed*
  end-to-end identifications versus the apex gate, because gating on the
  rescorer's own best feature enriches hard decoys. "Best discriminator" is the
  wrong criterion for a gate; the lever is the rescorer. This is why every gate
  change must pass an entrapment-holdout gate before being enabled by default.
- **The MS2 peak budget is set in `convert`, not here.** `convert.rs:76-79` is
  the only MS2 peak truncation in the chain; extract reads whatever
  `spectra_ms2.parquet` contains. The two peak limits in the pipeline are
  independent: on a 50-window Orbitrap DIA run, moving `--top-peaks-ms2` from
  300 to uncapped left seed output identical (80,474 seed PSMs) while moving
  extract's accepted-candidate count from 188,027 to 2,286,840 (12.2x) and
  extract wall clock from 57.9 s to 91.9 s. Uncapping is cheap at this stage
  relative to the identification difference it produces, but the accepted-row
  count it feeds into `features` and `compete` grows by an order of magnitude,
  so size the downstream stages accordingly.
- **Mass recal is a divisor, not a subtraction**: observed m/z is corrected by
  `q_mz = peak.mz / factor_at(peak.mz)` where `factor_at` returns
  `1 + ppm(mz)*1e-6` (`extract.rs:640`, applied at `extract.rs:705`); a
  missing/absent `masscal.json` yields offset 0 and the config tolerance. The ppm
  term may be m/z-dependent, so do not assume one constant offset per run.
- **Empty vs zero traces**: a never-observed predicted fragment carries an empty
  trace, not a grid-length zero vector (`extract.rs:2098`). Downstream code must
  treat an empty trace as `obs_apex = 0`.
- **`apex_im` is null on every 3D run** (v2); only a v4 table, written from spectra
  with 1/K0, fills it, together with `apex_im_mad`, `ms1_apex_im` and `im_pred_cal`.
  Read the version (`extract::psms_schema`) or the column, never assume a value.
- **`restrict_candidates` only forces the serial path in the non-two-pass case.**
  In the non-two-pass branch the parallel window accumulation is used only when a
  fragindex is present **and** there is no `restrict` list (`extract.rs:1455`); a
  `restrict` list therefore routes to the slower serial single-pass loop, which
  honors the allowlist and every non-co-elution `peak_claim` strategy. In the
  two-pass branch (`Coelution*` or `emit_contested_features`) extraction stays on
  the parallel `extract_twopass_windows` path regardless of `restrict`, which
  applies the allowlist inside each pass's push closure (`extract.rs:853`,
  `extract.rs:907`, wired at `extract.rs:1578`).
- **`peaks.parquet` is not scored** and never enters FDR; it is a research
  sidecar. The `peak_rank` **column** of `psms_extracted` is a different thing:
  rank-0 rows are the ordinary selected apex, and ranks >= 1 appear only under
  `promote_top_peaks > 1`, where they are scored and do enter FDR.
- Chromatogram list columns are `LargeListF32` on purpose (`extract.rs:2534`); do
  not downgrade to 32-bit `ListF32` or wide-open gates overflow the offset buffer.

## Tests

Only the pure gate-scoring helpers, the mass-offset interpolator, and the peak
enumerator are unit-tested; there is no stage-level test of `run` itself. The
tests encode the behavioral invariants that gate tuning must preserve.

- `mass_offset_tests` (`extract.rs:2604`): scalar and m/z-grid ppm interpolation
  agree at the anchors and clamp outside them.
- `coelution_tests` (`extract.rs:2630`): co-eluting fragments score `> 0.95`
  (`extract.rs:2651`); a strong non-co-eluting interferent drops the co-elution
  score (`extract.rs:2665`); fewer than 3 scan groups returns `1.0`
  (do-not-reject) rather than a low score (`extract.rs:2674`); `peak_spectral`
  scores `> 0.99` when the peak-integrated pattern matches predicted
  (`extract.rs:2688`) and still recovers a fragment that is momentarily unsampled
  at the apex scan by integrating over the peak (`extract.rs:2705`, the DIA
  scan-gap case the single-scan apex Pearson fails).
- `the_im_gate_drops_out_of_window_peaks_only`: the 1/K0 gate removes exactly the
  hits outside a candidate's window, on the task-local and the global index, and is
  inert on 3D scans and with the gate off; `a_hit_carries_its_mobility_at_no_size_cost`
  pins `Hit` at 16 bytes. `the_streamed_psms_table_is_the_write_table_file` covers the
  v4 columns, and `extract_writes_the_1k0_evidence_only_for_spectra_with_ion_mobility`
  (`tests/pipeline.rs`) the stage end to end: 3D bytes unchanged under every 1/K0
  setting, v4 evidence values, chromatograms v2/v3/v4, the gate and `auto`.
- `peaks::tests` (`peaks.rs:154`): empty/all-zero profiles yield no peaks; a
  clean triangular peak resolves apex and 1/3-height boundaries; `k=1` keeps only
  the strongest-area peak; a dominant interference peak with `k=1` discards a
  weaker true peak but `k>=2` retains it (the core sensitivity behavior,
  `peaks.rs:190`); two maxima rank by area; a left-edge-truncated peak apexes at
  index 0; the prominence filter suppresses a noise bump; a shoulder inside a
  stronger envelope collapses into it; and repeated calls are bit-identical with
  ties broken by earliest apex (`peaks.rs:260`).

## How to extend / modify

- **A new gate metric**: add a variant to `GateMode` (`config.rs:735`), add a lazy
  score closure next to `apex_pearson`/`peak_spec`/`coel` (`extract.rs:1989`), and
  a match arm in the acceptance gate (`extract.rs:2010`). If it is worth
  diagnosing, also wire it into the `emit_gate_diagnostics` tuple
  (`extract.rs:2035`) and the conditional column block (`extract.rs:2511`). Default
  the gate off and validate against entrapment before enabling.
- **A new peak-claim strategy**: add a `PeakClaim` variant (`config.rs:132`); if it
  needs elution profiles, extend the two-pass trigger (`extract.rs:1443`), the
  `reassign` predicate (`extract.rs:1557`), and the reassignment match in
  `extract_twopass_windows` (`extract.rs:1236`); otherwise add it to the serial
  single-pass match (`extract.rs:1500`) and the parallel accumulation
  (`extract.rs:718`). Keep tie-breaks deterministic (lowest `candidate_id`). Prefer
  a cue in `ClaimCues` (`config.rs:192`) over a new variant when the change is
  really a new per-claimant weight: `claim_cue_multiplier` (`extract.rs:163`)
  already composes them and defaults each to 1.0.
- **Scoring the top-K peaks**: the `retain_top_peaks` sidecar peaks are unscored.
  `promote_top_peaks` is the partial closure of that loop: it emits alternate peaks
  as real `psms_extracted` rows carrying `peak_rank`, so they pick up the full
  feature vector downstream. What is still missing is per-candidate q-collapse over
  `(candidate_id, peak_rank)` and entrapment validation, so it stays default-off.
- **New per-PSM columns**: append to the `CandOut` struct (`extract.rs:2682`), set
  it in the `rank0` construction (`extract.rs:2203`) and in the promoted-alternate
  construction (`extract.rs:2297`), add a vector to `PsmRows`, push it in
  `PsmRows::push` and emit its `Col` in `PsmRows::take_cols`, which writes the
  columns in the table's order for every streamed chunk. Gate any non-production
  column behind an `emit_*` flag to
  preserve the byte-identical default schema, and bump the schema version in
  `schema.rs` if the default schema changes.
- **IM / 4D**: section 8. A new 1/K0 use goes through `RtWindows::admits` for the
  window test and through `has_im` (the scans' `im` lists) for anything it writes, so
  a 3D run keeps its bytes. A new 4D column goes into the v4 branch of
  `PsmRows::take_cols`; a new chromatogram list needs a new number in
  `artifact::chromatograms`.
