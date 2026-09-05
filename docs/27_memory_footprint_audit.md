# 27. Memory footprint audit and reduction plan

Scope: the single-run default workflow driven by `mumdia run`
(`stages/run.rs:208-489`): convert, search-seed, optional DeepLC fine-tune,
rt-im-train, extract, features, compete, rescore, quant, report. Imported-library
mode is the validated production path (`CLAUDE.md`, "Validated sensitivity
workflow") and is the case analysed here. Target machine: a 32 GB laptop.

Two facts frame the whole plan:

1. Stages do not share memory. Every `::run` in `run.rs` takes paths, reloads
   its inputs, and drops everything on return. Peak RSS of `mumdia run` is
   therefore the *maximum* over stages, not the sum. Reducing the tallest stage
   is the only thing that moves the peak.
2. The dominant pattern is "materialise everything, then compute". The IO layer
   (`mumdia-io/src/table.rs:282-286`) reads a whole parquet file into
   `Vec<RecordBatch>` and the typed getters (`table.rs:387-425`) then copy each
   column into an owned `Vec`. During a stage's load phase both copies coexist.
   That doubling is the single most common cause of peak RSS in the codebase and
   is fixable once, in one crate.

## 0. Measured status (2026-09-04)

Section 1 below is the code-derived model the plan was written from. This section is what
was then measured, on doxy (128 cores, 755 GB) with 32 threads, one HYE file
(`LFQ_Orbitrap_AIF_Condition_B_Sample_Alpha_01.mzML`), the imported HYE library, a
pre-fine-tuned precursor table (`rt_im_train.finetune_deeplc = false`, so the RT model is
identical across arms), Extended features, `compete.group_by = peptidoform_charge` and
`nn_torch`. Peak resident set is `/usr/bin/time -v` "Maximum resident set size", which is
authoritative; `bench/mem_profile.py` sampled the process tree in parallel.

Whole-run A/B, both arms stopped at 42 min during rescore, after the peak was set:

| arm | peak RSS |
|-----|----------|
| before (`52c2927`) | 231.0 GiB |
| streaming + f32 (`503eadb`) | 86.6 GiB |

Per-buffer payload from `mumdia::memlog` on the instrumented build (`1d1ed7f`), at
2,603,894 candidates and 38.9M chromatogram rows:

| buffer | GiB |
|--------|-----|
| library steady state | 2.26 |
| MS2 scans | 1.41 |
| MS1 scans | 0.02 |
| extract traces, largest chunk in flight | 0.074 (61.8 streamed) |
| features chromatogram store | 62.68 (fragment 48.22 + MS1 XIC 14.46) |
| features `fmap` value matrix | 7.51 |
| rescore feature matrix | 4.05 (features 3.75 + metadata 0.30) |

The incremental writer removed extract's whole-run trace residency as intended, and the
peak moved to features, which re-materialised the traces the writer had just streamed out.
The gap between the 70.2 GiB of named payload and the 86.6 GiB sampled peak is Arrow
batches, the per-PSM extended-value vectors, and allocator slack. A dhat profile of that
features stage at full scale (`bench/dhat_top.py`, 81.9 GiB live at the dhat peak, 5.5 h
under instrumentation) attributed it directly: two `TableFile::list_f32` calls, the `rt` and
`intensity` trace columns as one `Vec<f32>` per row, held 30.2 GiB each (73.8% of the
peak), the `fmap` closure 6.7 GiB and `Vec::push` growth 4.8 GiB. That is the structure
section 3.4 replaced.

Per stage, from the instrumented build's complete run (1:07:30 wall, whole-run peak
86.6 GiB, 66,081 target PSMs and 59,515 peptides at 1%). This is the first profile with
usable stage labels; the earlier A/B arms predate the `stage=` regex fix in `5aa0eb6`:

| stage | wall s | peak tree RSS GiB |
|-------|--------|-------------------|
| convert | 15.9 | 0.20 |
| search-seed | 23.9 | 6.81 |
| rt-im-train | 2.4 | 5.19 |
| extract | 335.6 | 61.38 |
| features | 348.7 | 86.62 |
| compete | 34.9 | 4.55 |
| rescore | 3186.8 | 33.04 |
| quant | 100.9 | 2.81 |
| report | 1.0 | 2.80 |

Rescore is 79% of the wall clock and the second tallest stage; its 33.04 GiB is the Rust
process and the NN sidecar together, which is what item 3.5 addresses.

State of the plan in section 3:

| item | state | measured effect |
|------|-------|-----------------|
| 3.1 streaming typed readers | shipped `503eadb` | part of 231.0 -> 86.6 GiB |
| 3.2 extract incremental write | shipped `503eadb` | trace residency 61.8 GiB -> 0.074 GiB in flight |
| 3.3 f32 bulk arrays | shipped `503eadb` | library 26 -> 22 B per fragment row; rescore matrix halved |
| 3.4 features chunked | shipped | features stage 86.6 -> 3.03 GiB |
| 3.10 extract window-closing flush | shipped | extract stage 61.33 -> 28.13 GiB |
| 3.10 incremental merge + `windows_in_flight` cap | shipped 2026-09-05 | extract stage 28.13 -> 16.57 GiB (12.31 at 8 in flight) |
| 3.5 rescore flat f32 matrix | half shipped `503eadb` | matrix is flat f32; the binary sidecar handoff and the early drop are not done |
| 3.6 quant reads accepted rows | shipped `503eadb` | |
| 3.7 compete key columns | shipped `503eadb` | |
| 3.8 convert batched write | shipped `503eadb` | |
| 3.9 library streamed load | shipped `503eadb` | removes the 23 GiB Arrow transient |

Extract was then instrumented the same way, because only 3.8 GiB of its 61.38 GiB was
named. The whole-run hit accumulator (`acc: HashMap<u32, Vec<Hit>>`) holds
1,607,661,233 hits over 8,761,136 candidates, 183 per candidate:

| part | GiB |
|------|-----|
| hits payload (24 B per `Hit`) | 35.93 |
| growth slack | 1.83 |
| `Vec` spine | 0.20 |
| hash table | 0.45 |

Flushing candidates as their windows close (section 3.10) took that stage to 28.13 GiB at
identical output. The tallest stage on this file is now rescore at 33.04 GiB, which is the
Rust process and the NN sidecar together, i.e. item 3.5.

## 1. Static memory model per stage

All sizes are resident-set estimates derived from the code. Symbols: `P` = MS2
peaks in the run, `S` = MS2 scans, `C` = library candidates, `F` = library
fragment rows, `N` = extracted candidate rows (PSM rows), `T` = trace length
(scans in the extraction window), `K` = fragments per candidate, `D` = feature
count (387 for `Extended`).

### 1.1 convert (`stages/convert.rs:107-182`)

Whole run is decoded into `Vec<Vec<f32>>` for m/z and intensity (MS1 and MS2)
before a single `write_table` call. Resident: `8P` bytes plus one `Vec` header
and one heap allocation per spectrum per column, plus the mzdata spectrum being
decoded (f64 arrays, transient). Estimate for a 2 h Orbitrap run with 60M MS2
peaks: about 0.5 GB plus MS1. Not the tallest stage, but it is the simplest to
stream and it sets the shape of the artifact every later stage reads.

### 1.2 Library load (`index.rs:60-200`), used by search-seed and extract

Load sequence: precursor table (Arrow batches plus owned column Vecs, then
`drop(pt)` at `index.rs:96`), then the fragment table projected to four columns
(`index.rs:102-110`), then owned Vecs, then `drop(ft)`, then the counting sort
into flat arrays, then the sorted m/z index (`idx_mz`, `idx_cid`, `idx_int`).

Resident after load, per fragment row: `frag_mz` f64 (8) + `frag_int` f32 (4) +
`frag_name_id` u16 (2) + `idx_mz` f32 (4) + `idx_cid` u32 (4) + `idx_int` f32
(4) = 26 bytes. Peak during load is higher: the projected Arrow batches for
`F` rows (`candidate_id` u32, `mz` f64, `predicted_intensity` f32, `name` utf8
with offsets and bytes, about 24-30 bytes per row) coexist with the owned
copies of the same four columns. The comment at `index.rs:93-95` records 23 GB
of Arrow batches at 657M fragment rows for the modification-expanded library.
For that library the load peak is roughly 40 GB and the steady state about 17 GB:
it cannot run on 32 GB at all. The AIF DIA-NN library is much smaller (about
1/20 of that), so on the benchmark this is not the wall, but it is the wall for
any PTM search and for the augmented HYE libraries.

### 1.3 search-seed (`stages/search_seed.rs:55`)

`load_ms2` materialises every MS2 scan (`spectra.rs:20`, f32 m/z and intensity,
one heap allocation per array per scan) plus the library. Hits accumulate per
scan and are reduced to the seed table. Resident: library steady state plus `8P`
plus hit buffers. Library-dominated.

### 1.4 rt-im-train

Reads the seed table and library precursors (`Table::read` twice). Small
compared with the others. The DeepLC fine-tune sidecar is the exception: it
holds a pandas frame of every library peptidoform and a torch model. For the
modification-expanded library that frame alone is several GB in Python object
overhead; it is outside the Rust heap and invisible to a Rust-only profile.

### 1.5 extract (`stages/extract.rs:1348-2560`)

Resident at once:

- all MS2 scans (`load_ms2`, `8P`) and, if MS1 is given, all MS1 scans
  (`spectra.rs:12-16`, `Ms1Scan.mz` is `Vec<f64>`, so 12 bytes per MS1 peak);
- the library (section 1.2 steady state);
- per-window narrowing caches (`window_narrow`, `extract.rs:699,836`), one per
  isolation window in flight under rayon;
- the full output before write: `Vec<ChromOutputRow>` (`extract.rs:1698,2081`)
  holding one `Vec<f32>` RT trace and one `Vec<f32>` intensity trace per
  fragment per candidate, i.e. `8 N K T` bytes plus two heap allocations per
  fragment, and `psms_cols` alongside. At `N` = 1.5M, `K` = 6, `T` = 60 that is
  about 4.3 GB of trace payload plus about 0.4 GB of allocation headers, all
  converted again into Arrow `LargeList<f32>` builders (`extract.rs:2537-2538`)
  at write time, so trace memory roughly doubles at the moment `write_table`
  runs.

Extract is the tallest single-run stage on a large library because it is the
only stage that holds spectra, library and full output simultaneously.

### 1.6 features (`stages/features.rs:363-560, 1284-1541`)

Three `Table::read` calls (PSMs, chromatograms, library precursors). The
chromatogram table's `LargeList<f32>` batches stay resident while
`traces_full: Vec<Vec<f64>>` (`features.rs:367,478`) is built from them: every
trace is widened to f64 and gets its own heap allocation, then `traces`
(`features.rs:526`) is a second copy over the peak window, and `ms1_xic`
(`features.rs:408,556`) a third family. The feature matrix is assembled as
`fmap: HashMap<String, Vec<f64>>` with `D` columns of `N` f64: `8 N D` = 4.6 GB
at 1.5M rows (the comment at `features.rs:1068-1071` records exactly this for
the PIN transpose that was already removed). Peak here is Arrow chromatogram
batches + f64 traces + f64 feature columns, roughly 3x the chromatogram file
size plus 4.6 GB.

### 1.7 compete (`stages/compete.rs`)

One `Table::read` of the full features table (Arrow) plus owned key columns,
then row selection, then a rewrite of the surviving rows. Peak is about two
copies of the features table, so `2 x 8 N D` plus strings. On the default key
about half the rows are deleted here (`CLAUDE.md`, competition key), which is
why the stages after compete are smaller than features.

### 1.8 rescore (`stages/rescore.rs:60-140, 831-846`)

Loads the competed table, then transposes into `feats: Vec<Vec<f64>>`
(`rescore.rs:60`), row-major with one heap allocation per PSM. Per row:
`8 D` = 3096 bytes payload + 24 bytes `Vec` header + allocator overhead. The
comment at `rescore.rs:831-832` records about 27 GB for an experiment-wide pool.
`fcols: Vec<Vec<f64>>` (`rescore.rs:94`) is the column-major copy that exists
during the transpose, so the transient peak is `2 x 8 N D` plus the Arrow
batches. The PIN TSV is then written for the sidecar (text, about 4 KB per row,
disk only).

The `nn_torch` worker (`scripts/nn_rescore_worker.py:20-25, 299-371`) reads the
PIN with pyarrow, builds `Xs` as float32 (`4 N D`), and standardises into it.
Below `MUMDIA_NN_STREAM_GB` (default 4) it holds the pyarrow table and the
float32 matrix together, then torch batches. Above it, the memmap path holds one
batch. A 4 GB matrix plus the pyarrow table plus torch working set is about
10-12 GB in the Python process, at the same time as the Rust process still holds
`feats` (it waits on the sidecar). Rust plus Python together is the true
rescore peak: about 6 GB Rust + 12 GB Python at 1.5M competed rows.

### 1.9 quant (`stages/quant.rs`, six `Table::read` calls)

Reads the scored table and the full chromatogram table again, although only the
accepted rows (typically 1-5% of `N` after `q_filter`) are quantified. Peak is
the whole chromatogram file in Arrow form plus the derived per-row traces.
Reading the chromatogram table dominates; the actual quantification is small.

### 1.10 report

Three `Table::read` calls on already-filtered tables. Small.

## 2. Ranking: what takes the memory

Ordered by contribution to single-run peak on 32 GB, largest first.

| Rank | Stage / structure | Why it is large | Mechanism |
|------|-------------------|-----------------|-----------|
| 1 | extract: chromatogram rows held until one final `write_table` | `8 N K T` traces + per-fragment allocations, then doubled into Arrow builders | materialise-then-write |
| 2 | library load: Arrow batches + owned copies of the fragment table | `F` up to hundreds of millions of rows, both copies alive | `Table::read` doubling |
| 3 | features: chromatogram Arrow + f64 traces (x2) + `8 N D` f64 feature columns | f32 traces widened to f64 and copied twice; 387 f64 columns | widening + copies |
| 4 | rescore (Rust + Python): `Vec<Vec<f64>>` row-major + column-major transient + float32 matrix in the worker, all alive together | two processes each holding a full matrix | transpose copy + sidecar handoff by TSV |
| 5 | quant: full chromatogram table read to quantify a few percent of rows | no predicate pushdown | read-everything |
| 6 | compete: two copies of the features table | full read then full rewrite | `Table::read` doubling |
| 7 | convert: whole run resident before write | single `write_table` | materialise-then-write |
| 8 | MS1 scans as f64 m/z (`spectra.rs:15`) | 12 B per MS1 peak where 8 would do | precision choice |

## 3. Reduction plan per step

Ordered by leverage divided by risk. Each item names what changes, the expected
effect, and how to check that results stay the same.

### 3.1 IO layer: streaming typed readers (one change, every stage benefits)

Add to `mumdia-io`:

```text
read_col_f64(path, name) -> Vec<f64>      // decode batch by batch straight into the output
read_col_f32(path, name) -> Vec<f32>
read_col_u32 / read_col_str / read_col_list_f32(path, name) -> (offsets, values)
for_each_batch(path, cols, |RecordBatch| ...)   // streaming visitor, one batch resident
```

Peak for a column becomes `output + one batch` instead of `whole file + output`.
Convert `Table::read` call sites in features, compete, rescore, quant, report,
rt-im-train and `index.rs` to these. Where a stage needs many columns, use
`for_each_batch` and push into preallocated Vecs sized from `nrows(path)`
(`table.rs:292-299` already reads that from the footer). Results: byte-identical
by construction. This alone removes the load-phase doubling in ranks 2, 3, 4, 6.

### 3.2 extract: write chromatograms and PSMs incrementally

Replace the `Vec<ChromOutputRow>` accumulation with a parquet writer opened
before the window loop that appends one row group per finished isolation
window (windows are independent, `extract.rs:621-626, 778-785`). Under rayon,
send finished window results through a channel to a single writer thread so
row-group order stays deterministic (sort by window key before writing, as the
current concatenation order already does). Peak drops from "all traces plus
Arrow builders" to "traces of the windows currently in flight". Expected: 4-9 GB
less at 1.5M rows. Results: identical rows; only row-group boundaries change.
Downstream readers already iterate batches, so nothing else changes.

### 3.3 Store bulk arrays as f32; keep accumulations in f64

Precision policy: f32 for storage of m/z, intensities, traces and feature
values; f64 for every reduction (sums, dot products, correlations, LOESS,
q-values). f32 m/z has a relative precision of 6e-8, i.e. 0.06 ppm at 1000 m/z,
against matching tolerances of 5-20 ppm, so matching decisions cannot change.
Concretely:

- `Library.frag_mz: Vec<f64>` to `Vec<f32>` (`index.rs:40`), saves 4 B per
  fragment row (15% of the library steady state). `prec_mz` stays f64 because the
  candidate range search compares against isolation window edges; it is only `C`
  rows.
- `Ms1Scan.mz: Vec<f64>` to `Vec<f32>` (`spectra.rs:15`); the artifact already
  stores f32, so this removes a widening copy, not information.
- features: keep `traces_full` as `Vec<f32>` flat with an offsets array
  (one allocation, not one per trace), and compute correlations in f64 from the
  f32 values. Emit the feature table as f32. The rescore stage already writes
  f32 features for the sidecar (`rescore.rs:826-829`) and the NN worker trains
  on float32, so the classifier input is unchanged. Native linear and mokapot
  paths gain a rounding at the 7th significant digit; treat as "mostly similar"
  and validate as in section 5.

The task note allows a precision change; the direction that saves memory is
f64 to f32 for storage, with f64 kept for arithmetic, and that is what this
plan does.

### 3.4 features: chunked computation (shipped)

The stage processes the run in chunks of at most `CHUNK_CHROM_ROWS` (2^20)
chromatogram rows, cut so that a candidate is never split, and appends one row
group per chunk. Three parts:

- `plan_chunks` fixes the boundaries from the `candidate_id` column of both
  tables, which is all that is read up front. Extract emits a PSM row and that
  row's chromatogram rows in one pass, so both tables carry the same candidates
  in the same order with each candidate's rows contiguous. That invariant is
  verified, not assumed: a table that cannot be joined this way is a hard error
  naming the artifact.
- the chunk store shares one RT axis across the rows of a candidate (window-grid
  extraction gives every fragment of a candidate the identical axis) and holds
  one flat buffer per array with interned fragment names, instead of two `Vec`s
  and a `String` per row. That halves the traces by itself: a 62.7 GiB store
  became 33.2 GiB streamed.
- `ValueMatrix` is one flat column-major buffer per chunk in place of the
  `HashMap<&str, Vec<f64>>`, and the PIN is written row by row as chunks are
  computed.

Cost: `bound_from_confident` (default on) needs the global elution half-widths
before the first feature, so it takes one extra streaming pass, which decodes but
never copies the rows of non-confident candidates.

Measured on the artifacts of the arm in section 0, against the features table
that arm wrote: peak RSS 86.6 -> 3.03 GiB, 38 chunks, largest chunk 0.92 GiB of
traces plus 0.21 GiB of feature values, all 398 columns of 2,603,894 rows
identical. Wall time 7:15 against 5:05; the extra pass is the difference.
`run_with_chunk_rows` exposes the chunk size, and the pipeline test runs one
chunk against one candidate per chunk, comparing every f64 column bit for bit
plus the PIN bytes.

### 3.5 rescore: flat f32 matrix and a binary sidecar handoff

- Build `feats` as one row-major `Vec<f32>` of `N D` values, not
  `Vec<Vec<f64>>`. Saves the per-row allocation and 4 B per value: 4x smaller,
  and no column-major transient because the streaming reader fills rows in place.
- Hand the sidecar a parquet or Arrow IPC file instead of the PIN TSV. The NN
  worker already reads with pyarrow; with IPC it can memory-map the matrix and
  build `Xs` from it with one copy instead of parse + table + copy. Then
  standardise in place.
- Lower `MUMDIA_NN_STREAM_GB` from a fixed 4 to a fraction of available RAM
  (for example 12%, read once at worker start) so a 32 GB machine streams above
  about 3.8 GB and a 128 GB server keeps the fast path up to 15 GB. This is the
  same knob that already bit once (`CLAUDE.md`, sidecar contracts).
- Drop `feats` in Rust before waiting on the sidecar; the scores come back by
  flat row index (`rescore.rs:760`), so only the metadata columns are needed
  afterwards. This removes the Rust-plus-Python double residency in rank 4.

### 3.6 quant: read only accepted rows

Compute the accepted `candidate_id` set from the scored table first, then read
the chromatogram table with `for_each_batch` and keep only rows whose id is in
the set. With sorted ids and per-window row groups the parquet row-group
statistics allow skipping most row groups without decoding them. Peak becomes
proportional to accepted rows (a few percent of `N`).

### 3.7 compete: key columns only, then a streaming rewrite

Read `base_peptide_id`, `label_code`, `charge`, `peak_rank`, `prelim_score`
with the typed readers, decide winners, then copy surviving rows batch by batch
into the competed parquet. Peak: key columns plus one batch, not two full tables.

### 3.8 convert: batched write

Open the parquet writer at the start and flush every 2,000 spectra as a row
group. Peak becomes one batch of spectra. `--top-peaks-ms2` semantics are
unchanged because the cap is applied per spectrum (`convert.rs:76-79`).

### 3.9 Library load: stream the fragment table into preallocated arrays

Read `nrows(fragments)` from the footer, allocate the four flat arrays once, and
fill them with `for_each_batch`. Then the counting sort. Peak: final arrays plus
one batch, i.e. about 26 B per fragment row (22 after 3.3) with no 23 GB Arrow
transient. This is what makes the modification-expanded library fit on 32 GB
(657M rows x 22 B = 14.5 GB).

### 3.10 extract: flush candidates as their windows close (shipped)

Extract probed every isolation window, merged all hits into one accumulator, and only then
scored candidates. Windows are probed in ascending m/z and precursors are sorted by m/z,
so once the next window's candidate range starts at some id, no later window can add a hit
below it: every candidate below that bound is final. The stage now probes a batch of
windows, merges the partials in window order, then scores and writes everything the next
batch cannot touch.

The ascending-bound argument does not assume disjoint windows, which matters here: the
benchmark acquisition is 151 staggered 8 Th windows stepping 4 Th, so every window overlaps
its neighbour and each candidate is matched by two of them. A decomposition that assumed
one window per candidate would be wrong on this data.

`GROUPS_IN_FLIGHT` is the thread count, because a window is both the unit of parallelism
and the unit of memory. Measured: 61.33 -> 28.13 GiB, largest open accumulator 11.94 GiB,
wall 6:08 against 6:10, both output tables identical value for value (2,603,894 PSM rows
over 21 columns, 38,889,646 chromatogram rows over 7, traces included).

Two follow-ups shipped on 2026-09-05, both value-preserving (PSM and chromatogram tables
identical to the byte, 2,603,894 and 38,889,646 rows):

- **Partials merged as windows complete.** The batch accumulator collected every window's
  partial map and only then merged, so the batch's hits existed twice at that moment.
  Workers now hand each finished partial over a channel and the calling thread merges in
  window order (a reorder buffer holds the ones that finished early), dropping each partial
  as it is merged. 28.13 -> 24.65 GiB, wall 6:08 -> 5:00.
- **`extract.windows_in_flight`.** The batch is the unit of accumulation parallelism, but
  that phase is a small, memory-bound part of the stage, so following the thread count
  bought little and cost a lot. Measured at 32 threads:

  | windows in flight | largest open hits | stage peak | wall |
  |---|---|---|---|
  | 32 (old default: thread count) | 11.94 GiB | 24.65 GiB | 5:00 |
  | **16 (new default cap)** | 6.16 GiB | **16.57 GiB** | 5:04 |
  | 8 | 3.29 GiB | 12.31 GiB | 5:26 |

  The default is now `min(threads, 16)`; a memory-bound machine sets 8 or 4 explicitly.
  With that, the tallest stage of a single HYE run is 16.6 GiB, inside the 32 GB laptop
  this audit set as its target, and 12.3 GiB is available for a 9% wall cost.

What remains: the open hits (6.16 GiB at 16 in flight) are the `Vec<Hit>` payload plus its
growth slack, and `Hit` is 24 bytes where 16 would do (`obs_mz` as an f32 ppm offset from
the theoretical m/z, `rt` as a scan index). That is a precision change and gated.

## 4. Rewrite candidates (structural changes beyond the items above)

- **Extract as a window-streaming pipeline.** Load only the scans of the
  isolation window being processed (the spectra artifact is already
  window-sorted after convert if written per window), not the whole run. Scans,
  narrowed library slice, traces, and output are all per window. Memory becomes
  independent of run length. This is the only change that makes extract scale to
  long gradients and to Astral-scale peak counts.
- **Features fused into extract.** Features are computed from traces that
  extract has just built. Computing them before the traces leave the window
  worker removes the chromatogram round trip (write, re-read in features,
  re-read in quant). The chromatogram artifact would then be written only for
  accepted rows after rescore (from a compact trace store keyed by candidate id).
  Larger change: it moves the feature schema version into extract and changes
  the stage boundary that `run.rs` and the docs describe.
- **Library on disk with a memory-mapped index.** Store `idx_mz`, `idx_cid` and
  the flat fragment arrays as a fixed-layout binary sidecar next to the parquet
  library, memory-map it, and let the OS page it. Steady-state RSS becomes the
  touched pages only. Worth it for PTM libraries; not needed for the AIF library.

## 5. Keeping results "mostly similar": acceptance gates

Items 3.1, 3.2, 3.4, 3.6, 3.7, 3.8 and 3.9 change layout only and must produce
byte-identical parquet payloads (row-group boundaries may differ; compare sorted
tables, not file hashes). Item 3.3 and the f32 matrix in 3.5 change numeric
precision and are gated as follows, run on the AIF benchmark and one second
acquisition (`CLAUDE.md`, "Validate new sensitivity defaults"):

- stripped-peptide count at 1% `peptide_q_value` within 0.3% of baseline;
- empirical decoy fraction unchanged to two decimals (0.98%);
- Jaccard of accepted peptidoform sets at 1% at least 0.995;
- quant: median log2 ratio bias and CV on the known-ratio HYE set within 0.005
  of baseline;
- `cal.json` RT residuals unchanged (they never pass through f32 storage).

Sidecar nondeterminism (DeepLC fine-tune draw variance, NN kernels) is larger
than any f32 rounding effect, so compare arms with `rt_im_train.finetune_deeplc
= false` on a pre-fine-tuned library and with a fixed `MUMDIA_NN_SEEDS`.

## 6. Profiling protocol

`bench/mem_profile.py` runs a command, samples the RSS of the process tree
(engine plus Python sidecars) every 0.2 s, and splits the samples into stages
either by `RUST_LOG=info` stage lines from `mumdia run` or by running each stage
as its own command. Output is a TSV of per-stage peak RSS, duration, and the
peak sidecar share. Use `--stage-regex` to match the actual banner text of the
build being profiled.

Recommended first measurement, on the AIF benchmark with the augmented library,
uncapped conversion, to establish the baseline the plan above is judged against:

```text
python bench/mem_profile.py --out mem_aif_baseline.tsv -- \
  mumdia run --lib-precursors lib/lib_precursors_aug.parquet \
  --lib-fragments lib/lib_fragments_aug.parquet \
  --mzml mzml_files/LFQ_Orbitrap_AIF_Ecoli_01.mzML \
  --out-dir out_memprof --config config.local-diann-lib.json
```

On the Linux server the same can be cross-checked per stage with
`/usr/bin/time -v` (field "Maximum resident set size"), which is the
authoritative number; the sampler can undercount transients shorter than the
sampling interval.

Status as of 2026-09-04: section 1 remains the code-derived (static) model; section 0
carries the measured numbers and the state of each plan item. Sections 3.1-3.3 and
3.6-3.9 shipped in `503eadb`, 3.4 and 3.10 in the commits that added those sections, and
3.5 is half done. Stage peaks on the benchmark are now extract 16.57 GiB (default) or 12.31
(8 windows in flight), rescore 8.95 GiB by default (parquet handoff, docs/28 section 11) or
5.5 GiB with the opt-in recipe, and features 3.03 GiB. A single HYE run fits the 32 GB target
machine. The measurement to take is a full `mumdia run` on the current build.
