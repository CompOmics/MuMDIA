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

### 3.4 features: chunked computation

With 3.1 in place, iterate the chromatogram table batch by batch (candidate
groups do not span batches if extract writes one row group per window, 3.2),
compute the `D` features for that chunk, and append a row group to the features
parquet. Resident: one chunk of traces plus `D x chunk` feature values. Removes
the `8 N D` matrix entirely. Requires the "rank-0 row carries the shared
chromatogram" convention (`extract.rs:2330`) to be satisfied within a chunk,
which it is when chunks align with extract's row groups.

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

Status as of 2026-09-04: this document is a code-derived (static) profile. The
harness is written but the baseline run has not been executed in this repository
yet; fill section 1 with measured numbers from the first run before changing
code, so that each item in section 3 can be credited with a measured delta.
