# Changelog

All notable changes to MuMDIA are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`0.1.0` is the first tagged release of the Rust engine. The superseded
Python implementation remains available at the tag `legacy-python-v1`.

`0.4.0` is a minor rather than a patch release: it changes results. Multi-head
retention-time calibration is on by default, which on a six-file Astral experiment
moved precursors from 113,961 to 131,646 and cost 1.4x to 1.7x wall clock; the
library it produces is then adapted once per experiment rather than once per run,
which gives about 1% of that back in exchange for removing N-1 full re-predictions.
A run repeated across the upgrade will not reproduce its old counts or its old
runtime. `predict_frag.predictor = "peptdeep"` is a new fragment-intensity
predictor, and `experiment.finetune_scope` is now `experiment.rt_library_scope`
(the old name still parses). The pooled `nn_torch` rescore also changed shape:
about half the memory, and on Intel desktops about a tenth of the time (a six-run
Astral pool went from 118 to 11 minutes on an i9-13900KS), because the trained
network no longer accumulates subnormal floats; and `rescore.train_neg_ratio`
defaults to 2 instead of 3, which raised identifications on every pool measured.

`0.3.0` raises the DeepLC floor to 4.4.0, so every environment must be rebuilt
before upgrading. It also adds multi-head retention-time calibration, off by
default.

`0.2.0` is a minor rather than a patch release: it changes behaviour a user can
see. The default MS2PIP model is `HCDch2`, configurations with out-of-range
numeric values are refused at load instead of running, both TSV reports gain two
columns, the candidate-audit rejection code `NO_PEAK_GROUP` is
`DID_NOT_SURVIVE_EXTRACTION`, a candidate no predictor covered is dropped rather
than given a substitute value, and a stage refuses to write its output over one of
its own inputs.

Two things are versioned independently of this file and matter when reading old
results: the per-artifact Parquet schema versions in
`rust/mumdia/crates/mumdia-core/src/schema.rs`, and the feature-set identity
`classifier_feature_schema_id`, which is a hash of the active feature list rather
than a number. Both are recorded in every run's `manifest.json`.

## [Unreleased]

### Added

- **`rescore.handoff = raw`, an opt-in handoff with no parquet codec on either side.** The
  `nn_torch` worker is given a `.raw.json` description naming a row-major little-endian
  f32 `.npy` matrix and a small parquet of the metadata columns. The engine writes each
  decoded batch straight into the matrix, and the worker copies it into its own with no
  decode and no column-to-row transpose, summing the float64 moments over the parquet
  handoff's 131,072-row groups, so the scores are byte-identical to `parquet` (checked on
  a fixture for both worker backends and on a 41,910-PSM and a 522,237-PSM competed
  table). The file is 4 bytes a value, so it pays where the codec, not the disk, is the
  limit. Validate a new host by rescoring one pool with each handoff and comparing
  `psms_scored.parquet` byte for byte.
- **`groups.pool_chromatograms = false` has quant read a grouped run's band chromatogram
  tables directly.** The pool writes the overlap losers it finds anyway to
  `groups/overlap_losers.parquet` (`band`, `candidate_id`), and quant reads the bands'
  tables in band order, dropping each band's losers, so the pooled `chromatograms.parquet`
  (about 68 GB a run on the immunopeptidomics experiment) is neither written, hashed nor
  read. It holds for overlapping bands too. Default `true`. The quant tables are
  byte-identical either way (tests with two overlapping bands and with a candidate
  straddling two band files; a smoke arm compares the quant tables and TSVs of the
  three-band fixture). What changes is the artifact set: no pooled chromatogram record,
  an `overlap_losers` record (schema `overlap_losers` 1), and the quant report lists the
  band tables with `chromatogram_dropped_candidates`. The band directories are then the
  run's only chromatograms. `mumdia quant` takes several `--chromatograms` and
  `--overlap-losers` for the same read by hand.
- **`groups.pool_competed = false` has rescore read a grouped run's band tables directly.**
  Rescore takes a table-to-source map (`competed_sources` in the scored report), so the
  bands' competed tables in band order give exactly the rows the pooled
  `psms_competed.parquet` holds, and that copy (about 83 GB a run on the
  immunopeptidomics experiment) is not written. It applies only when the bands' library
  row spans are disjoint and neither the candidate audit nor match-between-runs is on;
  otherwise the table is pooled and the log says why. Default `true`. `psms_scored.parquet`
  is byte-identical either way (a smoke arm compares it); the manifest then has no pooled
  competed record, and `mumdia pool --groups-dir` rebuilds the table from the bands.

- **`mumdia sub-library` subsets a library to a set of candidates.** A second pass searches
  the survivors of a first pass (or of `prescan`) as a library of their own, which means
  keeping those precursors, renumbering them to the contiguous `0..n` the fragment index
  requires, and remapping their fragment rows. `scripts/assemble_survivors.py` did that by
  holding the whole precursor table plus a Python string per label: 44 GB and several
  minutes per call on the 142.7M-precursor 8-12-mer immunopeptidomics library. The engine
  streams both tables one batch at a time and keeps two `u32` per library precursor, about
  1.6 GB on that library. The keep decision is unioned over `peptidoform_id`
  (`--no-pair-link` opts out), so a target and its decoy are kept or dropped together and
  exchangeability is what it was in the full library, which is the semantics the script had.
  Not to be confused with an m/z band of a library, which is a row range the engine reads
  directly with no fragment table of its own. The script stays for existing recipes, with a
  note pointing at the command.

### Changed

- **Quant reads less of its two inputs.** The scored table's identity columns
  (`peptidoform`, `protein_group`, `charge`, `base_peptide_id`) are read at the accepted
  rows only, and the identification-apex map holds only the candidates whose
  chromatograms are loaded; the report's `candidates_with_scored_apex` still counts the
  whole table. The chromatogram table is opened with its offset index: a row group whose
  `candidate_id` statistics hold no accepted id is not opened, and in an opened group each
  column skips, unread, every data page that holds no kept row. Rows are never skipped
  inside a page, because parquet-rs steps over a long list row slower than it decodes it.
  Every quant table and report is byte-identical (checked against the previous binary on
  a per-run split of the six-file Astral experiment and on the AIF benchmark, in both the
  old single-row-group and the current chromatogram layouts). On those tables every list
  page holds an accepted row, so no page is skipped and the read time does not move; the
  gain is on tables whose accepted candidates cluster more coarsely than their pages,
  which `selective_read_on_a_real_artifact` measures from a real footer.
  `MUMDIA_QUANT_SELECTIVE_READ=0` turns the page selection off for an A/B.
- **`features.parquet` and `psms_competed.parquet` store the feature columns as float32.**
  Every classifier narrows every feature to f32 before it sees it, so the features stage
  now stores `v as f32` and the classifier inputs, the scores and every scored output are
  unchanged (byte-identical `psms_scored.parquet` on the smoke fixture, and in a test that
  runs both layouts through compete and rescore, `unique_evidence` included). Five columns
  that compete or rescore read as f64 before narrowing stay float64 (`charge`,
  `n_matched_fragments`, `unique_fragment_count`, `peak_contested_frac`,
  `contested_frac`), as do the bookkeeping columns. The two tables are about half the
  bytes, which is the widest write, copy and read of a run (about 83 GB a run of
  `psms_competed` on the immunopeptidomics experiment). The schema versions are now
  `features` 2 and `psms_competed` 4. Compete and rescore still read the previous
  versions: a v1 features table is rewritten into the v4 layout, and a v3 competed table
  scores exactly as its v4 counterpart. An external reader of these files sees float32
  columns. Band tables from before and after the change cannot be pooled into one table;
  re-run the bands with one binary. The PIN (`features.emit_pin`) is written from the f64
  values and does not change.
- **Rescore removes its sidecar files once the scores are read back.** The handoff, the
  fold keys and the worker's output were named after the output and the PID and never
  removed, so they piled up: 7.7 GB per HYE rescore, 359 GB per immunopeptidomics pool.
  They are now removed after `align_sidecar_scores` accepts the scores (a failed worker
  still leaves its input), unless `MUMDIA_KEEP_HANDOFF=1`. `MUMDIA_SIDECAR_DIR` moves the
  work directory for `run`, `run-experiment` and `rescore`, and `mumdia rescore --work-dir`
  names it for one call. The file names carry a nonce beside the PID, so runs that share
  one `MUMDIA_SIDECAR_DIR` (two containers with the same PID) cannot touch each other's
  files, and a handoff or fold-keys write that fails removes what it wrote. Before the
  handoff is written the engine asks the sidecar interpreter for the directory's free
  space. It refuses a run only below what the files cannot be smaller than, which a PIN
  or raw handoff has and a parquet handoff does not, and it warns below their usual size,
  the NN worker's streaming memmap included when the worker may stream; the messages name
  the directory and the way out (`MUMDIA_SIDECAR_SPACE_CHECK=0` skips it). The scores are
  unchanged.
- **`run-experiment` splices the per-run scored tables.** A run's rows are contiguous in the
  pooled scored table, so the by-source split copies every single-source row group as
  bytes and re-encodes only the boundary groups. The per-run `<run>/scored.parquet` tables
  hold the same rows, order and values, so quant and the reports are unchanged, but their
  bytes and the `scored[<run>]` hashes in `experiment_manifest.json` change: a spliced
  group keeps the scored table's 1,048,576-row layout. A nullable `source` (the MBR
  worker's output) is re-encoded as before.

- **The engine holds far fewer heap blocks, because that, not memory, is what a banded
  search runs out of.** A grouped search of a 203M-precursor library died at about 290 GB
  resident with 1.7 TB free, reporting a failed 3 KB allocation. Sampled on the live
  process at 180 GB it held 129,393 memory mappings of the kernel's 1,048,576 per-process
  limit, 92,030 of them 64-256 KB and 36,368 of them 256 KB-1 MB: one heap block per item,
  which mimalloc maps individually. Six subsystems now use flat buffers where they used a
  block per row: the library's `label` column is read as one bit per row rather than one
  `String` (203M blocks), compete groups PSMs through one sorted vector rather than a
  `Vec` per competition group, features holds one buffer rather than a `Vec<f64>` per PSM
  and dense ids rather than a `HashSet` per peptidoform, quant reads chromatograms into
  flat buffers with an axis store rather than three blocks per row, and rescoring's
  per-fold training matrix is one buffer rather than one per training row. Rescore also
  stops materialising the engine's own feature matrix when a sidecar classifier will read
  the handoff parquet instead, which at experiment scale is about 250 GB that need not
  exist. Artifacts are unchanged except where noted below.
- **Pooling a grouped run's bands is a byte copy.** The rows are already in the pooled
  table's order and already encoded, so `pool` splices the bands' parquet column chunks
  into the output and decodes only the row groups whose `candidate_id` statistics say they
  hold a candidate the overlap dedup drops. Measured on one run of the production
  experiment, 63 bands and 136 GB: 3 minutes 12 seconds at 2.6 GB resident, against
  3.5 MB/s at 110% CPU before, which had reached 42.7 GB in 2 hours 50 minutes when it was
  stopped, on a disk that reads 221 MB/s. The pooled tables hold the same rows in the same
  order with the same values; their row groups are the bands' own, so their content hashes
  differ from a re-encoded pool's.
- **A grouped run no longer pools `psms_extracted`.** Its only reader is the candidate
  audit, so it is pooled when `extract.emit_candidate_audit` is set and left per band
  otherwise. The band tables are written either way.
- **`groups.parallel` is clamped to one less than the thread count.** A band in flight
  parks a rayon worker on its extraction's accumulation channel, so as many bands as
  threads leaves no worker to feed them and the run deadlocks with no error and no output
  (reproduced on the fixture at `parallel = 2, --threads 2`). A larger value now warns and
  uses `threads - 1`.
- **`psms_extracted.parquet` and `run_windows.parquet` change content hash.** `write_table`
  now encodes in 65,536-row chunks instead of building one Arrow copy of the whole table,
  and on a column that is entirely NULL -- `apex_im` and the three ion-mobility columns
  always are -- the definition levels are run-encoded, which moves the writer's internal
  mini-batch size and so the page framing. Measured at 196,615 rows: 16 bytes, with every
  row, value and row-group boundary unchanged.
- **`psms_competed.parquet` is published as the features file's own bytes when compete
  removes no row, which is the shipped default.** Under `group_by = peptidoform_charge` the
  competed table has the features table's columns, rows, order and values, and it was
  decoded and re-encoded column by column for nothing (estimated at about 10 s per HYE
  file). When the features file is exactly what that rewrite would write, compete now
  hard-links it to the competed name, falls back to a byte copy where the filesystem refuses
  a link, and rewrites only if both fail. The competed file then has the features file's
  65,536-row groups instead of 131,072-row ones, and its `content_hash` IS the features
  hash, so it differs from an earlier run's whenever the table holds more than one row
  group. On a grouped run the pooled `psms_competed.parquet` inherits the bands' row groups,
  and its hash differs for the same reason. When compete does remove rows and the untouched
  row groups hold at least half of the table's rows, those groups are spliced as bytes and
  only the others are rewritten; a splice that fails falls back to the full rewrite. The
  report records the path in `stats.publish` (`hard_link`, `byte_copy`, `spliced` with
  `rewritten_row_groups` and `row_groups`, or `rewritten`). `psms_scored.parquet` and every
  artifact after it are byte-identical. After a hard link the two names share one file:
  rewriting either through the engine leaves the other alone, but a tool that edits either
  file in place, including pandas `to_parquet` or pyarrow `write_table` onto the existing
  path, changes both.
- **Each artifact is hashed once.** The orchestrators record a stage's outputs in
  `manifest.json` with the hash the stage already computed for its own report instead of
  reading and hashing every file again, and a grouped run under `run-experiment` no longer
  builds the band records it then dropped. Reusing the stage hashes changes no recorded
  hash value.
- **The large artifacts are hashed while they are written.** A writer opened with
  `WriteOptions::content_hash` feeds its bytes to blake3 on the way to the file and returns
  the digest when it closes, so convert, predict-frag, search-seed, rt-im-train, extract,
  features, compete, the pool and rescore no longer read their outputs back to hash them.
  The digest is the same blake3 over the same bytes, so hashing while writing changes no
  artifact byte and no recorded hash by itself (docs/03 "Hash on write"). The capped-writer
  layout changes below (page cut, float plan, PLAIN chromatogram `rt`) do change the bytes
  and content hashes of the files they write.
- **`TableFile::scan` can read each row group's projection in one sequential read.**
  `ScanOptions::coalesced()` gives the parquet reader a span cache that reads every selected
  row group's projected column chunks as one byte span (split where unprojected columns
  leave a gap over 1 MB), serves the page reads from memory, prefetches the next span on a
  helper thread and releases a span once its column chunks are read. The batches are
  identical to the plain reader's. Nothing uses it by default; it is for wide full scans on
  spinning storage, where the page-at-a-time reader is seek-bound (docs/03 "Sequential
  row-group reads").
- **Capped writers cut data pages by size, not every 20,000 rows.** A scalar column of a
  65,536-row group is one page instead of four, so the plain reader, which seeks once per
  page, reads a wide table with a quarter of the seeks: on the AIF artifacts features went
  from 2,003 to 1,039 data pages and psms_competed from 1,592 to 399, at +0.5% bytes, and
  the writer's in-progress buffer of one 131,072-row competed group from 552 to 620 MB.
  Values are unchanged; the bytes and content hashes of files from capped writers change,
  and uncapped writers are byte-identical to before (docs/03 "Page layout of capped
  writers").
- **Capped writers plan their float encodings from their first rows.** A writer with a
  row-group cap holds its first quarter row group, writes every float leaf whose sampled
  values are more than 80% distinct PLAIN from the first page (instead of paying for a
  dictionary prefix until the dictionary limit fills), and sizes the dictionary limit of the
  other float leaves from their values per row group rather than their rows, which gives the
  chromatogram traces back the dictionary the row-sized limit cut at 128 KB. Against the
  unplanned layout on the AIF artifacts: features -8.6%, psms_competed -14.9%,
  chromatograms -6.3%, spectra -0.2%, and the competed rewrite encodes in 0.54 s against
  1.31 s with half the writer buffer. Values are unchanged; bytes and content hashes of
  capped writers change; `MUMDIA_PARQUET_PLAN=0` restores the unplanned layout (docs/03
  "Float encodings planned from the first rows").
- **The chromatogram `rt` axis is written PLAIN.** Each fragment row of a candidate repeats
  the candidate's axis, which snappy shortens in PLAIN form and cannot find in bit-packed
  dictionary indices, so the AIF chromatograms are 12.0% smaller than with the planned
  dictionary (160.9 against 182.8 MB) with identical values. Writers can name such columns
  with `WriteOptions::plain_column` (docs/09 "Output: chromatograms").
- **Parquet columns are encoded in parallel.** Every writer encodes a row group's columns
  concurrently on a dedicated codec pool (at most 8 threads, `--threads` when it is lower,
  serial at `--threads 1` or `MUMDIA_PARQUET_THREADS=1`) and appends them in schema order,
  which writes the serial writer's file byte for byte. The codec pool is a second pool
  beside the global one: `--threads N` now bounds the global pool at N and the codec pool
  at min(N, 8), and the two can be busy at once, so a run can keep up to N + min(N, 8)
  threads busy; `MUMDIA_PARQUET_THREADS=1` restores the previous bound. Encoding the AIF features table took
  0.14 s on 8 threads against 0.45 s, the competed table 0.16 against 0.42 s, the
  chromatograms 4.4 against 8.1 s. Writers called from inside a rayon pool keep encoding on
  their own thread, and so does a writer that finds as many callers already waiting on the
  pool as it has threads, so concurrent band writers under `groups.parallel` never have less
  than a thread each (docs/03 "Parallel column codec").
- **Multi-column scans decode their columns in parallel.** `TableFile::scan` and
  `TableFile::batches` split the projection into contiguous column groups, one reader each,
  decode the groups of every batch on the codec pool and join them column-wise; the batches
  are the single reader's exactly. Automatic by default (up to 8 groups, at least 4 MB of
  data each, the single reader from inside a rayon pool and for a coalesced scan, which
  keeps its one forward read per row group); `ScanOptions::decode_threads` sets it, and
  `MUMDIA_PARQUET_DECODE_THREADS=1` keeps every scan of a process on one reader. A full scan of the AIF features table went from 1.18 to 0.51 s, the competed
  table from 1.22 to 0.45 s, the chromatograms from 5.1 to 3.3 s (docs/03 "Parallel
  decode").

- **Library writers emit fragment tables sorted by `candidate_id`.** `import_diann_lib.py`,
  `make_reverse_decoys.py` and `make_shift_decoys.py` finish with a streaming bucket sort
  (`_lib_io.sort_fragments_by_candidate`: partition by candidate-id range into temporary
  files, sort each bucket in memory, append), so the parquet row-group statistics of
  `candidate_id` are monotonic and the engine's range load can read one candidate range of
  the table without scanning it (`Library::load_range_with`, the isolation-window-group
  search). Stable, so each candidate's fragments keep their stored order; the engine never
  depended on the previous order. `scripts/sort_fragments.py` applies the same rewrite to a
  table written before this change; the engine still loads an unsorted table through a
  filtered scan, with a warning.
- **The generated config reference cites functions, not line numbers.**
  `ci/gen_config_reference.py` cited every environment read as `path:line` and every
  config struct and enum by its line, and `configs/config-schema.json` carried a
  `source_line` per setting. Any merge that moved lines in `rescore.rs`, `config.rs`,
  `main.rs` or a sidecar script therefore made both files stale on every other open pull
  request, although no variable, field or default had changed. A read is now cited as
  `path::function` (`Type::method`, `Trait::method`, `module::function` and
  `outer::inner` in Rust, `Class.method` in Python, `<module>` outside any function), a
  struct or enum by its name, and the schema field
  `source_line` is replaced by `source_struct`, the declaring struct (the desktop editor
  never read either). `--check` also regenerates from copies of every input with blank
  lines inserted and fails if either artifact differs;
  `tests/python/test_gen_config_reference.py` pins the same property on synthetic
  sources. Two reads in one function now share one citation, so the `Read at` column
  has fewer entries. In the list of reads whose name is not a literal, such reads share
  one entry that gives their number, and the header still counts reads. The variables,
  their defaults, the fields and the settings are unchanged.
  The generator also skips a `#[cfg(test)]` element by the element's own extent
  (`cfg_test_item_end`), not by counting braces to the next balanced `}`. The old count
  ran past an attribute on something without a body: on an enum variant it took the
  enum's closing brace and the next item's header, so a source with a test-only variant
  was rejected as unbalanced, and on a statement (`inject(Fault::Publish)?;`) it silently
  blanked real code up to the end of the next block. The reference generated from main is
  unchanged by this.

### Performance

- **Rescore's feature stream and its post-classifier tail do less.** The feature stream
  and compete's pass-through copy can read each row group's projected column chunks in
  one sequential read (the span cache, `MUMDIA_WIDE_SCAN=coalesced`), and the feature
  stream can decode one row group a batch (`MUMDIA_WIDE_SCAN=rowgroup`). Both are opt-in
  for seek-bound storage, where neither is measured yet: the plain reader stays the
  default, the faster one from the page cache (1.54 s against 2.50 s coalesced on the HYE
  competed table). The parquet handoff is staged column by column
  from the decoded batches, with no row-major round trip: the streamed handoff of the HYE
  competed table (879,018 x 387) went from 8.4-8.7 s to 3.0-4.1 s of process wall, to the
  same bytes. The top-K collapse map is skipped when no candidate repeats (a 9.1 GB
  transient on the immunopeptidomics pool), the q columns are computed in place and per
  source by slice, and a grouped run whose bands' library row spans are disjoint skips
  the overlap dedup. Each run logs `rescore: phase timings` (the metadata pass, the
  feature stream and its encode, the classifier, the tail) and `rescore: sidecar
  timings`. `psms_scored.parquet` is byte-identical.
- **Both TSV reports read the scored table in two passes.** The rows that can be printed
  come from `label`, the transfer flag and the q columns, 3 bytes a row; the printed
  columns are then read for those rows only. The one-pass read held four string columns
  for every row, an estimated 55 GB on the 258.75M-row pooled table, to print about 10^5
  rows. Both TSVs are byte-identical (compared against the one-pass code in the tests).
- **The MBR worker keeps only the confident candidates.** Its apex maps, metadata and
  selected-peak lookup hold only candidates that are a confident target somewhere, and the
  transfers are flagged by one sorted-key lookup instead of a Python loop over every scored
  row: an estimated 100 GB of worker memory and 4-6 minutes on the pooled
  immunopeptidomics experiment with MBR on. Every output is byte-identical to the worker
  before the change, which the tests run from git history.

- **The `nn_torch` worker spends less time outside training, with byte-identical
  scores.** The parquet load decodes the next row group on a reader thread
  (`pre_buffer=True`) while up to 8 threads write the current one straight into the
  matrix, keeping the float64 moment partition and order (1,000,000 x 387: 11.8 s to
  2.4 s); the init feature scan counts its columns on a thread pool (400,000 x 120:
  24.9 s to 4.3 s), with at most 1 GiB of sort transients in flight
  (`MUMDIA_NN_SCAN_MEM_GB`) when the init sample escalates toward the whole fold;
  scoring batches are gathered with `torch.index_select` into one reused
  numpy-allocated buffer (4.7x on the gather; identity checked on Windows x86-64 with
  torch 2.6, CPU and CUDA, not yet on the Linux fleet); each round's positives come from a certified top window instead of a
  full stable sort (10M scores: 1.69 s to 0.08 s), and the decoy order of the hybrid cap
  from one uint64 key sort; the training pool is no longer scored after the last round,
  whose scores fed only a log line. Every change keeps a switch back to the code it
  replaced (`docs/13_sidecars.md`). Tests assert equal score bytes with all of them set
  back and against the worker before the change (extracted from git history), for the
  in-memory, streaming and TSV paths. The worker also prints read, fill, standardise
  and selection sub-timers, and removes its memmap after a failed run as well.

### Added

- **Opt-in concurrent fold training for `nn_torch` (`MUMDIA_NN_PARALLEL=K`).** The
  (seed, fold) tasks train in K spawned processes at a fixed per-process thread count
  (`MUMDIA_NN_PARALLEL_THREADS`), sharing the matrix through a read-only memmap. The
  epoch shuffle is then keyed per (seed, fold, iteration, epoch), which changes the scores
  once, like a seed change; they do not depend on K. Off by default; validate it as a seed
  change (three seeds, two pools, entrapment) before relying on it.

- **`psms_scored.parquet.report.json` records the NN worker's inherited environment.**
  When `nn_torch` ran, `params.nn_env` lists every `MUMDIA_NN_*` variable the worker
  inherited beyond the ones the engine sets. `MUMDIA_NN_SEED`, `MUMDIA_NN_THREADS` and
  `MUMDIA_NN_PARALLEL` change the scores and reach the worker only this way, so two runs
  of one configuration that differ in them are now told apart by the report.

- **`mumdia pool` pools a grouped run's band artifacts from the command line.** `run` does
  this itself at the end of a grouped search; standalone it is for the case where the
  search finished and the run did not, which now costs a pool rather than a re-search.
- **`groups.window_groups` searches a run one isolation-window group at a time.** The
  run's windows are cut, in ascending m/z, into contiguous groups whose library bands hold
  about the same number of precursors (planned from the precursor table's row-group
  statistics, no table read). Each group's band is written as a precursor table of its own
  and searched as a library of its own: seed, RT calibration windows, extract, features and
  compete see one band's precursors, fragments (by id range from the shared fragment table)
  and accepted rows, so the search's memory is one band's worth rather than the library's.
  A `seed-pool` stage then puts the bands' seeds on one q scale with library-wide ids and
  combines their mass calibrations by calibrant count; under `groups.calibration = global`
  (the default) every band's RT model and windows are fitted on the pooled anchors, under
  `per_group` on its own. A `pool` stage rewrites the band tables with library-wide ids,
  keeps one row per candidate where overlapping windows searched it twice (the higher
  `prelim_score`), and writes the standard `psms_extracted`, `chromatograms`, `features`
  and `psms_competed`, so rescore, quant and report run unchanged and the manifest, the
  run-level `cal.json` and the per-artifact reports keep their shape (band artifacts are
  recorded as `name[gNN]`, under `groups/gNN/`). Measured on the CI fixture with three and
  five groups against the ungrouped run: 151 against 150 stripped peptides, 149 shared,
  every smoke assertion passing and the grouped run byte-identical on repetition.
  At real scale (AT10234AUH against the 142.7M-precursor 8-12-mer immunopeptidomics
  library, eight groups, measured with a zero-code harness) the grouped extraction with the
  run-wide calibration reproduced the monolithic extraction row for row (22,850,003
  candidates, identical apex retention times) at 22-96 GB per group against 173 GB, and the
  pooled rescore gave the same 11,271 precursors at 1% (10,346 peptides against 10,213,
  within seed spread) in 107 min against 329 min for the same stages. Per-group calibration
  failed on the same file: three of seven groups had no confident seed on their own q scale
  and ran unbounded, one aborting at 664 GB.
  `docs/33_window_groups.md` has the layout, the semantics of the two calibration modes,
  and what a band cannot see. Groups run one after another in one process; child-process
  parallelism and `run-experiment` support are the next steps. A grouped run refuses
  `rt_im_train.finetune_deeplc`, which would train a different model per group. A sweep of
  the group count on the same file (48, 64, 96 and 192 asked for, one per host) accepted
  22,850,003 PSMs in every arm, the monolithic count exactly; it also showed that a band
  cannot be smaller than one isolation window (114 here, so 192 became 94 bands), that the
  largest band falls only from 44 to 23 GB as bands shrink, because each pays a fixed cost,
  and that CPU roughly doubles past about 64 bands.
- **`Library::load_range_with` loads one precursor m/z band of a library.** The precursor
  table is m/z-sorted with row-aligned ids, so a band is a row span: it is found from the
  parquet row-group statistics plus one decode of `precursor_mz` over the boundary groups,
  and `TableFile::open_rows` then reads only the row groups that cover it, trimmed by a row
  selection. Fragments come the same way when their table is sorted by `candidate_id` at
  row-group granularity; an unsorted table still loads through a filtered scan, with a
  warning. The slice carries local ids `0..n` and `Library::global_offset`, the file row of
  local id 0. This is the load an isolation-window-group search needs: a group of windows
  can only select precursors in its band, so a run searched group by group never holds the
  rest of the library. `TableFile::row_group_stats` exposes the footer statistics for
  planning such reads.
### Changed

- **`mumdia doctor` and the interpreter resolver say what a missing DeepLC costs.** The
  multi-head retention-time calibration is the default whenever a DeepLC interpreter is
  configured or discovered (`predict_frag.deeplc_python` absent or `"auto"`: `MUMDIA_PYTHON_DEEPLC`,
  `CONDA_PREFIX`, `VIRTUAL_ENV`, then `python3`/`python` on `PATH`), and a machine without one
  runs on the imported iRT. The note printed in that case now names the calibration and its
  measured value (+4.8% peptides on AIF, +14.3% on Astral) instead of only "keeps the imported
  iRT", so the loss is visible where the decision is made.
### Added

- **Helpers for very large predicted libraries** and `docs/32_large_libraries.md`, from the
  immunopeptidomics case study (59M and 203M precursors on one Astral run):
  `scripts/mz_range_survivors.py` (candidates inside the run's isolation range),
  `scripts/assemble_survivors.py` (survivors -> renumbered library, target/decoy pair kept
  together on `peptidoform_id`), `scripts/shard_parquet.py` (row-group-aligned split and
  concatenate) and `scripts/mh_shard_predict.py` (deduplicated, sharded multi-head DeepLC
  calibration: 125.9M unique sequences in 42 minutes over 12 CPU shards instead of 6 hours in
  one process). Measured yields and costs are in the document, including the seven-file
  orchestrated first pass and the second pass from the union of first-pass identifications
  (17,829 peptides pooled at 1%, 94-100% of DIA-NN's empirical-library second pass per file)
  and a measurement of what the sequence-tag screen can and cannot prune on DIA
  immunopeptidomics data, including the negative result for predicted-intensity-weighted
  tags.


### Fixed

- **`extract --restrict-candidates` now runs on the streaming path.** A candidate allowlist
  routed extract to the serial path, whose whole-run hit accumulator ignores
  `extract.windows_in_flight`, so a prescan-restricted extract had the memory profile of
  the pre-streaming engine: measured on an immunopeptidomics library, 35M allowed
  candidates took 2:08 h at a 325 GB peak, and 83M candidates aborted with
  `memory allocation of 12288 bytes failed` at 471 GB on a host with a 1 TB commit limit.
  The allowlist is now applied inside the streaming probe at the point the serial path
  applies it, before the peak claim, so a listed candidate collects the same hits and an
  unlisted one neither collects hits nor competes for a shared peak; the serial path is
  unchanged and still serves the two-pass peak-claim strategies.
- **`nn_torch` rescoring no longer aborts on a pool that is overwhelmingly false.** The
  worker picks its initial ranking feature on a 300k-row sample of the training fold and
  then requires at least one target at the training FDR. On an 8.07M-PSM immunopeptidomics
  pool (34.9M candidates screened, a few thousand true) a 300k sample was 3.7% of the rows
  and held too few true PSMs for any of 347 features to reach 1%, so the scan returned an
  arbitrary feature with 0 targets and every fold aborted with `selected no positive
  targets at training FDR 0.01`, while the same features gave 150-217 targets at 1% on a
  2M-PSM pool of the same run. Two changes: when no feature passes on the sample the scan
  is repeated on 4x the rows up to the whole fold, and when the chosen feature still
  selects no positive over the whole fold the first (and only the first) selection is
  loosened in steps to `MUMDIA_NN_INIT_FDR_MAX` (default 0.05; 0 restores the hard error).
  Later iterations re-select at the training FDR on the model's own scores as before, so a
  pool where the init already works is unchanged.
- **`scripts/_lib_io.py` writes precursor tables of any size.** `Table.from_pandas` converts
  a column as one arrow array, and a `string` array holds at most 2 GiB of characters, so on
  a 285M-row precursor table (the 8-12-mer immunopeptidomics library with reverse decoys;
  `peptidoform` alone is 4.5 GB) pandas 3 / pyarrow 25 produced a `large_string` array that
  the engine-encoding cast refused (`Failed casting from large_string to string: input array
  too large`), and `make_reverse_decoys.py` failed at its final write after 70 minutes. The
  frame is now converted in 4M-row slices into a chunked table; parquet writes the chunks as
  row groups, which is how the engine reads them.
- **`scripts/import_diann_lib.py` fragment m/z cardinality is bit-identical to the
  whole-table importer.** The streaming importer binned `Product.Mz` in float64 where the
  original rounded the float32 column, and it counted a bin twice when one precursor's
  fragments straddled a parquet row-group boundary. Bins are computed in float32 and the last
  precursor's bins are carried into the next row group; measured identical `cardinality` on
  the 29.5M-precursor 9-mer library.


### Performance

- **`scripts/import_diann_lib.py` streams the DIA-NN library** instead of reading the
  whole fragment-level parquet into pandas. Two passes over the row groups: the first
  collects the precursor table, the fragment counts and the fragment m/z cardinality, the
  second writes the fragment table one row group at a time. Output contract unchanged
  (same candidate ids, same columns, same types; the 11 contract tests pass), fragments are
  no longer globally sorted by `candidate_id`, which the engine never required. Measured on
  a 29.5M-precursor, 354M-row immunopeptidomics library: 19:54 at a 15.0 GB peak against
  21:18 at 202.6 GB before, output identical column for column (cardinality included); the 142.7M-precursor, 1.69-billion-row 8-12-mer library of the
  same set could not be imported at all before (about a terabyte in pandas).
- **`scripts/make_reverse_decoys.py` streams the fragment table** and computes the decoy
  fragment m/z from per-decoy cumulative residue masses with array lookups instead of a
  Python loop over every fragment row. Same decoys (same reversal, same seeded scramble, same
  collision rules, same mass model; the 18 decoy-builder tests pass). Measured on the same 29.5M-precursor library: 18:20 at a 59.6 GB peak against 59:00 at 321.4 GB before, identical decoy statistics (26,438 collisions, 22,242 resolved by scramble, 4,196 pairs dropped, 58,996,242 precursors, 707,582,432 fragment rows).



### Added

- `predict_frag.predictor = "peptdeep"`: AlphaPeptDeep fragment intensities, a third
  option beside `native` and `ms2pip`. With `rt_predictor = "deeplc"` this is a
  library built end to end from a FASTA with open predictors and no DIA-NN anywhere,
  which also means nothing re-predicts or replaces its retention times later:
  `rt_im_train.library_irt` applies to imported libraries only, and a library you
  built already is the DeepLC prediction. The per-run LOESS calibration,
  `finetune_deeplc` and `multihead_calibration` all still apply.

  New settings: `peptdeep_python`, `peptdeep_model` (default `generic`),
  `peptdeep_nce` (default 30.0) and `peptdeep_instrument` (default `Lumos`).
  Collision energy and instrument change the predicted spectrum, so both are part of
  the artifact's `model_identity` (`peptdeep-1.5.1-generic-nce30-Lumos`): two
  libraries built at different NCE are not the same library. The device comes from
  `MUMDIA_PEPTDEEP_DEVICE` (auto|cuda|cpu), as the rescorer takes `MUMDIA_NN_DEVICE`.

  Opt-in and benchmark-gated. No entrapment or second-acquisition measurement exists
  for it yet, and a seed-PSM count alone does not promote a default.

- The desktop Setup screen has a **Managed data** card. Everything the application
  downloads or builds is written at runtime under `%LOCALAPPDATA%\MuMDIA` (or
  `~/.local/share/MuMDIA`), and an installer removes only what it placed under the
  program folder, so an uninstall left all of it behind: 8.9 GB on one development
  machine, with nothing in the interface that could remove it. The card lists each
  item with its size and removes it in two clicks, naming the exact paths first.
  Removal is refused while a search, an installation or a library build is running.

  Deliberately not done as an uninstaller action: an upgrade reuses the same data
  directory, and an MSI uninstall also runs during some upgrade paths, so a silent
  delete would throw away a spectral library that costs hours to predict as a side
  effect of a version change.

### Changed

- **`rescore.train_neg_ratio` defaults to 2 instead of 3.** Measured with three seeds per
  pool against 3: Astral six-run pool 116,711 against 116,309 peptides (+0.35%), HYE B01
  63,096 against 63,004 (+0.15%), AIF entrapment library +0.56% real peptides at an
  empirical FDP of 1.025% against 1.044% with the decoy fraction unchanged, at 16% less
  rescore wall. 1.5 and 1 were measured too: 1 buys its count with a looser null (FDP
  1.099%). A config that sets the ratio explicitly is unaffected.
- Measured and recorded rather than changed (CLAUDE.md "Rescore cost"): `folds: 2` is
  -45% rescore wall at -0.06% peptides on the large pool but -1.1% on HYE B01;
  `train_subsample: 0.5` -33% at -0.3% / -0.9%; batch 16384 -20% at +0.07% / -0.6%;
  `folds: 5` +0.09% at twice the wall; `folds: 1` (in-sample scoring) is refuted by
  entrapment, +2.8% real peptides at an FDP of 1.42% against 1.00%. A single seed's count
  moves by up to 0.4% under any change of arithmetic (thread count, CPU generation, column
  set), so these are means over seeds on two pools.
- **`experiment.finetune_scope` is now `experiment.rt_library_scope`, and it governs
  multi-head calibration as well as the DeepLC fine-tune.** The old name still parses.

  Multi-head re-predicted the whole library once per run, unconditionally. On a six-file
  Astral experiment that is six 362 MB re-predictions of the same 9.4M-row library, and
  it is where the measured 1.4x-1.7x wall clock goes. The reasoning was that the
  calibration is fitted against THIS run's chromatography -- true, and equally true of
  the fine-tune, which has been shareable all along. What a shared library fixes is
  elution ORDER, which replicate injections on one LC method share; each run still fits
  its own LOESS on top, and that per-run fit is what absorbs drift.

  Under the default `first_run_only` the first run adapts the library and the rest reuse
  it. Set `per_run` for a batch that genuinely reorders -- different gradients or
  columns, a method change part-way -- or a long batch where drift accumulates; the
  Measured on the six-file Astral experiment: sharing costs 0.97% of precursors and
  1.05% of peptides (128,006 -> 126,762 and 115,924 -> 114,704), gains 0.70% of protein
  groups, and leaves the empirical decoy fraction at 0.0100, so the 1% is lost
  identifications rather than a moved threshold. It gives back 55% of the window
  narrowing multi-head's gain came from -- the five reusing runs go from a mean `w_rt`
  of 22.0 s to 34.1 s -- and in exchange removes five full re-predictions of a 9.4M-row
  library, finishing those files' chains in 1.9-3.4 minutes each.

  `first_run_only` stays the default on that trade. Set `per_run` when the last percent
  matters more than the hours, when the runs do not share an elution order, or on a long
  batch where drift accumulates.

- **`rt_im_train.multihead_calibration` is now the default.** Left unset it calibrates the
  DeepLC base model over 80 of its best-correlating LC-setup heads against each run's own
  confident seed PSMs, in place of the single head `deeplc.predict` returns. Measured on
  two acquisitions, six pooled runs each, at an unchanged empirical decoy fraction of
  0.0100 in all four arms: AIF 80,842 -> 84,725 peptides (+4.8%), Astral 102,942 ->
  117,652 (+14.3%), protein groups +2.4% and +7.1%, precursors +4.8% and +15.5%. Fewer
  candidates reach rescore and more of them are real. An entrapment arm supplies the
  empirical null: +4.28% real peptides at an FDP of 0.995% against the baseline's 0.995%,
  identical to three decimal places on 138 and 144 accepted spike-in peptides.

  It costs **1.4x to 1.7x wall clock**, because the calibration is fitted against each
  run's own anchors and so cannot be shared across an experiment. Set
  `multihead_calibration: 0` to restore the previous behaviour exactly.

  The default is scoped rather than unconditional, and the field is now nullable to
  express that. It applies only where an interpreter is available AND the run's retention
  times are DeepLC's already: an imported library under `library_irt = auto` or `deeplc`,
  or FASTA with `rt_predictor = deeplc`. A native, Python-free run is unaffected and still
  starts; a configuration asking for the native retention-time model keeps it. An explicit
  count is a hard requirement, so `multihead_calibration: 80` without an interpreter
  still fails rather than doing without. `finetune_deeplc` keeps its own slot instead of
  colliding with the default, so a configuration that enables the fine-tune is no longer
  refused for a conflict it never wrote; asking for both explicitly is still an error.

  Two consequences worth knowing: the separate `library_irt` base-model re-prediction no
  longer runs under the defaults, because multi-head re-predicts the library itself (about
  27 minutes saved on a 10.9M-row library, whose only output it would overwrite); and
  `model_identities.rt_predictor` now reads `multihead-80` rather than
  `deeplc-4.4.0-base`, which is how to tell which one a given run used.

### Performance

- **The pooled `nn_torch` rescore holds about half the memory it did.** Under
  `rescore.strict` with a sidecar classifier the engine releases its own feature matrix
  as soon as the handoff table is written (strict has no native fallback that could still
  read it); the handoff parquet is written in 131,072-row groups; and the worker reads it
  one row group at a time, because pyarrow's read-ahead had buffered a second copy of the
  matrix. Six-run Astral pool (3,133,636 PSMs x 387 features): process-tree peak 17.9 ->
  9.3 GB; HYE B01 (1,838,344 PSMs): 12.0 -> 5.05 GB; identical identifications and
  unchanged wall on the fleet. The disk-backed memmap stays the last resort.
- **Intel desktops rescore at fleet speed.** The same six-run pool took 118 minutes on an
  i9-13900KS against 19 on an EPYC 9354, with identical single-threaded epoch speed. Two
  causes, both removed. The worker ran one OpenMP thread per logical CPU, and on a hybrid
  (P + E core) CPU every parallel op waits for its slowest thread; torch threads are now
  capped at the performance-core count on Windows or 16 elsewhere, the engine's
  `--threads` being an upper bound (`MUMDIA_NN_THREAD_CAP`). And the trained network
  accumulated subnormal float32 values, which Intel cores handle through microcode assists
  at about a hundred times the cost of a normal multiply-add: Adam with L2 decay shrinks
  every parameter that receives no data gradient geometrically until it crosses 1.2e-38
  (the constant features' weights, dead units, their BatchNorm buffers, and Adam's own
  moments). The worker now drops constant feature columns from the parquet footer's
  statistics (`MUMDIA_NN_DROP_CONSTANT`), zeroes every parameter, buffer and optimizer
  moment below 1e-20 once per epoch (`MUMDIA_NN_CLAMP_TINY`), and flushes subnormals to
  zero as a second layer (`MUMDIA_NN_FLUSH_DENORMAL`). Desktop, six-run pool: 118 -> 74
  minutes from the thread cap alone, 11.4 with everything, 11.9 with flush-to-zero off and
  the clamps alone; 4.6 minutes on an RTX 4090 with `MUMDIA_NN_DEVICE=cuda`. Paired on one
  host, the clamps and the flush move a single seed's count by less than the seed-to-seed
  spread and are bit-identical on HYE B01. `MUMDIA_NN_DEBUG_DENORMALS=1` prints a per-round
  census.

### Fixed

- Under multi-head calibration both orchestrators warned that no
  `predict_frag.deeplc_python` was configured when the base-model re-prediction of the
  library iRT was skipped; the real reason was that the calibration re-predicts the library
  itself on the first run. The warning is now reserved for a missing interpreter and the
  multi-head case is logged as what it is.
- The `nn_torch` rescorer fell to its disk-backed memmap far too eagerly.
  `MUMDIA_NN_STREAM_GB` was a fixed 4 GB, so a 4.52 GiB feature matrix on a 96 GiB
  machine crossed it by 13% and took a path measured at about 9x slower: **166 minutes
  against roughly 20**. Unset, the threshold is now twice free physical memory, never
  below the historical 4 GB, so the memmap is a last resort rather than a safety margin
  and a matrix that merely overflows RAM is paged by the operating system instead --
  much cheaper for this access pattern. Setting `MUMDIA_NN_STREAM_GB` still overrides it
  exactly as before, which is what a machine with no page file wants.

  The worker already logged which backend it chose; it now also says where the threshold
  came from, and warns explicitly when the slow path is taken for want of memory.
- The desktop progress ladder stopped moving after the first file of a multi-file
  experiment. Stages were aggregated by NAME across the whole output tree, so once file
  1 had reached the last stage, file 2's `convert` could not pull "the furthest stage
  seen" backwards, and the remaining files ran with the display frozen on the first
  file's finish. The row and time figures were a sum across files as well, describing no
  file in particular: on a six-file Astral experiment `extract` read 87.8M rows from 12
  artifacts. The run snapshot now carries per-file progress (`runs`) and the pooled tail
  (`root_stages`) separately, and the display follows the file that is actually running
  and says "file k of n".
- The desktop application's run log, the terminal and any redirected log file were
  almost entirely blank lines whenever retention times were predicted. DeepLC's
  progress writer emits a bare carriage return per update, which renders as nothing
  when stdout is not a terminal, and the engine inherits a worker's stdout rather than
  capturing it (deliberately, so long-running progress reaches the user live). Measured:
  5,697 blank lines from one 2.9M-peptide prediction, and 98-99% of two real run logs.
  Both DeepLC workers now filter their own stdout, dropping only what is empty once
  carriage returns and whitespace are stripped; everything DeepLC actually says still
  comes through, in order, and stderr is untouched. `MUMDIA_DEEPLC_RAW_OUTPUT=1`
  restores the unfiltered output for debugging.

  This affected every run that predicts retention times, which since multi-head
  calibration became the default is every run with a DeepLC interpreter.

### Security

- `rustls` 0.23.43 -> 0.23.45 in the desktop application, closing RUSTSEC-2026-0285
  (TLS 1.3 handshake messages incorrectly accepted across encryption level boundaries,
  medium, 5.3). It reaches the application through `ureq`, which is what downloads
  DIA-NN 1.8.1 and ThermoRawFileParser, so it is on a path that fetches executables.
  Lock-only; no manifest constraint changed.

## [0.3.1] - 2026-09-10

Fixes a v0.3.0 desktop installer that could not create its Python environment at all.

### Fixed

- The desktop application's "installing the analysis packages" step failed on every
  machine with `Because only setuptools<=78.1.0 is available and you require
  setuptools>=83, we can conclude that your requirements are unsatisfiable`.
  `env/console-requirements.txt` adds the PyTorch CPU index for `torch==2.14.0+cpu`, uv
  gives an `--extra-index-url` priority over PyPI, and that index also carries an old
  vendored `setuptools`, so under uv's default `first-index` strategy the resolver never
  reached PyPI's setuptools 84 and the `setuptools>=83` pin that closes PYSEC-2026-3447
  made the environment unresolvable. The installer passes
  `--index-strategy unsafe-best-match`, which is what uv's own PyTorch documentation
  prescribes and which both first-party indexes already justify; the environment resolves
  to setuptools 84.0.0 from PyPI and torch 2.14.0+cpu from the PyTorch index. Verified by
  building the environment end to end, not only by resolving it.
- `ci/check_console_envs.py` resolves both `env/console-*.txt` files with uv and asserts
  the installer still passes that flag. Nothing in CI read those files before, which is
  how a broken installer shipped: the conda specifications beside them are exercised by
  the sidecar-import jobs, but the desktop application installs neither of those.

## [0.3.0] - 2026-09-09

Multi-head retention-time calibration and the DeepLC 4.4.0 floor it needs, plus
the sha2 0.11 bump. `rt_im_train.multihead_calibration` is off by default;
turning it on measured +4.8% stripped peptides on the HYE AIF benchmark and
+14.3% on HYE Astral, at an unchanged empirical decoy fraction.

### Added

- `rt_im_train.multihead_calibration` (default 0, off): calibrate the DeepLC base model
  against this run's confident seed PSMs across its N best-correlating LC-setup heads,
  instead of fine-tuning. `deeplc.predict` returns one of the model's 6,543 heads, the
  setup its `DEFAULT_TASK_NAME` names, on that setup's gradient; the per-run LOESS then
  maps that column onto observed retention time, and a smooth increasing curve can stretch
  and bend the axis but cannot reorder two peptides, so that one setup's elution order
  survives into the calibrated result. Different chromatography reorders peptides.
  `MultiHeadRidgeCalibration` ranks every head against the run's own anchors,
  spline-calibrates the best ones and ridge-combines them, and never fits more head
  weights than half the reference. It occupies the fine-tune's slot in the chain, needs
  the same anchors, and validation refuses both at once. Off by default: no entrapment or
  second-acquisition measurement exists for it in this repository yet. Measured on HYE
  B01's own anchors, split in half: the default head with the best possible monotone map
  scores 79.5 s median error, the multi-head combination 28.6 s. The monotone oracle,
  fitted on the scoring half itself, also scores 79.5 s, so on that run the LOESS is
  already doing everything a monotone map can and the whole gap is ordering rather than
  fit quality (`docs/08_rt_im_train.md` section 4d). The best head there is 1229, not the
  default 938. Measured end to end on both HYE acquisitions, four pooled six-file
  experiments from one binary differing only in this field: AIF 80,842 to 84,725 stripped
  peptides at 1% (+4.8%), Astral 102,942 to 117,652 (+14.3%), protein groups +2.4% and
  +7.1%, at an empirical decoy fraction of 0.0100 in all four arms. On Astral the window
  narrows from 84 s to about 20 s and half as many candidates reach the rescore, so the
  gain is interference removed rather than a loosened threshold. It costs 1.4x to 1.7x
  wall clock, because the calibration re-predicts the library per run rather than once per
  experiment. Still off by default: entrapment has not been run, and CLAUDE.md requires an
  empirical null as well as two acquisitions.

### Changed

- The DeepLC floor is 4.4.0, raised from 4.1.1, for
  `deeplc.calibration.MultiHeadRidgeCalibration` and the lazy head source that
  `predict_and_calibrate` uses to avoid materialising all 6,543 head columns. 4.1.1 has
  neither: its `calibrate` picks a single best-correlating head. `mumdia doctor`, the
  interpreter discovery, `sidecar::require_deeplc_version` and both worker scripts enforce
  it from the one constant, and every shipped environment pins `deeplc==4.4.0`.
- The desktop application builds against `sha2` 0.11. Its `finalize()` returns
  `hybrid_array::Array` rather than the old `GenericArray`, which does not implement
  `LowerHex`, so the four `format!("{:x}", ..)` sites move to a `components::hex`
  helper. The strings are unchanged, and a test pins them against the canonical
  SHA-256 vectors, because they are compared with published checksums and used as
  cache directory names.

### Fixed

- Nine desktop test fixtures used a fixed temporary directory and deleted it on entry, so
  two `cargo test` runs on one machine raced and one lost its files mid-test. They are
  unique per process, like every other temporary fixture in the workspace (`docs/14`). Hit
  twice while preparing releases; two concurrent runs now both pass.

## [0.2.0] - 2026-09-08

Three code reviews and their fixes (`docs/29_code_review_2026-09-07.md`,
`docs/30_code_review_2026-09-08.md`, `docs/31_code_review_2026-09-08_full.md`),
the MS2PIP charge-2 library path, and the desktop settings work. Validated on the
six-file HYE benchmark end to end: 80,803 experiment-wide stripped peptides at 1%
at a 1.00% decoy fraction.

### Added

- MS2PIP charge-2 fragment predictions reach the library. The worker emits a
  `frag_charge` column from the `b2`/`y2` series that the `*ch2` models (`HCDch2`,
  `CIDch2`) predict, and `predict-frag` keys its lookup on `(ion, ordinal, charge)`.
  When the model emitted charge-2 series, every fragment carries a model intensity on
  one scale; single-charge models keep the previous behaviour (native heuristic for
  charge-2 fragments, each charge group normalised to its own peak), so existing
  `HCD2021` libraries are unchanged. Why: on the HYE FASTA library built with `HCD2021`,
  78.6% of the top-6 fragments were charge-2 heuristics and the seed search separated
  targets from decoys no better than chance (41.6% decoys among the top 1,000 seed
  scores, against 0% with the DIA-NN library on the same spectra), so retention-time
  calibration found no anchors and the run proceeded with an unbounded RT window.
  Measured on that run with 12 fragments: `HCDch2` 19,308 confident seeds (0% decoys
  in the top 5,000; fragment charges 79% / 21%), `HCD2021` with charge-2 fragments
  only from precursor charge 3 14,412, the DIA-NN library 21,856.

- `peptides.tsv` and `proteins.tsv` (single-run and experiment-wide) carry
  `is_transferred` and `transfer_q`, the acceptance basis of a match-between-runs row:
  a transferred row keeps its grouped q next to the transfer q it was accepted at, a
  tighter report threshold does not revoke a transfer that passed `mbr.q_transfer`, and a
  protein group admitted through a transferred row carries the flag. The MBR worker's
  augmented scored table gains `transfer_q` for it (docs/29 #19).
- `experiment_manifest.json` records the resolved `config_json` next to its hash, the
  `model_identities` that produced the artifacts (RT source, fragment predictor, the
  classifier that actually ran, feature schema, MBR strategy), the configured and
  effective `quant.q_filter`, and input hashes taken at the start of the run rather than
  at its end (docs/29 #15).
- Dependabot covers the desktop application's Cargo dependencies (`/desktop`), and CI
  audits `desktop/Cargo.lock` with `cargo audit` next to the engine's lockfile, with one
  documented ignore (RUSTSEC-2024-0429: glib 0.18 through Tauri 2's gtk 0.18).

### Changed

- Model identities carry the installed predictor versions: `deeplc-4.1.1-base`,
  `deeplc-4.1.1-finetuned`, `ms2pip-4.2.0-HCDch2` in the library report and the manifests,
  in place of the family labels `deeplc-4.0-mt` and `ms2pip-<model>` (docs/30).
- With nothing to transfer, the MBR worker writes a transfer table with its ten columns and
  zero rows, and the requested augmented scored table with every row unflagged, instead of
  a one-column placeholder and no scored table (docs/30).
- The candidate-audit rejection code `NO_PEAK_GROUP` is `DID_NOT_SURVIVE_EXTRACTION`
  (`RejectionReason::DidNotSurviveExtraction`). The audit assigns it to every candidate
  with no extracted row, and `extract` does not write the per-candidate table that would
  separate presence, matched-fraction and gate failures, so the old name claimed a cause
  the audit cannot see (docs/29 #16). The audit table has no versioned schema; the
  metrics JSON gains `q_unit`.
- MS2PIP 4.2.0 in every shipped environment (`env/docker-rescore.yml`,
  `env/console-ms2pip-requirements.txt`), and `env/mumdia-deeplc.yml` now carries
  `ms2pip==4.2.0` too, so one host environment serves DeepLC, MS2PIP and the `nn_torch`
  rescorer and `configs/examples/fasta-sidecars.json` runs from the shipped
  specifications with its interpreters at `auto`. Before this no host specification
  provided MS2PIP at all. 4.0.0 needed `sqlalchemy<2` and could not share an
  environment with DeepLC; 4.2.0 is the version behind the FASTA-mode measurements in
  `docs/28` section 22. The image's smoke test imports `ms2pip` in the rescore
  environment.
- `predict_frag.ms2pip_model` defaults to `HCDch2` (was `HCD`), for the reason above.
  `configs/examples/fasta-sidecars.json` and the image's `config.dia.json` set
  `top_n_fragments: 12` explicitly, the count the DIA-NN library ships and the one the
  measurement used; the engine default stays 6, because the native predictor was not
  re-measured.
- FASTA mode measured end to end on both HYE acquisitions with that library
  (`docs/28_feature_selection_analysis.md` section 22): six pooled AIF runs 69,091
  experiment-wide peptides at 1% in 45 min of search (imported DIA-NN library: 72,344),
  six pooled Astral runs 85,644 in 34 min; library prediction 54 min once per FASTA
  (DeepLC 19, MS2PIP 35 at 32 processes). DIA-NN 2.2.0 library-free on the same files:
  61 and 67 min including its library.

- The MS2PIP worker uses the engine's thread count for its process pool (passed as a
  fourth argument by `run_ms2pip`) instead of capping itself at eight processes, and
  assembles its output from numpy arrays per chunk instead of four per-fragment Python
  lists. Measured on the 9.8M-peptidoform HYE library (one missed cleavage, 7-30, charges
  2-3, one oxidation), the old worker ran on six to eight cores of the 32 requested and
  held 13 GB of Python objects before writing. Predictions are unchanged: the same
  rows in the same order, the same float64 arithmetic stored as float32.

### Fixed

- Code review A, data integrity (`docs/29_code_review_2026-09-07.md`, findings 1, 2, 4,
  9, 11, 17, 18, 21):
  - quant's refusal of a pooled scored table read `source` as i32 while rescore writes
    it as u32, treated the type error as "no such column", and so never ran on the
    engine's own output; a pooled table quantified against one run's chromatograms
    produced one identical row per run. The column is now read in its declared type and
    a present column of another type is an error (#1).
  - The streamed library loader checked fragment `mz` and `predicted_intensity` for
    finiteness on the physical Arrow buffers, which ignore the validity bitmap, and then
    turned NULL cells into NaN, NULL names into `""` and a NULL `candidate_id` into
    candidate 0. Every required fragment column now rejects NULLs before its values are
    read, through the same contract the typed getters enforce, with a fixture per
    column (#2).
  - `AtomicPath` removed the destination before renaming, so a failed publication had
    already destroyed the previous artifact and readers saw a window with no file; two
    writers for one destination in one process shared a temporary name. The rename
    replaces in place on every platform, the temporary name carries a counter, and the
    failure case is tested (#4).
  - N-terminal methionine excision was skipped whenever the Met-retained peptide fell
    outside the length window, so an N-terminal peptide of `max_len + 1` residues yielded
    nothing although its excised form was in range. Both forms are judged on their own
    length (#9).
  - `rescore.max_feature_matrix_gib` was checked after the matrix had been filled,
    against an estimate of the old `Vec<Vec<f64>>` layout, so it could neither prevent the
    allocation nor describe it; it is now checked from the parquet footers and the
    selected feature count before allocation, on the flat f32 layout, with checked
    arithmetic (#11).
  - A candidate DeepLC or MS2PIP returned nothing for received a substitute (iRT 0.0, or
    the native intensities under an MS2PIP model identity). It is now dropped together
    with its paired decoy or target, the counts are in the library report and a warning,
    and a worker id that was not requested or appears twice is an error (#17).
  - Numeric configuration domains are validated at load: thresholds and fractions within
    their unit interval, positive multipliers and widths, ordered `min_len <= max_len` and
    `charge_min <= charge_max`, counts at least one, with documented zero meanings kept
    (#18). `quant.q_threshold = -0.1`, `rt_im_train.rt_window_multiplier = -1.0` and
    `rescore.train_margin_frac = 2.0` were accepted before.
  - `ci/gen_config_reference.py` and `ci/check_workflows.py` scan the files git tracks
    rather than everything on disk, so scratch copies beside the sources no longer enter
    the generated reference or the workflow check (#21).
- Code review B, workers (`docs/29`, findings 3, 6, 7, 8, 12, 20):
  - The entrapment worker skipped a training fold whose training side held one class
    and then scored that fold's held-out rows with the final model, trained on those
    very rows, so in-sample scores entered the entrapment FDR. A single-class training
    fold is now an error that names the condition; the final model scores the decoys
    only (#3).
  - The MBR worker printed an "empirical decoy fraction" over accepted transfers, a
    population that cannot contain a decoy, and computed the transfer q as
    `null / targets`, which is exactly 0 for any pool no permuted residual undercuts, so
    a three-candidate pool was accepted whole at 1%. The q uses the engine's `+1`
    pseudocount and the summary names the permuted-null draws inside the accepted window
    instead (#6, #7).
  - With `extract.retain_top_peaks` above 1 the MBR worker measured the transfer on the
    last competed peak of a candidate, not the one rescore selected and quant integrates;
    it now joins `selected_peak_rank` and falls back to the highest `prelim_score` peak
    (#8).
  - `augment_library.py` gave every added precursor a fresh `base_peptide_id`, so an
    added charge state or modform of an existing peptide left its peptide's competition
    group and fold; added forms of existing sequences keep the imported id (#12).
  - `bench/feature_selection/fs_lib.py` hashed the peptide with its `DECOY_` prefix for
    fold assignment, splitting pairs; it hashes the base sequence, and every benchmark
    row records the code revision, fold rule, feature count, seed and training recipe
    (#20).
- Code review C, desktop and output ownership (`docs/29`, findings 5, 13, 14):
  - A repeated Start in the desktop application could launch a second engine into the
    same results folder: the start flow had several awaits and no in-progress guard, and
    the backend launched every request. A Start is now refused while one is in progress
    or while the run the interface follows is still running, and the backend reserves a
    run's results folder (by canonical path) before spawning the engine and releases it
    when the run's end is published, so a request for an active folder is refused with
    the owning run named (#5).
  - `run-experiment --run-names` compared names case-sensitively, so `RunA` and `runa`
    passed and addressed one directory on Windows, macOS and most network shares. Names
    that differ only in case are rejected on every platform (#5).
  - Desktop preflight asked the engine about converters without the request's
    configuration, so a converter named in `convert.thermo_raw_parser` or
    `convert.msconvert` was reported missing and the search refused, and it required
    ThermoRawFileParser for Thermo `.raw` even when msconvert, the engine's own fallback
    for a parser left at `auto`, was present. The probe now carries the configuration and
    the verdict follows the engine's rule: only msconvert present runs, with a note; an
    explicitly configured parser that is missing blocks, as it errors in the engine (#13).
  - A cancelled desktop run could be published as failed. `cancel` and the process
    waiter both wrote the terminal status and whichever ran second won, while the
    cancellation flag was written and never read. The waiter is now the only writer: it
    reads the intent after reaping the engine and publishes `cancelled`, `done` when the
    engine had already finished, or `failed`; until then the run shows "Stopping" (#14).
- The committed CycloneDX SBOM, which ships in every release archive, referenced
  `pkg:cargo/mumdia-core` and `pkg:cargo/mumdia-io` in its dependency graph while
  excluding them from its component list, so it failed validation and `--check`
  regenerated the same broken document. Only the application crate is excluded now,
  because `metadata.component` describes it (`docs/31` F11).
- The desktop DIA-NN cache-key test used a fixed temporary directory and deleted it on
  entry, so two `cargo test` runs on one machine raced and one lost its fixture
  mid-test. It is unique per process, like every other temporary fixture in the
  workspace (`docs/14`).
- Code review F, whole-repository review (`docs/31_code_review_2026-09-08_full.md`,
  F1 to F10):
  - `prescan` read the infinite-bounds sentinel that `rt-im-train` writes for "calibration
    unavailable, search the whole gradient" as "cannot be screened" and dropped the
    candidate, so a run with no confident seeds discarded the entire library and exited 0
    with a zero-row survivors table. An unbounded window now screens over the whole
    gradient, a candidate with no window row is treated the same, both are counted, and
    screening every candidate away is an error (F1).
  - A present-but-wrong-typed `is_transferred` was swallowed as "no transfers", silently
    removing every match-between-runs identification from `peptides.tsv` and
    `proteins.tsv` while the parquet still carried them. Present columns are read in their
    declared type and a mismatch is an error; only an absent column falls back (F2).
  - `sidecar::resolve_script` tried the working directory before the directory beside the
    binary, the ordering `python::resolve_script_dir` was hardened against, so a `scripts/`
    directory inside an untrusted dataset could have its worker executed. An absolute
    directory is taken as given, then the executable's directory, then `<exe>/scripts`, and
    the working directory last (F3).
  - `Loess::predict` indexed before the start of its grid for a non-finite query, so one
    library row with a null `predicted_irt` could abort or misread memory at rt-im-train.
    It returns NaN, and rt-im-train treats a non-finite library iRT as "no calibrated RT"
    and counts those rows (F4).
  - The rescorer's in-memory TSV backend standardised with median/IQR while the parquet and
    streaming backends used mean/std, so the same pool scored differently depending on
    `rescore.handoff` and on the 4 GB streaming threshold. All three use mean/std, which
    leaves the shipped parquet default and every published benchmark unchanged. The
    `MUMDIA_NN_FOLD_KEYS` companion is length-checked instead of being sliced short, which
    used to leave the tail rows unscored at a fabricated mid-rank score, and the estimate
    that picks the backend counts feature columns by name (F5).
  - `refuse_output_over_input` was wired into two of eighteen stages, so
    `compete --features f.parquet --out f.parquet` replaced the widest artifact of the run
    with the competed subset at exit 0. It now guards every output of `search-seed`,
    `rt-im-train`, `extract`, `features`, `compete`, `rescore`, `quant` and `audit` (F6).
  - The LOESS boundary extrapolation slope introduced in the previous package was the
    pointwise local slope at the sparsest, most one-sided point of the fit: unbounded, free
    to be negative, and multiplying an unbounded distance. It is the secant of the fitted
    curve over its end decile, clamped non-negative and to at most four times the global
    slope, and the test uses noisy anchors rather than a noiseless quadratic (F7).
    Measured on HYE B01 against the previous behaviour, same library and settings: 48,533
    stripped peptides at 1%, 53,127 PSM-q 1% targets, 6,519 protein groups and 1,961,800
    extracted rows in both arms, identical to the row. The two extrapolations agree
    wherever the anchors are dense and differ only outside the anchor range.
  - A desktop stop arriving between the reap and the end of `publish_exit` could pass a
    recycled process id to the tree kill. The waiter retires the id the instant `wait`
    returns, before it reads the output directory (F8).
  - The conversion lock added in the previous package spun without pause on an undeletable
    stale lock, mistook clock skew and a peer's partial file for evidence about its own
    holder, could be held by two processes at once, and left every interrupted conversion's
    partial mzML behind for ever. Take-overs are bounded and paced, the holder is
    identified by a token it reads back, a future modification time counts as fresh, the
    partial-file probe matches this destination only, and abandoned partials are swept
    under the lock (F9).
  - Dropping an unpredicted candidate with everything sharing its pair key also removed
    positional isomers that predicted correctly, bounded only by the library being emptied.
    The direct misses and the collateral are counted separately and exceeding 2% of the
    library is an error naming the sidecar. The key stays position-free deliberately: a
    positional key would stop matching a reverse decoy to its target, trading a sensitivity
    defect for an FDR one (F10).
- Code review E, follow-up (`docs/30_code_review_2026-09-08.md`, R1 to R9):
  - Enabling DeepLC fine-tuning with its own defaults was rejected at load, because the
    documented automatic batch size is `finetune_batch = 0` and the new validation demanded
    a positive batch. Only the epoch count has a lower bound now (R1; a regression from
    review A).
  - `run-experiment --run-names` accepted `a` and `a.`, one directory on Windows, and the
    second run overwrote the first with exit 0. Names ending in a dot or a space, containing
    `<>:"|?*` or a control character, or naming a Windows reserved device are rejected on
    every platform before anything is written (R2).
  - A desktop stop could sweep temporary files that belonged to the next run in the same
    folder: cancellation swept after the reservation had been released, and a stop on a
    finished run swept as well. Cancellation is now intent and kill only and inert once the
    run is terminal; the sweep happens in the waiter, after the reap and before the release,
    and a stop still killing finishes before the folder changes hands (R3).
  - Two searches converting the same vendor file concurrently shared one temporary output
    and one could publish the other's bytes. Each conversion writes a unique partial file
    under a lock beside the destination; a concurrent converter waits and reuses the
    result (R4).
  - Domain checks for the numeric settings review A left unchecked: `mbr.q_anchor`,
    `min_anchor_runs`, `extract.min_matched_fraction`, `features.bound_peak_fraction`,
    `quant.reliable_q` and the remaining fractions, correlations, tolerances and counts (R5).
  - The DeepLC fine-tune and re-prediction worker zipped predictions with peptidoforms
    without checking the count and silently kept the imported iRT for anything missing. A
    count mismatch is an error; rows that keep their imported value are counted in
    `<lib_out>.summary.json` and the engine warns when there are any (R6).
  - The audit's `reported` flag repeated the precursor gate, so it could read `true` next to
    `FAILED_PEPTIDE_FDR`, and a decoy could be `REPORTED`; the flag now follows the reason,
    a decoy past both gates is `REMOVED_DURING_REPORTING`, and a present `precursor_q` of the
    wrong type is an error rather than a fallback (R7).
  - The desktop results-folder reservation compared exact folders only, so a search into a
    child of an active experiment's folder was allowed; ancestors and descendants are
    refused, siblings are not (R8).
  - The Windows debug binary overflowed its 1 MiB main-thread stack on `--version`; the CLI
    runs on a thread with a 256 MiB reservation and an integration test runs the built
    binary (R9).
- Code review D, calibration, provenance, reporting (`docs/29`, findings 10, 15, 16, 19):
  - LOESS retention-time calibration switched to the global least-squares line the
    moment a query left the anchor range, while the grid just inside used the local fit,
    and the two need not agree: on `y = 200 + 10x^2` (span 0.3) the prediction jumped
    from 193.4 at `x = 1e-6` to 38.3 at `x = 0`, and from 1173.5 to 1018.4 at the top,
    about 155 s discontinuities that misplaced gradient-edge peptides relative to their
    extraction window. The map now continues the boundary local fit (its value and
    slope) outside the range, and is continuous at both ends; the global line remains
    only the degenerate fallback (#10). Measured on HYE B01 with the imported DIA-NN iRT
    as the RT source and `native_tda`: 45,946 stripped peptides at `peptide_q_value` 1%
    before, 45,957 after, at an unchanged 1.0% PSM-level decoy fraction and 6,410 protein
    groups in both arms. 208,130 of 10.88 M candidates (1.9%) received a different window,
    194,698 of them with iRT above the anchor range, which the global line had placed
    past the end of the 9,000 s run; the local fit places them at 8,578 to 9,100 s, and
    extract accepted 454 more rows from them. A second pair on the DeepLC 4.1.1
    re-predicted precursor table (`w_rt` 414 s against 691 s): 48,533 stripped peptides in
    both arms, PSM-q 1% targets 53,124 against 53,127 at the same decoy fraction, 6,519
    protein groups in both, 0.2% of candidates with a different window. Neutral on both RT
    sources, which is what a boundary correction should be.
  - The candidate audit's `passed_precursor_fdr` gate and `FAILED_PRECURSOR_FDR` reason
    read the PSM `q_value`; they read `precursor_q`, the unit the label names, with the
    PSM q as a recorded fallback on tables without it. A pooled scored table (several
    `source` values) is refused, because the audit keys on `candidate_id` and would
    attribute the last run's fate to every run (#16).
- Desktop: the digest fields on the Search screen (missed cleavages, peptide length,
  charge range, carbamidomethyl, oxidation) now reach the engine on the built-in
  library path. They were read only by the DIA-NN library build, so with the built-in
  predictors the engine digested with the preset's values; the block also showed the
  two modification checkboxes twice under the same ids. The run's configuration is now
  the selected preset with the fields merged on top (`derive_config`), validated by
  the engine before the search starts.
- Desktop: the settings editor starts from the preset selected on the Search screen
  instead of from the engine defaults, so "Save and use" writes the preset plus the
  edits rather than silently dropping the preset's predictor, rescorer and interpreter
  choices. Engine fields the schema marks `not yet wired` (the match-between-runs
  tiers) are labelled as such and cannot be edited; list-valued settings display and
  accept JSON.

## [0.1.1] - 2026-09-07

### Fixed

- The 0.1.0 desktop installers (`.msi`, `.AppImage`) shipped without the Python
  workers: `binaries/scripts/` held its README alone, because the release workflow
  staged the engine and `uv` but never copied `scripts/*.py`, and the Tauri resource
  glob was satisfied by the README. The engine accepts a script directory only when it
  holds a worker file, so the installed application had no sidecars and DeepLC, the
  neural rescorer, mokapot and the DIA-NN import failed at the point of use. The
  workflow now stages the workers and, before uploading, unpacks every bundle it built
  (`msiexec /a`, `--appimage-extract`) to assert the console, the engine, `uv` and each
  worker are inside and the bundled engine runs. Found by unpacking the published
  installers; the Setup page installs the Python environment, not these files.
- A SCIEX `.wiff` without its `.wiff.scan` companion fails in msconvert with
  `Could not open data stream. Is a required 'scan' file missing?`, which names no
  file. The engine now appends the missing companion's path to that error. Measured on
  PRIDE-archived `.wiff` files that had been downloaded without their companions.

## [0.1.0] - 2026-09-06

### Pre-release audit (2026-08-28)

A six-way audit of the tree ([`docs/25_release_readiness_review.md`](docs/25_release_readiness_review.md))
found three release blockers and about thirty further defects. All three blockers
and most of the rest are fixed; that document's status section records what is
deliberately still open. The entries below fold into the sections that follow.

**Second audit (2026-08-28, external).** A second review of the same tree raised
four further blockers, two of which were defects in the first audit's own output: a
generated document quoting a value from a test as though the engine set it
(`ci/gen_config_reference.py` did not skip `#[cfg(test)]`), and a `build.rs` that
guessed at the git directory and stamped a stale commit into every manifest. Both
fixed, the second verified across a commit. The other two were release mechanics and
are covered under Added below: a release archive that could not run the verification
its own documentation prescribes, and a `v*` tag that could publish any commit with
no CI behind it.

**Two changes require re-measurement rather than only review.**

- Three defaults changed on correctness grounds, not from a count:
  `extract.apex_evidence_rank` to `true` (the legacy apex silently selected the
  lowest-RT qualifying scan when none of the top-K predicted fragments was
  observed anywhere) and `features.emit_pin` to `false` (no stage reads the file;
  it is a ~5.4 GB write per run). `extract.gate_min_score` was also briefly changed
  to `0.6` and then measured back to `0.2`: 0.6 costs 4.4% of peptides for
  `native_tda` and 4.7% for `nn_torch` at an unchanged decoy fraction, because the
  gate sweep that motivated it predates the current defaults and its optimum has
  moved to the loose end for both rescorers.
- The `nn_torch` CV fold is keyed on `base_peptide_id` supplied by the engine
  rather than on a hash of the peptidoform, so a target and its paired decoy now
  share a fold as `percolator_lite` always did and as `docs/11` always claimed.

Both move `nn_torch` counts, and the entrapment fixes below invalidate any
entrapment measurement previously taken through the native rescorer.

**Breaking: interface renames with no compatibility aliases.** A tag freezes
these, so they are free now and a major bump later. CLI:
`--library-precursors`/`--library-fragments` to `--lib-`, `--out-chrom` to
`--out-chromatograms`, `--scored` to `--psms-scored`, `--out-scored` to
`--out-psms-scored`, `--psms` to `--psms-extracted`, and `--seed`/`--seeds` to
`--seed-psms`. Config: `extract.min_frag_corr` to `extract.gate_min_score` (it is
not a correlation under any `gate_mode`), and `compete.group_by = "precursor"` to
`"base_peptide"` (it keys on the stripped sequence). An old name now fails with
the offending key and the valid alternatives listed. Local configs need:

```bash
sed -i 's/"min_frag_corr"/"gate_min_score"/; s/"group_by": *"precursor"/"group_by": "base_peptide"/' config.*.json
```

### Added

- **Several files pool by default.** `run` given more than one `--mzml` dispatches to
  `run-experiment`: files provided together are rescored together (one pooled FDR),
  quantified per run and aligned across runs (MaxLFQ). Searching files separately is
  the opt-in, one `run` per file.
- **Experiment-wide report.** `run-experiment` writes `peptides.tsv` and `proteins.tsv`
  at the experiment root, selected on the experiment-wide `peptide_q_value` and
  `pg_q_value`, with an `n_runs` column (per-run acceptances on `run_psm_q`) and one
  `quantity_<run>` and `lfq_<run>` column per run. `mumdia report --experiment-dir`
  rewrites the pair at another threshold. No per-run TSVs are written, because the
  grouped q columns go to each group's experiment-wide winner only.
- **Vendor formats.** A vendor file given as `--mzml` is converted to mzML first
  (`convert`, `run`, `run-experiment`, `peak-census`): Thermo `.raw` through
  ThermoRawFileParser (or msconvert), Bruker and Agilent `.d`, SCIEX `.wiff` and
  Waters `.raw` through ProteoWizard `msconvert`. Converters are located
  (`convert.thermo_raw_parser`, `convert.msconvert`, both `auto`, or
  `MUMDIA_THERMO_PARSER` / `MUMDIA_MSCONVERT`), never shipped. The mzML is written
  beside the input (or into the output directory when that is not writable) through
  a `<name>.partial.mzML` temporary and reused on later runs when newer than its
  source (`convert.reuse_converted`). `mumdia doctor` reports both converters and does
  not fail for their absence. Only Thermo is exercised end to end (a 3.7 GB Astral
  run, 6:40 through ThermoRawFileParser 2.0.0); the four msconvert formats are wired
  and unverified, and ion mobility is discarded.
- **Library retention time from the DeepLC base model.** `rt_im_train.library_irt`
  (`auto`, `library`, `deeplc`) re-predicts an imported library's iRT with the DeepLC
  base model once per experiment when a DeepLC interpreter is configured, because the
  imported DIA-NN iRT is the worst RT source measured: AIF 10,416 peptides against
  10,015 raw; HYE B01 58,842 against 56,556 raw over three NN seeds. The optional
  fine-tune (`finetune_deeplc`) remains available and is still +2.4% on HYE.
- **DeepLC 4.1.1 is a floor.** `mumdia doctor`, the sidecar launch
  (`sidecar::require_deeplc_version`) and both worker scripts refuse an older DeepLC
  (`mumdia_core::constants::MIN_DEEPLC_VERSION`), because the default
  prediction-plus-calibration workflow is only sound on a base model that does not
  memorise its anchors.
- **Rescore handoff and training recipe.** `rescore.handoff = parquet` replaces the
  TSV handoff to the Python worker (rescore peak 29.96 to 8.95 GB, wall 8:35 to 6:33
  on the HYE competed table, identical identifications; mokapot still receives a PIN).
  The worker trains on the targets at 1% plus a capped, hybrid-selected decoy sample
  with warm refits (`train_neg_ratio 3`, `train_neg_select hybrid`,
  `train_warm_epochs 5`): HYE A01 +1.0%, HYE B01 +2.2%, AIF -0.1%, entrapment +3.3% at
  an unchanged spike-in FDP, at 9 to 19x less training time. `rescore.features` /
  `features_file` project the classifier's input columns; `feature_preset = compact`
  (114 features) is an opt-in memory lever, not a sensitivity one;
  `max_feature_matrix_gib` turns an oversized matrix into an error at startup;
  `MUMDIA_NN_SEED` sets the worker's base seed.
- **Memory footprint.** Streaming Parquet readers, incremental extract output flushed
  as isolation windows close (`extract.windows_in_flight`, auto, capped at 16), f32
  bulk arrays and a chunked features stage. HYE B01 single run: 231 GiB and 1:07:30
  before, 16.5 GiB and 17:52 after (compact preset; about 20:40 with every feature).
  Six pooled HYE runs rescored in 18 minutes at 15.9 GB against 4:34:42 at 40 GB.
- **N-terminal methionine excision** in the native digest
  (`digest.n_term_met_excision`, default on, matching DIA-NN `--met-excision`); old
  configurations still parse. `scripts/augment_library.py` uses the same digest to add
  the tryptic peptides an imported library is missing.
- **Imported libraries with empty protein cells load.** An empty `protein` (DIA-NN
  writes the iRT-kit standards that way) is grouped as `UNASSIGNED` with a warning
  that counts the rows; `scripts/import_diann_lib.py` writes the same group. An empty
  `peptidoform` is still an error.
- **Desktop application** (`desktop/`, "MuMDIA Console"): a Windows `.msi` and a Linux
  `.AppImage` built by the release workflow, bundling the engine and `uv`. It creates
  its own Python environment (no conda), installs ThermoRawFileParser on request,
  locates msconvert and DIA-NN, and rescores all files provided together by default.
  Its backend is unit-tested and both bundles were built and inspected; nobody has yet
  clicked through the interface end to end.

- The release archive verifies itself. It now carries `ci/smoke.sh`, its two helper
  scripts and `test_data/fixture.fasta`, and `release.yml` unpacks every archive it
  builds into a clean directory and runs that archive's own smoke test, on every
  target. `docs/19` told the reader to run exactly this while the archive shipped
  neither `ci/` nor `test_data/`; testing the artifact rather than the tree it came
  from is also the only check that can catch a packaging mistake, and it gives macOS
  its first end-to-end coverage.
- `validate-tag`, a release job every build depends on: the tag must equal the
  workspace version, the tagged commit must be an ancestor of `main`, and `ci.yml`
  must have a successful run for that exact SHA. A tag push does not trigger
  `ci.yml`, so the only checks behind a release were previously `--version`,
  `--help` and `doctor`.
- `run-experiment` coverage: `ci/smoke.sh` now runs the multi-run orchestrator over
  two copies of the fixture and asserts the pooled rescore, the by-source split, the
  per-run quantification and the cross-run LFQ. The multi-run path had no test of
  any kind, and its split had a silent data-loss case (see Fixed).
- The experiment manifest records one artifact per output it writes, each with a
  content hash, row count and schema version. It previously listed output paths and
  nothing else, so two experiment results could not be compared. New artifact
  identity `lfq_maxlfq` for the cross-run table.
- `sbom.cdx.json`: a CycloneDX 1.5 software bill of materials generated from
  `cargo metadata --locked` by `ci/gen_sbom.py`, covering all 173 components with
  purls and the full dependency graph. Shipped in the release archive and at
  `/opt/mumdia/sbom.cdx.json` in the image, and checked for staleness in CI.
  `THIRD_PARTY_LICENSES.md` is a notice document for a human reader; this is the
  machine inventory a vulnerability scanner or a software inventory consumes.
- `pip-audit` over both resolved sidecar environments, strict on the weekly
  scheduled run and advisory on pull requests, plus `pip freeze --all` uploaded per
  environment as a 90-day artifact. This is what covers the Python dependency
  surface, which has no Dependabot support: the pins live in the `pip:` sections of
  the `env/` conda specifications, which Dependabot cannot parse, and a mirror
  requirements file would be a second list that nothing installs. Reasoning in
  `docs/14`.
- Release platforms are now Linux (musl), Windows and Apple silicon. The Intel Mac
  target was removed: it required GitHub's `macos-13` label, which no longer receives
  a runner (measured 2026-08-28: queued over two hours with none assigned, in two
  separate rehearsals, while every other target finished in about three minutes), so
  a real tag would have hung until GitHub's 24-hour queue timeout and then failed (a
  rehearsal job was observed reporting exactly `24h0m0s`; that is the limit on waiting
  for a runner, not the six-hour limit on a running job). Cross-compiling it on
  the Apple silicon runner was rejected because the result cannot be executed there,
  and publishing the one archive nobody ran is what the verification step above exists
  to prevent. Intel Mac users build from source or use the container image.
- Docker base images pinned by digest, with a Dependabot `docker` entry to keep the
  pins current. A tag is mutable, so a rebuild of the same commit could previously
  produce a different image.
- `--min-assertions` on `ci/check_smoke.py`: the smoke run fails if fewer assertions
  execute than the count quoted in the documentation. The documented count was 112
  while 117 ran, and a guard block that stops executing fails no assertion, so it
  reads as a pass.
- `quant.fragment_selection = predicted` ranks a precursor's fragments for the
  top-N sum by their library intensity instead of by their own integrated area.
  Ranking by observed area preferentially selects interfered fragments, because
  interference inflates the very quantity the ranking rewards, and the selected
  set then varies between runs.
- `quant.fixed_scan_halfwidth` and `quant.fixed_window_s` integrate a fixed
  window centred on the identification apex instead of the descent-walk bounds.
  The seconds form is instrument-independent and overrides the scan form. On the
  ProteoBench Astral HYE set these two options together moved median absolute
  epsilon from 0.273 to 0.195 and CV from 0.175 to 0.107.
- `quant.baseline_subtract`, with `baseline_flank_scans` and
  `baseline_quantile`, subtracts a flank-quantile background inside the fixed
  window.
- `prescan` stage: a native per-run sequence-tag prescan that prunes
  modification-bearing candidate hypotheses with no anchored tag support, 11.6
  times faster than the previous Python screen. Only modform hypotheses are ever
  pruned.
- `rt_im_train.window_holdout_frac` sizes the RT window from held-out anchors
  rather than from the anchors the calibration was fitted on. Benchmark-gated and
  off by default: it gained 1.1% of peptides with DeepLC 4.1.0 but lost 1.5% with
  the overfitting 4.0.0a2 model, so it interacts with RT-model quality.
- `env/mumdia-deeplc.yml`: a portable conda spec for the DeepLC sidecars. They
  previously had no committed local environment, so running them meant
  reconstructing a developer machine by hand.
- Sidecar interpreter discovery. A `python` field may be `"auto"` or absent, and
  the engine finds an interpreter from `MUMDIA_PYTHON_<ROLE>`, `MUMDIA_PYTHON`,
  `CONDA_PREFIX`, `VIRTUAL_ENV`, or `PATH`, accepting a candidate only after it
  imports what that role's workers import. A role is resolved only if the
  configuration uses it, so a default native run still needs no Python at all.
  Explicit paths behave exactly as before.
- Global CLI flags, accepted before or after the subcommand: `--threads N`
  bounds the engine's rayon pool and is forwarded to the sidecars as
  `MUMDIA_NN_THREADS` and `OMP_NUM_THREADS` when those are unset;
  `--log-level`, `-v`/`-vv` and `-q` set verbosity. Previously the only control
  was `RUST_LOG`, which is not discoverable from `--help`, and thread count could
  not be bounded at all: the engine never read `RAYON_NUM_THREADS`, so a run took
  every core on a shared machine.
- `configs/examples/{native,fasta-sidecars,diann-library}.json`, portable
  starting points that use `"auto"`, with `configs/README.md` explaining the
  resolution order and the environment specs. These replace the only tracked
  config, which named one developer's interpreters and OneDrive path and was the
  config the documentation told everyone to run.
- End-to-end smoke test, run in CI on Linux and Windows: `ci/smoke.sh` builds a
  synthetic library from `test_data/fixture.fasta`, generates a matching mzML from
  the engine's own library so the planted peaks cannot disagree with the mass
  model, runs the single-run pipeline twice, runs `run-experiment` over two copies
  of the fixture, and asserts 136 things. It covers mzML parsing, the library
  build, the `run` orchestrator and its manifest, retention-time calibration, the
  report writers, and the multi-run path (pooled rescore, by-source split, per-run
  quant, cross-run LFQ), none of which had any test. `--min-assertions` fails the
  run if fewer assertions execute than the count quoted here, so a guard block
  that stops running cannot pass silently.
- A further CI job asserts the two platforms produced byte-identical
  `peptides.tsv` and `proteins.tsv`. The native pipeline turns out to be
  byte-reproducible across operating systems, not only across runs.
- `tests/python`: 71 tests over the Python worker contracts, run in CI. Tests
  needing torch, mokapot, deeplc or ms2pip skip rather than fail.
- `docs/23_cli_reference.md` and `docs/24_config_reference.md`, generated from
  `--help` and from `config.rs` by `ci/gen_*_reference.py` and checked for
  freshness in CI, so a new flag or field lands with its documentation. The second
  includes the environment-variable table that existed nowhere: 47 variables read
  across engine and sidecars, plus the 11 the code sets.
- `bench/`: the portable part of the ProteoBench scoring path, the two recorded
  results with their row units and q columns, and the measured resource profile of
  a reference run (85 minutes and 13.1 GB of artifacts from a 1.94 GB mzML,
  rescoring 80% of it).
- `ci/check_doc_refs.py`, run in CI: fails when a tracked file cites a Markdown
  document the repository does not ship.
- `CONTRIBUTING.md`, `SECURITY.md`, this changelog, and
  `docs/22_release_plan.md`.

All new quantification options default to off, so an existing configuration
produces bit-identical results.

### Changed

- **Competition key: `compete.group_by = peptidoform_charge`** (keys
  `(pform_id, label, charge, peak_rank)`). Sibling charge states and modforms of one
  peptide are separate precursors that compete only against their own alternative
  peaks, the unit DIA-NN reports at and the key every benchmark in
  `docs/28_feature_selection_analysis.md` ran under (entrapment FDP flat at
  0.48-0.64%). The previous default `base_peptide` (renamed from `precursor`, which it
  was not) deleted every charge and modification variant of a peptide but the highest
  `prelim_score` before rescore: 23% of the extracted candidates on HYE B01, 46.6% on a
  modification-rich library, at an unchanged peptide count. It stays available as an
  explicit peptide-level population and must not be used for a PTM search.

- CPU PyTorch in the three DeepLC-bearing environment sets (`env/docker-deeplc.yml`,
  `env/mumdia-deeplc.yml`, `env/console-requirements.txt`) moves from `2.12.1+cpu` to
  `2.14.0+cpu`, the first version whose metadata allows a `setuptools` without
  PYSEC-2026-3447 (`>=77.0.3` instead of `<82`). Resolved on 2026-09-06 with
  `deeplc==4.1.1`: numpy 2.4.6, pandas 2.3.3, psm-utils 1.5.5, scikit-learn 1.9.0,
  setuptools 84.0.0, i.e. the same scientific stack as before with only torch
  changed. DeepLC 4.1.1 declares `torch<3,>=2.6.0`. Neural-network training is not
  bit-deterministic across torch versions, so expect seed-level, not result-level,
  differences in `nn_torch` rescoring.
- CI now enforces the full stated gate: `cargo fmt --check` and
  `cargo clippy --workspace --all-targets -- -D warnings` in addition to the
  build and tests, plus `python -m compileall` over the sidecars, a JSON parse of
  every tracked configuration, a YAML parse of the environment specs, and the
  documentation-reference check. Formatting and clippy were previously a local
  responsibility, so `main` could carry a tree that failed them.
- The Docker DeepLC environment pins `deeplc==4.1.1` from PyPI instead of a git
  commit on the 4.0 multitask branch, and no longer caps `numpy<2`, which 4.1.1
  does not require. 4.1.1 is a floor, not merely the current release: the 4.0.0a2
  multitask preview overfits per-run fine-tuning badly enough to invert RT-model
  rankings. Verified by building the image and importing the workers' graph in
  the worker's own order: DeepLC 4.1.1, torch 2.12.1+cpu, numpy 2.4.6 in the
  `deeplc` environment and mokapot 0.10.0 in `rescore`, with `mumdia doctor`
  passing on both baked configurations.
- The image no longer runs as root after setup. It needs
  `--user "$(id -u):$(id -g)"` to write into a bind mount, which the documented
  invocation now passes and the Docker workflow now asserts.
- Under a fixed integration window, the reported `integration_lo_rt` and
  `integration_hi_rt` are now the retention-time extent actually integrated
  rather than the walked bounds that were ignored. Measured on the AIF benchmark
  run behind the ProteoBench submission: 72,168 quantified precursors, every
  `quantity`, `n_fragments_used`, `quant_status` and `integration_apex_rt`
  bit-identical, and the reported window corrected from a 29.1 s median (the
  descent walk) to 34.9 s (the fixed window that produced the numbers).
- `mumdia doctor` reports whether the configuration can actually run: the
  interpreter each role resolves to and how it was found, the versions of the
  packages whose version changes results, whether the worker scripts are where
  the engine will look, and a warning when DeepLC is older than 4.1.1. It now
  covers `mbr.python` and the script directory, neither of which it checked
  before, and it no longer fails a native configuration over a worker directory
  that configuration never opens.
- `predict_frag.sidecar_script_dir` is resolved against the config file's own
  directory and against the executable's directory, not only the current working
  directory. The same config invoked from elsewhere used to silently change which
  worker scripts ran.
- `run-experiment` warns when it overrides the configured `quant.q_filter` to
  gate per-run quantification on the pooled q value, instead of doing it
  silently.
- Dependabot keeps Cargo and GitHub Actions dependencies current, with
  `arrow`/`parquet` grouped separately because they carry the on-disk contract.

### Fixed

- The sidecar environment specifications pin `setuptools>=83`. The CI audit of the
  resolved DeepLC environment found `setuptools 81.0.0` (PYSEC-2026-3447,
  CVE-2026-59890, fixed in 83.0.0) and failed the main branch after the merge of #54.
  The conda-level pin alone did not hold: `torch 2.12.1` declares `setuptools<82`, so
  pip downgraded the conda-installed 84.0.0 to 81.0.0 underneath it. torch is now
  2.14.0 (see Changed) and the floor is repeated in the pip sections and in the
  desktop requirement set. The audit step is advisory on pushes to main as well as
  on pull requests, as its comment already intended; the weekly scheduled run and a
  manual dispatch stay strict.
- A single malformed retention time in an mzML aborted the whole run. `convert`
  validated peak m/z and intensity but not the scan start time, so one `NaN` value
  passed unchecked into the spectra artifact and then panicked inside extract with
  `called `Option::unwrap()` on a `None` value`, naming neither the file, nor the
  scan, nor the value. Reproduced by editing one value in the fixture mzML. Such
  spectra are now dropped with a count and the first offending scan id, which loses
  nothing (a spectrum with no retention time cannot be placed in a chromatogram) and
  leaves identifications unchanged; `ci/smoke.sh` asserts all three.
- Every float ordering in the workspace now uses `total_cmp` rather than
  `partial_cmp(..).unwrap()` (25 sites) or `partial_cmp(..).unwrap_or(Equal)` (36
  sites). The first panics on NaN; the second is worse, because `Equal`-on-NaN is an
  intransitive comparator and `sort_by` has detected that and panicked since Rust
  1.81, so it converted a deterministic failure into an intermittent one. `total_cmp`
  agrees with both on every finite value: the fixture's `peptides.tsv` and
  `proteins.tsv` hashes are byte-identical across the change. One of the rewritten
  comparators picks the competition winner, where treating every NaN as equal made
  the surviving row depend on iteration order.
- `compete` panicked instead of erroring when a `.schema.json` companion named a
  feature column the parquet does not have, which a stale companion beside a
  rewritten table produces. It now names the column and the file and says to delete
  the companion.
- `scripts/make_reverse_decoys.py` silently assigned 0 Da to any modification outside
  its eight-name table, so those decoys got fragment m/z for the wrong molecule and
  could never match. A decoy that cannot match does not compete, which makes the
  target-decoy null optimistic for exactly the peptides carrying that modification,
  and nothing in the output distinguished such a decoy from a good one. The sampled
  calculator check could not catch it: it compares 500 precursors at the 99th
  percentile. Unknown modifications now raise, `valid()` rejects the peptidoform so
  no decoy is written for it, and the script reports the names and counts. This
  matches the engine's own parser, which has always returned
  `MassError::UnknownModification`.
- `run-experiment` dropped PSMs silently when splitting the pooled scored table by
  run. `split_by_source` filtered on `source == i` for each output table and returned
  `Ok` regardless, so any row whose `source` had no output table went nowhere: every
  per-run quantity and the cross-run LFQ were then computed from a smaller population
  with no error and no warning. It now counts the rows it placed and refuses if that
  is not all of them.
- The library helpers write parquet the engine can read on any pandas.
  `DataFrame.to_parquet` chooses the arrow string width itself, and pandas 3
  writes `large_string`, which the engine rejects at load with
  `column 'peptidoform' is not utf8`. Every library built by
  `import_diann_lib.py`, `make_shift_decoys.py`, `make_reverse_decoys.py` or
  `augment_library.py` on a current pandas was therefore unreadable, breaking the
  imported-library path. They now write through `scripts/_lib_io.py`. Found by the
  new sidecar contract tests.
- MBR transfers are now quantified. The augmented scored table lowered only
  `q_value`, while quantification gates on `quant.q_filter`, which the experiment
  path sets to `run_psm_q`. An accepted transfer therefore kept a sub-threshold
  `run_psm_q` and was dropped: 34,280 of 34,664 transfers on a six-run HYE
  experiment, so match-between-runs appeared to run and changed almost nothing.
- `quant` reads `predicted_intensity` as an optional chromatogram column.
  Requiring it made every chromatogram artifact written before that column
  existed unquantifiable; the `predicted` ranking, its only consumer, now fails
  with an actionable message instead.
- `scripts/deeplc_worker.py` imports `deeplc` before `numpy` and `pyarrow`. The
  wrong order aborts torch DLL initialization on Windows with
  `OSError: [WinError 1114] ... c10.dll`. The failure was latent because
  imported-library mode skips the stage that reaches it.
- `.gitignore` covers `rust/mumdia/target` (the root-anchored `/target` never
  matched it), the experiment configurations that carry machine-specific
  interpreter paths, and the local benchmark data directories. It no longer
  matches `docs/22_release_plan.md`, which an unanchored `*_plan.md` rule had
  silently excluded from version control.
- Source comments no longer cite untracked local design notes. About 130
  references pointed at documents a clone does not receive; they now point at the
  tracked `docs/` guide.

### Known limitations

- The sidecar contract tests cover the workers' file contracts, not the science:
  the tests that need torch, mokapot, DeepLC or MS2PIP skip on a runner without
  them, so CI does not validate rescoring or retention-time prediction behaviour.
- The end-to-end smoke test runs on Linux and Windows, not macOS, and uses the
  native predictors only. A separate job imports DeepLC, mokapot and MS2PIP in
  real conda environments on any pull request touching `scripts/`, `env/`,
  `tests/python/` or the Dockerfile, but no CI job runs the sidecar path end to
  end on data.
- Under an experiment-wide rescore no per-run TSV report is written; per-run counts
  come from the split scored tables on `run_psm_q`, and the experiment-wide
  `peptides.tsv` / `proteins.tsv` are the reports.
- `mbr.strategy` distinguishes only none from not-none; `rt_window_s`,
  `decoy_transfer` and `requant_all` are accepted but not wired.
- `extract.retain_top_peaks > 1` writes diagnostic peak alternatives that do not
  reach features or rescoring.
- No ion mobility support: a Bruker diaPASEF `.d` is converted to 3D spectra and
  searched with more interference than a 4D engine would see. Vendor formats other
  than Thermo `.raw` are converted through msconvert but have not been exercised on
  real files. No wildcard or terminal variable modifications.
- The desktop application has not been clicked through end to end; its backend is
  unit-tested and both bundles were built and inspected.
