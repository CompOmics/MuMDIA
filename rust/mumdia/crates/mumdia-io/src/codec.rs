//! Parallel parquet column encoding (docs/03_io_layer.md, "Parallel column codec").
//!
//! parquet-rs's [`ArrowWriter`] encodes one row group's columns one after another on the
//! calling thread: dictionary interning, statistics, page assembly and snappy for every leaf
//! of every batch. On a wide artifact that is the whole write. The streamed rescore handoff
//! of the six-run HYE pool spent about 26 s of one core there, and an immunopeptidomics
//! handoff about 26 minutes; the features writer thread encodes all ~398 columns alone.
//!
//! Columns are independent inside a row group, so `ColumnEncoder` encodes them
//! concurrently and appends the finished column chunks in schema order. It is built from the
//! same parts [`ArrowWriter`] uses (`into_serialized_writer`, then one
//! [`ArrowColumnWriter`] per leaf from the [`ArrowRowGroupWriterFactory`]) and feeds every
//! column writer exactly the leaf slices the serial writer would feed it, in the same order,
//! so the file is the serial writer's file byte for byte:
//!
//! * a batch is split at the row-group cap where `ArrowWriter::write` splits it, and an
//!   empty batch is skipped as it skips one;
//! * each column writer receives one `write` per (split) batch, so the encoder's 1,024-value
//!   mini-batches and therefore its page boundaries fall where they fell;
//! * the `ARROW:schema` key-value metadata is added by `ArrowWriter::try_new` itself before
//!   the serialized writer is taken from it.
//!
//! `the_parallel_encoder_writes_the_serial_writers_bytes` pins all three.
//!
//! The work runs on a DEDICATED rayon pool, never on the global one: a writer called from
//! a global worker that queued its columns on the global pool and waited for them could wait
//! forever at `--threads 1`. And a writer called from inside ANY rayon pool encodes on its
//! own thread instead of waiting for the codec pool, because rayon lets a worker that waits
//! on another pool steal jobs of its own pool meanwhile, and such a job could try to take a
//! lock the waiting writer's caller holds. The parallel path is therefore taken from plain
//! OS threads only: the main thread, extract's chromatogram writer thread, the features
//! writer thread. A band or run already running on the global pool is parallel at that
//! level. The codec pool's jobs never wait on anything but their own column.
//!
//! The pool is shared by every plain-thread writer and decoder of the process, so it counts
//! the callers inside it. When as many callers are waiting on it as it has threads, the
//! next caller encodes (or decodes) on its own thread instead of queueing behind them
//! ([`claim`]). Without that, more than eight concurrent writers (one extract chromatogram
//! writer per band in flight under `groups.parallel`, plus features and compete writers)
//! would share eight codec threads where they used to have one core each, and a band's
//! candidate loop would stall on its writer's channel. Both paths write the same bytes.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Context, Result};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::{
    compute_leaves, ArrowColumnChunk, ArrowColumnWriter, ArrowRowGroupWriterFactory,
};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use rayon::prelude::*;

/// The codec pool's size when neither `MUMDIA_PARQUET_THREADS` nor [`set_codec_threads`]
/// says otherwise, and the ceiling applied to the machine's parallelism. Encoding one row
/// group is a few hundred column jobs of ~1-10 ms each, so a handful of threads takes most
/// of the gain and more would only compete with the stage's own pool.
pub const DEFAULT_CODEC_THREADS: usize = 8;

/// Set by the CLI from `--threads` before any artifact is written; 0 means unset.
static REQUESTED: AtomicUsize = AtomicUsize::new(0);

static POOL: OnceLock<Option<Arc<CodecPool>>> = OnceLock::new();

/// A rayon pool for column jobs, with a count of the plain-thread callers inside it.
pub(crate) struct CodecPool {
    pool: rayon::ThreadPool,
    /// Callers currently in [`Claim::install`], each a plain thread waiting on its own jobs.
    callers: AtomicUsize,
}

impl CodecPool {
    /// A codec pool of `threads` threads named `mumdia-codec-{i}`.
    pub(crate) fn new(threads: usize) -> Result<Arc<CodecPool>> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("mumdia-codec-{i}"))
            .build()
            .map_err(|e| anyhow!("building the parquet codec pool: {e}"))?;
        Ok(Arc::new(CodecPool {
            pool,
            callers: AtomicUsize::new(0),
        }))
    }

    /// Threads of the pool.
    pub(crate) fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }
}

/// One caller's turn on a [`CodecPool`]; releases its place when dropped.
pub(crate) struct Claim<'a>(&'a CodecPool);

impl Claim<'_> {
    /// Run `op` on the pool and wait for it.
    pub(crate) fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        self.0.pool.install(op)
    }
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        self.0.callers.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A turn on `pool` for the calling thread, or `None` when the caller should do the work on
/// its own thread: always from inside any rayon pool (module docs), and when as many callers
/// already wait on the pool as it has threads, so that concurrent writers beyond the pool's
/// size keep one core each instead of queueing (module docs).
pub(crate) fn claim(pool: &Option<Arc<CodecPool>>) -> Option<Claim<'_>> {
    let pool = pool.as_deref()?;
    if rayon::current_thread_index().is_some() {
        return None;
    }
    let limit = pool.threads();
    pool.callers
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
            (n < limit).then_some(n + 1)
        })
        .ok()?;
    Some(Claim(pool))
}

/// Size the codec pool from `--threads`: at most `threads` codec threads, and none (the
/// serial path) at `--threads 1`. Call it before the first artifact is written; the pool is
/// built once, on first use, and a later call changes nothing.
pub fn set_codec_threads(threads: usize) {
    REQUESTED.store(threads.max(1), Ordering::Relaxed);
}

/// The number of threads the codec pool has or will have; 1 means columns are encoded on
/// the calling thread. `MUMDIA_PARQUET_THREADS` wins (0 and 1 both mean serial), then
/// [`set_codec_threads`] capped at [`DEFAULT_CODEC_THREADS`], then the machine's
/// parallelism capped at [`DEFAULT_CODEC_THREADS`].
pub fn codec_threads() -> usize {
    if let Some(n) = std::env::var("MUMDIA_PARQUET_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        return n.max(1);
    }
    let requested = REQUESTED.load(Ordering::Relaxed);
    let ceiling = if requested > 0 {
        requested
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    };
    ceiling.clamp(1, DEFAULT_CODEC_THREADS)
}

/// The process-wide codec pool, or `None` when the codec runs serially.
pub(crate) fn codec_pool() -> Option<Arc<CodecPool>> {
    POOL.get_or_init(|| {
        let n = codec_threads();
        if n <= 1 {
            return None;
        }
        match CodecPool::new(n) {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), "parquet codec pool unavailable; encoding serially");
                None
            }
        }
    })
    .clone()
}

/// One row group being encoded: the column writers of each root field, in schema order.
struct RowGroup {
    roots: Vec<Vec<ArrowColumnWriter>>,
    rows: usize,
}

/// A parquet writer that encodes a row group's columns concurrently and writes the file the
/// serial [`ArrowWriter`] writes, byte for byte (module docs).
///
/// Only the row-count row-group cap is supported; a byte cap or content-defined chunking
/// would split rows on measured sizes, and no writer in this crate sets either. The
/// constructor refuses them rather than writing a different file.
pub(crate) struct ColumnEncoder<W: Write + Send> {
    file: SerializedFileWriter<W>,
    factory: ArrowRowGroupWriterFactory,
    schema: SchemaRef,
    /// Parquet leaves under each root field, in schema order.
    leaves_per_root: Vec<usize>,
    max_rows: Option<usize>,
    in_progress: Option<RowGroup>,
    pool: Option<Arc<CodecPool>>,
}

impl<W: Write + Send> ColumnEncoder<W> {
    /// An encoder on `sink` for `schema` under `props`, with the columns of a row group
    /// encoded on `pool` (serially when `None`, from inside a rayon pool, or while the pool
    /// is saturated: [`claim`]).
    pub(crate) fn try_new(
        sink: W,
        schema: SchemaRef,
        props: WriterProperties,
        pool: Option<Arc<CodecPool>>,
    ) -> Result<ColumnEncoder<W>> {
        if props.max_row_group_bytes().is_some() || props.content_defined_chunking().is_some() {
            return Err(anyhow!(
                "ColumnEncoder supports a row-count row-group cap only, not a byte cap or \
                 content-defined chunking"
            ));
        }
        let max_rows = props.max_row_group_row_count();
        // `try_new` adds the ARROW:schema metadata to the properties the file writer is
        // built with, which is the one part of the serial writer this could get wrong.
        let (file, factory) =
            ArrowWriter::try_new(sink, schema.clone(), Some(props))?.into_serialized_writer()?;
        let descr = file.schema_descr();
        let mut leaves_per_root = vec![0usize; schema.fields().len()];
        for c in 0..descr.num_columns() {
            let root = descr.get_column_root_idx(c);
            *leaves_per_root
                .get_mut(root)
                .ok_or_else(|| anyhow!("parquet leaf {c} has no arrow root field"))? += 1;
        }
        Ok(ColumnEncoder {
            file,
            factory,
            schema,
            leaves_per_root,
            max_rows,
            in_progress: None,
            pool,
        })
    }

    /// Encode `batch`, splitting it at the row-group cap exactly where
    /// `ArrowWriter::write` splits it.
    pub(crate) fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        if batch.num_columns() != self.schema.fields().len() {
            return Err(anyhow!(
                "batch has {} columns, the writer's schema {}",
                batch.num_columns(),
                self.schema.fields().len()
            ));
        }
        if self.in_progress.is_none() {
            let index = self.file.flushed_row_groups().len();
            let mut flat = self.factory.create_column_writers(index)?.into_iter();
            let roots = self
                .leaves_per_root
                .iter()
                .map(|&n| flat.by_ref().take(n).collect())
                .collect();
            self.in_progress = Some(RowGroup { roots, rows: 0 });
        }
        let buffered = self.in_progress.as_ref().map_or(0, |rg| rg.rows);
        if let Some(max) = self.max_rows {
            if buffered + batch.num_rows() > max {
                let to_write = max - buffered;
                self.write(&batch.slice(0, to_write))?;
                return self.write(&batch.slice(to_write, batch.num_rows() - to_write));
            }
        }
        let rg = self.in_progress.as_mut().expect("row group opened above");
        rg.rows += batch.num_rows();
        let fields = self.schema.fields();
        let columns = batch.columns();
        let encode = |(i, writers): (usize, &mut Vec<ArrowColumnWriter>)| -> Result<()> {
            let leaves = compute_leaves(fields[i].as_ref(), &columns[i])?;
            if leaves.len() != writers.len() {
                return Err(anyhow!(
                    "column '{}' has {} parquet leaves, the writer {}",
                    fields[i].name(),
                    leaves.len(),
                    writers.len()
                ));
            }
            for (w, leaf) in writers.iter_mut().zip(&leaves) {
                w.write(leaf)?;
            }
            Ok(())
        };
        {
            // The turn is released before `flush`, which takes its own.
            let turn = if rg.roots.len() > 1 {
                claim(&self.pool)
            } else {
                None
            };
            match turn {
                Some(turn) => {
                    turn.install(|| rg.roots.par_iter_mut().enumerate().try_for_each(encode))?
                }
                None => rg.roots.iter_mut().enumerate().try_for_each(encode)?,
            }
        }
        if self.max_rows.is_some_and(|max| rg.rows >= max) {
            self.flush()?;
        }
        Ok(())
    }

    /// Close the in-progress row group and append its column chunks in schema order.
    fn flush(&mut self) -> Result<()> {
        let Some(rg) = self.in_progress.take() else {
            return Ok(());
        };
        let close = |writers: Vec<ArrowColumnWriter>| -> Result<Vec<ArrowColumnChunk>> {
            writers
                .into_iter()
                .map(|w| w.close().map_err(anyhow::Error::from))
                .collect()
        };
        let turn = if rg.roots.len() > 1 {
            claim(&self.pool)
        } else {
            None
        };
        let chunks: Vec<Vec<ArrowColumnChunk>> = match turn {
            Some(turn) => turn.install(|| {
                rg.roots
                    .into_par_iter()
                    .map(close)
                    .collect::<Result<Vec<_>>>()
            })?,
            None => rg.roots.into_iter().map(close).collect::<Result<_>>()?,
        };
        let mut out = self.file.next_row_group()?;
        for chunk in chunks.into_iter().flatten() {
            chunk.append_to_row_group(&mut out)?;
        }
        out.close()?;
        Ok(())
    }

    /// Flush the last row group, write the footer and hand back the sink.
    pub(crate) fn into_inner(mut self) -> Result<W> {
        self.flush()?;
        self.file.into_inner().context("writing the parquet footer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        ArrayRef, Float32Array, Float64Array, Int32Array, LargeListArray, ListArray, StringArray,
    };
    use arrow::datatypes::{DataType, Field, Float32Type, Schema};
    use parquet::basic::Compression;
    use parquet::file::properties::EnabledStatistics;
    use parquet::schema::types::ColumnPath;

    /// A table with every shape the engine writes: required and nullable scalars of each
    /// width, a low-cardinality float, strings, and List and LargeList float columns with
    /// empty and null rows.
    fn batch(first: usize, n: usize) -> RecordBatch {
        let rows = first..first + n;
        let schema = Arc::new(Schema::new(vec![
            Field::new("candidate_id", DataType::Int32, false),
            Field::new("mz", DataType::Float64, false),
            Field::new("irt", DataType::Float32, true),
            Field::new("flag", DataType::Float64, false),
            Field::new("peptidoform", DataType::Utf8, false),
            Field::new(
                "rt",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ),
            Field::new(
                "intensity",
                DataType::LargeList(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]));
        let list = |r: usize| -> Option<Vec<Option<f32>>> {
            match r % 7 {
                0 => Some(Vec::new()),
                3 => None,
                k => Some((0..k).map(|j| Some((r * 3 + j) as f32 * 0.25)).collect()),
            }
        };
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from_iter_values(
                rows.clone().map(|r| (r / 3) as i32),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.clone().map(|r| r as f64 * 1.000_7),
            )),
            Arc::new(Float32Array::from_iter(
                rows.clone()
                    .map(|r| (r % 11 != 0).then_some(r as f32 * 0.5)),
            )),
            Arc::new(Float64Array::from_iter_values(
                rows.clone().map(|r| (r % 2) as f64),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.clone().map(|r| format!("PEP{}K", r % 97)),
            )),
            Arc::new(ListArray::from_iter_primitive::<Float32Type, _, _>(
                rows.clone().map(list),
            )),
            Arc::new(LargeListArray::from_iter_primitive::<Float32Type, _, _>(
                rows.clone().map(|r| Some(list(r).unwrap_or_default())),
            )),
        ];
        RecordBatch::try_new(schema, cols).unwrap()
    }

    fn props(cap: Option<usize>, plain_mz: bool) -> WriterProperties {
        let mut b = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(cap)
            .set_statistics_enabled(EnabledStatistics::Page);
        if plain_mz {
            b = b
                .set_column_dictionary_enabled(ColumnPath::from("mz"), false)
                .set_column_dictionary_page_size_limit(ColumnPath::from("rt.list.item"), 512)
                .set_data_page_row_count_limit(cap.unwrap_or(1 << 20));
        }
        b.build()
    }

    fn serial(batches: &[RecordBatch], props: WriterProperties) -> Vec<u8> {
        let mut out = Vec::new();
        let mut w = ArrowWriter::try_new(&mut out, batches[0].schema(), Some(props)).unwrap();
        for b in batches {
            w.write(b).unwrap();
        }
        w.close().unwrap();
        out
    }

    fn parallel(
        batches: &[RecordBatch],
        props: WriterProperties,
        pool: Option<Arc<CodecPool>>,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        let mut w = ColumnEncoder::try_new(&mut out, batches[0].schema(), props, pool).unwrap();
        for b in batches {
            w.write(b).unwrap();
        }
        w.into_inner().unwrap();
        out
    }

    /// The contract of the module: over several row groups, list columns, nulls, empty
    /// batches, batches that straddle a row-group boundary and batches that are not a
    /// multiple of the encoder's 1,024-value mini-batch, the parallel encoder writes the
    /// serial [`ArrowWriter`]'s file byte for byte, on a pool of any size and without one.
    #[test]
    fn the_parallel_encoder_writes_the_serial_writers_bytes() {
        let pools: Vec<Option<Arc<CodecPool>>> = vec![
            None,
            Some(CodecPool::new(1).unwrap()),
            Some(CodecPool::new(4).unwrap()),
        ];
        // (batch sizes, row-group cap, planned properties)
        let cases: Vec<(Vec<usize>, Option<usize>, bool)> = vec![
            (vec![5_000], Some(1_024), false),
            (
                vec![0, 1_000, 0, 3_333, 1, 0, 2_047, 2_500],
                Some(2_048),
                false,
            ),
            (vec![700; 13], Some(3_000), true),
            (vec![10_000, 10_000], None, false),
            (vec![4_096, 4_096, 4_096], Some(4_096), true),
            (vec![1], Some(1), false),
            (vec![0], Some(10), false),
        ];
        for (sizes, cap, planned) in cases {
            let mut first = 0usize;
            let batches: Vec<RecordBatch> = sizes
                .iter()
                .map(|&n| {
                    let b = batch(first, n);
                    first += n;
                    b
                })
                .collect();
            let want = serial(&batches, props(cap, planned));
            for pool in &pools {
                let got = parallel(&batches, props(cap, planned), pool.clone());
                assert!(
                    got == want,
                    "sizes {sizes:?} cap {cap:?} planned {planned} threads {:?}: {} bytes \
                     against the serial writer's {}",
                    pool.as_ref().map(|p| p.threads()),
                    got.len(),
                    want.len()
                );
            }
        }
    }

    /// A features-shaped table: many float columns of every cardinality, so the columns of a
    /// row group really are spread over the pool's threads, some falling back from their
    /// dictionary part-way through a chunk.
    fn wide(first: usize, n: usize) -> RecordBatch {
        let mut fields = Vec::new();
        let mut cols: Vec<ArrayRef> = Vec::new();
        for c in 0..48usize {
            fields.push(Field::new(format!("f{c}"), DataType::Float64, c % 5 == 0));
            let modulus = [2usize, 20, 700, 5_000, usize::MAX][c % 5];
            cols.push(Arc::new(Float64Array::from_iter((first..first + n).map(
                |r| {
                    let v = ((r * (c + 1) * 2_654_435_761) % modulus.min(1 << 40)) as f64 * 0.37;
                    (c % 5 != 0 || r % 13 != 0).then_some(v)
                },
            ))));
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
    }

    #[test]
    fn a_wide_table_on_a_pool_is_the_serial_writers_file() {
        let pool = Some(CodecPool::new(6).unwrap());
        let mut first = 0usize;
        let batches: Vec<RecordBatch> = [3_001usize, 0, 4_999, 12_000, 1, 7_777]
            .iter()
            .map(|&n| {
                let b = wide(first, n);
                first += n;
                b
            })
            .collect();
        for cap in [Some(5_000usize), Some(16_384), None] {
            let props = || {
                let mut b = WriterProperties::builder()
                    .set_compression(Compression::SNAPPY)
                    .set_max_row_group_row_count(cap)
                    .set_data_page_row_count_limit(cap.unwrap_or(20_000));
                for c in (0..48).step_by(3) {
                    b = b.set_column_dictionary_page_size_limit(
                        ColumnPath::from(format!("f{c}").as_str()),
                        4_096,
                    );
                }
                b.build()
            };
            let want = serial(&batches, props());
            let got = parallel(&batches, props(), pool.clone());
            assert!(
                got == want,
                "cap {cap:?}: {} bytes against {}",
                got.len(),
                want.len()
            );
        }
    }

    /// From inside a rayon pool the encoder does not wait on the codec pool (module docs),
    /// and the file is the same.
    #[test]
    fn a_writer_inside_a_rayon_pool_encodes_on_its_own_thread() {
        let codec = Some(CodecPool::new(3).unwrap());
        let outer = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let batches = vec![wide(0, 9_000), wide(9_000, 2_000)];
        let props = || {
            WriterProperties::builder()
                .set_max_row_group_row_count(Some(4_000))
                .build()
        };
        let want = serial(&batches, props());
        assert!(claim(&codec).is_some(), "a plain thread uses the pool");
        let got = outer.install(|| {
            assert!(claim(&codec).is_none(), "a rayon worker does not");
            parallel(&batches, props(), codec.clone())
        });
        assert!(got == want);
    }

    /// A pool gives out as many turns as it has threads and no more, so a caller beyond
    /// that works on its own thread, and a released turn is available again.
    #[test]
    fn a_saturated_pool_sends_the_next_caller_to_its_own_thread() {
        let codec = Some(CodecPool::new(2).unwrap());
        let a = claim(&codec).expect("first turn");
        let b = claim(&codec).expect("second turn");
        assert!(
            claim(&codec).is_none(),
            "a third caller encodes on its own thread"
        );
        drop(a);
        let c = claim(&codec).expect("a released turn is given out again");
        drop((b, c));
        assert!(claim(&codec).is_some());
        assert!(claim(&None).is_none(), "no pool, no turn");
    }

    /// More concurrent writers than the pool has threads: whichever of them get a turn and
    /// whichever encode on their own thread, every file is the serial writer's.
    #[test]
    fn concurrent_writers_beyond_the_pool_size_write_the_serial_writers_file() {
        let codec = Some(CodecPool::new(2).unwrap());
        let batches = vec![wide(0, 6_000), wide(6_000, 3_500)];
        let props = || {
            WriterProperties::builder()
                .set_max_row_group_row_count(Some(2_500))
                .build()
        };
        let want = serial(&batches, props());
        let files: Vec<Vec<u8>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..6)
                .map(|_| s.spawn(|| parallel(&batches, props(), codec.clone())))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for got in files {
            assert!(got == want);
        }
        assert_eq!(
            codec.as_ref().unwrap().callers.load(Ordering::Acquire),
            0,
            "every turn was released"
        );
    }

    /// The encoder refuses the row-group splits it does not reproduce rather than writing a
    /// file the serial writer would not have written.
    #[test]
    fn a_byte_cap_is_refused() {
        let props = WriterProperties::builder()
            .set_max_row_group_bytes(Some(1 << 20))
            .build();
        let b = batch(0, 10);
        assert!(ColumnEncoder::try_new(Vec::new(), b.schema(), props, None).is_err());
    }
}
