//! Coalesced row-group reads for wide projections (docs/03_io_layer.md, "Sequential
//! row-group reads").
//!
//! parquet-rs's synchronous reader fetches every page on its own: one read for the header
//! and one for the body (`serialized_reader.rs`, `get_read` then `get_bytes`), through a
//! fresh seek each time. The arrow reader advances every projected column in lockstep, one
//! batch at a time, so a batch of a 398-column table issues about 400 page reads, each one
//! column chunk away from the previous: a row group is covered in several strided passes
//! instead of one forward sweep. That costs nothing on an SSD and is seek-bound on a
//! spinning array, where one reader of the immunopeptidomics competed tables measured
//! 44 MB/s against a 133 MB/s sequential ceiling.
//!
//! [`SpanCache`] is a [`ChunkReader`] that reads each selected row group's projected column
//! chunks as one byte span, with one sequential read, and serves every page request inside
//! it from memory. The decoder receives the same bytes through the same metadata, so the
//! decoded batches are identical to the plain [`std::fs::File`] reader's.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};

use bytes::{Buf, Bytes};
use parquet::errors::{ParquetError, Result as PqResult};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::reader::{ChunkReader, Length};

/// How [`SpanCache`] plans and holds its spans.
#[derive(Clone, Debug)]
pub struct SpanReadOptions {
    /// Two projected column chunks of one row group separated by at most this many bytes
    /// of unprojected columns are read as one span, the gap included: a short gap costs
    /// less than a seek. A larger gap splits the row group into separate spans, so a narrow
    /// projection never reads the columns it skipped wholesale.
    pub max_gap_bytes: u64,
    /// A span larger than this is not cached: its pages are read directly, one at a time,
    /// exactly as the plain reader reads them. It bounds the memory one span can take.
    pub max_span_bytes: u64,
    /// The spans held at once, prefetched ones included, may not exceed this. A span is
    /// released as soon as every column chunk in it has been read to its end. The decoder
    /// never waits on the budget: a span it needs is read even when the budget is full,
    /// and only the prefetch is skipped.
    pub max_resident_bytes: u64,
    /// Read the next span on a helper thread while the decoder works through this one.
    pub prefetch: bool,
}

impl Default for SpanReadOptions {
    fn default() -> Self {
        SpanReadOptions {
            max_gap_bytes: 1 << 20,
            max_span_bytes: 512 << 20,
            max_resident_bytes: 1 << 30,
            prefetch: true,
        }
    }
}

/// One byte range of the file that is read as a whole.
#[derive(Debug)]
struct Span {
    start: u64,
    end: u64,
    /// End offsets of the column chunks inside this span. A read that ends at one of them
    /// is the last page of that chunk, which is how the cache learns a span is finished.
    chunk_ends: Vec<u64>,
    /// Over `max_span_bytes`: served by direct reads, never held.
    direct: bool,
}

#[derive(Debug)]
enum State {
    Idle,
    Loading,
    Ready(Bytes),
}

struct Slot {
    state: State,
    /// Column chunks of the span read to their end since it was loaded.
    finished: usize,
}

struct Shared {
    spans: Vec<Span>,
    slots: Mutex<Vec<Slot>>,
    loaded: Condvar,
    options: SpanReadOptions,
    stats: Stats,
    /// One past the highest span the decoder has asked for. Only written and read under
    /// the `slots` lock. A prefetch request for a span below it is stale: the decoder got
    /// there first, and loading the span again would hold bytes nobody reads.
    reached: AtomicUsize,
}

/// What the cache did, for the debug line it logs when it is dropped and for the tests.
#[derive(Default)]
struct Stats {
    /// Spans read whole, by the decoder or by the helper.
    span_reads: AtomicU64,
    span_bytes: AtomicU64,
    /// Spans the helper read ahead of the decoder.
    prefetched: AtomicU64,
    /// Requests that fell outside every cacheable span and were read from the file.
    direct_reads: AtomicU64,
    /// Requests served from span bytes.
    served: AtomicU64,
}

impl Stats {
    fn count_span(&self, bytes: u64, prefetched: bool) {
        self.span_reads.fetch_add(1, Ordering::Relaxed);
        self.span_bytes.fetch_add(bytes, Ordering::Relaxed);
        if prefetched {
            self.prefetched.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Shared {
    fn resident(slots: &[Slot], spans: &[Span]) -> u64 {
        slots
            .iter()
            .zip(spans)
            .filter(|(s, _)| !matches!(s.state, State::Idle))
            .map(|(_, sp)| sp.end - sp.start)
            .sum()
    }
}

/// Read one span of `file` with a single positioned read.
fn read_span(file: &File, start: u64, end: u64) -> std::io::Result<Bytes> {
    let mut f = file.try_clone()?;
    f.seek(SeekFrom::Start(start))?;
    let len = (end - start) as usize;
    // Read into reserved capacity rather than a zeroed buffer: zero-filling hundreds of
    // megabytes before overwriting them is a measurable share of a span read from cache.
    let mut buf = Vec::with_capacity(len);
    let n = f.take(len as u64).read_to_end(&mut buf)?;
    if n != len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("span {start}..{end}: read {n} of {len} bytes"),
        ));
    }
    Ok(Bytes::from(buf))
}

/// A [`ChunkReader`] over a parquet file that reads the planned byte spans whole.
///
/// Build it with [`SpanCache::plan`] for the row groups and leaf columns the reader will
/// decode, then hand it to `ParquetRecordBatchReaderBuilder::new_with_metadata` in place of
/// the `File`. A request that lies inside a planned span is sliced out of the span's bytes;
/// anything else (the footer, a column that was not planned, an oversized span) is read
/// directly, as the plain reader would.
pub struct SpanCache {
    file: File,
    len: u64,
    shared: Arc<Shared>,
    prefetch: Option<mpsc::Sender<usize>>,
    helper: Option<std::thread::JoinHandle<()>>,
}

impl SpanCache {
    /// Plan the spans of `row_groups` (file row-group indices, in the order they will be
    /// read) over the parquet leaf columns `leaves`, and open `path` for them.
    pub fn plan(
        path: &str,
        meta: &ParquetMetaData,
        row_groups: &[usize],
        leaves: &[usize],
        options: SpanReadOptions,
    ) -> anyhow::Result<SpanCache> {
        let file = File::open(path).map_err(|e| anyhow::anyhow!("opening {path}: {e}"))?;
        let len = file.metadata()?.len();
        let mut spans: Vec<Span> = Vec::new();
        for &rg in row_groups {
            let group = meta.row_group(rg);
            let mut ranges: Vec<(u64, u64)> = leaves
                .iter()
                .map(|&c| {
                    let (start, n) = group.column(c).byte_range();
                    (start, start + n)
                })
                .filter(|(s, e)| e > s)
                .collect();
            ranges.sort_unstable();
            let mut open: Option<Span> = None;
            for (s, e) in ranges {
                match open.as_mut() {
                    Some(sp) if s <= sp.end + options.max_gap_bytes && s >= sp.end => {
                        sp.end = sp.end.max(e);
                        sp.chunk_ends.push(e);
                    }
                    _ => {
                        if let Some(sp) = open.take() {
                            spans.push(sp);
                        }
                        open = Some(Span {
                            start: s,
                            end: e,
                            chunk_ends: vec![e],
                            direct: false,
                        });
                    }
                }
            }
            if let Some(sp) = open.take() {
                spans.push(sp);
            }
        }
        // Row groups are planned in reading order, which for this engine's files is also
        // file order; lookups need the spans sorted by offset, and they never overlap
        // because column chunks do not.
        spans.sort_by_key(|s| s.start);
        for sp in &mut spans {
            sp.chunk_ends.sort_unstable();
            sp.direct = sp.end - sp.start > options.max_span_bytes;
        }
        let slots = spans
            .iter()
            .map(|_| Slot {
                state: State::Idle,
                finished: 0,
            })
            .collect();
        let shared = Arc::new(Shared {
            spans,
            slots: Mutex::new(slots),
            loaded: Condvar::new(),
            options,
            stats: Stats::default(),
            reached: AtomicUsize::new(0),
        });
        let (prefetch, helper) = if shared.options.prefetch && shared.spans.len() > 1 {
            // The helper has its own handle: a cloned `File` shares the cursor of the one
            // it was cloned from, and the decoder's direct reads seek the main handle.
            let own = File::open(path).map_err(|e| anyhow::anyhow!("opening {path}: {e}"))?;
            let (tx, rx) = mpsc::channel::<usize>();
            let sh = shared.clone();
            let h = std::thread::Builder::new()
                .name("mumdia-span-prefetch".into())
                .spawn(move || {
                    for i in rx {
                        prefetch_one(&sh, &own, i);
                    }
                })?;
            (Some(tx), Some(h))
        } else {
            (None, None)
        };
        Ok(SpanCache {
            file,
            len,
            shared,
            prefetch,
            helper,
        })
    }

    /// The span that holds `[start, start + len)` entirely, if one was planned and is
    /// cacheable.
    fn find(&self, start: u64, len: u64) -> Option<usize> {
        let spans = &self.shared.spans;
        let i = spans.partition_point(|s| s.start <= start).checked_sub(1)?;
        let s = &spans[i];
        (!s.direct && start + len <= s.end).then_some(i)
    }

    /// The bytes of span `i`, reading them if nobody has, waiting if the helper is. The
    /// first request for a span asks the helper for the next one.
    fn acquire(&self, i: usize) -> PqResult<Bytes> {
        let sh = &self.shared;
        let mut slots = sh.slots.lock().expect("span cache lock");
        if sh.reached.fetch_max(i + 1, Ordering::Relaxed) <= i {
            self.request_prefetch(i + 1);
        }
        loop {
            match &slots[i].state {
                State::Ready(b) => return Ok(b.clone()),
                State::Loading => {
                    slots = sh.loaded.wait(slots).expect("span cache lock");
                }
                State::Idle => {
                    slots[i].state = State::Loading;
                    slots[i].finished = 0;
                    drop(slots);
                    let sp = &sh.spans[i];
                    let got = read_span(&self.file, sp.start, sp.end);
                    if got.is_ok() {
                        sh.stats.count_span(sp.end - sp.start, false);
                    }
                    let mut slots = sh.slots.lock().expect("span cache lock");
                    return match got {
                        Ok(b) => {
                            slots[i].state = State::Ready(b.clone());
                            sh.loaded.notify_all();
                            Ok(b)
                        }
                        Err(e) => {
                            slots[i].state = State::Idle;
                            sh.loaded.notify_all();
                            Err(ParquetError::External(Box::new(e)))
                        }
                    };
                }
            }
        }
    }

    fn request_prefetch(&self, next: usize) {
        if let Some(tx) = &self.prefetch {
            if next < self.shared.spans.len() {
                let _ = tx.send(next);
            }
        }
    }

    /// Record that a read ended at `end`; release the span once each of its column chunks
    /// has been read to its end. A forward scan never returns to a finished chunk, and if
    /// something does, the span is simply read again.
    fn note_read(&self, i: usize, end: u64) {
        let sh = &self.shared;
        if sh.spans[i].chunk_ends.binary_search(&end).is_err() {
            return;
        }
        let mut slots = sh.slots.lock().expect("span cache lock");
        let slot = &mut slots[i];
        if !matches!(slot.state, State::Ready(_)) {
            return;
        }
        slot.finished += 1;
        if slot.finished >= sh.spans[i].chunk_ends.len() {
            slot.state = State::Idle;
            slot.finished = 0;
        }
    }

    fn direct_bytes(&self, start: u64, length: usize) -> PqResult<Bytes> {
        self.shared
            .stats
            .direct_reads
            .fetch_add(1, Ordering::Relaxed);
        self.file.get_bytes(start, length)
    }
}

/// The helper thread's half of [`SpanCache::acquire`]: read span `i` ahead of the decoder
/// when it is idle, cacheable, still ahead of the decoder and fits the resident budget.
fn prefetch_one(sh: &Shared, file: &File, i: usize) {
    {
        let mut slots = sh.slots.lock().expect("span cache lock");
        let sp = &sh.spans[i];
        if sp.direct
            || !matches!(slots[i].state, State::Idle)
            || i < sh.reached.load(Ordering::Relaxed)
        {
            return;
        }
        if Shared::resident(&slots, &sh.spans) + (sp.end - sp.start) > sh.options.max_resident_bytes
        {
            return;
        }
        slots[i].state = State::Loading;
        slots[i].finished = 0;
    }
    let sp = &sh.spans[i];
    let got = read_span(file, sp.start, sp.end);
    if got.is_ok() {
        sh.stats.count_span(sp.end - sp.start, true);
    }
    let mut slots = sh.slots.lock().expect("span cache lock");
    // A failed prefetch leaves the span idle; the decoder then reads it itself and reports
    // the error from its own read.
    slots[i].state = match got {
        Ok(b) => State::Ready(b),
        Err(_) => State::Idle,
    };
    sh.loaded.notify_all();
}

impl Drop for SpanCache {
    fn drop(&mut self) {
        // Closing the channel ends the helper's loop after its current span.
        self.prefetch.take();
        if let Some(h) = self.helper.take() {
            let _ = h.join();
        }
        let st = &self.shared.stats;
        tracing::debug!(
            spans = self.shared.spans.len(),
            span_reads = st.span_reads.load(Ordering::Relaxed),
            span_mb = st.span_bytes.load(Ordering::Relaxed) as f64 / 1e6,
            prefetched = st.prefetched.load(Ordering::Relaxed),
            served_from_spans = st.served.load(Ordering::Relaxed),
            direct_reads = st.direct_reads.load(Ordering::Relaxed),
            "span cache: done"
        );
    }
}

impl Length for SpanCache {
    fn len(&self) -> u64 {
        self.len
    }
}

/// A page reader over either cached span bytes or the file.
pub enum SpanRead {
    Mem(bytes::buf::Reader<Bytes>),
    File(BufReader<File>),
}

impl Read for SpanRead {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            SpanRead::Mem(r) => r.read(buf),
            SpanRead::File(r) => r.read(buf),
        }
    }
}

impl ChunkReader for SpanCache {
    type T = SpanRead;

    fn get_read(&self, start: u64) -> PqResult<SpanRead> {
        if let Some(i) = self.find(start, 1) {
            let bytes = self.acquire(i)?;
            self.shared.stats.served.fetch_add(1, Ordering::Relaxed);
            let off = (start - self.shared.spans[i].start) as usize;
            return Ok(SpanRead::Mem(bytes.slice(off..).reader()));
        }
        self.shared
            .stats
            .direct_reads
            .fetch_add(1, Ordering::Relaxed);
        Ok(SpanRead::File(self.file.get_read(start)?))
    }

    fn get_bytes(&self, start: u64, length: usize) -> PqResult<Bytes> {
        if let Some(i) = self.find(start, length as u64) {
            let bytes = self.acquire(i)?;
            self.shared.stats.served.fetch_add(1, Ordering::Relaxed);
            let off = (start - self.shared.spans[i].start) as usize;
            let out = bytes.slice(off..off + length);
            self.note_read(i, start + length as u64);
            return Ok(out);
        }
        self.direct_bytes(start, length)
    }
}

#[cfg(test)]
impl SpanCache {
    /// Spans planned, and how many of them are cacheable.
    fn plan_shape(&self) -> (usize, usize) {
        let spans = &self.shared.spans;
        (spans.len(), spans.iter().filter(|s| !s.direct).count())
    }

    /// Bytes currently held.
    fn resident_bytes(&self) -> u64 {
        let slots = self.shared.slots.lock().unwrap();
        Shared::resident(&slots, &self.shared.spans)
    }

    /// (span reads, direct reads).
    fn reads(&self) -> (u64, u64) {
        let st = &self.shared.stats;
        (
            st.span_reads.load(Ordering::Relaxed),
            st.direct_reads.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::{Col, ScanOptions, TableFile, TableWriter};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};

    /// A shared handle, so a test can inspect the cache while a reader owns it.
    struct Handle(Arc<SpanCache>);

    impl Length for Handle {
        fn len(&self) -> u64 {
            self.0.len()
        }
    }

    impl ChunkReader for Handle {
        type T = SpanRead;
        fn get_read(&self, start: u64) -> PqResult<SpanRead> {
            self.0.get_read(start)
        }
        fn get_bytes(&self, start: u64, length: usize) -> PqResult<Bytes> {
            self.0.get_bytes(start, length)
        }
    }

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia_span_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// 14 columns, scalars and lists, several row groups, written by the engine's writer.
    fn fixture(path: &str, rows: usize, rg: usize) {
        let mut cols = vec![
            Col::U32("candidate_id".into(), (0..rows as u32).collect()),
            Col::Str(
                "label".into(),
                (0..rows)
                    .map(|i| if i % 2 == 0 { "target" } else { "decoy" }.to_string())
                    .collect(),
            ),
        ];
        for j in 0..10 {
            cols.push(Col::F64(
                format!("f{j}"),
                (0..rows).map(|i| (i * (j + 3)) as f64 * 0.37).collect(),
            ));
        }
        cols.push(Col::OptF32(
            "im".into(),
            (0..rows)
                .map(|i| (i % 7 != 3).then_some(i as f32))
                .collect(),
        ));
        cols.push(Col::LargeListF32(
            "trace".into(),
            (0..rows)
                .map(|i| (0..(i % 11)).map(|k| (i + k) as f32).collect())
                .collect(),
        ));
        let mut w = TableWriter::new(path).with_row_group_rows(rg);
        w.write_cols(cols).unwrap();
        w.close().unwrap();
    }

    fn collect(
        t: &TableFile,
        cols: Option<&[&str]>,
        bs: usize,
        o: &ScanOptions,
    ) -> Vec<RecordBatch> {
        t.scan(cols, bs, o)
            .unwrap()
            .collect::<anyhow::Result<Vec<_>>>()
            .unwrap()
    }

    #[test]
    fn coalesced_reads_yield_the_plain_readers_batches() {
        let d = dir("same");
        let p = d.join("t.parquet").to_string_lossy().to_string();
        fixture(&p, 10_000, 1_500);
        let whole = TableFile::open(&p).unwrap();
        let spans = [
            whole.span(0, 10_000).unwrap(),
            whole.span(1_234, 5_000).unwrap(),
            whole.span(2_999, 2).unwrap(),
            whole.span(9_999, 1).unwrap(),
        ];
        let options = [
            SpanReadOptions::default(),
            SpanReadOptions {
                prefetch: false,
                ..SpanReadOptions::default()
            },
            // Every span over the limit: all pages read directly.
            SpanReadOptions {
                max_span_bytes: 1,
                ..SpanReadOptions::default()
            },
            // No gap tolerated, and a budget too small to prefetch anything.
            SpanReadOptions {
                max_gap_bytes: 0,
                max_resident_bytes: 1,
                ..SpanReadOptions::default()
            },
        ];
        let projections: [Option<&[&str]>; 4] = [
            None,
            Some(&["f3", "trace"]),
            Some(&["candidate_id", "f9", "im"]),
            Some(&["label"]),
        ];
        for t in std::iter::once(&whole).chain(spans.iter()) {
            for cols in projections {
                // Batch sizes that divide the row group, that straddle row groups, and one
                // larger than several row groups.
                for bs in [500usize, 1_024, 4_096] {
                    let plain = collect(t, cols, bs, &ScanOptions::default());
                    for o in &options {
                        let so = ScanOptions {
                            coalesce: Some(o.clone()),
                            decode_threads: Some(1),
                        };
                        assert_eq!(
                            collect(t, cols, bs, &so),
                            plain,
                            "{cols:?} batch {bs} under {o:?} on {} rows",
                            t.nrows
                        );
                    }
                }
            }
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn each_span_is_read_once_and_released_when_its_chunks_are_done() {
        let d = dir("once");
        let p = d.join("t.parquet").to_string_lossy().to_string();
        fixture(&p, 12_000, 2_000);
        let file = File::open(&p).unwrap();
        let am = ArrowReaderMetadata::load(&file, Default::default()).unwrap();
        let meta = am.metadata().clone();
        let n_rg = meta.num_row_groups();
        let leaves: Vec<usize> = (0..meta.file_metadata().schema_descr().num_columns()).collect();
        for prefetch in [false, true] {
            let cache = Arc::new(
                SpanCache::plan(
                    &p,
                    &meta,
                    &(0..n_rg).collect::<Vec<_>>(),
                    &leaves,
                    SpanReadOptions {
                        prefetch,
                        ..SpanReadOptions::default()
                    },
                )
                .unwrap(),
            );
            // All columns of a row group are adjacent: one span per row group.
            assert_eq!(cache.plan_shape(), (n_rg, n_rg));
            let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(
                Handle(cache.clone()),
                am.clone(),
            )
            .with_batch_size(700)
            .build()
            .unwrap();
            let mut rows = 0;
            let largest = (0..n_rg)
                .map(|i| meta.row_group(i).compressed_size() as u64)
                .max()
                .unwrap();
            for b in reader {
                rows += b.unwrap().num_rows();
                // Never more than the two row groups one batch can straddle plus the one
                // being prefetched.
                assert!(cache.resident_bytes() <= 3 * largest);
            }
            assert_eq!(rows, 12_000);
            let (span_reads, direct) = cache.reads();
            assert_eq!(
                span_reads, n_rg as u64,
                "prefetch {prefetch}: one read per span"
            );
            assert_eq!(
                direct, 0,
                "prefetch {prefetch}: every page served from a span"
            );
            assert_eq!(cache.resident_bytes(), 0, "every span released at the end");
        }
        // A narrow projection with wide gaps between its columns plans separate spans.
        let cache = SpanCache::plan(
            &p,
            &meta,
            &[0, 1],
            &[0, 13],
            SpanReadOptions {
                max_gap_bytes: 0,
                ..SpanReadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(cache.plan_shape(), (4, 4));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A prefetch request that reaches the helper after the decoder has already read and
    /// released its span must not load the span again: nobody would read it, so it would
    /// stay resident until the cache is dropped. On a loaded host the helper can fall that
    /// far behind; this replays the late request directly.
    #[test]
    fn a_late_prefetch_does_not_reload_a_finished_span() {
        let d = dir("late");
        let p = d.join("t.parquet").to_string_lossy().to_string();
        fixture(&p, 6_000, 2_000);
        let file = File::open(&p).unwrap();
        let am = ArrowReaderMetadata::load(&file, Default::default()).unwrap();
        let meta = am.metadata().clone();
        let n_rg = meta.num_row_groups();
        let leaves: Vec<usize> = (0..meta.file_metadata().schema_descr().num_columns()).collect();
        let cache = SpanCache::plan(
            &p,
            &meta,
            &(0..n_rg).collect::<Vec<_>>(),
            &leaves,
            SpanReadOptions {
                prefetch: false,
                ..SpanReadOptions::default()
            },
        )
        .unwrap();
        // Read span 0 to the end of every column chunk in it, so it is released.
        let sp = &cache.shared.spans[0];
        let (start, ends) = (sp.start, sp.chunk_ends.clone());
        let mut from = start;
        for end in ends {
            cache.get_bytes(from, (end - from) as usize).unwrap();
            from = end;
        }
        assert_eq!(cache.resident_bytes(), 0);
        assert_eq!(cache.reads(), (1, 0));
        // The late request for span 0, and a timely one for the next span.
        prefetch_one(&cache.shared, &file, 0);
        assert_eq!(
            cache.resident_bytes(),
            0,
            "a finished span is not loaded again"
        );
        assert_eq!(cache.reads(), (1, 0));
        prefetch_one(&cache.shared, &file, 1);
        assert!(
            cache.resident_bytes() > 0,
            "a span ahead of the decoder is prefetched"
        );
        assert_eq!(cache.reads(), (2, 0));
        drop(cache);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The full-projection scan of a real artifact under each read option, timed, with the
    /// batches checked against the plain reader's:
    ///
    /// ```text
    /// MUMDIA_BENCH_PARQUET=out_aif02/psms_competed.parquet \
    ///   cargo test -p mumdia-io --release -- --ignored --nocapture bench_scan
    /// ```
    ///
    /// On a local SSD or from the page cache the arms should tie; the coalesced read is for
    /// spinning storage, which this bench cannot emulate. The arms are interleaved over
    /// five rounds and the median is reported, because a single shot on a shared host
    /// moved by 40% between two runs of the identical arm. The equality check zips each
    /// arm against the plain reader batch by batch, outside the timed passes, so no pass
    /// holds the decoded table.
    #[test]
    #[ignore = "benchmark; needs MUMDIA_BENCH_PARQUET"]
    fn bench_scan_a_real_artifact() {
        let Ok(src) = std::env::var("MUMDIA_BENCH_PARQUET") else {
            println!("set MUMDIA_BENCH_PARQUET to a real artifact to run this");
            return;
        };
        let t = TableFile::open(&src).unwrap();
        let serial = ScanOptions::default().with_decode_threads(1);
        let arms: Vec<(&str, ScanOptions)> = vec![
            ("plain, one reader", serial.clone()),
            (
                "plain, 2 decode groups",
                serial.clone().with_decode_threads(2),
            ),
            (
                "plain, 4 decode groups",
                serial.clone().with_decode_threads(4),
            ),
            (
                "plain, 8 decode groups",
                serial.clone().with_decode_threads(8),
            ),
            ("plain, automatic", ScanOptions::default()),
            (
                "coalesced, one reader",
                ScanOptions::coalesced().with_decode_threads(1),
            ),
            ("coalesced, automatic", ScanOptions::coalesced()),
            (
                "coalesced, no prefetch, one reader",
                ScanOptions {
                    coalesce: Some(SpanReadOptions {
                        prefetch: false,
                        ..SpanReadOptions::default()
                    }),
                    decode_threads: Some(1),
                },
            ),
        ];
        let bs = 1 << 14;
        for (label, o) in &arms[1..] {
            let a = t.scan(None, bs, &serial).unwrap();
            let b = t.scan(None, bs, o).unwrap();
            for (x, y) in a.zip(b) {
                assert_eq!(x.unwrap(), y.unwrap(), "{label} decoded a different batch");
            }
        }
        let mut secs: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for _ in 0..5 {
            for (k, (_, o)) in arms.iter().enumerate() {
                let t0 = std::time::Instant::now();
                let mut rows = 0usize;
                for b in t.scan(None, bs, o).unwrap() {
                    rows += b.unwrap().num_rows();
                }
                assert_eq!(rows, t.nrows);
                secs[k].push(t0.elapsed().as_secs_f64());
            }
        }
        for ((label, o), s) in arms.iter().zip(secs.iter_mut()) {
            s.sort_by(f64::total_cmp);
            let groups = t.scan(None, bs, o).unwrap().decode_groups();
            println!(
                "{label} ({groups} readers): {} rows, median {:.2} s (min {:.2}, max {:.2})",
                t.nrows, s[2], s[0], s[4]
            );
        }
    }
}
