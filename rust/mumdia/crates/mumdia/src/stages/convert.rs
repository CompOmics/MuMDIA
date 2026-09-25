//! Stage 0 `mumdia convert`: read an mzML run into the normalized spectra
//! artifact set (docs/04_convert.md). MVP is mzML-only and 3D, so ion-mobility
//! columns are absent. Profile spectra are centroided (simple local-maxima)
//! so downstream matching sees discrete peaks.

use std::io::SeekFrom;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, TableWriter};
use mzdata::io::mzml::MzMLReaderType;
use mzdata::io::{DetailLevel, MZReaderType};
use mzdata::prelude::*;
use mzdata::spectrum::SignalContinuity;
use serde_json::json;
use tracing::{debug, info, warn};

/// Centroid a profile spectrum by local maxima with 3-point parabolic m/z
/// refinement. Peaks below `noise_floor` (relative to the max) are dropped.
fn centroid(mz: &[f64], inten: &[f32]) -> (Vec<f64>, Vec<f32>) {
    // `.min()`, not `mz.len()`. The two arrays are decoded independently from the file
    // and either decode is allowed to fail (`unwrap_or_default` in `peaks_of`), while
    // mzdata never checks `defaultArrayLength` on read. So a profile spectrum with at
    // least 3 m/z values and a shorter or undecodable intensity array indexed
    // `inten[i + 1]` out of bounds and panicked on the first iteration. Checked Rust, so
    // the outcome was always a panic rather than an out-of-bounds read; still a crash on
    // a plain `mumdia convert` of a damaged file. `spectra.rs:68` already had this guard.
    let n = mz.len().min(inten.len());
    if n < 3 {
        return (mz.to_vec(), inten.to_vec());
    }
    let max_i = inten.iter().cloned().fold(0.0f32, f32::max);
    let floor = max_i * 1e-4;
    let mut out_mz = Vec::new();
    let mut out_in = Vec::new();
    for i in 1..n - 1 {
        let y0 = inten[i - 1];
        let y1 = inten[i];
        let y2 = inten[i + 1];
        if y1 <= floor || !(y1 >= y0 && y1 > y2) {
            continue;
        }
        // Parabolic peak apex refinement on m/z.
        let denom = (y0 - 2.0 * y1 + y2) as f64;
        let delta = if denom.abs() > 1e-12 {
            0.5 * (y0 - y2) as f64 / denom
        } else {
            0.0
        };
        let spacing = (mz[i + 1] - mz[i - 1]) * 0.5;
        let cm = mz[i] + delta * spacing;
        out_mz.push(cm);
        out_in.push(y1);
    }
    if out_mz.is_empty() {
        (mz.to_vec(), inten.to_vec())
    } else {
        (out_mz, out_in)
    }
}

/// Extract (m/z, intensity) as centroided, m/z-sorted, non-zero peaks, capped
/// to `top_n` most intense (0 = no cap).
fn peaks_of<S: SpectrumLike>(spec: &S, top_n: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let (mut mz, mut inten): (Vec<f64>, Vec<f32>) = match spec.raw_arrays() {
        Some(arrays) => {
            let m = arrays.mzs().map(|c| c.to_vec()).unwrap_or_default();
            let it = arrays.intensities().map(|c| c.to_vec()).unwrap_or_default();
            // Truncate both to the shorter length rather than carrying a mismatch
            // downstream. `zip` below would silently drop the tail anyway; doing it here
            // means `centroid` and every later reader see a consistent pair.
            let k = m.len().min(it.len());
            (m[..k].to_vec(), it[..k].to_vec())
        }
        None => (Vec::new(), Vec::new()),
    };
    if spec.signal_continuity() == SignalContinuity::Profile {
        let (cm, ci) = centroid(&mz, &inten);
        mz = cm;
        inten = ci;
    }
    // Drop zero/negative intensity, and NON-FINITE m/z or intensity.
    //
    // The intensity filter `*i > 0.0` is already false for NaN, so a NaN intensity was
    // dropped by accident. A NaN m/z with a positive intensity was not, and reached the
    // sorts below, where `partial_cmp(...).unwrap()` panics: a single malformed value in
    // an mzML aborted `mumdia convert` with an unwrap message naming neither the spectrum
    // nor the value. An infinite m/z was worse than a panic -- it survived, and one
    // non-finite fragment m/z collapses the whole fragment index range (see
    // `FragIndex::build`).
    //
    // Dropping rather than erroring, because a peak list is a measurement and one bad
    // peak in one spectrum is not a reason to refuse a whole run; the caller reports the
    // count so the loss is visible rather than silent.
    let n_before = mz.len();
    let mut pairs: Vec<(f64, f32)> = mz
        .into_iter()
        .zip(inten)
        .filter(|(m, i)| *i > 0.0 && m.is_finite() && i.is_finite())
        .collect();
    let dropped_nonfinite = n_before.saturating_sub(pairs.len());
    if top_n > 0 && pairs.len() > top_n {
        // `total_cmp` rather than `partial_cmp(..).unwrap()`. Every value here is finite
        // by the filter above, so the two agree; `total_cmp` is a genuine total order, so
        // it cannot panic and cannot trip the "comparison function does not correctly
        // implement a total order" check that `sort_by` has had since Rust 1.81.
        pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
        pairs.truncate(top_n);
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let out_mz: Vec<f32> = pairs.iter().map(|(m, _)| *m as f32).collect();
    let out_in: Vec<f32> = pairs.iter().map(|(_, i)| *i).collect();
    (out_mz, out_in, dropped_nonfinite)
}

pub struct ConvertParams<'a> {
    pub mzml: &'a str,
    pub out_dir: &'a str,
    pub max_spectra: usize,
    pub top_peaks_ms2: usize,
    pub top_peaks_ms1: usize,
    pub config_hash: &'a str,
}

/// Result paths for chaining.
pub struct ConvertOutputs {
    pub ms1: String,
    pub ms2: String,
    pub isolation_windows: String,
    pub ms2_to_ms1: String,
    /// The content hash each artifact's `<artifact>.report.json` records, so an
    /// orchestrator can record the four artifacts without reading and hashing them again.
    pub hashes: ConvertHashes,
}

/// The report content hash of each convert artifact, field for field with the paths of
/// [`ConvertOutputs`].
pub struct ConvertHashes {
    pub ms1: String,
    pub ms2: String,
    pub isolation_windows: String,
    pub ms2_to_ms1: String,
}

/// Spectra per flushed chunk and per parquet row group. The peak columns are
/// `LargeListF32`: a 32-bit arrow list offset saturates above 2^31-1 values, and the
/// readers accept either width.
///
/// Spectra per flushed chunk and per parquet row group. MS2 scans at ~2,000 peaks are
/// ~8 MB per list column per chunk before compression, so the conversion's resident set is
/// a few tens of MB regardless of run length.
const SPECTRA_CHUNK: usize = 2048;

/// Column accumulators for one chunk of MS1 scans. `cols()` drains them into exactly the
/// column set `spectra_ms1.parquet` has always had.
#[derive(Default)]
struct Ms1Chunk {
    idx: Vec<u32>,
    rt: Vec<f64>,
    mz: Vec<Vec<f32>>,
    inten: Vec<Vec<f32>>,
}

impl Ms1Chunk {
    fn cols(&mut self) -> Vec<Col> {
        vec![
            Col::U32("scan_index".into(), std::mem::take(&mut self.idx)),
            Col::F64("rt_seconds".into(), std::mem::take(&mut self.rt)),
            Col::LargeListF32("mz".into(), std::mem::take(&mut self.mz)),
            Col::LargeListF32("intensity".into(), std::mem::take(&mut self.inten)),
        ]
    }
}

/// Column accumulators for one chunk of MS2 scans (the `spectra_ms2.parquet` columns).
#[derive(Default)]
struct Ms2Chunk {
    idx: Vec<u32>,
    id: Vec<String>,
    rt: Vec<f64>,
    win_id: Vec<u32>,
    wt: Vec<f64>,
    wl: Vec<f64>,
    wu: Vec<f64>,
    pmz: Vec<Option<f64>>,
    pz: Vec<Option<i32>>,
    mz: Vec<Vec<f32>>,
    inten: Vec<Vec<f32>>,
}

impl Ms2Chunk {
    fn cols(&mut self) -> Vec<Col> {
        vec![
            Col::U32("scan_index".into(), std::mem::take(&mut self.idx)),
            Col::Str("id".into(), std::mem::take(&mut self.id)),
            Col::F64("rt_seconds".into(), std::mem::take(&mut self.rt)),
            Col::U32("window_id".into(), std::mem::take(&mut self.win_id)),
            Col::F64("window_target".into(), std::mem::take(&mut self.wt)),
            Col::F64("window_lower".into(), std::mem::take(&mut self.wl)),
            Col::F64("window_upper".into(), std::mem::take(&mut self.wu)),
            Col::OptF64("precursor_mz".into(), std::mem::take(&mut self.pmz)),
            Col::OptI32("precursor_charge".into(), std::mem::take(&mut self.pz)),
            Col::LargeListF32("mz".into(), std::mem::take(&mut self.mz)),
            Col::LargeListF32("intensity".into(), std::mem::take(&mut self.inten)),
        ]
    }
}

/// The `count` attribute of `<spectrumList>`, read from the head of an mzML.
///
/// `None` when the file is compressed, the attribute is absent, or the head cannot be
/// read: the caller then skips the completeness check rather than guessing.
fn declared_spectrum_count(path: &str) -> Option<usize> {
    use std::io::Read as _;
    let mut head = vec![0u8; 1 << 20];
    let mut f = std::fs::File::open(path).ok()?;
    let n = f.read(&mut head).ok()?;
    let text = String::from_utf8_lossy(&head[..n]);
    let at = text.find("<spectrumList")?;
    let rest = &text[at..];
    let c = rest.find("count=")? + "count=".len();
    let rest = rest[c..].trim_start();
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let digits: String = rest[1..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Everything ONE spectrum contributes, computed from that spectrum alone.
///
/// Splitting the per-spectrum work out of the fold is what makes the mzML parse
/// parallelisable: nothing here reads any other spectrum, and every piece of
/// order-dependent state -- the scan index, the preceding MS1, the isolation-window
/// id map, the two drop counters and the first offending scan id -- stays in
/// [`Fold`], which runs on one thread in index order. The variants also encode the
/// ORDER of the checks the sequential loop performed, which is load-bearing: a
/// non-finite retention time is decided before any peak is decoded, so such a scan
/// contributes nothing to `nonfinite_peaks`, and a spectrum of any other MS level
/// decodes no peaks at all.
enum Decoded {
    /// Scan start time is not finite. Dropped, with its id kept for the warning.
    BadRt {
        id: String,
        rt_s: f64,
    },
    Ms1 {
        rt_s: f64,
        mz: Vec<f32>,
        inten: Vec<f32>,
        nonfinite_peaks: usize,
    },
    Ms2(Box<Ms2Row>),
    /// MS3 and above, or an absent level: counted as read, nothing written.
    Other,
}

/// The MS2 payload, boxed so `Decoded` is not dominated by its largest variant.
struct Ms2Row {
    rt_s: f64,
    id: String,
    mz: Vec<f32>,
    inten: Vec<f32>,
    nonfinite_peaks: usize,
    wt: f64,
    wl: f64,
    wu: f64,
    pmz: Option<f64>,
    pz: Option<i32>,
}

fn decode_one<S: SpectrumLike>(spec: &S, top_ms1: usize, top_ms2: usize) -> Decoded {
    let rt_s = spec.start_time() * 60.0; // mzdata returns minutes
    if !rt_s.is_finite() {
        return Decoded::BadRt {
            id: spec.id().to_string(),
            rt_s,
        };
    }
    match spec.ms_level() {
        1 => {
            let (mz, inten, nonfinite_peaks) = peaks_of(spec, top_ms1);
            Decoded::Ms1 {
                rt_s,
                mz,
                inten,
                nonfinite_peaks,
            }
        }
        2 => {
            let (mz, inten, nonfinite_peaks) = peaks_of(spec, top_ms2);
            let prec = spec.precursor();
            let iw = prec.map(|pr| pr.isolation_window.clone());
            let (wt, wl, wu) = match &iw {
                Some(w) if !(w.lower_bound == 0.0 && w.upper_bound == 0.0) => {
                    (w.target as f64, w.lower_bound as f64, w.upper_bound as f64)
                }
                // AIF / all-ion: no quad isolation -> full-range window.
                _ => (0.0, 0.0, 1.0e6),
            };
            let (pmz, pz) = match prec.and_then(|pr| pr.ions.first()) {
                Some(ion) => (Some(ion.mz), ion.charge),
                None => (None, None),
            };
            Decoded::Ms2(Box::new(Ms2Row {
                rt_s,
                id: spec.id().to_string(),
                mz,
                inten,
                nonfinite_peaks,
                wt,
                wl,
                wu,
                pmz,
                pz,
            }))
        }
        _ => Decoded::Other,
    }
}

/// The order-dependent half of the conversion: the accumulators, the parquet
/// writers, and the four folds whose values depend on what came before.
struct Fold {
    ms1_w: TableWriter,
    ms2_w: TableWriter,
    ms1: Ms1Chunk,
    ms2: Ms2Chunk,
    /// Distinct isolation windows (id, target, lower, upper); ids by first
    /// appearance in scan order.
    uniq: Vec<(u64, f64, f64, f64)>,
    seen: std::collections::HashMap<(u64, u64), u32>,
    /// MS2 -> preceding MS1 scan map (two ints per MS2 scan; kept whole).
    map_ms2: Vec<u32>,
    map_ms1: Vec<i32>,
    last_ms1_index: Option<u32>,
    nonfinite_peaks: usize,
    nonfinite_rt: usize,
    first_bad_rt: Option<String>,
}

impl Fold {
    fn new(ms1_path: &str, ms2_path: &str) -> Self {
        Self {
            ms1_w: TableWriter::new(ms1_path).with_row_group_rows(SPECTRA_CHUNK),
            ms2_w: TableWriter::new(ms2_path).with_row_group_rows(SPECTRA_CHUNK),
            ms1: Ms1Chunk::default(),
            ms2: Ms2Chunk::default(),
            uniq: Vec::new(),
            seen: std::collections::HashMap::new(),
            map_ms2: Vec::new(),
            map_ms1: Vec::new(),
            last_ms1_index: None,
            nonfinite_peaks: 0,
            nonfinite_rt: 0,
            first_bad_rt: None,
        }
    }

    fn absorb(&mut self, scan_index: u32, d: Decoded) -> Result<()> {
        match d {
            Decoded::BadRt { id, rt_s } => {
                self.nonfinite_rt += 1;
                if self.first_bad_rt.is_none() {
                    self.first_bad_rt = Some(format!("{id} (rt={rt_s})"));
                }
            }
            Decoded::Ms1 {
                rt_s,
                mz,
                inten,
                nonfinite_peaks,
            } => {
                self.nonfinite_peaks += nonfinite_peaks;
                self.ms1.idx.push(scan_index);
                self.ms1.rt.push(rt_s);
                self.ms1.mz.push(mz);
                self.ms1.inten.push(inten);
                self.last_ms1_index = Some(scan_index);
                if self.ms1.idx.len() >= SPECTRA_CHUNK {
                    self.ms1_w.write_cols(self.ms1.cols())?;
                }
            }
            Decoded::Ms2(r) => {
                let Ms2Row {
                    rt_s,
                    id,
                    mz,
                    inten,
                    nonfinite_peaks,
                    wt,
                    wl,
                    wu,
                    pmz,
                    pz,
                } = *r;
                self.nonfinite_peaks += nonfinite_peaks;
                let uniq = &mut self.uniq;
                let win_id = *self
                    .seen
                    .entry((wl.to_bits(), wu.to_bits()))
                    .or_insert_with(|| {
                        let id = uniq.len() as u32;
                        uniq.push((id as u64, wt, wl, wu));
                        id
                    });
                self.ms2.idx.push(scan_index);
                self.ms2.id.push(id);
                self.ms2.rt.push(rt_s);
                self.ms2.win_id.push(win_id);
                self.ms2.wt.push(wt);
                self.ms2.wl.push(wl);
                self.ms2.wu.push(wu);
                self.ms2.pmz.push(pmz);
                self.ms2.pz.push(pz);
                self.ms2.mz.push(mz);
                self.ms2.inten.push(inten);
                self.map_ms2.push(scan_index);
                self.map_ms1
                    .push(self.last_ms1_index.map(|x| x as i32).unwrap_or(-1));
                if self.ms2.idx.len() >= SPECTRA_CHUNK {
                    self.ms2_w.write_cols(self.ms2.cols())?;
                }
            }
            Decoded::Other => {}
        }
        Ok(())
    }
}

/// The one-thread decode, unchanged in every observable respect: the iterator
/// yields spectra in file order, `scan_index` is the ordinal, and a parse error
/// part-way through simply ends the iterator.
fn drive_sequential<S: SpectrumLike>(
    reader: impl Iterator<Item = S>,
    fold: &mut Fold,
    p: &ConvertParams,
) -> Result<usize> {
    let mut read = 0usize;
    for (count, (scan_index, spec)) in (0_u32..).zip(reader).enumerate() {
        if p.max_spectra > 0 && count >= p.max_spectra {
            break;
        }
        // Count what the READER yielded, here at the top of the body, not at the
        // bottom.
        //
        // `declared` is the number of `<spectrum>` elements the header advertises,
        // not the number this stage finds usable, so the completeness check in
        // `run` compares like with like only if every spectrum handed over
        // advances `read`. As the last statement of the body it was skipped by the
        // non-finite-RT `continue`, and a bad retention time in the FINAL spectrum
        // then left `read == declared - 1` and made a whole file fail as
        // "truncated or corrupt", telling the user to re-transfer a multi-gigabyte
        // mzML. Anywhere but the tail the next good spectrum re-assigned `read`
        // and hid it.
        //
        // Position relative to the `break` above is load-bearing and must not
        // move: `--max-spectra N` breaks on `count == N` BEFORE this assignment,
        // and the previous iteration already set `read = N`, so `capped` in `run`
        // still evaluates exactly as it did.
        read = count + 1;
        fold.absorb(
            scan_index,
            decode_one(&spec, p.top_peaks_ms1, p.top_peaks_ms2),
        )?;
    }
    Ok(read)
}

/// Converts running concurrently in this process (`experiment.parallel_runs`).
///
/// The decode threads are drawn from the rayon pool's size, and `run-experiment`
/// can call `convert::run` from inside a `par_iter` over runs, so without this the
/// process would hold `parallel_runs x pool` decode threads. Each conversion takes
/// its share of the pool instead.
static LIVE_CONVERTS: AtomicUsize = AtomicUsize::new(0);

struct LiveConvert;

impl LiveConvert {
    fn enter() -> (Self, usize) {
        let live = LIVE_CONVERTS.fetch_add(1, Ordering::SeqCst) + 1;
        (Self, live)
    }
}

impl Drop for LiveConvert {
    fn drop(&mut self) {
        LIVE_CONVERTS.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Decode threads for one conversion. `MUMDIA_CONVERT_THREADS` overrides, and
/// `0` or `1` forces the sequential path.
fn convert_threads(live: usize) -> usize {
    if let Ok(v) = std::env::var("MUMDIA_CONVERT_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            return n;
        }
    }
    (rayon::current_num_threads() / live.max(1)).max(1)
}

/// A chunk is a contiguous run of spectra spanning about this many bytes of the
/// mzML. Sizing in BYTES rather than in spectra keeps the decoded chunk (and so
/// the in-flight memory) roughly constant whether a spectrum holds 100 peaks or
/// 10,000, and keeps each worker's `seek` amortised over a useful read.
const CHUNK_TARGET_BYTES: u64 = 1 << 20;
/// Belt and braces for a file of very small spectra, so a chunk cannot become
/// unboundedly many rows.
const CHUNK_MAX_SPECTRA: usize = 4096;
/// Chunks a worker may have produced but not yet handed over. Total in-flight
/// decoded bytes are bounded by `threads x (QUEUE + 1) x CHUNK_TARGET_BYTES`-ish.
const CHUNK_QUEUE: usize = 2;

/// One worker's output for one chunk.
struct ChunkOut {
    rows: Vec<Decoded>,
    /// The global index at which `read_next` gave up, if it did. Reproduces the
    /// sequential iterator's "a parse error ends the run" semantics: the consumer
    /// stops at the FIRST such index and everything after it is discarded, so a
    /// truncated or damaged file still yields exactly the contiguous prefix it
    /// yields today, and the completeness check below still refuses it.
    failed_at: Option<usize>,
}

fn plan_chunks(offsets: &[u64], n: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < n {
        let base = offsets[start];
        let mut end = start + 1;
        while end < n
            && end - start < CHUNK_MAX_SPECTRA
            && offsets[end].saturating_sub(base) < CHUNK_TARGET_BYTES
        {
            end += 1;
        }
        out.push((start, end));
        start = end;
    }
    out
}

/// A decode reader: the header parsed (so `referenceableParamGroupRef` and the
/// instrument configurations resolve exactly as they do for the sequential reader)
/// and NO offset index built, because the byte offsets are handed to it.
fn open_decode_reader(path: &str) -> Result<MzMLReaderType<std::fs::File>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {path}"))?;
    Ok(
        MzMLReaderType::<std::fs::File>::with_buffer_capacity_and_detail_level(
            file,
            READER_BUFFER_BYTES,
            DetailLevel::Full,
        ),
    )
}

/// Can the offset index drive a seek-based decode of this file?
///
/// Probes the last spectrum, and the middle one when there is more than one: seeks
/// to the byte offset the index gives for it and checks that the spectrum found
/// there calls itself that index. Two spectra, so it costs nothing measurable.
///
/// It catches, BEFORE any row is produced, the three ways this could go wrong: a
/// stale index whose offsets no longer line up; a file whose `<spectrum>` elements
/// omit the `index` attribute the mzML schema requires, where mzdata reports 0 for
/// all of them and a seek-driven decode would mislabel every scan; and a file that
/// ends before its index says. Any of them sends the whole run down the sequential
/// path, which is the one whose truncation behaviour the completeness check in `run`
/// is written against. That in turn is what lets the same check inside a worker be a
/// hard error: by then it is a should-never-happen.
fn offsets_are_trustworthy(path: &str, offsets: &[u64], p: &ConvertParams) -> bool {
    let n = offsets.len();
    if n == 0 {
        return false;
    }
    let Ok(mut r) = open_decode_reader(path) else {
        return false;
    };
    let mut probes = vec![n - 1];
    if n > 2 {
        probes.push(n / 2);
    }
    for i in probes {
        match produce_chunk(&mut r, offsets, (i, i + 1), p) {
            Ok(out) if out.failed_at.is_none() => {}
            _ => return false,
        }
    }
    true
}

/// Decode one chunk on a worker's own reader.
fn produce_chunk(
    reader: &mut MzMLReaderType<std::fs::File>,
    offsets: &[u64],
    range: (usize, usize),
    p: &ConvertParams,
) -> Result<ChunkOut> {
    let (start, end) = range;
    reader
        .seek(SeekFrom::Start(offsets[start]))
        .with_context(|| format!("seek to spectrum {start}"))?;
    let mut rows = Vec::with_capacity(end - start);
    for i in start..end {
        let Some(spec) = reader.read_next() else {
            return Ok(ChunkOut {
                rows,
                failed_at: Some(i),
            });
        };
        // The offset index says this byte offset is spectrum `i`. If the file
        // disagrees the index is stale and every `scan_index` after this point
        // would be wrong, which is silent corruption rather than a slow run, so
        // refuse instead of guessing.
        if spec.index() != i {
            anyhow::bail!(
                concat!(
                    "the mzML offset index is inconsistent with the file: the entry for ",
                    "spectrum {} points at a spectrum that calls itself {}. Re-index the ",
                    "file, or set MUMDIA_CONVERT_THREADS=1 to read it sequentially."
                ),
                i,
                spec.index()
            );
        }
        rows.push(decode_one(&spec, p.top_peaks_ms1, p.top_peaks_ms2));
    }
    Ok(ChunkOut {
        rows,
        failed_at: None,
    })
}

/// The many-thread decode. Returns the number of spectra the readers yielded
/// contiguously from the start of the file, which is what `read` means on the
/// sequential path too.
///
/// Equality: every float in `decode_one` is confined to one spectrum, so nothing
/// is reordered; the consumer folds chunks in index order on one thread, so the
/// window-id map, the preceding-MS1 link, the two counters, the first offending
/// scan id and the parquet row order are all assigned exactly as they are
/// sequentially. `scan_index` is the mzML `index` attribute here and the iteration
/// ordinal there; `produce_chunk` refuses the file if the two could differ.
fn drive_parallel(
    path: &str,
    offsets: &[u64],
    n_total: usize,
    threads: usize,
    fold: &mut Fold,
    p: &ConvertParams,
) -> Result<usize> {
    let chunks = plan_chunks(offsets, n_total);
    let workers = threads.min(chunks.len()).max(1);
    let chunks = &chunks;

    let mut read = 0usize;
    let mut err: Option<anyhow::Error> = None;
    std::thread::scope(|scope| -> Result<()> {
        // One bounded queue per worker, and worker `w` owns chunks
        // `w, w + workers, w + 2 * workers, ...`. The consumer therefore always
        // knows which queue the next chunk will arrive on and needs no reorder
        // buffer, and a worker blocks once its queue is full, which is what bounds
        // the memory. A single shared queue plus a reorder map does not bound it:
        // the consumer has to keep draining to avoid deadlock, so one slow chunk
        // lets every other worker run to the end of the file.
        let mut txs = Vec::with_capacity(workers);
        let mut rxs = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Result<ChunkOut>>(CHUNK_QUEUE);
            txs.push(tx);
            rxs.push(rx);
        }
        for (w, tx) in txs.into_iter().enumerate() {
            scope.spawn(move || {
                let mut reader = match open_decode_reader(path) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                };
                for c in (w..chunks.len()).step_by(workers) {
                    let out = produce_chunk(&mut reader, offsets, chunks[c], p);
                    let failed = matches!(&out, Ok(o) if o.failed_at.is_some()) || out.is_err();
                    if tx.send(out).is_err() || failed {
                        return;
                    }
                }
            });
        }

        for (c, &(start, end)) in chunks.iter().enumerate() {
            let out = match rxs[c % workers].recv() {
                Ok(Ok(out)) => out,
                Ok(Err(e)) => {
                    err = Some(e);
                    break;
                }
                // A worker died without sending. Nothing sane to salvage.
                Err(_) => {
                    err = Some(anyhow::anyhow!(
                        "convert: an mzML decode worker stopped unexpectedly"
                    ));
                    break;
                }
            };
            debug_assert_eq!(
                out.rows.len(),
                end - start - out.failed_at.map_or(0, |f| end - f)
            );
            for (k, d) in out.rows.into_iter().enumerate() {
                let i = start + k;
                if p.max_spectra > 0 && i >= p.max_spectra {
                    break;
                }
                fold.absorb(i as u32, d)?;
                read = i + 1;
            }
            if out.failed_at.is_some() || (p.max_spectra > 0 && read >= p.max_spectra) {
                break;
            }
        }
        // Release the workers still blocked on a full queue before the scope joins.
        drop(rxs);
        Ok(())
    })?;
    if let Some(e) = err {
        return Err(e);
    }
    Ok(read)
}

/// Read buffer per decode reader. mzdata's own default is 10,000 bytes, which is
/// about 154,000 read syscalls on a 1.5 GB file.
const READER_BUFFER_BYTES: usize = 1 << 18;

// Which decode path the last `run_inner` on THIS thread took. Test-only, and
// thread-local rather than a global counter so the equality tests can assert it
// while the rest of the suite runs alongside them.
#[cfg(test)]
thread_local! {
    static LAST_DECODE_PARALLEL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn note_decode_path(parallel: bool) {
    LAST_DECODE_PARALLEL.with(|c| c.set(parallel));
}

#[cfg(not(test))]
fn note_decode_path(_parallel: bool) {}

pub fn run(p: ConvertParams) -> Result<ConvertOutputs> {
    run_inner(p, None)
}

/// `force_threads` exists so a test can run the same input down both decode paths
/// and diff the artifacts, without an environment variable that every other test
/// in the process would see.
fn run_inner(p: ConvertParams, force_threads: Option<usize>) -> Result<ConvertOutputs> {
    let t0 = Instant::now();
    std::fs::create_dir_all(p.out_dir).ok();
    info!(mzml = p.mzml, "convert: opening mzML");
    let reader = mzdata::MZReader::open_path(p.mzml).with_context(|| format!("open {}", p.mzml))?;
    // The number of spectra the file SAYS it has, read from its own header.
    //
    // The iteration below yields `Spectrum`, not `Result<Spectrum>`, so a parse error
    // part-way through a file simply ends the iterator: a truncated download or a
    // dropped network share produced a complete-looking artifact set covering the first
    // fraction of the gradient, at exit 0, with no error. mzdata does log the parse
    // failure, but a log line does not stop the pipeline and is easy to miss in a batch.
    //
    // Deliberately NOT `SpectrumSource::len()`: that comes from the index, and in an
    // indexedmzML the index is at the END of the file, so exactly the truncation this
    // guard is for makes it read 0. `spectrumList count` is in the header, a few hundred
    // KiB in at most, so it survives any truncation long enough to be worth checking.
    let declared = declared_spectrum_count(p.mzml).unwrap_or(0);

    let ms1_path = format!("{}/spectra_ms1.parquet", p.out_dir);
    let ms2_path = format!("{}/spectra_ms2.parquet", p.out_dir);
    let iw_path = format!("{}/isolation_windows.parquet", p.out_dir);
    let map_path = format!("{}/ms2_to_ms1.parquet", p.out_dir);

    // Spectra stream to parquet in SPECTRA_CHUNK-scan row groups as they are decoded, so
    // the run is never resident as a whole: the previous single `write_table` per MS level
    // held every peak of the run, plus the encoder's copy, before the first byte reached
    // disk. Rows and their order are unchanged; only the row-group layout differs.
    //
    // `Fold` also holds the two drop counters. Peaks dropped for a non-finite m/z or
    // intensity are counted rather than ignored: a handful is a damaged spectrum, and a
    // large fraction means the file or the converter that produced it is wrong, which is
    // worth knowing before the numbers are believed.
    //
    // Spectra with a non-finite retention time are dropped and counted the same way.
    // Retention time was the one externally supplied float this stage did not check, and
    // it is the value everything downstream keys on. A single `NaN` scan start time in one
    // scan of an mzML -- reproduced by editing one `scan start time` value in the fixture
    // -- passed convert without a warning, was written into the spectra artifact, and then
    // aborted `mumdia run` inside extract with `called `Option::unwrap()` on a `None`
    // value` at `extract.rs`, naming neither the file, nor the scan, nor the value.
    // Dropped rather than fatal, for the same reason as a bad peak: one unusable scan is
    // not a reason to refuse a run, and a spectrum with no retention time cannot be placed
    // in a chromatogram. The "no MS2 spectra" bail below is the backstop for a file whose
    // retention times are ALL unusable.
    let mut fold = Fold::new(&ms1_path, &ms2_path);

    // Decode on several threads when the file's own offset index makes it safe.
    //
    // `MZReader::open_path` has ALREADY built or read that index (`new_indexed`), so the
    // byte offset of every spectrum is in hand and the cost is sunk whether or not it is
    // used. The parallel path is taken only when the index is initialised, is exactly as
    // long as the header's `spectrumList count`, and the file is mzML: any disagreement
    // means the index is stale, rebuilt over a short file, or absent, and then the
    // sequential reader -- whose truncation behaviour the completeness check below is
    // written against -- is both correct and the right thing to be running.
    let (_live_guard, live) = LiveConvert::enter();
    let threads = force_threads.unwrap_or_else(|| convert_threads(live));
    let offsets: Option<Vec<u64>> = if threads > 1 && declared > 0 {
        match &reader {
            MZReaderType::MzML(_) => {
                let idx = reader.get_index();
                if idx.init && idx.len() == declared {
                    Some(idx.offsets.values().copied().collect())
                } else {
                    None
                }
            }
            _ => None,
        }
    } else {
        None
    };
    // And the index has to survive a two-spectrum probe before any row is produced.
    let offsets = offsets.filter(|o| offsets_are_trustworthy(p.mzml, o, &p));

    note_decode_path(offsets.is_some());
    let read = match offsets {
        Some(offsets) => {
            let n_total = if p.max_spectra > 0 {
                offsets.len().min(p.max_spectra)
            } else {
                offsets.len()
            };
            debug!(
                threads,
                spectra = n_total,
                "convert: decoding the mzML in parallel from its offset index"
            );
            drop(reader);
            drive_parallel(p.mzml, &offsets, n_total, threads, &mut fold, &p)?
        }
        None => {
            debug!(threads, "convert: decoding the mzML sequentially");
            drive_sequential(reader, &mut fold, &p)?
        }
    };

    // Final chunks (possibly empty: they fix the schema for a level with no scans). The
    // writers are closed (published) only after the checks below: an abandoned
    // `TableWriter` removes its temp file, so a refused run leaves no half-written
    // spectra artifact behind.
    let Fold {
        mut ms1_w,
        mut ms2_w,
        mut ms1,
        mut ms2,
        uniq,
        map_ms2,
        map_ms1,
        nonfinite_peaks,
        nonfinite_rt,
        first_bad_rt,
        ..
    } = fold;
    ms1_w.write_cols(ms1.cols())?;
    ms2_w.write_cols(ms2.cols())?;

    // A short read means the file ended before the index said it would. `--max-spectra`
    // truncates deliberately, so it is excluded.
    let capped = p.max_spectra > 0 && read >= p.max_spectra;
    if !capped && declared > 0 && read < declared {
        anyhow::bail!(
            concat!(
                "{} declares {} spectra in its header but only {} could be read, so it ",
                "is truncated or corrupt. An interrupted transfer of a large mzML is the ",
                "usual cause. Continuing would produce a complete-looking artifact set ",
                "covering only the first {:.0}% of the run. Re-transfer or re-convert ",
                "the file and compare sizes."
            ),
            p.mzml,
            declared,
            read,
            100.0 * read as f64 / declared as f64
        );
    }
    // Zero MS2 is never a legitimate DIA run, and it is the shape every downstream stage
    // silently tolerates: search-seed finds nothing, extract runs the whole library
    // against an empty spectrum list, and report writes a header-only peptides.tsv, all
    // at exit 0. Fail here, where the cause is still visible.
    if nonfinite_peaks > 0 {
        warn!(
            nonfinite_peaks,
            mzml = p.mzml,
            "convert: dropped peaks with a non-finite m/z or intensity. A few indicate a              damaged spectrum; a large number indicates a problem with the file or with              the converter that wrote it"
        );
    }
    if nonfinite_rt > 0 {
        warn!(
            nonfinite_rt,
            first = first_bad_rt.as_deref().unwrap_or("?"),
            mzml = p.mzml,
            concat!(
                "convert: dropped spectra whose scan start time is not finite. Such a ",
                "spectrum cannot be placed in a chromatogram, and left in place it ",
                "propagates NaN into the retention-time windows and aborts a later stage"
            )
        );
    }
    if ms2_w.rows() == 0 {
        anyhow::bail!(
            concat!(
                "{} yielded no MS2 spectra ({} MS1). A DIA run must contain MS2, so this ",
                "is the wrong file, an MS1-only acquisition, or a conversion that dropped ",
                "the MS2 level. Every later stage tolerates an empty spectrum list ",
                "silently -- search-seed finds nothing, extract runs the whole library ",
                "against nothing, report writes a header-only peptides.tsv -- so this has ",
                "to fail here."
            ),
            p.mzml,
            ms1_w.rows()
        );
    }

    let n_ms1 = ms1_w.close()?;
    let n_ms2 = ms2_w.close()?;

    let n_iw = write_table(
        &iw_path,
        vec![
            Col::U32(
                "window_id".into(),
                uniq.iter().map(|w| w.0 as u32).collect(),
            ),
            Col::F64("target".into(), uniq.iter().map(|w| w.1).collect()),
            Col::F64("lower".into(), uniq.iter().map(|w| w.2).collect()),
            Col::F64("upper".into(), uniq.iter().map(|w| w.3).collect()),
        ],
    )?;

    let n_map = write_table(
        &map_path,
        vec![
            Col::U32("ms2_scan_index".into(), map_ms2),
            Col::I32("ms1_scan_index".into(), map_ms1),
        ],
    )?;

    let elapsed = t0.elapsed().as_millis();
    let mut hashes = write_reports(
        &[
            (&ms1_path, artifact::SPECTRA_MS1, n_ms1),
            (&ms2_path, artifact::SPECTRA_MS2, n_ms2),
            (&iw_path, artifact::ISOLATION_WINDOWS, n_iw),
            (&map_path, artifact::MS2_TO_MS1, n_map),
        ],
        elapsed,
        json!({
            "mzml": p.mzml,
            "max_spectra": p.max_spectra,
            "top_peaks_ms2": p.top_peaks_ms2,
            "top_peaks_ms1": p.top_peaks_ms1,
            "config_hash": p.config_hash,
        }),
    )?;

    info!(
        ms1 = n_ms1,
        ms2 = n_ms2,
        windows = n_iw,
        elapsed_ms = elapsed,
        "convert: done"
    );
    // `write_reports` returns the hashes in the order it was given the artifacts.
    let map_hash = hashes.pop().expect("four reports written");
    let iw_hash = hashes.pop().expect("four reports written");
    let ms2_hash = hashes.pop().expect("four reports written");
    let ms1_hash = hashes.pop().expect("four reports written");
    Ok(ConvertOutputs {
        ms1: ms1_path,
        ms2: ms2_path,
        isolation_windows: iw_path,
        ms2_to_ms1: map_path,
        hashes: ConvertHashes {
            ms1: ms1_hash,
            ms2: ms2_hash,
            isolation_windows: iw_hash,
            ms2_to_ms1: map_hash,
        },
    })
}

/// Write one report per artifact and return the content hashes they record, in `items`
/// order.
fn write_reports(
    items: &[(&String, (&str, u32), u64)],
    elapsed_ms: u128,
    params: serde_json::Value,
) -> Result<Vec<String>> {
    let mut hashes = Vec::with_capacity(items.len());
    for (path, schema, rows) in items {
        let rep = ArtifactReport {
            logical_name: schema.0.to_string(),
            schema_name: schema.0.to_string(),
            schema_version: schema.1,
            stage: "convert".to_string(),
            rows: *rows,
            content_hash: mumdia_io::hash::blake3_file(path)?,
            params: params.clone(),
            stats: Default::default(),
            model_identity: None,
            elapsed_ms,
        };
        rep.write_for(path)?;
        hashes.push(rep.content_hash);
    }
    Ok(hashes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Base64, written out rather than pulled in as a dependency: the tests need
    /// a few hundred bytes of it and `mumdia` has no base64 crate.
    fn b64(bytes: &[u8]) -> String {
        const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for c in bytes.chunks(3) {
            let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
            let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
            out.push(A[(n >> 18) as usize & 63] as char);
            out.push(A[(n >> 12) as usize & 63] as char);
            out.push(if c.len() > 1 {
                A[(n >> 6) as usize & 63] as char
            } else {
                '='
            });
            out.push(if c.len() > 2 {
                A[n as usize & 63] as char
            } else {
                '='
            });
        }
        out
    }

    fn f64_array(v: &[f64]) -> String {
        let mut raw = Vec::new();
        for x in v {
            raw.extend_from_slice(&x.to_le_bytes());
        }
        b64(&raw)
    }

    fn f32_array(v: &[f32]) -> String {
        let mut raw = Vec::new();
        for x in v {
            raw.extend_from_slice(&x.to_le_bytes());
        }
        b64(&raw)
    }

    /// One `<spectrum>`. `rt` is written verbatim, so a test can plant `NaN`.
    fn spectrum(index: usize, ms_level: u8, rt: &str, mz: &[f64], inten: &[f32]) -> String {
        let level_cv = if ms_level == 1 {
            r#"<cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" value=""/>"#
        } else {
            r#"<cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" value=""/>"#
        };
        let precursor = if ms_level == 1 {
            ""
        } else {
            concat!(
                r#"<precursorList count="1"><precursor><isolationWindow>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" value="500.0"/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="5.0"/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="5.0"/>"#,
                "</isolationWindow><activation>",
                r#"<cvParam cvRef="MS" accession="MS:1000133" name="collision-induced dissociation" value=""/>"#,
                "</activation></precursor></precursorList>",
            )
        };
        let mz_b64 = f64_array(mz);
        let in_b64 = f32_array(inten);
        format!(
            concat!(
                r#"<spectrum index="{index}" id="scan={scan}" defaultArrayLength="{n}">"#,
                "{level_cv}",
                r#"<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{ms_level}"/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" value=""/>"#,
                r#"<scanList count="1">"#,
                r#"<cvParam cvRef="MS" accession="MS:1000795" name="no combination" value=""/>"#,
                "<scan>",
                r#"<cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt}""#,
                r#" unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"/>"#,
                "</scan></scanList>",
                "{precursor}",
                r#"<binaryDataArrayList count="2">"#,
                r#"<binaryDataArray encodedLength="{mz_len}">"#,
                r#"<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>"#,
                "<binary>{mz_b64}</binary></binaryDataArray>",
                r#"<binaryDataArray encodedLength="{in_len}">"#,
                r#"<cvParam cvRef="MS" accession="MS:1000521" name="32-bit float" value=""/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>"#,
                "<binary>{in_b64}</binary></binaryDataArray>",
                "</binaryDataArrayList></spectrum>",
            ),
            index = index,
            scan = index + 1,
            n = mz.len(),
            level_cv = level_cv,
            ms_level = ms_level,
            rt = rt,
            precursor = precursor,
            mz_len = mz_b64.len(),
            in_len = in_b64.len(),
            mz_b64 = mz_b64,
            in_b64 = in_b64,
        )
    }

    /// One MS2 `<spectrum>` whose binary arrays are ZLIB-COMPRESSED, so the inflate
    /// backend is on the path of at least one test.
    ///
    /// The two blobs are pre-computed constants rather than compressed here: DEFLATE
    /// output depends on the compressor, the engine only ever inflates, and a literal
    /// leaves the test with no compression dependency of its own. They hold
    /// `[100.5, 200.25, 300.125, 400.0625]` as little-endian f64 and
    /// `[11, 22, 33, 44]` as little-endian f32, all exactly representable in f32, so
    /// the expected output is exact.
    fn zlib_ms2_spectrum(index: usize, rt: &str) -> String {
        const MZ_B64: &str = "eJxjYGBgYFCIdABRDByZEPpQEYRmrHQAAC9DA5k=";
        const IN_B64: &str = "eJxjYDBwZGDY4MjAwOLEwGDgBAARKgIb";
        format!(
            concat!(
                r#"<spectrum index="{index}" id="scan={scan}" defaultArrayLength="4">"#,
                r#"<cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" value=""/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2"/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" value=""/>"#,
                r#"<scanList count="1">"#,
                r#"<cvParam cvRef="MS" accession="MS:1000795" name="no combination" value=""/>"#,
                "<scan>",
                r#"<cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt}""#,
                r#" unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"/>"#,
                "</scan></scanList>",
                r#"<precursorList count="1"><precursor><isolationWindow>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" value="500.0"/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="5.0"/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="5.0"/>"#,
                "</isolationWindow><activation>",
                r#"<cvParam cvRef="MS" accession="MS:1000133" name="collision-induced dissociation" value=""/>"#,
                "</activation></precursor></precursorList>",
                r#"<binaryDataArrayList count="2">"#,
                r#"<binaryDataArray encodedLength="{mz_len}">"#,
                r#"<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>"#,
                "<binary>{mz_b64}</binary></binaryDataArray>",
                r#"<binaryDataArray encodedLength="{in_len}">"#,
                r#"<cvParam cvRef="MS" accession="MS:1000521" name="32-bit float" value=""/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>"#,
                r#"<cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>"#,
                "<binary>{in_b64}</binary></binaryDataArray>",
                "</binaryDataArrayList></spectrum>",
            ),
            index = index,
            scan = index + 1,
            rt = rt,
            mz_len = MZ_B64.len(),
            in_len = IN_B64.len(),
            mz_b64 = MZ_B64,
            in_b64 = IN_B64,
        )
    }

    /// An mzML whose `<spectrumList count=...>` is `declared` and whose body is
    /// `spectra`, so a test can make the two disagree deliberately.
    fn mzml(declared: usize, spectra: &[String]) -> String {
        let mut s = String::from(concat!(
            r#"<?xml version="1.0" encoding="utf-8"?>"#,
            "\n",
            r#"<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0" id="t">"#,
            r#"<cvList count="2">"#,
            r#"<cv id="MS" fullName="PSI-MS" URI="https://example.invalid/psi-ms.obo"/>"#,
            r#"<cv id="UO" fullName="UO" URI="https://example.invalid/unit.obo"/>"#,
            "</cvList>",
            "<fileDescription><fileContent>",
            r#"<cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" value=""/>"#,
            "</fileContent></fileDescription>",
            r#"<softwareList count="1"><software id="sw" version="1">"#,
            r#"<cvParam cvRef="MS" accession="MS:1000799" name="custom unreleased software tool" value="t"/>"#,
            "</software></softwareList>",
            r#"<instrumentConfigurationList count="1"><instrumentConfiguration id="IC1">"#,
            r#"<cvParam cvRef="MS" accession="MS:1000031" name="instrument model" value="synthetic"/>"#,
            "</instrumentConfiguration></instrumentConfigurationList>",
            r#"<dataProcessingList count="1"><dataProcessing id="DP1">"#,
            r#"<processingMethod order="1" softwareRef="sw">"#,
            r#"<cvParam cvRef="MS" accession="MS:1000544" name="Conversion to mzML" value=""/>"#,
            "</processingMethod></dataProcessing></dataProcessingList>",
            r#"<run id="r" defaultInstrumentConfigurationRef="IC1">"#,
        ));
        s.push_str(&format!(
            r#"<spectrumList count="{declared}" defaultDataProcessingRef="DP1">"#
        ));
        for sp in spectra {
            s.push_str(sp);
        }
        s.push_str("</spectrumList></run></mzML>\n");
        s
    }

    /// A run of `cycles` cycles, each one MS1 followed by `windows` MS2 scans in
    /// distinct isolation windows, with a deterministic peak list per spectrum.
    /// Big enough that `plan_chunks` produces several chunks and the parallel
    /// decode really does hand work to more than one worker.
    fn synthetic_run(cycles: usize, windows: usize, peaks: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut index = 0usize;
        for c in 0..cycles {
            let rt = format!("{:.6}", 0.01 * c as f64);
            let mz: Vec<f64> = (0..peaks).map(|k| 300.0 + k as f64 * 0.7).collect();
            let inten: Vec<f32> = (0..peaks).map(|k| 10.0 + (k % 37) as f32).collect();
            out.push(spectrum(index, 1, &rt, &mz, &inten));
            index += 1;
            for w in 0..windows {
                let mz: Vec<f64> = (0..peaks)
                    .map(|k| 120.0 + w as f64 * 3.0 + k as f64 * 0.9)
                    .collect();
                let inten: Vec<f32> = (0..peaks).map(|k| 5.0 + ((k + w) % 29) as f32).collect();
                // `spectrum` writes one fixed isolation window, so vary it here by
                // rewriting the target/offsets: distinct windows exercise the
                // first-appearance id map that the fold owns.
                let s = spectrum(index, 2, &rt, &mz, &inten)
                    .replace(
                        r#"name="isolation window target m/z" value="500.0""#,
                        &format!(
                            r#"name="isolation window target m/z" value="{:.1}""#,
                            400.0 + 20.0 * w as f64
                        ),
                    )
                    .replace(
                        r#"name="isolation window lower offset" value="5.0""#,
                        &format!(
                            r#"name="isolation window lower offset" value="{:.1}""#,
                            5.0 + w as f64
                        ),
                    );
                out.push(s);
                index += 1;
            }
        }
        out
    }

    /// Four spectra, alternating MS1/MS2, with `rt` supplying each scan start
    /// time verbatim.
    fn four_spectra(rt: [&str; 4]) -> Vec<String> {
        vec![
            spectrum(0, 1, rt[0], &[300.0, 400.0], &[10.0, 20.0]),
            spectrum(1, 2, rt[1], &[100.0, 200.0], &[30.0, 40.0]),
            spectrum(2, 1, rt[2], &[300.0, 400.0], &[11.0, 21.0]),
            spectrum(3, 2, rt[3], &[100.0, 200.0], &[31.0, 41.0]),
        ]
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "mumdia_convert_{tag}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn took_parallel_path() -> bool {
        LAST_DECODE_PARALLEL.with(|c| c.get())
    }

    fn rows_in_report(path: &str) -> u64 {
        let text = std::fs::read_to_string(format!("{path}.report.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        v["rows"].as_u64().unwrap()
    }

    /// A file the offset index cannot drive must fall back, not fail and not
    /// mislabel.
    ///
    /// The mzML schema requires `index` on every `<spectrum>`, and a seek-driven
    /// decode takes the scan index from it, so a file that writes the same value
    /// everywhere would silently mislabel every scan. Here every spectrum claims
    /// `index="0"`. The probe must catch it, the run must go sequential, and the
    /// artifacts must be exactly what the sequential decode has always produced --
    /// which is asserted by diffing against the same file with correct indices.
    #[test]
    fn a_file_whose_index_attributes_are_wrong_falls_back_to_the_sequential_decode() {
        let base = tmpdir("bad_index_attr");
        let spectra = synthetic_run(80, 5, 400);
        let good_path = base.join("good.mzML");
        let bad_path = base.join("bad.mzML");
        std::fs::write(&good_path, mzml(spectra.len(), &spectra)).unwrap();
        let bad: Vec<String> = spectra
            .iter()
            .map(|s| {
                let at = s.find(r#" id="scan="#).unwrap();
                format!(r#"<spectrum index="0"{}"#, &s[at..])
            })
            .collect();
        std::fs::write(&bad_path, mzml(bad.len(), &bad)).unwrap();

        let mut outs = Vec::new();
        for (tag, path, want_parallel) in [("good", &good_path, true), ("bad", &bad_path, false)] {
            let d = base.join(tag);
            std::fs::create_dir_all(&d).unwrap();
            let o = run_inner(
                ConvertParams {
                    mzml: path.to_str().unwrap(),
                    out_dir: d.to_str().unwrap(),
                    max_spectra: 0,
                    top_peaks_ms2: 0,
                    top_peaks_ms1: 0,
                    config_hash: "test",
                },
                Some(4),
            )
            .unwrap_or_else(|e| panic!("{tag}: {e:#}"));
            assert_eq!(took_parallel_path(), want_parallel, "{tag}");
            outs.push([o.ms1, o.ms2, o.isolation_windows, o.ms2_to_ms1]);
        }
        // The only difference between the two inputs is an attribute convert never
        // writes out, so the artifacts must match.
        for (a, b) in outs[0].iter().zip(outs[1].iter()) {
            assert_eq!(std::fs::read(a).unwrap(), std::fs::read(b).unwrap(), "{a}");
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The decode-path A/B, on a REAL mzML named by `MUMDIA_BENCH_MZML`.
    ///
    /// `#[ignore]`d, and it takes no fixture. The synthetic fixture above is a few
    /// megabytes, which is three chunks: at that size the measurement is process
    /// setup and the parquet write, and it cannot resolve the change at all. Saying
    /// so is the point of putting the harness here rather than a fixture benchmark
    /// that would produce a number meaning nothing.
    ///
    ///   cargo test --release -p mumdia a_b_decode_paths -- --ignored --nocapture
    ///
    /// Both arms pay for everything: the same binary, the same output directory
    /// shape, the full parquet write and the blake3 of every artifact. The only
    /// difference is the worker count. It also asserts the artifacts are
    /// byte-identical, so a run that is fast and wrong fails rather than scores.
    #[test]
    #[ignore = "needs a real mzML in MUMDIA_BENCH_MZML"]
    fn a_b_decode_paths_on_a_real_mzml() {
        let mzml = std::env::var("MUMDIA_BENCH_MZML").expect(
            "set MUMDIA_BENCH_MZML to a real mzML; the committed fixture is far too \
             small to resolve the decode paths",
        );
        let base = tmpdir("ab_decode");
        let mut hashes = Vec::new();
        for threads in [1usize, 2, 4, 8, rayon::current_num_threads()] {
            let d = base.join(format!("t{threads}"));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            let t0 = Instant::now();
            let o = run_inner(
                ConvertParams {
                    mzml: &mzml,
                    out_dir: d.to_str().unwrap(),
                    max_spectra: 0,
                    top_peaks_ms2: 0,
                    top_peaks_ms1: 0,
                    config_hash: "bench",
                },
                Some(threads),
            )
            .unwrap();
            let ms = t0.elapsed().as_millis();
            let h: Vec<u64> = [&o.ms1, &o.ms2, &o.isolation_windows, &o.ms2_to_ms1]
                .iter()
                .map(|p| {
                    let b = std::fs::read(p).unwrap();
                    let mut x = 1469598103934665603u64;
                    for byte in b {
                        x ^= byte as u64;
                        x = x.wrapping_mul(1099511628211);
                    }
                    x
                })
                .collect();
            println!(
                "threads={threads:<3} {ms:>7} ms  parallel={}",
                took_parallel_path()
            );
            hashes.push(h);
        }
        for h in &hashes[1..] {
            assert_eq!(h, &hashes[0], "a decode path produced different artifacts");
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The inflate backend, end to end through the stage.
    ///
    /// Nothing in the suite read a zlib-compressed binary array before, and every
    /// real mzML this engine sees is zlib-compressed, so the workspace could have
    /// selected a broken or absent codec and only a real file would have said so.
    /// Values, not just row counts: `defaultArrayLength` is never checked on read,
    /// so a wrong inflate produces a plausible-looking short or garbled peak list
    /// rather than an error.
    #[test]
    fn a_zlib_compressed_binary_array_decodes_to_the_right_peaks() {
        let d = tmpdir("zlib");
        let mzml_path = d.join("zlib.mzML");
        let spectra = vec![
            spectrum(0, 1, "0.10", &[300.0, 400.0], &[10.0, 20.0]),
            zlib_ms2_spectrum(1, "0.11"),
        ];
        std::fs::write(&mzml_path, mzml(2, &spectra)).unwrap();

        let out = run(ConvertParams {
            mzml: mzml_path.to_str().unwrap(),
            out_dir: d.to_str().unwrap(),
            max_spectra: 0,
            top_peaks_ms2: 0,
            top_peaks_ms1: 0,
            config_hash: "test",
        })
        .expect("a zlib-compressed mzML must convert");

        let t = mumdia_io::table::Table::read_cols(&out.ms2, &["mz", "intensity"]).unwrap();
        let mz = t.list_f32("mz").unwrap();
        let inten = t.list_f32("intensity").unwrap();
        assert_eq!(mz.len(), 1);
        assert_eq!(mz[0], vec![100.5f32, 200.25, 300.125, 400.0625]);
        assert_eq!(inten[0], vec![11.0f32, 22.0, 33.0, 44.0]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Rule 1 of this change, asserted rather than argued: the parallel decode
    /// must produce byte-identical artifacts.
    ///
    /// Not "the same rows": the same FILES. Every one of the four parquet outputs
    /// and all four `.report.json` content hashes are compared, so a difference in
    /// row order, in the isolation-window ids (which are assigned by first
    /// appearance and are the one thing a reordered fold would get wrong), in the
    /// MS2-to-MS1 links, or in the row-group boundaries would fail here.
    #[test]
    fn the_parallel_decode_is_byte_identical_to_the_sequential_one() {
        let base = tmpdir("par_eq");
        let mzml_path = base.join("run.mzML");
        // 80 cycles x (1 + 5) scans of 400 peaks: about 3.6 MB of mzML, which is
        // several 1 MiB chunks, so more than one worker gets work.
        let spectra = synthetic_run(80, 5, 400);
        std::fs::write(&mzml_path, mzml(spectra.len(), &spectra)).unwrap();
        assert!(std::fs::metadata(&mzml_path).unwrap().len() > 2 * CHUNK_TARGET_BYTES);

        let mut out_files = Vec::new();
        for (tag, threads) in [("seq", 1usize), ("par", 4usize)] {
            let d = base.join(tag);
            std::fs::create_dir_all(&d).unwrap();
            let o = run_inner(
                ConvertParams {
                    mzml: mzml_path.to_str().unwrap(),
                    out_dir: d.to_str().unwrap(),
                    max_spectra: 0,
                    top_peaks_ms2: 0,
                    top_peaks_ms1: 0,
                    config_hash: "test",
                },
                Some(threads),
            )
            .unwrap_or_else(|e| panic!("{tag}: {e:#}"));
            // The arms are only a comparison if they really are different code.
            assert_eq!(
                took_parallel_path(),
                threads > 1,
                "{tag}: wrong decode path"
            );
            out_files.push([o.ms1, o.ms2, o.isolation_windows, o.ms2_to_ms1]);
        }
        let [seq, par] = [&out_files[0], &out_files[1]];
        for (a, b) in seq.iter().zip(par.iter()) {
            let name = std::path::Path::new(a).file_name().unwrap().to_owned();
            assert_eq!(
                std::fs::read(a).unwrap(),
                std::fs::read(b).unwrap(),
                "{name:?} differs between the sequential and parallel decode"
            );
            // And the recorded content hash, which is what the manifest compares.
            let ha: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(format!("{a}.report.json")).unwrap())
                    .unwrap();
            let hb: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(format!("{b}.report.json")).unwrap())
                    .unwrap();
            assert_eq!(ha["content_hash"], hb["content_hash"], "{name:?}");
            assert_eq!(ha["rows"], hb["rows"], "{name:?}");
        }
        // And the run is not trivially empty.
        assert_eq!(rows_in_report(&seq[0]), 80);
        assert_eq!(rows_in_report(&seq[1]), 400);
        assert_eq!(rows_in_report(&seq[2]), 5);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The same equality under `--max-spectra`, which is the one place the two
    /// drivers stop counting differently (a `break` on the iteration ordinal
    /// against a bound inside a chunk).
    #[test]
    fn the_two_decode_paths_agree_under_max_spectra() {
        let base = tmpdir("par_eq_capped");
        let mzml_path = base.join("run.mzML");
        let spectra = synthetic_run(80, 5, 400);
        std::fs::write(&mzml_path, mzml(spectra.len(), &spectra)).unwrap();

        // 77 is deliberately not a chunk or cycle boundary.
        let mut outs = Vec::new();
        for (tag, threads) in [("seq", 1usize), ("par", 4usize)] {
            let d = base.join(tag);
            std::fs::create_dir_all(&d).unwrap();
            let o = run_inner(
                ConvertParams {
                    mzml: mzml_path.to_str().unwrap(),
                    out_dir: d.to_str().unwrap(),
                    max_spectra: 77,
                    top_peaks_ms2: 0,
                    top_peaks_ms1: 0,
                    config_hash: "test",
                },
                Some(threads),
            )
            .unwrap_or_else(|e| panic!("{tag}: {e:#}"));
            assert_eq!(
                took_parallel_path(),
                threads > 1,
                "{tag}: wrong decode path"
            );
            outs.push([o.ms1, o.ms2, o.isolation_windows, o.ms2_to_ms1]);
        }
        for (a, b) in outs[0].iter().zip(outs[1].iter()) {
            assert_eq!(std::fs::read(a).unwrap(), std::fs::read(b).unwrap(), "{a}");
        }
        // 77 spectra = 12 full cycles (72) plus MS1 + 4 MS2.
        assert_eq!(rows_in_report(&outs[0][0]), 13);
        assert_eq!(rows_in_report(&outs[0][1]), 64);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// A trailing non-finite retention time is the bug fixed below, and the
    /// parallel driver has its own `read` accounting, so it gets the same test.
    #[test]
    fn the_parallel_decode_also_survives_a_trailing_bad_rt() {
        let base = tmpdir("par_tail_nan");
        let mzml_path = base.join("run.mzML");
        let mut spectra = synthetic_run(80, 5, 400);
        let last = spectra.len() - 1;
        spectra[last] = spectra[last].replace(r#"value="0.790000""#, r#"value="NaN""#);
        assert!(spectra[last].contains(r#"value="NaN""#));
        std::fs::write(&mzml_path, mzml(spectra.len(), &spectra)).unwrap();

        let d = base.join("out");
        std::fs::create_dir_all(&d).unwrap();
        let o = run_inner(
            ConvertParams {
                mzml: mzml_path.to_str().unwrap(),
                out_dir: d.to_str().unwrap(),
                max_spectra: 0,
                top_peaks_ms2: 0,
                top_peaks_ms1: 0,
                config_hash: "test",
            },
            Some(4),
        )
        .expect("a whole file with one unusable retention time must convert");
        assert!(took_parallel_path());
        assert_eq!(rows_in_report(&o.ms2), 399);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The bug this pins: `read` was advanced by the LAST statement of the loop
    /// body, and the non-finite-RT arm reaches it through `continue`. A bad
    /// retention time in the final spectrum therefore left `read` one short of
    /// `declared`, and the completeness check called a whole file truncated and
    /// told the user to re-transfer a multi-gigabyte mzML.
    ///
    /// The POSITION is the whole bug. A bad RT anywhere else is re-covered by the
    /// next good spectrum, which assigns `read` again, so only a trailing run of
    /// them bites and only this arrangement fails.
    #[test]
    fn a_non_finite_rt_in_the_last_spectrum_is_not_a_truncated_file() {
        let d = tmpdir("tail_nan_rt");
        let mzml_path = d.join("tail.mzML");
        std::fs::write(
            &mzml_path,
            mzml(4, &four_spectra(["0.10", "0.11", "0.20", "NaN"])),
        )
        .unwrap();

        let out = run(ConvertParams {
            mzml: mzml_path.to_str().unwrap(),
            out_dir: d.to_str().unwrap(),
            max_spectra: 0,
            top_peaks_ms2: 0,
            top_peaks_ms1: 0,
            config_hash: "test",
        })
        .expect("a whole file with one unusable retention time must convert");

        // The one bad scan is dropped, as the RT guard intends; everything else
        // is kept, and nothing declares the file truncated.
        assert_eq!(rows_in_report(&out.ms1), 2);
        assert_eq!(rows_in_report(&out.ms2), 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// `ConvertOutputs::hashes` is what an orchestrator records in the manifest instead of
    /// hashing the four files again, so each field must be the hash of the file at the
    /// path of the same name, and the hash its own report records. A swap between two
    /// fields would put one artifact's hash on another's record; the four files differ,
    /// so any swap fails here.
    #[test]
    fn the_returned_hashes_are_those_of_the_files_written() {
        let d = tmpdir("hashes");
        let mzml_path = d.join("run.mzML");
        std::fs::write(
            &mzml_path,
            mzml(4, &four_spectra(["0.10", "0.11", "0.20", "0.21"])),
        )
        .unwrap();
        let out = run(ConvertParams {
            mzml: mzml_path.to_str().unwrap(),
            out_dir: d.to_str().unwrap(),
            max_spectra: 0,
            top_peaks_ms2: 0,
            top_peaks_ms1: 0,
            config_hash: "test",
        })
        .unwrap();
        for (path, hash) in [
            (&out.ms1, &out.hashes.ms1),
            (&out.ms2, &out.hashes.ms2),
            (&out.isolation_windows, &out.hashes.isolation_windows),
            (&out.ms2_to_ms1, &out.hashes.ms2_to_ms1),
        ] {
            assert_eq!(hash, &mumdia_io::hash::blake3_file(path).unwrap(), "{path}");
            let rep: mumdia_io::report::ArtifactReport =
                mumdia_io::json::read_json(&format!("{path}.report.json")).unwrap();
            assert_eq!(hash, &rep.content_hash, "{path}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The same shape one scan earlier, which the old code survived because the
    /// following good spectrum re-assigned `read`. Kept so the pair states where
    /// the boundary was rather than only that it moved.
    #[test]
    fn a_non_finite_rt_in_the_middle_was_already_fine_and_stays_fine() {
        let d = tmpdir("mid_nan_rt");
        let mzml_path = d.join("mid.mzML");
        std::fs::write(
            &mzml_path,
            mzml(4, &four_spectra(["0.10", "NaN", "0.20", "0.21"])),
        )
        .unwrap();

        let out = run(ConvertParams {
            mzml: mzml_path.to_str().unwrap(),
            out_dir: d.to_str().unwrap(),
            max_spectra: 0,
            top_peaks_ms2: 0,
            top_peaks_ms1: 0,
            config_hash: "test",
        })
        .expect("a mid-file bad retention time was never a truncation");
        assert_eq!(rows_in_report(&out.ms1), 2);
        assert_eq!(rows_in_report(&out.ms2), 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The guard the fix must not disable: a file that really does end early
    /// still refuses. `count="5"` against four spectra present.
    #[test]
    fn a_genuinely_short_file_still_refuses() {
        let d = tmpdir("short");
        let mzml_path = d.join("short.mzML");
        std::fs::write(
            &mzml_path,
            mzml(5, &four_spectra(["0.10", "0.11", "0.20", "0.21"])),
        )
        .unwrap();

        let res = run(ConvertParams {
            mzml: mzml_path.to_str().unwrap(),
            out_dir: d.to_str().unwrap(),
            max_spectra: 0,
            top_peaks_ms2: 0,
            top_peaks_ms1: 0,
            config_hash: "test",
        });
        let Err(err) = res else {
            panic!("a file shorter than its own header must be refused");
        };
        let msg = format!("{err:#}");
        assert!(msg.contains("truncated or corrupt"), "{msg}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// `--max-spectra` is a deliberate short read and stays excluded from the
    /// completeness check, including now that `read` advances earlier in the
    /// body: the `break` still precedes the assignment, so `capped` is unchanged.
    #[test]
    fn max_spectra_still_suppresses_the_completeness_check() {
        let d = tmpdir("capped");
        let mzml_path = d.join("capped.mzML");
        std::fs::write(
            &mzml_path,
            mzml(4, &four_spectra(["0.10", "0.11", "0.20", "0.21"])),
        )
        .unwrap();

        let out = run(ConvertParams {
            mzml: mzml_path.to_str().unwrap(),
            out_dir: d.to_str().unwrap(),
            max_spectra: 2,
            top_peaks_ms2: 0,
            top_peaks_ms1: 0,
            config_hash: "test",
        })
        .expect("--max-spectra is a deliberate short read");
        assert_eq!(rows_in_report(&out.ms1), 1);
        assert_eq!(rows_in_report(&out.ms2), 1);
        let _ = std::fs::remove_dir_all(&d);
    }
}
