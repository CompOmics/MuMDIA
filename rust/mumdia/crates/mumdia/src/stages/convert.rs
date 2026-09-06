//! Stage 0 `mumdia convert`: read an mzML run into the normalized spectra
//! artifact set (docs/04_convert.md). MVP is mzML-only and 3D, so ion-mobility
//! columns are absent. Profile spectra are centroided (simple local-maxima)
//! so downstream matching sees discrete peaks.

use std::time::Instant;

use anyhow::{Context, Result};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, TableWriter};
use mzdata::prelude::*;
use mzdata::spectrum::SignalContinuity;
use serde_json::json;
use tracing::{info, warn};

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

pub fn run(p: ConvertParams) -> Result<ConvertOutputs> {
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
    let mut ms1_w = TableWriter::new(&ms1_path).with_row_group_rows(SPECTRA_CHUNK);
    let mut ms2_w = TableWriter::new(&ms2_path).with_row_group_rows(SPECTRA_CHUNK);
    let mut ms1 = Ms1Chunk::default();
    let mut ms2 = Ms2Chunk::default();
    // Distinct isolation windows (id, target, lower, upper); ids by first appearance in
    // scan order, as before, now assigned on the fly.
    let mut uniq: Vec<(u64, f64, f64, f64)> = Vec::new();
    let mut seen: std::collections::HashMap<(u64, u64), u32> = std::collections::HashMap::new();
    // MS2 -> preceding MS1 scan map (two ints per MS2 scan; kept whole).
    let (mut map_ms2, mut map_ms1) = (Vec::new(), Vec::new());

    let mut last_ms1_index: Option<u32> = None;
    let mut read = 0usize;
    // Peaks dropped for a non-finite m/z or intensity. Counted rather than ignored:
    // a handful is a damaged spectrum, and a large fraction means the file or the
    // converter that produced it is wrong, which is worth knowing before the numbers
    // are believed.
    let mut nonfinite_peaks = 0usize;
    // Spectra dropped for a non-finite retention time, and the first one's id.
    //
    // Retention time was the one externally supplied float this stage did not check,
    // and it is the value everything downstream keys on. A single `NaN` scan start time
    // in one scan of an mzML -- reproduced by editing one `scan start time` value in the
    // fixture -- passed convert without a warning, was written into the spectra
    // artifact, and then aborted `mumdia run` inside extract with
    // `called `Option::unwrap()` on a `None` value` at `extract.rs`, naming neither the
    // file, nor the scan, nor the value. The peak filter in `peaks_of` had exactly this
    // hole closed for m/z and intensity; RT was missed.
    //
    // Dropped rather than fatal, for the same reason as a bad peak: one unusable scan is
    // not a reason to refuse a run, and a spectrum with no retention time cannot be
    // placed in a chromatogram, so there is nothing else to do with it. The count and
    // the first offending scan id are reported, and the existing "no MS2 spectra" bail
    // below is the backstop for a file whose retention times are ALL unusable.
    let mut nonfinite_rt = 0usize;
    let mut first_bad_rt: Option<String> = None;
    for (count, (scan_index, spec)) in (0_u32..).zip(reader).enumerate() {
        if p.max_spectra > 0 && count >= p.max_spectra {
            break;
        }
        let rt_s = spec.start_time() * 60.0; // mzdata returns minutes
        if !rt_s.is_finite() {
            nonfinite_rt += 1;
            if first_bad_rt.is_none() {
                first_bad_rt = Some(format!("{} (rt={})", spec.id(), rt_s));
            }
            continue;
        }
        match spec.ms_level() {
            1 => {
                let (mz, inten, nf) = peaks_of(&spec, p.top_peaks_ms1);
                nonfinite_peaks += nf;
                ms1.idx.push(scan_index);
                ms1.rt.push(rt_s);
                ms1.mz.push(mz);
                ms1.inten.push(inten);
                last_ms1_index = Some(scan_index);
                if ms1.idx.len() >= SPECTRA_CHUNK {
                    ms1_w.write_cols(ms1.cols())?;
                }
            }
            2 => {
                let (mz, inten, nf) = peaks_of(&spec, p.top_peaks_ms2);
                nonfinite_peaks += nf;
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
                let win_id = *seen.entry((wl.to_bits(), wu.to_bits())).or_insert_with(|| {
                    let id = uniq.len() as u32;
                    uniq.push((id as u64, wt, wl, wu));
                    id
                });
                ms2.idx.push(scan_index);
                ms2.id.push(spec.id().to_string());
                ms2.rt.push(rt_s);
                ms2.win_id.push(win_id);
                ms2.wt.push(wt);
                ms2.wl.push(wl);
                ms2.wu.push(wu);
                ms2.pmz.push(pmz);
                ms2.pz.push(pz);
                ms2.mz.push(mz);
                ms2.inten.push(inten);
                map_ms2.push(scan_index);
                map_ms1.push(last_ms1_index.map(|x| x as i32).unwrap_or(-1));
                if ms2.idx.len() >= SPECTRA_CHUNK {
                    ms2_w.write_cols(ms2.cols())?;
                }
            }
            _ => {}
        }
        read = count + 1;
    }
    // Final chunks (possibly empty: they fix the schema for a level with no scans). The
    // writers are closed (published) only after the checks below: an abandoned
    // `TableWriter` removes its temp file, so a refused run leaves no half-written
    // spectra artifact behind.
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
    write_reports(
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
    Ok(ConvertOutputs {
        ms1: ms1_path,
        ms2: ms2_path,
        isolation_windows: iw_path,
        ms2_to_ms1: map_path,
    })
}
fn write_reports(
    items: &[(&String, (&str, u32), u64)],
    elapsed_ms: u128,
    params: serde_json::Value,
) -> Result<()> {
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
    }
    Ok(())
}
