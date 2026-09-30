//! Chromatogram traces rebuilt from the raw timsTOF events (`retrace`, diaPASEF only;
//! docs/TIMS_ROADMAP_bis.md).
//!
//! Extract's trace point is the intensity of one centroid, and convert's centroiding
//! drops or merges weak signal (single-linkage m/z chaining, `tdf_min_points`). Here
//! every point of a fragment trace becomes the sum of all raw events in the grid point's
//! frame, in the quad slots whose isolation window holds the precursor, within the
//! learned fragment tolerance of the calibrated fragment m/z and inside the candidate's
//! 1/K0 band `apex_im +/- retrace.im_half_width`; the point's `im` is their
//! intensity-weighted 1/K0. The MS1 isotope traces (`ms1_mono` / `ms1_iso1` /
//! `ms1_iso2`) are rebuilt from the MS1 frame nearest each grid point, as extract samples
//! them, within `extract.prec_tol_ppm` and the MS1 band. Every other row and column is
//! copied, and the rows keep their order, so features (which joins the chromatograms by
//! stream order) and quant read the result unchanged. Label-blind.
//!
//! The frames inside the grids' RT range are decoded once and held (about the raw size
//! of the run); the chromatograms are then streamed one row group at a time, so memory
//! does not grow with the number of trace points.

use std::time::Instant;

use anyhow::{bail, Context, Result};
use mumdia_core::config::RetraceConfig;
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{Col, ParallelTableWriter, TableFile};
use rayon::prelude::*;
use serde_json::json;
use std::sync::Arc;
use timsrust::{Frame, MSLevel, QuadrupoleSettings};
use tracing::{info, warn};

use crate::stages::convert::RawTdf;
use crate::stages::extract::{nearest_index, read_mass_cal};

/// Frames within this of the grid's RT range are decoded (the nearest MS1 frame of an
/// edge grid point can lie just outside it).
const RT_MARGIN_S: f64 = 5.0;
/// Rows per parquet row group, as extract writes the chromatograms.
const ROW_GROUP_ROWS: usize = 1 << 16;
/// Row groups the writer encodes at once, one thread each: the parquet encode of the
/// list columns is otherwise the slowest stage of the pipeline.
const ENCODE_AT_ONCE: usize = 16;
/// Reader threads decoding the input row groups (the read is one thread per row group).
const READERS: usize = 8;
/// A fragment grid point must sit on an MS2 frame's RT within this (f32 rounding of the
/// frame RT is ~1e-4 s at the end of a gradient; frames are ~60 ms apart).
const RT_MATCH_S: f64 = 1e-3;

pub struct RetraceParams<'a> {
    /// The diaPASEF `.d` the run was converted from.
    pub raw: &'a str,
    pub chromatograms: &'a str,
    pub psms_extracted: &'a str,
    pub run_windows: &'a str,
    pub library_precursors: &'a str,
    /// search-seed `masscal.json`; `frag_tol_fallback_ppm` without one.
    pub mass_cal: Option<&'a str>,
    pub frag_tol_fallback_ppm: f64,
    /// `extract.prec_tol_ppm`, the MS1 tolerance extract's traces used.
    pub prec_tol_ppm: f64,
    pub out: &'a str,
    pub cfg: &'a RetraceConfig,
    pub config_hash: &'a str,
}

/// The `.d` a spectra artifact was converted from, from its report (`params.mzml`, with
/// `params.reader = timsrust`). `None` for any other reader: nothing to retrace.
pub fn raw_path_from_spectra(ms2_spectra: &str) -> Result<Option<String>> {
    let rep = format!("{ms2_spectra}.report.json");
    let v: serde_json::Value = mumdia_io::json::read_json(&rep)?;
    let params = &v["params"];
    if params["reader"].as_str() != Some("timsrust") {
        return Ok(None);
    }
    Ok(params["mzml"].as_str().map(str::to_string))
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Copy,
    Frag,
    Ms1,
}

/// One output row: its kind, intensity and 1/K0 lists, raw events summed, points without a
/// decoded frame.
type RowOut = (Kind, Vec<f32>, Vec<f32>, u64, u64);
/// One planned grid point: its decoded frame (`None`: no frame at that RT) and the scan
/// ranges to sum in it.
type Point<'a> = (Option<&'a TofFrame>, Vec<(usize, usize)>);

/// One row group of the input chromatograms, flat.
struct SpanIn {
    n: usize,
    cid: Vec<u32>,
    name_off: Vec<usize>,
    names: String,
    fmz: Vec<f64>,
    obs: Vec<f64>,
    pint: Vec<f32>,
    rt_off: Vec<usize>,
    rt: Vec<f32>,
    int_off: Vec<usize>,
    int: Vec<f32>,
    im_off: Vec<usize>,
    im: Vec<f32>,
}

impl SpanIn {
    fn read(ch: &TableFile, a: usize, n: usize) -> Result<SpanIn> {
        let t = ch.span(a, n)?;
        let (name_off, names) = t.str_flat("frag_name")?;
        let (rt_off, rt) = t.list_f32_flat("rt")?;
        let (int_off, int) = t.list_f32_flat("intensity")?;
        let (im_off, im) = t.list_f32_flat("im")?;
        Ok(SpanIn {
            n,
            cid: t.u32("candidate_id")?,
            name_off,
            names,
            fmz: t.f64("frag_mz")?,
            obs: t.f64("frag_obs_mz")?,
            pint: t.f32("predicted_intensity")?,
            rt_off,
            rt,
            int_off,
            int,
            im_off,
            im,
        })
    }
}

/// One rebuilt point: summed intensity, intensity-weighted 1/K0 sum, raw event count.
#[derive(Default, Clone, Copy, Debug, PartialEq)]
struct Sum {
    inten: f64,
    imw: f64,
    n: u32,
}

/// First scan (inclusive) and last scan (exclusive) with 1/K0 in `[lo, hi]`, given
/// `scan_im` strictly decreasing in the scan index, as TIMS mobility is.
fn scan_band(scan_im: &[f64], lo: f64, hi: f64) -> (usize, usize) {
    let a = scan_im.partition_point(|&v| v > hi);
    let b = scan_im.partition_point(|&v| v >= lo);
    (a, b.max(a))
}

/// A held frame: its quad settings and its raw events sorted by TOF, each packed as
/// `tof << 44 | scan << 32 | intensity`, so a TOF window is one binary search over the whole
/// frame instead of two per TIMS scan in the band. 8 bytes per event, as the decoded frame.
struct TofFrame {
    quad: Arc<QuadrupoleSettings>,
    ev: Vec<u64>,
}

const TOF_SHIFT: u32 = 44;
const SCAN_SHIFT: u32 = 32;

impl TofFrame {
    fn new(f: Frame) -> Result<TofFrame> {
        let n_scans = f.scan_offsets.len().saturating_sub(1);
        if n_scans >= 1 << (TOF_SHIFT - SCAN_SHIFT) {
            bail!(
                "frame {}: {n_scans} scans do not fit the event packing",
                f.index
            );
        }
        let mut ev = Vec::with_capacity(f.tof_indices.len());
        for sc in 0..n_scans {
            for k in f.scan_offsets[sc]..f.scan_offsets[sc + 1] {
                let t = f.tof_indices[k] as u64;
                if t >= 1 << (64 - TOF_SHIFT) {
                    bail!(
                        "frame {}: TOF index {t} does not fit the event packing",
                        f.index
                    );
                }
                ev.push(t << TOF_SHIFT | (sc as u64) << SCAN_SHIFT | f.intensities[k] as u64);
            }
        }
        ev.sort_unstable();
        Ok(TofFrame {
            quad: f.quadrupole_settings,
            ev,
        })
    }
}

/// Sum the raw events of `f` with TOF index in `[t_lo, t_hi]` whose scan lies in one of
/// `scans` (half-open ranges).
fn sum_events(
    f: &TofFrame,
    scans: &[(usize, usize)],
    t_lo: u32,
    t_hi: u32,
    scan_im: &[f64],
) -> Sum {
    let mut s = Sum::default();
    let i0 = f.ev.partition_point(|&e| e < (t_lo as u64) << TOF_SHIFT);
    for &e in &f.ev[i0..] {
        if (e >> TOF_SHIFT) as u32 > t_hi {
            break;
        }
        let sc = ((e >> SCAN_SHIFT) & ((1 << (TOF_SHIFT - SCAN_SHIFT)) - 1)) as usize;
        if sc < scan_im.len() && scans.iter().any(|&(a, b)| (a..b).contains(&sc)) {
            let w = (e & 0xffff_ffff) as f64;
            s.inten += w;
            s.imw += w * scan_im[sc];
            s.n += 1;
        }
    }
    s
}

/// The scan ranges of the quad slots whose isolation window holds `pmz`, clipped to the
/// band `(a, b)`, merged so an event is never counted twice.
fn slot_scans(q: &QuadrupoleSettings, pmz: f64, (a, b): (usize, usize)) -> Vec<(usize, usize)> {
    let mut r: Vec<(usize, usize)> = (0..q.len())
        .filter(|&k| (pmz - q.isolation_mz[k]).abs() <= q.isolation_width[k] / 2.0)
        .map(|k| (q.scan_starts[k].max(a), q.scan_ends[k].min(b)))
        .filter(|&(x, y)| x < y)
        .collect();
    r.sort_unstable();
    let mut m: Vec<(usize, usize)> = Vec::with_capacity(r.len());
    for (x, y) in r {
        match m.last_mut() {
            Some(last) if x <= last.1 => last.1 = last.1.max(y),
            _ => m.push((x, y)),
        }
    }
    m
}

/// Inclusive TOF index window of the m/z interval `[lo, hi]`.
fn tof_window(raw: &RawTdf, lo: f64, hi: f64) -> (u32, u32) {
    let a = raw.tof_of(lo).ceil().max(0.0);
    let b = raw.tof_of(hi).floor().max(0.0);
    (a as u32, b as u32)
}

/// Per-candidate lookups indexed by `candidate_id`, NaN where absent.
fn by_candidate(ids: &[u32], vals: impl Iterator<Item = f64>, n: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; n];
    for (&c, x) in ids.iter().zip(vals) {
        if let Some(slot) = v.get_mut(c as usize) {
            *slot = x;
        }
    }
    v
}

pub fn run(p: RetraceParams) -> Result<u64> {
    let t0 = Instant::now();
    mumdia_io::refuse_output_over_input(
        p.out,
        &[
            ("--chromatograms", p.chromatograms),
            ("--psms-extracted", p.psms_extracted),
        ],
    )?;
    let raw = RawTdf::open(p.raw)?;
    let (mass_off, frag_tol) = read_mass_cal(p.mass_cal, p.frag_tol_fallback_ppm)?;

    // Frame index without decoding: RT, MS level, quad settings.
    let n_frames = raw.reader.len();
    let heads: Vec<Frame> = (0..n_frames)
        .into_par_iter()
        .map(|i| raw.reader.get_frame_without_coordinates(i))
        .collect::<std::result::Result<_, _>>()
        .with_context(|| format!("{}: frame headers", p.raw))?;
    let (mut ms2_rt, mut ms2_fi, mut ms1_rt, mut ms1_fi) = (vec![], vec![], vec![], vec![]);
    for (i, f) in heads.iter().enumerate() {
        match f.ms_level {
            MSLevel::MS1 => {
                ms1_rt.push(f.rt_in_seconds);
                ms1_fi.push(i);
            }
            MSLevel::MS2 => {
                ms2_rt.push(f.rt_in_seconds);
                ms2_fi.push(i);
            }
            MSLevel::Unknown => {}
        }
    }
    let ascending = |v: &[f64]| v.windows(2).all(|w| w[0] <= w[1]);
    if !(ascending(&ms1_rt) && ascending(&ms2_rt)) || ms1_rt.is_empty() || ms2_rt.is_empty() {
        bail!(
            "{}: frames are not in RT order, or MS1/MS2 frames are missing",
            p.raw
        );
    }
    let n_scans = raw
        .reader
        .get(ms1_fi[0])
        .with_context(|| format!("{}: frame {}", p.raw, ms1_fi[0]))?
        .scan_offsets
        .len()
        .saturating_sub(1);
    let scan_im: Vec<f64> = (0..n_scans).map(|s| raw.im_of_scan(s as f64)).collect();
    if !scan_im.windows(2).all(|w| w[0] > w[1]) {
        bail!(
            "{}: 1/K0 is not strictly decreasing in the scan index",
            p.raw
        );
    }

    // Per candidate: precursor m/z and the band centre.
    let lib = TableFile::open(p.library_precursors)?;
    let lib_cid = lib.u32("candidate_id")?;
    let n_cand = lib_cid.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
    let pmz = by_candidate(&lib_cid, lib.f64("precursor_mz")?.into_iter(), n_cand);
    let rw = TableFile::open(p.run_windows)?;
    let rw_cid = rw.u32("candidate_id")?;
    let rw_col = |name: &str| -> Result<Vec<f64>> {
        let v = rw.opt_f64(name)?;
        Ok(by_candidate(
            &rw_cid,
            v.into_iter().map(|x| x.unwrap_or(f64::NAN)),
            n_cand,
        ))
    };
    let (rt_lo_c, rt_hi_c) = (rw_col("rt_lo")?, rw_col("rt_hi")?);
    let mut centre = by_candidate(
        &rw_cid,
        rw.opt_f64("im_pred_cal")?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
        n_cand,
    );
    let ex = TableFile::open(p.psms_extracted)?;
    let (ex_cid, ex_rank, ex_im) = (
        ex.u32("candidate_id")?,
        ex.i32("peak_rank")?,
        ex.opt_f64("apex_im")?,
    );
    for i in 0..ex.nrows {
        if let (0, Some(v)) = (ex_rank[i], ex_im[i]) {
            if v.is_finite() {
                if let Some(slot) = centre.get_mut(ex_cid[i] as usize) {
                    *slot = v;
                }
            }
        }
    }

    // Decode, once, every frame inside the RT range of the candidates' grids (plus a
    // margin, for the nearest-MS1-frame lookup) and hold them: about the raw size of that
    // part of the run, whatever the number of trace points (HYE has 30x E. coli's).
    let ch = TableFile::open(p.chromatograms)?;
    if !ch.has_column("im") {
        bail!(
            "{}: no per-point `im` column; retrace needs a 4D (diaPASEF) run",
            p.chromatograms
        );
    }
    // A candidate's grid lies inside its RT window, so the extracted candidates' windows
    // bound the frames needed; one without a finite window needs the whole run.
    let (mut rt_min, mut rt_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &c in &ex_cid {
        let (lo, hi) = (rt_lo_c.get(c as usize), rt_hi_c.get(c as usize));
        match (lo, hi) {
            (Some(&lo), Some(&hi)) if lo.is_finite() && hi.is_finite() => {
                rt_min = rt_min.min(lo);
                rt_max = rt_max.max(hi);
            }
            _ => {
                (rt_min, rt_max) = (f64::NEG_INFINITY, f64::INFINITY);
                break;
            }
        }
    }
    let (rt_min, rt_max) = (rt_min - RT_MARGIN_S, rt_max + RT_MARGIN_S);
    // ponytail: all needed frames are resident, so the peak is about the run's raw size
    // (21 GB on the E. coli diaPASEF run, 36 GB on a HYE run). Past that: decode RT blocks of frames and visit
    // the rows by RT, which needs the rows grouped by RT (an external sort) first.
    let frames: Vec<Option<TofFrame>> = heads
        .par_iter()
        .enumerate()
        .map(|(i, h)| -> Result<Option<TofFrame>> {
            if !(rt_min..=rt_max).contains(&h.rt_in_seconds) {
                return Ok(None);
            }
            let f = raw
                .reader
                .get(i)
                .with_context(|| format!("{}: frame {i}", p.raw))?;
            Ok(Some(TofFrame::new(f)?))
        })
        .collect::<Result<_>>()?;
    drop(heads);
    let decoded = frames.iter().filter(|f| f.is_some()).count();
    info!(frames = decoded, rt_min, rt_max, "retrace: frames decoded");

    let (ms1_rt, ms1_fi, ms2_rt, ms2_fi) = (&ms1_rt, &ms1_fi, &ms2_rt, &ms2_fi);
    let (hw, hw1) = (p.cfg.im_half_width, p.cfg.ms1_im_half_width);
    let nrows = ch.nrows;
    let (mut n_frag, mut n_ms1, mut unmatched, mut events) = (0u64, 0u64, 0u64, 0u64);
    let mut t_sum = 0u128;
    let t_decoded = t0.elapsed().as_millis();
    // Three stages over row groups, joined by bounded channels so read, sum and write
    // overlap: a reader thread, the parallel sum here, a writer thread. Rows keep the
    // input order. The writer publishes only on the explicit end marker (`None`), so an
    // error anywhere leaves no partial table.
    let n_spans = nrows.max(1).div_ceil(ROW_GROUP_ROWS);
    let (tx_out, rx_out) = std::sync::mpsc::sync_channel::<Option<Vec<Col>>>(ENCODE_AT_ONCE);
    let out_path = p.out.to_string();
    let (rows, t_read, t_write) = std::thread::scope(|sc| -> Result<(u64, u128, u128)> {
        // READERS threads take the spans in turn; the sum receives them round-robin, which
        // is the input order.
        let ch = &ch;
        let (mut rx_ins, mut readers) = (Vec::new(), Vec::new());
        for j in 0..READERS {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Result<SpanIn>>(1);
            rx_ins.push(rx);
            readers.push(sc.spawn(move || {
                let mut t_read = 0u128;
                for i in (j..n_spans).step_by(READERS) {
                    let a = i * ROW_GROUP_ROWS;
                    let tp = Instant::now();
                    let span = SpanIn::read(ch, a, ROW_GROUP_ROWS.min(nrows - a));
                    t_read += tp.elapsed().as_millis();
                    if tx.send(span).is_err() {
                        break;
                    }
                }
                t_read
            }));
        }
        let writer = sc.spawn(move || -> Result<Option<(u64, u128)>> {
            let mut w = ParallelTableWriter::new(&out_path, ROW_GROUP_ROWS);
            let (mut pending, mut t_write) = (Vec::with_capacity(ENCODE_AT_ONCE), 0u128);
            loop {
                let msg = rx_out.recv();
                let end = matches!(msg, Ok(None));
                match msg {
                    Ok(Some(cols)) => pending.push(cols),
                    Ok(None) => {}
                    Err(_) => return Ok(None), // aborted upstream: publish nothing
                }
                if pending.len() == ENCODE_AT_ONCE || (end && !pending.is_empty()) {
                    let tp = Instant::now();
                    w.write_row_groups(std::mem::take(&mut pending))?;
                    t_write += tp.elapsed().as_millis();
                }
                if end {
                    let tp = Instant::now();
                    let rows = w.close()?;
                    return Ok(Some((rows, t_write + tp.elapsed().as_millis())));
                }
            }
        });
        for i in 0..n_spans {
            let span = rx_ins[i % READERS]
                .recv()
                .map_err(|_| anyhow::anyhow!("retrace: a reader stopped early"))?;
            let SpanIn {
                n,
                cid,
                name_off,
                names,
                fmz,
                obs,
                pint,
                rt_off,
                rt,
                int_off,
                int,
                im_off,
                im,
            } = span?;
            let name = |r: usize| &names[name_off[r]..name_off[r + 1]];
            let tp = Instant::now();
            // Rows of one candidate are contiguous and share one grid, so the frame and the
            // quad-slot scan ranges of every grid point are planned once per candidate (per
            // distinct grid) and reused by each of its fragment rows.
            let mut groups: Vec<(usize, usize)> = Vec::new();
            for r in 0..n {
                match groups.last_mut() {
                    Some(g) if cid[g.0] == cid[r] => g.1 = r + 1,
                    _ => groups.push((r, r + 1)),
                }
            }
            let plan = |pts: &[f32], frag: bool, c: usize| -> Vec<Point> {
                let band = if frag {
                    scan_band(&scan_im, centre[c] - hw, centre[c] + hw)
                } else {
                    scan_band(&scan_im, centre[c] - hw1, centre[c] + hw1)
                };
                pts.iter()
                    .map(|&tf| {
                        let t = tf as f64;
                        let fi = if frag {
                            let j = nearest_index(ms2_rt, t);
                            ((ms2_rt[j] - t).abs() <= RT_MATCH_S).then(|| ms2_fi[j])
                        } else {
                            Some(ms1_fi[nearest_index(ms1_rt, t)])
                        };
                        match fi.and_then(|i| frames[i].as_ref()) {
                            Some(f) if frag => (Some(f), slot_scans(&f.quad, pmz[c], band)),
                            Some(f) => (Some(f), vec![band]),
                            None => (None, Vec::new()),
                        }
                    })
                    .collect()
            };
            let rows: Vec<RowOut> = groups
                .par_iter()
                .map(|&(g0, g1)| {
                    let mut cache: [Option<(&[f32], Vec<Point>)>; 2] = [None, None];
                    (g0..g1)
                        .map(|r| {
                            let pts = &rt[rt_off[r]..rt_off[r + 1]];
                            let old_int = &int[int_off[r]..int_off[r + 1]];
                            let old_im = &im[im_off[r]..im_off[r + 1]];
                            let c = cid[r] as usize;
                            let ok = !pts.is_empty()
                                && c < n_cand
                                && pmz[c].is_finite()
                                && centre[c].is_finite();
                            let ms1 = name(r).starts_with("ms1_");
                            let kind = match (ok, ms1) {
                                (true, true) if p.cfg.ms1 => Kind::Ms1,
                                (true, false) if pint[r] > 0.0 => Kind::Frag,
                                _ => Kind::Copy,
                            };
                            if kind == Kind::Copy {
                                return (kind, old_int.to_vec(), old_im.to_vec(), 0, 0);
                            }
                            let frag = kind == Kind::Frag;
                            let slot = &mut cache[frag as usize];
                            if slot.as_ref().is_none_or(|(g, _)| *g != pts) {
                                *slot = Some((pts, plan(pts, frag, c)));
                            }
                            let points = &slot.as_ref().expect("planned above").1;
                            let (t_lo, t_hi) = if frag {
                                let m = fmz[r] * mass_off.factor_at(fmz[r]);
                                let tol = frag_tol * 1e-6;
                                tof_window(&raw, m * (1.0 - tol), m * (1.0 + tol))
                            } else {
                                let (m, tol) = (fmz[r], p.prec_tol_ppm * 1e-6);
                                tof_window(&raw, m * (1.0 - tol), m * (1.0 + tol))
                            };
                            let (mut ints, mut ims) = (Vec::with_capacity(pts.len()), Vec::new());
                            let (mut ev, mut miss) = (0u64, 0u64);
                            for (k, (f, scans)) in points.iter().enumerate() {
                                let s = match f {
                                    Some(f) => sum_events(f, scans, t_lo, t_hi, &scan_im),
                                    None => {
                                        miss += 1;
                                        Sum::default()
                                    }
                                };
                                ev += s.n as u64;
                                let hit = s.n >= p.cfg.min_events && s.inten > 0.0;
                                ints.push(if hit { s.inten as f32 } else { 0.0 });
                                if frag {
                                    ims.push(if hit {
                                        (s.imw / s.inten) as f32
                                    } else if old_im.len() == pts.len() {
                                        old_im[k]
                                    } else {
                                        0.0
                                    });
                                }
                            }
                            if !frag {
                                ims = old_im.to_vec();
                            }
                            (kind, ints, ims, ev, miss)
                        })
                        .collect::<Vec<RowOut>>()
                })
                .flatten_iter()
                .collect();
            let (mut ints, mut ims) = (Vec::with_capacity(n), Vec::with_capacity(n));
            for (kind, i, m, ev, miss) in rows {
                match kind {
                    Kind::Frag => n_frag += 1,
                    Kind::Ms1 => n_ms1 += 1,
                    Kind::Copy => {}
                }
                events += ev;
                unmatched += miss;
                ints.push(i);
                ims.push(m);
            }
            t_sum += tp.elapsed().as_millis();
            let cols = vec![
                Col::U32("candidate_id".into(), cid),
                Col::Str(
                    "frag_name".into(),
                    (0..n).map(|r| name(r).to_string()).collect(),
                ),
                Col::F64("frag_mz".into(), fmz),
                Col::F64("frag_obs_mz".into(), obs),
                Col::F32("predicted_intensity".into(), pint),
                Col::LargeListF32(
                    "rt".into(),
                    (0..n)
                        .map(|r| rt[rt_off[r]..rt_off[r + 1]].to_vec())
                        .collect(),
                ),
                Col::LargeListF32("intensity".into(), ints),
                Col::LargeListF32("im".into(), ims),
            ];
            if tx_out.send(Some(cols)).is_err() {
                break;
            }
        }
        let sent_end = tx_out.send(None).is_ok();
        drop(rx_ins);
        let mut t_read = 0u128;
        for r in readers {
            t_read += r
                .join()
                .map_err(|_| anyhow::anyhow!("retrace: a reader panicked"))?;
        }
        let written = writer
            .join()
            .map_err(|_| anyhow::anyhow!("retrace: writer panicked"))??;
        match (sent_end, written) {
            (true, Some((rows, t_write))) => Ok((rows, t_read, t_write)),
            _ => bail!("retrace: the writer stopped before the end; nothing published"),
        }
    })?;
    if unmatched > 0 {
        warn!(
            points = unmatched,
            "retrace: grid points without a decoded frame at their RT; set to 0"
        );
    }

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("rows".to_string(), json!(rows));
    stats.insert("rebuilt_fragment_rows".to_string(), json!(n_frag));
    stats.insert("rebuilt_ms1_rows".to_string(), json!(n_ms1));
    stats.insert("copied_rows".to_string(), json!(rows - n_frag - n_ms1));
    stats.insert("frames_decoded".to_string(), json!(decoded));
    stats.insert("ms_setup_and_decode".to_string(), json!(t_decoded));
    stats.insert("ms_read".to_string(), json!(t_read));
    stats.insert("ms_sum".to_string(), json!(t_sum));
    stats.insert("ms_write".to_string(), json!(t_write));
    stats.insert("raw_events_summed".to_string(), json!(events));
    stats.insert("unmatched_fragment_points".to_string(), json!(unmatched));
    ArtifactReport {
        logical_name: artifact::CHROMATOGRAMS.0.to_string(),
        schema_name: artifact::CHROMATOGRAMS.0.to_string(),
        schema_version: artifact::CHROMATOGRAMS.1,
        stage: "retrace".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({
            "trace_source": "raw_tdf",
            "raw": p.raw,
            "centroid_chromatograms": p.chromatograms,
            "im_half_width": hw,
            "ms1": p.cfg.ms1,
            "ms1_im_half_width": hw1,
            "min_events": p.cfg.min_events,
            "frag_tol_ppm": frag_tol,
            "prec_tol_ppm": p.prec_tol_ppm,
            "config_hash": p.config_hash,
        }),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;
    info!(
        rows,
        fragment_rows = n_frag,
        ms1_rows = n_ms1,
        frames = decoded,
        elapsed_ms = elapsed,
        "retrace: done"
    );
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two slots: scans 0..4 isolate 500 +/- 10, scans 4..8 isolate 600 +/- 10.
    fn frame() -> Frame {
        // scan s holds TOF indices [100, 200, 300] with intensity s + 1 each
        let mut f = Frame::default();
        let mut off = vec![0usize];
        for s in 0..8u32 {
            for t in [100u32, 200, 300] {
                f.tof_indices.push(t);
                f.intensities.push(s + 1);
            }
            off.push(f.tof_indices.len());
        }
        f.scan_offsets = off;
        f.quadrupole_settings = Arc::new(QuadrupoleSettings {
            index: 1,
            scan_starts: vec![0, 4],
            scan_ends: vec![4, 8],
            isolation_mz: vec![500.0, 600.0],
            isolation_width: vec![20.0, 20.0],
            collision_energy: vec![0.0, 0.0],
        });
        f
    }

    #[test]
    fn scan_band_takes_the_scans_inside_the_mobility_interval() {
        let im = [1.4, 1.3, 1.2, 1.1, 1.0, 0.9];
        assert_eq!(scan_band(&im, 1.05, 1.25), (2, 4));
        assert_eq!(scan_band(&im, 1.0, 1.1), (3, 5));
        assert_eq!(scan_band(&im, 2.0, 3.0), (0, 0));
        assert_eq!(scan_band(&im, 0.1, 0.2), (6, 6));
    }

    #[test]
    fn slot_scans_keeps_the_covering_slot_clipped_to_the_band() {
        let f = frame();
        assert_eq!(
            slot_scans(&f.quadrupole_settings, 505.0, (0, 8)),
            vec![(0, 4)]
        );
        assert_eq!(
            slot_scans(&f.quadrupole_settings, 605.0, (2, 6)),
            vec![(4, 6)]
        );
        assert_eq!(slot_scans(&f.quadrupole_settings, 700.0, (0, 8)), vec![]);
        assert_eq!(slot_scans(&f.quadrupole_settings, 505.0, (5, 8)), vec![]);
    }

    #[test]
    fn slot_scans_merges_overlapping_slots() {
        let mut f = frame();
        f.quadrupole_settings = Arc::new(QuadrupoleSettings {
            index: 1,
            scan_starts: vec![0, 2],
            scan_ends: vec![4, 6],
            isolation_mz: vec![500.0, 505.0],
            isolation_width: vec![20.0, 20.0],
            collision_energy: vec![0.0, 0.0],
        });
        assert_eq!(
            slot_scans(&f.quadrupole_settings, 502.0, (0, 8)),
            vec![(0, 6)]
        );
    }

    #[test]
    fn sum_events_sums_the_tof_window_over_the_scans_with_weighted_mobility() {
        let f = TofFrame::new(frame()).unwrap();
        let im: Vec<f64> = (0..8).map(|s| 1.5 - 0.1 * s as f64).collect();
        // scans 1..3, TOF 150..=300 -> TOF 200 and 300 in scans 1 and 2: 2*2 + 2*3 = 10
        let s = sum_events(&f, &[(1, 3)], 150, 300, &im);
        assert_eq!((s.inten, s.n), (10.0, 4));
        let mean = s.imw / s.inten;
        assert!((mean - (4.0 * 1.4 + 6.0 * 1.3) / 10.0).abs() < 1e-12);
        // an empty TOF window and a band past the last scan sum to nothing
        assert_eq!(sum_events(&f, &[(0, 8)], 101, 199, &im), Sum::default());
        assert_eq!(sum_events(&f, &[(9, 12)], 0, 999, &im), Sum::default());
    }
}
