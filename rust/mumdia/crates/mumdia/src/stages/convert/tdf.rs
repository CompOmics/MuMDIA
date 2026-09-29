//! Native timsTOF reading (`.d` holding `analysis.tdf`), with ion mobility kept
//! (docs/TIMS_ROADMAP.md, P1).
//!
//! A diaPASEF MS2 frame holds several isolation windows ("slots") that differ in both
//! m/z and 1/K0: the quadrupole steps through them as the TIMS scans. Each slot becomes
//! its own MS2 spectrum, carrying the slot's m/z bounds and 1/K0 bounds. An MS1 frame
//! becomes one spectrum over all of its scans. Within a spectrum the raw TOF x scan
//! points are centroided in m/z x mobility ([`centroid_2d`]), so every peak carries an
//! intensity-weighted 1/K0.
//!
//! Frames are decoded in parallel and handed to the sink in frame order, so the output
//! does not depend on the thread count.

use anyhow::{anyhow, bail, Context, Result};
use mumdia_core::config::ConvertConfig;
use rayon::prelude::*;
use timsrust::converters::{ConvertableDomain, Scan2ImConverter, Tof2MzConverter};
use timsrust::readers::{FrameReader, MetadataReader};
use timsrust::{AcquisitionType, Frame, MSLevel};
use tracing::{info, warn};

use super::{Decoded, Ms2Row};

/// Frames decoded per parallel batch. Bounds the resident raw frames (an MS1 frame is
/// ~700k points, ~8 MB) while keeping every thread busy.
const FRAME_CHUNK: usize = 512;

/// Centroiding and noise-floor settings (`convert.tdf_*`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TdfParams {
    pub mz_ppm: f64,
    pub im_gap_scans: u32,
    pub min_points: u32,
    /// Also compute each centroid's mobility width (`Centroids::width`).
    pub im_width: bool,
    /// Valley depth that splits a cluster along m/z (0 = off; `valley_cuts`).
    pub mz_valley: f64,
    /// Half-width in ppm of the m/z profile smoothing used by the valley split.
    pub mz_smooth_ppm: f64,
    /// Valley depth that splits a cluster along mobility (0 = off; `valley_cuts`).
    pub im_valley: f64,
    /// Half-width in TIMS scans of the mobility profile smoothing.
    pub im_smooth_scans: f64,
}

impl TdfParams {
    pub fn from_config(c: &ConvertConfig) -> Self {
        Self {
            mz_ppm: c.tdf_mz_ppm,
            im_gap_scans: c.tdf_im_gap_scans,
            min_points: c.tdf_min_points,
            im_width: c.tdf_im_width,
            mz_valley: c.tdf_mz_valley,
            mz_smooth_ppm: c.tdf_mz_smooth_ppm,
            im_valley: c.tdf_im_valley,
            im_smooth_scans: c.tdf_im_smooth_scans,
        }
    }
}

impl Default for TdfParams {
    fn default() -> Self {
        Self::from_config(&ConvertConfig::default())
    }
}

/// Scan index -> 1/K0 (V s cm^-2).
///
/// timsrust only interpolates linearly between the acquisition range bounds, which on the
/// Ultra 2 benchmark run is off by up to 0.030 at the high-mobility end: the width of a
/// calibrated IM window. Bruker's `TimsCalibration` model 2 is used when present
/// (the same form mzdata implements, `io/tdf/calibration.rs`).
enum ImCal {
    Model2 {
        c6: f64,
        c7: f64,
        offset: f64,
        slope: f64,
    },
    Linear(Scan2ImConverter),
}

impl ImCal {
    fn read(path: &str, linear: Scan2ImConverter) -> Result<Self> {
        let db = format!("{path}/analysis.tdf");
        let con =
            rusqlite::Connection::open_with_flags(&db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
                .with_context(|| format!("open {db}"))?;
        let mut st = con.prepare(
            "SELECT ModelType, C0, C1, C2, C3, C4, C6, C7 FROM TimsCalibration ORDER BY Id",
        )?;
        let rows: Vec<(i64, [f64; 7])> = st
            .query_map([], |r| {
                let mut c = [0.0; 7];
                for (k, v) in c.iter_mut().enumerate() {
                    *v = r.get(k + 1)?;
                }
                Ok((r.get(0)?, c))
            })?
            .collect::<rusqlite::Result<_>>()?;
        // ponytail: one calibration per run. Frames can reference different rows (the
        // `Frames.TimsCalibration` column); add per-frame lookup if a file ever has more.
        if rows.len() > 1 {
            warn!(
                rows = rows.len(),
                "convert: several TimsCalibration rows; using the first for every frame"
            );
        }
        match rows.first() {
            Some(&(2, [c0, c1, c2, c3, c4, c6, c7])) => {
                let slope = if c1 == 0.0 { 0.0 } else { (c3 - c2) / c1 };
                Ok(ImCal::Model2 {
                    c6,
                    c7,
                    offset: c2 - slope * (c4 + c0),
                    slope,
                })
            }
            other => {
                warn!(
                    model = ?other.map(|r| r.0),
                    "convert: no TimsCalibration model 2; 1/K0 interpolated linearly \
                     between the acquisition range bounds"
                );
                Ok(ImCal::Linear(linear))
            }
        }
    }

    fn im(&self, scan: f64) -> f64 {
        match self {
            ImCal::Model2 {
                c6,
                c7,
                offset,
                slope,
            } => 1.0 / (c6 + c7 / (offset + slope * scan)),
            ImCal::Linear(l) => l.convert(scan),
        }
    }
}

/// One centroided spectrum: m/z-sorted peaks with a per-peak 1/K0.
#[derive(Debug, Default, PartialEq)]
struct Centroids {
    mz: Vec<f32>,
    inten: Vec<f32>,
    im: Vec<f32>,
    /// Per-peak mobility width in 1/K0 (empty unless `TdfParams::im_width`).
    width: Vec<f32>,
}

/// Centroid raw `(tof_index, scan, intensity)` points in m/z x mobility.
///
/// Points are ordered by TOF (monotone in m/z) and joined into one m/z trace while each
/// neighbour is within `mz_ppm` of the previous one. Each trace is then ordered by scan
/// and split where two consecutive points are more than `im_gap_scans` scans apart. A
/// cluster yields its summed intensity and its intensity-weighted m/z and 1/K0, and is
/// dropped when it holds fewer than `min_points` raw points.
///
/// With `im_width`, a cluster also yields its mobility width: the intensity-weighted SD
/// of its scans, with the 1/12 variance of one scan's quantisation added so a
/// single-scan cluster is not a zero-width point, converted to 1/K0 with the local slope
/// of the calibration at the weighted mean scan.
///
/// With `mz_valley` > 0, each such cluster is further cut at the valleys of its m/z
/// profile, and with `im_valley` > 0 each resulting piece at the valleys of its mobility
/// profile ([`valley_cuts`]). Every piece is a cluster of its own, floor included.
///
/// ponytail: single linkage, so a dense m/z region can chain two peaks together, and two
/// ions of one m/z whose mobility profiles touch without a scan gap merge, unless the
/// valley splits (both default off) cut them apart again.
fn centroid_2d(
    pts: &mut [(u32, u32, u32)],
    mz_of: impl Fn(u32) -> f64,
    im_of: impl Fn(f64) -> f64,
    p: &TdfParams,
) -> Centroids {
    // Total order on the full tuple, so the result does not depend on the input order.
    pts.sort_unstable();
    let emit = |c: &[(u32, u32, u32)], peaks: &mut Vec<(f64, f32, f32, f32)>| {
        if c.len() < p.min_points as usize {
            return;
        }
        let (mut w, mut wm, mut ws, mut wss) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for &(t, s, x) in c {
            let x = x as f64;
            w += x;
            wm += x * mz_of(t);
            ws += x * s as f64;
            wss += x * (s as f64) * (s as f64);
        }
        if w > 0.0 {
            let mean = ws / w;
            let width = if p.im_width {
                let var = (wss / w - mean * mean).max(0.0) + 1.0 / 12.0;
                (var.sqrt() * (im_of(mean + 0.5) - im_of(mean - 0.5)).abs()) as f32
            } else {
                0.0
            };
            peaks.push((wm / w, w as f32, im_of(mean) as f32, width));
        }
    };
    // Mobility valley split of one piece, then emit.
    let im_split = |c: &mut [(u32, u32, u32)], peaks: &mut Vec<(f64, f32, f32, f32)>| {
        if p.im_valley <= 0.0 {
            return emit(c, peaks);
        }
        c.sort_unstable_by_key(|&(t, s, x)| (s, t, x));
        let cuts = valley_cuts(c, |q| q.1, |s| s as f64, |_| p.im_smooth_scans, p.im_valley);
        let mut a = 0;
        for b in cuts.into_iter().chain([c.len()]) {
            emit(&c[a..b], peaks);
            a = b;
        }
    };
    let mut peaks: Vec<(f64, f32, f32, f32)> = Vec::new();
    let mut i = 0;
    while i < pts.len() {
        let mut j = i + 1;
        let mut prev = mz_of(pts[i].0);
        while j < pts.len() {
            let m = mz_of(pts[j].0);
            if m - prev > prev * p.mz_ppm * 1e-6 {
                break;
            }
            prev = m;
            j += 1;
        }
        let trace = &mut pts[i..j];
        trace.sort_unstable_by_key(|&(t, s, x)| (s, t, x));
        let mut k = 0;
        while k < trace.len() {
            let mut e = k + 1;
            while e < trace.len() && trace[e].1 - trace[e - 1].1 <= p.im_gap_scans {
                e += 1;
            }
            let cluster = &mut trace[k..e];
            if p.mz_valley > 0.0 {
                cluster.sort_unstable();
                let cuts = valley_cuts(
                    cluster,
                    |q| q.0,
                    &mz_of,
                    |m| m * p.mz_smooth_ppm * 1e-6,
                    p.mz_valley,
                );
                let mut a = 0;
                for b in cuts.into_iter().chain([cluster.len()]) {
                    im_split(&mut cluster[a..b], &mut peaks);
                    a = b;
                }
            } else {
                im_split(cluster, &mut peaks);
            }
            k = e;
        }
        i = j;
    }
    peaks.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.total_cmp(&b.2)));
    Centroids {
        mz: peaks.iter().map(|p| p.0 as f32).collect(),
        inten: peaks.iter().map(|p| p.1).collect(),
        im: peaks.iter().map(|p| p.2).collect(),
        width: if p.im_width {
            peaks.iter().map(|p| p.3).collect()
        } else {
            Vec::new()
        },
    }
}

/// Cut positions (indices into `c`, sorted by `key`) where one cluster is split at the
/// valleys of its profile along one axis: m/z (`convert.tdf_mz_valley`, key the TOF index)
/// or mobility (`convert.tdf_im_valley`, key the TIMS scan).
///
/// The profile has one entry per distinct key, at position `pos(key)`, with the summed
/// intensity, smoothed with a triangular kernel of half-width `tol(position)`. A local
/// minimum is a cut when it is below `valley` times the smaller of the highest point since
/// the previous cut and the highest point after it; the minimum itself starts the
/// right-hand piece.
///
/// ponytail: the right-hand hump is the maximum of the whole remainder, not the next
/// hump only, so a small shoulder between two large ions may stay joined to one of them.
/// Track the next maximum instead if the centroid-pair histogram shows it matters.
fn valley_cuts(
    c: &[(u32, u32, u32)],
    key: impl Fn(&(u32, u32, u32)) -> u32,
    pos: impl Fn(u32) -> f64,
    tol: impl Fn(f64) -> f64,
    valley: f64,
) -> Vec<usize> {
    // (first index in `c`, position, summed intensity) per distinct key.
    let mut prof: Vec<(usize, f64, f64)> = Vec::new();
    for (i, q) in c.iter().enumerate() {
        match prof.last_mut() {
            Some(l) if key(&c[l.0]) == key(q) => l.2 += q.2 as f64,
            _ => prof.push((i, pos(key(q)), q.2 as f64)),
        }
    }
    let n = prof.len();
    if n < 3 {
        return Vec::new();
    }
    // ponytail: O(n x window) smoothing; clusters span tens to hundreds of TOF indices.
    let sm: Vec<f64> = (0..n)
        .map(|i| {
            let (m, tol) = (prof[i].1, tol(prof[i].1));
            let w = |d: f64| if tol > 0.0 { 1.0 - d / tol } else { 1.0 };
            let mut s = 0.0;
            for q in prof[..i].iter().rev().take_while(|q| m - q.1 <= tol) {
                s += q.2 * w(m - q.1);
            }
            for q in prof[i..].iter().take_while(|q| q.1 - m <= tol) {
                s += q.2 * w(q.1 - m);
            }
            s
        })
        .collect();
    let mut rmax = sm.clone();
    for i in (0..n - 1).rev() {
        rmax[i] = rmax[i].max(rmax[i + 1]);
    }
    let (mut cuts, mut lmax) = (Vec::new(), sm[0]);
    for m in 1..n - 1 {
        if sm[m] < sm[m - 1] && sm[m] <= sm[m + 1] && sm[m] < valley * lmax.min(rmax[m + 1]) {
            cuts.push(prof[m].0);
            lmax = sm[m];
        } else {
            lmax = lmax.max(sm[m]);
        }
    }
    cuts
}

/// Keep the `top` most intense peaks (0 = all), m/z order preserved.
fn cap(c: Centroids, top: usize) -> Centroids {
    if top == 0 || c.mz.len() <= top {
        return c;
    }
    let mut idx: Vec<usize> = (0..c.mz.len()).collect();
    idx.sort_by(|&a, &b| c.inten[b].total_cmp(&c.inten[a]).then(a.cmp(&b)));
    idx.truncate(top);
    idx.sort_unstable();
    Centroids {
        mz: idx.iter().map(|&k| c.mz[k]).collect(),
        inten: idx.iter().map(|&k| c.inten[k]).collect(),
        im: idx.iter().map(|&k| c.im[k]).collect(),
        width: if c.width.is_empty() {
            Vec::new()
        } else {
            idx.iter().map(|&k| c.width[k]).collect()
        },
    }
}

struct Ctx {
    mz: Tof2MzConverter,
    im: ImCal,
    p: TdfParams,
    top_ms1: usize,
    top_ms2: usize,
}

impl Ctx {
    fn spectrum(&self, f: &Frame, scans: std::ops::Range<usize>, top: usize) -> Centroids {
        let n_scans = f.scan_offsets.len().saturating_sub(1);
        let scans = scans.start.min(n_scans)..scans.end.min(n_scans);
        let mut pts = Vec::with_capacity(f.scan_offsets[scans.end] - f.scan_offsets[scans.start]);
        for s in scans {
            for k in f.scan_offsets[s]..f.scan_offsets[s + 1] {
                pts.push((f.tof_indices[k], s as u32, f.intensities[k]));
            }
        }
        // ponytail: raw detector counts; `intensity_correction_factor` (accumulation-time
        // normalisation) is ignored, which is exact while MS2 frames share one ramp time.
        let c = centroid_2d(&mut pts, |t| self.mz.convert(t), |s| self.im.im(s), &self.p);
        cap(c, top)
    }

    fn frame(&self, f: &Frame) -> Vec<Decoded> {
        let rt_s = f.rt_in_seconds;
        if !rt_s.is_finite() {
            return vec![Decoded::BadRt {
                id: format!("frame={}", f.index),
                rt_s,
            }];
        }
        match f.ms_level {
            MSLevel::MS1 => {
                let c = self.spectrum(f, 0..usize::MAX, self.top_ms1);
                vec![Decoded::Ms1 {
                    rt_s,
                    mz: c.mz,
                    inten: c.inten,
                    im: Some(c.im),
                    im_width: self.p.im_width.then_some(c.width),
                    nonfinite_peaks: 0,
                }]
            }
            MSLevel::MS2 => {
                let q = &f.quadrupole_settings;
                (0..q.len())
                    .map(|k| {
                        let (s0, s1) = (q.scan_starts[k], q.scan_ends[k]);
                        let c = self.spectrum(f, s0..s1, self.top_ms2);
                        let (a, b) = (self.im.im(s0 as f64), self.im.im(s1 as f64));
                        let half = q.isolation_width[k] / 2.0;
                        let target = q.isolation_mz[k];
                        Decoded::Ms2(Box::new(Ms2Row {
                            rt_s,
                            id: format!("frame={} slot={k}", f.index),
                            mz: c.mz,
                            inten: c.inten,
                            nonfinite_peaks: 0,
                            wt: target,
                            wl: target - half,
                            wu: target + half,
                            pmz: Some(target),
                            pz: None,
                            im: Some(c.im),
                            im_width: self.p.im_width.then_some(c.width),
                            im_lo: Some(a.min(b) as f32),
                            im_hi: Some(a.max(b) as f32),
                        }))
                    })
                    .collect()
            }
            MSLevel::Unknown => vec![Decoded::Other],
        }
    }
}

/// Decode `path` frame by frame and hand every spectrum to `sink` with its scan index
/// (the emission ordinal). `max_frames > 0` reads only the first frames of the run.
/// Returns the number of spectra emitted.
pub(super) fn drive(
    path: &str,
    max_frames: usize,
    p: &TdfParams,
    top_ms1: usize,
    top_ms2: usize,
    mut sink: impl FnMut(u32, Decoded) -> Result<()>,
) -> Result<usize> {
    let meta = MetadataReader::new(path).map_err(|e| anyhow!("{path}: {e}"))?;
    let reader = FrameReader::new(path).map_err(|e| anyhow!("{path}: {e}"))?;
    match reader.get_acquisition() {
        AcquisitionType::DIAPASEF => {}
        other => bail!(
            "{path}: acquisition type {other:?}. The native timsTOF reader handles \
             diaPASEF only; set convert.bruker_reader = \"msconvert\" for this file"
        ),
    }
    let ctx = Ctx {
        mz: meta.mz_converter,
        im: ImCal::read(path, meta.im_converter)?,
        p: *p,
        top_ms1,
        top_ms2,
    };
    let n = match max_frames {
        0 => reader.len(),
        m => m.min(reader.len()),
    };
    info!(frames = n, params = ?p, "convert: decoding timsTOF frames");
    let mut emitted = 0usize;
    for start in (0..n).step_by(FRAME_CHUNK) {
        let batch: Vec<Result<Vec<Decoded>>> = (start..(start + FRAME_CHUNK).min(n))
            .into_par_iter()
            .map(|i| {
                let f = reader
                    .get(i)
                    .map_err(|e| anyhow!("{path}: frame {i}: {e}"))?;
                Ok(ctx.frame(&f))
            })
            .collect();
        for spectra in batch {
            for d in spectra? {
                sink(emitted as u32, d)?;
                emitted += 1;
            }
        }
    }
    Ok(emitted)
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: TdfParams = TdfParams {
        mz_ppm: 10.0,
        im_gap_scans: 5,
        min_points: 2,
        im_width: false,
        mz_valley: 0.0,
        mz_smooth_ppm: 4.0,
        im_valley: 0.0,
        im_smooth_scans: 4.0,
    };
    const V: TdfParams = TdfParams {
        mz_valley: 0.5,
        ..P
    };

    /// TOF index t sits at m/z 500 + t/1000 (2 ppm per index near 500), scan s at 1/K0 s.
    fn run(pts: &[(u32, u32, u32)], p: &TdfParams) -> Centroids {
        let mut v = pts.to_vec();
        centroid_2d(&mut v, |t| 500.0 + t as f64 / 1000.0, |s| s, p)
    }

    #[test]
    fn neighbouring_points_merge_into_one_weighted_peak() {
        let c = run(&[(1000, 10, 1), (1001, 11, 3)], &P);
        assert_eq!(c.inten, vec![4.0]);
        assert!((c.mz[0] as f64 - (501.0 * 1.0 + 501.001 * 3.0) / 4.0).abs() < 1e-4);
        assert!((c.im[0] - 10.75).abs() < 1e-6);
    }

    #[test]
    fn one_mz_at_distant_scans_is_two_peaks() {
        let c = run(
            &[(1000, 10, 1), (1000, 11, 1), (1000, 50, 2), (1000, 52, 2)],
            &P,
        );
        assert_eq!(c.inten, vec![2.0, 4.0]);
        assert_eq!(c.im, vec![10.5, 51.0]);
    }

    #[test]
    fn distant_mz_is_two_peaks_and_a_singleton_is_below_the_floor() {
        // 1000 -> 1100 is ~200 ppm apart; the lone point at 5000 has one raw point.
        let c = run(
            &[
                (1000, 10, 1),
                (1000, 11, 1),
                (1100, 10, 1),
                (1101, 10, 1),
                (5000, 10, 9),
            ],
            &P,
        );
        assert_eq!(c.inten, vec![2.0, 2.0]);
        let all = run(&[(5000, 10, 9)], &TdfParams { min_points: 1, ..P });
        assert_eq!(all.inten, vec![9.0]);
    }

    #[test]
    fn the_result_does_not_depend_on_input_order() {
        let pts = [
            (1000, 10, 1),
            (1003, 12, 5),
            (1000, 50, 2),
            (1001, 51, 7),
            (2000, 3, 4),
            (2001, 3, 1),
        ];
        let mut rev = pts;
        rev.reverse();
        assert_eq!(run(&pts, &P), run(&rev, &P));
    }

    /// One scan, consecutive TOF indices from 1000 (2 ppm apart), given intensities.
    fn profile(x: &[u32]) -> Vec<(u32, u32, u32)> {
        x.iter()
            .enumerate()
            .map(|(i, &x)| (1000 + i as u32, 10, x))
            .collect()
    }

    #[test]
    fn two_humps_with_a_deep_valley_split_and_off_keeps_one_peak() {
        // Humps at 1002 and 1007 (10 ppm apart), one chain under 10 ppm linkage.
        let pts = profile(&[1, 4, 9, 4, 1, 1, 4, 9, 4, 1]);
        assert_eq!(run(&pts, &P).inten, vec![38.0]);
        // Smoothed (4 ppm = 2 indices, triangular): the minimum 3.5 at 1004 is below
        // 0.5 x 13 and starts the right-hand piece.
        let c = run(&pts, &V);
        assert_eq!(c.inten, vec![18.0, 20.0]);
        assert!(c.mz[0] < c.mz[1]);
    }

    #[test]
    fn a_lumpy_single_ion_stays_one_peak_after_smoothing() {
        let pts = profile(&[2, 6, 3, 6, 2]);
        let raw = TdfParams {
            mz_valley: 0.6,
            mz_smooth_ppm: 0.0,
            ..P
        };
        // Unsmoothed, the dip 3 is below 0.6 x 6 and cuts; smoothed there is no minimum.
        assert_eq!(run(&pts, &raw).inten, vec![8.0, 11.0]);
        assert_eq!(
            run(
                &pts,
                &TdfParams {
                    mz_valley: 0.6,
                    ..P
                }
            )
            .inten,
            vec![19.0]
        );
        // A shallow dip does not cut even unsmoothed.
        assert_eq!(run(&profile(&[2, 6, 5, 6, 2]), &raw).inten, vec![21.0]);
    }

    #[test]
    fn valley_pieces_meet_the_floor_and_do_not_depend_on_input_order() {
        // The dip 1 is below 0.5 x min(4, 8) and cuts; the left piece is the single point
        // 4 and falls below the floor of 2 points.
        let floor = TdfParams {
            mz_smooth_ppm: 0.0,
            ..V
        };
        assert_eq!(run(&profile(&[4, 1, 8, 3]), &floor).inten, vec![12.0]);
        let mut pts = profile(&[1, 4, 9, 4, 1, 1, 4, 9, 4, 1]);
        pts.extend([(1003, 11, 2), (1007, 12, 3), (1000, 50, 2), (1001, 51, 7)]);
        let mut rev = pts.clone();
        rev.reverse();
        assert_eq!(run(&pts, &V), run(&rev, &V));
    }

    #[test]
    fn two_mobility_humps_split_and_off_keeps_one_peak() {
        // One TOF index, scans 10..=19 (one scan apart, no gap), humps at 12 and 17.
        let x = [1, 4, 9, 4, 1, 1, 4, 9, 4, 1];
        let pts: Vec<_> = x
            .iter()
            .enumerate()
            .map(|(i, &x)| (1000, 10 + i as u32, x))
            .collect();
        assert_eq!(run(&pts, &P).inten, vec![38.0]);
        // Smoothing 1 scan: no weight on the neighbours, the raw dip 1 at scan 14 cuts.
        let on = TdfParams {
            im_valley: 0.5,
            im_smooth_scans: 1.0,
            ..P
        };
        let c = run(&pts, &on);
        assert_eq!(c.inten, vec![18.0, 20.0]);
        assert!((c.im[0] - 214.0 / 18.0).abs() < 1e-4 && c.im[1] > 16.0);
        // A wide smoothing fills the valley: one peak.
        assert_eq!(
            run(
                &pts,
                &TdfParams {
                    im_smooth_scans: 6.0,
                    ..on
                }
            )
            .inten,
            vec![38.0]
        );
        // Both splits on, input order irrelevant.
        let both = TdfParams {
            mz_valley: 0.5,
            ..on
        };
        let mut rev = pts.clone();
        rev.reverse();
        assert_eq!(run(&pts, &both), run(&rev, &both));
    }

    #[test]
    fn cap_keeps_the_most_intense_in_mz_order() {
        let c = Centroids {
            mz: vec![1.0, 2.0, 3.0],
            inten: vec![5.0, 1.0, 9.0],
            im: vec![0.7, 0.8, 0.9],
            width: vec![0.01, 0.02, 0.03],
        };
        let k = cap(c, 2);
        assert_eq!(k.mz, vec![1.0, 3.0]);
        assert_eq!(k.im, vec![0.7, 0.9]);
        assert_eq!(k.width, vec![0.01, 0.03]);
    }

    #[test]
    fn width_is_the_weighted_scan_sd_plus_quantisation_in_im_units() {
        let on = TdfParams {
            im_width: true,
            ..P
        };
        // Equal weight at scans 10 and 12: SD 1 scan; one scan is 1/K0 0.5 here.
        let mut v = vec![(1000, 10, 2), (1000, 12, 2)];
        let c = centroid_2d(&mut v, |t| 500.0 + t as f64 / 1000.0, |s| 0.5 * s, &on);
        assert!((c.width[0] as f64 - 0.5 * (1.0f64 + 1.0 / 12.0).sqrt()).abs() < 1e-6);
        // A single-scan cluster keeps the quantisation term only.
        let mut v = vec![(1000, 10, 1), (1001, 10, 3)];
        let c = centroid_2d(&mut v, |t| 500.0 + t as f64 / 1000.0, |s| 0.5 * s, &on);
        assert!((c.width[0] as f64 - 0.5 / 12f64.sqrt()).abs() < 1e-6);
        // Off: no widths, and the other columns are those of the on run.
        let mut v = vec![(1000, 10, 2), (1000, 12, 2)];
        let off = centroid_2d(&mut v, |t| 500.0 + t as f64 / 1000.0, |s| 0.5 * s, &P);
        assert!(off.width.is_empty());
        assert_eq!(off.im, vec![5.5]);
    }

    /// Real-data check: `MUMDIA_BENCH_TDF=/path/run.d cargo test -- --ignored tdf_real`.
    #[test]
    #[ignore]
    fn tdf_real_diapasef_frames_split_into_slots_with_mobility_inside_them() {
        let Ok(path) = std::env::var("MUMDIA_BENCH_TDF") else {
            return;
        };
        let (mut ms1, mut ms2) = (0, 0);
        drive(&path, 60, &TdfParams::default(), 0, 0, |_, d| {
            match d {
                Decoded::Ms1 { mz, im, .. } => {
                    ms1 += 1;
                    assert_eq!(im.unwrap().len(), mz.len());
                }
                Decoded::Ms2(r) => {
                    ms2 += 1;
                    let (lo, hi) = (r.im_lo.unwrap(), r.im_hi.unwrap());
                    let im = r.im.unwrap();
                    assert_eq!(im.len(), r.mz.len());
                    assert!(im.iter().all(|&v| v >= lo - 1e-3 && v <= hi + 1e-3));
                    assert!(r.mz.windows(2).all(|w| w[0] <= w[1]));
                }
                _ => {}
            }
            Ok(())
        })
        .unwrap();
        assert!(ms1 > 0 && ms2 > 0);
        eprintln!("first 60 frames: {ms1} MS1, {ms2} MS2 slot spectra");
    }
}
