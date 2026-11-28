//! Native timsTOF diaPASEF reading (a `.d` holding `analysis.tdf`), with ion mobility kept
//! (docs/35_diapasef.md, "Conversion").
//!
//! A diaPASEF MS2 frame holds several isolation windows ("slots") that differ in both m/z
//! and 1/K0: the quadrupole steps through them as the TIMS scans. Each slot becomes its own
//! MS2 spectrum, carrying the slot's m/z bounds and 1/K0 bounds. An MS1 frame becomes one
//! spectrum over all of its scans. Within a spectrum the raw TOF x scan points are
//! centroided in m/z x mobility ([`centroid_2d`]), so every peak carries an
//! intensity-weighted 1/K0.
//!
//! Only files that `raw::detect` classifies as `Vendor::BrukerTdf` reach this reader:
//! diaPASEF frames (`Frames.MsMsType` 9, none 8) in the compression timsrust decodes.
//! Every other Bruker `.d` is converted by msconvert.
//!
//! Frames are decoded in parallel and handed to the sink in frame order, so the output does
//! not depend on the thread count.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, bail, Context, Result};
use mumdia_core::config::ConvertConfig;
use rayon::prelude::*;
use timsrust::converters::{ConvertableDomain, Scan2ImConverter, Tof2MzConverter};
use timsrust::readers::{FrameReader, FrameReaderError, MetadataReader};
use timsrust::{AcquisitionType, Frame, MSLevel};
use tracing::{info, warn};

use super::ms1_denoise::Ms1Denoise;
use super::{top_n_by_intensity, Decoded, Ms2Row};

/// Frames decoded per parallel batch. Bounds the resident raw frames (an MS1 frame is
/// ~700k points, ~8 MB) while keeping every thread busy.
const FRAME_CHUNK: usize = 512;

/// Centroiding and noise-floor settings (`convert.tdf_*`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TdfParams {
    pub mz_ppm: f64,
    pub im_gap_scans: u32,
    pub min_points: u32,
    /// Valley depth that splits a cluster along m/z (0 = off; `valley_cuts`).
    pub mz_valley: f64,
    /// Half-width in ppm of the m/z profile smoothing used by the valley split.
    pub mz_smooth_ppm: f64,
    /// Valley depth that splits a cluster along mobility (0 = off; `valley_cuts`).
    pub im_valley: f64,
    /// Half-width in TIMS scans of the mobility profile smoothing.
    pub im_smooth_scans: f64,
    /// Remove MS1 noise before centroiding (`convert.tdf_ms1_denoise`, `ms1_denoise::Ms1Denoise`).
    pub ms1_denoise: bool,
}

impl TdfParams {
    pub fn from_config(c: &ConvertConfig) -> Self {
        Self {
            mz_ppm: c.tdf_mz_ppm,
            im_gap_scans: c.tdf_im_gap_scans,
            min_points: c.tdf_min_points,
            mz_valley: c.tdf_mz_valley,
            mz_smooth_ppm: c.tdf_mz_smooth_ppm,
            im_valley: c.tdf_im_valley,
            im_smooth_scans: c.tdf_im_smooth_scans,
            ms1_denoise: c.tdf_ms1_denoise,
        }
    }

    /// The settings as the `convert.tdf_*` keys they come from, for the spectra reports.
    pub fn report_params(&self) -> serde_json::Value {
        let mut v = serde_json::json!({
            "tdf_mz_ppm": self.mz_ppm,
            "tdf_im_gap_scans": self.im_gap_scans,
            "tdf_min_points": self.min_points,
            "tdf_mz_valley": self.mz_valley,
            "tdf_mz_smooth_ppm": self.mz_smooth_ppm,
            "tdf_im_valley": self.im_valley,
            "tdf_im_smooth_scans": self.im_smooth_scans,
        });
        // Only when on, so a report at the default is what it was.
        if self.ms1_denoise {
            if let Some(o) = v.as_object_mut() {
                o.insert("tdf_ms1_denoise".into(), serde_json::json!(true));
            }
        }
        v
    }
}

impl Default for TdfParams {
    fn default() -> Self {
        Self::from_config(&ConvertConfig::default())
    }
}

/// TIMS scan index -> 1/K0 (V s cm^-2), as `convert` writes it.
///
/// timsrust only interpolates linearly between the acquisition range bounds, which on the
/// Ultra 2 benchmark run is off by up to 0.030 at the high-mobility end: the width of a
/// calibrated 1/K0 window. Bruker's `TimsCalibration` model 2 is used when present (the
/// same form mzdata implements, `io/tdf/calibration.rs`). Not to be confused with the
/// anchor-fitted 1/K0 calibration of `rt_im_train`, which maps predicted onto observed
/// 1/K0; this one maps the instrument's scan axis onto 1/K0.
enum ScanImCalibration {
    Model2 {
        c6: f64,
        c7: f64,
        offset: f64,
        slope: f64,
    },
    Linear(Scan2ImConverter),
}

/// One `TimsCalibration` row: `ModelType` and `C0, C1, C2, C3, C4, C6, C7`.
type CalibrationRow = (i64, [f64; 7]);

impl ScanImCalibration {
    fn read(path: &str, linear: Scan2ImConverter) -> Result<Self> {
        let db = format!("{path}/analysis.tdf");
        let con =
            rusqlite::Connection::open_with_flags(&db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
                .with_context(|| format!("open {db}"))?;
        let mut st = con.prepare(
            "SELECT ModelType, C0, C1, C2, C3, C4, C6, C7 FROM TimsCalibration ORDER BY Id",
        )?;
        let rows: Vec<CalibrationRow> = st
            .query_map([], |r| {
                let mut c = [0.0; 7];
                for (k, v) in c.iter_mut().enumerate() {
                    *v = r.get(k + 1)?;
                }
                Ok((r.get(0)?, c))
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(Self::from_rows(&rows, linear))
    }

    /// The calibration of the first row, or the linear fallback when that row is not a
    /// model 2 (or there is none).
    ///
    /// One calibration per run. Frames can reference different rows (the
    /// `Frames.TimsCalibration` column); a per-frame lookup is needed if a file ever has
    /// more than one, and until then the first is used for every frame, with a warning.
    fn from_rows(rows: &[CalibrationRow], linear: Scan2ImConverter) -> Self {
        if rows.len() > 1 {
            warn!(
                rows = rows.len(),
                "convert: several TimsCalibration rows; using the first for every frame"
            );
        }
        match rows.first() {
            Some(&(2, [c0, c1, c2, c3, c4, c6, c7])) => {
                let slope = if c1 == 0.0 { 0.0 } else { (c3 - c2) / c1 };
                ScanImCalibration::Model2 {
                    c6,
                    c7,
                    offset: c2 - slope * (c4 + c0),
                    slope,
                }
            }
            other => {
                warn!(
                    model = ?other.map(|r| r.0),
                    "convert: no TimsCalibration model 2; 1/K0 interpolated linearly \
                     between the acquisition range bounds"
                );
                ScanImCalibration::Linear(linear)
            }
        }
    }

    fn im(&self, scan: f64) -> f64 {
        match self {
            ScanImCalibration::Model2 {
                c6,
                c7,
                offset,
                slope,
            } => 1.0 / (c6 + c7 / (offset + slope * scan)),
            ScanImCalibration::Linear(l) => l.convert(scan),
        }
    }

    /// Which calibration this is, for a report.
    fn name(&self) -> &'static str {
        match self {
            ScanImCalibration::Model2 { .. } => "TimsCalibration model 2",
            ScanImCalibration::Linear(_) => "linear between the acquisition range bounds",
        }
    }
}

/// TOF index -> m/z, as `convert` writes it.
///
/// timsrust interpolates linearly in sqrt(m/z) between the acquisition range bounds
/// (`Tof2MzConverter::from_boundaries`). That is exact only when the instrument's own
/// calibration is a pure square law. Bruker's `MzCalibration` rows carry a quadratic (and
/// cubic) TOF term and a constant m/z offset; where they are not zero the linear form is off
/// by an m/z-dependent amount. Measured on a 2020 timsTOF Pro single-cell run
/// (`hela_singlecell_timstof_diapasef`, `C2` = -0.005): -98 ppm at m/z 300, -43 ppm at 700,
/// -21 ppm at 1,000, 0 at 1,600, against the fragment m/z of DIA-NN's identifications in the
/// same run; at such an error no seed PSM passes, and the run returned nothing.
///
/// The calibrated form (the model mzdata implements, `io/tdf/calibration.rs`, Apache-2.0):
/// with `tof = index * DigitizerTimebase + DigitizerDelay`, solve
/// `tof = C0 + beta s + c2 s^2 + C3 s^3` for `s`, and `m/z = s^2 - C4`, where
/// `beta = sqrt(1e12 / (C1 cf))`, `c2 = C2 / cf` and the temperature factor
/// `cf = 1 + (dC1 (T1_cal - T1) + dC2 (T2_cal - T2)) / 1e6` uses the median `T1`, `T2` of the
/// run's frames. One calibration per run: the `MzCalibration` row most frames reference.
#[derive(Clone)]
enum TofMzCalibration {
    Model {
        model: MzModel,
        /// m/z of every integer TOF index below the digitizer's sample count, so that the
        /// per-point conversion of centroiding is a lookup.
        table: std::sync::Arc<Vec<f64>>,
    },
    Linear(Tof2MzConverter),
}

/// The solved form of one `MzCalibration` row (temperature factor applied).
#[derive(Clone, Copy, Debug, PartialEq)]
struct MzModel {
    timebase: f64,
    delay: f64,
    c0: f64,
    beta: f64,
    c2: f64,
    c3: f64,
    c4: f64,
}

impl MzModel {
    /// m/z at a fractional TOF index. Newton on the cubic, from the square-law solution; the
    /// quadratic and cubic terms are small, so it converges in a few steps.
    fn mz(&self, tof_index: f64) -> f64 {
        let tof = tof_index.mul_add(self.timebase, self.delay);
        let mut s = (tof - self.c0) / self.beta;
        if self.c2 != 0.0 || self.c3 != 0.0 {
            for _ in 0..12 {
                let f = self.c0 + self.beta * s + self.c2 * s * s + self.c3 * s * s * s - tof;
                let d = self.beta + 2.0 * self.c2 * s + 3.0 * self.c3 * s * s;
                if d == 0.0 {
                    break;
                }
                let step = f / d;
                s -= step;
                if step.abs() < 1e-13 * s.abs().max(1.0) {
                    break;
                }
            }
        }
        s * s - self.c4
    }

    /// The fractional TOF index of an m/z (closed form).
    fn tof(&self, mz: f64) -> f64 {
        let s = (mz + self.c4).max(0.0).sqrt();
        (self.c0 + self.beta * s + self.c2 * s * s + self.c3 * s * s * s - self.delay)
            / self.timebase
    }
}

/// `ModelType, DigitizerTimebase, DigitizerDelay, T1, T2, dC1, dC2, C0, C1, C2, C3, C4` of
/// one `MzCalibration` row.
type MzCalibrationRow = (i64, [f64; 11]);

impl TofMzCalibration {
    /// The model of a row at the run's frame temperatures `t1`, `t2`, or the linear form
    /// when there is no usable row (model type 1 or 2, positive `C1` and time base).
    fn from_parts(
        row: Option<MzCalibrationRow>,
        t1: f64,
        t2: f64,
        n_samples: usize,
        linear: Tof2MzConverter,
    ) -> Self {
        match row {
            Some((kind, [timebase, delay, t1_cal, t2_cal, dc1, dc2, c0, c1, c2, c3, c4]))
                if (kind == 1 || kind == 2) && c1 > 0.0 && timebase > 0.0 =>
            {
                let cf = 1.0 + (dc1 * (t1_cal - t1) + dc2 * (t2_cal - t2)) / 1.0e6;
                let model = MzModel {
                    timebase,
                    delay,
                    c0,
                    beta: (1.0e12 / (c1 * cf)).sqrt(),
                    c2: c2 / cf,
                    c3,
                    c4,
                };
                let table = (0..n_samples).map(|i| model.mz(i as f64)).collect();
                TofMzCalibration::Model {
                    model,
                    table: std::sync::Arc::new(table),
                }
            }
            other => {
                warn!(
                    model = ?other.map(|r| r.0),
                    "convert: no usable MzCalibration row; m/z interpolated in sqrt(m/z) \
                     between the acquisition range bounds"
                );
                TofMzCalibration::Linear(linear)
            }
        }
    }

    /// The row most frames reference, the median frame temperatures and the digitizer's
    /// sample count. A file without the table or columns gives no row (the linear form).
    fn read_row(path: &str) -> Result<(Option<MzCalibrationRow>, f64, f64, usize)> {
        let db = format!("{path}/analysis.tdf");
        let con =
            rusqlite::Connection::open_with_flags(&db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
                .with_context(|| format!("open {db}"))?;
        let used: Vec<(i64, i64)> = con
            .prepare(
                "SELECT MzCalibration, COUNT(*) FROM Frames GROUP BY MzCalibration \
                 ORDER BY 2 DESC, 1",
            )?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        if used.len() > 1 {
            warn!(
                rows = used.len(),
                "convert: frames reference several MzCalibration rows; using the most \
                 referenced for every frame"
            );
        }
        let Some(&(id, _)) = used.first() else {
            return Ok((None, 0.0, 0.0, 0));
        };
        let row: Option<MzCalibrationRow> = con
            .prepare(
                "SELECT ModelType, DigitizerTimebase, DigitizerDelay, T1, T2, dC1, dC2, \
                 C0, C1, C2, C3, C4 FROM MzCalibration WHERE Id = ?1",
            )?
            .query_map([id], |r| {
                let mut c = [0.0; 11];
                for (k, v) in c.iter_mut().enumerate() {
                    *v = r.get::<_, Option<f64>>(k + 1)?.unwrap_or(f64::NAN);
                }
                Ok((r.get(0)?, c))
            })?
            .next()
            .transpose()?
            .filter(|(_, c)| c.iter().all(|v| v.is_finite()));
        let median = |col: &str| -> Result<f64> {
            let mut v: Vec<f64> = con
                .prepare(&format!(
                    "SELECT {col} FROM Frames WHERE MzCalibration = ?1 AND {col} IS NOT NULL"
                ))?
                .query_map([id], |r| r.get::<_, f64>(0))?
                .collect::<rusqlite::Result<_>>()?;
            v.sort_by(|a, b| a.total_cmp(b));
            Ok(v.get(v.len() / 2).copied().unwrap_or(f64::NAN))
        };
        let (t1, t2) = (median("T1")?, median("T2")?);
        let n_samples: usize = con
            .query_row(
                "SELECT Value FROM GlobalMetadata WHERE Key = 'DigitizerNumSamples'",
                [],
                |r| r.get::<_, String>(0),
            )
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0);
        // A missing frame temperature leaves the factor at the calibration's own (cf = 1).
        let row = row.map(|(m, mut c)| {
            if !t1.is_finite() {
                c[4] = 0.0;
            }
            if !t2.is_finite() {
                c[5] = 0.0;
            }
            (m, c)
        });
        let t = |x: f64| if x.is_finite() { x } else { 0.0 };
        Ok((row, t(t1), t(t2), n_samples))
    }

    fn read(path: &str, linear: Tof2MzConverter) -> Self {
        match Self::read_row(path) {
            Ok((row, t1, t2, n)) => Self::from_parts(row, t1, t2, n, linear),
            Err(e) => {
                warn!(error = %e, "convert: MzCalibration not readable");
                Self::from_parts(None, 0.0, 0.0, 0, linear)
            }
        }
    }

    /// m/z of a fractional TOF index.
    fn mz(&self, tof: f64) -> f64 {
        match self {
            TofMzCalibration::Model { model, table } => {
                if tof >= 0.0 && tof.fract() == 0.0 && (tof as usize) < table.len() {
                    table[tof as usize]
                } else {
                    model.mz(tof)
                }
            }
            TofMzCalibration::Linear(l) => l.convert(tof),
        }
    }

    /// m/z of an integer TOF index (centroiding).
    fn mz_u32(&self, tof: u32) -> f64 {
        match self {
            TofMzCalibration::Model { table, .. } if (tof as usize) < table.len() => {
                table[tof as usize]
            }
            _ => self.mz(tof as f64),
        }
    }

    /// The fractional TOF index of an m/z.
    fn tof(&self, mz: f64) -> f64 {
        match self {
            TofMzCalibration::Model { model, .. } => model.tof(mz),
            TofMzCalibration::Linear(l) => l.invert(mz),
        }
    }

    /// Which calibration this is, for a report.
    fn name(&self) -> &'static str {
        match self {
            TofMzCalibration::Model { .. } => "MzCalibration model",
            TofMzCalibration::Linear(_) => "sqrt(m/z) linear between the acquisition range bounds",
        }
    }
}

/// One centroided spectrum: m/z-sorted peaks with a per-peak 1/K0.
#[derive(Debug, Default, PartialEq)]
struct Centroids {
    mz: Vec<f32>,
    inten: Vec<f32>,
    im: Vec<f32>,
}

/// Centroid raw `(tof_index, scan, intensity)` points in m/z x mobility.
///
/// Points are ordered by TOF (monotone in m/z) and joined into one m/z trace while each
/// neighbour is within `mz_ppm` of the previous one. Each trace is then ordered by scan
/// and split where two consecutive points are more than `im_gap_scans` scans apart. A
/// cluster yields its summed intensity and its intensity-weighted m/z and 1/K0, and is
/// dropped when it holds fewer than `min_points` raw points.
///
/// With `mz_valley` > 0, each such cluster is further cut at the valleys of its m/z
/// profile, and with `im_valley` > 0 each resulting piece at the valleys of its mobility
/// profile ([`valley_cuts`]). Every piece is a cluster of its own, floor included.
///
/// Single linkage, so a dense m/z region can chain two peaks together, and two ions of one
/// m/z whose mobility profiles touch without a scan gap merge, unless the valley splits
/// (both off by default) cut them apart again.
fn centroid_2d(
    pts: &mut [(u32, u32, u32)],
    mz_of: impl Fn(u32) -> f64,
    im_of: impl Fn(f64) -> f64,
    p: &TdfParams,
) -> Centroids {
    // Total order on the full tuple, so the result does not depend on the input order.
    pts.sort_unstable();
    let emit = |c: &[(u32, u32, u32)], peaks: &mut Vec<(f64, f32, f32)>| {
        if c.len() < p.min_points as usize {
            return;
        }
        let (mut w, mut wm, mut ws) = (0.0f64, 0.0f64, 0.0f64);
        for &(t, s, x) in c {
            let x = x as f64;
            w += x;
            wm += x * mz_of(t);
            ws += x * s as f64;
        }
        if w > 0.0 {
            peaks.push((wm / w, w as f32, im_of(ws / w) as f32));
        }
    };
    // Mobility valley split of one piece, then emit.
    let im_split = |c: &mut [(u32, u32, u32)], peaks: &mut Vec<(f64, f32, f32)>| {
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
    let mut peaks: Vec<(f64, f32, f32)> = Vec::new();
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
    }
}

/// Cut positions (indices into `c`, sorted by `key`) where one cluster is split at the
/// valleys of its profile along one axis: m/z (`convert.tdf_mz_valley`, key the TOF index)
/// or mobility (`convert.tdf_im_valley`, key the TIMS scan). One function serves both axes.
///
/// The profile has one entry per distinct key, at position `pos(key)`, with the summed
/// intensity, smoothed with a triangular kernel of half-width `tol(position)`. A local
/// minimum is a cut when it is below `valley` times the smaller of the highest point since
/// the previous cut and the highest point after it; the minimum itself starts the
/// right-hand piece.
///
/// The right-hand hump is the maximum of the whole remainder, not the next hump only, so a
/// small shoulder between two large ions may stay joined to one of them.
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
    // O(n x window) smoothing; clusters span tens to hundreds of TOF indices.
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

/// Keep the `top` most intense peaks (0 = all), in m/z order.
///
/// The selection is the one the mzML path uses ([`top_n_by_intensity`]: intensity
/// descending, the lower index on ties). The kept peaks then return to index order, which
/// is m/z order here because [`centroid_2d`] emits them sorted.
fn cap(c: Centroids, top: usize) -> Centroids {
    if top == 0 || c.mz.len() <= top {
        return c;
    }
    let mut idx = top_n_by_intensity(&c.inten, top);
    idx.sort_unstable();
    Centroids {
        mz: idx.iter().map(|&k| c.mz[k]).collect(),
        inten: idx.iter().map(|&k| c.inten[k]).collect(),
        im: idx.iter().map(|&k| c.im[k]).collect(),
    }
}

struct Ctx {
    mz: TofMzCalibration,
    im: ScanImCalibration,
    p: TdfParams,
    top_ms1: usize,
    top_ms2: usize,
    /// The run's MS1 noise filter under `convert.tdf_ms1_denoise`, and the MS1 points it
    /// was given and kept.
    ms1: Option<Ms1Denoise>,
    ms1_points: AtomicU64,
    ms1_kept: AtomicU64,
}

/// The raw `(tof_index, scan, intensity)` points of a frame's scans.
fn scan_points(f: &Frame, scans: std::ops::Range<usize>) -> Vec<(u32, u32, u32)> {
    let n_scans = f.scan_offsets.len().saturating_sub(1);
    let scans = scans.start.min(n_scans)..scans.end.min(n_scans);
    let mut pts = Vec::with_capacity(f.scan_offsets[scans.end] - f.scan_offsets[scans.start]);
    for s in scans {
        for k in f.scan_offsets[s]..f.scan_offsets[s + 1] {
            pts.push((f.tof_indices[k], s as u32, f.intensities[k]));
        }
    }
    pts
}

impl Ctx {
    fn spectrum(
        &self,
        f: &Frame,
        scans: std::ops::Range<usize>,
        top: usize,
        denoise: Option<&Ms1Denoise>,
    ) -> Centroids {
        let n_scans = f.scan_offsets.len().saturating_sub(1);
        let mut pts = scan_points(f, scans);
        if let Some(d) = denoise {
            self.ms1_points
                .fetch_add(pts.len() as u64, Ordering::Relaxed);
            d.apply(&mut pts, n_scans);
            self.ms1_kept.fetch_add(pts.len() as u64, Ordering::Relaxed);
        }
        // Raw detector counts: `intensity_correction_factor` (accumulation-time
        // normalisation) is ignored, which is exact while frames share one ramp time.
        let c = centroid_2d(&mut pts, |t| self.mz.mz_u32(t), |s| self.im.im(s), &self.p);
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
                let c = self.spectrum(f, 0..usize::MAX, self.top_ms1, self.ms1.as_ref());
                vec![Decoded::Ms1 {
                    rt_s,
                    mz: c.mz,
                    inten: c.inten,
                    im: Some(c.im),
                    nonfinite_peaks: 0,
                }]
            }
            MSLevel::MS2 => {
                let q = &f.quadrupole_settings;
                (0..q.len())
                    .map(|k| {
                        let (s0, s1) = (q.scan_starts[k], q.scan_ends[k]);
                        let c = self.spectrum(f, s0..s1, self.top_ms2, None);
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
                            im_lo: Some(a.min(b) as f32),
                            im_hi: Some(a.max(b) as f32),
                            target_only: None,
                            widened: false,
                        }))
                    })
                    .collect()
            }
            MSLevel::Unknown => vec![Decoded::Other],
        }
    }
}

/// A diaPASEF `.d` opened for decoding: the frame reader with the TOF -> m/z and
/// scan -> 1/K0 calibration that convert centroids with. Retrace reads the raw events
/// through the same handle ([`RawTdf::tof_of`], [`RawTdf::im_of_scan`]), so raw events,
/// centroids and every 1/K0 derived from them share one m/z and one 1/K0 scale.
pub(crate) struct RawTdf {
    pub(crate) reader: FrameReader,
    mz: TofMzCalibration,
    im: ScanImCalibration,
    /// The scan count of every frame recorded without peaks, by frame index ([`empty_frames`]).
    empty: BTreeMap<usize, usize>,
}

/// Frame `i` with its peaks. A frame in `empty` is returned with its scans and no peaks: it
/// stores no compressed payload, which the frame reader would reject.
fn read_frame(
    reader: &FrameReader,
    empty: &BTreeMap<usize, usize>,
    i: usize,
) -> Result<Frame, FrameReaderError> {
    match empty.get(&i) {
        Some(&n_scans) => {
            let mut f = reader.get_frame_without_coordinates(i)?;
            f.scan_offsets = vec![0; n_scans + 1];
            Ok(f)
        }
        None => reader.get(i),
    }
}

/// The scan count of every frame recorded without peaks (`Frames.NumPeaks` 0), keyed by its
/// frame-reader index. Such a frame's blob in `analysis.tdf_bin` is the 8-byte header alone,
/// and the frame reader fails on it with "Decompression fails"; in some acquisitions the
/// first frame of every run is one.
fn empty_frames(path: &str, reader: &FrameReader) -> Result<BTreeMap<usize, usize>> {
    let db = format!("{path}/analysis.tdf");
    let con =
        rusqlite::Connection::open_with_flags(&db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("open {db}"))?;
    let mut st = con.prepare("SELECT Id, NumScans FROM Frames WHERE NumPeaks = 0")?;
    let by_id: BTreeMap<usize, usize> = st
        .query_map([], |r| {
            Ok((r.get::<_, i64>(0)? as usize, r.get::<_, i64>(1)? as usize))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut out = BTreeMap::new();
    if by_id.is_empty() {
        return Ok(out);
    }
    // The reader indexes frames by table row, not by `Id`; map through the id it records.
    for i in 0..reader.len() {
        let id = reader
            .get_frame_without_coordinates(i)
            .map_err(|e| anyhow!("{path}: frame header {i}: {e}"))?
            .index;
        if let Some(&n_scans) = by_id.get(&id) {
            out.insert(i, n_scans);
        }
    }
    info!(
        frames = out.len(),
        "timsTOF: frames recorded without peaks are read as empty frames"
    );
    Ok(out)
}

impl RawTdf {
    /// Frame `i` with its peaks, through [`read_frame`].
    pub(crate) fn frame(&self, i: usize) -> Result<Frame, FrameReaderError> {
        read_frame(&self.reader, &self.empty, i)
    }

    /// The fractional TOF index of an m/z: the inverse of the converter convert uses.
    pub(crate) fn tof_of(&self, mz: f64) -> f64 {
        self.mz.tof(mz)
    }

    /// The m/z of a (fractional) TOF index, as convert computes it.
    pub(crate) fn mz_of(&self, tof: f64) -> f64 {
        self.mz.mz(tof)
    }

    /// The 1/K0 of a TIMS scan index, as convert computes it.
    pub(crate) fn im_of_scan(&self, scan: f64) -> f64 {
        self.im.im(scan)
    }

    /// The scan -> 1/K0 calibration in use, named for a report.
    pub(crate) fn im_calibration(&self) -> &'static str {
        self.im.name()
    }

    pub(crate) fn open(path: &str) -> Result<Self> {
        let meta = MetadataReader::new(path).map_err(|e| anyhow!("{path}: {e}"))?;
        let reader = FrameReader::new(path).map_err(|e| anyhow!("{path}: {e}"))?;
        match reader.get_acquisition() {
            AcquisitionType::DIAPASEF => {}
            other => bail!(
                "{path}: acquisition type {other:?}. The native timsTOF reader handles \
                 diaPASEF only, and the input routing sends every other Bruker .d to \
                 msconvert, so the file changed or its frame table is unusual. Convert it to \
                 mzML with msconvert and pass the mzML."
            ),
        }
        Ok(Self {
            im: ScanImCalibration::read(path, meta.im_converter)?,
            mz: TofMzCalibration::read(path, meta.mz_converter),
            empty: empty_frames(path, &reader)?,
            reader,
        })
    }
}

/// Decode `path` frame by frame and hand every spectrum to `sink` with its scan index
/// (the emission ordinal). `max_frames > 0` reads only the first frames of the run: on this
/// path `--max-spectra` counts frames, not spectra. Returns the number of spectra emitted.
pub(super) fn drive(
    path: &str,
    max_frames: usize,
    p: &TdfParams,
    top_ms1: usize,
    top_ms2: usize,
    mut sink: impl FnMut(u32, Decoded) -> Result<()>,
) -> Result<usize> {
    let RawTdf {
        reader,
        mz,
        im,
        empty,
    } = RawTdf::open(path)?;
    // The MS1 filter's window gate is built once per run, from its diaPASEF windows and
    // the scan count of its first frame.
    let ms1 = if p.ms1_denoise {
        let n_scans = match reader.len() {
            0 => 0,
            _ => read_frame(&reader, &empty, 0)
                .map_err(|e| anyhow!("{path}: frame 0: {e}"))?
                .scan_offsets
                .len()
                .saturating_sub(1),
        };
        let windows = reader.get_dia_windows().unwrap_or_default();
        Some(Ms1Denoise::new(
            &windows,
            n_scans,
            |m| mz.tof(m),
            |s| im.im(s),
        ))
    } else {
        None
    };
    let ctx = Ctx {
        mz,
        im,
        p: *p,
        top_ms1,
        top_ms2,
        ms1,
        ms1_points: AtomicU64::new(0),
        ms1_kept: AtomicU64::new(0),
    };
    let n = match max_frames {
        0 => reader.len(),
        m => m.min(reader.len()),
    };
    info!(frames = n, params = ?p, mz_calibration = ctx.mz.name(), "convert: decoding timsTOF frames");
    let mut emitted = 0usize;
    for start in (0..n).step_by(FRAME_CHUNK) {
        let batch: Vec<Result<Vec<Decoded>>> = (start..(start + FRAME_CHUNK).min(n))
            .into_par_iter()
            .map(|i| {
                let f = read_frame(&reader, &empty, i)
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
    if let Some(d) = &ctx.ms1 {
        let (points, kept) = (
            ctx.ms1_points.load(Ordering::Relaxed),
            ctx.ms1_kept.load(Ordering::Relaxed),
        );
        info!(
            points,
            kept,
            kept_pct = 100.0 * kept as f64 / points.max(1) as f64,
            window_gate = d.gated(),
            "convert: MS1 noise filter (convert.tdf_ms1_denoise)"
        );
    }
    Ok(emitted)
}

/// MS2 frames [`load`] decodes, spread evenly over the run.
pub const LOAD_FRAMES: usize = 256;

/// The MS2 load of a diaPASEF run, from `LOAD_FRAMES` (256) of its MS2 frames spread evenly over
/// the run and centroided with the run's settings: how much of the fragment signal
/// `convert.tdf_min_points` 2 would drop. What `"auto"` decides on (docs/35_diapasef.md,
/// "Automatic centroid floor").
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct TdfLoad {
    /// MS2 frames in the run, and of them decoded.
    pub ms2_frames: usize,
    pub sampled_frames: usize,
    /// Slot spectra of the sampled frames.
    pub spectra: u64,
    /// Centroids of those spectra at `tdf_min_points` 1, and of them the ones below 2 raw
    /// points (a single ion event), which `tdf_min_points` 2 drops.
    pub centroids: u64,
    pub single_ion: u64,
    /// Summed intensity of those centroids, and of the single-ion ones.
    pub intensity: f64,
    pub single_ion_intensity: f64,
}

impl TdfLoad {
    /// The share of MS2 centroids that are single ion events.
    pub fn single_ion_share(&self) -> f64 {
        self.single_ion as f64 / self.centroids.max(1) as f64
    }

    /// The share of MS2 intensity in single ion events.
    pub fn single_ion_intensity_share(&self) -> f64 {
        self.single_ion_intensity / self.intensity.max(f64::MIN_POSITIVE)
    }

    /// Centroids of at least 2 raw points per MS2 spectrum (what `tdf_min_points` 2 keeps).
    pub fn multi_point_per_spectrum(&self) -> f64 {
        (self.centroids - self.single_ion) as f64 / self.spectra.max(1) as f64
    }

    /// Summed MS2 intensity per spectrum.
    pub fn intensity_per_spectrum(&self) -> f64 {
        self.intensity / self.spectra.max(1) as f64
    }

    /// Every measure, for a log line and the manifest.
    pub fn report(&self) -> serde_json::Value {
        let mut v = serde_json::to_value(self).unwrap_or_default();
        if let Some(o) = v.as_object_mut() {
            o.insert("single_ion_share".into(), self.single_ion_share().into());
            o.insert(
                "single_ion_intensity_share".into(),
                self.single_ion_intensity_share().into(),
            );
            o.insert(
                "multi_point_per_spectrum".into(),
                self.multi_point_per_spectrum().into(),
            );
            o.insert(
                "intensity_per_spectrum".into(),
                self.intensity_per_spectrum().into(),
            );
        }
        v
    }
}

/// `convert.tdf_min_points = "auto"` chooses 2 when the runs' median share of MS2 intensity in
/// single ion events ([`TdfLoad::single_ion_intensity_share`]) is below this, and 1 otherwise.
///
/// Measured on 256 sampled MS2 frames of two runs per set (one where the set has one): 0.72-0.89
/// on the low-input sets (200 pg HeLa and its blanks, 1 ng HeLa/E. coli, 250 pg 5-min Ultra,
/// three single-cell HeLa plates), 0.15-0.41 on the bulk sets (ProteoBench HYE, 50 ng 5-min
/// Ultra 2, the E. coli entrapment run, a synthetic HLA-I pool, 60-min A549/K562). The threshold
/// sits closer to the bulk side because the costs are not symmetric: 2 on a low-input run lost
/// 18-74% of the precursors, 1 on a bulk run costs memory and wall.
pub const AUTO_SINGLE_ION_INTENSITY_SHARE: f64 = 0.5;

/// `convert.tdf_min_points = "auto"` chooses 2, whatever the load, for a search whose library
/// holds more precursors than this (or, built from the FASTA, digests non-specifically).
///
/// Immunopeptidomics searches are the case: non-specific libraries of 309-379 M precursors
/// against 14.7 M for the tryptic HYE library, and runs that can sit on the low-load side of
/// the intensity share (0.55-0.64 on two timsTOF SCP HLA-I runs). On a 379 M-precursor HLA-I
/// search, 2 (0.13.0) gave 1,368 peptides per run at 195 GB; 1 (0.16.0) cut the confident seed
/// anchors of one run from 203 to 0 and stopped at 407 GB.
pub const AUTO_LIBRARY_PRECURSORS: u64 = 100_000_000;

/// What `convert.tdf_min_points = "auto"` chose for a search, and on what.
#[derive(Clone, Debug, PartialEq)]
pub struct MinPointsAuto {
    pub min_points: u32,
    /// The median over the runs of [`TdfLoad::single_ion_intensity_share`].
    pub measure: f64,
    pub threshold: f64,
    /// Why the search space alone decided for 2 (`AUTO_LIBRARY_PRECURSORS`), if it did.
    pub large_search_space: Option<String>,
}

/// The floor `"auto"` chooses for runs of these loads: one value for every run, from the
/// median run, so one outlying run (a blank, a failed injection) does not decide for the
/// rest; 2 whatever the load when `large_search_space` names a reason. No load (no
/// diaPASEF run) keeps the default 1, which nothing else reads.
pub fn auto_min_points(loads: &[TdfLoad], large_search_space: Option<String>) -> MinPointsAuto {
    let mut v: Vec<f64> = loads
        .iter()
        .map(TdfLoad::single_ion_intensity_share)
        .collect();
    v.sort_by(f64::total_cmp);
    let measure = match v.len() {
        0 => 1.0,
        n if n % 2 == 1 => v[n / 2],
        n => (v[n / 2 - 1] + v[n / 2]) / 2.0,
    };
    let large_search_space = large_search_space.filter(|_| !v.is_empty());
    let two = !v.is_empty()
        && (measure < AUTO_SINGLE_ION_INTENSITY_SHARE || large_search_space.is_some());
    MinPointsAuto {
        min_points: if two { 2 } else { 1 },
        measure,
        threshold: AUTO_SINGLE_ION_INTENSITY_SHARE,
        large_search_space,
    }
}

/// [`TdfLoad`] of `path` under the centroiding settings `p` (its floor ignored).
pub fn load(path: &str, p: &TdfParams) -> Result<TdfLoad> {
    let raw = RawTdf::open(path)?;
    let ms2: Vec<usize> = (0..raw.reader.len())
        .filter(|&i| {
            raw.reader
                .get_frame_without_coordinates(i)
                .is_ok_and(|f| f.ms_level == MSLevel::MS2)
        })
        .collect();
    let n = ms2.len().min(LOAD_FRAMES);
    let picks: Vec<usize> = (0..n).map(|k| ms2[k * ms2.len() / n]).collect();
    let (one, two) = (
        TdfParams {
            min_points: 1,
            ..*p
        },
        TdfParams {
            min_points: 2,
            ..*p
        },
    );
    let frames: Vec<Result<TdfLoad>> = picks
        .par_iter()
        .map(|&i| {
            let f = raw
                .frame(i)
                .map_err(|e| anyhow!("{path}: frame {i}: {e}"))?;
            let q = &f.quadrupole_settings;
            let mut l = TdfLoad::default();
            for k in 0..q.len() {
                let mut pts = scan_points(&f, q.scan_starts[k]..q.scan_ends[k]);
                let mz = |t: u32| raw.mz.mz_u32(t);
                let im = |s: f64| raw.im.im(s);
                let all = centroid_2d(&mut pts.clone(), mz, im, &one);
                let kept = centroid_2d(&mut pts, mz, im, &two);
                let sum = |c: &Centroids| c.inten.iter().map(|&x| x as f64).sum::<f64>();
                l.spectra += 1;
                l.centroids += all.inten.len() as u64;
                l.single_ion += (all.inten.len() - kept.inten.len()) as u64;
                l.intensity += sum(&all);
                l.single_ion_intensity += sum(&all) - sum(&kept);
            }
            Ok(l)
        })
        .collect();
    // Summed in frame order, so the result does not depend on the thread count.
    let mut out = TdfLoad {
        ms2_frames: ms2.len(),
        sampled_frames: n,
        ..TdfLoad::default()
    };
    for l in frames {
        let l = l?;
        out.spectra += l.spectra;
        out.centroids += l.centroids;
        out.single_ion += l.single_ion;
        out.intensity += l.intensity;
        out.single_ion_intensity += l.single_ion_intensity;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: TdfParams = TdfParams {
        mz_ppm: 10.0,
        im_gap_scans: 5,
        min_points: 2,
        mz_valley: 0.0,
        mz_smooth_ppm: 4.0,
        im_valley: 0.0,
        im_smooth_scans: 4.0,
        ms1_denoise: false,
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

    /// The `MzCalibration` model against a published reference row (the test vector of
    /// mzdata's `io/tdf/calibration.rs`, a timsTOF with `dC1` = 20 and a frame 0.0044 degree
    /// warmer than the calibration): index 0 at m/z 99.990834, index 636,029 at 1,700.005.
    #[test]
    fn mz_calibration_model_matches_the_reference_row() {
        let row = (
            2,
            [
                0.125,
                25741.0,
                20.9410989491122,
                0.0,
                20.0,
                0.0,
                286.065160463331,
                154317.348188993,
                0.0,
                0.0,
                0.0,
            ],
        );
        let linear = Tof2MzConverter::from_boundaries(100.0, 1700.0, 636030);
        let c = TofMzCalibration::from_parts(Some(row), 20.9455139021767, 0.0, 636_030, linear);
        assert_eq!(c.name(), "MzCalibration model");
        assert!((c.mz_u32(0) - 99.990834).abs() < 1e-3, "{}", c.mz_u32(0));
        assert!(
            (c.mz_u32(636_029) - 1700.005).abs() < 1e-3,
            "{}",
            c.mz_u32(636_029)
        );
        // table and direct evaluation agree; tof() inverts mz()
        for t in [0u32, 1, 12_345, 400_000, 636_029] {
            assert!((c.mz_u32(t) - c.mz(t as f64 + 0.0)).abs() < 1e-9);
            assert!((c.tof(c.mz(t as f64)) - t as f64).abs() < 1e-4, "{t}");
        }
        assert!((c.tof(c.mz(1234.5)) - 1234.5).abs() < 1e-4);
    }

    /// A quadratic and a constant term (a 2020 timsTOF Pro row) move m/z by tens of ppm against
    /// the square law; the cubic solve still inverts exactly.
    #[test]
    fn mz_calibration_model_with_quadratic_and_offset_terms() {
        let row = (
            1,
            [
                0.2,
                24888.4,
                25.082415568717185,
                20.33742811348631,
                0.0,
                0.0,
                300.80436361188765,
                157290.6485284086,
                -0.005015829952769195,
                0.0,
                0.09391436164385845,
            ],
        );
        let linear = Tof2MzConverter::from_boundaries(95.0, 1705.0, 397_605);
        let c = TofMzCalibration::from_parts(Some(row), 25.078, 20.157, 397_605, linear);
        let lin = TofMzCalibration::Linear(linear);
        assert!((c.mz_u32(0) - 95.0).abs() < 0.01, "{}", c.mz_u32(0));
        assert!(
            (c.mz_u32(397_604) - 1705.0).abs() < 0.01,
            "{}",
            c.mz_u32(397_604)
        );
        let t400 = c.tof(400.0);
        let ppm = (lin.mz(t400) - 400.0) / 400.0 * 1e6;
        assert!(
            (-90.0..-70.0).contains(&ppm),
            "linear form at m/z 400: {ppm} ppm"
        );
        for mz in [150.0, 400.0, 999.0, 1600.0] {
            assert!((c.mz(c.tof(mz)) - mz).abs() < 1e-7, "{mz}");
        }
    }

    /// No usable row falls back to the linear form.
    #[test]
    fn mz_calibration_without_a_row_is_linear() {
        let linear = Tof2MzConverter::from_boundaries(100.0, 1700.0, 1000);
        let c = TofMzCalibration::from_parts(None, 0.0, 0.0, 1000, linear);
        assert_eq!(
            c.name(),
            "sqrt(m/z) linear between the acquisition range bounds"
        );
        assert!((c.mz(0.0) - 100.0).abs() < 1e-9);
        assert!((c.mz_u32(1000) - 1700.0).abs() < 1e-6);
    }

    #[test]
    fn the_shipped_defaults_are_the_measured_settings() {
        let d = TdfParams::default();
        assert_eq!((d.mz_ppm, d.im_gap_scans, d.min_points), (10.0, 30, 1));
        assert_eq!((d.mz_valley, d.im_valley), (0.0, 0.0));
        assert_eq!((d.mz_smooth_ppm, d.im_smooth_scans), (8.0, 8.0));
    }

    /// A load whose intensity is `share` in single ion events.
    fn load_of(share: f64) -> TdfLoad {
        TdfLoad {
            spectra: 10,
            centroids: 1000,
            single_ion: 900,
            intensity: 1000.0,
            single_ion_intensity: share * 1000.0,
            ..TdfLoad::default()
        }
    }

    /// `"auto"` keeps single ion events (1) at or above the threshold share and drops them (2)
    /// below it, from the median run, and keeps the default without a diaPASEF run.
    #[test]
    fn auto_min_points_decides_on_the_median_single_ion_intensity_share() {
        let t = AUTO_SINGLE_ION_INTENSITY_SHARE;
        let mp = |l: &[TdfLoad]| auto_min_points(l, None).min_points;
        assert_eq!(mp(&[]), 1);
        assert_eq!(mp(&[load_of(t)]), 1);
        assert_eq!(mp(&[load_of(t - 0.01)]), 2);
        assert_eq!(mp(&[load_of(0.99)]), 1);
        // One outlying run does not decide for the rest, in either direction.
        let low = [load_of(0.8), load_of(0.75), load_of(0.2)];
        assert_eq!(mp(&low), 1);
        let bulk = [load_of(0.15), load_of(0.4), load_of(0.9)];
        let d = auto_min_points(&bulk, None);
        assert_eq!((d.min_points, d.measure, d.threshold), (2, 0.4, t));
        // An even count takes the mean of the middle two, whatever the input order.
        let even = auto_min_points(
            &[load_of(0.8), load_of(0.2), load_of(0.4), load_of(0.7)],
            None,
        );
        assert!((even.measure - 0.55).abs() < 1e-12);
        assert_eq!(even.min_points, 1);
    }

    /// A large search space (an immunopeptidomics library) decides for 2 on low-load runs
    /// too, and is moot without a diaPASEF run.
    #[test]
    fn a_large_search_space_decides_for_two_on_any_load() {
        let why = || Some("the library holds 379312896 precursors".to_string());
        let d = auto_min_points(&[load_of(0.64), load_of(0.55)], why());
        assert_eq!(d.min_points, 2);
        assert!(d.measure > AUTO_SINGLE_ION_INTENSITY_SHARE);
        assert_eq!(d.large_search_space, why());
        let none = auto_min_points(&[], why());
        assert_eq!((none.min_points, none.large_search_space), (1, None));
    }

    #[test]
    fn tdf_load_shares_and_rates() {
        let l = TdfLoad {
            spectra: 4,
            centroids: 100,
            single_ion: 75,
            intensity: 400.0,
            single_ion_intensity: 100.0,
            ..TdfLoad::default()
        };
        assert_eq!(l.single_ion_share(), 0.75);
        assert_eq!(l.single_ion_intensity_share(), 0.25);
        assert_eq!(l.multi_point_per_spectrum(), 6.25);
        assert_eq!(l.intensity_per_spectrum(), 100.0);
        // An empty sample reads as shares of 0, not NaN.
        let e = TdfLoad::default();
        assert_eq!(e.single_ion_share(), 0.0);
        assert_eq!(e.single_ion_intensity_share(), 0.0);
        let r = l.report();
        assert_eq!(r["single_ion_share"], 0.75);
        assert_eq!(r["centroids"], 100);
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

    /// One generic cut finder serves both axes: the same profile cut along TOF and along
    /// scans gives the same cut positions.
    #[test]
    fn valley_cuts_is_axis_agnostic() {
        let x = [1u32, 4, 9, 4, 1, 1, 4, 9, 4, 1];
        let along_tof: Vec<_> = x
            .iter()
            .enumerate()
            .map(|(i, &x)| (1000 + i as u32, 7, x))
            .collect();
        let along_scan: Vec<_> = x
            .iter()
            .enumerate()
            .map(|(i, &x)| (1000, 10 + i as u32, x))
            .collect();
        let a = valley_cuts(&along_tof, |q| q.0, |t| t as f64, |_| 1.0, 0.5);
        let b = valley_cuts(&along_scan, |q| q.1, |s| s as f64, |_| 1.0, 0.5);
        assert_eq!(a, vec![4]);
        assert_eq!(a, b);
        // Off at any depth that no valley undercuts.
        assert!(valley_cuts(&along_scan, |q| q.1, |s| s as f64, |_| 1.0, 0.1).is_empty());
    }

    #[test]
    fn cap_keeps_the_most_intense_in_mz_order() {
        let c = Centroids {
            mz: vec![1.0, 2.0, 3.0],
            inten: vec![5.0, 1.0, 9.0],
            im: vec![0.7, 0.8, 0.9],
        };
        let k = cap(c, 2);
        assert_eq!(k.mz, vec![1.0, 3.0]);
        assert_eq!(k.im, vec![0.7, 0.9]);
        // Ties keep the lower index, as on the mzML path.
        let t = Centroids {
            mz: vec![1.0, 2.0, 3.0],
            inten: vec![4.0, 4.0, 4.0],
            im: vec![0.7, 0.8, 0.9],
        };
        assert_eq!(cap(t, 2).mz, vec![1.0, 2.0]);
    }

    fn linear() -> Scan2ImConverter {
        // 1/K0 0.6 at scan 0 to 1.6 at scan 1000.
        Scan2ImConverter::from_boundaries(0.6, 1.6, 1000)
    }

    #[test]
    fn calibration_model_2_is_used_from_the_first_row() {
        // C0 offset 0, C1 1, C2 0, C3 1, C4 0: offset 0, slope 1, so 1/K0 = 1/(c6 + c7/scan).
        let m2 = |c6: f64, c7: f64| (2i64, [0.0, 1.0, 0.0, 1.0, 0.0, c6, c7]);
        let cal = ScanImCalibration::from_rows(&[m2(0.5, 100.0)], linear());
        assert!((cal.im(200.0) - 1.0 / (0.5 + 100.0 / 200.0)).abs() < 1e-12);
        // Several rows: only the first is used.
        let two = ScanImCalibration::from_rows(&[m2(0.5, 100.0), m2(0.9, 10.0)], linear());
        assert_eq!(two.im(200.0).to_bits(), cal.im(200.0).to_bits());
    }

    #[test]
    fn calibration_falls_back_to_the_linear_range_without_model_2() {
        let l = linear();
        for rows in [
            Vec::new(),
            vec![(1i64, [0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 100.0])],
        ] {
            let cal = ScanImCalibration::from_rows(&rows, linear());
            assert!(matches!(cal, ScanImCalibration::Linear(_)));
            assert_eq!(cal.im(250.0).to_bits(), l.convert(250.0).to_bits());
        }
        // A model 2 behind a first row of another model is not used either.
        let later = ScanImCalibration::from_rows(
            &[(1, [0.0; 7]), (2, [0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 100.0])],
            linear(),
        );
        assert!(matches!(later, ScanImCalibration::Linear(_)));
    }

    /// The calibration is read from a real SQLite table, rows in `Id` order.
    #[test]
    fn calibration_rows_are_read_from_the_tdf_in_id_order() {
        let d = std::env::temp_dir().join(format!("mumdia_tdf_cal_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        {
            let con = rusqlite::Connection::open(d.join("analysis.tdf")).unwrap();
            con.execute_batch(
                "CREATE TABLE TimsCalibration (Id INTEGER PRIMARY KEY, ModelType INTEGER,
                   C0 REAL, C1 REAL, C2 REAL, C3 REAL, C4 REAL, C5 REAL, C6 REAL, C7 REAL);
                 INSERT INTO TimsCalibration VALUES (2, 2, 0, 1, 0, 1, 0, 0, 0.9, 10);
                 INSERT INTO TimsCalibration VALUES (1, 2, 0, 1, 0, 1, 0, 0, 0.5, 100);",
            )
            .unwrap();
        }
        let cal = ScanImCalibration::read(d.to_str().unwrap(), linear()).unwrap();
        assert!((cal.im(200.0) - 1.0 / (0.5 + 100.0 / 200.0)).abs() < 1e-12);
        std::fs::remove_dir_all(&d).ok();
    }

    /// A two-frame diaPASEF `.d` (an MS1 frame, then an MS2 frame of two slots) whose frames
    /// store no payload: each blob in `analysis.tdf_bin` is the 8-byte header alone. The MS2
    /// frame declares `ms2_peaks` peaks.
    fn headers_only_d(tag: &str, ms2_peaks: i64) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia_tdf_{tag}_{}.d", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let blob = [8u32.to_le_bytes(), 100u32.to_le_bytes()].concat();
        std::fs::write(d.join("analysis.tdf_bin"), [blob.clone(), blob].concat()).unwrap();
        let con = rusqlite::Connection::open(d.join("analysis.tdf")).unwrap();
        con.execute_batch(&format!(
            "CREATE TABLE GlobalMetadata (Key TEXT, Value TEXT);
             INSERT INTO GlobalMetadata VALUES ('TimsCompressionType', '2'),
               ('AcquisitionSoftware', 'test'), ('MzAcqRangeLower', '100'),
               ('MzAcqRangeUpper', '1700'), ('OneOverK0AcqRangeLower', '0.6'),
               ('OneOverK0AcqRangeUpper', '1.6'), ('DigitizerNumSamples', '400000');
             CREATE TABLE Frames (Id INTEGER PRIMARY KEY, Time REAL, ScanMode INTEGER,
               MsMsType INTEGER, NumPeaks INTEGER, NumScans INTEGER, TimsId INTEGER,
               AccumulationTime REAL);
             INSERT INTO Frames VALUES (1, 1.0, 9, 0, 0, 100, 0, 100.0),
               (2, 1.1, 9, 9, {ms2_peaks}, 100, 8, 100.0);
             CREATE TABLE DiaFrameMsMsInfo (Frame INTEGER, WindowGroup INTEGER);
             INSERT INTO DiaFrameMsMsInfo VALUES (2, 1);
             CREATE TABLE DiaFrameMsMsWindows (WindowGroup INTEGER, ScanNumBegin INTEGER,
               ScanNumEnd INTEGER, IsolationMz REAL, IsolationWidth REAL, CollisionEnergy REAL);
             INSERT INTO DiaFrameMsMsWindows VALUES (1, 10, 40, 500.0, 25.0, 30.0),
               (1, 50, 80, 700.0, 25.0, 30.0);
             CREATE TABLE TimsCalibration (Id INTEGER PRIMARY KEY, ModelType INTEGER,
               C0 REAL, C1 REAL, C2 REAL, C3 REAL, C4 REAL, C5 REAL, C6 REAL, C7 REAL);"
        ))
        .unwrap();
        d
    }

    /// A frame recorded without peaks has no payload to decompress. It becomes empty
    /// spectra in frame order (one MS1, one per MS2 slot), with and without the MS1 filter,
    /// whose scan count is read from the first frame.
    #[test]
    fn frames_recorded_without_peaks_decode_as_empty_spectra() {
        let d = headers_only_d("empty", 0);
        for ms1_denoise in [false, true] {
            let p = TdfParams {
                ms1_denoise,
                ..TdfParams::default()
            };
            let mut got = Vec::new();
            let n = drive(d.to_str().unwrap(), 0, &p, 0, 0, |i, s| {
                got.push(match s {
                    Decoded::Ms1 { mz, .. } => (i, "ms1", 0.0, mz.len()),
                    Decoded::Ms2(r) => (i, "ms2", r.wt, r.mz.len()),
                    _ => (i, "other", 0.0, 0),
                });
                Ok(())
            })
            .unwrap();
            assert_eq!(n, 3);
            assert_eq!(
                got,
                vec![
                    (0, "ms1", 0.0, 0),
                    (1, "ms2", 500.0, 0),
                    (2, "ms2", 700.0, 0)
                ]
            );
        }
        std::fs::remove_dir_all(&d).ok();
    }

    /// The load probe decodes MS2 frames only, every slot of them, and reads an empty frame
    /// as no centroids.
    #[test]
    fn load_samples_ms2_frames_and_their_slots() {
        let d = headers_only_d("load", 0);
        let l = load(d.to_str().unwrap(), &TdfParams::default()).unwrap();
        assert_eq!((l.ms2_frames, l.sampled_frames, l.spectra), (1, 1, 2));
        assert_eq!((l.centroids, l.single_ion), (0, 0));
        std::fs::remove_dir_all(&d).ok();
    }

    /// Only a frame that declares no peaks is read as empty: a frame that declares peaks
    /// and stores no payload is still an error.
    #[test]
    fn a_frame_with_peaks_and_no_payload_is_still_an_error() {
        let d = headers_only_d("corrupt", 3);
        let err = drive(
            d.to_str().unwrap(),
            0,
            &TdfParams::default(),
            0,
            0,
            |_, _| Ok(()),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("frame 1") && err.contains("Decompression"),
            "{err}"
        );
        std::fs::remove_dir_all(&d).ok();
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
