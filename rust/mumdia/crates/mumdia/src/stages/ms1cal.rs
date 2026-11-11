//! MS1 precursor mass offset (`extract.ms1_calibrate`).
//!
//! The fragment offset is learned by the seed (`masscal`), the MS1 one was not: on the
//! single-cell and HeLa Astral runs the MS1 mono peaks sit +1.9 to +2.35 ppm from the
//! theoretical m/z, so a +-5 ppm MS1 read is centred off the signal and reaches further into
//! interfering peaks on the other side. The offset is the median signed ppm error of the
//! most intense MS1 peak within `search_ppm` of each confident seed anchor's precursor m/z,
//! in the MS1 scan nearest its observed RT and the scans either side.
//!
//! On spectra that carry 1/K0 (a diaPASEF `.d`, seed v2 with `observed_im`) the most intense
//! peak within 15 ppm is often another ion in a different mobility band, so an anchor with
//! an `observed_im` takes, over the same scans, the in-tolerance peak whose 1/K0 lies
//! nearest it (`spectra::ms1_peak_near`, the rule extract's `ms1_apex_im` uses). Every other
//! anchor, and every 3D run, keeps the most-intense rule.
//!
//! `extract.ms1_calibrate = "surface"` fits the same anchors' errors as a smooth function of
//! precursor m/z and retention time instead of one median ([`fit_surface`]), and
//! `extract.prec_tol_auto` sets `extract.prec_tol_ppm` from their residuals after the
//! calibration ([`tolerance_from`]). Measured on a ZenoTOF 8600 ZT Scan DIA run (24,124
//! anchors with an MS1 peak): median +2.84 ppm, robust SD 2.67 ppm, the median drifting
//! from +3.4 to +2.3 ppm over the gradient and from +3.4 to +2.6 ppm over m/z.
use anyhow::Result;
use mumdia_core::config::{interp_held, Ms1Cal, Ms1Surface};
use mumdia_io::table::TableFile;

use crate::calibrate::{percentile, Loess};
use crate::spectra::load_ms1;

/// Fewest anchors that still give a usable median.
const MIN_ANCHORS: usize = 30;

/// Fewest anchors a surface is fitted from; with fewer, `"surface"` keeps the one offset.
const MIN_SURFACE_ANCHORS: usize = 200;

/// Most anchors a surface is fitted from: above it every k-th anchor in seed order, so the
/// fit stays a few seconds on any run.
const MAX_SURFACE_ANCHORS: usize = 40_000;

/// Points of each grid of the surface.
const SURFACE_GRID: usize = 20;

/// LOESS span, robustness iterations and backfitting rounds of the surface.
const SURFACE_SPAN: f64 = 0.3;
const SURFACE_ROBUST_ITERS: usize = 2;
const SURFACE_ROUNDS: usize = 3;

/// `extract.prec_tol_auto`: the tolerance is this many robust standard deviations of the
/// calibrated residuals, clamped to `[PREC_TOL_AUTO_MIN, PREC_TOL_AUTO_MAX]` ppm (the upper
/// bound is the window the anchors' peaks are searched in).
pub const PREC_TOL_AUTO_K: f64 = 3.0;
const PREC_TOL_AUTO_MIN: f64 = 1.0;
const PREC_TOL_AUTO_MAX: f64 = 15.0;

/// The window (ppm) each anchor's MS1 peak is searched in.
const SEARCH_PPM: f64 = 15.0;

/// The robust standard deviation of a normal sample per unit of its median absolute
/// deviation.
const MAD_TO_SIGMA: f64 = 1.4826;

/// `(offset_ppm, anchors used)`, or `None` with fewer than `MIN_ANCHORS` anchors.
pub fn estimate(
    seed_psms: &str,
    ms1_path: &str,
    q_max: f64,
    search_ppm: f64,
) -> Result<Option<(f64, usize)>> {
    Ok(
        anchor_errors(seed_psms, ms1_path, q_max, search_ppm)?.map(|e| {
            let ppm: Vec<f64> = e.iter().map(|a| a.ppm).collect();
            (percentile(&ppm, 0.5), ppm.len())
        }),
    )
}

/// One anchor's MS1 error: its precursor m/z, its observed retention time and the signed ppm
/// error of the MS1 peak it took.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchorError {
    pub mz: f64,
    pub rt: f64,
    pub ppm: f64,
}

/// The MS1 error of every confident seed anchor with an MS1 peak within `search_ppm`, in
/// seed order, or `None` with fewer than `MIN_ANCHORS` such anchors.
pub fn anchor_errors(
    seed_psms: &str,
    ms1_path: &str,
    q_max: f64,
    search_ppm: f64,
) -> Result<Option<Vec<AnchorError>>> {
    let seed = TableFile::open(seed_psms)?;
    let label = seed.str("label")?;
    let q = seed.f64("spectrum_q")?;
    let mz = seed.f64("precursor_mz")?;
    let rt = seed.f64("observed_rt")?;
    // The seed's apex 1/K0 (seed v2), read only when the seed has it: a 3D seed has none.
    let obs_im: Option<Vec<Option<f64>>> = if seed.has_column("observed_im") {
        Some(seed.opt_f64("observed_im")?)
    } else {
        None
    };
    let anchors: Vec<(f64, f64, Option<f64>)> = (0..seed.nrows)
        .filter(|&i| label[i] == "target" && q[i] < q_max && mz[i].is_finite() && rt[i].is_finite())
        .map(|i| (mz[i], rt[i], obs_im.as_ref().and_then(|v| v[i])))
        .collect();
    if anchors.len() < MIN_ANCHORS {
        return Ok(None);
    }
    let ms1 = load_ms1(ms1_path)?;
    if ms1.is_empty() {
        return Ok(None);
    }
    let rts: Vec<f64> = ms1.iter().map(|s| s.rt_seconds).collect();
    let mut ppm = Vec::with_capacity(anchors.len());
    for &(m, r, im) in &anchors {
        let j = rts.partition_point(|&x| x < r).min(rts.len() - 1);
        let j = if j > 0 && (r - rts[j - 1]).abs() < (rts[j] - r).abs() {
            j - 1
        } else {
            j
        };
        let scans = &ms1[j.saturating_sub(1)..=(j + 1).min(ms1.len() - 1)];
        // 4D: the in-tolerance peak nearest the anchor's 1/K0 over the same scans, the
        // first scan winning a tie, as the most-intense rule below lets the first win.
        if let Some(im) = im.filter(|_| scans.iter().any(|s| s.im.is_some())) {
            let mut best: Option<(f64, f64)> = None; // (1/K0 distance, observed m/z)
            for s in scans {
                let Some(k) = crate::spectra::ms1_peak_near(s, m, search_ppm, Some(im)) else {
                    continue;
                };
                let d = (s.im.as_ref().expect("ms1_peak_near needs 1/K0")[k] as f64 - im).abs();
                if best.is_none_or(|(b, _)| d < b) {
                    best = Some((d, s.mz[k] as f64));
                }
            }
            if let Some((_, obs)) = best {
                ppm.push(AnchorError {
                    mz: m,
                    rt: r,
                    ppm: (obs - m) / m * 1e6,
                });
            }
            continue;
        }
        let (lo, hi) = (m * (1.0 - search_ppm * 1e-6), m * (1.0 + search_ppm * 1e-6));
        let mut best: Option<(f32, f64)> = None;
        for s in scans {
            let a = s.mz.partition_point(|&x| (x as f64) < lo);
            let mut i = a;
            while i < s.mz.len() && (s.mz[i] as f64) <= hi {
                if best.is_none_or(|(b, _)| s.intensity[i] > b) {
                    best = Some((s.intensity[i], s.mz[i] as f64));
                }
                i += 1;
            }
        }
        if let Some((_, obs)) = best {
            ppm.push(AnchorError {
                mz: m,
                rt: r,
                ppm: (obs - m) / m * 1e6,
            });
        }
    }
    if ppm.len() < MIN_ANCHORS {
        return Ok(None);
    }
    Ok(Some(ppm))
}

/// Median and robust standard deviation (1.4826 MAD) of `v`.
fn median_sd(v: &[f64]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let med = percentile(v, 0.5);
    let dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
    (med, MAD_TO_SIGMA * percentile(&dev, 0.5))
}

/// The robust standard deviation `extract.prec_tol_auto` uses: 1.4826 MAD of the residuals,
/// then again over the residuals within four of those of their median, so the random
/// matches a 15 ppm search admits (the most intense peak in the window) widen it less.
pub fn residual_sd(resid: &[f64]) -> f64 {
    let (med, sd0) = median_sd(resid);
    if sd0.is_nan() || sd0 <= 0.0 {
        return sd0;
    }
    let kept: Vec<f64> = resid
        .iter()
        .copied()
        .filter(|r| (r - med).abs() <= 4.0 * sd0)
        .collect();
    median_sd(&kept).1
}

/// The tolerance `extract.prec_tol_auto` sets from a residual standard deviation.
pub fn tolerance_from(sd: f64) -> f64 {
    (PREC_TOL_AUTO_K * sd).clamp(PREC_TOL_AUTO_MIN, PREC_TOL_AUTO_MAX)
}

/// `f` sampled on `SURFACE_GRID` points from `lo` to `hi`.
fn sample(f: &Loess, lo: f64, hi: f64) -> (Vec<f64>, Vec<f64>) {
    let n = SURFACE_GRID;
    let g: Vec<f64> = (0..n)
        .map(|k| lo + (hi - lo) * k as f64 / (n - 1) as f64)
        .collect();
    let v: Vec<f64> = g.iter().map(|&x| f.predict(x)).collect();
    (g, v)
}

/// The smallest and largest of `v`.
fn span(v: &[f64]) -> (f64, f64) {
    v.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        })
}

/// The MS1 error surface of `extract.ms1_calibrate = "surface"`: ppm = f(m/z) + g(rt), each
/// a robust LOESS (span 0.3, two bisquare iterations), fitted by backfitting over three
/// rounds on the anchors whose error lies within four robust standard deviations of the
/// median (a peak another ion put in the 15 ppm window is not an MS1 error), then sampled
/// on 20-point grids over the anchors' range. g is centred on the anchors and f refitted on
/// the centred g, so f carries the level. `None` with fewer than `MIN_SURFACE_ANCHORS`
/// inliers. Deterministic: the anchors are taken in seed order and the LOESS sorts its own
/// points.
pub fn fit_surface(errors: &[AnchorError]) -> Option<Ms1Surface> {
    let all: Vec<f64> = errors.iter().map(|e| e.ppm).collect();
    let (med, sd) = median_sd(&all);
    let mut kept: Vec<AnchorError> = errors
        .iter()
        .copied()
        .filter(|e| sd > 0.0 && (e.ppm - med).abs() <= 4.0 * sd)
        .collect();
    if kept.len() < MIN_SURFACE_ANCHORS {
        return None;
    }
    if kept.len() > MAX_SURFACE_ANCHORS {
        let k = kept.len().div_ceil(MAX_SURFACE_ANCHORS);
        kept = kept.into_iter().step_by(k).collect();
    }
    let mz: Vec<f64> = kept.iter().map(|e| e.mz).collect();
    let rt: Vec<f64> = kept.iter().map(|e| e.rt).collect();
    let y: Vec<f64> = kept.iter().map(|e| e.ppm).collect();
    let ((mlo, mhi), (rlo, rhi)) = (span(&mz), span(&rt));
    let n = kept.len();
    let fit_f = |g_at: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let ym: Vec<f64> = (0..n).map(|i| y[i] - g_at[i]).collect();
        let f = Loess::fit_robust(&mz, &ym, SURFACE_SPAN, 200, SURFACE_ROBUST_ITERS);
        sample(&f, mlo, mhi)
    };
    let mut g_at = vec![0.0f64; n];
    let (mut rt_grid, mut rt_ppm) = (Vec::new(), Vec::new());
    for _ in 0..SURFACE_ROUNDS {
        let (mz_grid, mz_ppm) = fit_f(&g_at);
        let f_at: Vec<f64> = mz
            .iter()
            .map(|&x| interp_held(&mz_grid, &mz_ppm, x))
            .collect();
        let yr: Vec<f64> = (0..n).map(|i| y[i] - f_at[i]).collect();
        let g = Loess::fit_robust(&rt, &yr, SURFACE_SPAN, 200, SURFACE_ROBUST_ITERS);
        (rt_grid, rt_ppm) = sample(&g, rlo, rhi);
        g_at = rt
            .iter()
            .map(|&x| interp_held(&rt_grid, &rt_ppm, x))
            .collect();
        // Centre g on the anchors; the level belongs to f.
        let mean = g_at.iter().sum::<f64>() / n as f64;
        for v in rt_ppm.iter_mut() {
            *v -= mean;
        }
        for v in g_at.iter_mut() {
            *v -= mean;
        }
    }
    let (mz_grid, mz_ppm) = fit_f(&g_at);
    Some(Ms1Surface {
        mz_grid,
        mz_ppm,
        rt_grid,
        rt_ppm,
    })
}

/// The one offset and the MS1 calibration record `extract_cfg` hands extract, from the
/// anchors' errors: a surface when `surface` (`ms1_calibrate = "surface"`), the tolerance
/// when `tol_auto` (`prec_tol_auto`), from the residuals after the surface or, without
/// one, after the offset.
fn calibrate(errors: &[AnchorError], surface: bool, tol_auto: bool) -> (f64, Ms1Cal) {
    let raw: Vec<f64> = errors.iter().map(|e| e.ppm).collect();
    let (raw_median, raw_sd) = median_sd(&raw);
    let offset = percentile(&raw, 0.5);
    let off_resid: Vec<f64> = raw.iter().map(|p| p - offset).collect();
    let offset_sd = median_sd(&off_resid).1;
    let fitted = if surface { fit_surface(errors) } else { None };
    let surf_resid: Option<Vec<f64>> = fitted.as_ref().map(|s| {
        errors
            .iter()
            .map(|e| e.ppm - s.ppm_at(e.mz, e.rt))
            .collect()
    });
    let surface_sd = surf_resid.as_ref().map(|r| median_sd(r).1);
    let (tol_sd, tol) = if tol_auto {
        let sd = residual_sd(surf_resid.as_deref().unwrap_or(&off_resid));
        (Some(sd), Some(tolerance_from(sd)))
    } else {
        (None, None)
    };
    (
        offset,
        Ms1Cal {
            mode: if fitted.is_some() {
                "surface"
            } else {
                "offset"
            }
            .to_string(),
            surface: fitted,
            n_anchors: errors.len(),
            raw_median_ppm: raw_median,
            raw_sd_ppm: raw_sd,
            offset_sd_ppm: offset_sd,
            surface_sd_ppm: surface_sd,
            tol_sd_ppm: tol_sd,
            prec_tol_ppm: tol,
        },
    )
}

/// `cfg` with `ms1_ppm_offset` learned from `seed_psms` when `ms1_calibrate` is set; `cfg`
/// unchanged otherwise, or when too few anchors find an MS1 peak (logged). Under
/// `ms1_calibrate = "surface"` or `prec_tol_auto` the surface, the tolerance and their
/// diagnostics go into `ms1_cal` (and `prec_tol_ppm`); with both off this is the one-offset
/// path, unchanged.
pub fn extract_cfg(
    cfg: &mumdia_core::config::ExtractConfig,
    seed_psms: &str,
    ms1_path: &str,
    q_max: f64,
) -> Result<mumdia_core::config::ExtractConfig> {
    let mut out = cfg.clone();
    if !cfg.ms1_calibrate.is_on() {
        return Ok(out);
    }
    if cfg.ms1_calibrate.is_surface() || cfg.prec_tol_auto {
        let Some(errors) = anchor_errors(seed_psms, ms1_path, q_max, SEARCH_PPM)? else {
            tracing::warn!(
                "extract: too few seed anchors with an MS1 peak; MS1 offset left at {}, no \
                 surface, prec_tol_ppm left at {}",
                cfg.ms1_ppm_offset,
                cfg.prec_tol_ppm
            );
            return Ok(out);
        };
        let surface = cfg.ms1_calibrate.is_surface();
        let (offset, cal) = calibrate(&errors, surface, cfg.prec_tol_auto);
        if surface && cal.surface.is_none() {
            tracing::warn!(
                anchors = errors.len(),
                "extract: too few anchors for an MS1 error surface; one offset instead"
            );
        }
        let (mz_lo, mz_hi) = cal.surface.as_ref().map_or((0.0, 0.0), |s| span(&s.mz_ppm));
        let (rt_lo, rt_hi) = cal.surface.as_ref().map_or((0.0, 0.0), |s| span(&s.rt_ppm));
        tracing::info!(
            anchors = cal.n_anchors,
            mode = %cal.mode,
            raw_median_ppm = cal.raw_median_ppm,
            raw_sd_ppm = cal.raw_sd_ppm,
            offset_ppm = offset,
            offset_sd_ppm = cal.offset_sd_ppm,
            surface_sd_ppm = cal.surface_sd_ppm,
            mz_term_ppm = %format!("{mz_lo:.3}..{mz_hi:.3}"),
            rt_term_ppm = %format!("{rt_lo:.3}..{rt_hi:.3}"),
            prec_tol_ppm = cal.prec_tol_ppm,
            "extract: MS1 mass calibration learned from the seed (residuals as robust SD, ppm)"
        );
        out.ms1_ppm_offset = offset;
        if let Some(t) = cal.prec_tol_ppm {
            out.prec_tol_ppm = t;
        }
        out.ms1_cal = Some(cal);
        return Ok(out);
    }
    match estimate(seed_psms, ms1_path, q_max, SEARCH_PPM)? {
        Some((off, n)) => {
            tracing::info!(
                ms1_ppm_offset = off,
                anchors = n,
                "extract: MS1 mass offset learned from the seed"
            );
            out.ms1_ppm_offset = off;
        }
        None => tracing::warn!(
            "extract: too few seed anchors with an MS1 peak; MS1 offset left at {}",
            cfg.ms1_ppm_offset
        ),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::{write_table, Col};

    fn scratch(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("mumdia_ms1cal_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_string_lossy().to_string()
    }

    /// One MS1 scan per anchor RT holding, around each anchor's m/z, an intense interferent
    /// at +6 ppm in another mobility band and the precursor at +2 ppm at the anchor's 1/K0.
    /// The 3D rule takes the interferent; with `observed_im` (4D) the precursor is taken.
    #[test]
    fn on_4d_the_peak_nearest_the_seed_mobility_is_taken() {
        let n = 40usize;
        let mz: Vec<f64> = (0..n).map(|i| 500.0 + i as f64 * 7.0).collect();
        let rt: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 10.0).collect();
        let im: Vec<f64> = (0..n).map(|i| 0.8 + i as f64 * 0.005).collect();
        let at = |m: f64, ppm: f64| (m * (1.0 + ppm * 1e-6)) as f32;
        let ms1 = scratch("ms1.parquet");
        let mzs: Vec<Vec<f32>> = (0..n)
            .map(|i| vec![at(mz[i], 2.0), at(mz[i], 6.0)])
            .collect();
        let ints: Vec<Vec<f32>> = vec![vec![10.0, 1000.0]; n];
        let ims: Vec<Vec<f32>> = (0..n)
            .map(|i| vec![im[i] as f32, im[i] as f32 + 0.2])
            .collect();
        let ms1_cols = |with_im: bool| {
            let mut c = vec![
                Col::U32("scan_index".into(), (0..n as u32).collect()),
                Col::F64("rt_seconds".into(), rt.clone()),
                Col::LargeListF32("mz".into(), mzs.clone()),
                Col::LargeListF32("intensity".into(), ints.clone()),
            ];
            if with_im {
                c.push(Col::LargeListF32("im".into(), ims.clone()));
            }
            c
        };
        let seed = |path: &str, with_im: bool| {
            let mut c = vec![
                Col::Str("label".into(), vec!["target".to_string(); n]),
                Col::F64("spectrum_q".into(), vec![0.001; n]),
                Col::F64("precursor_mz".into(), mz.clone()),
                Col::F64("observed_rt".into(), rt.clone()),
            ];
            if with_im {
                c.push(Col::OptF64(
                    "observed_im".into(),
                    im.iter().map(|&v| Some(v)).collect(),
                ));
            }
            write_table(path, c).unwrap();
        };
        let (seed3, seed4) = (scratch("seed3.parquet"), scratch("seed4.parquet"));
        seed(&seed3, false);
        seed(&seed4, true);
        write_table(&ms1, ms1_cols(false)).unwrap();
        let (off3, n3) = estimate(&seed3, &ms1, 0.01, 15.0).unwrap().unwrap();
        assert_eq!(n3, n);
        assert!(
            (off3 - 6.0).abs() < 0.1,
            "3D takes the most intense: {off3}"
        );
        // A seed with 1/K0 against MS1 without it keeps the most-intense rule too.
        let (off, _) = estimate(&seed4, &ms1, 0.01, 15.0).unwrap().unwrap();
        assert_eq!(off.to_bits(), off3.to_bits());
        let ms1_4d = scratch("ms1_4d.parquet");
        write_table(&ms1_4d, ms1_cols(true)).unwrap();
        let (off4, n4) = estimate(&seed4, &ms1_4d, 0.01, 15.0).unwrap().unwrap();
        assert_eq!(n4, n);
        assert!(
            (off4 - 2.0).abs() < 0.1,
            "4D takes the 1/K0-nearest: {off4}"
        );
        // A 3D seed on 4D MS1: no 1/K0 to be near, the most-intense rule.
        let (off, _) = estimate(&seed3, &ms1_4d, 0.01, 15.0).unwrap().unwrap();
        assert_eq!(off.to_bits(), off3.to_bits());
    }

    /// Real-data check of a finished run: `MUMDIA_BENCH_MS1CAL=<run out dir> cargo test
    /// --release -- --ignored ms1cal_real --nocapture`. Prints the calibration of the run's
    /// seed anchors under the one offset and under the surface, with the tolerance
    /// `prec_tol_auto` would set, and the surface's residuals by m/z and retention time.
    #[test]
    #[ignore]
    fn ms1cal_real_run() {
        let Ok(run) = std::env::var("MUMDIA_BENCH_MS1CAL") else {
            return;
        };
        let errors = anchor_errors(
            &format!("{run}/seed_psms.parquet"),
            &format!("{run}/spectra/spectra_ms1.parquet"),
            0.01,
            SEARCH_PPM,
        )
        .unwrap()
        .expect("anchors");
        for surface in [false, true] {
            let (offset, cal) = calibrate(&errors, surface, true);
            println!(
                "offset {offset:.4} ppm; {}",
                serde_json::to_string(&cal).unwrap()
            );
        }
        let (_, cal) = calibrate(&errors, true, true);
        let s = cal.surface.unwrap();
        let mut by: Vec<(f64, f64, f64, f64)> = errors
            .iter()
            .map(|e| {
                (
                    e.mz,
                    e.rt,
                    e.ppm - percentile(&errors.iter().map(|a| a.ppm).collect::<Vec<_>>(), 0.5),
                    e.ppm - s.ppm_at(e.mz, e.rt),
                )
            })
            .collect();
        for (name, key) in [("m/z", 0usize), ("rt", 1usize)] {
            by.sort_by(|a, b| {
                if key == 0 {
                    a.0.total_cmp(&b.0)
                } else {
                    a.1.total_cmp(&b.1)
                }
            });
            let n = by.len();
            let line: Vec<String> = (0..8)
                .map(|b| {
                    let part = &by[b * n / 8..(b + 1) * n / 8];
                    let off: Vec<f64> = part.iter().map(|x| x.2).collect();
                    let sur: Vec<f64> = part.iter().map(|x| x.3).collect();
                    format!(
                        "[{:.0}] {:+.2}/{:+.2}",
                        if key == 0 { part[0].0 } else { part[0].1 },
                        median_sd(&off).0,
                        median_sd(&sur).0
                    )
                })
                .collect();
            println!(
                "median residual by {name} (offset/surface): {}",
                line.join(" ")
            );
        }
    }

    /// Anchors whose error drifts with m/z and retention time, with noise of 1 ppm and one
    /// in ten a random match anywhere in the 15 ppm window. The surface recovers the drift,
    /// shrinks the residuals against the one offset, and the tolerance follows the noise.
    #[test]
    fn the_surface_recovers_a_drift_in_mz_and_rt() {
        let truth = |mz: f64, rt: f64| 3.0 - 0.004 * (mz - 600.0) + 0.001 * (rt - 1500.0);
        // A deterministic pseudo-random sequence in [0, 1).
        let mut s = 12345u64;
        let mut u = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut errors = Vec::new();
        for _ in 0..4000 {
            let (mz, rt) = (400.0 + 500.0 * u(), 900.0 + 1400.0 * u());
            // Sum of four uniforms: mean 2, sd 1/sqrt(3); scaled to sd 1.
            let noise = (u() + u() + u() + u() - 2.0) * 3f64.sqrt();
            let ppm = if u() < 0.1 {
                -15.0 + 30.0 * u()
            } else {
                truth(mz, rt) + noise
            };
            errors.push(AnchorError { mz, rt, ppm });
        }
        let surf = fit_surface(&errors).expect("enough anchors");
        for &(mz, rt) in &[(450.0, 1000.0), (600.0, 1500.0), (850.0, 2200.0)] {
            let d = surf.ppm_at(mz, rt) - truth(mz, rt);
            assert!(d.abs() < 0.25, "{mz} {rt}: {d}");
        }
        let (offset, cal) = calibrate(&errors, true, true);
        assert_eq!(cal.mode, "surface");
        assert!((offset - 3.0).abs() < 0.3, "{offset}");
        let (off_sd, surf_sd) = (cal.offset_sd_ppm, cal.surface_sd_ppm.unwrap());
        assert!(surf_sd < 0.9 * off_sd, "{surf_sd} against {off_sd}");
        let tol = cal.prec_tol_ppm.unwrap();
        assert!((2.5..4.0).contains(&tol), "{tol}");
        // Without a surface the tolerance comes from the offset's residuals, which carry
        // the drift as well: wider.
        let (_, flat) = calibrate(&errors, false, true);
        assert_eq!(flat.mode, "offset");
        assert!(flat.surface.is_none() && flat.prec_tol_ppm.unwrap() > tol);
        // Too few anchors: no surface.
        assert!(fit_surface(&errors[..100]).is_none());
        assert_eq!(tolerance_from(0.1), 1.0);
        assert_eq!(tolerance_from(9.0), 15.0);
    }
}
