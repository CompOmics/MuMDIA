//! MS1 precursor mass offset (`extract.ms1_calibrate`).
//!
//! The fragment offset is learned by the seed (`masscal`), the MS1 one was not: on the
//! single-cell and HeLa Astral runs the MS1 mono peaks sit +1.9 to +2.35 ppm from the
//! theoretical m/z, so a +-5 ppm MS1 read is centred off the signal and reaches further into
//! interfering peaks on the other side. The offset is the median signed ppm error of the
//! most intense MS1 peak within `search_ppm` of each confident seed anchor's precursor m/z,
//! in the MS1 scan nearest its observed RT and the scans either side.
use anyhow::Result;
use mumdia_io::table::TableFile;

use crate::calibrate::percentile;
use crate::spectra::load_ms1;

/// Fewest anchors that still give a usable median.
const MIN_ANCHORS: usize = 30;

/// `(offset_ppm, anchors used)`, or `None` with fewer than [`MIN_ANCHORS`] anchors.
pub fn estimate(
    seed_psms: &str,
    ms1_path: &str,
    q_max: f64,
    search_ppm: f64,
) -> Result<Option<(f64, usize)>> {
    let seed = TableFile::open(seed_psms)?;
    let label = seed.str("label")?;
    let q = seed.f64("spectrum_q")?;
    let mz = seed.f64("precursor_mz")?;
    let rt = seed.f64("observed_rt")?;
    let anchors: Vec<(f64, f64)> = (0..seed.nrows)
        .filter(|&i| label[i] == "target" && q[i] < q_max && mz[i].is_finite() && rt[i].is_finite())
        .map(|i| (mz[i], rt[i]))
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
    for &(m, r) in &anchors {
        let j = rts.partition_point(|&x| x < r).min(rts.len() - 1);
        let j = if j > 0 && (r - rts[j - 1]).abs() < (rts[j] - r).abs() {
            j - 1
        } else {
            j
        };
        let (lo, hi) = (m * (1.0 - search_ppm * 1e-6), m * (1.0 + search_ppm * 1e-6));
        let mut best: Option<(f32, f64)> = None;
        for s in &ms1[j.saturating_sub(1)..=(j + 1).min(ms1.len() - 1)] {
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
            ppm.push((obs - m) / m * 1e6);
        }
    }
    if ppm.len() < MIN_ANCHORS {
        return Ok(None);
    }
    Ok(Some((percentile(&ppm, 0.5), ppm.len())))
}

/// `cfg` with `ms1_ppm_offset` learned from `seed_psms` when `ms1_calibrate` is set; `cfg`
/// unchanged otherwise, or when too few anchors find an MS1 peak (logged).
pub fn extract_cfg(
    cfg: &mumdia_core::config::ExtractConfig,
    seed_psms: &str,
    ms1_path: &str,
    q_max: f64,
) -> Result<mumdia_core::config::ExtractConfig> {
    let mut out = cfg.clone();
    if !cfg.ms1_calibrate {
        return Ok(out);
    }
    match estimate(seed_psms, ms1_path, q_max, 15.0)? {
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
