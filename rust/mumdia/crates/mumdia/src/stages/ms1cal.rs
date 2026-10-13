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
use anyhow::Result;
use mumdia_io::table::TableFile;

use crate::calibrate::percentile;
use crate::spectra::load_ms1;

/// Fewest anchors that still give a usable median.
const MIN_ANCHORS: usize = 30;

/// `(offset_ppm, anchors used)`, or `None` with fewer than `MIN_ANCHORS` anchors.
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
                ppm.push((obs - m) / m * 1e6);
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
}
