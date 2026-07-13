//! Stage 0 `mumdia convert`: read an mzML run into the normalized spectra
//! artifact set (PLAN.md Stage 0). MVP is mzML-only and 3D, so ion-mobility
//! columns are absent. Profile spectra are centroided (simple local-maxima)
//! so downstream matching sees discrete peaks.

use std::time::Instant;

use anyhow::{Context, Result};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col};
use mzdata::prelude::*;
use mzdata::spectrum::SignalContinuity;
use serde_json::json;
use tracing::info;

/// Centroid a profile spectrum by local maxima with 3-point parabolic m/z
/// refinement. Peaks below `noise_floor` (relative to the max) are dropped.
fn centroid(mz: &[f64], inten: &[f32]) -> (Vec<f64>, Vec<f32>) {
    let n = mz.len();
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
fn peaks_of<S: SpectrumLike>(spec: &S, top_n: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut mz, mut inten): (Vec<f64>, Vec<f32>) = match spec.raw_arrays() {
        Some(arrays) => {
            let m = arrays.mzs().map(|c| c.to_vec()).unwrap_or_default();
            let it = arrays.intensities().map(|c| c.to_vec()).unwrap_or_default();
            (m, it)
        }
        None => (Vec::new(), Vec::new()),
    };
    if spec.signal_continuity() == SignalContinuity::Profile {
        let (cm, ci) = centroid(&mz, &inten);
        mz = cm;
        inten = ci;
    }
    // Drop zero/negative intensity.
    let mut pairs: Vec<(f64, f32)> = mz
        .into_iter()
        .zip(inten)
        .filter(|(_, i)| *i > 0.0)
        .collect();
    if top_n > 0 && pairs.len() > top_n {
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        pairs.truncate(top_n);
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let out_mz: Vec<f32> = pairs.iter().map(|(m, _)| *m as f32).collect();
    let out_in: Vec<f32> = pairs.iter().map(|(_, i)| *i).collect();
    (out_mz, out_in)
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

pub fn run(p: ConvertParams) -> Result<ConvertOutputs> {
    let t0 = Instant::now();
    std::fs::create_dir_all(p.out_dir).ok();
    info!(mzml = p.mzml, "convert: opening mzML");
    let reader = mzdata::MZReader::open_path(p.mzml).with_context(|| format!("open {}", p.mzml))?;

    // MS1 accumulators
    let (mut ms1_idx, mut ms1_rt) = (Vec::new(), Vec::new());
    let (mut ms1_mz, mut ms1_in): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
    // MS2 accumulators
    let (mut m2_idx, mut m2_id, mut m2_rt) = (Vec::new(), Vec::new(), Vec::new());
    let (mut m2_wt, mut m2_wl, mut m2_wu) = (Vec::new(), Vec::new(), Vec::new());
    let (mut m2_pmz, mut m2_pz): (Vec<Option<f64>>, Vec<Option<i32>>) = (Vec::new(), Vec::new());
    let (mut m2_mz, mut m2_in): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
    let (mut map_ms2, mut map_ms1) = (Vec::new(), Vec::new());

    let mut last_ms1_index: Option<u32> = None;
    let mut count = 0usize;
    let mut idx: u32 = 0;

    for spec in reader {
        if p.max_spectra > 0 && count >= p.max_spectra {
            break;
        }
        count += 1;
        let scan_index = idx;
        idx += 1;
        let rt_s = spec.start_time() * 60.0; // mzdata returns minutes
        match spec.ms_level() {
            1 => {
                let (mz, inten) = peaks_of(&spec, p.top_peaks_ms1);
                ms1_idx.push(scan_index);
                ms1_rt.push(rt_s);
                ms1_mz.push(mz);
                ms1_in.push(inten);
                last_ms1_index = Some(scan_index);
            }
            2 => {
                let (mz, inten) = peaks_of(&spec, p.top_peaks_ms2);
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
                m2_idx.push(scan_index);
                m2_id.push(spec.id().to_string());
                m2_rt.push(rt_s);
                m2_wt.push(wt);
                m2_wl.push(wl);
                m2_wu.push(wu);
                m2_pmz.push(pmz);
                m2_pz.push(pz);
                m2_mz.push(mz);
                m2_in.push(inten);
                map_ms2.push(scan_index);
                map_ms1.push(last_ms1_index.map(|x| x as i32).unwrap_or(-1));
            }
            _ => {}
        }
    }

    let ms1_path = format!("{}/spectra_ms1.parquet", p.out_dir);
    let ms2_path = format!("{}/spectra_ms2.parquet", p.out_dir);
    let iw_path = format!("{}/isolation_windows.parquet", p.out_dir);
    let map_path = format!("{}/ms2_to_ms1.parquet", p.out_dir);

    let n_ms1 = write_table(
        &ms1_path,
        vec![
            Col::U32("scan_index".into(), ms1_idx),
            Col::F64("rt_seconds".into(), ms1_rt),
            Col::ListF32("mz".into(), ms1_mz),
            Col::ListF32("intensity".into(), ms1_in),
        ],
    )?;

    // Distinct isolation windows (id, target, lower, upper).
    let mut uniq: Vec<(u64, f64, f64, f64)> = Vec::new();
    let mut win_id_col = Vec::with_capacity(m2_idx.len());
    {
        use std::collections::HashMap;
        let mut seen: HashMap<(u64, u64), u32> = HashMap::new();
        for i in 0..m2_idx.len() {
            let key = (m2_wl[i].to_bits(), m2_wu[i].to_bits());
            let id = *seen.entry(key).or_insert_with(|| {
                let id = uniq.len() as u32;
                uniq.push((id as u64, m2_wt[i], m2_wl[i], m2_wu[i]));
                id
            });
            win_id_col.push(id);
        }
    }

    let n_ms2 = write_table(
        &ms2_path,
        vec![
            Col::U32("scan_index".into(), m2_idx),
            Col::Str("id".into(), m2_id),
            Col::F64("rt_seconds".into(), m2_rt),
            Col::U32("window_id".into(), win_id_col),
            Col::F64("window_target".into(), m2_wt),
            Col::F64("window_lower".into(), m2_wl),
            Col::F64("window_upper".into(), m2_wu),
            Col::OptF64("precursor_mz".into(), m2_pmz),
            Col::OptI32("precursor_charge".into(), m2_pz),
            Col::ListF32("mz".into(), m2_mz),
            Col::ListF32("intensity".into(), m2_in),
        ],
    )?;

    let n_iw = write_table(
        &iw_path,
        vec![
            Col::U32("window_id".into(), uniq.iter().map(|w| w.0 as u32).collect()),
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
        p.config_hash,
        elapsed,
        json!({"mzml": p.mzml, "max_spectra": p.max_spectra}),
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
    config_hash: &str,
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
