//! Shared spectrum reading: load the normalized spectra artifacts (Stage 0
//! output) back into memory for the seed search and extractor. Downstream
//! stages consume this artifact set, never the raw vendor file (PLAN.md Stage 0).

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, Float32Array, ListArray};
use mumdia_core::types::{IsolationWindow, Ms2Scan, Peak};
use mumdia_io::table::Table;

/// An MS1 scan with centroided peaks.
#[derive(Clone, Debug)]
pub struct Ms1Scan {
    pub scan_index: u32,
    pub rt_seconds: f64,
    pub mz: Vec<f64>,
    pub intensity: Vec<f32>,
}

/// Load MS2 scans (spectra_ms2.parquet) into memory, RT-sorted.
pub fn load_ms2(path: &str) -> Result<Vec<Ms2Scan>> {
    let t = Table::read(path).with_context(|| format!("loading ms2 {path}"))?;
    let scan_index = t.u32("scan_index")?;
    let id = t.str("id")?;
    let rt = t.f64("rt_seconds")?;
    let wlo = t.f64("window_lower")?;
    let whi = t.f64("window_upper")?;
    let wtarget = t.f64("window_target")?;
    // Build the per-scan peak lists in a single pass over the "mz"/"intensity"
    // ListArrays, downcasting each row's inner Float32Array once, without the
    // intermediate Vec<Vec<f32>> that list_f32() would materialize. m/z is stored
    // as f32 for size and widened to f64 here. The global row counter `i` tracks
    // the same batch-then-row order the scalar getters used, so it stays aligned.
    let mz_i = t.schema.index_of("mz").map_err(|_| anyhow!("column 'mz' not found"))?;
    let in_i = t
        .schema
        .index_of("intensity")
        .map_err(|_| anyhow!("column 'intensity' not found"))?;
    let mut out = Vec::with_capacity(t.nrows);
    let mut i = 0usize;
    for b in &t.batches {
        let mza = b
            .column(mz_i)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("column 'mz' is not a list"))?;
        let ina = b
            .column(in_i)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("column 'intensity' is not a list"))?;
        for k in 0..mza.len() {
            let peaks: Vec<Peak> = if mza.is_null(k) || ina.is_null(k) {
                Vec::new()
            } else {
                let mv = mza.value(k);
                let mf = mv
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("list 'mz' inner is not f32"))?;
                let iv = ina.value(k);
                let iff = iv
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("list 'intensity' inner is not f32"))?;
                let n = mf.len().min(iff.len());
                let mut peaks = Vec::with_capacity(n);
                for j in 0..n {
                    peaks.push(Peak {
                        mz: mf.value(j) as f64,
                        intensity: iff.value(j),
                        ion_mobility: None,
                    });
                }
                peaks
            };
            out.push(Ms2Scan {
                scan_index: scan_index[i],
                id: id[i].clone(),
                rt_seconds: rt[i],
                window: IsolationWindow {
                    target_mz: wtarget[i],
                    lower_mz: wlo[i],
                    upper_mz: whi[i],
                    im_lower: None,
                    im_upper: None,
                },
                peaks,
            });
            i += 1;
        }
    }
    out.sort_by(|a, b| a.rt_seconds.partial_cmp(&b.rt_seconds).unwrap());
    Ok(out)
}

/// Load MS1 scans (spectra_ms1.parquet), RT-sorted.
pub fn load_ms1(path: &str) -> Result<Vec<Ms1Scan>> {
    let t = Table::read(path).with_context(|| format!("loading ms1 {path}"))?;
    let scan_index = t.u32("scan_index")?;
    let rt = t.f64("rt_seconds")?;
    // Build mz (widened to f64) and intensity in a single pass over the ListArrays,
    // downcasting each row's inner Float32Array once, without the intermediate
    // Vec<Vec<f32>> (and the extra intensity clone) that list_f32() would incur.
    let mz_i = t.schema.index_of("mz").map_err(|_| anyhow!("column 'mz' not found"))?;
    let in_i = t
        .schema
        .index_of("intensity")
        .map_err(|_| anyhow!("column 'intensity' not found"))?;
    let mut out = Vec::with_capacity(t.nrows);
    let mut i = 0usize;
    for b in &t.batches {
        let mza = b
            .column(mz_i)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("column 'mz' is not a list"))?;
        let ina = b
            .column(in_i)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("column 'intensity' is not a list"))?;
        for k in 0..mza.len() {
            let mz: Vec<f64> = if mza.is_null(k) {
                Vec::new()
            } else {
                let mv = mza.value(k);
                let mf = mv
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("list 'mz' inner is not f32"))?;
                (0..mf.len()).map(|j| mf.value(j) as f64).collect()
            };
            let intensity: Vec<f32> = if ina.is_null(k) {
                Vec::new()
            } else {
                let iv = ina.value(k);
                let iff = iv
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("list 'intensity' inner is not f32"))?;
                (0..iff.len()).map(|j| iff.value(j)).collect()
            };
            out.push(Ms1Scan {
                scan_index: scan_index[i],
                rt_seconds: rt[i],
                mz,
                intensity,
            });
            i += 1;
        }
    }
    out.sort_by(|a, b| a.rt_seconds.partial_cmp(&b.rt_seconds).unwrap());
    Ok(out)
}
