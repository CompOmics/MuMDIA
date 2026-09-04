//! Shared spectrum reading: load the normalized spectra artifacts (Stage 0
//! output) back into memory for the seed search and extractor. Downstream
//! stages consume this artifact set, never the raw vendor file (PLAN.md Stage 0).
//!
//! Both loaders stream: the scalar columns are decoded one column at a time and the
//! two peak-list columns one batch of scans at a time, so nothing but the scans being
//! built is resident. The old `Table::read` path held the whole artifact as Arrow
//! batches (the peak lists twice: Arrow plus the `Vec<Peak>`s) until the function
//! returned.

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, ArrayRef, Float32Array, ListArray};
use arrow::record_batch::RecordBatch;
use mumdia_core::types::{IsolationWindow, Ms2Scan, Peak};
use mumdia_io::table::TableFile;

/// Scans per decoded batch of the peak-list columns. Scan rows are long lists (thousands
/// of peaks), so keep this small: 1024 scans x ~2,000 peaks x 4 B is ~8 MB per column.
const SCAN_BATCH_ROWS: usize = 1024;

/// An MS1 scan with centroided peaks.
#[derive(Clone, Debug)]
pub struct Ms1Scan {
    pub scan_index: u32,
    pub rt_seconds: f64,
    /// Peak m/z, f32 exactly as the spectra artifact stores it. Consumers widen to f64 at
    /// the comparison (`extract::sum_near`), which yields the very values the previous
    /// `Vec<f64>` copy held, at two-thirds of the footprint.
    pub mz: Vec<f32>,
    pub intensity: Vec<f32>,
}

fn list_col<'a>(b: &'a RecordBatch, name: &str) -> Result<&'a ListArray> {
    let i = b
        .schema()
        .index_of(name)
        .map_err(|_| anyhow!("column '{name}' not found"))?;
    b.column(i)
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("column '{name}' is not a list"))
}

fn inner_f32<'a>(v: &'a ArrayRef, name: &str) -> Result<&'a Float32Array> {
    v.as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| anyhow!("list '{name}' inner is not f32"))
}

/// Load MS2 scans (spectra_ms2.parquet) into memory, RT-sorted.
pub fn load_ms2(path: &str) -> Result<Vec<Ms2Scan>> {
    let t = TableFile::open(path).with_context(|| format!("loading ms2 {path}"))?;
    let scan_index = t.u32("scan_index")?;
    let mut id = t.str("id")?;
    let rt = t.f64("rt_seconds")?;
    let wlo = t.f64("window_lower")?;
    let whi = t.f64("window_upper")?;
    let wtarget = t.f64("window_target")?;
    // Build the per-scan peak lists in a single pass over the "mz"/"intensity"
    // ListArrays, downcasting each row's inner Float32Array once, without the
    // intermediate Vec<Vec<f32>> that list_f32() would materialize. m/z is stored
    // as f32 for size and widened to f64 here. The global row counter `i` tracks
    // the same batch-then-row order the scalar getters used, so it stays aligned.
    let mut out = Vec::with_capacity(t.nrows);
    let mut i = 0usize;
    t.for_each_batch(Some(&["mz", "intensity"]), SCAN_BATCH_ROWS, |b| {
        let mza = list_col(b, "mz")?;
        let ina = list_col(b, "intensity")?;
        for k in 0..mza.len() {
            let peaks: Vec<Peak> = if mza.is_null(k) || ina.is_null(k) {
                Vec::new()
            } else {
                let mv = mza.value(k);
                let mf = inner_f32(&mv, "mz")?;
                let iv = ina.value(k);
                let iff = inner_f32(&iv, "intensity")?;
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
                id: std::mem::take(&mut id[i]),
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
        Ok(())
    })?;
    out.sort_by(|a, b| a.rt_seconds.partial_cmp(&b.rt_seconds).unwrap());
    Ok(out)
}

/// Load MS1 scans (spectra_ms1.parquet), RT-sorted.
pub fn load_ms1(path: &str) -> Result<Vec<Ms1Scan>> {
    let t = TableFile::open(path).with_context(|| format!("loading ms1 {path}"))?;
    let scan_index = t.u32("scan_index")?;
    let rt = t.f64("rt_seconds")?;
    // mz and intensity are copied straight out of each row's inner Float32Array (no
    // widening, no intermediate Vec<Vec<f32>>).
    let mut out = Vec::with_capacity(t.nrows);
    let mut i = 0usize;
    t.for_each_batch(Some(&["mz", "intensity"]), SCAN_BATCH_ROWS, |b| {
        let mza = list_col(b, "mz")?;
        let ina = list_col(b, "intensity")?;
        for k in 0..mza.len() {
            let mz: Vec<f32> = if mza.is_null(k) {
                Vec::new()
            } else {
                let mv = mza.value(k);
                inner_f32(&mv, "mz")?.values().to_vec()
            };
            let intensity: Vec<f32> = if ina.is_null(k) {
                Vec::new()
            } else {
                let iv = ina.value(k);
                inner_f32(&iv, "intensity")?.values().to_vec()
            };
            out.push(Ms1Scan {
                scan_index: scan_index[i],
                rt_seconds: rt[i],
                mz,
                intensity,
            });
            i += 1;
        }
        Ok(())
    })?;
    out.sort_by(|a, b| a.rt_seconds.partial_cmp(&b.rt_seconds).unwrap());
    Ok(out)
}
