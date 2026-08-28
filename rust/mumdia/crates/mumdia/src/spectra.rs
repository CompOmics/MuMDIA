//! Shared spectrum reading: load the normalized spectra artifacts (Stage 0
//! output) back into memory for the seed search and extractor. Downstream
//! stages consume this artifact set, never the raw vendor file (docs/04_convert.md).

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, ArrayRef, Float32Array, LargeListArray, ListArray};
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
/// A float list column, whichever offset width it was written with.
///
/// `convert` writes the peak columns as `LargeListF32` (64-bit offsets), because a
/// 32-bit arrow list offset saturates above 2^31-1 total values and the builder then
/// unwraps a `None` deep inside arrow. A reader that downcasts only to `ListArray`
/// therefore fails with `column 'mz' is not a list` -- which is exactly how this was
/// found, by CI, after the writer moved and the reader did not.
///
/// Accepting both is the right shape regardless: spectra artifacts written by an
/// earlier version carry `List`, and there is no reason to refuse them.
enum FloatList<'a> {
    Small(&'a ListArray),
    Large(&'a LargeListArray),
}

impl<'a> FloatList<'a> {
    fn new(col: &'a dyn Array, name: &str) -> Result<FloatList<'a>> {
        if let Some(a) = col.as_any().downcast_ref::<ListArray>() {
            return Ok(FloatList::Small(a));
        }
        if let Some(a) = col.as_any().downcast_ref::<LargeListArray>() {
            return Ok(FloatList::Large(a));
        }
        Err(anyhow!(
            "column '{name}' is not a float list (arrow type {:?}); expected List or              LargeList of f32",
            col.data_type()
        ))
    }

    fn len(&self) -> usize {
        match self {
            FloatList::Small(a) => a.len(),
            FloatList::Large(a) => a.len(),
        }
    }

    fn is_null(&self, k: usize) -> bool {
        match self {
            FloatList::Small(a) => a.is_null(k),
            FloatList::Large(a) => a.is_null(k),
        }
    }

    fn value(&self, k: usize) -> ArrayRef {
        match self {
            FloatList::Small(a) => a.value(k),
            FloatList::Large(a) => a.value(k),
        }
    }
}

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
    let mz_i = t
        .schema
        .index_of("mz")
        .map_err(|_| anyhow!("column 'mz' not found"))?;
    let in_i = t
        .schema
        .index_of("intensity")
        .map_err(|_| anyhow!("column 'intensity' not found"))?;
    let mut out = Vec::with_capacity(t.nrows);
    let mut i = 0usize;
    for b in &t.batches {
        let mza = FloatList::new(b.column(mz_i).as_ref(), "mz")?;
        let ina = FloatList::new(b.column(in_i).as_ref(), "intensity")?;
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
    let mz_i = t
        .schema
        .index_of("mz")
        .map_err(|_| anyhow!("column 'mz' not found"))?;
    let in_i = t
        .schema
        .index_of("intensity")
        .map_err(|_| anyhow!("column 'intensity' not found"))?;
    let mut out = Vec::with_capacity(t.nrows);
    let mut i = 0usize;
    for b in &t.batches {
        let mza = FloatList::new(b.column(mz_i).as_ref(), "mz")?;
        let ina = FloatList::new(b.column(in_i).as_ref(), "intensity")?;
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
            // Truncate to the shorter list, as `load_ms2` does at :68. The two list
            // columns are decoded independently and either can be null, so a spectra
            // artifact whose m/z and intensity lists disagree in length would otherwise
            // be carried into `sum_near` and the MS1 isotope features, where the loop
            // bound comes from one array and the body indexes the other.
            let n = mz.len().min(intensity.len());
            out.push(Ms1Scan {
                scan_index: scan_index[i],
                rt_seconds: rt[i],
                mz: mz[..n].to_vec(),
                intensity: intensity[..n].to_vec(),
            });
            i += 1;
        }
    }
    out.sort_by(|a, b| a.rt_seconds.partial_cmp(&b.rt_seconds).unwrap());
    Ok(out)
}
