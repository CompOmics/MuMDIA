//! Shared spectrum reading: load the normalized spectra artifacts (Stage 0
//! output) back into memory for the seed search and extractor. Downstream
//! stages consume this artifact set, never the raw vendor file (docs/04_convert.md).
//!
//! Both loaders stream: the scalar columns are decoded one column at a time and the
//! two peak-list columns one batch of scans at a time, so nothing but the scans being
//! built is resident. The old `Table::read` path held the whole artifact as Arrow
//! batches (the peak lists twice: Arrow plus the `Vec<Peak>`s) until the function
//! returned.

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, ArrayRef, Float32Array, LargeListArray, ListArray};
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

/// The named peak-list column of one batch, whichever offset width it was written with
/// (see [`FloatList`]).
fn list_col<'a>(b: &'a RecordBatch, name: &str) -> Result<FloatList<'a>> {
    let i = b
        .schema()
        .index_of(name)
        .map_err(|_| anyhow!("column '{name}' not found"))?;
    FloatList::new(b.column(i).as_ref(), name)
}

fn inner_f32<'a>(v: &'a ArrayRef, name: &str) -> Result<&'a Float32Array> {
    v.as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| anyhow!("list '{name}' inner is not f32"))
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
    out.sort_by(|a, b| a.rt_seconds.total_cmp(&b.rt_seconds));
    let peak_bytes: usize = out
        .iter()
        .map(|s| std::mem::size_of_val(s.peaks.as_slice()))
        .sum();
    crate::memlog::report(
        "ms2 scans",
        &[
            ("peaks", peak_bytes),
            ("scan_spine", std::mem::size_of_val(out.as_slice())),
        ],
    );
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
        Ok(())
    })?;
    out.sort_by(|a, b| a.rt_seconds.total_cmp(&b.rt_seconds));
    let mz_bytes: usize = out
        .iter()
        .map(|s| std::mem::size_of_val(s.mz.as_slice()))
        .sum();
    let int_bytes: usize = out
        .iter()
        .map(|s| std::mem::size_of_val(s.intensity.as_slice()))
        .sum();
    crate::memlog::report(
        "ms1 scans",
        &[
            ("mz", mz_bytes),
            ("intensity", int_bytes),
            ("scan_spine", std::mem::size_of_val(out.as_slice())),
        ],
    );
    Ok(out)
}
