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
use arrow::array::Array;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use mumdia_core::types::{IsolationWindow, Ms2Scan, Peak};
use mumdia_io::table::{ListF32, TableFile};
use rayon::prelude::*;

/// Most parts a spectra table is decoded in at once ([`TableFile::row_parts`]).
///
/// Every concurrent part decoder costs working set of its own: its page and dictionary
/// buffers, and the allocator's per-thread pages for the peak lists it builds, which stay
/// with the scans for the whole stage. Measured on the AIF MS2 table (465,806 scans, 39.6M
/// peaks) inside a 16-thread seed: 1 part 534-575 ms decode and 1,098-1,108 MB seed peak,
/// 4 parts 256-274 ms and 1,137-1,179 MB, 8 parts 171 ms and 1,239 MB, 16 parts 123-140
/// ms and 1,334-1,341 MB, so about 14 MB per part. Eight keeps most of the speed-up at a
/// bounded cost on any host; the thread count alone would put it at 1.8 GB on 128 threads.
/// The extract stage's peak did not move (its own parallel allocations reuse those pages).
const DECODE_PARTS_MAX: usize = 8;

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

/// The named peak-list column of one batch as a borrowed view ([`ListF32`]), whichever
/// offset width it was written with.
///
/// `convert` writes the peak columns as `LargeListF32` (64-bit offsets), because a
/// 32-bit arrow list offset saturates above 2^31-1 total values and the builder then
/// unwraps a `None` deep inside arrow. A reader that downcasts only to `ListArray`
/// therefore fails with `column 'mz' is not a list` -- which is exactly how this was
/// found, by CI, after the writer moved and the reader did not. Accepting both is the
/// right shape regardless: spectra artifacts written by an earlier version carry `List`,
/// and there is no reason to refuse them.
///
/// The view resolves the offsets and the child values once per batch; reaching a row
/// through `ListArray::value(k)` built an `Arc`'d `ArrayRef` per row per column and threw
/// it away after the copy (2x per row, measured in mumdia-io's
/// `bench_list_view_against_the_per_row_arrayref`).
fn list_col<'a>(b: &'a RecordBatch, name: &str) -> Result<ListF32<'a>> {
    let i = b
        .schema()
        .index_of(name)
        .map_err(|_| anyhow!("column '{name}' not found"))?;
    let col = b.column(i);
    if !matches!(col.data_type(), DataType::List(_) | DataType::LargeList(_)) {
        return Err(anyhow!(
            "column '{name}' is not a float list (arrow type {:?}); expected List or \
             LargeList of f32",
            col.data_type()
        ));
    }
    ListF32::of(col, name)
}

/// Load MS2 scans (spectra_ms2.parquet) into memory, RT-sorted.
///
/// Decoded in parallel: the table is cut into row-contiguous parts
/// ([`TableFile::row_parts`]; `convert` writes an offset index, so even its one row group
/// splits at page granularity), each part decodes its own rows, and the parts are
/// concatenated in order before the (stable) retention-time sort. Row order before the
/// sort is therefore file order, as in the serial decode, and the result is the same
/// `Vec` bit for bit (`the_parallel_ms2_decode_is_the_serial_decode`).
pub fn load_ms2(path: &str) -> Result<Vec<Ms2Scan>> {
    let t0 = std::time::Instant::now();
    let t = TableFile::open(path).with_context(|| format!("loading ms2 {path}"))?;
    let parts = t.row_parts(rayon::current_num_threads().min(DECODE_PARTS_MAX))?;
    let decoded: Vec<Result<Vec<Ms2Scan>>> = parts.par_iter().map(decode_ms2).collect();
    let mut out = Vec::with_capacity(t.nrows);
    // The first error in part order is the one the serial decode would have hit first.
    for d in decoded {
        out.extend(d?);
    }
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
    // The decode's own wall time. The stage logs bracket it together with the library load
    // and the index build, so without this line no log could say which of the three a seed
    // or an extract spent its load phase on.
    tracing::info!(
        ms2 = path,
        scans = out.len(),
        peaks = peak_bytes / std::mem::size_of::<Peak>(),
        parts = parts.len(),
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "spectra: decoded MS2"
    );
    Ok(out)
}

/// The MS2 scans of one part, in file order.
fn decode_ms2(t: &TableFile) -> Result<Vec<Ms2Scan>> {
    let scan_index = t.u32("scan_index")?;
    let rt = t.f64("rt_seconds")?;
    let wlo = t.f64("window_lower")?;
    let whi = t.f64("window_upper")?;
    let wtarget = t.f64("window_target")?;
    // The "id" column is NOT read. It stays in the artifact, where an external consumer can
    // find the mzML native id of any scan by `scan_index`, but decoding it here built one
    // String per scan (~72 B of header plus payload each, ~17 MB and 233 k allocations on
    // an AIF run) that no stage ever looked at.
    //
    // Build the per-scan peak lists in a single pass over the "mz"/"intensity" lists,
    // without the intermediate Vec<Vec<f32>> that list_f32() would materialize. m/z is
    // copied at the artifact's own f32 width and widened by the consumers at the
    // comparison, which is exact; the widening used to happen here and cost 8 B per peak
    // for nothing. The row counter `i` tracks the same batch-then-row order the scalar
    // getters used, so it stays aligned.
    let mut out = Vec::with_capacity(t.nrows);
    let mut i = 0usize;
    t.for_each_batch(Some(&["mz", "intensity"]), SCAN_BATCH_ROWS, |b| {
        let mza = list_col(b, "mz")?;
        let ina = list_col(b, "intensity")?;
        for k in 0..mza.len() {
            // A null list is an empty slice, so a scan with either list null is empty.
            let (mf, iff) = (mza.row_slice(k, "mz")?, ina.row_slice(k, "intensity")?);
            let n = mf.len().min(iff.len());
            let peaks: Vec<Peak> = mf[..n]
                .iter()
                .zip(&iff[..n])
                .map(|(&mz, &intensity)| Peak { mz, intensity })
                .collect();
            out.push(Ms2Scan {
                scan_index: scan_index[i],
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
    Ok(out)
}

/// Load MS1 scans (spectra_ms1.parquet), RT-sorted. Decoded in parallel parts exactly as
/// [`load_ms2`] is.
pub fn load_ms1(path: &str) -> Result<Vec<Ms1Scan>> {
    let t0 = std::time::Instant::now();
    let t = TableFile::open(path).with_context(|| format!("loading ms1 {path}"))?;
    let parts = t.row_parts(rayon::current_num_threads().min(DECODE_PARTS_MAX))?;
    let decoded: Vec<Result<Vec<Ms1Scan>>> = parts.par_iter().map(decode_ms1).collect();
    let mut out = Vec::with_capacity(t.nrows);
    for d in decoded {
        out.extend(d?);
    }
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
    tracing::info!(
        ms1 = path,
        scans = out.len(),
        peaks = mz_bytes / std::mem::size_of::<f32>(),
        parts = parts.len(),
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "spectra: decoded MS1"
    );
    Ok(out)
}

/// The MS1 scans of one part, in file order.
fn decode_ms1(t: &TableFile) -> Result<Vec<Ms1Scan>> {
    let scan_index = t.u32("scan_index")?;
    let rt = t.f64("rt_seconds")?;
    // mz and intensity are copied straight out of each row's slice of the batch's values
    // (no widening, no intermediate Vec<Vec<f32>>).
    let mut out = Vec::with_capacity(t.nrows);
    let mut i = 0usize;
    t.for_each_batch(Some(&["mz", "intensity"]), SCAN_BATCH_ROWS, |b| {
        let mza = list_col(b, "mz")?;
        let ina = list_col(b, "intensity")?;
        for k in 0..mza.len() {
            // Truncate to the shorter list, as `load_ms2` does. The two list columns are
            // decoded independently and either can be null, so a spectra artifact whose
            // m/z and intensity lists disagree in length would otherwise be carried into
            // `sum_near` and the MS1 isotope features, where the loop bound comes from one
            // array and the body indexes the other. A null list is an empty slice.
            //
            // The length is taken from the decoded slices before either is copied: each
            // list was previously copied whole and then copied again truncated, which on a
            // run with MS1 scans of tens of thousands of peaks is two allocations per scan
            // per column that are freed immediately.
            let (mf, iff) = (mza.row_slice(k, "mz")?, ina.row_slice(k, "intensity")?);
            let n = mf.len().min(iff.len());
            out.push(Ms1Scan {
                scan_index: scan_index[i],
                rt_seconds: rt[i],
                mz: mf[..n].to_vec(),
                intensity: iff[..n].to_vec(),
            });
            i += 1;
        }
        Ok(())
    })?;
    Ok(out)
}

/// The serial decoders the parallel ones replaced, kept verbatim as the reference the tests
/// compare against scan for scan, bit for bit.
#[cfg(test)]
mod serial {
    use super::*;
    use arrow::array::{ArrayRef, Float32Array, LargeListArray, ListArray};

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
            Err(anyhow!("column '{name}' is not a float list"))
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

    fn col<'a>(b: &'a RecordBatch, name: &str) -> Result<FloatList<'a>> {
        let i = b.schema().index_of(name)?;
        FloatList::new(b.column(i).as_ref(), name)
    }

    fn inner_f32<'a>(v: &'a ArrayRef, name: &str) -> Result<&'a Float32Array> {
        v.as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow!("list '{name}' inner is not f32"))
    }

    pub fn load_ms2(path: &str) -> Result<Vec<Ms2Scan>> {
        let t = TableFile::open(path)?;
        let scan_index = t.u32("scan_index")?;
        let rt = t.f64("rt_seconds")?;
        let wlo = t.f64("window_lower")?;
        let whi = t.f64("window_upper")?;
        let wtarget = t.f64("window_target")?;
        let mut out = Vec::with_capacity(t.nrows);
        let mut i = 0usize;
        t.for_each_batch(Some(&["mz", "intensity"]), SCAN_BATCH_ROWS, |b| {
            let mza = col(b, "mz")?;
            let ina = col(b, "intensity")?;
            for k in 0..mza.len() {
                let peaks: Vec<Peak> = if mza.is_null(k) || ina.is_null(k) {
                    Vec::new()
                } else {
                    let mv = mza.value(k);
                    let mf = inner_f32(&mv, "mz")?;
                    let iv = ina.value(k);
                    let iff = inner_f32(&iv, "intensity")?;
                    let n = mf.len().min(iff.len());
                    (0..n)
                        .map(|j| Peak {
                            mz: mf.value(j),
                            intensity: iff.value(j),
                        })
                        .collect()
                };
                out.push(Ms2Scan {
                    scan_index: scan_index[i],
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
        Ok(out)
    }

    pub fn load_ms1(path: &str) -> Result<Vec<Ms1Scan>> {
        let t = TableFile::open(path)?;
        let scan_index = t.u32("scan_index")?;
        let rt = t.f64("rt_seconds")?;
        let mut out = Vec::with_capacity(t.nrows);
        let mut i = 0usize;
        t.for_each_batch(Some(&["mz", "intensity"]), SCAN_BATCH_ROWS, |b| {
            let mza = col(b, "mz")?;
            let ina = col(b, "intensity")?;
            for k in 0..mza.len() {
                let mv = (!mza.is_null(k)).then(|| mza.value(k));
                let iv = (!ina.is_null(k)).then(|| ina.value(k));
                let mf = mv.as_ref().map(|v| inner_f32(v, "mz")).transpose()?;
                let iff = iv.as_ref().map(|v| inner_f32(v, "intensity")).transpose()?;
                let n = match (&mf, &iff) {
                    (Some(m), Some(x)) => m.len().min(x.len()),
                    _ => 0,
                };
                out.push(Ms1Scan {
                    scan_index: scan_index[i],
                    rt_seconds: rt[i],
                    mz: mf.map(|m| m.values()[..n].to_vec()).unwrap_or_default(),
                    intensity: iff.map(|x| x.values()[..n].to_vec()).unwrap_or_default(),
                });
                i += 1;
            }
            Ok(())
        })?;
        out.sort_by(|a, b| a.rt_seconds.total_cmp(&b.rt_seconds));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::{write_table, Col};

    /// A spectra table shaped like `convert`'s (LargeList peak columns), with the awkward
    /// rows the loaders must keep: null lists, lists of unequal length, empty lists, and
    /// retention times out of order with ties, written in `row_group` rows per group and
    /// small data pages, with or without an offset index. `list32` writes 32-bit list
    /// offsets, as an older convert did.
    fn write_spectra(path: &str, n: usize, row_group: usize, offset_index: bool, list32: bool) {
        use arrow::array::{
            Float32Builder, Float64Array, LargeListBuilder, ListBuilder, UInt32Array,
        };
        use arrow::datatypes::Field;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::{EnabledStatistics, WriterProperties};
        use std::sync::Arc;
        let rt: Vec<f64> = (0..n).map(|i| ((i * 7919) % 97) as f64 * 1.5).collect();
        let fill = |i: usize, which: usize| -> Option<Vec<f32>> {
            if i % 29 == 3 && which == 0 || i % 31 == 5 && which == 1 {
                return None;
            }
            let len = (i * 13 + which) % 9 + usize::from(i.is_multiple_of(17) && which == 1);
            Some(
                (0..len)
                    .map(|k| 100.0 + i as f32 * 0.37 + k as f32 * 11.0)
                    .collect(),
            )
        };
        let list_field = Arc::new(Field::new_list_field(DataType::Float32, true));
        let lt = if list32 {
            DataType::List(list_field.clone())
        } else {
            DataType::LargeList(list_field.clone())
        };
        let mut lists: Vec<arrow::array::ArrayRef> = Vec::new();
        for which in 0..2 {
            if list32 {
                let mut b = ListBuilder::new(Float32Builder::new());
                for i in 0..n {
                    match fill(i, which) {
                        Some(v) => {
                            b.values().append_slice(&v);
                            b.append(true);
                        }
                        None => b.append(false),
                    }
                }
                lists.push(Arc::new(b.finish()));
            } else {
                let mut b = LargeListBuilder::new(Float32Builder::new());
                for i in 0..n {
                    match fill(i, which) {
                        Some(v) => {
                            b.values().append_slice(&v);
                            b.append(true);
                        }
                        None => b.append(false),
                    }
                }
                lists.push(Arc::new(b.finish()));
            }
        }
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            Field::new("scan_index", DataType::UInt32, false),
            Field::new("rt_seconds", DataType::Float64, false),
            Field::new("window_lower", DataType::Float64, false),
            Field::new("window_upper", DataType::Float64, false),
            Field::new("window_target", DataType::Float64, false),
            Field::new("mz", lt.clone(), true),
            Field::new("intensity", lt, true),
        ]));
        let win = |i: usize| 400.0 + (i % 8) as f64 * 25.0;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from((0..n as u32).collect::<Vec<_>>())),
                Arc::new(Float64Array::from(rt)),
                Arc::new(Float64Array::from((0..n).map(win).collect::<Vec<_>>())),
                Arc::new(Float64Array::from(
                    (0..n).map(|i| win(i) + 25.0).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    (0..n).map(|i| win(i) + 12.5).collect::<Vec<_>>(),
                )),
                lists[0].clone(),
                lists[1].clone(),
            ],
        )
        .unwrap();
        let mut props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(row_group))
            .set_data_page_row_count_limit(23)
            .set_write_batch_size(23);
        if !offset_index {
            props = props
                .set_statistics_enabled(EnabledStatistics::Chunk)
                .set_offset_index_disabled(true);
        }
        let mut w = ArrowWriter::try_new(
            std::fs::File::create(path).unwrap(),
            schema,
            Some(props.build()),
        )
        .unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    /// A scan as bits: index, RT, window (target, lower, upper), peaks.
    type ScanBits = (u32, u64, [u64; 3], Vec<(u32, u32)>);

    fn ms2_bits(v: &[Ms2Scan]) -> Vec<ScanBits> {
        v.iter()
            .map(|s| {
                (
                    s.scan_index,
                    s.rt_seconds.to_bits(),
                    [
                        s.window.target_mz.to_bits(),
                        s.window.lower_mz.to_bits(),
                        s.window.upper_mz.to_bits(),
                    ],
                    s.peaks
                        .iter()
                        .map(|p| (p.mz.to_bits(), p.intensity.to_bits()))
                        .collect(),
                )
            })
            .collect()
    }

    fn ms1_bits(v: &[Ms1Scan]) -> Vec<(u32, u64, Vec<u32>, Vec<u32>)> {
        v.iter()
            .map(|s| {
                (
                    s.scan_index,
                    s.rt_seconds.to_bits(),
                    s.mz.iter().map(|x| x.to_bits()).collect(),
                    s.intensity.iter().map(|x| x.to_bits()).collect(),
                )
            })
            .collect()
    }

    /// The parallel decode (row parts, the borrowed list view) returns exactly the serial
    /// decode's scans, in the same order, on many row groups, on one row group split at
    /// page granularity through its offset index, on one group without an index, and with
    /// 32-bit list offsets; the peak lists keep their nulls-as-empty and shorter-list
    /// rules and the RT sort keeps its tie order.
    #[test]
    fn the_parallel_ms2_decode_is_the_serial_decode() {
        let dir = std::env::temp_dir().join(format!("mumdia_spectra_par_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = 3_000usize;
        for (tag, rg, oi, l32) in [
            ("many", 211usize, true, false),
            ("one_indexed", n, true, false),
            ("one_flat", n, false, false),
            ("list32", 500, true, true),
        ] {
            let p = dir
                .join(format!("{tag}.parquet"))
                .to_str()
                .unwrap()
                .to_string();
            write_spectra(&p, n, rg, oi, l32);
            let want2 = serial::load_ms2(&p).unwrap();
            let want1 = serial::load_ms1(&p).unwrap();
            assert!(
                want2.iter().any(|s| s.peaks.is_empty()) && want2.len() == n,
                "{tag}: the fixture must hold empty scans"
            );
            for threads in [1usize, 3, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let (got2, got1) = pool.install(|| (load_ms2(&p).unwrap(), load_ms1(&p).unwrap()));
                assert_eq!(ms2_bits(&got2), ms2_bits(&want2), "{tag} ms2 @ {threads}");
                assert_eq!(ms1_bits(&got1), ms1_bits(&want1), "{tag} ms1 @ {threads}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The MS1 loader truncates a scan to the shorter of its two peak lists and keeps the
    /// scans in retention-time order. Both are contracts the isotope features depend on:
    /// they take the loop bound from one array and index the other.
    #[test]
    fn ms1_scans_are_rt_sorted_and_truncated_to_the_shorter_list() {
        let dir = std::env::temp_dir().join(format!("mumdia_ms1_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ms1.parquet").to_str().unwrap().to_string();
        write_table(
            &path,
            vec![
                Col::U32("scan_index".into(), vec![7, 3]),
                Col::F64("rt_seconds".into(), vec![20.0, 10.0]),
                // The second scan carries one more intensity than it has m/z values.
                Col::ListF32("mz".into(), vec![vec![100.0, 200.0], vec![300.0]]),
                Col::ListF32("intensity".into(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
            ],
        )
        .unwrap();
        let scans = load_ms1(&path).unwrap();
        assert_eq!(scans.len(), 2);
        assert_eq!(scans[0].scan_index, 3, "sorted by retention time");
        assert_eq!(scans[0].mz, vec![300.0]);
        assert_eq!(scans[0].intensity, vec![3.0], "truncated to the m/z count");
        assert_eq!(scans[1].mz, vec![100.0, 200.0]);
        assert_eq!(scans[1].intensity, vec![1.0, 2.0]);
    }

    /// The MS2 loader hands each peak m/z on at the artifact's own f32 width, and a
    /// consumer's `peak.mz as f64` is the same f64 the loader used to store in the peak.
    ///
    /// This pins the OLD behaviour of the narrowed field. The loader ran
    /// `mz: mf.value(j) as f64` and every consumer read that f64; it now runs
    /// `mz: mf.value(j)` and every consumer says `as f64`. The two differ only if
    /// `f32 -> f64` were lossy, so the assertion is written as the old expression against
    /// the new one, over values chosen to be awkward at f32: a decimal that is not
    /// representable, a value at the top of the fragment m/z range where f32 spacing is
    /// ~1.2e-4, and the subnormal boundary.
    #[test]
    fn ms2_peak_mz_is_the_artifact_f32_and_widens_to_the_f64_the_loader_used_to_store() {
        let dir = std::env::temp_dir().join(format!("mumdia_ms2_mz_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ms2.parquet").to_str().unwrap().to_string();
        let mzs: Vec<f32> = vec![0.1, 133.107_1, 1_999.999_9, f32::MIN_POSITIVE, 700.325_44];
        write_table(
            &path,
            vec![
                Col::U32("scan_index".into(), vec![1]),
                Col::Str("id".into(), vec!["controllerType=0 scan=1".into()]),
                Col::F64("rt_seconds".into(), vec![12.5]),
                Col::F64("window_lower".into(), vec![400.0]),
                Col::F64("window_upper".into(), vec![410.0]),
                Col::F64("window_target".into(), vec![405.0]),
                Col::ListF32("mz".into(), vec![mzs.clone()]),
                Col::ListF32("intensity".into(), vec![vec![1.0; mzs.len()]]),
            ],
        )
        .unwrap();
        let scans = load_ms2(&path).unwrap();
        assert_eq!(scans.len(), 1);
        assert_eq!(scans[0].scan_index, 1);
        assert_eq!(scans[0].peaks.len(), mzs.len());
        for (p, &m) in scans[0].peaks.iter().zip(&mzs) {
            assert_eq!(p.mz.to_bits(), m.to_bits(), "stored f32 was not preserved");
            // `m as f64` is exactly what the loader wrote into the old `mz: f64` field.
            assert_eq!(
                (p.mz as f64).to_bits(),
                (m as f64).to_bits(),
                "widening at the consumer does not reproduce the old stored f64"
            );
        }
        // The artifact still carries the native id; the in-memory scan no longer does.
        assert!(TableFile::open(&path).unwrap().str("id").is_ok());
    }
}
