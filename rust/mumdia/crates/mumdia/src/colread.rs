//! Column-parallel, pipelined decoding of a parquet handle.
//!
//! A table read through one multi-column reader decodes its columns one after another on
//! one thread, and the stage consuming the batches waits for each. [`for_each_zipped`]
//! gives every column its own reader and decodes the next batch of all of them in parallel
//! WHILE the caller processes the current one, so the wall time of a load becomes roughly
//! the slowest column's decode rather than the sum of all of them plus the processing.
//!
//! It uses `rayon::join` and nothing that blocks: no channel, no dedicated thread. On a
//! one-thread pool the two halves simply run one after the other, which is the serial
//! read, and it cannot deadlock however deeply it is nested inside other rayon work.
//!
//! The batches the caller sees are exactly the multi-column reader's: every column reader
//! uses the same batch size over the same rows, so batch `k` of each column covers the
//! same rows, and that is checked rather than assumed.

use anyhow::{bail, Result};
use arrow::array::ArrayRef;
use mumdia_io::table::{BatchReader, TableFile};
use rayon::prelude::*;

/// Call `f(first_row, columns)` for every batch of `cols` (in the order given, one
/// `ArrayRef` per name) over the rows of `t`, where `first_row` counts the handle's rows.
/// Columns named in `dict` are read through their dictionary
/// ([`TableFile::batches_dict`]). The first error in row order is returned: `f`'s error
/// for batch `k` wins over a decode error of batch `k + 1`, as in a serial read.
pub fn for_each_zipped<F>(
    t: &TableFile,
    cols: &[&str],
    dict: &[&str],
    batch_rows: usize,
    mut f: F,
) -> Result<()>
where
    F: FnMut(usize, &[ArrayRef]) -> Result<()> + Send,
{
    let mut readers: Vec<BatchReader> = cols
        .iter()
        .map(|c| t.batches_dict(Some(&[*c]), batch_rows, dict))
        .collect::<Result<_>>()?;
    let mut next = step(&mut readers, cols)?;
    let mut row = 0usize;
    while let Some(cur) = next {
        let n = cur[0].len();
        let (done, ahead) = rayon::join(|| f(row, &cur), || step(&mut readers, cols));
        done?;
        next = ahead?;
        row += n;
    }
    Ok(())
}

/// The next batch of every column, decoded in parallel; `None` when all are exhausted.
fn step(readers: &mut [BatchReader], cols: &[&str]) -> Result<Option<Vec<ArrayRef>>> {
    let got: Vec<Option<Result<arrow::record_batch::RecordBatch>>> =
        readers.par_iter_mut().map(|r| r.next()).collect();
    let mut out: Vec<ArrayRef> = Vec::with_capacity(got.len());
    let mut ended = 0usize;
    for g in got {
        match g {
            None => ended += 1,
            Some(b) => out.push(b?.column(0).clone()),
        }
    }
    if ended == readers.len() {
        return Ok(None);
    }
    if ended > 0 {
        bail!("columns {cols:?} of one table ended at different rows");
    }
    let n = out[0].len();
    if out.iter().any(|a| a.len() != n) {
        bail!("columns {cols:?} of one table decoded batches of different lengths");
    }
    Ok(Some(out))
}

/// The first error of `results`, in order, or all the values.
///
/// A parallel collect into `Result<Vec<_>>` returns whichever error a worker hit first in
/// time. A load that reports the first bad ROW needs the first error in part order, which
/// is the first bad row in file order, because every part stops at its own first error.
pub fn first_err<T>(results: Vec<Result<T>>) -> Result<Vec<T>> {
    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::{Col, TableWriter};

    /// The zipped read sees the same rows, in the same batches, as one multi-column reader,
    /// and reports row bases that count the handle's rows, on a whole file and on a span.
    #[test]
    fn zipped_batches_are_the_multi_column_batches() {
        let dir = std::env::temp_dir().join(format!("mumdia_colread_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.parquet").to_str().unwrap().to_string();
        let n = 10_007usize;
        let mut w = TableWriter::new(&p).with_row_group_rows(1_000);
        w.write_cols(vec![
            Col::U32("a".into(), (0..n as u32).collect()),
            Col::F64("b".into(), (0..n).map(|i| i as f64 * 0.5).collect()),
            Col::Str("c".into(), (0..n).map(|i| format!("n{}", i % 7)).collect()),
        ])
        .unwrap();
        w.close().unwrap();
        let t = TableFile::open(&p).unwrap();
        for handle in [t.span(0, n).unwrap(), t.span(333, 5_000).unwrap()] {
            let mut seen_a = Vec::new();
            let mut seen_b = Vec::new();
            let mut seen_c = Vec::new();
            let mut bases = Vec::new();
            for_each_zipped(&handle, &["a", "b", "c"], &["c"], 512, |row, cols| {
                bases.push(row);
                let a = cols[0]
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .unwrap();
                let b = cols[1]
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();
                seen_a.extend_from_slice(a.values());
                seen_b.extend_from_slice(b.values());
                let view = mumdia_io::table::StrBatch::of(&cols[2]).unwrap();
                assert!(matches!(view, mumdia_io::table::StrBatch::Dict { .. }));
                let mut it = mumdia_io::table::StrInterner::new();
                it.begin(&view);
                for k in 0..a.len() {
                    let id = it.row(&view, k).unwrap();
                    seen_c.push(it.values()[id as usize].clone());
                }
                Ok(())
            })
            .unwrap();
            assert_eq!(seen_a, handle.u32("a").unwrap());
            assert_eq!(seen_b, handle.f64("b").unwrap());
            assert_eq!(seen_c, handle.str("c").unwrap());
            let want_bases: Vec<usize> = (0..handle.nrows).step_by(512).collect();
            assert_eq!(
                bases, want_bases,
                "one batch per 512 rows, bases in handle rows"
            );
        }
        // An error from the callback is returned, and stops the read.
        let mut calls = 0;
        let err = for_each_zipped(&t, &["a"], &[], 512, |row, _| {
            calls += 1;
            if row >= 1024 {
                bail!("stop at {row}");
            }
            Ok(())
        })
        .unwrap_err();
        assert_eq!(err.to_string(), "stop at 1024");
        assert_eq!(calls, 3);
        // A missing column is refused up front.
        assert!(for_each_zipped(&t, &["a", "zz"], &[], 512, |_, _| Ok(())).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
