//! The chromatogram table's two on-disk layouts, and the encoder and decoder between them
//! (docs/15_data_dictionary.md, `chromatograms.parquet`).
//!
//! Both layouts hold the same rows in the same order: one per predicted fragment of an
//! accepted candidate, then, in window-grid mode, the candidate's three MS1 isotope XICs.
//! Candidates are ascending and each candidate's rows are contiguous. The layouts differ
//! only in how a row's two traces are stored.
//!
//! **v1** (`extract.chromatogram_schema = 1`, the default) stores each trace whole. `rt` is
//! the row's retention-time axis and `intensity` its values on that axis, the same length,
//! and both are empty for a predicted fragment that was never observed. In window-grid mode
//! every observed row of a candidate repeats the same axis, and a fragment's trace is zero
//! over most of the candidate's RT window. On the AIF run of docs/15 ("Layout v2"), the
//! `rt` column held 10.8 times the values that one axis per candidate needs, and 57% of
//! the `intensity` values lay outside their trace's nonzero run.
//!
//! **v2** (`extract.chromatogram_schema = 2`) adds two `u32` columns and changes what the
//! two list columns hold:
//!
//! * `trace_len` is the length of the row's full trace, and 0 for a fragment that was never
//!   observed. `trace_offset` is the position, in that full trace, of the first stored
//!   intensity.
//! * `intensity` holds the full trace from its first to its last value that is not `+0.0`.
//!   The comparison is on the bit pattern, so a `-0.0` or a NaN is stored as it is. Every
//!   value outside the stored run is `+0.0`, and a trace that is `+0.0` throughout stores
//!   no value at all (`trace_len` still says how long it is).
//! * `rt` holds the row's axis only where a reader cannot know it already. A row with a
//!   non-empty trace writes its axis, unless the last row that wrote an axis in the same
//!   parquet row group belongs to the same candidate and wrote a bit-identical axis. Then
//!   the row writes an empty `rt`, and its axis is that one. In window-grid mode this is one
//!   axis per candidate per row group. In sparse mode (`extract.emit_window_grid = false`)
//!   each fragment has an axis of its own, so nearly every row writes one.
//!
//! The rule restarts at every row group, so **each row group can be read on its own**. That
//! is what makes the layout safe at the seams the readers cut. Quant reads the table row
//! group by row group, the pool splices whole row groups, and the features main pass starts
//! every chunk at a candidate's first row. Only the features confident-bounds pass starts
//! inside a candidate, because its sub-chunks follow an absolute row grid; it first reads
//! the rows between the start of the row group and its own first row through
//! [`Decoder::skip`] (`features::ChromStream::open_at`).
//!
//! [`Decoder::row`] turns every v2 row back into its v1 row: the same axis values and the
//! same full trace, bit for bit. Every stage downstream of extract builds its in-memory
//! store from those rows, so it writes the same bytes from either layout.

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, StringArray, UInt32Array};
use arrow::record_batch::RecordBatch;
use mumdia_io::table::{Col, ListF32, TableFile, TableWriter};

/// The v2 column holding where a row's stored intensities start in its full trace.
pub const TRACE_OFFSET: &str = "trace_offset";
/// The v2 column holding the length of a row's full trace (0: never observed).
pub const TRACE_LEN: &str = "trace_len";

/// Rows per parquet row group of a chromatogram table written by extract: about 64k rows of
/// two ~60-point traces, ~30 MB uncompressed in v1, which bounds the encoder's in-progress
/// buffer. The v2 axis rule restarts at these boundaries, so the writer and
/// [`Encoder`] must agree on it; both take it from here.
pub const ROW_GROUP_ROWS: usize = 1 << 16;

/// Test knob: rows per chromatogram row group, instead of [`ROW_GROUP_ROWS`]. It moves the
/// row-group seams of the chromatogram table and nothing else, so every downstream table
/// must come out byte-identical whatever it is set to. `ci/smoke.sh` sets it to 1 to put a
/// seam at every row of a v2 table. Unset, empty, 0 or unparsable means the default.
pub const ROW_GROUP_ROWS_ENV: &str = "MUMDIA_CHROM_ROW_GROUP_ROWS";

/// The rows per row group extract writes: [`ROW_GROUP_ROWS`], or [`ROW_GROUP_ROWS_ENV`].
pub fn row_group_rows() -> usize {
    // The literal rather than `ROW_GROUP_ROWS_ENV`, so the generated configuration
    // reference (docs/24) lists the variable by name.
    std::env::var("MUMDIA_CHROM_ROW_GROUP_ROWS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(ROW_GROUP_ROWS)
}

/// Which of the two layouts a chromatogram table has (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    V1,
    V2,
}

impl Layout {
    /// The layout `extract.chromatogram_schema` names.
    pub fn from_schema_version(version: u32) -> Result<Layout> {
        match version {
            1 => Ok(Layout::V1),
            2 => Ok(Layout::V2),
            v => bail!("chromatogram schema {v} does not exist; use 1 or 2"),
        }
    }

    /// The schema version recorded for a table in this layout.
    pub fn version(self) -> u32 {
        match self {
            Layout::V1 => 1,
            Layout::V2 => 2,
        }
    }

    /// The layout of an open table, from its columns: v2 carries both trace columns and v1
    /// neither. One without the other is an error rather than a guess.
    pub fn of(tf: &TableFile) -> Result<Layout> {
        match (tf.has_column(TRACE_OFFSET), tf.has_column(TRACE_LEN)) {
            (false, false) => Ok(Layout::V1),
            (true, true) => Ok(Layout::V2),
            _ => bail!(
                "{} has one of the chromatogram v2 columns '{TRACE_OFFSET}' and \
                 '{TRACE_LEN}' but not the other; it is neither layout",
                tf.path()
            ),
        }
    }

    /// The columns a reader projects in addition to the v1 ones.
    pub fn trace_columns(self) -> &'static [&'static str] {
        match self {
            Layout::V1 => &[],
            Layout::V2 => &[TRACE_OFFSET, TRACE_LEN],
        }
    }
}

/// The two trace columns of one decoded batch of a v2 table.
pub struct TraceCols<'a> {
    offset: &'a UInt32Array,
    len: &'a UInt32Array,
}

impl<'a> TraceCols<'a> {
    /// The trace columns of `b`, by name. Both must be `u32` without nulls: a null slot
    /// would be read as whatever value lies behind the validity bitmap.
    pub fn of(b: &'a RecordBatch) -> Result<TraceCols<'a>> {
        let col = |name: &str| -> Result<&'a UInt32Array> {
            let i = b
                .schema()
                .index_of(name)
                .map_err(|_| anyhow!("chromatogram v2 batch has no column '{name}'"))?;
            let a = b
                .column(i)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| anyhow!("chromatogram column '{name}' is not u32"))?;
            if a.null_count() > 0 {
                bail!("chromatogram column '{name}' holds a null");
            }
            Ok(a)
        };
        Ok(TraceCols {
            offset: col(TRACE_OFFSET)?,
            len: col(TRACE_LEN)?,
        })
    }

    #[inline]
    pub fn offset(&self, k: usize) -> u32 {
        self.offset.value(k)
    }

    #[inline]
    pub fn len(&self, k: usize) -> u32 {
        self.len.value(k)
    }
}

/// Whether two axes are the same bit patterns. `==` would call `-0.0` and `0.0` equal and a
/// NaN unequal to itself; the v2 reader hands back the stored bits, so the writer may only
/// drop an axis that is bit-identical to the one it refers to.
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The run `lo..hi` of `v` from its first to its last value that is not `+0.0`, by bit
/// pattern; `(0, 0)` when there is none.
fn stored_run(v: &[f32]) -> (usize, usize) {
    match v.iter().position(|x| x.to_bits() != 0) {
        None => (0, 0),
        Some(lo) => {
            let hi = v.iter().rposition(|x| x.to_bits() != 0).map_or(lo, |i| i) + 1;
            (lo, hi)
        }
    }
}

/// One row as the v2 layout stores it.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedRow {
    pub rt: Vec<f32>,
    pub intensity: Vec<f32>,
    pub trace_offset: u32,
    pub trace_len: u32,
}

/// Encodes a table's rows, in table order, into the v2 layout. It counts the rows it is
/// given, so it knows where each row group starts; every row of the table must pass
/// through it exactly once, in the order it is written, and the writer must cut row groups
/// every `row_group_rows` rows (what [`TableWriter::with_row_group_rows`] does).
pub struct Encoder {
    row_group_rows: u64,
    row: u64,
    /// The candidate whose axis was the last one written in this row group, and that axis.
    open: Option<u32>,
    axis: Vec<f32>,
}

impl Encoder {
    pub fn new(row_group_rows: usize) -> Encoder {
        Encoder {
            row_group_rows: row_group_rows.max(1) as u64,
            row: 0,
            open: None,
            axis: Vec::new(),
        }
    }

    /// Rows encoded so far.
    pub fn rows(&self) -> u64 {
        self.row
    }

    /// Encode the next row: candidate `cid`, with its v1 axis and trace.
    pub fn encode(&mut self, cid: u32, rt: Vec<f32>, intensity: Vec<f32>) -> Result<EncodedRow> {
        if self.row.is_multiple_of(self.row_group_rows) {
            // A new row group: nothing written before it may be referred to.
            self.open = None;
        }
        self.row += 1;
        if rt.len() != intensity.len() {
            bail!(
                "chromatogram row for candidate_id {cid} has {} retention-time points but {} \
                 intensity points; the v2 layout stores one trace length for both",
                rt.len(),
                intensity.len()
            );
        }
        let n = rt.len();
        let trace_len = u32::try_from(n).map_err(|_| {
            anyhow!("chromatogram row for candidate_id {cid} has {n} points, above u32")
        })?;
        if n == 0 {
            return Ok(EncodedRow {
                rt,
                intensity,
                trace_offset: 0,
                trace_len: 0,
            });
        }
        let rt = if self.open == Some(cid) && same_bits(&self.axis, &rt) {
            Vec::new()
        } else {
            self.open = Some(cid);
            self.axis.clear();
            self.axis.extend_from_slice(&rt);
            rt
        };
        let (lo, hi) = stored_run(&intensity);
        let intensity = if lo == 0 && hi == n {
            intensity
        } else {
            intensity[lo..hi].to_vec()
        };
        Ok(EncodedRow {
            rt,
            intensity,
            // `lo < n`, which fits: `n` did.
            trace_offset: lo as u32,
            trace_len,
        })
    }
}

/// Decodes the rows of a v2 table, in table order, back into their v1 form. One decoder per
/// read scope: it must see every row of the scope in order, from the start of a row group
/// or from a candidate's first row (module docs).
#[derive(Default)]
pub struct Decoder {
    /// The candidate of the last row that carried an axis in this scope, and that axis.
    open: Option<u32>,
    axis: Vec<f32>,
    /// The last trace rebuilt with its zero margins.
    dense: Vec<f32>,
}

impl Decoder {
    pub fn new() -> Decoder {
        Decoder::default()
    }

    /// Check one stored row and take its axis when it carries one.
    fn resolve(
        &mut self,
        cid: u32,
        rt: &[f32],
        intensity: &[f32],
        trace_offset: u32,
        trace_len: u32,
    ) -> Result<()> {
        let n = trace_len as usize;
        if n == 0 {
            if !rt.is_empty() || !intensity.is_empty() || trace_offset != 0 {
                bail!(
                    "chromatogram row for candidate_id {cid} has trace_len 0 (a fragment \
                     that was never observed) but {} retention-time and {} intensity values \
                     at offset {trace_offset}",
                    rt.len(),
                    intensity.len()
                );
            }
            return Ok(());
        }
        if !rt.is_empty() {
            if rt.len() != n {
                bail!(
                    "chromatogram row for candidate_id {cid} has {} retention-time points \
                     but a trace_len of {n}",
                    rt.len()
                );
            }
            self.open = Some(cid);
            self.axis.clear();
            self.axis.extend_from_slice(rt);
        } else if self.open != Some(cid) {
            bail!(
                "chromatogram row for candidate_id {cid} has a {n}-point trace but no \
                 retention-time axis, and no earlier row of the candidate in this read \
                 carried one; a v2 table must be read from the start of a row group or from \
                 a candidate's first row"
            );
        } else if self.axis.len() != n {
            bail!(
                "chromatogram row for candidate_id {cid} has a trace_len of {n} but the \
                 candidate's retention-time axis has {} points",
                self.axis.len()
            );
        }
        if (trace_offset as usize)
            .checked_add(intensity.len())
            .is_none_or(|end| end > n)
        {
            bail!(
                "chromatogram row for candidate_id {cid} stores {} intensity values at \
                 offset {trace_offset} of a {n}-point trace",
                intensity.len()
            );
        }
        Ok(())
    }

    /// The v1 form of one stored row: its axis and its full trace, the same values bit for
    /// bit as the v1 layout stores. Both slices borrow from the row or from the decoder, so
    /// they are valid until the next call.
    pub fn row<'a>(
        &'a mut self,
        cid: u32,
        rt: &'a [f32],
        intensity: &'a [f32],
        trace_offset: u32,
        trace_len: u32,
    ) -> Result<(&'a [f32], &'a [f32])> {
        self.resolve(cid, rt, intensity, trace_offset, trace_len)?;
        let n = trace_len as usize;
        if n == 0 {
            return Ok((&[], &[]));
        }
        let Decoder { axis, dense, .. } = self;
        let axis: &'a Vec<f32> = axis;
        let rt: &'a [f32] = if rt.is_empty() { axis } else { rt };
        let o = trace_offset as usize;
        if o == 0 && intensity.len() == n {
            return Ok((rt, intensity));
        }
        dense.clear();
        dense.resize(n, 0.0);
        dense[o..o + intensity.len()].copy_from_slice(intensity);
        Ok((rt, dense))
    }

    /// Check one stored row and follow its axis, without rebuilding its trace: for a row
    /// the caller does not keep, and for the rows before a read that starts inside a
    /// candidate.
    pub fn skip(
        &mut self,
        cid: u32,
        rt: &[f32],
        intensity: &[f32],
        trace_offset: u32,
        trace_len: u32,
    ) -> Result<()> {
        self.resolve(cid, rt, intensity, trace_offset, trace_len)
    }
}

/// Which of the columns no reader requires a chromatogram table has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Optional {
    pub frag_mz: bool,
    pub frag_obs_mz: bool,
}

impl Optional {
    /// Every column: what extract writes.
    pub const ALL: Optional = Optional {
        frag_mz: true,
        frag_obs_mz: true,
    };

    pub fn of(tf: &TableFile) -> Optional {
        Optional {
            frag_mz: tf.has_column("frag_mz"),
            frag_obs_mz: tf.has_column("frag_obs_mz"),
        }
    }
}

/// Chromatogram rows gathered into one chunk of columns, in either layout. Extract fills it
/// and so does [`rewrite`], so the column set and order are defined once.
#[derive(Default)]
pub struct Rows {
    pub cid: Vec<u32>,
    pub name: Vec<String>,
    pub frag_mz: Vec<f64>,
    pub frag_obs_mz: Vec<f64>,
    pub predicted_intensity: Vec<f32>,
    pub rt: Vec<Vec<f32>>,
    pub intensity: Vec<Vec<f32>>,
    /// v2 only.
    pub trace_offset: Vec<u32>,
    pub trace_len: Vec<u32>,
}

impl Rows {
    /// The chunk's traces in bytes, for the writer's memory report.
    pub fn trace_bytes(&self) -> usize {
        crate::memlog::bytes_of_nested(&self.rt) + crate::memlog::bytes_of_nested(&self.intensity)
    }

    /// The columns of a table in `layout`, with the optional ones `opt` names: every table
    /// extract writes has both, a table from before `frag_obs_mz` existed has no such
    /// column, and quant's test tables carry neither.
    pub fn into_cols(self, layout: Layout, opt: Optional) -> Vec<Col> {
        let mut cols = vec![
            Col::U32("candidate_id".into(), self.cid),
            Col::Str("frag_name".into(), self.name),
        ];
        if opt.frag_mz {
            cols.push(Col::F64("frag_mz".into(), self.frag_mz));
        }
        if opt.frag_obs_mz {
            cols.push(Col::F64("frag_obs_mz".into(), self.frag_obs_mz));
        }
        cols.extend([
            Col::F32("predicted_intensity".into(), self.predicted_intensity),
            // LargeList (64-bit offsets): the total chromatogram list-value count can exceed
            // the ~2.1B limit of a 32-bit ListArray offset buffer when extraction accepts a
            // very large candidate set (e.g. gates opened up).
            Col::LargeListF32("rt".into(), self.rt),
            Col::LargeListF32("intensity".into(), self.intensity),
        ]);
        if layout == Layout::V2 {
            cols.push(Col::U32(TRACE_OFFSET.into(), self.trace_offset));
            cols.push(Col::U32(TRACE_LEN.into(), self.trace_len));
        }
        cols
    }
}

/// A writer for a chromatogram table, laid out as extract lays it out: `row_group_rows`
/// rows per row group and the `rt` axis written PLAIN. Every fragment row of a v1 candidate
/// carries the same axis, and snappy shortens those repeated PLAIN runs far better than a
/// dictionary's bit-packed indices (AIF chromatograms 12.0% smaller; docs/03 "Float
/// encodings planned from the first rows"). In v2 the column holds about one axis per
/// candidate and mostly empty lists, and PLAIN is kept there as well: against a dictionary
/// `rt` it measured 2.7% smaller on an AIF run and 3.2% and 0.3% larger on an entrapment
/// and an Astral run (`v2_on_a_real_artifact`, docs/15_data_dictionary.md "Layout v2"),
/// which is no case for a second rule.
pub fn writer(path: &str, row_group_rows: usize) -> TableWriter {
    TableWriter::new(path)
        .with_row_group_rows(row_group_rows)
        .with_plain_column("rt")
}

/// One row of a chromatogram table in its v1 form, as [`for_each_row`] hands it over.
pub struct DenseRow<'a> {
    pub cid: u32,
    pub name: &'a str,
    /// `None` when the table has no such column.
    pub frag_mz: Option<f64>,
    pub frag_obs_mz: Option<f64>,
    pub predicted_intensity: f32,
    pub rt: &'a [f32],
    pub intensity: &'a [f32],
}

/// Call `f` with every row of `tf`, in table order and in its v1 form, whichever layout the
/// table has. `tf` is a whole file, or a span that starts at a row group or at a
/// candidate's first row.
pub fn for_each_row(tf: &TableFile, mut f: impl FnMut(&DenseRow) -> Result<()>) -> Result<()> {
    let layout = Layout::of(tf)?;
    let opt = Optional::of(tf);
    let mut cols = vec![
        "candidate_id",
        "frag_name",
        "predicted_intensity",
        "rt",
        "intensity",
    ];
    if opt.frag_mz {
        cols.push("frag_mz");
    }
    if opt.frag_obs_mz {
        cols.push("frag_obs_mz");
    }
    cols.extend_from_slice(layout.trace_columns());
    let mut dec = Decoder::new();
    for b in tf.batches(Some(&cols), 1 << 12)? {
        let b = b?;
        let col = |name: &str| -> Result<&ArrayRef> {
            let i = b
                .schema()
                .index_of(name)
                .map_err(|_| anyhow!("chromatograms batch has no column '{name}'"))?;
            Ok(b.column(i))
        };
        let cid = col("candidate_id")?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| anyhow!("chromatograms column 'candidate_id' is not u32"))?;
        let name = col("frag_name")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("chromatograms column 'frag_name' is not utf8"))?;
        let fmz = if opt.frag_mz {
            Some(
                col("frag_mz")?
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| anyhow!("chromatograms column 'frag_mz' is not f64"))?,
            )
        } else {
            None
        };
        let obsmz = if opt.frag_obs_mz {
            Some(
                col("frag_obs_mz")?
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| anyhow!("chromatograms column 'frag_obs_mz' is not f64"))?,
            )
        } else {
            None
        };
        let pint = col("predicted_intensity")?
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow!("chromatograms column 'predicted_intensity' is not f32"))?;
        let rt = ListF32::of(col("rt")?, "rt")?;
        let int = ListF32::of(col("intensity")?, "intensity")?;
        let trace = match layout {
            Layout::V1 => None,
            Layout::V2 => Some(TraceCols::of(&b)?),
        };
        for k in 0..b.num_rows() {
            let c = cid.value(k);
            let rt_k = rt.row_slice(k, "rt")?;
            let int_k = int.row_slice(k, "intensity")?;
            let (rt_k, int_k) = match &trace {
                None => (rt_k, int_k),
                Some(t) => dec.row(c, rt_k, int_k, t.offset(k), t.len(k))?,
            };
            f(&DenseRow {
                cid: c,
                name: if name.is_null(k) { "" } else { name.value(k) },
                frag_mz: fmz.map(|a| a.value(k)),
                frag_obs_mz: obsmz.map(|a| a.value(k)),
                predicted_intensity: pint.value(k),
                rt: rt_k,
                intensity: int_k,
            })?;
        }
    }
    Ok(())
}

/// Rewrite the chromatogram table `src` (either layout) as `out` in `layout`, with
/// `row_group_rows` rows per row group. The rows, their order and every value are kept; the
/// layout, and so the bytes, change. For converting an existing artifact, for readers that
/// only understand v1, and for the tests that compare the layouts at every row-group size.
pub fn rewrite(src: &str, out: &str, layout: Layout, row_group_rows: usize) -> Result<u64> {
    if std::path::Path::new(src) == std::path::Path::new(out) {
        bail!("chromatograms rewrite: the output {out} is the input");
    }
    rewrite_into(src, layout, row_group_rows, writer(out, row_group_rows))
}

/// [`rewrite`] into a writer of the caller's, which must cut row groups every
/// `row_group_rows` rows.
fn rewrite_into(
    src: &str,
    layout: Layout,
    row_group_rows: usize,
    mut w: TableWriter,
) -> Result<u64> {
    let tf = TableFile::open(src)?;
    let opt = Optional::of(&tf);
    let mut enc = Encoder::new(row_group_rows);
    let mut rows = Rows::default();
    // Flushed on the row-group grid, which keeps the chunks bounded; the writer cuts the
    // row groups itself.
    let chunk = row_group_rows.clamp(1, ROW_GROUP_ROWS);
    for_each_row(&tf, |r| {
        rows.cid.push(r.cid);
        rows.name.push(r.name.to_string());
        rows.frag_mz.push(r.frag_mz.unwrap_or(0.0));
        rows.frag_obs_mz.push(r.frag_obs_mz.unwrap_or(0.0));
        rows.predicted_intensity.push(r.predicted_intensity);
        match layout {
            Layout::V1 => {
                rows.rt.push(r.rt.to_vec());
                rows.intensity.push(r.intensity.to_vec());
            }
            Layout::V2 => {
                let e = enc.encode(r.cid, r.rt.to_vec(), r.intensity.to_vec())?;
                rows.rt.push(e.rt);
                rows.intensity.push(e.intensity);
                rows.trace_offset.push(e.trace_offset);
                rows.trace_len.push(e.trace_len);
            }
        }
        if rows.cid.len() >= chunk {
            w.write_cols(std::mem::take(&mut rows).into_cols(layout, opt))?;
        }
        Ok(())
    })
    .with_context(|| format!("rewriting {src}"))?;
    // Also the chunk that fixes the schema when the source has no row.
    w.write_cols(rows.into_cols(layout, opt))?;
    w.close()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("mumdia_chromatograms_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_string_lossy().to_string()
    }

    /// Rows that reach every branch of the encoder: shared grid axes, an absent fragment
    /// first and in the middle, all-zero traces, a trace with no zero margin, `-0.0` and NaN
    /// inside and at the edges, a candidate whose rows change axis part-way (sparse mode),
    /// an axis that differs from the grid only in the sign of a zero, and a one-row
    /// candidate.
    fn rows_fixture() -> Vec<(u32, Vec<f32>, Vec<f32>)> {
        let grid = |n: usize, t0: f32| (0..n).map(|k| t0 + k as f32).collect::<Vec<f32>>();
        let mut v = vec![
            // Candidate 3: absent first, then grid rows, then MS1-like rows on the same grid.
            (3, vec![], vec![]),
            (3, grid(6, 10.0), vec![0.0, 0.0, 5.0, 7.0, 0.0, 0.0]),
            (3, grid(6, 10.0), vec![1.0, 0.0, 5.0, 7.0, 0.0, 2.0]),
            (3, vec![], vec![]),
            (3, grid(6, 10.0), vec![0.0; 6]),
            (3, grid(6, 10.0), vec![-0.0, 0.0, 0.0, 0.0, 0.0, f32::NAN]),
            (3, grid(6, 10.0), vec![0.0, 0.0, 0.0, 0.0, 0.0, 3.0]),
            // Candidate 4: sparse mode, a different axis on most rows, one repeated, and one
            // that differs from the one before only in the sign of a zero.
            (4, vec![1.0, 3.0], vec![2.0, 0.0]),
            (4, vec![0.0, 4.0, 9.0], vec![0.0, 1.5, 0.0]),
            (4, vec![0.0, 4.0, 9.0], vec![0.0, 0.0, 0.0]),
            (4, vec![-0.0, 4.0, 9.0], vec![0.0, 0.0, 8.0]),
            // Candidate 7: a single row.
            (7, grid(3, 50.0), vec![0.5, 0.25, 0.125]),
            // Candidate 9: only absent rows.
            (9, vec![], vec![]),
            (9, vec![], vec![]),
        ];
        // Candidate 12: a long grid.
        let g = grid(40, 100.0);
        for f in 0..5 {
            let t: Vec<f32> = (0..40)
                .map(|k| {
                    if (10 + f..20 + f).contains(&k) {
                        (k * (f + 1)) as f32
                    } else {
                        0.0
                    }
                })
                .collect();
            v.push((12, g.clone(), t));
        }
        v
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Encode `rows` at `rg` rows per row group, then decode every row group on its own.
    fn round_trip(rows: &[(u32, Vec<f32>, Vec<f32>)], rg: usize) -> Vec<EncodedRow> {
        let mut enc = Encoder::new(rg);
        let encoded: Vec<EncodedRow> = rows
            .iter()
            .map(|(c, rt, it)| enc.encode(*c, rt.clone(), it.clone()).unwrap())
            .collect();
        for (g, group) in encoded.chunks(rg).enumerate() {
            let mut dec = Decoder::new();
            for (j, e) in group.iter().enumerate() {
                let (c, rt, it) = &rows[g * rg + j];
                let (drt, dit) = dec
                    .row(*c, &e.rt, &e.intensity, e.trace_offset, e.trace_len)
                    .unwrap();
                assert_eq!(
                    bits(drt),
                    bits(rt),
                    "rt of row {} at {rg} rows per group",
                    g * rg + j
                );
                assert_eq!(
                    bits(dit),
                    bits(it),
                    "intensity of row {} at {rg}",
                    g * rg + j
                );
            }
        }
        encoded
    }

    #[test]
    fn every_row_group_decodes_on_its_own_bit_for_bit() {
        let rows = rows_fixture();
        for rg in 1..=rows.len() + 1 {
            round_trip(&rows, rg);
        }
        // At one row per group every observed row carries its own axis.
        let one = round_trip(&rows, 1);
        for (e, (_, rt, _)) in one.iter().zip(&rows) {
            assert_eq!(e.rt.len(), rt.len());
        }
    }

    #[test]
    fn the_encoder_drops_repeated_axes_and_zero_margins() {
        let rows = rows_fixture();
        let e = round_trip(&rows, 1 << 16);
        // Candidate 3: the first observed row carries the grid, the later ones do not.
        assert_eq!(e[1].rt.len(), 6);
        for i in [2, 4, 5, 6] {
            assert!(e[i].rt.is_empty(), "row {i} repeats the grid");
        }
        assert_eq!((e[0].trace_len, e[3].trace_len), (0, 0));
        assert_eq!(
            (e[1].trace_offset, e[1].intensity.clone()),
            (2, vec![5.0, 7.0])
        );
        // No zero margin: stored whole, from offset 0.
        assert_eq!((e[2].trace_offset, e[2].intensity.len()), (0, 6));
        // All zero: nothing stored, the length kept.
        assert_eq!((e[4].trace_len, e[4].intensity.len()), (6, 0));
        // `-0.0` and NaN are values, not margins.
        assert_eq!((e[5].trace_offset, e[5].intensity.len()), (0, 6));
        assert_eq!((e[6].trace_offset, e[6].intensity.clone()), (5, vec![3.0]));
        // Sparse candidate 4: a new axis is written, a repeated one is not, and an axis that
        // differs only in the sign of a zero is a different axis.
        assert!(!e[7].rt.is_empty() && !e[8].rt.is_empty());
        assert!(e[9].rt.is_empty());
        assert_eq!(bits(&e[10].rt), bits(&[-0.0, 4.0, 9.0]));
        // Candidate 12 carries its axis once.
        let axes: usize = e[14..19].iter().filter(|r| !r.rt.is_empty()).count();
        assert_eq!(axes, 1);
    }

    #[test]
    fn a_row_without_its_axis_is_an_error_not_a_guess() {
        let rows = rows_fixture();
        let e = round_trip(&rows, 1 << 16);
        // Row 2 needs the axis row 1 carried. Read from row 2, it has none.
        let mut dec = Decoder::new();
        let err = dec
            .row(
                3,
                &e[2].rt,
                &e[2].intensity,
                e[2].trace_offset,
                e[2].trace_len,
            )
            .unwrap_err();
        assert!(format!("{err}").contains("no retention-time axis"), "{err}");
        // Another candidate's axis is not borrowed either.
        let mut dec = Decoder::new();
        dec.skip(
            7,
            &e[11].rt,
            &e[11].intensity,
            e[11].trace_offset,
            e[11].trace_len,
        )
        .unwrap();
        assert!(dec
            .row(
                3,
                &e[2].rt,
                &e[2].intensity,
                e[2].trace_offset,
                e[2].trace_len
            )
            .is_err());
        // `skip` over the rows before it is what makes the read possible.
        let mut dec = Decoder::new();
        dec.skip(
            3,
            &e[1].rt,
            &e[1].intensity,
            e[1].trace_offset,
            e[1].trace_len,
        )
        .unwrap();
        let (rt, it) = dec
            .row(
                3,
                &e[2].rt,
                &e[2].intensity,
                e[2].trace_offset,
                e[2].trace_len,
            )
            .unwrap();
        assert_eq!((bits(rt), bits(it)), (bits(&rows[2].1), bits(&rows[2].2)));
        // Stored values past the trace, and an absent row that stores values, are refused.
        assert!(Decoder::new()
            .row(1, &[1.0, 2.0], &[1.0, 1.0], 1, 2)
            .is_err());
        assert!(Decoder::new().row(1, &[], &[1.0], 0, 0).is_err());
        assert!(Decoder::new().row(1, &[1.0], &[], 0, 2).is_err());
    }

    #[test]
    fn a_mismatched_v1_row_is_refused_by_the_encoder() {
        let mut enc = Encoder::new(8);
        assert!(enc.encode(1, vec![1.0, 2.0], vec![1.0]).is_err());
    }

    /// Write `rows` as a v1 table of `rg`-row groups (List or LargeList per `large`).
    fn write_v1(path: &str, rows: &[(u32, Vec<f32>, Vec<f32>)], rg: usize) {
        let mut w = writer(path, rg);
        let mut r = Rows::default();
        for (i, (c, rt, it)) in rows.iter().enumerate() {
            r.cid.push(*c);
            r.name.push(format!("y{i}"));
            r.frag_mz.push(100.0 + i as f64);
            r.frag_obs_mz.push(100.0 + i as f64 + 1e-4);
            r.predicted_intensity.push(1.0 / (i + 1) as f32);
            r.rt.push(rt.clone());
            r.intensity.push(it.clone());
        }
        w.write_cols(r.into_cols(Layout::V1, Optional::ALL))
            .unwrap();
        w.close().unwrap();
    }

    /// One row as [`for_each_row`] hands it over, every float as its bit pattern.
    type RowBits = (u32, String, u64, u64, u32, Vec<u32>, Vec<u32>);

    fn dense_rows(path: &str) -> Vec<RowBits> {
        let tf = TableFile::open(path).unwrap();
        let mut out = Vec::new();
        for_each_row(&tf, |r| {
            out.push((
                r.cid,
                r.name.to_string(),
                r.frag_mz.unwrap().to_bits(),
                r.frag_obs_mz.unwrap().to_bits(),
                r.predicted_intensity.to_bits(),
                bits(r.rt),
                bits(r.intensity),
            ));
            Ok(())
        })
        .unwrap();
        out
    }

    #[test]
    fn a_rewritten_table_reads_back_as_the_source_at_every_row_group_size() {
        let rows = rows_fixture();
        let v1 = scratch("rw_v1.parquet");
        write_v1(&v1, &rows, 1 << 16);
        let want = dense_rows(&v1);
        assert_eq!(want.len(), rows.len());
        for rg in 1..=rows.len() + 1 {
            let v2 = scratch(&format!("rw_v2_{rg}.parquet"));
            assert_eq!(
                rewrite(&v1, &v2, Layout::V2, rg).unwrap(),
                rows.len() as u64
            );
            let tf = TableFile::open(&v2).unwrap();
            assert_eq!(Layout::of(&tf).unwrap(), Layout::V2);
            // The writer cut its row groups where the encoder restarted its rule.
            let groups = tf.row_group_rows();
            assert!(
                groups[..groups.len() - 1].iter().all(|&n| n == rg),
                "{groups:?}"
            );
            assert_eq!(dense_rows(&v2), want, "{rg} rows per row group");
            // Every row group on its own, as quant and the pool read them.
            let mut first = 0usize;
            let mut got = Vec::new();
            for n in groups {
                let span = tf.span(first, n).unwrap();
                for_each_row(&span, |r| {
                    got.push((r.cid, bits(r.rt), bits(r.intensity)));
                    Ok(())
                })
                .unwrap();
                first += n;
            }
            let want_traces: Vec<_> = want
                .iter()
                .map(|w| (w.0, w.5.clone(), w.6.clone()))
                .collect();
            assert_eq!(got, want_traces, "row groups read one by one at {rg}");
            // And back to v1: the same rows as the source.
            let back = scratch(&format!("rw_back_{rg}.parquet"));
            rewrite(&v2, &back, Layout::V1, 1 << 16).unwrap();
            assert_eq!(
                Layout::of(&TableFile::open(&back).unwrap()).unwrap(),
                Layout::V1
            );
            assert_eq!(dense_rows(&back), want);
        }
    }

    #[test]
    fn an_empty_table_rewrites_to_an_empty_table_of_the_layout() {
        let v1 = scratch("empty_v1.parquet");
        write_v1(&v1, &[], 16);
        let v2 = scratch("empty_v2.parquet");
        assert_eq!(rewrite(&v1, &v2, Layout::V2, 16).unwrap(), 0);
        let tf = TableFile::open(&v2).unwrap();
        assert_eq!((tf.nrows, Layout::of(&tf).unwrap()), (0, Layout::V2));
    }

    #[test]
    fn half_a_v2_schema_is_neither_layout() {
        let p = scratch("half.parquet");
        let mut w = writer(&p, 16);
        let mut cols = Rows::default().into_cols(Layout::V1, Optional::ALL);
        cols.push(Col::U32(TRACE_LEN.into(), Vec::new()));
        w.write_cols(cols).unwrap();
        w.close().unwrap();
        assert!(Layout::of(&TableFile::open(&p).unwrap()).is_err());
    }

    /// The layouts measured on a real run, and held to the same downstream bytes.
    ///
    /// `MUMDIA_CHROM_V2_DIR` is a run directory holding `chromatograms.parquet`,
    /// `psms_extracted.parquet`, `seed_psms.parquet` and `psms_scored.parquet` (or
    /// `scored.parquet`);
    /// `MUMDIA_CHROM_V2_OUT` is a scratch directory. The artifact is rewritten in the
    /// current v1 layout (so both arms come from today's writer), in v2, and in v2 with a
    /// dictionary-encoded `rt`; the sizes and write times are printed. Features (the
    /// Extended set with the confident bounds) and quant then run on v1 and v2, and every
    /// table they write must be byte-identical; their wall times are printed as well.
    ///
    /// ```text
    /// MUMDIA_CHROM_V2_DIR=run MUMDIA_CHROM_V2_OUT=scratch \
    ///   cargo test -p mumdia --release --lib -- --ignored --nocapture v2_on_a_real_artifact
    /// ```
    #[test]
    #[ignore = "measurement; needs MUMDIA_CHROM_V2_DIR and MUMDIA_CHROM_V2_OUT"]
    fn v2_on_a_real_artifact() {
        use crate::stages::{features, quant};
        use std::time::Instant;
        let (Ok(dir), Ok(out)) = (
            std::env::var("MUMDIA_CHROM_V2_DIR"),
            std::env::var("MUMDIA_CHROM_V2_OUT"),
        ) else {
            println!("set MUMDIA_CHROM_V2_DIR and MUMDIA_CHROM_V2_OUT to run this");
            return;
        };
        std::fs::create_dir_all(&out).unwrap();
        let at = |d: &str, f: &str| {
            std::path::Path::new(d)
                .join(f)
                .to_string_lossy()
                .to_string()
        };
        let src = at(&dir, "chromatograms.parquet");
        let size = |p: &str| std::fs::metadata(p).unwrap().len();
        let mb = |b: u64| b as f64 / 1e6;
        println!("source {src}: {:.1} MB", mb(size(&src)));
        let arm = |name: &str, layout: Layout, w: TableWriter, path: &str| {
            let t = Instant::now();
            let rows = rewrite_into(&src, layout, ROW_GROUP_ROWS, w).unwrap();
            println!(
                "{name}: {rows} rows, {:.1} MB, rewritten in {:.1} s",
                mb(size(path)),
                t.elapsed().as_secs_f64()
            );
        };
        let v1 = at(&out, "chrom_v1.parquet");
        let v2 = at(&out, "chrom_v2.parquet");
        let v2_dict = at(&out, "chrom_v2_dict_rt.parquet");
        arm("v1", Layout::V1, writer(&v1, ROW_GROUP_ROWS), &v1);
        arm("v2", Layout::V2, writer(&v2, ROW_GROUP_ROWS), &v2);
        arm(
            "v2, rt dictionary",
            Layout::V2,
            TableWriter::new(&v2_dict).with_row_group_rows(ROW_GROUP_ROWS),
            &v2_dict,
        );
        println!(
            "v2 against v1: {:.1}% smaller",
            100.0 * (1.0 - size(&v2) as f64 / size(&v1) as f64)
        );
        // Stored values, the survey's measure.
        let values = |p: &str| -> (u64, u64) {
            let tf = TableFile::open(p).unwrap();
            let (mut r, mut i) = (0u64, 0u64);
            for b in tf.batches(Some(&["rt", "intensity"]), 1 << 14).unwrap() {
                let b = b.unwrap();
                let s = b.schema();
                let rt = ListF32::of(b.column(s.index_of("rt").unwrap()), "rt").unwrap();
                let it = ListF32::of(b.column(s.index_of("intensity").unwrap()), "i").unwrap();
                for k in 0..b.num_rows() {
                    r += rt.row_slice(k, "rt").unwrap().len() as u64;
                    i += it.row_slice(k, "i").unwrap().len() as u64;
                }
            }
            (r, i)
        };
        let (r1, i1) = values(&v1);
        let (r2, i2) = values(&v2);
        println!("list values: rt {r1} -> {r2}, intensity {i1} -> {i2}");

        let fcfg = mumdia_core::config::FeaturesConfig {
            set: mumdia_core::config::FeatureSet::Extended,
            bound_from_confident: true,
            ..Default::default()
        };
        let seed = at(&dir, "seed_psms.parquet");
        let psms = at(&dir, "psms_extracted.parquet");
        // `scored.parquet` where a benchmark directory renamed it.
        let scored = ["psms_scored.parquet", "scored.parquet"]
            .iter()
            .map(|f| at(&dir, f))
            .find(|p| std::path::Path::new(p).exists())
            .expect("a psms_scored.parquet (or scored.parquet) in MUMDIA_CHROM_V2_DIR");
        let mut outputs: Vec<Vec<String>> = Vec::new();
        for (tag, chrom) in [("v1", &v1), ("v2", &v2)] {
            let feats = at(&out, &format!("features_{tag}.parquet"));
            let t = Instant::now();
            features::run(features::FeaturesParams {
                psms: &psms,
                chromatograms: chrom,
                seed: Some(&seed),
                out: &feats,
                out_pin: "",
                cfg: &fcfg,
                config_hash: "test",
            })
            .unwrap();
            let tf = t.elapsed().as_secs_f64();
            let (pep, prot, frag) = (
                at(&out, &format!("peptide_{tag}.parquet")),
                at(&out, &format!("protein_{tag}.parquet")),
                at(&out, &format!("fragment_{tag}.parquet")),
            );
            let t = Instant::now();
            quant::run(quant::QuantParams {
                psms_scored: &scored,
                chromatograms: &[quant::ChromTable::whole(chrom)],
                out_peptide: &pep,
                out_protein: &prot,
                out_fragment: Some(&frag),
                out_peak_bounds: None,
                cfg: &mumdia_core::config::QuantConfig::default(),
                config_hash: "test",
            })
            .unwrap();
            println!(
                "{tag}: features {tf:.1} s, quant {:.1} s",
                t.elapsed().as_secs_f64()
            );
            outputs.push(vec![feats, pep, prot, frag]);
        }
        for (a, b) in outputs[0].iter().zip(&outputs[1]) {
            assert!(
                std::fs::read(a).unwrap() == std::fs::read(b).unwrap(),
                "{a} and {b} differ"
            );
            println!("identical: {a} = {b}");
        }
    }
}
