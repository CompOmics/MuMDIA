//! A small typed column/table layer over Arrow + Parquet so each stage writes
//! and reads its declared schema without hand-rolling RecordBatches.
//!
//! Tables are written as Parquet (SNAPPY), the open, self-describing interstage
//! format (PLAN.md Section 3.3). Ion-mobility and other conditional columns are
//! nullable via the `Opt*` variants (PLAN.md Section 2 missing-value policy).

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float32Builder, Float64Array, Int32Array,
    Int64Array, LargeListArray, LargeListBuilder, ListArray, ListBuilder, StringArray, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// One named, typed column for writing.
pub enum Col {
    I64(String, Vec<i64>),
    I32(String, Vec<i32>),
    U32(String, Vec<u32>),
    F64(String, Vec<f64>),
    F32(String, Vec<f32>),
    Bool(String, Vec<bool>),
    Str(String, Vec<String>),
    OptF64(String, Vec<Option<f64>>),
    OptF32(String, Vec<Option<f32>>),
    OptI32(String, Vec<Option<i32>>),
    OptStr(String, Vec<Option<String>>),
    ListF32(String, Vec<Vec<f32>>),
    ListF64(String, Vec<Vec<f64>>),
    /// Like `ListF32` but encoded as an Arrow `LargeList` (64-bit offsets).
    /// Required for columns whose total list-value count can exceed the ~2.1B
    /// limit of the 32-bit `ListArray` offset buffer (e.g. per-fragment
    /// chromatograms when extraction accepts a very large candidate set).
    LargeListF32(String, Vec<Vec<f32>>),
}

impl Col {
    fn name(&self) -> &str {
        match self {
            Col::I64(n, _)
            | Col::I32(n, _)
            | Col::U32(n, _)
            | Col::F64(n, _)
            | Col::F32(n, _)
            | Col::Bool(n, _)
            | Col::Str(n, _)
            | Col::OptF64(n, _)
            | Col::OptF32(n, _)
            | Col::OptI32(n, _)
            | Col::OptStr(n, _)
            | Col::ListF32(n, _)
            | Col::ListF64(n, _)
            | Col::LargeListF32(n, _) => n,
        }
    }

    fn len(&self) -> usize {
        match self {
            Col::I64(_, v) => v.len(),
            Col::I32(_, v) => v.len(),
            Col::U32(_, v) => v.len(),
            Col::F64(_, v) => v.len(),
            Col::F32(_, v) => v.len(),
            Col::Bool(_, v) => v.len(),
            Col::Str(_, v) => v.len(),
            Col::OptF64(_, v) => v.len(),
            Col::OptF32(_, v) => v.len(),
            Col::OptI32(_, v) => v.len(),
            Col::OptStr(_, v) => v.len(),
            Col::ListF32(_, v) => v.len(),
            Col::ListF64(_, v) => v.len(),
            Col::LargeListF32(_, v) => v.len(),
        }
    }

    fn field(&self) -> Field {
        let item32 = || Arc::new(Field::new("item", DataType::Float32, true));
        let item64 = || Arc::new(Field::new("item", DataType::Float64, true));
        match self {
            Col::I64(n, _) => Field::new(n, DataType::Int64, false),
            Col::I32(n, _) => Field::new(n, DataType::Int32, false),
            Col::U32(n, _) => Field::new(n, DataType::UInt32, false),
            Col::F64(n, _) => Field::new(n, DataType::Float64, false),
            Col::F32(n, _) => Field::new(n, DataType::Float32, false),
            Col::Bool(n, _) => Field::new(n, DataType::Boolean, false),
            Col::Str(n, _) => Field::new(n, DataType::Utf8, false),
            Col::OptF64(n, _) => Field::new(n, DataType::Float64, true),
            Col::OptF32(n, _) => Field::new(n, DataType::Float32, true),
            Col::OptI32(n, _) => Field::new(n, DataType::Int32, true),
            Col::OptStr(n, _) => Field::new(n, DataType::Utf8, true),
            Col::ListF32(n, _) => Field::new(n, DataType::List(item32()), true),
            Col::ListF64(n, _) => Field::new(n, DataType::List(item64()), true),
            Col::LargeListF32(n, _) => Field::new(n, DataType::LargeList(item32()), true),
        }
    }

    /// Consuming counterpart to a borrowing `array()`: moves the inner `Vec`
    /// into the Arrow array instead of cloning it, so `write_table` copies the
    /// column data only once. Field name/type/nullability are unchanged.
    fn into_array(self) -> ArrayRef {
        match self {
            Col::I64(_, v) => Arc::new(Int64Array::from(v)),
            Col::I32(_, v) => Arc::new(Int32Array::from(v)),
            Col::U32(_, v) => Arc::new(UInt32Array::from(v)),
            Col::F64(_, v) => Arc::new(Float64Array::from(v)),
            Col::F32(_, v) => Arc::new(Float32Array::from(v)),
            Col::Bool(_, v) => Arc::new(BooleanArray::from(v)),
            Col::Str(_, v) => Arc::new(StringArray::from(v)),
            Col::OptF64(_, v) => Arc::new(Float64Array::from(v)),
            Col::OptF32(_, v) => Arc::new(Float32Array::from(v)),
            Col::OptI32(_, v) => Arc::new(Int32Array::from(v)),
            Col::OptStr(_, v) => Arc::new(StringArray::from(v)),
            Col::ListF32(_, v) => {
                // Reserve the exact total up front and CONSUME the source rows, so each
                // inner Vec is freed as it is copied. Building without capacity reallocated
                // the values buffer repeatedly, and borrowing kept the whole source
                // Vec<Vec<f32>> alive alongside the finished array -- two full copies of
                // the chromatogram values at peak.
                let total: usize = v.iter().map(|r| r.len()).sum();
                let mut b =
                    ListBuilder::with_capacity(Float32Builder::with_capacity(total), v.len());
                for row in v {
                    b.values().append_slice(&row);
                    b.append(true);
                }
                Arc::new(b.finish())
            }
            Col::ListF64(_, v) => {
                use arrow::array::Float64Builder;
                // See ListF32: reserve exactly, and consume so inner Vecs free as copied.
                let total: usize = v.iter().map(|r| r.len()).sum();
                let mut b =
                    ListBuilder::with_capacity(Float64Builder::with_capacity(total), v.len());
                for row in v {
                    b.values().append_slice(&row);
                    b.append(true);
                }
                Arc::new(b.finish())
            }
            Col::LargeListF32(_, v) => {
                // Same as ListF32, and this is the variant that carries the very large
                // chromatogram columns (tens of millions of rows), so the reserve and the
                // progressive free matter most here.
                let total: usize = v.iter().map(|r| r.len()).sum();
                let mut b =
                    LargeListBuilder::with_capacity(Float32Builder::with_capacity(total), v.len());
                for row in v {
                    b.values().append_slice(&row);
                    b.append(true);
                }
                Arc::new(b.finish())
            }
        }
    }
}

/// Validate a set of typed columns and turn them into one Arrow schema + record batch.
/// Shared by [`write_table`] (one batch = the whole table) and [`TableWriter`] (one batch
/// per chunk), so both write paths declare a column identically.
fn cols_to_batch(path: &str, cols: Vec<Col>) -> Result<(Arc<Schema>, RecordBatch)> {
    if cols.is_empty() {
        return Err(anyhow!("write_table: no columns for {path}"));
    }
    // Reject duplicate column names: Arrow allows them but readers resolve a
    // name to the first match, silently hiding the second column.
    let mut names = std::collections::HashSet::new();
    for c in &cols {
        if !names.insert(c.name()) {
            return Err(anyhow!(
                "write_table: duplicate column '{}' for {path}",
                c.name()
            ));
        }
    }
    let nrows = cols[0].len();
    for c in &cols {
        if c.len() != nrows {
            return Err(anyhow!(
                "write_table: column '{}' has {} rows, expected {}",
                c.name(),
                c.len(),
                nrows
            ));
        }
    }
    let fields: Vec<Field> = cols.iter().map(|c| c.field()).collect();
    let schema = Arc::new(Schema::new(fields));
    // Consume the columns so each Vec is moved into its Arrow array rather than
    // cloned. `fields` above already captured everything the schema needs.
    let arrays: Vec<ArrayRef> = cols.into_iter().map(|c| c.into_array()).collect();
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .with_context(|| format!("building record batch for {path}"))?;
    Ok((schema, batch))
}

fn snappy_props(row_group_rows: Option<usize>) -> WriterProperties {
    let mut b = WriterProperties::builder().set_compression(Compression::SNAPPY);
    if let Some(n) = row_group_rows {
        b = b.set_max_row_group_row_count(Some(n.max(1)));
    }
    b.build()
}

/// Write columns to a Parquet file. Returns the row count. All columns must
/// share the same length.
pub fn write_table(path: &str, cols: Vec<Col>) -> Result<u64> {
    let (schema, batch) = cols_to_batch(path, cols)?;
    let nrows = batch.num_rows();
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = std::fs::File::create(path).with_context(|| format!("creating {path}"))?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(snappy_props(None)))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(nrows as u64)
}

/// Incremental typed writer: the chunked counterpart of [`write_table`]. Feed `Vec<Col>`
/// chunks that all declare the same columns (names, types, order) and each chunk is encoded
/// as it arrives, so the resident set is one chunk plus the encoder's in-progress row group
/// instead of the whole table. The arrow writer encodes and compresses pages as rows come in;
/// what stays in memory between flushes is the compressed in-progress row group, which
/// [`TableWriter::with_row_group_rows`] bounds.
///
/// The first chunk fixes the schema (a zero-row chunk is enough), so a caller must write at
/// least one chunk before [`TableWriter::close`]; every later chunk must match it exactly.
/// Rows are stored in write order, so an output written in N chunks reads back identical to
/// the same rows written once with [`write_table`]; only the row-group boundaries differ.
pub struct TableWriter {
    path: String,
    schema: Option<Arc<Schema>>,
    writer: Option<ArrowWriter<std::fs::File>>,
    rows: u64,
    row_group_rows: Option<usize>,
}

impl TableWriter {
    /// Create a writer for `path`. Nothing is opened until the first chunk arrives.
    pub fn new(path: &str) -> TableWriter {
        TableWriter {
            path: path.to_string(),
            schema: None,
            writer: None,
            rows: 0,
            row_group_rows: None,
        }
    }

    /// Cap the rows per parquet row group (default: the parquet writer's 1,048,576).
    /// Smaller row groups bound the compressed in-progress buffer for wide list columns
    /// (chromatogram traces, spectra peak lists); keep them at tens of thousands of rows
    /// so the footer stays small and readers still get large batches.
    pub fn with_row_group_rows(mut self, rows: usize) -> TableWriter {
        self.row_group_rows = Some(rows.max(1));
        self
    }

    /// Append one chunk. Empty chunks are accepted (they only fix or check the schema).
    pub fn write_cols(&mut self, cols: Vec<Col>) -> Result<()> {
        let (schema, batch) = cols_to_batch(&self.path, cols)?;
        match &self.schema {
            None => {
                if let Some(parent) = std::path::Path::new(&self.path).parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                let file = std::fs::File::create(&self.path)
                    .with_context(|| format!("creating {}", self.path))?;
                self.writer = Some(ArrowWriter::try_new(
                    file,
                    schema.clone(),
                    Some(snappy_props(self.row_group_rows)),
                )?);
                self.schema = Some(schema);
            }
            Some(first) => {
                if first.as_ref() != schema.as_ref() {
                    return Err(anyhow!(
                        "TableWriter: chunk schema for {} differs from the first chunk\n  first: {:?}\n  chunk: {:?}",
                        self.path,
                        first.fields(),
                        schema.fields()
                    ));
                }
            }
        }
        if batch.num_rows() > 0 {
            self.writer
                .as_mut()
                .expect("writer opened on the first chunk")
                .write(&batch)
                .with_context(|| format!("writing parquet chunk to {}", self.path))?;
            self.rows += batch.num_rows() as u64;
        }
        Ok(())
    }

    /// Rows written so far.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Finish the file (the footer is written here) and return the row count.
    pub fn close(mut self) -> Result<u64> {
        let w = self.writer.take().ok_or_else(|| {
            anyhow!(
                "TableWriter: no chunk written for {}; write one (possibly empty) chunk to fix the schema",
                self.path
            )
        })?;
        w.close()
            .with_context(|| format!("closing parquet writer {}", self.path))?;
        Ok(self.rows)
    }
}

/// Write pre-built Arrow record batches to a Snappy Parquet file, preserving
/// their schema exactly. Unlike [`write_table`] (which builds columns from typed
/// vecs) this is for passing an existing schema through unchanged, e.g. filtering
/// a scored table by run without re-declaring its column set. Returns the row count.
/// Incremental parquet writer: feed `RecordBatch`es one at a time and the rows are encoded
/// and flushed as they arrive, so peak memory is one batch rather than the whole table.
///
/// [`write_table`] and [`write_batches`] both need every column materialised up front, which
/// is fine for ordinary artifacts but not for the rescoring feature matrix - hundreds of
/// columns over millions of rows, where the caller already holds the data once.
pub struct BatchWriter {
    writer: Option<ArrowWriter<std::fs::File>>,
    rows: u64,
}

impl BatchWriter {
    pub fn new(path: &str, schema: Arc<Schema>) -> Result<BatchWriter> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let file = std::fs::File::create(path).with_context(|| format!("creating {path}"))?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        Ok(BatchWriter {
            writer: Some(ArrowWriter::try_new(file, schema, Some(props))?),
            rows: 0,
        })
    }

    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        self.rows += batch.num_rows() as u64;
        self.writer
            .as_mut()
            .expect("writer closed")
            .write(batch)
            .context("writing parquet batch")?;
        Ok(())
    }

    /// Finish the file and return the row count. Must be called: the footer is written here.
    pub fn close(mut self) -> Result<u64> {
        if let Some(w) = self.writer.take() {
            w.close().context("closing parquet writer")?;
        }
        Ok(self.rows)
    }
}

pub fn write_batches(path: &str, schema: Arc<Schema>, batches: &[RecordBatch]) -> Result<u64> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = std::fs::File::create(path).with_context(|| format!("creating {path}"))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    let mut n = 0u64;
    for b in batches {
        writer.write(b)?;
        n += b.num_rows() as u64;
    }
    writer.close()?;
    Ok(n)
}

/// A read-back table: all batches concatenated logically, accessed by column
/// name with typed getters.
pub struct Table {
    pub schema: Arc<Schema>,
    pub batches: Vec<RecordBatch>,
    pub nrows: usize,
}

/// Row count straight from the parquet footer metadata, without decoding any column
/// data. `Table::read(path)?.nrows` materialises the whole file (hundreds of millions of
/// rows for a fragment library) just to learn its length; use this instead whenever only
/// the count is needed.
pub fn nrows(path: &str) -> Result<u64> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet footer {path}"))?;
    Ok(builder.metadata().file_metadata().num_rows() as u64)
}

/// Column names straight from the parquet footer metadata, in file order, without decoding
/// any column data. The counterpart to [`nrows`] for callers that need the schema rather
/// than the contents -- a features/competed artifact carries ~390 columns, so
/// `Table::read(path)?.column_names()` decodes every one of them to answer a question the
/// footer already contains.
pub fn column_names(path: &str) -> Result<Vec<String>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet footer {path}"))?;
    Ok(builder
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect())
}

impl Table {
    pub fn read(path: &str) -> Result<Table> {
        Self::read_inner(path, None)
    }

    /// Read only the named columns. Every other column is skipped in the parquet reader,
    /// so its pages are never fetched or decoded. Useful for the wide artifacts: a
    /// competed/features table carries ~390 columns and most callers want a handful, but
    /// `read` decodes all of them and holds the batches for the table's lifetime.
    ///
    /// Names not present in the file are ignored (the resulting `Table` simply will not
    /// have them, and the typed getters report the missing column as they always do).
    pub fn read_cols(path: &str, columns: &[&str]) -> Result<Table> {
        Self::read_inner(path, Some(columns))
    }

    fn read_inner(path: &str, columns: Option<&[&str]>) -> Result<Table> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("reading parquet {path}"))?;
        let builder = match columns {
            None => builder,
            Some(want) => {
                // Map requested names to leaf indices in the parquet schema. A projection
                // mask over root fields is enough here: all artifact columns are flat
                // primitives or a single list level.
                let parquet_schema = builder.parquet_schema();
                let mut roots: Vec<usize> = Vec::new();
                for (i, f) in parquet_schema.root_schema().get_fields().iter().enumerate() {
                    if want.contains(&f.name()) {
                        roots.push(i);
                    }
                }
                let mask = parquet::arrow::ProjectionMask::roots(parquet_schema, roots);
                builder.with_projection(mask)
            }
        };
        let reader = builder.build()?;
        // Take the schema from the READER, not the builder: the builder reports the full
        // file schema, so under a projection it would disagree with the batches (which
        // carry only the selected columns) and the typed getters would resolve a name to
        // the wrong column index.
        let schema = arrow::array::RecordBatchReader::schema(&reader);
        let mut batches = Vec::new();
        let mut nrows = 0;
        for b in reader {
            let b = b?;
            nrows += b.num_rows();
            batches.push(b);
        }
        Ok(Table {
            schema,
            batches,
            nrows,
        })
    }

    pub fn column_names(&self) -> Vec<String> {
        self.schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    fn idx(&self, name: &str) -> Result<usize> {
        self.schema
            .index_of(name)
            .map_err(|_| anyhow!("column '{name}' not found in {:?}", self.column_names()))
    }

    pub fn f64(&self, name: &str) -> Result<Vec<f64>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            push_f64(&mut out, b.column(i), name)?;
        }
        Ok(out)
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            push_f32(&mut out, b.column(i), name)?;
        }
        Ok(out)
    }

    pub fn i64(&self, name: &str) -> Result<Vec<i64>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            push_i64(&mut out, b.column(i), name)?;
        }
        Ok(out)
    }

    pub fn i32(&self, name: &str) -> Result<Vec<i32>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            push_i32(&mut out, b.column(i), name)?;
        }
        Ok(out)
    }

    pub fn u32(&self, name: &str) -> Result<Vec<u32>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            push_u32(&mut out, b.column(i), name)?;
        }
        Ok(out)
    }

    pub fn bool(&self, name: &str) -> Result<Vec<bool>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            push_bool(&mut out, b.column(i), name)?;
        }
        Ok(out)
    }

    pub fn str(&self, name: &str) -> Result<Vec<String>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            push_str(&mut out, b.column(i), name)?;
        }
        Ok(out)
    }

    pub fn opt_f64(&self, name: &str) -> Result<Vec<Option<f64>>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            push_opt_f64(&mut out, b.column(i), name)?;
        }
        Ok(out)
    }

    /// Read an f32 list column. Accepts both `List` (32-bit offsets) and
    /// `LargeList` (64-bit offsets, written by `Col::LargeListF32`) encodings,
    /// so chromatogram artifacts written by either binary read back the same.
    pub fn list_f32(&self, name: &str) -> Result<Vec<Vec<f32>>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            push_list_f32(&mut out, b.column(i), name)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Column decoding shared by `Table` (materialised) and `TableFile` (streaming).
//
// Both read paths must decode identically, because a stage that moves from one to
// the other must see the same values; so the per-array rules live here exactly
// once. Null policy: f64/f32 nulls read as NaN, utf8 nulls as "", integer/bool
// nulls as the underlying buffer value, and `opt_f64` keeps them as `None`.
// ---------------------------------------------------------------------------

fn downcast<'a, T: 'static>(col: &'a ArrayRef, name: &str, what: &str) -> Result<&'a T> {
    col.as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| anyhow!("column '{name}' is not {what}"))
}

fn push_f64(out: &mut Vec<f64>, col: &ArrayRef, name: &str) -> Result<()> {
    let a: &Float64Array = downcast(col, name, "f64")?;
    if a.null_count() == 0 {
        out.extend_from_slice(a.values());
    } else {
        for k in 0..a.len() {
            out.push(if a.is_null(k) { f64::NAN } else { a.value(k) });
        }
    }
    Ok(())
}

fn push_f32(out: &mut Vec<f32>, col: &ArrayRef, name: &str) -> Result<()> {
    let a: &Float32Array = downcast(col, name, "f32")?;
    if a.null_count() == 0 {
        out.extend_from_slice(a.values());
    } else {
        for k in 0..a.len() {
            out.push(if a.is_null(k) { f32::NAN } else { a.value(k) });
        }
    }
    Ok(())
}

fn push_i64(out: &mut Vec<i64>, col: &ArrayRef, name: &str) -> Result<()> {
    let a: &Int64Array = downcast(col, name, "i64")?;
    if a.null_count() == 0 {
        out.extend_from_slice(a.values());
    } else {
        for k in 0..a.len() {
            out.push(a.value(k));
        }
    }
    Ok(())
}

fn push_i32(out: &mut Vec<i32>, col: &ArrayRef, name: &str) -> Result<()> {
    let a: &Int32Array = downcast(col, name, "i32")?;
    if a.null_count() == 0 {
        out.extend_from_slice(a.values());
    } else {
        for k in 0..a.len() {
            out.push(a.value(k));
        }
    }
    Ok(())
}

fn push_u32(out: &mut Vec<u32>, col: &ArrayRef, name: &str) -> Result<()> {
    let a: &UInt32Array = downcast(col, name, "u32")?;
    if a.null_count() == 0 {
        out.extend_from_slice(a.values());
    } else {
        for k in 0..a.len() {
            out.push(a.value(k));
        }
    }
    Ok(())
}

fn push_bool(out: &mut Vec<bool>, col: &ArrayRef, name: &str) -> Result<()> {
    let a: &BooleanArray = downcast(col, name, "bool")?;
    for k in 0..a.len() {
        out.push(a.value(k));
    }
    Ok(())
}

fn push_str(out: &mut Vec<String>, col: &ArrayRef, name: &str) -> Result<()> {
    let a: &StringArray = downcast(col, name, "utf8")?;
    for k in 0..a.len() {
        out.push(if a.is_null(k) {
            String::new()
        } else {
            a.value(k).to_string()
        });
    }
    Ok(())
}

fn push_opt_f64(out: &mut Vec<Option<f64>>, col: &ArrayRef, name: &str) -> Result<()> {
    let a: &Float64Array = downcast(col, name, "f64")?;
    for k in 0..a.len() {
        out.push(if a.is_null(k) { None } else { Some(a.value(k)) });
    }
    Ok(())
}

/// Visit each row of an f32 list column (`List` or `LargeList`); `None` for a null row.
/// The inner array is the row's own f32 slice.
fn for_each_list_f32(
    col: &ArrayRef,
    name: &str,
    mut f: impl FnMut(Option<&Float32Array>) -> Result<()>,
) -> Result<()> {
    fn inner<'a>(v: &'a ArrayRef, name: &str) -> Result<&'a Float32Array> {
        v.as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow!("list '{name}' inner is not f32"))
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeListArray>() {
        for k in 0..a.len() {
            if a.is_null(k) {
                f(None)?;
            } else {
                let v = a.value(k);
                f(Some(inner(&v, name)?))?;
            }
        }
    } else if let Some(a) = col.as_any().downcast_ref::<ListArray>() {
        for k in 0..a.len() {
            if a.is_null(k) {
                f(None)?;
            } else {
                let v = a.value(k);
                f(Some(inner(&v, name)?))?;
            }
        }
    } else {
        return Err(anyhow!("column '{name}' is not a list"));
    }
    Ok(())
}

fn push_list_f32(out: &mut Vec<Vec<f32>>, col: &ArrayRef, name: &str) -> Result<()> {
    for_each_list_f32(col, name, |row| {
        out.push(match row {
            Some(a) => a.values().to_vec(),
            None => Vec::new(),
        });
        Ok(())
    })
}

/// Flat layout: one values buffer plus `offsets` (row `r` is `values[offsets[r]..offsets[r+1]]`).
/// One allocation instead of one per row. Null rows are empty.
fn push_list_f32_flat(
    offsets: &mut Vec<usize>,
    values: &mut Vec<f32>,
    col: &ArrayRef,
    name: &str,
) -> Result<()> {
    if offsets.is_empty() {
        offsets.push(0);
    }
    for_each_list_f32(col, name, |row| {
        if let Some(a) = row {
            values.extend_from_slice(a.values());
        }
        offsets.push(values.len());
        Ok(())
    })
}

/// Borrowed view of an f32 list column (`List` or `LargeList`) for per-row access while
/// iterating a batch: [`ListF32::row`] is row `k` as an owned `Vec<f32>` (empty for null).
pub enum ListF32<'a> {
    Small(&'a ListArray),
    Large(&'a LargeListArray),
}

impl<'a> ListF32<'a> {
    pub fn of(col: &'a ArrayRef, name: &str) -> Result<ListF32<'a>> {
        if let Some(a) = col.as_any().downcast_ref::<LargeListArray>() {
            Ok(ListF32::Large(a))
        } else if let Some(a) = col.as_any().downcast_ref::<ListArray>() {
            Ok(ListF32::Small(a))
        } else {
            Err(anyhow!("column '{name}' is not a list"))
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ListF32::Small(a) => a.len(),
            ListF32::Large(a) => a.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append row `k` to `out` and return the number of values appended (0 for a null
    /// row). The caller keeps one flat buffer plus its own offsets, so a column of tens
    /// of millions of short traces costs one allocation instead of one per row, which
    /// [`ListF32::row`] cannot avoid.
    pub fn append_row(&self, k: usize, out: &mut Vec<f32>, name: &str) -> Result<usize> {
        let v: Option<ArrayRef> = match self {
            ListF32::Small(a) => (!a.is_null(k)).then(|| a.value(k)),
            ListF32::Large(a) => (!a.is_null(k)).then(|| a.value(k)),
        };
        match v {
            None => Ok(0),
            Some(v) => {
                let a = v
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("list '{name}' inner is not f32"))?;
                out.extend_from_slice(a.values());
                Ok(a.len())
            }
        }
    }

    /// Row `k` as an owned `Vec<f32>`; a null row is empty.
    pub fn row(&self, k: usize, name: &str) -> Result<Vec<f32>> {
        let v: Option<ArrayRef> = match self {
            ListF32::Small(a) => (!a.is_null(k)).then(|| a.value(k)),
            ListF32::Large(a) => (!a.is_null(k)).then(|| a.value(k)),
        };
        match v {
            None => Ok(Vec::new()),
            Some(v) => Ok(v
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow!("list '{name}' inner is not f32"))?
                .values()
                .to_vec()),
        }
    }
}

/// Rows per decoded batch for scalar columns (a batch is ~0.5 MB of f64).
const SCALAR_BATCH_ROWS: usize = 1 << 16;
/// Rows per decoded batch for list columns, whose rows are hundreds of values each.
const LIST_BATCH_ROWS: usize = 1 << 12;

/// Streaming record-batch iterator over a parquet file (see [`TableFile::batches`]).
/// One batch is resident at a time; nothing is retained across `next` calls.
pub struct BatchReader {
    inner: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    schema: Arc<Schema>,
}

impl BatchReader {
    /// Schema of the batches this reader yields (the projected columns, in file order).
    pub fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
}

impl Iterator for BatchReader {
    type Item = Result<RecordBatch>;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|r| r.map_err(|e| anyhow!("reading parquet batch: {e}")))
    }
}

/// A parquet table opened by its footer only: schema and row count are known up front,
/// nothing is decoded until asked, and every typed getter streams just its own column,
/// batch by batch, straight into the returned `Vec`.
///
/// This is the low-memory counterpart of [`Table`]. `Table::read` decodes the whole file
/// into Arrow batches and each getter then copies its column out of them, so during a
/// stage's load phase the table exists twice (Arrow plus owned `Vec`s). With `TableFile`
/// the peak for a column is the output `Vec` plus one decoded batch, and columns a stage
/// never asks for are never fetched. Getters share their decoding rules with `Table`
/// (the `push_*` helpers above), so a stage moved from one to the other reads the same
/// values, including the null policy.
///
/// Use [`TableFile::for_each_batch`] when a stage wants several columns row by row
/// without materialising any of them, or when it can keep a subset of rows only.
pub struct TableFile {
    path: String,
    /// Arrow schema of the file, from the parquet footer.
    pub schema: Arc<Schema>,
    /// Row count from the parquet footer.
    pub nrows: usize,
}

impl TableFile {
    /// Open `path` and read its footer. No column data is decoded.
    pub fn open(path: &str) -> Result<TableFile> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("reading parquet footer {path}"))?;
        let nrows = builder.metadata().file_metadata().num_rows().max(0) as usize;
        let schema = builder.schema().clone();
        Ok(TableFile {
            path: path.to_string(),
            schema,
            nrows,
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn column_names(&self) -> Vec<String> {
        self.schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.schema.index_of(name).is_ok()
    }

    fn idx(&self, name: &str) -> Result<usize> {
        self.schema
            .index_of(name)
            .map_err(|_| anyhow!("column '{name}' not found in {:?}", self.column_names()))
    }

    /// Stream the file as record batches of at most `batch_size` rows. `columns` projects
    /// to the named root columns (all columns when `None`); a name that is not in the file
    /// is an error, unlike [`Table::read_cols`] which silently drops it. Batches carry the
    /// projected columns in FILE order, so look them up by name (`batch.schema().index_of`)
    /// rather than by the order of `columns`.
    pub fn batches(&self, columns: Option<&[&str]>, batch_size: usize) -> Result<BatchReader> {
        let file =
            std::fs::File::open(&self.path).with_context(|| format!("opening {}", self.path))?;
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("reading parquet {}", self.path))?
            .with_batch_size(batch_size.max(1));
        if let Some(want) = columns {
            let mask = {
                let parquet_schema = builder.parquet_schema();
                let fields = parquet_schema.root_schema().get_fields();
                let mut roots: Vec<usize> = Vec::with_capacity(want.len());
                for w in want {
                    let i = fields.iter().position(|f| f.name() == *w).ok_or_else(|| {
                        anyhow!("column '{w}' not found in {:?}", self.column_names())
                    })?;
                    roots.push(i);
                }
                roots.sort_unstable();
                roots.dedup();
                parquet::arrow::ProjectionMask::roots(parquet_schema, roots)
            };
            builder = builder.with_projection(mask);
        }
        let reader = builder.build()?;
        // Schema from the READER: under a projection it carries only the selected columns.
        let schema = arrow::array::RecordBatchReader::schema(&reader);
        Ok(BatchReader {
            inner: reader,
            schema,
        })
    }

    /// Run `f` over every batch of the projected `columns` (all when `None`). One batch is
    /// resident at a time.
    pub fn for_each_batch(
        &self,
        columns: Option<&[&str]>,
        batch_size: usize,
        mut f: impl FnMut(&RecordBatch) -> Result<()>,
    ) -> Result<()> {
        for b in self.batches(columns, batch_size)? {
            f(&b?)?;
        }
        Ok(())
    }

    fn column(&self, name: &str, batch_size: usize) -> Result<BatchReader> {
        self.idx(name)?;
        self.batches(Some(&[name]), batch_size)
    }

    pub fn f64(&self, name: &str) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(self.nrows);
        for b in self.column(name, SCALAR_BATCH_ROWS)? {
            push_f64(&mut out, b?.column(0), name)?;
        }
        Ok(out)
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(self.nrows);
        for b in self.column(name, SCALAR_BATCH_ROWS)? {
            push_f32(&mut out, b?.column(0), name)?;
        }
        Ok(out)
    }

    pub fn i64(&self, name: &str) -> Result<Vec<i64>> {
        let mut out = Vec::with_capacity(self.nrows);
        for b in self.column(name, SCALAR_BATCH_ROWS)? {
            push_i64(&mut out, b?.column(0), name)?;
        }
        Ok(out)
    }

    pub fn i32(&self, name: &str) -> Result<Vec<i32>> {
        let mut out = Vec::with_capacity(self.nrows);
        for b in self.column(name, SCALAR_BATCH_ROWS)? {
            push_i32(&mut out, b?.column(0), name)?;
        }
        Ok(out)
    }

    pub fn u32(&self, name: &str) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(self.nrows);
        for b in self.column(name, SCALAR_BATCH_ROWS)? {
            push_u32(&mut out, b?.column(0), name)?;
        }
        Ok(out)
    }

    pub fn bool(&self, name: &str) -> Result<Vec<bool>> {
        let mut out = Vec::with_capacity(self.nrows);
        for b in self.column(name, SCALAR_BATCH_ROWS)? {
            push_bool(&mut out, b?.column(0), name)?;
        }
        Ok(out)
    }

    pub fn str(&self, name: &str) -> Result<Vec<String>> {
        let mut out = Vec::with_capacity(self.nrows);
        for b in self.column(name, SCALAR_BATCH_ROWS)? {
            push_str(&mut out, b?.column(0), name)?;
        }
        Ok(out)
    }

    pub fn opt_f64(&self, name: &str) -> Result<Vec<Option<f64>>> {
        let mut out = Vec::with_capacity(self.nrows);
        for b in self.column(name, SCALAR_BATCH_ROWS)? {
            push_opt_f64(&mut out, b?.column(0), name)?;
        }
        Ok(out)
    }

    /// Read an f32 list column (`List` or `LargeList`) as one `Vec` per row.
    pub fn list_f32(&self, name: &str) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(self.nrows);
        for b in self.column(name, LIST_BATCH_ROWS)? {
            push_list_f32(&mut out, b?.column(0), name)?;
        }
        Ok(out)
    }

    /// Read an f32 list column into one flat values buffer plus `nrows + 1` offsets
    /// (row `r` is `values[offsets[r]..offsets[r + 1]]`): one allocation for the whole
    /// column instead of one per row, which is what makes a chromatogram table with tens
    /// of millions of short traces affordable to hold.
    pub fn list_f32_flat(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let mut offsets = Vec::with_capacity(self.nrows + 1);
        let mut values = Vec::new();
        for b in self.column(name, LIST_BATCH_ROWS)? {
            push_list_f32_flat(&mut offsets, &mut values, b?.column(0), name)?;
        }
        if offsets.is_empty() {
            offsets.push(0);
        }
        Ok((offsets, values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_mixed_columns() {
        let dir = std::env::temp_dir().join("mumdia_table_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.parquet");
        let p = path.to_str().unwrap();
        let n = write_table(
            p,
            vec![
                Col::U32("id".into(), vec![1, 2, 3]),
                Col::F64("mz".into(), vec![100.0, 200.5, 300.25]),
                Col::Str("name".into(), vec!["a".into(), "b".into(), "c".into()]),
                Col::OptF64("cal".into(), vec![Some(1.0), None, Some(3.0)]),
                Col::ListF32("trace".into(), vec![vec![1.0, 2.0], vec![], vec![9.0]]),
                Col::LargeListF32("big".into(), vec![vec![5.0], vec![6.0, 7.0], vec![]]),
            ],
        )
        .unwrap();
        assert_eq!(n, 3);
        let t = Table::read(p).unwrap();
        assert_eq!(t.nrows, 3);
        assert_eq!(t.u32("id").unwrap(), vec![1, 2, 3]);
        assert_eq!(t.f64("mz").unwrap()[2], 300.25);
        assert_eq!(t.str("name").unwrap()[1], "b");
        assert_eq!(t.opt_f64("cal").unwrap(), vec![Some(1.0), None, Some(3.0)]);
        assert_eq!(t.list_f32("trace").unwrap()[0], vec![1.0, 2.0]);
        assert!(t.list_f32("trace").unwrap()[1].is_empty());
        // LargeListF32 (64-bit offsets) reads back through the same list_f32 path.
        assert_eq!(t.list_f32("big").unwrap()[1], vec![6.0, 7.0]);
        assert!(t.list_f32("big").unwrap()[2].is_empty());
    }
}

#[cfg(test)]
mod projection_tests {
    use super::*;

    /// `read_cols` must return byte-identical data to `read` for the columns it selects.
    /// Projection is a read-path optimisation; if it altered values, every stage that
    /// adopts it would silently change results.
    #[test]
    fn read_cols_matches_read_for_selected_columns() {
        let dir = std::env::temp_dir().join("mumdia_projection_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.parquet");
        let p = path.to_str().unwrap();
        write_table(
            p,
            vec![
                Col::U32("id".into(), vec![1, 2, 3]),
                Col::F64("keep_f64".into(), vec![1.5, -2.25, f64::MIN_POSITIVE]),
                Col::Str(
                    "keep_str".into(),
                    vec!["a".into(), "".into(), "yz^2".into()],
                ),
                Col::F32("skip_me".into(), vec![9.0, 9.0, 9.0]),
                Col::ListF32("keep_list".into(), vec![vec![1.0, 2.0], vec![], vec![3.5]]),
            ],
        )
        .unwrap();

        let full = Table::read(p).unwrap();
        let proj = Table::read_cols(p, &["id", "keep_f64", "keep_str", "keep_list"]).unwrap();

        assert_eq!(full.nrows, proj.nrows);
        assert_eq!(full.u32("id").unwrap(), proj.u32("id").unwrap());
        assert_eq!(full.f64("keep_f64").unwrap(), proj.f64("keep_f64").unwrap());
        assert_eq!(full.str("keep_str").unwrap(), proj.str("keep_str").unwrap());
        assert_eq!(
            full.list_f32("keep_list").unwrap(),
            proj.list_f32("keep_list").unwrap()
        );
        // The unprojected column is absent, not silently zero-filled.
        assert!(proj.f32("skip_me").is_err());
        assert!(full.f32("skip_me").is_ok());
        std::fs::remove_file(p).ok();
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    fn tmp(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("mumdia_table_stream_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_str().unwrap().to_string()
    }

    fn mixed_cols(n: usize) -> Vec<Col> {
        let f = |i: usize| i as f64 * 1.5 - 3.0;
        vec![
            Col::U32("id".into(), (0..n as u32).collect()),
            Col::I32("z".into(), (0..n).map(|i| i as i32 - 2).collect()),
            Col::I64(
                "big".into(),
                (0..n).map(|i| i as i64 * 1_000_000_007).collect(),
            ),
            Col::F64("mz".into(), (0..n).map(f).collect()),
            Col::F32("irt".into(), (0..n).map(|i| f(i) as f32).collect()),
            Col::Bool("flag".into(), (0..n).map(|i| i % 3 == 0).collect()),
            Col::Str(
                "name".into(),
                (0..n)
                    .map(|i| {
                        if i % 4 == 1 {
                            String::new()
                        } else {
                            format!("y{i}^2")
                        }
                    })
                    .collect(),
            ),
            Col::OptF64(
                "cal".into(),
                (0..n)
                    .map(|i| if i % 5 == 2 { None } else { Some(f(i)) })
                    .collect(),
            ),
            Col::OptStr(
                "note".into(),
                (0..n)
                    .map(|i| {
                        if i % 2 == 0 {
                            None
                        } else {
                            Some(format!("n{i}"))
                        }
                    })
                    .collect(),
            ),
            Col::ListF32(
                "trace".into(),
                (0..n)
                    .map(|i| (0..(i % 7)).map(|k| k as f32 + i as f32).collect())
                    .collect(),
            ),
            Col::LargeListF32(
                "rt".into(),
                (0..n)
                    .map(|i| (0..(i % 5)).map(|k| k as f32 * 0.5).collect())
                    .collect(),
            ),
        ]
    }

    /// `TableFile` getters must decode exactly what `Table` getters decode, including the
    /// null policy (NaN / "" / None) and both list encodings, across batch boundaries.
    #[test]
    fn table_file_matches_table() {
        let p = tmp("mixed.parquet");
        let n = 10_000; // several 4096-row list batches
        write_table(&p, mixed_cols(n)).unwrap();
        let t = Table::read(&p).unwrap();
        let f = TableFile::open(&p).unwrap();
        assert_eq!(f.nrows, n);
        assert_eq!(f.nrows, t.nrows);
        assert_eq!(f.column_names(), t.column_names());
        assert!(f.has_column("mz") && !f.has_column("nope"));
        assert_eq!(f.u32("id").unwrap(), t.u32("id").unwrap());
        assert_eq!(f.i32("z").unwrap(), t.i32("z").unwrap());
        assert_eq!(f.i64("big").unwrap(), t.i64("big").unwrap());
        assert_eq!(f.f64("mz").unwrap(), t.f64("mz").unwrap());
        assert_eq!(f.f32("irt").unwrap(), t.f32("irt").unwrap());
        assert_eq!(f.bool("flag").unwrap(), t.bool("flag").unwrap());
        assert_eq!(f.str("name").unwrap(), t.str("name").unwrap());
        assert_eq!(f.str("note").unwrap(), t.str("note").unwrap());
        assert_eq!(f.opt_f64("cal").unwrap(), t.opt_f64("cal").unwrap());
        // Nullable f64 through the plain getter: nulls are NaN on both paths.
        let (a, b) = (f.f64("cal").unwrap(), t.f64("cal").unwrap());
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(a[2].is_nan());
        assert_eq!(f.list_f32("trace").unwrap(), t.list_f32("trace").unwrap());
        assert_eq!(f.list_f32("rt").unwrap(), t.list_f32("rt").unwrap());
        // Flat layout is the same data.
        let rows = t.list_f32("trace").unwrap();
        let (off, val) = f.list_f32_flat("trace").unwrap();
        assert_eq!(off.len(), n + 1);
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(&val[off[r]..off[r + 1]], row.as_slice());
        }
        // Missing column: same error wording as Table.
        assert_eq!(
            f.f64("nope").unwrap_err().to_string(),
            t.f64("nope").unwrap_err().to_string()
        );
        // Wrong type: same error wording as Table.
        assert_eq!(
            f.f64("id").unwrap_err().to_string(),
            t.f64("id").unwrap_err().to_string()
        );
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn batches_project_in_file_order_and_reject_unknown_columns() {
        let p = tmp("proj.parquet");
        write_table(&p, mixed_cols(100)).unwrap();
        let f = TableFile::open(&p).unwrap();
        // Requested out of file order; delivered in file order (id before mz before trace).
        let r = f.batches(Some(&["trace", "id", "mz"]), 30).unwrap();
        assert_eq!(
            r.schema()
                .fields()
                .iter()
                .map(|x| x.name().clone())
                .collect::<Vec<_>>(),
            vec!["id", "mz", "trace"]
        );
        let mut rows = 0;
        let mut nb = 0;
        for b in r {
            let b = b.unwrap();
            assert_eq!(b.num_columns(), 3);
            assert!(b.num_rows() <= 30);
            rows += b.num_rows();
            nb += 1;
        }
        assert_eq!(rows, 100);
        assert_eq!(nb, 4);
        assert!(f.batches(Some(&["id", "missing"]), 30).is_err());
        // for_each_batch sees the same rows.
        let mut seen = 0;
        f.for_each_batch(None, 64, |b| {
            seen += b.num_rows();
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, 100);
        std::fs::remove_file(&p).ok();
    }

    fn slice_cols(cols: Vec<Col>, a: usize, b: usize) -> Vec<Col> {
        cols.into_iter()
            .map(|c| match c {
                Col::U32(nm, v) => Col::U32(nm, v[a..b].to_vec()),
                Col::I32(nm, v) => Col::I32(nm, v[a..b].to_vec()),
                Col::I64(nm, v) => Col::I64(nm, v[a..b].to_vec()),
                Col::F64(nm, v) => Col::F64(nm, v[a..b].to_vec()),
                Col::F32(nm, v) => Col::F32(nm, v[a..b].to_vec()),
                Col::Bool(nm, v) => Col::Bool(nm, v[a..b].to_vec()),
                Col::Str(nm, v) => Col::Str(nm, v[a..b].to_vec()),
                Col::OptF64(nm, v) => Col::OptF64(nm, v[a..b].to_vec()),
                Col::OptF32(nm, v) => Col::OptF32(nm, v[a..b].to_vec()),
                Col::OptI32(nm, v) => Col::OptI32(nm, v[a..b].to_vec()),
                Col::OptStr(nm, v) => Col::OptStr(nm, v[a..b].to_vec()),
                Col::ListF32(nm, v) => Col::ListF32(nm, v[a..b].to_vec()),
                Col::ListF64(nm, v) => Col::ListF64(nm, v[a..b].to_vec()),
                Col::LargeListF32(nm, v) => Col::LargeListF32(nm, v[a..b].to_vec()),
            })
            .collect()
    }

    /// Chunked writes must read back identical to one `write_table` call.
    #[test]
    fn table_writer_chunks_match_write_table() {
        let p_once = tmp("once.parquet");
        let p_chunk = tmp("chunk.parquet");
        let n = 9_000;
        write_table(&p_once, mixed_cols(n)).unwrap();
        let mut w = TableWriter::new(&p_chunk).with_row_group_rows(1_000);
        // Slice the same columns into uneven chunks, including empty ones.
        let bounds = [0usize, 0, 1, 2_500, 2_500, 7_777, n];
        for k in 0..bounds.len() - 1 {
            w.write_cols(slice_cols(mixed_cols(n), bounds[k], bounds[k + 1]))
                .unwrap();
        }
        assert_eq!(w.rows(), n as u64);
        assert_eq!(w.close().unwrap(), n as u64);
        let once = Table::read(&p_once).unwrap();
        let chunk = Table::read(&p_chunk).unwrap();
        assert_eq!(once.schema, chunk.schema);
        assert_eq!(once.nrows, chunk.nrows);
        let n_row_groups = |path: &str| {
            ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path).unwrap())
                .unwrap()
                .metadata()
                .num_row_groups()
        };
        assert_eq!(n_row_groups(&p_once), 1);
        assert_eq!(
            n_row_groups(&p_chunk),
            9,
            "1,000-row groups over 9,000 rows"
        );
        let a = arrow::compute::concat_batches(&once.schema, &once.batches).unwrap();
        let b = arrow::compute::concat_batches(&chunk.schema, &chunk.batches).unwrap();
        for (i, name) in once.column_names().iter().enumerate() {
            assert_eq!(a.column(i), b.column(i), "column {name} differs");
        }
        std::fs::remove_file(&p_once).ok();
        std::fs::remove_file(&p_chunk).ok();
    }

    #[test]
    fn table_writer_rejects_schema_drift_and_empty_close() {
        let p = tmp("drift.parquet");
        let mut w = TableWriter::new(&p);
        w.write_cols(vec![Col::U32("id".into(), vec![1, 2])])
            .unwrap();
        // Different type for the same name.
        assert!(w.write_cols(vec![Col::I32("id".into(), vec![3])]).is_err());
        // Extra column.
        assert!(w
            .write_cols(vec![
                Col::U32("id".into(), vec![3]),
                Col::F64("x".into(), vec![1.0])
            ])
            .is_err());
        assert_eq!(w.close().unwrap(), 2);
        assert!(TableWriter::new(&tmp("never.parquet")).close().is_err());
        std::fs::remove_file(&p).ok();
    }
}
