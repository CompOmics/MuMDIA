//! A small typed column/table layer over Arrow + Parquet so each stage writes
//! and reads its declared schema without hand-rolling RecordBatches.
//!
//! Tables are written as Parquet (SNAPPY), the open, self-describing interstage
//! format (docs/03_io_layer.md). Ion-mobility and other conditional columns are
//! nullable via the `Opt*` variants (docs/02_config_and_data_model.md
//! missing-value policy).

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float32Builder, Float64Array, Int32Array,
    Int64Array, LargeListArray, LargeListBuilder, ListArray, ListBuilder, StringArray, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
    RowSelector,
};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, LogicalType};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::properties::WriterProperties;
use parquet::file::statistics::Statistics;

/// The codec every artifact is written with. Snappy by default, which is what released
/// artifacts use and what the sidecars' pyarrow reads without configuration;
/// `MUMDIA_PARQUET_COMPRESSION=zstd` writes zstd instead, which is much smaller on the
/// float-heavy chromatogram and feature tables and therefore that much less to write on a
/// run whose wall clock is disk-bound. Both are read transparently, whatever wrote them.
/// It changes every artifact's bytes, so two runs compared by content hash must agree on it.
fn codec() -> Compression {
    match std::env::var("MUMDIA_PARQUET_COMPRESSION")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "zstd" => Compression::ZSTD(Default::default()),
        "uncompressed" | "none" => Compression::UNCOMPRESSED,
        _ => Compression::SNAPPY,
    }
}

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

/// Validate a set of typed columns: at least one column, unique names, equal lengths.
/// Returns the row count.
///
/// Split out of [`cols_to_batch`] so [`write_table`] can check the WHOLE column set once,
/// before it starts consuming the columns chunk by chunk, and still report exactly the
/// errors it always reported.
fn validate_cols(path: &str, cols: &[Col]) -> Result<usize> {
    if cols.is_empty() {
        return Err(anyhow!("write_table: no columns for {path}"));
    }
    // Reject duplicate column names: Arrow allows them but readers resolve a
    // name to the first match, silently hiding the second column.
    let mut names = std::collections::HashSet::new();
    for c in cols {
        if !names.insert(c.name()) {
            return Err(anyhow!(
                "write_table: duplicate column '{}' for {path}",
                c.name()
            ));
        }
    }
    let nrows = cols[0].len();
    for c in cols {
        if c.len() != nrows {
            return Err(anyhow!(
                "write_table: column '{}' has {} rows, expected {}",
                c.name(),
                c.len(),
                nrows
            ));
        }
    }
    Ok(nrows)
}

/// Validate a set of typed columns and turn them into one Arrow schema + record batch.
/// Shared by [`write_table`] (one batch per chunk) and [`TableWriter`] (one batch per
/// caller chunk), so both write paths declare a column identically.
fn cols_to_batch(path: &str, cols: Vec<Col>) -> Result<(Arc<Schema>, RecordBatch)> {
    validate_cols(path, &cols)?;
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
    let mut b = WriterProperties::builder().set_compression(codec());
    if let Some(n) = row_group_rows {
        b = b.set_max_row_group_row_count(Some(n.max(1)));
    }
    b.build()
}

/// Rows per internal chunk in [`write_table`].
///
/// A multiple of the parquet writer's 1,024-value write batch, so the encoder sees the
/// same sequence of mini-batches it would see from one big batch, and a divisor of the
/// writer's default 1,048,576-row row group, so the row-group boundaries are where they
/// were. 65,536 rows is ~0.5 MB for an f64 column and ~1.5 MB for a string column.
const WRITE_TABLE_CHUNK_ROWS: usize = 1 << 16;

/// One column of [`write_table`] mid-flight: the source `Vec` turned into an iterator so
/// each chunk MOVES its rows out of it. A `String` or an inner `Vec<f32>` is handed to the
/// chunk and freed with the chunk's Arrow arrays, rather than every column being copied
/// into Arrow in full while the source is still resident.
macro_rules! col_chunks {
    ($($v:ident => $t:ty),+ $(,)?) => {
        enum ColChunks { $($v(String, std::vec::IntoIter<$t>)),+ }

        impl ColChunks {
            fn of(c: Col) -> ColChunks {
                match c { $(Col::$v(n, v) => ColChunks::$v(n, v.into_iter())),+ }
            }

            /// The next `k` rows as a `Col` of the same variant and name. Fewer than `k`
            /// only when the column is exhausted, which `write_table` never asks for.
            fn take(&mut self, k: usize) -> Col {
                match self {
                    $(ColChunks::$v(n, it) => Col::$v(n.clone(), it.by_ref().take(k).collect())),+
                }
            }
        }
    };
}

col_chunks!(
    I64 => i64,
    I32 => i32,
    U32 => u32,
    F64 => f64,
    F32 => f32,
    Bool => bool,
    Str => String,
    OptF64 => Option<f64>,
    OptF32 => Option<f32>,
    OptI32 => Option<i32>,
    OptStr => Option<String>,
    ListF32 => Vec<f32>,
    ListF64 => Vec<f64>,
    LargeListF32 => Vec<f32>,
);

/// Write columns to a Parquet file. Returns the row count. All columns must
/// share the same length.
///
/// The columns are validated as a set, then encoded in [`WRITE_TABLE_CHUNK_ROWS`] chunks
/// through [`TableWriter`]. Building ONE record batch for the whole table first, as this
/// used to, meant a second full Arrow copy of every column existed beside the source
/// vectors: on a wide artifact that is a second copy of the whole table, and each column
/// is one very large heap block, which on a grouped run is what the per-process mapping
/// limit counts. Chunking holds one chunk's Arrow arrays instead.
///
/// Rows are written in order and row groups still fall where the parquet writer's default
/// 1,048,576-row maximum puts them (the writer accumulates across chunks; the chunk size
/// divides it), so the file is the same file:
/// `write_table_matches_one_batch_byte_for_byte_on_scalars` asserts exactly that, and
/// `..._row_for_row_on_lists` asserts the rows where the encoder's page boundaries are its
/// own business.
pub fn write_table(path: &str, cols: Vec<Col>) -> Result<u64> {
    let nrows = validate_cols(path, &cols)?;
    let mut chunks: Vec<ColChunks> = cols.into_iter().map(ColChunks::of).collect();
    let mut w = TableWriter::new(path);
    let mut written = 0usize;
    loop {
        let k = (nrows - written).min(WRITE_TABLE_CHUNK_ROWS);
        // A zero-row table still writes its one empty chunk, which fixes the schema, so
        // an empty artifact keeps its columns.
        w.write_cols(chunks.iter_mut().map(|c| c.take(k)).collect())?;
        written += k;
        if written >= nrows {
            break;
        }
    }
    w.close()
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
    target: Option<AtomicPath>,
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
            target: None,
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
                // Written to a sibling temp path and renamed on `close`, like every other
                // writer here: a chunked artifact is the one most likely to be interrupted
                // part-way, and an abandoned writer takes its temp file with it.
                let target = AtomicPath::new(&self.path)?;
                let file = std::fs::File::create(target.tmp())
                    .with_context(|| format!("creating {}", target.tmp().display()))?;
                self.target = Some(target);
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
        if let Some(t) = self.target.take() {
            t.publish()?;
        }
        Ok(self.rows)
    }
}

/// A sibling temp path for an artifact, and the rename that publishes it.
///
/// Every artifact used to be written with `File::create` directly AT its final path,
/// which truncates on open. Three consequences, all real:
///
/// - a rerun destroyed the previous good artifact BEFORE producing a replacement, so
///   an interrupted rerun left neither;
/// - Ctrl-C or a crash part-way through a multi-gigabyte write left rubble under the
///   canonical name. The parquet footer is written last and every reader requires it,
///   so such a file fails to open rather than reading as a short table, but a
///   directory of unopenable artifacts is still worse than a directory of intact ones;
/// - nothing distinguished "this run wrote it" from "a previous run left it".
///
/// Writing to `<path>.tmp-<pid>-<n>` and renaming on success addresses all three: the
/// rename is atomic on both POSIX and Windows for a same-directory target, so a reader
/// sees either the old artifact or the new one, never a partial one. A killed run
/// leaves at most a recognisable `.tmp-<pid>-<n>` file, which is inert.
///
/// Two guarantees this makes, and one it does not (docs/29 #4): the destination is never
/// removed before the rename, so a failed publication leaves the previous artifact in
/// place; and `n` is a process-wide counter, so two writers for one destination in one
/// process cannot share a temporary file. It does not promise that the rename succeeds
/// while another Windows process holds the destination open without delete sharing;
/// then the rename fails and the old file stays, which is the first guarantee at work.
static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub struct AtomicPath {
    tmp: std::path::PathBuf,
    final_path: std::path::PathBuf,
    published: bool,
}

impl AtomicPath {
    pub fn new(path: &str) -> Result<AtomicPath> {
        let final_path = std::path::PathBuf::from(path);
        if let Some(parent) = final_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating output directory {}", parent.display()))?;
            }
        }
        let n = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = std::path::PathBuf::from(format!("{path}.tmp-{}-{n}", std::process::id()));
        Ok(AtomicPath {
            tmp,
            final_path,
            published: false,
        })
    }

    pub fn tmp(&self) -> &std::path::Path {
        &self.tmp
    }

    /// Move the completed temp file onto the final path.
    pub fn publish(mut self) -> Result<()> {
        // `std::fs::rename` replaces an existing destination FILE on POSIX and, through
        // `MoveFileExW(MOVEFILE_REPLACE_EXISTING)`, on Windows, so the destination is not
        // removed first. Removing it first meant a rename that then failed had already
        // destroyed the previous result, and gave every reader a window with no file at
        // the final path at all (docs/29 #4). Now a failed rename is an error with the
        // previous artifact still where it was.
        std::fs::rename(&self.tmp, &self.final_path).with_context(|| {
            format!(
                "publishing {} -> {}",
                self.tmp.display(),
                self.final_path.display()
            )
        })?;
        self.published = true;
        Ok(())
    }
}

impl Drop for AtomicPath {
    fn drop(&mut self) {
        // Abandoned write (an error return or an unwind): take the rubble with us so
        // the output directory does not accumulate temp files.
        if !self.published {
            let _ = std::fs::remove_file(&self.tmp);
        }
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
    target: Option<AtomicPath>,
}

impl BatchWriter {
    pub fn new(path: &str, schema: Arc<Schema>) -> Result<BatchWriter> {
        Self::open(path, schema, None)
    }

    /// Like [`BatchWriter::new`], with row groups capped at `rows` rows.
    ///
    /// A reader that iterates the file in batches decodes one row group at a time, so on
    /// a wide table the row-group size IS the reader's working set: at 387 f32 columns
    /// the parquet default of 1,048,576 rows is 1.6 GB decoded per group, regardless of
    /// how small the batches it asks for are. Batches larger than `rows` are split.
    pub fn with_row_group_rows(
        path: &str,
        schema: Arc<Schema>,
        rows: usize,
    ) -> Result<BatchWriter> {
        Self::open(path, schema, Some(rows))
    }

    fn open(path: &str, schema: Arc<Schema>, row_group_rows: Option<usize>) -> Result<BatchWriter> {
        let target = AtomicPath::new(path)?;
        let file = std::fs::File::create(target.tmp())
            .with_context(|| format!("creating {}", target.tmp().display()))?;
        Ok(BatchWriter {
            writer: Some(ArrowWriter::try_new(
                file,
                schema,
                Some(snappy_props(row_group_rows)),
            )?),
            rows: 0,
            target: Some(target),
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
        if let Some(t) = self.target.take() {
            t.publish()?;
        }
        Ok(self.rows)
    }
}

pub fn write_batches(path: &str, schema: Arc<Schema>, batches: &[RecordBatch]) -> Result<u64> {
    let target = AtomicPath::new(path)?;
    let file = std::fs::File::create(target.tmp())
        .with_context(|| format!("creating {}", target.tmp().display()))?;
    let props = WriterProperties::builder().set_compression(codec()).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    let mut n = 0u64;
    for b in batches {
        writer.write(b)?;
        n += b.num_rows() as u64;
    }
    writer.close()?;
    target.publish()?;
    Ok(n)
}

/// A read-back table: all batches concatenated logically, accessed by column
/// name with typed getters.
pub struct Table {
    pub schema: Arc<Schema>,
    pub batches: Vec<RecordBatch>,
    pub nrows: usize,
    /// The file this was read from, kept only for error messages.
    ///
    /// Without it, a missing or mistyped column reported `column 'x' not found in [...]`
    /// and dumped up to 390 column names with no indication of WHICH artifact was being
    /// read -- in a pipeline where several tables share most of their schema.
    path: String,
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
            path: path.to_string(),
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
            .map_err(|_| missing_column(name, &self.path, &self.column_names()))
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

    /// A string column as `column == value`, row by row: exactly
    /// `self.str(name)?.iter().map(|s| s == value).collect()`, without building the
    /// `String`s. See [`TableFile::str_eq`] for what that costs on a two-valued column.
    pub fn str_eq(&self, name: &str, value: &str) -> Result<Vec<bool>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            push_str_eq(&mut out, b.column(i), name, value)?;
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

/// The error for a column the table does not have: the file first, then the columns it
/// does have. A long list is still useful, but only once the reader knows which artifact
/// produced it. Shared by the whole-table and the streaming readers so the wording is one.
fn missing_column(name: &str, path: &str, have: &[String]) -> anyhow::Error {
    let shown = if have.len() > 24 {
        format!("{:?} ... and {} more", &have[..24], have.len() - 24)
    } else {
        format!("{have:?}")
    };
    anyhow!(
        "column '{name}' not found in {path} ({} columns: {shown})",
        have.len()
    )
}

/// Reject a NULL in a column read through a non-optional accessor.
///
/// The `Col` enum has `OptF64`, `OptF32`, `OptI32` and `OptStr` for genuinely nullable
/// columns, so a plain accessor is a statement that the column is required. It did not
/// behave like one: `i32`/`i64`/`u32` pushed `a.value(k)` for a null row, which is the
/// raw buffer value and in practice 0, and `str` pushed an empty `String`. Both are
/// silent substitutions of a plausible value for a missing one. The concrete
/// consequence: a null `charge` in an imported library became 0, and a 0 charge reaches
/// an isotope-spacing division in extract. `bool` did not check nulls at all. Shared by
/// the whole-table and the streaming accessors, which is why it is a free function.
fn reject_null(name: &str, row: usize) -> anyhow::Error {
    anyhow!(
        "column '{name}' has a NULL at row {row}, but it is a required column. Nullable \
         columns are written through the Opt* variants and read through the opt_* \
         accessors; a NULL here would otherwise be substituted with 0 or an empty string. \
         Fix or drop the row"
    )
}

/// The same contract as `reject_null`, for a reader that walks Arrow batches itself.
///
/// The streaming library loader reads fragment columns straight from record batches, and
/// `values()` on an Arrow array is the physical buffer: it ignores the validity bitmap,
/// so a NULL reads as 0 or 0.0 and a finiteness check over it proves nothing. Call this
/// on every required column of a batch before touching its values (docs/29 #2).
/// `row_offset` is the absolute row of the batch's first element, so the message names
/// the row a person can find in the file.
pub fn require_no_nulls(
    array: &dyn arrow::array::Array,
    name: &str,
    path: &str,
    row_offset: usize,
) -> Result<()> {
    if array.null_count() == 0 {
        return Ok(());
    }
    let row = (0..array.len()).find(|&i| array.is_null(i)).unwrap_or(0);
    Err(reject_null(name, row_offset + row).context(format!("in {path}")))
}

fn push_i64(out: &mut Vec<i64>, col: &ArrayRef, name: &str) -> Result<()> {
    let a: &Int64Array = downcast(col, name, "i64")?;
    if a.null_count() == 0 {
        out.extend_from_slice(a.values());
    } else {
        for k in 0..a.len() {
            if a.is_null(k) {
                return Err(reject_null(name, out.len()));
            }
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
            if a.is_null(k) {
                return Err(reject_null(name, out.len()));
            }
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
            if a.is_null(k) {
                return Err(reject_null(name, out.len()));
            }
            out.push(a.value(k));
        }
    }
    Ok(())
}

fn push_bool(out: &mut Vec<bool>, col: &ArrayRef, name: &str) -> Result<()> {
    let a: &BooleanArray = downcast(col, name, "bool")?;
    for k in 0..a.len() {
        if a.is_null(k) {
            return Err(reject_null(name, out.len()));
        }
        out.push(a.value(k));
    }
    Ok(())
}

fn push_str(out: &mut Vec<String>, col: &ArrayRef, name: &str) -> Result<()> {
    let a: &StringArray = downcast(col, name, "utf8")?;
    for k in 0..a.len() {
        if a.is_null(k) {
            return Err(reject_null(name, out.len()));
        }
        out.push(a.value(k).to_string());
    }
    Ok(())
}

/// Flat layout: one `String` of concatenated values plus `offsets` (row `r` is
/// `data[offsets[r]..offsets[r + 1]]`). One allocation instead of one per row. Same null
/// policy as `push_str`: a NULL in a required column is an error.
fn push_str_flat(
    offsets: &mut Vec<usize>,
    data: &mut String,
    col: &ArrayRef,
    name: &str,
) -> Result<()> {
    let a: &StringArray = downcast(col, name, "utf8")?;
    if offsets.is_empty() {
        offsets.push(0);
    }
    for k in 0..a.len() {
        if a.is_null(k) {
            // `offsets` holds one entry per row pushed plus the leading 0, so this is the
            // same absolute row `push_str` would name.
            return Err(reject_null(name, offsets.len() - 1));
        }
        data.push_str(a.value(k));
        offsets.push(data.len());
    }
    Ok(())
}

/// The equality test of `push_str`'s output against `value`, without the `String` per row.
/// Same null policy: a NULL in a required column is an error, not `false`.
fn push_str_eq(out: &mut Vec<bool>, col: &ArrayRef, name: &str, value: &str) -> Result<()> {
    let a: &StringArray = downcast(col, name, "utf8")?;
    for k in 0..a.len() {
        if a.is_null(k) {
            return Err(reject_null(name, out.len()));
        }
        out.push(a.value(k) == value);
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
    /// Row count from the parquet footer, or the span's row count for a partial open.
    pub nrows: usize,
    /// The parsed parquet footer, held for the life of the handle.
    ///
    /// Every reader this handle builds reuses it instead of re-reading and re-parsing the
    /// footer from disk, which is what [`ParquetRecordBatchReaderBuilder::try_new`] does
    /// each time it is called. The footer is not small on the library tables -- a
    /// 1.7-billion-row fragment table has ~13,000 row groups, and one
    /// `ColumnChunkMetaData` with statistics per column per group -- and it was being
    /// parsed once per open, once per `row_group_stats` and once per typed getter, so
    /// reading nine columns of one band cost ten parses. Measured on a 13,000-row-group,
    /// six-column, 1.3-million-row file: 26.6 ms per parse against 118 ms for the whole f64
    /// column, so every getter was paying about a fifth of its own cost again for a footer
    /// it already had. Cloning it into a span ([`TableFile::span`]) is an `Arc` clone.
    meta: ArrowReaderMetadata,
    /// `Some` when the table was opened on a row span ([`TableFile::open_rows`]): the row
    /// groups that cover the span and the selection that trims them to it. Every getter and
    /// batch reader applies it, so a partial table behaves as a smaller file.
    selection: Option<RowSpan>,
}

/// A contiguous row span of a parquet file: the row groups that cover it, in file order,
/// and how many rows to skip at the front of the first and the back of the last.
#[derive(Clone, Debug)]
struct RowSpan {
    row_groups: Vec<usize>,
    skip_before: usize,
    take: usize,
    skip_after: usize,
}

/// Per-row-group facts a caller can plan a partial read from without decoding anything:
/// the row count and the writer's min/max statistics of one numeric column, as f64.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowGroupStats {
    pub rows: usize,
    /// `None` when the writer recorded no statistics for the column.
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl RowGroupStats {
    /// Whether every group carries statistics and the groups are in non-decreasing order
    /// of the column, i.e. the file is sorted by it at row-group granularity. That is the
    /// precondition for reading a value range as a contiguous run of row groups.
    pub fn sorted_by_column(stats: &[RowGroupStats]) -> bool {
        stats.iter().all(|s| s.min.is_some() && s.max.is_some())
            && stats.windows(2).all(|w| w[0].max <= w[1].min)
    }
}

impl TableFile {
    /// Open `path` and read its footer. No column data is decoded. The parsed footer is
    /// kept on the handle, so every getter, batch reader and [`TableFile::span`] taken
    /// from it reads the file without parsing the footer again.
    pub fn open(path: &str) -> Result<TableFile> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
        let meta = ArrowReaderMetadata::load(&file, ArrowReaderOptions::default())
            .with_context(|| format!("reading parquet footer {path}"))?;
        let nrows = meta.metadata().file_metadata().num_rows().max(0) as usize;
        let schema = meta.schema().clone();
        Ok(TableFile {
            path: path.to_string(),
            schema,
            nrows,
            meta,
            selection: None,
        })
    }

    /// Open only the rows `[first_row, first_row + n_rows)` of `path`. The row groups that
    /// cover the span are the only ones ever decoded and a row selection trims the first
    /// and last of them, so a getter costs the span's rows plus at most one row group of
    /// decoding at each end. `nrows` is `n_rows`; the span must lie inside the file.
    ///
    /// This is what lets a stage load one slice of a sorted table: a library's precursors
    /// are ordered by m/z with row-aligned ids, so an isolation-window group is a row span,
    /// and the group's search never holds the rest of the library.
    pub fn open_rows(path: &str, first_row: usize, n_rows: usize) -> Result<TableFile> {
        TableFile::open(path)?.span(first_row, n_rows)
    }

    /// The row span `[first_row, first_row + n_rows)` of an ALREADY-OPEN handle, reusing
    /// the footer this handle has already parsed. [`TableFile::open_rows`] is this plus the
    /// open, and is the right call for a single span; this is the right call for a grouped
    /// search, which cuts one library into many bands and would otherwise re-read and
    /// re-parse a ~13,000-row-group footer for each band of each stage.
    ///
    /// Spans do not nest: take every span from the whole-file handle.
    pub fn span(&self, first_row: usize, n_rows: usize) -> Result<TableFile> {
        if self.selection.is_some() {
            anyhow::bail!(
                "TableFile::span: {} is already a row span; take spans from the whole-file \
                 handle so their rows are file rows",
                self.path
            );
        }
        let path = &self.path;
        let meta: &ParquetMetaData = self.meta.metadata();
        let total = meta.file_metadata().num_rows().max(0) as usize;
        if first_row.saturating_add(n_rows) > total {
            anyhow::bail!(
                "row span {first_row}..{} lies outside {path}, which has {total} rows",
                first_row + n_rows
            );
        }
        let mut row_groups = Vec::new();
        let mut skip_before = 0usize;
        let mut covered = 0usize;
        let mut start = 0usize;
        for i in 0..meta.num_row_groups() {
            let rows = meta.row_group(i).num_rows().max(0) as usize;
            let end = start + rows;
            if n_rows > 0 && end > first_row && start < first_row + n_rows {
                if row_groups.is_empty() {
                    skip_before = first_row - start;
                }
                row_groups.push(i);
                covered += rows;
            }
            start = end;
        }
        let skip_after = covered.saturating_sub(skip_before + n_rows);
        Ok(TableFile {
            path: self.path.clone(),
            schema: self.schema.clone(),
            nrows: n_rows,
            meta: self.meta.clone(),
            selection: Some(RowSpan {
                row_groups,
                skip_before,
                take: n_rows,
                skip_after,
            }),
        })
    }

    /// Row count and min/max statistics of a numeric column per row group, in file order,
    /// from the footer this handle already holds -- no file access at all. Integer and
    /// floating columns are reported as f64; unsigned integers are read back through their
    /// logical type so a value above `i32::MAX` is not returned as its two's-complement
    /// image.
    ///
    /// On a span handle these are the statistics of the WHOLE file, as they always were:
    /// the caller plans a span from them.
    pub fn row_group_stats(&self, name: &str) -> Result<Vec<RowGroupStats>> {
        let meta: &ParquetMetaData = self.meta.metadata();
        let mut out = Vec::with_capacity(meta.num_row_groups());
        for i in 0..meta.num_row_groups() {
            let rg = meta.row_group(i);
            let col = (0..rg.num_columns())
                .map(|j| rg.column(j))
                .find(|c| c.column_descr().name() == name)
                .ok_or_else(|| missing_column(name, &self.path, &self.column_names()))?;
            let unsigned = matches!(
                col.column_descr().logical_type_ref(),
                Some(LogicalType::Integer(t)) if !t.is_signed
            );
            let (min, max) = match col.statistics() {
                Some(Statistics::Double(s)) => (s.min_opt().copied(), s.max_opt().copied()),
                Some(Statistics::Float(s)) => (
                    s.min_opt().map(|v| f64::from(*v)),
                    s.max_opt().map(|v| f64::from(*v)),
                ),
                Some(Statistics::Int32(s)) => {
                    let conv = |v: &i32| {
                        if unsigned {
                            f64::from(*v as u32)
                        } else {
                            f64::from(*v)
                        }
                    };
                    (s.min_opt().map(conv), s.max_opt().map(conv))
                }
                Some(Statistics::Int64(s)) => {
                    let conv = |v: &i64| {
                        if unsigned {
                            *v as u64 as f64
                        } else {
                            *v as f64
                        }
                    };
                    (s.min_opt().map(conv), s.max_opt().map(conv))
                }
                _ => (None, None),
            };
            out.push(RowGroupStats {
                rows: rg.num_rows().max(0) as usize,
                min,
                max,
            });
        }
        Ok(out)
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
            .map_err(|_| missing_column(name, &self.path, &self.column_names()))
    }

    /// Stream the file as record batches of at most `batch_size` rows. `columns` projects
    /// to the named root columns (all columns when `None`); a name that is not in the file
    /// is an error, unlike [`Table::read_cols`] which silently drops it. Batches carry the
    /// projected columns in FILE order, so look them up by name (`batch.schema().index_of`)
    /// rather than by the order of `columns`.
    pub fn batches(&self, columns: Option<&[&str]>, batch_size: usize) -> Result<BatchReader> {
        let file =
            std::fs::File::open(&self.path).with_context(|| format!("opening {}", self.path))?;
        // The footer this handle parsed at `open`, not a fresh parse: a typed getter is one
        // call to this, and a stage reads a dozen columns.
        let mut builder =
            ParquetRecordBatchReaderBuilder::new_with_metadata(file, self.meta.clone())
                .with_batch_size(batch_size.max(1));
        if let Some(span) = &self.selection {
            // The selection counts rows of the SELECTED row groups only, front to back, so
            // it is skip / take / skip over exactly the groups named here.
            let mut sel = Vec::with_capacity(3);
            if span.skip_before > 0 {
                sel.push(RowSelector::skip(span.skip_before));
            }
            sel.push(RowSelector::select(span.take));
            if span.skip_after > 0 {
                sel.push(RowSelector::skip(span.skip_after));
            }
            builder = builder
                .with_row_groups(span.row_groups.clone())
                .with_row_selection(RowSelection::from(sel));
        }
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

    /// A string column as `column == value`, row by row: exactly
    /// `self.str(name)?.iter().map(|s| s == value).collect()`, without building the
    /// `String`s.
    ///
    /// For the two-valued columns -- `label` is "target" or "decoy" on every row of a
    /// 203-million-row precursor table -- `str` returns one `String` per row: a 24-byte
    /// spine plus its own heap block, so about 4.9 GB of spine and 203 million live
    /// allocations to carry one bit per row. This returns the bit: one byte per row and one
    /// allocation. The null policy is `str`'s, so a NULL is still refused rather than
    /// quietly reading as `false`.
    pub fn str_eq(&self, name: &str, value: &str) -> Result<Vec<bool>> {
        let mut out = Vec::with_capacity(self.nrows);
        for b in self.column(name, SCALAR_BATCH_ROWS)? {
            push_str_eq(&mut out, b?.column(0), name, value)?;
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

    /// Read a string column into one buffer plus `nrows + 1` byte offsets (row `r` is
    /// `data[offsets[r]..offsets[r + 1]]`, always a char boundary because whole values are
    /// concatenated): one allocation for the column instead of one `String` per row.
    ///
    /// The counterpart of [`TableFile::list_f32_flat`] for the high-cardinality string
    /// columns, where an equality test will not do. `peptidoform` and `protein` on a
    /// 203-million-row precursor library are 203 million live allocations and 4.9 GB of
    /// `String` spine through [`TableFile::str`] before a byte of text; here they are one
    /// `Vec` and one `String`. Same null policy as `str`.
    pub fn str_flat(&self, name: &str) -> Result<(Vec<usize>, String)> {
        let mut offsets = Vec::with_capacity(self.nrows + 1);
        let mut data = String::new();
        for b in self.column(name, SCALAR_BATCH_ROWS)? {
            push_str_flat(&mut offsets, &mut data, b?.column(0), name)?;
        }
        if offsets.is_empty() {
            offsets.push(0);
        }
        Ok((offsets, data))
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
    fn open_rows_reads_exactly_the_span_and_stats_describe_the_groups() {
        let dir = std::env::temp_dir().join(format!("mumdia_table_rows_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("ten.parquet").to_str().unwrap().to_string();
        // Ten rows in row groups of three: 3, 3, 3, 1.
        let mut w = TableWriter::new(&p).with_row_group_rows(3);
        w.write_cols(vec![
            Col::U32("id".into(), (0..10).collect()),
            Col::F64("v".into(), (0..10).map(|i| i as f64 * 1.5).collect()),
        ])
        .unwrap();
        w.close().unwrap();

        let stats = TableFile::open(&p).unwrap().row_group_stats("v").unwrap();
        assert_eq!(
            stats.iter().map(|s| s.rows).collect::<Vec<_>>(),
            vec![3, 3, 3, 1]
        );
        assert_eq!((stats[1].min, stats[1].max), (Some(4.5), Some(7.5)));
        assert_eq!((stats[3].min, stats[3].max), (Some(13.5), Some(13.5)));
        assert!(RowGroupStats::sorted_by_column(&stats));
        let ids = TableFile::open(&p).unwrap().row_group_stats("id").unwrap();
        assert_eq!((ids[2].min, ids[2].max), (Some(6.0), Some(8.0)));

        // Rows 4..9 start inside the second group and end inside the third.
        let part = TableFile::open_rows(&p, 4, 5).unwrap();
        assert_eq!(part.nrows, 5);
        assert_eq!(part.u32("id").unwrap(), vec![4, 5, 6, 7, 8]);
        assert_eq!(part.f64("v").unwrap(), vec![6.0, 7.5, 9.0, 10.5, 12.0]);
        let mut seen = 0;
        part.for_each_batch(Some(&["id"]), 2, |b| {
            seen += b.num_rows();
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, 5);
        // A span that is exactly one whole group, an empty span, and one past the end.
        assert_eq!(
            TableFile::open_rows(&p, 3, 3).unwrap().u32("id").unwrap(),
            vec![3, 4, 5]
        );
        assert!(TableFile::open_rows(&p, 0, 0)
            .unwrap()
            .u32("id")
            .unwrap()
            .is_empty());
        assert!(TableFile::open_rows(&p, 8, 5).is_err());
    }

    #[test]
    fn roundtrip_mixed_columns() {
        // Unique per process: a fixed name races when two `cargo test` runs share a
        // machine, which is the convention docs/14 states and four other test modules
        // already follow.
        let dir = std::env::temp_dir().join(format!("mumdia_table_test_{}", std::process::id()));
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
        // Unique per process: a fixed name races when two `cargo test` runs share a
        // machine, which is the convention docs/14 states and four other test modules
        // already follow.
        let dir =
            std::env::temp_dir().join(format!("mumdia_projection_test_{}", std::process::id()));
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
mod write_chunking_tests {
    use super::*;

    fn tmp(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("mumdia_table_chunk_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_str().unwrap().to_string()
    }

    /// Scalars and a string column: one value per row per column, so the parquet encoder
    /// sees the same 1,024-value mini-batches whether the rows arrive in one batch or in
    /// chunks that are multiples of 1,024.
    fn scalar_cols(n: usize) -> Vec<Col> {
        vec![
            Col::U32("id".into(), (0..n as u32).collect()),
            Col::F64("mz".into(), (0..n).map(|i| i as f64 * 1.5 - 3.0).collect()),
            Col::F32("irt".into(), (0..n).map(|i| (i % 977) as f32).collect()),
            Col::Bool("flag".into(), (0..n).map(|i| i % 3 == 0).collect()),
            Col::Str(
                "label".into(),
                (0..n)
                    .map(|i| if i % 2 == 0 { "target" } else { "decoy" }.to_string())
                    .collect(),
            ),
            Col::OptF64(
                "cal".into(),
                (0..n)
                    .map(|i| (i % 5 != 2).then_some(i as f64 * 0.25))
                    .collect(),
            ),
        ]
    }

    fn list_cols(n: usize) -> Vec<Col> {
        vec![
            Col::U32("id".into(), (0..n as u32).collect()),
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

    /// The pre-change write path, verbatim: validate, build ONE record batch for the whole
    /// table, write it with the same Snappy properties.
    fn write_one_batch(path: &str, cols: Vec<Col>) {
        let (schema, batch) = cols_to_batch(path, cols).unwrap();
        write_batches(path, schema, &[batch]).unwrap();
    }

    /// `write_table` now encodes in 65,536-row chunks. On scalar columns that must produce
    /// the same FILE, not merely the same values: the chunk size divides the writer's
    /// 1,048,576-row row group, so the row groups fall where they fell, and it is a
    /// multiple of the 1,024-value write batch, so the pages do too.
    #[test]
    fn write_table_matches_one_batch_byte_for_byte_on_scalars() {
        let n = 3 * WRITE_TABLE_CHUNK_ROWS + 7; // several chunks and a short last one
        let chunked = tmp("scalars_chunked.parquet");
        let once = tmp("scalars_once.parquet");
        assert_eq!(write_table(&chunked, scalar_cols(n)).unwrap(), n as u64);
        write_one_batch(&once, scalar_cols(n));
        assert_eq!(
            std::fs::read(&chunked).unwrap(),
            std::fs::read(&once).unwrap(),
            "chunked write_table must produce the same parquet file as the single-batch write"
        );
        // And the row groups are the writer's default, not one per chunk.
        let meta = TableFile::open(&chunked).unwrap();
        assert_eq!(
            meta.row_group_stats("id").unwrap().len(),
            1,
            "{n} rows is one 1,048,576-row row group, chunked or not"
        );
        assert_eq!(meta.nrows, n);
        std::fs::remove_file(&chunked).ok();
        std::fs::remove_file(&once).ok();
    }

    /// List columns hold a variable number of leaf values per row, so a row-chunk boundary
    /// need not fall on a 1,024-level mini-batch boundary and the data pages could in
    /// principle be cut elsewhere. The contract there is the rows: same order, same values,
    /// same row groups. (Measured on parquet 59.3.0 the files came out byte-identical here
    /// too, at 0-6 and at 0-136 values per row, but that is the encoder's business, not a
    /// promise this layer makes.)
    #[test]
    fn write_table_matches_one_batch_row_for_row_on_lists() {
        let n = WRITE_TABLE_CHUNK_ROWS + 1_234;
        let chunked = tmp("lists_chunked.parquet");
        let once = tmp("lists_once.parquet");
        write_table(&chunked, list_cols(n)).unwrap();
        write_one_batch(&once, list_cols(n));
        let (a, b) = (Table::read(&chunked).unwrap(), Table::read(&once).unwrap());
        assert_eq!(a.schema, b.schema);
        assert_eq!(a.nrows, n);
        assert_eq!(a.u32("id").unwrap(), b.u32("id").unwrap());
        assert_eq!(a.list_f32("trace").unwrap(), b.list_f32("trace").unwrap());
        assert_eq!(a.list_f32("rt").unwrap(), b.list_f32("rt").unwrap());
        assert_eq!(
            TableFile::open(&chunked)
                .unwrap()
                .row_group_stats("id")
                .unwrap()
                .len(),
            TableFile::open(&once)
                .unwrap()
                .row_group_stats("id")
                .unwrap()
                .len()
        );
        std::fs::remove_file(&chunked).ok();
        std::fs::remove_file(&once).ok();
    }

    /// A table shorter than one chunk, and an empty one: both must still be the file the
    /// single-batch path wrote, including the schema of a zero-row artifact.
    #[test]
    fn write_table_matches_one_batch_when_short_or_empty() {
        for n in [0usize, 1, 1_000] {
            let chunked = tmp(&format!("short_chunked_{n}.parquet"));
            let once = tmp(&format!("short_once_{n}.parquet"));
            assert_eq!(write_table(&chunked, scalar_cols(n)).unwrap(), n as u64);
            write_one_batch(&once, scalar_cols(n));
            assert_eq!(
                std::fs::read(&chunked).unwrap(),
                std::fs::read(&once).unwrap(),
                "{n} rows"
            );
            let t = Table::read(&chunked).unwrap();
            assert_eq!(t.nrows, n);
            assert_eq!(
                t.column_names().len(),
                6,
                "a zero-row table keeps its columns"
            );
            std::fs::remove_file(&chunked).ok();
            std::fs::remove_file(&once).ok();
        }
    }

    /// The validation is of the whole column set, before any chunk is encoded: a length
    /// mismatch past the first chunk must still be refused, with the same message, and
    /// must not leave a partial file behind.
    #[test]
    fn write_table_validates_the_whole_column_set_before_writing() {
        let p = tmp("rejected.parquet");
        let n = WRITE_TABLE_CHUNK_ROWS + 10;
        let mut cols = scalar_cols(n);
        cols.push(Col::I32("short".into(), vec![1, 2, 3]));
        let err = write_table(&p, cols).unwrap_err().to_string();
        assert!(err.contains("'short'") && err.contains("expected"), "{err}");
        assert!(
            !std::path::Path::new(&p).exists(),
            "a refused write must not publish a file"
        );
        let dup = write_table(
            &p,
            vec![
                Col::U32("id".into(), vec![1]),
                Col::U32("id".into(), vec![2]),
            ],
        )
        .unwrap_err()
        .to_string();
        assert!(dup.contains("duplicate column 'id'"), "{dup}");
        assert!(write_table(&p, vec![])
            .unwrap_err()
            .to_string()
            .contains("no columns"));
    }
}

#[cfg(test)]
mod batch_writer_tests {
    use super::*;
    use arrow::array::{ArrayRef, Float32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::file::reader::{FileReader, SerializedFileReader};

    #[test]
    fn row_group_cap_splits_large_batches() {
        let dir =
            std::env::temp_dir().join(format!("mumdia_batch_writer_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("capped.parquet").to_string_lossy().to_string();
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("x", DataType::Float32, false),
        ]));
        let n = 10_000usize;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from((0..n as i32).collect::<Vec<_>>())) as ArrayRef,
                Arc::new(Float32Array::from(vec![1.5f32; n])) as ArrayRef,
            ],
        )
        .unwrap();
        // One 10,000-row batch, groups capped at 3,000 rows: 3,000 + 3,000 + 3,000 + 1,000.
        let mut w = BatchWriter::with_row_group_rows(&path, schema, 3_000).unwrap();
        w.write(&batch).unwrap();
        assert_eq!(w.close().unwrap(), n as u64);
        let reader = SerializedFileReader::new(std::fs::File::open(&path).unwrap()).unwrap();
        let meta = reader.metadata();
        assert_eq!(meta.num_row_groups(), 4);
        assert_eq!(meta.row_group(0).num_rows(), 3_000);
        assert_eq!(meta.row_group(3).num_rows(), 1_000);
        let _ = std::fs::remove_dir_all(&dir);
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
    /// null policy (NaN for floats, an error for required columns, None through the Opt*
    /// readers) and both list encodings, across batch boundaries.
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
        // A NULL in a column read through the plain string getter is refused on both
        // paths, with the column named: `note` is nullable and has no Opt* reader yet.
        for e in [f.str("note").unwrap_err(), t.str("note").unwrap_err()] {
            let msg = e.to_string();
            assert!(msg.contains("'note'") && msg.contains("NULL"), "{msg}");
        }
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

    /// A span taken from an open handle must be the table `open_rows` opens, column for
    /// column, and must be the matching slice of the whole table. That equality is what
    /// lets a grouped search take every band from one handle and parse the footer once
    /// instead of once per band per stage.
    #[test]
    fn span_of_an_open_handle_matches_open_rows() {
        let p = tmp("spans.parquet");
        let n = 10_000;
        let mut w = TableWriter::new(&p).with_row_group_rows(1_000);
        w.write_cols(mixed_cols(n)).unwrap();
        w.close().unwrap();

        let whole = TableFile::open(&p).unwrap();
        let all_id = whole.u32("id").unwrap();
        let all_name = whole.str("name").unwrap();
        let all_trace = whole.list_f32("trace").unwrap();
        // Empty, whole-file, inside one row group, across several, and the last row.
        for (first, len) in [(0, 0), (0, n), (999, 1), (1_500, 3_000), (n - 1, 1)] {
            let a = whole.span(first, len).unwrap();
            let b = TableFile::open_rows(&p, first, len).unwrap();
            assert_eq!(a.nrows, len);
            assert_eq!(b.nrows, len);
            assert_eq!(a.u32("id").unwrap(), b.u32("id").unwrap(), "{first}+{len}");
            assert_eq!(a.f64("mz").unwrap(), b.f64("mz").unwrap());
            assert_eq!(a.str("name").unwrap(), b.str("name").unwrap());
            assert_eq!(a.list_f32("trace").unwrap(), b.list_f32("trace").unwrap());
            // ... and the same rows as the whole table's slice.
            assert_eq!(a.u32("id").unwrap(), all_id[first..first + len]);
            assert_eq!(a.str("name").unwrap(), all_name[first..first + len]);
            assert_eq!(a.list_f32("trace").unwrap(), all_trace[first..first + len]);
        }
        // Out of range is refused with the same message as open_rows.
        let msg = |r: Result<TableFile>| r.err().expect("must be refused").to_string();
        assert_eq!(
            msg(whole.span(n - 1, 2)),
            msg(TableFile::open_rows(&p, n - 1, 2))
        );
        // Spans do not nest: a span's rows would otherwise mean two different things.
        let part = whole.span(10, 20).unwrap();
        let err = msg(part.span(0, 1));
        assert!(err.contains("already a row span"), "{err}");
        // Statistics are the whole file's on either handle, which is how a caller plans
        // a span from them.
        assert_eq!(
            part.row_group_stats("id").unwrap(),
            whole.row_group_stats("id").unwrap()
        );
        assert_eq!(whole.row_group_stats("id").unwrap().len(), 10);
        std::fs::remove_file(&p).ok();
    }

    /// The handle carries its parsed footer across getters, spans and batch readers. If
    /// anything in that plumbing went wrong it would show up as a wrong row count or a
    /// wrong column, so read every getter twice from one handle and once through a span.
    #[test]
    fn one_handle_serves_repeated_reads() {
        let p = tmp("reuse.parquet");
        let n = 3_000;
        write_table(&p, mixed_cols(n)).unwrap();
        let f = TableFile::open(&p).unwrap();
        for _ in 0..3 {
            assert_eq!(f.u32("id").unwrap().len(), n);
            assert_eq!(f.f64("mz").unwrap().len(), n);
            assert_eq!(f.str("name").unwrap().len(), n);
            assert_eq!(f.list_f32("trace").unwrap().len(), n);
            let mut seen = 0;
            f.for_each_batch(Some(&["id", "mz"]), 512, |b| {
                seen += b.num_rows();
                Ok(())
            })
            .unwrap();
            assert_eq!(seen, n);
        }
        assert_eq!(f.span(100, 50).unwrap().u32("id").unwrap()[0], 100);
        std::fs::remove_file(&p).ok();
    }

    /// `str_eq` must be `str` plus a comparison, error for error: it is offered as a
    /// drop-in for a caller that only asks whether a two-valued column equals a value.
    #[test]
    fn str_eq_is_the_string_comparison_without_the_strings() {
        let p = tmp("labels.parquet");
        let n: usize = 5_000;
        write_table(
            &p,
            vec![
                Col::U32("id".into(), (0..n as u32).collect()),
                Col::Str(
                    "label".into(),
                    (0..n)
                        .map(|i| {
                            match i % 3 {
                                0 => "target",
                                1 => "decoy",
                                _ => "target_x",
                            }
                            .to_string()
                        })
                        .collect(),
                ),
                Col::OptStr(
                    "note".into(),
                    (0..n)
                        .map(|i| (i % 7 != 4).then(|| format!("n{i}")))
                        .collect(),
                ),
            ],
        )
        .unwrap();
        let t = Table::read(&p).unwrap();
        let f = TableFile::open(&p).unwrap();
        let want: Vec<bool> = t
            .str("label")
            .unwrap()
            .iter()
            .map(|s| s == "target")
            .collect();
        assert_eq!(want.iter().filter(|b| **b).count(), n.div_ceil(3));
        assert_eq!(t.str_eq("label", "target").unwrap(), want);
        assert_eq!(f.str_eq("label", "target").unwrap(), want);
        // A prefix is not a match, and a value the column does not hold matches nothing.
        assert!(f.str_eq("label", "targe").unwrap().iter().all(|b| !b));
        assert!(f.str_eq("label", "TARGET").unwrap().iter().all(|b| !b));
        // A span reads its own rows only: rows 10..16, where 10 % 3 == 1 is the decoy.
        assert_eq!(
            f.span(10, 6).unwrap().str_eq("label", "decoy").unwrap(),
            vec![true, false, false, true, false, false]
        );
        // A NULL is refused exactly as `str` refuses it, and so are a missing column and
        // the wrong type -- same wording on both read paths.
        for (a, b) in [
            (f.str_eq("note", "n0"), f.str("note")),
            (f.str_eq("nope", "x"), f.str("nope")),
            (f.str_eq("id", "x"), f.str("id")),
        ] {
            assert_eq!(
                a.unwrap_err().to_string(),
                b.unwrap_err().to_string(),
                "str_eq must fail exactly as str does"
            );
        }
        assert_eq!(
            t.str_eq("note", "n0").unwrap_err().to_string(),
            f.str_eq("note", "n0").unwrap_err().to_string()
        );
        std::fs::remove_file(&p).ok();
    }

    /// `str_flat` must rebuild exactly what `str` returns, row for row, including empty
    /// values and multi-byte ones, and refuse a NULL the same way.
    #[test]
    fn str_flat_rebuilds_the_string_column() {
        let p = tmp("flat_str.parquet");
        let n = 10_000; // several SCALAR_BATCH_ROWS-independent batches of rows
        write_table(&p, mixed_cols(n)).unwrap();
        let f = TableFile::open(&p).unwrap();
        let rows = f.str("name").unwrap();
        let (off, data) = f.str_flat("name").unwrap();
        assert_eq!(off.len(), n + 1);
        assert_eq!(off[0], 0);
        assert_eq!(*off.last().unwrap(), data.len());
        for (r, want) in rows.iter().enumerate() {
            assert_eq!(&data[off[r]..off[r + 1]], want, "row {r}");
        }
        // A span reads its own rows only, and a NULL is refused as `str` refuses it.
        let part = f.span(3, 4).unwrap();
        let (o, d) = part.str_flat("name").unwrap();
        assert_eq!(o.len(), 5);
        assert_eq!(&d[o[0]..o[1]], &rows[3]);
        assert_eq!(
            f.str_flat("note").unwrap_err().to_string(),
            f.str("note").unwrap_err().to_string()
        );
        assert_eq!(
            f.str_flat("id").unwrap_err().to_string(),
            f.str("id").unwrap_err().to_string()
        );
        // Multi-byte values: the offsets are byte offsets, and every one of them must land
        // on a char boundary or the slicing panics.
        let u = tmp("flat_unicode.parquet");
        write_table(
            &u,
            vec![Col::Str(
                "s".into(),
                vec![
                    String::new(),
                    "\u{03bc}\u{00b2}".into(),
                    "plain".into(),
                    "\u{1f9ea}\u{00e9}".into(),
                ],
            )],
        )
        .unwrap();
        let tu = TableFile::open(&u).unwrap();
        let want = tu.str("s").unwrap();
        let (o, d) = tu.str_flat("s").unwrap();
        for (r, w) in want.iter().enumerate() {
            assert_eq!(&d[o[r]..o[r + 1]], w, "row {r}");
        }
        // An empty table still returns the leading offset.
        let e = tmp("flat_empty.parquet");
        write_table(&e, vec![Col::Str("s".into(), vec![])]).unwrap();
        let (o, d) = TableFile::open(&e).unwrap().str_flat("s").unwrap();
        assert_eq!((o, d.as_str()), (vec![0], ""));
        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&u).ok();
        std::fs::remove_file(&e).ok();
    }

    /// A grouped search hands band handles to worker threads. Holding the parsed footer on
    /// the handle must not have taken that away.
    #[test]
    fn table_handles_stay_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TableFile>();
        assert_send_sync::<Table>();
        // A reader is moved to the thread that drains it, not shared.
        fn assert_send<T: Send>() {}
        assert_send::<BatchReader>();
    }
}

#[cfg(test)]
mod atomic_path_tests {
    use super::*;

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia_atomic_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn two_writers_for_one_destination_get_different_temporary_files() {
        let d = dir("two_writers");
        let final_path = d.join("out.parquet").to_str().unwrap().to_string();
        let a = AtomicPath::new(&final_path).unwrap();
        let b = AtomicPath::new(&final_path).unwrap();
        assert_ne!(
            a.tmp(),
            b.tmp(),
            "the suffix must not be the process id alone"
        );
        drop(a);
        drop(b);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_failed_publication_leaves_the_previous_result_in_place() {
        // The destination is a non-empty directory, which a file cannot be renamed onto
        // on any platform, so publication fails. Before the fix the destination was
        // removed first and the failure left nothing behind.
        let d = dir("failed_publish");
        let final_path = d.join("out.parquet");
        std::fs::create_dir_all(&final_path).unwrap();
        std::fs::write(final_path.join("previous"), b"previous result").unwrap();
        let ap = AtomicPath::new(final_path.to_str().unwrap()).unwrap();
        std::fs::write(ap.tmp(), b"new result").unwrap();
        assert!(
            ap.publish().is_err(),
            "renaming a file onto a directory must fail"
        );
        assert!(
            final_path.join("previous").is_file(),
            "the previous result must survive a failed publication"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_successful_publication_replaces_the_previous_file() {
        let d = dir("replace");
        let final_path = d.join("out.parquet");
        let fp = final_path.to_str().unwrap().to_string();
        let first = AtomicPath::new(&fp).unwrap();
        std::fs::write(first.tmp(), b"v1").unwrap();
        first.publish().unwrap();
        let second = AtomicPath::new(&fp).unwrap();
        std::fs::write(second.tmp(), b"v2").unwrap();
        second.publish().unwrap();
        assert_eq!(std::fs::read(&final_path).unwrap(), b"v2");
        assert!(
            std::fs::read_dir(&d).unwrap().count() == 1,
            "no temporary file may remain after publication"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn require_no_nulls_names_the_column_and_the_absolute_row() {
        let a = arrow::array::Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let err = require_no_nulls(&a, "mz", "lib.parquet", 1000).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("'mz'") && msg.contains("row 1001"), "{msg}");
        let ok = arrow::array::Float64Array::from(vec![Some(1.0), Some(2.0)]);
        assert!(require_no_nulls(&ok, "mz", "lib.parquet", 0).is_ok());
    }
}
