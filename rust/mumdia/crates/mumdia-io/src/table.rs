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
use parquet::arrow::ArrowSchemaConverter;
#[cfg(test)]
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, LogicalType, Type as PhysicalType};
use parquet::column::writer::ColumnCloseResult;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};
use parquet::file::properties::WriterProperties;
use parquet::file::statistics::Statistics;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::ColumnPath;

use crate::codec::{codec_pool, ColumnEncoder};
pub use crate::report::Written;
pub use crate::span_cache::SpanReadOptions;

/// The codec every artifact is written with. Snappy by default, which is what released
/// artifacts use and what the sidecars' pyarrow reads without configuration;
/// `MUMDIA_PARQUET_COMPRESSION=zstd` writes zstd instead. Both are read transparently,
/// whatever wrote them. It changes every artifact's bytes, so two runs compared by content
/// hash must agree on it.
///
/// zstd is a DISK-FOOTPRINT lever. This docstring used to call it a wall-clock lever
/// ("that much less to write on a run whose wall clock is disk-bound"), which the
/// measurement does not support: the write is not where it helps.
///
/// Measured through `bench_rewrite_a_real_artifact` on `out_aif02/chromatograms.parquet`
/// (1,028,155 rows), parquet-rs 59.3 at `ZstdLevel::default()` = 1, which is what this
/// function selects, both arms with the float dictionaries off (see [`writer_props`]):
/// snappy 171.6 MB against zstd 70.4 MB, -59%. So it more than halves the largest artifact.
/// Neither the write nor the read time is claimed in either direction: the whole spread
/// measured (write 6.7-7.0 s, read 3.9-4.3 s) is inside the run-to-run noise of repeats of
/// the IDENTICAL arm on this host, which is 10-20%, and each arm was a single shot.
///
/// Set it where the disk is the constraint -- a network share, OneDrive, a spinning disk,
/// or a run that is out of space -- and re-measure the decode there before assuming it is
/// free on slower storage than this. On footprint, be careful which baseline a ratio is
/// applied to: -59% is zstd against snappy with the float dictionaries ALREADY limited on
/// both sides. Against the pre-change dictionary-on snappy file (216.8 MB) the zstd one is
/// 70.4/216.8 = 32.5%, so the immuno run's 136 GB of band artifacts, measured before either
/// change, would land near 44 GB rather than 60 -- and only to the extent they compress
/// like a chromatogram table, which the wide feature and competed tables in that 136 GB
/// do not.
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

/// The FLOAT and DOUBLE leaves of `schema` as (path, width), for the tests and benches.
#[cfg(test)]
fn float_leaf_paths(schema: &Schema) -> Vec<(ColumnPath, usize)> {
    float_leaves(schema)
        .into_iter()
        .map(|l| (l.path, l.width))
        .collect()
}

/// The dictionary page size limit for one FLOAT or DOUBLE leaf of a chunk of
/// `row_group_rows` rows, or `None` to leave parquet-rs's 1 MB default in place.
///
/// A dictionary pays off on CARDINALITY RELATIVE TO THE CHUNK, and an absolute byte limit
/// is the one thing that cannot express it. Measured
/// (`bench_float_dictionary_limit_by_cardinality`, f64, snappy, 65,536-row groups, against
/// parquet-rs's default), for a column whose per-row-group cardinality is a fraction `c`
/// of the chunk's rows, disabling the dictionary costs +380% at c = 0.01, +852% at 0.05,
/// +231% at 0.125, +108% at 0.25 and +34% at 0.5, breaks even at 0.75, and only gains
/// (-19.7%) at c = 1. So the fallback must fire near c = 1 and nowhere below it.
///
/// `0.5 * rows * leaf_bytes` is the shipped point: half the chunk's values, whatever they
/// weigh. The width matters and is not cosmetic -- a limit sized for f64 is 1.5 times a
/// whole f32 chunk, so every scalar f32 column would keep its dictionary unconditionally
/// and the rule would silently do nothing for `predicted_intensity`, `irt` and every other
/// f32 leaf. It is deliberately short of the c = 0.75 break-even, because the fallback is
/// PREFIX-based: the pages written before it fires keep their dictionary and cannot be
/// undone, so a fallback that fires only at the break-even recovers very little. Measured
/// at c = 1 (65,536-row groups): -19.7% for a disable, -15.1% at c = 0.25, -10.1% at
/// c = 0.5, -4.9% at c = 0.75, 0% at c = 1.
///
/// Uncapped chunks keep the 1 MB default. Their break-even would be 6 MB, so parquet's own
/// default is if anything too EAGER there -- a 1,048,576-row chunk falls back at c = 0.125,
/// where the sweep says the dictionary was still worth +231% -- but raising a limit is a
/// different change with a different memory profile and is not in this one. This function
/// only ever LOWERS the limit, so it cannot move a byte of a file written without a cap.
///
/// The 0.75 is a conservatism choice and the alternatives are measured. Over the 92 real
/// artifacts of `ci/smoke.sh`, rewritten at each artifact's own row-group size
/// (`bench_rewrite_a_real_artifact`), against parquet-rs's default:
///
/// | rule | total | worst single artifact |
/// |---|---|---|
/// | dictionary disabled on float leaves | -0.93% | +18.7% (`fragment_library_fragments`) |
/// | 16 KiB fixed limit on float leaves | +0.48% | +30.7% (same) |
/// | 16 KiB limit on EVERY leaf | +1.39% | +30.7% (same) |
/// | c = 0.25 | -1.09% | +17.4% (`scored_combined`) |
/// | **c = 0.5, shipped** | **-2.24%** | **none** |
/// | c = 0.75 | -2.19% | none |
///
/// c = 0.5 and c = 0.75 are the only two that regress nothing, and they tie on this set;
/// c = 0.5 is taken because it is twice as good on the shape the change exists for
/// (`bench_dictionary_rules_on_a_features_shaped_table`: -10.1% against -4.9%). Its cost
/// is a wider synthetic loss band, 0.5 < c < 0.9, peaking at +50.6%, against 0.75 < c < 0.95
/// peaking at +18.4%. No artifact measured here falls in either band, but neither set is
/// production-scale; re-derive from the two benches before moving it.
fn float_dictionary_page_size_limit(
    row_group_rows: Option<usize>,
    leaf_bytes: usize,
) -> Option<usize> {
    row_group_rows
        .map(|rows| rows.saturating_mul(leaf_bytes) / 2)
        .filter(|&limit| limit < parquet::file::properties::DEFAULT_DICTIONARY_PAGE_SIZE_LIMIT)
}

/// The most rows a capped writer puts in one data page (see [`writer_props`], "PAGES").
///
/// A column writer holds its open page's values until the page is cut, so a larger row
/// limit is a larger in-progress buffer per column. Measured through the writer's own
/// `memory_size` (`bench_rewrite_a_real_artifact`, peak over the write): the 398-column
/// competed table in one 131,072-row group went from 551.9 to 620.2 MB (+12%), the features
/// table at 65,536-row groups from 327.8 to 316.3 MB (fewer page headers outweigh the
/// buffer), the chromatograms from 24.3 to 24.9 MB. Every cap the engine sets is at most
/// this value, so the bound only stops a future, larger cap from buffering a whole
/// multi-million-row chunk per column.
const MAX_DATA_PAGE_ROWS: usize = 1 << 17;

/// Above this fraction of distinct values in the planning sample, a float leaf of a capped
/// writer is written without a dictionary at all ([`EncodingPlan`]).
///
/// The break-even of a disable against a full dictionary is c = 0.75 of the chunk's values
/// distinct, and a disable gains 19.7% at c = 1 and costs 34% at c = 0.5
/// ([`float_dictionary_page_size_limit`] has the sweep). The threshold is applied to the
/// sample, a quarter of a row group ([`PLAN_SAMPLE_FRACTION`]), so it only means the same c
/// when a leaf's repeats scale with the rows, which is what the engine's float columns look
/// like: a few values repeated often (zeros, a noise floor, a clamp) and a near-unique rest.
/// Measured on the AIF artifacts rewritten at their own row-group sizes
/// (`bench_rewrite_a_real_artifact`), 0.8 is the best of the three thresholds tried on
/// every table, against the unplanned layout:
///
/// | artifact (row group) | 0.8 | 0.9 | 0.95 |
/// |---|---|---|---|
/// | features (65,536) | -8.6% | -7.9% | -7.3% |
/// | psms_competed (131,072) | -14.9% | -13.8% | -12.6% |
/// | chromatograms (65,536) | -6.3% | -5.8% | -5.8% |
/// | spectra_ms2 (2,048) | -0.2% | +1.8% | +1.8% |
///
/// The case it can get wrong is a leaf drawn evenly from a vocabulary of about twice the
/// sample's rows: its sample is 80% distinct while the whole chunk is only about 46%, where
/// the disable costs about 40%. No measured column has that shape; a quantised feature
/// that does would show up in the bench as a leaf the plan lost on.
const PLAN_PLAIN_ABOVE_DISTINCT: f64 = 0.8;

/// A sample with fewer non-null values of a leaf than this does not disable that leaf's
/// dictionary: the distinct fraction of a handful of values says little about a row group.
const PLAN_MIN_VALUES: usize = 4_096;

/// A capped writer plans its float encodings from the first `cap / PLAN_SAMPLE_FRACTION`
/// rows it is given, whatever the chunks they arrive in.
const PLAN_SAMPLE_FRACTION: usize = 4;

/// The rows a writer capped at `row_group_rows` samples before it plans.
fn plan_sample_rows(row_group_rows: usize) -> usize {
    row_group_rows.div_ceil(PLAN_SAMPLE_FRACTION).max(1)
}

/// Whether capped writers plan their float encodings. On unless `MUMDIA_PARQUET_PLAN` is
/// `0`, `off`, `false` or `no`, which restores the unplanned layout of [`writer_props`] for a
/// byte comparison against a binary from before the plan.
fn plan_enabled() -> bool {
    !matches!(
        std::env::var("MUMDIA_PARQUET_PLAN")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "off" | "false" | "no"
    )
}

/// One FLOAT or DOUBLE parquet leaf of an Arrow schema: its path as the writer spells it,
/// its value width, and the Arrow root column it lives under.
struct FloatLeaf {
    path: ColumnPath,
    width: usize,
    root: usize,
}

/// The parquet leaves of `schema` whose physical type is FLOAT or DOUBLE, as the writer
/// will name them, with their Arrow root columns. Derived with the same converter the
/// [`ArrowWriter`](parquet::arrow::ArrowWriter) uses, so a list column's leaf path (`trace.list.item`) is the writer's own
/// spelling rather than a guess. A schema the converter rejects yields no leaves; the writer
/// reports that failure itself.
fn float_leaves(schema: &Schema) -> Vec<FloatLeaf> {
    let Ok(desc) = ArrowSchemaConverter::new()
        .with_coerce_types(false)
        .convert(schema)
    else {
        return Vec::new();
    };
    (0..desc.num_columns())
        .filter_map(|i| {
            let c = desc.column(i);
            let width = match c.physical_type() {
                PhysicalType::FLOAT => 4,
                PhysicalType::DOUBLE => 8,
                _ => return None,
            };
            Some(FloatLeaf {
                path: c.path().clone(),
                width,
                root: desc.get_column_root_idx(i),
            })
        })
        .collect()
}

/// What a capped writer learned about one float leaf from its first rows.
#[derive(Clone, Debug, PartialEq)]
struct LeafPlan {
    path: ColumnPath,
    /// Rows of the sample.
    rows: usize,
    /// Non-null values of the leaf in the sample: at most one per row for a scalar, the
    /// summed list lengths for a list leaf.
    values: usize,
    /// At least [`PLAN_MIN_VALUES`] values, of which more than [`PLAN_PLAIN_ABOVE_DISTINCT`]
    /// are distinct.
    near_unique: bool,
}

impl LeafPlan {
    /// Values of the leaf per row, rounded up and at least one.
    fn values_per_row(&self) -> usize {
        self.values.div_ceil(self.rows.max(1)).max(1)
    }
}

/// The float encodings of a capped writer, planned from the first rows it is given
/// (docs/03_io_layer.md, "Float encodings planned from the first rows").
///
/// [`writer_props`] decides each float leaf's dictionary before a single value is seen, so
/// it sizes a dictionary LIMIT from the row-group cap and lets parquet fall back to PLAIN
/// when the limit fills. The fallback is prefix-based: the pages written before it fires
/// keep their dictionary, so a near-unique column still pays for a dictionary page and for
/// the index pages that address it. And the limit counts ROWS, which is wrong for a list
/// leaf, whose chunk holds a list's worth of values per row: the chromatogram `rt` and
/// `intensity` leaves (about 50 values a row) had their dictionary cut at 128 KB of a 13 MB
/// chunk, where the dictionary was worth keeping.
///
/// The plan looks at the first [`plan_sample_rows`] rows before the first byte is encoded:
///
/// * a leaf whose sampled values are near-unique ([`PLAN_PLAIN_ABOVE_DISTINCT`]) is written
///   PLAIN from its first page, with no dictionary to pay for;
/// * every other float leaf keeps the dictionary limit of [`writer_props`], sized from the
///   leaf's VALUES per row group (the sampled values per row times the cap) instead of its
///   rows, so a list leaf keeps parquet's 1 MB default and a scalar leaf is as before.
///
/// The sample is taken by rows, not by chunks: the same rows written in any chunking plan
/// the same encodings. Distinct values are counted by bit pattern, which is how parquet's
/// dictionary interns floats, and a distinct COUNT does not depend on hash iteration order,
/// so a plan is a deterministic function of the first rows.
#[derive(Clone, Debug, Default, PartialEq)]
struct EncodingPlan {
    leaves: Vec<LeafPlan>,
}

impl EncodingPlan {
    /// Plan the float leaves of `schema` from the first `sample_rows` rows of `batches`.
    fn of(schema: &Schema, batches: &[RecordBatch], sample_rows: usize) -> EncodingPlan {
        Self::with_threshold(schema, batches, sample_rows, PLAN_PLAIN_ABOVE_DISTINCT)
    }

    /// [`EncodingPlan::of`] at another distinct-fraction threshold (the benches sweep it).
    fn with_threshold(
        schema: &Schema,
        batches: &[RecordBatch],
        sample_rows: usize,
        threshold: f64,
    ) -> EncodingPlan {
        let mut sample: Vec<RecordBatch> = Vec::new();
        let mut rows = 0usize;
        for b in batches {
            if rows >= sample_rows {
                break;
            }
            let take = b.num_rows().min(sample_rows - rows);
            if take > 0 {
                sample.push(b.slice(0, take));
                rows += take;
            }
        }
        let leaves = float_leaves(schema)
            .into_iter()
            .filter_map(|leaf| {
                let mut bits: Vec<u64> = Vec::new();
                for b in &sample {
                    push_float_bits(b.column(leaf.root), &mut bits)?;
                }
                Some(LeafPlan {
                    path: leaf.path,
                    rows,
                    values: bits.len(),
                    near_unique: near_unique(&bits, threshold),
                })
            })
            .collect();
        EncodingPlan { leaves }
    }

    fn leaf(&self, path: &ColumnPath) -> Option<&LeafPlan> {
        self.leaves.iter().find(|l| &l.path == path)
    }
}

/// Append the bit patterns of the non-null float values under `col` (a Float32 or Float64
/// array, or a List or LargeList of one) to `out`. `None` for any other shape, whose leaf
/// then keeps the unplanned rule.
fn push_float_bits(col: &ArrayRef, out: &mut Vec<u64>) -> Option<()> {
    fn flat(values: &ArrayRef, out: &mut Vec<u64>) -> Option<()> {
        match values.data_type() {
            DataType::Float64 => {
                let a = values.as_any().downcast_ref::<Float64Array>()?;
                out.extend(a.iter().flatten().map(f64::to_bits));
            }
            DataType::Float32 => {
                let a = values.as_any().downcast_ref::<Float32Array>()?;
                out.extend(a.iter().flatten().map(|v| u64::from(v.to_bits())));
            }
            _ => return None,
        }
        Some(())
    }
    match col.data_type() {
        DataType::Float64 | DataType::Float32 => flat(col, out),
        DataType::List(_) => {
            let l = col.as_any().downcast_ref::<ListArray>()?;
            let offsets = l.value_offsets();
            let (lo, hi) = (*offsets.first()? as usize, *offsets.last()? as usize);
            flat(&l.values().slice(lo, hi - lo), out)
        }
        DataType::LargeList(_) => {
            let l = col.as_any().downcast_ref::<LargeListArray>()?;
            let offsets = l.value_offsets();
            let (lo, hi) = (*offsets.first()? as usize, *offsets.last()? as usize);
            flat(&l.values().slice(lo, hi - lo), out)
        }
        _ => None,
    }
}

/// Whether `bits` holds at least [`PLAN_MIN_VALUES`] values of which more than `threshold`
/// ([`PLAN_PLAIN_ABOVE_DISTINCT`] in the writers) are distinct. The count stops as soon as
/// too many repeats have been seen to pass, so a low-cardinality leaf costs a fraction of a
/// pass.
fn near_unique(bits: &[u64], threshold: f64) -> bool {
    if bits.len() < PLAN_MIN_VALUES {
        return false;
    }
    let allowed_repeats = ((1.0 - threshold) * bits.len() as f64) as usize;
    let mut seen = std::collections::HashSet::with_capacity(bits.len());
    let mut repeats = 0usize;
    for &b in bits {
        if !seen.insert(b) {
            repeats += 1;
            if repeats > allowed_repeats {
                return false;
            }
        }
    }
    true
}

/// Writer properties for one artifact: the codec, an optional row-group cap, and a
/// row-group-sized dictionary page size limit on the f32/f64 leaves.
///
/// parquet-rs enables dictionary encoding for every column and keeps it until the
/// dictionary reaches `dictionary_page_size_limit`, 1 MB by default, i.e. 131,072 distinct
/// f64 or 262,144 distinct f32; past that the column chunk writes the rest of its pages
/// PLAIN. A near-unique float column is the case the dictionary can never pay for -- the
/// dictionary page holds essentially every value and the bit-packed index is pure overhead
/// on top -- and at every row-group cap this engine uses (`FEATURE_ROW_GROUP_ROWS` 65,536,
/// `COMPETED_ROW_GROUP_ROWS`, `HANDOFF_ROW_GROUP_ROWS` and the library/pool `ROW_GROUP_ROWS`
/// 131,072) the chunk ends before the 1 MB limit can be reached, so the fallback never
/// fires and the whole chunk is written RLE_DICTIONARY. The chromatogram list leaves are
/// NOT the exception one might expect: `rt.list.item` and `intensity.list.item` on a
/// shipped chromatograms.parquet both report RLE_DICTIONARY and a dictionary page offset,
/// and they are the largest column chunks in the pipeline.
///
/// [`float_dictionary_page_size_limit`] lowers that limit, per float leaf, to HALF of
/// what the leaf's own values would weigh in the chunk; it has the whole argument and
/// the numbers. Three things follow that are easy to get wrong:
///
/// * It is a LIMIT, not a disable. Disabling the dictionary on float leaves keys on
///   physical type, but the discriminator is cardinality, and the engine writes both kinds
///   in the same table: CLAUDE.md records 10-11 constant columns among the 387 Extended
///   features, plus indicator and small-count features carried as f64. A constant f64
///   column costs +8,130% without its dictionary, a binary one +3,363%, a 0..20 count
///   +855% and a 1,001-valued quantised f32 +9%
///   (`bench_dictionary_rules_by_column_shape`, 1,000,000 rows, 131,072-row groups). Under
///   this rule all four are byte-for-byte what the dictionary produced.
/// * It applies to float leaves ONLY. The same limit set globally would also recover the
///   id columns (unique i32 6.050 -> 4.054 MB, -33%; run-length i32 2.074 -> 1.236 MB,
///   -40%) and costs nothing on two- and three-valued columns, which makes it tempting. It
///   is refused on a shape those probes do not contain: a 20,000-accession `protein` column
///   over 1,000,000 rows goes 3.337 -> 9.091 MB, +172% capped and +340% uncapped, because
///   20,000 accessions need about 320 KB of dictionary. Over the real smoke artifacts it is
///   the worst of every rule tried, +1.39%. A per-column INTEGER rule could still take the
///   id-column gain; it is unmeasured on the real library tables and is not in this change.
/// * It applies to CAPPED writers only, and three writers here are not capped.
///   `write_table`, `write_batches` and `BatchWriter::new` pass no cap and inherit
///   parquet-rs's `DEFAULT_MAX_ROW_GROUP_ROW_COUNT` of 1,048,576; `stages/rescore.rs`
///   writes psms_scored.parquet through `BatchWriter::new`, so it is one of them. Those
///   files are byte-identical to what the previous binary wrote.
///
/// WHAT THIS IS WORTH, and it is deliberately less than a disable would be. Over the 92
/// real artifacts of `ci/smoke.sh`, each rewritten at its own row-group size, this rule is
/// -2.24% with no artifact regressed, against -0.93% and a +18.7% worst artifact for
/// disabling the dictionary outright; on a features-shaped table it is -10.1% against the
/// disable's -18.5%. So it takes a bit over half the gain and none of the risk.
///
/// The ceiling is structural, not a tuning failure. parquet's fallback is PREFIX-based:
/// the pages written before it fires keep their dictionary, so a rule that waits to see
/// the data can never recover what a rule that decided in advance would have. The -13.2%
/// a disable measured on a shipped `features.parquet` and -20.9% on a
/// `chromatograms.parquet` are real, and they are the price of being wrong by two orders
/// of magnitude on any float column that turns out to be low-cardinality. Expect roughly
/// half of those numbers here.
///
/// The read side is not claimed in either direction and neither is the write: repeats of
/// the IDENTICAL arm spread 10-20% on this host, which is wider than any difference
/// measured between arms.
///
/// EQUALITY. Every artifact written through a CAPPED writer whose schema has a float leaf
/// gets different bytes and a different blake3 content hash, exactly as the
/// `MUMDIA_PARQUET_COMPRESSION` knob already does. The decoded f32/f64 values are
/// identical, the dictionary rule leaves every non-float column chunk as it would be
/// without it, and files from the uncapped writers do not move at all
/// (`the_float_limit_shrinks_high_cardinality_leaves_and_touches_nothing_else`); the page
/// rule below changes the page boundaries of every column of a capped writer.
/// [`SpliceWriter`] copies column chunks without re-encoding, so a pooled table assembled
/// from a mix of pre- and post-change band artifacts carries both encodings in different
/// row groups. That is legal parquet and reads correctly, but such a file is reproducible
/// from neither binary alone; re-run the bands rather than pooling across the upgrade.
///
/// PAGES. A capped writer also cuts its data pages by size only (a data page row limit at
/// the row-group cap, [`MAX_DATA_PAGE_ROWS`] at most), not every 20,000 rows as parquet-rs
/// does by default. parquet-rs's synchronous reader fetches every page with its own seek, so
/// a scalar column of a 65,536-row group was four page reads and is now one; on a spinning
/// array, where a reader of a 398-column table is seek-bound (docs/03_io_layer.md,
/// "Sequential row-group reads"), the page count is the read cost. Pages still end at
/// parquet's 1 MB data page size, so list leaves, whose pages were already cut by size,
/// barely change. Measured on the AIF artifacts rewritten at their own row-group sizes
/// (`bench_rewrite_a_real_artifact`, against the c = 0.5 rule with 20,000-row pages):
/// features.parquet (65,536-row groups) 2,003 -> 1,039 data pages at +0.5% bytes,
/// psms_competed.parquet (131,072) 1,592 -> 399 at +0.5%, chromatograms.parquet (65,536)
/// 1,141 -> 947 at +0.05%. Read and write times from the page cache did not move beyond the
/// run-to-run noise. The values are unchanged, and the uncapped writers' files do not move.
///
/// PLAN. Every float leaf that `plan` covers is decided by the plan instead: written PLAIN
/// when its sample was near-unique, otherwise given the dictionary limit sized from its
/// values per row group rather than its rows ([`EncodingPlan`]). A leaf the plan does not
/// cover keeps the unplanned rule above, and so does every leaf when `plan` is `None`.
/// Every leaf under a root column named in `plain` is written without a dictionary whatever
/// the plan says ([`WriteOptions::plain_column`]).
fn writer_props(
    schema: &Schema,
    row_group_rows: Option<usize>,
    plan: Option<&EncodingPlan>,
    plain: &[String],
) -> WriterProperties {
    let mut b = WriterProperties::builder().set_compression(codec());
    if let Some(n) = row_group_rows {
        b = b
            .set_max_row_group_row_count(Some(n.max(1)))
            .set_data_page_row_count_limit(data_page_rows(n));
    }
    let plain_leaves = plain_leaf_paths(schema, plain);
    for leaf in &plain_leaves {
        b = b.set_column_dictionary_enabled(leaf.clone(), false);
    }
    for leaf in float_leaves(schema) {
        if plain_leaves.contains(&leaf.path) {
            continue;
        }
        let planned = plan.and_then(|p| p.leaf(&leaf.path));
        if planned.is_some_and(|l| l.near_unique) {
            b = b.set_column_dictionary_enabled(leaf.path, false);
            continue;
        }
        let chunk_values = match planned {
            Some(l) => row_group_rows.map(|r| r.saturating_mul(l.values_per_row())),
            None => row_group_rows,
        };
        if let Some(limit) = float_dictionary_page_size_limit(chunk_values, leaf.width) {
            b = b.set_column_dictionary_page_size_limit(leaf.path, limit);
        }
    }
    b.build()
}

/// The parquet leaves under the root columns of `schema` named in `plain`.
fn plain_leaf_paths(schema: &Schema, plain: &[String]) -> Vec<ColumnPath> {
    if plain.is_empty() {
        return Vec::new();
    }
    let Ok(desc) = ArrowSchemaConverter::new()
        .with_coerce_types(false)
        .convert(schema)
    else {
        return Vec::new();
    };
    (0..desc.num_columns())
        .filter(|&i| {
            let root = desc.get_column_root_idx(i);
            plain.iter().any(|p| p == schema.field(root).name())
        })
        .map(|i| desc.column(i).path().clone())
        .collect()
}

/// The data page row limit of a writer capped at `row_group_rows` rows per row group.
fn data_page_rows(row_group_rows: usize) -> usize {
    row_group_rows.clamp(1, MAX_DATA_PAGE_ROWS)
}

/// The buffer in front of the hasher of a hashed artifact. parquet hands its sink page
/// headers of a few dozen bytes between page bodies, and blake3 is several times faster on
/// large contiguous inputs than on many small ones, so the hashed sink collects 1 MB before
/// it hashes and writes. An unhashed sink has no buffer of its own (capacity 0 passes every
/// write straight through), so it issues exactly the writes it always did.
const HASH_BUFFER_BYTES: usize = 1 << 20;

/// The file an artifact is written to, optionally hashed on the way (docs/03_io_layer.md,
/// "Hash on write").
///
/// Every stage used to publish its output and then read the whole file back through
/// [`crate::hash::blake3_file`] for the content hash in its report, so each artifact byte
/// crossed the disk or the page cache twice. parquet's writers only ever append to their
/// sink, so the digest of the stream is the digest of the file, and a writer opened with
/// [`WriteOptions::content_hash`] returns it from its `close_hashed` without a read-back.
struct Sink(std::io::BufWriter<crate::hash::HashingWrite<std::fs::File>>);

impl Sink {
    fn create(path: &std::path::Path, hash: bool) -> Result<Sink> {
        let file =
            std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let cap = if hash { HASH_BUFFER_BYTES } else { 0 };
        Ok(Sink(std::io::BufWriter::with_capacity(
            cap,
            crate::hash::HashingWrite::new(file, hash),
        )))
    }

    /// Flush every byte to the file, close it, and return the digest of what was written
    /// when one was asked for. Called after parquet has written the footer.
    fn finish(self) -> Result<Option<String>> {
        let inner = self
            .0
            .into_inner()
            .map_err(|e| anyhow!("flushing an artifact to disk: {}", e.error()))?;
        let (file, digest) = inner.finish();
        drop(file);
        Ok(digest)
    }
}

impl std::io::Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.0.write_all(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

/// The parquet encoder behind [`TableWriter`], [`BatchWriter`] and [`write_batches`].
///
/// An uncapped writer, or one whose schema has no float leaf, encodes from the first batch
/// with [`writer_props`]. A capped writer first holds its first [`plan_sample_rows`] rows,
/// plans its float encodings from them ([`EncodingPlan`]), and then encodes the held batches
/// in the order and the chunks they arrived in, so every column writer sees the sequence of
/// writes it would have seen without the plan. The columns of a row group are encoded on the
/// codec pool ([`crate::codec`]), which writes the serial arrow writer's bytes.
struct Encoder {
    state: EncoderState,
}

enum EncoderState {
    /// Holding the first rows of a capped writer until there are enough to plan from.
    Sampling {
        sink: Box<Sink>,
        schema: Arc<Schema>,
        row_group_rows: usize,
        pending: Vec<RecordBatch>,
        rows: usize,
        /// [`WriteOptions::plain_column`].
        plain: Vec<String>,
    },
    Writing(Box<ColumnEncoder<Sink>>),
    /// Only while a transition is in flight, or after one failed.
    Poisoned,
}

impl Encoder {
    fn new(sink: Sink, schema: Arc<Schema>, opts: &WriteOptions) -> Result<Encoder> {
        if let Some(missing) = opts.plain.iter().find(|p| schema.index_of(p).is_err()) {
            return Err(anyhow!(
                "WriteOptions::plain_column({missing:?}): no such column in {:?}",
                schema
                    .fields()
                    .iter()
                    .map(|f| f.name().as_str())
                    .collect::<Vec<_>>()
            ));
        }
        let plain = opts.plain.clone();
        let row_group_rows = opts.row_group_rows;
        let state = match row_group_rows {
            Some(cap) if plan_enabled() && !float_leaves(&schema).is_empty() => {
                EncoderState::Sampling {
                    sink: Box::new(sink),
                    schema,
                    row_group_rows: cap.max(1),
                    pending: Vec::new(),
                    rows: 0,
                    plain,
                }
            }
            _ => {
                let props = writer_props(&schema, row_group_rows, None, &plain);
                EncoderState::Writing(Box::new(ColumnEncoder::try_new(
                    sink,
                    schema,
                    props,
                    codec_pool(),
                )?))
            }
        };
        Ok(Encoder { state })
    }

    fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        match &mut self.state {
            EncoderState::Writing(w) => Ok(w.write(batch)?),
            EncoderState::Sampling {
                pending,
                rows,
                row_group_rows,
                ..
            } => {
                // An empty batch writes nothing, as the arrow writer skips it too.
                if batch.num_rows() > 0 {
                    pending.push(batch.clone());
                    *rows += batch.num_rows();
                }
                if *rows >= plan_sample_rows(*row_group_rows) {
                    self.start()?;
                }
                Ok(())
            }
            EncoderState::Poisoned => Err(anyhow!("parquet writer used after a failed write")),
        }
    }

    /// Plan from the held rows, open the arrow writer and encode them. A no-op once writing.
    fn start(&mut self) -> Result<()> {
        if !matches!(self.state, EncoderState::Sampling { .. }) {
            return Ok(());
        }
        let EncoderState::Sampling {
            sink,
            schema,
            row_group_rows,
            pending,
            plain,
            ..
        } = std::mem::replace(&mut self.state, EncoderState::Poisoned)
        else {
            unreachable!("checked above");
        };
        let plan = EncodingPlan::of(&schema, &pending, plan_sample_rows(row_group_rows));
        let props = writer_props(&schema, Some(row_group_rows), Some(&plan), &plain);
        let mut w = ColumnEncoder::try_new(*sink, schema, props, codec_pool())?;
        for b in &pending {
            w.write(b)?;
        }
        self.state = EncoderState::Writing(Box::new(w));
        Ok(())
    }

    /// Encode whatever is still held, write the footer and return the sink.
    fn finish(mut self) -> Result<Sink> {
        self.start()?;
        match self.state {
            EncoderState::Writing(w) => w.into_inner(),
            _ => Err(anyhow!("parquet writer used after a failed write")),
        }
    }
}

/// How a writer lays out and accounts for the file it writes. The default is what every
/// writer here did before the options existed: parquet-rs's row-group maximum, no digest.
#[derive(Clone, Debug, Default)]
pub struct WriteOptions {
    row_group_rows: Option<usize>,
    content_hash: bool,
    plain: Vec<String>,
}

impl WriteOptions {
    pub fn new() -> WriteOptions {
        WriteOptions::default()
    }

    /// Cap the rows per row group (see [`TableWriter::with_row_group_rows`]).
    pub fn row_group_rows(mut self, rows: usize) -> WriteOptions {
        self.row_group_rows = Some(rows.max(1));
        self
    }

    /// Hash the file as it is written, so the writer's `close_hashed` returns the same
    /// blake3 digest [`crate::hash::blake3_file`] would compute from the published file,
    /// without reading it back. Leave it off for files nobody records: a temporary splice
    /// source, a sidecar handoff.
    pub fn content_hash(mut self) -> WriteOptions {
        self.content_hash = true;
        self
    }

    /// Write every leaf of the root column `name` without a dictionary, whatever its type
    /// and whatever the float plan says. For a column whose repeats snappy shortens better
    /// in PLAIN form than a dictionary's bit-packed indices allow: the chromatogram `rt`
    /// axis repeats one list per fragment row of a candidate, and PLAIN made the AIF
    /// chromatograms 12.0% smaller than the planned dictionary did (docs/03_io_layer.md,
    /// "Float encodings planned from the first rows"). A name that is not a column of the
    /// written schema is an error when the writer opens.
    pub fn plain_column(mut self, name: &str) -> WriteOptions {
        if !self.plain.iter().any(|p| p == name) {
            self.plain.push(name.to_string());
        }
        self
    }
}

/// The digest a hashed writer returns, or an error naming the writer that was never asked
/// to compute one.
fn require_digest(rows: u64, digest: Option<String>, what: &str) -> Result<Written> {
    match digest {
        Some(content_hash) => Ok(Written { rows, content_hash }),
        None => Err(anyhow!(
            "{what}: close_hashed on a writer opened without WriteOptions::content_hash"
        )),
    }
}

/// A parquet file assembled from the row groups of other parquet files, copied as bytes.
///
/// Pooling a grouped run's band artifacts is a concatenation: the rows are already in the
/// order the pooled table wants, already encoded and already compressed. Decoding and
/// re-encoding them is what the pool used to do, and on the chromatogram tables it ran at
/// 3.5 MB/s on one core against a disk that does 221 MB/s -- five hours for one run's 68 GB
/// (measured on the production seven-file experiment). Splicing the column chunks byte for
/// byte costs a copy, so the pool becomes disk-bound.
///
/// The output carries the template's parquet schema and its Arrow metadata, so a spliced
/// table reads back as the same types as the tables it came from -- including the
/// `LargeList` columns, whose distinction from `List` lives only in that metadata. Every
/// source must have that same schema; splicing is a byte copy and cannot convert anything.
///
/// The values and the row order are exactly those of the sources. The row-group boundaries
/// are the sources' own, so a spliced file is not byte-identical to a re-encoded one.
pub struct SpliceWriter {
    writer: Option<SerializedFileWriter<Sink>>,
    /// The Arrow schema the sources must share, for the caller's own checks.
    pub schema: Arc<Schema>,
    rows: u64,
    target: Option<AtomicPath>,
}

/// The parsed footer of a file to splice from, with its page index when it has one.
fn splice_meta(path: &str) -> Result<(std::fs::File, ParquetMetaData)> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
    let meta = ParquetMetaDataReader::new()
        .with_page_index_policy(PageIndexPolicy::Optional)
        .parse_and_finish(&file)
        .with_context(|| format!("reading the parquet footer of {path}"))?;
    Ok((file, meta))
}

impl SpliceWriter {
    /// Create `out`, taking the schema and the Arrow metadata from `template`, which is
    /// normally the first file whose row groups will be spliced in.
    pub fn create(out: &str, template: &str) -> Result<SpliceWriter> {
        Self::open(out, template, false)
    }

    /// [`SpliceWriter::create`], hashing the spliced file as it is written so
    /// [`SpliceWriter::close_hashed`] returns its content hash without reading it back.
    pub fn create_hashed(out: &str, template: &str) -> Result<SpliceWriter> {
        Self::open(out, template, true)
    }

    fn open(out: &str, template: &str, hash: bool) -> Result<SpliceWriter> {
        let (_, meta) = splice_meta(template)?;
        let fm = meta.file_metadata();
        let props = WriterProperties::builder()
            .set_key_value_metadata(fm.key_value_metadata().cloned())
            .build();
        let schema =
            parquet::arrow::parquet_to_arrow_schema(fm.schema_descr(), fm.key_value_metadata())
                .with_context(|| format!("reading the arrow schema of {template}"))?;
        let target = AtomicPath::new(out)?;
        let file = Sink::create(target.tmp(), hash)?;
        let writer =
            SerializedFileWriter::new(file, fm.schema_descr().root_schema_ptr(), Arc::new(props))
                .with_context(|| format!("opening {out} for splicing"))?;
        Ok(SpliceWriter {
            writer: Some(writer),
            schema: Arc::new(schema),
            rows: 0,
            target: Some(target),
        })
    }

    /// Splice row groups of `src`: those whose index `keep` accepts, or all of them when
    /// `keep` accepts everything. Returns the rows appended.
    pub fn append_row_groups(&mut self, src: &str, keep: impl Fn(usize) -> bool) -> Result<u64> {
        let (file, meta) = splice_meta(src)?;
        let w = self.writer.as_mut().expect("writer closed");
        if meta.file_metadata().schema_descr() != w.schema_descr() {
            return Err(anyhow!(
                "{src} has a different parquet schema from the table being spliced into;                  the band artifacts must come from the same configuration"
            ));
        }
        let column_indexes = meta.column_index();
        let offset_indexes = meta.offset_index();
        let mut rows = 0u64;
        for (i, rg) in meta.row_groups().iter().enumerate() {
            if !keep(i) {
                continue;
            }
            let rg_column = column_indexes.and_then(|ci| ci.get(i));
            let rg_offset = offset_indexes.and_then(|oi| oi.get(i));
            let mut out = w.next_row_group()?;
            for (j, col) in rg.columns().iter().enumerate() {
                out.append_column(
                    &file,
                    ColumnCloseResult {
                        bytes_written: col.compressed_size() as u64,
                        rows_written: rg.num_rows() as u64,
                        metadata: col.clone(),
                        // The engine writes no bloom filters; the page index is carried
                        // through when the source has one.
                        bloom_filter: None,
                        column_index: rg_column.and_then(|r| r.get(j)).cloned(),
                        offset_index: rg_offset.and_then(|r| r.get(j)).cloned(),
                    },
                )?;
            }
            out.close()?;
            rows += rg.num_rows() as u64;
        }
        self.rows += rows;
        Ok(rows)
    }

    /// The row groups of `src`, as (first row, row count) in file order.
    pub fn row_group_spans(src: &str) -> Result<Vec<(usize, usize)>> {
        let (_, meta) = splice_meta(src)?;
        let mut out = Vec::with_capacity(meta.num_row_groups());
        let mut start = 0usize;
        for rg in meta.row_groups() {
            let n = rg.num_rows() as usize;
            out.push((start, n));
            start += n;
        }
        Ok(out)
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn close(self) -> Result<u64> {
        Ok(self.finish()?.0)
    }

    /// Close, and return the rows and the content hash of the spliced file. The writer
    /// must have been opened with [`SpliceWriter::create_hashed`].
    pub fn close_hashed(self) -> Result<Written> {
        let (rows, digest) = self.finish()?;
        require_digest(rows, digest, "SpliceWriter")
    }

    fn finish(mut self) -> Result<(u64, Option<String>)> {
        let mut digest = None;
        if let Some(w) = self.writer.take() {
            // `into_inner` writes the footer, then hands the sink back so its digest can
            // include the footer's bytes; `close` wrote the same bytes and dropped the sink.
            let sink = w.into_inner().context("closing the spliced parquet file")?;
            digest = sink.finish()?;
        }
        if let Some(t) = self.target.take() {
            t.publish()?;
        }
        Ok((self.rows, digest))
    }
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
/// The columns are validated as a set, then encoded in `WRITE_TABLE_CHUNK_ROWS` chunks
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
    write_table_to(TableWriter::new(path), path, cols)?.close()
}

/// [`write_table`], hashing the file as it is written: returns the rows and the content
/// hash [`crate::hash::blake3_file`] would compute, without reading the file back.
pub fn write_table_hashed(path: &str, cols: Vec<Col>) -> Result<Written> {
    write_table_to(TableWriter::new(path).with_content_hash(), path, cols)?.close_hashed()
}

/// Feed `cols` to `w` in [`WRITE_TABLE_CHUNK_ROWS`] chunks, leaving it open.
fn write_table_to(mut w: TableWriter, path: &str, cols: Vec<Col>) -> Result<TableWriter> {
    let nrows = validate_cols(path, &cols)?;
    let mut chunks: Vec<ColChunks> = cols.into_iter().map(ColChunks::of).collect();
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
    Ok(w)
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
    writer: Option<Encoder>,
    target: Option<AtomicPath>,
    rows: u64,
    opts: WriteOptions,
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
            opts: WriteOptions::default(),
        }
    }

    /// Cap the rows per parquet row group (default: the parquet writer's 1,048,576).
    /// Smaller row groups bound the compressed in-progress buffer for wide list columns
    /// (chromatogram traces, spectra peak lists); keep them at tens of thousands of rows
    /// so the footer stays small and readers still get large batches.
    pub fn with_row_group_rows(mut self, rows: usize) -> TableWriter {
        self.opts = self.opts.row_group_rows(rows);
        self
    }

    /// Hash the file as it is written ([`WriteOptions::content_hash`]); close it with
    /// [`TableWriter::close_hashed`] to get the digest.
    pub fn with_content_hash(mut self) -> TableWriter {
        self.opts = self.opts.content_hash();
        self
    }

    /// Write the column `name` without a dictionary ([`WriteOptions::plain_column`]).
    pub fn with_plain_column(mut self, name: &str) -> TableWriter {
        self.opts = self.opts.plain_column(name);
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
                let file = Sink::create(target.tmp(), self.opts.content_hash)?;
                self.target = Some(target);
                self.writer = Some(Encoder::new(file, schema.clone(), &self.opts)?);
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
    pub fn close(self) -> Result<u64> {
        Ok(self.finish()?.0)
    }

    /// Finish the file and return its rows and content hash, computed while it was written.
    /// The writer must have been built [`TableWriter::with_content_hash`].
    pub fn close_hashed(self) -> Result<Written> {
        let path = self.path.clone();
        let (rows, digest) = self.finish()?;
        require_digest(rows, digest, &format!("TableWriter for {path}"))
    }

    fn finish(mut self) -> Result<(u64, Option<String>)> {
        let w = self.writer.take().ok_or_else(|| {
            anyhow!(
                "TableWriter: no chunk written for {}; write one (possibly empty) chunk to fix the schema",
                self.path
            )
        })?;
        // `finish` writes the footer (the same bytes `close` writes) and returns the sink,
        // whose digest therefore covers the whole file.
        let sink = w
            .finish()
            .with_context(|| format!("closing parquet writer {}", self.path))?;
        let digest = sink.finish()?;
        if let Some(t) = self.target.take() {
            t.publish()?;
        }
        Ok((self.rows, digest))
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

/// How [`publish_copy_of`] put the bytes at the destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileCopy {
    /// A second directory entry for the same file: no byte was read or written.
    HardLink,
    /// The filesystem refused the link, so the bytes were copied.
    ByteCopy,
}

impl FileCopy {
    /// The spelling a report records.
    pub fn as_str(self) -> &'static str {
        match self {
            FileCopy::HardLink => "hard_link",
            FileCopy::ByteCopy => "byte_copy",
        }
    }
}

/// Publish the bytes of `src` at `out`, unchanged and without decoding them.
///
/// A hard link into the [`AtomicPath`] temp name, then the usual rename, so a reader of
/// `out` sees its previous content or the complete new one and never a partial file. A
/// filesystem that cannot link (another volume, FAT, some network and sync folders) falls
/// back to a byte copy into the same temp name. An error here leaves `out` as it was.
///
/// The two names share one file after a link. Every writer in this crate (the parquet
/// writers, [`write_batches`], this function and `json::write_json`) publishes by renaming
/// a new file over its destination, which replaces the directory entry and leaves the other
/// name's file alone, so rewriting either artifact through them never changes the other.
/// That does not hold for a writer that opens its destination with `File::create`, which
/// truncates the shared file and so writes through both names. The engine has two, the
/// `features` PIN and rescore's tab-separated handoff, and neither writes a parquet
/// artifact path. A tool that edits one of the linked files IN PLACE changes both.
pub fn publish_copy_of(src: &str, out: &str) -> Result<FileCopy> {
    publish_copy_of_with(src, out, true)
}

fn publish_copy_of_with(src: &str, out: &str, allow_link: bool) -> Result<FileCopy> {
    let target = AtomicPath::new(out)?;
    let tmp = target.tmp().to_path_buf();
    // A temp name left by a killed process with the same pid and counter would make the
    // link fail, and a byte copy onto it would write THROUGH it if it is itself a link to
    // `src`, truncating the source. Start both from a fresh name.
    let _ = std::fs::remove_file(&tmp);
    let linked = allow_link && std::fs::hard_link(src, &tmp).is_ok();
    let how = if linked {
        FileCopy::HardLink
    } else {
        std::fs::copy(src, &tmp).with_context(|| format!("copying {src} -> {}", tmp.display()))?;
        FileCopy::ByteCopy
    };
    target.publish()?;
    // POSIX `rename` does nothing and succeeds when both names already refer to the same
    // file, which is the case when `out` is a link to `src` from a previous run. The temp
    // name then survives the rename; it is a third name for that same file, so removing it
    // loses nothing.
    if tmp.exists() {
        let _ = std::fs::remove_file(&tmp);
    }
    Ok(how)
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
    writer: Option<Encoder>,
    rows: u64,
    target: Option<AtomicPath>,
}

impl BatchWriter {
    pub fn new(path: &str, schema: Arc<Schema>) -> Result<BatchWriter> {
        Self::with_options(path, schema, WriteOptions::default())
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
        Self::with_options(path, schema, WriteOptions::new().row_group_rows(rows))
    }

    /// A writer for `path` laid out and accounted for as `opts` says.
    pub fn with_options(
        path: &str,
        schema: Arc<Schema>,
        opts: WriteOptions,
    ) -> Result<BatchWriter> {
        let target = AtomicPath::new(path)?;
        let file = Sink::create(target.tmp(), opts.content_hash)?;
        Ok(BatchWriter {
            writer: Some(Encoder::new(file, schema, &opts)?),
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
    pub fn close(self) -> Result<u64> {
        Ok(self.finish()?.0)
    }

    /// Finish the file and return its rows and content hash, computed while it was written.
    /// The writer must have been opened with [`WriteOptions::content_hash`].
    pub fn close_hashed(self) -> Result<Written> {
        let (rows, digest) = self.finish()?;
        require_digest(rows, digest, "BatchWriter")
    }

    /// Finish the file and return its rows, and its content hash when the writer was opened
    /// with [`WriteOptions::content_hash`]: for a caller that hashes only some of the files
    /// one code path writes.
    pub fn close_with_digest(self) -> Result<(u64, Option<String>)> {
        self.finish()
    }

    fn finish(mut self) -> Result<(u64, Option<String>)> {
        let mut digest = None;
        if let Some(w) = self.writer.take() {
            let sink = w.finish().context("closing parquet writer")?;
            digest = sink.finish()?;
        }
        if let Some(t) = self.target.take() {
            t.publish()?;
        }
        Ok((self.rows, digest))
    }
}

pub fn write_batches(path: &str, schema: Arc<Schema>, batches: &[RecordBatch]) -> Result<u64> {
    Ok(write_batches_with(path, schema, batches, false)?.0)
}

/// [`write_batches`], hashing the file as it is written.
pub fn write_batches_hashed(
    path: &str,
    schema: Arc<Schema>,
    batches: &[RecordBatch],
) -> Result<Written> {
    let (rows, digest) = write_batches_with(path, schema, batches, true)?;
    require_digest(rows, digest, "write_batches_hashed")
}

fn write_batches_with(
    path: &str,
    schema: Arc<Schema>,
    batches: &[RecordBatch],
    hash: bool,
) -> Result<(u64, Option<String>)> {
    let target = AtomicPath::new(path)?;
    let file = Sink::create(target.tmp(), hash)?;
    // The same encoder [`TableWriter`] uses, minus the row-group cap: this path and the
    // chunked one must produce the same file, which
    // `write_table_matches_one_batch_byte_for_byte_on_scalars` asserts.
    let mut writer = Encoder::new(file, schema, &WriteOptions::default())?;
    let mut n = 0u64;
    for b in batches {
        writer.write(b)?;
        n += b.num_rows() as u64;
    }
    let digest = writer.finish()?.finish()?;
    target.publish()?;
    Ok((n, digest))
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
    // Same `null_count() == 0` fast path as the numeric decoders: on a required column the
    // per-row `is_null` is a validity-bitmap load and a branch for a bit that is always the
    // same. Values and null policy are unchanged.
    if a.null_count() == 0 {
        out.reserve(a.len());
        out.extend(a.values().iter());
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
    out.reserve(a.len());
    if a.null_count() == 0 {
        for k in 0..a.len() {
            out.push(a.value(k) == value);
        }
    } else {
        for k in 0..a.len() {
            if a.is_null(k) {
                return Err(reject_null(name, out.len()));
            }
            out.push(a.value(k) == value);
        }
    }
    Ok(())
}

fn push_opt_f64(out: &mut Vec<Option<f64>>, col: &ArrayRef, name: &str) -> Result<()> {
    let a: &Float64Array = downcast(col, name, "f64")?;
    if a.null_count() == 0 {
        out.extend(a.values().iter().copied().map(Some));
    } else {
        for k in 0..a.len() {
            out.push(if a.is_null(k) { None } else { Some(a.value(k)) });
        }
    }
    Ok(())
}

/// Visit each row of an f32 list column (`List` or `LargeList`) as its own f32 slice;
/// an empty slice for a null row (the rows this layer writes are never null, and the
/// readers that use this treat a null row as empty).
fn for_each_list_f32(col: &ArrayRef, name: &str, mut f: impl FnMut(&[f32])) -> Result<()> {
    let list = ListF32::of(col, name)?;
    for k in 0..list.len() {
        f(list.row_slice(k, name)?);
    }
    Ok(())
}

fn push_list_f32(out: &mut Vec<Vec<f32>>, col: &ArrayRef, name: &str) -> Result<()> {
    out.reserve(col.len());
    for_each_list_f32(col, name, |row| out.push(row.to_vec()))
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
    offsets.reserve(col.len());
    for_each_list_f32(col, name, |row| {
        values.extend_from_slice(row);
        offsets.push(values.len());
    })
}

/// The 32- or 64-bit offset buffer of a list column, already sliced to the array's own
/// window by Arrow, so `bounds[k]..bounds[k + 1]` indexes the child's logical values.
enum ListOffsets<'a> {
    Small(&'a [i32]),
    Large(&'a [i64]),
}

/// Borrowed view of an f32 list column (`List` or `LargeList`) for per-row access while
/// iterating a batch. [`ListF32::row_slice`] is row `k` as a borrowed `&[f32]`,
/// [`ListF32::append_row`] copies it into a caller-owned flat buffer, and
/// [`ListF32::row`] is the owned `Vec<f32>`. A null row is empty in all three.
///
/// The offsets, the child's values and the validity bitmap are resolved ONCE, in
/// [`ListF32::of`], because they are the same for every row of the batch. Reaching a row
/// through `ListArray::value(k)` instead returns an owned `ArrayRef`, which is an
/// `Arc::new(PrimitiveArray)` heap allocation plus an atomic refcount bump on the shared
/// values buffer, per row, thrown away immediately after the copy.
///
/// Measured (`bench_list_view_against_the_per_row_arrayref`, 200,000 rows of 40 values,
/// both arms copying into the same pre-reserved buffer): 7.1-7.4 ms against 14.3-14.8 ms,
/// so 2.0x and 36-37 ns per row. Two list columns (`rt`, `intensity`) are read twice per
/// run, by features and then by quant, which is 108 M rows on the HYE benchmark's 27 M
/// chromatogram rows (about 4 s) and 2.15 billion on the immuno run's 537 M (about 78 s),
/// plus the same number of contended refcount operations. It is also exactly the
/// one-heap-block-per-item pattern the per-process mapping-limit investigation named.
///
/// TWO THINGS ARE NOT BACKWARD-COMPATIBLE, and neither is a values change.
///
/// 1. This was `pub enum ListF32 { Small(&ListArray), Large(&LargeListArray) }`, and those
///    variants were part of the published surface of `mumdia-io` 0.4.0. It is now a struct
///    with private fields, so an external crate that matched on the variants no longer
///    compiles. Nothing in this repository did: the only uses are `ListF32::of` in
///    `stages/features.rs` and `stages/quant.rs`.
/// 2. `of` downcasts the child array eagerly, so a list column whose inner type is not f32
///    is refused HERE rather than at the first non-null row. For every column this engine
///    writes that is the same error at a different moment, but a foreign list column with
///    zero rows, or with every row null, used to decode as empty rows and is now an error
///    ("list '...' inner is not f32"). Failing on the schema rather than silently returning
///    empty traces is the better behaviour, which is why it is kept, but it IS a change,
///    and `a_non_f32_list_is_refused` pins the new one.
pub struct ListF32<'a> {
    offsets: ListOffsets<'a>,
    values: &'a [f32],
    nulls: Option<&'a arrow::buffer::NullBuffer>,
    len: usize,
}

impl<'a> ListF32<'a> {
    pub fn of(col: &'a ArrayRef, name: &str) -> Result<ListF32<'a>> {
        fn child<'b>(v: &'b ArrayRef, name: &str) -> Result<&'b [f32]> {
            // The physical values buffer, as the per-row path already read it: the inner
            // validity bitmap is ignored here exactly as it was before.
            Ok(v.as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow!("list '{name}' inner is not f32"))?
                .values())
        }
        if let Some(a) = col.as_any().downcast_ref::<LargeListArray>() {
            Ok(ListF32 {
                offsets: ListOffsets::Large(a.value_offsets()),
                values: child(a.values(), name)?,
                nulls: a.nulls(),
                len: a.len(),
            })
        } else if let Some(a) = col.as_any().downcast_ref::<ListArray>() {
            Ok(ListF32 {
                offsets: ListOffsets::Small(a.value_offsets()),
                values: child(a.values(), name)?,
                nulls: a.nulls(),
                len: a.len(),
            })
        } else {
            Err(anyhow!("column '{name}' is not a list"))
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Row `k` as a borrowed slice of the batch's own values buffer; empty for a null row.
    ///
    /// The offsets come from the file, so they are checked rather than trusted: a corrupt
    /// pair names the column and the row instead of panicking on the slice.
    pub fn row_slice(&self, k: usize, name: &str) -> Result<&'a [f32]> {
        if self.nulls.is_some_and(|n| n.is_null(k)) {
            return Ok(&[]);
        }
        let (lo, hi) = match &self.offsets {
            ListOffsets::Small(o) => (o[k] as usize, o[k + 1] as usize),
            ListOffsets::Large(o) => (o[k] as usize, o[k + 1] as usize),
        };
        self.values.get(lo..hi).ok_or_else(|| {
            anyhow!(
                "list '{name}' row {k} spans {lo}..{hi} of a {} value buffer",
                self.values.len()
            )
        })
    }

    /// Append row `k` to `out` and return the number of values appended (0 for a null
    /// row). The caller keeps one flat buffer plus its own offsets, so a column of tens
    /// of millions of short traces costs one allocation instead of one per row, which
    /// [`ListF32::row`] cannot avoid.
    pub fn append_row(&self, k: usize, out: &mut Vec<f32>, name: &str) -> Result<usize> {
        let row = self.row_slice(k, name)?;
        out.extend_from_slice(row);
        Ok(row.len())
    }

    /// Row `k` as an owned `Vec<f32>`; a null row is empty.
    pub fn row(&self, k: usize, name: &str) -> Result<Vec<f32>> {
        Ok(self.row_slice(k, name)?.to_vec())
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

/// How [`TableFile::scan`] gets the bytes to the decoder. The default is the plain reader
/// every getter uses: one `File`, pages fetched one at a time.
#[derive(Clone, Debug, Default)]
pub struct ScanOptions {
    /// Read each selected row group's projected column chunks as one byte span, with one
    /// sequential read, and decode from memory ([`crate::span_cache::SpanCache`]). For a
    /// wide projection on a spinning disk this turns a row group's several hundred strided
    /// page reads into one forward read; on an SSD or from the page cache it changes
    /// nothing measurable. Costs up to the options' resident budget in memory.
    pub coalesce: Option<SpanReadOptions>,
}

impl ScanOptions {
    /// Coalesced reads ([`ScanOptions::coalesce`]) with the default span budget.
    pub fn coalesced() -> ScanOptions {
        ScanOptions {
            coalesce: Some(SpanReadOptions::default()),
        }
    }
}

/// Everything a record-batch reader over one file is built from, owned, so the same reader
/// can be built on another thread.
#[derive(Clone)]
struct ReadSpec {
    path: String,
    meta: ArrowReaderMetadata,
    selection: Option<RowSpan>,
    /// Sorted, unique root columns; `None` reads every column.
    roots: Option<Vec<usize>>,
    batch_size: usize,
    coalesce: Option<SpanReadOptions>,
}

impl ReadSpec {
    /// The row groups the reader decodes, in reading order.
    fn row_groups(&self) -> Vec<usize> {
        match &self.selection {
            Some(span) => span.row_groups.clone(),
            None => (0..self.meta.metadata().num_row_groups()).collect(),
        }
    }

    /// The parquet leaf columns under the projected roots.
    fn leaves(&self) -> Vec<usize> {
        let descr = self.meta.metadata().file_metadata().schema_descr();
        (0..descr.num_columns())
            .filter(|&c| match &self.roots {
                None => true,
                Some(r) => r.binary_search(&descr.get_column_root_idx(c)).is_ok(),
            })
            .collect()
    }

    fn build(&self) -> Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader> {
        match &self.coalesce {
            None => {
                let file = std::fs::File::open(&self.path)
                    .with_context(|| format!("opening {}", self.path))?;
                self.build_with(file)
            }
            Some(o) => {
                let cache = crate::span_cache::SpanCache::plan(
                    &self.path,
                    self.meta.metadata(),
                    &self.row_groups(),
                    &self.leaves(),
                    o.clone(),
                )?;
                self.build_with(cache)
            }
        }
    }

    fn build_with<T: parquet::file::reader::ChunkReader + 'static>(
        &self,
        input: T,
    ) -> Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader> {
        // The footer the handle parsed at `open`, not a fresh parse: a typed getter is one
        // call to this, and a stage reads a dozen columns.
        let mut builder =
            ParquetRecordBatchReaderBuilder::new_with_metadata(input, self.meta.clone())
                .with_batch_size(self.batch_size);
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
        if let Some(roots) = &self.roots {
            let mask =
                parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), roots.clone());
            builder = builder.with_projection(mask);
        }
        Ok(builder.build()?)
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

    /// Rows of each row group of the file, in file order, from the footer this handle
    /// holds. On a span handle these are still the whole file's row groups.
    pub fn row_group_rows(&self) -> Vec<usize> {
        let meta: &ParquetMetaData = self.meta.metadata();
        (0..meta.num_row_groups())
            .map(|i| meta.row_group(i).num_rows().max(0) as usize)
            .collect()
    }

    /// Whether every parquet leaf column is REQUIRED (maximum definition level 0). Then no
    /// value in the file can be null, whatever the Arrow schema in its metadata claims.
    pub fn all_leaves_required(&self) -> bool {
        self.meta
            .metadata()
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .all(|c| c.max_def_level() == 0)
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
        self.scan(columns, batch_size, &ScanOptions::default())
    }

    /// [`TableFile::batches`] under explicit read options. With [`ScanOptions::default`]
    /// this is `batches` exactly. The batches are identical whatever the options: the same
    /// rows, the same row-group boundaries, the same values, because the options change
    /// only how the bytes reach the decoder.
    pub fn scan(
        &self,
        columns: Option<&[&str]>,
        batch_size: usize,
        opts: &ScanOptions,
    ) -> Result<BatchReader> {
        let spec = self.read_spec(columns, batch_size, opts.coalesce.clone())?;
        let reader = spec.build()?;
        // Schema from the READER: under a projection it carries only the selected columns.
        let schema = arrow::array::RecordBatchReader::schema(&reader);
        Ok(BatchReader {
            inner: reader,
            schema,
        })
    }

    /// The root columns `columns` names, sorted and unique, or `None` for every column.
    fn projection_roots(&self, columns: Option<&[&str]>) -> Result<Option<Vec<usize>>> {
        let Some(want) = columns else {
            return Ok(None);
        };
        let parquet_schema = self.meta.metadata().file_metadata().schema_descr();
        let fields = parquet_schema.root_schema().get_fields();
        let mut roots: Vec<usize> = Vec::with_capacity(want.len());
        for w in want {
            let i = fields
                .iter()
                .position(|f| f.name() == *w)
                .ok_or_else(|| anyhow!("column '{w}' not found in {:?}", self.column_names()))?;
            roots.push(i);
        }
        roots.sort_unstable();
        roots.dedup();
        Ok(Some(roots))
    }

    fn read_spec(
        &self,
        columns: Option<&[&str]>,
        batch_size: usize,
        coalesce: Option<SpanReadOptions>,
    ) -> Result<ReadSpec> {
        Ok(ReadSpec {
            path: self.path.clone(),
            meta: self.meta.clone(),
            selection: self.selection.clone(),
            roots: self.projection_roots(columns)?,
            batch_size: batch_size.max(1),
            coalesce,
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
    fn row_group_rows_and_required_leaves_describe_the_footer() {
        // compete reuses a features file's bytes only when both hold (docs/11), so each is
        // pinned here on its own: the row-group sizes in file order, also through a span
        // handle, and REQUIRED leaves for plain columns but not for an optional one.
        let dir = std::env::temp_dir().join(format!("mumdia_table_footer_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let req = dir.join("required.parquet").to_str().unwrap().to_string();
        let mut w = TableWriter::new(&req).with_row_group_rows(4);
        w.write_cols(vec![
            Col::U32("id".into(), (0..10).collect()),
            Col::F64("v".into(), (0..10).map(|i| i as f64).collect()),
            Col::Str("s".into(), (0..10).map(|i| format!("s{i}")).collect()),
        ])
        .unwrap();
        w.close().unwrap();
        let t = TableFile::open(&req).unwrap();
        assert_eq!(t.row_group_rows(), vec![4, 4, 2]);
        assert_eq!(
            TableFile::open_rows(&req, 5, 2).unwrap().row_group_rows(),
            vec![4, 4, 2]
        );
        assert!(t.all_leaves_required());

        let opt = dir.join("optional.parquet").to_str().unwrap().to_string();
        write_table(
            &opt,
            vec![
                Col::U32("id".into(), vec![0, 1]),
                Col::OptF64("v".into(), vec![Some(1.0), Some(2.0)]),
            ],
        )
        .unwrap();
        let t = TableFile::open(&opt).unwrap();
        assert_eq!(t.row_group_rows(), vec![2]);
        assert!(!t.all_leaves_required());
        let _ = std::fs::remove_dir_all(&dir);
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

    /// An entirely NULL optional column, which is what every real run writes: `apex_im` on
    /// `psms_extracted` and the three ion-mobility columns on `run_windows` have exactly one
    /// push site each and it pushes `None`.
    ///
    /// Here the chunked write is NOT byte-identical to the single-batch write, and this test
    /// pins that rather than hiding it. An all-null column's definition levels are RLE-run
    /// encoded, which changes the size the writer takes its internal mini-batches in, so a
    /// row-chunk boundary no longer coincides with a mini-batch boundary and one page header
    /// lands elsewhere: measured on parquet 59.3.0, 2,287,903 bytes against 2,287,919 at
    /// 196,615 rows, a 16-byte difference in page framing. The ROWS are identical, which is
    /// the contract this layer makes. The consequence to know about is that
    /// `psms_extracted.parquet` and `run_windows.parquet` have different content hashes from
    /// the ones a pre-chunking build wrote, exactly as the row-group caps elsewhere do.
    #[test]
    fn an_entirely_null_column_keeps_its_rows_but_not_its_page_framing() {
        let n = 3 * WRITE_TABLE_CHUNK_ROWS + 7;
        let cols = |n: usize| -> Vec<Col> {
            vec![
                Col::U32("id".into(), (0..n as u32).collect()),
                Col::F64("mz".into(), (0..n).map(|i| i as f64 * 1.5 - 3.0).collect()),
                Col::OptF64("apex_im".into(), vec![None; n]),
                Col::OptF32("im_lower".into(), vec![None; n]),
                Col::OptI32("im_bin".into(), vec![None; n]),
                Col::OptStr("note".into(), vec![None; n]),
            ]
        };
        // Its own directory: another test in this module wipes the shared one with
        // `remove_dir_all` while these run in parallel threads.
        let dir = std::env::temp_dir().join(format!("mumdia_allnull_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = |n: &str| dir.join(n).to_str().unwrap().to_string();
        let chunked = p("allnull_chunked.parquet");
        let once = p("allnull_once.parquet");
        assert_eq!(write_table(&chunked, cols(n)).unwrap(), n as u64);
        write_one_batch(&once, cols(n));
        let (a, b) = (Table::read(&chunked).unwrap(), Table::read(&once).unwrap());
        assert_eq!(a.schema, b.schema);
        assert_eq!(a.nrows, n);
        assert_eq!(a.u32("id").unwrap(), b.u32("id").unwrap());
        assert_eq!(a.f64("mz").unwrap(), b.f64("mz").unwrap());
        assert_eq!(a.opt_f64("apex_im").unwrap(), b.opt_f64("apex_im").unwrap());
        assert!(
            a.opt_f64("apex_im").unwrap().iter().all(|v| v.is_none()),
            "the fixture's point is that the column is entirely null"
        );
        let (ta, tb) = (
            TableFile::open(&chunked).unwrap(),
            TableFile::open(&once).unwrap(),
        );
        assert_eq!(
            ta.row_group_stats("id").unwrap().len(),
            tb.row_group_stats("id").unwrap().len(),
            "the row groups fall in the same places whichever way the table was written"
        );
        std::fs::remove_file(&chunked).ok();
        std::fs::remove_file(&once).ok();
    }

    /// Past the writer's 1,048,576-row row-group maximum the chunked write must still put
    /// the group boundaries where the single-batch write puts them: the 65,536-row chunk
    /// divides that maximum, so a row group closes on the same row either way. Nothing
    /// tested that before -- the other chunking tests are all under one row group -- and it
    /// is the property the whole chunk-size choice rests on.
    #[test]
    fn the_row_groups_fall_in_the_same_places_past_the_row_group_maximum() {
        let n = 1_048_576 + 7;
        let dir = std::env::temp_dir().join(format!("mumdia_bigrg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = |x: &str| dir.join(x).to_str().unwrap().to_string();
        let cols = |n: usize| -> Vec<Col> {
            vec![
                Col::U32("id".into(), (0..n as u32).collect()),
                Col::F64("mz".into(), (0..n).map(|i| i as f64 * 0.5).collect()),
            ]
        };
        let (chunked, once) = (p("big_chunked.parquet"), p("big_once.parquet"));
        write_table(&chunked, cols(n)).unwrap();
        write_one_batch(&once, cols(n));
        let a = TableFile::open(&chunked)
            .unwrap()
            .row_group_stats("id")
            .unwrap();
        let b = TableFile::open(&once)
            .unwrap()
            .row_group_stats("id")
            .unwrap();
        let rows_a: Vec<usize> = a.iter().map(|g| g.rows).collect();
        let rows_b: Vec<usize> = b.iter().map(|g| g.rows).collect();
        assert_eq!(
            rows_a, rows_b,
            "row groups moved: {rows_a:?} against {rows_b:?}"
        );
        assert_eq!(rows_a, vec![1_048_576, 7]);
        assert_eq!(
            std::fs::read(&chunked).unwrap(),
            std::fs::read(&once).unwrap(),
            "and the file is still byte-identical across a row-group boundary"
        );
        let _ = std::fs::remove_dir_all(&dir);
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

/// The digest a hashed writer returns is the blake3 of the published file, for every writer
/// type and shape, and hashing does not change a byte of the file.
#[cfg(test)]
mod hash_on_write_tests {
    use super::*;
    use crate::hash::blake3_file;

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "mumdia_hash_on_write_{}_{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn cols(n: usize) -> Vec<Col> {
        vec![
            Col::U32("id".into(), (0..n as u32).collect()),
            Col::F64("mz".into(), (0..n).map(|i| i as f64 * 1.000_7).collect()),
            Col::Str(
                "label".into(),
                (0..n)
                    .map(|i| if i % 3 == 0 { "decoy" } else { "target" }.to_string())
                    .collect(),
            ),
            Col::OptF32(
                "im".into(),
                (0..n).map(|i| (i % 4 != 1).then_some(i as f32)).collect(),
            ),
            Col::ListF32(
                "trace".into(),
                (0..n)
                    .map(|i| (0..(i % 9)).map(|k| (i * k) as f32).collect())
                    .collect(),
            ),
            Col::LargeListF32(
                "rt".into(),
                (0..n)
                    .map(|i| (0..(i % 5)).map(|k| k as f32 * 0.25).collect())
                    .collect(),
            ),
        ]
    }

    fn check(path: &str, w: &Written, rows: usize) {
        assert_eq!(w.rows, rows as u64, "{path}");
        assert_eq!(w.content_hash, blake3_file(path).unwrap(), "{path}");
    }

    #[test]
    fn write_table_and_table_writer_digests_are_the_file_digests() {
        let d = dir("table");
        // Empty (schema only), one row, a sub-chunk table, and several 65,536-row chunks.
        for n in [0usize, 1, 1_000, 2 * WRITE_TABLE_CHUNK_ROWS + 17] {
            let hashed = d
                .join(format!("hashed_{n}.parquet"))
                .to_string_lossy()
                .to_string();
            let plain = d
                .join(format!("plain_{n}.parquet"))
                .to_string_lossy()
                .to_string();
            let w = write_table_hashed(&hashed, cols(n)).unwrap();
            check(&hashed, &w, n);
            write_table(&plain, cols(n)).unwrap();
            assert_eq!(
                std::fs::read(&hashed).unwrap(),
                std::fs::read(&plain).unwrap(),
                "hashing must not change the file ({n} rows)"
            );
        }
        // Many row groups through a capped TableWriter, fed in uneven chunks.
        let p = d.join("capped.parquet").to_string_lossy().to_string();
        let mut w = TableWriter::new(&p)
            .with_row_group_rows(1_000)
            .with_content_hash();
        let n = 7_777;
        let mut rows = 0;
        for (a, b) in [
            (0usize, 10usize),
            (10, 3_000),
            (3_000, 3_000),
            (3_000, 7_777),
        ] {
            let all = cols(n);
            w.write_cols(all.into_iter().map(|c| slice_col(c, a, b)).collect())
                .unwrap();
            rows += b - a;
        }
        let written = w.close_hashed().unwrap();
        check(&p, &written, rows);
        assert_eq!(TableFile::open(&p).unwrap().row_group_rows().len(), 8);
        let _ = std::fs::remove_dir_all(&d);
    }

    fn slice_col(c: Col, a: usize, b: usize) -> Col {
        match c {
            Col::U32(n, v) => Col::U32(n, v[a..b].to_vec()),
            Col::F64(n, v) => Col::F64(n, v[a..b].to_vec()),
            Col::Str(n, v) => Col::Str(n, v[a..b].to_vec()),
            Col::OptF32(n, v) => Col::OptF32(n, v[a..b].to_vec()),
            Col::ListF32(n, v) => Col::ListF32(n, v[a..b].to_vec()),
            Col::LargeListF32(n, v) => Col::LargeListF32(n, v[a..b].to_vec()),
            _ => unreachable!("not in the fixture"),
        }
    }

    #[test]
    fn batch_writer_write_batches_and_splice_digests_are_the_file_digests() {
        let d = dir("batch");
        let (schema, batch) = cols_to_batch("fixture", cols(5_000)).unwrap();
        let p = |x: &str| d.join(x).to_string_lossy().to_string();

        // BatchWriter: no batch at all, one batch, and one batch split into many row groups.
        for (name, batches, cap) in [
            ("bw_empty", vec![], Some(100)),
            ("bw_one", vec![batch.clone()], None),
            (
                "bw_many",
                vec![batch.clone(), batch.slice(0, 1_234)],
                Some(700),
            ),
        ] {
            let path = p(&format!("{name}.parquet"));
            let mut opts = WriteOptions::new().content_hash();
            if let Some(c) = cap {
                opts = opts.row_group_rows(c);
            }
            let mut w = BatchWriter::with_options(&path, schema.clone(), opts).unwrap();
            let mut rows = 0;
            for b in &batches {
                w.write(b).unwrap();
                rows += b.num_rows();
            }
            check(&path, &w.close_hashed().unwrap(), rows);
        }

        // write_batches: empty, one, two batches; and the same bytes as the unhashed path.
        for (name, batches) in [
            ("wb_empty", vec![]),
            ("wb_one", vec![batch.clone()]),
            ("wb_two", vec![batch.clone(), batch.slice(100, 900)]),
        ] {
            let path = p(&format!("{name}.parquet"));
            let plain = p(&format!("{name}_plain.parquet"));
            let rows: usize = batches.iter().map(|b: &RecordBatch| b.num_rows()).sum();
            check(
                &path,
                &write_batches_hashed(&path, schema.clone(), &batches).unwrap(),
                rows,
            );
            write_batches(&plain, schema.clone(), &batches).unwrap();
            assert_eq!(
                std::fs::read(&path).unwrap(),
                std::fs::read(&plain).unwrap()
            );
        }

        // SpliceWriter over two capped sources, hashed and not.
        let a = p("src_a.parquet");
        let b = p("src_b.parquet");
        let mut w = TableWriter::new(&a).with_row_group_rows(1_500);
        w.write_cols(cols(4_000)).unwrap();
        w.close().unwrap();
        let mut w = TableWriter::new(&b).with_row_group_rows(1_500);
        w.write_cols(cols(2_000)).unwrap();
        w.close().unwrap();
        let spliced = p("spliced.parquet");
        let mut s = SpliceWriter::create_hashed(&spliced, &a).unwrap();
        s.append_row_groups(&a, |_| true).unwrap();
        s.append_row_groups(&b, |k| k == 1).unwrap();
        let written = s.close_hashed().unwrap();
        check(&spliced, &written, 4_000 + 500);
        let unhashed = p("spliced_plain.parquet");
        let mut s = SpliceWriter::create(&unhashed, &a).unwrap();
        s.append_row_groups(&a, |_| true).unwrap();
        s.append_row_groups(&b, |k| k == 1).unwrap();
        s.close().unwrap();
        assert_eq!(
            std::fs::read(&spliced).unwrap(),
            std::fs::read(&unhashed).unwrap()
        );
        // An empty splice (footer only) is hashed too.
        let empty = p("spliced_empty.parquet");
        let s = SpliceWriter::create_hashed(&empty, &a).unwrap();
        check(&empty, &s.close_hashed().unwrap(), 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn close_hashed_refuses_a_writer_that_was_not_asked_to_hash() {
        let d = dir("refuse");
        let p = d.join("x.parquet").to_string_lossy().to_string();
        let mut w = TableWriter::new(&p);
        w.write_cols(cols(3)).unwrap();
        let e = w.close_hashed().unwrap_err().to_string();
        assert!(e.contains("content_hash"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
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
    fn a_published_copy_is_the_source_bytes_by_link_and_by_copy() {
        let d = dir("copy_of");
        let src = d.join("features.parquet");
        std::fs::write(&src, b"PAR1 the source bytes PAR1").unwrap();
        let src_s = src.to_str().unwrap();
        for (allow_link, want) in [(true, FileCopy::HardLink), (false, FileCopy::ByteCopy)] {
            let out = d.join(format!("competed_{allow_link}.parquet"));
            let out_s = out.to_str().unwrap();
            // Twice: the second publication lands on a destination that already holds the
            // same bytes, and for the link case already IS the same file.
            for _ in 0..2 {
                let how = publish_copy_of_with(src_s, out_s, allow_link).unwrap();
                // A filesystem without hard links degrades to the copy; both are correct.
                assert!(how == want || how == FileCopy::ByteCopy, "{how:?}");
                assert_eq!(std::fs::read(&out).unwrap(), std::fs::read(&src).unwrap());
            }
        }
        // The source is intact, and no temporary name survived either publication: the
        // second link publication renamed a link onto a link to the same file, which POSIX
        // turns into a no-op that leaves the temporary name behind.
        assert_eq!(std::fs::read(&src).unwrap(), b"PAR1 the source bytes PAR1");
        let names: Vec<String> = std::fs::read_dir(&d)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.iter().all(|n| !n.contains(".tmp-")),
            "a temporary file survived: {names:?}"
        );
        assert_eq!(names.len(), 3, "{names:?}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn republishing_either_name_leaves_the_other_file_alone() {
        // After a link the two names share one file. The writers publish by renaming a new
        // file over the destination, so rewriting the source must not change the copy.
        let d = dir("link_independence");
        let src = d.join("features.parquet");
        let out = d.join("competed.parquet");
        std::fs::write(&src, b"v1").unwrap();
        publish_copy_of(src.to_str().unwrap(), out.to_str().unwrap()).unwrap();
        let rewrite = AtomicPath::new(src.to_str().unwrap()).unwrap();
        std::fs::write(rewrite.tmp(), b"v2").unwrap();
        rewrite.publish().unwrap();
        assert_eq!(std::fs::read(&src).unwrap(), b"v2");
        assert_eq!(std::fs::read(&out).unwrap(), b"v1");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_failed_copy_leaves_the_destination_as_it_was() {
        let d = dir("copy_fails");
        let out = d.join("competed.parquet");
        std::fs::write(&out, b"previous").unwrap();
        let missing = d.join("no_such_features.parquet");
        assert!(publish_copy_of(missing.to_str().unwrap(), out.to_str().unwrap()).is_err());
        assert_eq!(std::fs::read(&out).unwrap(), b"previous");
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 1);
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

/// The float columns are written without a dictionary and everything else keeps one, the
/// per-row decoders agree on both sides of their null fast path, and [`ListF32`] returns
/// exactly what the per-row `ArrayRef` path returned.
#[cfg(test)]
mod encoding_tests {
    use super::*;
    use parquet::basic::Encoding;

    fn tmp(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("mumdia_table_enc_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_str().unwrap().to_string()
    }

    /// The encodings parquet recorded for one leaf of row group 0, by its dotted path.
    fn encodings(path: &str, leaf: &str) -> Vec<Encoding> {
        with_leaf(path, leaf, |c| c.encodings().collect())
    }

    /// One leaf's compressed chunk size in row group 0. This, not the encoding name, is
    /// what the rule is chosen on: a column that falls back part-way still reports
    /// RLE_DICTIONARY for the pages written before the fallback.
    fn leaf_bytes(path: &str, leaf: &str) -> i64 {
        with_leaf(path, leaf, |c| c.compressed_size())
    }

    fn with_leaf<T>(
        path: &str,
        leaf: &str,
        f: impl Fn(&parquet::file::metadata::ColumnChunkMetaData) -> T,
    ) -> T {
        let file = std::fs::File::open(path).unwrap();
        let b = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let rg = b.metadata().row_group(0);
        let c = (0..rg.num_columns())
            .map(|i| rg.column(i))
            .find(|c| c.column_descr().path().string() == leaf)
            .unwrap_or_else(|| panic!("no leaf '{leaf}' in {path}"));
        f(c)
    }

    /// The row-group cap the assertions below run at. The float rule only applies to a
    /// CAPPED writer, so a test that wrote through the uncapped `write_table` would
    /// compare the shipped properties against themselves and pass vacuously.
    const TEST_ROW_GROUP: usize = 4_096;

    /// Write `cols` with parquet-rs's own dictionary defaults: a dictionary on every column,
    /// 1 MB limit. The pre-change baseline every assertion below is against. Its data pages
    /// are cut the way a capped writer here cuts them ([`writer_props`], "PAGES"), so a
    /// comparison against it isolates the dictionary rule.
    fn write_with_parquet_defaults(path: &str, cols: Vec<Col>) {
        let (schema, batch) = cols_to_batch(path, cols).unwrap();
        let props = WriterProperties::builder()
            .set_compression(codec())
            .set_max_row_group_row_count(Some(TEST_ROW_GROUP))
            .set_data_page_row_count_limit(data_page_rows(TEST_ROW_GROUP))
            .build();
        let f = std::fs::File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    /// Write `cols` with the SHIPPED properties at the same cap.
    fn write_with_shipped_props(path: &str, cols: Vec<Col>) {
        let mut w = TableWriter::new(path).with_row_group_rows(TEST_ROW_GROUP);
        w.write_cols(cols).unwrap();
        w.close().unwrap();
    }

    fn mixed(n: usize) -> Vec<Col> {
        vec![
            // Near-unique floats: the case the dictionary can never pay for.
            Col::F64("mz".into(), (0..n).map(|i| i as f64 * 1.000_007).collect()),
            Col::F32("irt".into(), (0..n).map(|i| i as f32 * 0.5).collect()),
            Col::OptF64("cal".into(), (0..n).map(|i| Some(i as f64)).collect()),
            // Low-cardinality floats: the case a physical-type rule gets wrong. CLAUDE.md
            // records 10-11 constant columns among the 387 Extended features, plus
            // indicator and small-count features carried as f64.
            Col::F64("const_feat".into(), vec![0.5; n]),
            Col::F64(
                "indicator_feat".into(),
                (0..n).map(|i| (i % 2) as f64).collect(),
            ),
            // Run-length ints and a two-valued string: the cases it pays for handsomely.
            Col::I32(
                "candidate_id".into(),
                (0..n).map(|i| (i / 6) as i32).collect(),
            ),
            Col::U32(
                "charge".into(),
                (0..n).map(|i| (i % 3) as u32 + 1).collect(),
            ),
            Col::Str(
                "label".into(),
                (0..n)
                    .map(|i| if i % 2 == 0 { "target" } else { "decoy" }.to_string())
                    .collect(),
            ),
            Col::ListF32(
                "trace".into(),
                (0..n).map(|i| vec![i as f32, i as f32 + 0.5]).collect(),
            ),
            Col::LargeListF32("big".into(), (0..n).map(|i| vec![i as f32 * 3.0]).collect()),
        ]
    }

    /// The rule, leaf by leaf, against parquet-rs's defaults on the same values:
    /// high-cardinality float leaves shrink, low-cardinality float leaves are untouched
    /// (this is the regression a `dictionary_enabled(false)` rule would cause), and no
    /// non-float leaf moves by a single byte.
    #[test]
    fn the_float_limit_shrinks_high_cardinality_leaves_and_touches_nothing_else() {
        let n = 65_536;
        let p = tmp("limited.parquet");
        let q = tmp("defaults.parquet");
        write_with_shipped_props(&p, mixed(n));
        write_with_parquet_defaults(&q, mixed(n));

        // f32/f64, scalar, nullable and inside a List or a LargeList. Every one of these
        // has far more than the 2,048 distinct f64 the limit allows, so it falls back to
        // PLAIN after a short dictionary prefix and comes out materially smaller.
        for leaf in ["mz", "irt", "cal", "trace.list.item", "big.list.item"] {
            let (a, b) = (leaf_bytes(&p, leaf), leaf_bytes(&q, leaf));
            assert!(a < b, "{leaf}: {a} should be below the dictionary's {b}");
        }
        // The low-cardinality float leaves keep the dictionary they had. A rule keyed on
        // physical type rather than cardinality inflates these by 800-12,000%; see
        // `the_rejected_disable_rule_inflates_a_low_cardinality_float_column`.
        for leaf in ["const_feat", "indicator_feat"] {
            assert_eq!(
                leaf_bytes(&p, leaf),
                leaf_bytes(&q, leaf),
                "{leaf} should be byte-for-byte what the dictionary produced"
            );
            assert!(
                encodings(&p, leaf).contains(&Encoding::RLE_DICTIONARY),
                "{leaf} should still be dictionary encoded"
            );
        }
        // No non-float leaf is touched at all: the rule names float leaves only.
        // `candidate_id` is the fragment library's run of repeated values and `label` is
        // two-valued over 203M rows; a global limit would cost both (see
        // `bench_dictionary_rules_by_column_shape`).
        for leaf in ["candidate_id", "charge", "label"] {
            assert_eq!(
                leaf_bytes(&p, leaf),
                leaf_bytes(&q, leaf),
                "{leaf} must be identical to what the default properties wrote"
            );
            assert!(
                encodings(&p, leaf).contains(&Encoding::RLE_DICTIONARY),
                "{leaf} should still be dictionary encoded"
            );
        }
        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&q).ok();
    }

    /// An UNCAPPED writer is byte-identical to what parquet-rs's own defaults produce, so
    /// every artifact written through `write_table`, `write_batches` or
    /// `BatchWriter::new` -- `psms_scored.parquet` among them -- is unchanged by this rule.
    /// The limit is a fraction of the chunk, and for an uncapped chunk that fraction is
    /// above parquet's own 1 MB, so [`float_dictionary_page_size_limit`] declines to set
    /// it rather than RAISING it.
    #[test]
    fn an_uncapped_write_is_byte_identical_to_the_parquet_defaults() {
        let n = 65_536;
        let p = tmp("uncapped_shipped.parquet");
        let q = tmp("uncapped_defaults.parquet");
        write_table(&p, mixed(n)).unwrap();
        let (schema, batch) = cols_to_batch(&q, mixed(n)).unwrap();
        let props = WriterProperties::builder().set_compression(codec()).build();
        let f = std::fs::File::create(&q).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        assert_eq!(
            std::fs::read(&p).unwrap(),
            std::fs::read(&q).unwrap(),
            "an uncapped write must not move a byte"
        );
        assert!(float_dictionary_page_size_limit(None, 8).is_none());
        // And a chunk so large that half of it exceeds parquet's own 1 MB keeps the
        // default too, for the same reason.
        assert!(float_dictionary_page_size_limit(Some(1_048_576), 8).is_none());
        // The caps the engine actually uses do get a limit, and it tracks the leaf width.
        assert_eq!(
            float_dictionary_page_size_limit(Some(65_536), 8),
            Some(262_144)
        );
        assert_eq!(
            float_dictionary_page_size_limit(Some(65_536), 4),
            Some(131_072)
        );
        assert_eq!(
            float_dictionary_page_size_limit(Some(131_072), 8),
            Some(524_288)
        );
        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&q).ok();
    }

    /// The page rule of [`writer_props`]: a capped writer cuts a scalar column's pages by
    /// size, so a row group of a narrow column is one page where parquet's default cut it
    /// every 20,000 rows, and the values are the ones the default layout holds.
    #[test]
    fn a_capped_writer_cuts_pages_by_size_not_every_20000_rows() {
        let n: usize = 3 * 65_536;
        let cols = || {
            vec![
                Col::F64("mz".into(), (0..n).map(|i| i as f64 * 1.000_7).collect()),
                Col::I32(
                    "candidate_id".into(),
                    (0..n).map(|i| i as i32 / 3).collect(),
                ),
            ]
        };
        let pages = |path: &str| -> Vec<usize> {
            let (_, meta) = splice_meta(path).unwrap();
            let oi = meta
                .offset_index()
                .expect("the writers write an offset index");
            oi.iter()
                .flat_map(|rg| rg.iter().map(|c| c.page_locations().len()))
                .collect()
        };
        let p = tmp("page_rows.parquet");
        let q = tmp("page_rows_default.parquet");
        let mut w = TableWriter::new(&p).with_row_group_rows(65_536);
        w.write_cols(cols()).unwrap();
        w.close().unwrap();
        let (schema, batch) = cols_to_batch(&q, cols()).unwrap();
        let props = WriterProperties::builder()
            .set_compression(codec())
            .set_max_row_group_row_count(Some(65_536))
            .build();
        let f = std::fs::File::create(&q).unwrap();
        let mut dw = ArrowWriter::try_new(f, schema, Some(props)).unwrap();
        dw.write(&batch).unwrap();
        dw.close().unwrap();

        // Three row groups of two columns. 65,536 f64 are 512 KB and the indices of a
        // run-length id column far less, both under the 1 MB page size, so each chunk is one
        // page. (`mz` is near-unique, so the plan writes it PLAIN; unplanned, the dictionary
        // fallback would close the dictionary-encoded prefix as a second page.)
        assert_eq!(pages(&p), vec![1; 6]);
        assert!(pages(&q).iter().all(|&k| k >= 4), "{:?}", pages(&q));
        let (a, b) = (Table::read(&p).unwrap(), Table::read(&q).unwrap());
        assert_eq!(a.f64("mz").unwrap(), b.f64("mz").unwrap());
        assert_eq!(
            a.i32("candidate_id").unwrap(),
            b.i32("candidate_id").unwrap()
        );
        // An uncapped writer keeps parquet's 20,000-row pages.
        write_table(&p, cols()).unwrap();
        assert!(pages(&p).iter().all(|&k| k > 1), "{:?}", pages(&p));
        assert_eq!(data_page_rows(65_536), 65_536);
        assert_eq!(data_page_rows(1 << 20), MAX_DATA_PAGE_ROWS);
        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&q).ok();
    }

    /// Write `cols` as a capped writer would WITHOUT the plan ([`writer_props`] at `cap`).
    fn write_unplanned(path: &str, cols: Vec<Col>, cap: usize) {
        let (schema, batch) = cols_to_batch(path, cols).unwrap();
        let props = writer_props(&schema, Some(cap), None, &[]);
        let f = std::fs::File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    /// Columns for the plan: near-unique scalars and lists, low-cardinality scalars, and a
    /// list leaf of ~50 values a row that repeats (a chromatogram trace), whose dictionary
    /// the unplanned rule cut at a row-sized limit.
    fn planned_cols(n: usize) -> Vec<Col> {
        vec![
            Col::F64("mz".into(), (0..n).map(|i| i as f64 * 1.000_7).collect()),
            Col::F64("const_feat".into(), vec![0.5; n]),
            Col::F64("count".into(), (0..n).map(|i| (i % 20) as f64).collect()),
            // Scattered draws from 100,000 values: a dictionary's 17-bit indices beat four
            // PLAIN bytes that snappy cannot shorten. (A leaf whose rows repeat whole runs
            // of values, like the chromatogram `rt` axis, is the opposite case: snappy
            // shortens the PLAIN runs, and that leaf is written PLAIN by its writer.)
            Col::ListF32(
                "trace".into(),
                (0..n)
                    .map(|i| {
                        (0..50u64)
                            .map(|j| {
                                ((i as u64 * 50 + j).wrapping_mul(2_654_435_761) % 100_000) as f32
                            })
                            .collect()
                    })
                    .collect(),
            ),
            Col::LargeListF32(
                "big".into(),
                (0..n)
                    .map(|i| vec![i as f32 * 3.0, i as f32 * 3.0 + 1.0])
                    .collect(),
            ),
        ]
    }

    /// The plan of a capped writer (F2 of the 2026-09-25 survey): near-unique float leaves,
    /// scalar or list, are written PLAIN from the first page; low-cardinality leaves keep
    /// exactly the dictionary they had; a repeating list leaf keeps a dictionary sized from
    /// its values, not its rows, and comes out smaller; every value is unchanged.
    #[test]
    fn the_plan_writes_near_unique_floats_plain_and_sizes_list_dictionaries_by_values() {
        let (n, cap) = (65_536usize, 65_536usize);
        let p = tmp("planned.parquet");
        let q = tmp("unplanned.parquet");
        let mut w = TableWriter::new(&p).with_row_group_rows(cap);
        w.write_cols(planned_cols(n)).unwrap();
        w.close().unwrap();
        write_unplanned(&q, planned_cols(n), cap);

        for leaf in ["mz", "big.list.item"] {
            assert!(
                !encodings(&p, leaf).contains(&Encoding::RLE_DICTIONARY),
                "{leaf} is near-unique and should have no dictionary"
            );
            assert!(encodings(&q, leaf).contains(&Encoding::RLE_DICTIONARY));
            assert!(
                leaf_bytes(&p, leaf) < leaf_bytes(&q, leaf),
                "{leaf}: {} against {}",
                leaf_bytes(&p, leaf),
                leaf_bytes(&q, leaf)
            );
        }
        for leaf in ["const_feat", "count"] {
            assert!(encodings(&p, leaf).contains(&Encoding::RLE_DICTIONARY));
            assert_eq!(leaf_bytes(&p, leaf), leaf_bytes(&q, leaf), "{leaf}");
        }
        // 100,000 distinct f32 are 400 KB of dictionary: under the 1 MB the plan leaves a
        // 50-values-a-row leaf, over the 128 KB the row-sized limit gave it.
        assert!(encodings(&p, "trace.list.item").contains(&Encoding::RLE_DICTIONARY));
        assert!(
            leaf_bytes(&p, "trace.list.item") < leaf_bytes(&q, "trace.list.item"),
            "trace: {} against {}",
            leaf_bytes(&p, "trace.list.item"),
            leaf_bytes(&q, "trace.list.item")
        );
        let (a, b) = (Table::read(&p).unwrap(), Table::read(&q).unwrap());
        assert_eq!(a.f64("mz").unwrap(), b.f64("mz").unwrap());
        assert_eq!(a.f64("count").unwrap(), b.f64("count").unwrap());
        assert_eq!(a.list_f32("trace").unwrap(), b.list_f32("trace").unwrap());
        assert_eq!(a.list_f32("big").unwrap(), b.list_f32("big").unwrap());
        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&q).ok();
    }

    /// The plan is a function of the first rows, not of the chunks they arrive in: the same
    /// rows split any way plan the same encodings, and scalar columns written in chunks that
    /// are whole multiples of the encoder's 1,024-value mini-batch make the same file. (A list
    /// leaf's page boundaries depend on the chunking with or without a plan;
    /// `write_table_matches_one_batch_row_for_row_on_lists` has that.)
    #[test]
    fn the_plan_depends_on_the_rows_not_the_chunks() {
        let n = 65_536usize;
        let (schema, whole) = cols_to_batch("plan", planned_cols(n)).unwrap();
        let sample = plan_sample_rows(n);
        let one = EncodingPlan::of(&schema, std::slice::from_ref(&whole), sample);
        let mut pieces = Vec::new();
        let mut at = 0usize;
        for k in [0usize, 1, 999, 0, 4_096, 7, 30_000, 30_433] {
            pieces.push(whole.slice(at, k));
            at += k;
        }
        assert_eq!(at, n);
        assert_eq!(one, EncodingPlan::of(&schema, &pieces, sample));
        assert_eq!(one.leaves.len(), 5);
        assert!(one.leaf(&ColumnPath::from("mz")).unwrap().near_unique);
        let trace = ColumnPath::new(vec!["trace".into(), "list".into(), "item".into()]);
        assert_eq!(one.leaf(&trace).unwrap().values_per_row(), 50);

        let scalars = || -> Vec<Col> {
            planned_cols(n)
                .into_iter()
                .filter(|c| matches!(c, Col::F64(..)))
                .collect()
        };
        let p = tmp("plan_whole.parquet");
        let q = tmp("plan_chunked.parquet");
        let mut w = TableWriter::new(&p).with_row_group_rows(n);
        w.write_cols(scalars()).unwrap();
        w.close().unwrap();
        let mut w = TableWriter::new(&q).with_row_group_rows(n);
        let cols = scalars();
        for c in 0..8 {
            let (a, b) = (c * 8_192, (c + 1) * 8_192);
            w.write_cols(cols.iter().map(|col| slice_col(col, a, b)).collect())
                .unwrap();
        }
        w.close().unwrap();
        let (a, b) = (std::fs::read(&p).unwrap(), std::fs::read(&q).unwrap());
        assert!(a == b, "{} bytes against {}", a.len(), b.len());
        assert!(!encodings(&p, "mz").contains(&Encoding::RLE_DICTIONARY));
        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&q).ok();
    }

    /// Rows `a..b` of one f64 column, for the chunking test.
    fn slice_col(c: &Col, a: usize, b: usize) -> Col {
        match c {
            Col::F64(n, v) => Col::F64(n.clone(), v[a..b].to_vec()),
            _ => unreachable!("the chunking test writes f64 columns only"),
        }
    }

    /// A table too short for the sample to hold [`PLAN_MIN_VALUES`] values of a scalar leaf
    /// plans nothing for it: its file is the unplanned writer's file byte for byte.
    #[test]
    fn a_sample_too_small_to_plan_leaves_the_unplanned_layout() {
        let cols = || {
            vec![
                Col::F64(
                    "mz".into(),
                    (0..3_000).map(|i| i as f64 * 1.000_7).collect(),
                ),
                Col::F32("irt".into(), (0..3_000).map(|i| i as f32 * 0.5).collect()),
            ]
        };
        let p = tmp("small_planned.parquet");
        let q = tmp("small_unplanned.parquet");
        let mut w = TableWriter::new(&p).with_row_group_rows(65_536);
        w.write_cols(cols()).unwrap();
        w.close().unwrap();
        write_unplanned(&q, cols(), 65_536);
        assert_eq!(std::fs::read(&p).unwrap(), std::fs::read(&q).unwrap());
        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&q).ok();
    }

    /// A column named with [`WriteOptions::plain_column`] loses its dictionary on every leaf
    /// whatever the plan says, the other columns keep what the plan gave them, the values
    /// are unchanged, and a name that is not in the schema is refused when the writer opens.
    #[test]
    fn a_plain_column_is_written_without_a_dictionary() {
        let n = 20_000usize;
        let cols = || {
            vec![
                Col::U32(
                    "candidate_id".into(),
                    (0..n).map(|i| (i / 6) as u32).collect(),
                ),
                // One axis per candidate, repeated on each of its six fragment rows: the
                // shape of the chromatogram `rt` list.
                Col::LargeListF32(
                    "rt".into(),
                    (0..n)
                        .map(|i| (0..40).map(|j| ((i / 6) * 3 + j) as f32 * 0.9).collect())
                        .collect(),
                ),
                Col::LargeListF32(
                    "intensity".into(),
                    (0..n)
                        .map(|i| (0..40).map(|j| ((i * 13 + j) % 97) as f32).collect())
                        .collect(),
                ),
            ]
        };
        let p = tmp("plain_rt.parquet");
        let q = tmp("planned_rt.parquet");
        let mut w = TableWriter::new(&p)
            .with_row_group_rows(8_192)
            .with_plain_column("rt");
        w.write_cols(cols()).unwrap();
        w.close().unwrap();
        let mut w = TableWriter::new(&q).with_row_group_rows(8_192);
        w.write_cols(cols()).unwrap();
        w.close().unwrap();

        assert!(!encodings(&p, "rt.list.item").contains(&Encoding::RLE_DICTIONARY));
        assert!(encodings(&q, "rt.list.item").contains(&Encoding::RLE_DICTIONARY));
        for leaf in ["candidate_id", "intensity.list.item"] {
            assert_eq!(leaf_bytes(&p, leaf), leaf_bytes(&q, leaf), "{leaf}");
        }
        let (a, b) = (Table::read(&p).unwrap(), Table::read(&q).unwrap());
        assert_eq!(a.list_f32("rt").unwrap(), b.list_f32("rt").unwrap());
        assert_eq!(
            a.list_f32("intensity").unwrap(),
            b.list_f32("intensity").unwrap()
        );

        // Uncapped writers honour it too.
        let (schema, batch) = cols_to_batch(&p, cols()).unwrap();
        let mut bw =
            BatchWriter::with_options(&p, schema.clone(), WriteOptions::new().plain_column("rt"))
                .unwrap();
        bw.write(&batch).unwrap();
        bw.close().unwrap();
        assert!(!encodings(&p, "rt.list.item").contains(&Encoding::RLE_DICTIONARY));
        assert_eq!(
            Table::read(&p).unwrap().list_f32("rt").unwrap(),
            b.list_f32("rt").unwrap()
        );

        let mut w = TableWriter::new(&q).with_plain_column("retention");
        let err = w.write_cols(cols()).unwrap_err().to_string();
        assert!(err.contains("retention"), "{err}");
        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&q).ok();
    }

    /// Why [`writer_props`] sets a per-column dictionary page size LIMIT on the float
    /// leaves rather than disabling their dictionary. The disable was the first rule
    /// written and it keys on physical type, but the discriminator is cardinality: a
    /// constant f64 column costs two orders of magnitude more without its dictionary, and
    /// the engine has 10-11 of them in every features table.
    #[test]
    fn the_rejected_disable_rule_inflates_a_low_cardinality_float_column() {
        let n = 65_536;
        let cols = || vec![Col::F64("const_feat".into(), vec![0.5; n])];
        let p = tmp("limit_rule.parquet");
        let q = tmp("disable_rule.parquet");
        write_with_shipped_props(&p, cols());

        let (schema, batch) = cols_to_batch(&q, cols()).unwrap();
        let mut b = WriterProperties::builder()
            .set_compression(codec())
            .set_max_row_group_row_count(Some(TEST_ROW_GROUP));
        for (leaf, _) in float_leaf_paths(&schema) {
            b = b.set_column_dictionary_enabled(leaf, false);
        }
        let f = std::fs::File::create(&q).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, Some(b.build())).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let (kept, disabled) = (leaf_bytes(&p, "const_feat"), leaf_bytes(&q, "const_feat"));
        assert!(
            disabled > kept * 10,
            "the disable rule should be the regression this one avoids: {disabled} vs {kept}"
        );
        assert_eq!(
            Table::read(&p).unwrap().f64("const_feat").unwrap(),
            Table::read(&q).unwrap().f64("const_feat").unwrap()
        );
        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&q).ok();
    }

    /// The point of the setting: the same values, fewer bytes, and every reader still
    /// decodes them. This fixture is 10 columns wide at a 4,096-row group, so the margin
    /// is not the one a real artifact shows; the direction is what this pins.
    #[test]
    fn limiting_the_float_dictionary_keeps_the_values_and_shrinks_the_file() {
        let n = 65_536;
        let p = tmp("nodict.parquet");
        let q = tmp("withdict.parquet");
        write_with_shipped_props(&p, mixed(n));
        write_with_parquet_defaults(&q, mixed(n));

        let a = Table::read(&p).unwrap();
        let b = Table::read(&q).unwrap();
        assert_eq!(a.f64("mz").unwrap(), b.f64("mz").unwrap());
        assert_eq!(a.f32("irt").unwrap(), b.f32("irt").unwrap());
        assert_eq!(a.opt_f64("cal").unwrap(), b.opt_f64("cal").unwrap());
        assert_eq!(a.list_f32("trace").unwrap(), b.list_f32("trace").unwrap());
        assert_eq!(
            a.i32("candidate_id").unwrap(),
            b.i32("candidate_id").unwrap()
        );
        assert_eq!(a.str("label").unwrap(), b.str("label").unwrap());

        let (sa, sb) = (
            std::fs::metadata(&p).unwrap().len(),
            std::fs::metadata(&q).unwrap().len(),
        );
        assert!(
            sa < sb,
            "the float dictionary limit should be smaller: {sa} vs {sb}"
        );
        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&q).ok();
    }

    /// Pins the OLD behaviour of the three decoders that gained a `null_count() == 0` fast
    /// path: identical values with no nulls, and the identical error, naming the same row,
    /// with one.
    #[test]
    fn the_decoder_fast_paths_agree_with_the_per_row_loops() {
        let flags: ArrayRef = Arc::new(BooleanArray::from(vec![true, false, true, true]));
        let mut out = Vec::new();
        push_bool(&mut out, &flags, "flag").unwrap();
        assert_eq!(out, vec![true, false, true, true]);
        let with_null: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)]));
        let mut out = Vec::new();
        let e = push_bool(&mut out, &with_null, "flag")
            .unwrap_err()
            .to_string();
        assert!(e.contains("'flag'") && e.contains("row 1"), "{e}");

        let names: ArrayRef = Arc::new(StringArray::from(vec!["target", "decoy", "target"]));
        let mut out = Vec::new();
        push_str_eq(&mut out, &names, "label", "target").unwrap();
        assert_eq!(out, vec![true, false, true]);
        let with_null: ArrayRef =
            Arc::new(StringArray::from(vec![Some("target"), Some("decoy"), None]));
        let mut out = Vec::new();
        let e = push_str_eq(&mut out, &with_null, "label", "target")
            .unwrap_err()
            .to_string();
        assert!(e.contains("'label'") && e.contains("row 2"), "{e}");

        let cal: ArrayRef = Arc::new(Float64Array::from(vec![1.5, -0.0, f64::NAN]));
        let mut out = Vec::new();
        push_opt_f64(&mut out, &cal, "cal").unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], Some(1.5));
        // -0.0 and NaN survive the fast path bit for bit.
        assert_eq!(out[1].unwrap().to_bits(), (-0.0f64).to_bits());
        assert!(out[2].unwrap().is_nan());
        let cal: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
        let mut out = Vec::new();
        push_opt_f64(&mut out, &cal, "cal").unwrap();
        assert_eq!(out, vec![Some(1.0), None, Some(3.0)]);
    }

    /// The same three decoders on SLICED arrays, which is what [`BatchReader`] can hand
    /// them. The fast paths read `a.values()`, so they are only correct if Arrow has
    /// already offset-sliced the value buffer; the slow paths index with `value(k)`, which
    /// applies the offset itself. Every window is checked both ways, including windows
    /// that exclude the null so `null_count()` flips to 0 and the arm changes.
    #[test]
    fn the_decoder_fast_paths_agree_with_the_per_row_loops_on_sliced_arrays() {
        let flags = BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(true),
            Some(false),
        ]);
        let names = StringArray::from(vec![
            Some("target"),
            Some("decoy"),
            None,
            Some("target"),
            Some("target"),
            Some("decoy"),
        ]);
        let cal = Float64Array::from(vec![
            Some(1.5),
            Some(-0.0),
            None,
            Some(f64::NAN),
            Some(3.0),
            Some(-2.5),
        ]);
        for (off, len) in [(0, 6), (0, 2), (1, 2), (2, 3), (3, 3), (4, 2), (5, 1)] {
            let window = format!("[{off}..{}]", off + len);

            let col: ArrayRef = Arc::new(flags.slice(off, len));
            let a = col.as_any().downcast_ref::<BooleanArray>().unwrap();
            let mut fast = Vec::new();
            let got = push_bool(&mut fast, &col, "flag");
            match (0..a.len()).find(|&k| a.is_null(k)) {
                Some(k) => assert!(
                    got.unwrap_err().to_string().contains(&format!("row {k}")),
                    "bool {window}"
                ),
                None => {
                    let slow: Vec<bool> = (0..a.len()).map(|k| a.value(k)).collect();
                    assert_eq!(fast, slow, "bool {window}");
                }
            }

            let col: ArrayRef = Arc::new(names.slice(off, len));
            let a = col.as_any().downcast_ref::<StringArray>().unwrap();
            let mut fast = Vec::new();
            let got = push_str_eq(&mut fast, &col, "label", "target");
            match (0..a.len()).find(|&k| a.is_null(k)) {
                Some(k) => assert!(
                    got.unwrap_err().to_string().contains(&format!("row {k}")),
                    "str {window}"
                ),
                None => {
                    let slow: Vec<bool> = (0..a.len()).map(|k| a.value(k) == "target").collect();
                    assert_eq!(fast, slow, "str {window}");
                }
            }

            let col: ArrayRef = Arc::new(cal.slice(off, len));
            let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
            let mut fast = Vec::new();
            push_opt_f64(&mut fast, &col, "cal").unwrap();
            let slow: Vec<Option<f64>> = (0..a.len())
                .map(|k| (!a.is_null(k)).then(|| a.value(k)))
                .collect();
            let bits = |v: &[Option<f64>]| {
                v.iter()
                    .map(|x| x.map(f64::to_bits))
                    .collect::<Vec<Option<u64>>>()
            };
            assert_eq!(bits(&fast), bits(&slow), "f64 {window}");
        }
    }

    fn list_with_a_null() -> ListArray {
        let mut b = ListBuilder::new(Float32Builder::new());
        b.values().append_slice(&[1.0, 2.0, 3.0]);
        b.append(true);
        b.append(false); // a null row
        b.values().append_slice(&[]);
        b.append(true); // an empty row
        b.values().append_slice(&[9.5, -0.0, f32::NAN]);
        b.append(true);
        b.finish()
    }

    /// Pins the OLD per-row path: whatever `ListArray::value(k)` yielded, including for a
    /// null row, an empty row and a SLICED array whose offsets no longer start at 0, is
    /// what the resolved-once view must yield.
    #[test]
    fn the_list_view_matches_the_per_row_arrayref_path() {
        fn old_row(a: &ListArray, k: usize) -> Vec<f32> {
            if a.is_null(k) {
                return Vec::new();
            }
            a.value(k)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .to_vec()
        }
        let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for (label, arr) in [
            ("whole", list_with_a_null()),
            ("sliced", list_with_a_null().slice(1, 3)),
        ] {
            let col: ArrayRef = Arc::new(arr.clone());
            let v = ListF32::of(&col, "trace").unwrap();
            assert_eq!(v.len(), arr.len(), "{label}");
            let mut flat = Vec::new();
            for k in 0..arr.len() {
                let old = old_row(&arr, k);
                assert_eq!(
                    bits(v.row_slice(k, "trace").unwrap()),
                    bits(&old),
                    "{label} {k}"
                );
                assert_eq!(bits(&v.row(k, "trace").unwrap()), bits(&old), "{label} {k}");
                let n = v.append_row(k, &mut flat, "trace").unwrap();
                assert_eq!(n, old.len(), "{label} {k}");
            }
            let total: usize = (0..arr.len()).map(|k| old_row(&arr, k).len()).sum();
            assert_eq!(flat.len(), total, "{label}");
        }
    }

    /// The same for `LargeList` (64-bit offsets), which is what the chromatogram columns
    /// actually are on a real run.
    #[test]
    fn the_list_view_handles_large_list_offsets() {
        let mut b = LargeListBuilder::new(Float32Builder::new());
        b.values().append_slice(&[4.0, 5.0]);
        b.append(true);
        b.append(false);
        b.values().append_slice(&[6.0]);
        b.append(true);
        let arr = b.finish();
        let col: ArrayRef = Arc::new(arr);
        let v = ListF32::of(&col, "trace").unwrap();
        assert_eq!(v.row(0, "trace").unwrap(), vec![4.0, 5.0]);
        assert_eq!(v.row(1, "trace").unwrap(), Vec::<f32>::new());
        assert_eq!(v.row(2, "trace").unwrap(), vec![6.0]);
    }

    /// A non-f32 list is still refused, only now at [`ListF32::of`] rather than at the
    /// first row. The last case is the one behaviour change in the list rework, pinned
    /// deliberately: an EMPTY list column of the wrong inner type used to decode as no
    /// rows, because the old per-row loop never reached the downcast, and is now the same
    /// error the non-empty one gives. See the note on [`ListF32`].
    #[test]
    fn a_non_f32_list_is_refused() {
        let mut b = ListBuilder::new(arrow::array::Float64Builder::new());
        b.values().append_slice(&[1.0]);
        b.append(true);
        let col: ArrayRef = Arc::new(b.finish());
        let e = ListF32::of(&col, "trace").err().unwrap().to_string();
        assert!(e.contains("'trace'") && e.contains("not f32"), "{e}");

        let col: ArrayRef = Arc::new(Float32Array::from(vec![1.0f32]));
        let e = ListF32::of(&col, "trace").err().unwrap().to_string();
        assert!(e.contains("is not a list"), "{e}");

        let empty: ArrayRef =
            Arc::new(ListBuilder::new(arrow::array::Float64Builder::new()).finish());
        assert_eq!(empty.len(), 0);
        let e = ListF32::of(&empty, "trace").err().unwrap().to_string();
        assert!(e.contains("'trace'") && e.contains("not f32"), "{e}");
        // A zero-row column of the RIGHT type is still fine, which is the case the engine
        // can actually produce.
        let empty: ArrayRef = Arc::new(ListBuilder::new(Float32Builder::new()).finish());
        assert_eq!(ListF32::of(&empty, "trace").unwrap().len(), 0);
    }
}

/// Microbenchmarks for the writer properties and the list view. Ignored by default; run
/// them with
///
/// ```text
/// cargo test -p mumdia-io --release -- --ignored --nocapture bench_
/// ```
///
/// Both arms of every A/B build their inputs the same way OUTSIDE the timer and the timer
/// covers only the operation being compared.
#[cfg(test)]
mod writer_bench {
    use super::*;
    use std::time::Instant;

    fn tmp(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("mumdia_table_bench_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_str().unwrap().to_string()
    }

    /// A cheap LCG, so the floats are near-unique and not compressible by accident.
    fn noise(seed: u64, n: usize) -> Vec<f64> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64) / ((1u64 << 53) as f64) * 1e4
            })
            .collect()
    }

    /// The dictionary rules the choice was made between. `Default` is parquet-rs as
    /// shipped and therefore the pre-change baseline every percentage is against. `Shipped`
    /// is the c = 0.5 dictionary rule alone, with parquet's 20,000-row data pages; `Writer`
    /// is what this crate's capped writers use today ([`writer_props`]), whatever that is.
    #[derive(Clone, Copy)]
    enum DictRule {
        Default,
        FloatOff,
        Float16K,
        Shipped,
        FloatC075,
        FloatC025,
        GlobalLimited,
        /// The unplanned [`writer_props`]: the c = 0.5 rule with pages cut by size.
        WriterUnplanned,
        /// [`writer_props`] with a plan from the first `cap / sample_div` rows at a
        /// distinct-fraction `threshold`; `rt_plain` also drops the chromatogram `rt` leaf's
        /// dictionary whatever the plan says (X6 of the 2026-09-25 survey).
        Planned {
            sample_div: usize,
            threshold: f64,
            rt_plain: bool,
        },
    }

    const WRITER: DictRule = DictRule::Planned {
        sample_div: PLAN_SAMPLE_FRACTION,
        threshold: PLAN_PLAIN_ABOVE_DISTINCT,
        rt_plain: false,
    };

    const DICT_RULES: &[(&str, DictRule)] = &[
        ("default", DictRule::Default),
        ("float-off", DictRule::FloatOff),
        ("float-16K", DictRule::Float16K),
        ("c0.5", DictRule::Shipped),
        ("c0.75", DictRule::FloatC075),
        ("c0.25", DictRule::FloatC025),
        ("global-16K", DictRule::GlobalLimited),
        ("unplanned", DictRule::WriterUnplanned),
        ("writer", WRITER),
        (
            "plan-0.9",
            DictRule::Planned {
                sample_div: PLAN_SAMPLE_FRACTION,
                threshold: 0.9,
                rt_plain: false,
            },
        ),
        (
            "plan-0.95",
            DictRule::Planned {
                sample_div: PLAN_SAMPLE_FRACTION,
                threshold: 0.95,
                rt_plain: false,
            },
        ),
        (
            "plan-group",
            DictRule::Planned {
                sample_div: 1,
                threshold: PLAN_PLAIN_ABOVE_DISTINCT,
                rt_plain: false,
            },
        ),
        (
            "writer+rt-plain",
            DictRule::Planned {
                sample_div: PLAN_SAMPLE_FRACTION,
                threshold: PLAN_PLAIN_ABOVE_DISTINCT,
                rt_plain: true,
            },
        ),
    ];

    /// The properties `rule` writes `batches` with. A planned rule samples `batches` the way
    /// a capped writer samples its first rows.
    fn props_for(
        rule: DictRule,
        schema: &Schema,
        cap: Option<usize>,
        batches: &[RecordBatch],
    ) -> WriterProperties {
        let DictRule::Planned {
            sample_div,
            threshold,
            rt_plain,
        } = rule
        else {
            return props_under(rule, schema, cap);
        };
        let Some(rows) = cap else {
            return writer_props(schema, None, None, &[]);
        };
        let plan = EncodingPlan::with_threshold(
            schema,
            batches,
            rows.div_ceil(sample_div).max(1),
            threshold,
        );
        if !rt_plain {
            return writer_props(schema, cap, Some(&plan), &[]);
        }
        writer_props(schema, cap, Some(&plan), &["rt".to_string()])
    }

    fn props_under(rule: DictRule, schema: &Schema, cap: Option<usize>) -> WriterProperties {
        match rule {
            DictRule::WriterUnplanned => return writer_props(schema, cap, None, &[]),
            DictRule::Planned { .. } => panic!("a planned rule needs the batches: props_for"),
            _ => {}
        }
        let mut b = WriterProperties::builder().set_compression(codec());
        if let Some(n) = cap {
            b = b.set_max_row_group_row_count(Some(n.max(1)));
        }
        match rule {
            DictRule::Default => {}
            DictRule::FloatOff => {
                for (leaf, _) in float_leaf_paths(schema) {
                    b = b.set_column_dictionary_enabled(leaf, false);
                }
            }
            DictRule::Float16K => {
                for (leaf, _) in float_leaf_paths(schema) {
                    b = b.set_column_dictionary_page_size_limit(leaf, 16 * 1024);
                }
            }
            DictRule::Shipped => {
                for (leaf, width) in float_leaf_paths(schema) {
                    if let Some(limit) = float_dictionary_page_size_limit(cap, width) {
                        b = b.set_column_dictionary_page_size_limit(leaf, limit);
                    }
                }
            }
            DictRule::FloatC075 => {
                for (leaf, width) in float_leaf_paths(schema) {
                    if let Some(rows) = cap {
                        b = b.set_column_dictionary_page_size_limit(leaf, rows * width / 4 * 3);
                    }
                }
            }
            DictRule::FloatC025 => {
                for (leaf, width) in float_leaf_paths(schema) {
                    if let Some(rows) = cap {
                        b = b.set_column_dictionary_page_size_limit(leaf, rows * width / 4);
                    }
                }
            }
            DictRule::GlobalLimited => {
                b = b.set_dictionary_page_size_limit(16 * 1024);
            }
            DictRule::WriterUnplanned | DictRule::Planned { .. } => unreachable!("above"),
        }
        b.build()
    }

    /// Write `cols` under one rule and return the file size. The batch is built outside
    /// any timer the caller keeps, so only the encode differs between arms.
    fn write_under(path: &str, cols: Vec<Col>, rule: DictRule, cap: Option<usize>) -> u64 {
        let (schema, batch) = cols_to_batch(path, cols).unwrap();
        let props = props_for(rule, &schema, cap, std::slice::from_ref(&batch));
        let f = std::fs::File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        std::fs::metadata(path).unwrap().len()
    }

    /// The same A/B on a REAL artifact, because a synthetic float column is pure noise and
    /// therefore the most favourable case a dictionary can be given. Point it at one:
    ///
    /// ```text
    /// MUMDIA_BENCH_PARQUET=out_aif02/features.parquet \
    ///   cargo test -p mumdia-io --release -- --ignored --nocapture bench_rewrite
    /// ```
    ///
    /// Both arms rewrite the SAME decoded batches, so the read and the decode are outside
    /// the timer and only the encode is compared.
    ///
    /// Both arms also use the SOURCE FILE'S OWN row-group size, not parquet-rs's default,
    /// because the dictionary fallback threshold is per column chunk and the two are not
    /// the same measurement. An earlier version of this bench passed no cap, which put
    /// 1,028,155 chromatogram rows into one row group where `stages/extract.rs` writes 16
    /// (`CHROM_ROW_GROUP_ROWS = 1 << 16`); the A/B was internally fair but it was not the
    /// production configuration. `MUMDIA_BENCH_ROW_GROUP` overrides, `0` means uncapped.
    /// The cap is printed with every number.
    #[test]
    #[ignore = "benchmark; needs MUMDIA_BENCH_PARQUET"]
    fn bench_rewrite_a_real_artifact() {
        let Ok(src) = std::env::var("MUMDIA_BENCH_PARQUET") else {
            println!("set MUMDIA_BENCH_PARQUET to a real artifact to run this");
            return;
        };
        let source_row_group = {
            let f = std::fs::File::open(&src).unwrap();
            let b = ParquetRecordBatchReaderBuilder::try_new(f).unwrap();
            let md = b.metadata();
            (0..md.num_row_groups())
                .map(|i| md.row_group(i).num_rows() as usize)
                .max()
                .unwrap_or(0)
        };
        let cap = match std::env::var("MUMDIA_BENCH_ROW_GROUP") {
            Ok(v) => v.parse::<usize>().ok().filter(|n| *n > 0),
            Err(_) => Some(source_row_group).filter(|n| *n > 0),
        };
        let table = Table::read(&src).unwrap();
        let floats = table
            .schema
            .fields()
            .iter()
            .filter(|f| {
                matches!(f.data_type(), DataType::Float32 | DataType::Float64)
                    || matches!(f.data_type(), DataType::List(i) | DataType::LargeList(i)
                    if matches!(i.data_type(), DataType::Float32 | DataType::Float64))
            })
            .count();
        // One rewrite under `rule`: the file size, the encode time, and the writer's peak
        // `memory_size` (its in-progress row group: compressed pages plus the buffered values
        // of each column's open page), sampled after every batch.
        let rewrite = |path: &str, rule: DictRule| -> (u64, f64, usize) {
            let props = props_for(rule, &table.schema, cap, &table.batches);
            let t = Instant::now();
            let f = std::fs::File::create(path).unwrap();
            let mut w = ArrowWriter::try_new(f, table.schema.clone(), Some(props)).unwrap();
            let mut peak = 0usize;
            for b in &table.batches {
                w.write(b).unwrap();
                peak = peak.max(w.memory_size());
            }
            w.close().unwrap();
            let secs = t.elapsed().as_secs_f64();
            (std::fs::metadata(path).unwrap().len(), secs, peak)
        };
        // The data pages of a rewritten file, from its offset index, and one full read.
        let pages_and_read = |path: &str| -> (usize, f64) {
            let (_, meta) = splice_meta(path).unwrap();
            let pages = meta
                .offset_index()
                .map(|oi| {
                    oi.iter()
                        .flat_map(|rg| rg.iter().map(|c| c.page_locations().len()))
                        .sum()
                })
                .unwrap_or(0);
            let t = Instant::now();
            let mut rows = 0usize;
            for b in TableFile::open(path)
                .unwrap()
                .batches(None, 1 << 14)
                .unwrap()
            {
                rows += b.unwrap().num_rows();
            }
            assert_eq!(rows, table.nrows);
            (pages, t.elapsed().as_secs_f64())
        };
        // `MUMDIA_BENCH_RULES=default,writer` restricts the arms (the first is the baseline
        // the percentages are against); `MUMDIA_BENCH_REPEATS` (default 3) sets how many
        // interleaved rounds the median times come from.
        let wanted: Option<Vec<String>> = std::env::var("MUMDIA_BENCH_RULES")
            .ok()
            .map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
        let repeats: usize = std::env::var("MUMDIA_BENCH_REPEATS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3)
            .max(1);
        let rules: Vec<(&str, DictRule)> = DICT_RULES
            .iter()
            .copied()
            .filter(|(label, _)| wanted.as_ref().is_none_or(|w| w.iter().any(|x| x == label)))
            .collect();
        let p = tmp("real_rule.parquet");
        let median = |mut v: Vec<f64>| -> f64 {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let mut writes: Vec<Vec<f64>> = vec![Vec::new(); rules.len()];
        let mut reads: Vec<Vec<f64>> = vec![Vec::new(); rules.len()];
        let mut shape: Vec<(u64, usize, usize)> = vec![(0, 0, 0); rules.len()];
        for _ in 0..repeats {
            for (k, &(_, rule)) in rules.iter().enumerate() {
                let (n, secs, peak) = rewrite(&p, rule);
                let (pages, read) = pages_and_read(&p);
                writes[k].push(secs);
                reads[k].push(read);
                shape[k] = (n, pages, peak);
            }
        }
        std::fs::remove_file(&p).ok();
        let base = shape[0].0 as f64;
        println!(
            "{src}: {} rows x {} columns ({floats} float), row group {}; on disk {:.3} MB; median write and read of {repeats} interleaved rounds",
            table.nrows,
            table.schema.fields().len(),
            match cap {
                Some(n) => format!("{n} rows"),
                None => "uncapped".to_string(),
            },
            std::fs::metadata(&src).unwrap().len() as f64 / 1e6,
        );
        for (k, (label, _)) in rules.iter().enumerate() {
            let (n, pages, peak) = shape[k];
            println!(
                "  {label:>12} {:>10.3} MB ({:+6.1}%)  write {:.2} s  read {:.2} s  {pages} data pages  writer peak {:.1} MB",
                n as f64 / 1e6,
                100.0 * (n as f64 / base - 1.0),
                median(writes[k].clone()),
                median(reads[k].clone()),
                peak as f64 / 1e6,
            );
        }
    }

    /// The parallel column codec against the serial one on a REAL artifact, under the
    /// shipped writer properties at the source's own row-group size:
    ///
    /// ```text
    /// MUMDIA_BENCH_PARQUET=out_aif02/features.parquet MUMDIA_BENCH_ROW_GROUP=65536 \
    ///   cargo test -p mumdia-io --release -- --ignored --nocapture bench_parallel_encode
    /// ```
    ///
    /// The batches are re-chunked to 65,536 rows first (the decoded ones are 1,024), which
    /// is what the engine's stage writers hand the codec. Every arm must produce the serial
    /// arm's bytes; the median of `MUMDIA_BENCH_REPEATS` (default 3) interleaved rounds is
    /// printed per thread count.
    #[test]
    #[ignore = "benchmark; needs MUMDIA_BENCH_PARQUET"]
    fn bench_parallel_encode_a_real_artifact() {
        let Ok(src) = std::env::var("MUMDIA_BENCH_PARQUET") else {
            println!("set MUMDIA_BENCH_PARQUET to a real artifact to run this");
            return;
        };
        let cap = std::env::var("MUMDIA_BENCH_ROW_GROUP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0);
        let repeats: usize = std::env::var("MUMDIA_BENCH_REPEATS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3)
            .max(1);
        let table = Table::read(&src).unwrap();
        let mut chunks = Vec::new();
        let mut at = 0usize;
        let whole = arrow::compute::concat_batches(&table.schema, &table.batches).unwrap();
        while at < table.nrows {
            let k = (table.nrows - at).min(1 << 16);
            chunks.push(whole.slice(at, k));
            at += k;
        }
        let plan = cap.map(|c| EncodingPlan::of(&table.schema, &chunks, plan_sample_rows(c)));
        let props = || writer_props(&table.schema, cap, plan.as_ref(), &[]);
        let write = |threads: usize| -> (Vec<u8>, f64) {
            let pool = (threads > 1).then(|| {
                Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap(),
                )
            });
            let t = Instant::now();
            let mut out = Vec::with_capacity(1 << 28);
            let mut w =
                ColumnEncoder::try_new(&mut out, table.schema.clone(), props(), pool).unwrap();
            for b in &chunks {
                w.write(b).unwrap();
            }
            w.into_inner().unwrap();
            (out, t.elapsed().as_secs_f64())
        };
        let arms = [1usize, 2, 4, 8];
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        let mut reference: Option<Vec<u8>> = None;
        for _ in 0..repeats {
            for (k, &threads) in arms.iter().enumerate() {
                let (bytes, secs) = write(threads);
                match &reference {
                    None => reference = Some(bytes),
                    Some(r) => assert!(*r == bytes, "{threads} threads: bytes differ"),
                }
                times[k].push(secs);
            }
        }
        println!(
            "{src}: {} rows x {} columns, row group {cap:?}, {:.1} MB; median encode of \
             {repeats} rounds (every arm byte-identical to 1 thread):",
            table.nrows,
            table.schema.fields().len(),
            reference.as_ref().map_or(0, |r| r.len()) as f64 / 1e6
        );
        for (k, &threads) in arms.iter().enumerate() {
            let mut v = times[k].clone();
            v.sort_by(f64::total_cmp);
            println!("  {threads} threads: {:.2} s", v[v.len() / 2]);
        }
    }

    /// The four candidate dictionary rules against every column shape this engine writes.
    /// This is the measurement [`writer_props`] rests on, and the one that rejected both
    /// of the simpler rules.
    ///
    /// * `default` -- parquet-rs as shipped: a dictionary on every column, 1 MB limit.
    ///   The pre-change baseline.
    /// * `float-off` -- `set_column_dictionary_enabled(float_leaf, false)`. The first
    ///   attempt; it keys on physical type, but the discriminator is CARDINALITY, so it
    ///   regresses every low-cardinality float column by an order of magnitude.
    /// * `float-16K` -- `set_column_dictionary_page_size_limit(float_leaf, 16 KiB)`, what
    ///   this module now does.
    /// * `global-16K` -- the same limit set globally, on every leaf.
    ///
    /// Run it at both a capped and an uncapped row group, because the fallback threshold
    /// is per column chunk and the engine has writers of both kinds:
    ///
    /// ```text
    /// cargo test -p mumdia-io --release -- --ignored --nocapture bench_dictionary_rules
    /// ```
    #[test]
    #[ignore = "benchmark"]
    fn bench_dictionary_rules_by_column_shape() {
        let rows = 1_000_000usize;
        type Shape<'a> = (&'a str, Box<dyn Fn() -> Col>);
        let shapes: Vec<Shape> = vec![
            (
                "near-unique f64",
                Box::new(move || Col::F64("mz".into(), noise(11, rows))),
            ),
            (
                "constant f64",
                Box::new(move || Col::F64("const_feat".into(), vec![0.0; rows])),
            ),
            (
                "binary 0/1 f64",
                Box::new(move || {
                    Col::F64(
                        "is_modified".into(),
                        (0..rows).map(|i| (i % 2) as f64).collect(),
                    )
                }),
            ),
            (
                "small-int f64 (0..20)",
                Box::new(move || {
                    Col::F64(
                        "n_fragments".into(),
                        (0..rows).map(|i| (i % 21) as f64).collect(),
                    )
                }),
            ),
            (
                "quantised f32 (1,001 values)",
                Box::new(move || {
                    Col::F32(
                        "corr".into(),
                        (0..rows).map(|i| (i % 1001) as f32 / 1000.0).collect(),
                    )
                }),
            ),
            (
                "repeated iRT f64 (5 modforms per peptide)",
                Box::new(move || {
                    let base = noise(29, rows / 5 + 1);
                    Col::F64(
                        "predicted_irt".into(),
                        (0..rows).map(|i| base[i / 5]).collect(),
                    )
                }),
            ),
            (
                "unique i32",
                Box::new(move || Col::I32("id".into(), (0..rows as i32).collect())),
            ),
            (
                "run-length i32 (6 rows per id)",
                Box::new(move || {
                    Col::I32(
                        "candidate_id".into(),
                        (0..rows).map(|i| (i / 6) as i32).collect(),
                    )
                }),
            ),
            (
                "three-valued u32",
                Box::new(move || {
                    Col::U32(
                        "charge".into(),
                        (0..rows).map(|i| (i % 3) as u32 + 1).collect(),
                    )
                }),
            ),
            (
                "two-valued utf8",
                Box::new(move || {
                    Col::Str(
                        "label".into(),
                        (0..rows)
                            .map(|i| if i % 2 == 0 { "target" } else { "decoy" }.to_string())
                            .collect(),
                    )
                }),
            ),
            (
                "20,000-accession utf8",
                Box::new(move || {
                    Col::Str(
                        "protein".into(),
                        (0..rows)
                            .map(|i| format!("sp|P{:05}|PROT{:05}_HUMAN", i % 20_000, i % 20_000))
                            .collect(),
                    )
                }),
            ),
        ];
        for cap in [Some(131_072usize), None] {
            println!(
                "--- row group {} ---",
                match cap {
                    Some(n) => format!("{n} rows"),
                    None => "uncapped (parquet-rs 1,048,576)".to_string(),
                }
            );
            for (what, make) in &shapes {
                let sizes: Vec<(&str, u64)> = DICT_RULES
                    .iter()
                    .map(|&(label, rule)| {
                        let p = tmp("rule_shape.parquet");
                        let n = write_under(&p, vec![make()], rule, cap);
                        std::fs::remove_file(&p).ok();
                        (label, n)
                    })
                    .collect();
                let base = sizes[0].1 as f64;
                let rendered = sizes
                    .iter()
                    .map(|(label, n)| {
                        format!(
                            "{label} {:.3} MB ({:+.0}%)",
                            *n as f64 / 1e6,
                            100.0 * (*n as f64 / base - 1.0)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                println!("{what:42} {rendered}");
            }
        }
    }

    /// How [`FLOAT_DICTIONARY_PAGE_SIZE_LIMIT`] was chosen. The limit is a cardinality
    /// threshold, so sweep the cardinality against it: an f64 column of 1,000,000 rows
    /// cycling through `d` distinct values, at the 131,072-row cap, for each candidate
    /// limit. A column whose per-row-group cardinality sits below the limit keeps its
    /// dictionary and is untouched; one above it falls back and pays a dictionary page
    /// plus indices for the prefix before the fallback, which is a LOSS whenever the full
    /// dictionary would still have fitted under parquet-rs's 1 MB.
    ///
    /// The loss band is what sizes the constant, and the engine writes into it: a smoke
    /// fragment library's `mz` is 7,002 distinct over 22,920 rows.
    #[test]
    #[ignore = "benchmark"]
    fn bench_float_dictionary_limit_by_cardinality() {
        for &(rows, cap) in &[(1_000_000usize, 65_536usize), (1_000_000, 131_072)] {
            // Limits as a fraction of the row group, because the break-even is
            // cardinality against CHUNK ROWS, not an absolute byte count.
            let limits: Vec<(String, usize)> = [8usize, 4, 2]
                .iter()
                .map(|d| (format!("R/{d}"), cap / d * 8))
                .chain([
                    ("6R (0.75R)".to_string(), cap * 6),
                    ("16K".to_string(), 16 * 1024),
                ])
                .collect();
            println!(
                "\nf64, {rows} rows, row group {cap}. Per-column dictionary limit as a \
                 fraction of the row group (R/8 = fall back above R/8 distinct):"
            );
            print!("{:>14} {:>10} {:>9}", "distinct/R", "default", "off");
            for (name, _) in &limits {
                print!(" {name:>9}");
            }
            println!();
            for frac in [0.01f64, 0.05, 0.125, 0.25, 0.5, 0.75, 1.0] {
                let d = ((cap as f64 * frac) as usize).max(1);
                let values = noise(97, d);
                let make = || {
                    Col::F64(
                        "x".into(),
                        (0..rows).map(|i| values[i % values.len()]).collect(),
                    )
                };
                let one = |props: WriterProperties| -> u64 {
                    let p = tmp("card.parquet");
                    let (schema, batch) = cols_to_batch(&p, vec![make()]).unwrap();
                    let f = std::fs::File::create(&p).unwrap();
                    let mut w = ArrowWriter::try_new(f, schema, Some(props)).unwrap();
                    w.write(&batch).unwrap();
                    w.close().unwrap();
                    let n = std::fs::metadata(&p).unwrap().len();
                    std::fs::remove_file(&p).ok();
                    n
                };
                let schema = cols_to_batch("card", vec![make()]).unwrap().0;
                let base = one(props_under(DictRule::Default, &schema, Some(cap))) as f64;
                let off = one(props_under(DictRule::FloatOff, &schema, Some(cap))) as f64;
                print!(
                    "{:>9} {frac:>4.2} {:>9.3}M {:>8.1}%",
                    d,
                    base / 1e6,
                    100.0 * (off / base - 1.0)
                );
                for (_, l) in &limits {
                    let mut b = WriterProperties::builder()
                        .set_compression(codec())
                        .set_max_row_group_row_count(Some(cap));
                    for (leaf, _) in float_leaf_paths(&schema) {
                        b = b.set_column_dictionary_page_size_limit(leaf, *l);
                    }
                    let n = one(b.build()) as f64;
                    print!(" {:>8.1}%", 100.0 * (n / base - 1.0));
                }
                println!();
            }
        }
    }

    /// A features-shaped table under the same four rules: the composition CLAUDE.md
    /// records for the 387 Extended features, at the cap `features.rs` actually uses.
    #[test]
    #[ignore = "benchmark"]
    fn bench_dictionary_rules_on_a_features_shaped_table() {
        // Above parquet-rs's 131,072-distinct-f64 dictionary limit, so the uncapped arm is
        // in the regime where the DEFAULT already falls back part-way through the chunk.
        let rows = 200_000usize;
        let cols = || -> Vec<Col> {
            let mut c = vec![
                Col::U32("candidate_id".into(), (0..rows as u32).collect()),
                Col::U32(
                    "charge".into(),
                    (0..rows).map(|i| (i % 3) as u32 + 2).collect(),
                ),
                Col::Str(
                    "label".into(),
                    (0..rows)
                        .map(|i| if i % 2 == 0 { "target" } else { "decoy" }.to_string())
                        .collect(),
                ),
            ];
            // 300 near-unique f64, the bulk of the table.
            for j in 0..300 {
                c.push(Col::F64(format!("f{j}"), noise(j as u64 + 7, rows)));
            }
            // 11 constant f64: `MUMDIA_NN_DROP_CONSTANT` counts 11 of 387 on the Astral pool.
            for j in 0..11 {
                c.push(Col::F64(format!("const{j}"), vec![j as f64; rows]));
            }
            // 40 indicator f64 and 36 small-count f64.
            for j in 0..40 {
                c.push(Col::F64(
                    format!("ind{j}"),
                    (0..rows).map(|i| ((i + j) % 2) as f64).collect(),
                ));
            }
            for j in 0..36 {
                c.push(Col::F64(
                    format!("cnt{j}"),
                    (0..rows).map(|i| ((i + j) % 21) as f64).collect(),
                ));
            }
            c
        };
        // Capped is `features.parquet` and `psms_competed.parquet`; uncapped is
        // `psms_scored.parquet`, which goes through `BatchWriter::new` and inherits
        // parquet-rs's 1,048,576-row group. The two are not the same measurement, because
        // the default already falls back to PLAIN part-way through an uncapped near-unique
        // float chunk.
        for cap in [Some(65_536usize), None] {
            let mut base = 0f64;
            for &(label, rule) in DICT_RULES {
                let p = tmp("features_shaped.parquet");
                let t = Instant::now();
                let n = write_under(&p, cols(), rule, cap);
                let secs = t.elapsed().as_secs_f64();
                std::fs::remove_file(&p).ok();
                if base == 0.0 {
                    base = n as f64;
                }
                println!(
                    "features-shaped {rows} x 390 ({}): {label} {:.1} MB ({:+.1}%), \
                     {:.0} bytes per row, write {secs:.2} s",
                    if cap.is_some() { "capped" } else { "uncapped" },
                    n as f64 / 1e6,
                    100.0 * (n as f64 / base - 1.0),
                    n as f64 / rows as f64,
                );
            }
        }
    }

    /// The chromatogram read path: `ListF32` resolved once against the per-row
    /// `ArrayRef`. Both arms walk the same decoded batches and copy the same values into
    /// the same pre-reserved buffer; only the way the row's bounds are reached differs.
    #[test]
    #[ignore = "benchmark"]
    fn bench_list_view_against_the_per_row_arrayref() {
        let (rows, per_row) = (200_000usize, 40usize);
        let p = tmp("traces.parquet");
        write_table(
            &p,
            vec![
                Col::U32("id".into(), (0..rows as u32).collect()),
                Col::LargeListF32(
                    "rt".into(),
                    (0..rows)
                        .map(|i| (0..per_row).map(|k| (i + k) as f32).collect())
                        .collect(),
                ),
            ],
        )
        .unwrap();
        let table = Table::read(&p).unwrap();
        let total = rows * per_row;

        let mut out = Vec::with_capacity(total);
        let t = Instant::now();
        for b in &table.batches {
            let col = b.column(b.schema().index_of("rt").unwrap()).clone();
            let v = ListF32::of(&col, "rt").unwrap();
            for k in 0..v.len() {
                v.append_row(k, &mut out, "rt").unwrap();
            }
        }
        let new = t.elapsed().as_secs_f64();
        assert_eq!(out.len(), total);

        // The previous implementation, verbatim: one owned `ArrayRef` per row.
        let mut old_out = Vec::with_capacity(total);
        let t = Instant::now();
        for b in &table.batches {
            let col = b.column(b.schema().index_of("rt").unwrap()).clone();
            let a = col.as_any().downcast_ref::<LargeListArray>().unwrap();
            for k in 0..a.len() {
                if a.is_null(k) {
                    continue;
                }
                let v = a.value(k);
                let f = v.as_any().downcast_ref::<Float32Array>().unwrap();
                old_out.extend_from_slice(f.values());
            }
        }
        let old = t.elapsed().as_secs_f64();
        assert_eq!(old_out, out);
        println!(
            "list rows {rows} x {per_row}: resolved once {:.1} ms, ArrayRef per row {:.1} ms \
             ({:.2}x), {:.1} ns per row saved",
            new * 1e3,
            old * 1e3,
            old / new,
            (old - new) * 1e9 / rows as f64,
        );
        std::fs::remove_file(&p).ok();
    }
}
