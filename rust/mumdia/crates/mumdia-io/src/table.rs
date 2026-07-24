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
                let mut b = ListBuilder::new(Float32Builder::new());
                for row in &v {
                    b.values().append_slice(row);
                    b.append(true);
                }
                Arc::new(b.finish())
            }
            Col::ListF64(_, v) => {
                use arrow::array::Float64Builder;
                let mut b = ListBuilder::new(Float64Builder::new());
                for row in &v {
                    b.values().append_slice(row);
                    b.append(true);
                }
                Arc::new(b.finish())
            }
            Col::LargeListF32(_, v) => {
                let mut b = LargeListBuilder::new(Float32Builder::new());
                for row in &v {
                    b.values().append_slice(row);
                    b.append(true);
                }
                Arc::new(b.finish())
            }
        }
    }
}

/// Write columns to a Parquet file. Returns the row count. All columns must
/// share the same length.
pub fn write_table(path: &str, cols: Vec<Col>) -> Result<u64> {
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

    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = std::fs::File::create(path).with_context(|| format!("creating {path}"))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(nrows as u64)
}

/// A read-back table: all batches concatenated logically, accessed by column
/// name with typed getters.
pub struct Table {
    pub schema: Arc<Schema>,
    pub batches: Vec<RecordBatch>,
    pub nrows: usize,
}

impl Table {
    pub fn read(path: &str) -> Result<Table> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("reading parquet {path}"))?;
        let schema = builder.schema().clone();
        let reader = builder.build()?;
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
            let a = b
                .column(i)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| anyhow!("column '{name}' is not f64"))?;
            if a.null_count() == 0 {
                out.extend_from_slice(a.values());
            } else {
                for k in 0..a.len() {
                    out.push(if a.is_null(k) { f64::NAN } else { a.value(k) });
                }
            }
        }
        Ok(out)
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            let a = b
                .column(i)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow!("column '{name}' is not f32"))?;
            if a.null_count() == 0 {
                out.extend_from_slice(a.values());
            } else {
                for k in 0..a.len() {
                    out.push(if a.is_null(k) { f32::NAN } else { a.value(k) });
                }
            }
        }
        Ok(out)
    }

    pub fn i64(&self, name: &str) -> Result<Vec<i64>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            let a = b
                .column(i)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| anyhow!("column '{name}' is not i64"))?;
            if a.null_count() == 0 {
                out.extend_from_slice(a.values());
            } else {
                for k in 0..a.len() {
                    out.push(a.value(k));
                }
            }
        }
        Ok(out)
    }

    pub fn i32(&self, name: &str) -> Result<Vec<i32>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            let a = b
                .column(i)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| anyhow!("column '{name}' is not i32"))?;
            if a.null_count() == 0 {
                out.extend_from_slice(a.values());
            } else {
                for k in 0..a.len() {
                    out.push(a.value(k));
                }
            }
        }
        Ok(out)
    }

    pub fn u32(&self, name: &str) -> Result<Vec<u32>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            let a = b
                .column(i)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| anyhow!("column '{name}' is not u32"))?;
            if a.null_count() == 0 {
                out.extend_from_slice(a.values());
            } else {
                for k in 0..a.len() {
                    out.push(a.value(k));
                }
            }
        }
        Ok(out)
    }

    pub fn bool(&self, name: &str) -> Result<Vec<bool>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            let a = b
                .column(i)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| anyhow!("column '{name}' is not bool"))?;
            for k in 0..a.len() {
                out.push(a.value(k));
            }
        }
        Ok(out)
    }

    pub fn str(&self, name: &str) -> Result<Vec<String>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            let a = b
                .column(i)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("column '{name}' is not utf8"))?;
            for k in 0..a.len() {
                out.push(if a.is_null(k) {
                    String::new()
                } else {
                    a.value(k).to_string()
                });
            }
        }
        Ok(out)
    }

    pub fn opt_f64(&self, name: &str) -> Result<Vec<Option<f64>>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        for b in &self.batches {
            let a = b
                .column(i)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| anyhow!("column '{name}' is not f64"))?;
            for k in 0..a.len() {
                out.push(if a.is_null(k) { None } else { Some(a.value(k)) });
            }
        }
        Ok(out)
    }

    /// Read an f32 list column. Accepts both `List` (32-bit offsets) and
    /// `LargeList` (64-bit offsets, written by `Col::LargeListF32`) encodings,
    /// so chromatogram artifacts written by either binary read back the same.
    pub fn list_f32(&self, name: &str) -> Result<Vec<Vec<f32>>> {
        let i = self.idx(name)?;
        let mut out = Vec::with_capacity(self.nrows);
        let push_inner = |out: &mut Vec<Vec<f32>>, v: ArrayRef| -> Result<()> {
            let f = v
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow!("list '{name}' inner is not f32"))?;
            out.push(f.values().to_vec());
            Ok(())
        };
        for b in &self.batches {
            let col = b.column(i);
            if let Some(a) = col.as_any().downcast_ref::<LargeListArray>() {
                for k in 0..a.len() {
                    if a.is_null(k) {
                        out.push(Vec::new());
                    } else {
                        push_inner(&mut out, a.value(k))?;
                    }
                }
            } else if let Some(a) = col.as_any().downcast_ref::<ListArray>() {
                for k in 0..a.len() {
                    if a.is_null(k) {
                        out.push(Vec::new());
                    } else {
                        push_inner(&mut out, a.value(k))?;
                    }
                }
            } else {
                return Err(anyhow!("column '{name}' is not a list"));
            }
        }
        Ok(out)
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
