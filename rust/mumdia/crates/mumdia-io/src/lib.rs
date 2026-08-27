//! mumdia-io: on-disk contracts (Parquet, JSON), content hashing, logging, and
//! the `inspect` helper (docs/03_io_layer.md).

pub mod hash;
pub mod json;
pub mod report;
pub mod table;

use anyhow::Result;
use mumdia_core::manifest::ArtifactRecord;

/// Initialize tracing once, honoring `RUST_LOG` (default `info`).
pub fn init_logging() {
    init_logging_level(None)
}

/// Initialize tracing with an explicit level, falling back to `RUST_LOG` and then
/// to `info`.
///
/// `RUST_LOG` was the only way to change verbosity, which is not discoverable
/// from `--help` and is awkward on Windows. An explicit level wins over it, so
/// `--log-level` and `-v/-q` behave as the flags a user expects; `RUST_LOG` still
/// works, and still offers per-module filtering that a single level cannot.
pub fn init_logging_level(level: Option<&str>) {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = match level {
        Some(l) => EnvFilter::new(l),
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
    };
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}

/// Build an [`ArtifactRecord`] by hashing the written file (docs/03_io_layer.md).
pub fn record_artifact(
    logical_name: &str,
    schema: (&str, u32),
    path: &str,
    rows: u64,
    stage: &str,
    config_hash: &str,
) -> Result<ArtifactRecord> {
    Ok(ArtifactRecord {
        logical_name: logical_name.to_string(),
        path: path.to_string(),
        format: "parquet".to_string(),
        schema_name: schema.0.to_string(),
        schema_version: schema.1,
        rows,
        content_hash: hash::blake3_file(path)?,
        producing_stage: stage.to_string(),
        config_hash: config_hash.to_string(),
    })
}

/// Human-readable schema + head sample + row count for any Parquet artifact
/// (`mumdia inspect`, docs/03_io_layer.md).
pub fn inspect(path: &str) -> Result<String> {
    let t = table::Table::read(path)?;
    let mut s = String::new();
    s.push_str(&format!("artifact: {path}\n"));
    s.push_str(&format!("rows: {}\n", t.nrows));
    s.push_str("schema:\n");
    for f in t.schema.fields() {
        s.push_str(&format!(
            "  {}: {:?}{}\n",
            f.name(),
            f.data_type(),
            if f.is_nullable() { " (nullable)" } else { "" }
        ));
    }
    if let Some(first) = t.batches.first() {
        let head_rows = first.num_rows().min(10);
        let head = first.slice(0, head_rows);
        if let Ok(p) = arrow::util::pretty::pretty_format_batches(&[head]) {
            s.push_str("head:\n");
            s.push_str(&p.to_string());
            s.push('\n');
        }
    }
    Ok(s)
}
