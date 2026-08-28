//! JSON read/write for scalars, config, and reports (docs/03_io_layer.md).

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Write JSON atomically: to a sibling temp file, then rename.
///
/// Same contract as the parquet writers (`table::AtomicPath`). It matters most for
/// `manifest.json`, which is the provenance record: a half-written manifest is worse
/// than none, because it parses far enough to look authoritative.
pub fn write_json<T: Serialize>(path: &str, value: &T) -> Result<()> {
    let target = crate::table::AtomicPath::new(path)?;
    let s = serde_json::to_string_pretty(value)?;
    std::fs::write(target.tmp(), s)
        .with_context(|| format!("writing json {}", target.tmp().display()))?;
    target.publish()?;
    Ok(())
}

pub fn read_json<T: DeserializeOwned>(path: &str) -> Result<T> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading json {path}"))?;
    let v = serde_json::from_str(&s).with_context(|| format!("parsing json {path}"))?;
    Ok(v)
}
