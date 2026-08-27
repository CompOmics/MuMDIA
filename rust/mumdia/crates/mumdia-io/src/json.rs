//! JSON read/write for scalars, config, and reports (docs/03_io_layer.md).

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

pub fn write_json<T: Serialize>(path: &str, value: &T) -> Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let s = serde_json::to_string_pretty(value)?;
    std::fs::write(path, s).with_context(|| format!("writing json {path}"))?;
    Ok(())
}

pub fn read_json<T: DeserializeOwned>(path: &str) -> Result<T> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading json {path}"))?;
    let v = serde_json::from_str(&s).with_context(|| format!("parsing json {path}"))?;
    Ok(v)
}
