//! Per-artifact `<artifact>.report.json` (docs/03_io_layer.md) so a stage
//! can be evaluated without loading the full table: row counts, key
//! distributions, the parameters used, model identity, and timing.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactReport {
    pub logical_name: String,
    pub schema_name: String,
    pub schema_version: u32,
    pub stage: String,
    pub rows: u64,
    pub content_hash: String,
    /// Parameters the stage actually used (resolved).
    pub params: Value,
    /// Summary key distributions / metrics.
    pub stats: BTreeMap<String, Value>,
    pub model_identity: Option<String>,
    pub elapsed_ms: u128,
}

impl ArtifactReport {
    /// Write next to the artifact as `<artifact>.report.json`.
    pub fn write_for(&self, artifact_path: &str) -> Result<()> {
        let report_path = format!("{artifact_path}.report.json");
        crate::json::write_json(&report_path, self)
    }

    /// The row count and content hash this report carries, for a caller that records
    /// the same artifact elsewhere.
    pub fn written(&self) -> Written {
        Written {
            rows: self.rows,
            content_hash: self.content_hash.clone(),
        }
    }
}

/// An artifact a stage has just written and reported: its row count and the content hash
/// the stage computed for its `<artifact>.report.json`.
///
/// Every stage hashes each output once for its report, after the file is published. The
/// orchestrators record the same files in the run manifest, and they used to do that with
/// [`crate::record_artifact`], which reads and hashes the whole file a second time. A stage
/// that returns this lets the caller record with [`Written::record`] instead, so the
/// manifest carries the identical hash for one read of the file rather than two.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Written {
    pub rows: u64,
    pub content_hash: String,
}

impl Written {
    /// The manifest record of `path` from this hash, without reading the file.
    pub fn record(
        &self,
        logical_name: &str,
        schema: (&str, u32),
        path: &str,
        stage: &str,
        config_hash: &str,
    ) -> mumdia_core::manifest::ArtifactRecord {
        crate::record_artifact_with_hash(
            logical_name,
            schema,
            path,
            self.rows,
            stage,
            config_hash,
            self.content_hash.clone(),
        )
    }
}
