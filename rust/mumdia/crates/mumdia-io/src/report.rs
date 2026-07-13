//! Per-artifact `<artifact>.report.json` (PLAN.md Section 3.5 rule 2) so a stage
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
}
