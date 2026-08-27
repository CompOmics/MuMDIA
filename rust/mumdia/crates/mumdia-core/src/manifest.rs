//! Run manifest (docs/03_io_layer.md): per-artifact provenance for a chained
//! run. The current orchestrator records but does not consume the manifest: it
//! does not cache, resume, or skip unchanged stages. Standalone stages remain
//! reusable because their inputs are path-addressable.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub logical_name: String,
    pub path: String,
    pub format: String,
    pub schema_name: String,
    pub schema_version: u32,
    pub rows: u64,
    pub content_hash: String,
    pub producing_stage: String,
    pub config_hash: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub mumdia_version: String,
    /// Resolved config JSON (fully expanded), and its hash.
    pub config_json: String,
    pub config_hash: String,
    /// Sidecar / predictor model identities recorded per stage.
    pub model_identities: BTreeMap<String, String>,
    pub artifacts: BTreeMap<String, ArtifactRecord>,
}

impl Manifest {
    pub fn new(config_json: String, config_hash: String) -> Self {
        Self {
            mumdia_version: env!("CARGO_PKG_VERSION").to_string(),
            config_json,
            config_hash,
            model_identities: BTreeMap::new(),
            artifacts: BTreeMap::new(),
        }
    }

    pub fn record(&mut self, r: ArtifactRecord) {
        self.artifacts.insert(r.logical_name.clone(), r);
    }

    pub fn get(&self, logical_name: &str) -> Option<&ArtifactRecord> {
        self.artifacts.get(logical_name)
    }
}
