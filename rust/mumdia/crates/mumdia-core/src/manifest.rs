//! Run manifest (PLAN.md Section 3.3, 3.5): per-artifact provenance so a chained
//! run is reproducible and reruns skip unchanged work. Provenance is recorded,
//! not required: because inputs are path-addressable, a stage never depends on
//! the manifest to run.

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
