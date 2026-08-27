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

/// One run input, hashed so a result can be tied back to the exact bytes it came
/// from. The engine's own artifacts were always recorded this way; its inputs were
/// not, so a manifest could not answer "which mzML produced this?".
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputRecord {
    pub path: String,
    pub bytes: u64,
    pub content_hash: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub mumdia_version: String,
    /// Source identity of the binary: the short commit it was built from, with a
    /// `-dirty` suffix when the worktree carried uncommitted changes, and that
    /// commit's date. `unknown` when the build had no git available. Stamped by
    /// this crate's `build.rs`.
    #[serde(default)]
    pub git_sha: String,
    #[serde(default)]
    pub commit_date: String,
    /// The command line, so a run can be repeated without reconstructing it from
    /// memory. Flags that are not in the config (`--top-peaks-ms2`, `--threads`,
    /// `--max-spectra`) live only here.
    #[serde(default)]
    pub cli_args: Vec<String>,
    /// Resolved config JSON (fully expanded), and its hash.
    pub config_json: String,
    pub config_hash: String,
    /// Sidecar / predictor model identities recorded per stage.
    pub model_identities: BTreeMap<String, String>,
    /// Hashed run inputs, keyed by role (`mzml`, `fasta`, `lib_precursors`, ...).
    #[serde(default)]
    pub inputs: BTreeMap<String, InputRecord>,
    pub artifacts: BTreeMap<String, ArtifactRecord>,
}

impl Manifest {
    pub fn new(config_json: String, config_hash: String) -> Self {
        Self {
            mumdia_version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha: env!("MUMDIA_GIT_SHA").to_string(),
            commit_date: env!("MUMDIA_COMMIT_DATE").to_string(),
            // Read here rather than threaded down from `main`: a manifest is only
            // ever built inside a CLI invocation, and every call site would
            // otherwise have to carry the same value unchanged.
            cli_args: std::env::args().collect(),
            config_json,
            config_hash,
            model_identities: BTreeMap::new(),
            inputs: BTreeMap::new(),
            artifacts: BTreeMap::new(),
        }
    }

    /// Record a hashed input. `role` is the logical slot, not the flag name, so a
    /// reader does not need to know which CLI spelling was used.
    pub fn record_input(&mut self, role: &str, path: &str, bytes: u64, content_hash: String) {
        self.inputs.insert(
            role.to_string(),
            InputRecord {
                path: path.to_string(),
                bytes,
                content_hash,
            },
        );
    }

    /// The one-line provenance stamp a benchmark record should quote.
    pub fn provenance(&self) -> String {
        format!(
            "mumdia {} ({}, {})",
            self.mumdia_version, self.git_sha, self.commit_date
        )
    }

    pub fn record(&mut self, r: ArtifactRecord) {
        self.artifacts.insert(r.logical_name.clone(), r);
    }

    pub fn get(&self, logical_name: &str) -> Option<&ArtifactRecord> {
        self.artifacts.get(logical_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(name: &str) -> ArtifactRecord {
        ArtifactRecord {
            logical_name: name.to_string(),
            path: format!("out/{name}.parquet"),
            format: "parquet".into(),
            schema_name: name.to_string(),
            schema_version: 1,
            rows: 7,
            content_hash: "a".repeat(64),
            producing_stage: "test".into(),
            config_hash: "b".repeat(64),
        }
    }

    #[test]
    fn new_stamps_the_build_identity() {
        let m = Manifest::new("{}".into(), "c".repeat(64));
        assert_eq!(m.mumdia_version, env!("CARGO_PKG_VERSION"));
        // build.rs always sets these, falling back to "unknown" without git, so an
        // empty value means the stamp was lost rather than unavailable.
        assert!(!m.git_sha.is_empty(), "git_sha not stamped");
        assert!(!m.commit_date.is_empty(), "commit_date not stamped");
        // Under `cargo test` the arguments are the harness's, which is fine: the
        // assertion is that something was captured, not what.
        assert!(!m.cli_args.is_empty(), "cli_args not captured");
        assert!(m.provenance().contains(&m.git_sha));
        assert!(m.provenance().starts_with("mumdia "));
    }

    #[test]
    fn inputs_and_artifacts_are_keyed_and_ordered() {
        let mut m = Manifest::new("{}".into(), "c".repeat(64));
        m.record_input("mzml", "run.mzML", 1234, "d".repeat(64));
        m.record_input("fasta", "proteome.fasta", 99, "e".repeat(64));
        m.record(record("features"));
        m.record(record("chromatograms"));

        assert_eq!(m.inputs["mzml"].bytes, 1234);
        assert_eq!(m.inputs["fasta"].path, "proteome.fasta");
        assert_eq!(m.get("features").unwrap().rows, 7);
        // BTreeMap, so the JSON key order is fixed rather than hash-dependent.
        // Determinism of the manifest matters: it is hashed and diffed.
        let keys: Vec<&str> = m.artifacts.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["chromatograms", "features"]);
        let ikeys: Vec<&str> = m.inputs.keys().map(String::as_str).collect();
        assert_eq!(ikeys, vec!["fasta", "mzml"]);
    }

    #[test]
    fn an_older_manifest_without_the_provenance_fields_still_parses() {
        // Manifests already exist on disk from before these fields were added.
        // They must keep loading, which is what `#[serde(default)]` buys; without
        // it every prior run's manifest would become unreadable.
        let old = r#"{
            "mumdia_version": "0.1.0",
            "config_json": "{}",
            "config_hash": "abc",
            "model_identities": {"rescorer": "native-percolator-lite-v1"},
            "artifacts": {}
        }"#;
        let m: Manifest = serde_json::from_str(old).expect("older manifest must parse");
        assert_eq!(m.mumdia_version, "0.1.0");
        assert!(m.git_sha.is_empty());
        assert!(m.cli_args.is_empty());
        assert!(m.inputs.is_empty());
        assert_eq!(m.model_identities["rescorer"], "native-percolator-lite-v1");
    }

    #[test]
    fn round_trips_through_json() {
        let mut m = Manifest::new("{\"a\":1}".into(), "c".repeat(64));
        m.record_input("mzml", "run.mzML", 5, "f".repeat(64));
        m.record(record("features"));
        m.model_identities
            .insert("rescorer".into(), "native-percolator-lite-v1".into());
        let json = serde_json::to_string(&m).unwrap();
        let back: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.config_hash, m.config_hash);
        assert_eq!(
            back.inputs["mzml"].content_hash,
            m.inputs["mzml"].content_hash
        );
        assert_eq!(back.artifacts["features"].rows, 7);
        assert_eq!(back.git_sha, m.git_sha);
    }
}
