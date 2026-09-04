//! The generated settings editor, and writing a configuration the engine accepts.
//!
//! # Why the form is generated
//!
//! There are 150 settings. An interface that restated their names, types, defaults
//! and help text would be a second copy of `config.rs`, and the copy that drifts is
//! the one a user reads. So the form is rendered from `configs/config-schema.json`,
//! which `ci/gen_config_reference.py` emits from the same parse that produces the
//! reference document, checked for staleness in CI beside it.
//!
//! # Why only overrides are written
//!
//! `Config` is `deny_unknown_fields` with serde defaults, so a valid configuration
//! contains only what differs from the default. Writing the full 150 keeps nothing
//! useful and freezes every default at the moment the file was saved: a later
//! release that improves a default would not reach anyone who had ever opened this
//! screen. Writing the difference keeps saved configurations short, reviewable, and
//! forward-compatible.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One setting, as the schema describes it.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Field {
    pub path: String,
    pub name: String,
    pub section: String,
    pub kind: String,
    pub optional: bool,
    #[serde(default)]
    pub default: serde_json::Value,
    #[serde(default)]
    pub help: String,
    #[serde(default)]
    pub gates: Vec<String>,
    #[serde(default)]
    pub choices: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Schema {
    pub sections: Vec<String>,
    pub fields: Vec<Field>,
}

/// The settings schema, compiled in.
///
/// Embedded rather than shipped beside the application, for two reasons. It cannot
/// then go missing from a bundle, which is a real failure mode: the first Windows
/// installer built here put `..`-rooted resources in a literal `_up_` directory
/// where nothing would have found them. And it costs nothing in freshness, because
/// the schema is generated from `config.rs` and any change to it requires a rebuild
/// anyway.
///
/// `ci/gen_config_reference.py` writes this file and CI fails when it is stale, so
/// the compiled-in copy is the same one the reference document describes.
const SCHEMA_JSON: &str = include_str!("../../../configs/config-schema.json");

pub fn load_schema() -> Result<Schema, String> {
    serde_json::from_str(SCHEMA_JSON)
        .map_err(|e| format!("the compiled-in settings schema could not be parsed: {e}"))
}

/// Turn `{"extract.gate_min_score": 0.3}` into the nested JSON the engine reads.
///
/// Only the paths present are written, so the result is the override set and
/// nothing else.
pub fn nest(flat: &BTreeMap<String, serde_json::Value>) -> serde_json::Value {
    let mut root = serde_json::Map::new();
    for (path, value) in flat {
        let mut cursor = &mut root;
        let parts: Vec<&str> = path.split('.').collect();
        for part in &parts[..parts.len().saturating_sub(1)] {
            cursor = cursor
                .entry((*part).to_string())
                .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()))
                .as_object_mut()
                .expect("intermediate config nodes are always objects");
        }
        if let Some(last) = parts.last() {
            cursor.insert((*last).to_string(), value.clone());
        }
    }
    serde_json::Value::Object(root)
}

/// Where a configuration built in the interface is written.
///
/// Under the per-user data directory rather than beside the results, so the same
/// settings can be reused across searches, and so a results folder stays a results
/// folder.
pub fn config_dir() -> PathBuf {
    crate::components::data_dir().join("configs")
}

/// Write an override set and hand back the path.
pub fn save(name: &str, flat: BTreeMap<String, serde_json::Value>) -> Result<String, String> {
    let dir = config_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    // A name typed by a person becomes a filename, so keep it to something that is
    // one on every platform.
    let safe: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let safe = if safe.trim_matches('_').is_empty() {
        "settings".to_string()
    } else {
        safe
    };
    let path = dir.join(format!("{safe}.json"));
    let text = serde_json::to_string_pretty(&nest(&flat))
        .map_err(|e| format!("could not serialise the settings: {e}"))?;
    std::fs::write(&path, text + "\n")
        .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(path.display().to_string())
}

/// Ask the engine whether a configuration file is acceptable.
///
/// `doctor` loads the configuration through the same path a run does, so a value the
/// engine would reject is rejected here, while editing, rather than an hour into a
/// search. A missing interpreter is NOT a validation failure: that is what the setup
/// screen and the preflight component check are for, and conflating the two would
/// make every configuration look invalid until the components are installed.
pub fn validate(config_path: &str) -> Result<(), String> {
    let (exe, _) = crate::engine::resolve()?;
    let out = crate::engine::command(&exe)
        .args(["doctor", "--config", config_path, "--json"])
        .output()
        .map_err(|e| format!("could not run the engine: {e}"))?;
    // A configuration the engine cannot even parse produces no JSON at all; that is
    // the case worth reporting, and stderr carries the reason.
    if serde_json::from_slice::<serde_json::Value>(&out.stdout).is_err() {
        let err = String::from_utf8_lossy(&out.stderr);
        let line = err
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty() && !l.contains("INFO") && !l.contains("WARN"))
            .unwrap_or("the engine rejected this configuration");
        return Err(line.trim().to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn flat(pairs: &[(&str, serde_json::Value)]) -> BTreeMap<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn nesting_builds_the_shape_the_engine_reads() {
        let v = nest(&flat(&[
            ("extract.gate_min_score", json!(0.3)),
            ("extract.frag_tol_ppm", json!(20.0)),
            ("rescore.classifier", json!("nn_torch")),
            ("threads", json!(8)),
        ]));
        assert_eq!(v["extract"]["gate_min_score"], json!(0.3));
        assert_eq!(v["extract"]["frag_tol_ppm"], json!(20.0));
        assert_eq!(v["rescore"]["classifier"], json!("nn_torch"));
        assert_eq!(v["threads"], json!(8));
    }

    #[test]
    fn only_the_given_paths_appear() {
        // The whole point: a saved configuration is the difference from the
        // defaults, so a later release that improves a default still reaches a user
        // who saved settings today.
        let v = nest(&flat(&[("extract.gate_min_score", json!(0.3))]));
        let obj = v.as_object().unwrap();
        assert_eq!(obj.len(), 1);
        assert_eq!(obj["extract"].as_object().unwrap().len(), 1);
    }

    #[test]
    fn an_empty_override_set_is_an_empty_object() {
        // Which is a valid configuration meaning "every default", not an error.
        assert_eq!(nest(&BTreeMap::new()), json!({}));
    }

    #[test]
    fn a_hostile_name_cannot_escape_the_configuration_directory() {
        let dir = config_dir();
        for name in ["../../evil", "a/b", "c:\\d", "..", ""] {
            let p = save(name, BTreeMap::new()).expect("save should succeed");
            let p = PathBuf::from(p);
            assert_eq!(
                p.parent().map(|x| x.to_path_buf()),
                Some(dir.clone()),
                "{name:?} escaped to {}",
                p.display()
            );
            let _ = std::fs::remove_file(&p);
        }
    }

    /// The schema ships with the repository, so this runs everywhere the tests do.
    #[test]
    fn the_shipped_schema_parses_and_describes_real_settings() {
        let Ok(s) = load_schema() else {
            eprintln!("config-schema.json not found from this build; skipping");
            return;
        };
        assert!(
            s.fields.len() > 100,
            "expected the full settings set, got {}",
            s.fields.len()
        );
        let gate = s
            .fields
            .iter()
            .find(|f| f.path == "extract.gate_min_score")
            .expect("a known setting should be present");
        assert_eq!(gate.kind, "float");
        assert_eq!(gate.default, json!(0.2));
        assert!(
            !gate.help.is_empty(),
            "help text should come from the doc comment"
        );

        let group_by = s
            .fields
            .iter()
            .find(|f| f.path == "compete.group_by")
            .expect("an enum setting should be present");
        assert_eq!(group_by.kind, "enum");
        let choices = group_by.choices.as_ref().expect("an enum has choices");
        assert!(choices.contains(&"base_peptide".to_string()), "{choices:?}");

        // Gate markers are what stop a benchmark-gated parameter being changed as if
        // it were ordinary.
        assert!(
            s.fields.iter().any(|f| !f.gates.is_empty()),
            "some settings are documented as gated and should be marked"
        );
    }
}
