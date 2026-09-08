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
//!
//! # Where a preset comes in
//!
//! The overrides are relative to the engine defaults, and the editor seeds them from
//! the preset selected on the Search screen (`overrides_of`), so what is saved is the
//! preset plus the edits and never silently less than the preset. The Search screen's
//! digest fields take the other route for the built-in library path: `derive` merges
//! them onto the preset file and writes a run configuration, because a person who typed
//! a missed-cleavage count expects it to reach the digest whichever predictor runs.

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

/// The inverse of [`nest`]: a configuration object as `{"extract.gate_min_score": 0.3}`.
///
/// Objects recurse into dotted paths; arrays and scalars are leaves, so a list-valued
/// setting such as `peptidoforms.fixed_mods` stays one entry. An empty object
/// contributes nothing, which is what "no override" means.
pub fn flatten(value: &serde_json::Value) -> BTreeMap<String, serde_json::Value> {
    fn walk(prefix: &str, v: &serde_json::Value, out: &mut BTreeMap<String, serde_json::Value>) {
        match v {
            serde_json::Value::Object(map) => {
                for (k, child) in map {
                    let path = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    walk(&path, child, out);
                }
            }
            other => {
                if !prefix.is_empty() {
                    out.insert(prefix.to_string(), other.clone());
                }
            }
        }
    }
    let mut out = BTreeMap::new();
    walk("", value, &mut out);
    out
}

/// Deep-merge `over` into `base`: objects merge key by key, everything else (arrays
/// included) is replaced. Replacing arrays is deliberate: the modification lists are
/// the whole list the user asked for, not additions to whatever the preset had.
pub fn merge(base: &mut serde_json::Value, over: &serde_json::Value) {
    match (base, over) {
        (serde_json::Value::Object(b), serde_json::Value::Object(o)) => {
            for (k, v) in o {
                match b.get_mut(k) {
                    Some(slot) if slot.is_object() && v.is_object() => merge(slot, v),
                    _ => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (b, o) => *b = o.clone(),
    }
}

/// The override set a configuration file amounts to, for seeding the editor.
pub fn overrides_of(path: &str) -> Result<BTreeMap<String, serde_json::Value>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read the preset {path}: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("the preset {path} is not valid JSON: {e}"))?;
    if !v.is_object() {
        return Err(format!("the preset {path} is not a JSON object"));
    }
    Ok(flatten(&v))
}

/// A filename a typed name can safely become, on every platform.
fn safe_name(name: &str) -> String {
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
    if safe.trim_matches('_').is_empty() {
        "settings".to_string()
    } else {
        safe
    }
}

/// Write an override set and hand back the path.
pub fn save(name: &str, flat: BTreeMap<String, serde_json::Value>) -> Result<String, String> {
    derive(name, None, flat)
}

/// Write `base` (a preset file, or nothing) with `flat` merged on top, and hand back
/// the path. This is how the Search screen's digest fields reach the engine on the
/// built-in library path: the run's configuration is the chosen preset plus the fields,
/// written under `config_dir` as `<name>.json`.
pub fn derive(
    name: &str,
    base: Option<&str>,
    flat: BTreeMap<String, serde_json::Value>,
) -> Result<String, String> {
    let dir = config_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let mut root = match base.filter(|b| !b.trim().is_empty()) {
        Some(b) => {
            let text = std::fs::read_to_string(b)
                .map_err(|e| format!("could not read the preset {b}: {e}"))?;
            let v: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| format!("the preset {b} is not valid JSON: {e}"))?;
            if !v.is_object() {
                return Err(format!("the preset {b} is not a JSON object"));
            }
            v
        }
        None => serde_json::Value::Object(serde_json::Map::new()),
    };
    merge(&mut root, &nest(&flat));
    let path = dir.join(format!("{}.json", safe_name(name)));
    let text = serde_json::to_string_pretty(&root)
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
    // The same environment every other engine invocation gets. This was the one
    // spawn site that omitted it, which breaks the invariant `stamp_env`'s own doc
    // states: validating a configuration against an engine that cannot see the
    // managed interpreters answers a different question from the one the run will
    // ask, and would report a perfectly good sidecar configuration as unusable.
    let mut cmd = crate::engine::command(&exe);
    crate::components::stamp_env(&mut cmd);
    let out = cmd
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
    fn flatten_inverts_nest_and_keeps_lists_whole() {
        let flat = flat(&[
            ("extract.gate_min_score", json!(0.3)),
            ("rescore.classifier", json!("nn_torch")),
            (
                "peptidoforms.fixed_mods",
                json!([{"residue": "C", "name": "Carbamidomethyl"}]),
            ),
            ("threads", json!(8)),
        ]);
        assert_eq!(flatten(&nest(&flat)), flat);
        // An empty object is "no override", not a key.
        assert!(flatten(&json!({"extract": {}})).is_empty());
    }

    #[test]
    fn merge_recurses_into_objects_and_replaces_lists() {
        let mut base = json!({
            "predict_frag": {"predictor": "ms2pip", "top_n_fragments": 12},
            "peptidoforms": {"fixed_mods": [{"residue": "C", "name": "Carbamidomethyl"}],
                             "charge_min": 2},
            "rescore": {"classifier": "nn_torch"}
        });
        merge(
            &mut base,
            &nest(&flat(&[
                ("peptidoforms.fixed_mods", json!([])),
                ("peptidoforms.charge_max", json!(3)),
                ("digest.missed_cleavages", json!(1)),
            ])),
        );
        // Untouched preset keys survive; the list is replaced, not appended to.
        assert_eq!(base["predict_frag"]["predictor"], json!("ms2pip"));
        assert_eq!(base["predict_frag"]["top_n_fragments"], json!(12));
        assert_eq!(base["rescore"]["classifier"], json!("nn_torch"));
        assert_eq!(base["peptidoforms"]["fixed_mods"], json!([]));
        assert_eq!(base["peptidoforms"]["charge_min"], json!(2));
        assert_eq!(base["peptidoforms"]["charge_max"], json!(3));
        assert_eq!(base["digest"]["missed_cleavages"], json!(1));
    }

    #[test]
    fn derive_writes_the_preset_plus_the_overrides() {
        // The built-in library path: the Search screen's digest fields on top of the
        // chosen preset, whose own choices (predictor, rescorer) must survive.
        let dir =
            std::env::temp_dir().join(format!("mumdia-console-derive-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let preset = dir.join("preset.json");
        std::fs::write(
            &preset,
            r#"{"predict_frag": {"predictor": "ms2pip", "rt_predictor": "deeplc"},
                "rescore": {"classifier": "nn_torch"},
                "digest": {"missed_cleavages": 2}}"#,
        )
        .unwrap();
        let out = derive(
            "derive-test",
            Some(preset.to_str().unwrap()),
            flat(&[
                ("digest.missed_cleavages", json!(1)),
                ("peptidoforms.variable_mods", json!([])),
            ]),
        )
        .expect("derive should succeed");
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(v["predict_frag"]["predictor"], json!("ms2pip"));
        assert_eq!(v["rescore"]["classifier"], json!("nn_torch"));
        assert_eq!(v["digest"]["missed_cleavages"], json!(1));
        assert_eq!(v["peptidoforms"]["variable_mods"], json!([]));
        assert_eq!(
            overrides_of(&out).unwrap()["digest.missed_cleavages"],
            json!(1),
            "a derived file reads back as the override set it is"
        );
        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_dir_all(&dir);

        // No base: the overrides alone, exactly what `save` always wrote.
        let out = derive("derive-test-nobase", None, flat(&[("threads", json!(4))])).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(v, json!({"threads": 4}));
        let _ = std::fs::remove_file(&out);
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
