//! Setups: a search setup kept under a name, or written to a file to move it to another
//! computer or send it to a colleague (#297).
//!
//! # What a setup holds
//!
//! The choices that decide a search, and nothing that belongs to one machine or one
//! dataset: the Search screen without its files, folders, conditions and thread counts,
//! and the Settings screen's engine settings as the preset they start from plus the
//! changes to it, without the paths of programs and interpreters, the cache locations and
//! the limits sized to this machine. The frontend builds the content from the screen and
//! applies it back (`captureSetup` and `applySetup` in `ui/app.js`), because that is where
//! the screen's state lives, and validates it there against the screen's own choices, the
//! modification catalogue and the settings schema. This module owns the file: its envelope,
//! the migration of an older format, and the named setups the application keeps.
//!
//! # The file
//!
//! ```json
//! {
//!   "format": "mumdia-desktop-settings",
//!   "version": 1,
//!   "name": "Phospho",
//!   "created_by": "MuMDIA 0.13.0",
//!   "created": "2026-11-12T09:30:00Z",
//!   "settings": { "engine": {...}, "digest": {...}, ... }
//! }
//! ```
//!
//! `version` is the version of this format, not of the application, and moves only when a
//! setting is renamed or reshaped. A file written in an older format is brought up to date
//! one version at a time by [`STEPS`], so a file saved today keeps loading. A file from a
//! newer format is read as far as this version understands it: the frontend lists every
//! setting it does not know as ignored, and says the file comes from a newer version.
//!
//! # Where named setups live
//!
//! `setups/` in the default folder ([`crate::storage::default_root`]), beside
//! `storage.json`, and not in the storage folder: that folder holds what can be built or
//! downloaded again, it can be changed on the Setup screen (nothing is copied), and its
//! `configs/` is offered for removal on the Managed data card. A setup is the user's own
//! work and can be rebuilt by nobody, so it stays where neither touches it. One file per
//! setup, in the format above, so a named setup can be copied out of that folder by hand
//! and loaded anywhere.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

/// The format identifier every settings file carries.
pub const FORMAT: &str = "mumdia-desktop-settings";

/// The format version this application writes, and the newest it reads without a warning.
pub const VERSION: u64 = 1;

/// The folder of named setups, inside the default folder.
pub const DIR_NAME: &str = "setups";

/// A settings file is a few kilobytes; anything near this is not one, and is not read.
pub const MAX_FILE_BYTES: u64 = 1_000_000;

/// The longest name a setup may have.
pub const NAME_MAX_CHARS: usize = 120;

/// The keys of the envelope; anything else at the top of a file is reported and ignored.
const ENVELOPE: &[&str] = &[
    "format",
    "version",
    "name",
    "created_by",
    "created",
    "settings",
];

/// One migration step: brings `settings` from one format version to the next, and says
/// what it changed in the notes the user is shown.
pub type Step = fn(&mut Value, &mut Vec<String>);

/// The migration from each older format: `STEPS[i]` brings a file from version `i + 1` to
/// version `i + 2`, so there are always `VERSION - 1` of them. A later format that renames
/// or reshapes a setting adds its step here, and a file written today keeps loading.
/// Version 1 is the first format, so there are none yet.
pub const STEPS: &[Step] = &[];

/// The named setups' folder (see the module comment for why it is not the storage folder).
pub fn dir() -> PathBuf {
    crate::storage::default_root().join(DIR_NAME)
}

/// What a file says wrote it: this application and its version, which is the release's.
pub fn created_by() -> String {
    format!("MuMDIA {}", env!("CARGO_PKG_VERSION"))
}

/// A settings file, read and brought up to the current format.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct Document {
    pub name: String,
    pub created_by: Option<String>,
    pub created: Option<String>,
    /// The format version the file was written in.
    pub version: u64,
    /// Written in a newer format than this application reads: what it does not know is
    /// ignored, and the frontend says so for every setting.
    pub newer: bool,
    /// The setup, migrated to [`VERSION`] when it was older. Validated by the frontend.
    pub settings: Value,
    /// What the reader noticed about the file itself (a newer format, a migration, keys
    /// at the top of the file it does not know), for the summary shown before applying.
    pub notes: Vec<String>,
}

/// Read a settings file's text. `fallback_name` names a file that carries no name of its
/// own. The error is the reason, for the caller to put after the file's name.
pub fn parse(text: &str, fallback_name: &str) -> Result<Document, String> {
    parse_with(text, fallback_name, STEPS)
}

fn parse_with(text: &str, fallback_name: &str, steps: &[Step]) -> Result<Document, String> {
    let current = steps.len() as u64 + 1;
    let value: Value =
        serde_json::from_str(text).map_err(|e| format!("it is not valid JSON ({e})"))?;
    let Value::Object(mut obj) = value else {
        return Err("it is not a JSON object".into());
    };
    match obj.get("format") {
        Some(Value::String(f)) if f == FORMAT => {}
        Some(other) => {
            return Err(format!(
                "its \"format\" is {other}, not \"{FORMAT}\", so it is not a MuMDIA settings file"
            ))
        }
        None => {
            // An engine configuration is the likeliest wrong file: it is JSON too.
            const SECTIONS: &[&str] = &["rescore", "extract", "quant", "predict_frag", "digest"];
            return Err(if SECTIONS.iter().any(|s| obj.contains_key(*s)) {
                format!(
                    "it has no \"format\": \"{FORMAT}\" and looks like an engine configuration \
                     file, which is not a setup. A setup is saved from the Settings screen."
                )
            } else {
                format!("it has no \"format\": \"{FORMAT}\", so it is not a MuMDIA settings file")
            });
        }
    }
    let version = match obj.get("version") {
        Some(Value::Number(n)) => n.as_u64().filter(|v| *v >= 1).ok_or_else(|| {
            format!("its \"version\" is {n}; a settings file's version is a whole number from 1")
        })?,
        Some(other) => {
            return Err(format!(
                "its \"version\" is {other}; a settings file's version is a whole number from 1"
            ))
        }
        None => return Err("it has no \"version\"".into()),
    };
    let name = match obj.get("name") {
        Some(Value::String(s)) if !s.trim().is_empty() => s.trim().to_string(),
        Some(Value::String(_)) | None => fallback_name.to_string(),
        Some(other) => return Err(format!("its \"name\" is {other}, not text")),
    };
    let mut notes = Vec::new();
    let text_field = |key: &str, notes: &mut Vec<String>| match obj.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        None | Some(Value::Null) => None,
        Some(other) => {
            notes.push(format!("\"{key}\" is {other}, not text, and is ignored."));
            None
        }
    };
    let created_by = text_field("created_by", &mut notes);
    let created = text_field("created", &mut notes);
    let mut settings = match obj.remove("settings") {
        Some(Value::Object(m)) => Value::Object(m),
        Some(other) => return Err(format!("its \"settings\" is {other}, not an object")),
        None => return Err("it has no \"settings\"".into()),
    };
    let newer = version > current;
    let unknown: Vec<&str> = obj
        .keys()
        .map(|k| k.as_str())
        .filter(|k| !ENVELOPE.contains(k))
        .collect();
    if newer {
        notes.push(format!(
            "Written by a newer version of MuMDIA (settings format {version}; this version \
             reads format {current} and older). What this version knows is loaded; what it \
             does not know is listed and ignored."
        ));
    }
    if !unknown.is_empty() {
        notes.push(format!(
            "Ignored at the top of the file, {}: {}.",
            if newer {
                "from the newer version"
            } else {
                "not known to this version"
            },
            unknown.join(", ")
        ));
    }
    if version < current {
        for step in &steps[(version - 1) as usize..] {
            step(&mut settings, &mut notes);
        }
        notes.push(format!(
            "Written in settings format {version}, and brought up to format {current}."
        ));
    }
    Ok(Document {
        name,
        created_by,
        created,
        version,
        newer,
        settings,
        notes,
    })
}

/// A setup's name, trimmed, or why it cannot be one.
fn checked_name(name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Give the setup a name.".into());
    }
    if name.chars().count() > NAME_MAX_CHARS {
        return Err(format!(
            "A setup's name has at most {NAME_MAX_CHARS} characters."
        ));
    }
    if name.chars().any(|c| c.is_control()) {
        return Err("A setup's name cannot contain line breaks or tabs.".into());
    }
    Ok(name.to_string())
}

/// The text of a settings file for `settings` under `name`, written now by this version.
pub fn render(name: &str, settings: &Value, created: &str) -> Result<String, String> {
    #[derive(Serialize)]
    struct Envelope<'a> {
        format: &'a str,
        version: u64,
        name: &'a str,
        created_by: String,
        created: &'a str,
        settings: &'a Value,
    }
    if !settings.is_object() {
        return Err("the settings of a setup are an object".into());
    }
    let name = checked_name(name)?;
    let text = serde_json::to_string_pretty(&Envelope {
        format: FORMAT,
        version: VERSION,
        name: &name,
        created_by: created_by(),
        created,
        settings,
    })
    .map_err(|e| format!("could not write the settings: {e}"))?;
    Ok(text + "\n")
}

/// The current time as an RFC 3339 UTC timestamp, to the second.
pub fn now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    utc_timestamp(secs)
}

fn utc_timestamp(secs: u64) -> String {
    // Days to a civil date, after Howard Hinnant's `civil_from_days`.
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let rem = secs % 86_400;
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3_600,
        rem % 3_600 / 60,
        rem % 60
    )
}

/// Device names Windows reserves in every folder, which a file cannot be called.
const RESERVED: &[&str] = &[
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// The file a named setup is kept in, from its name: lower case, so two names that differ
/// only in case are the same setup on every platform (Windows and macOS folders do not
/// tell them apart), and only letters, digits, `-` and `_`.
pub fn id_of(name: &str) -> String {
    let mut id: String = name
        .trim()
        .chars()
        .flat_map(|c| c.to_lowercase())
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    if id.trim_matches('_').is_empty() {
        id = "setup".into();
    }
    if RESERVED.contains(&id.as_str()) {
        id = format!("setup_{id}");
    }
    id
}

/// An id handed in by the interface, refused unless it is one [`id_of`] could produce, so
/// it names a file inside the setups folder and nothing else.
fn checked_id(id: &str) -> Result<&str, String> {
    let ok = !id.is_empty()
        && id.chars().count() <= 70
        && id
            .chars()
            .all(|c| (c.is_alphanumeric() && !c.is_uppercase()) || c == '-' || c == '_')
        && !RESERVED.contains(&id);
    if ok {
        Ok(id)
    } else {
        Err(format!("\"{id}\" is not the name of a saved setup"))
    }
}

/// A named setup, as the list shows it.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct Item {
    pub id: String,
    pub name: String,
    pub created: Option<String>,
    pub created_by: Option<String>,
    /// The preset its engine settings start from, for the list; `None` for the engine
    /// defaults or when it does not say.
    pub preset: Option<String>,
    /// Why the file could not be read, when it could not: it is listed rather than left
    /// out, so it can be deleted.
    pub error: Option<String>,
}

impl Item {
    fn of(id: &str, doc: &Document) -> Item {
        Item {
            id: id.to_string(),
            name: doc.name.clone(),
            created: doc.created.clone(),
            created_by: doc.created_by.clone(),
            preset: doc
                .settings
                .pointer("/engine/preset")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            error: None,
        }
    }
}

/// What saving a named setup did: the setup saved, or the name of the one already saved
/// under that name, which is replaced only when the user says so.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct SaveOutcome {
    pub item: Option<Item>,
    pub existing: Option<String>,
}

fn read_text(path: &Path) -> Result<String, String> {
    let meta =
        std::fs::metadata(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("{} is not a file", path.display()));
    }
    if meta.len() > MAX_FILE_BYTES {
        return Err(format!(
            "{} is {:.1} MB; a settings file is a few kilobytes, so this is not one",
            path.display(),
            meta.len() as f64 / 1e6
        ));
    }
    std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

/// Write `text` to `path` through a temporary file renamed into place, so a crash never
/// leaves half a setup.
fn write_atomic(path: &Path, text: &str) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "settings.json".into());
    let tmp = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    std::fs::write(&tmp, text).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot write {}: {e}", path.display())
    })
}

/// A path the user chose for a settings file, with `.json` added unless it ends in it.
pub fn json_path(path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    let is_json = p
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("json"));
    if is_json {
        p
    } else {
        let mut s = p.into_os_string();
        s.push(".json");
        PathBuf::from(s)
    }
}

/// Read a settings file the user chose.
pub fn read_file(path: &str) -> Result<Document, String> {
    let p = Path::new(path);
    let text = read_text(p)?;
    let stem = p
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("settings")
        .to_string();
    parse(&text, &stem).map_err(|e| format!("{} cannot be loaded: {e}", p.display()))
}

/// Write `settings` as a settings file named `name` at the path the user chose (with
/// `.json` added when it lacks it), and return where it went.
pub fn write_file(path: &str, name: &str, settings: &Value) -> Result<String, String> {
    if path.trim().is_empty() {
        return Err("Choose where to save the settings file.".into());
    }
    let dest = json_path(path);
    write_atomic(&dest, &render(name, settings, &now())?)?;
    Ok(dest.display().to_string())
}

/// The named setups in `dir`, by name. A file that cannot be read is listed with the
/// reason, so it can be deleted rather than lingering unseen.
pub fn list_in(dir: &Path) -> Vec<Item> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut items: Vec<Item> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "json"))
        .filter_map(|p| {
            let id = p.file_stem()?.to_str()?.to_string();
            // A file that is not one of ours (an uppercase name, a dot) is not listed: it
            // could not be loaded or deleted by its id.
            checked_id(&id).ok()?;
            Some(match read_text(&p).and_then(|t| parse(&t, &id)) {
                Ok(doc) => Item::of(&id, &doc),
                Err(e) => Item {
                    id: id.clone(),
                    name: id,
                    created: None,
                    created_by: None,
                    preset: None,
                    error: Some(e),
                },
            })
        })
        .collect();
    items.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.id.cmp(&b.id))
    });
    items
}

/// Keep `settings` as the named setup `name`. A setup already saved under that name is
/// replaced only with `overwrite`; otherwise its name comes back in `existing`.
pub fn save_in(
    dir: &Path,
    name: &str,
    settings: &Value,
    overwrite: bool,
) -> Result<SaveOutcome, String> {
    let name = checked_name(name)?;
    let id = id_of(&name);
    let path = dir.join(format!("{id}.json"));
    if path.exists() && !overwrite {
        let existing = read_text(&path)
            .and_then(|t| parse(&t, &id))
            .map(|d| d.name)
            .unwrap_or_else(|_| id.clone());
        return Ok(SaveOutcome {
            item: None,
            existing: Some(existing),
        });
    }
    let text = render(&name, settings, &now())?;
    write_atomic(&path, &text)?;
    let doc = parse(&text, &id)?;
    Ok(SaveOutcome {
        item: Some(Item::of(&id, &doc)),
        existing: None,
    })
}

/// Read the named setup `id`.
pub fn read_in(dir: &Path, id: &str) -> Result<Document, String> {
    let id = checked_id(id)?;
    let text = read_text(&dir.join(format!("{id}.json")))?;
    parse(&text, id).map_err(|e| format!("The saved setup \"{id}\" cannot be loaded: {e}"))
}

/// Give the named setup `id` another name. Everything else in its file is kept as it is,
/// including a format newer than this version's; the file follows the name.
pub fn rename_in(dir: &Path, id: &str, name: &str) -> Result<Item, String> {
    let id = checked_id(id)?;
    let name = checked_name(name)?;
    let from = dir.join(format!("{id}.json"));
    let text = read_text(&from)?;
    let doc = parse(&text, id)?;
    let mut raw: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    raw["name"] = Value::String(name.clone());
    let new_id = id_of(&name);
    let to = dir.join(format!("{new_id}.json"));
    if new_id != id && to.exists() {
        let other = read_text(&to)
            .and_then(|t| parse(&t, &new_id))
            .map(|d| d.name)
            .unwrap_or_else(|_| new_id.clone());
        return Err(format!(
            "A setup named \"{other}\" is already saved; choose another name, or delete that one first."
        ));
    }
    let out = serde_json::to_string_pretty(&raw).map_err(|e| e.to_string())? + "\n";
    write_atomic(&to, &out)?;
    if new_id != id {
        std::fs::remove_file(&from)
            .map_err(|e| format!("cannot remove {}: {e}", from.display()))?;
    }
    Ok(Item::of(&new_id, &Document { name, ..doc }))
}

/// Delete the named setup `id`.
pub fn delete_in(dir: &Path, id: &str) -> Result<(), String> {
    let id = checked_id(id)?;
    let path = dir.join(format!("{id}.json"));
    std::fs::remove_file(&path).map_err(|e| format!("cannot delete {}: {e}", path.display()))
}

/// Write the named setup `id` to a file the user chose, exactly as it is kept, and return
/// where it went.
pub fn export_in(dir: &Path, id: &str, path: &str) -> Result<String, String> {
    let id = checked_id(id)?;
    if path.trim().is_empty() {
        return Err("Choose where to save the settings file.".into());
    }
    let text = read_text(&dir.join(format!("{id}.json")))?;
    // Refused rather than copied when it cannot be read: a broken file is not worth sending.
    parse(&text, id)
        .map_err(|e| format!("The saved setup \"{id}\" cannot be saved to a file: {e}"))?;
    let dest = json_path(path);
    write_atomic(&dest, &text)?;
    Ok(dest.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tempdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "mumdia-console-setups-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn sample() -> Value {
        json!({
            "engine": {"preset": "diann-library", "changes": {"rescore": {"seeds": 3}}, "reset": []},
            "digest": {"enzyme": "trypsin_p", "missed_cleavages": 1, "min_length": 7},
            "modifications": {
                "fixed": [{"residue": "C", "name": "Carbamidomethyl"}],
                "variable": [{"residue": "*", "name": "Acetyl", "position": "protein_n_term"}],
                "max_variable": 2
            },
            "ztscan": {"analysis": true, "q1_half_width": null}
        })
    }

    #[test]
    fn a_written_file_reads_back_as_the_same_settings() {
        let text = render("Phospho", &sample(), "2026-11-12T09:30:00Z").unwrap();
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["format"], json!(FORMAT));
        assert_eq!(v["version"], json!(VERSION));
        assert_eq!(v["name"], json!("Phospho"));
        assert_eq!(v["created_by"], json!(created_by()));
        assert!(created_by().starts_with("MuMDIA "));
        let doc = parse(&text, "x").unwrap();
        assert_eq!(
            doc.settings,
            sample(),
            "the settings survive the file unchanged"
        );
        assert_eq!(doc.name, "Phospho");
        assert_eq!(doc.version, VERSION);
        assert!(!doc.newer);
        assert!(doc.notes.is_empty(), "{:?}", doc.notes);
        // The envelope reads in the order a person expects, whatever the map type.
        let at = |k: &str| text.find(&format!("\"{k}\"")).unwrap();
        assert!(at("format") < at("version") && at("version") < at("name"));
        assert!(at("name") < at("created_by") && at("created") < at("settings"));
    }

    #[test]
    fn a_file_that_is_not_a_settings_file_is_refused_with_the_reason() {
        let cases: &[(&str, &str)] = &[
            ("not json", "not valid JSON"),
            ("[1, 2]", "not a JSON object"),
            (
                r#"{"rescore": {"seeds": 3}}"#,
                "looks like an engine configuration",
            ),
            (r#"{"settings": {}}"#, "has no \"format\""),
            (
                r#"{"format": "something-else", "version": 1, "settings": {}}"#,
                "\"something-else\"",
            ),
            (
                r#"{"format": "mumdia-desktop-settings", "settings": {}}"#,
                "no \"version\"",
            ),
            (
                r#"{"format": "mumdia-desktop-settings", "version": 0, "settings": {}}"#,
                "whole number from 1",
            ),
            (
                r#"{"format": "mumdia-desktop-settings", "version": "1", "settings": {}}"#,
                "whole number from 1",
            ),
            (
                r#"{"format": "mumdia-desktop-settings", "version": 1.5, "settings": {}}"#,
                "whole number from 1",
            ),
            (
                r#"{"format": "mumdia-desktop-settings", "version": 1}"#,
                "no \"settings\"",
            ),
            (
                r#"{"format": "mumdia-desktop-settings", "version": 1, "settings": [1]}"#,
                "not an object",
            ),
            (
                r#"{"format": "mumdia-desktop-settings", "version": 1, "name": 3, "settings": {}}"#,
                "\"name\" is 3",
            ),
        ];
        for (text, why) in cases {
            let e = parse(text, "f").expect_err(text);
            assert!(e.contains(why), "{text}: {e}");
        }
    }

    #[test]
    fn a_file_from_a_newer_version_is_read_with_a_warning() {
        let text = json!({
            "format": FORMAT, "version": VERSION + 1, "name": "Later",
            "settings": {"digest": {"missed_cleavages": 2}, "new_card": {"x": 1}},
            "signature": "abc"
        })
        .to_string();
        let doc = parse(&text, "f").unwrap();
        assert!(doc.newer);
        assert_eq!(doc.version, VERSION + 1);
        // Read as far as this version understands it; the frontend names the unknown keys.
        assert_eq!(doc.settings["new_card"], json!({"x": 1}));
        assert!(
            doc.notes.iter().any(|n| n.contains("newer version")),
            "{:?}",
            doc.notes
        );
        assert!(
            doc.notes
                .iter()
                .any(|n| n.contains("signature") && n.contains("newer")),
            "{:?}",
            doc.notes
        );
    }

    #[test]
    fn an_older_format_is_migrated_one_step_at_a_time() {
        // There is no older format yet, so the hook is shown working with two made-up steps:
        // format 1 named the prescreen "screen", format 2 kept bands as a bare number.
        fn one_to_two(s: &mut Value, notes: &mut Vec<String>) {
            if let Some(v) = s.as_object_mut().and_then(|o| o.remove("screen")) {
                s["prescreen"] = v;
                notes.push("\"screen\" is now \"prescreen\".".into());
            }
        }
        fn two_to_three(s: &mut Value, _: &mut Vec<String>) {
            if let Some(n) = s.get("bands").and_then(|b| b.as_u64()) {
                s["bands"] = json!({"mode": "custom", "count": n});
            }
        }
        let steps: &[Step] = &[one_to_two, two_to_three];
        let from_one = json!({"format": FORMAT, "version": 1, "settings": {"screen": {"mode": "off"}, "bands": 4}});
        let doc = parse_with(&from_one.to_string(), "old", steps).unwrap();
        assert_eq!(
            doc.settings,
            json!({"prescreen": {"mode": "off"}, "bands": {"mode": "custom", "count": 4}})
        );
        assert!(doc.notes.iter().any(|n| n.contains("now \"prescreen\"")));
        assert!(doc
            .notes
            .iter()
            .any(|n| n.contains("format 1, and brought up to format 3")));
        assert_eq!(doc.name, "old", "a file without a name takes the file's");
        let from_two = json!({"format": FORMAT, "version": 2, "settings": {"bands": 2}});
        let doc = parse_with(&from_two.to_string(), "f", steps).unwrap();
        assert_eq!(
            doc.settings,
            json!({"bands": {"mode": "custom", "count": 2}})
        );
        // The shipped table has one step per older version.
        assert_eq!(STEPS.len() as u64, VERSION - 1);
    }

    #[test]
    fn timestamps_are_rfc3339_utc() {
        assert_eq!(utc_timestamp(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_timestamp(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(utc_timestamp(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(utc_timestamp(4_107_542_399), "2100-02-28T23:59:59Z");
        assert_eq!(now().len(), "2026-11-12T09:30:00Z".len());
    }

    #[test]
    fn names_become_safe_file_names_that_ignore_case() {
        assert_eq!(id_of("Phospho"), "phospho");
        assert_eq!(id_of("phospho"), id_of("PHOSPHO"));
        assert_eq!(id_of("Label-free / HYE"), "label-free___hye");
        assert_eq!(id_of("../../evil"), "______evil");
        assert_eq!(id_of("..."), "setup");
        assert_eq!(id_of("CON"), "setup_con");
        assert_eq!(id_of("Phospho écrémé"), "phospho_écrémé");
        assert_eq!(id_of(&"x".repeat(200)).chars().count(), 64);
        for name in ["../../evil", "a/b", "c:\\d", "..", "", "nul", "Ünïcode"] {
            assert!(checked_id(&id_of(name)).is_ok(), "{name}");
        }
        for bad in ["../x", "a/b", "A", "a.b", "", "con", "a\\b"] {
            assert!(checked_id(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn named_setups_are_saved_listed_renamed_exported_and_deleted() {
        let dir = tempdir("store");
        assert!(list_in(&dir).is_empty());
        let saved = save_in(&dir, "  Phospho  ", &sample(), false).unwrap();
        let item = saved.item.unwrap();
        assert_eq!(
            (item.id.as_str(), item.name.as_str()),
            ("phospho", "Phospho")
        );
        assert_eq!(item.preset.as_deref(), Some("diann-library"));
        assert_eq!(item.created_by.as_deref(), Some(created_by().as_str()));
        save_in(&dir, "label-free", &json!({}), false).unwrap();

        // The same name in another case is the same setup, replaced only when asked.
        let again = save_in(&dir, "PHOSPHO", &json!({"digest": {}}), false).unwrap();
        assert_eq!(
            (again.item, again.existing.as_deref()),
            (None, Some("Phospho"))
        );
        assert_eq!(read_in(&dir, "phospho").unwrap().settings, sample());
        let replaced = save_in(&dir, "PHOSPHO", &json!({"digest": {}}), true).unwrap();
        assert_eq!(replaced.item.unwrap().name, "PHOSPHO");
        assert_eq!(
            read_in(&dir, "phospho").unwrap().settings,
            json!({"digest": {}})
        );

        let names: Vec<String> = list_in(&dir).into_iter().map(|i| i.name).collect();
        assert_eq!(names, ["label-free", "PHOSPHO"]);

        // Renaming moves the file and keeps the settings; a taken name is refused.
        let r = rename_in(&dir, "phospho", "Phospho STY").unwrap();
        assert_eq!(
            (r.id.as_str(), r.name.as_str()),
            ("phospho_sty", "Phospho STY")
        );
        assert!(!dir.join("phospho.json").exists());
        assert_eq!(
            read_in(&dir, "phospho_sty").unwrap().settings,
            json!({"digest": {}})
        );
        let e = rename_in(&dir, "phospho_sty", "Label-Free").unwrap_err();
        assert!(e.contains("\"label-free\" is already saved"), "{e}");
        // A change of case only keeps the file.
        let r = rename_in(&dir, "phospho_sty", "phospho sty").unwrap();
        assert_eq!(r.id, "phospho_sty");

        // Exported exactly as kept, with .json added to a name without it.
        let out = export_in(
            &dir,
            "phospho_sty",
            dir.join("out/shared").to_str().unwrap(),
        )
        .unwrap();
        assert!(out.ends_with("shared.json"), "{out}");
        assert_eq!(
            std::fs::read_to_string(&out).unwrap(),
            std::fs::read_to_string(dir.join("phospho_sty.json")).unwrap()
        );
        let doc = read_file(&out).unwrap();
        assert_eq!(doc.name, "phospho sty");

        delete_in(&dir, "phospho_sty").unwrap();
        assert!(read_in(&dir, "phospho_sty").is_err());
        assert!(delete_in(&dir, "../escape").is_err());
        assert!(read_in(&dir, "..").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_broken_named_setup_is_listed_with_its_reason() {
        let dir = tempdir("broken");
        std::fs::write(dir.join("broken.json"), "{not json").unwrap();
        // Not ours (an id `id_of` cannot produce): not listed.
        std::fs::write(dir.join("Other.json"), "{}").unwrap();
        let items = list_in(&dir);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, "broken");
        assert!(items[0]
            .error
            .as_deref()
            .unwrap()
            .contains("not valid JSON"));
        assert!(export_in(&dir, "broken", dir.join("x.json").to_str().unwrap()).is_err());
        delete_in(&dir, "broken").unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_is_written_where_the_user_chose_and_read_back() {
        let dir = tempdir("file");
        let path = dir.join("my setup.JSON");
        let out = write_file(path.to_str().unwrap(), "Label-free", &sample()).unwrap();
        assert_eq!(
            PathBuf::from(&out),
            path,
            "an existing .json extension is kept"
        );
        let doc = read_file(&out).unwrap();
        assert_eq!(
            (doc.name.as_str(), &doc.settings),
            ("Label-free", &sample())
        );
        let txt = write_file(dir.join("notes.txt").to_str().unwrap(), "A", &json!({})).unwrap();
        assert!(txt.ends_with("notes.txt.json"), "{txt}");
        assert!(write_file("", "A", &json!({})).is_err());
        assert!(write_file(path.to_str().unwrap(), "  ", &json!({})).is_err());
        assert!(write_file(path.to_str().unwrap(), "a\nb", &json!({})).is_err());
        assert!(write_file(path.to_str().unwrap(), &"n".repeat(121), &json!({})).is_err());
        assert!(write_file(path.to_str().unwrap(), "A", &json!([])).is_err());
        // Too large to be a settings file: not read at all.
        let big = dir.join("big.json");
        std::fs::write(&big, " ".repeat(MAX_FILE_BYTES as usize + 1)).unwrap();
        assert!(read_file(big.to_str().unwrap())
            .unwrap_err()
            .contains("not one"));
        let e = read_file(dir.join("absent.json").to_str().unwrap()).unwrap_err();
        assert!(e.contains("cannot read"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn named_setups_are_kept_outside_the_storage_folder() {
        // The storage folder can be changed (nothing is copied) and its `configs/` is offered
        // for removal; a setup must survive both.
        assert_eq!(dir(), crate::storage::default_root().join(DIR_NAME));
        assert_ne!(DIR_NAME, crate::settings::CONFIG_DIR_NAME);
    }
}
