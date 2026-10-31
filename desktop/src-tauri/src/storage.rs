//! The storage folder: one setting for where the application and the engine write.
//!
//! # Why one setting
//!
//! Everything this application writes on its own lived in fixed per-user places on the
//! system drive: the managed Python environments, the downloaded converters, the
//! predicted DIA-NN libraries and the saved settings under `%LOCALAPPDATA%\MuMDIA`
//! (`~/.local/share/MuMDIA`), the engine's caches beside them, temporary files in the
//! system's temporary folder, the predictors' model downloads in the home folder, and
//! `uv`'s package cache in its own. On an account whose profile is small, roams, or may
//! not be written (a managed workstation), a search failed in whichever of these came
//! first, and nothing on the screen could move it.
//!
//! The Storage folder setting on the Setup screen moves all of the bulky and
//! re-creatable data at once: the predicted libraries and saved settings (`libraries/`,
//! `configs/`), the engine's caches (`cache/`, exported as `MUMDIA_CACHE_DIR`), temporary
//! files (`temp/`, `MUMDIA_TEMP_DIR`), model downloads (`models/`, `MUMDIA_MODEL_DIR`) and
//! `uv`'s package cache (`uv-cache/`, `UV_CACHE_DIR`).
//!
//! The Python environments and the downloaded tools (ThermoRawFileParser, DIA-NN) stay in
//! the default folder unless the user also asks to move them, because a moved environment
//! has to be installed again (an environment cannot be relocated: its interpreter paths are
//! absolute) and that is a few hundred megabytes of download nobody should get by surprise.
//! With that option `uv`'s interpreters move as well (`uv-python/`, `UV_PYTHON_INSTALL_DIR`).
//!
//! # Where the setting itself lives
//!
//! In `storage.json` in the default folder, which is the one place the application can
//! find before it knows the setting. `MUMDIA_DATA_DIR` replaces that default folder, for an
//! account whose profile cannot be written at all (set it before starting the
//! application). The setting is never a configuration value, so a configuration's hash and
//! a search's output do not depend on it.
//!
//! An exported variable never overrides one the user set before starting the application:
//! a `MUMDIA_CACHE_DIR` in the environment is a deliberate choice.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The file in the default folder that records the storage folder.
pub const BOOTSTRAP_NAME: &str = "storage.json";

/// Sub-folders of the storage folder for what the application exports to the engine.
pub const CACHE_SUBDIR: &str = "cache";
pub const TEMP_SUBDIR: &str = "temp";
pub const MODELS_SUBDIR: &str = "models";
pub const UV_CACHE_SUBDIR: &str = "uv-cache";
pub const UV_PYTHON_SUBDIR: &str = "uv-python";

/// Free space below which the Setup screen and the preflight warn about the storage
/// folder: the engine's caches alone may grow to 100 GiB, a predicted whole-proteome
/// library is gigabytes, and a search's temporary files are written there too.
pub const LOW_SPACE_BYTES: u64 = 20_000_000_000;

/// What `storage.json` holds.
#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq, Eq)]
pub struct Bootstrap {
    /// The storage folder, when one was chosen; the default folder otherwise.
    #[serde(default)]
    pub folder: Option<PathBuf>,
    /// Whether the Python environments and the downloaded tools live in the storage folder
    /// too, rather than in the default folder.
    #[serde(default)]
    pub move_environments: bool,
}

/// The folders the application uses, resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Where `storage.json` is, and everything lives when no storage folder is chosen.
    pub default_root: PathBuf,
    /// Libraries, saved settings, caches, temporary files, model downloads.
    pub storage: PathBuf,
    /// Python environments, ThermoRawFileParser, DIA-NN.
    pub tools: PathBuf,
    /// Whether a storage folder other than the default was chosen.
    pub configured: bool,
    pub move_environments: bool,
}

/// The default folder, from a variable lookup: `MUMDIA_DATA_DIR` when it is set to an
/// absolute path, else `%LOCALAPPDATA%\MuMDIA` on Windows and `$XDG_DATA_HOME/MuMDIA` or
/// `~/.local/share/MuMDIA` elsewhere, else the temporary folder.
fn default_root_with(env: &dyn Fn(&str) -> Option<std::ffi::OsString>, windows: bool) -> PathBuf {
    if let Some(d) = env("MUMDIA_DATA_DIR").map(PathBuf::from) {
        if d.is_absolute() {
            return d;
        }
    }
    let base = if windows {
        env("LOCALAPPDATA").map(PathBuf::from)
    } else {
        env("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| env("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    };
    base.unwrap_or_else(std::env::temp_dir).join("MuMDIA")
}

/// The default folder of this process (see [`default_root_with`]).
pub fn default_root() -> PathBuf {
    default_root_with(&|k| std::env::var_os(k), cfg!(windows))
}

/// `storage.json` in `root`; a missing or unreadable file is the default setting.
fn load_from(root: &Path) -> Bootstrap {
    std::fs::read_to_string(root.join(BOOTSTRAP_NAME))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Write `storage.json` in `root` (a temporary file renamed into place, so a crash never
/// leaves half a setting), or remove it when `b` is the default.
fn save_to(root: &Path, b: &Bootstrap) -> Result<(), String> {
    let path = root.join(BOOTSTRAP_NAME);
    if *b == Bootstrap::default() {
        return match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("cannot remove {}: {e}", path.display())),
        };
    }
    std::fs::create_dir_all(root).map_err(|e| format!("cannot create {}: {e}", root.display()))?;
    let text = serde_json::to_string_pretty(b).map_err(|e| e.to_string())?;
    let tmp = root.join(format!("{BOOTSTRAP_NAME}.tmp-{}", std::process::id()));
    std::fs::write(&tmp, text).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot write {}: {e}", path.display())
    })
}

fn layout_of(default_root: PathBuf, b: &Bootstrap) -> Layout {
    let storage = b
        .folder
        .clone()
        .filter(|f| f.is_absolute())
        .unwrap_or_else(|| default_root.clone());
    let configured = storage != default_root;
    let move_environments = configured && b.move_environments;
    Layout {
        tools: if move_environments {
            storage.clone()
        } else {
            default_root.clone()
        },
        storage,
        default_root,
        configured,
        move_environments,
    }
}

/// The folders of this process, read from `storage.json` on every call so a change on the
/// Setup screen applies to the next thing started.
pub fn layout() -> Layout {
    let root = default_root();
    let b = load_from(&root);
    layout_of(root, &b)
}

/// Libraries, saved settings, caches, temporary files and model downloads.
pub fn storage_dir() -> PathBuf {
    layout().storage
}

/// The Python environments and the downloaded tools.
pub fn tools_dir() -> PathBuf {
    layout().tools
}

/// Where the application's own temporary files go: `temp/` in a chosen storage folder,
/// else the system's temporary folder.
pub fn temp_dir() -> PathBuf {
    let l = layout();
    if l.configured {
        l.storage.join(TEMP_SUBDIR)
    } else {
        std::env::temp_dir()
    }
}

/// The variables a child process gets for `l`, leaving out every one `present` says the
/// application's own environment already has. Nothing when no storage folder is chosen:
/// the engine's and the packages' defaults then hold, as before the setting existed.
fn child_env_of(l: &Layout, present: &dyn Fn(&str) -> bool) -> Vec<(&'static str, PathBuf)> {
    if !l.configured {
        return Vec::new();
    }
    let mut out = vec![
        ("MUMDIA_CACHE_DIR", l.storage.join(CACHE_SUBDIR)),
        ("MUMDIA_TEMP_DIR", l.storage.join(TEMP_SUBDIR)),
        ("MUMDIA_MODEL_DIR", l.storage.join(MODELS_SUBDIR)),
        ("UV_CACHE_DIR", l.storage.join(UV_CACHE_SUBDIR)),
    ];
    if l.move_environments {
        out.push(("UV_PYTHON_INSTALL_DIR", l.storage.join(UV_PYTHON_SUBDIR)));
    }
    out.into_iter().filter(|(k, _)| !present(k)).collect()
}

/// Give `cmd` the storage locations: the engine (through `components::stamp_env`) and
/// `uv` (through `components::install`).
pub fn apply(cmd: &mut std::process::Command) {
    let l = layout();
    for (k, v) in child_env_of(&l, &|k| std::env::var_os(k).is_some_and(|v| !v.is_empty())) {
        cmd.env(k, v);
    }
}

/// Create `dir` if needed and check that a file can be written in it and renamed there,
/// which is what the engine's caches and the installers do. The probe file is removed.
pub fn probe_writable(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    if !dir.is_dir() {
        return Err(format!("{} is not a folder", dir.display()));
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stem = format!(".mumdia-write-probe-{}-{nanos}", std::process::id());
    let a = dir.join(format!("{stem}.tmp"));
    let b = dir.join(format!("{stem}.done"));
    std::fs::write(&a, b"mumdia").map_err(|e| {
        let _ = std::fs::remove_file(&a);
        format!("cannot write a file in {}: {e}", dir.display())
    })?;
    if let Err(e) = std::fs::rename(&a, &b) {
        let _ = std::fs::remove_file(&a);
        return Err(format!(
            "can write in {} but cannot rename a file there: {e}",
            dir.display()
        ));
    }
    let _ = std::fs::remove_file(&b);
    Ok(())
}

/// Whether `dir`, which may not exist yet, could be written, asked without creating it:
/// probed in the folder itself, or in its nearest existing ancestor.
pub fn writable_without_creating(dir: &Path) -> Result<(), String> {
    let mut base = dir.to_path_buf();
    while !base.exists() {
        match base.parent() {
            Some(p) if !p.as_os_str().is_empty() => base = p.to_path_buf(),
            _ => return Err(format!("no part of {} exists", dir.display())),
        }
    }
    if !base.is_dir() {
        return Err(format!("{} is a file, not a folder", base.display()));
    }
    // The probe file is written and removed in an existing folder, so nothing is created.
    probe_writable(&base)
}

/// Whether `path` is a network share (`\\server\share`, `//server/share`).
pub fn is_network_path(path: &Path) -> bool {
    let s = path.to_string_lossy();
    if s.starts_with(r"\\?\UNC\") {
        return true;
    }
    if s.starts_with(r"\\?\") || s.starts_with(r"\\.\") {
        return false;
    }
    s.starts_with(r"\\") || s.starts_with("//")
}

/// Whether `path` is in a OneDrive folder: under a folder in `roots` (the OneDrive
/// client's `OneDrive`, `OneDriveCommercial`, `OneDriveConsumer`), or below a folder whose
/// name starts with `OneDrive`.
fn in_onedrive(path: &Path, roots: &[PathBuf]) -> bool {
    let norm = |p: &Path| {
        p.to_string_lossy()
            .replace('\\', "/")
            .trim_end_matches('/')
            .to_lowercase()
    };
    let p = norm(path);
    let under = |r: &PathBuf| {
        let r = norm(r);
        !r.is_empty() && (p == r || p.starts_with(&format!("{r}/")))
    };
    roots.iter().any(under)
        || path.components().any(|c| match c {
            std::path::Component::Normal(n) => {
                n.to_string_lossy().to_lowercase().starts_with("onedrive")
            }
            _ => false,
        })
}

fn onedrive_roots() -> Vec<PathBuf> {
    ["OneDrive", "OneDriveCommercial", "OneDriveConsumer"]
        .iter()
        .filter_map(std::env::var_os)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .collect()
}

/// Warnings about where a folder is (`what` names it in the sentence): a OneDrive folder
/// or a network share. Neither is refused; both are slower and can fail where a local
/// folder does not.
fn location_warnings_with(what: &str, path: &Path, onedrive: &[PathBuf]) -> Vec<String> {
    let mut out = Vec::new();
    if is_network_path(path) {
        out.push(format!(
            "The {what} {} is on a network share. Searches write and rename many large files \
             there, which is slow over a network and can fail; a folder on a local disk is \
             faster and safer.",
            path.display()
        ));
    }
    if in_onedrive(path, onedrive) {
        out.push(format!(
            "The {what} {} is inside OneDrive. OneDrive can lock files while a search writes \
             and renames them, and uploads every intermediate file; a local folder outside \
             OneDrive is faster and safer.",
            path.display()
        ));
    }
    out
}

/// [`location_warnings_with`] for this account's OneDrive folders.
pub fn location_warnings(what: &str, path: &Path) -> Vec<String> {
    location_warnings_with(what, path, &onedrive_roots())
}

/// The file that marks a folder as a MuMDIA storage folder.
pub const MARKER_NAME: &str = ".mumdia-storage";

/// The folder MuMDIA uses for a folder the user picked: the folder itself when it is new,
/// empty or already a MuMDIA storage folder ([`MARKER_NAME`]), else a `MuMDIA` sub-folder
/// of it. The Managed data card removes `libraries/` and `configs/` of the storage folder,
/// so a picked folder that already holds the user's own files (Documents, a data drive)
/// must not have its sub-folders of those names mixed with, or removed with, MuMDIA's.
fn resolve_choice(p: &Path) -> PathBuf {
    let ours = !p.exists()
        || p.join(MARKER_NAME).is_file()
        || std::fs::read_dir(p).is_ok_and(|mut d| d.next().is_none());
    if ours {
        p.to_path_buf()
    } else {
        p.join("MuMDIA")
    }
}

/// A folder the user chose, checked: absolute, a folder (or creatable as one), and
/// writable with a rename. Returns the folder that will be used ([`resolve_choice`]).
pub fn validate(folder: &str) -> Result<PathBuf, String> {
    let f = folder.trim();
    if f.is_empty() {
        return Err("no folder was given".into());
    }
    let p = PathBuf::from(f);
    if !p.is_absolute() {
        return Err(format!(
            "{f} is not a full path; choose the folder with the folder picker"
        ));
    }
    if p.is_file() {
        return Err(format!("{f} is a file, not a folder"));
    }
    let target = resolve_choice(&p);
    probe_writable(&target).map_err(|e| {
        format!(
            "MuMDIA cannot use {} as its storage folder: {e}",
            target.display()
        )
    })?;
    Ok(target)
}

/// Choose the storage folder (`None`: back to the default) and whether the Python
/// environments move with it. Saved in `storage.json` in the default folder.
pub fn set(folder: Option<&str>, move_environments: bool) -> Result<Layout, String> {
    let root = default_root();
    let chosen = match folder {
        Some(f) => Some(validate(f)?),
        None => None,
    };
    let b = match chosen {
        // Choosing the default folder is the same as resetting.
        Some(p) if p != root => {
            let marker = p.join(MARKER_NAME);
            std::fs::write(
                &marker,
                "This folder is a MuMDIA storage folder (the Setup screen's Storage folder).\n",
            )
            .map_err(|e| format!("cannot write {}: {e}", marker.display()))?;
            Bootstrap {
                folder: Some(p),
                move_environments,
            }
        }
        _ => Bootstrap::default(),
    };
    save_to(&root, &b)?;
    Ok(layout_of(root, &b))
}

/// The storage folder as the Setup screen shows it.
#[derive(Serialize, Clone, Debug)]
pub struct Info {
    pub folder: String,
    pub default_folder: String,
    pub configured: bool,
    pub move_environments: bool,
    /// Where the Python environments and the downloaded tools are.
    pub environments: String,
    pub free_bytes: Option<u64>,
    /// Whether a file can be written and renamed in the folder (or where it would be).
    pub writable: bool,
    pub error: Option<String>,
    /// Location and low-space warnings.
    pub warnings: Vec<String>,
    /// Whether `MUMDIA_DATA_DIR` decides the default folder.
    pub data_dir_env: bool,
}

/// What the Setup screen shows about the storage folder.
pub fn info() -> Info {
    let l = layout();
    let check = writable_without_creating(&l.storage);
    let free = crate::preflight::free_space(&l.storage);
    let mut warnings = location_warnings("storage folder", &l.storage);
    if let Some(f) = free.filter(|f| *f < LOW_SPACE_BYTES) {
        warnings.push(low_space(&l.storage, f));
    }
    Info {
        folder: l.storage.display().to_string(),
        default_folder: l.default_root.display().to_string(),
        configured: l.configured,
        move_environments: l.move_environments,
        environments: l.tools.display().to_string(),
        free_bytes: free,
        writable: check.is_ok(),
        error: check.err(),
        warnings,
        data_dir_env: std::env::var_os("MUMDIA_DATA_DIR").is_some_and(|v| !v.is_empty()),
    }
}

fn low_space(dir: &Path, free: u64) -> String {
    format!(
        "The storage folder {} has {:.1} GB free. Caches, predicted libraries and temporary \
         files can take tens of gigabytes; choose a folder on a larger disk on the Setup \
         screen.",
        dir.display(),
        free as f64 / 1e9
    )
}

/// The storage part of a search's preflight: `(blockers, warnings)`. The results folder
/// must be writable (checked without creating it) and so must a chosen storage folder;
/// a network or OneDrive location and a nearly full storage folder are warnings.
pub fn preflight(out_dir: &str) -> (Vec<String>, Vec<String>) {
    let mut blockers = Vec::new();
    let mut warnings = Vec::new();
    if !out_dir.trim().is_empty() {
        let out = Path::new(out_dir);
        if let Err(e) = writable_without_creating(out) {
            blockers.push(format!(
                "The results folder cannot be written ({e}). Choose a folder you can write \
                 to, for example one in your Documents on a local disk."
            ));
        }
        warnings.extend(location_warnings("results folder", out));
    }
    let l = layout();
    if l.configured {
        match probe_writable(&l.storage) {
            Err(e) => blockers.push(format!(
                "The storage folder chosen on the Setup screen cannot be written ({e}). \
                 Choose another one there, or go back to the default."
            )),
            Ok(()) => {
                warnings.extend(location_warnings("storage folder", &l.storage));
                if let Some(f) = crate::preflight::free_space(&l.storage) {
                    if f < LOW_SPACE_BYTES {
                        warnings.push(low_space(&l.storage, f));
                    }
                }
            }
        }
    }
    (blockers, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia-storage-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_default_folder_follows_mumdia_data_dir_then_the_platform() {
        use std::ffi::OsString;
        let env = |pairs: Vec<(&'static str, String)>| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| OsString::from(v))
            }
        };
        let abs = std::env::temp_dir().join("mumdia-data-dir");
        let a = abs.display().to_string();
        assert_eq!(
            default_root_with(&env(vec![("MUMDIA_DATA_DIR", a.clone())]), true),
            abs
        );
        // A relative MUMDIA_DATA_DIR is ignored rather than resolved against wherever the
        // application happened to start.
        assert_eq!(
            default_root_with(
                &env(vec![
                    ("MUMDIA_DATA_DIR", "rel".into()),
                    ("LOCALAPPDATA", a.clone())
                ]),
                true
            ),
            abs.join("MuMDIA")
        );
        assert_eq!(
            default_root_with(&env(vec![("HOME", a.clone())]), false),
            abs.join(".local/share").join("MuMDIA")
        );
    }

    #[test]
    fn the_setting_round_trips_and_the_default_leaves_no_file() {
        let root = scratch("roundtrip");
        let chosen = scratch("roundtrip-chosen");
        assert_eq!(load_from(&root), Bootstrap::default());
        let b = Bootstrap {
            folder: Some(chosen.clone()),
            move_environments: true,
        };
        save_to(&root, &b).unwrap();
        assert_eq!(load_from(&root), b);
        let l = layout_of(root.clone(), &b);
        assert!(l.configured && l.move_environments);
        assert_eq!(
            (l.storage.clone(), l.tools.clone()),
            (chosen.clone(), chosen.clone())
        );
        // Without the option the environments stay in the default folder.
        let keep = Bootstrap {
            folder: Some(chosen.clone()),
            move_environments: false,
        };
        let l = layout_of(root.clone(), &keep);
        assert_eq!((l.storage, l.tools), (chosen.clone(), root.clone()));
        // Back to the default: the file is removed.
        save_to(&root, &Bootstrap::default()).unwrap();
        assert!(!root.join(BOOTSTRAP_NAME).exists());
        // A damaged file is the default, never an error that stops the application.
        std::fs::write(root.join(BOOTSTRAP_NAME), b"{not json").unwrap();
        assert_eq!(load_from(&root), Bootstrap::default());
        // The default folder chosen explicitly, or a relative path, is not "configured",
        // and moving the environments means nothing without a folder.
        for f in [Some(root.clone()), Some(PathBuf::from("relative")), None] {
            let l = layout_of(
                root.clone(),
                &Bootstrap {
                    folder: f,
                    move_environments: true,
                },
            );
            assert!(!l.configured && !l.move_environments, "{l:?}");
            assert_eq!(l.tools, root);
        }
    }

    #[test]
    fn a_folder_is_refused_unless_it_is_absolute_and_writable() {
        assert!(validate("").is_err());
        assert!(validate("relative/folder")
            .unwrap_err()
            .contains("not a full path"));
        let base = scratch("validate");
        let file = base.join("file");
        std::fs::write(&file, b"x").unwrap();
        assert!(validate(&file.display().to_string())
            .unwrap_err()
            .contains("is a file"));
        assert!(validate(&file.join("below").display().to_string()).is_err());
        let ok = base.join("new");
        assert_eq!(validate(&ok.display().to_string()).unwrap(), ok);
        assert!(
            ok.is_dir(),
            "a chosen folder that does not exist yet is created"
        );
        assert_eq!(
            std::fs::read_dir(&ok).unwrap().count(),
            0,
            "the probe is removed"
        );
        // An empty folder is used as it is; one holding someone's files gets a MuMDIA
        // sub-folder, unless it is already a storage folder.
        assert_eq!(validate(&ok.display().to_string()).unwrap(), ok);
        let busy = base.join("documents");
        std::fs::create_dir_all(busy.join("configs")).unwrap();
        assert_eq!(
            validate(&busy.display().to_string()).unwrap(),
            busy.join("MuMDIA")
        );
        std::fs::write(busy.join(MARKER_NAME), b"").unwrap();
        assert_eq!(validate(&busy.display().to_string()).unwrap(), busy);
    }

    #[test]
    fn the_child_environment_is_exported_only_when_configured_and_never_overrides() {
        let root = PathBuf::from(if cfg!(windows) { r"C:\d" } else { "/d" });
        let chosen = PathBuf::from(if cfg!(windows) { r"E:\s" } else { "/s" });
        let none = |_: &str| false;
        assert!(child_env_of(&layout_of(root.clone(), &Bootstrap::default()), &none).is_empty());
        let b = Bootstrap {
            folder: Some(chosen.clone()),
            move_environments: false,
        };
        let env = child_env_of(&layout_of(root.clone(), &b), &none);
        let names: Vec<&str> = env.iter().map(|(k, _)| *k).collect();
        assert_eq!(
            names,
            [
                "MUMDIA_CACHE_DIR",
                "MUMDIA_TEMP_DIR",
                "MUMDIA_MODEL_DIR",
                "UV_CACHE_DIR"
            ]
        );
        assert_eq!(env[0].1, chosen.join(CACHE_SUBDIR));
        assert_eq!(env[2].1, chosen.join(MODELS_SUBDIR));
        // A variable the user set is left to the user.
        let set = |k: &str| k == "MUMDIA_CACHE_DIR";
        let env = child_env_of(&layout_of(root.clone(), &b), &set);
        assert!(env.iter().all(|(k, _)| *k != "MUMDIA_CACHE_DIR"));
        // Moving the environments moves uv's interpreters too.
        let moved = Bootstrap {
            folder: Some(chosen.clone()),
            move_environments: true,
        };
        let env = child_env_of(&layout_of(root, &moved), &none);
        assert!(env
            .iter()
            .any(|(k, v)| *k == "UV_PYTHON_INSTALL_DIR" && *v == chosen.join(UV_PYTHON_SUBDIR)));
    }

    #[test]
    fn network_and_onedrive_folders_are_warned_about() {
        assert!(is_network_path(Path::new(r"\\srv\share\x")));
        assert!(is_network_path(Path::new(r"\\?\UNC\srv\share")));
        assert!(!is_network_path(Path::new(r"\\?\C:\x")));
        assert!(!is_network_path(Path::new("/home/u")));
        let roots = [PathBuf::from(r"C:\Users\u\Contoso Files")];
        assert!(in_onedrive(
            Path::new(r"C:\Users\u\Contoso Files\r"),
            &roots
        ));
        assert!(!in_onedrive(
            Path::new(r"C:\Users\u\Contoso Filesx"),
            &roots
        ));
        assert!(in_onedrive(Path::new("/Users/u/OneDrive - UGent/r"), &[]));
        let w = location_warnings_with("results folder", Path::new("//srv/OneDrive/r"), &[]);
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].contains("network share") && w[1].contains("OneDrive"));
        assert!(location_warnings_with("x", Path::new("/data/r"), &[]).is_empty());
    }

    #[test]
    fn checking_a_folder_that_does_not_exist_creates_nothing() {
        let base = scratch("nocreate");
        let missing = base.join("a/b");
        writable_without_creating(&missing).unwrap();
        assert!(!base.join("a").exists());
        let file = base.join("f");
        std::fs::write(&file, b"x").unwrap();
        assert!(writable_without_creating(&file.join("x")).is_err());
    }

    #[test]
    fn an_unwritable_results_folder_blocks_the_search() {
        let base = scratch("preflight");
        let file = base.join("f");
        std::fs::write(&file, b"x").unwrap();
        let (blockers, _) = preflight(&file.join("results").display().to_string());
        assert!(
            blockers
                .iter()
                .any(|b| b.contains("results folder cannot be written")),
            "{blockers:?}"
        );
        let (blockers, _) = preflight(&base.join("results").display().to_string());
        assert!(
            blockers.iter().all(|b| !b.contains("results folder")),
            "{blockers:?}"
        );
    }
}
