//! The engine's on-disk caches: where they live and how large they may grow.
//!
//! Two caches share one root. `libraries/` holds FASTA-built spectral libraries
//! (`predict_frag.library_cache`, [`crate::library_cache`]), keyed by everything that
//! determines a library, so a later search of the same FASTA under the same settings
//! skips digest, peptidoforms and predict-frag. `deeplc_projections/` holds DeepLC's
//! run-independent trunk projection (`rt_im_train.deeplc_projection_cache`, the
//! `--projection-cache` of `deeplc_finetune.py`), which every multi-head calibration of
//! the same sequence list would otherwise recompute.
//!
//! Both settings default to `"auto"`: that cache's sub-directory of [`root`]. The root is
//! `MUMDIA_CACHE_DIR` when it is set, else the platform's per-user cache directory:
//! `$XDG_CACHE_HOME/mumdia` or `~/.cache/mumdia` on Linux and other Unix systems,
//! `~/Library/Caches/mumdia` on macOS, `%LOCALAPPDATA%\mumdia\cache` on Windows. It is an
//! environment variable rather than a configuration value so that a configuration and
//! its hash do not name one machine's directories. `MUMDIA_CACHE_DIR=off` turns every
//! `"auto"` cache off; a path in the configuration is used as given, and `null` (or
//! `"off"`) turns that one cache off.
//!
//! The caches are bounded. After a run stores into them, [`enforce_budget`] removes the
//! least recently used entries until the total is within `MUMDIA_CACHE_MAX_GB` (default
//! 100, in GiB like the other `_GB` variables; `0` or `unlimited` for no bound). An entry
//! used within the last hour is never removed, so a run that is reading one keeps it, and
//! an entry is renamed aside before it is deleted, so a reader finds a whole entry or
//! none. Only directories named like the engine's own entries are ever touched: a
//! 24-character (library) or 40-character (projection) lowercase hex key, and the
//! temporary directories derived from one.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use mumdia_core::config::Config;
use tracing::{debug, info, warn};

/// The setting that means "the default location".
pub const AUTO: &str = "auto";
/// Sub-directory of the root for [`crate::library_cache`].
pub const LIBRARIES: &str = "libraries";
/// Sub-directory of the root for DeepLC's projection cache.
pub const PROJECTIONS: &str = "deeplc_projections";
/// `MUMDIA_CACHE_MAX_GB` when it is unset.
pub const DEFAULT_MAX_GB: f64 = 100.0;
/// The file whose modification time records an entry's last use.
pub const LAST_USED: &str = "last_used";
/// An entry used this recently is never evicted, and a leftover temporary directory this
/// recently written is left alone: a run that is reading or writing it may still be alive.
/// A copy of the largest library takes minutes, not an hour.
const PROTECT: Duration = Duration::from_secs(60 * 60);
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The platform the default root is chosen for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Os {
    Windows,
    Mac,
    Unix,
}

impl Os {
    fn current() -> Os {
        if cfg!(windows) {
            Os::Windows
        } else if cfg!(target_os = "macos") {
            Os::Mac
        } else {
            Os::Unix
        }
    }
}

/// Where the cache root is, or why there is none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Root {
    /// The directory, and where it came from (`MUMDIA_CACHE_DIR`, `XDG_CACHE_HOME`, ...).
    Dir(PathBuf, &'static str),
    /// No default cache, with the reason.
    Off(String),
}

fn is_off_word(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "off" | "0" | "false" | "no" | "none"
    )
}

fn root_with(env: &dyn Fn(&str) -> Option<String>, os: Os) -> Root {
    let set = |k: &str| env(k).filter(|v| !v.trim().is_empty());
    if let Some(v) = set("MUMDIA_CACHE_DIR") {
        if is_off_word(&v) {
            return Root::Off(format!("MUMDIA_CACHE_DIR={}", v.trim()));
        }
        return Root::Dir(PathBuf::from(v.trim()), "MUMDIA_CACHE_DIR");
    }
    match os {
        Os::Windows => match set("LOCALAPPDATA") {
            Some(v) => Root::Dir(
                PathBuf::from(v).join("mumdia").join("cache"),
                "LOCALAPPDATA",
            ),
            None => Root::Off("neither MUMDIA_CACHE_DIR nor LOCALAPPDATA is set".into()),
        },
        Os::Mac => match set("HOME") {
            Some(h) => Root::Dir(PathBuf::from(h).join("Library/Caches/mumdia"), "HOME"),
            None => Root::Off("neither MUMDIA_CACHE_DIR nor HOME is set".into()),
        },
        Os::Unix => {
            // The XDG base-directory specification ignores a relative XDG_CACHE_HOME. A Unix
            // absolute path starts with `/` (not `Path::is_absolute`, which would judge it
            // by the host's rules when this runs elsewhere, as the tests do).
            if let Some(x) = set("XDG_CACHE_HOME").filter(|x| x.starts_with('/')) {
                return Root::Dir(PathBuf::from(x).join("mumdia"), "XDG_CACHE_HOME");
            }
            match set("HOME") {
                Some(h) => Root::Dir(PathBuf::from(h).join(".cache/mumdia"), "HOME"),
                None => Root::Off("neither MUMDIA_CACHE_DIR nor HOME is set".into()),
            }
        }
    }
}

/// The cache root of this process's environment. Every variable is read by its literal
/// name, so the generated configuration reference (docs/24) lists each one.
pub fn root() -> Root {
    let vars = [
        ("MUMDIA_CACHE_DIR", std::env::var("MUMDIA_CACHE_DIR").ok()),
        ("XDG_CACHE_HOME", std::env::var("XDG_CACHE_HOME").ok()),
        ("HOME", std::env::var("HOME").ok()),
        ("LOCALAPPDATA", std::env::var("LOCALAPPDATA").ok()),
    ];
    let lookup = |k: &str| {
        vars.iter()
            .find(|(name, _)| *name == k)
            .and_then(|(_, v)| v.clone())
    };
    root_with(&lookup, Os::current())
}

/// A cache directory a setting resolved to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheDir {
    pub path: PathBuf,
    /// Whether it came from `"auto"` (the root's sub-directory) rather than a path the
    /// configuration names.
    pub auto: bool,
}

impl CacheDir {
    pub fn as_str(&self) -> String {
        self.path.to_string_lossy().into_owned()
    }
}

/// The directory of one cache setting: `None` (and `"off"`) is off, `"auto"` is `sub`
/// under the root (off when there is no root), anything else is the directory it names.
fn resolve_with(setting: Option<&str>, sub: &str, root: &Root) -> Option<CacheDir> {
    let s = setting?.trim();
    if s.is_empty() || is_off_word(s) {
        return None;
    }
    if s.eq_ignore_ascii_case(AUTO) {
        return match root {
            Root::Dir(d, _) => Some(CacheDir {
                path: d.join(sub),
                auto: true,
            }),
            Root::Off(_) => None,
        };
    }
    Some(CacheDir {
        path: PathBuf::from(s),
        auto: false,
    })
}

/// Where `predict_frag.library_cache` points in this environment, if anywhere.
pub fn library_dir(cfg: &Config) -> Option<CacheDir> {
    resolve_with(
        cfg.predict_frag.library_cache.as_deref(),
        LIBRARIES,
        &root(),
    )
}

/// Where `rt_im_train.deeplc_projection_cache` points in this environment, if anywhere.
pub fn projection_dir(cfg: &Config) -> Option<CacheDir> {
    resolve_with(
        cfg.rt_im_train.deeplc_projection_cache.as_deref(),
        PROJECTIONS,
        &root(),
    )
}

/// The byte budget from `MUMDIA_CACHE_MAX_GB` (GiB): `None` is no bound.
fn budget_with(env: &dyn Fn(&str) -> Option<String>) -> Option<u64> {
    let default = Some((DEFAULT_MAX_GB * GIB) as u64);
    let Some(v) = env("MUMDIA_CACHE_MAX_GB").filter(|v| !v.trim().is_empty()) else {
        return default;
    };
    let v = v.trim();
    if v.eq_ignore_ascii_case("unlimited") || is_off_word(v) {
        return None;
    }
    match v.parse::<f64>() {
        Ok(g) if g.is_finite() && g > 0.0 => Some((g * GIB) as u64),
        _ => {
            warn!(
                value = v,
                "MUMDIA_CACHE_MAX_GB is not a positive number, 0 or `unlimited`; using the \
                 default of {DEFAULT_MAX_GB} GiB"
            );
            default
        }
    }
}

/// The byte budget of this process's environment.
pub fn budget() -> Option<u64> {
    let max_gb = std::env::var("MUMDIA_CACHE_MAX_GB").ok();
    budget_with(&|k| {
        (k == "MUMDIA_CACHE_MAX_GB")
            .then(|| max_gb.clone())
            .flatten()
    })
}

/// The key of an engine cache entry's directory name: 24 (library) or 40 (projection)
/// lowercase hex characters.
fn entry_key(name: &str) -> Option<&str> {
    let is_key = |s: &str| {
        matches!(s.len(), 24 | 40)
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    is_key(name).then_some(name)
}

/// Whether a name is a temporary directory derived from an entry key: a store in progress
/// or abandoned (`.partial-`, `.tmp-`), an entry set aside (`.broken-`) or being evicted
/// (`.evicted-`).
fn is_leftover(name: &str) -> bool {
    let Some((key, rest)) = name.split_once('.') else {
        return false;
    };
    entry_key(key).is_some()
        && ["partial-", "tmp-", "broken-", "evicted-"]
            .iter()
            .any(|p| rest.starts_with(p))
}

/// Total size of the files under `dir`.
fn dir_bytes(dir: &Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            match e.file_type() {
                Ok(t) if t.is_dir() => stack.push(e.path()),
                Ok(_) => total += e.metadata().map(|m| m.len()).unwrap_or(0),
                Err(_) => {}
            }
        }
    }
    total
}

/// When an entry was last used: its [`LAST_USED`] file's modification time, else the
/// directory's own (the store that published it).
fn last_used(entry: &Path) -> SystemTime {
    let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    mtime(&entry.join(LAST_USED))
        .or_else(|| mtime(entry))
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Record that the entry at `entry` was used now. A cache that cannot be written (a
/// shared, read-only one) still serves hits, so a failure is a debug line only.
pub fn mark_used(entry: &Path) {
    let stamp = format!(
        "{}\n",
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    if let Err(e) = std::fs::write(entry.join(LAST_USED), stamp) {
        debug!(entry = %entry.display(), error = %e, "cache: could not record the entry's use");
    }
}

/// Whether `t` lies within `window` of `now` (a time in the future counts as recent: clock
/// skew on a shared cache is not evidence that nobody is using the entry).
fn recent(t: SystemTime, now: SystemTime, window: Duration) -> bool {
    match now.duration_since(t) {
        Ok(age) => age < window,
        Err(_) => true,
    }
}

/// One entry found in a cache directory.
#[derive(Debug)]
struct Entry {
    path: PathBuf,
    bytes: u64,
    last_used: SystemTime,
}

/// What a cache directory holds.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Usage {
    pub entries: usize,
    pub bytes: u64,
}

/// The entries of `dir` (not its leftovers), with their sizes and last uses.
fn scan(dir: &Path) -> Vec<Entry> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<Entry> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter(|e| entry_key(&e.file_name().to_string_lossy()).is_some())
        .map(|e| {
            let path = e.path();
            Entry {
                bytes: dir_bytes(&path),
                last_used: last_used(&path),
                path,
            }
        })
        .collect();
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// What `dir` holds: its entries and their total size.
pub fn usage(dir: &Path) -> Usage {
    let entries = scan(dir);
    Usage {
        entries: entries.len(),
        bytes: entries.iter().map(|e| e.bytes).sum(),
    }
}

/// Remove the leftover temporary directories of `dir` that nothing has written for
/// [`PROTECT`]: a killed store's, an entry set aside as broken, an eviction that did not
/// finish. Returns the bytes removed.
fn remove_stale_leftovers(dir: &Path, now: SystemTime) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut freed = 0u64;
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !is_leftover(&name) || !e.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        let path = e.path();
        let newest = std::fs::read_dir(&path)
            .map(|rd| {
                rd.flatten()
                    .filter_map(|f| f.metadata().and_then(|m| m.modified()).ok())
                    .max()
            })
            .ok()
            .flatten();
        let written = [
            newest,
            std::fs::metadata(&path).and_then(|m| m.modified()).ok(),
        ]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or(SystemTime::UNIX_EPOCH);
        if recent(written, now, PROTECT) {
            continue;
        }
        let bytes = dir_bytes(&path);
        match std::fs::remove_dir_all(&path) {
            Ok(()) => freed += bytes,
            Err(e) => {
                debug!(dir = %path.display(), error = %e, "cache: could not remove a leftover")
            }
        }
    }
    freed
}

/// What [`enforce_budget`] did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Enforced {
    /// Entries and bytes left in the caches.
    pub kept: Usage,
    /// Entries removed to meet the budget, and their bytes.
    pub evicted: Usage,
    /// Bytes of stale leftover directories removed.
    pub leftovers_freed: u64,
    /// The caches are still above the budget: what is left was used within the last hour.
    pub over_budget: bool,
}

/// Remove the least recently used entries of `dirs`, taken together, until their total
/// size is within `budget` bytes (`None`: no bound). An entry used within [`PROTECT`] of
/// `now` is kept whatever the total. Stale leftovers are removed first.
fn enforce_at(dirs: &[PathBuf], budget: Option<u64>, now: SystemTime) -> Enforced {
    let mut out = Enforced::default();
    let mut entries: Vec<Entry> = Vec::new();
    let mut seen: Vec<PathBuf> = Vec::new();
    for d in dirs {
        // Two settings may name the same directory; count it once.
        if seen.contains(d) {
            continue;
        }
        seen.push(d.clone());
        out.leftovers_freed += remove_stale_leftovers(d, now);
        entries.extend(scan(d));
    }
    let mut total: u64 = entries.iter().map(|e| e.bytes).sum();
    let mut kept = entries.len();
    if let Some(budget) = budget {
        // Oldest use first; the path breaks ties, so the order never depends on the
        // directory listing.
        entries.sort_by(|a, b| a.last_used.cmp(&b.last_used).then(a.path.cmp(&b.path)));
        for e in &entries {
            if total <= budget {
                break;
            }
            if recent(e.last_used, now, PROTECT) {
                continue;
            }
            // Renamed aside first: a reader that opens the entry after this sees no
            // entry (a miss), never a half-deleted one.
            let aside = e.path.with_file_name(format!(
                "{}.evicted-{}-{}",
                e.path.file_name().unwrap_or_default().to_string_lossy(),
                std::process::id(),
                now.duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            if let Err(err) = std::fs::rename(&e.path, &aside) {
                debug!(entry = %e.path.display(), error = %err, "cache: could not set an entry aside");
                continue;
            }
            if let Err(err) = std::fs::remove_dir_all(&aside) {
                debug!(dir = %aside.display(), error = %err, "cache: could not delete an evicted entry; a later run removes it");
            }
            total -= e.bytes;
            kept -= 1;
            out.evicted.entries += 1;
            out.evicted.bytes += e.bytes;
        }
        out.over_budget = total > budget;
    }
    out.kept = Usage {
        entries: kept,
        bytes: total,
    };
    out
}

/// Remove the least recently used entries of `dirs`, taken together, until their total
/// size is within `budget` bytes (`None`: no bound). An entry used within the last hour
/// is kept whatever the total, and leftover temporary directories unwritten for an hour
/// are removed first.
pub fn enforce_budget(dirs: &[PathBuf], budget: Option<u64>) -> Enforced {
    enforce_at(dirs, budget, SystemTime::now())
}

/// The cache directories this configuration uses in this environment.
pub fn dirs_of(cfg: &Config) -> Vec<PathBuf> {
    [library_dir(cfg), projection_dir(cfg)]
        .into_iter()
        .flatten()
        .map(|d| d.path)
        .collect()
}

/// Bring this configuration's caches within `MUMDIA_CACHE_MAX_GB`, logging what was
/// removed. Never fails: a cache that cannot be trimmed costs disk, not a run.
pub fn enforce_for(cfg: &Config) {
    let dirs = dirs_of(cfg);
    if dirs.is_empty() {
        return;
    }
    let budget = budget();
    let r = enforce_budget(&dirs, budget);
    let gib = |b: u64| format!("{:.2} GiB", b as f64 / GIB);
    let limit = budget.map_or("unlimited".to_string(), gib);
    if r.evicted.entries > 0 || r.leftovers_freed > 0 {
        info!(
            evicted = r.evicted.entries,
            evicted_size = %gib(r.evicted.bytes),
            leftovers_freed = %gib(r.leftovers_freed),
            kept = r.kept.entries,
            kept_size = %gib(r.kept.bytes),
            limit = %limit,
            "cache: removed the least recently used entries (MUMDIA_CACHE_MAX_GB)"
        );
    }
    if r.over_budget {
        warn!(
            size = %gib(r.kept.bytes),
            limit = %limit,
            "cache: above MUMDIA_CACHE_MAX_GB, and every remaining entry was used within the \
             last hour, so none was removed; raise the limit or point MUMDIA_CACHE_DIR at a \
             larger disk"
        );
    }
}

/// One cache as `mumdia doctor` reports it.
#[derive(Debug, serde::Serialize)]
pub struct CacheReport {
    /// The configuration field.
    pub field: &'static str,
    /// Its value.
    pub setting: Option<String>,
    /// The directory it resolved to, if it is on.
    pub dir: Option<String>,
    pub auto: bool,
    pub entries: usize,
    pub bytes: u64,
}

/// The caches of `cfg` in this environment: the root, the budget and each cache.
#[derive(Debug, serde::Serialize)]
pub struct CachesReport {
    pub root: Option<String>,
    pub root_source: String,
    pub max_bytes: Option<u64>,
    pub caches: Vec<CacheReport>,
}

pub fn report(cfg: &Config) -> CachesReport {
    let (root, root_source) = match root() {
        Root::Dir(d, src) => (Some(d.to_string_lossy().into_owned()), src.to_string()),
        Root::Off(why) => (None, why),
    };
    let one = |field: &'static str, setting: Option<&String>, dir: Option<CacheDir>| {
        let u = dir.as_ref().map(|d| usage(&d.path)).unwrap_or_default();
        CacheReport {
            field,
            setting: setting.cloned(),
            auto: dir.as_ref().is_some_and(|d| d.auto),
            dir: dir.map(|d| d.as_str()),
            entries: u.entries,
            bytes: u.bytes,
        }
    };
    CachesReport {
        root,
        root_source,
        max_bytes: budget(),
        caches: vec![
            one(
                "predict_frag.library_cache",
                cfg.predict_frag.library_cache.as_ref(),
                library_dir(cfg),
            ),
            one(
                "rt_im_train.deeplc_projection_cache",
                cfg.rt_im_train.deeplc_projection_cache.as_ref(),
                projection_dir(cfg),
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia_cache_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_root_follows_the_environment_then_the_platform_convention() {
        let r = |pairs: &[(&str, &str)], os| root_with(&env_of(pairs), os);
        assert_eq!(
            r(
                &[("MUMDIA_CACHE_DIR", "/scratch/c"), ("HOME", "/h")],
                Os::Unix
            ),
            Root::Dir(PathBuf::from("/scratch/c"), "MUMDIA_CACHE_DIR")
        );
        for off in ["off", "OFF", "0", "false", "none"] {
            assert!(matches!(
                r(&[("MUMDIA_CACHE_DIR", off), ("HOME", "/h")], Os::Unix),
                Root::Off(_)
            ));
        }
        // An empty value is unset, not a directory named "".
        assert_eq!(
            r(&[("MUMDIA_CACHE_DIR", " "), ("HOME", "/h")], Os::Unix),
            Root::Dir(PathBuf::from("/h/.cache/mumdia"), "HOME")
        );
        assert_eq!(
            r(&[("XDG_CACHE_HOME", "/x"), ("HOME", "/h")], Os::Unix),
            Root::Dir(PathBuf::from("/x/mumdia"), "XDG_CACHE_HOME")
        );
        // A relative XDG_CACHE_HOME is ignored, as the specification says.
        assert_eq!(
            r(&[("XDG_CACHE_HOME", "rel"), ("HOME", "/h")], Os::Unix),
            Root::Dir(PathBuf::from("/h/.cache/mumdia"), "HOME")
        );
        assert_eq!(
            r(&[("HOME", "/Users/u")], Os::Mac),
            Root::Dir(PathBuf::from("/Users/u/Library/Caches/mumdia"), "HOME")
        );
        assert_eq!(
            r(&[("LOCALAPPDATA", "C:/Users/u/AppData/Local")], Os::Windows),
            Root::Dir(
                PathBuf::from("C:/Users/u/AppData/Local")
                    .join("mumdia")
                    .join("cache"),
                "LOCALAPPDATA"
            )
        );
        assert!(matches!(r(&[], Os::Unix), Root::Off(_)));
        assert!(matches!(r(&[("HOME", "/h")], Os::Windows), Root::Off(_)));
    }

    #[test]
    fn a_setting_is_off_auto_or_a_directory() {
        let root = Root::Dir(PathBuf::from("/r"), "HOME");
        assert_eq!(resolve_with(None, LIBRARIES, &root), None);
        assert_eq!(resolve_with(Some("off"), LIBRARIES, &root), None);
        assert_eq!(resolve_with(Some(""), LIBRARIES, &root), None);
        assert_eq!(
            resolve_with(Some("auto"), LIBRARIES, &root),
            Some(CacheDir {
                path: PathBuf::from("/r").join(LIBRARIES),
                auto: true
            })
        );
        assert_eq!(
            resolve_with(Some("AUTO"), PROJECTIONS, &root),
            Some(CacheDir {
                path: PathBuf::from("/r").join(PROJECTIONS),
                auto: true
            })
        );
        // `auto` without a root is off; a named directory does not need one.
        let off = Root::Off("MUMDIA_CACHE_DIR=off".into());
        assert_eq!(resolve_with(Some("auto"), LIBRARIES, &off), None);
        assert_eq!(
            resolve_with(Some("/data/libs"), LIBRARIES, &off),
            Some(CacheDir {
                path: PathBuf::from("/data/libs"),
                auto: false
            })
        );
    }

    #[test]
    fn the_budget_defaults_to_100_gib_and_accepts_no_bound() {
        let b = |pairs: &[(&str, &str)]| budget_with(&env_of(pairs));
        assert_eq!(b(&[]), Some(100 * 1024 * 1024 * 1024));
        assert_eq!(
            b(&[("MUMDIA_CACHE_MAX_GB", "2.5")]),
            Some((2.5 * GIB) as u64)
        );
        assert_eq!(b(&[("MUMDIA_CACHE_MAX_GB", "0")]), None);
        assert_eq!(b(&[("MUMDIA_CACHE_MAX_GB", "unlimited")]), None);
        assert_eq!(
            b(&[("MUMDIA_CACHE_MAX_GB", "-3")]),
            Some(100 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            b(&[("MUMDIA_CACHE_MAX_GB", "lots")]),
            Some(100 * 1024 * 1024 * 1024)
        );
    }

    #[test]
    fn only_the_engines_own_names_are_entries_or_leftovers() {
        let k24 = "0123456789abcdef01234567";
        let k40 = "0123456789abcdef0123456789abcdef01234567";
        assert_eq!(entry_key(k24), Some(k24));
        assert_eq!(entry_key(k40), Some(k40));
        assert_eq!(
            entry_key("0123456789ABCDEF01234567"),
            None,
            "lowercase only"
        );
        assert_eq!(entry_key("notes"), None);
        assert_eq!(entry_key(&k24[..23]), None);
        assert!(is_leftover(&format!("{k24}.partial-12-34")));
        assert!(is_leftover(&format!("{k40}.tmp-99.ab_c")));
        assert!(is_leftover(&format!("{k24}.broken-1-2")));
        assert!(is_leftover(&format!("{k40}.evicted-1-2")));
        assert!(!is_leftover(&format!("{k24}.backup")));
        assert!(!is_leftover("mine.tmp-1"));
    }

    /// An entry directory of `bytes` bytes, last used `age` before `now`.
    fn entry(dir: &Path, key: &str, bytes: usize, used: SystemTime) -> PathBuf {
        let e = dir.join(key);
        std::fs::create_dir_all(&e).unwrap();
        std::fs::write(e.join("data.bin"), vec![7u8; bytes]).unwrap();
        mark_used(&e);
        let f = std::fs::File::options()
            .write(true)
            .open(e.join(LAST_USED))
            .unwrap();
        f.set_modified(used).unwrap();
        e
    }

    #[test]
    fn eviction_removes_the_least_recently_used_entries_until_the_budget_holds() {
        let libs = tmp("evict_libs");
        let proj = tmp("evict_proj");
        let now = SystemTime::now();
        let ago = |h: u64| now - Duration::from_secs(h * 3600);
        let old_lib = entry(&libs, &"a".repeat(24), 400, ago(30));
        let mid_proj = entry(&proj, &"b".repeat(40), 300, ago(20));
        let new_lib = entry(&libs, &"c".repeat(24), 200, ago(10));
        // Not the engine's: never counted, never removed.
        std::fs::create_dir_all(libs.join("my_notes")).unwrap();
        std::fs::write(libs.join("my_notes/x"), vec![0u8; 5000]).unwrap();
        let r = enforce_at(&[libs.clone(), proj.clone()], Some(600), now);
        assert_eq!(
            r.evicted,
            Usage {
                entries: 1,
                bytes: 400 + LAST_USED_BYTES
            }
        );
        assert!(
            !old_lib.exists(),
            "the least recently used entry goes first"
        );
        assert!(mid_proj.exists() && new_lib.exists());
        assert!(libs.join("my_notes/x").exists());
        assert!(!r.over_budget);
        assert_eq!(r.kept.entries, 2);
        // No bound: nothing is removed.
        let r = enforce_at(&[libs.clone(), proj.clone()], None, now);
        assert_eq!(r.evicted, Usage::default());
        assert_eq!(r.kept.entries, 2);
    }

    /// The size of the `last_used` stamp `mark_used` writes (the seconds and a newline).
    const LAST_USED_BYTES: u64 = 11;

    #[test]
    fn an_entry_used_within_the_hour_is_kept_even_over_the_budget() {
        let libs = tmp("evict_recent");
        let now = SystemTime::now();
        let busy = entry(&libs, &"d".repeat(24), 1000, now - Duration::from_secs(60));
        let r = enforce_at(std::slice::from_ref(&libs), Some(10), now);
        assert!(busy.exists());
        assert_eq!(r.evicted, Usage::default());
        assert!(r.over_budget);
    }

    #[test]
    fn stale_leftovers_are_removed_and_fresh_ones_kept() {
        let libs = tmp("leftovers");
        let now = SystemTime::now();
        let stale = libs.join(format!("{}.partial-1-2", "e".repeat(24)));
        let fresh = libs.join(format!("{}.tmp-3.x", "f".repeat(40)));
        for d in [&stale, &fresh] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join("part"), vec![1u8; 100]).unwrap();
        }
        let later = now + PROTECT + Duration::from_secs(60);
        // Seen from two hours later only the stale one is old enough; the fresh one is
        // rewritten "then" to stay recent.
        std::fs::File::options()
            .write(true)
            .open(fresh.join("part"))
            .unwrap()
            .set_modified(later)
            .unwrap();
        let r = enforce_at(std::slice::from_ref(&libs), None, later);
        assert!(!stale.exists());
        assert!(fresh.exists());
        assert_eq!(r.leftovers_freed, 100);
    }

    #[test]
    fn marking_an_entry_used_moves_its_last_use() {
        let libs = tmp("mark");
        let now = SystemTime::now();
        let e = entry(
            &libs,
            &"9".repeat(24),
            10,
            now - Duration::from_secs(5 * 3600),
        );
        assert!(!recent(last_used(&e), now, PROTECT));
        mark_used(&e);
        assert!(recent(last_used(&e), SystemTime::now(), PROTECT));
    }
}
