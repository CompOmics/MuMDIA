//! Where the engine writes, and whether it can.
//!
//! A run writes to five places, and only the first is named on the command line of a
//! stage:
//!
//! | what | where | moved by |
//! |---|---|---|
//! | results | `--out-dir` | the flag |
//! | rescore sidecar files | `<out-dir>/sidecar_work` | `MUMDIA_SIDECAR_DIR` |
//! | caches ([`crate::cache`]) | the per-user cache directory | `--cache-dir`, `MUMDIA_CACHE_DIR` |
//! | temporary files of the engine and of every process it starts | the system's | `--temp-dir`, `MUMDIA_TEMP_DIR` |
//! | predictor model downloads (MS2PIP, AlphaPeptDeep) | each package's default under the home directory | `MUMDIA_MODEL_DIR` |
//!
//! None of them is a configuration value, so moving one changes neither the configuration
//! hash nor a byte of the output. A flag wins over its environment variable, which wins
//! over the default, and every relative path is made absolute against the working
//! directory once, at startup ([`apply`]), so the engine and the workers it starts agree on
//! one directory whatever directory each of them runs in.
//!
//! A directory counts as writable when a file can be created in it AND renamed there
//! ([`probe_writable`]): the caches publish an entry by renaming it into place, and a
//! cloud-synced folder or a network share can accept the write and refuse the rename.
//! [`prepare_output`] runs that probe at the start of `run` and `run-experiment`, before
//! any compute, so an unwritable results folder fails with a sentence naming the fix
//! instead of a stage error minutes later; an unwritable cache root switches the default
//! caches off for the process with one warning, because a run without its cache builds
//! what it would have reused and loses nothing else.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::Result;
use tracing::{info, warn};

/// Which step of [`probe_writable`] failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeStep {
    /// The directory could not be created (or a file stands where it should be).
    Create,
    /// A file could not be written in it.
    Write,
    /// A file written in it could not be renamed there.
    Rename,
}

/// Why a directory is not writable.
#[derive(Debug)]
pub struct ProbeFailure {
    pub dir: PathBuf,
    pub step: ProbeStep,
    pub error: std::io::Error,
}

impl std::fmt::Display for ProbeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.dir.display();
        let e = &self.error;
        match self.step {
            ProbeStep::Create => write!(f, "cannot create the directory {d}: {e}"),
            ProbeStep::Write => write!(f, "cannot write a file in {d}: {e}"),
            ProbeStep::Rename => write!(
                f,
                "can write in {d} but cannot rename a file there: {e} (a cloud-synced \
                 folder or a network share can refuse renames)"
            ),
        }
    }
}

impl std::error::Error for ProbeFailure {}

/// Create `dir` if needed and check that a file can be written in it and renamed there.
/// The probe file is removed again; nothing else in the directory is touched.
pub fn probe_writable(dir: &Path) -> std::result::Result<(), ProbeFailure> {
    let fail = |step, error| ProbeFailure {
        dir: dir.to_path_buf(),
        step,
        error,
    };
    std::fs::create_dir_all(dir).map_err(|e| fail(ProbeStep::Create, e))?;
    if !dir.is_dir() {
        return Err(fail(
            ProbeStep::Create,
            std::io::Error::other("it exists and is not a directory"),
        ));
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stem = format!(".mumdia-write-probe-{}-{nanos}", std::process::id());
    let written = dir.join(format!("{stem}.tmp"));
    let renamed = dir.join(format!("{stem}.done"));
    std::fs::write(&written, b"mumdia").map_err(|e| {
        let _ = std::fs::remove_file(&written);
        fail(ProbeStep::Write, e)
    })?;
    if let Err(e) = std::fs::rename(&written, &renamed) {
        let _ = std::fs::remove_file(&written);
        return Err(fail(ProbeStep::Rename, e));
    }
    let _ = std::fs::remove_file(&renamed);
    Ok(())
}

/// `dir`, or its nearest ancestor that exists, or `None` when no part of it does.
pub fn nearest_existing(dir: &Path) -> Option<PathBuf> {
    let mut probe = dir.to_path_buf();
    while !probe.exists() {
        probe = probe.parent()?.to_path_buf();
        if probe.as_os_str().is_empty() {
            return None;
        }
    }
    Some(probe)
}

/// What [`check`] found about one directory, without creating it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct DirCheck {
    pub path: String,
    pub exists: bool,
    /// Whether a file can be written and renamed in it or, when it does not exist yet, in
    /// its nearest existing ancestor (where it would be created).
    pub writable: bool,
    /// The failed probe, when it is not writable.
    pub error: Option<String>,
    /// Free bytes on its volume, when that can be determined.
    pub free_bytes: Option<u64>,
    /// Location warnings ([`location_notes`]): a cloud-synced folder, a network share.
    pub notes: Vec<String>,
}

/// Check `dir` for `doctor`: writable (probed in the directory, or in its nearest existing
/// ancestor when it does not exist, so nothing is created), free space and location.
pub fn check(dir: &Path) -> DirCheck {
    let exists = dir.is_dir();
    let probed = if exists {
        probe_writable(dir).map_err(|f| f.to_string())
    } else {
        match nearest_existing(dir) {
            Some(base) if base.is_dir() => probe_writable(&base)
                .map_err(|f| format!("{} does not exist, and {f}", dir.display())),
            Some(base) => Err(format!(
                "{} is a file, so {} cannot be created",
                base.display(),
                dir.display()
            )),
            None => Err(format!("no part of {} exists", dir.display())),
        }
    };
    DirCheck {
        path: dir.display().to_string(),
        exists,
        writable: probed.is_ok(),
        error: probed.err(),
        free_bytes: free_space(dir),
        notes: location_notes(dir),
    }
}

/// Free space in bytes on the file system that holds `dir` (or its nearest existing
/// ancestor), or `None` when it cannot be determined. The one free-space function of the
/// engine: the library cache's store, the rescore sidecar space check and `doctor` all
/// ask it.
///
/// The standard library has no free-space call and the workspace forbids `unsafe`, so
/// this asks the platform's own tool: POSIX `df -Pk` on Unix and macOS, Windows PowerShell
/// on Windows (.NET `DriveInfo` for a drive letter, the Scripting runtime's `GetDrive` for
/// a `\\server\share` path, which `DriveInfo` refuses), the path passed in the child's
/// environment so no quoting can go wrong. A tool that is missing or refuses gives `None`,
/// and a caller then proceeds as it would without the check.
pub fn free_space(dir: &Path) -> Option<u64> {
    let probe = nearest_existing(dir)?;
    let out = if cfg!(windows) {
        std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$p = [System.IO.Path]::GetFullPath($env:MUMDIA_FREE_SPACE_PATH); \
                 $r = [System.IO.Path]::GetPathRoot($p); \
                 if ($r.StartsWith('\\\\')) { \
                 [uint64](New-Object -ComObject Scripting.FileSystemObject).GetDrive($r.TrimEnd('\\')).AvailableSpace \
                 } else { [uint64]([System.IO.DriveInfo]::new($r).AvailableFreeSpace) }",
            ])
            .env("MUMDIA_FREE_SPACE_PATH", plain_windows_path(&probe))
            .output()
            .ok()?
    } else {
        std::process::Command::new("df")
            .arg("-Pk")
            .arg("--")
            .arg(&probe)
            .output()
            .ok()?
    };
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    if cfg!(windows) {
        return text.trim().parse().ok();
    }
    parse_df_available(&text)
}

/// A Windows path without the verbatim prefix `canonicalize` adds (`\\?\C:\x` is `C:\x`,
/// `\\?\UNC\server\share` is `\\server\share`), which .NET's path functions do not read as
/// a drive or a share.
fn plain_windows_path(p: &Path) -> String {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix("\\\\?\\UNC\\") {
        format!("\\\\{rest}")
    } else if let Some(rest) = s.strip_prefix("\\\\?\\") {
        rest.to_string()
    } else {
        s.into_owned()
    }
}

/// The available bytes of `df -Pk` output: the field before the capacity percentage on
/// the last line, so a file-system or mount name with spaces in it cannot shift it.
pub(crate) fn parse_df_available(text: &str) -> Option<u64> {
    let line = text.lines().rev().find(|l| !l.trim().is_empty())?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    let pct = fields.iter().rposition(|f| f.ends_with('%'))?;
    let kib: u64 = fields.get(pct.checked_sub(1)?)?.parse().ok()?;
    Some(kib * 1024)
}

/// Whether `path` names a network share: `\\server\share` (or the verbatim
/// `\\?\UNC\server\share`) and `//server/share`.
pub fn is_network_path(path: &Path) -> bool {
    let s = path.to_string_lossy();
    if s.starts_with("\\\\?\\UNC\\") {
        return true;
    }
    if s.starts_with("\\\\?\\") || s.starts_with("\\\\.\\") {
        return false;
    }
    s.starts_with("\\\\") || s.starts_with("//")
}

/// Whether `path` lies in a OneDrive folder: under one of the folders the OneDrive client
/// publishes in `OneDrive`, `OneDriveCommercial` or `OneDriveConsumer`, or below a
/// directory whose name starts with `OneDrive` (`OneDrive`, `OneDrive - <organisation>`).
fn in_onedrive_with(path: &Path, roots: &[String]) -> bool {
    let lower = |s: &str| s.replace('\\', "/").trim_end_matches('/').to_lowercase();
    let p = lower(&path.to_string_lossy());
    if roots
        .iter()
        .map(|r| lower(r))
        .any(|r| !r.is_empty() && (p == r || p.starts_with(&format!("{r}/"))))
    {
        return true;
    }
    // By the text, not `components()`: on Windows the share of `//server/OneDrive`
    // is part of the prefix component.
    p.split('/').any(|seg| seg.starts_with("onedrive"))
}

/// The OneDrive folders of this account, from the variables the OneDrive client sets.
fn onedrive_roots() -> Vec<String> {
    [
        std::env::var("OneDrive").ok(),
        std::env::var("OneDriveCommercial").ok(),
        std::env::var("OneDriveConsumer").ok(),
    ]
    .into_iter()
    .flatten()
    .filter(|v| !v.trim().is_empty())
    .collect()
}

/// Location warnings for a directory MuMDIA writes to: a OneDrive folder, whose sync
/// client can lock a file while it is written, holds files the caches memory-map and
/// rename, and uploads every intermediate; and a network share, where renames and
/// memory-mapped reads are slow and can fail. Neither is refused: both work for many
/// people, and the person who chose the folder may have no other.
pub fn location_notes(path: &Path) -> Vec<String> {
    location_notes_with(path, &onedrive_roots())
}

fn location_notes_with(path: &Path, onedrive: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    if is_network_path(path) {
        out.push(format!(
            "{} is on a network share: renames and memory-mapped reads there are slow and \
             can fail; a local disk is faster and safer",
            path.display()
        ));
    }
    if in_onedrive_with(path, onedrive) {
        out.push(format!(
            "{} is inside a OneDrive folder: the sync client can lock files while they are \
             written and renamed and uploads every intermediate; a local folder outside \
             OneDrive is faster and safer",
            path.display()
        ));
    }
    out
}

/// `p` made absolute against `cwd`, with `.` components dropped. `..` is kept: through a
/// symbolic link it does not mean the lexical parent.
fn absolute_in(p: &Path, cwd: &Path) -> PathBuf {
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    };
    joined
        .components()
        .filter(|c| !matches!(c, std::path::Component::CurDir))
        .collect()
}

/// The command-line flags that name a storage location (`--cache-dir`, `--temp-dir`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StorageFlags {
    pub cache_dir: Option<String>,
    pub temp_dir: Option<String>,
}

/// The environment [`apply`] establishes, decided from the flags, the environment and the
/// working directory without touching the process.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Decision {
    /// The value `MUMDIA_CACHE_DIR` is set to, when it changes: the flag's, or a relative
    /// path made absolute. An off word (`off`, `0`, ...) passes through as given.
    pub cache_dir: Option<String>,
    /// The temporary directory, when one is chosen, and the flag or variable it came from.
    pub temp_dir: Option<(PathBuf, &'static str)>,
    /// Whether `MPLCONFIGDIR` is set to `<temp>/matplotlib` (it is left alone when set).
    pub mplconfigdir: bool,
    /// The value `MUMDIA_MODEL_DIR` is set to, when a relative one is made absolute.
    pub model_dir: Option<String>,
}

/// A value that is set and not blank.
fn given(v: Option<String>) -> Option<String> {
    v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Decide what [`apply`] sets: a flag wins over its variable, every path is made absolute
/// against `cwd`, and an off word for the cache is passed through.
pub fn decide(flags: &StorageFlags, env: &dyn Fn(&str) -> Option<String>, cwd: &Path) -> Decision {
    let mut d = Decision::default();
    let env_cache = given(env("MUMDIA_CACHE_DIR"));
    if let Some(v) = given(flags.cache_dir.clone()).or_else(|| env_cache.clone()) {
        let value = if crate::cache::is_off_word(&v) {
            v
        } else {
            absolute_in(Path::new(&v), cwd)
                .to_string_lossy()
                .into_owned()
        };
        if Some(&value) != env_cache.as_ref() {
            d.cache_dir = Some(value);
        }
    }
    let temp = match given(flags.temp_dir.clone()) {
        Some(v) => Some((v, "--temp-dir")),
        None => given(env("MUMDIA_TEMP_DIR")).map(|v| (v, "MUMDIA_TEMP_DIR")),
    };
    if let Some((v, source)) = temp {
        d.temp_dir = Some((absolute_in(Path::new(&v), cwd), source));
        d.mplconfigdir = given(env("MPLCONFIGDIR")).is_none();
    }
    if let Some(v) = given(env("MUMDIA_MODEL_DIR")) {
        let abs = absolute_in(Path::new(&v), cwd)
            .to_string_lossy()
            .into_owned();
        if abs != v {
            d.model_dir = Some(abs);
        }
    }
    d
}

/// Establish the storage locations for this process and every process it starts, from
/// `--cache-dir` / `--temp-dir` and `MUMDIA_CACHE_DIR` / `MUMDIA_TEMP_DIR` /
/// `MUMDIA_MODEL_DIR`. Runs once at startup, before any thread pool exists, because it
/// sets environment variables.
///
/// A temporary directory that was asked for must be usable: it is created if need be, and
/// one that takes no write and rename is an error rather than a silent fallback to the
/// system's, which may be the very place that cannot hold the files. It becomes `TMP`,
/// `TEMP` and `TMPDIR`, which the engine and Python's `tempfile` both read, and
/// `MPLCONFIGDIR` is put under it unless already set.
pub fn apply(flags: &StorageFlags) -> Result<Decision> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let vars = [
        ("MUMDIA_CACHE_DIR", std::env::var("MUMDIA_CACHE_DIR").ok()),
        ("MUMDIA_TEMP_DIR", std::env::var("MUMDIA_TEMP_DIR").ok()),
        ("MUMDIA_MODEL_DIR", std::env::var("MUMDIA_MODEL_DIR").ok()),
        ("MPLCONFIGDIR", std::env::var("MPLCONFIGDIR").ok()),
    ];
    let lookup = |k: &str| {
        vars.iter()
            .find(|(name, _)| *name == k)
            .and_then(|(_, v)| v.clone())
    };
    let d = decide(flags, &lookup, &cwd);
    if let Some((t, source)) = &d.temp_dir {
        probe_writable(t).map_err(|f| {
            anyhow::anyhow!(
                "the temporary directory from {source} is not usable: {f}. Point --temp-dir \
                 or MUMDIA_TEMP_DIR at a folder you can write to, or unset it to use the \
                 system's temporary directory"
            )
        })?;
    }
    if let Some(v) = &d.cache_dir {
        std::env::set_var("MUMDIA_CACHE_DIR", v);
    }
    if let Some((t, _)) = &d.temp_dir {
        std::env::set_var("MUMDIA_TEMP_DIR", t);
        for var in ["TMP", "TEMP", "TMPDIR"] {
            std::env::set_var(var, t);
        }
        if d.mplconfigdir {
            std::env::set_var("MPLCONFIGDIR", t.join("matplotlib"));
        }
    }
    if let Some(v) = &d.model_dir {
        std::env::set_var("MUMDIA_MODEL_DIR", v);
    }
    if d.cache_dir.is_some() || d.temp_dir.is_some() || d.model_dir.is_some() {
        info!(
            cache_dir = d.cache_dir.as_deref().unwrap_or("-"),
            temp_dir = %d.temp_dir.as_ref().map_or("-".into(), |(t, _)| t.display().to_string()),
            model_dir = d.model_dir.as_deref().unwrap_or("-"),
            "storage: locations set for this run and the processes it starts"
        );
    }
    Ok(d)
}

/// The writes a run cannot do without, checked before any compute: the output directory
/// (created here) and an explicitly set `MUMDIA_SIDECAR_DIR` must take a write and a
/// rename, or the run stops with a sentence naming the fix. The default cache root is
/// checked when `cfg` uses a default (`"auto"`) cache, and an unwritable one switches the
/// default caches off for this process ([`crate::cache::disable_default_root`]; the DeepLC
/// worker learns its cache from its arguments, so it is off there too) with one warning: a
/// run without its cache only builds what it would have reused. Called by `run` and `run-experiment`; the cache warning is
/// given once per process.
pub fn prepare_output(out_dir: &str, cfg: &mumdia_core::config::Config) -> Result<()> {
    probe_writable(Path::new(out_dir)).map_err(|f| {
        anyhow::anyhow!(
            "the output directory is not writable: {f}. Choose an --out-dir you can write \
             to (a folder in your own user directory, on a local disk, works)"
        )
    })?;
    for note in location_notes(Path::new(out_dir)) {
        warn!("storage: the output directory {note}");
    }
    if let Some(dir) = std::env::var("MUMDIA_SIDECAR_DIR")
        .ok()
        .filter(|d| !d.trim().is_empty())
    {
        probe_writable(Path::new(&dir)).map_err(|f| {
            anyhow::anyhow!(
                "MUMDIA_SIDECAR_DIR is not writable: {f}. Point it at a folder you can write \
                 to, or unset it to keep the sidecar files under the output directory"
            )
        })?;
    }
    check_cache_root(cfg);
    Ok(())
}

/// The once-per-process cache root check of [`prepare_output`].
fn check_cache_root(cfg: &mumdia_core::config::Config) {
    static CHECKED: OnceLock<()> = OnceLock::new();
    if CHECKED.set(()).is_err() {
        return;
    }
    let uses_auto = [
        crate::cache::library_dir(cfg),
        crate::cache::projection_dir(cfg),
    ]
    .into_iter()
    .flatten()
    .any(|d| d.auto);
    if !uses_auto {
        return;
    }
    let crate::cache::Root::Dir(root, source) = crate::cache::root() else {
        return;
    };
    match probe_writable(&root) {
        Ok(()) => {
            for note in location_notes(&root) {
                warn!("storage: the cache root {note}; --cache-dir or MUMDIA_CACHE_DIR moves it");
            }
        }
        Err(f) => {
            warn!(
                "==== storage: the cache root from {source} is not writable ({f}), so the \
                 default caches are OFF for this run: a FASTA library and the DeepLC \
                 projection are computed again rather than reused. Point --cache-dir or \
                 MUMDIA_CACHE_DIR at a folder you can write to, or set MUMDIA_CACHE_DIR=off \
                 to silence this. ===="
            );
            crate::cache::disable_default_root(format!(
                "the cache root {} is not writable",
                root.display()
            ));
        }
    }
}

/// One storage location as `mumdia doctor` reports it.
#[derive(Debug, serde::Serialize)]
pub struct Location {
    /// `output`, `sidecar`, `cache_root`, `temp`, `models`.
    pub role: &'static str,
    /// `ok`, `warn`, `fail` or `skip`.
    pub status: String,
    /// Where the location came from: a flag, a variable, `default`, or why there is none.
    pub source: String,
    /// The directory, when there is one, with what [`check`] found.
    pub dir: Option<DirCheck>,
    /// What else is worth saying (where the default sits, what a failure means).
    pub detail: Option<String>,
}

/// The `storage` section of `mumdia doctor`.
#[derive(Debug, serde::Serialize)]
pub struct StorageReport {
    /// False when a location the run must write to cannot be written: the output
    /// directory (with `--out-dir`), an explicit `MUMDIA_SIDECAR_DIR` or temporary
    /// directory. An unwritable cache root is a warning: the run switches the cache off.
    pub ok: bool,
    pub locations: Vec<Location>,
}

/// Every storage location of this process, checked without creating anything.
/// `out_dir` is `doctor --out-dir`, the results folder a run would use.
pub fn report(out_dir: Option<&str>) -> StorageReport {
    let mut locations = Vec::new();
    let mut ok = true;
    // A location the run must write to: unwritable fails the report, a location note is
    // a warning.
    fn required(role: &'static str, source: &str, dir: &Path, ok: &mut bool) -> Location {
        let c = check(dir);
        let status = if !c.writable {
            *ok = false;
            "fail"
        } else if c.notes.is_empty() {
            "ok"
        } else {
            "warn"
        };
        Location {
            role,
            status: status.into(),
            source: source.into(),
            dir: Some(c),
            detail: None,
        }
    }
    if let Some(o) = out_dir {
        locations.push(required("output", "--out-dir", Path::new(o), &mut ok));
    }
    match std::env::var("MUMDIA_SIDECAR_DIR")
        .ok()
        .filter(|d| !d.trim().is_empty())
    {
        Some(d) => locations.push(required(
            "sidecar",
            "MUMDIA_SIDECAR_DIR",
            Path::new(&d),
            &mut ok,
        )),
        None => locations.push(Location {
            role: "sidecar",
            status: "skip".into(),
            source: "default".into(),
            dir: None,
            detail: Some("<out-dir>/sidecar_work (MUMDIA_SIDECAR_DIR moves it)".into()),
        }),
    }
    match std::env::var("MUMDIA_TEMP_DIR")
        .ok()
        .filter(|d| !d.trim().is_empty())
    {
        Some(t) => locations.push(required("temp", "MUMDIA_TEMP_DIR", Path::new(&t), &mut ok)),
        None => {
            let t = std::env::temp_dir();
            let c = check(&t);
            locations.push(Location {
                role: "temp",
                status: if c.writable { "ok" } else { "warn" }.into(),
                source: "system".into(),
                dir: Some(c),
                detail: Some("--temp-dir or MUMDIA_TEMP_DIR moves it".into()),
            });
        }
    }
    match crate::cache::root() {
        crate::cache::Root::Dir(root, source) => {
            let c = check(&root);
            let status = if !c.writable || !c.notes.is_empty() {
                "warn"
            } else {
                "ok"
            };
            locations.push(Location {
                role: "cache_root",
                status: status.into(),
                source: source.into(),
                detail: (!c.writable).then(|| {
                    "a run switches the default caches off with a warning; --cache-dir or \
                     MUMDIA_CACHE_DIR moves them"
                        .to_string()
                }),
                dir: Some(c),
            });
        }
        crate::cache::Root::Off(why) => locations.push(Location {
            role: "cache_root",
            status: "skip".into(),
            source: why,
            dir: None,
            detail: None,
        }),
    }
    match std::env::var("MUMDIA_MODEL_DIR")
        .ok()
        .filter(|d| !d.trim().is_empty())
    {
        Some(m) => {
            let c = check(Path::new(&m));
            locations.push(Location {
                role: "models",
                status: if !c.writable || !c.notes.is_empty() {
                    "warn"
                } else {
                    "ok"
                }
                .into(),
                source: "MUMDIA_MODEL_DIR".into(),
                detail: Some(
                    "MS2PIP downloads into <dir>/ms2pip, AlphaPeptDeep into <dir>/peptdeep".into(),
                ),
                dir: Some(c),
            });
        }
        None => locations.push(Location {
            role: "models",
            status: "skip".into(),
            source: "default".into(),
            dir: None,
            detail: Some(
                "each predictor's own: ~/.ms2pip (MS2PIP), ~/peptdeep (AlphaPeptDeep); \
                 MUMDIA_MODEL_DIR moves both"
                    .into(),
            ),
        }),
    }
    StorageReport { ok, locations }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia_storage_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn the_probe_creates_writes_renames_and_leaves_nothing_behind() {
        let d = scratch("probe").join("a/b");
        probe_writable(&d).unwrap();
        assert!(d.is_dir());
        assert_eq!(
            std::fs::read_dir(&d).unwrap().count(),
            0,
            "the probe file is removed"
        );
    }

    #[test]
    fn the_probe_names_the_step_that_failed() {
        let base = scratch("probe_file");
        std::fs::create_dir_all(&base).unwrap();
        let file = base.join("not_a_dir");
        std::fs::write(&file, b"x").unwrap();
        // A file where the directory should be, and a directory below a file.
        for target in [file.clone(), file.join("below")] {
            let e = probe_writable(&target).unwrap_err();
            assert_eq!(e.step, ProbeStep::Create, "{e}");
            assert!(e.to_string().contains("cannot create the directory"), "{e}");
        }
        // `check` creates nothing and says the same.
        let c = check(&file.join("below"));
        assert!(!c.writable && !c.exists, "{c:?}");
        assert!(c.error.unwrap().contains("is a file"));
    }

    #[cfg(unix)]
    #[test]
    fn a_read_only_directory_fails_at_the_write() {
        use std::os::unix::fs::PermissionsExt;
        let d = scratch("probe_ro");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o555)).unwrap();
        let r = probe_writable(&d);
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
        // Root writes anywhere; the assertion holds for everyone else.
        if let Err(e) = r {
            assert_eq!(e.step, ProbeStep::Write, "{e}");
        }
    }

    #[test]
    fn check_does_not_create_a_missing_directory() {
        let d = scratch("check_missing").join("x/y");
        let c = check(&d);
        assert!(!c.exists);
        assert!(c.writable, "{c:?}");
        assert!(!d.exists(), "doctor's check must not create the directory");
    }

    #[test]
    fn a_flag_wins_over_its_variable_and_relative_paths_become_absolute() {
        let cwd = if cfg!(windows) {
            PathBuf::from(r"C:\work")
        } else {
            PathBuf::from("/work")
        };
        let flags = |c: Option<&str>, t: Option<&str>| StorageFlags {
            cache_dir: c.map(str::to_string),
            temp_dir: t.map(str::to_string),
        };
        // Nothing given: nothing changes.
        assert_eq!(
            decide(&flags(None, None), &env_of(&[]), &cwd),
            Decision::default()
        );
        // The variable alone, relative: made absolute.
        let d = decide(
            &flags(None, None),
            &env_of(&[("MUMDIA_CACHE_DIR", "./c"), ("MUMDIA_TEMP_DIR", "t")]),
            &cwd,
        );
        assert_eq!(
            d.cache_dir,
            Some(cwd.join("c").to_string_lossy().into_owned())
        );
        assert_eq!(d.temp_dir, Some((cwd.join("t"), "MUMDIA_TEMP_DIR")));
        assert!(d.mplconfigdir);
        // An absolute variable is left as it is.
        let abs = cwd.join("abs").to_string_lossy().into_owned();
        let d = decide(
            &flags(None, None),
            &env_of(&[("MUMDIA_CACHE_DIR", &abs)]),
            &cwd,
        );
        assert_eq!(d.cache_dir, None);
        // The flag wins over the variable.
        let d = decide(
            &flags(Some("fc"), Some("ft")),
            &env_of(&[
                ("MUMDIA_CACHE_DIR", &abs),
                ("MUMDIA_TEMP_DIR", "/elsewhere"),
                ("MPLCONFIGDIR", "/mine"),
            ]),
            &cwd,
        );
        assert_eq!(
            d.cache_dir,
            Some(cwd.join("fc").to_string_lossy().into_owned())
        );
        assert_eq!(d.temp_dir, Some((cwd.join("ft"), "--temp-dir")));
        assert!(!d.mplconfigdir, "a set MPLCONFIGDIR is left alone");
        // An off word passes through, from either.
        let d = decide(&flags(Some("off"), None), &env_of(&[]), &cwd);
        assert_eq!(d.cache_dir.as_deref(), Some("off"));
        let d = decide(
            &flags(None, None),
            &env_of(&[("MUMDIA_CACHE_DIR", "OFF")]),
            &cwd,
        );
        assert_eq!(d.cache_dir, None);
        // A blank value is unset.
        let d = decide(
            &flags(Some(" "), None),
            &env_of(&[("MUMDIA_TEMP_DIR", "")]),
            &cwd,
        );
        assert_eq!(d, Decision::default());
        // The model directory is only made absolute.
        let d = decide(
            &flags(None, None),
            &env_of(&[("MUMDIA_MODEL_DIR", "m")]),
            &cwd,
        );
        assert_eq!(
            d.model_dir,
            Some(cwd.join("m").to_string_lossy().into_owned())
        );
        let d = decide(
            &flags(None, None),
            &env_of(&[("MUMDIA_MODEL_DIR", &abs)]),
            &cwd,
        );
        assert_eq!(d.model_dir, None);
    }

    #[test]
    fn network_and_onedrive_locations_are_named() {
        assert!(is_network_path(Path::new(r"\\server\share\x")));
        assert!(is_network_path(Path::new(r"\\?\UNC\server\share\x")));
        assert!(is_network_path(Path::new("//server/share/x")));
        assert!(!is_network_path(Path::new(r"\\?\C:\x")));
        assert!(!is_network_path(Path::new("/home/u/x")));
        assert!(!is_network_path(Path::new(r"C:\x")));

        let roots = vec![r"C:\Users\u\Company Files".to_string()];
        assert!(in_onedrive_with(
            Path::new(r"C:\Users\u\Company Files\r"),
            &roots
        ));
        assert!(!in_onedrive_with(
            Path::new(r"C:\Users\u\Company Filesx"),
            &roots
        ));
        assert!(in_onedrive_with(
            Path::new("/Users/u/OneDrive - UGent/results"),
            &[]
        ));
        assert!(!in_onedrive_with(Path::new("/data/results"), &[]));

        let notes = location_notes_with(Path::new("//srv/OneDrive/r"), &[]);
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(notes[0].contains("network share") && notes[1].contains("OneDrive"));
        assert!(location_notes_with(Path::new("/data/r"), &[]).is_empty());
    }

    #[test]
    fn the_verbatim_prefix_is_removed_for_the_free_space_tool() {
        assert_eq!(plain_windows_path(Path::new(r"\\?\C:\x")), r"C:\x");
        assert_eq!(
            plain_windows_path(Path::new(r"\\?\UNC\srv\share\x")),
            r"\\srv\share\x"
        );
        assert_eq!(plain_windows_path(Path::new(r"D:\y")), r"D:\y");
    }

    #[test]
    fn the_df_available_column_is_found_by_the_capacity_field() {
        let out = "Filesystem     1024-blocks      Used Available Capacity Mounted on\n\
                   /dev/sdb1       15000000000 1300000000 13000000 10% /public/local\n";
        assert_eq!(parse_df_available(out), Some(13_000_000 * 1024));
        // A mount point with a space in it does not move the column.
        let spaced = "Filesystem 1024-blocks Used Available Capacity Mounted on\n\
                      my disk 100 40 60 40% /Volumes/My Disk\n";
        assert_eq!(parse_df_available(spaced), Some(60 * 1024));
        assert_eq!(parse_df_available("garbage"), None);
    }

    #[test]
    fn free_space_answers_for_the_temporary_directory() {
        // Every platform the engine ships on has the tool; a missing one is `None`, which
        // callers treat as "no check".
        let t = std::env::temp_dir().join("mumdia-free-space-probe-does-not-exist/x");
        if let Some(free) = free_space(&t) {
            assert!(free > 0);
        }
    }

    #[test]
    fn the_report_fails_on_an_unwritable_output_directory_only() {
        let base = scratch("report");
        std::fs::create_dir_all(&base).unwrap();
        let file = base.join("file");
        std::fs::write(&file, b"x").unwrap();
        let bad = file.join("out");
        let r = report(Some(&bad.to_string_lossy()));
        assert!(!r.ok);
        let out = r.locations.iter().find(|l| l.role == "output").unwrap();
        assert_eq!(out.status, "fail");
        let good = report(Some(&base.join("new_out").to_string_lossy()));
        assert_eq!(
            good.locations
                .iter()
                .find(|l| l.role == "output")
                .unwrap()
                .status,
            "ok"
        );
        for role in ["sidecar", "temp", "cache_root", "models"] {
            assert!(
                good.locations.iter().any(|l| l.role == role),
                "no {role} location in the report"
            );
        }
    }
}
