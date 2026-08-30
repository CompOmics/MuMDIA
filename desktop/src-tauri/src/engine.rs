//! Locating the engine binary and asking it what it is.
//!
//! The application ships the engine beside itself rather than linking it, so the
//! first question at startup is "where is it, and does it run". Resolution order
//! mirrors `ci/smoke.sh`'s `find_bin` in spirit: an explicit override wins, then the
//! bundled copy, then whatever is on PATH.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

/// Name of the engine executable on this platform.
pub const EXE: &str = if cfg!(windows) {
    "mumdia.exe"
} else {
    "mumdia"
};

/// What the application knows about the engine it will run.
#[derive(Serialize, Clone, Debug)]
pub struct Info {
    /// Absolute path to the binary that will be spawned.
    pub path: String,
    /// The `--version` line, verbatim.
    pub version: String,
    /// How the path was found, for the "why is it using that one" question.
    pub source: &'static str,
}

/// Every place the engine may live, in the order they are tried.
///
/// `MUMDIA_BIN` first so a developer can point the application at a freshly built
/// engine without reinstalling it. Then the two bundled locations: Tauri puts
/// declared resources next to the executable, and the `binaries/` subdirectory is
/// where the release workflow stages the engine. PATH last, because a copy the user
/// installed separately is the least likely to be the one we were tested against.
fn candidates() -> Vec<(PathBuf, &'static str)> {
    let mut out = Vec::new();

    if let Some(p) = std::env::var_os("MUMDIA_BIN") {
        out.push((PathBuf::from(p), "MUMDIA_BIN"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push((dir.join(EXE), "bundled"));
            out.push((dir.join("binaries").join(EXE), "bundled"));
            // `cargo run` from src-tauri puts the app in target/debug, with the
            // repository three levels up. Convenient during development and
            // harmless in a release, where the path simply does not exist.
            out.push((
                dir.join("../../../../rust/mumdia/target/release").join(EXE),
                "repository build",
            ));
        }
    }
    out
}

/// Resolve the engine, or explain every place that was tried.
pub fn resolve() -> Result<(PathBuf, &'static str), String> {
    let cands = candidates();
    for (path, source) in &cands {
        if path.is_file() {
            let abs = path.canonicalize().unwrap_or_else(|_| path.clone());
            return Ok((strip_unc(abs), source));
        }
    }
    if let Ok(path) = which_on_path() {
        return Ok((path, "PATH"));
    }
    let tried: Vec<String> = cands
        .iter()
        .map(|(p, s)| format!("  {} ({s})", p.display()))
        .collect();
    Err(format!(
        "could not find the {EXE} engine. Tried:\n{}\n  and every directory on PATH.\n\
         Set MUMDIA_BIN to the engine binary, or reinstall the application.",
        tried.join("\n")
    ))
}

/// `Path::canonicalize` returns a `\\?\C:\...` extended-length path on Windows, which
/// many programs render but few accept back. Strip the prefix so the path we display
/// is the path a user could paste into a terminal.
fn strip_unc(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy().to_string();
    match s.strip_prefix(r"\\?\") {
        Some(rest) => PathBuf::from(rest),
        None => p,
    }
}

fn which_on_path() -> Result<PathBuf, ()> {
    let path = std::env::var_os("PATH").ok_or(())?;
    for dir in std::env::split_paths(&path) {
        let cand = dir.join(EXE);
        if cand.is_file() {
            return Ok(cand);
        }
    }
    Err(())
}

/// Resolve the engine and ask it for its version.
///
/// Running it is the point: a binary that exists but cannot execute (wrong
/// architecture, missing library, quarantined by antivirus) fails here, at startup,
/// rather than at the moment someone starts an hour-long search.
pub fn info() -> Result<Info, String> {
    let (path, source) = resolve()?;
    let out = command(&path)
        .arg("--version")
        .output()
        .map_err(|e| format!("found {} but could not run it: {e}", path.display()))?;
    if !out.status.success() {
        return Err(format!(
            "{} --version exited with {}",
            path.display(),
            out.status
        ));
    }
    Ok(Info {
        path: path.display().to_string(),
        version: String::from_utf8_lossy(&out.stdout).trim().to_string(),
        source,
    })
}

/// A `Command` for the engine with the console window suppressed on Windows.
///
/// Without this every engine invocation flashes a console window, including the
/// version probe at startup.
pub fn command(path: &Path) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(path);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}
