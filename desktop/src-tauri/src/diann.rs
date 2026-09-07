//! Obtaining, locating, and driving DIA-NN.
//!
//! # Why this is not an install like the Python components
//!
//! `components.rs` installs DeepLC, torch and mokapot outright: they are
//! open-source packages on PyPI under licences that permit redistribution, so `uv`
//! can fetch and place them however is convenient. None of that is true of DIA-NN,
//! and the difference decides the shape of this module.
//!
//! DIA-NN is closed source. Releases from 1.9 onward forbid redistribution, which
//! is why it is absent from Bioconda; from 1.9.2 it requires a licence file to
//! activate; and 2.x is split into a paid Enterprise edition and an Academia
//! edition restricted to non-profit academic research. 1.8.1, the last release
//! predating activation, is commonly described as "the redistributable one", and
//! that is wrong: its own LICENSE.txt bars derivative works and bars renting,
//! leasing, lending and sublicensing, permitting only a one-time permanent transfer
//! of all rights. The claim comes from community container images, not from the
//! licence text.
//!
//! So MuMDIA never ships DIA-NN bytes, never mirrors them, and never proxies them.
//! Two things it does do, and both avoid distributing anything:
//!
//!   - **Locate** a DIA-NN the user installed and licensed themselves, and drive
//!     it. Running a program already on the machine redistributes nothing.
//!   - **Fetch** the pinned 1.8.1 asset from the vendor's own release URL onto the
//!     user's machine. The vendor distributes; the user obtains; the application
//!     automates a download the user could perform by hand.
//!
//! That second one is a deliberate, narrow judgement and not a general licence to
//! bundle. If it is ever widened -- a mirror, a vendored copy, a "latest" lookup,
//! another host -- the reasoning above no longer covers it. `LICENCE_NOTICE` and
//! the tests around `asset()` exist to make that hard to do by accident.
//!
//! # What it produces
//!
//! DIA-NN predicts a spectral library from a FASTA; the two sidecars already in
//! `scripts/` convert it to the MuMDIA schema and add the decoy population. The
//! command line is the one recorded in `README.md`, so what this runs is what a
//! user following the manual instructions would have typed.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

/// Shown before DIA-NN is located or run, and acknowledged once per machine.
///
/// Deliberately states what MuMDIA does *not* do, because the common assumption
/// after seeing the components screen install Python is that this screen installs
/// DIA-NN the same way.
pub const LICENCE_NOTICE: &str = "\
DIA-NN is separate software, by a separate author, under its own licence. MuMDIA \
does not include, modify or redistribute it, and you use it under your own licence.

The DIA-NN \"Academia\" edition is free for non-profit academic research only. \
Commercial and industrial use requires a paid Enterprise licence. If you are not \
certain which applies to you, check before continuing.

If you use the download button, MuMDIA fetches DIA-NN 1.8.1 from the author's own \
release page and checks it against a known checksum. It is not hosted, mirrored or \
altered by MuMDIA, and accepting this notice is not a substitute for DIA-NN's own \
licence terms.

DIA-NN's version affects the library it predicts, and therefore affects your \
results. Record the version you used.";

/// Where a located DIA-NN, and the acknowledgement, are remembered.
fn state_file() -> PathBuf {
    crate::components::data_dir().join("diann.json")
}

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
struct Saved {
    path: Option<String>,
    licence_acknowledged: bool,
}

fn load() -> Saved {
    std::fs::read_to_string(state_file())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

fn store(s: &Saved) -> Result<(), String> {
    let dir = crate::components::data_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let text = serde_json::to_string_pretty(s).map_err(|e| e.to_string())?;
    std::fs::write(state_file(), text)
        .map_err(|e| format!("cannot write {}: {e}", state_file().display()))
}

/// Whether DIA-NN was found, where, and whether it runs.
#[derive(Serialize, Clone, Debug, Default)]
pub struct Status {
    pub path: Option<String>,
    /// `configured` | `environment` | `path` | `installed` -- how it was found,
    /// so the interface can say why it is using this copy and not another.
    pub source: Option<String>,
    /// The version banner, when the binary printed one that could be parsed.
    pub version: Option<String>,
    /// True only when the binary was executed successfully. A file that exists but
    /// cannot run (wrong architecture, missing licence file) must not read as ready.
    pub runs: bool,
    pub licence_acknowledged: bool,
    /// False for DIA-NN 1.8.x, which cannot write the Parquet spectral library the
    /// library build needs. Reported so the interface can say so before a
    /// whole-proteome prediction rather than after it.
    pub writes_parquet: bool,
    pub error: Option<String>,
}

/// Directories a DIA-NN installer is known to use, searched one level deep so a
/// version-numbered subdirectory is found without a recursive walk of the disk.
fn install_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        for var in ["ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"] {
            if let Some(p) = std::env::var_os(var) {
                roots.push(PathBuf::from(p).join("DIA-NN"));
            }
        }
        roots.push(PathBuf::from("C:\\DIA-NN"));
    } else {
        roots.push(PathBuf::from("/usr/diann"));
        roots.push(PathBuf::from("/opt/diann"));
        roots.push(PathBuf::from("/opt/DIA-NN"));
        if let Some(home) = std::env::var_os("HOME") {
            roots.push(PathBuf::from(&home).join("diann"));
            roots.push(PathBuf::from(&home).join("DIA-NN"));
        }
    }
    roots
}

/// Executable names DIA-NN has shipped under. 1.8.x used `DiaNN.exe` and
/// `diann-linux`; 2.x uses `diann`. The Linux 1.8.1 tarball names its binary after
/// the version, so a managed install is found by the same search as any other.
fn exe_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["diann.exe", "DiaNN.exe", "diann-cli.exe"]
    } else {
        &["diann", "diann-linux", "diann-1.8.1"]
    }
}

/// Spawn DIA-NN with its own directory on the shared-library path.
///
/// The Linux 1.8.1 tarball is flat: `diann-1.8.1` sits beside `libtorch_cpu.so`,
/// `libc10.so` and the rest, with no enclosing directory. Extracted anywhere other
/// than a location already on the loader path, it cannot find its own libraries, so
/// the directory is added here. Harmless for a `.deb` install, whose libraries are
/// already resolvable, and for Windows, which does not read this variable.
fn diann_command(exe: &Path) -> std::process::Command {
    // Mutated only under `cfg(unix)`, exactly as `engine::command` is mutated only
    // under `cfg(windows)`.
    #[allow(unused_mut)]
    let mut cmd = crate::engine::command(exe);
    #[cfg(unix)]
    if let Some(dir) = exe.parent() {
        let joined = match std::env::var_os("LD_LIBRARY_PATH") {
            Some(existing) => {
                let mut paths = vec![dir.to_path_buf()];
                paths.extend(std::env::split_paths(&existing));
                std::env::join_paths(paths).ok()
            }
            None => std::env::join_paths([dir.to_path_buf()]).ok(),
        };
        if let Some(v) = joined {
            cmd.env("LD_LIBRARY_PATH", v);
        }
    }
    cmd
}

fn first_exe_in(dir: &Path) -> Option<PathBuf> {
    for name in exe_names() {
        let c = dir.join(name);
        if c.is_file() {
            return Some(c);
        }
    }
    None
}

/// Candidate binaries, most deliberate choice first: what the user pointed at,
/// then what they set in the environment, then PATH, then the installer defaults.
fn candidates(saved: &Saved) -> Vec<(String, PathBuf)> {
    let mut out: Vec<(String, PathBuf)> = Vec::new();

    if let Some(p) = saved.path.as_ref() {
        out.push(("configured".into(), PathBuf::from(p)));
    }
    if let Some(p) = std::env::var_os("MUMDIA_DIANN") {
        out.push(("environment".into(), PathBuf::from(p)));
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            if let Some(exe) = first_exe_in(&dir) {
                out.push(("path".into(), exe));
                break;
            }
        }
    }
    for root in install_roots() {
        if let Some(exe) = first_exe_in(&root) {
            out.push(("installed".into(), exe));
            continue;
        }
        // One level deeper, for the version-numbered directory the Windows
        // installer creates (`C:\DIA-NN\2.0\diann.exe`). Sorted, so the choice
        // between several installed versions is deterministic rather than
        // whatever order the filesystem returns.
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        let mut subs: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        subs.sort();
        for sub in subs.iter().rev() {
            if let Some(exe) = first_exe_in(sub) {
                out.push(("installed".into(), exe));
                break;
            }
        }
    }
    out
}

/// Run the binary and read the version out of its banner.
///
/// DIA-NN has no `--version`; it prints `DIA-NN 1.8.1 (Data-Independent
/// Acquisition by Neural Networks)` on startup and then exits complaining that it
/// has nothing to do. A non-zero exit is therefore expected and is not a failure:
/// the question this answers is only "does this file execute on this machine",
/// which catches a Linux binary on Windows, a truncated download, and a 1.9.2+
/// build with no licence file, all of which otherwise surface hours later.
/// How long a probe may take before it is assumed not to be a console program.
///
/// DIA-NN prints its banner and exits within a second. Ten seconds is slow enough
/// for a cold start off a network share and short enough that a mistake does not
/// look like a frozen application.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// A DIA-NN installation contains a graphical front end as well as the
/// command-line tool, and pointing at the wrong one is the obvious mistake.
///
/// DIA-NN's own documentation describes the pair: `diann.exe` is the command-line
/// tool, and `DIA-NN.exe` is the GUI that invokes it. Only the former is usable
/// here. Recognised by name and refused before it is executed, because executing it
/// opens a window on the user's desktop and never exits.
fn is_graphical_frontend(exe: &Path) -> bool {
    let name = exe
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    // `diann.exe` and `dia-nn.exe` differ only by the hyphen, so compare exactly
    // rather than by a substring.
    name == "dia-nn.exe" || name == "dia-nn"
}

/// Run the binary and read the version out of its banner.
///
/// DIA-NN has no `--version`; it prints `DIA-NN 1.8.1 (Data-Independent
/// Acquisition by Neural Networks)` on startup and then exits complaining that it
/// has nothing to do. A non-zero exit is therefore expected and is not a failure:
/// the question this answers is only "does this file execute on this machine",
/// which catches a Linux binary on Windows, a truncated download, and a 1.9.2+
/// build with no licence file, all of which otherwise surface hours later.
///
/// Bounded by `PROBE_TIMEOUT` and killed on expiry. Without that, pointing this at
/// a graphical program hung the Setup screen for ever: `Command::output` waits for
/// the child to exit, a GUI does not exit, and the Tauri command that called this
/// never returned. `is_graphical_frontend` catches the DIA-NN case by name; the
/// timeout is the backstop for every other program that does not exit on its own.
pub fn probe(exe: &Path) -> (bool, Option<String>, Option<String>) {
    if is_graphical_frontend(exe) {
        return (
            false,
            None,
            Some(format!(
                "{} is DIA-NN's graphical interface, which cannot be driven from here. \
                 Choose diann.exe in the same folder, which is DIA-NN's command-line \
                 tool.",
                exe.file_name().unwrap_or_default().to_string_lossy()
            )),
        );
    }

    let mut cmd = diann_command(exe);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return (
                false,
                None,
                Some(format!("{} did not run: {e}", exe.display())),
            )
        }
    };

    // Read both streams on their own threads. Reading after `wait` would deadlock a
    // child that fills a pipe, and reading before it means the timeout below can
    // still fire while output is arriving.
    let mut readers = Vec::new();
    for stream in [
        child
            .stdout
            .take()
            .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        child
            .stderr
            .take()
            .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
    ]
    .into_iter()
    .flatten()
    {
        readers.push(std::thread::spawn(move || {
            // Bytes, then lossy UTF-8. `read_to_string` RETURNS an error on invalid
            // UTF-8 and leaves the buffer untouched, i.e. empty, so a single OEM or
            // CP1252 byte -- an accented character in a path DIA-NN echoes back --
            // emptied the output and the banner scan then reported a perfectly good
            // DIA-NN as "ran but did not identify itself as DIA-NN", permanently
            // disabling Locate and the library build. `thermo::probe` already used
            // `from_utf8_lossy`; the two probes disagreed.
            let mut bytes = Vec::new();
            let mut r = stream;
            let _ = std::io::Read::read_to_end(&mut r, &mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        }));
    }

    let started = std::time::Instant::now();
    let timed_out = loop {
        match child.try_wait() {
            Ok(Some(_)) => break false,
            Ok(None) => {}
            // The child is gone in a way we cannot interrogate; treat it as finished
            // and let the banner check decide.
            Err(_) => break false,
        }
        if started.elapsed() >= PROBE_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            break true;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };

    let mut text = String::new();
    for r in readers {
        if let Ok(part) = r.join() {
            text.push_str(&part);
        }
    }

    if timed_out {
        return (
            false,
            None,
            Some(format!(
                "{} did not exit within {} seconds, so it is not a command-line \
                 program. If this is DIA-NN's graphical interface, choose diann.exe \
                 instead.",
                exe.file_name().unwrap_or_default().to_string_lossy(),
                PROBE_TIMEOUT.as_secs()
            )),
        );
    }

    let version = text
        .lines()
        .find(|l| l.trim_start().starts_with("DIA-NN"))
        .map(|l| l.trim().to_string());

    // The banner is the evidence that this is DIA-NN and not some other binary
    // that happens to be called `diann`.
    if version.is_none() {
        return (
            false,
            None,
            Some(format!(
                "{} ran but did not identify itself as DIA-NN",
                exe.display()
            )),
        );
    }
    (true, version, None)
}

/// Find DIA-NN, if it is there.
pub fn detect() -> Status {
    let saved = load();
    let mut status = Status {
        licence_acknowledged: saved.licence_acknowledged,
        ..Default::default()
    };

    let cands = candidates(&saved);
    if cands.is_empty() {
        return status;
    }

    // Report the first candidate that actually runs. A configured path that has
    // stopped working is reported as an error rather than silently falling through
    // to a different copy, because a silent switch would change results.
    for (source, exe) in &cands {
        if !exe.is_file() {
            if source == "configured" {
                status.path = Some(exe.display().to_string());
                status.source = Some(source.clone());
                status.error = Some(format!("{} is no longer there", exe.display()));
                return status;
            }
            continue;
        }
        let (runs, version, err) = probe(exe);
        if runs {
            status.path = Some(exe.display().to_string());
            status.source = Some(source.clone());
            status.writes_parquet = version
                .as_deref()
                .map(writes_parquet_libraries)
                .unwrap_or(true);
            status.version = version;
            status.runs = true;
            return status;
        }
        if source == "configured" {
            status.path = Some(exe.display().to_string());
            status.source = Some(source.clone());
            status.error = err;
            return status;
        }
    }

    status.error = Some("no working DIA-NN was found".into());
    status
}

/// Can this DIA-NN write a Parquet spectral library?
///
/// 1.8.x cannot, and that matters more than a version note: `build` needs a Parquet
/// library, the re-export step is the only way to get one from a predicted
/// `.speclib`, and 1.8.x will not perform it. Without this check the interface
/// offered the download, predicted a whole proteome, ran the re-export, and only then
/// failed -- after the expensive step, on a path it had recommended.
///
/// Parsed from the banner (`DIA-NN 1.8.1 (...)`). An unparseable banner is treated as
/// capable: refusing on a version we cannot read would block working installations,
/// and the build still reports the real failure if it turns out not to be.
pub fn writes_parquet_libraries(version: &str) -> bool {
    let Some(rest) = version.split_whitespace().nth(1) else {
        return true;
    };
    let mut parts = rest.split('.');
    let (Some(major), minor) = (parts.next(), parts.next()) else {
        return true;
    };
    let Ok(major) = major.trim().parse::<u32>() else {
        return true;
    };
    if major >= 2 {
        return true;
    }
    // 1.9 introduced Parquet library output; 1.8.x did not have it.
    minor
        .and_then(|m| m.trim().parse::<u32>().ok())
        .map(|m| m >= 9)
        .unwrap_or(true)
}

/// Remember a DIA-NN the user pointed at. `None` forgets it and returns to search.
pub fn set_path(path: Option<&str>) -> Result<Status, String> {
    let mut saved = load();
    saved.path = match path {
        Some(p) if !p.trim().is_empty() => {
            let p = Path::new(p.trim());
            if !p.is_file() {
                return Err(format!("{} is not a file", p.display()));
            }
            Some(p.display().to_string())
        }
        _ => None,
    };
    store(&saved)?;
    Ok(detect())
}

/// Record that the licence notice was shown and accepted.
pub fn acknowledge_licence(accepted: bool) -> Result<Status, String> {
    let mut saved = load();
    saved.licence_acknowledged = accepted;
    store(&saved)?;
    Ok(detect())
}

// ── obtaining 1.8.1 ─────────────────────────────────────────────────────────
//
// The only version this will fetch, and the reason is the licence rather than a
// preference for old software.
//
// 1.8.1 is the last release predating the activation requirement introduced in
// 1.9.2, and the last that a user can obtain and run without a licence file. It is
// not, however, freely redistributable: its own LICENSE.txt bars derivative works
// and bars renting, leasing, lending and sublicensing, permitting only a one-time
// permanent transfer of all rights. The widespread claim that "1.8.1 is the
// redistributable one" comes from community container images, not from that text.
//
// So MuMDIA does not ship these bytes and does not mirror them. What this does is
// fetch the vendor's own release asset, from the vendor's own URL, onto the user's
// machine. Nothing is redistributed: the vendor distributes, the user obtains, and
// the application automates a download the user could perform by hand. That
// distinction is the whole basis on which this is offered, so do not "simplify" it
// later by vendoring the asset or proxying it through another host.
//
// Two further rules hold it together:
//
//   - the URL and digest are PINNED. Never resolve "latest": a newer release is
//     one this reasoning has not been applied to, and it would also silently change
//     the predicted library.
//   - on Windows the vendor's own installer is launched rather than silently
//     extracted, because that installer presents the vendor's own licence terms and
//     takes the user's acceptance directly. Ours is not a substitute for theirs.

/// What to do with the asset once it is on disk.
#[derive(Clone, Copy, PartialEq)]
enum AssetKind {
    /// The vendor's installer. Launched interactively so its own licence screen is
    /// the one the user accepts.
    Installer,
    /// A flat tarball, extracted into the managed directory. Needs no root.
    Tarball,
}

struct Asset {
    url: &'static str,
    /// Verified before the bytes are executed or extracted.
    sha256: &'static str,
    size: u64,
    kind: AssetKind,
    file_name: &'static str,
}

/// The pinned 1.8.1 asset for this platform, if there is one.
///
/// Digests taken from the release assets on 2026-08-30. A mismatch is a hard
/// failure: these bytes get executed.
fn asset() -> Option<Asset> {
    if cfg!(windows) {
        Some(Asset {
            url:
                "https://github.com/vdemichev/DiaNN/releases/download/1.8.1/DIA-NN.1_8_1.Setup.exe",
            sha256: "83c788d532b5b173c0945467fd18ff12e2fc3d337d1ca328ffe20016d1fc0fb3",
            size: 137_983_408,
            kind: AssetKind::Installer,
            file_name: "DIA-NN.1_8_1.Setup.exe",
        })
    } else if cfg!(target_os = "linux") {
        Some(Asset {
            url: "https://github.com/vdemichev/DiaNN/releases/download/1.8.1/diann_1.8.1.tar.gz",
            sha256: "fb239a1191ae9f3aa497d4e933ef9435bd9fd4795222f100530d1e22e550bb2c",
            size: 142_368_752,
            kind: AssetKind::Tarball,
            file_name: "diann_1.8.1.tar.gz",
        })
    } else {
        None
    }
}

/// Progress of an install, polled by the interface.
#[derive(Serialize, Clone, Debug, Default)]
pub struct InstallState {
    /// `idle` | `running` | `done` | `failed` | `handoff`
    ///
    /// `handoff` is the Windows outcome: the vendor's installer was launched and
    /// the rest is between the user and it, so there is nothing further to report.
    pub status: String,
    pub step: String,
    /// 0-100 for the download, which is the only part with a known length.
    pub percent: u8,
    pub log: Vec<String>,
    pub error: Option<String>,
}

#[derive(Default)]
pub struct Installer {
    state: Mutex<InstallState>,
}

impl Installer {
    pub fn snapshot(&self) -> InstallState {
        self.state
            .lock()
            .map(|s| s.clone())
            .unwrap_or_else(|_| InstallState {
                status: "failed".into(),
                error: Some("internal state is poisoned".into()),
                ..Default::default()
            })
    }

    fn log(&self, line: String) {
        if let Ok(mut s) = self.state.lock() {
            s.log.push(line);
            if s.log.len() > 500 {
                let drop = s.log.len() - 500;
                s.log.drain(0..drop);
            }
        }
    }

    fn set(&self, status: &str, step: &str) {
        if let Ok(mut s) = self.state.lock() {
            s.status = status.into();
            s.step = step.into();
        }
    }

    fn fail(&self, msg: String) {
        if let Ok(mut s) = self.state.lock() {
            s.status = "failed".into();
            s.error = Some(msg);
        }
    }
}

/// Where a managed 1.8.1 is extracted. Per-user, so no administrator rights.
fn managed_dir() -> PathBuf {
    crate::components::data_dir().join("diann-1.8.1")
}

/// Whether this platform can be offered the download at all, and what it costs.
pub fn offer() -> serde_json::Value {
    match asset() {
        Some(a) => serde_json::json!({
            "available": true,
            "version": "1.8.1",
            "url": a.url,
            "download_bytes": a.size,
            // The Linux tarball unpacks to about 490 MB, nearly all of it
            // libtorch_cpu.so, and someone on a small volume should be told before
            // a 142 MB download rather than after it.
            "disk_bytes": if a.kind == AssetKind::Tarball { 500_000_000u64 } else { a.size * 2 },
            "hands_off": a.kind == AssetKind::Tarball,
        }),
        None => serde_json::json!({ "available": false }),
    }
}

/// Stream the asset to `dest`, hashing as it goes, and verify the digest.
fn download_verified(installer: &Arc<Installer>, a: &Asset, dest: &Path) -> Result<(), String> {
    use sha2::{Digest, Sha256};

    installer.log(format!("== downloading {}", a.url));
    let resp = ureq::get(a.url)
        .call()
        .map_err(|e| format!("the download failed: {e}"))?;
    let mut reader = resp.into_body().into_reader();

    let mut file =
        std::fs::File::create(dest).map_err(|e| format!("cannot write {}: {e}", dest.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total: u64 = 0;
    let mut last_report: u64 = 0;

    loop {
        let n = std::io::Read::read(&mut reader, &mut buf)
            .map_err(|e| format!("the download was interrupted: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        std::io::Write::write_all(&mut file, &buf[..n])
            .map_err(|e| format!("cannot write {}: {e}", dest.display()))?;
        total += n as u64;

        if let Ok(mut s) = installer.state.lock() {
            s.percent = ((total.min(a.size) as f64 / a.size as f64) * 100.0) as u8;
        }
        // Every 16 MB, so a slow link still shows movement without flooding the log.
        if total - last_report >= 16 << 20 {
            last_report = total;
            installer.log(format!(
                "   {} MB of {} MB",
                total / 1_000_000,
                a.size / 1_000_000
            ));
        }
    }
    std::io::Write::flush(&mut file).map_err(|e| e.to_string())?;
    drop(file);

    let got = format!("{:x}", hasher.finalize());
    if got != a.sha256 {
        // The file is removed rather than left for someone to run by hand.
        let _ = std::fs::remove_file(dest);
        return Err(format!(
            "the downloaded file does not match the expected checksum and has been \
             deleted. Expected {}, got {got}. Do not run it; download DIA-NN from \
             the vendor instead.",
            a.sha256
        ));
    }
    installer.log(format!("== checksum verified ({} bytes)", total));
    Ok(())
}

/// Fetch DIA-NN 1.8.1 from the vendor's release and make it usable.
///
/// Refuses until the licence notice has been acknowledged, in the backend, for the
/// same reason `build` does: a disabled button is not an enforcement point.
pub fn install(installer: Arc<Installer>) -> Result<(), String> {
    if !load().licence_acknowledged {
        return Err("the DIA-NN licence notice has not been acknowledged".into());
    }
    let Some(a) = asset() else {
        return Err("no DIA-NN 1.8.1 release is published for this platform".into());
    };

    {
        let mut s = installer
            .state
            .lock()
            .map_err(|_| "internal state is poisoned".to_string())?;
        if s.status == "running" {
            return Err("an install is already running".into());
        }
        *s = InstallState {
            status: "running".into(),
            step: "downloading DIA-NN 1.8.1".into(),
            ..Default::default()
        };
    }

    std::thread::spawn(move || {
        let dir = crate::components::data_dir();
        if let Err(e) = std::fs::create_dir_all(&dir) {
            installer.fail(format!("cannot create {}: {e}", dir.display()));
            return;
        }
        let archive = dir.join(a.file_name);

        if let Err(e) = download_verified(&installer, &a, &archive) {
            installer.fail(e);
            return;
        }

        match a.kind {
            // Hand over to the vendor's installer. It shows the vendor's licence and
            // takes the acceptance, which ours does not replace.
            AssetKind::Installer => {
                installer.set("running", "starting the DIA-NN installer");
                installer.log("== launching the DIA-NN installer".into());
                match crate::engine::command(&archive).spawn() {
                    Ok(_) => {
                        installer.set("handoff", "finish in the DIA-NN installer");
                        installer.log(
                            "== the DIA-NN installer is now running. Complete it, accept \
                             DIA-NN's own licence, then press Locate."
                                .into(),
                        );
                    }
                    Err(e) => installer.fail(format!("could not start the installer: {e}")),
                }
            }

            AssetKind::Tarball => {
                installer.set("running", "extracting");
                let target = managed_dir();
                // A previous partial extraction would otherwise leave a mixture of
                // two versions' libraries in one directory.
                let _ = std::fs::remove_dir_all(&target);
                if let Err(e) = std::fs::create_dir_all(&target) {
                    installer.fail(format!("cannot create {}: {e}", target.display()));
                    return;
                }
                installer.log(format!("== extracting into {}", target.display()));
                // `tar` rather than a Rust decoder: it is present on every Linux
                // this application supports, and the archive is flat, so there is
                // nothing here a dedicated crate would handle better.
                let out = std::process::Command::new("tar")
                    .arg("xzf")
                    .arg(&archive)
                    .arg("-C")
                    .arg(&target)
                    .output();
                match out {
                    Ok(o) if o.status.success() => {}
                    Ok(o) => {
                        installer.fail(format!(
                            "extraction failed: {}",
                            String::from_utf8_lossy(&o.stderr).trim()
                        ));
                        return;
                    }
                    Err(e) => {
                        installer.fail(format!("could not run tar: {e}"));
                        return;
                    }
                }
                let _ = std::fs::remove_file(&archive);

                let Some(exe) = first_exe_in(&target) else {
                    installer.fail(format!(
                        "the archive extracted but no DIA-NN binary was found in {}",
                        target.display()
                    ));
                    return;
                };
                // The tarball carries permissions, but an extraction that lost them
                // would leave a binary that cannot be executed.
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    if let Ok(md) = std::fs::metadata(&exe) {
                        let mut perms = md.permissions();
                        perms.set_mode(perms.mode() | 0o755);
                        let _ = std::fs::set_permissions(&exe, perms);
                    }
                }

                // Verify by running it, so a broken install is reported now and not
                // at the start of a two-hour prediction.
                let (runs, version, err) = probe(&exe);
                if !runs {
                    installer.fail(err.unwrap_or_else(|| {
                        format!("{} did not run after extraction", exe.display())
                    }));
                    return;
                }
                if let Err(e) = set_path(Some(&exe.display().to_string())) {
                    installer.fail(e);
                    return;
                }
                installer.log(format!("== installed: {}", version.unwrap_or_default()));
                installer.set("done", "installed");
            }
        }
    });

    Ok(())
}

/// What to predict, and how. The defaults match `README.md`'s worked example.
#[derive(Deserialize, Clone, Debug)]
pub struct BuildRequest {
    pub fasta: String,
    pub out_dir: String,
    #[serde(default = "d_missed")]
    pub missed_cleavages: u32,
    #[serde(default = "d_min_len")]
    pub min_pep_len: u32,
    #[serde(default = "d_max_len")]
    pub max_pep_len: u32,
    #[serde(default = "d_min_z")]
    pub min_charge: u32,
    #[serde(default = "d_max_z")]
    pub max_charge: u32,
    #[serde(default = "d_threads")]
    pub threads: u32,
    /// Fixed Carbamidomethyl (DIA-NN `--unimod4`).
    #[serde(default = "d_true")]
    pub carbamidomethyl: bool,
    /// Variable Oxidation of methionine.
    #[serde(default = "d_true")]
    pub oxidation: bool,
}

fn d_missed() -> u32 {
    1
}
fn d_min_len() -> u32 {
    7
}
fn d_max_len() -> u32 {
    30
}
fn d_min_z() -> u32 {
    2
}
fn d_max_z() -> u32 {
    4
}
fn d_threads() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(8)
}
fn d_true() -> bool {
    true
}

// ── reusing a predicted library ─────────────────────────────────────────────
//
// A predicted library depends on the FASTA and the digest parameters and on nothing
// else: not on the mzML, not on the thread count. Predicting one takes DIA-NN a long
// time on a whole proteome, so a search that starts from a FASTA must not pay for it
// twice. These two functions are what let the search screen offer "predict the
// library for me" without that being a promise to rebuild it on every run.

/// A content-addressed directory for the library this request describes.
///
/// Keyed on the FASTA's own bytes plus every parameter that changes the output plus
/// the DIA-NN version, because DIA-NN's version changes what it predicts and reusing
/// a library across versions would silently change results. The thread count is
/// deliberately excluded: it does not change the library, and including it would miss
/// the cache for no reason.
/// Bumped whenever the DIA-NN command line this module builds changes.
///
/// The cache key must move with the recipe, not just with the parameters: a library
/// built by an older recipe is a different library. Raised to 2 when `--met-excision`
/// was added, because every entry written before that lacks the initiator-Met-excised
/// peptides and reusing one would silently reinstate the defect the flag fixes.
const RECIPE_VERSION: u32 = 2;

pub fn library_cache_dir(req: &BuildRequest, diann_version: &str) -> Result<PathBuf, String> {
    use sha2::{Digest, Sha256};
    let fasta = std::fs::read(&req.fasta).map_err(|e| format!("cannot read {}: {e}", req.fasta))?;
    let mut h = Sha256::new();
    h.update(&fasta);
    h.update(
        format!(
            "|recipe={}|mc={}|len={}-{}|z={}-{}|cam={}|ox={}|diann={}",
            RECIPE_VERSION,
            req.missed_cleavages,
            req.min_pep_len,
            req.max_pep_len,
            req.min_charge,
            req.max_charge,
            req.carbamidomethyl,
            req.oxidation,
            diann_version,
        )
        .as_bytes(),
    );
    let key = format!("{:x}", h.finalize());
    Ok(crate::components::data_dir()
        .join("libraries")
        .join(&key[..16]))
}

/// The two tables in `dir`, if a previous build finished there.
///
/// Both must be present. A build interrupted between writing the two would otherwise
/// look like a usable cache entry, and the search would fail on a missing file after
/// the interface had already said it was reusing a library.
pub fn cached_library(dir: &Path) -> Option<(String, String)> {
    let p = dir.join("lib_precursors.parquet");
    let f = dir.join("lib_fragments.parquet");
    if !p.is_file() || !f.is_file() {
        return None;
    }
    // A completion marker, written last. Existence alone was not enough: a
    // `make_reverse_decoys.py` killed mid-write leaves a TRUNCATED
    // `lib_fragments.parquet` that satisfied `is_file()` for ever, so `library_plan`
    // reported `ready` and the interface promised an immediate search, after which the
    // engine either rejected the library outright or accepted a partial decoy
    // population. There was no eviction and no way to force a rebuild.
    if !dir.join(CACHE_MARKER).is_file() {
        return None;
    }
    Some((p.display().to_string(), f.display().to_string()))
}

/// Written only after both tables are complete; its absence means "not usable".
pub const CACHE_MARKER: &str = "mumdia-library-complete.json";

/// Record a finished library, so the next run may reuse it.
///
/// Carries provenance rather than only acting as a flag: a one-way 16-hex directory
/// name is not something anyone can debug from, and the FASTA and parameters that
/// produced a cached library are exactly what someone asks about when a count looks
/// wrong.
fn mark_cache_complete(dir: &Path, req: &BuildRequest, version: &str) {
    let body = serde_json::json!({
        "fasta": req.fasta,
        "diann_version": version,
        "recipe_version": RECIPE_VERSION,
        "missed_cleavages": req.missed_cleavages,
        "pep_len": [req.min_pep_len, req.max_pep_len],
        "charge": [req.min_charge, req.max_charge],
        "carbamidomethyl": req.carbamidomethyl,
        "oxidation": req.oxidation,
    });
    if let Ok(text) = serde_json::to_string_pretty(&body) {
        let _ = std::fs::write(dir.join(CACHE_MARKER), text);
    }
}

/// What a search starting from a FASTA would do: reuse a library, or build one.
///
/// Answered before anything is started, so the interface can say "this will take a
/// while" or "this is already built" rather than discovering it afterwards.
pub fn library_plan(mut req: BuildRequest) -> Result<serde_json::Value, String> {
    let status = detect();
    if !status.runs {
        return Err("no working DIA-NN was found".into());
    }
    if !status.licence_acknowledged {
        return Err("the DIA-NN licence notice has not been acknowledged".into());
    }
    if !status.writes_parquet {
        return Err(format!(
            "{} cannot write a Parquet spectral library, so it cannot build one here. \
             Use DIA-NN 2.x, or select a prebuilt library instead.",
            status
                .version
                .clone()
                .unwrap_or_else(|| "this DIA-NN".into())
        ));
    }
    let version = status.version.clone().unwrap_or_default();
    let dir = library_cache_dir(&req, &version)?;
    req.out_dir = dir.display().to_string();

    let cached = cached_library(&dir);
    Ok(serde_json::json!({
        "cache_dir": dir.display().to_string(),
        "diann_version": version,
        "cached": cached.as_ref().map(|(p, f)| serde_json::json!({
            "precursors": p,
            "fragments": f,
        })),
        "ready": cached.is_some(),
    }))
}

/// Progress of a library build, polled by the interface.
#[derive(Serialize, Clone, Debug, Default)]
pub struct BuildState {
    /// `idle` | `running` | `done` | `failed`
    pub status: String,
    /// Which of the three steps is running, for a caption above the log.
    pub step: String,
    pub log: Vec<String>,
    /// The two tables a search consumes, once they exist.
    pub precursors: Option<String>,
    pub fragments: Option<String>,
    pub error: Option<String>,
}

#[derive(Default)]
pub struct Builder {
    state: Mutex<BuildState>,
    /// The converter or worker currently running, so a build can be stopped.
    ///
    /// Without this a library build was uncancellable: the interface switched to the
    /// progress screen and Stop called `cancel_run`, which only knows about engine
    /// runs, so a whole-proteome DIA-NN prediction could not be interrupted at all.
    child: Mutex<Option<u32>>,
    cancelled: std::sync::atomic::AtomicBool,
}

impl Builder {
    /// Stop the running build. Safe to call when nothing is running.
    pub fn cancel(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let pid = self.child.lock().ok().and_then(|p| *p);
        if let Some(pid) = pid {
            crate::run::kill_tree(pid);
        }
        if let Ok(mut s) = self.state.lock() {
            if s.status == "running" {
                s.status = "failed".into();
                s.step = "cancelled".into();
                s.error = Some("The library build was cancelled.".into());
            }
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn snapshot(&self) -> BuildState {
        self.state
            .lock()
            .map(|s| s.clone())
            .unwrap_or_else(|_| BuildState {
                status: "failed".into(),
                error: Some("internal state is poisoned".into()),
                ..Default::default()
            })
    }

    fn log(&self, line: String) {
        if let Ok(mut s) = self.state.lock() {
            s.log.push(line);
            if s.log.len() > 4000 {
                let drop = s.log.len() - 4000;
                s.log.drain(0..drop);
            }
        }
    }

    fn fail(&self, msg: String) {
        if let Ok(mut s) = self.state.lock() {
            s.status = "failed".into();
            s.error = Some(msg);
        }
    }
}

/// Clear a previous cancellation so a new build is not stopped before it starts.
fn installer_reset(b: &Arc<Builder>) {
    b.cancelled
        .store(false, std::sync::atomic::Ordering::SeqCst);
    if let Ok(mut c) = b.child.lock() {
        *c = None;
    }
}

/// Run one child, streaming both of its streams into the log.
fn run_step(
    builder: &Arc<Builder>,
    exe: &Path,
    args: &[String],
    cwd: Option<&Path>,
    // DIA-NN needs its own directory on the loader path; the Python workers do
    // not, and adding it for them would put a virtualenv's `bin` on
    // `LD_LIBRARY_PATH` for no reason.
    is_diann: bool,
) -> Result<(), String> {
    let mut cmd = if is_diann {
        diann_command(exe)
    } else {
        crate::engine::command(exe)
    };
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    // Refuse to start another step once cancelled, or a Stop during the prediction
    // would be followed by the import and decoy steps running anyway.
    if builder.is_cancelled() {
        return Err("cancelled".into());
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("{} did not start: {e}", exe.display()))?;
    if let Ok(mut c) = builder.child.lock() {
        *c = Some(child.id());
    }

    // Both streams, on two threads, so a child that writes only to stderr still
    // shows progress and a full pipe cannot deadlock the child.
    let mut joins = Vec::new();
    for stream in [
        child
            .stdout
            .take()
            .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        child
            .stderr
            .take()
            .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
    ]
    .into_iter()
    .flatten()
    {
        let sink = Arc::clone(builder);
        joins.push(std::thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                sink.log(line);
            }
        }));
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    for j in joins {
        let _ = j.join();
    }
    if let Ok(mut c) = builder.child.lock() {
        *c = None;
    }
    if builder.is_cancelled() {
        return Err("cancelled".into());
    }
    if !status.success() {
        return Err(format!(
            "{} exited with {}",
            exe.file_name().unwrap_or_default().to_string_lossy(),
            status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "a signal".into())
        ));
    }
    Ok(())
}

/// The fragment-level Parquet library, if DIA-NN has written one.
///
/// The name has varied between versions, so it is found by prefix rather than
/// assumed.
fn find_lib_parquet(dir: &Path, stem: &str) -> Option<PathBuf> {
    let direct = dir.join(format!("{stem}.parquet"));
    if direct.is_file() {
        return Some(direct);
    }
    // Anchored on `<stem>.`, NOT `starts_with(stem)`.
    //
    // The unanchored prefix matched this pipeline's OWN outputs, which land in the
    // same directory: `lib_precursors.parquet`, `lib_fragments.parquet` and their
    // `_targets` variants all start with "lib". On DIA-NN 2.x, which writes
    // `lib.predicted.parquet`, a rebuild in a directory that had completed once
    // sorted ['lib.predicted.parquet', 'lib_fragments...', 'lib_precursors...']
    // ('.' is 0x2E, '_' is 0x5F) and `pop()` returned `lib_precursors_targets.parquet`
    // -- so MuMDIA's own schema table was imported as the DIA-NN library and the run
    // reported success on a library that did not match the requested FASTA.
    let prefix = format!("{stem}.");
    let mut hits: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "parquet")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with(&prefix))
        })
        .collect();
    hits.sort();
    hits.pop()
}

/// The predicted library DIA-NN actually writes.
///
/// A PREDICTED library is always DIA-NN's own compact binary format,
/// `<name>.predicted.speclib`; only an empirical, DIA-derived library is written as
/// Parquet. `README.md` had a command that asked for `--out-lib lib` and a next step
/// that read `lib.parquet`, and those cannot both be right: the file in between is a
/// `.speclib`, which nothing in MuMDIA reads. The one hint anybody had written down
/// was the phrase "the re-exported speclib" in a comment in
/// `scripts/import_diann_lib.py`.
fn find_predicted_speclib(dir: &Path, stem: &str) -> Option<PathBuf> {
    let direct = dir.join(format!("{stem}.predicted.speclib"));
    if direct.is_file() {
        return Some(direct);
    }
    let mut hits: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "speclib")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with(stem))
        })
        .collect();
    hits.sort();
    hits.pop()
}

/// The DIA-NN command line for predicting a library from a FASTA.
///
/// Extracted so it can be asserted rather than described. `--met-excision` was
/// missing here for the whole of this feature's life and nothing caught it, because
/// the arguments were built inline and the only statement that they were right was a
/// comment.
fn predict_args(req: &BuildRequest, out_lib: &Path) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "--fasta".into(),
        req.fasta.clone(),
        "--fasta-search".into(),
        "--gen-spec-lib".into(),
        "--predictor".into(),
        "--cut".into(),
        "K*,R*".into(),
        // Emit N-terminal initiator-Met-excised forms.
        //
        // Not optional, and its absence was a silent scientific defect rather than a
        // missing convenience. MuMDIA's native digest defaults to
        // `digest.n_term_met_excision = true` precisely "matching DIA-NN
        // `--met-excision`" (CLAUDE.md), and without it the search database
        // structurally misses those peptides. A raw imported DIA-NN library built
        // without it was missing 209 of DIA-NN's own 1% peptides on the AIF benchmark,
        // ALL of them this form -- the entire reason `augment_library.py` exists. This
        // recipe omitted the flag, so every library the interface built lacked them,
        // with nothing saying so, and the loss would have read as a MuMDIA sensitivity
        // result rather than a library-recipe defect.
        "--met-excision".into(),
        "--missed-cleavages".into(),
        req.missed_cleavages.to_string(),
        "--min-pep-len".into(),
        req.min_pep_len.to_string(),
        "--max-pep-len".into(),
        req.max_pep_len.to_string(),
        "--min-pr-charge".into(),
        req.min_charge.to_string(),
        "--max-pr-charge".into(),
        req.max_charge.to_string(),
    ];
    if req.carbamidomethyl {
        args.push("--unimod4".into());
    }
    if req.oxidation {
        args.push("--var-mods".into());
        args.push("1".into());
        args.push("--var-mod".into());
        args.push("UniMod:35,15.994915,M".into());
    }
    args.push("--out-lib".into());
    args.push(out_lib.display().to_string());
    args.push("--threads".into());
    args.push(req.threads.to_string());
    args
}

/// Predict a library with the user's DIA-NN, then convert it to MuMDIA's schema.
///
/// Three steps, all of which must succeed: DIA-NN predicts, `import_diann_lib.py`
/// maps it into the target schema, and `make_reverse_decoys.py` adds the decoy
/// population -- the last of which also sorts by precursor m/z and re-indexes
/// `candidate_id`, both of which the fragment index rejects a library for lacking.
pub fn build(builder: Arc<Builder>, req: BuildRequest) -> Result<(), String> {
    let status = detect();
    if !status.licence_acknowledged {
        return Err("the DIA-NN licence notice has not been acknowledged".into());
    }
    let exe = match (&status.path, status.runs) {
        (Some(p), true) => PathBuf::from(p),
        _ => return Err("no working DIA-NN was found".into()),
    };
    // Before the expensive step, not after it.
    if !status.writes_parquet {
        return Err(format!(
            "{} cannot write a Parquet spectral library, which this needs: 1.8.x has \
             no Parquet library output. Predicting would take a long time and then \
             fail at the conversion step. Use DIA-NN 2.x for library building, or \
             build the library manually and select the two tables directly.",
            status.version.unwrap_or_else(|| "this DIA-NN".into())
        ));
    }
    // The importer and the decoy builder need pandas and pyarrow, which the
    // primary managed environment has.
    let python = crate::components::managed_python(crate::components::Env::Primary);
    if !python.is_file() {
        return Err(
            "the analysis components are not installed, and the library \
                    conversion needs them"
                .into(),
        );
    }
    let scripts = crate::engine::scripts_dir()
        .ok_or_else(|| "the conversion scripts were not found".to_string())?;

    let fasta = PathBuf::from(&req.fasta);
    if !fasta.is_file() {
        return Err(format!("{} is not a file", fasta.display()));
    }
    let out_dir = PathBuf::from(&req.out_dir);
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;

    {
        let mut s = builder
            .state
            .lock()
            .map_err(|_| "internal state is poisoned".to_string())?;
        if s.status == "running" {
            return Err("a library build is already running".into());
        }
        *s = BuildState {
            status: "running".into(),
            step: "predicting the library with DIA-NN".into(),
            ..Default::default()
        };
        installer_reset(&builder);
    }

    std::thread::spawn(move || {
        let stem = "lib";
        let out_lib = out_dir.join(stem);

        // The command line recorded in README.md, so this reproduces what a user
        // following the written instructions would have run.
        let args = predict_args(&req, &out_lib);

        let version_for_cache = status.version.clone();
        builder.log(format!("== DIA-NN: {}", status.version.unwrap_or_default()));
        builder.log(format!("== {} {}", exe.display(), args.join(" ")));
        if let Err(e) = run_step(&builder, &exe, &args, Some(&out_dir), true) {
            builder.fail(e);
            return;
        }

        // DIA-NN writes a predicted library as its own binary `.speclib`, so a
        // second DIA-NN run is needed to re-export it as the Parquet the importer
        // reads. Parquet is looked for first anyway: a version that writes one
        // directly, or a `--out-lib` that already named one, needs no re-export and
        // paying for one would be minutes wasted on a large library.
        let predicted = match find_lib_parquet(&out_dir, stem) {
            Some(p) => {
                builder.log(format!("== predicted library (Parquet): {}", p.display()));
                p
            }
            None => {
                let Some(speclib) = find_predicted_speclib(&out_dir, stem) else {
                    builder.fail(format!(
                        "DIA-NN finished but wrote neither a Parquet nor a .speclib \
                         library in {}. Check the log: a licence or prediction failure \
                         can still exit successfully.",
                        out_dir.display()
                    ));
                    return;
                };
                if let Ok(mut s) = builder.state.lock() {
                    s.step = "re-exporting the library as Parquet".into();
                }
                builder.log(format!(
                    "== DIA-NN wrote {}, which MuMDIA cannot read; re-exporting it as \
                     Parquet",
                    speclib.display()
                ));
                let target = out_dir.join(format!("{stem}.parquet"));
                let reexport = vec![
                    "--lib".to_string(),
                    speclib.to_string_lossy().into_owned(),
                    "--gen-spec-lib".to_string(),
                    "--out-lib".to_string(),
                    target.to_string_lossy().into_owned(),
                    "--threads".to_string(),
                    req.threads.to_string(),
                ];
                builder.log(format!("== {} {}", exe.display(), reexport.join(" ")));
                if let Err(e) = run_step(&builder, &exe, &reexport, Some(&out_dir), true) {
                    builder.fail(format!(
                        "{e}\n\nThe prediction succeeded and {} exists, but converting \
                         it to Parquet failed. Convert it by hand with:\n  {} --lib {} \
                         --gen-spec-lib --out-lib {}\nthen point the search at the \
                         resulting tables.",
                        speclib.display(),
                        exe.display(),
                        speclib.display(),
                        target.display()
                    ));
                    return;
                }
                match find_lib_parquet(&out_dir, stem) {
                    Some(p) => {
                        builder.log(format!("== re-exported: {}", p.display()));
                        p
                    }
                    None => {
                        builder.fail(format!(
                            "the re-export ran but produced no Parquet in {}. Your DIA-NN \
                             version may not support Parquet library output; 1.8.x does \
                             not. Use DIA-NN 2.x, or export the library as .tsv and \
                             convert it yourself.",
                            out_dir.display()
                        ));
                        return;
                    }
                }
            }
        };

        let tgt_prec = out_dir.join("lib_precursors_targets.parquet");
        let tgt_frag = out_dir.join("lib_fragments_targets.parquet");
        let prec = out_dir.join("lib_precursors.parquet");
        let frag = out_dir.join("lib_fragments.parquet");

        let steps: Vec<(&str, Vec<String>)> = vec![
            (
                "converting to the MuMDIA schema",
                vec![
                    scripts.join("import_diann_lib.py").display().to_string(),
                    predicted.display().to_string(),
                    tgt_prec.display().to_string(),
                    tgt_frag.display().to_string(),
                ],
            ),
            (
                "adding the decoy population",
                vec![
                    scripts.join("make_reverse_decoys.py").display().to_string(),
                    tgt_prec.display().to_string(),
                    tgt_frag.display().to_string(),
                    prec.display().to_string(),
                    frag.display().to_string(),
                ],
            ),
        ];

        for (title, args) in steps {
            if let Ok(mut s) = builder.state.lock() {
                s.step = title.into();
            }
            builder.log(format!("== {title}"));
            if let Err(e) = run_step(&builder, &python, &args, Some(&scripts), false) {
                builder.fail(e);
                return;
            }
        }

        // The marker last, after both tables are on disk. Anything that dies before
        // this leaves a directory `cached_library` will not accept, which is the
        // point: a half-written library must not read as a finished one.
        mark_cache_complete(&out_dir, &req, version_for_cache.as_deref().unwrap_or(""));

        if let Ok(mut s) = builder.state.lock() {
            s.status = "done".into();
            s.step = "finished".into();
            s.precursors = Some(prec.display().to_string());
            s.fragments = Some(frag.display().to_string());
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_licence_notice_says_what_mumdia_does_not_do() {
        // The whole point of the notice is the distinction from the cards above it,
        // which do install software. If it stops saying so, it is decoration.
        let n = LICENCE_NOTICE;
        assert!(
            n.contains("does not include, modify or redistribute"),
            "the notice must state that MuMDIA does not ship DIA-NN: {n}"
        );
        // It must not claim MuMDIA never downloads DIA-NN, because the download
        // button does exactly that. An inaccurate notice is worse than none.
        assert!(
            n.contains("release page") && n.contains("checksum"),
            "the notice must describe the download honestly: {n}"
        );
        assert!(
            n.contains("Academia") && n.contains("non-profit"),
            "the notice must name the restriction a commercial user can breach: {n}"
        );
        // Version changes the predicted library, so it changes results.
        assert!(
            n.contains("version"),
            "the notice must mention the version: {n}"
        );
    }

    #[test]
    fn a_build_is_refused_until_the_licence_is_acknowledged() {
        // The acknowledgement is a precondition inside `build`, not only a disabled
        // button in the interface, so that it cannot be bypassed by calling the
        // command directly.
        let saved = load();
        if saved.licence_acknowledged {
            // The developer machine has already acknowledged; this assertion is
            // about the refusal path, so skip rather than clobber their state.
            return;
        }
        let err = build(
            Arc::new(Builder::default()),
            BuildRequest {
                fasta: "nonexistent.fasta".into(),
                out_dir: ".".into(),
                missed_cleavages: 1,
                min_pep_len: 7,
                max_pep_len: 30,
                min_charge: 2,
                max_charge: 4,
                threads: 1,
                carbamidomethyl: true,
                oxidation: true,
            },
        )
        .unwrap_err();
        assert!(
            err.contains("licence"),
            "refused for the licence reason: {err}"
        );
    }

    #[test]
    fn a_missing_binary_is_reported_rather_than_reported_as_ready() {
        // A path that is not there must never read as usable: the failure would
        // otherwise appear only when DIA-NN was supposed to run.
        let (runs, version, err) = probe(Path::new("definitely-not-a-real-diann-binary"));
        assert!(!runs);
        assert!(version.is_none());
        assert!(err.is_some());
    }

    #[test]
    fn a_configured_path_wins_over_everything_else() {
        // Order matters for reproducibility: a user who pointed at one DIA-NN must
        // not silently get a different one from PATH, because the version changes
        // the library.
        let saved = Saved {
            path: Some("/somewhere/diann".into()),
            licence_acknowledged: true,
        };
        let c = candidates(&saved);
        assert_eq!(c.first().map(|(s, _)| s.as_str()), Some("configured"));
    }

    #[test]
    fn the_predicted_parquet_is_found_by_name_or_by_search() {
        let dir = std::env::temp_dir().join("mumdia-diann-test-lib");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        assert!(find_lib_parquet(&dir, "lib").is_none());

        // The name DIA-NN has actually used varies between versions, so a
        // prefix match is the fallback.
        std::fs::write(dir.join("lib.predicted.parquet"), b"x").unwrap();
        assert!(find_lib_parquet(&dir, "lib").is_some());

        // The exact name wins when it exists.
        std::fs::write(dir.join("lib.parquet"), b"x").unwrap();
        assert_eq!(
            find_lib_parquet(&dir, "lib").unwrap().file_name().unwrap(),
            "lib.parquet"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_build_request_defaults_match_the_documented_command_line() {
        // README.md's worked example is the reference; the form must not quietly
        // search a different space than the manual instructions do.
        let req: BuildRequest =
            serde_json::from_str(r#"{"fasta":"a.fasta","out_dir":"out"}"#).unwrap();
        assert_eq!(req.missed_cleavages, 1);
        assert_eq!((req.min_pep_len, req.max_pep_len), (7, 30));
        assert_eq!((req.min_charge, req.max_charge), (2, 4));
        assert!(req.carbamidomethyl && req.oxidation);
    }

    #[test]
    fn the_download_is_pinned_to_one_version_from_the_vendors_own_host() {
        // Three properties, each of which the module comment says the legality of
        // offering this rests on, and each of which is easy to break later.
        let Some(a) = asset() else { return };

        // Not a mirror, not a proxy, not our own storage.
        assert!(
            a.url
                .starts_with("https://github.com/vdemichev/DiaNN/releases/download/"),
            "the asset must come from the vendor's own release: {}",
            a.url
        );
        // Pinned, never "latest": a newer release is one this reasoning has not been
        // applied to, and it would silently change the predicted library.
        assert!(a.url.contains("/1.8.1/"), "pinned to 1.8.1: {}", a.url);
        assert!(!a.url.contains("latest"), "never resolve latest: {}", a.url);
        // The bytes get executed, so the digest must be a real one.
        assert_eq!(a.sha256.len(), 64, "a sha256 is 64 hex characters");
        assert!(a.sha256.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(a.size > 0);
    }

    #[test]
    fn an_install_is_refused_until_the_licence_is_acknowledged() {
        // Same enforcement point as `build`: the interface disables the button, but
        // the button is not the thing that must hold.
        if load().licence_acknowledged {
            return;
        }
        let err = install(Arc::new(Installer::default())).unwrap_err();
        assert!(
            err.contains("licence"),
            "refused for the licence reason: {err}"
        );
    }

    #[test]
    fn the_offer_is_honest_about_what_it_costs() {
        let o = offer();
        if !o["available"].as_bool().unwrap_or(false) {
            return;
        }
        assert_eq!(o["version"], "1.8.1");
        // The Linux tarball is 142 MB and unpacks to roughly 490 MB, nearly all of
        // it libtorch. Reporting only the download would understate it threefold.
        let dl = o["download_bytes"].as_u64().unwrap();
        let disk = o["disk_bytes"].as_u64().unwrap();
        assert!(dl > 100_000_000, "the real asset is over 100 MB");
        assert!(disk >= dl, "disk cost is never less than the download");
    }

    #[test]
    fn the_managed_install_is_per_user_and_not_beside_the_application() {
        // Program Files needs administrator rights, and an install that needs them
        // is not the easy installation this exists to provide.
        let d = managed_dir();
        assert!(d.starts_with(crate::components::data_dir()));
        assert!(d.to_string_lossy().contains("1.8.1"));
    }

    #[test]
    fn the_linux_binary_name_from_the_tarball_is_recognised() {
        // The 1.8.1 tarball is flat and names its binary `diann-1.8.1`, which the
        // detection search has to know or a managed install becomes invisible.
        if cfg!(windows) {
            return;
        }
        assert!(exe_names().contains(&"diann-1.8.1"));
    }

    /// A one-request HTTP server on loopback, so the download path is exercised for
    /// real -- streaming, hashing, verification -- without touching the network or
    /// pulling 142 MB in a test.
    fn serve_once(body: Vec<u8>) -> String {
        use std::io::Write as _;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut head = Vec::new();
                // Read past the request head; the body is irrelevant for GET.
                let mut buf = [0u8; 1024];
                loop {
                    match std::io::Read::read(&mut sock, &mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            head.extend_from_slice(&buf[..n]);
                            if head.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                let _ = write!(
                    sock,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(&body);
                let _ = sock.flush();
            }
        });
        format!("http://127.0.0.1:{port}/asset")
    }

    fn tmp_path(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia-diann-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d.join("asset.bin")
    }

    #[test]
    fn a_download_that_matches_its_digest_is_kept() {
        use sha2::{Digest, Sha256};
        let body = b"the quick brown fox jumps over the lazy dog".to_vec();
        let digest = format!("{:x}", Sha256::digest(&body));
        let url: &'static str = Box::leak(serve_once(body.clone()).into_boxed_str());
        let sha: &'static str = Box::leak(digest.into_boxed_str());

        let dest = tmp_path("ok");
        let a = Asset {
            url,
            sha256: sha,
            size: body.len() as u64,
            kind: AssetKind::Tarball,
            file_name: "asset.bin",
        };
        download_verified(&Arc::new(Installer::default()), &a, &dest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        let _ = std::fs::remove_dir_all(dest.parent().unwrap());
    }

    #[test]
    fn a_download_that_fails_its_digest_is_deleted_not_left_to_be_run() {
        // The whole point of pinning: these bytes would otherwise be executed, or
        // handed to the operating system's installer. A mismatch must not leave a
        // runnable file behind for someone to double-click.
        let body = b"not what was expected".to_vec();
        let url: &'static str = Box::leak(serve_once(body.clone()).into_boxed_str());

        let dest = tmp_path("bad");
        let a = Asset {
            url,
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
            size: body.len() as u64,
            kind: AssetKind::Installer,
            file_name: "asset.bin",
        };
        let err = download_verified(&Arc::new(Installer::default()), &a, &dest).unwrap_err();
        assert!(
            err.contains("checksum"),
            "the reason must be the checksum: {err}"
        );
        assert!(
            !dest.exists(),
            "a file that failed verification must not be left on disk"
        );
        let _ = std::fs::remove_dir_all(dest.parent().unwrap());
    }

    #[test]
    fn the_graphical_front_end_is_refused_by_name_before_it_is_run() {
        // DIA-NN ships a GUI (`DIA-NN.exe`) beside the command-line tool
        // (`diann.exe`), and pointing at the GUI is the obvious mistake. Executing it
        // would open a window and never exit, which used to hang the Setup screen for
        // ever, so it is refused by name rather than probed.
        assert!(is_graphical_frontend(Path::new("C:/DIA-NN/2.0/DIA-NN.exe")));
        assert!(is_graphical_frontend(Path::new("/opt/diann/dia-nn")));
        // The hyphen is the whole difference; the command-line tool must pass.
        assert!(!is_graphical_frontend(Path::new("C:/DIA-NN/2.0/diann.exe")));
        assert!(!is_graphical_frontend(Path::new("/usr/diann/diann")));
        assert!(!is_graphical_frontend(Path::new("diann-1.8.1")));

        let (runs, version, err) = probe(Path::new("C:/DIA-NN/2.0/DIA-NN.exe"));
        assert!(!runs);
        assert!(version.is_none());
        let msg = err.expect("a refusal must explain itself");
        assert!(
            msg.contains("diann.exe"),
            "it must name the right file: {msg}"
        );
    }

    #[test]
    fn a_program_that_never_exits_is_given_up_on_rather_than_hanging() {
        // The backstop behind the name check. `Command::output` waits for the child,
        // so any program that does not exit blocked the Tauri command that called
        // `probe` for ever. Uses a real long-running process, and asserts only that
        // the probe returns well within the process's own lifetime.
        let sleeper = if cfg!(windows) {
            // `timeout` needs a console; `ping` is the portable "wait a while".
            ("ping", vec!["-n", "60", "127.0.0.1"])
        } else {
            ("sleep", vec!["60"])
        };
        let Ok(which) = which_on_path(sleeper.0) else {
            eprintln!("{} not on PATH; skipping", sleeper.0);
            return;
        };

        // A shorter bound than PROBE_TIMEOUT would need the constant to be
        // injectable; instead this asserts the probe returns at all, and well before
        // the child would have finished on its own.
        let started = std::time::Instant::now();
        let (runs, _, err) = probe_with_args(&which, &sleeper.1);
        let elapsed = started.elapsed();

        assert!(!runs, "a sleeper must not report as a working DIA-NN");
        assert!(err.is_some());
        assert!(
            elapsed < std::time::Duration::from_secs(40),
            "the probe must give up, not wait for the child: took {elapsed:?}"
        );
    }

    /// `probe` with extra arguments, so the timeout can be exercised against a real
    /// long-running program. Production `probe` passes none.
    fn probe_with_args(exe: &Path, args: &[&str]) -> (bool, Option<String>, Option<String>) {
        let mut cmd = diann_command(exe);
        cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());
        let Ok(mut child) = cmd.spawn() else {
            return (false, None, Some("did not spawn".into()));
        };
        let started = std::time::Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(_)) => return (false, None, Some("exited".into())),
                Ok(None) => {}
                Err(_) => return (false, None, Some("unknown".into())),
            }
            if started.elapsed() >= PROBE_TIMEOUT {
                let _ = child.kill();
                let _ = child.wait();
                return (false, None, Some("timed out".into()));
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    fn which_on_path(name: &str) -> Result<PathBuf, ()> {
        let exe = if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.to_string()
        };
        let path = std::env::var_os("PATH").ok_or(())?;
        std::env::split_paths(&path)
            .map(|d| d.join(&exe))
            .find(|p| p.is_file())
            .ok_or(())
    }

    #[test]
    fn a_predicted_speclib_is_recognised_so_it_can_be_re_exported() {
        // The bug this guards. DIA-NN writes a PREDICTED library as its own binary
        // `.speclib`; only an empirical one is Parquet. Without recognising the
        // `.speclib`, a successful prediction was reported as "DIA-NN finished but
        // wrote no Parquet library" and the user was left with a file nothing reads.
        let dir = std::env::temp_dir().join("mumdia-diann-speclib");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        assert!(find_predicted_speclib(&dir, "lib").is_none());
        assert!(find_lib_parquet(&dir, "lib").is_none());

        // What DIA-NN actually writes.
        std::fs::write(dir.join("lib.predicted.speclib"), b"x").unwrap();
        assert_eq!(
            find_predicted_speclib(&dir, "lib")
                .unwrap()
                .file_name()
                .unwrap(),
            "lib.predicted.speclib"
        );
        // And it is not mistaken for the Parquet the importer needs.
        assert!(find_lib_parquet(&dir, "lib").is_none());

        // Once re-exported, Parquet wins and no second re-export is paid for.
        std::fs::write(dir.join("lib.parquet"), b"x").unwrap();
        assert!(find_lib_parquet(&dir, "lib").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn a_request(fasta: &str) -> BuildRequest {
        BuildRequest {
            fasta: fasta.to_string(),
            out_dir: String::new(),
            missed_cleavages: 1,
            min_pep_len: 7,
            max_pep_len: 30,
            min_charge: 2,
            max_charge: 4,
            threads: 8,
            carbamidomethyl: true,
            oxidation: true,
        }
    }

    #[test]
    fn the_library_cache_key_covers_what_changes_the_library_and_nothing_else() {
        let dir = std::env::temp_dir().join("mumdia-diann-cachekey");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let fa = dir.join("a.fasta");
        std::fs::write(&fa, b">sp|P1|X\nPEPTIDEK\n").unwrap();
        let path = fa.to_str().unwrap();

        let base = library_cache_dir(&a_request(path), "DIA-NN 2.0").unwrap();

        // The thread count does not change the library, so it must not change the
        // key: including it would miss the cache for no reason.
        let mut threads = a_request(path);
        threads.threads = 1;
        assert_eq!(library_cache_dir(&threads, "DIA-NN 2.0").unwrap(), base);

        // Everything that DOES change the library must change the key.
        for mutate in [
            (|r: &mut BuildRequest| r.missed_cleavages = 2) as fn(&mut BuildRequest),
            |r: &mut BuildRequest| r.min_pep_len = 6,
            |r: &mut BuildRequest| r.max_pep_len = 40,
            |r: &mut BuildRequest| r.min_charge = 1,
            |r: &mut BuildRequest| r.max_charge = 5,
            |r: &mut BuildRequest| r.carbamidomethyl = false,
            |r: &mut BuildRequest| r.oxidation = false,
        ] {
            let mut r = a_request(path);
            mutate(&mut r);
            assert_ne!(
                library_cache_dir(&r, "DIA-NN 2.0").unwrap(),
                base,
                "a parameter that changes the library must change the cache key"
            );
        }

        // DIA-NN's version changes what it predicts, so a library must not be reused
        // across versions.
        assert_ne!(
            library_cache_dir(&a_request(path), "DIA-NN 1.8.1").unwrap(),
            base
        );

        // And the FASTA's contents, not just its name.
        std::fs::write(&fa, b">sp|P1|X\nPEPTIDEKR\n").unwrap();
        assert_ne!(
            library_cache_dir(&a_request(path), "DIA-NN 2.0").unwrap(),
            base
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cached_library_needs_both_tables_and_a_completion_marker() {
        // Two failures this guards, and the second is the one that bites in practice.
        //
        // One table is obviously not a library: a build interrupted between writing
        // them would otherwise read as reusable and the search would fail on a missing
        // file after the interface had said it was reusing one.
        //
        // Two tables are ALSO not enough. `make_reverse_decoys.py` killed mid-write
        // leaves a truncated `lib_fragments.parquet` that satisfies `is_file()` for
        // ever, so `library_plan` reported ready, the interface promised an immediate
        // search, and the engine then either rejected the library or accepted a
        // partial decoy population -- with no eviction and no way to force a rebuild.
        // The marker is written last, after both tables are complete.
        let dir = std::env::temp_dir().join("mumdia-diann-halfcache");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        assert!(
            cached_library(&dir).is_none(),
            "an empty directory is not a cache hit"
        );

        std::fs::write(dir.join("lib_precursors.parquet"), b"x").unwrap();
        assert!(cached_library(&dir).is_none(), "one table is not a library");

        std::fs::write(dir.join("lib_fragments.parquet"), b"x").unwrap();
        assert!(
            cached_library(&dir).is_none(),
            "two tables without the completion marker are a possibly-truncated build"
        );

        std::fs::write(dir.join(CACHE_MARKER), b"{}").unwrap();
        assert!(
            cached_library(&dir).is_some(),
            "complete builds are reusable"
        );

        // And the marker alone proves nothing if a table is later removed.
        std::fs::remove_file(dir.join("lib_fragments.parquet")).unwrap();
        assert!(cached_library(&dir).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn diann_1_8_cannot_build_a_library_and_is_refused_before_predicting() {
        // The interface offered 1.8.1 as its download and then offered a library build
        // that 1.8.x cannot finish: it has no Parquet library output, so the re-export
        // fails -- after a whole-proteome prediction, on a path the interface itself
        // recommended.
        assert!(!writes_parquet_libraries(
            "DIA-NN 1.8.1 (Data-Independent Acquisition)"
        ));
        assert!(!writes_parquet_libraries("DIA-NN 1.8"));
        assert!(!writes_parquet_libraries("DIA-NN 1.8.2"));

        // 1.9 introduced it; 2.x has it.
        assert!(writes_parquet_libraries("DIA-NN 1.9"));
        assert!(writes_parquet_libraries("DIA-NN 1.9.2"));
        assert!(writes_parquet_libraries("DIA-NN 2.0"));
        assert!(writes_parquet_libraries("DIA-NN 2.2.0"));

        // An unreadable banner is treated as capable: refusing on a version we cannot
        // parse would block working installations, and the build still reports the
        // real failure if it turns out not to be.
        assert!(writes_parquet_libraries(""));
        assert!(writes_parquet_libraries("DIA-NN"));
        assert!(writes_parquet_libraries("DIA-NN unreleased-build"));
    }

    #[test]
    fn the_prediction_recipe_emits_met_excised_peptides() {
        // Asserted against the argument vector the build actually uses, which is why
        // `predict_args` exists as a function. The flag was absent for this feature's
        // whole life and nothing caught it: the arguments were built inline and the
        // only claim they were right was a comment.
        //
        // MuMDIA's native digest defaults to Met-excision, so a library built without
        // it structurally misses those peptides -- 209 of DIA-NN's own 1% peptides on
        // the AIF benchmark, every one of them this form.
        let req = a_request("x.fasta");
        let args = predict_args(&req, Path::new("out/lib"));
        assert!(
            args.iter().any(|a| a == "--met-excision"),
            "the recipe must emit Met-excised peptides: {args:?}"
        );
        // The flags the importer's modification mapping depends on.
        assert!(args.iter().any(|a| a == "--unimod4"));
        assert!(args.iter().any(|a| a == "UniMod:35,15.994915,M"));
        // And it must still be a prediction run writing where it was told to.
        assert!(args.iter().any(|a| a == "--predictor"));
        assert_eq!(
            args.iter()
                .position(|a| a == "--out-lib")
                .map(|i| &args[i + 1]),
            Some(&"out/lib".to_string())
        );
    }

    #[test]
    fn the_parquet_search_never_matches_mumdias_own_output_tables() {
        // The bug: an unanchored `starts_with("lib")` matched the pipeline's own
        // outputs, which are written into the same directory. With DIA-NN 2.x naming
        // its library `lib.predicted.parquet`, '.' (0x2E) sorts before '_' (0x5F), so
        // `pop()` returned `lib_precursors_targets.parquet` and MuMDIA's own schema
        // table was imported as the DIA-NN library, reporting success on a library
        // that did not match the FASTA.
        let dir = std::env::temp_dir().join("mumdia-diann-anchor");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Exactly what a completed build leaves behind.
        for n in [
            "lib_precursors.parquet",
            "lib_fragments.parquet",
            "lib_precursors_targets.parquet",
            "lib_fragments_targets.parquet",
        ] {
            std::fs::write(dir.join(n), b"x").unwrap();
        }
        assert!(
            find_lib_parquet(&dir, "lib").is_none(),
            "our own output tables must never be mistaken for the DIA-NN library"
        );

        // DIA-NN's actual name is still found, even surrounded by them.
        std::fs::write(dir.join("lib.predicted.parquet"), b"x").unwrap();
        assert_eq!(
            find_lib_parquet(&dir, "lib").unwrap().file_name().unwrap(),
            "lib.predicted.parquet"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
