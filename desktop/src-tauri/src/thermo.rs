//! Installing ThermoRawFileParser, so a user with Thermo `.raw` files and no mzML
//! can search them.
//!
//! # Why this one is a plain install, unlike DIA-NN
//!
//! `diann.rs` goes to considerable lengths -- a licence notice, an acknowledgement
//! gate, a pinned single version, a hand-off to the vendor's own installer --
//! because DIA-NN is closed source and not redistributable. None of that applies
//! here. ThermoRawFileParser is Apache-2.0, from CompOmics, and redistributable, so
//! this is an ordinary managed component in the mould of `components.rs`: press
//! Install, and it installs.
//!
//! The URL and digest are still pinned, for the reason every downloaded executable
//! should be: the bytes get run. That is a supply-chain measure, not a licence one.
//!
//! # Why the engine does not do this itself
//!
//! The engine locates a converter (`raw::locate_parser`) and refuses clearly when
//! there is none. It does not fetch one, because a search engine that downloads
//! software mid-run is a worse thing to operate than one that tells you what is
//! missing. Installing is the application's job, and the application tells the
//! engine where the result is through `MUMDIA_THERMO_PARSER`
//! (`components::stamp_env`).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::Serialize;

/// The pinned release, per platform.
///
/// 2.0.0's self-contained builds are chosen over the much smaller 1.4.5 zip
/// deliberately. 1.4.5 is a managed .NET Framework build that needs Mono on Linux,
/// and "install Mono first" is exactly the step that loses the user this feature
/// exists for. The self-contained builds carry their own runtime, so Install is the
/// only step. The engine still accepts a 1.4.x install found on the machine and
/// runs it under Mono (`raw::parser_command`).
///
/// Digests taken from the release assets on 2026-08-31.
struct Asset {
    url: &'static str,
    sha256: &'static str,
    size: u64,
}

fn asset() -> Option<Asset> {
    if cfg!(windows) {
        Some(Asset {
            url: "https://github.com/compomics/ThermoRawFileParser/releases/download/v.2.0.0-dev/ThermoRawFileParser-v.2.0.0-dev-win.zip",
            sha256: "c5629c42c55ff7fbfa1d0ed1ba71fbb30681ab3eb4acd0ed80042c016c6d3602",
            size: 51_090_469,
        })
    } else if cfg!(target_os = "linux") {
        Some(Asset {
            url: "https://github.com/compomics/ThermoRawFileParser/releases/download/v.2.0.0-dev/ThermoRawFileParser-v.2.0.0-dev-linux.zip",
            sha256: "19566762ce6759a93cee9aa4cef50de1e9ae2ad9078bd95826e8e733d4bb0d52",
            size: 48_627_521,
        })
    } else {
        None
    }
}

/// Which converter, if any, a spectra path needs.
///
/// Mirrors `raw::detect` in the engine deliberately rather than importing it: the
/// application does not depend on the engine crate, it spawns the engine binary.
/// The two must agree, which is what `vendor_detection_matches_the_engines_own_rule`
/// asserts against the same cases.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Needs {
    /// Readable as-is.
    Nothing,
    /// Thermo `.raw`: ThermoRawFileParser, which this application installs.
    ThermoParser,
    /// Bruker, SCIEX, Agilent or Waters: msconvert, which it does not.
    Msconvert,
}

/// Classify a spectra path the same way the engine does.
///
/// The `.raw` collision is the part that matters: Thermo's `.raw` is a file and
/// Waters' is a directory, and they route to different converters.
pub fn needs(path: &str) -> Needs {
    let p = std::path::Path::new(path);
    let ext = p
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "raw" if p.is_dir() => Needs::Msconvert,
        "raw" => Needs::ThermoParser,
        "d" | "wiff" | "wiff2" => Needs::Msconvert,
        _ => Needs::Nothing,
    }
}

/// A human label for what the file is, for the note under the picker.
pub fn label(path: &str) -> &'static str {
    let p = std::path::Path::new(path);
    let ext = p
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "raw" if p.is_dir() => "Waters .raw",
        "raw" => "Thermo .raw",
        "d" => {
            if p.join("analysis.tdf").exists() || p.join("analysis.baf").exists() {
                "Bruker .d"
            } else if p.join("AcqData").exists() {
                "Agilent .d"
            } else {
                "Bruker .d"
            }
        }
        "wiff" | "wiff2" => "SCIEX .wiff",
        _ => "mzML",
    }
}

/// Is this path a Thermo `.raw`?
pub fn is_raw(path: &str) -> bool {
    needs(path) == Needs::ThermoParser
}

/// Is msconvert available? Asked of the engine, which owns the search order.
///
/// Shelling out rather than reimplementing: the engine searches
/// `MUMDIA_MSCONVERT`, its own directory, the version-stamped ProteoWizard
/// directories under Program Files, then `PATH`, and a second implementation here
/// would drift from that.
pub fn msconvert_available() -> Option<String> {
    converters(None).and_then(|c| c.msconvert.path)
}

/// One converter as the engine reports it for a configuration: what was configured,
/// what was found, and when nothing was, why.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Converter {
    /// `convert.thermo_raw_parser` or `convert.msconvert` as the engine read it:
    /// `auto`, or a path.
    pub configured: String,
    pub path: Option<String>,
    pub detail: Option<String>,
}

impl Converter {
    /// True when the configuration names a converter rather than leaving the search
    /// to the engine. The engine treats a wrong explicit path as an error and never as
    /// a reason to use a different converter (`raw::ensure_mzml`), because vendor
    /// conversion is not reproducible across converters; preflight says the same.
    pub fn explicit(&self) -> bool {
        !self.configured.is_empty() && self.configured != "auto"
    }
}

/// Both converters, as the engine resolves them for one configuration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Converters {
    pub thermo: Converter,
    pub msconvert: Converter,
}

/// Ask the engine which converters a request with this configuration would use.
///
/// `doctor --json --config <file>`: the engine resolves `convert.thermo_raw_parser`
/// and `convert.msconvert` from that file, its environment (`MUMDIA_THERMO_PARSER`,
/// `MUMDIA_MSCONVERT`), its own directory and `PATH`, so a converter is found here
/// exactly when the run would find it. The probe used to run without the configuration
/// (docs/29 #13): it answered for the defaults, so a converter the configuration named
/// at an off-`PATH` location was reported missing and the search refused, while the
/// engine would have converted without complaint. `None` when the engine could not be
/// asked at all, which is a different situation from "asked, and nothing found".
pub fn converters(config: Option<&str>) -> Option<Converters> {
    let (exe, _) = crate::engine::resolve().ok()?;
    let mut cmd = crate::engine::command(&exe);
    crate::components::stamp_env(&mut cmd);
    // `doctor` exits non-zero when the configuration's interpreters do not resolve,
    // but it prints the report first, and the converter half is what is wanted here.
    let out = cmd.args(doctor_args(config)).output().ok()?;
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    Some(Converters {
        thermo: converter_of(v.get("thermo")),
        msconvert: converter_of(v.get("msconvert")),
    })
}

/// The `doctor` invocation for a configuration, or for the defaults without one.
fn doctor_args(config: Option<&str>) -> Vec<String> {
    let mut args = vec!["doctor".to_string(), "--json".to_string()];
    if let Some(c) = config {
        args.push("--config".to_string());
        args.push(c.to_string());
    }
    args
}

/// One converter entry of the `doctor --json` report; an absent entry is "nothing".
fn converter_of(v: Option<&serde_json::Value>) -> Converter {
    let field = |k: &str| {
        v.and_then(|c| c.get(k))
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
    };
    Converter {
        configured: field("configured").unwrap_or_default(),
        path: field("path"),
        detail: field("detail"),
    }
}

/// What preflight says about the converters the selected files need: hard blockers,
/// and notes worth reading before an hour is spent.
///
/// The rule is the engine's own (`raw::ensure_mzml`), restated rather than imported
/// because this application spawns the engine and does not link it:
///
/// - a Thermo `.raw` goes to ThermoRawFileParser when one is found;
/// - when `convert.thermo_raw_parser` is `auto` and none is found, msconvert converts
///   it instead, with a note, because that is what the engine will do;
/// - a parser the configuration names explicitly and that is not found is an error
///   and never a fallback, matching the engine;
/// - every other vendor format needs msconvert.
///
/// Preflight used to require the Thermo parser specifically, so a machine with only
/// msconvert was refused a search the engine would have run (docs/29 #13).
pub fn converter_verdict(files: &[String], conv: &Converters) -> (Vec<String>, Vec<String>) {
    let mut blockers = Vec::new();
    let mut notes = Vec::new();
    let thermo: Vec<&str> = files
        .iter()
        .filter(|m| needs(m) == Needs::ThermoParser)
        .map(|s| s.as_str())
        .collect();
    let other: Vec<&str> = files
        .iter()
        .filter(|m| needs(m) == Needs::Msconvert)
        .map(|s| s.as_str())
        .collect();

    if let (Some(first), None) = (thermo.first(), &conv.thermo.path) {
        let n = thermo.len();
        if conv.thermo.explicit() {
            let detail = conv
                .thermo
                .detail
                .as_deref()
                .map(|d| format!(": {d}"))
                .unwrap_or_default();
            blockers.push(format!(
                "{n} selected file(s) are Thermo .raw and the converter named in the \
                 configuration was not found (convert.thermo_raw_parser = {}){detail}.\n\
                 Fix that path, or set it to \"auto\" to let MuMDIA search for a converter.\n\
                 First: {first}",
                conv.thermo.configured
            ));
        } else if let Some(ms) = conv.msconvert.path.as_deref() {
            notes.push(format!(
                "{n} Thermo .raw file(s) will be converted with ProteoWizard msconvert ({ms}) \
                 because ThermoRawFileParser was not found.\nInstall it on the Setup screen to \
                 use the licence-free converter instead.\nFirst: {first}"
            ));
        } else {
            blockers.push(format!(
                "{n} selected file(s) are Thermo .raw and no converter is installed.\n\
                 Install ThermoRawFileParser on the Setup screen, install ProteoWizard \
                 msconvert, or convert them to mzML yourself.\nFirst: {first}"
            ));
        }
    }
    if let (Some(first), None) = (other.first(), &conv.msconvert.path) {
        blockers.push(format!(
            "{} selected file(s) need ProteoWizard msconvert, which was not found.\n\
             MuMDIA does not install it; see the Setup screen.\nFirst: {first} ({})",
            other.len(),
            label(first)
        ));
    }
    (blockers, notes)
}

/// State of the converter: installed or not, and the last install's progress.
#[derive(Serialize, Clone, Debug, Default)]
pub struct Status {
    /// Absolute path to the managed converter, if it is installed.
    pub path: Option<String>,
    /// True when the converter is present and reported its own version.
    pub ready: bool,
    pub version: Option<String>,
    /// `idle` | `installing` | `done` | `failed`
    pub install_status: String,
    pub percent: u8,
    pub step: String,
    pub log: Vec<String>,
    pub error: Option<String>,
    /// Whether a download is published for this platform at all.
    pub available: bool,
    pub download_bytes: u64,
}

#[derive(Default)]
pub struct Installer {
    state: Mutex<Status>,
}

impl Installer {
    /// Probe the disk, preserving any terminal state from an install.
    ///
    /// Same reasoning as `components::Installer::refresh`: a fresh probe carries
    /// `idle`, and letting that overwrite `done` or `failed` would lose the outcome
    /// a caller is watching for.
    pub fn refresh(&self) -> Status {
        let a = asset();
        let mut fresh = Status {
            available: a.is_some(),
            download_bytes: a.map(|x| x.size).unwrap_or(0),
            install_status: "idle".into(),
            ..Default::default()
        };
        if let Some(p) = crate::components::thermo_parser() {
            fresh.path = Some(p.display().to_string());
            let (ok, version) = probe(&p);
            fresh.ready = ok;
            fresh.version = version;
        }
        if let Ok(mut s) = self.state.lock() {
            let keep = s.install_status.clone();
            let log = s.log.clone();
            let err = s.error.clone();
            let step = s.step.clone();
            let pct = s.percent;
            *s = fresh;
            if keep != "idle" {
                s.install_status = keep;
                s.log = log;
                s.step = step;
                s.percent = pct;
                if s.error.is_none() {
                    s.error = err;
                }
            }
            return s.clone();
        }
        fresh
    }

    fn log(&self, line: String) {
        if let Ok(mut s) = self.state.lock() {
            s.log.push(line);
            if s.log.len() > 300 {
                let drop = s.log.len() - 300;
                s.log.drain(0..drop);
            }
        }
    }

    fn fail(&self, msg: String) {
        if let Ok(mut s) = self.state.lock() {
            s.install_status = "failed".into();
            s.error = Some(msg);
        }
    }
}

/// Run the converter with no arguments and read its version banner.
///
/// Verified by execution rather than by the file existing, for the same reason as
/// everywhere else here: a self-contained .NET build that cannot start on this
/// machine is not a working converter, and finding that out at the start of a
/// search is worse than finding it out now. ThermoRawFileParser exits non-zero when
/// given no input, so the exit code is not the signal; the banner is.
/// Does this converter execute on this machine?
///
/// Separate from "is it on disk": a half-unpacked or non-startable build is a file
/// that is not a converter, and the difference decides whether the engine should be
/// told about it at all (`components::runnable_thermo_parser`).
pub fn runs(exe: &std::path::Path) -> bool {
    probe(exe).0
}

/// How long a probe may take before the converter is assumed not to be a console
/// program. Same reasoning and same value as `diann::PROBE_TIMEOUT`.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

fn probe(exe: &std::path::Path) -> (bool, Option<String>) {
    // Bounded, and for the reason `diann.rs` already documents: `Command::output`
    // waits for the child to exit, and this runs on the Tauri main thread from
    // `thermo_status` -- including a 700 ms poll during install -- and from
    // `preflight`. A self-contained .NET build that stalls on a missing runtime, an
    // antivirus scan or a cold network share froze the window with no recovery. The
    // sibling probe was bounded and this one was not.
    let out = match crate::engine::command(exe)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(mut child) => {
            let started = std::time::Instant::now();
            loop {
                match child.try_wait() {
                    Ok(Some(_)) | Err(_) => break,
                    Ok(None) => {}
                }
                if started.elapsed() >= PROBE_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    return (false, None);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            match child.wait_with_output() {
                Ok(o) => o,
                Err(_) => return (false, None),
            }
        }
        Err(_) => return (false, None),
    };
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    let version = text
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("ThermoRawFileParser"))
        .map(|l| l.to_string());
    // Usage text also proves it started. Either is acceptable evidence; neither
    // being present means it did not run.
    let started = version.is_some() || text.contains("--input") || text.contains("Usage");
    (started, version)
}

/// Download and unpack the converter.
pub fn install(installer: Arc<Installer>) -> Result<(), String> {
    let Some(a) = asset() else {
        return Err(
            "no ThermoRawFileParser build is published for this platform; convert to \
             mzML with msconvert instead"
                .into(),
        );
    };
    {
        let mut s = installer
            .state
            .lock()
            .map_err(|_| "internal state is poisoned".to_string())?;
        if s.install_status == "installing" {
            return Err("an install is already running".into());
        }
        s.install_status = "installing".into();
        s.step = "downloading ThermoRawFileParser".into();
        s.percent = 0;
        s.log.clear();
        s.error = None;
    }

    std::thread::spawn(move || {
        let dir = crate::components::data_dir();
        if let Err(e) = std::fs::create_dir_all(&dir) {
            installer.fail(format!("cannot create {}: {e}", dir.display()));
            return;
        }
        let archive = dir.join("ThermoRawFileParser.zip");
        if let Err(e) = download(&installer, &a, &archive) {
            installer.fail(e);
            return;
        }

        if let Ok(mut s) = installer.state.lock() {
            s.step = "unpacking".into();
        }
        let target = crate::components::thermo_dir();
        // A previous partial unpack would otherwise leave a mixture of two
        // releases' assemblies in one directory, which is the kind of failure that
        // presents as a mysterious runtime error rather than a missing file.
        let _ = std::fs::remove_dir_all(&target);
        if let Err(e) = unzip(&archive, &target) {
            installer.fail(e);
            return;
        }
        let _ = std::fs::remove_file(&archive);

        let Some(exe) = crate::components::thermo_parser() else {
            installer.fail(format!(
                "the archive unpacked but no ThermoRawFileParser executable was found in {}",
                target.display()
            ));
            return;
        };
        // The zip does not carry the Unix execute bit for the launcher.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(md) = std::fs::metadata(&exe) {
                let mut perms = md.permissions();
                perms.set_mode(perms.mode() | 0o755);
                let _ = std::fs::set_permissions(&exe, perms);
            }
        }

        let (ok, version) = probe(&exe);
        if !ok {
            installer.fail(format!(
                "{} unpacked but did not run on this machine",
                exe.display()
            ));
            return;
        }
        installer.log(format!("== installed: {}", version.unwrap_or_default()));
        if let Ok(mut s) = installer.state.lock() {
            s.install_status = "done".into();
            s.step = "installed".into();
            s.percent = 100;
        }
    });

    Ok(())
}

/// Stream the asset to `dest`, hashing as it goes, and verify the digest.
///
/// Deliberately a near-twin of `diann::download_verified`: the two differ only in
/// what they do afterwards, and sharing one function would have meant a parameter
/// that means "and also accept a licence", which is not a thing this one has.
fn download(installer: &Arc<Installer>, a: &Asset, dest: &std::path::Path) -> Result<(), String> {
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
    }
    std::io::Write::flush(&mut file).map_err(|e| e.to_string())?;
    drop(file);

    let got = crate::components::hex(hasher.finalize());
    if got != a.sha256 {
        let _ = std::fs::remove_file(dest);
        return Err(format!(
            "the downloaded file does not match the expected checksum and has been \
             deleted. Expected {}, got {got}.",
            a.sha256
        ));
    }
    installer.log(format!("== checksum verified ({total} bytes)"));
    Ok(())
}

/// Unpack a zip into `target`.
///
/// Entry names are validated rather than trusted. A zip is an untrusted input even
/// from a trusted publisher, and an entry named `../../x` would otherwise write
/// outside the target directory.
fn unzip(archive: &std::path::Path, target: &PathBuf) -> Result<(), String> {
    let file = std::fs::File::open(archive)
        .map_err(|e| format!("cannot read {}: {e}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
        .map_err(|e| format!("{} is not a readable zip: {e}", archive.display()))?;
    std::fs::create_dir_all(target)
        .map_err(|e| format!("cannot create {}: {e}", target.display()))?;

    for i in 0..zip.len() {
        let mut entry = zip
            .by_index(i)
            .map_err(|e| format!("cannot read entry {i}: {e}"))?;
        // `enclosed_name` rejects absolute paths and any `..` component, which is
        // exactly the traversal check this needs.
        let Some(rel) = entry.enclosed_name() else {
            return Err(format!(
                "the archive contains an unsafe path ({}); refusing to unpack it",
                entry.name()
            ));
        };
        let out = target.join(rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out)
                .map_err(|e| format!("cannot create {}: {e}", out.display()))?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        let mut w = std::fs::File::create(&out)
            .map_err(|e| format!("cannot write {}: {e}", out.display()))?;
        std::io::copy(&mut entry, &mut w)
            .map_err(|e| format!("cannot write {}: {e}", out.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Some(mode) = entry.unix_mode() {
                let _ = std::fs::set_permissions(&out, std::fs::Permissions::from_mode(mode));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(thermo: (&str, Option<&str>), msconvert: Option<&str>) -> Converters {
        Converters {
            thermo: Converter {
                configured: thermo.0.into(),
                path: thermo.1.map(String::from),
                detail: thermo
                    .1
                    .is_none()
                    .then(|| "no ThermoRawFileParser found".to_string()),
            },
            msconvert: Converter {
                configured: "auto".into(),
                path: msconvert.map(String::from),
                detail: None,
            },
        }
    }

    fn files(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_thermo_file_with_only_msconvert_is_allowed_with_a_note() {
        // The engine's fallback for a parser left at `auto` (docs/29 #13): preflight
        // used to demand the parser and block this.
        let (blockers, notes) = converter_verdict(
            &files(&["a.raw"]),
            &conv(("auto", None), Some("C:/pwiz/msconvert.exe")),
        );
        assert!(blockers.is_empty(), "{blockers:?}");
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("msconvert"), "{}", notes[0]);
    }

    #[test]
    fn an_explicit_parser_that_is_missing_blocks_even_with_msconvert_present() {
        // The engine's rule: a configured path that is wrong is an error, not a reason
        // to convert with a program the configuration did not name.
        let (blockers, notes) = converter_verdict(
            &files(&["a.raw"]),
            &conv(
                ("D:/tools/ThermoRawFileParser.exe", None),
                Some("msconvert"),
            ),
        );
        assert_eq!(blockers.len(), 1, "{blockers:?}");
        assert!(
            blockers[0].contains("D:/tools/ThermoRawFileParser.exe"),
            "{}",
            blockers[0]
        );
        assert!(
            blockers[0].contains("no ThermoRawFileParser found"),
            "{}",
            blockers[0]
        );
        assert!(notes.is_empty(), "{notes:?}");
    }

    #[test]
    fn converters_the_configuration_names_off_path_are_enough() {
        // The probe carries the request's configuration, so a converter found only
        // through it arrives with a path, and nothing is blocked.
        let mut c = conv(
            (
                "D:/tools/ThermoRawFileParser.exe",
                Some("D:/tools/ThermoRawFileParser.exe"),
            ),
            Some("E:/pwiz/msconvert.exe"),
        );
        c.msconvert.configured = "E:/pwiz/msconvert.exe".into();
        assert!(c.thermo.explicit() && c.msconvert.explicit());
        let (blockers, notes) = converter_verdict(&files(&["a.raw", "b.d"]), &c);
        assert!(
            blockers.is_empty() && notes.is_empty(),
            "{blockers:?} {notes:?}"
        );
    }

    #[test]
    fn nothing_installed_blocks_thermo_and_bruker_and_leaves_mzml_alone() {
        let (blockers, _) = converter_verdict(
            &files(&["a.raw", "b.d", "c.raw"]),
            &conv(("auto", None), None),
        );
        assert_eq!(blockers.len(), 2, "{blockers:?}");
        assert!(
            blockers[0].starts_with("2 selected file(s) are Thermo"),
            "{}",
            blockers[0]
        );
        assert!(blockers[0].contains("msconvert"), "{}", blockers[0]);
        assert!(blockers[1].contains("msconvert"), "{}", blockers[1]);
        let (b, n) = converter_verdict(&files(&["x.mzML"]), &conv(("auto", None), None));
        assert!(b.is_empty() && n.is_empty());
    }

    #[test]
    fn the_probe_carries_the_requests_configuration() {
        assert_eq!(doctor_args(None), vec!["doctor", "--json"]);
        assert_eq!(
            doctor_args(Some("C:/cfg/run.json")),
            vec!["doctor", "--json", "--config", "C:/cfg/run.json"]
        );
    }

    #[test]
    fn the_report_is_read_as_the_engine_writes_it() {
        let v = serde_json::json!({
            "thermo": {"status": "none", "configured": "auto", "path": null,
                       "detail": "no ThermoRawFileParser found"},
            "msconvert": {"status": "ok", "configured": "auto",
                          "path": "/opt/pwiz/msconvert", "detail": null}
        });
        let t = converter_of(v.get("thermo"));
        assert_eq!(t.configured, "auto");
        assert!(!t.explicit());
        assert_eq!(t.path, None);
        assert_eq!(t.detail.as_deref(), Some("no ThermoRawFileParser found"));
        let m = converter_of(v.get("msconvert"));
        assert_eq!(m.path.as_deref(), Some("/opt/pwiz/msconvert"));
        assert_eq!(converter_of(None), Converter::default());
    }

    #[test]
    fn the_download_is_pinned_to_the_publishers_own_release() {
        let Some(a) = asset() else { return };
        assert!(
            a.url
                .starts_with("https://github.com/compomics/ThermoRawFileParser/releases/download/"),
            "the asset must come from the publisher's own release: {}",
            a.url
        );
        assert!(!a.url.contains("latest"), "pinned, never latest: {}", a.url);
        assert_eq!(a.sha256.len(), 64);
        assert!(a.sha256.chars().all(|c| c.is_ascii_hexdigit()));
        // The self-contained builds are tens of megabytes; the 3.6 MB managed zip
        // would mean requiring Mono, which this deliberately does not.
        assert!(
            a.size > 20_000_000,
            "expected a self-contained build, got {} bytes",
            a.size
        );
    }

    #[test]
    fn vendor_detection_matches_the_engines_own_rule() {
        // If these two disagree, the interface either blocks a file the engine would
        // have converted, or admits one it will not. Mirrors `raw::detect`.
        assert_eq!(needs("a.raw"), Needs::ThermoParser);
        assert_eq!(needs("a.RAW"), Needs::ThermoParser);
        assert_eq!(needs("/d/LFQ_01.Raw"), Needs::ThermoParser);
        assert_eq!(needs("a.mzML"), Needs::Nothing);
        assert_eq!(needs("a.mzml"), Needs::Nothing);
        assert_eq!(needs("noextension"), Needs::Nothing);
        // The formats msconvert owns.
        assert_eq!(needs("a.d"), Needs::Msconvert);
        assert_eq!(needs("a.wiff"), Needs::Msconvert);
        assert_eq!(needs("a.wiff2"), Needs::Msconvert);

        assert!(is_raw("a.raw"));
        assert!(!is_raw("a.d"));
    }

    #[test]
    fn a_raw_directory_is_waters_and_routes_to_msconvert() {
        // The collision. A Waters `.raw` directory sent to ThermoRawFileParser fails
        // with something unhelpful, and the interface would have offered the wrong
        // Install button.
        // Unique per process, like every other temp fixture in the workspace: a fixed name
        // plus the delete-on-entry below races a second `cargo test` on one machine
        // (docs/14). Hit twice during the 0.2.0 and 0.3.0 releases.
        let d = std::env::temp_dir().join(format!("mumdia-thermo-waters-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("waters.raw")).unwrap();
        let p = d.join("waters.raw");
        let s = p.to_str().unwrap();
        assert_eq!(needs(s), Needs::Msconvert);
        assert_eq!(label(s), "Waters .raw");
        assert!(!is_raw(s));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_d_directory_is_labelled_by_its_contents() {
        let d = std::env::temp_dir().join(format!("mumdia-thermo-dlabel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let tims = d.join("tims.d");
        std::fs::create_dir_all(&tims).unwrap();
        std::fs::write(tims.join("analysis.tdf"), b"x").unwrap();
        assert_eq!(label(tims.to_str().unwrap()), "Bruker .d");

        let ag = d.join("ag.d");
        std::fs::create_dir_all(ag.join("AcqData")).unwrap();
        assert_eq!(label(ag.to_str().unwrap()), "Agilent .d");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_managed_converter_lives_under_the_per_user_data_directory() {
        let d = crate::components::thermo_dir();
        assert!(d.starts_with(crate::components::data_dir()));
    }

    #[test]
    fn a_zip_entry_that_escapes_the_target_is_refused() {
        // A zip is untrusted input even from a trusted publisher. Without the
        // `enclosed_name` check this entry would be written outside the target.
        let dir =
            std::env::temp_dir().join(format!("mumdia-thermo-traversal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let archive = dir.join("evil.zip");

        {
            let f = std::fs::File::create(&archive).unwrap();
            let mut w = zip::ZipWriter::new(f);
            let opts: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            // `start_file` rejects some names, so the raw name goes in through the
            // path-less API to build the hostile archive this must refuse.
            w.start_file("../escaped.txt", opts).unwrap();
            std::io::Write::write_all(&mut w, b"x").unwrap();
            w.finish().unwrap();
        }

        let target = dir.join("out");
        let err = unzip(&archive, &target);
        // Either the archive is refused outright, or the entry was normalised into
        // the target. What must never happen is a file appearing beside `out`.
        assert!(
            !dir.join("escaped.txt").exists(),
            "an entry escaped the target directory"
        );
        if let Err(msg) = err {
            assert!(msg.contains("unsafe path"), "{msg}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_normal_zip_round_trips_with_its_directory_structure() {
        let dir = std::env::temp_dir().join(format!("mumdia-thermo-unzip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let archive = dir.join("ok.zip");
        {
            let f = std::fs::File::create(&archive).unwrap();
            let mut w = zip::ZipWriter::new(f);
            let opts: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            w.start_file("ThermoRawFileParser.exe", opts).unwrap();
            std::io::Write::write_all(&mut w, b"launcher").unwrap();
            w.start_file("lib/dep.dll", opts).unwrap();
            std::io::Write::write_all(&mut w, b"dep").unwrap();
            w.finish().unwrap();
        }
        let target = dir.join("out");
        unzip(&archive, &target).unwrap();
        assert_eq!(
            std::fs::read(target.join("ThermoRawFileParser.exe")).unwrap(),
            b"launcher"
        );
        assert_eq!(std::fs::read(target.join("lib/dep.dll")).unwrap(), b"dep");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
