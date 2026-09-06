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
    converter_path("msconvert")
}

/// The Thermo converter the ENGINE would use, which is not the same question as
/// whether this application installed one.
///
/// `Installer::refresh` only ever looks in `data_dir()/ThermoRawFileParser/`, while
/// the engine's `raw::locate_parser` also accepts an explicit
/// `convert.thermo_raw_parser`, `MUMDIA_THERMO_PARSER`, a binary beside the engine or
/// one on `PATH`. Preflight asked the narrow question and hard-blocked users whose
/// converter the engine would have found perfectly well -- and the GUI was the only
/// path that refused. `doctor --json` already reports both converters from the
/// engine's own search; nothing was reading the `thermo` half of it.
pub fn engine_thermo_parser() -> Option<String> {
    converter_path("thermo")
}

/// One `doctor --json` probe, shared by both converter questions.
fn converter_path(key: &str) -> Option<String> {
    let (exe, _) = crate::engine::resolve().ok()?;
    let mut cmd = crate::engine::command(&exe);
    crate::components::stamp_env(&mut cmd);
    let out = cmd.arg("doctor").arg("--json").output().ok()?;
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    v.get(key)?.get("path")?.as_str().map(|s| s.to_string())
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

    let got = format!("{:x}", hasher.finalize());
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
        let d = std::env::temp_dir().join("mumdia-thermo-waters");
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
        let d = std::env::temp_dir().join("mumdia-thermo-dlabel");
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
        let dir = std::env::temp_dir().join("mumdia-thermo-traversal");
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
        let dir = std::env::temp_dir().join("mumdia-thermo-unzip");
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
