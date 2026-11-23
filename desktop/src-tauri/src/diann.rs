//! Locating and driving DIA-NN.
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
//! edition restricted to non-profit academic research.
//!
//! So MuMDIA never ships, mirrors, proxies or downloads DIA-NN. It **locates** a
//! DIA-NN the user installed and licensed themselves, and drives it. Running a
//! program already on the machine redistributes nothing.
//!
//! # What it produces
//!
//! DIA-NN predicts a spectral library from a FASTA; the two sidecars already in
//! `scripts/` convert it to the MuMDIA schema and add the decoy population. The
//! command line is the one recorded in `README.md`, so what this runs is what a
//! user following the manual instructions would have typed.

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

Accepting this notice is not a substitute for DIA-NN's own licence terms.

DIA-NN's version affects the library it predicts, and therefore affects your \
results. Record the version you used.";

/// Where a located DIA-NN, and the acknowledgement, are remembered.
///
/// Per-user in both scopes (`crate::storage::Layout::user_tools`): the licence answer is
/// the user's own, and in the machine scope the components folder beside which this file
/// sits in the user scope is shared and read-only.
fn state_file() -> PathBuf {
    crate::storage::user_tools_dir().join(STATE_FILE_NAME)
}

/// Names under the data directory, shared with `components::inventory` so that the
/// removal offered on the Setup screen cannot drift out of step with what is written.
pub const STATE_FILE_NAME: &str = "diann.json";
pub const MANAGED_DIR_NAME: &str = "diann-1.8.1";
pub const LIBRARY_CACHE_NAME: &str = "libraries";

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
struct Saved {
    path: Option<String>,
    /// The user's own answer to the licence notice: `None` until they tick or untick
    /// it. An older state file stored a plain boolean, which reads as `Some`.
    #[serde(default)]
    licence_acknowledged: Option<bool>,
}

/// Whether the licence counts as acknowledged, and whether that was assumed.
///
/// An explicit answer always wins. Without one, a DIA-NN that is installed and runs
/// is taken as acknowledged: someone who installed DIA-NN has accepted its licence
/// from the vendor, and asking again before the first library build only adds a step.
fn effective_licence(saved: Option<bool>, runs: bool) -> (bool, bool) {
    match saved {
        Some(answer) => (answer, false),
        None => (runs, runs),
    }
}

fn load() -> Saved {
    std::fs::read_to_string(state_file())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

fn store(s: &Saved) -> Result<(), String> {
    let dir = crate::storage::user_tools_dir();
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
    /// True when `licence_acknowledged` was assumed from a working installation
    /// rather than answered by the user, so the interface can say so.
    pub licence_assumed: bool,
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
    let mut status = detect_unacknowledged(&saved);
    let (ack, assumed) = effective_licence(saved.licence_acknowledged, status.runs);
    status.licence_acknowledged = ack;
    status.licence_assumed = assumed;
    status
}

/// `detect` before the licence answer is applied to it.
fn detect_unacknowledged(saved: &Saved) -> Status {
    let mut status = Status::default();

    let cands = candidates(saved);
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
/// predicted a whole proteome, ran the re-export, and only then
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
    saved.licence_acknowledged = Some(accepted);
    store(&saved)?;
    Ok(detect())
}

/// What to predict, and how. The defaults match `README.md`'s worked example.
#[derive(Deserialize, Clone, Debug)]
pub struct BuildRequest {
    pub fasta: String,
    pub out_dir: String,
    /// The cleavage preset (`enzymes.json`: `trypsin_p`, `lys_c`, `non_specific`, ...,
    /// `custom`). Trypsin/P by default, the rule this builder always used.
    #[serde(default = "d_enzyme")]
    pub enzyme: String,
    /// The rule of `enzyme = "custom"`, in DIA-NN `--cut` syntax.
    #[serde(default)]
    pub cleavage_rule: Option<String>,
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
    /// The Search screen's full modification choice. When both lists are empty the two
    /// flags above decide, which is how every earlier request (and cache entry) reads.
    #[serde(default)]
    pub fixed_mods: Vec<ModSite>,
    #[serde(default)]
    pub variable_mods: Vec<ModSite>,
    /// Variable modifications per peptide (DIA-NN `--var-mods`).
    #[serde(default = "d_max_var")]
    pub max_variable_mods: u32,
    /// Precursor m/z limits (DIA-NN `--min-pr-mz` / `--max-pr-mz`). Unset by default, when
    /// the builder applies its own defaults and the arguments are not passed; set, each one
    /// is passed and is part of the library's cache key.
    #[serde(default)]
    pub min_pr_mz: Option<f64>,
    #[serde(default)]
    pub max_pr_mz: Option<f64>,
    /// Fragment m/z limits (DIA-NN `--min-fr-mz` / `--max-fr-mz`), likewise. They are also
    /// passed to the decoys' prediction, so targets and decoys keep fragments from the same
    /// range.
    #[serde(default)]
    pub min_fr_mz: Option<f64>,
    #[serde(default)]
    pub max_fr_mz: Option<f64>,
}

impl BuildRequest {
    /// The four m/z limits of the request.
    pub fn mz_limits(&self) -> crate::mzlimits::MzLimits {
        crate::mzlimits::MzLimits {
            precursor_min: self.min_pr_mz,
            precursor_max: self.max_pr_mz,
            fragment_min: self.min_fr_mz,
            fragment_max: self.max_fr_mz,
        }
    }

    /// The precursor and fragment m/z limits as DIA-NN arguments, each only when set;
    /// refused when one is not a plausible m/z or a range is empty (`mzlimits`).
    fn mz_args(&self) -> Result<Vec<String>, String> {
        self.mz_limits().check()?;
        let mut args = Vec::new();
        for (flag, v) in [
            ("--min-pr-mz", self.min_pr_mz),
            ("--max-pr-mz", self.max_pr_mz),
            ("--min-fr-mz", self.min_fr_mz),
            ("--max-fr-mz", self.max_fr_mz),
        ] {
            if let Some(v) = v {
                args.push(flag.to_string());
                args.push(v.to_string());
            }
        }
        Ok(args)
    }

    /// The search space of the request, for a message about a library that came out
    /// empty.
    fn describe(&self) -> String {
        format!(
            "peptide length {} to {}, charge {} to {}, {} missed cleavage(s), {}",
            self.min_pep_len,
            self.max_pep_len,
            self.min_charge,
            self.max_charge,
            self.missed_cleavages,
            self.mz_limits().describe()
        )
    }

    /// The fragment m/z arguments alone, as `predict_decoys.py --diann-arg` values, so the
    /// decoys' spectra are predicted over the same fragment range as the targets'.
    fn decoy_diann_args(&self) -> Vec<String> {
        [
            ("--min-fr-mz", self.min_fr_mz),
            ("--max-fr-mz", self.max_fr_mz),
        ]
        .into_iter()
        .filter_map(|(flag, v)| v.map(|v| (flag, v)))
        .flat_map(|(flag, v)| [format!("--diann-arg={flag}"), format!("--diann-arg={v}")])
        .collect()
    }
}

/// One modification on one residue, by catalogue name, or on a terminus.
#[derive(Deserialize, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModSite {
    pub name: String,
    pub residue: char,
    /// `protein_n_term` or `peptide_n_term` for a terminal site, whose residue is `*`;
    /// empty (or `anywhere`) for a residue, which is how every earlier request reads.
    #[serde(default)]
    pub position: String,
}

impl ModSite {
    /// DIA-NN's site token for a terminal site: `*n` for the protein N-terminus, `n` for
    /// the peptide N-terminus (the `--var-mod`/`--fixed-mod` site syntax).
    fn terminal_token(&self) -> Option<&'static str> {
        match self.position.as_str() {
            "protein_n_term" => Some("*n"),
            "peptide_n_term" => Some("n"),
            _ => None,
        }
    }
}

/// The modifications a request asks for, normalised: catalogue order is irrelevant, the
/// two legacy flags become sites, and duplicates go.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModPlan {
    pub fixed: Vec<ModSite>,
    pub variable: Vec<ModSite>,
    pub max_variable: u32,
}

fn site(name: &str, residue: char) -> ModSite {
    ModSite {
        name: name.into(),
        residue,
        position: String::new(),
    }
}

impl ModPlan {
    pub fn of(req: &BuildRequest) -> Result<Self, String> {
        let (mut fixed, mut variable) = if req.fixed_mods.is_empty() && req.variable_mods.is_empty()
        {
            (
                if req.carbamidomethyl {
                    vec![site("Carbamidomethyl", 'C')]
                } else {
                    vec![]
                },
                if req.oxidation {
                    vec![site("Oxidation", 'M')]
                } else {
                    vec![]
                },
            )
        } else {
            (req.fixed_mods.clone(), req.variable_mods.clone())
        };
        for m in fixed.iter_mut().chain(variable.iter_mut()) {
            if m.position == "anywhere" {
                m.position.clear();
            }
        }
        for m in fixed.iter().chain(&variable) {
            if crate::mods::lookup(&m.name).is_none() {
                return Err(format!(
                    "{} is not a modification MuMDIA knows by name",
                    m.name
                ));
            }
            if m.position.is_empty() {
                if !m.residue.is_ascii_uppercase() {
                    return Err(format!("{} on '{}': not a residue", m.name, m.residue));
                }
            } else if m.terminal_token().is_none() {
                return Err(format!(
                    "{} at '{}': the terminal positions are protein_n_term and peptide_n_term",
                    m.name, m.position
                ));
            } else if m.residue != '*' {
                return Err(format!(
                    "{} at {} names residue '{}'; a terminal site takes '*'",
                    m.name, m.position, m.residue
                ));
            }
        }
        // One group per terminus, as the engine requires: at most one fixed N-terminal site,
        // and none variable beside it.
        let n_fixed = fixed
            .iter()
            .filter(|m| m.terminal_token().is_some())
            .count();
        let n_var = variable
            .iter()
            .filter(|m| m.terminal_token().is_some())
            .count();
        if n_fixed > 1 || (n_fixed == 1 && n_var > 0) {
            return Err(
                "a peptide N-terminus carries one group: at most one fixed N-terminal \
                 modification, and no variable one beside it"
                    .into(),
            );
        }
        fixed.sort();
        fixed.dedup();
        variable.sort();
        variable.dedup();
        Ok(Self {
            fixed,
            variable,
            max_variable: req.max_variable_mods,
        })
    }

    /// The selection every library before the full choice was built with: carbamidomethyl
    /// on C fixed and oxidation on M variable, each optional, one variable per peptide. Its
    /// cache key, arguments and import are exactly the earlier ones.
    pub fn is_legacy(&self) -> bool {
        self.fixed
            .iter()
            .all(|m| *m == site("Carbamidomethyl", 'C'))
            && self.variable.iter().all(|m| *m == site("Oxidation", 'M'))
            && (self.variable.is_empty() || self.max_variable == 1)
    }

    /// Canonical text for the cache key. A residue site is `Name@R` as it always was; a
    /// terminal site is `Name@*n` or `Name@n`, its DIA-NN site token.
    fn key(&self) -> String {
        let list = |v: &[ModSite]| {
            v.iter()
                .map(|m| match m.terminal_token() {
                    Some(t) => format!("{}@{t}", m.name),
                    None => format!("{}@{}", m.name, m.residue),
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            "fixed={};var={};max={}",
            list(&self.fixed),
            list(&self.variable),
            self.max_variable
        )
    }

    /// `name -> residues` in name order, for one DIA-NN argument per modification. A
    /// terminal site is an argument of its own after the residue one, with its site token
    /// (`*n`, `n`), so a residue list never mixes with a terminus.
    fn grouped(v: &[ModSite]) -> Vec<(&'static crate::mods::Modification, String)> {
        let mut by: std::collections::BTreeMap<(&str, &str), String> =
            std::collections::BTreeMap::new();
        for m in v {
            match m.terminal_token() {
                Some(t) => by.entry((m.name.as_str(), t)).or_default().push_str(t),
                None => by.entry((m.name.as_str(), "")).or_default().push(m.residue),
            }
        }
        by.into_iter()
            .filter_map(|((n, _), r)| crate::mods::lookup(n).map(|m| (m, r)))
            .collect()
    }

    /// `UniMod:<id>,<mass>` for every modification of the plan, in accession order: what each
    /// DIA-NN run after the prediction declares with `--mod` (see `mod_declarations`).
    pub fn declarations(&self) -> Vec<String> {
        declarations(
            self.fixed
                .iter()
                .chain(&self.variable)
                .filter_map(|m| crate::mods::lookup(&m.name)),
        )
    }

    /// The UniMod accessions the import must keep.
    pub fn unimod_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .fixed
            .iter()
            .chain(&self.variable)
            .filter_map(|m| crate::mods::lookup(&m.name).map(|c| c.unimod))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

/// `UniMod:<id>,<mass>` per modification, in accession order, once each, the mass written as
/// the prediction's `--fixed-mod`/`--var-mod` writes it.
fn declarations<'a>(mods: impl Iterator<Item = &'a crate::mods::Modification>) -> Vec<String> {
    let by: std::collections::BTreeMap<u32, f64> = mods.map(|m| (m.unimod, m.mass)).collect();
    by.into_iter()
        .map(|(id, mass)| format!("UniMod:{id},{mass:.6}"))
        .collect()
}

/// `--mod <declaration>` per declaration.
///
/// DIA-NN knows a few modifications by itself (UniMod 4, 35 and 21 among them) and must be
/// told the mass of every other one, in every run that reads a library carrying it, not
/// only in the run that predicted it. A re-export without the declaration logs "unknown
/// modifications found in the library" and turns its fragmentation-based algorithms off; a
/// prediction without it refuses ("cannot generate a predicted library without knowing the
/// modification masses") and its re-export saves 0 precursors, which is how methyl
/// (UniMod:34) and trimethyl (UniMod:37) variable modifications broke the decoy step.
/// `--mod` only declares a mass: unlike `--var-mod` or `--fixed-mod` it adds no
/// peptidoform, and DIA-NN 2.7 writes the same library with it as without it.
fn mod_declaration_args(mods: &[String]) -> Vec<String> {
    mods.iter()
        .flat_map(|d| ["--mod".to_string(), d.clone()])
        .collect()
}

fn d_max_var() -> u32 {
    1
}

fn d_missed() -> u32 {
    1
}
fn d_enzyme() -> String {
    crate::enzymes::DEFAULT_ENZYME.to_string()
}

/// The DIA-NN `--cut` rule of a request: its preset's, or its custom rule, checked.
pub fn cut_rule(req: &BuildRequest) -> Result<String, String> {
    crate::enzymes::resolve(&req.enzyme, req.cleavage_rule.as_deref())
}

/// The rule every library before the cleavage choice was built with; it keeps the cache
/// key it always had.
const LEGACY_CUT: &str = "K*,R*";
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
/// peptides and reusing one would silently reinstate the defect the flag fixes. Raised to 3
/// when the decoys' spectra became DIA-NN predictions (`predict_decoys.py`): an entry
/// written before carries copied-intensity decoys, which made the reported FDR optimistic.
const RECIPE_VERSION: u32 = 3;

pub fn library_cache_dir(req: &BuildRequest, diann_version: &str) -> Result<PathBuf, String> {
    use sha2::{Digest, Sha256};
    let plan = ModPlan::of(req)?;
    let fasta = std::fs::read(&req.fasta).map_err(|e| format!("cannot read {}: {e}", req.fasta))?;
    let mut h = Sha256::new();
    h.update(&fasta);
    // The standard selection keeps the key it always had, so libraries built before the
    // full modification choice existed are still found; any other selection adds itself.
    let (cam, ox) = if plan.is_legacy() {
        (!plan.fixed.is_empty(), !plan.variable.is_empty())
    } else {
        (req.carbamidomethyl, req.oxidation)
    };
    h.update(
        format!(
            "|recipe={}|mc={}|len={}-{}|z={}-{}|cam={}|ox={}|diann={}",
            RECIPE_VERSION,
            req.missed_cleavages,
            req.min_pep_len,
            req.max_pep_len,
            req.min_charge,
            req.max_charge,
            cam,
            ox,
            diann_version,
        )
        .as_bytes(),
    );
    if !plan.is_legacy() {
        h.update(format!("|mods={}", plan.key()).as_bytes());
    }
    // Likewise the cleavage: Trypsin/P keeps the key it always had.
    let cut = cut_rule(req)?;
    if cut != LEGACY_CUT {
        h.update(format!("|cut={cut}").as_bytes());
    }
    // And the m/z limits: a library built without them keeps the key it always had.
    let end = |v: Option<f64>| v.map(|v| v.to_string()).unwrap_or_default();
    if req.min_pr_mz.is_some() || req.max_pr_mz.is_some() {
        h.update(format!("|prmz={}-{}", end(req.min_pr_mz), end(req.max_pr_mz)).as_bytes());
    }
    if req.min_fr_mz.is_some() || req.max_fr_mz.is_some() {
        h.update(format!("|frmz={}-{}", end(req.min_fr_mz), end(req.max_fr_mz)).as_bytes());
    }
    let key = crate::components::hex(h.finalize());
    Ok(crate::components::data_dir()
        .join(LIBRARY_CACHE_NAME)
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
    let mut body = serde_json::json!({
        "fasta": req.fasta,
        "diann_version": version,
        "recipe_version": RECIPE_VERSION,
        "cleavage": cut_rule(req).ok(),
        "missed_cleavages": req.missed_cleavages,
        "pep_len": [req.min_pep_len, req.max_pep_len],
        "charge": [req.min_charge, req.max_charge],
        "carbamidomethyl": req.carbamidomethyl,
        "oxidation": req.oxidation,
        "modifications": ModPlan::of(req).ok().map(|p| p.key()),
    });
    for (name, v) in [
        ("min_pr_mz", req.min_pr_mz),
        ("max_pr_mz", req.max_pr_mz),
        ("min_fr_mz", req.min_fr_mz),
        ("max_fr_mz", req.max_fr_mz),
    ] {
        if let Some(v) = v {
            body[name] = serde_json::json!(v);
        }
    }
    if let Ok(text) = serde_json::to_string_pretty(&body) {
        let _ = std::fs::write(dir.join(CACHE_MARKER), text);
    }
}

/// What a search starting from a FASTA would do: reuse a library, or build one.
///
/// Answered before anything is started, so the interface can say "this will take a
/// while" or "this is already built" rather than discovering it afterwards.
pub fn library_plan(req: BuildRequest) -> Result<serde_json::Value, String> {
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
    /// Every command this build has started, in order, the running one last, so the
    /// progress screen can show what is being run as it does for a search.
    pub commands: Vec<String>,
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
    /// Forget a finished build, after the library cache it wrote was removed.
    ///
    /// `BuildState` carries the paths of the two tables the build produced, and the
    /// search screen offers to reuse them. Removing the cache without clearing this
    /// would offer a library that is no longer on disk.
    pub fn forget(&self) {
        if let Ok(mut s) = self.state.lock() {
            *s = BuildState::default();
        }
    }

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

/// A command as a person would type it: arguments with spaces quoted.
fn command_line(exe: &Path, args: &[String]) -> String {
    let quote = |a: &str| {
        if a.is_empty() || a.contains(char::is_whitespace) {
            format!("\"{a}\"")
        } else {
            a.to_string()
        }
    };
    std::iter::once(quote(&exe.display().to_string()))
        .chain(args.iter().map(|a| quote(a)))
        .collect::<Vec<_>>()
        .join(" ")
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
        // The library helpers are Python: UTF-8 on the pipes whatever the code page,
        // so a path with a non-ASCII character reaches the log as written (issue #281).
        let mut c = crate::engine::command(exe);
        crate::childio::python_utf8(&mut c);
        c
    };
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());
    // Its own process group, so Stop reaches DIA-NN under a helper script (`run::kill_tree`).
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    // Refuse to start another step once cancelled, or a Stop during the prediction
    // would be followed by the import and decoy steps running anyway.
    if builder.is_cancelled() {
        return Err("cancelled".into());
    }
    if let Ok(mut s) = builder.state.lock() {
        s.commands.push(command_line(exe, args));
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("{} did not start: {e}", exe.display()))?;
    if let Ok(mut c) = builder.child.lock() {
        *c = Some(child.id());
    }

    // Both streams, on two threads, so a child that writes only to stderr still
    // shows progress and a full pipe cannot deadlock the child. The last lines of each
    // are kept apart from the log, for the error message when the step fails.
    let mut joins = Vec::new();
    let tails = [
        Arc::new(Mutex::new(Tail::default())),
        Arc::new(Mutex::new(Tail::default())),
    ];
    for (stream, tail) in [
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
    .zip(tails.iter().cloned())
    .filter_map(|(s, t)| s.map(|s| (s, t)))
    {
        let sink = Arc::clone(builder);
        joins.push(std::thread::spawn(move || {
            // Lossy and to the end: DIA-NN echoes paths in the OEM code page, and the
            // Python helpers write in the ANSI one; a reader that stopped on such a line
            // closed the pipe under a still-running child.
            crate::childio::for_each_line_lossy(stream, |line| {
                if let Ok(mut t) = tail.lock() {
                    t.push(&line);
                }
                sink.log(line);
            });
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
        let [out, err] = tails.map(|t| t.lock().map(|t| t.lines()).unwrap_or_default());
        return Err(step_failure(
            exe,
            args,
            is_diann,
            status.code(),
            // What a failing Python helper says is on stderr (a `SystemExit` message or a
            // traceback); DIA-NN writes everything to stdout.
            if err.is_empty() { &out } else { &err },
        ));
    }
    Ok(())
}

/// How many of a failed step's last lines its error message quotes.
const TAIL_LINES: usize = 4;

/// The last non-empty lines of one of a child's streams.
#[derive(Default)]
struct Tail(std::collections::VecDeque<String>);

impl Tail {
    fn push(&mut self, line: &str) {
        let line = line.trim_end();
        if line.trim().is_empty() {
            return;
        }
        if self.0.len() == TAIL_LINES {
            self.0.pop_front();
        }
        self.0.push_back(line.to_string());
    }

    fn lines(&self) -> Vec<String> {
        self.0.iter().cloned().collect()
    }
}

/// The error of a step that exited non-zero: what ran (the script, for a Python helper),
/// its exit code, and the last lines it wrote, so the reason is in the message and not
/// only in the log on the Progress screen.
fn step_failure(
    exe: &Path,
    args: &[String],
    is_diann: bool,
    code: Option<i32>,
    tail: &[String],
) -> String {
    let file = |p: &Path| {
        p.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    };
    let program = file(exe);
    let what = match args.first().map(Path::new) {
        Some(script)
            if !is_diann
                && script
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("py")) =>
        {
            format!("{} ({program})", file(script))
        }
        _ => program,
    };
    let code = code
        .map(|c| c.to_string())
        .unwrap_or_else(|| "a signal".into());
    let mut msg = format!("{what} exited with {code}");
    if tail.is_empty() {
        msg.push_str(", and wrote nothing.");
    } else {
        msg.push_str(". Its last output:");
        for line in tail {
            msg.push_str(
                "
  ",
            );
            msg.push_str(line);
        }
    }
    msg
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
fn predict_args(req: &BuildRequest, plan: &ModPlan, out_lib: &Path) -> Result<Vec<String>, String> {
    let cut = cut_rule(req)?;
    let rule = crate::enzymes::Rule::parse(&cut)?;
    // Under a rule that cuts every bond a peptide of n residues spans n - 1 cleavage
    // sites, so DIA-NN needs that many missed cleavages to enumerate it; the setting is
    // not a choice there.
    let missed = if rule.is_non_specific() {
        req.missed_cleavages.max(req.max_pep_len.saturating_sub(1))
    } else {
        req.missed_cleavages
    };
    let mut args: Vec<String> = vec![
        "--fasta".into(),
        req.fasta.clone(),
        "--fasta-search".into(),
        "--gen-spec-lib".into(),
        "--predictor".into(),
        "--cut".into(),
        cut.clone(),
        // Emit N-terminal initiator-Met-excised forms.
        //
        // Not optional, and its absence was a silent scientific defect rather than a
        // missing convenience. MuMDIA's native digest defaults to
        // `digest.n_term_met_excision = true` precisely "matching DIA-NN
        // `--met-excision`" (CLAUDE.md), and without it the search database
        // structurally misses those peptides. A raw imported DIA-NN library built
        // without it was missing 209 of DIA-NN's own 1% peptides on the AIF benchmark,
        // ALL of them this form. This recipe omitted the flag, so every library the interface built lacked them,
        // with nothing saying so, and the loss would have read as a MuMDIA sensitivity
        // result rather than a library-recipe defect.
        "--met-excision".into(),
        "--missed-cleavages".into(),
        missed.to_string(),
        "--min-pep-len".into(),
        req.min_pep_len.to_string(),
        "--max-pep-len".into(),
        req.max_pep_len.to_string(),
        "--min-pr-charge".into(),
        req.min_charge.to_string(),
        "--max-pr-charge".into(),
        req.max_charge.to_string(),
    ];
    // The m/z limits, only when set, so a request without them is the argument list it
    // always was.
    args.extend(req.mz_args()?);
    // Fixed carbamidomethyl on C is DIA-NN's `--unimod4`; every other fixed modification is
    // `--fixed-mod UniMod:<id>,<mass>,<residues>`, and each variable one `--var-mod` in the
    // same form, under one `--var-mods <max per peptide>`. The standard selection produces
    // exactly the arguments it always did.
    for (m, residues) in ModPlan::grouped(&plan.fixed) {
        if m.name == "Carbamidomethyl" && residues == "C" {
            args.push("--unimod4".into());
        } else {
            args.push("--fixed-mod".into());
            args.push(format!("UniMod:{},{:.6},{}", m.unimod, m.mass, residues));
        }
    }
    if !plan.variable.is_empty() {
        args.push("--var-mods".into());
        args.push(plan.max_variable.to_string());
        for (m, residues) in ModPlan::grouped(&plan.variable) {
            args.push("--var-mod".into());
            args.push(format!("UniMod:{},{:.6},{}", m.unimod, m.mass, residues));
        }
    }
    // Trypsin does not cleave after a GlyGly-modified lysine, so its peptides carry the
    // remnant on an internal K; DIA-NN documents this flag for that search. It only means
    // something for a rule that cuts after K, and a non-specific rule enumerates every
    // span anyway.
    if rule.cuts_after(b'K')
        && !rule.is_non_specific()
        && plan
            .fixed
            .iter()
            .chain(&plan.variable)
            .any(|m| m.name == "GlyGly" && m.residue == 'K')
    {
        args.push("--no-cut-after-mod".into());
        args.push("UniMod:121".into());
    }
    args.push("--out-lib".into());
    args.push(out_lib.display().to_string());
    args.push("--threads".into());
    args.push(req.threads.to_string());
    Ok(args)
}

/// Predict a library with the user's DIA-NN, then convert it to MuMDIA's schema.
///
/// Four steps, all of which must succeed: DIA-NN predicts, `import_diann_lib.py`
/// maps it into the target schema, `make_reverse_decoys.py` adds the reversed decoy
/// population, and `predict_decoys.py` replaces the decoys' copied intensities with
/// DIA-NN's prediction for their sequences -- the last of which also sorts by precursor
/// m/z and re-indexes `candidate_id`, both of which the fragment index rejects a library
/// for lacking. Everything after the prediction is `convert`, which `import` shares.
pub fn build(builder: Arc<Builder>, req: BuildRequest) -> Result<(), String> {
    // Before anything runs: an implausible m/z limit (a highest fragment m/z of 1) makes
    // DIA-NN generate nothing, and the build then failed steps later with a message about
    // fragment rows.
    req.mz_limits().check()?;
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
    let (python, scripts) = conversion_tools()?;

    let fasta = PathBuf::from(&req.fasta);
    if !fasta.is_file() {
        return Err(format!("{} is not a file", fasta.display()));
    }
    let out_dir = PathBuf::from(&req.out_dir);
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;

    begin(&builder, "predicting the library with DIA-NN")?;

    std::thread::spawn(move || {
        let stem = "lib";
        let out_lib = out_dir.join(stem);

        // The command line recorded in README.md, so this reproduces what a user
        // following the written instructions would have run.
        let plan = match ModPlan::of(&req) {
            Ok(p) => p,
            Err(e) => {
                builder.fail(e);
                return;
            }
        };
        let args = match predict_args(&req, &plan, &out_lib) {
            Ok(a) => a,
            Err(e) => {
                builder.fail(e);
                return;
            }
        };

        let version_for_cache = status.version.clone();
        builder.log(format!("== DIA-NN: {}", status.version.unwrap_or_default()));
        builder.log(format!("== {} {}", exe.display(), args.join(" ")));
        // A failed or stopped earlier build in this folder may have left its Parquet,
        // which `library_parquet` would take before this run's `.speclib`.
        let _ = std::fs::remove_file(out_dir.join(format!("{stem}.parquet")));
        if let Err(e) = run_step(&builder, &exe, &args, Some(&out_dir), true) {
            builder.fail(e);
            return;
        }
        // DIA-NN exits 0 having generated nothing when the search space is empty; the
        // re-export and the importer then failed on an empty library.
        if generated_precursors(&builder.snapshot().log) == Some(0) {
            builder.fail(empty_build_message(&req));
            return;
        }

        let conversion = Conversion {
            exe,
            python,
            scripts,
            out_dir,
            threads: req.threads,
            keep_unimod: build_keep_unimod(&plan),
            decoy_args: req.decoy_diann_args(),
            diann_mods: plan.declarations(),
            fasta: Some(PathBuf::from(&req.fasta)),
            remap: None,
            empty: empty_build_message(&req),
        };
        let result = convert(&builder, &conversion, Source::Predicted(stem));
        // The marker last, after both tables are on disk. Anything that dies before
        // this leaves a directory `cached_library` will not accept, which is the
        // point: a half-written library must not read as a finished one.
        finish(&builder, result, |dir| {
            mark_cache_complete(dir, &req, version_for_cache.as_deref().unwrap_or(""))
        });
    });

    Ok(())
}

/// The precursor count DIA-NN reported for a library it generated ("N precursors
/// generated"), the last such line of the log; `None` when it printed none.
fn generated_precursors(log: &[String]) -> Option<u64> {
    log.iter().rev().find_map(|line| {
        let at = line.find("precursors generated")?;
        let count = line[..at].split_whitespace().last()?;
        count.replace([',', '\''], "").parse().ok()
    })
}

/// The error for a build whose prediction came out empty, with the settings it used.
fn empty_build_message(req: &BuildRequest) -> String {
    format!(
        "DIA-NN generated 0 precursors from the FASTA, so there is no library to search. \
         Check the digest settings and the m/z limits on the Search screen ({}).",
        req.describe()
    )
}

/// The row count of a Parquet file, from its footer; `None` when it cannot be read.
///
/// The footer is a Thrift compact-protocol `FileMetaData`, whose field 3 is `num_rows`.
/// Fields 1 (`version`) and 2 (`schema`) come before it and are skipped.
fn parquet_num_rows(path: &Path) -> Option<i64> {
    footer_num_rows(&parquet_footer(path).ok()?)
}

fn footer_num_rows(footer: &[u8]) -> Option<i64> {
    let mut r = Compact {
        buf: footer,
        pos: 0,
    };
    let mut last = 0i16;
    loop {
        let b = r.byte()?;
        if b == 0 {
            return None;
        }
        let ty = b & 0x0f;
        let delta = (b >> 4) as i16;
        let id = if delta == 0 {
            r.zigzag()? as i16
        } else {
            last + delta
        };
        last = id;
        if id == 3 && ty == 6 {
            return r.zigzag();
        }
        r.skip(ty, 0)?;
    }
}

/// Just enough of the Thrift compact protocol to skip values.
struct Compact<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Compact<'_> {
    fn byte(&mut self) -> Option<u8> {
        let b = *self.buf.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }

    fn varint(&mut self) -> Option<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }

    fn zigzag(&mut self) -> Option<i64> {
        let v = self.varint()?;
        Some((v >> 1) as i64 ^ -((v & 1) as i64))
    }

    fn advance(&mut self, n: usize) -> Option<()> {
        let end = self.pos.checked_add(n).filter(|e| *e <= self.buf.len())?;
        self.pos = end;
        Some(())
    }

    /// Skip one value of compact type `ty`. `depth` bounds the recursion on a footer
    /// that is not what it claims to be.
    fn skip(&mut self, ty: u8, depth: u32) -> Option<()> {
        if depth > 32 {
            return None;
        }
        match ty {
            // A bool field carries its value in the type nibble.
            1 | 2 => Some(()),
            3 => self.advance(1),
            4..=6 => self.varint().map(|_| ()),
            7 => self.advance(8),
            8 => {
                let n = self.varint()? as usize;
                self.advance(n)
            }
            9 | 10 => {
                let h = self.byte()?;
                let n = if h >> 4 == 15 {
                    self.varint()? as usize
                } else {
                    (h >> 4) as usize
                };
                let et = h & 0x0f;
                for _ in 0..n {
                    // A bool element of a list is one byte.
                    if et == 1 || et == 2 {
                        self.advance(1)?;
                    } else {
                        self.skip(et, depth + 1)?;
                    }
                }
                Some(())
            }
            11 => {
                let n = self.varint()? as usize;
                if n == 0 {
                    return Some(());
                }
                let kv = self.byte()?;
                for _ in 0..n {
                    self.skip(kv >> 4, depth + 1)?;
                    self.skip(kv & 0x0f, depth + 1)?;
                }
                Some(())
            }
            12 => loop {
                let b = self.byte()?;
                if b == 0 {
                    return Some(());
                }
                // A field id that does not fit the header's delta follows it.
                if b >> 4 == 0 {
                    self.zigzag()?;
                }
                self.skip(b & 0x0f, depth + 1)?;
            },
            _ => None,
        }
    }
}

/// The accessions a built library's import keeps.
///
/// The importer keeps the standard accessions by default; any other selection names
/// what DIA-NN was asked to place, or the precursors carrying it would be dropped as
/// unmapped.
fn build_keep_unimod(plan: &ModPlan) -> Option<Vec<u32>> {
    (!plan.is_legacy()).then(|| plan.unimod_ids())
}

// ── the conversion after DIA-NN, shared by `build` and `import` ─────────────

/// The Python interpreter and the scripts directory the conversion runs.
fn conversion_tools() -> Result<(PathBuf, PathBuf), String> {
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
    if let Some(e) = crate::components::base_interpreter_gone(&crate::components::venv_dir(
        crate::components::Env::Primary,
    )) {
        return Err(e);
    }
    let scripts = crate::engine::scripts_dir()
        .ok_or_else(|| "the conversion scripts were not found".to_string())?;
    Ok((python, scripts))
}

/// Claim the builder for a new build or import, refusing while one runs.
fn begin(builder: &Arc<Builder>, step: &str) -> Result<(), String> {
    let mut s = builder
        .state
        .lock()
        .map_err(|_| "internal state is poisoned".to_string())?;
    if s.status == "running" {
        return Err("a library build is already running".into());
    }
    *s = BuildState {
        status: "running".into(),
        step: step.into(),
        ..Default::default()
    };
    installer_reset(builder);
    Ok(())
}

/// What turning a DIA-NN library into the two engine tables needs.
struct Conversion {
    /// The user's DIA-NN: it re-exports a `.speclib` and predicts the decoys.
    exe: PathBuf,
    python: PathBuf,
    scripts: PathBuf,
    /// Where the intermediate tables and the two final ones are written.
    out_dir: PathBuf,
    threads: u32,
    /// `--keep-unimod` for the importer and the decoy prediction; `None` keeps the
    /// importer's default set.
    keep_unimod: Option<Vec<u32>>,
    /// Extra `predict_decoys.py` arguments (`--diann-arg=...`): the fragment m/z limits of
    /// the build, empty for an import or a build without them.
    decoy_args: Vec<String>,
    /// The modifications every DIA-NN run of the conversion declares (`UniMod:<id>,<mass>`):
    /// the re-export of a `.speclib` (`--mod`) and the decoys' prediction and re-export
    /// (`predict_decoys.py --diann-mod`). A build's are its own modifications; an import's
    /// the whole catalogue, since the chosen library may carry any of them.
    diann_mods: Vec<String>,
    /// The FASTA a build predicted from, given to the importer (`--fasta`) so it can name what
    /// the DIA-NN library leaves without an accession, entry name or gene; `None` for an
    /// import, whose library names its proteins itself.
    fasta: Option<PathBuf>,
    /// An import's re-mapping to another FASTA (`--remap-fasta`), which then names every
    /// protein and its annotations; `None` keeps the library's proteins.
    remap: Option<Remap>,
    /// The error when the DIA-NN library has no rows, refused before the importer runs.
    empty: String,
}

/// A DIA-NN library a user chose, by the format it is in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiannLibrary {
    /// A fragment-level Parquet library (DIA-NN 1.9 and 2.x), read as it is.
    Parquet(PathBuf),
    /// DIA-NN's binary `.speclib`, re-exported as Parquet with the user's DIA-NN first.
    Speclib(PathBuf),
}

/// The DIA-NN library the conversion starts from.
enum Source {
    /// What `build` predicted under this stem in the output folder: DIA-NN's Parquet
    /// when it wrote one, otherwise its `.speclib`, re-exported.
    Predicted(&'static str),
    /// A library the user chose (`import`).
    Chosen(DiannLibrary),
}

/// The stem an imported `.speclib` is re-exported under, in the output folder.
const IMPORT_REEXPORT_STEM: &str = "diann_library";

/// DIA-NN's command line for writing a `.speclib` as Parquet, declaring `mods`.
fn reexport_args(speclib: &Path, target: &Path, threads: u32, mods: &[String]) -> Vec<String> {
    let mut args = vec![
        "--lib".to_string(),
        speclib.to_string_lossy().into_owned(),
        "--gen-spec-lib".to_string(),
        "--out-lib".to_string(),
        target.to_string_lossy().into_owned(),
        "--threads".to_string(),
        threads.to_string(),
    ];
    args.extend(mod_declaration_args(mods));
    args
}

/// Re-export a `.speclib` as `<stem>.parquet` in the output folder, and find what
/// DIA-NN wrote.
fn reexport(
    builder: &Arc<Builder>,
    c: &Conversion,
    speclib: &Path,
    stem: &str,
) -> Result<PathBuf, String> {
    if let Ok(mut s) = builder.state.lock() {
        s.step = "re-exporting the library as Parquet".into();
    }
    builder.log(format!(
        "== DIA-NN wrote {}, which MuMDIA cannot read; re-exporting it as \
         Parquet",
        speclib.display()
    ));
    let target = c.out_dir.join(format!("{stem}.parquet"));
    let args = reexport_args(speclib, &target, c.threads, &c.diann_mods);
    builder.log(format!("== {} {}", c.exe.display(), args.join(" ")));
    if let Err(e) = run_step(builder, &c.exe, &args, Some(&c.out_dir), true) {
        return Err(format!(
            "{e}\n\nThe prediction succeeded and {} exists, but converting \
             it to Parquet failed. Convert it by hand with:\n  {} --lib {} \
             --gen-spec-lib --out-lib {}{}\nthen point the search at the \
             resulting tables.",
            speclib.display(),
            c.exe.display(),
            speclib.display(),
            target.display(),
            mod_declaration_args(&c.diann_mods)
                .chunks(2)
                .map(|p| format!(" {} {}", p[0], p[1]))
                .collect::<String>()
        ));
    }
    match find_lib_parquet(&c.out_dir, stem) {
        Some(p) => {
            builder.log(format!("== re-exported: {}", p.display()));
            Ok(p)
        }
        None => Err(format!(
            "the re-export ran but produced no Parquet in {}. Your DIA-NN \
             version may not support Parquet library output; 1.8.x does \
             not. Use DIA-NN 2.x, or export the library as .tsv and \
             convert it yourself.",
            c.out_dir.display()
        )),
    }
}

/// The fragment-level Parquet library the importer reads, re-exporting when needed.
fn library_parquet(
    builder: &Arc<Builder>,
    c: &Conversion,
    source: Source,
) -> Result<PathBuf, String> {
    match source {
        // DIA-NN writes a predicted library as its own binary `.speclib`, so a
        // second DIA-NN run is needed to re-export it as the Parquet the importer
        // reads. Parquet is looked for first anyway: a version that writes one
        // directly, or a `--out-lib` that already named one, needs no re-export and
        // paying for one would be minutes wasted on a large library.
        Source::Predicted(stem) => match find_lib_parquet(&c.out_dir, stem) {
            Some(p) => {
                builder.log(format!("== predicted library (Parquet): {}", p.display()));
                Ok(p)
            }
            None => {
                let Some(speclib) = find_predicted_speclib(&c.out_dir, stem) else {
                    return Err(format!(
                        "DIA-NN finished but wrote neither a Parquet nor a .speclib \
                         library in {}. Check the log: a licence or prediction failure \
                         can still exit successfully.",
                        c.out_dir.display()
                    ));
                };
                reexport(builder, c, &speclib, stem)
            }
        },
        Source::Chosen(DiannLibrary::Parquet(p)) => {
            builder.log(format!("== DIA-NN library (Parquet): {}", p.display()));
            Ok(p)
        }
        // Always re-exported, never a Parquet already in the folder: one left there by
        // an earlier import may be of another `.speclib`.
        Source::Chosen(DiannLibrary::Speclib(s)) => reexport(builder, c, &s, IMPORT_REEXPORT_STEM),
    }
}

/// `--keep-unimod <id,id,...>`, or nothing for the importer's default set.
fn keep_unimod_args(ids: Option<&[u32]>) -> Vec<String> {
    match ids {
        Some(ids) => vec![
            "--keep-unimod".into(),
            ids.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ],
        None => vec![],
    }
}

/// The two tables a conversion writes into `out_dir`. The protein annotations (accessions,
/// entry names, genes) are columns of `lib_precursors.parquet` itself, which the importer
/// writes and the decoy steps keep (#299), so the library carries them wherever it is copied.
/// A library converted by 0.10.0 to 0.12.0 has them in a `protein_annotations.parquet` beside
/// it instead, which the engine still reads; one converted before 0.10.0 has none, and a
/// search on it warns and leaves the reports' `protein_ids`, `protein_names` and `genes`
/// empty.
fn final_tables(out_dir: &Path) -> (PathBuf, PathBuf) {
    (
        out_dir.join("lib_precursors.parquet"),
        out_dir.join("lib_fragments.parquet"),
    )
}

/// The three Python steps once the DIA-NN library is Parquet, as (caption, arguments).
///
/// A function of its own so that the builder's steps can be asserted, and so that the
/// import runs exactly these and no copy of them.
fn conversion_steps(c: &Conversion, predicted: &Path) -> Vec<(&'static str, Vec<String>)> {
    let out_dir = &c.out_dir;
    let scripts = &c.scripts;
    let tgt_prec = out_dir.join("lib_precursors_targets.parquet");
    let tgt_frag = out_dir.join("lib_fragments_targets.parquet");
    let rev_prec = out_dir.join("lib_precursors_copied_decoys.parquet");
    let rev_frag = out_dir.join("lib_fragments_copied_decoys.parquet");
    let (prec, frag) = final_tables(out_dir);
    let keep = keep_unimod_args(c.keep_unimod.as_deref());

    vec![
        ("converting to the MuMDIA schema", {
            let mut a = vec![
                scripts.join("import_diann_lib.py").display().to_string(),
                predicted.display().to_string(),
                tgt_prec.display().to_string(),
                tgt_frag.display().to_string(),
            ];
            a.extend(keep.iter().cloned());
            if let Some(r) = &c.remap {
                a.extend([
                    "--remap-fasta".to_string(),
                    r.fasta.display().to_string(),
                    "--remap-cleavage-rule".to_string(),
                    r.rule.clone(),
                ]);
            } else if let Some(fasta) = &c.fasta {
                a.extend(["--fasta".to_string(), fasta.display().to_string()]);
            }
            a
        }),
        (
            "adding the decoy population",
            vec![
                scripts.join("make_reverse_decoys.py").display().to_string(),
                tgt_prec.display().to_string(),
                tgt_frag.display().to_string(),
                rev_prec.display().to_string(),
                rev_frag.display().to_string(),
            ],
        ),
        // The reversed decoys above carry their targets' intensities, copied ion for
        // ion; DIA-NN now predicts their spectra instead. Copied decoys made the
        // reported FDR optimistic: paired entrapment FDP 1.68% at a reported 1% on the
        // ProteoBench Astral entrapment module, against 0.85% with predicted decoys.
        ("predicting the decoy spectra with DIA-NN", {
            let mut a = vec![
                scripts.join("predict_decoys.py").display().to_string(),
                "build".into(),
                rev_prec.display().to_string(),
                rev_frag.display().to_string(),
                prec.display().to_string(),
                frag.display().to_string(),
                "--diann".into(),
                c.exe.display().to_string(),
                "--threads".into(),
                c.threads.to_string(),
                "--work".into(),
                out_dir.join("decoy_prediction").display().to_string(),
            ];
            a.extend(keep);
            a.extend(c.decoy_args.iter().cloned());
            a.extend(c.diann_mods.iter().map(|d| format!("--diann-mod={d}")));
            a
        }),
    ]
}

/// Everything after DIA-NN's prediction: the re-export when the library is a
/// `.speclib`, the import, the decoys and their prediction. Returns the two tables.
fn convert(
    builder: &Arc<Builder>,
    c: &Conversion,
    source: Source,
) -> Result<(PathBuf, PathBuf), String> {
    // The user's own Parquet library is never deleted; a prediction or re-export is ours.
    let own = !matches!(source, Source::Chosen(DiannLibrary::Parquet(_)));
    let predicted = library_parquet(builder, c, source)?;
    // An empty library made the importer fail with "no target b/y fragment rows survived
    // the filters", which says nothing about why. Only a footer that reads is trusted.
    if parquet_num_rows(&predicted) == Some(0) {
        return Err(c.empty.clone());
    }
    for (title, args) in conversion_steps(c, &predicted) {
        if let Ok(mut s) = builder.state.lock() {
            s.step = title.into();
        }
        builder.log(format!("== {title}"));
        run_step(builder, &c.python, &args, Some(&c.scripts), false)?;
    }
    // Each step's input is scratch once the two tables exist, about three times their
    // size. Delete errors are ignored: a finished library must not fail on cleanup.
    for name in [
        "lib_precursors_targets.parquet",
        "lib_fragments_targets.parquet",
        "lib_precursors_copied_decoys.parquet",
        "lib_fragments_copied_decoys.parquet",
    ] {
        let _ = std::fs::remove_file(c.out_dir.join(name));
    }
    let _ = std::fs::remove_dir_all(c.out_dir.join("decoy_prediction"));
    if own {
        let _ = std::fs::remove_file(&predicted);
    }
    Ok(final_tables(&c.out_dir))
}

/// Record the outcome of `convert`: on success `mark` writes the folder's completion
/// marker first, then the state reports the two tables.
fn finish(
    builder: &Arc<Builder>,
    result: Result<(PathBuf, PathBuf), String>,
    mark: impl FnOnce(&Path),
) {
    match result {
        Err(e) => builder.fail(e),
        Ok((prec, frag)) => {
            if let Some(dir) = prec.parent() {
                mark(dir);
            }
            if let Ok(mut s) = builder.state.lock() {
                s.status = "done".into();
                s.step = "finished".into();
                s.precursors = Some(prec.display().to_string());
                s.fragments = Some(frag.display().to_string());
            }
        }
    }
}

// ── importing a DIA-NN library the user already has ────────────────────────
//
// A library-mode search needs the two engine tables, and a DIA-NN library is one
// fragment-level Parquet (or a `.speclib`). The conversion is the builder's from the
// step after its prediction. No FASTA is needed: the protein identifiers come from the
// library's own `Protein.Names` / `Protein.Ids` columns, and the decoys are built from
// the target sequences. There is deliberately no fallback to copied-intensity decoys
// when DIA-NN is missing: those made the reported FDR optimistic (paired entrapment
// FDP 1.68% at a reported 1%, against 0.85% with predicted decoys).

/// A DIA-NN library to convert, and where to write the two tables.
#[derive(Deserialize, Clone, Debug)]
pub struct ImportRequest {
    /// The DIA-NN library: a `.parquet` or a `.speclib`.
    pub library: String,
    pub out_dir: String,
    #[serde(default = "d_threads")]
    pub threads: u32,
    /// A FASTA to map the library's peptides to instead of keeping the proteins the library
    /// names (#319, `import_diann_lib.py --remap-fasta`); empty or absent keeps them.
    #[serde(default)]
    pub remap_fasta: Option<String>,
    /// The cleavage preset the re-mapping holds an occurrence to (`enzymes.json`), Trypsin/P
    /// when absent.
    #[serde(default)]
    pub remap_enzyme: Option<String>,
    /// The rule of `remap_enzyme = "custom"`.
    #[serde(default)]
    pub remap_cleavage_rule: Option<String>,
}

/// The FASTA an import re-maps its peptides to, with the cleavage rule (rule text) an
/// occurrence must satisfy.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Remap {
    fasta: PathBuf,
    rule: String,
}

impl ImportRequest {
    /// The re-mapping this request asks for, checked: `None` without a FASTA.
    fn remap(&self) -> Result<Option<Remap>, String> {
        let Some(fasta) = self
            .remap_fasta
            .as_deref()
            .map(str::trim)
            .filter(|f| !f.is_empty())
        else {
            return Ok(None);
        };
        let path = PathBuf::from(fasta);
        if !path.is_file() {
            return Err(format!("the FASTA to re-map to, {fasta}, is not a file"));
        }
        let enzyme = self.remap_enzyme.as_deref().unwrap_or("trypsin_p");
        let rule = crate::enzymes::resolve(enzyme, self.remap_cleavage_rule.as_deref())?;
        Ok(Some(Remap { fasta: path, rule }))
    }
}

/// The columns `import_diann_lib.py` requires of a DIA-NN library (its `_META_COLS` and
/// `_FRAG_COLS`). Change both together.
pub const DIANN_LIBRARY_COLUMNS: &[&str] = &[
    "Modified.Sequence",
    "Stripped.Sequence",
    "Precursor.Charge",
    "Precursor.Mz",
    "RT",
    "Product.Mz",
    "Relative.Intensity",
    "Fragment.Type",
    "Fragment.Series.Number",
    "Fragment.Charge",
];

/// The footer of a Parquet file, which holds its schema as Thrift bytes.
fn parquet_footer(path: &Path) -> Result<Vec<u8>, String> {
    use std::io::{Read, Seek, SeekFrom};
    let not_parquet = || {
        format!(
            "{} is not a Parquet file (it does not start and end with the Parquet \
             magic bytes); a DIA-NN library is a .parquet or a .speclib",
            path.display()
        )
    };
    let io = |e: std::io::Error| format!("cannot read {}: {e}", path.display());
    let mut f = std::fs::File::open(path).map_err(io)?;
    let len = f.metadata().map_err(io)?.len();
    if len < 12 {
        return Err(not_parquet());
    }
    let mut head = [0u8; 4];
    f.read_exact(&mut head).map_err(io)?;
    let mut tail = [0u8; 8];
    f.seek(SeekFrom::End(-8)).map_err(io)?;
    f.read_exact(&mut tail).map_err(io)?;
    if &head != b"PAR1" || &tail[4..] != b"PAR1" {
        return Err(not_parquet());
    }
    let footer_len = u64::from(u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]));
    if footer_len + 12 > len {
        return Err(format!(
            "{} is a truncated Parquet file: its footer is longer than the file",
            path.display()
        ));
    }
    f.seek(SeekFrom::Start(len - 8 - footer_len)).map_err(io)?;
    let mut footer = vec![0u8; footer_len as usize];
    f.read_exact(&mut footer).map_err(io)?;
    Ok(footer)
}

/// Whether a Parquet footer names a column, read without a Parquet library.
///
/// The schema's names are Thrift binary fields, a varint length and then the bytes, so
/// a name is matched with its length in front of it: `RT` is not found inside `iRT`.
fn footer_names(footer: &[u8], name: &str) -> bool {
    let n = name.as_bytes();
    let mut needle = Vec::with_capacity(n.len() + 2);
    let mut len = n.len();
    loop {
        let b = (len & 0x7f) as u8;
        len >>= 7;
        if len == 0 {
            needle.push(b);
            break;
        }
        needle.push(b | 0x80);
    }
    needle.extend_from_slice(n);
    footer.windows(needle.len()).any(|w| w == needle.as_slice())
}

/// Check that `path` looks like a DIA-NN spectral library, before anything expensive.
///
/// A `.parquet` must carry every column the importer reads. Only its footer is read for
/// that, so the check is immediate on a library of any size. A `.speclib` is DIA-NN's
/// own binary format, which only DIA-NN reads; the re-export is its check. A text
/// library is refused by name.
pub fn inspect_library(path: &Path) -> Result<DiannLibrary, String> {
    if !path.is_file() {
        return Err(format!("{} is not a file", path.display()));
    }
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "speclib" => {
            let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            if len == 0 {
                return Err(format!("{} is empty", path.display()));
            }
            Ok(DiannLibrary::Speclib(path.to_path_buf()))
        }
        "parquet" => {
            let footer = parquet_footer(path)?;
            if footer_names(&footer, "peptidoform") && footer_names(&footer, "candidate_id") {
                return Err(format!(
                    "{} is a MuMDIA library table already, not a DIA-NN library: select \
                     it and its partner directly as the precursor and fragment tables",
                    path.display()
                ));
            }
            let missing: Vec<&str> = DIANN_LIBRARY_COLUMNS
                .iter()
                .copied()
                .filter(|c| !footer_names(&footer, c))
                .collect();
            if missing.is_empty() {
                return Ok(DiannLibrary::Parquet(path.to_path_buf()));
            }
            let fragment_level = ["Product.Mz", "Relative.Intensity", "Fragment.Type"];
            let what = if fragment_level.iter().all(|c| missing.contains(c))
                && footer_names(&footer, "Precursor.Mz")
            {
                "it has precursor columns but no fragment columns, which is what a \
                 DIA-NN search report looks like; a spectral library has one row per \
                 fragment"
            } else {
                "it is not a DIA-NN fragment-level spectral library"
            };
            Err(format!(
                "{}: {what} (missing {}). Choose the library DIA-NN wrote with \
                 --gen-spec-lib, as a .parquet or a .speclib.",
                path.display(),
                missing.join(", ")
            ))
        }
        "tsv" | "csv" | "txt" | "xls" => Err(format!(
            "{} is a text library, which this import does not read. Choose the DIA-NN \
             library as a .parquet or a .speclib; DIA-NN 2.x writes a text library as \
             Parquet with --lib <library.tsv> --gen-spec-lib --out-lib <name>.parquet.",
            path.display()
        )),
        _ => Err(format!(
            "{} is neither a .parquet nor a .speclib DIA-NN library",
            path.display()
        )),
    }
}

/// Why an import cannot run with this DIA-NN, or else the DIA-NN it runs.
///
/// Refused rather than degraded: without DIA-NN the decoys could only copy their
/// targets' fragment intensities, and those make the reported FDR optimistic.
pub fn import_refusal(status: &Status, library: &DiannLibrary) -> Result<PathBuf, String> {
    const WHY: &str = "Importing a DIA-NN library needs a working DIA-NN, because MuMDIA \
         has DIA-NN predict the spectra of the decoys it adds: decoys that copy their \
         targets' fragment intensities make the reported FDR optimistic, so the import \
         does not fall back to them.";
    let what = match library {
        DiannLibrary::Speclib(_) => {
            "re-export this .speclib as the Parquet MuMDIA reads nor write its \
             prediction of the decoys' spectra"
        }
        DiannLibrary::Parquet(_) => {
            "write its prediction of the decoys' spectra in a form MuMDIA reads, nor \
             re-export one"
        }
    };
    diann_refusal(status, WHY, what)
}

/// The DIA-NN a step that predicts decoy spectra runs, or why it cannot: `why` opens every
/// refusal, `what` says what a DIA-NN without Parquet output cannot do.
fn diann_refusal(status: &Status, why: &str, what: &str) -> Result<PathBuf, String> {
    let exe = match (&status.path, status.runs) {
        (Some(p), true) => PathBuf::from(p),
        _ => {
            let detail = status
                .error
                .as_deref()
                .filter(|e| !e.is_empty() && *e != "no working DIA-NN was found")
                .map(|e| format!(" ({e})"))
                .unwrap_or_default();
            return Err(format!(
                "{why} No working DIA-NN was found{detail}. Install DIA-NN 2.x, locate it \
                 on the Setup screen and acknowledge its licence there."
            ));
        }
    };
    if !status.licence_acknowledged {
        return Err(format!(
            "{why} The DIA-NN licence notice has not been acknowledged: acknowledge it on \
             the Setup screen."
        ));
    }
    if !status.writes_parquet {
        let version = status.version.as_deref().unwrap_or("this DIA-NN");
        return Err(format!(
            "{version} cannot write a Parquet spectral library (1.8.x has no Parquet \
             library output), so it can neither {what}. Install DIA-NN 2.x and locate it \
             on the Setup screen."
        ));
    }
    Ok(exe)
}

/// Refuse an output folder that already holds a converted library, so that two
/// libraries cannot be mixed in one pair of tables.
fn check_import_out_dir(out_dir: &Path) -> Result<(), String> {
    let (p, f) = final_tables(out_dir);
    let taken: Vec<String> = [p, f]
        .iter()
        .filter(|t| t.exists())
        .map(|t| {
            t.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    if taken.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} already holds {}; choose an empty folder for the converted library",
            out_dir.display(),
            taken.join(" and ")
        ))
    }
}

/// Every UniMod accession of the modification catalogue, for `--keep-unimod`.
///
/// The builder keeps what it asked DIA-NN to place; an imported library's author chose
/// its modifications, so the import keeps every one the engine can name and drops (in the
/// importer's count) only precursors that carry a modification outside the catalogue.
fn catalogue_unimod_ids() -> Vec<u32> {
    let mut ids: Vec<u32> = crate::mods::catalogue().iter().map(|m| m.unimod).collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// The completion marker of an imported library, with where it came from.
fn mark_import_complete(
    dir: &Path,
    library: &str,
    version: &str,
    kept: &[u32],
    remap: Option<&Remap>,
) {
    let mut body = serde_json::json!({
        "source": "diann-import",
        "diann_library": library,
        "diann_version": version,
        "recipe_version": RECIPE_VERSION,
        "keep_unimod": kept,
    });
    if let Some(r) = remap {
        body["remap_fasta"] = serde_json::json!(r.fasta.display().to_string());
        body["remap_cleavage_rule"] = serde_json::json!(r.rule);
    }
    if let Ok(text) = serde_json::to_string_pretty(&body) {
        let _ = std::fs::write(dir.join(CACHE_MARKER), text);
    }
}

/// Convert a DIA-NN library the user already has into the two tables a search reads.
///
/// The builder's steps after its prediction (`convert`): a `.speclib` is re-exported as
/// Parquet with the user's DIA-NN, `import_diann_lib.py` maps it, `make_reverse_decoys.py`
/// adds the decoys and `predict_decoys.py` has DIA-NN predict their spectra. Progress is
/// the builder's state, so the interface follows it exactly as it follows a build.
pub fn import(builder: Arc<Builder>, req: ImportRequest) -> Result<(), String> {
    // The file first: the check is immediate and says what is wrong with the choice.
    let library = inspect_library(Path::new(&req.library))?;
    let remap = req.remap()?;
    let status = detect();
    let exe = import_refusal(&status, &library)?;
    let (python, scripts) = conversion_tools()?;
    if req.out_dir.trim().is_empty() {
        return Err("choose a folder for the converted library".into());
    }
    let out_dir = PathBuf::from(&req.out_dir);
    check_import_out_dir(&out_dir)?;
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;

    begin(&builder, "converting the DIA-NN library")?;

    std::thread::spawn(move || {
        let version = status.version.clone().unwrap_or_default();
        builder.log(format!("== DIA-NN: {version}"));
        builder.log(format!("== importing {}", req.library));
        if let Some(r) = &remap {
            builder.log(format!(
                "== re-mapping the peptides to {} (cleavage rule {})",
                r.fasta.display(),
                r.rule
            ));
        }
        let kept = catalogue_unimod_ids();
        let conversion = Conversion {
            exe,
            python,
            scripts,
            out_dir,
            threads: req.threads,
            keep_unimod: Some(kept.clone()),
            decoy_args: Vec::new(),
            diann_mods: declarations(crate::mods::catalogue().iter()),
            fasta: None,
            remap: remap.clone(),
            empty: format!("the DIA-NN library {} holds no precursors", req.library),
        };
        let result = convert(&builder, &conversion, Source::Chosen(library));
        finish(&builder, result, |dir| {
            mark_import_complete(dir, &req.library, &version, &kept, remap.as_ref())
        });
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
        if detect().licence_acknowledged {
            // The developer machine has already acknowledged, or has a DIA-NN from
            // which acknowledgement is assumed; this assertion is about the refusal
            // path, so skip rather than clobber their state.
            return;
        }
        let err = build(
            Arc::new(Builder::default()),
            BuildRequest {
                enzyme: d_enzyme(),
                cleavage_rule: None,
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
                fixed_mods: vec![],
                variable_mods: vec![],
                max_variable_mods: 1,
                min_pr_mz: None,
                max_pr_mz: None,
                min_fr_mz: None,
                max_fr_mz: None,
            },
        )
        .unwrap_err();
        assert!(
            err.contains("licence"),
            "refused for the licence reason: {err}"
        );
    }

    #[test]
    fn an_installed_diann_counts_as_acknowledged_until_the_user_says_otherwise() {
        // Nothing answered and nothing that runs: not acknowledged.
        assert_eq!(effective_licence(None, false), (false, false));
        // Nothing answered and a DIA-NN that runs: assumed, and reported as assumed.
        assert_eq!(effective_licence(None, true), (true, true));
        // An explicit answer always wins, in both directions.
        assert_eq!(effective_licence(Some(false), true), (false, false));
        assert_eq!(effective_licence(Some(true), false), (true, false));
        // An older state file stored a plain boolean; it still reads as an answer.
        let old: Saved =
            serde_json::from_str(r#"{"path":null,"licence_acknowledged":false}"#).unwrap();
        assert_eq!(old.licence_acknowledged, Some(false));
        let fresh: Saved = serde_json::from_str(r#"{"path":null}"#).unwrap();
        assert_eq!(fresh.licence_acknowledged, None);
    }

    #[test]
    fn a_build_step_is_shown_as_a_command_a_person_could_type() {
        let line = command_line(
            Path::new("C:/DIA-NN/2.7.0/diann.exe"),
            &[
                "--lib".into(),
                "my lib.tsv".into(),
                "--cut".into(),
                "".into(),
            ],
        );
        assert_eq!(
            line,
            r#"C:/DIA-NN/2.7.0/diann.exe --lib "my lib.tsv" --cut """#
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
            licence_acknowledged: Some(true),
        };
        let c = candidates(&saved);
        assert_eq!(c.first().map(|(s, _)| s.as_str()), Some("configured"));
    }

    #[test]
    fn the_predicted_parquet_is_found_by_name_or_by_search() {
        // Unique per process, like every other temp fixture in the workspace: a fixed name
        // plus the delete-on-entry below races a second `cargo test` on one machine
        // (docs/14). Hit twice during the 0.2.0 and 0.3.0 releases.
        let dir =
            std::env::temp_dir().join(format!("mumdia-diann-test-lib-{}", std::process::id()));
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
    fn the_state_file_is_the_users_own() {
        // In the user scope beside the components, as it always was, so an existing
        // licence answer and located path are still found.
        assert_eq!(
            state_file(),
            crate::components::tools_dir().join(STATE_FILE_NAME)
        );
        assert!(state_file().starts_with(crate::storage::user_tools_dir()));
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
        let dir = std::env::temp_dir().join(format!("mumdia-diann-speclib-{}", std::process::id()));
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
            enzyme: d_enzyme(),
            cleavage_rule: None,
            missed_cleavages: 1,
            min_pep_len: 7,
            max_pep_len: 30,
            min_charge: 2,
            max_charge: 4,
            threads: 8,
            carbamidomethyl: true,
            oxidation: true,
            fixed_mods: vec![],
            variable_mods: vec![],
            max_variable_mods: 1,
            min_pr_mz: None,
            max_pr_mz: None,
            min_fr_mz: None,
            max_fr_mz: None,
        }
    }

    /// The m/z limits reach DIA-NN only when set: without them the argument list is the
    /// one it always was; set, each one is passed, and the fragment limits also reach the
    /// decoys' prediction. A reversed or non-positive range is refused.
    #[test]
    fn the_mz_limits_reach_the_diann_arguments_only_when_set() {
        let base = a_request("x.fasta");
        let plan = ModPlan::of(&base).unwrap();
        let args = predict_args(&base, &plan, Path::new("o")).unwrap();
        assert!(!args.iter().any(|a| a.ends_with("-mz")), "{args:?}");
        assert!(base.decoy_diann_args().is_empty());

        let mut r = a_request("x.fasta");
        r.min_pr_mz = Some(380.0);
        r.max_pr_mz = Some(980.5);
        r.min_fr_mz = Some(150.0);
        let args = predict_args(&r, &plan, Path::new("o")).unwrap();
        let value = |flag: &str| -> Option<String> {
            args.iter()
                .position(|a| a == flag)
                .map(|i| args[i + 1].clone())
        };
        assert_eq!(value("--min-pr-mz").as_deref(), Some("380"));
        assert_eq!(value("--max-pr-mz").as_deref(), Some("980.5"));
        assert_eq!(value("--min-fr-mz").as_deref(), Some("150"));
        assert_eq!(value("--max-fr-mz"), None);
        assert_eq!(
            r.decoy_diann_args(),
            ["--diann-arg=--min-fr-mz", "--diann-arg=150"]
        );

        let mut bad = a_request("x.fasta");
        bad.min_pr_mz = Some(900.0);
        bad.max_pr_mz = Some(400.0);
        assert!(predict_args(&bad, &plan, Path::new("o")).is_err());
        bad.max_pr_mz = None;
        bad.min_fr_mz = Some(-1.0);
        assert!(predict_args(&bad, &plan, Path::new("o")).is_err());
        // The reported case: positive, but not an m/z.
        bad.min_fr_mz = None;
        bad.max_fr_mz = Some(1.0);
        let e = predict_args(&bad, &plan, Path::new("o")).unwrap_err();
        assert!(e.contains("Highest fragment m/z is 1,"), "{e}");

        // An old request without the fields reads as no limit.
        let old: BuildRequest =
            serde_json::from_str(r#"{"fasta": "x.fasta", "out_dir": "o"}"#).unwrap();
        assert_eq!((old.min_pr_mz, old.max_fr_mz), (None, None));
    }

    /// A build with an implausible m/z limit is refused when it is asked for, before
    /// DIA-NN is looked for or anything starts, naming the field and its value.
    #[test]
    fn a_build_with_an_implausible_mz_limit_is_refused_before_it_starts() {
        let mut r = a_request("x.fasta");
        r.max_fr_mz = Some(1.0);
        let builder = Arc::new(Builder::default());
        let e = build(Arc::clone(&builder), r).unwrap_err();
        assert!(e.contains("Highest fragment m/z is 1,"), "{e}");
        assert_eq!(builder.snapshot().status, "", "nothing started");
        assert!(builder.snapshot().commands.is_empty());
    }

    /// DIA-NN's "N precursors generated" is read from the log, the last one counting; a
    /// log without it says nothing. The message for an empty build names the settings.
    #[test]
    fn an_empty_prediction_is_recognised_from_the_log() {
        let log = |lines: &[&str]| lines.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            generated_precursors(&log(&[
                "[0:00] Processing FASTA",
                "[0:01] 0 precursors generated"
            ])),
            Some(0)
        );
        assert_eq!(
            generated_precursors(&log(&[
                "12 precursors generated",
                "1,234,567 precursors generated"
            ])),
            Some(1_234_567)
        );
        assert_eq!(
            generated_precursors(&log(&["== DIA-NN: 2.2.0", "Finished"])),
            None
        );

        let mut r = a_request("x.fasta");
        r.max_fr_mz = Some(1.0);
        let m = empty_build_message(&r);
        assert!(
            m.starts_with("DIA-NN generated 0 precursors from the FASTA"),
            "{m}"
        );
        assert!(
            m.contains(
                "peptide length 7 to 30, charge 2 to 4, 1 missed cleavage(s), \
                        precursor m/z any to any, fragment m/z any to 1"
            ),
            "{m}"
        );
    }

    /// The row count is read from a Parquet footer (Thrift compact `FileMetaData`, field
    /// 3), skipping the version and the schema list before it; a footer that does not
    /// parse gives no count rather than a wrong one.
    #[test]
    fn the_row_count_is_read_from_a_parquet_footer() {
        let footer = |rows: &[u8]| {
            let mut f = vec![0x15, 0x02]; // 1: version = 1
            f.extend([0x19, 0x2C]); // 2: list of two structs
            f.extend([0x48, 6]); // 4: name, binary of 6 bytes
            f.extend(b"schema");
            f.extend([0x15, 0x02, 0x21, 0x00]); // 5: i32, 7: bool true, end
            f.extend([0x18, 4]); // 1: binary of 4 bytes (a field id by delta)
            f.extend(b"Prot");
            f.extend([0x06, 0x0E, 0x02, 0x00]); // id 7 (zigzag 14) i64, end
            f.push(0x16); // 3: num_rows, i64
            f.extend(rows);
            f.extend([0x19, 0x0C]); // 4: an empty list of row groups
            f.push(0x00);
            f
        };
        assert_eq!(footer_num_rows(&footer(&[0x00])), Some(0));
        assert_eq!(footer_num_rows(&footer(&[0xD8, 0x04])), Some(300));
        assert_eq!(footer_num_rows(&footer(&[0x00])[..12]), None, "truncated");
        assert_eq!(footer_num_rows(&[0x15]), None);
        assert_eq!(footer_num_rows(&[0x00]), None);
    }

    /// A step that exits non-zero says what it was and what it last wrote: the script of a
    /// Python helper by name, its exit code, and the tail of its output, not only
    /// "python.exe exited with 1".
    #[test]
    fn a_failed_step_quotes_its_last_output() {
        let mut t = Tail::default();
        for line in ["one", "", "two", "three", "   ", "four", "five"] {
            t.push(line);
        }
        assert_eq!(t.lines(), ["two", "three", "four", "five"]);

        let args = [
            "C:/app/scripts/import_diann_lib.py".to_string(),
            "lib.parquet".to_string(),
        ];
        let tail = [
            "C:/cache/lib.parquet: no target b/y fragment rows survived the filters".to_string(),
        ];
        let m = step_failure(Path::new("C:/env/python.exe"), &args, false, Some(1), &tail);
        assert_eq!(
            m,
            "import_diann_lib.py (python.exe) exited with 1. Its last output:\n  \
             C:/cache/lib.parquet: no target b/y fragment rows survived the filters"
        );
        let m = step_failure(
            Path::new("/opt/diann/diann-linux"),
            &["--lib".into()],
            true,
            None,
            &[],
        );
        assert_eq!(m, "diann-linux exited with a signal, and wrote nothing.");
    }

    /// The cleavage choice reaches DIA-NN's `--cut`; a non-specific rule asks for the
    /// missed cleavages a peptide of the maximum length spans and drops the GlyGly
    /// no-cut rule; a custom rule passes through; Trypsin/P is the default and the
    /// argument list it always was; a malformed custom rule is refused.
    #[test]
    fn the_cleavage_choice_reaches_the_diann_arguments() {
        let value = |args: &[String], flag: &str| -> String {
            let i = args.iter().position(|a| a == flag).unwrap();
            args[i + 1].clone()
        };
        let base = a_request("x.fasta");
        let args = predict_args(&base, &ModPlan::of(&base).unwrap(), Path::new("o")).unwrap();
        assert_eq!(value(&args, "--cut"), "K*,R*");
        assert_eq!(value(&args, "--missed-cleavages"), "1");

        let mut ns = a_request("x.fasta");
        ns.enzyme = "non_specific".into();
        ns.min_pep_len = 10;
        ns.max_pep_len = 10;
        ns.variable_mods = vec![site("GlyGly", 'K')];
        let args = predict_args(&ns, &ModPlan::of(&ns).unwrap(), Path::new("o")).unwrap();
        assert_eq!(value(&args, "--cut"), "**");
        assert_eq!(value(&args, "--missed-cleavages"), "9");
        assert!(!args.iter().any(|a| a == "--no-cut-after-mod"), "{args:?}");

        let mut cu = a_request("x.fasta");
        cu.enzyme = "custom".into();
        cu.cleavage_rule = Some("E*,D*".into());
        let args = predict_args(&cu, &ModPlan::of(&cu).unwrap(), Path::new("o")).unwrap();
        assert_eq!(value(&args, "--cut"), "E*,D*");
        cu.cleavage_rule = Some("E*,DXP".into());
        assert!(predict_args(&cu, &ModPlan::of(&cu).unwrap(), Path::new("o")).is_err());

        // An old request without the fields reads as Trypsin/P.
        let old: BuildRequest =
            serde_json::from_str(r#"{"fasta": "x.fasta", "out_dir": "o"}"#).unwrap();
        assert_eq!(cut_rule(&old).unwrap(), "K*,R*");
    }

    #[test]
    fn the_library_cache_key_covers_what_changes_the_library_and_nothing_else() {
        // Unique per process, like every other temp fixture in the workspace: this one
        // used a fixed name AND deleted the directory on entry, so two `cargo test` runs
        // on one machine raced and one of them lost its fasta mid-test (docs/14).
        let dir =
            std::env::temp_dir().join(format!("mumdia-diann-cachekey-{}", std::process::id()));
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
            |r: &mut BuildRequest| r.enzyme = "lys_c".into(),
            |r: &mut BuildRequest| r.enzyme = "non_specific".into(),
            |r: &mut BuildRequest| {
                r.enzyme = "custom".into();
                r.cleavage_rule = Some("K*,R*,D*,!*P".into());
            },
            |r: &mut BuildRequest| r.min_pr_mz = Some(400.0),
            |r: &mut BuildRequest| r.max_pr_mz = Some(1000.0),
            |r: &mut BuildRequest| r.min_fr_mz = Some(150.0),
            |r: &mut BuildRequest| r.max_fr_mz = Some(2000.0),
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
        let dir =
            std::env::temp_dir().join(format!("mumdia-diann-halfcache-{}", std::process::id()));
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
        let args = predict_args(&req, &ModPlan::of(&req).unwrap(), Path::new("out/lib")).unwrap();
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
    fn the_standard_selection_keeps_its_arguments_key_and_import() {
        // The two flags and the same choice spelled as lists are one library: same
        // arguments, same cache key, and the importer's default kept set.
        let dir = std::env::temp_dir().join(format!("mumdia-diann-mods-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fa = dir.join("a.fasta");
        std::fs::write(
            &fa,
            b">sp|P1|X
PEPTIDEK
",
        )
        .unwrap();
        let flags = a_request(fa.to_str().unwrap());
        let mut lists = flags.clone();
        lists.fixed_mods = vec![site("Carbamidomethyl", 'C')];
        lists.variable_mods = vec![site("Oxidation", 'M')];
        let (pf, pl) = (ModPlan::of(&flags).unwrap(), ModPlan::of(&lists).unwrap());
        assert!(pf.is_legacy() && pl.is_legacy());
        let out = Path::new("out/lib");
        assert_eq!(
            predict_args(&flags, &pf, out).unwrap(),
            predict_args(&lists, &pl, out).unwrap()
        );
        assert_eq!(
            library_cache_dir(&flags, "DIA-NN 2.0").unwrap(),
            library_cache_dir(&lists, "DIA-NN 2.0").unwrap()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_wider_selection_becomes_dia_nn_modifications_and_a_new_key() {
        let dir = std::env::temp_dir().join(format!("mumdia-diann-mods2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fa = dir.join("a.fasta");
        std::fs::write(
            &fa,
            b">sp|P1|X
PEPTIDEK
",
        )
        .unwrap();
        let base = a_request(fa.to_str().unwrap());
        let mut r = base.clone();
        r.fixed_mods = vec![site("Carbamidomethyl", 'C')];
        r.variable_mods = vec![
            site("Phospho", 'T'),
            site("Oxidation", 'M'),
            site("Phospho", 'S'),
            site("Phospho", 'Y'),
            site("GlyGly", 'K'),
        ];
        r.max_variable_mods = 2;
        let plan = ModPlan::of(&r).unwrap();
        assert!(!plan.is_legacy());
        let args = predict_args(&r, &plan, Path::new("out/lib")).unwrap();
        let after = |flag: &str| -> Vec<String> {
            args.iter()
                .enumerate()
                .filter(|(_, a)| *a == flag)
                .map(|(i, _)| args[i + 1].clone())
                .collect()
        };
        assert!(args.iter().any(|a| a == "--unimod4"));
        assert_eq!(after("--var-mods"), vec!["2".to_string()]);
        assert_eq!(
            after("--var-mod"),
            vec![
                "UniMod:121,114.042927,K".to_string(),
                "UniMod:35,15.994915,M".to_string(),
                "UniMod:21,79.966331,STY".to_string(),
            ]
        );
        assert_eq!(plan.unimod_ids(), vec![4, 21, 35, 121]);
        assert_eq!(after("--no-cut-after-mod"), vec!["UniMod:121".to_string()]);
        assert_ne!(
            library_cache_dir(&r, "DIA-NN 2.0").unwrap(),
            library_cache_dir(&base, "DIA-NN 2.0").unwrap()
        );
        // Homocysteinylation of K (issue #234) is passed by accession and mass like any
        // other catalogue entry, and the importer is told to keep its accession.
        let mut hcy = base.clone();
        hcy.fixed_mods = vec![site("Carbamidomethyl", 'C')];
        hcy.variable_mods = vec![site("HCysThiolactone", 'K'), site("Oxidation", 'M')];
        let hplan = ModPlan::of(&hcy).unwrap();
        assert!(!hplan.is_legacy());
        let hargs = predict_args(&hcy, &hplan, Path::new("out/lib")).unwrap();
        let hvar: Vec<&String> = hargs
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == "--var-mod")
            .map(|(i, _)| &hargs[i + 1])
            .collect();
        assert_eq!(
            hvar,
            vec!["UniMod:1270,117.024835,K", "UniMod:35,15.994915,M"]
        );
        assert!(!hargs.iter().any(|a| a == "--no-cut-after-mod"));
        assert_eq!(hplan.unimod_ids(), vec![4, 35, 1270]);
        // A name outside the catalogue is refused before DIA-NN runs.
        let mut bad = base.clone();
        bad.variable_mods = vec![site("NotAModification", 'K')];
        assert!(ModPlan::of(&bad).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn terminal(name: &str, position: &str) -> ModSite {
        ModSite {
            name: name.into(),
            residue: '*',
            position: position.into(),
        }
    }

    /// Protein N-terminal acetylation is DIA-NN's `*n` site, an argument of its own next to
    /// a residue acetyl, a cache key of its own, and its accession is kept at import.
    #[test]
    fn a_protein_n_terminal_site_is_the_star_n_token() {
        let dir = std::env::temp_dir().join(format!("mumdia-diann-nterm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fa = dir.join("a.fasta");
        std::fs::write(&fa, b">sp|P1|X\nMPEPTIDEK\n").unwrap();
        let base = a_request(fa.to_str().unwrap());
        let mut r = base.clone();
        r.fixed_mods = vec![site("Carbamidomethyl", 'C')];
        r.variable_mods = vec![
            site("Oxidation", 'M'),
            terminal("Acetyl", "protein_n_term"),
            site("Acetyl", 'K'),
        ];
        let plan = ModPlan::of(&r).unwrap();
        assert!(!plan.is_legacy());
        let args = predict_args(&r, &plan, Path::new("out/lib")).unwrap();
        let after = |flag: &str| -> Vec<String> {
            args.iter()
                .enumerate()
                .filter(|(_, a)| *a == flag)
                .map(|(i, _)| args[i + 1].clone())
                .collect()
        };
        assert_eq!(
            after("--var-mod"),
            vec![
                "UniMod:1,42.010565,K".to_string(),
                "UniMod:1,42.010565,*n".to_string(),
                "UniMod:35,15.994915,M".to_string(),
            ]
        );
        assert!(args.iter().any(|a| a == "--met-excision"));
        assert_eq!(plan.unimod_ids(), vec![1, 4, 35]);
        assert!(plan.key().contains("Acetyl@*n"), "{}", plan.key());
        let mut residue_only = r.clone();
        residue_only.variable_mods = vec![site("Oxidation", 'M'), site("Acetyl", 'K')];
        assert_ne!(
            library_cache_dir(&r, "DIA-NN 2.0").unwrap(),
            library_cache_dir(&residue_only, "DIA-NN 2.0").unwrap()
        );
        // `anywhere` is the residue site it always was.
        let mut spelled = residue_only.clone();
        spelled.variable_mods[1].position = "anywhere".into();
        assert_eq!(
            ModPlan::of(&spelled).unwrap(),
            ModPlan::of(&residue_only).unwrap()
        );
        // Refused: a residue on a terminal site, a C-terminus, fixed and variable N-termini.
        let mut bad = base.clone();
        bad.variable_mods = vec![ModSite {
            residue: 'S',
            ..terminal("Acetyl", "protein_n_term")
        }];
        assert!(ModPlan::of(&bad).is_err());
        bad.variable_mods = vec![terminal("Acetyl", "protein_c_term")];
        assert!(ModPlan::of(&bad).is_err());
        bad.variable_mods = vec![terminal("Acetyl", "protein_n_term")];
        bad.fixed_mods = vec![terminal("Propionyl", "peptide_n_term")];
        assert!(ModPlan::of(&bad).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The ZT Scan search space the Search screen sets with the ZT Scan recipe
    /// (docs/37_ztscan.md, section 11) reaches the prediction as the arguments the measured
    /// library was built with, and its accessions (1, 4, 35) reach both the importer and the
    /// decoy prediction, so no N-terminally acetylated precursor is dropped as unmapped.
    #[test]
    fn the_ztscan_search_space_reaches_the_prediction_and_both_imports() {
        let mut r = a_request("k562.fasta");
        r.min_pep_len = 6;
        r.max_pep_len = 40;
        r.min_charge = 1;
        r.max_charge = 4;
        r.max_variable_mods = 2;
        r.min_pr_mz = Some(392.0);
        r.max_pr_mz = Some(901.0);
        r.fixed_mods = vec![site("Carbamidomethyl", 'C')];
        r.variable_mods = vec![site("Oxidation", 'M'), terminal("Acetyl", "protein_n_term")];
        let plan = ModPlan::of(&r).unwrap();
        let args = predict_args(&r, &plan, Path::new("out/lib"))
            .unwrap()
            .join(" ");
        for want in [
            "--cut K*,R*",
            "--met-excision",
            "--missed-cleavages 1",
            "--min-pep-len 6",
            "--max-pep-len 40",
            "--min-pr-charge 1",
            "--max-pr-charge 4",
            "--min-pr-mz 392",
            "--max-pr-mz 901",
            "--unimod4",
            "--var-mods 2",
            "--var-mod UniMod:1,42.010565,*n",
            "--var-mod UniMod:35,15.994915,M",
        ] {
            assert!(args.contains(want), "{want} not in {args}");
        }
        let c = a_conversion(build_keep_unimod(&plan));
        let steps = conversion_steps(&c, Path::new("/cache/lib0/lib.parquet"));
        assert_eq!(
            steps[0].1[steps[0].1.len() - 2..],
            ["--keep-unimod", "1,4,35"]
        );
        assert_eq!(
            steps[2].1[steps[2].1.len() - 2..],
            ["--keep-unimod", "1,4,35"]
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
        let dir = std::env::temp_dir().join(format!("mumdia-diann-anchor-{}", std::process::id()));
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

    // ── the shared conversion and the import ────────────────────────────────

    fn a_conversion(keep_unimod: Option<Vec<u32>>) -> Conversion {
        Conversion {
            exe: PathBuf::from("/opt/diann/diann-linux"),
            python: PathBuf::from("/py/bin/python"),
            scripts: PathBuf::from("/app/scripts"),
            out_dir: PathBuf::from("/cache/lib0"),
            threads: 6,
            keep_unimod,
            decoy_args: Vec::new(),
            diann_mods: Vec::new(),
            fasta: None,
            remap: None,
            empty: String::new(),
        }
    }

    /// The builder's three conversion steps as they were spelled inline in `build`
    /// before the conversion was shared with the import, argument for argument.
    fn builder_steps_before_the_refactor(
        c: &Conversion,
        predicted: &Path,
        plan: &ModPlan,
    ) -> Vec<(&'static str, Vec<String>)> {
        let (out_dir, scripts) = (&c.out_dir, &c.scripts);
        let tgt_prec = out_dir.join("lib_precursors_targets.parquet");
        let tgt_frag = out_dir.join("lib_fragments_targets.parquet");
        let rev_prec = out_dir.join("lib_precursors_copied_decoys.parquet");
        let rev_frag = out_dir.join("lib_fragments_copied_decoys.parquet");
        let prec = out_dir.join("lib_precursors.parquet");
        let frag = out_dir.join("lib_fragments.parquet");
        let ids = || {
            plan.unimod_ids()
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let mut import = vec![
            scripts.join("import_diann_lib.py").display().to_string(),
            predicted.display().to_string(),
            tgt_prec.display().to_string(),
            tgt_frag.display().to_string(),
        ];
        if !plan.is_legacy() {
            import.push("--keep-unimod".into());
            import.push(ids());
        }
        let mut decoys = vec![
            scripts.join("predict_decoys.py").display().to_string(),
            "build".into(),
            rev_prec.display().to_string(),
            rev_frag.display().to_string(),
            prec.display().to_string(),
            frag.display().to_string(),
            "--diann".into(),
            c.exe.display().to_string(),
            "--threads".into(),
            c.threads.to_string(),
            "--work".into(),
            out_dir.join("decoy_prediction").display().to_string(),
        ];
        if !plan.is_legacy() {
            decoys.push("--keep-unimod".into());
            decoys.push(ids());
        }
        vec![
            ("converting to the MuMDIA schema", import),
            (
                "adding the decoy population",
                vec![
                    scripts.join("make_reverse_decoys.py").display().to_string(),
                    tgt_prec.display().to_string(),
                    tgt_frag.display().to_string(),
                    rev_prec.display().to_string(),
                    rev_frag.display().to_string(),
                ],
            ),
            ("predicting the decoy spectra with DIA-NN", decoys),
        ]
    }

    #[test]
    fn the_builder_runs_the_same_conversion_steps_as_before_they_were_shared() {
        let predicted = Path::new("/cache/lib0/lib.parquet");
        // The standard selection: the importer's default set, no --keep-unimod.
        let legacy = ModPlan::of(&a_request("x.fasta")).unwrap();
        let c = a_conversion(build_keep_unimod(&legacy));
        assert!(c.keep_unimod.is_none());
        assert_eq!(
            conversion_steps(&c, predicted),
            builder_steps_before_the_refactor(&c, predicted, &legacy)
        );
        // A wider selection: its accessions reach the importer and the decoy prediction,
        // and not the reversal, exactly as before.
        let mut req = a_request("x.fasta");
        req.variable_mods = vec![site("Oxidation", 'M'), site("Phospho", 'S')];
        req.fixed_mods = vec![site("Carbamidomethyl", 'C')];
        let wide = ModPlan::of(&req).unwrap();
        let c = a_conversion(build_keep_unimod(&wide));
        let steps = conversion_steps(&c, predicted);
        assert_eq!(
            steps,
            builder_steps_before_the_refactor(&c, predicted, &wide)
        );
        assert_eq!(
            steps[0].1[steps[0].1.len() - 2..],
            ["--keep-unimod", "4,21,35"]
        );
        assert!(!steps[1].1.iter().any(|a| a == "--keep-unimod"));
        assert_eq!(
            steps[2].1[steps[2].1.len() - 2..],
            ["--keep-unimod", "4,21,35"]
        );
        // The final tables are the ones `cached_library` and the search screen expect.
        assert_eq!(
            final_tables(&c.out_dir),
            (
                c.out_dir.join("lib_precursors.parquet"),
                c.out_dir.join("lib_fragments.parquet")
            )
        );
    }

    /// The importer writes the protein annotations into its precursor table's columns, which
    /// the decoy steps keep, so the final `lib_precursors.parquet` carries them. Every step
    /// writes into the one library folder, and the decoy prediction imports its own DIA-NN
    /// library in a subfolder, so its (decoy) import touches nothing of the targets'. An
    /// import has no FASTA to give the importer; a build gives its own (below).
    #[test]
    fn the_protein_annotations_land_beside_the_final_precursor_table() {
        let c = a_conversion(None);
        let steps = conversion_steps(&c, Path::new("/in/lib.parquet"));
        let (prec, _) = final_tables(&c.out_dir);
        assert_eq!(prec.parent(), Some(c.out_dir.as_path()));
        // import: <script> <library> <target precursors> <target fragments>
        assert_eq!(
            Path::new(&steps[0].1[2]).parent(),
            Some(c.out_dir.as_path())
        );
        assert!(!steps[0]
            .1
            .iter()
            .any(|a| a == "--no-annotations" || a == "--annotations" || a == "--fasta"));
        // reversal: its output precursors are in the same folder.
        assert_eq!(
            Path::new(&steps[1].1[3]).parent(),
            Some(c.out_dir.as_path())
        );
        // decoy prediction: the work folder is below the library folder, not the folder.
        let w = steps[2].1.iter().position(|a| a == "--work").unwrap();
        let work = PathBuf::from(&steps[2].1[w + 1]);
        assert_ne!(work, c.out_dir);
        assert_eq!(work.parent(), Some(c.out_dir.as_path()));
    }

    /// A build gives the importer the FASTA it predicted from, so the importer can name from
    /// its headers what the DIA-NN library leaves empty; nothing else in the steps changes.
    #[test]
    fn a_build_gives_the_importer_its_fasta() {
        let plain = a_conversion(None);
        let mut c = a_conversion(None);
        c.fasta = Some(PathBuf::from("/data/proteome.fasta"));
        let (with, without) = (
            conversion_steps(&c, Path::new("/in/lib.parquet")),
            conversion_steps(&plain, Path::new("/in/lib.parquet")),
        );
        let n = without[0].1.len();
        assert_eq!(with[0].1[..n], without[0].1[..]);
        assert_eq!(with[0].1[n..], ["--fasta", "/data/proteome.fasta"]);
        assert_eq!(with[1..], without[1..]);
    }

    /// An import that re-maps to a FASTA hands the importer that FASTA and the cleavage rule;
    /// nothing else in the steps changes, and the decoys follow the remapped targets.
    #[test]
    fn an_import_can_re_map_its_peptides_to_a_fasta() {
        let plain = a_conversion(None);
        let mut c = a_conversion(None);
        c.remap = Some(Remap {
            fasta: PathBuf::from("/data/new.fasta"),
            rule: "K*,R*".into(),
        });
        let (with, without) = (
            conversion_steps(&c, Path::new("/in/lib.parquet")),
            conversion_steps(&plain, Path::new("/in/lib.parquet")),
        );
        let n = without[0].1.len();
        assert_eq!(with[0].1[..n], without[0].1[..]);
        assert_eq!(
            with[0].1[n..],
            [
                "--remap-fasta",
                "/data/new.fasta",
                "--remap-cleavage-rule",
                "K*,R*"
            ]
        );
        assert_eq!(with[1..], without[1..]);
    }

    #[test]
    fn an_import_request_checks_its_re_mapping() {
        let mut req: ImportRequest =
            serde_json::from_str(r#"{"library": "a.parquet", "out_dir": "o"}"#).unwrap();
        assert_eq!(req.remap().unwrap(), None);
        req.remap_fasta = Some("  ".into());
        assert_eq!(req.remap().unwrap(), None);
        req.remap_fasta = Some("/no/such/proteins.fasta".into());
        assert!(req.remap().unwrap_err().contains("not a file"));
        let dir = std::env::temp_dir().join(format!("mumdia-diann-remap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fa = dir.join("p.fasta");
        std::fs::write(&fa, b">P\nPEPTIDEK\n").unwrap();
        req.remap_fasta = Some(fa.display().to_string());
        assert_eq!(req.remap().unwrap().unwrap().rule, "K*,R*");
        req.remap_enzyme = Some("custom".into());
        assert!(req.remap().is_err());
        req.remap_cleavage_rule = Some("K*,!*P".into());
        assert_eq!(req.remap().unwrap().unwrap().rule, "K*,!*P");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The reported failure: a build with methyl on H and trimethyl on K, which DIA-NN does
    /// not know by itself. The prediction declared them (`--var-mod`), the re-export and the
    /// decoys' prediction did not, and the decoys' prediction then saved 0 precursors. Every
    /// DIA-NN run after the prediction now declares each modification of the build.
    #[test]
    fn every_dia_nn_run_after_the_prediction_declares_the_builds_modifications() {
        let dir = std::env::temp_dir().join(format!("mumdia-diann-decl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fa = dir.join("a.fasta");
        std::fs::write(&fa, b">sp|P1|X\nMPEPTIDEK\n").unwrap();
        let mut r = a_request(fa.to_str().unwrap());
        r.fixed_mods = vec![site("Carbamidomethyl", 'C')];
        r.variable_mods = vec![
            site("Methyl", 'H'),
            site("Oxidation", 'M'),
            site("Phospho", 'H'),
            site("Phospho", 'S'),
            site("Phospho", 'T'),
            site("Phospho", 'Y'),
            site("Trimethyl", 'K'),
        ];
        let plan = ModPlan::of(&r).unwrap();
        let want = [
            "UniMod:4,57.021464",
            "UniMod:21,79.966331",
            "UniMod:34,14.015650",
            "UniMod:35,15.994915",
            "UniMod:37,42.046950",
        ];
        assert_eq!(plan.declarations(), want);
        // The masses are the ones the prediction itself was given.
        let predict = predict_args(&r, &plan, Path::new("out/lib"))
            .unwrap()
            .join(" ");
        assert!(
            predict.contains("--var-mod UniMod:34,14.015650,H"),
            "{predict}"
        );
        assert!(
            predict.contains("--var-mod UniMod:37,42.046950,K"),
            "{predict}"
        );

        let mut c = a_conversion(build_keep_unimod(&plan));
        c.diann_mods = plan.declarations();
        // The `.speclib` re-export.
        let re = reexport_args(
            Path::new("/cache/lib0/lib.predicted.speclib"),
            Path::new("/cache/lib0/lib.parquet"),
            6,
            &c.diann_mods,
        );
        let declared: Vec<&str> = re
            .windows(2)
            .filter(|w| w[0] == "--mod")
            .map(|w| w[1].as_str())
            .collect();
        assert_eq!(declared, want);
        // `--mod` declares a mass; it must never become a modification of the library.
        assert!(!re.iter().any(|a| a == "--var-mod" || a == "--fixed-mod"));
        // The decoys' prediction and its re-export, through `predict_decoys.py`.
        let steps = conversion_steps(&c, Path::new("/cache/lib0/lib.parquet"));
        let decoys = &steps[2].1;
        assert!(decoys[0].ends_with("predict_decoys.py"));
        let passed: Vec<String> = decoys
            .iter()
            .filter_map(|a| a.strip_prefix("--diann-mod=").map(str::to_string))
            .collect();
        assert_eq!(passed, want);
        // The importer and the decoy builder are not DIA-NN runs and take no declaration.
        assert!(!steps[0]
            .1
            .iter()
            .chain(&steps[1].1)
            .any(|a| a.contains("diann-mod")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_import_declares_every_catalogue_modification() {
        let all = declarations(crate::mods::catalogue().iter());
        assert_eq!(all.len(), catalogue_unimod_ids().len());
        for d in [
            "UniMod:34,14.015650",
            "UniMod:37,42.046950",
            "UniMod:4,57.021464",
        ] {
            assert!(all.iter().any(|x| x == d), "{d} in {all:?}");
        }
    }

    #[test]
    fn the_re_export_is_the_command_line_the_builder_always_ran() {
        let args = reexport_args(
            Path::new("/cache/lib0/lib.predicted.speclib"),
            Path::new("/cache/lib0/lib.parquet"),
            6,
            &[],
        );
        assert_eq!(
            args,
            [
                "--lib",
                "/cache/lib0/lib.predicted.speclib",
                "--gen-spec-lib",
                "--out-lib",
                "/cache/lib0/lib.parquet",
                "--threads",
                "6"
            ]
        );
    }

    #[test]
    fn the_import_keeps_every_modification_the_catalogue_names() {
        let ids = catalogue_unimod_ids();
        // The importer's default set and the usual variable modifications are all in it.
        for id in [1, 4, 21, 35, 44, 48, 376] {
            assert!(ids.contains(&id), "UniMod:{id} is kept");
        }
        assert!(ids.windows(2).all(|w| w[0] < w[1]), "sorted and unique");
        let c = a_conversion(Some(ids.clone()));
        let steps = conversion_steps(&c, Path::new("/in/lib.parquet"));
        let joined = ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(steps[0].1.last(), Some(&joined));
        assert_eq!(steps[2].1.last(), Some(&joined));
        // The import reads the chosen library where it is; it is not copied.
        assert_eq!(
            steps[0].1[1],
            Path::new("/in/lib.parquet").display().to_string()
        );
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("mumdia-diann-import-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A file with Parquet's framing and a footer naming `columns` the way Thrift's
    /// compact protocol does (field header, varint length, bytes). Enough for the
    /// footer check, which reads nothing else.
    fn fake_parquet(path: &Path, columns: &[&str]) {
        let mut footer = vec![0x15, 0x02, 0x19, 0x6c];
        for c in columns {
            footer.push(0x48);
            footer.push(c.len() as u8);
            footer.extend_from_slice(c.as_bytes());
        }
        let mut bytes = b"PAR1".to_vec();
        bytes.extend_from_slice(&[0u8; 64]);
        bytes.extend_from_slice(&footer);
        bytes.extend_from_slice(&(footer.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"PAR1");
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn a_diann_parquet_library_is_recognised_from_its_footer() {
        let dir = scratch_dir("ok");
        let lib = dir.join("report-lib.parquet");
        let mut cols = DIANN_LIBRARY_COLUMNS.to_vec();
        cols.extend(["Protein.Ids", "Decoy", "IM"]);
        fake_parquet(&lib, &cols);
        assert_eq!(
            inspect_library(&lib).unwrap(),
            DiannLibrary::Parquet(lib.clone())
        );
        // The extension is matched without regard to case.
        let upper = dir.join("LIB.PARQUET");
        fake_parquet(&upper, &cols);
        assert!(matches!(
            inspect_library(&upper),
            Ok(DiannLibrary::Parquet(_))
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_column_name_is_matched_whole_not_inside_another() {
        let mut footer = vec![0x48, 3];
        footer.extend_from_slice(b"iRT");
        assert!(!footer_names(&footer, "RT"));
        assert!(footer_names(&footer, "iRT"));
        // A name longer than 127 bytes takes a two-byte varint length.
        let long = "x".repeat(200);
        let mut footer = vec![0x48, 0xc8, 0x01];
        footer.extend_from_slice(long.as_bytes());
        assert!(footer_names(&footer, &long));
    }

    #[test]
    fn what_is_not_a_diann_library_is_refused_before_anything_runs() {
        let dir = scratch_dir("refused");

        let e = inspect_library(&dir.join("absent.parquet")).unwrap_err();
        assert!(e.contains("is not a file"), "{e}");

        // A search report: precursor columns, no fragment columns.
        let report = dir.join("report.parquet");
        fake_parquet(
            &report,
            &[
                "Modified.Sequence",
                "Stripped.Sequence",
                "Precursor.Charge",
                "Precursor.Mz",
                "RT",
            ],
        );
        let e = inspect_library(&report).unwrap_err();
        assert!(e.contains("search report"), "{e}");
        assert!(e.contains("Product.Mz"), "names what is missing: {e}");

        // MuMDIA's own table is sent where it belongs.
        let ours = dir.join("lib_precursors.parquet");
        fake_parquet(
            &ours,
            &["candidate_id", "peptidoform", "precursor_mz", "label"],
        );
        let e = inspect_library(&ours).unwrap_err();
        assert!(e.contains("MuMDIA library table"), "{e}");

        // Something else entirely.
        let other = dir.join("other.parquet");
        fake_parquet(&other, &["a", "b"]);
        let e = inspect_library(&other).unwrap_err();
        assert!(
            e.contains("not a DIA-NN fragment-level spectral library"),
            "{e}"
        );

        // Not Parquet at all, and a Parquet cut short.
        let text = dir.join("renamed.parquet");
        std::fs::write(&text, "Modified.Sequence\tPrecursor.Mz\n").unwrap();
        assert!(inspect_library(&text)
            .unwrap_err()
            .contains("not a Parquet file"));
        let cut = dir.join("cut.parquet");
        let mut bytes = b"PAR1".to_vec();
        bytes.extend_from_slice(&1000u32.to_le_bytes());
        bytes.extend_from_slice(b"PAR1");
        std::fs::write(&cut, bytes).unwrap();
        assert!(inspect_library(&cut).unwrap_err().contains("truncated"));

        // A text library is refused by name, with the way to make it Parquet.
        let tsv = dir.join("lib.tsv");
        std::fs::write(&tsv, "PrecursorMz\tProductMz\n").unwrap();
        let e = inspect_library(&tsv).unwrap_err();
        assert!(
            e.contains("text library") && e.contains("--gen-spec-lib"),
            "{e}"
        );

        // A .speclib is DIA-NN's to check, but an empty one is not a library.
        let empty = dir.join("lib.predicted.speclib");
        std::fs::write(&empty, b"").unwrap();
        assert!(inspect_library(&empty).unwrap_err().contains("empty"));
        std::fs::write(&empty, b"\x01\x02\x03").unwrap();
        assert_eq!(
            inspect_library(&empty).unwrap(),
            DiannLibrary::Speclib(empty.clone())
        );

        let e = inspect_library(&tsv.with_extension("msp")).unwrap_err();
        assert!(e.contains("is not a file"), "{e}");
        let msp = dir.join("lib.msp");
        std::fs::write(&msp, b"x").unwrap();
        assert!(inspect_library(&msp).unwrap_err().contains("neither"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn a_status(runs: bool, ack: bool, version: &str) -> Status {
        Status {
            path: runs.then(|| "/opt/diann/diann-linux".to_string()),
            source: runs.then(|| "path".to_string()),
            version: (!version.is_empty()).then(|| version.to_string()),
            runs,
            licence_acknowledged: ack,
            licence_assumed: false,
            writes_parquet: version.is_empty() || writes_parquet_libraries(version),
            error: (!runs).then(|| "no working DIA-NN was found".to_string()),
        }
    }

    #[test]
    fn an_import_without_a_working_diann_is_refused_and_says_why() {
        let parquet = DiannLibrary::Parquet(PathBuf::from("lib.parquet"));
        let speclib = DiannLibrary::Speclib(PathBuf::from("lib.speclib"));

        // No DIA-NN: refused, never converted with copied decoys.
        for lib in [&parquet, &speclib] {
            let e = import_refusal(&a_status(false, false, ""), lib).unwrap_err();
            assert!(e.contains("predict the spectra of the decoys"), "{e}");
            assert!(e.contains("FDR optimistic"), "{e}");
            assert!(e.contains("Install DIA-NN 2.x"), "{e}");
            assert!(e.contains("licence"), "{e}");
            assert!(
                !e.contains("(no working DIA-NN was found)"),
                "no repetition: {e}"
            );
        }
        // A configured DIA-NN that stopped working says what went wrong.
        let mut broken = a_status(false, true, "");
        broken.error = Some("/opt/diann is no longer there".into());
        let e = import_refusal(&broken, &parquet).unwrap_err();
        assert!(e.contains("(/opt/diann is no longer there)"), "{e}");

        // Licence not acknowledged.
        let e = import_refusal(&a_status(true, false, "DIA-NN 2.7.0"), &parquet).unwrap_err();
        assert!(
            e.contains("licence notice has not been acknowledged"),
            "{e}"
        );
        assert!(e.contains("Setup"), "{e}");

        // 1.8.x writes no Parquet: a .speclib cannot even be re-exported, and the decoy
        // prediction would fail for either input.
        let old = a_status(
            true,
            true,
            "DIA-NN 1.8.1 (Data-Independent Acquisition by Neural Networks)",
        );
        assert!(!old.writes_parquet);
        let e = import_refusal(&old, &speclib).unwrap_err();
        assert!(
            e.contains("DIA-NN 1.8.1") && e.contains("re-export this .speclib"),
            "{e}"
        );
        assert!(e.contains("Install DIA-NN 2.x"), "{e}");
        let e = import_refusal(&old, &parquet).unwrap_err();
        assert!(e.contains("decoys' spectra"), "{e}");

        // A working, acknowledged 2.x is what runs.
        assert_eq!(
            import_refusal(&a_status(true, true, "DIA-NN 2.7.0 Academia"), &speclib).unwrap(),
            PathBuf::from("/opt/diann/diann-linux")
        );
    }

    #[test]
    fn an_import_never_writes_into_a_folder_that_holds_a_library() {
        let dir = scratch_dir("outdir");
        check_import_out_dir(&dir).unwrap();
        check_import_out_dir(&dir.join("not-yet-created")).unwrap();
        std::fs::write(dir.join("lib_fragments.parquet"), b"x").unwrap();
        let e = check_import_out_dir(&dir).unwrap_err();
        assert!(
            e.contains("lib_fragments.parquet") && e.contains("empty folder"),
            "{e}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_import_is_refused_for_the_file_before_diann_is_asked() {
        // The file is checked first, so the refusal names the file whatever DIA-NN state
        // the machine running the tests is in.
        let dir = scratch_dir("order");
        let tsv = dir.join("lib.tsv");
        std::fs::write(&tsv, "x").unwrap();
        let e = import(
            Arc::new(Builder::default()),
            ImportRequest {
                library: tsv.display().to_string(),
                out_dir: dir.join("out").display().to_string(),
                threads: 1,
                remap_fasta: None,
                remap_enzyme: None,
                remap_cleavage_rule: None,
            },
        )
        .unwrap_err();
        assert!(e.contains("text library"), "{e}");
        assert!(
            !dir.join("out").exists(),
            "nothing is created for a refused import"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The footer check on a real DIA-NN library: `MUMDIA_TEST_DIANN_LIB=<lib.parquet>
    /// cargo test --lib -- --ignored real_diann_library`.
    #[test]
    #[ignore]
    fn a_real_diann_library_passes_the_footer_check() {
        let Ok(p) = std::env::var("MUMDIA_TEST_DIANN_LIB") else {
            return;
        };
        let p = PathBuf::from(p);
        assert_eq!(
            inspect_library(&p).unwrap(),
            DiannLibrary::Parquet(p.clone())
        );
    }
}
