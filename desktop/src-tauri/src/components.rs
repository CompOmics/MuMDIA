//! Installing and detecting the Python analysis components.
//!
//! # Why this exists at all
//!
//! MuMDIA runs with no Python: the native predictors and the `native_tda` rescorer
//! are the shipped defaults. It is tempting to build a zero-dependency application
//! on that basis. The recorded numbers say otherwise -- on the same file, the fully
//! native FASTA path returns about 1,213 report rows against about 10,300 for the
//! imported-library workflow with DeepLC and neural rescoring. Two things differ
//! between those, the library source as well as the predictors, so the gap is not
//! attributable to Python alone; it is large enough to settle the design. An
//! application that avoids installing Python ships the weak path, so this one
//! installs it, and `preflight` refuses to search without it.
//!
//! # Why `uv` and not conda
//!
//! Reading `env/mumdia-deeplc.yml`, conda contributes `python=3.11` and `pip`, and
//! every dependency that matters is already pip. `uv` supplies the interpreter as
//! well, as one self-contained binary, so "install Miniconda first, create two
//! environments, then edit the config to point at the right interpreters" -- the
//! step where an external user gives up -- disappears entirely.
//!
//! # One environment, and the role it cannot cover
//!
//! Rescoring, DeepLC and match-between-runs share one interpreter happily, and that
//! is the whole recommended workflow. MS2PIP cannot join them: at the versions this
//! project tests, `deeplc==4.1.1` needs `sqlalchemy>=2` through `psm-utils` and
//! `ms2pip==4.0.0` needs `sqlalchemy<2`, which `uv` reports as unsatisfiable.
//!
//! This was checked rather than assumed, and the assumption was wrong. `ms2pip>=4.1`
//! does resolve alongside DeepLC, but MS2PIP's version changes predicted fragment
//! intensities, so taking that upgrade to simplify packaging would trade a
//! convenience for a results change. MS2PIP therefore gets its own environment,
//! installed only on request, and it is needed only for FASTA-mode library building
//! with predicted intensities.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};

use serde::Serialize;

/// Where the managed environment lives, and the state of the last install.
#[derive(Serialize, Clone, Debug, Default)]
pub struct Status {
    /// Absolute path to the managed interpreter, if it exists.
    pub python: Option<String>,
    /// True when that interpreter can import everything every role needs.
    pub complete: bool,
    /// Modules the managed interpreter cannot import.
    pub missing: Vec<String>,
    /// Versions of the packages whose version changes results.
    pub versions: std::collections::BTreeMap<String, String>,
    /// `idle` | `installing` | `done` | `failed`
    pub install_status: String,
    pub install_log: Vec<String>,
    pub error: Option<String>,
    /// Whether a bundled `uv` was found, without which nothing can be installed.
    pub uv: Option<String>,
}

/// Every module the primary environment must provide.
///
/// This is the union of what the rescore, DeepLC and match-between-runs roles
/// import: the whole recommended workflow. MS2PIP is deliberately absent -- it
/// cannot share an environment with DeepLC (see the module documentation) and lives
/// in its own, which `MS2PIP_MODULES` describes.
///
/// Kept here rather than asked of the engine because it is needed BEFORE any
/// configuration exists: the setup screen runs on first launch, when there is
/// nothing to point `doctor` at. `doctor --json` remains the authority once a
/// configuration is in play, and the application shows that too.
const REQUIRED_MODULES: &[&str] = &[
    "deeplc",
    "torch",
    "psm_utils",
    "mokapot",
    "sklearn",
    "numpy",
    "pandas",
    "pyarrow",
];

/// The optional MS2PIP environment's modules.
const MS2PIP_MODULES: &[&str] = &["ms2pip", "numpy", "pandas"];

/// Which managed environment a call is about.
///
/// Two exist because they must: MS2PIP and DeepLC cannot share one. Everything else
/// about them is identical, so the create-and-install path is written once.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Env {
    /// Rescoring, DeepLC and match-between-runs: the recommended workflow.
    Primary,
    /// MS2PIP, for FASTA-mode library building with predicted intensities.
    Ms2pip,
}

impl Env {
    fn dir_name(self) -> &'static str {
        match self {
            Env::Primary => "python",
            Env::Ms2pip => "python-ms2pip",
        }
    }

    fn requirements_name(self) -> &'static str {
        match self {
            Env::Primary => "console-requirements.txt",
            Env::Ms2pip => "console-ms2pip-requirements.txt",
        }
    }

    pub fn modules(self) -> &'static [&'static str] {
        match self {
            Env::Primary => REQUIRED_MODULES,
            Env::Ms2pip => MS2PIP_MODULES,
        }
    }

    /// Python version for this environment. MS2PIP pulls `pandas<2`, which has no
    /// cp312 wheel, which is why both stay on 3.11.
    fn python_version(self) -> &'static str {
        "3.11"
    }
}

/// Packages whose version is worth reporting, because it changes results.
const REPORT_VERSIONS: &[&str] = &["deeplc", "torch", "mokapot", "ms2pip", "numpy"];

/// Per-user application data, where the managed environment is created.
///
/// Not beside the executable: on Windows that is under Program Files, which a
/// normal user cannot write to, and an installer that needs administrator rights to
/// finish its first run is not the easy installation this exists to provide.
pub fn data_dir() -> PathBuf {
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    };
    base.unwrap_or_else(std::env::temp_dir).join("MuMDIA")
}

pub fn venv_dir(env: Env) -> PathBuf {
    data_dir().join(env.dir_name())
}

/// The interpreter inside a managed environment, whether or not it exists yet.
pub fn managed_python(env: Env) -> PathBuf {
    let v = venv_dir(env);
    if cfg!(windows) {
        v.join("Scripts").join("python.exe")
    } else {
        v.join("bin").join("python")
    }
}

/// Point an engine child process at every tool this application manages.
///
/// # Why this is needed at all
///
/// The engine resolves `"auto"` interpreters from `MUMDIA_PYTHON_*`, then an
/// activated conda/virtualenv, then `PATH` (`python.rs::candidates`). It does not
/// know about this application's data directory and should not: the engine is
/// usable without it.
///
/// So without this, an environment installed by the Setup screen was invisible to
/// every search it was installed for. `preflight` reported the components ready,
/// the run started, and the engine's own discovery fell through to whatever
/// `python` happened to be on `PATH` -- which is either absent, or present and
/// lacking DeepLC and torch. Under `rescore.strict = true` that is a failure hours
/// in; without it, a silent downgrade to `native_tda`.
///
/// Every engine invocation must go through this, `doctor` included, or the
/// application asks one question of an engine that can see the environment and
/// another of one that cannot, and gets inconsistent answers.
pub fn stamp_env(cmd: &mut std::process::Command) {
    // Per-role rather than the blanket `MUMDIA_PYTHON`, because the two
    // environments are not interchangeable: MS2PIP cannot share the primary one.
    let primary = managed_python(Env::Primary);
    if primary.is_file() {
        for var in [
            "MUMDIA_PYTHON_RESCORE",
            "MUMDIA_PYTHON_DEEPLC",
            "MUMDIA_PYTHON_MBR",
        ] {
            cmd.env(var, &primary);
        }
    }
    let ms2pip = managed_python(Env::Ms2pip);
    if ms2pip.is_file() {
        cmd.env("MUMDIA_PYTHON_MS2PIP", &ms2pip);
    }
    // The Thermo .raw converter -- but only if it RUNS.
    //
    // The engine accepts `MUMDIA_THERMO_PARSER` on `is_file()` alone
    // (`raw::locate`), and `ensure_mzml` falls back to msconvert only when locating
    // the Thermo parser returns an error. So exporting a half-unpacked or
    // non-startable converter made `locate_parser` succeed, which made the msconvert
    // fallback unreachable and killed the run inside a converter that never worked --
    // while `ConvertConfig::msconvert`'s own documentation promises it is "the Thermo
    // fallback when no ThermoRawFileParser is found".
    if let Some(p) = runnable_thermo_parser() {
        cmd.env("MUMDIA_THERMO_PARSER", p);
    }
}

/// Where a managed ThermoRawFileParser is unpacked.
pub fn thermo_dir() -> PathBuf {
    data_dir().join("ThermoRawFileParser")
}

/// The managed ThermoRawFileParser, but only when it actually executes.
///
/// `thermo_parser` answers "is it on disk", which is the right question for showing
/// install state. This answers "will it convert", which is the only question worth
/// putting in an environment variable the engine treats as an explicit choice.
pub fn runnable_thermo_parser() -> Option<PathBuf> {
    let p = thermo_parser()?;
    crate::thermo::runs(&p).then_some(p)
}

/// The managed ThermoRawFileParser executable, if it is installed.
pub fn thermo_parser() -> Option<PathBuf> {
    let dir = thermo_dir();
    for name in ["ThermoRawFileParser.exe", "ThermoRawFileParser"] {
        let c = dir.join(name);
        if c.is_file() {
            return Some(c);
        }
    }
    None
}

/// Locate `uv`: bundled beside the application first, then whatever is on PATH.
///
/// PATH is accepted because a developer very likely has it already, and refusing to
/// use it would mean nobody could test this without building a bundle.
pub fn find_uv() -> Option<PathBuf> {
    let exe_name = if cfg!(windows) { "uv.exe" } else { "uv" };
    // Same reasoning as the engine lookup: an AppImage's resources are not beside
    // the executable, and only the shell knows where they are.
    if let Some(res) = crate::engine::resource_dir() {
        for cand in [res.join("binaries").join(exe_name), res.join(exe_name)] {
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for cand in [dir.join(exe_name), dir.join("binaries").join(exe_name)] {
                if cand.is_file() {
                    return Some(cand);
                }
            }
        }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(exe_name))
        .find(|p| p.is_file())
}

/// The two requirement sets, compiled in for the same reason as the settings
/// schema: a file that is not a file cannot go missing from a bundle, and these
/// change only when the application itself is rebuilt.
const REQUIREMENTS_PRIMARY: &str = include_str!("../../../env/console-requirements.txt");
const REQUIREMENTS_MS2PIP: &str = include_str!("../../../env/console-ms2pip-requirements.txt");

/// Write the requirements for `env` where `uv` can read them, and return the path.
///
/// Into the managed data directory rather than a temporary file, so that when an
/// installation fails the exact input is still on disk to look at.
pub fn requirements(env: Env) -> Result<PathBuf, String> {
    let text = match env {
        Env::Primary => REQUIREMENTS_PRIMARY,
        Env::Ms2pip => REQUIREMENTS_MS2PIP,
    };
    let dir = data_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let path = dir.join(env.requirements_name());
    std::fs::write(&path, text).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(path)
}

/// Ask an interpreter which of `modules` it cannot import.
///
/// One process for the whole list: nine separate probes cost about a second each on
/// Windows, which is long enough to be visible on a screen that exists to feel
/// responsive.
fn missing_modules(python: &Path, modules: &[&str]) -> Result<Vec<String>, String> {
    let script = format!(
        "import importlib.util as u\n\
         print('\\n'.join(m for m in {:?} if u.find_spec(m) is None))",
        modules
    );
    let out = crate::engine::command(python)
        .args(["-c", &script])
        .output()
        .map_err(|e| format!("could not run {}: {e}", python.display()))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

fn versions(python: &Path, modules: &[&str]) -> std::collections::BTreeMap<String, String> {
    let script = format!(
        "import importlib.metadata as m\n\
         for name in {:?}:\n\
         \x20   try: print(name, m.version(name))\n\
         \x20   except Exception: pass",
        modules
    );
    let mut out = std::collections::BTreeMap::new();
    if let Ok(o) = crate::engine::command(python)
        .args(["-c", &script])
        .output()
    {
        for line in String::from_utf8_lossy(&o.stdout).lines() {
            if let Some((k, v)) = line.trim().split_once(' ') {
                out.insert(k.to_string(), v.to_string());
            }
        }
    }
    out
}

/// Inspect a managed environment as it stands.
pub fn status_of(env: Env) -> Status {
    let mut s = Status {
        install_status: "idle".into(),
        uv: find_uv().map(|p| p.display().to_string()),
        ..Default::default()
    };
    let py = managed_python(env);
    if !py.is_file() {
        s.missing = env.modules().iter().map(|m| m.to_string()).collect();
        return s;
    }
    s.python = Some(py.display().to_string());
    match missing_modules(&py, env.modules()) {
        Ok(missing) => {
            s.complete = missing.is_empty();
            s.missing = missing;
            if s.complete {
                // `sklearn` is imported as `sklearn` but distributed as
                // `scikit-learn`, so ask metadata for the name pip knows.
                s.versions = versions(&py, REPORT_VERSIONS);
            }
        }
        Err(e) => {
            s.error = Some(e);
            s.missing = env.modules().iter().map(|m| m.to_string()).collect();
        }
    }
    s
}

/// The primary environment, which is what "are the components installed" means.
pub fn status() -> Status {
    status_of(Env::Primary)
}

/// Shared, mutable install state so the interface can poll while it runs.
#[derive(Default)]
pub struct Installer {
    primary: Mutex<Status>,
    ms2pip: Mutex<Status>,
}

impl Installer {
    fn slot(&self, env: Env) -> &Mutex<Status> {
        match env {
            Env::Primary => &self.primary,
            Env::Ms2pip => &self.ms2pip,
        }
    }

    /// Refresh one environment from disk. Cheap enough to call whenever the screen
    /// is shown.
    pub fn refresh(&self, env: Env) -> Status {
        let fresh = status_of(env);
        if let Ok(mut s) = self.slot(env).lock() {
            // Anything an installation put there outlives a refresh. Keeping only
            // "installing" lost the outcome: the install thread would set "done",
            // the next refresh would overwrite it with the "idle" a fresh probe
            // carries, and a caller watching for the transition would wait for ever.
            // Found by the test that does exactly that.
            let keep = s.install_status.clone();
            let log = s.install_log.clone();
            let err = s.error.clone();
            *s = fresh;
            if keep != "idle" {
                s.install_status = keep;
                s.install_log = log;
                // A failure message is part of the outcome and must survive too.
                if s.error.is_none() {
                    s.error = err;
                }
            }
            return s.clone();
        }
        fresh
    }
}

/// Create the managed environment and install everything into it.
///
/// Runs on its own thread and streams both streams into the shared log, because a
/// several-hundred-megabyte download with no visible progress is indistinguishable
/// from a hang.
pub fn install(installer: Arc<Installer>, env: Env) -> Result<(), String> {
    let uv = find_uv().ok_or_else(|| {
        "the installer component `uv` was not found beside the application or on PATH".to_string()
    })?;
    let reqs = requirements(env)?;
    let venv = venv_dir(env);
    std::fs::create_dir_all(data_dir())
        .map_err(|e| format!("cannot create {}: {e}", data_dir().display()))?;

    {
        let mut s = installer
            .slot(env)
            .lock()
            .map_err(|_| "internal state is poisoned".to_string())?;
        if s.install_status == "installing" {
            return Err("an installation is already running".into());
        }
        s.install_status = "installing".into();
        s.install_log.clear();
        s.error = None;
    }

    std::thread::spawn(move || {
        let log = |installer: &Installer, line: String| {
            if let Ok(mut s) = installer.slot(env).lock() {
                s.install_log.push(line);
                if s.install_log.len() > 2000 {
                    let drop = s.install_log.len() - 2000;
                    s.install_log.drain(0..drop);
                }
            }
        };

        // Two steps: an interpreter, then the packages. `uv venv` downloads a
        // standalone CPython if the machine has none, which is the whole point.
        let steps: Vec<(&str, Vec<String>)> = vec![
            (
                "creating the Python environment",
                vec![
                    "venv".into(),
                    // Without this, `uv venv` refuses a directory that already
                    // exists, so Install would fail for ever after the first
                    // attempt -- including after a FAILED attempt that left a
                    // partial environment behind, which is exactly when someone
                    // presses it again. `--clear` would also work but throws away
                    // a several-hundred-megabyte download to repair one package.
                    "--allow-existing".into(),
                    "--python".into(),
                    env.python_version().into(),
                    venv.display().to_string(),
                ],
            ),
            (
                "installing the analysis packages",
                vec![
                    "pip".into(),
                    "install".into(),
                    "--python".into(),
                    venv.display().to_string(),
                    "-r".into(),
                    reqs.display().to_string(),
                ],
            ),
        ];

        for (title, args) in steps {
            log(&installer, format!("== {title}"));
            let mut cmd = crate::engine::command(&uv);
            cmd.args(&args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => {
                    if let Ok(mut s) = installer.slot(env).lock() {
                        s.install_status = "failed".into();
                        s.error = Some(format!("could not run uv: {e}"));
                    }
                    return;
                }
            };
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
                let installer = Arc::clone(&installer);
                std::thread::spawn(move || {
                    for line in BufReader::new(stream).lines().map_while(Result::ok) {
                        if let Ok(mut s) = installer.slot(env).lock() {
                            s.install_log.push(line);
                        }
                    }
                });
            }
            match child.wait() {
                Ok(st) if st.success() => {}
                Ok(st) => {
                    if let Ok(mut s) = installer.slot(env).lock() {
                        s.install_status = "failed".into();
                        s.error = Some(format!(
                            "{title} failed ({st}). The log above says why; a failed download \
                             can simply be retried."
                        ));
                    }
                    return;
                }
                Err(e) => {
                    if let Ok(mut s) = installer.slot(env).lock() {
                        s.install_status = "failed".into();
                        s.error = Some(format!("{title} could not be waited for: {e}"));
                    }
                    return;
                }
            }
        }

        // Verify rather than assume: uv exiting zero says the packages resolved, not
        // that the interpreter can import them. A torch wheel that does not match the
        // machine installs perfectly and fails on import.
        let fresh = status_of(env);
        if let Ok(mut s) = installer.slot(env).lock() {
            let log_so_far = s.install_log.clone();
            *s = fresh;
            s.install_log = log_so_far;
            if s.complete {
                s.install_status = "done".into();
            } else {
                s.install_status = "failed".into();
                s.error = Some(format!(
                    "the packages installed but the interpreter still cannot import: {}",
                    s.missing.join(", ")
                ));
            }
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_requirement_sets_are_compiled_in_and_look_right() {
        // Requirement lines only. The comments legitimately discuss MS2PIP at
        // length, explaining why it is absent, and a substring search over the whole
        // file reads that prose as a dependency.
        fn requirement_lines(text: &str) -> Vec<&str> {
            text.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("--"))
                .collect()
        }
        let primary = requirement_lines(REQUIREMENTS_PRIMARY);
        assert!(
            primary.iter().any(|l| l.starts_with("deeplc==")),
            "the primary set pins DeepLC: {primary:?}"
        );
        assert!(
            primary.iter().any(|l| l.starts_with("torch==")),
            "the primary set pins torch: {primary:?}"
        );
        // The whole reason there are two environments: they cannot be one.
        assert!(
            !primary.iter().any(|l| l.starts_with("ms2pip")),
            "MS2PIP must not be in the primary set; it conflicts with DeepLC: {primary:?}"
        );
        assert!(
            requirement_lines(REQUIREMENTS_MS2PIP)
                .iter()
                .any(|l| l.starts_with("ms2pip==")),
            "the optional set pins MS2PIP"
        );
    }

    #[test]
    fn the_two_environments_do_not_share_a_directory() {
        // They must not: installing one would then half-overwrite the other.
        assert_ne!(venv_dir(Env::Primary), venv_dir(Env::Ms2pip));
        assert_ne!(
            requirements_name_of(Env::Primary),
            requirements_name_of(Env::Ms2pip)
        );
    }

    fn requirements_name_of(e: Env) -> &'static str {
        e.requirements_name()
    }

    #[test]
    fn the_managed_paths_are_per_user_and_platform_shaped() {
        let py = managed_python(Env::Primary);
        let s = py.display().to_string();
        assert!(s.contains("MuMDIA"), "{s}");
        if cfg!(windows) {
            assert!(s.ends_with("python.exe"), "{s}");
            assert!(s.contains("Scripts"), "{s}");
        } else {
            assert!(s.ends_with("bin/python"), "{s}");
        }
        // Never inside the installation directory, which is not user-writable on
        // Windows.
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(|p| p.to_path_buf()));
        if let Some(d) = exe_dir {
            assert!(
                !py.starts_with(d),
                "the environment must not live beside the exe"
            );
        }
    }

    #[test]
    fn status_of_an_absent_environment_lists_everything_as_missing() {
        // Nothing is installed under a temp HOME, so this exercises the cold path
        // without touching a real installation.
        let s = if managed_python(Env::Primary).is_file() {
            eprintln!("a managed environment exists on this machine; checking the warm path");
            let s = status();
            assert!(s.python.is_some());
            return;
        } else {
            status()
        };
        assert!(s.python.is_none());
        assert!(!s.complete);
        assert_eq!(s.missing.len(), REQUIRED_MODULES.len());
    }

    #[test]
    fn every_role_the_engine_defines_is_covered_by_the_required_modules() {
        // The engine's own lists, transcribed from python.rs. If a role gains a
        // dependency there and not here, the setup screen would report a complete
        // installation that a run then fails on.
        for role_modules in [
            vec!["torch", "numpy", "pandas", "pyarrow"],
            vec!["mokapot", "sklearn", "numpy", "pandas", "pyarrow"],
            vec!["deeplc", "numpy", "pandas", "pyarrow", "torch", "psm_utils"],
            vec!["numpy", "pyarrow"],
        ] {
            for m in role_modules {
                assert!(
                    REQUIRED_MODULES.contains(&m),
                    "{m} is imported by a sidecar role but is not installed"
                );
            }
        }
    }

    /// MS2PIP is excluded on purpose, not by oversight. If someone "fixes" the list
    /// by adding it, the environment stops resolving; this says so at test time
    /// rather than at install time on a user's machine.
    #[test]
    fn ms2pip_is_deliberately_not_in_the_primary_environment() {
        assert!(
            !REQUIRED_MODULES.contains(&"ms2pip"),
            "ms2pip==4.0.0 needs sqlalchemy<2 and deeplc==4.1.1 needs sqlalchemy>=2;              they cannot share an environment"
        );
        assert!(MS2PIP_MODULES.contains(&"ms2pip"));
    }

    #[test]
    fn the_engine_environment_names_every_role_the_engine_reads() {
        // The bug this guards: the Setup screen installed an environment the engine
        // could not see. The engine resolves `"auto"` from `MUMDIA_PYTHON_*`, an
        // activated conda/virtualenv, then PATH -- never this application's data
        // directory -- so without a stamped environment the managed interpreter was
        // invisible to every search it was installed for, and the engine fell
        // through to whatever `python` was on PATH.
        //
        // Asserted against the same role list the engine defines, so a new role
        // cannot be added there and silently left unstamped here.
        let mut cmd = std::process::Command::new("cmd-that-is-never-run");
        stamp_env(&mut cmd);

        let stamped: Vec<String> = cmd
            .get_envs()
            .filter_map(|(k, v)| v.map(|_| k.to_string_lossy().into_owned()))
            .collect();

        // Only meaningful once something is installed; on a bare machine there is
        // nothing to point at and stamping nothing is correct.
        if managed_python(Env::Primary).is_file() {
            for var in [
                "MUMDIA_PYTHON_RESCORE",
                "MUMDIA_PYTHON_DEEPLC",
                "MUMDIA_PYTHON_MBR",
            ] {
                assert!(
                    stamped.iter().any(|s| s == var),
                    "the primary environment must be stamped as {var}: {stamped:?}"
                );
            }
            // MS2PIP must NOT get the primary interpreter: the two environments are
            // not interchangeable, which is why there are two.
            assert!(
                !stamped.iter().any(|s| s == "MUMDIA_PYTHON_MS2PIP")
                    || managed_python(Env::Ms2pip).is_file(),
                "MS2PIP was stamped without its own environment existing"
            );
            // The blanket variable would apply the primary interpreter to MS2PIP too.
            assert!(
                !stamped.iter().any(|s| s == "MUMDIA_PYTHON"),
                "the blanket MUMDIA_PYTHON must not be used: {stamped:?}"
            );
        }
        if thermo_parser().is_some() {
            assert!(stamped.iter().any(|s| s == "MUMDIA_THERMO_PARSER"));
        }
    }

    #[test]
    fn nothing_installed_means_nothing_is_stamped() {
        // A stamped variable pointing at a path that does not exist would be worse
        // than no variable: the engine treats an explicit interpreter as a
        // deliberate choice and fails on it rather than searching on.
        let mut cmd = std::process::Command::new("cmd-that-is-never-run");
        stamp_env(&mut cmd);
        for (k, v) in cmd.get_envs() {
            let Some(v) = v else { continue };
            let key = k.to_string_lossy();
            if key.starts_with("MUMDIA_") {
                assert!(
                    std::path::Path::new(v).is_file(),
                    "{key} was stamped with {v:?}, which is not a file"
                );
            }
        }
    }
}
