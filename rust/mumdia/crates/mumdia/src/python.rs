//! Sidecar interpreter resolution: turn "which Python should run this worker?"
//! into one answer, computed once per run and recorded in the manifest.
//!
//! Every ML predictor and rescorer in MuMDIA is a Python worker the engine launches
//! by path (docs/13_sidecars.md). Those paths used to be the only answer available:
//! a config named `.../envs/whatever/bin/python` outright, with no lookup and no
//! discovery, so a config was bound to the machine that wrote it. The only tracked
//! example config carried one developer's `C:/Users/...` and OneDrive paths, and a
//! collaborator's first act was always to edit it.
//!
//! Two things change that. A field may be absent or set to `"auto"`, in which case
//! [`resolve`] discovers an interpreter; and discovery VALIDATES a candidate by
//! importing the worker's own dependency list before accepting it, so it cannot
//! silently pick a Python without torch and defer the failure to hour three of a
//! run. Explicit paths keep working unchanged and are never second-guessed.
//!
//! Resolution happens before the config hash is taken, so `manifest.json` records
//! the interpreter that actually ran rather than the word `auto`. That makes the
//! hash machine-specific for an `auto` config, which is correct: two runs whose
//! rescorer came from different environments are not the same configuration.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Result};
use mumdia_core::config::{
    Config, FragPredictorKind, ImPredictorKind, LibraryIrt, RescorerKind, RtPredictorKind,
};
use tracing::{info, warn};

/// The literal a config uses to ask for discovery rather than naming a path.
pub const AUTO: &str = "auto";

/// One sidecar interpreter slot. Each maps to a config field, a set of workers,
/// and the modules those workers import.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    /// `rescore.python`: `mokapot_worker.py`, `nn_rescore_worker.py`,
    /// `entrapment_worker.py`.
    Rescore,
    /// `predict_frag.deeplc_python`: `deeplc_worker.py`, `deeplc_finetune.py`.
    DeepLc,
    /// `predict_frag.ms2pip_python`: `ms2pip_worker.py`.
    Ms2pip,
    /// `predict_frag.peptdeep_python`: `peptdeep_worker.py`.
    Peptdeep,
    /// `mbr.python`: `mbr_worker.py`.
    Mbr,
    /// `predict_frag.im2deep_python`: `im2deep_worker.py`.
    Im2deep,
}

impl Role {
    pub fn field(self) -> &'static str {
        match self {
            Role::Rescore => "rescore.python",
            Role::DeepLc => "predict_frag.deeplc_python",
            Role::Ms2pip => "predict_frag.ms2pip_python",
            Role::Peptdeep => "predict_frag.peptdeep_python",
            Role::Mbr => "mbr.python",
            Role::Im2deep => "predict_frag.im2deep_python",
        }
    }

    /// The distribution whose version changes this role's results, and the oldest release
    /// accepted. Discovery skips an interpreter below it, `doctor` fails on it and the
    /// sidecar launch refuses it (`sidecar::require_version`).
    pub fn version_floor(self) -> Option<(&'static str, (u32, u32, u32))> {
        match self {
            Role::DeepLc => Some(("deeplc", mumdia_core::constants::MIN_DEEPLC_VERSION)),
            Role::Im2deep => Some(("im2deep", mumdia_core::constants::MIN_IM2DEEP_VERSION)),
            Role::Rescore | Role::Ms2pip | Role::Peptdeep | Role::Mbr => None,
        }
    }

    /// Environment variable that overrides discovery for this role alone.
    pub fn env_var(self) -> &'static str {
        match self {
            Role::Rescore => "MUMDIA_PYTHON_RESCORE",
            Role::DeepLc => "MUMDIA_PYTHON_DEEPLC",
            Role::Ms2pip => "MUMDIA_PYTHON_MS2PIP",
            Role::Peptdeep => "MUMDIA_PYTHON_PEPTDEEP",
            Role::Mbr => "MUMDIA_PYTHON_MBR",
            Role::Im2deep => "MUMDIA_PYTHON_IM2DEEP",
        }
    }

    /// Modules the role's workers import. Asserting what the scripts actually
    /// import, not what their dependency trees imply: probing only `deeplc` once
    /// let a green `doctor` precede a fine-tune crash, which on a batch is
    /// discovered long after the run was launched.
    pub fn modules(self, cfg: &Config) -> &'static [&'static str] {
        match self {
            // `nn_rescore_worker.py` needs torch; mokapot and the entrapment GBM
            // need mokapot/sklearn instead.
            Role::Rescore => match cfg.rescore.classifier {
                RescorerKind::NnTorch => &["torch", "numpy", "pandas", "pyarrow"],
                _ => &["mokapot", "sklearn", "numpy", "pandas", "pyarrow"],
            },
            // `deeplc_finetune.py` imports pyarrow, torch and psm_utils on top of
            // deeplc itself.
            Role::DeepLc => &["deeplc", "numpy", "pandas", "pyarrow", "torch", "psm_utils"],
            Role::Ms2pip => &["ms2pip", "numpy", "pandas"],
            // `peptdeep_worker.py` imports torch directly to decide the device, and
            // alphabase to check a modification name against the installed vocabulary,
            // so both are probed rather than assumed from peptdeep's dependency tree.
            Role::Peptdeep => &[
                "peptdeep",
                "alphabase",
                "torch",
                "numpy",
                "pandas",
                "pyarrow",
            ],
            Role::Mbr => &["numpy", "pyarrow"],
            // `im2deep_worker.py` builds a `psm_utils.PSMList` and sizes torch's thread pool.
            Role::Im2deep => &["im2deep", "psm_utils", "torch", "numpy", "pyarrow"],
        }
    }

    /// Worker scripts this role runs, for the file-presence check in `doctor`.
    pub fn workers(self) -> &'static [&'static str] {
        match self {
            Role::Rescore => &[
                "mokapot_worker.py",
                "nn_rescore_worker.py",
                "entrapment_worker.py",
            ],
            Role::DeepLc => &["deeplc_worker.py", "deeplc_finetune.py"],
            Role::Ms2pip => &["ms2pip_worker.py"],
            Role::Peptdeep => &["peptdeep_worker.py"],
            Role::Mbr => &["mbr_worker.py"],
            Role::Im2deep => &["im2deep_worker.py"],
        }
    }

    fn get(self, cfg: &Config) -> Option<&str> {
        match self {
            Role::Rescore => cfg.rescore.python.as_deref(),
            Role::DeepLc => cfg.predict_frag.deeplc_python.as_deref(),
            Role::Ms2pip => cfg.predict_frag.ms2pip_python.as_deref(),
            Role::Peptdeep => cfg.predict_frag.peptdeep_python.as_deref(),
            Role::Mbr => cfg.mbr.python.as_deref(),
            Role::Im2deep => cfg.predict_frag.im2deep_python.as_deref(),
        }
    }

    fn set(self, cfg: &mut Config, value: String) {
        match self {
            Role::Rescore => cfg.rescore.python = Some(value),
            Role::DeepLc => cfg.predict_frag.deeplc_python = Some(value),
            Role::Ms2pip => cfg.predict_frag.ms2pip_python = Some(value),
            Role::Peptdeep => cfg.predict_frag.peptdeep_python = Some(value),
            Role::Mbr => cfg.mbr.python = Some(value),
            Role::Im2deep => cfg.predict_frag.im2deep_python = Some(value),
        }
    }

    /// Drop the field entirely. Used for a role that asked for discovery but is not
    /// used by this configuration: leaving the literal "auto" in place made the
    /// resolved config self-inconsistent, and every later `Path::exists` check on it
    /// then failed on a program named "auto" for a sidecar that never runs. That is
    /// how `run-experiment` came to reject a config `run` accepted. None also records
    /// the truth in the manifest: no interpreter was selected, because none was needed.
    fn clear(self, cfg: &mut Config) {
        match self {
            Role::Rescore => cfg.rescore.python = None,
            Role::DeepLc => cfg.predict_frag.deeplc_python = None,
            Role::Ms2pip => cfg.predict_frag.ms2pip_python = None,
            Role::Peptdeep => cfg.predict_frag.peptdeep_python = None,
            Role::Mbr => cfg.mbr.python = None,
            Role::Im2deep => cfg.predict_frag.im2deep_python = None,
        }
    }

    /// Whether this run needs the role at all. A role that is not needed is never
    /// discovered and never probed, so a FASTA-free native run still works on a
    /// machine with no Python.
    pub fn required_by(self, cfg: &Config) -> bool {
        match self {
            Role::Rescore => matches!(
                cfg.rescore.classifier,
                RescorerKind::Mokapot | RescorerKind::NnTorch | RescorerKind::Entrapment
            ),
            Role::DeepLc => {
                cfg.predict_frag.rt_predictor == RtPredictorKind::Deeplc
                    || cfg.rt_im_train.finetune_deeplc
                    || cfg.rt_im_train.library_irt == LibraryIrt::Deeplc
                    // An EXPLICIT head count. The automatic default is `wanted_by`
                    // instead, so a run with no interpreter degrades rather than failing.
                    || cfg.rt_im_train.multihead_calibration.is_some_and(|n| n > 0)
            }
            Role::Ms2pip => cfg.predict_frag.predictor == FragPredictorKind::Ms2pip,
            Role::Peptdeep => cfg.predict_frag.predictor == FragPredictorKind::Peptdeep,
            Role::Mbr => cfg.mbr.strategy.transfers(),
            Role::Im2deep => cfg.predict_frag.im_predictor == ImPredictorKind::Im2deep,
        }
    }

    /// Whether the configuration would USE the role if an interpreter is available,
    /// without requiring one. IM2Deep has this state under `predict_frag.im_predictor =
    /// auto`, which predicts the library's 1/K0 when an interpreter resolves and builds a
    /// library without it otherwise. DeepLC has it by default: under the default
    /// `rt_im_train.library_irt = auto` a library-input run re-predicts the imported iRT
    /// with the DeepLC base model when `predict_frag.deeplc_python` resolves and keeps the
    /// imported values, with a warning, when it does not. Discovery therefore runs for a
    /// wanted role, and a miss is a note rather than an error. `library_irt = deeplc`
    /// turns the same role into a required one (`required_by`), and a FASTA-mode run,
    /// which ignores `library_irt`, simply never launches the worker.
    pub fn wanted_by(self, cfg: &Config) -> bool {
        match self {
            Role::DeepLc => {
                // The automatic multi-head default needs no clause of its own. Its
                // scope IS a DeepLC retention-time source: in imported-library mode that
                // is `library_irt == Auto`, which already wants an interpreter, and in
                // FASTA mode it is `rt_predictor == Deeplc`, which already requires one.
                // An earlier version added a clause here and made a fully native
                // configuration want DeepLC, which three tests caught.
                !self.required_by(cfg) && cfg.rt_im_train.library_irt == LibraryIrt::Auto
            }
            Role::Im2deep => cfg.predict_frag.im_predictor == ImPredictorKind::Auto,
            _ => false,
        }
    }
}

pub const ALL_ROLES: [Role; 6] = [
    Role::Rescore,
    Role::DeepLc,
    Role::Ms2pip,
    Role::Peptdeep,
    Role::Mbr,
    Role::Im2deep,
];

/// How a role's interpreter was determined, for logging and for `doctor`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolution {
    pub role: Role,
    pub python: Option<String>,
    /// Human-readable provenance: `configured`, the environment variable name,
    /// `CONDA_PREFIX`, `VIRTUAL_ENV`, `PATH`, or `not required`.
    pub source: &'static str,
}

/// Candidate interpreters, in the order they are tried. Each carries the
/// provenance string reported for it.
fn candidates(role: Role) -> Vec<(PathBuf, &'static str)> {
    let mut out: Vec<(PathBuf, &'static str)> = Vec::new();
    let mut push_env = |var: &'static str| {
        if let Ok(v) = std::env::var(var) {
            if !v.trim().is_empty() {
                out.push((PathBuf::from(v), var));
            }
        }
    };
    // Most specific first: this role, then all roles.
    push_env(role.env_var());
    push_env("MUMDIA_PYTHON");

    // An activated environment is the strongest implicit signal: the user chose it
    // by activating it. Windows puts the interpreter at the prefix root, POSIX in
    // bin/.
    for (var, tag) in [
        ("CONDA_PREFIX", "CONDA_PREFIX"),
        ("VIRTUAL_ENV", "VIRTUAL_ENV"),
    ] {
        if let Ok(prefix) = std::env::var(var) {
            if !prefix.trim().is_empty() {
                let p = Path::new(&prefix);
                out.push((p.join("bin").join("python"), tag));
                out.push((p.join("python.exe"), tag));
                out.push((p.join("Scripts").join("python.exe"), tag));
            }
        }
    }

    // Finally the plain interpreters on PATH. `Command` resolves a bare name
    // against PATH itself, so no manual search is needed.
    out.push((PathBuf::from("python3"), "PATH"));
    out.push((PathBuf::from("python"), "PATH"));
    out
}

/// Ask an interpreter which of `modules` it cannot import. `Ok(vec![])` means all
/// of them are importable.
///
/// `find_spec` rather than a real import: importing torch or deeplc costs seconds,
/// and discovery runs this for every candidate interpreter at the start of every run.
/// The trade is that a package which is present but cannot load (a torch whose DLLs
/// fail to initialise, for instance) still passes here. [`import_probe`] is the real
/// import that catches it, and `doctor` runs it by default.
pub fn missing_modules(python: &Path, modules: &[&str]) -> Result<Vec<String>> {
    let out = Command::new(python)
        .args(["-c", &missing_modules_script(modules)])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            Ok(if s.is_empty() {
                Vec::new()
            } else {
                s.split(',').map(str::to_string).collect()
            })
        }
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr).trim().to_string();
            bail!(
                "{} exited {}: {}",
                python.display(),
                o.status.code().unwrap_or(-1),
                if err.is_empty() { "no output" } else { &err }
            )
        }
        Err(e) => bail!("cannot run {}: {e}", python.display()),
    }
}

/// The script `missing_modules` runs: `find_spec` over `modules`, printing the absent
/// ones comma-separated.
fn missing_modules_script(modules: &[&str]) -> String {
    format!(
        "import importlib.util as u; \
         print(','.join(p for p in '{}'.split(',') if u.find_spec(p) is None))",
        modules.join(",")
    )
}

/// How long `doctor`'s real import of one role's modules may take before it is reported
/// as hung. A cold `import deeplc` (which loads torch) takes seconds; the first one after
/// an installation on Windows can take tens of seconds while the new files are scanned.
pub const IMPORT_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// The bound on the import probe that follows a failed worker (`sidecar`): shorter,
/// because it only adds a diagnosis to an error the run already has.
pub const FAILURE_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// What to do when PyTorch cannot load its DLLs on Windows. Repeated in the desktop
/// application (`desktop/src-tauri/src/loadcheck.rs`), which cannot depend on this crate;
/// the two are kept word for word.
pub const VC_RUNTIME_HINT: &str = "PyTorch cannot load its DLLs on this machine. This is \
     usually a missing or outdated Microsoft Visual C++ 2015-2022 x64 runtime: install the \
     latest one from https://aka.ms/vs/17/release/vc_redist.x64.exe, reboot, then check \
     again. If it is already installed, an older msvcp140.dll or vcruntime140.dll earlier \
     on PATH can be loaded instead of it: `where msvcp140.dll` lists every copy Windows \
     finds, in the order it tries them. See docs/17_troubleshooting.md.";

/// The fix for an import failure, when its output names one this module recognises.
///
/// Recognised: PyTorch's Windows DLL failures, `[WinError 1114]` (a DLL initialization
/// routine failed) or `[WinError 126]` (a module could not be found) while loading a file
/// under `torch\lib`, and the `DLL load failed while importing` form of the same, all of
/// which a missing or outdated Visual C++ runtime produces. The text is matched without
/// regard to case or path separators, so the same function reads a probe's stderr and a
/// worker's traceback.
pub fn load_failure_hint(text: &str) -> Option<&'static str> {
    let t = text.to_ascii_lowercase();
    let names_torch = t.contains("c10.dll") || t.contains("torch");
    let dll_error = t.contains("winerror 1114")
        || t.contains("winerror 126")
        || t.contains("dll load failed")
        || t.contains("dll initialization routine failed");
    (names_torch && dll_error).then_some(VC_RUNTIME_HINT)
}

/// What a real import of a role's modules found (`import_probe`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportProbe {
    /// Every installed module imported, and torch, when it is one of them, ran an
    /// operation.
    Loaded,
    /// `module` is installed but raising, or crashing the interpreter, on import.
    /// `detail` is the tail of the interpreter's stderr, traceback included.
    Unloadable { module: String, detail: String },
    /// The interpreter had not finished importing after `seconds`.
    TimedOut { seconds: u64 },
}

/// The script `import_probe` runs: a real import of each of `modules` in order, skipping
/// the ones `find_spec` cannot locate (absence is `missing_modules`'s report, not this
/// one's), and one torch operation once torch is loaded.
///
/// `deeplc` is moved to the front when present, because that is the order every DeepLC
/// worker imports in (docs/13_sidecars.md) and a probe that imported numpy first could
/// fail where the workers do not. Each import is announced before it is attempted, so a
/// module that kills the interpreter outright (an access violation in a native extension)
/// is still named.
pub fn import_probe_script(modules: &[&str]) -> String {
    let mut order: Vec<&str> = modules.iter().copied().filter(|m| *m == "deeplc").collect();
    order.extend(modules.iter().copied().filter(|m| *m != "deeplc"));
    let list = order
        .iter()
        .map(|m| format!("'{m}'"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "import importlib.util, sys, traceback\n\
         for m in [{list}]:\n\
         \x20   if importlib.util.find_spec(m) is None:\n\
         \x20       continue\n\
         \x20   print('MUMDIA_TRY ' + m, flush=True)\n\
         \x20   try:\n\
         \x20       __import__(m)\n\
         \x20       if m == 'torch':\n\
         \x20           sys.modules['torch'].zeros(1)\n\
         \x20   except BaseException:\n\
         \x20       traceback.print_exc()\n\
         \x20       sys.stderr.flush()\n\
         \x20       print('MUMDIA_UNLOADABLE ' + m, flush=True)\n\
         \x20       sys.exit(3)\n\
         print('MUMDIA_IMPORT_OK', flush=True)\n"
    )
}

/// Read an import probe's output. Separate from the process handling so the rules can
/// be tested on fixed text.
pub fn judge_import_probe(success: bool, stdout: &str, stderr: &str) -> ImportProbe {
    let mut tried = None;
    for line in stdout.lines() {
        let line = line.trim();
        if line == "MUMDIA_IMPORT_OK" && success {
            return ImportProbe::Loaded;
        }
        if let Some(m) = line.strip_prefix("MUMDIA_UNLOADABLE ") {
            return ImportProbe::Unloadable {
                module: m.to_string(),
                detail: tail_lines(stderr, 15),
            };
        }
        if let Some(m) = line.strip_prefix("MUMDIA_TRY ") {
            tried = Some(m.to_string());
        }
    }
    // No verdict line: the interpreter died, most likely inside the last import it
    // announced.
    ImportProbe::Unloadable {
        module: tried.unwrap_or_else(|| "python".to_string()),
        detail: tail_lines(stderr, 15),
    }
}

/// The last `n` non-empty lines of `s`.
pub fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// Run `cmd` to completion with both streams captured, or kill it after `timeout`.
/// `Ok(None)` is a timeout.
pub fn output_with_timeout(
    mut cmd: Command,
    timeout: std::time::Duration,
) -> std::io::Result<Option<std::process::Output>> {
    use std::io::Read;
    use std::process::Stdio;
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    // Both pipes are drained on their own threads: a traceback larger than the pipe
    // buffer would otherwise block the child and turn a failure into a timeout.
    fn drain<R: Read + Send + 'static>(r: Option<R>) -> std::thread::JoinHandle<Vec<u8>> {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut r) = r {
                let _ = r.read_to_end(&mut buf);
            }
            buf
        })
    }
    let out = drain(child.stdout.take());
    let err = drain(child.stderr.take());
    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        if let Some(st) = child.try_wait()? {
            break st;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(None);
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    Ok(Some(std::process::Output {
        status,
        stdout: out.join().unwrap_or_default(),
        stderr: err.join().unwrap_or_default(),
    }))
}

/// The command `import_probe` runs. UTF-8 output, because a traceback quotes paths, and a
/// Windows user's profile path need not be representable in the console code page.
fn import_probe_command(python: &Path, modules: &[&str]) -> Command {
    let mut cmd = Command::new(python);
    cmd.args(["-c", &import_probe_script(modules)])
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8");
    cmd
}

/// Import `modules` for real in `python`, which `missing_modules` deliberately does not.
///
/// `find_spec` only locates a package; it never loads a native library. A torch whose
/// DLLs cannot initialise on the machine, typically for want of the Visual C++ runtime,
/// therefore passes `missing_modules` and then kills the first worker that imports it,
/// minutes into a search. This is the check that catches it: `doctor` runs it by default,
/// and a failed worker runs it to explain itself (`sidecar`). `Err` means the
/// interpreter could not be started at all.
pub fn import_probe(
    python: &Path,
    modules: &[&str],
    timeout: std::time::Duration,
) -> Result<ImportProbe> {
    match output_with_timeout(import_probe_command(python, modules), timeout) {
        Ok(Some(o)) => Ok(judge_import_probe(
            o.status.success(),
            &String::from_utf8_lossy(&o.stdout),
            &String::from_utf8_lossy(&o.stderr),
        )),
        Ok(None) => Ok(ImportProbe::TimedOut {
            seconds: timeout.as_secs(),
        }),
        Err(e) => bail!("cannot run {}: {e}", python.display()),
    }
}

/// What a version probe says about a floor (`Role::version_floor`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FloorCheck {
    /// The reported version, at or above the floor.
    Met(String),
    /// The reported version, below the floor or not a parseable release.
    TooOld(String),
    /// The interpreter reported no version.
    Unknown,
}

/// Judge a probed version against a floor. Discovery, `doctor` and the sidecar launch
/// (`sidecar::require_version`) all decide through this, so they cannot disagree.
pub fn judge_version(version: Option<String>, floor: (u32, u32, u32)) -> FloorCheck {
    match version {
        Some(v) if mumdia_core::constants::parse_version3(&v).is_some_and(|t| t >= floor) => {
            FloorCheck::Met(v)
        }
        Some(v) => FloorCheck::TooOld(v),
        None => FloorCheck::Unknown,
    }
}

/// `major.minor.patch` of a floor, for messages.
pub fn fmt_version((ma, mi, pa): (u32, u32, u32)) -> String {
    format!("{ma}.{mi}.{pa}")
}

/// Report a package's version, or `None` when the interpreter cannot tell us.
/// Used by `doctor` to show what a user has, since a present-but-old DeepLC
/// changes results rather than only failing.
pub fn module_version(python: &Path, module: &str) -> Option<String> {
    let code = format!(
        "import importlib.metadata as m\n\
         try: print(m.version('{module}'))\n\
         except Exception: print('')"
    );
    let out = Command::new(python).args(["-c", &code]).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Find an interpreter for `role` that can import everything the role's workers
/// need. Returns the first candidate that passes, with its provenance.
pub fn discover(role: Role, cfg: &Config) -> Option<(String, &'static str)> {
    let modules = role.modules(cfg);
    let mut tried: Vec<String> = Vec::new();
    for (path, source) in candidates(role) {
        match missing_modules(&path, modules) {
            Ok(missing) if missing.is_empty() => {
                // Presence is not enough for DeepLC: the default RT workflow calibrates
                // base-model predictions without a fine-tune, which is only sound from
                // `MIN_DEEPLC_VERSION` on, and the sidecar launch refuses anything older.
                // Rejecting it here keeps discovery from handing back an interpreter the
                // run cannot use.
                if let Some((module, floor)) = role.version_floor() {
                    let want = fmt_version(floor);
                    match judge_version(module_version(&path, module), floor) {
                        FloorCheck::Met(_) => {}
                        FloorCheck::TooOld(v) => {
                            tried.push(format!(
                                "{} ({module} {v} is older than the required {want})",
                                path.display()
                            ));
                            continue;
                        }
                        FloorCheck::Unknown => {
                            tried.push(format!(
                                "{} ({module} version unknown; {want} or newer is required)",
                                path.display()
                            ));
                            continue;
                        }
                    }
                }
                return Some((path.to_string_lossy().into_owned(), source));
            }
            Ok(missing) => tried.push(format!(
                "{} (missing {})",
                path.display(),
                missing.join(",")
            )),
            // A candidate that cannot even be executed is not interesting to
            // report unless nothing works, which the caller handles.
            Err(_) => {}
        }
    }
    if !tried.is_empty() {
        info!(
            role = ?role,
            rejected = tried.join("; "),
            "python discovery: candidates found but incomplete"
        );
    }
    None
}

/// How hard to insist that a role the CONFIGURATION needs actually resolves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Strictness {
    /// A role the configuration uses but which cannot be resolved is a hard error.
    ///
    /// For the orchestrators, where preflight exists so a misconfiguration costs
    /// seconds rather than the hours already spent by the time rescore is reached.
    Require,
    /// Resolve what can be resolved; leave the rest as `None` and carry on.
    ///
    /// For a single stage, where `Role::required_by` answers the wrong question. It says
    /// "does this CONFIGURATION use the role", not "does the stage I am about to run use
    /// it" -- so under `Require` a `mumdia search-seed` with
    /// `rescore.classifier = entrapment` failed at config load demanding a mokapot
    /// interpreter, for a rescorer that stage never invokes. Found by running a
    /// benchmark, not by a test.
    ///
    /// An unresolvable role becomes `None` rather than staying as the literal `"auto"`,
    /// so a stage that DOES need it still fails with "this classifier requires
    /// rescore.python" instead of trying to execute a program called `auto`, which is
    /// the confusion this whole mechanism exists to remove.
    BestEffort,
}

/// Resolve every interpreter this run needs, mutating `cfg` so downstream stages
/// see concrete paths. Absent or `"auto"` triggers discovery; an explicit path is
/// taken as given. Returns one [`Resolution`] per role for logging.
pub fn resolve(cfg: &mut Config) -> Result<Vec<Resolution>> {
    resolve_with(cfg, Strictness::Require)
}

pub fn resolve_with(cfg: &mut Config, strictness: Strictness) -> Result<Vec<Resolution>> {
    let mut out = Vec::new();
    for role in ALL_ROLES {
        let required = role.required_by(cfg);
        let configured = role.get(cfg).map(str::to_string);
        let wants_discovery = match configured.as_deref() {
            None => true,
            Some(v) => v.eq_ignore_ascii_case(AUTO),
        };

        if !wants_discovery {
            let path = configured.expect("checked above");
            // An explicit path is honored as-is. Existence is checked here for a
            // clear message; whether it can import the modules is `doctor`'s
            // question, because a user may knowingly point at an environment they
            // are about to fix.
            if required && !Path::new(&path).exists() {
                let msg = format!(
                    "{} points at {}, which does not exist. Fix the path, set it to \
                     \"auto\" to discover an interpreter, or set {}.",
                    role.field(),
                    path,
                    role.env_var()
                );
                if strictness == Strictness::Require {
                    bail!(msg);
                }
                // The stage being run may not touch this role at all; let the one that
                // does report it.
                warn!(field = role.field(), "python: {msg}");
            }
            out.push(Resolution {
                role,
                python: Some(path),
                source: "configured",
            });
            continue;
        }

        let wanted = role.wanted_by(cfg);
        if !required && !wanted {
            // Nothing to probe: the native path is in use. Clear the field so the
            // literal "auto" does not survive into the resolved config, where a later
            // existence check would try to stat a program called "auto".
            role.clear(cfg);
            out.push(Resolution {
                role,
                python: None,
                source: "not required",
            });
            continue;
        }

        match discover(role, cfg) {
            Some((path, source)) => {
                info!(
                    field = role.field(),
                    python = %path,
                    source,
                    "python: resolved sidecar interpreter"
                );
                role.set(cfg, path.clone());
                out.push(Resolution {
                    role,
                    python: Some(path),
                    source,
                });
            }
            None if !required => {
                // Wanted, not required: the run goes on without it. `run` says what that
                // means for the retention times when it gets there.
                role.clear(cfg);
                if role == Role::Im2deep {
                    info!(
                        field = role.field(),
                        "python: no IM2Deep interpreter found, so predict_frag.im_predictor = \
                         auto resolves to none and a library built from the FASTA carries no \
                         predicted 1/K0 (set {} or name one)",
                        role.env_var()
                    );
                } else {
                    info!(
                        field = role.field(),
                        "python: no interpreter found for an optional role; a library-input \
                         run will keep the imported iRT and skip the multi-head RT calibration, \
                         which is worth 4.8% of peptides on AIF and 14.3% on Astral (set {} or \
                         name one)",
                        role.env_var()
                    );
                }
                out.push(Resolution {
                    role,
                    python: None,
                    source: "not found (optional)",
                });
            }
            None => {
                let msg = format!(
                    "{} is required by this configuration but no usable interpreter was \
                     found. Tried {}, CONDA_PREFIX, VIRTUAL_ENV, and python3/python on \
                     PATH; each must import {}. Install one (see env/ for conda specs), \
                     then either activate it, set {}, or name it in the config.",
                    role.field(),
                    role.env_var(),
                    role.modules(cfg).join(", "),
                    role.env_var()
                );
                if strictness == Strictness::Require {
                    bail!(msg);
                }
                // Clear rather than leave `"auto"` behind, so a stage that does need this
                // role fails on the missing interpreter and not on a program named `auto`.
                role.clear(cfg);
                warn!(field = role.field(), "python: {msg}");
                out.push(Resolution {
                    role,
                    python: None,
                    source: "unresolved",
                });
            }
        }
    }

    // A configured-but-unused interpreter is a silent no-op today; say so, because
    // it usually means the classifier or predictor is not the one the user thinks.
    for r in &out {
        if r.source == "configured" && !r.role.required_by(cfg) && !r.role.wanted_by(cfg) {
            warn!(
                field = r.role.field(),
                "python: interpreter configured but this run does not use it"
            );
        }
    }
    Ok(out)
}

/// Resolve the directory holding the worker scripts.
///
/// The configured value is relative to the CWD, which silently changed which
/// scripts ran depending on where the command was invoked. Preference order:
/// the configured directory if it exists, then the same path relative to the
/// config file's own directory, then `scripts/` next to the executable (the
/// release-archive layout), then the configured value unchanged so the eventual
/// error names what was asked for.
pub fn resolve_script_dir(configured: &str, config_path: Option<&str>) -> String {
    let has_workers =
        |dir: &Path| dir.join("mbr_worker.py").exists() || dir.join("deeplc_worker.py").exists();

    // An ABSOLUTE configured path is taken as given: the user named a directory, and
    // naming one is the way to be unambiguous.
    let direct = Path::new(configured);
    if direct.is_absolute() && has_workers(direct) {
        return configured.to_string();
    }

    // A RELATIVE path resolves against the config file first, then the executable, and
    // only then against the current working directory.
    //
    // The old order tried the working directory first, and the shipped default is the
    // relative `"scripts"`, which both sidecar example configs carry literally. So:
    // unpack a dataset archive, `cd` into it, run
    // `mumdia run --config configs/examples/diann-library.json`, and if that archive
    // happens to contain a `scripts/` directory with a worker file in it, the whole
    // directory won resolution and every worker in it ran as the user. That needs no
    // hostile CONFIGURATION -- which the security policy treats as trusted, like a shell
    // script -- only an untrusted input directory, which is an ordinary Tuesday in this
    // field.
    //
    // Config-relative first is also the more useful order: it makes a config portable
    // with its scripts, which is what the resolution exists for.
    if let Some(cfg_path) = config_path {
        if let Some(base) = Path::new(cfg_path).parent() {
            let rel = base.join(configured);
            if has_workers(&rel) {
                return rel.to_string_lossy().into_owned();
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(base) = exe.parent() {
            for cand in [base.join(configured), base.join("scripts")] {
                if has_workers(&cand) {
                    return cand.to_string_lossy().into_owned();
                }
            }
        }
    }
    // Working directory last. Still reached, because running from a checkout with a
    // relative `scripts` is the normal development case, but now only when neither the
    // config's directory nor the executable's supplies the workers -- so a planted
    // directory can no longer displace the ones that ship with the configuration or the
    // binary.
    if has_workers(direct) {
        tracing::warn!(
            script_dir = configured,
            "python: resolved the sidecar script directory against the current working \
             directory, because neither the config file's directory nor the executable's \
             contains the workers. Prefer an absolute sidecar_script_dir, or keep the \
             scripts beside the config"
        );
        return configured.to_string();
    }
    configured.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_version_floor_is_judged_the_same_way_everywhere() {
        let floor = (4, 5, 0);
        assert_eq!(
            judge_version(Some("4.5.1".into()), floor),
            FloorCheck::Met("4.5.1".into())
        );
        assert_eq!(
            judge_version(Some("4.5.0".into()), floor),
            FloorCheck::Met("4.5.0".into())
        );
        assert_eq!(
            judge_version(Some("4.4.0".into()), floor),
            FloorCheck::TooOld("4.4.0".into())
        );
        assert_eq!(
            judge_version(Some("4.0.0a2".into()), floor),
            FloorCheck::TooOld("4.0.0a2".into())
        );
        assert_eq!(judge_version(None, floor), FloorCheck::Unknown);
        assert_eq!(fmt_version(floor), "4.5.0");
        // DeepLC and IM2Deep have a floor; every other role reports none.
        for role in ALL_ROLES {
            match (role, role.version_floor()) {
                (Role::DeepLc, f) => assert_eq!(
                    f,
                    Some(("deeplc", mumdia_core::constants::MIN_DEEPLC_VERSION))
                ),
                (Role::Im2deep, f) => assert_eq!(
                    f,
                    Some(("im2deep", mumdia_core::constants::MIN_IM2DEEP_VERSION))
                ),
                (_, f) => assert_eq!(f, None, "{role:?}"),
            }
        }
        assert_eq!(mumdia_core::constants::MIN_IM2DEEP_VERSION, (2, 0, 0));
        assert_eq!(
            judge_version(Some("1.1.0".into()), (2, 0, 0)),
            FloorCheck::TooOld("1.1.0".into())
        );
    }

    #[test]
    fn the_im2deep_role_follows_the_im_predictor() {
        let mut cfg = Config::default();
        assert!(!Role::Im2deep.required_by(&cfg) && !Role::Im2deep.wanted_by(&cfg));
        assert_eq!(Role::Im2deep.field(), "predict_frag.im2deep_python");
        assert_eq!(Role::Im2deep.env_var(), "MUMDIA_PYTHON_IM2DEEP");
        assert_eq!(Role::Im2deep.workers(), &["im2deep_worker.py"]);
        assert!(Role::Im2deep.modules(&cfg).contains(&"im2deep"));
        // An explicit `im2deep` is a hard requirement; `auto` only wants an interpreter.
        cfg.predict_frag.im_predictor = ImPredictorKind::Im2deep;
        assert!(Role::Im2deep.required_by(&cfg) && !Role::Im2deep.wanted_by(&cfg));
        cfg.predict_frag.im_predictor = ImPredictorKind::Auto;
        assert!(!Role::Im2deep.required_by(&cfg) && Role::Im2deep.wanted_by(&cfg));
        // Under `auto` a miss resolves to no prediction rather than an error, and the field
        // never keeps the literal "auto".
        cfg.rt_im_train.library_irt = LibraryIrt::Library;
        cfg.predict_frag.im2deep_python = Some(AUTO.into());
        std::env::set_var("MUMDIA_PYTHON_IM2DEEP", "/definitely/not/an/interpreter");
        let res = resolve(&mut cfg);
        std::env::remove_var("MUMDIA_PYTHON_IM2DEEP");
        let res = res.expect("an optional role never fails resolution");
        let r = res.iter().find(|r| r.role == Role::Im2deep).unwrap();
        assert_eq!(cfg.predict_frag.im2deep_python, r.python);
        assert!(cfg
            .predict_frag
            .im2deep_python
            .as_deref()
            .is_none_or(|p| !p.eq_ignore_ascii_case(AUTO)));
        let resolved = cfg.predict_frag.im_predictor_for(true);
        assert_eq!(
            resolved == ImPredictorKind::Im2deep,
            cfg.predict_frag.im2deep_python.is_some()
        );
    }

    #[test]
    fn roles_report_their_config_field_and_env_var() {
        for role in ALL_ROLES {
            assert!(role.field().contains("python"), "{:?}", role);
            assert!(role.env_var().starts_with("MUMDIA_PYTHON"), "{:?}", role);
            assert!(!role.workers().is_empty());
        }
    }

    #[test]
    fn required_by_follows_the_configuration() {
        // Defaults are the fully native path: nothing needs Python, so a machine
        // without an interpreter still runs.
        let mut cfg = Config::default();
        for role in ALL_ROLES {
            assert!(!role.required_by(&cfg), "{:?} required by default", role);
        }

        cfg.rescore.classifier = RescorerKind::NnTorch;
        assert!(Role::Rescore.required_by(&cfg));
        assert_eq!(
            Role::Rescore.modules(&cfg),
            &["torch", "numpy", "pandas", "pyarrow"]
        );
        cfg.rescore.classifier = RescorerKind::Mokapot;
        assert!(Role::Rescore.modules(&cfg).contains(&"mokapot"));

        // DeepLC is WANTED, not required, by default: `library_irt = auto` uses it for a
        // library-input run when it resolves and does without it otherwise.
        assert!(Role::DeepLc.wanted_by(&cfg));
        cfg.rt_im_train.library_irt = LibraryIrt::Library;
        assert!(!Role::DeepLc.wanted_by(&cfg) && !Role::DeepLc.required_by(&cfg));
        cfg.rt_im_train.library_irt = LibraryIrt::Deeplc;
        assert!(Role::DeepLc.required_by(&cfg) && !Role::DeepLc.wanted_by(&cfg));
        cfg.rt_im_train.library_irt = LibraryIrt::Auto;
        for role in [Role::Rescore, Role::Ms2pip, Role::Mbr] {
            assert!(!role.wanted_by(&cfg), "{:?} has no optional state", role);
        }

        cfg.rt_im_train.finetune_deeplc = true;
        assert!(Role::DeepLc.required_by(&cfg) && !Role::DeepLc.wanted_by(&cfg));
        cfg.rt_im_train.finetune_deeplc = false;
        cfg.predict_frag.rt_predictor = RtPredictorKind::Deeplc;
        assert!(Role::DeepLc.required_by(&cfg));

        cfg.predict_frag.predictor = FragPredictorKind::Ms2pip;
        assert!(Role::Ms2pip.required_by(&cfg));
        assert!(
            !Role::Peptdeep.required_by(&cfg),
            "one fragment predictor is chosen, so the other role stays unrequired"
        );
        cfg.predict_frag.predictor = FragPredictorKind::Peptdeep;
        assert!(Role::Peptdeep.required_by(&cfg));
        assert!(!Role::Ms2pip.required_by(&cfg));
        // An explicit head count makes DeepLC required. The automatic default does not:
        // it is scoped to a run whose retention times are DeepLC's already, so a native
        // run neither requires nor wants an interpreter and still starts without Python.
        cfg.predict_frag.rt_predictor = RtPredictorKind::Native;
        cfg.rt_im_train.library_irt = LibraryIrt::Library;
        cfg.rt_im_train.multihead_calibration = Some(80);
        assert!(
            Role::DeepLc.required_by(&cfg),
            "an explicit count requires it"
        );
        cfg.rt_im_train.multihead_calibration = None;
        assert!(
            !Role::DeepLc.required_by(&cfg) && !Role::DeepLc.wanted_by(&cfg),
            "the default asks nothing of a native configuration"
        );
        // Under the default `library_irt = auto` it is wanted but not required, which is
        // the state the automatic multi-head default rides on.
        cfg.rt_im_train.library_irt = LibraryIrt::Auto;
        assert!(!Role::DeepLc.required_by(&cfg) && Role::DeepLc.wanted_by(&cfg));

        cfg.mbr.strategy = mumdia_core::config::MbrStrategy::RtTransfer;
        assert!(Role::Mbr.required_by(&cfg));
    }

    #[test]
    fn resolve_leaves_a_native_config_untouched() {
        let mut cfg = Config::default();
        // The default `library_irt = auto` makes DeepLC an OPTIONAL role that discovery
        // probes for (next test); the fully native case is the one that keeps the
        // imported iRT.
        cfg.rt_im_train.library_irt = LibraryIrt::Library;
        let res = resolve(&mut cfg).unwrap();
        assert_eq!(res.len(), ALL_ROLES.len());
        assert!(res.iter().all(|r| r.source == "not required"));
        assert!(cfg.rescore.python.is_none());
        assert!(cfg.predict_frag.deeplc_python.is_none());
    }

    #[test]
    fn an_explicit_path_is_honored_and_a_missing_one_is_named() {
        let mut cfg = Config::default();
        // Not required: an unusable path is accepted, because nothing will run it.
        cfg.rescore.python = Some("/definitely/not/here/python".into());
        let res = resolve(&mut cfg).unwrap();
        let r = res.iter().find(|r| r.role == Role::Rescore).unwrap();
        assert_eq!(r.source, "configured");

        // Required: the same path must fail, and the message must name the field
        // and the way out.
        cfg.rescore.classifier = RescorerKind::NnTorch;
        let err = resolve(&mut cfg).unwrap_err().to_string();
        assert!(err.contains("rescore.python"), "{err}");
        assert!(err.contains("/definitely/not/here/python"), "{err}");
        assert!(err.contains("auto"), "{err}");
    }

    #[test]
    fn discovery_failure_names_every_place_it_looked() {
        // Point the role's own variable at something unusable and ask for
        // discovery. Whether a system python exists is machine-dependent, so
        // accept either outcome and only assert the failure message's content.
        let mut cfg = Config::default();
        cfg.rescore.classifier = RescorerKind::NnTorch;
        cfg.rescore.python = Some(AUTO.into());
        match resolve(&mut cfg) {
            Ok(res) => {
                let r = res.iter().find(|r| r.role == Role::Rescore).unwrap();
                assert!(r.python.is_some());
                assert_ne!(r.source, "configured");
            }
            Err(e) => {
                let m = e.to_string();
                assert!(m.contains("MUMDIA_PYTHON_RESCORE"), "{m}");
                assert!(m.contains("CONDA_PREFIX"), "{m}");
                assert!(m.contains("torch"), "{m}");
            }
        }
    }

    #[test]
    fn auto_is_case_insensitive() {
        let mut cfg = Config::default();
        cfg.rescore.python = Some("AUTO".into());
        // Not required, so discovery is skipped and the field is left alone rather
        // than being treated as a literal path.
        let res = resolve(&mut cfg).unwrap();
        let r = res.iter().find(|r| r.role == Role::Rescore).unwrap();
        assert_eq!(r.source, "not required");
        assert_eq!(r.python, None);
    }

    #[test]
    fn script_dir_prefers_a_directory_that_holds_the_workers() {
        let tmp = std::env::temp_dir().join(format!("mumdia_pyres_{}", std::process::id()));
        let real = tmp.join("scripts");
        std::fs::create_dir_all(&real).unwrap();
        std::fs::write(real.join("mbr_worker.py"), b"# test\n").unwrap();

        // Configured value that exists and holds workers is used unchanged.
        let got = resolve_script_dir(&real.to_string_lossy(), None);
        assert_eq!(got, real.to_string_lossy());

        // Relative to the config file: the config lives beside the scripts dir.
        let cfg_file = tmp.join("config.json");
        std::fs::write(&cfg_file, b"{}").unwrap();
        let got = resolve_script_dir("scripts", Some(&cfg_file.to_string_lossy()));
        assert!(
            Path::new(&got).join("mbr_worker.py").exists(),
            "expected the config-relative scripts dir, got {got}"
        );

        // Nothing found: the configured value is returned so the error names it.
        assert_eq!(resolve_script_dir("no_such_dir", None), "no_such_dir");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolve_clears_an_unused_auto_role() {
        // Regression: `run` accepted a config that `run-experiment` rejected with
        // "mbr.python points at an interpreter that does not exist: auto", because
        // resolution left the literal "auto" in place for a role this configuration
        // never uses, and the experiment preflight then stat'ed it. After resolution an
        // unused role must be None: no interpreter was selected, because none is needed.
        let mut cfg = Config::default();
        cfg.rt_im_train.library_irt = LibraryIrt::Library;
        cfg.mbr.python = Some(AUTO.to_string());
        cfg.rescore.python = Some(AUTO.to_string());
        assert!(
            !Role::Mbr.required_by(&cfg),
            "default config must not need MBR"
        );
        assert!(
            !Role::Rescore.required_by(&cfg),
            "default classifier must be native"
        );

        let res = resolve(&mut cfg).expect("resolving unused roles must not fail");
        assert_eq!(cfg.mbr.python, None);
        assert_eq!(cfg.rescore.python, None);
        assert!(res.iter().all(|r| r.python.is_none()));
        assert!(res.iter().all(|r| r.source == "not required"));
    }

    #[test]
    fn resolve_treats_deeplc_as_optional_under_library_irt_auto() {
        // `library_irt = auto` (the default) WANTS DeepLC without requiring it: discovery
        // runs, a miss is not an error, and the field never keeps the literal "auto".
        // Whether a machine has a usable DeepLC is not something a unit test can assume,
        // so the assertion is on the contract, not on the outcome.
        let mut cfg = Config::default();
        cfg.predict_frag.deeplc_python = Some(AUTO.to_string());
        assert!(Role::DeepLc.wanted_by(&cfg) && !Role::DeepLc.required_by(&cfg));
        let res = resolve(&mut cfg).expect("an optional role never fails resolution");
        let r = res.iter().find(|r| r.role == Role::DeepLc).unwrap();
        assert_ne!(r.source, "not required");
        assert_eq!(r.python.is_none(), r.source == "not found (optional)");
        assert_eq!(cfg.predict_frag.deeplc_python, r.python);
        assert!(cfg
            .predict_frag
            .deeplc_python
            .as_deref()
            .is_none_or(|p| !p.eq_ignore_ascii_case(AUTO)));
    }

    #[test]
    fn resolve_honours_a_configured_path_for_an_unused_role() {
        // An explicit path is not discovery, so it is preserved rather than cleared --
        // the run still warns that it is unused, which is the useful signal.
        let mut cfg = Config::default();
        cfg.mbr.python = Some("/some/explicit/python".to_string());
        resolve(&mut cfg).expect("an unused explicit path must not be validated");
        assert_eq!(cfg.mbr.python.as_deref(), Some("/some/explicit/python"));
    }

    #[test]
    fn best_effort_does_not_demand_a_role_the_running_stage_never_uses() {
        // Regression, found by running a benchmark rather than by a test.
        //
        // `Role::required_by` answers "does this CONFIGURATION use the role", which is
        // not "does the subcommand I am about to run use it". Resolving strictly inside
        // `load_config` therefore made `mumdia search-seed` with
        // `rescore.classifier = entrapment` fail at config load, demanding a mokapot
        // interpreter for a rescorer that stage never invokes.
        //
        // Under BestEffort an unresolvable role must NOT error, and must be left as None
        // rather than as the literal "auto" -- so a stage that does need it fails on the
        // missing interpreter instead of trying to execute a program called `auto`.
        let mut cfg = Config::default();
        cfg.rescore.classifier = RescorerKind::Mokapot;
        cfg.rescore.python = Some("/definitely/not/an/interpreter".to_string());
        assert!(Role::Rescore.required_by(&cfg));

        let res = resolve_with(&mut cfg, Strictness::BestEffort)
            .expect("BestEffort must not fail on an unresolvable required role");
        assert!(res.iter().any(|r| r.role == Role::Rescore));

        // Require, on the same config, still fails: that is what the orchestrators use,
        // and early failure is the whole point there.
        let mut strict_cfg = Config::default();
        strict_cfg.rescore.classifier = RescorerKind::Mokapot;
        strict_cfg.rescore.python = Some("/definitely/not/an/interpreter".to_string());
        let err = resolve_with(&mut strict_cfg, Strictness::Require)
            .expect_err("Require must reject an interpreter path that does not exist");
        assert!(
            format!("{err}").contains("rescore.python"),
            "the error should name the field: {err}"
        );
    }

    #[test]
    fn best_effort_clears_an_auto_it_cannot_resolve() {
        // A role asking for discovery that finds nothing must end up None, never "auto".
        let mut cfg = Config::default();
        cfg.rescore.classifier = RescorerKind::Mokapot;
        cfg.rescore.python = Some(AUTO.to_string());
        // Point discovery at nothing so it cannot succeed regardless of the host.
        std::env::set_var("MUMDIA_PYTHON_RESCORE", "/definitely/not/an/interpreter");
        let out = resolve_with(&mut cfg, Strictness::BestEffort);
        std::env::remove_var("MUMDIA_PYTHON_RESCORE");
        out.expect("BestEffort must not fail");
        assert_ne!(
            cfg.rescore.python.as_deref(),
            Some(AUTO),
            "an unresolved \"auto\" must not survive into the resolved config"
        );
    }
    /// The traceback a Windows user's DeepLC worker printed, cut to the lines that matter.
    const WINERROR_1114: &str = r#"Traceback (most recent call last):
  File "C:\Program Files\MuMDIA\binaries\scripts/deeplc_finetune.py", line 67, in <module>
    import deeplc   # import before numpy (OpenMP load order)
  File "C:\Users\x\AppData\Roaming\MuMDIA\python\Lib\site-packages\deeplc\core.py", line 10, in <module>
    import torch
  File "C:\Users\x\AppData\Roaming\MuMDIA\python\Lib\site-packages\torch\__init__.py", line 292, in _load_dll_libraries
    raise err
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. Error loading "C:\Users\x\AppData\Roaming\MuMDIA\python\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies."#;

    #[test]
    fn a_torch_dll_failure_is_answered_with_the_runtime_hint() {
        assert_eq!(load_failure_hint(WINERROR_1114), Some(VC_RUNTIME_HINT));
        for text in [
            r#"OSError: [WinError 126] The specified module could not be found. Error loading "C:\env\Lib\site-packages\torch\lib\fbgemm.dll" or one of its dependencies."#,
            "ImportError: DLL load failed while importing _C: The specified module could not \
             be found. (torch/_C)",
            "oserror: [winerror 1114] ... error loading \"c:/env/torch/lib/c10.dll\"",
        ] {
            assert_eq!(load_failure_hint(text), Some(VC_RUNTIME_HINT), "{text}");
        }
        // Not a DLL failure, or not torch's: no hint, rather than a wrong one.
        for text in [
            "ModuleNotFoundError: No module named 'torch'",
            r#"OSError: [WinError 126] The specified module could not be found. Error loading "C:\env\Lib\site-packages\duckdb\duckdb.dll""#,
            "RuntimeError: CUDA out of memory",
            "",
        ] {
            assert_eq!(load_failure_hint(text), None, "{text}");
        }
        let hint = VC_RUNTIME_HINT;
        assert!(hint.contains("https://aka.ms/vs/17/release/vc_redist.x64.exe"));
        assert!(hint.contains("where msvcp140.dll"));
        assert!(hint.contains("reboot"));
    }

    #[test]
    fn the_probe_imports_deeplc_first_and_runs_torch() {
        let s = import_probe_script(&["numpy", "pandas", "pyarrow", "torch", "deeplc"]);
        let list = s.lines().find(|l| l.starts_with("for m in")).unwrap();
        assert_eq!(
            list, "for m in ['deeplc', 'numpy', 'pandas', 'pyarrow', 'torch']:",
            "deeplc must be imported before numpy, as every DeepLC worker does"
        );
        assert!(s.contains("sys.modules['torch'].zeros(1)"));
        assert!(s.contains("find_spec(m) is None"));
        // Without deeplc the order is the role's own.
        let s = import_probe_script(&["torch", "numpy"]);
        assert!(s.contains("for m in ['torch', 'numpy']:"));
    }

    #[test]
    fn a_probe_is_judged_by_its_marker_lines() {
        assert_eq!(
            judge_import_probe(true, "MUMDIA_TRY deeplc\nMUMDIA_IMPORT_OK\n", ""),
            ImportProbe::Loaded
        );
        assert_eq!(
            judge_import_probe(
                false,
                "MUMDIA_TRY deeplc\nMUMDIA_UNLOADABLE deeplc\n",
                WINERROR_1114
            ),
            ImportProbe::Unloadable {
                module: "deeplc".into(),
                detail: tail_lines(WINERROR_1114, 15),
            }
        );
        // The interpreter died inside an import: the last one announced is named.
        match judge_import_probe(false, "MUMDIA_TRY deeplc\nMUMDIA_TRY torch\n", "") {
            ImportProbe::Unloadable { module, .. } => assert_eq!(module, "torch"),
            other => panic!("{other:?}"),
        }
        // A verdict line from a process that then failed is not a success.
        assert!(matches!(
            judge_import_probe(false, "MUMDIA_IMPORT_OK\n", ""),
            ImportProbe::Unloadable { .. }
        ));
        assert_eq!(tail_lines("a\n\nb\nc\n", 2), "b\nc");
    }

    /// A Python interpreter on this host, or `None` to skip the tests that need one.
    fn any_python() -> Option<PathBuf> {
        ["python3", "python"]
            .into_iter()
            .map(PathBuf::from)
            .find(|p| {
                Command::new(p)
                    .args(["-c", "import sys; sys.exit(0)"])
                    .output()
                    .is_ok_and(|o| o.status.success())
            })
    }

    #[test]
    fn a_package_that_cannot_load_passes_find_spec_and_fails_the_real_import() {
        let Some(py) = any_python() else {
            eprintln!("skipped: no Python interpreter on this host");
            return;
        };
        let dir = std::env::temp_dir().join(format!("mumdia-loadprobe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for (name, body) in [
            ("okpkg", "VALUE = 1\n"),
            (
                "brokenpkg",
                "raise OSError(r'[WinError 1114] A dynamic link library (DLL) initialization \
                 routine failed. Error loading \"C:\\env\\torch\\lib\\c10.dll\" or one of its \
                 dependencies.')\n",
            ),
            ("crashpkg", "import os\nos._exit(9)\n"),
        ] {
            std::fs::create_dir_all(dir.join(name)).unwrap();
            std::fs::write(dir.join(name).join("__init__.py"), body).unwrap();
        }
        let path = dir.to_string_lossy().into_owned();

        // find_spec, as `missing_modules` runs it: every package is present.
        let o = Command::new(&py)
            .args([
                "-c",
                &missing_modules_script(&["okpkg", "brokenpkg", "crashpkg"]),
            ])
            .env("PYTHONPATH", &path)
            .output()
            .unwrap();
        assert!(o.status.success());
        assert_eq!(String::from_utf8_lossy(&o.stdout).trim(), "");

        let probe = |mods: &[&str]| {
            let mut cmd = import_probe_command(&py, mods);
            cmd.env("PYTHONPATH", &path);
            let o = output_with_timeout(cmd, IMPORT_PROBE_TIMEOUT)
                .unwrap()
                .expect("the probe must not time out");
            judge_import_probe(
                o.status.success(),
                &String::from_utf8_lossy(&o.stdout),
                &String::from_utf8_lossy(&o.stderr),
            )
        };
        // The real import: the broken package is named, with its traceback, and the
        // traceback earns the runtime hint.
        match probe(&["okpkg", "brokenpkg"]) {
            ImportProbe::Unloadable { module, detail } => {
                assert_eq!(module, "brokenpkg");
                assert!(detail.contains("WinError 1114"), "{detail}");
                assert_eq!(load_failure_hint(&detail), Some(VC_RUNTIME_HINT));
            }
            other => panic!("expected the broken package to be reported: {other:?}"),
        }
        // A package that kills the interpreter is still named.
        match probe(&["okpkg", "crashpkg"]) {
            ImportProbe::Unloadable { module, .. } => assert_eq!(module, "crashpkg"),
            other => panic!("{other:?}"),
        }
        // Absent modules are skipped (that is `missing_modules`' report); the rest load.
        assert_eq!(
            probe(&["okpkg", "not_a_module_mumdia"]),
            ImportProbe::Loaded
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_probe_that_hangs_is_killed_at_the_timeout() {
        let Some(py) = any_python() else {
            eprintln!("skipped: no Python interpreter on this host");
            return;
        };
        let mut cmd = Command::new(&py);
        cmd.args(["-c", "import time; time.sleep(30)"]);
        let t = std::time::Instant::now();
        let out = output_with_timeout(cmd, std::time::Duration::from_millis(500)).unwrap();
        assert!(out.is_none());
        assert!(t.elapsed() < std::time::Duration::from_secs(20));
    }
}
