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
use mumdia_core::config::{Config, FragPredictorKind, MbrStrategy, RescorerKind, RtPredictorKind};
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
    /// `mbr.python`: `mbr_worker.py`.
    Mbr,
}

impl Role {
    pub fn field(self) -> &'static str {
        match self {
            Role::Rescore => "rescore.python",
            Role::DeepLc => "predict_frag.deeplc_python",
            Role::Ms2pip => "predict_frag.ms2pip_python",
            Role::Mbr => "mbr.python",
        }
    }

    /// Environment variable that overrides discovery for this role alone.
    pub fn env_var(self) -> &'static str {
        match self {
            Role::Rescore => "MUMDIA_PYTHON_RESCORE",
            Role::DeepLc => "MUMDIA_PYTHON_DEEPLC",
            Role::Ms2pip => "MUMDIA_PYTHON_MS2PIP",
            Role::Mbr => "MUMDIA_PYTHON_MBR",
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
            Role::Mbr => &["numpy", "pyarrow"],
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
            Role::Mbr => &["mbr_worker.py"],
        }
    }

    fn get(self, cfg: &Config) -> Option<&str> {
        match self {
            Role::Rescore => cfg.rescore.python.as_deref(),
            Role::DeepLc => cfg.predict_frag.deeplc_python.as_deref(),
            Role::Ms2pip => cfg.predict_frag.ms2pip_python.as_deref(),
            Role::Mbr => cfg.mbr.python.as_deref(),
        }
    }

    fn set(self, cfg: &mut Config, value: String) {
        match self {
            Role::Rescore => cfg.rescore.python = Some(value),
            Role::DeepLc => cfg.predict_frag.deeplc_python = Some(value),
            Role::Ms2pip => cfg.predict_frag.ms2pip_python = Some(value),
            Role::Mbr => cfg.mbr.python = Some(value),
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
            Role::Mbr => cfg.mbr.python = None,
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
            }
            Role::Ms2pip => cfg.predict_frag.predictor == FragPredictorKind::Ms2pip,
            Role::Mbr => cfg.mbr.strategy != MbrStrategy::None,
        }
    }
}

pub const ALL_ROLES: [Role; 4] = [Role::Rescore, Role::DeepLc, Role::Ms2pip, Role::Mbr];

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
/// `find_spec` rather than a real import: importing torch or deeplc costs seconds
/// and, for DeepLC specifically, has an ordering constraint the workers satisfy but
/// a probe would not. The trade is that a package which is present but broken still
/// passes here, which is why the container image runs a real import in CI.
pub fn missing_modules(python: &Path, modules: &[&str]) -> Result<Vec<String>> {
    let code = format!(
        "import importlib.util as u; \
         print(','.join(p for p in '{}'.split(',') if u.find_spec(p) is None))",
        modules.join(",")
    );
    let out = Command::new(python).args(["-c", &code]).output();
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

/// Resolve every interpreter this run needs, mutating `cfg` so downstream stages
/// see concrete paths. Absent or `"auto"` triggers discovery; an explicit path is
/// taken as given. Returns one [`Resolution`] per role for logging.
///
/// A role that is needed but cannot be resolved is a hard error here rather than
/// at the stage that needs it: preflight exists so a misconfiguration costs
/// seconds, not the hours already spent when rescore is reached.
pub fn resolve(cfg: &mut Config) -> Result<Vec<Resolution>> {
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
                bail!(
                    "{} points at {}, which does not exist. Fix the path, set it to \
                     \"auto\" to discover an interpreter, or set {}.",
                    role.field(),
                    path,
                    role.env_var()
                );
            }
            out.push(Resolution {
                role,
                python: Some(path),
                source: "configured",
            });
            continue;
        }

        if !required {
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
            None => bail!(
                "{} is required by this configuration but no usable interpreter was found. \
                 Tried {}, CONDA_PREFIX, VIRTUAL_ENV, and python3/python on PATH; each must \
                 import {}. Install one (see env/ for conda specs), then either activate it, \
                 set {}, or name it in the config.",
                role.field(),
                role.env_var(),
                role.modules(cfg).join(", "),
                role.env_var()
            ),
        }
    }

    // A configured-but-unused interpreter is a silent no-op today; say so, because
    // it usually means the classifier or predictor is not the one the user thinks.
    for r in &out {
        if r.source == "configured" && !r.role.required_by(cfg) {
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

        cfg.rt_im_train.finetune_deeplc = true;
        assert!(Role::DeepLc.required_by(&cfg));
        cfg.rt_im_train.finetune_deeplc = false;
        cfg.predict_frag.rt_predictor = RtPredictorKind::Deeplc;
        assert!(Role::DeepLc.required_by(&cfg));

        cfg.predict_frag.predictor = FragPredictorKind::Ms2pip;
        assert!(Role::Ms2pip.required_by(&cfg));

        cfg.mbr.strategy = MbrStrategy::RtTransfer;
        assert!(Role::Mbr.required_by(&cfg));
    }

    #[test]
    fn resolve_leaves_a_native_config_untouched() {
        let mut cfg = Config::default();
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
    fn resolve_honours_a_configured_path_for_an_unused_role() {
        // An explicit path is not discovery, so it is preserved rather than cleared --
        // the run still warns that it is unused, which is the useful signal.
        let mut cfg = Config::default();
        cfg.mbr.python = Some("/some/explicit/python".to_string());
        resolve(&mut cfg).expect("an unused explicit path must not be validated");
        assert_eq!(cfg.mbr.python.as_deref(), Some("/some/explicit/python"));
    }
}
