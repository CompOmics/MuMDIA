//! Whether a managed environment's packages LOAD, not only whether they are installed.
//!
//! # Why `find_spec` is not enough
//!
//! The routine check (`components::status_of`) asks `importlib.util.find_spec`, which
//! locates a package without executing it and therefore never loads a native library.
//! That is what makes it cheap enough to run every time the setup screen is shown, and
//! it is also why it passed an environment whose torch could not load at all: on a
//! Windows machine without a current Microsoft Visual C++ runtime, every component showed
//! green, and the first search stopped two minutes in, when the DeepLC worker's
//! `import deeplc` reached `torch\lib\c10.dll`:
//!
//! ```text
//! OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
//! Error loading "...\torch\lib\c10.dll" or one of its dependencies.
//! ```
//!
//! So the application imports the packages for real, in a subprocess with a time limit,
//! at the two moments that decide whether an environment is usable: when an installation
//! finishes, and the first time a session finds an environment already installed. The
//! refresh in between stays on `find_spec`.
//!
//! # Shared with the engine
//!
//! The probe script, the reading of its output and the hint are the engine's
//! (`rust/mumdia/crates/mumdia/src/python.rs`, which `mumdia doctor` runs), repeated here
//! because this crate does not link the engine. Change both together.
//!
//! Only `std`: the module compiles and tests without Tauri.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// How long one environment's import may take. A cold `import deeplc` loads torch and
/// takes seconds; the first one after an installation can take much longer on Windows
/// while the new files are scanned.
pub const TIMEOUT: Duration = Duration::from_secs(120);

/// What to do when PyTorch cannot load its DLLs on Windows. Word for word the engine's
/// `python::VC_RUNTIME_HINT`.
pub const VC_RUNTIME_HINT: &str = "PyTorch cannot load its DLLs on this machine. This is \
     usually a missing or outdated Microsoft Visual C++ 2015-2022 x64 runtime: install the \
     latest one from https://aka.ms/vs/17/release/vc_redist.x64.exe, reboot, then check \
     again. If it is already installed, an older msvcp140.dll or vcruntime140.dll earlier \
     on PATH can be loaded instead of it: `where msvcp140.dll` lists every copy Windows \
     finds, in the order it tries them. See docs/17_troubleshooting.md.";

/// The fix for an import failure, when its output names one this module recognises:
/// PyTorch's Windows DLL failures (`[WinError 1114]`, `[WinError 126]`, `DLL load failed`
/// under `torch`), which a missing or outdated Visual C++ runtime produces.
pub fn load_failure_hint(text: &str) -> Option<&'static str> {
    let t = text.to_ascii_lowercase();
    let names_torch = t.contains("c10.dll") || t.contains("torch");
    let dll_error = t.contains("winerror 1114")
        || t.contains("winerror 126")
        || t.contains("dll load failed")
        || t.contains("dll initialization routine failed");
    (names_torch && dll_error).then_some(VC_RUNTIME_HINT)
}

/// What a real import found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Every installed module imported, and torch ran an operation.
    Loaded,
    /// `module` is installed but raising, or crashing the interpreter, on import.
    /// `detail` is the end of the interpreter's stderr.
    Unloadable { module: String, detail: String },
    /// The import had not finished after `seconds`.
    TimedOut { seconds: u64 },
    /// The interpreter could not be started.
    NotRun(String),
}

/// The probe: import each module for real, `deeplc` first because that is the order every
/// DeepLC worker imports in (docs/13_sidecars.md), skip what `find_spec` cannot locate (the
/// routine check reports that), and run one torch operation. Each import is announced
/// first, so a module that kills the interpreter is still named.
pub fn script(modules: &[&str]) -> String {
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

/// Read the probe's output.
pub fn judge(success: bool, stdout: &str, stderr: &str) -> Outcome {
    let mut tried = None;
    for line in stdout.lines() {
        let line = line.trim();
        if line == "MUMDIA_IMPORT_OK" && success {
            return Outcome::Loaded;
        }
        if let Some(m) = line.strip_prefix("MUMDIA_UNLOADABLE ") {
            return Outcome::Unloadable {
                module: m.to_string(),
                detail: tail(stderr, 15),
            };
        }
        if let Some(m) = line.strip_prefix("MUMDIA_TRY ") {
            tried = Some(m.to_string());
        }
    }
    Outcome::Unloadable {
        module: tried.unwrap_or_else(|| "python".to_string()),
        detail: tail(stderr, 15),
    }
}

/// The last `n` non-empty lines of `s`.
pub fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// Run the probe for `modules` with `cmd`, an interpreter command the caller built (the
/// application's own `engine::command`, which keeps a console window from opening).
pub fn probe(mut cmd: Command, modules: &[&str], timeout: Duration) -> Outcome {
    cmd.args(["-c", &script(modules)])
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return Outcome::NotRun(e.to_string()),
    };
    // Drained on their own threads: a traceback larger than the pipe buffer would
    // otherwise block the child and turn a failure into a timeout.
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
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Outcome::TimedOut {
                    seconds: timeout.as_secs(),
                };
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => return Outcome::NotRun(e.to_string()),
        }
    };
    let stdout = out.join().unwrap_or_default();
    let stderr = err.join().unwrap_or_default();
    judge(
        status.success(),
        &String::from_utf8_lossy(&stdout),
        &String::from_utf8_lossy(&stderr),
    )
}

/// What the interface shows for a failed probe: the sentence for the banner, and the
/// lines for the installation log (the end of the traceback, then the hint). `None` for
/// an environment that loads.
pub fn report(outcome: &Outcome) -> Option<(String, Vec<String>)> {
    let mut log = vec!["== checking that the packages load".to_string()];
    let message = match outcome {
        Outcome::Loaded => return None,
        Outcome::Unloadable { module, detail } => {
            log.extend(detail.lines().map(str::to_string));
            match load_failure_hint(detail) {
                Some(hint) => {
                    log.push(hint.to_string());
                    format!(
                        "The packages are installed, but `{module}` cannot be loaded. {hint} \
                         Re-check runs this test again."
                    )
                }
                None => format!(
                    "The packages are installed, but importing `{module}` fails, so a search \
                     would stop the first time it needs it. The end of the error is in the \
                     log below; installing again repairs a damaged package."
                ),
            }
        }
        Outcome::TimedOut { seconds } => {
            log.push(format!("the import did not finish within {seconds} s"));
            format!(
                "Importing the packages did not finish within {seconds} s. The first import \
                 after an installation can be slow while the new files are scanned; press \
                 Re-check. If it stops again, a search would stop the same way."
            )
        }
        Outcome::NotRun(e) => {
            log.push(e.clone());
            format!("The environment's interpreter could not be started to test it: {e}")
        }
    };
    Some((message, log))
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINERROR_1114: &str = r#"Traceback (most recent call last):
  File "C:\Program Files\MuMDIA\binaries\scripts/deeplc_finetune.py", line 67, in <module>
    import deeplc   # import before numpy (OpenMP load order)
  File "C:\Users\x\AppData\Roaming\MuMDIA\python\Lib\site-packages\deeplc\core.py", line 10, in <module>
    import torch
  File "C:\Users\x\AppData\Roaming\MuMDIA\python\Lib\site-packages\torch\__init__.py", line 292, in _load_dll_libraries
    raise err
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. Error loading "C:\Users\x\AppData\Roaming\MuMDIA\python\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies."#;

    #[test]
    fn the_windows_torch_dll_failures_get_the_runtime_hint() {
        assert_eq!(load_failure_hint(WINERROR_1114), Some(VC_RUNTIME_HINT));
        assert_eq!(
            load_failure_hint(
                r#"OSError: [WinError 126] The specified module could not be found. Error loading "C:\env\torch\lib\fbgemm.dll" or one of its dependencies."#
            ),
            Some(VC_RUNTIME_HINT)
        );
        assert_eq!(
            load_failure_hint("ImportError: DLL load failed while importing _C (torch)"),
            Some(VC_RUNTIME_HINT)
        );
        for text in [
            "ModuleNotFoundError: No module named 'torch'",
            r#"OSError: [WinError 126] Error loading "C:\env\duckdb\duckdb.dll""#,
            "",
        ] {
            assert_eq!(load_failure_hint(text), None, "{text}");
        }
        assert!(VC_RUNTIME_HINT.contains("https://aka.ms/vs/17/release/vc_redist.x64.exe"));
        assert!(VC_RUNTIME_HINT.contains("where msvcp140.dll"));
    }

    #[test]
    fn the_hint_is_the_engines_word_for_word() {
        // The engine's copy, read from the source tree so the two cannot drift apart
        // silently. The desktop crate does not link the engine, hence the text match.
        let engine = include_str!("../../../rust/mumdia/crates/mumdia/src/python.rs");
        let start = engine
            .find("pub const VC_RUNTIME_HINT: &str = ")
            .expect("the engine defines VC_RUNTIME_HINT");
        let ours = include_str!("loadcheck.rs");
        let ours_start = ours.find("pub const VC_RUNTIME_HINT: &str = ").unwrap();
        let body = |s: &str| s[..s.find("\";").unwrap() + 2].to_string();
        assert_eq!(body(&engine[start..]), body(&ours[ours_start..]));
    }

    #[test]
    fn the_probe_imports_deeplc_first_and_runs_torch() {
        let s = script(&["torch", "psm_utils", "numpy", "deeplc"]);
        assert!(
            s.contains("for m in ['deeplc', 'torch', 'psm_utils', 'numpy']:"),
            "{s}"
        );
        assert!(s.contains("sys.modules['torch'].zeros(1)"));
    }

    #[test]
    fn the_output_is_judged_by_its_marker_lines() {
        assert_eq!(
            judge(true, "MUMDIA_TRY a\nMUMDIA_IMPORT_OK\n", ""),
            Outcome::Loaded
        );
        assert_eq!(
            judge(
                false,
                "MUMDIA_TRY deeplc\nMUMDIA_UNLOADABLE deeplc\n",
                WINERROR_1114
            ),
            Outcome::Unloadable {
                module: "deeplc".into(),
                detail: tail(WINERROR_1114, 15)
            }
        );
        assert!(matches!(
            judge(false, "MUMDIA_TRY deeplc\nMUMDIA_TRY torch\n", ""),
            Outcome::Unloadable { ref module, .. } if module == "torch"
        ));
    }

    #[test]
    fn a_failure_reports_a_sentence_and_the_log_lines() {
        assert!(report(&Outcome::Loaded).is_none());
        let (msg, log) = report(&Outcome::Unloadable {
            module: "deeplc".into(),
            detail: tail(WINERROR_1114, 15),
        })
        .unwrap();
        assert!(msg.contains("`deeplc` cannot be loaded"));
        assert!(msg.contains("vc_redist.x64.exe"));
        assert!(msg.contains("Re-check"));
        assert!(log.iter().any(|l| l.contains("WinError 1114")));
        assert_eq!(log.last().map(String::as_str), Some(VC_RUNTIME_HINT));

        let (msg, _) = report(&Outcome::Unloadable {
            module: "mokapot".into(),
            detail: "ImportError: cannot import name 'x'".into(),
        })
        .unwrap();
        assert!(!msg.contains("Visual C++"));
        assert!(msg.contains("installing again"));
        let (msg, _) = report(&Outcome::TimedOut { seconds: 120 }).unwrap();
        assert!(msg.contains("120 s"));
    }

    fn any_python() -> Option<&'static str> {
        ["python3", "python"].into_iter().find(|p| {
            Command::new(p)
                .args(["-c", "pass"])
                .output()
                .is_ok_and(|o| o.status.success())
        })
    }

    /// The incident in miniature: a package `find_spec` locates, whose import raises the
    /// error a torch without its runtime raises. The routine check passes it; the probe
    /// names it and earns the hint.
    #[test]
    fn a_package_that_cannot_load_passes_find_spec_and_fails_the_probe() {
        let Some(py) = any_python() else {
            eprintln!("skipped: no Python interpreter on this host");
            return;
        };
        let dir =
            std::env::temp_dir().join(format!("mumdia-console-loadcheck-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for (name, body) in [
            ("okpkg", "VALUE = 1\n"),
            (
                "brokenpkg",
                "raise OSError(r'[WinError 1114] A dynamic link library (DLL) initialization \
                 routine failed. Error loading \"C:\\env\\torch\\lib\\c10.dll\" or one of its \
                 dependencies.')\n",
            ),
        ] {
            std::fs::create_dir_all(dir.join(name)).unwrap();
            std::fs::write(dir.join(name).join("__init__.py"), body).unwrap();
        }
        let path = dir.to_string_lossy().into_owned();

        let located = Command::new(py)
            .args([
                "-c",
                "import importlib.util as u; \
                 print([m for m in ['okpkg', 'brokenpkg'] if u.find_spec(m) is None])",
            ])
            .env("PYTHONPATH", &path)
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&located.stdout).trim(), "[]");

        let mut cmd = Command::new(py);
        cmd.env("PYTHONPATH", &path);
        match probe(cmd, &["okpkg", "brokenpkg"], TIMEOUT) {
            Outcome::Unloadable { module, detail } => {
                assert_eq!(module, "brokenpkg");
                assert_eq!(load_failure_hint(&detail), Some(VC_RUNTIME_HINT));
            }
            other => panic!("the broken package must be reported: {other:?}"),
        }
        let mut cmd = Command::new(py);
        cmd.env("PYTHONPATH", &path);
        assert_eq!(
            probe(cmd, &["okpkg", "absent_mumdia"], TIMEOUT),
            Outcome::Loaded
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_probe_that_hangs_is_stopped() {
        let Some(py) = any_python() else {
            return;
        };
        // `-c` is given twice; Python runs the first and ignores what follows it as
        // arguments, so the hang stands in for an import that never returns.
        let mut cmd = Command::new(py);
        cmd.args(["-c", "import time; time.sleep(30)"]);
        let t = Instant::now();
        assert_eq!(
            probe(cmd, &["os"], Duration::from_millis(500)),
            Outcome::TimedOut { seconds: 0 }
        );
        assert!(t.elapsed() < Duration::from_secs(20));
    }
}
