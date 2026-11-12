//! Changing the components in the program folder as an administrator.
//!
//! # Why a second process
//!
//! In the machine scope (`crate::scope`) the components are in `components\` in the
//! program folder, which a standard user cannot write to. The Setup screen's Install,
//! Update, Reinstall and Remove then start this same executable again, elevated, with
//! `--install-components <id>` or `--remove-components <id>` ([`run_helper`]). Windows shows
//! its permission prompt, where a standard user can enter an administrator's name and
//! password. The elevated copy does exactly what the button does in the user scope, in
//! its own process ([`helper_main`]), and writes every line to a log in
//! `components\logs\`, which every user can read. The window reads that log as it grows
//! and shows it in the installation log. When the prompt is declined, the screen says so
//! and names the command an administrator can run instead.
//!
//! Elevating the whole window instead would run searches and save settings as the
//! administrator, in the administrator's folders. Only the change to the shared folder
//! needs the rights, so only that runs elevated.
//!
//! # The command line
//!
//! The same flags are what an administrator runs for a scripted or silent deployment,
//! from an elevated prompt or a deployment tool running as SYSTEM, which needs no prompt:
//!
//! ```text
//! mumdia-console.exe --install-components primary,thermo
//! mumdia-console.exe --install-components all
//! mumdia-console.exe --remove-components primary
//! ```
//!
//! The exit code is 0 on success, 1 when an installation or removal failed, 2 for a
//! command line it does not understand and 3 when it cannot write to the components folder
//! (not run as administrator). In the machine scope the log is also written to
//! `components\logs\`. These flags belong to the desktop application; the engine's own
//! command line is unchanged.
//!
//! `--install-components-on-start` is not a helper flag. It opens the window and starts
//! installing the required components, and is what the installer's Finish button passes
//! when the administrator asked for the components to be installed
//! (`wix/components-scope.wxs`).

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::vcruntime::Launch;

/// Install (or update) the named components, then exit.
pub const INSTALL_FLAG: &str = "--install-components";
/// Remove the named components, then exit.
pub const REMOVE_FLAG: &str = "--remove-components";
/// The name of the log file in `components\logs\`, chosen by the window that waits for it.
pub const LOG_FLAG: &str = "--log-name";
/// Open the window and start installing the required components.
pub const ON_START_FLAG: &str = "--install-components-on-start";

/// What `--install-components` accepts, in the order `all` installs them.
pub const INSTALLABLE: &[&str] = &["primary", "ms2pip", "visdia", "thermo"];

pub const EXIT_OK: i32 = 0;
pub const EXIT_FAILED: i32 = 1;
pub const EXIT_USAGE: i32 = 2;
pub const EXIT_CANNOT_WRITE: i32 = 3;

static IN_HELPER: AtomicBool = AtomicBool::new(false);

/// Whether this process is the helper. An installation started here never asks for
/// elevation again (`components::needs_administrator`), so a helper that somehow cannot
/// write fails instead of starting another helper.
pub fn in_helper() -> bool {
    IN_HELPER.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Install,
    Remove,
}

/// A helper command line, parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invocation {
    pub action: Action,
    /// Component ids, without repeats, in the order given.
    pub ids: Vec<String>,
    pub log_name: Option<String>,
}

/// Whether the window was opened to install the components ([`ON_START_FLAG`]).
pub fn wants_install_on_start(args: &[String]) -> bool {
    args.iter().any(|a| a == ON_START_FLAG)
}

/// The helper command in `args` (without the program name), or `None` when there is none
/// and the window should open.
pub fn parse(args: &[String]) -> Option<Result<Invocation, String>> {
    if !args.iter().any(|a| a == INSTALL_FLAG || a == REMOVE_FLAG) {
        return None;
    }
    Some(parse_helper(args))
}

fn parse_helper(args: &[String]) -> Result<Invocation, String> {
    let mut action = None;
    let mut ids: Vec<String> = Vec::new();
    let mut log_name = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            INSTALL_FLAG | REMOVE_FLAG => {
                if action.is_some() {
                    return Err(format!("give {INSTALL_FLAG} or {REMOVE_FLAG}, once"));
                }
                action = Some(if a == INSTALL_FLAG {
                    Action::Install
                } else {
                    Action::Remove
                });
                let list = it.next().ok_or_else(|| {
                    format!("{a} needs the components to act on, for example {a} primary,thermo")
                })?;
                ids.extend(
                    list.split(',')
                        .map(|x| x.trim().to_ascii_lowercase())
                        .filter(|x| !x.is_empty()),
                );
            }
            LOG_FLAG => {
                let n = it
                    .next()
                    .ok_or_else(|| format!("{LOG_FLAG} needs a file name"))?;
                if !valid_log_name(n) {
                    return Err(format!(
                        "{n} is not a log file name: letters, digits, '.', '-' and '_', \
                         ending in .log"
                    ));
                }
                log_name = Some(n.clone());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let Some(action) = action else {
        return Err(format!("give {INSTALL_FLAG} or {REMOVE_FLAG}"));
    };
    let known: &[&str] = match action {
        Action::Install => INSTALLABLE,
        Action::Remove => crate::components::COMPONENT_IDS,
    };
    let mut out: Vec<String> = Vec::new();
    for id in ids {
        let expanded: Vec<String> = if id == "all" && action == Action::Install {
            INSTALLABLE.iter().map(|s| s.to_string()).collect()
        } else {
            vec![id]
        };
        for id in expanded {
            if !known.contains(&id.as_str()) {
                return Err(format!(
                    "unknown component {id}; the components are {}",
                    known.join(", ")
                ));
            }
            if !out.contains(&id) {
                out.push(id);
            }
        }
    }
    if out.is_empty() {
        return Err("no component was named".into());
    }
    Ok(Invocation {
        action,
        ids: out,
        log_name,
    })
}

/// A file name for a log in `components\logs\`: no folder, nothing hidden, and `.log`.
/// The helper runs elevated, so it must not write wherever a command line says.
fn valid_log_name(n: &str) -> bool {
    (5..=100).contains(&n.len())
        && n.ends_with(".log")
        && !n.starts_with('.')
        && n.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
}

/// Where the helper writes: standard output (for a deployment script that captures it)
/// and, in the machine scope or when the window named one, a log file in
/// `components\logs\`.
struct Log {
    file: Option<std::fs::File>,
    path: Option<PathBuf>,
}

impl Log {
    fn open(tools: &Path, name: Option<&str>, machine: bool, action: Action) -> Log {
        let name = match name {
            Some(n) => n.to_string(),
            None if machine => format!(
                "{}-{}.log",
                match action {
                    Action::Install => "install",
                    Action::Remove => "remove",
                },
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
            ),
            None => return Log::none(),
        };
        let dir = tools.join(crate::components::LOGS_DIR_NAME);
        let path = dir.join(name);
        match std::fs::create_dir_all(&dir).and_then(|_| std::fs::File::create(&path)) {
            Ok(f) => Log {
                file: Some(f),
                path: Some(path),
            },
            Err(_) => Log::none(),
        }
    }

    fn none() -> Log {
        Log {
            file: None,
            path: None,
        }
    }

    fn line(&mut self, text: &str) {
        // Ignored: a release build has no console, and a deployment tool may close the pipe.
        let _ = writeln!(std::io::stdout(), "{text}");
        if let Some(f) = self.file.as_mut() {
            let _ = writeln!(f, "{text}");
            let _ = f.flush();
        }
    }
}

/// Run a helper command in this process and return the exit code.
pub fn helper_main(inv: &Invocation) -> i32 {
    IN_HELPER.store(true, Ordering::Relaxed);
    let layout = crate::storage::layout();
    let machine = layout.scope == crate::scope::Scope::Machine;
    if let Err(e) = crate::storage::probe_writable(&layout.tools) {
        let _ = writeln!(
            std::io::stdout(),
            "MuMDIA cannot write to {} ({e}). {}",
            layout.tools.display(),
            if machine {
                "The components on this computer are in the program folder: run this from \
                 an elevated prompt (Run as administrator)."
            } else {
                ""
            }
        );
        return EXIT_CANNOT_WRITE;
    }
    let mut log = Log::open(&layout.tools, inv.log_name.as_deref(), machine, inv.action);
    log.line(&format!(
        "MuMDIA {}: {} {} in {} ({})",
        env!("CARGO_PKG_VERSION"),
        match inv.action {
            Action::Install => "installing",
            Action::Remove => "removing",
        },
        inv.ids.join(", "),
        layout.tools.display(),
        if machine {
            "the program folder, shared by every user"
        } else {
            "this user's own folder"
        }
    ));
    if let Some(p) = &log.path {
        log.line(&format!("log: {}", p.display()));
    }
    for id in &inv.ids {
        let done = match inv.action {
            Action::Install => install_one(id, &mut log),
            Action::Remove => crate::components::remove(id).map(|bytes| {
                log.line(&format!("== removed {id}, {} MB freed", bytes / 1_000_000));
            }),
        };
        if let Err(e) = done {
            log.line(&format!("FAILED: {e}"));
            return EXIT_FAILED;
        }
    }
    log.line("== done");
    EXIT_OK
}

/// Install one component with the same code as the Setup screen, printing its log as it
/// grows.
fn install_one(id: &str, log: &mut Log) -> Result<(), String> {
    let pause = std::time::Duration::from_millis(250);
    let mut seen = 0u64;
    if id == "thermo" {
        let inst = Arc::new(crate::thermo::Installer::default());
        crate::thermo::install(Arc::clone(&inst))?;
        loop {
            let (lines, s) = inst.log_since(&mut seen);
            lines.iter().for_each(|l| log.line(l));
            match s.install_status.as_str() {
                "installing" => std::thread::sleep(pause),
                "done" => return Ok(()),
                _ => return Err(s.error.unwrap_or_else(|| "the installation failed".into())),
            }
        }
    }
    let env =
        crate::components::Env::from_id(id).ok_or_else(|| format!("unknown component {id}"))?;
    let inst = Arc::new(crate::components::Installer::default());
    crate::components::install(Arc::clone(&inst), env)?;
    loop {
        let (lines, s) = inst.log_since(env, &mut seen);
        lines.iter().for_each(|l| log.line(l));
        match s.install_status.as_str() {
            "installing" => std::thread::sleep(pause),
            "done" => return Ok(()),
            _ => return Err(s.error.unwrap_or_else(|| "the installation failed".into())),
        }
    }
}

/// A command line for `ShellExecuteExW`: arguments with a space or a quote are quoted.
#[cfg_attr(not(windows), allow(dead_code))]
fn command_line(args: &[String]) -> String {
    args.iter()
        .map(|a| {
            if a.is_empty() || a.contains([' ', '\t', '"']) {
                format!("\"{}\"", a.replace('"', "\\\""))
            } else {
                a.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Reads a log file as another process appends to it, line by line.
#[cfg_attr(not(windows), allow(dead_code))]
#[derive(Default)]
struct Tail {
    offset: u64,
    partial: Vec<u8>,
}

#[cfg_attr(not(windows), allow(dead_code))]
impl Tail {
    /// Pass every complete line added since the last call to `on_line`. A file that does
    /// not exist yet (the helper has not started) is simply not read.
    fn read(&mut self, path: &Path, on_line: &mut dyn FnMut(String)) {
        let Ok(mut f) = std::fs::File::open(path) else {
            return;
        };
        if f.seek(SeekFrom::Start(self.offset)).is_err() {
            return;
        }
        let mut buf = Vec::new();
        let Ok(n) = f.read_to_end(&mut buf) else {
            return;
        };
        self.offset += n as u64;
        self.partial.extend_from_slice(&buf);
        while let Some(pos) = self.partial.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.partial.drain(..=pos).collect();
            on_line(
                String::from_utf8_lossy(&line)
                    .trim_end_matches(['\r', '\n'])
                    .to_string(),
            );
        }
    }

    /// The last line, when the file did not end with a line break.
    fn finish(&mut self, on_line: &mut dyn FnMut(String)) {
        if !self.partial.is_empty() {
            on_line(
                String::from_utf8_lossy(&self.partial)
                    .trim_end()
                    .to_string(),
            );
            self.partial.clear();
        }
    }
}

/// What the helper's end means for the window: `Ok` when it exited with 0, otherwise the
/// sentence the Setup screen shows.
#[cfg_attr(not(windows), allow(dead_code))]
fn outcome(launch: Launch, args: &[&str], tools: &Path, exe: &Path) -> Result<(), String> {
    let removing = args.first() == Some(&REMOVE_FLAG);
    let what = if removing { "removal" } else { "installation" };
    match launch {
        Launch::Exited(0) => Ok(()),
        Launch::Exited(c) if c as i32 == EXIT_CANNOT_WRITE => Err(format!(
            "Even as administrator, MuMDIA could not write to {}. Check the permissions of the \
             program folder.",
            tools.display()
        )),
        Launch::Exited(c) => Err(format!(
            "The {what} as administrator failed (exit code {c}). The installation log says why."
        )),
        Launch::NotStarted(1223) => Err(format!(
            "Windows asked for administrator permission and it was not given, so nothing was \
             changed. The analysis components on this computer are in the program folder ({}), \
             shared by every user, and only an administrator can {} them. Ask an \
             administrator to press the button again and approve the prompt, or to run this \
             from an elevated prompt:\n\"{}\" {}",
            tools.display(),
            if removing {
                "remove"
            } else {
                "install or update"
            },
            exe.display(),
            args.join(" ")
        )),
        Launch::NotStarted(e) => Err(format!(
            "Windows could not start MuMDIA as administrator (error {e}), so nothing was changed."
        )),
    }
}

/// Run `args` (a helper command) in an elevated copy of this application and wait for
/// it, passing each line of its log to `on_line` as it is written. Windows shows its
/// permission prompt first.
pub fn run_helper(args: &[&str], on_line: &mut dyn FnMut(String)) -> Result<(), String> {
    #[cfg(windows)]
    {
        use std::sync::mpsc::RecvTimeoutError;

        let exe = std::env::current_exe()
            .map_err(|e| format!("cannot find this application's own executable: {e}"))?;
        let tools = crate::components::tools_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let name = format!(
            "{}-{}-{nanos}.log",
            if args.first() == Some(&REMOVE_FLAG) {
                "remove"
            } else {
                "install"
            },
            std::process::id()
        );
        let log_path = tools.join(crate::components::LOGS_DIR_NAME).join(&name);
        let mut all: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        all.push(LOG_FLAG.into());
        all.push(name);
        let params = command_line(&all);
        let (tx, rx) = std::sync::mpsc::channel();
        let launched = exe.clone();
        std::thread::spawn(move || {
            let _ = tx.send(crate::vcruntime::win::run_elevated_with(&launched, &params));
        });
        let mut tail = Tail::default();
        loop {
            tail.read(&log_path, on_line);
            match rx.recv_timeout(std::time::Duration::from_millis(400)) {
                Ok(launch) => {
                    tail.read(&log_path, on_line);
                    tail.finish(on_line);
                    return outcome(launch, args, &tools, &exe);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("the elevated process ended without reporting how".into())
                }
            }
        }
    }
    #[cfg(not(windows))]
    {
        let _ = (args, on_line);
        Err("the components are installed in the program folder only on Windows".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_window_opens_unless_a_helper_command_is_given() {
        assert_eq!(parse(&args(&[])), None);
        assert_eq!(parse(&args(&[ON_START_FLAG])), None);
        assert!(wants_install_on_start(&args(&[ON_START_FLAG])));
        assert!(!wants_install_on_start(&args(&[])));
    }

    #[test]
    fn a_helper_command_names_known_components() {
        let inv = parse(&args(&[INSTALL_FLAG, "primary,Thermo"]))
            .unwrap()
            .unwrap();
        assert_eq!(inv.action, Action::Install);
        assert_eq!(inv.ids, ["primary", "thermo"]);
        assert_eq!(inv.log_name, None);
        let all = parse(&args(&[INSTALL_FLAG, "all,primary"]))
            .unwrap()
            .unwrap();
        assert_eq!(all.ids, INSTALLABLE, "all, in order, without repeats");
        let rm = parse(&args(&[REMOVE_FLAG, "runtime", LOG_FLAG, "remove-1-2.log"]))
            .unwrap()
            .unwrap();
        assert_eq!(rm.action, Action::Remove);
        assert_eq!(rm.ids, ["runtime"]);
        assert_eq!(rm.log_name.as_deref(), Some("remove-1-2.log"));

        for bad in [
            vec![INSTALL_FLAG],
            vec![INSTALL_FLAG, ""],
            vec![INSTALL_FLAG, "diann"],
            vec![INSTALL_FLAG, "settings"],
            vec![REMOVE_FLAG, "all"],
            vec![REMOVE_FLAG, "settings"],
            vec![INSTALL_FLAG, "primary", REMOVE_FLAG, "primary"],
            vec![INSTALL_FLAG, "primary", "--verbose"],
            vec![INSTALL_FLAG, "primary", LOG_FLAG],
        ] {
            assert!(parse(&args(&bad)).unwrap().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_log_name_cannot_point_anywhere_else() {
        assert!(valid_log_name("install-123-456.log"));
        for bad in [
            "..\\..\\Windows\\x.log",
            "../x.log",
            "C:x.log",
            "a b.log",
            ".log",
            ".hidden.log",
            "install.txt",
            "",
        ] {
            assert!(!valid_log_name(bad), "{bad}");
        }
        let long = format!("{}.log", "a".repeat(200));
        assert!(!valid_log_name(&long));
    }

    #[test]
    fn the_command_line_quotes_only_what_needs_it() {
        assert_eq!(
            command_line(&args(&[INSTALL_FLAG, "primary", LOG_FLAG, "x.log"])),
            "--install-components primary --log-name x.log"
        );
        assert_eq!(command_line(&args(&["a b", ""])), "\"a b\" \"\"");
    }

    #[test]
    fn the_log_is_read_as_it_grows_one_whole_line_at_a_time() {
        let dir = std::env::temp_dir().join(format!("mumdia-elevate-tail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("install.log");
        let mut lines = Vec::new();
        let mut tail = Tail::default();
        // Not there yet: nothing, and no error.
        tail.read(&path, &mut |l| lines.push(l));
        assert!(lines.is_empty());
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"== creating the Python environment\r\nhalf a li")
            .unwrap();
        f.flush().unwrap();
        tail.read(&path, &mut |l| lines.push(l));
        assert_eq!(lines, ["== creating the Python environment"]);
        f.write_all(b"ne\nlast").unwrap();
        f.flush().unwrap();
        tail.read(&path, &mut |l| lines.push(l));
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1], "half a line");
        tail.finish(&mut |l| lines.push(l));
        assert_eq!(lines[2], "last");
        drop(f);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn each_end_of_the_helper_is_explained() {
        let tools = PathBuf::from(r"C:\Program Files\MuMDIA\components");
        let exe = PathBuf::from(r"C:\Program Files\MuMDIA\mumdia-console.exe");
        let install = [INSTALL_FLAG, "primary"];
        assert_eq!(outcome(Launch::Exited(0), &install, &tools, &exe), Ok(()));
        let declined = outcome(Launch::NotStarted(1223), &install, &tools, &exe).unwrap_err();
        assert!(declined.contains("permission"), "{declined}");
        assert!(declined.contains("install or update"), "{declined}");
        assert!(
            declined.contains("mumdia-console.exe\" --install-components primary"),
            "the command an administrator can run instead: {declined}"
        );
        let removal = outcome(
            Launch::NotStarted(1223),
            &[REMOVE_FLAG, "primary"],
            &tools,
            &exe,
        )
        .unwrap_err();
        assert!(removal.contains("can remove them"), "{removal}");
        let cannot = outcome(Launch::Exited(3), &install, &tools, &exe).unwrap_err();
        assert!(cannot.contains("could not write"), "{cannot}");
        let failed = outcome(Launch::Exited(1), &install, &tools, &exe).unwrap_err();
        assert!(failed.contains("exit code 1"), "{failed}");
        let other = outcome(Launch::NotStarted(5), &install, &tools, &exe).unwrap_err();
        assert!(other.contains("error 5"), "{other}");
    }
}
