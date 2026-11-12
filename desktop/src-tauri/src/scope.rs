//! Where the managed components live: each user's own folder, or the program folder.
//!
//! # The two scopes
//!
//! `user`, the default and the only scope before this module existed: each user installs
//! the Python environments and ThermoRawFileParser into their own data folder
//! (`%LOCALAPPDATA%\MuMDIA`, see `crate::storage`) from the Setup screen, with no
//! administrator rights.
//!
//! `machine`: the administrator who runs the Windows installer chooses the program folder.
//! The components then live in `components\` beside the application
//! (`C:\Program Files\MuMDIA\components` for the default installation folder). An
//! administrator installs and updates them once, and every user of the computer runs them
//! without being able to change them. What a user builds or saves stays per-user in both
//! scopes: libraries, settings, setups, caches, temporary files, model downloads, results,
//! run logs, and the DIA-NN path and licence answer.
//!
//! # Where the choice is recorded
//!
//! The installer writes it to `HKLM\Software\CompOmics\MuMDIA`, value `ComponentsScope`
//! (`user` or `machine`), and reads it back the next time it runs, so an upgrade keeps it
//! (`wix/components-scope.wxs`). The application reads the same value. A missing or
//! unreadable value, another spelling, another platform, or an executable whose folder
//! cannot be found all mean `user`, the scope that needs nothing from an administrator.
//! `desktop/README.md`, "Where the components are installed", is the guide.

use std::path::{Path, PathBuf};

use serde::Serialize;

/// The registry key the installer writes the choice under, in the 64-bit view of
/// `HKEY_LOCAL_MACHINE`. Must match `wix/components-scope.wxs`.
pub const REGISTRY_KEY: &str = r"Software\CompOmics\MuMDIA";
/// The value under [`REGISTRY_KEY`].
pub const REGISTRY_VALUE: &str = "ComponentsScope";
/// The folder beside the application that holds the components in the machine scope.
pub const COMPONENTS_DIR_NAME: &str = "components";

/// Where the managed components live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// Each user's own data folder.
    User,
    /// The program folder, shared by every user of the computer.
    Machine,
}

impl Scope {
    /// The installer's spelling of a scope, compared without case and surrounding space,
    /// as `msiexec` accepts `COMPONENTS_SCOPE=Machine` too.
    pub fn parse(text: &str) -> Option<Scope> {
        match text.trim().to_ascii_lowercase().as_str() {
            "user" => Some(Scope::User),
            "machine" => Some(Scope::Machine),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Scope::User => "user",
            Scope::Machine => "machine",
        }
    }
}

/// The program folder's components folder when the recorded choice is `machine` and the
/// application's folder is known; `None` (the user scope) otherwise.
///
/// `recorded` is the registry value as read, `program_dir` the folder of the running
/// executable. Pure, so both scopes are tested without a registry.
pub fn machine_dir(recorded: Option<&str>, program_dir: Option<&Path>) -> Option<PathBuf> {
    match recorded.and_then(Scope::parse) {
        Some(Scope::Machine) => program_dir
            .filter(|d| d.is_absolute())
            .map(|d| d.join(COMPONENTS_DIR_NAME)),
        _ => None,
    }
}

/// The folder of the running executable: the installation folder of an installed
/// application, `target\<profile>` of a development build.
pub fn program_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
}

/// The installer's recorded choice, as written.
#[cfg(windows)]
fn recorded() -> Option<String> {
    win::read_hklm_string(REGISTRY_KEY, REGISTRY_VALUE)
}

#[cfg(not(windows))]
fn recorded() -> Option<String> {
    None
}

/// The components folder of the machine scope for this process, or `None` in the user
/// scope.
///
/// Read once: the choice is the installer's, and nothing changes it while the application
/// runs. Unit tests always see the user scope, so a developer machine with an installed
/// MuMDIA in the machine scope does not change what they test; [`machine_dir`] is where
/// both scopes are tested.
pub fn machine_components_dir() -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    static DIR: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| machine_dir(recorded().as_deref(), program_dir().as_deref()))
        .clone()
}

/// The scope of this process.
pub fn current() -> Scope {
    if machine_components_dir().is_some() {
        Scope::Machine
    } else {
        Scope::User
    }
}

#[cfg(windows)]
mod win {
    //! One `REG_SZ` read from the 64-bit view of `HKEY_LOCAL_MACHINE`.

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// The string value `name` under `HKLM\<key>`, or `None` when the key or the value is
    /// missing, is not a string, or cannot be read.
    pub fn read_hklm_string(key: &str, name: &str) -> Option<String> {
        use windows_sys::Win32::System::Registry::{
            RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ, RRF_SUBKEY_WOW6464KEY,
        };
        let key = wide(key);
        let name = wide(name);
        let mut buf = vec![0u16; 64];
        for _ in 0..2 {
            let mut bytes = (buf.len() * 2) as u32;
            // SAFETY: plain FFI. `key` and `name` are NUL-terminated and outlive the call,
            // `buf` holds `bytes` bytes, and the call writes at most that many.
            let rc = unsafe {
                RegGetValueW(
                    HKEY_LOCAL_MACHINE,
                    key.as_ptr(),
                    name.as_ptr(),
                    RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY,
                    std::ptr::null_mut(),
                    buf.as_mut_ptr().cast(),
                    &mut bytes,
                )
            };
            const ERROR_MORE_DATA: u32 = 234;
            match rc {
                0 => {
                    let n = (bytes as usize / 2).min(buf.len());
                    let text = String::from_utf16_lossy(&buf[..n]);
                    return Some(text.trim_end_matches('\0').to_string());
                }
                ERROR_MORE_DATA => buf = vec![0u16; bytes as usize / 2 + 1],
                _ => return None,
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn installed() -> PathBuf {
        if cfg!(windows) {
            PathBuf::from(r"C:\Program Files\MuMDIA")
        } else {
            PathBuf::from("/opt/MuMDIA")
        }
    }

    #[test]
    fn the_installer_spellings_are_read_without_case() {
        assert_eq!(Scope::parse("machine"), Some(Scope::Machine));
        assert_eq!(Scope::parse(" Machine "), Some(Scope::Machine));
        assert_eq!(Scope::parse("USER"), Some(Scope::User));
        assert_eq!(Scope::parse("program"), None);
        assert_eq!(Scope::parse(""), None);
        assert_eq!(Scope::Machine.as_str(), "machine");
        assert_eq!(Scope::User.as_str(), "user");
        assert_eq!(
            serde_json::to_value(Scope::Machine).unwrap(),
            serde_json::json!("machine")
        );
    }

    #[test]
    fn only_a_recorded_machine_choice_puts_the_components_in_the_program_folder() {
        let prog = installed();
        assert_eq!(
            machine_dir(Some("machine"), Some(&prog)),
            Some(prog.join(COMPONENTS_DIR_NAME))
        );
        // The default, and everything that is not a machine choice, is the user scope.
        assert_eq!(machine_dir(Some("user"), Some(&prog)), None);
        assert_eq!(machine_dir(None, Some(&prog)), None);
        assert_eq!(machine_dir(Some("everyone"), Some(&prog)), None);
        // Without a known, absolute program folder there is nowhere shared to look.
        assert_eq!(machine_dir(Some("machine"), None), None);
        assert_eq!(
            machine_dir(Some("machine"), Some(Path::new("relative"))),
            None
        );
    }

    #[test]
    fn unit_tests_run_in_the_user_scope() {
        assert_eq!(machine_components_dir(), None);
        assert_eq!(current(), Scope::User);
    }

    #[cfg(windows)]
    #[test]
    fn a_missing_registry_value_reads_as_none() {
        assert_eq!(
            win::read_hklm_string(r"Software\CompOmics\MuMDIA-test-absent", "Nothing"),
            None
        );
        // A value that exists on every Windows, read through the same call.
        let root = win::read_hklm_string(
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            "SystemRoot",
        );
        assert!(
            root.as_deref()
                .is_some_and(|r| r.to_ascii_lowercase().contains("windows")),
            "{root:?}"
        );
    }
}
