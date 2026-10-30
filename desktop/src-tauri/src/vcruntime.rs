//! Installing the Microsoft Visual C++ x64 runtime that PyTorch needs, on request.
//!
//! # Why the application offers this
//!
//! PyTorch's Windows wheels load `torch\lib\c10.dll` and its neighbours against the
//! Microsoft Visual C++ 2015-2022 x64 runtime (`vcruntime140.dll`, `msvcp140.dll`). On a
//! machine without a current one the import fails with `[WinError 1114]` or
//! `[WinError 126]`, and `loadcheck` reports it with `VC_RUNTIME_HINT`. The installer
//! (MSI) carries the runtime as Microsoft's merge module, so a fresh installation has it;
//! this module is for the machine where that did not take effect (an older MuMDIA
//! installer, a runtime removed afterwards, a portable copy of the application), and it
//! turns the hint's link into one button.
//!
//! # What it does, and what it refuses
//!
//! Only on the user's click, and only on Windows:
//!
//! 1. download `URL` (Microsoft's own redirect to the current redistributable) into a new
//!    directory under the user's temporary folder;
//! 2. hold the file open without write sharing, so it cannot be replaced between the
//!    check and the launch, and check its Authenticode signature: it must be valid
//!    (chained to a trusted root, contents unmodified) and the signer's organisation
//!    must be `SIGNER_ORGANIZATION`. Anything else is refused and the file deleted;
//! 3. launch it elevated with `ARGUMENTS` (Windows shows its permission prompt), wait for
//!    it to exit, and read the exit code (`interpret_exit`);
//! 4. after a success, or "a newer version is already installed", or "restart
//!    required", run the real import again (the caller's Re-check).
//!
//! The steps are a trait (`Steps`) so the sequence and every message can be tested
//! without Windows, a network or an elevation prompt; the Windows implementation is
//! `win::WindowsSteps`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;

/// Microsoft's permanent link to the current Visual C++ 2015-2022 x64 redistributable,
/// the same one `loadcheck::VC_RUNTIME_HINT` names.
pub const URL: &str = "https://aka.ms/vs/17/release/vc_redist.x64.exe";

/// Install, with a progress window and no prompts beyond Windows' own permission
/// request, and never restart the machine on its own: the user is told instead.
pub const ARGUMENTS: &str = "/install /passive /norestart";

/// The organisation (`O=`) the installer's signing certificate must name.
pub const SIGNER_ORGANIZATION: &str = "Microsoft Corporation";

/// Larger than any redistributable Microsoft has published (about 25 MB), small enough
/// that a wrong answer from a proxy cannot fill the disk.
const MAX_BYTES: u64 = 256 << 20;

/// Whether an environment's load failure is the one this module fixes: the real import's
/// message carries `VC_RUNTIME_HINT` exactly when the traceback was a torch DLL failure.
pub fn applies(load_error: Option<&str>) -> bool {
    load_error.is_some_and(|e| e.contains(crate::loadcheck::VC_RUNTIME_HINT))
}

/// Whether this build can install the runtime at all. Only Windows has it to install.
pub const fn available() -> bool {
    cfg!(windows)
}

// ---------------------------------------------------------------------------
// The sequence
// ---------------------------------------------------------------------------

/// Where an installation is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Nothing has been asked for in this session.
    #[default]
    Idle,
    Downloading,
    /// Checking the downloaded file's signature.
    Verifying,
    /// The Microsoft installer is running, or Windows is asking for permission to run it.
    Installing,
    /// The installer exited with a code that leaves the runtime installed.
    Done,
    Failed,
}

/// What moves an installation from one phase to the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// The user pressed the button.
    Start,
    Downloaded,
    /// The signature is valid and Microsoft's.
    Verified,
    /// The installer exited with a code `interpret_exit` reads as installed.
    Installed,
    /// Anything that stops the installation: no network, a refused signature, a declined
    /// permission prompt, an installer error.
    Failed,
}

impl Phase {
    /// Whether an installation is under way, so a second one must not start.
    pub fn busy(self) -> bool {
        matches!(
            self,
            Phase::Downloading | Phase::Verifying | Phase::Installing
        )
    }

    /// The phase `event` leads to, or `None` when it does not apply here: a second
    /// Start while busy, or a step reported out of order.
    pub fn next(self, event: Event) -> Option<Phase> {
        match (self, event) {
            (p, Event::Start) if !p.busy() => Some(Phase::Downloading),
            (Phase::Downloading, Event::Downloaded) => Some(Phase::Verifying),
            (Phase::Verifying, Event::Verified) => Some(Phase::Installing),
            (Phase::Installing, Event::Installed) => Some(Phase::Done),
            (p, Event::Failed) if p.busy() => Some(Phase::Failed),
            _ => None,
        }
    }
}

/// What the Microsoft installer's exit code means.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitMeaning {
    /// 0: installed or repaired.
    Installed,
    /// 1638: this version or a newer one is already installed.
    AlreadyNewer,
    /// 3010 (or 1641): installed, and Windows must restart before the new files are used.
    RebootRequired,
    /// 1602: the user closed the installer.
    Cancelled,
    /// 1618: another installation (Windows Update, another installer) is running.
    Busy,
    /// Anything else.
    Failed,
}

impl ExitMeaning {
    /// Whether the runtime is installed afterwards, so the import is worth testing again.
    pub fn installed(self) -> bool {
        matches!(
            self,
            ExitMeaning::Installed | ExitMeaning::AlreadyNewer | ExitMeaning::RebootRequired
        )
    }
}

/// Read the installer's exit code. The installer reports Windows Installer codes, either
/// as the plain number or as the HRESULT that wraps it (`0x8007xxxx`), so both are read.
pub fn interpret_exit(code: u32) -> ExitMeaning {
    let plain = if code & 0xFFFF_0000 == 0x8007_0000 {
        code & 0xFFFF
    } else {
        code
    };
    match plain {
        0 => ExitMeaning::Installed,
        1638 => ExitMeaning::AlreadyNewer,
        3010 | 1641 => ExitMeaning::RebootRequired,
        1602 => ExitMeaning::Cancelled,
        1618 => ExitMeaning::Busy,
        _ => ExitMeaning::Failed,
    }
}

/// The sentence the interface shows for an exit code.
pub fn exit_message(code: u32) -> String {
    match interpret_exit(code) {
        ExitMeaning::Installed => "The Visual C++ runtime is installed. MuMDIA is testing the \
             analysis components again."
            .into(),
        ExitMeaning::AlreadyNewer => "This Visual C++ runtime, or a newer one, was already \
             installed, so a missing runtime is not the cause. MuMDIA is testing the analysis \
             components again; if they still cannot load, an older msvcp140.dll or \
             vcruntime140.dll earlier on PATH is being loaded instead: `where msvcp140.dll` \
             in a command prompt lists every copy, in the order Windows tries them."
            .into(),
        ExitMeaning::RebootRequired => "The Visual C++ runtime is installed, and Windows must \
             restart before programs use it. Restart Windows, then open MuMDIA and press \
             Re-check."
            .into(),
        ExitMeaning::Cancelled => "The Visual C++ installer was closed before it finished; \
             nothing was changed. Press Install Visual C++ runtime to try again."
            .into(),
        ExitMeaning::Busy => "Another installation is running (Windows Update, or another \
             installer). Wait for it to finish, then press Install Visual C++ runtime again."
            .into(),
        ExitMeaning::Failed => format!(
            "The Visual C++ installer stopped with exit code {code}. Install the runtime \
             yourself from {URL}, restart Windows, then press Re-check. The installer's log \
             is in your temporary folder (dd_vcredist_amd64_*.log)."
        ),
    }
}

/// The sentence for an installer that Windows did not start. `code` is the Win32 error
/// from the launch.
pub fn launch_message(code: u32) -> String {
    match code {
        // ERROR_CANCELLED: the permission prompt was declined.
        1223 => "Windows asked for permission to install the Visual C++ runtime and it was not \
             given, so nothing was installed. Press Install Visual C++ runtime to ask again, \
             or install it yourself from {URL}."
            .replace("{URL}", URL),
        _ => format!(
            "Windows could not start the Visual C++ installer (error {code}). Install the \
             runtime yourself from {URL}, restart Windows, then press Re-check."
        ),
    }
}

/// The sentence for a download that failed. `detail` is the transport's own message.
pub fn download_message(detail: &str) -> String {
    format!(
        "The Visual C++ runtime could not be downloaded ({detail}). Check the network \
         connection. Behind a proxy, set HTTPS_PROXY before starting MuMDIA, or download and \
         run the installer yourself from {URL}, restart Windows, then press Re-check."
    )
}

// ---------------------------------------------------------------------------
// The signature
// ---------------------------------------------------------------------------

/// What Windows reports about a file's Authenticode signature
/// (`Get-AuthenticodeSignature`, as the JSON object `SIGNATURE_SCRIPT` prints).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    /// `Valid`, `NotSigned`, `HashMismatch`, `NotTrusted`, `UnknownError`, ...
    pub status: String,
    /// Windows' explanation of `status`.
    pub message: String,
    /// The signing certificate's subject, e.g. `CN=Microsoft Corporation,
    /// O=Microsoft Corporation, L=Redmond, S=Washington, C=US`; empty when unsigned.
    pub subject: String,
}

/// Read `SIGNATURE_SCRIPT`'s output: one JSON object on the last non-empty line.
pub fn parse_signature(stdout: &str) -> Result<Signature, String> {
    let line = stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .ok_or("the signature check printed nothing")?;
    let v: serde_json::Value = serde_json::from_str(line)
        .map_err(|e| format!("the signature check printed something unexpected ({e}): {line}"))?;
    let field = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let status = field("status");
    if status.is_empty() {
        return Err(format!("the signature check reported no status: {line}"));
    }
    Ok(Signature {
        status,
        message: field("message"),
        subject: field("subject"),
    })
}

/// Every value of attribute `key` (`O`, `CN`, ...) in an X.500 distinguished name as
/// Windows formats it: comma-separated, a value with a comma in it double-quoted.
pub fn dn_values(subject: &str, key: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    for ch in subject.chars() {
        match ch {
            '"' => {
                quoted = !quoted;
                cur.push(ch);
            }
            ',' | ';' if !quoted => parts.push(std::mem::take(&mut cur)),
            _ => cur.push(ch),
        }
    }
    parts.push(cur);
    parts
        .iter()
        .filter_map(|p| p.split_once('='))
        .filter(|(k, _)| k.trim().eq_ignore_ascii_case(key))
        .map(|(_, v)| {
            let v = v.trim();
            v.strip_prefix('"')
                .and_then(|x| x.strip_suffix('"'))
                .unwrap_or(v)
                .replace("\"\"", "\"")
        })
        .collect()
}

/// Accept a signature only when Windows calls it valid and every organisation in the
/// signer's name is `SIGNER_ORGANIZATION`.
pub fn judge_signature(sig: &Signature) -> Result<(), String> {
    let refuse = |why: String| {
        Err(format!(
            "The downloaded installer was not run, and has been deleted: {why}. Download the \
             runtime yourself from {URL} and check it before running it."
        ))
    };
    if sig.status != "Valid" {
        let detail = if sig.message.is_empty() {
            String::new()
        } else {
            format!(" ({})", sig.message.trim())
        };
        return refuse(format!(
            "its signature is not valid: Windows reports {}{detail}",
            sig.status
        ));
    }
    let orgs = dn_values(&sig.subject, "O");
    if orgs.is_empty() || orgs.iter().any(|o| o != SIGNER_ORGANIZATION) {
        return refuse(format!(
            "it is signed by \"{}\", not by {SIGNER_ORGANIZATION}",
            sig.subject
        ));
    }
    Ok(())
}

/// Ask Windows PowerShell for a file's Authenticode signature. The path arrives in the
/// environment variable `MUMDIA_VCREDIST_FILE` rather than in the script text, so no
/// character in it can change the script.
pub const SIGNATURE_SCRIPT: &str = "$ErrorActionPreference = 'Stop'\n\
     $s = Get-AuthenticodeSignature -LiteralPath $env:MUMDIA_VCREDIST_FILE\n\
     $subject = ''\n\
     if ($s.SignerCertificate) { $subject = $s.SignerCertificate.Subject }\n\
     [pscustomobject]@{ status = [string]$s.Status; message = [string]$s.StatusMessage; \
     subject = $subject } | ConvertTo-Json -Compress\n";

// ---------------------------------------------------------------------------
// State and the driver
// ---------------------------------------------------------------------------

/// What the interface polls.
#[derive(Clone, Debug, Default, Serialize)]
pub struct State {
    pub phase: Phase,
    /// 0-100 while downloading, when the size is known.
    pub percent: u8,
    /// The sentence for the user: what is happening, or what happened and what next.
    pub message: String,
    /// The installer's exit code, once it has exited.
    pub exit_code: Option<u32>,
    /// What that code means.
    pub outcome: Option<ExitMeaning>,
    /// Windows must restart before the runtime is used.
    pub reboot: bool,
}

/// The shared state of the one installation a session can run at a time.
#[derive(Default)]
pub struct Installer {
    state: Mutex<State>,
}

impl Installer {
    pub fn snapshot(&self) -> State {
        self.state.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Apply `event`; false (and nothing changed) when it does not apply in this phase.
    fn step(&self, event: Event, message: String) -> bool {
        let Ok(mut s) = self.state.lock() else {
            return false;
        };
        match s.phase.next(event) {
            Some(p) => {
                if event == Event::Start {
                    *s = State::default();
                }
                s.phase = p;
                s.message = message;
                true
            }
            None => false,
        }
    }

    fn progress(&self, done: u64, total: Option<u64>) {
        if let Ok(mut s) = self.state.lock() {
            s.percent = match total {
                Some(t) if t > 0 => ((done.min(t) as f64 / t as f64) * 100.0) as u8,
                _ => 0,
            };
            s.message = match total {
                Some(t) => format!(
                    "Downloading the Visual C++ runtime from Microsoft: {} of {} MB.",
                    done / 1_000_000,
                    t.div_ceil(1_000_000)
                ),
                None => format!(
                    "Downloading the Visual C++ runtime from Microsoft: {} MB.",
                    done / 1_000_000
                ),
            };
        }
    }

    /// Record the installer's exit. `on_installed` runs before the phase becomes `Done`,
    /// so an interface that sees `Done` and refreshes already finds the import being
    /// tested again, rather than the old failure.
    fn finish(&self, code: u32, on_installed: &mut dyn FnMut()) -> bool {
        let meaning = interpret_exit(code);
        if meaning.installed() {
            on_installed();
        }
        if let Ok(mut s) = self.state.lock() {
            s.exit_code = Some(code);
            s.outcome = Some(meaning);
            s.reboot = meaning == ExitMeaning::RebootRequired;
        }
        let event = if meaning.installed() {
            Event::Installed
        } else {
            Event::Failed
        };
        self.step(event, exit_message(code)) && meaning.installed()
    }
}

/// How the installer is launched: `Exited` with its code, or `NotStarted` with the Win32
/// error Windows gave instead (1223 when the permission prompt was declined).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Launch {
    Exited(u32),
    NotStarted(u32),
}

/// The three things an installation does to the outside world.
pub trait Steps {
    /// Fetch the installer to `dest`, calling `progress(bytes, total)` as it goes.
    fn download(
        &mut self,
        dest: &Path,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), String>;
    /// Hold `file` against replacement and return its signature. The hold lasts until
    /// `release`.
    fn verify(&mut self, file: &Path) -> Result<Signature, String>;
    /// Run the held installer elevated and wait for it.
    fn launch(&mut self, file: &Path) -> Launch;
    /// Let go of the file.
    fn release(&mut self) {}
}

/// Run one installation to its end with `steps`, recording each phase in `inst`. When
/// the runtime is installed afterwards, `on_installed` (the caller's Re-check) runs just
/// before the phase becomes `Done`, and the result is true.
pub fn drive(
    inst: &Installer,
    steps: &mut dyn Steps,
    work: &Path,
    on_installed: &mut dyn FnMut(),
) -> bool {
    let file = work.join("vc_redist.x64.exe");
    let fail = |msg: String| {
        inst.step(Event::Failed, msg);
        false
    };

    if let Err(e) = std::fs::create_dir_all(work) {
        return fail(format!(
            "Cannot create {} for the download: {e}. Install the runtime yourself from {URL}.",
            work.display()
        ));
    }
    let mut progress = |done, total| inst.progress(done, total);
    if let Err(e) = steps.download(&file, &mut progress) {
        let _ = std::fs::remove_file(&file);
        return fail(download_message(&e));
    }
    inst.step(
        Event::Downloaded,
        "Checking that the installer is signed by Microsoft.".into(),
    );

    let checked = steps.verify(&file).and_then(|sig| judge_signature(&sig));
    if let Err(e) = checked {
        steps.release();
        let _ = std::fs::remove_file(&file);
        return fail(e);
    }
    inst.step(
        Event::Verified,
        "The Microsoft installer is running. Windows asks for permission first; the \
         installer's own window shows its progress."
            .into(),
    );

    let launched = steps.launch(&file);
    steps.release();
    let _ = std::fs::remove_file(&file);
    match launched {
        Launch::Exited(code) => inst.finish(code, on_installed),
        Launch::NotStarted(code) => fail(launch_message(code)),
    }
}

/// A new directory for one download, under the user's temporary folder.
fn work_dir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("mumdia-vcredist-{}-{nanos}", std::process::id()))
}

/// Start an installation on its own thread. `then` runs after an installation that left
/// the runtime installed (the caller's Re-check). Refuses off Windows and while one is
/// running.
pub fn install(inst: Arc<Installer>, then: impl FnOnce() + Send + 'static) -> Result<(), String> {
    if !available() {
        return Err("the Visual C++ runtime is a Windows component".into());
    }
    if !inst.step(
        Event::Start,
        "Downloading the Visual C++ runtime from Microsoft.".into(),
    ) {
        return Err("the Visual C++ runtime is already being installed".into());
    }
    std::thread::spawn(move || {
        let work = work_dir();
        let mut then = Some(then);
        run_platform(&inst, &work, &mut || {
            if let Some(f) = then.take() {
                f();
            }
        });
        let _ = std::fs::remove_dir_all(&work);
    });
    Ok(())
}

#[cfg(windows)]
fn run_platform(inst: &Installer, work: &Path, on_installed: &mut dyn FnMut()) -> bool {
    drive(inst, &mut win::WindowsSteps::default(), work, on_installed)
}

#[cfg(not(windows))]
fn run_platform(inst: &Installer, _work: &Path, _on_installed: &mut dyn FnMut()) -> bool {
    inst.step(
        Event::Failed,
        "the Visual C++ runtime is a Windows component".into(),
    );
    false
}

/// Stream `url` to `dest`, refusing more than `MAX_BYTES`. Bounded in time so a stalled
/// connection ends in an error rather than a spinner that never stops. `ureq` reads
/// `HTTPS_PROXY` / `ALL_PROXY` from the environment.
pub fn download(
    url: &str,
    dest: &Path,
    progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<(), String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .timeout_recv_body(Some(Duration::from_secs(30 * 60)))
        .build()
        .into();
    let resp = agent.get(url).call().map_err(|e| e.to_string())?;
    let total = resp.body().content_length();
    if total.is_some_and(|t| t > MAX_BYTES) {
        return Err(format!(
            "the server offered {} MB, far more than the installer's size",
            total.unwrap_or(0) / 1_000_000
        ));
    }
    let mut reader = resp.into_body().into_reader();
    let mut file =
        std::fs::File::create(dest).map_err(|e| format!("cannot write {}: {e}", dest.display()))?;
    let mut buf = vec![0u8; 1 << 16];
    let mut done: u64 = 0;
    let mut reported: u64 = 0;
    progress(0, total);
    loop {
        let n = std::io::Read::read(&mut reader, &mut buf)
            .map_err(|e| format!("the download was interrupted: {e}"))?;
        if n == 0 {
            break;
        }
        done += n as u64;
        if done > MAX_BYTES {
            return Err("the download is far larger than the installer".into());
        }
        std::io::Write::write_all(&mut file, &buf[..n])
            .map_err(|e| format!("cannot write {}: {e}", dest.display()))?;
        if done - reported >= 1 << 20 {
            reported = done;
            progress(done, total);
        }
    }
    std::io::Write::flush(&mut file).map_err(|e| e.to_string())?;
    if done == 0 {
        return Err("the server sent an empty file".into());
    }
    progress(done, total.or(Some(done)));
    Ok(())
}

#[cfg(windows)]
pub mod win {
    //! The Windows side: the signature check through Windows PowerShell, the hold on the
    //! file, and the elevated launch through `ShellExecuteExW` with the `runas` verb.

    use super::{Launch, Signature, Steps, ARGUMENTS, SIGNATURE_SCRIPT, URL};
    use std::fs::File;
    use std::path::{Path, PathBuf};

    /// Windows PowerShell 5.1, which every supported Windows ships, by its full path so a
    /// `powershell.exe` earlier on PATH is never the one asked.
    fn powershell() -> PathBuf {
        let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
        PathBuf::from(root).join(r"System32\WindowsPowerShell\v1.0\powershell.exe")
    }

    /// Open `path` for reading and allow others only to read it, so nothing can write,
    /// rename or delete it while the handle lives. Reading is all the signature check and
    /// the program loader need.
    pub fn hold(path: &Path) -> std::io::Result<File> {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ: u32 = 0x0000_0001;
        std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(path)
    }

    /// The file's Authenticode signature, as Windows judges it.
    pub fn signature(path: &Path) -> Result<Signature, String> {
        let out = crate::engine::command(&powershell())
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                SIGNATURE_SCRIPT,
            ])
            .env("MUMDIA_VCREDIST_FILE", path)
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|e| format!("Windows PowerShell could not be started to check it: {e}"))?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        super::parse_signature(&stdout).map_err(|e| {
            let stderr = String::from_utf8_lossy(&out.stderr);
            format!(
                "its signature could not be checked: {e} {}",
                crate::loadcheck::tail(&stderr, 3)
            )
        })
    }

    fn wide(s: &std::ffi::OsStr) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        s.encode_wide().chain(std::iter::once(0)).collect()
    }

    /// Run `file` with `ARGUMENTS`, elevated, and wait for it. Windows shows its
    /// permission prompt; declining it is `NotStarted(1223)`.
    pub fn run_elevated(file: &Path) -> Launch {
        use windows_sys::Win32::Foundation::{CloseHandle, GetLastError};
        use windows_sys::Win32::System::Com::{
            CoInitializeEx, COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE,
        };
        use windows_sys::Win32::System::Threading::{
            GetExitCodeProcess, WaitForSingleObject, INFINITE,
        };
        use windows_sys::Win32::UI::Shell::{
            ShellExecuteExW, SEE_MASK_FLAG_NO_UI, SEE_MASK_NOASYNC, SEE_MASK_NOCLOSEPROCESS,
            SHELLEXECUTEINFOW,
        };
        use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

        let verb = wide("runas".as_ref());
        let path = wide(file.as_os_str());
        let params = wide(ARGUMENTS.as_ref());
        // SAFETY: plain FFI. Every pointer in `info` points into a buffer that outlives
        // the call; the struct is zero-initialised as the API documents, and the process
        // handle it returns is closed exactly once below.
        unsafe {
            // ShellExecuteExW may hand the request to shell extensions, which expect COM
            // on the calling thread. This thread is ours and does nothing else.
            CoInitializeEx(
                std::ptr::null(),
                (COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) as u32,
            );
            let mut info: SHELLEXECUTEINFOW = std::mem::zeroed();
            info.cbSize = std::mem::size_of::<SHELLEXECUTEINFOW>() as u32;
            info.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NOASYNC | SEE_MASK_FLAG_NO_UI;
            info.lpVerb = verb.as_ptr();
            info.lpFile = path.as_ptr();
            info.lpParameters = params.as_ptr();
            info.nShow = SW_SHOWNORMAL;
            if ShellExecuteExW(&mut info) == 0 {
                return Launch::NotStarted(GetLastError());
            }
            if info.hProcess.is_null() {
                // Started, but Windows gave no handle to wait on (it does this only when
                // the request went to an already running program, never for an .exe).
                return Launch::NotStarted(0);
            }
            WaitForSingleObject(info.hProcess, INFINITE);
            let mut code: u32 = 0;
            let got = GetExitCodeProcess(info.hProcess, &mut code);
            CloseHandle(info.hProcess);
            if got == 0 {
                return Launch::NotStarted(GetLastError());
            }
            Launch::Exited(code)
        }
    }

    /// The real steps: `URL` over the network, the signature from Windows, the
    /// elevated launch.
    #[derive(Default)]
    pub struct WindowsSteps {
        held: Option<File>,
    }

    impl Steps for WindowsSteps {
        fn download(
            &mut self,
            dest: &Path,
            progress: &mut dyn FnMut(u64, Option<u64>),
        ) -> Result<(), String> {
            super::download(URL, dest, progress)
        }

        fn verify(&mut self, file: &Path) -> Result<Signature, String> {
            // A virus scanner that opened the new file a moment ago can refuse the hold
            // for a moment; a few short retries are cheaper than a failed installation.
            let mut last = None;
            for _ in 0..10 {
                match hold(file) {
                    Ok(f) => {
                        self.held = Some(f);
                        return signature(file);
                    }
                    Err(e) => last = Some(e),
                }
                std::thread::sleep(std::time::Duration::from_millis(300));
            }
            Err(format!(
                "it could not be opened for checking ({})",
                last.map(|e| e.to_string()).unwrap_or_default()
            ))
        }

        fn launch(&mut self, file: &Path) -> Launch {
            run_elevated(file)
        }

        fn release(&mut self) {
            self.held = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_are_read_as_the_installer_means_them() {
        assert_eq!(interpret_exit(0), ExitMeaning::Installed);
        assert_eq!(interpret_exit(1638), ExitMeaning::AlreadyNewer);
        assert_eq!(interpret_exit(0x8007_0666), ExitMeaning::AlreadyNewer);
        assert_eq!(interpret_exit(3010), ExitMeaning::RebootRequired);
        assert_eq!(interpret_exit(0x8007_0BC2), ExitMeaning::RebootRequired);
        assert_eq!(interpret_exit(1641), ExitMeaning::RebootRequired);
        assert_eq!(interpret_exit(1602), ExitMeaning::Cancelled);
        assert_eq!(interpret_exit(1618), ExitMeaning::Busy);
        assert_eq!(interpret_exit(1603), ExitMeaning::Failed);
        assert_eq!(interpret_exit(0x8007_0643), ExitMeaning::Failed);
        assert_eq!(interpret_exit(u32::MAX), ExitMeaning::Failed);
        for ok in [0, 1638, 3010] {
            assert!(interpret_exit(ok).installed());
        }
        for bad in [1602, 1618, 1603] {
            assert!(!interpret_exit(bad).installed());
        }
    }

    #[test]
    fn each_outcome_tells_the_user_what_to_do_next() {
        assert!(exit_message(0).contains("testing the analysis components again"));
        let newer = exit_message(1638);
        assert!(newer.contains("already installed"));
        assert!(newer.contains("where msvcp140.dll"));
        let reboot = exit_message(3010);
        assert!(reboot.contains("Restart Windows"));
        assert!(reboot.contains("Re-check"));
        let failed = exit_message(1603);
        assert!(failed.contains("1603"));
        assert!(failed.contains(URL));
        assert!(launch_message(1223).contains("permission"));
        assert!(launch_message(1223).contains(URL));
        assert!(!launch_message(1223).contains("{URL}"));
        assert!(launch_message(5).contains("error 5"));
        let net = download_message("dns failed");
        assert!(net.contains("dns failed"));
        assert!(net.contains("HTTPS_PROXY"));
        assert!(net.contains(URL));
    }

    #[test]
    fn the_phases_only_move_forward_and_one_installation_runs_at_a_time() {
        use Event::{Downloaded, Installed, Start, Verified};
        use Phase::{Done, Downloading, Idle, Installing, Verifying};
        assert_eq!(Idle.next(Start), Some(Downloading));
        assert_eq!(Downloading.next(Start), None);
        assert_eq!(Verifying.next(Start), None);
        assert_eq!(Installing.next(Start), None);
        assert_eq!(Done.next(Start), Some(Downloading));
        assert_eq!(Phase::Failed.next(Start), Some(Downloading));
        assert_eq!(Downloading.next(Downloaded), Some(Verifying));
        assert_eq!(Verifying.next(Verified), Some(Installing));
        assert_eq!(Installing.next(Installed), Some(Done));
        // No skipping the signature check.
        assert_eq!(Downloading.next(Verified), None);
        assert_eq!(Downloading.next(Installed), None);
        assert_eq!(Verifying.next(Installed), None);
        for p in [Downloading, Verifying, Installing] {
            assert_eq!(p.next(Event::Failed), Some(Phase::Failed));
        }
        assert_eq!(Idle.next(Event::Failed), None);
        assert_eq!(Done.next(Event::Failed), None);
    }

    #[test]
    fn the_signer_is_read_from_the_distinguished_name() {
        let ms = "CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US";
        assert_eq!(dn_values(ms, "O"), vec!["Microsoft Corporation"]);
        assert_eq!(dn_values(ms, "cn"), vec!["Microsoft Corporation"]);
        assert_eq!(
            dn_values(r#"CN=x, O="Microsoft Corporation, Inc", C=US"#, "O"),
            vec!["Microsoft Corporation, Inc"]
        );
        assert!(dn_values("CN=Microsoft Corporation", "O").is_empty());
        assert!(dn_values("", "O").is_empty());
    }

    fn sig(status: &str, subject: &str) -> Signature {
        Signature {
            status: status.into(),
            message: String::new(),
            subject: subject.into(),
        }
    }

    #[test]
    fn only_a_valid_microsoft_signature_is_accepted() {
        let ms = "CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US";
        assert!(judge_signature(&sig("Valid", ms)).is_ok());
        // Windows' own binaries are signed as "Microsoft Windows", same organisation.
        assert!(judge_signature(&sig(
            "Valid",
            "CN=Microsoft Windows, O=Microsoft Corporation, L=Redmond, S=Washington, C=US"
        ))
        .is_ok());
        for (s, subject, why) in [
            ("HashMismatch", ms, "HashMismatch"),
            ("NotSigned", "", "NotSigned"),
            ("NotTrusted", ms, "NotTrusted"),
            ("UnknownError", ms, "UnknownError"),
            (
                "Valid",
                "CN=Microsoft Corporation, O=Contoso Ltd, C=US",
                "Contoso",
            ),
            ("Valid", "CN=Microsoft Corporation", "not by Microsoft"),
            (
                "Valid",
                "CN=x, O=Microsoft Corporation, O=Contoso Ltd",
                "not by Microsoft",
            ),
            (
                "Valid",
                "CN=x, O=Microsoft Corporation Evil",
                "not by Microsoft",
            ),
        ] {
            let err = judge_signature(&sig(s, subject)).unwrap_err();
            assert!(err.contains(why), "{s} {subject}: {err}");
            assert!(err.contains("was not run"), "{err}");
        }
    }

    #[test]
    fn the_powershell_answer_is_parsed() {
        // Verbatim from Windows PowerShell 5.1 on the vc_redist.x64.exe that URL served on
        // 2026-10-30.
        let out = r#"{"status":"Valid","message":"Signature verified.","subject":"CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US"}"#;
        let s = parse_signature(&format!("\r\n{out}\r\n\r\n")).unwrap();
        assert_eq!(s.status, "Valid");
        assert_eq!(s.message, "Signature verified.");
        assert!(judge_signature(&s).is_ok());
        let tampered = r#"{"status":"HashMismatch","message":"The contents of file ... might have been changed by an unauthorized user or process, because the hash of the file does not match the hash stored in the digital signature.","subject":"CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US"}"#;
        let s = parse_signature(tampered).unwrap();
        assert!(judge_signature(&s).unwrap_err().contains("HashMismatch"));
        assert!(parse_signature("").is_err());
        assert!(parse_signature("Get-AuthenticodeSignature : File x was not found.").is_err());
        assert!(parse_signature(r#"{"subject":"x"}"#).is_err());
    }

    #[test]
    fn the_script_takes_the_path_from_the_environment_only() {
        assert!(SIGNATURE_SCRIPT.contains("-LiteralPath $env:MUMDIA_VCREDIST_FILE"));
        assert!(SIGNATURE_SCRIPT.contains("ConvertTo-Json -Compress"));
    }

    #[test]
    fn only_the_torch_dll_failure_offers_the_installer() {
        let msg = crate::loadcheck::report(&crate::loadcheck::Outcome::Unloadable {
            module: "deeplc".into(),
            detail: "OSError: [WinError 1114] A dynamic link library (DLL) initialization \
                     routine failed. Error loading \"C:\\env\\torch\\lib\\c10.dll\""
                .into(),
        })
        .unwrap()
        .0;
        assert!(applies(Some(&msg)));
        let other = crate::loadcheck::report(&crate::loadcheck::Outcome::Unloadable {
            module: "mokapot".into(),
            detail: "ImportError: cannot import name 'x'".into(),
        })
        .unwrap()
        .0;
        assert!(!applies(Some(&other)));
        assert!(!applies(None));
        assert!(URL == "https://aka.ms/vs/17/release/vc_redist.x64.exe");
        assert!(crate::loadcheck::VC_RUNTIME_HINT.contains(URL));
    }

    /// Steps that record what was asked of them.
    struct Fake {
        download: Result<(), String>,
        signature: Result<Signature, String>,
        launch: Launch,
        calls: Vec<&'static str>,
    }

    impl Fake {
        fn ok(launch: Launch) -> Self {
            Fake {
                download: Ok(()),
                signature: Ok(sig(
                    "Valid",
                    "CN=Microsoft Corporation, O=Microsoft Corporation, C=US",
                )),
                launch,
                calls: Vec::new(),
            }
        }
    }

    impl Steps for Fake {
        fn download(
            &mut self,
            dest: &Path,
            progress: &mut dyn FnMut(u64, Option<u64>),
        ) -> Result<(), String> {
            self.calls.push("download");
            std::fs::write(dest, b"MZ").unwrap();
            progress(1, Some(2));
            self.download.clone()
        }
        fn verify(&mut self, file: &Path) -> Result<Signature, String> {
            assert!(file.exists(), "verified before it was downloaded");
            self.calls.push("verify");
            self.signature.clone()
        }
        fn launch(&mut self, file: &Path) -> Launch {
            assert!(file.exists());
            self.calls.push("launch");
            self.launch
        }
        fn release(&mut self) {
            self.calls.push("release");
        }
    }

    fn work(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "mumdia-vcruntime-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn started() -> Installer {
        let inst = Installer::default();
        assert!(inst.step(Event::Start, "start".into()));
        inst
    }

    /// `drive`, counting the Re-checks it asks for and the phase each one saw.
    fn run(inst: &Installer, steps: &mut dyn Steps, dir: &Path) -> (bool, Vec<Phase>) {
        let mut seen = Vec::new();
        let ok = drive(inst, steps, dir, &mut || seen.push(inst.snapshot().phase));
        (ok, seen)
    }

    #[test]
    fn a_successful_installation_ends_done_and_asks_for_the_recheck() {
        let inst = started();
        let dir = work("ok");
        let mut f = Fake::ok(Launch::Exited(0));
        let (ok, rechecks) = run(&inst, &mut f, &dir);
        assert!(ok);
        // Asked for once, before the phase said Done, so a refresh on Done already
        // finds the import being tested again.
        assert_eq!(rechecks, [Phase::Installing]);
        assert_eq!(f.calls, ["download", "verify", "launch", "release"]);
        let s = inst.snapshot();
        assert_eq!(s.phase, Phase::Done);
        assert_eq!(s.exit_code, Some(0));
        assert_eq!(s.outcome, Some(ExitMeaning::Installed));
        assert!(!s.reboot);
        assert!(!dir.join("vc_redist.x64.exe").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn newer_already_installed_and_reboot_required_also_recheck() {
        for (code, reboot) in [(1638, false), (3010, true)] {
            let inst = started();
            let dir = work(&format!("c{code}"));
            let (ok, rechecks) = run(&inst, &mut Fake::ok(Launch::Exited(code)), &dir);
            assert!(ok);
            assert_eq!(rechecks.len(), 1);
            let s = inst.snapshot();
            assert_eq!(s.phase, Phase::Done);
            assert_eq!(s.reboot, reboot);
            assert_eq!(s.message, exit_message(code));
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn a_refused_signature_is_never_launched_and_the_file_is_deleted() {
        let inst = started();
        let dir = work("sig");
        let mut f = Fake::ok(Launch::Exited(0));
        f.signature = Ok(sig("Valid", "CN=Someone, O=Someone Else, C=US"));
        assert_eq!(run(&inst, &mut f, &dir), (false, vec![]));
        assert_eq!(f.calls, ["download", "verify", "release"]);
        assert!(!dir.join("vc_redist.x64.exe").exists());
        let s = inst.snapshot();
        assert_eq!(s.phase, Phase::Failed);
        assert!(
            s.message.contains("not by Microsoft Corporation"),
            "{}",
            s.message
        );

        let inst = started();
        let mut f = Fake::ok(Launch::Exited(0));
        f.signature = Err("its signature could not be checked".into());
        assert_eq!(run(&inst, &mut f, &dir), (false, vec![]));
        assert!(!f.calls.contains(&"launch"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_network_fails_with_the_manual_link() {
        let inst = started();
        let dir = work("net");
        let mut f = Fake::ok(Launch::Exited(0));
        f.download = Err("io: Connection refused".into());
        assert_eq!(run(&inst, &mut f, &dir), (false, vec![]));
        assert_eq!(f.calls, ["download"]);
        let s = inst.snapshot();
        assert_eq!(s.phase, Phase::Failed);
        assert!(s.message.contains("Connection refused"));
        assert!(s.message.contains(URL));
        assert!(!dir.join("vc_redist.x64.exe").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_declined_prompt_or_a_failed_installer_does_not_recheck() {
        let inst = started();
        let dir = work("uac");
        let (ok, rechecks) = run(&inst, &mut Fake::ok(Launch::NotStarted(1223)), &dir);
        assert!(!ok);
        assert!(rechecks.is_empty());
        let s = inst.snapshot();
        assert_eq!(s.phase, Phase::Failed);
        assert!(s.message.contains("permission"));
        assert_eq!(s.exit_code, None);

        let inst = started();
        let (ok, rechecks) = run(&inst, &mut Fake::ok(Launch::Exited(1603)), &dir);
        assert!(!ok);
        assert!(rechecks.is_empty());
        let s = inst.snapshot();
        assert_eq!(s.phase, Phase::Failed);
        assert_eq!(s.exit_code, Some(1603));
        assert_eq!(s.outcome, Some(ExitMeaning::Failed));
        // And it can be tried again.
        assert!(inst.step(Event::Start, "again".into()));
        assert_eq!(inst.snapshot().exit_code, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_second_start_while_one_runs_is_refused() {
        let inst = started();
        assert!(!inst.step(Event::Start, "again".into()));
        assert_eq!(inst.snapshot().phase, Phase::Downloading);
    }

    #[cfg(not(windows))]
    #[test]
    fn off_windows_nothing_is_offered_or_started() {
        assert!(!available());
        let inst = Arc::new(Installer::default());
        assert!(install(Arc::clone(&inst), || panic!("must not recheck")).is_err());
        assert_eq!(inst.snapshot().phase, Phase::Idle);
    }

    /// A one-request HTTP server on loopback, so the download is exercised for real
    /// without the network.
    fn serve_once(status: &str, body: Vec<u8>) -> String {
        use std::io::Write as _;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let status = status.to_string();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut head = Vec::new();
                let mut buf = [0u8; 1024];
                loop {
                    match std::io::Read::read(&mut sock, &mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            head.extend_from_slice(&buf[..n]);
                            if head.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                    }
                }
                let _ = write!(
                    sock,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(&body);
                let _ = sock.flush();
            }
        });
        format!("http://127.0.0.1:{port}/vc_redist.x64.exe")
    }

    #[test]
    fn the_download_streams_with_progress() {
        let body: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        let url = serve_once("200 OK", body.clone());
        let dir = work("dl");
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("f.exe");
        let mut seen = Vec::new();
        download(&url, &dest, &mut |d, t| seen.push((d, t))).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert_eq!(seen.first(), Some(&(0, Some(3_000_000))));
        assert_eq!(seen.last(), Some(&(3_000_000, Some(3_000_000))));
        assert!(seen.len() >= 3, "{seen:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_download_is_an_error_with_the_reason() {
        let dir = work("dl404");
        std::fs::create_dir_all(&dir).unwrap();
        let url = serve_once("404 Not Found", b"no".to_vec());
        let err = download(&url, &dir.join("f.exe"), &mut |_, _| {}).unwrap_err();
        assert!(err.contains("404"), "{err}");
        // Nothing listening: the error a machine without a network sees.
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let err = download(
            &format!("http://127.0.0.1:{port}/x"),
            &dir.join("g.exe"),
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(!err.is_empty());
        assert!(download_message(&err).contains(URL));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(windows)]
    mod on_windows {
        use super::super::win::*;
        use super::super::*;

        fn system_exe(name: &str) -> PathBuf {
            let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
            PathBuf::from(root).join("System32").join(name)
        }

        /// The signature check on files whose answer is known: a Windows binary (signed by
        /// Microsoft, through the catalogue) passes; an unsigned file and a modified copy
        /// do not.
        #[test]
        fn windows_judges_real_signatures() {
            let whoami = system_exe("whoami.exe");
            let s = signature(&whoami).unwrap();
            assert_eq!(s.status, "Valid", "{s:?}");
            assert!(judge_signature(&s).is_ok(), "{s:?}");

            let dir = std::env::temp_dir().join(format!("mumdia-vcrt-sig-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let unsigned = dir.join("unsigned.exe");
            std::fs::write(&unsigned, b"MZ not a program").unwrap();
            let s = signature(&unsigned).unwrap();
            assert_ne!(s.status, "Valid");
            assert!(judge_signature(&s).is_err());
            // A path with characters PowerShell would otherwise interpret.
            let odd = dir.join("a $(x) 'b' ;c.exe");
            std::fs::write(&odd, b"MZ").unwrap();
            let s = signature(&odd).unwrap();
            assert_ne!(s.status, "Valid");
            let _ = std::fs::remove_dir_all(&dir);
        }

        /// The hold blocks replacement but not reading, checking or running the file.
        #[test]
        fn a_held_file_can_be_checked_and_run_but_not_replaced() {
            let dir = std::env::temp_dir().join(format!("mumdia-vcrt-hold-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let exe = dir.join("whoami.exe");
            std::fs::copy(system_exe("whoami.exe"), &exe).unwrap();
            let held = hold(&exe).unwrap();
            assert!(std::fs::OpenOptions::new().write(true).open(&exe).is_err());
            assert!(std::fs::remove_file(&exe).is_err());
            assert!(std::fs::rename(&exe, dir.join("other.exe")).is_err());
            assert_eq!(signature(&exe).unwrap().status, "Valid");
            let out = std::process::Command::new(&exe).output().unwrap();
            assert!(out.status.success());
            drop(held);
            std::fs::remove_file(&exe).unwrap();
            let _ = std::fs::remove_dir_all(&dir);
        }

        /// The whole sequence against Microsoft's server: download, signature, elevated
        /// launch, exit code. It installs the runtime, so it runs only when asked
        /// (`cargo test --lib -- --ignored real_runtime_install`); the Windows CI job does,
        /// on a runner where the prompt is not shown and a runtime is already present.
        #[test]
        #[ignore]
        fn real_runtime_install() {
            let inst = Installer::default();
            assert!(inst.step(Event::Start, "start".into()));
            let dir = std::env::temp_dir().join(format!("mumdia-vcrt-real-{}", std::process::id()));
            let mut rechecks = 0;
            let installed = drive(&inst, &mut WindowsSteps::default(), &dir, &mut || {
                rechecks += 1
            });
            assert_eq!(rechecks, usize::from(installed));
            let s = inst.snapshot();
            eprintln!("final state: {s:?}");
            assert!(installed, "{s:?}");
            assert_eq!(s.phase, Phase::Done);
            assert!(s.outcome.is_some_and(ExitMeaning::installed));
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
