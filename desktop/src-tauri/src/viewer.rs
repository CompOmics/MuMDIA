//! visDIA, the results viewer: start it on a results folder and show it in the browser.
//!
//! visDIA (`mumdia-viewer`, github.com/CompOmics/visDIA) is a local web application. It
//! serves one results folder at `http://127.0.0.1:<port>/<token>/`, where the random
//! token keeps other users of a shared machine out, and runs until it is stopped. This
//! module starts it from the managed environment (`components::Env::Visdia`) with
//! `--no-browser`, reads the address it prints, and opens that address itself, so the
//! same code path serves a first open and a repeat.
//!
//! One viewer per folder. Opening a folder whose viewer still runs shows that viewer
//! again instead of starting a second server on the same data. Every viewer is stopped
//! when the application closes; nothing else would, because it has no window of its own.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Stdio};
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::components::{self, Env};

/// How long to wait for the viewer to print its address. Opening a large experiment
/// indexes its tables first, which takes a while on a slow disk.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);

/// The lines of the viewer's output kept for an error message.
const TAIL_LINES: usize = 20;

struct Running {
    child: Child,
    url: String,
}

/// The viewers this application started, keyed by results folder.
#[derive(Default)]
pub struct Viewers {
    live: Mutex<BTreeMap<String, Running>>,
}

impl Viewers {
    /// Whether any viewer is still running (it holds the visDIA environment open).
    pub fn any_running(&self) -> bool {
        self.live
            .lock()
            .map(|mut m| {
                m.retain(|_, r| matches!(r.child.try_wait(), Ok(None)));
                !m.is_empty()
            })
            .unwrap_or(false)
    }

    /// Stop every viewer. Called when the application closes.
    pub fn stop_all(&self) {
        if let Ok(mut m) = self.live.lock() {
            for (_, mut r) in std::mem::take(&mut *m) {
                crate::run::kill_tree(r.child.id());
                let _ = r.child.wait();
            }
        }
    }
}

/// The address in a line the viewer prints once it is serving:
/// `mumdia-viewer: serving <dir> (<kind>) at http://127.0.0.1:<port>/<token>/`.
fn serving_url(line: &str) -> Option<String> {
    if !line.contains("mumdia-viewer: serving ") {
        return None;
    }
    let url = line.rsplit(" at ").next()?.trim();
    is_local_url(url).then(|| url.to_string())
}

/// Only a loopback http address is ever opened: it is the one thing the viewer prints,
/// and this keeps a malformed line from becoming an arbitrary launch.
fn is_local_url(url: &str) -> bool {
    url.starts_with("http://127.0.0.1:") && !url.contains(char::is_whitespace)
}

fn open_in_browser(url: &str) -> Result<(), String> {
    if !is_local_url(url) {
        return Err(format!("{url} is not a local viewer address"));
    }
    // For the integration test, which must not open windows on the machine running it.
    if std::env::var_os("MUMDIA_VIEWER_NO_BROWSER").is_some() {
        return Ok(());
    }
    #[cfg(windows)]
    let r = std::process::Command::new("rundll32")
        .args(["url.dll,FileProtocolHandler", url])
        .spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(url).spawn();
    r.map(|_| ())
        .map_err(|e| format!("could not open {url}: {e}"))
}

/// Show `dir` in visDIA, starting a viewer if none is running for it, and return the
/// address. `fasta` gives the protein sequences for the coverage views when the run
/// did not record one (a search on a library DIA-NN built from that FASTA).
pub fn open(viewers: &Viewers, dir: &str, fasta: Option<&str>) -> Result<String, String> {
    let python = components::managed_python(Env::Visdia);
    if !python.is_file() {
        return Err(
            "visDIA is not installed. Install it on the Setup screen, under \"visDIA results \
             viewer\"."
                .into(),
        );
    }
    let root = Path::new(dir);
    if !root.is_dir() {
        return Err(format!("{dir} is not a folder"));
    }
    let key = std::fs::canonicalize(root)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| dir.to_string());

    // A viewer for this folder that still runs: show it again.
    if let Ok(mut m) = viewers.live.lock() {
        if let Some(r) = m.get_mut(&key) {
            if matches!(r.child.try_wait(), Ok(None)) {
                let url = r.url.clone();
                drop(m);
                open_in_browser(&url)?;
                return Ok(url);
            }
            m.remove(&key);
        }
    }

    let mut cmd = crate::engine::command(&python);
    cmd.args(["-m", "mumdia_viewer", dir, "--no-browser"]);
    if let Some(f) = fasta.filter(|f| Path::new(f).is_file()) {
        cmd.args(["--fasta", f]);
    }
    // Python buffers a piped stdout, and the address line would then arrive only when
    // the buffer filled, which for a quiet server is never.
    cmd.env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("visDIA did not start ({}): {e}", python.display()))?;

    // Both streams on their own threads, for the life of the viewer: a pipe nobody
    // reads fills up and stalls the server.
    let (tx, rx) = mpsc::channel::<String>();
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
        let tx = tx.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                // After the address has been read nobody receives; keep draining.
                let _ = tx.send(line);
            }
        });
    }
    drop(tx);

    let started = Instant::now();
    let mut tail: Vec<String> = Vec::new();
    let url = loop {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(line) => {
                if let Some(url) = serving_url(&line) {
                    break url;
                }
                tail.push(line);
                if tail.len() > TAIL_LINES {
                    tail.remove(0);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Both streams closed: the viewer has exited.
                let _ = child.wait();
                return Err(failure("visDIA stopped before it was ready", &tail));
            }
        }
        if let Ok(Some(status)) = child.try_wait() {
            // Collect what it said on the way out before reporting.
            while let Ok(line) = rx.recv_timeout(Duration::from_millis(200)) {
                tail.push(line);
            }
            return Err(failure(&format!("visDIA exited ({status})"), &tail));
        }
        if started.elapsed() > STARTUP_TIMEOUT {
            crate::run::kill_tree(child.id());
            let _ = child.wait();
            return Err(failure(
                "visDIA did not report an address within five minutes",
                &tail,
            ));
        }
    };

    open_in_browser(&url)?;
    if let Ok(mut m) = viewers.live.lock() {
        m.insert(
            key,
            Running {
                child,
                url: url.clone(),
            },
        );
    }
    Ok(url)
}

/// An error that carries the viewer's own last words, which name the actual problem
/// (a folder that is not a MuMDIA result, a schema it does not read).
fn failure(what: &str, tail: &[String]) -> String {
    let said: Vec<&str> = tail
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if said.is_empty() {
        format!("{what}.")
    } else {
        format!("{what}:\n{}", said.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_address_is_read_from_the_serving_line_only() {
        let line = "mumdia-viewer: serving C:\\res\\run (run) at http://127.0.0.1:8050/AbC_12xyz/";
        assert_eq!(
            serving_url(line).as_deref(),
            Some("http://127.0.0.1:8050/AbC_12xyz/")
        );
        // The remote-server hint carries an address too, and must not be taken for it.
        assert_eq!(
            serving_url("remote server: ssh -L 8050:127.0.0.1:8050 <user>@<server>, then open http://127.0.0.1:8050/x/"),
            None
        );
        // Anything that is not a loopback http address is refused.
        assert_eq!(
            serving_url("mumdia-viewer: serving x (run) at http://0.0.0.0:8050/x/"),
            None
        );
        assert_eq!(
            serving_url("mumdia-viewer: serving x (run) at file:///etc/passwd"),
            None
        );
    }

    #[test]
    fn opening_without_the_environment_says_where_to_install_it() {
        if components::managed_python(Env::Visdia).is_file() {
            return; // installed on this machine; the refusal path is not reachable
        }
        let err = open(&Viewers::default(), ".", None).unwrap_err();
        assert!(err.contains("Setup"), "{err}");
    }

    /// The real thing: start visDIA from the managed environment on a MuMDIA results
    /// folder, fetch its page, and stop it. Needs the environment installed and
    /// `MUMDIA_TEST_VISDIA_DIR` naming a results folder.
    #[test]
    #[ignore]
    fn a_results_folder_is_served_and_stopped() {
        use std::io::{Read, Write};
        let Some(dir) = std::env::var_os("MUMDIA_TEST_VISDIA_DIR") else {
            return;
        };
        std::env::set_var("MUMDIA_VIEWER_NO_BROWSER", "1");
        let viewers = Viewers::default();
        let dir = dir.to_string_lossy().to_string();
        let url = open(&viewers, &dir, None).expect("the viewer starts");
        assert!(is_local_url(&url), "{url}");
        // A second open of the same folder reuses the running viewer.
        assert_eq!(open(&viewers, &dir, None).unwrap(), url);
        assert!(viewers.any_running());

        let rest = url.trim_start_matches("http://");
        let (host, path) = rest.split_once('/').unwrap();
        let mut s = std::net::TcpStream::connect(host).unwrap();
        write!(
            s,
            "GET /{path} HTTP/1.0
Host: {host}

"
        )
        .unwrap();
        let mut body = String::new();
        s.read_to_string(&mut body).unwrap();
        assert!(
            body.starts_with("HTTP/1.1 200") || body.starts_with("HTTP/1.0 200"),
            "{}",
            &body[..body.len().min(200)]
        );

        viewers.stop_all();
        assert!(!viewers.any_running());
        assert!(
            std::net::TcpStream::connect(host).is_err(),
            "the server is gone"
        );
    }

    #[test]
    fn a_failure_carries_what_the_viewer_said() {
        let msg = failure(
            "visDIA exited (exit code: 1)",
            &["".into(), "mumdia-viewer: x is not a MuMDIA result".into()],
        );
        assert!(
            msg.ends_with("mumdia-viewer: x is not a MuMDIA result"),
            "{msg}"
        );
        assert_eq!(failure("visDIA stopped", &[]), "visDIA stopped.");
    }
}
