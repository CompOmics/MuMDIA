//! Starting, watching and stopping one search.
//!
//! # Why the engine is a child process and not a linked library
//!
//! The engine is a library crate, so linking it and calling stages in-process looks
//! attractive: one binary, no path resolution, no version skew. It is the wrong
//! choice, for one decisive reason and two supporting ones.
//!
//! The engine installs no signal handler anywhere, so stopping a run is a kill, and
//! a Rust thread cannot be killed. Linked in-process there would be no Stop button
//! at all, only a window that ignores you for an hour. Supporting reasons: a stage
//! panic would take the whole application down with it rather than ending one run,
//! and rayon's global pool can only be built once per process, so `--threads` could
//! not change between runs.
//!
//! # How progress is observed
//!
//! Not by parsing the log. Every stage writes `<artifact>.report.json` beside its
//! output, carrying the producing stage, row count, elapsed time and per-stage
//! statistics. Polling the output directory for those files is a structured progress
//! feed that costs the engine nothing and works identically for a run this
//! application started and one it is merely looking at.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::engine;

/// How often the output directory is rescanned for stage reports.
const POLL: Duration = Duration::from_millis(700);
/// Log lines kept in memory. The pane shows the tail; the full log is on disk.
const LOG_TAIL: usize = 4000;

/// The results folders of the runs in flight, by canonical path, each with the id of
/// the run that owns it.
///
/// Two engines writing one artifact set interleave their output with no error from
/// either (docs/29 #5): a repeated Start launched a second engine into the same folder
/// and the interface kept only the latest run id. A folder is reserved here before the
/// engine is spawned and released when the run's end is published, so while a run is
/// active nothing else can be started into its folder.
static ACTIVE_OUT_DIRS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

/// The identity of a results folder for ownership purposes.
///
/// The canonical path folds the ways one directory can be spelled: relative against
/// absolute, `.` and `..` segments, symbolic links and, on Windows, case (`Out` and
/// `out` canonicalise to the on-disk spelling). The directory has to exist for that,
/// which is why `start` creates it first. When it cannot be canonicalised the path is
/// used as given, made absolute, so a reservation is still taken.
fn ownership_key(dir: &Path) -> String {
    let p = std::fs::canonicalize(dir).unwrap_or_else(|_| {
        if dir.is_absolute() {
            dir.to_path_buf()
        } else {
            std::env::current_dir()
                .map(|c| c.join(dir))
                .unwrap_or_else(|_| dir.to_path_buf())
        }
    });
    let s = p.to_string_lossy().into_owned();
    if cfg!(windows) {
        s.to_lowercase()
    } else {
        s
    }
}

/// Reserve `dir` for run `id`, or say which run already owns it.
///
/// Returns the key to release with. Held by the `Run`, released by `publish_exit`.
pub fn reserve_out_dir(dir: &Path, id: &str) -> Result<String, String> {
    let key = ownership_key(dir);
    let mut active = ACTIVE_OUT_DIRS.lock().unwrap_or_else(|e| e.into_inner());
    // Equal keys, and also one folder inside the other (docs/30 R8): an experiment writes
    // into its per-run subfolders and cleanup walks its whole folder, so a search into a
    // child of an active experiment, or an experiment over the parent of an active search,
    // is an overlapping writer. Component-wise, so `out` and `out2` stay independent.
    if let Some((held, owner)) = active
        .iter()
        .find(|(held, _)| **held == key || paths_nest(held, &key))
    {
        let relation = if *held == key {
            "is in use".to_string()
        } else if Path::new(&key).starts_with(Path::new(held)) {
            format!("is inside the results folder {held}, which is in use")
        } else {
            format!("contains the results folder {held}, which is in use")
        };
        return Err(format!(
            "the results folder {} {relation} by a search that is still running ({owner}). \
             Wait for it to finish or stop it, or choose another folder: two searches \
             writing one folder tree overwrite each other's results.",
            dir.display()
        ));
    }
    active.insert(key.clone(), id.to_string());
    Ok(key)
}

/// True when one path is an ancestor of the other, by path components.
fn paths_nest(a: &str, b: &str) -> bool {
    let (pa, pb) = (Path::new(a), Path::new(b));
    pa.starts_with(pb) || pb.starts_with(pa)
}

/// Give a reserved results folder back.
pub fn release_out_dir(key: &str) {
    let mut active = ACTIVE_OUT_DIRS.lock().unwrap_or_else(|e| e.into_inner());
    active.remove(key);
}

/// What the interface asks for when it starts a search.
#[derive(Deserialize, Debug, Clone)]
pub struct Request {
    /// One or more spectra files. Each may be an mzML or a vendor path.
    ///
    /// A single entry is an ordinary `run`. Several entries mean one of two very
    /// different things, chosen by `experiment`, and conflating them would be a
    /// scientific error rather than a UI simplification: separate searches share
    /// nothing, while an experiment pools the FDR across runs.
    pub mzml: Vec<String>,
    /// Pool the runs into one `run-experiment`: combined rescore, optional MBR
    /// transfer, per-run quant and cross-run LFQ.
    ///
    /// Ignored for a single file, where there is nothing to pool.
    #[serde(default)]
    pub experiment: bool,
    pub out_dir: String,
    /// FASTA mode. Mutually exclusive with the library pair.
    pub fasta: Option<String>,
    pub lib_precursors: Option<String>,
    pub lib_fragments: Option<String>,
    pub config: Option<String>,
    pub threads: Option<usize>,
}

/// One stage, as observed from the artifact reports it produced.
///
/// A stage can write several artifacts (`convert` writes four), so rows and elapsed
/// time are summed and the artifact count is kept, which is more honest than
/// reporting whichever file happened to be read last.
#[derive(Serialize, Clone, Debug, Default)]
pub struct Stage {
    pub name: String,
    pub rows: u64,
    pub elapsed_ms: u64,
    pub artifacts: usize,
}

/// Everything the results panel shows, taken from the scored table's own report.
///
/// Read from disk rather than recomputed: `psms_scored.parquet.report.json` records
/// the classifier that ACTUALLY ran alongside the one that was requested, and those
/// differ when a sidecar fails and `rescore.strict` is false.
#[derive(Serialize, Clone, Debug, Default)]
pub struct Results {
    pub classifier: String,
    pub classifier_requested: String,
    pub config_hash: String,
    pub peptides_1pct: u64,
    pub precursors_1pct: u64,
    pub protein_groups_1pct: u64,
    pub psms: u64,
    pub has_peptides_tsv: bool,
    pub has_proteins_tsv: bool,
    /// True when these counts came from a pooled `run-experiment`.
    ///
    /// They are then EXPERIMENT-WIDE, not per file, and the difference is not
    /// cosmetic: an experiment-wide rescore groups the q columns experiment-wide, so
    /// `peptide_q_value`, `precursor_q` and `pg_q_value` are written only to each
    /// group's single winning row across the whole experiment. A per-run count on
    /// those columns is diluted by roughly 1/n_runs and is meaningless; the correct
    /// per-file unit there is `run_psm_q`, from the split tables. The interface has to
    /// say which it is showing.
    pub experiment_wide: bool,
}

/// The whole observable state of a run. Serialised to the interface on every poll.
#[derive(Serialize, Clone, Debug)]
pub struct Snapshot {
    pub id: String,
    /// `starting` | `running` | `done` | `failed` | `cancelled`
    pub status: String,
    pub exit_code: Option<i32>,
    pub error: Option<String>,
    pub stages: Vec<Stage>,
    pub log: Vec<String>,
    pub out_dir: String,
    /// The exact command line, so it can be shown, copied and reproduced.
    pub command: String,
    pub started_unix_ms: u64,
    pub elapsed_ms: u64,
    pub results: Option<Results>,
    /// True in library-input mode, which skips digest, peptidoforms and predict-frag.
    pub library_mode: bool,
    /// True when this is a pooled `run-experiment` rather than a single `run`.
    ///
    /// The interface needs this to label the result counts: an experiment-wide
    /// rescore groups the q columns experiment-wide, so those counts are NOT per
    /// file, and the `peptides.tsv` / `proteins.tsv` at the experiment root are the
    /// experiment-wide report, not a per-run one.
    pub experiment: bool,
    /// Stop was requested. The status stays `running` until the engine has actually
    /// been reaped, and the interface shows "Stopping" meanwhile.
    pub cancel_requested: bool,
}

pub struct Run {
    pub snapshot: Mutex<Snapshot>,
    /// Process id of the engine. On Unix this is also its process-group id, because
    /// it is spawned into a new group.
    pid: Mutex<Option<u32>>,
    cancelled: AtomicBool,
    /// The results-folder reservation, until `publish_exit` releases it.
    reservation: Mutex<Option<String>>,
}

impl Run {
    fn set<F: FnOnce(&mut Snapshot)>(&self, f: F) {
        if let Ok(mut s) = self.snapshot.lock() {
            f(&mut s);
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        self.snapshot
            .lock()
            .map(|s| s.clone())
            .unwrap_or_else(|e| e.into_inner().clone())
    }

    /// Stop the run: record the intent, kill the process tree, then remove the rubble.
    ///
    /// Both halves of the kill matter. The engine spawns Python workers, so killing
    /// only the engine would orphan a process that may hold tens of gigabytes. And a
    /// hard kill skips destructors, so the atomic-write layer never removes its
    /// `.tmp-<pid>` files; without a sweep the next run starts in a dirty directory.
    ///
    /// What this does NOT do is write the terminal status. That is `publish_exit`'s,
    /// once the process has been reaped, and it reads the intent recorded here. The
    /// flag used to be written and never read while the status was written from here
    /// as well, racing the waiter: it could wake from the dying process first and
    /// publish `failed`, with the last log line as the "error", and this method then
    /// declined to replace a terminal status (docs/29 #14).
    pub fn cancel(&self) {
        // Inert once terminal: there is no process to kill, and the folder may already
        // belong to a later run (docs/30 R3). Cleanup is not done here at all any more:
        // it belongs to `publish_exit`, which runs after the reap and before the
        // reservation is released, so it can only ever touch this run's own files.
        if !self.is_active() {
            return;
        }
        self.cancelled.store(true, Ordering::SeqCst);
        self.set(|s| s.cancel_requested = true);
        // The pid lock is held across the kill, and `publish_exit` retires the pid under
        // the same lock before it sweeps and releases. A stop still in flight when the
        // engine is reaped therefore finishes before the folder changes hands, and a stop
        // that arrives after the reap finds no pid.
        let guard = self.pid.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pid) = *guard {
            kill_tree(pid);
        }
        drop(guard);
    }

    fn is_active(&self) -> bool {
        matches!(self.snapshot().status.as_str(), "running" | "starting")
    }

    /// Apply `f` only while the run is still active; returns whether it was.
    fn set_if_active<F: FnOnce(&mut Snapshot)>(&self, f: F) -> bool {
        if let Ok(mut s) = self.snapshot.lock() {
            if matches!(s.status.as_str(), "running" | "starting") {
                f(&mut s);
                return true;
            }
        }
        false
    }

    /// Publish the terminal state of the run from how its process ended.
    ///
    /// The one place a run becomes terminal, so the outcome does not depend on which
    /// thread ran first. Everything a finished run displays, its stages and results,
    /// is read from disk BEFORE the status stops being `running`; the other way round
    /// leaves a window in which the run says it is finished but has no stages, which an
    /// interface polling for completion reliably catches.
    ///
    /// A process that exited successfully is `done` even under a cancel request: the
    /// engine finished before the kill landed and its outputs are complete, and
    /// calling them cancelled would hide a finished result.
    fn publish_exit(&self, outcome: std::io::Result<std::process::ExitStatus>, out_dir: &Path) {
        // Retire the pid first. This waits for a stop that is still killing (it holds the
        // same lock), so nothing below overlaps a kill, and a later stop finds nothing to
        // signal (docs/30 R3).
        if let Ok(mut p) = self.pid.lock() {
            *p = None;
        }
        let cancelled = self.cancelled.load(Ordering::SeqCst);
        if cancelled {
            // The only sweep: after the reap, before the release, inside this run's
            // ownership of the folder.
            sweep_temp_files(out_dir);
        }
        let stages = scan_stages(out_dir);
        let results = read_results(out_dir);
        let status = terminal_status(cancelled, &outcome);
        // Released before the status is published, so a caller that sees the run end
        // can start another into the same folder without being refused.
        self.release_reservation();
        self.set(|s| {
            s.stages = stages;
            s.results = results;
            s.exit_code = outcome.as_ref().ok().and_then(|st| st.code());
            s.error = match (&outcome, status) {
                (_, "done" | "cancelled") => None,
                // The last stderr line is almost always the anyhow error chain, which
                // is the sentence worth showing.
                (Ok(_), _) => s.log.iter().rev().find(|l| !l.trim().is_empty()).cloned(),
                (Err(e), _) => Some(format!("could not wait for the engine: {e}")),
            };
            s.status = status.into();
        });
    }

    fn release_reservation(&self) {
        let key = self.reservation.lock().ok().and_then(|mut r| r.take());
        if let Some(key) = key {
            release_out_dir(&key);
        }
    }
}

/// The status a finished process publishes. Pure, so the interleavings of a Stop with
/// the engine's own exit can be pinned down in tests.
fn terminal_status(
    cancelled: bool,
    outcome: &std::io::Result<std::process::ExitStatus>,
) -> &'static str {
    match outcome {
        Ok(s) if s.success() => "done",
        _ if cancelled => "cancelled",
        _ => "failed",
    }
}

/// A run handle in its initial state. Shared by `start` and by the tests, which drive
/// `publish_exit` and `cancel` directly to pin down their interleavings.
fn new_run(id: &str, req: &Request, command: String, reservation: Option<String>) -> Arc<Run> {
    Arc::new(Run {
        snapshot: Mutex::new(Snapshot {
            id: id.to_string(),
            status: "starting".into(),
            exit_code: None,
            error: None,
            stages: Vec::new(),
            log: Vec::new(),
            out_dir: req.out_dir.clone(),
            command,
            started_unix_ms: now_ms(),
            elapsed_ms: 0,
            results: None,
            library_mode: req.lib_precursors.is_some(),
            experiment: req.experiment,
            cancel_requested: false,
        }),
        pid: Mutex::new(None),
        cancelled: AtomicBool::new(false),
        reservation: Mutex::new(reservation),
    })
}

/// Kill a process and everything it spawned.
///
/// Deliberately shelling out rather than calling the platform APIs directly: both
/// would need `unsafe`, and this is not on any hot path. `taskkill /T` walks the
/// tree at kill time and `kill` on a negative pid signals the whole group, so a
/// Python worker the application never knew about is included either way.
pub fn kill_tree(pid: u32) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .creation_flags(CREATE_NO_WINDOW)
            .status();
    }
    #[cfg(unix)]
    {
        // Signalling a process group is how the engine's Python workers get
        // included, and it is also how you kill everything you are running inside
        // if the group turns out to be your own. That is not hypothetical: an
        // earlier version of this function, exercised by its own test, terminated a
        // CI runner.
        //
        // So the group is verified before it is signalled. `Command::process_group`
        // is asked for at spawn time, but if it did not take effect the child sits
        // in OUR group, and a group kill would take down the application, the shell
        // that started it, and on a shared machine whatever else shares that group.
        // When the guard trips the child is still killed, just individually.
        let target_pgid = pgid_of(pid);
        let own_pgid = pgid_of(std::process::id());
        let group_is_safe = match (target_pgid, own_pgid) {
            // A group of its own: signal the group, which is the whole point.
            (Some(t), Some(o)) => t != o && t == pid,
            // Unknown either way: do not guess with SIGKILL.
            _ => false,
        };
        let target = if group_is_safe {
            format!("-{pid}")
        } else {
            pid.to_string()
        };
        // TERM first so the engine can unwind and remove its own temp files, KILL
        // shortly after for anything that ignored it.
        let _ = std::process::Command::new("kill")
            .args(["-TERM", &target])
            .status();
        std::thread::sleep(Duration::from_millis(1500));
        let _ = std::process::Command::new("kill")
            .args(["-KILL", &target])
            .status();
    }
}

/// The process-group id of `pid`, via `ps`, or `None` if it cannot be determined.
///
/// Shelling out rather than calling `getpgid`, which would need `unsafe` in a crate
/// that has none. This runs twice per cancellation, not in a loop.
#[cfg(unix)]
fn pgid_of(pid: u32) -> Option<u32> {
    let out = std::process::Command::new("ps")
        .args(["-o", "pgid=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u32>()
        .ok()
}

/// Remove `*.tmp-<pid>` files left by a killed run, recursively.
fn sweep_temp_files(dir: &Path) {
    for f in walk(dir) {
        if f.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains(".tmp-"))
        {
            let _ = std::fs::remove_file(&f);
        }
    }
}

/// Every file under `dir`, recursively. Small hand-rolled walk to avoid a dependency
/// for one function; output directories are shallow.
fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            match e.file_type() {
                Ok(t) if t.is_dir() => stack.push(p),
                Ok(t) if t.is_file() => out.push(p),
                _ => {}
            }
        }
    }
    out
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build the argument list, and reject the input combinations the engine would
/// reject anyway — here, where the message can point at a field.
fn argv(req: &Request) -> Result<Vec<String>, String> {
    let lib = req.lib_precursors.is_some() || req.lib_fragments.is_some();
    if lib && req.fasta.is_some() {
        return Err("choose either a FASTA or a spectral library, not both".into());
    }
    if !lib && req.fasta.is_none() {
        return Err("select a FASTA file or a spectral library".into());
    }
    if lib && (req.lib_precursors.is_none() || req.lib_fragments.is_none()) {
        return Err("a spectral library needs both the precursor and the fragment table".into());
    }
    if req.mzml.is_empty() || req.mzml.iter().all(|m| m.trim().is_empty()) {
        return Err("select at least one spectra file".into());
    }
    if req.experiment && req.mzml.len() < 2 {
        return Err("a pooled experiment needs at least two files".into());
    }
    // The engine's `run` takes a single `--mzml` (main.rs: `mzml: String`, no append
    // action), so emitting several would produce "the argument '--mzml <MZML>' cannot
    // be used multiple times" -- an error the user cannot act on. Separate searches
    // over several files are N single-file requests, which is the interface's job;
    // this refuses the invalid state rather than letting clap report it.
    if !req.experiment && req.mzml.len() > 1 {
        return Err(format!(
            "{} files were given for a single search. Either search them separately,              which runs one search per file, or choose a pooled experiment.",
            req.mzml.len()
        ));
    }
    // Two runs pointed at the same file would collide: `run-experiment` derives each
    // run's subdirectory from its name, and the engine's own duplicate-name guard
    // catches the name collision but not two entries for one path.
    {
        let mut seen = std::collections::HashSet::new();
        for m in &req.mzml {
            if !seen.insert(m.as_str()) {
                return Err(format!("{m} was selected more than once"));
            }
        }
    }
    if req.out_dir.trim().is_empty() {
        return Err("choose a folder for the results".into());
    }

    // `req.experiment` alone: the check above already rejected an experiment with
    // fewer than two files, so re-testing the length here would be a second, quieter
    // rule that disagreed with the first. It did, and a test caught it.
    let mut a: Vec<String> = vec![if req.experiment {
        "run-experiment"
    } else {
        "run"
    }
    .into()];
    for m in &req.mzml {
        a.push("--mzml".into());
        a.push(m.clone());
    }
    a.push("--out-dir".into());
    a.push(req.out_dir.clone());
    if let Some(f) = &req.fasta {
        a.push("--fasta".into());
        a.push(f.clone());
    }
    if let (Some(p), Some(g)) = (&req.lib_precursors, &req.lib_fragments) {
        a.push("--lib-precursors".into());
        a.push(p.clone());
        a.push("--lib-fragments".into());
        a.push(g.clone());
    }
    if let Some(c) = &req.config {
        if !c.trim().is_empty() {
            a.push("--config".into());
            a.push(c.clone());
        }
    }
    if let Some(t) = req.threads {
        if t > 0 {
            a.push("--threads".into());
            a.push(t.to_string());
        }
    }
    Ok(a)
}

/// Quote an argument for display only. This string is shown and copied, never
/// executed, so it just has to be pasteable.
fn quote(s: &str) -> String {
    if s.contains(' ') {
        format!("\"{s}\"")
    } else {
        s.to_string()
    }
}

/// Does this request describe a search that needs no Python at all?
///
/// That is the configuration the application refuses to run. The predicate is
/// deliberately narrow. The measured gap that motivates the refusal is about 1,213
/// report rows against about 10,300 on the same file, and that is the fully native
/// FASTA path against the imported-library workflow -- but the rescorer is not what
/// separates them. On an imported library with retention-time modelling in place,
/// `native_tda` measured 10,847 against `nn_torch`'s 10,914, a difference of 0.6%.
/// Refusing every configuration that mentions `native_tda` would block one that is
/// within noise of the best.
///
/// So the rule is "needs no sidecar at all", which is exactly the zero-component
/// path the 1,213 figure describes.
///
/// The authority for this is the engine, not a list kept here: `mumdia doctor
/// --json` reports `required` per role from the configuration it is given, and if
/// every role is unrequired then the run is the minimal path.
pub fn needs_no_sidecar(engine: &Path, config: Option<&str>) -> Result<bool, String> {
    let mut cmd = engine::command(engine);
    // The same environment the run itself gets. Asking `doctor` without it would
    // answer for a different engine than the one that will run.
    crate::components::stamp_env(&mut cmd);
    cmd.arg("doctor").arg("--json");
    if let Some(c) = config {
        if !c.trim().is_empty() {
            cmd.arg("--config").arg(c);
        }
    }
    let out = cmd
        .output()
        .map_err(|e| format!("could not ask the engine what this configuration needs: {e}"))?;
    // `doctor` exits non-zero when the configuration cannot run, which is exactly
    // the case where an interpreter is required and missing. The report on stdout is
    // still valid and still says which roles are required, so the exit status is
    // deliberately not checked here.
    let text = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("could not read the engine's configuration report: {e}"))?;
    let roles = v
        .get("roles")
        .and_then(|r| r.as_array())
        .ok_or_else(|| "the engine's configuration report has no roles section".to_string())?;
    Ok(!roles
        .iter()
        .any(|r| r.get("required").and_then(|b| b.as_bool()).unwrap_or(false)))
}

/// Start a search. Returns immediately with a handle; progress arrives by polling.
pub fn start(id: String, req: Request) -> Result<Arc<Run>, String> {
    let args = argv(&req)?;
    let (exe, _source) = engine::resolve()?;

    std::fs::create_dir_all(&req.out_dir)
        .map_err(|e| format!("cannot create the results folder {}: {e}", req.out_dir))?;
    // Taken before the engine exists, so a second request for this folder is refused
    // while this one is still being spawned, not only once it runs.
    let reservation = reserve_out_dir(Path::new(&req.out_dir), &id)?;

    let display = format!(
        "{} {}",
        quote(&exe.display().to_string()),
        args.iter().map(|a| quote(a)).collect::<Vec<_>>().join(" ")
    );

    let run = new_run(&id, &req, display, Some(reservation));

    let mut cmd = engine::command(&exe);
    // Without this the managed Python environment and the managed .raw converter
    // are invisible to the engine; see `components::stamp_env`.
    crate::components::stamp_env(&mut cmd);
    cmd.args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null());

    // A new process group, so cancelling can signal the engine AND the Python
    // workers it spawns. Windows gets the same effect from `taskkill /T`.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            run.release_reservation();
            return Err(format!("could not start {}: {e}", exe.display()));
        }
    };

    let pid = child.id();
    if let Ok(mut p) = run.pid.lock() {
        *p = Some(pid);
    }
    run.set(|s| s.status = "running".into());

    // stderr carries the log; stdout carries result summaries. Both are shown.
    for (stream, tag) in [
        (
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
            "",
        ),
        (
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
            "",
        ),
    ] {
        let Some(stream) = stream else { continue };
        let run = Arc::clone(&run);
        let _ = tag;
        std::thread::spawn(move || {
            let reader = BufReader::new(stream);
            for line in reader.lines().map_while(Result::ok) {
                run.set(|s| {
                    s.log.push(line);
                    if s.log.len() > LOG_TAIL {
                        let drop = s.log.len() - LOG_TAIL;
                        s.log.drain(0..drop);
                    }
                });
            }
        });
    }

    // Progress: rescan the output directory until the run stops.
    {
        let run = Arc::clone(&run);
        let out_dir = PathBuf::from(&req.out_dir);
        std::thread::spawn(move || {
            let started = Instant::now();
            loop {
                let stages = scan_stages(&out_dir);
                // Written only while the run is still active, under the snapshot lock:
                // once `publish_exit` has published, a scan that was in flight must not
                // replace the finished snapshot's stages with whatever the folder holds
                // now, which may already be a later run's contents (docs/30).
                let still_active = run.set_if_active(|s| {
                    s.stages = stages;
                    s.elapsed_ms = started.elapsed().as_millis() as u64;
                });
                if !still_active {
                    // The final scan belongs to the waiter, not here: it has to happen
                    // BEFORE the status becomes terminal, or a caller that polls until
                    // the run is finished can read a snapshot whose stages and results
                    // have not been filled in yet.
                    break;
                }
                std::thread::sleep(POLL);
            }
        });
    }

    // Reap the child, then publish the terminal state in one step (`publish_exit`).
    {
        let run = Arc::clone(&run);
        let out_dir = PathBuf::from(&req.out_dir);
        std::thread::spawn(move || {
            let outcome = child.wait();
            run.publish_exit(outcome, &out_dir);
        });
    }

    Ok(run)
}

/// One past run, reconstructed from what it left on disk.
///
/// There is no history database. A finished run already carries a complete record
/// in its own output folder: `manifest.json` for the engine version, the commit and
/// the hashed inputs, and `psms_scored.parquet.report.json` for the counts and the
/// classifier that actually ran. Reading those back is both less code and more
/// honest than a separate index, which could disagree with the folder it describes.
#[derive(Serialize, Clone, Debug)]
pub struct HistoryEntry {
    pub out_dir: String,
    pub name: String,
    pub finished_unix_ms: u64,
    pub results: Option<Results>,
    /// Present when the run wrote a manifest, which is every completed run.
    pub engine_version: Option<String>,
}

/// Read one output directory as a history entry, or `None` if it is not one.
pub fn history_entry(dir: &Path) -> Option<HistoryEntry> {
    let scored = dir.join("psms_scored.parquet.report.json");
    let manifest = dir.join("manifest.json");
    if !scored.is_file() && !manifest.is_file() {
        return None;
    }
    let finished = scored
        .metadata()
        .or_else(|_| manifest.metadata())
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let engine_version = std::fs::read_to_string(&manifest)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| {
            v.get("mumdia_version")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string())
        });
    Some(HistoryEntry {
        name: dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("run")
            .to_string(),
        out_dir: dir.display().to_string(),
        finished_unix_ms: finished,
        results: read_results(dir),
        engine_version,
    })
}

/// Fold every `*.report.json` under `dir` into one row per producing stage.
fn scan_stages(dir: &Path) -> Vec<Stage> {
    let mut by_stage: BTreeMap<String, Stage> = BTreeMap::new();
    for f in walk(dir) {
        if !f
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".report.json"))
        {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&f) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let Some(stage) = v.get("stage").and_then(|s| s.as_str()) else {
            continue;
        };
        let e = by_stage.entry(stage.to_string()).or_insert_with(|| Stage {
            name: stage.to_string(),
            ..Default::default()
        });
        e.rows += v.get("rows").and_then(|r| r.as_u64()).unwrap_or(0);
        e.elapsed_ms += v.get("elapsed_ms").and_then(|r| r.as_u64()).unwrap_or(0);
        e.artifacts += 1;
    }
    by_stage.into_values().collect()
}

/// Read the results panel out of the scored table's report.
fn read_results(dir: &Path) -> Option<Results> {
    // A single `run` writes `psms_scored.parquet`; a pooled `run-experiment` writes
    // `scored_combined.parquet` (its counts) plus an experiment-wide `peptides.tsv`
    // and `proteins.tsv` at the root, one quantity column per run. Reading only the
    // first name left the results screen blank after every experiment.
    let (path, experiment_wide) = {
        let single = dir.join("psms_scored.parquet.report.json");
        if single.is_file() {
            (single, false)
        } else {
            (dir.join("scored_combined.parquet.report.json"), true)
        }
    };
    let text = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let params = v.get("params");
    let stats = v.get("stats");
    let s = |o: Option<&serde_json::Value>, k: &str| -> String {
        o.and_then(|p| p.get(k))
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string()
    };
    let n = |o: Option<&serde_json::Value>, k: &str| -> u64 {
        o.and_then(|p| p.get(k))
            .and_then(|x| x.as_u64())
            .unwrap_or(0)
    };
    Some(Results {
        classifier: s(stats, "classifier"),
        classifier_requested: s(params, "classifier_requested"),
        config_hash: s(params, "config_hash"),
        peptides_1pct: n(stats, "target_peptides_at_1pct"),
        precursors_1pct: n(stats, "target_precursors_at_1pct"),
        protein_groups_1pct: n(stats, "target_protein_groups_at_1pct"),
        psms: n(stats, "psms"),
        has_peptides_tsv: dir.join("peptides.tsv").is_file(),
        has_proteins_tsv: dir.join("proteins.tsv").is_file(),
        experiment_wide,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> Request {
        Request {
            mzml: vec!["a.mzML".into()],
            experiment: false,
            out_dir: "out".into(),
            fasta: None,
            lib_precursors: None,
            lib_fragments: None,
            config: None,
            threads: None,
        }
    }

    fn exit_status(code: i32) -> std::process::ExitStatus {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            std::process::ExitStatus::from_raw(code << 8)
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::ExitStatusExt;
            std::process::ExitStatus::from_raw(code as u32)
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mumdia_run_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A run in the `running` state with no process behind it, so `publish_exit` and
    /// `cancel` can be interleaved by hand.
    fn running(name: &str) -> (Arc<Run>, PathBuf) {
        let dir = scratch(name);
        let mut r = req();
        r.out_dir = dir.display().to_string();
        let run = new_run(&format!("run-{name}"), &r, "mumdia run ...".into(), None);
        run.set(|s| s.status = "running".into());
        (run, dir)
    }

    #[test]
    fn a_stop_that_lands_before_the_waiter_wakes_is_published_as_cancelled() {
        // The interleaving of docs/29 #14: Stop was pressed and the kill issued, and
        // the waiter wakes from the dying process before `cancel` could have written
        // anything. The waiter used to publish `failed` here, with the last log line
        // as the error, and `cancel` then left that terminal status alone.
        let (run, dir) = running("cancel_first");
        run.set(|s| s.log.push("thread 'main' panicked".into()));
        run.cancelled.store(true, Ordering::SeqCst);
        run.publish_exit(Ok(exit_status(1)), &dir);
        let s = run.snapshot();
        assert_eq!(s.status, "cancelled");
        assert_eq!(s.error, None, "a stopped run has no error to show");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_run_that_failed_on_its_own_stays_failed_when_stop_arrives_late() {
        let (run, dir) = running("failed_first");
        run.set(|s| s.log.push("Error: no such file".into()));
        run.publish_exit(Ok(exit_status(1)), &dir);
        run.cancel();
        let s = run.snapshot();
        assert_eq!(s.status, "failed");
        assert_eq!(s.error.as_deref(), Some("Error: no such file"));
        // A stop that arrives after the end is inert and records nothing (docs/30 R3).
        assert!(!s.cancel_requested);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_run_that_finished_before_the_kill_landed_is_done() {
        let (run, dir) = running("done_under_cancel");
        run.cancelled.store(true, Ordering::SeqCst);
        run.publish_exit(Ok(exit_status(0)), &dir);
        assert_eq!(run.snapshot().status, "done");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_wait_error_is_failed_unless_a_stop_was_requested() {
        let (run, dir) = running("wait_error");
        run.publish_exit(Err(std::io::Error::other("gone")), &dir);
        let s = run.snapshot();
        assert_eq!(s.status, "failed");
        assert!(
            s.error.as_deref().unwrap_or("").contains("could not wait"),
            "{:?}",
            s.error
        );
        let _ = std::fs::remove_dir_all(&dir);

        let (run, dir) = running("wait_error_cancelled");
        run.cancelled.store(true, Ordering::SeqCst);
        run.publish_exit(Err(std::io::Error::other("gone")), &dir);
        assert_eq!(run.snapshot().status, "cancelled");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cancel_records_the_intent_without_publishing_a_terminal_status() {
        // Until the engine is reaped the run is still running, whatever the button
        // says; `cancel_requested` is what the interface shows "Stopping" from.
        let (run, dir) = running("intent_only");
        run.cancel();
        let s = run.snapshot();
        assert_eq!(s.status, "running");
        assert!(s.cancel_requested);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_late_stop_after_the_run_ended_leaves_the_folder_alone() {
        // docs/30 R3: A finished and released its folder; B took it and is writing. A
        // stop delivered to A must neither sweep B's temporary file nor change A's state.
        let (run, dir) = running("late_stop");
        let key = reserve_out_dir(&dir, "run-A").unwrap();
        *run.reservation.lock().unwrap() = Some(key);
        run.publish_exit(Ok(exit_status(1)), &dir);
        assert_eq!(run.snapshot().status, "failed");
        let key_b = reserve_out_dir(&dir, "run-B").expect("A released its folder");
        let b_file = dir.join("new.parquet.tmp-999-1");
        std::fs::write(&b_file, b"B's partial write").unwrap();
        run.cancel();
        assert!(
            b_file.is_file(),
            "a late stop must not sweep another run's files"
        );
        let s = run.snapshot();
        assert_eq!(s.status, "failed");
        assert!(!s.cancel_requested, "a terminal run records no stop");
        release_out_dir(&key_b);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stop_still_in_flight_finishes_before_the_folder_is_released() {
        // docs/30 R3, the concurrent route: the engine is reaped while the stop thread is
        // still inside the kill. Publication must wait for the kill to finish, so the
        // reservation cannot be released, and taken by a new run, while the stop is
        // still active in that folder. The kill is simulated by holding the pid lock.
        let (run, dir) = running("inflight_stop");
        let key = reserve_out_dir(&dir, "run-A").unwrap();
        *run.reservation.lock().unwrap() = Some(key);
        run.cancelled.store(true, Ordering::SeqCst);
        let killing = run.pid.lock().unwrap();
        let (r2, d2) = (Arc::clone(&run), dir.clone());
        let waiter = std::thread::spawn(move || r2.publish_exit(Ok(exit_status(1)), &d2));
        std::thread::sleep(Duration::from_millis(300));
        assert!(
            reserve_out_dir(&dir, "run-B").is_err(),
            "the folder must stay reserved while the stop is in flight"
        );
        assert_eq!(
            run.snapshot().status,
            "running",
            "nothing is published mid-kill"
        );
        drop(killing);
        waiter.join().unwrap();
        assert_eq!(run.snapshot().status, "cancelled");
        let k = reserve_out_dir(&dir, "run-B").expect("released once the stop completed");
        release_out_dir(&k);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn overlapping_result_folders_are_refused_in_both_orders_but_siblings_are_not() {
        // docs/30 R8: an experiment owns its per-run subfolders and its cleanup walks the
        // whole tree, so a parent and a child are one writer.
        let parent = scratch("nest");
        let child = parent.join("run1");
        let sibling = std::env::temp_dir().join(format!("mumdia_run_{}_nest2", std::process::id()));
        std::fs::create_dir_all(&child).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();
        let k = reserve_out_dir(&parent, "run-1").unwrap();
        let e = reserve_out_dir(&child, "run-2").unwrap_err();
        assert!(e.contains("is inside") && e.contains("run-1"), "{e}");
        let ks = reserve_out_dir(&sibling, "run-3").expect("a sibling is independent");
        release_out_dir(&k);
        release_out_dir(&ks);
        let kc = reserve_out_dir(&child, "run-2").unwrap();
        let e = reserve_out_dir(&parent, "run-1").unwrap_err();
        assert!(e.contains("contains") && e.contains("run-2"), "{e}");
        release_out_dir(&kc);
        let _ = std::fs::remove_dir_all(&parent);
        let _ = std::fs::remove_dir_all(&sibling);
    }

    #[test]
    fn a_results_folder_owned_by_an_active_run_is_refused_to_a_second() {
        let dir = scratch("owned");
        let key = reserve_out_dir(&dir, "run-1").unwrap();
        let e = reserve_out_dir(&dir, "run-2").unwrap_err();
        assert!(e.contains("run-1") && e.contains("still running"), "{e}");
        // Another spelling of the same folder is the same folder.
        let e2 = reserve_out_dir(&dir.join("."), "run-3").unwrap_err();
        assert!(e2.contains("run-1"), "{e2}");
        release_out_dir(&key);
        let key2 = reserve_out_dir(&dir, "run-2").expect("free again once released");
        release_out_dir(&key2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_case_alias_of_an_active_folder_is_refused_where_the_filesystem_folds_case() {
        // `..._Case` and `..._case` are one directory on NTFS and APFS and two on ext4.
        // The rule follows the filesystem, which is what the canonical path reports:
        // the same directory is refused, a different one is free.
        let dir = scratch("Case");
        let alias = std::env::temp_dir().join(format!("mumdia_run_{}_case", std::process::id()));
        let key = reserve_out_dir(&dir, "run-1").unwrap();
        let same = std::fs::canonicalize(&alias).ok() == std::fs::canonicalize(&dir).ok();
        let second = reserve_out_dir(&alias, "run-2");
        if same {
            let e = second.unwrap_err();
            assert!(e.contains("run-1"), "{e}");
        } else {
            release_out_dir(&second.expect("a different directory is free"));
        }
        release_out_dir(&key);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn publishing_the_end_of_a_run_releases_its_folder() {
        let dir = scratch("release");
        let key = reserve_out_dir(&dir, "run-9").unwrap();
        let mut r = req();
        r.out_dir = dir.display().to_string();
        let run = new_run("run-9", &r, String::new(), Some(key));
        run.set(|s| s.status = "running".into());
        assert!(reserve_out_dir(&dir, "run-10").is_err());
        run.publish_exit(Ok(exit_status(0)), &dir);
        let k = reserve_out_dir(&dir, "run-10").expect("released when the run ended");
        release_out_dir(&k);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fasta_and_library_together_is_rejected() {
        let mut r = req();
        r.fasta = Some("p.fasta".into());
        r.lib_precursors = Some("p.parquet".into());
        r.lib_fragments = Some("f.parquet".into());
        let e = argv(&r).unwrap_err();
        assert!(e.contains("not both"), "{e}");
    }

    #[test]
    fn a_search_space_is_required() {
        let e = argv(&req()).unwrap_err();
        assert!(e.contains("FASTA") && e.contains("library"), "{e}");
    }

    #[test]
    fn half_a_library_is_rejected() {
        // The engine would reject this too, but only after starting up. Catching it
        // here lets the message name the missing field.
        let mut r = req();
        r.lib_precursors = Some("p.parquet".into());
        let e = argv(&r).unwrap_err();
        assert!(e.contains("both"), "{e}");
    }

    #[test]
    fn fasta_mode_builds_the_documented_invocation() {
        let mut r = req();
        r.fasta = Some("p.fasta".into());
        r.threads = Some(8);
        r.config = Some("c.json".into());
        assert_eq!(
            argv(&r).unwrap(),
            vec![
                "run",
                "--mzml",
                "a.mzML",
                "--out-dir",
                "out",
                "--fasta",
                "p.fasta",
                "--config",
                "c.json",
                "--threads",
                "8",
            ]
        );
    }

    #[test]
    fn library_mode_passes_both_tables_and_no_fasta() {
        let mut r = req();
        r.lib_precursors = Some("p.parquet".into());
        r.lib_fragments = Some("f.parquet".into());
        let a = argv(&r).unwrap();
        assert!(a.contains(&"--lib-precursors".to_string()));
        assert!(a.contains(&"--lib-fragments".to_string()));
        assert!(!a.contains(&"--fasta".to_string()));
    }

    #[test]
    fn an_empty_config_is_omitted_rather_than_passed_as_an_empty_path() {
        let mut r = req();
        r.fasta = Some("p.fasta".into());
        r.config = Some("   ".into());
        assert!(!argv(&r).unwrap().contains(&"--config".to_string()));
    }

    /// Fold real artifact reports, written by a real run, into stage rows.
    #[test]
    fn stages_are_folded_from_real_artifact_reports() {
        let dir =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/run_out");
        if !dir.is_dir() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let stages = scan_stages(&dir);
        let names: Vec<&str> = stages.iter().map(|s| s.name.as_str()).collect();
        for expected in [
            "convert",
            "digest",
            "peptidoforms",
            "predict-frag",
            "search-seed",
            "rt-im-train",
            "extract",
            "features",
            "compete",
            "rescore",
            "quant",
        ] {
            assert!(names.contains(&expected), "missing {expected} in {names:?}");
        }
        // `convert` writes four artifacts under spectra/; the walk must recurse and
        // the four must fold into one row.
        let convert = stages.iter().find(|s| s.name == "convert").unwrap();
        assert_eq!(convert.artifacts, 4);
        assert_eq!(convert.rows, 8 + 480 + 60 + 480);
    }

    #[test]
    fn results_come_from_the_scored_report_not_from_the_request() {
        let dir =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/run_out");
        if !dir.is_dir() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let r = read_results(&dir).expect("scored report should parse");
        assert_eq!(r.classifier, "native_tda");
        assert_eq!(r.peptides_1pct, 151);
        assert_eq!(r.psms, 152);
        assert!(r.has_peptides_tsv && r.has_proteins_tsv);
    }

    /// The engine spawns Python workers, so a kill that reaches only the process we
    /// launched would leave one behind holding tens of gigabytes. This spawns a
    /// parent that spawns its own child and checks the kill lands.
    ///
    /// The end-to-end cancel test cannot cover this: the fixture search finishes in
    /// under a second, faster than a stop can be issued.
    /// The guard that stops a cancellation killing the application itself.
    ///
    /// This is the check whose absence terminated a CI runner: without it,
    /// `kill_tree` would signal whatever group the child happened to be in, and if
    /// that is our own group the kill reaches the process doing the killing.
    #[cfg(unix)]
    #[test]
    fn a_process_in_our_own_group_is_never_group_killed() {
        // A child spawned WITHOUT `process_group` inherits ours, which is exactly
        // the situation the guard exists for.
        let mut cmd = std::process::Command::new("sh");
        cmd.args(["-c", "sleep 30"])
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let mut child = cmd.spawn().expect("could not spawn the test process");
        let own = pgid_of(std::process::id());
        let theirs = pgid_of(child.id());
        assert_eq!(
            own, theirs,
            "a child spawned without process_group should share our group"
        );
        // The decision the guard makes, without acting on it: this must NOT be a
        // group kill.
        let group_is_safe = match (theirs, own) {
            (Some(t), Some(o)) => t != o && t == child.id(),
            _ => false,
        };
        assert!(
            !group_is_safe,
            "killing this group would kill the test process itself"
        );
        let _ = child.kill();
        let _ = child.wait();
    }

    /// Opt-in with `MUMDIA_TEST_KILL=1`, and never in shared CI.
    ///
    /// This test terminated a GitHub runner twice. The first time is explained: the
    /// group kill had no guard, so it could signal the runner's own process group.
    /// The second time it did it again WITH the guard, which should have made a
    /// group kill possible only when the child is verifiably in a group of its own,
    /// and I cannot account for that. Two possibilities remain open: the guard's
    /// reasoning is wrong in a way I have not seen, or something about the runner's
    /// process arrangement makes any group signal fatal there.
    ///
    /// What follows from not knowing is the gating, not a guess. A test that can
    /// take down the machine it runs on does not belong in a shared pipeline while
    /// its failure mode is unexplained, and the thing it checks is verified on
    /// Windows, where `taskkill /T` addresses a process tree rather than a group.
    ///
    /// The consequence to be honest about: the Unix group-kill path in `kill_tree`
    /// is exercised by nothing automated. `a_process_in_our_own_group_is_never_group_killed`
    /// covers the guard's decision without acting on it, which is the part that can
    /// be tested safely.
    #[test]
    fn kill_tree_terminates_the_process_it_is_given() {
        if std::env::var("MUMDIA_TEST_KILL").ok().as_deref() != Some("1") {
            eprintln!("MUMDIA_TEST_KILL=1 not set; skipping (see the comment above)");
            return;
        }
        let mut cmd = if cfg!(windows) {
            let mut c = std::process::Command::new("cmd");
            c.args(["/C", "ping -n 30 127.0.0.1"]);
            c
        } else {
            let mut c = std::process::Command::new("sh");
            c.args(["-c", "sleep 30 & wait"]);
            c
        };
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let mut child = cmd.spawn().expect("could not spawn the test process");
        std::thread::sleep(Duration::from_millis(400));
        assert!(
            matches!(child.try_wait(), Ok(None)),
            "the test process exited on its own; the test proves nothing"
        );

        kill_tree(child.id());

        let start = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                _ if start.elapsed() > Duration::from_secs(15) => {
                    let _ = child.kill();
                    panic!("the process survived kill_tree");
                }
                _ => std::thread::sleep(Duration::from_millis(100)),
            }
        }
    }

    #[test]
    fn the_temp_sweep_removes_only_temp_files() {
        let dir = std::env::temp_dir().join(format!("mumdia_sweep_{}", std::process::id()));
        let nested = dir.join("spectra");
        std::fs::create_dir_all(&nested).unwrap();
        let keep = dir.join("peptides.tsv");
        let kill = dir.join("psms_scored.parquet.tmp-12345");
        let kill_nested = nested.join("spectra_ms2.parquet.tmp-9");
        for f in [&keep, &kill, &kill_nested] {
            std::fs::write(f, b"x").unwrap();
        }
        sweep_temp_files(&dir);
        assert!(keep.is_file(), "a real output must survive the sweep");
        assert!(!kill.is_file(), "the temp file must go");
        assert!(!kill_nested.is_file(), "the sweep must recurse");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod history_tests {
    use super::*;

    #[test]
    fn a_finished_run_folder_reads_back_as_history() {
        // The same real artifact reports the stage test uses: a history entry must
        // come from the folder, not from anything the application remembered.
        let dir =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/run_out");
        if !dir.is_dir() {
            eprintln!("fixture missing, skipping");
            return;
        }
        let e = history_entry(&dir).expect("a completed run folder is a history entry");
        assert_eq!(e.name, "run_out");
        let r = e.results.expect("results come from the scored report");
        assert_eq!(r.classifier, "native_tda");
        assert_eq!(r.peptides_1pct, 151);
        assert!(
            e.engine_version.is_some(),
            "the manifest names the engine version"
        );
        assert!(e.finished_unix_ms > 0);
    }

    #[test]
    fn a_folder_that_is_not_a_run_is_not_history() {
        // A user picks output folders by hand, so the list will contain directories
        // that never held a search. They must drop out rather than appear empty.
        let dir = std::env::temp_dir().join(format!("mumdia_not_a_run_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.txt"), b"hello").unwrap();
        assert!(history_entry(&dir).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Multi-file semantics, in their own module because the two meanings of "several
/// files" are the thing most likely to be conflated by a later change.
#[cfg(test)]
mod multifile_tests {
    use super::*;

    fn req() -> Request {
        Request {
            mzml: vec!["a.mzML".into()],
            experiment: false,
            out_dir: "out".into(),
            fasta: Some("f.fasta".into()),
            lib_precursors: None,
            lib_fragments: None,
            config: None,
            threads: None,
        }
    }

    #[test]
    fn a_one_file_experiment_is_refused_rather_than_quietly_demoted() {
        // Refused, not silently turned into a single run. Pooled and separate are
        // scientifically different -- that difference is the whole reason the
        // interface offers a choice -- so a user who asked for an experiment and got
        // one plain run must be told, not left to infer it from the output tree.
        let mut r = req();
        r.experiment = true;
        let e = argv(&r).unwrap_err();
        assert!(e.contains("at least two files"), "{e}");

        // Without the flag the same single file is an ordinary run.
        r.experiment = false;
        let a = argv(&r).unwrap();
        assert_eq!(a[0], "run");
        assert_eq!(a.iter().filter(|x| *x == "--mzml").count(), 1);
    }

    #[test]
    fn several_files_are_separate_runs_unless_pooling_is_asked_for() {
        // The distinction the interface must not blur. Separate searches share
        // nothing; an experiment pools the FDR across runs. Batch mode is N calls
        // with one file each, so a multi-file request WITHOUT the flag is a mistake
        // rather than an implicit experiment.
        let mut r = req();
        r.mzml = vec!["a.mzML".into(), "b.mzML".into()];

        r.experiment = true;
        let pooled = argv(&r).unwrap();
        assert_eq!(pooled[0], "run-experiment");
        assert_eq!(pooled.iter().filter(|x| *x == "--mzml").count(), 2);

        // Without the flag, several files are REFUSED rather than emitted as one
        // `run`. The engine's `run` takes a single `--mzml` (`mzml: String`, no append
        // action), so N of them would die on "cannot be used multiple times" -- an
        // error a user cannot act on. Separate searches are N single-file requests,
        // which is the interface's job, so a multi-file non-experiment request is a
        // mistake and is named as one.
        r.experiment = false;
        let e = argv(&r).unwrap_err();
        assert!(e.contains("search them separately"), "{e}");
    }

    #[test]
    fn a_pooled_experiment_needs_at_least_two_files() {
        let mut r = req();
        r.experiment = true;
        // Zero files is an error whatever the flag says.
        r.mzml = Vec::new();
        let e = argv(&r).unwrap_err();
        assert!(e.contains("at least one"), "{e}");
    }

    #[test]
    fn the_same_file_twice_is_refused() {
        // `run-experiment` derives each run's output subdirectory from its name, and
        // the engine guards duplicate NAMES. Two entries for one path is a different
        // mistake it does not catch, and it would search the same file twice and
        // pool the result with itself, inflating the evidence for those peptides.
        let mut r = req();
        r.experiment = true;
        r.mzml = vec!["a.mzML".into(), "b.mzML".into(), "a.mzML".into()];
        let e = argv(&r).unwrap_err();
        assert!(e.contains("more than once"), "{e}");
    }

    #[test]
    fn a_pooled_experiment_reports_its_combined_table_and_says_so() {
        // `run-experiment` writes `scored_combined.parquet` (and an experiment-wide
        // TSV pair at the root), so reading only `psms_scored.parquet.report.json` left
        // the results screen blank after every experiment. And the counts it does
        // yield are experiment-wide: the grouped q columns are grouped across the whole
        // experiment, so a per-file reading of them is diluted by ~1/n_runs.
        let dir = std::env::temp_dir().join("mumdia-results-experiment");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        assert!(
            read_results(&dir).is_none(),
            "no report at all is no results"
        );

        let body = r#"{"stats":{"classifier":"nn_torch","psms":10,
            "target_peptides_at_1pct":7,"target_precursors_at_1pct":8,
            "target_protein_groups_at_1pct":3}}"#;
        std::fs::write(dir.join("scored_combined.parquet.report.json"), body).unwrap();
        let r = read_results(&dir).expect("the combined table must be read");
        assert!(r.experiment_wide, "a combined table is experiment-wide");
        assert_eq!(r.peptides_1pct, 7);
        assert_eq!(r.precursors_1pct, 8);
        // This fixture wrote no TSV, so neither is reported present.
        assert!(!r.has_peptides_tsv && !r.has_proteins_tsv);

        // A single run's own report wins, and is not labelled experiment-wide.
        std::fs::write(dir.join("psms_scored.parquet.report.json"), body).unwrap();
        let r = read_results(&dir).unwrap();
        assert!(!r.experiment_wide);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
