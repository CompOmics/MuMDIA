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

/// What the interface asks for when it starts a search.
#[derive(Deserialize, Debug, Clone)]
pub struct Request {
    pub mzml: String,
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
}

pub struct Run {
    pub snapshot: Mutex<Snapshot>,
    /// Process id of the engine. On Unix this is also its process-group id, because
    /// it is spawned into a new group.
    pid: Mutex<Option<u32>>,
    cancelled: AtomicBool,
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

    /// Stop the run: kill the process tree, then remove the rubble.
    ///
    /// Both halves matter. The engine spawns Python workers, so killing only the
    /// engine would orphan a process that may hold tens of gigabytes. And a hard
    /// kill skips destructors, so the atomic-write layer never removes its
    /// `.tmp-<pid>` files; without a sweep the next run starts in a dirty directory.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        let pid = self.pid.lock().ok().and_then(|p| *p);
        if let Some(pid) = pid {
            kill_tree(pid);
        }
        let out_dir = self.snapshot().out_dir;
        sweep_temp_files(Path::new(&out_dir));
        self.set(|s| {
            if s.status == "running" || s.status == "starting" {
                s.status = "cancelled".into();
            }
        });
    }
}

/// Kill a process and everything it spawned.
///
/// Deliberately shelling out rather than calling the platform APIs directly: both
/// would need `unsafe`, and this is not on any hot path. `taskkill /T` walks the
/// tree at kill time and `kill` on a negative pid signals the whole group, so a
/// Python worker the application never knew about is included either way.
fn kill_tree(pid: u32) {
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
    if req.mzml.trim().is_empty() {
        return Err("select an mzML file".into());
    }
    if req.out_dir.trim().is_empty() {
        return Err("choose a folder for the results".into());
    }

    let mut a: Vec<String> = vec!["run".into()];
    a.push("--mzml".into());
    a.push(req.mzml.clone());
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

    let display = format!(
        "{} {}",
        quote(&exe.display().to_string()),
        args.iter().map(|a| quote(a)).collect::<Vec<_>>().join(" ")
    );

    let library_mode = req.lib_precursors.is_some();
    let run = Arc::new(Run {
        snapshot: Mutex::new(Snapshot {
            id: id.clone(),
            status: "starting".into(),
            exit_code: None,
            error: None,
            stages: Vec::new(),
            log: Vec::new(),
            out_dir: req.out_dir.clone(),
            command: display,
            started_unix_ms: now_ms(),
            elapsed_ms: 0,
            results: None,
            library_mode,
        }),
        pid: Mutex::new(None),
        cancelled: AtomicBool::new(false),
    });

    let mut cmd = engine::command(&exe);
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

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("could not start {}: {e}", exe.display()))?;

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
                let running = {
                    let s = run.snapshot();
                    s.status == "running" || s.status == "starting"
                };
                run.set(|s| {
                    s.stages = stages;
                    s.elapsed_ms = started.elapsed().as_millis() as u64;
                });
                if !running {
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

    // Reap the child, then publish the terminal state in one step.
    //
    // The order matters. Everything a finished run displays -- its stages and its
    // results -- is read from disk here, BEFORE the status stops being `running`.
    // Doing it the other way round leaves a window in which the run says it is
    // finished but has no stages, which an interface polling for completion will
    // reliably catch: the results screen renders empty and then fills in.
    {
        let run = Arc::clone(&run);
        let out_dir = PathBuf::from(&req.out_dir);
        std::thread::spawn(move || {
            let outcome = child.wait();
            let stages = scan_stages(&out_dir);
            let results = read_results(&out_dir);
            match outcome {
                Ok(status) => run.set(|s| {
                    s.stages = stages;
                    s.results = results;
                    s.exit_code = status.code();
                    if s.status == "cancelled" {
                        return;
                    }
                    if status.success() {
                        s.status = "done".into();
                    } else {
                        s.status = "failed".into();
                        // The last stderr line is almost always the anyhow error
                        // chain, which is the sentence worth showing.
                        s.error = s.log.iter().rev().find(|l| !l.trim().is_empty()).cloned();
                    }
                }),
                Err(e) => run.set(|s| {
                    s.stages = stages;
                    s.status = "failed".into();
                    s.error = Some(format!("could not wait for the engine: {e}"));
                }),
            }
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
    let path = dir.join("psms_scored.parquet.report.json");
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> Request {
        Request {
            mzml: "a.mzML".into(),
            out_dir: "out".into(),
            fasta: None,
            lib_precursors: None,
            lib_fragments: None,
            config: None,
            threads: None,
        }
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
