// A release build must not open a console window behind the application. Debug
// builds keep it, because that is where the developer's own `println!` goes.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use mumdia_console::{components, engine, preflight as pf, run, settings};

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tauri::Manager;

#[derive(Default)]
struct AppState {
    runs: Mutex<HashMap<String, Arc<run::Run>>>,
    next_id: AtomicU64,
    installer: Arc<components::Installer>,
}

/// Which engine will be used, and does it execute.
#[tauri::command]
fn engine_info() -> Result<engine::Info, String> {
    engine::info()
}

/// The state of the managed Python environment.
#[tauri::command]
fn components_status(state: tauri::State<'_, AppState>) -> serde_json::Value {
    serde_json::json!({
        "primary": state.installer.refresh(components::Env::Primary),
        // Reported separately because it is optional and cannot share the primary
        // environment: MS2PIP and DeepLC pin incompatible sqlalchemy majors.
        "ms2pip": state.installer.refresh(components::Env::Ms2pip),
    })
}

/// Create the managed environment and install the analysis packages into it.
///
/// Returns as soon as the work is under way; the interface polls
/// `components_status` for the log and the outcome.
#[tauri::command]
fn components_install(
    state: tauri::State<'_, AppState>,
    env: Option<components::Env>,
) -> Result<(), String> {
    components::install(
        Arc::clone(&state.installer),
        env.unwrap_or(components::Env::Primary),
    )
}

/// Everything that must be true before a search can start.
///
/// Checked here rather than at the moment of starting, so the interface can explain
/// and offer a fix instead of reporting a failure.
#[tauri::command]
fn preflight(
    state: tauri::State<'_, AppState>,
    req: run::Request,
) -> Result<serde_json::Value, String> {
    let (exe, _) = engine::resolve()?;
    let mut blockers: Vec<String> = Vec::new();

    // The minimal path is not offered. See `run::needs_no_sidecar` for why the
    // predicate is "needs no sidecar at all" rather than anything about the
    // rescorer.
    match run::needs_no_sidecar(&exe, req.config.as_deref()) {
        Ok(true) => blockers.push(
            "This configuration would run without any of the analysis components, which \
             identifies far fewer peptides. Choose a preset that uses retention-time \
             modelling, or install the components."
                .into(),
        ),
        Ok(false) => {}
        // A configuration the engine cannot even read is a real problem, but it is
        // the engine's message that says what is wrong with it.
        Err(e) => blockers.push(e),
    }

    let comp = state.installer.refresh(components::Env::Primary);
    if !comp.complete {
        blockers.push(format!(
            "The analysis components are not installed{}.",
            if comp.missing.is_empty() {
                String::new()
            } else {
                format!(" (missing {})", comp.missing.join(", "))
            }
        ));
    }

    // Room on disk. The engine cannot resume, so filling the volume at hour three
    // loses the whole search; this is the cheapest possible moment to notice.
    let disk = pf::disk(&req.mzml, &req.out_dir);
    let mut warnings: Vec<String> = Vec::new();
    if !disk.unknown && !disk.enough {
        let gb = |b: u64| format!("{:.1} GB", b as f64 / 1e9);
        warnings.push(format!(
            "This search may need about {} and the drive has {} free. A search cannot \
             resume, so running out part-way loses all of it.",
            gb(disk.estimated_output_bytes),
            gb(disk.free_bytes)
        ));
    }

    Ok(serde_json::json!({
        "ok": blockers.is_empty(),
        "blockers": blockers,
        "warnings": warnings,
        "disk": disk,
        "components_complete": comp.complete,
    }))
}

/// Past runs, read back from the folders they wrote.
///
/// The interface remembers which folders it has used; the content of each entry
/// comes from the folder itself, so a run deleted or moved on disk simply stops
/// appearing rather than lingering as a stale row.
#[tauri::command]
fn history(dirs: Vec<String>) -> Vec<run::HistoryEntry> {
    let mut out: Vec<run::HistoryEntry> = dirs
        .iter()
        .filter_map(|d| run::history_entry(Path::new(d)))
        .collect();
    // Newest first.
    out.sort_by_key(|e| std::cmp::Reverse(e.finished_unix_ms));
    out
}

/// Peaks per MS2 spectrum for a chosen file, so the peak cap can be set from the
/// file rather than from another acquisition.
#[tauri::command]
fn peak_census(mzml: String) -> Result<serde_json::Value, String> {
    pf::peak_census(&mzml)
}

/// The settings schema the editor renders its form from.
#[tauri::command]
fn config_schema() -> Result<settings::Schema, String> {
    settings::load_schema()
}

/// Write an override set, then ask the engine whether it accepts it.
///
/// Validating here rather than at run time is the point: a value the engine would
/// reject is reported next to the field while it is being edited.
#[tauri::command]
fn save_settings(
    name: String,
    overrides: std::collections::BTreeMap<String, serde_json::Value>,
) -> Result<String, String> {
    let path = settings::save(&name, overrides)?;
    settings::validate(&path)?;
    Ok(path)
}

/// Start a search. Returns the run id used by every subsequent call.
#[tauri::command]
fn start_run(state: tauri::State<'_, AppState>, req: run::Request) -> Result<String, String> {
    let id = format!("run-{}", state.next_id.fetch_add(1, Ordering::SeqCst) + 1);
    let handle = run::start(id.clone(), req)?;
    state
        .runs
        .lock()
        .map_err(|_| "internal state is poisoned".to_string())?
        .insert(id.clone(), handle);
    Ok(id)
}

/// Poll one run. The interface calls this on a timer; everything it displays is here.
#[tauri::command]
fn run_state(state: tauri::State<'_, AppState>, id: String) -> Option<run::Snapshot> {
    let runs = state.runs.lock().ok()?;
    runs.get(&id).map(|r| r.snapshot())
}

/// Stop a run: kill the process tree, then sweep the temporary files it left.
#[tauri::command]
fn cancel_run(state: tauri::State<'_, AppState>, id: String) -> Result<(), String> {
    let handle = {
        let runs = state
            .runs
            .lock()
            .map_err(|_| "internal state is poisoned".to_string())?;
        runs.get(&id).cloned()
    };
    match handle {
        Some(r) => {
            // Killing waits for SIGTERM to be given a chance on Unix, so do it off
            // the command thread and let the interface keep polling meanwhile.
            std::thread::spawn(move || r.cancel());
            Ok(())
        }
        None => Err(format!("no such run: {id}")),
    }
}

/// Open a folder in the platform file manager.
#[tauri::command]
fn reveal(path: String) -> Result<(), String> {
    let p = Path::new(&path);
    if !p.exists() {
        return Err(format!("{path} does not exist"));
    }
    #[cfg(windows)]
    let r = std::process::Command::new("explorer").arg(p).spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(p).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(p).spawn();
    // `explorer` returns a non-zero exit code even on success, so only the spawn
    // itself is checked.
    r.map(|_| ())
        .map_err(|e| format!("could not open {path}: {e}"))
}

/// Configuration presets shipped beside the engine, for the input screen.
///
/// Found rather than hard-coded: the release archive carries `configs/examples/`, and
/// listing what is actually there means a preset cannot be offered that does not
/// exist. An empty list is a valid answer and the interface says so.
#[tauri::command]
fn presets() -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            dirs.push(d.join("configs").join("examples"));
            dirs.push(d.join("../../../../configs/examples"));
        }
    }
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut found: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        found.sort();
        for p in found {
            let name = p
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("config")
                .to_string();
            out.push(serde_json::json!({
                "name": name,
                "path": p.display().to_string(),
            }));
        }
        if !out.is_empty() {
            break;
        }
    }
    out
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .setup(|app| {
            // Tell the engine and component lookups where the bundler put things.
            // Without this an AppImage cannot find its own engine: its resources live
            // under `usr/lib/<app>/`, not beside the executable, so every
            // `exe.parent()` candidate misses.
            match app.path().resource_dir() {
                Ok(dir) => engine::set_resource_dir(dir),
                Err(e) => eprintln!("could not resolve the resource directory: {e}"),
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            engine_info,
            components_status,
            components_install,
            config_schema,
            save_settings,
            peak_census,
            history,
            preflight,
            start_run,
            run_state,
            cancel_run,
            reveal,
            presets
        ])
        .on_window_event(|window, event| {
            // Closing the window must not leave an engine, or a Python worker, running
            // with no way to reach it. Every live run is stopped first.
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.app_handle().try_state::<AppState>() {
                    let handles: Vec<_> = state
                        .runs
                        .lock()
                        .map(|r| r.values().cloned().collect())
                        .unwrap_or_default();
                    for h in handles {
                        if h.snapshot().status == "running" {
                            h.cancel();
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to start the MuMDIA console");
}
