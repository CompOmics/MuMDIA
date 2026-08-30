// A release build must not open a console window behind the application. Debug
// builds keep it, because that is where the developer's own `println!` goes.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use mumdia_console::{engine, run};

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tauri::Manager;

#[derive(Default)]
struct AppState {
    runs: Mutex<HashMap<String, Arc<run::Run>>>,
    next_id: AtomicU64,
}

/// Which engine will be used, and does it execute.
#[tauri::command]
fn engine_info() -> Result<engine::Info, String> {
    engine::info()
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
        .invoke_handler(tauri::generate_handler![
            engine_info,
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
