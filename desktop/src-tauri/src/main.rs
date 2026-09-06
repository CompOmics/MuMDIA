// A release build must not open a console window behind the application. Debug
// builds keep it, because that is where the developer's own `println!` goes.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use mumdia_console::{components, diann, engine, preflight as pf, run, settings, thermo};

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
    diann: Arc<diann::Builder>,
    diann_installer: Arc<diann::Installer>,
    thermo: Arc<thermo::Installer>,
}

/// Whether a DIA-NN the user installed is present, and whether it runs.
///
/// MuMDIA does not install DIA-NN. See `diann::LICENCE_NOTICE` for why this is a
/// detection rather than an installation like the Python components.
#[tauri::command]
fn diann_status() -> serde_json::Value {
    serde_json::json!({
        "status": diann::detect(),
        "notice": diann::LICENCE_NOTICE,
    })
}

/// Remember a DIA-NN the user pointed at, or forget it when `path` is absent.
#[tauri::command]
fn diann_set_path(path: Option<String>) -> Result<diann::Status, String> {
    diann::set_path(path.as_deref())
}

/// Record that the licence notice was read and accepted. Nothing runs before it is.
#[tauri::command]
fn diann_acknowledge(accepted: bool) -> Result<diann::Status, String> {
    diann::acknowledge_licence(accepted)
}

/// Predict a spectral library from a FASTA with the user's DIA-NN, then convert it
/// into the two tables a search consumes.
#[tauri::command]
fn diann_build(state: tauri::State<'_, AppState>, req: diann::BuildRequest) -> Result<(), String> {
    diann::build(Arc::clone(&state.diann), req)
}

/// Progress of the library build, polled while it runs.
#[tauri::command]
fn diann_build_state(state: tauri::State<'_, AppState>) -> diann::BuildState {
    state.diann.snapshot()
}

/// Whether this platform can be offered the pinned DIA-NN 1.8.1 download, and what
/// it costs in bandwidth and disk.
#[tauri::command]
fn diann_offer() -> serde_json::Value {
    diann::offer()
}

/// Fetch DIA-NN 1.8.1 from the vendor's own release URL onto this machine.
///
/// MuMDIA neither ships nor mirrors those bytes; see the module comment in
/// `diann.rs` for why that distinction is the entire basis for offering this.
#[tauri::command]
fn diann_install(state: tauri::State<'_, AppState>) -> Result<(), String> {
    diann::install(Arc::clone(&state.diann_installer))
}

/// Progress of the download, polled while it runs.
#[tauri::command]
fn diann_install_state(state: tauri::State<'_, AppState>) -> diann::InstallState {
    state.diann_installer.snapshot()
}

/// Whether the Thermo `.raw` converter is installed, and the last install's progress.
#[tauri::command]
fn thermo_status(state: tauri::State<'_, AppState>) -> thermo::Status {
    state.thermo.refresh()
}

/// Download and unpack ThermoRawFileParser.
///
/// Unlike DIA-NN this needs no licence gate: it is Apache-2.0 and redistributable.
#[tauri::command]
fn thermo_install(state: tauri::State<'_, AppState>) -> Result<(), String> {
    thermo::install(Arc::clone(&state.thermo))
}

/// Open one of a fixed set of project URLs in the user's browser.
///
/// An allowlist rather than an arbitrary opener. The frontend is our own code, but
/// `open_url` is reachable from anything running in the webview, and handing a
/// shell-adjacent opener an arbitrary string is how a link becomes a command.
#[tauri::command]
fn open_url(url: String) -> Result<(), String> {
    const ALLOWED: &[&str] = &[
        "https://proteowizard.sourceforge.io/",
        "https://github.com/compomics/ThermoRawFileParser",
        "https://github.com/vdemichev/DiaNN",
    ];
    if !ALLOWED.contains(&url.as_str()) {
        return Err(format!(
            "{url} is not one of the links this application opens"
        ));
    }
    #[cfg(windows)]
    let r = std::process::Command::new("rundll32")
        .args(["url.dll,FileProtocolHandler", &url])
        .spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(&url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(&url).spawn();
    r.map(|_| ())
        .map_err(|e| format!("could not open {url}: {e}"))
}

/// What a chosen spectra path is, and which converter it needs.
///
/// Answered in the backend because the file-versus-directory distinction decides it
/// -- a `.raw` file is Thermo, a `.raw` directory is Waters -- and the webview
/// cannot stat a path.
#[tauri::command]
fn vendor_of(path: String) -> serde_json::Value {
    serde_json::json!({
        "label": thermo::label(&path),
        "needs": match thermo::needs(&path) {
            thermo::Needs::Nothing => "Nothing",
            thermo::Needs::ThermoParser => "ThermoParser",
            thermo::Needs::Msconvert => "Msconvert",
        },
    })
}

/// Path to msconvert, if the engine can find one.
#[tauri::command]
fn msconvert_status() -> Option<String> {
    thermo::msconvert_available()
}

/// What a FASTA-mode search would do about its library: reuse one, or build it.
///
/// Asked before the search starts, so the interface can say "already built" or "this
/// takes a while" instead of the user discovering which after pressing Start.
#[tauri::command]
fn diann_library_plan(req: diann::BuildRequest) -> Result<serde_json::Value, String> {
    diann::library_plan(req)
}

/// Stop a running library build.
///
/// Separate from `cancel_run`, which only knows about engine runs: a DIA-NN
/// prediction is not one, so before this a multi-hour build could not be stopped.
#[tauri::command]
fn diann_cancel(state: tauri::State<'_, AppState>) {
    state.diann.cancel();
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

    // A Thermo .raw with no converter fails inside the engine, after the run has
    // been launched and the interface has switched to the progress screen. Caught
    // here it is a sentence on the screen the user is already looking at.
    // Every selected file, not just the first. A mixed selection where only one file
    // needs a converter would otherwise pass preflight and fail mid-experiment, after
    // the other runs had already been searched.
    // Both answers come from the engine's own search (`doctor --json`), not from this
    // application's install directory. Asking the narrow question blocked a user whose
    // ThermoRawFileParser was on PATH, or named in the configuration, for a file the
    // engine converts without complaint -- and the GUI was the only path that refused.
    let thermo_missing = thermo::engine_thermo_parser().is_none();
    let msconvert_missing = thermo::msconvert_available().is_none();
    let mut needs_thermo: Vec<&str> = Vec::new();
    let mut needs_msconvert: Vec<&str> = Vec::new();
    for m in &req.mzml {
        match thermo::needs(m) {
            thermo::Needs::ThermoParser if thermo_missing => needs_thermo.push(m),
            thermo::Needs::Msconvert if msconvert_missing => needs_msconvert.push(m),
            _ => {}
        }
    }
    if !needs_thermo.is_empty() {
        blockers.push(format!(
            "{} selected file(s) are Thermo .raw and the converter is not installed. 
Install it on the Setup screen, or convert them to mzML yourself. 
First: {}",
            needs_thermo.len(),
            needs_thermo[0]
        ));
    }
    if !needs_msconvert.is_empty() {
        blockers.push(format!(
            "{} selected file(s) need ProteoWizard msconvert, which was not found. 
MuMDIA does not install it; see the Setup screen. 
First: {} ({})",
            needs_msconvert.len(),
            needs_msconvert[0],
            thermo::label(needs_msconvert[0])
        ));
    }

    // Room on disk. The engine cannot resume, so filling the volume at hour three
    // loses the whole search; this is the cheapest possible moment to notice.
    // One run scales with its own file. A pooled experiment keeps every run's
    // intermediates under one output directory, so the relevant input size is the
    // total: estimating from a single file would understate an eight-run experiment
    // roughly eightfold, and the engine cannot resume.
    let disk = if req.experiment && req.mzml.len() > 1 {
        pf::disk_multi(&req.mzml, &req.out_dir)
    } else {
        pf::disk(
            req.mzml.first().map(|s| s.as_str()).unwrap_or(""),
            &req.out_dir,
        )
    };

    // The INPUT volume, separately. `raw::ensure_mzml` writes the converted mzML
    // beside the input whenever that directory is writable, typically two to four
    // times the vendor file, but the estimate above only ever asked about the output
    // volume. A vendor file on a nearly-full acquisition drive with results aimed at a
    // roomy one passed preflight and then filled the acquisition drive mid-conversion,
    // and the engine cannot resume.
    let conversion = pf::conversion_space(&req.mzml);
    let mut warnings: Vec<String> = Vec::new();
    for w in conversion {
        warnings.push(w);
    }
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
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();

    // Materialised copies first: in a packaged build this is the only place they
    // exist. See `materialise_presets` for why they are compiled in rather than
    // shipped as bundle resources.
    match materialise_presets() {
        Ok(d) => dirs.push(d),
        Err(e) => eprintln!("could not write the bundled presets: {e}"),
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            // A copy placed beside the application by hand, or by a release archive,
            // wins over the compiled-in ones: someone who drops a config there means
            // it.
            dirs.insert(0, d.join("configs").join("examples"));
            // Three levels from `desktop/target/<profile>/` to the repository root,
            // not four. The development case, and it must beat the materialised
            // copies so an edited example config takes effect without reinstalling.
            dirs.insert(1, d.join("../../../configs/examples"));
        }
    }

    // A preset this application refuses to run must not be offered. `preflight`
    // blocks a configuration needing no Python sidecar, so `native.json` -- a valid
    // configuration for a command-line user -- was listed and then rejected, and the
    // rejection told the reader to choose a preset that uses retention-time
    // modelling. Offering a choice that cannot be taken is worse than not offering
    // it.
    //
    // Decided by asking the engine rather than by filename, which is the same
    // authority `preflight` uses; a renamed or added native-only config is filtered
    // for the same reason without anyone remembering to update a list.
    let engine = engine::resolve().ok().map(|(exe, _)| exe);

    let mut out = Vec::new();
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
            let path = p.display().to_string();
            if let Some(exe) = &engine {
                // Only a definite "needs nothing" excludes a preset. An engine that
                // cannot answer must not silently empty the list.
                if run::needs_no_sidecar(exe, Some(&path)) == Ok(true) {
                    continue;
                }
            }
            let name = p
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("config")
                .to_string();
            out.push(serde_json::json!({
                "name": name,
                "path": path,
            }));
        }
        if !out.is_empty() {
            break;
        }
    }
    out
}

/// The example configurations, compiled in.
///
/// # Why compiled in and not bundle resources
///
/// They live at `configs/examples/` in the repository, outside `src-tauri`, and a
/// Tauri resource path containing `..` does not work: the list form places it in a
/// literal `_up_` directory and the map form fails with "Access is denied". The
/// settings schema and the two requirement files are compiled in for exactly this
/// reason, so this follows them.
///
/// # Why this is not optional
///
/// `preflight` refuses to start a search that needs no Python sidecar at all,
/// because that is the weak native path. The engine's own defaults ARE that path, so
/// a run with no configuration is always blocked. With no presets to choose from,
/// the packaged application therefore could not run anything: the blocker told the
/// user to "choose a preset that uses retention-time modelling" while the preset
/// list was empty. Found by extracting the MSI, not by any test.
const PRESET_FILES: &[(&str, &str)] = &[
    (
        "diann-library.json",
        include_str!("../../../configs/examples/diann-library.json"),
    ),
    (
        "fasta-sidecars.json",
        include_str!("../../../configs/examples/fasta-sidecars.json"),
    ),
    (
        "native.json",
        include_str!("../../../configs/examples/native.json"),
    ),
];

/// Write the compiled-in presets into the per-user data directory and return it.
///
/// Rewritten every call rather than only when absent, so that upgrading the
/// application updates the configurations it ships. A file the user edited in place
/// is therefore overwritten, which is why `save_settings` writes elsewhere and why
/// the interface never offers these as editable.
fn materialise_presets() -> Result<std::path::PathBuf, String> {
    let dir = components::data_dir().join("configs");
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    for (name, body) in PRESET_FILES {
        let path = dir.join(name);
        // Compare before writing: this runs on every visit to the search screen, and
        // rewriting three files each time would churn the disk for nothing.
        let same = std::fs::read_to_string(&path).is_ok_and(|existing| existing == *body);
        if !same {
            std::fs::write(&path, body)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
    }
    Ok(dir)
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
            presets,
            diann_status,
            diann_set_path,
            diann_acknowledge,
            diann_build,
            diann_build_state,
            diann_offer,
            diann_install,
            diann_install_state,
            thermo_status,
            thermo_install,
            open_url,
            vendor_of,
            msconvert_status,
            diann_library_plan,
            diann_cancel
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_compiled_in_preset_parses_and_needs_a_sidecar() {
        // Two properties, and the second is the one whose absence made a packaged
        // application unable to run anything at all.
        //
        // `preflight` blocks a configuration that needs no Python sidecar, because
        // that is the weak native path. The engine's defaults ARE that path, so a run
        // with no configuration is always blocked. If the preset list is empty, or
        // every preset in it is native-only, the user is told to choose a preset that
        // uses retention-time modelling and has nothing to choose.
        assert!(!PRESET_FILES.is_empty(), "there must be presets to offer");

        let mut sidecar_presets = 0;
        for (name, body) in PRESET_FILES {
            let v: serde_json::Value = serde_json::from_str(body)
                .unwrap_or_else(|e| panic!("{name} is not valid JSON: {e}"));

            // A preset that reaches for a Python role. Checked structurally rather
            // than by name, so the assertion survives a renamed file.
            let reaches_python = ["rescore", "predict_frag", "mbr"].iter().any(|section| {
                v.get(section).is_some_and(|s| {
                    s.get("python").is_some()
                        || s.get("deeplc_python").is_some()
                        || s.get("ms2pip_python").is_some()
                        || s.get("rt_predictor").is_some_and(|r| r != "native")
                        || s.get("predictor").is_some_and(|r| r != "native")
                        || s.get("classifier").is_some_and(|c| c != "native_tda")
                })
            });
            if reaches_python {
                sidecar_presets += 1;
            }
        }
        assert!(
            sidecar_presets > 0,
            "at least one preset must use a Python sidecar, or `preflight` blocks \
             every run a packaged application can start"
        );
    }

    #[test]
    fn presets_are_materialised_where_a_packaged_build_can_find_them() {
        // The packaged MSI ships no `configs/` directory: the examples live outside
        // `src-tauri` and a Tauri resource path containing `..` does not work. So they
        // are compiled in and written out, and this asserts the writing works.
        let dir = materialise_presets().expect("presets must materialise");
        for (name, body) in PRESET_FILES {
            let path = dir.join(name);
            assert!(path.is_file(), "{} was not written", path.display());
            assert_eq!(
                std::fs::read_to_string(&path).unwrap(),
                *body,
                "{name} was written with different content"
            );
        }
        // Idempotent: this runs on every visit to the search screen.
        let again = materialise_presets().expect("second call must also succeed");
        assert_eq!(again, dir);
    }

    #[test]
    fn a_preset_the_application_refuses_to_run_is_not_offered() {
        // `native.json` is native-only, and `preflight` blocks exactly that. It was
        // offered and then rejected, with the rejection advising the reader to pick a
        // preset that uses retention-time modelling.
        //
        // Requires a resolvable engine, because the filter asks the engine.
        if engine::resolve().is_err() {
            eprintln!("no engine resolvable; skipping");
            return;
        }
        let names: Vec<String> = presets()
            .into_iter()
            .map(|p| p["name"].as_str().unwrap_or_default().to_string())
            .collect();
        assert!(
            !names.iter().any(|n| n == "native"),
            "the native-only preset must not be offered: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "diann-library"),
            "the sidecar presets must survive the filter: {names:?}"
        );
    }

    #[test]
    fn presets_lists_something() {
        // The end-to-end version of the above, through the command the interface
        // actually calls. An empty list here is the blocker.
        let list = presets();
        assert!(
            !list.is_empty(),
            "the interface must be offered at least one preset"
        );
        for p in &list {
            let path = p["path"].as_str().expect("a preset needs a path");
            assert!(
                std::path::Path::new(path).is_file(),
                "{path} is offered but does not exist"
            );
        }
    }
}
