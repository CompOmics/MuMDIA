// MuMDIA console frontend.
//
// No framework and no build step, deliberately: the heaviest screen here is a list
// of stages and a handful of fields, and adding Node to the release pipeline for
// that would be a poor trade. If this grows into the generated 86-field settings
// editor, revisit it then.
//
// State lives in the backend. This polls `run_state` and renders what it returns,
// so a reload or a reopened window shows the truth rather than a stale copy.

const { invoke } = window.__TAURI__.core;
const dialog = window.__TAURI__.dialog;

const $ = (id) => document.getElementById(id);

// The preset selected when the application starts.
const DEFAULT_PRESET = "diann-library";

// ── pipeline order ──────────────────────────────────────────────────────────
// The engine reports the stage that produced each artifact, but not what is still
// to come, so the expected sequence lives here. Library-input mode skips the three
// library-building stages, exactly as the engine does.
const STAGES_FASTA = [
  ["convert", "Reading spectra"],
  ["digest", "Digesting the FASTA"],
  ["peptidoforms", "Expanding peptidoforms"],
  ["predict-frag", "Predicting the library"],
  ["search-seed", "First-pass search"],
  ["rt-im-train", "Retention-time model"],
  ["extract", "Extracting chromatograms"],
  ["features", "Computing features"],
  ["compete", "Competition"],
  ["rescore", "Rescoring"],
  ["quant", "Quantification"],
  ["report", "Writing the report"],
];
const SKIP_IN_LIBRARY_MODE = new Set(["digest", "peptidoforms", "predict-frag"]);
// A library build without a search (`build-library`) runs only these.
const LIBRARY_BUILD_STAGES = SKIP_IN_LIBRARY_MODE;

const state = {
  mode: "fasta",
  picks: {
    // A list. Several spectra files mean one of two different analyses, chosen by
    // `runMode`; a single file is an ordinary search either way.
    mzml: [], fasta: "", lib_precursors: "", lib_fragments: "", out_dir: "",
    lib_fasta: "", lib_out: "",
    // A DIA-NN library to convert into the two tables, and the folder they go in.
    diann_lib: "", diann_lib_out: "",
    // The condition of each spectra file of an experiment, by path
    // (`quant.normalisation_scope = condition`, `run-experiment --condition`).
    conditions: {},
  },
  // The files whose condition the user typed; the others follow the suggestion from the
  // file names.
  conditionTyped: {},
  diannTimer: null,
  diannGetTimer: null,
  thermoTimer: null,
  thermoReady: false,
  msconvert: null,
  // DIA-NN by default: it is far more sensitive than the built-in predictors. Falls
  // back to "builtin" while DIA-NN is unavailable, unless the user chose one.
  libSrc: "diann",
  libSrcChosen: false,
  // On the spectral-library tab: two MuMDIA tables ("tables"), or a DIA-NN library to
  // convert into them first ("diann").
  libIn: "tables",
  buildingLibrary: false,
  // Whether the in-depth settings were opened, so a re-render keeps them open.
  advancedOpen: false,
  // The visDIA environment is installed, so "Open in visDIA" can work.
  visdiaReady: false,
  // "separate" = one search per file; "experiment" = one pooled run-experiment.
  runMode: "experiment",
  // Batch progress: which file of how many, and what each one produced.
  batch: null,
  diannOffer: null,
  runId: null,
  timer: null,
  lastStatus: null,
  outDir: "",
  // Where the application keeps its downloaded and generated files, for the button
  // that opens it. Reported by the backend rather than guessed here.
  managedDir: "",
  componentsReady: false,
  // The engine was found at startup. Start is only ever enabled when it was.
  engineOk: false,
  // A Start is in progress: library build, derived configuration, preflight, launch.
  starting: false,
  setupTimer: null,
  // Polls the Visual C++ runtime installation while it runs.
  vcrtTimer: null,
  schema: null,
  overrides: {},
  savedConfig: null,
  // The presets the Search screen offers, by name and path (`presets`).
  presets: [],
  // The built-in preset the Settings screen's values start from ("" for the engine
  // defaults), and the one "My settings" was saved from: a setup names it and lists the
  // changes from it (#297).
  settingsPreset: "",
  savedPreset: "",
  // The override set of the configuration the next search uses, to tell edits on the
  // Settings screen that are not saved and used yet.
  inUse: {},
  // Named setups (`setups_list`), and a setup read and checked, waiting for Apply.
  setups: { list: [], dir: "", pending: null },
  // The Search tab's modification choice: "name|residue" -> "var" or "fix", and the
  // engine's catalogue it is chosen from.
  mods: new Map(),
  catalogue: null,
  // The last search-space size and band plan, and the request counter that drops a
  // stale answer when the inputs changed while it was being computed.
  space: null,
  spaceSeq: 0,
  spaceTimer: null,
  // The built-in path's predictor plan (`builtin_plan`), with the same stale-answer
  // counter: each check runs `mumdia doctor` once per recipe it considers.
  builtinSeq: 0,
  builtinTimer: null,
  // The built-in path would build its library on the engine's native models, which the
  // engine refuses: Start and Build library only stay disabled, with the reason shown.
  builtinBlocked: false,
  // The library on the results screen after "Build library only": its two tables, the
  // folder they are in, and the engine's library.json for a built-in build.
  libraryResult: null,
  // Output folders this application has used. The only thing it remembers; what
  // each one contains is read back from the folder.
  known: readStored("mumdia.folders", []),
  // How many of the oldest remembered folders the cap has pushed out, so History can
  // say so instead of losing them silently (#282).
  knownDropped: readStored("mumdia.folders.dropped", 0),
  // SCIEX ZT Scan DIA files (`refreshZtscanNote`): the engine's isolation census of each
  // selected mzML by path (null when it could not be read), whether the user typed the
  // half-width, the overrides that turn the ZT Scan analysis off once read (`ztscan_off`),
  // whether the user changed the analysis checkbox (until then it is ticked whenever the
  // analysis applies) and what they chose (null when only `analysisTouched` says so), and
  // what the analysis's library settings changed in the Search space card
  // (`syncZtscanLibrary`).
  ztscan: {
    probes: {}, hwTyped: false, off: null, analysisTouched: false, analysisChoice: null, lib: null,
  },
};

function readStored(key, fallback) {
  try {
    const v = JSON.parse(localStorage.getItem(key) ?? "null");
    const sameKind = Array.isArray(fallback) ? Array.isArray(v) : typeof v === typeof fallback;
    return sameKind ? v : fallback;
  } catch {
    // A storage that cannot be read, or holds something that is not JSON, must not stop
    // the application from starting.
    return fallback;
  }
}

// The remembered folders are paths only, so the cap is generous; History says how many
// older folders it pushed out.
const KNOWN_FOLDERS_MAX = 500;

function saveKnownFolders() {
  try {
    localStorage.setItem("mumdia.folders", JSON.stringify(state.known));
    localStorage.setItem("mumdia.folders.dropped", JSON.stringify(state.knownDropped));
  } catch {
    // A browser storage that refuses to write is not worth failing a search over.
  }
}

function rememberFolder(dir) {
  if (!dir || state.known.includes(dir)) return;
  state.known.unshift(dir);
  if (state.known.length > KNOWN_FOLDERS_MAX) {
    state.knownDropped += state.known.length - KNOWN_FOLDERS_MAX;
    state.known = state.known.slice(0, KNOWN_FOLDERS_MAX);
  }
  saveKnownFolders();
}

function forgetFolder(dir) {
  state.known = state.known.filter((d) => d !== dir);
  saveKnownFolders();
}

// ── small helpers ───────────────────────────────────────────────────────────
const fmtInt = (n) => (n ?? 0).toLocaleString("en-GB");

function fmtDuration(ms) {
  if (!ms || ms < 0) return "";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

function baseName(p) {
  if (!p) return "";
  const parts = p.split(/[\\/]/);
  return parts[parts.length - 1] || p;
}

function show(el, on) {
  if (on) el.removeAttribute("hidden");
  else el.setAttribute("hidden", "");
}

function node(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function banner(el, message) {
  if (!message) {
    show(el, false);
    return;
  }
  el.textContent = message;
  show(el, true);
}

function screen(name) {
  for (const s of document.querySelectorAll(".screen")) {
    s.classList.toggle("on", s.id === `screen-${name}`);
  }
  for (const b of document.querySelectorAll(".nav")) {
    b.classList.toggle("on", b.dataset.screen === name);
  }
}

// ── startup ─────────────────────────────────────────────────────────────────
async function init() {
  const cores = navigator.hardwareConcurrency;
  if (cores) {
    $("cores").textContent = `${cores} cores available`;
    // All but two, so the machine stays usable while DIA-NN predicts.
    $("d-threads").value = String(Math.max(1, cores - 2));
    $("d-threads").max = String(cores);
    $("d-threads-note").textContent = `of ${cores}; two are left free by default`;
  }

  try {
    const info = await invoke("engine_info");
    $("engine-line").textContent = `${info.version}\n${info.path}`;
    $("engine-line").title = `${info.version} — found via ${info.source}\n${info.path}`;
    state.engineOk = true;
  } catch (e) {
    $("engine-line").textContent = "engine not found";
    banner($("engine-error"), String(e));
    $("start").disabled = true;
    $("build-library").disabled = true;
  }

  try {
    const list = await invoke("presets");
    state.presets = list.map((p) => ({ name: p.name, path: p.path }));
    const sel = $("preset");
    for (const p of list) {
      const o = document.createElement("option");
      o.value = p.path;
      o.textContent = p.name;
      sel.appendChild(o);
    }
    // The preset the published benchmarks use, and the one a DIA-NN library needs.
    const preferred = [...sel.options].find((o) => o.textContent === DEFAULT_PRESET);
    if (preferred) sel.value = preferred.value;
  } catch {
    /* Presets are a convenience; engine defaults remain available without them. */
  }

  for (const b of document.querySelectorAll(".nav")) {
    b.addEventListener("click", () => {
      if (b.disabled) return;
      screen(b.dataset.screen);
      if (b.dataset.screen === "setup") {
        refreshComponents();
        refreshDiann();
        refreshThermo();
        refreshStorage();
        refreshManaged();
      }
      if (b.dataset.screen === "settings") {
        loadSettings();
        refreshSetups();
      }
      if (b.dataset.screen === "history") loadHistory();
    });
  }
  for (const t of document.querySelectorAll(".tab")) {
    t.addEventListener("click", () => setMode(t.dataset.mode));
  }
  await initMods();
  await initCleavage();
  initSearchOptions();
  for (const r of document.querySelectorAll('input[name="runmode"]')) {
    r.addEventListener("change", () => {
      state.runMode = r.value;
      refreshNormOptions();
    });
  }
  $("norm-scope").addEventListener("change", refreshNormOptions);
  // A half-width the user typed is theirs: a later probe no longer replaces it. The precursor
  // range the recipe sets follows it.
  $("ztscan-hw").addEventListener("input", () => {
    state.ztscan.hwTyped = true;
    syncZtscanLibrary();
  });
  // Once the user ticks or clears the ZT Scan analysis, the choice is theirs.
  $("ztscan-analysis").addEventListener("change", () => {
    state.ztscan.analysisTouched = true;
    state.ztscan.analysisChoice = $("ztscan-analysis").checked;
    renderZtscanNote();
  });
  // Setups, on the Settings screen (#297).
  $("setups-save").addEventListener("click", saveNamedSetup);
  $("setups-export").addEventListener("click", exportCurrentSetup);
  $("setups-import").addEventListener("click", importSetupFile);
  $("setups-apply").addEventListener("click", applyPendingSetup);
  $("setups-cancel").addEventListener("click", cancelPendingSetup);
  $("suggest-conditions").addEventListener("click", () => {
    state.conditionTyped = {};
    renderMzmlList();
  });
  for (const r of document.querySelectorAll('input[name="libsrc"]')) {
    r.addEventListener("change", () => {
      state.libSrcChosen = true;
      selectLibSrc(r.value);
      if (r.value === "diann") refreshLibSrcNote();
    });
  }
  for (const r of document.querySelectorAll('input[name="libin"]')) {
    r.addEventListener("change", () => selectLibIn(r.value));
  }
  $("diann-import").addEventListener("click", importDiannLibrary);
  // The built-in path's predictors depend on the preset as well as on the files.
  $("preset").addEventListener("change", scheduleBuiltinPlan);
  for (const b of document.querySelectorAll("[data-pick]")) {
    b.addEventListener("click", () => pick(b.dataset.pick));
  }
  $("start").addEventListener("click", start);
  $("build-library").addEventListener("click", buildLibraryOnly);
  $("lib-for").addEventListener("change", () => {
    refreshLibForNote();
    scheduleBuiltinPlan();
  });
  refreshLibForNote();
  $("lib-search").addEventListener("click", searchWithLibrary);
  $("lib-copy").addEventListener("click", copyLibrary);
  $("lib-reveal").addEventListener("click", () => {
    const L = state.libraryResult;
    if (L) invoke("reveal", { path: L.dir }).catch((e) => banner($("lib-error"), String(e)));
  });
  $("install-primary").addEventListener("click", () => installComponents("primary"));
  $("install-ms2pip").addEventListener("click", () => installComponents("ms2pip"));
  $("install-visdia").addEventListener("click", () => installComponents("visdia"));
  $("recheck-primary").addEventListener("click", () => recheckComponents("primary"));
  $("vcrt-install").addEventListener("click", installVcRuntime);
  $("open-visdia").addEventListener("click", (e) =>
    openVisdia(state.outDir, e.currentTarget, $("visdia-error"), state.picks.fasta)
  );
  $("install-thermo").addEventListener("click", installThermo);
  $("msconvert-get").addEventListener("click", () =>
    invoke("open_url", { url: "https://proteowizard.sourceforge.io/" }).catch((e) =>
      banner($("setup-error"), String(e))
    )
  );
  $("diann-locate").addEventListener("click", locateDiann);
  $("diann-download").addEventListener("click", downloadDiann);
  $("diann-ack").addEventListener("change", async (e) => {
    try {
      await invoke("diann_acknowledge", { accepted: e.target.checked });
    } catch (err) {
      banner($("setup-error"), String(err));
    }
    refreshDiann();
  });
  $("managed-reveal").addEventListener("click", () =>
    invoke("reveal", { path: state.managedDir }).catch((e) =>
      banner($("managed-error"), String(e))
    )
  );
  refreshComponents();
  refreshDiann();
  refreshThermo();
  // Setup is the first screen, so its storage and managed-data cards are filled from the
  // start.
  $("storage-change").addEventListener("click", chooseStorage);
  $("history-add").addEventListener("click", addHistoryFolder);
  $("storage-reset").addEventListener("click", () => setStorage(null));
  // With a folder chosen, the checkbox applies at once; without one, with the next choice.
  $("storage-move-envs").addEventListener("change", () => {
    if (state.storage && state.storage.configured) setStorage(state.storage.folder);
  });
  refreshStorage();
  refreshManaged();
  $("stop").addEventListener("click", stop);
  $("another").addEventListener("click", () => screen("input"));
  $("open-folder").addEventListener("click", () => invoke("reveal", { path: state.outDir }));
  $("copy-cmd").addEventListener("click", async () => {
    await navigator.clipboard.writeText($("cmd").textContent);
    $("copy-cmd").textContent = "Copied";
    setTimeout(() => ($("copy-cmd").textContent = "Copy"), 1200);
  });
}


// ── settings editor ─────────────────────────────────────────────────────────
// The form is generated from configs/config-schema.json, which the engine's own
// documentation generator emits from config.rs. Nothing about a setting -- its
// name, type, default or help text -- is written here, so this cannot describe a
// parameter the engine does not have.
//
// The one exception is which settings count as common: the handful a search may
// reasonably change, shown first under a plain name. Everything else stays in the
// in-depth section. A path listed here that the schema does not have is skipped, so a
// renamed setting drops out of the list rather than breaking the screen.
const COMMON_SETTINGS = [
  ["rescore.classifier", "Rescoring model"],
  ["quant.q_threshold", "FDR threshold of the reports"],
  ["extract.prec_tol_ppm", "Precursor (MS1) tolerance, ppm"],
  ["search_seed.fragment_tol_ppm", "Fragment tolerance of the first pass, ppm"],
  ["rt_im_train.rt_window_multiplier", "Retention-time window width"],
  ["rt_im_train.library_irt", "Where the library's retention times come from"],
  ["rt_im_train.multihead_calibration", "DeepLC multi-head calibration"],
  ["experiment.rt_library_scope", "Adapt library retention times once or per run"],
  ["extract.gate_min_score", "Extraction gate threshold"],
  ["compete.group_by", "Competition unit"],
  ["mbr.strategy", "Match-between-runs"],
  ["rescore.seeds", "Rescoring seeds"],
];

// Load the schema and seed the form once. A setup can be saved or loaded while the first
// load is still under way, so a second call waits for the first rather than starting
// another (which would wire every listener twice).
async function loadSettings() {
  if (state.schema) return;
  if (!state.settingsLoading) {
    state.settingsLoading = loadSettingsOnce().finally(() => {
      state.settingsLoading = null;
    });
  }
  return state.settingsLoading;
}

async function loadSettingsOnce() {
  let schema;
  try {
    schema = await invoke("config_schema");
  } catch (e) {
    banner($("settings-error"), String(e));
    return;
  }
  state.schema = schema;
  await seedOverridesFromPreset();
  $("settings-search").addEventListener("input", renderSettings);
  $("only-changed").addEventListener("change", renderSettings);
  $("save-settings").addEventListener("click", saveSettings);
  $("settings-advanced").addEventListener("toggle", (e) => {
    // Remember only a choice made by hand; a search opens it without changing that.
    if (!$("settings-search").value.trim() && !$("only-changed").checked) {
      state.advancedOpen = e.target.open;
    }
  });
  // A preset chosen on the Search screen is the new starting point here. Edits made
  // before the switch are dropped, and the banner says so.
  $("preset").addEventListener("change", async () => {
    const had = Object.keys(state.overrides).length;
    await seedOverridesFromPreset();
    if (had) banner($("settings-saved"), "Settings reset to the newly selected preset.");
    renderSettings();
  });
  renderSettings();
}

// Start from the preset selected on the Search screen, not from the engine defaults.
//
// A saved settings file used to hold only what was edited on this screen, relative
// to the engine defaults, so saving with a preset selected silently dropped the
// preset's own choices: predictor, rescorer, interpreters. Seeding the overrides with
// the preset's content makes the saved file the preset plus the edits.
async function seedOverridesFromPreset() {
  state.overrides = {};
  const path = $("preset").value;
  // Which built-in preset these values start from: "My settings" keeps the one it was
  // saved from, so a setup can name it.
  state.settingsPreset =
    path && path === state.savedConfig ? state.savedPreset : presetNameOf(path);
  if (path) {
    try {
      state.overrides = await seededOverrides(path);
    } catch (e) {
      banner($("settings-error"), `Could not read the selected preset: ${e}`);
    }
  }
  state.inUse = { ...state.overrides };
}

/// The override set a configuration file amounts to, as the form holds it: the settings
/// the schema describes, and only those that differ from the engine defaults.
async function seededOverrides(path) {
  const out = {};
  if (!path) return out;
  const flat = await invoke("config_overrides", { path });
  for (const [k, v] of Object.entries(flat)) {
    const f = state.schema.fields.find((x) => x.path === k);
    if (!f || sameAsDefault(f, v)) continue;
    out[k] = v;
  }
  return out;
}

/// Whether `value` is the engine default for `f`. List-valued settings carry their
/// default as JSON text in the schema, so both sides are compared as parsed JSON.
function sameAsDefault(f, value) {
  let dflt = f.default;
  if (typeof dflt === "string" && f.kind === "other") {
    try {
      dflt = JSON.parse(dflt);
    } catch {
      /* a plain text default stays text */
    }
  }
  return JSON.stringify(value) === JSON.stringify(dflt);
}

/// A value as the text input shows it: lists and objects as JSON, scalars as they are.
function displayValue(v) {
  if (v === null || v === undefined) return "";
  return typeof v === "object" ? JSON.stringify(v) : v;
}

/// The value currently shown for a setting: an override if one was typed, else the
/// engine's default.
function currentValue(f) {
  return f.path in state.overrides ? state.overrides[f.path] : f.default;
}

const escHtml = (t) =>
  String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
           .replace(/"/g, "&quot;");

/// One setting as a row: its control, its markers and the engine's help text. `label`
/// is the plain name the common section puts above the path; the in-depth section
/// shows the path alone.
function settingRow(f, label) {
  const esc = escHtml;
  const changed = f.path in state.overrides;
  const v = currentValue(f);
  let control;
  if (f.kind === "bool") {
    control =
      `<select data-path="${esc(f.path)}">` +
      `<option value="true"${v === true ? " selected" : ""}>true</option>` +
      `<option value="false"${v === false ? " selected" : ""}>false</option></select>`;
  } else if (f.kind === "enum" && f.choices) {
    control =
      `<select data-path="${esc(f.path)}">` +
      f.choices
        .map((c) => `<option value="${esc(c)}"${String(c) === String(v) ? " selected" : ""}>${esc(c)}</option>`)
        .join("") +
      `</select>`;
  } else {
    control = `<input type="text" data-path="${esc(f.path)}" value="${esc(displayValue(v))}">`;
  }
  // A field the engine accepts but does not act on yet is shown as exactly that,
  // and cannot be edited: a value typed there would be saved, validated and then
  // ignored by the run, which is the one outcome worse than not offering it.
  // A gated parameter is one the project documents as not to be changed from a
  // single benchmark count. Saying so where the decision is made is the whole
  // reason the schema carries the marker.
  const unwired = f.gates.some((g) => g.includes("wired"));
  if (unwired) control = control.replace(/^<(select|input)/, "<$1 disabled");
  const gate = unwired
    ? `<span class="pill warn" title="${esc(f.gates.join(", "))}">not wired yet</span>`
    : f.gates.length
      ? `<span class="pill warn" title="${esc(f.gates.join(", "))}">gated</span>`
      : "";
  const badge = changed ? `<span class="pill info">changed</span>` : "";
  const name = label
    ? `<b>${esc(label)}</b> <span class="spath">${esc(f.path)}</span>`
    : esc(f.path);
  return (
    `<div class="setting${changed ? " changed" : ""}">` +
    `<div class="sname">${name} ${gate} ${badge}</div>` +
    `<div class="shelp">${esc(f.help || "No description in the engine source.")}` +
    (f.default !== null && f.default !== undefined ? ` <em>Default: ${esc(f.default)}</em>` : "") +
    `</div><div class="sctl">${control}</div></div>`
  );
}

function renderSettings() {
  const q = $("settings-search").value.trim().toLowerCase();
  const onlyChanged = $("only-changed").checked;
  const matches = (f) =>
    (!onlyChanged || f.path in state.overrides) &&
    (!q || f.path.toLowerCase().includes(q) || (f.help || "").toLowerCase().includes(q));

  // Common: the curated list, in its own order, under plain names.
  const byPath = new Map(state.schema.fields.map((f) => [f.path, f]));
  const commonPaths = new Set();
  const commonRows = [];
  for (const [path, label] of COMMON_SETTINGS) {
    const f = byPath.get(path);
    if (!f) continue;
    commonPaths.add(path);
    if (matches(f) || (q && label.toLowerCase().includes(q))) commonRows.push(settingRow(f, label));
  }

  // In-depth: everything else, grouped by section as the schema orders it.
  const bySection = new Map();
  for (const f of state.schema.fields) {
    if (commonPaths.has(f.path) || !matches(f)) continue;
    const sec = f.section || "(top level)";
    if (!bySection.has(sec)) bySection.set(sec, []);
    bySection.get(sec).push(f);
  }
  const parts = [];
  for (const [sec, fields] of bySection) {
    parts.push(`<div class="sec-title">${escHtml(sec)}</div><div class="card">`);
    for (const f of fields) parts.push(settingRow(f));
    parts.push(`</div>`);
  }

  const common = $("settings-common");
  common.innerHTML = commonRows.length
    ? `<div class="card">${commonRows.join("")}</div>`
    : `<p class="hint">${q || onlyChanged ? "None of the common settings match." : ""}</p>`;
  const list = $("settings-list");
  list.innerHTML = parts.join("") || `<p class="hint">Nothing else matches.</p>`;

  // A search or "only changed" looks through everything, so the in-depth section opens
  // for it; otherwise it stays as the user left it.
  $("settings-advanced").open = state.advancedOpen || !!q || onlyChanged;

  for (const host of [common, list]) {
    for (const el of host.querySelectorAll("[data-path]")) {
      el.addEventListener("change", () => onSettingChanged(el));
    }
  }

  const n = Object.keys(state.overrides).length;
  const presetName = $("preset").selectedOptions[0]?.textContent || "Engine defaults";
  $("settings-sub").textContent =
    `${state.schema.fields.length} settings, starting from "${presetName}" (chosen on the ` +
    `Search screen). ` +
    (n
      ? `${n} differ from the engine defaults (the preset's own values included); those are what is saved.`
      : "None differ from the engine defaults.");
}

/// Record a change, or drop it when the value returns to the default.
///
/// Dropping matters: a saved configuration is meant to be the difference from the
/// defaults, and a value typed and then typed back should leave nothing behind.
function onSettingChanged(el) {
  const path = el.dataset.path;
  const f = state.schema.fields.find((x) => x.path === path);
  if (!f) return;
  let raw = el.value;
  let value = raw;
  if (f.kind === "bool") value = raw === "true";
  // A choice spelled `true` or `false` is a JSON boolean to the engine (the ZT Scan switches,
  // `quant.baseline_subtract`, `extract.ms1_calibrate`), not the string.
  else if (f.kind === "enum" && (raw === "true" || raw === "false")) value = raw === "true";
  else if (f.kind === "integer" || f.kind === "float") {
    // An OPTIONAL number left blank means "unset", which is a real value for these
    // (`rt_im_train.multihead_calibration` unset is what turns the automatic default
    // on). A required one left blank is still an error.
    if (raw.trim() === "" && f.optional) {
      value = null;
    } else {
      const n = Number(raw);
      if (raw.trim() === "" || Number.isNaN(n)) {
        banner($("settings-error"), `${path} must be a number.`);
        return;
      }
      value = n;
    }
  } else if (f.kind === "other" && /^\s*[\[{]/.test(raw)) {
    // A list or object typed as JSON (the modification lists, for instance).
    try {
      value = JSON.parse(raw);
    } catch {
      banner($("settings-error"), `${path} must be valid JSON.`);
      return;
    }
  }
  banner($("settings-error"), "");
  if (sameAsDefault(f, value)) delete state.overrides[path];
  else state.overrides[path] = value;
  renderSettings();
}

// True when the settings were saved and are used by the next search.
async function saveSettings() {
  banner($("settings-error"), "");
  banner($("settings-saved"), "");
  try {
    const used = await useSettings(state.overrides, state.settingsPreset);
    banner(
      $("settings-saved"),
      used.asPreset
        ? `No setting differs from ${presetText(used.presetName)}, so the next search uses it as it is.`
        : `Saved and accepted by the engine: the selected preset plus your edits. The next search uses this file.`
    );
    return true;
  } catch (e) {
    banner($("settings-error"), String(e));
    return false;
  }
}

// Make `overrides` (the Settings screen's values, relative to the engine defaults, starting
// from the built-in preset `presetName`) the settings the next search uses: the preset
// itself when they are exactly its values, so the search reads the preset file as it would
// had nothing been edited; otherwise "My settings", written and checked by the engine.
// Throws the engine's reason when it refuses them, having changed nothing.
async function useSettings(overrides, presetName) {
  presetName = presetName || "";
  const presetPath = presetPathOf(presetName);
  const sel = $("preset");
  let asPreset = false;
  if (presetPath !== null && sameOverrides(overrides, await seededOverrides(presetPath))) {
    sel.value = presetPath;
    asPreset = true;
  } else {
    const path = await invoke("save_settings", { name: "console", overrides });
    state.savedConfig = path;
    state.savedPreset = presetName;
    // The saved file becomes the preset the next search uses, so the settings on
    // this screen and the settings a run uses cannot diverge.
    const options = sel.options ? [...sel.options] : [];
    if (!options.some((o) => o.value === path)) {
      const opt = document.createElement("option");
      opt.value = path;
      opt.textContent = "My settings";
      sel.appendChild(opt);
    }
    sel.value = path;
  }
  state.overrides = { ...overrides };
  state.settingsPreset = presetName;
  state.inUse = { ...overrides };
  return { asPreset, presetName };
}

// Edits on the Settings screen that the next search would not use, because they were not
// saved and used.
function settingsUnsaved() {
  return !sameOverrides(state.overrides, state.inUse);
}

// ── setups ──────────────────────────────────────────────────────────────────
// A setup is how a search is set up, kept under a name or written to a file so it can be
// used again, moved to another computer or sent to a colleague (#297). The file's
// envelope, its format version and the named setups are the backend's (`setups.rs`); what
// is in it is read from the screens and put back here, because this is where their state
// lives:
//
// - the Search screen's choices, without its files, folders, conditions and thread counts;
// - the Settings screen's engine settings, as the built-in preset they start from, the
//   settings changed from it and the preset's settings returned to the engine default,
//   without the settings that describe a computer rather than a search (`LOCAL_SETTINGS`).
//
// Loading one is checked first and shown as the list of what it changes, and only Apply
// changes anything. A value this version cannot use refuses the whole setup, naming the
// field; a setting it does not know is listed and left out.

// Engine settings that describe a computer, or one dataset, rather than a search: where the
// programs and caches are, how much of the machine a search may use, and the conditions of a
// set of files by run name. A setup leaves them out, and loading one keeps this computer's
// own values.
const LOCAL_SETTINGS = new Set([
  "convert.thermo_raw_parser",
  "convert.msconvert",
  "predict_frag.sidecar_script_dir",
  "predict_frag.library_cache",
  "rt_im_train.deeplc_projection_cache",
  "rescore.percolator_bin",
  "rescore.features_file",
  "convert.parallel_conversions",
  "extract.windows_in_flight",
  "extract.max_footprint_gib",
  "features.chrom_loaders",
  "rescore.max_feature_matrix_gib",
  "rt_im_train.deeplc_predict_shards",
  "experiment.parallel_runs",
  "experiment.overlap_front_threads",
  "experiment.conditions",
]);

// The interpreters. A path is this computer's, but a search that needs an interpreter cannot
// do without one (the engine refuses `rescore.classifier = nn_torch` without `rescore.python`,
// MS2PIP, DeepLC and IM2Deep without theirs), so a setup carries "auto" for a path, which
// finds the interpreter on the computer the setup is loaded on, and loading one keeps an
// interpreter this computer named itself.
const INTERPRETER_SETTINGS = new Set([
  "predict_frag.ms2pip_python",
  "predict_frag.peptdeep_python",
  "predict_frag.deeplc_python",
  "predict_frag.im2deep_python",
  "rescore.python",
  "mbr.python",
]);

// A setting's value as a setup carries it: an interpreter's path as "auto".
function portableValue(path, v) {
  return INTERPRETER_SETTINGS.has(path) && typeof v === "string" && v !== "auto" ? "auto" : v;
}

// Settings whose value on this computer a loaded setup keeps, where the user set one.
function isComputersOwn(path) {
  return LOCAL_SETTINGS.has(path) || INTERPRETER_SETTINGS.has(path);
}

// Every setting a setup holds outside the engine settings, with how it is checked and the
// name the summary uses. `select`, `radio` and `id` name the control on the Search screen
// (`ci/test_desktop_ui.mjs` checks the values against index.html); `names` are the words
// for a value the summary shows when the control does not say them.
const SETUP_SPEC = [
  { path: "search_space.source", kind: "choice", values: ["fasta", "library"], label: "Search space",
    tabs: true, names: { fasta: "From a FASTA", library: "From a spectral library" } },
  { path: "search_space.library_input", kind: "choice", values: ["tables", "diann"],
    label: "Spectral library input", radio: "libin",
    names: { tables: "MuMDIA library tables", diann: "a DIA-NN library" } },
  { path: "search_space.library_build", kind: "choice", values: ["diann", "builtin"],
    label: "Library from a FASTA built with", radio: "libsrc",
    names: { diann: "DIA-NN", builtin: "the built-in predictors" } },
  { path: "search_space.library_for", kind: "choice", values: ["other", "diapasef"],
    label: "Library for", select: "lib-for" },
  { path: "digest.enzyme", kind: "enzyme", label: "Cleavage", select: "d-enzyme" },
  { path: "digest.cleavage_rule", kind: "text", label: "Cleavage rule", id: "d-rule" },
  { path: "digest.missed_cleavages", kind: "int", min: 0, label: "Missed cleavages", id: "d-missed" },
  { path: "digest.min_length", kind: "int", min: 1, label: "Shortest peptide", id: "d-minlen" },
  { path: "digest.max_length", kind: "int", min: 1, label: "Longest peptide", id: "d-maxlen" },
  { path: "digest.min_charge", kind: "int", min: 1, label: "Lowest charge", id: "d-minz" },
  { path: "digest.max_charge", kind: "int", min: 1, label: "Highest charge", id: "d-maxz" },
  { path: "digest.precursor_mz_min", kind: "mz", label: "Lowest precursor m/z", id: "d-prmin" },
  { path: "digest.precursor_mz_max", kind: "mz", label: "Highest precursor m/z", id: "d-prmax" },
  { path: "digest.fragment_mz_min", kind: "mz", label: "Lowest fragment m/z", id: "d-frmin" },
  { path: "digest.fragment_mz_max", kind: "mz", label: "Highest fragment m/z", id: "d-frmax" },
  { path: "modifications.fixed", kind: "sites", label: "Fixed modifications" },
  { path: "modifications.variable", kind: "sites", label: "Variable modifications" },
  { path: "modifications.max_variable", kind: "int", min: 0, label: "Variable modifications per peptide",
    id: "d-maxvar" },
  { path: "experiment.mode", kind: "choice", values: ["experiment", "separate"], label: "Several files",
    radio: "runmode", names: { experiment: "One experiment", separate: "Search each separately" } },
  { path: "experiment.normalisation", kind: "choice", values: ["none", "all", "condition"],
    label: "Normalisation", select: "norm-scope" },
  { path: "prescreen.mode", kind: "choice", values: ["off", "before", "after", "tags"], label: "Prescreen",
    select: "prescreen" },
  { path: "prescreen.strength", kind: "choice", values: ["light", "balanced", "stringent"],
    label: "Prescreen strength", select: "prescreen-strength" },
  { path: "bands.mode", kind: "choice", values: ["auto", "off", "custom"], label: "Bands", select: "bands" },
  { path: "bands.count", kind: "int", min: 1, label: "Custom bands", id: "bands-n" },
  { path: "bands.at_a_time", kind: "int", min: 1, label: "Custom bands at a time", id: "bands-parallel" },
  { path: "ztscan.analysis", kind: "bool", label: "ZT Scan analysis",
    names: { true: "on for ZT Scan files (the default)", false: "off" } },
  { path: "ztscan.q1_half_width", kind: "positive", label: "ZT Scan Q1 half-width (m/z)",
    id: "ztscan-hw", empty: "from each file" },
  { path: "engine.preset", kind: "preset", label: "Preset" },
  { path: "engine.changes", kind: "changes", label: "Engine settings changed from the preset" },
  { path: "engine.reset", kind: "reset", label: "Preset settings returned to the engine default" },
];

const SETUP_FILTERS = [{ name: "MuMDIA settings", extensions: ["json"] }];

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

// JSON with every object's keys sorted, so two values compare equal whatever their order.
function canonical(v) {
  if (v === undefined) return undefined;
  return JSON.stringify(v, (_k, x) =>
    isObject(x) ? Object.fromEntries(Object.keys(x).sort().map((k) => [k, x[k]])) : x
  );
}

function sameOverrides(a, b) {
  return canonical(a || {}) === canonical(b || {});
}

function getPath(obj, path) {
  let cur = obj;
  for (const part of path.split(".")) {
    if (!isObject(cur) || !(part in cur)) return undefined;
    cur = cur[part];
  }
  return cur;
}

// `{"rescore.seeds": 3}` as the nested object a configuration file holds.
function nestPaths(flat) {
  const root = {};
  for (const [path, value] of Object.entries(flat)) {
    const parts = path.split(".");
    let cur = root;
    for (const part of parts.slice(0, -1)) {
      if (!isObject(cur[part])) cur[part] = {};
      cur = cur[part];
    }
    cur[parts[parts.length - 1]] = value;
  }
  return root;
}

// The inverse of `nestPaths`, which stops at a setting the schema knows, so an object-valued
// setting stays one value.
function flattenConfig(obj, known, prefix = "", out = {}) {
  for (const [k, v] of Object.entries(obj || {})) {
    const path = prefix ? `${prefix}.${k}` : k;
    if (isObject(v) && !known.has(path)) flattenConfig(v, known, path, out);
    else out[path] = v;
  }
  return out;
}

function schemaPaths() {
  return new Set((state.schema?.fields || []).map((f) => f.path));
}

function fieldOf(path) {
  return (state.schema?.fields || []).find((f) => f.path === path);
}

// A built-in preset's path from its name: "" for the engine defaults, null for a name this
// application does not offer.
function presetPathOf(name) {
  if (!name) return "";
  const p = state.presets.find((x) => x.name === name);
  return p ? p.path : null;
}

function presetNameOf(path) {
  if (!path) return "";
  return state.presets.find((x) => x.path === path)?.name || "";
}

function presetText(name) {
  return name ? `the preset "${name}"` : "the engine defaults";
}

// The ZT Scan analysis as a setup holds it: on (the engine's default for ZT Scan files)
// unless the user cleared it.
function ztscanAnalysisSetting() {
  const z = state.ztscan;
  if (!z.analysisTouched) return true;
  if (typeof z.analysisChoice === "boolean") return z.analysisChoice;
  return !(ztscanAnalysisAllowed() && !$("ztscan-analysis").checked);
}

// The Search screen as a setup holds it. The ZT Scan analysis sets the library's search
// space while it is on; what it set is read as the value it replaced (and is set again
// when the setup is loaded for ZT Scan files), so a setup holds the user's own choices.
function captureSearch() {
  const z = state.ztscan;
  const own = (id) => {
    const el = $(id);
    if (!el) return undefined;
    if (z.lib && id in z.lib.set && el.value === z.lib.set[id]) return z.lib.before[id] ?? "";
    return el.value;
  };
  const d = libraryParams("", own);
  const mods = new Map(state.mods);
  for (const key of z.lib?.mods || []) if (mods.get(key) === "var") mods.delete(key);
  const { fixed, variable } = selectedMods(mods);
  const count = (id) => {
    const v = parseInt($(id).value, 10);
    return Number.isFinite(v) && v > 0 ? v : 1;
  };
  const hw = parseFloat($("ztscan-hw").value);
  return {
    search_space: {
      source: state.mode,
      library_input: state.libIn,
      // DIA-NN unless the user chose the built-in predictors: a fallback while DIA-NN
      // cannot be used is this computer's, not the setup's.
      library_build: state.libSrcChosen ? state.libSrc : "diann",
      library_for: $("lib-for").value || "other",
    },
    digest: {
      enzyme: d.enzyme,
      cleavage_rule: d.enzyme === "custom" ? d.cleavage_rule || null : null,
      missed_cleavages: d.missed_cleavages,
      min_length: d.min_pep_len,
      max_length: d.max_pep_len,
      min_charge: d.min_charge,
      max_charge: d.max_charge,
      precursor_mz_min: d.min_pr_mz,
      precursor_mz_max: d.max_pr_mz,
      fragment_mz_min: d.min_fr_mz,
      fragment_mz_max: d.max_fr_mz,
    },
    modifications: { fixed, variable, max_variable: maxVariableMods(own("d-maxvar")) },
    experiment: { mode: state.runMode, normalisation: $("norm-scope").value },
    prescreen: { mode: $("prescreen").value, strength: $("prescreen-strength").value },
    bands: { mode: $("bands").value, count: count("bands-n"), at_a_time: count("bands-parallel") },
    ztscan: {
      analysis: ztscanAnalysisSetting(),
      q1_half_width: z.hwTyped && Number.isFinite(hw) && hw > 0 ? hw : null,
    },
  };
}

// The Settings screen's values as a setup holds them: the preset they start from, the
// settings that differ from it and the preset's settings returned to the engine default.
// `local` names the settings left out because they describe this computer, `paths` the
// interpreters whose path was written as "auto".
async function captureEngine() {
  let presetName = state.settingsPreset || "";
  let presetPath = presetPathOf(presetName);
  if (presetPath === null) {
    // The preset these values started from is no longer offered: they are written out in
    // full, relative to the engine defaults.
    presetName = "";
    presetPath = "";
  }
  const base = await seededOverrides(presetPath);
  const changes = {};
  const reset = [];
  const local = [];
  const paths = [];
  const keys = [...new Set([...Object.keys(state.overrides), ...Object.keys(base)])].sort();
  for (const k of keys) {
    const has = k in state.overrides;
    const value = has ? portableValue(k, state.overrides[k]) : undefined;
    if (has && value !== state.overrides[k]) paths.push(k);
    if (has && canonical(value) === canonical(base[k])) continue;
    if (LOCAL_SETTINGS.has(k)) {
      local.push(k);
      continue;
    }
    if (has) changes[k] = value;
    else reset.push(k);
  }
  return { engine: { preset: presetName || null, changes: nestPaths(changes), reset }, local, paths };
}

// The current setup, for saving it: the Settings screen's edits are saved and used first
// when they are not yet, so a setup is always what the next search here would run.
async function currentSetup() {
  await loadSettings();
  if (!state.schema) throw "The settings could not be read, so a setup cannot be saved yet.";
  const savedEdits = settingsUnsaved();
  if (savedEdits) {
    try {
      await useSettings(state.overrides, state.settingsPreset);
    } catch (e) {
      throw `The changes on the Settings screen were refused by the engine, so no setup was saved: ${e}`;
    }
  }
  const { engine, local, paths } = await captureEngine();
  const settings = { engine, ...captureSearch() };
  // The same check a load makes, so a setup that is saved can always be loaded.
  const v = validateSetup(settings);
  if (v.errors.length) {
    throw `This setup cannot be saved: ${v.errors.join(" ")} Correct it on the Search screen first.`;
  }
  return { settings: v.setup, local, paths, savedEdits };
}

function siteText(s) {
  return s.position ? `${s.name} (${TERMINI[s.position]?.text || s.position})` : `${s.name} (${s.residue})`;
}

// Check one setting of a setup; `{ value }` (normalised) or `{ error }`.
function checkSetupValue(e, v, where, out) {
  const name = `${where} (${e.label})`;
  const not = (what) => ({ error: `${name} must be ${what}, not ${JSON.stringify(v)}.` });
  switch (e.kind) {
    case "choice":
      return e.values.includes(v)
        ? { value: v }
        : not(`one of ${e.values.map((x) => JSON.stringify(x)).join(", ")}`);
    case "int":
      return Number.isSafeInteger(v) && v >= e.min ? { value: v } : not(`a whole number of ${e.min} or more`);
    case "mz":
      return v === null || (typeof v === "number" && Number.isFinite(v) && v > 0)
        ? { value: v }
        : not("a positive number, or null for no limit");
    case "positive":
      return v === null || (typeof v === "number" && Number.isFinite(v) && v > 0)
        ? { value: v }
        : not("a positive number, or null");
    case "bool":
      return typeof v === "boolean" ? { value: v } : not("true or false");
    case "text":
      return v === null || typeof v === "string"
        ? { value: typeof v === "string" && v.trim() ? v.trim() : null }
        : not("text, or null");
    case "enzyme":
      return ENZYMES.some((x) => x.name === v)
        ? { value: v }
        : not(`a cleavage this version knows (${ENZYMES.map((x) => x.name).join(", ")})`);
    case "preset": {
      if (v === null || v === "") return { value: null };
      if (typeof v === "string" && presetPathOf(v) !== null) return { value: v };
      const offered = state.presets.map((p) => p.name).join(", ") || "none";
      return { error: `${name} names ${JSON.stringify(v)}, a preset this version does not offer (it offers: ${offered}).` };
    }
    case "sites": {
      if (!Array.isArray(v)) return not("a list of modification sites");
      const known = new Set((state.catalogue?.modifications || []).map((m) => m.name));
      const sites = [];
      for (const [i, s] of v.entries()) {
        const at = `${where}[${i}]`;
        if (!isObject(s)) return { error: `${at} must be an object with a residue and a name.` };
        for (const k of Object.keys(s)) {
          if (!["residue", "name", "position"].includes(k)) out.unknown.push(`${at}.${k}`);
        }
        if (typeof s.name !== "string" || !known.has(s.name)) {
          return { error: `${at}: ${JSON.stringify(s.name)} is not a modification this version knows.` };
        }
        const position = s.position ?? "";
        const ok = position
          ? s.residue === "*" && position in TERMINI
          : typeof s.residue === "string" && /^[A-Z]$/.test(s.residue);
        if (!ok) {
          return {
            error:
              `${at}: ${s.name} at ${JSON.stringify(s.residue)}` +
              (position ? ` ${JSON.stringify(position)}` : "") +
              " is not a site: a residue is one capital letter, a terminus is the residue \"*\" " +
              `with the position ${Object.keys(TERMINI).map((t) => JSON.stringify(t)).join(" or ")}.`,
          };
        }
        sites.push(position ? { residue: "*", name: s.name, position } : { residue: s.residue, name: s.name });
      }
      // The order the screen writes them in (`selectedMods`), each site once.
      const map = new Map(sites.map((s) => [modKey(s.name, s.residue, s.position), "var"]));
      return { value: selectedMods(map).variable };
    }
    case "changes": {
      if (!isObject(v)) return not("an object of engine settings");
      const known = schemaPaths();
      const flat = flattenConfig(v, known);
      const kept = {};
      for (const [path, value] of Object.entries(flat)) {
        const f = fieldOf(path);
        if (!f) {
          out.unknown.push(`${where}.${path}`);
          continue;
        }
        if (LOCAL_SETTINGS.has(path)) {
          out.local.push(path);
          continue;
        }
        const problem = engineValueProblem(f, value);
        if (problem) return { error: `${where}.${path} ${problem}, not ${JSON.stringify(value)}.` };
        const portable = portableValue(path, value);
        if (portable !== value) out.paths.push(path);
        kept[path] = normaliseEngineValue(f, portable);
      }
      return { value: nestPaths(kept) };
    }
    case "reset": {
      if (!Array.isArray(v) || v.some((x) => typeof x !== "string")) {
        return not("a list of engine setting names");
      }
      const out2 = [];
      for (const path of v) {
        if (!fieldOf(path)) out.unknown.push(`${where}: ${path}`);
        else if (LOCAL_SETTINGS.has(path)) out.local.push(path);
        else if (!out2.includes(path)) out2.push(path);
      }
      return { value: out2.sort() };
    }
    default:
      return { error: `${name} cannot be read.` };
  }
}

// Why `v` is not a value of the engine setting `f`, or null. The engine has the last word:
// a setup whose engine settings change is checked by it before it is shown.
function engineValueProblem(f, v) {
  const nullable = f.optional || f.default === null || f.default === "null" || f.kind === "other";
  if (v === null) return nullable ? null : "cannot be empty";
  switch (f.kind) {
    case "bool":
      return typeof v === "boolean" ? null : "must be true or false";
    case "integer":
      return Number.isSafeInteger(v) ? null : "must be a whole number";
    case "float":
      return typeof v === "number" && Number.isFinite(v) ? null : "must be a number";
    case "enum": {
      const choices = (f.choices || []).map(String);
      return choices.includes(String(v)) && (typeof v === "string" || typeof v === "boolean")
        ? null
        : `must be one of ${choices.join(", ")}`;
    }
    case "string":
      return typeof v === "string" ? null : "must be text";
    default:
      return null;
  }
}

// A choice spelled `true` or `false` is a JSON boolean to the engine, as the editor sends it.
function normaliseEngineValue(f, v) {
  if (f.kind === "enum" && (v === "true" || v === "false")) return v === "true";
  return v;
}

// Check a setup's settings against this version: the normalised setup holds every setting
// it could use; `errors` refuse it, `unknown` and `local` are reported and left out, and
// `paths` are interpreters whose path is read as "auto".
function validateSetup(settings) {
  const out = { setup: {}, errors: [], unknown: [], local: [], paths: [] };
  if (!isObject(settings)) {
    out.errors.push("Its settings are not an object.");
    return out;
  }
  const sections = new Map();
  for (const e of SETUP_SPEC) {
    const [sec, key] = e.path.split(".");
    if (!sections.has(sec)) sections.set(sec, new Map());
    sections.get(sec).set(key, e);
  }
  for (const [sec, body] of Object.entries(settings)) {
    const spec = sections.get(sec);
    if (!spec) {
      out.unknown.push(`settings.${sec}`);
      continue;
    }
    if (!isObject(body)) {
      out.errors.push(`settings.${sec} must be an object, not ${JSON.stringify(body)}.`);
      continue;
    }
    for (const [key, value] of Object.entries(body)) {
      const e = spec.get(key);
      const where = `settings.${sec}.${key}`;
      if (!e) {
        out.unknown.push(where);
        continue;
      }
      const r = checkSetupValue(e, value, where, out);
      if (r.error) out.errors.push(r.error);
      else {
        if (!out.setup[sec]) out.setup[sec] = {};
        out.setup[sec][key] = r.value;
      }
    }
  }
  const m = out.setup.modifications || {};
  if (m.fixed && m.variable) {
    const fixed = new Set(m.fixed.map((s) => modKey(s.name, s.residue, s.position)));
    const both = m.variable.filter((s) => fixed.has(modKey(s.name, s.residue, s.position)));
    if (both.length) {
      out.errors.push(
        `settings.modifications: ${both.map(siteText).join(", ")} ${both.length === 1 ? "is" : "are"} ` +
          "both fixed and variable."
      );
    }
  }
  const eng = out.setup.engine || {};
  if (eng.changes && eng.reset) {
    const changed = new Set(Object.keys(flattenConfig(eng.changes, schemaPaths())));
    const both = eng.reset.filter((p) => changed.has(p));
    if (both.length) {
      out.errors.push(`settings.engine: ${both.join(", ")} ${both.length === 1 ? "is" : "are"} both changed and reset.`);
    }
  }
  return out;
}

// What loading `doc` would do, checked: the normalised setup, the engine settings it makes
// (`overrides`, relative to the engine defaults, as the Settings screen holds them) and what
// the summary must say. Throws the reason a setup is refused, before anything changes.
async function planSetup(doc) {
  await loadSettings();
  if (!state.schema) throw "The settings could not be read, so a setup cannot be loaded yet.";
  const name = doc.name || "This setup";
  const v = validateSetup(doc.settings);
  if (v.errors.length) {
    const more = v.errors.length > 8 ? ` And ${v.errors.length - 8} more.` : "";
    throw `"${name}" cannot be loaded, so nothing was changed. ${v.errors.slice(0, 8).join(" ")}${more}`;
  }
  const warnings = [...(doc.notes || [])];
  if (v.unknown.length) {
    warnings.push(
      `${doc.newer ? "From the newer version and not known to this one" : "Not known to this version"}, ` +
        `so left out: ${v.unknown.join(", ")}.`
    );
  }
  if (v.local.length) {
    warnings.push(
      "Left out because they describe a computer rather than a search, so this one keeps its " +
        `own: ${v.local.join(", ")}.`
    );
  }
  if (v.paths.length) {
    warnings.push(
      `Interpreter paths of another computer, read as "auto", which finds this one's: ` +
        `${v.paths.join(", ")}.`
    );
  }
  const setup = v.setup;
  let engine = null;
  if (setup.engine) {
    const presetName = "preset" in setup.engine ? setup.engine.preset || "" : state.settingsPreset || "";
    const presetPath = presetPathOf(presetName);
    const base = await seededOverrides(presetPath);
    const next = { ...base };
    for (const k of setup.engine.reset || []) delete next[k];
    for (const [k, val] of Object.entries(flattenConfig(setup.engine.changes || {}, schemaPaths()))) {
      if (sameAsDefault(fieldOf(k), val)) delete next[k];
      else next[k] = val;
    }
    // This computer's own settings of that kind, where the user set them, are kept.
    const lineage = await seededOverrides(presetPathOf(state.settingsPreset || "") || "");
    const kept = [];
    for (const [k, val] of Object.entries(state.overrides)) {
      if (!isComputersOwn(k) || canonical(lineage[k]) === canonical(val)) continue;
      if (canonical(next[k]) !== canonical(val)) kept.push(k);
      next[k] = val;
    }
    if (kept.length) {
      warnings.push(`This computer's own ${kept.join(", ")} ${kept.length === 1 ? "is" : "are"} kept.`);
    }
    const mine = !sameOverrides(next, base);
    if (mine && state.engineOk) {
      try {
        await invoke("check_settings", { overrides: next });
      } catch (e) {
        throw `"${name}" cannot be loaded, so nothing was changed: the engine refuses its settings. ${e}`;
      }
    }
    if (settingsUnsaved()) {
      warnings.push("The changes on the Settings screen that were not saved and used are replaced.");
    }
    engine = { presetName, presetPath, overrides: next, mine };
  }
  if (setup.search_space?.library_build === "diann" && $("libsrc-diann").disabled) {
    warnings.push(
      "This setup builds a library from a FASTA with DIA-NN, which cannot be used on this " +
        "computer yet (see Setup): the built-in predictors are used until it can."
    );
  }
  return { setup, engine, warnings };
}

// A setup's value as the summary shows it.
function setupValueText(e, v) {
  if (v === undefined) return "not set";
  switch (e.kind) {
    case "sites":
      return v.length ? v.map(siteText).join(", ") : "none";
    case "mz":
      return v === null ? "no limit" : String(v);
    case "positive":
      return v === null ? e.empty || "none" : String(v);
    case "text":
      return v === null ? "none" : v;
    case "enzyme":
      return ENZYMES.find((x) => x.name === v)?.label || v;
    default: {
      if (e.names && String(v) in e.names) return e.names[String(v)];
      const sel = e.select ? $(e.select) : null;
      const opt = sel?.options ? [...sel.options].find((o) => o.value === v) : null;
      return opt ? opt.textContent : String(v);
    }
  }
}

function engineValueText(path, v) {
  if (v === undefined) {
    const d = fieldOf(path)?.default;
    return d === null || d === undefined ? "the engine default" : `the engine default (${displayValue(d)})`;
  }
  return typeof v === "string" ? v : JSON.stringify(v);
}

// What loading a checked setup changes, one line each.
function setupChanges(before, plan) {
  const lines = [];
  for (const e of SETUP_SPEC) {
    if (e.path.startsWith("engine.")) continue;
    const after = getPath(plan.setup, e.path);
    if (after === undefined) continue;
    const was = getPath(before, e.path);
    if (canonical(was) === canonical(after)) continue;
    lines.push(`${e.label}: ${setupValueText(e, was)} → ${setupValueText(e, after)}`);
  }
  if (plan.engine) {
    if ((plan.engine.presetName || "") !== (state.settingsPreset || "")) {
      lines.push(`Preset: ${presetText(state.settingsPreset)} → ${presetText(plan.engine.presetName)}`);
    }
    const next = plan.engine.overrides;
    const keys = [...new Set([...Object.keys(state.overrides), ...Object.keys(next)])].sort();
    for (const k of keys) {
      if (canonical(state.overrides[k]) === canonical(next[k])) continue;
      lines.push(`${k}: ${engineValueText(k, state.overrides[k])} → ${engineValueText(k, next[k])}`);
    }
  }
  return lines;
}

// Put a checked setup on the screens. The engine settings go first, because saving them is
// the one step that can still fail, and a setup that fails must change nothing.
async function applySetup(plan) {
  if (plan.engine) {
    await useSettings(plan.engine.overrides, plan.engine.presetName);
    if (state.schema) renderSettings();
  }
  await applySearchSetup(plan.setup);
}

// The Search screen part of a setup. Only the settings the setup holds are set; the rest
// stay as they are.
async function applySearchSetup(s) {
  const z = state.ztscan;
  const sp = s.search_space || {};
  if (sp.source !== undefined) setMode(sp.source);
  if (sp.library_input !== undefined) selectLibIn(sp.library_input);
  if (sp.library_build !== undefined) {
    // DIA-NN is the default and comes back by itself once it can be used.
    state.libSrcChosen = sp.library_build === "builtin";
    selectLibSrc(sp.library_build);
  }
  if (sp.library_for !== undefined) {
    $("lib-for").value = sp.library_for;
    refreshLibForNote();
  }

  // A field the setup sets is the user's own value: the ZT Scan analysis, when it applies,
  // sets its library over it again and takes it back to this value when it goes off.
  const own = (id) => {
    if (z.lib) {
      delete z.lib.set[id];
      delete z.lib.before[id];
    }
  };
  const d = s.digest || {};
  for (const e of SETUP_SPEC) {
    if (!e.path.startsWith("digest.") || !e.id || e.kind === "text") continue;
    const v = d[e.path.slice("digest.".length)];
    if (v === undefined) continue;
    $(e.id).value = v === null ? "" : String(v);
    own(e.id);
  }
  if (d.enzyme !== undefined) $("d-enzyme").value = d.enzyme;
  if (typeof d.cleavage_rule === "string") $("d-rule").value = d.cleavage_rule;
  if (d.enzyme !== undefined || d.cleavage_rule !== undefined) cleavageChanged();

  const m = s.modifications || {};
  for (const [key, kind] of [["fixed", "fix"], ["variable", "var"]]) {
    if (!m[key]) continue;
    for (const [k, v] of [...state.mods]) if (v === kind) state.mods.delete(k);
    for (const site of m[key]) state.mods.set(modKey(site.name, site.residue, site.position), kind);
    if (kind === "var" && z.lib) z.lib.mods = [];
  }
  if (m.max_variable !== undefined) {
    $("d-maxvar").value = String(m.max_variable);
    own("d-maxvar");
  }
  if (m.fixed || m.variable || m.max_variable !== undefined) modsChanged();

  const ex = s.experiment || {};
  if (ex.mode !== undefined) {
    state.runMode = ex.mode;
    $(`runmode-${ex.mode}`).checked = true;
  }
  if (ex.normalisation !== undefined) $("norm-scope").value = ex.normalisation;
  refreshNormOptions();

  const pre = s.prescreen || {};
  if (pre.mode !== undefined) $("prescreen").value = pre.mode;
  if (pre.strength !== undefined) $("prescreen-strength").value = pre.strength;
  refreshPrescreenNote();

  const b = s.bands || {};
  if (b.mode !== undefined) {
    $("bands").value = b.mode;
    show($("bands-custom"), b.mode === "custom");
  }
  if (b.count !== undefined) $("bands-n").value = String(b.count);
  if (b.at_a_time !== undefined) $("bands-parallel").value = String(b.at_a_time);
  renderSpace();

  const zt = s.ztscan || {};
  if (zt.analysis !== undefined) {
    z.analysisTouched = !zt.analysis;
    z.analysisChoice = zt.analysis ? null : false;
  }
  if (zt.q1_half_width !== undefined) {
    if (zt.q1_half_width === null) z.hwTyped = false;
    else {
      $("ztscan-hw").value = String(zt.q1_half_width);
      z.hwTyped = true;
    }
  }
  renderZtscanNote();
  summariseMods();
  scheduleSpace();
  scheduleBuiltinPlan();
  if (state.libSrc === "diann") await refreshLibSrcNote();
}

// ── setups: the Settings screen card ─────────────────────────────────────────

function setupsMessage(error, done) {
  banner($("setups-error"), error || "");
  banner($("setups-done"), done || "");
}

function setupName() {
  return ($("setups-name").value || "").trim();
}

// What saving said about the settings it left out or saved on the way.
function savedNotes(cur) {
  const parts = [];
  if (cur.savedEdits) {
    parts.push("The changes on this screen were saved and are used by the next search, as the setup has them.");
  }
  if (cur.local.length) {
    parts.push(`Not in the setup, as they describe this computer: ${cur.local.join(", ")}.`);
  }
  if (cur.paths.length) {
    parts.push(
      `Saved as "auto" rather than this computer's interpreter path, so the other computer ` +
        `finds its own: ${cur.paths.join(", ")}.`
    );
  }
  return parts.length ? ` ${parts.join(" ")}` : "";
}

// Ask a yes-or-no question; true to go on.
async function askYesNo(message, title, okLabel) {
  try {
    return await dialog.ask(message, { title, kind: "warning", okLabel, cancelLabel: "Cancel" });
  } catch {
    return window.confirm(message);
  }
}

// Keep `settings` as a named setup, asking before one of the same name is replaced. The
// item saved, or null when the user kept the one there.
async function keepSetup(name, settings) {
  let r = await invoke("setups_save", { name, settings, overwrite: false });
  if (r.existing) {
    const replace = await askYesNo(
      `A setup named "${r.existing}" is already saved. Replace it?`,
      "Replace a saved setup",
      "Replace"
    );
    if (!replace) return null;
    r = await invoke("setups_save", { name, settings, overwrite: true });
  }
  return r.item;
}

async function saveNamedSetup() {
  setupsMessage();
  const name = setupName();
  if (!name) return setupsMessage("Give the setup a name first, for example Phospho or Label-free.");
  let cur;
  try {
    cur = await currentSetup();
    const item = await keepSetup(name, cur.settings);
    if (!item) return setupsMessage("", "Nothing was saved; the setup already saved under that name is kept.");
    setupsMessage("", `Saved "${item.name}" in My setups.${savedNotes(cur)}`);
  } catch (e) {
    return setupsMessage(String(e));
  }
  await refreshSetups();
}

// A file name a setup's name can become.
function setupFileName(name) {
  const safe = name.replace(/[\\/:*?"<>|\u0000-\u001f]/g, "_").trim().replace(/^\.+/, "");
  return `${safe || "settings"}.json`;
}

async function exportCurrentSetup() {
  setupsMessage();
  const name = setupName();
  if (!name) {
    return setupsMessage(
      "Give the setup a name first: the file carries it, and it is shown when the file is loaded."
    );
  }
  try {
    const cur = await currentSetup();
    const path = await dialog.save({ defaultPath: setupFileName(name), filters: SETUP_FILTERS });
    if (!path) return;
    const out = await invoke("setup_write", { path, name, settings: cur.settings });
    setupsMessage("", `Saved "${name}" to ${out}.${savedNotes(cur)}`);
  } catch (e) {
    setupsMessage(String(e));
  }
}

async function importSetupFile() {
  setupsMessage();
  let path;
  try {
    path = await dialog.open({ multiple: false, filters: SETUP_FILTERS });
  } catch (e) {
    return setupsMessage(String(e));
  }
  if (!path) return;
  if (Array.isArray(path)) path = path[0];
  let doc;
  try {
    doc = await invoke("setup_read", { path });
  } catch (e) {
    return setupsMessage(String(e));
  }
  await reviewSetup(doc, { kind: "file", path });
}

// Check a setup read from a file or from My setups and show what it would change, with
// Apply and Cancel. Nothing changes until Apply.
async function reviewSetup(doc, origin) {
  cancelPendingSetup(true);
  // Checking the engine settings runs the engine once, which takes a moment.
  setupsMessage("", `Checking "${doc.name}"…`);
  let plan;
  try {
    plan = await planSetup(doc);
  } catch (e) {
    setupsMessage(String(e));
    return null;
  }
  setupsMessage();
  const changes = setupChanges(captureSearch(), plan);
  state.setups.pending = { doc, plan, origin, changes };

  $("setups-review-title").textContent = `Load "${doc.name}"?`;
  const meta = [];
  if (doc.created_by) meta.push(`Saved by ${doc.created_by}`);
  if (doc.created) meta.push(`${meta.length ? "on" : "Saved on"} ${fmtWhen(doc.created)}`);
  if (origin.kind === "file") meta.push(`${meta.length ? "in" : "From"} ${origin.path}`);
  $("setups-review-meta").textContent = meta.length ? `${meta.join(" ")}.` : "";
  banner($("setups-review-warn"), plan.warnings.join("\n"));
  const list = $("setups-review-changes");
  list.textContent = "";
  const lines = changes.length
    ? changes.slice(0, 40)
    : ["Nothing changes: the current settings are already this setup's."];
  for (const line of lines) list.appendChild(node("li", "", line));
  if (changes.length > 40) list.appendChild(node("li", "", `… and ${changes.length - 40} more.`));
  show($("setups-keep-row"), origin.kind === "file");
  $("setups-keep").checked = true;
  $("setups-apply").disabled = false;
  $("setups-cancel").disabled = false;
  show($("setups-review"), true);
  $("setups-apply").focus?.();
  return plan;
}

// Close the summary without applying. `quiet` closes it without a message, for a new
// summary replacing it.
function cancelPendingSetup(quiet) {
  const had = !!state.setups.pending;
  state.setups.pending = null;
  show($("setups-review"), false);
  if (had && quiet !== true) setupsMessage("", "Nothing was changed.");
}

async function applyPendingSetup() {
  const p = state.setups.pending;
  if (!p) return;
  $("setups-apply").disabled = true;
  $("setups-cancel").disabled = true;
  try {
    await applySetup(p.plan);
  } catch (e) {
    $("setups-apply").disabled = false;
    $("setups-cancel").disabled = false;
    return setupsMessage(`"${p.doc.name}" was not loaded, and nothing was changed: ${e}`);
  }
  state.setups.pending = null;
  show($("setups-review"), false);
  const n = p.changes.length;
  let message = `Loaded "${p.doc.name}": ${n ? `${n} change${n === 1 ? "" : "s"}` : "nothing changed"}.`;
  if (p.plan.engine) {
    message += p.plan.engine.mine
      ? " Its engine settings are saved as My settings and used by the next search."
      : ` The next search uses ${presetText(p.plan.engine.presetName)}.`;
  }
  if (p.origin.kind === "file" && $("setups-keep").checked) {
    try {
      const item = await keepSetup(p.doc.name, p.plan.setup);
      message += item ? ` Kept in My setups as "${item.name}".` : " Not added to My setups.";
    } catch (e) {
      message += ` It could not be kept in My setups: ${e}`;
    }
    await refreshSetups();
  }
  setupsMessage("", message);
}

function fmtWhen(iso) {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return iso;
  return new Date(t).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
}

async function refreshSetups() {
  let r;
  try {
    r = await invoke("setups_list");
  } catch (e) {
    return setupsMessage(String(e));
  }
  state.setups.list = r.items || [];
  state.setups.dir = r.dir || "";
  $("setups-where").textContent = state.setups.dir
    ? `Kept in ${state.setups.dir}, outside the storage folder: changing the storage folder, ` +
      "removing managed data or uninstalling MuMDIA does not remove them."
    : "";
  show($("setups-empty"), state.setups.list.length === 0);
  const host = $("setups-list");
  host.textContent = "";
  for (const it of state.setups.list) host.appendChild(setupRow(it));
}

// One named setup: Load, Rename, Save to a file and Delete, the rename and the
// confirmation opening in place as on the Managed data card.
function setupRow(it) {
  const row = node("div", "dataitem");
  const title = node("div", "dname", it.name);
  const when = node("div", "dsize", it.created ? fmtWhen(it.created) : "");
  const detail = it.error
    ? `Cannot be read: ${it.error}`
    : [it.preset ? `Starts from the preset "${it.preset}"` : "Starts from the engine defaults", it.created_by]
        .filter(Boolean)
        .join(" · ");
  row.append(title, when, node("div", "dhelp", detail));

  const act = node("div", "dact");
  const button = (text, label) => {
    const b = node("button", "btn quiet", text);
    b.type = "button";
    b.setAttribute("aria-label", label);
    return b;
  };
  const load = button("Load", `Load the setup ${it.name}`);
  const rename = button("Rename", `Rename the setup ${it.name}`);
  const save = button("Save to a file…", `Save the setup ${it.name} to a file`);
  const del = button("Delete", `Delete the setup ${it.name}`);
  if (!it.error) act.append(load, rename, save);
  act.appendChild(del);
  row.appendChild(act);

  load.addEventListener("click", async () => {
    setupsMessage();
    let doc;
    try {
      doc = await invoke("setups_read", { id: it.id });
    } catch (e) {
      return setupsMessage(String(e));
    }
    await reviewSetup(doc, { kind: "named", id: it.id });
  });

  save.addEventListener("click", async () => {
    setupsMessage();
    try {
      const path = await dialog.save({ defaultPath: setupFileName(it.name), filters: SETUP_FILTERS });
      if (!path) return;
      const out = await invoke("setups_export", { id: it.id, path });
      setupsMessage("", `Saved "${it.name}" to ${out}.`);
    } catch (e) {
      setupsMessage(String(e));
    }
  });

  const edit = node("div", "dconfirm");
  edit.hidden = true;
  const input = document.createElement("input");
  input.type = "text";
  input.value = it.name;
  input.maxLength = 120;
  input.setAttribute("aria-label", `New name for ${it.name}`);
  const editButtons = node("div", "dact");
  const ok = node("button", "btn", "Rename");
  const back = node("button", "btn quiet", "Cancel");
  editButtons.append(ok, back);
  edit.append(input, editButtons);
  row.appendChild(edit);
  rename.addEventListener("click", () => {
    edit.hidden = false;
    rename.disabled = true;
    input.focus?.();
  });
  back.addEventListener("click", () => {
    edit.hidden = true;
    rename.disabled = false;
  });
  ok.addEventListener("click", async () => {
    setupsMessage();
    try {
      const r = await invoke("setups_rename", { id: it.id, name: input.value });
      setupsMessage("", `Renamed "${it.name}" to "${r.name}".`);
    } catch (e) {
      return setupsMessage(String(e));
    }
    await refreshSetups();
  });

  const ask = node("div", "dconfirm");
  ask.hidden = true;
  ask.appendChild(node("p", "dwarn", `Delete the setup "${it.name}"? This cannot be undone.`));
  const askButtons = node("div", "dact");
  const yes = node("button", "btn", "Delete permanently");
  const no = node("button", "btn quiet", "Cancel");
  askButtons.append(yes, no);
  ask.appendChild(askButtons);
  row.appendChild(ask);
  del.addEventListener("click", () => {
    ask.hidden = false;
    del.disabled = true;
    no.focus?.();
  });
  no.addEventListener("click", () => {
    ask.hidden = true;
    del.disabled = false;
  });
  yes.addEventListener("click", async () => {
    setupsMessage();
    try {
      await invoke("setups_delete", { id: it.id });
      setupsMessage("", `Deleted "${it.name}".`);
    } catch (e) {
      return setupsMessage(String(e));
    }
    await refreshSetups();
  });
  return row;
}

// ── components ──────────────────────────────────────────────────────────────
// Polled while an installation runs, then left alone. The backend owns the state,
// so closing and reopening this screen shows the truth rather than a stale copy.

// The three environments, in the order the log pane prefers them, with the labels the
// banner uses.
const COMPONENT_ENVS = [
  ["primary", "Analysis components"],
  ["ms2pip", "MS2PIP"],
  ["visdia", "visDIA"],
];

// The real import of the packages (`loadcheck`), run once an installation ends and the
// first time a session finds an environment: `find_spec` alone passed a torch that could
// not load its DLLs.
function componentTesting(e) {
  return !!e && e.load_check === "running";
}

function componentInstalling(e) {
  return !!e && e.install_status === "installing";
}

// What an optional environment's card shows: the pill, its class, the button label and
// whether the button is disabled. Pure, so the state table is tested
// (`ci/test_desktop_ui.mjs`) without a window.
function optionalCardView(e, uv) {
  e = e || {};
  if (componentInstalling(e)) {
    return { pill: "installing…", cls: "pill warn", label: "Installing…", disabled: true };
  }
  if (componentTesting(e)) {
    return { pill: "testing…", cls: "pill warn", label: "Testing…", disabled: true };
  }
  if (e.load_check === "failed") {
    // Installed, but a package does not load. Installing again repairs a damaged one.
    return { pill: "cannot load", cls: "pill bad", label: "Reinstall", disabled: !uv };
  }
  if (e.complete) {
    return { pill: "installed", cls: "pill ok", label: "Installed", disabled: true };
  }
  if ((e.outdated || []).length) {
    // Installed by an older release, or the viewer from another pinned commit: it
    // imports, but installing again is what brings it to the pins.
    return { pill: "update needed", cls: "pill warn", label: "Update", disabled: !uv };
  }
  if (e.install_status === "failed") {
    return { pill: "failed", cls: "pill bad", label: "Reinstall", disabled: !uv };
  }
  return { pill: "optional", cls: "pill mute", label: "Install", disabled: !uv };
}

// Whether anything on the Setup screen is still moving, which is what keeps the poll on.
function componentsBusy(c) {
  return COMPONENT_ENVS.some(([k]) => componentInstalling(c[k]) || componentTesting(c[k]));
}

// One log pane, showing whichever installation is talking; otherwise the environment
// whose last installation failed or whose packages do not load, so the traceback stays
// readable; otherwise the last finished installation's log. Returns `[key, status, open]`,
// `open` being whether the card is shown at all.
function installLogSource(c) {
  const envs = COMPONENT_ENVS.map(([k]) => [k, c[k] || {}]);
  const hasLog = (e) => (e.install_log || []).length > 0;
  const talking = envs.find(([, e]) => componentInstalling(e));
  if (talking) return [talking[0], talking[1], true];
  const failed = envs.find(([, e]) => e.install_status === "failed" && hasLog(e));
  if (failed) return [failed[0], failed[1], true];
  const unloadable = envs.find(([, e]) => e.load_check === "failed");
  if (unloadable) return [unloadable[0], unloadable[1], true];
  const done = envs.find(([, e]) => e.install_status === "done" && hasLog(e));
  if (done) return [done[0], done[1], true];
  return ["primary", envs[0][1], false];
}

// The Setup banner: the first thing that needs the user, across all three environments.
// The optional environments were once read here before they were declared, which threw
// on every refresh and left their cards and the poll frozen; this reads only `c`.
function componentsBannerText(c) {
  const p = c.primary || {};
  if (!p.uv && !p.complete) {
    return (
      "The installer component `uv` was not found beside the application or on PATH, " +
      "so components cannot be installed automatically."
    );
  }
  if (p.error) return p.error;
  for (const [k, label] of COMPONENT_ENVS.slice(1)) {
    const e = c[k] || {};
    if (componentInstalling(e)) continue;
    if (e.error) return `${label}: ${e.error}`;
    if (e.load_error) return `${label}: ${e.load_error}`;
  }
  if ((p.outdated || []).length) {
    return (
      `The analysis environment needs an update: ${p.outdated.join(", ")}. ` +
      "Press Update; it upgrades the installed packages in place."
    );
  }
  return "";
}

async function refreshComponents() {
  let c;
  try {
    c = await invoke("components_status");
  } catch (e) {
    banner($("setup-error"), String(e));
    return;
  }
  try {
    renderComponents(c);
  } catch (e) {
    // A rendering fault must not freeze the screen silently: an exception here once
    // left every optional card on its first state and stopped the poll.
    banner($("setup-error"), `The Setup screen could not show the component state: ${e}`);
    console.error(e);
  }
  // Decided from the status alone, after rendering, so even a rendering fault keeps
  // polling while an installation runs.
  const busy = componentsBusy(c);
  if (busy && !state.setupTimer) {
    state.setupTimer = setInterval(refreshComponents, 900);
  } else if (!busy && state.setupTimer) {
    clearInterval(state.setupTimer);
    state.setupTimer = null;
    // An installation just finished, so the managed-data sizes are stale. Once, on
    // the transition: walking a Python environment is too slow for the poll itself.
    refreshManaged();
  }
}

function renderComponents(c) {
  const p = c.primary;
  const m = c.ms2pip || {};
  const v = c.visdia || {};
  state.componentsReady = !!p.complete;

  const pill = $("primary-pill");
  const btn = $("install-primary");
  const installing = componentInstalling(p);
  const loadFailed = p.load_check === "failed";
  show($("primary-bar"), installing || componentTesting(p));
  show($("recheck-primary"), loadFailed);
  renderVcRuntime(c.vc_runtime);

  if (installing) {
    pill.textContent = "installing…";
    pill.className = "pill warn";
    btn.disabled = true;
  } else if (componentTesting(p)) {
    pill.textContent = "testing…";
    pill.className = "pill warn";
    btn.disabled = true;
  } else if (loadFailed) {
    // Installed, but a package does not load. Installing again repairs a damaged
    // package; a missing system runtime needs Re-check once it is fixed.
    pill.textContent = "cannot load";
    pill.className = "pill bad";
    btn.disabled = !c.primary.uv;
    btn.textContent = "Reinstall";
  } else if (p.complete) {
    pill.textContent = "installed";
    pill.className = "pill ok";
    btn.disabled = true;
    btn.textContent = "Installed";
  } else if ((p.outdated || []).length) {
    // Installed by an older release: it imports, but a pinned package is behind what
    // this release needs (the engine refuses a DeepLC below its floor). Installing again
    // upgrades the environment in place.
    pill.textContent = "update needed";
    pill.className = "pill warn";
    btn.disabled = !c.primary.uv;
    btn.textContent = "Update";
  } else {
    pill.textContent = "not installed";
    pill.className = "pill bad";
    btn.disabled = !c.primary.uv;
    btn.textContent = "Install";
  }

  banner($("setup-error"), componentsBannerText(c));

  const uv = !!c.primary.uv;
  for (const [e, pillId, btnId] of [
    [m, "ms2pip-pill", "install-ms2pip"],
    [v, "visdia-pill", "install-visdia"],
  ]) {
    const view = optionalCardView(e, uv);
    $(pillId).textContent = view.pill;
    $(pillId).className = view.cls;
    $(btnId).textContent = view.label;
    $(btnId).disabled = view.disabled;
  }
  state.visdiaReady = !!v.complete;
  refreshVisdiaButtons();

  const [, source, open] = installLogSource(c);
  show($("install-log-card"), open);
  const logEl = $("install-log");
  const text = (source.install_log || []).join("\n");
  if (logEl.textContent !== text) {
    logEl.textContent = text;
    logEl.scrollTop = logEl.scrollHeight;
  }

  const versions = Object.entries(p.versions || {});
  $("setup-versions").textContent = versions.length
    ? versions.map(([k, v]) => `${k} ${v}`).join(" · ")
    : "";
}

// Test again whether the packages load, after something outside the application was
// fixed (the Visual C++ runtime, for one). The backend forgets its last result and the
// next status poll starts the import.
async function recheckComponents(env) {
  banner($("setup-error"), "");
  try {
    await invoke("components_recheck", { env });
  } catch (e) {
    banner($("setup-error"), String(e));
    return;
  }
  refreshComponents();
}

// "Install Visual C++ runtime" (`vcruntime.rs`): offered on Windows when the real import
// failed on a torch DLL, and only ever run on this click. The backend downloads
// Microsoft's installer, refuses it unless its signature is valid and Microsoft's, runs it
// elevated (Windows asks for permission), and tests the packages again afterwards.
function renderVcRuntime(vc) {
  if (!vc) return;
  const s = vc.state || {};
  const busy = ["downloading", "verifying", "installing"].includes(s.phase);
  const spoke = busy || s.phase === "done" || s.phase === "failed";
  show($("vcrt-install"), vc.available && vc.needed && !busy);
  $("vcrt-install").disabled = busy;
  show($("vcrt-status"), vc.available && spoke && !!s.message);
  $("vcrt-status").textContent = s.message || "";
  const bar = $("vcrt-bar");
  show(bar, vc.available && busy);
  bar.classList.toggle("pct", s.phase === "downloading" && s.percent > 0);
  bar.firstElementChild.style.width =
    s.phase === "downloading" && s.percent > 0 ? `${s.percent}%` : "";
  if (busy && !state.vcrtTimer) {
    state.vcrtTimer = setInterval(pollVcRuntime, 700);
  } else if (!busy && state.vcrtTimer) {
    clearInterval(state.vcrtTimer);
    state.vcrtTimer = null;
  }
}

async function pollVcRuntime() {
  let s;
  try {
    s = await invoke("vcruntime_state");
  } catch {
    return;
  }
  const busy = ["downloading", "verifying", "installing"].includes(s.phase);
  if (busy) {
    renderVcRuntime({ available: true, needed: true, state: s });
  } else {
    // Finished: the backend has asked for the packages to be tested again, so the whole
    // screen is refreshed, which shows "testing..." and then the new result.
    clearInterval(state.vcrtTimer);
    state.vcrtTimer = null;
    refreshComponents();
  }
}

async function installVcRuntime() {
  banner($("setup-error"), "");
  try {
    await invoke("vcruntime_install");
  } catch (e) {
    banner($("setup-error"), String(e));
    return;
  }
  pollVcRuntime();
}

async function installComponents(env) {
  banner($("setup-error"), "");
  // Feedback before the backend answers. The command returns as soon as the installation
  // has started (it runs on a thread of its own), and the poll takes over from there.
  const btn = $(`install-${env}`);
  const pill = $(`${env}-pill`);
  btn.disabled = true;
  btn.textContent = "Installing…";
  pill.textContent = "installing…";
  pill.className = "pill warn";
  let failure = null;
  try {
    await invoke("components_install", { env });
  } catch (e) {
    failure = String(e);
  }
  // Back to what the backend says the cards are, which also starts the poll.
  await refreshComponents();
  // A refusal (no `uv`, an installation already running) is the answer to this click,
  // so it is shown over whatever the refresh put in the banner.
  if (failure) banner($("setup-error"), failure);
}

// -- managed data ------------------------------------------------------------
// Everything the application downloads or builds lives under one per-user folder,
// and the installer does not remove it: an MSI uninstalls what it installed, and all
// of this is written at runtime. Several gigabytes left behind with no visible way to
// remove them is the reason this card exists.
//
// Nothing is deleted on a single click. The confirmation names the exact paths,
// because "Required components" is a label and `C:\Users\...\MuMDIA\python` is what
// actually disappears, and the spectral library cache in particular can be hours of
// prediction rather than a download.
const SIZE_UNITS = ["B", "KB", "MB", "GB", "TB"];

function bytesLabel(n) {
  let v = Number(n) || 0;
  let i = 0;
  while (v >= 1024 && i < SIZE_UNITS.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${i === 0 ? v : v.toFixed(v < 10 ? 1 : 0)} ${SIZE_UNITS[i]}`;
}

function managedRow(item) {
  const row = node("div", "dataitem");
  row.append(
    node("div", "dname", item.label),
    node("div", "dsize", bytesLabel(item.bytes)),
    node("div", "dhelp", item.detail)
  );

  const act = node("div", "dact");
  const remove = node("button", "btn quiet", "Remove");
  act.appendChild(remove);
  row.appendChild(act);

  const ask = node("div", "dconfirm");
  ask.hidden = true;
  ask.append(
    node("p", "dpaths", item.paths.join("\n")),
    node(
      "p",
      "dwarn",
      item.costly
        ? "This is not downloaded again. Getting it back means running the prediction " +
            "once more, which takes hours on a whole proteome."
        : "This is deleted permanently. Reinstall it from this screen when you need it."
    )
  );
  const buttons = node("div", "dact");
  const yes = node("button", "btn", "Delete permanently");
  const no = node("button", "btn quiet", "Cancel");
  buttons.append(yes, no);
  ask.appendChild(buttons);
  row.appendChild(ask);

  remove.addEventListener("click", () => {
    ask.hidden = false;
    remove.disabled = true;
  });
  no.addEventListener("click", () => {
    ask.hidden = true;
    remove.disabled = false;
  });
  yes.addEventListener("click", async () => {
    yes.disabled = true;
    no.disabled = true;
    yes.textContent = "Removing\u2026";
    banner($("managed-error"), "");
    banner($("managed-freed"), "");
    try {
      const r = await invoke("components_remove", { id: item.id });
      renderManaged(r);
      banner($("managed-freed"), `Removed ${item.label}. ${bytesLabel(r.freed_bytes)} freed.`);
      // The other Setup cards read the same folder, so they are now stale.
      refreshComponents();
      refreshThermo();
      refreshDiann();
    } catch (e) {
      banner($("managed-error"), String(e));
      yes.disabled = false;
      no.disabled = false;
      yes.textContent = "Delete permanently";
    }
  });
  return row;
}

function renderManaged(m) {
  state.managedDir = m.dir;
  $("managed-where").textContent = m.dir;
  const items = m.items || [];
  $("managed-total").textContent = items.length
    ? bytesLabel(items.reduce((a, x) => a + x.bytes, 0))
    : "empty";
  show($("managed-empty"), items.length === 0);
  const host = $("managed-items");
  host.textContent = "";
  for (const it of items) host.appendChild(managedRow(it));
}

// Sizing the folder walks every file in it, so this runs when the screen is opened
// and after a removal -- never on the installation poll, which ticks every 900 ms.
async function refreshManaged() {
  try {
    renderManaged(await invoke("components_removable"));
  } catch (e) {
    banner($("managed-error"), String(e));
  }
}

// -- storage folder ------------------------------------------------------
// One setting for where MuMDIA writes (desktop/src-tauri/src/storage.rs). The folder
// picker and the reset change it; the checkbox decides whether the Python environments
// move too, which means installing them again, so it applies with the next change.
function renderStorage(s) {
  state.storage = s;
  const pill = $("storage-pill");
  if (!s.writable) {
    pill.textContent = "cannot be written";
    pill.className = "pill bad";
  } else {
    pill.textContent = s.configured ? "chosen" : "default";
    pill.className = s.configured ? "pill ok" : "pill mute";
  }
  const free = s.free_bytes == null ? "" : ` (${bytesLabel(s.free_bytes)} free)`;
  $("storage-where").textContent = s.folder + free;
  $("storage-envs").textContent =
    s.environments === s.folder
      ? "The Python environments and downloaded tools are in this folder too."
      : `Python environments and downloaded tools: ${s.environments}`;
  $("storage-move-envs").checked = !!s.move_environments;
  show($("storage-reset"), s.configured);
  banner($("storage-error"), s.error ? `MuMDIA cannot write here: ${s.error}` : "");
  banner($("storage-warn"), (s.warnings || []).join("\n"));
}

async function refreshStorage() {
  try {
    renderStorage(await invoke("storage_info"));
  } catch (e) {
    banner($("storage-error"), String(e));
  }
}

async function setStorage(folder) {
  const moveEnvironments = $("storage-move-envs").checked;
  const before = state.storage;
  if (moveEnvironments && folder && before && !before.move_environments) {
    const ok = await dialog.ask(
      "The Python environments and downloaded tools will be looked for in the new " +
        "folder, so they have to be installed again from this screen. Continue?",
      { title: "Move the Python environments", kind: "warning" }
    );
    if (!ok) {
      $("storage-move-envs").checked = false;
      return;
    }
  }
  banner($("storage-error"), "");
  try {
    renderStorage(await invoke("storage_set", { folder, moveEnvironments }));
    // Every other Setup card reads the folders that just moved.
    refreshComponents();
    refreshThermo();
    refreshDiann();
    refreshManaged();
  } catch (e) {
    banner($("storage-error"), String(e));
  }
}

async function chooseStorage() {
  const chosen = await dialog.open({ directory: true, multiple: false });
  if (!chosen) return;
  await setStorage(Array.isArray(chosen) ? chosen[0] : chosen);
}

// -- DIA-NN --------------------------------------------------------------
// Detection, not installation. DIA-NN is closed source and its licence forbids
// redistribution from 1.9 onward, so MuMDIA cannot fetch it the way it fetches
// DeepLC and torch; it can only drive a copy the user licensed themselves. The
// notice is shown here because that distinction is invisible from a screen whose
// other cards do install things.
async function refreshDiann() {
  let r;
  try {
    r = await invoke("diann_status");
  } catch (e) {
    banner($("setup-error"), String(e));
    return;
  }
  const d = r.status;
  $("diann-notice-body").textContent = r.notice;
  $("diann-ack").checked = !!d.licence_acknowledged;
  show($("diann-ack-note"), !!d.licence_assumed);

  const pill = $("diann-pill");
  if (d.runs) {
    pill.textContent = "found";
    pill.className = "pill ok";
  } else if (d.path) {
    pill.textContent = "does not run";
    pill.className = "pill bad";
  } else {
    pill.textContent = "not found";
    pill.className = "pill mute";
  }

  // Which copy, and why that one: with an environment variable, a PATH entry and
  // an installer directory all possible, "found" alone does not say what will run,
  // and DIA-NN's version changes the library it predicts.
  const where = $("diann-where");
  if (d.error) {
    where.textContent = d.error;
  } else if (d.path) {
    const via = { configured: "you chose it", environment: "MUMDIA_DIANN", path: "on PATH", installed: "installed" };
    where.textContent = `${d.version || "DIA-NN"} — ${d.path} (${via[d.source] || d.source})`;
  } else {
    where.textContent =
      "No DIA-NN found. Install it yourself from github.com/vdemichev/DiaNN, then " +
      "use Locate. MuMDIA does not download it.";
  }

  // Building a library happens on the Search screen now; Setup only decides whether
  // DIA-NN is available and licensed. What it decides changes what the Search screen
  // can offer, so that is refreshed too.

  refreshDiannOffer(d);
  refreshLibSrcNote();
}

async function refreshDiannOffer(d) {
  if (!state.diannOffer) {
    try {
      state.diannOffer = await invoke("diann_offer");
    } catch {
      return;
    }
  }
  const o = state.diannOffer;
  // Only worth showing when there is nothing working already.
  show($("diann-get"), !!o.available && !d.runs);
  if (!o.available || d.runs) return;

  const gb = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);
  $("diann-get-sub").textContent =
    `DIA-NN ${o.version} from the vendor's release page. ${gb(o.download_bytes)} download, ` +
    `about ${gb(o.disk_bytes)} on disk. MuMDIA does not host or modify these files, and ` +
    `verifies their checksum before use.` +
    (o.hands_off ? "" : " The DIA-NN installer opens at the end and you complete it there.");
  $("diann-download").disabled = !$("diann-ack").checked;
  if (!$("diann-ack").checked) {
    $("diann-get-step").textContent = "Acknowledge the licence above first.";
  }
  refreshDiannInstall();
}

async function refreshDiannInstall() {
  let s;
  try {
    s = await invoke("diann_install_state");
  } catch {
    return;
  }
  const running = s.status === "running";
  const bar = $("diann-get-bar");
  show(bar, running);
  bar.classList.toggle("pct", s.percent > 0);
  bar.firstElementChild.style.width = s.percent > 0 ? `${s.percent}%` : "";

  show($("diann-get-log-card"), s.log.length > 0);
  const logEl = $("diann-get-log");
  const text = (s.log || []).join("\n");
  if (logEl.textContent !== text) {
    logEl.textContent = text;
    logEl.scrollTop = logEl.scrollHeight;
  }

  $("diann-download").disabled = running || !$("diann-ack").checked;
  if (running) {
    $("diann-get-step").textContent = s.percent > 0 ? `${s.step} — ${s.percent}%` : s.step;
  } else if (s.status === "handoff") {
    // Windows: the vendor's installer is now the thing the user is talking to.
    $("diann-get-step").textContent =
      "The DIA-NN installer is open. Finish it, then press Locate.";
  } else if (s.status === "done") {
    $("diann-get-step").textContent = "Installed.";
  } else if (!$("diann-ack").checked) {
    $("diann-get-step").textContent = "Acknowledge the licence above first.";
  } else {
    $("diann-get-step").textContent = "";
  }
  banner($("diann-get-error"), s.status === "failed" ? s.error || "The download failed." : "");

  if (running && !state.diannGetTimer) {
    state.diannGetTimer = setInterval(refreshDiannInstall, 700);
  } else if (!running && state.diannGetTimer) {
    clearInterval(state.diannGetTimer);
    state.diannGetTimer = null;
    // A finished install changes what detection sees.
    if (s.status === "done" || s.status === "handoff") refreshDiann();
  }
}

async function downloadDiann() {
  banner($("diann-get-error"), "");
  try {
    await invoke("diann_install");
  } catch (e) {
    banner($("diann-get-error"), String(e));
    return;
  }
  refreshDiannInstall();
}

async function locateDiann() {
  const chosen = await dialog.open({
    multiple: false,
    filters: window.navigator.userAgent.includes("Windows")
      ? [{ name: "Programs", extensions: ["exe"] }]
      : undefined,
  });
  if (!chosen) return;
  const path = Array.isArray(chosen) ? chosen[0] : chosen;
  try {
    await invoke("diann_set_path", { path });
  } catch (e) {
    banner($("setup-error"), String(e));
  }
  refreshDiann();
}

async function refreshThermo() {
  let t;
  try {
    t = await invoke("thermo_status");
  } catch (e) {
    banner($("setup-error"), String(e));
    return;
  }
  state.thermoReady = !!t.ready;

  const pill = $("thermo-pill");
  const btn = $("install-thermo");
  const installing = t.install_status === "installing";
  show($("thermo-bar"), installing);
  const bar = $("thermo-bar");
  bar.classList.toggle("pct", t.percent > 0);
  bar.firstElementChild.style.width = t.percent > 0 ? `${t.percent}%` : "";

  if (!t.available) {
    pill.textContent = "unavailable";
    pill.className = "pill mute";
    btn.disabled = true;
    $("thermo-where").textContent =
      "No build is published for this platform. Convert .raw to mzML with msconvert instead.";
  } else if (installing) {
    pill.textContent = t.percent > 0 ? `${t.percent}%` : "installing…";
    pill.className = "pill warn";
    btn.disabled = true;
    $("thermo-where").textContent = t.step;
  } else if (t.ready) {
    pill.textContent = "installed";
    pill.className = "pill ok";
    btn.disabled = true;
    btn.textContent = "Installed";
    $("thermo-where").textContent = `${t.version || "ThermoRawFileParser"} — ${t.path}`;
  } else {
    pill.textContent = "optional";
    pill.className = "pill mute";
    btn.disabled = false;
    $("thermo-where").textContent = t.error
      ? t.error
      : `About ${Math.round(t.download_bytes / 1e6)} MB. Includes its own runtime, so nothing else is needed.`;
  }

  // msconvert is detect-only: MuMDIA does not install it, so there is no button
  // state to manage beyond found / not found.
  try {
    state.msconvert = await invoke("msconvert_status");
  } catch {
    state.msconvert = null;
  }
  const mp = $("msconvert-pill");
  if (state.msconvert) {
    mp.textContent = "found";
    mp.className = "pill ok";
    $("msconvert-where").textContent = state.msconvert;
  } else {
    mp.textContent = "not found";
    mp.className = "pill mute";
    $("msconvert-where").textContent =
      "Not found. Install ProteoWizard, or set MUMDIA_MSCONVERT to its msconvert.";
  }
  banner(
    $("bruker-note"),
    "Bruker: a timsTOF .d with diaPASEF frames is read directly, with its ion mobility " +
      "kept, and needs no converter. Every other Bruker .d is converted with msconvert, " +
      "which discards ion mobility: it will search, with calibrated q values, but a " +
      "mobility-separated acquisition loses the separation that makes it selective."
  );

  if (installing && !state.thermoTimer) {
    state.thermoTimer = setInterval(refreshThermo, 700);
  } else if (!installing && state.thermoTimer) {
    clearInterval(state.thermoTimer);
    state.thermoTimer = null;
  }
  // Files may already be selected on the search screen, and whether they can be
  // searched has just changed -- both the note and the per-file tags.
  updateRawNote();
  renderMzmlList();
}

async function installThermo() {
  banner($("setup-error"), "");
  try {
    await invoke("thermo_install");
  } catch (e) {
    banner($("setup-error"), String(e));
    return;
  }
  refreshThermo();
}

// Mirrors `thermo::needs` in the backend. A `.raw` that is a directory is Waters,
// not Thermo, and routes to a different converter -- but the frontend cannot stat a
// path, so it asks the backend which converter a file needs rather than guessing.
function isRaw(path) {
  return /\.raw$/i.test(path || "");
}
function isVendor(path) {
  return /\.(raw|d|wiff|wiff2)$/i.test(path || "");
}

// Shown when a .raw is selected. Two things the user needs and cannot infer: that
// conversion happens and costs minutes, and that the peak statistics shown for an
// mzML are not available until it has.
async function updateRawNote() {
  // The note describes the selection as a whole, so it speaks about the first vendor
  // file in it; the per-file list carries the detail.
  const path = state.picks.mzml.find((f) => isVendor(f)) || state.picks.mzml[0] || "";
  const vendor = isVendor(path);
  show($("raw-note"), vendor);
  if (!vendor) return;

  // The backend owns the file-versus-directory distinction, so it decides which
  // converter this needs and what to call the format.
  let v;
  try {
    v = await invoke("vendor_of", { path });
  } catch {
    return;
  }
  $("raw-note-title").textContent = v.label;
  const body = $("raw-note-body");
  const common =
    " Conversion takes several minutes and writes an mzML next to the input, which " +
    "later searches of the same file reuse. Peak statistics are reported during the run.";

  if (v.needs === "Nothing") {
    // A diaPASEF .d: the engine reads it natively, so nothing is converted.
    body.textContent =
      "This timsTOF diaPASEF .d is read directly, with its ion mobility kept. No " +
      "conversion and no converter are needed. Peak statistics are reported during the run.";
  } else if (v.needs === "ThermoParser") {
    body.textContent = state.thermoReady
      ? `MuMDIA reads mzML, so this is converted first with ThermoRawFileParser.${common}`
      : "This needs the Thermo .raw converter, which is not installed. Go to Setup and " +
        "install it, or convert to mzML yourself and select that.";
  } else {
    let extra = "";
    if (v.label === "Bruker .d") {
      // The thing the user must be told, and cannot infer: this .d is not diaPASEF, so
      // it goes through msconvert and loses its ion mobility.
      extra =
        " This .d is not a timsTOF diaPASEF acquisition, so msconvert converts it and " +
        "its ion mobility is discarded. For a mobility-separated acquisition that " +
        "removes the separation it exists to produce. The q values stay calibrated; " +
        "the sensitivity does not.";
    }
    body.textContent = state.msconvert
      ? `MuMDIA reads mzML, so this is converted first with msconvert.${common}${extra}`
      : "This needs ProteoWizard msconvert, which was not found. MuMDIA does not " +
        "install it; see Setup." + extra;
  }
}

function setMode(mode) {
  state.mode = mode;
  for (const t of document.querySelectorAll(".tab")) t.classList.toggle("on", t.dataset.mode === mode);
  show($("mode-fasta"), mode === "fasta");
  show($("mode-library"), mode === "library");
  // A library is built from a FASTA; there is nothing to build from a library.
  show($("build-library"), mode === "fasta");
  scheduleSpace();
  scheduleBuiltinPlan();
}

// ── file pickers ────────────────────────────────────────────────────────────
const FILTERS = {
  mzml: [
    {
      name: "Spectra",
      extensions: ["mzML", "mzml", "raw", "RAW", "wiff", "wiff2"],
    },
  ],
  fasta: [{ name: "FASTA", extensions: ["fasta", "fa", "fas"] }],
  lib_precursors: [{ name: "Parquet", extensions: ["parquet"] }],
  lib_fragments: [{ name: "Parquet", extensions: ["parquet"] }],
  lib_fasta: [{ name: "FASTA", extensions: ["fasta", "fa", "fas"] }],
  // Parquet (DIA-NN 1.9 and 2.x) or DIA-NN's binary .speclib; a .tsv is not accepted.
  diann_lib: [{ name: "DIA-NN library", extensions: ["parquet", "speclib"] }],
};
const LABEL = {
  mzml: "p-mzml",
  fasta: "p-fasta",
  lib_precursors: "p-libp",
  lib_fragments: "p-libf",
  out_dir: "p-out",
  mzml_dir: "p-mzml",
  lib_fasta: "p-lib-fasta",
  lib_out: "p-lib-out",
  diann_lib: "p-diann-lib",
  diann_lib_out: "p-diann-lib-out",
};
// Pickers that choose a folder rather than a file.
// `mzml_dir` is here because Bruker/Agilent `.d` and Waters `.raw` are
// directories, and a file dialog cannot select one.
const DIR_PICKS = new Set(["out_dir", "lib_out", "mzml_dir", "diann_lib_out"]);

// Render the selected spectra, with the vendor note each one needs.
//
// The list is the whole selection UI: a single `.picked` line could not show which
// of eight files is a Bruker directory that needs a converter, and that is exactly
// what preflight will refuse on.
async function renderMzmlList() {
  const list = $("mzml-list");
  const files = state.picks.mzml;
  show(list, files.length > 0);
  show($("multi-mode"), files.length > 1);

  const label = $("p-mzml");
  if (files.length === 0) {
    label.textContent = "Nothing selected";
    label.classList.remove("set");
  } else {
    label.textContent =
      files.length === 1 ? baseName(files[0]) : `${files.length} files selected`;
    label.classList.add("set");
  }

  list.innerHTML = "";
  // The condition of each file: what the user typed, else the suggestion from the names.
  const suggested = suggestConditions(files);
  for (const [i, f] of files.entries()) {
    if (!state.conditionTyped[f]) state.picks.conditions[f] = suggested[i];
  }
  // The experiment's options follow the file count now, not after each file's vendor
  // check below; the call after the list also shows its new condition fields.
  refreshNormOptions();
  let anyDiapasef = false;
  for (const [i, f] of files.entries()) {
    const li = document.createElement("li");

    const name = document.createElement("span");
    name.className = "fname";
    name.textContent = baseName(f);
    li.appendChild(name);

    if (files.length > 1) {
      const cond = document.createElement("input");
      cond.type = "text";
      cond.className = "fcond";
      cond.setAttribute("list", "condition-labels");
      cond.placeholder = "Condition";
      cond.title = "The condition of this run, for normalisation within each condition";
      cond.setAttribute("aria-label", `Condition of ${baseName(f)}`);
      cond.value = state.picks.conditions[f] || "";
      cond.addEventListener("input", () => {
        state.picks.conditions[f] = cond.value;
        state.conditionTyped[f] = true;
        refreshNormOptions();
      });
      li.appendChild(cond);
    }

    const path = document.createElement("span");
    path.className = "fpath";
    path.textContent = f;
    path.title = f;
    li.appendChild(path);

    // Say which files need something that is not installed, per file, here rather
    // than as one aggregate blocker after Start.
    let v = null;
    try {
      v = await invoke("vendor_of", { path: f });
    } catch {
      /* Detection is advisory; preflight is the authority. */
    }
    if (v && v.label === DIAPASEF_LABEL) anyDiapasef = true;
    if (v && v.needs !== "Nothing") {
      const ok =
        v.needs === "ThermoParser" ? state.thermoReady : !!state.msconvert;
      const tag = document.createElement("span");
      tag.className = ok ? "hint inline" : "fbad";
      tag.textContent = ok ? v.label : `${v.label} — converter missing`;
      li.appendChild(tag);
    }

    const drop = document.createElement("button");
    drop.className = "fdrop";
    drop.type = "button";
    drop.textContent = "\u00d7";
    drop.title = "Remove";
    drop.setAttribute("aria-label", `Remove ${baseName(f)}`);
    drop.addEventListener("click", () => {
      state.picks.mzml.splice(i, 1);
      delete state.picks.conditions[f];
      delete state.conditionTyped[f];
      renderMzmlList();
      updateRawNote();
    });
    li.appendChild(drop);

    list.appendChild(li);
  }
  refreshNormOptions();
  // The library is for the spectra that are selected, when there are any; the user may
  // still change it.
  if (files.length) {
    $("lib-for").value = anyDiapasef ? "diapasef" : "other";
    refreshLibForNote();
  }
  // Which of them the engine would refuse for want of a Q1 half-width; asked once per file.
  refreshZtscanNote();
}

// -- SCIEX ZT Scan DIA ------------------------------------------------------------------
// A ZT Scan DIA mzML names each MS2 spectrum's quadrupole position and not the window it
// isolated. The engine recognises such a file by its isolation census, reads it with the
// half-width the census suggests and searches it with the ZT Scan analysis by itself
// (docs/37_ztscan.md, "Automatic for ZT Scan files"). The same census of each selected mzML
// (`scanning_probe`) tells this card which files are such: it says what they are, shows the
// half-width (a value the user types is sent instead), offers to turn the analysis off, and
// sets the library's search space while the analysis is on. A file that names targets
// without offsets in any other layout is refused without a half-width the user enters.
// Vendor files are not probed: the census would convert them first.
async function refreshZtscanNote() {
  const z = state.ztscan;
  const todo = state.picks.mzml.filter((f) => !isVendor(f) && !(f in z.probes));
  if (todo.length) {
    // Marked before the call, so a second render while it runs does not ask again.
    for (const f of todo) z.probes[f] = null;
    let got = [];
    try {
      got = await invoke("scanning_probe", { mzml: todo });
    } catch {
      /* Advisory: the engine reports the refusal itself if the file is used. */
    }
    for (const p of got || []) z.probes[p.path] = p.isolation || null;
  }
  if (ztscanFiles().length && !z.off) {
    try {
      z.off = await invoke("ztscan_off");
    } catch {
      z.off = null;
    }
  }
  renderZtscanNote();
}

// The selected files that need a half-width (`convert_refuses`), in selection order: the
// ZT Scan files, whose half-width the engine takes from the census, and any other file that
// names targets without offsets, which it refuses without one.
function ztscanFiles() {
  return state.picks.mzml.filter((f) => state.ztscan.probes[f]?.convert_refuses);
}

function ztscanScanning() {
  return ztscanFiles().filter((f) => state.ztscan.probes[f].scanning_quadrupole);
}

// The engine runs the ZT Scan analysis when every file of the search is a ZT Scan run (its
// keys read the Q1 sweep, and a pooled rescore needs one feature set), so it is offered then.
function ztscanAnalysisAllowed() {
  const scanning = ztscanScanning();
  return scanning.length > 0 && scanning.length === state.picks.mzml.length;
}

// Whether the search runs the ZT Scan analysis: the engine's default for these files, unless
// the user cleared the box.
function ztscanAnalysisOn() {
  return ztscanAnalysisAllowed() && !!$("ztscan-analysis").checked;
}

// The user cleared the analysis but the overrides that turn it off are not known: the engine
// would run it anyway, so the search is not started.
function ztscanAllowedOffMissing() {
  return ztscanAnalysisAllowed() && !ztscanAnalysisOn() && !state.ztscan.off;
}

function renderZtscanNote() {
  const refused = ztscanFiles();
  show($("ztscan-note"), refused.length > 0);
  renderZtscanModsNote();
  if (!refused.length) {
    syncZtscanLibrary();
    return;
  }
  const scanning = ztscanScanning();
  const other = refused.filter((f) => !scanning.includes(f));
  const n = state.picks.mzml.length;
  const parts = [];
  if (scanning.length) {
    const l = state.ztscan.probes[scanning[0]].layout || {};
    const fmt = (v, d) => (Number.isFinite(v) ? v.toFixed(d) : "?");
    const which =
      n === 1
        ? "This file is"
        : `${scanning.length} of the ${n} selected files ${scanning.length === 1 ? "is" : "are"}`;
    parts.push(
      `${which} a scanning-quadrupole acquisition (SCIEX ZT Scan DIA): every MS2 spectrum ` +
        `names the quadrupole's position (${fmtInt(l.distinct_targets)} positions from ` +
        `${fmt(l.lowest_target, 1)} to ${fmt(l.highest_target, 1)} m/z, ` +
        `${fmt(l.median_step, 4)} m/z apart, repeating every ${fmt(l.cycle_s, 2)} s), but not ` +
        "the width of the window it isolated. MuMDIA reads it with the half-width below, " +
        "which assumes a window five positions wide, as on the measured 3.8 m/z method, " +
        "where half-widths from 1.5 to 2.3 m/z identified the same; enter half of your " +
        "method's Q1 window if it differs."
    );
  }
  if (other.length) {
    parts.push(
      `${other.map(baseName).join(", ")} ${other.length === 1 ? "names" : "name"} an ` +
        "isolation target in every MS2 spectrum but no window width, in a layout that is " +
        "not a scanning quadrupole: the converter most likely left the window offsets out. " +
        "Converting the vendor file again with one that writes them is the better fix; the " +
        "half-width below applies to these files too."
    );
  }
  $("ztscan-body").textContent = parts.join(" ");
  if (!state.ztscan.hwTyped) {
    const first = state.ztscan.probes[scanning[0] || refused[0]];
    const hw = first?.suggested_half_width;
    $("ztscan-hw").value = Number.isFinite(hw) ? String(hw) : "";
  }
  // The engine runs the analysis for these files by default, so the box is ticked whenever
  // it applies, until the user ticks or clears it themselves.
  // A choice the user made (or a loaded setup made) holds again once the analysis applies
  // again, after a selection it did not apply to had cleared the box.
  const allowed = ztscanAnalysisAllowed();
  $("ztscan-analysis").disabled = !allowed;
  if (!allowed) $("ztscan-analysis").checked = false;
  else if (!state.ztscan.analysisTouched) $("ztscan-analysis").checked = true;
  else if (typeof state.ztscan.analysisChoice === "boolean") {
    $("ztscan-analysis").checked = state.ztscan.analysisChoice;
  }
  $("ztscan-analysis-note").textContent = allowed
    ? "Measured on one ZT Scan acquisition (K562 on a ZenoTOF 8600): with its library " +
      "settings, about 9% more precursors than the half-width alone, a measured false " +
      "discovery proportion of about 1.2% at a reported 1% (1.4% with the half-width alone), " +
      "and under half the replicate CV of the default quantification; extraction and " +
      "rescoring take longer. Not yet validated on a second acquisition. Clearing the box " +
      "searches the files with the half-width alone."
    : "The ZT Scan analysis needs every selected file to be a ZT Scan run (a pooled search " +
      "scores every run with one feature set), so this search reads the ZT Scan files with " +
      "the half-width and leaves the analysis out. Search the ZT Scan files on their own to " +
      "use it.";
  syncZtscanLibrary();
}

// On ZT Scan the Q1 window passes a precursor's whole isotope envelope, so deamidation of N or
// Q (+0.984 Da) is matched by the 13C isotope (+1.003 Da) of the unmodified peptide eluting with
// it: measured on the K562 runs, half of the precursors a deamidation search reported were such
// forms, at a decoy and entrapment FDP that looked valid, while its peptides and protein groups
// fell (docs/37_ztscan.md, section 13). Said on the card, not refused.
function ztscanDeamidationWarning() {
  if (!ztscanFiles().length) return "";
  const picked = [...state.mods.keys()]
    .map(parseModKey)
    .some((x) => x.name === "Deamidated" && (x.residue === "N" || x.residue === "Q"));
  if (!picked) return "";
  return (
    "Deamidation of N or Q is selected. On ZT Scan the Q1 window passes the whole isotope " +
    "envelope, so a deamidated form (+0.984 Da) is matched by the 13C isotope (+1.003 Da) of " +
    "the unmodified peptide eluting with it: measured on one ZT Scan acquisition, half of the " +
    "precursors such a search reported were these forms, while its peptides and protein groups " +
    "fell, and neither the decoys nor an entrapment library detects it. Leave deamidation out " +
    "of a ZT Scan search."
  );
}

function renderZtscanModsNote() {
  $("ztscan-mods-note").textContent = ztscanDeamidationWarning();
}

// The search space a library built from a FASTA takes under the ZT Scan analysis: the library
// settings measured on the K562 ZT Scan runs (docs/37_ztscan.md, "The library for a ZT Scan
// search"): charges 1 to 4, lengths 6 to 40, methionine oxidation and protein N-terminal
// acetylation as variable modifications, at most two a peptide, and precursors inside the
// swept Q1 range (`ztscanPrecursorRange`).
const ZTSCAN_LIBRARY = {
  fields: { "d-minlen": "6", "d-maxlen": "40", "d-minz": "1", "d-maxz": "4", "d-maxvar": "2" },
  variable: [modKey("Oxidation", "M"), modKey("Acetyl", "*", "protein_n_term")],
};

// The precursor m/z the selected ZT Scan files sweep: the lowest target minus the half-width
// to the highest plus it, rounded outward to whole m/z; null while either is unknown. A
// precursor outside it is never isolated, so leaving it out of the library loses nothing.
function ztscanPrecursorRange() {
  const hw = parseFloat($("ztscan-hw").value);
  if (!(hw > 0)) return null;
  let lo = Infinity;
  let hi = -Infinity;
  for (const f of ztscanScanning()) {
    const l = state.ztscan.probes[f]?.layout || {};
    if (Number.isFinite(l.lowest_target)) lo = Math.min(lo, l.lowest_target);
    if (Number.isFinite(l.highest_target)) hi = Math.max(hi, l.highest_target);
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  return { min: Math.floor(lo - hw), max: Math.ceil(hi + hw) };
}

// The values the recipe sets in the Search space card, by element id.
function ztscanLibraryFields() {
  const out = { ...ZTSCAN_LIBRARY.fields };
  const r = ztscanPrecursorRange();
  if (r) {
    out["d-prmin"] = String(r.min);
    out["d-prmax"] = String(r.max);
  }
  return out;
}

// While the analysis is on, the Search space card holds the ZT Scan library settings, so a
// library built from a FASTA (by DIA-NN or the built-in predictors), the size estimate and
// the cache key all follow them; when it goes off they are taken back. Only what the analysis
// set and the user has not changed since is taken back: a value edited after is the user's,
// and stays. The two modifications are added to the user's choice (the default one,
// carbamidomethyl fixed and oxidation variable, becomes the measured one) and are not saved
// as the user's preference.
function syncZtscanLibrary() {
  const z = state.ztscan;
  let changed = false;
  if (ztscanAnalysisOn()) {
    if (!z.lib) z.lib = { before: {}, set: {}, mods: [] };
    for (const [id, v] of Object.entries(ztscanLibraryFields())) {
      const el = $(id);
      if (!(id in z.lib.set)) z.lib.before[id] = el.value;
      else if (el.value !== z.lib.set[id]) continue;
      if (el.value !== v) {
        el.value = v;
        changed = true;
      }
      z.lib.set[id] = v;
    }
    for (const key of ZTSCAN_LIBRARY.variable) {
      if (!state.mods.has(key) && !z.lib.mods.includes(key)) {
        state.mods.set(key, "var");
        z.lib.mods.push(key);
        changed = true;
      }
    }
  } else if (z.lib) {
    for (const [id, v] of Object.entries(z.lib.set)) {
      const el = $(id);
      if (el.value === v) {
        el.value = z.lib.before[id] ?? "";
        changed = true;
      }
    }
    for (const key of z.lib.mods) {
      if (state.mods.get(key) === "var") {
        state.mods.delete(key);
        changed = true;
      }
    }
    z.lib = null;
  }
  const r = z.lib ? ztscanPrecursorRange() : null;
  $("ztscan-library-note").textContent = z.lib
    ? "A library built from a FASTA for this search covers the ZT Scan search space, set " +
      "under Search space: charges 1 to 4, lengths 6 to 40, methionine oxidation and protein " +
      "N-terminal acetylation as variable modifications, at most two a peptide" +
      (r ? `, and precursors from ${r.min} to ${r.max} m/z, the range the files sweep.` : ".")
    : "";
  if (!changed) return;
  if (state.catalogue) {
    renderMods();
    summariseMods();
  }
  if (state.libSrc === "diann") refreshLibSrcNote();
  scheduleSpace();
}

// What the ZT Scan card adds to the run's configuration. The engine takes the half-width of
// a ZT Scan file from its census and runs the ZT Scan analysis by itself, so only the user's
// choices are written: the half-width when they typed one, or when a selected file names
// targets without offsets in another layout (the engine has no half-width for it), and the
// analysis's keys set off when they cleared the box.
function ztscanOverrides() {
  const files = ztscanFiles();
  if (!files.length) return {};
  const out = {};
  if (ztscanAnalysisAllowed() && !ztscanAnalysisOn() && state.ztscan.off) {
    Object.assign(out, state.ztscan.off);
  }
  const hw = parseFloat($("ztscan-hw").value);
  const other = files.some((f) => !state.ztscan.probes[f].scanning_quadrupole);
  if (Number.isFinite(hw) && hw > 0 && (state.ztscan.hwTyped || other)) {
    out["convert.isolation_half_width"] = hw;
  }
  return out;
}

// What `vendor_of` calls a .d the engine reads natively, with its ion mobility.
const DIAPASEF_LABEL = "Bruker timsTOF diaPASEF .d";

// What the "Library for" choice changes, under the library source chosen above it.
function refreshLibForNote() {
  const diapasef = $("lib-for").value === "diapasef";
  let note;
  if (state.libSrc === "diann") {
    note =
      "DIA-NN predicts one library whatever the instrument, so this choice changes nothing " +
      "for it.";
  } else if (diapasef) {
    note =
      "For a timsTOF diaPASEF .d: the diaPASEF recipe (AlphaPeptDeep, DeepLC and IM2Deep) " +
      "predicts the library with its ion mobility (1/K0) when those are installed, and a " +
      "search of diaPASEF spectra finds it.";
  } else {
    note =
      "For spectra without ion mobility (Orbitrap, Astral and the other instruments); a " +
      "search of such spectra finds it.";
  }
  if (state.picks.mzml.length) note += " Chosen from the selected spectra.";
  $("lib-for-note").textContent = note;
}

// A condition for every file, suggested from the file names: the rule of visDIA's
// Quant QC page (`suggest_conditions`), so the two agree.
//
// 1. The name is cut at its extension (and a compression suffix) and split into tokens at
//    `_`, `-`, `.` and spaces.
// 2. A token `Condition` or `Cond` (any case) names the condition with the token after it,
//    and `ConditionX` / `CondX` names it `X`: `..._Condition_A_REP1.mzML` gives `A`.
// 3. Otherwise a final replicate token is removed (`REP1`, `rep_2`, `R3`, `run4`, `inj1`, a
//    bare number), then the tokens every such file shares at the start and at the end.
//    What is left, joined with `_`, is the condition: `HeLa_ctrl_01` and `HeLa_treat_02`
//    give `ctrl` and `treat`.
// 4. A file whose name leaves nothing in step 3 takes its name after the replicate token
//    is removed.
const COND_WORDS = new Set(["condition", "cond"]);
const COND_JOINED = /^(?:condition|cond)([A-Za-z0-9]+)$/i;
const REP_WORDS = new Set(["rep", "replicate", "r", "run", "inj", "injection", "tech", "techrep", "bio", "biorep"]);
const REP_TOKEN = /^(?:rep|replicate|r|run|inj|injection|tech|techrep|bio|biorep|br|tr)?\d+$/i;

function conditionTokens(path) {
  let name = baseName(path).replace(/\.(?:gz|bz2|xz|zst|zip)$/i, "");
  name = name.replace(/\.[A-Za-z][A-Za-z0-9]{0,5}$/, "");
  return name.split(/[_\-.\s]+/).filter(Boolean);
}

function suggestConditions(files) {
  const out = new Array(files.length).fill("");
  const cores = [];
  for (const [i, f] of files.entries()) {
    const t = conditionTokens(f);
    if (!t.length) {
      out[i] = baseName(f);
      continue;
    }
    let named = null;
    for (const [k, tok] of t.entries()) {
      if (COND_WORDS.has(tok.toLowerCase()) && k + 1 < t.length) {
        named = t[k + 1];
        break;
      }
      const m = COND_JOINED.exec(tok);
      if (m) {
        named = m[1];
        break;
      }
    }
    if (named !== null) {
      out[i] = named;
      continue;
    }
    let core = t;
    if (t.length >= 3 && REP_WORDS.has(t[t.length - 2].toLowerCase()) && /^\d+$/.test(t[t.length - 1])) {
      core = t.slice(0, -2);
    } else if (t.length >= 2 && REP_TOKEN.test(t[t.length - 1])) {
      core = t.slice(0, -1);
    }
    cores.push([i, core]);
  }
  if (cores.length) {
    const lists = cores.map(([, c]) => c);
    const n = Math.min(...lists.map((x) => x.length));
    let pre = 0;
    while (pre < n && lists.every((x) => x[pre] === lists[0][pre])) pre++;
    let suf = 0;
    while (
      suf < n - pre &&
      lists.every((x) => x[x.length - 1 - suf] === lists[0][lists[0].length - 1 - suf])
    ) {
      suf++;
    }
    for (const [i, c] of cores) {
      const rest = c.slice(pre, c.length - suf);
      out[i] = (rest.length ? rest : c).join("_");
    }
  }
  return out;
}

// The condition of each selected file, in file order, trimmed ("" when it has none).
function runConditions() {
  return state.picks.mzml.map((f) => (state.picks.conditions[f] || "").trim());
}

const NORM_NOTES = {
  none:
    "No normalisation between runs: the cross-run quantities (lfq_<run> in proteins.tsv, " +
    "normalised_<run> in precursors.tsv) are the runs' own quantities.",
  all:
    "One size factor per run, estimated over all runs (median of ratios). It assumes most " +
    "proteins do not change between any two runs.",
  condition:
    "Size factors estimated within each condition's runs only: replicates are made " +
    "comparable, and the differences between conditions stay as measured. A loading " +
    "difference between conditions is not corrected. Quantification uses the pattern-fit " +
    "recipe (profile-centred window, pattern fit, background chosen from the data) without " +
    "the stable-set normalisation, which is estimated over every run: the per-run " +
    "quantities are not normalised, and MaxLFQ is normalised within each condition.",
};

// The normalisation choice and its note, shown for an experiment of several files: they
// sit under "One experiment" and are hidden and disabled for "Search each separately" and
// for a single file. `normOverrides` checks the same condition, so a hidden choice is
// never sent.
function refreshNormOptions() {
  const several = experimentWithSeveral();
  show($("norm-options"), several);
  $("norm-scope").disabled = !several;
  $("suggest-conditions").disabled = !several;
  for (const el of document.querySelectorAll("#mzml-list .fcond")) show(el, several);
  if (!several) return;
  const scope = $("norm-scope").value;
  const conds = runConditions();
  const labels = [...new Set(conds.filter(Boolean))];
  $("condition-labels").innerHTML = labels
    .map((c) => `<option value="${escHtml(c)}"></option>`)
    .join("");
  let note = NORM_NOTES[scope] || "";
  if (scope === "condition") {
    const missing = conds.filter((c) => !c).length;
    if (missing) note += ` ${missing} file(s) have no condition yet.`;
    else {
      note += ` Conditions: ${labels
        .map((c) => `${c} (${conds.filter((x) => x === c).length})`)
        .join(", ")}.`;
    }
  }
  $("norm-note").textContent = note;
}

// The configuration the normalisation choice adds: nothing at the default.
function normOverrides() {
  const scope = $("norm-scope").value;
  return experimentWithSeveral() && scope !== "all"
    ? { "quant.normalisation_scope": scope }
    : {};
}

function currentRunMode() {
  // One file is a plain search whatever the radio says; the backend refuses a
  // one-file experiment rather than quietly demoting it, so do not send one.
  return state.picks.mzml.length > 1 && state.runMode === "experiment"
    ? "experiment"
    : "separate";
}

async function pick(what) {
  // Spectra accept several at once; everything else is a single choice.
  const many = what === "mzml";
  const chosen = DIR_PICKS.has(what)
    ? await dialog.open({ directory: true, multiple: false })
    : await dialog.open({ multiple: many, filters: FILTERS[what] });
  if (!chosen) return;

  if (what === "mzml" || what === "mzml_dir") {
    const added = (Array.isArray(chosen) ? chosen : [chosen]).filter(Boolean);
    // Silently dropping a duplicate is right: the backend refuses the same path
    // twice (it would search one file twice and pool the result with itself), and a
    // user who picks a file again means "have it", not "have it twice".
    for (const f of added) {
      if (!state.picks.mzml.includes(f)) state.picks.mzml.push(f);
    }
    await renderMzmlList();
    updateRawNote();
    refreshPrescreenNote();
    scheduleBuiltinPlan();
    // The peak census reads one file, and only an mzML: with several selected there
    // is no single answer to show, and for a vendor path it would convert first.
    if (state.picks.mzml.length === 1 && !isVendor(state.picks.mzml[0])) {
      showPeakCensus(state.picks.mzml[0]);
    } else {
      show($("peak-note"), false);
    }
    banner($("start-error"), "");
    return;
  }

  const path = Array.isArray(chosen) ? chosen[0] : chosen;
  state.picks[what] = path;
  if (what === "fasta") refreshLibSrcNote();
  if (what === "fasta" || what === "lib_precursors") scheduleSpace();
  if (what === "diann_lib" || what === "diann_lib_out") banner($("diann-import-error"), "");
  const el = $(LABEL[what]);
  // Shown right-to-left so the filename stays visible on a long path; the full
  // path is the tooltip.
  el.textContent = path;
  el.title = path;
  el.classList.add("set");
  banner($("start-error"), "");
}

// ── the peak cap, answered from the file ────────────────────────────────────
// The documentation is emphatic that a peak cap is acquisition-specific, and that
// one carried from another run deletes fragment evidence rather than failing: on a
// 50-window Orbitrap DIA run a 300-peak cap discarded 78.6% of MS2 peaks and cost
// 60% of the peptides. The application has the file, so it answers the question
// rather than leaving a number box for someone to guess into.
async function showPeakCensus(mzml) {
  const panel = $("peak-note");
  const body = $("peak-body");
  body.textContent = "Reading the file…";
  show(panel, true);
  let c;
  try {
    c = await invoke("peak_census", { mzml });
  } catch {
    // Not being able to read it here is not a reason to say anything alarming; the
    // engine will report the real problem if the file is unusable.
    show(panel, false);
    return;
  }
  const p = c.peaks_per_ms2;
  const at300 = (c.caps || []).find((x) => x.cap === 300);
  const lines = [
    `${fmtInt(c.ms2_spectra)} MS2 spectra sampled. Peaks per spectrum: ` +
      `p25 ${fmtInt(p.p25)}, median ${fmtInt(p.p50)}, p95 ${fmtInt(p.p95)}, ` +
      `max ${fmtInt(p.max)}.`,
  ];
  if (at300 && at300.fraction_of_peaks_discarded > 0.02) {
    lines.push(
      `A 300-peak cap would truncate ${(at300.fraction_of_spectra_truncated * 100).toFixed(0)}% ` +
        `of spectra and discard ${(at300.fraction_of_peaks_discarded * 100).toFixed(0)}% of all ` +
        `peaks. Uncapped is the default, and is right for this file.`
    );
  } else {
    lines.push("No cap is applied by default, which is right for this file.");
  }
  if (c.profile_ms2_spectra > 0) {
    lines.push(
      `${fmtInt(c.profile_ms2_spectra)} spectra are profile mode, so these are raw ` +
        `sample counts rather than centroided peaks.`
    );
  }
  body.textContent = lines.join(" ");
}

// ── starting and polling ────────────────────────────────────────────────────
// -- FASTA -> DIA-NN library, driven from the Search tab -----------------
// The library depends only on the FASTA and the digest parameters, so it is built
// into a content-addressed cache and reused. That is what makes offering this on the
// search screen honest: it is a one-time cost per FASTA, not a cost per search.
//
// Chained here rather than in the backend, deliberately. `start_run` is the one path
// with end-to-end tests and it stays untouched; the cost is that a webview reload
// mid-build loses the chain. The cache makes that harmless: press Start again and the
// library is already there.
// Read from the Search screen, where these inputs now live.
//
// They used to sit on the Setup screen inside a separate "Predict a library" card
// while the control that consumed them was here, so invisible state on a screen the
// user need never have opened decided the search space, the cache key, and whether
// the "already built" note was even true. The `if (!el) return dflt` guards made
// every mismatch silent. That card is gone; this flow is the only way to build one.
// `read` gives a field's text by element id; a setup reads the fields as they would be with
// the ZT Scan analysis's library taken back (`captureSearch`).
function libraryParams(fasta, read = (id) => $(id)?.value) {
  const n = (id, dflt) => {
    const raw = read(id);
    if (raw === undefined || raw === null) return dflt;
    const v = parseInt(raw, 10);
    return Number.isFinite(v) ? v : dflt;
  };
  const mz = (id) => {
    const raw = read(id);
    if (raw === undefined || raw === null || String(raw).trim() === "") return null;
    const v = parseFloat(raw);
    return Number.isFinite(v) && v > 0 ? v : null;
  };
  const enzyme = read("d-enzyme") ?? "trypsin_p";
  return {
    fasta,
    out_dir: "",
    enzyme,
    cleavage_rule: enzyme === "custom" ? (read("d-rule") || "").trim() : null,
    missed_cleavages: n("d-missed", 1),
    min_pep_len: n("d-minlen", 7),
    max_pep_len: n("d-maxlen", 30),
    min_charge: n("d-minz", 2),
    max_charge: n("d-maxz", 4),
    // Empty means no limit; only a set limit reaches the build or the configuration.
    min_pr_mz: mz("d-prmin"),
    max_pr_mz: mz("d-prmax"),
    min_fr_mz: mz("d-frmin"),
    max_fr_mz: mz("d-frmax"),
    threads: n("d-threads", 8),
    carbamidomethyl: state.mods.get(modKey("Carbamidomethyl", "C")) === "fix",
    oxidation: state.mods.get(modKey("Oxidation", "M")) === "var",
    // The full choice, for a DIA-NN build: `--fixed-mod` / `--var-mod` per modification.
    // The two flags above stay for the standard selection's cache key.
    fixed_mods: state.catalogue ? selectedMods().fixed : [],
    variable_mods: state.catalogue ? selectedMods().variable : [],
    max_variable_mods: maxVariableMods(),
  };
}

// The same digest fields as engine configuration paths, for the built-in library path.
//
// Before this the fields were read only when DIA-NN built the library; with the
// built-in predictors the engine digested with whatever the preset said, and a
// missed-cleavage count typed on this screen reached nothing. The values go on top of
// the chosen preset through `derive_config`, so the preset's predictor and rescorer
// choices survive. Modification lists are the whole list: an unticked box means none.
function digestOverrides() {
  const d = libraryParams("");
  const cleavage = { "digest.enzyme": d.enzyme };
  if (d.enzyme === "custom") cleavage["digest.cleavage_rule"] = d.cleavage_rule;
  return {
    ...cleavage,
    "digest.missed_cleavages": d.missed_cleavages,
    "digest.min_len": d.min_pep_len,
    "digest.max_len": d.max_pep_len,
    "peptidoforms.charge_min": d.min_charge,
    "peptidoforms.charge_max": d.max_charge,
    "peptidoforms.fixed_mods": selectedMods().fixed,
    "peptidoforms.variable_mods": selectedMods().variable,
    "peptidoforms.max_variable_mods": maxVariableMods(),
    // Only a limit that is set: an unset one stays out of the configuration, and so out
    // of its hash and the library cache key.
    ...mzOverrides(d),
  };
}

function mzOverrides(d) {
  const out = {};
  for (const [key, v] of [
    ["peptidoforms.precursor_mz_min", d.min_pr_mz],
    ["peptidoforms.precursor_mz_max", d.max_pr_mz],
    ["predict_frag.fragment_mz_min", d.min_fr_mz],
    ["predict_frag.fragment_mz_max", d.max_fr_mz],
  ]) {
    if (v !== null) out[key] = v;
  }
  return out;
}

// ── modifications ───────────────────────────────────────────────────────────
// The list is the engine's own catalogue (`modifications.json` in mumdia-core, through
// the `modifications` command), so every name offered is one the engine accepts. Each
// residue chip cycles not searched -> variable -> fixed.
const MOD_DEFAULTS = [
  ["Carbamidomethyl", "C", "fix"],
  ["Oxidation", "M", "var"],
];
const FALLBACK_CATALOGUE = {
  modifications: [
    { name: "Carbamidomethyl", unimod: 4, mass: 57.021463735, residues: "C", group: "Alkylation" },
    { name: "Oxidation", unimod: 35, mass: 15.99491462, residues: "M", group: "Oxidation and artefacts" },
  ],
  sets: [],
};

// A residue site is "name|R", as it always was; a terminal site is "name|*|position"
// (`protein_n_term` or `peptide_n_term`), so saved residue choices still read.
const TERMINI = {
  protein_n_term: { chip: "Prot N-term", text: "protein N-terminus" },
  peptide_n_term: { chip: "Pep N-term", text: "peptide N-terminus" },
};

function modKey(name, residue, position) {
  return position ? `${name}|${residue}|${position}` : `${name}|${residue}`;
}

function parseModKey(key) {
  const [name, residue, position] = String(key).split("|");
  return { name, residue, position: position || "" };
}

function maxVariableMods(raw = $("d-maxvar").value) {
  const v = parseInt(raw, 10);
  return Number.isFinite(v) && v >= 0 ? v : 1;
}

// The choice as engine configuration, in catalogue order then residue order, so the
// same choice always derives the same configuration.
function selectedMods(mods = state.mods) {
  const fixed = [];
  const variable = [];
  for (const m of state.catalogue?.modifications || []) {
    const picked = [...mods.entries()]
      .map(([k, kind]) => ({ ...parseModKey(k), kind }))
      .filter((x) => x.name === m.name)
      .sort((a, b) => (a.residue + a.position).localeCompare(b.residue + b.position));
    for (const x of picked) {
      const site = x.position
        ? { residue: x.residue, name: m.name, position: x.position }
        : { residue: x.residue, name: m.name };
      (x.kind === "fix" ? fixed : variable).push(site);
    }
  }
  return { fixed, variable };
}

// What the engine would refuse, said before the search starts rather than by the
// peptidoform stage.
function modProblems() {
  const { fixed, variable } = selectedMods();
  const out = [];
  // A terminal site is keyed by the terminus: the engine places one group there.
  const byResidue = (list) => {
    const m = new Map();
    for (const x of list) {
      const at = x.position ? "the N-terminus" : x.residue;
      m.set(at, [...(m.get(at) || []), x.name]);
    }
    return m;
  };
  const fx = byResidue(fixed);
  const vr = byResidue(variable);
  for (const [r, names] of fx) {
    if (names.length > 1) {
      out.push(`${r} has two fixed modifications (${names.join(", ")}); a residue can carry one.`);
    }
    if (vr.has(r)) {
      out.push(
        `${r} is fixed (${names.join(", ")}) and also variable (${vr.get(r).join(", ")}); ` +
          "the engine refuses both on one residue. Make the fixed one variable, or remove one."
      );
    }
  }
  return out;
}

function saveMods() {
  try {
    localStorage.setItem("mumdia.mods", JSON.stringify([...state.mods]));
    localStorage.setItem("mumdia.maxvar", String(maxVariableMods()));
  } catch {
    // A convenience only: the defaults come back next time.
  }
}

function loadMods() {
  state.mods = new Map();
  let saved = null;
  try {
    saved = JSON.parse(localStorage.getItem("mumdia.mods") || "null");
    const mv = localStorage.getItem("mumdia.maxvar");
    if (mv !== null && Number.isFinite(parseInt(mv, 10))) $("d-maxvar").value = mv;
  } catch {
    saved = null;
  }
  const known = new Set(state.catalogue.modifications.map((m) => m.name));
  const entries = Array.isArray(saved)
    ? saved
    : MOD_DEFAULTS.map(([n, r, k]) => [modKey(n, r), k]);
  for (const e of entries) {
    if (!Array.isArray(e) || e.length !== 2) continue;
    const [key, kind] = e;
    const { name, residue, position } = parseModKey(key);
    const site = position ? residue === "*" && position in TERMINI : /^[A-Z]$/.test(residue || "");
    if (known.has(name) && site && (kind === "var" || kind === "fix")) {
      state.mods.set(key, kind);
    }
  }
}

// ── cleavage ────────────────────────────────────────────────────────────────
// The presets are the engine's own `enzymes.json` (through the `enzymes` command), so a
// rule offered here is one the engine's digest and the DIA-NN build both apply. A custom
// rule uses DIA-NN's `--cut` pair syntax. Under a rule that cuts every bond peptides are
// enumerated by length, so the missed-cleavage count does not apply.
let ENZYMES = [{ name: "trypsin_p", label: "Trypsin/P", note: "after K or R, also before P", rule: "K*,R*" }];

function cleavageChanged() {
  const name = $("d-enzyme").value;
  const p = ENZYMES.find((e) => e.name === name) || ENZYMES[0];
  show($("d-rule-row"), name === "custom");
  const nonSpecific = p.rule === "**" || (name === "custom" && ($("d-rule").value || "").trim() === "**");
  $("d-missed").disabled = nonSpecific;
  const rule = name === "custom" ? "pairs around the cut, * any residue, ! never cut" : p.rule;
  $("d-enzyme-note").textContent =
    `${p.label}: ${p.note}${rule ? ` (rule ${rule})` : ""}.` +
    (nonSpecific ? " Missed cleavages do not apply; the search space grows with the length range." : "");
  try { localStorage.setItem("mumdia.cleavage", JSON.stringify({ name, rule: $("d-rule").value })); } catch {}
  if (state.libSrc === "diann") refreshLibSrcNote();
  scheduleSpace();
}

async function initCleavage() {
  try {
    ENZYMES = await invoke("enzymes");
  } catch {}
  const sel = $("d-enzyme");
  sel.innerHTML = "";
  for (const e of ENZYMES) {
    const o = document.createElement("option");
    o.value = e.name;
    o.textContent = e.label;
    sel.appendChild(o);
  }
  sel.value = "trypsin_p";
  try {
    const saved = JSON.parse(localStorage.getItem("mumdia.cleavage") || "null");
    if (saved && ENZYMES.some((e) => e.name === saved.name)) {
      sel.value = saved.name;
      $("d-rule").value = saved.rule || "";
    }
  } catch {}
  sel.addEventListener("change", cleavageChanged);
  $("d-rule").addEventListener("change", cleavageChanged);
  cleavageChanged();
}

async function initMods() {
  try {
    state.catalogue = await invoke("modifications");
  } catch {
    state.catalogue = FALLBACK_CATALOGUE;
  }
  loadMods();
  const sel = $("mod-set");
  (state.catalogue.sets || []).forEach((set, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = set.name;
    sel.appendChild(o);
  });
  sel.addEventListener("change", () => {
    const set = (state.catalogue.sets || [])[parseInt(sel.value, 10)];
    sel.value = "";
    if (!set) return;
    if (set.replace) state.mods.clear();
    for (const [n, r, pos] of set.fixed || []) state.mods.set(modKey(n, r, pos), "fix");
    for (const [n, r, pos] of set.variable || []) state.mods.set(modKey(n, r, pos), "var");
    modsChanged();
  });
  $("mod-filter").addEventListener("input", renderMods);
  $("d-maxvar").addEventListener("change", modsChanged);
  renderMods();
  summariseMods();
}

function modsChanged() {
  saveMods();
  renderMods();
  summariseMods();
  renderZtscanModsNote();
  if (state.libSrc === "diann") refreshLibSrcNote();
  scheduleSpace();
}

function renderMods() {
  const list = $("mod-list");
  const scroll = list.scrollTop;
  const words = $("mod-filter").value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  list.textContent = "";
  let group = null;
  for (const m of state.catalogue.modifications) {
    // The catalogue's residues, plus any a saved or set choice put elsewhere, so a
    // selected site is never invisible.
    const chosen = [...state.mods.keys()].map(parseModKey).filter((x) => x.name === m.name);
    const extra = chosen.filter((x) => !x.position).map((x) => x.residue);
    const residues = [...new Set([...m.residues, ...extra])];
    // The catalogue's termini, plus any a saved or set choice put there.
    const termini = [
      ...new Set([...(m.termini || []), ...chosen.filter((x) => x.position).map((x) => x.position)]),
    ].filter((t) => t in TERMINI);
    const sites = [
      ...residues.map((r) => ({ key: modKey(m.name, r), label: r, text: r })),
      ...termini.map((t) => ({
        key: modKey(m.name, "*", t),
        label: TERMINI[t].chip,
        text: `the ${TERMINI[t].text}`,
      })),
    ];
    // The optional label is a plain-language description ("N-homocysteinylation") for a
    // UniMod name that does not say what it is, so it is shown and matched by the filter.
    const hay = `${m.name} ${m.label || ""} ${m.group} ${residues.join(" ")} ${termini
      .map((t) => `${TERMINI[t].chip} ${TERMINI[t].text}`)
      .join(" ")} unimod:${m.unimod}`.toLowerCase();
    if (words.length && !words.every((w) => hay.includes(w))) continue;
    if (m.group !== group) {
      group = m.group;
      const h = document.createElement("div");
      h.className = "modgroup";
      h.textContent = group;
      list.appendChild(h);
    }
    const row = document.createElement("div");
    row.className = "modrow";
    const label = document.createElement("div");
    const name = document.createElement("span");
    name.className = "mname";
    name.textContent = m.name;
    const mass = document.createElement("span");
    mass.className = "mmass";
    mass.textContent = `${m.mass >= 0 ? "+" : ""}${m.mass.toFixed(4)} Da · UniMod:${m.unimod}`;
    label.append(name, mass);
    if (m.label) {
      const desc = document.createElement("div");
      desc.className = "mlabel";
      desc.textContent = m.label;
      label.appendChild(desc);
    }
    const chips = document.createElement("div");
    chips.className = "chips";
    let on = false;
    for (const { key, label: chipLabel, text } of sites) {
      const kind = state.mods.get(key);
      if (kind) on = true;
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip" + (kind === "var" ? " var" : kind === "fix" ? " fix" : "");
      b.textContent = chipLabel;
      b.title = `${m.name} on ${text}: ${
        kind === "var" ? "variable" : kind === "fix" ? "fixed" : "not searched"
      }`;
      b.addEventListener("click", () => {
        const cur = state.mods.get(key);
        if (!cur) state.mods.set(key, "var");
        else if (cur === "var") state.mods.set(key, "fix");
        else state.mods.delete(key);
        modsChanged();
      });
      chips.appendChild(b);
    }
    row.classList.toggle("on", on);
    row.append(label, chips);
    list.appendChild(row);
  }
  if (!list.childElementCount) {
    const none = document.createElement("div");
    none.className = "modrow";
    none.textContent = "No modification matches the filter.";
    list.appendChild(none);
  }
  list.scrollTop = scroll;
}

function describeMods(list) {
  const by = new Map();
  for (const x of list) {
    const at = x.position ? TERMINI[x.position]?.text || x.position : x.residue;
    by.set(x.name, [...(by.get(x.name) || []), at]);
  }
  return [...by].map(([n, rs]) => `${n} (${rs.join(", ")})`).join(", ");
}

function summariseMods() {
  const { fixed, variable } = selectedMods();
  const parts = [];
  parts.push(fixed.length ? `Fixed: ${describeMods(fixed)}.` : "No fixed modification.");
  parts.push(
    variable.length
      ? `Variable: ${describeMods(variable)}, at most ${maxVariableMods()} per peptide.`
      : "No variable modification."
  );
  const problems = modProblems();
  if (problems.length) parts.push(problems.join(" "));
  const space = spaceLine(state.space);
  if (state.mode === "fasta" && space) parts.push(space);
  $("mod-summary").textContent = parts.join(" ");
}

// ── prescreen and bands ─────────────────────────────────────────────────────
// Measured numbers are from docs/34_prescreen.md: the fraction of candidates removed,
// and the fraction of an unfiltered search's accepted precursors still accepted.
const PRESCREEN_NOTES = {
  off:
    "Off: every library candidate is searched, which keeps every identification within " +
    "reach. The default.",
  before:
    "Scores every candidate against the spectra of its isolation window before DeepLC and " +
    "MS2PIP run, without retention times or predicted intensities, so predictions are made " +
    "only for the candidates kept; everything after it uses predicted retention times as " +
    "usual. Measured at balanced: 54% removed with 99.0% of an unfiltered search's accepted " +
    "precursors kept on immunopeptidomics Astral data, 56% with 100% on Orbitrap AIF, and " +
    "52% with 95.2% on HYE Astral, where light kept 99.6%. What it removes cannot be " +
    "identified, so the kept fraction is the cost in identifications. One file at a time.",
  after:
    "Scores every candidate inside its retention-time window after the calibration. " +
    "Predictions are still made for the whole library; the saving is in extraction, " +
    "features and rescoring. Measured at balanced: 54% removed with 99.4% kept on HYE " +
    "Astral, 54% with 98.9% on immunopeptidomics. What it removes cannot be identified. " +
    "Cannot be combined with bands yet.",
  tags:
    "Keeps a candidate only when one of its three-residue sequence tags was read from a " +
    "spectrum of its isolation window; database-free, before prediction and without " +
    "retention times. Measured: 56-58% removed with 100% kept on Orbitrap AIF and on a " +
    "FASTA E. coli search, 17% on HYE Astral, nothing on immunopeptidomics. A peptide " +
    "whose tags were all missed cannot be identified. One file at a time.",
};
// The fraction of the library each choice is expected to keep, for the band plan.
const PRESCREEN_KEEP = { light: 0.75, balanced: 0.46, stringent: 0.25 };

function initSearchOptions() {
  $("prescreen").addEventListener("change", () => {
    refreshPrescreenNote();
    scheduleSpace();
  });
  $("prescreen-strength").addEventListener("change", scheduleSpace);
  $("bands").addEventListener("change", () => {
    show($("bands-custom"), $("bands").value === "custom");
    renderSpace();
  });
  for (const id of ["bands-n", "bands-parallel"]) $(id).addEventListener("input", renderSpace);
  for (const id of ["d-missed", "d-minlen", "d-maxlen", "d-minz", "d-maxz", "d-prmin", "d-prmax", "threads"]) {
    $(id).addEventListener("change", scheduleSpace);
  }
  for (const r of document.querySelectorAll('input[name="runmode"]')) {
    r.addEventListener("change", refreshPrescreenNote);
  }
  refreshPrescreenNote();
  scheduleSpace();
}

function refreshPrescreenNote() {
  const mode = $("prescreen").value;
  show($("prescreen-strength-row"), mode === "before" || mode === "after");
  let note = PRESCREEN_NOTES[mode] || "";
  if ((mode === "before" || mode === "tags") && experimentWithSeveral()) {
    note += " With several files searched as one experiment it is refused; choose \"Search each separately\".";
  }
  $("prescreen-note").textContent = note;
}

function experimentWithSeveral() {
  return state.picks.mzml.length > 1 && currentRunMode() === "experiment";
}

function prescreenOverrides() {
  const mode = $("prescreen").value;
  const strength = $("prescreen-strength").value;
  const preset =
    strength === "light"
      ? { "prescreen.preset": "custom", "prescreen.target": 0.25 }
      : { "prescreen.preset": strength };
  if (mode === "before") {
    // Crowding 0.25: the no-retention-time score of docs/34, measured with it.
    return {
      "prescreen.score_before_prediction": true,
      "prescreen.crowding_exponent": 0.25,
      ...preset,
    };
  }
  if (mode === "after") return { "prescreen.enabled": true, ...preset };
  if (mode === "tags") return { "prescreen.tag_prefilter": true };
  return {};
}

function prescreenKeep() {
  return $("prescreen").value === "before"
    ? PRESCREEN_KEEP[$("prescreen-strength").value] ?? 1
    : 1;
}

function spaceRequest(libPrecursors) {
  const d = libraryParams("");
  const threads = parseInt($("threads").value, 10);
  const library = libPrecursors || (state.mode === "library" ? state.picks.lib_precursors : "");
  return {
    mode: library ? "library" : "fasta",
    fasta: state.picks.fasta || null,
    lib_precursors: library || null,
    enzyme: d.enzyme,
    cleavage_rule: d.cleavage_rule,
    missed_cleavages: d.missed_cleavages,
    min_len: d.min_pep_len,
    max_len: d.max_pep_len,
    min_charge: d.min_charge,
    max_charge: d.max_charge,
    precursor_mz_min: d.min_pr_mz,
    precursor_mz_max: d.max_pr_mz,
    variable: selectedMods().variable,
    max_variable_mods: maxVariableMods(),
    keep_fraction: prescreenKeep(),
    threads: Number.isFinite(threads) && threads > 0 ? threads : null,
  };
}

function scheduleSpace() {
  clearTimeout(state.spaceTimer);
  state.spaceTimer = setTimeout(refreshSpace, 400);
}

async function refreshSpace() {
  const seq = ++state.spaceSeq;
  let s = null;
  try {
    s = await invoke("search_space", { req: spaceRequest() });
  } catch (e) {
    s = { reason: `Could not size the search: ${e}`, plan: null, machine: null };
  }
  if (seq !== state.spaceSeq) return;
  state.space = s;
  renderSpace();
  summariseMods();
}

function fmtCount(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e4) return `${Math.round(n / 1e3)}k`;
  return String(n);
}

function spaceLine(s) {
  if (!s) return "";
  if (s.precursors == null) return s.reason || "";
  let line = s.exact
    ? `The library has ${fmtCount(s.precursors)} precursors.`
    : `About ${fmtCount(s.peptides)} peptides and ${fmtCount(s.precursors)} library precursors ` +
      "with their decoys, estimated from the FASTA.";
  if (s.searched_precursors != null && s.searched_precursors < s.precursors) {
    line += ` About ${fmtCount(s.searched_precursors)} after the prescreen.`;
  }
  return line;
}

function customBands() {
  const n = parseInt($("bands-n").value, 10);
  const p = parseInt($("bands-parallel").value, 10);
  const bands = Number.isFinite(n) && n > 0 ? n : 1;
  return { bands, parallel: Math.min(Number.isFinite(p) && p > 0 ? p : 1, bands) };
}

function renderSpace() {
  const s = state.space;
  const plan = s?.plan;
  const mode = $("bands").value;
  const parts = [];
  if (state.mode === "library") {
    const line = spaceLine(s);
    if (line) parts.push(line);
  }
  if (mode === "auto") {
    parts.push(plan ? plan.note : "Choose the search space to size it.");
  } else if (mode === "off") {
    parts.push(
      plan?.unbanded_gb != null
        ? `Everything at once: about ${plan.unbanded_gb.toFixed(0)} GB estimated, on a ` +
            `machine that can give a search about ${plan.budget_gb.toFixed(0)} GB.`
        : "Everything at once."
    );
  } else {
    const c = customBands();
    let line = `${c.bands} band${c.bands === 1 ? "" : "s"}, ${c.parallel} at a time.`;
    if (plan?.unbanded_gb != null) {
      const per = 4 + (plan.unbanded_gb - 4) / c.bands;
      line += ` About ${(per * c.parallel).toFixed(0)} GB in flight against a budget of ${plan.budget_gb.toFixed(0)} GB.`;
    }
    parts.push(line);
  }
  const m = s?.machine;
  if (m?.total_memory_bytes) {
    parts.push(`This machine: ${(m.total_memory_bytes / 1e9).toFixed(0)} GB of memory, ${m.threads} threads.`);
  }
  $("bands-note").textContent = parts.join(" ");
}

// The band count this search uses. Automatic sizes the inputs being started now, so a
// stale plan from before the last change cannot decide it.
async function resolveBands(libPrecursors) {
  const mode = $("bands").value;
  if (mode === "off") return { bands: 1, parallel: 1, explicit: true };
  if (mode === "custom") return { ...customBands(), explicit: true };
  try {
    const s = await invoke("search_space", { req: spaceRequest(libPrecursors) });
    state.space = s;
    renderSpace();
    return { bands: s.plan.bands, parallel: s.plan.parallel, explicit: false };
  } catch {
    return { bands: 1, parallel: 1, explicit: false };
  }
}

// What the viewer reads beyond the defaults. `quant.fragment_detail` writes, beside each
// run's quantities, which fragments made each quantity (`quant_fragments.parquet`), so the
// viewer can mark them on the chromatograms. It costs one small table per run and changes no
// other output, so every search started here asks for it.
function viewerOverrides() {
  return { "quant.fragment_detail": true };
}

// What the Search screen adds to the run's configuration, on top of the chosen preset: the
// digest and modification fields for a library the engine builds from a FASTA, and the
// prescreen, the band plan, the normalisation scope, a library's retention-time handling
// and the ZT Scan choices in every mode. `bands` is `resolveBands`'s answer, `built` the
// tables a DIA-NN build produced, `libraryIrt` what a library's library.json names.
function runOverrides({ bands, built = null, libraryIrt = null }) {
  return {
    ...viewerOverrides(),
    ...(state.mode === "fasta" && !built ? digestOverrides() : {}),
    ...prescreenOverrides(),
    ...bandOverrides(bands),
    ...normOverrides(),
    ...(libraryIrt ? { "rt_im_train.library_irt": libraryIrt } : {}),
    ...ztscanOverrides(),
  };
}

function bandOverrides(b) {
  // Automatic with one band leaves whatever the preset says; Off and Custom are explicit.
  if (b.bands > 1 || b.explicit) {
    return { "groups.window_groups": b.bands, "groups.parallel": b.parallel };
  }
  return {};
}

// Choices the engine would refuse, found before anything is built or started.
function searchProblems() {
  const out = [];
  if (state.mode === "fasta") {
    out.push(...modProblems());
  }
  const pre = $("prescreen").value;
  if ((pre === "before" || pre === "tags") && experimentWithSeveral()) {
    out.push(
      "The prescreen before prediction searches one file at a time. Choose \"Search each " +
        "separately\" for these files, or the prescreen after retention-time calibration."
    );
  }
  if (experimentWithSeveral() && $("norm-scope").value === "condition") {
    const missing = state.picks.mzml.filter((f, i) => !runConditions()[i]);
    if (missing.length) {
      out.push(
        "Normalisation within each condition needs the condition of every file. Without " +
          `one: ${missing.map(baseName).join(", ")}.`
      );
    }
  }
  if (pre === "after" && $("bands").value === "custom" && customBands().bands > 1) {
    out.push(
      "The prescreen after retention-time calibration cannot be combined with bands yet. " +
        "Set Bands to Off, or choose the prescreen before prediction."
    );
  }
  const zt = ztscanFiles();
  if (zt.length && !(parseFloat($("ztscan-hw").value) > 0)) {
    out.push(
      "These files name the quadrupole's position but not the width of the window it " +
        "isolated, and cannot be searched without the Q1 half-width: " +
        `${zt.map(baseName).join(", ")}. Enter it in the SCIEX ZT Scan DIA card.`
    );
  }
  if (ztscanAllowedOffMissing()) {
    out.push(
      "The ZT Scan analysis could not be turned off: the settings that do so could not be " +
        "read. Tick it again, or restart the application."
    );
  }
  if (ztscanAnalysisOn() && $("bands").value === "custom" && customBands().bands > 1) {
    out.push(
      "The ZT Scan analysis calibrates the Q1 sweep of the whole run, which cannot be split " +
        "into bands. Set Bands to Off, or clear the ZT Scan analysis."
    );
  }
  return out;
}

// Say, on the search screen, whether choosing DIA-NN means waiting.
//
// DIA-NN is the default source. While it cannot be used (not installed, or its licence
// not acknowledged) the built-in predictors are selected instead, and DIA-NN comes back
// by itself once it can be used, unless the user picked one of the two explicitly.
async function refreshLibSrcNote() {
  const note = $("libsrc-diann-note");
  const radio = $("libsrc-diann");
  if (!note || !radio) return;
  const unavailable = (why) => {
    radio.disabled = true;
    note.textContent = why;
    if (state.libSrc === "diann") selectLibSrc("builtin");
  };
  const available = () => {
    radio.disabled = false;
    if (!state.libSrcChosen && state.libSrc !== "diann") selectLibSrc("diann");
  };

  const fasta = state.picks.fasta;
  if (!fasta) {
    let d;
    try {
      d = (await invoke("diann_status")).status;
    } catch (e) {
      return unavailable(`Unavailable: ${e}`);
    }
    if (!d.runs) return unavailable("No working DIA-NN was found; see Setup.");
    if (!d.licence_acknowledged) {
      return unavailable("Acknowledge the DIA-NN licence on the Setup screen first.");
    }
    available();
    note.textContent = "Choose a FASTA and DIA-NN predicts the library from it before the search.";
    return;
  }
  let plan;
  try {
    plan = await invoke("diann_library_plan", { req: libraryParams(fasta) });
  } catch (e) {
    // The reason is the backend's: no DIA-NN, or the licence not acknowledged.
    return unavailable(`Unavailable: ${e}`);
  }
  available();
  note.textContent = plan.ready
    ? `Already built for this FASTA — the search will start immediately (${plan.diann_version}).`
    : `Not built yet: DIA-NN will predict it first, which takes a while on a whole proteome (${plan.diann_version}).`;
}

function selectLibSrc(which) {
  state.libSrc = which;
  const r = $(which === "diann" ? "libsrc-diann" : "libsrc-builtin");
  if (r) r.checked = true;
  show($("libsrc-params"), which === "diann");
  refreshLibForNote();
  scheduleBuiltinPlan();
}

// ── built-in predictors ─────────────────────────────────────────────────────
// A FASTA search on the built-in path predicts its library with what its configuration
// names, and the preset selected by default (the imported-library one) names no fragment
// predictor, which left such searches on the engine's native models. `builtin_plan`
// replaces a preset that names none with the first recipe whose interpreters the engine
// finds (the diaPASEF recipe for a timsTOF diaPASEF .d), and says when the search would
// still run on native predictions, which blocks Start: the engine refuses them. Shown
// here while the choice is being made, and checked again at Start.
function scheduleBuiltinPlan() {
  clearTimeout(state.builtinTimer);
  state.builtinTimer = setTimeout(refreshBuiltinPlan, 400);
}

// `diapasef` is the acquisition a library built without spectra is for; a search leaves
// it unset and the backend reads it from the selected spectra.
async function builtinPlan(diapasef = null) {
  return invoke("builtin_plan", {
    preset: $("preset").value || null,
    mzml: state.picks.mzml,
    diapasef,
  });
}

// Start and Build library only are enabled when the engine was found, no Start is in
// progress, and the built-in path is not blocked on the native models.
function refreshStartButtons() {
  const off = !state.engineOk || state.starting || state.builtinBlocked;
  $("start").disabled = off;
  $("build-library").disabled = off;
}

// Show the plan's block (red, Start disabled) or its warning (amber, asked at Start).
function showBuiltinPlan(plan) {
  const warn = $("builtin-warn");
  const blocked = Boolean(plan && plan.blocked);
  warn.classList.toggle("bad", blocked);
  warn.classList.toggle("warn", !blocked);
  banner(warn, (plan && (plan.blocked || plan.warning)) || "");
  state.builtinBlocked = blocked;
  refreshStartButtons();
}

async function refreshBuiltinPlan() {
  const note = $("libsrc-builtin-note");
  const warn = $("builtin-warn");
  const seq = ++state.builtinSeq;
  if (state.mode !== "fasta" || state.libSrc !== "builtin") {
    note.textContent = "";
    showBuiltinPlan(null);
    return;
  }
  note.textContent = "Checking which predictors are installed…";
  let plan;
  try {
    // Without spectra, the plan is for the instrument chosen under "Library for".
    plan = await builtinPlan(
      state.picks.mzml.length ? null : $("lib-for").value === "diapasef"
    );
  } catch (e) {
    if (seq !== state.builtinSeq) return;
    note.textContent = `Could not check the predictors: ${e}`;
    banner(warn, "");
    state.builtinBlocked = false;
    refreshStartButtons();
    return;
  }
  if (seq !== state.builtinSeq) return;
  note.textContent = plan.note;
  showBuiltinPlan(plan);
}

// Ask before a search on diaPASEF spectra without predicted 1/K0. True to go on. A search
// on the native models is not asked about: it is blocked (`plan.blocked`).
async function confirmPredictors(warning) {
  const message = `${warning}\n\nStart the search anyway?`;
  try {
    return await dialog.ask(message, {
      title: "This search uses less of the diaPASEF data",
      kind: "warning",
      okLabel: "Start anyway",
      cancelLabel: "Go back",
    });
  } catch {
    // A dialog plugin that cannot ask must not turn the warning into a silent start.
    return window.confirm(message);
  }
}

/// Build the library if needed and return its two tables, or null on failure.
// Put the progress screen back the way a search expects to find it.
function leaveLibraryPhase() {
  state.buildingLibrary = false;
  $("prog-title").textContent = "Searching";
}

async function ensureLibrary(fasta) {
  const req = libraryParams(fasta);
  let plan;
  try {
    plan = await invoke("diann_library_plan", { req });
  } catch (e) {
    banner($("start-error"), String(e));
    return null;
  }
  if (plan.cached) {
    return [plan.cached.precursors, plan.cached.fragments];
  }

  // Not cached: build into the cache directory the backend chose.
  req.out_dir = plan.cache_dir;
  // The progress screen's own log and subtitle, not the Setup screen's DIA-NN panes:
  // those live inside #screen-setup and would not be visible from here.
  enterLibraryPhase("Predicting the library", "DIA-NN is predicting the spectral library");
  try {
    await invoke("diann_build", { req });
  } catch (e) {
    banner($("start-error"), String(e));
    leaveLibraryPhase();
    screen("input");
    return null;
  }
  const built = await followLibraryBuild($("start-error"));
  if (built) {
    // Hand the screen back to the search, which is about to start on it.
    $("prog-sub").textContent = "Starting";
    $("log").textContent = "";
  }
  return built;
}

// Show the progress screen for a library build or import.
function enterLibraryPhase(title, sub) {
  screen("progress");
  state.buildingLibrary = true;
  // Reachable from the menu for the whole build: it used to be enabled only once the
  // search started, so leaving this screen during a prediction left no way back.
  $("nav-progress").disabled = false;
  $("nav-results").disabled = true;
  $("stop").disabled = false;
  $("stop").textContent = "Stop";
  banner($("run-error"), "");
  $("stages").innerHTML = "";
  $("prog-title").textContent = title;
  $("prog-sub").textContent = sub;
  $("log").textContent = "";
  $("cmd").textContent = "";
}

// Follow a library build, or a DIA-NN library import (which shares its state), to the
// end: its two tables, or null after showing its error in `errorEl` on the input screen.
async function followLibraryBuild(errorEl) {
  // Poll to completion, mirroring the build log onto the progress screen.
  for (;;) {
    await new Promise((r) => setTimeout(r, 1000));
    let b;
    try {
      b = await invoke("diann_build_state");
    } catch {
      continue;
    }
    const logEl = $("log");
    const text = (b.log || []).join("\n");
    if (logEl && logEl.textContent !== text) {
      logEl.textContent = text;
      logEl.scrollTop = logEl.scrollHeight;
    }
    // Every command the build has run, the current one last: DIA-NN's prediction,
    // its re-export, and the conversion and decoy steps after it.
    const cmds = (b.commands || []).join("\n\n");
    if ($("cmd").textContent !== cmds) $("cmd").textContent = cmds;
    $("prog-sub").textContent = b.step || "working";
    if (b.status === "failed") {
      banner(errorEl, b.error || "The library build failed.");
      leaveLibraryPhase();
      screen("input");
      return null;
    }
    if (b.status === "done" && b.precursors) {
      leaveLibraryPhase();
      return [b.precursors, b.fragments];
    }
  }
}

// -- a DIA-NN library as the search's library ------------------------------
// A DIA-NN library (.parquet or .speclib) is converted into the two tables by the
// builder's own steps after its prediction (`diann::import`): re-export, import,
// reversed decoys, DIA-NN's prediction of the decoys. On success the two tables are
// selected under "MuMDIA library tables", ready for the search.
function selectLibIn(which) {
  state.libIn = which;
  const r = $(which === "diann" ? "libin-diann" : "libin-tables");
  if (r) r.checked = true;
  show($("libin-tables-pane"), which === "tables");
  show($("libin-diann-pane"), which === "diann");
  if (which === "diann") refreshImportNote();
}

// Say, before Convert is pressed, whether the DIA-NN the conversion needs is there.
async function refreshImportNote() {
  const note = $("diann-import-note");
  let d;
  try {
    d = (await invoke("diann_status")).status;
  } catch (e) {
    note.textContent = String(e);
    return;
  }
  if (!d.runs) {
    note.textContent =
      "No working DIA-NN was found: install DIA-NN 2.x and locate it on the Setup screen.";
  } else if (!d.licence_acknowledged) {
    note.textContent = "Acknowledge the DIA-NN licence on the Setup screen first.";
  } else if (!d.writes_parquet) {
    note.textContent = `${d.version || "This DIA-NN"} cannot write Parquet; DIA-NN 2.x is needed.`;
  } else {
    note.textContent = `Uses ${d.version || "DIA-NN"}.`;
  }
}

async function importDiannLibrary() {
  const err = $("diann-import-error");
  banner(err, "");
  if (state.buildingLibrary) {
    banner(err, "A library build is already running.");
    return;
  }
  const library = state.picks.diann_lib;
  const outDir = state.picks.diann_lib_out;
  if (!library) return banner(err, "Choose the DIA-NN library (.parquet or .speclib).");
  if (!outDir) return banner(err, "Choose an empty folder for the converted library.");
  const threads = parseInt($("d-threads").value, 10);
  const req = { library, out_dir: outDir };
  if (Number.isFinite(threads) && threads > 0) req.threads = threads;
  try {
    // Refuses at once, with the reason, for a file that is not a DIA-NN library and
    // without a DIA-NN 2.x whose licence is acknowledged.
    await invoke("diann_import", { req });
  } catch (e) {
    banner(err, String(e));
    return;
  }
  enterLibraryPhase("Converting the library", "Converting the DIA-NN library");
  const built = await followLibraryBuild(err);
  screen("input");
  if (!built) return;
  for (const [what, path] of [
    ["lib_precursors", built[0]],
    ["lib_fragments", built[1]],
  ]) {
    state.picks[what] = path;
    const el = $(LABEL[what]);
    el.textContent = path;
    el.title = path;
    el.classList.add("set");
  }
  selectLibIn("tables");
  scheduleSpace();
  banner($("start-error"), "");
}

// One search per file, in sequence.
//
// Sequential and not parallel on purpose: a search already saturates the machine, and
// two at once would compete for the same cores and memory while making the progress
// display meaningless. The engine's own `experiment.parallel_runs` exists for the
// pooled case where it can be reasoned about.
async function runBatch(p, built, threads, config) {
  const files = p.mzml.slice();
  state.batch = { total: files.length, done: 0, failed: 0, results: [] };

  for (const [i, file] of files.entries()) {
    if (state.batch.stopped) break;
    state.batch.done = i;

    // Its own folder, named after the file, so results are findable and two runs
    // cannot overwrite each other. A stem collision between two directories is
    // possible, so the index disambiguates.
    const stem = baseName(file).replace(/\.[^.]+$/, "");
    const sub = `${String(i + 1).padStart(2, "0")}_${stem}`;
    const req = {
      mzml: [file],
      experiment: false,
      out_dir: `${p.out_dir}/${sub}`,
      fasta: state.mode === "fasta" && !built ? p.fasta || null : null,
      lib_precursors: built ? built[0] : state.mode === "library" ? p.lib_precursors || null : null,
      lib_fragments: built ? built[1] : state.mode === "library" ? p.lib_fragments || null : null,
      config,
      threads: Number.isFinite(threads) && threads > 0 ? threads : null,
    };

    // Preflight each file. A converter missing for file 5 should not be discovered
    // after files 1 to 4 have been searched.
    try {
      const pf = await invoke("preflight", { req });
      if (!pf.ok) {
        state.batch.failed += 1;
        state.batch.results.push({ file, error: pf.blockers.join(" ") });
        continue;
      }
    } catch {
      /* Preflight that cannot run is not a reason to refuse; the engine reports. */
    }

    try {
      state.runId = await invoke("start_run", { req });
    } catch (e) {
      state.batch.failed += 1;
      state.batch.results.push({ file, error: String(e) });
      continue;
    }

    state.outDir = req.out_dir;
    rememberFolder(req.out_dir);
    screen("progress");
    $("prog-title").textContent = `Searching ${i + 1} of ${files.length}`;
    $("prog-sub").textContent = baseName(file);
    for (const b of document.querySelectorAll(".nav")) {
      if (b.dataset.screen === "progress") b.disabled = false;
    }

    // Wait for this one before starting the next.
    const outcome = await awaitRun();
    state.batch.results.push({ file, out_dir: req.out_dir, ...outcome });
    if (outcome.error) state.batch.failed += 1;
  }

  state.batch.done = files.length;
  renderBatchSummary();
}

/// Poll the current run to a terminal state, resolving with what it produced.
function awaitRun() {
  return new Promise((resolve) => {
    const tick = async () => {
      let s;
      try {
        s = await invoke("run_state", { id: state.runId });
      } catch (e) {
        return resolve({ error: String(e) });
      }
      render(s);
      if (s.status === "done") return resolve({ results: s.results });
      if (s.status === "failed" || s.status === "cancelled") {
        return resolve({ error: s.error || s.status });
      }
      setTimeout(tick, 1000);
    };
    tick();
  });
}

// What the batch produced, per file, once every run has finished.
function renderBatchSummary() {
  const b = state.batch;
  if (!b) return;
  screen("results");
  $("res-title").textContent = `${b.total} searches`;
  const ok = b.total - b.failed;
  $("res-sub").textContent =
    b.failed === 0
      ? `All ${b.total} finished.`
      : `${ok} finished, ${b.failed} did not. Each file has its own folder.`;

  // Reuse the results screen's file list area for a per-file breakdown, because a
  // single set of counts would be a lie: these runs share no FDR.
  show($("res-batch-card"), true);
  show($("res-library-card"), false);
  show($("open-visdia"), true);
  show($("kpis"), false);
  const host = $("res-files");
  if (!host) return;
  host.innerHTML = "";
  for (const r of b.results) {
    const row = document.createElement("div");
    const n = r.results ? fmtInt(r.results.peptides_1pct) : null;
    const uncalibrated = (r.results?.rt_calibration_missing || []).length > 0;
    row.textContent = r.error
      ? `${baseName(r.file)} — failed: ${r.error}`
      : `${baseName(r.file)} — ${n ?? "?"} peptides at 1%` +
        (uncalibrated ? " — DEGRADED: no retention-time calibration" : "");
    // Each file of a separate-files batch is its own result, in its own folder.
    if (!r.error && r.out_dir) {
      const open = node("button", "btn quiet small", "Open in visDIA");
      open.dataset.visdia = r.out_dir;
      open.addEventListener("click", () =>
        openVisdia(r.out_dir, open, $("visdia-error"), state.picks.fasta)
      );
      row.append(" ", open);
    }
    host.appendChild(row);
  }
  refreshVisdiaButtons();
}

// ── a library without a search (issue #233) ─────────────────────────────────
// "Build library only" predicts the library a FASTA search would, and stops there: no
// spectra are needed. DIA-NN builds into its own cache, exactly as before a search. The
// built-in predictors run the engine's `build-library` through the ordinary run machinery
// (one engine process, its log, Stop, the results-folder reservation), which writes the
// library into the results folder and stores it in the engine's library cache, where a
// later FASTA search with the same settings finds it.
async function buildLibraryOnly() {
  if (state.starting) return;
  state.starting = true;
  $("start").disabled = true;
  $("build-library").disabled = true;
  try {
    await startLibraryBuild();
  } finally {
    state.starting = false;
    refreshStartButtons();
  }
}

async function startLibraryBuild() {
  banner($("start-error"), "");
  banner($("preflight-block"), "");
  const p = state.picks;
  if (state.mode !== "fasta") {
    banner($("start-error"), "A library is built from a FASTA: choose the From a FASTA tab.");
    return;
  }
  if (!p.fasta) {
    banner($("start-error"), "Choose a FASTA first.");
    return;
  }
  const active = await activeRun();
  if (active) {
    banner(
      $("start-error"),
      `A search is still running in ${active.out_dir}. Stop it, or wait for it to finish, first.`
    );
    return;
  }
  const problems = modProblems();
  if (problems.length) {
    banner($("start-error"), problems.join("\n\n"));
    return;
  }
  const diapasef = $("lib-for").value === "diapasef";

  if (state.libSrc === "diann") {
    const built = await ensureLibrary(p.fasta);
    if (!built) return;
    let plan = null;
    try {
      plan = await invoke("diann_library_plan", { req: libraryParams(p.fasta) });
    } catch {
      /* Only the version line is lost. */
    }
    let count = null;
    try {
      count = (await invoke("search_space", { req: spaceRequest(built[0]) })).precursors;
    } catch {
      /* The counts are a convenience. */
    }
    $("nav-results").disabled = false;
    renderLibraryResult({
      source: "diann",
      precursors: built[0],
      fragments: built[1],
      diannVersion: plan?.diann_version || "",
      precursorCount: count,
      info: null,
    });
    screen("results");
    return;
  }

  if (!p.out_dir) {
    banner($("start-error"), "Choose a results folder: the library is written there.");
    return;
  }
  let plan;
  try {
    plan = await builtinPlan(diapasef);
  } catch (e) {
    banner($("start-error"), `Could not check the predictors: ${e}`);
    return;
  }
  $("libsrc-builtin-note").textContent = plan.note;
  showBuiltinPlan(plan);
  if (plan.blocked) {
    banner($("start-error"), plan.blocked);
    return;
  }
  if (plan.warning && !(await confirmPredictors(plan.warning))) {
    banner($("start-error"), plan.warning);
    return;
  }
  // The digest and modification choices, as a search derives them, so the library is the
  // one a search with these settings looks up. Prescreen and bands do not change it.
  let config = plan.base;
  try {
    config = await invoke("derive_config", { base: config, overrides: digestOverrides() });
  } catch (e) {
    banner($("start-error"), String(e));
    return;
  }
  const threads = parseInt($("threads").value, 10);
  const req = {
    mzml: [],
    experiment: false,
    out_dir: p.out_dir,
    fasta: p.fasta,
    lib_precursors: null,
    lib_fragments: null,
    config,
    threads: Number.isFinite(threads) && threads > 0 ? threads : null,
    build_library: true,
    ion_mobility: diapasef,
  };
  try {
    const pf = await invoke("preflight", { req });
    if (!pf.ok) {
      banner(
        $("preflight-block"),
        pf.blockers.join("\n\n") +
          (pf.components_complete ? "" : "\n\nOpen Setup to install the components.")
      );
      return;
    }
  } catch (e) {
    banner($("preflight-block"), `Could not check before starting: ${e}`);
  }
  try {
    state.runId = await invoke("start_run", { req });
  } catch (e) {
    banner($("start-error"), String(e));
    return;
  }
  state.outDir = p.out_dir;
  state.lastStatus = null;
  state.batch = null;
  $("nav-progress").disabled = false;
  $("nav-results").disabled = true;
  $("stop").disabled = false;
  $("log").textContent = "";
  banner($("run-error"), "");
  screen("progress");
  clearInterval(state.timer);
  state.timer = setInterval(poll, 700);
  poll();
}

// The folder a file is in.
function dirOf(path) {
  return (path || "").replace(/[\\/][^\\/]*$/, "");
}

// The library of a finished `build-library` run, from its library.json.
function libraryOfBuild(s) {
  const info = s.library || {};
  const precursors = info.precursors?.path || `${s.out_dir}/fragment_library_precursors.parquet`;
  return {
    source: "builtin",
    precursors,
    fragments: info.fragments?.path || `${s.out_dir}/fragment_library_fragments.parquet`,
    info: s.library || null,
    elapsedMs: s.elapsed_ms,
  };
}

// The result card of a library build: what was built, where, and how to search with it.
function renderLibraryResult(L) {
  L.dir = dirOf(L.precursors);
  state.libraryResult = L;
  state.outDir = L.dir;
  show($("res-batch-card"), false);
  show($("kpis"), false);
  show($("open-visdia"), false);
  show($("res-library-card"), true);
  banner($("res-warn"), "");
  banner($("res-degraded"), "");
  banner($("lib-error"), "");
  banner($("lib-copied"), "");
  $("res-title").textContent = "Library built";

  const info = L.info;
  const lines = [];
  const sub = [];
  if (L.source === "diann") {
    sub.push(`predicted by DIA-NN${L.diannVersion ? ` ${L.diannVersion}` : ""}`);
    if (L.precursorCount != null) lines.push(`${fmtInt(L.precursorCount)} precursors`);
  } else if (info) {
    sub.push(info.origin === "library-cache" ? "restored from the library cache" : "built");
    if (L.elapsedMs) sub.push(fmtDuration(L.elapsedMs));
    const pr = info.precursors || {};
    lines.push(
      `${fmtInt(pr.rows)} precursors (${fmtInt(pr.targets)} targets, ${fmtInt(pr.decoys)} ` +
        `decoys), ${fmtInt(info.fragments?.rows)} fragments`
    );
    const m = info.models || {};
    const models = [`fragments: ${m.fragment ?? "?"}`, `retention time: ${m.rt ?? "?"}`];
    models.push(`ion mobility: ${m.im ?? "none"}`);
    lines.push(`Predicted with ${models.join(", ")}.`);
    lines.push(
      `For ${info.ion_mobility ? "timsTOF diaPASEF spectra" : "spectra without ion mobility"}` +
        (info.predicts_ion_mobility ? ", with predicted 1/K0." : ".")
    );
    if (info.cache?.key) lines.push(`Library cache key ${info.cache.key}.`);
  }
  $("res-sub").textContent = sub.join(" · ");
  $("lib-summary").innerHTML = "";
  for (const l of lines) $("lib-summary").appendChild(node("div", "", l));

  let reuse;
  if (L.source === "diann") {
    reuse =
      "Kept in the DIA-NN library cache for this FASTA and these digest settings: a search " +
      "from the FASTA with DIA-NN uses it without predicting again.";
  } else {
    const irt = info?.reuse?.library_irt;
    reuse =
      (info?.cache?.stored
        ? "A search from the same FASTA with the same settings and installed predictors finds " +
          "this library in the library cache and skips the prediction. "
        : "The library cache is off, so a search from the FASTA predicts the library again. ") +
      "Search with this library opens the spectral library tab with these tables" +
      (irt
        ? `, and the search then sets rt_im_train.library_irt = "${irt}", which handles ` +
          "the retention times as a search from the FASTA would."
        : ".");
  }
  $("lib-reuse").textContent = reuse;

  const files = [
    [L.precursors, "precursor table"],
    [L.fragments, "fragment table"],
  ];
  if (L.source === "builtin") {
    files.push([`${L.dir}/library.json`, "counts, models, cache key and how to search with it"]);
    files.push([`${L.dir}/manifest.json`, "engine version, commit, and a hash of the FASTA"]);
  }
  $("files").innerHTML = "";
  for (const [f, d] of files) {
    const row = node("div");
    row.append(node("code", "", f), " ", node("span", "hint inline", d));
    $("files").appendChild(row);
  }
}

// Put the library in the spectral library tab, ready to search.
function searchWithLibrary() {
  const L = state.libraryResult;
  if (!L) return;
  for (const [what, path] of [
    ["lib_precursors", L.precursors],
    ["lib_fragments", L.fragments],
  ]) {
    state.picks[what] = path;
    const el = $(LABEL[what]);
    el.textContent = path;
    el.title = path;
    el.classList.add("set");
  }
  setMode("library");
  banner($("start-error"), "");
  screen("input");
}

// The `rt_im_train.library_irt` the library.json beside a precursor table names, or null
// for a table no `build-library` wrote.
async function libraryIrtOf(precursors) {
  if (!precursors) return null;
  try {
    const info = await invoke("library_info", { precursors });
    return info?.reuse?.library_irt || null;
  } catch {
    return null;
  }
}

async function copyLibrary() {
  const L = state.libraryResult;
  if (!L) return;
  banner($("lib-error"), "");
  banner($("lib-copied"), "");
  const dest = await dialog.open({ directory: true, multiple: false });
  if (!dest) return;
  const btn = $("lib-copy");
  btn.disabled = true;
  btn.textContent = "Copying…";
  try {
    const [precursors] = await invoke("copy_library", {
      precursors: L.precursors,
      fragments: L.fragments,
      dest,
    });
    banner($("lib-copied"), `Copied to ${dirOf(precursors)}.`);
  } catch (e) {
    banner($("lib-error"), String(e));
  } finally {
    btn.disabled = false;
    btn.textContent = "Copy to folder…";
  }
}

// Start is asynchronous with several awaits before the run exists (library build,
// derived configuration, preflight), and a second click during any of them launched a
// second engine into the same results folder while the interface kept only the latest
// run id (docs/29 #5). This guard sits before the first await and is the first line of
// defence; the backend's reservation of active results folders is the second.
async function start() {
  if (state.starting) return;
  state.starting = true;
  $("start").disabled = true;
  try {
    await startSearch();
  } finally {
    state.starting = false;
    refreshStartButtons();
  }
}

// The run the interface follows, when it has not reached a terminal state.
async function activeRun() {
  if (!state.runId) return null;
  try {
    const s = await invoke("run_state", { id: state.runId });
    return s && (s.status === "running" || s.status === "starting") ? s : null;
  } catch {
    return null;
  }
}

async function startSearch() {
  banner($("start-error"), "");
  const p = state.picks;
  const threads = parseInt($("threads").value, 10);

  if (p.mzml.length === 0) {
    banner($("start-error"), "Add at least one spectra file.");
    return;
  }
  // A DIA-NN library is searched through the two tables its conversion writes.
  if (state.mode === "library" && state.libIn === "diann" && !p.lib_precursors) {
    banner(
      $("start-error"),
      "Convert the DIA-NN library first: the search reads the two tables it writes."
    );
    return;
  }

  // The interface follows one run. Starting another while it is in progress would
  // leave the first running with nothing showing it and nothing able to stop it.
  const active = await activeRun();
  if (active) {
    banner(
      $("start-error"),
      `A search is still running in ${active.out_dir}. Stop it, or wait for it to finish, before starting another.`
    );
    return;
  }

  const problems = searchProblems();
  if (problems.length) {
    banner($("start-error"), problems.join("\n\n"));
    return;
  }

  // FASTA mode with DIA-NN: the search is a library-mode search whose library is
  // produced first. Everything after this point is the ordinary library path, which
  // is also the tested one. Built ONCE for the whole selection, not per file: the
  // library depends on the FASTA and the digest parameters, not on the spectra.
  let built = null;
  if (state.mode === "fasta" && state.libSrc === "diann") {
    if (!p.fasta) {
      banner($("start-error"), "Choose a FASTA first.");
      return;
    }
    built = await ensureLibrary(p.fasta);
    if (!built) return;
  }

  // The Search tab's choices become part of the run's configuration, on top of the
  // chosen preset: the digest and modification fields for the built-in predictors, and
  // the prescreen and band plan in every mode. Derived once for the whole selection and
  // validated by the engine before anything starts. With nothing chosen the preset is
  // passed unchanged, as before.
  let config = $("preset").value || null;

  // The built-in path: the predictors the library is built with, chosen and checked now
  // (`builtin_plan`). A search that would still build its library on the native models
  // does not start (the engine refuses it); one on a diaPASEF .d without predicted 1/K0
  // starts only when the user says so.
  if (state.mode === "fasta" && !built) {
    let plan = null;
    try {
      plan = await builtinPlan();
    } catch (e) {
      banner($("start-error"), `Could not check the predictors: ${e}`);
      screen("input");
      return;
    }
    config = plan.base;
    $("libsrc-builtin-note").textContent = plan.note;
    showBuiltinPlan(plan);
    if (plan.blocked) {
      banner($("start-error"), plan.blocked);
      screen("input");
      return;
    }
    if (plan.warning && !(await confirmPredictors(plan.warning))) {
      banner($("start-error"), plan.warning);
      screen("input");
      return;
    }
  }

  const bands = await resolveBands(built ? built[0] : null);
  if (ztscanAnalysisOn() && bands.bands > 1) {
    banner(
      $("start-error"),
      `This search would be split into ${bands.bands} bands to fit the memory of this ` +
        "machine, and the ZT Scan analysis calibrates the Q1 sweep of the whole run, which " +
        "cannot be split. Clear the ZT Scan analysis, or set Bands to Off if the machine has " +
        "the memory."
    );
    screen("input");
    return;
  }
  if ($("prescreen").value === "after" && bands.bands > 1) {
    banner(
      $("start-error"),
      `This search would be split into ${bands.bands} bands to fit the memory of this ` +
        "machine, and the prescreen after retention-time calibration cannot be combined " +
        "with bands yet. Choose the prescreen before prediction, or set Bands to Off if " +
        "the machine has the memory."
    );
    screen("input");
    return;
  }
  // A library this application built from a FASTA (`build-library`) is searched with the
  // `rt_im_train.library_irt` its library.json names, so its retention times are handled
  // as the FASTA search would have handled them.
  const libraryIrt =
    state.mode === "library" ? await libraryIrtOf(p.lib_precursors) : null;
  const overrides = runOverrides({ bands, built, libraryIrt });
  if (Object.keys(overrides).length) {
    try {
      config = await invoke("derive_config", { base: config, overrides });
    } catch (e) {
      banner($("start-error"), String(e));
      screen("input");
      return;
    }
  }

  // Several files searched separately are N independent runs. Queued in the frontend
  // so `start_run` stays the single tested path and each run is an ordinary one; the
  // cost is that closing the window ends the queue, which the per-file output folders
  // make recoverable.
  if (p.mzml.length > 1 && currentRunMode() === "separate") {
    return runBatch(p, built, threads, config);
  }

  const req = {
    // A sequence: `run::Request.mzml` is `Vec<String>`. Sending the bare string made
    // serde reject every payload, so preflight AND start_run both failed and the Start
    // button did nothing in every mode. `ci/check_desktop_ui.py` cannot catch this --
    // it checks element ids and command names, never payload types.
    mzml: p.mzml,
    experiment: currentRunMode() === "experiment",
    // One per file, recorded in the manifest and read by the normalisation within each
    // condition; empty when no file has one.
    conditions:
      currentRunMode() === "experiment" && runConditions().some(Boolean) ? runConditions() : [],
    out_dir: p.out_dir,
    fasta: state.mode === "fasta" && !built ? p.fasta || null : null,
    lib_precursors: built ? built[0] : state.mode === "library" ? p.lib_precursors || null : null,
    lib_fragments: built ? built[1] : state.mode === "library" ? p.lib_fragments || null : null,
    config,
    threads: Number.isFinite(threads) && threads > 0 ? threads : null,
  };

  // Ask the backend whether this is runnable before starting it, so a missing
  // component or a configuration that needs no components at all is explained here
  // rather than failing once the engine is under way.
  banner($("preflight-block"), "");
  try {
    const pf = await invoke("preflight", { req });
    if (!pf.ok) {
      banner(
        $("preflight-block"),
        pf.blockers.join("\n\n") +
          (pf.components_complete ? "" : "\n\nOpen Setup to install the components.")
      );
      // The library build may already have moved us to the progress screen, and both
      // banners live on the input screen. Without this the message rendered onto a
      // hidden screen: the user sat on "Searching" with an empty log, Progress still
      // disabled, and `state.runId` null so Stop and the poller both did nothing.
      screen("input");
      return;
    }
    // Not blocking, but worth saying before an hour is spent on it.
    banner($("preflight-note"), (pf.warnings || []).join("\n\n"));
  } catch (e) {
    // A preflight that cannot run is not a reason to refuse: say so and let the
    // engine be the judge, since it reports its own errors perfectly well.
    banner($("preflight-block"), `Could not check before starting: ${e}`);
  }

  try {
    state.runId = await invoke("start_run", { req });
  } catch (e) {
    banner($("start-error"), String(e));
    screen("input");
    return;
  }

  state.outDir = p.out_dir;
  rememberFolder(p.out_dir);
  state.lastStatus = null;
  $("nav-progress").disabled = false;
  $("nav-results").disabled = true;
  $("stop").disabled = false;
  $("log").textContent = "";
  banner($("run-error"), "");
  screen("progress");

  clearInterval(state.timer);
  state.timer = setInterval(poll, 700);
  poll();
}

async function stop() {
  // During the library phase there is no run to cancel: `cancel_run` only knows about
  // engine runs, so Stop did nothing at all through a whole-proteome prediction.
  // A batch: stop the queue as well as the current run, or the next file starts the
  // moment this one is cancelled.
  if (state.batch && state.batch.done < state.batch.total) {
    state.batch.stopped = true;
  }
  if (state.buildingLibrary) {
    try {
      await invoke("diann_cancel");
    } catch (e) {
      banner($("run-error"), String(e));
    }
    return;
  }

  if (!state.runId) return;
  $("stop").disabled = true;
  $("stop").textContent = "Stopping…";
  try {
    await invoke("cancel_run", { id: state.runId });
  } catch (e) {
    banner($("run-error"), String(e));
  }
}

async function poll() {
  if (!state.runId) return;
  let s;
  try {
    s = await invoke("run_state", { id: state.runId });
  } catch {
    return;
  }
  if (!s) return;
  render(s);

  if (s.status !== "running" && s.status !== "starting") {
    clearInterval(state.timer);
    state.timer = null;
    $("stop").disabled = true;
    $("stop").textContent = "Stop";
    if (s.status === "done") {
      $("nav-results").disabled = false;
      if (s.build_library) {
        renderLibraryResult(libraryOfBuild(s));
      } else {
        renderResults(s);
      }
      screen("results");
    }
  }
}

// ── visDIA ──────────────────────────────────────────────────────────────────
// The results viewer runs as its own local server (`viewer.rs`). Opening a folder
// starts one, or shows the one already serving that folder; the browser opens on it.

/// The hint every "Open in visDIA" button carries, which says what a click will do.
function refreshVisdiaButtons() {
  const title = state.visdiaReady
    ? "Show these results in visDIA, in your browser"
    : "visDIA is not installed yet: install it on the Setup screen";
  for (const b of document.querySelectorAll("#open-visdia, [data-visdia]")) b.title = title;
}

async function openVisdia(dir, btn, errEl, fasta) {
  if (!dir) return;
  banner(errEl, "");
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Opening visDIA…";
  try {
    await invoke("visdia_open", { dir, fasta: fasta || null });
  } catch (e) {
    banner(errEl, String(e));
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

// ── history ─────────────────────────────────────────────────────────────────
// Every remembered folder comes back from `history` with a status (#282): completed
// searches first, newest first, then the folders that are not one, greyed, with the
// reason and a Remove button that forgets the folder (it is never deleted).
const HISTORY_STATUS = {
  running: ["Running", "warn"],
  incomplete: ["Did not finish", "bad"],
  not_a_search: ["Not a completed search", "mute"],
  not_found: ["Folder not found", "mute"],
};

function historyNote() {
  const n = state.knownDropped;
  banner(
    $("history-note"),
    n
      ? `${fmtInt(n)} older folder${n === 1 ? " was" : "s were"} dropped from this ` +
          `list, which keeps the ${fmtInt(KNOWN_FOLDERS_MAX)} most recent. Add one back ` +
          `with "Add an existing output folder…".`
      : ""
  );
}

function historyCard(i) {
  const esc = escHtml;
  const e = i.entry;
  if (i.status === "completed" && e) {
    const r = e.results;
    const when = e.finished_unix_ms ? new Date(e.finished_unix_ms).toLocaleString() : "";
    const counts = r
      ? `${fmtInt(r.peptides_1pct)} peptides at peptide_q_value 0.01 · ` +
        `${fmtInt(r.protein_groups_1pct)} protein groups · ${esc(r.classifier)}` +
        (e.experiment ? ` · experiment of ${fmtInt(r.n_runs)} runs` : "")
      : "no scored table in this folder";
    return (
      `<div class="card"><div class="card-head"><div>` +
      `<div class="card-title">${esc(e.name)}</div>` +
      `<div class="card-sub">${counts}<br>${esc(when)} · ${esc(e.out_dir)}` +
      (e.engine_version ? ` · ${esc(e.engine_version)}` : "") +
      `</div></div>` +
      `<div class="row">` +
      (r ? `<button class="btn quiet" data-visdia="${esc(e.out_dir)}">Open in visDIA</button>` : "") +
      `<button class="btn quiet" data-open="${esc(e.out_dir)}">Open folder</button>` +
      `</div></div></div>`
    );
  }
  const [label, tone] = HISTORY_STATUS[i.status] || [i.status, "mute"];
  return (
    `<div class="card dim"><div class="card-head"><div>` +
    `<div class="card-title">${esc(i.name)} <span class="pill ${tone}">${esc(label)}</span></div>` +
    `<div class="card-sub">${esc(i.reason || "")}<br>${esc(i.out_dir)}</div></div>` +
    `<div class="row">` +
    (i.status !== "not_found"
      ? `<button class="btn quiet" data-open="${esc(i.out_dir)}">Open folder</button>`
      : "") +
    (i.status !== "running"
      ? `<button class="btn quiet" data-forget="${esc(i.out_dir)}">Remove</button>`
      : "") +
    `</div></div></div>`
  );
}

async function loadHistory() {
  const list = $("history-list");
  historyNote();
  let items = [];
  try {
    items = await invoke("history", { dirs: state.known });
  } catch (e) {
    list.innerHTML = `<p class="hint">Could not read past searches: ${escHtml(e)}</p>`;
    return;
  }
  if (!items.length) {
    list.innerHTML = `<p class="hint">No past searches yet.</p>`;
    return;
  }
  list.innerHTML = items.map(historyCard).join("");
  for (const b of list.querySelectorAll("[data-open]")) {
    b.addEventListener("click", () =>
      invoke("reveal", { path: b.dataset.open }).catch((e) =>
        banner($("history-error"), String(e))
      )
    );
  }
  for (const b of list.querySelectorAll("[data-forget]")) {
    b.addEventListener("click", () => {
      forgetFolder(b.dataset.forget);
      loadHistory();
    });
  }
  for (const b of list.querySelectorAll("[data-visdia]")) {
    b.addEventListener("click", () => openVisdia(b.dataset.visdia, b, $("history-error")));
  }
  refreshVisdiaButtons();
}

// "Add an existing output folder…": a search run elsewhere, or one this list lost. Added
// only when the folder is a completed search; otherwise the reason is shown.
async function addHistoryFolder() {
  banner($("history-error"), "");
  const chosen = await dialog.open({ directory: true, multiple: false });
  if (!chosen) return;
  const dir = Array.isArray(chosen) ? chosen[0] : chosen;
  let item;
  try {
    item = await invoke("history_check", { dir });
  } catch (e) {
    banner($("history-error"), String(e));
    return;
  }
  if (item.status !== "completed" && item.status !== "running") {
    banner($("history-error"), `Not added: ${dir}. ${item.reason || item.status}`);
    return;
  }
  // Already listed: move it to the front.
  forgetFolder(dir);
  rememberFolder(dir);
  loadHistory();
}

// ── rendering ───────────────────────────────────────────────────────────────
function render(s) {
  // In a pooled experiment the engine runs the per-file chain -- convert through
  // compete -- once per input file, in that file's own subdirectory, and then does a
  // single rescore/quant/report at the experiment root. Aggregating by stage NAME
  // across the whole tree, which is all this had before, meant the ladder stopped
  // moving after the first file: once file 1 had reached the end, file 2's `convert`
  // could not pull "the furthest stage seen" backwards, so the remaining files ran
  // with the display frozen. The counts were a sum across files as well, so they
  // described no file in particular.
  //
  // So track the file that is actually running: the last one with any stage, in input
  // order (the backend orders the run folders, which are named after the files). Its
  // per-file stages plus the root's pooled ones are the ladder; the subtitle says
  // which file of how many.
  const active = (s.runs || []).length ? s.runs[s.runs.length - 1] : null;
  const perFile = active
    ? [...(active.stages || []), ...(s.root_stages || [])]
    : s.stages;
  const seen = new Map(perFile.map((x) => [x.name, x]));
  const expected = STAGES_FASTA.filter(([key]) =>
    s.build_library
      ? LIBRARY_BUILD_STAGES.has(key)
      : !(s.library_mode && SKIP_IN_LIBRARY_MODE.has(key))
  );

  // The furthest stage with a report is the one in progress; everything before it
  // is done. `report` writes TSVs rather than an artifact report, so it is only
  // ever complete once the process itself has finished successfully.
  let lastSeen = -1;
  expected.forEach(([key], i) => {
    if (seen.has(key)) lastSeen = i;
  });
  const finished = s.status === "done";

  const rows = expected.map(([key, label], i) => {
    const st = seen.get(key);
    let cls, mark;
    if (finished) {
      cls = "done";
      mark = "✓";
    } else if (st && i < lastSeen) {
      cls = "done";
      mark = "✓";
    } else if (st && i === lastSeen) {
      cls = s.status === "running" ? "now" : "done";
      mark = s.status === "running" ? "●" : "✓";
    } else if (i === lastSeen + 1 && s.status === "running") {
      cls = "now";
      mark = "●";
    } else {
      cls = "todo";
      mark = "·";
    }
    const stat = st && st.rows ? `${fmtInt(st.rows)} rows` : "";
    const time = st && st.elapsed_ms ? fmtDuration(st.elapsed_ms) : "";
    return `<div class="stage ${cls}">
      <span class="mark">${mark}</span>
      <span class="name">${label}</span>
      <span class="stat">${stat}</span>
      <span class="time">${time}</span>
    </div>`;
  });
  $("stages").innerHTML = rows.join("");

  const titles = s.build_library
    ? {
        starting: "Starting",
        running: "Building the library",
        done: "Library built",
        failed: "Library build failed",
        cancelled: "Library build stopped",
      }
    : {
        starting: "Starting",
        running: "Searching",
        done: "Finished",
        failed: "Search failed",
        cancelled: "Search stopped",
      };
  $("prog-title").textContent = titles[s.status] || s.status;
  // Stop was pressed but the engine has not been reaped yet: the status stays
  // `running` until the waiter publishes the outcome, and the title says so.
  if (s.cancel_requested && (s.status === "running" || s.status === "starting")) {
    $("prog-title").textContent = "Stopping";
  }

  const done = expected.filter(([k]) => seen.has(k)).length;
  const parts = [];
  // `s.runs` only has entries once a file has written something, so during the very
  // first file it reads "file 1 of n" rather than "0".
  if (active && (s.runs || []).length && s.status === "running") {
    const total = state.picks.mzml.length || s.runs.length;
    parts.push(`file ${Math.min(s.runs.length, total)} of ${total}`);
  }
  if (s.status === "running") parts.push(`stage ${Math.min(done + 1, expected.length)} of ${expected.length}`);
  if (s.elapsed_ms) parts.push(fmtDuration(s.elapsed_ms));
  if (s.status === "cancelled") parts.push("partial results were discarded");
  $("prog-sub").textContent = parts.join(" · ");

  $("cmd").textContent = s.command;

  if (s.error && (s.status === "failed" || s.status === "cancelled")) {
    banner($("run-error"), s.error);
  }

  // Only touch the log when it changed, so a user scrolled up to read something
  // is not yanked back to the bottom on every poll.
  const log = $("log");
  const text = s.log.join("\n");
  if (log.textContent !== text) {
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    log.textContent = text;
    if (atBottom) log.scrollTop = log.scrollHeight;
  }
}

// The runs the engine recorded without a retention-time calibration
// (`rt_calibration_missing` in its manifest), as the sentence the results screen leads
// with, or "" for a calibrated search. Such a run's first-pass search found too few
// confident anchors, so its candidates were extracted over the whole gradient (or a broad
// fixed window): the search finished, slowly, with far fewer identifications than the
// data hold, and nothing else on this screen would say so.
function degradedMessage(r) {
  const gaps = r?.rt_calibration_missing || [];
  if (!gaps.length) return "";
  const runs = (pred) => new Set(gaps.filter(pred).map((g) => g.run)).size;
  const first = runs((g) => g.pass !== "second_pass");
  const second = runs((g) => g.pass === "second_pass");
  const n = r.n_runs || Math.max(first, second);
  const parts = [];
  if (first) {
    parts.push(
      n > 1
        ? `${first} of ${n} runs had no retention-time calibration`
        : "This run had no retention-time calibration"
    );
  }
  if (second) parts.push(`${second} of ${n} runs had none in the second pass`);
  return (
    `SEARCH DEGRADED: ${parts.join("; ")}. The first-pass search found too few ` +
    "confident anchors, so candidates were extracted over the whole gradient: the " +
    "search was slow and the counts below are far lower than these data should give. " +
    "The usual cause is the native fragment predictor; others are a mass offset beyond " +
    "the search tolerance or a library that does not fit the sample. Search again with " +
    "MS2PIP or AlphaPeptDeep and DeepLC (with IM2Deep for diaPASEF), or with a predicted " +
    "or imported library. The log names the cause it found."
  );
}

function renderResults(s) {
  // The per-file breakdown belongs to a batch; a single run shows its own counts.
  show($("res-batch-card"), false);
  show($("res-library-card"), false);
  show($("open-visdia"), true);
  show($("kpis"), true);
  const r = s.results;
  $("res-title").textContent = "Finished";
  const bits = [fmtDuration(s.elapsed_ms)];
  if (r && r.classifier) bits.push(`rescoring: ${r.classifier}`);
  if (r && r.config_hash) bits.push(`settings ${r.config_hash.slice(0, 8)}`);
  $("res-sub").textContent = bits.filter(Boolean).join(" · ");

  // The requested classifier and the one that ran can differ, when a sidecar fails
  // and strict mode is off. Saying so is the whole point of reading the artifact
  // report rather than echoing the request.
  const requested = (r?.classifier_requested || "").toLowerCase().replace(/_/g, "");
  const actual = (r?.classifier || "").toLowerCase().replace(/_/g, "");
  if (r && requested && actual && requested !== actual) {
    banner(
      $("res-warn"),
      `Rescoring fell back to ${r.classifier}; ${r.classifier_requested} was requested. ` +
        `The counts below come from ${r.classifier}.`
    );
  } else if (r && r.experiment_wide) {
    // Not a caveat, a unit change. `run-experiment` groups the q columns across the
    // whole experiment, so these counts are experiment-wide and a per-file reading of
    // them is diluted by roughly 1/n_runs. Its peptides.tsv and proteins.tsv are the
    // experiment-wide report, with one quantity column per run.
    banner(
      $("res-warn"),
      "These are EXPERIMENT-WIDE counts, pooled across every run, not per file. " +
        "An experiment-wide rescore groups the q values across the whole experiment, " +
        "so dividing by the number of runs does not give a per-file number; the " +
        "per-file unit is run_psm_q in the split tables. peptides.tsv, precursors.tsv and " +
        "proteins.tsv here are the experiment-wide report, with one quantity column per run."
    );
  } else {
    banner($("res-warn"), "");
  }

  banner($("res-degraded"), degradedMessage(r));

  // Every count names its row unit and its q-value column. A number without both
  // is not interpretable, and this is where a screenshot gets taken.
  const kpis = r
    ? [
        [fmtInt(r.peptides_1pct), "peptides", "peptide_q_value ≤ 0.01"],
        [fmtInt(r.precursors_1pct), "precursors", "precursor_q ≤ 0.01"],
        [fmtInt(r.protein_groups_1pct), "protein groups", "pg_q_value ≤ 0.01"],
        [fmtInt(r.psms), "PSMs scored", "all, before thresholding"],
      ]
    : [];
  $("kpis").innerHTML = kpis
    .map(([v, k, u]) => `<div class="kpi"><div class="v">${v}</div><div class="k">${k}</div><div class="u">${u}</div></div>`)
    .join("");

  const files = [];
  if (r?.has_peptides_tsv) files.push(["peptides.tsv", "one row per (peptidoform, charge), selected by peptide q"]);
  if (r?.has_precursors_tsv) files.push(["precursors.tsv", "every accepted (peptidoform, charge), selected by precursor q"]);
  if (r?.has_proteins_tsv) files.push(["proteins.tsv", "protein groups"]);
  files.push(["psms_scored.parquet", "every scored match, with all features"]);
  files.push(["manifest.json", "engine version, commit, and a hash of every input"]);
  $("files").innerHTML = files
    .map(([f, d]) => `<div><code>${f}</code> &nbsp;<span class="hint inline">${d}</span></div>`)
    .join("");
}

init();
