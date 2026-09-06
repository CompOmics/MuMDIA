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

const state = {
  mode: "fasta",
  picks: {
    // A list. Several spectra files mean one of two different analyses, chosen by
    // `runMode`; a single file is an ordinary search either way.
    mzml: [], fasta: "", lib_precursors: "", lib_fragments: "", out_dir: "",
    lib_fasta: "", lib_out: "",
  },
  diannTimer: null,
  diannGetTimer: null,
  thermoTimer: null,
  thermoReady: false,
  msconvert: null,
  libSrc: "builtin",
  buildingLibrary: false,
  // "separate" = one search per file; "experiment" = one pooled run-experiment.
  runMode: "separate",
  // Batch progress: which file of how many, and what each one produced.
  batch: null,
  diannOffer: null,
  runId: null,
  timer: null,
  lastStatus: null,
  outDir: "",
  componentsReady: false,
  setupTimer: null,
  schema: null,
  overrides: {},
  savedConfig: null,
  // Output folders this application has used. The only thing it remembers; what
  // each one contains is read back from the folder.
  known: JSON.parse(localStorage.getItem("mumdia.folders") || "[]"),
};

function rememberFolder(dir) {
  if (!dir || state.known.includes(dir)) return;
  state.known.unshift(dir);
  state.known = state.known.slice(0, 50);
  try {
    localStorage.setItem("mumdia.folders", JSON.stringify(state.known));
  } catch {
    // A browser storage that refuses to write is not worth failing a search over.
  }
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
  if (cores) $("cores").textContent = `${cores} cores available`;

  try {
    const info = await invoke("engine_info");
    $("engine-line").textContent = `${info.version}\n${info.path}`;
    $("engine-line").title = `${info.version} — found via ${info.source}\n${info.path}`;
  } catch (e) {
    $("engine-line").textContent = "engine not found";
    banner($("engine-error"), String(e));
    $("start").disabled = true;
  }

  try {
    const list = await invoke("presets");
    const sel = $("preset");
    for (const p of list) {
      const o = document.createElement("option");
      o.value = p.path;
      o.textContent = p.name;
      sel.appendChild(o);
    }
  } catch {
    /* Presets are a convenience; engine defaults remain available without them. */
  }

  for (const b of document.querySelectorAll(".nav")) {
    b.addEventListener("click", () => {
      if (b.disabled) return;
      screen(b.dataset.screen);
      if (b.dataset.screen === "setup") { refreshComponents(); refreshDiann(); refreshThermo(); }
      if (b.dataset.screen === "settings") loadSettings();
      if (b.dataset.screen === "history") loadHistory();
    });
  }
  for (const t of document.querySelectorAll(".tab")) {
    t.addEventListener("click", () => setMode(t.dataset.mode));
  }
  for (const r of document.querySelectorAll('input[name="runmode"]')) {
    r.addEventListener("change", () => (state.runMode = r.value));
  }
  for (const r of document.querySelectorAll('input[name="libsrc"]')) {
    r.addEventListener("change", () => {
      selectLibSrc(r.value);
      if (r.value === "diann") refreshLibSrcNote();
    });
  }
  for (const b of document.querySelectorAll("[data-pick]")) {
    b.addEventListener("click", () => pick(b.dataset.pick));
  }
  $("start").addEventListener("click", start);
  $("install-primary").addEventListener("click", () => installComponents("primary"));
  $("install-ms2pip").addEventListener("click", () => installComponents("ms2pip"));
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
  refreshComponents();
  refreshDiann();
  refreshThermo();
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

async function loadSettings() {
  if (state.schema) return;
  try {
    state.schema = await invoke("config_schema");
  } catch (e) {
    banner($("settings-error"), String(e));
    return;
  }
  $("settings-search").addEventListener("input", renderSettings);
  $("only-changed").addEventListener("change", renderSettings);
  $("save-settings").addEventListener("click", saveSettings);
  renderSettings();
}

/// The value currently shown for a setting: an override if one was typed, else the
/// engine's default.
function currentValue(f) {
  return f.path in state.overrides ? state.overrides[f.path] : f.default;
}

function renderSettings() {
  const q = $("settings-search").value.trim().toLowerCase();
  const onlyChanged = $("only-changed").checked;
  const bySection = new Map();

  for (const f of state.schema.fields) {
    const changed = f.path in state.overrides;
    if (onlyChanged && !changed) continue;
    if (q && !(f.path.toLowerCase().includes(q) || f.help.toLowerCase().includes(q))) continue;
    const sec = f.section || "(top level)";
    if (!bySection.has(sec)) bySection.set(sec, []);
    bySection.get(sec).push(f);
  }

  const esc = (t) =>
    String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;");

  const parts = [];
  for (const [sec, fields] of bySection) {
    parts.push(`<div class="sec-title">${esc(sec)}</div><div class="card">`);
    for (const f of fields) {
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
            .map((c) => `<option value="${esc(c)}"${c === v ? " selected" : ""}>${esc(c)}</option>`)
            .join("") +
          `</select>`;
      } else {
        control = `<input type="text" data-path="${esc(f.path)}" value="${esc(v ?? "")}">`;
      }
      // A gated parameter is one the project documents as not to be changed from a
      // single benchmark count. Saying so where the decision is made is the whole
      // reason the schema carries the marker.
      const gate = f.gates.length
        ? `<span class="pill warn" title="${esc(f.gates.join(", "))}">gated</span>`
        : "";
      const badge = changed ? `<span class="pill info">changed</span>` : "";
      parts.push(
        `<div class="setting${changed ? " changed" : ""}">` +
          `<div class="sname">${esc(f.path)} ${gate} ${badge}</div>` +
          `<div class="shelp">${esc(f.help || "No description in the engine source.")}` +
          (f.default !== null && f.default !== undefined
            ? ` <em>Default: ${esc(f.default)}</em>`
            : "") +
          `</div><div class="sctl">${control}</div></div>`
      );
    }
    parts.push(`</div>`);
  }

  const list = $("settings-list");
  list.innerHTML =
    parts.join("") || `<p class="hint">Nothing matches that search.</p>`;

  for (const el of list.querySelectorAll("[data-path]")) {
    el.addEventListener("change", () => onSettingChanged(el));
  }

  const n = Object.keys(state.overrides).length;
  $("settings-sub").textContent =
    `${state.schema.fields.length} settings. ` +
    (n ? `${n} changed from the defaults; only those are saved.` : "None changed.");
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
  else if (f.kind === "integer" || f.kind === "float") {
    const n = Number(raw);
    if (raw.trim() === "" || Number.isNaN(n)) {
      banner($("settings-error"), `${path} must be a number.`);
      return;
    }
    value = n;
  }
  banner($("settings-error"), "");
  if (JSON.stringify(value) === JSON.stringify(f.default)) delete state.overrides[path];
  else state.overrides[path] = value;
  renderSettings();
}

async function saveSettings() {
  banner($("settings-error"), "");
  banner($("settings-saved"), "");
  try {
    const path = await invoke("save_settings", {
      name: "console",
      overrides: state.overrides,
    });
    state.savedConfig = path;
    // The saved file becomes the preset the next search uses, so the settings on
    // this screen and the settings a run uses cannot diverge.
    const sel = $("preset");
    let opt = [...sel.options].find((o) => o.value === path);
    if (!opt) {
      opt = document.createElement("option");
      opt.value = path;
      opt.textContent = "My settings";
      sel.appendChild(opt);
    }
    sel.value = path;
    banner(
      $("settings-saved"),
      `Saved and accepted by the engine. The next search will use these settings.`
    );
  } catch (e) {
    banner($("settings-error"), String(e));
  }
}

// ── components ──────────────────────────────────────────────────────────────
// Polled while an installation runs, then left alone. The backend owns the state,
// so closing and reopening this screen shows the truth rather than a stale copy.
async function refreshComponents() {
  let c;
  try {
    c = await invoke("components_status");
  } catch (e) {
    banner($("setup-error"), String(e));
    return;
  }
  const p = c.primary;
  state.componentsReady = !!p.complete;

  const pill = $("primary-pill");
  const btn = $("install-primary");
  const installing = p.install_status === "installing";
  show($("primary-bar"), installing);
  show($("install-log-card"), installing || p.install_status === "failed");

  if (installing) {
    pill.textContent = "installing…";
    pill.className = "pill warn";
    btn.disabled = true;
  } else if (p.complete) {
    pill.textContent = "installed";
    pill.className = "pill ok";
    btn.disabled = true;
    btn.textContent = "Installed";
  } else {
    pill.textContent = "not installed";
    pill.className = "pill bad";
    btn.disabled = !c.primary.uv;
    btn.textContent = "Install";
  }

  if (!c.primary.uv && !p.complete) {
    banner(
      $("setup-error"),
      "The installer component `uv` was not found beside the application or on PATH, " +
        "so components cannot be installed automatically."
    );
  } else if (p.error) {
    banner($("setup-error"), p.error);
  } else {
    banner($("setup-error"), "");
  }

  const m = c.ms2pip;
  const mpill = $("ms2pip-pill");
  const mbtn = $("install-ms2pip");
  if (m.install_status === "installing") {
    mpill.textContent = "installing…";
    mpill.className = "pill warn";
    mbtn.disabled = true;
  } else if (m.complete) {
    mpill.textContent = "installed";
    mpill.className = "pill ok";
    mbtn.disabled = true;
    mbtn.textContent = "Installed";
  } else {
    mpill.textContent = "optional";
    mpill.className = "pill mute";
    mbtn.disabled = false;
  }

  // One log pane, showing whichever installation is talking.
  const active = m.install_status === "installing" ? m : p;
  const logEl = $("install-log");
  const text = (active.install_log || []).join("\n");
  if (logEl.textContent !== text) {
    logEl.textContent = text;
    logEl.scrollTop = logEl.scrollHeight;
  }

  const versions = Object.entries(p.versions || {});
  $("setup-versions").textContent = versions.length
    ? versions.map(([k, v]) => `${k} ${v}`).join(" · ")
    : "";

  // The DIA-NN card's build button depends on `componentsReady`, which is only
  // known here.
  const busy = installing || m.install_status === "installing";
  if (busy && !state.setupTimer) {
    state.setupTimer = setInterval(refreshComponents, 900);
  } else if (!busy && state.setupTimer) {
    clearInterval(state.setupTimer);
    state.setupTimer = null;
  }
}

async function installComponents(env) {
  banner($("setup-error"), "");
  try {
    await invoke("components_install", { env });
  } catch (e) {
    banner($("setup-error"), String(e));
    return;
  }
  refreshComponents();
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
  // DIA-NN is available and licensed.

  refreshDiannOffer(d);
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
    "Bruker diaPASEF: MuMDIA's pipeline is 3D and discards ion mobility, so a " +
      "diaPASEF file loses the mobility separation that makes it selective. It will " +
      "search, with calibrated q values, but with considerably fewer identifications " +
      "than a 4D engine on the same data."
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

  if (v.needs === "ThermoParser") {
    body.textContent = state.thermoReady
      ? `MuMDIA reads mzML, so this is converted first with ThermoRawFileParser.${common}`
      : "This needs the Thermo .raw converter, which is not installed. Go to Setup and " +
        "install it, or convert to mzML yourself and select that.";
  } else {
    let extra = "";
    if (v.label === "Bruker .d") {
      // The thing a diaPASEF user must be told, and cannot infer.
      extra =
        " Note that MuMDIA's pipeline is 3D and discards ion mobility. For diaPASEF " +
        "that removes the separation the acquisition exists to produce, so expect " +
        "considerably fewer identifications than a 4D engine. The q values stay " +
        "calibrated; the sensitivity does not.";
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
};
// Pickers that choose a folder rather than a file.
// `mzml_dir` is here because Bruker/Agilent `.d` and Waters `.raw` are
// directories, and a file dialog cannot select one.
const DIR_PICKS = new Set(["out_dir", "lib_out", "mzml_dir"]);

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
  for (const [i, f] of files.entries()) {
    const li = document.createElement("li");

    const name = document.createElement("span");
    name.className = "fname";
    name.textContent = baseName(f);
    li.appendChild(name);

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
      renderMzmlList();
      updateRawNote();
    });
    li.appendChild(drop);

    list.appendChild(li);
  }
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
function libraryParams(fasta) {
  const n = (id, dflt) => {
    const el = $(id);
    if (!el) return dflt;
    const v = parseInt(el.value, 10);
    return Number.isFinite(v) ? v : dflt;
  };
  const cb = (id, dflt) => {
    const el = $(id);
    return el ? el.checked : dflt;
  };
  return {
    fasta,
    out_dir: "",
    missed_cleavages: n("d-missed", 1),
    min_pep_len: n("d-minlen", 7),
    max_pep_len: n("d-maxlen", 30),
    min_charge: n("d-minz", 2),
    max_charge: n("d-maxz", 4),
    threads: n("d-threads", 8),
    carbamidomethyl: cb("d-cam", true),
    oxidation: cb("d-ox", true),
  };
}

// Say, on the search screen, whether choosing DIA-NN means waiting.
async function refreshLibSrcNote() {
  const note = $("libsrc-diann-note");
  const radio = $("libsrc-diann");
  if (!note || !radio) return;
  const fasta = state.picks.fasta;
  if (!fasta) {
    radio.disabled = true;
    note.textContent = "Choose a FASTA first.";
    return;
  }
  let plan;
  try {
    plan = await invoke("diann_library_plan", { req: libraryParams(fasta) });
  } catch (e) {
    radio.disabled = true;
    // The reason is the backend's: no DIA-NN, or the licence not acknowledged.
    note.textContent = `Unavailable: ${e}`;
    if (state.libSrc === "diann") selectLibSrc("builtin");
    return;
  }
  radio.disabled = false;
  note.textContent = plan.ready
    ? `Already built for this FASTA — the search will start immediately (${plan.diann_version}).`
    : `Not built yet: DIA-NN will predict it first, which takes a while on a whole proteome (${plan.diann_version}).`;
}

function selectLibSrc(which) {
  state.libSrc = which;
  const r = $(which === "diann" ? "libsrc-diann" : "libsrc-builtin");
  if (r) r.checked = true;
  show($("libsrc-params"), which === "diann");
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
  screen("progress");
  state.buildingLibrary = true;
  $("stages").innerHTML = "";
  $("prog-title").textContent = "Predicting the library";
  $("prog-sub").textContent = "DIA-NN is predicting the spectral library";
  $("log").textContent = "";
  try {
    await invoke("diann_build", { req });
  } catch (e) {
    banner($("start-error"), String(e));
    leaveLibraryPhase();
    screen("input");
    return null;
  }

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
    $("prog-sub").textContent = b.step || "working";
    if (b.status === "failed") {
      banner($("start-error"), b.error || "The library build failed.");
      leaveLibraryPhase();
      screen("input");
      return null;
    }
    if (b.status === "done" && b.precursors) {
      // Hand the screen back to the search, which is about to start on it.
      leaveLibraryPhase();
      $("prog-sub").textContent = "Starting";
      $("log").textContent = "";
      return [b.precursors, b.fragments];
    }
  }
}

// One search per file, in sequence.
//
// Sequential and not parallel on purpose: a search already saturates the machine, and
// two at once would compete for the same cores and memory while making the progress
// display meaningless. The engine's own `experiment.parallel_runs` exists for the
// pooled case where it can be reasoned about.
async function runBatch(p, built, threads) {
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
      config: $("preset").value || null,
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
    state.batch.results.push({ file, ...outcome });
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
  show($("kpis"), false);
  const host = $("res-files");
  if (!host) return;
  host.innerHTML = "";
  for (const r of b.results) {
    const row = document.createElement("div");
    const n = r.results ? fmtInt(r.results.peptides_1pct) : null;
    row.textContent = r.error
      ? `${baseName(r.file)} — failed: ${r.error}`
      : `${baseName(r.file)} — ${n ?? "?"} peptides at 1%`;
    host.appendChild(row);
  }
}

async function start() {
  banner($("start-error"), "");
  const p = state.picks;
  const threads = parseInt($("threads").value, 10);

  if (p.mzml.length === 0) {
    banner($("start-error"), "Add at least one spectra file.");
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

  // Several files searched separately are N independent runs. Queued in the frontend
  // so `start_run` stays the single tested path and each run is an ordinary one; the
  // cost is that closing the window ends the queue, which the per-file output folders
  // make recoverable.
  if (p.mzml.length > 1 && currentRunMode() === "separate") {
    return runBatch(p, built, threads);
  }

  const req = {
    // A sequence: `run::Request.mzml` is `Vec<String>`. Sending the bare string made
    // serde reject every payload, so preflight AND start_run both failed and the Start
    // button did nothing in every mode. `ci/check_desktop_ui.py` cannot catch this --
    // it checks element ids and command names, never payload types.
    mzml: p.mzml,
    experiment: currentRunMode() === "experiment",
    out_dir: p.out_dir,
    fasta: state.mode === "fasta" && !built ? p.fasta || null : null,
    lib_precursors: built ? built[0] : state.mode === "library" ? p.lib_precursors || null : null,
    lib_fragments: built ? built[1] : state.mode === "library" ? p.lib_fragments || null : null,
    config: $("preset").value || null,
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
      renderResults(s);
      screen("results");
    }
  }
}

// ── history ─────────────────────────────────────────────────────────────────
async function loadHistory() {
  const list = $("history-list");
  let entries = [];
  try {
    entries = await invoke("history", { dirs: state.known });
  } catch (e) {
    list.innerHTML = `<p class="hint">Could not read past searches: ${e}</p>`;
    return;
  }
  if (!entries.length) {
    list.innerHTML = `<p class="hint">No past searches yet.</p>`;
    return;
  }
  const esc = (t) =>
    String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  list.innerHTML = entries
    .map((e) => {
      const r = e.results;
      const when = e.finished_unix_ms
        ? new Date(e.finished_unix_ms).toLocaleString()
        : "";
      const counts = r
        ? `${fmtInt(r.peptides_1pct)} peptides at peptide_q_value 0.01 · ` +
          `${fmtInt(r.protein_groups_1pct)} protein groups · ${esc(r.classifier)}`
        : "no scored table in this folder";
      return (
        `<div class="card"><div class="card-head"><div>` +
        `<div class="card-title">${esc(e.name)}</div>` +
        `<div class="card-sub">${counts}<br>${esc(when)} · ${esc(e.out_dir)}` +
        (e.engine_version ? ` · ${esc(e.engine_version)}` : "") +
        `</div></div>` +
        `<button class="btn quiet" data-open="${esc(e.out_dir)}">Open folder</button>` +
        `</div></div>`
      );
    })
    .join("");
  for (const b of list.querySelectorAll("[data-open]")) {
    b.addEventListener("click", () => invoke("reveal", { path: b.dataset.open }));
  }
}

// ── rendering ───────────────────────────────────────────────────────────────
function render(s) {
  const seen = new Map(s.stages.map((x) => [x.name, x]));
  const expected = STAGES_FASTA.filter(
    ([key]) => !(s.library_mode && SKIP_IN_LIBRARY_MODE.has(key))
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

  const titles = {
    starting: "Starting",
    running: "Searching",
    done: "Finished",
    failed: "Search failed",
    cancelled: "Search stopped",
  };
  $("prog-title").textContent = titles[s.status] || s.status;

  const done = expected.filter(([k]) => seen.has(k)).length;
  const parts = [];
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

function renderResults(s) {
  // The per-file breakdown belongs to a batch; a single run shows its own counts.
  show($("res-batch-card"), false);
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
    // them is diluted by roughly 1/n_runs. It also never calls the report stage, so
    // there is no peptides.tsv or proteins.tsv to fall back on.
    banner(
      $("res-warn"),
      "These are EXPERIMENT-WIDE counts, pooled across every run, not per file. " +
        "An experiment-wide rescore groups the q values across the whole experiment, " +
        "so dividing by the number of runs does not give a per-file number; the " +
        "per-file unit is run_psm_q in the split tables. This mode writes no " +
        "peptides.tsv or proteins.tsv."
    );
  } else {
    banner($("res-warn"), "");
  }

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
  if (r?.has_proteins_tsv) files.push(["proteins.tsv", "protein groups"]);
  files.push(["psms_scored.parquet", "every scored match, with all features"]);
  files.push(["manifest.json", "engine version, commit, and a hash of every input"]);
  $("files").innerHTML = files
    .map(([f, d]) => `<div><code>${f}</code> &nbsp;<span class="hint inline">${d}</span></div>`)
    .join("");
}

init();
