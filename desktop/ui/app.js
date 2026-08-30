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
  picks: { mzml: "", fasta: "", lib_precursors: "", lib_fragments: "", out_dir: "" },
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
      if (b.dataset.screen === "setup") refreshComponents();
      if (b.dataset.screen === "settings") loadSettings();
      if (b.dataset.screen === "history") loadHistory();
    });
  }
  for (const t of document.querySelectorAll(".tab")) {
    t.addEventListener("click", () => setMode(t.dataset.mode));
  }
  for (const b of document.querySelectorAll("[data-pick]")) {
    b.addEventListener("click", () => pick(b.dataset.pick));
  }
  $("start").addEventListener("click", start);
  $("install-primary").addEventListener("click", () => installComponents("primary"));
  $("install-ms2pip").addEventListener("click", () => installComponents("ms2pip"));
  refreshComponents();
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

function setMode(mode) {
  state.mode = mode;
  for (const t of document.querySelectorAll(".tab")) t.classList.toggle("on", t.dataset.mode === mode);
  show($("mode-fasta"), mode === "fasta");
  show($("mode-library"), mode === "library");
}

// ── file pickers ────────────────────────────────────────────────────────────
const FILTERS = {
  mzml: [{ name: "mzML", extensions: ["mzML", "mzml"] }],
  fasta: [{ name: "FASTA", extensions: ["fasta", "fa", "fas"] }],
  lib_precursors: [{ name: "Parquet", extensions: ["parquet"] }],
  lib_fragments: [{ name: "Parquet", extensions: ["parquet"] }],
};
const LABEL = {
  mzml: "p-mzml",
  fasta: "p-fasta",
  lib_precursors: "p-libp",
  lib_fragments: "p-libf",
  out_dir: "p-out",
};

async function pick(what) {
  const chosen =
    what === "out_dir"
      ? await dialog.open({ directory: true, multiple: false })
      : await dialog.open({ multiple: false, filters: FILTERS[what] });
  if (!chosen) return;
  const path = Array.isArray(chosen) ? chosen[0] : chosen;
  state.picks[what] = path;
  if (what === "mzml") showPeakCensus(path);
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
async function start() {
  banner($("start-error"), "");
  const p = state.picks;
  const threads = parseInt($("threads").value, 10);
  const req = {
    mzml: p.mzml,
    out_dir: p.out_dir,
    fasta: state.mode === "fasta" ? p.fasta || null : null,
    lib_precursors: state.mode === "library" ? p.lib_precursors || null : null,
    lib_fragments: state.mode === "library" ? p.lib_fragments || null : null,
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
