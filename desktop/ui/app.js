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
};

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
    b.addEventListener("click", () => !b.disabled && screen(b.dataset.screen));
  }
  for (const t of document.querySelectorAll(".tab")) {
    t.addEventListener("click", () => setMode(t.dataset.mode));
  }
  for (const b of document.querySelectorAll("[data-pick]")) {
    b.addEventListener("click", () => pick(b.dataset.pick));
  }
  $("start").addEventListener("click", start);
  $("stop").addEventListener("click", stop);
  $("another").addEventListener("click", () => screen("input"));
  $("open-folder").addEventListener("click", () => invoke("reveal", { path: state.outDir }));
  $("copy-cmd").addEventListener("click", async () => {
    await navigator.clipboard.writeText($("cmd").textContent);
    $("copy-cmd").textContent = "Copied";
    setTimeout(() => ($("copy-cmd").textContent = "Copy"), 1200);
  });
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
  const el = $(LABEL[what]);
  // Shown right-to-left so the filename stays visible on a long path; the full
  // path is the tooltip.
  el.textContent = path;
  el.title = path;
  el.classList.add("set");
  banner($("start-error"), "");
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

  try {
    state.runId = await invoke("start_run", { req });
  } catch (e) {
    banner($("start-error"), String(e));
    return;
  }

  state.outDir = p.out_dir;
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
