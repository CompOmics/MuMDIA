// Behavioural tests of the desktop frontend's Setup screen, run without a window.
//
// `ci/check_desktop_ui.py` checks that the frontend and the backend agree on names; it
// cannot see a fault that only shows when the code runs. One did: the Setup screen read
// the optional environments' status before declaring it, so every refresh threw, the
// MS2PIP and visDIA cards never left their first state and the poll never started.
// Clicking Install looked like nothing happened while the installation ran to the end.
//
// This loads `desktop/ui/app.js` into a `vm` context with a minimal document and a fake
// Tauri `invoke`, then drives the component functions with the statuses
// `components_status` returns. Node only, no dependencies.
//
// Usage:
//     node ci/test_desktop_ui.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";
import assert from "node:assert/strict";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
// `APP_JS` points the harness at another copy of the frontend, to show a test failing
// on the code it was written against.
const SOURCE = readFileSync(process.env.APP_JS || join(ROOT, "desktop", "ui", "app.js"), "utf8");
// `INDEX_HTML` does the same for the markup.
const HTML = readFileSync(
  process.env.INDEX_HTML || join(ROOT, "desktop", "ui", "index.html"),
  "utf8"
);

// Enough of a parse of `index.html` to say where an element sits: the elements that
// enclose each element with an id (outermost first, as tag and id), and the order in
// which the ids appear. The markup closes its elements explicitly; an end tag closes the
// nearest open element of its name and whatever is still open inside it.
function htmlTree(html) {
  const VOID = new Set([
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
    "track", "wbr",
  ]);
  const stack = [];
  const ancestors = new Map();
  const order = [];
  const text = html.replace(/<!--[\s\S]*?-->/g, "");
  for (const [, close, name, attrs] of text.matchAll(/<(\/?)([a-zA-Z][\w-]*)([^>]*)>/g)) {
    const tag = name.toLowerCase();
    if (close) {
      const at = stack.map((e) => e.tag).lastIndexOf(tag);
      assert.notEqual(at, -1, `index.html: </${tag}> closes nothing`);
      stack.length = at;
      continue;
    }
    const id = /(?:^|\s)id="([^"]+)"/.exec(attrs)?.[1] ?? null;
    if (id) {
      ancestors.set(id, stack.map((e) => ({ ...e })));
      order.push(id);
    }
    if (!VOID.has(tag) && !/\/\s*$/.test(attrs)) stack.push({ tag, id });
  }
  return { ancestors, order };
}

function element(id) {
  const attrs = new Map();
  return {
    id,
    textContent: "",
    className: "",
    disabled: false,
    title: "",
    value: "",
    scrollTop: 0,
    scrollHeight: 0,
    style: {},
    dataset: {},
    firstElementChild: { style: {} },
    classList: { toggle() {}, add() {}, remove() {}, contains: () => false },
    selectedOptions: [],
    setAttribute: (k, v) => attrs.set(k, v),
    removeAttribute: (k) => attrs.delete(k),
    hasAttribute: (k) => attrs.has(k),
    addEventListener() {},
    appendChild() {},
    append() {},
    focus() {},
    querySelectorAll: () => [],
    get hidden() {
      return attrs.has("hidden");
    },
  };
}

// A fresh frontend: its own document, timers and backend answers.
function load() {
  const els = new Map();
  const timers = new Set();
  const calls = [];
  // A working, acknowledged DIA-NN: the Search screen asks for it whenever the library
  // settings change (the ZT Scan analysis sets them).
  const answers = {
    components_status: () => status(),
    diann_status: () => ({ status: { runs: true, licence_acknowledged: true } }),
  };
  // The dialog plugin: a test sets `save`, `open` or `ask` when it needs one.
  const dialog = {};
  const document = {
    getElementById(id) {
      if (!els.has(id)) els.set(id, element(id));
      return els.get(id);
    },
    querySelectorAll: () => [],
    createElement: () => element(""),
  };
  const ctx = {
    window: {
      __TAURI__: {
        core: {
          invoke: async (name, args) => {
            calls.push([name, args]);
            const a = answers[name];
            return a ? a(args) : {};
          },
        },
        dialog,
      },
    },
    document,
    navigator: {},
    console,
    setInterval: (fn) => {
      const t = { fn };
      timers.add(t);
      return t;
    },
    clearInterval: (t) => timers.delete(t),
    setTimeout: () => 0,
    clearTimeout() {},
  };
  vm.createContext(ctx);
  // Everything but the start-up call, which would wire the whole window.
  const body = SOURCE.replace(/^init\(\);\s*$/m, "");
  assert.notEqual(body, SOURCE, "app.js no longer ends with init(); update this harness");
  // Each name guarded, so an older frontend without the helpers still loads and fails
  // only the tests that need them.
  const names = [
    "refreshComponents",
    "installComponents",
    "optionalCardView",
    "state",
    "refreshNormOptions",
    "normOverrides",
    "searchProblems",
    "refreshZtscanNote",
    "ztscanOverrides",
    "initMods",
    "initCleavage",
    "loadSettings",
    "seedOverridesFromPreset",
    "saveSettings",
    "setMode",
    "resolveBands",
    "runOverrides",
    "captureSearch",
    "currentSetup",
    "validateSetup",
    "planSetup",
    "reviewSetup",
    "applyPendingSetup",
    "cancelPendingSetup",
    "saveNamedSetup",
    "exportCurrentSetup",
    "importSetupFile",
    "refreshSetups",
    "nestPaths",
    "canonical",
    "SETUP_SPEC",
    "LOCAL_SETTINGS",
    "INTERPRETER_SETTINGS",
  ];
  const exported = names.map((n) => `${n}: typeof ${n} === "undefined" ? undefined : ${n}`);
  const api = vm.runInContext(`${body}\n;({ ${exported.join(", ")} });`, ctx);
  return { api, el: (id) => document.getElementById(id), timers, calls, answers, dialog };
}

function env(over = {}) {
  return {
    python: null,
    complete: false,
    missing: [],
    versions: {},
    install_status: "idle",
    install_log: [],
    error: null,
    uv: "/bin/uv",
    outdated: [],
    load_check: "",
    load_error: null,
    ...over,
  };
}

function status({ primary = {}, ms2pip = {}, visdia = {} } = {}) {
  return {
    primary: env({ complete: true, load_check: "ok", ...primary }),
    ms2pip: env(ms2pip),
    visdia: env(visdia),
    vc_runtime: { available: false, needed: false, state: {} },
  };
}

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test("an installing optional environment shows on its card and keeps the poll on", async () => {
  for (const [key, pillId, btnId] of [
    ["visdia", "visdia-pill", "install-visdia"],
    ["ms2pip", "ms2pip-pill", "install-ms2pip"],
  ]) {
    const f = load();
    f.answers.components_status = () =>
      status({ [key]: { install_status: "installing", install_log: ["Resolved 42 packages"] } });
    await f.api.refreshComponents();
    assert.equal(f.el(pillId).textContent, "installing…", key);
    assert.equal(f.el(btnId).disabled, true, key);
    assert.equal(f.el("install-log").textContent, "Resolved 42 packages", key);
    assert.equal(f.el("install-log-card").hidden, false, key);
    assert.equal(f.el("setup-error").hidden, true, `${key}: ${f.el("setup-error").textContent}`);
    assert.equal(f.timers.size, 1, `${key}: the poll must run while it installs`);
  }
});

test("the poll follows the installation through the load check to installed", async () => {
  const f = load();
  const phases = [
    { install_status: "installing", install_log: ["downloading"] },
    { install_status: "installing", install_log: ["downloading", "testing the packages"] },
    {
      install_status: "done",
      complete: true,
      load_check: "ok",
      install_log: ["downloading", "testing the packages", "done"],
    },
  ];
  let i = 0;
  f.answers.components_status = () => status({ visdia: phases[Math.min(i, phases.length - 1)] });
  await f.api.refreshComponents();
  assert.equal(f.timers.size, 1);
  const [timer] = f.timers;
  for (i = 1; i < phases.length; i++) await timer.fn();
  assert.equal(f.el("visdia-pill").textContent, "installed");
  assert.equal(f.el("install-visdia").textContent, "Installed");
  assert.equal(f.el("install-visdia").disabled, true);
  assert.equal(f.api.state.visdiaReady, true);
  assert.equal(f.timers.size, 0, "the poll stops once nothing is moving");
  assert.match(f.el("install-log").textContent, /done$/);
});

test("clicking Install gives feedback before the backend answers", async () => {
  for (const key of ["visdia", "ms2pip", "primary"]) {
    const f = load();
    let release;
    f.answers.components_install = () => new Promise((r) => (release = r));
    f.answers.components_status = () =>
      status({ [key]: { install_status: "installing", complete: false } });
    const done = f.api.installComponents(key);
    // Still waiting on the backend.
    assert.equal(f.el(`${key}-pill`).textContent, "installing…", key);
    assert.equal(f.el(`install-${key}`).disabled, true, key);
    release();
    await done;
    // Objects from the vm context are of another realm, so compare their JSON.
    assert.equal(
      JSON.stringify(f.calls.find(([n]) => n === "components_install")),
      JSON.stringify(["components_install", { env: key }])
    );
    assert.equal(f.el(`${key}-pill`).textContent, "installing…", key);
    assert.equal(f.timers.size, 1, key);
  }
});

test("a refused installation is shown in the banner and the card is restored", async () => {
  const f = load();
  f.answers.components_install = () => Promise.reject("an installation is already running");
  await f.api.installComponents("visdia");
  assert.equal(f.el("setup-error").hidden, false);
  assert.equal(f.el("setup-error").textContent, "an installation is already running");
  assert.equal(f.el("visdia-pill").textContent, "optional");
  assert.equal(f.el("install-visdia").textContent, "Install");
  assert.equal(f.el("install-visdia").disabled, false);
});

test("a failed optional installation shows its error, its log and Reinstall", async () => {
  const f = load();
  f.answers.components_status = () =>
    status({
      visdia: {
        install_status: "failed",
        error: "uv pip install exited with status 1",
        install_log: ["error: no matching distribution"],
      },
    });
  await f.api.refreshComponents();
  assert.equal(f.el("visdia-pill").textContent, "failed");
  assert.equal(f.el("install-visdia").textContent, "Reinstall");
  assert.equal(f.el("install-visdia").disabled, false);
  assert.equal(f.el("setup-error").textContent, "visDIA: uv pip install exited with status 1");
  assert.equal(f.el("install-log").textContent, "error: no matching distribution");
  assert.equal(f.el("install-log-card").hidden, false);
  assert.equal(f.timers.size, 0);
});

test("the optional card state table", () => {
  const { api } = load();
  const v = (over, uv = true) => {
    const r = api.optionalCardView(env(over), uv);
    return [r.pill, r.label, r.disabled];
  };
  assert.deepEqual(v({}), ["optional", "Install", false]);
  assert.deepEqual(v({}, false), ["optional", "Install", true]);
  assert.deepEqual(v({ install_status: "installing" }), ["installing…", "Installing…", true]);
  assert.deepEqual(v({ load_check: "running" }), ["testing…", "Testing…", true]);
  assert.deepEqual(v({ load_check: "failed", install_status: "failed" }), [
    "cannot load",
    "Reinstall",
    false,
  ]);
  assert.deepEqual(v({ complete: true, load_check: "ok" }), ["installed", "Installed", true]);
  assert.deepEqual(v({ outdated: ["mumdia-viewer 1a2b3c4 -> cb8b5bc"] }), [
    "update needed",
    "Update",
    false,
  ]);
  assert.deepEqual(v({ install_status: "failed" }), ["failed", "Reinstall", false]);
});

test("an optional environment that cannot load is named in the banner", async () => {
  const f = load();
  f.answers.components_status = () =>
    status({ ms2pip: { load_check: "failed", load_error: "torch cannot load", install_status: "failed" } });
  await f.api.refreshComponents();
  assert.equal(f.el("setup-error").textContent, "MS2PIP: torch cannot load");
  assert.equal(f.el("ms2pip-pill").textContent, "cannot load");
});

test("the normalisation note says what quantification runs under each scope", () => {
  const f = load();
  const { state } = f.api;
  state.picks.mzml = ["/d/a.mzML", "/d/b.mzML", "/d/c.mzML"];
  state.picks.conditions = { "/d/a.mzML": "A", "/d/b.mzML": "A", "/d/c.mzML": "B" };
  state.runMode = "experiment";
  f.el("norm-scope").value = "condition";
  f.api.refreshNormOptions();
  const note = f.el("norm-note").textContent;
  assert.match(note, /pattern-fit recipe/);
  assert.match(note, /without the stable-set normalisation/);
  assert.match(note, /MaxLFQ is normalised within each condition/);
  assert.match(note, /Conditions: A \(2\), B \(1\)\./);
  assert.deepEqual({ ...f.api.normOverrides() }, { "quant.normalisation_scope": "condition" });
  // Only the scope is sent: the engine keeps the rest of the recipe itself, and a sent
  // normalisation key would be an explicit request it refuses under `condition`.
  for (const scope of ["all", "none"]) {
    f.el("norm-scope").value = scope;
    f.api.refreshNormOptions();
    assert.doesNotMatch(f.el("norm-note").textContent, /stable-set/);
  }
  f.el("norm-scope").value = "all";
  assert.deepEqual({ ...f.api.normOverrides() }, {});
});

// The normalisation options belong to "One experiment". Placed after both run-mode
// options, they read as options of "Search each separately" (#298).
test("the normalisation options sit inside the One experiment choice, above the other", () => {
  const { ancestors, order } = htmlTree(HTML);
  const inside = (id) => {
    assert.ok(ancestors.has(id), `index.html has no element ${id}`);
    return ancestors.get(id).some((e) => e.id === "runmode-experiment-choice");
  };
  for (const id of [
    "runmode-experiment",
    "norm-options",
    "norm-scope",
    "suggest-conditions",
    "norm-note",
    "condition-labels",
  ]) {
    assert.ok(inside(id), `${id} is not inside the One experiment choice`);
  }
  assert.ok(!inside("runmode-separate"), "Search each separately is inside the experiment choice");
  // Beside the radio's label, not in it: a label holds one control, and a click on the
  // selector would select the radio as well.
  assert.ok(
    !ancestors.get("norm-options").some((e) => e.tag === "label"),
    "the normalisation options are inside a label"
  );
  const at = (id) => order.indexOf(id);
  assert.ok(at("runmode-experiment") < at("norm-options"), "under One experiment");
  assert.ok(at("norm-options") < at("runmode-separate"), "above Search each separately");
});

test("the normalisation options show only for one experiment of several files", () => {
  const f = load();
  const { state } = f.api;
  const two = ["/d/a.mzML", "/d/b.mzML"];
  for (const [files, mode, shown] of [
    [two, "experiment", true],
    [two, "separate", false],
    [["/d/a.mzML"], "experiment", false],
    [["/d/a.mzML"], "separate", false],
    [[], "experiment", false],
    [two, "experiment", true],
  ]) {
    state.picks.mzml = files;
    state.runMode = mode;
    f.api.refreshNormOptions();
    const what = `${files.length} file(s), ${mode}`;
    assert.equal(f.el("norm-options").hidden, !shown, what);
    assert.equal(f.el("norm-scope").disabled, !shown, what);
    assert.equal(f.el("suggest-conditions").disabled, !shown, what);
  }
});

test("a normalisation choice hidden by Search each separately is not sent", () => {
  const f = load();
  const { state } = f.api;
  state.mode = "library";
  state.picks.mzml = ["/d/a.mzML", "/d/b.mzML"];
  // b has no condition, which blocks "Within each condition" for an experiment.
  state.picks.conditions = { "/d/a.mzML": "A" };
  for (const scope of ["none", "condition"]) {
    f.el("norm-scope").value = scope;
    state.runMode = "experiment";
    f.api.refreshNormOptions();
    assert.deepEqual({ ...f.api.normOverrides() }, { "quant.normalisation_scope": scope });
    state.runMode = "separate";
    f.api.refreshNormOptions();
    assert.deepEqual({ ...f.api.normOverrides() }, {}, `${scope}, separate`);
  }
  // The start-time check of a condition per file follows the same choice.
  const blocked = () =>
    [...f.api.searchProblems()].some((s) => /Normalisation within each condition/.test(s));
  state.runMode = "experiment";
  assert.ok(blocked(), "an experiment without every condition must be refused");
  state.runMode = "separate";
  assert.ok(!blocked(), "separate searches have no normalisation to refuse");
  // One file is a plain search whatever the radio says.
  state.runMode = "experiment";
  state.picks.mzml = ["/d/a.mzML"];
  assert.deepEqual({ ...f.api.normOverrides() }, {});
  assert.ok(!blocked());
});

// The isolation census `scanning_probe` returns for a SCIEX ZT Scan DIA mzML.
function ztProbe(path, over = {}) {
  return {
    path,
    isolation: {
      ms2_spectra: 1996,
      with_offsets: 0,
      no_isolation: 0,
      target_only: 1996,
      convert_refuses: true,
      scanning_quadrupole: true,
      suggested_half_width: 1.9,
      suggested_half_width_steps: 2.5,
      layout: {
        distinct_targets: 663,
        lowest_target: 394.256,
        highest_target: 898.271,
        median_step: 0.7614,
        spectra_per_target: 3.01,
        cycle_s: 1.0,
      },
      ...over,
    },
    error: null,
  };
}

// What `ztscan_off` returns: the keys of the ZT Scan analysis, off.
const OFF = {
  "extract.q1_fragments": false,
  "quant.q1_cycle_halfwidth": 0,
  "rt_im_train.q1_calibrate": false,
};

test("a ZT Scan mzML is explained, its half-width shown and the analysis left to the engine", async () => {
  const f = load();
  const { state } = f.api;
  f.answers.scanning_probe = ({ mzml }) => mzml.map((p) => ztProbe(p));
  f.answers.ztscan_off = () => OFF;
  state.picks.mzml = ["/d/REP01.mzML", "/d/REP02.mzML"];
  await f.api.refreshZtscanNote();
  assert.equal(f.el("ztscan-note").hidden, false);
  assert.match(f.el("ztscan-body").textContent, /2 of the 2 selected files are a scanning-quadrupole/);
  assert.match(f.el("ztscan-body").textContent, /663 positions from 394\.3 to 898\.3 m\/z/);
  assert.equal(f.el("ztscan-hw").value, "1.9");
  assert.equal(f.el("ztscan-analysis").disabled, false);
  // The analysis is ticked for ZT Scan files until the user changes it, and the engine runs
  // it, with the census's half-width, by itself: nothing is written.
  assert.equal(f.el("ztscan-analysis").checked, true);
  assert.deepEqual({ ...f.api.ztscanOverrides() }, {});
  // What the checkbox listener records when the user clears it: the analysis's keys, off, and
  // a re-render keeps their choice.
  f.el("ztscan-analysis").checked = false;
  state.ztscan.analysisTouched = true;
  await f.api.refreshZtscanNote();
  assert.equal(f.el("ztscan-analysis").checked, false);
  assert.deepEqual({ ...f.api.ztscanOverrides() }, OFF);
  assert.match(f.el("ztscan-analysis-note").textContent, /half-width alone/);
  // Off with no known way to turn it off: the search is not started.
  state.ztscan.off = null;
  assert.ok([...f.api.searchProblems()].some((s) => /could not be turned off/.test(s)));
  state.ztscan.off = OFF;
  f.el("ztscan-analysis").checked = true;
  assert.ok(![...f.api.searchProblems()].some((s) => /could not be turned off/.test(s)));
  // What the input listener records when the user types: their half-width is written.
  f.el("ztscan-hw").value = "2.1";
  state.ztscan.hwTyped = true;
  assert.deepEqual({ ...f.api.ztscanOverrides() }, { "convert.isolation_half_width": 2.1 });
  f.el("ztscan-analysis").checked = false;
  assert.deepEqual({ ...f.api.ztscanOverrides() }, { ...OFF, "convert.isolation_half_width": 2.1 });
  f.el("ztscan-analysis").checked = true;
  // Each file is asked once, and a typed half-width survives the re-render.
  const probes = () => f.calls.filter(([n]) => n === "scanning_probe").length;
  assert.equal(probes(), 1);
  await f.api.refreshZtscanNote();
  assert.equal(probes(), 1);
  assert.equal(f.el("ztscan-hw").value, "2.1");
  // An empty half-width blocks the start.
  f.el("ztscan-hw").value = "";
  assert.ok([...f.api.searchProblems()].some((s) => /Q1 half-width/.test(s)));
});

test("the ZT Scan analysis is held back for a mixed selection, and other files need nothing", async () => {
  const f = load();
  const { state } = f.api;
  f.answers.scanning_probe = ({ mzml }) =>
    mzml.map((p) =>
      p.includes("orbi")
        ? ztProbe(p, { convert_refuses: false, scanning_quadrupole: false, target_only: 0 })
        : p.includes("fixed")
          ? ztProbe(p, { scanning_quadrupole: false, suggested_half_width: null })
          : ztProbe(p)
    );
  f.answers.ztscan_off = () => OFF;
  // A ZT Scan file with an Orbitrap file and a vendor file: the engine reads the ZT Scan file
  // with the census's half-width and leaves the analysis out; nothing is written, and the
  // vendor file is not probed.
  state.picks.mzml = ["/d/REP01.mzML", "/d/orbi.mzML", "/d/run.raw"];
  await f.api.refreshZtscanNote();
  const asked = f.calls.filter(([n]) => n === "scanning_probe").flatMap(([, a]) => a.mzml);
  assert.deepEqual([...asked], ["/d/REP01.mzML", "/d/orbi.mzML"]);
  assert.equal(f.el("ztscan-note").hidden, false);
  assert.equal(f.el("ztscan-analysis").disabled, true);
  assert.equal(f.el("ztscan-analysis").checked, false);
  assert.match(f.el("ztscan-analysis-note").textContent, /leaves the analysis out/);
  assert.deepEqual({ ...f.api.ztscanOverrides() }, {});
  // Only files with offsets: nothing to say and nothing added.
  state.picks.mzml = ["/d/orbi.mzML"];
  await f.api.refreshZtscanNote();
  assert.equal(f.el("ztscan-note").hidden, true);
  assert.deepEqual({ ...f.api.ztscanOverrides() }, {});
  assert.ok(![...f.api.searchProblems()].some((s) => /Q1 half-width/.test(s)));
  // A file that names targets without offsets in a fixed-window layout: the engine has no
  // half-width for it, so the one the user enters is written.
  state.picks.mzml = ["/d/fixed.mzML"];
  await f.api.refreshZtscanNote();
  assert.equal(f.el("ztscan-hw").value, "");
  assert.ok([...f.api.searchProblems()].some((s) => /Q1 half-width/.test(s)));
  f.el("ztscan-hw").value = "12.5";
  assert.deepEqual({ ...f.api.ztscanOverrides() }, { "convert.isolation_half_width": 12.5 });
});

test("the ZT Scan analysis sets the library's search space and takes back what it set", async () => {
  const f = load();
  const { state } = f.api;
  f.answers.scanning_probe = ({ mzml }) =>
    mzml.map((p) =>
      p.includes("orbi")
        ? ztProbe(p, { convert_refuses: false, scanning_quadrupole: false, target_only: 0 })
        : ztProbe(p)
    );
  f.answers.ztscan_off = () => OFF;
  // The Search space card as the page starts it, with the default modifications.
  const start = { "d-minlen": "7", "d-maxlen": "30", "d-minz": "2", "d-maxz": "3", "d-maxvar": "1", "d-prmin": "", "d-prmax": "" };
  const fields = () => Object.fromEntries(Object.keys(start).map((id) => [id, f.el(id).value]));
  for (const [id, v] of Object.entries(start)) f.el(id).value = v;
  state.mods = new Map([["Carbamidomethyl|C", "fix"], ["Oxidation|M", "var"]]);
  const acetyl = "Acetyl|*|protein_n_term";
  // A selection without ZT Scan files changes nothing.
  state.picks.mzml = ["/d/orbi.mzML"];
  await f.api.refreshZtscanNote();
  assert.deepEqual(fields(), start);
  assert.equal(state.ztscan.lib, null);
  // A ZT Scan file: the analysis is ticked, and with it the measured search space, the
  // precursors inside the swept range (394.256 - 1.9 to 898.271 + 1.9, rounded outward).
  state.picks.mzml = ["/d/REP01.mzML"];
  await f.api.refreshZtscanNote();
  assert.equal(f.el("ztscan-analysis").checked, true);
  assert.deepEqual(fields(), {
    "d-minlen": "6", "d-maxlen": "40", "d-minz": "1", "d-maxz": "4", "d-maxvar": "2", "d-prmin": "392", "d-prmax": "901",
  });
  assert.equal(state.mods.get(acetyl), "var");
  assert.equal(state.mods.get("Oxidation|M"), "var");
  assert.equal(state.mods.get("Carbamidomethyl|C"), "fix");
  assert.match(f.el("ztscan-library-note").textContent, /charges 1 to 4, lengths 6 to 40/);
  assert.match(f.el("ztscan-library-note").textContent, /from 392 to 901 m\/z/);
  // The range follows the half-width while the analysis set it.
  f.el("ztscan-hw").value = "3.0";
  state.ztscan.hwTyped = true;
  await f.api.refreshZtscanNote();
  assert.equal(f.el("d-prmin").value, "391");
  assert.equal(f.el("d-prmax").value, "902");
  // A value the user edits is theirs: it is neither replaced nor taken back.
  f.el("d-maxlen").value = "35";
  await f.api.refreshZtscanNote();
  assert.equal(f.el("d-maxlen").value, "35");
  // Clearing the analysis takes back the rest, and the modification it added.
  f.el("ztscan-analysis").checked = false;
  state.ztscan.analysisTouched = true;
  await f.api.refreshZtscanNote();
  assert.deepEqual(fields(), { ...start, "d-maxlen": "35" });
  assert.equal(state.mods.has(acetyl), false);
  assert.equal(state.mods.get("Oxidation|M"), "var");
  assert.equal(f.el("ztscan-library-note").textContent, "");
  // Ticking it again applies it again; removing the ZT Scan file takes it back.
  f.el("ztscan-analysis").checked = true;
  await f.api.refreshZtscanNote();
  assert.equal(f.el("d-minz").value, "1");
  assert.equal(state.mods.get(acetyl), "var");
  state.picks.mzml = ["/d/orbi.mzML"];
  await f.api.refreshZtscanNote();
  assert.equal(f.el("d-minz").value, "2");
  assert.equal(state.mods.has(acetyl), false);
  // A modification the user had chosen before is left alone when the analysis goes.
  state.mods.set(acetyl, "var");
  state.picks.mzml = ["/d/REP01.mzML"];
  await f.api.refreshZtscanNote();
  state.picks.mzml = ["/d/orbi.mzML"];
  await f.api.refreshZtscanNote();
  assert.equal(state.mods.get(acetyl), "var");
});

test("a deamidation choice for a ZT Scan run is warned about on the card", async () => {
  const f = load();
  const { state } = f.api;
  f.answers.scanning_probe = ({ mzml }) =>
    mzml.map((p) =>
      p.includes("orbi")
        ? ztProbe(p, { convert_refuses: false, scanning_quadrupole: false, target_only: 0 })
        : ztProbe(p)
    );
  f.answers.ztscan_off = () => OFF;
  state.mods = new Map([["Carbamidomethyl|C", "fix"], ["Oxidation|M", "var"]]);
  state.picks.mzml = ["/d/REP01.mzML"];
  await f.api.refreshZtscanNote();
  assert.equal(f.el("ztscan-mods-note").textContent, "");
  state.mods.set("Deamidated|N", "var");
  await f.api.refreshZtscanNote();
  assert.match(f.el("ztscan-mods-note").textContent, /13C isotope/);
  // Deamidation of R is not the N/Q alias; another instrument's file needs no warning.
  state.mods.delete("Deamidated|N");
  state.mods.set("Deamidated|R", "var");
  await f.api.refreshZtscanNote();
  assert.equal(f.el("ztscan-mods-note").textContent, "");
  state.mods.set("Deamidated|Q", "var");
  state.picks.mzml = ["/d/orbi.mzML"];
  await f.api.refreshZtscanNote();
  assert.equal(f.el("ztscan-mods-note").textContent, "");
});

// ── setups (#297) ──────────────────────────────────────────────────────────────────
// A setup saved on one profile and loaded on a fresh one must give the engine the same
// configuration. The frontends below read the real settings schema, modification catalogue,
// cleavage presets and example presets, and the backend commands they call are emulated
// from the Rust they stand for: `config_overrides` is `settings::flatten` of a file,
// `save_settings` writes `settings::nest` of the overrides, and a search's configuration is
// `settings::derive`, the preset (or the saved file) with the Search screen's overrides
// merged on top.

// A value from the frontend's context as one of this one, for a deep comparison.
const host = (v) => JSON.parse(JSON.stringify(v));

const readJson = (...parts) => JSON.parse(readFileSync(join(ROOT, ...parts), "utf8"));
const SCHEMA = readJson("configs", "config-schema.json");
// `settings::NOT_OFFERED` leaves the engine's opt-in for its native models out.
SCHEMA.fields = SCHEMA.fields.filter((f) => f.path !== "predict_frag.allow_native_predictors");
const CATALOGUE = readJson("rust", "mumdia", "crates", "mumdia-core", "src", "modifications.json");
const ENZYME_LIST = Object.entries(
  readJson("rust", "mumdia", "crates", "mumdia-core", "src", "enzymes.json").presets
).map(([name, p]) => ({ name, label: p.label || name, note: p.note || "", rule: p.rule ?? null }));
const PRESET_NAMES = ["diann-library", "fasta-sidecars", "ztscan"];
const PRESET_PATH = Object.fromEntries(PRESET_NAMES.map((n) => [n, `/presets/${n}.json`]));
const PRESET_CONTENT = Object.fromEntries(
  PRESET_NAMES.map((n) => [PRESET_PATH[n], readJson("configs", "examples", `${n}.json`)])
);
const SAVED = "/profile/configs/console.json";

// `settings::flatten`: objects recurse into dotted paths, everything else is a leaf.
function flattenJson(v, prefix = "", out = {}) {
  if (v && typeof v === "object" && !Array.isArray(v)) {
    for (const [k, c] of Object.entries(v)) flattenJson(c, prefix ? `${prefix}.${k}` : k, out);
  } else if (prefix) out[prefix] = v;
  return out;
}

function nest(flat) {
  const root = {};
  for (const [path, value] of Object.entries(flat)) {
    const parts = path.split(".");
    let cur = root;
    for (const p of parts.slice(0, -1)) cur = cur[p] ??= {};
    cur[parts[parts.length - 1]] = value;
  }
  return root;
}

// `settings::merge`: objects merge key by key, everything else is replaced.
function merge(base, over) {
  const isObj = (x) => x && typeof x === "object" && !Array.isArray(x);
  if (!isObj(base) || !isObj(over)) return structuredClone(over);
  const out = structuredClone(base);
  for (const [k, v] of Object.entries(over)) out[k] = isObj(out[k]) && isObj(v) ? merge(out[k], v) : structuredClone(v);
  return out;
}

function sortKeys(v) {
  if (Array.isArray(v)) return v.map(sortKeys);
  if (v && typeof v === "object") return Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])]));
  return v;
}

// The values index.html starts its fields with: an input's `value`, a select's selected
// (or first) option.
function htmlDefaults(html) {
  const out = {};
  for (const [, attrs] of html.matchAll(/<input([^>]*)>/g)) {
    const id = /\bid="([^"]+)"/.exec(attrs)?.[1];
    const value = /\bvalue="([^"]*)"/.exec(attrs)?.[1];
    if (id && value !== undefined && !/type="(radio|checkbox)"/.test(attrs)) out[id] = value;
  }
  for (const [, id, body] of html.matchAll(/<select id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
    const opts = [...body.matchAll(/<option value="([^"]*)"([^>]*)>/g)];
    const sel = opts.find(([, , a]) => /selected/.test(a)) || opts[0];
    if (sel) out[id] = sel[1];
  }
  return out;
}

function selectOptions(html, id) {
  const m = new RegExp(`<select id="${id}"[^>]*>([\\s\\S]*?)</select>`).exec(html);
  assert.ok(m, `index.html has no select ${id}`);
  return [...m[1].matchAll(/<option value="([^"]*)"/g)].map((x) => x[1]);
}

// A frontend on a profile of its own: its saved settings file and its named setups.
async function profile() {
  const f = load();
  const files = { ...structuredClone(PRESET_CONTENT) };
  const store = new Map();
  Object.assign(f.answers, {
    config_schema: () => structuredClone(SCHEMA),
    modifications: () => structuredClone(CATALOGUE),
    enzymes: () => structuredClone(ENZYME_LIST),
    config_overrides: ({ path }) => {
      assert.ok(path in files, `no file ${path}`);
      return flattenJson(files[path]);
    },
    save_settings: ({ overrides }) => {
      const flat = { ...overrides };
      delete flat["predict_frag.allow_native_predictors"];
      files[SAVED] = nest(flat);
      return SAVED;
    },
    // The engine's rule that matters here (`config.rs`: a sidecar classifier needs its
    // interpreter), so a setup that leaves the interpreter out is refused as the engine would.
    check_settings: ({ overrides }) => {
      const sidecar = ["nn_torch", "mokapot", "entrapment"].includes(overrides["rescore.classifier"]);
      if (sidecar && !overrides["rescore.python"]) {
        throw "invalid config: rescore.classifier=NnTorch requires rescore.python";
      }
      return null;
    },
    search_space: () => ({ plan: { bands: 1, parallel: 1, note: "" }, machine: null }),
    builtin_plan: ({ preset }) => ({ base: preset, note: "", blocked: null, warning: null }),
    diann_library_plan: () => ({ ready: false, diann_version: "DIA-NN 2.2.0" }),
    setups_list: () => ({
      dir: "/profile/setups",
      items: [...store.entries()].map(([id, s]) => ({ id, name: s.name, preset: null, error: null })),
    }),
    setups_save: ({ name, settings, overwrite }) => {
      const id = name.toLowerCase();
      if (store.has(id) && !overwrite) return { item: null, existing: store.get(id).name };
      store.set(id, { name, settings: structuredClone(settings) });
      return { item: { id, name, preset: settings.engine?.preset ?? null, error: null }, existing: null };
    },
    setup_write: ({ path, name, settings }) => {
      files[path] = { format: "mumdia-desktop-settings", version: 1, name, settings: structuredClone(settings) };
      return path;
    },
  });
  const { api } = f;
  for (const [id, v] of Object.entries(htmlDefaults(HTML))) f.el(id).value = v;
  api.state.presets = PRESET_NAMES.map((n) => ({ name: n, path: PRESET_PATH[n] }));
  api.state.engineOk = true;
  await api.initMods();
  await api.initCleavage();
  // The preset the application selects at start.
  f.el("preset").value = PRESET_PATH["diann-library"];
  await api.loadSettings();
  return { ...f, files, store };
}

// The configuration a search started now would hand the engine (`startSearch`): the base it
// derives from and the Search screen's overrides, merged as `settings::derive` merges them.
async function searchConfig(p) {
  const base = p.el("preset").value;
  const bands = await p.api.resolveBands(null);
  const overrides = { ...p.api.runOverrides({ bands, built: null, libraryIrt: null }) };
  const content = base ? p.files[base] : {};
  const derived = merge(content, nest(overrides));
  if (derived.predict_frag) delete derived.predict_frag.allow_native_predictors;
  return { base, text: JSON.stringify(sortKeys(derived)), derived };
}

// What `setup_read` hands back for a file written by `setup_write`.
const asRead = (doc) => ({
  name: doc.name,
  created_by: "MuMDIA 0.12.0",
  created: "2026-11-12T09:30:00Z",
  version: 1,
  newer: false,
  notes: [],
  settings: JSON.parse(JSON.stringify(doc.settings)),
});

async function loadInto(p, doc, origin = { kind: "file", path: "/shared/setup.json" }) {
  const plan = await p.api.reviewSetup(doc, origin);
  assert.ok(plan, `the setup was refused: ${p.el("setups-error").textContent}`);
  p.dialog.ask = async () => true;
  await p.api.applyPendingSetup();
  assert.equal(p.el("setups-error").hidden, true, p.el("setups-error").textContent);
  return plan;
}

test("a setup saved on one profile gives a fresh profile the same engine configuration", async () => {
  const a = await profile();
  const s = a.api.state;
  // The Settings screen: two edits to the preset, this machine's interpreter and a limit
  // sized to this machine.
  s.overrides["rescore.seeds"] = 3;
  delete s.overrides["extract.apex_rt_prior_s"]; // the preset's 120, returned to the default
  s.overrides["rescore.python"] = "C:/envs/rescore/python.exe";
  s.overrides["extract.windows_in_flight"] = 8;
  assert.equal(await a.api.saveSettings(), true);
  assert.equal(a.el("preset").value, SAVED, "the edits are My settings");
  // The Search screen: a phospho search from a FASTA with the built-in predictors.
  a.api.setMode("fasta");
  s.libSrcChosen = true;
  s.libSrc = "builtin";
  for (const [id, v] of Object.entries({
    "d-enzyme": "custom", "d-rule": "K*,R*,!*P", "d-missed": "2", "d-minlen": "8", "d-maxlen": "35",
    "d-minz": "2", "d-maxz": "4", "d-prmin": "400", "d-prmax": "1200.5", "d-maxvar": "3",
    prescreen: "before", "prescreen-strength": "light", bands: "custom", "bands-n": "4",
    "bands-parallel": "2", "norm-scope": "none", "ztscan-hw": "2.1",
  })) a.el(id).value = v;
  s.mods = new Map([
    ["Carbamidomethyl|C", "fix"], ["Phospho|S", "var"], ["Phospho|T", "var"], ["Phospho|Y", "var"],
    ["Acetyl|*|protein_n_term", "var"],
  ]);
  s.runMode = "separate";
  s.ztscan.hwTyped = true;
  s.ztscan.analysisTouched = true;
  s.ztscan.analysisChoice = false;
  // Saved to a file, through its text.
  a.dialog.save = async () => "/shared/Phospho.json";
  a.el("setups-name").value = "Phospho";
  await a.api.exportCurrentSetup();
  assert.equal(a.el("setups-error").hidden, true, a.el("setups-error").textContent);
  const done = a.el("setups-done").textContent;
  assert.match(done, /Not in the setup, as they describe this computer: extract\.windows_in_flight\./);
  assert.match(done, /Saved as "auto" rather than this computer's interpreter path, so the other computer finds its own: rescore\.python\./);
  const file = a.files["/shared/Phospho.json"];
  assert.equal(file.name, "Phospho");
  const st = file.settings;
  assert.deepEqual(st.engine, {
    preset: "diann-library",
    changes: { rescore: { seeds: 3 } },
    reset: ["extract.apex_rt_prior_s"],
  });
  assert.equal(JSON.stringify(st).includes("python.exe"), false, "no path of this machine");
  assert.equal(JSON.stringify(st).includes("windows_in_flight"), false, "no limit of this machine");
  assert.deepEqual(st.digest, {
    enzyme: "custom", cleavage_rule: "K*,R*,!*P", missed_cleavages: 2, min_length: 8, max_length: 35,
    min_charge: 2, max_charge: 4, precursor_mz_min: 400, precursor_mz_max: 1200.5,
    fragment_mz_min: null, fragment_mz_max: null,
  });
  // In the catalogue's order, as the screen writes them (`selectedMods`).
  assert.deepEqual(st.modifications.variable.map((x) => `${x.name}|${x.residue}|${x.position ?? ""}`), [
    "Phospho|S|", "Phospho|T|", "Phospho|Y|", "Acetyl|*|protein_n_term",
  ]);
  assert.deepEqual(st.bands, { mode: "custom", count: 4, at_a_time: 2 });
  assert.deepEqual(st.ztscan, { analysis: false, q1_half_width: 2.1 });
  assert.deepEqual(st.search_space, {
    source: "fasta", library_input: "tables", library_build: "builtin", library_for: "other",
  });

  const b = await profile();
  await loadInto(b, asRead(file));
  assert.equal(b.el("preset").value, SAVED, "changes from the preset are My settings here too");
  assert.equal(b.store.get("phospho")?.name, "Phospho", "kept in My setups");
  // The loaded profile saves the same setup, and its searches get the same configuration
  // but for the left-out settings: this profile's own (the preset's "auto", no limit).
  const again = await b.api.currentSetup();
  assert.deepEqual(sortKeys(host(again.settings)), sortKeys(file.settings));
  const ca = await searchConfig(a);
  const cb = await searchConfig(b);
  assert.equal(cb.derived.rescore.python, "auto");
  assert.equal(ca.derived.rescore.python, "C:/envs/rescore/python.exe");
  assert.equal(cb.derived.extract?.windows_in_flight, undefined);
  delete ca.derived.rescore.python;
  delete cb.derived.rescore.python;
  delete ca.derived.extract.windows_in_flight;
  if (!Object.keys(ca.derived.extract).length) delete ca.derived.extract;
  assert.equal(JSON.stringify(sortKeys(cb.derived)), JSON.stringify(sortKeys(ca.derived)));
  assert.equal(cb.derived.prescreen.score_before_prediction, true);
  assert.equal(cb.derived.groups.window_groups, 4);
  assert.equal(cb.derived.digest.cleavage_rule, "K*,R*,!*P");
  assert.equal(cb.derived.extract?.apex_rt_prior_s, undefined, "the reset preset setting");
});

test("without a machine setting the configuration is identical, and a plain preset stays the preset", async () => {
  // Edits: My settings on both sides, byte for byte.
  const a = await profile();
  a.api.state.overrides["quant.q_threshold"] = 0.05;
  a.api.state.overrides["mbr.strategy"] = "none";
  assert.equal(await a.api.saveSettings(), true);
  a.el("d-missed").value = "0";
  const cur = await a.api.currentSetup();
  const b = await profile();
  await loadInto(b, asRead({ name: "Fast", settings: cur.settings }));
  const [ca, cb] = [await searchConfig(a), await searchConfig(b)];
  assert.equal(cb.base, ca.base);
  assert.equal(JSON.stringify(sortKeys(b.files[SAVED])), JSON.stringify(sortKeys(a.files[SAVED])));
  assert.equal(cb.text, ca.text);

  // Nothing edited: both search with the preset file itself, and nothing is written.
  const c = await profile();
  c.el("preset").value = PRESET_PATH["fasta-sidecars"];
  // What choosing another preset on the Search screen does to the Settings screen.
  await c.api.seedOverridesFromPreset();
  const plain = await c.api.currentSetup();
  assert.deepEqual(host(plain.settings.engine), { preset: "fasta-sidecars", changes: {}, reset: [] });
  const d = await profile();
  await loadInto(d, asRead({ name: "Plain", settings: plain.settings }));
  assert.equal(d.el("preset").value, c.el("preset").value);
  assert.equal(d.calls.some(([n]) => n === "save_settings"), false);
  assert.equal((await searchConfig(d)).text, (await searchConfig(c)).text);
});

test("a ZT Scan setup keeps the user's own search space, which the analysis sets again", async () => {
  const OFF_KEYS = { "extract.q1_fragments": false };
  const zt = (p) => {
    p.answers.scanning_probe = ({ mzml }) => mzml.map((x) => ztProbe(x));
    p.answers.ztscan_off = () => OFF_KEYS;
    p.api.state.picks.mzml = ["/d/REP01.mzML"];
  };
  const a = await profile();
  zt(a);
  a.el("d-maxvar").value = "1";
  await a.api.refreshZtscanNote();
  // The analysis set the measured library; the setup holds what it replaced.
  assert.equal(a.el("d-maxz").value, "4");
  const cur = await a.api.currentSetup();
  assert.equal(cur.settings.digest.max_charge, 3);
  assert.equal(cur.settings.digest.min_length, 7);
  assert.equal(cur.settings.modifications.max_variable, 1);
  assert.deepEqual(host(cur.settings.modifications.variable.map((x) => x.name)), ["Oxidation"]);
  assert.equal(cur.settings.ztscan.analysis, true);

  // Loaded where the same file is selected, the analysis sets its library again.
  const b = await profile();
  zt(b);
  await b.api.refreshZtscanNote();
  b.el("d-maxz").value = "3";
  await loadInto(b, asRead({ name: "ZT", settings: cur.settings }));
  assert.equal(b.el("d-maxz").value, "4");
  assert.equal(b.api.state.mods.get("Acetyl|*|protein_n_term"), "var");
  assert.equal((await searchConfig(b)).text, (await searchConfig(a)).text);
  // Loaded with no ZT Scan file, it is the user's own search space.
  const c = await profile();
  await loadInto(c, asRead({ name: "ZT", settings: cur.settings }));
  assert.equal(c.el("d-maxz").value, "3");
  assert.equal(c.api.state.mods.has("Acetyl|*|protein_n_term"), false);
  // Cleared, the analysis's keys are written off on the loaded profile too.
  a.el("ztscan-analysis").checked = false;
  a.api.state.ztscan.analysisTouched = true;
  a.api.state.ztscan.analysisChoice = false;
  await a.api.refreshZtscanNote();
  const off = await a.api.currentSetup();
  assert.equal(off.settings.ztscan.analysis, false);
  const d = await profile();
  zt(d);
  await d.api.refreshZtscanNote();
  await loadInto(d, asRead({ name: "ZT off", settings: off.settings }));
  assert.equal(d.el("ztscan-analysis").checked, false);
  assert.deepEqual({ ...d.api.ztscanOverrides() }, OFF_KEYS);
  assert.equal((await searchConfig(d)).text, (await searchConfig(a)).text);
});

test("a setup is checked before anything changes: refused values name the field", async () => {
  const p = await profile();
  const good = (await p.api.currentSetup()).settings;
  const before = await searchConfig(p);
  const refuse = async (mutate, why) => {
    const settings = structuredClone(good);
    mutate(settings);
    const plan = await p.api.reviewSetup(asRead({ name: "Bad", settings }), { kind: "file", path: "/x.json" });
    assert.equal(plan, null, why);
    assert.match(p.el("setups-error").textContent, why);
    assert.match(p.el("setups-error").textContent, /nothing was changed/);
    assert.equal(p.el("setups-review").hidden, true);
  };
  await refuse((s) => (s.digest.missed_cleavages = "two"), /settings\.digest\.missed_cleavages \(Missed cleavages\) must be a whole number of 0 or more, not "two"/);
  await refuse((s) => (s.prescreen.mode = "sometimes"), /settings\.prescreen\.mode \(Prescreen\) must be one of "off", "before", "after", "tags"/);
  await refuse((s) => (s.digest.enzyme = "pepsin"), /settings\.digest\.enzyme \(Cleavage\) must be a cleavage this version knows/);
  await refuse((s) => s.modifications.variable.push({ residue: "S", name: "Phosho" }), /settings\.modifications\.variable\[1\]: "Phosho" is not a modification this version knows/);
  await refuse((s) => s.modifications.variable.push({ residue: "s", name: "Phospho" }), /is not a site/);
  await refuse((s) => s.modifications.variable.push({ residue: "C", name: "Carbamidomethyl" }), /Carbamidomethyl \(C\) is both fixed and variable/);
  await refuse((s) => (s.engine.preset = "phospho-magic"), /names "phospho-magic", a preset this version does not offer \(it offers: diann-library, fasta-sidecars, ztscan\)/);
  await refuse((s) => (s.engine.changes = { rescore: { seeds: "three" } }), /settings\.engine\.changes\.rescore\.seeds must be a whole number, not "three"/);
  await refuse((s) => (s.engine.changes = { compete: { group_by: "protein" } }), /settings\.engine\.changes\.compete\.group_by must be one of base_peptide, apex, peptidoform_charge/);
  await refuse((s) => (s.bands = "custom"), /settings\.bands must be an object/);
  await refuse((s) => (s.ztscan.analysis = "yes"), /settings\.ztscan\.analysis \(ZT Scan analysis\) must be true or false/);
  // The engine has the last word on engine settings.
  p.answers.check_settings = () => {
    throw "rescore.seeds must be at least 1";
  };
  await refuse((s) => (s.engine.changes = { rescore: { seeds: 0 } }), /the engine refuses its settings\. rescore\.seeds must be at least 1/);
  // Nothing of any of them reached the screens.
  assert.equal((await searchConfig(p)).text, before.text);
  assert.equal(p.calls.some(([n]) => n === "save_settings"), false);
});

test("settings this version does not know, or that belong to a computer, are listed and left out", async () => {
  const p = await profile();
  p.api.state.overrides["rescore.python"] = "/home/me/env/bin/python";
  await p.api.saveSettings();
  const settings = structuredClone((await p.api.currentSetup()).settings);
  settings.digest.missed_cleavages = 2;
  settings.digest.semi_specific = true;
  settings.colour_scheme = { name: "dark" };
  settings.engine.changes = {
    rescore: { seeds: 2, python: "C:/other/python.exe", new_model: "x" },
    predict_frag: { deeplc_python: "D:/deeplc/python.exe" },
    extract: { windows_in_flight: 16 },
  };
  settings.modifications.fixed[0].note = "always";
  const doc = { ...asRead({ name: "Later", settings }), version: 2, newer: true, notes: ["Written by a newer version of MuMDIA."] };
  const plan = await p.api.reviewSetup(doc, { kind: "file", path: "/x.json" });
  assert.ok(plan, p.el("setups-error").textContent);
  const warn = p.el("setups-review-warn").textContent;
  assert.match(warn, /Written by a newer version/);
  assert.match(warn, /From the newer version and not known to this one, so left out: /);
  for (const key of [
    "settings.digest.semi_specific",
    "settings.modifications.fixed[0].note",
    "settings.colour_scheme",
    "settings.engine.changes.rescore.new_model",
  ]) {
    assert.ok(warn.includes(key), `${key} is not listed: ${warn}`);
  }
  assert.match(warn, /describe a computer rather than a search, so this one keeps its own: extract\.windows_in_flight\./);
  assert.match(warn, /Interpreter paths of another computer, read as "auto", which finds this one's: rescore\.python, predict_frag\.deeplc_python\./);
  assert.match(warn, /This computer's own rescore\.python is kept\./);
  const changes = [...p.api.state.setups.pending.changes];
  assert.ok(changes.includes("Missed cleavages: 1 \u2192 2"), changes.join("\n"));
  assert.ok(changes.includes("rescore.seeds: the engine default (1) \u2192 2"), changes.join("\n"));
  assert.ok(!changes.some((l) => /python|windows_in_flight/.test(l)), changes.join("\n"));
  p.dialog.ask = async () => true;
  await p.api.applyPendingSetup();
  assert.equal(p.api.state.overrides["rescore.python"], "/home/me/env/bin/python");
  assert.equal(p.api.state.overrides["predict_frag.deeplc_python"], "auto");
  assert.equal(p.api.state.overrides["extract.windows_in_flight"], undefined);
  assert.equal(p.api.state.overrides["rescore.seeds"], 2);
  assert.equal(p.el("d-missed").value, "2");
  assert.match(p.el("setups-done").textContent, /Loaded "Later": 2 changes\./);
});

test("saving a setup saves and uses the Settings screen's edits first, and asks before replacing one", async () => {
  const p = await profile();
  p.api.state.overrides["rescore.seeds"] = 3;
  assert.equal(p.el("preset").value, PRESET_PATH["diann-library"], "not yet in use");
  p.el("setups-name").value = "Seeds";
  await p.api.saveNamedSetup();
  assert.equal(p.el("setups-error").hidden, true, p.el("setups-error").textContent);
  assert.equal(p.el("preset").value, SAVED, "the edits are now what the next search uses");
  assert.match(p.el("setups-done").textContent, /saved and are used by the next search/);
  assert.equal(p.store.get("seeds").settings.engine.changes.rescore.seeds, 3);
  // The same name again: asked, and kept when the answer is no.
  p.api.state.overrides["rescore.seeds"] = 5;
  p.dialog.ask = async () => false;
  await p.api.saveNamedSetup();
  assert.equal(p.store.get("seeds").settings.engine.changes.rescore.seeds, 3);
  assert.match(p.el("setups-done").textContent, /Nothing was saved/);
  p.dialog.ask = async () => true;
  await p.api.saveNamedSetup();
  assert.equal(p.store.get("seeds").settings.engine.changes.rescore.seeds, 5);
  // No name, no setup.
  p.el("setups-name").value = " ";
  await p.api.saveNamedSetup();
  assert.match(p.el("setups-error").textContent, /Give the setup a name first/);
  // "Save and use" with nothing changed from the preset uses the preset itself.
  const q = await profile();
  assert.equal(await q.api.saveSettings(), true);
  assert.equal(q.el("preset").value, PRESET_PATH["diann-library"]);
  assert.equal(q.calls.some(([n]) => n === "save_settings"), false);
  assert.match(q.el("settings-saved").textContent, /No setting differs from the preset "diann-library"/);
});

test("Cancel changes nothing, and a setup that matches the screens says so", async () => {
  const p = await profile();
  const cur = (await p.api.currentSetup()).settings;
  const other = structuredClone(cur);
  other.prescreen.mode = "after";
  await p.api.reviewSetup(asRead({ name: "After", settings: other }), { kind: "named", id: "after" });
  assert.equal(p.el("setups-review").hidden, false);
  assert.equal(p.el("setups-keep-row").hidden, true, "a named setup is in My setups already");
  assert.deepEqual([...p.api.state.setups.pending.changes], ["Prescreen: off \u2192 after"]);
  p.api.cancelPendingSetup();
  assert.equal(p.el("setups-review").hidden, true);
  assert.equal(p.el("prescreen").value, "off");
  assert.match(p.el("setups-done").textContent, /Nothing was changed/);
  await p.api.reviewSetup(asRead({ name: "Same", settings: cur }), { kind: "file", path: "/same.json" });
  assert.deepEqual([...p.api.state.setups.pending.changes], []);
  assert.equal(p.el("setups-keep-row").hidden, false);
  // A partial file sets only what it holds.
  await loadInto(p, asRead({ name: "Partial", settings: { prescreen: { mode: "tags" } } }));
  assert.equal(p.el("prescreen").value, "tags");
  assert.equal(p.el("d-missed").value, "1");
});

test("an interpreter is carried as auto, so a setup that needs one still runs elsewhere", async () => {
  // Started from the engine defaults, which name no interpreter: the neural rescorer needs
  // `rescore.python`, and this computer's path is not the other's.
  const a = await profile();
  a.el("preset").value = "";
  await a.api.seedOverridesFromPreset();
  a.api.state.overrides["rescore.classifier"] = "nn_torch";
  a.api.state.overrides["rescore.python"] = "C:/envs/rescore/python.exe";
  assert.equal(await a.api.saveSettings(), true);
  const cur = await a.api.currentSetup();
  assert.deepEqual(host(cur.settings.engine), {
    preset: null,
    changes: { rescore: { classifier: "nn_torch", python: "auto" } },
    reset: [],
  });
  assert.deepEqual([...cur.paths], ["rescore.python"]);
  // A fresh profile: accepted by the engine, and it finds its own interpreter.
  const b = await profile();
  await loadInto(b, asRead({ name: "NN", settings: cur.settings }));
  assert.equal(b.api.state.overrides["rescore.python"], "auto");
  assert.equal(b.el("preset").value, SAVED);
  // A profile that named its own interpreter keeps it.
  const c = await profile();
  c.el("preset").value = "";
  await c.api.seedOverridesFromPreset();
  c.api.state.overrides["rescore.python"] = "/opt/mumdia/env/bin/python";
  await c.api.saveSettings();
  await loadInto(c, asRead({ name: "NN", settings: cur.settings }));
  assert.equal(c.api.state.overrides["rescore.python"], "/opt/mumdia/env/bin/python");
  // Without the interpreter the engine refuses it, and so does the load.
  const d = await profile();
  const bare = structuredClone(cur.settings);
  delete bare.engine.changes.rescore.python;
  d.el("preset").value = "";
  await d.api.seedOverridesFromPreset();
  const refused = await d.api.reviewSetup(asRead({ name: "Bare", settings: bare }), { kind: "file", path: "/b.json" });
  assert.equal(refused, null);
  assert.match(d.el("setups-error").textContent, /requires rescore\.python/);
});

test("a setup's choices are the Search screen's choices", () => {
  const { api } = load();
  const radios = (name) =>
    [...HTML.matchAll(new RegExp(`<input[^>]*name="${name}"[^>]*>`, "g"))].map((m) => /value="([^"]*)"/.exec(m[0])[1]);
  const tabs = [...HTML.matchAll(/data-mode="([^"]+)"/g)].map((m) => m[1]);
  const ids = new Set([...HTML.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
  for (const e of api.SETUP_SPEC) {
    if (e.select && e.kind === "choice") assert.deepEqual([...e.values].sort(), selectOptions(HTML, e.select).sort(), e.path);
    if (e.select) assert.ok(ids.has(e.select), `${e.path}: no ${e.select}`);
    if (e.radio) assert.deepEqual([...e.values].sort(), radios(e.radio).sort(), e.path);
    if (e.tabs) assert.deepEqual([...e.values].sort(), [...tabs].sort(), e.path);
    if (e.id) assert.ok(ids.has(e.id), `${e.path}: no ${e.id}`);
    if (e.kind === "choice") assert.ok(e.select || e.radio || e.tabs, `${e.path} names no control`);
  }
  // A machine's own settings are settings the engine has.
  const paths = new Set(SCHEMA.fields.map((f) => f.path));
  for (const k of [...api.LOCAL_SETTINGS, ...api.INTERPRETER_SETTINGS]) {
    assert.ok(paths.has(k), `${k} is not an engine setting`);
  }
  for (const k of api.INTERPRETER_SETTINGS) assert.ok(!api.LOCAL_SETTINGS.has(k), k);
});

let failed = 0;
for (const [name, fn] of tests) {
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${name}\n${e && e.stack ? e.stack : e}`);
  }
}
if (failed) {
  console.log(`${failed} of ${tests.length} desktop UI tests failed`);
  process.exit(1);
}
console.log(`desktop UI tests ok: ${tests.length}`);
