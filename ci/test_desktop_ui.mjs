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
    setAttribute: (k, v) => attrs.set(k, v),
    removeAttribute: (k) => attrs.delete(k),
    hasAttribute: (k) => attrs.has(k),
    addEventListener() {},
    appendChild() {},
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
  // settings change (the ZT Scan recipe sets them).
  const answers = {
    components_status: () => status(),
    diann_status: () => ({ status: { runs: true, licence_acknowledged: true } }),
  };
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
        dialog: {},
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
  ];
  const exported = names.map((n) => `${n}: typeof ${n} === "undefined" ? undefined : ${n}`);
  const api = vm.runInContext(`${body}\n;({ ${exported.join(", ")} });`, ctx);
  return { api, el: (id) => document.getElementById(id), timers, calls, answers };
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

const RECIPE = {
  "convert.isolation_half_width": 1.9,
  "extract.q1_fragments": true,
  "quant.q1_cycle_halfwidth": 4,
};

test("a ZT Scan mzML is explained, its half-width filled and the recipe offered", async () => {
  const f = load();
  const { state } = f.api;
  f.answers.scanning_probe = ({ mzml }) => mzml.map((p) => ztProbe(p));
  f.answers.ztscan_recipe = () => RECIPE;
  state.picks.mzml = ["/d/REP01.mzML", "/d/REP02.mzML"];
  await f.api.refreshZtscanNote();
  assert.equal(f.el("ztscan-note").hidden, false);
  assert.match(f.el("ztscan-body").textContent, /2 of the 2 selected files are a scanning-quadrupole/);
  assert.match(f.el("ztscan-body").textContent, /663 positions from 394\.3 to 898\.3 m\/z/);
  assert.equal(f.el("ztscan-hw").value, "1.9");
  assert.equal(f.el("ztscan-recipe").disabled, false);
  // The recipe is ticked for ZT Scan files until the user changes it: the recipe and the
  // confirmed width.
  assert.equal(f.el("ztscan-recipe").checked, true);
  assert.deepEqual({ ...f.api.ztscanOverrides() }, { ...RECIPE, "convert.isolation_half_width": 1.9 });
  // What the checkbox listener records when the user clears it: the half-width alone, and a
  // re-render keeps their choice.
  f.el("ztscan-recipe").checked = false;
  state.ztscan.recipeTouched = true;
  await f.api.refreshZtscanNote();
  assert.equal(f.el("ztscan-recipe").checked, false);
  assert.deepEqual({ ...f.api.ztscanOverrides() }, { "convert.isolation_half_width": 1.9 });
  f.el("ztscan-recipe").checked = true;
  // What the input listener records when the user types.
  f.el("ztscan-hw").value = "2.1";
  state.ztscan.hwTyped = true;
  assert.deepEqual({ ...f.api.ztscanOverrides() }, { ...RECIPE, "convert.isolation_half_width": 2.1 });
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

test("the ZT Scan recipe is held back for a mixed selection, and other files need nothing", async () => {
  const f = load();
  const { state } = f.api;
  f.answers.scanning_probe = ({ mzml }) =>
    mzml.map((p) =>
      p.includes("orbi")
        ? ztProbe(p, { convert_refuses: false, scanning_quadrupole: false, target_only: 0 })
        : ztProbe(p)
    );
  f.answers.ztscan_recipe = () => RECIPE;
  // A ZT Scan file with an Orbitrap file and a vendor file: the half-width only, no recipe,
  // and the vendor file is not probed.
  state.picks.mzml = ["/d/REP01.mzML", "/d/orbi.mzML", "/d/run.raw"];
  await f.api.refreshZtscanNote();
  const asked = f.calls.filter(([n]) => n === "scanning_probe").flatMap(([, a]) => a.mzml);
  assert.deepEqual([...asked], ["/d/REP01.mzML", "/d/orbi.mzML"]);
  assert.equal(f.el("ztscan-note").hidden, false);
  assert.equal(f.el("ztscan-recipe").disabled, true);
  f.el("ztscan-recipe").checked = true;
  assert.deepEqual({ ...f.api.ztscanOverrides() }, { "convert.isolation_half_width": 1.9 });
  // Only files with offsets: nothing to say and nothing added.
  state.picks.mzml = ["/d/orbi.mzML"];
  await f.api.refreshZtscanNote();
  assert.equal(f.el("ztscan-note").hidden, true);
  assert.deepEqual({ ...f.api.ztscanOverrides() }, {});
  assert.ok(![...f.api.searchProblems()].some((s) => /Q1 half-width/.test(s)));
});

test("the ZT Scan recipe sets the library's search space and takes back what it set", async () => {
  const f = load();
  const { state } = f.api;
  f.answers.scanning_probe = ({ mzml }) =>
    mzml.map((p) =>
      p.includes("orbi")
        ? ztProbe(p, { convert_refuses: false, scanning_quadrupole: false, target_only: 0 })
        : ztProbe(p)
    );
  f.answers.ztscan_recipe = () => RECIPE;
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
  // A ZT Scan file: the recipe is ticked, and with it the measured search space, the
  // precursors inside the swept range (394.256 - 1.9 to 898.271 + 1.9, rounded outward).
  state.picks.mzml = ["/d/REP01.mzML"];
  await f.api.refreshZtscanNote();
  assert.equal(f.el("ztscan-recipe").checked, true);
  assert.deepEqual(fields(), {
    "d-minlen": "6", "d-maxlen": "40", "d-minz": "1", "d-maxz": "4", "d-maxvar": "2", "d-prmin": "392", "d-prmax": "901",
  });
  assert.equal(state.mods.get(acetyl), "var");
  assert.equal(state.mods.get("Oxidation|M"), "var");
  assert.equal(state.mods.get("Carbamidomethyl|C"), "fix");
  assert.match(f.el("ztscan-library-note").textContent, /charges 1 to 4, lengths 6 to 40/);
  assert.match(f.el("ztscan-library-note").textContent, /from 392 to 901 m\/z/);
  // The range follows the half-width while the recipe set it.
  f.el("ztscan-hw").value = "3.0";
  state.ztscan.hwTyped = true;
  await f.api.refreshZtscanNote();
  assert.equal(f.el("d-prmin").value, "391");
  assert.equal(f.el("d-prmax").value, "902");
  // A value the user edits is theirs: it is neither replaced nor taken back.
  f.el("d-maxlen").value = "35";
  await f.api.refreshZtscanNote();
  assert.equal(f.el("d-maxlen").value, "35");
  // Clearing the recipe takes back the rest, and the modification it added.
  f.el("ztscan-recipe").checked = false;
  state.ztscan.recipeTouched = true;
  await f.api.refreshZtscanNote();
  assert.deepEqual(fields(), { ...start, "d-maxlen": "35" });
  assert.equal(state.mods.has(acetyl), false);
  assert.equal(state.mods.get("Oxidation|M"), "var");
  assert.equal(f.el("ztscan-library-note").textContent, "");
  // Ticking it again applies it again; removing the ZT Scan file takes it back.
  f.el("ztscan-recipe").checked = true;
  await f.api.refreshZtscanNote();
  assert.equal(f.el("d-minz").value, "1");
  assert.equal(state.mods.get(acetyl), "var");
  state.picks.mzml = ["/d/orbi.mzML"];
  await f.api.refreshZtscanNote();
  assert.equal(f.el("d-minz").value, "2");
  assert.equal(state.mods.has(acetyl), false);
  // A modification the user had chosen before is left alone when the recipe goes.
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
  f.answers.ztscan_recipe = () => RECIPE;
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
