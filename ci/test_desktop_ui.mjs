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
  const answers = { components_status: () => status() };
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
