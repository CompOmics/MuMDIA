# MuMDIA Console

Desktop interface for the MuMDIA search engine, for people who should not have to use
a terminal. Windows and Linux.

This directory is a **separate Cargo workspace**. It is not a member of
`rust/mumdia`'s workspace and does not depend on the `mumdia` crate, so
`cargo test --workspace` in the engine never compiles a webview.

## Running it during development

```bash
cd desktop/src-tauri
cargo run
```

The application needs an engine binary. It looks in this order:

1. `$MUMDIA_BIN`, if set;
2. beside its own executable, and in a `binaries/` subdirectory beside it (this is
   where a release bundle puts it);
3. `rust/mumdia/target/release/mumdia` relative to a `cargo run` build, which is the
   convenient case while developing;
4. anything named `mumdia` on `PATH`.

It runs `--version` on whatever it finds at startup, so a binary that exists but
cannot execute fails immediately rather than an hour into a search.

To point it at a specific build:

```bash
MUMDIA_BIN=/path/to/mumdia cargo run          # Linux
$env:MUMDIA_BIN = "C:\path\to\mumdia.exe"; cargo run   # Windows PowerShell
```

## Why the engine is a subprocess and not a linked library

The engine is a library crate, so linking it looks attractive. It is the wrong
choice, for one decisive reason: **the engine installs no signal handler anywhere**,
so stopping a run is a kill, and a Rust thread cannot be killed. Linked in-process
there would be no Stop button at all.

Two supporting reasons. A stage panic would take the whole application down rather
than ending one run — and the engine does still panic on some malformed input. And
rayon's global pool can only be built once per process, so `--threads` could not
change between runs.

The cost is that the application must resolve a path and manage a process tree. That
is `src/engine.rs` and `src/run.rs`.

## Process control

The tree is three deep: application, engine, and the Python workers the engine
spawns. Killing only the engine orphans a worker that may hold tens of gigabytes.

- **Linux**: the engine is spawned into a new process group, and cancelling signals
  the group (`TERM`, then `KILL`).
- **Windows**: `taskkill /T /F` walks the tree at kill time.

A hard kill skips destructors, so the engine's atomic-write layer never removes its
`.tmp-<pid>` files. Cancelling therefore sweeps them from the output directory, or the
next run would start in a dirty folder.

Closing the window cancels every running search, for the same reason.

## How progress works

No log parsing. Every engine stage writes `<artifact>.report.json` beside its output,
carrying the producing stage, row count, elapsed time and per-stage statistics. The
application polls the output directory and folds those into one row per stage.

The results panel is read from `psms_scored.parquet.report.json`, which records the
classifier that **actually** ran alongside the one requested. Those differ when a
sidecar fails and `rescore.strict` is false, and the interface says so rather than
echoing the request.

## Frontend

Plain ES modules, no framework and no build step, so the release pipeline needs no
Node. `ui/` is served as static files. If this grows to include the generated
settings editor, revisit that decision then — it is much easier to add a bundler
later than to remove one.

## What is not here yet

This is milestone 1 of the plan: it runs a search and shows the result. Component
installation (bundled `uv`), the generated settings editor, pre-flight disk and
peak-cap checks, and the release bundles are milestones 2 to 4.
