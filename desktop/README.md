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
settings editor, revisit that decision then: it is much easier to add a bundler
later than to remove one.

## Analysis components

The application installs its own Python environment with `uv`, so conda is never
needed. It goes under the per-user data directory (`%LOCALAPPDATA%\MuMDIA` or
`~/.local/share/MuMDIA`), not beside the executable, because on Windows that is
Program Files and an installer that needs administrator rights on first run is not
an easy install.

**Searching without the components is refused.** MuMDIA does run with no Python at
all, but the recorded numbers make that a bad default to offer: on the same file the
fully native FASTA path returns about 1,213 report rows against about 10,300 for the
imported-library workflow with DeepLC and neural rescoring. The refusal predicate is
narrow on purpose -- "this configuration requires no sidecar at all", asked of
`mumdia doctor --json` rather than kept as a list here. Refusing anything mentioning
`native_tda` would be wrong: on an imported library it measured 10,847 against
`nn_torch`'s 10,914.

### Two environments, not one

MS2PIP cannot share an environment with DeepLC at the versions this project tests:

    deeplc==4.1.1  -> psm-utils>=1.5 -> sqlalchemy>=2
    ms2pip==4.0.0  ->                   sqlalchemy>=1.3,<2

`uv` reports the pair as unsatisfiable. `ms2pip>=4.1` does resolve alongside DeepLC,
but MS2PIP's version changes predicted fragment intensities, and
`env/docker-rescore.yml` pins 4.0.0 deliberately as "a separate, testable upgrade".
So the primary environment covers rescoring, DeepLC and match-between-runs, which is
the whole recommended workflow, and MS2PIP gets its own, installed on request and
needed only for FASTA-mode library building with predicted intensities.

## Settings

The editor is generated from `configs/config-schema.json`, which
`ci/gen_config_reference.py` emits from the same parse of `config.rs` that produces
the reference document, staleness-checked in CI beside it. Nothing about a setting
is written in the interface, so it cannot describe a parameter the engine does not
have.

Saving writes only the difference from the defaults. `Config` is
`deny_unknown_fields` with serde defaults, so that is a valid configuration, and it
means a later release that improves a default still reaches someone who saved
settings today. Every save is validated by the engine before it is offered for use.

## Testing

    cargo test --lib                 # unit tests, no engine needed

    # end to end, against a real engine and the fixture ci/smoke.sh generates
    MUMDIA_BIN=... MUMDIA_TEST_MZML=... MUMDIA_TEST_FASTA=... cargo test

    # the real component installation; downloads several hundred megabytes
    MUMDIA_TEST_INSTALL=1 cargo test the_primary_environment

## Packaging

`cargo tauri build` produces an `.msi` on Windows and an `.AppImage` on Linux. Only
two files are shipped as Tauri resources, both executables: the engine and `uv`, in
`binaries/` beside the application.

Everything else the application needs from the repository is compiled in with
`include_str!`: the settings schema, and the two requirement sets. That is both
simpler and more robust than shipping them as files, and it costs nothing in
freshness, because all three are generated from sources that require a rebuild
anyway. It also avoids two Tauri packaging traps found while building the first
installer:

- in the resource LIST form, a `..` source keeps its shape, so `"../../configs/*"`
  installs to `<install>/_up_/_up_/configs/`, where nothing looks for it. This was
  read out of the generated WiX source, not guessed;
- in the resource MAP form, which does let a destination be named, a `..` source
  fails the build outright with `Access is denied`.

Verified on Windows: a 29 MB installer containing `mumdia-console.exe` with
`binaries/mumdia.exe` and `binaries/uv.exe` beside it, and no `_up_` directory.

### The Linux engine is a GNU build, not musl

The engine's own release archives are musl and stay musl. Inside an AppImage they do
not survive: `linuxdeploy` runs `patchelf` over every ELF binary it bundles, and a
static-pie musl binary comes out with `RUNPATH [$ORIGIN]` injected and segfaults
immediately. Verified by extracting a built AppImage and running the engine inside
it; `uv`, dynamically linked, survived the same treatment untouched.

Nothing is lost. musl would buy portability if the bundle had no other glibc floor,
but the Tauri host links WebKitGTK and sets that floor regardless.

### Where the engine actually sits in each bundle

    MSI       <install>/mumdia-console.exe
              <install>/binaries/mumdia.exe

    AppImage  usr/bin/mumdia-console
              usr/lib/MuMDIA/binaries/mumdia

Note that the AppImage does NOT put the engine beside the executable. That is why
the lookup asks Tauri for the resource directory first; without it the application
would search `usr/bin/` and report that it cannot find its own engine.

## What is not here yet

Nobody has clicked through the interface. The backend it drives is covered by tests,
and both bundles have been built and inspected, but the buttons themselves rest on
inspection.
