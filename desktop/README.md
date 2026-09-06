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

## DIA-NN

The interface can predict a spectral library from a FASTA using DIA-NN, which is
worth doing because an imported library identifies far more peptides than digesting
a FASTA with the built-in predictors.

It gets DIA-NN one of two ways, and neither distributes anything. It **locates** a
copy the user installed, or it **fetches** the pinned 1.8.1 release from the
vendor's own URL onto the user's machine. The difference from the Python components
above is a licence constraint, not a style choice: DeepLC, torch and mokapot are
open-source PyPI packages whose licences permit redistribution, so `uv` can place
them anywhere convenient. DIA-NN is closed source, absent from PyPI and Bioconda,
forbids redistribution from 1.9 onward, and from 1.9.2 requires a licence file to
activate.

1.8.1 gets special treatment because it is the last release predating activation.
It is **not**, however, freely redistributable, despite that being widely repeated:
its own `LICENSE.txt` bars derivative works and bars renting, leasing, lending and
sublicensing, permitting only a one-time permanent transfer of all rights. The
claim traces to community container images, not to the licence text. So MuMDIA does
not bundle it. It downloads it from the vendor's release URL, on the user's
machine, which distributes nothing: the vendor distributes, the user obtains, and
the application automates a download the user could do by hand.

That is a narrow judgement and not a licence to bundle. A mirror, a vendored copy,
a `latest` lookup or any other host voids the reasoning that permits it, which is
what `the_download_is_pinned_to_one_version_from_the_vendors_own_host` exists to
catch.

Five things hold the boundary, four of them enforced in `diann.rs` rather than only
in the interface:

- `diann::build` and `diann::install` both refuse until the licence notice is
  acknowledged, so calling either command directly does not bypass it. The Academia
  edition is non-profit-only, and that is a restriction a commercial user can
  breach without ever noticing it exists.
- The URL and SHA-256 are pinned per platform, and the digest is verified while
  streaming. A mismatch deletes the file: these bytes are executed or handed to the
  operating system's installer, and a failed verification must not leave something
  runnable behind for someone to double-click.
- On Windows the vendor's **own installer** is launched rather than silently
  unpacked, so the licence screen the user accepts is DIA-NN's, not a paraphrase of
  it in our dialog. The interface says so and then gets out of the way.
- Detection reports a binary as usable only after **running** it and reading the
  `DIA-NN ...` banner out of its output. A file that exists but cannot execute -- a
  Linux binary on Windows, a truncated download, a 1.9.2+ build with no licence
  file -- must not read as ready, or the failure surfaces at the point of use.
- A path the user chose wins over `MUMDIA_DIANN`, PATH and the installer
  directories, and if it stops working that is reported rather than silently
  replaced. DIA-NN's version changes the library it predicts, so a silent switch is
  a silent change of results.

### The Linux tarball is flat

`diann_1.8.1.tar.gz` has no enclosing directory: `diann-1.8.1` sits beside
`libtorch_cpu.so`, `libc10.so`, `libtimsdata.so` and `unimod.obo`. Two consequences,
both handled and both easy to reintroduce. It must be extracted into a directory of
its own, or it scatters half a gigabyte of shared libraries into whatever it was
unpacked in. And the binary cannot find those libraries unless its own directory is
on the loader path, which is what `diann_command` adds; the Python workers
deliberately do not go through it, or a virtualenv's `bin` would end up on
`LD_LIBRARY_PATH` for no reason.

Note the size: 142 MB compressed, about 490 MB extracted, nearly all of it
`libtorch_cpu.so`. `offer()` reports both figures, because reporting only the
download would understate the disk cost threefold.

### Building the library from the search screen

The digest parameters live on the Search screen, beside the control that consumes
them. They used to sit on Setup in a separate "Predict a library" card while the
radio that used them was here, so invisible state on a screen the user need never
have opened decided the search space, the cache key, and whether the "already built"
note was true; the `if (!el) return dflt` guards made every mismatch silent. That
card is gone and this is the only way to build a library, which also removed the
branch that silently overwrote a manually chosen library pair on every visit to
Setup.

FASTA mode offers two ways to get a library: the engine's built-in predictors, or
DIA-NN predicting one first. The second is the sensitive path -- the ~1,213 against
~10,300 figure is largely this difference -- so a user starting from a FASTA should be
able to take it without first understanding that a library is a separate artifact
built on another screen.

**A predicted library is cached, content-addressed.** It depends on the FASTA's bytes
and the digest parameters and on nothing else: not the mzML, not the thread count. So
`library_cache_dir` keys it on a digest of the FASTA plus every parameter that changes
the output plus the DIA-NN version, and the search reuses it. Without that, offering
this on the search screen would promise a whole-proteome prediction on every run. The
DIA-NN version is in the key because its version changes what it predicts, and reusing
a library across versions would silently change results. The thread count is
deliberately out, because including it would miss the cache for nothing.

`cached_library` requires **both** tables. A build interrupted between writing them
would otherwise read as a usable cache entry and the search would fail on a missing
file after the interface had said it was reusing a library.

**The chain lives in the frontend, deliberately.** `start_run` is the one path with
end-to-end tests and it stays untouched: by the time it is called, this is an ordinary
library-mode search. The cost is that a webview reload during the build loses the
chain. The cache is what makes that acceptable -- press Start again and the library is
already there -- and moving the chain into `run.rs` would mean a second state machine
in the code that actually spawns searches.

### What it runs

The three steps documented in the top-level `README.md`, in order: DIA-NN predicts,
`import_diann_lib.py` maps the result into the MuMDIA schema, and
`make_reverse_decoys.py` adds the decoy population. The last also sorts by precursor
m/z and re-indexes `candidate_id`, both of which the fragment index rejects a
library for lacking. On success the two tables are selected on the search screen
automatically, because the alternative is retyping two long paths.

MuMDIA reads only the library. Invoking DIA-NN is not reading its source, so the
clean-room boundary is unchanged.

## Several files

The Spectra picker takes a list. One file is an ordinary search; several mean one of
two different analyses, and the interface makes you choose because conflating them
would be a scientific error rather than a UI simplification.

**Search each separately** runs one `run` per file into its own numbered subfolder,
queued in the frontend. Sequential rather than parallel on purpose: a search already
saturates the machine, and two at once would compete for cores and memory while making
the progress display meaningless. The queue lives here so `start_run` stays the single
tested path and each run is an ordinary one; the cost is that closing the window ends
the queue, which the per-file folders make recoverable. The results screen then shows a
per-file breakdown and says the counts are not additive, because these runs share no
FDR estimate and a peptide found in three files is counted three times.

**One experiment** sends every file to `run-experiment`: one pooled rescore, optional
MBR, per-run quant and cross-run LFQ. Three consequences the interface has to state
rather than let a user infer:

- the counts are **experiment-wide**, not per file. The grouped q columns are grouped
  across the whole experiment, so dividing by the number of runs does not give a
  per-file number; the per-file unit is `run_psm_q` in the split tables.
- it writes **no report stage**, so there is no `peptides.tsv` or `proteins.tsv`
  anywhere in the tree. `read_results` falls back to `scored_combined.parquet` and
  flags the result `experiment_wide`, which is what drives the banner.
- per-run artifacts go to `<out_dir>/<name>/`, so the stage list looks sparse until
  the pooled phase.

A DIA-NN-predicted library is built **once** for the whole selection, not per file:
the library depends on the FASTA and the digest parameters, not on the spectra.

The backend refuses two things rather than guessing: a pooled experiment with fewer
than two files, and the same path selected twice (which would search one file twice
and pool the result with itself, inflating the evidence for those peptides).

## Vendor formats

The Search screen accepts vendor formats as well as mzML, because a user who has no
mzML is exactly the user this application is for. The engine does the conversion
(`raw.rs`); the application installs what it can and locates the rest.

**Two pickers, not one.** Three of the five vendor formats are directories: Bruker
and Agilent `.d`, and Waters `.raw`. A file dialog cannot select a directory, so
Spectra has both "Choose file..." and "Choose folder...", and both write the same
state slot.

**Thermo is installed; everything else is located.** ThermoRawFileParser is
Apache-2.0 and from CompOmics, so this is an ordinary managed component and the
contrast with `diann.rs` is the point: no licence notice, no acknowledgement gate,
no installer hand-off. Press Install. The URL and SHA-256 are still pinned and
verified, because the bytes get executed, but that is a supply-chain measure rather
than a licence one.

ProteoWizard `msconvert` covers Bruker, SCIEX, Agilent and Waters and is **not**
installed, because its vendor readers bundle each instrument maker's own libraries
under those makers' terms, which the user accepts when they obtain ProteoWizard. The
Setup screen shows whether one was found and offers a link to get it; `open_url`
is an allowlist of three project URLs rather than a general opener, because a
shell-adjacent opener reachable from the webview is how a link becomes a command.

The 2.0.0 self-contained builds are used rather than the much smaller 1.4.5 zip.
1.4.5 is a managed .NET Framework build needing Mono on Linux, and "install Mono
first" is the step that loses the user this feature exists for. The self-contained
builds carry their own runtime, at about 50 MB. A 1.4.x install already on the
machine still works: the engine runs a managed `.exe` under Mono.

Two things worth knowing:

- **The peak census is skipped for any vendor format.** `peak-census` would convert
  the whole file first, which is minutes of apparent hang the moment someone picks a
  file. The note under the picker says conversion happens at search time and that
  peak statistics arrive then. This is a deliberate gap: the users most likely to
  need advice on `--top-peaks-ms2` are the ones least likely to have an mzML, and
  they get it during the run rather than before it.
- **Preflight blocks a vendor format whose converter is missing**, naming which
  converter, rather than letting the engine fail after the interface has switched to
  the progress screen.
- **Bruker gets an ion-mobility warning** on the Setup screen and under the picker.
  MuMDIA's pipeline is 3D, so diaPASEF loses the separation that makes it selective.
  Saying so is the difference between a user reading a low count as a MuMDIA result
  and reading it as the cost of a discarded dimension.

`thermo::needs` and `thermo::label` duplicate the engine's `raw::detect` rather than
importing it, because the application spawns the engine binary and does not depend on
its crate. `vendor_detection_matches_the_engines_own_rule` asserts the two agree; if
they drift, the interface either blocks a file the engine would convert or admits one
it will not. The file-versus-directory question is answered in the backend, through
`vendor_of`, because the webview cannot stat a path.

## Settings

The editor is generated from `configs/config-schema.json`, which
`ci/gen_config_reference.py` emits from the same parse of `config.rs` that produces
the reference document, staleness-checked in CI beside it. Nothing about a setting
is written in the interface, so it cannot describe a parameter the engine does not
have.

### The example configurations are compiled in

`configs/examples/*.json` are `include_str!`-ed into the binary and written to the
per-user data directory on demand, the same treatment the settings schema and the
requirement files get, and for the same reason: they live outside `src-tauri` and a
Tauri resource path containing `..` does not work (the list form produces a literal
`_up_` directory, the map form fails with "Access is denied").

This is not a packaging nicety. `preflight` refuses a configuration that needs no
Python sidecar, and the engine's own defaults are exactly that configuration, so a
run with no `--config` is always blocked. With no presets to offer, the packaged
application could not start anything: the blocker told the user to choose a preset
that uses retention-time modelling while the list was empty. A copy beside the
executable, or in the repository during development, still wins over the compiled-in
ones.

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

    # the process-tree kill. NOT run in CI, and not on a machine you share
    MUMDIA_TEST_KILL=1 cargo test kill_tree

`MUMDIA_TEST_KILL` is opt-in because that test terminated a GitHub runner twice. The
first time is explained: the group kill had no guard and could signal the runner's
own process group. The second time it did it again with the guard in place, which
should have permitted a group signal only for a child verifiably in its own group,
and that is not accounted for. The gating follows from not knowing rather than from
a diagnosis.

The consequence, stated plainly: the Unix group-kill path in `kill_tree` is covered
by nothing automated. The guard's decision is tested without acting on it, and the
kill itself is verified on Windows, where `taskkill /T` addresses a process tree
rather than a group.

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

The DIA-NN path is the least exercised part of that. No DIA-NN is installed on the
development machine, so detection, the licence gate, the argument construction and
the Parquet search are unit-tested, and the three-step build has never been run end
to end against a real DIA-NN. What it runs is the command line from `README.md`,
which has been run manually, but that is an argument from equivalence and not a
test.

The vendor-format path is partly verified now. A real 887 MB Thermo `.raw`
(236,041 scans) was converted end to end through **both** converters, and the engine
read both results to identical counts (`ms1=7 ms2=493 windows=151`): 191 s through
ThermoRawFileParser, 139 s through msconvert. So the conversion machinery, argument
construction and reuse logic are exercised, not just unit-tested.

What is **not** verified is the four msconvert-only formats. Bruker `.d`, SCIEX
`.wiff`, Agilent `.d` and Waters `.raw` have no test file on this machine, so their
vendor-specific readers, the directory-input handling and
`--combineIonMobilitySpectra` have never run. The dispatch that routes to them is
unit-tested; the conversions themselves are not.

The application's own install and detect paths remain unclicked: the zip download,
extraction (including a path-traversal refusal) and probe are unit-tested, but nobody
has pressed Install.

The download is tested where it can be. Streaming, hashing, digest verification and
the delete-on-mismatch path all run against a loopback HTTP server in the unit
tests, and the pinned URLs, sizes and SHA-256 digests were taken from the real
release assets on 2026-08-30 by downloading and hashing both. What has not run is
the last mile on either platform: the Windows installer hand-off, and the Linux
extract-then-probe. Both are short, and both are untested.
