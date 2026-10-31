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

The terminal state is published in one place. `cancel` records the intent and kills
the tree; the thread that reaps the engine reads that intent and publishes `cancelled`,
`done` (the engine finished before the kill landed, so its outputs are complete) or
`failed`. Until then the status stays `running` with `cancel_requested` set and the
interface shows "Stopping". Two writers used to race here, and a stopped run could be
shown as failed with the last log line as its error (docs/29 #14).

## Output ownership

Two engines writing one results folder interleave their artifacts with no error from
either. A run reserves its results folder before the engine is spawned, by canonical
path so that spellings and, on Windows, case name one folder, and releases it when its
end is published; a second Start into an active folder is refused with the owning run
named. The frontend also refuses to start while a Start is in progress or while the run
it follows is still running, because it can show and stop only one (docs/29 #5).

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

**Installed is tested by importing, not by finding.** The routine status check asks
`importlib.util.find_spec`, which is cheap enough to run on every refresh because it
only locates a package; it never loads a native library. That let an environment whose
torch could not initialise its DLLs (a Windows machine without a current Microsoft
Visual C++ runtime) show every component green, and the first search stopped two
minutes in with `OSError: [WinError 1114] ... torch\lib\c10.dll`. So `loadcheck`
imports the packages for real, in a subprocess bounded at 120 s, `deeplc` first as the
workers do and with one torch operation: when an installation finishes, and the first
time a session finds an environment already installed (in the background; the card
reads "testing..." meanwhile and a search waits for it). An environment that fails is
marked "cannot load", the installation log shows the end of the traceback, the banner
says what to do (for a torch DLL failure: install the runtime from
https://aka.ms/vs/17/release/vc_redist.x64.exe, reboot, check for an older
`msvcp140.dll` on `PATH` with `where msvcp140.dll`), and **Re-check** runs the import
again. The result is kept for the session and per set of package versions, so an
upgrade is tested again. The application's own `doctor` calls, which ask only which
roles a configuration needs and which converters exist, set
`MUMDIA_DOCTOR_IMPORT_CHECK=0` so they do not repeat that import.

### The Visual C++ runtime

PyTorch's Windows wheels need the Microsoft Visual C++ 2015-2022 x64 runtime
(`vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll`) at 14.40 or newer, and a
fresh Windows does not always have one; without it the real import above fails on
`torch\lib\c10.dll`. Two things provide it.

**The installer carries it.** The release MSI merges Microsoft's merge module for the
runtime (`Microsoft_VC14x_CRT_x64.msm`) from the Visual Studio installation on the
build runner (`src-tauri/wix/vc-runtime.wxs`, added by `tauri.vcruntime.conf.json`).
Installing MuMDIA therefore installs the runtime into `System32` with Windows
Installer's own rules: no network at install time, and a file is never replaced by an
older one, so a machine that already has a newer runtime keeps it and the
installation succeeds. The runtime files are shared, reference-counted components:
uninstalling MuMDIA leaves them for the other programs that installed them too. The
options considered, and why this one:

- **the merge module (chosen)**: native to the MSI that Tauri already builds, works
  offline, and needs no custom action. Microsoft describes merge modules for the
  runtime as deprecated in favour of the redistributable installer, but still ships
  and supports them; the cost of the choice is that a runtime installed this way is
  updated by installing a newer MuMDIA or the redistributable, not on its own;
- **running `vc_redist.x64.exe` from a custom action**: it is itself a Windows
  Installer package, and Windows Installer runs one installation at a time, so
  launched from inside the MSI's own installation it fails with 1618 ("another
  installation is in progress") unless it is scheduled after the installation has
  finished, outside its transaction and its elevation. It also adds 25 MB that must be pinned and
  re-pinned;
- **an NSIS bundle**: Tauri can build one and run the redistributable from it, but
  it would replace the MSI, which existing installations upgrade from, and give up
  the administrative install the release workflow inspects.

`release.yml` locates the module with `ci/find_vc_merge_module.ps1`, checks it before
the build and the built MSI after it with `ci/check_msi_runtime.ps1` (the merge module
recorded in `ModuleSignature`, the three DLLs in the `File` table at 14.40 or newer,
installed into `System64Folder`), and the Windows CI job also installs the MSI on a
runner that already has a newer runtime and uninstalls it again. A plain
`cargo tauri build` does not read the overlay and produces an MSI without the
runtime, which is the right default on a developer machine without Visual Studio's
merge modules.

The runtime is redistributed under the Microsoft Visual Studio license terms, which
allow the files on Visual Studio's redistribution list (REDIST.TXT), the runtime's
merge modules among them, to be distributed as part of an application. It is not a
Rust dependency, so it is not in `THIRD_PARTY_LICENSES.md` or `sbom.cdx.json`, both
of which describe the engine binary; in Windows' list of installed programs it is
part of MuMDIA rather than an entry of its own.

**The Components screen installs it on request.** For a machine where the installer
did not provide it (an installation from an older MSI, or a runtime removed
afterwards), a failed real import whose traceback is a torch DLL failure shows
**Install Visual C++ runtime** next to Re-check, on Windows only. Nothing happens until it is pressed.
Then `vcruntime.rs`:

1. downloads https://aka.ms/vs/17/release/vc_redist.x64.exe, Microsoft's permanent
   link to the current redistributable, into a new directory under `%TEMP%`, with
   progress (a proxy is read from `HTTPS_PROXY` / `ALL_PROXY`; a failed download says
   so and gives the link to install it by hand);
2. holds the file open so that it can be read but not replaced, and asks Windows for
   its Authenticode signature (`Get-AuthenticodeSignature`, with the path passed in
   an environment variable, never in the script). Anything but a valid signature
   whose signer's organisation is `Microsoft Corporation` is refused and the file
   deleted;
3. launches it elevated with `/install /passive /norestart`: Windows shows its
   permission prompt, and declining it is reported as such. The installer's own
   window shows its progress;
4. reads the exit code: 0 installed; 1638, this or a newer runtime was already there,
   so the failure is more likely an older `msvcp140.dll` earlier on `PATH`
   (`where msvcp140.dll` lists them in the order Windows tries them); 3010, installed
   but Windows must restart first, which the screen says; 1602 closed, 1618 another
   installation running, anything else with the code and the manual link;
5. after 0, 1638 or 3010, tests every environment whose import failed again, as
   Re-check does, so the cards move to "testing..." and then to the new result.

The sequence is a small state machine over three steps behind a trait, so it and
every message are unit-tested without Windows; the Windows CI job runs the real
signature check on system binaries, the hold, and the whole sequence against
Microsoft's server on the runner (which takes the 1638 path). Not tested anywhere: a
real Windows machine without the runtime, and the permission prompt itself, which
the hosted runners do not show.

### Uninstalling does not remove them

An MSI removes exactly what it placed under Program Files. Everything the
application downloads or builds is written at runtime under the data directory, so
an uninstall leaves all of it behind: two Python environments, ThermoRawFileParser,
an optionally downloaded DIA-NN, the predicted-library cache and the saved settings.
Measured on one development machine, 8.9 GB of it, with nothing in the interface
that could remove it.

That the installer leaves it is not itself the bug, and the fix is deliberately not
a WiX uninstall custom action:

- an upgrade reinstalls over the same data directory and reuses a
  several-hundred-megabyte download and a library cache that costs hours to rebuild;
- an MSI "uninstall" also runs during some upgrade paths, so a silent delete there
  would destroy a predicted library as a side effect of a version change.

The Setup screen therefore has a **Managed data** card. `components::inventory`
lists what exists, item by item, with the bytes each occupies; each row names the
exact paths and takes two clicks to remove. The items are `primary`, `ms2pip`,
`thermo`, `diann`, `libraries` and `settings`, and each directory name comes from a
constant exported by the module that writes there, so a rename cannot leave
something behind that nothing offers to remove.

One more row, `engine_caches` ("Search caches"), is not the application's own: it is
the engine's cache of FASTA-built libraries and DeepLC projections (docs/14, "The
engine's caches"), which the engine keeps within `MUMDIA_CACHE_MAX_GB` (100 GB by
default). On Windows it lives in `%LOCALAPPDATA%\MuMDIA\cache`, inside the data
directory; on Linux and macOS in `~/.cache/mumdia` or `~/Library/Caches/mumdia`, outside
it. Only the engine knows which names in that directory are its own, so the row is listed
through `mumdia cache --json` and removed through `mumdia cache clear --json`, not through
`remove_in`; an engine without the command offers no row.

Two properties matter and are tested. `remove_in` resolves the real parent directory
and refuses anything outside the data directory, and refuses a symbolic link
outright, because this is a recursive delete driven by a string from the frontend.
`components_remove` refuses entirely while a search, an installation or a library
build is running, since those hold the very files it would delete.

By hand, the same thing:

```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\MuMDIA"           # everything
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\MuMDIA\python"    # just the environment
mumdia cache clear                                               # just the search caches
```

```bash
rm -rf ~/.local/share/MuMDIA
mumdia cache clear      # the search caches, in ~/.cache/mumdia
```

### Storage folder

Before the Storage folder existed, everything above lived in fixed per-user places on the
system drive: the data directory (`%LOCALAPPDATA%\MuMDIA`, `~/.local/share/MuMDIA`), the
engine's caches, the system's temporary folder, the predictors' model downloads in the home
folder, and `uv`'s package cache. On an account whose profile is small, roams, is synced or
may not be written, a search failed in whichever came first, and nothing on the screen
could move it.

The Setup screen's **Storage folder** card shows the folder in use, its free space and
whether it can be written (a file created and renamed there, which is what the caches and
installers do), with **Change...** (a folder picker) and **Use the default**. A picked
folder that is not new, not empty and not already a storage folder (marked by
`.mumdia-storage`) gets a `MuMDIA` sub-folder, so a user's own `configs` or `libraries`
folder in, say, Documents is never mixed with MuMDIA's or offered for removal on the
Managed data card. The storage folder holds:

| sub-folder | what | exported to the engine and `uv` as |
|---|---|---|
| `libraries/`, `configs/` | predicted DIA-NN libraries, saved settings | (the application's own) |
| `cache/` | the engine's caches | `MUMDIA_CACHE_DIR` |
| `temp/` | temporary files of the engine, its Python workers and this application (the Visual C++ runtime download) | `MUMDIA_TEMP_DIR` |
| `models/` | MS2PIP and AlphaPeptDeep model downloads | `MUMDIA_MODEL_DIR` |
| `uv-cache/` | `uv`'s package cache | `UV_CACHE_DIR` |

The Python environments and the downloaded tools (ThermoRawFileParser, DIA-NN) stay in the
default folder unless **Also keep the Python environments and downloaded tools there** is
ticked: an environment cannot be relocated (its interpreter paths are absolute), so moving
it means installing it again, a few hundred megabytes nobody should get by surprise. With
the option, `uv`'s interpreters move too (`uv-python/`, `UV_PYTHON_INSTALL_DIR`). Nothing is
copied in either case: the caches, libraries and models are rebuilt or downloaded where the
new folder is, and the old folder's items can be removed from the Managed data card first.

The setting is kept in `storage.json` in the default folder, the one place the application
can find before it knows the setting. `MUMDIA_DATA_DIR`, set before the application starts,
replaces that default folder (environments and `storage.json` included), for an account
whose profile cannot be written at all. A variable exported to the engine never overrides
one already in the application's environment, and nothing is exported while no folder is
chosen, so the engine's own defaults then hold exactly as before. The folder is never a
configuration value, so a configuration's hash and a search's output do not depend on it.

A change is refused while a search, an installation, a library build or the viewer is
running, as a removal is. The search preflight blocks a results folder or a chosen storage
folder that cannot be written (checked without creating the results folder), and warns
about a folder inside OneDrive or on a network share and about a storage folder with less
than 20 GB free. The run log (`mumdia-console.log`) is opened before the engine starts, and
a results folder that cannot take it stops the start with the reason instead of running a
search with no log. The pure logic is `src/storage.rs`, with its tests.

### Two environments, not one (historical)

MS2PIP could not share an environment with DeepLC at the versions this project
tested until 2026-09-07:

    deeplc==4.4.0  -> psm-utils>=1.5 -> sqlalchemy>=2
    ms2pip==4.0.0  ->                   sqlalchemy>=1.3,<2

`uv` reported the pair as unsatisfiable, so the primary environment covers
rescoring, DeepLC and match-between-runs, and MS2PIP got its own, installed on
request and needed only for FASTA-mode library building with predicted
intensities. The shipped MS2PIP is now 4.2.0, which resolves next to DeepLC (the
host specification `env/mumdia-deeplc.yml` holds all three), and the FASTA path
was measured with it (`docs/28`, section 22). The application still installs it as
the separate optional component; folding it into the primary environment and
removing the second `Env` is a follow-up, not a requirement.

### visDIA, the results viewer

A third, optional environment (`Env::Visdia`, `python-visdia`) holds
[visDIA](https://github.com/CompOmics/visDIA) (`mumdia-viewer`), the interactive viewer
for MuMDIA results. Its own environment because its Dash and DuckDB stack has nothing to
do with the analysis packages, and neither should constrain the other.
`env/console-visdia-requirements.txt` pins one visDIA commit through GitHub's source
archive, so no git is needed and everyone on a release gets the same viewer; that commit
was checked against this release's outputs (experiment and single run: counts,
identification table, precursor detail, mirror spectrum). About 350 MB on Windows.

"Open in visDIA" sits on the Results screen, on each History entry with a scored table,
and on each file of a separate-files batch. `viewer::open` starts
`python -m mumdia_viewer <dir> --no-browser` from that environment, reads the
`serving ... at http://127.0.0.1:<port>/<token>/` line (with `PYTHONUNBUFFERED`, or a
piped stdout holds it back), and opens only such a loopback address. One viewer per
folder: opening the same folder again shows the running one. A viewer that exits before
it serves is reported with its own last lines, which name the actual problem. Viewers
have no window of their own, so all of them are stopped when the application closes,
and removing the environment is refused while one runs. For the search just finished,
the FASTA chosen on the Search screen is passed with `--fasta`, so a library DIA-NN
built from it still gets sequence coverage.

`viewer::tests::a_results_folder_is_served_and_stopped` (ignored by default; set
`MUMDIA_TEST_VISDIA_DIR` to a results folder, with the environment installed) starts
the real viewer, fetches its page, checks the reuse, and stops it.

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
  breach without ever noticing it exists. Since 2026-10-10 a DIA-NN that is installed
  and runs counts as acknowledged until the user answers (`effective_licence`):
  installing it meant accepting its licence from the vendor. The box is shown ticked
  with a note saying why, an explicit untick is stored and respected, and the download
  of 1.8.1 still needs an explicit tick, because nothing is installed at that point.
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

The digest parameters live on the Search screen, beside the controls that consume
them, and they apply to both ways of building the library. They used to sit on Setup
in a separate "Predict a library" card while the radio that used them was here, so
invisible state on a screen the user need never have opened decided the search space,
the cache key, and whether the "already built" note was true; the `if (!el) return
dflt` guards made every mismatch silent. That card is gone and this is the only way to
build a library, which also removed the branch that silently overwrote a manually
chosen library pair on every visit to Setup.

A second version of the same mistake lived on until 2026-09-07: the fields sat inside
the DIA-NN block and were read only by the DIA-NN library build, so with the built-in
predictors the engine digested with the preset's values and a missed-cleavage count
typed on the screen reached nothing (the block also carried the two modification
checkboxes twice, with the same ids; only the first pair was read). Now the fields sit
above the two radios, and for the built-in path `start()` asks `derive_config` for a
run configuration that is the selected preset with the fields merged on top
(`settings::derive`), validated by the engine like any saved settings file. `start_run`
still receives an ordinary `--config` path and stays the one tested entry point.

FASTA mode offers two ways to get a library: the engine's built-in predictors, or
DIA-NN predicting one first. The second is the sensitive path -- the ~1,213 against
~10,300 figure is largely this difference -- so a user starting from a FASTA should be
able to take it without first understanding that a library is a separate artifact
built on another screen. It is the selected option whenever DIA-NN can be used (found,
runs, licence acknowledged); otherwise the built-in predictors are selected and DIA-NN
comes back by itself once it can be used, unless the user picked one explicitly. DIA-NN
runs on all cores but two by default, so the machine stays usable during a prediction.

While DIA-NN builds the library the Progress screen is reachable from the menu, shows
the build log, and its "Command being run" panel lists every command the build has
started (prediction, re-export, conversion, decoys, decoy prediction), from
`BuildState::commands`.

**The cleavage rule is one of the digest parameters.** The Cleavage choice lists the
engine's presets (`mumdia-core/src/enzymes.json` through the `enzymes` command:
Trypsin/P, Trypsin, Lys-C, Lys-N, Arg-C, Asp-N, Glu-C, chymotrypsin, non-specific) and
a custom rule in DIA-NN's `--cut` pair syntax (`K*,R*,D*,!*P`). It reaches the DIA-NN
build as `--cut`, the built-in path as `digest.enzyme` / `digest.cleavage_rule`, and the
search-space estimate. Under a rule that cuts every bond peptides are enumerated by
length, so the missed-cleavage field is disabled and the DIA-NN build asks for
`max_len - 1` missed cleavages, the sites a peptide of the maximum length spans; the
GlyGly `--no-cut-after-mod` rule is passed only for a rule that cuts after K. The
application's rule parser (`enzymes.rs`) mirrors the engine's and is tested on the same
`test_vectors`.

**A predicted library is cached, content-addressed.** It depends on the FASTA's bytes
and the digest parameters and on nothing else: not the mzML, not the thread count. So
`library_cache_dir` keys it on a digest of the FASTA plus every parameter that changes
the output plus the DIA-NN version, and the search reuses it. Without that, offering
this on the search screen would promise a whole-proteome prediction on every run. The
DIA-NN version is in the key because its version changes what it predicts, and reusing
a library across versions would silently change results. The thread count is
deliberately out, because including it would miss the cache for nothing. The cleavage
rule enters the key only when it is not Trypsin/P, the rule every earlier build used,
so libraries cached before the choice existed are still found.

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
`make_reverse_decoys.py` adds the decoy population, `predict_decoys.py` has DIA-NN predict
the decoys' spectra in place of the copied target intensities. The last also sorts by precursor
m/z and re-indexes `candidate_id`, both of which the fragment index rejects a
library for lacking. On success the two tables are selected on the search screen
automatically, because the alternative is retyping two long paths.

MuMDIA reads only the library. Invoking DIA-NN is not reading its source, so the
clean-room boundary is unchanged.

### Which predictors the built-in path uses

The built-in path derives the run's configuration from the selected preset, and until
this section existed it took the preset as it was. The preset selected by default is
`diann-library`, which is right for a library predicted or imported before the search and
names no fragment predictor, so a FASTA search that fell back to the built-in path (the
first library option unavailable) predicted its library with the engine's native models. A desktop user searched
twelve timsTOF diaPASEF runs that way: the native intensities left the seed of every run
without a confident anchor, every run was extracted over the whole gradient, and the search
took about ten hours for a small fraction of the identifications, with nothing on the
screen saying why.

`builtin_plan` (`predictors.rs`) now decides, from the selected preset and the selected
files:

- a preset or saved settings file that names `predict_frag.predictor` is the user's choice
  and is kept;
- a preset this application ships that names none (or the engine defaults) is replaced by
  the first recipe whose interpreters `mumdia doctor --json` reports as usable: the
  diaPASEF recipe (`diapasef.json`: AlphaPeptDeep, DeepLC, IM2Deep and the diaPASEF
  extraction, retrace and quantification settings) when any file is a timsTOF diaPASEF
  `.d`, then `fasta-sidecars.json` (MS2PIP, DeepLC), then `fasta-peptdeep.json`
  (AlphaPeptDeep, DeepLC). The digest and modification fields are merged on top of the
  recipe exactly as they were on top of the preset;
- a saved settings file that names no predictor is kept, because replacing it would drop
  the user's edits, and is warned about.

The Search screen shows the plan under the built-in option (`libsrc-builtin-note`), and a
warning (`builtin-warn`) when the search would still run on native fragment intensities, or
on a diaPASEF `.d` without predicted 1/K0. Start asks the plan again and, when it carries a
warning, opens a dialog that says what the search would cost and starts only on "Start
anyway". `preflight` adds the same warning for any FASTA configuration, so a batch or a
configuration from the settings editor is covered too. Each recipe considered costs one
`mumdia doctor`, which imports the interpreters' packages, so the plan runs off the
interface thread and a stale answer is dropped.

The engine reports the other half: a run whose seed left no retention-time calibration is
recorded under `rt_calibration_missing` in its manifest (docs/17, "Seed: no confident
targets"). `read_results` reads it, and the results screen leads with a red banner
(`res-degraded`), "SEARCH DEGRADED: N of M runs had no retention-time calibration", with the
usual causes; a separate-files batch marks each such file in its per-file list.

## Several files

The Spectra picker takes a list. One file is an ordinary search; several mean one of
two different analyses, and the interface shows the choice because conflating them
would be a scientific error rather than a UI simplification. **One experiment is the
default**: files provided together are rescored together, which is also what
`mumdia run` does with several `--mzml`. Searching each separately is the opt-in.

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
- its `peptides.tsv` and `proteins.tsv` are **experiment-wide**: selected on the
  experiment-wide q columns, one quantity column per run (`quantity_<run>` for
  precursors, `lfq_<run>` for protein groups), no per-run TSVs. `read_results` takes
  the counts from `scored_combined.parquet` and flags the result `experiment_wide`,
  which is what drives the banner.
- per-run artifacts go to `<out_dir>/<name>/`, so the stage list looks sparse until
  the pooled phase.

A DIA-NN-predicted library is built **once** for the whole selection, not per file:
the library depends on the FASTA and the digest parameters, not on the spectra.

The backend refuses two things rather than guessing: a pooled experiment with fewer
than two files, and the same path selected twice (which would search one file twice
and pool the result with itself, inflating the evidence for those peptides).

## Prescreen, bands and modifications on the Search screen

Three engine settings that change how large a search is are on the Search screen, so
that turning them on does not need a settings file. Each becomes a configuration path
merged on top of the chosen preset through `derive_config`, the same route the digest
fields take, and the engine validates the result before anything starts. With all three
at their defaults (prescreen off, automatic bands that resolve to one band, the standard
modifications) the preset reaches the engine unchanged in library mode.

**Prescreen** (`prescreen.*`, docs/34). The screen says what it is for and what it
costs: a candidate it removes can never be identified, so it trades identifications for
time and memory (immunopeptidomics end to end: -3.5% peptides, -25% wall time, half the
memory). Off; before prediction without retention times
(`score_before_prediction` with `crowding_exponent = 0.25`, the no-RT score docs/34
measured); after the retention-time calibration (`enabled`); or the database-free tag
prefilter (`tag_prefilter`). Light, balanced and stringent map to a target of 0.25 and
to the `balanced` and `stringent` presets. The two modes before prediction are
single-file only (`run-experiment` refuses them), and the mode after calibration cannot
be combined with bands (config validation refuses it), so the screen refuses both
combinations before a library is built or a run starts, and says which choice to change.

**Bands** (`groups.window_groups`, `groups.parallel`, docs/33). Automatic, off, or a
custom count. The automatic plan (`sizing::band_plan`) takes the machine's physical
memory (`GlobalMemoryStatusEx` on Windows, `/proc/meminfo` and the cgroup v2 limit on
Linux, `sysctl hw.memsize` on macOS), a budget of 60% of it and never more than all but
6 GB, and a memory model fitted to the docs/33 peaks: about 4 GB per band plus about 2
GB per million library precursors in the band (the 44.6M-precursor band of the 203M
immunopeptidomics library peaked at 96 GB; the unbanded 10.9M HYE library at 16.5 GiB,
so the model is conservative on tryptic data). It bands only when the unbanded estimate
exceeds the budget, because bands repeat fixed work and do not change identifications,
uses the fewest bands that fit, at most 64, and runs as many at once as the budget
holds, below the thread count with at least four threads each. Automatic re-sizes the
inputs at Start, so a plan computed before the last change cannot decide the run.
"Automatic" that resolves to one band leaves the preset's own `groups` block alone;
"Off" and "Custom" write it.

The size comes from the library footer in library mode (`mumdia inspect`, rows only)
and from the FASTA otherwise (`sizing::fasta_space`): a digest under the screen's
cleavage rule with its missed cleavages and lengths (every window of the length range
for a non-specific rule), N-terminal Met excision, every charge in range,
one paired decoy per target, and each distinct peptide's modified forms counted exactly
as the elementary symmetric sums of its sites' alternatives up to the variable maximum.
On the E. coli FASTA with the engine defaults it gives 1,925,388 precursors against the
engine's 1,922,388, in 0.23 s for a debug build. A prescreen before prediction scales
the planned size by the fraction its strength is expected to keep.

**Modifications** (`peptidoforms.fixed_mods`, `variable_mods`, `max_variable_mods`).
The list is the engine's catalogue, `rust/mumdia/crates/mumdia-core/src/modifications.json`,
embedded in both the engine's mass table and this application (`modifications`
command), so every name offered is one the engine accepts and a new entry appears in
both at once. Each residue chip cycles variable, fixed, not searched; the quick sets come
from the same file; a filter narrows the list by name, label, residue, group or accession.
An entry may carry a plain-language `label`, shown under its name, for a UniMod name that
does not say what it is: typing "homocysteine" finds `HCysThiolactone` (UniMod 1270,
N-homocysteinylation of K) and `HCysteinyl` (UniMod 1271, S-homocysteinylation of C). Two
fixed modifications on one residue, or a fixed and a variable one, are reported as the
selection is made and refused at Start, because the peptidoform stage would refuse them
mid-run. A DIA-NN build carries the whole selection (see below). The selection is
remembered between sessions in browser storage, as a convenience only.

Validated 2026-10-05: the Search screen driven in headless Edge with a mocked command
bridge (catalogue, sets, filter, conflicts, the derived overrides at Start), and the
derived configuration run end to end on the smoke fixture: the prescreen before
prediction and two bands each identify the planted peptides as the plain run does
(110 and 115 of 160 against 116). The fixture cannot judge a large modification set:
with about 150 true peptides the `(d + 1) / t` floor puts one extra decoy peptide above
1%, so that measurement is a real FASTA search.

On the Orbitrap AIF E. coli file (`bench/prescreen/ps_mods.sh`, MS2PIP + DeepLC, `nn_torch`,
stripped peptides at 1%, mean of 3 NN seeds, EPYC 9354, 64 threads):

| arm | peptidoforms predicted | peptides at 1% | wall | peak RSS |
|---|---|---|---|---|
| carbamidomethyl C, oxidation M | 1,922,388 | 10,632 | 8:39 | 26.7 GB |
| + Phospho STY, Acetyl K, Deamidated NQ | 8,750,640 | 9,987 | 38:37 | 89.6 GB |
| the same + prescreen before prediction | 3,217,290 | 9,945 | 12:15 | 26.4 GB |

The larger search space costs 6.1% of the peptides, which is the FDR price of 4.6x the
candidates on a sample without enriched modifications. The prescreen before prediction
then removes 63% of the peptidoforms before MS2PIP and DeepLC, for -0.4% peptides (seed
spreads of 40-90), 3.2x less wall time and 3.4x less memory.

DIA-NN builds take the same selection (`ModPlan` in `diann.rs`): carbamidomethyl C is
`--unimod4`, every other modification `--fixed-mod` / `--var-mod UniMod:<id>,<mass>,<residues>`
under `--var-mods <max>`, plus `--no-cut-after-mod UniMod:121` for GlyGly K. The standard
selection keeps its arguments and cache key, so existing cached libraries are reused, and
the importer is told which accessions to keep (`--keep-unimod`). Checked with DIA-NN 2.2.0
on 60 E. coli proteins with Phospho STY, GlyGly K and Oxidation M: all 33,885 target
precursors imported, and the decoy builder's mass check at 0.02 ppm median.

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
  the progress screen. The converters are asked of the engine with the request's own
  configuration (`doctor --json --config`), so one named in `convert.thermo_raw_parser`
  or `convert.msconvert` counts, and the rule is the engine's: a Thermo `.raw` with the
  parser at `auto` and only msconvert present runs, with a note; a parser the
  configuration names and that is missing blocks, as it errors in the engine
  (docs/29 #13).
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

The Settings screen shows a short list of common settings first, under plain names
(`COMMON_SETTINGS` in `app.js`: rescoring model, FDR threshold, tolerances, retention-time
window and calibration, gate, competition unit, match-between-runs, rescoring seeds), and
every other parameter in a collapsed "In-depth settings" section that a search or "only
changed" opens. Both are generated from the schema; only the choice of the common few is
written by hand, and a path the schema lacks is skipped.

The application opens on the Setup screen, and the Search screen's preset is
`diann-library` when that preset is available.

The editor starts from the preset selected on the Search screen, not from the engine
defaults: `config_overrides` flattens the preset file into the same dotted paths the
form uses (`settings::flatten`, the inverse of `nest`), so the saved file is the
preset plus the edits. Until 2026-09-07 it started from the defaults, and saving with
a preset selected silently dropped the preset's predictor, rescorer and interpreter
choices. Selecting another preset re-seeds the form and says so. Fields the engine
accepts but does not act on yet (the schema marks them `not yet wired`: the
match-between-runs tiers) are shown with that label and cannot be edited; a value
typed there would have been saved, validated and then ignored by the run.

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

`cargo tauri build` produces an `.msi` on Windows and an `.AppImage` on Linux. The
Tauri resources are the engine and `uv` in `binaries/` beside the application, and
the Python workers in `binaries/scripts/`, all staged there by `release.yml` from the
same checkout. The workers are not optional: the engine resolves its relative
`sidecar_script_dir` against its own directory and accepts that directory only when
it holds a worker file. The 0.1.0 installers shipped `binaries/scripts/` with its
README alone, so DeepLC, the neural rescorer, mokapot and the DIA-NN import all
failed at the point of use; `release.yml` now stages `scripts/*.py` and opens every
bundle it built (`msiexec /a` on Windows, `--appimage-extract` on Linux) to assert
the console, the engine, `uv` and the workers are inside and the engine runs.
On Windows the build also passes `--config src-tauri/tauri.vcruntime.conf.json`,
which merges the Visual C++ runtime into the MSI ("The Visual C++ runtime" above), and
the same inspection checks that it is there.

Everything else the application needs from the repository is compiled in with
`include_str!`: the settings schema, and the three requirement sets. That is both
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
`binaries/mumdia.exe` and `binaries/uv.exe` beside it, and no `_up_` directory. That
inspection did not look inside `binaries/scripts/`, which is how the missing workers
reached 0.1.0; the workflow check above does.

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
