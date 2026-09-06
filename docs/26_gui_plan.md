# 26. Graphical interface: design and plan

Plan for a graphical way to run MuMDIA. Written 2026-08-28 from a survey of eight
comparable tools, a measured audit of what the engine can already support, and the
group's actual deployment (analyses run on a shared compute node reached through a
jump host).

The recommendation is a **web UI served by the engine itself** (`mumdia serve`),
with each run executed as a detached child process and its state kept on disk. The
reasoning below matters more than the conclusion, because two of the four
candidates are eliminated by a constraint that is easy to miss and expensive to
discover late.

## 1. What the survey settles

Eight tools were examined: DIA-NN, FragPipe, AlphaDIA, AlphaPept, MaxQuant,
Skyline, PeptideShaker, DeepLC.

| | toolkit | compute | shared params file | binds a port | cancel | survives GUI exit |
|---|---|---|---|---|---|---|
| DIA-NN | C# WinForms | child process | `--cfg`, only if args > 32k | no | yes | no ("progress will be lost") |
| FragPipe | Java Swing | child processes (DAG) | `.workflow` + `.fp-manifest` | no | yes | no; use `--headless` |
| AlphaDIA | Electron + React | child process | `config.yaml` | no | yes | no, no reattach |
| AlphaPept | Streamlit | detached process | settings YAML | 8505 | not located | **yes** |
| MaxQuant | C# WinForms | same process | `mqpar.xml` | no | not documented | no |
| Skyline | C# WinForms | same process | flags | no | unconfirmed | no |
| PeptideShaker | Java Swing | same JVM | params file | no | yes | no |
| DeepLC | NiceGUI | same process | none | 8080 | no | no |

Four conclusions follow, and three of them are about behaviour rather than
appearance.

**A GUI is expected to be a launcher over the command line, not a separate
application.** Six of eight keep the CLI as the real interface and five build or
write the exact invocation. DIA-NN's own README advertises that its log "prints
the exact set of commands it used". This is the shape users can debug, script and
publish, and it is the shape to copy.

**Progress is expected to be a log tail.** Not one of the eight shows a calibrated
per-stage percentage or an ETA for the search itself. AlphaPept's coarse markers
and FragPipe's batch bar are the only progress bars in the set. Modest
instrumentation is therefore differentiating rather than table stakes.

**Job lifetime is the discriminating requirement, and almost everyone gets it
wrong.** Seven of eight tie the run to the UI process. Only AlphaPept decouples
them, and it does so with three cheap primitives: a queue directory, a PID file
and a log file. For an 85-minute run started from a laptop that will sleep, a GUI
that loses the run when it closes is not usable, and no toolkit choice fixes that.

**Nobody has solved a local GUI driving remote compute.** In eight tools there is
not one such path. The documented answers are "run it all remotely from the CLI",
a container, or X forwarding, which is FragPipe's official answer and the worst
option on a slow link. We should not expect to invent this cheaply.

## 2. The constraint that eliminates two candidates

Measured crate counts, resolved with default features into an empty crate:

| approach | crates (Windows) | crates (linux-gnu) | system requirements |
|---|---|---|---|
| `tiny_http` | **6** | 6 | none |
| `axum` | 50 | 49 | none of consequence |
| `eframe` (egui) | 173 | **263** | `pkg-config` at build; libX11, libxcb, libGL, libEGL, libwayland at run time |
| `tauri` | 236 | **278** | GTK3 + WebKit2GTK via `-sys`; Xcode on macOS; MSVC plus WebView2 on Windows |

The workspace has deliberately avoided C toolchains: `parquet` uses `snap` rather
than zlib-ng "needs cmake", `mzdata` uses `miniz_oxide` for the same reason,
`mimalloc` was chosen partly for being pure Rust. The only C-toolchain crate today
is `cc`, as a build dependency of `blake3` and `libmimalloc-sys`. `ci.yml` contains
no `apt-get` line at all, and the only system package in `release.yml` is
`musl-tools`.

**The Linux release target is a fully static musl binary, and both desktop
toolkits are incompatible with it.** They dynamically link system graphics or
webview libraries and, in `eframe`'s case, `dlopen` a Vulkan loader, which a
statically linked binary cannot do. Choosing either means giving up the
single-file Linux artifact and adding system-package steps to two workflows.
Neither can run in the shipped container, which has no display.

That removes `eframe` and `tauri`. The choice is between a web UI inside the
binary and a separate Python application.

## 3. Recommendation

**Launcher: a web UI served by the engine, `mumdia serve`, using `tiny_http`.**

- Six crates, no system dependency on any platform, builds for musl, ships inside
  the existing binary and release archive. The container needs one `EXPOSE` line.
- Reached over SSH the way this group already works: bind loopback, `ssh -L 8080:localhost:8080 doxy`, open the page. Both CompOmics precedents already do
  exactly this (DeepLC's `[web]` extra on 8080, AlphaPept on 8505).
- The server owns the runs, so a closed browser, a sleeping laptop and a dropped
  tunnel are all survivable by construction.
- `tiny_http` rather than `axum` because the UI is single-user and local: blocking
  request handling is adequate, and it avoids pulling tokio and hyper into a
  binary that currently has no async runtime.

**Results viewer: separate, later, and probably Python.** A launcher needs no
Parquet reader, because everything it must show is already on disk as text or JSON
(section 4). A viewer that draws chromatograms and annotated spectra needs lazy
access to a 4.53 GB, 27-million-row `chromatograms.parquet`, and the group has
already specified that work in Python with Plotly and polars or duckdb. Keeping
the two apart lets the launcher ship in the engine and the viewer follow in the
ecosystem where DeepLC already lives. Revisit only if the viewer turns out to need
nothing but tables.

**Runs execute as detached child `mumdia run` processes, never on a server
thread.** Three independent reasons:

1. `rayon::ThreadPoolBuilder::build_global()` can be called once per process, and
   `--threads` also sets process-global `MUMDIA_NN_THREADS` and
   `OMP_NUM_THREADS`. A server hosting many runs on threads would freeze one
   thread budget at startup for every run.
2. The sidecar workers inherit stdout deliberately, so a child process with a
   piped stdout captures the engine's own lines *and* the Python worker's
   per-fold progress in one stream, attributable to that run. On a server thread
   the worker's output would land on the server's stdout, belonging to nobody.
3. A crashing or cancelled run cannot take the server with it.

## 4. What already exists, and what has to be built

Verified against the current tree, because the honest starting point matters.

**Usable today, with no engine change:**

- **A 21-step progress bar for free.** Every stage writes
  `<artifact>.report.json` on completion, at 17 call sites, carrying `stage`,
  `rows`, `elapsed_ms`, `content_hash` and a `stats` map. Polling the output
  directory yields real per-stage completion with row counts. This is the single
  most valuable existing hook.
- **The results summary needs no Parquet reader.**
  `psms_scored.parquet.report.json` already carries `psms`, `classifier`,
  `target_psms_at_1pct`, `target_peptides_at_1pct`,
  `target_protein_groups_at_1pct`, `target_precursors_at_1pct`, and records the
  classifier that actually ran against the one requested. `peptides.tsv` and
  `proteins.tsv` give the tables.
- **Provenance for a results page.** `manifest.json` carries the engine version,
  the short commit with a `-dirty` marker, the commit date, the full command line,
  and a blake3 hash of every input. It is written once at the end, so it is a
  completion record, not a status file.
- **A reliable setup check.** `mumdia doctor` computes per-role interpreter path,
  how it was resolved, missing modules and package versions, and its exit code is
  trustworthy.
- **Per-run progress in an experiment.** `run-experiment` logs
  `run = <name>, i, n`, so "run 3 of 8" is available.
- **A resume shortcut.** All 21 subcommands are standalone on path-addressable
  inputs. Cancelling at minute 70 discards the rescore but leaves a valid
  `psms_competed.parquet`, and `mumdia rescore --competed ...` restarts from
  there. That is worth surfacing as a button.

**Missing, and needed:**

| # | gap | why the GUI needs it | size |
|---|---|---|---|
| 1 | **No incremental progress in the stage that dominates the run.** Rescore is 80% of an 85-minute run and emits one line at its start and one at its end. The only live signal in between is roughly one unstructured Python print every two to four minutes. | Without this the GUI shows an elapsed clock and nothing else for most of the run. | medium |
| 2 | **No machine-readable log.** `tracing-subscriber` is built without its `json` feature. Worse, the human format writes to **stdout** with ANSI escapes even when piped, wrapping each field name and value separately, and `println!` output interleaves on the same stream. A naive `key=value` parser cannot work. | A GUI parsing stdout needs `--log-format json` and `--no-color`. | small |
| 3 | **No `doctor --json`.** | The setup screen is the highest-value screen, since misconfigured Python environments are the failure every surveyed tool documents at length. All the fields already exist; only the output format is missing. | small |
| 4 | **No signal handling anywhere.** No `ctrlc`, no `SIGINT`/`SIGTERM` handler, no cancellation flag. Ctrl-C kills the process wherever it is. | A Stop button, which four of eight surveyed tools have. | small |
| 5 | **Artifact writes are not atomic.** `File::create` targets the final path with no temp-then-rename, so an interrupted run leaves a truncated Parquet at the canonical name. It fails to open rather than reading as valid, so it is detectably broken, but it is still rubble in the output directory. | Cancellation has to leave a clean directory. Worth fixing regardless of the GUI. | small |
| 6 | **No per-run state on disk.** | Reattaching after a server restart, and listing what ran. AlphaPept's queue-directory plus PID-file plus log-file design is the proven minimum. | medium |

Items 2, 3, 4 and 5 are worth doing for the command line alone, which is the
argument for doing them first.

## 5. Scope of the first version

Three screens. Deliberately not a configuration editor: there are 160 leaf
configuration fields across 17 structs, and the three shipped profiles set only 4,
12 and 12 of them respectively.

**Setup.** Pick input mzML files from the server's filesystem (the data lives on
the compute node, so this is a server-side browser, not an upload). Pick a library
source: a FASTA, or the precursor and fragment pair. Pick one of the three shipped
profiles. Set the output directory and `--threads`. A "check environment" button
runs `doctor` and renders one row per sidecar role.

**Run.** The 21-step stage list with per-stage elapsed time and row counts from
the `report.json` files, a live log tail, and a Stop button. When a run is stopped
or fails after competition, offer the `rescore --competed` restart.

**Results.** The counts from the scored artifact's report, the provenance stamp,
a preview of `peptides.tsv` and `proteins.tsv`, and links to the files.

**Explicitly out of scope for the first version:** a general configuration editor
(offer a JSON text area for advanced users instead), plots, vendor-format
conversion, multi-user accounts, and file upload.

**Two traps to avoid, both documented in this repository.** Do not offer
`--top-peaks-ms2` as a general control: carrying the AIF-specific value 300 to a
50-window run cost 60% of the peptides, and a slider would manufacture that
failure. And do not label a `peptides.tsv` row count as a precursor count at a
precursor q threshold: the rows are precursors but the filter column is a
base-peptide q.

## 6. Security, which a launcher makes non-optional

A server that accepts a file path and starts a process is a remote-code-execution
surface. On a shared compute node, loopback is **not** a boundary: every other user
on that host can reach `127.0.0.1`. A config field names the Python interpreter to
execute, so an unauthenticated request can run an arbitrary program as the
operator.

Requirements, not options:

- bind `127.0.0.1` by default, and require an explicit flag to bind anything else;
- print a URL containing a random per-process token, and require it on every
  request, the way Jupyter does. This is the single control that makes a shared
  compute node safe;
- reject paths outside an allowed root, so the file browser cannot be walked to
  arbitrary locations;
- never accept an interpreter path from the request; take it from the config or
  from `"auto"` resolution only.

Note as a live trap rather than a hypothetical: NiceGUI's non-native default binds
`0.0.0.0`, so the DeepLC precedent would expose the service to the network if
copied without thought.

## 7. Sequencing

**Phase 0, useful on its own, no GUI (small).** `--log-format json` and
`--no-color`; `doctor --json`; a `SIGINT`/`SIGTERM` handler that stops cleanly;
temp-then-rename for artifact writes. This is a robustness and scriptability
improvement whether or not the GUI is ever built, and it is the prerequisite for
everything else.

**Phase 1, incremental progress (medium).** A progress event per unit of work in
the loops that dominate: extract over candidates, features over rows, and above all
an iteration counter in the NN rescore worker, which is 80% of a run. Emit it on
the structured log so both the CLI and the GUI can consume it.

**Phase 2, `mumdia serve` (medium).** `tiny_http`, one embedded page, no build
step and no npm. Runs as detached children with per-run state on disk. The three
screens of section 5, with the token and path-root controls of section 6.

**Phase 3, results viewer (separate project).** Python, lazy Parquet, per the
existing specification.

Phase 0 should not wait for a decision about the GUI. Phase 2 should not start
before Phase 1, because a launcher that cannot show progress through the stage that
takes 80% of the run will be judged by exactly that.

## 8. What this plan does not claim

The survey found no comparable tool that solves remote GUI over remote compute, and
this plan does not solve it either: it sidesteps it by putting the UI on the compute
node and reaching it through the tunnel the group already uses. If the requirement
ever becomes a desktop application on a laptop driving a remote engine, the work is
mostly SSH plumbing (agent-held passphrase keys, a jump host, correct detachment,
and reattachment across GUI restarts), not user interface, and reattachment is
precisely the part every surveyed tool skips.
