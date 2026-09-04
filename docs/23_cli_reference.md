# MuMDIA command-line reference

GENERATED FILE. Do not edit. `ci/gen_cli_reference.py` writes it from the
`mumdia` binary's own `--help` output, so the interface described here is
the interface the binary actually has. The source of truth is the `clap`
derive in `rust/mumdia/crates/mumdia/src/main.rs`: change the flag
or its doc comment there, rebuild, and regenerate. An edit made to this
file is lost on the next run.

```text
python ci/gen_cli_reference.py            # regenerate
python ci/gen_cli_reference.py --check    # fail if this file is stale
```

For what each stage does with these arguments, read `docs/README.md` and the
per-stage documents it routes to. For the configuration file passed with
`--config`, read `docs/24_config_reference.md`.

## How to read this document

Every block below is the binary's own help text, right-stripped and with
the program name normalized to `mumdia` (a Windows build reports
`mumdia.exe`). Nothing is paraphrased. Two mechanical edits are applied:

- the 4 flags clap marks `global = true`, and `-h, --help`, are removed
  from the per-subcommand blocks and documented once under "Global flags"
  below, because clap repeats them in every subcommand;
- `mumdia help` gets a table row but no section, because
  `mumdia help --help` reprints the top-level help unchanged.

## Top-level help

```text
MuMDIA DIA search engine (Rust MVP)

Usage: mumdia [OPTIONS] <COMMAND>

Commands:
  convert         Read an mzML run into the normalized spectra artifact set
  digest          Fully-tryptic digest + decoy pairing -> peptides.parquet
  peptidoforms    Fixed+variable modification and charge enumeration -> peptidoforms.parquet
  predict-frag    Spectral library: b/y m/z + predicted intensity + iRT -> fragment_library
  prescan         Sequence-tag prescan: keep only modification-bearing candidates whose anchored trimers are observed in this run -> prescan_survivors.parquet. Label-blind by construction, so it prunes search space without touching target-decoy exchangeability
  search-seed     Native broad DIA seed search over the fragment index -> seed_psms.parquet
  rt-im-train     Per-run RT calibration + windows -> run_windows.parquet, cal.json
  extract         Targeted 3D extraction (peak-major cascade) -> psms_extracted, chromatograms
  features        Compute the minimal feature set -> features.parquet + PIN
  compete         Keep the best candidate per competition group -> psms_competed.parquet
  rescore         Rescore + native target-decoy q-values -> psms_scored.parquet
  quant           Quantify identified peptides + roll up to protein groups
  quant-lfq       Combine per-run quant tables into a protein-by-run matrix (cross-run LFQ)
  run             Orchestrate the full MVP pipeline on one run and write a manifest
  run-experiment  Experiment-wide orchestrator: run the per-file search chain over N runs, then one combined rescore, optional rescuable MBR transfer, per-run quant, and cross-run LFQ. Pass --mzml once per run (>= 2)
  align           Cross-run RT alignment (experiment-level) -> alignment.parquet
  mbr             Match-between-runs identification transfer (Stage D3) -> transferred.parquet
  inspect         Print schema, head sample, and row count for any artifact
  peak-census     Peaks per MS2 spectrum for an mzML, as JSON: percentiles plus what each candidate `--top-peaks-ms2` cap would discard
  audit           Candidate audit: reconstruct per-candidate stage flags + earliest rejection reason across the artifact chain and write candidate_audit.parquet (sensitivity program, P0.3/P0.4). Non-destructive; reruns no compute
  report          Write peptides.tsv + proteins.tsv from a scored PSM table
  doctor          Check that the configured Python sidecar environments are usable
  help            Print this message or the help of the given subcommand(s)

Options:
      --threads <N>
          Maximum worker threads. Default: every core.

          Bounds the engine's rayon pool and is forwarded to the Python sidecars as `MUMDIA_NN_THREADS` and `OMP_NUM_THREADS` unless those are already set. Without this there was no way to bound MuMDIA at all except the undocumented `RAYON_NUM_THREADS`, which the engine never read and which does not reach the sidecars; on a shared machine that made a run antisocial. Note the NN rescore worker measured FASTER on 8 threads than on 32 (docs/13_sidecars.md).

      --log-level <LEVEL>
          Log level: `error`, `warn`, `info` (default), `debug`, or `trace`. Accepts any `RUST_LOG` filter, so `mumdia=debug,extract=trace` also works

  -v, --verbose...
          More detail: `-v` for debug, `-vv` for trace. Overridden by --log-level

  -q, --quiet
          Warnings and errors only. Overridden by --log-level

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

## Global flags

These 4 options are declared `global = true`, so they are accepted on
EITHER side of the subcommand: `mumdia --threads 8 extract ...` and
`mumdia extract --threads 8 ...` are equivalent and reach the same value.
They are removed from the per-subcommand blocks below to keep this document
readable, as is `-h, --help`, which every subcommand also accepts.

| Flag | Purpose |
|---|---|
| `--log-level <LEVEL>` | Log level: `error`, `warn`, `info` (default), `debug`, or `trace`. Accepts any `RUST_LOG` filter, so `mumdia=debug,extract=trace` also works |
| `-q, --quiet` | Warnings and errors only. Overridden by --log-level |
| `--threads <N>` | Maximum worker threads. Default: every core. Bounds the engine's rayon pool and is forwarded to the Python sidecars as `MUMDIA_NN_THREADS` and `OMP_NUM_THREADS` unless those are already set. Without this there was no way to bound MuMDIA at all except the undocumented `RAYON_NUM_THREADS`, which the engine never read and which does not reach the sidecars; on a shared machine that made a run antisocial. Note the NN rescore worker measured FASTER on 8 threads than on 32 (docs/13_sidecars.md). |
| `-v, --verbose...` | More detail: `-v` for debug, `-vv` for trace. Overridden by --log-level |
| `-h, --help` | Print help (see a summary with '-h') |

The same text as the binary prints it:

```text
Options:
      --log-level <LEVEL>
          Log level: `error`, `warn`, `info` (default), `debug`, or `trace`. Accepts any `RUST_LOG` filter, so `mumdia=debug,extract=trace` also works

  -q, --quiet
          Warnings and errors only. Overridden by --log-level

      --threads <N>
          Maximum worker threads. Default: every core.

          Bounds the engine's rayon pool and is forwarded to the Python sidecars as `MUMDIA_NN_THREADS` and `OMP_NUM_THREADS` unless those are already set. Without this there was no way to bound MuMDIA at all except the undocumented `RAYON_NUM_THREADS`, which the engine never read and which does not reach the sidecars; on a shared machine that made a run antisocial. Note the NN rescore worker measured FASTER on 8 threads than on 32 (docs/13_sidecars.md).

  -v, --verbose...
          More detail: `-v` for debug, `-vv` for trace. Overridden by --log-level

  -h, --help
          Print help (see a summary with '-h')
```

## Subcommands

One row per subcommand. `--config` says whether the subcommand reads a JSON
config file (see `docs/24_config_reference.md`); the purpose column is the
first sentence of the description, with the full text in the section below.

| Subcommand | `--config` | Purpose |
|---|---|---|
| [`convert`](#convert) | no | Read an mzML run into the normalized spectra artifact set |
| [`digest`](#digest) | yes | Fully-tryptic digest + decoy pairing -> peptides.parquet |
| [`peptidoforms`](#peptidoforms) | yes | Fixed+variable modification and charge enumeration -> peptidoforms.parquet |
| [`predict-frag`](#predict-frag) | yes | Spectral library: b/y m/z + predicted intensity + iRT -> fragment_library |
| [`prescan`](#prescan) | yes | Sequence-tag prescan: keep only modification-bearing candidates whose anchored trimers are observed in this run -> prescan_survivors.parquet. |
| [`search-seed`](#search-seed) | yes | Native broad DIA seed search over the fragment index -> seed_psms.parquet |
| [`rt-im-train`](#rt-im-train) | yes | Per-run RT calibration + windows -> run_windows.parquet, cal.json |
| [`extract`](#extract) | yes | Targeted 3D extraction (peak-major cascade) -> psms_extracted, chromatograms |
| [`features`](#features) | yes | Compute the minimal feature set -> features.parquet + PIN |
| [`compete`](#compete) | yes | Keep the best candidate per competition group -> psms_competed.parquet |
| [`rescore`](#rescore) | yes | Rescore + native target-decoy q-values -> psms_scored.parquet |
| [`quant`](#quant) | yes | Quantify identified peptides + roll up to protein groups |
| [`quant-lfq`](#quant-lfq) | no | Combine per-run quant tables into a protein-by-run matrix (cross-run LFQ) |
| [`run`](#run) | yes | Orchestrate the full MVP pipeline on one run and write a manifest |
| [`run-experiment`](#run-experiment) | yes | Experiment-wide orchestrator: run the per-file search chain over N runs, then one combined rescore, optional rescuable MBR transfer, per-run quant, and cross-run LFQ. |
| [`align`](#align) | yes | Cross-run RT alignment (experiment-level) -> alignment.parquet |
| [`mbr`](#mbr) | yes | Match-between-runs identification transfer (Stage D3) -> transferred.parquet |
| [`inspect`](#inspect) | no | Print schema, head sample, and row count for any artifact |
| [`peak-census`](#peak-census) | no | Peaks per MS2 spectrum for an mzML, as JSON: percentiles plus what each candidate `--top-peaks-ms2` cap would discard |
| [`audit`](#audit) | no | Candidate audit: reconstruct per-candidate stage flags + earliest rejection reason across the artifact chain and write candidate_audit.parquet (sensitivity program, P0.3/P0.4). |
| [`report`](#report) | yes | Write peptides.tsv + proteins.tsv from a scored PSM table |
| [`doctor`](#doctor) | yes | Check that the configured Python sidecar environments are usable |
| `help` | n/a | Print this message or the help of the given subcommand(s) |

17 of the 22 documented subcommands accept `--config`:
 `align`, `compete`, `digest`, `doctor`, `extract`, `features`, `mbr`, `peptidoforms`, `predict-frag`, `prescan`, `quant`, `report`, `rescore`, `rt-im-train`, `run`, `run-experiment`, `search-seed`.

5 do not, so every setting they use comes from their own flags:
 `audit`, `convert`, `inspect`, `peak-census`, `quant-lfq`.

## convert

```text
Read an mzML run into the normalized spectra artifact set

Usage: mumdia convert [OPTIONS] --mzml <MZML> --out-dir <OUT_DIR>

Options:
      --mzml <MZML>

      --out-dir <OUT_DIR>

      --max-spectra <MAX_SPECTRA>
          Limit spectra read (0 = all), for fast iteration

          [default: 0]

      --top-peaks-ms2 <TOP_PEAKS_MS2>
          Keep at most this many MS2 peaks in the normalized artifact (0 = all).

          This is an irreversible conversion-time cap that also affects extraction, features, and quantification. Use `search_seed.top_n_peaks` for a seed-only limit.

          [default: 0]

      --top-peaks-ms1 <TOP_PEAKS_MS1>
          Keep at most this many MS1 peaks per scan (0 = all)

          [default: 0]
```

Plus the 5 repeated flags removed above: see "Global flags".

## digest

```text
Fully-tryptic digest + decoy pairing -> peptides.parquet

Usage: mumdia digest [OPTIONS] --fasta <FASTA> --out <OUT>

Options:
      --fasta <FASTA>

      --out <OUT>

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## peptidoforms

```text
Fixed+variable modification and charge enumeration -> peptidoforms.parquet

Usage: mumdia peptidoforms [OPTIONS] --peptides <PEPTIDES> --out <OUT>

Options:
      --peptides <PEPTIDES>

      --out <OUT>

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## predict-frag

```text
Spectral library: b/y m/z + predicted intensity + iRT -> fragment_library

Usage: mumdia predict-frag [OPTIONS] --peptidoforms <PEPTIDOFORMS> --out-precursors <OUT_PRECURSORS> --out-fragments <OUT_FRAGMENTS>

Options:
      --peptidoforms <PEPTIDOFORMS>

      --out-precursors <OUT_PRECURSORS>

      --out-fragments <OUT_FRAGMENTS>

      --work-dir <WORK_DIR>
          Working directory for sidecar request/response files

          [default: sidecar_work]

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## prescan

```text
Sequence-tag prescan: keep only modification-bearing candidates whose anchored trimers are observed in this run -> prescan_survivors.parquet. Label-blind by construction, so it prunes search space without touching target-decoy exchangeability

Usage: mumdia prescan [OPTIONS] --ms2 <MS2> --isolation-windows <ISOLATION_WINDOWS> --lib-precursors <LIB_PRECURSORS> --run-windows <RUN_WINDOWS> --out <OUT>

Options:
      --ms2 <MS2>

      --isolation-windows <ISOLATION_WINDOWS>

      --lib-precursors <LIB_PRECURSORS>

      --run-windows <RUN_WINDOWS>
          Per-candidate RT bounds (candidate_id, rt_lo, rt_hi); a run_windows-shaped table

      --out <OUT>

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## search-seed

```text
Native broad DIA seed search over the fragment index -> seed_psms.parquet

Usage: mumdia search-seed [OPTIONS] --ms2 <MS2> --lib-precursors <LIB_PRECURSORS> --lib-fragments <LIB_FRAGMENTS> --out <OUT>

Options:
      --ms2 <MS2>

      --lib-precursors <LIB_PRECURSORS>

      --lib-fragments <LIB_FRAGMENTS>

      --out <OUT>

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## rt-im-train

```text
Per-run RT calibration + windows -> run_windows.parquet, cal.json

Usage: mumdia rt-im-train [OPTIONS] --seed-psms <SEED_PSMS> --lib-precursors <LIB_PRECURSORS> --out-windows <OUT_WINDOWS> --out-cal <OUT_CAL>

Options:
      --seed-psms <SEED_PSMS>

      --lib-precursors <LIB_PRECURSORS>

      --out-windows <OUT_WINDOWS>

      --out-cal <OUT_CAL>

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## extract

```text
Targeted 3D extraction (peak-major cascade) -> psms_extracted, chromatograms

Usage: mumdia extract [OPTIONS] --ms2 <MS2> --lib-precursors <LIB_PRECURSORS> --lib-fragments <LIB_FRAGMENTS> --run-windows <RUN_WINDOWS> --out-psms <OUT_PSMS> --out-chromatograms <OUT_CHROMATOGRAMS>

Options:
      --ms2 <MS2>

      --lib-precursors <LIB_PRECURSORS>

      --lib-fragments <LIB_FRAGMENTS>

      --run-windows <RUN_WINDOWS>

      --ms1 <MS1>
          Optional MS1 spectra for isotope-envelope features

      --mass-cal <MASS_CAL>
          Optional mass recalibration json (search-seed <seed>.masscal.json)

      --out-psms <OUT_PSMS>

      --out-chromatograms <OUT_CHROMATOGRAMS>

      --restrict-candidates <RESTRICT_CANDIDATES>
          Optional candidate allowlist (a prior run's psms.parquet): restrict extraction to these candidate_ids. For "gate first, then compete" - re-extract with a peak_claim strategy over only the gate-accepted survivors, keeping the two-pass profile map small

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## features

```text
Compute the minimal feature set -> features.parquet + PIN

Usage: mumdia features [OPTIONS] --psms-extracted <PSMS_EXTRACTED> --chromatograms <CHROMATOGRAMS> --out <OUT> --out-pin <OUT_PIN>

Options:
      --psms-extracted <PSMS_EXTRACTED>

      --chromatograms <CHROMATOGRAMS>

      --seed-psms <SEED_PSMS>
          Optional seed_psms for search-engine corroboration features

      --out <OUT>

      --out-pin <OUT_PIN>

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## compete

```text
Keep the best candidate per competition group -> psms_competed.parquet

Usage: mumdia compete [OPTIONS] --features <FEATURES> --out <OUT>

Options:
      --features <FEATURES>

      --out <OUT>

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## rescore

```text
Rescore + native target-decoy q-values -> psms_scored.parquet

Usage: mumdia rescore [OPTIONS] --out <OUT>

Options:
      --competed <COMPETED>...
          One or more competed feature tables

      --out <OUT>

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## quant

```text
Quantify identified peptides + roll up to protein groups

Usage: mumdia quant [OPTIONS] --psms-scored <PSMS_SCORED> --chromatograms <CHROMATOGRAMS> --out-peptide <OUT_PEPTIDE> --out-protein <OUT_PROTEIN>

Options:
      --psms-scored <PSMS_SCORED>

      --chromatograms <CHROMATOGRAMS>

      --out-peptide <OUT_PEPTIDE>

      --out-protein <OUT_PROTEIN>

      --out-fragment <OUT_FRAGMENT>
          Optional per-fragment area export (for ion-level directLFQ)

      --out-peak-bounds <OUT_PEAK_BOUNDS>
          Optional per-candidate peak-window diagnostic (candidate_id, lo_rt, hi_rt, width_s)

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## quant-lfq

```text
Combine per-run quant tables into a protein-by-run matrix (cross-run LFQ)

Usage: mumdia quant-lfq [OPTIONS] --out <OUT>

Options:
      --inputs <INPUTS>...
          One per-run table per run: peptide_quant.parquet for maxlfq, fragment_quant.parquet for directlfq

      --method <METHOD>
          `maxlfq` (peptide-level) or `directlfq` (ion/fragment-level)

          [default: maxlfq]

      --normalize <NORMALIZE>
          Cross-run normalization: `median_ratio` (default), `median`, or `none`

          [default: median_ratio]

      --out <OUT>
```

Plus the 5 repeated flags removed above: see "Global flags".

## run

```text
Orchestrate the full MVP pipeline on one run and write a manifest

Usage: mumdia run [OPTIONS] --mzml <MZML> --out-dir <OUT_DIR>

Options:
      --fasta <FASTA>
          FASTA to digest into the library. Omit when supplying a prebuilt library via --lib-precursors + --lib-fragments (library-input mode)

      --mzml <MZML>

      --out-dir <OUT_DIR>

      --lib-precursors <LIB_PRECURSORS>
          Library-input mode: consume a prebuilt precursor library (e.g. an imported DIA-NN speclib) instead of digesting --fasta. Requires --lib-fragments; skips digest/peptidoforms/predict-frag

      --lib-fragments <LIB_FRAGMENTS>
          Prebuilt fragment library paired with --lib-precursors

      --config <CONFIG>

      --profile <PROFILE>
          Named tuning preset applied on top of --config/defaults. "dia" = the validated DIA preset (Extended features, rolling-window apex, RT prior)

      --max-spectra <MAX_SPECTRA>
          [default: 0]

      --top-peaks-ms2 <TOP_PEAKS_MS2>
          Irreversible conversion-time MS2 cap (0 = all). Seed-only peak limiting is configured by `search_seed.top_n_peaks`

          [default: 0]
```

Plus the 5 repeated flags removed above: see "Global flags".

## run-experiment

```text
Experiment-wide orchestrator: run the per-file search chain over N runs, then one combined rescore, optional rescuable MBR transfer, per-run quant, and cross-run LFQ. Pass --mzml once per run (>= 2)

Usage: mumdia run-experiment [OPTIONS] --out-dir <OUT_DIR>

Options:
      --fasta <FASTA>

      --mzml <MZML>
          One per run; repeat the flag (>= 2 runs)

      --run-names <RUN_NAMES>
          Optional per-run labels / subdir names (default r0..rN-1)

      --out-dir <OUT_DIR>

      --lib-precursors <LIB_PRECURSORS>

      --lib-fragments <LIB_FRAGMENTS>

      --config <CONFIG>

      --profile <PROFILE>

      --max-spectra <MAX_SPECTRA>
          [default: 0]

      --top-peaks-ms2 <TOP_PEAKS_MS2>
          [default: 0]
```

Plus the 5 repeated flags removed above: see "Global flags".

## align

```text
Cross-run RT alignment (experiment-level) -> alignment.parquet

Usage: mumdia align [OPTIONS] --out <OUT>

Options:
      --seed-psms <SEED_PSMS>...
          One seed_psms.parquet per run; the first is the reference

      --out <OUT>

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## mbr

```text
Match-between-runs identification transfer (Stage D3) -> transferred.parquet

Usage: mumdia mbr [OPTIONS] --psms-scored <PSMS_SCORED> --out <OUT>

Options:
      --psms-scored <PSMS_SCORED>
          Experiment-wide scored_combined.parquet (has the `source` column)

      --psms-extracted <PSMS_EXTRACTED>...
          Per-run psms.parquet in `source` order (one per run)

      --out <OUT>

      --out-psms-scored <OUT_PSMS_SCORED>
          Optional augmented scored table: input scored with accepted transfers' q_value lowered + is_transferred flag (for quant/report with q_filter=psm_q)

      --frag [<FRAG>...]
          Optional per-run fragment_quant.parquet (source order) for the fragment-consensus guard (needs mbr.consensus_corr_min > 0)

      --config <CONFIG>
```

Plus the 5 repeated flags removed above: see "Global flags".

## inspect

```text
Print schema, head sample, and row count for any artifact

Usage: mumdia inspect [OPTIONS] <ARTIFACT>

Arguments:
  <ARTIFACT>
```

Plus the 5 repeated flags removed above: see "Global flags".

## peak-census

```text
Peaks per MS2 spectrum for an mzML, as JSON: percentiles plus what each candidate `--top-peaks-ms2` cap would discard.

The pre-flight for a decision the documentation says must be made per acquisition. Reading it before setting a cap is the difference between bounding peak volume and deleting fragment evidence from most spectra.

Usage: mumdia peak-census [OPTIONS] --mzml <MZML>

Options:
      --mzml <MZML>

      --max-spectra <MAX_SPECTRA>
          Stop after this many spectra from the head of the file (0 = all)

          [default: 0]
```

Plus the 5 repeated flags removed above: see "Global flags".

## audit

```text
Candidate audit: reconstruct per-candidate stage flags + earliest rejection reason across the artifact chain and write candidate_audit.parquet (sensitivity program, P0.3/P0.4). Non-destructive; reruns no compute

Usage: mumdia audit [OPTIONS] --lib-precursors <LIB_PRECURSORS> --psms-extracted <PSMS_EXTRACTED> --competed <COMPETED> --psms-scored <PSMS_SCORED> --out <OUT>

Options:
      --lib-precursors <LIB_PRECURSORS>
          Library precursors parquet (the full candidate search space)

      --psms-extracted <PSMS_EXTRACTED>
          psms parquet from `extract`

      --competed <COMPETED>
          competed parquet from `compete`

      --psms-scored <PSMS_SCORED>
          scored parquet from `rescore`

      --out <OUT>
          Output candidate_audit.parquet

      --q <Q>
          Precursor q-value threshold for passed_precursor_fdr / reported

          [default: 0.01]

      --run-id <RUN_ID>
          Run identifier stamped on every row

          [default: run]

      --entrapment-substr <ENTRAPMENT_SUBSTR>
          Optional protein substring marking entrapment candidates (e.g. _HUMAN)

          [default: ""]
```

Plus the 5 repeated flags removed above: see "Global flags".

## report

```text
Write peptides.tsv + proteins.tsv from a scored PSM table

Usage: mumdia report [OPTIONS] --psms-scored <PSMS_SCORED> --out-dir <OUT_DIR>

Options:
      --psms-scored <PSMS_SCORED>

      --out-dir <OUT_DIR>

      --peptide-quant <PEPTIDE_QUANT>

      --protein-quant <PROTEIN_QUANT>

      --q <Q>
          Reported q threshold. Defaults to `quant.q_threshold` from `--config` when that is given, otherwise 0.01. An explicit value always wins

      --config <CONFIG>
          Read `quant.q_threshold` from this config, so a standalone report uses the same threshold as the `run` that produced the table.

          Without it, a config setting `quant.q_threshold = 0.05` yielded 0.05 from `run` and 0.01 from `report` on the same scored table, silently.
```

Plus the 5 repeated flags removed above: see "Global flags".

## doctor

```text
Check that the configured Python sidecar environments are usable

Usage: mumdia doctor [OPTIONS]

Options:
      --config <CONFIG>

      --json
          Emit the report as JSON on stdout instead of prose on stdout.

          For a caller that has to act on the result rather than read it: the desktop application renders one row per role and offers to install what is missing, which means it needs the modules and versions as data, not a paragraph to regex. The exit status is unchanged.
```

Plus the 5 repeated flags removed above: see "Global flags".
