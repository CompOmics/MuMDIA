# Changelog

All notable changes to MuMDIA are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`0.1.0` is the first tagged release of the Rust engine. The superseded
Python implementation remains available at the tag `legacy-python-v1`.

`0.3.0` raises the DeepLC floor to 4.4.0, so every environment must be rebuilt
before upgrading. It also adds multi-head retention-time calibration, off by
default.

`0.2.0` is a minor rather than a patch release: it changes behaviour a user can
see. The default MS2PIP model is `HCDch2`, configurations with out-of-range
numeric values are refused at load instead of running, both TSV reports gain two
columns, the candidate-audit rejection code `NO_PEAK_GROUP` is
`DID_NOT_SURVIVE_EXTRACTION`, a candidate no predictor covered is dropped rather
than given a substitute value, and a stage refuses to write its output over one of
its own inputs.

Two things are versioned independently of this file and matter when reading old
results: the per-artifact Parquet schema versions in
`rust/mumdia/crates/mumdia-core/src/schema.rs`, and the feature-set identity
`classifier_feature_schema_id`, which is a hash of the active feature list rather
than a number. Both are recorded in every run's `manifest.json`.

## [Unreleased]

### Fixed

- Nine desktop test fixtures used a fixed temporary directory and deleted it on entry, so
  two `cargo test` runs on one machine raced and one lost its files mid-test. They are
  unique per process, like every other temporary fixture in the workspace (`docs/14`). Hit
  twice while preparing releases; two concurrent runs now both pass.

## [0.3.0] - 2026-09-09

Multi-head retention-time calibration and the DeepLC 4.4.0 floor it needs, plus
the sha2 0.11 bump. `rt_im_train.multihead_calibration` is off by default;
turning it on measured +4.8% stripped peptides on the HYE AIF benchmark and
+14.3% on HYE Astral, at an unchanged empirical decoy fraction.

### Added

- `rt_im_train.multihead_calibration` (default 0, off): calibrate the DeepLC base model
  against this run's confident seed PSMs across its N best-correlating LC-setup heads,
  instead of fine-tuning. `deeplc.predict` returns one of the model's 6,543 heads, the
  setup its `DEFAULT_TASK_NAME` names, on that setup's gradient; the per-run LOESS then
  maps that column onto observed retention time, and a smooth increasing curve can stretch
  and bend the axis but cannot reorder two peptides, so that one setup's elution order
  survives into the calibrated result. Different chromatography reorders peptides.
  `MultiHeadRidgeCalibration` ranks every head against the run's own anchors,
  spline-calibrates the best ones and ridge-combines them, and never fits more head
  weights than half the reference. It occupies the fine-tune's slot in the chain, needs
  the same anchors, and validation refuses both at once. Off by default: no entrapment or
  second-acquisition measurement exists for it in this repository yet. Measured on HYE
  B01's own anchors, split in half: the default head with the best possible monotone map
  scores 79.5 s median error, the multi-head combination 28.6 s. The monotone oracle,
  fitted on the scoring half itself, also scores 79.5 s, so on that run the LOESS is
  already doing everything a monotone map can and the whole gap is ordering rather than
  fit quality (`docs/08_rt_im_train.md` section 4d). The best head there is 1229, not the
  default 938. Measured end to end on both HYE acquisitions, four pooled six-file
  experiments from one binary differing only in this field: AIF 80,842 to 84,725 stripped
  peptides at 1% (+4.8%), Astral 102,942 to 117,652 (+14.3%), protein groups +2.4% and
  +7.1%, at an empirical decoy fraction of 0.0100 in all four arms. On Astral the window
  narrows from 84 s to about 20 s and half as many candidates reach the rescore, so the
  gain is interference removed rather than a loosened threshold. It costs 1.4x to 1.7x
  wall clock, because the calibration re-predicts the library per run rather than once per
  experiment. Still off by default: entrapment has not been run, and CLAUDE.md requires an
  empirical null as well as two acquisitions.

### Changed

- The DeepLC floor is 4.4.0, raised from 4.1.1, for
  `deeplc.calibration.MultiHeadRidgeCalibration` and the lazy head source that
  `predict_and_calibrate` uses to avoid materialising all 6,543 head columns. 4.1.1 has
  neither: its `calibrate` picks a single best-correlating head. `mumdia doctor`, the
  interpreter discovery, `sidecar::require_deeplc_version` and both worker scripts enforce
  it from the one constant, and every shipped environment pins `deeplc==4.4.0`.
- The desktop application builds against `sha2` 0.11. Its `finalize()` returns
  `hybrid_array::Array` rather than the old `GenericArray`, which does not implement
  `LowerHex`, so the four `format!("{:x}", ..)` sites move to a `components::hex`
  helper. The strings are unchanged, and a test pins them against the canonical
  SHA-256 vectors, because they are compared with published checksums and used as
  cache directory names.

## [0.2.0] - 2026-09-08

Three code reviews and their fixes (`docs/29_code_review_2026-09-07.md`,
`docs/30_code_review_2026-09-08.md`, `docs/31_code_review_2026-09-08_full.md`),
the MS2PIP charge-2 library path, and the desktop settings work. Validated on the
six-file HYE benchmark end to end: 80,803 experiment-wide stripped peptides at 1%
at a 1.00% decoy fraction.

### Added

- MS2PIP charge-2 fragment predictions reach the library. The worker emits a
  `frag_charge` column from the `b2`/`y2` series that the `*ch2` models (`HCDch2`,
  `CIDch2`) predict, and `predict-frag` keys its lookup on `(ion, ordinal, charge)`.
  When the model emitted charge-2 series, every fragment carries a model intensity on
  one scale; single-charge models keep the previous behaviour (native heuristic for
  charge-2 fragments, each charge group normalised to its own peak), so existing
  `HCD2021` libraries are unchanged. Why: on the HYE FASTA library built with `HCD2021`,
  78.6% of the top-6 fragments were charge-2 heuristics and the seed search separated
  targets from decoys no better than chance (41.6% decoys among the top 1,000 seed
  scores, against 0% with the DIA-NN library on the same spectra), so retention-time
  calibration found no anchors and the run proceeded with an unbounded RT window.
  Measured on that run with 12 fragments: `HCDch2` 19,308 confident seeds (0% decoys
  in the top 5,000; fragment charges 79% / 21%), `HCD2021` with charge-2 fragments
  only from precursor charge 3 14,412, the DIA-NN library 21,856.

- `peptides.tsv` and `proteins.tsv` (single-run and experiment-wide) carry
  `is_transferred` and `transfer_q`, the acceptance basis of a match-between-runs row:
  a transferred row keeps its grouped q next to the transfer q it was accepted at, a
  tighter report threshold does not revoke a transfer that passed `mbr.q_transfer`, and a
  protein group admitted through a transferred row carries the flag. The MBR worker's
  augmented scored table gains `transfer_q` for it (docs/29 #19).
- `experiment_manifest.json` records the resolved `config_json` next to its hash, the
  `model_identities` that produced the artifacts (RT source, fragment predictor, the
  classifier that actually ran, feature schema, MBR strategy), the configured and
  effective `quant.q_filter`, and input hashes taken at the start of the run rather than
  at its end (docs/29 #15).
- Dependabot covers the desktop application's Cargo dependencies (`/desktop`), and CI
  audits `desktop/Cargo.lock` with `cargo audit` next to the engine's lockfile, with one
  documented ignore (RUSTSEC-2024-0429: glib 0.18 through Tauri 2's gtk 0.18).

### Changed

- Model identities carry the installed predictor versions: `deeplc-4.1.1-base`,
  `deeplc-4.1.1-finetuned`, `ms2pip-4.2.0-HCDch2` in the library report and the manifests,
  in place of the family labels `deeplc-4.0-mt` and `ms2pip-<model>` (docs/30).
- With nothing to transfer, the MBR worker writes a transfer table with its ten columns and
  zero rows, and the requested augmented scored table with every row unflagged, instead of
  a one-column placeholder and no scored table (docs/30).
- The candidate-audit rejection code `NO_PEAK_GROUP` is `DID_NOT_SURVIVE_EXTRACTION`
  (`RejectionReason::DidNotSurviveExtraction`). The audit assigns it to every candidate
  with no extracted row, and `extract` does not write the per-candidate table that would
  separate presence, matched-fraction and gate failures, so the old name claimed a cause
  the audit cannot see (docs/29 #16). The audit table has no versioned schema; the
  metrics JSON gains `q_unit`.
- MS2PIP 4.2.0 in every shipped environment (`env/docker-rescore.yml`,
  `env/console-ms2pip-requirements.txt`), and `env/mumdia-deeplc.yml` now carries
  `ms2pip==4.2.0` too, so one host environment serves DeepLC, MS2PIP and the `nn_torch`
  rescorer and `configs/examples/fasta-sidecars.json` runs from the shipped
  specifications with its interpreters at `auto`. Before this no host specification
  provided MS2PIP at all. 4.0.0 needed `sqlalchemy<2` and could not share an
  environment with DeepLC; 4.2.0 is the version behind the FASTA-mode measurements in
  `docs/28` section 22. The image's smoke test imports `ms2pip` in the rescore
  environment.
- `predict_frag.ms2pip_model` defaults to `HCDch2` (was `HCD`), for the reason above.
  `configs/examples/fasta-sidecars.json` and the image's `config.dia.json` set
  `top_n_fragments: 12` explicitly, the count the DIA-NN library ships and the one the
  measurement used; the engine default stays 6, because the native predictor was not
  re-measured.
- FASTA mode measured end to end on both HYE acquisitions with that library
  (`docs/28_feature_selection_analysis.md` section 22): six pooled AIF runs 69,091
  experiment-wide peptides at 1% in 45 min of search (imported DIA-NN library: 72,344),
  six pooled Astral runs 85,644 in 34 min; library prediction 54 min once per FASTA
  (DeepLC 19, MS2PIP 35 at 32 processes). DIA-NN 2.2.0 library-free on the same files:
  61 and 67 min including its library.

- The MS2PIP worker uses the engine's thread count for its process pool (passed as a
  fourth argument by `run_ms2pip`) instead of capping itself at eight processes, and
  assembles its output from numpy arrays per chunk instead of four per-fragment Python
  lists. Measured on the 9.8M-peptidoform HYE library (one missed cleavage, 7-30, charges
  2-3, one oxidation), the old worker ran on six to eight cores of the 32 requested and
  held 13 GB of Python objects before writing. Predictions are unchanged: the same
  rows in the same order, the same float64 arithmetic stored as float32.

### Fixed

- Code review A, data integrity (`docs/29_code_review_2026-09-07.md`, findings 1, 2, 4,
  9, 11, 17, 18, 21):
  - quant's refusal of a pooled scored table read `source` as i32 while rescore writes
    it as u32, treated the type error as "no such column", and so never ran on the
    engine's own output; a pooled table quantified against one run's chromatograms
    produced one identical row per run. The column is now read in its declared type and
    a present column of another type is an error (#1).
  - The streamed library loader checked fragment `mz` and `predicted_intensity` for
    finiteness on the physical Arrow buffers, which ignore the validity bitmap, and then
    turned NULL cells into NaN, NULL names into `""` and a NULL `candidate_id` into
    candidate 0. Every required fragment column now rejects NULLs before its values are
    read, through the same contract the typed getters enforce, with a fixture per
    column (#2).
  - `AtomicPath` removed the destination before renaming, so a failed publication had
    already destroyed the previous artifact and readers saw a window with no file; two
    writers for one destination in one process shared a temporary name. The rename
    replaces in place on every platform, the temporary name carries a counter, and the
    failure case is tested (#4).
  - N-terminal methionine excision was skipped whenever the Met-retained peptide fell
    outside the length window, so an N-terminal peptide of `max_len + 1` residues yielded
    nothing although its excised form was in range. Both forms are judged on their own
    length (#9).
  - `rescore.max_feature_matrix_gib` was checked after the matrix had been filled,
    against an estimate of the old `Vec<Vec<f64>>` layout, so it could neither prevent the
    allocation nor describe it; it is now checked from the parquet footers and the
    selected feature count before allocation, on the flat f32 layout, with checked
    arithmetic (#11).
  - A candidate DeepLC or MS2PIP returned nothing for received a substitute (iRT 0.0, or
    the native intensities under an MS2PIP model identity). It is now dropped together
    with its paired decoy or target, the counts are in the library report and a warning,
    and a worker id that was not requested or appears twice is an error (#17).
  - Numeric configuration domains are validated at load: thresholds and fractions within
    their unit interval, positive multipliers and widths, ordered `min_len <= max_len` and
    `charge_min <= charge_max`, counts at least one, with documented zero meanings kept
    (#18). `quant.q_threshold = -0.1`, `rt_im_train.rt_window_multiplier = -1.0` and
    `rescore.train_margin_frac = 2.0` were accepted before.
  - `ci/gen_config_reference.py` and `ci/check_workflows.py` scan the files git tracks
    rather than everything on disk, so scratch copies beside the sources no longer enter
    the generated reference or the workflow check (#21).
- Code review B, workers (`docs/29`, findings 3, 6, 7, 8, 12, 20):
  - The entrapment worker skipped a training fold whose training side held one class
    and then scored that fold's held-out rows with the final model, trained on those
    very rows, so in-sample scores entered the entrapment FDR. A single-class training
    fold is now an error that names the condition; the final model scores the decoys
    only (#3).
  - The MBR worker printed an "empirical decoy fraction" over accepted transfers, a
    population that cannot contain a decoy, and computed the transfer q as
    `null / targets`, which is exactly 0 for any pool no permuted residual undercuts, so
    a three-candidate pool was accepted whole at 1%. The q uses the engine's `+1`
    pseudocount and the summary names the permuted-null draws inside the accepted window
    instead (#6, #7).
  - With `extract.retain_top_peaks` above 1 the MBR worker measured the transfer on the
    last competed peak of a candidate, not the one rescore selected and quant integrates;
    it now joins `selected_peak_rank` and falls back to the highest `prelim_score` peak
    (#8).
  - `augment_library.py` gave every added precursor a fresh `base_peptide_id`, so an
    added charge state or modform of an existing peptide left its peptide's competition
    group and fold; added forms of existing sequences keep the imported id (#12).
  - `bench/feature_selection/fs_lib.py` hashed the peptide with its `DECOY_` prefix for
    fold assignment, splitting pairs; it hashes the base sequence, and every benchmark
    row records the code revision, fold rule, feature count, seed and training recipe
    (#20).
- Code review C, desktop and output ownership (`docs/29`, findings 5, 13, 14):
  - A repeated Start in the desktop application could launch a second engine into the
    same results folder: the start flow had several awaits and no in-progress guard, and
    the backend launched every request. A Start is now refused while one is in progress
    or while the run the interface follows is still running, and the backend reserves a
    run's results folder (by canonical path) before spawning the engine and releases it
    when the run's end is published, so a request for an active folder is refused with
    the owning run named (#5).
  - `run-experiment --run-names` compared names case-sensitively, so `RunA` and `runa`
    passed and addressed one directory on Windows, macOS and most network shares. Names
    that differ only in case are rejected on every platform (#5).
  - Desktop preflight asked the engine about converters without the request's
    configuration, so a converter named in `convert.thermo_raw_parser` or
    `convert.msconvert` was reported missing and the search refused, and it required
    ThermoRawFileParser for Thermo `.raw` even when msconvert, the engine's own fallback
    for a parser left at `auto`, was present. The probe now carries the configuration and
    the verdict follows the engine's rule: only msconvert present runs, with a note; an
    explicitly configured parser that is missing blocks, as it errors in the engine (#13).
  - A cancelled desktop run could be published as failed. `cancel` and the process
    waiter both wrote the terminal status and whichever ran second won, while the
    cancellation flag was written and never read. The waiter is now the only writer: it
    reads the intent after reaping the engine and publishes `cancelled`, `done` when the
    engine had already finished, or `failed`; until then the run shows "Stopping" (#14).
- The committed CycloneDX SBOM, which ships in every release archive, referenced
  `pkg:cargo/mumdia-core` and `pkg:cargo/mumdia-io` in its dependency graph while
  excluding them from its component list, so it failed validation and `--check`
  regenerated the same broken document. Only the application crate is excluded now,
  because `metadata.component` describes it (`docs/31` F11).
- The desktop DIA-NN cache-key test used a fixed temporary directory and deleted it on
  entry, so two `cargo test` runs on one machine raced and one lost its fixture
  mid-test. It is unique per process, like every other temporary fixture in the
  workspace (`docs/14`).
- Code review F, whole-repository review (`docs/31_code_review_2026-09-08_full.md`,
  F1 to F10):
  - `prescan` read the infinite-bounds sentinel that `rt-im-train` writes for "calibration
    unavailable, search the whole gradient" as "cannot be screened" and dropped the
    candidate, so a run with no confident seeds discarded the entire library and exited 0
    with a zero-row survivors table. An unbounded window now screens over the whole
    gradient, a candidate with no window row is treated the same, both are counted, and
    screening every candidate away is an error (F1).
  - A present-but-wrong-typed `is_transferred` was swallowed as "no transfers", silently
    removing every match-between-runs identification from `peptides.tsv` and
    `proteins.tsv` while the parquet still carried them. Present columns are read in their
    declared type and a mismatch is an error; only an absent column falls back (F2).
  - `sidecar::resolve_script` tried the working directory before the directory beside the
    binary, the ordering `python::resolve_script_dir` was hardened against, so a `scripts/`
    directory inside an untrusted dataset could have its worker executed. An absolute
    directory is taken as given, then the executable's directory, then `<exe>/scripts`, and
    the working directory last (F3).
  - `Loess::predict` indexed before the start of its grid for a non-finite query, so one
    library row with a null `predicted_irt` could abort or misread memory at rt-im-train.
    It returns NaN, and rt-im-train treats a non-finite library iRT as "no calibrated RT"
    and counts those rows (F4).
  - The rescorer's in-memory TSV backend standardised with median/IQR while the parquet and
    streaming backends used mean/std, so the same pool scored differently depending on
    `rescore.handoff` and on the 4 GB streaming threshold. All three use mean/std, which
    leaves the shipped parquet default and every published benchmark unchanged. The
    `MUMDIA_NN_FOLD_KEYS` companion is length-checked instead of being sliced short, which
    used to leave the tail rows unscored at a fabricated mid-rank score, and the estimate
    that picks the backend counts feature columns by name (F5).
  - `refuse_output_over_input` was wired into two of eighteen stages, so
    `compete --features f.parquet --out f.parquet` replaced the widest artifact of the run
    with the competed subset at exit 0. It now guards every output of `search-seed`,
    `rt-im-train`, `extract`, `features`, `compete`, `rescore`, `quant` and `audit` (F6).
  - The LOESS boundary extrapolation slope introduced in the previous package was the
    pointwise local slope at the sparsest, most one-sided point of the fit: unbounded, free
    to be negative, and multiplying an unbounded distance. It is the secant of the fitted
    curve over its end decile, clamped non-negative and to at most four times the global
    slope, and the test uses noisy anchors rather than a noiseless quadratic (F7).
    Measured on HYE B01 against the previous behaviour, same library and settings: 48,533
    stripped peptides at 1%, 53,127 PSM-q 1% targets, 6,519 protein groups and 1,961,800
    extracted rows in both arms, identical to the row. The two extrapolations agree
    wherever the anchors are dense and differ only outside the anchor range.
  - A desktop stop arriving between the reap and the end of `publish_exit` could pass a
    recycled process id to the tree kill. The waiter retires the id the instant `wait`
    returns, before it reads the output directory (F8).
  - The conversion lock added in the previous package spun without pause on an undeletable
    stale lock, mistook clock skew and a peer's partial file for evidence about its own
    holder, could be held by two processes at once, and left every interrupted conversion's
    partial mzML behind for ever. Take-overs are bounded and paced, the holder is
    identified by a token it reads back, a future modification time counts as fresh, the
    partial-file probe matches this destination only, and abandoned partials are swept
    under the lock (F9).
  - Dropping an unpredicted candidate with everything sharing its pair key also removed
    positional isomers that predicted correctly, bounded only by the library being emptied.
    The direct misses and the collateral are counted separately and exceeding 2% of the
    library is an error naming the sidecar. The key stays position-free deliberately: a
    positional key would stop matching a reverse decoy to its target, trading a sensitivity
    defect for an FDR one (F10).
- Code review E, follow-up (`docs/30_code_review_2026-09-08.md`, R1 to R9):
  - Enabling DeepLC fine-tuning with its own defaults was rejected at load, because the
    documented automatic batch size is `finetune_batch = 0` and the new validation demanded
    a positive batch. Only the epoch count has a lower bound now (R1; a regression from
    review A).
  - `run-experiment --run-names` accepted `a` and `a.`, one directory on Windows, and the
    second run overwrote the first with exit 0. Names ending in a dot or a space, containing
    `<>:"|?*` or a control character, or naming a Windows reserved device are rejected on
    every platform before anything is written (R2).
  - A desktop stop could sweep temporary files that belonged to the next run in the same
    folder: cancellation swept after the reservation had been released, and a stop on a
    finished run swept as well. Cancellation is now intent and kill only and inert once the
    run is terminal; the sweep happens in the waiter, after the reap and before the release,
    and a stop still killing finishes before the folder changes hands (R3).
  - Two searches converting the same vendor file concurrently shared one temporary output
    and one could publish the other's bytes. Each conversion writes a unique partial file
    under a lock beside the destination; a concurrent converter waits and reuses the
    result (R4).
  - Domain checks for the numeric settings review A left unchecked: `mbr.q_anchor`,
    `min_anchor_runs`, `extract.min_matched_fraction`, `features.bound_peak_fraction`,
    `quant.reliable_q` and the remaining fractions, correlations, tolerances and counts (R5).
  - The DeepLC fine-tune and re-prediction worker zipped predictions with peptidoforms
    without checking the count and silently kept the imported iRT for anything missing. A
    count mismatch is an error; rows that keep their imported value are counted in
    `<lib_out>.summary.json` and the engine warns when there are any (R6).
  - The audit's `reported` flag repeated the precursor gate, so it could read `true` next to
    `FAILED_PEPTIDE_FDR`, and a decoy could be `REPORTED`; the flag now follows the reason,
    a decoy past both gates is `REMOVED_DURING_REPORTING`, and a present `precursor_q` of the
    wrong type is an error rather than a fallback (R7).
  - The desktop results-folder reservation compared exact folders only, so a search into a
    child of an active experiment's folder was allowed; ancestors and descendants are
    refused, siblings are not (R8).
  - The Windows debug binary overflowed its 1 MiB main-thread stack on `--version`; the CLI
    runs on a thread with a 256 MiB reservation and an integration test runs the built
    binary (R9).
- Code review D, calibration, provenance, reporting (`docs/29`, findings 10, 15, 16, 19):
  - LOESS retention-time calibration switched to the global least-squares line the
    moment a query left the anchor range, while the grid just inside used the local fit,
    and the two need not agree: on `y = 200 + 10x^2` (span 0.3) the prediction jumped
    from 193.4 at `x = 1e-6` to 38.3 at `x = 0`, and from 1173.5 to 1018.4 at the top,
    about 155 s discontinuities that misplaced gradient-edge peptides relative to their
    extraction window. The map now continues the boundary local fit (its value and
    slope) outside the range, and is continuous at both ends; the global line remains
    only the degenerate fallback (#10). Measured on HYE B01 with the imported DIA-NN iRT
    as the RT source and `native_tda`: 45,946 stripped peptides at `peptide_q_value` 1%
    before, 45,957 after, at an unchanged 1.0% PSM-level decoy fraction and 6,410 protein
    groups in both arms. 208,130 of 10.88 M candidates (1.9%) received a different window,
    194,698 of them with iRT above the anchor range, which the global line had placed
    past the end of the 9,000 s run; the local fit places them at 8,578 to 9,100 s, and
    extract accepted 454 more rows from them. A second pair on the DeepLC 4.1.1
    re-predicted precursor table (`w_rt` 414 s against 691 s): 48,533 stripped peptides in
    both arms, PSM-q 1% targets 53,124 against 53,127 at the same decoy fraction, 6,519
    protein groups in both, 0.2% of candidates with a different window. Neutral on both RT
    sources, which is what a boundary correction should be.
  - The candidate audit's `passed_precursor_fdr` gate and `FAILED_PRECURSOR_FDR` reason
    read the PSM `q_value`; they read `precursor_q`, the unit the label names, with the
    PSM q as a recorded fallback on tables without it. A pooled scored table (several
    `source` values) is refused, because the audit keys on `candidate_id` and would
    attribute the last run's fate to every run (#16).
- Desktop: the digest fields on the Search screen (missed cleavages, peptide length,
  charge range, carbamidomethyl, oxidation) now reach the engine on the built-in
  library path. They were read only by the DIA-NN library build, so with the built-in
  predictors the engine digested with the preset's values; the block also showed the
  two modification checkboxes twice under the same ids. The run's configuration is now
  the selected preset with the fields merged on top (`derive_config`), validated by
  the engine before the search starts.
- Desktop: the settings editor starts from the preset selected on the Search screen
  instead of from the engine defaults, so "Save and use" writes the preset plus the
  edits rather than silently dropping the preset's predictor, rescorer and interpreter
  choices. Engine fields the schema marks `not yet wired` (the match-between-runs
  tiers) are labelled as such and cannot be edited; list-valued settings display and
  accept JSON.

## [0.1.1] - 2026-09-07

### Fixed

- The 0.1.0 desktop installers (`.msi`, `.AppImage`) shipped without the Python
  workers: `binaries/scripts/` held its README alone, because the release workflow
  staged the engine and `uv` but never copied `scripts/*.py`, and the Tauri resource
  glob was satisfied by the README. The engine accepts a script directory only when it
  holds a worker file, so the installed application had no sidecars and DeepLC, the
  neural rescorer, mokapot and the DIA-NN import failed at the point of use. The
  workflow now stages the workers and, before uploading, unpacks every bundle it built
  (`msiexec /a`, `--appimage-extract`) to assert the console, the engine, `uv` and each
  worker are inside and the bundled engine runs. Found by unpacking the published
  installers; the Setup page installs the Python environment, not these files.
- A SCIEX `.wiff` without its `.wiff.scan` companion fails in msconvert with
  `Could not open data stream. Is a required 'scan' file missing?`, which names no
  file. The engine now appends the missing companion's path to that error. Measured on
  PRIDE-archived `.wiff` files that had been downloaded without their companions.

## [0.1.0] - 2026-09-06

### Pre-release audit (2026-08-28)

A six-way audit of the tree ([`docs/25_release_readiness_review.md`](docs/25_release_readiness_review.md))
found three release blockers and about thirty further defects. All three blockers
and most of the rest are fixed; that document's status section records what is
deliberately still open. The entries below fold into the sections that follow.

**Second audit (2026-08-28, external).** A second review of the same tree raised
four further blockers, two of which were defects in the first audit's own output: a
generated document quoting a value from a test as though the engine set it
(`ci/gen_config_reference.py` did not skip `#[cfg(test)]`), and a `build.rs` that
guessed at the git directory and stamped a stale commit into every manifest. Both
fixed, the second verified across a commit. The other two were release mechanics and
are covered under Added below: a release archive that could not run the verification
its own documentation prescribes, and a `v*` tag that could publish any commit with
no CI behind it.

**Two changes require re-measurement rather than only review.**

- Three defaults changed on correctness grounds, not from a count:
  `extract.apex_evidence_rank` to `true` (the legacy apex silently selected the
  lowest-RT qualifying scan when none of the top-K predicted fragments was
  observed anywhere) and `features.emit_pin` to `false` (no stage reads the file;
  it is a ~5.4 GB write per run). `extract.gate_min_score` was also briefly changed
  to `0.6` and then measured back to `0.2`: 0.6 costs 4.4% of peptides for
  `native_tda` and 4.7% for `nn_torch` at an unchanged decoy fraction, because the
  gate sweep that motivated it predates the current defaults and its optimum has
  moved to the loose end for both rescorers.
- The `nn_torch` CV fold is keyed on `base_peptide_id` supplied by the engine
  rather than on a hash of the peptidoform, so a target and its paired decoy now
  share a fold as `percolator_lite` always did and as `docs/11` always claimed.

Both move `nn_torch` counts, and the entrapment fixes below invalidate any
entrapment measurement previously taken through the native rescorer.

**Breaking: interface renames with no compatibility aliases.** A tag freezes
these, so they are free now and a major bump later. CLI:
`--library-precursors`/`--library-fragments` to `--lib-`, `--out-chrom` to
`--out-chromatograms`, `--scored` to `--psms-scored`, `--out-scored` to
`--out-psms-scored`, `--psms` to `--psms-extracted`, and `--seed`/`--seeds` to
`--seed-psms`. Config: `extract.min_frag_corr` to `extract.gate_min_score` (it is
not a correlation under any `gate_mode`), and `compete.group_by = "precursor"` to
`"base_peptide"` (it keys on the stripped sequence). An old name now fails with
the offending key and the valid alternatives listed. Local configs need:

```bash
sed -i 's/"min_frag_corr"/"gate_min_score"/; s/"group_by": *"precursor"/"group_by": "base_peptide"/' config.*.json
```

### Added

- **Several files pool by default.** `run` given more than one `--mzml` dispatches to
  `run-experiment`: files provided together are rescored together (one pooled FDR),
  quantified per run and aligned across runs (MaxLFQ). Searching files separately is
  the opt-in, one `run` per file.
- **Experiment-wide report.** `run-experiment` writes `peptides.tsv` and `proteins.tsv`
  at the experiment root, selected on the experiment-wide `peptide_q_value` and
  `pg_q_value`, with an `n_runs` column (per-run acceptances on `run_psm_q`) and one
  `quantity_<run>` and `lfq_<run>` column per run. `mumdia report --experiment-dir`
  rewrites the pair at another threshold. No per-run TSVs are written, because the
  grouped q columns go to each group's experiment-wide winner only.
- **Vendor formats.** A vendor file given as `--mzml` is converted to mzML first
  (`convert`, `run`, `run-experiment`, `peak-census`): Thermo `.raw` through
  ThermoRawFileParser (or msconvert), Bruker and Agilent `.d`, SCIEX `.wiff` and
  Waters `.raw` through ProteoWizard `msconvert`. Converters are located
  (`convert.thermo_raw_parser`, `convert.msconvert`, both `auto`, or
  `MUMDIA_THERMO_PARSER` / `MUMDIA_MSCONVERT`), never shipped. The mzML is written
  beside the input (or into the output directory when that is not writable) through
  a `<name>.partial.mzML` temporary and reused on later runs when newer than its
  source (`convert.reuse_converted`). `mumdia doctor` reports both converters and does
  not fail for their absence. Only Thermo is exercised end to end (a 3.7 GB Astral
  run, 6:40 through ThermoRawFileParser 2.0.0); the four msconvert formats are wired
  and unverified, and ion mobility is discarded.
- **Library retention time from the DeepLC base model.** `rt_im_train.library_irt`
  (`auto`, `library`, `deeplc`) re-predicts an imported library's iRT with the DeepLC
  base model once per experiment when a DeepLC interpreter is configured, because the
  imported DIA-NN iRT is the worst RT source measured: AIF 10,416 peptides against
  10,015 raw; HYE B01 58,842 against 56,556 raw over three NN seeds. The optional
  fine-tune (`finetune_deeplc`) remains available and is still +2.4% on HYE.
- **DeepLC 4.1.1 is a floor.** `mumdia doctor`, the sidecar launch
  (`sidecar::require_deeplc_version`) and both worker scripts refuse an older DeepLC
  (`mumdia_core::constants::MIN_DEEPLC_VERSION`), because the default
  prediction-plus-calibration workflow is only sound on a base model that does not
  memorise its anchors.
- **Rescore handoff and training recipe.** `rescore.handoff = parquet` replaces the
  TSV handoff to the Python worker (rescore peak 29.96 to 8.95 GB, wall 8:35 to 6:33
  on the HYE competed table, identical identifications; mokapot still receives a PIN).
  The worker trains on the targets at 1% plus a capped, hybrid-selected decoy sample
  with warm refits (`train_neg_ratio 3`, `train_neg_select hybrid`,
  `train_warm_epochs 5`): HYE A01 +1.0%, HYE B01 +2.2%, AIF -0.1%, entrapment +3.3% at
  an unchanged spike-in FDP, at 9 to 19x less training time. `rescore.features` /
  `features_file` project the classifier's input columns; `feature_preset = compact`
  (114 features) is an opt-in memory lever, not a sensitivity one;
  `max_feature_matrix_gib` turns an oversized matrix into an error at startup;
  `MUMDIA_NN_SEED` sets the worker's base seed.
- **Memory footprint.** Streaming Parquet readers, incremental extract output flushed
  as isolation windows close (`extract.windows_in_flight`, auto, capped at 16), f32
  bulk arrays and a chunked features stage. HYE B01 single run: 231 GiB and 1:07:30
  before, 16.5 GiB and 17:52 after (compact preset; about 20:40 with every feature).
  Six pooled HYE runs rescored in 18 minutes at 15.9 GB against 4:34:42 at 40 GB.
- **N-terminal methionine excision** in the native digest
  (`digest.n_term_met_excision`, default on, matching DIA-NN `--met-excision`); old
  configurations still parse. `scripts/augment_library.py` uses the same digest to add
  the tryptic peptides an imported library is missing.
- **Imported libraries with empty protein cells load.** An empty `protein` (DIA-NN
  writes the iRT-kit standards that way) is grouped as `UNASSIGNED` with a warning
  that counts the rows; `scripts/import_diann_lib.py` writes the same group. An empty
  `peptidoform` is still an error.
- **Desktop application** (`desktop/`, "MuMDIA Console"): a Windows `.msi` and a Linux
  `.AppImage` built by the release workflow, bundling the engine and `uv`. It creates
  its own Python environment (no conda), installs ThermoRawFileParser on request,
  locates msconvert and DIA-NN, and rescores all files provided together by default.
  Its backend is unit-tested and both bundles were built and inspected; nobody has yet
  clicked through the interface end to end.

- The release archive verifies itself. It now carries `ci/smoke.sh`, its two helper
  scripts and `test_data/fixture.fasta`, and `release.yml` unpacks every archive it
  builds into a clean directory and runs that archive's own smoke test, on every
  target. `docs/19` told the reader to run exactly this while the archive shipped
  neither `ci/` nor `test_data/`; testing the artifact rather than the tree it came
  from is also the only check that can catch a packaging mistake, and it gives macOS
  its first end-to-end coverage.
- `validate-tag`, a release job every build depends on: the tag must equal the
  workspace version, the tagged commit must be an ancestor of `main`, and `ci.yml`
  must have a successful run for that exact SHA. A tag push does not trigger
  `ci.yml`, so the only checks behind a release were previously `--version`,
  `--help` and `doctor`.
- `run-experiment` coverage: `ci/smoke.sh` now runs the multi-run orchestrator over
  two copies of the fixture and asserts the pooled rescore, the by-source split, the
  per-run quantification and the cross-run LFQ. The multi-run path had no test of
  any kind, and its split had a silent data-loss case (see Fixed).
- The experiment manifest records one artifact per output it writes, each with a
  content hash, row count and schema version. It previously listed output paths and
  nothing else, so two experiment results could not be compared. New artifact
  identity `lfq_maxlfq` for the cross-run table.
- `sbom.cdx.json`: a CycloneDX 1.5 software bill of materials generated from
  `cargo metadata --locked` by `ci/gen_sbom.py`, covering all 173 components with
  purls and the full dependency graph. Shipped in the release archive and at
  `/opt/mumdia/sbom.cdx.json` in the image, and checked for staleness in CI.
  `THIRD_PARTY_LICENSES.md` is a notice document for a human reader; this is the
  machine inventory a vulnerability scanner or a software inventory consumes.
- `pip-audit` over both resolved sidecar environments, strict on the weekly
  scheduled run and advisory on pull requests, plus `pip freeze --all` uploaded per
  environment as a 90-day artifact. This is what covers the Python dependency
  surface, which has no Dependabot support: the pins live in the `pip:` sections of
  the `env/` conda specifications, which Dependabot cannot parse, and a mirror
  requirements file would be a second list that nothing installs. Reasoning in
  `docs/14`.
- Release platforms are now Linux (musl), Windows and Apple silicon. The Intel Mac
  target was removed: it required GitHub's `macos-13` label, which no longer receives
  a runner (measured 2026-08-28: queued over two hours with none assigned, in two
  separate rehearsals, while every other target finished in about three minutes), so
  a real tag would have hung until GitHub's 24-hour queue timeout and then failed (a
  rehearsal job was observed reporting exactly `24h0m0s`; that is the limit on waiting
  for a runner, not the six-hour limit on a running job). Cross-compiling it on
  the Apple silicon runner was rejected because the result cannot be executed there,
  and publishing the one archive nobody ran is what the verification step above exists
  to prevent. Intel Mac users build from source or use the container image.
- Docker base images pinned by digest, with a Dependabot `docker` entry to keep the
  pins current. A tag is mutable, so a rebuild of the same commit could previously
  produce a different image.
- `--min-assertions` on `ci/check_smoke.py`: the smoke run fails if fewer assertions
  execute than the count quoted in the documentation. The documented count was 112
  while 117 ran, and a guard block that stops executing fails no assertion, so it
  reads as a pass.
- `quant.fragment_selection = predicted` ranks a precursor's fragments for the
  top-N sum by their library intensity instead of by their own integrated area.
  Ranking by observed area preferentially selects interfered fragments, because
  interference inflates the very quantity the ranking rewards, and the selected
  set then varies between runs.
- `quant.fixed_scan_halfwidth` and `quant.fixed_window_s` integrate a fixed
  window centred on the identification apex instead of the descent-walk bounds.
  The seconds form is instrument-independent and overrides the scan form. On the
  ProteoBench Astral HYE set these two options together moved median absolute
  epsilon from 0.273 to 0.195 and CV from 0.175 to 0.107.
- `quant.baseline_subtract`, with `baseline_flank_scans` and
  `baseline_quantile`, subtracts a flank-quantile background inside the fixed
  window.
- `prescan` stage: a native per-run sequence-tag prescan that prunes
  modification-bearing candidate hypotheses with no anchored tag support, 11.6
  times faster than the previous Python screen. Only modform hypotheses are ever
  pruned.
- `rt_im_train.window_holdout_frac` sizes the RT window from held-out anchors
  rather than from the anchors the calibration was fitted on. Benchmark-gated and
  off by default: it gained 1.1% of peptides with DeepLC 4.1.0 but lost 1.5% with
  the overfitting 4.0.0a2 model, so it interacts with RT-model quality.
- `env/mumdia-deeplc.yml`: a portable conda spec for the DeepLC sidecars. They
  previously had no committed local environment, so running them meant
  reconstructing a developer machine by hand.
- Sidecar interpreter discovery. A `python` field may be `"auto"` or absent, and
  the engine finds an interpreter from `MUMDIA_PYTHON_<ROLE>`, `MUMDIA_PYTHON`,
  `CONDA_PREFIX`, `VIRTUAL_ENV`, or `PATH`, accepting a candidate only after it
  imports what that role's workers import. A role is resolved only if the
  configuration uses it, so a default native run still needs no Python at all.
  Explicit paths behave exactly as before.
- Global CLI flags, accepted before or after the subcommand: `--threads N`
  bounds the engine's rayon pool and is forwarded to the sidecars as
  `MUMDIA_NN_THREADS` and `OMP_NUM_THREADS` when those are unset;
  `--log-level`, `-v`/`-vv` and `-q` set verbosity. Previously the only control
  was `RUST_LOG`, which is not discoverable from `--help`, and thread count could
  not be bounded at all: the engine never read `RAYON_NUM_THREADS`, so a run took
  every core on a shared machine.
- `configs/examples/{native,fasta-sidecars,diann-library}.json`, portable
  starting points that use `"auto"`, with `configs/README.md` explaining the
  resolution order and the environment specs. These replace the only tracked
  config, which named one developer's interpreters and OneDrive path and was the
  config the documentation told everyone to run.
- End-to-end smoke test, run in CI on Linux and Windows: `ci/smoke.sh` builds a
  synthetic library from `test_data/fixture.fasta`, generates a matching mzML from
  the engine's own library so the planted peaks cannot disagree with the mass
  model, runs the single-run pipeline twice, runs `run-experiment` over two copies
  of the fixture, and asserts 136 things. It covers mzML parsing, the library
  build, the `run` orchestrator and its manifest, retention-time calibration, the
  report writers, and the multi-run path (pooled rescore, by-source split, per-run
  quant, cross-run LFQ), none of which had any test. `--min-assertions` fails the
  run if fewer assertions execute than the count quoted here, so a guard block
  that stops running cannot pass silently.
- A further CI job asserts the two platforms produced byte-identical
  `peptides.tsv` and `proteins.tsv`. The native pipeline turns out to be
  byte-reproducible across operating systems, not only across runs.
- `tests/python`: 71 tests over the Python worker contracts, run in CI. Tests
  needing torch, mokapot, deeplc or ms2pip skip rather than fail.
- `docs/23_cli_reference.md` and `docs/24_config_reference.md`, generated from
  `--help` and from `config.rs` by `ci/gen_*_reference.py` and checked for
  freshness in CI, so a new flag or field lands with its documentation. The second
  includes the environment-variable table that existed nowhere: 47 variables read
  across engine and sidecars, plus the 11 the code sets.
- `bench/`: the portable part of the ProteoBench scoring path, the two recorded
  results with their row units and q columns, and the measured resource profile of
  a reference run (85 minutes and 13.1 GB of artifacts from a 1.94 GB mzML,
  rescoring 80% of it).
- `ci/check_doc_refs.py`, run in CI: fails when a tracked file cites a Markdown
  document the repository does not ship.
- `CONTRIBUTING.md`, `SECURITY.md`, this changelog, and
  `docs/22_release_plan.md`.

All new quantification options default to off, so an existing configuration
produces bit-identical results.

### Changed

- **Competition key: `compete.group_by = peptidoform_charge`** (keys
  `(pform_id, label, charge, peak_rank)`). Sibling charge states and modforms of one
  peptide are separate precursors that compete only against their own alternative
  peaks, the unit DIA-NN reports at and the key every benchmark in
  `docs/28_feature_selection_analysis.md` ran under (entrapment FDP flat at
  0.48-0.64%). The previous default `base_peptide` (renamed from `precursor`, which it
  was not) deleted every charge and modification variant of a peptide but the highest
  `prelim_score` before rescore: 23% of the extracted candidates on HYE B01, 46.6% on a
  modification-rich library, at an unchanged peptide count. It stays available as an
  explicit peptide-level population and must not be used for a PTM search.

- CPU PyTorch in the three DeepLC-bearing environment sets (`env/docker-deeplc.yml`,
  `env/mumdia-deeplc.yml`, `env/console-requirements.txt`) moves from `2.12.1+cpu` to
  `2.14.0+cpu`, the first version whose metadata allows a `setuptools` without
  PYSEC-2026-3447 (`>=77.0.3` instead of `<82`). Resolved on 2026-09-06 with
  `deeplc==4.1.1`: numpy 2.4.6, pandas 2.3.3, psm-utils 1.5.5, scikit-learn 1.9.0,
  setuptools 84.0.0, i.e. the same scientific stack as before with only torch
  changed. DeepLC 4.1.1 declares `torch<3,>=2.6.0`. Neural-network training is not
  bit-deterministic across torch versions, so expect seed-level, not result-level,
  differences in `nn_torch` rescoring.
- CI now enforces the full stated gate: `cargo fmt --check` and
  `cargo clippy --workspace --all-targets -- -D warnings` in addition to the
  build and tests, plus `python -m compileall` over the sidecars, a JSON parse of
  every tracked configuration, a YAML parse of the environment specs, and the
  documentation-reference check. Formatting and clippy were previously a local
  responsibility, so `main` could carry a tree that failed them.
- The Docker DeepLC environment pins `deeplc==4.1.1` from PyPI instead of a git
  commit on the 4.0 multitask branch, and no longer caps `numpy<2`, which 4.1.1
  does not require. 4.1.1 is a floor, not merely the current release: the 4.0.0a2
  multitask preview overfits per-run fine-tuning badly enough to invert RT-model
  rankings. Verified by building the image and importing the workers' graph in
  the worker's own order: DeepLC 4.1.1, torch 2.12.1+cpu, numpy 2.4.6 in the
  `deeplc` environment and mokapot 0.10.0 in `rescore`, with `mumdia doctor`
  passing on both baked configurations.
- The image no longer runs as root after setup. It needs
  `--user "$(id -u):$(id -g)"` to write into a bind mount, which the documented
  invocation now passes and the Docker workflow now asserts.
- Under a fixed integration window, the reported `integration_lo_rt` and
  `integration_hi_rt` are now the retention-time extent actually integrated
  rather than the walked bounds that were ignored. Measured on the AIF benchmark
  run behind the ProteoBench submission: 72,168 quantified precursors, every
  `quantity`, `n_fragments_used`, `quant_status` and `integration_apex_rt`
  bit-identical, and the reported window corrected from a 29.1 s median (the
  descent walk) to 34.9 s (the fixed window that produced the numbers).
- `mumdia doctor` reports whether the configuration can actually run: the
  interpreter each role resolves to and how it was found, the versions of the
  packages whose version changes results, whether the worker scripts are where
  the engine will look, and a warning when DeepLC is older than 4.1.1. It now
  covers `mbr.python` and the script directory, neither of which it checked
  before, and it no longer fails a native configuration over a worker directory
  that configuration never opens.
- `predict_frag.sidecar_script_dir` is resolved against the config file's own
  directory and against the executable's directory, not only the current working
  directory. The same config invoked from elsewhere used to silently change which
  worker scripts ran.
- `run-experiment` warns when it overrides the configured `quant.q_filter` to
  gate per-run quantification on the pooled q value, instead of doing it
  silently.
- Dependabot keeps Cargo and GitHub Actions dependencies current, with
  `arrow`/`parquet` grouped separately because they carry the on-disk contract.

### Fixed

- The sidecar environment specifications pin `setuptools>=83`. The CI audit of the
  resolved DeepLC environment found `setuptools 81.0.0` (PYSEC-2026-3447,
  CVE-2026-59890, fixed in 83.0.0) and failed the main branch after the merge of #54.
  The conda-level pin alone did not hold: `torch 2.12.1` declares `setuptools<82`, so
  pip downgraded the conda-installed 84.0.0 to 81.0.0 underneath it. torch is now
  2.14.0 (see Changed) and the floor is repeated in the pip sections and in the
  desktop requirement set. The audit step is advisory on pushes to main as well as
  on pull requests, as its comment already intended; the weekly scheduled run and a
  manual dispatch stay strict.
- A single malformed retention time in an mzML aborted the whole run. `convert`
  validated peak m/z and intensity but not the scan start time, so one `NaN` value
  passed unchecked into the spectra artifact and then panicked inside extract with
  `called `Option::unwrap()` on a `None` value`, naming neither the file, nor the
  scan, nor the value. Reproduced by editing one value in the fixture mzML. Such
  spectra are now dropped with a count and the first offending scan id, which loses
  nothing (a spectrum with no retention time cannot be placed in a chromatogram) and
  leaves identifications unchanged; `ci/smoke.sh` asserts all three.
- Every float ordering in the workspace now uses `total_cmp` rather than
  `partial_cmp(..).unwrap()` (25 sites) or `partial_cmp(..).unwrap_or(Equal)` (36
  sites). The first panics on NaN; the second is worse, because `Equal`-on-NaN is an
  intransitive comparator and `sort_by` has detected that and panicked since Rust
  1.81, so it converted a deterministic failure into an intermittent one. `total_cmp`
  agrees with both on every finite value: the fixture's `peptides.tsv` and
  `proteins.tsv` hashes are byte-identical across the change. One of the rewritten
  comparators picks the competition winner, where treating every NaN as equal made
  the surviving row depend on iteration order.
- `compete` panicked instead of erroring when a `.schema.json` companion named a
  feature column the parquet does not have, which a stale companion beside a
  rewritten table produces. It now names the column and the file and says to delete
  the companion.
- `scripts/make_reverse_decoys.py` silently assigned 0 Da to any modification outside
  its eight-name table, so those decoys got fragment m/z for the wrong molecule and
  could never match. A decoy that cannot match does not compete, which makes the
  target-decoy null optimistic for exactly the peptides carrying that modification,
  and nothing in the output distinguished such a decoy from a good one. The sampled
  calculator check could not catch it: it compares 500 precursors at the 99th
  percentile. Unknown modifications now raise, `valid()` rejects the peptidoform so
  no decoy is written for it, and the script reports the names and counts. This
  matches the engine's own parser, which has always returned
  `MassError::UnknownModification`.
- `run-experiment` dropped PSMs silently when splitting the pooled scored table by
  run. `split_by_source` filtered on `source == i` for each output table and returned
  `Ok` regardless, so any row whose `source` had no output table went nowhere: every
  per-run quantity and the cross-run LFQ were then computed from a smaller population
  with no error and no warning. It now counts the rows it placed and refuses if that
  is not all of them.
- The library helpers write parquet the engine can read on any pandas.
  `DataFrame.to_parquet` chooses the arrow string width itself, and pandas 3
  writes `large_string`, which the engine rejects at load with
  `column 'peptidoform' is not utf8`. Every library built by
  `import_diann_lib.py`, `make_shift_decoys.py`, `make_reverse_decoys.py` or
  `augment_library.py` on a current pandas was therefore unreadable, breaking the
  imported-library path. They now write through `scripts/_lib_io.py`. Found by the
  new sidecar contract tests.
- MBR transfers are now quantified. The augmented scored table lowered only
  `q_value`, while quantification gates on `quant.q_filter`, which the experiment
  path sets to `run_psm_q`. An accepted transfer therefore kept a sub-threshold
  `run_psm_q` and was dropped: 34,280 of 34,664 transfers on a six-run HYE
  experiment, so match-between-runs appeared to run and changed almost nothing.
- `quant` reads `predicted_intensity` as an optional chromatogram column.
  Requiring it made every chromatogram artifact written before that column
  existed unquantifiable; the `predicted` ranking, its only consumer, now fails
  with an actionable message instead.
- `scripts/deeplc_worker.py` imports `deeplc` before `numpy` and `pyarrow`. The
  wrong order aborts torch DLL initialization on Windows with
  `OSError: [WinError 1114] ... c10.dll`. The failure was latent because
  imported-library mode skips the stage that reaches it.
- `.gitignore` covers `rust/mumdia/target` (the root-anchored `/target` never
  matched it), the experiment configurations that carry machine-specific
  interpreter paths, and the local benchmark data directories. It no longer
  matches `docs/22_release_plan.md`, which an unanchored `*_plan.md` rule had
  silently excluded from version control.
- Source comments no longer cite untracked local design notes. About 130
  references pointed at documents a clone does not receive; they now point at the
  tracked `docs/` guide.

### Known limitations

- The sidecar contract tests cover the workers' file contracts, not the science:
  the tests that need torch, mokapot, DeepLC or MS2PIP skip on a runner without
  them, so CI does not validate rescoring or retention-time prediction behaviour.
- The end-to-end smoke test runs on Linux and Windows, not macOS, and uses the
  native predictors only. A separate job imports DeepLC, mokapot and MS2PIP in
  real conda environments on any pull request touching `scripts/`, `env/`,
  `tests/python/` or the Dockerfile, but no CI job runs the sidecar path end to
  end on data.
- Under an experiment-wide rescore no per-run TSV report is written; per-run counts
  come from the split scored tables on `run_psm_q`, and the experiment-wide
  `peptides.tsv` / `proteins.tsv` are the reports.
- `mbr.strategy` distinguishes only none from not-none; `rt_window_s`,
  `decoy_transfer` and `requant_all` are accepted but not wired.
- `extract.retain_top_peaks > 1` writes diagnostic peak alternatives that do not
  reach features or rescoring.
- No ion mobility support: a Bruker diaPASEF `.d` is converted to 3D spectra and
  searched with more interference than a 4D engine would see. Vendor formats other
  than Thermo `.raw` are converted through msconvert but have not been exercised on
  real files. No wildcard or terminal variable modifications.
- The desktop application has not been clicked through end to end; its backend is
  unit-tested and both bundles were built and inspected.
