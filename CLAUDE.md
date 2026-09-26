# CLAUDE.md

Repository guide for coding agents and maintainers. Read this first, then use
`docs/README.md` to route to the code-grounded subsystem reference. `plan.md` is
the long algorithmic design history, untracked and local-only; when it disagrees
with executable code, tests and the tracked `docs/` guide describe current
behavior.

## Project and scientific objective

MuMDIA is a clean-room Rust DIA proteomics search engine. It converts mzML,
builds or imports a spectral library, performs a broad calibration search,
calibrates retention time, extracts chromatographic evidence, computes a
versioned feature vector, competes candidates, rescores against decoys, and
reports identifications and label-free quantities.

Optimization has three separate objectives:

1. identification sensitivity: more true discoveries at a stated q threshold;
2. FDR validity: the q threshold must remain calibrated under entrapment and
   exchangeable paired decoys;
3. quantification accuracy: low bias/CV/missingness on known-ratio data.

Do not use a higher identification count as evidence that the other two
improved. `docs/20_sensitivity_and_quantification_playbook.md` is the operational
policy for tuning and validation.

## Repository map

- `rust/mumdia/`: Cargo workspace.
  - `mumdia-core`: typed config, schemas, manifest, masses/constants.
  - `mumdia-io`: Arrow/Parquet table layer, hashes, JSON, artifact reports.
  - `mumdia`: CLI/library, fragment index, FDR/rescoring, and stages.
- `scripts/`: eight engine-invoked Python workers plus four imported-library
  helpers (twelve scripts), and `_lib_io.py`, the shared writer the helpers use so
  they cannot emit a parquet the engine rejects. Includes `augment_library.py`, which adds the
  tryptic FASTA peptides an imported library is missing. Sidecars use positional
  file contracts.
- `docs/`: tracked developer guide (`01` through `21`); start at
  `docs/README.md`.
- `env/`, `docker/`, `Dockerfile`: sidecar environments and deployable configs.
- `lib/`, `fasta/`, `mzml_files/`: large local inputs, intentionally untracked.
- Root comparison/design notes other than this file and `README.md` remain local
  under the root-Markdown ignore rule.

Preserve the untracked `lib/` data. Do not treat a dirty worktree as disposable.

## Build and validation

The workspace target directory is redirected off OneDrive on the development
machine. Do not move it back into the synced tree.

```text
cd rust/mumdia
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release --locked
```

Python changes must at least pass `python -m compileall -q scripts`; JSON
configs must parse. Real DeepLC, mokapot, NN, and MBR behavior is not exercised
by the Rust unit suite, so do not present a passing CI run as sidecar validation.

## Actual workflow

Library construction/import and mzML conversion are independent branches:

```text
FASTA -> digest -> peptidoforms -> predict-frag --+
                                                   +-> search-seed
imported library ---------------------------------+       |
mzML or vendor file -> convert -------------------+       v
        (optional: prescan, per-run tag pruning    |
         of modform hypotheses for a PTM search)   |
                                    library iRT re-prediction (DeepLC 4.1.1 base
                                    model, default) or optional DeepLC fine-tune
                                                           |
                                                           v
                                      rt-im-train -> extract -> features
                                      -> compete -> rescore -> quant -> report
```

Key semantics:

- `run` is a single-run orchestrator when given one `--mzml`; given several it
  dispatches to `run-experiment`, because files provided together are rescored
  together by default (one pooled FDR, per-run quant, cross-run LFQ). Searching
  files separately is the opt-in: one `run` per file. `run` always recomputes and
  overwrites its named outputs; the manifest is provenance, not a cache or resume
  database. Use a fresh output directory. The one opt-in exception is
  `predict_frag.library_cache`: a FASTA-mode run then reuses a library an earlier run
  stored under a key of the FASTA hash, the build settings, the predictor versions and
  the engine binary, and skips digest, peptidoforms and predict-frag
  (`library_cache.rs`). Without it, a FASTA run logs the `--lib-*` command, with the
  `rt_im_train.library_irt` value that keeps its retention-time handling, that would
  reuse the library it just built.
- Standalone stages can be reused manually because inputs are path-addressable.
- Both `convert` and `run` default `--top-peaks-ms2` to `0` (uncapped). The cap
  is destructive: `convert.rs:76-79` keeps only the top N peaks per MS2 spectrum
  and bakes the truncation into the spectra artifact, and extract applies no cap
  of its own. `search_seed.top_n_peaks` (`config.rs:410-415`, default 300) is a
  separate, non-destructive limit that only bounds seed index-probing cost. The
  seed selects from what convert wrote, so a conversion cap below `top_n_peaks`
  also shrinks the seed's input; above it the two do not interact, and seed
  output was identical with and without a 300-peak conversion cap.
- `--max-spectra N` reads the head of the mzML. It does not select a
  mid-gradient slice.
- A vendor path given as `--mzml` is converted to mzML first by
  `raw::ensure_mzml` (`convert`, `run`, `run-experiment`, `peak-census`): Thermo
  `.raw` by ThermoRawFileParser, Bruker/Agilent `.d`, SCIEX `.wiff` and Waters
  `.raw` by ProteoWizard `msconvert`, both located (`convert.thermo_raw_parser`,
  `convert.msconvert`, `auto` by default) and never shipped. The mzML lands
  beside the input and is reused when newer than it (`convert.reuse_converted`).
  `mumdia doctor` reports both converters and never fails for their absence.
  Only Thermo is exercised end to end; `docs/04_convert.md` "Vendor formats" has
  the table, the ion-mobility caveat for Bruker, and the extension collisions.
- An imported library row with an empty `protein` is grouped as `UNASSIGNED` at
  load, with a warning that counts the rows (DIA-NN writes the iRT-kit standards
  without a protein); `scripts/import_diann_lib.py` writes the same group at
  import time. An empty `peptidoform` is still a hard error.
- The native digest emits N-terminal Met-excised forms by default
  (`digest.n_term_met_excision = true`, matching DIA-NN `--met-excision`).
  Excision keys on protein position 0 with a leading `M`, not any interior `M`.
  Without it the search database structurally misses those peptides; old configs
  still parse because the field defaults on. `augment_library.py` reuses this
  same digest to fill an imported library's missing tryptic peptides.
- Imported-library mode skips digest, peptidoform expansion, and initial
  prediction. Under the default `rt_im_train.library_irt = auto` the imported
  iRT is re-predicted with the DeepLC base model when a DeepLC interpreter is
  configured (a new precursor table; once per experiment under `run-experiment`),
  because the imported DIA-NN iRT is the worst RT source measured: AIF 10,015
  peptides raw, 10,181 per-run fine-tuned, 10,416 re-predicted with DeepLC 4.1.1;
  HYE B01 (NN seeds 1-3) 56,556 raw, 60,278 with a once-fine-tuned library, 58,842
  re-predicted (`docs/08_rt_im_train.md` section 4c). The re-prediction is
  deterministic and costs about 27 minutes once for the 10.9M-row HYE library on
  64 threads. The once-per-library fine-tune is still +2.4% on HYE (18.6k anchors)
  and -2.3% on AIF (5.6k anchors), so it stays the recommended extra step on a
  large reference rather than the default. Optional DeepLC fine-tuning still runs
  after seed search and writes a new precursor table rather than modifying the
  input.
- Stage-level candidate competition is within label, so it does not directly
  eliminate a target against its decoy. Peptide-level q estimation subsequently
  performs picked target-decoy competition through the shared
  `base_peptide_id`; keep that pairing intact.
- `extract.retain_top_peaks > 1` (default 1) writes the alternative peaks as
  additional `psms_extracted` rows with `peak_rank >= 1` (plus a diagnostic
  `.peaks.parquet`), `features` carries `peak_rank`, `compete` keys on it, and
  `rescore` keeps one row per candidate and records `selected_peak_rank`. The
  plumbing exists; what the default still lacks is entrapment validation on two
  acquisitions.

## Validated sensitivity workflow

The strongest measured workflow on the chimeric AIF benchmark run
`LFQ_Orbitrap_AIF_Ecoli_01.mzML` uses the imported DIA-NN library, DeepLC
fine-tuning, Extended features, the loose `apex_pearson` extraction gate, and
`nn_torch` rescoring:

```text
mumdia doctor --config configs/examples/diann-library.json

mumdia run \
  --lib-precursors lib/lib_precursors.parquet \
  --lib-fragments  lib/lib_fragments.parquet \
  --mzml mzml_files/LFQ_Orbitrap_AIF_Ecoli_01.mzML \
  --out-dir out_aif_nn \
  --config configs/examples/diann-library.json \
  --top-peaks-ms2 300
```

Use the original imported precursor library here. Do not pass
`lib_precursors_ft.parquet` while also enabling fine-tuning. The
`--top-peaks-ms2 300` in that command reproduces this one benchmark. Do not
carry it to another acquisition; see the next subsection.

`configs/examples/diann-library.json` sets both interpreters to `"auto"`, so it
runs unchanged only where an environment with torch and DeepLC is discoverable
(an activated conda env, or `MUMDIA_PYTHON_RESCORE` / `MUMDIA_PYTHON_DEEPLC`).
Run `mumdia doctor --config ...` first: it prints the interpreter it resolved and
the versions, or names exactly what is missing. On a machine with several
candidate environments, copy the example and write the two paths in, which is
what the untracked `config.local-*.json` files are for.

The historical result is about 10.3k confident precursor-shaped report rows,
selected by stripped-peptide q at 1%, versus roughly 9.3-9.5k with linear/native
or mokapot rescoring. It is a benchmark target, not a CI assertion or a universal
instrument preset. Measured 2026-08-24 against DIA-NN 2.2.0 library-free with
`--reanalyse` on the same file (11,817 stripped peptides at 1%), this workflow
reached 90.4-91.6% depending on the DeepLC version and window sizing.

Prefer the `augment_library.py`-completed precursor/fragment tables over the raw
imported library when both exist. The raw imported DIA-NN library is missing 209
of DIA-NN's own 1% peptides on this benchmark, all N-terminal Met-excision forms;
the augmented tables (+18,903 tryptic base peptides) recover about 80 of them at
an unchanged 0.98% empirical decoy fraction and identification parity elsewhere.
The remaining ~130 enter the search space but stay below threshold, which is an
abundance limit, not a library hole. One caveat measured on the same day: DeepLC
per-run fine-tuning is not deterministic, and under held-out window sizing a poor
fine-tune draw widens `w_rt` (held-out p95 varied 150-211 s across two draws of
one arm) and can cost about 2% of peptides; judge any single-run comparison of
window sizing or library variants against that draw variance before concluding.

### The peak cap is acquisition-specific, and 300 is not a default

`--top-peaks-ms2 300` belongs to that chimeric AIF run, where only 47.8% of MS2
spectra saturate the cap. Elsewhere it is actively harmful. On a 50-window
Orbitrap DIA run the same cap discarded 78.6% of all MS2 peaks and cost 60% of
the peptides (25,425 capped versus 63,237 uncapped) at an unchanged 0.99%
empirical decoy fraction, so the loss is sensitivity, not a loosened threshold.
The mechanism is peak-group formation rather than scoring: with most peaks gone,
`presence_min_fragments` cannot be met and real peptides never assemble a peak
group.

That reading comes from the cap dose-response, not from the audit ladder's own
label. `DID_NOT_SURVIVE_EXTRACTION` cannot be used as evidence for it: `audit.rs` reads a
per-candidate audit table that `extract` does not write (`emit_candidate_audit`
is unwired), so the reason map is always empty and the `_ => DidNotSurviveExtraction`
catch-all absorbs presence failures, matched-fraction failures AND every
extraction-gate rejection alike. Treat the label as "did not survive extract",
and do not decompose it further until the audit table is actually produced.

Rules that follow from this:

- prefer uncapped, which is the shipped default at both conversion entry points;
- never carry a cap across acquisition schemes;
- before setting any cap, compute the peaks-per-MS2-spectrum percentiles for the
  run. If p25 already exceeds the intended cap, the cap is deleting fragment
  evidence from most spectra and must be raised or removed;
- when peak volume must be bounded, take the cap from an end-to-end sweep on that
  acquisition, not from another run.

`docs/04_convert.md` ("Choosing `--top-peaks-ms2`") is the canonical treatment:
peak census, end-to-end effect, audit ladder, cap dose-response, and the
extraction cost of uncapping. `docs/20_sensitivity_and_quantification_playbook.md`
has the pre-flight saturation check as runnable code and the promotion policy;
`docs/09_extract.md` has the extraction-side rejection path; `docs/18` finding A3
is the decision record.

### Extraction gate and retention time

`extract.gate_min_score` is named historically. Under the default
`gate_mode = apex_pearson`, it thresholds observed-versus-predicted fragment
intensities at one apex; it is not a chromatographic co-elution correlation.
**The default is 0.2, and loose is now better for both rescorers.** Measured
2026-08-28 on the AIF file under the current defaults, at an unchanged empirical
decoy fraction of 0.0097-0.0098:

| gate | `native_tda` | `nn_torch` |
|---|---|---|
| 0.2 | 10,847 | 10,914 |
| 0.6 | 10,369 | 10,399 |

0.6 costs 4.4% of peptides for `native_tda` and 4.7% for `nn_torch`, and halves
what extract accepts (45,338 against 21,979).

This corrects an earlier reading. The `docs/18` gate sweep, taken on the RAW
imported library before the augmented tables, before `apex_evidence_rank` became
the default and before the CV fold was paired, showed `native_tda` peaking at 0.6
(9,503) and `nn_torch` at the loose end. On that basis the default was briefly
changed to 0.6 to match the default classifier. `native_tda` has since risen to
10,847 and its optimum has moved to the loose end, so the sweep no longer
describes this configuration and the default is back to 0.2. Re-derive the optimum
rather than inheriting it if you change the library, the apex mode or the folds.

DeepLC 4.4.0 or newer is required wherever DeepLC runs (`predict-frag` in FASTA mode, the
optional fine-tune): `mumdia doctor` fails on an older one, `sidecar::require_deeplc_version`
refuses to launch either worker, and both worker scripts repeat the check
(`mumdia_core::constants::MIN_DEEPLC_VERSION` is the single Rust constant). The default retention-time workflow is
prediction plus per-run LOESS calibration with `finetune_deeplc = false`, and that default
is only sound on a base model that does not memorise its anchors (4.0.0a2 did).

`rt_im_train.multihead_calibration` (default: automatic) is the fourth RT lever, the
reason the floor is 4.4.0, and since 2026-09-11 the default. `deeplc.predict` returns ONE of the base model's 6,543
LC-setup heads, the one its `DEFAULT_TASK_NAME` names, on that setup's own gradient. The
per-run LOESS then maps that column onto observed RT, and a smooth increasing curve can
stretch and bend the axis but cannot reorder two peptides, so that setup's elution order
survives into the result. Different chromatography reorders peptides, which is what the
multitask model exists to represent. Set to N and the worker fits
`MultiHeadRidgeCalibration` over the N best-correlating heads against this run's confident
seed PSMs instead of fine-tuning, so the ordering is assembled from the setups that
resemble the run.

Left unset it is AUTOMATIC, and the scope matters: 80 heads when a DeepLC interpreter is
available and the run's retention times are DeepLC's already (imported library under
`library_irt = auto`/`deeplc`, or FASTA with `rt_predictor = deeplc`), and nothing
otherwise. It never turns a native, Python-free run into a startup error, and it never
replaces a native retention time because an unrelated interpreter happened to be
discoverable. `0` disables it; any other number is an explicit request and then a hard
requirement. `finetune_deeplc` keeps its own slot rather than colliding with the default;
asking for both explicitly is still refused.

Measured (`docs/08_rt_im_train.md` section 4d), two acquisitions, six pooled runs each, at
an unchanged empirical decoy fraction of 0.0100 in all four arms: AIF 80,842 -> 84,725
peptides (+4.8%), Astral 102,942 -> 117,652 (+14.3%), protein groups +2.4% and +7.1%. The
entrapment arm closes the other half of the gate: +4.28% real peptides at an empirical FDP
of 0.995% against the baseline's 0.995%, on 138 and 144 accepted spike-ins.
Fewer candidates reach rescore and more of them are real, which is interference removed
rather than a threshold traded. It costs 1.4x to 1.7x wall clock, because the calibration
is fitted against each run's own anchors. Since 2026-09-14 `experiment.rt_library_scope`
(default `first_run_only`, formerly `finetune_scope`) amortises it exactly as it already
amortised the fine-tune: the first run adapts the library, the rest reuse it and fit their
own LOESS on top. Use `per_run` where the runs do not share an elution order.

Three RT rules, each measured in `docs/08_rt_im_train.md` and restated from the
failure side in `docs/17_troubleshooting.md`:

- DeepLC fine-tuning of library iRT is the largest RT lever (historically
  reducing residuals from about 110 s to 13-27 s) after the base-model
  re-prediction that `library_irt = auto` now does by default, and it need not
  happen per file. A library fine-tuned once and predicted over every
  peptidoform, combined with the per-run LOESS calibration and
  `rt_im_train.finetune_deeplc = false`, measured equal or marginally better
  residuals than per-file fine-tuning while removing about 36 minutes per file.
  This does not license reusing a stale per-run `_ft` table built on a different
  file, which has previously underperformed a fresh fit.
- The `cal.json` RT residuals are in-sample and optimistic. The loess is fit and
  `w_rt` derived from a residual percentile on the same anchor points (see
  `docs/08_rt_im_train.md` section 4), so the reported
  `rt_residual_abs_median_s` was roughly 3x better than the same calibration
  scored out-of-sample. Worse than optimism: in-sample residuals can rank two RT
  models backwards, because a higher-capacity model memorizes anchors (measured
  2026-08-24: in-sample 15.9 s vs 24.9 s, held-out 195.1 s vs 46.4 s, for the
  same model pair). Treat them as fit diagnostics, never as error estimates or
  model rankings; size any external RT tolerance from out-of-sample numbers.
  `rt_im_train.window_holdout_frac` (benchmark-gated, default off) sizes `w_rt`
  itself from held-out anchors; `cal.json.w_rt_sizing` records which sizing ran.
- Check modform iRT variance before trusting RT windows in a PTM search. On one
  modification-expanded imported library most stripped-peptide groups shared an
  identical raw `predicted_irt` across all their modforms: the modified forms had
  inherited the unmodified retention time and the modification was never
  modelled. If that variance is zero for most groups, re-predict iRT per
  peptidoform.

NnTorch seeds NumPy and PyTorch, but training kernels are not guaranteed
bit-for-bit deterministic; `MUMDIA_NN_SEEDS > 1` is an explicit ensemble. DeepLC
fine-tuning also is not guaranteed deterministic.

## FDR and sidecar rules

- A search library must contain valid `target` and `decoy` labels. Native digest
  decoys are collision-checked; imported-library helpers must preserve paired
  populations.
- Production and benchmark configs use `rescore.strict = true`. An explicitly
  requested external classifier must not silently become `native_tda`.
- The source of truth for the classifier actually used is
  `psms_scored.parquet.report.json`, not the configured enum or an old stdout
  line. The orchestrated manifest is expected to carry that actual identity.
- Sidecar output must cover every flat input row exactly once with finite scores.
  Mokapot must provide complete out-of-fold confidence scores; in-sample
  fold-model averaging is not an acceptable fallback.
- Q-value columns have different units:
  - `q_value` / `experiment_psm_q`: pooled PSM;
  - `run_psm_q`: within-run PSM;
  - `precursor_q`: peptidoform plus charge under the default
    `compete.group_by = peptidoform_charge`. Under `base_peptide` the sibling rows
    were already deleted, so it then counts base peptides (measured 1.000
    precursors per peptide, against 1.174 with `peptidoform_charge`);
  - `peptide_q_value`: base/stripped peptide;
  - `pg_q_value`: protein-accession-set group.
- The grouped q columns (`peptide_q_value`, `precursor_q`, `pg_q_value`) are
  written only to each group's single winning row (`rescore.rs:721-728`); losers
  get 1.0. Under an experiment-wide rescore the grouping is experiment-wide, so
  a per-run count on those columns is diluted by roughly 1/n_runs and is
  meaningless. The correct per-file unit there is `run_psm_q`.
- Pooling more runs does not tighten q. `fdr.rs` computes
  `q = (decoys + 1) / max(1, targets)`, whose only pool-size term is the `+1`
  pseudocount, and that makes a larger pool marginally LOOSER, never tighter.
  Do not attribute per-run count changes to pool size.

  "Scale-invariant" overstates it, though, and the overstatement matters at the
  top of the list: the floor is exactly `1/T`, so it scales with the pool.
  Measured on the real kernel, replicating a five-row population once moved q
  from `[0.5, 0.5, 0.667, 0.667, 1.0]` to `[0.25, 0.25, 0.5, 0.5, 0.833]`. The
  per-source `run_psm_q` is exactly per-run and genuinely unaffected; the pooled
  `q_value` -- which is what `run-experiment` gates quant on -- is not.
- Reported benchmark counts must name their row and q-value unit. `peptides.tsv`
  contains `(peptidoform, charge)` rows but is selected with
  `peptide_q_value`; it is not a precursor-q report.
- Validate new sensitivity defaults with entrapment or another empirical null,
  plus at least two datasets/acquisition contexts. Count gains alone are
  insufficient.

### Experiment-wide rescore

- `run-experiment` writes one experiment-wide `peptides.tsv` and `proteins.tsv`
  at the experiment root (`report::run_experiment`): rows selected on the
  experiment-wide `peptide_q_value` / `pg_q_value`, an `n_runs` column counting
  per-run acceptances on `run_psm_q`, and one quantity column per run
  (`quantity_<run>` from each run's quant, `lfq_<run>` from the cross-run MaxLFQ
  matrix). It writes no per-run TSVs, because the grouped q columns are written
  to each group's experiment-wide winner only. Per-run counts come from the
  split scored tables on `run_psm_q`; `mumdia report --experiment-dir` rewrites
  the experiment-wide pair at another threshold.
- `run-experiment` overrides the configured `quant.q_filter` and gates per-run
  quant on the pooled `q_value`. It warns rather than doing so silently.
- `rescore --competed` accepts many tables, stamps `source` with the index of
  the input table each PSM came from (`rescore.rs:65-70,108`), and computes a
  per-source `run_psm_q` alongside the pooled `q_value`
  (`rescore.rs:403-408`). Pooling therefore never costs per-run FDR, and
  sub-batching is free in `run_psm_q` terms. It is not free in pooled-`q_value`
  terms, because the `1/T` floor moves with the pool, so a batched run and a
  single pooled run do not produce identical `q_value` columns. Batch to fit RAM,
  and compare per-run counts on `run_psm_q`.
- Pooled rescore scales linearly, measured 0.834 ms/PSM on the streaming
  backend. Two feature matrices, one width: the Python worker's is
  `n_psms x n_features x 4` bytes (f32), and the Rust `FeatureMatrix` that
  `rescore` builds (`rescoring.rs`) is flat f32 as well, so the same
  `n_psms x n_features x 4`. `native_tda` fits its folds one at a time and holds
  one standardised copy of the training slice, so its peak is
  `1 + (folds - 1) / folds` times the matrix: 1.67x at the default 3 folds, 1.80x
  at 5, and never above 2x. `rescore.max_feature_matrix_gib` is
  checked against that layout, from the parquet footers and the selected feature
  count, before the allocation (docs/29 #11), so exceeding the ceiling is an error
  at startup rather than an OS kill hours in.

### Rescore cost: handoff, feature selection, training-set reduction

Measured 2026-09-05 on the HYE competed table (2,603,894 PSMs x 387 features), docs/28
sections 10-16:

- `rescore.handoff` defaults to `parquet` since 2026-09-05. The TSV path made the worker
  parse every column into a float64 pandas frame before building its float32 matrix; parquet
  took the rescore peak from 29.96 to 8.95 GB and the wall from 8:35 to 6:33 at identical
  identifications. mokapot and entrapment sidecars still receive the tab-separated PIN
  (`mokapot.read_pin` cannot read parquet), automatically and with a warning.
- The rescore process tree is about half as tall since 2026-09-16, at identical
  identifications. Under `rescore.strict` (the production setting) with a sidecar classifier
  the engine releases its own `FeatureMatrix` as soon as the handoff parquet is written,
  because strict has no native fallback that could still read it; the handoff is written in
  131,072-row groups; and the worker loads it row group by row group, holding at most two
  decoded groups (one filling, one read ahead on a reader thread since 2026-09-25), since
  pyarrow's `iter_batches` reads ahead without bound and its buffered batches were a second
  copy of the matrix (the worker climbed to 11.2 GB while filling a 4.85 GB matrix, then
  fell to 6.3). Measured on
  the fleet (EPYC 9354, 32 threads, process-tree peaks): the six-run Astral pool
  (3,133,636 x 387) 17.9 GB -> 9.3 GB in 19.4 against 19.4 min, HYE B01
  (1,838,344 x 387) 12.0 GB -> 5.05 GB in 5.2 against 5.0 min, 63,270 peptides in both HYE
  arms. The disk-backed memmap remains the last resort it was made in #88.
- The worker caps its torch CPU threads at 16, or at the performance-core count on a
  hybrid CPU (Windows, `GetLogicalProcessorInformationEx`; `MUMDIA_NN_THREAD_CAP` overrides,
  `0` = none), and treats the engine's `--threads` as an upper bound rather than a target.
  Measured: the MLP is flat from 16 to 64 threads on an EPYC 7H12 (one Astral run: 253 s at
  16, 254 at 32, 281 at 64) and from 8 on an EPYC 9354, while on an i9-13900KS (8 P + 16 E
  cores) every OpenMP-parallel op waits for its slowest thread: one run 200 s at 30 threads
  against 152 s at 4, and the six-run Astral pool 118 min at 30 threads against
  74 min at 8 on the same machine, and 11.4 min with the thread cap, flush-to-zero and the
  memory diet together (116,258 peptides, inside the seed spread; 4.6 min on its RTX 4090).
  A desktop rescore that is many times slower than a server's on the same pool is this, not
  the pool.
- The worker flushes subnormal float32 to zero (`torch.set_flush_denormal`,
  `MUMDIA_NN_FLUSH_DENORMAL`, default on) since 2026-09-16. Intel cores handle subnormals
  through microcode assists: measured on the i9-13900KS, one 16384 x 387 `Linear` takes
  4.2 ms with normal inputs, 475 ms with subnormal inputs, 150 ms with subnormal weights and
  3.5 ms again with flush-to-zero, while an EPYC 9354 pays nothing (12.1 against 11.4 ms).
  The competed features carry none (0 subnormal cells in the 3.1M x 387 table, raw or
  standardised); they are the first-layer weights of the constant features. A constant
  column standardises to exactly 0, so its 128 weights receive no data gradient, only the
  L2 term, and Adam walks them under 1.2e-38: the census reads 1 subnormal parameter after
  the cold round and about 1,180 (10 features x 128 units) after the first warm-started
  rounds, with 0 subnormal buffers or activations. Every FMA on those 10 columns then took
  the assist, which is why the desktop's iterations slowed as the run went on and why
  py-spy found 97% of the worker's samples inside `Linear.forward`. A
  value below 1.2e-38 contributes nothing to a score; `MUMDIA_NN_DEBUG_DENORMALS=1` prints a
  per-round census of parameters, buffers and activations. Desktop A/B on a two-run pool at 8
  threads: flush off 7.64 min, flush on 5.07 min, 96,137 peptides in both. This, not the
  thread count alone, is why the same pool took 118 min on the desktop and 19 on the fleet.
  The worker also drops the constant columns before training (`MUMDIA_NN_DROP_CONSTANT`,
  from the parquet footer's per-column min/max, 11 of 387 on the Astral pool), which removes
  that subnormal source but not the others: dead hidden units and their BatchNorm variances
  decay the same way (census on the six-run pool: up to ~6,400 parameters, 11 buffers,
  ~3,000 activations in a round), and the desktop still took 61.8 min without flushing.
  Since 2026-09-17 the worker removes that source as well: once per epoch every parameter,
  floating buffer and Adam moment with |value| < 1e-20 is set to exactly 0
  (`MUMDIA_NN_CLAMP_TINY`). A value that small adds nothing to any float32 sum here, a weight
  of exactly 0 stays 0 under Adam, and 0 times anything is 0, so no subnormal can form in a
  weight, a buffer, an activation or the optimizer state on any CPU, flush-to-zero or not.
  The moments matter: for a zero-gradient parameter Adam's first moment decays as 0.9^k and
  its second as 0.999^k, and every step touches them elementwise; with parameters and
  buffers clamped but the moments not, the six-run desktop pool still took 18.8 min with
  flush off (train 939 s), with the moments clamped 11.9 min (train 527 s, the flushed run's
  513), census 0 everywhere, 116,873 peptides. Two-run pool: 4.31 min against 7.64 without
  the clamp. Flush-to-zero stays on as the second layer; clamping training logits was tried
  and changed nothing (no row saturates).
- Single-seed counts are not a measurement. Any change to the arithmetic reshuffles one
  seed's peptide count by up to ~0.4% on the Astral pool and ~1% on HYE B01: seed 0 gave
  116,405 at 32 threads, 116,192 at 8 threads, 116,025 with the 11 constant columns dropped
  (a mathematically identical model) and 115,937 with flush-to-zero, while the means over
  three seeds sit within 0.1% of each other (flushed: -0.24 / -0.04 / +0.03%). Paired on the
  same host with the constant columns dropped, flush-to-zero is bit-identical on HYE B01 for
  three seeds and -0.04% on Astral seed 1. Same seed on the same CPU model reproduces
  exactly; a different CPU generation does not (EPYC 7H12 against 9354), so pair A/B arms on
  one host. Judge a change on the mean over seeds on two pools,
  which is what CLAUDE.md already asks, and treat a 0.3% single-seed delta as nothing.
- Training is 85% of the rescore wall (fleet baseline: worker 1,106 s of a 19.4 min stage,
  944 s of it `train`; engine load 11 s, handoff write 26 s, post-processing 5 s), so the
  only cheaper recipes are ones that train less. Measured 2026-09-16 with seeds on two pools
  (Astral six-run pool, HYE B01): `folds: 2` is -45% wall (10.6 against 19.4 min) at -0.06%
  peptides on Astral (3 seeds, 116,236 against 116,309) but -1.1% on HYE B01 (62,297 against
  63,004), the smaller pool with the smaller training folds; `train_subsample: 0.5` -33% wall
  at -0.3% / -0.9%; `MUMDIA_NN_EARLY_STOP_TOL=0.03` -17% wall at -1.9%; `MUMDIA_NN_BATCH=16384`
  with `MUMDIA_NN_LR=2e-3` -10% to -24% wall at +0.07% on Astral and -0.6% on HYE B01 (3 seeds
  each; the notebook's 2.3x was measured on a GPU, where per-step overhead dominates);
  `MUMDIA_NN_INIT_TOPK=20000` -6% wall at -0.16% / identical, within seed spread; the three
  combined (`folds 2` + batch 16384 + init_topk) -55% wall at -0.34% on Astral and -2.0% on
  HYE B01 (3 seeds); progressive subsampling (30% of the training rows for the first seven
  rounds, all of them for the last three; experiment knob, not shipped) -47% wall at -0.14%
  on Astral and -0.85% on HYE B01 (3 seeds); `folds: 5` +0.09% at 2.0x the wall. None is a
  default; `folds: 2` is the fast option for a large pool. The one lever that gained on every
  pool is `train_neg_ratio: 2` (default 3): Astral +0.35% (116,711 against 116,309, 3 seeds),
  HYE B01 +0.15% (63,096 against 63,004, 3 seeds), entrapment +0.56% real peptides at an FDP
  of 1.025% against 1.044%, at -16% wall; it is the default since 2026-09-16 (this is the
  measurement the promotion rests on). `folds: 1` (in-sample scoring) is refuted
  by entrapment on the AIF spike-in library: +2.8% real peptides at an empirical FDP of 1.42%
  against 1.00% for folds 3, with the decoy fraction unchanged at 0.98%, so the decoys do not
  see the overfit and the count is not a gain.
- `rescore.features` / `features_file` project the classifier's input columns. 43 of the 387
  Extended features are dead by construction (10 constant under the default configuration, 20
  bit-identical, 13 affine duplicates), and about 114 chosen multivariately reproduce all 387
  within seed noise on both DIA-NN-library benchmarks. The shipped list is
  `bench/feature_selection/fs_union75_dedup.txt`. It was selected on a DIA-NN-library search
  and costs 2.1% on a FASTA-built entrapment library, so re-derive it per library type.
- `rescore.train_neg_ratio` / `train_neg_select` / `train_subsample` / `train_warm_epochs`
  thin what the sidecar trains on. The worker refits 30 times over the targets at 1% plus
  every decoy in the fold, which is 12-18 decoys per positive on HYE and 3:1 on AIF, and the
  gain from thinning is proportional to that imbalance. A cap is self-limiting and a quota is
  not: `train_neg_ratio: 5` never binds on a balanced pool and is 2.2x on an imbalanced one.
- Feature selection buys memory, not time: MLP training time per row is flat in the feature
  count from 387 down to 25. Training-set reduction buys time.
- The training recipe (`train_neg_ratio: 2` since 2026-09-16, previously 3; `train_neg_select:
  hybrid, train_warm_epochs: 5`) is the shipped default since 2026-09-05: measured with seeds
  against the previous defaults
  (every decoy, cold refits) on four pools, HYE A01 +1.0%, HYE B01 +2.2%, AIF -0.1%,
  entrapment +3.3% with the spike-in FDP unchanged, at 9-19x less training time (docs/28
  section 21; B01's baseline training took 50 minutes per seed against 2.6). `train_neg_ratio:
  0, train_neg_select: random, train_warm_epochs: 0` restore the previous behaviour exactly.
- `rescore.feature_preset = compact` (the embedded 114-feature list) stays opt-in. It is a
  memory lever, not a sensitivity one: the rescore matrix shrinks 3.4x (full-scale HYE
  rescore 5.49 GB / 3:19 against 13.5 GB / 6:20 with every feature, process-tree peaks; both
  sit under extract's 16.5 GiB), but under the
  default training it measured +0.2% / -1.2% / -0.1% / +1.5% on A01 / B01 / AIF / entrapment,
  and B01 is the pool the list was never fitted on. Use it for pooled rescoring on machines
  where the matrix would not fit (six HYE runs: 15.9 GB with it), not by default.

  It is CLASSIFIER-specific as well as library-specific, which is the stronger reason it
  cannot be a default. Every number above was measured with `nn_torch`. Under
  `native_tda` -- the shipped default classifier -- the same list produced ZERO
  identifications on the smoke fixture: 0 of 152 planted peptides, 0 peptides at 1%, and
  empty quant tables, against SMOKE_OK from the same binary with `feature_preset: all`
  (measured 2026-09-16 while trying to promote it to the default). Pair it with the
  classifier it was selected for, and re-derive the list for any other combination.
- The "sensitivity" recipe adds `folds: 5, train_margin_frac: 0.75, seeds: 3`: +0.4 / +0.4 /
  +0.6 pp over the fast recipe on HYE A01 / AIF / entrapment with the FDP unchanged, for 5.3x
  the rescore wall through the engine (18:38 against 3:31 on HYE B01, +0.2% peptides there);
  it is the option for a final pass, not a default.
- Do not set `train_neg_select: margin` with `train_neg_ratio: 1`. It is the fastest recipe
  and +1.27% on HYE, and it loses 10.35% on the entrapment pool.
- Judge any of these on at least two pools and with seeds. Seed 0 of the HYE baseline scored
  59,046 against 59,611 and 59,619 for seeds 1 and 2, which inflated every seed-0 comparison
  by about a percent.
- The rescorer's own hyperparameters (hidden 128-64, lr 1e-3, 25 epochs, dropout 0.3, batch
  4096, 10 iterations, train FDR 0.01, weight decay 1e-4) were swept with seeds on three
  pools (docs/28 section 17): every one is at or within noise of the optimum of its column,
  and every departure that helps one pool costs another. Do not retune them from a single
  benchmark. The only knob positive on every pool is `MUMDIA_NN_SEEDS=3` (+0.1 to +0.3 pp at
  3x training).
- The extraction and RT defaults (`gate_min_score` 0.2, `rt_window_multiplier` 1.5,
  `apex_count_window` 5) are likewise a measured local optimum on HYE end to end (docs/28
  section 18); `window_holdout_frac` is neutral there with a pre-fine-tuned library.
- Reference point, 2026-09-16: the six Astral files end to end (`mumdia run` with six
  `--mzml`, imported HYE library, multi-head calibration on the first run and reuse, pooled
  `nn_torch` rescore) take 52.5 min at a 13.25 GB process-tree peak on an EPYC 9354 with 32
  threads: per file convert 1.9-2.5 min, extract 1.6-2.0, features 0.5-0.8, search-seed 0.4,
  compete 0.2, quant 0.3 (about 5 min per file, 30 of the 52), then the pooled rescore 21.9
  min, and the multi-head calibration of the first run 11.7 min, which is also where the 13.2 GB
  peak sits (`extract.windows_in_flight: 8` left the peak at 13.5 GB: it is not extract's on this
  data). 113,160 peptides, 126,224 precursors, 12,132 protein groups at 1%. The rescore is 42%
  of the whole, the multi-head calibration 22% and the per-file chains the rest, so after this PR
  the next levers are the calibration's library re-prediction and convert + extract, not the
  classifier.
- Reference point, 2026-09-05: a complete HYE single run is 17:52 at 16.5 GiB on 32 threads
  (extract 16.5 GiB is the tallest stage, `extract.windows_in_flight: 8` takes it to 12.3),
  and the six-run pooled rescore is 18 minutes at 15.9 GB for 72,344 peptides.

### Sidecar and IO contracts

- `scripts/deeplc_worker.py` must `import deeplc` before numpy/pyarrow. DeepLC
  4.x is torch-backed, and on Windows the wrong order aborts torch DLL init with
  `OSError: [WinError 1114] ... Error loading torch\lib\c10.dll`.
  `deeplc_finetune.py` already orders its imports this way; keep both that way.
  The failure is latent because imported-library mode skips predict-frag, so
  only a FASTA-mode library build reaches it.
- `mumdia doctor` probes `deeplc,numpy,pandas,pyarrow,torch,psm_utils` for the
  DeepLC interpreter, because `deeplc_finetune.py` imports the last three too.
- Every DeepLC call site asks for the engine's thread count (the fine-tune's training
  pool keeps its own bound of 8), and both DeepLC workers cap what they give torch at the
  physical cores available to the process: the distinct sysfs `core_cpus_list` sets
  under `sched_getaffinity` on Linux, every physical core on Windows.
  `MUMDIA_DEEPLC_THREAD_CAP=N` overrides it and `0` disables it; the resolved numbers are
  under `torch_threads` in `<lib_out>.summary.json`. Measured on doxy (64 cores, 128
  CPUs): the multi-head step took 10:41 at 96 threads and 18:09 at 128. The cap is a
  ceiling, and a request at or below it is taken as given, but the engine asks for every
  logical CPU unless `--threads` says otherwise, so on an SMT host the cap binds by
  default (128 to 64 on doxy). Where it binds, the output changes as between any two
  `--threads` values: most rows in the last bits, and under the multi-head calibration a
  few sequences at the edge of the reference range by up to about two minutes (63 of
  12,002 fixture rows above 1 s, at most 129 s, same heads). Separately from the cap,
  the prediction after a fine-tune now runs on the engine thread count instead of the 8
  training threads, which moves the `finetune_deeplc` output in the last bits on every
  host with more than 8 cores (`DEEPLC_FT_THREADS` bounds training only). The default
  still owes the survey's doxy sweep (32-128 threads on HYE and AIF, head set, peptides at
  1% on `run_psm_q` over three NN seeds; docs/13, "DeepLC thread cap").
- `rt_im_train.deeplc_predict_shards` (default 1) splits that whole-library prediction
  across processes of `budget / K` threads, with the calibration or fine-tuned model
  fitted once in the parent and no refit per shard. Bit-identical to one process at
  equal threads per process, float-equivalent at the same `--threads`. It is not known to
  pay: on the one desktop measured the forward pass dominated and scaled with threads,
  so 4 x 2 threads equalled 1 x 8 (docs/08, "Sharded whole-library prediction"); it stays
  opt-in until measured where one process stops scaling, on two acquisitions.
- Any parquet written outside `mumdia-io` and read by the engine must be
  snappy-compressed with arrow `utf8` string columns. Polars defaults to zstd
  and `large_utf8`, and the engine rejects both ("Disabled feature at compile
  time: zstd", "column 'peptidoform' is not utf8").
- A library must carry `candidate_id` as the contiguous row-aligned range
  `0..ncand` (`index.rs:215-245`) and precursors ascending by `precursor_mz`
  (`index.rs:1288-1299`). Both are hard errors. Fragments are grouped by a
  counting sort, so they need valid ids but not a sorted order; a table whose
  ids ascend (what every library writer produces) takes the parallel fill.
- The `nn_torch` worker selects its backend at `MUMDIA_NN_STREAM_GB`
  (default 4). A feature matrix marginally over the threshold silently falls to
  the much slower disk-backed streaming memmap; a 4.31 GB matrix against the
  4.00 GB default took the slow path.
- `chromatograms.parquet` has two layouts. v1 (the default, schema 1) stores every row's
  whole `rt` axis and `intensity` trace. v2 (`extract.chromatogram_schema = 2`, schema
  2, `CHROMATOGRAMS_V2`) stores the lists as `rt_axis` and `intensity_trimmed`, plus
  `trace_offset` and `trace_len`: each candidate's axis once per parquet row group (an
  empty `rt_axis` means the last axis that candidate wrote in the same row group) and each
  trace trimmed to its first-to-last run of values that are not `+0.0`. A v2 row group
  decodes on its own, so a reader must start at a row group or at a candidate's first
  row. `mumdia::chromatograms::Decoder` is the reference reader and
  `mumdia::chromatograms::rewrite` converts either way; convert to v1 before handing the
  table to a tool outside the engine. The lists are renamed so that such a tool, or an
  engine binary from before v2, stops at the missing `rt` instead of misreading v2.

## Quantification rules

Identification and quantifiability are distinct. Keep an accepted ID even when
its signal cannot support a quantity.

Current correctness contract:

- the identification apex and feature bounds are carried into the scored row;
  quantification recomputes its configured integration bounds around that same
  apex, with compatibility fallback for older scored artifacts;
- absent/all-zero fragment evidence is unquantifiable (nullable quantity and
  status), not a valid abundance of zero;
- protein Top-N operates on unique `base_peptide_id` values rather than counting
  charge/modification rows as separate peptides;
- `precursor_q` is available for a single-run precursor output and is a genuine
  precursor unit under the default `compete.group_by = peptidoform_charge` (see
  the competition key below);
- for an experiment-wide rescore, split the scored table by `source` before
  invoking quant with each run's chromatograms. Changing `q_filter` does not
  select a source.

The default competition key is `peptidoform_charge` since 2026-09-06 (keys
`(pform_id, label, charge, peak_rank)`): sibling charge states and modforms of
one peptide are separate precursors that compete only against their own
alternative peaks, which is the unit DIA-NN reports at and the key every
docs/28 benchmark ran under (entrapment FDP flat at 0.48-0.64%, HYE, AIF). The
previous default `base_peptide` (renamed from `precursor`, which it was not)
keys the group on `(base_peptide_id, label_code, 0, peak_rank)`, and
`base_peptide_id` comes from the stripped sequence, so `compete.rs` deleted
every charge and every modification variant of one peptide but the highest
`prelim_score` before rescore: 23% of the extracted candidates on HYE B01, 46.6%
on a modification-rich library, at an unchanged peptide count. It remains
available as an explicit peptide-level population; never use it for a PTM
search.

Use Parquet quantities for analysis; TSV values are rounded for presentation.
Cross-run consensus ions, interference-aware ion selection, minimum clean-ion
rules, connected-component LFQ diagnostics, and coherent MBR requantification
remain open high-priority work.

## Defaults promoted on correctness grounds, not on a count

Four defaults changed because the previous value was wrong on its own terms, not
because a benchmark improved. None was promoted from a sensitivity measurement,
and each still needs entrapment plus a second acquisition before anyone claims a
sensitivity result for it.

- `extract.apex_evidence_rank` is now `true`. The legacy signature-intensity
  apex scores a scan group by the summed observed intensity of only the top-K
  *predicted* fragments, so when none of those K is observed at any qualifying
  scan the score is `0.0` everywhere, the strict `>` never replaces the first
  candidate, and the apex silently becomes the *lowest-RT qualifying scan*: up to
  a full RT window away, or anywhere in the gradient for a candidate with no
  window row. The RT prior cannot rescue it, because the combination is
  multiplicative and a zero annihilates the prior in exactly the case the prior
  exists for. Evidence rank scores `(n_distinct_fragments + tie) * prior`, which
  is always positive, so the fallback is unreachable. The wrong apex propagated
  into `prelim_score`, which decides the pre-FDR competition winner, and into
  quant's integration centre.

  The quantification question this left open is now measured (`bench/README.md`,
  HYE 3+3, 2026-08-29): quantification does not distinguish the two settings, every
  accuracy difference being under 0.013 in median |epsilon| and 0.002 in median CV,
  which is inside the DeepLC fine-tune draw variance the two arms carry. Extraction
  does distinguish them, in the promoted default's favour: the legacy apex pushes
  27.6% MORE candidates through extract (14.29 M against 11.20 M over six runs) and
  returns 1.0% FEWER peptides from them, costing an hour of pooled rescore. That is
  the fallback showing up as measured cost. Still one acquisition and no entrapment,
  so the promotion stays a correctness result rather than a sensitivity one.
- `extract.gate_min_score` stays `0.2`. It was briefly changed to `0.6`, the
  documented optimum for the default `native_tda` rescorer, and then measured: 0.6
  costs 4.4% of peptides for `native_tda` and 4.7% for `nn_torch` at an unchanged
  decoy fraction, because the gate sweep in `docs/18` was taken on the raw library
  before the current defaults and its optimum has since moved to the loose end for
  both rescorers. A default changed from a documentation claim, reverted by
  measurement.
- `features.emit_pin` is now `false`. No MuMDIA stage reads the file, because
  rescore builds its own PIN, and it is a ~5.4 GB text write per run on a real
  library.
- `predict_frag.ms2pip_model` is now `HCDch2` (2026-09-07). MS2PIP's single-charge
  models predict the singly charged b/y series only; the engine filled charge-2
  fragments with its native heuristic and max-normalised each charge group to its
  own peak, so the heuristics tied at 1.0 and crowded the predictions out of the
  top-N. On the HYE FASTA library (9.8M peptidoforms, one missed cleavage, 7-30,
  charges 2-3) built that way, 78.6% of the kept fragments were heuristics and the
  seed search found 0 confident PSMs at 1% on a real run (41.6% decoys among the
  top 1,000 seed scores; the DIA-NN library on the same spectra: 21,856 confident,
  0%), so `rt-im-train` had no anchors and the run continued with an unbounded RT
  window. `HCDch2` predicts the doubly charged series too; the worker returns it as
  `frag_charge` 2, `predict_frag::ms2pip_values` puts every fragment on the model's
  scale, and the same library gave 19,308 confident seeds (`HCD2021` with 12
  fragments and charge-2 only from charge 3: 14,412). The FASTA + sidecar example
  configurations set `top_n_fragments: 12`, the count the DIA-NN library ships;
  the engine default stays 6 because the native predictor was not re-measured.
  Precursor and fragment m/z of the FASTA library agree with the DIA-NN library to
  0.1 ppm on 3.66M shared keys, so this was intensity ranking, not mass. Library
  build for those 9.8M peptidoforms: DeepLC 19 min, MS2PIP 4.2.0 35-39 min at 32
  processes (docs/13). `MUMDIA_PREDICT_FRAG_CONCURRENT=1` (opt-in) runs the two
  workers at the same time, which leaves the library byte-identical; it is off because
  each worker sizes itself from the whole thread count and neither the wall time nor the
  peak of the pair was measured at that scale. `predict_frag.defer_deeplc_to_multihead`
  (opt-in) skips the DeepLC pass when the multi-head calibration re-predicts every row
  anyway (docs/06).

## Changes that remain benchmark-gated

Do not enable these by default from a single AIF count:

- model-visible top-K peaks (`extract.retain_top_peaks > 1`; implemented through
  features, compete and rescore, default 1);
- adaptive RT windows;
- held-out RT window sizing (`rt_im_train.window_holdout_frac`). Implemented and
  measured on the AIF benchmark: +1.1% peptides with DeepLC 4.1.0 at unchanged
  0.98% decoy, but -1.5% with the overfitting 4.0.0a2 model, so it interacts
  with RT-model quality and must not ship as default without the model switch
  plus entrapment and a second acquisition. `docs/08_rt_im_train.md` section 4b
  has the mechanism and numbers;
- alternative hard/soft extraction gates or peak apportionment;
- margin competition or unique-evidence competition;
- MBR transfer/re-extraction;
- acquisition-specific fragment/peak caps. The shipped default stays uncapped;
  see the peak-cap subsection above.

`compete.group_by = peptidoform_charge` is the default (2026-09-06) and is
required, not optional, for a PTM or modification search. Under `base_peptide`
the modified form is deleted whenever an unmodified or alkylated sibling scores
higher, which is usually. Measured on a modification-rich library, `base_peptide`
deleted 880,464 of 1,890,239 extracted candidates (46.6%); `peptidoform_charge`
removed 0 rows and moved precursors per peptide from 1.000 to 1.174 (DIA-NN
reports about 1.126 on comparable data), with an unchanged peptide count. The
gate for making it the default was the one CLAUDE.md sets for a changed training
and FDR population: the entrapment pool, HYE and AIF of docs/28 all ran under it.

The selected apex was historically correct/strongest only about 48-52% of the
time while the correct peak appeared in the top five about 86-88%. Promoting
top-K alternatives through features/rescore is therefore the best plausible
sensitivity project. The `candidate_id + peak_rank` contract exists (`peak_rank`
on every extracted row, `selected_peak_rank` on the scored row, and the MBR worker
joins it); what default activation still needs is the entrapment validation.

## Coding conventions

- Preserve deterministic ordering wherever floats are reduced. Use ordered maps
  or sort explicit keys; never depend on `HashMap` iteration for fits/sums.
- Keep target/decoy labels and grouping keys out of predictive features.
- Reuse the shared mass/constants and stats kernels; do not duplicate physical
  constants or correlation implementations.
- Config is serde-typed with `deny_unknown_fields`, but validation is targeted,
  not proof that every parameter combination is meaningful.
- Schema-changing columns require artifact version bumps and compatibility
  behavior where old artifacts are reasonably supportable.
- Use paired/collision-free decoys and retain target/decoy exchangeability.
- Maintain the clean-room boundary. Do not copy proprietary constants, maps, or
  code from DIA-NN or other closed implementations.

## Deployment

The Docker image contains:

- `/opt/mumdia/config.dia.json`: FASTA + MS2PIP/DeepLC + strict mokapot;
- `/opt/mumdia/config.diann-lib.json`: imported library + multi-head calibration of the
  library's retention times against each run, then the per-run LOESS (no fine-tune;
  `rt_im_train.finetune_deeplc` is available for a once-per-library fine-tune instead) +
  strict `nn_torch`. The multi-head step is the default rather than something this config
  sets, and it needs the image's DeepLC interpreter; without one the run still works and
  says in the log that it is not calibrating.

The Dockerfile copies both configs. MuMDIA consumes but does not ship or invoke a
DIA-NN binary; users create imported libraries under their own DIA-NN license.
