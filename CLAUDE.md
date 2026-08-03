# CLAUDE.md

Repository guide for coding agents and maintainers. Read this first, then use
`docs/README.md` to route to the code-grounded subsystem reference. `plan.md` is
the long algorithmic design history; when it disagrees with executable code,
tests and the tracked `docs/` guide describe current behavior.

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
- `scripts/`: seven engine-invoked Python workers plus four imported-library
  helpers (eleven scripts total), including `augment_library.py`, which adds the
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
mzML -> convert ----------------------------------+       v
        (optional: prescan, per-run tag pruning    |
         of modform hypotheses for a PTM search)   |
                                                  optional DeepLC fine-tune
                                                           |
                                                           v
                                      rt-im-train -> extract -> features
                                      -> compete -> rescore -> quant -> report
```

Key semantics:

- `run` is a single-run orchestrator. It always recomputes and overwrites its
  named outputs; the manifest is provenance, not a cache or resume database.
  Use a fresh output directory.
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
- The native digest emits N-terminal Met-excised forms by default
  (`digest.n_term_met_excision = true`, matching DIA-NN `--met-excision`).
  Excision keys on protein position 0 with a leading `M`, not any interior `M`.
  Without it the search database structurally misses those peptides; old configs
  still parse because the field defaults on. `augment_library.py` reuses this
  same digest to fill an imported library's missing tryptic peptides.
- Imported-library mode skips digest, peptidoform expansion, and initial
  prediction. Optional DeepLC fine-tuning still runs after seed search and
  writes a new precursor table rather than modifying the input.
- Stage-level candidate competition is within label, so it does not directly
  eliminate a target against its decoy. Peptide-level q estimation subsequently
  performs picked target-decoy competition through the shared
  `base_peptide_id`; keep that pairing intact.
- `retain_top_peaks > 1` currently writes diagnostic peak alternatives only.
  Those alternatives do not yet become feature/rescore rows.

## Validated sensitivity workflow

The strongest measured workflow on the chimeric AIF benchmark run
`LFQ_Orbitrap_AIF_Ecoli_01.mzML` uses the imported DIA-NN library, DeepLC
fine-tuning, Extended features, the loose `apex_pearson` extraction gate, and
`nn_torch` rescoring:

```text
mumdia doctor --config config.local-diann-lib.json

mumdia run \
  --lib-precursors lib/lib_precursors.parquet \
  --lib-fragments  lib/lib_fragments.parquet \
  --mzml mzml_files/LFQ_Orbitrap_AIF_Ecoli_01.mzML \
  --out-dir out_aif_nn \
  --config config.local-diann-lib.json \
  --top-peaks-ms2 300
```

Use the original imported precursor library here. Do not pass
`lib_precursors_ft.parquet` while also enabling fine-tuning. The
`--top-peaks-ms2 300` in that command reproduces this one benchmark. Do not
carry it to another acquisition; see the next subsection.

The historical result is about 10.3k confident precursor-shaped report rows,
selected by stripped-peptide q at 1%, versus roughly 9.3-9.5k with linear/native
or mokapot rescoring. It is a benchmark target, not a CI assertion or a universal
instrument preset.

### The peak cap is acquisition-specific, and 300 is not a default

`--top-peaks-ms2 300` belongs to that chimeric AIF run, where only 47.8% of MS2
spectra saturate the cap. Elsewhere it is actively harmful. On a 50-window
Orbitrap DIA run the same cap discarded 78.6% of all MS2 peaks and cost 60% of
the peptides (25,425 capped versus 63,237 uncapped) at an unchanged 0.99%
empirical decoy fraction, so the loss is sensitivity, not a loosened threshold.
The mechanism is peak-group formation rather than scoring: with most peaks gone,
`presence_min_fragments` cannot be met and real peptides are recorded
`NO_PEAK_GROUP`.

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

`extract.min_frag_corr` is named historically. Under the default
`gate_mode = apex_pearson`, it thresholds observed-versus-predicted fragment
intensities at one apex; it is not a chromatographic co-elution correlation.
The observed optimum was about 0.2 with the NN and tighter with the native
linear model.

Three RT rules, each measured in `docs/08_rt_im_train.md` and restated from the
failure side in `docs/17_troubleshooting.md`:

- DeepLC fine-tuning of library iRT is the largest RT lever (historically
  reducing residuals from about 110 s to 13-27 s) and must happen, but it need
  not happen per file. A library fine-tuned once and predicted over every
  peptidoform, combined with the per-run LOESS calibration and
  `rt_im_train.finetune_deeplc = false`, measured equal or marginally better
  residuals than per-file fine-tuning while removing about 36 minutes per file.
  This does not license reusing a stale per-run `_ft` table built on a different
  file, which has previously underperformed a fresh fit.
- The `cal.json` RT residuals are in-sample and optimistic. `rt_im_train.rs:137`
  fits the loess and `rt_im_train.rs:177-185` derives `w_rt` from a residual
  percentile on the same anchor points, so the reported
  `rt_residual_abs_median_s` was roughly 3x better than the same calibration
  scored out-of-sample. Treat it as a fit diagnostic, not an error estimate, and
  size any external RT tolerance from out-of-sample numbers.
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
  - `precursor_q`: peptidoform plus charge, but only under
    `compete.group_by = peptidoform_charge`. The default key already deleted the
    sibling rows, so it then counts base peptides (measured 1.000 precursors per
    peptide, against 1.174 with `peptidoform_charge`);
  - `peptide_q_value`: base/stripped peptide;
  - `pg_q_value`: protein-accession-set group.
- The grouped q columns (`peptide_q_value`, `precursor_q`, `pg_q_value`) are
  written only to each group's single winning row (`rescore.rs:721-728`); losers
  get 1.0. Under an experiment-wide rescore the grouping is experiment-wide, so
  a per-run count on those columns is diluted by roughly 1/n_runs and is
  meaningless. The correct per-file unit there is `run_psm_q`.
- Pooling more runs does not tighten q. `fdr.rs:38` computes
  `q = (decoys + 1) / max(1, targets)`, which is scale-invariant under
  replicating the population. The only pool-size term is the +1 pseudocount,
  which makes a larger pool marginally looser. Do not attribute per-run count
  changes to pool size.
- Reported benchmark counts must name their row and q-value unit. `peptides.tsv`
  contains `(peptidoform, charge)` rows but is selected with
  `peptide_q_value`; it is not a precursor-q report.
- Validate new sensitivity defaults with entrapment or another empirical null,
  plus at least two datasets/acquisition contexts. Count gains alone are
  insufficient.

### Experiment-wide rescore

- `run-experiment` never calls the report stage. There is no `peptides.tsv` or
  `proteins.tsv` anywhere in its output tree. Per-run counts come from the split
  scored tables or from `mumdia report` invoked manually.
- `run-experiment` overrides the configured `quant.q_filter` and gates per-run
  quant on the pooled `q_value`. It warns rather than doing so silently.
- `rescore --competed` accepts many tables, stamps `source` with the index of
  the input table each PSM came from (`rescore.rs:65-70,108`), and computes a
  per-source `run_psm_q` alongside the pooled `q_value`
  (`rescore.rs:403-408`). Pooling therefore never costs per-run FDR, and
  sub-batching a large experiment is statistically free. Batch only to fit RAM.
- Pooled rescore scales linearly, measured 0.834 ms/PSM on the streaming
  backend. The feature matrix is `n_psms x n_features x 4` bytes; size batches
  from that.

### Sidecar and IO contracts

- `scripts/deeplc_worker.py` must `import deeplc` before numpy/pyarrow. DeepLC
  4.x is torch-backed, and on Windows the wrong order aborts torch DLL init with
  `OSError: [WinError 1114] ... Error loading torch\lib\c10.dll`.
  `deeplc_finetune.py` already orders its imports this way; keep both that way.
  The failure is latent because imported-library mode skips predict-frag, so
  only a FASTA-mode library build reaches it.
- `mumdia doctor` probes `deeplc,numpy,pandas,pyarrow,torch,psm_utils` for the
  DeepLC interpreter, because `deeplc_finetune.py` imports the last three too.
- Any parquet written outside `mumdia-io` and read by the engine must be
  snappy-compressed with arrow `utf8` string columns. Polars defaults to zstd
  and `large_utf8`, and the engine rejects both ("Disabled feature at compile
  time: zstd", "column 'peptidoform' is not utf8").
- A library must carry `candidate_id` as the contiguous row-aligned range
  `0..ncand` (`index.rs:112-125`) and precursors ascending by `precursor_mz`
  (`index.rs:215-231`). Both are hard errors. Fragments are grouped by a
  counting sort, so they need valid ids but not a sorted order.
- The `nn_torch` worker selects its backend at `MUMDIA_NN_STREAM_GB`
  (default 4). A feature matrix marginally over the threshold silently falls to
  the much slower disk-backed streaming memmap; a 4.31 GB matrix against the
  4.00 GB default took the slow path.

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
- `precursor_q` is available for a single-run precursor output, but it is a
  genuine precursor unit only under `compete.group_by = peptidoform_charge` (see
  the competition key below);
- for an experiment-wide rescore, split the scored table by `source` before
  invoking quant with each run's chromatograms. Changing `q_filter` does not
  select a source.

The default competition key `precursor` is a misnomer. `compete.rs:88` keys the
group on `(base_peptide_id, label_code, 0, peak_rank)`, and `base_peptide_id`
comes from the stripped sequence (`import_diann_lib.py:137` factorises
`Stripped.Sequence`). `compete.rs:319-340` keeps only the highest `prelim_score`
per group and deletes the rest before rescore, so every charge and every
modification variant of one peptide collapses to a single winner pre-FDR.
`peptidoform_charge` (`compete.rs:93-98`, keys
`(pform_id, label, charge, peak_rank)`) is the quant-oriented alternative; see
the benchmark-gating section for when it is required rather than optional.

Use Parquet quantities for analysis; TSV values are rounded for presentation.
Cross-run consensus ions, interference-aware ion selection, minimum clean-ion
rules, connected-component LFQ diagnostics, and coherent MBR requantification
remain open high-priority work.

## Changes that remain benchmark-gated

Do not enable these by default from a single AIF count:

- model-visible top-K peaks (currently diagnostic sidecar only);
- adaptive RT windows;
- alternative hard/soft extraction gates or peak apportionment;
- `peptidoform_charge` competition as a general default, margin competition, or
  unique-evidence competition;
- MBR transfer/re-extraction;
- acquisition-specific fragment/peak caps. The shipped default stays uncapped;
  see the peak-cap subsection above.

`compete.group_by = peptidoform_charge` is required, not optional, for a PTM or
modification search. Under the default key the modified form is deleted whenever
an unmodified or alkylated sibling scores higher, which is usually. Measured on
a modification-rich library, the default key deleted 880,464 of 1,890,239
extracted candidates (46.6%); `peptidoform_charge` removed 0 rows and moved
precursors per peptide from 1.000 to 1.174 (DIA-NN reports about 1.126 on
comparable data), with an unchanged peptide count. It stays gated only as a
change to the shipped default for non-PTM searches, because it changes the
training and FDR population.

The selected apex was historically correct/strongest only about 48-52% of the
time while the correct peak appeared in the top five about 86-88%. Promoting
top-K alternatives through features/rescore is therefore the best plausible
sensitivity project, but it needs a `candidate_id + peak_rank` contract and
entrapment validation before default activation.

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
- `/opt/mumdia/config.diann-lib.json`: imported library + per-run DeepLC
  fine-tune + strict `nn_torch`.

The Dockerfile copies both configs. MuMDIA consumes but does not ship or invoke a
DIA-NN binary; users create imported libraries under their own DIA-NN license.
