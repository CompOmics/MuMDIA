# Python sidecars (the 10 scripts) + conda envs
> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

MuMDIA has deterministic native implementations for its default predictor and
rescorer paths, so it runs with zero Python. The sidecars are the opt-in path to real
predictors and rescorers that raise identification counts over the native
defaults. Each sidecar is a standalone Python worker invoked as a subprocess over
a **positional-CLI file contract** (the sidecar file contract; see
`docs/18_findings_and_decisions.md`): the Rust caller writes an input file
(Parquet or PIN), runs `python <worker> <argv...>`, and reads an output Parquet
keyed by `id` or `candidate_id`. There is no JSON request file, no long-lived
server, and no stdin/stdout data channel; the files on disk are the entire
interface. Any sidecar can be swapped for a native Rust implementation (or a
different tool) without touching the callers. Failure handling is **not uniform**:
only the rescore sidecars fall back to a native rescorer, and only when
`rescore.strict = false`. The predictor sidecars and MBR have no fallback and no
strict gate, so a crashed MS2PIP, DeepLC, or DeepLC fine-tune aborts the whole
run. The matrix in **Failure behavior** below is authoritative.

The 10 scripts split into four groups: **predictors** (`ms2pip_worker`,
`deeplc_worker`, `deeplc_finetune`) feed the run-independent library;
**rescorers** (`mokapot_worker`, `nn_rescore_worker`, `entrapment_worker`) score
the competed PSMs in Stage F; **MBR** (`mbr_worker`) transfers identifications
across runs (Stage D3); and the **DIA-NN recipe** (`import_diann_lib`,
`make_reverse_decoys`, `make_shift_decoys`) is an offline, one-time toolchain to
convert a user-produced DIA-NN library into MuMDIA's target+decoy library schema.

## Failure behavior

Two groups, verified in the source. Only the rescore sidecars fall back; the
predictor sidecars and MBR abort the run on any nonzero exit. `run_worker` turns a
nonzero exit into an `Err` (`sidecar.rs:229-231`); the call site decides whether
that `Err` aborts or is caught.

| worker (stage) | on nonzero exit | strict gate | fallback |
|---|---|---|---|
| ms2pip_worker (predict-frag) | aborts run | none | none: `run_ms2pip(...)?` propagates (`predict_frag.rs:333-341`); an empty result map is also a hard `bail!` (`predict_frag.rs:342-344`) |
| deeplc_worker (predict-frag) | aborts run | none | none: `run_deeplc(...)?` propagates (`predict_frag.rs:291`) |
| deeplc_finetune (run) | aborts run | none | none: `run_deeplc_finetune(...)?` propagates (`run.rs:253-263`) |
| mbr_worker (`mumdia mbr`) | aborts command | none | none: `run_mbr(...)?` propagates (`main.rs:697-710`) |
| mokapot_worker (rescore) | falls back / bails | `rescore.strict` | `native_tda` when `strict=false`, else bail (`rescore.rs:172-180`) |
| nn_rescore_worker (rescore) | falls back / bails | `rescore.strict` | `native_tda` when `strict=false`, else bail (`rescore.rs:199-207`) |
| percolator (rescore, unwired) | falls back / bails | `rescore.strict` | `native_tda` when `strict=false`, else bail (`rescore.rs:209-215`) |
| entrapment_worker (rescore) | falls back / bails | `rescore.strict` | native linear entrapment rescorer (still `QMode::Entrapment`) when `strict=false`, else bail (`rescore.rs:255-264`) |

A crashed MS2PIP, DeepLC, DeepLC fine-tune, or MBR worker aborts. A crashed
rescorer aborts only under `rescore.strict = true`; otherwise its scores are
silently replaced by the native path.

### Argv contract

The positional arguments each Rust caller passes, the output file it reads, and
the column it keys the readback on.

| worker | positional args in | output file | key column |
|---|---|---|---|
| ms2pip_worker | `<in.parquet> <out.parquet> <model>` (`sidecar.rs:63`) | `ms2pip_out.parquet` | `id` |
| deeplc_worker | `<in.parquet> <out.parquet>` (`sidecar.rs:99`) | `deeplc_out.parquet` | `id` |
| deeplc_finetune | `<lib_in> <seed> <lib_out> --epochs --patience --q-train --batch` (`sidecar.rs:139-151`) | `<lib_out>` (= `fragment_library_precursors_ft.parquet`) | `peptidoform` (new table with replaced `predicted_irt`; input unchanged) |
| mokapot_worker / nn_rescore_worker | `<rescore.pin> <out.parquet>` + env `MUMDIA_NN_FOLDS/ITERS/TRAIN_FDR` (`rescore.rs:781-792`) | `rescore_sidecar_out.parquet` | `candidate_id` (echoes the flat row index) |
| entrapment_worker | `<in.parquet> <out.parquet> <folds>` (`rescore.rs:718-724`) | `entrapment_out.parquet` | `row_id` |
| mbr_worker | `<scored> <psms_csv> <out> --q-anchor --min-anchor-runs --q-transfer --seed [--out-scored] [--frag-csv --consensus-corr-min]` (`sidecar.rs:193-211`) | `<out>.parquet` | `candidate_id` |

For the mapping from each conda environment to the config field that points at it
(`predict_frag.ms2pip_python`, `predict_frag.deeplc_python`, `rescore.python`,
`mbr.python`) and the interpreter paths on this machine, see
`docs/19_getting_started.md`.

## Files

| path | role |
|---|---|
| `scripts/ms2pip_worker.py` | Predictor: MS2PIP b/y fragment intensities per peptidoform+charge |
| `scripts/deeplc_worker.py` | Predictor: DeepLC iRT per peptidoform (uncalibrated) |
| `scripts/deeplc_finetune.py` | Predictor: transfer-learn DeepLC on this run's seed, rewrite library iRT |
| `scripts/mokapot_worker.py` | Rescorer: mokapot brew over a PIN (model env-switchable: nn/logreg/xgb/percolator) |
| `scripts/nn_rescore_worker.py` | Rescorer: PyTorch semi-supervised MLP over a PIN, in-memory or streaming memmap |
| `scripts/entrapment_worker.py` | Rescorer: GBM/NN on real-target-vs-spike-in negatives, out-of-fold by base peptide |
| `scripts/mbr_worker.py` | MBR (Stage D3): cross-run RT transfer + permuted-RT decoy-transfer FDR |
| `scripts/import_diann_lib.py` | Recipe: DIA-NN fragment-level parquet -> MuMDIA target `lib_precursors`+`lib_fragments` |
| `scripts/make_reverse_decoys.py` | Recipe: reverse-sequence decoys with no-target-overlap invariant |
| `scripts/make_shift_decoys.py` | Recipe: fragment-shift (CH2) decoys, DIA-NN-style terminal shift |
| `rust/mumdia/crates/mumdia/src/sidecar.rs` | Rust clients: `resolve_script`, `run_ms2pip`, `run_deeplc`, `run_deeplc_finetune`, `run_mbr`, `run_worker` |
| `rust/mumdia/crates/mumdia/src/stages/predict_frag.rs` | Call sites for MS2PIP + DeepLC (Stage C) |
| `rust/mumdia/crates/mumdia/src/stages/run.rs` | Call site for DeepLC fine-tune (between search-seed and rt-im-train) |
| `rust/mumdia/crates/mumdia/src/stages/rescore.rs` | Call sites for mokapot/nn_torch (PIN) + entrapment (Parquet) sidecars |
| `rust/mumdia/crates/mumdia/src/main.rs` | Call site for MBR (`Cmd::Mbr`) + `doctor` env probe |
| `env/docker-rescore.yml` | Docker env `rescore`: mokapot 0.10.0 + ms2pip 4.0.0.dev9 (py3.11) |
| `env/docker-deeplc.yml` | Docker env `deeplc`: DeepLC 4.0 multitask (pinned commit) + CPU torch (py3.11) |
| `env/mumdia-rescore.yml` | Minimal portable env for the mokapot logreg rescore path (py3.12) |

## Inputs and outputs

Each sidecar's on-disk contract, with the exact column schema read/written by the
code.

**ms2pip_worker** (`sidecar.rs:42` `run_ms2pip`)
- IN `ms2pip_in.parquet`: `id` u32, `peptidoform` str (ProForma), `charge` i32.
- OUT `ms2pip_out.parquet`: `id` u32, `ion_type` str (`"b"`/`"y"`), `ordinal` i32
  (1-based), `intensity` f32 (linear). Rust folds this into
  `HashMap<u32, HashMap<(u8 ion_byte, u16 ordinal), f32>>` (`sidecar.rs:70-76`).

**deeplc_worker** (`sidecar.rs:81` `run_deeplc`)
- IN `deeplc_in.parquet`: `id` u32, `peptidoform` str.
- OUT `deeplc_out.parquet`: `id` u32, `predicted_rt` f32. Rust returns
  `HashMap<u32, f32>` (`sidecar.rs:104`).

**deeplc_finetune** (`sidecar.rs:111` `run_deeplc_finetune`)
- IN `<lib_in>` = `fragment_library_precursors.parquet` (needs `peptidoform`,
  `predicted_irt`); `<seed>` = seed PSMs (`peptidoform`, `label`, `spectrum_q`,
  `observed_rt`).
- OUT `<lib_out>` = `fragment_library_precursors_ft.parquet`: the input table with
  the `predicted_irt` column replaced (`deeplc_finetune.py:156-159`). Same schema,
  values rewritten.

**mokapot_worker** / **nn_rescore_worker** (`rescore.rs:740` `run_pin_sidecar`)
- IN `rescore.pin`: Percolator tab-separated. Fixed columns
  `SpecId Label ScanNr ExpMass CalcMass <features...> Peptide Proteins`
  (`rescore.rs:760-777`). `SpecId = psm_<i>`, `ScanNr = <i>` where `i` is the
  unique flat row index (NOT `candidate_id`, which repeats across runs);
  `ExpMass=CalcMass=precursor_mz` (`{:.5}`); features are `{:.6}`; `Label` +1
  target / -1 decoy; `Peptide = -.<peptidoform>.-` (Percolator flanking dots);
  `Proteins = <protein>` (single column, `rescore.rs:776`). `nn_rescore_worker.py`
  builds its fold key from the `Peptide` column, stripping the mod brackets and
  the `X.`/`.X` flanks (`strip_pep`, `nn_rescore_worker.py:70-74`) before the
  `md5 % FOLDS` hash. The feature column order is taken from the first competed
  input's `FeatureSchema` companion (`rescore.rs:74`), so all competed inputs must
  share one schema.
- OUT `rescore_sidecar_out.parquet`: `candidate_id` u32 (echoes the SpecId tail =
  row index), `score` f64, `q_value` f64 (written as zeros; Rust computes q). Rust
  maps `score` back by row index and validates exact/unique/finite coverage
  (`align_sidecar_scores`, `rescore.rs:802-805`).

**entrapment_worker** (`rescore.rs:675` `run_entrapment_gbm`)
- IN `entrapment_in.parquet`: `row_id` u32 (unique flat index), `candidate_id`
  u32, `base_peptide_id` u32, `is_entrapment` i32 (0/1), `is_decoy` i32 (0/1),
  then one f64 column per feature (`rescore.rs:693-714`).
- OUT `entrapment_out.parquet`: `row_id` u32, `candidate_id` u32, `score` f64.
  Rust maps back by `row_id` (`rescore.rs:729-732`).

**mbr_worker** (`sidecar.rs:162` `run_mbr`)
- IN `<scored_combined>`: experiment-wide scored table, columns read =
  `candidate_id, source, label, q_value, peptidoform, charge, protein_group`
  (`mbr_worker.py:81-82`). `<psms_csv>`: comma-joined per-run psms.parquet paths
  in `source` order, each read for `candidate_id, apex_rt` (`mbr_worker.py:96`).
  Optional `--frag-csv`: per-run fragment_quant paths
  (`candidate_id, fragment_name, quantity`).
- OUT `<out>.parquet`: one row per accepted transfer with
  `candidate_id, source, peptidoform, charge, protein_group, label, expected_rt,
  observed_rt, rt_delta, transfer_q` (`mbr_worker.py:254-265`). Optional
  `--out-scored` writes the scored table with accepted transfers' `q_value`
  lowered to `transfer_q` and an `is_transferred` flag added. Optional
  `--emit-transfer-targets` writes per-run `run_windows`-format tables
  (`candidate_id, rt_pred_cal, rt_lo, rt_hi, im_*`) plus a permuted-RT decoy file
  for the re-extraction tier (`mbr_worker.py:142-151`).

**import_diann_lib** (offline; no Rust caller)
- IN `<diann_lib.parquet>`: DIA-NN fragment-level speclib (columns
  `Decoy, Fragment.Loss.Type, Fragment.Type, Modified.Sequence, Precursor.Charge,
  Precursor.Mz, RT, Stripped.Sequence, Product.Mz, Relative.Intensity,
  Fragment.Series.Number, Fragment.Charge, Protein.Names`/`Protein.Ids`).
- OUT `<out_precursors>`: `candidate_id, peptidoform_id, base_peptide_id,
  peptidoform, charge, precursor_mz, predicted_irt, label(="target"), protein,
  n_fragments`. `<out_fragments>`: `candidate_id, mz, predicted_intensity, name,
  ion_type, ordinal, frag_charge` (`import_diann_lib.py:59-83`).

**make_reverse_decoys** / **make_shift_decoys** (offline; no Rust caller)
- IN/OUT the same precursor+fragment schema as `import_diann_lib`, reading a
  target-only (or target-half) library and emitting a target+decoy library
  re-sorted by `precursor_mz` with contiguous re-indexed `candidate_id`.

## How it works

### Predictors (Stage C, and one Stage-B step)

**MS2PIP** (`predict_frag.rs:324-393`, worker `ms2pip_worker.py`). Selected by
`predict_frag.predictor = "ms2pip"`. `assign_intensities` collects one `id` per
`Raw` candidate, its peptidoform and charge (the `id` is the flat `raws` index,
not `candidate_id`, `predict_frag.rs:330`), calls `run_ms2pip`, then for each
fragment looks up `(ion_byte, ordinal)` for **charge-1** fragments only; charge-2
fragments fall back to the native heuristic (`predict_frag.rs:356-363`). Because
MS2PIP charge-1 (TIC-fraction, ~0.02-0.3) and the native charge-2 fallback
(max-normalized, ~0.19-0.5) live on different scales, ranking them together in
top-N would bury MS2PIP, so each charge group is max-normalized to its own peak
before they compete (`predict_frag.rs:365-384`). Two native-fallback edge cases:
`run_ms2pip` returning an empty map is a hard error (`bail!("MS2PIP returned no
predictions")`, `predict_frag.rs:342-344`), while a single candidate that MS2PIP
returned nothing for (missing/empty per-id entry) falls back wholesale to the
native intensities for that candidate (`predict_frag.rs:387-389`). The
worker builds `psm_utils.PSMList` in 100k-row chunks, calls
`ms2pip.predict_batch(model, processes=min(8, cpu_count))`, and converts MS2PIP's
log2 intensities to linear via `2**x - 0.001` clipped at 0
(`ms2pip_worker.py:52-53`). Ordinals are emitted 1-based
(`ms2pip_worker.py:57`). The `__main__` guard makes the Windows `spawn` start
method safe for multiprocessing.

**DeepLC predict** (`predict_frag.rs:274-312`, worker `deeplc_worker.py`).
Selected by `predict_frag.rt_predictor = "deeplc"`. `assign_rt` deduplicates by
peptidoform (RT is charge-independent, `predict_frag.rs:281-290`), calls
`run_deeplc`, and writes `r.irt`. Peptidoforms with no returned iRT are anchored
at `0.0` with a warning (`predict_frag.rs:293-308`; this is the "unmatched
peptidoforms silently get iRT 0.0" foot-gun noted in CLAUDE.md). The worker calls
`deeplc.predict` in 200k chunks and, when the multitask model returns an
ensemble matrix `(N, n_models)`, averages across models (`deeplc_worker.py:34-35`).
Predictions are uncalibrated; rt-im-train's per-run LOESS/linear maps them onto
observed RT.

**DeepLC fine-tune** (`run.rs:242-263`, worker `deeplc_finetune.py`). Wired into
`run` only when `rt_im_train.finetune_deeplc = true`; runs **between**
search-seed and rt-im-train, rewriting `predicted_irt` in a copy of the library
(`fragment_library_precursors_ft.parquet`) that rt-im-train and extract then read.
Algorithm: (1) build the reference from confident **target** seed PSMs with
`spectrum_q <= q_train` and a standard-AA sequence (`deeplc_finetune.py:99-104`);
(2) auto-scale batch size so each epoch runs >= ~30 gradient steps, clamped to
[16, 512] (`deeplc_finetune.py:114-117`) because a fixed 512 underfits a small
(~4k) E.coli seed; (3) `deeplc.finetune(ref_psms, train_kwargs)` transfer-learns
the weights (`deeplc_finetune.py:128`); (4) predict every unique standard
peptidoform on its **`DECOY_`-stripped** underlying sequence so decoys land on the
same iRT scale as targets (`deeplc_finetune.py:44-45, 135-156`). A peptidoform
that is non-standard (`is_std` false, e.g. a terminal mod outside `STD`) or was
not predicted keeps its **original** `predicted_irt` unchanged, because the
write-back is `preds.get(base_pf(pf), orig[i])` (`deeplc_finetune.py:156`), so
only the sequences DeepLC actually re-predicted move onto the fine-tuned scale.
Beyond the four flags Rust passes (`--epochs/--patience/--q-train/--batch`,
`sidecar.rs:139-151`), the worker exposes CLI-only knobs that `run` never sets:
`--device cpu|cuda` (cuda aborts with `SystemExit` if `torch.cuda.is_available()`
is false, `deeplc_finetune.py:79-81`), `--threads` (torch CPU pool, defaults to
`DEEPLC_FT_THREADS`), `--max-ref N` (cap reference PSMs), `--predict-limit N`
(cap peptidoforms predicted), and `--skip-predict` (fine-tune only, exercise the
crash path without the full-library prediction, `deeplc_finetune.py:71-76,
131-133`). The long
docstring and the thread-cap block at the top exist to prevent an intermittent
machine crash: numpy's OpenBLAS (GNU OpenMP) and torch's Intel OpenMP coexist
under `KMP_DUPLICATE_LIB_OK=TRUE`, and without pinning `OMP/MKL/OPENBLAS` to 1
thread and bounding torch's pool, the two full thread pools oversubscribe the CPU
during the backward pass (`deeplc_finetune.py:6-28, 82-91`). `--device cuda`
sidesteps this entirely by moving compute off the CPU pools.

### Rescorers (Stage F)

**mokapot** and **nn_torch** share the exact PIN contract via `run_pin_sidecar`
(`rescore.rs:740-806`). Rust concatenates the competed feature tables, writes the
PIN, spawns the worker, and passes `MUMDIA_NN_FOLDS/ITERS/TRAIN_FDR` from
`rescore.{folds,num_iter,train_fdr}` as env vars (`rescore.rs:790-792`) so the
report's recorded params match what ran; mokapot ignores those three. On success
the classifier label and `model_identity` are recorded; on failure the path falls
back to `native_scores` only when `rescore.strict = false`; strict is the
production default. The authoritative actual path is recorded in
`psms_scored.parquet.report.json`.

- `mokapot_worker.py` reads the PIN with `mokapot.read_pin`, builds a model
  chosen by `MUMDIA_RESCORE_MODEL` (`make_model`, `mokapot_worker.py:35-98`:
  `nn` -> sklearn `MLPClassifier`; `logreg` -> `LogisticRegression`; `xgb` ->
  `XGBClassifier`; `percolator`/`linear`/`svm` -> mokapot's default, `model=None`),
  runs `mokapot.brew(..., rng=0, max_workers=MUMDIA_MOKAPOT_WORKERS)`, and uses
  mokapot's **out-of-fold** confidence scores only (each PSM scored by the fold
  that did not train on it, `mokapot_worker.py:128-163`). There is deliberately no
  in-sample fallback: `_oof_scores` raises `RuntimeError` unless the merged
  target+decoy confidence tables cover the PIN rows exactly once with finite
  scores (`mokapot_worker.py:152-162`), matching the CLAUDE.md rule that
  fold-model averaging is not an acceptable fallback. That `RuntimeError` is a
  nonzero worker exit, which the Rust caller then treats per `rescore.strict`
  (bail if strict, else `native_tda`). Every PSM (targets and decoys) is scored,
  `SpecId` tail parsed to `candidate_id` (`mokapot_worker.py:171`).
- `nn_rescore_worker.py` implements the semi-supervised scheme itself in PyTorch.
  Fold assignment is `md5(stripped_peptide) % FOLDS` (`nn_rescore_worker.py:127`),
  so peptides never leak across folds and the split is deterministic. Per fold it
  selects the initial feature+sign using a deterministic sample of training rows
  only, then iterates {recompute target-decoy q on the training folds -> targets
  at `q<=TRAIN_FDR` positive, all decoys negative -> train MLP from scratch ->
  rescore} for `ITERS` rounds, then scores the held-out fold. Empty,
  single-class, and zero-positive training folds hard-error; held-out labels do
  not influence model selection. Two feature
  backends behind one accessor `get`: in-memory (median/IQR standardisation,
  `:130-145`) or a disk-backed float32 **memmap** streamed in `MUMDIA_NN_CHUNK`
  chunks with mean/std accumulated in one text pass (`:146-177`), auto-selected
  when the PIN exceeds `MUMDIA_NN_STREAM_GB` (4 GB). This is what makes an
  experiment-wide multi-run rescore tractable: the full PIN never lives in RAM.
  `tda_q` (`:77-87`) is the shared q formula `(decoys+1)/max(1,targets)`, running
  min from the tail. Seeds are ensembled by averaging rank-normalised OOF scores
  (`:281-288`).

**entrapment** (`rescore.rs:675-733`, worker `entrapment_worker.py`). Selected by
`rescore.classifier = "entrapment"` with `rescore.entrapment_marker` set and
`rescore.python` present; otherwise it falls back to a native linear entrapment
rescorer or `native_tda` (`rescore.rs:217-276`). `classify_entrapment`
(`rescore.rs:577-604`) marks a target as entrapment when its protein contains the
marker, does not contain `entrapment_exclude`, and matches none of
`entrapment_contaminant_markers`. The worker trains real-target (positive) vs
spike-in (negative), decoys excluded from training (`entrapment_worker.py:80-83`),
out-of-fold with `GroupKFold` grouped by `base_peptide_id`
(`entrapment_worker.py:93-101`); a final model fit on all non-decoy PSMs scores
decoys and any single-class-fold gaps (`:103-108`). Model is `gbm`
(`HistGradientBoostingClassifier`, `early_stopping=False` so `random_state=0` is
reproducible) or `nn` (StandardScaler + MLP pipeline) via
`MUMDIA_ENTRAPMENT_MODEL` (`entrapment_worker.py:28-60`). The rationale: spike-in
negatives experience the same chimeric DIA interference as real targets, so a
flexible model helps (AUC ~0.97 vs ~0.62 on in-silico decoys), unlike the
decoy-trained regime, where a native linear model is all in-silico decoys can
support. Selecting entrapment (whether the GBM
sidecar or the native linear fallback) flips the internal `QMode` from `Decoy` to
`Entrapment` (`rescore.rs:145, 252/262/273`), so every q level (PSM, per-run,
peptide, protein-group, precursor) is computed by `entrapment_q` against the
real-target-vs-spike-in null scaled by `rescore.entrapment_ratio`, and the
reported IDs are the real targets only (spike-in excluded, `rescore.rs:400-403`).
The report also records `entrapment_peptides_at_1pct`, a leak check on spike-in
peptides passing the 1% gate (`rescore.rs:437-445, 492`).

### MBR (Stage D3)

`mbr_worker.py` (`main.rs:671-711` `Cmd::Mbr`, `sidecar.rs:162` `run_mbr`).
Requires `mbr.strategy != none`, `>= 2` runs, and `mbr.python`
(`main.rs:680-692`). Reads the experiment-wide scored table and per-run apex RTs.
It builds per-run to-reference / from-reference RT maps by monotone binned-median
calibration against run 0 (`binned_map`, `mbr_worker.py:31-43`, needs >= 200
shared anchors else identity). For a precursor confident (target, `q<=q_anchor`)
in `>= min_anchor_runs` OTHER runs but sub-threshold in a target run where it was
still extracted (the **rescuable** tier), it predicts RT as the from-ref map of
the median of the other runs' to-ref-mapped apex RTs (`expected_rt`,
`mbr_worker.py:116-121`). The false-transfer FDR uses a **permuted-RT
decoy-transfer null**: each candidate is assigned a shuffled candidate's predicted
RT, and transfer q is standard target/decoy competition on `|observed - predicted|`
(`mbr_worker.py:190-199`). An optional fragment-consensus cosine guard
(`--frag-csv`/`--consensus-corr-min`) rejects RT-concordant interference
(`mbr_worker.py:209-241`). `--emit-transfer-targets` instead emits per-run
`run_windows` for the **absent** set (confident elsewhere, not extracted here) so
`extract --restrict-candidates --run-windows` can re-extract them (the
re-extraction tier). `run_mbr` (`sidecar.rs:162-213`) drives only the **rescuable
tier**: it passes `<scored> <psms_csv> <out>`, the `--q-anchor/--min-anchor-runs/
--q-transfer/--seed` values, and optionally `--out-scored` and (only when `frag`
paths are given and `consensus_corr_min > 0`) `--frag-csv/--consensus-corr-min`.
It never passes `--emit-transfer-targets` or `--rt-window`, so the re-extraction
tier is a manual worker invocation. `--seed` receives the engine-wide
`rng_seed` (`main.rs:709`), not an MBR-specific field. Note the MBR sidecar is
validated as a prototype but Stage D3 is a stub in the engine (config hooks only;
not in the `run` chain).

### DIA-NN recipe (offline, license-clean)

Run once by the user, who must hold their own DIA-NN license (MuMDIA ships no
DIA-NN). `import_diann_lib.py` filters to targets, b/y no-loss fragments, and
peptides carrying only Carbamidomethyl/Oxidation (the only mapped mods,
`import_diann_lib.py:28-37`), rewrites `(UniMod:4/35)` to ProForma bracket names,
sorts precursors by m/z with a stable mergesort and assigns contiguous
`candidate_id` (`:52-56`), derives `base_peptide_id` by factorizing
`Stripped.Sequence` (`:55`), and builds each fragment `name` as
`<Fragment.Type><Fragment.Series.Number>` with a `^<z>` suffix appended when
`Fragment.Charge > 1` (`:72-74`), preserving species-flagged protein names for the
ProteoBench metric. Then either decoy builder adds the null population:
`make_reverse_decoys.py` reverses each target keeping the C-terminal residue,
recomputes the real b/y m/z of the reversed sequence from a 20-residue monoisotopic
mass table plus a `UNIMOD` mod-mass dict (Carbamidomethyl, Oxidation, Acetyl,
Phospho, Deamidated, Methyl, Dimethyl, Carbamyl; an unknown bracket name falls
back to parsing a numeric `+mass` string, `make_reverse_decoys.py:26-50`),
validated against the library's own target m/z to < 5 ppm at the 99th percentile
over the first 500 target precursors (`make_reverse_decoys.py:97-107`), and
enforces a hard no-overlap
invariant: any reversed stripped sequence colliding with a real target (palindrome
or reverse-equals-another-target) or with a decoy sequence already owned by a
different target base sequence is re-scrambled by a per-peptide-seeded
Fisher-Yates, dropped after `MAX_TRIES=30`, and a final assertion requires
`decoy_stripped ∩ target_stripped == {}` (`:115-132, 166-169`). `make_shift_decoys.py` is
the alternative: copy intensities+iRT, keep precursor m/z, and shift each fragment
in **m/z space** by `-DELTA/z` (b ions) or `+DELTA/z` (y ions) where
`DELTA = 14.015650 Da` (one CH2) and `z = frag_charge`, net precursor shift zero
(`make_shift_decoys.py:17, 41-44`). Its decoy `peptidoform` and `protein` are the
target strings prefixed with `DECOY_` (`make_shift_decoys.py:35-36`); the reverse
builder instead sets `peptidoform = "DECOY_" + <reversed-sequence ProForma>` and
`protein = "DECOY_" + <target protein>` (`make_reverse_decoys.py:145-146`). Both
concatenate
target+decoy, re-sort by `precursor_mz`, reassign contiguous `candidate_id`, and
re-sort fragments by `candidate_id` so the fragment index's contiguous-range
precondition holds (`make_reverse_decoys.py:156-161`, `make_shift_decoys.py:47-55`).

## Key types and functions

| name | file:line | what it does |
|---|---|---|
| `resolve_script` | `sidecar.rs:20` | Resolve a worker path: CWD-relative dir, then `<exe_dir>/<dir>`, then `<exe_dir>/scripts`, else CWD-relative fallback |
| `run_worker` | `sidecar.rs:217` | Spawn `python <script> <argv...>`; `utf8=true` sets `PYTHONUTF8`/`PYTHONIOENCODING` (DeepLC/Keras crash on Windows cp1252) |
| `run_ms2pip` | `sidecar.rs:42` | Write ms2pip_in.parquet, run worker, fold output to `HashMap<u32,HashMap<(u8,u16),f32>>` |
| `run_deeplc` | `sidecar.rs:81` | Write deeplc_in.parquet, run worker, return `HashMap<u32,f32>` (`utf8=true`) |
| `run_deeplc_finetune` | `sidecar.rs:111` | Run `deeplc_finetune.py <lib_in> <seed> <lib_out> --epochs --patience --q-train --batch` (`utf8=true`) |
| `run_mbr` | `sidecar.rs:162` | Run `mbr_worker.py <scored> <psms_csv> <out> [--out-scored] [--frag-csv --consensus-corr-min] --q-anchor --min-anchor-runs --q-transfer --seed` |
| `run_pin_sidecar` | `rescore.rs:740` | Write PIN keyed by row index, run mokapot/nn worker, map `score` back by row index |
| `run_entrapment_gbm` | `rescore.rs:675` | Write features+meta Parquet, run entrapment worker, map `score` back by `row_id` |
| `align_sidecar_scores` | `rescore.rs:811` | Validate + align a sidecar's `(row_id, score)`: exact, unique, finite coverage or bail |
| `make_model` | `mokapot_worker.py:35` | Build the mokapot model from `MUMDIA_RESCORE_MODEL` (nn/logreg/xgb/percolator) |
| `tda_q` | `nn_rescore_worker.py:77` | Target-decoy q: `(decoys+1)/max(1,targets)`, running min from the tail |
| `one_pass` | `nn_rescore_worker.py:225` | One CV pass -> OOF scores; per fold iterate positive re-selection + MLP retrain |
| `binned_map` | `mbr_worker.py:31` | Monotone binned-median RT calibration x->y (80 bins) |
| `expected_rt` | `mbr_worker.py:116` | Cross-run predicted RT of a candidate in a run from the other runs' anchors |
| `frag_mz` / `reverse_keep_cterm` / `stable_seed` / `splitmix` | `make_reverse_decoys.py:58/56/74/65` | Residue-mass b/y m/z; C-term-fixed reversal; process-independent FNV-1a seed; seeded PRNG for scramble |

## Configuration

Config was recently pruned; every field below exists in `mumdia-core/src/config.rs`
today. Predictor/rescorer selection is by strategy enum, and sidecars are engaged
only when a non-native enum is set AND the corresponding Python interpreter is
configured.

| field | default | effect |
|---|---|---|
| `predict_frag.predictor` | `native` | `ms2pip` engages `ms2pip_worker.py` (requires `ms2pip_python`) |
| `predict_frag.rt_predictor` | `native` | `deeplc` engages `deeplc_worker.py` (requires `deeplc_python`) |
| `predict_frag.ms2pip_model` | `"HCD"` | 3rd positional arg to `ms2pip_worker.py` |
| `predict_frag.ms2pip_python` | `None` | interpreter for MS2PIP (env with ms2pip+pyarrow) |
| `predict_frag.deeplc_python` | `None` | interpreter for DeepLC predict AND fine-tune |
| `predict_frag.sidecar_script_dir` | `"scripts"` | dir passed to `resolve_script` for all workers |
| `rt_im_train.finetune_deeplc` | `false` | engage `deeplc_finetune.py` in `run` (requires `deeplc_python`) |
| `rt_im_train.finetune_epochs` | `25` | `--epochs` upper bound (early stopping usually halts first) |
| `rt_im_train.finetune_patience` | `10` | `--patience` epochs without val-loss improvement |
| `rt_im_train.finetune_batch` | `0` | `--batch`; 0 = auto-scale to seed size in the worker |
| `rt_im_train.q_train` | `0.01` | `--q-train` max seed `spectrum_q` for the fine-tune reference |
| `rescore.classifier` | `native_tda` | `mokapot`/`nn_torch` -> PIN sidecar; `entrapment` -> Parquet sidecar; `percolator` -> unwired |
| `rescore.python` | `None` | interpreter for the rescore/entrapment sidecar |
| `rescore.folds` | `3` | passed as `MUMDIA_NN_FOLDS` + entrapment `folds` argv |
| `rescore.num_iter` | `10` | passed as `MUMDIA_NN_ITERS` (native semi-supervised iterations) |
| `rescore.train_fdr` | `0.01` | passed as `MUMDIA_NN_TRAIN_FDR` |
| `rescore.strict` | `true` | fail on any rescorer sidecar failure/misconfiguration; false explicitly enables compatibility fallback |
| `rescore.entrapment_marker` | `None` | protein substring marking spike-in negatives (required for `entrapment`) |
| `rescore.entrapment_exclude` | `None` | substring that un-marks the sample's own species |
| `rescore.entrapment_contaminant_markers` | `[]` | substrings for genuine contaminants (kept as real targets) |
| `rescore.entrapment_ratio` | `1.0` | `N_real_lib / N_entrap_lib` scaling of the entrapment FDR |
| `mbr.strategy` | `none` | `empirical_library`/`rt_transfer`/`full` engage `mbr_worker.py` |
| `mbr.python` | `None` | interpreter for MBR (required when `strategy != none`) |
| `mbr.q_anchor` / `min_anchor_runs` / `q_transfer` | `0.01` / `2` / `0.01` | `--q-anchor` / `--min-anchor-runs` / `--q-transfer` (anchor/transfer FDR + min supporting runs) |
| `mbr.consensus_corr_min` | `0.0` | `--consensus-corr-min` fragment-consensus guard threshold (0 = off; only passed when `frag` paths are also supplied, `sidecar.rs:209`) |
| `mbr.rt_window_s` | `20.0` | transfer half-window (>= p95 M2 residual ~17 s) for the `--emit-transfer-targets` re-extraction tier. **Not wired through `run_mbr`** (`sidecar.rs:193-211` passes neither `--rt-window` nor `--emit-transfer-targets`); the worker uses its own `--rt-window` default of 20.0 when run by hand. |
| `mbr.decoy_transfer` | `permuted_rt` | `DecoyTransfer` enum (`permuted_rt`/`reverse_sequence`/`both`, `config.rs:857-869`) selecting the false-transfer null. **Unwired**: `run_mbr` never passes it and the worker implements only the permuted-RT null (`mbr_worker.py:176-199`). |
| `mbr.requant_all` | `false` | requantify already-identified precursors, not only transfers; only meaningful for `strategy = full`. **Unwired** in the current `mbr_worker.py`. |

**Worker-only env knobs** (not config fields; set in the process environment).
`mokapot_worker.py`: `MUMDIA_RESCORE_MODEL` (default `nn`; accepts the aliases
`logreg`/`logistic`/`lr`, `xgb`/`xgboost`, and `percolator`/`linear`/`svm`,
`mokapot_worker.py:36-76`), `MUMDIA_BREW_ITERS` (20, the `Model.max_iter`
semi-supervised count), `MUMDIA_NN_HIDDEN` (`128,64,64,32`), `MUMDIA_NN_SOLVER`
(adam), `MUMDIA_NN_MAX_ITER` (200), `MUMDIA_NN_ALPHA` (1e-4); logreg reads
`MUMDIA_LR_C` (1.0) and `MUMDIA_LR_MAX_ITER` (1000); xgb reads `MUMDIA_XGB_TREES`
(200), `MUMDIA_XGB_DEPTH` (6), `MUMDIA_XGB_LR` (0.1), `MUMDIA_XGB_JOBS` (0 = all
cores); `MUMDIA_MOKAPOT_WORKERS` (3, thread-based CV-fold parallelism).
`nn_rescore_worker.py`: `MUMDIA_NN_EPOCHS` (25), `MUMDIA_NN_HIDDEN` (`128,64`),
`MUMDIA_NN_DROPOUT` (0.3), `MUMDIA_NN_LR` (1e-3), `MUMDIA_NN_WD` (1e-4),
`MUMDIA_NN_BATCH` (4096), `MUMDIA_NN_SEEDS` (1), `MUMDIA_NN_STREAM` (auto),
`MUMDIA_NN_STREAM_GB` (4), `MUMDIA_NN_CHUNK` (250000), `MUMDIA_NN_INIT_SAMPLE`
(300000) plus the three the Rust caller injects: `MUMDIA_NN_FOLDS` (worker default
3), `MUMDIA_NN_ITERS` (worker default 5, but `run_pin_sidecar` overrides it with
`rescore.num_iter` = 10), `MUMDIA_NN_TRAIN_FDR` (0.01). These worker defaults
apply only when the sidecar is run standalone. `entrapment_worker.py`:
`MUMDIA_ENTRAPMENT_MODEL` (`gbm`|`nn`). `deeplc_finetune.py`: `DEEPLC_FT_THREADS`
(8) plus argparse flags. Note the mokapot worker's **code default model is `nn`**
(the sklearn MLP inside mokapot), even though the recommended portable path
(`env/mumdia-rescore.yml`) sets `MUMDIA_RESCORE_MODEL=logreg`; the Rust caller
does not set `MUMDIA_RESCORE_MODEL`, so unless the environment sets it you get the
MLP. Set it explicitly for the logreg path.

## Invariants, determinism, gotchas

- **File contract only.** No sidecar reads stdin or emits data on stdout (only log
  lines). The output Parquet must key on `id` or `candidate_id`/`row_id` exactly
  as the caller expects, or the readback map silently assigns the worst score.
- **Row index vs candidate_id.** In multi-run rescoring `candidate_id` is the
  library index and repeats across runs, so the PIN keys on a unique flat row
  index (`SpecId=psm_<i>`, `ScanNr=<i>`, `rescore.rs:763-771`) and the entrapment
  Parquet carries a separate `row_id` (`rescore.rs:697`). Keying on
  `candidate_id` would collide and collapse runs.
- **resolve_script Windows-path gotcha.** The build target dir is redirected off
  the OneDrive tree (`C:/Users/robbi/mumdia_build/...`), while `scripts/` lives
  under the OneDrive project. So `<exe_dir>/scripts` does NOT exist next to the
  binary on this machine; `resolve_script` finds the workers only via the
  CWD-relative branch (`sidecar.rs:19-21`). Run from the project root, or set
  `predict_frag.sidecar_script_dir` (and it is reused for rescore/MBR) to an
  absolute path, e.g. `/opt/mumdia/scripts` in the Docker configs. If none of the
  three candidates exist it returns the CWD-relative path so the eventual spawn
  error names it (`sidecar.rs:32`).
- **UTF-8.** `run_deeplc`/`run_deeplc_finetune` pass `utf8=true`; the PIN and
  entrapment sidecars set `PYTHONUTF8=1` directly. MS2PIP and MBR do not
  (`sidecar.rs:63, 99, 152, 212`).
- **Determinism (`docs/18_findings_and_decisions.md`, determinism contract).** MS2PIP predictions are deterministic
  regardless of process count (`ms2pip_worker.py:40-41`). DeepLC predict is
  deterministic given fixed weights. **DeepLC fine-tune is nondeterministic**: no
  torch/numpy seed is set (CLAUDE.md and MEMORY both flag this), so the rewritten
  iRT and thus the whole downstream run vary. `mokapot_worker.py` pins
  `rng=0`/`random_state=0`/`np.random.seed(0)`, but with `solver=adam` + BLAS
  threading the NN scores drift slightly; logreg is near bit-exact.
  `nn_rescore_worker.py` seeds torch/numpy per seed and the fold split is a
  content hash, but training is only approximately reproducible (use
  `MUMDIA_NN_SEEDS>1` to average out variance). `entrapment_worker.py` GBM is
  reproducible (`early_stopping=False`, `random_state=0`); the NN variant is not.
  `mbr_worker.py` is deterministic given `--seed` (`np.random.default_rng(seed)`
  for the permuted-RT null).
- **make_reverse_decoys scramble is deterministic and process-independent.** The
  re-scramble PRNG for the collision/palindrome path is a SplitMix64 (`splitmix`,
  `make_reverse_decoys.py:65`) seeded from a process-independent FNV-1a hash of the
  stripped sequence (`stable_seed`, `make_reverse_decoys.py:74`, applied at
  `make_reverse_decoys.py:119`), not Python's randomized builtin `hash`. The
  scrambled decoys for those few peptides are therefore reproducible across
  library-build runs without setting `PYTHONHASHSEED`.
- **DIA-NN index precondition.** Both decoy builders and the importer re-sort by
  `precursor_mz` (stable mergesort) and reassign contiguous `candidate_id`, and
  sort fragments by `candidate_id`, so `index.rs load()`'s
  contiguous-id + m/z-sorted precondition holds. Do not reorder afterward.
  `make_reverse_decoys.py` additionally aborts if its residue-mass calculator
  disagrees with the library's own target fragment m/z by > 5 ppm at p99.
- **Fallback exists only when explicitly requested (rescore only).** With
  `rescore.strict = false` a crashed or misconfigured **rescore** sidecar is
  logged and the run continues on `native_tda`; the default `strict = true`
  makes the failure fatal. Verify the actual classifier in the scored artifact
  report. The predictor sidecars (MS2PIP, DeepLC, DeepLC
  fine-tune) and MBR have no such gate: any nonzero exit aborts the run (see
  **Failure behavior**).
- **MS2PIP charge coverage.** MS2PIP emits singly-charged b/y only; charge-2
  fragments keep the native heuristic intensity (`predict_frag.rs:357-363`).
- **Entrapment worker needs both classes.** `entrapment_worker.py` raises
  `SystemExit` if the training set (all non-decoy rows) is single-class, i.e. no
  real-target or no entrapment PSMs (`entrapment_worker.py:86-87`); the Rust side
  already guards this (`n_ent == 0` falls back to `native_tda`, `rescore.rs:218-234`).
  Its fold count is `k = max(2, min(folds, n_groups))` over the `base_peptide_id`
  groups (`:89-90`); folds whose training side is single-class are left NaN and
  filled by the final full-data model, which also scores every decoy
  (`:97-108`).
- **Sidecar device selection.** `nn_rescore_worker.py` uses CUDA automatically when
  `torch.cuda.is_available()`, else CPU (`nn_rescore_worker.py:112`); it is not a
  config knob. The DeepLC fine-tune stays on CPU unless `--device cuda` is passed
  (which `run` never does). `mokapot_worker.py` and `entrapment_worker.py` are
  scikit-learn/CPU only.
- **mokapot uses OOF scores only, with no fallback.** `_oof_scores`
  (`mokapot_worker.py:136-163`) merges mokapot's held-out target and decoy
  confidence tables and raises `RuntimeError` unless they cover the PIN rows
  exactly once (`mokapot_worker.py:152-159`) with finite scores
  (`mokapot_worker.py:161-162`). There is deliberately no in-sample or
  fold-averaging fallback: an incomplete confidence table is a hard worker error,
  which the Rust caller then treats per `rescore.strict` (bail if strict, else
  `native_tda`). Confirm the OOF branch ran on real runs (it prints "using
  complete out-of-fold confidence scores", `mokapot_worker.py:166`).

## How to extend / modify

- **Add a predictor/rescorer sidecar.** Add the enum variant in `config.rs`
  (`FragPredictorKind`/`RtPredictorKind`/`RescorerKind`), a `Config` field for its
  interpreter, a dispatch arm in `predict_frag.rs`/`rescore.rs`, and a thin client
  in `sidecar.rs` (or reuse `run_pin_sidecar` if it consumes the PIN and emits
  `candidate_id`+`score`). Keep the positional-CLI-plus-Parquet contract; do not
  add a JSON request file or a server.
- **Reuse the PIN contract.** A new rescorer that reads the PIN and writes
  `candidate_id`+`score`+`q_value` needs no new Rust plumbing beyond a
  `RescorerKind` arm calling `run_pin_sidecar` with its script name; the
  `MUMDIA_NN_*` env vars are already injected.
- **Change env probing.** `doctor` (`main.rs:313-374`) hard-codes the package list
  per interpreter and switches the rescore packages on the classifier: `nn_torch`
  -> `torch,numpy,pandas,pyarrow`; every other classifier (mokapot, entrapment,
  percolator, native) -> `mokapot,sklearn,numpy,pandas,pyarrow`
  (`main.rs:318-324`); `predict_frag.deeplc_python` -> `deeplc,numpy,pandas`;
  `predict_frag.ms2pip_python` -> `ms2pip,numpy,pandas` (`main.rs:327-336`). An
  interpreter left `None` prints `[skip]` (native path). It probes with
  `importlib.util.find_spec` and reports `MISSING <pkgs>`; note `mbr.python` is
  **not** probed by `doctor`. Update these lists when a worker's imports change so
  `mumdia doctor` stays truthful.
- **Conda envs.** The committed reproducible specs are `env/docker-rescore.yml`
  (env `rescore`: mokapot 0.10.0 + ms2pip 4.0.0.dev9, py3.11) and
  `env/docker-deeplc.yml` (env `deeplc`: DeepLC 4.0 multitask at a pinned commit +
  CPU torch, py3.11); the Docker configs point interpreters at
  `/opt/conda/envs/{rescore,deeplc}/bin/python` (`docker/config.dia.json`,
  `docker/config.diann-lib.json`). `env/mumdia-rescore.yml` is the minimal
  portable env for the mokapot logreg path (no torch/DeepLC/MS2PIP). On the
  developer machine the workers also import from the local `py312_mumdia` env
  (general env: torch + mokapot + ms2pip + sklearn + pyarrow), `ms2rescore`
  (mokapot + MS2PIP), and `deeplc_mt` for fine-tune (the a2 build; NOT
  `deeplc_multitask`/a1, whose predict crashes, nor `py310_deeplc`, whose ms2pip
  is broken). Anchor only the tool version and let pip resolve its scientific-Python
  graph, since exact old pins (pandas < 2) have no cp312 wheel.
- **nn_torch scaling.** For a many-run experiment-wide rescore, force the streaming
  backend with `MUMDIA_NN_STREAM=1` (or rely on the 4 GB auto threshold) so peak
  RAM is one minibatch, not the whole PIN; the memmap sidecar file is written next
  to the output and deleted on exit (`nn_rescore_worker.py:270-275`).
