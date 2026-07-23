# Python sidecars (the 10 scripts) + conda envs
> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

MuMDIA is a pure-Rust engine with deterministic native fallbacks for every ML
step, so it runs with zero Python. The sidecars are the opt-in path to real
predictors and rescorers that raise identification counts over the native
defaults. Each sidecar is a standalone Python worker invoked as a subprocess over
a **positional-CLI file contract** (plan.md Section 3.2): the Rust caller writes
an input file (Parquet or PIN), runs `python <worker> <argv...>`, and reads an
output Parquet keyed by `id` or `candidate_id`. There is no JSON request file, no
long-lived server, and no stdin/stdout data channel; the files on disk are the
entire interface. This means any sidecar can be swapped for a native Rust
implementation (or a different tool) without touching the callers, and a broken
sidecar can fall back to the native path (`rescore.strict = false`).

The 10 scripts split into four groups: **predictors** (`ms2pip_worker`,
`deeplc_worker`, `deeplc_finetune`) feed the run-independent library;
**rescorers** (`mokapot_worker`, `nn_rescore_worker`, `entrapment_worker`) score
the competed PSMs in Stage F; **MBR** (`mbr_worker`) transfers identifications
across runs (Stage D3); and the **DIA-NN recipe** (`import_diann_lib`,
`make_reverse_decoys`, `make_shift_decoys`) is an offline, one-time toolchain to
convert a user-produced DIA-NN library into MuMDIA's target+decoy library schema.

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

**ms2pip_worker** (`sidecar.rs:37` `run_ms2pip`)
- IN `ms2pip_in.parquet`: `id` u32, `peptidoform` str (ProForma), `charge` i32.
- OUT `ms2pip_out.parquet`: `id` u32, `ion_type` str (`"b"`/`"y"`), `ordinal` i32
  (1-based), `intensity` f32 (linear). Rust folds this into
  `HashMap<u32, HashMap<(u8 ion_byte, u16 ordinal), f32>>` (`sidecar.rs:66-73`).

**deeplc_worker** (`sidecar.rs:77` `run_deeplc`)
- IN `deeplc_in.parquet`: `id` u32, `peptidoform` str.
- OUT `deeplc_out.parquet`: `id` u32, `predicted_rt` f32. Rust returns
  `HashMap<u32, f32>` (`sidecar.rs:100`).

**deeplc_finetune** (`sidecar.rs:106` `run_deeplc_finetune`)
- IN `<lib_in>` = `fragment_library_precursors.parquet` (needs `peptidoform`,
  `predicted_irt`); `<seed>` = seed PSMs (`peptidoform`, `label`, `spectrum_q`,
  `observed_rt`).
- OUT `<lib_out>` = `fragment_library_precursors_ft.parquet`: the input table with
  the `predicted_irt` column replaced (`deeplc_finetune.py:156-159`). Same schema,
  values rewritten.

**mokapot_worker** / **nn_rescore_worker** (`rescore.rs:563` `run_pin_sidecar`)
- IN `rescore.pin`: Percolator tab-separated. Fixed columns
  `SpecId Label ScanNr ExpMass CalcMass <features...> Peptide Proteins`
  (`rescore.rs:585-601`). `SpecId = psm_<i>`, `ScanNr = <i>` where `i` is the
  unique flat row index (NOT `candidate_id`, which repeats across runs);
  `ExpMass=CalcMass=precursor_mz`; `Label` +1 target / -1 decoy.
- OUT `rescore_sidecar_out.parquet`: `candidate_id` u32 (echoes the SpecId tail =
  row index), `score` f64, `q_value` f64 (written as zeros; Rust computes q). Rust
  maps `score` back by row index (`rescore.rs:625-630`).

**entrapment_worker** (`rescore.rs:503` `run_entrapment_gbm`)
- IN `entrapment_in.parquet`: `row_id` u32 (unique flat index), `candidate_id`
  u32, `base_peptide_id` u32, `is_entrapment` i32 (0/1), `is_decoy` i32 (0/1),
  then one f64 column per feature (`rescore.rs:521-534`).
- OUT `entrapment_out.parquet`: `row_id` u32, `candidate_id` u32, `score` f64.
  Rust maps back by `row_id` (`rescore.rs:549-555`).

**mbr_worker** (`sidecar.rs:136` `run_mbr`)
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

**MS2PIP** (`predict_frag.rs:305-315`, worker `ms2pip_worker.py`). Selected by
`predict_frag.predictor = "ms2pip"`. `assign_intensities` collects one `id` per
`Raw` candidate, its peptidoform and charge, calls `run_ms2pip`, then for each
fragment looks up `(ion_byte, ordinal)` for **charge-1** fragments only; charge-2
fragments fall back to the native heuristic (`predict_frag.rs:320-338`). The
worker builds `psm_utils.PSMList` in 100k-row chunks, calls
`ms2pip.predict_batch(model, processes=min(8, cpu_count))`, and converts MS2PIP's
log2 intensities to linear via `2**x - 0.001` clipped at 0
(`ms2pip_worker.py:52-53`). Ordinals are emitted 1-based
(`ms2pip_worker.py:57`). The `__main__` guard makes the Windows `spawn` start
method safe for multiprocessing.

**DeepLC predict** (`predict_frag.rs:254-291`, worker `deeplc_worker.py`).
Selected by `predict_frag.rt_predictor = "deeplc"`. `assign_rt` deduplicates by
peptidoform (RT is charge-independent, `predict_frag.rs:262-271`), calls
`run_deeplc`, and writes `r.irt`. Peptidoforms with no returned iRT are anchored
at `0.0` with a warning (`predict_frag.rs:278-289`; this is the "unmatched
peptidoforms silently get iRT 0.0" foot-gun noted in CLAUDE.md). The worker calls
`deeplc.predict` in 200k chunks and, when the multitask model returns an
ensemble matrix `(N, n_models)`, averages across models (`deeplc_worker.py:34-35`).
Predictions are uncalibrated; rt-im-train's per-run LOESS/linear maps them onto
observed RT.

**DeepLC fine-tune** (`run.rs:187-208`, worker `deeplc_finetune.py`). Wired into
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
same iRT scale as targets (`deeplc_finetune.py:44-45, 135-156`). The long
docstring and the thread-cap block at the top exist to prevent an intermittent
machine crash: numpy's OpenBLAS (GNU OpenMP) and torch's Intel OpenMP coexist
under `KMP_DUPLICATE_LIB_OK=TRUE`, and without pinning `OMP/MKL/OPENBLAS` to 1
thread and bounding torch's pool, the two full thread pools oversubscribe the CPU
during the backward pass (`deeplc_finetune.py:6-28, 82-91`). `--device cuda`
sidesteps this entirely by moving compute off the CPU pools.

### Rescorers (Stage F)

**mokapot** and **nn_torch** share the exact PIN contract via `run_pin_sidecar`
(`rescore.rs:563-631`). Rust concatenates the competed feature tables, writes the
PIN, spawns the worker, and passes `MUMDIA_NN_FOLDS/ITERS/TRAIN_FDR` from
`rescore.{folds,num_iter,train_fdr}` as env vars (`rescore.rs:614-616`) so the
report's recorded params match what ran; mokapot ignores those three. On success
the classifier label and `model_identity` are recorded; on failure the path falls
back to `native_scores` unless `rescore.strict = true` (`rescore.rs:105-134`).

- `mokapot_worker.py` reads the PIN with `mokapot.read_pin`, builds a model
  chosen by `MUMDIA_RESCORE_MODEL` (`make_model`, `mokapot_worker.py:34-97`:
  `nn` -> sklearn `MLPClassifier`; `logreg` -> `LogisticRegression`; `xgb` ->
  `XGBClassifier`; `percolator`/`linear`/`svm` -> mokapot's default, `model=None`),
  runs `mokapot.brew(..., rng=0, max_workers=MUMDIA_MOKAPOT_WORKERS)`, and prefers
  mokapot's **out-of-fold** confidence scores (each PSM scored by the fold that
  did not train on it, `mokapot_worker.py:135-156`). It falls back to averaging
  all fold models over all rows only if the confidence API differs; that fallback
  is not truly OOF and inflates apparent sensitivity, so it is a last resort.
  Every PSM (targets and decoys) is scored, `SpecId` tail parsed to
  `candidate_id` (`mokapot_worker.py:167`).
- `nn_rescore_worker.py` implements the semi-supervised scheme itself in PyTorch.
  Fold assignment is `md5(stripped_peptide) % FOLDS` (`nn_rescore_worker.py:123`),
  so peptides never leak across folds and the split is deterministic. Per fold it
  initialises from the single best feature+sign (by targets at `TRAIN_FDR` on a
  sample, `nn_rescore_worker.py:175-188`), then iterates {recompute target-decoy q
  on the training folds -> targets at `q<=TRAIN_FDR` positive, all decoys negative
  -> train MLP from scratch -> rescore} for `ITERS` rounds, then scores the
  held-out fold (`one_pass`, `nn_rescore_worker.py:233-253`). Two feature
  backends behind one accessor `get`: in-memory (median/IQR standardisation,
  `:127-141`) or a disk-backed float32 **memmap** streamed in `MUMDIA_NN_CHUNK`
  chunks with mean/std accumulated in one text pass (`:143-173`), auto-selected
  when the PIN exceeds `MUMDIA_NN_STREAM_GB` (4 GB). This is what makes an
  experiment-wide multi-run rescore tractable: the full PIN never lives in RAM.
  `tda_q` (`:77-87`) is the shared q formula `(decoys+1)/max(1,targets)`, running
  min from the tail. Seeds are ensembled by averaging rank-normalised OOF scores
  (`:255-262`).

**entrapment** (`rescore.rs:503-556`, worker `entrapment_worker.py`). Selected by
`rescore.classifier = "entrapment"` with `rescore.entrapment_marker` set and
`rescore.python` present; otherwise it falls back to a native linear entrapment
rescorer or `native_tda` (`rescore.rs:143-191`). `classify_entrapment`
(`rescore.rs:411-438`) marks a target as entrapment when its protein contains the
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
flexible model helps (AUC ~0.97 vs ~0.62 on in-silico decoys), unlike the linear
default which is all decoys can support.

### MBR (Stage D3)

`mbr_worker.py` (`main.rs:637-657` `Cmd::Mbr`, `sidecar.rs:136` `run_mbr`).
Requires `mbr.strategy != none`, `>= 2` runs, and `mbr.python`
(`main.rs:639-649`). Reads the experiment-wide scored table and per-run apex RTs.
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
re-extraction tier). Note the MBR sidecar is validated as a prototype but Stage
D3 is a stub in the engine (config hooks only; not in the `run` chain).

### DIA-NN recipe (offline, license-clean)

Run once by the user, who must hold their own DIA-NN license (MuMDIA ships no
DIA-NN). `import_diann_lib.py` filters to targets, b/y no-loss fragments, and
peptides carrying only Carbamidomethyl/Oxidation (the only mapped mods,
`import_diann_lib.py:28-37`), rewrites `(UniMod:4/35)` to ProForma bracket names,
sorts precursors by m/z with a stable mergesort and assigns contiguous
`candidate_id` (`:52-56`), and preserves species-flagged protein names for the
ProteoBench metric. Then either decoy builder adds the null population:
`make_reverse_decoys.py` reverses each target keeping the C-terminal residue,
recomputes the real b/y m/z of the reversed sequence from a residue-mass table
(validated against the library's own target m/z to < 5 ppm at the 99th
percentile, `make_reverse_decoys.py:88-98`), and enforces a hard no-overlap
invariant: any reversed stripped sequence colliding with a real target (palindrome
or reverse-equals-another-target) is re-scrambled by a per-peptide-seeded
Fisher-Yates, dropped after `MAX_TRIES=30`, and a final assertion requires
`decoy_stripped ∩ target_stripped == {}` (`:100-145`). `make_shift_decoys.py` is
the alternative: copy intensities+iRT, keep precursor m/z, shift b-ion m/z by
`-DELTA` and y-ion by `+DELTA` where `DELTA = 14.01565 Da` (one CH2), net
precursor shift zero (`make_shift_decoys.py:17, 38-44`). Both concatenate
target+decoy, re-sort by `precursor_mz`, reassign contiguous `candidate_id`, and
re-sort fragments by `candidate_id` so the fragment index's contiguous-range
precondition holds (`make_reverse_decoys.py:132-140`, `make_shift_decoys.py:47-55`).

## Key types and functions

| name | file:line | what it does |
|---|---|---|
| `resolve_script` | `sidecar.rs:18` | Resolve a worker path: CWD-relative dir, then `<exe_dir>/<dir>`, then `<exe_dir>/scripts`, else CWD-relative fallback |
| `run_worker` | `sidecar.rs:174` | Spawn `python <script> <argv...>`; `utf8=true` sets `PYTHONUTF8`/`PYTHONIOENCODING` (DeepLC/Keras crash on Windows cp1252) |
| `run_ms2pip` | `sidecar.rs:37` | Write ms2pip_in.parquet, run worker, fold output to `HashMap<u32,HashMap<(u8,u16),f32>>` |
| `run_deeplc` | `sidecar.rs:77` | Write deeplc_in.parquet, run worker, return `HashMap<u32,f32>` (`utf8=true`) |
| `run_deeplc_finetune` | `sidecar.rs:106` | Run `deeplc_finetune.py <lib_in> <seed> <lib_out> --epochs --patience --q-train --batch` (`utf8=true`) |
| `run_mbr` | `sidecar.rs:136` | Run `mbr_worker.py <scored> <psms_csv> <out> [--out-scored] [--frag-csv --consensus-corr-min] --q-anchor --min-anchor-runs --q-transfer --seed` |
| `run_pin_sidecar` | `rescore.rs:563` | Write PIN keyed by row index, run mokapot/nn worker, map `score` back by row index |
| `run_entrapment_gbm` | `rescore.rs:503` | Write features+meta Parquet, run entrapment worker, map `score` back by `row_id` |
| `make_model` | `mokapot_worker.py:34` | Build the mokapot model from `MUMDIA_RESCORE_MODEL` (nn/logreg/xgb/percolator) |
| `tda_q` | `nn_rescore_worker.py:77` | Target-decoy q: `(decoys+1)/max(1,targets)`, running min from the tail |
| `one_pass` | `nn_rescore_worker.py:233` | One CV pass -> OOF scores; per fold iterate positive re-selection + MLP retrain |
| `binned_map` | `mbr_worker.py:31` | Monotone binned-median RT calibration x->y (80 bins) |
| `expected_rt` | `mbr_worker.py:116` | Cross-run predicted RT of a candidate in a run from the other runs' anchors |
| `frag_mz` / `reverse_keep_cterm` / `splitmix` | `make_reverse_decoys.py:57/55/64` | Residue-mass b/y m/z; C-term-fixed reversal; seeded PRNG for scramble |

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
| `rescore.strict` | `false` | true makes any sidecar failure/misconfig a hard error (no native fallback) |
| `rescore.entrapment_marker` | `None` | protein substring marking spike-in negatives (required for `entrapment`) |
| `rescore.entrapment_exclude` | `None` | substring that un-marks the sample's own species |
| `rescore.entrapment_contaminant_markers` | `[]` | substrings for genuine contaminants (kept as real targets) |
| `rescore.entrapment_ratio` | `1.0` | `N_real_lib / N_entrap_lib` scaling of the entrapment FDR |
| `mbr.strategy` | `none` | `empirical_library`/`rt_transfer`/`full` engage `mbr_worker.py` |
| `mbr.python` | `None` | interpreter for MBR (required when `strategy != none`) |
| `mbr.q_anchor` / `min_anchor_runs` / `q_transfer` | `0.01` / `2` / `0.01` | anchor/transfer FDR + min supporting runs |
| `mbr.rt_window_s` | `20.0` | transfer half-window (>= p95 M2 residual ~17 s) for `--emit-transfer-targets` |
| `mbr.consensus_corr_min` | `0.0` | fragment-consensus guard threshold (0 = off) |

**Worker-only env knobs** (not config fields; set in the process environment).
`mokapot_worker.py`: `MUMDIA_RESCORE_MODEL` (default `nn`), `MUMDIA_BREW_ITERS`
(20), `MUMDIA_NN_HIDDEN` (`128,64,64,32`), `MUMDIA_NN_SOLVER` (adam),
`MUMDIA_NN_MAX_ITER` (200), `MUMDIA_NN_ALPHA` (1e-4), `MUMDIA_LR_C`,
`MUMDIA_LR_MAX_ITER`, `MUMDIA_XGB_*`, `MUMDIA_MOKAPOT_WORKERS` (3).
`nn_rescore_worker.py`: `MUMDIA_NN_EPOCHS` (25), `MUMDIA_NN_HIDDEN` (`128,64`),
`MUMDIA_NN_DROPOUT` (0.3), `MUMDIA_NN_LR` (1e-3), `MUMDIA_NN_WD` (1e-4),
`MUMDIA_NN_BATCH` (4096), `MUMDIA_NN_SEEDS` (1), `MUMDIA_NN_STREAM` (auto),
`MUMDIA_NN_STREAM_GB` (4), `MUMDIA_NN_CHUNK` (250000), `MUMDIA_NN_INIT_SAMPLE`
(300000) plus the three the Rust caller injects. `entrapment_worker.py`:
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
  index (`SpecId=psm_<i>`, `ScanNr=<i>`, `rescore.rs:588-596`) and the entrapment
  Parquet carries a separate `row_id` (`rescore.rs:521-525`). Keying on
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
  (`sidecar.rs:58, 95, 122, 169`).
- **Determinism (plan.md Section 7).** MS2PIP predictions are deterministic
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
- **make_reverse_decoys scramble seed is process-salted.** The re-scramble PRNG is
  seeded from Python's builtin `hash(stripped(t))` (`make_reverse_decoys.py:104`),
  which for strings is randomized per process unless `PYTHONHASHSEED` is fixed.
  Only the collision/palindrome path uses it, but that means the scrambled decoys
  for those few peptides differ between library-build runs. Set `PYTHONHASHSEED=0`
  to make the built decoy library reproducible.
- **DIA-NN index precondition.** Both decoy builders and the importer re-sort by
  `precursor_mz` (stable mergesort) and reassign contiguous `candidate_id`, and
  sort fragments by `candidate_id`, so `index.rs load()`'s
  contiguous-id + m/z-sorted precondition holds. Do not reorder afterward.
  `make_reverse_decoys.py` additionally aborts if its residue-mass calculator
  disagrees with the library's own target fragment m/z by > 5 ppm at p99.
- **Fallback masks failure unless strict.** With `rescore.strict = false` a
  crashed or misconfigured sidecar is logged and the run silently continues on
  `native_tda` (`rescore.rs:112-118, 127-133`). Production configs should set
  `strict = true` so a broken rescorer never passes as native scores.
- **MS2PIP charge coverage.** MS2PIP emits singly-charged b/y only; charge-2
  fragments keep the native heuristic intensity (`predict_frag.rs:324-337`).
- **mokapot OOF fallback is not OOF.** If mokapot's confidence API differs, the
  worker averages all fold models over all rows, which trained on ~2/3 of each row
  and inflates apparent sensitivity to ~2.5% external FDR at a nominal 1% cut
  (`mokapot_worker.py:129-134`). Confirm the OOF branch is taken on real runs
  (it prints "using out-of-fold confidence scores").

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
- **Change env probing.** `doctor` (`main.rs:311-361`) hard-codes the package list
  per interpreter and switches the rescore packages on the classifier
  (`torch...` for `nn_torch`, else `mokapot,sklearn,...`). Update both lists when a
  worker's imports change so `mumdia doctor` stays truthful.
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
