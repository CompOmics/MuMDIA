# Python sidecars (12 scripts + one shared writer) + conda envs
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

The 12 scripts split into four groups: **predictors** (`ms2pip_worker`,
`peptdeep_worker`, `deeplc_worker`, `deeplc_finetune`) feed the run-independent
library;
**rescorers** (`mokapot_worker`, `nn_rescore_worker`, `entrapment_worker`) score
the competed PSMs in Stage F; **MBR** (`mbr_worker`) transfers identifications
across runs (Stage D3); and the **DIA-NN recipe** (`import_diann_lib`,
`make_reverse_decoys`, `make_shift_decoys`, `augment_library`) is an offline,
one-time toolchain that builds MuMDIA's target+decoy library schema. Three of the
four convert a user-produced DIA-NN library into that schema; `augment_library`
is the fourth, filling an imported library with the tryptic FASTA peptides it is
missing (a completeness fix) before decoy generation.

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
| ms2pip_worker | `<in.parquet> <out.parquet> <model> <processes>` (`sidecar.rs`, `run_ms2pip`; `processes` is the engine's thread count) | `ms2pip_out.parquet` | `id` |
| deeplc_worker | `<in.parquet> <out.parquet> [threads]` (`sidecar.rs`, `run_deeplc`; `threads` is the engine's thread count, capped by the worker, see **DeepLC thread cap**) | `deeplc_out.parquet` | `id` |
| deeplc_finetune | `<lib_in> <seed> <lib_out> --epochs --patience --q-train --batch --window-holdout-frac --seed --predict-threads` (`sidecar.rs`, `run_deeplc_finetune`) | `<lib_out>` (= `fragment_library_precursors_ft.parquet`) | `peptidoform` (new table with replaced `predicted_irt`; input unchanged) |
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
| `scripts/peptdeep_worker.py` | Predictor: AlphaPeptDeep b/y fragment intensities per peptidoform+charge, conditioned on collision energy and instrument |
| `scripts/deeplc_worker.py` | Predictor: DeepLC iRT per peptidoform (uncalibrated) |
| `scripts/deeplc_finetune.py` | Predictor: transfer-learn DeepLC on this run's seed and rewrite the library iRT; with `--no-finetune` (seed `-`) rewrite it with base-model predictions instead (`rt_im_train.library_irt`) |
| `scripts/mokapot_worker.py` | Rescorer: mokapot brew over a PIN (model env-switchable: nn/logreg/xgb/percolator) |
| `scripts/nn_rescore_worker.py` | Rescorer: PyTorch semi-supervised MLP over a PIN, in-memory or streaming memmap |
| `scripts/entrapment_worker.py` | Rescorer: GBM/NN on real-target-vs-spike-in negatives, out-of-fold by base peptide |
| `scripts/mbr_worker.py` | MBR (Stage D3): cross-run RT transfer + permuted-RT decoy-transfer FDR |
| `scripts/import_diann_lib.py` | Recipe: DIA-NN fragment-level parquet -> MuMDIA target `lib_precursors`+`lib_fragments`; streams the input row group by row group (two passes), so peak memory is the precursor table, not the fragment table |
| `scripts/make_reverse_decoys.py` | Recipe: reverse-sequence decoys with no-target-overlap invariant |
| `scripts/make_shift_decoys.py` | Recipe: fragment-shift (CH2) decoys, DIA-NN-style terminal shift |
| `scripts/augment_library.py` | Recipe: augment an imported library with its missing tryptic FASTA peptides, then hand off to a decoy builder |
| `scripts/sort_fragments.py` | Recipe: rewrite a fragment table in `candidate_id` order (streaming bucket sort, in place); the writers above do this themselves since the range load, this is for tables written before |
| `scripts/mz_range_survivors.py` | Recipe: candidate ids inside the run's isolation range, as a prescan-style survivors table (docs/32) |
| `scripts/assemble_survivors.py` | Recipe: survivors table -> renumbered library, target/decoy pair kept together on `peptidoform_id`, fragments streamed (docs/21, docs/32) |
| `scripts/shard_parquet.py` | Recipe: row-group-aligned parquet split / concatenate |
| `scripts/mh_shard_predict.py` | Recipe: deduplicated, sharded multi-head DeepLC calibration of a very large precursor table (`uniq` / `predict` / `merge`; reuses `deeplc_finetune.py`) |
| `rust/mumdia/crates/mumdia/src/sidecar.rs` | Rust clients: `resolve_script`, `run_ms2pip`, `run_peptdeep`, `run_deeplc`, `run_deeplc_finetune`, `run_mbr`, `run_worker`, and the shared `fragment_request` / `read_fragment_intensities` the two intensity predictors both use |
| `rust/mumdia/crates/mumdia/src/stages/predict_frag.rs` | Call sites for MS2PIP + DeepLC (Stage C) |
| `rust/mumdia/crates/mumdia/src/stages/run.rs` | Call site for DeepLC fine-tune (between search-seed and rt-im-train) |
| `rust/mumdia/crates/mumdia/src/stages/rescore.rs` | Call sites for mokapot/nn_torch (PIN) + entrapment (Parquet) sidecars |
| `rust/mumdia/crates/mumdia/src/main.rs` | Call site for MBR (`Cmd::Mbr`) + `doctor` env probe |
| `env/docker-rescore.yml` | Docker env `rescore`: mokapot 0.10.0 + ms2pip 4.0.0.dev9 (py3.11) |
| `env/docker-deeplc.yml` | Docker env `deeplc`: `deeplc==4.4.0` (PyPI; the engine's floor) + CPU torch (py3.11) |
| `env/mumdia-rescore.yml` | Minimal portable env for the mokapot logreg rescore path (py3.12) |
| `env/mumdia-deeplc.yml` | Portable local env for the DeepLC sidecars: DeepLC 4.4.0 + CPU torch (py3.11) |
| `env/mumdia-peptdeep.yml` | Portable local env for the AlphaPeptDeep sidecar: peptdeep 1.5.1 (py3.11), separate because it pins the alphabase stack |

## Inputs and outputs

Each sidecar's on-disk contract, with the exact column schema read/written by the
code.

**ms2pip_worker** (`sidecar.rs:42` `run_ms2pip`)
- IN `ms2pip_in.parquet`: `id` u32, `peptidoform` str (ProForma), `charge` i32.
- OUT `ms2pip_out.parquet`: `id` u32, `ion_type` str (`"b"`/`"y"`), `ordinal` i32, `frag_charge` i32 (1, or 2 for the `b2`/`y2` series of the `*ch2` models; an output without the column is read as charge 1)
  (1-based), `intensity` f32 (linear). Rust folds this into
  `HashMap<u32, HashMap<(u8 ion_byte, u16 ordinal), f32>>` (`sidecar.rs:70-76`).

**peptdeep_worker** (`sidecar.rs` `run_peptdeep`)
- IN `peptdeep_in.parquet`: `id` u32, `peptidoform` str (ProForma), `charge` i32.
  Byte-for-byte the same request `ms2pip_worker` reads; both are written by
  `fragment_request`.
- OUT `peptdeep_out.parquet`: the same five columns `ms2pip_worker` writes, read back
  by the same `read_fragment_intensities`. The two intensity predictors are therefore
  interchangeable at this boundary, which is the point: the engine enumerates its own
  fragments and asks only for an intensity per `(ion_type, ordinal, frag_charge)`.
- ARGV `<in> <out> <model> <nce> <instrument> <processes>`; the device comes from
  `MUMDIA_PEPTDEEP_DEVICE` (auto|cuda|cpu), matching `MUMDIA_NN_DEVICE` in the
  rescorer rather than inventing a second convention.

**deeplc_worker** (`sidecar.rs` `run_deeplc`)
- ARGV `<in> <out> [threads]`; `threads` sizes torch's CPU pool under the
  **DeepLC thread cap**. A worker run without it keeps torch's own default, under
  the same cap.
- IN `deeplc_in.parquet`: `id` u32, `peptidoform` str.
- OUT `deeplc_out.parquet`: `id` u32, `predicted_rt` f32. Rust returns
  `HashMap<u32, f32>` (`sidecar.rs:104`).

**deeplc_finetune** (`sidecar.rs:111` `run_deeplc_finetune`)
- IN `<lib_in>` = `fragment_library_precursors.parquet` (needs `peptidoform`,
  `predicted_irt`); `<seed>` = seed PSMs (`peptidoform`, `label`, `spectrum_q`,
  `observed_rt`, and `base_peptide_id` under `--window-holdout-frac`; only these
  columns are read).
- OUT `<lib_out>` = `fragment_library_precursors_ft.parquet`: the input table with
  the `predicted_irt` column replaced (`rewrite_irt`). Same schema, values rewritten.
  Beside it `<lib_out>.summary.json` counts where each row's value came from
  (`rows`, `repredicted`, `retained_imported` and its two parts), the number of unique
  sequences predicted (`unique_predicted`), the torch threads used (`torch_threads`)
  and the wall time per phase in seconds (`timings_s`: `read_library`, `model_load`,
  `reference`, `fit`, `unique`, `predict`, `featurisation`, `forward`, `rewrite`,
  `write`). `featurisation` (PSM parsing, dataset construction, length bucketing and
  batch encoding) and `forward` (the model's forward calls) are measured inside
  `predict` by wrapping DeepLC's own steps (`PredictTimers`); the rest of `predict` is
  the calibration transform and copies. A phase that did not run, or that DeepLC's
  internals did not let the worker measure, is `null`; under a fine-tune the model
  load is part of `fit`.

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
  `--out-scored` writes the scored table with accepted transfers' PSM q columns
  lowered to `transfer_q`, an `is_transferred` flag and a `transfer_q` column (NaN
  on non-transferred rows) added; with no transfer candidates the transfer table has
  its ten columns and zero rows and the augmented table is the input, unflagged. Optional
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

**augment_library** (offline; no Rust caller, but shells out to the `mumdia`
binary and to a decoy builder)
- IN `--fasta` (protein FASTA), `--imported-precursors` / `--imported-fragments`
  (the imported target+decoy library in the `import_diann_lib` schema), plus
  `--mumdia-bin` and `--work-dir` (all required). The imported library supplies
  the set of TARGET base sequences already present (rows with `label == "target"`,
  `augment_library.py:100-114`); the FASTA plus `mumdia digest`/`peptidoforms`
  supplies the candidate set to complete against.
- OUT `--out-precursors` / `--out-fragments`: a full target+decoy library in the
  same `import_diann_lib` schema, containing the imported targets plus the missing
  tryptic peptides plus paired decoys. No schema/artifact-version bump: the output
  is the same fragment_library precursor/fragment column layout, only re-indexed.
- FLAGS: `--config` (default None), `--match-level {base_sequence(default),
  peptidoform_charge}`, `--decoy-strategy {shift(default),reverse}`
  (`augment_library.py:67-91`).

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
predictions")`), while a single candidate that MS2PIP returned nothing for
(missing/empty per-id entry) is dropped from the library together with its pair
(`drop_unpredicted`; docs/29 #17) instead of falling back to the native intensities
under an MS2PIP model identity. `run_ms2pip` rejects an id it did not request and a
fragment reported twice. The worker builds `psm_utils.PSMList` in chunks of
`max(100k, 5k x processes)` rows,
calls `ms2pip.predict_batch(model, processes=N)` with `N` the engine's thread count
passed as the fourth argument (the old cap `min(8, cpu_count)` applies only when the
argument is absent), and converts MS2PIP's log2 intensities to linear via
`2**x - 0.001` clipped at 0, in float64, stored as float32
(`ms2pip_worker.py`, `fragment_rows`). Ordinals are emitted 1-based; rows come out
per result, ions b then y. The output is assembled from numpy arrays per chunk, not
from per-fragment Python list appends, which at 9.8M peptidoforms were about 300M
appends and 13 GB of Python objects. The `__main__` guard makes the Windows `spawn`
start method safe for multiprocessing.

**AlphaPeptDeep** (`predict_frag.rs`, worker `peptdeep_worker.py`). Selected by
`predict_frag.predictor = "peptdeep"`. The call site is the MS2PIP arm with a
different worker behind it, down to reusing `ms2pip_values` for the lookup and
normalisation, and the empty-map and per-candidate-miss behaviour is identical
(`bail!` on an empty map, `drop_unpredicted` for a candidate the model did not
cover).

Three things are specific to it:

- **Both fragment charges are always predicted.** `FRAG_TYPES` in the worker is fixed
  at `b_z1, b_z2, y_z1, y_z2` and zero intensities are written out rather than
  dropped. The engine decides a model predicts the doubly charged series by finding
  ANY charge-2 key in that candidate's map; a precursor whose charge-2 predictions
  happened to be all zero would otherwise be scored as though the model were
  charge-1-only, fall back to native heuristic values there and max-normalise the two
  charge groups separately. That is the `HCD2021` failure mode, which measured 0
  confident PSMs at 1% on a real run. Measured on a 40-protein E. coli library,
  AlphaPeptDeep keeps 59,942 charge-2 fragments in the top 12 against the native
  heuristic's 9,892.
- **Collision energy and instrument change the prediction**, and are recorded in the
  artifact's `model_identity` for that reason:
  `peptdeep-1.5.1-generic-nce30-Lumos`. Two libraries built at different NCE are not
  the same library. The worker refuses an instrument outside the installed
  AlphaPeptDeep's own `instrument_group` vocabulary, because AlphaPeptDeep maps an
  unknown one onto its default silently.
- **ProForma is translated in the worker.** MuMDIA writes modifications as UniMod
  names (`PEC[Carbamidomethyl]TIDE`), which is also alphabase's `Name@Residue`
  convention, so `parse_peptidoform` maps them mechanically with 1-based sites, 0 for
  the N terminus and -1 for the C terminus. A bare mass delta (`[+79.96633]`) is
  refused rather than matched by mass, because the nearest UniMod entry is not
  necessarily the right one; those candidates are left unpredicted and the engine
  drops them with their pairs, failing above 2%.

Ordinals follow AlphaPeptDeep's fragment dataframe: within a precursor's
`[frag_start_idx, frag_stop_idx)` slice, row `i` holds b(i+1) and y(nAA-1-i). Verified
against its own `fragment_mz_df` (for a 9-mer, row 0 is b1 = 72.0444 and y8; row 7 is
b8 and y1 = 147.1128). Output is streamed through a `ParquetWriter` in 200k-precursor
chunks rather than concatenated at the end.

What it does not do: retention time or ion mobility (AlphaPeptDeep can; MuMDIA's RT is
DeepLC and its calibration is built around that), and a/c/x/z or neutral-loss ions
(`mumdia_core::mass::IonType` is `B | Y`, so the engine never generates those
fragments and an intensity for one has nowhere to go).

**DeepLC predict** (`predict_frag.rs:274-312`, worker `deeplc_worker.py`).
Selected by `predict_frag.rt_predictor = "deeplc"`. `assign_rt` deduplicates by
peptidoform (RT is charge-independent, `predict_frag.rs:281-290`), calls
`run_deeplc`, and writes `r.irt`. Peptidoforms with no returned iRT are dropped
from the library with their pairs and counted in the library report (they used to
be anchored at `0.0` with a warning, the "unmatched peptidoforms silently get iRT
0.0" foot-gun; docs/29 #17). `run_deeplc` rejects an id it did not request and a
duplicate id. The worker calls
`deeplc.predict` in 200k chunks and, when the multitask model returns an
ensemble matrix `(N, n_models)`, averages across models (`deeplc_worker.py:44-47`).
Predictions are uncalibrated; rt-im-train's per-run LOESS/linear maps them onto
observed RT. Its module-level imports are order-dependent: `import deeplc` must
precede numpy and pyarrow (`deeplc_worker.py:13-29`, see **DeepLC import order**
under gotchas).

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
same iRT scale as targets (`base_pf`, and `library_bases` for the whole column; both
strip only a leading `DECOY_`). A peptidoform that is non-standard (`is_std` false,
e.g. a terminal mod outside `STD`), was not predicted, or came back non-finite keeps
its **original** `predicted_irt` unchanged (`rewrite_irt`), so only the sequences
DeepLC actually re-predicted move onto the fine-tuned scale. The unique set, the
seed filter and the write-back run in Arrow and the base model is loaded once
(`load_base_model`), which measured byte-identical to the per-row Python loop they
replace; `tests/python/test_deeplc_predict.py` pins the equivalence without DeepLC.
Beyond the seven flags Rust passes for a fine-tune (`--epochs/--patience/--q-train/--batch/
--window-holdout-frac/--seed/--predict-threads`, the last one the engine's rayon thread
count for the forward-only library prediction while training keeps its bounded pool), and
the three it passes for a base-model re-prediction (`--no-finetune --threads N
--predict-threads N` with `-` as the seed path, `sidecar::run_deeplc_repredict`, N = the
engine's rayon thread count because prediction is forward-only), the worker exposes
CLI-only knobs that `run` never sets:
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

**DeepLC thread cap.** Every DeepLC call site asks for the engine's rayon thread count
(the fine-tune's training pool keeps its own bound), and both workers cap what they
give torch at the physical cores available to the process: the unique
`(physical_package_id, core_id)` pairs from sysfs over `sched_getaffinity(0)` on Linux,
so a container or `taskset` mask is respected, and every physical core
(`GetLogicalProcessorInformationEx`) on Windows. Elsewhere the count is unknown and
nothing is capped. The cap is a ceiling, never a target: a request at or below it is
taken as given, so the default path changes only on a host where the engine's thread
count exceeds the physical cores, which is the default on any SMT machine run without
`--threads`. `deeplc_worker.py` took torch's own default before (the physical cores on
an MKL build), so it now also follows an explicit `--threads` below that. Measured on
doxy (64 cores, 128 CPUs), the multi-head step took 10:41 at 96 threads and 18:09 at
128, because every OpenMP-parallel op waits for its slowest thread.
`MUMDIA_DEEPLC_THREAD_CAP=N` sets the cap explicitly, and `0` disables it. The
fine-tune worker prints the resolved numbers and records them under `torch_threads` in
`<lib_out>.summary.json` (requested and used training and prediction threads, the cap
and where it came from); `deeplc_worker.py` prints them.

Where the cap binds, the numbers change the way any change of `--threads` changes them.
Measured on the CPU (DeepLC 4.5.0, one i9-13900KS with 24 physical cores): torch's
kernels round differently at different thread counts, so 10 to 89 of 3,002 base-model
predictions differed in the last bits between 8 threads and each of 1, 2, 4, 16 and 24
(at most 3.1e-5), and 1,701 at 32 threads, past the physical cores (at most 1.4e-4). The
multi-head calibration amplifies this for a few sequences: DeepLC's per-head spline
hands over to a linear trail outside the reference's range, so a sequence at that edge
can move by tens of seconds. On a 12,002-row synthetic library, 32 threads against the
capped 24 moved the median row by 0.002 s, 63 rows by more than 1 s and one by 129 s,
with the same 80 heads, ridge strength and best head; the unmodified worker at 8
against 32 threads differs the same way (75 rows above 1 s, 179 s at most). Where the
cap does not bind, old and new worker wrote byte-identical libraries. Predictions on a
GPU do not depend on the torch thread count. The cap-binding sweep on a 64-core host,
with peptides at 1% on `run_psm_q` over seeds, is the validation that remains
(`docs/08_rt_im_train.md` section 4d).

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
  `:358-413`) or a disk-backed float32 **memmap** streamed in `MUMDIA_NN_CHUNK`
  chunks with mean/std accumulated in one pass (`:415-463`), selected at
  `nn_rescore_worker.py:290-302` by comparing an estimated decoded size against
  `MUMDIA_NN_STREAM_GB` (auto: twice the free physical memory, never below 4 GB;
  #88). For a tab-separated PIN the compared
  quantity is the on-disk file size; for a Parquet feature table
  (`rescore.handoff = parquet`, accepted by this worker only, `rescore.rs:943-959`)
  it is `num_rows * (num_columns - 3) * 4`, the decoded float32 feature matrix,
  because the column store on disk is several times smaller than the memory a
  full read needs (`nn_rescore_worker.py:292-298`). The streaming backend is what
  makes an experiment-wide multi-run rescore tractable: the full matrix never
  lives in RAM.
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
out-of-fold with `GroupKFold` grouped by `base_peptide_id`; a training fold with a
single class is an error, never a gap filled with in-sample scores (docs/29 #3), and
a final model fit on all non-decoy PSMs scores the decoys only. Model is `gbm`
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
target+decoy, re-sort by `precursor_mz`, and reassign contiguous `candidate_id`,
which is what satisfies `index.rs load()`'s contiguous-id and m/z-ordering
preconditions; they also re-sort fragments by `candidate_id`, which the index no
longer requires (`make_reverse_decoys.py:156-161`,
`make_shift_decoys.py:47-55`).

`augment_library.py` closes a different gap: an imported DIA-NN library can be
missing tryptic peptides that are present in the FASTA, so the search DB
structurally cannot find them. It fixes this completeness gap by reusing the
engine's own stages, which guarantees the augmented peptidoform strings are
byte-identical to what the native path would emit
(`augment_library.py:1-37`). The flow is: (1) run `mumdia digest` (with
N-terminal Met-excision on) then `peptidoforms` over the FASTA
(`augment_library.py:96-98`); (2) set-diff the resulting base sequences against
the imported library's TARGET base (stripped) sequences, keeping only the missing
ones (`augment_library.py:103-104, 110-114`); (3) run `mumdia predict-frag` on the
missing set to produce native predicted spectra and iRT
(`augment_library.py:126-131`); (4) offset `peptidoform_id`/`base_peptide_id`/
`candidate_id` so the new entries are disjoint from the imported ids
(`augment_library.py:137-144`); (5) per-precursor (`groupby candidate_id`)
max-normalize `predicted_intensity` (`augment_library.py:148-149`); (6) merge the
imported targets with the missing targets (`augment_library.py:156-162`); (7) hand
the merged target library to `make_shift_decoys.py` (default) or
`make_reverse_decoys.py` for the paired, collision-free decoy population
(`augment_library.py:166-171`); (8) validate load invariants on the final library:
`candidate_id` contiguous over `0..N-1`, `precursor_mz` monotonically increasing,
and both `target` and `decoy` labels present (`augment_library.py:178-193`). The
pairing and collision-free guarantee lives in the downstream decoy builder, not in
`augment_library.py` itself; its own validate step only checks contiguity,
`precursor_mz` ordering, both-labels presence, and (for `--decoy-strategy reverse`
only) target/decoy stripped-sequence non-overlap
(`augment_library.py:190-193`). The predicted entries' RT axis need not match the
imported DIA-NN iRT axis, because the per-run DeepLC fine-tune
(`rt_im_train.finetune_deeplc`) re-predicts iRT for the whole library, putting
every entry on one axis before extraction, so no explicit reconciliation is done
(`augment_library.py:21-23`). CLI contract (all positional-style flags): required
`--fasta --imported-precursors --imported-fragments --out-precursors
--out-fragments --mumdia-bin --work-dir`, optional `--config` (default None),
`--match-level {base_sequence(default),peptidoform_charge}`, and `--decoy-strategy
{shift(default),reverse}` (`augment_library.py:67-91`).

## Key types and functions

| name | file:line | what it does |
|---|---|---|
| `resolve_script` | `sidecar.rs:20` | Resolve a worker path: CWD-relative dir, then `<exe_dir>/<dir>`, then `<exe_dir>/scripts`, else CWD-relative fallback |
| `run_worker` | `sidecar.rs:217` | Spawn `python <script> <argv...>`; `utf8=true` sets `PYTHONUTF8`/`PYTHONIOENCODING` (DeepLC/Keras crash on Windows cp1252) |
| `run_ms2pip` | `sidecar.rs:42` | Write ms2pip_in.parquet, run worker, fold output to `HashMap<u32,HashMap<(u8,u16),f32>>` |
| `run_deeplc` | `sidecar.rs:81` | Write deeplc_in.parquet, run worker with the engine's thread count, return `HashMap<u32,f32>` (`utf8=true`) |
| `run_deeplc_finetune` | `sidecar.rs:111` | Run `deeplc_finetune.py <lib_in> <seed> <lib_out> --epochs --patience --q-train --batch --window-holdout-frac --seed --predict-threads` (`utf8=true`) |
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

## Interpreter resolution

Each sidecar role has one config field naming the Python that runs it:
`rescore.python`, `predict_frag.deeplc_python`, `predict_frag.ms2pip_python`,
`mbr.python`. Implemented in `rust/mumdia/crates/mumdia/src/python.rs`.

A field may hold an absolute path, which is used as given and never
second-guessed, or the string `"auto"` (or be absent), which asks the engine to
find one. Discovery order, from `python.rs`:

| order | source | provenance reported |
|---|---|---|
| 1 | `MUMDIA_PYTHON_RESCORE` / `_DEEPLC` / `_MS2PIP` / `_MBR` | the variable name |
| 2 | `MUMDIA_PYTHON` (all roles) | `MUMDIA_PYTHON` |
| 3 | `CONDA_PREFIX`: `bin/python`, `python.exe`, `Scripts/python.exe` | `CONDA_PREFIX` |
| 4 | `VIRTUAL_ENV`, same three layouts | `VIRTUAL_ENV` |
| 5 | `python3`, then `python`, on `PATH` | `PATH` |

A candidate is accepted only after it imports the role's own module list, so
discovery cannot pick a Python without torch and defer the failure to the rescore
stage hours later. The module lists live on `Role::modules` and are the same ones
`doctor` probes:

| role | modules |
|---|---|
| `Rescore`, `classifier = nn_torch` | torch, numpy, pandas, pyarrow |
| `Rescore`, mokapot or entrapment | mokapot, sklearn, numpy, pandas, pyarrow |
| `DeepLc` | deeplc, numpy, pandas, pyarrow, torch, psm_utils |
| `Ms2pip` | ms2pip, numpy, pandas |
| `Mbr` | numpy, pyarrow |

A role is resolved only when the configuration actually uses it
(`Role::required_by`): the rescorer classifier, `rt_predictor`/`finetune_deeplc`,
`predictor`, and `mbr.strategy` respectively. A default native run therefore
needs no interpreter, probes nothing, and works on a machine with no Python. A
role that is needed but unresolvable is a hard error in preflight, naming the
field, the environment variable, and the modules required.

Resolution runs before the config hash is computed, so `manifest.json` records
the interpreter that actually ran rather than the word `auto`. This makes the hash
machine-specific for an `auto` config, which is the honest outcome: two runs whose
rescorer came from different environments are not the same configuration.

`predict_frag.sidecar_script_dir` is resolved the same way
(`python::resolve_script_dir`): the configured value if it holds the workers, then
the same path relative to the config file's own directory, then `scripts/` beside
the executable, which is the release-archive layout. Previously it was interpreted
against the current working directory alone, so invoking the same config from
another directory silently changed which scripts ran.

## Configuration

Config was recently pruned; every field below exists in `mumdia-core/src/config.rs`
today. Predictor/rescorer selection is by strategy enum, and sidecars are engaged
only when a non-native enum is set AND the corresponding Python interpreter is
configured.

| field | default | effect |
|---|---|---|
| `predict_frag.predictor` | `native` | `ms2pip` engages `ms2pip_worker.py` (requires `ms2pip_python`) |
| `predict_frag.rt_predictor` | `native` | `deeplc` engages `deeplc_worker.py` (requires `deeplc_python`) |
| `predict_frag.ms2pip_model` | `"HCDch2"` | 3rd positional arg to `ms2pip_worker.py`; `*ch2` models emit the `b2`/`y2` series as `frag_charge` 2 |
| `predict_frag.ms2pip_python` | `None` | interpreter for MS2PIP (env with ms2pip+pyarrow) |
| `predict_frag.deeplc_python` | `None` | interpreter for DeepLC predict AND fine-tune |
| `predict_frag.sidecar_script_dir` | `"scripts"` | dir passed to `resolve_script` for all workers |
| `rt_im_train.finetune_deeplc` | `false` | engage `deeplc_finetune.py` in `run` (requires `deeplc_python`) |
| `rt_im_train.finetune_epochs` | `25` | `--epochs` upper bound (early stopping usually halts first) |
| `rt_im_train.finetune_patience` | `10` | `--patience` epochs without val-loss improvement |
| `rt_im_train.finetune_batch` | `0` | `--batch`; 0 = auto-scale to seed size in the worker |
| `rt_im_train.q_train` | `0.01` | `--q-train` max seed `spectrum_q` for the fine-tune reference |
| `rt_im_train.window_holdout_frac` | `0.0` | `--window-holdout-frac`; excludes `base_peptide_id %% 1000 < round(frac*1000)` anchors from the fine-tune reference so rt-im-train can size `w_rt` on them held-out (rule duplicated in `rt_im_train.rs::is_holdout`; see `docs/08` section 4b) |
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
`MUMDIA_NN_BATCH` (4096), `MUMDIA_NN_SEEDS` (1), `MUMDIA_NN_SEED` (0, base seed; ensemble member s uses SEED + s, so seeded repeats of one configuration set it to 1, 2, ...), `MUMDIA_NN_STREAM` (auto),
`MUMDIA_NN_STREAM_GB` (4), `MUMDIA_NN_CHUNK` (250000), `MUMDIA_NN_INIT_SAMPLE`
(300000; when no feature reaches the training FDR on that sample the init scan is
repeated on 4x the rows up to the whole fold), `MUMDIA_NN_INIT_FDR_MAX` (0.05; ceiling
of the first-iteration bootstrap ladder 0.02/0.05/0.1 used only when the init feature
selects no positive at the training FDR over the whole fold, 0 = hard error as before)
plus the three the Rust caller injects: `MUMDIA_NN_FOLDS` (worker default
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
- **DeepLC import order is load-bearing.** In both DeepLC workers `import deeplc`
  must execute before numpy and pyarrow at module scope
  (`deeplc_worker.py:13-29`; `deeplc_finetune.py` was already ordered this way).
  DeepLC 4.x is torch-backed, and on Windows importing numpy (and the pyarrow
  that follows it) first aborts torch's DLL initialisation with
  `OSError: [WinError 1114] ... Error loading "...\torch\lib\c10.dll"`.
  `deeplc_worker.py` previously deferred `import deeplc` into `main()`, which put
  it after the module-level numpy/pyarrow and reproduced the crash. The fault was
  latent because imported-library mode skips predict-frag entirely, so only a
  FASTA-mode library build exercises `deeplc_worker.py`. Do not let an import
  sorter reorder these lines.
- **Parquet written outside `mumdia-io` must be SNAPPY plus arrow `utf8`.** The
  engine's `parquet` dependency is built with `default-features = false,
  features = ["arrow","snap"]` (`rust/mumdia/Cargo.toml:23-24`), so SNAPPY is the
  only codec compiled in and a zstd file fails at read with
  `Parquet error: Disabled feature at compile time: zstd`. `Table::str`
  downcasts to arrow `StringArray` only and rejects anything else with
  `column '<name>' is not utf8` (`mumdia-io/src/table.rs:503-511`), so a
  64-bit-offset `large_utf8` string column is also refused. `mumdia-io` itself
  always writes SNAPPY (`mumdia-io/src/table.rs:205`), and the pandas-based
  recipe scripts get both defaults right through pyarrow
  (`import_diann_lib.py:175-176`). A hand-written helper does not: Polars
  defaults to zstd and `large_utf8` and produces a library the engine cannot
  load. Cast string columns to `pa.string()` and write with
  `compression="snappy"`.
- **The nn_torch backend threshold is a cliff, not a preference.** A feature
  matrix marginally over `MUMDIA_NN_STREAM_GB` takes the disk-backed memmap path,
  which is much slower than in-memory; a 4.31 GB matrix against the 4.00 GB
  default was observed doing so, and a 4.52 GB matrix cost 166 minutes against 20
  before the threshold was sized from free memory (#88).
- **The in-memory backend holds one matrix and little else** (2026-09-16). The
  engine writes the handoff parquet in 131,072-row groups and releases its own
  `FeatureMatrix` before the worker starts (under `rescore.strict`, which has no
  native fallback), and the worker reads one row group at a time; pyarrow's
  `iter_batches` reads ahead and had buffered a second copy of the matrix. Six-run
  Astral pool: process tree 17.9 GB before, under 10 after; HYE B01 12.0 -> 5.05 GB,
  identical identifications (`docs/27` section 0.1).
- **Torch CPU threads are capped** at 16, or at the performance-core count on a
  hybrid CPU (Windows `GetLogicalProcessorInformationEx`); the engine's `--threads`
  arrives as `MUMDIA_NN_THREADS` and is an upper bound. The MLP is flat past 16
  threads on two EPYC generations, and on an i9-13900KS (8P + 16E) 30 threads made
  the pooled Astral rescore 118 minutes against 74 at 8, before the subnormal fix below. The
  worker prints `torch cpu threads=N (asked A from ...; cap C: why)` at startup;
  `MUMDIA_NN_THREAD_CAP` overrides the cap, `0` removes it.
- **Constant feature columns are dropped before training** (`MUMDIA_NN_DROP_CONSTANT`,
  default 1; 2026-09-16), identified from the parquet footer's per-column min/max
  without a read (11 of the 387 Extended features on the Astral pool, `has_ms1` and
  ten all-zero deconvolution/peak-sharing fractions). A constant column standardises
  to exactly 0 and contributes nothing to any prediction; its first-layer weights are
  the main subnormal source (below), so dropping it is free.
- **Tiny parameters, buffers and Adam moments are clamped to zero once per epoch**
  (`MUMDIA_NN_CLAMP_TINY`, default 1e-20; 2026-09-17). Adam with L2 decay shrinks any
  parameter that receives no data gradient geometrically until it crosses 1.2e-38:
  dead hidden units' weights, their BatchNorm scale and shift, the running variances,
  and Adam's own first and second moments (0.9^k and 0.999^k). Setting them to
  exactly 0 below 1e-20 changes no prediction and leaves nothing that can become
  subnormal, on any CPU. Desktop six-run pool with flush-to-zero off: 11.9 min and a
  census of 0 everywhere (18.8 min with the moments left unclamped, 61.8 with no
  clamp at all); two-run pool 4.31 min against 7.64.
- **Subnormal floats are flushed to zero** (`MUMDIA_NN_FLUSH_DENORMAL`, default 1;
  2026-09-16). On Intel cores a subnormal operand turns a 4 ms `Linear` into a
  475 ms one (measured, i9-13900KS; an EPYC 9354 is unaffected), and the trained
  network accumulates them in the first-layer weights of the constant features
  (standardised to exactly 0, so Adam's L2 term is their only gradient: census
  1 -> ~1,180 subnormal parameters over the first rounds, 0 in buffers or
  activations) while the input features carry none. `MUMDIA_NN_DEBUG_DENORMALS=1` prints a census after every
  training round. Desktop A/B on a two-run pool at 8 threads: 7.64 min with the
  FPU default, 5.07 min with flush-to-zero, identical peptides (96,137). The worker prints the size, the threshold, and
  the chosen backend before it starts (`nn_rescore_worker.py:306-311`), so check
  that line rather than inferring the backend from the wall clock. Raise
  `MUMDIA_NN_STREAM_GB` when the RAM is available, or force the choice with
  `MUMDIA_NN_STREAM=1`/`0`.
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
- **DIA-NN index precondition.** `index.rs load()` enforces two hard
  preconditions on any library, imported or native, and bails with a message
  naming the offending row rather than degrading silently. First, precursor
  `candidate_id` must be the contiguous row-aligned range `0..ncand`, checked
  row by row (`index.rs:112-125`). Second, precursors must be ascending by
  `precursor_mz`, because the fragment index's `partition_point` search over
  `prec_mz` assumes it and an unsorted import would return wrong candidate
  windows (`index.rs:215-231`). Both decoy builders and the importer satisfy
  these by re-sorting on `precursor_mz` (stable mergesort) and reassigning
  contiguous ids; do not reorder the precursor table afterward. Fragment **order**
  is not a precondition: fragments are grouped by a counting sort that preserves
  stored order (`index.rs:126-153`), so a fragment table only needs every
  `candidate_id` to be less than the precursor count (`index.rs:133-139`). The
  recipe scripts still sort fragments by `candidate_id`, which is harmless but no
  longer load-bearing. `make_reverse_decoys.py` additionally aborts if its
  residue-mass calculator disagrees with the library's own target fragment m/z by
  > 5 ppm at p99.
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
  add a JSON request file or a server. Write the output Parquet with SNAPPY
  compression and arrow `utf8` string columns (pyarrow and pandas defaults
  satisfy both; Polars does not, see the gotcha above), and if the worker loads a
  torch-backed package, import that package first.
- **Reuse the PIN contract.** A new rescorer that reads the PIN and writes
  `candidate_id`+`score`+`q_value` needs no new Rust plumbing beyond a
  `RescorerKind` arm calling `run_pin_sidecar` with its script name; the
  `MUMDIA_NN_*` env vars are already injected.
- **Change env probing.** `doctor` (`main.rs:346-413`) hard-codes the package list
  per interpreter and switches the rescore packages on the classifier: `nn_torch`
  -> `torch,numpy,pandas,pyarrow`; every other classifier (mokapot, entrapment,
  percolator, native) -> `mokapot,sklearn,numpy,pandas,pyarrow`
  (`main.rs:351-357`); `predict_frag.deeplc_python` ->
  `deeplc,numpy,pandas,pyarrow,torch,psm_utils` (`main.rs:368-371`);
  `predict_frag.ms2pip_python` -> `ms2pip,numpy,pandas` (`main.rs:372-376`). The
  DeepLC list covers both scripts that run on that interpreter, because
  `deeplc_finetune.py` imports pyarrow, torch and psm_utils on top of deeplc
  itself; probing only `deeplc,numpy,pandas` let a green `doctor` precede a crash
  at the fine-tune step, which on an experiment-wide batch surfaces long after the
  run is launched. An interpreter left `None` prints `[skip]` (native path,
  `main.rs:381`). It probes with `importlib.util.find_spec` and reports
  `MISSING <pkgs>`; note `mbr.python` is **not** probed by `doctor`. The probe
  only asks whether a module is importable, so it cannot catch an ordering fault
  like the DeepLC/torch one above. Update these lists when a worker's imports
  change so `mumdia doctor` stays truthful.
- **Conda envs.** The committed reproducible specs are `env/docker-rescore.yml`
  (env `rescore`: mokapot 0.10.0 + ms2pip 4.0.0.dev9, py3.11) and
  `env/docker-deeplc.yml` (env `deeplc`: DeepLC 4.4.0 + CPU torch, py3.11); the
  Docker configs point interpreters at `/opt/conda/envs/{rescore,deeplc}/bin/python`
  (`docker/config.dia.json`, `docker/config.diann-lib.json`). For a native install
  the portable equivalents are `env/mumdia-rescore.yml` (mokapot logreg path, no
  torch/DeepLC/MS2PIP) and `env/mumdia-deeplc.yml` (the DeepLC sidecars).
  **DeepLC 4.4.0 is a floor, not merely the current release**: the 4.0.0a2
  multitask preview overfits per-run fine-tuning badly enough to invert RT-model
  rankings (`docs/08_rt_im_train.md` section 4b), so an older DeepLC changes
  results and not only performance. The engine enforces it: `mumdia doctor` fails
  below 4.4.0 and `sidecar::require_deeplc_version` refuses to launch a DeepLC worker
  (one constant, `mumdia_core::constants::MIN_DEEPLC_VERSION`). Anchor the tool version and let pip resolve
  its scientific-Python graph; do not re-add an exact `pandas < 2` style pin,
  which has no cp312 wheel. A developer machine may also have older local envs
  (`deeplc_mt` holds the superseded a2 build); prefer the committed specs.
- **nn_torch scaling.** For a many-run experiment-wide rescore, force the streaming
  backend with `MUMDIA_NN_STREAM=1` (or rely on the `MUMDIA_NN_STREAM_GB`
  threshold, default 4 GB) so peak RAM is one minibatch, not the whole feature
  matrix; the memmap sidecar file is written next to the output as
  `<out>.feat.mm` and deleted on exit (`nn_rescore_worker.py:424-425, 719-722`).
  Going the other way is also a deliberate choice: raising `MUMDIA_NN_STREAM_GB`
  above the estimated matrix size keeps a merely large rescore in memory, and the
  worker's startup line reports which side of the threshold it landed on.
