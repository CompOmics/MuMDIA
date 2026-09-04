# Python sidecar contract tests

Contract tests for the eleven Python workers in `scripts/`. The Rust unit suite
does not exercise any of them: it can pass while a worker violates a file
contract, and the violation then surfaces hours into a real run, after the
search compute has already been spent.

Every test here builds a tiny synthetic input, invokes the worker as a
subprocess with `sys.executable` exactly as the engine does
(`docs/13_sidecars.md`, "Argv contract"), and asserts the on-disk output
contract. Nothing reads real data: no `lib/`, no `mzml_files/`, no `out_*/`, no
network, no absolute path from any developer machine.

## Running

```bash
cd <repository root>
python -m pip install -r tests/python/requirements.txt
python -m pytest tests/python -q
python -m pytest tests/python -q -rs      # show what skipped and why
```

Run it with the interpreter you want to test. The suite invokes every worker
with `sys.executable`, so running it under `env/mumdia-rescore.yml`'s Python
tests the mokapot path, and running it under `env/mumdia-deeplc.yml`'s Python
tests the DeepLC import order. It never reads `rescore.python`,
`predict_frag.deeplc_python` or any other configured interpreter path.

The whole suite finishes in well under two minutes with torch present, and in a
few seconds when the ML tests skip.

## What skips, and when

Optional dependencies are gated with `importorskip_any` (`conftest.py`), a
variant of `pytest.importorskip` that also skips when a package is installed
but unimportable, naming the real exception in the skip reason. A stale
environment therefore reports "sklearn is not usable here: PydanticUserError:
..." instead of failing a test that has nothing to do with that package.

| needs | skips | unlocks |
|---|---|---|
| nothing | never | `test_mbr_worker.py`, `test_import_diann_lib.py`, `test_decoy_builders.py`, the static half of `test_predictor_workers.py` |
| scikit-learn | `test_entrapment_worker.py` | the entrapment rescorer |
| mokapot | `test_mokapot_worker.py` | mokapot coverage + model selection |
| torch | `test_nn_rescore_worker.py` | `nn_torch` coverage, three input paths |
| deeplc | 2 tests in `test_predictor_workers.py` | the real module import order |
| psm_utils | 1 of those 2 | `deeplc_finetune.py`'s import order |
| ms2pip | 1 test in `test_predictor_workers.py` | the MS2PIP output schema |

No test is expected to fail. Three of them assert that the library helpers
write arrow `utf8` string columns; that assertion was red when this suite was
first written, because `DataFrame.to_parquet` chooses the string width itself
and pandas 3 writes `large_string`, which the engine rejects with
`column 'peptidoform' is not utf8`. The helpers now write through
`scripts/_lib_io.write_engine_parquet`, which narrows the large variants and
pins snappy, and these assertions are what keeps that true: they must fail
loudly if a helper goes back to a bare `to_parquet`.

The import-order tests for the DeepLC workers deliberately gate on package
PRESENCE (`importlib.util.find_spec`) rather than importability, and do the
import in a fresh subprocess. Inside pytest, numpy and pyarrow are already
loaded, which is the broken order; importing a torch-backed DeepLC in that
state reproduces the very `WinError 1114` the ordering rule exists to prevent.

## What each file asserts

### `test_mbr_worker.py` (no ML dependency, always runs)

`scripts/mbr_worker.py` aborts `mumdia mbr` on any nonzero exit: no strict gate,
no fallback.

* **M5 augmentation** (the reason this suite exists). `q_value`, `run_psm_q` and
  `experiment_psm_q` must every one be lowered on an accepted transfer, and
  `is_transferred` set. Lowering only `q_value` left 34,280 of 34,664 transfers
  unquantified on the HYE pooled run, because quant gates on `run_psm_q`.
* The augmentation is a **minimum, not an assignment**: a q already below
  `transfer_q` must not be raised.
* Only the matching **`(candidate_id, source)`** row may change. The fixture
  carries a fourth run holding the same `candidate_id` values with high q and no
  extraction, which a `candidate_id`-keyed augmentation would wrongly promote.
* A scored table **missing** `run_psm_q` and/or `experiment_psm_q` must still
  work, and the worker must not invent the absent column.
* `transferred.parquet` row count equals the number of accepted transfers, and
  its documented columns are present.
* The **permuted-RT decoy null** accepts the RT-concordant candidates and
  rejects the ~500 s discordant ones; every accepted row satisfies
  `--q-transfer`; the accepted set is monotone in that threshold.
* `binned_map` removes a systematic inter-run RT offset, and below its
  200-shared-anchor floor it silently falls back to the identity, leaving the
  full offset in the prediction.
* Determinism for a fixed `--seed`; empty-transfer case writes an empty table
  and no `--out-scored` file; a missing psms path exits nonzero.

### `test_nn_rescore_worker.py` (needs torch)

The hard contract from `CLAUDE.md`, enforced by `align_sidecar_scores`
(`rescore.rs:1046-1082`): every input row gets exactly one finite out-of-fold
score, output rows equal input rows, no row silently dropped. Asserted for all
three input paths - the tab-separated PIN, the `rescore.handoff = parquet`
feature table, and the streaming memmap backend, which must also delete its
`<out>.feat.mm`. Plus: `q_value` written as zeros (the engine computes q),
targets outscore decoys (so the scores are aligned to the rows), a single-class
PIN exits nonzero, and an unknown `MUMDIA_NN_FEATURES` name aborts.

### `test_mokapot_worker.py` (needs mokapot)

The same complete-coverage contract, and that the out-of-fold branch ran (there
is deliberately no in-sample fallback). Plus model selection:
`MUMDIA_RESCORE_MODEL=logreg` and its aliases build a `LogisticRegression`, the
unset default is the sklearn MLP, `percolator`/`linear`/`svm` mean `model=None`,
and an unknown name aborts instead of falling back.

### `test_entrapment_worker.py` (needs scikit-learn)

Complete, finite coverage of every input row keyed on `row_id`, decoys included
even though they are excluded from training. `candidate_id` is echoed per row
and never used to deduplicate: the fixture makes it collide the way it does in a
competed multi-run pool. Real targets outscore the spike-in negatives; a missing
`row_id` falls back to the positional index; a single-class input exits nonzero.

### `test_import_diann_lib.py` (no ML dependency)

The invariants `index.rs load()` hard-errors on: `candidate_id` is the
contiguous row-aligned range `0..ncand`, precursors ascend by `precursor_mz`.
Plus: only mapped targets survive (DIA-NN's own decoys, unmapped UniMods,
neutral-loss and non-b/y fragments dropped); `(UniMod:4)` is not matched inside
`(UniMod:44)`; charges of one stripped sequence share a `base_peptide_id`;
`n_fragments` matches the fragment table; species-flagged protein names survive;
the parquet codec is one the engine can decode; string columns are arrow
`utf8`; `--charge-by-basic-residues` drops rows before ids are assigned; and
the output feeds straight into a decoy builder.

### `test_decoy_builders.py` (no ML dependency)

For both `make_shift_decoys.py` and `make_reverse_decoys.py`: index validity,
one decoy per target with names mapping 1:1, decoys keeping the target
precursor m/z, iRT and charge so they co-isolate and co-elute, and
reproducibility across two builds. Shift-specific: b ions move by `-DELTA/z` and
y ions by `+DELTA/z` for one CH2, net precursor shift zero, intensities copied.
Reverse-specific: `decoy_stripped` disjoint from `target_stripped`, the
C-terminal residue and the residue composition preserved, the three colliding
peptides in the fixture resolved by scramble, distinct base sequences never
sharing a decoy sequence, decoy fragment m/z recomputed as the real b/y of the
decoy sequence, and the 5 ppm calculator check aborting a library whose
fragment m/z do not agree with a residue-mass calculation.

The residue-mass table used to build the fixture lives in `conftest.py` and is
transcribed independently of `make_reverse_decoys.py`, so the reverse builder's
own calculator check is a real cross-validation rather than a tautology.

### `test_predictor_workers.py` (static tests always run)

`import deeplc` must precede numpy, pyarrow and torch at module scope in both
DeepLC workers, and must be at module scope rather than deferred into `main()`.
That ordering is load bearing: DeepLC 4.x is torch-backed and on Windows the
wrong order aborts torch's DLL init with
`OSError: [WinError 1114] ... Error loading "...\torch\lib\c10.dll"`. The fault
is latent because imported-library mode skips predict-frag, so only a FASTA-mode
library build reaches `deeplc_worker.py`, and `mumdia doctor` cannot see it
because `find_spec` only asks whether a module is importable. The same file also
asserts that `deeplc_finetune.py` sets its `OMP`/`MKL`/`OPENBLAS`/
`KMP_DUPLICATE_LIB_OK` caps before importing numpy and torch (setting them
afterwards is a no-op and the two OpenMP runtimes then oversubscribe the CPU),
that `ms2pip_worker.py` keeps `ms2pip`/`psm_utils` inside `main()`, that all
three workers carry the `__main__` guard the Windows `spawn` start method needs,
and - with the packages installed - that a fresh interpreter really can import
each worker and that MS2PIP emits `id`/`ion_type`/1-based `ordinal`/linear
`intensity`.

## Conventions

* Fixtures and helpers live in `conftest.py`: `run_worker` /
  `run_worker_ok(script, *args, env=None)` returning `(rc, stdout, stderr)` with
  stderr in the failure message, PIN and parquet-handoff writers mirroring
  `rescore.rs`, scored/psms writers mirroring `mbr_worker.py`'s inputs, the
  `align_sidecar_scores` mirror, the `index.rs` load-invariant checks, and the
  independent peptide mass arithmetic.
* Every test is deterministic. Anything random is seeded, and no test asserts an
  exact neural-network score: MLP training is only approximately reproducible
  (`docs/13_sidecars.md`, determinism), so the assertions are coverage,
  finiteness and the direction of separation.
* Every test docstring states what breaks in production if the assertion fails,
  not merely what it checks.

## Not covered

* `augment_library.py`: it shells out to the `mumdia` binary for `digest`,
  `peptidoforms` and `predict-frag`, so a contract test needs a built engine.
* The MBR fragment-consensus guard (`--frag-csv` / `--consensus-corr-min`) and
  the re-extraction tier (`--emit-transfer-targets`); `run_mbr` passes the
  former only when both flags are supplied and never passes the latter.
* Real predictor behaviour: MS2PIP intensity values and DeepLC retention times
  are not checked against reference numbers, only the output schema.
* `deeplc_finetune.py` end to end. Fine-tuning is not deterministic and needs a
  DeepLC model plus a seed PSM table; only its import and thread-cap ordering is
  asserted here.
