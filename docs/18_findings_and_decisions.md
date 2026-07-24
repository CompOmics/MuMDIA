# 18. Findings, decisions, and contracts

This document is a self-contained reference. It restates the validated findings,
the interstage and determinism and sidecar contracts, the current best workflow,
and the ranked roadmap, so that no other document is required to understand how
MuMDIA behaves and why. It does not depend on any gitignored specification file;
everything needed is stated inline here.

MuMDIA is a clean-room Rust reimplementation of a data-independent-acquisition
(DIA) proteomics search engine. It runs the full DIA chain end to end: mzML in,
identified peptides and proteins at target-decoy-controlled FDR out. It builds a
spectral library from either an in-silico FASTA digest or an imported predicted
library, extracts candidates against a peak-major inverted fragment index, and
rescores with a semi-supervised classifier. The pipeline is a DAG: mzML
conversion and library creation/import are independent branches that converge at
`search-seed`; optional DeepLC fine-tuning then sits between seed search and RT
calibration. Extraction, features, competition, rescoring, quantification, and
reporting follow. `align`, `mbr`, and `quant-lfq` are separate experiment-level
commands.

---

## Part A: Validated findings

Steer development by these. Unless noted, the benchmark is
`LFQ_Orbitrap_AIF_Ecoli_01.mzML` with an imported DIA-NN E. coli library whose
iRT has been fine-tuned (the `lib/lib_precursors_ft.parquet` library), the
Extended feature set, and all reported points are FDR-valid (measured decoy
fraction near 0.98 to 0.99 percent at the 1 percent threshold). Peptide counts
are at 1 percent FDR on the `peptide_q_value` column. The extraction threshold is
`extract.min_frag_corr`; with the default `gate_mode=apex_pearson`, it is the
Pearson correlation between observed and predicted fragment intensities at one
apex scan. It is temporal co-elution only under `gate_mode=coelution` (or the
co-elution half of `combined`).

### A1. The rescorer is the dominant sensitivity lever

The choice of rescoring classifier moves identifications more than any other
single lever. Comparing `nn_torch` (a PyTorch MLP semi-supervised rescorer) to
`native_tda` (the native L2-regularized logistic-regression `percolator_lite`),
on spectra capped at 300 MS2 peaks, sweeping the extraction gate:

| gate | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|
| `native_tda` | 9014 | 9110 | 9274 | 9444 | **9503** | 9402 | 9261 |
| `nn_torch`   | **10308** | 10256 | 10232 | 10147 | 10062 | 9857 | 9354 |

Best `native_tda` = 9,503 peptides at gate 0.6. Best `nn_torch` = 10,308 at gate
0.2. That is a +8.5 percent gain from the rescorer alone. For reference, the
zero-dependency native FASTA-digest mode (no imported library, native predictors)
finds about 1,213 target peptides. Invest in the rescorer before almost anything
else. The cost of `nn_torch`: it needs a Python interpreter with PyTorch, and its
training is only approximately reproducible (see the determinism contract in Part
B).

The gate optimum inverts by rescorer. `native_tda` shows an inverted-U: the count
rises to a peak at a tight gate near 0.6, then falls. `nn_torch` is monotonic
across the tested range, looser-is-better, peaking at the loosest gate tested
(0.2), and would likely gain further below 0.2. A linear rescorer cannot
represent a hard floor on the co-elution feature, so a pre-gate does real work
for it; a strong nonlinear rescorer absorbs the loose-gate flood and just wants
recall. The hard gate is a linear-rescorer crutch. The extraction gate is a
finite floor in `[0, 1]` at `extract.min_frag_corr`, default 0.2
(`rust/mumdia/crates/mumdia-core/src/config.rs:522`).

### A2. Mechanism of the gate

Target-decoy FDR depends on the decoy density near the score boundary, and
semi-supervised rescoring learns from the pool of candidates it is handed.
Loosening the gate does not add signal; it adds decoy-enriched junk. Measured on
this benchmark, the decoy fraction of the competed pool rose from 44.8 percent at
gate 0.5 to 47.3 percent at gate 0.2 as the gate loosened. That flood does two
harmful things at once: it pushes the q-value threshold stricter (more decoys
near the boundary), and it corrupts the learned model (the training pool is
contaminated), which degrades the scores of the good candidates too. So the gate
is training-pool curation plus null curation, not feature engineering. A hard
fixed threshold is the wrong instrument. The right fix is a soft or budgeted gate:
prune the worst junk to protect the null and the training pool, keep the
borderline candidates, and pass the gate score through to the rescorer as a
  feature rather than using it as a drop criterion. These mechanism claims refer
  to the default apex-intensity Pearson gate used in the benchmark, not a
  chromatographic co-elution threshold.

### A3. MS2 peak cap: keep 300 on AIF

Uncapping the per-scan MS2 peak count (from a cap of 300 up to roughly 970 peaks
on some scans) is neutral to slightly negative under both rescorers on this
chimeric AIF file. For example, at gate 0.2 `nn_torch` gives 10,308 peptides
capped versus 10,251 uncapped. On chimeric all-ion-fragmentation data the extra
low-intensity peaks are interference, not signal, so the cap helps. This is
instrument-dependent: narrow-window Orbitrap-Astral data behaves differently and
should be re-evaluated on its own. The cap is applied at conversion time by
`peaks_of`, which keeps the top-N most intense peaks with 0 meaning no cap.
Both `run --top-peaks-ms2` and standalone `convert --top-peaks-ms2` default to
0 (uncapped). Reproducing this AIF result therefore requires the explicit
conversion flag `--top-peaks-ms2 300`. This is independent of
`search_seed.top_n_peaks=300`, which limits seed probing only and does not remove
peaks from downstream extraction or quantification.

### A4. DeepLC multitask fine-tune of iRT is essential in library mode

Raw DIA-NN predicted iRT is locally noisy, with a residual around 110 seconds MAD
against observed RT on both the E. coli and HYE datasets. A per-run DeepLC 4.0
multitask fine-tune on the confident seed PSMs tightens this to roughly 13 to 27
seconds MAD, which narrows the RT windows and materially lifts identifications.
The fine-tune is therefore essential, not optional, in library-input mode. Do not
use an already-fine-tuned library (a `_ft` or reversed-decoy library) as a "raw
DIA-NN" baseline; the raw baseline is the un-fine-tuned precursor library. The
fine-tune runs between search-seed and RT calibration and writes a new
`fragment_library_precursors_ft.parquet` with replaced `predicted_irt`; the input
library is unchanged and downstream stages are rebound to the new table. It is
nondeterministic (see Part B).

### A5. q-value units are independent per level, not a rollup

MuMDIA computes q-values independently at each grouping level. They are not a
monotone rollup of one another. The columns written by `rescore` are:

- `q_value`, equal to `experiment_psm_q` and to the alias `global_q_value`: the
  pooled per-PSM FDR.
- `run_psm_q`: target-decoy FDR recomputed within each source (run) separately.
- `precursor_q`: FDR grouped on peptidoform plus charge.
- `peptide_q_value`: FDR grouped on stripped sequence.
- `pg_q_value`: protein-group FDR.

See `rust/mumdia/crates/mumdia/src/stages/rescore.rs:288` through `:396` for the
computation (PSM, peptide, protein-group, per-run, and precursor levels) and
`:447` through `:477` for the emitted columns. The estimator is
`q = (n_decoys + 1) / max(1, n_targets)`, monotonized, with tied-score blocks
collapsed to one q (`rust/mumdia/crates/mumdia/src/fdr.rs:38`).

Because coarser grouping pools evidence across PSMs, a peptide count can exceed a
precursor count at the same threshold (here precursors run about 1 percent below
peptides). Practical consequences: for a ProteoBench precursor matrix, filter on
`precursor_q`; do not threshold PSM q and then deduplicate, which overstates the
peptide-level FDR. For cross-run library-mode quantification, filter on
`run_psm_q`, not `peptide_q_value` (the latter is the global-best-PSM value under
experiment-wide rescoring and yields near-disjoint per-run sets).

### A6. The selected apex is the correct peak only about half the time

Diagnostics on this benchmark show that the single apex the extractor selects is
the strongest or correct chromatographic peak only about 48 to 52 percent of the
time; the correct peak is within the top 5 about 86 percent of the time.
Intensity is chimeric in DIA, so ranking candidate peaks by intensity is weak;
ranking by co-eluting-fragment breadth (how many predicted fragments co-elute at
that peak) beats ranking by intensity. This motivates making the top-K peaks
model-visible: retain the top-K peaks per candidate, carry them as
`candidate_id + peak_rank` rows through features and rescore, and let `compete`
collapse them after scoring rather than committing to one apex before the model
sees them.

### A7. Instrument validation

MuMDIA has been validated on three instrument classes: Orbitrap AIF (E. coli and
HYE), SCIEX TripleTOF SWATH (E. coli, about 11,855 peptides, which is about 74.8
percent of the DIA-NN count on the same file), and Orbitrap-Astral narrow-window
DIA (HYE). Where compared against DIA-NN, MuMDIA identifications are about 91 to
97 percent sequence-concordant. The engine is high-precision; sensitivity is the
gap, and it is closed most by the imported-library recipe plus the `nn_torch`
rescorer.

---

## Part B: Contracts

These three contracts are what let the pipeline be modular, reproducible, and
extensible. They are stated here in full so that no separate specification is
needed.

### B1. Interstage contract

Computational stages exchange path-addressable Parquet (and JSON for scalars and
configuration) rather than shared in-memory state. Most can run standalone on
prior outputs. `run` does not cache, skip unchanged work, or resume: it
unconditionally recomputes its chain, so use a fresh output directory and invoke
selected downstream stages manually when reusing artifacts. Most primary
Parquets receive `<artifact>.report.json`, and selected primary Parquets are
recorded in `manifest.json`; coverage is not universal for diagnostics,
PIN/schema companions, TSVs, or Python-written outputs.

Where implemented, the per-artifact report is written by
`ArtifactReport::write_for`, which writes
`<artifact_path>.report.json` next to the Parquet
(`rust/mumdia/crates/mumdia-io/src/report.rs:28`). The report carries the logical
name, schema name and version, stage, row count, a content hash, the resolved
parameters the stage actually used, summary statistics, an optional model
identity, and elapsed time
(`rust/mumdia/crates/mumdia-io/src/report.rs:10` through `:24`). This means a
stage can be evaluated, and its provenance checked, without loading the full
table.

### B2. Determinism contract

Identical inputs and configuration must produce byte-identical outputs. Two rules
enforce this: seed every RNG, and keep numeric summation and iteration order
fixed. Ordered maps or sorted iteration are used wherever floats are summed,
because floating-point addition is not associative and an unordered sum can shift
a result (a HashMap f32 sum once shifted a chromatographic apex and broke
reproducibility).

Concrete guarantees in code:

- Decoy generation is seeded. The scramble decoy runs a deterministic
  Fisher-Yates shuffle driven by a splitmix64 PRNG seeded per peptide from the
  configured `rng_seed` XORed with an FNV-1a hash of the sequence
  (`rust/mumdia/crates/mumdia/src/stages/digest.rs:117` for the seeding, `:159`
  for `splitmix64`, `:167` for `fnv1a`). Native digest decoys are additionally
  collision-checked by `collision_safe_decoy`, which deterministically retries
  with independently seeded interior scrambles when a transform collides with a
  target or an already-emitted decoy while keeping the C-terminal residue fixed
  (`rust/mumdia/crates/mumdia/src/stages/digest.rs:137`).
- FDR q-values are order-independent. Tied-score blocks are processed together so
  every PSM in a block gets the same q regardless of its arbitrary within-tie
  order, and monotonization runs worst-to-best
  (`rust/mumdia/crates/mumdia/src/fdr.rs:26` through `:52`).
- Per-run q-values iterate sources through a `BTreeMap` (sorted keys), and the
  code notes explicitly that no floats are summed there
  (`rust/mumdia/crates/mumdia/src/stages/rescore.rs:347` through `:376`).

Two known exceptions are nondeterministic by design and are documented as such:

- The DeepLC multitask fine-tune (`scripts/deeplc_finetune.py`) sets no torch or
  numpy seed, so refitting the RT model is not bit-reproducible.
- The `nn_torch` rescorer (`scripts/nn_rescore_worker.py`) sets
  `torch.manual_seed` and `np.random.seed` per pass
  (`scripts/nn_rescore_worker.py:197`, `:284`, `:285`), but its own header states
  that NN training is only approximately reproducible, and it offers
  `MUMDIA_NN_SEEDS > 1` to ensemble seeds and average rank-normalized
  out-of-fold scores for stability (`scripts/nn_rescore_worker.py:29` through
  `:30`). Treat `nn_torch` as seeded and approximately reproducible, but not
  bit-deterministic.

### B3. Sidecar contract

Real ML predictors and rescorers (MS2PIP, DeepLC, the DeepLC fine-tune, mokapot,
the PyTorch NN, entrapment, MBR) run as opt-in Python workers behind one file
contract. The Rust caller writes an input Parquet (or a Percolator PIN for the
rescorers), invokes the worker as a subprocess with positional command-line
arguments (input path, output path, then any extra flags), and reads back an
output Parquet. Predictor output is keyed by stable per-candidate `id`. PIN
rescorers instead encode the concatenated flat row ordinal in `SpecId` and echo
that ordinal in a column named `candidate_id`; it is deliberately not the
library candidate ID, which repeats across source runs. There is no JSON request
file; argv positions and the output schema are the contract.

Concrete points in code:

- The subprocess invocation is `python script arg...` with positional args
  (`rust/mumdia/crates/mumdia/src/sidecar.rs:217` through `:233`).
- MS2PIP is invoked as `<in> <out> <model>` and its output is read back keyed by
  `id` (`rust/mumdia/crates/mumdia/src/sidecar.rs:63`, `:66` through `:76`).
- DeepLC is invoked as `<in> <out>` and read back keyed by `id`
  (`rust/mumdia/crates/mumdia/src/sidecar.rs:99` through `:104`).
- The DeepLC fine-tune is invoked as `<lib_in> <seed> <lib_out>` plus flags
  (`rust/mumdia/crates/mumdia/src/sidecar.rs:111` through `:155`).
- The NN rescorer reads a PIN and writes the flat PIN-row ordinal back in its
  `candidate_id` column; Rust maps scores by that unique row index.

Worker scripts are located by `sidecar::resolve_script`, which tries the
configured directory relative to the working directory, then relative to the
binary's own directory, then `<exe_dir>/scripts`
(`rust/mumdia/crates/mumdia/src/sidecar.rs:20` through `:38`). On Windows, set
`predict_frag.sidecar_script_dir` to an absolute path with a drive letter
(`c:/...`), not a git-bash `/c/...` path, or the binary cannot find the worker
and the strict default aborts. Only an explicit `rescore.strict=false`
compatibility config falls back to `native_tda`.

---

## Current best workflow and rationale

Library-input mode is the recipe that wins:

1. Import a DIA-NN predicted library (fragment intensities plus iRT) and generate
   paired decoys (license-clean: MuMDIA ships no DIA-NN; the user runs DIA-NN
   under their own license, then `import_diann_lib.py` and `make_reverse_decoys.py`
   or `make_shift_decoys.py`).
2. Enable the per-run DeepLC multitask fine-tune of iRT
   (`rt_im_train.finetune_deeplc = true`). Rationale: finding A4, raw iRT is too
   noisy.
3. Use the Extended feature set (`features.set = extended`).
4. Use the `nn_torch` rescorer (`rescore.classifier = nn_torch`, with
   `rescore.python` pointing at a torch-capable interpreter and
   `rescore.strict = true`). Rationale: finding A1, the rescorer is the dominant
   lever; strict mode prevents a failed sidecar from masquerading as the intended
   model. Verify `params.classifier` in `psms_scored.parquet.report.json`.
5. Use a loose extraction gate (`extract.min_frag_corr` near 0.2). Rationale:
   finding A1 and A2, the nonlinear rescorer prefers looser-is-better.
6. Keep the conversion-time MS2 peak cap at 300 on AIF by passing
   `--top-peaks-ms2 300`. Rationale: finding A3, uncapping adds interference on
   chimeric data. The seed-only `search_seed.top_n_peaks=300` is separate.

On `LFQ_Orbitrap_AIF_Ecoli_01` this historically reaches about 10,300
`(peptidoform, charge)` rows in `peptides.tsv`, selected with
`peptide_q_value <= 0.01`. That is a precursor-shaped report count under a
peptide-level q filter, not a stripped-peptide count or a precursor-q count. The
zero-dependency native FASTA-digest mode gives about 1,213 rows under the same
definition. See docs/20 for the exact command and promotion gates.

Report the right q-unit for the question asked (finding A5): `precursor_q` for a
ProteoBench precursor matrix, `run_psm_q` for cross-run quantification,
`peptide_q_value` for peptide-level identification counts.

---

## Ranked roadmap

1. **Rescorer.** The biggest lever (finding A1). Make a strong NN rescorer
   first-class and well-tested, and ideally deterministic (seed ensembling for
   stability).
2. **Soft or budgeted extraction gate plus model-visible top-K peaks.** Replace
   the hard `min_frag_corr` drop with a score-ranked per-window candidate budget
   and pass the gate score through as a feature (findings A1, A2). Promote the
   retained top-K peaks (`extract.retain_top_peaks`, currently written to a
   sidecar but not scored) into real `candidate_id + peak_rank` rows through
   features and rescore, and collapse them in `compete` after scoring (finding
   A6). Both need `precursor_q` to FDR-count correctly.
3. **Wire match-between-runs fully and add a run-set orchestrator.** Forward the
   unwired `MbrConfig` knobs, differentiate the `MbrStrategy` variants, keep the
   transfer q as a separate column rather than overwriting `q_value`, and chain
   transfer, search, and MBR under one config.
4. **Quantification maturity.** The current path now consumes the identified
   apex, preserves missing quantities as null with `quant_status`, and prevents
   sibling precursor forms from multiplying protein Top-N abundance. Next,
   benchmark quantifiability gates (minimum clean ions/co-elution), a cross-run
   consensus ion/window policy, and missingness rules on known-ratio datasets.
   Keep identification acceptance separate from quantifiability.
5. **Per-run figure-of-merit tolerance-optimization loop** (set each tolerance
   from a high percentile of the 1-percent-FDR error distribution and iterate to a
   stable precursor count).
6. **Beyond-MVP:** learned protein-group FDR plus parsimony, native vendor
   readers (Thermo `.raw`, Bruker `.d`/TDF), and ion mobility / diaPASEF (full 4D;
   the data model is already IM-nullable throughout).
