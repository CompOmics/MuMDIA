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

### A3. The MS2 peak cap is destructive and acquisition-specific

This finding is restated here as a decision record, self-contained by the charter
of this document. The canonical reference for the flag itself is
docs/04_convert.md ("Choosing `--top-peaks-ms2`"), and docs/20 holds the
pre-flight saturation check to run before setting any cap.

The cap is applied at conversion time by `peaks_of`, which sorts each MS2
spectrum by intensity, truncates to the top N, and writes only the survivors
(`rust/mumdia/crates/mumdia/src/stages/convert.rs:76` through `:79`). The
truncation is baked into the `spectra_ms2.parquet` artifact: extraction applies
no cap of its own and consumes whatever `convert` wrote. This is a different
mechanism from `search_seed.top_n_peaks` (default 300,
`rust/mumdia/crates/mumdia-core/src/config.rs:410` through `:415`), which is
non-destructive and only bounds the per-peak index-probing cost of the seed
search. The independence is measured, not just documented: on one file the seed
output was identical (80,474 PSMs, 14,877 confident) with and without the
conversion cap.

On the chimeric AIF benchmark a cap of 300 is neutral to slightly positive. At
gate 0.2, `nn_torch` gives 10,308 peptides capped versus 10,251 uncapped, the
uncapped scans reach roughly 970 peaks, and only 47.8 percent of MS2 spectra
there hold more than 300 peaks. On chimeric all-ion-fragmentation data the extra
low-intensity peaks are largely interference.

That result does not transfer, and applying it blindly is destructive. On a
50-window Orbitrap DIA run (32,950 MS2 and 660 MS1 spectra) the uncapped peak
count per MS2 spectrum is p25 572, p50 1,320, p95 2,756, max 3,596, for 43.4M
peaks in total. A cap of 300 keeps 9.3M of them: 78.6 percent of all MS2 peaks
are discarded and 85.5 percent of spectra are truncated, because even the 25th
percentile spectrum exceeds the cap. End-to-end on that file, with only this flag
changed:

| conversion cap | `peptides.tsv` rows @ `peptide_q_value` <= 0.01 | protein groups | share of DIA-NN 2.2.0 library-free |
|---|---|---|---|
| 300 | 25,425 | 4,554 | 32.3% |
| uncapped | **63,237** | **7,336** | 80.3% |

The empirical decoy fraction was 0.99 percent in both arms, so this is a
sensitivity difference, not a loosened threshold.

The mechanism is peak-group formation, not scoring. With most peaks removed,
`extract.presence_min_fragments = 3`
(`rust/mumdia/crates/mumdia-core/src/config.rs:523`, default at `:690`) cannot be
satisfied, so real peptides never form a peak group. `mumdia audit` on the capped
arm, restricted to peptides DIA-NN confirms are present, shows 49,105 of 78,782
(62.3 percent) stopped at `candidate_generated` with `NO_PEAK_GROUP`, against
only 5,380 lost to FDR and 355 lost to competition. A counterfactual replay on
the uncapped artifact recovered 41,948 of those 49,105 (85.4 percent).

The response to the cap is smooth, so a cap remains usable when peak volume is
constrained. Fraction of the lost peptides recovered on that file:

| cap | 300 | 400 | 600 | 900 | 1400 | 2000 | uncapped |
|---|---|---|---|---|---|---|---|
| recovered | 0% | 17.8% | 44.3% | 69.0% | 83.1% | 86.8% | 87.2% |

A cap of 1,400 buys 83 of the 87 achievable percentage points at about 77 percent
of the peak volume. Uncapping is not free but is cheap on that file: accepted candidates
rose from 188,027 to 2,286,840 (12.2x) and extract wall clock from 57.9 s to
91.9 s.

Both `run --top-peaks-ms2` and standalone `convert --top-peaks-ms2` default to 0
(uncapped), which is the correct default. Reproducing the AIF result requires
passing `--top-peaks-ms2 300` explicitly, and that flag must not be carried to
another acquisition without the per-acquisition pre-flight and cap sweep in
docs/20.

### A4. DeepLC fine-tune of iRT is essential in library mode, but not per file

Raw DIA-NN predicted iRT is locally noisy, with a residual around 110 seconds MAD
against observed RT on both the E. coli and HYE datasets. A DeepLC 4.0 multitask
fine-tune on confident seed PSMs tightens this to roughly 13 to 27 seconds MAD,
which narrows the RT windows and materially lifts identifications. Fine-tuning is
therefore essential, not optional, in library-input mode. Do not use an
already-fine-tuned library (a `_ft` or reversed-decoy library) as a "raw DIA-NN"
baseline; the raw baseline is the un-fine-tuned precursor library.

Addendum 2026-09-05 (docs/08 section 4c): most of that gain is available without a
fine-tune. Re-predicting the imported iRT with the DeepLC 4.1.1 base model
(`rt_im_train.library_irt = auto`, the default) is +4.0% peptides over the raw iRT on
HYE B01 (NN seeds 1-3) and +4.0% on AIF; the once-per-library fine-tune is a further
+2.4% on HYE (18.6k anchors) and -2.3% on AIF (5.6k anchors). The essential step is
therefore DeepLC 4.1.1 predictions, with the fine-tune as the recommended extra on a
large reference. Single-seed comparisons of these arms misled by two points because NN
seed 0 depressed two of the three arms; compare with seeds.

What is not established is that the fine-tune must be repeated for every file.
Measured: a library whose `predicted_irt` was fine-tuned once and then predicted
across every peptidoform in that library, combined with the ordinary per-run
LOESS calibration in `rt-im-train` and `rt_im_train.finetune_deeplc = false`,
gives median absolute RT residual 6.06 s, MAD 6.11 s, slope 0.9907, intercept
16.4 s, against 6.14 s and 6.18 s with per-file fine-tuning. That is equal or
marginally better, and it removes about 36 minutes per file (2,166 s of a 5,127 s
single-file run, covering the fine-tune plus whole-library iRT prediction over
roughly 5M peptidoforms). The distinction that matters is between a library
fine-tuned once and re-predicted in full, which is what was measured, and a stale
per-run `_ft` table produced on one file and reused on another, which has
previously underperformed a fresh per-run fit. Per-run fine-tuning remains the
default in the AIF reference recipe; on a multi-file experiment against a fixed
library, the once-per-library variant is the cheaper equal-quality option.

The fine-tune runs between search-seed and RT calibration and writes a new
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

Pooling more runs into one rescore does not tighten q. That estimator is
scale-invariant: replicating the population leaves the decoy-to-target ratio
unchanged. The only pool-size-dependent term is the `+1` pseudocount, whose
relative weight shrinks as the pool grows, so a larger pool is if anything
marginally looser at a given score. Do not attribute a per-run count change to
the number of runs pooled; the cause is the score distribution or the q column
used, not the pool size.

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

### Session findings, 2026-07-24 (provisional)

Findings A8 through A12 are from one session (commit `f8f6c08`, on top of the
previously synced `2cbbc20`) and carry shared caveats. They rest on a single
dataset (`LFQ_Orbitrap_AIF_Ecoli_01`), on decoy-based FDR only (not
entrapment-validated), and use the nondeterministic `nn_torch` rescorer. Any
default change they motivate remains benchmark-gated per repository policy.
Treat them as provisional signals below the validation bar of A1 through A7.

### A8. N-terminal Met-excision closes the database-completeness gap

The native digest now models cotranslational N-terminal methionine excision. A
protein's initiator methionine is frequently cleaved in vivo, so a search
database built without the Met-removed form structurally misses those peptides.
`DigestConfig` gained `n_term_met_excision: bool` (default `true`,
`rust/mumdia/crates/mumdia-core/src/config.rs:194`, matching DIA-NN
`--met-excision`). In `digest_protein`, for a peptide anchored at protein
position 0 whose first residue is M, the Met-removed form is also emitted (start
shifted to 1) and re-checked against min_len/max_len and the standard-residue
rule (`rust/mumdia/crates/mumdia/src/stages/digest.rs:101` through `:110`).
Excision keys on protein position 0, not any leading M. The struct carries
`#[serde(default, deny_unknown_fields)]` so existing configs still parse. Two
unit tests fix the behavior: `met_excision_emits_both_n_term_forms` and
`met_excision_only_at_protein_n_terminus` (interior M is not excised,
`rust/mumdia/crates/mumdia/src/stages/digest.rs:384`, `:418`).

On this benchmark every missed peptide absent from the search database (209 of
them) was a Met-excision peptide. Augmenting the imported DIA-NN library with the
missing FASTA peptides through the new `scripts/augment_library.py` helper, which
reuses `mumdia digest` (Met-excision on) and `mumdia predict-frag` so peptidoform
strings are byte-identical, drove the not-in-database count from 209 to 0.
Met-excision is standard-proteomics-correct, but because it changes native-digest
output it remains entrapment-plus-second-dataset gated before it is trusted as a
default.

### A9. The missed-peptide funnel after augmentation is a downstream, faint-signal ceiling

With the database-completeness gap closed, the remaining missed peptides are
downstream and faint, about 10x lower in abundance than identified peptides.
Partitioning the misses by the stage that dropped them: extraction presence/apex
accounts for about 49 percent (dominated by the presence_min_matched and
minimum-co-elution checks, not the `min_frag_corr` gate), rescoring about 26
percent, and seed search about 25 percent. The ceiling is therefore not one gate;
it is split across extraction presence requirements, rescorer discrimination, and
seed sensitivity, all on low-abundance precursors.

### A10. The nn_torch rescorer is converged, not training-limited

An epoch and round sweep of `nn_torch` on the augmented candidate pool shows the
rescorer is feature-limited, not training-limited. 10 epochs undertrains (about
150 to 180 fewer peptides); 25 epochs at 10 rounds is the knee and the default;
50 epochs at 20 rounds gives no real gain (+5 peptides, within noise).
Separately, removing the extraction presence/apex filters wholesale was a wash
(+96 peptides net, roughly one-to-one churn) even though the candidate pool
flooded about 18x, and decoy-based FDR stayed valid at 0.98 percent throughout.
The lever is therefore better features or empirical-library spectra, not more
training epochs or an opened gate.

The `nn_torch` training loop is seed x fold x round x epoch. The knobs:
`rescore.folds` maps to `MUMDIA_NN_FOLDS`, `rescore.num_iter` maps to
`MUMDIA_NN_ITERS` (rounds), `MUMDIA_NN_EPOCHS` is an environment variable only
(not engine-set, default 25), and `MUMDIA_NN_SEEDS` defaults to 1.

### A11. `compete.group_by = precursor` is a misnomer and deletes modforms

The enum variant is named `Precursor`, but the group key it builds is
`(base_peptide_id, label_code, 0, peak_rank)`
(`rust/mumdia/crates/mumdia/src/stages/compete.rs:88`), and `base_peptide_id` is
the stripped sequence: `scripts/import_diann_lib.py:137` factorizes
`Stripped.Sequence` into it. `resolve_competition` then keeps only the highest
`prelim` row per group and deletes the rest
(`rust/mumdia/crates/mumdia/src/stages/compete.rs:321` through `:344`), before
rescoring. Every charge and every modification variant of one peptide therefore
collapses to a single winner pre-FDR, and the default is not a precursor-level
count.

`group_by = peptidoform_charge` keys `(peptidoform_id, label_code, charge,
peak_rank)` (`rust/mumdia/crates/mumdia/src/stages/compete.rs:93` through `:98`)
and is the true precursor-level competition. Measured on a modification-rich
imported library, the default deleted 880,464 of 1,890,239 extracted candidates
(46.6 percent). Under `peptidoform_charge`, competition removed 0 rows and
precursors per peptide moved from 1.000 to 1.174 (DIA-NN reports about 1.126 on
comparable data). The peptide count was unchanged, so the precursor-level key
cost nothing on that library.

The consequence for a modification search is not cosmetic. The modified form is
deleted whenever an unmodified or otherwise-modified sibling of the same stripped
sequence scores higher, which is the common case. Treat `peptidoform_charge` as
required for PTM work rather than benchmark-gated, and required for any
precursor-level identification count or precursor matrix submission. Changing the
key still changes the training and FDR population, so for base-peptide
sensitivity benchmarks the default remains the comparison baseline (see finding
A5 on q-value units and the quantification rules in CLAUDE.md).

### A12. Comparison against DIA-NN 2.2.0

Against DIA-NN 2.2.0 run library-free from the same E. coli FASTA with a matched
search space, both at 1 percent FDR, MuMDIA reached about 90 to 92 percent of the
DIA-NN peptide count, about 89 to 91 percent of precursors (peptidoform plus
charge), and about 99 to 101 percent of protein groups. The comparison is not
symmetric: DIA-NN ran its full library-free double pass, while MuMDIA ran a
single library-based pass on a library derived from DIA-NN. Protein-group parity
alongside a peptide and precursor gap is consistent with MuMDIA being
high-precision and recall-limited on faint precursors (finding A9).

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
2. Make sure the library iRT is DeepLC fine-tuned. The reference recipe does this
   per run (`rt_im_train.finetune_deeplc = true`). Rationale: finding A4, raw iRT
   is too noisy. On a multi-file experiment against a fixed library, fine-tuning
   the library once and predicting it in full, then leaving
   `finetune_deeplc = false`, matched the per-file residuals and saved about 36
   minutes per file (finding A4).
3. Use the Extended feature set (`features.set = extended`).
4. Use the `nn_torch` rescorer (`rescore.classifier = nn_torch`, with
   `rescore.python` pointing at a torch-capable interpreter and
   `rescore.strict = true`). Rationale: finding A1, the rescorer is the dominant
   lever; strict mode prevents a failed sidecar from masquerading as the intended
   model. Verify `params.classifier` in `psms_scored.parquet.report.json`.
5. Use a loose extraction gate (`extract.min_frag_corr` near 0.2). Rationale:
   finding A1 and A2, the nonlinear rescorer prefers looser-is-better.
6. On this AIF file only, set the conversion-time MS2 peak cap to 300 by passing
   `--top-peaks-ms2 300`. Rationale: finding A3, uncapping adds interference on
   chimeric data. The seed-only `search_seed.top_n_peaks=300` is separate. On any
   other acquisition, leave the cap at the default 0 until a per-acquisition
   sweep says otherwise; finding A3 measures a 300 cap costing 60 percent of the
   peptides on a 50-window Orbitrap DIA run.

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
