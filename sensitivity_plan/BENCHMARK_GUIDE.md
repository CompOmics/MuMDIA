# Sensitivity Benchmark Guide

How to run the diagnostics added by the sensitivity program and how to read them.
All tools are non-destructive: they read existing artifacts and change no engine
output. Python tools use the `py312_mumdia` conda interpreter (pyarrow, pandas,
numpy, scikit-learn).

Paths below use the E. coli / HYE example in `C:/proteobench/out_ecoli/`; substitute
your own run directory.

## 1. Candidate audit (identification-loss waterfall)

Reconstructs, per candidate, the pipeline stage flags and the EARLIEST rejection
reason from the artifact chain (library -> psms -> competed -> scored).

```
mumdia audit \
  --library-precursors lib/lib_precursors_ft.parquet \
  --psms      out_ecoli/psms.parquet \
  --competed  out_ecoli/comp.parquet \
  --scored    out_ecoli/scored.parquet \
  --out       out_ecoli/candidate_audit.parquet \
  --q 0.01 --run-id ecoli --entrapment-substr _HUMAN
```

Or automatically as part of a full run: set `extract.emit_candidate_audit = true`
in the config; `mumdia run` then writes `candidate_audit.parquet` after rescore.

Outputs:
- `candidate_audit.parquet`: one row per candidate with `run_id, precursor_id,
  modified_sequence, charge, target_decoy_label, entrapment_label,
  candidate_generated, traces_extracted, peak_generated, peak_selected,
  variant_selected, target_decoy_winner, passed_precursor_fdr, passed_peptide_fdr,
  reported, rejection_reason`.
- `<out>.metrics.json`: the waterfall counts + stage recalls.

Example waterfall (E. coli file vs the 8.3M-candidate HYE library):

```
search_space = 8,334,126   extracted = 341,754   reported = 8,568
NO_PEAK_GROUP = 7,992,372   FAILED_PRECURSOR_FDR = 332,120
FAILED_PEPTIDE_FDR = 1,066   REPORTED = 8,568
```

Reading it (spec 05 §5 decision rules):
- Large `NO_PEAK_GROUP` with a cross-species library is expected (most library
  peptides are absent from a single-species sample). To make it actionable, restrict
  the audit input library to candidates a reference tool (DIA-NN) identifies, or
  stratify by `entrapment_label`.
- Large `FAILED_PRECURSOR_FDR` among candidates that DID extract points at scoring /
  decoy / FDR (prioritize rescoring, decoy design). Large `OUTCOMPETED_*` points at
  competition (try `compete.mode = none`). Large extraction loss among candidates
  that should be present points at peak selection (top-K) or calibration.

Limitation: at artifact resolution, extraction losses collapse to `NO_PEAK_GROUP`.
The in-extract audit sidecar (NEXT_STEPS #2) refines them to `NO_FRAGMENT_TRACES` /
`NO_VALID_FRAGMENTS` / `PEAK_NOT_SELECTED` / `RT_PRUNED`; `mumdia audit` already
reads `<psms>.audit.parquet` when present.

## 2. Reference-apex top-K peak recall

Quantifies the top-K peak opportunity: how often the selected apex is the strongest
peak, and (with a DIA-NN report) whether the reference apex is within the top-K
MuMDIA peaks.

```
python scripts/reference_apex_topk.py \
  --psms  out_ecoli/psms.parquet \
  --chrom out_ecoli/chrom.parquet \
  [--diann diann_report.tsv] \
  [--out out_ecoli/topk_metrics.json] \
  [--max-candidates 20000] [--bound-fraction 0.333] [--rt-tol-s 10]
```

Self analysis (no DIA-NN needed) on 20,000 E. coli candidates:

| metric | value |
|---|---|
| mean peaks / candidate | 10.35 (median 9) |
| candidates with >= 2 peaks | 95.2% |
| selected apex is peak rank-1 | 52.5% |
| selected apex in top-3 | 79.2% |
| selected apex in top-5 | 88.3% |
| selected apex in top-10 | 94.8% |
| selected apex in no enumerated peak | 3.5% |

Reading it: the selected apex is the strongest peak only about half the time, and a
correct alternative peak exists within the top 5 for ~88% of candidates. This is the
quantitative case for `extract.retain_top_peaks = 5` (NEXT_STEPS #1): retain the
alternative peaks and let a peak-selection model choose, instead of committing to one
apex. Run this before and after the top-K wiring to measure recovered peak recall.

With `--diann`: reports `reference_apex_in_top_{1,3,5,10}` by matching stripped
sequence + charge and comparing the DIA-NN apex RT (auto-converted minutes->seconds)
to the MuMDIA peak apexes. This is the spec's peak-oracle metric (02 §6).

## 3. Feature-family ablation

Grouped cross-validated ablation with leakage guards (in-fold standardization,
group by peptidoform+charge, targets at empirical FDP).

```
python scripts/feature_ablation.py \
  --features out_ecoli/comp.parquet \
  --registry feature_registry.yaml \
  --out out_ecoli/ablation \
  [--folds 3] [--fdp 0.01] [--model both] [--max-rows 60000] [--clip-sd 8.0]
```

Outputs a CSV + JSON per model with columns `feature_family, baseline_identifications,
new_identifications, relative_gain, delta_vs_full, model, recommendation`.

Smoke (60,000 rows, logreg, 3 folds): 355 features / 17 families; full model = 509,
minimal baseline = 414 targets @1% FDP. Most useful families (removal hurts):
`similarity` (-393), `rt` (-382), `entropy` (-6). On this subset removing
`interference` / `rich` / `coelution` raised the count (+267 / +204 / +119),
indicating redundancy or subset overfit.

Caveats (spec 03 §9, 05 §7): this is ONE dataset subset and (in the smoke) ONE model.
Do not retain or drop a family on a single favourable subset. Rerun with `--model
both`, all rows, and a second dataset before acting; a family that only helps on one
subset or whose gain flips sign across models/datasets is not a keep.

## 4. Empirical entrapment FDP (the acceptance gate)

Already present. Trains on E. coli targets vs a human-TRAIN half and evaluates on the
unseen human-TEST half, giving a leakage-free identification count at a genuinely
controlled FDP.

```
python scripts/entrapment_holdout.py out_ecoli/comp.parquet --q 0.01
```

Use this as the accept/reject gate for every change (spec 05 §6): a change ships only
if held-out entrapment identifications rise without FDP inflation, reproduced on a
second dataset.

## 5. End-to-end recipe for one experiment (spec 05)

1. `mumdia run ... --config <cfg>` (set `extract.emit_candidate_audit=true`).
2. `mumdia audit ...` (or read the run's `candidate_audit.parquet`) -> waterfall.
3. `python scripts/reference_apex_topk.py ...` -> peak recall.
4. `python scripts/feature_ablation.py ...` -> family contributions.
5. `python scripts/entrapment_holdout.py ...` -> honest FDP + identification count.
6. Change ONE component (e.g. `compete.mode`, `retain_top_peaks`, a feature family),
   rerun 1-5, and compare at matched empirical FDP. Keep raw candidate outputs.

## Determinism and cost

- All Rust stages are deterministic under a fixed seed; `mumdia audit` is a pure join.
- The Python tools seed RNG and use stable (SHA1) fold hashing.
- `reference_apex_topk.py` and `feature_ablation.py` support `--max-candidates` /
  `--max-rows` for fast passes; full runs over ~1.3M rows are heavier (minutes).
