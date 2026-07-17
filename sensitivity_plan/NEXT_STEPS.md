# Sensitivity Program — Next Steps

Prioritized, with exact hook sites (from `ARCHITECTURE_MAP.md`). Items are ordered
by expected sensitivity value per unit risk. Everything below builds on the
`feat/sensitivity-improvements` branch.

## 1. Wire top-K peak retention into extraction (highest value, highest risk)

The enumerator (`mumdia::peaks::enumerate_peaks`), config (`ExtractConfig.retain_top_peaks`),
and tests exist; extraction still emits one apex per candidate.

- **Hook:** `extract.rs:718` (single-argmax apex loop) and `CandOut` (`extract.rs:569`),
  which today carries one apex + one `Vec<chrom>`. Multiply it to `Vec<CandOut>` (one
  per retained peak) when `retain_top_peaks > 1`.
- **Approach:** after the scan-group build (`extract.rs:608`) and rolling count
  (`extract.rs:681`), build the signature-ion summed profile over the grid and call
  `enumerate_peaks(profile, k, bound_peak_fraction, prominence)`. For each returned
  `PeakGroup`, run the existing apex/chrom emission restricted to `[start_idx, end_idx]`,
  stamping a new `peak_rank` column on `psms_extracted` and `chromatograms`.
- **Compatibility:** `K == 1` must bypass the enumerator and keep the current path
  (a regression test must show byte-identical `psms.parquet` for K=1 on a fixed input).
- **Downstream:** the features stage and `compete` key currently assume one row per
  `candidate_id`. Add `peak_rank` to the competition key or add a peak-selection pass
  (below) so multiple peaks of one candidate do not all survive to the report.
- **Validation:** run `scripts/reference_apex_topk.py` before/after; the SELF top-K
  distribution predicts the achievable gain. Confirm entrapment FDP is not inflated.

## 2. In-extract candidate-audit emitter (precise extract-stage reasons)

`mumdia audit` already reconstructs the ladder from artifacts, but extraction losses
collapse to `NO_PEAK_GROUP`. Emit the precise reason per candidate.

- **Hook:** the per-candidate cascade at `extract.rs:566` returns `Option<CandOut>`;
  every `return None` (lines 600, 750-753, 808) is a mapped reason (see the drop
  table in `ARCHITECTURE_MAP.md` §3). Change it to return
  `Result<CandOut, RejectionReason>` (or a small `enum CandEval`), collect via rayon,
  and when `extract.emit_candidate_audit` is set write `<out-psms>.audit.parquet`
  (candidate_id, rejection_reason). Also diff the library candidate range against the
  accumulator keys (`extract.rs:302`) to emit `NO_FRAGMENT_TRACES` for never-materialized
  candidates.
- `stages::audit::load_extract_reasons` already reads this sidecar and refines the
  waterfall, so no audit-stage change is needed.
- **Cost guard:** only allocate the audit vector when the flag is set.

## 3. Competition after an initial rescoring pass (spec 04 §11)

`compete` runs before `rescore` on the heuristic `prelim_score` (`run.rs:243` before
`:252`), so a candidate a trained model would keep can be removed early.

- **Interim (already available):** run with `compete.mode = none` (or `features_only`)
  so competition removes nothing and the rescorer arbitrates. Benchmark this against
  `winner_take_all` at matched empirical FDP (experiment E14).
- **Full:** add a first-pass rescore (native `percolator_lite` is cheap) that writes an
  out-of-fold score, then run `compete` on that score instead of `prelim_score`. Keep
  the fold grouping by peptidoform+charge to avoid leakage.

## 4. Fragment claimant / conflict-graph features (spec 04 §5, P2.1-2.3)

The nucleus exists: the per-peak `claimants` buffer (`extract.rs:304`) and the
two-pass `contested` map (`extract.rs:308`, feature `contested_frac` at
`extract.rs:814`). Extend to a candidate-level family:

- `claimant_count` / `contested_fragment_count`, `unique_fragment_count`,
  `unique_intensity_fraction`, `shared_trace_correlation`, `conflict_group_size`,
  `strongest_competitor_score`, `score_margin`.
- **Hook:** compute in the two-pass arbitration (`extract.rs:456-511`) and emit as new
  `psms_extracted` columns, or as a new `stages/features/*.rs` family reading the
  chromatogram overlaps. Register in `feature_registry.yaml`. Keep target/decoy
  computation symmetric (audit per spec 04 §9).
- These directly feed `compete.mode = unique_evidence`, which currently approximates
  unique evidence from `n_matched_fragments * (1 - contested_frac)`.

## 5. New feature families with existing data (spec 03 §8, P5)

Lowest risk, additive. Priorities by data availability (see `FEATURE_REGISTRY.md`
gap table):

- **Uncertainty-normalized residuals** (P5.1): `abs(rt_residual)/local_rt_sigma`,
  `abs(mass_error)/local_mz_sigma`. Needs local uncertainty from calibration (item 6).
- **Apex dispersion** (P5.3): fragment-apex RT stddev/MAD, precursor-fragment apex
  delta. Data already in the chromatograms; compute in a new features family.
- **Candidate ambiguity** (P5.5): margins to alternative peaks/peptides using
  `prelim_score` (an earlier-stage score, not the final model, to avoid circularity).

## 6. Two-pass calibration + local uncertainty (spec 03 §5, P3)

`rt_im_train` (`rt_im_train.rs`) fits per-run RT calibration; mass calibration is in
`search_seed` (`masscal.json`). Add: robust two-pass precursor+fragment mass
calibration, a monotonic nonlinear RT map, and a LOCAL uncertainty estimate exported
per region. The uncertainty unlocks item 5's normalized residuals and adaptive
extraction windows (`window = max(min, scale * local_sigma)`), spec P3.3.

## 7. Empirical-FDP-first evaluation loop

`scripts/entrapment_holdout.py` already gives a leakage-free held-out entrapment count
on an unseen human null. Make it the acceptance gate for every change above (spec 05
§6): a change ships only if held-out entrapment identifications rise without FDP
inflation, reproduced on a second dataset. The single E. coli/HYE file here is one
dataset; a second (the TTOF SWATH file, or a ProteoBench HYE run) is needed for the
spec's held-out reproduction criterion.

## Cannot be completed in this environment

- Held-out reproduction on a second dataset (needs a second labelled run loaded).
- Reference-apex recall vs DIA-NN (needs a DIA-NN report; `scripts/reference_apex_topk.py`
  computes it once `--diann` is supplied).
- Ion-mobility / diaPASEF families (no 4D data).
