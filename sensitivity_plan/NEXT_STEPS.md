# Sensitivity Program - Next Steps

Prioritized, with exact hook sites (from `ARCHITECTURE_MAP.md`). Items are ordered
by expected sensitivity value per unit risk. Everything below builds on the
`feat/sensitivity-improvements` branch. The large scaffolding is now in place:
top-K peak retention, competition modes, two feature families, two-pass mass
calibration, and the adaptive RT window are all implemented and default-off. What
remains is closing loops and validating each knob, not building new plumbing.

## 1. Close the top-K peak loop (highest value, highest risk)

Peak RETENTION is done: `extract.retain_top_peaks > 1` enumerates top-K peak groups
with `peaks::enumerate_peaks` and writes `<psms>.peaks.parquet`
(`extract.rs:927-929`, `:1068`; 7 cols `candidate_id, peak_rank, apex_rt, start_rt,
end_rt, evidence_count, area`). Validated K=5 on E. coli = 1,699,995 peaks
(mean ~4.97/candidate) with the scored path unchanged. The offline peak-selection
prototype `scripts/peak_selection_model.py` learns to rank the retained peaks
(E. coli weak-self-label: learned top1 0.426 / top3 0.802; evidence-rank 0.418 beats
area-rank 0.390). What is NOT done: the engine still reports the single heuristic
apex; the retained peaks are a diagnostic sidecar, not fed back into scoring.

- **Remaining hook (close the loop):** the retained peaks carry only coarse
  descriptors today (apex/start/end RT, evidence count, area). To re-select a peak
  before rescore, each retained `PeakGroup` needs the FULL per-peak feature vector.
  Emit one `psms_extracted` + `chromatograms` row per `(candidate_id, peak_rank)`
  (peak-restricted apex/chrom emission from `CandOut`, `extract.rs:569`), stamp a
  `peak_rank` column, and let `features.rs` compute per-peak features (chrom grouping
  must key by `(candidate_id, peak_rank)`).
- **Re-selection:** run the peak-selection ranker (port `peak_selection_model.py`)
  or a first native pass, keep the winning peak, then collapse to one row before
  `compete` (`compete.rs:72` key) and `rescore` (peptide/protein grouping) so
  multiple peaks of one candidate do not all reach the report.
- **Compatibility:** `K == 1` must keep the current byte-identical path (regression
  test on a fixed input).
- **Validation:** `scripts/reference_apex_topk.py` before/after; confirm the
  entrapment FDP is not inflated (item 7).

## 2. In-core wiring of the conflict / localization sidecar features

The conflict graph and localization competition exist as non-invasive Python
sidecars: `scripts/conflict_features.py` -> `conflict.parquet` (fragment-claimant
index + peak-group conflict graph + contested/unique/ambiguity features, joinable by
`candidate_id`) and `scripts/localization.py` -> `localization.parquet`
(localization-variant grouping + site-determining ions; 0 ambiguity groups on
E. coli, as expected). They are not yet in the rescorer feature schema.

- **Hook:** add the cross-candidate pass inside `features.rs` (precedent: the
  cross_charge extended-extras block), add the new fields to `Evidence`
  (`features.rs:269`), and register a new family module following the
  `NAMES` + `values(&Evidence)` contract appended to `FAMILIES` (`features.rs:49`);
  dedup / schema-id / PIN flow then handle it automatically.
- **Schema:** bump the family count and update the `feature_sets_sized` test
  (`features.rs:1263`) and `feature_registry.yaml`.
- These directly feed `compete.mode = unique_evidence`, which currently approximates
  unique evidence from the existing features.

## 3. P6.3 staged modification search

Not started. A calibration stage (unmodified / common-mod search) followed by an
extended-mod stage that opens the search space only where the calibration stage
found evidence.

- **Prerequisite:** site-determining-ion scoring; `scripts/localization.py` is the
  offline prototype for the localization half.
- **Hook:** a second `peptidoforms` -> `predict-frag` -> `extract` pass gated on the
  first pass's confident precursors; needs a new orchestration branch in `run.rs`
  and a config flag. This is genuinely new plumbing (unlike the items above) and is
  the largest remaining feature.

## 4. Entrapment-gate validation of every default-off knob on >=2 datasets

Every knob shipped this program is OFF by default because none has passed the
acceptance gate. This is the next ACTION, not more code.

- **Knobs to validate:** `extract.retain_top_peaks` (once item 1 closes the loop),
  `extract.apex_evidence_rank`, `rt_im_train.adaptive_rt_window`,
  `search_seed.two_pass_mass_cal`, and `compete.mode`
  (`none`/`features_only`/`unique_evidence`/`margin_gated`).
- **Gate:** `scripts/entrapment_holdout.py` gives a leakage-free held-out entrapment
  count on an unseen human null. A knob ships as a default only if held-out
  entrapment identifications rise WITHOUT FDP inflation, reproduced on a SECOND
  dataset (spec 05 §6).
- **Blocker:** only one valid dataset is loaded here (the E. coli / HYE file). The
  on-disk SWATH TTOF file is NOT a valid second dataset: it produced 0 IDs against
  the Ox HYE library (the sample does not match the library). A genuine second
  labelled run (a matching TTOF library, or a ProteoBench HYE run) is required.

## 5. In-extract candidate-audit emitter (precise extract-stage reasons)

`emit_candidate_audit` makes `run` write `candidate_audit.parquet` after rescore, but
extraction losses still collapse to `NO_PEAK_GROUP` at artifact resolution.

- **Hook:** the per-candidate cascade at `extract.rs:566` returns `Option<CandOut>`;
  every `return None` (the mapped reasons in `ARCHITECTURE_MAP.md` §3) plus the
  never-materialized cohort (library candidate range minus accumulator keys,
  `extract.rs:302`) should write a `<psms>.audit.parquet` (candidate_id,
  rejection_reason). `stages::audit::load_extract_reasons` already reads that sidecar
  and refines the waterfall, so no audit-stage change is needed.
- **Cost guard:** only allocate the audit vector when the flag is set.

## 6. Competition after an initial rescoring pass (spec 04 §11)

`compete` runs before `rescore` on the heuristic `prelim_score` (`run.rs` order), so
a candidate a trained model would keep can be removed early. The modes are wired;
the reordering is not.

- **Interim (available now):** run with `compete.mode = none` (or `features_only`) so
  competition removes nothing and the rescorer arbitrates. Benchmark against
  `winner_take_all` at matched empirical FDP (validate per item 4).
- **Full:** add a first-pass native `percolator_lite` rescore that writes an
  out-of-fold score, then run `compete` on that score instead of `prelim_score`. Keep
  the fold grouping by peptidoform+charge to avoid leakage.

## 7. Local calibration uncertainty (finish spec 03 §5 / P3)

Two-pass mass calibration (`search_seed.two_pass_mass_cal`) and the adaptive RT
window (`rt_im_train.adaptive_rt_window`) are done. The remaining piece is exporting a
LOCAL per-region uncertainty estimate so features can normalize residuals
(`abs(rt_residual)/local_rt_sigma`, `abs(mass_error)/local_mz_sigma`).

- **Hook:** capture the residual distribution into `cal.json` (`rt_im_train.rs`) and
  `masscal.json` (`search_seed.rs`); add a `pred_sigma` field to `Evidence`
  (`features.rs:269`) fed by a new library/chromatogram column. The `mass_uncertainty`
  and `apex_dispersion` families are the consumers already in place.

## Cannot be completed in this environment

- Held-out reproduction on a second dataset (needs a valid second labelled run; the
  on-disk TTOF file is the wrong sample for the Ox HYE library).
- Reference-apex recall vs DIA-NN (needs a DIA-NN report; `scripts/reference_apex_topk.py`
  and `scripts/peak_selection_model.py` compute it once `--diann` is supplied).
- Ion-mobility / diaPASEF families (no 4D data).
