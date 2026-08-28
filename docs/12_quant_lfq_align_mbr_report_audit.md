# quant, quant-lfq, align, mbr, report, audit

> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

This document covers the six "tail" subcommands of the pipeline: the ones that run
after `rescore` has produced `psms_scored.parquet`, plus the two experiment-level
stages that operate across multiple runs.

- **quant** (Stage G, `mumdia quant`): integrate per-fragment chromatograms over the
  elution peak, sum the top-N fragments into a per-peptidoform quantity, and roll up
  to protein groups. Single-run.
- **quant-lfq** (Stage G, `mumdia quant-lfq`): combine several per-run quant tables
  into a protein-by-run abundance matrix by MaxLFQ (peptide-level) or directLFQ
  (ion/fragment-level), with optional cross-run median-ratio normalization.
- **align** (Stage D2, `mumdia align`): put >=2 runs on a common RT coordinate by
  fitting a reference LOESS RT map per run and recording the residual spread.
- **mbr** (Stage D3, `mumdia mbr`): partially wired match-between-runs identification
  transfer. A Rust CLI + config gate that shells out to `scripts/mbr_worker.py`.
- **report** (`mumdia report`): emit human-readable `peptides.tsv` + `proteins.tsv`
  from the scored PSM table, joined to quant. Invoked by the single-run `run`
  chain and by the CLI subcommand only; `run-experiment` never runs it.
- **audit** (`mumdia audit`): reconstruct, per candidate, the pipeline stage flags and
  the earliest rejection reason across the artifact chain, without re-running compute.

## Files

| path | role |
|---|---|
| `rust/mumdia/crates/mumdia/src/stages/quant.rs` | `quant` stage + `run_lfq_combine`/`size_factors` for `quant-lfq` |
| `rust/mumdia/crates/mumdia/src/quant_lfq.rs` | MaxLFQ/directLFQ ratio-alignment core (`lfq_profile`, `maxlfq`, `directlfq`) |
| `rust/mumdia/crates/mumdia/src/stages/align.rs` | Stage D2 cross-run RT alignment |
| `rust/mumdia/crates/mumdia/src/stages/report.rs` | `peptides.tsv` + `proteins.tsv` writer |
| `rust/mumdia/crates/mumdia/src/stages/audit.rs` | candidate identification-loss ladder |
| `rust/mumdia/crates/mumdia-core/src/rejection.rs` | `RejectionReason` enum + ladder ordering |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `QuantConfig`, `MbrConfig`, `RtImTrainConfig`, and the strategy enums |
| `scripts/mbr_worker.py` | MBR transfer sidecar (rescuable + re-extraction tiers) |
| `rust/mumdia/crates/mumdia/src/main.rs` | CLI definitions + handlers for all six subcommands |
| `rust/mumdia/crates/mumdia/src/sidecar.rs` | `run_mbr` (builds argv for `mbr_worker.py`) |

## Inputs and outputs

### quant
Consumes `psms_scored.parquet` (from `rescore`) and `chromatograms.parquet` (from
`extract`). Produces two mandatory artifacts and two optional ones.

`peptide_quant.parquet` (schema `PEPTIDE_QUANT` v2):

| column | type | meaning |
|---|---|---|
| `candidate_id` | u32 | library candidate id |
| `base_peptide_id` | u32 | stripped/base-peptide key used to deduplicate protein rollup |
| `peptidoform` | str | ProForma peptidoform |
| `charge` | i32 | precursor charge |
| `protein_group` | str | protein-group key |
| `quantity` | nullable f64 | sum of the top-N positive finite fragment areas; null when not quantifiable |
| `quant_status` | str | `quantified` or the explicit reason quantity is missing |
| `n_fragments_used` | i32 | number of positive finite fragments actually summed |
| `integration_apex_rt` | nullable f64 | apex actually used for integration |
| `integration_lo_rt` / `integration_hi_rt` | nullable f64 | integration bounds actually applied |

`protein_group_quant.parquet` (schema `PROTEIN_GROUP_QUANT` v2):
`protein_group` (str), nullable `quantity` (f64), `quant_status` (str), and
`n_peptides` (i32, number of unique positive base peptides before Top-N
truncation). Charge/modification siblings contribute only their maximum
single-run quantity to the base-peptide representative.

`fragment_quant.parquet` (optional, `--out-fragment`; `quant.rs:558`): one row per
fragment area, `candidate_id`, `peptidoform`, `charge`, `protein_group`,
`fragment_name` (str), `quantity` (f64). Only positive finite fragment areas are
emitted. This is the input for ion-level directLFQ.

`<peak_bounds>.parquet` (optional diagnostic, `--out-peak-bounds`, only when
`bound_peak` is on; `quant.rs:419`): `candidate_id`, `lo_rt`, `hi_rt`, `width_s`
(all f64). Not part of the quant contract; it is a view of the integration windows.
A row is emitted only for a candidate whose chosen `(lo_rt, hi_rt)` are both finite
(`quant.rs:399`), so unbounded/degenerate windows are absent.

`peptide_quant`, `protein_group_quant`, and an emitted `fragment_quant` receive a
sidecar `<artifact>.report.json`; the peak-bounds diagnostic does not. Recorded
params include the q filter, integration settings, config hash, and exact scored
and chromatogram inputs. Stats distinguish row counts from quantified and
nonquantifiable rows.

### quant-lfq
Consumes N per-run tables (peptide_quant for maxlfq, fragment_quant for directlfq).
Produces a long-form matrix (`quant.rs:720`): `protein_group` (str), `run` (i32,
0-based input index), `quantity` (f64), `n_features` (i32). `n_features` is the
count of feature keys for that protein group, constant across its rows (`quant.rs:717`),
not the per-run non-missing count. One row per `(protein_group, run)` is written for
every run, including runs where the protein's profile is 0. Two sibling matrices are
also written next to `<out>`: `<out>.peptide.parquet` (rolled up by stripped base
sequence) and `<out>.precursor.parquet` (by peptidoform + charge), each with columns
`group` (str), `charge` (i32; -1 for peptide level), `run`, `quantity`, `n_features`,
from the same normalized feature matrix and LFQ engine. The protein output is
unchanged. Unlike `quant` and
`align`, `run_lfq_combine` writes no `<out>.report.json`: it calls `write_table` and
logs, with no `ArtifactReport` (`quant.rs:720`).

### align
Consumes one `seed_psms.parquet` per run (`--seed-psms`, first is the reference).
Produces `alignment.parquet` (`align.rs:130`): `run_id` (u32), `source_rt` (f64),
`reference_rt` (f64), `residual_spread` (f64). The mapping is emitted on a grid of
`grid_n` points per run (`grid_n = 100`, hardcoded in `main.rs:667`).

### mbr
Consumes an experiment-wide `scored_combined.parquet` (must carry a `source` column)
and one `psms.parquet` per run in `source` order. Produces `<out>.parquet`, one row
per accepted transfer (`mbr_worker.py:254`): `candidate_id`, `source`, `peptidoform`,
`charge`, `protein_group`, `label`, `expected_rt`, `observed_rt`, `rt_delta`,
`transfer_q` (10 columns). When there are no transfer candidates at all the worker
short-circuits and writes a placeholder table with a single empty `candidate_id`
column (`pa_write_empty`, `mbr_worker.py:289`), so `<out>.parquet` always exists.
Optionally writes an augmented scored table (`--out-psms-scored`) that lowers each accepted
transfer's `q_value` to `min(q_value, transfer_q)` on the matching `(candidate_id,
source)` row and adds an `is_transferred` bool (`mbr_worker.py:272`); this requires the
scored table to carry a `source` column. All MBR outputs are written by Python and
have no `report.json`.

The unreachable re-extraction tier (`--emit-transfer-targets <dir>`) instead writes
per-run `transfer_targets_<i>.parquet` and `transfer_decoys_<i>.parquet` in
`run_windows` format (`mbr_worker.py:142`): `candidate_id` (u32), `rt_pred_cal`,
`rt_lo`, `rt_hi` (f64), plus null `im_pred_cal`/`im_lo`/`im_hi` (f64) columns for the
3D data model. `run_mbr` never passes this flag, so `mumdia mbr` never produces these.

### report
Consumes `psms_scored.parquet` and optionally the two quant tables. The CLI takes
`--out-dir` and derives `<out-dir>/peptides.tsv` + `<out-dir>/proteins.tsv`
(`main.rs:796-797`); `run` passes the run out-dir and
`q_threshold = quant.q_threshold` (`run.rs:484-491`). `peptides.tsv` header
(`report.rs:95`): `precursor`, `stripped_sequence`, `charge`, `protein`,
`q_value`, `score`, `quantity`. `proteins.tsv` header (`report.rs:134`):
`protein_group`, `q_value`, `quantity`. No Parquet or `report.json` is written;
`report::run` returns `(n_precursors, n_protein_groups)`, and the `mumdia report`
handler prints a one-line summary (`main.rs:806`).

The only two call sites are `run.rs:484` and the `mumdia report` handler at
`main.rs:798`. The experiment orchestrator does not call the stage, so a
`run-experiment` output tree contains no `peptides.tsv` and no `proteins.tsv` at
any level.

### audit
Consumes `library_precursors.parquet` (the full search space) plus `psms`
(extract), `competed` (compete), `scored` (rescore). It also attempts to read
`<psms>.audit.parquet` (`load_extract_reasons`, `audit.rs:51`) for extract-reason
refinement, but no stage in the current chain writes that file (see the gotchas), so
the read returns empty and refinement is inert. Writes `candidate_audit.parquet`
(16 columns, `audit.rs:180`) and `<out>.metrics.json` (`audit.rs:216`).

`candidate_audit.parquet` schema (`audit.rs:180`):

| column | type | meaning |
|---|---|---|
| `run_id` | str | run identifier (`--run-id`; in `run` it is the out-dir path) |
| `precursor_id` | u32 | library `candidate_id` |
| `modified_sequence` | str | peptidoform (ProForma) |
| `charge` | i32 | precursor charge |
| `target_decoy_label` | str | `target`/`decoy` from the library |
| `entrapment_label` | bool | `protein` contains `entrapment_substr` (empty substr = always false) |
| `candidate_generated` | bool | always true (in the search space by construction, `audit.rs:168`) |
| `traces_extracted` | bool | present in `psms` (extract produced an accepted peak) |
| `peak_generated` | bool | equals `traces_extracted` (artifact resolution, `audit.rs:170`) |
| `peak_selected` | bool | equals `traces_extracted` |
| `variant_selected` | bool | present in `competed` |
| `target_decoy_winner` | bool | present in `scored` (survived to rescore) |
| `passed_precursor_fdr` | bool | scored `q_value <= q_threshold` |
| `passed_peptide_fdr` | bool | `passed_pep && passed_prec` (`audit.rs:175`) |
| `reported` | bool | equals `passed_precursor_fdr` (`audit.rs:176`) |
| `rejection_reason` | str | earliest-loss `RejectionReason::code()` |

`<out>.metrics.json` (`audit.rs:206`) records `run_id`, `q_threshold`,
`search_space`, `extracted`, `competed`, `reported`, `trace_recall`
(`extracted / max(1, search_space)`), and the per-reason `waterfall` map (as a sorted
`BTreeMap`).

## How it works

### quant (`quant.rs:259`, `run`)

1. Load `psms_scored`. The q-value column to filter on is selected by
   `cfg.q_filter`:

   | `q_filter` | column read | row unit that q is computed over | valid context |
   |---|---|---|---|
   | `PeptideQ` (default) | `peptide_q_value` | stripped/base peptide | single-run rescore |
   | `PrecursorQ` | `precursor_q` | peptidoform + charge | single-run rescore |
   | `PsmQ` | pooled `q_value` | PSM, pooled over all `source` values | any, but the gate stays experiment-wide |
   | `RunPsmQ` | `run_psm_q` | PSM within one `source` | a `source` slice of an experiment-wide rescore |

   The three grouped columns (`peptide_q_value`, `precursor_q`, `pg_q_value`) are
   sparse: `grouped_q` assigns the computed q only to each group's single
   best-scoring row and writes 1.0 to every losing sibling (`rescore.rs:721-728`).
   After an experiment-wide rescore the grouping is experiment-wide too, so each
   group's winner lives in exactly one run and a per-run slice retains only about
   `1/n_runs` of the populated rows. `PeptideQ`/`PrecursorQ` are therefore
   meaningful only when rescore itself was single-run; for a slice of a pooled
   rescore use `RunPsmQ`. Quant has no source selector: before per-run
   quantification, slice a pooled scored table by `source` and pair that slice
   with the matching chromatograms. Changing `q_filter` does not select a run.
   `run-experiment` performs that split itself but forces `PsmQ`; see the
   gotchas.
2. Group chromatogram rows by `candidate_id` into `cand_rows` (`quant.rs:301`). Rows
   whose `frag_name` starts with `ms1_` are MS1 isotope XIC pseudo-traces, not
   fragment ions; they are excluded from both peak detection and the top-N sum.
3. **Phase 1** (only when `cfg.bound_peak`): compute a per-candidate elution
   window `(lo_rt, hi_rt, apex_rt)` via `peak_window`. Schema-v3 scored tables
   carry the exact identification apex through compete/rescore; quant anchors the
   window at that apex. A missing, null, or non-finite apex from an older scored
   artifact falls back to the legacy robust summed-XIC apex detector. The
   summed XIC across all of a candidate's fragments is built in a `BTreeMap` keyed by
   the f32 RT bit pattern, so both the union RT axis and the f64 summation order are
   fixed (determinism, docs/18_findings_and_decisions.md contract B2; non-negative RTs make bit order equal value
   order). When the summed XIC has fewer than two distinct RT samples nothing can be
   bound: `peak_window` returns `(NEG_INFINITY, INFINITY)` with the lone RT as apex
   (or NaN when empty), an unbounded window (`quant.rs:107`). Otherwise the apex is
   chosen by a co-elution rule (`quant.rs:134`): among scans whose co-eluting
   nonzero-fragment count is at least `thresh = max(max_cnt - 1, 1)` (`quant.rs:135`,
   the `-1` for robustness, the `.max(1)` floor requiring at least one co-eluting
   fragment), take the highest summed intensity; fall back to a plain summed argmax
   only if no scan qualifies. That detector is now a compatibility fallback, not
   the normal source of the quant apex. `peak_bounds` then walks out from the apex with
   `peak_fraction` and `peak_grace`. A collapsed `lo==hi` window is widened to the
   adjacent grid scans so `trapezoid_window` never returns a raw height (units bug,
   `quant.rs:164`).
4. **Consensus mode** (`quant.rs:342`, `peak_window_mode == Consensus`): peak width is
   treated as a near-constant instrument/gradient property. Over confident target
   peptides (`pep_q <= reliable_q`) it takes the median left half-width `apex - lo` and
   right half-width `hi - apex`, and applies `(apex - ml, apex + mr)` around each
   candidate's apex. Requires `>= 20` anchors (`quant.rs:364`), else falls back to the
   per-candidate windows. The consensus is local to one quant invocation; separate
   runs estimate separate widths unless an external workflow supplies a shared
   policy.
5. **Phase 2** (`quant.rs:385`): integrate each fragment trace. With `bound_peak` off,
   integrate the whole trace with `trapezoid` (`quant.rs:38`). With it on, restrict to
   the chosen window via `trapezoid_window` (`quant.rs:53`). Areas are accumulated per
   candidate in `areas` and `frag_areas`.
6. **Top-N sum:** for each accepted target (`passes_quant_filter`, `quant.rs:177`),
   `summarize_fragment_areas` (`quant.rs:185`) retains only positive finite fragment
   areas, sorts descending, and sums the top `top_n_fragments`. Missing traces,
   all-zero/non-finite areas, or `top_n_fragments=0` produce a null quantity plus
   an explicit `quant_status`; they are not converted to biological zero. The
   applied apex and bounds are written with the row.
7. **Protein rollup:** within each protein group, `add_protein_base_quantity`
   (`quant.rs:220`) deduplicates charge/modification siblings by `base_peptide_id`
   using their maximum positive quantity. `rollup_protein_bases` (`quant.rs:238`)
   then sums the top `top_n_peptides` unique base peptides under `TopNSum`; `Sum` uses
   all unique bases. A group with no quantifiable base peptide has null quantity
   and `quant_status=no_quantifiable_peptide`.
8. Optional fragment export (`quant.rs:531`) and peak-bounds diagnostic
   (`quant.rs:419`). `ArtifactReport` records params + stats for each table
   (`quant.rs:603`).

Trapezoid math: `trapezoid` sums `dt*(y_i+y_{i+1})/2` in f64 over consecutive RT
samples; a single sample returns its raw intensity (`quant.rs:39`). `trapezoid_window`
first filters to samples with `lo <= rt <= hi`, then calls `trapezoid` on the subset
so the single-sample rule is identical; an empty window integrates to 0.

**Remaining limits.** A single positive fragment can still yield a quantity; the
status and `n_fragments_used` expose that evidence level but do not enforce a
minimum-clean-ion rule. Peak-window and fragment-selection policies remain
single-run choices rather than a learned cross-run consensus. Evaluate changes on
known-ratio data rather than identification count alone.

### quant-lfq (`quant.rs:657`, `run_lfq_combine`)

Reads each input table and builds `data: protein_group -> feature_key -> Vec<Option<f64>>`
of length N runs (`quant.rs:666`). The feature key is `peptidoform|charge` for MaxLFQ
and `peptidoform|charge|fragment_name` for directLFQ (`quant.rs:682`). Missing entries
stay `None`. `size_factors` (`quant.rs:753`) computes one global size factor per run;
when `normalize != None` every present value is divided by its run factor before
rollup (`quant.rs:703`). For each protein group the feature-by-run matrix is passed to
`lfq_profile` (`quant_lfq.rs:84`) and the per-run abundances are written long-form.

`lfq_profile` is the MaxLFQ least-squares reconstruction:
- Column sums/counts per sample give the fallback and the anchoring total
  (`quant_lfq.rs:89`). Single sample returns the column sum (`quant_lfq.rs:101`).
- For each sample pair `(a,b)`, the median over shared features of `ln(va)-ln(vb)` is
  an edge weight (`quant_lfq.rs:106`). Median-of-log-ratios is robust to a minority of
  genuinely changing features.
- Connected components of the sample graph are found by union-find (`quant_lfq.rs:126`).
  A singleton component falls back to its column sum.
- Each multi-sample component solves a Laplacian normal system `L x = c` with the first
  variable fixed at 0 (`solve_fixed`, `quant_lfq.rs:28`, dense Gaussian elimination
  with partial pivoting), giving log-abundances. The exp-profile is then scaled so the
  component preserves its measured total intensity (`quant_lfq.rs:176`).

`size_factors` methods (`quant.rs:753`):
- `MedianRatio` (default, DESeq-style): over complete-case features (positive in all
  runs) take each run's log2 deviation from the per-feature mean; the run factor is
  `2^median` of those deviations. Robust so a spike-in design's real fold changes are
  not flattened (test `median_ratio_recovers_global_scale_not_real_changes`,
  `quant.rs:918`).
- `Median`: align each run's median log2 intensity to the median of the per-run
  medians.
- `None`: all factors 1.0.

Determinism: medians sort in place, the matrix is iterated in `BTreeMap` key order.
With a single input, `run_lfq_combine` reduces to the per-run sum.
If no positive feature is complete across all runs, `MedianRatio` returns identity
factors (`1.0`) without estimating a scale correction. Inspect the logged
`size_factors`; identity may mean either balanced data or no usable complete cases.

### align (`align.rs:53`, `run`)

`confident_rts` (`align.rs:32`) reads a seed table's `base_peptide_id`, `spectrum_q`,
`observed_rt`, `score`, `label`, validates labels, and returns the best-scoring
observed RT per base peptide among confident targets (`q <= q_train`, target only).
The reference is `seeds[0]`. For each run, shared base peptides with the reference give
paired `(this_rt, ref_rt)`. A LOESS map is fit only when there are `>= 4` shared
peptides (`align.rs:93`, `Loess::fit(xs, ys, span=0.4, grid_n)`); the span is hardcoded
at 0.4 and is independent of `rt_im_train.loess_span`. The residual spread is the p95 of
`|ref - loess(this)|` on the shared set (`align.rs:105`); it is what sets how tight an
MBR window can be. The mapping is emitted on `grid_n.max(2)` evenly-spaced grid points;
the reference-run grid spans `[min(ref_rt, 0.0), max(ref_rt, 1.0)]` (`align.rs:58`) and
each other run's grid extends that span to cover its own shared RTs (`align.rs:88`). The
reference run and any run with too few shared peptides emit the identity map with
residual 0 (the insufficient-anchor guard, `align.rs:93`/`align.rs:113`). `align::run`
asserts at least one seed (`align.rs:55`) and writes an `alignment.report.json`
(schema `alignment` v1, `align.rs:141`, the only literal-string schema here since there
is no `schema.rs` constant for it). This is an experiment-level stage: with one run it
degenerates to identity, and it is not part of the `run` chain. Real multi-run
validation needs a multi-file experiment; only crafted two-run unit input exercises it.

### mbr (partially wired Stage D3)

The Rust side is a gate, not the algorithm. `Cmd::Mbr` (`main.rs:671`): loads config;
bails if `cfg.mbr.strategy == None` (`main.rs:680`); bails if fewer than 2 psms paths
(`main.rs:685`); requires `cfg.mbr.python` (`main.rs:688`); resolves `mbr_worker.py`
relative to the binary; calls `sidecar::run_mbr`. `run_mbr` (`sidecar.rs:162`) joins
the psms paths into a comma-separated `psms_csv` and forwards a fixed set of flags.

**Which `MbrConfig` knobs are forwarded, and which are NOT.** Forwarded to the worker:
`q_anchor` (`--q-anchor`), `min_anchor_runs` (`--min-anchor-runs`), `q_transfer`
(`--q-transfer`), `consensus_corr_min` (`--consensus-corr-min`, only together with
`--frag-csv` and only when `frag` is non-empty and `consensus_corr_min > 0`,
`sidecar.rs:209`), plus `cfg.rng_seed` as `--seed` and the optional `--out-scored`.
NOT forwarded (present in `MbrConfig`, `config.rs:873`, but dead in the wired path):
- `strategy` beyond the None/not-None gate: `EmpiricalLibrary`, `RtTransfer`, `Full`
  are indistinguishable to the worker, which always runs the rescuable transfer tier.
- `rt_window_s` (`config.rs:884`): the worker's `--rt-window` (default 20 s) is only
  read by the `--emit-transfer-targets` re-extraction tier, which `run_mbr` never
  invokes, so this knob has no effect on the wired path.
- `decoy_transfer` (`config.rs:886`): the worker hardcodes the permuted-RT null;
  `ReverseSequence`/`Both` are not implemented in the worker.
- `requant_all` (`config.rs:892`): `Full`-only requantification, unused.

The worker reads the scored table columns `candidate_id`, `source`, `label`, `q_value`,
`peptidoform`, `charge`, `protein_group` (`mbr_worker.py:81`) and per-run `apex_rt` from
each psms path. Run index 0 is the RT reference `REF` (`mbr_worker.py:101`); the
cross-run RT calibration (`to_ref`/`from_ref`, `mbr_worker.py:102`) uses `binned_map`
only when a run shares `>= 200` confident anchors with the reference, else the identity
map. The confident-anchor set uses targets only (a decoy anchor would inject a random
cross-run RT pair); decoys ride the same transfer test to measure the empirical
decoy-transfer fraction (`mbr_worker.py:87`).

The worker (`scripts/mbr_worker.py`) implements two tiers:
- **Rescuable transfer** (default path, `mbr_worker.py:156`): for each precursor
  confident (`q <= q_anchor`, target) in `>= min_anchor_runs` OTHER runs, sub-threshold
  in a target run where it WAS extracted (`c in rt_all[i]`), predict its RT in that run
  from the median of the other runs' binned-median-aligned apex RTs (`expected_rt`,
  `mbr_worker.py:116`; calibration via `binned_map`, `mbr_worker.py:31`, default 80
  bins). The false-transfer FDR is a permuted-RT decoy-transfer null: each candidate is
  given a shuffled candidate's predicted RT (`mbr_worker.py:177`). The transfer q-value
  is standard target/decoy competition on `rt_delta = |observed - predicted|` (smaller
  is better), computed as a running min from the tail (`mbr_worker.py:198`). Accept when
  `q <= q_transfer`. When no transfer candidates exist at all it writes the empty
  placeholder and returns (`mbr_worker.py:187`).
- **Fragment-consensus guard** (`mbr_worker.py:209`, M4 enhancement, active only with
  `--frag-csv` and `--consensus-corr-min > 0`): reject an accepted transfer whose
  observed fragment pattern in the target run has cosine `< corr_min` with the
  empirical consensus (per-run L1-normalized, averaged over the confident runs) over its
  confident runs. A transfer is also rejected if the candidate has no fragment data in
  the target run or fewer than `min_anchor_runs` confident runs with fragments
  (`mbr_worker.py:229`). Removes RT-concordant interference.
- **Re-extraction tier** (`--emit-transfer-targets`, `mbr_worker.py:126`): for the
  ABSENT set (confident elsewhere, not confident here, and NOT extracted here) emit
  per-run `run_windows`-format tables (`transfer_targets_<i>.parquet`) at the tight
  predicted-RT window plus a permuted-RT decoy target file (`transfer_decoys_<i>.parquet`),
  to feed `extract --restrict-candidates --run-windows`. This tier is fully implemented
  in the worker but has no Rust CLI plumbing, so it is unreachable from `mumdia mbr`.

The M5 augmented scored output (`--out-scored`, `mbr_worker.py:272`) requires the
scored table to have a `source` column and matches transfers on `(candidate_id,
source)`, taking `min(q_value, transfer_q)` and setting `is_transferred`. The worker
prints a validation summary (accepted counts per run, empirical decoy fraction, and the
RT window `delta_star` at `q_transfer`, `mbr_worker.py:245`).

### report (`report.rs:49`, `run`)

Reads the scored table columns `peptidoform`, `charge`, `protein`, `label`,
`peptide_q_value`, `protein_group`, `pg_q_value`, `score`. Builds two quant lookup
maps: peptide quant keyed `(peptidoform, charge)` (`report.rs:61`), protein quant keyed
`protein_group` (`report.rs:73`). Peptides: sort rows by `peptide_q_value` ascending,
keep the first (best-q) row per unique `(peptidoform, charge)`, targets only with
`peptide_q_value <= q_threshold` (`report.rs:101`). The row unit is the precursor
(peptidoform + charge), not the stripped sequence; `strip` (`report.rs:24`) removes a
`DECOY_` prefix and bracketed/parenthesized mod blocks for the `stripped_sequence`
column only, and a separate stripped-sequence count is logged. `q_value` is printed at
6 decimals, `score` at 4, `quantity` via `qcell` (1 decimal, empty on NaN,
`report.rs:39`). Proteins: sort by `pg_q_value`, unique non-empty protein groups,
targets with `pg_q_value <= q_threshold` (`report.rs:137`). Returns `(n_precursors,
n_protein_groups)`.

This is a hybrid identification report: `peptides.tsv` has precursor-shaped rows
but is filtered and labeled with peptide-level q. It is neither a stripped-peptide
table nor a `precursor_q`-controlled precursor table. Report filtering is also
independent of `quant.q_filter`, so a report row can legitimately have a blank
quantity when quant excluded it or marked it nonquantifiable. Use
`peptide_quant.parquet`/`protein_group_quant.parquet` for numerical analysis:
Parquet retains nullable f64 precision and status/bounds, while TSV quantity is a
presentation value rounded to one decimal.

### audit (`audit.rs:64`, `run`)

The search space is every library precursor (`candidate_id`, `peptidoform`, `charge`,
`label`, `protein`). Survivor sets are built as `HashSet<u32>` of `candidate_id` from
`psms` (extracted), `competed`, and `scored`; scored also yields `q_by_cid` and
optional `pepq_by_cid` (peptide-level q, only present in some scored schemas,
`audit.rs:90`). `load_extract_reasons` (`audit.rs:51`) tries to read a
`<psms>.audit.parquet` sidecar to refine the extract-stage bucket, but nothing in the
current chain writes that file (see the gotchas), so the map is empty and every
extract-stage loss buckets to `NO_PEAK_GROUP`. For each candidate the earliest
rejection reason is assigned along the ladder (`audit.rs:136`):
- not in `extracted` -> refined from the sidecar
  (`NO_FRAGMENT_TRACES`/`NO_VALID_FRAGMENTS`/`PEAK_NOT_SELECTED`/`RT_PRUNED`/
  `WRONG_ISOLATION_WINDOW`, `audit.rs:139`) or the generic `NO_PEAK_GROUP` when no
  sidecar (the only outcome today);
- extracted but not in `competed` -> `OUTCOMPETED_BY_DECOY` (decoy) or
  `OUTCOMPETED_BY_TARGET` (target);
- competed but `q > q_threshold` -> `FAILED_PRECURSOR_FDR`;
- passes precursor but fails peptide q -> `FAILED_PEPTIDE_FDR` (peptide q falls back to
  the precursor gate when absent, `audit.rs:133`);
- else `REPORTED`.

The waterfall counts per reason and is logged sorted by descending count
(`audit.rs:220`); `<out>.metrics.json` records the fields listed under
`## Inputs and outputs`. This stage never re-runs compute and never mutates a pipeline
output, so it is safe to run after any search. In the `run` orchestrator the stage is
invoked only when `extract.emit_candidate_audit` is set (`run.rs:428`), with
`q_threshold = 0.01`, `run_id = out_dir`, and no entrapment substring; standalone,
`mumdia audit` takes those as CLI args.

`RejectionReason` (`rejection.rs:19`) is a 17-variant enum with a stable
SCREAMING_SNAKE_CASE `code()` (`rejection.rs:50`), a `stage_order()` ladder position
(0 earliest, `Reported` = 255, `rejection.rs:76`), an `is_rejection()` predicate
(true for any non-`Reported` reason, `rejection.rs:100`), and `earliest()` to keep the
smaller stage (`rejection.rs:106`). The variants are grouped by pipeline stage in the
source (search space, candidate generation/pruning, extraction, ranking, competition,
FDR/reporting).

## Key types and functions

| name | file:line | what it does |
|---|---|---|
| `QuantParams` | quant.rs:21 | inputs/outputs + `&QuantConfig` for the quant stage |
| `trapezoid` | quant.rs:38 | f64 trapezoidal integral of one trace; single point = raw intensity |
| `trapezoid_window` | quant.rs:53 | trapezoid restricted to `[lo,hi]`; empty window = 0 |
| `peak_window` | quant.rs:80 | summed-XIC apex (co-elution rule) + descent-walk window |
| `passes_quant_filter` | quant.rs:177 | accept a scored row: non-decoy (`label != "decoy"`), finite q, q <= threshold |
| `summarize_fragment_areas` | quant.rs:185 | top-N positive fragment sum + quant_status; missing/zero -> null |
| `add_protein_base_quantity` | quant.rs:220 | record one base peptide's max quantity for protein rollup |
| `rollup_protein_bases` | quant.rs:238 | TopNSum/Sum over unique quantifiable base peptides |
| `quant::run` | quant.rs:259 | full quant stage |
| `run_lfq_combine` | quant.rs:657 | build the protein-by-run matrix for `quant-lfq` |
| `size_factors` | quant.rs:753 | per-run normalization factors (MedianRatio/Median/None) |
| `median_sorted` | quant.rs:818 | in-place median helper (empty = 0.0; assumes finite, no NaN) |
| `lfq_profile` | quant_lfq.rs:84 | MaxLFQ least-squares per-sample profile |
| `median` | quant_lfq.rs:12 | private in-place median (duplicate of `median_sorted`, local to `quant_lfq`) |
| `solve_fixed` | quant_lfq.rs:28 | dense Laplacian solve, first var fixed at 0 |
| `maxlfq` / `directlfq` | quant_lfq.rs:191 / 198 | granularity-specific wrappers over `lfq_profile` |
| `AlignParams` | align.rs:22 | seeds + q_train + grid_n |
| `confident_rts` | align.rs:32 | best observed RT per confident target base peptide |
| `align::run` | align.rs:53 | reference LOESS RT map per run |
| `run_mbr` | sidecar.rs:162 | build argv and spawn `mbr_worker.py` |
| `binned_map` | mbr_worker.py:31 | monotone binned-median RT calibration |
| `expected_rt` | mbr_worker.py:116 | cross-run predicted RT for a candidate in a run |
| `pa_write_empty` | mbr_worker.py:289 | placeholder output when there are no transfer candidates |
| `ReportParams` / `report::run` | report.rs:13 / 49 | TSV writer |
| `strip` | report.rs:24 | stripped sequence from a peptidoform |
| `qcell` | report.rs:39 | quantity cell formatting (1 decimal; empty on NaN) |
| `AuditParams` / `audit::run` | audit.rs:28 / 64 | candidate loss ladder |
| `load_extract_reasons` | audit.rs:51 | optional `<psms>.audit.parquet` refinement (no producer yet) |
| `RejectionReason` | rejection.rs:19 | earliest-loss category enum (`code`/`stage_order`/`is_rejection`/`earliest`) |
| `resolve_script` | sidecar.rs:20 | locate a worker script relative to CWD or the binary |
| `run_worker` | sidecar.rs:217 | spawn `python <script> <args>`; error on non-zero exit |

## Configuration

**`QuantConfig`** (`config.rs:793`, defaults at `config.rs:822`), read by `quant`:
- `q_threshold` (0.01): peptide q cutoff for inclusion in the quant tables.
- `top_n_fragments` (3): fragments summed per peptidoform.
- `top_n_peptides` (3): peptides summed per protein group under `TopNSum`.
- `rollup` (`TopNSum`): `TopNSum` or `Sum` (`config.rs:706`).
- `bound_peak` (true): integrate over the detected window vs the whole trace.
- `peak_fraction` (1/6): descent threshold, fraction of apex height.
- `peak_grace` (1): consecutive sub-threshold scans bridged during the walk.
- `peak_window_mode` (`PerCandidate`): `PerCandidate` or `Consensus` (`config.rs:717`).
- `reliable_q` (0.001): confident-set cutoff calibrating the consensus half-widths.
- `q_filter` (`PeptideQ`): which q column to filter on:
  `PeptideQ`/`PrecursorQ`/`PsmQ`/`RunPsmQ`. Use `PeptideQ`/`PrecursorQ` only
  after a single-run rescore, because the grouped columns are populated on group
  winners only (`rescore.rs:721-728`). For experiment-wide rescoring, slice by
  `source` and normally use `RunPsmQ`; `q_filter` alone does not select the run.
  `run-experiment` ignores this field and forces `PsmQ` (`run_experiment.rs:498`).

**`quant-lfq`** takes no config struct; `--method` (maxlfq/directlfq) and `--normalize`
(median_ratio/median/none, parsed by `NormalizeMethod::from_token`, `config.rs:754`)
are CLI args. Default normalize is `median_ratio`.

**align** reads `rt_im_train.q_train` (0.01, `config.rs:326`) as the anchor cutoff;
`grid_n` is hardcoded to 100 in `main.rs:667` and the LOESS span to 0.4 in `align.rs:93`.

**`MbrConfig`** (`config.rs:873`, defaults at `config.rs:898`): `strategy` (`None`;
`MbrStrategy` at `config.rs:845`), `q_anchor` (0.01), `min_anchor_runs` (2),
`q_transfer` (0.01), `rt_window_s` (20.0, unwired), `decoy_transfer` (`PermutedRt`,
`DecoyTransfer` at `config.rs:864`, unwired), `consensus_corr_min` (0.0), `requant_all`
(false, unwired), `python` (None, required when `strategy != None`). See the mbr
section for exactly which of these reach the worker.

**report** takes `q_threshold` (`--q`, default 0.01) and optional quant table paths as
CLI args; no config struct.

**audit** takes `q` (0.01), `run_id` ("run"), `entrapment_substr` ("") as CLI args.

The config was recently pruned of dead fields. The `MbrConfig` knobs noted as
"unwired" above are intentionally documented as such: they still exist in the struct
but do not affect the wired `mumdia mbr` path.

## Invariants, determinism, gotchas

- **Determinism.** quant uses `BTreeMap` for candidate iteration, the RT union axis,
  and f64 summation order; quant-lfq iterates `BTreeMap` keys and sorts medians in
  place; align iterates via `HashMap` but writes a fixed grid. The `mbr_worker.py`
  permutation is seeded from `cfg.rng_seed` (`--seed`). The DeepLC fine-tune elsewhere
  is nondeterministic, but none of these stages depend on it.
- **quant units bug guard.** A collapsed `lo==hi` window would make `trapezoid_window`
  return a raw height (intensity, not intensity*seconds), mixing units against
  broad-peak peptides; `peak_window` widens to two grid scans to prevent this
  (`quant.rs:164`, test `peak_window_never_collapses_to_single_sample`).
- **Quant anchors at the identification apex.** Schema-v3 scored rows carry
  `apex_rt`; quant uses it and only falls back to XIC apex detection for legacy,
  missing, or non-finite values. Applied apex/bounds are emitted for QC.
- **Missing is not zero.** Nonquantifiable accepted IDs receive null `quantity`
  and an explicit `quant_status`. LFQ ignores null, non-finite, and nonpositive
  values.
- **Competition affects matrix completeness.** The default upstream
  `group_by=precursor` collapses charge/modification siblings separately within
  each label. `peptidoform_charge` preserves them, but changes the rescoring/FDR
  population and must be benchmarked before production use.
- **MS1 traces excluded.** `ms1_*` chromatogram rows are precursor channels and never
  enter peak detection or the top-N sum (`quant.rs:303`).
- **Grouped q columns are populated on group winners only.** `grouped_q`
  (`rescore.rs:673`) writes the computed q to the single best-scoring row of each
  group and 1.0 to every loser (`rescore.rs:721-728`). This keeps a losing
  charge/modification sibling from inheriting the winner's q, but it means
  `peptide_q_value`, `precursor_q`, and `pg_q_value` are sparse. After an
  experiment-wide rescore each group has one winner across the whole experiment,
  so filtering or counting a single run's slice on one of those columns keeps
  roughly `1/n_runs` of the rows it should and is not a per-run number. The
  correct per-file PSM unit for a pooled rescore is `run_psm_q`.
- **`run-experiment` overrides `quant.q_filter`.** The experiment orchestrator
  splits the scored table by `source` (`run_experiment.rs:474-477`) and then sets
  `q_filter = PsmQ` for every per-run quant call (`run_experiment.rs:498`),
  because the downstream cross-run LFQ step assumes the pooled column. A
  configured value other than `psm_q` is logged as a warning
  (`run_experiment.rs:490-497`) and then ignored; no artifact records the
  substitution. Per-run quantities out of `run-experiment` are therefore gated on
  the pooled `q_value`, not on `run_psm_q` and not on the configured column.
- **`run-experiment` produces no TSV report.** `report::run` is called only from
  `run.rs:484` and the `mumdia report` handler (`main.rs:798`), so an experiment
  output tree has scored, per-run split, quant, and LFQ artifacts but no
  `peptides.tsv` or `proteins.tsv`. Take per-run identification counts from the
  split `<run>/scored.parquet` tables gated on `run_psm_q`, or run `mumdia report`
  by hand on a split table.
- **report row unit.** `peptides.tsv` rows are precursors (peptidoform + charge), not
  stripped sequences, but selection is by `peptide_q_value`, not `precursor_q`.
  A separate stripped-sequence count is only logged. TSV quantity is rounded; use
  Parquet for quantitative work.
- **`<psms>.audit.parquet` has no producer.** `emit_candidate_audit` (`config.rs:483`)
  documents that extraction "writes `<out-psms>.audit.parquet`", but no stage in the
  current tree writes it (grep: only `compete` writes `<out>.compete_audit.parquet` and
  the `audit` stage writes `candidate_audit.parquet`). What `emit_candidate_audit`
  actually does in `run` is gate whether the `audit` STAGE is invoked at all
  (`run.rs:428`), not emit an in-extract per-candidate sidecar. So `load_extract_reasons`
  always reads nothing and the refined extract reasons are unreachable today.
- **audit artifact resolution.** `peak_generated`, `peak_selected`, and
  `traces_extracted` are all set from the same `traces` flag (`audit.rs:170`) because
  the artifacts only record presence in `psms`; the (not-yet-written) in-extract sidecar
  would be the only way to split "no traces" from "traces but no accepted peak".
- **audit `reported` vs `REPORTED`.** The `reported` bool column is set from
  `passed_prec` alone (`audit.rs:176`), while the `REPORTED` rejection reason additionally
  requires the peptide gate. A candidate can therefore have `reported=true` yet
  `rejection_reason=FAILED_PEPTIDE_FDR`. Treat `rejection_reason` as authoritative.
- **audit reason coverage.** In the current chain `audit.rs` can only ever emit
  `NO_PEAK_GROUP`, `OUTCOMPETED_BY_TARGET`/`OUTCOMPETED_BY_DECOY`, `FAILED_PRECURSOR_FDR`,
  `FAILED_PEPTIDE_FDR`, and `REPORTED`. The five refined extract codes
  (`NO_FRAGMENT_TRACES`, `NO_VALID_FRAGMENTS`, `PEAK_NOT_SELECTED`, `RT_PRUNED`,
  `WRONG_ISOLATION_WINDOW`) are matched by `load_extract_reasons` (`audit.rs:139`) but
  need the absent sidecar to fire. The six remaining enum variants
  (`PEPTIDE_NOT_GENERATED`, `MODIFICATION_NOT_ALLOWED`, `CHARGE_OUT_OF_RANGE`,
  `PRECURSOR_MZ_OUT_OF_RANGE`, `CANDIDATE_CAP_REACHED`, `REMOVED_DURING_REPORTING`) have
  no producer at all: even a sidecar string of that name falls to the `_ => NoPeakGroup`
  arm (`audit.rs:145`).
- **report.json coverage is partial.** `quant` writes reports for
  `peptide_quant`, `protein_quant`, and emitted `fragment_quant`; the peak-bounds
  diagnostic gets none. `quant-lfq` and `mbr` write no `report.json`; `align` and `audit` do
  (the latter as `<out>.metrics.json`). Do not assume every artifact here has a sidecar
  report.
- **align / mbr need >=2 runs.** align degenerates to identity on one run; `mumdia mbr`
  hard-bails on `<2` psms paths (`main.rs:685`). Neither is in the single-run `run`
  chain; from `run`, only `quant` (with fragment export, no peak-bounds,
  `run.rs:445`), `report` (`run.rs:484`), and the optional `audit` stage
  (`run.rs:430`) fire.
- **quant-lfq single input** reduces to the per-run sum; MaxLFQ/directLFQ have only
  synthetic-matrix unit tests, no real multi-run validation.
- **two median helpers.** `quant::median_sorted` (`quant.rs:818`) and
  `quant_lfq::median` (`quant_lfq.rs:12`) are independent in-place median functions with
  the same semantics; they are not the shared stats kernel and are easy to drift apart.

## How to extend / modify

- **New rollup or normalization strategy:** add a variant to `RollupMethod`
  (`config.rs:706`) or `NormalizeMethod` (`config.rs:737`) and a match arm in
  `rollup_protein_bases` (`quant.rs:248`) or `size_factors` (`quant.rs:758`). Keep
  defaults conservative and iterate `BTreeMap` in key order for determinism.
- **Quantifiability gate:** filter `summarize_fragment_areas` (`quant.rs:185`) on
  `n_fragments_used` or a minimum window width; emit the reason so `report` can surface
  low-evidence quantities.
- **Wire the MBR strategy tiers:** thread `cfg.mbr.strategy` past the None gate in
  `main.rs:680` and forward `rt_window_s`/`decoy_transfer`/`requant_all` in
  `run_mbr` (`sidecar.rs:193`). The re-extraction tier already exists in the worker
  (`--emit-transfer-targets`); it needs a Rust subcommand that then calls
  `extract --restrict-candidates --run-windows` and re-scores. Implement the
  `ReverseSequence`/`Both` nulls in the worker before honoring `decoy_transfer`.
- **Consume `alignment.parquet` in MBR** instead of the worker's internal
  `binned_map`, so the RT map is computed once and shared; align already records the
  p95 residual that should set `rt_window_s`.
- **Add an audit reason:** extend `RejectionReason` (`rejection.rs:19`) with a
  `code()`, `stage_order()`, and a branch in the ladder (`audit.rs:136`); prefer
  emitting it from the in-extract sidecar so `load_extract_reasons` can pass it through.
- **New quant artifact column:** add the `Col` in the relevant `write_table`
  (`quant.rs:487`/`519`/`558`), bump the schema version in
  `mumdia-core/src/schema.rs`, and update the consumer (`report.rs`, `run_lfq_combine`).
