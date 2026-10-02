# TIMS roadmap, part 2: where diaPASEF identifications are lost, and what to try next

Status: proposed 2026-09-25; sections 3 and 6 record what was measured and implemented since. Branch `IM`.
It continues [TIMS_ROADMAP.md](TIMS_ROADMAP.md), whose phases P0-P7 are done. The objective
order and the governance rules are the same:
- identification sensitivity at 1% first, then FDR validity, quantification and runtime;
- every arm is measured against the P6 base on the same host;
- a claimed gain needs 3 paired `nn_torch` seeds;
- a gain is checked by entrapment, and a default needs a second diaPASEF acquisition
  (the ProteoBench HYE diaPASEF set, six runs; changed from PYE by the user on 2026-09-25).

## 1. Starting point

**Base.** The P6 configuration
(`/public/compomics2/Robbe/MuMDIA_data/output_mumdia_p6_best/config.json`) on
`LFQ_Ultra2_diaPASEF_15min_50ng_Ecoli_01.d` gives 11,037 precursors, 9,507 peptides and
1,472 protein groups at 1% (3-seed means, units of TIMS_ROADMAP.md section 2), against
DIA-NN 2.5.0's 18,673 / 15,404 / 1,919. That is 0.62x in peptides. Entrapment on this
configuration gives an empirical FDP of 0.36-0.38%. P7 (per-peak IM width and apex
peak-shape features) gave no gain (-0.9%, 3 seeds).

**Why the existing ladder is not enough.** `bench/tims_compare.py` counts at the
stripped-peptide level. It calls a peptide "extracted" when any target row with that
sequence exists, whatever its charge, modform or apex. So its "extracted, below 1%" group
(5,943 peptides) mixes two different failures: the right peak scored too low, and the wrong
peak scored. `mumdia audit` cannot separate them either. Above extract it knows only
"outcompeted" and "failed FDR". Below it, every loss is `DID_NOT_SURVIVE_EXTRACTION`,
because the per-candidate table it reads (`<psms>.audit.parquet`) is written by nothing
(`extract.emit_candidate_audit` is unwired; CLAUDE.md, peak-cap subsection).

**A precursor-level cut** on the P6 run (read-only, 2026-09-25). DIA-NN's 1% precursors
(`Q.Value` <= 0.01, `Decoy` = 0) are joined to MuMDIA on the I/L-merged stripped sequence
plus the charge. Each DIA-NN precursor is paired with the best-scoring MuMDIA target row
of that key. "Right peak" means that MuMDIA's apex lies within 5 s of DIA-NN's `RT`. The RT
and IM windows come from `run_windows`.

| group | precursors | share |
|---|---|---|
| DIA-NN 1% precursors (distinct sequence, charge) | 17,591 | 100% |
| accepted by MuMDIA (`q_value` <= 0.01) | 10,138 | 57.6% |
| not extracted | 762 | 4.3% |
| extracted, rejected, right peak | 4,327 | 24.6% |
| extracted, rejected, wrong peak, DIA-NN RT inside the MuMDIA RT window | 1,384 | 7.9% |
| extracted, rejected, wrong peak, DIA-NN RT outside the window | 980 | 5.6% |

The counts are those of `bench/tims_loss.py` (D1 result). The first ad hoc cut gave 4,328 /
1,387 / 976; the difference is which DIA-NN row represents the 1,082 keys that DIA-NN reports
under several modforms.

Signatures of the groups:

| | accepted | rejected, right peak | rejected, wrong peak |
|---|---|---|---|
| apex within 5 s of DIA-NN RT | 96.8% | 100% (by definition) | 0% |
| apex 1/K0 error against DIA-NN `IM`, median | 0.0029 | 0.0086 | |
| apex 1/K0 error > 0.02 | 3.0% | 20.2% | |
| DIA-NN `IM` inside the MuMDIA IM window | 98.7% | 95.1% | 89.5% |
| calibrated RT prediction error against DIA-NN RT, median / p95 (s) | 2.9 / 11.6 | 4.1 / 65.3 (all rejects) | |
| matched fragments at the apex, median | 12 | 11 (all rejects) | |
| DIA-NN `Precursor.Quantity`, median | 1.01e5 | 2.09e4 (all rejects) | |

The rejected precursors have these properties:
- **They are not borderline.** Their `q_value` has quartiles 0.09 / 0.54 / 0.89. Only 6.9%
  lie at 1-2% and 17.6% at or below 5%. Relaxing the threshold or small rescore gains will
  not recover them.
- **They are faint.** MuMDIA accepts 19.6 / 38.7 / 59.2 / 77.8 / 92.9% of DIA-NN's
  precursors from the lowest to the highest `Precursor.Quantity` quintile (D1 keys).
- **The right-peak group is the largest** (65% of rejects). At the correct retention time,
  the apex 1/K0 is three times further from DIA-NN's than for accepted precursors. The
  matched fragment peaks there partly belong to other ions inside the IM window (half-width
  0.040). Among right-peak rejects the `q_value` quartiles are 0.05 / 0.21 / 0.76.
- **The wrong-peak group splits in two.**
  - 59% have DIA-NN's apex inside MuMDIA's RT window, so the right peak was available and
    peak selection lost it.
  - 41% lie outside the window, a retention-time prediction tail: the calibrated
    prediction error has a p95 of 65 s against 11.6 s for accepted precursors.

The cut is a first reading. D1 below turns it into a reproducible tool.

## 2. Diagnosis phase (D): measure before adding levers

Each item is read-only on existing outputs, or uses the fast loop
(`/public/compomics2/Robbe/MuMDIA_data/p6/chain.sh`: extract -> features -> compete ->
rescore on an existing run).

### D1: precursor-level loss classifier (`bench/tims_loss.py`)

- **Inputs:** DIA-NN `report.parquet`, `psms_scored`, `psms_extracted`, `run_windows`, the
  library precursors, and optionally the top-K peak sidecar (`<psms>.peaks.parquet`).
- **Join:** as in section 1. It additionally reports the exact-peptidoform join, so that
  modform mismatches are visible.
- **Categories:**
  - not in library;
  - not extracted, split by whether DIA-NN's RT and IM lie inside the candidate's windows
    (outside RT, outside IM, inside both = gate or presence loss);
  - extracted with the wrong peak, inside or outside the RT window;
  - right peak with a low score;
  - accepted.
- **Per category:** count, DIA-NN intensity quintile, apex RT and 1/K0 error, matched
  fragments, `q_value`.
- **Outputs:** a summary table and a per-precursor parquet with the category label, which
  D2, D4 and every later arm reuse. Every later arm reports its effect as movement between
  these categories, not only as a count.
- **Replaces** the stripped-peptide ladder of `tims_compare.py` for this work. That script
  stays for the headline counts.
- **Not planned:** wiring `emit_candidate_audit` in extract. D1 covers the extract split at
  this scale (762 precursors). It is the fix if a finer extract reason (presence, gate,
  presence-minimum) is needed later.

### D1 result (2026-09-25)

`bench/tims_loss.py` implements the classifier. Unit: a DIA-NN 1% precursor (`Q.Value` <=
0.01, `Decoy` = 0) keyed on the I/L-merged stripped sequence and the charge (17,591 keys; for
the 1,082 keys with several DIA-NN modforms the lowest `Q.Value` row is kept). Each key is
paired with the highest-`score` MuMDIA target row of that key; "accepted" is that row's
`q_value` (pooled PSM q) <= 0.01. Apex RT and 1/K0 come from `psms_extracted` at
(`candidate_id`, `selected_peak_rank`). Windows come from `run_windows` for that row's
candidate, or for the key's library candidate (DIA-NN's peptidoform when present) if nothing
was extracted. Outputs of the P6 base run: `/public/compomics2/Robbe/MuMDIA_data/bis/d1/`
(`p6_loss.parquet` with one row per key and the columns `category`, `category_exact_pf`,
`quintile`, `rt_err`, `im_err`, `rt_pred_err`, `q_value`; `p6_loss.txt`). Runtime 4 s.

```text
python bench/tims_loss.py --diann output_diann/report.parquet --run-dir output_mumdia_p6_best \
  [--scored ... --extracted ... --peaks <psms_extracted>.peaks.parquet] --out loss.parquet
```

| category | n | share | median quintile | median abs RT err (s) | median abs 1/K0 err | 1/K0 err > 0.02 | p95 abs RT pred err (s) | median matched frags | `q_value` quartiles | best row is DIA-NN's peptidoform |
|---|---|---|---|---|---|---|---|---|---|---|
| not in library | 0 | 0% | | | | | | | | |
| not extracted, RT outside window | 194 | 1.1% | 1 | | | | 76.2 | | | |
| not extracted, RT in, IM outside window | 173 | 1.0% | 2 | | | | 11.0 | | | |
| not extracted, both inside (gate or presence) | 395 | 2.2% | 1 | | | | 12.2 | | | |
| accepted | 10,138 | 57.6% | 4 | 0.97 | 0.0029 | 3.0% | 11.7 | 12 | 0.000 / 0.000 / 0.001 | 97.6% |
| right peak, low score | 4,327 | 24.6% | 2 | 0.97 | 0.0086 | 20.2% | 11.3 | 11 | 0.049 / 0.209 / 0.758 | 99.9% |
| wrong peak, DIA-NN RT inside window | 1,384 | 7.9% | 2 | 9.7 | 0.0142 | 33.2% | 11.5 | 11 | 0.676 / 0.876 / 0.953 | 99.8% |
| wrong peak, DIA-NN RT outside window | 980 | 5.6% | 2 | 36.8 | 0.0131 | 33.1% | 130.6 | 10 | 0.437 / 0.835 / 0.934 | 55.5% |

Categories per DIA-NN `Precursor.Quantity` quintile (1 = faintest; 3,518-3,519 keys each):

| category | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| accepted | 691 | 1,360 | 2,082 | 2,736 | 3,269 |
| right peak, low score | 1,559 | 1,343 | 878 | 437 | 110 |
| wrong peak, inside | 453 | 401 | 291 | 173 | 66 |
| wrong peak, outside | 384 | 269 | 179 | 104 | 44 |
| not extracted (all three) | 432 | 145 | 88 | 68 | 29 |

Reading:
- The section 1 cut is reproduced to within 4 keys per group (table updated above).
- **The out-of-window group is 44% modform confusion.** In 436 of its 980 keys the best
  MuMDIA row is another modform of the same sequence and charge (431 differ in the number
  of Met oxidations, 409 of them with the oxidation on the MuMDIA side), so its RT window belongs to a different
  molecule. Paired with DIA-NN's own peptidoform (`category_exact_pf`), those 980 split into
  595 still outside the window, 175 right peak with a low score, 155 wrong peak inside the
  window and 55 not extracted. The RT-tail population that L3 addresses is therefore about
  600 keys, not 980. In the other groups the two pairings agree on 97.6-99.9% of keys.
- On the exact-peptidoform pairing 10,076 keys are accepted (against 10,138): 62 keys are
  accepted only on a different modform than the one DIA-NN reports.
- The not-extracted group splits into 194 RT, 173 IM and 395 gate or presence losses, the
  last almost entirely in quintile 1 (270 of 395).
- The wrong-peak groups have 1/K0 errors (median 0.013-0.014) above the right-peak rejects
  (0.0086), as expected for a different ion. Their `q_value` is not borderline (median 0.88
  inside the window).

### D2: which evidence makes a true precursor look like a decoy

For the D1 right-peak rejects, compute the AUC and the median of every one of the 396
features (402 with the P7 block) against:
- accepted targets;
- decoys at a matched score.

This extends `bench/tims_im_features.py` from the IM block to all columns, with the D1
label as the population. It answers whether these precursors fail on library intensity
agreement, co-elution, fragment count, mass error, IM or MS1. That tells whether L1a, L1c
or L1d below is the right first lever. The docs/28 section 3 analysis is the template.

### D2 result (2026-09-25)

`bench/tims_loss_features.py` (sibling of `tims_im_features.py`). Populations, one
`features.parquet` row each at (`candidate_id`, `selected_peak_rank`) of the P6 base run:
- R: the 4,327 D1 right-peak rejects (best row per key);
- A: the 10,138 D1 accepted keys;
- Aq: A resampled to R's DIA-NN `Precursor.Quantity` distribution (4,327 draws). This is the
  abundance control, because R is fainter (median quintile 2 against 4);
- Dm: decoys resampled to R's rescore `score` distribution (4,327 draws, median score 0.957
  in both). These are the decoys R competes with;
- D: all 250,843 decoys.

Resampling uses 20 quantile bins, with replacement, seed 0. AUC is P(R > other); 0.5 means no
separation. All 396 rescore features were scored; the table is in
`/public/compomics2/Robbe/MuMDIA_data/bis/d2/right_peak.tsv`. 64 features have
|AUC(R, Aq) - 0.5| >= 0.25 together with |AUC(R, Dm) - 0.5| <= 0.05. Representatives per
evidence family:

| family | feature | AUC R vs Aq | AUC R vs Dm | AUC A vs D | median R / Aq / Dm |
|---|---|---|---|---|---|
| library intensity agreement | `spectral_angle` | 0.208 | 0.522 | 0.932 | 0.367 / 0.611 / 0.352 |
| | `kl_obs_pred` | 0.853 | 0.487 | 0.040 | 1.10 / 0.45 / 1.16 |
| | `scribe_score` | 0.157 | 0.518 | 0.952 | 1.52 / 2.68 / 1.45 |
| fragment co-elution | `frag_ref_corr_mean_full` | 0.115 | 0.512 | 0.979 | 0.231 / 0.430 / 0.229 |
| | `ref_corr` | 0.230 | 0.500 | 0.903 | 0.181 / 0.367 / 0.181 |
| | `coelution_mean` | 0.241 | 0.482 | 0.890 | 0.003 / 0.110 / 0.005 |
| | `evidence` | 0.227 | 0.500 | 0.902 | 2.13 / 4.37 / 2.13 |
| fragment count | `n_matched_fragments` | 0.438 | 0.579 | 0.762 | 11 / 11 / 10 |
| | `matched_fraction` | 0.463 | 0.585 | 0.760 | 0.917 / 0.917 / 0.917 |
| signal | `log_apex_intensity` | 0.431 | 0.607 | 0.907 | 8.71 / 8.90 / 8.26 |
| mass error | `weighted_mass_error` | 0.570 | 0.504 | 0.338 | 2.65 / 2.37 / 2.57 ppm |
| ion mobility | `im_error_abs` | 0.578 | 0.495 | 0.385 | 0.012 / 0.009 / 0.012 |
| | `im_elution_sd` | 0.654 | 0.485 | 0.197 | 0.0090 / 0.0070 / 0.0090 |
| MS1 | `isotope_corr` | 0.461 | 0.501 | 0.667 | 0 / 0.109 / 0 |
| RT | `rt_error_abs` | 0.565 | 0.430 | 0.284 | 3.5 / 2.7 / 4.6 s |

Where R is more target-like than its score-matched decoys (AUC R vs Dm 0.60-0.62):
signal volume (`total_xic_log`, `log_apex_intensity`, `sum_y_intensity`), fragments present
in the peak (`n_frag_present_inpeak`, `frac_frag_present_inpeak`) and the cross-correlation
shape (`xcorr_shape`). The classifier credits R for having signal and fragments. It then
loses R on what that signal looks like.

Within-group checks (Spearman, peak rank 0):
- In R, `spectral_angle` falls with the apex 1/K0 error (rho -0.22; median 0.42 / 0.38 /
  0.31 over the |1/K0 error| terciles). Accepted targets show the same trend (rho -0.24).
- In R, `frag_ref_corr_mean_full` is uncorrelated with the apex 1/K0 error (rho -0.02) and
  with DIA-NN abundance (rho 0.12). It sits at 0.23 in every 1/K0-error tercile. Among
  accepted targets it follows abundance (rho 0.58).
- An MS1 monoisotopic peak is found for 66.6% of accepted keys, 49.5% of R and 46.4% of
  decoys.

Reading:
- **The missing evidence is fragment agreement, in two independent families.** On library
  intensity agreement and on fragment co-elution, the right-peak rejects are
  indistinguishable from score-matched decoys (AUC 0.48-0.52). They are far from
  abundance-matched accepted targets (AUC 0.12-0.24, or 0.76-0.85 for distances). Abundance
  does not explain it: Aq has almost the same apex intensity (median 8.90 against 8.71).
- **Not missing:** fragment count, mass error, IM and RT. On these R lies between Aq and Dm
  or at Aq (AUC R vs Aq 0.44-0.65). The IM features separate accepted targets from decoys
  (A vs D 0.20-0.39), but they do not tell R from its decoys. P5 and P7 found the same
  redundancy.
- **Library agreement tracks apex IM contamination; co-elution does not.** The first fits L1a:
  a brighter co-isolated ion inside the IM window distorts the apex fragment pattern. The
  second is flat in the apex 1/K0 error and at decoy level even in the best tercile. The
  fragment traces themselves do not co-elute. A per-scan choice of the most intense
  in-tolerance peak inside a ±0.04 window can make each trace jump between ions from scan to
  scan. So L1a must cover the trace build as well as the apex post-pass, and co-elution is
  its test.
- Library quality (L1c) can explain part of the first family but none of the second, because
  co-elution does not depend on predicted intensities. D3 measures that share.
- MS1 is weak evidence throughout. A third of accepted precursors have no MS1 monoisotopic
  peak at the apex, and in R MS1 presence is at decoy level. This is a candidate lever that
  the plan does not yet list (MS1 extraction on diaPASEF). It is noted here, not planned.

### D2b: do the fragment traces jump between ions? (added after D2)

D2 found that the right-peak rejects fail on fragment co-elution at decoy level, and that this
does not follow the apex 1/K0 error. Two explanations predict different fixes:
- **IM jitter.** Each trace takes the most intense in-tolerance peak per scan inside the
  ±0.04 IM window, so it can switch between ions from scan to scan. Fix: L1a on the trace
  build.
- **Sparsity.** A 3-scan FWHM at 50 ng leaves too few non-zero points for a stable
  correlation. Fix: smoothing, or scoring over RT x IM rather than RT alone.

Read-only on chromatograms v2, which carry a per-point `im`. For R, Aq and Dm of D2, and per
fragment trace inside the peak bounds:
- the 1/K0 jitter (the SD of the per-point `im`, and the median step between scans);
- the share of points more than one IM tolerance from the candidate's calibrated 1/K0;
- the number of non-zero points.

There is already a weak signal: `im_elution_frag_sd` is 0.007 / 0.005 / 0.007 for R / Aq /
Dm (AUC R vs Aq 0.66, R vs Dm 0.50).

### D2b result (2026-09-25)

`bench/tims_trace_im.py`, on the D2 populations (`tims_loss_features.py --save-pops`; R, Aq, Dm
of 4,327 each, A 10,138). Per fragment trace (chromatograms v2) inside the scored peak bounds,
then the median over each candidate's observed traces. `off_frac` is the share of a trace's
intensity carried by points more than 0.01 in 1/K0 from the candidate's apex 1/K0 (about 1.4x
the single-ion SD of 0.007). Outputs: `/public/compomics2/Robbe/MuMDIA_data/bis/d2b/`.

| statistic | R | Aq | Dm | A | AUC R vs Aq | AUC R vs Dm |
|---|---|---|---|---|---|---|
| grid points in the peak bounds | 3 | 3 | 3 | 3 | 0.50 | 0.50 |
| non-zero points per trace in the peak | 2 | 2.5 | 2 | 3 | 0.35 | 0.53 |
| non-zero points per trace, bounds ± 5 s (13 grid points) | 5 | 5 | 4 | 6.5 | 0.47 | 0.56 |
| 1/K0 SD per trace in the peak | 0.0043 | 0.0033 | 0.0047 | 0.0027 | 0.62 | 0.46 |
| median 1/K0 step between points | 0.0070 | 0.0053 | 0.0077 | 0.0038 | 0.65 | 0.45 |
| `off_frac` in the peak | 0.22 | 0 | 0.28 | 0 | 0.67 | 0.48 |
| `off_frac`, bounds ± 5 s | 0.41 | 0.21 | 0.43 | 0.11 | 0.69 | 0.49 |

Reading:
- **Both explanations hold, but only IM jitter is specific to R.**
- **Sparsity** is a property of the acquisition. A scored peak spans 3 scan-grid points
  (1.94 s median bounds at a ~0.97 s cycle, matching DIA-NN's FWHM of 3.1 scans), for accepted
  targets too. Over ±5 s, R has as many non-zero points as abundance-matched accepted targets
  (5 against 5, AUC 0.47). Every co-elution statistic here rests on 3-6 points.
- **IM jitter** separates R from Aq and puts R at decoy level on every jitter statistic (AUC R
  vs Dm 0.45-0.49). A fifth of R's in-peak trace intensity comes from points more than 0.01
  from the fragment consensus 1/K0; for the median accepted target it is none. The trace build
  admits other ions inside the ±0.04 window, as L1a assumes.
- **The anchor matters for L1a.** Measured from `im_pred_cal` instead of the apex consensus,
  `off_frac` is 0.48 for accepted targets and 0.70 for R. The calibrated prediction is
  therefore too coarse to anchor a 0.01 choice. L1a should anchor on the candidate's own
  fragment consensus (apex 1/K0, or a running consensus), with `im_pred_cal` only as the
  window centre.
- Because the peaks are 3 points wide, a gain from L1a should appear as fewer off-mobility
  points, not as more points.

### D3: library quality against engine scoring

The search space is held fixed and only the library changes:
- DIA-NN's predicted library for the same FASTA and settings (`report-lib.predicted.speclib`
  is on disk). The user converts it to parquet with DIA-NN under their own licence; MuMDIA
  does not invoke DIA-NN.
- Import with `scripts/import_diann_lib.py` and complete with `scripts/augment_library.py`.
- Search with the P6 engine configuration. RT and IM stay per-run calibrated as in P6.

The same FASTA, digest and paired decoys give the same FDR population. The delta against
P6 therefore attributes the gap between library quality (MS2PIP `timsTOF2024`, DeepLC,
IM2Deep) and engine scoring.

Add a per-precursor spectral angle, MS2PIP against DIA-NN predicted intensities, for each
D1 category. DIA-NN's empirical library (`report-lib.parquet`, its 1% IDs only) is not a
fair search library, because it holds only confident precursors. Use it for the
spectral-angle comparison only.

### D3 result (2026-09-25)

**Setup.** DIA-NN 2.5.0 converted `report-lib.predicted.speclib` to parquet (the user ran
`diann-linux --lib ... --out-lib bis/d3/diann_predicted.parquet --gen-spec-lib`; 594,840
target precursors, 7.1M fragment rows, 29% of them at fragment charge 2). Imported with
`scripts/import_diann_lib.py`. The search spaces nearly coincide: every P6 target precursor
in the 400-1000 isolation range is in DIA-NN's library, and 99.99% of DIA-NN's are in P6's, so
`augment_library.py` was not needed. Both arms are restricted to the 594,756 shared
(peptidoform, charge) keys.

The P6 native decoys are reversed sequences with their own MS2PIP / DeepLC / IM2Deep
predictions. `make_reverse_decoys.py` copies the target's intensities, iRT and 1/K0 onto the
reversed sequence. That is a different null, so the DIA-NN arm is compared with a **control**
that uses P6's own targets and the same decoy script. All arms are full `mumdia run`s in
library mode with the P6 engine config and the P7 binary. `library_irt = auto` re-predicts
iRT with DeepLC in every arm, followed by the multi-head calibration, so the RT source is
the same everywhere. Libraries and outputs are in `/public/compomics2/Robbe/MuMDIA_data/bis/d3/`.

| arm | fragments | 1/K0 | precursors | peptides (range) | PGs | decoy fraction | seed anchors | `w_im` | wall |
|---|---|---|---|---|---|---|---|---|---|
| P6 base (native decoys) | MS2PIP `timsTOF2024` | IM2Deep | 11,037 | 9,507 (9,460-9,570) | 1,472 | 0.98-0.99% | | | |
| control | MS2PIP `timsTOF2024` | IM2Deep | 11,271 | 9,654 (9,630-9,680) | 1,526 | 0.98-0.99% | 3,490 | 0.034 | 4:14 |
| DIA-NN library | DIA-NN | DIA-NN | 12,540 | **10,455** (10,366-10,565), **+8.3%** | 1,573 | 0.98-0.99% | 3,914 | 0.041 | 4:30 |
| H1 | DIA-NN | IM2Deep | 12,735 | **10,695** (10,676-10,708), **+10.8%** | 1,568 | 0.98-0.99% | | 0.040 | 4:34 |
| H2 | MS2PIP | DIA-NN | 11,092 | 9,540 (9,454-9,614), -1.2% | 1,493 | 0.98-0.99% | | 0.033 | 4:32 |
| H3 (seed 0) | DIA-NN, charge-1 fragments only | IM2Deep | 11,427 | 9,813 (+1.6% against control seed 0) | 1,523 | 0.99% | | 0.036 | 3:59 |

Three seeds (0-2) except H3; percentages against the control; units of TIMS_ROADMAP.md section 2. Peak RSS was
7.5-7.8 GB in every library-mode run.

D1 (seed 0, best-row pairing), control -> DIA-NN library: accepted keys 10,186 -> 11,246
(+10.4%). Not extracted 694 -> 217, of which "IM outside window" 158 -> 19. Right peak, low
score 4,250 -> 3,871 (1,143 of them accepted). The accepted share per DIA-NN quintile moves
from 19.4 / 39.6 / 58.8 / 78.7 / 93.1% to 24.0 / 48.5 / 67.7 / 84.3 / 95.3%: the faint
precursors gain most.

Library spectral angle, MS2PIP against DIA-NN (square-root intensities over the union of
predicted fragments, missing = 0), per control D1 category: accepted 0.730, right peak with a
low score 0.676, wrong peak inside the window 0.742, outside 0.635, not extracted 0.39-0.54.
Per quintile, from faint to bright: 0.650 / 0.691 / 0.707 / 0.733 / 0.758.

Reading:
- **Library fragment intensities are worth +8.3% peptides on this file**, with the decoy
  fraction unchanged, the same search space and the same decoy construction. This is the
  largest single lever measured on this file since P6's gate change.
- **The gain is DIA-NN's charge-2 fragments.** Restricted to charge 1 (H3; 611 target/decoy
  pairs without a charge-1 fragment dropped), DIA-NN's intensities are +1.6% on one seed,
  within noise. The charge-1 intensities of the two predictors are therefore about equally
  useful. What MuMDIA lacks is charge-2 fragments: MS2PIP `timsTOF2024` predicts none, and
  P6 had to request charge 1 only (TIMS_ROADMAP.md, "P6 result").
- **The IM source does not matter, and IM2Deep is slightly better.** H1 keeps the gain with
  IM2Deep 1/K0 and is +2.3% over the DIA-NN arm (non-overlapping seed ranges). H2 does not
  gain with DIA-NN's 1/K0. The drop in "IM outside window" follows `w_im`, which rt-im-train sizes from
  the anchors: DIA-NN fragments give more anchors (3,914 against 3,490), fainter ones among
  them, and so a wider held-out window. The IM predictor itself is not the lever.
- The script-built decoys cost nothing (control 9,654 against native 9,507).
- The two libraries disagree most on the precursors MuMDIA loses (spectral angle 0.39-0.68
  against 0.73 on accepted keys). Library quality therefore explains part of D2's
  library-agreement family, and L1c moves up the order.
- No entrapment yet, and one acquisition only. DIA-NN's library is not a MuMDIA default in
  any case: the lever is a fragment predictor that gives good charge-2 fragments for
  timsTOF (L1c).

### D4: counterfactual replays (diagnostic only)

These replays use DIA-NN's answers for targets only. They break target/decoy
exchangeability, so they bound a lever's ceiling and make no FDR claim.
- **Apex oracle.** Rerun extract with `extract.retain_top_peaks: 5` (unscored sidecar) and
  count, for the 1,384 in-window wrong-peak precursors, how often the DIA-NN apex is among
  the top K. This is the ceiling of L2. Earlier AIF measurement: the correct peak was in
  the top five in 86-88% of cases (docs/18 A6).
- **RT-window oracle.** Centre the RT window on DIA-NN's RT for the shared precursors and
  count how many of the 980 out-of-window precursors are then accepted. This is the ceiling
  of L3.

### D4 result (2026-09-25)

**Apex oracle.** Extract of the P6 base with `extract.retain_top_peaks: 5`, P7 binary
(`/public/compomics2/Robbe/MuMDIA_data/bis/d4_top5`). `psms_extracted` is identical to the base
on all 24 base columns; the P7 binary adds its 5 v5 columns, null on v2 spectra. The sidecar
holds 2,517,835 peaks (500,748 of the 504,608 candidates have 5). Extract wall 30.5 s against
21-22 s, peak RSS 4.7 GB. `tims_loss.py --peaks` gives `oracle_rank`, the lowest sidecar rank
whose apex lies within 5 s of DIA-NN's RT.

| D1 category (best-row pairing) | n | DIA-NN apex in top 1 / 2 / 3 / 5 sidecar peaks |
|---|---|---|
| accepted | 10,138 | 92.0 / 96.4 / 97.1 / 97.3% |
| right peak, low score | 4,327 | 80.0 / 95.4 / 98.7 / 99.9% |
| wrong peak, DIA-NN RT inside window | 1,384 | 31.3 / 67.6 / 86.6 / 98.7% |
| wrong peak, DIA-NN RT outside window | 980 | 1.0 / 3.6 / 6.1 / 12.7% |

The sidecar ranks by count-profile area, which is not how extract selects its apex (evidence
rank with the RT prior). For 433 of the 1,384 in-window wrong-peak keys, the sidecar's rank-0
peak is DIA-NN's apex while the selected apex is another peak.

**L2 ceiling.** An emulation of `promote_top_peaks` on the sidecar. It enumerates the top K
peaks, drops the envelope holding the selected apex, requires >= 5 s separation and >= 10% of
the selected envelope's area, and keeps K-1 alternates (`extract.rs` promote block; the
matched-fragment floor is not emulated). Count: in-window wrong-peak keys whose DIA-NN apex
lies within 5 s of an alternate.

| K | alternate rows per candidate | DIA-NN apex among the alternates (of 1,384) |
|---|---|---|
| 2 | 0.75 | 783 (56.6%) |
| 3 | 1.45 | 1,096 (79.2%) |
| 5 | 2.85 | 1,301 (94.0%) |

This is a ceiling on candidacy, not on identifications. A recovered peak must still score
above threshold, and D2 shows that right peaks in this data often do not. K = 3 reaches most
of the ceiling at 2.45x the rows (about 1.24M) through features, compete and rescore.

**RT-window oracle** (`/public/compomics2/Robbe/MuMDIA_data/bis/d4_rtoracle`). In
`run_windows` the RT window of 992 library candidates was moved, at unchanged width, onto
DIA-NN's RT. These are DIA-NN's own peptidoforms, for the keys that are out-of-window under
either pairing. Their `rt_pred_cal` was set to DIA-NN's RT plus an error drawn from the
accepted group's calibrated-prediction residuals (seed 0), so the rows carry a realistic,
not a perfect, RT feature. Then the fast loop, P7 binary, seed 0. This uses DIA-NN answers
for targets and makes no FDR claim.

| | precursors (`precursor_q`) | peptides (`peptide_q_value`) | PGs | decoy fraction |
|---|---|---|---|---|
| P6 base, seed 0 | 11,136 | 9,570 | 1,468 | 0.98% |
| RT oracle, seed 0 | 11,318 | 9,765 (+2.0%) | 1,484 | 0.98% |

Movement of the 607 shifted keys that are out-of-window on the exact-peptidoform pairing:
219 accepted (36%), 244 right peak with a low score (40%), 129 still at a wrong peak, and 15
not extracted. D1 keys accepted (best-row pairing, `q_value` <= 0.01): 10,138 -> 10,322. On the
unshifted keys it was 10,128 -> 10,083, a single-seed reshuffle.

Reading:
- **L2 has the larger ceiling.** Up to about 1,100 in-window wrong-peak keys get their right
  peak as a scored row at K = 3.
- **L3's ceiling is small.** A perfect window for every out-of-window key adds about 220
  accepted keys (+2.0% peptides, single seed). The rest land at the right peak and then fail
  on score: the L1 problem again. RT refinement (L1d, L3) is worth doing only together with
  the L1 fixes.

### D5: why MS1 is missing (added after D2, low priority)

No MS1 monoisotopic peak is found at the apex for 33% of accepted keys, 50% of right-peak
rejects and 54% of decoys. Read-only check: for accepted keys without an MS1 peak, look up the
MS1 spectra near DIA-NN's m/z, RT and 1/K0. This separates three causes: the peak is absent;
it was merged into a neighbouring centroid (L1b); or the IM gate or `prec_tol_ppm` removed it.

### D5 result (2026-09-25)

`bench/tims_ms1_check.py` on the 10,138 D1 accepted keys of the P6 base (read-only; outputs in
`/public/compomics2/Robbe/MuMDIA_data/bis/d5/`). P6 does not IM-gate MS1 (`im_gate:
fragments`), so the gate is not the cause. Extract found an MS1 monoisotopic peak
(`ms1_mono` > 0) for 6,754 keys and none for 3,384.

| | MS1 found (6,754) | MS1 missing (3,384) |
|---|---|---|
| apex MS1 scan: nearest centroid to the mono m/z, any 1/K0, median abs / signed ppm | 4.9 / -0.8 | 39.6 / -34.2 |
| ... share within 20 ppm (the extract tolerance) | 100% | 0.1% |
| apex scan, nearest centroid within 0.03 of DIA-NN's 1/K0, within 20 ppm | 85.5% | 0.0% |
| ± 2 MS1 scans, same, within 20 ppm | 93.3% | 35.4% |
| DIA-NN `Ms1.Apex.Area` > 0 | 99.9% | 100% (median 40,558) |
| DIA-NN quintile, median | 4 | 3 |

MS1 found by charge: z2 64.2%, z3 76.2%, z4 79.4%.

Reading:
- **The MS1 peaks are lost in convert, not in extract.** DIA-NN measures a precursor MS1
  signal for every one of these keys, but MuMDIA's MS1 spectra have no centroid within 20 ppm
  of the monoisotopic m/z at the apex scan. The nearest centroid sits a median 34 ppm below
  it.
- **Likely cause: m/z single linkage across all TIMS scans.** `centroid_2d`
  (`stages/convert/tdf.rs`) links neighbouring TOF points within `tdf_mz_ppm` (10 ppm) over
  every scan of the frame before it splits by mobility gap. MS1 frames are dense (the msconvert
  baseline read a median 701,377 points per MS1 spectrum), so the chains can bridge
  neighbouring ions, and the resulting centroid's m/z is an
  intensity-weighted blend. This is the same merging the P7 width measurement showed (MS1
  width 1.6x the fragment width), now seen in m/z. It is inferred from the ppm shift, not yet
  shown on raw frames.
- The MS2 side is affected less, because its frames are sparser, but the same mechanism
  applies. That makes this a candidate cause of part of D2's library-agreement deficit too.
- **Next check (cheap).** A convert arm with `tdf_mz_ppm` 5, then the fast loop: MS1 found
  share, D5 table, MS2 peaks per spectrum, IDs. If the chains are the cause, a tighter
  linkage, or a split at m/z profile minima (the L1b upgrade path, applied in m/z), is the fix.

**D5 convert check (2026-09-25): `convert.tdf_mz_ppm` 5** (default 10). A full run of the P6
config, P7 binary; seeds 1-2 re-run `rescore`. An MS1-only arm re-ran the fast loop on the P6
run with the 5 ppm MS1 spectra and P6's MS2 spectra, seed, windows and masscal. Outputs in
`/public/compomics2/Robbe/MuMDIA_data/bis/d5/`.

| | P6 (10 ppm) | 5 ppm |
|---|---|---|
| MS1 centroids, total / median per spectrum | 41.5M / 26,574 | 102.6M / 58,541 |
| MS2 centroids, total / median per spectrum | 71.0M / 684 | 75.1M / 473 |
| seed confident PSMs / RT anchors | 3,587 / 3,370 | **615 / 606** |
| learned fragment tolerance (deviations) | 12.5 ppm (41,742) | 9.6 ppm (7,091) |
| MS1 found, D1 accepted keys | 66.6% | 80.5% (of that run's own accepted keys) |

| arm | precursors | peptides (range) | PGs | decoy fraction | wall |
|---|---|---|---|---|---|
| P6 base | 11,037 | 9,507 (9,460-9,570) | 1,472 | 0.98-0.99% | 13:17 |
| 5 ppm, MS1 and MS2 | 11,530 | **9,846 (9,815-9,875), +3.6%** | 1,493 | 0.98-0.99% | 13:17 |
| 5 ppm MS1 only (seed 0) | 11,156 | 9,618 (+0.5% against seed 0) | 1,475 | 0.99% | fast loop |

In the MS1-only arm, MS1 is found for 79.9% of P6's accepted keys (from 66.6%), but also for
64.2% of decoys (from 46.4%). The 5 ppm run's missing-MS1 keys still sit at a median of
34.5 ppm from the nearest centroid.

Reading (partly superseded by the follow-up below):
- **Recovering MS1 peaks adds no identifications** (+0.5%), because MS1 presence rises almost
  as much for decoys as for targets.
- **The gain is on the MS2 side:** +3.6% over 3 seeds, although the seed collapsed to 615
  confident PSMs. With fewer anchors the RT calibration and the learned tolerance rest on
  one sixth of the data. The seed collapse is unexplained. Fixing it could make the 5 ppm arm
  gain more, and it is the next thing to diagnose. Candidate causes: the per-spectrum peak
  distribution (the median falls from 684 to 473 while the p95 rises), and the seed's
  `top_n_peaks` 300 selection.
- Not measured: a sweep between 5 and 10 ppm.

**Follow-up: 5 ppm over-splits, and the seed needs the 10 ppm centroids (2026-09-25).**
- **Over-splitting.** In 200 spectra per level, pairs of centroids at the same 1/K0 (within
  0.005) and 4-10 ppm apart go from 1 (MS2) and 1,204 (MS1) at 10 ppm to 3,807 and 153,897 at
  5 ppm. The instrument resolves about 20-25 ppm FWHM, so such pairs are mostly one ion cut
  into pieces. The 2.5x MS1 count is therefore largely splitting, not separated ions. The
  missing-MS1 keys still sit about 34 ppm from the nearest centroid. D5's MS1 loss is not
  explained by chaining; `tdf_min_points` (2) on faint MS1 ions is the remaining candidate,
  unmeasured.
- **Seed collapse.** On the 5 ppm spectra the seed finds 615 confident PSMs at `top_n_peaks`
  300, 147 at 1,000 and 0 with all peaks; on the 10 ppm spectra, with the same library, 3,587.
  The split pieces act as noise in the index probe.
- **Hybrid (zero code, label-blind).** Seed, `run_windows` and mass calibration from the
  10 ppm conversion, extraction on the 5 ppm spectra. Fast loop; seeds 1-2 re-run `rescore`:

| arm | precursors | peptides (range) | PGs | decoy fraction |
|---|---|---|---|---|
| P6 base | 11,037 | 9,507 (9,460-9,570) | 1,472 | 0.98-0.99% |
| 5 ppm spectra, P6 seed/windows, tolerance 12.5 ppm (seed 0) | 11,540 | 9,921 (+3.7% against seed 0) | 1,476 | 0.99% |
| 5 ppm spectra, P6 seed/windows, tolerance 9.6 ppm | 11,766 | **10,054 (10,018-10,106), +5.8%** | 1,479 | 0.99% |
| L1c NCE 35 (full run) | 13,009 | 10,899 (10,811-10,981) | 1,580 | 0.98-0.99% |
| **L1c NCE 35 + 5 ppm spectra, tolerance 9.6 ppm** | **14,055** | **11,757 (11,707-11,828), +23.7% over P6, +7.9% over L1c** | **1,637** | 0.99% |

The 9.6 ppm tolerance is the one the collapsed 5 ppm seed learned (offset 0). The last row
takes the seed, windows and library from `bis/l1c/nce35`. MuMDIA / DIA-NN in peptides:
0.76x.

- The MS2 gain is real and stacks with L1c. Its mechanism is not settled. Over-splitting
  costs intensity per piece, yet the arm gains. The likely reading is that the 10 ppm chains
  also absorb neighbouring fragment ions 10-20 ppm away, and that the tighter 9.6 ppm
  tolerance then rejects them.
- The clean fix is not a smaller linkage distance. It is to split along m/z at profile
  minima (the L1b upgrade path applied in m/z), so one ion stays one centroid and two ions
  become two. Until then the hybrid needs two conversions: 10 ppm for the seed, 5 ppm for
  extraction.
- Needs entrapment before any claim; one acquisition.

**Hybrid entrapment (2026-09-25), L1c NCE 35 + 5 ppm spectra.** The fast loop
(`bis/d5/chain_spec.sh`) on the L1c entrapment run (`/public/local/MuMDIA_entrap/l1c/s0`: its
library, seed and `run_windows`), with the 5 ppm MS1 and MS2 spectra of `bis/d5/mzppm5/run`, the
9.6 ppm mass calibration and `l1c/config.run.json`. The spectra depend on the raw file only, so
no new conversion was needed. Seeds 1-2 re-run `rescore`; `bench/entrapment_fdp.py` at the
peptide level, `entrapment_ratio` 0.563430. Outputs in `/public/local/MuMDIA_entrap/l1c_hyb`.

| seed | real peptides at 1% | spike-in peptides | empirical FDP | decoy fraction |
|---|---|---|---|---|
| 0 | 10,741 | 67 | 0.361% | 0.99% |
| 1 | 10,748 | 66 | 0.355% | 0.99% |
| 2 | 10,713 | 73 | 0.393% | 0.99% |
| L1c NCE 35 (seeds 0-2) | 9,989-10,011 | 76-84 | 0.438-0.483% | 0.98-0.99% |
| P6 (seeds 0-2) | 8,612-8,790 | 53-56 | 0.355-0.378% | 0.98-0.99% |

- The FDP is below 1% in every seed, and below the L1c entrapment run in every seed. The FDR
  control holds for the hybrid on this acquisition.
- Real peptides (seed means): 10,734 against 10,003 for L1c (+7.3%) and 8,702 for P6 (+23.4%).
  This matches the E. coli-only search (+7.9% over L1c, +23.7% over P6).
- Spike-ins fall from 76-84 to 66-73 while real peptides rise. The extra identifications from
  the 5 ppm extraction are not bought with entrapment hits.
- Fast-loop cost: extract 66 s (11.1 GB), features 29 s, compete 20 s, rescore 150-158 s.
- Still one acquisition; the HYE set is not run for the hybrid. Nothing is promoted.

**m/z valley split (2026-09-25): one conversion replaces the hybrid.** New default-off keys
`convert.tdf_mz_valley` and `convert.tdf_mz_smooth_ppm` (`stages/convert/tdf.rs`,
`mz_valley_cuts`). After the mobility split, each cluster's m/z profile (summed intensity
per TOF index, triangular smoothing of half-width `tdf_mz_smooth_ppm`) is cut at every local
minimum below `tdf_mz_valley` times the smaller of the two humps beside it. The 10 ppm
linkage is kept. With the valley key at 0 the spectra are identical to the L1c 10 ppm
spectra (pyarrow table equality, MS1 and MS2), and the fast loop with the new binary on the
L1c NCE 35 run gives a `psms_scored` identical to that run's on score, q_value,
peptide_q_value and precursor_q (`p6/same.py`, 569,529 rows). Outputs in
`/public/local/MuMDIA_valley`.

TOF grid. One TOF index is about 7 ppm wide at m/z 200, 4.4 ppm at 500 and 2.8 ppm at 1,200
on this run (from the TDF acquisition range and digitizer samples, assuming m/z proportional
to the square of the TOF index). An ion of 20-25 ppm FWHM therefore spans only 4-7 TOF
indices. Two consequences:
- a 5 ppm linkage cannot join adjacent TOF indices below m/z of about 400, and breaks at any
  skipped index above it. This is the over-splitting of the D5 follow-up;
- a smoothing half-width of 4 ppm gives no weight to the neighbouring index below m/z of
  about 900, so it does not smooth at all.

Centroid-pair histogram (`bench/tims_centroid_pairs.py`, new): all pairs at the same 1/K0
(within 0.005) in 200 evenly spaced spectra per level, by m/z distance. It counts all pairs,
not only adjacent ones, so its counts are not comparable with the ad hoc count of the D5
follow-up.

| arm | MS1 centroids | MS1 pairs 4-10 / 10-25 ppm | MS2 centroids | MS2 pairs 4-10 / 10-25 ppm |
|---|---|---|---|---|
| 10 ppm (P6, L1c) | 41.5M | 7 / 46,452 | 71.0M | 0 / 6,891 |
| 5 ppm | 102.6M | 246,090 / 968,597 | 75.1M | 7,916 / 32,737 |
| valley 0.5, smoothing 4 ppm | 95.8M | 22,718 / 697,492 | 85.3M | 788 / 23,725 |
| **valley 0.5, smoothing 8 ppm** | **65.2M** | **7 / 89,212** | **75.4M** | **0 / 8,564** |
| valley 0.5, smoothing 12 ppm | 53.4M | 7 / 50,373 | 72.5M | 0 / 7,029 |
| valley 0.3, smoothing 8 ppm | 49.1M | 7 / 49,804 | 72.2M | 0 / 7,080 |

At smoothing 8 ppm or more the 4-10 ppm pairs fall back to the 10 ppm level, so single ions
are not cut. Valley 0.5 at 8 ppm separates the most centroids (MS2 +4.4M, close to the 5 ppm
count), and was the only arm searched.

Full L1c NCE 35 run (FASTA, `bis/l1c/config_nce35.json` plus the two keys), valley binary;
seeds 1-2 re-run `rescore`. Units of TIMS_ROADMAP.md section 2 (precursor_q / stripped
peptide_q_value / pg_q_value at 1%):

| arm | precursors | peptides (range) | PGs | decoy fraction |
|---|---|---|---|---|
| P6 base | 11,037 | 9,507 (9,460-9,570) | 1,472 | 0.98-0.99% |
| L1c NCE 35 | 13,009 | 10,899 (10,811-10,981) | 1,580 | 0.98-0.99% |
| L1c + 5 ppm hybrid (two conversions) | 14,055 | 11,757 (11,707-11,828) | 1,637 | 0.99% |
| **L1c + valley 0.5 / 8 ppm (one conversion)** | **14,318** | **11,990 (11,958-12,040)** | **1,661** | 0.99% |

Peptides: +26.1% over P6, +10.0% over L1c, +2.0% over the hybrid (3 seeds each; the ranges of
the valley arm and the hybrid do not overlap). MuMDIA / DIA-NN in peptides: 0.78x.

| | L1c NCE 35 | valley 0.5 / 8 ppm |
|---|---|---|
| seed confident PSMs / multi-head anchors | 4,793 / 4,333 | 5,200 / 4,685 |
| learned fragment tolerance (deviations) | 12.7 ppm (56,233) | 11.2 ppm (61,086), offset -0.17 |
| D1 accepted keys (seed 0) | 11,443 | 12,664 |
| D1 right peak, low score / wrong peak in window / out of window | 3,923 / 1,157 / 795 | 3,072 / 846 / 675 |
| D1 not extracted (RT out / IM out / in window) | 77 / 61 / 135 | 129 / 66 / 139 |
| MS1 found, D1 accepted keys (`tims_ms1_check.py`) | 66.5% (7,606) | **97.3% (12,318)** |
| ... median abs ppm of the found MS1 peak | 5.0 | 3.3 |
| MS1 found, all rank-0 extracted rows, targets / decoys | 48.0% / 47.0% | 76.2% / 75.0% |
| wall / peak RSS / convert stage | 14:42 / 8.66 GB / 14.5 s | 12:33 / 8.65 GB / 13.1 s |

Reading:
- **One conversion now does better than the hybrid,** and the seed does not collapse: it
  gains 8.5% confident PSMs over L1c. The split keeps one ion as one centroid (4-10 ppm pairs
  unchanged) and separates neighbours (10-25 ppm pairs +24% in MS2, +92% in MS1).
- **The D5 MS1 loss was chaining after all.** The missing-MS1 accepted keys fall from 3,837 to
  346, and the found MS1 peaks are closer to the monoisotopic m/z. D5's follow-up had ruled
  chaining out because 5 ppm did not help; the TOF grid above explains that: 5 ppm cannot
  link a single ion's indices, so it never tested a clean split. The `tdf_min_points`
  hypothesis (item 3) is therefore largely moot.
- MS1 presence rises as much for decoys as for targets over all extracted rows, as in D5.
  The gain comes from the accepted population, not from MS1 presence as a discriminant.
- The convert cost is unchanged. The histogram timings (13-44 s) were taken under a load
  average of 22 and are not usable.
- Entrapment and the HYE acquisition follow below ("Valley entrapment"; section 3, "Valley
  split on ProteoBench HYE diaPASEF"). Not yet done: 3 seeds on HYE, and a sweep of valley
  and smoothing on IDs (only one arm was searched). Nothing is promoted.

**Valley entrapment (2026-09-25).** A full run of `l1c/config.run.json` plus
`tdf_mz_valley` 0.5 and `tdf_mz_smooth_ppm` 8 on the E. coli file against E. coli + 1:1 human
(same setup as the L1c entrapment; `entrapment_ratio` 0.563430; peptide level). Wall 30:56,
peak RSS 23.7 GB. Seeds 1-2 re-run `rescore`. Outputs in `/public/local/MuMDIA_entrap/valley`.

| arm (seeds 0-2) | real peptides at 1% | spike-in peptides | empirical FDP | decoy fraction |
|---|---|---|---|---|
| **valley 0.5 / 8 ppm** | 11,490 / 11,429 / 11,463 | 89 / 90 / 91 | 0.445 / 0.452 / 0.456% | 0.99% |
| L1c + 5 ppm hybrid | 10,741 / 10,748 / 10,713 | 67 / 66 / 73 | 0.361 / 0.355 / 0.393% | 0.99% |
| L1c NCE 35 | 10,011 / 9,989 / 10,010 | 76 / 81 / 84 | 0.438 / 0.467 / 0.483% | 0.98-0.99% |
| P6 | 8,612-8,790 | 53-56 | 0.355-0.378% | 0.98-0.99% |

- The FDP is below 1% in every seed and inside the L1c range. FDR control holds for the
  valley split on this acquisition.
- Real peptides (seed means): 11,461, +14.6% over L1c (10,003), +6.8% over the hybrid
  (10,734), +31.7% over P6 (8,702). The gain over L1c is larger here than in the E. coli-only
  search (+10.0%).
- Spike-ins rise from 76-84 (L1c) to 89-91, in proportion to the real peptides, so the FDP
  stays at the L1c level. The hybrid's lower FDP is not kept.

**MS1 point floor (2026-09-25): `convert.tdf_min_points` 1** (default 2). The key also
affects MS2, so only the MS1 spectra of a `tdf_min_points` 1 conversion were swapped into the
fast loop (`chain_spec.sh` with `MS1=`); MS2, seed, windows, library and mass calibration are
those of the base run. Two bases: (a) L1c NCE 35 at the 10 ppm linkage, as D5 proposed; (b)
the valley run (valley 0.5 / 8 ppm, with `tdf_min_points` 1 in the MS1 conversion). The fast
loop reproduces its base exactly (see the identity check above), so the base runs are the
controls. `tims_ms1_check.py` on D1 tables built from each arm. Outputs in
`/public/local/MuMDIA_valley` (`fl_l1c_ms1min1`, `fl_v05_ms1min1`).

| arm | MS1 centroids | MS1 found, D1 accepted keys | missing keys: nearest centroid, signed median ppm | MS1 found, rank-0 extracted targets / decoys | peptides |
|---|---|---|---|---|---|
| (a) L1c, `min_points` 2 | 41.5M | 66.5% (7,606 of 11,443) | -34.4 | 48.0% / 47.0% | 10,899 (10,811-10,981) |
| (a) L1c, MS1 `min_points` 1 | 112.2M | 70.8% (8,310 of 11,745) | -30.8 | 62.4% / 61.8% | 10,926 (10,848-11,025) |
| (b) valley, `min_points` 2 | 65.2M | 97.3% (12,318 of 12,664) | -22.5 | 76.2% / 75.0% | 11,958 (seed 0) |
| (b) valley, MS1 `min_points` 1 | 136.0M | 97.6% (12,376 of 12,681) | -22.2 | 85.3% / 84.6% | 11,940 (seed 0) |

Peptides are stripped peptides at 1% `peptide_q_value`. Arm (a) has 3 seeds against the L1c
seeds (paired deltas +214, -133, -1; mean +0.25%). Arm (b) has seed 0 only: -0.2% against the
valley seed 0. Precursors (a): 13,112 against 13,009; PGs 1,585 against 1,580. The decoy
fraction is 0.99% in every arm.

Reading:
- **The point floor is not the cause of the D5 MS1 loss.** At the 10 ppm linkage,
  `min_points` 1 nearly triples the MS1 centroids but recovers MS1 for only 4 points more of
  the accepted keys. The keys that are still missing keep their nearest centroid about 31 ppm
  below the monoisotopic m/z, which is the chaining signature. The valley split recovers them
  (the previous subsection).
- It adds no identifications on either base, because MS1 presence rises for decoys almost
  exactly as for targets.
- `tdf_min_points` stays at 2. Item closed.

**Mobility valley split (L1b) and m/z valley sweep (2026-09-26).** New default-off keys
`convert.tdf_im_valley` and `convert.tdf_im_smooth_scans`: after the m/z valley split, each
piece is cut at valleys of its mobility profile (summed intensity per TIMS scan, triangular
smoothing in scans), with the same rule as the m/z split (`valley_cuts` in
`stages/convert/tdf.rs` now serves both axes). With every key off, and with only the m/z
split on, the spectra are identical to those of the previous binary (pyarrow table
equality). Outputs in `/public/local/MuMDIA_imvalley`.

`bench/tims_centroid_pairs.py` gained a mobility histogram: pairs within 3 ppm in m/z,
binned by 1/K0 distance (0-0.005 / 0.005-0.01 / 0.01-0.02 / 0.02-0.05), plus the P7 width
percentiles (conversions with `tdf_im_width`). A mobility peak is about 0.02 wide (FWHM),
so pairs below 0.01 are one ion cut in two. All arms on top of the m/z split 0.5 / 8 ppm:

| mobility split | MS1 centroids | MS1 pairs by 1/K0 distance | MS1 width p50 / p95 | MS2 centroids | MS2 pairs by 1/K0 distance | MS2 width p50 / p95 |
|---|---|---|---|---|---|---|
| off | 65.2M | 0 / 0 / 0 / 31,317 | 0.0184 / 0.0996 | 75.4M | 0 / 0 / 0 / 1,863 | 0.0084 / 0.0256 |
| 0.5, smoothing 2 scans | 143.2M | 9,176 / 193,938 / 726,598 / 1,954,234 | 0.0088 / 0.0497 | 93.1M | 123 / 2,836 / 10,250 / 21,937 | 0.0068 / 0.0192 |
| 0.5, smoothing 4 scans | 120.2M | 0 / 6,423 / 155,554 / 1,018,031 | 0.0113 / 0.0520 | 86.5M | 0 / 149 / 2,877 / 14,239 | 0.0075 / 0.0197 |
| **0.5, smoothing 8 scans** | **99.2M** | **0 / 2 / 7,306 / 384,733** | **0.0140 / 0.0547** | **81.5M** | **0 / 0 / 238 / 7,902** | **0.0080 / 0.0207** |

Smoothing 2 and 4 scans over-split; 8 scans does not, and halves the MS1 width p95.

Full L1c NCE 35 runs (one after the other, valley binary with the mobility keys; seeds 1-2
re-run `rescore`). Units of TIMS_ROADMAP.md section 2 (precursor_q / stripped
peptide_q_value / pg_q_value at 1%):

| arm | precursors | peptides (range) | PGs | decoy fraction | seed confident / anchors | learned tolerance |
|---|---|---|---|---|---|---|
| m/z 0.5 / 8 ppm (previous subsection) | 14,318 | 11,990 (11,958-12,040) | 1,661 | 0.99% | 5,200 / 4,685 | 11.2 ppm |
| m/z 0.5 / 8 ppm + mobility 0.5 / 8 scans | 14,735 | **12,185 (12,163-12,204), +1.6%** | 1,672 | 0.98-0.99% | **3,331 / 2,995** | 9.2 ppm |
| m/z 0.3 / 8 ppm | 13,402 | 11,262 (11,222-11,304), -6.1% | 1,607 | 0.99% | 5,003 / 4,501 | 11.9 ppm |
| m/z 0.7 / 8 ppm | 14,689 | **12,177 (12,085-12,229), +1.6%** | 1,711 | 0.98-0.99% | 4,962 / 4,483 | 10.9 ppm |
| m/z 0.5 / 12 ppm | 13,649 | 11,462 (11,429-11,482), -4.4% | 1,593 | 0.98-0.99% | 5,127 / 4,618 | 11.8 ppm |

Deltas are against the m/z 0.5 / 8 ppm run. Against L1c (10,899): +11.8% for both +1.6% arms;
against P6 (9,507): +28%. Wall 12:17-12:23 per run.

Which level carries the mobility gain (fast loop on the m/z 0.5 / 8 ppm run, its seed,
windows, library and mass calibration, seed 0; control 11,958 peptides):

| spectra swapped in | precursors | peptides | PGs |
|---|---|---|---|
| MS1 with mobility split, MS2 m/z only | 14,323 | 11,983 (+0.2%) | 1,655 |
| MS1 and MS2 with mobility split | 15,100 | **12,535 (+4.8%)** | 1,671 |

Reading:
- **The mobility split helps through MS2, and the seed caps it.** With the seed from the
  m/z-only spectra the gain is +4.8% (seed 0); in the full run, whose seed runs on the
  mobility-split spectra, it is +1.9% (seed 0). The seed loses 36% of its confident PSMs
  although MS2 gains only 8% more centroids.
- **Likely cause: the seed counts peaks, not fragments.** `search_seed` adds one match per
  observed peak within tolerance of a library fragment (`search_seed.rs`, the
  `page_search` loop), with no dedup per fragment. A diaPASEF slot spectrum merges all its
  scans, so two mobility pieces of one fragment ion sit at the same m/z and both count, and
  `ln(matched!)` in the hyperscore rewards the duplicate. The same mechanism would explain
  the 5 ppm seed collapse. Not yet tested.
- **The m/z sweep is monotone in how much it splits:** 0.3 and a wider smoothing lose, 0.7
  gains. The 0.7 arm keeps the 4-10 ppm bin clean (MS1 33, MS2 0) but its 10-25 ppm pairs
  rise 3.5x in MS1 and 1.6x in MS2 (313,395 and 13,770), so part of its gain may be mild
  over-splitting. No entrapment on either +1.6% arm.
- MS1 is not where the mobility split pays, although its merging was the motivation (P7).
  The P7 shape features were not re-tested yet.

**Seed: count fragments, not peaks (2026-09-26).** New default-off key
`search_seed.unique_fragment_matches` (fragindex matcher; `SeedScratch::unique` in
`matchers/fragindex.rs`). Per spectrum, each (candidate, predicted fragment) counts at most
once, with the intensity of its most intense matching peak, instead of once per matching
peak. With the key off the seed table is identical to the previous binary's (pyarrow table
equality on the mobility-split spectra); nothing downstream of the seed changed. Outputs in
`/public/local/MuMDIA_useed`.

Seed only (standalone `search-seed`, same library), confident seed PSMs:

| spectra | key off | key on |
|---|---|---|
| m/z 0.5 / 8 ppm + mobility 0.5 / 8 scans | 3,331 | **5,534** |
| m/z 0.5 / 8 ppm | 5,200 | 5,446 |
| m/z 0.7 / 8 ppm | 4,962 | 5,449 |

Full L1c NCE 35 runs with the key on (one after the other; seeds 1-2 re-run `rescore`; units
of TIMS_ROADMAP.md section 2):

| arm | precursors | peptides (range) | PGs | decoy fraction | seed confident / anchors | learned tolerance |
|---|---|---|---|---|---|---|
| m/z 0.5 / 8 ppm, key off (reference) | 14,318 | 11,990 (11,958-12,040) | 1,661 | 0.99% | 5,200 / 4,685 | 11.2 ppm |
| m/z 0.5 / 8 ppm + mobility, key off | 14,735 | 12,185 (12,163-12,204) | 1,672 | 0.98-0.99% | 3,331 / 2,995 | 9.2 ppm |
| m/z 0.5 / 8 ppm, key on | 14,575 | 12,047 (12,006-12,084), +0.5% | 1,688 | 0.98-0.99% | 5,446 / 4,905 | 11.1 ppm |
| **m/z 0.5 / 8 ppm + mobility, key on** | **15,153** | **12,451 (12,387-12,533), +3.8%** | **1,689** | 0.98-0.99% | **5,534 / 5,000** | 10.1 ppm |
| m/z 0.7 / 8 ppm + mobility, key on | 14,985 | 12,464 (12,391-12,516), +4.0% | 1,695 | 0.98-0.99% | 5,558 / 5,028 | 9.7 ppm |

Deltas are against the reference row. The best single-conversion arm (m/z 0.5 + mobility +
key on) is +14.2% peptides over L1c (10,899) and +31.0% over P6 (9,507). MuMDIA / DIA-NN in
peptides: 0.81x. Wall 12:13-12:26 per run.

Reading:
- **The duplicate counting was the seed's problem.** With the key on the mobility-split
  spectra give the most confident seed PSMs of any conversion (5,534), and the full run
  recovers the gain the fast loop with the m/z-only seed showed (+3.8% over 3 seeds, against
  +4.8% on seed 0 there).
- On spectra without the mobility split the key is neutral (+0.5%, the ranges overlap), as
  expected when few duplicates exist.
- Once the mobility split and the key are on, m/z valley 0.7 adds nothing over 0.5 (12,464
  against 12,451). 0.5 splits less and stays the choice.
- Not validated: entrapment and HYE for the mobility split and the seed key. Nothing is
  promoted.

**L1d prototype: refit the RT and IM calibration on pass-1 identifications (2026-09-26).**
Zero code. Pass 1 is the best arm above (`MuMDIA_useed/run_im_u`, seed 0). Its accepted
targets at PSM `q_value` 1% (15,207 rows, 12,425 peptides) become a pseudo-seed table: apex
RT as `observed_rt`, extract's `apex_im` at the selected peak as `observed_im`, `q_value` as
`spectrum_q`. `rt-im-train` is re-run on it against the same multi-head library. Pass 2 is
the fast loop (`chain_spec.sh`) with the new `run_windows`. It keeps the run's spectra,
library, mass calibration and the ORIGINAL seed table, which `features` reads, so pass-1
scores cannot enter the features. The multi-head calibration was not refitted. Outputs in
`/public/local/MuMDIA_l1d`.

| | seed anchors (pass 1) | pass-1 accepted IDs |
|---|---|---|
| RT anchors (peptides) | 4,645 | 12,425 |
| RT half-window `w_rt` (in-sample p95) | 16.9 s | 8.0 s |
| in-sample RT residual, median | 2.66 s | 2.11 s |
| IM anchors / `w_im` (held-out p95) | 5,534 / 0.039 | 15,207 / 0.029 |
| held-out IM residual, median | 0.0101 | 0.0098 |

The new windows are narrower because of selection: pass-1 IDs can only have been found
inside the pass-1 windows, so their residuals are truncated at the old window edge.
Iterating would shrink the windows further. Two pass-2 arms: A1 uses the refit windows as
they are; A2 uses the refit centres (`rt_pred_cal`, `im_pred_cal`) with the old per-candidate
half-widths. The RT centres moved by a median of 1.65 s (p95 8.3 s), the IM centres by 0.0011.

| arm | precursors | peptides (range) | PGs | decoy fraction |
|---|---|---|---|---|
| pass 1 (best arm, 3 seeds) | 15,153 | 12,451 (12,387-12,533) | 1,689 | 0.98-0.99% |
| A1, refit windows (seed 0) | 14,409 | 12,035 (-4.0% against seed 0) | 1,685 | 0.99% |
| **A2, refit centres, old widths (3 seeds)** | **15,320** | **12,641 (12,614-12,666), +1.5%** | **1,697** | 0.99% |

A2 paired seed deltas: +109, +181, +279 peptides. Against L1c (10,899) A2 is +16.0%, against
P6 (9,507) +33.0%; MuMDIA / DIA-NN in peptides 0.82x. Pass-2 cost: extract 31 s, features 11 s,
compete 7 s, rescore 113 s.

Reading:
- **Better-centred windows gain; narrower windows lose.** The refit centres from 2.7x more
  anchors add 1.5%. The truncated widths cost 4.0%, so pass-1 IDs must not size the windows.
- A2 does not address the out-of-window group directly, since the widths are unchanged.
  `never_extracted` falls from 106 to 77 (seed 0).
- Leakage risk: the centres are fitted on targets accepted by target-decoy competition. The
  LOESS is one smooth curve over 12k anchors, so a single target's own apex barely moves its
  prediction, but this is not zero. Needs entrapment before any claim. Refitting the 80-head
  RT calibration on pass-1 IDs (variant B) has a larger risk of this kind and would need
  cross-fitting.
- Not tried: refitting the mass tolerance on pass-1 IDs.

**L1d variant B: refit the 80-head RT calibration on pass-1 IDs, cross-fitted
(2026-09-26).** The multi-head worker (`scripts/deeplc_finetune.py --multihead 80`, as the
engine calls it) is run on the pass-1 pseudo-seed instead of the seed. Cross-fitting: the
anchors are split into two folds by `base_peptide_id` (bit 1: the digest numbers
target/decoy pairs in steps of 2, so every id is even and parity would put everything in
one fold; a target and its decoy share the id and so the fold). Each fold gets its own fit
(6,733 and 6,631 anchor peptidoforms), and every library candidate, target or decoy, takes
the prediction of the fit on the OTHER fold. No anchor is then predicted by a model that saw
it, and targets and decoys are treated alike. `rt-im-train` then refits the LOESS and IM on
the pass-1 pseudo-seed, and the A2 rule is applied (new centres, old half-widths). For
scale, an in-sample arm fits all 13,364 anchors at once. Fast loop as for A2.

| | pass 1 (seed anchors) | A2 (pass-1 LOESS) | B cross-fitted | B in-sample |
|---|---|---|---|---|
| RT residual after LOESS, median (on the 12,425 pass-1 anchors) | 2.66 s (on 4,645 seed anchors) | 2.11 s | 1.43 s | 1.41 s |
| RT centre shift against pass 1, median | | 1.65 s | 3.67 s | 3.64 s |

| arm | precursors | peptides (range) | PGs | decoy fraction |
|---|---|---|---|---|
| pass 1 (3 seeds) | 15,153 | 12,451 (12,387-12,533) | 1,689 | 0.98-0.99% |
| A2 (3 seeds) | 15,320 | 12,641 (12,614-12,666) | 1,697 | 0.99% |
| **B cross-fitted (3 seeds)** | **15,774** | **12,919 (12,913-12,929), +3.8% over pass 1, +2.2% over A2** | **1,716** | 0.99% |
| B in-sample (seed 0) | 15,646 | 12,871 | 1,705 | 0.99% |

B cross-fitted paired seed deltas against pass 1: +383, +480, +542 peptides. Against L1c
(10,899): +18.5%; against P6 (9,507): +35.9%; MuMDIA / DIA-NN in peptides 0.84x.
`never_extracted` 62 (pass 1: 106).

Reading:
- **Refitting the multi-head RT model on pass-1 IDs is the larger part of L1d:** +2.2% on top
  of the LOESS refit. The RT residual falls by a third against A2.
- **No sign of leakage from the multi-head fit:** the cross-fitted and in-sample fits give
  the same residual (1.43 against 1.41 s) and the same IDs (12,916 against 12,871 on seed 0,
  cross-fitted higher). The 80-head ridge does not memorise its anchors at this size. The
  LOESS on top is still fitted in-sample; entrapment decides.

**L1d variant C: refit the fragment mass calibration on pass-1 IDs (2026-09-26).** Fragment
ppm deviations of the pass-1 accepted targets from the chromatogram table (`frag_obs_mz`,
the raw peak m/z; extract offset-corrects only its query m/z). 24% of the rows have a
deviation of exactly 0: every zero-intensity trace, plus fragments whose `frag_obs_mz`
appears to default to the theoretical m/z. These were excluded. On the other 173,604
fragments: offset -0.41 ppm (seed: -0.14), MAD 1.35 ppm, tolerance 7.99 ppm by the seed's
rule (`frag_tol_mad_k` 4; seed: 10.11 ppm). Fast loop on top of B cross-fitted, seed 0:

| arm | precursors | peptides | PGs | decoy fraction |
|---|---|---|---|---|
| B cross-fitted | 15,820 | 12,916 | 1,721 | 0.99% |
| C: offset -0.41, tolerance 7.99 ppm | 15,328 | 12,566 (-2.7%) | 1,684 | 0.99% |
| C2: offset -0.41, tolerance 10.11 ppm | 15,723 | 12,883 (-0.3%) | 1,705 | 0.99% |

- The refit offset is neutral; the tighter tolerance loses 2.7%. The accepted IDs' fragments
  are a biased sample (pass 1 extracted within 10.1 ppm, and accepted IDs have strong
  fragments), so their MAD underestimates the error of the weak fragments that the
  identifications also need. The mass calibration stays as the seed sets it.

**Entrapment: mobility split + seed key (pass 1) and L1d variant B (pass 2) (2026-09-27).**
One full run of `l1c/config.run.json` plus m/z valley 0.5 / 8 ppm, mobility valley 0.5 / 8
scans and `search_seed.unique_fragment_matches` on the E. coli file against E. coli + 1:1
human (`entrapment_ratio` 0.563430; peptide level). Wall 30:53, peak RSS 23.7 GB. Pass 2 is
the L1d variant B recipe on that run, as on the E. coli-only search: pseudo-seed from its
pass-1 accepted targets (14,247 rows, entrapment hits included, as the method would do on
real data), cross-fitted 80-head fit, LOESS refit, new centres with old widths, fast loop.
Seeds 1-2 re-run `rescore` in both passes. Scripts `make_seed.py` and `prep_arm.py` in
`/public/local/MuMDIA_l1d`; outputs in `/public/local/MuMDIA_entrap/l1d`.

| arm (seeds 0-2) | real peptides at 1% | spike-in peptides | empirical FDP | decoy fraction |
|---|---|---|---|---|
| **pass 1: m/z + mobility split, seed key** | 11,697 / 11,647 / 11,727 | 83 / 82 / 82 | 0.408 / 0.405 / 0.403% | 0.99% |
| **pass 2: + L1d variant B** | 11,985 / 12,057 / 11,924 | 86 / 92 / 91 | 0.413 / 0.438 / 0.438% | 0.99% |
| m/z valley only (earlier) | 11,490 / 11,429 / 11,463 | 89 / 90 / 91 | 0.445 / 0.452 / 0.456% | 0.99% |
| L1c NCE 35 | 10,011 / 9,989 / 10,010 | 76 / 81 / 84 | 0.438 / 0.467 / 0.483% | 0.98-0.99% |
| P6 | 8,612-8,790 | 53-56 | 0.355-0.378% | 0.98-0.99% |

- The FDP is below 1% in every seed of both passes. FDR control holds for the mobility split
  with the seed key and for the L1d refit on this acquisition.
- Real peptides (seed means): pass 1 11,690, +2.0% over the m/z valley alone (E. coli-only
  search: +3.8%), with a lower FDP (0.41% against 0.45%). Pass 2 11,989, +2.6% over pass 1
  (E. coli-only: +3.8%), +19.9% over L1c (10,003), +37.8% over P6 (8,702).
- Pass 2 raises the spike-ins by 9% for 2.6% more real peptides, so its FDP rises from 0.41%
  to 0.43%. That is inside the L1c and m/z valley range and far below 1%, but the refit is
  not free of extra false hits. The LOESS on the pass-1 IDs is fitted in-sample, and the
  pseudo-seed contains the pass-1 entrapment hits; either could contribute. Worth watching
  on the next acquisition.

**L1d in the engine: `rt_im_train.refit` (2026-09-27).** Variant B is now a default-off
key of `run` and ungrouped `run-experiment` (`stages/im_rt_refit.rs`, docs/08 section 8).
The recipe is the prototype's: pseudo-seed from the pass-1 targets at PSM `q_value` at
most `q_train` (`run_psm_q` per run under `run-experiment`), the cross-fitted 80-head fit
when multi-head calibration is active, the rt-im-train refit, new centres with the pass-1
half-widths, then extract, features with the original seed, compete and rescore. Pass-1
artifacts go to `pass1/`. The fold is the parity of a base peptide's rank among the
library's distinct `base_peptide_id` values, which equals the prototype's
`(id >> 1) & 1` on a native digest and is the id parity on an imported library. Under
`run-experiment` with `rt_library_scope = first_run_only` only the first run refits the
library. Verified on the E. coli file with the best-arm configuration (`config_run_im_u`),
seed 0, runs one after the other on the same host; outputs in `/public/local/MuMDIA_refit`:

| check | result |
|---|---|
| key off: `psms_scored` against `run_im_u` (`same.py`) | identical, 578,683 rows |
| key on: `pass1/psms_scored` against `run_im_u` | identical, 578,683 rows |
| pseudo-seed and both fold tables against the prototype | identical (15,207 / 7,650 / 7,557 rows) |
| cross-fitted library and pass-2 `run_windows` against the prototype | identical (1,333,950 rows) |
| key on: `psms_scored` against the prototype pass 2 (`fl_Bcf`) | identical, 546,848 rows |

Counts with the key on (seed 0; precursors on `precursor_q`, peptides on stripped
`peptide_q_value`, protein groups on `pg_q_value`, all at 1%): 15,820 precursors, 12,916
peptides, 1,721 protein groups, decoy fraction 0.99%, the prototype's numbers exactly.
Wall 17:23 against 14:19 with the key off (+21%), peak RSS 8.9 against 8.7 GB. The
fixture smoke passes, and the refit paths of `run` and `run-experiment` complete on it.

**L1d on HYE diaPASEF: the multi-head refit is unstable (2026-09-27).** One six-run
`mumdia run` (dispatched to `run-experiment`) of the best arm with `rt_im_train.refit`: the
ProteoBench HYE diaPASEF config with m/z valley 0.5 / 8 ppm, mobility valley 0.5 / 8 scans
and `search_seed.unique_fragment_matches`. Wall 8:10:32, peak RSS 92.4 GB. Outputs in
`/public/local/ProteoBench/HYE_diaPASEF_mumdia/l1d`; ProteoBench inputs
`proteobench_pass1/custom_input.tsv` (pass 1, per-run quant gated on the pooled `q_value`) and
`proteobench_pass2/custom_input.tsv`. The pass-2 file is not a valid L1d measurement.

| run | target PSMs at `run_psm_q` 1%, pass 1 | pass 2 | in-sample LOESS residual, pass 1 / pass 2 |
|---|---|---|---|
| A01 | 63,307 | 66,535 | 7.5 / 4.4 s |
| A02 | 63,632 | 57,731 | 7.7 / 9.1 s |
| A03 | 64,522 | 66,089 | 7.5 / 6.9 s |
| B01 | 63,670 | 64,638 | 7.4 / 8.5 s |
| B02 | 64,331 | 68,320 | 7.5 / 3.9 s |
| B03 | 64,362 | 28,334 | 7.7 / 50.9 s |

Cause (notebook `refit_instability/refit_instability.ipynb` in the output directory, on the
real tables):
- The cross-fitted 80-head refit gives some library rows absurd iRTs (-1.9e5 to 3.8e5 s on a
  3,500 s gradient). The two fold fits disagree by more than 300 s on 733,710 of 14,731,188 HYE
  rows (5.0%), against 1,023 of 1,333,950 on E. coli (0.08%).
- The absurd values come from the per-head spline in DeepLC's `MultiHeadRidgeCalibration`
  (`SplineTransformerCalibration`, degree 4, `n / 500 + 5` evenly spaced knots, no penalty).
  At 29,419 anchors (63 knots) one head's spline reaches 2e5 s at the sparse top end of its
  anchor range; at 11,357 anchors (27 knots) it stays inside the gradient.
- The ridge that combines the heads selects its penalty by leave-one-out and chose 0.001, the
  smallest value on its grid, for the 29,419-anchor fold fit (pass-1 fit on 11,357 seed
  anchors: 1e5). Over subsets of the fold anchors the choice is erratic (2,000: 1e5; 5,000:
  1e2; 11,357: 1e-3; 20,000: 1e3), and the share of a 10,000-peptidoform library sample outside
  0-7,200 s rises from 0 to 2.8%.
- In rt-im-train, 11 of 64,362 B03 anchors carry such values. The LOESS grid spans the anchor
  iRT range evenly, so the gradient falls into one or two grid cells, and the curve is offset by
  a median 51 s against a 38.7 s median half-width.

Fix 1, `rt_im_train.robust_calibration` (default false; docs/08 section 8): outlying anchors are
removed before the fit. Standalone rt-im-train on the six pass-2 pseudo-seeds: in-sample
residuals 3.7 to 4.0 s in every run, 362 to 489 anchors removed (0.6-0.8%), linear slope about
1.00 (0.03 to 0.13 without it); key off identical to the run's windows.
The absurd library iRTs themselves remain, for 1.95% of target and 1.97% of decoy rows of the
cross-fitted library (pass 1: 0.001%), so the loss is symmetric and does not bias the decoy
estimate; only 11 of 64,362 B03 pass-1 identifications are among them. A change in the DeepLC
calibration (fix 2) would remove them; not started.

Pass 2 with fix 1 (2026-09-28): the pass-2 chain rerun from the pass-1 artifacts of the same
experiment, with the robust rt-im-train refit, the new centres and the pass-1 half-widths, then
extract, features (original seed), compete, one pooled rescore, split by `source` and per-run
quant gated on the pooled `q_value`. Scripts and outputs in `l1d/pass2_robust`; ProteoBench input
`l1d/pass2_robust/proteobench/custom_input.tsv`. Here the robust fit applies to the pass-2 refit
only; an engine run with the key on also applies it to the pass-1 fit, whose anchors were clean
(7.5 s residual). One `nn_torch` seed, experiment-wide q columns at 1%:

| arm | precursors (`precursor_q`) | peptides (`peptide_q_value`) | PGs (`pg_q_value`) | decoy fraction | target PSMs per run at `run_psm_q` 1% |
|---|---|---|---|---|---|
| m/z valley only (earlier) | 74,775 | 67,768 | 9,814 | 0.99% | 57,717-59,041 |
| pass 1: m/z and mobility valley, seed key | 81,265 | 73,169 (+8.0%) | 10,303 | 0.99% | 63,307-64,522 |
| **pass 2 with fix 1** | **86,048** | **77,734 (+6.2% over pass 1, +14.7% over m/z valley)** | **10,657** | 0.99% | 67,573-69,288 |

Per-run ions in the ProteoBench input: 67,556 to 69,164 (pass 1: 63,349 to 64,634), even over
the six runs. Wall of the pass-2 chain: extract 8-9 min, features 4-5 min, compete 1.6 min per
run, pooled rescore 56 min. On E. coli L1d gave +3.8% in peptides (3 seeds), here +6.2% (one
seed, no entrapment on this set).

## 3. Levers, ranked by the population they address

Each lever is:
- behind a key that defaults to off;
- bit-identical when off;
- measured against the P6 base with 3 seeds and reported as D1 category movement plus the
  decoy fraction.

### L1: right peak, low score (4,327 precursors, 65% of rejects)

**L1a. IM-consistent fragment peak choice.**
- Current behaviour, at a scan:
  - extract keeps, per (scan, fragment), the most intense hit (the scan grouping in the
    `per_candidate` closure of `stages/extract.rs`);
  - the apex IM post-pass takes the peak nearest in m/z
    (`search_seed::frag_peak_indices`).

  Both admit a brighter co-isolated ion anywhere inside the ±0.04 IM window.
- Proposed: choose among in-tolerance peaks by a joint m/z and 1/K0 distance to the
  candidate's calibrated 1/K0 (`im_pred_cal`), or to the running consensus of its
  fragments.
- Scope (widened after D2): the per-(scan, fragment) choice in the trace build as well as
  the apex post-pass. D2 shows two failures: library agreement, which tracks the apex 1/K0
  error, and trace co-elution, which does not. Only the trace build can fix the second.
  D2b decides whether IM jitter causes it.
- Target: the 3x apex IM error of this group, and its decoy-level co-elution.
- Measure: the co-elution family on R (`frag_ref_corr_mean_full`, `coel_clean`,
  `coelution_mean`), the apex 1/K0 error per D1 category, IDs, and the decoy fraction.
- Cost: small; the gate code already carries the window per candidate.

**L1a prototype result (2026-09-25).** Zero-code version: `run_windows` copies whose IM
window is each candidate's P6 `apex_im` (the fragment consensus at the extraction apex) ± δ,
for the 504,603 candidates that have one (the original half-width is 0.034; `im_pred_cal`
unchanged). The rule is label-blind. Fast loop, P7 binary, seed 0, units of
TIMS_ROADMAP.md section 2. Outputs in `/public/compomics2/Robbe/MuMDIA_data/bis/l1a/`.

| δ | precursors | peptides | PGs | decoy fraction | extract accepted | never extracted (peptides) | D1 accepted keys | extract / rescore (s) |
|---|---|---|---|---|---|---|---|---|
| base (± 0.034 around `im_pred_cal`) | 11,136 | 9,570 | 1,468 | 0.98% | 504,608 | 203 | 10,138 | 20.7 / 112.6 |
| 0.008 | 10,154 | 8,806 (-8.0%) | 1,470 | 0.99% | 285,871 | 1,148 | 9,289 | 15.8 / 63.0 |
| 0.012 | 10,936 | 9,433 (-1.4%) | 1,517 | 0.99% | 375,353 | 610 | 9,941 | 17.9 / 88.6 |
| 0.020 | 11,269 | 9,701 (+1.4%) | 1,482 | 0.99% | 453,246 | 316 | 10,235 | 20.3 / 101.4 |

At δ 0.012 the narrow window moved 561 right-peak rejects to accepted, but it moved 618
accepted keys to "right peak, low score" and 95 accepted plus 351 right-peak keys to "not
extracted". On the D2 populations (D2b statistics, bounds ± 5 s) it removed the jitter by
construction: R's median 1/K0 step fell from 0.0070 to 0.0052 and `off_frac` to 0. **R's
evidence did not move**: `frag_ref_corr_mean_full` 0.234 (decoys 0.221; base 0.231),
`spectral_angle` 0.355 (decoys 0.303; base 0.367), `coel_clean` 0.036 (base 0.065).

Reading:
- The off-mobility trace points are not what makes R's traces fail to co-elute, at least
  not the part a window around the fragment consensus can remove. D2b's jitter is a
  correlate of R's weak signal, not its cause. The hard window also strips real fragment
  signal, which is why extraction losses dominate at 0.008.
- R's consensus 1/K0 may itself come from a contaminating ion (its apex 1/K0 error is 3x
  that of accepted keys). In that case an anchor on the fragment consensus cannot help, and
  an anchor on `im_pred_cal` is too coarse (D2b). An engine version of L1a (nearest-1/K0 peak
  choice instead of a hard window) would face the same anchor problem. It is **not
  recommended** before the library question (D3) is settled.

**L1f. The existing interference levers in extract (added after D2).** These are all
default-off and listed as "not measured" in docs/09 (default-off knob index):
- the spectrum-centric NNLS demix (`emit_demix_features`; `peak_claim: CoelutionDemix`);
- solver-free shadow subtraction (`peak_claim: CoelutionShadow`);
- two-pass contested-peak accounting (`emit_contested_features`).

Their feature columns are constant 0 in the P6 base. The features-only keys are
non-destructive; the `peak_claim` modes rewrite extracted intensity. They were written for
3D chimeric DIA, so the first step is to read whether their co-isolation sets and shared
peaks respect the IM window. Then fast-loop arms, measured as for L1a. No new code unless the
IM check fails.

**L1f result (2026-09-25).** Code reading first: the demix (`demix_solve_scan` in
`stages/extract.rs`) admits a peak as a candidate's claimant only inside that candidate's IM
window (`ImWin::rejects`), as the other two-pass paths do. It is therefore IM-gated, but it
does not use 1/K0 to separate candidates whose windows overlap. Fast loop on the P6 base, P7
binary, seed 0, one arm after the other (`/public/compomics2/Robbe/MuMDIA_data/bis/l1f/`):

| arm | precursors | peptides | PGs | decoy fraction | extract wall |
|---|---|---|---|---|---|
| base | 11,136 | 9,570 | 1,468 | 0.98% | 20.7 s |
| `emit_demix_features` + `emit_contested_features` | 11,056 | 9,534 (-0.4%) | 1,488 | 0.99% | 29.3 s |
| `peak_claim: coelution_shadow` | 10,102 | 8,741 (-8.7%) | 1,421 | 0.98% | 25.7 s |
| `peak_claim: coelution_demix` | stopped | | | | > 44 min |

- The non-destructive demix and contested features add nothing: the rescorer already has
  equivalent information.
- Shadow subtraction, which rewrites extracted intensity, loses 8.7%.
- The destructive demix does not finish at gate 0 on this file. It solves an NNLS problem of
  up to 64 columns at every scan (`demix_scan_stride` 1) with about 500k candidates in play,
  and was stopped at 44 minutes against 21 s for the base. It was not retried with a stride,
  because the non-destructive demix features, which come from the same solve at the apex,
  carry no signal.
- L1f is closed with no gain.

**Not planned: new interference-robust rescore features.** The existing ones already score
at decoy level for R (`coel_clean`, `frag_loo_ref_corr_mean`, AUC 0.50-0.53 against Dm). The
evidence is contaminated before `features` sees it.

**L1b. Centroid splitting at mobility-profile minima.**
- The `ponytail:` upgrade path in `centroid_2d` (`stages/convert/tdf.rs`). P7 measured MS1
  widths at 1.6x the fragment widths (p95 54 scans). Centroids therefore merge neighbouring
  ions at the 30-scan gap, and the merged 1/K0 is a blend.
- Measure: peaks per spectrum, seed anchors, the P7 width distribution, IDs.
- Once split, re-test the P7 shape block (`features.im_shape_features`). Its MS1 comparison
  was limited by this merging.

**L1c. Library fragment intensities, depending on D3.** If D3 shows a large
library-attributable gap:
- the AlphaPeptDeep timsTOF model (the P6 item that is still open);
- MS2PIP settings (`top_n_fragments`, fragment m/z floor; DIA-NN used `--min-fr-mz 200`);
- a fine-tune of the fragment model on pass-1 IDs.

**L1c result: AlphaPeptDeep (2026-09-25).** A full FASTA run with the P6 config and the
P7 binary, except:
- `predict_frag.predictor: peptdeep` (`generic` model, `peptdeep_instrument: timsTOF`,
  `peptdeep_nce: 30`, the AlphaPeptDeep default; the NCE was not swept);
- `charge2_from_precursor_charge` back at its default 2, so charge-2 fragments are requested
  (25.8% of the kept top-12 fragments);
- interpreter `~/.pyenv/versions/mumdia-peptdeep/bin/python`.

Native reversed decoys get their own AlphaPeptDeep predictions, exactly as in P6, so the
comparison with the P6 base is direct. Seeds 1-2 re-run `rescore`. Outputs in
`/public/compomics2/Robbe/MuMDIA_data/bis/l1c/`.

| arm | precursors | peptides (range) | PGs | decoy fraction | never extracted | seed anchors | wall / peak RSS |
|---|---|---|---|---|---|---|---|
| P6 base (MS2PIP `timsTOF2024`, charge 1) | 11,037 | 9,507 (9,460-9,570) | 1,472 | 0.98-0.99% | 203 | 4,313 | 13:17 / 17.1 GB |
| AlphaPeptDeep `timsTOF`, charge 1 + 2 | **12,835 (+16.3%)** | **10,797 (10,725-10,845), +13.6%** | 1,564 (+6.3%) | 0.99% | 92 | 4,371 | 13:09 / 8.7 GB |

Three seeds each; units of TIMS_ROADMAP.md section 2. predict-frag (DeepLC + AlphaPeptDeep +
IM2Deep, 1.33M peptidoforms, 64 CPU processes) took 9:06. The peak RSS is not comparable
with the P6 figure, which was taken on a loaded host with a different peak stage.

D1 (seed 0, best-row pairing), P6 base -> AlphaPeptDeep: accepted keys 10,138 -> 11,523
(+13.7%). 1,290 right-peak rejects, 214 in-window and 186 out-of-window wrong-peak keys and 177
not-extracted keys became accepted; 482 accepted keys were lost. Accepted share per DIA-NN
quintile: 19.6 / 38.7 / 59.2 / 77.8 / 92.9% -> 27.5 / 49.8 / 69.2 / 85.6 / 95.5%.

Reading:
- **AlphaPeptDeep with charge-2 fragments is the largest lever on this file:** +13.6% peptides
  over 3 seeds, at an unchanged decoy fraction. MuMDIA / DIA-NN in peptides moves from 0.62x
  to 0.70x. It exceeds the DIA-NN-fragment hybrid of D3 (H1, +10.8% over its control), so
  MuMDIA's own open predictor now closes the whole library gap that D3 measured.
- The gain is concentrated in the faint quintiles and in the right-peak-low-score group
  (4,327 -> 3,699), which is D2's library-agreement family.
- Not yet validated: entrapment, a second acquisition, and the NCE setting. Nothing is
  promoted.

**L1c NCE sweep (2026-09-25).** The same run at `peptdeep_nce` 25, 30, 35 and 40 (full FASTA
runs, one after the other; seeds 1-2 re-run `rescore`; units of TIMS_ROADMAP.md section 2):

| NCE | precursors | peptides (range) | PGs | decoy fraction | wall |
|---|---|---|---|---|---|
| 25 | 12,722 | 10,728 (10,703-10,751) | 1,535 | 0.99% | 13:32 |
| 30 | 12,835 | 10,797 (10,725-10,845) | 1,564 | 0.99% | 13:09 |
| **35** | **13,009** | **10,899 (10,811-10,981)** | **1,580** | 0.98-0.99% | 14:42 |
| 40 | 12,724 | 10,631 (10,582-10,679) | 1,559 | 0.98-0.99% | |

The optimum is NCE 35, +14.6% peptides over the P6 base (9,507). The step from 30 is +0.9%,
at the noise limit, but the curve is unimodal over four points. Entrapment and the
ProteoBench HYE diaPASEF set run at NCE 35 (outputs in `/public/local/MuMDIA_entrap/l1c` and
`/public/local/ProteoBench/HYE_diaPASEF_mumdia/l1c`).

**L1c entrapment (2026-09-25), NCE 35.** Same setup as TIMS_ROADMAP.md "P6 result": the
E. coli file against E. coli + 1:1 human (`ecoli_human_entrap.fasta`), `entrapment_ratio`
0.563430 (it depends on the digest only, so it is unchanged), the ordinary `nn_torch` accepted
set, `bench/entrapment_fdp.py` at the peptide level. One full run (38:26 wall, 23.6 GB peak
RSS); seeds 1-2 re-run `rescore`.

| seed | real peptides at 1% | spike-in peptides | empirical FDP | decoy fraction |
|---|---|---|---|---|
| 0 | 10,011 | 76 | 0.438% | 0.99% |
| 1 | 9,989 | 81 | 0.467% | 0.99% |
| 2 | 10,010 | 84 | 0.483% | 0.98% |
| P6 (for reference, seeds 0-2) | 8,612-8,790 | 53-56 | 0.355-0.378% | 0.98-0.99% |

- The FDP is below the 1% threshold in every seed. The FDR control holds with AlphaPeptDeep
  and charge-2 fragments on this acquisition.
- Real peptides are +15.0% over the P6 entrapment run (10,003 against 8,702, seed means),
  consistent with the +14.6% of the E. coli-only search.
- The FDP rose from 0.37% to 0.46%. That is a small increase, on 76-84 spike-ins, and still
  about half the nominal level.

**L1c on ProteoBench HYE diaPASEF (2026-09-25), NCE 35.** The TIMS_ROADMAP.md "ProteoBench
HYE diaPASEF" setup: six runs, one `mumdia run` with six `--mzml` (pooled rescore, per-run
quant, MBR off), P7 binary, `config.p6.json` with the L1c predictor changes. Quant arms:
"both" (`fragment_selection: predicted` + `interference_envelope`, configured in the run) and
the default quant, re-run on the same identifications. Scored offline with proteobench 0.18.4
(`score.sh`, `bench/pb_eval.py`); nothing uploaded. Outputs in
`/public/local/ProteoBench/HYE_diaPASEF_mumdia/l1c` (`run`, `defq`).

Identifications (seed 0):

| arm | stripped peptides (pooled `peptide_q_value` 1%) | PGs (`pg_q_value`) | PSMs per run (`run_psm_q` 1%) | decoy fraction |
|---|---|---|---|---|
| P6 | 54,895 | 9,070 | 45.8-47.0k | 0.99% |
| L1c, NCE 35 | **60,547 (+10.3%)** | **9,591 (+5.7%)** | 50.9-52.7k | 1.0% |

ProteoBench at `min_obs` 3 (expected log2 A/B: E. coli -2, yeast +1, human 0):

| arm | ions | median abs eps | species-equalised | CV median | E. coli | yeast | human |
|---|---|---|---|---|---|---|---|
| P6, default quant | 47,309 | 0.222 | 0.549 | 0.122 | -1.03 | +0.55 | +0.06 |
| P6, quant both | 47,309 | 0.182 | 0.307 | 0.114 | -1.57 | +0.76 | +0.04 |
| L1c, default quant | 52,591 | 0.229 | 0.560 | 0.125 | -0.99 | +0.57 | +0.07 |
| **L1c, quant both** | **52,591** | **0.185** | **0.313** | 0.117 | -1.56 | +0.78 | +0.05 |
| DIA-NN 2.5.0, no MBR (public) | 98,694 | 0.125 | 0.180 | 0.089 | -1.79 | +0.84 | |
| AlphaDIA 1.12.1, no MBR (public) | 61,982 | 0.182 | 0.259 | 0.113 | -1.74 | +0.79 | |
| Spectronaut 21 (public) | 148,030 | 0.198 | 0.302 | 0.148 | -1.69 | +0.78 | |

Cost: wall 5:24:24 against 4:31:48, peak RSS 92.4 GB against 70.9 GB. predict-frag took 93 min
for 14.7M peptidoforms (DeepLC + AlphaPeptDeep on 64 CPU processes + IM2Deep). The pooled
rescore took 62 min over 44.2M PSMs, against 45 min over 42.5M.

Reading:
- **The identification gain transfers to a second acquisition context** (a different
  sample, species mix and library size): +10.3% peptides and +11.2% ProteoBench ions, at a
  1.0% decoy fraction. It is one seed and has no entrapment on this set.
- **Quantification accuracy is unchanged per ion:** eps 0.185 against 0.182, species-equalised
  0.313 against 0.307, on 11% more ions. The quant setting still matters more than the
  library: "both" is required in both arms.
- **Against the state of the art (no MBR):** MuMDIA now matches AlphaDIA on the global
  median eps (0.185 against 0.182) at 0.85x its ions, and matches Spectronaut's
  species-equalised eps (0.313 against 0.302). It trails DIA-NN in completeness (0.53x ions)
  and in accuracy (0.185 against 0.125). The E. coli ratio stays compressed (-1.56 against
  DIA-NN's -1.79).
- The peak RSS rose by 21 GB. I have not identified which stage it comes from.

**Valley split on ProteoBench HYE diaPASEF (2026-09-26).** The L1c HYE run above with
`convert.tdf_mz_valley` 0.5 and `tdf_mz_smooth_ppm` 8 added (`valley/config.json`), valley
binary, quant "both" as configured in the run. The default-quant requant was not run. Seed 0.
Outputs in `/public/local/ProteoBench/HYE_diaPASEF_mumdia/valley`.

| arm | stripped peptides (pooled `peptide_q_value` 1%) | PGs (`pg_q_value`) | PSMs per run (`run_psm_q` 1%) | decoy fraction |
|---|---|---|---|---|
| P6 | 54,895 | 9,070 | 45.8-47.0k | 0.99% |
| L1c, NCE 35 | 60,547 | 9,591 | 50.9-52.7k | 1.0% |
| **L1c + valley 0.5 / 8 ppm** | **67,768 (+11.9% over L1c, +23.4% over P6)** | **9,814 (+2.3%, +8.2%)** | 57.7-59.0k | 1.0% |

ProteoBench at `min_obs` 3, quant "both":

| arm | ions | median abs eps | species-equalised | CV median | E. coli | yeast | human |
|---|---|---|---|---|---|---|---|
| P6 | 47,309 | 0.182 | 0.307 | 0.114 | -1.57 | +0.76 | +0.04 |
| L1c | 52,591 | 0.185 | 0.313 | 0.117 | -1.56 | +0.78 | +0.05 |
| **L1c + valley** | **59,639 (+13.4%)** | **0.184** | 0.326 | 0.115 | -1.53 | +0.75 | +0.04 |

Cost: wall 5:24:49 against 5:24:24 for L1c, peak RSS 92.4 GB in both. The pooled rescore took
79 min over 43.7M PSMs, against 62 min over 44.2M.

Reading:
- **The identification gain transfers to the second acquisition:** +11.9% peptides and +13.4%
  ProteoBench ions over L1c at a 1.0% decoy fraction, which is close to the +10.0% of the
  E. coli file. One seed, and no entrapment on this set.
- **Per-ion quantification is unchanged:** global eps 0.184 against 0.185 and CV 0.115
  against 0.117 on 13% more ions. The species-equalised eps is slightly worse (0.326 against
  0.313), and the E. coli and yeast ratios are slightly more compressed (-1.53 against -1.56,
  +0.75 against +0.78). The added ions are probably fainter; this was not checked.
- MuMDIA / DIA-NN in ProteoBench ions (no MBR): 0.60x, from 0.53x. Against AlphaDIA 1.12.1:
  0.96x its ions at about its global eps (0.184 against 0.182).

**L1d. Two-pass refinement with the full search space.**
- DIA-NN's empirical-library workflow is the model: docs/22 WP7, and docs/32 "Orchestrated
  runs, seven files" (Astral immunopeptidomics, first pass about 10k peptides per file,
  pass 2 about 15k).
- docs/32's pass 2 searched a survivors-only library, whose target population is almost
  entirely real. For a single file that is circular.
- The single-run version here keeps the full library and uses pass-1 accepted IDs (about
  10k, against 4.3k seed anchors) to refit:
  - the RT calibration (this also serves L3);
  - the per-charge IM calibration (the P2 levers that are still deferred: non-linear z3,
    anchor weighting);
  - optionally, fine-tunes of the RT, IM and fragment predictors.
- Then re-extract and rescore. The FDR population is unchanged, because the refit touches
  predictions for targets and decoys alike.
- Cost: one extra extract -> rescore chain (about 4.5 min on this file).

**L1e. Rescore capacity.** `MUMDIA_NN_SEEDS=3` and the "sensitivity" recipe (`folds: 5`,
`train_margin_frac: 0.75`) are +0.1 to +0.6 pp on other pools (CLAUDE.md, "Rescore cost";
docs/28). These are known and small, so they come last. They do not address a group whose
median q is 0.21.

### L2: wrong peak, DIA-NN RT inside the window (1,384)

- Lever: `extract.promote_top_peaks` 2-3 (with `retain_top_peaks` >= K). It is implemented
  through compete (`peak_rank` in every group key) and rescore (best row per candidate,
  `selected_peak_rank`), but has never been measured end to end.
- The D4 apex oracle gives its ceiling. Cost: more rows per candidate.
- Entrapment is mandatory, because the rescore sees several chances per target and decoy.
- Measure: D1 movement out of "wrong peak, inside", and the `selected_peak_rank`
  distribution.
- CLAUDE.md calls this lever `retain_top_peaks > 1`. The key that creates scored rows is
  `promote_top_peaks`; `retain_top_peaks` alone writes only the unscored sidecar.

### L2 result (2026-09-25)

Fast loop on the P6 base, P7 binary, seed 0, one arm after the other on a quiet host (load
average about 1). Config: P6 plus `extract.promote_top_peaks: K` and `retain_top_peaks: K`.
Outputs in `/public/compomics2/Robbe/MuMDIA_data/bis/l2/` (`base`, `k2`, `k3`). The base replay
is bit-identical to `output_mumdia_p6_best` (`p6/same.py`, 504,608 rows).

| arm | precursors (`precursor_q`) | peptides (`peptide_q_value`) | PGs | decoy fraction | extracted rows | extract / features / compete / rescore (s) | features.parquet |
|---|---|---|---|---|---|---|---|
| base | 11,136 | 9,570 | 1,468 | 0.98% | 504,608 | 20.7 / 11.1 / 9.2 / 112.6 | 1.2 GB |
| K = 2 | 10,352 | 9,027 (-5.7%) | 1,431 | 0.99% | 879,676 | 22.0 / 15.4 / 17.0 / 139.0 | 2.0 GB |
| K = 3 | 10,441 | 9,003 (-5.9%) | 1,407 | 0.99% | 1,220,107 | 21.5 / 21.1 / 24.3 / 170.3 | 2.7 GB |

The peak RSS of every stage stayed at or below 4.9 GB. The loss is far outside the ~1%
single-seed noise, so no seeds or entrapment were run.

D1 movement (keys; best-row pairing; `q_value` <= 0.01):

| from \ to (K = 2) | accepted | right peak, low | wrong peak, in | wrong peak, out |
|---|---|---|---|---|
| accepted (10,138) | 9,241 | 847 | 31 | 19 |
| right peak, low score (4,327) | 157 | 3,411 | 665 | 94 |
| wrong peak, in window (1,384) | 90 | 400 | 865 | 29 |
| wrong peak, out of window (980) | 21 | 85 | 66 | 808 |

K = 3 is similar: 102 in-window wrong-peak keys recovered, 791 accepted keys lost to "right
peak, low" and 1,138 right-peak rejects moved to a wrong peak.

`selected_peak_rank`, all scored rows: K = 2 targets 66.0 / 34.0%, decoys 65.3 / 34.7%; K = 3
targets 44.8 / 38.6 / 16.6%, decoys 43.2 / 39.6 / 17.3%. Among D1 keys accepted at K = 2, 2,667
are accepted on rank 1, and only 6.8% of those rank-1 apexes lie within 5 s of DIA-NN's RT.

**Mechanism.** The per-row worker scores (`sidecar_work/rescore_psms_scored_*_out.parquet`,
aligned to `psms_competed` by flat row id) show that the classifier scores the candidate, not
the peak:
- Over all K = 2 pairs, rank 1 outscores rank 0 in 46% of pairs, for targets and decoys alike
  (score correlation 0.70).
- For D1 accepted keys whose rank 0 is the right peak and rank 1 is not (5,743 pairs), the
  wrong alternate wins 40% of the time. The median scores are 0.990 and 0.989, although the
  alternate has far worse evidence (median `spectral_angle` 0.26 against 0.76,
  `coelution_mean` -0.01 against 0.56, `log_apex_intensity` 8.4 against 11.1).
- For in-window wrong-peak keys whose rank 1 is the right peak (780), rank 1 wins only 58.5%.
- 42 of the 396 features are identical between rank 0 and rank 1 in >= 99% of pairs. Some are
  candidate-level by design (charge, length, predicted RT, protein count). Others describe one
  peak and are copied from rank 0: `seed_score`, `seed_identified`, `log_seed_hyperscore`,
  `coelution_run`, `mean_mass_error`, `apex_centering_offset`, `rt_diff_profile_apex`,
  `frag_ref_corr_mean_full`, `cosine_fullwindow`. `frag_ref_corr_mean_full` is the strongest
  single feature in D2 (AUC A vs D 0.979).
- The semi-supervised training takes targets at 1% as positives. Wrong alternates of true
  candidates inherit their candidate-level evidence, score about 0.99 and enter the positive
  set. That teaches the model that per-peak evidence does not matter. Decoys gain the same
  max-of-K lift (decoy score median 0.48 -> 0.58), and the 1% score threshold rises from
  0.978 to 0.983.

Reading:
- As implemented, `promote_top_peaks` is not a test of peak selection. The alternates are not
  peak-specific rows, so the rescorer cannot choose between them and the extra rows only add
  noise to training and to the null.
- A fair L2 needs two changes, both inside the default-off path: per-peak values for the
  copied apex features (seed features only when the seed apex matches the row's apex,
  `coelution_run`, mass error, the full-window features per row or excluded), and training on
  rank-0 rows only, with the alternates scored but not learned from. Not implemented; this
  goes to planning (item 6).

### L3: wrong peak, DIA-NN RT outside the window (980; about 600 on the exact peptidoform)

- The RT prediction tail (p95 65 s among rejects). Options, in order:
  1. the pass-1 RT refit of L1d;
  2. `rt_im_train.window_holdout_frac`;
  3. `rt_im_train.adaptive_rt_window`;
  4. `rt_im_train.finetune_deeplc` on the anchors.
- The D4 RT oracle gives the ceiling. The P6 sweep found the RT window multiplier optimum
  at 1.0, so widening alone is not the lever.

### L4: not extracted (762)

Lowest priority. D1 splits them first (RT window, IM window, gate or presence). With
`gate_min_score` 0 the spectral gate no longer binds (TIMS_ROADMAP.md, P6 result).

### L5: MS1 evidence on diaPASEF (added after D2, low priority)

Depends on D5. If MS1 peaks are merged, L1b covers it. If they are removed by the gate or the
tolerance, the fix is a setting.

### Cross-cutting: faint precursors

Every group is dominated by the lowest DIA-NN intensity quintiles. Report each lever's gain
per quintile, so that a lever that only helps bright precursors is visible as such.

## 4. Order and gates

1. D1, then D2.
2. D2b, D3 and D4 in parallel (D3 needs the user to convert the DIA-NN library).
3. L2, which is implemented and cheap, with D4 as its ceiling.
4. L1a as a zero-code prototype first (agreed 2026-09-25): a `run_windows` copy whose IM
   window is narrowed to each candidate's pass-1 fragment-consensus `apex_im` ± δ
   (δ = 0.008 / 0.012 / 0.020; `im_pred_cal` unchanged; the rule is label-blind), fast loop.
   The engine version (a default-off two-pass key) only if an arm gains. Then L1f (read
   whether the demix code respects the IM window first), then L1b. D5 runs alongside.
   The L2 fix (per-peak alternate features, rank-0-only training) is parked behind L1a.
   **Revised after D3 (agreed 2026-09-25):** L1c first, as AlphaPeptDeep (`generic` model,
   instrument `timsTOF`, charge-2 fragments at the default rule) in its own pyenv env
   `mumdia-peptdeep` (peptdeep 1.5.1, torch 2.14 CPU). Then the D5 convert check
   (`tdf_mz_ppm` 5), then L1f. The L1a engine version is dropped for now (prototype result
   above). The L2 fix stays parked.
5. L1c or L1d, whichever D2 and D3 point to. L1d also covers L3.
6. The remaining L3 options.
7. L1e.
8. D5, then L5.

For every arm:
- 3 paired `nn_torch` seeds against the P6 base on the same host;
- D1 categories before and after, and the decoy fraction;
- convert and extract wall time, and peak RSS where the lever touches them;
- entrapment for any gain (`bench/make_entrapment_fasta.py`, `bench/entrapment_fdp.py`,
  set up as in TIMS_ROADMAP.md, "P6 result");
- the ProteoBench HYE diaPASEF set (the second acquisition) before any default, on request;

## 5. Housekeeping (noted, not planned here)

- CLAUDE.md names the top-K lever `retain_top_peaks > 1`. The scored key is
  `promote_top_peaks`.
- `extract.emit_candidate_audit` is documented as writing a per-candidate extract table,
  but nothing writes it (`audit.rs` treats it as future work).

## 6. After L1d (2026-09-28 to 2026-09-30)

Base: L1d in the engine (`rt_im_train.refit`), E. coli file, 12,919 stripped peptides at
`peptide_q_value` 1% over 3 `nn_torch` seeds (0.84x DIA-NN's 15,404). Prototypes ran outside
the engine (side scripts and patched copies of workers) until a lever held over seeds,
entrapment and HYE. Scripts and results are in `/public/local/MuMDIA_raw` (`RESULTS_ms1.txt`
has every number below with its directory).

### Closed on the L1d base (2026-09-28)

| lever | result | status |
|---|---|---|
| L2 top-3 peaks, trained on rank 0 only | +1.3% peptides | dropped: too small for the code it adds |
| P7 apex shape features on valley-split spectra | -0.1% | closed |
| `top_n_fragments` 8 / 10 / 16 / 20 / 24, 200 m/z floor | lose or tie against 12 | closed |
| AlphaPeptDeep MS2 fine-tune on pass-1 IDs, cross-fitted | held-out PCC 0.887 -> 0.902, but -1.5% peptides and -5% seed anchors | closed |
| shared-Gaussian co-elution fit on the chromatograms | univariate AUC 0.57-0.59 against `xcorr_shape` 0.594; +0.010 CV AUC on a supervised model | closed without a rescore |

### Raw fragment traces, prototype (2026-09-28)

`rebuild_traces.py` (alphatims 1.0.9) rebuilt every top-12 fragment trace on extract's grid
from the raw TIMS events: the sum of the events in the grid point's frame and quad slots
holding the precursor, within the learned fragment tolerance, with 1/K0 in
`apex_im +/- 0.015` and no noise floor. Features, compete and rescore were unchanged.

- E. coli, 3 seeds: 13,405 / 13,437 / 13,526, mean 13,456 (+4.2%, 0.87x DIA-NN). The width is
  flat from +/-0.012 to 0.020; noise floors of 2 and 3 events lose, and so does the whole
  calibrated IM window as the band.
- Entrapment (the L1d pass-2 inputs): real peptides 12,636 / 12,743 / 12,732 against 11,985
  (+6.0%), FDP 0.42-0.46% against 0.41%.
- HYE diaPASEF (six runs, seed 0, from the `l1d/pass2_robust` inputs): 97,407 precursors /
  87,502 peptides / 11,578 PGs against 86,048 / 77,734 / 10,657. ProteoBench at k = 3 with
  quant on the new traces: 76,370 ions, median |epsilon| 0.170, CV 0.107, against 69,856 /
  0.185 / 0.118. Quant on the old traces is worse.

### Raw MS1 traces (2026-09-29)

On the same base, `ms1_mono` / `iso1` / `iso2` rebuilt from the raw MS1 frame nearest each
grid point (as extract samples them), 20 ppm, 1/K0 in `apex_im +/- 0.0XX`. The old MS1 traces
are ungated in 1/K0 (`extract.im_gate = fragments`); +/-0.015 keeps 31% of their signal. On
accepted targets the MS1 1/K0 centre agrees with the fragment `apex_im` (median |difference|
0.004).

| arm (E. coli, peptides at 1%) | seeds 0 / 1 / 2 | mean |
|---|---|---|
| raw fragment traces | 13,405 / 13,437 / 13,526 | 13,456 |
| + raw MS1, +/-0.015 | 13,651 / 13,739 / 13,629 | 13,673 (+1.6%) |
| + raw MS1, +/-0.025 | 13,626 / 13,659 / 13,704 | 13,663 (+1.5%) |

Entrapment (+/-0.015): real peptides 12,939 against 12,704 (+1.85%), FDP 0.44-0.46% against
0.42-0.46%. Decoy fraction 0.99% in every arm.

### Apex re-pick on the raw traces (2026-09-29)

A first split, pairing each DIA-NN 1% precursor with every extracted candidate of its
sequence, counted 3,021 of 6,004 unaccepted pairs outside the RT window. That unit double
counts (other modforms, candidates removed in compete); the D1 sorter below pairs each
precursor with its best-scoring row and finds far fewer (section "Loss split on the retrace
baseline").
The best offline rule (cosine to the prediction x square root of the predicted-weighted
signal x a Gaussian RT prior with sigma 8 s, moving only when the new score is more than
twice the score at extract's apex) gave 13,823 peptides (+1.1%, 3 seeds), entrapment +0.4%.
`extract.apex_rt_prior_s: 8` (the default 120 is flat over the +/-17 s window) gives the same,
13,803 (+0.95%), entrapment +0.5% at an unchanged FDP, with no code. No re-pick code; the prior
is a config setting, not yet measured on HYE.

### Raw traces (retrace)

The engine stage, `retrace.enabled` (docs/09 section 6c): fragment and MS1 traces as in the
two prototypes, on convert's own m/z and 1/K0 scale and extract's mass calibration, in both
passes of `rt_im_train.refit`.

Parity with the prototype on the `run_on` inputs: fragment intensity sum 1.013x, r 0.974;
MS1 1.001x, r 0.986; the row-level log2 ratio has median 0.000 and an interquartile range of
+/-0.15, the edge events of two +/-10 ppm windows on m/z scales about 3 ppm apart. Fast loop:
13,578 (3 seeds; prototype 13,673). Entrapment: real peptides 12,922 against the prototype's
12,939, FDP 0.40-0.44%.

Full `mumdia run` on the E. coli file, one binary, 3 seeds:

| | peptides | precursors | PGs | decoy fraction |
|---|---|---|---|---|
| retrace off | 12,927 / 12,950 / 12,874 (12,917) | 15,679 | 1,720 | 0.99% |
| **retrace on** | **13,619 / 13,679 / 13,735 (13,678, +5.9%)** | **16,737 (+6.7%)** | **1,820 (+5.8%)** | 0.99% |

13,678 is 0.89x DIA-NN.

HYE diaPASEF, six runs, seed 0, the stage on the `l1d/pass2_robust` inputs, then features and
compete per run, one pooled rescore and per-run quant (`eng_retrace2`; ProteoBench input
`eng_retrace2/proteobench/custom_input.tsv`):

| arm | precursors | peptides | PGs | PB ions (k = 3) | median abs epsilon | CV |
|---|---|---|---|---|---|---|
| L1d pass 2 (control) | 86,048 | 77,734 | 10,657 | 69,856 | 0.185 | 0.118 |
| prototype, fragments only | 97,407 | 87,502 | 11,578 | 76,370 | 0.170 | 0.107 |
| **engine, fragments + MS1** | **99,634** | **89,424 (+15.0%)** | **11,719** | **77,627** | 0.171 | 0.108 |

Decoy fraction 0.010. The precursor, peptide and PG counts are experiment-wide at 1% on
`precursor_q`, `peptide_q_value` and `pg_q_value`; the ProteoBench ions come from the per-run
quant tables (per-run PSM q) and count ions quantified in at least 3 of the 6 runs, so the two
are not comparable with each other.

Cost: 12 s and 21 GB on the E. coli file; 160 s and 36 GB on one HYE run (7.4e9 trace points),
after the speed work (the first working version took 20 min per HYE run and a version holding
every point reached 240 GB before it was stopped). Every speed step was checked byte-identical
on E. coli and HYE r0.

Not a default yet: HYE has one seed, the stage is diaPASEF only, and `apex_rt_prior_s: 8` on top
is untested on HYE.

### Loss split on the retrace baseline, and wider RT windows (2026-09-30)

`bench/tims_loss.py` (D1) on the full E. coli runs (`/public/local/MuMDIA_retrace`), 17,591
DIA-NN 1% precursors (I/L-merged stripped sequence + charge), accepted on the best row's pooled
PSM `q_value` at 1%, right peak within 5 s:

| category | retrace off | retrace on | on, `rt_window_multiplier` 1.5 |
|---|---|---|---|
| accepted | 13,652 (77.6%) | 14,220 (80.8%) | 14,047 |
| right peak, low score | 2,484 | 1,882 (10.7%) | 1,792 |
| wrong peak, DIA-NN RT inside the window | 829 | 847 (4.8%) | 1,220 |
| wrong peak, DIA-NN RT outside the window | 415 | 433 (2.5%) | 355 |
| not extracted (RT out / IM out / in window) | 35 / 53 / 123 | 32 / 54 / 123 | 14 / 51 / 112 |

Accepted share per DIA-NN intensity quintile with retrace on: 0.50 / 0.75 / 0.87 / 0.94 / 0.98
(faintest first; off: 0.42 / 0.70 / 0.85 / 0.93 / 0.98). Retrace moved about 600 precursors
from "right peak, low score" to accepted. 57% of the 433 outside-window rows are not on
DIA-NN's exact peptidoform.

Wider RT windows (`rt_im_train.rt_window_multiplier`, default 1.0, full runs with retrace on,
3 seeds): 1.5 gives 13,429 peptides (-1.8%), 2.0 gives 13,282 (-2.9%), decoy fraction 0.99%.
At 1.5 about 100 precursors come back inside the window and 373 more lose to a wrong peak
inside it. Closed: the RT-window loss is small and a wider window costs more than it recovers.
The largest remaining group is "right peak, low score", dominated by the faintest quintile.


## 7. "Right peak, low score" on the retrace baseline (2026-09-30)

The target was the 1,882 D1 rejects on the correct peak (section 6, last table). All
counts are stripped peptides at `peptide_q_value` 1%, over 3 `nn_torch` seeds, against the
retrace baseline of 13,678. The decoy fraction was 0.98-0.99% in every arm. Scripts and
outputs are in `/public/local/MuMDIA_raw/d2r`, and every number is in `RESULTS_ms1.txt`.

**Diagnosis (D2 and D2b rerun).**
- On the raw traces, the rejects still match their score-matched decoys on library
  agreement, fragment co-elution, MS1 co-elution and mobility agreement (AUC R vs Dm
  0.45-0.55). They are above those decoys on signal, coverage and RT (0.61-0.67).
- Because R and Dm are matched on the classifier score, an AUC near 0.5 on the model's
  main features is partly built in. The useful question is whether new evidence separates
  them.
- Retrace removed the off-mobility points (`off_frac` 0 in every group). Inside the
  ±0.015 band, R's per-point 1/K0 still jitters at decoy level.

**Stale centroid scalars.**
- `apex_intensity`, `n_matched_fragments` and `ms1_mono` / `iso1` / `iso2` were
  recomputed from the raw traces in a patched `psms_extracted`.
- The fragment count changed in only 3% of rows.
- Result: 13,759 (+0.6%). The fast-loop control reproduced the baseline exactly.

**Mobility agreement on raw events (`imc_ref_w`).**
- At the apex, each fragment's 1/K0 profile is built from the raw events and correlated
  with the weighted sum of the other fragments' profiles (definition in docs/09 section 6c).
- AUC R vs Dm is 0.71. The best of the 396 existing features reaches 0.665.
- AUC A vs D is 0.986.
- It holds within every stratum of observed fragment count (0.67-0.78).
- An RT × 1/K0 version carries the same information (Spearman 0.98).

| arm | peptides (seeds 0 / 1 / 2) | mean | precursors |
|---|---|---|---|
| baseline (fast-loop control) | 13,619 / 13,679 / 13,735 | 13,678 | 16,737 |
| stale scalars refreshed | 13,752 / 13,800 / 13,726 | 13,759 (+0.6%) | |
| + `imc_ref_w` only | 13,845 / 13,807 / 13,905 | 13,852 (+1.3%) | 17,008 |
| both, prototype | 13,972 / 13,843 / 13,960 | 13,925 (+1.8%) | 17,123 |
| **both, engine (`features.retrace_apex`)** | **13,866 / 13,946 / 13,947** | **13,920 (+1.8%)** | 17,098 |

With `imc_ref_w`, 13% of R are accepted, and 0.5% of the previously accepted are lost.

**Entrapment** (retrace-binary control on `entrap_in`, 3 seeds):
- real peptides: 12,922 for the control, 13,210 for both levers (+2.2%);
- FDP: 0.40-0.44% for the control, 0.43-0.51% for both levers.

The added peptides bring spike-ins at a higher rate than the base set: about 14 more per
288 more real peptides, which is an FDP of about 2.7% on the added peptides alone.

**Engine parity.**
- The four refreshed scalars match the prototype exactly.
- `imc_ref_w` correlates at r = 0.93 with the prototype and separates equally well (AUC R
  vs Dm 0.708 against 0.712). The two m/z scales are about 3 ppm apart, and the Pearson
  over few events reacts to the edge events.
- With the key off, the features are identical to the control, and the chromatograms are
  byte-identical with the key on or off.
- The stage took 11.7 s on E. coli.

**Closed on the same baseline.** More library fragments with retrace on (full runs):
- 16 fragments: 13,225 (−3.3%);
- 24 fragments: 12,585 (−8.0%).

Raw traces do not change the centroid result of section 6.

**HYE diaPASEF** (`eng_apex`: six runs, seed 0, from the `l1d/pass2_robust` inputs; no
paired control, against `eng_retrace2`, which used an older chain binary):

| | `eng_retrace2` | `eng_apex` |
|---|---|---|
| precursors | 99,634 | 103,641 (+4.0%) |
| peptides | 89,424 | 92,951 (+3.9%) |
| PGs | 11,719 | 11,887 (+1.4%) |
| ProteoBench ions (k = 3) | 77,627 | 80,359 (+3.5%) |
| median abs epsilon / CV | 0.171 / 0.108 | 0.176 / 0.110 |

The decoy fraction is 0.010 in both. The units are those of section 6.

Cost:
- The first binary held every candidate's 1/K0 profiles until the end of the stage, so
  retrace reached 70-73 GB per HYE run (three runs at once, 466-731 s).
- Each candidate is now scored when it is complete and its profiles are dropped. The
  sidecar is byte-identical, and the E. coli peak fell from 21.2 to 18.7 GB.

Not a default yet: HYE has one seed.

**Closed on the `retrace_apex` base (2026-09-30), E. coli, 3 seeds.**

`extract.apex_rt_prior_s: 8` (full runs; control 13,883):
- E. coli: 14,109 (+1.6%).
- Entrapment (full runs on the E. coli + 1:1 human FASTA): real peptides 13,258 for the
  control and 13,349 for the prior (+0.7%).
- FDP: 0.45-0.47% for the control, 0.49-0.50% for the prior.
- About 9 more spike-ins came with 91 more real peptides, so about 6% of the added
  peptides are false. The gain is paid for in false identifications, so it is closed.

L1e (rescore only, against 13,920):
- `rescore.seeds: 3`: 13,986 (+0.5%).
- The sensitivity recipe (`seeds: 3`, `folds: 5`, `train_margin_frac: 0.75`): 13,998
  (+0.6%).
- Both take about 3x the training time. They remain the option for a final pass.

## 8. Where DIA-NN still finds more, and the apex re-pick (2026-09-30 to 2026-10-01)

The counts are stripped peptides at `peptide_q_value` 1%, over 3 `nn_torch` seeds, on the
E. coli diaPASEF file. The baseline is the full run with retrace and `retrace_apex` (13,883).
Everything is recorded in `/public/local/MuMDIA_raw/RESULTS_ms1.txt`.

**What the gap is not.**
- **FDR calibration.** DIA-NN, run on the entrapment FASTA with the reference settings,
  reports 14,761 real peptides at an FDP of 0.41%. MuMDIA reports 13,258 at 0.45%. So the
  gap holds at the same empirical FDP.
- **Library.** On the same 594,756 keys with the same decoy script, DIA-NN's fragments with
  IM2Deep 1/K0 give 13,483 peptides, against 13,970 for ours (−3.5%). D3's result on the P6
  base has reversed.
- **Tolerance.** A retrace fragment tolerance of 7 or 5 ppm loses 1.0% or 2.3% (DIA-NN chose
  8 ppm).
- **Sampling.** Every window is revisited every 0.968 s, and the trace grid holds every
  sample.
- **Marginal calls.** DIA-NN's median q for the precursors we miss is 5×10⁻⁴.

**What the gap is.** Extract's apex, in RT and in 1/K0, was compared with DIA-NN's observed
values (more than 0.015 in 1/K0, or more than 5 s in RT, counts as off):

| population | `apex_im` off | `im_pred_cal` off | wrong RT peak |
|---|---|---|---|
| accepted | 2% | 33% | 2.5% |
| right peak, low score | 42% | 36% | 0% (by definition) |
| wrong peak, in window | 47% | 38% | 100% (by definition) |

**Levers, E. coli.**

| lever | result |
|---|---|
| 1/K0 band only (rule D: best band by library cosine × √I), prototype | 14,145 (+1.9%) |
| joint RT peak and band (extract's apex + 5 sidecar peaks), prototype | 14,769 (+6.4%) |
| joint choice, engine (`retrace.repick`), fast loop | 14,832 |
| **`retrace.repick`, full run** | **14,869 / 14,911 / 14,953 (14,911, +7.4%)**, 18,544 precursors |

- The full-run result is 0.97x DIA-NN's peptides and 0.99x its precursors.
- Offline, against DIA-NN's RT, the joint choice puts 73% of the wrong-peak keys on the right
  peak, keeps 87% of the right-peak rejects there, and leaves accepted precursors unchanged.

**Entrapment** (full runs):
- `retrace.repick`: 13,906 / 13,948 / 14,071 real peptides (13,975, +5.4%), FDP 0.37-0.47%.
- Control: 13,258 real peptides at an FDP of 0.45-0.47%.
- 13,975 is 0.95x DIA-NN at a lower FDP.
- The fast-loop prototype had added 644 real peptides and 2 spike-ins.

**HYE diaPASEF** (`eng_repick`: six runs, seed 0, against `eng_apex`, which is the same
pipeline without the re-pick):
- The extract rerun with `retain_top_peaks: 5` gave a `psms_extracted` identical to
  `pass2_robust` in all six runs.

| | `eng_apex` | `eng_repick` |
|---|---|---|
| precursors | 103,641 | 112,800 (+8.8%) |
| peptides | 92,951 | 100,790 (+8.4%) |
| PGs | 11,887 | 12,331 (+3.7%) |
| ProteoBench ions (k = 3) | 80,359 | 92,540 (+15.2%) |
| median abs epsilon / CV | 0.176 / 0.110 | 0.172 / 0.107 |

The decoy fraction is 0.010 in both.

**Tuning** (fast loop, control 14,832), all within seed noise:

| arm | peptides |
|---|---|
| `retain_top_peaks` 3 | 14,805 |
| `retain_top_peaks` 8 | 14,783 |
| seed rows used only within 3 s of the chosen apex | 14,817 |

`retain_top_peaks` stays at 5.

**Engine notes.**
- The band centre is the prediction-weighted 1/K0 centroid of the raw profile inside the
  winning band. The first maximum of the band score sits at the low-1/K0 edge of a plateau,
  because an ion is narrower than the band; using that edge gave 14,782.
- Retrace including the repick takes 15 s and 20.8 GB on E. coli, and 6-7.5 min and 36-39 GB
  per HYE run (three runs at once).

## 9. The re-pick has no RT prior (2026-10-01, HYE diaPASEF run 0)

Diagnosis for the MBR-off ion gap against DIA-NN 2.5.0 (docs/TIMS_QUANT_ROADMAP.md sections 2b
and 4h): per run we identify about 7% fewer ions, and 53% of the DIA-NN rows we extract but do not
accept have our apex on another RT peak. Scripts in
`/public/local/ProteoBench/HYE_diaPASEF_mumdia/quant_diag/`: `repick_diag.py`, `repick_prior.py`.

**Who chooses the wrong peak.** The 93,307 candidates of run 0 that DIA-NN reports (`Q.Value` <=
0.01) and we extract, with extract's apex (E) and the re-pick (P) against DIA-NN's RT (right =
within 5 s). "From pred" is the median |apex - `rt_pred_cal`|.

| class | candidates | accepted (q <= 0.01) | chosen from pred | right peak from pred |
|---|---|---|---|---|
| E right, P right | 74,563 | 92.8% | 3.8 s | 4.1 s |
| E right, P wrong (re-pick broke it) | 2,464 | 5.0% | 20.9 s | 4.4 s |
| E wrong, P right (re-pick fixed it) | 9,014 | 68.5% | 4.7 s | 4.8 s |
| both wrong, right peak in top 5 | 3,009 | 5.1% | 20.9 s | 5.0 s |
| right peak not in top 5 | 4,257 | 4.9% | 20.9 s | |

- The re-pick score (`cos x ln(1 + I)` on the best 1/K0 band, section 8) reads no RT. Extract's
  apex does (`extract.apex_rt_prior_s`, 120 s in this config).
- Where the re-pick breaks a right apex, the peak it leaves is closer to the predicted RT in 82%
  of cases; in the "both wrong" class, 85%. Where it fixes one, the new peak is closer in 83%.

**Counterfactual RT prior.** `MUMDIA_REPICK_DUMP=<path>` (diagnostic, off by default) makes
retrace write every scored peak (candidate, order, apex RT, score). On run 0 the re-run reproduces
the shipped `psms_extracted.repick.parquet` exactly (333 s, 38.5 GB). Offline, each score is
multiplied by exp(-0.5 (d / sigma)^2), d = |peak apex - `rt_pred_cal`|, and the first maximum
wins. Sigma in seconds, or as a multiple of the candidate's RT-window half-width (median 37.4 s).

| sigma | on DIA-NN's peak | E right, P right kept | broken repaired | fixed kept | both-wrong rescued |
|---|---|---|---|---|---|
| none (shipped) | 83,577 | 100% | 0% | 100% | 0% |
| 10 s | 78,188 | 90.1% | 75.2% | 79.4% | 67.3% |
| 20 s | 84,236 | 97.1% | 73.8% | 90.2% | 63.7% |
| 30 s | 85,598 | 98.8% | 68.7% | 94.6% | 55.9% |
| 40 s | 85,815 | 99.5% | 61.6% | 96.5% | 47.3% |
| 60 s | 85,360 | 99.8% | 44.2% | 98.2% | 32.2% |
| 0.5 x half-width | 83,869 | 96.6% | 74.5% | 89.2% | 64.5% |
| 0.75 x half-width | 85,450 | 98.6% | 70.0% | 94.1% | 57.2% |
| **1 x half-width** | **85,834** | 99.4% | 63.6% | 96.2% | 49.8% |
| 1.25 x half-width | 85,714 | 99.7% | 56.0% | 97.3% | 41.3% |
| 2 x half-width | 85,019 | 99.9% | 36.0% | 98.9% | 24.1% |

- At sigma = 1 x half-width, 2,257 more of DIA-NN's candidates (2.4%) sit on DIA-NN's peak in this
  run. It is a peak-choice count, not an identification count: the repaired and rescued candidates
  are faint (5% accepted today), and the prior also moves decoys toward their predicted RT (45% of
  decoy picks change against 45% of target picks), which shifts the RT features the rescorer sees.
- The rule reads no label and generalises with the window (it scales with `rt_im_train`'s window).
- Next: an engine key (`retrace.repick_rt_prior`, multiple of the window half-width, 0 = off), then
  the identification gates: E. coli full run over 3 seeds, entrapment, HYE six runs.

**Gates: `retrace.repick_rt_prior: 1` (2026-10-02).** Engine key for this measurement (sigma =
1 x the RT-window half-width, 0 = off), removed again after the gates. On HYE run 0 the engine picks equal the offline counterfactual on all
7,086,111 candidates. Binary `~/bin/mumdia-rtprior/mumdia`; full runs in
`/public/local/MuMDIA_repick/prior1`, HYE in `eng_prior1` (seed 0, quant with the diaPASEF preset
including `cross_run_width`).

| gate | without the prior | with the prior |
|---|---|---|
| E. coli peptides, 3 seeds | 14,869 / 14,911 / 14,953 (14,911) | 14,873 / 14,951 / 15,021 (14,948, +0.25%) |
| entrapment real peptides, 3 seeds | 13,906 / 13,948 / 14,071 (13,975) | 13,927 / 13,865 / 13,864 (13,885, -0.6%) |
| entrapment FDP | 0.47 / 0.37 / 0.43% | 0.42 / 0.41 / 0.42% |
| HYE precursors / peptides / PGs (seed 0) | 112,800 / 100,790 / 12,331 | 112,872 / 100,823 / 12,318 |
| HYE target rows at pooled q 1%, six runs | 543,082 | 545,769 (+0.5%) |
| HYE ProteoBench ions (k = 3) | 92,540 | 93,142 (+0.65%) |
| HYE global / eq / CV | 0.130 / 0.177 / 0.083 | 0.130 / 0.178 / 0.084 |

HYE run 0, by what the prior did to the candidates DIA-NN reports:

| candidates | n | accepted before | accepted after |
|---|---|---|---|
| moved onto DIA-NN's peak | 3,064 | 3.4% | 37.6% |
| moved off DIA-NN's peak | 807 | 42.6% | 12.3% |
| right before and after | 82,770 | 90.6% | 90.2% |
| wrong before and after | 6,666 | 5.7% | 5.3% |

- The prior repairs the peak choice as predicted, but 62% of the candidates it moves onto the
  right peak still fail scoring (median q 0.048, half at q <= 0.05). On these faint precursors the
  score is the bottleneck, not the peak.
- Net: +0.25% / -0.6% / +0.5% (E. coli / entrapment / HYE rows), inside seed noise. Not promoted;
  the key is not kept (the change is 10 lines in `retrace.rs` `repick`, multiplying the score).
- Next: the "right peak, low score" population (docs/TIMS_QUANT_ROADMAP.md section 4h: 45,651
  run-level HYE rows, 20,000 at q 0.01-0.05), now including these repaired candidates.
- The pooled HYE rescore took 2,093 s at a 134 GB peak (eng_repick: 3,751 s, 69 GB).

## 10. "Right peak, low score" on the re-pick base (2026-10-02, HYE diaPASEF run 0)

Scripts in `/public/local/ProteoBench/HYE_diaPASEF_mumdia/quant_diag/`: `rplow_features.py`
(D2 method on `eng_repick`), `rplow_local.py`. Populations, one competed row per candidate:
- R: DIA-NN 2.5.0 reports it, our re-picked apex within 5 s of DIA-NN's RT, our q > 0.01 (8,219;
  3,307 at q 0.01-0.05);
- A: the same, accepted (75,358); Aq: A resampled to R's DIA-NN `Precursor.Normalised`;
- Dm: decoys resampled to R's rescore score; D: 300,000 decoys.

**Every evidence family is at decoy level.** Against Aq (same DIA-NN abundance), the largest gaps,
with R at Dm level on all of them (AUC R vs Dm 0.44-0.56):

| family | feature | AUC R vs Aq | median R / Aq / Dm |
|---|---|---|---|
| peak contrast in the RT window | `peak_to_full_area_ratio_frag_mean` | 0.167 | 0.097 / 0.186 / 0.107 |
| | `n_competing_peaks_in_window` | 0.764 | 8 / 2 / 7 |
| library intensity agreement | `scribe_score_area` | 0.183 | 2.65 / 3.66 / 2.76 |
| fragment co-elution | `frag_ref_corr_mean_full` | 0.188 | 0.254 / 0.388 / 0.281 |
| 1/K0 agreement | `imc_ref_w` | 0.236 | 0.309 / 0.501 / 0.303 |
| MS1 | `ms1_isotope_cosine_apex` | 0.275 | 0.956 / 0.989 / 0.944 |
| | `ms1_ms2_time_corr` | 0.375 | 0.53 / 0.79 / 0.51 |
| signal | `sum_y_intensity` | 0.328 | 2,521 / 3,952 / 2,536 |

- The RT window is the same for R and Aq (37.4 s half-width), `contested_frac` is 0 in both, and
  R sits at lower precursor m/z (582 against 667, AUC 0.36).
- A local contrast from the raw traces (+-4 scans over +-10 to +-40 scans) behaves the same: at
  +-10 scans R's summed trace has 3 maxima above half the apex, Aq 1, Dm 3 (AUC R vs Dm
  0.47-0.57). It is not new evidence.
- R's apex is less precise: |apex - DIA-NN RT| median 0.97 s against 0 (p90 2.9 against 0.97 s),
  |apex 1/K0 - DIA-NN IM| > 0.015 in 9.8% against 1.3%. DIA-NN's peak is wider for R (8.7 against
  7.7 s).

**Oracle apex (diagnostic only).** `rpdump/r0_oracle`: the DIA-NN-reported targets get DIA-NN's RT
and IM as `apex_rt` / `apex_im` (93,307 rows), then retrace (re-pick off) and features; same
populations. R moves part of the way: over the 82 features with |AUC(R, Aq) - 0.5| >= 0.2, the
median gap falls from 0.257 to 0.188, and R's separation from Dm doubles (median |AUC - 0.5| 0.032
to 0.066). Library and 1/K0 agreement gain most (`scribe_score_area` AUC 0.18 to 0.28, `imc_ref_w`
0.24 to 0.33, `ms1_isotope_cosine_apex` 0.28 to 0.35); co-elution and contrast hardly move
(`frag_ref_corr_mean_full` 0.19 to 0.22). At the oracle apex R's `sum_y_intensity` is still 30%
below Aq's (2,844 against 4,052).

Reading:
- A perfect apex would recover about a quarter of R's evidence gap. It is an upper bound, and the
  RT part was already tried (section 9).
- The rest is the signal itself: at the same DIA-NN abundance our traces hold about 30% less of
  R's fragment signal, and what they hold is crowded (several peaks of similar height within
  +-10 s). Either we capture less of these ions (tolerance, 1/K0 band, fragment choice), or DIA-NN's
  abundance estimate for them leans on evidence we do not use.

**Signal capture is not the loss** (`rplow_capture.py`, traces rebuilt at the oracle apex of run 0
with the re-pick off). Median gain of the +-4-scan summed fragment area over the current settings
(1/K0 band +-0.015, 16.5 ppm from the mass calibration), with the local co-elution (median Pearson
of each fragment with the sum of the others over +-10 scans):

| arm | area gain R / Aq / Dm / D | co-elution R / Aq / Dm | co-elution AUC R vs Dm |
|---|---|---|---|
| base | 1 / 1 / 1 / 1 | 0.257 / 0.540 / 0.145 | 0.655 |
| 1/K0 band +-0.025 | 1.38 / 1.26 / 1.39 / 1.44 | 0.200 / 0.460 / 0.132 | 0.602 |
| 25 ppm | 1.35 / 1.28 / 1.43 / 1.44 | 0.253 / 0.523 / 0.159 | 0.625 |

R gains what the decoys gain, and contrast and co-elution get worse for every population: the
extra signal is background, not R's ion. The trace build does not lose these ions; the current
band and tolerance are right.

**Local co-elution is weak new evidence.** At our own apex (`rplow_coel_sweep.py`, median over
fragments of the Pearson with the sum of the others over +-L scans):

| L (scans) | 4 | 6 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|
| AUC R vs Dm | 0.570 | **0.591** | 0.590 | 0.563 | 0.534 | 0.494 |
| AUC Aq vs D | 0.922 | 0.932 | 0.929 | 0.918 | 0.907 | 0.883 |

On every existing co-elution feature R is at or below its score-matched decoys (AUC 0.42-0.52;
the features use the whole window or the peak). The +-6-scan version is the only co-elution
measure above them. For scale, `imc_ref_w` separated at 0.71 on E. coli and gave +1.3% (section 7).

Reading: the right-peak rejects of HYE are signal-limited. A perfect apex recovers about a quarter
of the evidence gap, widening the trace build adds only background, and the one new measure found
here is weak.

## 11. DIA-NN's peak not among extract's top 5 (2026-10-02, HYE diaPASEF run 0)

The 4,257 candidates of run 0 (section 9 classes) whose DIA-NN peak is neither extract's apex nor
one of the five sidecar peaks (`quant_diag/nottop5.py`). Extract enumerates its peaks on the
per-scan count of distinct matched fragments (`peaks::enumerate_peaks`, bound 1/3, prominence
0.1, ranked by the summed count), from centroid matches over the run-window 1/K0 range (median
0.079). The rebuild from `l1d/pass2_robust` reproduces the shipped sidecar for 3,000 of 3,000
candidates (count window 1; the sidecar uses the unsmoothed count). Rank of DIA-NN's peak (a peak
apex within 5 s of DIA-NN's RT) in that profile, and in the same profile rebuilt from raw traces
at DIA-NN's 1/K0 (+-0.015, `rpdump/r0_oracle`):

| profile | population | rank 0-4 | 5-9 | 10-19 | no peak |
|---|---|---|---|---|---|
| centroid, run-window 1/K0 (extract) | not in top 5 (4,257) | 0 | 69.0% | 11.0% | 19.7% |
| | in top 5 (4,000 sampled) | 100% | 0 | 0 | 0 |
| raw, DIA-NN's 1/K0 +-0.015 | not in top 5 | 62.4% | 14.6% | 3.5% | 19.3% |
| | in top 5 | 96.3% | 3.5% | 0.1% | 0.1% |

Fragment count at DIA-NN's RT against the top peak's (medians): 6 against 7 in the centroid
profile, 9 against 9 in the raw one. Each profile has about 21 peaks.

- 905 of the 4,257 (21%) are outside our RT window: DIA-NN's RT is a median 55 s from
  `rt_pred_cal` (p25-p75 46-78 s) against a 37.4 s half-width, and 28% are beyond twice the
  half-width. Over all 93,307 DIA-NN candidates of the run, 1,014 (1.1%) are outside the window.
  This is RT prediction, not peak detection (wider windows: section 6).
- The rest are near misses of the count ranking: in extract's profile DIA-NN's peak is rank 5-9 for
  69%. Counted in a narrow band at the right 1/K0 it would be in the top five for 62%. The wide
  1/K0 range lets random matches build peaks of the same height.
- The narrow-band figure uses DIA-NN's 1/K0, so it is an upper bound: extract's `im_pred_cal` is
  more than 0.015 from DIA-NN's IM for a third of candidates (section 8).
- Size: 3,350 candidates per run (3.6% of DIA-NN's). Section 9 showed that a candidate moved onto
  the right peak converts at about 38%, so the reachable gain is of the order of 1,000 run-level
  rows per run (about 1%).

## 12. Where the per-run gap sits, and zero-intensity library fragments (2026-10-02, HYE diaPASEF)

`quant_diag/gap_strata.py` on run 0: DIA-NN 2.5.0 rows (`Q.Value` <= 0.01, 94,872) against our
accepted rows (pooled q <= 0.01, 89,674); net gap 5,198 (5.5%). Net gap = DIA-NN rows - DIA-NN rows
we accept - our rows DIA-NN does not report.

| stratum | share of DIA-NN rows | share of the net gap | our acceptance of DIA-NN's rows |
|---|---|---|---|
| precursor charge 2 | 79.6% | 97.4% | 0.805 |
| precursor charge 3 | 19.3% | 1.5% | 0.777 |
| length 7-9 | 24.7% | 47.7% | 0.749 |
| length 10-12 | 33.7% | 36.8% | 0.820 |
| length 13-16 | 29.0% | 15.8% | 0.828 |
| precursor m/z <= 500 | 19.2% | 30.3% | 0.744 |
| precursor m/z 500-600 | 26.9% | 36.0% | 0.794 |
| RT 15-25 min | 2.2% | 13.9% | 0.518 |
| 2 missed cleavages (not in our library) | 0.4% | 8.1% | 0 |

The gap is short, doubly charged, low-m/z precursors. Charge 3 is net even.

**Library slots.** The fragment library keeps the top 12 fragments by predicted intensity, so a
precursor with fewer than 12 positive predictions keeps fragments predicted at 0. 21% of all
176.8M library fragments are predicted 0. Per precursor (means of 1,500 sampled per stratum):

| precursor | slots predicted 0 | fragment charge 2 (predicted 0) | m/z < 200 | useful (> 0, m/z >= 200) |
|---|---|---|---|---|
| charge 2, 7-9 aa | 3.97 | 3.33 (2.46) | 1.74 | 7.96 |
| charge 2, 10-12 aa | 2.05 | 2.05 (1.09) | 0.70 | 9.93 |
| charge 2, 13-16 aa | 1.99 | 2.08 (1.09) | 0.56 | 10.01 |
| charge 3, 13-16 aa | 1.42 | 3.30 (0.69) | 0.52 | 10.58 |

**Test: drop the fragments predicted 0** (`eng_libnz`; `lib_nz/fragments_nz.parquet`, 79.1% of
the fragments; the 11,821 candidates without a positive fragment keep theirs). Same binary and
inputs as `eng_repick` (paired), seed 0. A filter to m/z >= 200 on top removes only 0.3% more,
because almost every fragment below 200 is already predicted 0, so it was not run.

| | `eng_repick` | zero-intensity fragments dropped |
|---|---|---|
| extracted candidates, run 2 (targets / decoys) | 3,572,711 / 3,567,127 | 3,384,767 / 3,379,196 (-5.3%) |
| DIA-NN candidates extracted, run 0 | 93,008 | 92,912 |
| precursors / peptides / PGs | 112,800 / 100,790 / 12,331 | 112,559 / 100,706 / 12,300 |
| target rows at pooled q 1%, six runs | 543,082 | 543,286 |
| decoy rows at pooled q 1% | 5,429 | 5,431 |
| run 0: DIA-NN 7-9 aa rows accepted | 0.749 | 0.752 |
| ProteoBench ions (k = 3) / global / eq / CV (quant with `cross_run_width`) | 92,540 / 0.130 / 0.177 / 0.083 | 92,531 / 0.129 / 0.177 / 0.083 |

- No effect on identification, also not on the short peptides. The zero-predicted slots collect
  noise matches, but the rescorer already discounts them. They are not why short peptides lose.
- The extract population shrinks by 5.3% without losing DIA-NN's candidates, so the filter is a
  cost lever (smaller extract, features and rescore), not a sensitivity one.
- The short-peptide deficit is then the small number of informative fragments itself (about 8 for
  a 7-9-mer at charge 2), consistent with sections 10 and 11: faint, crowded precursors with
  little evidence.

## 13. Systemic causes checked for the 6% per-run gap (2026-10-02, HYE diaPASEF)

Precursors at 1% against DIA-NN 2.5.0 (MBR off): mean per run 90,514 against 96,692 (-6.4%; run-level
q in both), union over runs 124,738 against 131,475 (-5.1%), experiment-wide 112,800 against 121,542
(-7.2%; our `precursor_q`, DIA-NN run and global q). The per-row levers of sections 9-12 sum to about
2%, so the search turned to factors every precursor shares. Scripts in `quant_diag/`.

| hypothesis | test | result |
|---|---|---|
| reverse decoys shadow their targets (same composition) | `shadow.py`: decoys above the 1% cut whose paired target is accepted with an apex within 5 s and 0.015 1/K0 | 1.2-2.4% of them; predicted RT of target and decoy differ by a median 38.7 s. Not it |
| plain target-decoy q is conservative | `picked.py`: picked competition per (base peptide, charge, Met-ox count) | +0.5% rows per run on HYE; entrapment precursor FDP 0.35-0.41% plain, 0.37-0.47% picked. Small |
| prior predictions (RT, 1/K0) worse than DIA-NN's | residuals on the 75,842 precursors both identify in run 0 | RT: ours median 3.8 s, DIA-NN `Predicted.RT` 18.7 s; 1/K0: ours 0.0099, DIA-NN 0.0082 (p95 equal). Not it |
| MS1 signal missing for our rejects | DIA-NN's own values for the right-peak rejects (R) against abundance-matched accepted (Aq) | `Ms1.Area` equal (25k against 23k); DIA-NN's evidence is lower for R too (`Evidence` 3.4 against 4.0, `Ms1.Profile.Corr` 0.66 against 0.82), but DIA-NN still accepts them (median q 9e-4, PEP 0.009) |
| fragment-intensity model not adapted to the data | `frag_headroom.py`: cosine (square-rooted areas) of prediction against run 0, and run 1 against run 0, 77,448 precursors accepted in both | prediction 0.967, run-to-run 0.986; run-to-run better in 81%; by quintile 0.952 / 0.960 (faint) to 0.984 / 0.998 (bright) |

Short peptides are over-represented among high-scoring decoys: 38% of the 856 decoys above the 1% cut
of run 0 are 7-9 aa, against about 23% of our accepted targets.

Reading: the per-run gap is a discrimination gap. DIA-NN's evidence for the precursors we reject is
weaker than for the rest, as ours is, but its targets stay separated from its decoys and ours do not.
Of the shared causes tested, only the fragment model shows real headroom: RT and 1/K0 are already
adapted to the data (multi-head refit, per-charge CCS calibration), the fragment intensities are
peptdeep's generic timsTOF model. Fine-tuning it on the dataset's confident identifications (the
transfer learning of AlphaPeptDeep and AlphaDIA) is the untested lever. It helps real precursors, whose
spectra exist, and cannot help reversed decoys, whose spectra do not.

## 14. Fine-tuning the fragment model on the dataset's own IDs (2026-10-02, HYE diaPASEF)

Prototype outside the engine, to find the cause, not a lever to implement as is. Scripts:
`bench/ms2_finetune/ms2_finetune.py` (fine-tune and held-out check) and `ms2_repredict.py` (library
re-prediction); both carry the HYE paths of this run and run in the `mumdia-peptdeep` env. Model in
`ms2ft/`, library in `lib_ft/`, arm `eng_ms2ft` (under `/public/local/ProteoBench/HYE_diaPASEF_mumdia`).

**Fine-tune.** peptdeep `generic` MS2 model (timsTOF, NCE 35, as the library build), trained 10
epochs at lr 1e-4 (34 min, CPU) on 79,937 precursors accepted at pooled q <= 0.001 in any of the six
`eng_repick` runs. Target pattern: mean over the confident runs of the max-normalised pass-1
fixed-window areas of the 12 library fragments; every other b/y position set to 0. Split by stripped
sequence (80 / 20). Held-out (12,479 precursors accepted in runs 0 and 1), cosine of square-rooted
areas over the fragments seen in both runs:

| | prediction vs run 0 | run-to-run better than prediction |
|---|---|---|
| generic model | 0.9720 | 82% |
| fine-tuned | **0.9834** | 65% |
| ceiling: run 1 vs run 0 | 0.9881 | |

**Library.** Every candidate, target and decoy, re-predicted with the fine-tuned model (14.7M
precursors, 57 min, CPU, 48 processes; the spawned workers keep the weights, checked). Only
`predicted_intensity` is replaced: the 12 fragments per candidate, their m/z and cardinality, and the
precursor table (RT, 1/K0) are unchanged. Median change 0.042 in normalised intensity.

**HYE, six runs, seed 0**, same binary and inputs as `eng_repick` (paired):

| | `eng_repick` | fine-tuned fragment model | change |
|---|---|---|---|
| target rows at pooled q 1%, mean per run | 90,514 | 93,604 | +3.4% (+3.1 to +3.6% per run) |
| decoy rows at pooled q 1%, six runs | 5,429 | 5,615 | |
| precursors / peptides / PGs | 112,800 / 100,790 / 12,331 | 116,095 / 103,324 / 12,655 | +2.9 / +2.5 / +2.6% |
| ProteoBench ions (k = 3) | 92,540 | 95,772 | +3.5% |
| global / eq / CV | 0.130 / 0.177 / 0.083 | 0.132 / 0.181 / 0.085 | |
| E. coli / human / yeast abs eps | 0.228 / 0.120 / 0.182 | 0.236 / 0.122 / 0.184 | |
| ions shared with DIA-NN 2.5.0 | 81,778 | 83,345 | |

Against DIA-NN 2.5.0 (96,692 rows per run, 98,694 ions): from -6.4% to -3.2% per run, from -6.2% to
-3.0% in ions.

Run 0, the right-peak rejects (R, section 10) on the new competed table, against score-matched decoys
(Dm; populations from the old q): library agreement now separates them (`scribe_score_area` AUC 0.459
to 0.547, `spectral_entropy_similarity_area` 0.497 to 0.589, `kl_obs_pred` 0.556 to 0.464), because R's
agreement rose (median scribe 2.65 to 2.76) while the decoys' fell (2.76 to 2.62). Co-elution and
contrast are unchanged, as they do not use the predictions.

Reading:
- The fragment-intensity model is a cause of the per-run gap, worth about half of it on this dataset.
  RT and 1/K0 were already adapted to the data; the fragment intensities were not.
- Not yet validated: one rescore seed, one dataset, and no entrapment. The model is trained on the
  same runs it scores. Decoys get the same model, so the target-decoy symmetry holds, but the FDR gate
  is the E. coli entrapment run with its own fine-tune, then seeds and a second acquisition.
- An earlier test points the other way: on 2026-09-28 a cross-fitted AlphaPeptDeep MS2 fine-tune on
  the E. coli pass-1 IDs (centroid traces, before retrace and the re-pick; `/public/local/MuMDIA_ft`)
  raised held-out PCC from 0.887 to 0.902 but lost 1.5% of peptides and 5% of seed anchors. Here the
  model is trained on all six runs' IDs and scores those runs, so part of the +3.4% may be the model
  favouring precursors it was trained on. The deciding test is a cross-fitted fine-tune (train on one
  half of the precursors, predict the other) on HYE, and the E. coli entrapment run.
- Open design points if pursued: the fragment set was frozen (re-choosing the top 12 with the new model
  is a second effect), the 0 targets for unobserved positions bias the model toward sparsity, and the
  training set uses six runs' IDs (a per-run or first-run-only fine-tune, as `rt_library_scope`, is the
  engine-shaped variant).
