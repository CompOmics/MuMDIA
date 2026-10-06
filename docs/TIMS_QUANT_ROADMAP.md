# TIMS quant roadmap: diaPASEF quantification at DIA-NN level

Goal: on ProteoBench `quant_lfq_DIA_ion_diaPASEF` (HYE, 3 + 3 runs, MBR off), reach DIA-NN's
ratio accuracy and precision without losing the identifications gained in `TIMS_ROADMAP.md`
and `TIMS_ROADMAP_bis.md`. The policy is that of
`docs/20_sensitivity_and_quantification_playbook.md` ("Quantification accuracy"). Report every
result on the fixed common ion set and on all ions.

## 1. Constraints

- **No loss of sensitivity.** Identification counts are not allowed to drop. Neither is the
  number of ProteoBench ions at `min_obs` 3. A quant change that improves epsilon by not
  reporting difficult ions does not count as a gain. A quality flag on an ion is allowed. A
  filter that removes the ion from the submission is not.
- **Quant changes do not change IDs.** Most levers below are requant-only. They run on the
  fixed `eng_repick` identifications, so they cost no sensitivity by construction and need no
  seed or entrapment arm. A lever that changes extraction, retrace or rescore still needs the
  usual seeds, entrapment and second-dataset gates.
- **Scope: HYE diaPASEF only, for now.** Every lever is measured and gated on the HYE diaPASEF
  set. A quant default that comes out of this roadmap applies to diaPASEF (TDF) input only.
  Making it general (Astral HYE, PYE diaPASEF, AIF) is a later step with its own measurement,
  so keep each lever a config key that other acquisitions can switch on, not a TDF-only code
  path.
- **Targets.** Since 2026-10-01 the targets are the two DIA-NN submissions of section 2b, one
  with MBR off and one with MBR on, in sensitivity (ProteoBench ions at `min_obs` 3) and in
  quantification (global, then species-equalised median |epsilon|, then CV). The earlier
  reference of section 2 (0.118 / 0.169 / 0.076) is superseded. When a lever trades accuracy
  against CV, accuracy wins. A CV gain that compresses the ratios is not accepted.
- **Clean room.** DIA-NN is a reference for the result only. Published method descriptions
  (for example the QuantUMS preprint) can guide the design. Code and constants cannot.

## 2. Starting point (2026-10-01)

Inputs:
- DIA-NN: `bench/diann_compare/input_file.tsv`, scored with `bench/pb_eval.py --format DIA-NN`.
  ProteoBench reads `Precursor.Normalised`.
- MuMDIA: `bench/proteobench_input/eng_repick/custom_input.tsv`. This is the `eng_repick` arm
  of `TIMS_ROADMAP_bis.md` section 8 (seed 0). Quant is
  `fragment_selection: predicted` + `interference_envelope`, top-3 fragments, per-candidate
  descent-walk bounds, `q_filter: psm_q`, on the retraced raw traces.

ProteoBench at `min_obs` 3 (expected log2 A/B: E. coli -2, yeast +1, human 0). Median
|epsilon| is global. "eq" is species-equalised. CV is ProteoBench's `CV_median`.

| arm | ions | median abs eps | eq | CV | E. coli | yeast | human |
|---|---|---|---|---|---|---|---|
| DIA-NN (`Precursor.Normalised`) | 88,924 | **0.118** | **0.169** | **0.076** | -1.84 | +0.81 | -0.03 |
| DIA-NN (`Precursor.Quantity`, no normalisation) | 88,924 | 0.120 | 0.172 | 0.080 | -1.79 | +0.86 | +0.02 |
| MuMDIA `eng_repick` | **92,540** | 0.172 | 0.358 | 0.107 | -1.43 | +0.66 | +0.03 |

MuMDIA reports 4% more ions than DIA-NN, but its epsilon is 46% higher and its E. coli
ratio is compressed by 0.4 log2.

This DIA-NN reference is superseded by section 2b. Sections 3 to 4g still compare against it.

## 2b. Targets (2026-10-01): DIA-NN 2.5.0 MBR off, DIA-NN 2.2.0 MBR on

Inputs are in `bench/diann_compare/`: `input_file.parquet` (ProteoBench's input),
`result_performance.csv` (the per-ion intermediate) and `param_0..txt` (the DIA-NN log).

| target | DIA-NN | MBR | FASTA | missed cleavages | charge | length | precursor m/z | variable mods |
|---|---|---|---|---|---|---|---|---|
| `250_noMBR` | 2.5.0 | off | `ProteoBenchFASTA_MixedSpecies_HYE.fasta` | 2 | 2-4 | 7-30 | 400-1200 | Met ox |
| `220_MBR` | 2.2.0 | `--reanalyse` | `ProteoBenchFASTA_DDAQuantification.fasta` | 1 | 1-5 | 6-30 | 400-1000 | Met ox, N-term acetyl, `--met-excision` |
| MuMDIA `eng_repick` | | | HYE FASTA | 1 | 2-4 | 7-30 | no cap | Met ox, Met excision |

The diaPASEF windows cover precursor m/z 400-1000 only (24 windows of 25 Th), so the m/z caps do
not differ in practice. Section 4h measures what the other differences cost.

Scored with `bench/pb_eval.py --format DIA-NN --module quant_lfq_DIA_ion_diaPASEF`; both
reproduce `result_performance.csv`. `min_obs` 3. "CV" is the median over ions of (CV_A + CV_B) / 2,
as in `shared.py`. "PB CV" is ProteoBench's `CV_median`, which equals (median CV_A + median CV_B) / 2
on all five arms here. (The CV values in the 2026-10-01 brief, 0.097 / 0.094 / 0.117 / 0.101, match
neither definition and are not used.) MuMDIA arms are seed 0 `eng_repick` IDs with the diaPASEF quant
preset (section 4g, `crwbg`); "+ MBR" adds the rescuable tier (section 4i).

| arm | ions | global | eq | CV | PB CV | E. coli / human / yeast abs eps | E. coli log2 |
|---|---|---|---|---|---|---|---|
| DIA-NN 2.5.0, MBR off | **98,694** | **0.125** | 0.180 | 0.093 | 0.089 | 0.237 / 0.113 / 0.189 | -1.79 |
| MuMDIA `crwbg`, MBR off | 92,540 | 0.134 | 0.180 | **0.090** | **0.087** | 0.231 / 0.123 / 0.185 | -1.84 |
| DIA-NN 2.2.0, MBR on | **118,326** | 0.143 | **0.207** | 0.114 | 0.107 | 0.282 / 0.125 / 0.215 | -1.77 |
| MuMDIA `crwbg` + MBR | 100,340 | 0.143 | 0.220 | 0.098 | 0.094 | 0.325 / 0.128 / 0.207 | -1.77 |
| MuMDIA preset with `cross_run_width` 2.5 + MBR | 100,340 | **0.140** | 0.218 | **0.093** | **0.089** | 0.326 / 0.124 / 0.204 | -1.76 |

On the ions shared with each target (`quant_diag/shared.py`, now reading `result_performance.csv`;
`DIANN=250` or `220`), eps / CV / E. coli log2:

| arm | shared | MuMDIA | DIA-NN on the same ions | MuMDIA-only |
|---|---|---|---|---|
| `crwbg` s0 against 2.5.0 | 81,778 | 0.127 / 0.088 / -1.84 | 0.115 / 0.088 / -1.80 | 10,762: 0.216 / 0.126 / -1.64 |
| `crwbg` s1 against 2.5.0 | 81,506 | 0.127 / 0.088 / -1.84 | 0.115 / 0.088 / -1.80 | 10,635: 0.213 / 0.126 / -1.57 |
| `crwbg` s2 against 2.5.0 | 81,755 | 0.127 / 0.088 / -1.84 | 0.115 / 0.088 / -1.80 | 10,720: 0.217 / 0.127 / -1.61 |
| `crwbg` + MBR s0 against 2.2.0 | 91,379 | 0.138 / 0.095 / -1.79 | 0.127 / 0.104 / -1.80 | 8,961: 0.237 / 0.141 / -1.09 |
| preset (`cross_run_width` 2.5) + MBR s0 against 2.2.0 | 91,379 | 0.134 / 0.091 / -1.78 | 0.127 / 0.104 / -1.80 | 8,961: 0.231 / 0.136 / -1.03 |

The last rows (2026-10-02) are the current diaPASEF preset on the `mbr_s0` IDs: binary
`~/bin/mumdia-ww/mumdia`, `requant_crw.sh mbr_ww_s0 mbr_s0` with `WIDTH=2.5`, scored with
`MBR=true score.sh` (the flag sets `enable_match_between_runs` in `user_input.json` and changes no
number). Against `crwbg` + MBR: global -0.003, eq -0.002, CV -0.005, the same gain as MBR off
(section 4h). Against DIA-NN 2.2.0: global -0.003, eq +0.011 (E. coli), CV -0.021, ions -15.2%.

- MBR off: eq and CV are at DIA-NN 2.5.0. Global epsilon is +0.009 on all ions and +0.012 on the
  shared ions, so it is a per-ion precision gap, not an effect of the extra ions. Ions: -6.2%.
- MBR on: CV is better than DIA-NN's (0.098 against 0.114), global epsilon is equal on all ions (+0.011 on shared), eq
  is +0.013 (E. coli). Ions: -15.2%.

## 3. Diagnosis (Q0, done 2026-10-01)

Scripts and results are in `/public/local/ProteoBench/HYE_diaPASEF_mumdia/quant_diag/`.
Ions are keyed on I/L-merged stripped sequence + charge.

**Q0.1 The gap is in quantification, not in the identification set.** At `min_obs` 3 there
are 77,148 shared ions, 15,235 MuMDIA-only ions and 11,753 DIA-NN-only ions. On the shared
ions:

| | median abs eps | CV | E. coli | yeast | human |
|---|---|---|---|---|---|
| DIA-NN | 0.111 | 0.079 | -1.85 | +0.82 | -0.03 |
| MuMDIA `eng_repick` | 0.161 | 0.114 | -1.46 | +0.70 | +0.03 |

CV in this table and in the later shared-ion tables is the median of (CV_A + CV_B) / 2. It is
close to, but not identical to, ProteoBench's `CV_median`.

**Q0.2 The apex is not the cause.** Over 431,666 run-level rows shared with DIA-NN, MuMDIA's
integration apex is within 5 s of DIA-NN's `RT` in 99.5% of rows in every species and
condition. The median offset is 0.96 s, one cycle. Excluding the 0.5% off-apex rows does not
change the ratios.

**Q0.3 The compression is additive and sits in the low condition.** The median
log2(MuMDIA / DIA-NN) is -1.66 / -1.67 for human (A / B). For E. coli, which is 4x lower in
A, it is -1.50 in A and -1.66 in B. MuMDIA's quantity carries a floor that matters only
where the signal is low.

**Q0.4 The floor comes from wide integration windows.** The descent walk (`peak_fraction`
1/6, `peak_grace` 1) runs on raw retrace traces, which have no noise floor. On a weak peak it
often never drops below 1/6 of the apex. The median integration width is 6.8 s (DIA-NN
`RT.Stop - RT.Start` 8.7 s). The p90 is 44 s against DIA-NN's 11.6 s, and 27% of rows
integrate over more than 15 s. The tail is larger in the low condition: 35% of E. coli rows in
A against 29% in B. Shared E. coli ions binned by the wider of their two condition widths:

| width (s) | ions | MuMDIA log2 A/B | DIA-NN log2 A/B |
|---|---|---|---|
| <= 5 | 513 | -1.74 | -1.82 |
| 5-8 | 492 | -1.73 | -1.85 |
| 8-12 | 261 | -1.64 | -1.86 |
| 12-20 | 242 | -1.52 | -1.81 |
| > 20 | 608 | -1.11 | -1.78 |

Ions with a DIA-NN-like width are close to DIA-NN's ratio. The wide tail causes most of the
compression.

**Q0.5 Normalisation is a small share.** Without DIA-NN's normalisation, DIA-NN still reaches
0.120 / CV 0.080 / E. coli -1.79. DIA-NN's normalisation factors are 0.97 (A) and 1.01 (B) on
median.

**Q0.6 DIA-NN's quantity is not a top-N fragment sum.** The median `Precursor.Quantity` is
1.5x the sum of all 12 `Fragment.Quant.Raw` values, and 2.7x the top-3 by area. This fits a
model-based estimate across fragments (and possibly MS1), as described for QuantUMS in DIA-NN
2.x. It is not a fixed top-N rule.

**Q0.7 The MuMDIA-only ions are faint and quantify worst.** They are about 0.6 log2 below the
shared ions (median max-condition log intensity 13.85 against 14.44), with |epsilon| 0.255,
CV 0.15 and E. coli at -1.0. About half are DIA-NN identifications that DIA-NN quantified in
fewer than 3 runs. Their cross-run apex spread is no worse than for the shared ions (global RT
offset per run, 30% against 24% above 10 s), so a cross-run apex inconsistency does not explain
them. A proper LOESS-aligned check is still open (Q4).

## 4. First lever measured: a fixed integration window (Q1)

Requant only, on the `eng_repick` identifications (`quant_diag/requant_repick.sh <name>
'<edit of q>'`), with the base quant config otherwise unchanged. All arms have 92,540 ions,
because the IDs are fixed.

| arm | median abs eps | eq | CV | E. coli | yeast | human |
|---|---|---|---|---|---|---|
| `eng_repick` (descent walk) | 0.172 | 0.358 | 0.107 | -1.43 | +0.66 | +0.03 |
| `peak_window_mode: consensus` | 0.153 | 0.242 | 0.108 | -1.66 | +0.80 | +0.02 |
| `fixed_window_s: 3` | 0.142 | 0.232 | 0.098 | -1.67 | +0.79 | +0.01 |
| **`fixed_window_s: 4`** | **0.136** | **0.230** | **0.089** | **-1.68** | **+0.79** | +0.01 |
| `fixed_window_s: 5` | 0.135 | 0.232 | 0.086 | -1.67 | +0.79 | +0.02 |
| `fixed_window_s: 4` + `baseline_subtract` | 0.138 | **0.217** | 0.092 | **-1.71** | +0.81 | +0.01 |
| `fixed_window_s: 4` + `top_n_fragments: 6` | **0.129** | 0.256 | **0.080** | -1.60 | +0.76 | +0.02 |
| `fixed_window_s: 4`, `interference_envelope` off | 0.131 | 0.260 | 0.082 | -1.60 | +0.75 | +0.02 |
| DIA-NN | 0.118 | 0.169 | 0.076 | -1.84 | +0.81 | -0.03 |

The same arms, split into shared and MuMDIA-only ions:

| arm | shared eps / CV / E. coli | MuMDIA-only eps / CV / E. coli |
|---|---|---|
| `eng_repick` | 0.161 / 0.114 / -1.46 | 0.255 / 0.149 / -1.02 |
| `fixed_window_s: 4` | 0.129 / 0.091 / -1.69 | 0.194 / 0.121 / -1.44 |
| + `baseline_subtract` | 0.130 / 0.095 / -1.72 | 0.202 / 0.129 / -1.47 |
| + `top_n_fragments: 6` | 0.122 / 0.082 / -1.60 | 0.180 / 0.108 / -1.33 |
| DIA-NN | 0.111 / 0.079 / -1.85 | |

Reading:
- One existing key closes about two thirds of the epsilon gap and 60% of the CV gap at the same
  ion count. 4 s and 5 s are equal. 3 s loses precision. Consensus widths fix the ratio but
  not the CV.
- On the shared ions, yeast is now at DIA-NN's value (+0.81 against +0.82). E. coli is still
  0.16 log2 short, and the CV is 0.091 against 0.079.
- **More fragments buy precision and cost accuracy.** Top-6, or switching the envelope off,
  gives DIA-NN-level CV (0.080 to 0.082) and compresses E. coli back to -1.60. Some of the
  added fragments are interfered. A fixed count cannot get both, so the next lever is
  fragment choice (Q2). Under the targets of section 1, top-6 is rejected: its global epsilon
  improves (0.129), but its species-equalised epsilon gets worse (0.256 against 0.230).
- Baseline subtraction removes part of the remaining floor (E. coli -1.71, eq 0.217) at a
  small CV cost. It gives the best species-equalised epsilon of all arms, and its global
  epsilon is within 0.002 of the best fixed window. It is therefore the working base for Q2
  and Q3. Remaining gap to DIA-NN: global 0.138 against 0.118, eq 0.217 against 0.169.
- The MuMDIA-only ions are still the worst group (0.194). They are 16% of the ions and
  carry a disproportionate share of the global epsilon.

### Q1 sweep: window unit and baseline (2026-10-01, seed 0)

Requants on the `eng_repick` IDs (92,540 ions in every arm). "Shared" is the 77,282 ions also
quantified by DIA-NN at `min_obs` 3, "only" the 15,258 MuMDIA-only ions (`quant_diag/shared.py`;
the shared count differs slightly from Q0.1 because the scripts differ). Shared and only columns are
eps / CV / E. coli.

| arm | global | eq | CV | E. coli | shared | only |
|---|---|---|---|---|---|---|
| `fixed_window_s: 4` + baseline (12, 0.25) | 0.138 | 0.217 | 0.092 | -1.71 | 0.130 / 0.092 / -1.72 | 0.202 / 0.128 / -1.47 |
| `fixed_scan_halfwidth: 4` | 0.136 | 0.230 | 0.089 | -1.68 | 0.129 / 0.089 / -1.69 | 0.194 / 0.121 / -1.44 |
| `fixed_scan_halfwidth: 4` + baseline (12, 0.25) | 0.138 | 0.217 | 0.092 | -1.71 | 0.130 / 0.092 / -1.72 | 0.202 / 0.128 / -1.47 |
| + `baseline_flank_scans: 6` | 0.140 | 0.215 | 0.095 | -1.72 | 0.131 / 0.094 / -1.73 | 0.207 / 0.134 / -1.47 |
| + `baseline_flank_scans: 24` | 0.138 | 0.218 | 0.091 | -1.70 | 0.129 / 0.091 / -1.71 | 0.198 / 0.125 / -1.46 |
| + `baseline_quantile: 0.1` | 0.137 | 0.226 | 0.090 | -1.69 | 0.129 / 0.090 / -1.70 | 0.196 / 0.123 / -1.45 |
| + `baseline_quantile: 0.4` | 0.139 | 0.207 | 0.095 | -1.74 | 0.130 / 0.094 / -1.75 | 0.206 / 0.132 / -1.47 |
| + `baseline_quantile: 0.5` | 0.142 | 0.198 | 0.098 | -1.77 | 0.133 / 0.098 / -1.78 | 0.216 / 0.140 / -1.52 |
| + `baseline_quantile: 0.6` | 0.145 | 0.195 | 0.101 | -1.81 | 0.135 / 0.101 / -1.82 | 0.224 / 0.145 / -1.58 |
| + `baseline_quantile: 0.75` | 0.153 | 0.196 | 0.109 | -1.87 | 0.142 / 0.108 / -1.88 | 0.239 / 0.159 / -1.61 |

The `fixed_window_s` arms with flank or quantile changes were run as `fixed_window_s: 4`; the
scan and seconds forms are identical on this data, so the rows are comparable.

Reading:
- `fixed_scan_halfwidth: 4` and `fixed_window_s: 4` give identical results, with and without the
  baseline. The scan form is preferred, as planned.
- The flank length (6, 12, 24) does not matter (within 0.002 in both epsilons).
- The baseline quantile is a monotone trade: a higher quantile removes more of the floor (E. coli
  -1.69 at 0.1 to -1.87 at 0.75), lowers eq down to 0.6 and raises global epsilon and CV. At 0.75 the
  ratios overshoot (E. coli -1.87, yeast +0.89), so the flank estimate there includes more than
  background. Without Q2, no quantile improves both epsilons.

## 4b. Fragment weighting by cross-run consistency (Q2 prototype, 2026-10-01, seed 0)

Offline Python on `chromatograms.parquet` (`quant_diag/q2_frag_areas.py`, `q2_combine.py`).
`q2_frag_areas.py` replicates quant's fixed window (`fixed_scan_halfwidth: 4`), flank baseline
and interference envelope per fragment; it reproduces the engine quantity on 99.92% of run 0
candidates, and its top-3 rule reproduces the engine arms exactly (0.138 / 0.217 / 0.092 at
quantile 0.25, 0.142 / 0.198 / 0.098 at 0.5). No variant uses condition or species labels.

**Correction: the prototype includes MS1.** `chromatograms.parquet` also holds the three MS1
traces of each candidate (`ms1_mono`, `ms1_iso1`, `ms1_iso2`, predicted intensity 0). Quant
never reads them, but the prototype did, so every `cons` and `wsum` row in this section treats them
as three more channels: these arms are fragments plus MS1, weighted. The top-3 and `fixed` rules
rank by predicted intensity and are unaffected. The fragment-only weighting, as implemented in the
engine, is in section 4c.

Variants, all one fragment set or one weight vector per precursor, used in every run:
- `fixed k`: top-k by predicted intensity among fragments with a positive area in at least half
  of the runs.
- `cons k`: fragments with a median apex correlation >= 0.5, ranked by the cross-run deviation of
  their log-share of the precursor total (low first), up to k. Correlation is the Pearson
  correlation of the fragment's 9 windowed samples with the sum of the other fragments, per run.
- `wsum`: every fragment, weight 1 / (median |log-share deviation| + 0.1) x max(median
  correlation, 0).
- Ion-preserving fallbacks: a precursor with no passing fragment keeps the `fixed` set (`cons`) or
  an unweighted sum (`wsum`); a run where the consistent quantity is zero takes the engine top-3
  rule for that run (7 of about 540,000 run-level rows for `wsum` at 0.5). The `cons` rows below
  predate the second fallback and lose 68 to 1,407 ions.

| arm | ions | global | eq | CV | E. coli | shared | only |
|---|---|---|---|---|---|---|---|
| baseline q 0.25, `fixed 6` | 92,540 | 0.131 | 0.234 | 0.083 | -1.65 | 0.123 / 0.083 / -1.67 | 0.188 / 0.116 / -1.37 |
| baseline q 0.25, `cons 6` | 91,158 | 0.121 | 0.238 | 0.071 | -1.65 | 0.112 / 0.071 / -1.67 | 0.190 / 0.102 / -1.38 |
| baseline q 0.25, `wsum` | 92,527 | 0.118 | 0.248 | 0.068 | -1.62 | 0.110 / 0.067 / -1.64 | 0.183 / 0.096 / -1.35 |
| baseline q 0.4, `wsum` | 92,540 | 0.121 | 0.223 | 0.071 | -1.69 | 0.112 / 0.070 / -1.70 | 0.187 / 0.102 / -1.42 |
| baseline q 0.5, `fixed 6` | 92,540 | 0.134 | 0.205 | 0.089 | -1.74 | 0.125 / 0.089 / -1.75 | 0.201 / 0.126 / -1.42 |
| baseline q 0.5, `cons 3` | 92,441 | 0.132 | 0.209 | 0.084 | -1.77 | 0.121 / 0.083 / -1.79 | 0.213 / 0.124 / -1.54 |
| baseline q 0.5, `cons 6` | 92,472 | 0.126 | 0.203 | 0.079 | -1.78 | 0.116 / 0.077 / -1.79 | 0.207 / 0.117 / -1.52 |
| **baseline q 0.5, `wsum`** | **92,540** | **0.123** | 0.203 | **0.074** | -1.76 | 0.114 / 0.074 / -1.77 | 0.197 / 0.109 / -1.50 |
| baseline q 0.6, `wsum` | 92,540 | 0.126 | **0.197** | 0.078 | -1.81 | 0.116 / 0.077 / -1.82 | 0.205 / 0.115 / -1.55 |
| `eng_repick` as shipped | 92,540 | 0.172 | 0.358 | 0.107 | -1.43 | 0.161 / 0.110 / -1.46 | 0.255 / 0.151 / -1.02 |
| DIA-NN | 88,924 | 0.118 | 0.169 | 0.076 | -1.84 | 0.111 / 0.079 / -1.85 | |

Reading:
- Consistency weighting buys precision and global epsilon. On the shared ions `wsum` reaches
  DIA-NN's epsilon and beats its CV (0.110 / 0.067 against 0.111 / 0.079 at quantile 0.25).
  Weighting matters: `fixed 6` gets a third of the gain, so it is not the fragment count alone.
- Alone it compresses the ratios (E. coli -1.62 at quantile 0.25), so eq gets worse. A stronger
  baseline removes that compression, and the two levers combine: `wsum` at quantile 0.5 is better
  than the Q1 base in both epsilons (0.123 / 0.203 against 0.138 / 0.217) and in CV (0.074 against
  0.092), at the same ion count. It meets both Q2 targets (eq < 0.217, global < 0.129).
- Quantile 0.5 and 0.6 are an even trade on seed 0 (global -0.003 / eq +0.006). The seed
  replicates decide.
- The MuMDIA-only ions remain the worst group (0.197), now Q4's target.

**Replication on two more ID sets (2026-10-01).** `eng_repick_s1` and `eng_repick_s2` re-run only
the pooled `nn_torch` rescore of `eng_repick` with `MUMDIA_NN_SEED` 1 and 2
(`quant_diag/seed_ids.sh`, `seed_eval.sh`). Peptides at 1%: 100,790 / 100,064 / 100,800 (seeds
0 / 1 / 2).

| seed | arm | ions | global | eq | CV | E. coli | shared | only |
|---|---|---|---|---|---|---|---|---|
| 0 | shipped | 92,540 | 0.172 | 0.358 | 0.107 | -1.43 | 0.161 / 0.110 / -1.46 | 0.255 / 0.151 / -1.02 |
| 0 | Q1 base | 92,540 | 0.138 | 0.217 | 0.092 | -1.71 | 0.130 / 0.092 / -1.72 | 0.202 / 0.128 / -1.47 |
| 0 | `wsum` q 0.5 | 92,540 | 0.123 | 0.203 | 0.074 | -1.76 | 0.114 / 0.074 / -1.77 | 0.197 / 0.109 / -1.50 |
| 0 | `wsum` q 0.6 | 92,540 | 0.126 | 0.197 | 0.078 | -1.81 | 0.116 / 0.077 / -1.82 | 0.205 / 0.115 / -1.55 |
| 1 | shipped | 92,141 | 0.172 | 0.354 | 0.107 | -1.44 | 0.161 / 0.110 / -1.47 | 0.255 / 0.152 / -1.03 |
| 1 | Q1 base | 92,141 | 0.138 | 0.217 | 0.092 | -1.71 | 0.130 / 0.092 / -1.72 | 0.201 / 0.128 / -1.47 |
| 1 | `wsum` q 0.5 | 92,141 | 0.122 | 0.202 | 0.074 | -1.76 | 0.114 / 0.074 / -1.77 | 0.194 / 0.108 / -1.51 |
| 1 | `wsum` q 0.6 | 92,141 | 0.125 | 0.197 | 0.078 | -1.81 | 0.116 / 0.077 / -1.82 | 0.199 / 0.114 / -1.55 |
| 2 | shipped | 92,475 | 0.172 | 0.357 | 0.107 | -1.43 | 0.161 / 0.110 / -1.46 | 0.252 / 0.151 / -1.02 |
| 2 | Q1 base | 92,475 | 0.138 | 0.218 | 0.092 | -1.71 | 0.130 / 0.092 / -1.72 | 0.203 / 0.129 / -1.49 |
| 2 | `wsum` q 0.5 | 92,475 | 0.123 | 0.203 | 0.075 | -1.76 | 0.114 / 0.074 / -1.77 | 0.197 / 0.108 / -1.49 |
| 2 | `wsum` q 0.6 | 92,475 | 0.126 | 0.198 | 0.078 | -1.81 | 0.116 / 0.077 / -1.82 | 0.203 / 0.115 / -1.55 |

Q1 base is `fixed_scan_halfwidth: 4` + `baseline_subtract` (12, 0.25). Shared ions: 77,282 /
77,129 / 77,257.

- Every arm reproduces within 0.004 across the three ID sets, and every gain has the same size on
  each: `wsum` q 0.5 against the Q1 base is -0.015 to -0.016 global, -0.015 eq, -0.017 to -0.018 CV.
  `wsum` keeps every ion of its ID set. The Q1 and Q2 gates on HYE diaPASEF are met.
- Quantile 0.6 against 0.5 is the same trade on every seed: eq -0.005 to -0.006, global +0.003,
  CV +0.003 to +0.004, and E. coli 0.05 closer. 0.6 was chosen on these numbers, which include
  MS1 (see the correction above); section 4c revisits it for the fragment-only engine lever.
- Submission: `bench/proteobench_input/q05_wsum/custom_input.tsv` (baseline q 0.5, `wsum`).
- Gap closed from `eng_repick` to DIA-NN: about 90% in global epsilon, 82% in eq.

## 4c. Engine lever: `quant.cross_run_weights` (2026-10-01)

Implemented as `quant.cross_run_weights` (`QuantConfig`, quant.rs `fit_fragment_weights`). Per-run
quant runs twice: pass 1 writes `fragment_quant` (v2, new `apex_corr` column: correlation of the
fragment's fixed-window samples with the sum of the candidate's other fragments); the weights are
fitted over all runs as in the prototype; pass 2 sums `weight * area`. A candidate with all-zero
weights takes equal weights; a run with no positive weighted area keeps the `top_n_fragments`
rule, so no ion is lost. `run-experiment` does both passes; standalone, `mumdia quant
--weights-from <pass-1 fragment tables>` (harness: `quant_diag/requant_crw.sh`). Fragments only:
the MS1 traces are not quant channels.

Parity: on seed 0 at quantile 0.6 the engine equals the prototype run without the MS1 rows
(0.130 / 0.190 / 0.087 all ions, 0.122 / 0.086 / -1.79 shared, in both).

All arms `fixed_scan_halfwidth: 4`, `baseline_subtract` (flank 12), predicted selection, envelope,
`cross_run_weights: true`; binary `~/bin/mumdia-crw/mumdia`.

| seed | baseline q | ions | global | eq | CV | E. coli | yeast | shared | only |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 92,540 | 0.129 | 0.200 | 0.084 | -1.74 | +0.82 | 0.120 / 0.083 / -1.75 | 0.200 / 0.121 / -1.47 |
| 0 | **0.6** | 92,540 | 0.130 | **0.190** | 0.087 | -1.78 | +0.84 | 0.122 / 0.086 / -1.79 | 0.206 / 0.126 / -1.51 |
| 1 | 0.5 | 92,141 | 0.129 | 0.200 | 0.084 | -1.74 | +0.82 | 0.120 / 0.083 / -1.75 | 0.198 / 0.121 / -1.47 |
| 1 | **0.6** | 92,141 | 0.130 | **0.190** | 0.086 | -1.78 | +0.84 | 0.122 / 0.086 / -1.79 | 0.205 / 0.125 / -1.51 |
| 2 | 0.5 | 92,475 | 0.129 | 0.201 | 0.084 | -1.74 | +0.82 | 0.120 / 0.083 / -1.75 | 0.201 / 0.121 / -1.47 |
| 2 | **0.6** | 92,475 | 0.131 | **0.191** | 0.087 | -1.78 | +0.84 | 0.122 / 0.086 / -1.79 | 0.209 / 0.126 / -1.52 |

Against the Q1 base on the same IDs (0.138 / 0.217 / 0.092): -0.008 global, -0.027 eq, -0.005 CV at
quantile 0.6, on every seed. Without MS1, quantile 0.6 is the better setting: eq -0.010 for
global +0.001 against 0.5. It is the quantile in the diaPASEF preset.

The diaPASEF preset (`QuantConfig::diapasef`): predicted selection, envelope,
`fixed_scan_halfwidth: 4`, `baseline_subtract` with quantile 0.6, `cross_run_weights`. It replaces
a `quant` block left at its defaults when every input is a timsTOF `.d`; any explicitly set quant
key keeps the block as written. Under single-run `run` the weights are not applied (nothing to fit).

Submissions: `bench/proteobench_input/crw_q06/` is the engine lever (seed 0). The earlier
`bench/proteobench_input/q05_wsum/` is the prototype with MS1 and quantile 0.5, not an engine result.

MS1 as weighted channels (the prototype, quantile 0.6, seed 0: 0.126 / 0.197 / 0.078) is a Q5 lever:
against the fragment-only engine it is -0.004 global and -0.009 CV for +0.007 eq.

## 4d. Q4 measured: the MuMDIA-only ions are faint, not mis-peaked (2026-10-01)

`quant_diag/q4_diag.py` on the engine lever (section 4c, quantile 0.6, seed 0; 15,258 MuMDIA-only
and 77,282 shared ions). Each run's apex RT is mapped onto the cross-run median by a binned-median
fit on confident shared ions (`run_psm_q` < 0.001); spread is the largest |run - median| per ion.

| group | apex spread | ions | abs eps | E. coli ions | E. coli log2 |
|---|---|---|---|---|---|
| only | <= 2 s | 11,326 | 0.200 | 927 | -1.41 |
| only | 2-5 s | 3,094 | 0.195 | 214 | -1.77 |
| only | 5-10 s | 476 | 0.282 | 25 | -1.77 |
| only | > 10 s | 362 | 0.541 | 16 | -1.26 |
| shared | <= 2 s | 58,751 | 0.120 | 3,668 | -1.79 |
| shared | > 10 s | 412 | 0.529 | 14 | -0.65 |

- Wrong peaks are rare: 5.4% of the MuMDIA-only ions (838) have a spread above 5 s. Re-integrating
  them at a consensus apex cannot move the median, so the consensus-apex lever of Q4 is dropped.
- The error follows intensity and completeness. By max-condition intensity quintile the
  MuMDIA-only ions go from 0.245 (E. coli -1.00) to 0.130 (-1.79); the shared ions from 0.189 to
  0.088. MuMDIA-only ions observed in all six runs quantify at 0.142 (E. coli -1.81, 4,233 ions);
  those observed in one or two runs of a condition at 0.19 to 0.32. 28% of the MuMDIA-only ions are
  complete, against 73% of the shared ions.
- Within-run confidence does not separate them: of the complete ions, all but 14 pass `run_psm_q`
  < 0.01 in every A run.
- Completeness is not a MuMDIA deficit. Over all ions at `min_obs` 3:

| | complete (3A + 3B) | abs eps complete | abs eps partial | global |
|---|---|---|---|---|
| DIA-NN | 57,882 (65.1%) | 0.096 | 0.196 | 0.118 |
| MuMDIA, section 4c | 60,285 (65.1%) | 0.107 | 0.214 | 0.131 |

The remaining gap is a per-ion gap on weak signal, present in both strata (+0.011 complete, +0.018
partial). That is Q3's and Q5's target, not Q4's. Q4 is closed; a per-ion quality value is still
possible as a report column and is not a priority.

## 4e. Q3 measured: neither a narrower band nor a mobility background helps (2026-10-01)

Requant only, on the seed 0 `eng_repick` IDs, against the engine lever of section 4c (fixed 9-scan
window, flank baseline q 0.6, cross-run weights).

**Fragment 1/K0 band.** `quant_diag/retrace_band.sh` rebuilds the traces with
`retrace.im_half_width` changed, centred on the repicked apices (`repick` off, the repick PSM table
as input, extract's centroid traces from `l1d/pass2_robust`). At 0.015 it reproduces the shipped
traces exactly (run 0: 89,853 of 89,853 quantities equal). Then the engine quant
(`requant_crw.sh`).

| band (1/K0) | global | eq | CV | E. coli | shared | only |
|---|---|---|---|---|---|---|
| +/- 0.0075 | 0.142 | 0.190 | 0.098 | -1.80 | 0.132 / 0.098 / -1.81 | 0.220 / 0.144 / -1.61 |
| +/- 0.010 | 0.134 | 0.186 | 0.091 | -1.80 | 0.125 / 0.090 / -1.81 | 0.210 / 0.133 / -1.59 |
| **+/- 0.015 (shipped)** | **0.130** | 0.190 | **0.087** | -1.78 | 0.122 / 0.086 / -1.79 | 0.206 / 0.126 / -1.51 |
| +/- 0.020 | 0.133 | 0.197 | 0.088 | -1.77 | 0.124 / 0.086 / -1.78 | 0.211 / 0.126 / -1.45 |

A narrower band removes a little interference (E. coli -1.80) and cuts more of the ion's own signal
(global and CV up); a wider one is worse on both epsilons. The best trade (0.010: eq -0.004, global
+0.004) is below the bar. The quant band stays the identification band.

**Mobility background.** `quant_diag/retrace_side.sh` builds the same traces with every apex_im
shifted by +0.05 and -0.05 1/K0 (same fragment m/z, same frames, a band next to the precursor's).
`q2_frag_areas.py` (`SIDE=`) subtracts the mean side-band trace per sample before the window; the
combiner is fragments only, as the engine. The control arm reproduces the engine (0.130 / 0.190 /
0.087).

| background | global | eq | CV | E. coli | shared | only |
|---|---|---|---|---|---|---|
| RT flank, q 0.6 (engine, control) | **0.130** | **0.190** | 0.087 | -1.78 | 0.122 / 0.086 / -1.79 | 0.206 / 0.126 / -1.51 |
| mobility side band only | **0.130** | 0.230 | **0.081** | -1.67 | 0.122 / 0.080 / -1.68 | 0.192 / 0.114 / -1.42 |
| mobility side band + RT flank | 0.134 | 0.189 | 0.090 | -1.80 | 0.125 / 0.088 / -1.81 | 0.212 / 0.131 / -1.58 |

- The side band removes much less of the floor than the RT flank (E. coli -1.67 against -1.78), so
  the floor is not co-mobile background in the next band. It sits at the ion's own mobility, which
  fits co-eluting signal in the same 1/K0 band (interference or the ion's own tail) rather than a
  diffuse background.
- On top of the RT flank it adds removal (E. coli -1.80) at a precision cost (global +0.004, CV
  +0.003). No gain on the targets.
- Q3 is closed with no lever. The narrow-band and side-band traces are in
  `/public/local/ProteoBench/HYE_diaPASEF_mumdia/q3_band/` and `q3_side/` (645 GB).

## 4f. Where eq is lost, and a cross-run pooled background (2026-10-01, seed 0)

**Eq is an E. coli problem.** Eq is the mean of the three species' median |epsilon|. Engine lever
(section 4c) against DIA-NN: E. coli 0.254 / 0.198, human 0.119 / 0.103, yeast 0.198 / 0.205. On
the complete shared ions both tools differ by a near-uniform offset (MuMDIA log2 A/B - DIA-NN:
human +0.045, yeast +0.040, E. coli +0.065), and on those ions E. coli sits at -1.82 against
DIA-NN's -1.88 and yeast at about +0.90 against +0.85. Both of our non-human species are compressed,
so a run-scale shift helps one and hurts the other.

**Rejected on the way** (prototype, fragments only, engine control 0.130 / 0.190 / 0.087):
- Per-run robust scaling of the cross-run fragment profile (weighted percentile of
  area / reference share): p50 0.136 / 0.190 / 0.094, p35 0.148 / 0.191 / 0.104 (E. coli -1.88). The
  floor is not interference on single fragments in single runs.
- Sharper cross-run weights, `(dev + d0)^-p`: every arm E. coli -1.78 and worse precision (best
  d0 0.03: 0.132 / 0.192 / 0.088).
- Trimmed flank mean instead of the flank quantile: full mean 0.134 / 0.184 / 0.090, 90% 0.130 /
  0.192 / 0.085, 75% 0.127 / 0.211 / 0.081. The same trade curve as the quantile.
- Label-free median run normalisation (Q6), one factor per run from the ions quantified in all six
  runs: engine 0.133 / 0.191 / 0.086, pooled arm below 0.134 / 0.180 / 0.087. It centres the mixture,
  not human (human -0.03, as DIA-NN's normalisation does), and moves nothing. Q6 is closed.

**The floor is sparse noise.** In run 0, half of all flank samples of quantified fragments are 0, and
the flank median is 0 for 48% of fragments. Per sample (median over about 1,500 precursors per
species): flank q 0.6 76, flank mean 88, window mean 237 (E. coli). A quantile of a sparse flank
underestimates the mean noise the window collects; the flank mean estimates it, but from 24 samples
of one run it is noisy, and subtracting a noisy level costs precision.

**Cross-run pooled background.** The fragment's background is in the same m/z and 1/K0 band at the
same aligned RT in every run, so one level per (candidate, fragment) is taken as the mean over the
six runs of the per-run flank means and subtracted per sample before the envelope; the cross-run
weights are then fitted as usual (`q2_frag_areas.py POOLRAW=1`, `q2_combine.py POOL=mean
POOL_SCALE=`). Label-blind.

| background | global | eq | CV | E. coli | yeast | E. coli / human / yeast abs eps | shared | only |
|---|---|---|---|---|---|---|---|---|
| per run, q 0.6 (engine) | **0.130** | 0.190 | 0.087 | -1.78 | +0.84 | 0.254 / **0.119** / 0.198 | 0.122 / 0.086 / -1.79 | 0.206 / 0.126 / -1.51 |
| per run, flank mean | 0.134 | 0.184 | 0.090 | -1.81 | +0.87 | 0.239 / 0.124 / 0.191 | 0.125 / 0.089 / -1.82 | 0.215 / 0.132 / -1.54 |
| pooled mean x 0.6 | **0.130** | 0.196 | **0.083** | -1.76 | +0.84 | 0.272 / 0.118 / 0.199 | 0.121 / 0.082 / -1.77 | 0.199 / 0.119 / -1.51 |
| pooled mean x 0.8 | 0.132 | 0.187 | 0.085 | -1.80 | +0.86 | 0.250 / 0.121 / 0.190 | 0.123 / 0.084 / -1.81 | 0.205 / 0.123 / -1.55 |
| pooled mean x 1.0 | 0.134 | **0.180** | 0.087 | **-1.84** | +0.89 | **0.232** / 0.124 / **0.185** | 0.125 / 0.086 / -1.84 | 0.211 / 0.127 / -1.58 |
| pooled median x 1.0 | 0.133 | 0.181 | 0.087 | -1.83 | +0.88 | 0.233 / 0.123 / 0.185 | 0.124 / 0.086 / -1.84 | 0.211 / 0.127 / -1.59 |
| DIA-NN | 0.118 | 0.169 | 0.076 | -1.84 | +0.81 | 0.198 / 0.103 / 0.205 | 0.111 / 0.079 / -1.85 | |

- Pooling removes the precision cost of a mean background: at full scale CV is unchanged (0.087)
  where the per-run mean costs +0.003, and eq falls to 0.180, with E. coli at DIA-NN's -1.84 and
  yeast nearer +1 than DIA-NN.
- It is still a trade with global epsilon: human |eps| rises (0.119 to 0.124), because subtracting
  the floor leaves faint human ions with less signal and the same noise. At equal global epsilon
  (x 0.6) it buys CV, not eq. No point is better than the engine on both epsilons.
- Open decision: whether eq -0.010 for global +0.004 (x 1.0) is worth taking.

**Replication and the MS1 combination** (`quant_diag/pool_eval.sh`; pooled mean x 1.0 with cross-run
weights; "+ MS1" also uses the three MS1 traces as weighted channels, with the same pooled background).

| seed | arm | ions | global | eq | CV | E. coli | E. coli / human / yeast abs eps | shared | only |
|---|---|---|---|---|---|---|---|---|---|
| 0 | engine (section 4c) | 92,540 | 0.130 | 0.190 | 0.087 | -1.78 | 0.254 / 0.119 / 0.198 | 0.122 / 0.086 / -1.79 | 0.206 / 0.126 / -1.51 |
| 0 | pooled | 92,540 | 0.134 | 0.180 | 0.087 | -1.84 | 0.232 / 0.124 / 0.185 | 0.125 / 0.086 / -1.84 | 0.211 / 0.127 / -1.58 |
| 0 | pooled + MS1 | 92,540 | 0.127 | 0.198 | 0.076 | -1.88 | 0.281 / 0.115 / 0.196 | 0.118 / 0.075 / -1.90 | 0.209 / 0.113 / -1.64 |
| 1 | engine | 92,141 | 0.130 | 0.190 | 0.086 | -1.78 | 0.254 / 0.118 / 0.198 | 0.122 / 0.086 / -1.79 | 0.205 / 0.125 / -1.51 |
| 1 | pooled | 92,141 | 0.133 | 0.180 | 0.087 | -1.84 | 0.232 / 0.123 / 0.185 | 0.124 / 0.086 / -1.84 | 0.210 / 0.127 / -1.58 |
| 1 | pooled + MS1 | 92,141 | 0.126 | 0.197 | 0.076 | -1.88 | 0.280 / 0.115 / 0.195 | 0.117 / 0.075 / -1.89 | 0.207 / 0.112 / -1.64 |
| 2 | engine | 92,475 | 0.131 | 0.191 | 0.087 | -1.78 | 0.256 / 0.119 / 0.199 | 0.122 / 0.086 / -1.79 | 0.209 / 0.126 / -1.52 |
| 2 | pooled | 92,475 | 0.134 | 0.180 | 0.087 | -1.84 | 0.232 / 0.124 / 0.185 | 0.124 / 0.086 / -1.85 | 0.213 / 0.128 / -1.59 |
| 2 | pooled + MS1 | 92,475 | 0.127 | 0.198 | 0.076 | -1.88 | 0.282 / 0.115 / 0.196 | 0.117 / 0.075 / -1.89 | 0.210 / 0.113 / -1.64 |

- The pooled background replicates exactly on three ID sets: eq -0.010 to -0.011, global +0.003 to
  +0.004, CV +0.000 to +0.001 against the engine.
- MS1 channels on top buy global epsilon and CV (0.127 / 0.076, DIA-NN's CV) but widen E. coli: its
  median moves closer to -2 (-1.88) while its median |eps| rises to 0.28. The MS1 traces of faint
  E. coli ions in condition A are not clean enough. Against the engine the combination is global
  -0.003, CV -0.011, eq +0.008: a trade the other way, not a gain on both.

## 4g. Engine lever: `quant.cross_run_background` (2026-10-01)

Decision: take the pooled background, fragments only (eq first); keep MS1 out (Q5 stays open with
the section 4f result). Implemented as `quant.cross_run_background` in the two-pass step of
`cross_run_weights` (`fit_cross_run`): pass 1 also exports each fragment's `flank_mean` (and,
with this key, zero-area fragments; `fragment_quant` v3); the fit averages it over all runs per
(candidate, fragment); pass 2 subtracts that level from every window sample in place of the per-run
quantile. The weights stay fitted on pass 1 (per-run baseline) areas: the prototype with weights
from the pooled-background areas (three passes) gives the same result (0.134 / 0.180 / 0.087, shared
0.125 / 0.086 / -1.84 against 0.124 / 0.086 / -1.85 for two passes). Part of the diaPASEF preset.

Engine (`quant_diag/requant_crw.sh`, `BG=1`), quantile 0.6 is still used in pass 1:

| seed | ions | global | eq | CV | E. coli | yeast | E. coli / human / yeast abs eps | shared | only |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 92,540 | 0.134 | 0.180 | 0.087 | -1.84 | +0.89 | 0.231 / 0.123 / 0.185 | 0.124 / 0.086 / -1.85 | 0.211 / 0.128 / -1.60 |
| 1 | 92,141 | 0.133 | 0.180 | 0.087 | -1.84 | +0.89 | 0.232 / 0.123 / 0.186 | 0.124 / 0.086 / -1.85 | 0.209 / 0.128 / -1.60 |
| 2 | 92,475 | 0.134 | 0.181 | 0.087 | -1.84 | +0.89 | 0.233 / 0.123 / 0.186 | 0.124 / 0.086 / -1.85 | 0.212 / 0.129 / -1.60 |

Against section 4c on the same IDs: eq -0.010, global +0.003 to +0.004, CV unchanged, E. coli at
DIA-NN's -1.84, same ion count. From the start of this roadmap (0.172 / 0.358 / 0.107): eq closed by
about 94% of the gap to DIA-NN, global by about 70%. Submission: `bench/proteobench_input/crwbg/`.

Remaining gap to DIA-NN (0.118 / 0.169 / 0.076): global +0.016, eq +0.011, CV +0.011. Human
precision on faint ions is now the largest share of the global gap (human |eps| 0.123 against
0.103).

## 4h. The MBR-off ion gap is per-run identification depth (2026-10-01, seed 0)

Scripts `quant_diag/gap.py`, `runrows.py` and `wrongpeak.py`; the scored DIA-NN intermediates and
the per-ion class tables are in `quant_diag/pb_targets/`. Key: I/L-merged precursor ion. "Identified" in a run means target with
pooled `q_value` <= 0.01, the gate quant uses.

**Ion level.** Of the 98,694 DIA-NN 2.5.0 ions at `min_obs` 3, 81,665 are ours too. The 17,029
DIA-NN-only ions:

| class | ions | E. coli / human / yeast |
|---|---|---|
| outside our library: 2 missed cleavages | 475 | 26 / 380 / 69 |
| in the library, never extracted | 683 | 20 / 556 / 107 |
| extracted, identified in no run | 5,837 | 284 / 4,470 / 1,083 |
| identified in 1 run | 4,570 | 319 / 3,305 / 946 |
| identified in 2 runs | 5,460 | 588 / 3,555 / 1,317 |
| identified in >= 3 runs, quantified in < 3 | 4 | |

DIA-NN observes the median DIA-NN-only ion in 4 of 6 runs. Our 10,754 MuMDIA-only ions are the
mirror image: DIA-NN reports 4,918 in no run, 2,618 in one and 3,218 in two.

- The search space costs 0.5% of DIA-NN's ions (2 missed cleavages). The other differences
  (Met excision on our side, no m/z cap) cost nothing, since the windows end at 1000.
- 59% of the gap is completeness (identified in 1 or 2 runs), 34% is never identified, 4% never
  extracted. Quant loses no identified ion.
- Per run, DIA-NN quantifies 93,217 to 96,773 ions, we quantify 87,998 to 89,540 (7% fewer), and
  about 75,000 are shared. The gap is per-run identification depth.

**Run level.** 580,023 DIA-NN rows (`Q.Value` <= 0.01). We accept 79.4%; 11,802 are not extracted.
The 107,504 extracted but not accepted rows, by our `q_value` and by our apex against DIA-NN's
(`right` = RT within 5 s and 1/K0 within 0.015):

| our q | right peak | RT right, 1/K0 off | wrong RT peak | all |
|---|---|---|---|---|
| 0.01-0.02 | 9,195 | 133 | 398 | 9,726 |
| 0.02-0.05 | 10,762 | 212 | 722 | 11,696 |
| 0.05-0.1 | 6,967 | 213 | 736 | 7,916 |
| 0.1-0.5 | 12,719 | 819 | 5,323 | 18,861 |
| > 0.5 | 6,008 | 3,647 | 49,650 | 59,305 |
| all | 45,651 | 5,024 | 56,829 | 107,504 |

On the rows we accept, the apex agrees with DIA-NN's in 98.8%. The missed rows are 0.4 log10
fainter (DIA-NN `Precursor.Normalised` median 10^4.33 against 10^4.73).

- Right peak, low score: 45,651 rows, 20,000 of them at q 0.01-0.05. This is the
  `TIMS_ROADMAP_bis.md` section 7 population.
- Wrong RT peak: 56,829 rows, almost all at q > 0.1. Run 0 (9,246 rows): DIA-NN's RT is inside
  our RT window in 91%; DIA-NN's peak is among extract's top-5 peaks in 54% (rank 0: 1,704, i.e.
  the repick moved away from extract's correct apex; ranks 1-4: 3,318); it is absent from the
  top 5 in 46%. Median offset 23 s, median window half-width 37 s.
- Both are identification levers (`TIMS_ROADMAP_bis.md` gates: seeds, entrapment, second
  dataset). No quant lever can recover these ions.

**The q unit is not the cause.** DIA-NN's report is filtered on run-level `Q.Value` <= 0.01 and
run-level `PG.Q.Value` <= 0.01, with no global filter (1.9% of its rows have `Global.Q.Value` >
0.01). Our closest unit is `run_psm_q`. Per run it accepts the same targets as the pooled `q_value`
that quant gates on to within 0.8% (543,248 against 543,082 over six runs), with the same decoy
counts. The completeness class is per-run depth, not a q-unit effect.

**The human epsilon gap: noise against bias.** `quant_diag/bias_split.py` on
the 43,223 complete human ions shared by DIA-NN 2.5.0 and both engine arms (seed 0). Per ion, e =
mean log2 A - mean log2 B (centred), noise sd = sqrt(sd_A^2 / 3 + sd_B^2 / 3), bias sd = sqrt(var(e)
- mean noise var). Intensity quartiles on DIA-NN's level.

| arm | abs e | noise sd | bias sd | bias sd, quartiles faint to bright |
|---|---|---|---|---|
| DIA-NN 2.5.0 | 0.086 | 0.098 | 0.097 | 0.089 / 0.100 / 0.088 / 0.104 |
| per-run background (`q_rp_crw_s0`, section 4c) | 0.093 | 0.099 | 0.106 | 0.114 / 0.113 / 0.088 / 0.102 |
| pooled background (`q_rp_crwbg_s0`, section 4g) | 0.094 | 0.099 | 0.119 | 0.123 / 0.124 / 0.099 / 0.124 |

- Within-condition noise is equal to DIA-NN's in the three fainter quartiles. In the brightest it
  is higher (0.070 against 0.061), and there abs e is too (0.072 against 0.061).
- Across all ions the error is larger than the noise predicts: median |z| 0.95 against DIA-NN's
  0.90, at equal noise sd.
- The bias sd is a variance estimate and is sensitive to outliers. The prototype pooled arm below
  matches the engine pooled arm in ions, global, eq and CV, but gives bias sd 0.111 against 0.119.
  The difference between the two engine arms (0.106 against 0.119) is therefore not evidence that
  the pooled background causes the bias. Median |z| moves only from 0.947 to 0.954.

**Background shrinkage: no gain** (`quant_diag/shrink_eval.sh`, `q2_combine.py POOL_SHRINK=w`;
background = (1 - w) x the run's own flank mean + w x the six-run pool, the same rule for every
run; cross-run weights, fragments only, seed 0). The w = 1 control reproduces the pooled arm.

| w | global | eq | CV | E. coli | shared | only | human median abs z | human bias sd |
|---|---|---|---|---|---|---|---|---|
| 0 (own flank mean) | 0.134 | 0.184 | 0.090 | -1.81 | 0.128 / 0.090 / -1.82 | 0.221 / 0.129 / -1.58 | 0.943 | 0.113 |
| 0.25 | 0.134 | 0.183 | 0.089 | -1.82 | 0.127 / 0.089 / -1.83 | 0.219 / 0.128 / -1.59 | 0.947 | 0.109 |
| 0.5 | 0.133 | 0.182 | 0.088 | -1.83 | 0.127 / 0.089 / -1.83 | 0.216 / 0.127 / -1.60 | 0.949 | 0.109 |
| 0.75 | 0.133 | 0.181 | 0.088 | -1.83 | 0.127 / 0.088 / -1.84 | 0.215 / 0.126 / -1.61 | 0.954 | 0.111 |
| 1 (pool, preset) | 0.134 | 0.180 | 0.087 | -1.84 | 0.127 / 0.088 / -1.84 | 0.215 / 0.125 / -1.63 | 0.954 | 0.111 |
| DIA-NN 2.5.0 | 0.125 | 0.180 | | -1.79 | 0.115 / 0.088 / -1.80 | | 0.897 | 0.097 |

- Global epsilon does not move (0.133 to 0.134), and neither does the human excess over noise.
  w = 1 is the best eq and CV. The preset stays.
- The human gap does not come from how the background level is estimated. The remaining clue is
  the brightest quartile, where our noise is higher than DIA-NN's and a background plays no part.

**The window is too narrow for bright peaks.** The same split on the Q1 window arms (top-3, no
baseline; section 4) shows that in the brightest human quartile the noise falls with the width:
0.096 / 0.071 / 0.061 at `fixed_window_s` 3 / 4 / 5 (DIA-NN 0.061), and abs e falls from 0.090 to
0.065 (DIA-NN 0.061). The faint quartile does not change (abs e 0.136 to 0.137). The preset's 9
scans equal `fixed_window_s: 4`.

Engine requant of the full diaPASEF preset with only `fixed_scan_halfwidth` changed
(`requant_crw.sh`, new `FSH` variable, default 4):

| seed | halfwidth | ions | global | eq | CV | E. coli / human / yeast abs eps | shared | only |
|---|---|---|---|---|---|---|---|---|
| 0 | 4 (preset) | 92,540 | 0.134 | **0.180** | 0.087 | 0.231 / 0.123 / 0.185 | 0.127 / 0.088 / -1.84 | 0.216 / 0.126 / -1.64 |
| 0 | **5** | 92,540 | 0.131 | 0.181 | 0.083 | 0.237 / 0.120 / 0.185 | 0.125 / 0.084 / -1.83 | 0.209 / 0.121 / -1.63 |
| 0 | 6 | 92,540 | **0.130** | 0.186 | **0.082** | 0.246 / 0.120 / 0.191 | 0.124 / 0.082 / -1.82 | 0.211 / 0.120 / -1.62 |
| 1 | 4 (preset) | 92,141 | 0.133 | 0.180 | 0.087 | 0.232 / 0.123 / 0.186 | 0.127 / 0.088 / -1.84 | 0.213 / 0.126 / -1.57 |
| 1 | 5 | 92,141 | 0.130 | 0.181 | 0.083 | 0.238 / 0.120 / 0.186 | 0.124 / 0.084 / -1.83 | 0.207 / 0.121 / -1.59 |
| 2 | 4 (preset) | 92,475 | 0.134 | 0.181 | 0.087 | 0.233 / 0.123 / 0.186 | 0.127 / 0.088 / -1.84 | 0.217 / 0.127 / -1.61 |
| 2 | 5 | 92,475 | 0.131 | 0.183 | 0.083 | 0.241 / 0.120 / 0.186 | 0.125 / 0.084 / -1.83 | 0.210 / 0.121 / -1.60 |
| | DIA-NN 2.5.0 | 98,694 | 0.125 | 0.180 | 0.089 | 0.237 / 0.113 / 0.189 | 0.115 / 0.088 / -1.80 | |

(CV here is ProteoBench's `CV_median`; the shared and only columns use the `shared.py` CV.)

- Halfwidth 5 against 4, the same on all three ID sets: global -0.003, CV -0.004, human abs eps
  -0.003, shared-ion eps -0.002 to -0.003, eq +0.001 to +0.002 (E. coli abs eps +0.006 to +0.008).
  It closes about a third of the global gap to DIA-NN 2.5.0, and our CV moves below DIA-NN's.
- Halfwidth 6 continues the trade: global -0.001 and CV -0.001 more, eq +0.005 (E. coli). The wider
  window helps bright ions and adds floor to faint E. coli ions in condition A.
- Under section 1 (accuracy first; eq is not to get worse), 5 is a near-even trade on eq and a
  gain on global and CV. Promoting it to `QuantConfig::diapasef` is a one-value change. It is not
  done yet: the eq cost needs a decision.
- The width that is right depends on the peak. The next lever is a window that follows the peak
  width: for example a width per precursor, learned across runs from its brightest runs and used
  in every run (label-blind, part of the cross-run step). Prototype it offline first.

**A halfwidth per precursor from its peak width (prototype, 2026-10-01).** `quant_diag/wwin.py`
(`wwin.sh <seed> <rule ...>`). `extract` keeps each fragment's raw samples at +-19 scans around
quant's apex; `combine` recomputes the diaPASEF preset (pooled flank-mean background, envelope,
cross-run weights, fragments only) at any halfwidth h per precursor, the same h in every run.
Rules, all label-blind:
- `fixed:h`: h for every precursor (controls).
- `width:c[:floor]`: h = clamp(round(c x HWHM), floor (default 3), 7). HWHM is the half width at
  half maximum, in scans, of the summed background-subtracted fragment trace in the precursor's
  brightest run (largest quantity at h 4). HWHM quartiles: 1.27 / 1.83 / 2.36 scans.
- `bright:q`: h = 5 above the q quantile of the brightest-run quantity, else 4.

Controls: `fixed:4` gives 0.134 / 0.180 / 0.087 (engine preset 0.134 / 0.180 / 0.087), `fixed:5`
0.130 / 0.181 / 0.083 (engine 0.131 / 0.181 / 0.083). Seed 0, 92,540 ions in every arm:

| rule | h = 3 / 4 / 5 / 6 / 7 (precursors) | global | eq | CV | E. coli / human / yeast abs eps | shared | only |
|---|---|---|---|---|---|---|---|
| `fixed:4` (preset) | all 4 | 0.134 | 0.180 | 0.087 | 0.232 / 0.124 / 0.185 | 0.127 / 0.088 / -1.84 | 0.215 / 0.125 / -1.63 |
| `fixed:5` | all 5 | 0.130 | 0.181 | 0.083 | 0.236 / 0.120 / 0.187 | 0.124 / 0.084 / -1.83 | 0.211 / 0.121 / -1.61 |
| `width:1` | 115k / 5.6k / 3.1k / 1.0k / 0.5k | 0.142 | 0.186 | 0.097 | 0.235 / 0.133 / 0.189 | 0.136 / 0.098 / -1.86 | 0.223 / 0.132 / -1.65 |
| `width:1.5` | 93k / 17k / 6.6k / 3.4k / 5.6k | 0.139 | 0.182 | 0.094 | 0.231 / 0.130 / 0.186 | 0.133 / 0.095 / -1.86 | 0.219 / 0.128 / -1.65 |
| `width:2` | 56k / 33k / 16k / 7.5k / 13k | 0.132 | **0.178** | 0.086 | **0.228** / 0.123 / **0.182** | 0.127 / 0.087 / -1.85 | 0.211 / 0.122 / -1.62 |
| **`width:2.5`** | 36k / 24k / 26k / 15k / 24k | 0.130 | **0.178** | 0.083 | 0.231 / 0.120 / 0.184 | 0.123 / 0.083 / -1.84 | 0.210 / 0.120 / -1.62 |
| `width:3` | 27k / 14k / 22k / 22k / 40k | **0.129** | 0.181 | **0.082** | 0.237 / **0.119** / 0.187 | 0.123 / 0.082 / -1.83 | 0.206 / 0.119 / -1.60 |
| `width:3.5` | 20k / 12k / 12k / 20k / 60k | 0.130 | 0.184 | **0.082** | 0.244 / 0.119 / 0.189 | 0.124 / 0.082 / -1.82 | 0.207 / 0.119 / -1.55 |
| `width:2.5:4` | 0 / 60k / 26k / 15k / 24k | 0.130 | 0.180 | **0.082** | 0.235 / 0.119 / 0.185 | 0.123 / 0.083 / -1.84 | 0.207 / 0.120 / -1.61 |
| `width:3:4` | 0 / 40k / 22k / 22k / 40k | 0.130 | 0.182 | **0.082** | 0.238 / 0.119 / 0.188 | 0.123 / 0.082 / -1.83 | 0.206 / 0.120 / -1.59 |
| `bright:0.5` | 0 / 62k / 62k / 0 / 0 | 0.131 | 0.181 | 0.084 | 0.236 / 0.121 / 0.186 | 0.125 / 0.084 / -1.83 | 0.210 / 0.122 / -1.62 |
| `bright:0.75` | 0 / 94k / 31k / 0 / 0 | 0.131 | 0.179 | 0.084 | 0.232 / 0.121 / 0.184 | 0.125 / 0.085 / -1.84 | 0.212 / 0.124 / -1.62 |

Precursor counts include those quantified in fewer than 3 runs.

Replication of `width:2.5` against the preset on the same IDs:

| seed | arm | ions | global | eq | CV | E. coli / human / yeast abs eps | shared | only |
|---|---|---|---|---|---|---|---|---|
| 0 | preset | 92,540 | 0.134 | 0.180 | 0.087 | 0.231 / 0.123 / 0.185 | 0.127 / 0.088 / -1.84 | 0.216 / 0.126 / -1.64 |
| 0 | `width:2.5` | 92,540 | 0.130 | 0.178 | 0.083 | 0.231 / 0.120 / 0.184 | 0.123 / 0.083 / -1.84 | 0.210 / 0.120 / -1.62 |
| 1 | preset | 92,141 | 0.133 | 0.180 | 0.087 | 0.232 / 0.123 / 0.186 | 0.127 / 0.088 / -1.84 | 0.213 / 0.126 / -1.57 |
| 1 | `width:2.5` | 92,141 | 0.129 | 0.178 | 0.083 | 0.232 / 0.119 / 0.182 | 0.123 / 0.083 / -1.84 | 0.206 / 0.119 / -1.57 |
| 2 | preset | 92,475 | 0.134 | 0.181 | 0.087 | 0.233 / 0.123 / 0.186 | 0.127 / 0.088 / -1.84 | 0.217 / 0.127 / -1.61 |
| 2 | `width:2.5` | 92,475 | 0.130 | 0.180 | 0.083 | 0.236 / 0.119 / 0.184 | 0.123 / 0.083 / -1.84 | 0.210 / 0.121 / -1.58 |
| | DIA-NN 2.5.0 | 98,694 | 0.125 | 0.180 | 0.089 | 0.237 / 0.113 / 0.189 | 0.115 / 0.088 / -1.80 | |

- On all three ID sets: global -0.004, eq -0.001 to -0.002, CV -0.004, shared-ion eps -0.004, same
  ion count. Unlike `fixed:5`, it costs no eq: narrow peaks keep a narrow window, so faint E. coli
  ions in condition A collect no more floor.
- Global gap to DIA-NN 2.5.0: 0.009 to 0.005 (about 45% closed). eq is at DIA-NN's value, CV 0.006
  below it.
- Human split (seed 0): in the brightest quartile the noise is now 0.056 (DIA-NN 0.061, preset
  0.069) and abs e 0.062 (DIA-NN 0.061). The faint quartile does not change (abs e 0.130 against
  DIA-NN 0.118). The remaining global gap is on faint ions.
- Submission (seed 0): `bench/proteobench_input/ww_width2.5/`.
- The background must come from the flank beyond the chosen window. With the flank fixed beyond
  +-7 scans for every precursor (`FLANK_AT=7`), `width:2.5` gives 0.130 / 0.180 / 0.083: the eq
  gain is lost, because a narrow peak then takes its background from too far away.

**Engine: `quant.cross_run_width` (2026-10-01).** In the two-pass cross-run step. Pass 1 exports
per fragment the flank mean beyond each halfwidth 3 to 7 (`flank_mean_h3`..`h7`) and per candidate
`peak_hwhm` (fragment flank mean subtracted, per run); `fit_cross_run` takes the HWHM of the
candidate's brightest run (largest summed pass-1 area), h = clamp(round(2.5 x HWHM), 3, 7), and
pools `flank_mean_h<h>`; pass 2 integrates 2h+1 scans. The weights stay fitted on the pass-1 areas.
`fragment_quant` v4. Part of `QuantConfig::diapasef()` (2.5). Binary `~/bin/mumdia-ww/mumdia`,
harness `requant_crw.sh` with `WIDTH=2.5`.

| seed | ions | global | eq | CV | E. coli / human / yeast abs eps | shared | only |
|---|---|---|---|---|---|---|---|
| 0 | 92,540 | 0.130 | 0.177 | 0.083 | 0.228 / 0.120 / 0.182 | 0.123 / 0.083 / -1.84 | 0.211 / 0.121 / -1.59 |
| 1 | 92,141 | 0.129 | 0.177 | 0.083 | 0.230 / 0.119 / 0.182 | 0.123 / 0.083 / -1.84 | 0.210 / 0.121 / -1.59 |
| 2 | 92,475 | 0.130 | 0.178 | 0.083 | 0.231 / 0.120 / 0.183 | 0.123 / 0.083 / -1.84 | 0.212 / 0.122 / -1.59 |
| DIA-NN 2.5.0 | 98,694 | 0.125 | 0.180 | 0.089 | 0.237 / 0.113 / 0.189 | 0.115 / 0.088 / -1.80 | |

Against the preset of section 4g on the same IDs: global -0.004, eq -0.003, CV -0.004 on every ID
set; the engine matches the prototype (seed 0 prototype 0.130 / 0.178 / 0.083). MBR off, eq and CV
are now better than DIA-NN 2.5.0; global epsilon is +0.005 (human faint ions) and ions -6.2%.

## 4i. MBR on: transferred values are the right peak with a weak signal (2026-10-01, seed 0)

`quant_diag/mbrdiag.py` on `mbr_s0` (39,071 transfers, RT window 1.7 s) and its quant
`q_rp_mbr_crwbg_s0`. Per row, `d` = log2(value) - median log2 of the confident values of the
other condition - the expected log2 ratio (0 for an accurate value; species labels are used for
this diagnosis only). Controls (C) are the confident rows, scored leave-one-out. Cosine: pass-1
fragment areas against the L1-normalised mean of the other confident runs.

| species, cond | group | rows | d median | abs d | 1/K0 off > 0.015 | cosine median | cos < 0.8 | in 2.2.0 | RT within 5 s of 2.2.0 | in 2.5.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| E. coli A (low) | C | 8,666 | +0.15 | 0.24 | 0.3% | 0.971 | 4.7% | 93% | 99.6% | 81% |
| | T | 3,997 | +0.52 | 0.63 | 5.9% | 0.874 | 26.9% | 56% | 99.2% | 20% |
| E. coli B (high) | C | 21,470 | -0.17 | 0.25 | 0.3% | 0.973 | 9.6% | 88% | 99.6% | 84% |
| | T | 556 | -1.87 | 1.87 | 2.2% | 0.945 | 7.6% | 65% | 98.6% | 48% |
| human A | T | 13,439 | -0.09 | 0.30 | 3.6% | 0.939 | 10.5% | 64% | 98.9% | 41% |
| human B | T | 11,996 | -0.09 | 0.30 | 3.8% | 0.940 | 10.3% | 63% | 98.6% | 43% |
| yeast A (high) | T | 1,685 | -0.60 | 0.65 | 3.6% | 0.940 | 11.8% | 63% | 98.5% | 49% |
| yeast B (low) | C | 32,319 | +0.11 | 0.20 | 0.2% | 0.978 | 3.5% | 92% | 99.3% | 83% |
| | T | 7,147 | +0.24 | 0.39 | 3.5% | 0.914 | 16.3% | 66% | 99.5% | 32% |

Human controls: d +0.02 / -0.02, abs d 0.15, cosine 0.98.

- **Right peak.** Where DIA-NN 2.2.0 reports the ion in the same run, our transfer apex is within
  5 s of its RT in 98.5-99.5% of rows, as for confident rows. 1/K0 is off by more than 0.015 in
  2-6% of transfers (0.3% of controls).
- **Weak signal.** Transfers are 2.4 log2 fainter than confident rows. In the low condition they
  are overestimated (E. coli A +0.52, yeast B +0.24); in the high condition they are
  underestimated (selection: a run where the ion failed to identify is a run where its signal
  came out low). Matched on the expected value, transfers are still worse than confident rows
  (low condition, expected log2 13.5-14.5: abs d 0.40 against 0.29).
- **DIA-NN has the same error on the same rows.** On the (ion, run) rows both report, DIA-NN's own
  d: E. coli A transfers +0.28 (ours +0.38), abs 0.38 (ours 0.52); human transfers abs 0.24-0.25
  (ours 0.29). DIA-NN's confident rows: E. coli A +0.16, as ours (+0.15).
- **Guards do not separate the error.** On the low-condition transfers, d by cosine bin is +0.39 /
  +0.26 / +0.27 / +0.30 / +0.40 (cosine <= 0.6 to > 0.95); by 1/K0 offset +0.28 (<= 0.005, 81% of
  rows), +0.38, +0.56, +0.91 (> 0.015, 485 rows). A 1/K0 guard at 0.015 would act on 4% of
  transfers; a cosine guard removes rows that are no worse than the rest. Both are dropped as
  quant levers. A 1/K0 check is still sound FDR hygiene, but it is about a 1% effect.
- The quant gap on transfers is the faint-ion floor of sections 4d to 4g, made larger by the
  selection. It is not a wrong-signal problem, so a separate quant path for transfers has no
  measured basis yet.

**MBR-on ion gap.** Of the 27,045 DIA-NN 2.2.0-only ions: outside our search space 2,173
(N-term acetyl 1,069, charge 1 or 5 576, length 6 528; the DDA FASTA adds none), never extracted
1,117, never identified 13,038, identified in 1 run 8,015, in 2 runs 2,699. Per run DIA-NN
quantifies 110,702 to 112,703 ions, we quantify 94,221 to 95,710.

`quant_diag/mbrgap.py`. The 109,717 (ion, run) rows DIA-NN reports and our MBR submission does not, by anchors (runs where
we are confident without transfer) and by our extracted apex against DIA-NN's RT:

| case | rows |
|---|---|
| no anchor run | 68,158 |
| 1 anchor, right peak | 15,501 |
| 1 anchor, wrong peak | 15,406 |
| >= 2 anchors, wrong peak | 6,577 |
| >= 1 anchor, not extracted | 2,633 |
| >= 2 anchors, right peak | 1,442 |

Upper bounds on DIA-NN-only ions that reach 3 runs (24,872 in our search space), before any FDR
test: right peak with >= 2 anchors 883; + right peak with 1 anchor 5,329; + integration at the
predicted RT where our peak is wrong or not extracted 10,706. 62% of the rows belong to ions we
never identify in any run; MBR cannot reach them, and the remaining MBR-on gap is mostly the
per-run depth of section 4h.

**`mbr.min_anchor_runs: 1`** (worker flag, no engine change; `mbr_a1_s0`, quant
`q_rp_mbr_a1_crwbg_s0`). Control: the worker at the default reproduces `mbr_s0` exactly.

| arm | transfers | RT window | ions | global | eq | CV | PB CV | E. coli / human / yeast abs eps | shared against 2.2.0 | MuMDIA-only |
|---|---|---|---|---|---|---|---|---|---|---|
| `min_anchor_runs` 2 | 39,071 | 1.7 s | 100,340 | 0.143 | 0.220 | 0.098 | 0.094 | 0.325 / 0.128 / 0.207 | 91,379: 0.138 / 0.095 / -1.79 | 8,961: 0.237 / 0.141 / -1.09 |
| `min_anchor_runs` 1 | 48,808 | 1.0 s | 105,092 | 0.148 | 0.229 | 0.096 | 0.093 | 0.333 / 0.132 / 0.221 | 92,913: 0.139 / 0.093 / -1.79 | 12,179: 0.270 / 0.137 / -0.63 |

+4,752 ions, but only 1,534 of them are DIA-NN ions. The other 3,218 move our-only E. coli from
-1.09 to -0.63, which fits false or mostly-noise transfers. One pooled transfer test also
tightens the window for the 2-anchor tier (1.7 s to 1.0 s). Rejected in this form. If revisited:
a separate transfer q per anchor tier.

**Next MBR lever, by size:** integration at the predicted RT for a precursor that is confident
elsewhere but whose extracted apex is elsewhere or missing (the re-extraction tier; up to about
5,400 ions beyond the 1-anchor tier). It needs Rust plumbing and an FDR null for the integrated
value, so it comes after a prototype on the chromatograms.

## 4j. MBR at the predicted RT, the 1-anchor tier, and transfer quant (2026-10-02, seeds 0-2)

Prototype outside the engine, on the current preset (`cross_run_width` 2.5, binary `~/bin/mumdia-ww/mumdia`).
Scripts in `quant_diag/`: `mbr_prt.py` (evidence and null), `mbr_prt_eval.py` (transfer q, arms), `prt_quant.py`
(per-row quant against DIA-NN 2.2.0), `robust_tr.py` (transfer quant), `seed_mbr.sh <seed>` (the whole chain on
`eng_repick_s<seed>`; the worker reads the shared `eng_repick` competed tables). Arms `mbr_prt2_s*`,
`mbr_prt12_s*`, quants `q_rp_mbr_*_s*` and `q_rp_rob1_*_s*`, all scored with `MBR=true score.sh`.

**Population.** (candidate, run), target, confident (q <= 0.01, not transferred) in at least one other run,
neither confident nor transferred in this run (so the rescuable tier rejected it, its apex being outside the
1.5-1.7 s window), extracted here. Expected RT as in `mbr_worker.py` (binned-median maps through run 0, median over the anchor
runs). Seed 0: 66,783 rows with >= 2 anchors (tier 2), 89,902 with 1 anchor (tier 1); for 1.4% the expected RT
is outside the candidate's trace, so those rows are not tested. Rows that were never extracted (2,633 in section
4i) need retrace at the expected RT and are not covered.

**Evidence and null.** At the expected RT, in this run's raw retrace traces: the cosine of square-rooted
fragment areas (+-4 scans) against the anchor runs' L1-normalised pass-1 areas, the median co-elution of each
fragment with the sum of the others (+-6 scans), and the log summed area. The null takes the same three values
in the same traces at K = 10 random positions at least 15 s from the expected RT: same candidate, same run, same
fragments, and no ion at that position. The score is a logistic regression of target against null on the three
values. It is cross-fitted in 2 folds by candidate, so no row is scored by a model that saw it. Transfer
q = (null >= s + 1) / K / (targets >= s), running minimum, as in the worker. **Each tier has its own pool and its
own q**, so tier 1 cannot move tier 2's threshold.

Separation (seed 0, AUC target against null): cosine 0.85-0.87, co-elution 0.65, area 0.61-0.63. Cosine alone
accepts 7,894 tier-2 rows at 1%, the combined score 12,883; co-elution alone accepts none.

| seed | tier | rows tested | accepted at q 0.01 | null draws (at or above the threshold) | expected false | DIA-NN 2.2.0 reports the row | expected RT within 5 s of DIA-NN's |
|---|---|---|---|---|---|---|---|
| 0 | 2 | 65,822 | 12,883 | 658,220 (1,287) | 129 | 81.4% | 99.0% |
| 0 | 1 | 88,506 | 15,006 | 885,060 (1,499) | 150 | 42.0% | 97.8% |
| 1 | 2 | 66,559 | 13,091 | 665,590 (1,307) | 131 | 81.3% | 98.9% |
| 1 | 1 | 87,687 | 15,128 | 876,870 (1,511) | 151 | 41.8% | 98.1% |
| 2 | 2 | 69,158 | 14,690 | 691,580 (1,468) | 147 | 80.0% | 98.9% |
| 2 | 1 | 86,879 | 15,404 | 868,790 (1,539) | 154 | 42.0% | 98.0% |

For comparison, DIA-NN reports 68.3% (tier 2) and 35.6% (tier 1) of all tested rows. The rescuable tier on seeds
1 and 2 (worker, unchanged): 37,434 and 35,106 transfers, 373 and 350 null draws inside a 1.6 s and 1.5 s window.

**Arms.** An accepted row gets `apex_rt` = the expected RT, its PSM q columns lowered to the transfer q and
`is_transferred` set; the engine quant then integrates there (two passes, preset). "Rescuable" is the worker tier
alone (the base of section 2b). "drop1" is the transfer quant below. CV is the median of (CV_A + CV_B) / 2,
PB CV ProteoBench's.

| seed | arm | ions | global | eq | CV | PB CV | E. coli / human / yeast abs eps | E. coli log2 |
|---|---|---|---|---|---|---|---|---|
| 0 | rescuable | 100,340 | 0.140 | 0.218 | 0.093 | 0.089 | 0.326 / 0.124 / 0.204 | -1.76 |
| 0 | rescuable, drop1 | 100,340 | 0.140 | **0.212** | 0.094 | 0.090 | 0.308 / 0.125 / 0.202 | -1.80 |
| 0 | + predicted RT, tier 2 | 101,268 | 0.141 | 0.221 | 0.096 | 0.091 | 0.333 / 0.125 / 0.206 | -1.76 |
| 0 | + predicted RT, tier 2, drop1 | 101,268 | 0.142 | 0.214 | 0.098 | 0.093 | 0.314 / 0.126 / 0.204 | -1.80 |
| 0 | + predicted RT, tiers 1 and 2 | 105,040 | 0.144 | 0.230 | 0.097 | 0.092 | 0.347 / 0.127 / 0.215 | -1.74 |
| 1 | rescuable | 99,775 | 0.139 | 0.217 | 0.093 | 0.089 | 0.323 / 0.124 / 0.203 | -1.77 |
| 1 | rescuable, drop1 | 99,775 | 0.140 | **0.210** | 0.094 | 0.090 | 0.304 / 0.125 / 0.201 | -1.80 |
| 1 | + predicted RT, tier 2 | 100,718 | 0.140 | 0.219 | 0.095 | 0.091 | 0.328 / 0.125 / 0.205 | -1.76 |
| 1 | + predicted RT, tier 2, drop1 | 100,718 | 0.141 | 0.213 | 0.097 | 0.092 | 0.310 / 0.126 / 0.203 | -1.80 |
| 1 | + predicted RT, tiers 1 and 2 | 104,539 | 0.143 | 0.228 | 0.096 | 0.091 | 0.344 / 0.127 / 0.214 | -1.74 |
| 2 | rescuable | 99,846 | 0.139 | 0.216 | 0.093 | 0.088 | 0.320 / 0.124 / 0.204 | -1.77 |
| 2 | rescuable, drop1 | 99,846 | 0.140 | **0.210** | 0.093 | 0.089 | 0.304 / 0.125 / 0.202 | -1.80 |
| 2 | + predicted RT, tier 2 | 100,929 | 0.140 | 0.220 | 0.095 | 0.091 | 0.328 / 0.125 / 0.206 | -1.76 |
| 2 | + predicted RT, tier 2, drop1 | 100,929 | 0.141 | 0.213 | 0.097 | 0.092 | 0.309 / 0.126 / 0.205 | -1.80 |
| 2 | + predicted RT, tiers 1 and 2 | 104,831 | 0.143 | 0.229 | 0.096 | 0.092 | 0.343 / 0.127 / 0.217 | -1.74 |
| | DIA-NN 2.2.0 | 118,326 | 0.143 | 0.207 | 0.114 | 0.107 | 0.282 / 0.125 / 0.215 | -1.77 |

Shared with DIA-NN 2.2.0 (`DIANN=220 shared.py`), eps / CV / E. coli log2:

| seed | arm | shared | MuMDIA | DIA-NN on the same ions | MuMDIA-only |
|---|---|---|---|---|---|
| 0 | rescuable | 91,379 | 0.134 / 0.091 / -1.78 | 0.127 / 0.104 / -1.80 | 8,961: 0.231 / 0.136 / -1.03 |
| 0 | rescuable, drop1 | 91,379 | 0.135 / 0.092 / -1.82 | 0.127 / 0.104 / -1.80 | 8,961: 0.236 / 0.141 / -1.19 |
| 0 | + tier 2 | 92,118 | 0.135 / 0.093 / -1.78 | 0.128 / 0.104 / -1.80 | 9,150: 0.233 / 0.139 / -1.00 |
| 0 | + tier 2, drop1 | 92,118 | 0.136 / 0.095 / -1.82 | 0.128 / 0.104 / -1.80 | 9,150: 0.237 / 0.144 / -1.14 |
| 0 | + tiers 1 and 2 | 93,894 | 0.137 / 0.094 / -1.78 | 0.129 / 0.105 / -1.80 | 11,146: 0.243 / 0.135 / -0.70 |
| 1 | rescuable | 90,896 | 0.134 / 0.090 / -1.79 | 0.127 / 0.104 / -1.80 | 8,879: 0.231 / 0.136 / -1.00 |
| 1 | rescuable, drop1 | 90,896 | 0.134 / 0.091 / -1.82 | 0.127 / 0.104 / -1.80 | 8,879: 0.237 / 0.141 / -1.10 |
| 1 | + tier 2 | 91,641 | 0.135 / 0.093 / -1.78 | 0.127 / 0.104 / -1.80 | 9,077: 0.234 / 0.139 / -0.95 |
| 1 | + tier 2, drop1 | 91,641 | 0.135 / 0.094 / -1.82 | 0.127 / 0.104 / -1.80 | 9,077: 0.240 / 0.144 / -1.06 |
| 1 | + tiers 1 and 2 | 93,402 | 0.136 / 0.093 / -1.78 | 0.129 / 0.105 / -1.80 | 11,137: 0.245 / 0.133 / -0.60 |
| 2 | rescuable | 91,024 | 0.134 / 0.090 / -1.79 | 0.127 / 0.104 / -1.80 | 8,822: 0.234 / 0.135 / -1.02 |
| 2 | rescuable, drop1 | 91,024 | 0.134 / 0.091 / -1.82 | 0.127 / 0.104 / -1.80 | 8,822: 0.239 / 0.140 / -1.14 |
| 2 | + tier 2 | 91,866 | 0.135 / 0.093 / -1.78 | 0.127 / 0.104 / -1.80 | 9,063: 0.237 / 0.139 / -0.97 |
| 2 | + tier 2, drop1 | 91,866 | 0.135 / 0.094 / -1.82 | 0.127 / 0.104 / -1.80 | 9,063: 0.240 / 0.145 / -1.09 |
| 2 | + tiers 1 and 2 | 93,687 | 0.136 / 0.093 / -1.78 | 0.129 / 0.105 / -1.80 | 11,144: 0.247 / 0.135 / -0.63 |

**Tier 2 (step 2).** 12,883-14,690 rows per seed pass, and 80-81% of them are rows DIA-NN reports, on DIA-NN's peak
in 99%. Most of them fill runs of ions that are already at `min_obs` 3: +928 / +943 / +1,083 ions, of which
+739 / +745 / +842 are DIA-NN's. Quant cost: global +0.001, eq +0.002 to +0.004, PB CV +0.002 to +0.003.

**Tier 1 (step 3).** With its own q the 1-anchor tier no longer tightens the 2-anchor window. It still adds
ions whose E. coli ratio is near 0 (our-only -1.00 to -0.60 across the three seeds). The cause is not the anchor: over the
3,749 new 1-anchor ions of seed 0, the E. coli log2 is -0.04 to -0.16 in every anchor-q bin from <= 1e-4 to
0.01, and the new ions DIA-NN also reports quantify badly too. On the new ions DIA-NN shares, its own E. coli log2 is
-1.38 (tier 1) and -1.49 (tier 2) against ours -0.32 and -0.45 (128 and 92 ions; human and yeast closer:
yeast 0.35 / 0.66 against DIA-NN 0.68 / 0.76). The identifications are DIA-NN's; their quantities are wrong.
Tier 1 is rejected in this form on quant (eq +0.012), not on its FDR.

**What is wrong with the transferred values.** `prt_quant.py`: d per row (section 4i), split by the signal over
the pooled floor (summed pass-1 area / 9 x the summed flank mean). In the low condition the error rises with the
signal, which a floor cannot cause. On the rows DIA-NN also reports, transfers (both tiers):

| SNR | E. coli A rows | ours d | DIA-NN d | yeast B rows | ours d | DIA-NN d |
|---|---|---|---|---|---|---|
| <= 0.5 | 728 | +0.19 | +0.32 | 1,311 | +0.08 | +0.20 |
| 0.5-1 | 921 | +0.42 | +0.29 | 2,070 | +0.21 | +0.19 |
| 1-2 | 651 | +0.60 | +0.25 | 1,566 | +0.28 | +0.17 |
| 2-5 | 246 | +0.70 | +0.19 | 674 | +0.35 | +0.13 |
| > 5 | 30 | +0.66 | +0.24 | 141 | +0.74 | +0.10 |

On faint transfers we are closer to the truth than DIA-NN. On bright transfers in the low condition, where the ion failed
to identify, our value carries signal that is not the ion's and DIA-NN's does not. One weight vector per precursor
(section 4c) cannot remove interference that is present in one run only.

**Transfer quant: drop1** (`robust_tr.py`, `DROP1=1`, transferred rows only). Per (candidate, run), from the pass-1
fragment areas A_f and the mean over the candidate's confident runs ref_f: c = (sum of A_f / sum of ref_f without the
fragment with the largest A_f / ref_f) / (sum of A_f / sum of ref_f), with at least 3 fragments; quantity x c. Interference on one
fragment in one run then does not raise the value. It reads no condition or species. The median of A_f / ref_f
instead of drop1 gives eq 0.216 and global 0.144 on seed 0 (noisier). Applied to every row, it gives global 0.158 and CV
0.115, so confident rows do not need it.

- On the rescuable tier alone, drop1 is the same on all three seeds: eq -0.006 to -0.007, global +0.000 to
  +0.001, PB CV +0.001, E. coli -1.76 / -1.77 to -1.80, same ions. Shared ions: eps +0.000 to +0.001, CV +0.001.
  Against DIA-NN 2.2.0: global -0.003, eq +0.003 to +0.005, CV 0.020 lower.
- With tier 2 and drop1 against the rescuable tier without it: +928 to +1,083 ions, eq -0.003 to -0.004,
  global +0.002, PB CV +0.003 to +0.004.
- Engine form, if taken: in `fit_cross_run`, a per-(candidate, run) factor for transferred rows from the pass-1
  table (the reference profile from the non-transferred runs), applied in pass 2. Small, but it needs the
  transferred flag in the pass-1 fragment table.

**Open.**
- Tier 2 in the engine is the re-extraction tier: retrace at the expected RT and the anchor runs' 1/K0, then this test.
  The prototype reads traces built in the band of this run's (wrong) apex 1/K0, so it measures a lower bound.
- Tier 1 waits for transfer quant that holds faint, partly interfered values to the right ratio. drop1 alone is not
  enough (E. coli on the new ions is still near 0).
- No entrapment check yet: the null is the FDR instrument here, and DIA-NN agreement (80-81% of accepted tier-2 rows,
  99% on its peak) is the only external check.

## 4k. The re-extraction tier: retrace at the expected RT and the anchors' 1/K0 (2026-10-02, seeds 0-2)

The section 4j test read traces built in the 1/K0 band of this run's own apex, which for these rows is on another
peak. Here the same rows get new traces from the existing stage, with no engine change:
- `reext_prep.py <seed> <dir>`: per run, the tested rows of section 4j (tiers 1 and 2), their re-pick rows with
  `apex_rt` = the expected RT and `apex_im` = the median over the anchor runs of their re-picked `apex_im`, each
  moved onto this run's scale by a per-run median offset (-0.002 to +0.0002 against run 0), plus extract's
  centroid traces of those candidates only. The 1/K0 centre moves by a median 0.0072-0.0077, half the band.
- `reext_retrace.sh <dir>`: `mumdia retrace` with `repick` off on those inputs (7-19 s per run, 28-30 GB,
  about 400,000 trace rows).
- `mbr_prt.py` (`CH=<dir>`) and `mbr_prt_eval.py`: the same evidence, null and cross-fitted transfer q as 4j.
- `reext_quant.sh`: the preset quant in two parts per run, the existing traces for every other row and the new
  traces for the new transfers, with all twelve pass-1 fragment tables in one cross-run fit. Control: with the
  existing traces in both parts it reproduces the one-part quant exactly (101,268 / 0.141 / 0.221 / 0.091).
- `reext_seed.sh <seed>`: the whole chain.

| seed | tier 2 accepted | null draws (at or above) | expected false | DIA-NN 2.2.0 reports the row | its RT within 5 s |
|---|---|---|---|---|---|
| 0 | 35,233 (4j: 12,883) | 658,220 (3,522) | 352 | 79.5% | 99.1% |
| 1 | 36,114 (4j: 13,091) | 665,590 (3,610) | 361 | 79.4% | 99.0% |
| 2 | 37,092 (4j: 14,690) | 691,580 (3,708) | 371 | 79.0% | 99.1% |

| seed | arm | ions | global | eq | CV | PB CV | E. coli / human / yeast abs eps | E. coli log2 |
|---|---|---|---|---|---|---|---|---|
| 0 | rescuable (4j) | 100,340 | 0.140 | 0.218 | 0.093 | 0.089 | 0.326 / 0.124 / 0.204 | -1.76 |
| 0 | + re-extraction, tier 2 | 102,640 | **0.139** | 0.223 | 0.098 | 0.093 | 0.339 / 0.123 / 0.206 | -1.75 |
| 0 | + re-extraction, tier 2, drop1 | 102,640 | 0.140 | **0.214** | 0.100 | 0.095 | 0.315 / 0.125 / 0.204 | -1.80 |
| 0 | + re-extraction, tiers 1 and 2 | 109,483 | 0.144 | 0.236 | 0.099 | 0.094 | 0.361 / 0.126 / 0.220 | -1.72 |
| 0 | + re-extraction, tiers 1 and 2, drop1 | 109,483 | 0.145 | 0.228 | 0.102 | 0.096 | 0.338 / 0.128 / 0.217 | -1.77 |
| 1 | rescuable (4j) | 99,775 | 0.139 | 0.217 | 0.093 | 0.089 | 0.323 / 0.124 / 0.203 | -1.77 |
| 1 | + re-extraction, tier 2 | 102,150 | 0.139 | 0.222 | 0.098 | 0.093 | 0.337 / 0.123 / 0.205 | -1.75 |
| 1 | + re-extraction, tier 2, drop1 | 102,150 | 0.140 | 0.214 | 0.100 | 0.095 | 0.315 / 0.125 / 0.203 | -1.80 |
| 2 | rescuable (4j) | 99,846 | 0.139 | 0.216 | 0.093 | 0.088 | 0.320 / 0.124 / 0.204 | -1.77 |
| 2 | + re-extraction, tier 2 | 102,362 | 0.139 | 0.221 | 0.098 | 0.093 | 0.335 / 0.123 / 0.206 | -1.75 |
| 2 | + re-extraction, tier 2, drop1 | 102,362 | 0.140 | 0.213 | 0.100 | 0.094 | 0.311 / 0.125 / 0.204 | -1.80 |

Shared with DIA-NN 2.2.0, eps / CV / E. coli log2:

| seed | arm | shared | MuMDIA | DIA-NN on the same ions | MuMDIA-only |
|---|---|---|---|---|---|
| 0 | + tier 2 | 93,181 | 0.134 / 0.095 / -1.78 | 0.128 / 0.104 / -1.80 | 9,459: 0.229 / 0.139 / -0.97 |
| 0 | + tier 2, drop1 | 93,181 | 0.134 / 0.097 / -1.82 | 0.128 / 0.104 / -1.80 | 9,459: 0.234 / 0.146 / -1.12 |
| 0 | + tiers 1 and 2 | 96,629 | 0.136 / 0.096 / -1.77 | 0.131 / 0.106 / -1.79 | 12,854: 0.244 / 0.133 / -0.58 |
| 0 | + tiers 1 and 2, drop1 | 96,629 | 0.137 / 0.098 / -1.81 | 0.131 / 0.106 / -1.79 | 12,854: 0.249 / 0.140 / -0.67 |
| 1 | + tier 2 | 92,753 | 0.133 / 0.095 / -1.78 | 0.128 / 0.104 / -1.80 | 9,397: 0.230 / 0.140 / -0.92 |
| 1 | + tier 2, drop1 | 92,753 | 0.134 / 0.097 / -1.82 | 0.128 / 0.104 / -1.80 | 9,397: 0.234 / 0.147 / -1.01 |
| 2 | + tier 2 | 92,963 | 0.133 / 0.095 / -1.78 | 0.128 / 0.104 / -1.80 | 9,399: 0.230 / 0.140 / -0.92 |
| 2 | + tier 2, drop1 | 92,963 | 0.134 / 0.097 / -1.82 | 0.128 / 0.104 / -1.80 | 9,399: 0.235 / 0.148 / -1.02 |

Quant of the re-extracted transfers on the rows DIA-NN also reports (`prt_quant.py rx2 0`), median d / abs d:

| species, cond | rows | ours | DIA-NN on the same rows | rescuable transfers, ours |
|---|---|---|---|---|
| E. coli A (low) | 1,898 | +0.30 / 0.44 | +0.27 / 0.37 | +0.42 / 0.53 |
| human A | 13,768 | -0.07 / 0.22 | -0.07 / 0.21 | -0.07 / 0.28 |
| human B | 12,406 | -0.04 / 0.22 | -0.00 / 0.20 | -0.08 / 0.29 |
| yeast B (low) | 4,255 | +0.18 / 0.30 | +0.20 / 0.27 | +0.23 / 0.37 |

Reading:
- The 1/K0 band is what the old-trace test missed: 2.7x the accepted rows at the same null, the same DIA-NN
  agreement (79-80% reported, 99% on its peak), and +2,300 to +2,516 ions against the rescuable tier
  (+1,802 / +1,857 / +1,939 of them DIA-NN's).
- The re-extracted values quantify like DIA-NN's on the same rows (human abs d 0.22 against 0.20-0.21), and better
  than the rescuable transfers, whose traces are still in the band of their own apex. So the rescuable tier's
  quant gap (section 4i) is partly its 1/K0 band.
- Tier 2 alone: global -0.000 to -0.001, eq +0.005, PB CV +0.004 to +0.005 against the rescuable tier. With drop1:
  eq -0.003 to -0.004, global +0.000 to +0.001, PB CV +0.006. Both replicate on three ID sets.
- Tiers 1 and 2 re-extracted: +9,143 ions on seed 0, but our-only E. coli -0.58 and eq +0.018. Tier 1 stays out.
- Against DIA-NN 2.2.0 (118,326 / 0.143 / 0.207 / PB CV 0.107), tier 2 + drop1 on seed 0: ions -13.3% (was
  -15.2%), global -0.003, eq +0.007, PB CV -0.012.

**Engine: `mbr.reextract` (2026-10-02).** `run-experiment`, default off, needs `retrace.enabled`
(docs/12, "mbr.reextract"). The worker is `scripts/mbr_reextract.py` (`prep`, `score`), NumPy only. Retrace runs
in-process on each run's targets, with its inputs read from the run's retrace report. Quant reads the accepted
transfers from the new traces through a second `ChromTable` with `drop` lists, so quant itself is unchanged.
Worker parity on the seed 0 inputs (`wk_s0`: the prototype's population, same retrace, same harness quant):
35,432 accepted rows against the prototype's 35,233 (NumPy logistic regression instead of sklearn), 3,542 null
draws at or above the threshold (expected false 354); 102,651 ions, 0.139 / 0.223 / PB CV 0.093, shared
93,189: 0.134 / 0.095 / -1.77 (prototype 102,640, 0.139 / 0.223 / 0.093). Prep 79 s, retrace 7-19 s per
run, score 122 s at 34 GB and 22 GB peak.

## 4l. Re-extraction first: order of the two tiers (2026-10-02/03, seeds 0-2)

**End to end, rescuable then re-extraction.** `mumdia run` with the six `.d` (`run-experiment`), binary
`~/bin/mumdia-rx/mumdia`, `eng_repick`'s config plus `retrace.enabled`, the quant preset written out
(`fixed_scan_halfwidth` 4, quantile 0.6, `cross_run_weights`, `cross_run_background`, `cross_run_width` 2.5),
`library_irt: library`, `multihead_calibration: 80`, MBR `rt_transfer` + `reextract`, and the `l1d` FASTA
library tables (`e2e_rx/`). 2:48:58 wall, 134 GB peak, exit 0. Rescore: 550,484 target PSMs at 1% over six
runs, 100,634 peptides (`eng_repick`: 543,082). Rescuable tier: window 2.5 s (`eng_repick` 1.7 s), 55,048
transfers, 549 null draws inside. Re-extraction: 44,111 targets tested, 17,710 accepted, 1,770 null draws at
or above the threshold (expected false 177). Paired against the rescuable tier alone requanted from the same
run (`e2e_resc`, `requant_crw.sh` with `q_filter: psm_q`):

| arm | ions | global | eq | PB CV | shared with 2.2.0 | MuMDIA-only |
|---|---|---|---|---|---|---|
| rescuable only | 102,617 | 0.143 | 0.227 | 0.093 | 93,273: 0.137 / 0.095 / -1.78 | 9,344: 0.247 / 0.147 / -0.87 |
| rescuable + re-extraction (engine) | 103,530 | 0.143 | 0.230 | 0.095 | 93,999: 0.137 / 0.097 / -1.77 | 9,531: 0.245 / 0.149 / -0.93 |

The engine runs (worker, retrace from the reports, two-table quant). The gain is +913 ions, not the +2,300 of
section 4k, because the rescuable tier here takes 55,048 rows at a 2.5 s window, and those rows keep traces in
the 1/K0 band of their own apex. Re-extracted values quantify better than rescuable ones (section 4k: human abs d
0.22 against 0.29), so the order of the tiers matters.

**Re-extraction only and the union** (harness, `quant_diag/wk_only.sh`, `wk_union.sh`). Re-extraction only:
`mbr_reextract.py` on the MBR-off scored table, so every precursor confident in >= 2 other runs and not confident
here is a target, also those the rescuable tier would take. Union: re-extraction first, then the rescuable
transfers of `mbr_s<seed>` for the rows re-extraction did not accept (16,056 on seed 0), every transfer
quantified on the re-extracted traces.

| seed | arm | transfers (null at or above; expected false) | ions | global | eq | CV | PB CV | E. coli / human / yeast abs eps | E. coli log2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | rescuable only | 39,071 | 100,340 | 0.140 | 0.218 | 0.093 | 0.089 | 0.326 / 0.124 / 0.204 | -1.76 |
| 0 | re-extraction only | 61,384 (6,137; 614) | 100,799 | **0.135** | **0.204** | 0.095 | 0.090 | 0.295 / 0.121 / 0.198 | -1.78 |
| 0 | union | 61,384 + 16,056 | **102,772** | 0.139 | 0.216 | 0.098 | 0.093 | 0.323 / 0.123 / 0.202 | -1.77 |
| 1 | rescuable only | 37,434 | 99,775 | 0.139 | 0.217 | 0.093 | 0.089 | 0.323 / 0.124 / 0.203 | -1.77 |
| 1 | re-extraction only | 60,806 (6,078; 608) | 100,371 | 0.135 | 0.204 | 0.095 | 0.090 | 0.295 / 0.121 / 0.197 | -1.78 |
| 1 | union | | 102,248 | 0.138 | 0.215 | 0.098 | 0.093 | 0.321 / 0.123 / 0.202 | -1.77 |
| 2 | rescuable only | 35,106 | 99,846 | 0.139 | 0.216 | 0.093 | 0.088 | 0.320 / 0.124 / 0.204 | -1.77 |
| 2 | re-extraction only | 60,670 (6,066; 607) | 100,607 | 0.135 | 0.205 | 0.095 | 0.090 | 0.295 / 0.121 / 0.198 | -1.78 |
| 2 | union | | 102,494 | 0.139 | 0.215 | 0.098 | 0.093 | 0.321 / 0.123 / 0.202 | -1.77 |
| | DIA-NN 2.2.0 | | 118,326 | 0.143 | 0.207 | 0.114 | 0.107 | 0.282 / 0.125 / 0.215 | -1.77 |

Shared with DIA-NN 2.2.0 (eps / CV / E. coli log2; DIA-NN on the same ions 0.127-0.128 / 0.104 / -1.80):

| seed | rescuable only | re-extraction only | union |
|---|---|---|---|
| 0 | 91,379: 0.134 / 0.091 / -1.78; only 8,961: 0.231 / 0.136 / -1.03 | 92,067: 0.130 / 0.093 / -1.80; only 8,732: 0.215 / 0.129 / -0.91 | 93,277: 0.133 / 0.096 / -1.79; only 9,495: 0.223 / 0.138 / -1.06 |
| 1 | 90,896: 0.134 / 0.090 / -1.79; only 8,879: 0.231 / 0.136 / -1.00 | 91,715: 0.130 / 0.092 / -1.80; only 8,656: 0.215 / 0.129 / -0.85 | 92,825: 0.133 / 0.095 / -1.79; only 9,423: 0.224 / 0.139 / -1.02 |
| 2 | 91,024: 0.134 / 0.090 / -1.79; only 8,822: 0.234 / 0.135 / -1.02 | 91,920: 0.130 / 0.093 / -1.80; only 8,687: 0.213 / 0.131 / -0.89 | 93,055: 0.133 / 0.095 / -1.79; only 9,439: 0.225 / 0.139 / -1.03 |

Against the rescuable tier alone, the same on all three seeds:
- Re-extraction only: +459 / +596 / +761 ions, global -0.004 to -0.005, eq -0.011 to -0.014, PB CV +0.001 to
  +0.002. Global and eq are better than DIA-NN 2.2.0's (0.135 / 0.204 against 0.143 / 0.207), at -15% ions.
- Union: +2,432 / +2,473 / +2,648 ions, global 0.000 to -0.001, eq -0.001 to -0.002, PB CV +0.004 to +0.005.
  Better than rescuable + re-extraction (section 4k) on every metric.
- FDR: each tier is tested at transfer q 0.01 against its own null. The union holds two tested sets, so its
  expected false count is at most the sum of the two (about 614 + 390 on seed 0, 1.3% of 77,440 transfers).

Engine consequence: re-extraction runs first, on the MBR-off scored table; `mbr.rescuable` (default true) then
adds the rescuable transfers for the rows it did not accept. `mbr.rescuable: false` is re-extraction only.
Worker check on seed 0 (`mbr_reextract.py score --rescuable`): 61,384 + 16,056 = 77,440 transfers, the harness
union exactly.

**End to end, re-extraction first** (`e2e_rx2/`, same binary path rebuilt with the new order, same config,
default `mbr.rescuable`). 3:29:14 wall, 135 GB peak, exit 0. The rescore is identical to `e2e_rx` (550,484 target
PSMs, 100,634 peptides), so `e2e_resc` is the paired base. Re-extraction: 97,837 targets tested, 52,554 accepted,
5,254 null draws at or above the threshold (expected false 525); 25,481 rescuable transfers added.

| arm (same IDs) | ions | global | eq | CV | PB CV | E. coli / human / yeast abs eps | E. coli log2 |
|---|---|---|---|---|---|---|---|
| rescuable only | 102,617 | 0.143 | 0.227 | 0.098 | 0.093 | 0.343 / 0.126 / 0.213 | -1.75 |
| rescuable, then re-extraction | 103,530 | 0.143 | 0.230 | 0.101 | 0.095 | 0.351 / 0.126 / 0.213 | -1.74 |
| re-extraction, then rescuable | **103,714** | **0.141** | **0.222** | 0.101 | 0.095 | 0.332 / 0.125 / 0.208 | -1.76 |

Shared with DIA-NN 2.2.0: 94,139 ions, 0.135 / 0.098 / -1.79 (DIA-NN 0.130 / 0.105 / -1.80); MuMDIA-only 9,575:
0.240 / 0.148 / -1.03. Against the paired base: +1,097 ions (+866 DIA-NN's), global -0.002, eq -0.005, PB CV
+0.002. The ion gain is smaller than in the harness (+2,432) because this run's rescuable window is 2.5 s, not
1.7 s. This run's base also quantifies worse than `eng_repick`'s (0.143 / 0.227 against 0.140 / 0.218 for the
rescuable tier alone); that is the end-to-end identification base, not MBR.

## 4m. Plan: a second-pass search as a third MBR strategy (2026-10-03, not measured)

The current MBR is called "transfer" below: the rescuable tier plus the re-extraction tier of sections
4i to 4l. This section plans a second strategy, a second-pass search with an empirical library. One
switch selects between them. Nothing here is measured yet. The numbers to beat are at the end.

### What DIA-NN does, and what is assumed

Sources: Demichev et al. 2020 (Nat Methods 17:41), Demichev et al. 2022 (Nat Commun 13:3944,
dia-PASEF), and the DIA-NN documentation of `--reanalyse`. These sources describe the following:
- MBR is a two-step analysis. The first pass searches every run. Its identifications form an empirical
  spectral library, with observed fragment intensities, retention time and, on dia-PASEF, ion mobility.
  The second pass searches every run again with that library.
- DIA-NN generates decoys for whatever library it searches, so the second-pass decoys come from the
  empirical library.
- The second pass is the reported result.

The following points are assumptions. The publications do not specify them, and nothing here copies
DIA-NN code or constants:
- the q unit and threshold for library inclusion;
- how intensities are combined over runs;
- the second-pass RT and 1/K0 window widths;
- whether a run's own data enter its library.

### SP1. The switch

`mbr.strategy` becomes `none | transfer | second_pass`.
- `transfer` is the behaviour of today's `rt_transfer`, without change. `mbr.reextract` and
  `mbr.rescuable` remain its sub-keys, with their current defaults and validation.
- `rt_transfer`, `empirical_library` and `full` remain enum variants, so old configs still parse
  under `deny_unknown_fields`. `validate()` maps them to `transfer` and logs a warning that names the
  replacement. Their meaning does not change, because today all three run the transfer tiers.
  `empirical_library` is not made an alias of `second_pass`: that would silently change what an old
  config does. The warning exists because the name now suggests the other strategy.
- The current "only none versus not-none is read" warning is removed. The inert fields
  (`rt_window_s`, `decoy_transfer`, `requant_all`) keep their own warning. They are out of scope here.
- New sub-table `mbr.second_pass` (serde-typed, `deny_unknown_fields`):
  - `lib_q` (0.01): the library cut;
  - `lib_min_runs` (1): the minimum number of runs that support the precursor;
  - `leave_run_out` (false = DIA-NN-like, the default; true is the later test arm of SP2);
  - `rt_halfwidth_s` and `im_halfwidth` (0 = sized from held-out cross-run residuals, SP2);
  - `shrink_runs` (2, SP2).
- Validation:
  - `second_pass` together with `reextract: true` is an error;
  - `leave_run_out: true` needs at least 2 runs (the default also works on one run, as DIA-NN does);
  - the two strategies are not combined in this plan.

### SP2. The empirical library

**Inclusion: unit, value and minimum runs.** Counts on the targets of `e2e_rx2/run/scored_combined`.
"Runs" means runs with pooled PSM `q_value` <= 0.01.

| cut | precursors | in 1 run | in 2 | in 3-5 | in 6 |
|---|---|---|---|---|---|
| `precursor_q` <= 0.01 | 112,463 | 8,566 | 9,122 | 30,922 | 63,853 |
| `precursor_q` <= 0.02 | 119,806 | 13,666 | 10,689 | 31,586 | 63,865 |
| `precursor_q` <= 0.05 | 130,822 | 18,408 (+5,780 in 0) | 11,125 | 31,644 | 63,865 |
| >= 1 run at pooled `q_value` <= 0.01 | 125,042 | 18,408 | 11,125 | 31,644 | 63,865 |
| >= 2 runs at pooled `q_value` <= 0.01 | 106,634 | 0 | 11,125 | 31,644 | 63,865 |

- **Unit: precursor-level q, not a per-row PSM q.** The library is a set of precursors, so its false
  fraction is controlled by a precursor-level q. A union of per-run PSM acceptances at 1% does not have
  that property. The pooled PSM cut of 1% admits about 5,500 false rows over six runs. A false row rarely
  repeats in another run, so most of those false rows become false precursors. They concentrate among
  the 18,408 single-run precursors. `precursor_q` <= 0.01 keeps 8,566 of those 18,408.
- **Value: 0.01, with min runs 1 (DIA-NN-like, the default).** One library from the experiment-wide
  `precursor_q` of the pooled pass-1 rescore (112,463 precursors on e2e_rx2), searched in every run.
  A looser cut reaches more ions: at 0.05, +18,359 precursors, of which 5,780 have no confident run.
  It also admits more false precursors, and in the default a false entry is searched in its anchor run
  with a spectrum and apex taken from that run (circularity, below). 0.02 and 0.05 are therefore
  prototype arms only, and the default stays 0.01 until entrapment says otherwise.
- **Where the 1-anchor rows are.** The 30,907 1-anchor rows of section 4i belong to the 18,408
  single-run precursors, of which the 0.01 cut admits 8,566. The first prototype step counts how many
  of the 30,907 rows, and how many of the 68,158 no-anchor rows, each cut reaches (`mbrgap.py` on the
  library lists).
- **Pass-1 rows outside the library.** 9,842 single-run precursors have a row at pooled `q_value`
  <= 0.01 but `precursor_q` > 0.01. They are not searched in pass 2, and as in DIA-NN only the second
  pass is reported, so their rows are dropped. Most cannot reach `min_obs` 3 in any case. The count per
  run is reported and is part of the sensitivity gate (SP7).

**Circularity, the known weakness of the default.** In the default, three things in an anchor run j were
selected on run j's own data: a false precursor's pass-1 match there, its empirical spectrum and its
apex. The target can then be favoured over its decoy in exactly its anchor runs, which makes q
optimistic. The size of this effect is bounded by the false library entries, about 1% of the library
at a 0.01 cut. The single-run E. coli entrapment of SP7 measures this case directly, because a
single-run second pass is all self-library.

**Later test: leave-run-out (`leave_run_out: true`).** The library that searches run j is built from
runs other than j only: inclusion (`q_-j`, the engine's `precursor_q` rule over the best row per
precursor and label in runs other than j), intensities, RT and 1/K0. Inclusion and library values are
then independent of run j, so a false entry costs sensitivity rather than FDR. That would also make
looser cuts safer. Consequence: a precursor supported in run j only is not in library j. It keeps its
pass-1 row in run j (a disjoint union, so the report stays at 1%), and it is searched in every other
run. It needs per-run fragment tables. This arm runs after the default holds, and its acceptance in
anchor runs, compared with the default on the same precursors, measures the circularity.

**Fragment intensities.** These are the pass-1 fixed-window fragment areas (`fragment_quant.parquet`
of the pass-1 quant, the input of `fit_cross_run`). They are taken over the precursor's confident runs
(run-level `run_psm_q` <= 0.01; if there is none, the best-scoring run). Under leave-run-out, run j is
excluded from this set.
- Each run's areas are max-normalised. The library value is the median over runs (3 or more runs) or
  the mean (1 or 2 runs). The median is robust to one interfered run (section 4j, drop1).
- With fewer than `shrink_runs` supporting runs, the library value is the mean of the empirical and the
  predicted pattern. Section 13 measured the run-to-run cosine at 0.986 and the prediction at 0.967. One
  run is noisier than that.
- Fragments: only the 12 library fragments, because pass 1 measures no others. A fragment with area 0
  in every supporting run gets intensity 0 and stays in the library. Section 12 found such slots
  harmless for identification. Choosing a new top 12 from all b/y ions needs a retrace with an expanded
  fragment list. That is a later step, and so is the fine-tuned model of section 14.

**RT and 1/K0, per run.**
- Expected RT in run j: the binned-median cross-run maps of `mbr_worker.py`, through run 0, as the
  median over the supporting runs (other than j under leave-run-out). In the default, a run where the
  precursor is confident contributes its own apex to that median.
- Expected 1/K0: the median of the supporting runs' re-picked `apex_im`, moved by run j's median offset.
  This is the `mbr_reextract.py prep` rule, reused.
- Half-widths: the p99 of the held-out residual (|observed - expected| on confident rows, each predicted
  without its own run), per run. This is not the in-sample residual (CLAUDE.md, RT rules). For scale,
  the rescuable windows were 1.5-2.5 s and pass 1 used about 14 s after the refit.
- These values go straight into per-run `run_windows` tables (`rt_pred_cal`, `rt_lo`, `rt_hi`,
  `im_pred_cal`, `im_lo`, `im_hi`). `rt-im-train` does not run in pass 2.

**Table layout.**
- One library (precursors and fragments) with contiguous `candidate_id` sorted by `precursor_mz`,
  written with `_lib_io.write_engine_parquet`, and one id space for the pooled rescore and
  `precursor_q`.
- Per run: its `run_windows`. Under leave-run-out, also its own fragment table and a
  `--restrict-candidates` list holding library j and its decoys, against a master library that holds
  the union.
- An old-to-new id map, used to carry the seed rows across.

### SP3. Decoys for the second pass

This is the central FDR risk. Targets with empirical values against decoys with predicted values let
the classifier learn the source. The construction below makes the two members of each pair identical in
everything except fragment m/z.
- **Sequence:** the target's native paired decoy from the search library: the same `base_peptide_id`,
  charge and modform, reversed with the C terminus fixed, already collision-checked against all 14.7M
  targets (`digest.rs` `collision_safe_decoy`). The decoy therefore never equals a real peptide of the
  search space. Its seed row (pass 1) can be carried over through the id map.
- **Fragments:** the target's 12 ion names (type, ordinal, charge), with m/z recomputed on the decoy
  sequence. Intensities are copied ion for ion. This is `make_reverse_decoys.py`, whose m/z calculator
  is checked against the library's target m/z before it writes.
- **Precursor m/z, RT centre and width, 1/K0 centre and width:** the target's. Reversal keeps the
  composition, so the precursor m/z is equal.
- **Per run:** the decoy copies the target's window for that run (and, under leave-run-out, its
  intensities for that run).

Exchangeability tests, in this order:
1. **Construction invariant, asserted by the writer.** Within each pair, precursor m/z, the window
   columns and the sorted intensity vector are equal. A classifier trained only on library-side
   columns (window width, centre, intensity statistics, number of nonzero fragments) must give
   AUC = 0.5. The test also lists which features read fragment m/z, such as the fraction of fragments
   above the precursor m/z. Those features are the only channel through which the source can leak.
2. **Low-score null,** as in TIMS_ROADMAP P5: per-feature AUC of target against decoy below the median
   pass-2 score. Report the largest deviations, and compare them with the same table for pass 1.
3. **Known-false library entries (entrapment, SP7).** Entrapment precursors that enter the library are
   false in every run. Their pass-2 score distribution must match the decoys' distribution (AUC of
   entrapment target against decoy about 0.5, KS test). This is the direct test that decoys represent
   false library entries. A single E. coli run gives only about 50-60 such entries at a 1% cut. That
   test therefore also forces entrapment precursors in at a loose cut (pass-1 q <= 0.2) to get hundreds.
4. **Circularity** (with the later leave-run-out arm): in the anchor runs, pass-2 acceptance of the
   default minus that of the leave-run-out arm, on the same precursors. A large excess means self-fit.
   Until that arm runs, compare instead the pass-2 decoy fraction and score distribution in anchor runs
   against those in non-anchor runs of the same precursors.

### SP4. The pipeline

```text
pass 1 (unchanged): per run convert, seed, rt-im-train, extract, retrace(+repick), features, compete
    -> pooled rescore -> [rt_im_train.refit: second extract..compete, pooled rescore]
    -> pass-1 quant (fragment areas only needed)
second pass: mbr_second_pass.py build (library, decoys, per-run windows, restrict lists, id map)
    -> per run: extract (empirical library, run-j windows)
                retrace (+repick), features (seed rows remapped), compete
    -> pooled rescore (same classifier and recipe), per-run run_psm_q and pooled q_value
    -> merge, quant (diaPASEF preset, cross-run fit on pass-2 fragment tables), report, manifest
```

- **Refit.** `rt_im_train.refit` stays inside pass 1 and is unchanged. The second pass starts from its
  rescore. Pass 2 does not run `rt-im-train` again, because its windows come from the empirical values.
  `experiment.rt_library_scope` and the multi-head calibration play no part in pass 2 (no DeepLC call).
- **Repick.** `retrace.repick` stays on. The narrow window bounds the re-pick to peaks near the
  expected RT, which is the RT prior that section 9 found missing, now with a few-second window.
- **Seed features.** Seed rows are remapped by the id map. A decoy without a seed row gets the
  missing-seed values, as in pass 1. In the default, a library target's seed score in its anchor run
  contributed to its inclusion, while its decoy's did not. This is part of the circularity, and
  SP3 tests 2 and 4 check it. If it shows, the fix is to project the seed features out of the pass-2
  classifier (`rescore.features`), not a new code path.
- **q.** Pass 2 computes per-run `run_psm_q`, pooled `q_value` and the grouped q columns over pass-2
  rows only. Scores from two classifiers never meet in one target-decoy competition.
- **Report: the second pass only (default, as in DIA-NN).** Pass-1 rows that pass 2 does not
  reproduce are dropped, both for library precursors and for the precursors outside the library.
  Keeping both would put two overlapping 1% sets in one report, which can approach 2%. The number
  dropped is reported per run, and it is a gate (SP7). Under leave-run-out the report is the disjoint
  union of SP2 instead.
- **Flags.** Rows accepted in pass 2 but not in pass 1 for that run get `is_transferred = true` (no new
  column, no schema bump). Report and ProteoBench handle them as today (`MBR=true score.sh`).
  `transfer_q` stays NaN.
- **Quant.** Quant is unchanged. It reads pass-2 chromatograms only, so the second `ChromTable` of the
  re-extraction path is not needed. `fragment_selection: predicted` now selects on empirical
  intensities. The `cross_run_*` keys fit on the pass-2 fragment tables. drop1 (section 4j) is a cheap
  requant arm for the new rows.
- **Manifest.** A `second_pass` block records the library and per-run table hashes, the cut and unit,
  target and decoy counts per run, the dropped pass-1 rows, and the pass-2 classifier identity from
  `psms_scored.parquet.report.json`. Pass-2 artifacts go to `<out>/second_pass/`, and pass-1 artifacts
  stay where they are.

### SP5. Cost

Estimate from the e2e_rx2 stage timings, 64 threads. The library has about 112k targets plus their
decoys, 225k candidates (1.5% of 14.7M).

| step | e2e_rx2 reference | second pass, estimate |
|---|---|---|
| pass 1 plus refit, through the pooled rescore | 3:00 | 3:00 (unchanged) |
| transfer MBR, re-extraction, quant, report | 29 min | not run |
| pass-1 quant (fragment areas) | 4-9 min of the 29 | 5-9 min |
| library build (Python) | | 2-5 min |
| per-run extract | 291-338 s at 14.7M candidates | about 1 min (mostly spectra decode and library load) |
| per-run retrace with repick | 234-287 s at 102M rows | 1-3 min (the re-extraction retrace took 29-89 s at 250k rows) |
| per-run features, compete | 100-110 s | under 20 s |
| pooled rescore | 22.8 min at 40.9M PSMs | 12-17 min at about 1.35M PSMs |
| quant, report | about 5 min | about 5 min |

- The rescore does not shrink with the PSM count. Training reads the targets at 1% plus up to 2 decoys
  per positive (`train_neg_ratio: 2`). In pass 1 that was about 550k plus 1.1M rows. In pass 2 it is
  about 600k plus all about 675k decoys.
- Total: second pass about 35-50 min, replacing the 29 min of the transfer path, so about 3:40-3:50 end
  to end against 3:29.
- The peak stays in pass 1 (135 GB). Pass-2 retrace holds the raw frames, 28-36 GB per run, so the
  per-run chains run three at a time.

### SP6. Prototype first, engine later

Offline on the harness, with no engine change. The stages are path-addressed, and the binary is
`~/bin/mumdia-rx/mumdia`. Scripts go in `quant_diag/`, arms in `sp_<idset>_<arm>/`.
1. **P0, populations** (minutes): the library lists for the `precursor_q` cuts 0.01 / 0.02 / 0.05.
   Run `mbrgap.py` coverage on them: how many of the 30,907 1-anchor and 68,158 no-anchor rows each cut
   reaches, and how many pass-1 rows per run fall outside each library.
2. **P1, `sp_lib.py`:** writes the library, the decoys, the per-run windows and the id map, plus the
   per-run fragment tables and restrict lists when `--leave-run-out` is given. It imports the m/z code
   of `make_reverse_decoys.py`, the RT and 1/K0 rules of `mbr_reextract.py` / `mbr_worker.py`, and
   `_lib_io`. It asserts the SP3 invariant.
3. **P2, `sp_chain.sh <idset> <arm>`:** per run, `mumdia extract` (`--lib-precursors`,
   `--lib-fragments`, `--run-windows`, and `--restrict-candidates` for leave-run-out), `retrace`
   (repick), `features` (remapped `--seed-psms`) and `compete`. Then pooled
   `mumdia rescore --competed r0..r5` with seeds 0-2.
4. **P3, `sp_eval.sh`:** report rows (SP4), then quant with the preset (as in `requant_crw.sh`), then
   `MBR=true score.sh`, `DIANN=220 shared.py` and `mbrgap.py`. It also runs exchangeability tests 1, 2
   and 4.
5. **Arms on `eng_repick` s0:**
   - SP-A (DIA-NN-like default, 0.01);
   - SP-A at 0.02 and 0.05;
   - SP-A with drop1 on the new rows;
   - SP-A with the seed features projected out (only if test 2 or 4 shows a seed effect).

   The best arm then runs on 3 rescore seeds, on `eng_repick_s1`, and on the e2e_rx2 pass-1 IDs.
6. **Entrapment (SP7) before any engine work.**
7. **Later:** SP-L (leave-run-out, 0.01, then looser cuts), against SP-A on the same ID sets.

Engine plumbing, only after the arm holds on at least two ID sets and passes entrapment:
- `config.rs`: the enum mapping and the `mbr.second_pass` sub-table of SP1, with validation and tests
  that old configs parse to `transfer`.
- `scripts/mbr_second_pass.py build` and `merge`, from `sp_lib.py` (NumPy, pyarrow), launched like
  `mbr_reextract.py`.
- `run_experiment.rs`: after the final pooled rescore, branch on the strategy. Reuse the refit pass's
  per-run chain call with other library, window and restrict paths. Reuse the pooled rescore and the
  per-source quant split.
- docs/12 ("mbr") and CLAUDE.md, after measurement.

### SP7. Validation and gates

Numbers to beat (ions at `min_obs` 3 / global / eq / PB CV):

| reference | ions | global | eq | PB CV |
|---|---|---|---|---|
| DIA-NN 2.2.0, MBR on (target) | 118,326 | 0.143 | 0.207 | 0.107 |
| e2e transfer, re-extraction then rescuable | 103,714 | 0.141 | 0.222 | 0.095 |
| e2e, same IDs, rescuable only | 102,617 | 0.143 | 0.227 | 0.093 |
| harness s0, rescuable | 100,340 | 0.140 | 0.218 | 0.089 |
| harness s0, union | 102,772 | 0.139 | 0.216 | 0.093 |
| harness s0, re-extraction only | 100,799 | 0.135 | 0.204 | 0.090 |

Gates (harness arms compare with the harness rows, the e2e arm with the e2e rows):
- **FDR.**
  - Entrapment on the E. coli diaPASEF setup of TIMS_ROADMAP_bis section 8: empirical FDP of the
    pass-2 rows at q 0.01 at or below pass 1's (0.37-0.47%), over 3 seeds, plus SP3 test 3.
  - That setup is one run, so its second pass is all self-library. That is the default's worst case of
    circularity, which makes it the right first test for the default. It cannot test cross-run
    propagation of a false entry (see R1).
- **Sensitivity.**
  - More ions than the transfer reference of the same ID set: harness above 102,772, e2e above 103,714.
  - Per-run rows at least pass 1's in every run (no net loss from the dropped pass-1 rows).
  - DIA-NN-shared ions up.
- **Accuracy before CV.** Global and eq at or below the transfer union of the same ID set, on all ions
  and on DIA-NN-shared ions (`shared.py`). PB CV is reported. MuMDIA-only E. coli log2 is checked
  against the tier-1 failure of section 4j (near 0 means faint false or wrong rows).
- **Gap decomposition** (`mbrgap.py`) for the new arm: the change in the 1-anchor (30,907),
  no-anchor (68,158) and remaining (10,652) rows.
- **Replication:** 3 rescore seeds on `eng_repick`, plus a second ID set.
- **Promotion** to a diaPASEF default also needs a second acquisition, per project policy.

### SP8. Risks and open decisions, ranked

1. **Circularity in the default, and no multi-run entrapment.** The default searches each anchor run
   with a spectrum and apex taken from that run. The E. coli entrapment measures that self-library case,
   but it is one run. A false entry that propagates to the other runs needs an entrapment experiment
   with several runs: the HYE runs searched with a library that adds a foreign proteome (for example
   1:1 by peptide count), through pass 1 and pass 2. That costs a library build and an e2e run (about
   5-7 h). Decision needed: build it, or gate on the E. coli entrapment plus SP3 tests 1, 2 and 4.
   **Drop if:** entrapment FDP of pass 2 exceeds 1% at q 0.01, or entrapment targets in the library
   score above decoys (SP3 test 3), and neither projecting out the seed features nor leave-run-out
   fixes it.
2. **Quant of the new rows.** Section 4j rejected the 1-anchor tier on quant, not on FDR: its new ions
   had E. coli log2 near 0 even where DIA-NN reports them. Pass 2 reaches the same population. A better
   library and window may not change a faint, partly interfered signal. **Drop or park if:** the ion
   gain comes with eq above the transfer union on two ID sets, and drop1 does not recover it.
3. **Pass 2 loses pass-1 rows.** A library of mostly true targets changes the classifier's training
   population. The 9,842 single-run precursors outside the 0.01 library are lost by construction. The
   dropped pass-1 rows could outnumber the new ones in some runs. **Drop if:** net per-run rows fall
   below pass 1 beyond seed noise, or ProteoBench ions fall.
4. **Noisy empirical patterns for 1- and 2-run precursors.** These are exactly the precursors pass 2
   is for. The fallback is shrinkage toward the prediction (`shrink_runs`). If shrinkage is not enough,
   the alternative is the fine-tuned fragment model of section 14 in place of raw empirical patterns.
5. **Too small a gain.** The project rule is to skip about 1% gains that need engine code. **Drop if:**
   the best arm adds under about 1,000 ions over the transfer union with no accuracy gain.
6. **Cost:** +10-20 min over transfer. Acceptable if it gains. Not a reason to drop.

Open decisions for review:
- **a.** Multi-run entrapment (R1).
- **b.** Reporting policy: the second pass only (recommended, DIA-NN-like, FDR-safe), or keep the
  unreproduced pass-1 rows with a flag (up to about 2% FDR on those rows).
- **c.** Later: combine `second_pass` with the re-extraction tier for library precursors that pass 2
  does not accept. Excluded from this plan.

## 4n. Second pass measured: more ions, worse eq (2026-10-03/04, eng_repick s0 and s1)

Offline prototype of section 4m (SP6 P0 to P3), no engine change. Binary `~/bin/mumdia-rx/mumdia`. Search
library and seeds: the `l1d/run` tables that `eng_repick` searched. Pass-1 fragment areas: `q_rp_crwbg_w25_s<seed>`
(the preset, MBR off). Scripts in `quant_diag/`:
- `sp_p0.py`: library populations (P0).
- `sp_lib.py`: library, decoys, per-run windows, id map, remapped seeds, and the SP3 test 1 invariant (P1).
- `sp_chain.sh <idset> <arm> [seeds]`: the per-run chain and the pooled rescore (P2).
- `sp_eval.py` / `sp_eval.sh`: report rows, SP3 tests 1, 2 and 4, then quant, `MBR=true score.sh`,
  `DIANN=220 shared.py` and `mbrgap.py` (P3). `mbrgap.py` now takes `Q=<arm>`.
- `sp_arms.sh`: library, chain and evaluation for several arms in turn.

Arms are in `sp_<idset>_<arm>/`. Library inclusion is `precursor_q` <= cut only (rule (a) of the P0 review).

**P0, library populations.** Precursor = `candidate_id`. Its `precursor_q` is the minimum over its rows. Runs are
counted at pooled `q_value` <= 0.01. The gap rows are those of `mbrgap.py` on `mbr_s0`.

| ID set | cut | precursors | 0 runs | 1 | 2 | 3-5 | 6 | 1-anchor rows reached (ions) | no-anchor rows reached (ions) | pass-1 rows outside |
|---|---|---|---|---|---|---|---|---|---|---|
| eng_repick | 0.01 | 112,800 | 0 | 9,360 | 9,728 | 32,046 | 61,666 | 18,878 of 30,907 (4,750) | 28 of 68,158 (6) | 15,171 |
| eng_repick | 0.02 | 119,855 | 0 | 14,411 | 11,187 | 32,588 | 61,669 | 25,965 (6,533) | 42 (9) | 5,380 |
| eng_repick | 0.05 | 130,807 | 6,069 | 18,853 | 11,579 | 32,637 | 61,669 | 30,907 (7,760) | 6,550 (1,324) | 0 |
| e2e_rx2 | 0.01 | 112,463 | 0 | 8,566 | 9,122 | 30,922 | 63,853 | 18,106 (4,495) | 8,195 (1,599) | 16,292 |
| e2e_rx2 | 0.02 | 119,806 | 0 | 13,666 | 10,689 | 31,586 | 63,865 | 22,583 (5,633) | 11,582 (2,287) | 5,794 |
| e2e_rx2 | 0.05 | 130,822 | 5,780 | 18,408 | 11,125 | 31,644 | 63,865 | 26,556 (6,647) | 17,490 (3,499) | 0 |

- A library made from the same ID set cannot reach the no-anchor rows. Those rows belong to precursors that no
  run accepts. The e2e_rx2 rows are a different ID set and are not comparable on this point.
- The pass-1 rows outside the 0.01 library are 15,171 on `eng_repick`, not the 9,842 of section 4m SP2. Multi-run
  precursors with `precursor_q` > 0.01 add the rest. About 594 of these precursors have 3 or more confident runs.

**Library and windows (P1, eng_repick s0, cut 0.01).**
- 112,800 targets and 112,800 native decoys. 112,751 decoys are paired through the reversed modform, and 49 are
  paired through equal mass, because their native decoy was scrambled.
- Decoy fragment m/z match the native decoy's own fragments to 0.000 ppm on 963,110 shared ions.
- 88,746 of 1,353,600 target fragments have intensity 0 in every supporting run and are kept.
- Held-out half-widths per run, p99: 7.6-8.8 s in RT and 0.008-0.010 in 1/K0. The held-out RT median is
  0.55-0.78 s.
- Build time 131 s, peak 7.2 GB.

**Deviation from SP4: two trace sets.** Extract and retrace build each trace only inside its RT window. Quant
integrates up to 2h+1 = 15 scans plus flank samples, which is about +-15 s at 0.97 s per scan. The p99 window
(about +-8 s) cannot hold that.
- The ID chain (extract, retrace with repick, features, compete) uses the p99 windows.
- A second extract, plus retrace with repick off, builds the quant traces (`chromatograms_q`). It uses the same
  centres and the pass-1 median half-widths (36.6-40.2 s and 0.040), with the 1/K0 centre at pass 2's re-picked
  `apex_im`.
- Quant reads only those traces. This is the arrangement of the re-extraction tier (section 4k).

**Cost per run.** Extract 86-92 s at 8 GB. Retrace 8-37 s at 27 GB. Features and compete under 3 s. The
quant-trace extract and retrace together take 25 s at 31 GB. Pooled rescore 146 s at 3.8 GB, over 936,930 PSMs.
Evaluation with quant 262 s. The whole arm takes about 20 minutes.

**First run: the 1/K0 window was too narrow** (`sp_eng_repick_A01`, 1/K0 half-width = the p99, 0.008-0.010).
`extract.im_gate: fragments` applies the window to each fragment peak, not only to the apex.
- Of the pass-1 accepted library rows, pass 2 did not reproduce 37,326, and extract did not extract 33,755 of
  them.
- On r0, 4,362 of 87,218 pass-1 accepted library rows were not extracted. With the pass-1 1/K0 width and the
  same RT window, the count was 564.
- Result: 99,711 ions / 0.135 / 0.221 / PB CV 0.089. All later arms use a 1/K0 half-width floor of 0.04
  (`--min-im-hw 0.04`).

**Arms** (eng_repick s0, pass-2 rescore seed 0; eng_repick_s1, pass-2 rescore seed 1). Ions at `min_obs` 3. drop1 is
`robust_tr.py` on the rows that pass 2 accepts and pass 1 did not (`is_transferred`).

| ID set | arm | ions | global | eq | PB CV | E. coli log2 |
|---|---|---|---|---|---|---|
| s0 | transfer union (4l) | 102,772 | 0.139 | **0.216** | 0.093 | -1.77 |
| s0 | re-extraction only (4l) | 100,799 | **0.135** | **0.204** | 0.090 | -1.78 |
| s0 | SP-A 0.01 | 105,801 | 0.139 | 0.235 | 0.093 | -1.72 |
| s0 | SP-A 0.01, drop1 | 105,801 | 0.141 | 0.226 | 0.095 | -1.78 |
| s0 | SP-A 0.02 | 111,354 | 0.143 | 0.247 | 0.094 | -1.70 |
| s0 | SP-A 0.02, drop1 | 111,354 | 0.145 | 0.237 | 0.097 | -1.76 |
| s0 | SP-A 0.05 | **118,146** | 0.148 | 0.262 | 0.096 | -1.67 |
| s0 | SP-A 0.05, drop1 | 118,146 | 0.150 | 0.252 | 0.098 | -1.74 |
| s1 | transfer union (4l) | 102,248 | 0.138 | **0.215** | 0.093 | -1.77 |
| s1 | SP-A 0.01 | 104,933 | 0.139 | 0.233 | 0.092 | -1.72 |
| s1 | SP-A 0.01, drop1 | 104,933 | 0.140 | 0.224 | 0.094 | -1.78 |
| | DIA-NN 2.2.0 | 118,326 | 0.143 | 0.207 | 0.107 | -1.77 |

Shared with DIA-NN 2.2.0 (eps / CV / E. coli log2):

| ID set | arm | shared: MuMDIA | DIA-NN on the same ions | MuMDIA-only |
|---|---|---|---|---|
| s0 | transfer union | 93,277: 0.133 / 0.096 / -1.79 | 0.127-0.128 / 0.104 / -1.80 | 9,495: 0.223 / 0.138 / -1.06 |
| s0 | SP-A 0.01 | 94,826: 0.132 / 0.095 / -1.76 | 0.130 / 0.105 / -1.79 | 10,975: 0.227 / 0.135 / -0.86 |
| s0 | SP-A 0.01, drop1 | 94,826: 0.134 / 0.097 / -1.81 | 0.130 / 0.105 / -1.79 | 10,975: 0.234 / 0.145 / -0.98 |
| s0 | SP-A 0.02 | 97,272: 0.134 / 0.095 / -1.75 | 0.132 / 0.106 / -1.79 | 14,082: 0.239 / 0.137 / -0.67 |
| s0 | SP-A 0.05 | 99,319: 0.136 / 0.096 / -1.75 | 0.133 / 0.107 / -1.79 | 18,827: 0.258 / 0.136 / -0.44 |
| s1 | transfer union | 92,825: 0.133 / 0.095 / -1.79 | 0.127-0.128 / 0.104 / -1.80 | 9,423: 0.224 / 0.139 / -1.02 |
| s1 | SP-A 0.01 | 94,222: 0.132 / 0.094 / -1.76 | 0.130 / 0.105 / -1.79 | 10,711: 0.223 / 0.134 / -0.84 |
| s1 | SP-A 0.01, drop1 | 94,222: 0.134 / 0.097 / -1.80 | 0.130 / 0.105 / -1.79 | 10,711: 0.229 / 0.143 / -0.98 |

**Per-run rows, against pass 1** (target rows at pooled `q_value` <= 0.01).

| ID set | cut | pass 1 | pass 2 | new | dropped, in library | dropped, outside | net per run |
|---|---|---|---|---|---|---|---|
| s0 | 0.01 | 543,082 | 621,490 | 106,093 | 12,514 | 15,171 | +12,453 to +14,040 |
| s0 | 0.02 | 543,082 | 651,273 | 127,221 | 13,650 | 5,380 | +17,526 to +19,112 |
| s0 | 0.05 | 543,082 | 687,050 | 158,729 | 14,761 | 0 | +23,343 to +25,261 |
| s1 | 0.01 | 540,862 | 616,911 | 103,987 | 12,593 | 15,345 | +11,987 to +13,743 |

**Gap decomposition** (`mbrgap.py`, s0). These are the (ion, run) rows that DIA-NN reports and the arm does not
quantify. The anchors are those of `mbr_s0`.

| case | transfer (4i) | SP-A 0.01 | SP-A 0.02 | SP-A 0.05 |
|---|---|---|---|---|
| no anchor run | 68,158 | 68,141 | 68,126 | 63,914 |
| 1 anchor, right peak | 15,501 | 11,928 | 7,189 | 3,921 |
| 1 anchor, wrong peak | 15,406 | 7,736 | 5,429 | 3,994 |
| >= 2 anchors, wrong peak | 6,577 | 1,605 | 1,353 | 1,332 |
| not extracted | 2,633 | 2,108 | 2,054 | 2,043 |
| >= 2 anchors, right peak | 1,442 | 1,246 | 908 | 861 |
| total | 109,717 | 92,764 | 85,059 | 76,065 |

**Where eq is lost** (s0, ions at `min_obs` 3, split by whether any of the ion's runs is a pass-2 new row; eq is the
mean of the per-species median abs epsilon).

| arm | ions, pass-1 rows only | eq | E. coli log2 | ions with new rows | eq | E. coli abs eps | E. coli log2 |
|---|---|---|---|---|---|---|---|
| SP-A 0.01 | 62,386 | 0.143 | -1.86 | 43,415 | 0.367 | 0.594 | -1.47 |
| SP-A 0.01, drop1 | 62,386 | 0.143 | -1.86 | 43,415 | 0.341 | 0.521 | -1.61 |
| SP-A 0.05 | 62,594 | 0.143 | -1.87 | 55,552 | 0.416 | 0.681 | -1.37 |
| SP-A 0.05, drop1 | 62,594 | 0.143 | -1.87 | 55,552 | 0.386 | 0.601 | -1.50 |

The two groups also differ in abundance, so this split locates the loss but does not prove its cause.

**Exchangeability (SP3).**
- **Test 1.** Every library-side column gives AUC 0.500, target against decoy: predicted RT and 1/K0, precursor
  m/z, intensity sum and spread, nonzero fragments, window width and centre.
- **Test 2** is not a null in pass 2. The pass-2 population is mostly true targets, so the targets below the
  median score are faint true peptides. Spectral-similarity features reach AUC 0.92 there, against 0.50 in pass 1.
  - Seed features: at most 0.515-0.520 (`seed_score`) in every arm. This is too small to require the arm that
    projects them out.
  - Fragment-m/z features: at most 0.594-0.608 (`mass_log_evidence`) in the 0.04 arms, against 0.719 in the narrow
    first run. This test cannot separate leakage from the mass accuracy of true matches.
  - A valid version needs another null region, for example targets with no pass-1 support in any run.
- **Test 4.** Fraction of decoys above the pass-2 acceptance threshold, relative to accepted targets:

  | arm | anchor runs | non-anchor runs |
  |---|---|---|
  | 0.01 (s0) | 0.85% | 1.71% |
  | 0.02 (s0) | 0.82% | 1.74% |
  | 0.05 (s0) | 0.76% | 1.80% |
  | 0.01 (s1) | 0.86% | 1.69% |

  The pooled 1% therefore puts about 2x the nominal FDR on the rows in runs where pass 1 did not accept the target,
  which is where most new rows are. Decoy scores are slightly lower in anchor runs (AUC 0.455-0.468, KS D
  0.045-0.077). The test shows no target excess in anchor runs, but it cannot measure circularity on its own.

**Reading.**
- Pass 2 gains ions and passes the per-run sensitivity gate: every run has 12,000-14,000 more accepted rows than
  pass 1 at the 0.01 cut.
  - SP-A 0.01 against the transfer union: +3,029 ions (s0) and +2,685 ions (s1).
  - DIA-NN-shared ions: +1,549 and +1,397.
  - Global epsilon is equal (0.139 against 0.139 and 0.138), and PB CV is equal.
  - SP-A 0.05 reaches the ion count of DIA-NN 2.2.0 (118,146 against 118,326).
- It fails the accuracy gate on both ID sets.
  - eq is +0.019 (s0) and +0.018 (s1) above the union, and +0.009 with drop1 on both.
  - eq rises with the cut: 0.235, 0.247, 0.262.
  - The ions with pass-2 new rows carry the loss: their E. coli ratio is compressed to -1.47, against -1.86 for
    the ions with pass-1 rows only. This is the failure of the 1-anchor tier in section 4j, now at larger scale.
  - The elevated decoy fraction on those rows (test 4) is a plausible contributor, because false rows compress
    ratios. It is not measured separately here.
- By SP8 R2 (eq above the union on two ID sets, not recovered by drop1), SP-A is parked in this form. The DIA-NN-like
  arms below replace it.
- Not done: SP-A at 0.02 and 0.05 on s1, rescore seeds 1-2 on s0, the projected-seed arm (not triggered), SP-L,
  entrapment.
- What it would need: an FDR control specific to the new rows (test 4), or transfer quant that holds faint, partly
  interfered values to the right ratio (the open item of section 4j).

**DIA-NN-like arms (2026-10-04).** Three differences from DIA-NN 2.2.0 come from its log of the 220_MBR run
(`bench/diann_compare/220_MBR/param_0..txt`) and its documentation:
- the empirical library is cut at global precursor q 0.05 (`--out-lib-qvalue` default; 125,271 target precursors),
  and the report keeps `Lib.Q.Value` <= 0.01 together with the run-specific `Q.Value` <= 0.01 (every row of its
  ProteoBench input);
- under `--rt-profiling` ("IDs, RT & IM profiling", the default) the library keeps the predicted fragment intensities
  and takes RT and 1/K0 from the data;
- it generates new decoys for the empirical library, by shuffling or by mutating one residue, with the termini kept.

Pass-2 windows in that log: RT 0.94 (minutes; pass 1 2.4-2.7) and IM 0.01 (pass 1 0.042). The log does not say whether
these are half-widths. The arms below add the three differences one at a time (cumulative). All use the 0.05 library
and the windows of SP-A (RT p99, 1/K0 0.04).
- B: `sp_eng_repick_A05i`, report limited to pass-1 `precursor_q` <= 0.01 (`LIBQ=0.01 TAG=_libq01 sp_eval.sh`). The
  other library targets are searched and compete, and their rows get q 1.0.
- C: B plus predicted intensities (`sp_lib.py --intensity predicted`), arm `A05p`.
- D: C plus new decoys (`--decoys reverse_nc`), arm `A05pd`.
  - Each library target gets a new decoy: the residues between the first and the last are reversed, and both
    termini stay.
  - A decoy equal (I = L) to any of the 7.4M search-library targets, or to the decoy of another target, is
    re-scrambled in its interior.
  - s0: 130,611 reversed, 193 scrambled, and 3 dropped with their targets.
  - The native decoy still provides the pair's ids and its pass-1 seed row, so the seed features of the new decoys
    are decoy values.

| ID set, rescore seed | arm | ions | global | eq | PB CV | E. coli abs eps | E. coli log2 |
|---|---|---|---|---|---|---|---|
| s0, 0 | transfer union (4l) | 102,772 | 0.139 | 0.216 | 0.093 | 0.323 | -1.77 |
| s0, 0 | B | 105,501 | 0.139 | 0.231 | 0.092 | 0.359 | -1.73 |
| s0, 0 | C | 105,827 | 0.138 | 0.212 | 0.093 | 0.308 | -1.78 |
| s0, 0 | D | 105,078 | 0.137 | 0.209 | 0.092 | 0.299 | -1.79 |
| s0, 1 | D | 104,953 | 0.136 | 0.208 | 0.092 | | -1.79 |
| s0, 2 | D | 104,926 | 0.136 | 0.209 | 0.092 | | -1.79 |
| s1, 1 | transfer union (4l) | 102,248 | 0.138 | 0.215 | 0.093 | 0.321 | -1.77 |
| s1, 1 | D | 104,368 | 0.136 | 0.208 | 0.092 | | -1.78 |
| | DIA-NN 2.2.0 | 118,326 | 0.143 | 0.207 | 0.107 | 0.282 | -1.77 |

Shared with DIA-NN 2.2.0 (eps / CV / E. coli log2):

| ID set, rescore seed | arm | shared: MuMDIA | DIA-NN on the same ions | MuMDIA-only |
|---|---|---|---|---|
| s0, 0 | transfer union | 93,277: 0.133 / 0.096 / -1.79 | 0.127-0.128 / 0.104 / -1.80 | 9,495: 0.223 / 0.138 / -1.06 |
| s0, 0 | B | 94,629: 0.132 / 0.094 / -1.76 | 0.130 / 0.105 / -1.79 | 10,872: 0.225 / 0.133 / -0.88 |
| s0, 0 | C | 95,147: 0.131 / 0.095 / -1.80 | 0.130 / 0.105 / -1.79 | 10,680: 0.225 / 0.136 / -1.12 |
| s0, 0 | D | 94,776: 0.131 / 0.094 / -1.80 | 0.130 / 0.105 / -1.79 | 10,302: 0.220 / 0.133 / -1.22 |
| s0, 1 | D | 94,705: 0.131 / 0.094 / -1.80 | 0.130 / 0.105 / -1.79 | 10,248: 0.219 / 0.131 / -1.21 |
| s0, 2 | D | 94,713: 0.131 / 0.093 / -1.80 | 0.130 / 0.105 / -1.79 | 10,213: 0.220 / 0.130 / -1.21 |
| s1, 1 | transfer union | 92,825: 0.133 / 0.095 / -1.79 | 0.127-0.128 / 0.104 / -1.80 | 9,423: 0.224 / 0.139 / -1.02 |
| s1, 1 | D | 94,286: 0.131 / 0.094 / -1.80 | 0.130 / 0.105 / -1.79 | 10,082: 0.219 / 0.132 / -1.20 |

Per-run rows against pass 1, and test 4 (decoy fraction above the threshold relative to accepted targets, anchor
runs / non-anchor runs). The test 4 counts include the unreported library targets with pass-1 `precursor_q` in
(0.01, 0.05].

| ID set, rescore seed | arm | pass 2 accepted | new | dropped, in library | dropped, outside | net per run | test 4 |
|---|---|---|---|---|---|---|---|
| s0, 0 | B | 618,821 | 104,218 | 13,308 | 15,171 | | 0.76% / 1.80% |
| s0, 0 | C | 617,038 | 98,135 | 9,008 | 15,171 | +11,607 to +13,182 | 0.77% / 1.81% |
| s0, 0 | D | 609,416 | 91,781 | 10,266 | 15,181 | +10,314 to +11,951 | 0.77% / 1.89% |
| s0, 1 | D | 608,759 | 91,125 | 10,267 | 15,181 | +10,169 to +11,848 | 0.78% / 1.88% |
| s0, 2 | D | 608,499 | 91,044 | 10,446 | 15,181 | +10,036 to +11,867 | 0.79% / 1.84% |
| s1, 1 | D | 605,736 | 90,600 | 10,365 | 15,361 | +10,090 to +11,753 | 0.77% / 1.89% |

Eq split (s0, seed 0; ions at `min_obs` 3 with and without a pass-2 new row):

| arm | pass-1 rows only: ions / eq / E. coli log2 | with new rows: ions / eq / E. coli abs eps / E. coli log2 |
|---|---|---|
| B | 62,582 / 0.143 / -1.87 | 42,919 / 0.361 / 0.581 / -1.49 |
| C | 63,404 / 0.143 / -1.86 | 42,423 / 0.319 / 0.471 / -1.60 |
| D | 64,120 / 0.144 / -1.86 | 40,958 / 0.314 / 0.460 / -1.62 |

Reading:
- The report filter (B) changes little: eq -0.004 against SP-A 0.01, at 300 fewer ions.
- Predicted intensities (C) carry the gain: eq -0.019 against B. The empirical patterns of 1- and 2-run precursors
  (SP8 R4) were the cause of the SP-A quant failure.
- The new decoys (D) add eq -0.003 and global -0.001 at about 750 fewer ions. The MuMDIA-only E. coli log2 moves from
  -1.12 to -1.22, which fits fewer false rows.
- D against the transfer union, on both ID sets:
  - s0 (3 seeds): +2,154 to +2,306 ions, of which +1,428 to +1,499 are DIA-NN-shared; global -0.002 to -0.003; eq
    -0.007 to -0.008; PB CV -0.001.
  - s1: +2,120 ions (+1,461 shared), global -0.002, eq -0.007, PB CV -0.001.
  - Every run has 10,000-11,900 more accepted rows than pass 1.
  - D passes the sensitivity and accuracy gates of SP7 on two ID sets.
  - Against DIA-NN 2.2.0: eq 0.208-0.209 against 0.207, global 0.136-0.137 against 0.143, PB CV 0.092 against 0.107,
    ions -11%.
- Test 4 is unchanged in D (1.84-1.89% in non-anchor runs against 0.77-0.79% in anchor runs). This is the open FDR
  point, and SP7 entrapment is the gate before engine work.
- The +2,100 to +2,300 ions are about 2% over the union. That is above the SP8 R5 floor of about 1,000 ions, and it
  comes with an accuracy gain.

**Entrapment, arm D (2026-10-04).** Setup: the single-run E. coli diaPASEF entrapment of TIMS_ROADMAP_bis section 8
(`LFQ_Ultra2_diaPASEF_15min_50ng_Ecoli_01.d` against E. coli + 1:1 human, `entrapment_ratio` 0.563430, peptide level).
- Pass 1 is the existing `retrace.repick` full run (`/public/local/Robbe/MuMDIA/MuMDIA_repick/full/entrap`, binary
  `~/bin/mumdia-repick2/mumdia`).
- The second pass is arm D, built on pass-1 seed 0 and run with binary `~/bin/mumdia-rx/mumdia`, then rescored with
  seeds 0-2.
  - Driver `quant_diag/sp_entrap.sh`; test 3 in `quant_diag/sp_entrap_t3.py`.
  - Outputs in `/public/local/Robbe/MuMDIA/MuMDIA_sp_entrap/`.
  - Library: 19,847 targets at `precursor_q` <= 0.05, of which 656 are entrapment. New decoys: 19,835 reversed, 12
    scrambled.
  - One run has no held-out residual, so the RT half-width is fixed at 8.8 s. That is the HYE p99, at the same
    0.968 s cycle. The 1/K0 half-width is 0.04.
- The whole second pass takes under 2 minutes on this run.

| | pass 1, seeds 0 / 1 / 2 | pass 2 (D), rescore seeds 0 / 1 / 2 |
|---|---|---|
| real peptides at 1% | 13,906 / 13,948 / 14,071 | 13,852 / 13,850 / 13,831 |
| spike-in peptides | 115 / 89 / 106 | 118 / 118 / 119 |
| empirical FDP | 0.473% / 0.367% / 0.432% | 0.487% / 0.487% / 0.492% |
| PSM decoy fraction | 0.99% | 1.11% |

The pass-2 decoy fraction is above 1% because the report filter sets the unreported targets to q 1.0 after q is
computed, while every decoy stays counted.

SP3 test 3 (pass-2 scores of the library's entrapment precursors, all false):

| library targets | in library | accepted at q <= 0.01 | AUC against decoys | beats its own decoy |
|---|---|---|---|---|
| entrapment, all | 656 | 534-536 | 0.977-0.978 | 98.2-98.5% |
| entrapment, reportable (pass-1 `precursor_q` <= 0.01) | 125 | 113-114 | 0.985-0.986 | 98.2-99.1% |
| real, reportable | 17,267 | 17,128-17,137 | 0.999 | 99.9% |

Reading:
- **Pass-2 target-decoy competition cannot reject a false library entry.** Pass 1 matched the false precursor to a
  real signal of another peptide, and the library takes its RT and 1/K0 from that match. Pass 2 finds the same signal
  again, while the decoy, with other fragment m/z, does not.
- This is a property of the two-pass design, not of this implementation. DIA-NN's MBR has the same structure. That
  is why DIA-NN reports `Lib.Q.Value` (the first-pass global q) and why its ProteoBench rows all pass
  `Lib.Q.Value` <= 0.01 as well as `Q.Value` <= 0.01.
- **The FDR of a second-pass report is set by the library q filter, not by the pass-2 q.** D reports only
  precursors with pass-1 `precursor_q` <= 0.01. The FDP stays near pass 1's because about 113 of the 125 reportable
  false entries come through again.
  - Against its own source library (pass-1 seed 0): 0.473% to 0.487%, below 1% but not at or below the pass-1
    range.
  - Real peptides: -0.4% against pass-1 seed 0. One run gains nothing from a second pass.
- **The SP-A arms at 0.02 and 0.05 above report every library precursor.** Their FDR is near their library cut, so
  their extra ions are not FDR-valid. B, C and D apply the library q filter.
- **Open, and tabled on 2026-10-04:**
  - A false precursor that enters the library can come back in every run where the same interfering signal elutes.
    A single run cannot measure this (SP8 R1, decision a).
  - What would measure it: the multi-run entrapment (six HYE runs against a library with a foreign proteome),
    through D and through DIA-NN `--reanalyse`.
  - A DIA-NN `--reanalyse` run on this single entrapment file would give DIA-NN's own MBR FDP for comparison. The
    existing DIA-NN entrapment run (`MuMDIA_diann_entrap`) is without MBR, at an FDP of 0.41%.
  - Test 4 (decoy fraction 1.84-1.89% in non-anchor runs) is also unresolved.

## 5. Plan

Ordered by expected gain per unit of work. Each phase states its target and its gate. Quant
levers are prototyped as requants and offline Python on `chromatograms.parquet` first. A lever
goes into the engine only when it holds on HYE diaPASEF over at least two ID sets. Gains of about
1% that need new engine code are skipped.

### Q1: Bounded integration window (measured, section 4)

Result: `fixed_scan_halfwidth: 4` (equal to `fixed_window_s: 4`) + `baseline_subtract`, flank 12;
the quantile is set together with Q2 (0.5 or 0.6). Holds on three ID sets.


- Working base: `fixed_window_s: 4` + `baseline_subtract` (global 0.138, eq 0.217).
- Confirm it on more HYE diaPASEF identifications than seed 0 of `eng_repick` (for example
  seeds 1 and 2 of the same pipeline). Quant is deterministic given the IDs, but the IDs are
  not.
- Sweep `baseline_flank_scans` and `baseline_quantile` on the fixed window. The defaults (12,
  0.25) were never tuned on diaPASEF.
- Decide the unit. `fixed_window_s` integrates a variable number of samples depending on where
  the apex falls on the grid (docs/12). On diaPASEF the cycle is 0.968 s, so 4 s covers 8 or 9
  samples. Compare with `fixed_scan_halfwidth: 4` (always 9 samples). Prefer the scan form if
  the two are equal, because it generalises across cycle times.
- An alternative with no new code path: cap the descent walk at a width learned from confident
  peptides, so wide windows are cut and narrow ones stay. Try it only if a fixed window turns
  out worse when generalising.
- Gate: global and species-equalised epsilon on HYE diaPASEF, over at least two ID sets. IDs
  are unchanged by construction. Then make it the diaPASEF quant default.

### Q2: Fragment choice by interference, consistent across runs

Result (section 4b): the `wsum` prototype meets the target on three ID sets. Next: an engine
implementation as a cross-run step on the per-fragment areas, behind a config key.

Target: species-equalised epsilon below the Q1 base (0.217) and global epsilon below 0.129
(the top-6 value), with no loss of ions. The CV of top-6 (0.080) would be a bonus.

- Per precursor, over every run in which it is quantified, take the per-fragment areas
  in the fixed window. A fragment that is clean has the same share of the precursor's total in
  every run. An interfered fragment has an inflated share in the runs where it is interfered,
  and most in the low-abundance condition.
- Prototype offline: for each of the 12 fragments, compute its log-share deviation from the
  cross-run median, and its apex correlation with the other fragments in each run (retrace
  traces). Choose the fragments with low deviation and high correlation, up to 6, as one set
  per precursor, used in every run. Compare with a weighted sum (weight inversely proportional
  to the deviation) over all 12.
- This uses no condition labels, so it is not tuned to the benchmark design. It does use
  every run of the experiment, so it is a cross-run step: run it after per-run quant, on the
  per-fragment areas (quant already supports the fragment export, `quant.rs:531`).
- Gate: shared-ion and all-ion global and species-equalised epsilon on HYE diaPASEF, then CV.
  Missingness must not rise. Report the number of fragments used per precursor.

### Q3: Additive background on the raw events

Result (section 4e): measured and closed. Neither a narrower band nor a side-band background gains.

Target: remove the residual E. coli floor (-1.71 against -1.85 on all ions with
`baseline_subtract`).

- The RT-flank baseline (`baseline_subtract`) helps the ratio and costs CV. It estimates the
  background from neighbouring scans, which on a crowded gradient also contain other peptides.
- On diaPASEF the background can be estimated in the mobility dimension instead: the same
  fragment m/z and the same frames, in a 1/K0 band next to the precursor's band. Retrace
  already reads those events. Prototype this offline before touching retrace.
- Also try a narrower quant band than retrace's `apex_im +/- 0.015`. It is tuned for
  identification. A narrower band removes more co-mobile interference but also cuts into the
  ion's own signal, so measure the trade-off against CV.
- Gate: as Q2. This lever is diaPASEF-specific by nature, which fits the current scope.

### Q4: The faint and MuMDIA-only ions

Result (section 4d): measured and closed. Not wrong peaks; faint and partial ions, like DIA-NN's.

Target: the 15,235 extra ions, now at |epsilon| 0.194 and E. coli -1.44 under Q1.

- First measure, then choose. The question is whether their error is a weak true signal (a
  floor problem: Q1 to Q3 apply to them, and they already gain the most from Q1) or a wrong
  peak in the low-abundance runs. Check with LOESS-aligned apex RT across the six runs, and with
  the per-run `run_psm_q` of each ion in condition A against condition B.
- If wrong peaks dominate: in a run where the ion is weak, integrate at the RT predicted from
  the runs where it is strong (aligned consensus apex). The ion is identified in that run
  anyway, so this changes the quantity, not the identification. MBR stays out of scope.
- Add a per-ion quantity quality value (like DIA-NN's `Quantity.Quality`) from the Q2
  consistency statistic. Report it. Do not filter the ProteoBench submission on it.
- A residual E. coli ratio near -1 can also mean that some extra ions are false. Entrapment on
  the E. coli file covers that question. It is not a quant lever, but it bounds what Q4 can
  recover.

### Q5: MS1 in the quantity

- Retrace writes raw MS1 traces (`ms1_mono`, `iso1`, `iso2`) gated in 1/K0. Quant excludes them
  today. Score MS1 alone as the quantity, then a combination with the fragment quantity (for
  example the fragment estimate, with MS1 used only when the fragments disagree across runs).
- The expected gain is on the faint ions, where few fragments are clean. Lower priority than Q2,
  because Q0.6 does not separate DIA-NN's MS1 and fragment contributions.

### Q6: Run normalisation (small, cheap)

- Q0.5 bounds the gain at about 0.002 in epsilon and 0.004 in CV. Apply median-ratio
  normalisation over the ions shared by all runs, with no species information, in the
  submission step or in quant-lfq. This is done last so that it does not hide Q1 to Q4 effects.

### Out of scope here

- MBR and transfer requant were out of scope while the target was MBR off. Since 2026-10-01 the
  MBR-on target is in scope (sections 2b and 4i).
- A second-pass MBR strategy (empirical library, second search) is planned in section 4m, not
  measured.
- Protein-level quant and MaxLFQ (ProteoBench scores precursor ions).
- Anything that changes the identification population. If a quant lever needs a rescore
  change, it moves to the identification roadmap with its gates.

## 6. Harness

- `quant_diag/requant_repick.sh <name> '<python edit of q>'`: six per-run quants on the
  `eng_repick` IDs into `q_rp_<name>`, then `score.sh`. About 3 min per arm.
- `quant_diag/shared_ions.py`: shared and unique ion split against the DIA-NN intermediate.
- `quant_diag/per_run_join.py`: run-level join with DIA-NN (apex RT, widths, per-condition
  intensity ratios).
- Every result is recorded with the binary, the quant config hash (in
  `peptide_quant.parquet.report.json`) and both the all-ion and shared-ion numbers.

## 7. Decisions (2026-10-01)

- Dataset: HYE diaPASEF only for now. Other known-ratio sets come in when the levers are
  generalised.
- Scope: diaPASEF defaults for now. Keep levers as generic config keys, because they may be
  generalised later.
- Targets: global and species-equalised median |epsilon| first, CV second.
