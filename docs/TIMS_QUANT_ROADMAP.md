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
| MuMDIA `crwbg` + MBR | 100,340 | 0.143 | 0.220 | **0.098** | **0.094** | 0.325 / 0.128 / 0.207 | -1.77 |

On the ions shared with each target (`quant_diag/shared.py`, now reading `result_performance.csv`;
`DIANN=250` or `220`), eps / CV / E. coli log2:

| arm | shared | MuMDIA | DIA-NN on the same ions | MuMDIA-only |
|---|---|---|---|---|
| `crwbg` s0 against 2.5.0 | 81,778 | 0.127 / 0.088 / -1.84 | 0.115 / 0.088 / -1.80 | 10,762: 0.216 / 0.126 / -1.64 |
| `crwbg` s1 against 2.5.0 | 81,506 | 0.127 / 0.088 / -1.84 | 0.115 / 0.088 / -1.80 | 10,635: 0.213 / 0.126 / -1.57 |
| `crwbg` s2 against 2.5.0 | 81,755 | 0.127 / 0.088 / -1.84 | 0.115 / 0.088 / -1.80 | 10,720: 0.217 / 0.127 / -1.61 |
| `crwbg` + MBR s0 against 2.2.0 | 91,379 | 0.138 / 0.095 / -1.79 | 0.127 / 0.104 / -1.80 | 8,961: 0.237 / 0.141 / -1.09 |

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
