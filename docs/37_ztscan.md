# 37. SCIEX ZT Scan DIA: scanning-quadrupole input
> Part of the MuMDIA developer documentation (see docs/README.md).

This guide describes how MuMDIA reads, searches and quantifies SCIEX ZT Scan DIA runs, in
which the quadrupole sweeps the precursor range instead of stepping through fixed windows.
Section 1 describes the acquisition, section 2 how a run is read and recognised before a
search, section 3 the opt-in model of the sweep layer by layer, section 4 how to run it,
section 5 what was measured, section 6 the quantification window in Q1 cycles, section 7 the
gate on the fragments that belong that was measured and not added, section 8 where time and
memory go, section 9 what is refused, section 10 the unfragmented precursor, multi-scale Q1
scores and Q1-deconvolved evidence, and section 11 what is open. Reading a ZT Scan file
needs one key (`convert.isolation_half_width`); everything that models the sweep is off by
default, and `configs/examples/ztscan.json` is the recipe that turns on what measured
positive. The mechanisms each have a canonical section elsewhere (docs/04, 08, 09, 10, 11 and
12), which this guide routes to rather than restates.

Every measurement here is on one acquisition: three K562 replicates on a ZenoTOF 8600 (100
ng, 15 min gradient, Q1 window 3.8 m/z), with an imported spectral library restricted to the
scanned 392-900 m/z and predicted decoys, and the same library extended with yeast and E.
coli entries for entrapment (r = 0.397, combined FDP = N_E (1 + 1/r) / (N_H + N_E)). That is
why no default changes: CLAUDE.md asks for a second acquisition, and the ZenoTOF ZT Scan
files of the benchmark collection are vendor files whose conversion with isolation targets
is a licence decision (section 11).

## 1. What a ZT Scan run is

A ZT Scan DIA method sweeps Q1 across the precursor range once a cycle while the TOF records
a spectrum every few milliseconds. On the K562 runs a cycle is 1 MS1 and 663 MS2 spectra
("steps" or "bins") whose isolation targets run from 394.256 to 898.271 m/z, 0.7614 m/z
apart, identical every cycle; a step is 1.345 ms and a cycle 1.00 s. The 3.8 m/z window
transmits a precursor through about five consecutive steps of every cycle, and a peptide
elutes over 3-6 cycles of a 15 min gradient. A run is 54 min, 2.15 million spectra and 3.34
billion MS2 centroids (42 GB of mzML).

Two properties set it apart from fixed-window DIA. Adjacent windows overlap almost entirely,
so a fragment m/z shared by two precursors 2 m/z apart shows up in partly the same steps,
but centred where each precursor is: where in the sweep a fragment's signal sits says which
precursor it came from. And a candidate's trace interleaves its five steps a cycle, 1.3 ms
apart, then a gap of a second, which every rule that counts samples reads wrongly unless it
knows the cycles (section 6).

## 2. Reading it, and knowing before a run

The SCIEX MS Data Converter writes each step as an MS2 spectrum that names its isolation
target and no window offsets. `convert` refuses such a file unless
`convert.isolation_half_width` gives the window's half-width: read as before, every
spectrum covered the whole m/z range, the run became one isolation window and extract grew
past 500 GB (docs/04_convert.md, "SCIEX ZT Scan DIA", has the converters, the refusal rule
and the measurements). With the key each step is a window `[target - h, target + h]` and
convert records the sweep (spectra_ms2 v3: `q1_cycle`, `q1_step`). The half-width the file
does not record has to come from the method: 1.9 m/z for the 3.8 m/z window, and 1.5, 1.9
and 2.3 identified the same within one seed's noise.

`mumdia peak-census` tells before a run whether a file needs the key. For an mzML its JSON
carries an `isolation` block, from the spectra it reads (`--max-spectra 10000`, the sample
`convert` decides its refusal on: 15 cycles of the K562 method, about 2 s on a 42 GB file,
most of it reading the file's index), computed with the rule `convert` refuses by
(`stages::convert::IsolationCensus`):

```json
"isolation": {
  "ms2_spectra": 9984, "with_offsets": 0, "no_isolation": 0, "target_only": 9984,
  "convert_refuses": true,
  "scanning_quadrupole": true,
  "layout": {"distinct_targets": 663, "lowest_target": 394.2558, "highest_target": 898.2707,
             "median_step": 0.7614, "spectra_per_target": 15.06, "cycle_s": 1.002},
  "suggested_half_width": 1.9,
  "suggested_half_width_steps": 2.5
}
```

`convert_refuses` is what `convert` does without the key: MS2 spectra that name differing
targets and no offsets. `scanning_quadrupole` is the layout the refusal names (at least 50
distinct targets repeating cycle after cycle in steps under 1.5 m/z). The suggestion is
`SCANNING_HALF_WIDTH_STEPS` (2.5) steps, a window five steps wide, rounded to 0.05 m/z: the
measured method's 3.8 m/z window over 0.7614 m/z steps. It is a starting point, not a
measurement of the file: the Q1 calibration (section 3) fits an effective window of about
5.1 m/z with soft edges on the same runs, and a method with a different window per step
needs its own half-width. A target-only file that is not a scanning quadrupole (fixed
windows whose converter dropped the offsets) is refused with no suggestion: converting the
vendor file again with offsets is the better fix.

The desktop application asks the same census of every selected mzML when it is added
(`scanning_probe`, of the first 10,000 spectra of each, four files at a time, vendor files not
probed because the census would convert them first). When one of them would be refused it
shows a "SCIEX ZT Scan DIA" card on the Search screen: what the file is (the positions, step
and cycle from the census), why it
cannot be read without the half-width, an input filled with the suggestion (a value the user
types is kept), and a checkbox for the ZT Scan recipe on top of the chosen preset, enabled only
when every selected file is a scanning-quadrupole run, because the recipe's keys refuse
spectra without a sweep. A search with such a file and no half-width is not started; the
recipe is refused with bands, which `rt_im_train.q1_calibrate` refuses.

## 3. The model of the sweep, layer by layer

Each layer is one opt-in key, off by default and omitted from the canonical configuration, so
every other search is byte-identical. Measured on REP01 with four NN seeds per arm (section
5); "in the recipe" is `configs/examples/ztscan.json`.

| layer | key | canonical section | measured | in the recipe |
|---|---|---|---|---|
| read each step as a window, record the sweep | `convert.isolation_half_width` | docs/04, "SCIEX ZT Scan DIA" | required to read the file at all | yes (1.9) |
| sum a cycle's steps, a Q1-profile family, a gate on it | `extract.q1_cycles`, `extract.q1_profile`, `extract.q1_gate_max_offset`, `features.q1_features` | docs/09, section 10 | summing -13.7%; the profile family +0.4% inside seed noise; the gate cuts targets and decoys alike | no |
| calibrate the Q1 window from the seed | `rt_im_train.q1_calibrate` | docs/08, section 4h | the target is the window centre; an effective window wider and softer than nominal; changes no identification by itself | yes (the decisions need it) |
| MS1 error surface and an MS1 tolerance from the seed | `extract.ms1_calibrate = "surface"`, `extract.prec_tol_auto` | docs/09, section 11 | +0.46% precursors, inside seed noise | no |
| decide per fragment whether it belongs | `extract.q1_fragments` | docs/09, section 12 | with its features: +1.26% human precursors at the base arm's measured FDP | yes |
| the decisions as features | `features.q1_fragment_features` | docs/10 | (with the decisions, above) | yes |
| re-pick the apex in RT x Q1 | `retrace.q1_repick` | docs/09, section 12 | +1.64 points on top, the largest single lever | yes |
| bound the peaks the re-pick compares | `retrace.q1_repick_peaks` | section 8 below | 2: extract -28% for -0.2% / -0.5% (entrapment at matched FDP / human library); 0: -37% for -1.2% / -1.1% | no (default 5) |
| the Q1 term of shadow demotion | `rescore.shadow_q1` | docs/11, "shadow demotion and its Q1 term" | counts within noise; lowers the entrapment precursors at 1% (the measured FDP at a reported 1%: 1.29% to 1.22%) | yes |
| quantify from the fragments that belong | `quant.q1_fragments` | docs/12, "Quantification from the fragments that belong" | replicate CV 0.120 against 0.273; restricting to belonging fragments costs precision against the collapse alone; the known-ratio gate is open | no |
| the quantification window in Q1 cycles | `quant.q1_cycle_halfwidth` | section 6 below | replicate CV 0.102 at 4 cycles against 0.273, the identifications unchanged | yes (4) |
| an MS1 isotope-envelope family | `features.ms1_envelope` | docs/10 | within noise (+0.01% at matched FDP) | no |
| the unfragmented precursor's Q1 profile in MS2 | `features.q1_precursor` | docs/09, section 14 | +0.13% precursors, +0.65% human precursors at matched FDP, inside the seed spread; in the three-family set it carries 0.16% | yes |
| multi-scale Q1 scores (3, 7, 11 steps) | `features.q1_multiscale` | docs/09, section 14 | +0.32% / +0.09%, inside the seed spread; nothing on top of the other two | yes |
| Q1-deconvolved evidence as features | `features.q1_deconvolved` | docs/09, section 14 | +0.51% precursors, +0.64% at matched FDP, just beyond the seed spread | yes |
| the three together | (the three keys) | section 10 below | +0.76% precursors, +1.38% human precursors at matched FDP, beyond the seed spread on both libraries | |
| count only Q1-consistent peaks in the seed, in extract | `search_seed.q1_deconvolved`, `extract.q1_deconvolved` | docs/07, docs/09 section 14 | -2.5% (seed), -25.6% (extract), -38% (both) | no |
| score each fragment's Q1 centre against its expected one | `features.q1_consistency` | docs/10, "q1_consistency" | on top of the four: +2.97% human precursors at the measured FDP and +2.48% precursors on the human library (the family alone, its columns kept out of the rescore prefilter: +2.28% and +1.85%) | yes |

The four decision keys of the recipe are the phase-2 set, measured together: +3.0%
precursors on the human library and +3.2% human precursors on the entrapment library at the
base arm's measured FDP, whose value at a reported precursor q of 1% falls from 1.39% to
1.22%. The three feature families of section 10 add +0.76% and +1.38% on top of them, with
the measured FDP unchanged, and `features.q1_consistency`, measured on top of the four
(section 5, "Features for the classifier"), adds +2.48% precursors and +2.97% human precursors
at the measured FDP. Sections 5 to 8 were measured with the recipe before those four families
were added; sections 5 and 10 have their numbers, and section 12 the recipe as released, with
the library of section 11.

## 4. How to run it

Check the environment against the recipe first. It is the imported-library recipe
(`configs/examples/diann-library.json`: the `nn_torch` classifier, DeepLC for the library's
retention times, both interpreters `"auto"`) plus the keys of section 3:

```text
mumdia doctor --config configs/examples/ztscan.json
```

One run, and the three replicates as one experiment (pooled rescore, second pass, per-run and
cross-run quantification):

```text
mumdia run --lib-precursors lib/lib_precursors.parquet --lib-fragments lib/lib_fragments.parquet \
  --mzml REP01.mzML --out-dir out_rep01 --config configs/examples/ztscan.json

mumdia run --lib-precursors lib/lib_precursors.parquet --lib-fragments lib/lib_fragments.parquet \
  --mzml REP01.mzML --mzml REP02.mzML --mzml REP03.mzML --out-dir out_k562 \
  --config configs/examples/ztscan.json
```

- Change `convert.isolation_half_width` to half of your method's Q1 window (`mumdia
  peak-census` suggests one, section 2).
- Restrict the library to the precursor range the method scanned (here 392-900 m/z); the
  sweep covers nothing outside it.
- Uncapped peaks (the default `--top-peaks-ms2 0`): ZT Scan spectra hold about 800 peaks
  each (p25 545, p95 1,078 at the start of the run).
- Expect about 20 min and 40 GB for one run with the human library on 32 threads, and about
  51 min and 78 GB for the three runs as one experiment, the peak from the concurrent
  second-pass extracts (sections 8 and 10.7; the recipe before the feature families of section
  10 took 18.5 min and 35 GB on the same host).
- The recipe is refused with `groups.window_groups > 1` (the Q1 calibration needs the whole
  sweep), so a machine too small for one run cannot band it; the base reading
  (`isolation_half_width` alone) can be banded.

## 5. Measured results

REP01 on hippogriff (human library) and dementor (entrapment library), each arm compared on its
own host with one binary; seeds 1-3 rescore the run's competed table with `MUMDIA_NN_SEED`.
Precursors at `precursor_q` 1%; matched FDP is the human precursors where the arm's combined
entrapment FDP reaches the base arm's mean at a reported 1% (1.386%), interpolated along the
`precursor_q` thresholds. Decoy fraction 0.0100 in every arm.

### Identification

| arm | precursors at 1%, mean of seeds 0-3 (range) | human precursors at matched FDP, mean (range) | combined FDP at a reported 1%, mean |
|---|---|---|---|
| base (`isolation_half_width` only) | 97,317 (96,873-97,541) | 95,135 (94,877-95,342) | 1.39% |
| + calibration (`q1_calibrate`, MS1 surface, `prec_tol_auto`) | 97,765 (97,509-97,951) | (95,295 at 1.40%) | 1.40% |
| + decisions and features | 98,728 (98,352-99,086) | 96,337 (95,885-96,496) | 1.31% |
| + decisions, features, re-pick | 100,194 (100,067-100,329) | 97,915 (97,705-98,222) | 1.29% |
| + decisions, features, shadow Q1 | 98,705 (98,356-99,046) | 96,624 (96,037-96,946) | 1.29% |
| the four (the recipe's identification keys) | 100,257 (100,132-100,520) | 98,155 (97,933-98,392) | 1.22% |
| the four + MS1 envelope family | | +0.01% against the four | 1.25% |

Peptides (mean): base 78,627, the four 80,726 (+2.7%); protein groups 7,721 and 7,750.

The recipe end to end (the shipped file, `"auto"` interpreters resolved through
`MUMDIA_PYTHON_DEEPLC` / `MUMDIA_PYTHON_RESCORE`), against the same arm run with the phase-2
and phase-3 binaries:

| library (host, threads) | precursors at 1%, seeds 0 / 1 / 2 / 3 (mean) | peptides / protein groups, seed 0 | human precursors at the matched FDP (1.224%), mean (range) | combined FDP at a reported 1% | wall, peak |
|---|---|---|---|---|---|
| human (hippogriff, 40) | 100,034 / 99,674 / 99,845 / 99,661 (99,804) | 80,604 / 7,750 | | | 19 min, 36.1 GB |
| entrapment (dementor, 64) | 97,446 / 97,688 / 97,468 / 97,843 (97,611) | 78,272 / 7,910 | 97,225 (96,992-97,540) | 1.20-1.24% | 28 min, 36.9 GB |

Both write their seed's scored table, and the extracted, chromatogram, feature and competed
tables, byte for byte as the four keys did in phase 3 on the same host and threads. The
human-library mean is 99,804 at 40 threads against 100,257 for the same arm at 64 threads in
the table above: the thread count moves the classifier's arithmetic, which is why every
comparison here is made on one host with one thread count.

### Three runs as one experiment

REP01-03 as one `mumdia run` (pooled rescore, second pass), one NN seed, horntail, 64 threads:

| experiment | precursors | peptides | protein groups | per run on `run_psm_q` | wall, peak |
|---|---|---|---|---|---|
| base (`isolation_half_width` only) | 110,433 | 88,958 | 8,981 | 106,485 / 106,561 / 106,583 | 40 min, 89.4 GB |
| the recipe | 113,013 (+2.3%) | 90,739 (+2.0%) | 9,027 (+0.5%) | 109,028 / 109,185 / 109,093 (+2.4%) | 60 min, 89.2 GB |

The recipe's experiment writes the same scored tables, first and second pass, byte for byte as
the four keys did in phases 2 and 3, and its quantification is the 4-cycle arm of the table
below (median precursor CV 0.102).

### Features for the classifier

**Where the information gap was.** With the four keys on, the precursors another engine
accepted at 1% on REP01 (92,322 on the entrapment library) fall into: accepted here 89%,
scored above 1% here 6.8% (6,301; 83% of them with the apex within 6 s of the other engine's,
so the peak is found and its score is weak), removed by the rescore prefilter 2.1%, outside the
RT window 1.3%, not extracted 0.9%. The scored-above-1% set carries a combined entrapment FDP
of 4.9%, so it is mostly real.
Against decoys and entrapment targets at a matched classifier score (40 bins of its NN score;
the Mann-Whitney AUC per bin, weighted by its rows), every one of the 434 features of the
vector stays between 0.36 and 0.64 (the largest `spectral_log_evidence_area`, 0.64; mass
error, ion series, MS1 co-elution, peak shape and the existing competition columns 0.43-0.61).
What separated them was the Q1 decision read against each fragment's expected centre: 60% of
them have no fragment that belongs, and for 82% of those the centre their fragments share (a fit
over every present fragment, interferers included) lies beyond the centre tolerance of the
expected one, while the share of predicted intensity in fragments within the offset tolerance
of their own expected centre reaches 0.74 against decoys and against entrapment targets alike
(0.50 between decoys and entrapment targets). The human-library run gives the same
ranking (0.74, the largest existing feature again 0.64). Decoys and entrapment targets are
exchangeable at matched score on every Q1 quantity; the one family that is not is the
retention-time position (`observed_rt_raw` and its relatives, 0.53-0.55), and leaving those
five columns out of the classifier changed nothing (below).

**The screen.** Each candidate family computed outside the engine (from psms_extracted, the
chromatograms and the library, label-blind) and appended to the recipe run's competed table,
rescored with two NN seeds on each library (entrapment library on one host, human library on
another, the base rescored the same way); the rescore prefilter (`rescore.prefilter_keep_decoys`,
active because the pool holds more than a million PSMs) reads its 32 fixed columns, so a new
family acts only on the rows it keeps. Human precursors at the base arm's combined FDP (1.205%)
and at 1.0%, interpolated along the `precursor_q` thresholds:

| arm | columns | family | entrapment: human at the base FDP | at FDP 1.0% | human library: precursors at 1% |
|---|---|---|---|---|---|
| base | 0 | the recipe as it is | 97,256 | 95,551 | 99,482 |
| noise10 | 10 | null: ten standard-normal columns | -0.05% | -0.03% | -0.51% |
| fr | 60 | per library fragment slot by predicted rank: prediction, observed share, apex, leave-one-out elution correlation, ppm error | -0.16% | +0.03% | -0.09% |
| frq | 60 | per slot: belongs, present, own Q1 offset, fit residual, 2D correlation | +0.51% | +0.81% | +0.96% |
| cc | 8 | competition context: co-eluting candidates within the Q1 reach that share fragments, the best one's score, Q1 distance | +0.06% | +0.16% | +0.58% |
| cyc | 10 | the elution profile collapsed to one value a cycle: Gaussian fit, width, asymmetry, co-elution, apex spread | -0.42% | -0.01% | +0.15% |
| q1s | 16 | similarity restricted to the fragments that belong and to the Q1-consistent ones (three-cycle sums) | +1.54% | +1.40% | +1.46% |
| q1c | 8 | Q1 consistency against the expected centre: counts, shares, entropy, cosine (three-cycle sums) | +1.15% | +1.46% | +1.44% |
| q1c, peak sums | 8 | the same on the elution-peak sums | +1.45% | +1.39% | +1.65% |
| q1c + graded | 16 | q1c plus graded and soft consistency, the 2D correlation, weighted and top-3 offsets (three-cycle sums) | +2.06% | +2.23% | +1.93% |
| **the family** | 16 | the same on the elution-peak sums: `features.q1_consistency` | **+1.99%** | **+1.93%** | **+1.70%** |
| family + 8 more | 24 | plus cosine within, log intensity, top-1, the belonging-set scores | +1.38% | +1.24% | +2.14% |
| fr + frq + q1s | 136 | | +1.15% | +1.27% | +1.52% |
| q1c + frq | 68 | | +1.83% | +2.05% | +1.81% |
| q1c + cc | 16 | | | | +2.01% |
| without RT position | -5 | `rescore.features` without `observed_rt_raw`, `predicted_rt_raw`, their fractions, `predicted_rt_in_gradient` | -0.00% | +0.10% | -0.14% |

The combined FDP at a reported 1% stayed 1.14-1.29% in every arm. What carries the gain is the
consistency of each fragment with its own expected Q1 centre; the fragment-resolved Q1 columns
let the network learn part of it (+0.5-1.0%), the explicit summaries more. The chromatogram
slots, the competition context and the cycle-collapsed shape add nothing the vector does not
have, and ten noise columns cost up to half a percent, which is why the family is the 16
columns and not the 24.

**Validated in the engine** (the recipe end to end with and without `features.q1_consistency`,
one binary, four NN seeds, seeds 1-3 rescoring each run's competed table; entrapment library on
ashwinder, human library on niffler, 64 threads each; the base reproduces the phase-4 recipe arm
of the human library byte for byte, every table to the scored one):

| library | arm | precursors at 1%, mean (range) | peptides / protein groups, mean | human precursors at the base arm's FDP (1.277%), mean (range) | at FDP 1.0% | combined FDP at a reported 1%, mean (range) |
|---|---|---|---|---|---|---|
| entrapment | the recipe | 97,857 (97,784-97,923) | 78,512 / 7,901 | 97,493 (97,247-97,809) | 95,285 (95,127-95,490) | 1.28% (1.25-1.31%) |
| entrapment | + `q1_consistency`, the family alone | 99,464 (99,351-99,629), +1.64% | 79,827 / 7,956 | **99,718 (99,341-99,999), +2.28%** | **97,404 (97,266-97,587), +2.22%** | 1.20% (1.18-1.23%) |
| entrapment | + `q1_consistency` with its prefilter columns | 100,524 (100,242-100,741), +2.73% | 80,567 / 7,972 | **100,394 (100,122-100,681), +2.97%** | **98,069 (97,880-98,213), +2.92%** | 1.25% (1.20-1.31%) |
| human | the recipe | 99,605 (99,331-99,817) | 80,153 / 7,737 | | | |
| human | + `q1_consistency`, the family alone | **101,446 (101,222-101,602), +1.85%** | 81,812 / 7,788 | | | |
| human | + `q1_consistency` with its prefilter columns | **102,077 (101,891-102,191), +2.48%** | 82,127 / 7,810 | | | |

"With its prefilter columns" is the shipped behaviour: with the family's columns in the
competed table, four of them (`q1c_pred_share`, `q1c_entropy`, `q1c_soft_pred`,
`q1c_offset_wmean`) join the rescore prefilter's 32 default columns (docs/11, "the prefilter
and feature families appended to the vector"); the family-alone rows are the same competed
tables rescored with the prefilter's default columns named explicitly. Beyond the seed ranges
on both libraries; the measured FDP at a reported 1% is 1.18-1.23% (the family alone) and
1.20-1.31% (with its prefilter columns) against 1.25-1.31%. The
extracted and chromatogram tables are the same with and without the family (it is computed in
the features stage); the features pass takes 83 s against 80 s and the rescore peak is 0.5 GB
higher. The recipe with the family's columns rescored with them projected out (and the
prefilter's default columns) writes the scored table of the run made without the family byte
for byte, which is how the same-host bases of the combined arms below were taken.

The three runs as one experiment (REP01-03, second pass, one NN seed, niffler, 64 threads, the
family alone): 113,013 to 115,152 precursors (+1.9%), 90,739 to 92,396 peptides (+1.8%), 9,027 to
9,116 protein groups (+1.0%), per run on `run_psm_q` 109,028 / 109,185 / 109,093 to 111,483 /
111,482 / 111,529 (+2.2%), first pass 101,104 / 101,212 / 100,449 to 102,945 / 103,101 / 102,613;
80 and 78 min on a shared host, 89 GB both. The base writes the phase-4 recipe experiment's
scored table byte for byte.

### Quantification

The three replicates as one experiment, quantified again with `mumdia quant` on the
experiment's scored tables in run-experiment's sequence, so the identifications are those of
the experiment byte for byte (113,013 precursors with the four keys). CV = sample SD over mean
of a precursor's three normalised quantities, over the precursors quantified in all three
runs; protein CV from the cross-run MaxLFQ. The common set is the 95,605 precursors and 8,445
proteins quantified in all three runs in every arm.

| identifications, quantification | median precursor CV | CV under 20% | median protein CV | common-set precursor CV |
|---|---|---|---|---|
| base identifications, default recipe | 0.319 | 25.3% | 0.133 | 0.317 |
| the four keys, default recipe | 0.273 | 34.6% | 0.111 | 0.269 |
| + `quant.q1_fragments` (collapse and restriction, 3 cycles) | 0.120 | 73.4% | 0.054 | 0.116 |
| + `quant.q1_fragments`, collapse only (`q1_min_belonging` 1000) | 0.106 | 82.4% | 0.047 | 0.104 |
| + `q1_cycle_halfwidth` 2 / 3 / 4 / 5 / 6 | 0.116 / 0.107 / 0.102 / 0.101 / 0.094 | 78.9% / 82.6% / 84.4% / 84.9% / 87.7% | 0.051 / 0.047 / 0.045 / 0.044 / 0.041 | 0.113 / 0.104 / 0.100 / 0.099 / 0.092 |
| + `fixed_scan_halfwidth` 20 samples | 0.102 | 84.3% | 0.045 | 0.100 |
| + collapse only, `q1_cycle_halfwidth` 4 | 0.102 | 84.3% | 0.045 | 0.100 |
| + `quant.q1_fragments`, `q1_cycle_halfwidth` 4 | 0.117 | 74.5% | 0.053 | 0.114 |

## 6. The quantification window in Q1 cycles

`quant.q1_cycle_halfwidth` (default 0, off; `stages::quant::q1_cycle_window`,
`q1quant::samples_per_cycle`) sets the fixed window's half-width in Q1 cycles. The fixed
window of the default recipe is `2h + 1` samples around the centre (`fixed_scan_halfwidth` 3),
which on Orbitrap and Astral is about the peak. On ZT Scan the five steps of a cycle are five
samples, so 7 samples cover about 1.4 cycles of a peak 3-6 cycles wide: most of the signal is
left out, and the replicate CV is 0.27. With the key set to `k`, each candidate's half-width is
`round(k x n)` samples, `n` its samples per cycle (the median over its trace's cycles of the
count of its samples in each, every sample located by retention time in the spectra's sweep,
`SweepScans::cycle_at`); under `quant.q1_fragments`, whose traces hold one value a cycle, it
is `k`. Everything that reads the window reads that half-width: the profile centre, the
pattern fit and its export, the background flanks beyond it. It applies only to a run whose
spectra record the sweep (spectra_ms2 v3); on any other run `fixed_scan_halfwidth` applies,
and the quant report says which (`params.q1_cycle_window`: the cycles, whether it applied,
the median samples per cycle and the half-widths chosen). `run` and `run-experiment` hand
quant the spectra; `mumdia quant` takes `--spectra-ms2`. It needs the scan form of the fixed
window and is refused with `fixed_scan_halfwidth_scale`, which also chooses a half-width per
candidate. No column changes.

On the K562 runs every candidate measured five samples a cycle (median; a few at the ends of
the sweep fewer), so 4 cycles are 20 samples: the key reproduces `fixed_scan_halfwidth` 20
to the last digit (0.1024), and on top of `quant.q1_fragments` with `q1_min_belonging` 1000
it reproduces the collapse at 4 cycles (0.1020). What it adds is that the width is stated in
the unit the chromatography has, whatever the window and step of the method.

**Why 4 in the recipe.** The CV falls with the span (table above) and keeps falling past the
peak: 0.116 at 2 cycles, 0.107 at 3, 0.102 at 4, 0.101 at 5, 0.094 at 6. Past the peak the
window integrates the background around it, and a background that is the same in every
replicate lowers the CV of a quantity while it compresses its ratios, which replicate CV
cannot see. Four cycles take the whole peak with about one cycle either side, where the
gain levels off (5 is 1.6% lower than 4); 3 is the more conservative choice. The choice
between this and `quant.q1_fragments` follows the same reasoning: the collapse alone and the
plain wider window measure the same at matched span (0.102), and restricting a candidate to
its fragments that belong costs precision (0.117 at 4 cycles), because replicates of one
sample carry the same interference. Removing interference can only pay off on known
ratios, and no ZT Scan data with known ratios has been converted (section 10). The recipe
therefore takes the simpler key, and `quant.q1_fragments` stays the option to test first
when such data exists.

## 7. A gate on the fragments that belong (measured, not added)

The plan's item 10 was a gate at extract on the number of fragments that belong
(`extract.q1_min_belonging`), to be added only if the statistic cut decoys more than targets.
Among the accepted targets 75.6% have at least three fragments that belong and among the
decoys 0.8%, which is the separation the classifier already reads through
`q1f_n_belong`. Measured on the whole extracted pool of the entrapment run (5,810,005 rows,
the four keys, before any classifier):

| class | rows | share with 0 / 1 / 2 / 3 / 4 / 5+ that belong | mean |
|---|---|---|---|
| decoys | 2,885,333 | 94.55 / 3.67 / 0.74 / 0.66 / 0.28 / 0.10% | 0.088 |
| entrapment targets | 827,230 | 94.54 / 3.65 / 0.74 / 0.65 / 0.30 / 0.12% | 0.089 |
| human targets | 2,097,442 | 90.29 / 4.12 / 1.03 / 1.07 / 0.79 / 2.69% | 0.336 |

The statistic is label-blind (decoys and entrapment targets, both false, have the same
distribution to the fourth digit), and it does not cut decoys more than false targets; what
it separates is real targets from everything else, which is a feature's job. As a gate it
removes real identifications before the classifier can weigh them against the other
evidence. Of the human precursors accepted at 1% (four seeds), 11.5-11.7% have no fragment
that belongs and 25.4-25.9% fewer than three, and they are mostly real: the precursors with
no fragment that belongs measure a combined FDP of about 4.8% (157 entrapment against 11,456
human, seed 0), against 1.3% overall. A gate at `N` therefore loses about 11% (N = 1) to 26%
(N = 3) of the identifications before the classifier sees them. Measured as the gate would
act, the competed table of the entrapment run filtered to the rows with at least `N` fragments
that belong (what extract would have written, as a row's decision depends on no other row)
and rescored with four NN seeds on the same host:

| gate | rows the classifier sees | human precursors at the ungated run's measured FDP (1.224%), mean (range) | vs ungated | combined FDP at a reported 1%, range | rescore |
|---|---|---|---|---|---|
| none (the recipe) | 5,810,005 | 97,225 (96,992-97,540) | | 1.20-1.24% | 155 s (the prefilter keeps 36%) |
| `N` = 1 | 406,171 (7.0%) | 88,120 (87,756-88,550) | -9.4% | 1.00-1.14% | 82-96 s, 1.8 GB |
| `N` = 3 | 134,543 (2.3%) | 73,863 (73,816-73,917) | -24.0% | 1.02-1.10% | 36-38 s, 1.3 GB |

The gate moves the measured FDP at a reported 1% toward 1%, but at a matched FDP it loses a
tenth to a quarter of the identifications. It is a strong speed lever (N = 1 hands the
classifier 7% of the rows, and extract would write as few chromatograms), but not one without
an identification loss, so the key was not added. The presence of a fragment (coherent signal
within the window's reach) separates nothing on ZT Scan: 11.6 of 12 fragments are present in
every class, because the dense spectra put a random match within 20 ppm in a few percent of
the cells.

## 8. Memory and speed

Measured on REP01 with the four identification keys, on hippogriff with 40 threads (a shared
host, so walls are indicative): one run end to end with every artifact kept, then extract
alone on that run's artifacts with and without the keys, and gdb stack samples every 10 s of
a build of the same code with its symbols kept, to see where the time goes.

One run: 1,126 s and a 35.0 GB peak, extract's. Convert 96 s, seed 36 s, the multi-head
retention-time calibration 229 s, the Q1 calibration 65 s, extract 496 s, features 66 s,
rescore 105 s, quant 23 s.

Extract alone (4,185,169 rows and 62,585,269 chromatogram rows in every arm; the busy shares
are of the thread samples that were not waiting):

| Q1 keys | wall, hippogriff | wall, demiguise | busy samples: Q1 patch reads / decisions / probing | bound by |
|---|---|---|---|---|
| none | 315 s | 315 s | 0 / 0 / 44% | the chromatogram writer (busy 223 s, producers blocked 118 s) |
| `extract.q1_fragments` | 324 s | 327 s | 41% / 11% / 18% | the writer (producers blocked 46 s) |
| + `retrace.q1_repick` | 503 s | 515-522 s | 59% / 20% / 7% | the Q1 patch reads |
| + `retrace.q1_repick`, `q1_repick_peaks` 2 | | 373 s | | |
| + `retrace.q1_repick`, `q1_repick_peaks` 0 | 331 s | 330 s | | |

**Where the memory goes.** Extract holds every MS2 peak of the run: 24.9 GB for 3.34 billion
peaks at 8 bytes (m/z and intensity in f32), of its 35.5 GB peak; the per-spectrum records
(retention time, window, the peak list's header) are 0.21 GB, the library 1.0 GB, the MS1
scans 0.4 GB and the resident hit store about 2.3 GB; the chromatograms are streamed. The Q1
decisions read any step of any cycle around a candidate, so the peaks stay in memory.

**Where the time goes.** Without the Q1 keys extract is bound by its single chromatogram
writer, which encodes 62.6 million rows of interleaved traces; the decisions use the idle
cores and cost 3% (9 s). The re-pick decides up to five more peaks per row, and each
decision reads its patch, the 7 cycles x 17 steps around the peak, with one binary search per
fragment and cell in that step's peak list: 1,428 searches a decision, in spectra of about
1,550 peaks spread over 25 GB, which is 59% of the busy samples. The fit itself is 20%.
Convert reads and inflates the 42 GB mzML (101 s alone, 2.1 GB; the threads waited in 96% of
the samples).

**Per-cycle storage would not help.** The plan's candidate was to store a cycle as one frame
with a step index per peak, as a diaPASEF frame carries a 1/K0 per peak. Memory cannot fall:
the peaks are what is held, and the per-spectrum records such a layout would replace are 0.21
GB; a two-byte step index per peak would add 6.7 GB. Time would not either: convert is
bound by reading and inflating the mzML, extract without the keys by the chromatogram writer,
and with them by searches that find one m/z in one step's peaks, which is the same search in
either layout. So it was not built.

**Cheaper re-picks.** Two were measured. Looking each cell's fragments up in ascending m/z,
each search starting from the previous one's position (an exponential search forward, the
same peaks by construction), wrote byte-identical tables but took 590 s against 515-522 s on
the same host: after the first search of a spectrum its upper levels are in cache, and the
forward search probes more. It was not kept. `retrace.q1_repick_peaks` (default 5) bounds the
peaks the re-pick compares, strongest first by area; at 0 the apex only moves to its own
peak's 2D apex. Each arm is the recipe with the key set, four NN seeds, against the recipe on
the same host:

| `q1_repick_peaks` | extract alone, demiguise | rows moved | entrapment library: human precursors at the matched FDP (1.224%), mean (range) | human library: precursors at 1%, mean (range) |
|---|---|---|---|---|
| 5 (the recipe) | 515-522 s | 131,168 | 97,225 (96,992-97,540) | 99,804 (99,661-100,034) |
| 2 | 373 s (-28%) | 108,394 | 97,074 (96,726-97,319), -0.16% | 99,305 (98,732-99,560), -0.50% |
| 0 | 330 s (-37%) | 87,359 | 96,093 (95,842-96,370), -1.16% | 98,733 (98,653-98,907), -1.07% |

The alternative peaks are worth about 1% of the identifications, and the strongest one beside
the apex's own carries most of it: at 2 the entrapment library is within the seed spread and
the human library half a percent lower, outside it, for 28% less extract time (about 12% of a
run). The three runs as one experiment at 2 (one seed) gave 113,051 precursors, 90,573
peptides and 9,056 protein groups (per run 108,892 / 109,219 / 108,998) against 113,013,
90,739 and 9,027 at 5, and the same replicate CV (0.1025 against 0.1023). The recipe keeps 5, the setting phase 2 measured and the one it reproduces byte for
byte; 2 is the setting to take when extract's time matters more than half a percent.

## 9. What is refused

- A file whose MS2 spectra name differing isolation targets and no offsets, without
  `convert.isolation_half_width` (convert, before the first 10,000 such spectra are read).
- An extraction whose estimated hit store passes the machine's memory, which is what a ZT
  Scan run read as one all-ion window does (`extract.max_footprint_gib`, docs/09).
- Every sweep key on a layout that is not a scanning quadrupole, on spectra with 1/K0, and,
  through `rt_im_train.q1_calibrate`, with `groups.window_groups > 1`.
- `extract.q1_fragments` without `rt_im_train.q1_calibrate`, with `extract.q1_cycles`, and on
  spectra_ms2 that is not v3 or whose recorded step is not the rank of its target.
- `features.q1_fragment_features`, `retrace.q1_repick` and `rescore.shadow_q1` without
  `extract.q1_fragments`; `retrace.q1_repick` with `promote_top_peaks > 1`,
  `retrace.repick` or `extract.apex_refine_intensity`; `rescore.shadow_q1` with
  `shadow_min_shared = 0`; `retrace.q1_repick_peaks` without `retrace.q1_repick`.
- `quant.q1_fragments` without `extract.q1_fragments` or with `fragment_selection =
  cross_run`; `q1_min_belonging = 0`; `q1_integration = elution_fit` without the scan form of
  the fixed window; its three companions without it.
- `quant.q1_cycle_halfwidth` without the scan form of the fixed window or with
  `fixed_scan_halfwidth_scale`; `mumdia quant` with it and no `--spectra-ms2`.
- `features.ms1_envelope` on spectra with 1/K0 and with window groups.
- `features.q1_precursor`, `features.q1_multiscale` and `features.q1_deconvolved` without
  `extract.q1_fragments`; `extract.q1_deconvolved` without `rt_im_train.q1_calibrate` or with
  `extract.q1_cycles`; `search_seed.q1_deconvolved` with `count_fragments_once`; both
  `q1_deconvolved` keys with `groups.window_groups > 1`, on spectra with 1/K0 and on
  spectra_ms2 without the recorded sweep.

## 10. The unfragmented precursor, multi-scale Q1 scores and Q1-deconvolved evidence

Three further readings of the sweep, each taken from scanning-quadrupole practice in the
literature: score each fragment's Q1 profile against the most intense fragment's and against
the unfragmented precursor's, its shape against the transmission and its centroid against the
precursor m/z, at several Q1 widths (Messner et al., 2021); and assign every fragment signal to
a precursor along the Q1 axis before it is counted (Q1 deconvolution). The mechanisms are in
docs/09_extract.md, section 14; this section has what they measured and what went into the
recipe. Every key is off by default and omitted from the canonical configuration; with all of
them off, the recipe on REP01, an Orbitrap AIF run, a native diaPASEF run and `ci/smoke.sh`
write every table byte for byte as before (section 10.7).

REP01 with the recipe of section 4 as the base, the human library on doxy and the entrapment
library on demiguise (32 threads each, both hosts shared, so walls there are indicative), NN
seeds 0-3 per arm (seeds 1-3 rescore the run's competed table), every arm of a library on one
host with one thread count. Matched FDP as in section 5: the human precursors where the arm's
combined entrapment FDP reaches the base arm's mean at a reported 1% (1.277% on this host),
interpolated along the `precursor_q` thresholds.

### 10.1 Where the identifications go

Against a reference search of REP01 in another engine's scanning-quadrupole mode (94,259
precursors at 1%), every reference precursor is put in the first class that applies
(`run_windows`, `psms_extracted`, `psms_scored` of the arm):

| class | base (human library) | + the three feature keys | base (entrapment library) | + the three feature keys |
|---|---|---|---|---|
| accepted at 1% | 84,215 (89.3%) | 84,546 (89.7%) | 82,288 (89.1%) | 82,566 (89.4%) |
| reference RT outside the candidate's window | 1,226 (1.3%) | 1,226 | 1,177 (1.3%) | 1,177 |
| inside the window, not extracted | 785 (0.8%) | 785 | 753 (0.8%) | 753 |
| extracted, removed by the rescore prefilter | 1,985 (2.1%) | 1,985 | 1,817 (2.0%) | 1,817 |
| scored, `precursor_q` above 1% | 6,048 (6.4%) | 5,717 (6.1%) | 6,287 (6.8%) | 6,009 (6.5%) |

The library covers every reference precursor. The largest loss is the scored bucket, of which
1,441 lie between 1% and 2% on the human library: the evidence is extracted and the classifier
ranks it below the threshold. Features can only act there (the prefilter reads a fixed set of
32 columns, so the 2.1% it removes are the same in every feature arm), and the three feature
keys take 4-5% of that bucket (331 and 278 reference precursors). The MuMDIA-only set grows from
15,006 to 15,782 human precursors on the entrapment library, at a combined FDP of 6.98% against
7.46%.

### 10.2 The unfragmented precursor is visible in MS2, and the decoys see it too

Measured first, on the base run (`vis.py`, 1,500 sampled seed PSMs per class): in the five
cycles around the seed scan, the precursor's M0 is present by the decision's coherent-cell rule
for 97.5% of the confident target seeds (M+1 97.3%, M+2 97.7%), but also for 83.9% of the decoy
seeds, 77.9% of the low-scoring target seeds and 72.1% at an m/z 0.37 below M0, where no isotope
of the candidate lies: the precursor region of a ZT Scan MS2 spectrum is crowded with the other
precursors the window transmits. The M0 share of M0 plus the six strongest fragments is 0.19
(median) for the confident seeds and 0.09 for decoys; their M0 Q1 profile correlates with the
transmission at 0.95 (median) against 0.81, and its centroid sits +0.27 steps from where the
window puts it (IQR -0.01 to +0.73). On the extracted rows of the arm with the family (4.18
million at peak rank 0): all three isotopes are present for 94.1% of the accepted targets,
80.8% of the decoys and 80.5% of the other targets, and the precursor holds half of the
precursor plus fragment intensity in every class (median 0.49-0.50). What separates is where:
the isotopes' fitted shift is 0.90 steps from the expected position (median) for the accepted
targets and 2.84 for the decoys, its distance from the fragments' shared centre 1.24 against
3.89, and the fragments' Q1 profiles correlate with the precursor's at 0.48 against 0.03. A
reversed decoy has its target's precursor m/z and Q1 position, so the family helps the
classifier only through those agreement terms.

### 10.3 Q1-deconvolved evidence: the separation, and what a hard count does with it

Every MS2 peak's Q1 centroid against the expected centre of the library fragment it matches,
in the steps whose window covers the precursor (the cells the seed and extract probe), as the
Q1 calibration learns the tolerance on the anchors and the decoy seeds (docs/08, section 4h):

| tolerance, steps | 0.5 | 0.75 | 1.0 (learned) | 1.5 | 2.0 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|---|
| anchors' peaks kept | 56.7% | 69.4% | 77.4% | 86.3% | 90.8% | 95.3% | 97.1% |
| decoys' peaks kept | 13.3% | 19.8% | 26.0% | 37.9% | 48.6% | 65.8% | 78.8% |

A real fragment's trace merges with any other ion of its m/z within the tolerance in the
neighbouring steps, which moves its centroid; 23% of the anchors' peaks fall outside one step.
Counting only the consistent peaks therefore removes real evidence along with the interference,
and every hard form lost identifications on the human library (four NN seeds each):

| arm | seed's confident targets | rows extracted | precursors at 1%, mean (range) | vs base | extract | wall |
|---|---|---|---|---|---|---|
| base | 25,305 | 4,179,683 | 99,902 (99,672-100,058) | | 559 s | 1,227 s |
| `extract.q1_deconvolved` (learned, 1.0 step) | 25,305 | 368,759 | 74,325 (74,151-74,461) | -25.6% | 209 s | 1,006 s |
| the same at 2.0 steps (development build) | 25,305 | 1,314,886 | 91,466 (91,345-91,694) | -8.4% | 400 s | 1,228 s |
| `search_seed.q1_deconvolved` (2 steps) | 21,482 | 4,072,093 | 97,379 (97,239-97,546) | -2.5% | 570 s | 1,257 s |
| both | 21,482 | 225,381 | 61,800 (61,615-61,940) | -38.1% | 152 s | 775 s |

The seed alone, searched again on the base run's spectra and library: 25,305 confident targets
without the key, 14,544 / 18,909 / 21,482 / 23,681 at 1 / 1.5 / 2 / 3 steps. In the ladder of
the extract arm (learned tolerance), 24.1% of the reference precursors are not extracted
(against 0.8%), while the scored bucket shrinks from 6.4% to 4.2%: the evidence that remains
is cleaner, but a quarter of the real precursors no longer meet the presence rules. On the
entrapment library the extract form keeps 73,653 human precursors at the matched FDP
(73,476-73,868, -24.5%) and both forms 61,671 (-36.7%), at a lower measured FDP at a reported
1% (1.09-1.19% and 1.05-1.13% against 1.25-1.31%): a smaller pool, not a better ranking. The
hard forms stay off: they are a speed lever (extract 209 s against 559 s) with an
identification loss no tolerance removed. The 2-step arm was run with a development build that
overrode the learned tolerance; the shipped keys have no such override.

The same evidence as features (`features.q1_deconvolved`, the extraction unchanged) separates
cleanly: among the accepted targets the consistent share of the matched intensity at the apex
is 0.53 (median) against 0.19 for decoys, and 10 fragments have a consistent match against 6.

### 10.4 Measured

| arm | human library: precursors at 1%, mean (range) | vs base | peptides / protein groups (mean) | entrapment: human precursors at the matched FDP (1.277%), mean (range) | vs base | at a combined FDP of 1.0% | combined FDP at a reported 1% |
|---|---|---|---|---|---|---|---|
| base (the recipe) | 99,902 (99,672-100,058) | | 80,456 / 7,739 | 97,493 (97,247-97,809) | | 95,285 | 1.25-1.31% |
| + `q1_precursor` | 100,033 (99,838-100,167) | +0.13% | 80,432 / 7,763 | 98,122 (97,773-98,759) | +0.65% | 95,744 | 1.18-1.26% |
| + `q1_multiscale` | 100,219 (99,946-100,408) | +0.32% | 80,635 / 7,723 | 97,580 (97,171-97,810) | +0.09% | 95,284 | 1.26-1.33% |
| + both | 100,112 (99,966-100,225) | +0.21% | 80,572 / 7,745 | 97,994 (97,671-98,248) | +0.51% | 95,528 | 1.26-1.31% |
| + `q1_deconvolved` (features) | 100,416 (100,167-100,546) | +0.51% | 80,916 / 7,764 | 98,116 (97,881-98,267) | +0.64% | 95,786 | 1.24-1.29% |
| + the three | 100,664 (100,424-100,840) | +0.76% | 81,019 / 7,765 | 98,841 (98,308-99,195) | +1.38% | 96,449 (+1.22%) | 1.21-1.30% |

Decoy fraction 0.0100 in every arm. On both libraries the three keys together are beyond the
seed spread of the base arm (the ranges do not overlap); each alone is inside it, except the
deconvolved evidence, whose ranges clear the base arm's on both libraries (narrowly on the
entrapment one).

### 10.5 Ablation

Each family left out of the three, by rescoring the competed table of the arm with all three
with `rescore.features_file` listing every other column (four NN seeds; this reproduces an arm
exactly: the three without `q1_deconvolved` gives the counts of "+ both" above to the last
digit, seed by seed):

| left out | human library: precursors, mean (range) | vs the three | entrapment: human at the matched FDP, mean (range) | vs the three |
|---|---|---|---|---|
| nothing (the three) | 100,664 (100,424-100,840) | | 98,845 (98,311-99,198) | |
| `q1_precursor` | 100,502 (100,346-100,658) | -0.16% | 98,251 (97,735-98,610) | -0.60% |
| `q1_multiscale` | 100,744 (100,620-100,811) | +0.08% | 98,507 (98,398-98,654) | -0.34% |
| `q1_deconvolved` | 100,112 (99,966-100,225) | -0.55% | 97,997 (97,674-98,251) | -0.86% |

The deconvolved evidence carries most of the gain on both libraries. The precursor family adds
on both; the multi-scale family adds on the entrapment library and nothing on the human one.
Each contribution alone is inside the seed spread; together they are not. The entrapment
ablations were rescored on doxy from the demiguise run's competed table, and the three there
reproduce the demiguise seeds' counts to the last digit, so the hosts' arithmetic agrees.

### 10.6 Label blindness

Among the rows whose untrained `prelim_score` is below the decoys' median (1.04 million decoys,
0.99 million targets, overwhelmingly false), the two-sample Kolmogorov-Smirnov statistic of
decoys against targets is 0.0019-0.0077 for every `q1p_*` feature, 0.0007-0.0049 for every
`q1m*` and 0.0020-0.0042 for every `q1d_*`, as for features already in the vector
(`n_matched_fragments` 0.0005, `frag_corr` 0.0037, `coelution_mean` 0.0034, `rt_error_abs`
0.0062). The accepted targets differ (`q1p_frag_corr_mean` median 0.48 against 0.03,
`q1m11_shape` 0.08 against 0.58, `q1d_share` 0.53 against 0.19).

### 10.7 Cost, byte identity

REP01 with the human library on thestral, 32 threads, the host otherwise idle, the recipe
without and with the three feature keys, one after the other (both write the scored table of
the doxy arms' seed 0: 99,915 and 100,840 precursors, so the hosts agree):

| stage | the recipe | + the three feature keys |
|---|---|---|
| convert / seed / multi-head RT calibration | 111 / 38 / 207 s | 115 / 38 / 204 s |
| Q1 calibration (with the deconvolution tolerance) | 64 s | 68 s |
| extract | 486 s | 522 s (+7%) |
| features / compete | 72 / 2 s | 83 / 3 s |
| rescore (459 features against 434) | 101 s | 152 s |
| quant | 19 s | 20 s |
| whole run, peak | 1,107 s, 35.3 GB | 1,217 s (+9.9%), 39.6 GB (+4.3 GB) |

The per-peak centroids are computed in about 6 s and hold 3.3 GB (one byte for each of the 3.34
billion MS2 peaks); the precursor patch adds one read of three m/z per row. The hard forms
make the run faster (1,006 s with extract's count on doxy against 1,227 s), which is all they
offer.

With every key off, the phase-5 binary against main's (both built from the same tree but for
this change, one after the other on one host): the recipe on REP01, 16 of 16 parquet tables
byte-identical and 19 of 20 reports (the multi-head summary: timings); the Orbitrap Q Exactive
AIF E. coli entrapment run, 15 of 15 and 19 of 20 (the same); the native diaPASEF run (timsTOF
Ultra 2, HYE 15 min, the diaPASEF recipe with retrace and second pass), 20 of 20 and 27 of 30
(retrace's stage timers and the multi-head summaries' timings); `ci/smoke.sh`, SMOKE_OK on both
and 1,863 of 1,863 parquet tables byte-identical, the reports identical but for the binary's
path in the manifests' `cli_args` and the library-cache key, which includes the binary.

### 10.8 Three runs as one experiment

REP01-03 as one `mumdia run` (pooled rescore, second pass), human library, one NN seed,
thestral, 32 threads, both arms with the phase-5 binary (the base with the three keys off):

| experiment | precursors | peptides | protein groups | per run on `run_psm_q` (second pass) | first pass per run | median precursor / protein CV | wall, peak |
|---|---|---|---|---|---|---|---|
| the recipe (section 4) | 113,439 | 91,063 | 9,041 | 109,526 / 109,600 / 109,604 | 101,518 / 101,425 / 100,851 | 0.1024 / 0.0448 | 53 min, 69.9 GB |
| + the three feature keys | 113,858 (+0.37%) | 91,270 (+0.23%) | 9,084 (+0.48%) | 109,482 / 110,141 / 110,308 (+0.26%) | 101,715 / 101,940 / 101,467 (+0.33%) | 0.1024 / 0.0449 | 51 min, 77.5 GB |

Smaller than the single-run gain, as with every identification key measured in an experiment
here (the second pass adds the same precursors to every run in both arms). The quantification
is unchanged, as it must be: the keys change no trace and no integration. The peak is 7.6 GB
higher: each concurrent second-pass extract holds the run's per-peak Q1 centroids (3.3 GB).

### 10.9 What went into the recipe

The three feature keys together are beyond the seed spread of the base arm on both libraries
(+0.76% precursors on the human library, +1.38% human precursors at the matched entrapment FDP
and +1.22% at a combined FDP of 1.0%), they do not raise the measured FDP at a reported 1%, and
their features are label-blind. So `configs/examples/ztscan.json` sets `features.q1_precursor`,
`features.q1_multiscale` and `features.q1_deconvolved`. All three, rather than the two that
carry most of it: leaving the multi-scale family out costs 0.34% on the entrapment library, and
the set was measured as a whole. The keys that count only Q1-consistent peaks
(`search_seed.q1_deconvolved`, `extract.q1_deconvolved`) lose identifications at every
tolerance and stay off and out of the recipe. No default changes (section 11).

## 11. Open items

- **A second acquisition, and known ratios.** Every number here is one ZenoTOF 8600 method on
  one sample. The benchmark collection holds the ZenoTOF 7600+ ZT Scan DIA arms of PXD070049
  (10 Da windows on a 15 min gradient and 5 Da on 5 min, 50 ng; three human/yeast/E. coli
  conditions with known ratios, six replicates each, and single-species runs that serve as
  foreign-species nulls), which would settle both gates at once, but as `.wiff`/`.wiff2`
  files. The converters that write their isolation targets (the SCIEX MS Data Converter, or
  ProteoWizard built with the vendor readers) carry a vendor licence, and whether to accept it
  is the user's decision; until then no ZT Scan key can become a default (CLAUDE.md), and the
  suggested half-width (five steps) is unverified on those methods. The collection's
  TripleTOF 6600 Scanning SWATH arm (PXD028735) has one pure-human run converted to mzML
  already, but its spectra carry the 1 m/z bin as their window rather than the quadrupole's,
  so the sweep model would first need a convert key that replaces those offsets with the
  method's window; not attempted.
- **Known ratios for quantification.** The quantification choices (section 6) are measured
  on replicate precision only. Whether leaving out interfered fragments (`quant.q1_fragments`)
  removes ratio bias, and how much background the 4-cycle window adds, needs ZT Scan data with
  known ratios (the arms above).
- **Calibration of the reported FDR.** With the four keys the measured entrapment FDP at a
  reported precursor q of 1% is 1.22% (1.39% without them), and with the recipe as it is now
  1.21-1.30% on another host (section 10.4); compare counts at a measured FDP.
- **The re-pick's cost** (section 8): `retrace.q1_repick_peaks` 2 takes 28% off extract for
  0.2-0.5% of the identifications; a cheaper statistic that ranks the alternatives without
  deciding them was not found.
- **The Q1 window's softness** (wider and softer than nominal, section 3) is not separated
  into quadrupole edges and ions in transit while the sweep moves on.
- **Q1-deconvolved evidence as a count.** Counting only the peaks whose Q1 centroid is where
  the candidate is expected loses identifications at every tolerance tried (section 10.3),
  because a real fragment's trace merges with other ions of its m/z; the same evidence as
  features gains. Weighting each hit by its Q1 consistency instead of cutting it, or
  splitting merged traces with the transmission model rather than at valleys, were not tried.
- **The precursor in MS2 against reversed decoys.** A reversed decoy shares its target's
  precursor m/z and Q1 position, so the unfragmented precursor cannot separate a decoy from a
  present target; the family adds through the agreement of the fragments with it (section
  10.2).
- **Where the excess of the measured FDP sits.** At a matched score the entrapment targets
  accepted at 1% carry more real co-elution than the decoys accepted with them (the
  fragment-versus-reference correlations of their strongest fragments, AUC 0.55-0.57): they
  borrow the signal of a co-eluting peptide that is present. The Q1 consistency does not
  separate the two (0.49), so `features.q1_consistency` lowers the measured FDP at a reported
  1% only slightly (1.28% to 1.20% the family alone); the borrowed co-elution is the next
  target for a feature.
- **The family on other scanning-quadrupole methods.** It reads the per-fragment decisions and
  is measured on one method; whether the offset tolerance learned there carries the same
  information on 5 and 10 Da ZT Scan methods needs the second acquisition above.
