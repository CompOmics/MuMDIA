# 37. SCIEX ZT Scan DIA: scanning-quadrupole input
> Part of the MuMDIA developer documentation (see docs/README.md).

This guide describes how MuMDIA reads, searches and quantifies SCIEX ZT Scan DIA runs, in
which the quadrupole sweeps the precursor range instead of stepping through fixed windows.
Section 1 describes the acquisition, section 2 how a run is read and recognised before a
search, section 3 the opt-in model of the sweep layer by layer, section 4 how to run it,
section 5 what was measured, section 6 the quantification window in Q1 cycles, section 7 the
gate on the fragments that belong that was measured and not added, section 8 where time and
memory go, section 9 what is refused, and section 10 what is open. Reading a ZT Scan file
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
is a licence decision (section 10).

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

The four identification keys of the recipe are the phase-2 set, measured together: +3.0%
precursors on the human library and +3.2% human precursors on the entrapment library at the
base arm's measured FDP, whose value at a reported precursor q of 1% falls from 1.39% to
1.22%.

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
- Expect about 19 min and 35 GB for one run with the human library on 40 threads (28 min and
  37 GB with the 4.1 million-target entrapment library on 64 threads of a busier host), and
  about 60 min and 89 GB for the three runs as one experiment on 64 threads, the peak from
  three concurrent second-pass extracts (section 8).
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
run). The recipe keeps 5, the setting phase 2 measured and the one it reproduces byte for
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

## 10. Open items

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
  reported precursor q of 1% is 1.22% (1.39% without them); compare counts at a measured
  FDP.
- **The re-pick's cost** (section 8): `retrace.q1_repick_peaks` 2 takes 28% off extract for
  0.2-0.5% of the identifications; a cheaper statistic that ranks the alternatives without
  deciding them was not found.
- **The Q1 window's softness** (wider and softer than nominal, section 3) is not separated
  into quadrupole edges and ions in transit while the sweep moves on.
