# TIMS roadmap: diaPASEF support at DIA-NN parity

Status: P0-P2 done, P3 done within the agreed scope, P4-P6 implemented and measured on the E. coli file (all new behaviour default off, P6 at 0.62x DIA-NN in peptides, entrapment FDP 0.36-0.38%); P7 (IM peak width) implemented and measured with no gain (all keys default off); 2026-09-24. Continued in TIMS_ROADMAP_bis.md (loss diagnosis and next levers). Code references are to `8d3db2e`. Branch `IM`. The objective order is fixed: identification
sensitivity at 1% first. FDR validity (entrapment), quantification accuracy and
runtime come after the identification gap is closed, and each keeps its own gate
from `docs/20_sensitivity_and_quantification_playbook.md`.

## 1. Why diaPASEF underperforms: MuMDIA discards ion mobility

MuMDIA reaches DIA-NN parity on Orbitrap DIA. On diaPASEF it does not, and the code
explains why: the ion mobility (IM) dimension is removed before the engine reads
any spectrum, and no stage downstream has an IM path.

| Stage | Current behaviour | Location |
|---|---|---|
| vendor conversion | msconvert `--combineIonMobilitySpectra` collapses each frame's TIMS scans into one 3D spectrum | `raw.rs:425-440` (Bruker arg at `:437`) |
| convert | reads only m/z and intensity arrays; no per-peak or per-window IM | `stages/convert.rs:70-71` (peak arrays), `:288` (isolation window) |
| window identity | keyed on `(lower_mz, upper_mz)` only | `convert.rs:396`, `extract.rs:1192` (`window_groups`), `search_seed.rs:583-584` |
| spectra schemas | `spectra_ms1/ms2`, `isolation_windows` carry no IM column (schema v1) | `stages/convert.rs` (`spectra_ms2` schema around `:191`) |
| read-back | `Peak.ion_mobility` and `IsolationWindow.im_lower/im_upper` hard-set `None` | `spectra.rs:150-151`, types at `mumdia-core/src/types.rs:14, 40-41` |
| library | no `predicted_im` in `fragment_library_precursors`; DIA-NN importer drops `IonMobility`; no IM predictor | `schema.rs:22`, `predict_frag.rs:288`, `scripts/import_diann_lib.py:92-95`, `predict.rs:14-20` |
| rt-im-train | `im_pred_cal/im_lo/im_hi` written null; no `w_im`; no IM config | `stages/rt_im_train.rs:421-435`, `config.rs:626` |
| search-seed / fragment index | candidate range on precursor m/z only; no IM gate | `matchers/fragindex.rs:154` (`candidate_range`) |
| extract | RT post-filter only (`extract.rs:1400, 1630, 1684, 1756, 1868, 1940, 2390`, demix `:826`); `Hit` has no IM; RT-only chromatograms; `apex_im` always null | `extract.rs:159` (`Hit`), `:494` (`ChromChunk`), `:3375` (`apex_im`) |
| features / rescore | no IM family among the 16 registered families (P5 adds an opt-in block outside `FAMILIES`) | `stages/features.rs:55` (`FAMILIES`) |

The unused hooks in the data model (`Option` IM fields in `types.rs`, nullable IM
columns in `run_windows`, nullable `apex_im`) mean the design anticipated 4D, but
nothing populates or consumes them.

### What the discarded dimension is worth on this data

The benchmark acquisition is a timsTOF Ultra 2, 15-min gradient, 50 ng E. coli,
read from `analysis.tdf`:

- 15,050 frames: 1,673 MS1 and 13,377 diaPASEF MS2; 936 TIMS scans per frame;
  1/K0 acquisition range 0.64-1.45 V s cm^-2; m/z 100-1700.
- 8 window groups x 3 IM slots = 24 isolation windows of 25 Th covering precursor
  m/z 400-1000. Each MS2 frame therefore contains three precursor windows that
  differ in both m/z and 1/K0. A cycle (1 MS1 + 8 MS2 frames) is about 0.97 s.
- DIA-NN's own report on this file (section 2): median |observed - predicted| 1/K0
  0.0080 (p95 0.029) V s cm^-2, against an IM slot width of roughly 0.2-0.3.
  A calibrated IM window of about +/-0.03 therefore excludes on the order of 80-90%
  of the co-isolated ions in a slot. Median elution width 6.8 s, median RT error
  8.3 s (p95 30 s).

Collapsing IM keeps all of that interference in every fragment trace. This agrees
with the published account of what makes diaPASEF analysis work: two-dimensional
(RT x IM) peak picking and IM-aware scoring (Demichev et al. 2022), and
feature-free scoring on the raw 4D signal (Wallmann et al. 2025).

## 2. Baseline (P0), 2026-09-23

Inputs:
- `LFQ_Ultra2_diaPASEF_15min_50ng_Ecoli_01.d` (4.7 GB)
- `fasta/ecoli_22032024.fasta` (4,390 proteins)
- host: 2x EPYC 9354, 755 GB RAM, 64 threads for both engines

Both engines use the same search space:
- trypsin `K*,R*`, 1 missed cleavage, length 7-30
- precursor charge 2-4
- fixed carbamidomethyl-C, at most 1 variable Met-oxidation
- N-terminal Met excision

The precursor m/z range is bounded by the isolation scheme (400-1000).

**DIA-NN 2.5.0**, library-free (`--fasta-search --predictor`), mass accuracy on auto.
Output is in `/public/compomics2/Robbe/MuMDIA_data/output_diann`, and the full command
line is in `report.log.txt` there.

**MuMDIA 0.4.0** (`8ea2b3d`, per the run's `manifest.json`), FASTA mode, from `configs/examples/fasta-sidecars.json`:
- MS2PIP HCDch2, 12 fragments
- DeepLC 4.4.0
- Extended features
- strict `nn_torch`
- uncapped peaks

The `.d` goes through the msconvert path, which this run exercises end to end for
the first time. The config is `config.local-tims-baseline.json` (untracked; the same
fields plus interpreter paths). Output is in
`/public/compomics2/Robbe/MuMDIA_data/output_mumdia`.

| at 1% | DIA-NN 2.5.0 | MuMDIA 0.4.0 | MuMDIA / DIA-NN |
|---|---|---|---|
| precursors | 18,673 | 8,721 | 0.47 |
| stripped peptides | 15,404 | 7,662 | 0.50 |
| protein groups | 1,919 | 1,333 | 0.69 |
| MuMDIA empirical decoy fraction (peptide level) | - | 0.98% | - |
| wall clock | 2:11 | 45:30 (28:00 of it msconvert) | 20.8x |
| peak RSS | 20.7 GB | 30.3 GB | 1.46x |

Stripped-peptide overlap, I/L merged: 7,473 shared, 7,931 found only by DIA-NN,
and 188 found only by MuMDIA. MuMDIA identifies a near-subset of DIA-NN, at a
calibrated decoy fraction. The deficit is sensitivity, not a loosened threshold.

Units are those printed by `bench/tims_compare.py`:
- DIA-NN precursors: `Q.Value` <= 0.01 (run-level precursor q)
- DIA-NN peptides: distinct stripped sequences among those precursors
- DIA-NN protein groups: `PG.Q.Value` <= 0.01
- MuMDIA precursors: `precursor_q`
- MuMDIA peptides: `peptide_q_value` (stripped)
- MuMDIA protein groups: `pg_q_value`

Reproduce the comparison with:

```text
python bench/tims_compare.py \
  --diann  /public/compomics2/Robbe/MuMDIA_data/output_diann/report.parquet \
  --mumdia /public/compomics2/Robbe/MuMDIA_data/output_mumdia/psms_scored.parquet
```

### Where the missing identifications are lost

Every one of DIA-NN's 15,404 peptides is in the MuMDIA library, so the search space
is not the cause. Of the 7,931 DIA-NN-only peptides:

| where lost | peptides | share |
|---|---|---|
| not in library | 0 | 0% |
| in library, never accepted by extract | 2,944 | 37% |
| extracted, below 1% after rescore | 4,987 | 63% |

The rescore-side losses are the larger group. For DIA-NN 1% precursors that MuMDIA
extracted but did not accept, 64% of the rank-0 apexes lie within 5 s of DIA-NN's
RT (median |dt| 1.9 s). For accepted ones the figure is 81% (median 1.0 s). The
right peak is usually found, but its fragment evidence does not separate from the
decoys. This is the signature of co-isolated interference in IM-collapsed traces,
which IM gating (P4) and IM features (P5) address directly.

### Stage diagnostics (MuMDIA)

| stage | observation | consequence |
|---|---|---|
| msconvert | 28 min; 20.7 GB mzML from a 4.7 GB `.d`; 41,804 spectra = 1,673 MS1 + 13,377 x 3 slots | slots stay separate spectra (24 m/z windows); the per-slot 1/K0 range survives only as mzML metadata, which `convert` ignores |
| convert | 2:27; 8.4 GB spectra artifacts; MS2 peaks/spectrum p5/25/50/75/95 = 2,111 / 4,416 / 7,600 / 25,729 / 61,793; MS1 median 701,377 points/spectrum (1.16e9 in total) | Bruker data arrive without vendor peak picking. MS1 in particular is effectively profile data. P1 must centroid in m/z x IM |
| predict-frag | 5:09 for 1.33M peptidoforms (MS2PIP + DeepLC) | library build, independent of IM |
| search-seed | 32 s; 61,257 PSMs, **only 130 confident** | far too few anchors: an imported DIA-NN library on a HYE run gave 21,856, and the HCDch2 FASTA library 19,308 (CLAUDE.md). The seed scores IM-collapsed spectra |
| multi-head RT + rt-im-train | calibration on 129 anchors; in-sample `w_rt` 10.1 s, median residual 2.0 s | against DIA-NN's RT, the calibrated prediction has median error 3.5 s but **p95 108.9 s**: a heavy tail from a 129-point fit over 80 heads |
| mass calibration | fragment ppm MAD 1.0 on 1,506 deviations; tolerance narrowed to 5 ppm | consistent with DIA-NN's median MS2 accuracy of 2.5 ppm; not the bottleneck |
| extract | 4:38; 243,369 candidates accepted; 3.65M chromatograms | |
| features, compete, rescore | 7 s, 5 s, 54 s; classifier `nn_torch` (from `psms_scored.parquet.report.json`) | |

Two levers come before any new IM feature, and both are caused by the collapse:
- The seed anchor count (130) limits RT calibration, and so every RT window
  downstream.
- The fragment traces carry all co-isolated ions of a 0.2-0.3 V s cm^-2 slot.

An IM-gated seed (P3/P4) should raise the anchor count. That is the first check to
run once P1 exists.

Update after P1: native 2D centroiding alone raised the anchors from 130 to 4,313,
but RT accuracy on DIA-NN's 1% precursors did not improve (see "P1 result").

## 3. Environment on the development host

- Python: one pyenv virtualenv, `mumdia-tims` (Python 3.11.11), contains every
  sidecar dependency:
  - torch 2.14.0+cpu, deeplc 4.4.0, ms2pip 4.2.0, psm-utils 1.5.5, pyarrow 25.0.1
  - mokapot 0.10.0, scikit-learn
  - im2deep 2.0.2

  im2deep 2.0.2 resolved against DeepLC 4.4.0 and torch 2.14 without downgrades (it
  pulls lightning 2.6.6). The older pyenv env `mumdia` (DeepLC 3.1.9, no torch) is
  below the DeepLC floor and must not be used.
- Rust: the pinned toolchain 1.96.1 via rustup. The first automatic install left
  the toolchain without `cargo`, and `rustup toolchain install 1.96.1 --component
  rustfmt,clippy` repaired it.
- msconvert: there is no native binary on the host. `~/bin/msconvert-docker/msconvert`
  wraps the `chambm/pwiz-skyline-i-agree-to-the-vendor-licenses` image. Wine inside
  that container needs `--security-opt seccomp=unconfined` on this kernel; without it,
  it fails with `wine: socket : Function not implemented`. The config points
  `convert.msconvert` at the wrapper. This is a baseline convenience only; P1
  removes msconvert from the diaPASEF path.
- DIA-NN: `~/bin/diann-2.5.0/diann-linux` (Academia build; ships `libtimsdata.so`
  and reads `.d` natively). MuMDIA does not ship or invoke DIA-NN.

## 4. Phases

Every phase is measured on the E. coli file with `bench/tims_compare.py`. Anything
that becomes a default additionally needs a second diaPASEF acquisition:
- ProteoBench PYE `A9_G_DIA_nLC_tTOF_R1.d` in `/public/local/ProteoBench/PYE_diaPASEF`
- the `nn_torch` mean over 3 seeds, since single-seed deltas under ~1% are noise
  (CLAUDE.md)

Count gains alone do not establish a default. Before promotion, check the decoy
fraction, and from P4 onward run entrapment.

### P0: Measurement harness (this document)

- `bench/tims_compare.py`: counts in named units, the MuMDIA empirical decoy
  fraction, and the I/L-merged stripped-peptide overlap.
- Baseline outputs are kept in the two output directories, with `/usr/bin/time -v`
  logs (`time.log`) for wall time and peak RSS.
- Diagnostics are kept for later phases: the DIA-NN IM/RT residuals above, and the
  window scheme from `analysis.tdf`.

### P1: Native TDF reader with IM preserved (I/O)

Goal: read `.d` directly with IM kept, and without an mzML round trip.

- **Reader.** `mzdata 0.66` (already a dependency) has an optional `bruker_tdf`
  feature built on `timsrust 0.4.1`, with frame-to-array mapping and TIMS/TOF
  calibration (`mzdata/src/io/tdf/`). Evaluate it first, because `convert` already
  drives `mzdata::MZReader`. Use `timsrust` directly if mzdata's spectrum model
  loses the per-slot structure or costs too much time.
  - Both pull `rusqlite`, which compiles bundled SQLite C code. This conflicts with
    the "no C toolchain" build goal stated in `raw.rs:5-13`. Resolve it by feature
    gating, or accept it for this format.
  - timsrust is the MannLabs reader that Sage uses for timsTOF data.
- **Representation.** Emit one MS2 spectrum per (frame, window slot), carrying:
  - the slot's m/z bounds and 1/K0 bounds;
  - peaks aggregated over the slot's TIMS scans, with a per-peak intensity-weighted
    1/K0.

  This matches how the three slots per frame are physically separate precursor
  windows. Aggregation is 2D centroiding in m/z x IM: merge TOF indices within a
  ppm tolerance, and scans within an IM tolerance. Apply a noise floor (a minimum
  peak count or intensity), because raw diaPASEF frames hold ~61k peaks each on
  average on this file.

  Emit MS1 frames the same way, with per-peak 1/K0.
- **Schemas.** Bump to v2:
  - `spectra_ms1`/`spectra_ms2` gain a per-peak `im` `LargeList<f32>`;
  - `isolation_windows` gains `im_lower`/`im_upper`;
  - window identity becomes (m/z range, IM range).

  Old v1 artifacts stay readable with `im = None`. `spectra.rs` fills
  `Peak.ion_mobility` and `IsolationWindow.im_*`, and `IsolationWindow::covers`
  gains an optional IM check.
- **Scope of the key change.** Every `(lower_mz, upper_mz)` grouping site listed in
  section 1, plus `groups.rs` window bands (`docs/33`).
- **Keep msconvert** for non-TIMS Bruker and as a fallback. It must not be the
  diaPASEF default.
- **Measure** against the msconvert baseline:
  - convert wall time and RSS;
  - artifact size;
  - peaks per spectrum (`docs/04` peak census);
  - end-to-end IDs with IM still unused downstream. This isolates the reader, since
    slot separation alone may change counts.

### P1 result (2026-09-23, `29575e1` + gap default 30)

Implemented as `stages/convert/tdf.rs` (docs/04_convert.md, "Native timsTOF reader").
- **Reader.** timsrust 0.4.2 with only `tdf`. timsrust 0.6.6 was rejected on
  dependencies: `timsrust-core` 0.6 depends on `filemanager` with default features,
  which pulls object_store (AWS/Azure/GCP), tokio and a second arrow/parquet (about
  2,000 lockfile lines). mzdata's `bruker_tdf` was rejected because it decodes each
  frame once per slot, has no parallel path, and computes slot bounds and peak 1/K0
  with different converters.
- **Calibration.** m/z is timsrust's TOF conversion. It agrees with msconvert's
  vendor-calibrated m/z to a median of 0.02 ppm on MS1 and -0.58 ppm on MS2
  (p5/p95 about +/-4 ppm). 1/K0 uses Bruker `TimsCalibration` model 2, because
  timsrust's linear interpolation is off by up to 0.030 V s cm^-2 on this file.
- **Schema v2** as planned. Windows are keyed on (m/z, 1/K0) via
  `IsolationWindow::key`, and `groups` deduplicates m/z ranges. `demix_apex_scan`,
  the covering-window grid and `prescan` stay m/z-only until a candidate IM exists
  (P3/P4). On this scheme the slot m/z ranges are disjoint.

**Centroiding parameters.** These were chosen on seed confident PSMs on this file
only. All values are provisional, and none is promoted.

| `tdf_min_points` / `tdf_im_gap_scans` | MS2 peaks (total / median per slot) | seed confident PSMs |
|---|---|---|
| msconvert baseline | 711 M / 7,600 | 130 |
| 2 / 5 | 55.5 M / 224 | **0** (the run then fails: no RT anchors) |
| 1 / 5 | 486 M / 7,243 | 367 |
| 2 / 10 | 60.4 M / 367 | 3,616 |
| 1 / 20 | 392 M / 6,780 | 4,161 |
| 2 / 20 | 66.6 M / 557 | 4,139 |
| **2 / 30 (default)** | 71.0 M / 684 | 4,313 |
| 2 / 50 | 77.3 M / 834 | 4,227 |
| 3 / 30 | 35.2 M / 176 | 4,307 |

At 50 ng most fragment ions arrive as sparse single counts spread over a mobility
profile of about 20 scans. A 5-scan gap splits them into singletons, which the
2-point floor then deletes. That is the peak-group failure CLAUDE.md describes for
the peak cap, reached by another route. The seed count plateaus from 20 to 50 scans.

**End to end** (FASTA mode, `config.local-tims-baseline.json`, 64 threads, IM unused
downstream; units as in section 2):

| at 1% | msconvert (P0) | native 2/30 | native 1/30 | native 2/30, seed tol 10 ppm |
|---|---|---|---|---|
| precursors | 8,721 | 8,302 | 6,911 | 8,301 |
| stripped peptides | 7,662 | 7,231 | 6,008 | 7,214 |
| protein groups | 1,333 | 1,265 | 1,166 | 1,243 |
| decoy fraction (peptide) | 0.98% | 0.98% | 0.98% | 0.98% |
| seed confident PSMs | 130 | 4,313 | 4,306 | 4,659 |
| learned fragment tolerance | 5.0 ppm | 20.4 ppm | 19.6 ppm | 17.2 ppm |
| wall clock | 45:30 | 8:45 | 10:02 | 8:03 |
| peak RSS | 30.3 GB | 18.6 GB | 18.9 GB | - |

Stage costs (baseline, then native 2/30):
- convert: 28:00 msconvert + 2:27, against 1:00 (48 s standalone at 7.9 GB RSS);
- artifacts: 8.5 GB against 1.2 GB;
- MS1 points: 1.16e9 against 4.2e7 (median 26,574 per spectrum);
- search-seed: 32 s against 10 s;
- extract: 4:38 against 0:26 (243,369 against 226,093 accepted).

Reading:
- The reader is **5.2x faster end to end at -5.6% peptides**, with IM still unused
  downstream. The empirical decoy fraction is unchanged, and the overlap stays a
  near-subset of DIA-NN (7,065 shared, 164 MuMDIA-only).
- **The 33x seed anchor gain did not translate into RT accuracy.** On DIA-NN's 1%
  unmodified precursors (n = 17,524), |calibrated RT - DIA-NN RT| is median 3.0 s /
  p95 14.7 s for the baseline and 4.0 s / 16.3 s for native 2/30. The p95 108.9 s in
  section 2 was measured on a different population. On the precursors that matter,
  RT calibration was never the bottleneck, so the "first lever" in section 2 is
  weaker than stated.
- **The noise floor helps.** Keeping singletons (1/30) costs 17% of peptides against
  2/30.
- **The learned fragment tolerance is window-bound, not data-bound.** The ppm MAD is
  2.1, but 1.5 x p95 of the deviations stays at 17-20 ppm whatever the seed tolerance
  is. That means random matches dominate the tail. The msconvert baseline learned
  5 ppm from 130 PSMs. This moves to P6: an estimator robust to the random-match
  floor, and extract at a fixed 5-10 ppm.
- **Second acquisition** (PYE `A9_G_DIA_nLC_tTOF_R1.d`, timsTOF Pro, 11 window groups
  x 2-3 slots = 26 windows): native convert in 8.9 s, 49,016 MS2 slot spectra.
  Under 2/30 its MS2 is much sparser (median 76 peaks, p5 1), so the floor may cost
  more there. An end-to-end PYE run (ProteoBench FASTA) is still open and is
  required before any `tdf_*` default is claimed. The same holds for 3 `nn_torch`
  seeds.

### P2: Library ion mobility (IM2Deep)

- Add `predicted_im` (1/K0, f64, nullable) to `fragment_library_precursors`. This
  is a schema bump, and old libraries read as null.
- Add a new sidecar, `scripts/im2deep_worker.py`, following the existing
  positional-file contract (`docs/13_sidecars.md`):
  - input: peptidoform, charge, precursor m/z;
  - output: CCS and 1/K0.
- Conversion: use IM2Deep's `im2ccs`/`ccs2im` (Mason-Schamp, N2) in the worker.
  Port it to `mumdia-core` only if it proves a measurable cost, or if the engine
  must itself convert a CCS-only library.
- Plumbing:
  - add a version floor and a `mumdia doctor` probe for `im2deep`;
  - add a `predict_frag.im_predictor` (`none | im2deep`) config key with an
    interpreter field.
- IM2Deep's multiconformer output (Devreese et al. 2025) can predict more than one
  CCS for a precursor. Only the single-conformer model is used. Multiconformer
  output is future work with no phase attached: IM2DeepMulti's two outputs are
  ordered by value (`FlexibleLossSorted`), not by abundance, so neither is a main
  conformer.
- The DIA-NN importer maps `IonMobility` to `predicted_im`. That gives a zero-cost
  comparison arm for isolating library IM quality from the engine.

### P2 result (2026-09-23)

Implemented as planned, and the calibration core of P3 is pulled forward so that the
predictions can be judged in the unit P4 will use.
- **Library.** `fragment_library_precursors` v2 has a nullable `predicted_im`, which is
  always written. It comes from `scripts/im2deep_worker.py` (IM2DeepUni, uncalibrated,
  converted with IM2Deep's `ccs2im`) under `predict_frag.im_predictor = im2deep`, or
  from the DIA-NN importer (`IM` or `IonMobility`). The default stays `none`. There is
  a new interpreter role, `predict_frag.im2deep_python`, with a `doctor` probe and a
  version floor of 2.0.0. A missing prediction is a hard error. v1 libraries load
  unchanged.
- **Seed anchors.** `seed_psms` v2 has `observed_im`: the intensity-weighted median
  1/K0 of the matched fragment peaks of each confident target, within the search
  tolerance.
- **Calibration.** In rt-im-train, a per-charge linear fit in CCS space on those
  anchors, with the global fit as fallback. `w_im` is always sized on held-out anchors
  (30% by base peptide). It fills `run_windows.im_pred_cal/im_lo/im_hi` and the `im_*`
  fields of `cal.json` (docs/08 section 7). Nothing reads the windows yet.

**Identifications are unchanged.** The E. coli run (`output_mumdia_p2`,
`config.local-tims-p2.json`) gives 8,302 precursors, 7,231 peptides and 1,265 protein
groups at a 0.98% decoy fraction, as P1. All 226,093 scored rows have bit-identical
`score` and `q_value`.

**Cost.** IM2Deep takes 244 s for 1,333,950 precursors on 64 threads, CPU torch. The
run takes 12:09 against 8:45; the difference is the prediction plus sidecar I/O. Peak
RSS is 18.7 GB, unchanged. The prediction is per library, not per run.

**Calibration on this run.**
- 4,313 anchors (charge 2: 3,189; charge 3: 1,008; charge 4: 116), 1,229 of them held
  out.
- Per-charge CCS fits: z2 `4.26 + 1.027 x`, z3 `-11.30 + 1.080 x`, z4 `-39.15 +
  1.120 x`.
- Held-out |residual| median 0.0105, p95 0.0398 V s cm^-2; in-sample median 0.0103. A
  linear fit does not memorise, so the two agree.
- `w_im` = 0.040.

**Accuracy against DIA-NN's observed `IM`** on its 18,673 1% precursors (all in the
library). |error| median / p95 in V s cm^-2 (`bench/tims_im_check.py`):

| source | all | z2 (13,398) | z3 (4,796) | z4 (479) |
|---|---|---|---|---|
| IM2Deep raw `predicted_im` | 0.0385 / 0.0695 | 0.0360 / 0.0597 | 0.0486 / 0.0826 | 0.0502 / 0.0852 |
| MuMDIA calibrated `im_pred_cal` | **0.0104 / 0.0329** | 0.0095 / 0.0285 | 0.0138 / 0.0419 | 0.0135 / 0.0434 |
| DIA-NN's own `Predicted.IM` (reference) | 0.0081 / 0.0291 | 0.0070 / 0.0232 | 0.0124 / 0.0357 | 0.0139 / 0.0340 |
| seed `observed_im` (anchors, n = 4,036) | 0.0021 / 0.0167 | 0.0018 / 0.0109 | 0.0033 / 0.0341 | 0.0027 / 0.0187 |
| native MS1 peak at DIA-NN apex (n = 8,339) | 0.0065 / 0.219 | 0.0055 / 0.218 | 0.0090 / 0.221 | 0.0068 / 0.227 |

- 97.6% of DIA-NN's 1% precursors have their observed 1/K0 inside
  [`im_lo`, `im_hi`]. That is the recall a P4 gate would have at this `w_im`
  (half-width 0.040, against a slot of 0.2-0.3).

Reading:
- **Uncalibrated IM2Deep is offset for this instrument.** The median error is 0.036 to
  0.050 and grows with charge. Its training scale is not the Ultra 2's, and per-run
  calibration is required, as with DeepLC.
- **After calibration, the error is 1.3x DIA-NN's median and 1.1x its p95.** z4 is as
  good as DIA-NN's. z2 and z3 carry the gap. P4 can gate on these windows.
- **The native reader's IM agrees with DIA-NN.** The fragment-based seed anchors
  deviate by a median of 0.0021 V s cm^-2. The MS1 check agrees at the median (0.0065)
  but has a heavy tail (p95 0.22) and matches only 45% of precursors. At 50 ng many
  precursors leave no MS1 centroid, and the most intense peak within 10 ppm is often a
  different ion in another mobility band. That column tests this crude lookup, not the
  reader.
- **Levers that remain for IM accuracy**, none measured yet:
  - IM2Deep fine-tuning on the anchors (P3);
  - a non-linear calibration for z3;
  - weighting anchors by their own IM agreement.

  Measure them against the P4 result, not before it.

### P3: IM calibration (rt-im-train)

Partly done in P2: the seed's apex 1/K0, the per-charge CCS fit, held-out `w_im`, and
the `run_windows`/`cal.json` outputs. What remains: the IM gate on the seed probe
(P4), the optional IM2Deep fine-tune, and carrying `observed_im` through `seed-pool`
for grouped runs.

- The seed reports the apex 1/K0 for its confident PSMs. This is the
  intensity-weighted 1/K0 of the matched fragment peaks, and needs P1 peaks.
- Fit predicted to observed IM:
  - linear first, since instrument drift is near-affine;
  - LOESS as the fallback via the existing `calibrate.rs` kernels.
- Size `w_im` from held-out residuals. Apply the RT lesson (`docs/08` section 4):
  in-sample residuals are optimistic and can rank models backwards, so
  `window_holdout_frac` must apply to IM from the start.
- Outputs:
  - fill `im_pred_cal`, `im_lo`, `im_hi` in `run_windows`;
  - add `w_im`, `im_residual_*` and `w_im_sizing` to `cal.json`;
  - add IM fields to `RtImTrainConfig`: method, multiplier, floor.
- Expected scale from DIA-NN's residuals: `w_im` ~0.02-0.04 after calibration.
- Optional: IM2Deep fine-tuning on the seed anchors, with the same caveats as the
  DeepLC fine-tune (non-deterministic; judge it held out).

### P3 result (2026-09-23)

Scope agreed for this session: the `seed-pool` carry only. The non-linear or weighted CCS
fit and the IM2Deep fine-tune stay deferred. The reason is below: after P4 the gate
optimum sits at the calibrated window (multiplier 1.0), so IM accuracy may be the limit,
but no measurement yet separates it from the other losses.
- `seed-pool` now carries `observed_im` through `run` and `refresh_irt`, so a grouped run
  (`groups.window_groups > 1`) is IM-calibrated. Before, it silently was not. A v1 seed
  under a library with `predicted_im` now warns.
- Unit test: `observed_im` follows the kept row through the pool and the refresh.

### P4: IM-aware extraction (the main selectivity gain)

- Add `im: f32` to `Hit`, and an IM post-filter
  `im < im_lo[c] || im > im_hi[c] -> skip` beside every RT filter site in section 1,
  including demix and the two-pass path.
- Fragment traces and MS1 XICs (`sum_near`) sum only peaks inside the candidate's
  IM window. This cuts interference at the source, before any feature or score
  sees it.
- Fill `psms_extracted.apex_im`. Optionally write per-fragment IM apexes or a
  coarse mobilogram at the apex scan for P5.
- Seed: the same IM gate on the seed probe, reusing the wide pre-calibration window,
  so the seed benefits before calibration exists. Target: raise the seed's 130
  confident PSMs (section 2) to thousands, which also shrinks the 109 s p95 tail of
  the RT calibration.
- Measure:
  - IDs;
  - candidates through extract (expected to drop sharply, as with the multi-head RT
    calibration: "fewer candidates reach rescore and more of them are real");
  - the empirical decoy fraction;
  - entrapment before any promotion.

### P4 result (2026-09-23)

Implemented behind three keys, all default off:
- `extract.im_gate` = `off | fragments | fragments_ms1` (docs/09 section 6b): a hard gate
  on `run_windows.im_lo/im_hi` at all eight RT guard sites;
- `search_seed.im_gate` = `off | fixed | two_pass`, with `search_seed.im_window`
  (docs/07, "Ion-mobility gate");
- the window width is swept through the existing `rt_im_train.im_window_multiplier`.

`psms_extracted` v3 fills `apex_im` on 4D data. Against DIA-NN's observed IM on the 12,202
DIA-NN 1% precursors that extract accepts (ungated), the error is median 0.0038 and p95
0.034. No feature reads it yet (P5).

**Identity with the gates off.** The E. coli run with the P4 binary and
`config.local-tims-p2.json` (`output_mumdia_p4_off`) matches P2 on all 226,093 scored
rows: `score`, `q_value`, `peptide_q_value` and `precursor_q` are bit-identical.

**Seed gate** (standalone `search-seed` + `rt-im-train` on the P2 spectra and library,
`seed_arms_p4/`). Each cell compares the seed arm against DIA-NN on its 1% precursors:

| seed arm | confident PSMs | learned frag tol (ppm) | `w_im` | anchor IM error median / p95 | window recall |
|---|---|---|---|---|---|
| off | 4,313 | 20.4 | 0.0398 | 0.0021 / 0.0167 | 97.6% |
| fixed 0.06 | 3,865 | 20.4 | 0.0247 | 0.0019 / 0.0115 | 84.2% |
| fixed 0.10 | 4,287 | 20.5 | 0.0350 | 0.0020 / 0.0144 | 96.0% |
| fixed 0.15 | 4,326 | 20.6 | 0.0384 | 0.0020 / 0.0164 | 97.2% |
| two_pass 0.04 | 4,153 | 20.4 | 0.0290 | 0.0019 / 0.0123 | 92.0% |
| two_pass 0.06 | 4,316 | 20.8 | 0.0351 | 0.0020 / 0.0142 | 96.0% |
| two_pass 0.10 | 4,358 | 20.7 | 0.0384 | 0.0020 / 0.0160 | 97.2% |

- The seed gate does not raise the anchor count. P1's centroiding already raised it from
  130 to 4,313, so the target stated in the P4 plan is met without the gate.
- **A narrow seed gate biases the IM calibration.** It removes the anchors whose observed
  IM disagrees with the prediction, so the held-out residuals and `w_im` shrink while the
  window recall on true precursors falls (84% at fixed 0.06). Any seed IM gate must stay
  wider than the calibrated error.
- **The learned fragment tolerance does not move** (20.4-20.8 ppm), even though the
  calibrant lookup skips out-of-window peaks. Random matches from other mobility bands are
  therefore not what holds it at 20 ppm. The P6 item stays open with that hypothesis
  weakened.

**Extraction gate, end to end** (FASTA mode, 64 threads, `nn_torch` single seed; units of
section 2; `frag` = `im_gate: fragments`, `m` = `im_window_multiplier`):

| arm | precursors | peptides | protein groups | decoy fraction | extract accepted | never extracted | extracted, below 1% | window recall | wall | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| gates off (= P2) | 8,302 | 7,231 | 1,265 | 0.98% | 226,093 | 3,260 | 5,079 | 97.6% | 12:18 | 18.6 GB |
| frag, m 0.5 (`w_im` 0.020) | 7,625 | 6,841 | 1,284 | 0.98% | 168,468 | 3,906 | 4,809 | 79.4% | 12:35 | 18.7 GB |
| frag, m 0.75 (0.030) | 8,580 | 7,542 | 1,300 | 0.98% | 191,513 | 3,220 | 4,801 | 92.9% | 12:23 | 18.6 GB |
| **frag, m 1.0 (0.040)** | **8,733** | **7,665** | 1,319 | 0.98% | 202,970 | 3,137 | 4,787 | 97.6% | 12:08 | 18.6 GB |
| frag, m 1.5 (0.060) | 8,668 | 7,510 | 1,270 | 0.99% | 212,659 | 3,147 | 4,916 | 99.7% | 12:15 | 18.7 GB |
| frag, m 2.0 (0.080) | 8,546 | 7,403 | 1,275 | 0.99% | 217,649 | 3,170 | 4,999 | 99.9% | 13:01 | 18.6 GB |
| frag + MS1, m 1.0 | 8,702 | 7,618 | 1,336 | 0.98% | 202,970 | 3,137 | 4,824 | 97.6% | 12:11 | 18.7 GB |
| frag m 1.0 + seed two_pass 0.10 | 8,767 | 7,661 | 1,314 | 0.98% | 198,723 | 3,151 | 4,757 | 97.2% | 12:51 | 18.7 GB |

The ladder columns count DIA-NN-only peptides (none is missing from the library in any
arm). `bench/tims_compare.py --extracted --lib` prints them.

Reading:
- **The fragment gate at the calibrated width gives +6.0% peptides and +5.2% precursors at
  an unchanged 0.98% decoy fraction.** This is one `nn_torch` seed on one acquisition, so it
  is a measurement to confirm, not a default. It brings the native reader level with the
  msconvert P0 baseline (7,662 peptides), at 5x less wall time. The gap to DIA-NN is still
  0.50x in peptides.
- **The optimum is the calibrated window.** Narrower loses recall (m 0.5: 79% of DIA-NN
  precursors keep their IM inside the window, and 6,841 peptides). Wider readmits
  interference. The optimum tracking `w_im` means better IM calibration (P3 deferred
  levers) could shift it.
- **The hard gate is not too hard.** Peptides fall as the window widens, and at m 1.0 the
  never-extracted count falls rather than rises. The soft-weighting contingency of the
  plan is not triggered.
- **Extract accepts 10% fewer candidates, not the sharp drop expected.** P1's 2D
  centroiding already separates ions in mobility, so the gate removes other mobility bands'
  peaks from a candidate's traces but rarely empties a candidate.
- **MS1 gating** (-0.6%) and **the two-pass seed gate** (-0.05%) are within single-seed
  noise.
- Cost: none measurable. Wall and peak RSS are within run-to-run variation; MS1 per-peak
  IM is now always loaded on 4D data (about 170 MB here).

Before any default: the PYE diaPASEF acquisition, 3 `nn_torch` seeds, and entrapment
(section 4).

### P5: IM features

A new `features/im.rs` family, appended to `FAMILIES`. The registry is append-only;
update the `feature_sets_sized` assertion. Candidate features:
- signed and absolute IM error against the calibrated prediction, and |error| / `w_im`;
- intensity-weighted IM dispersion of the matched fragments at the apex. Co-eluting
  interferers differ in IM, while a true precursor's fragments share it;
- precursor MS1 IM error, and MS1-fragment IM agreement;
- fragment-versus-precursor mobilogram correlation at the apex (DIA-NN-style 2D
  peak evidence);
- nearest-conformer IM error, if P2 carries multiconformer CCS.

Rescore picks these up with no IM-specific code. Re-check the compact
`feature_preset`, which is library- and classifier-specific (CLAUDE.md), because it
was never selected with IM features present.

### P5 result (2026-09-23)

Implemented behind `features.im_features` (default off). The block is not a `FAMILIES`
entry: that registry is always on under Extended, so appending to it would change every
run's feature vector. It is appended after all other columns only when the key is on
(docs/10_features.md, "Ion-mobility features"). Nine features:
- from psms_extracted **v4**: `im_error`, `im_error_abs` (apex 1/K0 against
  `im_pred_cal`), `im_frag_mad` (intensity-weighted MAD of the apex fragment 1/K0),
  `ms1_im_error_abs`, `ms1_frag_im_diff`, `has_ms1_im` (the MS1 peak within tolerance
  nearest the fragment 1/K0);
- from chromatograms **v2** (a per-point 1/K0 list on every fragment trace, 4D only), over
  the elution peak: `im_elution_error_abs`, `im_elution_sd`, `im_elution_frag_sd`.

Extract always writes the v4 columns (null on 3D data) and, on 4D data, the v2 `im` list.
`Hit` carries the peak's 1/K0 as a u16 and stays 24 bytes. |err|/`w_im` was left out: within
one run it is a constant rescaling of |err|, so it matters only for a pooled rescore.

Not implemented:
- **Mobilogram correlation.** P1's 2D centroiding merges each ion's TIMS scans into one
  peak with one weighted 1/K0, so no per-ion mobility profile reaches extract. It needs
  convert to keep an IM profile or width per peak.
- **Nearest-conformer error.** Deferred with IM2Deep multiconformer output.

**Identity with the key off.** The P5 binary with the P4 best-arm config
(`output_mumdia_p5_frag_off`) matches `output_mumdia_p4_frag_m1.0` on all 202,970 scored
rows: `score`, `q_value`, `peptide_q_value` and `precursor_q` are bit-identical.

**Leakage checks.**
- Decoy against paired target `predicted_im` (IM2Deep on the reversed sequence), 1.13M
  pairs: median +0.0001, SD 0.036 V s cm^-2, per charge +0.0001 to +0.0002. The decoy
  predictions are not offset.
- Every feature has AUC 0.499-0.508 between targets and decoys scoring below the median
  decoy score, a population in which almost every target is false too
  (`bench/tims_im_features.py`, "null" column).

**Separation at the rescore input** (AUC of target against decoy; below 0.5 means lower
values are more target-like; "accepted" = targets at `q_value` <= 1% against all decoys):

| feature | gated, all | gated, accepted | ungated, all | ungated, accepted |
|---|---|---|---|---|
| `im_error_abs` | 0.487 | 0.368 | 0.483 | 0.289 |
| `im_frag_mad` | 0.481 | 0.247 | 0.476 | 0.187 |
| `ms1_im_error_abs` | 0.505 | 0.546 | 0.504 | 0.547 |
| `has_ms1_im` | 0.511 | 0.611 | 0.510 | 0.618 |
| `im_elution_error_abs` | 0.490 | 0.401 | 0.483 | 0.301 |
| `im_elution_sd` | 0.474 | **0.176** | 0.473 | **0.156** |
| `im_elution_frag_sd` | 0.475 | 0.196 | 0.474 | 0.162 |

- The dispersion features separate far more than the IM error: a true precursor's
  fragments share one mobility over the whole peak, while a decoy's matches do not.
  `im_elution_sd` is the strongest single IM feature in both arms.
- As expected, the gate bounds the error features (`im_error_abs` 0.368 gated against 0.289
  ungated), but it leaves most of the dispersion signal.
- Over all targets the AUC stays near 0.5, because most targets in the pool are false.
- The MS1 features are weak: 48% of rows have an MS1 peak within tolerance.

**End to end** (FASTA mode, 64 threads; units of section 2; 3 paired `nn_torch` seeds
(0, 1, 2) per arm, seeds 1-2 by re-running `rescore` on each arm's `psms_competed`;
`seeds_p5/`). Mean over seeds, with the seed range for peptides:

| arm | precursors | peptides (range) | protein groups | decoy fraction |
|---|---|---|---|---|
| gates off (P4 off) | 8,273 | 7,201 (7,170-7,231) | 1,266 | 0.98-0.99% |
| gates off + IM features | 8,404 (+1.6%) | 7,309 (7,283-7,338), **+1.5%** | 1,291 (+2.0%) | 0.97-0.99% |
| frag m 1.0 (P4 best) | 8,703 | 7,615 (7,555-7,665) | 1,315 | 0.98% |
| frag m 1.0 + IM features | 8,840 (+1.6%) | 7,753 (7,732-7,767), **+1.8%** | 1,344 (+2.2%) | 0.98% |

Loss ladder (seed 0, DIA-NN-only peptides; `tims_compare.py`): gated, "extracted, below 1%"
4,787 -> 4,730, "never extracted" unchanged at 3,137; ungated, 5,079 -> 5,020, and
3,260 unchanged. The features move only the rescore side, as they should.

Cost: extract 16 s in both arms, features +1.0 s (6.5 against 5.4 s), wall 12:04 against
12:09, peak RSS 18.6 GB unchanged. chromatograms.parquet +38% (214 -> 295 MB gated,
250 -> 350 MB ungated), the per-point `im` list.

Reading:
- **+1.8% peptides on top of the P4 gate, at an unchanged decoy fraction.** Each IM seed
  scores above every baseline seed in both pairs. The gain is real on this file, but small.
  The gate itself is still worth +6.1% with the IM features on (7,753 against 7,309).
- **Strong univariate separation, small classifier gain.** Accepted targets are already
  well separated by the existing features, so the IM evidence is largely redundant for
  them. Where IM would matter (the 4,730 DIA-NN peptides below 1%), it adds only about 60.
  The gap to DIA-NN stays at 0.50x in peptides.
- The gap is therefore not explained by missing IM evidence at the rescore input. The
  remaining levers are on the extraction and scoring side: fragment tolerance (P6, still
  20.4 ppm), library fragment intensities for timsTOF (MS2PIP `timsTOF2024`, P6), and 2D
  (RT x IM) peak picking, which needs an IM profile per peak.
- `rescore.feature_preset = compact` excludes the IM block, because the compact list was
  selected without it.

Before any default: the PYE diaPASEF acquisition and entrapment (section 4). The 3 seeds
are done for this file.

### P6: timsTOF-specific tuning

Each item is measured with seeds on two diaPASEF acquisitions:
- the MS2PIP `timsTOF2024` model against HCDch2;
- the AlphaPeptDeep `timsTOF` instrument against both;
- fragment and precursor ppm tolerances from the observed error distribution. The
  default is 20 ppm; DIA-NN's median MS2 error on this file is 2.5 ppm and its optimised
  MS2 tolerance 8 ppm, so 20 ppm may be loose;
- `top_n_fragments`, `presence_min_fragments` (fewer fragments per slot after IM
  gating);
- RT window behaviour and `apex_count_window` on a 15-min gradient with ~7 points
  per peak;
- the extraction gate (`gate_min_score`), whose optimum must be re-derived after P4.

### P6 result (2026-09-24)

Base: the P5 arm (gate `fragments`, `im_window_multiplier` 1.0, `im_features` on), whose
3-seed mean is 8,840 precursors / 7,753 peptides / 1,344 protein groups. Every change below is
a documented setting in a local config, or behind a new key that defaults to off. None is
promoted. The E. coli file only; PYE and entrapment are still open.

Method. Library-model arms are full runs. Tolerance and extract arms re-run
`extract -> features -> compete -> rescore` on an existing run's spectra, library,
`run_windows` and masscal (`p6/chain.sh` in the data directory). Replayed on the P5 run, that
chain reproduces `psms_scored` bit for bit. Seeds 1-2 re-run `rescore` on each arm's
`psms_competed`. Units as in section 2. The ladder counts DIA-NN-only peptides. Every arm has
a 0.98-0.99% empirical decoy fraction and 0 peptides missing from the library.

**1. Library intensity model** (full runs, P5 binary, tolerance as learned):

| arm | precursors | peptides (range) | PGs | never extracted | extracted, below 1% | seed confident | learned frag tol |
|---|---|---|---|---|---|---|---|
| P5 base: HCDch2, charge-2 fragments on | 8,840 | 7,753 (7,732-7,767) | 1,344 | 3,137 | 4,730 | 4,313 | 20.4 |
| A: HCDch2, charge-1 only (seed 0) | 8,091 | 7,131 | 1,273 | 3,536 | 4,881 | 3,559 | 20.9 |
| **B: `timsTOF2024`, charge-1 only** | 8,974 | 7,959 (7,947-7,983), **+2.7%** | 1,368 | 3,297 | 4,340 | 3,587 | 21.8 |

- MS2PIP `timsTOF2024` predicts singly charged b/y only. Under the default
  `charge2_from_precursor_charge` 2 the library would request charge-2 fragments, and
  `ms2pip_values` would fill them with the native heuristic, which crowds the top-N (the
  CLAUDE.md `ms2pip_model` note). `charge2_from_precursor_charge: 99` requests charge 1 only,
  so that path is never reached. No code change is needed.
- DIA-NN's library for this run holds only charge-1 fragments (211k rows, for z2, z3 and z4
  precursors alike). The HCDch2 top-12 was 12.6% / 26.3% / 29.3% charge-2 for z2 / z3 / z4.
- Arm A isolates the charge-2 removal: it costs 8.0% with HCDch2. The model is therefore
  worth about +11% over HCDch2 at charge 1 only (seed 0: 7,947 against
  7,131), and it more than pays for the lost
  charge-2 fragments. It is also the only arm whose seed calibrates on fewer anchors than it
  identifies more with: 3,587 confident seed PSMs against 4,313.
- Library build time was not measured cleanly. Another job loaded the host (load average
  ~50) during both library arms.

**2. Fragment tolerance** (on arm B). Fixed tolerance through extract, offset kept at
-0.46 ppm, seed 0:

| tol (ppm) | 6 | 8 | 10 | 12 | 15 | 18 | 21.8 (learned) |
|---|---|---|---|---|---|---|---|
| peptides | 7,814 | 8,121 | 8,335 | 8,436 | 8,453 | 8,212 | 7,947 |
| extract accepted | 102,323 | 125,766 | 143,191 | 156,455 | 171,990 | 182,944 | 194,456 |
| never extracted | 4,433 | 3,922 | 3,570 | 3,443 | 3,268 | 3,278 | 3,297 |
| extracted, below 1% | 3,369 | 3,591 | 3,744 | 3,776 | 3,950 | 4,132 | 4,340 |

3 seeds: 12 ppm 9,528 / 8,437 (8,382-8,492) / 1,447; 15 ppm 9,461 / 8,350 / 1,392.
- The deviation distribution explains why the learned value stayed at ~20 ppm. See
  docs/07_search_seed.md, step 6. In short, there is no uniform random-match floor (~2.5% of
  deviations lie beyond ±20 ppm). The p95 is set by a 6-20 ppm shoulder of weak-peak
  centroid error, while the MAD (2.1 ppm) follows the core.
- DIA-NN settled on 8 ppm for MS2 on this file ("Optimised mass accuracy: 8 ppm"; 2.5 ppm is
  its median error). For MuMDIA 8 ppm is too tight (-3.7% against 12 ppm), because it
  starts to lose extraction (never extracted 3,922).
- The new key **`search_seed.frag_tol_mad_k`** (default 0, the p95 rule, bit-identical)
  sizes the tolerance as `k * 1.4826 * MAD`. With `k = 4` the orchestrated run learns
  12.54 ppm and gives 9,526 / **8,412** (8,369-8,437) / 1,440, which is **+8.5% peptides
  over the P5 base**. Wall 12:21, peak RSS 17.1 GB.
- At 12 ppm these were neutral (seed 0): `search_seed.mass_cal_loess` (8,448), MS1
  `prec_tol_ppm` 10 in extract and features (8,433), and `im_window_multiplier` 1.25
  (8,439). 0.75 lost (8,226), so the P4 optimum holds at the new tolerance.

**3. Extract knobs** (on B + `frag_tol_mad_k` 4, fast loop, seed 0, base 8,437):

| arm | precursors | peptides | PGs | extract accepted | never extracted | extracted, below 1% |
|---|---|---|---|---|---|---|
| base (gate 0.2, `apex_count_window` 5) | 9,528 | 8,437 | 1,446 | 159,422 | 3,411 | 3,804 |
| `presence_min_*` 2 / 4 | 9,555 / 9,530 | 8,445 / 8,426 | 1,416 / 1,432 | ~159k | 3,411 / 3,413 | 3,795 / 3,806 |
| `gate_min_score` 0.4 | 8,721 | 7,815 | 1,396 | 94,994 | 5,007 | 2,809 |
| `gate_min_score` 0.1 | 9,863 | 8,657 | 1,456 | 200,538 | 2,690 | 4,298 |
| `gate_min_score` 0.05 | 10,073 | 8,863 | 1,472 | 224,281 | 2,341 | 4,481 |
| **`gate_min_score` 0 (off)** | 10,740 | 9,369 | 1,466 | 504,608 | 203 | 6,136 |
| `apex_count_window` 1 / 3 | 9,168 / 9,644 | 8,135 / 8,493 | 1,404 / 1,427 | 169,809 / 161,321 | 3,336 / 3,269 | 4,148 / 3,864 |
| gate 0, `presence_min_*` 4 / 5 | 10,902 / 10,929 | 9,396 / 9,459 | 1,474 / 1,491 | ~503k | 206 / 218 | 6,086 / 6,035 |
| **gate 0, `apex_count_window` 3** | 11,136 | 9,570 | 1,468 | 504,608 | 203 | 5,943 |
| gate 0, acw 3, `rt_window_multiplier` 0.75 / 1.5 | 10,716 / 11,152 | 9,309 / 9,560 | 1,482 / 1,454 | 480,714 / 537,475 | 274 / 127 | 6,145 / 6,019 |

3 seeds: gate 0.05: 10,064 / 8,863 / 1,464. **Gate 0: 10,848 / 9,397 (9,365-9,458) / 1,484.**
**Gate 0 + `apex_count_window` 3: 11,037 / 9,507 (9,460-9,570) / 1,472.**

- **The spectral-agreement gate is the binding extraction loss on this file.** At the
  default 0.2 it rejects most of the 3,411 never-extracted DIA-NN peptides: at 0 only 203
  are never extracted. The presence thresholds do not bind, either at gate 0.2 or at 0.
  The gate's HYE optimum (docs/28 section 18) does not transfer to a 15-min diaPASEF
  gradient at 50 ng, where the apex spectrum rests on few points.
- The cost is volume: 3.2x the candidates (504,608), extract 35 against 12 s, and rescore
  175 against 50 s in the fast loop. Classifier memory stays small at this scale.
- **Gate 0 is a loosened pool, so the decoy fraction alone does not validate it.** One
  weak check is consistent with a real gain: of the ~930 peptides that gate 0 adds on
  seed 0, 876 are DIA-NN 1% peptides, and MuMDIA-only peptides go from 248 to 303. It needs
  entrapment before anything is claimed.
- `apex_count_window` 3 (DIA-NN reports an FWHM of 3.1 scans here) adds +1.2% peptides at
  gate 0 over 3 seeds, at the edge of noise. Window 1 loses 3.6%. The RT window multiplier
  optimum stays at 1.0.

**Summary, 3-seed means against the P5 base** (7,753 peptides):

| step | precursors | peptides | PGs | vs P5 |
|---|---|---|---|---|
| + `timsTOF2024`, charge-1 fragments | 8,974 | 7,959 | 1,368 | +2.7% |
| + `frag_tol_mad_k` 4 (12.5 ppm) | 9,526 | 8,412 | 1,440 | +8.5% |
| + `gate_min_score` 0 | 10,848 | 9,397 | 1,484 | +21.2% |
| + `apex_count_window` 3 | 11,037 | 9,507 | 1,472 | +22.6% |
| DIA-NN 2.5.0 | 18,673 | 15,404 | 1,919 | |

MuMDIA / DIA-NN moves from 0.50x to **0.62x** in peptides. The loss ladder at the last row
(seed 0) is 203 never extracted and 5,943 extracted but below 1%. The remaining gap is now
almost entirely on the scoring side.

**Orchestrated check** (`output_mumdia_p6_best`, the last row's config, one `mumdia run`, P6
binary). The run learns 12.54 ppm. Its `psms_scored` is bit-identical to the fast-loop arm
(504,608 rows): 11,136 / 9,570 / 1,468 on seed 0, 9,258 peptides shared with DIA-NN and 311
MuMDIA-only. Cost against the P5 arm, per stage: extract 21 against 16 s, features 12
against 7 s, compete 11 against 5 s, rescore 107 against 47 s. The chain from convert to
report takes 4.5 against 2.6 min. chromatograms.parquet is 461 against 282 MB and
features.parquet 1,136 against 469 MB. Peak RSS is 17.2 GB (18.6 GB for P5). The total wall
of 27:39 is not comparable, because the host was loaded (load average 50-70) during library
prediction.

**Entrapment (2026-09-24), the last row's config.** One full run of the E. coli file against
E. coli + 1:1 human (`bench/make_entrapment_fasta.py`, seed 20260828, `REAL_`/`ENTRAP_`
prefixes; marker `ENTRAP_`, exclude `REAL_`, contaminant markers `KRT`, `K1C`, `K2C`, `ALBU`,
`TRYP`). `entrapment_ratio` was measured from the built library: 670,590 real against 1,190,193
entrapment target candidates = 0.563430. The run config carries no entrapment keys, so the
accepted set is the ordinary `nn_torch` one. `bench/entrapment_fdp.py`, peptide level:

| seed | real peptides at 1% | spike-in peptides | empirical FDP | decoy fraction |
|---|---|---|---|---|
| 0 | 8,705 | 53 | 0.355% | 0.99% |
| 1 | 8,790 | 56 | 0.370% | 0.98% |
| 2 | 8,612 | 56 | 0.378% | 0.99% |

- The FDP is below the 1% threshold in every seed, so the P6 settings, gate 0 included,
  show no FDR inflation on this acquisition. The estimate rests on 53-56 spike-ins, so it
  excludes gross inflation, not small differences.
- Real peptides are 8.9% below the E. coli-only search (9,570 on seed 0), which is the cost
  of doubling the search space. That is expected and not an FDR effect.
- Wall 32:19, peak RSS 23.8 GB; outputs in `/public/local/MuMDIA_entrap`.

Before any default: a second acquisition with 3 seeds. Entrapment is done (above).

### ProteoBench HYE diaPASEF (2026-09-24)

Module `quant_lfq_DIA_ion_diaPASEF`: six `ttSCP_diaPASEF_Condition_{A,B}_Sample_Alpha_0{1,2,3}`
runs from `/public/local/ProteoBench/HYE_diaPASEF`, `ProteoBenchFASTA_MixedSpecies_HYE.fasta`
(14.7M precursors with decoys), one `mumdia run` with six `--mzml` (pooled rescore, per-run
quant, MBR off). Scored offline with proteobench 0.18.4 (`bench/pb_eval.py`, Custom format);
nothing uploaded. Work tree and submission files: `/public/local/ProteoBench/HYE_diaPASEF_mumdia`
(`<arm>/proteobench/custom_input.tsv`).

**Pre-roadmap (`8ea2b3d`, msconvert, HCDch2, the P0 config): no result.**
- The seed finds 0 confident PSMs on the IM-collapsed spectra (130 on the E. coli file), so the
  DeepLC multi-head calibration aborts with no anchors (`p0_failed_multihead/`).
- With `multihead_calibration: 0` the run continues with unbounded RT windows. Extract then
  reaches 607 GB RSS within minutes and the host's earlyoom kills it (`p0_oom/`).
  `extract.windows_in_flight: 2` does not bound it (639 GB).
- The msconvert conversion alone takes 52 min per file (30 GB mzML each, six in parallel).

**P6 (the `output_mumdia_p6_best` config).** One run of 4:32 wall, 71 GB peak RSS, 380 GB of
artifacts. Library 1:44, per-run chain about 17 min, pooled rescore 45 min over 42.5M PSMs.
- 54,895 stripped peptides at 1% (pooled `peptide_q_value`, decoy fraction 0.99%), 9,070
  protein groups, 45.8-47.0k PSMs per run at `run_psm_q` 1%.
- `frag_tol_mad_k: 4` learns 18.4-18.5 ppm on these runs (12.5 on the E. coli file).
- Gate 0 accepts 7.1M candidates per run, against 2.2M at 0.2.

ProteoBench at `min_obs` 3 (expected log2 A/B: E. coli -2, yeast +1, human 0):

| arm | ions | median abs eps | species-equalised | CV median | E. coli | yeast | human |
|---|---|---|---|---|---|---|---|
| P6 | 47,309 | 0.222 | 0.549 | 0.122 | -1.03 | +0.55 | +0.06 |
| P6, `gate_min_score` 0.2 | 42,968 | 0.219 | 0.543 | 0.117 | -1.03 | +0.56 | +0.05 |
| P6, quant `fragment_selection: predicted` | 47,309 | 0.202 | 0.390 | 0.122 | -1.39 | +0.68 | +0.05 |
| P6, quant `interference_envelope` | 47,309 | 0.198 | 0.458 | 0.111 | -1.22 | +0.62 | +0.05 |
| **P6, quant both** | 47,309 | **0.182** | **0.307** | 0.114 | **-1.57** | +0.76 | +0.04 |
| DIA-NN 2.5.0, no MBR (public datapoint) | 98,694 | 0.125 | 0.180 | 0.089 | -1.79 | +0.84 | |
| Spectronaut 21 (public) | 148,030 | 0.198 | 0.302 | 0.148 | -1.69 | +0.78 | |
| AlphaDIA 1.12.1, no MBR (public) | 61,982 | 0.182 | 0.259 | 0.113 | -1.74 | +0.79 | |

Reading:
- **Gate 0 costs no accuracy here.** Against gate 0.2 it adds 10% ions (and 4.8% peptides) at
  the same ratios, so the compression is not caused by the loosened pool.
- **Default quant compresses the ratios strongly**, and at every intensity (E. coli -0.95 in
  the brightest quintile). It concentrates in low-scoring identifications (E. coli -0.61 in the
  lowest score quintile, -1.50 in the highest). Ranking quant fragments by observed area picks
  the interfered ones on these traces: ranking by library intensity plus the interference
  envelope, two existing default-off keys, moves E. coli from -1.03 to -1.57 on the same ions.
  The residual gap to DIA-NN's -1.79 is still open.
- Completeness is 0.48x DIA-NN's no-MBR datapoint in ions.

### P7: Ion-mobility peak shape (2D RT x IM evidence)

Why this is next. After P6 the E. coli loss ladder is 203 never extracted against 5,943
extracted but below 1%: the remaining gap is scoring. P5 showed that the IM evidence MuMDIA
has (one intensity-weighted 1/K0 per centroid, the per-point 1/K0 of each trace) is largely
redundant with the existing features. What it lacks is the evidence DIA-NN scores on: the
shape of each ion's mobility profile, and whether a candidate's fragments share the
precursor's mobilogram at the apex (Demichev et al. 2022). P1's 2D centroiding
(`convert/tdf.rs`, `centroid_2d`) collapses each ion's TIMS scans into one peak, so no stage
downstream can compute it.

Scope:
- **Convert: keep a per-peak mobility profile.** `centroid_2d` already holds each cluster's
  raw `(tof, scan, intensity)` points. The cheapest carrier is a per-peak IM width (the
  intensity-weighted SD of the scans, or a FWHM) next to the existing `im`; a richer one is a
  compact profile (for example intensities in a few fixed scan bins around the centroid) for
  MS2 and MS1. Choose on measured artifact size and on what the feature needs. This is a
  spectra schema bump (v3) with v2 artifacts still readable.
- **Extract: carry it to the apex.** Beside the per-point `im`, keep the width or profile of
  the peak each hit came from, at least at the apex scan (chromatograms or psms_extracted
  schema bump).
- **Features: mobilogram agreement.** Fragment-versus-precursor (MS1) mobilogram correlation
  at the apex, fragment-versus-fragment profile agreement, and width agreement with the
  expected peak width. Behind a new key that defaults to off, appended after every other
  column as `features.im_features` is (docs/10_features.md), so the default feature vector
  and scores stay bit-identical.
- Optionally, if the profile proves informative: split centroids at minima of the smoothed
  mobility profile (the `ponytail:` upgrade path in `centroid_2d`), so that two ions of one
  m/z whose profiles touch are not merged.

Measure on the E. coli file against the P6 configuration (`output_mumdia_p6_best`):
- univariate AUC on accepted targets against decoys, and the low-score null
  (`bench/tims_im_features.py`);
- IDs at 1%, the loss ladder (the "extracted, below 1%" group should shrink), decoy fraction,
  3 paired `nn_torch` seeds;
- convert/extract wall and RSS, and artifact size;
- then entrapment (`bench/make_entrapment_fasta.py`, `bench/entrapment_fdp.py`) and the
  ProteoBench HYE diaPASEF set before any default.

### P7 result (2026-09-24)

Scope as agreed: a per-peak mobility width as the carrier (not a binned profile), apex-only
evidence (no `Hit` or chromatogram change), centroid splitting deferred. Two keys, both
default off:
- `convert.tdf_im_width`: `centroid_2d` also writes each centroid's width, the
  intensity-weighted SD of its TIMS scans plus 1/12 scan^2 for quantisation, in 1/K0 via the
  local calibration slope (spectra **v3**, `im_width` list, MS1 and MS2; docs/04).
- `features.im_shape_features`: six features after the P5 block (docs/10), from
  psms_extracted **v5** columns that extract's apex post-pass fills from the same fragment
  and MS1 peaks as `apex_im`. Each peak is a Gaussian (centroid, width); agreement is the
  Bhattacharyya coefficient: fragment width and its spread, fragment-versus-consensus
  overlap, MS1 width, MS1-versus-fragment overlap and the MS1/fragment width log ratio.

Outputs are in `/public/compomics2/Robbe/MuMDIA_data/p7` (`off`, `on`, `seeds/`). Both arms
are full runs of the P7 binary on the P6 base config, one after the other on a quiet host
(load average ~2).

**Identity with the keys off.** `p7/off` matches `output_mumdia_p6_best` on all 504,608
scored rows (`score`, `q_value`, `peptide_q_value`, `precursor_q` bit-identical), and the
spectra artifacts have the same byte size.

**Widths on this run** (1/K0 SD; one scan is ~0.00087 V s cm^-2):

| | p5 | p25 | median | p75 | p95 |
|---|---|---|---|---|---|
| MS2 peaks (scans) | 1.1 | 4.8 | 9.4 | 14.4 | 26.9 |
| MS1 peaks (scans) | 1.6 | 6.9 | 13.2 | 23.7 | 54.0 |

On the 7,305 accepted targets with both widths, the apex fragment width has a median of
0.0112 and the MS1 width 0.0202 (ratio 1.6, Spearman 0.23). One ion at TIMS resolution is
about 0.007. The MS1 tail shows centroids that merge several ions at the 30-scan gap: the
failure the `ponytail:` note in `centroid_2d` names.

**Separation** (`bench/tims_im_features.py`, seed 0; below 0.5 means lower is more
target-like):

| feature | all | accepted | null |
|---|---|---|---|
| `im_frag_width` | 0.495 | 0.411 | 0.498 |
| `im_frag_width_mad` | 0.501 | 0.496 | 0.499 |
| `im_frag_overlap` | 0.508 | **0.734** | 0.496 |
| `ms1_im_width` | 0.505 | 0.596 | 0.499 |
| `ms1_frag_overlap` | 0.509 | 0.677 | 0.500 |
| `ms1_frag_width_logratio` | 0.504 | 0.585 | 0.498 |
| `im_elution_sd` (P5, for reference) | 0.488 | 0.200 | 0.502 |

No leakage: every feature is 0.496-0.509 in the low-score null. The MS1 features partly
encode MS1 presence (`has_ms1_im` 0.600), because a missing MS1 peak gives 0.

**End to end** (units of section 2; 3 paired `nn_torch` seeds; P6 seeds 1-2 from
`p6/seeds/g0_acw3_s{1,2}`, the fast-loop arm that is bit-identical to the P6 run):

| arm | precursors | peptides (range) | PGs | decoy fraction |
|---|---|---|---|---|
| P6 base | 11,037 | 9,507 (9,460-9,570) | 1,472 | 0.98-0.99% |
| + width + shape features | 10,867 (-1.5%) | 9,417 (9,193-9,542), **-0.9%** | 1,484 (+0.8%) | 0.98-0.99% |

Loss ladder (seed 0, DIA-NN-only peptides): never extracted 203 against 203, extracted but
below 1% 5,960 against 5,943.

**Cost.** Convert 44.6 against 38.3 s; extract 22.0 against 20.4 s; features 402 against
396 columns; wall 13:16 against 13:17; peak RSS 17.17 against 17.14 GB (the peak is not in
convert or extract). Spectra 1,656 against 1,206 MB (+37%), psms_extracted 59.0 against
43.2 MB.

Reading:
- **No gain.** The block is -0.9% in peptides over 3 seeds, within the seed spread (seed 2
  alone is -2.8%), and it does not move the loss ladder. Entrapment was not run, because
  there is no gain to validate.
- **The width evidence is real but redundant.** `im_frag_overlap` separates accepted
  targets (0.734), but less than P5's `im_elution_sd` (0.200), which already measures
  fragment mobility agreement over the whole peak. A single apex scan at 50 ng carries few
  counts per fragment, so the apex width adds little to the elution-wide centroid spread.
- **The widths are inflated by merging, MS1 most.** The MS1/fragment ratio of 1.6 and the
  54-scan MS1 p95 mean that the MS1 comparison measures the merge of neighbouring ions
  more than the precursor's own profile. Splitting centroids at mobility-profile minima is
  therefore a precondition for any profile-agreement feature, not an optional extra.
  Measure it (peaks per spectrum, seed anchors, the width distribution above, and IDs)
  before re-testing this block.
- None of the P7 keys is a candidate for promotion.

The next steps, from a precursor-level loss analysis against DIA-NN, are in
[TIMS_ROADMAP_bis.md](TIMS_ROADMAP_bis.md).

### P8: Performance

- Memory: `Peak` is 24 B, and the IM slot is already reserved. Per-peak IM adds
  4 B to the artifacts.
- Parallel frame decode, and the effect of the noise floor on peak volume.
- Window-group sharing (`docs/33`) with the (m/z, IM) key.
- Compare wall time and peak RSS against DIA-NN (2:11 and 20.7 GB on this file,
  section 2).

### Later: FDR validity, quantification, runtime

- **FDR:** entrapment on diaPASEF. Build an E. coli + foreign-proteome FASTA with
  paired decoys, and report the empirical FDP at 1% (`bench/entrapment_fdp.py`).
  This is a required gate for P4/P5 defaults.
- **Quantification:** the ProteoBench `quant_lfq_DIA_ion_diaPASEF` module on the
  PYE diaPASEF 6+6 runs (`bench/pb_eval.py`). IM-gated integration should lower
  CV and ratio bias.
- **Runtime:** match DIA-NN wall time on single runs and on a pooled experiment.

## 5. Risks and open questions

- **Slot separation in the msconvert path (resolved).** msconvert's combined output
  keeps the three slots of a frame as separate spectra, and on this scheme each slot
  has its own m/z range, so the m/z-only window key did not merge slots. That holds
  only while no two slots share an m/z range. Schemes that tile one m/z range over
  several IM ranges (for example synchro-PASEF, or overlapping window groups) would
  merge, so P1 must still key windows on (m/z, IM).
- **Build-toolchain boundary.** The TDF reader brings bundled SQLite (C). Decide
  whether this is acceptable, or whether it goes behind a feature.
- **IM2Deep coverage.** The training data covers charge states and modifications
  unevenly. Check calibrated IM residuals per charge, and keep a wide-window
  fallback when a prediction is missing.
- **Candidate-per-slot overlap.** A precursor near a slot boundary can appear in two
  adjacent slots or window groups. Extraction must not double-count it, and
  competition must see one row.

## References

- Meier F. et al. diaPASEF: parallel accumulation-serial fragmentation combined
  with data-independent acquisition. *Nat. Methods* 17, 1229-1236 (2020).
- Demichev V. et al. dia-PASEF data analysis using FragPipe and DIA-NN for deep
  proteomics of low sample amounts. *Nat. Commun.* 13, 3944 (2022).
  https://www.nature.com/articles/s41467-022-31492-0
- Wallmann G. et al. AlphaDIA enables DIA transfer learning for feature-free
  proteomics. *Nat. Biotechnol.* (2025).
  https://www.nature.com/articles/s41587-025-02791-w
- Declercq A. et al. IM2Deep. *J. Proteome Res.* (2025); Devreese R. et al.
  Collisional cross-section prediction for multiconformational peptide ions with
  IM2Deep. *Anal. Chem.* 97, 15113 (2025). https://github.com/compomics/IM2Deep
- timsrust (MannLabs), Rust reader for Bruker TDF, used by Sage.
  https://github.com/mannlabs/timsrust
- Mason E. A., McDaniel E. W. *Transport Properties of Ions in Gases*
  (Wiley, 1988), for the Mason-Schamp relation between CCS and reduced mobility.
