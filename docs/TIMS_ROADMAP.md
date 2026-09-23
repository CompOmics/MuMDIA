# TIMS roadmap: diaPASEF support at DIA-NN parity

Status: plan, 2026-09-23. Code references are to `8d3db2e`. Branch `IM`. The objective order is fixed: identification
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
| features / rescore | no IM family among the 16 registered families | `stages/features.rs:55` (`FAMILIES`) |

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

**MuMDIA 0.4.0** (`8d3db2e`), FASTA mode, from `configs/examples/fasta-sidecars.json`:
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
  CCS for a precursor. Start with the main conformer. Treat the rest as P5
  candidates (a nearest-conformer IM error), not as separate library rows.
- The DIA-NN importer maps `IonMobility` to `predicted_im`. That gives a zero-cost
  comparison arm for isolating library IM quality from the engine.

### P3: IM calibration (rt-im-train)

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

### P6: timsTOF-specific tuning

Each item is measured with seeds on two diaPASEF acquisitions:
- the MS2PIP `timsTOF2024` model against HCDch2;
- the AlphaPeptDeep `timsTOF` instrument against both;
- fragment and precursor ppm tolerances from the observed error distribution. The
  default is 20 ppm; DIA-NN reports a median MS2 mass accuracy of 2.5 ppm on this
  file, so 20 ppm may be loose;
- `top_n_fragments`, `presence_min_fragments` (fewer fragments per slot after IM
  gating);
- RT window behaviour and `apex_count_window` on a 15-min gradient with ~7 points
  per peak;
- the extraction gate (`gate_min_score`), whose optimum must be re-derived after P4.

### P7: Performance

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
