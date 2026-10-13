# 35. diaPASEF: timsTOF input with ion mobility
> Part of the MuMDIA developer documentation (see docs/README.md).

This document describes how MuMDIA reads and searches diaPASEF data from a Bruker timsTOF.
It is filled in as the support lands, one pull request at a time. A section that names a later
pull request describes work that is not in the tree yet.

## 1. What diaPASEF support consists of

A diaPASEF run adds one separation to DIA: the trapped ion mobility (TIMS) dimension, reported as
the reduced ion mobility 1/K0 (V s cm^-2). The quadrupole steps through its isolation windows as
the TIMS scans, so each window ("slot") covers an m/z range and a 1/K0 range, and every fragment
peak carries a 1/K0. Using that dimension takes seven layers:

1. read the `.d` natively and keep the 1/K0 of every peak (section 2, in the tree);
2. tell apart the windows that share an m/z range and differ in 1/K0 (section 3, in the tree);
3. predict and calibrate the 1/K0 of each precursor (section 4, in the tree);
4. gate and score candidates with 1/K0 (section 5, in the tree);
5. rebuild traces from the raw TIMS events and choose the apex jointly in retention time and
   1/K0 (section 6);
6. extend retention-time recalibration and match-between-runs with a 1/K0 window (section 7);
7. quantify with cross-run fragment consistency (section 8).

Until a layer lands, the 1/K0 columns it would read are written but ignored. A run on a diaPASEF
`.d` works end to end after layers 1 and 2: the seed search, extraction and features see the
4D spectra artifacts and use only their m/z, intensity and retention time. Layer 4 adds the
1/K0 evidence to every 4D run and makes the gate and the feature family available behind keys
that default to off.

The one predicate for 4D behaviour is "the spectra carry per-peak 1/K0" (`Reader::has_im` in
`crate::instrument`). After conversion it is read from the record `convert` writes
(`params.reader = "timsrust"`, `instrument::spectra_has_im`), never from the instrument model:
a timsTOF mzML converted by msconvert has no 1/K0 and is a 3D run. Before conversion, the
orchestrators use the input routing (`instrument::input_has_im`, from `raw::Vendor::BrukerTdf`).
`convert` chooses its reader through the same function (`instrument::reader_for_input`), so the
two answers cannot disagree. The settings that depend on it resolve from one of the two:
library-build keys from the input, before anything is converted, and a stage that reads
converted spectra from the record.

## 2. Conversion

### Routing

`raw::detect` reads a Bruker `.d` natively only when it holds diaPASEF frames that timsrust
decodes (`raw::is_native_diapasef`):

- `analysis.tdf` and `analysis.tdf_bin` both exist;
- the `Frames` table has diaPASEF frames (`MsMsType` 9) and no ddaPASEF frames (8), which is
  the rule timsrust classifies by;
- `GlobalMetadata.TimsCompressionType` is 2.

The check is a read-only SQLite query of the metadata and never touches the frame data. Such a
`.d` is `Vendor::BrukerTdf`, which needs no conversion: `raw::ensure_mzml` returns the path
unchanged and `convert` decodes it. Every other Bruker `.d` (BAF, ddaPASEF, PRM, an older
compression, anything unrecognised) keeps the msconvert route with
`--combineIonMobilitySpectra`, and only those files get the warning that ion mobility is
discarded. There is no automatic msconvert retry after a native decode failure: the error names
the file, and converting it to mzML by hand is the way around it. The desktop application applies
the same rule, so it neither asks for msconvert nor reserves disk space for an mzML that is never
written.

msconvert is not used for diaPASEF because it collapses the mobility dimension, and because it
failed in practice on this data: on the six-run HYE diaPASEF set the msconvert route found no seed
anchors and then ran out of memory, and on the E. coli run it took 45:30 against 8:45 for the
native reader.

### The native reader

`stages::convert::tdf` reads frames with timsrust 0.4.2 (`tdf` feature only):

- **Calibration.** TOF index to m/z uses timsrust's converter. TIMS scan to 1/K0 uses Bruker's
  `TimsCalibration` model 2 when present (`ScanImCalibration`), because timsrust's linear
  interpolation between the acquisition range bounds is off by up to 0.030 1/K0 at the
  high-mobility end of the Ultra 2 benchmark run, the width of a calibrated 1/K0 window. Without a
  model 2 row the linear interpolation is used, with a warning. Only the first calibration row is
  used; a file with several logs a warning.
- **Spectra.** An MS1 frame is one spectrum over all of its scans. An MS2 frame becomes one
  spectrum per diaPASEF slot, with the slot's m/z bounds (isolation m/z plus or minus half the
  isolation width) and its 1/K0 bounds (the calibrated 1/K0 of its first and last scan). The
  spectrum id is `frame=<i> slot=<k>`.
- **Centroiding.** Within a spectrum the raw (TOF, scan, intensity) points are centroided in
  m/z x mobility: points join one m/z trace while each neighbour is within `convert.tdf_mz_ppm`
  (10) of the previous one; a trace splits where two consecutive points are more than
  `convert.tdf_im_gap_scans` (30) scans apart; a cluster of fewer than `convert.tdf_min_points`
  (2) raw points is dropped. A peak is the cluster's summed intensity at its intensity-weighted
  m/z and 1/K0. The result does not depend on the input order.
- **Valley splits.** With `convert.tdf_mz_valley` above 0 a cluster is also cut at the valleys
  of its m/z profile, and with `convert.tdf_im_valley` above 0 each piece at the valleys of its
  mobility profile. One function (`valley_cuts`) serves both axes: the profile is smoothed with a
  triangular kernel of half-width `tdf_mz_smooth_ppm` or `tdf_im_smooth_scans` (8 each), and a
  local minimum below the given fraction of the smaller neighbouring hump starts a new piece.
  Both fractions default to 0, which leaves the centroids unchanged.
- **Intensities** are raw detector counts. `intensity_correction_factor` is ignored, which is
  exact while frames share one ramp time.
- **Determinism.** Frames are decoded in parallel batches and handed to the writer in frame
  order, so the output does not depend on the thread count.
- **Peak caps.** `--top-peaks-ms1/ms2` select the kept peaks with the same step as the mzML path
  (intensity descending, the lower index on ties); the kept peaks stay in m/z order.
- **`--max-spectra` counts frames**, not spectra, on this path: N frames yield N spectra for MS1
  frames and one spectrum per slot for MS2 frames.

The reader settings are provisional. `tdf_im_gap_scans` 30 and `tdf_min_points` 2 rest on a peak
census of one run and the seed plateau. The valley fractions stay at 0 until they are measured
as one package with the seed's count-once matching on the six-run HYE set with three `nn_torch`
seeds: on its own, the mobility split cut the seed anchors from 5,200 to 3,331. The smoothing
half-width of 8 replaced 4, the value measured to over-split on both axes; it is inert while
the fractions are 0.

Reader scope: timsrust 0.4.2 classifies frames by `MsMsType` only. Synchro-PASEF, midiaPASEF and
slice-PASEF, and older TDF compression, are untested. Only Ultra 2 E. coli and HYE runs have been
searched end to end; a timsTOF Pro run has only been converted.

### Artifacts and provenance

The spectra and isolation-window artifacts of a diaPASEF run are v2 (`artifact::spectra_ms1`,
`spectra_ms2` and `isolation_windows` selectors in `schema.rs`); an mzML run keeps v1 and its
exact bytes.

| artifact | v2 adds |
|---|---|
| `spectra_ms1` | `im`: per-peak 1/K0 list, parallel to `mz` |
| `spectra_ms2` | `window_im_lower`, `window_im_upper` (nullable f32, the slot's 1/K0 bounds); `im` per-peak list |
| `isolation_windows` | `im_lower`, `im_upper`: one row per slot, so slots that share an m/z range are separate rows. Provenance only; no stage reads them. |

The spectra reports of a diaPASEF run add `params.reader = "timsrust"` and the seven
`convert.tdf_*` settings; `params.instrument` is the `GlobalMetadata.InstrumentName` of
`analysis.tdf`, with no PSI-MS accession. An mzML report has no `reader` field, and an absent
field reads as mzML. `run` records the version each artifact was written with, and a later pass
over existing spectra reads it back from the reports.

`mumdia peak-census --mzml <run>.d` counts peaks per MS2 spectrum after the native centroiding,
uncapped, which is what `convert` writes; the census JSON then carries `"reader": "timsrust"`, and
`--max-spectra` counts frames.

The manifest hashes a `.d` input as one value over the files directly inside it (name and blake3
of each, in name order).

## 3. Window identity

Several stages group scans by isolation window: extraction's per-window scan groups, its
zero-filled chromatogram grid, the seed search's window groups, and convert's window ids. Each
keys the group on `IsolationWindow::key()`: the bits of the m/z bounds and of the 1/K0 bounds,
with an absent 1/K0 bound sorting last. A key on the m/z pair alone would merge the scans of two
slots that share an m/z range into one chromatogram cadence. On 3D data the key orders exactly as
the m/z pair did, so 3D output is unchanged.

The chromatogram grid (`extract.emit_window_grid`) is an ordered map on the key, ascending by
lower then upper m/z, so slots that share an m/z range keep a fixed order. Its coverage test is
on m/z only: every slot whose m/z range holds a precursor contributes its scan times to that
precursor's grid, whatever its 1/K0 range. That is exact while the slots of a scheme have
disjoint m/z ranges, as in both benchmark schemes. A scheme that tiles one m/z range over
several 1/K0 slots would need the slots' 1/K0 bounds in the coverage test, or the grid gains the
scan times of slots that cannot hold the candidate; that is not done (section 5 notes the limit).

Isolation-window groups (`groups.rs`, `run_groups.rs`) keep their costs on the m/z pair, because
bands are cut in m/z.

Refused on diaPASEF input until they carry the 1/K0 slots, each with a message naming the setting
to change:

- grouped runs (`groups.window_groups > 1`): the window list, the seed pool, the per-band
  calibration and chromatogram schema all key on m/z only. They come back together with a
  measured grouped diaPASEF run;
- the prescreen (`prescreen.enabled`, `tag_prefilter`, `score_before_prediction`): its window
  groups are keyed on the m/z pair;
- the experiment second pass (`mbr.strategy = second_pass`, the default of `run-experiment`): it
  has no 1/K0 window for the transferred precursors. Set `mbr.strategy = none`. Section 7 lifts
  this refusal.

Shadow demotion (`rescore.shadow_min_shared`) keys on m/z isolation windows; on diaPASEF it may
demote a precursor that shares fragment m/z with a co-isolated ion in another mobility band. Its
optional 1/K0 term (`rescore.shadow_im_tolerance`, off by default) is described in section 5.

## 4. Library and calibration 1/K0

Library 1/K0. A precursor library carries `predicted_im` (nullable Float64, 1/K0 in V s cm^-2,
`fragment_library_precursors` v2) only when it has an ion-mobility source: predict-frag with
`predict_frag.im_predictor = im2deep` (or `auto` with an IM2Deep interpreter), or an imported
library with at least one finite positive `IM`/`IonMobility`. Otherwise the table is v1, byte
for byte. IM2Deep (>= 2.0.0, `scripts/im2deep_worker.py`) predicts one single-conformer CCS per
(peptidoform, charge), converted to 1/K0 at the precursor m/z with IM2Deep's Mason-Schamp
constants (`mumdia_core::constants::im_to_ccs` inverts it). A candidate without a prediction is
dropped with its pair under the 2% ceiling. A decoy's 1/K0 follows its iRT: script-built decoys
copy the target's value, native FASTA decoys get their own prediction. `augment_library.py` adds
entries with null 1/K0. Re-predicting an imported library's 1/K0 is deferred (IM D3 H1: +2.3%,
three seeds, one acquisition, no entrapment).

`im_predictor = auto` resolves per run on the predicate of section 1: it is `im2deep` only when
the run's input carries 1/K0 (`instrument::input_has_im`, decided before conversion) and an
IM2Deep interpreter resolves (`PredictFragConfig::im_predictor_for`). A 3D run, and a standalone
`predict-frag`, which has no run, predict no 1/K0 under `auto`, with a log line; an explicit
`im2deep` is a hard requirement everywhere. The library cache keys the resolved choice. The
desktop application points `MUMDIA_PYTHON_IM2DEEP` at its primary environment, where IM2Deep is
installed.

Seed 1/K0. When the searched spectra carry per-peak 1/K0, the seed writes `observed_im` and
`seed_psms` v2 (docs/07 step 6): for each confident target, the intensity-weighted median 1/K0
of its matched fragment peaks at its best scan, each peak being the calibrant's own nearest peak
(`spectra::nearest_peak`) and counted only within `fragment_tol_ppm`. A 3D seed is v1, byte for
byte, and its calibrants are unchanged. The seed is ungated in 1/K0: it runs before any
calibration exists. `search_seed.count_fragments_once` (default false, `matcher = fragindex`
only) counts each predicted fragment once per spectrum at its most intense peak, so the mobility
pieces of one fragment ion at one m/z do not count as several matched fragments; it is
benchmark-gated together with the reader's valley fractions (section 9).

Calibration 1/K0. rt-im-train fits, on the same confident targets, a per-charge linear map of
the library's CCS onto the observed CCS, and sizes the half-window `w_im` on held-out anchors
(30% of the base peptides, `IM_HOLDOUT_FRAC`) at `p_im` times `im_window_multiplier`, floored at
`im_window_min` (docs/08 section 7). It fills the existing `im_pred_cal`, `im_lo` and `im_hi`
columns of `run_windows` (still v1) and writes the `im_*` block and `w_im_sizing` of `cal.json`,
both only on a run whose seed carries `observed_im`. Extraction reads the IM windows for its
1/K0 gate and its `im_pred_cal` column (section 5). The IM anchors follow the relaxed anchor cut of
`anchor_q_ladder`. There is no refit of the calibration on identifications.

## 5. Extraction and features

Everything in this section applies only when the spectra carry per-peak 1/K0. A 3D run writes
the artifacts, reports and manifest entries it wrote before, and every new key is omitted from
the canonical configuration at its default, so 3D config hashes do not change.

### One window type and one reader

The run's windows are one type, `rt_im_train::RtWindows`: the RT arrays plus the 1/K0 window
(`im_lo`, `im_hi`, `-inf`/`+inf` for a candidate without one) and the calibrated 1/K0
(`im_cal`), the 1/K0 fields `None` when no row has a 1/K0 window. Extract's
`read_run_windows` is the one reader of `run_windows`, and rt-im-train fills the same fields
when it hands the windows over in memory (`run`, `run-experiment` reuse, the RT bootstrap), so
the handover equals the file, 1/K0 included. Every window test of extract is
`RtWindows::admits(c, rt, im)`, which with no 1/K0 to test is the RT test it replaced
(docs/09 section 8).

### The 1/K0 gate

`extract.im_gate = off | fragments | auto`, default `off`. Under `fragments` a fragment peak
outside the candidate's 1/K0 window is not matched to it; `auto` is `fragments` when the spectra
and the windows carry 1/K0. The MS1 reads are not gated (the IM branch's `fragments_ms1`
variant measured -0.6%, within noise, and is not ported); `claim_cues.ms1_support` and
`ms1_rescue` stay ungated on 4D data. Evidence: +5.7% on E. coli (P4, three seeds), the
isolated gate effect on that acquisition only, and entrapment only for the whole P6
configuration. The optimum follows the `w_im` calibration: a window multiplier of 0.5 lost
11%. The default moves to `auto` only with a paired gate-on/gate-off entrapment arm (T12) and
HYE diaPASEF seeds. The extraction gate (`gate_min_score` under `window_mean`) was never measured
on sparse diaPASEF input; arm T4 sweeps it with the 1/K0 gate on.

### Evidence and chromatograms

`psms_extracted` is v4 on a 4D run (`artifact::psms_extracted(has_im)`): `apex_im` (the
intensity-weighted median 1/K0 of the matched fragment peaks in the apex scan, the seed's
`observed_im` rule), `apex_im_mad`, `ms1_apex_im` (the MS1 precursor peak nearest `apex_im`,
not the most intense, read at the learned MS1 offset; not gated) and `im_pred_cal`. Each row
carries the values of its own apex, a promoted alternative peak included. The MS1 mass
calibration takes, on 4D data, the in-tolerance peak nearest the seed's `observed_im` instead
of the most intense one (`ms1cal::estimate`), by the same `spectra::ms1_peak_near` rule.

Chromatograms carry the per-point 1/K0 only while a consumer is on (the feature family below;
retrace in section 6): version 3 is version 1 plus `im`, version 4 is version 2 plus
`im_trimmed`, the IM branch's numbering, so `extract.chromatogram_schema = 1` keeps working. The
list adds about 38% to the table. Every recording site reads the version back from the table
(`chromatograms::recorded_schema`). The 4D chromatogram writer encodes whole row groups as jobs
on the codec pool while the next one fills (`with_row_groups_in_flight`, bounded by the pool and
off at `--threads 1`; docs/03 "Parallel column codec"); a 3D table is written as before. The
extract reports of a 4D run carry `params.reader = "timsrust"` and the resolved `im_gate`, so a
stage reading only these artifacts resolves the TIMS predicate from them.

The chromatogram grid's coverage test stays on m/z (section 3).

### The ion-mobility feature family

`features.im_features = off | on | auto`, default `off`; `auto` is `on` when the spectra carry
1/K0. `run` resolves it once from the input routing and `run-experiment` once for the whole
experiment (any run with 1/K0), before the configuration is hashed
(`Config::resolve_im_features`), and every stage sees `on` or `off`: the schema id the manifest
records, the chromatograms extract writes, the features stage and the rescore projection read
one list. A standalone `mumdia features` resolves `auto` from the chromatograms report. The
family is `extended`-only, appended after `window` and `rt_window` and before the six
psms-derived columns, so its column position differs from the IM branch, which appended it after
those columns for every set. Nine names (docs/10, `ion_mobility`): the apex 1/K0 error against
the prediction, the fragment 1/K0 dispersion, the MS1 precursor's 1/K0 agreement, and three
over the elution peak from the per-point 1/K0. A run without 1/K0 gets constant zero columns.
The compact rescore preset has none of them. A pool that mixes runs with and without the
family fails the rescore schema check, whose message names the family. Evidence: +1.8% on
E. coli (P5, three seeds), entrapment only with the block on, one unpaired HYE seed; the default
moves to `auto` with arm T12 and seeds on a second acquisition. The IM branch's 1/K0 peak-shape
block (P7, `im_shape_features`: -0.9%, re-test -0.1%) is not ported. `imc_ref_w`, the retrace
apex co-elution in 1/K0, comes with retrace (section 6). An open question is a pooled
`|im_error| / w_im` feature, by analogy with `rtw_*`: the IM branch dropped it as constant
within one run, but `run-experiment` pools runs by default.

Memory: on a 4D run with the family on, the per-point 1/K0 roughly doubles the resident trace
memory of a features chunk (to be recorded under G5).

### Fragment tolerance

`search_seed.frag_tol_mad_k` (default 0): the learned fragment tolerance as
`mad_k x 1.4826 x MAD` instead of `1.5 x p95`, floored at `frag_tol_floor_ppm`; under
`two_pass_mass_cal` its second pass refits the offset only (docs/07). At 0 it is bit-identical.
Every IM number for it is single-pass: +5.7% over the p95 arm on E. coli (three seeds), HYE not
isolated. Arm T3 decides between the estimators on E. coli and HYE diaPASEF and on the 3D
benchmarks; if one ties or wins everywhere, the other is retired. Sizing the tolerance from
accepted identifications was measured negative (variant C, 7.99 ppm: -2.7%; a fixed 8 ppm:
-3.7%).

### Shadow demotion 1/K0 term

`rescore.shadow_im_tolerance` (default 0, off): two rows count as sharing fragments only when
their apex 1/K0 lie within the tolerance. The apex 1/K0 is joined from each run's
`psms_extracted` on `(candidate_id, selected_peak_rank)` through one ordered-map helper
(`extract::apex_im_by_peak`, which the second pass's consensus 1/K0 is meant to reuse). A row
without an apex 1/K0, every row of a 3D run, and a rescore given no `psms_extracted`
(`--psms-extracted` for a standalone `mumdia rescore`) keep the m/z-and-RT test, so with the
term off the demotion is bit-identical. Arm T6 compares the demotion off, on without the term,
and on with it. The second-pass rescore gets no `psms_extracted` yet; the second pass is
refused on 4D input until section 7.

## 6. Retrace

Filled by PR 4: rebuilding the chromatogram traces from the raw TIMS events of the `.d`, and
re-picking the apex jointly in retention time and 1/K0.

## 7. Retention-time guards and match-between-runs

Filled by PR 5: the anchor filters of the retention-time calibration and the 1/K0 extension of
the experiment second pass, which lifts the refusal of section 3.

## 8. Quantification

Filled by PR 6: the cross-run fragment selection and background for diaPASEF quantification.

## 9. Recipe, results and open levers

Filled by PR 7: the example configuration, the measured ladder with its units, seeds and
entrapment results, the closed levers, and the open ones.
