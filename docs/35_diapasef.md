# 35. diaPASEF: timsTOF input with ion mobility
> Part of the MuMDIA developer documentation (see docs/README.md).

This document describes how MuMDIA reads, searches and quantifies diaPASEF data from a Bruker
timsTOF. Sections 1 to 8 describe the mechanisms, section 9 how to run them, section 10 the
artifacts that carry 1/K0, sections 11 to 13 what was measured and how the losses were
diagnosed, and sections 14 and 15 what is open and what is not supported. Every TIMS key
is off by default; `configs/examples/diapasef.json` is the recipe that turns them on.

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
   1/K0 (section 6, in the tree);
6. extend retention-time recalibration and match-between-runs with a 1/K0 window (section 7, in
   the tree);
7. quantify with cross-run fragment consistency (section 8, in the tree).

A run on a diaPASEF `.d` with the default configuration uses layers 1 and 2 and records the
1/K0 evidence of layer 4: the seed search, extraction and features use the m/z, intensity and
retention time, and the 1/K0 columns are written for the keys that read them. The 1/K0 gate,
the feature family, retrace, the re-pick and the cross-run quant selectors are each behind a
key that defaults to off.

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
  groups are keyed on the m/z pair.

The experiment second pass (`mbr.strategy = second_pass`, the default of `run-experiment`) was
refused here until it had a 1/K0 window for the precursors it searches again; section 7 adds it
and lifts the refusal.

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
benchmark-gated together with the reader's valley fractions (section 14, arm T13).

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
retrace, section 6): version 3 is version 1 plus `im`, version 4 is version 2 plus
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
apex co-elution in 1/K0, is the family's tenth feature under `retrace.apex_features` (section 6). An open question is a pooled
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
and on with it. The second-pass rescore reads each run's second-pass `psms_extracted`, whose
candidate ids are the empirical library's (section 7).

## 6. Retrace

An extract trace point on a diaPASEF run is the intensity of one convert centroid, and the
centroiding loses weak signal. Retrace rebuilds the traces from the raw TIMS events of the
`.d`, after extract and before features, and can choose the apex again, jointly in retention
time and 1/K0, on the same events. docs/09 section 9 describes the stage; this section gives
its place in the diaPASEF work and its measurements.

### What it does

- Every fragment trace point becomes the sum of the raw events in the grid point's frame, in
  the quad slots whose isolation window holds the precursor, within the learned fragment
  tolerance of the calibrated fragment m/z and inside `apex_im +/- retrace.im_half_width`
  (0.015). The MS1 isotope traces are rebuilt the same way from the MS1 frames extract
  samples (`ms1_neighbours`, `extract.ms1_scan_halfwidth`), within `extract.prec_tol_ppm` of
  the MS1 row's m/z, which already carries the learned MS1 offset.
- `retrace.repick` (default off) chooses, among extract's apex and the
  `extract.retain_top_peaks` sidecar peaks, the peak whose best 1/K0 band agrees best with
  the library, and moves `apex_rt` and `apex_im` there before the traces are rebuilt.
- `psms_extracted` is patched once, in place (v4 kept): the re-picked apex, the apex scalars
  read from the rebuilt traces, and under `retrace.apex_features` the column `imc_ref_w`,
  which the ion-mobility feature family carries as its tenth feature.

Keys: `retrace.enabled` (false), `retrace.repick` (false), `retrace.im_half_width` (0.015,
fragments and MS1), `retrace.ms1` (true) and `retrace.apex_features` (false), all omitted
from the canonical configuration at their defaults. `Config::validate` refuses `repick`
without `extract.retain_top_peaks >= 2`, with `extract.apex_refine_intensity`, and
`repick` or `apex_features` without `enabled`; `apex_features` with
`features.im_features = off`; and `enabled` with `groups.window_groups > 1` or with
`extract.promote_top_peaks > 1`. A run whose spectra were not read natively from a `.d`
is refused when the chain reaches retrace, with the reason.

### How it differs from the IM branch

- It reads and writes the v2 chromatogram layout through `crate::chromatograms` and records
  the version through `chromatograms(layout, has_im)`; the IM branch forced
  `chromatogram_schema = 1` under retrace. The rows are read by the table's actual row
  groups, and the writer is the chromatogram writer with whole row groups encoded on the
  codec pool (`with_row_groups_in_flight`); `ParallelTableWriter` is not ported. Retrace's
  bytes therefore differ from the IM branch's through encoding only, and a comparison with an
  IM-branch retrace artifact is a content comparison.
- One `psms_extracted` patch replaces the IM branch's `psms_extracted.repick.parquet` and
  `<chromatograms>.apex.parquet`; the apex scalars are refreshed whenever retrace rewrites
  traces, with no key, and `features.retrace_apex` became `retrace.apex_features`.
  `ms1_isom1` has no trace to rebuild from and stays IM-blind (docs/09 section 9).
- Retrace runs inside the extract-to-compete core. The RT bootstrap pass does not retrace;
  `run`, `run-experiment` and the second pass do. The centroid table is deleted after
  retrace succeeds, through the helper that also deletes a grouped run's band tables; there
  is no `MUMDIA_KEEP_INTERMEDIATE`.
- The mass calibration is read through the one reader of `masscal.json`
  (`masscal::read_mass_cal`), and the MS1 frames through the selection extract's MS1 read
  uses. A half-width of 0 there is the IM branch's nearest-frame rule; the default of 1 takes
  the largest sum over the nearest frame and its two neighbours.
- A fragment grid point whose raw sum is empty gets the 1/K0 `+0.0` (the IM branch kept the
  centroid value), so the trace keeps the invariant both chromatogram layouts rely on. The
  1/K0 at a zero-intensity point is read by no feature.
- `min_events` (a no-op at its default of 1) and `ms1_im_half_width` are not ported.
- `imc_ref_w` goes through the one leave-one-out reference-correlation kernel
  (`stats::loo_ref_corr`), which the co-elution and interference families now share; the 3D
  feature values are bit-identical (`the_loo_kernel_is_bit_identical_to_the_copies_it_replaced`).
- The chromatogram readers beyond the first are leased from a process-wide pool
  (`sched::DecoderLease`, the features stage's lease, lifted to the crate) instead of a fixed
  eight.

### Measured on the IM branch

All counts are stripped peptides at `peptide_q_value` 1% with `nn_torch`, on the base the IM
branch had at the time (the L1d configuration with its RT refit); the integrated pipeline is
re-measured under gates G1 to G5 and arms T3b and T10 (section 14).

| arm | E. coli peptides (seeds 0 / 1 / 2, mean) | entrapment | HYE diaPASEF, six runs, seed 0 |
|---|---|---|---|
| retrace off | 12,927 / 12,950 / 12,874 (12,917) | | 77,734 peptides |
| `retrace.enabled` | 13,619 / 13,679 / 13,735 (13,678, +5.9%) | real peptides at the prototype's level, FDP 0.40-0.44% | 89,424 peptides (+15.0%); ProteoBench ions 69,856 -> 77,627, median abs epsilon 0.185 -> 0.171, CV 0.118 -> 0.108 |
| + `retrace.repick` | 14,869 / 14,911 / 14,953 (14,911, +7.4%) | 13,975 real peptides (+5.4% over the control's 13,258) at an FDP of 0.37-0.47% | 100,790 peptides (+8.4% over the same pipeline without the re-pick); ProteoBench ions +15.2% |

Decoy fraction 0.99% in every E. coli arm. Precursors on E. coli rose from 15,679 to 16,737
with retrace and to 18,544 with the re-pick.

Levers measured on the retrace base:

- The apex scalars refreshed from the raw traces alone: +0.6%. Main refreshes them with every
  retrace, so a table never holds raw traces beside centroid scalars.
- `imc_ref_w` alone: +1.3%, both together +1.8% (13,920); entrapment for both +2.2% real
  peptides at an FDP of 0.43-0.51% against 0.40-0.44%. The added peptides carried about 2.7%
  FDP, so `retrace.apex_features` stays off until an entrapment recheck of the added set.
- The raw MS1 band: +1.6% (three seeds, entrapment-clean) at 20 ppm and the nearest frame.
  `retrace.ms1` stays a switch until an engine ablation confirms it (arm T10, which also
  compares `ms1_scan_halfwidth` 0 with 1 and 3 ppm with the diaPASEF value).
- Narrowing the summation tolerance to 7 or 5 ppm lost 1.0% or 2.3%; retrace sums within the
  learned tolerance. A band half-width from 0.012 to 0.020 was flat. Event floors of 2 or 3
  lost identifications.
- An RT prior on the re-pick (sigma = the RT-window half-width) moved 3,064 candidates of one
  HYE run onto the reference peak, but identifications stayed within seed noise: E. coli
  +0.25%, entrapment -0.6% real peptides at an FDP of 0.41-0.42%, HYE target rows +0.5%. Not
  kept. `MUMDIA_REPICK_DUMP` writes every scored peak, which is what this counterfactual was
  computed from.
- K = 3 or 8 sidecar peaks instead of 5, and aligning the seed rows to the chosen apex, were
  within noise.

### Cost

Retrace holds every frame inside the candidates' RT windows: 21 GB on the E. coli file and
36-39 GB per HYE diaPASEF run, on top of extract's own peak, which is why it stays a stage
after extract rather than folding into it. On the IM branch it took 12 s on E. coli and 160 s
per HYE run (7.4e9 trace points) before the TOF bucket table, with the re-pick 15 s and
6-7.5 min. Profiles showed 70-80% of the busy time in the whole-frame binary search for the
first event of each TOF window; the per-frame bucket table (`TofFrame::first_at`) took one
HYE retrace call from 2:37 to 1:44 at 0.9 GB more memory. The integrated stage is re-measured
under G5 (one retrace and three concurrent ones), because the v2 layout, the codec-pool
writer and the leased readers all change its wall time.

## 7. Retention-time guards and match-between-runs

### Retention time from identifications

`rt_bootstrap` (`rt_im_train.multihead_anchors`, docs/08 section 4f) is the one mechanism that
refits retention time on identifications. The IM branch's `rt_im_train.refit` (a full first
pass, a pooled rescore, a cross-fitted 80-head refit, a LOESS and 1/K0 recentre, and a second
extraction per run) is not ported: cross-fitting measured equal to in-sample fitting (12,916
against 12,871), and its gain cannot be split onto the bootstrap's design without an arm on
TIMS data (arm T1 below).

What the refit taught is kept as two guards (docs/08 section 4g). With many anchors the
multi-head calibration returned absurd iRTs for about 2% of library rows (up to 3.8e5 s) and an
erratic ridge alpha; on HYE diaPASEF the first pass accepted 63-64k target PSMs per run, well
past the 29,419 anchors at which the spline failed, so both guards are preconditions of
`multihead_anchors = search` on TIMS data.

- **Anchor x-domain filter** (`rt_im_train.anchor_x_filter`, default off): the anchors outside
  `[Q1 - 3 IQR, Q3 + 3 IQR]` of their predicted iRT are dropped before any fit, so a few absurd
  anchors no longer stretch the LOESS grid. It replaces the IM branch's `robust_calibration`,
  whose p0.5 / p99.5 x fences break down at 0.5% contamination per side; the branch's y-outlier
  cut is not ported. On one HYE diaPASEF run the IM filter took the global slope from 0.03-0.13
  to about 1.00 and the in-sample residual to 3.7-4.0 s, removing 362-489 anchors. It becomes
  always on only if it removes 0 anchors on the 3D benchmark anchors (G0).
- **Multi-head output-range guard** (`deeplc_finetune.py`, no key): calibrated iRTs more than
  one anchor RT span beyond the anchors are clamped before the library is written, on every
  prediction path, with the count, fraction and ridge alpha in the library's `summary.json`. It
  is a no-op when nothing is out of range, which is expected on 3D data (to be verified under
  G0 on Astral and AIF).

No anchor cap is added (`--max-ref` takes the first rows in m/z order, and a cap did not
prevent the failure).

Arm **T2**, `multihead_anchors = search` against the anchor-count `auto` on TIMS after both
guards, is measured (2026-10-14, section 11, "ProteoBench HYE diaPASEF"): +5.7% peptides on the
six 60 min HYE runs (11,144 seed anchors, one seed) and +0.6% on the E. coli run (4,639, three
seeds) at an entrapment FDP of 0.39-0.46% against 0.42-0.51%. `auto` therefore chooses `search`
on every run whose spectra carry 1/K0 (docs/08 section 4f, "the 1/K0 rule"). The IM branch's
per-run refit still leads by 2.0% peptides on the HYE runs (100,392 against 98,373, same library
and raw files); `experiment.rt_library_scope = per_run` and a 1/K0 recentre are the next arms.
Planned: **T1**, the RT
recentre increment, the bootstrap with and without a second LOESS on the search anchors, three
seeds, E. coli entrapment and quantification ions. A recentre that gains becomes one default-off
sub-option of `rt_bootstrap`, not a refit path, and widths are never sized from the refit (IM
arm A1, refit widths: -4.0%). Runs without a DeepLC interpreter get no bootstrap, so they would
not get a recentre either.

### Match-between-runs: the second pass on 4D data

Main's second pass (docs/12, "Second pass") is the one mechanism; the IM branch's
`mbr.second_pass` keys, `scripts/mbr_second_pass.py`, its separate decoys and its re-extraction
tier are not ported (its arms B to D showed that the split library and report cut and the new
decoys add nothing worth a second mechanism: eq -0.004 at 300 fewer ions, and -0.003 at about
750 fewer ions). Main's second pass gains three things on data with 1/K0, each a no-op on 3D
data:

- **Consensus 1/K0.** The median of the runs' re-picked apex 1/K0 among the confident apexes,
  joined from each run's `psms_extracted` on `(candidate_id, selected_peak_rank)` through the
  ordered-map join the shadow term uses (`extract::apex_im_where`), each run moved onto the
  retention-time reference by its median offset over at least `MIN_SHARED` shared precursors
  (IM measured offsets of -0.002 to +0.0002). Fallbacks: the winning row's apex 1/K0, then the
  library's `predicted_im`. One override per library row carries the retention time and the
  1/K0 (`sub_library::LibOverride`), and `plan_rows` gives it to both members of each pair, so
  the decoy shares its target's values (section 4 rule). Offsets and counts go to
  `empirical_library.json` (`im`).
- **1/K0 window floor.** With an observed consensus the second-pass `w_im` falls to about
  0.008-0.010, and the IM branch lost 8% of the reproducible rows to the 1/K0 gate (37,326
  first-pass accepted library rows not reproduced, 33,755 not extracted; 99,711 ions against
  105,801 with a 0.04 floor). Each run's `im_window_min` is raised to its first-pass `w_im`;
  `cal.json` records the floor and its source. It is derived from the data and has no key.
- **Quantification trace pad.** Extract and retrace record traces `(fixed integration
  half-width + baseline flanks) x cycle time` beyond the narrow second-pass RT window (IM's
  held-out p99 of the consensus residual: +/-7.6-8.8 s, against about +/-15 s that a 4-scan
  half-width plus flanks needs at 0.97 s per scan), while the apex, the gate and every feature
  stay inside it. It replaces the IM branch's separate quantification traces.

Retrace runs in the second pass (the chain hook of section 6, on there), and the second pass
reuses the first pass's run plan. Two things remain to be checked on one diaPASEF experiment:
that the second-pass chain's peak with retrace stays at or below the first pass's, and how often
quantification windows and flanks were clipped before the pad (G4).

**Diagnostics** (IM test 4): after the pooled second-pass rescore, per run the targets accepted
by each pass, the new ones and the dropped ones, and the accepted decoys per accepted target in
anchor and non-anchor runs, logged and written to `empirical_library.json` (`diagnostics`). On
the IM branch this decoy fraction was 0.77-0.79% in anchor runs against 1.84-1.89% in non-anchor
runs, about twice the nominal 1% where most new rows are; it is the open FDR point of the second
pass on TIMS data, and the multi-run entrapment (G2) is its gate.

**The library cut is the report cut.** In IM entrapment test 3, 113-114 of 125 reportable false
library entries came back in pass 2. `Config::validate` warns when `mbr.q_anchor > 0.01`.

Planned arms: **T8**, consensus 1/K0 against the calibrated library 1/K0 in the second pass,
both under the floor (G2 multi-run entrapment, G3, G4); the override's 1/K0 stays only if T8
shows a gain. **T9**, the first-pass `w_im` floor against IM's fixed 0.04, with the first-pass
`w_im` values and the 1/K0-gate losses on second-pass rows.

Main's second pass on the six 5 min HYE diaPASEF runs (section 11, seed 0): +18.7% peptides per
run at an unchanged pooled decoy fraction of 1.07%, with the decoy fraction at 0.82% in anchor
runs and 1.63% in non-anchor runs, the same pattern as on the IM branch.

## 8. Quantification

The cross-run quant levers (`quant.fragment_selection = cross_run`, `quant.baseline_source =
experiment`, `quant.fixed_scan_halfwidth_scale`) are described, with their diaPASEF
measurements, in docs/12_quant_lfq_align_mbr_report_audit.md, "Cross-run fragment selection
and background". The record below used predicted fragment selection and the interference
envelope, which are not defaults. Main's default quantification and the cross-run levers were
measured on main only on the six 5 min HYE runs (section 11), with the median-ratio metrics of
that section rather than ProteoBench.

### The quantification record, condensed

HYE diaPASEF, ProteoBench `quant_lfq_DIA_ion_diaPASEF` at `min_obs` 3, measured on the IM
branch with identifications fixed at its re-pick configuration (seed 0: 92,540 ions; seeds 1
and 2: 92,141 and 92,475). Global is the median |log2 ratio error|, eq the mean of the
per-species medians (species-equalised), CV the median of the per-condition CVs.

| step | global / eq / CV | commit |
|---|---|---|
| default quant, earlier identification base | 0.222 / 0.549 / 0.122 | `c5d41bb` |
| `fragment_selection = predicted` + `interference_envelope` | 0.182 / 0.307 / 0.114 | `c5d41bb` |
| the same on the re-pick identifications (the IM branch's shipped quant) | 0.172 / 0.358 / 0.107 | `7ac12e7` |
| fixed window of 4 scans | 0.136 / 0.230 / 0.089 | `7ac12e7` |
| + `baseline_subtract` (12 flank scans, quantile 0.25): the base of docs/12 | 0.138 / 0.217 / 0.092 | `7ac12e7` |
| + cross-run weights and experiment background (three ID sets) | 0.133-0.134 / 0.180-0.181 / 0.087 | `7ac12e7`, `1bc59a2` |
| + cross-run width, scale 2.5 (three ID sets) | 0.129-0.130 / 0.177-0.178 / 0.083 | `42ccfe2` |

The recipe (section 9) sets the measured base and the first two cross-run selectors. The
width (`quant.fixed_scan_halfwidth_scale`) is the next lever: it gained on all three ID sets
but rests on one known-ratio dataset, like the rest of the record (G4).

Match-between-runs on the IM branch, same data (its mechanisms are not ported; main's second
pass is the one mechanism, section 7):

- **Transfer** (`42ccfe2`): 98.5-99.5% of transferred values sit within 5 s of the
  reference retention time, so the transfers take the right peak, but their signal is 2.4
  log2 units fainter. Cosine and 1/K0 guards did not separate the error. Lowering
  `min_anchor_runs` to 1 added 4,752 ions and worsened eq from 0.220 to 0.229.
- **Integration at the predicted retention time with a cross-fitted null** (`3f8516c`, three
  seeds): the second anchor tier added 928 to 1,083 ions at eq +0.002 to +0.004; the
  first tier was rejected on quant (eq +0.012).
- **Re-extraction at the anchors' 1/K0 band** (`3f8516c`, three seeds): 2.7 times the
  accepted rows of the integration arm and 2,300 to 2,516 more ions; re-extraction alone gave
  the best eq of any MBR arm (100,799 ions, global 0.135, eq 0.204, seed 0), and re-extraction
  followed by the rescuable tier 102,772 ions at eq 0.216.
- **Second search on an empirical library** (`0fd81bc`, `940ee33`): with the empirical
  library's own intensities at q 0.01, 105,801 ions but eq 0.235; with predicted intensities
  and new decoys (arm D), 105,078 ions, global 0.137, eq 0.209 (seed 0), which is 2,154 to
  2,306 more ions than the transfer union at eq -0.007 to -0.008. Single-run E. coli
  entrapment of arm D: FDP 0.487-0.492% against 0.367-0.473% for pass 1. The library cut is
  the report cut: 534-536 of 656 entrapment library entries were accepted in pass 2.

MBR of any kind compresses ratios, on diaPASEF as elsewhere (docs/12). The IM branch's arm D
is the reference for main's second pass on TIMS data. On main, with the cross-run levers, the
second pass raised the 5 min HYE species-equalised error from 0.301 to 0.363 and quantified 40%
more precursors (section 11); ProteoBench on the 60 min set (G4) is still open.

## 9. How to run diaPASEF

The recipe is `configs/examples/diapasef.json`. It derives from
`configs/examples/fasta-peptdeep.json` and sets:

| section | values | why |
|---|---|---|
| `convert` | `tdf_mz_valley 0.5`, `tdf_im_valley 0.5` | valley splits of the native centroids (section 2) |
| `predict_frag` | `predictor peptdeep`, `peptdeep_model generic`, `peptdeep_instrument timsTOF`, `peptdeep_nce 35`, `top_n_fragments 12`; `im_predictor im2deep`; every interpreter `auto` | AlphaPeptDeep at timsTOF settings (no `charge2_from_precursor_charge` change: it predicts charge-2 fragments), IM2Deep 1/K0 |
| `search_seed` | `count_fragments_once true` | one count per predicted fragment and spectrum (section 4) |
| `rt_im_train` | `rt_window_multiplier 1.0` | see below |
| `extract` | `im_gate fragments`, `retain_top_peaks 5`; `gate_min_score 0.6` and `apex_rt_prior_s 120` as in the FASTA example | the 1/K0 gate (section 5) and the sidecar peaks the re-pick chooses among |
| `features` | `set extended`, `im_features on` | the ion-mobility family (section 5) |
| `retrace` | `enabled true`, `repick true` | section 6 |
| `rescore` | `nn_torch`, `strict true` | |
| `mbr` | `strategy second_pass` | the default of `run-experiment`, written out; supported on 4D since section 7 |
| `quant` | `fixed_scan_halfwidth 4`, `interference_envelope true`, `baseline_subtract true`, `fragment_selection cross_run`, `baseline_source experiment` | the measured base and the first two cross-run selectors (section 8) |

`rt_window_multiplier` stays at 1.0, the engine default. The 1.5 of the Orbitrap and Astral
examples (`diann-library.json`, the prescreen examples) must not be carried over: on
diaPASEF 1.5 lost 1.8% and 2.0 lost 2.9% of peptides (section 12).

Every TIMS key the recipe sets is off by default, and no existing default depends on the
instrument; the recipe is how a diaPASEF run gets them. Check the environment first, then
search:

```text
mumdia doctor --config configs/examples/diapasef.json

mumdia run --fasta proteome.fasta --mzml run01.d \
  --out-dir out_run01 --config configs/examples/diapasef.json

mumdia run --fasta proteome.fasta --mzml run01.d --mzml run02.d --mzml run03.d \
  --out-dir out_exp --config configs/examples/diapasef.json
```

`doctor` resolves the AlphaPeptDeep, DeepLC, IM2Deep and rescore interpreters as the run will
(`MUMDIA_PYTHON_PEPTDEEP`, `MUMDIA_PYTHON_DEEPLC`, `MUMDIA_PYTHON_IM2DEEP`,
`MUMDIA_PYTHON_RESCORE`, or an activated environment) and checks the DeepLC and IM2Deep
floors. The environments are `env/mumdia-peptdeep.yml` (AlphaPeptDeep) and
`env/mumdia-deeplc.yml` (DeepLC, IM2Deep and the NN rescorer). A single `.d` searches as one
run; several are one experiment with a pooled rescore, the second pass and the cross-run
quant (a single run falls back to the per-run quant settings, with a log line). The `.d`
must be a diaPASEF acquisition that timsrust reads (section 2); any other Bruker `.d` goes
to msconvert as a 3D run, and the recipe is then refused before the search, because retrace
needs the raw events.

The recipe has no MS2PIP arm. The engine ladder of section 11 used MS2PIP `timsTOF2024`; with
a single-charge MS2PIP model set `charge2_from_precursor_charge` to 99 (docs/06).

## 10. Data model

Which artifact carries which 1/K0 column. Each 4D version is chosen by a selector in
`mumdia_core::schema::artifact` and written only when the run carries 1/K0; a 3D run writes
the 3D version and its exact bytes. docs/15 has the column definitions.

| artifact | 3D | 4D | 1/K0 content |
|---|---|---|---|
| `spectra_ms1` | v1 | v2 | `im`: per-peak 1/K0 list |
| `spectra_ms2` | v1 | v2 | `im` per-peak list; `window_im_lower`, `window_im_upper` (the slot's bounds) |
| `isolation_windows` | v1 | v2 | `im_lower`, `im_upper`, one row per slot (provenance only) |
| spectra reports | | | `params.reader = "timsrust"`, the `convert.tdf_*` settings |
| `fragment_library_precursors` | v1 | v2 when a 1/K0 source exists | `predicted_im` (nullable) |
| `seed_psms` | v1 | v2 | `observed_im` |
| `run_windows` | v1 | v1 | `im_pred_cal`, `im_lo`, `im_hi` filled |
| `cal.json` | | | `im_*` block, `w_im_sizing`; `im_window_floor` in the second pass |
| `psms_extracted` | v2 | v4 | `apex_im`, `apex_im_mad`, `ms1_apex_im`, `im_pred_cal`; `imc_ref_w` only under `retrace.apex_features`; retrace patches values in place |
| `chromatograms` | v1, v2 | v3 (v1 + `im`), v4 (v2 + `im_trimmed`) | per-point 1/K0, written only while a consumer is on |
| `chromatograms.centroid.parquet` | | | extract's table before retrace; deleted after retrace succeeds |
| `features` | | | the `ion_mobility` family (nine columns, ten with `imc_ref_w`) |
| `fragment_quant` | v1 | v1 | v5 under the cross-run export, on any acquisition (section 8) |
| `empirical_library.json` | | | `im` (consensus offsets and counts) and `diagnostics`, on experiments with 1/K0 |

`ms2_to_ms1`, `psms_competed`, `psms_scored` and the quant tables carry no 1/K0 column; the
second pass and the shadow term join `apex_im` from `psms_extracted` on
`(candidate_id, selected_peak_rank)`.

## 11. Measured results

### On main

E. coli, timsTOF Ultra 2, 15 min gradient, 50 ng, one run. FASTA library with MS2PIP
`timsTOF2024`, DeepLC 4.5.0 and IM2Deep 2.0.2; strict `nn_torch`; one seed per arm. Counts are
stripped peptides at `peptide_q_value` 1%, precursors at `precursor_q` 1% and protein groups at
`pg_q_value` 1%. The decoy fraction was 0.98% in every arm. Each row adds to the one above.

| step (PR of the series) | peptides | precursors | protein groups | wall | change |
|---|---|---|---|---|---|
| native reader, 1/K0 unused (PR 1) | 8,735 | 10,227 | 1,571 | 7:55 | |
| + valley splits 0.5 / 0.5, count-once seed, IM2Deep, 1/K0 calibration not yet used by extract (PR 2) | 10,544 | 12,478 | 1,722 | 10:54 | +20.7% |
| + `im_gate fragments`, `im_features on` (PR 3) | 11,072 | 13,032 | 1,774 | 12:40 | +5.0% |
| + `retrace.enabled` (PR 4) | 11,688 | 13,920 | 1,833 | 12:42 | +5.6% |
| + `retrace.repick`, `retain_top_peaks 5` (PR 4) | 12,624 | 15,093 | 1,899 | 12:56 | +8.0% |

The 1/K0 calibration of the PR 2 arm used 4,024 training and 1,166 held-out anchors, with a
held-out median |residual| of 0.0097 and `w_im` 0.033. The retrace arms peaked at 18.8 and
18.7 GB. `search_seed.frag_tol_mad_k 4` on the PR 3 arm gave 11,037 (-0.3%): with two-pass
mass calibration it learned 10.0 ppm against 11.7 ppm for the default estimator, and no
gain. The converted spectra (`spectra_ms1`, `spectra_ms2`, `isolation_windows`,
`ms2_to_ms1`) are identical, column for column, to the IM branch's, and conversion takes 50 s.

Single seeds: a difference under about 1% between two of these arms is not a measurement
(docs/28). The arms that are still to be run on this base are in section 14.

**The recipe.** `configs/examples/diapasef.json` (section 9: AlphaPeptDeep `generic`,
`timsTOF`, NCE 35, and the PR 2 to PR 6 keys) on the same run, three `nn_torch` seeds:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| peptides | 14,004 | 13,934 | 13,948 |
| precursors | 17,086 | 17,046 | 16,898 |
| protein groups | 1,977 | 1,963 | 1,973 |

The decoy fraction was 0.98% in every seed, the process peak 19.9 GB, and the wall 19:40 to
23:04 with two searches sharing the host at 32 threads each. A single-seed PR 4 arm with
AlphaPeptDeep and without the recipe's extraction and RT keys gave 14,139 peptides in 9:46.

Entrapment, the same three seeds against the E. coli FASTA plus a 1:1 human entrapment
proteome (`bench/make_entrapment_fasta.py`; the built library's ratio is 0.574592, 988,128 real
against 1,719,704 entrapment target candidates): 13,293 / 13,392 / 13,331 real and 96 / 116 /
106 entrapment peptides at `peptide_q_value` 1%. The engine's estimator
(`fdr::entrapment_q`, ratio x entrapment + 1 over real) gives an FDP of 0.42 / 0.51 / 0.46%,
in the range of the IM branch's re-pick arm (0.37-0.47%). The combined upper bound,
entrapment x (1 + ratio) over all accepted peptides, is 1.13 / 1.35 / 1.24%.

**Six-run HYE diaPASEF experiment.** PXD070049, timsTOF Ultra 2, 5 min gradient, 50 ng,
conditions A and B in triplicate (expected log2 B/A: human 0, yeast -1, E. coli +2). One
`run-experiment` per arm with an imported FASTA library built once (missed cleavages 1, length
7-30, charges 2-4, MS2PIP `timsTOF2024`, DeepLC, IM2Deep; 14.7M precursors with decoys,
`library_irt = library`), the PR 4 settings of the table above, strict `nn_torch`, seed 0.
`library_irt = library` on an imported library also switches the automatic multi-head
calibration off, so these arms and the recipe arm below ran without it; the recipe itself builds
its library from FASTA with DeepLC and runs it (the E. coli recipe runs above did). On the
60 min set the omission cost 12% of the peptides (next subsection), so these numbers are a lower
bound for the recipe. Peptides per run are stripped peptides at `run_psm_q` 1%, the mean of the six runs; the
quantified precursors are `peptides.tsv` rows with a quantity in at least two runs of each
condition, and the ratios are their medians.

| arm | peptides per run | decoy fraction | precursors / protein groups (experiment) | quantified precursors | human | yeast | E. coli | species-equalised error | CV |
|---|---|---|---|---|---|---|---|---|---|
| no MBR | 27,951 | 1.07% | 36,698 / 8,345 | 22,619 | +0.121 | -0.434 | +1.419 | 0.450 | 0.061 |
| no MBR, cross-run quant | 27,951 | 1.07% | 36,698 / 8,345 | 22,618 | +0.164 | -0.633 | +1.808 | 0.301 | 0.067 |
| second pass | 33,190 (+18.7%) | 1.07% | 36,664 / 9,073 | 31,673 | +0.113 | -0.318 | +1.009 | 0.616 | 0.063 |
| second pass, cross-run quant | 33,190 | 1.07% | 36,664 / 9,073 | 31,673 | +0.171 | -0.560 | +1.657 | 0.363 | 0.076 |

Cross-run quant is `fragment_selection = cross_run`, `baseline_subtract` and
`baseline_source = experiment`. It changed no identification. On the cross-run MaxLFQ protein
groups the species-equalised error fell from 0.465 to 0.255 without the second pass (6,620
groups) and from 0.642 to 0.345 with it (8,169 groups).

The second pass built 40,017 empirical library entries, aligned all six runs (1/K0 offsets at
most 0.0014 against the reference run), took every entry's 1/K0 from the runs, and set each
run's 1/K0 window floor to its first-pass `w_im` (0.033-0.034). Per run it added 7,395-7,817
accepted rows and dropped 1,578-2,564 against the first pass. The decoy fraction was 0.82%
among rows accepted in a run that anchored the entry and 1.63% among rows accepted in the other
runs: the pooled level is unchanged, but the transferred subset carries twice the decoy rate.
Without cross-run quant the extra rows compress the ratios (E. coli +1.009 against +1.419);
with it the compression mostly goes (+1.657).

The no-MBR arm ran alone on a 128-core host at 64 threads and took 44 min; the other three ran
concurrently at 40 threads each and took 1:53 each, so wall times are not comparable between
the rows. The largest process peaked at 107 GB in every arm; the pooled rescore scored 31.2M
PSMs.

The IM branch's HYE column below is the ProteoBench HYE diaPASEF set, a different acquisition
(about a 60 min gradient), and its counts are not comparable with these.

**The recipe on the six runs.** The same data with `configs/examples/diapasef.json`: the
library rebuilt with AlphaPeptDeep (`generic`, `timsTOF`, NCE 35) on the same digest, DeepLC
and IM2Deep (1:45 for 14.7M precursors at 48 threads), and the recipe's extraction, second-pass
and quant keys; seed 0, 64 threads, alone on the host.

| | recipe | second pass, cross-run quant (PR 4 settings, table above) |
|---|---|---|
| peptides per run | 39,507 | 33,190 |
| decoy fraction | 1.12% | 1.07% |
| precursors / protein groups (experiment) | 43,518 / 9,984 | 36,664 / 9,073 |
| quantified precursors | 37,887 | 31,673 |
| human / yeast / E. coli | +0.149 / -0.612 / +1.661 | +0.171 / -0.560 / +1.657 |
| species-equalised error / CV | 0.349 / 0.080 | 0.363 / 0.076 |
| protein groups, cross-run MaxLFQ: n / species-equalised error | 9,027 / 0.335 | 8,169 / 0.345 |
| wall / peak | 39:13 / 121 GB | 1:53 (shared host) / 107 GB |

Per run the second pass added 9,119-9,879 accepted rows and dropped 1,818-2,487. The decoy
fraction was 0.84% in anchor runs and 1.56% in non-anchor runs, the same split as with the PR 4
settings.

**ProteoBench HYE diaPASEF.** The six public runs of the ProteoBench module
`quant_lfq_DIA_ion_diaPASEF` (timsTOF SCP, about 60 min, `ttSCP_diaPASEF_Condition_{A,B}`),
the module's FASTA, the AlphaPeptDeep library of the recipe (14.7M precursors with decoys,
identical row for row to the IM branch's l1d library in a sample of 18,477 precursors and
221,724 fragments), the recipe's extraction and quant keys, seed 0, 120 threads, scored offline
with ProteoBench 0.18.6 (`bench/pb_eval.py`, Custom format, k = 3); nothing uploaded. Peptides
are stripped peptides at the experiment-wide `peptide_q_value` 1%.

| arm | ions (k = 3) | global / eq / CV | E. coli log2 A/B | peptides | `w_rt` | wall / peak |
|---|---|---|---|---|---|---|
| no MBR, no multi-head calibration (`library_irt = library`) | 72,051 | 0.119 / 0.158 / 0.076 | -1.86 | 82,148 | 155 s | 1:41 / 480 GB |
| no MBR, multi-head 80 | 84,841 | 0.127 / 0.168 / 0.083 | -1.86 | 93,061 | 40-45 s | 0:48 / 258 GB |
| no MBR, multi-head 80, search anchors | 90,363 | 0.130 / 0.175 / 0.085 | -1.86 | 98,373 | 24-27 s | 1:03 / 199 GB |
| second pass, multi-head 80, search anchors | 103,185 | 0.139 / 0.208 / 0.092 | -1.78 | 97,979 | 25-33 s | 1:03 / 196 GB |
| IM branch binary and configuration, no MBR | 94,099 | 0.131 / 0.178 / 0.084 | -1.84 | 100,392 | 27.6 s | 1:15 / 172 GB |

Global is the median |log2 ratio error|, eq the mean of the per-species medians, CV the median
per-condition CV. The IM branch arm reproduces its own thestral result (92,540 ions, 100,790
peptides) on these raw files with this library, so the remaining gap is in the engine path: its
per-run RT refit with a 1/K0 recentre (arm T1, sections 7 and 14). The wide RT windows without a multi-head
calibration are also why that arm needed 120-136 GB per run in extract. The second pass added
14.2% ions; its decoy fraction was 0.87% in anchor runs and 1.70% in non-anchor runs, and the
ions it adds carry most of the ratio compression (E. coli log2 -0.86 for ions accepted in one
first-pass run, -1.87 for ions accepted in all six).


### On the IM branch

The prototype ladder of the IM branch (PR #140). Same E. coli run, three `nn_torch` seeds
unless marked, decoy fraction 0.98-0.99% in every arm. Each row was measured on the base the
branch had at the time: the extraction gate `apex_pearson` at 0.2 (gate 0 from P6c on),
single-pass mass calibration, and none of the defaults main gained since (among them the
`window_mean` gate, two-pass mass calibration, `rt_bootstrap` and the second pass).
The HYE column is the six-run HYE diaPASEF experiment, pooled, seed 0, no entrapment.
Entrapment is E. coli plus a 1:1 human entrapment proteome (section 14), FDP at the peptide
level.

| step | commit | peptides (mean) | change | entrapment | HYE peptides |
|---|---|---|---|---|---|
| P0: msconvert route (one seed) | `7f9aa3f` | 7,662 | | | no result: 0 seed anchors, then out of memory |
| P1: native reader, 1/K0 unused (one seed) | `29575e1`, `76d57b6` | 7,231 | -5.6%, 5.2x faster (45:30 to 8:45) | | |
| P2: IM2Deep and 1/K0 calibration, unused | `c5d41bb` | identical to P1 | | | |
| P4: + fragment 1/K0 gate | `c5d41bb` | 7,615 | +5.7% | | |
| P5: + ion-mobility features | `c5d41bb` | 7,753 | +1.8% | | |
| P6: + MS2PIP `timsTOF2024` (charge-1 fragments), `frag_tol_mad_k 4`, gate 0, `apex_count_window 3` | `c5d41bb` | 9,507 | +22.6% | FDP 0.355-0.378% | 54,895 |
| L1c: + AlphaPeptDeep `generic`, `timsTOF`, NCE 35 | `31cba11` | 10,899 | +14.6% | FDP 0.438-0.483% | 60,547 (+10.3%) |
| + m/z valley split 0.5 | `ba1b3bf`, `31cba11` | 11,990 | +10.0% | FDP 0.445-0.456% | 67,768 (+11.9%) |
| + mobility valley split 0.5 and count-once seed | `ba1b3bf`, `1b186c6`, `31cba11` | 12,451 | +3.8% | FDP 0.403-0.408% | 73,169 (+8.0%) |
| L1d: + RT refit on identifications (not ported, section 7) | `029ac82`, `db6d434` | 12,919 | +3.8% | FDP 0.413-0.438% | 77,734 (+6.2%) |
| + retrace | `907652f` | 13,678 | +5.9% | 12,922 real, FDP 0.40-0.44% | 89,424 (+15.0%) |
| + raw apex scalars and `imc_ref_w` | `2739f88` | 13,920 | +1.8% | +2.2% real, FDP 0.43-0.51% | 92,951 (+3.9%) |
| + re-pick (`retain_top_peaks 5`) | `f544cd7` | 14,911 | +7.4% | 13,975 real (+5.4%), FDP 0.37-0.47% | 100,790 (+8.4%) |

Components of P6, each over P5 (three seeds): `timsTOF2024` with charge-1 fragments +2.7%;
with `frag_tol_mad_k 4` (12.54 ppm) +8.5%; with gate 0 +21.2%; with `apex_count_window 3`
+22.6%. On HYE, k = 4 learned 18.4-18.5 ppm. AlphaPeptDeep NCE: 25 gave 10,728, 30 gave 10,797,
35 gave 10,899 and 40 gave 10,631.

What separates the two ladders is not one switch. Main's base has the `window_mean` gate at
0.6 (the IM branch ran gate 0 from P6), the p95 tolerance with two-pass mass calibration (the
IM branch: MAD k = 4, single pass), `apex_count_window` 5 (the IM branch: 3), no RT refit
(the IM branch's L1d, +3.8%), and MS2PIP rather than AlphaPeptDeep in the measured arms (the
IM branch's L1c, +14.6%). Arms T3, T4, T5 and T1 of section 14 and the recipe arm measure
them one at a time; a shortfall is bisected with them before any IM mechanism is restored.

## 12. Closed levers

Measured and not kept. Commits are on the IM branch; seeds are `nn_torch` seeds on the E. coli
run unless marked.

| lever | result | commit |
|---|---|---|
| 1/K0 width and peak-shape features (P7, `im_shape_features`, `tdf_im_width`) | -0.9% (three seeds; seed 2 -2.8%); re-test on valley-split spectra -0.1% | `c5d41bb`, `907652f` |
| seed 1/K0 gate (`search_seed.im_gate`, `im_window`) | no more anchors (4,313 off, 3,865 at a fixed 0.06, 4,358 two-pass 0.10); a narrow gate biased the 1/K0 calibration (`w_im` 0.0247, window recall 84.2% against 97.6%); end to end -0.05% | `c5d41bb` |
| MS1 reads in the 1/K0 gate (`im_gate = fragments_ms1`) | -0.6% (one seed) | `c5d41bb` |
| `top_n_fragments` other than 12 | 8, 10, 16, 20, 24 and a 200 m/z floor lose or tie against 12; with retrace on, 16 -3.3% and 24 -8.0% | `907652f`, `2739f88` |
| fragment-model fine-tune, cross-fitted (E. coli) | held-out PCC 0.887 to 0.902, but -1.5% peptides and -5% seed anchors; open on HYE (section 14) | `907652f` |
| shared-Gaussian co-elution fit | univariate AUC 0.57-0.59 against 0.594 for `xcorr_shape`; +0.010 cross-validated AUC; not taken to a rescore | `907652f` |
| `extract.apex_rt_prior_s 8` | +1.6% on the `retrace_apex` base, but entrapment real +0.7% at an FDP of 0.49-0.50% against 0.45-0.47%: about 6% of the added peptides were false | `907652f`, `2739f88` |
| RT prior on the re-pick | +0.25%; entrapment -0.6% real at FDP 0.41-0.42%; HYE rows +0.5% | `caeede4`, `63d104f` |
| `extract.promote_top_peaks` (scored alternatives) | K = 2 -5.7%, K = 3 -5.9% (P6 base, seed 0): about 40 features are copied from rank 0 onto every alternative, and a wrong alternative beat the right rank-0 peak 40% of the time; a top-3 model trained on rank 0 only, +1.3%, was dropped | `31cba11`, `907652f` |
| retrace summation tolerance 7 or 5 ppm | -1.0%, -2.3% | `f544cd7` |
| `rt_im_train.rt_window_multiplier` 1.5 or 2.0 with retrace on | -1.8% (13,429), -2.9% (13,282), decoy fraction 0.99%; at 1.5 about 100 precursors came back inside the window and 373 more lost to a wrong peak inside it | `1c5b9a1` |
| 1/K0 window narrowed to `apex_im` +/- delta | delta 0.008 -8.0%, 0.012 -1.4%, 0.020 +1.4% (seed 0); not recommended | `31cba11` |
| `convert.tdf_mz_ppm 5` | +3.6%, but the seed fell to 615 anchors; superseded by the valley split | `31cba11` |
| `tdf_min_points 1` | -17% (P1); for MS1 only +0.25% / -0.2% | `76d57b6`, `31cba11` |
| valley fraction and smoothing | m/z 0.3 -6.1%; smoothing 12 ppm -4.4%; m/z 0.7 +1.6%, nothing once the mobility split and count-once are on | `31cba11` |
| refit widths as fitted, and fragment mass recalibration on identifications | -4.0%; 7.99 ppm -2.7%, 10.11 ppm -0.3% | `31cba11` |
| `peak_claim = coelution_shadow`, contested-evidence features | -8.7%, -0.4% (seed 0) | `31cba11` |

## 13. How losses were diagnosed

The IM branch's levers were chosen from a precursor-level loss classifier
(`bench/tims/tims_loss.py`, from `31cba11`), not from count sweeps. It takes a reference set of
precursors identified at 1% by an independent search of the same run (17,591 on E. coli),
keyed on the stripped sequence with I and L merged plus the charge, pairs each with
MuMDIA's best-scoring target row, and puts it in exactly one class:

- not in the library;
- not extracted: retention time outside the window, 1/K0 outside the window, or both inside
  (the extraction gate or the presence rules);
- accepted at pooled PSM `q_value` 1%;
- right peak, low score: the apex within 5 s of the reference retention time, not accepted;
- wrong peak inside the RT window;
- wrong peak outside the RT window.

On the P6 base: 0 not in the library; 194, 173 and 395 not extracted (RT, 1/K0, gate or
presence; 270 of the 395 in the faintest quintile); 10,138 accepted (57.6%); 4,327 right peak
with a low score (24.6%, q quartiles 0.049 / 0.209 / 0.758); 1,384 wrong peak inside the
window (7.9%); 980 wrong peak outside it (5.6%, of which 436 were modform confusions, 431 of
them in the Met-oxidation count). The losses were faint precursors, not borderline ones:
acceptance by abundance quintile, faintest first, was 19.6 / 38.7 / 59.2 / 77.8 / 92.9%, and
only 6.9% of the rejects had q between 1% and 2%. With retrace on (`1c5b9a1`) the accepted
share rose from 77.6% to 80.8% and the right-peak rejects fell from 2,484 to 1,882; the
wrong-peak classes did not move (829 to 847 inside, 415 to 433 outside), which is what the
re-pick then addressed.

Follow-up diagnostics, each from the classifier's classes (`31cba11` unless marked):

- Right-peak rejects carry evidence at decoy level in every family (library intensity
  agreement and fragment co-elution: AUC 0.48-0.52 against score-matched decoys); fragment
  count, mass error, 1/K0 and retention time are not what they lack. On HYE (`0bb0984`) the
  same class had 30% less fragment signal than accepted precursors of the same reference
  abundance, and a perfect apex recovered about a quarter of that gap.
- An apex oracle: the reference apex was among the top 1 / 2 / 3 / 5 extracted peaks for
  31.3 / 67.6 / 86.6 / 98.7% of the wrong-peak-in-window class, the evidence behind
  `retain_top_peaks 5` and the re-pick.
- A retention-time oracle gave +2.0% (seed 0, diagnostic only).
- An imported spectral library on the same keys gave +8.3%, and the gain was its charge-2
  fragments; on the retrace base the same imported fragments lost 3.5% (`f544cd7`).

**D5, corrected.** The first MS1 diagnostic (D5) found the MS1 monoisotopic peak missing for
3,384 of 10,138 accepted precursors (33%), the nearest centroid a median -34.2 ppm from the
monoisotopic m/z, and attributed it to single-linkage chaining across scans at
`tdf_mz_ppm` 10. A 5 ppm follow-up seemed to rule chaining out; it had not tested a clean
split, because one TOF index is about 4.4-7 ppm wide below m/z 500 and 5 ppm linkage cuts
single ions. The m/z valley split settled it: MS1 found for 97.3% of accepted precursors
(from 66.5%), missing-MS1 precursors 3,837 to 346, median |ppm| of the found peak 3.3 (from
5.0). So D5's loss was chaining, and the valley split is its correction. Its effect is part
of the valley-split row of section 11, not a separate gain: MS1 presence is not a
discriminant by itself (targets 76.2%, decoys 75.0%).

**The raw-MS1 gain is a separate item.** Rebuilding `ms1_mono`, `ms1_iso1` and `ms1_iso2`
from the raw MS1 frame nearest each grid point, at 20 ppm and within `apex_im +/- 0.015`, gave
+1.6% (three seeds; +/- 0.025: +1.5%) with entrapment real peptides +1.85% at an FDP of
0.44-0.46% against 0.42-0.46% (`907652f`). It is the origin of `retrace.ms1`, which stays a
switch until arm T10 repeats the ablation on main.

On HYE run 0 (`caeede4`, `0bb0984`), against the reference search's precursors: the re-pick
fixed 9,014 and broke 2,464; the reference peak was outside the top 5 for 4,257, 905 of them
outside the RT window. Of the remaining per-run gap, 97.4% is charge 2 and 47.7% is peptides
of length 7 to 9. Retention-time error (median 3.8 s), 1/K0 error (0.0099), decoy shadowing
and picked competition were ruled out as causes.

## 14. Open levers and open measurements

**Fragment-model fine-tuning.** On HYE diaPASEF, AlphaPeptDeep `generic` fine-tuned for 10
epochs (learning rate 1e-4) on 79,937 precursors at pooled q 0.001 raised held-out cosine from
0.9720 to 0.9834 and target rows per run by 3.4% (90,514 to 93,604), with peptides +2.5%,
ProteoBench ions +3.5% and global / eq / CV 0.132 / 0.181 / 0.085 against 0.130 / 0.177 / 0.083
(`0bb0984`, with the `bench/ms2_finetune` scripts that stay on the IM branch). It was not
cross-fitted: the model was trained on the runs it then scored. The earlier E. coli fine-tune,
cross-fitted, lost 1.5% (section 12; recalled in `f90809c`). A cross-fitted HYE run and an E.
coli entrapment run decide between a real gain and train/score overlap. It is not ported; it
can return as its own measured proposal.

**Second-pass and quant measurements on TIMS data.** Main's second pass and the cross-run
quant have run on one diaPASEF pool, the six 5 min HYE runs of section 11, with one seed and
no multi-run entrapment. The second pass held the pooled decoy fraction, but its transferred
rows carried twice the decoy rate of the anchored rows (1.63% against 0.82%). A multi-run
entrapment on HYE and a second known-ratio diaPASEF set decide whether the second pass and
the cross-run quant become TIMS defaults.

Planned arms, each against main's integrated base with the recipe. The gates are named G0 to
G5: G0 byte identity of every 3D artifact, report, manifest entry and config hash, G1 the E. coli run with three paired seeds on one host, G2 E. coli entrapment (ratio
0.574592 for the recipe's library) and, for the second pass, a multi-run entrapment on HYE, G3 the six-run HYE
diaPASEF set, G4 ProteoBench HYE at `min_obs` 3, G5 wall time and process-tree peak against a
64 GB budget.

| arm | question |
|---|---|
| T1 | does an RT recentre on identifications add to `rt_bootstrap`? |
| T2 | `multihead_anchors = search` against `auto` on TIMS, after the guards: measured, `auto` now chooses `search` on 1/K0 data (section 7) |
| T3 | fragment tolerance estimator: p95 two-pass, MAD k = 4 two-pass and single-pass, p95 single-pass; also on the 3D benchmarks, to retire one |
| T3b | T3 with retrace and the re-pick on |
| T4 | `gate_min_score` 0, 0.3, 0.45, 0.6 under `window_mean` with the 1/K0 gate on |
| T5 | `apex_count_window` 5 against 3 |
| T6 | shadow demotion off, on, and on with `rescore.shadow_im_tolerance` |
| T7 | decoy 1/K0: copied against predicted, entrapment |
| T8 | second-pass 1/K0: consensus against calibrated library 1/K0 |
| T9 | second-pass 1/K0 floor: first-pass `w_im` against a fixed 0.04 |
| T10 | retrace MS1: `retrace.ms1` on and off, `ms1_scan_halfwidth` 0 against 1, `prec_tol_ppm` 3 against a TIMS value |
| T11 | `rt_window_multiplier` 1.0 against 1.5 on main |
| T12 | `im_gate` and `im_features` paired on and off, with entrapment and HYE seeds |
| T13 | the valley fractions (m/z 0.5 to 0.7) and count-once as one package on HYE, three seeds |

Each TIMS-only key moves to `auto` (or, for the valley fractions, to a TIMS default) only in
its own change, with entrapment and a second acquisition. The entrapment FASTA is made with
`bench/make_entrapment_fasta.py`; its ratio is a property of the library (0.563430 for the
IM branch's E. coli diaPASEF library, 0.574592 for the recipe's), so it is computed for each library (`bench/README.md`).

## 15. Limits

- **Reader scope.** timsrust 0.4.2 classifies frames by `MsMsType` only, and MuMDIA reads a
  `.d` natively only when it holds diaPASEF frames (type 9) and no ddaPASEF frames.
  Synchro-PASEF, midiaPASEF, slice-PASEF and older TDF compression are untested. Only Ultra 2
  E. coli and HYE runs were searched end to end; a timsTOF Pro run was only converted.
- **Calibration.** Only the first `TimsCalibration` row is used; a file with several logs a
  warning. Without a model 2 row the scan-to-1/K0 conversion is linear, with a warning.
- **Intensities.** `intensity_correction_factor` is ignored, which is exact only while the
  frames share one ramp time.
- **Refused on 4D input:** grouped runs (`groups.window_groups > 1`) and the prescreen, both
  keyed on m/z windows (section 3); retrace on a run not read natively from a `.d`; the
  cross-run quant step under grouped runs or `rt_library_scope = per_run`.
- **Window coverage.** The chromatogram grid's coverage test is on m/z only, exact while the
  slots of a scheme have disjoint m/z ranges (section 3).
- **Memory.** Retrace holds every frame inside the candidates' RT windows (21 GB on the
  E. coli run on the IM branch; the E. coli runs with retrace on main peaked at 18.7-18.8 GB;
  36-39 GB per HYE run on the IM branch). The pooled HYE rescore needed `MUMDIA_NN_STORE=f16` on a 64 GB node on the
  IM branch.
- **Evidence.** Most HYE numbers are one seed, no multi-run diaPASEF entrapment exists, and
  the quantification record rests on one known-ratio dataset.

## 16. Memory and speed

Measured on hippogriff (128 cores, 755 GB) on the six ProteoBench HYE diaPASEF runs (60 min
gradient, timsTOF SCP) at 120 threads, NN seed 0, with `MUMDIA_CACHE_DIR=off`, so both DeepLC
calibrations of the first run start cold. Peaks are GNU `time` maximum RSS of the engine
process unless named otherwise.

### Where it goes

| phase | wall | what takes it |
|---|---|---|
| first run alone | 18.8 min | conversion and seeds of all runs (1.6 min), the multi-head calibration (3.2), the RT bootstrap search (5.4), the refit and the first run's own chain (8.6) |
| five runs in parallel | 14.3 min | per run: retrace about 5.5 min, extract about 3, features 1.5-4 |
| pooled rescore | 14.9 min | `nn_torch` over 28.5 million PSMs |
| quantification and report | 1.9 min | |

The same engine and settings on six Astral runs (15 min gradients) took 12:07 at 23.3 GiB, and
on six Orbitrap AIF runs 13:33 at 30.1 GiB. A diaPASEF run costs more for three reasons that
the measurements below separate:

- **Extraction rejects little.** 4.9 million of the 14.7 million library candidates pass per
  run, against about 5% on Astral. On dense diaPASEF spectra almost every candidate matches
  its fragments: the identified ones match 11.3 of 12 on average, the rest 10.9. Retrace,
  features, the chromatogram tables and the pooled rescore all scale with that count.
- **Retrace holds the run's raw events** (section 6), 2.86 billion of them in a 60 min HYE
  run, two thirds of them MS1.
- **The fragment-peak matches of extract** are held for every window group in flight.

### Extract: windows in flight

`extract.windows_in_flight` defaults to 4 on spectra that carry 1/K0 (16 otherwise). Measured
standalone on run A_02 at 20 threads, the accepted candidates (4,907,155) and chromatogram rows
(73,607,325) identical at every setting:

| windows in flight | peak (anonymous heap) | wall |
|---|---|---|
| 16 (the previous default) | 45.1 GiB | 410 s (contended) |
| 8 | 31.9 GiB | 194 s |
| 4 | 21.4 GiB | 154 s |

None of the peak is file-backed. Threads change little: 16 in flight took 51.5 GiB at 60
threads and 51.9 at 120. On the six-run experiment the peak fell from 181 to 94 GiB, with every
identification, quantity and loss class bit-identical, in 50 against 51 minutes. The memory
record of the accumulator (`mem: extract hit accumulator`) reads its size after each batch,
not inside it, so it reported 0.0 throughout.

### Retrace

- **Readers.** The chromatogram readers come from a process-wide budget. With one call's
  worth (7 extra readers), the first of five concurrent retraces took all of them and the
  others read on one thread: the slowest spent 390 s of its 489 s reading. The budget is now
  four calls' worth (about 0.1 GB per reader).
- **MS1 sums once per row.** An MS1 grid point takes the largest sum over its nearest MS1
  frame and `extract.ms1_scan_halfwidth` neighbours either side, and consecutive points of a
  row share all of those frames but one, so every MS1 frame of a row was summed three times
  at the default. Each is now summed once per row, with every value bit-identical.
- **Skeleton centroid table.** Retrace rebuilds about 90% of the rows from raw events and
  reads only their axis and keys, but it decoded every centroid trace: the intensity and
  1/K0 lists are 86% of the centroid table's 11.8 GiB, and a read benchmark of one run's table
  took 13.9 s at 8 threads with them and 0.7 s without. Extract now writes the rows retrace
  rebuilds as zeros on their axis (`retrace::Skeleton`, recorded in the table's report), with
  the rule retrace classifies by, and retrace refuses a skeleton it would not rebuild in full.

Three changes to the summing kernel were measured on one run (standalone, 20 threads, 3:46
and 162 s of summing as the baseline) and left out, every output byte-identical: events
grouped by mobility scan instead of by TOF (22:12; a fragment's TOF window covers only a few
TOF indices, so one bucket lookup beats about 28 per-scan searches), candidates visited in
retention-time order (3:46), and every lookup of a row group sorted by frame before summing
(10:48: building and sorting the requests cost more than the cache misses they saved).
Smaller TOF buckets (16 or 4 indices) and huge pages changed nothing. A micro-benchmark on the
run's real frames puts one lookup at about 110 ns in retrace's order and 54 ns frame by frame.

### MS1 noise

The dnoise tool (P. T. Garrett, J. K. Diedrich and J. R. Yates III, bioRxiv 2026,
doi:10.64898/2026.08.27.747603, MIT licence) removes MS1 points that do not form a run of at
least 5 occupied mobility scans, points in the halo of a stronger ion, and MS1 points outside
every diaPASEF window. On these runs its default (MS1 only) kept 8.7% of the MS1 points and
40% of all points, cut the MS1 centroids from 254 million to 8.6 million per run, and left the
identifications unchanged (89,973 against 90,363 ions at k = 3, precursors 109,962 against
109,965) at 51 against 63 minutes. Its MS/MS mode kept 17% of all points and lost 9.2% of the
ions, as `convert.tdf_min_points` 3 to 6 did (-5.9% to -20.1%): no MS/MS peak filter tried
here is free.

The native reader now applies the same MS1 filter itself (`convert.tdf_ms1_denoise`, on by
default since 2026-10-17), and retrace applies it to the raw MS1 frames it holds, so no
pre-processing step is needed. On the six runs it gives exactly the identifications of the
tool's own output (98,524 peptides at 1%, 89,973 ions at k = 3), at about 33 s more conversion
per run than reading files the tool had cleaned. On the 15 min E. coli run, three NN seeds per
arm: 14,121 against 14,044 peptides at 1%, and on the entrapment library 13,384 against 13,375
real peptides at an entrapment FDP of 0.30-0.38% against 0.39-0.46%.
