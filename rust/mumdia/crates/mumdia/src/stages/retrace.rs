//! Chromatogram traces rebuilt from the raw timsTOF events (`retrace`, diaPASEF only;
//! docs/35_diapasef.md section 6, docs/09_extract.md section 9).
//!
//! Extract's trace point is the intensity of one centroid, and convert's centroiding drops
//! or merges weak signal (single-linkage m/z chaining, `convert.tdf_min_points`, the scan
//! gap). Here every point of a fragment trace becomes the sum of all raw events in the grid
//! point's frame, in the quad slots whose isolation window holds the precursor, within the
//! learned fragment tolerance of the calibrated fragment m/z and inside the candidate's
//! 1/K0 band `apex_im +/- retrace.im_half_width`; the point's 1/K0 is their
//! intensity-weighted 1/K0. The MS1 isotope traces (`ms1_mono`, `ms1_iso1`, `ms1_iso2`) are
//! rebuilt from the MS1 frames extract samples (`extract::ms1_neighbours`), within
//! `extract.prec_tol_ppm` of the MS1 row's own m/z and inside the same band. Every other
//! row and column is copied, the rows keep their order and the table keeps its layout, so
//! features (which joins the chromatograms by stream order) and quant read the result
//! unchanged. Label-blind.
//!
//! Under `retrace.repick` the apex retention time and 1/K0 centre of every candidate are
//! first chosen again on the raw events, among extract's apex and the peaks of the
//! `extract.retain_top_peaks` sidecar (`joint_score`), and the traces are rebuilt around
//! the chosen centre.
//!
//! Retrace then patches `psms_extracted` once, in place: `apex_rt` and `apex_im` of a
//! re-picked candidate, the apex scalars read from the rebuilt traces (`apex_intensity`,
//! `n_matched_fragments`, `ms1_mono`, `ms1_iso1`, `ms1_iso2`), and under
//! `retrace.apex_features` the column `imc_ref_w`. The table keeps its schema version (v4).
//!
//! The frames inside the candidates' RT windows are decoded once and held
//! (`convert::tims_frames::TofFrame`); the chromatograms are then streamed one row group at
//! a time, so memory does not grow with the number of trace points.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, Int32Array, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use mumdia_core::config::{Config, ExtractConfig, RetraceConfig};
use mumdia_io::report::{ArtifactReport, Written};
use mumdia_io::table::{write_table, BatchWriter, Col, TableFile, WriteOptions};
use rayon::prelude::*;
use serde_json::json;
use timsrust::{Frame, MSLevel};
use tracing::{info, warn};

use crate::chromatograms::{self, Layout, Optional, Rows};
use crate::masscal::{read_mass_cal, MassOffset};
use crate::sched::DecoderLease;
use crate::stages::convert::ms1_denoise::Ms1Denoise;
use crate::stages::convert::tims_frames::{scan_band, slot_scans, sum_events, Sum, TofFrame};
use crate::stages::convert::RawTdf;
use crate::stages::extract::{ms1_neighbours, nearest_index, read_run_windows};
use crate::stats::{loo_ref_corr, LooRef};

/// The file extract writes its centroid traces to when retrace runs after it, beside
/// `chromatograms.parquet`. Deleted once retrace has written the rebuilt table.
pub const CENTROID_NAME: &str = "chromatograms.centroid.parquet";

/// Frames within this of the candidates' RT range are decoded (the nearest MS1 frame of an
/// edge grid point can lie just outside it).
const RT_MARGIN_S: f64 = 5.0;
/// A fragment grid point must sit on an MS2 frame's RT within this (f32 rounding of the
/// frame RT is ~1e-4 s at the end of a gradient; frames are ~60 ms apart).
const RT_MATCH_S: f64 = 1e-3;

/// Chromatogram readers a call takes beyond its first: up to its row-group count and its
/// thread pool allow, so `--threads 1` reads on one thread; 1 + 7 is the eight readers the
/// stage was measured with.
const READER_EXTRAS: usize = 7;

/// Concurrent retrace calls that each get all [`READER_EXTRAS`] from the process-wide
/// budget ([`DecoderLease`]). Each reader holds about one decoded row group of traces
/// (~0.1 GB), so the budget stays process-wide rather than multiplying freely with
/// `experiment.parallel_runs`. With a budget of one call's extras, the first of five
/// concurrent retraces on HYE diaPASEF took all seven, the others read on one thread, and
/// the slowest spent 390 s of its 489 s reading: one call's readers then set the wall time
/// of the whole parallel phase. An automatic `run-experiment` plan grows the budget to
/// every concurrent chain's share while its chains run ([`grow_readers_for_chains`]).
const READER_CALLS: usize = 4;

/// Extra readers of the process-wide pool not currently leased.
static READER_BUDGET: AtomicUsize = AtomicUsize::new(READER_EXTRAS * READER_CALLS);

/// Grow the reader pool so that `chains` concurrent retrace calls can each take all
/// [`READER_EXTRAS`], until the guard drops; `None` when the pool already holds that many.
/// For the per-run chains of an automatic `run-experiment` plan (`sched::BudgetGrowth`).
pub(crate) fn grow_readers_for_chains(chains: usize) -> Option<crate::sched::BudgetGrowth> {
    let add = (READER_EXTRAS * chains).saturating_sub(READER_EXTRAS * READER_CALLS);
    (add > 0).then(|| crate::sched::BudgetGrowth::add(&READER_BUDGET, add))
}

// `retrace.apex_features` (docs/35_diapasef.md section 6): the apex mobility agreement is
// scored on each fragment's raw 1/K0 profile over `apex_im +/- APEX_IM_HW`, in APEX_BINS
// bins, summed over the MS2 frames within APEX_RT_HW_S of the apex.
const APEX_IM_HW: f64 = 0.04;
const APEX_BINS: usize = 40;
const APEX_IM_BIN: f64 = 2.0 * APEX_IM_HW / APEX_BINS as f64;
const APEX_RT_HW_S: f64 = 1.5;

// `retrace.repick`: the 1/K0 profile of each candidate peak is binned at REPICK_BIN over the
// run_windows 1/K0 window and scored in bands of 2 * REPICK_BAND_BINS + 1 bins (+/- 0.015),
// over the MS2 frames within REPICK_RT_HW_S of the peak's apex.
const REPICK_BIN: f64 = 0.002;
const REPICK_BAND_BINS: usize = 7;
const REPICK_RT_HW_S: f64 = 1.5;

pub struct RetraceParams<'a> {
    /// The diaPASEF `.d` the run was converted from.
    pub raw: &'a str,
    /// Extract's chromatograms (centroid traces), in either layout, with the per-point
    /// 1/K0 list.
    pub chromatograms: &'a str,
    /// Extract's `psms_extracted`, with its top-K peak sidecar beside it under
    /// `retrace.repick`.
    pub psms_extracted: &'a str,
    /// Where the patched `psms_extracted` goes: the input itself in the orchestrators (a
    /// patch in place), another path for a standalone call that keeps its input.
    pub out_psms: &'a str,
    pub run_windows: &'a str,
    pub library_precursors: &'a str,
    /// search-seed `masscal.json`; without one, `extract.frag_tol_ppm` and no offset.
    pub mass_cal: Option<&'a str>,
    /// The extraction settings the centroid traces were built with: `frag_tol_ppm` (the
    /// fallback tolerance), `prec_tol_ppm` and `ms1_scan_halfwidth`.
    pub extract: &'a ExtractConfig,
    /// The rebuilt chromatograms.
    pub out: &'a str,
    pub cfg: &'a RetraceConfig,
    pub config_hash: &'a str,
    /// `convert.tdf_ms1_denoise`: the MS1 frames held here pass the MS1 noise filter convert
    /// applied, so the rebuilt MS1 traces read the same MS1 events as extract's.
    pub ms1_denoise: bool,
}

/// The two tables retrace wrote, with the hashes their reports carry.
pub struct RetraceWritten {
    pub chromatograms: Written,
    pub psms_extracted: Written,
}

/// The `.d` the spectra artifact `ms2_spectra` was converted from, by the native reader.
/// Retrace sums raw events, so a run whose spectra came from any other reader (an mzML,
/// including a timsTOF mzML written by msconvert) is refused with the reason.
pub fn raw_input(ms2_spectra: &str) -> Result<String> {
    let prov = crate::instrument::provenance(ms2_spectra);
    match prov {
        Some(p) if p.has_im() => {
            let raw = p.source.ok_or_else(|| {
                anyhow!(
                    "retrace.enabled: {ms2_spectra} was read natively from a timsTOF .d, but \
                     its report does not name the .d (params.mzml)"
                )
            })?;
            if !std::path::Path::new(&raw).is_dir() {
                bail!(
                    "retrace.enabled: {ms2_spectra} was converted from {raw}, which is no \
                     longer there; retrace reads the raw events from it"
                );
            }
            Ok(raw)
        }
        _ => bail!(
            "retrace.enabled needs a diaPASEF .d read by the native timsTOF reader, and \
             {ms2_spectra} was not (an mzML input, including a timsTOF mzML converted by \
             msconvert, has no raw events to sum). Set retrace.enabled = false for this \
             input (docs/35_diapasef.md section 6)"
        ),
    }
}

/// Refuse `retrace.enabled` before any stage runs when an input will not be read natively
/// from a diaPASEF `.d` (the raw dispatch, `instrument::input_has_im`, which decides the
/// library build as well), so a run does not fail at its first retrace after the search.
pub fn refuse_inputs_without_raw_events<S: AsRef<str>>(cfg: &Config, inputs: &[S]) -> Result<()> {
    if !cfg.retrace.enabled {
        return Ok(());
    }
    for input in inputs {
        let input = input.as_ref();
        if !crate::instrument::input_has_im(input) {
            bail!(
                "retrace.enabled needs every input to be a diaPASEF .d read by the native \
                 timsTOF reader, and {input} is not (an mzML, including a timsTOF mzML \
                 converted by msconvert, has no raw events to sum). Set retrace.enabled = \
                 false for this input (docs/35_diapasef.md section 6)"
            );
        }
    }
    Ok(())
}

/// The orchestrators' first hook: the `.d` to retrace when `retrace.enabled` and the pass
/// allows it (`allowed`; the RT bootstrap pass does not), `None` otherwise. Called before
/// extract, because extract then writes its centroid traces to [`CENTROID_NAME`] with their
/// per-point 1/K0.
pub fn raw_if_enabled(cfg: &Config, allowed: bool, ms2_spectra: &str) -> Result<Option<String>> {
    if !(allowed && cfg.retrace.enabled) {
        return Ok(None);
    }
    raw_input(ms2_spectra).map(Some)
}

/// The paths of one run's chain that retrace reads and writes.
pub struct ChainFiles<'a> {
    /// Extract's centroid chromatograms ([`CENTROID_NAME`]); deleted after success.
    pub centroid: &'a str,
    pub psms_extracted: &'a str,
    pub run_windows: &'a str,
    pub library_precursors: &'a str,
    pub mass_cal: &'a str,
    /// `chromatograms.parquet`.
    pub out: &'a str,
}

/// The orchestrators' second hook, after extract and before features: rebuild the traces
/// when [`raw_if_enabled`] returned a `.d`, then delete the centroid intermediate. `None`
/// when retrace does not run in this pass.
pub fn run_if_enabled(
    raw: Option<&str>,
    cfg: &Config,
    extract: &ExtractConfig,
    config_hash: &str,
    f: ChainFiles,
) -> Result<Option<RetraceWritten>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    info!(stage = %"retrace", "run: stage start");
    let w = run_hashed(RetraceParams {
        raw,
        chromatograms: f.centroid,
        psms_extracted: f.psms_extracted,
        out_psms: f.psms_extracted,
        run_windows: f.run_windows,
        library_precursors: f.library_precursors,
        mass_cal: Some(f.mass_cal),
        extract,
        out: f.out,
        cfg: &cfg.retrace,
        config_hash,
        ms1_denoise: cfg.convert.tdf_ms1_denoise,
    })?;
    // The centroid table is read by nothing once the rebuilt one is written (12-17 GB per
    // HYE diaPASEF run).
    let (files, bytes) = crate::stages::delete_intermediate(f.centroid, "retrace intermediate")?;
    info!(
        files,
        bytes, "retrace: deleted the centroid chromatograms it rebuilt"
    );
    Ok(Some(w))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Copy,
    Frag,
    Ms1,
}

/// How retrace treats one chromatogram row: rebuilt from the raw events as a fragment or an
/// MS1 isotope trace, or copied as extract wrote it. A row is rebuilt when it has a grid,
/// its precursor and its own m/z are finite, its candidate has a finite 1/K0 centre, and it
/// is an MS1 row under `retrace.ms1` or a fragment with a positive prediction. Extract
/// evaluates the same rule when it writes a skeleton table ([`Skeleton`]), so a row it
/// leaves as zeros is one this stage rebuilds.
fn row_kind(
    has_points: bool,
    precursor_mz: f64,
    centre: f64,
    row_mz: f64,
    name: &str,
    predicted_intensity: f32,
    ms1: bool,
) -> Kind {
    let ok = has_points && precursor_mz.is_finite() && centre.is_finite() && row_mz.is_finite();
    match (ok, name.starts_with("ms1_")) {
        (true, true) if ms1 => Kind::Ms1,
        (true, false) if predicted_intensity > 0.0 => Kind::Frag,
        _ => Kind::Copy,
    }
}

/// A centroid table written for this stage to rebuild (`ExtractParams::retrace_skeleton`):
/// the traces of the rows retrace rebuilds from raw events are left as zeros on their axis,
/// because retrace reads only the axis and the keys of those rows, and decoding centroid
/// values it then discards was most of its read time (95% of the decode on a HYE diaPASEF
/// run). `ms1`: whether the MS1 isotope rows were among them (`retrace.ms1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Skeleton {
    pub ms1: bool,
}

impl Skeleton {
    /// The report parameter of a skeleton centroid table: `{"ms1": bool, "rows": n}`.
    pub const PARAM: &'static str = "retrace_skeleton";

    /// Whether extract leaves the row as zeros: exactly the rows `row_kind` rebuilds.
    pub fn zeroes(
        &self,
        has_points: bool,
        precursor_mz: f64,
        centre: f64,
        row_mz: f64,
        name: &str,
        predicted_intensity: f32,
    ) -> bool {
        row_kind(
            has_points,
            precursor_mz,
            centre,
            row_mz,
            name,
            predicted_intensity,
            self.ms1,
        ) != Kind::Copy
    }

    /// The skeleton a centroid table records in its report, if any.
    pub fn recorded(path: &str) -> Result<Option<Skeleton>> {
        let Ok(text) = std::fs::read_to_string(format!("{path}.report.json")) else {
            return Ok(None);
        };
        let v: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("{path}.report.json is not valid JSON"))?;
        match v.get("params").and_then(|p| p.get(Skeleton::PARAM)) {
            None => Ok(None),
            Some(s) => {
                let ms1 = s.get("ms1").and_then(|x| x.as_bool()).ok_or_else(|| {
                    anyhow!(
                        "{path}.report.json: {} without a boolean 'ms1'",
                        Skeleton::PARAM
                    )
                })?;
                Ok(Some(Skeleton { ms1 }))
            }
        }
    }
}

/// One rebuilt row: its kind, its intensity and 1/K0 lists, the raw events summed, the
/// grid points without a decoded frame, and for a rebuilt `ms1_mono` row the
/// intensity-weighted ppm residual of its raw events against the row's m/z.
struct RowOut {
    kind: Kind,
    int: Vec<f32>,
    im: Vec<f32>,
    events: u64,
    misses: u64,
    ms1_residual_ppm: Option<f64>,
}

/// One planned grid point: the decoded frames to sum in (none: no frame at that RT; one for
/// a fragment point; the [`ms1_neighbours`] for an MS1 point) and the scan ranges to sum.
type Point<'a> = (Vec<&'a TofFrame>, Vec<(usize, usize)>);

/// One row group of the input chromatograms, in the v1 form.
#[derive(Default)]
struct SpanIn {
    cid: Vec<u32>,
    name: Vec<String>,
    fmz: Vec<f64>,
    obs: Vec<f64>,
    pint: Vec<f32>,
    rt: Vec<Vec<f32>>,
    int: Vec<Vec<f32>>,
    im: Vec<Vec<f32>>,
}

impl SpanIn {
    /// The rows `[first, first + n)`, a whole row group, decoded to their v1 form.
    fn read(ch: &TableFile, first: usize, n: usize) -> Result<SpanIn> {
        let t = ch.span(first, n)?;
        let mut s = SpanIn::default();
        chromatograms::for_each_row(&t, |r| {
            s.cid.push(r.cid);
            s.name.push(r.name.to_string());
            s.fmz.push(r.frag_mz.unwrap_or(f64::NAN));
            s.obs.push(r.frag_obs_mz.unwrap_or(0.0));
            s.pint.push(r.predicted_intensity);
            s.rt.push(r.rt.to_vec());
            s.int.push(r.intensity.to_vec());
            s.im.push(r.im.unwrap_or(&[]).to_vec());
            Ok(())
        })?;
        if s.cid.len() != n {
            bail!(
                "{}: rows {first}..{} read back as {} rows",
                ch.path(),
                first + n,
                s.cid.len()
            );
        }
        Ok(s)
    }
}

/// The apex record of one candidate from its rebuilt rows. A candidate can straddle two
/// input row groups, so partial records are merged ([`Apex::merge`]) before
/// [`Apex::finish`].
struct Apex {
    cid: u32,
    /// Fragment rows rebuilt from raw events. With none, extract's fragment scalars stay.
    rebuilt: u32,
    /// Sum over fragment rows of the trace point nearest the apex (`apex_intensity`).
    inten: f64,
    /// Fragment rows with any signal in the trace (`n_matched_fragments`).
    n_matched: i32,
    /// `ms1_mono`, `ms1_iso1`, `ms1_iso2` at the point nearest the apex, from a rebuilt MS1
    /// row; `None` where that row was not rebuilt, and extract's value stays.
    ms1: [Option<f64>; 3],
    /// (predicted intensity, raw 1/K0 profile) per rebuilt fragment row, under
    /// `retrace.apex_features`.
    prof: Vec<(f64, [f64; APEX_BINS])>,
}

/// A finished candidate: id, the fragment scalars (`None`: not refreshed), the MS1 apex
/// values and `imc_ref_w`. The profiles are dropped here, so only these stay resident (a
/// HYE run has ~7M candidates; their profiles are ~28 GB).
#[derive(Clone, Copy, Debug, PartialEq)]
struct ApexRow {
    cid: u32,
    frag: Option<(f32, i32)>,
    ms1: [Option<f64>; 3],
    imc: f64,
}

impl Apex {
    fn new(cid: u32) -> Apex {
        Apex {
            cid,
            rebuilt: 0,
            inten: 0.0,
            n_matched: 0,
            ms1: [None; 3],
            prof: Vec::new(),
        }
    }

    fn finish(self) -> ApexRow {
        ApexRow {
            cid: self.cid,
            frag: (self.rebuilt > 0).then_some((self.inten as f32, self.n_matched)),
            ms1: self.ms1,
            imc: self.imc(),
        }
    }

    fn merge(&mut self, o: Apex) {
        self.rebuilt += o.rebuilt;
        self.inten += o.inten;
        self.n_matched += o.n_matched;
        for (a, b) in self.ms1.iter_mut().zip(o.ms1) {
            if a.is_none() {
                *a = b;
            }
        }
        self.prof.extend(o.prof);
    }

    /// `imc_ref_w`: over the fragments with signal, the predicted-intensity-weighted mean
    /// leave-one-out reference correlation of each 1/K0 profile against the weighted sum of
    /// the others (`stats::loo_ref_corr`). 0 when none has signal.
    fn imc(&self) -> f64 {
        let mut tot = [0.0f64; APEX_BINS];
        for (w, p) in &self.prof {
            for (t, v) in tot.iter_mut().zip(p) {
                *t += w * v;
            }
        }
        let profiles: Vec<&[f64]> = self.prof.iter().map(|(_, p)| &p[..]).collect();
        let weights: Vec<f64> = self.prof.iter().map(|(w, _)| *w).collect();
        let (mut num, mut den) = (0.0, 0.0);
        for (f, (w, p)) in self.prof.iter().enumerate() {
            if p.iter().sum::<f64>() > 0.0 {
                num += w * loo_ref_corr(&profiles, &weights, f, LooRef::Subtract(&tot));
                den += w;
            }
        }
        if den > 0.0 {
            num / den
        } else {
            0.0
        }
    }
}

/// One candidate peak on the raw events: over the 1/K0 bands of `[a, b]`, the band with the
/// best `cos x sqrt(I / I_max)`, where `cos` is the cosine of the square-rooted band sums of
/// the fragments against the square-rooted predictions and `I` the prediction-weighted
/// band intensity. Returns that band's `cos x ln(1 + I)` and its centre; (0, NaN) without
/// signal. `wins` holds each fragment's TOF window, `pred` its predicted intensity.
fn joint_score(
    fl: &[(&TofFrame, Vec<(usize, usize)>)],
    wins: &[(u32, u32)],
    pred: &[f64],
    scan_im: &[f64],
    a: f64,
    b: f64,
) -> (f64, f64) {
    let nb = ((b - a) / REPICK_BIN).ceil().max(1.0) as usize;
    let nf = wins.len();
    let mut prof = vec![0.0f64; nf * nb];
    for (f, &(t_lo, t_hi)) in wins.iter().enumerate() {
        let row = &mut prof[f * nb..(f + 1) * nb];
        for (fr, scans) in fl {
            fr.visit(scans, t_lo, t_hi, scan_im.len(), |sc, _, w| {
                let k = ((scan_im[sc] - a) / REPICK_BIN) as isize;
                row[k.clamp(0, nb as isize - 1) as usize] += w;
            });
        }
    }
    // Band sums, fragment-major.
    let k = REPICK_BAND_BINS;
    let mut band = vec![0.0f64; nf * nb];
    for f in 0..nf {
        let row = &prof[f * nb..(f + 1) * nb];
        for i in 0..nb {
            band[f * nb + i] = row[i.saturating_sub(k)..(i + k + 1).min(nb)].iter().sum();
        }
    }
    let inten: Vec<f64> = (0..nb)
        .map(|i| (0..nf).map(|f| pred[f] * band[f * nb + i]).sum())
        .collect();
    let mx = inten.iter().cloned().fold(0.0, f64::max);
    if mx <= 0.0 {
        return (0.0, f64::NAN);
    }
    let norm_p = pred.iter().sum::<f64>().sqrt();
    let cos = |i: usize| {
        let (mut dot, mut n2) = (0.0, 0.0);
        for f in 0..nf {
            let w = band[f * nb + i];
            dot += pred[f].sqrt() * w.sqrt();
            n2 += w;
        }
        dot / (norm_p * n2.sqrt() + 1e-12)
    };
    let (mut bi, mut bkey, mut bcos) = (0usize, f64::NEG_INFINITY, 0.0);
    for (i, &v) in inten.iter().enumerate() {
        let c = cos(i);
        let key = c * (v / mx).sqrt();
        if key > bkey {
            (bi, bkey, bcos) = (i, key, c);
        }
    }
    // An ion is narrower than the band, so the score is flat wherever the band holds all of
    // it and the first maximum sits at that plateau's low-1/K0 end. The centre is therefore
    // the prediction-weighted 1/K0 centroid of the profile inside the winning band.
    let (j0, j1) = (bi.saturating_sub(k), (bi + k + 1).min(nb));
    let (mut sw, mut sx) = (0.0, 0.0);
    for j in j0..j1 {
        let v: f64 = (0..nf).map(|f| pred[f] * prof[f * nb + j]).sum();
        sw += v;
        sx += v * (a + (j as f64 + 0.5) * REPICK_BIN);
    }
    (bcos * inten[bi].ln_1p(), sx / sw)
}

/// The value of the trace point nearest `t` (the first on a tie), 0 for an empty trace.
fn nearest_point(rt: &[f32], int: &[f32], t: f64) -> f64 {
    let mut best = (f64::MAX, 0.0);
    for (&r, &v) in rt.iter().zip(int) {
        let d = (r as f64 - t).abs();
        if d < best.0 {
            best = (d, v as f64);
        }
    }
    best.1
}

/// Inclusive TOF index window of the m/z interval `[lo, hi]`.
fn tof_window(raw: &RawTdf, lo: f64, hi: f64) -> (u32, u32) {
    let a = raw.tof_of(lo).ceil().max(0.0);
    let b = raw.tof_of(hi).floor().max(0.0);
    (a as u32, b as u32)
}

/// The decoded MS1 frames an MS1 grid point at `t` sums: the frame nearest `t` and its
/// `half` neighbours either side ([`ms1_neighbours`], extract's `ms1_scan_halfwidth`),
/// those that were decoded. `half = 0` is the nearest frame alone.
fn ms1_frames<'a>(
    t: f64,
    half: usize,
    ms1_rt: &[f64],
    ms1_fi: &[usize],
    frames: &'a [Option<TofFrame>],
) -> Vec<&'a TofFrame> {
    ms1_neighbours(nearest_index(ms1_rt, t), half, ms1_rt.len())
        .filter_map(|k| frames[ms1_fi[k]].as_ref())
        .collect()
}

/// The raw sum of one grid point: per frame the events in the TOF window and `scans`, and
/// the frame with the largest sum kept (the first on a tie), as extract's MS1 read takes
/// the maximum over its scans. A fragment point has one frame.
fn point_sum(
    frames: &[&TofFrame],
    scans: &[(usize, usize)],
    tof: (u32, u32),
    scan_im: &[f64],
) -> Sum {
    let mut best = Sum::default();
    for f in frames {
        let s = sum_events(f, scans, tof.0, tof.1, scan_im);
        if s.inten > best.inten {
            best = s;
        }
    }
    best
}

/// The frame sums of one MS1 row, kept while its grid is walked. An MS1 grid point takes the
/// largest sum over its nearest MS1 frame and `extract.ms1_scan_halfwidth` neighbours either
/// side, and consecutive points of a row share all of those frames but one, so each MS1
/// frame of a row was summed `2 * halfwidth + 1` times (3 at the default): about two thirds
/// of retrace's MS1 lookups. Every point of a row sums the same TOF window in the same scans,
/// so a frame's sum does not depend on the point that asks, and [`Ms1Seen::point_sum`]
/// returns bit for bit what [`point_sum`] returns.
#[derive(Default)]
struct Ms1Seen {
    /// (frame address, its sum), oldest first.
    sums: Vec<(usize, Sum)>,
}

impl Ms1Seen {
    /// The grid walks forward in retention time, so only the last few frames come back.
    const KEEP: usize = 16;

    fn point_sum(
        &mut self,
        frames: &[&TofFrame],
        scans: &[(usize, usize)],
        tof: (u32, u32),
        scan_im: &[f64],
    ) -> Sum {
        let mut best = Sum::default();
        for f in frames {
            let key = *f as *const TofFrame as usize;
            let s = match self.sums.iter().rev().find(|(k, _)| *k == key) {
                Some(&(_, s)) => s,
                None => {
                    let s = sum_events(f, scans, tof.0, tof.1, scan_im);
                    if self.sums.len() == Self::KEEP {
                        self.sums.remove(0);
                    }
                    self.sums.push((key, s));
                    s
                }
            };
            if s.inten > best.inten {
                best = s;
            }
        }
        best
    }
}

/// Per-candidate lookups indexed by `candidate_id`, NaN where absent.
fn by_candidate(ids: &[u32], vals: impl Iterator<Item = f64>, n: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; n];
    for (&c, x) in ids.iter().zip(vals) {
        if let Some(slot) = v.get_mut(c as usize) {
            *slot = x;
        }
    }
    v
}

/// Median of `v` (NaN when empty), in a total order so it does not depend on the input
/// order.
fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// What the re-pick reads once per call of the stage (`retrace.repick`): extract's top-K
/// peak sidecar `(candidate, rank, apex RT)` and every candidate's predicted fragment rows
/// `(candidate, calibrated m/z, predicted intensity)`, both sorted by candidate.
struct RepickData {
    alt: Vec<(u32, i32, f64)>,
    fr: Vec<(u32, f64, f64)>,
}

/// A re-pick: candidate, apex RT, 1/K0 centre.
type Pick = (u32, f64, f64);
/// One scored peak of the re-pick dump: candidate, order (0 = extract's apex, then the
/// sidecar ranks), apex RT, score.
type ScoredPeak = (u32, u32, f64, f64);

impl RepickData {
    fn load(p: &RetraceParams, ch: &TableFile, mass_off: &MassOffset) -> Result<RepickData> {
        let pk_path = format!("{}.peaks.parquet", p.psms_extracted);
        let pk = TableFile::open(&pk_path).with_context(|| {
            format!(
                "retrace.repick needs extract's top-K peak sidecar {pk_path} \
                 (extract.retain_top_peaks >= 2)"
            )
        })?;
        let mut alt: Vec<(u32, i32, f64)> = pk
            .u32("candidate_id")?
            .into_iter()
            .zip(pk.i32("peak_rank")?)
            .zip(pk.f64("apex_rt")?)
            .map(|((c, r), t)| (c, r, t))
            .collect();
        alt.sort_by_key(|x| (x.0, x.1));
        // Fragment rows (predicted, not MS1) by candidate: calibrated m/z and prediction.
        // These are plain columns in both layouts.
        let (name_id, names) = ch.str_interned("frag_name")?;
        let is_ms1: Vec<bool> = names.iter().map(|n| n.starts_with("ms1")).collect();
        let (f_cid, f_mz, f_pi) = (
            ch.u32("candidate_id")?,
            ch.f64("frag_mz")?,
            ch.f32("predicted_intensity")?,
        );
        let mut fr: Vec<(u32, f64, f64)> = (0..ch.nrows)
            .filter(|&r| f_pi[r] > 0.0 && !is_ms1[name_id[r] as usize])
            .map(|r| {
                (
                    f_cid[r],
                    f_mz[r] * mass_off.factor_at(f_mz[r]),
                    f_pi[r] as f64,
                )
            })
            .collect();
        fr.sort_by_key(|x| x.0);
        Ok(RepickData { alt, fr })
    }
}

/// Re-choose the apex RT and 1/K0 centre of the rank-0 candidates of the psms_extracted
/// `rows` on the raw events (`retrace.repick`): the picks `(candidate, apex RT, 1/K0 centre)`
/// in the order of `rows`, and with `dump` every scored peak `(candidate, order 0 = extract's
/// apex then the sidecar ranks, apex RT, score)`, which `MUMDIA_REPICK_DUMP` writes. Every
/// frame that isolates a candidate's precursor within [`REPICK_RT_HW_S`] of its peaks must be
/// decoded in `frames`.
#[allow(clippy::too_many_arguments)]
fn repick_rows(
    raw: &RawTdf,
    data: &RepickData,
    frag_tol: f64,
    rows: &[usize],
    ex_cid: &[u32],
    ex_rank: &[i32],
    apex_rt: &[f64],
    pmz: &[f64],
    (im_lo, im_hi): (&[f32], &[f32]),
    ms2_rt: &[f64],
    ms2_fi: &[usize],
    frames: &[Option<TofFrame>],
    scan_im: &[f64],
    dump: bool,
) -> (Vec<Pick>, Vec<ScoredPeak>) {
    let (alt, fr) = (&data.alt, &data.fr);
    let n_cand = pmz.len();
    let range = |v: &[(u32, i32, f64)], c: u32| {
        v.partition_point(|x| x.0 < c)..v.partition_point(|x| x.0 <= c)
    };
    let frange = |c: u32| fr.partition_point(|x| x.0 < c)..fr.partition_point(|x| x.0 <= c);
    let tol = frag_tol * 1e-6;
    // (pick: candidate, apex RT, 1/K0 centre; dumped rows: candidate, order, apex RT, score)
    type Scored = (Option<Pick>, Vec<ScoredPeak>);
    let scored: Vec<Scored> = rows
        .par_iter()
        .map(|&i| {
            let (c, ci) = (ex_cid[i], ex_cid[i] as usize);
            let mut dumped: Vec<ScoredPeak> = Vec::new();
            if ex_rank[i] != 0 || ci >= n_cand {
                return (None, dumped);
            }
            let (a, b) = (im_lo[ci] as f64, im_hi[ci] as f64);
            let fs = &fr[frange(c)];
            if !(a.is_finite() && b > a && pmz[ci].is_finite()) || fs.is_empty() {
                return (None, dumped);
            }
            let wins: Vec<(u32, u32)> = fs
                .iter()
                .map(|&(_, m, _)| tof_window(raw, m * (1.0 - tol), m * (1.0 + tol)))
                .collect();
            let pred: Vec<f64> = fs.iter().map(|x| x.2).collect();
            let band = scan_band(scan_im, a, b);
            let peaks = std::iter::once(apex_rt[ci]).chain(alt[range(alt, c)].iter().map(|x| x.2));
            // First maximum wins, so ties go to extract's apex, then to the lower rank.
            let mut best = (0.0f64, f64::NAN, f64::NAN);
            for (order, t0) in peaks.enumerate().filter(|(_, t)| t.is_finite()) {
                let lo = ms2_rt.partition_point(|&t| t < t0 - REPICK_RT_HW_S);
                let hi = ms2_rt.partition_point(|&t| t < t0 + REPICK_RT_HW_S);
                let fl: Vec<(&TofFrame, Vec<(usize, usize)>)> = (lo..hi)
                    .filter_map(|j| frames[ms2_fi[j]].as_ref())
                    .map(|f| (f, slot_scans(f.quad(), pmz[ci], band)))
                    .filter(|(_, s)| !s.is_empty())
                    .collect();
                let (s, cen) = joint_score(&fl, &wins, &pred, scan_im, a, b);
                if dump {
                    dumped.push((c, order as u32, t0, s));
                }
                if s > best.0 {
                    best = (s, t0, cen);
                }
            }
            ((best.0 > 0.0).then_some((c, best.1, best.2)), dumped)
        })
        .collect();
    let mut picks = Vec::new();
    let mut dumped = Vec::new();
    for (pick, rows) in scored {
        picks.extend(pick);
        dumped.extend(rows);
    }
    (picks, dumped)
}

/// The MS2 frames that isolate a window in `want` (inside the RT range), decoded into
/// `frames`, and every other decoded MS2 frame dropped: retrace holds the frames of the
/// windows its current candidates sit in, not the run. Returns the frames decoded.
#[allow(clippy::too_many_arguments)]
fn hold_window_frames(
    frames: &mut [Option<TofFrame>],
    want: &[bool],
    ms2_wins: &[Vec<u16>],
    ms2_fi: &[usize],
    ms2_rt: &[f64],
    (rt_min, rt_max): (f64, f64),
    raw: &RawTdf,
    path: &str,
) -> Result<usize> {
    let mut to_load = Vec::new();
    for (j, ws) in ms2_wins.iter().enumerate() {
        let fi = ms2_fi[j];
        let needed = (rt_min..=rt_max).contains(&ms2_rt[j]) && ws.iter().any(|&w| want[w as usize]);
        match (needed, frames[fi].is_some()) {
            (true, false) => to_load.push(fi),
            (false, true) => frames[fi] = None,
            _ => {}
        }
    }
    let loaded: Vec<(usize, TofFrame)> = to_load
        .par_iter()
        .map(|&fi| -> Result<(usize, TofFrame)> {
            let f = raw
                .reader
                .get(fi)
                .map_err(|e| anyhow!("{path}: frame {fi}: {e}"))?;
            Ok((fi, TofFrame::new(f)?))
        })
        .collect::<Result<_>>()?;
    let n = loaded.len();
    for (fi, f) in loaded {
        frames[fi] = Some(f);
    }
    Ok(n)
}

/// What [`patch_psms`] changed.
#[derive(Default)]
struct PatchStats {
    rows: u64,
    repicked: u64,
    moved: u64,
    refreshed_fragment_scalars: u64,
    refreshed_ms1_scalars: u64,
    without_record: u64,
}

/// The column `name` of `batch` as `T`.
fn col<'b, T: 'static>(batch: &'b RecordBatch, name: &str) -> Result<&'b T> {
    let i = batch
        .schema()
        .index_of(name)
        .map_err(|_| anyhow!("psms_extracted has no column '{name}'"))?;
    batch
        .column(i)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| anyhow!("psms_extracted column '{name}' has an unexpected type"))
}

/// Patch `psms_extracted` (`input`, written to `output`, which may be `input` itself),
/// once: `apex_rt` and `apex_im` of the re-picked
/// candidates (`picks`), the apex scalars of every rank-0 row whose candidate has an apex
/// record (`apex`), and under `imc` the column `imc_ref_w` (0.0 without a record). Every
/// other value, the columns' order and the rows' order are kept. A rank-0 row without an
/// apex record keeps extract's values (counted, and warned about by the caller).
///
/// The null rule of the MS1 scalars: a value refreshed from a rebuilt row is written, and
/// stays null only where extract had none and the raw trace has no signal there either.
fn patch_psms(
    input: &str,
    output: &str,
    picks: &[(u32, f64, f64)],
    apex: &[ApexRow],
    imc: bool,
) -> Result<(Written, PatchStats)> {
    let tf = TableFile::open(input)?;
    let rg = tf
        .row_group_rows()
        .first()
        .copied()
        .unwrap_or(1 << 16)
        .max(1);
    let mut reader = tf.batches(None, 1 << 16)?;
    let schema = reader.schema();
    let imc_at = schema.index_of("imc_ref_w").ok();
    let out_schema = match (imc, imc_at) {
        (true, None) => {
            let mut fields: Vec<Field> =
                schema.fields().iter().map(|f| f.as_ref().clone()).collect();
            fields.push(Field::new("imc_ref_w", DataType::Float64, false));
            Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
        }
        _ => schema.clone(),
    };
    let mut w = BatchWriter::with_options(
        output,
        out_schema.clone(),
        WriteOptions::new().row_group_rows(rg).content_hash(),
    )?;
    let pick_of = |c: u32| -> Option<(f64, f64)> {
        picks
            .binary_search_by_key(&c, |x| x.0)
            .ok()
            .map(|i| (picks[i].1, picks[i].2))
    };
    let apex_of = |c: u32| -> Option<&ApexRow> {
        apex.binary_search_by_key(&c, |x| x.cid)
            .ok()
            .map(|i| &apex[i])
    };
    let mut st = PatchStats::default();
    for batch in &mut reader {
        let batch = batch?;
        let n = batch.num_rows();
        let cid = col::<UInt32Array>(&batch, "candidate_id")?;
        let rank = col::<Int32Array>(&batch, "peak_rank")?;
        let rt = col::<Float64Array>(&batch, "apex_rt")?;
        let im = col::<Float64Array>(&batch, "apex_im")?;
        let ai = col::<Float32Array>(&batch, "apex_intensity")?;
        let nm = col::<Int32Array>(&batch, "n_matched_fragments")?;
        let ms1: Vec<&Float64Array> = ["ms1_mono", "ms1_iso1", "ms1_iso2"]
            .iter()
            .map(|c| col::<Float64Array>(&batch, c))
            .collect::<Result<_>>()?;
        let (mut new_rt, mut new_im, mut new_ai, mut new_nm) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        let mut new_ms1: [Vec<Option<f64>>; 3] = Default::default();
        let mut new_imc: Vec<f64> = Vec::with_capacity(if imc { n } else { 0 });
        for r in 0..n {
            let c = cid.value(r);
            let old_im = im.is_valid(r).then(|| im.value(r));
            let old_ms1: Vec<Option<f64>> = ms1
                .iter()
                .map(|a| a.is_valid(r).then(|| a.value(r)))
                .collect();
            let (mut t, mut m) = (rt.value(r), old_im);
            let (mut a, mut k) = (ai.value(r), nm.value(r));
            let mut v_ms1 = old_ms1.clone();
            let mut v_imc = 0.0;
            if rank.value(r) == 0 {
                if let Some((pt, pm)) = pick_of(c) {
                    st.repicked += 1;
                    st.moved += (pt != t) as u64;
                    (t, m) = (pt, Some(pm));
                }
                match apex_of(c) {
                    Some(rec) => {
                        if let Some((ra, rk)) = rec.frag {
                            (a, k) = (ra, rk);
                            st.refreshed_fragment_scalars += 1;
                        }
                        let mut any = false;
                        for (slot, (raw, old)) in v_ms1.iter_mut().zip(rec.ms1.iter().zip(&old_ms1))
                        {
                            if let Some(x) = raw {
                                any = true;
                                if old.is_some() || *x > 0.0 {
                                    *slot = Some(*x);
                                }
                            }
                        }
                        st.refreshed_ms1_scalars += any as u64;
                        v_imc = rec.imc;
                    }
                    None => st.without_record += 1,
                }
            }
            new_rt.push(t);
            new_im.push(m);
            new_ai.push(a);
            new_nm.push(k);
            for (dst, v) in new_ms1.iter_mut().zip(v_ms1) {
                dst.push(v);
            }
            if imc {
                new_imc.push(v_imc);
            }
        }
        let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
        let set = |cols: &mut Vec<ArrayRef>, name: &str, a: ArrayRef| -> Result<()> {
            let i = schema
                .index_of(name)
                .map_err(|_| anyhow!("psms_extracted has no column '{name}'"))?;
            cols[i] = a;
            Ok(())
        };
        set(&mut cols, "apex_rt", Arc::new(Float64Array::from(new_rt)))?;
        set(&mut cols, "apex_im", Arc::new(Float64Array::from(new_im)))?;
        set(
            &mut cols,
            "apex_intensity",
            Arc::new(Float32Array::from(new_ai)),
        )?;
        set(
            &mut cols,
            "n_matched_fragments",
            Arc::new(Int32Array::from(new_nm)),
        )?;
        for (name, v) in ["ms1_mono", "ms1_iso1", "ms1_iso2"].iter().zip(new_ms1) {
            set(&mut cols, name, Arc::new(Float64Array::from(v)))?;
        }
        if imc {
            let a: ArrayRef = Arc::new(Float64Array::from(new_imc));
            match imc_at {
                Some(i) => cols[i] = a,
                None => cols.push(a),
            }
        }
        w.write(&RecordBatch::try_new(out_schema.clone(), cols)?)?;
        st.rows += n as u64;
    }
    // Every handle on the input goes before the new file takes its place.
    drop(reader);
    drop(tf);
    Ok((w.close_hashed()?, st))
}

/// Rebuild the chromatograms of one run from its raw events, and patch its
/// `psms_extracted` (module docs). Returns both tables with their hashes.
pub fn run_hashed(p: RetraceParams) -> Result<RetraceWritten> {
    let t0 = Instant::now();
    mumdia_io::refuse_output_over_input(
        p.out,
        &[
            ("--chromatograms", p.chromatograms),
            ("--psms-extracted", p.psms_extracted),
            ("--out-psms", p.out_psms),
        ],
    )?;
    mumdia_io::refuse_output_over_input(p.out_psms, &[("--chromatograms", p.chromatograms)])?;
    let raw = RawTdf::open(p.raw)?;
    let (mass_off, frag_tol) = read_mass_cal(p.mass_cal, p.extract.frag_tol_ppm)?;

    // Frame index without decoding: RT, MS level, quad settings.
    let n_frames = raw.reader.len();
    let heads: Vec<Frame> = (0..n_frames)
        .into_par_iter()
        .map(|i| {
            raw.reader
                .get_frame_without_coordinates(i)
                .map_err(|e| anyhow!("{}: frame header {i}: {e}", p.raw))
        })
        .collect::<Result<_>>()?;
    let (mut ms2_rt, mut ms2_fi, mut ms1_rt, mut ms1_fi) = (vec![], vec![], vec![], vec![]);
    for (i, f) in heads.iter().enumerate() {
        match f.ms_level {
            MSLevel::MS1 => {
                ms1_rt.push(f.rt_in_seconds);
                ms1_fi.push(i);
            }
            MSLevel::MS2 => {
                ms2_rt.push(f.rt_in_seconds);
                ms2_fi.push(i);
            }
            MSLevel::Unknown => {}
        }
    }
    let ascending = |v: &[f64]| v.windows(2).all(|w| w[0] <= w[1]);
    if !(ascending(&ms1_rt) && ascending(&ms2_rt)) || ms1_rt.is_empty() || ms2_rt.is_empty() {
        bail!(
            "{}: frames are not in RT order, or MS1/MS2 frames are missing",
            p.raw
        );
    }
    // The isolation windows of the MS2 frames, for holding them window by window below: the
    // distinct (isolation m/z, width) of every slot as inclusive m/z bounds, and per MS2 frame
    // (in `ms2_fi` order) the windows its slots isolate.
    let mut win_of: std::collections::BTreeMap<(u64, u64), u16> = std::collections::BTreeMap::new();
    let mut win_bounds: Vec<(f64, f64)> = Vec::new();
    let mut ms2_wins: Vec<Vec<u16>> = Vec::with_capacity(ms2_fi.len());
    for &i in &ms2_fi {
        let q = &heads[i].quadrupole_settings;
        let mut ws: Vec<u16> = Vec::with_capacity(q.isolation_mz.len());
        for k in 0..q.isolation_mz.len() {
            let key = (q.isolation_mz[k].to_bits(), q.isolation_width[k].to_bits());
            let w = match win_of.get(&key) {
                Some(&w) => w,
                None => {
                    let half = q.isolation_width[k] / 2.0;
                    win_bounds.push((q.isolation_mz[k] - half, q.isolation_mz[k] + half));
                    let w = u16::try_from(win_bounds.len() - 1)
                        .map_err(|_| anyhow!("{}: more than 65,535 isolation windows", p.raw))?;
                    win_of.insert(key, w);
                    w
                }
            };
            ws.push(w);
        }
        ws.sort_unstable();
        ws.dedup();
        ms2_wins.push(ws);
    }
    drop(win_of);
    let n_scans = raw
        .reader
        .get(ms1_fi[0])
        .map_err(|e| anyhow!("{}: frame {}: {e}", p.raw, ms1_fi[0]))?
        .scan_offsets
        .len()
        .saturating_sub(1);
    let scan_im: Vec<f64> = (0..n_scans).map(|s| raw.im_of_scan(s as f64)).collect();
    if !scan_im.windows(2).all(|w| w[0] > w[1]) {
        bail!(
            "{}: 1/K0 is not strictly decreasing in the scan index",
            p.raw
        );
    }
    // The MS1 noise filter convert applied (`convert.tdf_ms1_denoise`), on the MS1 frames
    // held here as well, built from the run's diaPASEF windows.
    let ms1_denoise = (p.ms1_denoise && p.cfg.ms1).then(|| {
        let windows = raw.reader.get_dia_windows().unwrap_or_default();
        Ms1Denoise::new(&windows, n_scans, |m| raw.tof_of(m), |s| raw.im_of_scan(s))
    });

    // Per candidate: precursor m/z, the windows, and the band centre.
    let lib = TableFile::open(p.library_precursors)?;
    let lib_cid = lib.u32("candidate_id")?;
    let n_cand = lib_cid.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
    let pmz = by_candidate(&lib_cid, lib.f64("precursor_mz")?.into_iter(), n_cand);
    drop((lib, lib_cid));
    let mut win = read_run_windows(p.run_windows, n_cand)?;
    // Owned here and lent to each span in turn: the re-pick of a span's candidates moves
    // their apex and centre just before their rows are rebuilt.
    let mut centre_own: Vec<f64> = match &win.im_cal {
        Some(v) => v.iter().map(|x| x.unwrap_or(f64::NAN)).collect(),
        None => vec![f64::NAN; n_cand],
    };
    let ex = TableFile::open(p.psms_extracted)?;
    let (ex_cid, ex_rank, ex_im, ex_rt) = (
        ex.u32("candidate_id")?,
        ex.i32("peak_rank")?,
        ex.opt_f64("apex_im")?,
        ex.f64("apex_rt")?,
    );
    drop(ex);
    let mut apex_own = vec![f64::NAN; n_cand];
    for i in 0..ex_cid.len() {
        if ex_rank[i] != 0 {
            continue;
        }
        if let Some(slot) = apex_own.get_mut(ex_cid[i] as usize) {
            *slot = ex_rt[i];
        }
        if let Some(v) = ex_im[i].filter(|v| v.is_finite()) {
            if let Some(slot) = centre_own.get_mut(ex_cid[i] as usize) {
                *slot = v;
            }
        }
    }

    let ch = TableFile::open(p.chromatograms)?;
    let layout = Layout::of(&ch)?;
    let opt = Optional::of(&ch);
    // A skeleton table holds zeros where this stage rebuilds; one written for MS1 rows can
    // only be read with them rebuilt.
    let skeleton = Skeleton::recorded(p.chromatograms)?;
    if skeleton.is_some_and(|k| k.ms1) && !p.cfg.ms1 {
        bail!(
            "{}: extract left its MS1 isotope traces for retrace to rebuild ({} with ms1 \
             true), and retrace.ms1 is false here, so they would be published as zeros; run \
             retrace with retrace.ms1 = true or extract this run again",
            p.chromatograms,
            Skeleton::PARAM
        );
    }
    let lost_skeleton_rows = AtomicU64::new(0);
    if !opt.im {
        bail!(
            "{}: no per-point 1/K0 list; retrace needs the chromatograms of a diaPASEF run \
             extracted with the list on (run and run-experiment turn it on under retrace)",
            p.chromatograms
        );
    }
    if !opt.frag_mz {
        bail!(
            "{}: no frag_mz column; retrace sums raw events around each row's m/z",
            p.chromatograms
        );
    }

    // Decode, once, every frame inside the RT range of the candidates' windows (plus a
    // margin, for the nearest-MS1-frame lookup) and hold them: about the raw size of that
    // part of the run, whatever the number of trace points. A candidate without a finite
    // window needs the whole run.
    let (mut rt_min, mut rt_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &c in &ex_cid {
        match (win.rt_lo.get(c as usize), win.rt_hi.get(c as usize)) {
            (Some(&lo), Some(&hi)) if lo.is_finite() && hi.is_finite() => {
                rt_min = rt_min.min(lo);
                rt_max = rt_max.max(hi);
            }
            _ => {
                (rt_min, rt_max) = (f64::NEG_INFINITY, f64::INFINITY);
                break;
            }
        }
    }
    // Traces padded beyond the windows for quantification (`extract::TracePad`, recorded in
    // the centroid table's report) reach this much further, and their points are rebuilt
    // like any other.
    let trace_pad = chromatograms::recorded_trace_pad(p.chromatograms);
    let (rt_min, rt_max) = (
        rt_min - RT_MARGIN_S - trace_pad,
        rt_max + RT_MARGIN_S + trace_pad,
    );
    // The MS1 frames inside that range are decoded once and held: every MS1 trace reads them.
    // The MS2 frames are held window by window in the span loop below
    // (`hold_window_frames`): a span's candidates are consecutive in precursor m/z, so it
    // needs only the frames of the windows that isolate that stretch, and retrace holds a
    // window's frames instead of the run's (36-39 GB per HYE run when it held them all).
    let mut frames_own: Vec<Option<TofFrame>> = heads
        .par_iter()
        .enumerate()
        .map(|(i, h)| -> Result<Option<TofFrame>> {
            // MS1 frames feed only the MS1 traces.
            if h.ms_level != MSLevel::MS1
                || !p.cfg.ms1
                || !(rt_min..=rt_max).contains(&h.rt_in_seconds)
            {
                return Ok(None);
            }
            let mut f = raw
                .reader
                .get(i)
                .map_err(|e| anyhow!("{}: frame {i}: {e}", p.raw))?;
            if let Some(d) = ms1_denoise.as_ref() {
                d.apply_to_frame(&mut f);
            }
            Ok(Some(TofFrame::new(f)?))
        })
        .collect::<Result<_>>()?;
    drop(heads);
    let mut decoded = frames_own.iter().filter(|f| f.is_some()).count();
    info!(
        ms1_frames = decoded,
        rt_min, rt_max, "retrace: MS1 frames decoded"
    );
    let t_decoded = t0.elapsed().as_millis();

    // `retrace.repick`: each candidate's apex RT and 1/K0 centre are chosen again on the raw
    // events, among extract's apex and the top-K sidecar peaks, before its rows are rebuilt.
    // It runs per span, on the span's frames, over a contiguous range of candidates (rank-0
    // psms_extracted rows ordered by candidate), so every candidate is re-picked once.
    let repick_data = match (p.cfg.repick, &win.im_lo, &win.im_hi) {
        (false, _, _) => None,
        (true, Some(_), Some(_)) => Some(RepickData::load(&p, &ch, &mass_off)?),
        (true, _, _) => {
            warn!(
                run_windows = p.run_windows,
                "retrace: retrace.repick needs the 1/K0 windows of run_windows, and this run \
                 has none; extract's apex is kept"
            );
            None
        }
    };
    let (im_lo_own, im_hi_own) = (
        win.im_lo.take().unwrap_or_default(),
        win.im_hi.take().unwrap_or_default(),
    );
    let mut ex_order: Vec<usize> = (0..ex_cid.len()).filter(|&i| ex_rank[i] == 0).collect();
    ex_order.sort_by_key(|&i| ex_cid[i]);
    let dump_path = std::env::var("MUMDIA_REPICK_DUMP").ok();
    let mut picks: Vec<(u32, f64, f64)> = Vec::new();
    let mut dumped: Vec<(u32, u32, f64, f64)> = Vec::new();
    let mut t_repick = 0u128;
    let mut t_windows = 0u128;
    // The windows whose (inclusive) m/z bounds reach the precursors of candidates `lo..hi`,
    // widened by a hair so a precursor on a window edge never misses its window.
    let want_windows = |lo: usize, hi: usize| -> Vec<bool> {
        const EDGE: f64 = 1e-6;
        let mut want = vec![false; win_bounds.len()];
        let (mut a, mut b) = (f64::INFINITY, f64::NEG_INFINITY);
        for &m in &pmz[lo.min(pmz.len())..hi.min(pmz.len())] {
            if m.is_finite() {
                (a, b) = (a.min(m), b.max(m));
            }
        }
        if a <= b {
            for (w, &(l, u)) in win_bounds.iter().enumerate() {
                want[w] = l <= b + EDGE && u >= a - EDGE;
            }
        }
        want
    };
    // The windows' RT bounds stay only for padded traces, which are counted inside them.
    let pad_win: Option<(Vec<f64>, Vec<f64>)> =
        (trace_pad > 0.0).then(|| (win.rt_lo.clone(), win.rt_hi.clone()));
    drop((ex_im, ex_rt, win));

    let hw = p.cfg.im_half_width;
    let half1 = p.extract.ms1_scan_halfwidth;
    let prec_tol = p.extract.prec_tol_ppm * 1e-6;
    let ftol = frag_tol * 1e-6;
    let want_prof = p.cfg.apex_features;
    let (ms1_rt, ms1_fi, ms2_rt, ms2_fi) = (&ms1_rt, &ms1_fi, &ms2_rt, &ms2_fi);
    let (scan_im, raw, mass_off) = (&scan_im, &raw, &mass_off);
    let pmz = &pmz;
    let pad_win = &pad_win;

    // The input's actual row groups: each is read whole, so a v2 table is always read from
    // the start of a row group.
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut first = 0usize;
    for n in ch.row_group_rows() {
        if n > 0 {
            spans.push((first, n));
        }
        first += n;
    }
    let n_spans = spans.len();
    let rg_rows = chromatograms::row_group_rows();
    let want_readers = n_spans
        .min(1 + READER_EXTRAS)
        .min(rayon::current_num_threads())
        .max(1);
    let lease = DecoderLease::take_from(&READER_BUDGET, want_readers - 1);
    let n_readers = 1 + lease.granted();

    let (mut n_frag, mut n_ms1, mut unmatched, mut events) = (0u64, 0u64, 0u64, 0u64);
    let mut residuals: Vec<f64> = Vec::new();
    let mut t_sum = 0u128;
    let (mut pending, mut apex_done): (Option<Apex>, Vec<ApexRow>) = (None, Vec::new());
    let writer_path = p.out.to_string();
    // Three stages joined by bounded channels so read, sum and write overlap: the readers,
    // the parallel sum on this thread's pool, and a writer thread whose `TableWriter`
    // encodes whole row groups on the codec pool. Rows keep the input order. The writer
    // publishes only on the explicit end marker (`None`); when the channel closes without
    // it, after an error anywhere, the writer is dropped and its temporary file with it, so
    // no partial table is left.
    let (written, t_read, t_write) = std::thread::scope(|sc| -> Result<(Written, u128, u128)> {
        let ch = &ch;
        let spans = &spans;
        let (mut rx_ins, mut readers) = (Vec::new(), Vec::new());
        for j in 0..n_readers {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Result<SpanIn>>(1);
            rx_ins.push(rx);
            readers.push(sc.spawn(move || {
                let mut t_read = 0u128;
                for &(a, n) in spans.iter().skip(j).step_by(n_readers) {
                    let tp = Instant::now();
                    let span = SpanIn::read(ch, a, n);
                    t_read += tp.elapsed().as_millis();
                    if tx.send(span).is_err() {
                        break;
                    }
                }
                t_read
            }));
        }
        let (tx_out, rx_out) = std::sync::mpsc::sync_channel::<Option<Vec<Col>>>(2);
        let writer = sc.spawn(move || -> Result<Option<(Written, u128)>> {
            let mut w = chromatograms::writer(&writer_path, rg_rows, layout)
                .with_row_groups_in_flight(chromatograms::ROW_GROUPS_IN_FLIGHT)
                .with_content_hash();
            let mut t_write = 0u128;
            loop {
                match rx_out.recv() {
                    Ok(Some(cols)) => {
                        let tp = Instant::now();
                        w.write_cols(cols)?;
                        t_write += tp.elapsed().as_millis();
                    }
                    Ok(None) => {
                        let tp = Instant::now();
                        let done = w.close_hashed()?;
                        return Ok(Some((done, t_write + tp.elapsed().as_millis())));
                    }
                    // Aborted upstream: publish nothing.
                    Err(_) => return Ok(None),
                }
            }
        });
        let mut enc = chromatograms::Encoder::new(rg_rows);
        let mut sent_all = true;
        let mut next_lo = 0usize;
        for i in 0..n_spans {
            let span = rx_ins[i % n_readers]
                .recv()
                .map_err(|_| anyhow!("retrace: a reader stopped early"))??;
            // Every candidate up to this span's last: the frames of their windows held, then
            // their re-pick, so `apex_rt` and `centre` are final for every row of the span.
            let hi = span
                .cid
                .last()
                .map_or(next_lo, |&c| c as usize + 1)
                .max(next_lo);
            let tw = Instant::now();
            decoded += hold_window_frames(
                &mut frames_own,
                &want_windows(next_lo, hi),
                &ms2_wins,
                ms2_fi,
                ms2_rt,
                (rt_min, rt_max),
                raw,
                p.raw,
            )?;
            t_windows += tw.elapsed().as_millis();
            if let Some(data) = &repick_data {
                let tr = Instant::now();
                let a = ex_order.partition_point(|&r| (ex_cid[r] as usize) < next_lo);
                let b = ex_order.partition_point(|&r| (ex_cid[r] as usize) < hi);
                let (pk, rows) = repick_rows(
                    raw,
                    data,
                    frag_tol,
                    &ex_order[a..b],
                    &ex_cid,
                    &ex_rank,
                    &apex_own,
                    pmz,
                    (&im_lo_own, &im_hi_own),
                    ms2_rt,
                    ms2_fi,
                    &frames_own,
                    scan_im,
                    dump_path.is_some(),
                );
                for &(c, t, im) in &pk {
                    apex_own[c as usize] = t;
                    centre_own[c as usize] = im;
                }
                picks.extend(pk);
                dumped.extend(rows);
                t_repick += tr.elapsed().as_millis();
            }
            next_lo = hi;
            let (frames, centre, apex_rt) = (&frames_own, &centre_own, &apex_own);
            let tp = Instant::now();
            let n = span.cid.len();
            // Rows of one candidate are contiguous and share one grid, so the frames and the
            // quad-slot scan ranges of every grid point are planned once per candidate (per
            // distinct grid) and reused by each of its rows.
            let mut groups: Vec<(usize, usize)> = Vec::new();
            for r in 0..n {
                match groups.last_mut() {
                    Some(g) if span.cid[g.0] == span.cid[r] => g.1 = r + 1,
                    _ => groups.push((r, r + 1)),
                }
            }
            let plan = |pts: &[f32], frag: bool, c: usize| -> Vec<Point> {
                let band = scan_band(scan_im, centre[c] - hw, centre[c] + hw);
                pts.iter()
                    .map(|&tf| {
                        let t = tf as f64;
                        if frag {
                            let j = nearest_index(ms2_rt, t);
                            let f = ((ms2_rt[j] - t).abs() <= RT_MATCH_S)
                                .then(|| frames[ms2_fi[j]].as_ref())
                                .flatten();
                            match f {
                                Some(f) => (vec![f], slot_scans(f.quad(), pmz[c], band)),
                                None => (Vec::new(), Vec::new()),
                            }
                        } else {
                            (ms1_frames(t, half1, ms1_rt, ms1_fi, frames), vec![band])
                        }
                    })
                    .collect()
            };
            let span_ref = &span;
            let done: Vec<(Vec<RowOut>, Apex)> = groups
                .par_iter()
                .map(|&(g0, g1)| {
                    let s = span_ref;
                    let mut cache: [Option<(&[f32], Vec<Point>)>; 2] = [None, None];
                    let c0 = s.cid[g0] as usize;
                    let t_apex = apex_rt.get(c0).copied().unwrap_or(f64::NAN);
                    let mut apex = Apex::new(s.cid[g0]);
                    // MS2 frames within APEX_RT_HW_S of the apex and their quad-slot scans in
                    // the wider profile band, planned once per candidate.
                    let apex_frames: Vec<(&TofFrame, Vec<(usize, usize)>)> = if want_prof
                        && t_apex.is_finite()
                        && c0 < pmz.len()
                        && pmz[c0].is_finite()
                        && centre[c0].is_finite()
                    {
                        let band =
                            scan_band(scan_im, centre[c0] - APEX_IM_HW, centre[c0] + APEX_IM_HW);
                        let a = ms2_rt.partition_point(|&t| t < t_apex - APEX_RT_HW_S);
                        let b = ms2_rt.partition_point(|&t| t <= t_apex + APEX_RT_HW_S);
                        (a..b)
                            .filter_map(|j| frames[ms2_fi[j]].as_ref())
                            .map(|f| (f, slot_scans(f.quad(), pmz[c0], band)))
                            .filter(|(_, sc)| !sc.is_empty())
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let im_lo = centre.get(c0).copied().unwrap_or(f64::NAN) - APEX_IM_HW;
                    let out: Vec<RowOut> = (g0..g1)
                        .map(|r| {
                            let pts = &s.rt[r][..];
                            let c = s.cid[r] as usize;
                            let kind = if c < pmz.len() {
                                row_kind(
                                    !pts.is_empty(),
                                    pmz[c],
                                    centre[c],
                                    s.fmz[r],
                                    &s.name[r],
                                    s.pint[r],
                                    p.cfg.ms1,
                                )
                            } else {
                                Kind::Copy
                            };
                            if kind == Kind::Copy {
                                // In a skeleton table a fragment row with a grid and an
                                // all-zero trace is one extract left for this stage to
                                // rebuild (an observed fragment always has a non-zero
                                // point). Copying it would publish zeros as the trace.
                                if skeleton.is_some()
                                    && !pts.is_empty()
                                    && !s.name[r].starts_with("ms1_")
                                    && s.int[r].iter().all(|&x| x == 0.0)
                                {
                                    lost_skeleton_rows.fetch_add(1, Ordering::Relaxed);
                                }
                                return RowOut {
                                    kind,
                                    int: s.int[r].clone(),
                                    im: s.im[r].clone(),
                                    events: 0,
                                    misses: 0,
                                    ms1_residual_ppm: None,
                                };
                            }
                            let frag = kind == Kind::Frag;
                            let slot = &mut cache[frag as usize];
                            if slot.as_ref().is_none_or(|(g, _)| *g != pts) {
                                *slot = Some((pts, plan(pts, frag, c)));
                            }
                            let points = &slot.as_ref().expect("planned above").1;
                            // A fragment's m/z is calibrated as extract matched it; an MS1
                            // row's m/z already carries extract's MS1 offset, which is
                            // therefore never applied again here.
                            let tof = if frag {
                                let m = s.fmz[r] * mass_off.factor_at(s.fmz[r]);
                                tof_window(raw, m * (1.0 - ftol), m * (1.0 + ftol))
                            } else {
                                let m = s.fmz[r];
                                tof_window(raw, m * (1.0 - prec_tol), m * (1.0 + prec_tol))
                            };
                            let (mut ints, mut ims) =
                                (Vec::with_capacity(pts.len()), Vec::with_capacity(pts.len()));
                            let (mut ev, mut miss) = (0u64, 0u64);
                            let (mut sw, mut stof) = (0.0f64, 0.0f64);
                            let mut seen = Ms1Seen::default();
                            for (fs, scans) in points {
                                if fs.is_empty() {
                                    miss += 1;
                                }
                                let sum = if frag {
                                    point_sum(fs, scans, tof, scan_im)
                                } else {
                                    seen.point_sum(fs, scans, tof, scan_im)
                                };
                                ev += sum.n as u64;
                                let hit = sum.inten > 0.0;
                                ints.push(if hit { sum.inten as f32 } else { 0.0 });
                                if frag {
                                    // +0.0 wherever the trace is +0.0, as extract writes it,
                                    // so both layouts decode to the same list.
                                    ims.push(if hit {
                                        (sum.imw / sum.inten) as f32
                                    } else {
                                        0.0
                                    });
                                } else if hit {
                                    sw += sum.inten;
                                    stof += sum.tofw;
                                }
                            }
                            if !frag {
                                // The MS1 lists carry no per-point 1/K0 (+0.0 throughout).
                                ims = s.im[r].clone();
                            }
                            let ms1_residual_ppm = (!frag && s.name[r] == "ms1_mono" && sw > 0.0)
                                .then(|| (raw.mz_of(stof / sw) - s.fmz[r]) / s.fmz[r] * 1e6);
                            if frag && want_prof && !apex_frames.is_empty() {
                                let mut pr = [0.0f64; APEX_BINS];
                                for (f, scans) in &apex_frames {
                                    f.visit(scans, tof.0, tof.1, scan_im.len(), |sc, _, w| {
                                        let b = ((scan_im[sc] - im_lo) / APEX_IM_BIN) as isize;
                                        pr[b.clamp(0, APEX_BINS as isize - 1) as usize] += w;
                                    });
                                }
                                apex.prof.push((s.pint[r] as f64, pr));
                            }
                            RowOut {
                                kind,
                                int: ints,
                                im: ims,
                                events: ev,
                                misses: if frag { miss } else { 0 },
                                ms1_residual_ppm,
                            }
                        })
                        .collect();
                    // The scalars read the final rows of the candidate at its (re-picked)
                    // apex: the fragment ones when a fragment row was rebuilt, an MS1 one
                    // when its row was rebuilt.
                    if t_apex.is_finite() {
                        for (r, o) in (g0..g1).zip(&out) {
                            let v = nearest_point(&s.rt[r], &o.int, t_apex);
                            match s.name[r].as_str() {
                                "ms1_mono" if o.kind == Kind::Ms1 => apex.ms1[0] = Some(v),
                                "ms1_iso1" if o.kind == Kind::Ms1 => apex.ms1[1] = Some(v),
                                "ms1_iso2" if o.kind == Kind::Ms1 => apex.ms1[2] = Some(v),
                                nm if nm.starts_with("ms1") => {}
                                _ => {
                                    apex.rebuilt += (o.kind == Kind::Frag) as u32;
                                    apex.inten += v;
                                    // A fragment counts as matched by its points inside the
                                    // RT window, as extract counted it, also on padded traces.
                                    let int: &[f32] = match pad_win {
                                        Some((lo, hi))
                                            if o.int.len() == s.rt[r].len() && c0 < lo.len() =>
                                        {
                                            let sp = chromatograms::window_span(
                                                &s.rt[r], lo[c0], hi[c0],
                                            );
                                            &o.int[sp]
                                        }
                                        _ => &o.int,
                                    };
                                    apex.n_matched += int.iter().any(|&x| x > 0.0) as i32;
                                }
                            }
                        }
                    }
                    (out, apex)
                })
                .collect();
            let SpanIn {
                cid,
                name,
                fmz,
                obs,
                pint,
                rt,
                ..
            } = span;
            let mut rows = Rows::default();
            for (out, a) in done {
                for o in out {
                    match o.kind {
                        Kind::Frag => n_frag += 1,
                        Kind::Ms1 => n_ms1 += 1,
                        Kind::Copy => {}
                    }
                    events += o.events;
                    unmatched += o.misses;
                    if let Some(x) = o.ms1_residual_ppm {
                        residuals.push(x);
                    }
                    rows.intensity.push(o.int);
                    rows.im.push(o.im);
                }
                if apex_rt.get(a.cid as usize).is_some_and(|t| t.is_finite()) {
                    match pending.as_mut() {
                        Some(pa) if pa.cid == a.cid => pa.merge(a),
                        _ => {
                            if let Some(pa) = pending.replace(a) {
                                apex_done.push(pa.finish());
                            }
                        }
                    }
                }
            }
            let (ints, ims) = (
                std::mem::take(&mut rows.intensity),
                std::mem::take(&mut rows.im),
            );
            for (k, ((rt_k, int_k), im_k)) in rt.into_iter().zip(ints).zip(ims).enumerate() {
                match layout {
                    Layout::V1 => {
                        rows.rt.push(rt_k);
                        rows.intensity.push(int_k);
                        rows.im.push(im_k);
                    }
                    Layout::V2 => {
                        let e = enc.encode_im(cid[k], rt_k, int_k, Some(im_k))?;
                        rows.rt.push(e.rt);
                        rows.intensity.push(e.intensity);
                        rows.trace_offset.push(e.trace_offset);
                        rows.trace_len.push(e.trace_len);
                        rows.im.push(e.im);
                    }
                }
            }
            rows.cid = cid;
            rows.name = name;
            rows.frag_mz = fmz;
            rows.frag_obs_mz = obs;
            rows.predicted_intensity = pint;
            t_sum += tp.elapsed().as_millis();
            if tx_out.send(Some(rows.into_cols(layout, opt))).is_err() {
                sent_all = false;
                break;
            }
        }
        // Candidates after the last span carry no rows; they are re-picked for the
        // psms_extracted patch, on the frames of their windows.
        if let (Some(data), true) = (&repick_data, sent_all && next_lo < pmz.len()) {
            let tw = Instant::now();
            decoded += hold_window_frames(
                &mut frames_own,
                &want_windows(next_lo, pmz.len()),
                &ms2_wins,
                ms2_fi,
                ms2_rt,
                (rt_min, rt_max),
                raw,
                p.raw,
            )?;
            t_windows += tw.elapsed().as_millis();
            let tr = Instant::now();
            let a = ex_order.partition_point(|&r| (ex_cid[r] as usize) < next_lo);
            let (pk, rows) = repick_rows(
                raw,
                data,
                frag_tol,
                &ex_order[a..],
                &ex_cid,
                &ex_rank,
                &apex_own,
                pmz,
                (&im_lo_own, &im_hi_own),
                ms2_rt,
                ms2_fi,
                &frames_own,
                scan_im,
                dump_path.is_some(),
            );
            picks.extend(pk);
            dumped.extend(rows);
            t_repick += tr.elapsed().as_millis();
        }
        // A skeleton row this stage did not rebuild would be published as zeros: nothing is
        // published then (the writer drops its file without the end marker).
        let lost = lost_skeleton_rows.load(Ordering::Relaxed);
        if lost > 0 {
            sent_all = false;
        }
        // The chunk that fixes the schema of a table without rows (empty otherwise), then
        // the end marker.
        if sent_all {
            sent_all = tx_out
                .send(Some(Rows::default().into_cols(layout, opt)))
                .is_ok()
                && tx_out.send(None).is_ok();
        }
        drop(tx_out);
        drop(rx_ins);
        let mut t_read = 0u128;
        for r in readers {
            t_read += r
                .join()
                .map_err(|_| anyhow!("retrace: a reader panicked"))?;
        }
        let written = writer
            .join()
            .map_err(|_| anyhow!("retrace: the writer panicked"))??;
        match (sent_all, written) {
            (true, Some((written, t_write))) => Ok((written, t_read, t_write)),
            _ if lost > 0 => bail!(
                "{}: {lost} fragment rows that extract left as zeros for retrace to rebuild \
                 were not rebuilt, so the skeleton and this retrace disagree (other run \
                 windows, psms_extracted or retrace settings than the extract that wrote it); \
                 nothing published",
                p.chromatograms
            ),
            _ => bail!("retrace: the writer stopped before the end; nothing published"),
        }
    })?;
    drop(lease);
    drop(frames_own);
    picks.sort_by_key(|x| x.0);
    if repick_data.is_some() {
        info!(
            picked = picks.len(),
            "retrace: apex re-picked on the raw events"
        );
    }
    if let Some(path) = &dump_path {
        write_table(
            path,
            vec![
                Col::U32("candidate_id".into(), dumped.iter().map(|x| x.0).collect()),
                Col::U32("order".into(), dumped.iter().map(|x| x.1).collect()),
                Col::F64("apex_rt".into(), dumped.iter().map(|x| x.2).collect()),
                Col::F64("score".into(), dumped.iter().map(|x| x.3).collect()),
            ],
        )?;
        info!(rows = dumped.len(), path, "retrace: re-pick scores dumped");
    }
    drop((
        repick_data,
        ex_cid,
        ex_rank,
        ex_order,
        im_lo_own,
        im_hi_own,
        dumped,
    ));
    apex_done.extend(pending.map(Apex::finish));
    apex_done.sort_by_key(|a| a.cid);
    if unmatched > 0 {
        warn!(
            points = unmatched,
            "retrace: fragment grid points without a decoded frame at their RT; set to 0"
        );
    }

    // The raw MS1 mass residual against the m/z extract wrote for the MS1 rows, which
    // carries extract's MS1 offset: near 0 when the inherited offset fits the raw events.
    let residual_rows = residuals.len();
    let ms1_residual = median(residuals);
    if ms1_residual.is_finite() && ms1_residual.abs() > 0.5 * p.extract.prec_tol_ppm {
        warn!(
            median_ppm = ms1_residual,
            rows = residual_rows,
            prec_tol_ppm = p.extract.prec_tol_ppm,
            "retrace: the raw MS1 events sit off the MS1 rows' m/z by more than half the \
             precursor tolerance; the MS1 offset extract applied does not fit the raw events"
        );
    }

    // The chromatograms report: extract's record of the centroid table (its parameters,
    // the instrument and the reader, which a stage reading only this table resolves the
    // TIMS predicate from) with retrace's own block.
    let elapsed = t0.elapsed().as_millis();
    let centroid_report: Option<ArtifactReport> =
        mumdia_io::json::read_json(&format!("{}.report.json", p.chromatograms)).ok();
    let mut params = centroid_report
        .as_ref()
        .map(|r| r.params.clone())
        .filter(|v| v.is_object())
        .unwrap_or_else(|| json!({}));
    let retrace_params = json!({
        "trace_source": "raw_tdf",
        "raw": p.raw,
        "im_half_width": hw,
        "ms1": p.cfg.ms1,
        "repick": p.cfg.repick,
        "apex_features": p.cfg.apex_features,
        "frag_tol_ppm": frag_tol,
        "prec_tol_ppm": p.extract.prec_tol_ppm,
        "ms1_scan_halfwidth": half1,
        "config_hash": p.config_hash,
    });
    if let Some(obj) = params.as_object_mut() {
        obj.insert("reader".into(), json!(crate::instrument::READER_TIMSRUST));
        obj.insert("retrace".into(), retrace_params.clone());
    }
    let mut stats = std::collections::BTreeMap::new();
    let rows = written.rows;
    stats.insert("rows".to_string(), json!(rows));
    stats.insert("rebuilt_fragment_rows".to_string(), json!(n_frag));
    stats.insert("rebuilt_ms1_rows".to_string(), json!(n_ms1));
    stats.insert("copied_rows".to_string(), json!(rows - n_frag - n_ms1));
    // Frame decodes: the MS1 frames once, and each MS2 frame once per run of spans that
    // needs one of its windows (a frame's slots isolate windows apart in m/z).
    stats.insert("frames_decoded".to_string(), json!(decoded));
    stats.insert("readers".to_string(), json!(n_readers));
    stats.insert("ms_setup_and_decode".to_string(), json!(t_decoded));
    stats.insert("ms_window_frames".to_string(), json!(t_windows));
    stats.insert("ms_repick".to_string(), json!(t_repick));
    stats.insert("ms_read".to_string(), json!(t_read));
    stats.insert("ms_sum".to_string(), json!(t_sum));
    stats.insert("ms_write".to_string(), json!(t_write));
    stats.insert("raw_events_summed".to_string(), json!(events));
    stats.insert("unmatched_fragment_points".to_string(), json!(unmatched));
    stats.insert("ms1_mass_residual_rows".to_string(), json!(residual_rows));
    if ms1_residual.is_finite() {
        stats.insert(
            "ms1_mass_residual_ppm_median".to_string(),
            json!(ms1_residual),
        );
    }
    let schema = mumdia_core::schema::artifact::chromatograms(layout.version(), opt.im);
    let report = ArtifactReport {
        logical_name: schema.0.to_string(),
        schema_name: schema.0.to_string(),
        schema_version: schema.1,
        stage: "retrace".to_string(),
        rows,
        content_hash: written.content_hash.clone(),
        params,
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    };
    report.write_for(p.out)?;

    // The one patch of psms_extracted, with its provenance in the table's report.
    let (psms_written, st) = patch_psms(
        p.psms_extracted,
        p.out_psms,
        &picks,
        &apex_done,
        p.cfg.apex_features,
    )?;
    if st.without_record > 0 {
        warn!(
            rows = st.without_record,
            psms = p.psms_extracted,
            "retrace: psms_extracted rows without a retrace apex record keep extract's values"
        );
    }
    let report_path = format!("{}.report.json", p.psms_extracted);
    let mut psms_report: ArtifactReport = mumdia_io::json::read_json(&report_path)
        .with_context(|| format!("retrace: reading {report_path} to record the patch"))?;
    psms_report.rows = psms_written.rows;
    psms_report.content_hash = psms_written.content_hash.clone();
    let mut patched = vec!["apex_intensity", "n_matched_fragments"];
    if p.cfg.ms1 {
        patched.extend(["ms1_mono", "ms1_iso1", "ms1_iso2"]);
    }
    if p.cfg.repick {
        patched.extend(["apex_rt", "apex_im"]);
    }
    if p.cfg.apex_features {
        patched.push("imc_ref_w");
    }
    let provenance = json!({
        "stage": "retrace",
        "chromatograms": p.out,
        "patched_columns": patched,
        "rows_repicked": st.repicked,
        "rows_moved": st.moved,
        "rows_fragment_scalars_refreshed": st.refreshed_fragment_scalars,
        "rows_ms1_scalars_refreshed": st.refreshed_ms1_scalars,
        "rows_without_apex_record": st.without_record,
        "params": retrace_params,
    });
    match psms_report.params.as_object_mut() {
        Some(obj) => {
            obj.insert("retrace".into(), provenance);
        }
        None => psms_report.params = json!({ "retrace": provenance }),
    }
    psms_report.write_for(p.out_psms)?;
    info!(
        rows,
        fragment_rows = n_frag,
        ms1_rows = n_ms1,
        frames = decoded,
        repicked = st.repicked,
        moved = st.moved,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "retrace: done"
    );
    Ok(RetraceWritten {
        chromatograms: written,
        psms_extracted: psms_report.written(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::convert::tims_frames::tests::frame;
    use timsrust::QuadrupoleSettings;

    #[test]
    fn apex_imc_scores_profile_agreement_and_survives_a_span_split() {
        let bump = |c: usize| {
            let mut p = [0.0; APEX_BINS];
            for (k, v) in p.iter_mut().enumerate() {
                *v = (-((k as f64 - c as f64).powi(2)) / 8.0).exp();
            }
            p
        };
        let rec = |prof| Apex {
            cid: 7,
            rebuilt: 1,
            inten: 1.0,
            n_matched: 1,
            ms1: [None, Some(2.0), None],
            prof,
        };
        // Three fragments on one mobility peak agree fully; an empty one is ignored.
        let same = rec(vec![
            (1.0, bump(20)),
            (2.0, bump(20)),
            (0.5, bump(20)),
            (3.0, [0.0; APEX_BINS]),
        ]);
        assert!((same.imc() - 1.0).abs() < 1e-12);
        // One fragment on another ion's peak pulls the score down.
        let off = rec(vec![(1.0, bump(20)), (1.0, bump(20)), (1.0, bump(5))]);
        assert!(off.imc() < 0.7);
        assert_eq!(rec(vec![]).imc(), 0.0);
        // A candidate split over two spans scores as the whole.
        let mut a = rec(vec![(1.0, bump(20)), (1.0, bump(20))]);
        let mut b = rec(vec![(1.0, bump(5))]);
        b.ms1 = [Some(3.0), Some(9.0), None];
        a.merge(b);
        assert_eq!(a.imc(), off.imc());
        assert_eq!((a.inten, a.n_matched, a.rebuilt), (2.0, 2, 2));
        assert_eq!(a.ms1, [Some(3.0), Some(2.0), None]);
        // Without a rebuilt fragment row the fragment scalars are not refreshed.
        let mut none = rec(vec![]);
        none.rebuilt = 0;
        assert_eq!(none.finish().frag, None);
    }

    /// `imc_ref_w` through the shared kernel is the IM branch's formula bit for bit: the
    /// Pearson of each profile against the weighted total minus its own weighted profile.
    #[test]
    fn imc_through_the_shared_kernel_is_the_branch_formula() {
        let prof: Vec<(f64, [f64; APEX_BINS])> = (0..5)
            .map(|f| {
                let mut p = [0.0; APEX_BINS];
                for (k, v) in p.iter_mut().enumerate() {
                    *v = ((k * (f + 3)) % 11) as f64 * 1.7 + f as f64;
                }
                (0.3 + f as f64, p)
            })
            .collect();
        let mut tot = [0.0f64; APEX_BINS];
        for (w, p) in &prof {
            for (t, v) in tot.iter_mut().zip(p) {
                *t += w * v;
            }
        }
        let (mut num, mut den) = (0.0, 0.0);
        for (w, p) in &prof {
            let other: Vec<f64> = tot.iter().zip(p).map(|(t, v)| t - w * v).collect();
            num += w * crate::stats::pearson(p, &other);
            den += w;
        }
        let a = Apex {
            prof,
            ..Apex::new(1)
        };
        assert_eq!(a.imc().to_bits(), (num / den).to_bits());
    }

    #[test]
    fn joint_score_finds_the_band_that_matches_the_library() {
        // 60 scans at 1/K0 1.30 - 0.002 s (strictly decreasing); window [1.18, 1.30].
        let im: Vec<f64> = (0..60).map(|s| 1.30 - 0.002 * s as f64).collect();
        // Two fragments (TOF 100 and 200), predicted 4:1. Scans 10-12 (1/K0 ~1.278) carry
        // them 4:1; scans 40-42 (1/K0 ~1.218) carry a brighter 1:4 ion.
        let mut f = Frame::default();
        let mut off = vec![0usize];
        for s in 0..60u32 {
            let (a, b) = match s {
                10..=12 => (400, 100),
                40..=42 => (200, 800),
                _ => (0, 0),
            };
            for (t, i) in [(100u32, a), (200, b)] {
                if i > 0 {
                    f.tof_indices.push(t);
                    f.intensities.push(i);
                }
            }
            off.push(f.tof_indices.len());
        }
        f.scan_offsets = off;
        f.quadrupole_settings = Arc::new(QuadrupoleSettings::default());
        let fr = TofFrame::new(f).unwrap();
        let fl = vec![(&fr, vec![(0usize, 60usize)])];
        let (s, c) = joint_score(&fl, &[(100, 100), (200, 200)], &[4.0, 1.0], &im, 1.18, 1.30);
        assert!(s > 0.0);
        assert!((c - 1.278).abs() < 0.004, "centre {c}");
        // No signal: no score, no centre.
        let (s0, c0) = joint_score(&fl, &[(900, 900)], &[1.0], &im, 1.18, 1.30);
        assert_eq!(s0, 0.0);
        assert!(c0.is_nan());
    }

    #[test]
    fn nearest_point_takes_the_first_of_a_tie_and_zero_when_empty() {
        assert_eq!(nearest_point(&[1.0, 2.0, 3.0], &[5.0, 6.0, 7.0], 2.5), 6.0);
        assert_eq!(nearest_point(&[], &[], 2.5), 0.0);
    }

    /// The MS1 frames of a grid point follow extract's MS1 read: the nearest frame and its
    /// `ms1_scan_halfwidth` neighbours, the largest sum kept. A half-width of 0 is the
    /// nearest frame alone, the IM branch's rule.
    #[test]
    fn ms1_frames_follow_extracts_selection_and_keep_the_largest_sum() {
        // Three MS1 frames at 10, 11 and 12 s; frame k scales every intensity by 1 + 2k,
        // except the middle one, which is the faintest.
        let scale = [3u32, 1, 5];
        let frames: Vec<Option<TofFrame>> = scale
            .iter()
            .map(|&m| {
                let mut f = frame();
                for x in f.intensities.iter_mut() {
                    *x *= m;
                }
                Some(TofFrame::new(f).unwrap())
            })
            .collect();
        let (rt, fi) = (vec![10.0, 11.0, 12.0], vec![0usize, 1, 2]);
        let im: Vec<f64> = (0..8).map(|s| 1.5 - 0.1 * s as f64).collect();
        let sum_at = |t: f64, half: usize| {
            let fs = ms1_frames(t, half, &rt, &fi, &frames);
            (fs.len(), point_sum(&fs, &[(0, 8)], (200, 200), &im).inten)
        };
        // The scans' intensities 1..=8 sum to 36 per frame before scaling.
        assert_eq!(sum_at(11.1, 0), (1, 36.0));
        assert_eq!(sum_at(11.1, 1), (3, 180.0));
        assert_eq!(sum_at(9.0, 1), (2, 108.0));
        // A frame that was not decoded is skipped, not summed as zero.
        let partial = vec![
            None,
            frames[1].as_ref().map(|_| TofFrame::new(frame()).unwrap()),
            None,
        ];
        assert_eq!(ms1_frames(11.0, 1, &rt, &fi, &partial).len(), 1);
        assert!(ms1_frames(10.0, 0, &rt, &fi, &partial).is_empty());
    }

    /// Extract zeroes exactly the rows retrace rebuilds: the skeleton rule is `row_kind`,
    /// for fragment and MS1 rows, with and without `retrace.ms1`, at every non-finite input.
    #[test]
    fn a_skeleton_zeroes_exactly_the_rows_retrace_rebuilds() {
        let nan = f64::NAN;
        for ms1 in [false, true] {
            let k = Skeleton { ms1 };
            for has in [false, true] {
                for pmz in [500.0, nan] {
                    for centre in [1.0, nan] {
                        for mz in [300.0, nan] {
                            for (name, pi) in [
                                ("y5", 0.4f32),
                                ("b3", 0.0),
                                ("ms1_mono", 0.0),
                                ("ms1_iso2", 0.0),
                            ] {
                                let kind = row_kind(has, pmz, centre, mz, name, pi, ms1);
                                assert_eq!(
                                    k.zeroes(has, pmz, centre, mz, name, pi),
                                    kind != Kind::Copy,
                                    "{name} ms1={ms1} has={has} pmz={pmz} centre={centre} mz={mz}"
                                );
                            }
                        }
                    }
                }
            }
        }
        // The rule itself: a fragment needs a prediction, an MS1 row needs retrace.ms1.
        assert!(row_kind(true, 500.0, 1.0, 300.0, "y5", 0.4, false) == Kind::Frag);
        assert!(row_kind(true, 500.0, 1.0, 300.0, "y5", 0.0, true) == Kind::Copy);
        assert!(row_kind(true, 500.0, 1.0, 500.0, "ms1_mono", 0.0, true) == Kind::Ms1);
        assert!(row_kind(true, 500.0, 1.0, 500.0, "ms1_mono", 0.0, false) == Kind::Copy);
        assert!(row_kind(true, 500.0, nan, 300.0, "y5", 0.4, true) == Kind::Copy);
    }

    /// The marker a skeleton table carries in its report, absent on an ordinary table and
    /// refused when malformed.
    #[test]
    fn a_skeleton_is_read_from_the_table_report() {
        let dir = std::env::temp_dir().join(format!("mumdia_skeleton_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chrom.parquet").to_str().unwrap().to_string();
        let report = format!("{path}.report.json");
        assert_eq!(Skeleton::recorded(&path).unwrap(), None);
        std::fs::write(&report, r#"{"params":{"trace_pad_s":2.0}}"#).unwrap();
        assert_eq!(Skeleton::recorded(&path).unwrap(), None);
        std::fs::write(
            &report,
            r#"{"params":{"retrace_skeleton":{"ms1":true,"rows":7}}}"#,
        )
        .unwrap();
        assert_eq!(
            Skeleton::recorded(&path).unwrap(),
            Some(Skeleton { ms1: true })
        );
        std::fs::write(&report, r#"{"params":{"retrace_skeleton":{"rows":7}}}"#).unwrap();
        assert!(Skeleton::recorded(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An MS1 row walked with the frame sums kept gives every point the same sum, bit for
    /// bit, as summing each point's frames afresh: at every half-width, with ties (equal
    /// frames), a frame that was not decoded, and points that share their nearest frame.
    #[test]
    fn ms1_seen_equals_summing_every_point_afresh() {
        let scale = [3u32, 1, 5, 5, 2, 7, 1, 4, 4, 6];
        let mut frames: Vec<Option<TofFrame>> = scale
            .iter()
            .map(|&m| {
                let mut f = frame();
                for x in f.intensities.iter_mut() {
                    *x *= m;
                }
                Some(TofFrame::new(f).unwrap())
            })
            .collect();
        frames[6] = None;
        let rt: Vec<f64> = (0..scale.len()).map(|k| 10.0 + k as f64).collect();
        let fi: Vec<usize> = (0..scale.len()).collect();
        let im: Vec<f64> = (0..8).map(|s| 1.5 - 0.1 * s as f64).collect();
        // Grid points every 0.6 s: some share their nearest MS1 frame.
        let grid: Vec<f64> = (0..15).map(|p| 9.7 + 0.6 * p as f64).collect();
        for half in 0..4 {
            for (scans, tof) in [(vec![(0, 8)], (200, 200)), (vec![(2, 5)], (100, 300))] {
                let mut seen = Ms1Seen::default();
                for &t in &grid {
                    let fs = ms1_frames(t, half, &rt, &fi, &frames);
                    let want = point_sum(&fs, &scans, tof, &im);
                    let got = seen.point_sum(&fs, &scans, tof, &im);
                    assert_eq!(got, want, "half {half} t {t} scans {scans:?}");
                }
            }
        }
    }

    /// The one patch of psms_extracted: re-picked apexes, refreshed scalars under the null
    /// rule, extract's values kept without a record, `imc_ref_w` appended (and replaced, not
    /// appended again, on a second patch), every other column and the row order kept.
    #[test]
    fn the_psms_patch_writes_each_rule_once() {
        let dir = std::env::temp_dir().join(format!("mumdia_retrace_patch_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("psms.parquet").to_str().unwrap().to_string();
        let out = dir.join("psms_out.parquet").to_str().unwrap().to_string();
        write_table(
            &path,
            vec![
                Col::U32("candidate_id".into(), vec![3, 5, 9, 9]),
                Col::I32("peak_rank".into(), vec![0, 0, 0, 1]),
                Col::F64("apex_rt".into(), vec![10.0, 20.0, 30.0, 31.0]),
                Col::OptF64(
                    "apex_im".into(),
                    vec![Some(0.9), None, Some(1.1), Some(1.2)],
                ),
                Col::F32("apex_intensity".into(), vec![1.0, 2.0, 3.0, 4.0]),
                Col::I32("n_matched_fragments".into(), vec![3, 4, 5, 6]),
                Col::Str(
                    "label".into(),
                    vec![
                        "target".into(),
                        "decoy".into(),
                        "target".into(),
                        "target".into(),
                    ],
                ),
                Col::OptF64("ms1_mono".into(), vec![Some(7.0), None, None, Some(1.0)]),
                Col::OptF64("ms1_iso1".into(), vec![Some(8.0), None, Some(2.0), None]),
                Col::OptF64("ms1_iso2".into(), vec![None, None, None, None]),
            ],
        )
        .unwrap();
        let picks = vec![(5u32, 22.5, 0.95)];
        let apex = vec![
            ApexRow {
                cid: 3,
                frag: Some((11.0, 7)),
                // A refreshed zero stays null where extract had none, and becomes 0.0 where
                // it had a value.
                ms1: [Some(0.0), Some(4.0), Some(0.0)],
                imc: 0.5,
            },
            ApexRow {
                cid: 5,
                frag: None,
                ms1: [None, None, None],
                imc: -0.25,
            },
        ];
        let (w, st) = patch_psms(&path, &out, &picks, &apex, true).unwrap();
        assert_eq!(w.rows, 4);
        assert_eq!(
            (st.repicked, st.moved, st.without_record),
            (1, 1, 1),
            "candidate 9 has no record"
        );
        assert_eq!(
            (st.refreshed_fragment_scalars, st.refreshed_ms1_scalars),
            (1, 1)
        );
        let t = TableFile::open(&out).unwrap();
        let names = t.column_names();
        assert_eq!(names.last().map(String::as_str), Some("imc_ref_w"));
        assert_eq!(
            names[..names.len() - 1],
            TableFile::open(&path).unwrap().column_names()[..]
        );
        assert_eq!(t.f64("apex_rt").unwrap(), vec![10.0, 22.5, 30.0, 31.0]);
        assert_eq!(
            t.opt_f64("apex_im").unwrap(),
            vec![Some(0.9), Some(0.95), Some(1.1), Some(1.2)]
        );
        assert_eq!(t.f32("apex_intensity").unwrap(), vec![11.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.i32("n_matched_fragments").unwrap(), vec![7, 4, 5, 6]);
        assert_eq!(
            t.opt_f64("ms1_mono").unwrap(),
            vec![Some(0.0), None, None, Some(1.0)]
        );
        assert_eq!(
            t.opt_f64("ms1_iso1").unwrap(),
            vec![Some(4.0), None, Some(2.0), None]
        );
        assert_eq!(t.opt_f64("ms1_iso2").unwrap(), vec![None, None, None, None]);
        assert_eq!(t.f64("imc_ref_w").unwrap(), vec![0.5, -0.25, 0.0, 0.0]);
        assert_eq!(
            t.str("label").unwrap(),
            vec!["target", "decoy", "target", "target"]
        );
        // A second patch, in place, replaces the column instead of adding another.
        drop(t);
        let (w2, _) = patch_psms(&out, &out, &[], &apex, true).unwrap();
        assert_eq!(w2.rows, 4);
        let t2 = TableFile::open(&out).unwrap();
        assert_eq!(
            t2.column_names()
                .iter()
                .filter(|c| *c == "imc_ref_w")
                .count(),
            1
        );
        // Without `retrace.apex_features` no column is added.
        let plain = dir.join("psms_plain.parquet").to_str().unwrap().to_string();
        patch_psms(&path, &plain, &picks, &apex, false).unwrap();
        assert!(!TableFile::open(&plain).unwrap().has_column("imc_ref_w"));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `retrace.enabled` on an mzML input is refused before any stage runs, and the
    /// default configuration passes any input.
    #[test]
    fn retrace_on_an_mzml_input_is_refused_up_front() {
        let off = Config::default();
        refuse_inputs_without_raw_events(&off, &["run.mzML"]).unwrap();
        let on = Config::from_json(r#"{"retrace":{"enabled":true}}"#).unwrap();
        let err = refuse_inputs_without_raw_events(&on, &["a.d", "run.mzML"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("a.d") || err.contains("run.mzML"), "{err}");
        assert!(err.contains("retrace.enabled = false"), "{err}");
        // The chain's own check names the spectra artifact the same way.
        assert!(raw_input("no/such/spectra_ms2.parquet").is_err());
    }

    #[test]
    fn the_median_is_order_free() {
        assert!(median(vec![]).is_nan());
        assert_eq!(median(vec![3.0, -1.0, 2.0]), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), 2.5);
    }
}
