//! `mumdia mobilogram`: the 1/K0 profile of a candidate's ions, read from the raw timsTOF
//! events of its run (diaPASEF only; docs/35_diapasef.md section 17).
//!
//! No artifact holds a mobility profile: convert keeps one intensity-weighted 1/K0 per
//! centroid, and retrace sums the raw events of a trace point into one value. This command
//! goes back to the `.d` for a few candidates at a time. For each candidate it takes the MS2
//! frames whose quad slots isolate the precursor m/z (the frame nearest the apex RT and
//! `frames` such frames either side) and the MS1 frames nearest the same RT, and bins the
//! raw events of every ion in 1/K0:
//!
//! - the fragments are the candidate's chromatogram rows with a positive prediction, at
//!   their library m/z corrected by the run's mass offset and within the learned fragment
//!   tolerance, the window retrace sums (`masscal::read_mass_cal`);
//! - the MS1 isotopes (`ms1_mono`, `ms1_iso1`, `ms1_iso2`) are the chromatogram's MS1 rows,
//!   whose m/z already carries extract's MS1 offset, within `extract.prec_tol_ppm`;
//! - an MS2 event counts when its scan lies in a slot that holds the precursor, with
//!   overlapping slots merged so no event is counted twice (`tims_frames::slot_scans`); an
//!   MS1 event when its 1/K0 lies in the bin range;
//! - the bin range spans the scans of those slots, so the MS1 profile is cut to the
//!   mobility range the fragments were measured in.
//!
//! The scan -> 1/K0 table and the m/z -> TOF window are convert's
//! (`tims_frames::scan_mobilities`, `tims_frames::tof_window`), so the profile sits on the
//! axes the centroids and retrace use. The MS1 events are not passed through convert's MS1
//! noise filter. Read-only: nothing in the run directory is written.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use mumdia_core::config::Config;
use mumdia_io::table::{write_table, Col, TableFile};
use rayon::prelude::*;
use serde::Serialize;
use timsrust::{Frame, MSLevel};
use tracing::{info, warn};

use crate::chromatograms;
use crate::masscal::read_mass_cal;
use crate::stages::convert::tims_frames::{
    bin_events, scan_band, scan_mobilities, slot_scans, tof_window, TofFrame,
};
use crate::stages::convert::RawTdf;
use crate::stages::extract::nearest_index;

/// The `format` of the JSON output, and its version.
pub const FORMAT: &str = "mumdia-mobilogram";
pub const FORMAT_VERSION: u32 = 1;
/// Default bin width in 1/K0 (V s cm^-2): about two TIMS scans.
pub const DEFAULT_BIN: f64 = 0.002;
/// Default number of frames either side of the apex frame.
pub const DEFAULT_FRAMES: usize = 2;
/// Upper bound on `frames`, so a typo does not decode a gradient.
pub const MAX_FRAMES: usize = 50;
/// Narrowest bin: well below one scan step, wider than f64 noise on the bin edges.
const MIN_BIN: f64 = 1e-5;

/// What `mumdia mobilogram` reads and how. Every path is optional when `run_dir` is given
/// ([`resolve`] fills them from the run's layout); without `run_dir`, `raw`,
/// `psms_extracted` and `chromatograms` are required.
pub struct MobilogramParams<'a> {
    pub run_dir: Option<&'a str>,
    /// The diaPASEF `.d`; default: the one the run's spectra report names.
    pub raw: Option<&'a str>,
    pub psms_extracted: Option<&'a str>,
    pub chromatograms: Option<&'a str>,
    pub run_windows: Option<&'a str>,
    pub mass_cal: Option<&'a str>,
    /// `--config`; `None`: the configuration the run recorded, else the defaults.
    pub config: Option<Config>,
    pub candidates: &'a [u32],
    /// The `psms_extracted` row of each candidate (`peak_rank`).
    pub peak_rank: i32,
    /// The RT the frames are centred on; default: each candidate's `apex_rt`.
    pub rt: Option<f64>,
    pub frames: usize,
    pub bin: f64,
}

/// The inputs [`resolve`] settled on.
#[derive(Debug, Clone, Serialize)]
pub struct Inputs {
    pub raw: String,
    pub psms_extracted: String,
    pub chromatograms: String,
    pub run_windows: Option<String>,
    pub mass_cal: Option<String>,
    /// Where the configuration came from: `--config`, a manifest path, or `defaults`.
    pub config: String,
}

/// One frame a profile read.
#[derive(Debug, Clone, Serialize)]
pub struct FrameRef {
    /// The frame's position in the `.d` frame reader (0-based).
    pub index: usize,
    /// The frame's `Frames.Id`, the `N` of convert's spectrum ids (`frame=N slot=k`).
    pub frame_id: usize,
    pub rt: f64,
    /// Position relative to the centre frame, in frames of the same kind.
    pub offset: i32,
}

/// The 1/K0 profile of one ion.
#[derive(Debug, Clone, Serialize)]
pub struct IonProfile {
    /// The chromatogram row's `frag_name` (`ms1_mono`, `ms1_iso1`, `ms1_iso2` for MS1).
    pub name: String,
    /// `fragment` or `ms1`.
    pub kind: &'static str,
    /// The chromatogram row's m/z (`frag_mz`).
    pub mz: f64,
    /// The m/z the events were taken around: `mz` times the mass-offset factor for a
    /// fragment, `mz` for an MS1 row.
    pub query_mz: f64,
    pub tol_ppm: f64,
    pub predicted_intensity: f32,
    /// Raw intensity per bin, summed over the frames.
    pub intensity: Vec<f64>,
    /// Raw intensity per bin of each frame, in the order of `ms2_frames` (fragments) or
    /// `ms1_frames` (MS1).
    pub per_frame: Vec<Vec<f64>>,
}

/// The profiles of one candidate.
#[derive(Debug, Clone, Serialize)]
pub struct CandidateMobilogram {
    pub candidate_id: u32,
    pub peak_rank: i32,
    pub precursor_mz: f64,
    pub apex_rt: f64,
    pub apex_im: Option<f64>,
    pub ms1_apex_im: Option<f64>,
    /// The RT the frames are centred on.
    pub centre_rt: f64,
    /// `run_windows` `im_lo`, `im_hi` and `im_pred_cal`, when the run has them.
    pub im_window: Option<[f64; 2]>,
    pub im_pred_cal: Option<f64>,
    /// Lower edge of bin 0 and the bin width: bin `k` is `[im_lo + k bin, im_lo + (k+1) bin)`.
    pub im_lo: f64,
    pub bin: f64,
    /// Bin centres, ascending.
    pub im: Vec<f64>,
    /// The 1/K0 range of the scans of the slots that hold the precursor in `ms2_frames`.
    pub slot_im: [f64; 2],
    pub ms2_frames: Vec<FrameRef>,
    pub ms1_frames: Vec<FrameRef>,
    pub ions: Vec<IonProfile>,
}

/// A candidate that could not be profiled, and why.
#[derive(Debug, Clone, Serialize)]
pub struct CandidateError {
    pub candidate_id: u32,
    pub error: String,
}

/// The whole result, as `--json` prints it.
#[derive(Debug, Clone, Serialize)]
pub struct Mobilograms {
    pub format: &'static str,
    pub format_version: u32,
    pub mumdia_version: &'static str,
    pub inputs: Inputs,
    pub bin: f64,
    pub frames: usize,
    pub peak_rank: i32,
    pub frag_tol_ppm: f64,
    /// The scalar fragment mass offset (ppm) of the mass calibration, and the points of its
    /// m/z grid (0: the scalar is used).
    pub frag_ppm_offset: f64,
    pub frag_offset_grid_points: usize,
    pub prec_tol_ppm: f64,
    pub im_calibration: &'static str,
    pub ms1_denoise: bool,
    pub candidates: Vec<CandidateMobilogram>,
    pub errors: Vec<CandidateError>,
}

/// `path` when it exists.
fn existing(path: PathBuf) -> Option<String> {
    path.exists().then(|| path.display().to_string())
}

/// The spectra artifact of the run at `dir`: `<dir>/spectra/spectra_ms2.parquet`, or for a
/// second-pass run directory (`<experiment>/second_pass/<run>`, which holds no spectra) the
/// first pass's `<experiment>/<run>/spectra/spectra_ms2.parquet`.
fn spectra_of(dir: &Path) -> Option<String> {
    let own = dir.join("spectra").join("spectra_ms2.parquet");
    if own.exists() {
        return existing(own);
    }
    let name = dir.file_name()?;
    let exp = dir.parent()?.parent()?;
    existing(exp.join(name).join("spectra").join("spectra_ms2.parquet"))
}

/// The diaPASEF `.d` the spectra at `spectra` were read from by the native reader.
fn raw_of_spectra(spectra: &str) -> Result<String> {
    match crate::instrument::provenance(spectra) {
        Some(p) if p.has_im() => {
            let raw = p.source.ok_or_else(|| {
                anyhow!("{spectra}: its report does not name the .d it was read from; pass --raw")
            })?;
            if !Path::new(&raw).is_dir() {
                bail!(
                    "{spectra} was converted from {raw}, which is not there any more; pass \
                     --raw with the .d's current path"
                );
            }
            Ok(raw)
        }
        _ => bail!(
            "{spectra} was not read natively from a diaPASEF .d (an mzML input, including a \
             timsTOF mzML written by msconvert, has no raw events); a mobilogram needs the \
             .d (docs/35_diapasef.md section 17)"
        ),
    }
}

/// The configuration a run recorded: `config_json` of `<dir>/manifest.json`, else of the
/// `experiment_manifest.json` one or two levels up (a run of an experiment, or of its second
/// pass). `None` when there is none or it does not parse.
fn recorded_config(dir: &Path) -> Option<(Config, String)> {
    let manifests = [
        Some(dir.join("manifest.json")),
        dir.parent().map(|d| d.join("experiment_manifest.json")),
        dir.parent()
            .and_then(Path::parent)
            .map(|d| d.join("experiment_manifest.json")),
    ];
    for m in manifests.into_iter().flatten() {
        let Ok(text) = std::fs::read_to_string(&m) else {
            continue;
        };
        let parsed = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v.get("config_json")?.as_str().map(str::to_string));
        let Some(cj) = parsed else {
            continue;
        };
        match Config::from_json(&cj) {
            Ok(c) => return Some((c, m.display().to_string())),
            Err(e) => warn!(
                manifest = %m.display(),
                error = %e,
                "mobilogram: the recorded configuration does not parse; using the defaults"
            ),
        }
    }
    None
}

/// Settle every input path and the configuration.
pub fn resolve(p: &mut MobilogramParams) -> Result<(Inputs, Config)> {
    let dir = p.run_dir.map(Path::new);
    if let Some(d) = dir {
        if !d.is_dir() {
            bail!("--run-dir {} is not a directory", d.display());
        }
    }
    let need = |given: Option<&str>, name: &str, flag: &str| -> Result<String> {
        if let Some(g) = given {
            return Ok(g.to_string());
        }
        let d = dir.ok_or_else(|| anyhow!("{flag} is required without --run-dir"))?;
        existing(d.join(name)).ok_or_else(|| anyhow!("{} has no {name}; pass {flag}", d.display()))
    };
    let psms_extracted = need(
        p.psms_extracted,
        "psms_extracted.parquet",
        "--psms-extracted",
    )?;
    let chromatograms = need(p.chromatograms, "chromatograms.parquet", "--chromatograms")?;
    let run_windows = match (p.run_windows, dir) {
        (Some(w), _) => Some(w.to_string()),
        (None, Some(d)) => existing(d.join("run_windows.parquet")),
        (None, None) => None,
    };
    let mass_cal = match (p.mass_cal, dir) {
        (Some(m), _) => Some(m.to_string()),
        (None, Some(d)) => existing(d.join("seed_psms.parquet.masscal.json")),
        (None, None) => None,
    };
    let raw = match (p.raw, dir) {
        (Some(r), _) => {
            if !Path::new(r).is_dir() {
                bail!("--raw {r} is not a timsTOF .d directory");
            }
            r.to_string()
        }
        (None, Some(d)) => {
            let spectra = spectra_of(d).ok_or_else(|| {
                anyhow!(
                    "{} holds no spectra/spectra_ms2.parquet to find the .d from; pass --raw",
                    d.display()
                )
            })?;
            raw_of_spectra(&spectra)?
        }
        (None, None) => bail!("--raw is required without --run-dir"),
    };
    let (cfg, config) = match p.config.take() {
        Some(c) => (c, "--config".to_string()),
        None => dir
            .and_then(recorded_config)
            .unwrap_or_else(|| (Config::default(), "defaults".to_string())),
    };
    Ok((
        Inputs {
            raw,
            psms_extracted,
            chromatograms,
            run_windows,
            mass_cal,
            config,
        },
        cfg,
    ))
}

/// The spans of `tf` (whole row groups, consecutive ones merged) whose `candidate_id`
/// statistics can hold one of `want`; a row group without statistics is always read. Taken
/// from the whole-file handle, and each starts at a row group, so a v2 chromatogram span
/// decodes on its own.
pub(crate) fn candidate_spans(tf: &TableFile, want: &BTreeSet<u32>) -> Result<Vec<TableFile>> {
    let stats = tf.row_group_stats("candidate_id")?;
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut first = 0usize;
    for s in &stats {
        let hit = match (s.min, s.max) {
            (Some(a), Some(b)) if a <= b => {
                let lo = a.max(0.0).floor() as u64;
                let hi = b.min(u32::MAX as f64).ceil() as u64;
                want.iter().any(|&c| (lo..=hi).contains(&(c as u64)))
            }
            (Some(_), Some(_)) => false,
            _ => true,
        };
        if hit && s.rows > 0 {
            match spans.last_mut() {
                Some(l) if l.0 + l.1 == first => l.1 += s.rows,
                _ => spans.push((first, s.rows)),
            }
        }
        first += s.rows;
    }
    spans.into_iter().map(|(a, n)| tf.span(a, n)).collect()
}

/// The `psms_extracted` row of a candidate this command reads.
#[derive(Debug, Clone, Copy)]
struct Peak {
    apex_rt: f64,
    apex_im: Option<f64>,
    ms1_apex_im: Option<f64>,
    precursor_mz: f64,
}

fn read_peaks(path: &str, want: &BTreeSet<u32>, rank: i32) -> Result<BTreeMap<u32, Peak>> {
    let tf = TableFile::open(path)?;
    let mut out = BTreeMap::new();
    for s in candidate_spans(&tf, want)? {
        let cid = s.u32("candidate_id")?;
        let ranks = if s.has_column("peak_rank") {
            s.i32("peak_rank")?
        } else {
            vec![0; cid.len()]
        };
        let opt = |name: &str| -> Result<Vec<Option<f64>>> {
            if s.has_column(name) {
                s.opt_f64(name)
            } else {
                Ok(vec![None; cid.len()])
            }
        };
        let (rt, pmz) = (s.f64_widening("apex_rt")?, s.f64_widening("precursor_mz")?);
        let (im, ms1_im) = (opt("apex_im")?, opt("ms1_apex_im")?);
        for i in 0..cid.len() {
            if ranks[i] == rank && want.contains(&cid[i]) {
                out.insert(
                    cid[i],
                    Peak {
                        apex_rt: rt[i],
                        apex_im: im[i].filter(|v| v.is_finite()),
                        ms1_apex_im: ms1_im[i].filter(|v| v.is_finite()),
                        precursor_mz: pmz[i],
                    },
                );
            }
        }
    }
    Ok(out)
}

/// A chromatogram row of a candidate: an ion to profile.
#[derive(Debug, Clone)]
struct Ion {
    name: String,
    ms1: bool,
    mz: f64,
    predicted: f32,
}

/// The ions of each wanted candidate, in chromatogram row order: the fragment rows with a
/// positive prediction and the MS1 isotope rows, each with a finite m/z.
fn read_ions(path: &str, want: &BTreeSet<u32>) -> Result<BTreeMap<u32, Vec<Ion>>> {
    let tf = TableFile::open(path)?;
    if !chromatograms::Optional::of(&tf).frag_mz {
        bail!("{path}: no frag_mz column, so the ions' m/z are unknown");
    }
    let mut out: BTreeMap<u32, Vec<Ion>> = BTreeMap::new();
    for s in candidate_spans(&tf, want)? {
        chromatograms::for_each_row(&s, |r| {
            if !want.contains(&r.cid) {
                return Ok(());
            }
            let ms1 = r.name.starts_with("ms1_");
            let mz = r.frag_mz.unwrap_or(f64::NAN);
            if mz.is_finite() && mz > 0.0 && (ms1 || r.predicted_intensity > 0.0) {
                out.entry(r.cid).or_default().push(Ion {
                    name: r.name.to_string(),
                    ms1,
                    mz,
                    predicted: r.predicted_intensity,
                });
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// `run_windows` 1/K0 window and calibrated prediction per wanted candidate, when the table
/// has them.
type ImWindow = (Option<[f64; 2]>, Option<f64>);

fn read_im_windows(path: Option<&str>, want: &BTreeSet<u32>) -> Result<BTreeMap<u32, ImWindow>> {
    let mut out = BTreeMap::new();
    let Some(path) = path else {
        return Ok(out);
    };
    let tf = TableFile::open(path)?;
    if !["im_lo", "im_hi", "im_pred_cal"]
        .iter()
        .all(|c| tf.has_column(c))
    {
        return Ok(out);
    }
    for s in candidate_spans(&tf, want)? {
        let cid = s.u32("candidate_id")?;
        let (lo, hi, pred) = (
            s.opt_f64("im_lo")?,
            s.opt_f64("im_hi")?,
            s.opt_f64("im_pred_cal")?,
        );
        for i in 0..cid.len() {
            if want.contains(&cid[i]) {
                let w = match (lo[i], hi[i]) {
                    (Some(a), Some(b)) if a.is_finite() && b.is_finite() => Some([a, b]),
                    _ => None,
                };
                out.insert(cid[i], (w, pred[i].filter(|v| v.is_finite())));
            }
        }
    }
    Ok(out)
}

/// The bin grid over `[im_bottom, im_top]`: `(im_lo, n_bins)` with `im_lo` a multiple of
/// `bin` at or below `im_bottom`, so grids of one bin width line up across candidates, and
/// enough bins that `im_top` falls in the last.
pub(crate) fn bin_grid(im_bottom: f64, im_top: f64, bin: f64) -> (f64, usize) {
    let mut lo = (im_bottom / bin).floor() * bin;
    if lo > im_bottom {
        lo -= bin;
    }
    let n = ((im_top - lo) / bin).floor() as usize + 1;
    (lo, n)
}

/// The frames `[j - n, j + n]` of `rts` (clipped), `j` the one nearest `t`, as positions and
/// their offsets from `j`.
pub(crate) fn frames_around(rts: &[f64], t: f64, n: usize) -> Vec<(usize, i32)> {
    if rts.is_empty() {
        return Vec::new();
    }
    let j = nearest_index(rts, t);
    let (a, b) = (j.saturating_sub(n), (j + n).min(rts.len() - 1));
    (a..=b).map(|k| (k, k as i32 - j as i32)).collect()
}

/// The frame index of a `.d`: headers only, by MS level, in RT order.
struct FrameIndex {
    heads: Vec<Frame>,
    ms1: Vec<usize>,
    ms1_rt: Vec<f64>,
    ms2: Vec<usize>,
}

impl FrameIndex {
    fn read(raw: &RawTdf, path: &str) -> Result<FrameIndex> {
        let heads: Vec<Frame> = (0..raw.reader.len())
            .into_par_iter()
            .map(|i| {
                raw.reader
                    .get_frame_without_coordinates(i)
                    .map_err(|e| anyhow!("{path}: frame header {i}: {e}"))
            })
            .collect::<Result<_>>()?;
        let (mut ms1, mut ms1_rt, mut ms2) = (Vec::new(), Vec::new(), Vec::new());
        for (i, f) in heads.iter().enumerate() {
            match f.ms_level {
                MSLevel::MS1 => {
                    ms1.push(i);
                    ms1_rt.push(f.rt_in_seconds);
                }
                MSLevel::MS2 => ms2.push(i),
                MSLevel::Unknown => {}
            }
        }
        let rt_order = |v: &[usize]| {
            v.windows(2)
                .all(|w| heads[w[0]].rt_in_seconds <= heads[w[1]].rt_in_seconds)
        };
        if ms1.is_empty() || ms2.is_empty() || !rt_order(&ms1) || !rt_order(&ms2) {
            bail!("{path}: frames are not in RT order, or MS1/MS2 frames are missing");
        }
        Ok(FrameIndex {
            heads,
            ms1,
            ms1_rt,
            ms2,
        })
    }
}

/// Half-open scan ranges of one frame.
type Scans = Vec<(usize, usize)>;

/// The planned reads of one candidate.
struct Plan {
    /// Chosen MS2 frames, each with the scans of its slots that hold the precursor.
    ms2: Vec<(FrameRef, Scans)>,
    /// Chosen MS1 frames.
    ms1: Vec<FrameRef>,
    im_lo: f64,
    n_bins: usize,
    slot_im: [f64; 2],
}

fn plan(
    fx: &FrameIndex,
    scan_im: &[f64],
    pmz: f64,
    centre: f64,
    frames: usize,
    bin: f64,
) -> Result<Plan> {
    let n_scans = scan_im.len();
    // MS2 frames with a slot that holds the precursor, in RT order.
    let cov: Vec<(usize, Scans)> = fx
        .ms2
        .iter()
        .filter_map(|&fi| {
            let sc = slot_scans(&fx.heads[fi].quadrupole_settings, pmz, (0, n_scans));
            (!sc.is_empty()).then_some((fi, sc))
        })
        .collect();
    if cov.is_empty() {
        bail!("no diaPASEF slot isolates the precursor m/z {pmz:.4}");
    }
    let rts: Vec<f64> = cov
        .iter()
        .map(|(fi, _)| fx.heads[*fi].rt_in_seconds)
        .collect();
    let ms2: Vec<(FrameRef, Scans)> = frames_around(&rts, centre, frames)
        .into_iter()
        .map(|(k, offset)| {
            let frame = FrameRef {
                index: cov[k].0,
                frame_id: fx.heads[cov[k].0].index,
                rt: rts[k],
                offset,
            };
            (frame, cov[k].1.clone())
        })
        .collect();
    let a = ms2
        .iter()
        .flat_map(|m| m.1.iter().map(|s| s.0))
        .min()
        .unwrap_or(0);
    let b = ms2
        .iter()
        .flat_map(|m| m.1.iter().map(|s| s.1))
        .max()
        .unwrap_or(0);
    if a >= b || b > n_scans {
        bail!("the slots holding the precursor cover no scans");
    }
    // Scan index runs against 1/K0: scan `a` has the highest 1/K0, scan `b - 1` the lowest.
    let (im_top, im_bottom) = (scan_im[a], scan_im[b - 1]);
    let (im_lo, n_bins) = bin_grid(im_bottom, im_top, bin);
    let ms1 = frames_around(&fx.ms1_rt, centre, frames)
        .into_iter()
        .map(|(k, offset)| FrameRef {
            index: fx.ms1[k],
            frame_id: fx.heads[fx.ms1[k]].index,
            rt: fx.ms1_rt[k],
            offset,
        })
        .collect();
    Ok(Plan {
        ms2,
        ms1,
        im_lo,
        n_bins,
        slot_im: [im_bottom, im_top],
    })
}

/// The mobilograms of `p.candidates`.
pub fn run(mut p: MobilogramParams) -> Result<Mobilograms> {
    if !(p.bin.is_finite() && p.bin >= MIN_BIN) {
        bail!(
            "--bin must be a 1/K0 width of at least {MIN_BIN}, got {}",
            p.bin
        );
    }
    if p.frames > MAX_FRAMES {
        bail!("--frames must be at most {MAX_FRAMES}, got {}", p.frames);
    }
    if p.candidates.is_empty() {
        bail!("no --candidate given");
    }
    if let Some(t) = p.rt {
        if !t.is_finite() {
            bail!("--rt must be finite, got {t}");
        }
    }
    let (inputs, cfg) = resolve(&mut p)?;
    let want: BTreeSet<u32> = p.candidates.iter().copied().collect();
    let peaks = read_peaks(&inputs.psms_extracted, &want, p.peak_rank)?;
    let ions = read_ions(&inputs.chromatograms, &want)?;
    let windows = read_im_windows(inputs.run_windows.as_deref(), &want)?;
    let (mass_off, frag_tol) = read_mass_cal(inputs.mass_cal.as_deref(), cfg.extract.frag_tol_ppm)?;
    let prec_tol = cfg.extract.prec_tol_ppm;

    let raw = RawTdf::open(&inputs.raw)?;
    let fx = FrameIndex::read(&raw, &inputs.raw)?;
    let n_scans = raw
        .frame(fx.ms1[0])
        .map_err(|e| anyhow!("{}: frame {}: {e}", inputs.raw, fx.ms1[0]))?
        .scan_offsets
        .len()
        .saturating_sub(1);
    let scan_im = scan_mobilities(n_scans, |s| raw.im_of_scan(s), &inputs.raw)?;

    let mut done = Vec::new();
    let mut errors = Vec::new();
    for &cid in &want {
        let r = (|| -> Result<CandidateMobilogram> {
            let pk = peaks.get(&cid).ok_or_else(|| {
                anyhow!(
                    "no psms_extracted row with peak_rank {} in {}",
                    p.peak_rank,
                    inputs.psms_extracted
                )
            })?;
            let ion_list = ions
                .get(&cid)
                .ok_or_else(|| anyhow!("no chromatogram rows with an m/z"))?;
            if !(pk.precursor_mz.is_finite() && pk.apex_rt.is_finite()) {
                bail!("the extracted row has no finite precursor m/z or apex RT");
            }
            let centre = p.rt.unwrap_or(pk.apex_rt);
            let pl = plan(&fx, &scan_im, pk.precursor_mz, centre, p.frames, p.bin)?;
            let band = scan_band(&scan_im, pl.im_lo, pl.im_lo + pl.n_bins as f64 * p.bin);
            let wanted: BTreeSet<usize> = pl
                .ms2
                .iter()
                .map(|m| m.0.index)
                .chain(pl.ms1.iter().map(|m| m.index))
                .collect();
            let decoded: BTreeMap<usize, TofFrame> = wanted
                .into_par_iter()
                .map(|fi| -> Result<(usize, TofFrame)> {
                    let f = raw
                        .frame(fi)
                        .map_err(|e| anyhow!("{}: frame {fi}: {e}", inputs.raw))?;
                    Ok((fi, TofFrame::new(f)?))
                })
                .collect::<Result<_>>()?;
            let profile = |ion: &Ion| -> IonProfile {
                let (query, tol) = if ion.ms1 {
                    (ion.mz, prec_tol)
                } else {
                    (ion.mz * mass_off.factor_at(ion.mz), frag_tol)
                };
                let t = tol * 1e-6;
                let tof = tof_window(&raw, query * (1.0 - t), query * (1.0 + t));
                let reads: Vec<(usize, Scans)> = if ion.ms1 {
                    pl.ms1.iter().map(|m| (m.index, vec![band])).collect()
                } else {
                    pl.ms2.iter().map(|m| (m.0.index, m.1.clone())).collect()
                };
                let per_frame: Vec<Vec<f64>> = reads
                    .iter()
                    .map(|(fi, scans)| {
                        let mut v = vec![0.0; pl.n_bins];
                        bin_events(
                            &decoded[fi],
                            scans,
                            tof.0,
                            tof.1,
                            &scan_im,
                            pl.im_lo,
                            p.bin,
                            &mut v,
                        );
                        v
                    })
                    .collect();
                let mut intensity = vec![0.0; pl.n_bins];
                for v in &per_frame {
                    for (s, x) in intensity.iter_mut().zip(v) {
                        *s += x;
                    }
                }
                IonProfile {
                    name: ion.name.clone(),
                    kind: if ion.ms1 { "ms1" } else { "fragment" },
                    mz: ion.mz,
                    query_mz: query,
                    tol_ppm: tol,
                    predicted_intensity: ion.predicted,
                    intensity,
                    per_frame,
                }
            };
            let (im_window, im_pred_cal) = windows.get(&cid).copied().unwrap_or((None, None));
            Ok(CandidateMobilogram {
                candidate_id: cid,
                peak_rank: p.peak_rank,
                precursor_mz: pk.precursor_mz,
                apex_rt: pk.apex_rt,
                apex_im: pk.apex_im,
                ms1_apex_im: pk.ms1_apex_im,
                centre_rt: centre,
                im_window,
                im_pred_cal,
                im_lo: pl.im_lo,
                bin: p.bin,
                im: (0..pl.n_bins)
                    .map(|k| pl.im_lo + (k as f64 + 0.5) * p.bin)
                    .collect(),
                slot_im: pl.slot_im,
                ions: ion_list.iter().map(profile).collect(),
                ms2_frames: pl.ms2.iter().map(|m| m.0.clone()).collect(),
                ms1_frames: pl.ms1.clone(),
            })
        })();
        match r {
            Ok(m) => done.push(m),
            Err(e) => {
                warn!(candidate_id = cid, error = %e, "mobilogram: candidate skipped");
                errors.push(CandidateError {
                    candidate_id: cid,
                    error: format!("{e:#}"),
                });
            }
        }
    }
    info!(
        candidates = done.len(),
        skipped = errors.len(),
        raw = %inputs.raw,
        "mobilogram: done"
    );
    Ok(Mobilograms {
        format: FORMAT,
        format_version: FORMAT_VERSION,
        mumdia_version: mumdia_core::version(),
        inputs,
        bin: p.bin,
        frames: p.frames,
        peak_rank: p.peak_rank,
        frag_tol_ppm: frag_tol,
        frag_ppm_offset: mass_off.scalar_ppm,
        frag_offset_grid_points: mass_off.grid_len(),
        prec_tol_ppm: prec_tol,
        im_calibration: raw.im_calibration(),
        ms1_denoise: false,
        candidates: done,
        errors,
    })
}

/// Write the profiles as a long table: one row per candidate, ion, frame and non-zero bin
/// (`candidate_id`, `peak_rank`, `ion`, `kind`, `ion_mz`, `frame_index`, `frame_id`,
/// `frame_rt`, `frame_offset`, `im`, `intensity`). A candidate in `errors` has no rows.
pub fn write_parquet(m: &Mobilograms, out: &str) -> Result<u64> {
    let (mut cid, mut rank, mut ion, mut kind, mut ion_mz) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut fidx, mut fid, mut frt, mut foff, mut im, mut inten) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for c in &m.candidates {
        for p in &c.ions {
            let frames = if p.kind == "ms1" {
                &c.ms1_frames
            } else {
                &c.ms2_frames
            };
            for (f, bins) in frames.iter().zip(&p.per_frame) {
                for (k, &v) in bins.iter().enumerate() {
                    if v == 0.0 {
                        continue;
                    }
                    cid.push(c.candidate_id);
                    rank.push(c.peak_rank);
                    ion.push(p.name.clone());
                    kind.push(p.kind.to_string());
                    ion_mz.push(p.mz);
                    fidx.push(f.index as i64);
                    fid.push(f.frame_id as i64);
                    frt.push(f.rt);
                    foff.push(f.offset);
                    im.push(c.im[k]);
                    inten.push(v);
                }
            }
        }
    }
    write_table(
        out,
        vec![
            Col::U32("candidate_id".into(), cid),
            Col::I32("peak_rank".into(), rank),
            Col::Str("ion".into(), ion),
            Col::Str("kind".into(), kind),
            Col::F64("ion_mz".into(), ion_mz),
            Col::I64("frame_index".into(), fidx),
            Col::I64("frame_id".into(), fid),
            Col::F64("frame_rt".into(), frt),
            Col::I32("frame_offset".into(), foff),
            Col::F64("im".into(), im),
            Col::F64("intensity".into(), inten),
        ],
    )
    .with_context(|| format!("writing {out}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bin_grid_starts_on_a_bin_multiple_and_holds_both_ends() {
        let (lo, n) = bin_grid(0.8123, 1.0456, 0.002);
        assert!((lo - 0.812).abs() < 1e-12);
        assert!(lo <= 0.8123);
        assert!(lo + n as f64 * 0.002 > 1.0456);
        assert!(lo + (n - 1) as f64 * 0.002 <= 1.0456);
        // An end on a bin edge still lies inside the grid.
        let (lo, n) = bin_grid(0.8, 0.9, 0.05);
        assert!(lo <= 0.8 && ((0.8 - lo) / 0.05) >= 0.0);
        assert!(((0.9 - lo) / 0.05).floor() < n as f64);
    }

    #[test]
    fn frames_around_takes_the_nearest_and_its_neighbours_clipped() {
        let rts = [10.0, 11.0, 12.0, 13.0, 14.0];
        assert_eq!(frames_around(&rts, 12.2, 1), vec![(1, -1), (2, 0), (3, 1)]);
        assert_eq!(frames_around(&rts, 9.0, 2), vec![(0, 0), (1, 1), (2, 2)]);
        assert_eq!(frames_around(&rts, 99.0, 0), vec![(4, 0)]);
        // A tie goes to the earlier frame, as extract's nearest_index does.
        assert_eq!(frames_around(&rts, 12.5, 0), vec![(2, 0)]);
        assert!(frames_around(&[], 1.0, 2).is_empty());
    }

    #[test]
    fn candidate_spans_read_only_the_row_groups_that_can_hold_a_candidate() {
        let dir = std::env::temp_dir().join(format!("mumdia_mobspan_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.parquet").display().to_string();
        // Ten rows in groups of three: ids 0 0 1 | 1 2 2 | 5 5 6 | 9.
        let ids: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 5, 5, 6, 9];
        let vals: Vec<f64> = (0..10).map(f64::from).collect();
        let mut w = mumdia_io::table::TableWriter::new(&path).with_row_group_rows(3);
        w.write_cols(vec![
            Col::U32("candidate_id".into(), ids),
            Col::F64("v".into(), vals),
        ])
        .unwrap();
        w.close().unwrap();
        let tf = TableFile::open(&path).unwrap();
        assert_eq!(tf.row_group_stats("candidate_id").unwrap().len(), 4);
        let rows = |want: &[u32]| -> Vec<u32> {
            let want: BTreeSet<u32> = want.iter().copied().collect();
            candidate_spans(&tf, &want)
                .unwrap()
                .iter()
                .flat_map(|s| s.u32("candidate_id").unwrap())
                .collect()
        };
        // Candidate 1 straddles groups 0 and 1, which are read as one span.
        assert_eq!(rows(&[1]), vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(rows(&[6]), vec![5, 5, 6]);
        assert_eq!(rows(&[3, 4]), Vec::<u32>::new());
        assert_eq!(rows(&[0, 9]), vec![0, 0, 1, 9]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_second_pass_run_finds_the_first_pass_spectra() {
        let root = std::env::temp_dir().join(format!("mumdia_mobsp_{}", std::process::id()));
        let first = root.join("r0").join("spectra");
        let second = root.join("second_pass").join("r0");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        std::fs::write(first.join("spectra_ms2.parquet"), b"x").unwrap();
        let got = spectra_of(&second).unwrap();
        assert!(got.ends_with("spectra_ms2.parquet") && got.contains("r0"));
        assert_eq!(spectra_of(&root.join("r0")), Some(got));
        assert_eq!(spectra_of(&root.join("second_pass")), None);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_mzml_run_is_refused_with_the_reason() {
        let dir = std::env::temp_dir().join(format!("mumdia_mobmz_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spectra = dir.join("spectra_ms2.parquet").display().to_string();
        std::fs::write(
            format!("{spectra}.report.json"),
            r#"{"params": {"mzml": "/x/run.mzML", "reader": "mzml"}}"#,
        )
        .unwrap();
        let e = raw_of_spectra(&spectra).unwrap_err().to_string();
        assert!(e.contains("not read natively"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Real-data check: `MUMDIA_BENCH_TDF=/path/run.d cargo test -- --ignored
    /// tdf_real_mobilogram`. On the first MS2 frames, a slot's most intense raw event is
    /// profiled through the same helpers `run` uses: the binned intensity equals the summed
    /// window, the profile peaks at that event's scan, and the bin range covers the slot.
    #[test]
    #[ignore]
    fn tdf_real_mobilogram_conserves_the_window_and_peaks_at_the_strongest_scan() {
        use crate::stages::convert::tims_frames::sum_events;
        let Ok(path) = std::env::var("MUMDIA_BENCH_TDF") else {
            return;
        };
        let raw = RawTdf::open(&path).unwrap();
        let fx = FrameIndex::read(&raw, &path).unwrap();
        let n_scans = raw.frame(fx.ms1[0]).unwrap().scan_offsets.len() - 1;
        let scan_im = scan_mobilities(n_scans, |s| raw.im_of_scan(s), &path).unwrap();
        let mut checked = 0;
        for &fi in fx.ms2.iter().take(40) {
            let f = raw.frame(fi).unwrap();
            let q = f.quadrupole_settings.clone();
            let tf = TofFrame::new(raw.frame(fi).unwrap()).unwrap();
            for k in 0..q.len() {
                let (s0, s1) = (q.scan_starts[k], q.scan_ends[k].min(n_scans));
                // The most intense event of the slot.
                let mut best = (0u32, 0u32, 0usize);
                for s in s0..s1 {
                    for j in f.scan_offsets[s]..f.scan_offsets[s + 1] {
                        if f.intensities[j] > best.0 {
                            best = (f.intensities[j], f.tof_indices[j], s);
                        }
                    }
                }
                if best.0 == 0 {
                    continue;
                }
                let pmz = q.isolation_mz[k];
                let scans = slot_scans(&q, pmz, (0, n_scans));
                let a = scans.iter().map(|s| s.0).min().unwrap();
                let b = scans.iter().map(|s| s.1).max().unwrap();
                let (lo, n) = bin_grid(scan_im[b - 1], scan_im[a], DEFAULT_BIN);
                let mut v = vec![0.0; n];
                let tof = (best.1, best.1);
                let got = bin_events(&tf, &scans, tof.0, tof.1, &scan_im, lo, DEFAULT_BIN, &mut v);
                let s = sum_events(&tf, &scans, tof.0, tof.1, &scan_im);
                assert_eq!(got, s.inten, "frame {fi} slot {k}");
                let centre = s.imw / s.inten;
                let com: f64 = v
                    .iter()
                    .enumerate()
                    .map(|(k, w)| w * (lo + (k as f64 + 0.5) * DEFAULT_BIN))
                    .sum::<f64>()
                    / got;
                assert!(
                    (com - centre).abs() <= DEFAULT_BIN / 2.0 + 1e-9,
                    "{com} {centre}"
                );
                let kbest = ((scan_im[best.2] - lo) / DEFAULT_BIN) as usize;
                assert!(v[kbest] >= best.0 as f64);
                checked += 1;
            }
        }
        assert!(checked > 0);
        eprintln!(
            "mobilogram: {checked} slot profiles checked; 1/K0 calibration {}",
            raw.im_calibration()
        );
    }
}
