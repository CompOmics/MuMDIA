//! Stage B `mumdia rt-im-train`: per-run RT calibration and windows
//! (docs/08_rt_im_train.md). Calibrates the run-independent predicted iRT to
//! observed RT from confident seed PSMs, then sets a per-candidate RT window from
//! the residuals. MVP is 3D, so IM columns are null. The sidecars are not re-run
//! here.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{CalibrationMethod, RtImTrainConfig};
use mumdia_core::schema::artifact;
use mumdia_io::report::{ArtifactReport, Written};
use mumdia_io::table::{write_table_chunked_hashed, Col, TableFile};
use serde_json::json;
use tracing::{info, warn};

use crate::calibrate::{linear_fit, percentile, Loess};

pub struct RtImTrainParams<'a> {
    pub seed_psms: &'a str,
    pub library_precursors: &'a str,
    pub out_windows: &'a str,
    pub out_cal: &'a str,
    pub cfg: &'a RtImTrainConfig,
    pub config_hash: &'a str,
    /// Take each anchor's predicted iRT from the seed table's own `predicted_irt` column
    /// instead of joining the library by `candidate_id`. For a pooled seed of a grouped
    /// run: its ids are library-wide while the band's table is local, and the pooling step
    /// has refreshed the column from the re-predicted band libraries, so the seed is the
    /// source of truth there. Off, anchors outside the library are dropped silently.
    pub anchor_irt_from_seed: bool,
    /// `library_precursors` is the WHOLE library and the windows are written for its rows
    /// `[first, first + n)` only, with local ids `0..n`: what the same rows written out as a
    /// band file would give. `None` reads the whole table.
    pub precursor_span: Option<(usize, usize)>,
}

const INSUFFICIENT_ANCHORS_STATUS: &str = "insufficient_anchors_unbounded";

#[derive(Clone, Copy, Debug, PartialEq)]
enum WindowPlan {
    /// No trustworthy iRT -> run-RT mapping exists. Extract over the complete
    /// isolation-window RT range and mark calibrated RT as unavailable.
    Unbounded,
    /// A linear mapping is available, but too few anchors exist to estimate its
    /// residual distribution. Retain the configured broad fixed half-window.
    Fixed(f64),
    /// Enough anchors exist to derive the residual-percentile half-window.
    Calibrated,
}

fn window_plan(n_train: usize, min_anchors: usize, fallback_width: f64) -> WindowPlan {
    if n_train < 2 {
        WindowPlan::Unbounded
    } else if n_train < min_anchors {
        WindowPlan::Fixed(fallback_width)
    } else {
        WindowPlan::Calibrated
    }
}

/// Convert the best-per-base-peptide map into a fixed anchor order before any
/// floating-point fit or reduction. HashMap iteration order is randomized.
fn sorted_anchor_vectors(
    best_per_pep: HashMap<u32, (f64, f64, f64)>,
) -> (Vec<u32>, Vec<f64>, Vec<f64>) {
    let mut anchors: Vec<(u32, (f64, f64, f64))> = best_per_pep.into_iter().collect();
    anchors.sort_by_key(|(base_peptide_id, _)| *base_peptide_id);
    let ids = anchors.iter().map(|(id, _)| *id).collect();
    let train_irt = anchors.iter().map(|(_, values)| values.1).collect();
    let train_rt = anchors.iter().map(|(_, values)| values.2).collect();
    (ids, train_irt, train_rt)
}

/// Deterministic anchor-peptide holdout for window sizing. The rule
/// (`base_peptide_id % 1000 < round(frac*1000)`) is duplicated verbatim in
/// `deeplc_finetune.py` so the fine-tune reference excludes exactly these
/// peptides; keying on the base peptide keeps every charge/modform of one
/// peptide on the same side of the split (a row-wise split leaks).
fn is_holdout(base_peptide_id: u32, frac: f64) -> bool {
    base_peptide_id % 1000 < (frac * 1000.0).round() as u32
}

/// Minimum held-out anchors for a trustworthy p95; below this, sizing falls
/// back to in-sample with a warning rather than trusting a noisy tail estimate.
const MIN_HOLDOUT_ANCHORS: usize = 20;

/// Result of held-out window sizing, kept for `cal.json` so a run records both
/// which sizing produced `w_rt` and the held-out residual scale it came from.
struct HoldoutSizing {
    width: f64,
    n_sizing_train: usize,
    n_holdout: usize,
    /// Held-out residual at the configured `p_rt` percentile, pre-multiplier.
    resid_p_rt_s: f64,
    resid_abs_median_s: f64,
}

/// The `candidate_id -> predicted_irt` join the anchors are read through.
///
/// It used to be a `HashMap<u32, f64>` over every library row, built unconditionally:
/// about 4.5 GB of transient on an unbanded 203M-row library, built even when
/// `anchor_irt_from_seed` meant it was never read. A library as the engine writes and
/// loads it carries `candidate_id` as the row-aligned range `0..n` (`index.rs` refuses
/// anything else), so the join is an index into the iRT column itself. The hash table
/// remains for any other id layout, and it keeps exactly the old meaning there: the LAST
/// row with a given id wins, as repeated `insert`s did.
enum IrtJoin<'a> {
    /// `anchor_irt_from_seed`: the seed table carries the iRT, nothing is joined.
    Unused,
    /// `candidate_id[i] == i` for every row: the id is the row.
    Dense(&'a [f32]),
    /// Any other layout: the historical hash join.
    Map(HashMap<u32, f64>),
}

impl<'a> IrtJoin<'a> {
    fn new(cid: &[u32], irt: &'a [f32]) -> IrtJoin<'a> {
        if cid.len() == irt.len() && cid.iter().enumerate().all(|(i, &c)| c as usize == i) {
            return IrtJoin::Dense(irt);
        }
        let mut m: HashMap<u32, f64> = HashMap::with_capacity(cid.len());
        for (&c, &v) in cid.iter().zip(irt) {
            m.insert(c, v as f64);
        }
        IrtJoin::Map(m)
    }

    /// The library iRT of candidate `c`, widened exactly as the hash join stored it;
    /// `None` when the library has no such candidate.
    #[inline]
    fn get(&self, c: u32) -> Option<f64> {
        match self {
            IrtJoin::Unused => None,
            IrtJoin::Dense(irt) => irt.get(c as usize).map(|&v| v as f64),
            IrtJoin::Map(m) => m.get(&c).copied(),
        }
    }
}

/// Materialize one candidate's RT metadata. `None` means calibration is
/// unavailable: NaN is an explicit internal sentinel for `rt_pred_cal`, while
/// infinite bounds make extraction recall-safe.
fn candidate_window(calibrated_rt: Option<f64>, width: Option<f64>) -> (f64, f64, f64) {
    match (calibrated_rt, width) {
        (Some(cal), Some(w)) => (cal, cal - w, cal + w),
        _ => (f64::NAN, f64::NEG_INFINITY, f64::INFINITY),
    }
}

/// A fitted retention-time calibration: everything the anchors determine, which
/// [`apply`] then maps over a library table.
///
/// Under `groups.calibration = global` every band of a grouped run fits the same pooled
/// anchors with their iRT read from the seed (`anchor_irt_from_seed`), so the fit depends on
/// the seed and the configuration only, never on the band's library. `run_groups` therefore
/// fits once per run ([`fit_from_seed`]) and applies the one fit to every band, which is
/// what each band computed for itself before: one seed decode and one fit per band
/// (docs/33 section 4).
pub struct RtFit {
    n_train: usize,
    calibration_available: bool,
    slope: f64,
    intercept: f64,
    use_loess: bool,
    loess: Option<Loess>,
    w_rt: Option<f64>,
    status: String,
    holdout_frac: f64,
    holdout_sizing: Option<HoldoutSizing>,
    adaptive: Option<(f64, f64, Vec<f64>)>,
    /// Signed median, absolute median and MAD of the in-sample residuals (seconds).
    residuals: (f64, f64, f64),
}

/// The fitted windows, handed from `rt-im-train` to `extract` in memory by the
/// orchestrators (`run`, `run-experiment`, the grouped band loop) instead of through a
/// re-read of `run_windows.parquet`.
///
/// It is exactly what `extract` would build from the file it was just written to: three
/// arrays indexed by candidate id over the library's `n` rows, `(NaN, -inf, +inf)` for a
/// candidate with no row, and for every row `i` with `candidate_id[i] < n`, in row order,
/// that row's `(rt_pred_cal, rt_lo, rt_hi)`. The file is still written and is still the
/// contract of the standalone stages.
///
/// The arrays carry the identity of what they were fitted for: the precursor table
/// `rt-im-train` read and the `run_windows` path it wrote. `extract` takes them only when
/// both are the paths it was itself given and its library has the same `n` candidates
/// ([`RtWindows::mismatch`]), and reads the file otherwise. A count alone would accept
/// windows fitted on a different library of the same size (a re-predicted or fine-tuned
/// precursor table, another band), which the file-based contract cannot do, because
/// `extract` reads the `run_windows` path it is given.
pub struct RtWindows {
    pub(crate) rt_cal: Vec<f64>,
    pub(crate) rt_lo: Vec<f64>,
    pub(crate) rt_hi: Vec<f64>,
    /// `(library_precursors, run_windows)` of the `rt-im-train` call that fitted these
    /// windows; `None` for windows read back from a file, which are never handed over.
    pub(crate) fitted_for: Option<(String, String)>,
}

impl RtWindows {
    /// Candidates covered (the library's row count).
    pub fn len(&self) -> usize {
        self.rt_cal.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rt_cal.is_empty()
    }

    /// Why these windows are NOT the ones `extract` would read from `run_windows` for the
    /// `ncand`-candidate library at `library_precursors`, or `None` when they are: the same
    /// precursor table and the same `run_windows` path as the fit, and the same candidate
    /// count. The paths are compared as given, so an equivalent path spelled differently
    /// reads the file, which is the safe direction.
    pub fn mismatch(
        &self,
        library_precursors: &str,
        run_windows: &str,
        ncand: usize,
    ) -> Option<String> {
        let Some((lib, rw)) = &self.fitted_for else {
            return Some("they were not fitted by rt-im-train in this process".to_string());
        };
        if lib != library_precursors {
            return Some(format!(
                "they were fitted on the precursor table {lib}, not {library_precursors}"
            ));
        }
        if rw != run_windows {
            return Some(format!(
                "they were written to {rw}, not to the run_windows path {run_windows}"
            ));
        }
        if self.len() != ncand {
            return Some(format!(
                "they cover {} candidates and this library has {ncand}",
                self.len()
            ));
        }
        None
    }
}

/// Collects the dense windows while the table is streamed out, or refuses to.
struct RtWindowsBuilder {
    w: RtWindows,
    /// A row with a NaN `rt_lo` or `rt_hi` was seen. `extract` rejects such a table with
    /// the row named; the in-memory form is then withheld, so `extract` reads the file
    /// and reports exactly that error.
    poisoned: bool,
}

impl RtWindowsBuilder {
    fn new(n: usize) -> RtWindowsBuilder {
        RtWindowsBuilder {
            w: RtWindows {
                rt_cal: vec![f64::NAN; n],
                rt_lo: vec![f64::NEG_INFINITY; n],
                rt_hi: vec![f64::INFINITY; n],
                fitted_for: None,
            },
            poisoned: false,
        }
    }

    /// One written row, in row order: the scatter `extract` applies to the file.
    #[inline]
    fn row(&mut self, cid: u32, cal: f64, lo: f64, hi: f64) {
        let c = cid as usize;
        if c < self.w.rt_cal.len() {
            if lo.is_nan() || hi.is_nan() {
                self.poisoned = true;
            }
            self.w.rt_cal[c] = cal;
            self.w.rt_lo[c] = lo;
            self.w.rt_hi[c] = hi;
        }
    }

    /// The windows, stamped with the precursor table they were fitted on and the
    /// `run_windows` path they were written to; `None` when a NaN bound was seen.
    fn finish(self, library_precursors: &str, run_windows: &str) -> Option<RtWindows> {
        let mut w = self.w;
        w.fitted_for = Some((library_precursors.to_string(), run_windows.to_string()));
        (!self.poisoned).then_some(w)
    }
}

impl RtFit {
    fn predict(&self, irt: f64) -> f64 {
        if !self.calibration_available {
            return f64::NAN;
        }
        match &self.loess {
            Some(l) => l.predict(irt),
            None => self.slope * irt + self.intercept,
        }
    }

    /// Anchors the fit was made on.
    pub fn n_train(&self) -> usize {
        self.n_train
    }
}

/// What [`apply`] writes: the windows of one library table and its `cal.json`.
pub struct ApplyParams<'a> {
    pub library_precursors: &'a str,
    /// As [`RtImTrainParams::precursor_span`].
    pub precursor_span: Option<(usize, usize)>,
    pub out_windows: &'a str,
    pub out_cal: &'a str,
    pub cfg: &'a RtImTrainConfig,
    pub config_hash: &'a str,
}

fn check_cfg(cfg: &RtImTrainConfig) -> Result<()> {
    let holdout_frac = cfg.window_holdout_frac;
    if !(0.0..=0.9).contains(&holdout_frac) {
        anyhow::bail!(
            "rt_im_train.window_holdout_frac must be in [0.0, 0.9], got {holdout_frac}; \
             0.0 disables held-out window sizing"
        );
    }
    if holdout_frac > 0.0 && cfg.adaptive_rt_window {
        anyhow::bail!(
            "rt_im_train.window_holdout_frac and rt_im_train.adaptive_rt_window are mutually \
             exclusive: the adaptive per-bin percentiles are in-sample and would silently undo \
             the held-out sizing; disable one of them"
        );
    }
    Ok(())
}

pub fn run(p: RtImTrainParams) -> Result<u64> {
    run_hashed(p).map(|w| w.rows)
}

/// [`run`], returning the output's row count and the content hash its report records, so
/// an orchestrator can record the artifact without reading and hashing it again.
pub fn run_hashed(p: RtImTrainParams) -> Result<Written> {
    run_impl(p, false).map(|(written, _)| written)
}

/// [`run_hashed`], also returning the fitted windows in the form `extract` consumes, so an
/// orchestrator can hand them over instead of having `extract` decode the table it just
/// wrote. `None` only when the table holds a NaN bound (see [`RtWindows`]).
pub fn run_in_memory(p: RtImTrainParams) -> Result<(Written, Option<RtWindows>)> {
    run_impl(p, true)
}

fn run_impl(p: RtImTrainParams, keep_windows: bool) -> Result<(Written, Option<RtWindows>)> {
    let t0 = Instant::now();
    // `--out` must not be one of this stage's own inputs: every input is read
    // before the output is published, so writing over one replaces it and exits 0
    // (docs/31 F6). The shared guard existed and was wired into two stages.
    for out in [p.out_windows, p.out_cal] {
        mumdia_io::refuse_output_over_input(
            out,
            &[
                ("--seed-psms", p.seed_psms),
                ("--lib-precursors", p.library_precursors),
            ],
        )?;
    }
    check_cfg(p.cfg)?;

    // Library predicted iRT, keyed by candidate_id (single source of truth, so
    // a patched/updated library iRT is used for both training and application).
    let (lib_cid, lib_irt) = library_irt(p.library_precursors, p.precursor_span)?;
    // The candidate_id -> library iRT join, built only when the anchors take their iRT from
    // the library, and without a hash table when the ids allow it (see `IrtJoin`).
    let irt_join = if p.anchor_irt_from_seed {
        IrtJoin::Unused
    } else {
        IrtJoin::new(&lib_cid, &lib_irt)
    };
    let fit = fit_anchors(p.seed_psms, p.cfg, &irt_join)?;
    // The join borrows `lib_irt`, which the application pass below takes by value; its
    // hash-map form has drop glue, so end it here explicitly.
    drop(irt_join);
    write_windows(
        &fit,
        &ApplyParams {
            library_precursors: p.library_precursors,
            out_windows: p.out_windows,
            out_cal: p.out_cal,
            cfg: p.cfg,
            config_hash: p.config_hash,
            precursor_span: p.precursor_span,
        },
        lib_cid,
        lib_irt,
        t0,
        keep_windows,
    )
}

/// A library's `candidate_id` and `predicted_irt`: the whole table, or its rows
/// `[first, first + n)` with the ids made local (`id - first`), which is what the same rows
/// written as a band file carry. The span's ids must be the row-aligned range the library
/// invariant promises.
fn library_irt(path: &str, span: Option<(usize, usize)>) -> Result<(Vec<u32>, Vec<f32>)> {
    match span {
        None => {
            let lib = TableFile::open(path)?;
            Ok((lib.u32("candidate_id")?, lib.f32("predicted_irt")?))
        }
        Some((first, n)) => {
            let lib = TableFile::open_rows(path, first, n)?;
            let mut cid = lib.u32("candidate_id")?;
            let first32 = u32::try_from(first)
                .map_err(|_| anyhow::anyhow!("row span start {first} does not fit u32"))?;
            for (k, c) in cid.iter_mut().enumerate() {
                if *c != first32 + k as u32 {
                    anyhow::bail!(
                        "{path}: row {} has candidate_id {c}, expected {}; the library must \
                         carry candidate_id as the contiguous row-aligned range",
                        first + k,
                        first32 + k as u32
                    );
                }
                *c -= first32;
            }
            Ok((cid, lib.f32("predicted_irt")?))
        }
    }
}

/// Fit the calibration on a seed table whose own `predicted_irt` column carries the anchors'
/// iRT (`RtImTrainParams::anchor_irt_from_seed`): the fit a grouped run's bands share under
/// `groups.calibration = global`. No library is read.
pub fn fit_from_seed(seed_psms: &str, cfg: &RtImTrainConfig) -> Result<RtFit> {
    check_cfg(cfg)?;
    fit_anchors(seed_psms, cfg, &IrtJoin::Unused)
}

/// Write the windows of `p.library_precursors` and its `cal.json` from a fit made by
/// [`fit_from_seed`]. With the fit [`run`] would have made and the same table, this writes
/// what [`run`] writes, byte for byte (the report's `elapsed_ms` aside).
pub fn apply(fit: &RtFit, p: &ApplyParams) -> Result<u64> {
    apply_impl(fit, p, false).map(|(written, _)| written.rows)
}

/// [`apply`], also returning the windows in the form `extract` consumes, as
/// [`run_in_memory`] does for [`run`].
pub fn apply_in_memory(fit: &RtFit, p: &ApplyParams) -> Result<(Written, Option<RtWindows>)> {
    apply_impl(fit, p, true)
}

fn apply_impl(
    fit: &RtFit,
    p: &ApplyParams,
    keep_windows: bool,
) -> Result<(Written, Option<RtWindows>)> {
    let t0 = Instant::now();
    for out in [p.out_windows, p.out_cal] {
        mumdia_io::refuse_output_over_input(out, &[("--lib-precursors", p.library_precursors)])?;
    }
    let (lib_cid, lib_irt) = library_irt(p.library_precursors, p.precursor_span)?;
    write_windows(fit, p, lib_cid, lib_irt, t0, keep_windows)
}

/// The anchors and the fit. `irt_join` joins each anchor's iRT from the library;
/// [`IrtJoin::Unused`] reads it from the seed's own `predicted_irt` column.
fn fit_anchors(seed_psms: &str, cfg: &RtImTrainConfig, irt_join: &IrtJoin) -> Result<RtFit> {
    let holdout_frac = cfg.window_holdout_frac;
    // Training rows: confident seed PSMs, one apex (best score) per peptide;
    // predicted iRT is joined from the library by candidate_id.
    let seed = TableFile::open(seed_psms)?;
    let s_cid = seed.u32("candidate_id")?;
    let s_base = seed.u32("base_peptide_id")?;
    let s_q = seed.f64("spectrum_q")?;
    let s_score = seed.f64("score")?;
    let s_rt = seed.f64("observed_rt")?;
    let s_label = seed.str("label")?;
    let s_irt: Option<Vec<f32>> = if matches!(irt_join, IrtJoin::Unused) {
        Some(seed.f32("predicted_irt")?)
    } else {
        None
    };

    let mut best_per_pep: HashMap<u32, (f64, f64, f64)> = HashMap::new(); // base -> (score, irt, rt)
    for i in 0..seed.nrows {
        if !s_q[i].is_finite()
            || !s_score[i].is_finite()
            || !s_rt[i].is_finite()
            || s_q[i] >= cfg.q_train
        {
            continue;
        }
        // Only target PSMs may anchor the RT calibration; a decoy anchor injects a
        // random iRT<->RT pair into the fit.
        if s_label[i] != "target" {
            continue;
        }
        let irt = match &s_irt {
            Some(col) => {
                let v = col[i] as f64;
                if !v.is_finite() {
                    continue;
                }
                v
            }
            None => match irt_join.get(s_cid[i]) {
                Some(v) if v.is_finite() => v,
                None => continue,
                Some(_) => continue,
            },
        };
        let e = best_per_pep
            .entry(s_base[i])
            .or_insert((f64::NEG_INFINITY, 0.0, 0.0));
        if s_score[i] > e.0 {
            *e = (s_score[i], irt, s_rt[i]);
        }
    }
    let (anchor_ids, train_irt, train_rt) = sorted_anchor_vectors(best_per_pep);
    let n_train = train_irt.len();
    info!(n_train, "rt-im-train: training points");

    // Fit calibration predicted_irt -> observed RT (seconds).
    // Zero or one point cannot define a useful mapping across the gradient.
    let calibration_available = n_train >= 2;
    let (slope, intercept) = if calibration_available {
        linear_fit(&train_irt, &train_rt)
    } else {
        (f64::NAN, f64::NAN)
    };
    let use_loess = calibration_available
        && matches!(cfg.calibration_method, CalibrationMethod::Loess)
        && n_train >= cfg.min_seed_for_calibration;
    let loess = if use_loess {
        Some(Loess::fit(&train_irt, &train_rt, cfg.loess_span, 200))
    } else {
        None
    };

    let mut fit = RtFit {
        n_train,
        calibration_available,
        slope,
        intercept,
        use_loess,
        loess,
        w_rt: None,
        status: String::new(),
        holdout_frac,
        holdout_sizing: None,
        adaptive: None,
        residuals: (f64::NAN, f64::NAN, f64::NAN),
    };

    // Residuals and RT window. Require enough anchors before trusting the
    // residual-percentile window: with only a handful of points a linear fit
    // passes ~exactly through them, so residuals ~0 and the window collapses to
    // the 1s floor (which then discards nearly every true co-elution). Below the
    // threshold, use the configured fixed fallback instead.
    let min_anchors = cfg.min_seed_for_calibration.max(2);
    let plan = window_plan(n_train, min_anchors, cfg.fallback_rt_window_s);

    // Held-out window sizing (`window_holdout_frac > 0`): fit the sizing curve on
    // the non-held-out anchors only and take the residual percentile of the
    // held-out anchors against it. In-sample residuals underestimate the tail an
    // unseen library peptide actually has, and they reward a memorizing RT model
    // with a narrower window; the held-out anchors never entered this fit nor
    // (when `finetune_deeplc` ran with the same fraction) the DeepLC fine-tune,
    // so their residuals are an honest error estimate. The final calibration
    // curve applied to the library still uses every anchor; only the width is
    // sized out-of-sample, which is slightly conservative (the all-anchor curve
    // is marginally better than the sizing curve).
    let holdout_sizing: Option<HoldoutSizing> =
        if holdout_frac > 0.0 && plan == WindowPlan::Calibrated {
            let (mut tr_x, mut tr_y, mut ho_x, mut ho_y) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for k in 0..n_train {
                if is_holdout(anchor_ids[k], holdout_frac) {
                    ho_x.push(train_irt[k]);
                    ho_y.push(train_rt[k]);
                } else {
                    tr_x.push(train_irt[k]);
                    tr_y.push(train_rt[k]);
                }
            }
            if ho_x.len() < MIN_HOLDOUT_ANCHORS || tr_x.len() < min_anchors {
                warn!(
                    n_holdout = ho_x.len(),
                    n_sizing_train = tr_x.len(),
                    min_holdout = MIN_HOLDOUT_ANCHORS,
                    min_anchors,
                    "rt-im-train: too few anchors on one side of the holdout split; \
                 falling back to in-sample window sizing"
                );
                None
            } else {
                // Same method selection as the main fit, refit on the sizing subset.
                let sizing_loess = use_loess.then(|| Loess::fit(&tr_x, &tr_y, cfg.loess_span, 200));
                let (s_slope, s_intercept) = if sizing_loess.is_none() {
                    linear_fit(&tr_x, &tr_y)
                } else {
                    (f64::NAN, f64::NAN)
                };
                let sizing_predict = |x: f64| -> f64 {
                    match &sizing_loess {
                        Some(l) => l.predict(x),
                        None => s_slope * x + s_intercept,
                    }
                };
                let resid: Vec<f64> = ho_x
                    .iter()
                    .zip(&ho_y)
                    .map(|(x, y)| (y - sizing_predict(*x)).abs())
                    .collect();
                let p_rt_s = percentile(&resid, cfg.p_rt);
                Some(HoldoutSizing {
                    width: (p_rt_s * cfg.rt_window_multiplier).max(1.0),
                    n_sizing_train: tr_x.len(),
                    n_holdout: ho_x.len(),
                    resid_p_rt_s: p_rt_s,
                    resid_abs_median_s: percentile(&resid, 0.5),
                })
            }
        } else {
            None
        };

    let (w_rt, status): (Option<f64>, String) = match plan {
        WindowPlan::Unbounded => {
            warn!(
                n_train,
                min_anchors,
                "rt-im-train: fewer than two target anchors; using unbounded RT windows"
            );
            (None, INSUFFICIENT_ANCHORS_STATUS.to_string())
        }
        WindowPlan::Fixed(width) => {
            warn!(
                n_train,
                min_anchors, "rt-im-train: too few target anchors; using fallback fixed RT window"
            );
            (Some(width), "fallback_fixed".to_string())
        }
        WindowPlan::Calibrated => {
            let width = match &holdout_sizing {
                Some(h) => h.width,
                None => {
                    let resid: Vec<f64> = train_irt
                        .iter()
                        .zip(&train_rt)
                        .map(|(x, y)| (y - fit.predict(*x)).abs())
                        .collect();
                    (percentile(&resid, cfg.p_rt) * cfg.rt_window_multiplier).max(1.0)
                }
            };
            let status = if use_loess { "loess" } else { "linear" };
            (Some(width), status.to_string())
        }
    };

    // Optional adaptive window: local residual-percentile half-width per
    // calibrated-RT bin, so well-calibrated regions get a tight window (less
    // interference) and poorly-calibrated regions a wider one (more recall).
    // `None` keeps the single global `w_rt`. Empty bins fall back to `w_rt`.
    let adaptive: Option<(f64, f64, Vec<f64>)> = if cfg.adaptive_rt_window && n_train >= min_anchors
    {
        let cals: Vec<f64> = train_irt.iter().map(|x| fit.predict(*x)).collect();
        let resid: Vec<f64> = cals
            .iter()
            .zip(&train_rt)
            .map(|(c, y)| (y - c).abs())
            .collect();
        let rt_min = cals.iter().cloned().fold(f64::INFINITY, f64::min);
        let rt_max = cals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let nb = cfg.adaptive_rt_bins.max(1);
        if rt_max > rt_min {
            let span = rt_max - rt_min;
            let mut per_bin: Vec<Vec<f64>> = vec![Vec::new(); nb];
            for (c, r) in cals.iter().zip(&resid) {
                let frac = ((c - rt_min) / span).clamp(0.0, 0.999_999);
                per_bin[(frac * nb as f64) as usize].push(*r);
            }
            let lo_clamp = cfg.rt_window_min_s.max(0.0);
            let hi_clamp = cfg.fallback_rt_window_s.max(lo_clamp);
            let widths: Vec<f64> = per_bin
                .iter()
                .map(|rs| {
                    if rs.is_empty() {
                        w_rt.expect("adaptive RT windows require a calibrated global width")
                    } else {
                        (percentile(rs, cfg.p_rt) * cfg.rt_window_multiplier)
                            .clamp(lo_clamp, hi_clamp)
                    }
                })
                .collect();
            Some((rt_min, span, widths))
        } else {
            None
        }
    } else {
        None
    };

    // RT calibration-quality residuals over the training anchors (seconds):
    // signed median = residual bias, absolute median = typical accuracy, MAD =
    // spread. Diagnostic only (the RT window already derives from these
    // residuals); surfaced so a run's RT calibration can be judged good or biased.
    let residuals = if calibration_available {
        let signed: Vec<f64> = train_irt
            .iter()
            .zip(&train_rt)
            .map(|(x, y)| y - fit.predict(*x))
            .collect();
        let med = percentile(&signed, 0.5);
        let absres: Vec<f64> = signed.iter().map(|r| r.abs()).collect();
        let mad: Vec<f64> = signed.iter().map(|r| (r - med).abs()).collect();
        (med, percentile(&absres, 0.5), percentile(&mad, 0.5))
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };

    fit.w_rt = w_rt;
    fit.status = status;
    fit.holdout_sizing = holdout_sizing;
    fit.adaptive = adaptive;
    fit.residuals = residuals;
    Ok(fit)
}

/// Apply `fit` to every row of one library table and write the windows, the `cal.json` and
/// the windows' report. With `keep_windows`, also return the windows in the form `extract`
/// consumes (see [`RtWindows`]; `None` when the table holds a NaN bound).
fn write_windows(
    fit: &RtFit,
    p: &ApplyParams,
    lib_cid: Vec<u32>,
    lib_irt: Vec<f32>,
    t0: Instant,
    keep_windows: bool,
) -> Result<(Written, Option<RtWindows>)> {
    // Apply to every library candidate, streamed: each 65,536-row chunk of the window
    // table is computed and written before the next, through the same writer and the same
    // chunk sequence `write_table` uses, so the file is byte-identical to writing the seven
    // whole columns at once. Holding them whole was 76 bytes per candidate (three
    // `Option<f64>` ion-mobility columns among them), 15 GB at 203M rows, for a table that
    // is written once and never read back here.
    let cid = lib_cid;
    let irt = lib_irt;
    let n = cid.len();
    let mut n_nonfinite_irt = 0u64;
    let mut kept = keep_windows.then(|| RtWindowsBuilder::new(n));
    let windows = write_table_chunked_hashed(p.out_windows, n, |r| {
        let k = r.len();
        let (mut cid_c, mut cal_c, mut lo_c, mut hi_c) = (
            Vec::with_capacity(k),
            Vec::with_capacity(k),
            Vec::with_capacity(k),
            Vec::with_capacity(k),
        );
        for i in r {
            // A row whose library iRT is not finite has no calibrated RT, which is the
            // documented "search the whole gradient" sentinel rather than an arithmetic
            // accident. The parquet reader maps a null f32 to NaN, so one failed prediction
            // in an imported library reaches here (docs/31 F4).
            let usable_irt = (irt[i] as f64).is_finite();
            if !usable_irt {
                n_nonfinite_irt += 1;
            }
            let calibrated_rt =
                (fit.calibration_available && usable_irt).then(|| fit.predict(irt[i] as f64));
            let width = calibrated_rt.map(|cal| match &fit.adaptive {
                Some((rt_min, span, widths)) => {
                    let nb = widths.len();
                    let frac = ((cal - rt_min) / span).clamp(0.0, 0.999_999);
                    widths[(frac * nb as f64) as usize]
                }
                None => fit
                    .w_rt
                    .expect("available RT calibration requires a bounded window"),
            });
            let (cal, lo, hi) = candidate_window(calibrated_rt, width);
            if let Some(b) = kept.as_mut() {
                b.row(cid[i], cal, lo, hi);
            }
            cid_c.push(cid[i]);
            cal_c.push(cal);
            lo_c.push(lo);
            hi_c.push(hi);
        }
        Ok(vec![
            Col::U32("candidate_id".into(), cid_c),
            Col::F64("rt_pred_cal".into(), cal_c),
            Col::F64("rt_lo".into(), lo_c),
            Col::F64("rt_hi".into(), hi_c),
            Col::OptF64("im_pred_cal".into(), vec![None; k]),
            Col::OptF64("im_lo".into(), vec![None; k]),
            Col::OptF64("im_hi".into(), vec![None; k]),
        ])
    })?;
    let rows = windows.rows;

    let method = if !fit.calibration_available {
        "unavailable"
    } else if fit.use_loess {
        "loess"
    } else {
        "linear"
    };
    let slope_report = fit.calibration_available.then_some(fit.slope);
    let intercept_report = fit.calibration_available.then_some(fit.intercept);
    let (rt_residual_median_s, rt_residual_abs_median_s, rt_residual_mad_s) = fit.residuals;
    let holdout_sizing = &fit.holdout_sizing;
    let holdout_frac = fit.holdout_frac;
    let w_rt = fit.w_rt;
    let status = fit.status.as_str();
    let n_train = fit.n_train;

    // cal.json
    mumdia_io::json::write_json(
        p.out_cal,
        &json!({
            "method": method,
            "slope": slope_report,
            "intercept": intercept_report,
            "w_rt": w_rt,
            "p_rt": p.cfg.p_rt,
            "multiplier": p.cfg.rt_window_multiplier,
            "n_train": n_train,
            "calibration_status": status,
            // Window sizing provenance. "holdout" means w_rt came from held-out
            // residuals; "holdout_fallback_in_sample" means it was requested but
            // an anchor-count guard fell back; "in_sample" is the historical
            // behavior. The holdout_* fields are null unless sizing ran held-out.
            "w_rt_sizing": match (holdout_sizing, holdout_frac > 0.0) {
                (Some(_), _) => "holdout",
                (None, true) => "holdout_fallback_in_sample",
                (None, false) => "in_sample",
            },
            "window_holdout_frac": holdout_frac,
            "n_sizing_train": holdout_sizing.as_ref().map(|h| h.n_sizing_train),
            "n_holdout": holdout_sizing.as_ref().map(|h| h.n_holdout),
            "holdout_resid_p_rt_s": holdout_sizing.as_ref().map(|h| h.resid_p_rt_s),
            "holdout_resid_abs_median_s": holdout_sizing.as_ref().map(|h| h.resid_abs_median_s),
            // Calibration-quality diagnostics (post-fit RT residuals, seconds).
            "rt_residual_median_s": rt_residual_median_s,
            "rt_residual_abs_median_s": rt_residual_abs_median_s,
            "rt_residual_mad_s": rt_residual_mad_s,
        }),
    )?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("n_train".to_string(), json!(n_train));
    stats.insert("w_rt".to_string(), json!(w_rt));
    stats.insert("calibration_status".to_string(), json!(status));
    stats.insert(
        "candidates_without_finite_irt".to_string(),
        json!(n_nonfinite_irt),
    );
    let report = ArtifactReport {
        logical_name: artifact::RUN_WINDOWS.0.to_string(),
        schema_name: artifact::RUN_WINDOWS.0.to_string(),
        schema_version: artifact::RUN_WINDOWS.1,
        stage: "rt-im-train".to_string(),
        rows,
        // Computed while the table was written.
        content_hash: windows.content_hash,
        params: json!({"q_train": p.cfg.q_train, "p_rt": p.cfg.p_rt, "method": format!("{:?}", p.cfg.calibration_method)}),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    };
    report.write_for(p.out_windows)?;

    if n_nonfinite_irt > 0 {
        tracing::warn!(
            candidates = n_nonfinite_irt,
            of = rows,
            "rt-im-train: these candidates have no finite library iRT, so they get the \
             unbounded RT window rather than a calibrated one; a null predicted_irt reads \
             as NaN (docs/31 F4)"
        );
    }
    info!(
        rows,
        w_rt = ?w_rt,
        status,
        without_finite_irt = n_nonfinite_irt,
        elapsed_ms = elapsed,
        "rt-im-train: done"
    );
    Ok((
        report.written(),
        kept.and_then(|b| b.finish(p.library_precursors, p.out_windows)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::write_table;

    #[test]
    fn anchor_vectors_are_sorted_by_base_peptide_id() {
        let mut first = HashMap::new();
        first.insert(30, (9.0, 3.0, 300.0));
        first.insert(10, (7.0, 1.0, 100.0));
        first.insert(20, (8.0, 2.0, 200.0));

        let mut shuffled = HashMap::new();
        shuffled.insert(20, (8.0, 2.0, 200.0));
        shuffled.insert(30, (9.0, 3.0, 300.0));
        shuffled.insert(10, (7.0, 1.0, 100.0));

        let expected = (
            vec![10, 20, 30],
            vec![1.0, 2.0, 3.0],
            vec![100.0, 200.0, 300.0],
        );
        assert_eq!(sorted_anchor_vectors(first), expected);
        assert_eq!(sorted_anchor_vectors(shuffled), expected);
    }

    /// The direct index must answer exactly what the hash join answered, for every id a
    /// seed table can carry, including ids past the library; and a layout that is not the
    /// row-aligned range must still get the hash join, with its last-row-wins meaning.
    #[test]
    fn the_irt_join_answers_what_the_hash_join_answered() {
        let reference = |cid: &[u32], irt: &[f32]| -> HashMap<u32, f64> {
            let mut m = HashMap::new();
            for (&c, &v) in cid.iter().zip(irt) {
                m.insert(c, v as f64);
            }
            m
        };
        let irt: Vec<f32> = vec![1.5, f32::NAN, -3.25, 1e-40, 7.0];
        let cases: Vec<Vec<u32>> = vec![
            vec![0, 1, 2, 3, 4],      // row-aligned: the direct index
            vec![0, 1, 2, 2, 4],      // a duplicate: the hash join, last row wins
            vec![4, 3, 2, 1, 0],      // permuted
            vec![10, 11, 12, 13, 14], // offset ids
        ];
        for cid in &cases {
            let join = IrtJoin::new(cid, &irt);
            let aligned = cid.iter().enumerate().all(|(i, &c)| c as usize == i);
            assert_eq!(matches!(join, IrtJoin::Dense(_)), aligned, "{cid:?}");
            let want = reference(cid, &irt);
            for c in 0..20u32 {
                let got = join.get(c).map(f64::to_bits);
                assert_eq!(got, want.get(&c).map(|v| v.to_bits()), "{cid:?} id {c}");
            }
        }
        assert_eq!(IrtJoin::Unused.get(0), None);
        // An empty library joins nothing.
        assert_eq!(IrtJoin::new(&[], &[]).get(0), None);
    }

    /// A per-process scratch path; tests in this module run concurrently.
    fn scratch(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("mumdia_rt_im_train_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_str().unwrap().to_string()
    }

    /// Bit patterns of a window set, for exact comparison (NaN included).
    fn bits(w: &RtWindows) -> Vec<(u64, u64, u64)> {
        (0..w.len())
            .map(|c| {
                (
                    w.rt_cal[c].to_bits(),
                    w.rt_lo[c].to_bits(),
                    w.rt_hi[c].to_bits(),
                )
            })
            .collect()
    }

    /// The windows handed to extract in memory are the arrays extract builds from the file
    /// the same call wrote, bit for bit, under each window plan: calibrated LOESS, adaptive
    /// per-bin widths, and the unbounded plan of a run without anchors. The library has
    /// candidates with a non-finite iRT, which take the unbounded sentinel.
    #[test]
    fn the_windows_kept_in_memory_are_the_windows_extract_reads_back() {
        let n = 3_000usize;
        let prec = scratch("inmem_prec.parquet");
        let irt: Vec<f32> = (0..n)
            .map(|i| {
                if i % 97 == 5 {
                    f32::NAN
                } else {
                    i as f32 * 0.05
                }
            })
            .collect();
        write_table(
            &prec,
            vec![
                Col::U32("candidate_id".into(), (0..n as u32).collect()),
                Col::F32("predicted_irt".into(), irt.clone()),
            ],
        )
        .unwrap();
        // One confident target anchor per base peptide on a curved gradient, plus decoys and
        // unconfident rows that must not anchor.
        let seed_rows: Vec<usize> = (0..n).step_by(7).filter(|i| irt[*i].is_finite()).collect();
        let seed_table = |path: &str, rows: &[usize]| {
            let m = rows.len();
            write_table(
                path,
                vec![
                    Col::U32(
                        "candidate_id".into(),
                        rows.iter().map(|&i| i as u32).collect(),
                    ),
                    Col::U32(
                        "base_peptide_id".into(),
                        rows.iter().map(|&i| i as u32).collect(),
                    ),
                    Col::F64(
                        "spectrum_q".into(),
                        rows.iter()
                            .map(|&i| if i % 5 == 0 { 0.5 } else { 0.001 })
                            .collect(),
                    ),
                    Col::F64("score".into(), (0..m).map(|k| 1.0 + k as f64).collect()),
                    Col::F64(
                        "observed_rt".into(),
                        rows.iter()
                            .map(|&i| {
                                let x = irt[i] as f64;
                                60.0 + 30.0 * x + 0.4 * x * x + ((i * 37) % 11) as f64
                            })
                            .collect(),
                    ),
                    Col::Str(
                        "label".into(),
                        rows.iter()
                            .map(|&i| if i % 3 == 0 { "decoy" } else { "target" }.to_string())
                            .collect(),
                    ),
                ],
            )
            .unwrap();
        };
        let seed = scratch("inmem_seed.parquet");
        seed_table(&seed, &seed_rows);
        let empty_seed = scratch("inmem_seed_empty.parquet");
        seed_table(&empty_seed, &[]);

        let adaptive = RtImTrainConfig {
            adaptive_rt_window: true,
            ..RtImTrainConfig::default()
        };
        for (tag, cfg, seed_path, want_status) in [
            ("loess", RtImTrainConfig::default(), &seed, "loess"),
            ("adaptive", adaptive, &seed, "loess"),
            (
                "unbounded",
                RtImTrainConfig::default(),
                &empty_seed,
                INSUFFICIENT_ANCHORS_STATUS,
            ),
        ] {
            let windows = scratch(&format!("inmem_windows_{tag}.parquet"));
            let cal = scratch(&format!("inmem_cal_{tag}.json"));
            let params = || RtImTrainParams {
                precursor_span: None,
                seed_psms: seed_path,
                library_precursors: &prec,
                out_windows: &windows,
                out_cal: &cal,
                cfg: &cfg,
                config_hash: "test",
                anchor_irt_from_seed: false,
            };
            let (written, kept) = run_in_memory(params()).unwrap();
            assert_eq!(written.rows, n as u64);
            let kept = kept.expect("no NaN bound, so the windows are kept");
            let status: serde_json::Value = mumdia_io::json::read_json(&cal).unwrap();
            assert_eq!(status["calibration_status"], want_status, "{tag}");
            let from_file = crate::stages::extract::read_run_windows(&windows, n).unwrap();
            assert_eq!(
                bits(&kept),
                bits(&from_file),
                "{tag}: in memory != read back"
            );
            // Handed over only to the extract of the library and the file they were
            // fitted for; a table of the same size elsewhere, another run_windows path or
            // another candidate count reads the file instead.
            assert_eq!(kept.mismatch(&prec, &windows, n), None, "{tag}");
            assert!(kept.mismatch("other_prec.parquet", &windows, n).is_some());
            assert!(kept.mismatch(&prec, "other_windows.parquet", n).is_some());
            assert!(kept.mismatch(&prec, &windows, n + 1).is_some());
            assert!(from_file.mismatch(&prec, &windows, n).is_some());
            // `run` writes the same file as `run_in_memory`.
            let bytes = std::fs::read(&windows).unwrap();
            assert_eq!(run(params()).unwrap(), n as u64);
            assert_eq!(
                std::fs::read(&windows).unwrap(),
                bytes,
                "{tag}: run vs run_in_memory"
            );
        }
    }

    /// The builder is the scatter extract applies: last row wins for a repeated id, ids past
    /// the library are ignored, and a NaN bound withholds the in-memory form so extract
    /// reads the file and names the row.
    #[test]
    fn the_windows_builder_is_the_scatter_extract_applies() {
        let rows: Vec<(u32, f64, f64, f64)> = vec![
            (2, 10.0, 5.0, 15.0),
            (0, f64::NAN, f64::NEG_INFINITY, f64::INFINITY),
            (2, 11.0, 6.0, 16.0), // repeated: this one wins
            (9, 1.0, 0.0, 2.0),   // past the library
        ];
        let path = scratch("builder_windows.parquet");
        let write = |path: &str, rows: &[(u32, f64, f64, f64)]| {
            write_table(
                path,
                vec![
                    Col::U32("candidate_id".into(), rows.iter().map(|r| r.0).collect()),
                    Col::F64("rt_pred_cal".into(), rows.iter().map(|r| r.1).collect()),
                    Col::F64("rt_lo".into(), rows.iter().map(|r| r.2).collect()),
                    Col::F64("rt_hi".into(), rows.iter().map(|r| r.3).collect()),
                ],
            )
            .unwrap();
        };
        write(&path, &rows);
        let mut b = RtWindowsBuilder::new(4);
        for &(c, cal, lo, hi) in &rows {
            b.row(c, cal, lo, hi);
        }
        let kept = b.finish("lib", "rw").expect("no NaN bound");
        let read = crate::stages::extract::read_run_windows(&path, 4).unwrap();
        assert_eq!(bits(&kept), bits(&read));
        assert_eq!(kept.rt_cal[2], 11.0);
        assert!(kept.rt_cal[1].is_nan() && kept.rt_lo[3] == f64::NEG_INFINITY);

        write(&path, &[(1u32, 3.0, f64::NAN, 4.0)]);
        let mut b = RtWindowsBuilder::new(4);
        b.row(1, 3.0, f64::NAN, 4.0);
        assert!(
            b.finish("lib", "rw").is_none(),
            "a NaN bound must not be handed over"
        );
        let err = crate::stages::extract::read_run_windows(&path, 4)
            .err()
            .unwrap();
        assert!(err.to_string().contains("NaN RT bound"), "{err}");
        // A NaN bound on a row past the library is not checked by extract either.
        let mut b = RtWindowsBuilder::new(1);
        b.row(1, 3.0, f64::NAN, 4.0);
        assert!(b.finish("lib", "rw").is_some());
    }

    #[test]
    fn holdout_split_is_deterministic_and_fraction_shaped() {
        // frac 0.0 holds out nothing; frac 0.9 (the cap) holds out ids 0..899 mod 1000.
        assert!(!is_holdout(0, 0.0));
        assert!(!is_holdout(999, 0.0));
        // The documented rule, verbatim: base_peptide_id % 1000 < round(frac*1000).
        // deeplc_finetune.py pins these same four cases; keep them in sync.
        assert!(is_holdout(299, 0.3));
        assert!(!is_holdout(300, 0.3));
        assert!(is_holdout(1_000_299, 0.3));
        assert!(!is_holdout(1_000_300, 0.3));
        // Empirical fraction over a contiguous id range is exactly frac.
        let n_held = (0u32..1_000_000).filter(|id| is_holdout(*id, 0.3)).count();
        assert_eq!(n_held, 300_000);
    }

    #[test]
    fn sparse_anchor_policy_is_unbounded_only_below_two() {
        assert_eq!(window_plan(0, 50, 120.0), WindowPlan::Unbounded);
        assert_eq!(window_plan(1, 50, 120.0), WindowPlan::Unbounded);
        assert_eq!(window_plan(2, 50, 120.0), WindowPlan::Fixed(120.0));
        assert_eq!(window_plan(49, 50, 120.0), WindowPlan::Fixed(120.0));
        assert_eq!(window_plan(50, 50, 120.0), WindowPlan::Calibrated);
        assert_eq!(
            INSUFFICIENT_ANCHORS_STATUS,
            "insufficient_anchors_unbounded"
        );
    }

    #[test]
    fn unavailable_calibration_emits_unbounded_window_and_nan_prediction() {
        let (cal, lo, hi) = candidate_window(None, None);
        assert!(cal.is_nan());
        assert_eq!(lo, f64::NEG_INFINITY);
        assert_eq!(hi, f64::INFINITY);

        assert_eq!(
            candidate_window(Some(300.0), Some(120.0)),
            (300.0, 180.0, 420.0)
        );
    }

    /// A seed of `n` confident target anchors (plus a decoy and an unconfident row, which
    /// the fit must skip) and a library of `n + 5` rows, both with iRT columns.
    fn anchors_and_library(dir: &std::path::Path, n: usize) -> (String, String) {
        let seed = dir.join("seed.parquet").to_str().unwrap().to_string();
        let lib = dir.join("lib.parquet").to_str().unwrap().to_string();
        let m = n + 2;
        let irt: Vec<f32> = (0..m).map(|i| (i as f32) * 0.7 + 3.0).collect();
        let rt: Vec<f64> = (0..m)
            .map(|i| 100.0 + 12.0 * (i as f64 * 0.7 + 3.0) + ((i * 37 % 11) as f64 - 5.0) * 4.0)
            .collect();
        let mut label = vec!["target".to_string(); m];
        label[n] = "decoy".to_string();
        let mut q = vec![0.001; m];
        q[n + 1] = 0.5;
        write_table(
            &seed,
            vec![
                Col::U32("candidate_id".into(), (0..m as u32).collect()),
                Col::U32("base_peptide_id".into(), (0..m as u32).collect()),
                Col::F64("spectrum_q".into(), q),
                Col::F64("score".into(), (0..m).map(|i| 10.0 + i as f64).collect()),
                Col::F64("observed_rt".into(), rt),
                Col::Str("label".into(), label),
                Col::F32("predicted_irt".into(), irt),
            ],
        )
        .unwrap();
        write_table(
            &lib,
            vec![
                Col::U32("candidate_id".into(), (0..(n + 5) as u32).collect()),
                Col::F32(
                    "predicted_irt".into(),
                    (0..n + 5)
                        .map(|i| if i == 3 { f32::NAN } else { i as f32 * 0.9 })
                        .collect(),
                ),
            ],
        )
        .unwrap();
        (seed, lib)
    }

    #[test]
    fn one_fit_applied_writes_what_the_whole_stage_writes() {
        // The grouped path fits once per run and applies the fit to every band; each band
        // used to run the whole stage. Same windows bytes and the same cal.json, under the
        // default, the held-out sizing, the adaptive window and a linear fit, and for too
        // few anchors.
        let dir = std::env::temp_dir().join(format!("mumdia_rt_fit_apply_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let holdout = RtImTrainConfig {
            window_holdout_frac: 0.3,
            ..Default::default()
        };
        let adaptive = RtImTrainConfig {
            adaptive_rt_window: true,
            ..Default::default()
        };
        let linear = RtImTrainConfig {
            calibration_method: CalibrationMethod::Linear,
            ..Default::default()
        };
        for (tag, cfg, n) in [
            ("default", RtImTrainConfig::default(), 400usize),
            ("holdout", holdout, 400),
            ("adaptive", adaptive, 400),
            ("linear", linear, 400),
            ("few", RtImTrainConfig::default(), 20),
            ("one", RtImTrainConfig::default(), 1),
        ] {
            let sub = dir.join(tag);
            std::fs::create_dir_all(&sub).unwrap();
            let (seed, lib) = anchors_and_library(&sub, n);
            let out = |name: &str| sub.join(name).to_str().unwrap().to_string();
            run(RtImTrainParams {
                precursor_span: None,
                seed_psms: &seed,
                library_precursors: &lib,
                out_windows: &out("w_run.parquet"),
                out_cal: &out("cal_run.json"),
                cfg: &cfg,
                config_hash: "h",
                anchor_irt_from_seed: true,
            })
            .unwrap();
            let fit = fit_from_seed(&seed, &cfg).unwrap();
            apply(
                &fit,
                &ApplyParams {
                    precursor_span: None,
                    library_precursors: &lib,
                    out_windows: &out("w_apply.parquet"),
                    out_cal: &out("cal_apply.json"),
                    cfg: &cfg,
                    config_hash: "h",
                },
            )
            .unwrap();
            assert_eq!(
                std::fs::read(out("w_run.parquet")).unwrap(),
                std::fs::read(out("w_apply.parquet")).unwrap(),
                "{tag}: the windows differ"
            );
            assert_eq!(
                std::fs::read(out("cal_run.json")).unwrap(),
                std::fs::read(out("cal_apply.json")).unwrap(),
                "{tag}: cal.json differs"
            );
            assert_eq!(
                fit.n_train(),
                n,
                "{tag}: the decoy and the unconfident row are not anchors"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
