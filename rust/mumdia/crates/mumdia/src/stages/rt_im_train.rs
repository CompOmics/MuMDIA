//! Stage B `mumdia rt-im-train`: per-run RT calibration and windows
//! (docs/08_rt_im_train.md). Calibrates the run-independent predicted iRT to
//! observed RT from confident seed PSMs, then sets a per-candidate RT window from
//! the residuals. When the seed carries `observed_im` (spectra with 1/K0) and the library
//! `predicted_im`, the same confident targets calibrate ion mobility: a per-charge linear
//! map in CCS space and a held-out half-window (`fit_im`). Otherwise the IM columns are
//! null and `cal.json` has no IM block. The sidecars are not re-run here.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{CalibrationMethod, RtImTrainConfig};
use mumdia_core::schema::artifact;
use mumdia_io::report::{ArtifactReport, Written};
use mumdia_io::table::{write_table_chunked_hashed, Col, TableFile};
use serde_json::json;
use tracing::{info, warn};

use crate::calibrate::{linear_fit, percentile, x_domain_inliers, Loess};

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
    /// The second pass of match-between-runs raised `cfg.im_window_min` to this run's
    /// first-pass `w_im`; recorded in `cal.json` as `im_window_floor`. `None` everywhere else.
    pub im_window_floor: Option<ImWindowFloor<'a>>,
    /// The seed's own confident cut (`search_seed.fdr_seed`): an ion-mobility anchor must be
    /// at or below it as well as below `cfg.q_train`. Until `im_window_sizing =
    /// seed_decoy_corrected` only those targets carried `observed_im`, so a relaxed anchor
    /// cut (`anchor_q_ladder`) never added IM anchors; the sizing gives weaker rows an
    /// `observed_im` for its window estimate, and this keeps them out of the calibration.
    /// `None`: `cfg.q_train` alone.
    pub im_anchor_q: Option<f64>,
}

/// Where a run's 1/K0 window floor came from, when the second pass raised it
/// (`run_experiment::search_second_pass`): the configured `im_window_min`, the first pass's
/// `w_im` and the `cal.json` it was read from. The floor applied is the larger of the two,
/// and it is what `cfg.im_window_min` already holds.
#[derive(Clone, Copy, Debug)]
pub struct ImWindowFloor<'a> {
    pub configured: f64,
    pub first_pass_w_im: f64,
    pub source: &'a str,
}

/// `calibration_status` of a run with fewer than two target anchors: no iRT -> RT mapping,
/// so every candidate is extracted over the whole gradient (`rt_health` reports it).
pub(crate) const INSUFFICIENT_ANCHORS_STATUS: &str = "insufficient_anchors_unbounded";
/// `calibration_status` of a run with too few anchors for a residual-sized window: the
/// broad `fallback_rt_window_s` around a fit on those few anchors.
pub(crate) const FALLBACK_FIXED_STATUS: &str = "fallback_fixed";

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

/// Whether a fit on `n_train` anchors leaves the RT windows unbounded: zero or one point
/// cannot define an iRT -> RT mapping across the gradient. The one rule [`window_plan`] and
/// the grouped run's refusal of unbounded windows (`groups.allow_unbounded_rt`,
/// `run_groups`) both read, so the two cannot disagree.
pub(crate) fn windows_unbounded(n_train: usize) -> bool {
    n_train < 2
}

fn window_plan(n_train: usize, min_anchors: usize, fallback_width: f64) -> WindowPlan {
    if windows_unbounded(n_train) {
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

/// Fraction of the ion-mobility anchors' base peptides held out to size `w_im`
/// ([`is_holdout`]). The IM window is always sized on held-out anchors: in-sample residuals
/// are optimistic and can rank two models backwards (docs/08_rt_im_train.md section 4). It
/// is a constant rather than `window_holdout_frac`, which also drives the DeepLC fine-tune
/// and multi-head exclusion and is exclusive with `adaptive_rt_window`.
const IM_HOLDOUT_FRAC: f64 = 0.3;

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

/// The `candidate_id -> library row` join the anchors read their library values through:
/// the predicted iRT, and on a run with ion mobility the predicted 1/K0 from the same row.
///
/// It used to be a `HashMap<u32, f64>` of the iRT over every library row, built
/// unconditionally: about 4.5 GB of transient on an unbanded 203M-row library, built even
/// when `anchor_irt_from_seed` meant it was never read. A library as the engine writes and
/// loads it carries `candidate_id` as the row-aligned range `0..n` (`index.rs` refuses
/// anything else), so the join is the id itself. The hash table remains for any other id
/// layout, and it keeps exactly the old meaning there: the LAST row with a given id wins,
/// as repeated `insert`s did.
enum CidJoin {
    /// `anchor_irt_from_seed`: the seed table carries the iRT, nothing is joined.
    Unused,
    /// `candidate_id[i] == i` for every one of these many rows: the id is the row.
    Dense(usize),
    /// Any other layout: the historical hash join, to the row.
    Map(HashMap<u32, usize>),
}

impl CidJoin {
    fn new(cid: &[u32]) -> CidJoin {
        if cid.iter().enumerate().all(|(i, &c)| c as usize == i) {
            return CidJoin::Dense(cid.len());
        }
        let mut m: HashMap<u32, usize> = HashMap::with_capacity(cid.len());
        for (row, &c) in cid.iter().enumerate() {
            m.insert(c, row);
        }
        CidJoin::Map(m)
    }

    /// The library row of candidate `c`; `None` when the library has no such candidate.
    #[inline]
    fn row(&self, c: u32) -> Option<usize> {
        match self {
            CidJoin::Unused => None,
            CidJoin::Dense(n) => ((c as usize) < *n).then_some(c as usize),
            CidJoin::Map(m) => m.get(&c).copied(),
        }
    }

    /// The library iRT of candidate `c`, widened exactly as the hash join stored it.
    #[inline]
    fn irt(&self, c: u32, irt: &[f32]) -> Option<f64> {
        self.row(c).map(|r| irt[r] as f64)
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

/// Composition features of a peptidoform and charge for the RT correction: intercept,
/// charge 3, charge >= 4, length, internal K/R, His, Met-ox. Modifications are written in
/// square brackets (MuMDIA) or round brackets (UniMod); only residues count toward length.
pub(crate) fn composition_features(peptidoform: &str, charge: i32) -> [f64; 7] {
    let mut seq: Vec<u8> = Vec::with_capacity(peptidoform.len());
    let mut depth = 0i32;
    for &b in peptidoform.as_bytes() {
        match b {
            b'[' | b'(' => depth += 1,
            b']' | b')' => depth -= 1,
            _ if depth == 0 && b.is_ascii_uppercase() => seq.push(b),
            _ => {}
        }
    }
    let n = seq.len();
    let internal_kr = seq
        .iter()
        .take(n.saturating_sub(1))
        .filter(|&&c| c == b'K' || c == b'R')
        .count();
    let his = seq.iter().filter(|&&c| c == b'H').count();
    let mox = peptidoform.matches("Oxidation").count() + peptidoform.matches("UniMod:35").count();
    [
        1.0,
        f64::from(u8::from(charge == 3)),
        f64::from(u8::from(charge >= 4)),
        n as f64,
        internal_kr as f64,
        his as f64,
        mox as f64,
    ]
}

/// Fewest anchors that must carry a composition term (nonzero feature) for it to be fitted.
const MIN_ANCHORS_PER_TERM: usize = 20;

/// Zero every non-intercept column that fewer than `min` rows carry, so its coefficient
/// stays at 0 under the ridge in [`huber_fit`].
fn drop_sparse_columns(mut x: Vec<[f64; 7]>, min: usize) -> Vec<[f64; 7]> {
    for c in 1..7 {
        let n = x.iter().filter(|r| r[c] != 0.0).count();
        if n < min {
            for r in x.iter_mut() {
                r[c] = 0.0;
            }
        }
    }
    x
}

/// Huber regression (k = 1.345 on a MAD scale, iteratively reweighted least squares) of
/// `y` on the rows of `x`. `None` when the normal equations are singular.
fn huber_fit(x: &[[f64; 7]], y: &[f64]) -> Option<Vec<f64>> {
    const P: usize = 7;
    let n = y.len();
    if n < 3 * P {
        return None;
    }
    let mut w = vec![1.0f64; n];
    let mut beta = vec![0.0f64; P];
    for _ in 0..30 {
        // weighted normal equations with a tiny ridge for unused columns (e.g. no charge 4)
        let mut a = [[0.0f64; P]; P];
        let mut b = [0.0f64; P];
        for i in 0..n {
            for r in 0..P {
                b[r] += w[i] * x[i][r] * y[i];
                for c in 0..P {
                    a[r][c] += w[i] * x[i][r] * x[i][c];
                }
            }
        }
        for (r, row) in a.iter_mut().enumerate() {
            row[r] += 1e-6;
        }
        let sol = solve(a, b)?;
        let resid: Vec<f64> = (0..n)
            .map(|i| y[i] - (0..P).map(|c| x[i][c] * sol[c]).sum::<f64>())
            .collect();
        let abs: Vec<f64> = resid.iter().map(|r| r.abs()).collect();
        let s = 1.4826 * percentile(&abs, 0.5);
        let done = sol.iter().zip(&beta).all(|(a, b)| (a - b).abs() < 1e-6);
        beta = sol;
        if !(s.is_finite() && s > 0.0) || done {
            break;
        }
        for (wi, r) in w.iter_mut().zip(&resid) {
            let k = 1.345 * s;
            *wi = if r.abs() <= k { 1.0 } else { k / r.abs() };
        }
    }
    Some(beta)
}

/// Gaussian elimination with partial pivoting.
fn solve(mut a: [[f64; 7]; 7], mut b: [f64; 7]) -> Option<Vec<f64>> {
    const P: usize = 7;
    for col in 0..P {
        let piv = (col..P).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..P {
            let f = a[r][col] / a[col][col];
            let pivot_row = a[col];
            for (c, v) in a[r].iter_mut().enumerate().skip(col) {
                *v -= f * pivot_row[c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0f64; P];
    for r in (0..P).rev() {
        let s: f64 = (r + 1..P).map(|c| a[r][c] * x[c]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

/// Early (`left`) and late (`right`) half-widths of the widen-only local RT window on an
/// ascending grid of calibrated RT; linear in between, the end values beyond the grid.
#[derive(Clone, Debug)]
struct LocalWindow {
    grid: Vec<f64>,
    left: Vec<f64>,
    right: Vec<f64>,
}

impl LocalWindow {
    fn at(&self, cal: f64) -> (f64, f64) {
        let g = &self.grid;
        let last = g.len() - 1;
        if cal <= g[0] {
            return (self.left[0], self.right[0]);
        }
        if cal >= g[last] {
            return (self.left[last], self.right[last]);
        }
        let j = g.partition_point(|&x| x < cal);
        let t = (cal - g[j - 1]) / (g[j] - g[j - 1]).max(1e-12);
        (
            self.left[j - 1] + t * (self.left[j] - self.left[j - 1]),
            self.right[j - 1] + t * (self.right[j] - self.right[j - 1]),
        )
    }

    /// Fit from the anchors' calibrated RT and signed residuals (observed - calibrated):
    /// on a grid over the anchors' calibrated range, the `k` anchors nearest each point give
    /// the `(1 - p_rt) / 2` and `(1 + p_rt) / 2` residual quantiles, times `mult`, floored at
    /// the global `w_rt` and capped at `cap`.
    fn fit(
        cal: &[f64],
        resid: &[f64],
        k: usize,
        p_rt: f64,
        mult: f64,
        w_rt: f64,
        cap: f64,
    ) -> Option<Self> {
        const GRID: usize = 200;
        let n = cal.len();
        if n < 2 {
            return None;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| cal[a].total_cmp(&cal[b]).then(a.cmp(&b)));
        let c: Vec<f64> = order.iter().map(|&i| cal[i]).collect();
        let r: Vec<f64> = order.iter().map(|&i| resid[i]).collect();
        let (lo, hi) = (c[0], c[n - 1]);
        if hi <= lo {
            return None;
        }
        let k = k.clamp(2, n);
        let (q_lo, q_hi) = ((1.0 - p_rt) / 2.0, (1.0 + p_rt) / 2.0);
        let cap = cap.max(w_rt);
        let mut grid = Vec::with_capacity(GRID);
        let mut left = Vec::with_capacity(GRID);
        let mut right = Vec::with_capacity(GRID);
        for gi in 0..GRID {
            let x = lo + (hi - lo) * gi as f64 / (GRID - 1) as f64;
            // The k anchors nearest x: grow [a, b) outward from the insertion point.
            let mut a = c.partition_point(|&v| v < x);
            let mut b = a;
            while b - a < k {
                let take_left = if a == 0 {
                    false
                } else if b == n {
                    true
                } else {
                    (x - c[a - 1]) <= (c[b] - x)
                };
                if take_left {
                    a -= 1;
                } else {
                    b += 1;
                }
            }
            let near = &r[a..b];
            grid.push(x);
            left.push((-percentile(near, q_lo) * mult).max(w_rt).min(cap));
            right.push((percentile(near, q_hi) * mult).max(w_rt).min(cap));
        }
        Some(LocalWindow { grid, left, right })
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
    /// Widen-only local window (`local_window_anchors`): the early and late half-widths on a
    /// grid of calibrated RT.
    local: Option<LocalWindow>,
    /// Composition correction coefficients (`composition_correction`), over
    /// [`composition_features`].
    composition: Option<Vec<f64>>,
    /// Signed median, absolute median and MAD of the in-sample residuals (seconds).
    residuals: (f64, f64, f64),
    /// The ion-mobility calibration, `None` without one.
    im: Option<ImCal>,
    /// The seed carries `observed_im` (a run whose spectra have 1/K0), so `cal.json` records
    /// the IM calibration, or that there is none. A 3D run records none of it, and its
    /// `cal.json` is what it was before IM calibration existed.
    im_reported: bool,
    /// Anchors `anchor_x_filter` removed before the fit; recorded in `cal.json` only when
    /// above 0, so a run without removals writes the `cal.json` it always wrote.
    n_rt_outliers_removed: usize,
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
///
/// On a run with an ion-mobility calibration the same holds for the 1/K0 window and the
/// calibrated 1/K0 of every candidate (`im_lo`, `im_hi`, `im_pred_cal`): `-inf`/`+inf` and
/// `None` for a candidate without one. They are `None` as a whole when no row carries a
/// 1/K0 window, which is every 3D run, so a 3D extract holds nothing for them.
pub struct RtWindows {
    pub(crate) rt_cal: Vec<f64>,
    pub(crate) rt_lo: Vec<f64>,
    pub(crate) rt_hi: Vec<f64>,
    /// Per-candidate 1/K0 window bounds (`run_windows.im_lo`/`im_hi`) at the width the
    /// peaks' 1/K0 is stored at, `-inf`/`+inf` for a candidate without one; `None` when no
    /// row has one.
    pub(crate) im_lo: Option<Vec<f32>>,
    pub(crate) im_hi: Option<Vec<f32>>,
    /// Per-candidate calibrated 1/K0 (`run_windows.im_pred_cal`, psms_extracted v4
    /// `im_pred_cal`); `None` exactly when `im_lo` is.
    pub(crate) im_cal: Option<Vec<Option<f64>>>,
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

    /// Whether candidate `c` may take a peak at retention time `rt` and, when `im` is
    /// given, at 1/K0 `im`. The one window test of extract: every probe, the two-pass
    /// arbitration, the demix and the promoted alternatives go through it. `im` is `None`
    /// when the 1/K0 gate is off (`extract.im_gate`) or the scan carries no 1/K0, and the
    /// test is then the retention-time test alone, `!(rt < rt_lo || rt > rt_hi)`, as it
    /// always was. A candidate without a 1/K0 window holds `-inf`/`+inf` and is never
    /// gated, as an unbounded RT window is not.
    #[inline]
    pub(crate) fn admits(&self, c: usize, rt: f64, im: Option<f32>) -> bool {
        if rt < self.rt_lo[c] || rt > self.rt_hi[c] {
            return false;
        }
        match (im, &self.im_lo, &self.im_hi) {
            (Some(x), Some(lo), Some(hi)) => !(x < lo[c] || x > hi[c]),
            _ => true,
        }
    }

    /// [`RtWindows::admits`] with the RT window widened by `pad` seconds on each side; the
    /// 1/K0 test is unchanged. Extract's trace pad (`extract::TracePad`) reads it.
    pub(crate) fn admits_padded(&self, c: usize, rt: f64, im: Option<f32>, pad: f64) -> bool {
        if rt < self.rt_lo[c] - pad || rt > self.rt_hi[c] + pad {
            return false;
        }
        match (im, &self.im_lo, &self.im_hi) {
            (Some(x), Some(lo), Some(hi)) => !(x < lo[c] || x > hi[c]),
            _ => true,
        }
    }

    /// Whether any candidate has a 1/K0 window, which `extract.im_gate = auto` needs.
    pub(crate) fn has_im(&self) -> bool {
        self.im_lo.is_some()
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
    /// A row with a NaN `rt_lo` or `rt_hi`, or a NaN 1/K0 bound, was seen. `extract`
    /// rejects such a table with the row named; the in-memory form is then withheld, so
    /// `extract` reads the file and reports exactly that error.
    poisoned: bool,
    /// Rows that carried a 1/K0 window. None at all leaves the 1/K0 fields `None`, as
    /// `read_run_windows` leaves them for a table whose 1/K0 columns are all null.
    n_im: usize,
}

impl RtWindowsBuilder {
    /// `with_im`: the fit has an ion-mobility calibration, so the rows carry 1/K0 windows.
    fn new(n: usize, with_im: bool) -> RtWindowsBuilder {
        RtWindowsBuilder {
            w: RtWindows {
                rt_cal: vec![f64::NAN; n],
                rt_lo: vec![f64::NEG_INFINITY; n],
                rt_hi: vec![f64::INFINITY; n],
                im_lo: with_im.then(|| vec![f32::NEG_INFINITY; n]),
                im_hi: with_im.then(|| vec![f32::INFINITY; n]),
                im_cal: with_im.then(|| vec![None; n]),
                fitted_for: None,
            },
            poisoned: false,
            n_im: 0,
        }
    }

    /// One written row's 1/K0 columns, in row order: the scatter `read_run_windows` applies
    /// to `im_pred_cal`, `im_lo` and `im_hi`. A row whose bounds are null keeps the
    /// unbounded window, and a NaN bound poisons the builder as a NaN RT bound does.
    #[inline]
    fn im_row(&mut self, cid: u32, cal: Option<f64>, lo: Option<f64>, hi: Option<f64>) {
        let c = cid as usize;
        let (Some(wlo), Some(whi), Some(wcal)) =
            (&mut self.w.im_lo, &mut self.w.im_hi, &mut self.w.im_cal)
        else {
            return;
        };
        if c >= wcal.len() {
            return;
        }
        wcal[c] = cal;
        if let (Some(l), Some(h)) = (lo, hi) {
            if l.is_nan() || h.is_nan() {
                self.poisoned = true;
            }
            wlo[c] = l as f32;
            whi[c] = h as f32;
            self.n_im += 1;
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
        if self.n_im == 0 {
            w.im_lo = None;
            w.im_hi = None;
            w.im_cal = None;
        }
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
    /// As [`RtImTrainParams::im_window_floor`].
    pub im_window_floor: Option<ImWindowFloor<'a>>,
}

fn check_cfg(cfg: &RtImTrainConfig) -> Result<()> {
    let holdout_frac = cfg.window_holdout_frac;
    if !(0.0..=0.9).contains(&holdout_frac) {
        anyhow::bail!(
            "rt_im_train.window_holdout_frac must be in [0.0, 0.9], got {holdout_frac}; \
             0.0 disables held-out window sizing"
        );
    }
    // Both size the window per calibrated RT. The local window is on by default, so an
    // explicit `adaptive_rt_window` takes the slot instead of failing the run.
    if cfg.local_window_anchors > 0 && cfg.adaptive_rt_window {
        info!("rt-im-train: adaptive_rt_window is set, so the local RT window is not used");
    }
    if !(cfg.local_window_multiplier.is_finite() && cfg.local_window_multiplier > 0.0) {
        anyhow::bail!(
            "rt_im_train.local_window_multiplier must be a positive number, got {}",
            cfg.local_window_multiplier
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
    run_impl(p, false, |_| Ok(())).map(|(written, _)| written)
}

/// [`run_hashed`], also returning the fitted windows in the form `extract` consumes, so an
/// orchestrator can hand them over instead of having `extract` decode the table it just
/// wrote. `None` only when the table holds a NaN bound (see [`RtWindows`]).
pub fn run_in_memory(p: RtImTrainParams) -> Result<(Written, Option<RtWindows>)> {
    run_impl(p, true, |_| Ok(()))
}

/// [`run_in_memory`], with `check` called on the fit before anything is written. An error
/// from `check` is returned as it is, and neither the windows nor `cal.json` are written:
/// a grouped run under `groups.calibration = per_group` refuses a band whose own anchors
/// leave the windows unbounded this way (`groups.allow_unbounded_rt`).
pub fn run_in_memory_checked(
    p: RtImTrainParams,
    check: impl FnOnce(&RtFit) -> Result<()>,
) -> Result<(Written, Option<RtWindows>)> {
    run_impl(p, true, check)
}

fn run_impl(
    p: RtImTrainParams,
    keep_windows: bool,
    check: impl FnOnce(&RtFit) -> Result<()>,
) -> Result<(Written, Option<RtWindows>)> {
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
    // a patched/updated library iRT is used for both training and application). The same
    // rows' `predicted_im` is decoded in the same open only when the seed has an observed
    // 1/K0 to calibrate it against, so a 3D run reads none of it.
    let with_im = !p.anchor_irt_from_seed && seed_has_observed_im(p.seed_psms)?;
    let lib = library_irt(p.library_precursors, p.precursor_span, with_im)?;
    // The candidate_id -> library row join, built only when the anchors take their iRT from
    // the library, and without a hash table when the ids allow it (see `CidJoin`).
    let join = if p.anchor_irt_from_seed {
        CidJoin::Unused
    } else {
        CidJoin::new(&lib.cid)
    };
    let fit = fit_anchors(
        p.seed_psms,
        p.cfg,
        &join,
        &lib.irt,
        lib.im.as_deref(),
        p.im_anchor_q,
    )?;
    // Its hash-map form has drop glue, so end it here explicitly.
    drop(join);
    check(&fit)?;
    write_windows(
        &fit,
        &ApplyParams {
            library_precursors: p.library_precursors,
            out_windows: p.out_windows,
            out_cal: p.out_cal,
            cfg: p.cfg,
            config_hash: p.config_hash,
            precursor_span: p.precursor_span,
            im_window_floor: p.im_window_floor,
        },
        lib,
        t0,
        keep_windows,
    )
}

/// The rows of a library table this stage reads: the whole table, or its rows
/// `[first, first + n)`. One opener for the iRT and 1/K0 read ([`library_irt`]) and the
/// application pass's composition and ion-mobility columns.
fn library_rows(path: &str, span: Option<(usize, usize)>) -> Result<TableFile> {
    match span {
        None => TableFile::open(path),
        Some((first, n)) => TableFile::open_rows(path, first, n),
    }
}

/// What this stage reads of a library table up front: `candidate_id`, `predicted_irt` and,
/// when asked for, `predicted_im`.
struct LibraryRows {
    cid: Vec<u32>,
    irt: Vec<f32>,
    /// `predicted_im` (1/K0) with NaN for a null, decoded only on a run whose seed carries
    /// `observed_im`; `None` otherwise and for a library without the column (v1).
    im: Option<Vec<f64>>,
}

/// Does the seed table carry `observed_im` (`seed_psms` v2, spectra with 1/K0)? Read from
/// the parquet footer.
fn seed_has_observed_im(seed_psms: &str) -> Result<bool> {
    Ok(mumdia_io::table::column_names(seed_psms)?
        .iter()
        .any(|c| c == "observed_im"))
}

/// A library's `candidate_id` and `predicted_irt`, and with `with_im` its `predicted_im`,
/// from one open: the whole table, or its rows `[first, first + n)` with the ids made local
/// (`id - first`), which is what the same rows written as a band file carry. The span's ids
/// must be the row-aligned range the library invariant promises. A null 1/K0 is NaN; it
/// is not a finite-value error, because a library may predict 1/K0 for some candidates
/// only.
fn library_irt(path: &str, span: Option<(usize, usize)>, with_im: bool) -> Result<LibraryRows> {
    let lib = library_rows(path, span)?;
    let im = if with_im && lib.has_column("predicted_im") {
        Some(
            lib.opt_f64("predicted_im")?
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN))
                .collect(),
        )
    } else {
        None
    };
    match span {
        None => Ok(LibraryRows {
            cid: lib.u32("candidate_id")?,
            irt: lib.f32("predicted_irt")?,
            im,
        }),
        Some((first, _)) => {
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
            Ok(LibraryRows {
                cid,
                irt: lib.f32("predicted_irt")?,
                im,
            })
        }
    }
}

/// The anchor cut a run actually calibrates with, from [`plan_anchors`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchorPlan {
    /// The `spectrum_q` cut used for the anchors: `q_train`, or the first rung of
    /// `anchor_q_ladder` that reaches `min_seed_for_calibration` anchors.
    pub q_train: f64,
    /// Distinct target base peptides at that cut.
    pub n_anchors: usize,
    /// Whether `n_anchors` reaches `min_seed_for_calibration`. When false, `q_train` is the
    /// last rung tried, and a calibration that needs anchors (multi-head) must be skipped.
    pub sufficient: bool,
}

/// Choose the anchor cut for a run. A seed search on a hard run (few peptides, a dominant
/// contaminant, a library whose decoys score high) can leave fewer than
/// `min_seed_for_calibration` target anchors at `q_train`, and then the multi-head
/// calibration has nothing to fit and the per-run LOESS falls back to a fixed window. The
/// cut is relaxed through `anchor_q_ladder` until enough anchors exist; anchors are only
/// an RT reference, and a few percent of false ones move a robust curve far less than no
/// curve at all. Counted as `fit_anchors` counts: finite target rows, one per base
/// peptide.
pub fn plan_anchors(seed_psms: &str, cfg: &RtImTrainConfig) -> Result<AnchorPlan> {
    let best_q = best_target_q(seed_psms)?;
    let count = |q: f64| best_q.values().filter(|&&v| v < q).count();
    let need = cfg.min_seed_for_calibration.max(2);
    let mut plan = AnchorPlan {
        q_train: cfg.q_train,
        n_anchors: count(cfg.q_train),
        sufficient: false,
    };
    plan.sufficient = plan.n_anchors >= need;
    if plan.sufficient {
        return Ok(plan);
    }
    for &q in cfg.anchor_q_ladder.iter().filter(|&&q| q > cfg.q_train) {
        plan = AnchorPlan {
            q_train: q,
            n_anchors: count(q),
            sufficient: false,
        };
        plan.sufficient = plan.n_anchors >= need;
        if plan.sufficient {
            break;
        }
    }
    if plan.sufficient {
        warn!(
            q_train = cfg.q_train,
            relaxed_q = plan.q_train,
            n_anchors = plan.n_anchors,
            need,
            "rt-im-train: too few confident seed anchors at q_train; relaxed the anchor cut"
        );
    } else {
        warn!(
            q_train = cfg.q_train,
            last_q = plan.q_train,
            n_anchors = plan.n_anchors,
            need,
            "rt-im-train: too few seed anchors even at the most relaxed cut; retention-time \
             calibration that needs anchors is skipped"
        );
    }
    Ok(plan)
}

/// Best (lowest) `spectrum_q` per target base peptide, over the seed rows with a finite q,
/// score and retention time: the rows `fit_anchors` may take as anchors before it joins
/// their iRT. The anchor count at a cut is then one comparison per peptide.
fn best_target_q(seed_psms: &str) -> Result<HashMap<u32, f64>> {
    let seed = TableFile::open(seed_psms)?;
    let s_base = seed.u32("base_peptide_id")?;
    let s_q = seed.f64("spectrum_q")?;
    let s_score = seed.f64("score")?;
    let s_rt = seed.f64("observed_rt")?;
    let s_label = seed.str("label")?;
    let mut best_q: HashMap<u32, f64> = HashMap::new();
    for i in 0..seed.nrows {
        if !s_q[i].is_finite() || !s_score[i].is_finite() || !s_rt[i].is_finite() {
            continue;
        }
        if s_label[i] != "target" {
            continue;
        }
        let e = best_q.entry(s_base[i]).or_insert(f64::INFINITY);
        if s_q[i] < *e {
            *e = s_q[i];
        }
    }
    Ok(best_q)
}

/// The most anchors a fit at `q_train` on `seed_psms` can keep: the distinct target base
/// peptides with a finite score and retention time at `spectrum_q < q_train`, counted as
/// [`plan_anchors`] counts them and without its log lines. `fit_anchors` takes the same rows
/// and also drops those without a finite iRT and those `anchor_x_filter` removes, and
/// `seed_pool::refresh_irt` changes only the iRT, so the fit's `n_train` is never larger.
/// A grouped run counts each seed with it before any retention-time step, so that a seed
/// that cannot give bounded windows ([`windows_unbounded`]) is refused before the DeepLC
/// sidecars and the extraction (`groups.allow_unbounded_rt`).
pub(crate) fn seed_anchor_bound(seed_psms: &str, q_train: f64) -> Result<usize> {
    Ok(best_target_q(seed_psms)?
        .values()
        .filter(|&&v| v < q_train)
        .count())
}

/// `cfg` with the anchor cut of `plan`.
pub fn with_anchor_cut(cfg: &RtImTrainConfig, plan: &AnchorPlan) -> RtImTrainConfig {
    let mut c = cfg.clone();
    c.q_train = plan.q_train;
    c
}

/// Fit the calibration on a seed table whose own `predicted_irt` column carries the anchors'
/// iRT (`RtImTrainParams::anchor_irt_from_seed`): the fit a grouped run's bands share under
/// `groups.calibration = global`. No library is read.
pub fn fit_from_seed(seed_psms: &str, cfg: &RtImTrainConfig) -> Result<RtFit> {
    check_cfg(cfg)?;
    fit_anchors(seed_psms, cfg, &CidJoin::Unused, &[], None, None)
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
    let lib = library_irt(p.library_precursors, p.precursor_span, fit.im.is_some())?;
    write_windows(fit, p, lib, t0, keep_windows)
}

/// The anchors and the fit. `join` joins each anchor to its library row, whose iRT is in
/// `lib_irt` and, on a run with ion mobility, whose 1/K0 is in `lib_im`;
/// [`CidJoin::Unused`] reads the iRT from the seed's own `predicted_irt` column and fits no
/// ion-mobility calibration.
fn fit_anchors(
    seed_psms: &str,
    cfg: &RtImTrainConfig,
    join: &CidJoin,
    lib_irt: &[f32],
    lib_im: Option<&[f64]>,
    im_anchor_q: Option<f64>,
) -> Result<RtFit> {
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
    let s_irt: Option<Vec<f32>> = if matches!(join, CidJoin::Unused) {
        Some(seed.f32("predicted_irt")?)
    } else {
        None
    };

    let mut best_per_pep: HashMap<u32, (f64, f64, f64)> = HashMap::new(); // base -> (score, irt, rt)
    let mut best_row: HashMap<u32, usize> = HashMap::new(); // base -> seed row of that apex
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
            None => match join.irt(s_cid[i], lib_irt) {
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
            best_row.insert(s_base[i], i);
        }
    }
    let (anchor_ids, train_irt, train_rt) = sorted_anchor_vectors(best_per_pep);
    // `anchor_x_filter`: the anchors outside the x-domain fences of their predicted iRT are
    // dropped here, before the first fit, so every consumer below (the linear and LOESS
    // fits, `boundary_slopes`, the held-out, in-sample and local window sizing and the
    // composition correction) sees the same filtered anchors.
    let (anchor_ids, train_irt, train_rt, n_rt_outliers_removed) = if cfg.anchor_x_filter {
        let keep = x_domain_inliers(&train_irt);
        let pick = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(&keep)
                .filter(|(_, k)| **k)
                .map(|(x, _)| *x)
                .collect()
        };
        let ids: Vec<u32> = anchor_ids
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(x, _)| *x)
            .collect();
        let removed = keep.iter().filter(|k| !**k).count();
        if removed > 0 {
            info!(
                removed,
                of = keep.len(),
                "rt-im-train: dropped anchors outside the x-domain fences of predicted iRT"
            );
        }
        (ids, pick(&train_irt), pick(&train_rt), removed)
    } else {
        (anchor_ids, train_irt, train_rt, 0)
    };
    let n_train = train_irt.len();
    info!(n_train, "rt-im-train: training points");
    // The ion-mobility calibration, on a seed with 1/K0 only.
    let im_reported = seed.has_column("observed_im");
    let im = if im_reported {
        im_from_seed(
            &seed,
            &s_cid,
            &s_base,
            &s_q,
            &s_label,
            cfg,
            join,
            lib_im,
            im_anchor_q,
        )?
    } else {
        None
    };

    // Fit calibration predicted_irt -> observed RT (seconds).
    // Zero or one point cannot define a useful mapping across the gradient.
    let calibration_available = !windows_unbounded(n_train);
    let (slope, intercept) = if calibration_available {
        linear_fit(&train_irt, &train_rt)
    } else {
        (f64::NAN, f64::NAN)
    };
    let use_loess = calibration_available
        && matches!(cfg.calibration_method, CalibrationMethod::Loess)
        && n_train >= cfg.min_seed_for_calibration;
    let loess = if use_loess {
        Some(Loess::fit_robust(
            &train_irt,
            &train_rt,
            cfg.loess_span,
            200,
            cfg.loess_robust_iters,
        ))
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
        local: None,
        composition: None,
        residuals: (f64::NAN, f64::NAN, f64::NAN),
        im,
        im_reported,
        n_rt_outliers_removed,
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
                let sizing_loess = use_loess.then(|| {
                    Loess::fit_robust(&tr_x, &tr_y, cfg.loess_span, 200, cfg.loess_robust_iters)
                });
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
            (Some(width), FALLBACK_FIXED_STATUS.to_string())
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
    let local = match (w_rt, &plan) {
        (Some(w), WindowPlan::Calibrated)
            if cfg.local_window_anchors > 0 && !cfg.adaptive_rt_window =>
        {
            let cals: Vec<f64> = train_irt.iter().map(|x| fit.predict(*x)).collect();
            let signed: Vec<f64> = cals.iter().zip(&train_rt).map(|(c, y)| y - c).collect();
            LocalWindow::fit(
                &cals,
                &signed,
                cfg.local_window_anchors,
                cfg.p_rt,
                cfg.local_window_multiplier,
                w,
                cfg.fallback_rt_window_s,
            )
        }
        _ => None,
    };
    if let Some(lw) = &local {
        let max_l = lw.left.iter().cloned().fold(0.0, f64::max);
        let max_r = lw.right.iter().cloned().fold(0.0, f64::max);
        info!(
            max_left_s = max_l,
            max_right_s = max_r,
            "rt-im-train: widen-only local RT window"
        );
    }
    // A seed table without the sequence or charge (older or hand-built anchors) skips the
    // correction rather than failing the stage.
    let composition = if cfg.composition_correction
        && calibration_available
        && n_train >= 50
        && seed.has_column("peptidoform")
        && seed.has_column("charge")
    {
        let pform = seed.str("peptidoform")?;
        let charge = seed.i32("charge")?;
        let x: Vec<[f64; 7]> = anchor_ids
            .iter()
            .map(|b| {
                let i = best_row[b];
                composition_features(&pform[i], charge[i])
            })
            .collect();
        let y: Vec<f64> = train_irt
            .iter()
            .zip(&train_rt)
            .map(|(xi, yi)| yi - fit.predict(*xi))
            .collect();
        // A composition term only enters when enough anchors carry it: with a handful of
        // charge-4 anchors (HeLa) the charge-4 coefficient came out at -65 to -74 s.
        let x = drop_sparse_columns(x, MIN_ANCHORS_PER_TERM);
        let beta = huber_fit(&x, &y);
        if let Some(b) = &beta {
            info!(
                intercept = b[0],
                charge3 = b[1],
                charge4plus = b[2],
                per_residue = b[3],
                internal_kr = b[4],
                his = b[5],
                met_ox = b[6],
                "rt-im-train: composition RT correction (s)"
            );
        }
        beta
    } else {
        None
    };
    fit.adaptive = adaptive;
    fit.local = local;
    fit.composition = composition;
    fit.residuals = residuals;
    Ok(fit)
}

/// Apply `fit` to every row of one library table and write the windows, the `cal.json` and
/// the windows' report. With `keep_windows`, also return the windows in the form `extract`
/// consumes (see [`RtWindows`]; `None` when the table holds a NaN bound).
fn write_windows(
    fit: &RtFit,
    p: &ApplyParams,
    lib: LibraryRows,
    t0: Instant,
    keep_windows: bool,
) -> Result<(Written, Option<RtWindows>)> {
    // Apply to every library candidate, streamed: each 65,536-row chunk of the window
    // table is computed and written before the next, through the same writer and the same
    // chunk sequence `write_table` uses, so the file is byte-identical to writing the seven
    // whole columns at once. Holding them whole was 76 bytes per candidate (three
    // `Option<f64>` ion-mobility columns among them), 15 GB at 203M rows, for a table that
    // is written once and never read back here.
    let LibraryRows {
        cid,
        irt,
        im: lib_im,
    } = lib;
    let n = cid.len();
    // The ion-mobility calibration and the rows' 1/K0, when the fit has one. A library
    // without `predicted_im` cannot have given IM anchors, so a calibration implies the
    // column was decoded.
    let im_cal = fit.im.as_ref().zip(lib_im.as_deref());
    // The composition correction and the IM application read more columns of the same rows,
    // through one open shared by both, made only when either needs it.
    let more = (fit.composition.is_some() || im_cal.is_some())
        .then(|| library_rows(p.library_precursors, p.precursor_span))
        .transpose()?;
    let charge: Option<Vec<i32>> = more.as_ref().map(|t| t.i32("charge")).transpose()?;
    // The IM application's m/z, per row.
    let im_mz: Option<Vec<f64>> = match (&more, im_cal) {
        (Some(t), Some(_)) => Some(t.f64("precursor_mz")?),
        _ => None,
    };
    // Composition shift per library row, read only when the fit carries a correction.
    let shift: Option<Vec<f64>> = match (&fit.composition, &more, &charge) {
        (Some(beta), Some(lib), Some(charge)) => {
            let pform = lib.str("peptidoform")?;
            Some(
                (0..n)
                    .map(|i| {
                        let f = composition_features(&pform[i], charge[i]);
                        let v: f64 = (0..7).map(|c| f[c] * beta[c]).sum();
                        // bounded: a shift beyond three window half-widths is a fit artefact
                        let cap = 3.0 * fit.w_rt.unwrap_or(f64::INFINITY);
                        v.clamp(-cap, cap)
                    })
                    .collect(),
            )
        }
        _ => None,
    };
    drop(more);
    if let (Some(im), Some(mz), Some(z)) = (lib_im.as_deref(), &im_mz, &charge) {
        if im.len() != n || mz.len() != n || z.len() != n {
            anyhow::bail!(
                "{}: {} predicted_im, {} precursor_mz and {} charge values for {n} windows",
                p.library_precursors,
                im.len(),
                mz.len(),
                z.len()
            );
        }
    }
    let mut n_nonfinite_irt = 0u64;
    let mut kept = keep_windows.then(|| RtWindowsBuilder::new(n, im_cal.is_some()));
    let windows = write_table_chunked_hashed(p.out_windows, n, |r| {
        let k = r.len();
        let (mut cid_c, mut cal_c, mut lo_c, mut hi_c) = (
            Vec::with_capacity(k),
            Vec::with_capacity(k),
            Vec::with_capacity(k),
            Vec::with_capacity(k),
        );
        for i in r.clone() {
            // A row whose library iRT is not finite has no calibrated RT, which is the
            // documented "search the whole gradient" sentinel rather than an arithmetic
            // accident. The parquet reader maps a null f32 to NaN, so one failed prediction
            // in an imported library reaches here (docs/31 F4).
            let usable_irt = (irt[i] as f64).is_finite();
            if !usable_irt {
                n_nonfinite_irt += 1;
            }
            let calibrated_rt = (fit.calibration_available && usable_irt)
                .then(|| fit.predict(irt[i] as f64) + shift.as_ref().map_or(0.0, |s| s[i]));
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
            let (cal, lo, hi) = match (&fit.local, calibrated_rt) {
                (Some(lw), Some(c)) => {
                    let (l, r) = lw.at(c);
                    (c, c - l, c + r)
                }
                _ => candidate_window(calibrated_rt, width),
            };
            if let Some(b) = kept.as_mut() {
                b.row(cid[i], cal, lo, hi);
            }
            cid_c.push(cid[i]);
            cal_c.push(cal);
            lo_c.push(lo);
            hi_c.push(hi);
        }
        // The IM columns: the fit's IM calibration applied to each row's `predicted_im`, at
        // the row's m/z and charge for the CCS conversion; null for a row without a finite,
        // positive 1/K0. Without a calibration they are null, as on every 3D run.
        let [im_c, imlo_c, imhi_c]: [Vec<Option<f64>>; 3] = match (im_cal, &im_mz, &charge) {
            (Some((cal, im)), Some(mz), Some(z)) => {
                let pred: Vec<Option<f64>> = r
                    .map(|i| {
                        (im[i].is_finite() && im[i] > 0.0).then(|| cal.predict(im[i], mz[i], z[i]))
                    })
                    .collect();
                let lo = pred.iter().map(|v| v.map(|c| c - cal.w_im)).collect();
                let hi = pred.iter().map(|v| v.map(|c| c + cal.w_im)).collect();
                [pred, lo, hi]
            }
            _ => [vec![None; k], vec![None; k], vec![None; k]],
        };
        if let (Some(b), true) = (kept.as_mut(), im_cal.is_some()) {
            for (j, &c) in cid_c.iter().enumerate() {
                b.im_row(c, im_c[j], imlo_c[j], imhi_c[j]);
            }
        }
        Ok(vec![
            Col::U32("candidate_id".into(), cid_c),
            Col::F64("rt_pred_cal".into(), cal_c),
            Col::F64("rt_lo".into(), lo_c),
            Col::F64("rt_hi".into(), hi_c),
            Col::OptF64("im_pred_cal".into(), im_c),
            Col::OptF64("im_lo".into(), imlo_c),
            Col::OptF64("im_hi".into(), imhi_c),
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
    let mut cal_json = json!({
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
            "local_window_anchors": p.cfg.local_window_anchors,
            "local_window": fit.local.as_ref().map(|lw| json!({
                "multiplier": p.cfg.local_window_multiplier,
                "max_left_s": lw.left.iter().cloned().fold(0.0, f64::max),
                "max_right_s": lw.right.iter().cloned().fold(0.0, f64::max),
                "grid_rt": lw.grid,
                "left_s": lw.left,
                "right_s": lw.right,
            })),
    });
    // The ion-mobility calibration, on a run whose seed carries 1/K0 only (null fields when
    // it could not be fitted), so a 3D `cal.json` is unchanged. `w_im` is sized on held-out
    // anchors unless `w_im_sizing` says otherwise; the in-sample residual is a fit
    // diagnostic only (docs/08_rt_im_train.md section 4).
    if fit.n_rt_outliers_removed > 0 {
        cal_json["n_rt_outliers_removed"] = json!(fit.n_rt_outliers_removed);
    }
    let im = fit.im.as_ref();
    if fit.im_reported {
        let im_fields = json!({
            "im_method": if im.is_some() { "per_charge_linear_ccs" } else { "unavailable" },
            "im_n_train": im.map(|c| c.n_train),
            "im_global": im.map(|c| json!({"a": c.coefs.0 .0, "b": c.coefs.0 .1})),
            "im_per_charge": im.map(|c| c.coefs.1.iter()
                .map(|(z, (a, b, n))| (z.to_string(), json!({"a": a, "b": b, "n": n})))
                .collect::<serde_json::Map<_, _>>()),
            "w_im": im.map(|c| c.w_im),
            "w_im_sizing": im.map(|c| c.sizing),
            "p_im": p.cfg.p_im,
            "im_window_multiplier": p.cfg.im_window_multiplier,
            "im_window_min": p.cfg.im_window_min,
            "im_holdout_frac": IM_HOLDOUT_FRAC,
            "im_n_holdout": im.map(|c| c.n_holdout),
            "im_holdout_resid_abs_median": im.and_then(|c| c.holdout_abs_median),
            "im_holdout_resid_p_im": im.and_then(|c| c.holdout_p_im),
            "im_in_sample_resid_abs_median": im.map(|c| c.in_sample_abs_median),
        });
        if let (Some(obj), serde_json::Value::Object(extra)) = (cal_json.as_object_mut(), im_fields)
        {
            obj.extend(extra);
        }
        // The decoy-corrected seed sizing (the default), on a run with an IM calibration only,
        // so a 3D cal.json is unchanged.
        if let Some(e) = im.and_then(|c| c.seed.as_ref()) {
            cal_json["im_window_sizing"] = json!({
                "mode": "seed_decoy_corrected",
                "q_max": p.cfg.im_window_sizing_q,
                "seed_targets": e.n_targets,
                "seed_decoys": e.n_decoys,
                "decoy_corrected_p_im": e.p_im_residual,
                "applied": e.applied,
            });
        }
        // The second pass's floor on `w_im` and where it came from.
        if let Some(f) = &p.im_window_floor {
            cal_json["im_window_floor"] = json!({
                "floor": p.cfg.im_window_min,
                "configured": f.configured,
                "first_pass_w_im": f.first_pass_w_im,
                "source": f.source,
            });
        }
    }
    mumdia_io::json::write_json(p.out_cal, &cal_json)?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("n_train".to_string(), json!(n_train));
    stats.insert("w_rt".to_string(), json!(w_rt));
    stats.insert("calibration_status".to_string(), json!(status));
    if fit.im_reported {
        stats.insert("im_n_train".to_string(), json!(im.map(|c| c.n_train)));
        stats.insert("w_im".to_string(), json!(im.map(|c| c.w_im)));
    }
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

/// The ion-mobility calibration from the confident target rows of a seed with
/// `observed_im` (`s_*` are the seed columns `fit_anchors` decoded): each anchor's observed
/// 1/K0 from the seed, its `predicted_im` from its library row (`join`, `lib_im`). The
/// confident cut is `cfg.q_train` and, when given, the seed's own cut `im_anchor_q`
/// (`RtImTrainParams::im_anchor_q`), so a relaxed `anchor_q_ladder` cut adds RT anchors but
/// no IM anchors. The seed holds one row per candidate,
/// and 1/K0 depends on charge where RT does not, so the anchors are per candidate, not per
/// base peptide. `None`, with a warning, when there is nothing to calibrate.
#[allow(clippy::too_many_arguments)]
fn im_from_seed(
    seed: &TableFile,
    s_cid: &[u32],
    s_base: &[u32],
    s_q: &[f64],
    s_label: &[String],
    cfg: &RtImTrainConfig,
    join: &CidJoin,
    lib_im: Option<&[f64]>,
    im_anchor_q: Option<f64>,
) -> Result<Option<ImCal>> {
    if matches!(join, CidJoin::Unused) {
        warn!(
            "rt-im-train: the anchors take their iRT from the seed, so no library row is \
             joined for their predicted 1/K0; ion mobility is not calibrated and the IM \
             windows are null"
        );
        return Ok(None);
    }
    let Some(lib_im) = lib_im else {
        warn!(
            "rt-im-train: the seed carries observed 1/K0 but the library has no predicted_im; \
             ion mobility is not calibrated and the IM windows are null"
        );
        return Ok(None);
    };
    let s_obs = seed.opt_f64("observed_im")?;
    let s_mz = seed.f64("precursor_mz")?;
    let s_z = seed.i32("charge")?;
    let mut anchors: Vec<ImAnchor> = Vec::new();
    for i in 0..seed.nrows {
        if !s_q[i].is_finite()
            || s_q[i] >= cfg.q_train
            || im_anchor_q.is_some_and(|c| s_q[i] > c)
            || s_label[i] != "target"
        {
            continue;
        }
        let (Some(obs), Some(row)) = (s_obs[i], join.row(s_cid[i])) else {
            continue;
        };
        let pred = lib_im[row];
        if obs.is_finite() && obs > 0.0 && pred.is_finite() && pred > 0.0 {
            anchors.push(ImAnchor {
                cid: s_cid[i],
                base_peptide_id: s_base[i],
                charge: s_z[i],
                mz: s_mz[i],
                pred_im: pred,
                obs_im: obs,
            });
        }
    }
    // A fixed order before any fit or reduction.
    anchors.sort_by_key(|a| a.cid);
    let mut fit = fit_im(&anchors, cfg);
    if let (Some(c), mumdia_core::config::ImWindowSizing::SeedDecoyCorrected) =
        (fit.as_mut(), cfg.im_window_sizing)
    {
        // Every seed PSM up to the cut, targets and decoys, against this calibration.
        let (mut t, mut d) = (Vec::new(), Vec::new());
        for i in 0..seed.nrows {
            if !(s_q[i].is_finite() && s_q[i] <= cfg.im_window_sizing_q) {
                continue;
            }
            let (Some(obs), Some(row)) = (s_obs[i], join.row(s_cid[i])) else {
                continue;
            };
            let pred = lib_im[row];
            if !(obs.is_finite() && obs > 0.0 && pred.is_finite() && pred > 0.0) {
                continue;
            }
            let r = (obs - im_predict(&c.coefs, pred, s_mz[i], s_z[i])).abs();
            if s_label[i] == "decoy" {
                d.push(r);
            } else {
                t.push(r);
            }
        }
        let (n_t, n_d) = (t.len(), d.len());
        // No decoy with a 1/K0: the seed was written without `observed_im_q` (an older
        // engine, or `search-seed` under `anchors`), so only its confident targets carry
        // one and there is nothing to correct against.
        let est = if n_d == 0 {
            warn!(
                seed_targets = n_t,
                "rt-im-train: no seed decoy carries an observed 1/K0, so the seed was not                  written for the decoy-corrected 1/K0 window; keeping the anchors' w_im"
            );
            None
        } else {
            decoy_corrected_quantile(&mut t, &mut d, cfg.p_im, SEED_SIZING_MIN_TRUE)
        };
        let sized = est.map(|e| (e * cfg.im_window_multiplier).max(cfg.im_window_min));
        let applied = sized.is_some_and(|w| w > c.w_im);
        info!(
            seed_targets = n_t,
            seed_decoys = n_d,
            q_max = cfg.im_window_sizing_q,
            decoy_corrected_p_im = est,
            anchors_w_im = c.w_im,
            applied,
            "rt-im-train: 1/K0 window from every seed PSM, decoy-corrected              (rt_im_train.im_window_sizing = seed_decoy_corrected)"
        );
        if est.is_none() && n_d > 0 {
            warn!(
                seed_targets = n_t,
                seed_decoys = n_d,
                min_true = SEED_SIZING_MIN_TRUE,
                "rt-im-train: too few seed targets above the decoys for the decoy-corrected                  1/K0 window; keeping the anchors' w_im"
            );
        }
        c.seed = Some(SeedImEstimate {
            p_im_residual: est,
            n_targets: n_t,
            n_decoys: n_d,
            applied,
        });
        if applied {
            c.w_im = sized.unwrap();
            c.sizing = "seed_decoy_corrected";
        }
    }
    match &fit {
        Some(c) => info!(
            n_anchors = c.n_train,
            w_im = c.w_im,
            sizing = c.sizing,
            "rt-im-train: ion-mobility calibration"
        ),
        None => warn!(
            n_anchors = anchors.len(),
            min_anchors = cfg.min_seed_for_calibration,
            "rt-im-train: too few ion-mobility anchors; ion mobility is not calibrated and \
             the IM windows are null"
        ),
    }
    Ok(fit)
}

/// Fewest true hits (targets minus decoys) the decoy-corrected 1/K0 window is taken from.
const SEED_SIZING_MIN_TRUE: usize = 100;

/// The decoy-corrected sizing of `rt_im_train.im_window_sizing = "seed_decoy_corrected"`,
/// for `cal.json`.
#[derive(Clone, Debug)]
struct SeedImEstimate {
    /// The `p_im` percentile of the true hits' |observed - calibrated| 1/K0.
    p_im_residual: Option<f64>,
    n_targets: usize,
    n_decoys: usize,
    /// Whether it widened `w_im` (it never narrows the anchors' window).
    applied: bool,
}

/// The `q` quantile of the TRUE hits among `targets` and `decoys` residuals: the smallest
/// target residual `x` at which `(T(<= x) - D(<= x)) / (T - D)`, made non-decreasing, reaches
/// `q`. A false target is distributed as a decoy, so the difference of the two cumulative
/// counts is the true hits' cumulative count. `None` when `T - D < min_true`. Both inputs are
/// sorted in place; the result depends on the values only, not their order.
fn decoy_corrected_quantile(
    targets: &mut [f64],
    decoys: &mut [f64],
    q: f64,
    min_true: usize,
) -> Option<f64> {
    if targets.len() < decoys.len() + min_true {
        return None;
    }
    targets.sort_by(f64::total_cmp);
    decoys.sort_by(f64::total_cmp);
    let n_true = (targets.len() - decoys.len()) as f64;
    let (mut j, mut best) = (0usize, 0.0f64);
    for (k, &x) in targets.iter().enumerate() {
        // Count every target at the same value before reading the fraction.
        if k + 1 < targets.len() && targets[k + 1] == x {
            continue;
        }
        while j < decoys.len() && decoys[j] <= x {
            j += 1;
        }
        best = best.max((k + 1) as f64 - j as f64);
        if best / n_true >= q {
            return Some(x);
        }
    }
    targets.last().copied()
}

/// One ion-mobility calibration anchor: a confident target seed PSM with an observed and a
/// library-predicted 1/K0.
#[derive(Clone, Debug)]
struct ImAnchor {
    cid: u32,
    base_peptide_id: u32,
    charge: i32,
    mz: f64,
    pred_im: f64,
    obs_im: f64,
}

/// `(global (a, b), per charge (a, b, n))` of `obs_ccs = a + b * pred_ccs`.
type ImCoefs = (
    (f64, f64),
    std::collections::BTreeMap<i32, (f64, f64, usize)>,
);

/// A fitted per-run ion-mobility calibration: `obs_ccs = a + b * pred_ccs`, per charge where
/// that charge has `im_min_anchors_per_charge` anchors, else the global fit, and the
/// half-window `w_im` around it.
struct ImCal {
    coefs: ImCoefs,
    w_im: f64,
    /// The decoy-corrected seed sizing, when configured.
    seed: Option<SeedImEstimate>,
    /// `holdout`, or `holdout_fallback_in_sample` (the `w_rt_sizing` vocabulary).
    sizing: &'static str,
    n_train: usize,
    n_holdout: usize,
    holdout_abs_median: Option<f64>,
    holdout_p_im: Option<f64>,
    in_sample_abs_median: f64,
}

impl ImCal {
    /// The calibrated 1/K0 of a library 1/K0 at a precursor m/z and charge.
    fn predict(&self, pred_im: f64, mz: f64, charge: i32) -> f64 {
        im_predict(&self.coefs, pred_im, mz, charge)
    }
}

fn im_predict(coefs: &ImCoefs, pred_im: f64, mz: f64, charge: i32) -> f64 {
    use mumdia_core::constants::{ccs_to_im, im_to_ccs};
    let (a, b) = coefs.1.get(&charge).map(|c| (c.0, c.1)).unwrap_or(coefs.0);
    ccs_to_im(a + b * im_to_ccs(pred_im, mz, charge), mz, charge)
}

/// Least-squares `obs_ccs = a + b * pred_ccs`, globally and per charge. The anchors are in a
/// fixed order (sorted by candidate id) and the per-charge map is ordered, so the fit is
/// deterministic.
fn fit_im_coefs(anchors: &[&ImAnchor], min_per_charge: usize) -> ImCoefs {
    use mumdia_core::constants::im_to_ccs;
    let ccs = |a: &ImAnchor| {
        (
            im_to_ccs(a.pred_im, a.mz, a.charge),
            im_to_ccs(a.obs_im, a.mz, a.charge),
        )
    };
    let (x, y): (Vec<f64>, Vec<f64>) = anchors.iter().map(|a| ccs(a)).unzip();
    let (b, a) = linear_fit(&x, &y);
    let mut by_z: std::collections::BTreeMap<i32, (Vec<f64>, Vec<f64>)> = Default::default();
    for an in anchors {
        let (px, py) = ccs(an);
        let e = by_z.entry(an.charge).or_default();
        e.0.push(px);
        e.1.push(py);
    }
    let per_charge = by_z
        .into_iter()
        .filter(|(_, (xs, _))| xs.len() >= min_per_charge.max(2))
        .map(|(z, (xs, ys))| {
            let (bz, az) = linear_fit(&xs, &ys);
            (z, (az, bz, xs.len()))
        })
        .collect();
    ((a, b), per_charge)
}

/// Fit the ion-mobility calibration on every anchor and size `w_im` from the anchors held
/// out of a refit ([`is_holdout`] on the base peptide at [`IM_HOLDOUT_FRAC`], as the RT
/// holdout splits). `None` below `min_seed_for_calibration` anchors: no IM window is safer
/// than one from a handful of points, and a null window means "do not gate". Too few
/// anchors on one side of the split size `w_im` in-sample, with a warning.
fn fit_im(anchors: &[ImAnchor], cfg: &RtImTrainConfig) -> Option<ImCal> {
    if anchors.len() < cfg.min_seed_for_calibration.max(2) {
        return None;
    }
    let all: Vec<&ImAnchor> = anchors.iter().collect();
    let coefs = fit_im_coefs(&all, cfg.im_min_anchors_per_charge);
    let resid = |c: &ImCoefs, set: &[&ImAnchor]| -> Vec<f64> {
        set.iter()
            .map(|a| (a.obs_im - im_predict(c, a.pred_im, a.mz, a.charge)).abs())
            .collect()
    };
    let in_sample = resid(&coefs, &all);
    let (train, held): (Vec<&ImAnchor>, Vec<&ImAnchor>) = all
        .iter()
        .partition(|a| !is_holdout(a.base_peptide_id, IM_HOLDOUT_FRAC));
    let (p_resid, sizing, holdout) = if held.len() >= MIN_HOLDOUT_ANCHORS
        && train.len() >= cfg.min_seed_for_calibration.max(2)
    {
        let r = resid(&fit_im_coefs(&train, cfg.im_min_anchors_per_charge), &held);
        let pr = percentile(&r, cfg.p_im);
        (pr, "holdout", Some((percentile(&r, 0.5), pr)))
    } else {
        warn!(
            n_holdout = held.len(),
            n_sizing_train = train.len(),
            "rt-im-train: too few ion-mobility anchors on one side of the holdout split; \
             sizing w_im in-sample"
        );
        (
            percentile(&in_sample, cfg.p_im),
            "holdout_fallback_in_sample",
            None,
        )
    };
    Some(ImCal {
        coefs,
        w_im: (p_resid * cfg.im_window_multiplier).max(cfg.im_window_min),
        seed: None,
        sizing,
        n_train: anchors.len(),
        n_holdout: held.len(),
        holdout_abs_median: holdout.map(|h| h.0),
        holdout_p_im: holdout.map(|h| h.1),
        in_sample_abs_median: percentile(&in_sample, 0.5),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::write_table;

    /// Anchors on a known per-charge CCS map with a +/-0.004 1/K0 jitter.
    fn im_anchors(n: u32) -> Vec<ImAnchor> {
        use mumdia_core::constants::{ccs_to_im, im_to_ccs};
        (0..n)
            .map(|i| {
                let z = if i % 3 == 0 { 3 } else { 2 };
                let mz = 400.0 + (i as f64) * 0.9;
                let pred = 0.7 + (i as f64) * 0.0008;
                let (a, b) = if z == 2 { (10.0, 1.02) } else { (-5.0, 0.98) };
                let jitter = if i % 2 == 0 { 0.004 } else { -0.004 };
                ImAnchor {
                    cid: i,
                    base_peptide_id: i,
                    charge: z,
                    mz,
                    pred_im: pred,
                    obs_im: ccs_to_im(a + b * im_to_ccs(pred, mz, z), mz, z) + jitter,
                }
            })
            .collect()
    }

    /// True residuals uniform on [0, 0.05) mixed with false targets and an equal number of
    /// decoys uniform on [0, 0.3): the decoy-corrected 95th percentile is the true one
    /// (0.0475), where the targets' plain percentile is far wider; too few true hits is None.
    #[test]
    fn the_decoy_corrected_quantile_is_the_true_hits_quantile() {
        let true_hits: Vec<f64> = (0..2000)
            .map(|i| 0.05 * (i as f64 + 0.5) / 2000.0)
            .collect();
        let false_hits: Vec<f64> = (0..1000).map(|i| 0.3 * (i as f64 + 0.5) / 1000.0).collect();
        let mut t: Vec<f64> = true_hits.iter().chain(&false_hits).copied().collect();
        let mut d = false_hits.clone();
        let est = decoy_corrected_quantile(&mut t, &mut d, 0.95, 100).unwrap();
        assert!((est - 0.0475).abs() < 0.001, "{est}");
        let mut plain = true_hits
            .iter()
            .chain(&false_hits)
            .copied()
            .collect::<Vec<_>>();
        plain.sort_by(f64::total_cmp);
        assert!(plain[(0.95 * plain.len() as f64) as usize] > 0.2);
        let (mut t2, mut d2) = (vec![0.01; 150], vec![0.2; 100]);
        assert!(decoy_corrected_quantile(&mut t2, &mut d2, 0.95, 100).is_none());
    }

    #[test]
    fn im_calibration_recovers_a_per_charge_ccs_map_and_sizes_w_im_held_out() {
        let anchors = im_anchors(600);
        let cfg = RtImTrainConfig::default();
        let cal = fit_im(&anchors, &cfg).expect("600 anchors calibrate");
        assert_eq!(cal.sizing, "holdout");
        assert_eq!(cal.n_holdout, 300, "ids 0..300 satisfy id % 1000 < 300");
        assert_eq!(cal.n_train, 600);
        let (a2, b2, n2) = cal.coefs.1[&2];
        assert!(
            (b2 - 1.02).abs() < 0.01 && (a2 - 10.0).abs() < 5.0,
            "{a2} {b2}"
        );
        assert_eq!(n2, 400);
        let (_, b3, _) = cal.coefs.1[&3];
        assert!((b3 - 0.98).abs() < 0.01, "{b3}");
        // The residuals are the jitter, so the held-out p95 is about 0.004.
        let p = cal.holdout_p_im.unwrap();
        assert!((p - 0.004).abs() < 0.001, "{p}");
        assert_eq!(
            cal.w_im,
            (p * cfg.im_window_multiplier).max(cfg.im_window_min)
        );
        // A charge below `im_min_anchors_per_charge` takes the global fit.
        let sparse = RtImTrainConfig {
            im_min_anchors_per_charge: 300,
            ..RtImTrainConfig::default()
        };
        let cal3 = fit_im(&anchors, &sparse).unwrap();
        assert!(cal3.coefs.1.contains_key(&2) && !cal3.coefs.1.contains_key(&3));
        // Every anchor on the training side of the split: sized in-sample, and said so.
        let mut none_held = anchors.clone();
        for a in none_held.iter_mut() {
            a.base_peptide_id += 300;
        }
        let fb = fit_im(&none_held[..600], &cfg).unwrap();
        assert_eq!(fb.sizing, "holdout_fallback_in_sample");
        assert_eq!((fb.n_holdout, fb.holdout_p_im), (0, None));
        // Too few anchors: no calibration rather than a window from a handful of points.
        assert!(fit_im(&anchors[..10], &cfg).is_none());
    }

    /// A library with `predicted_im` and a seed with `observed_im`: the streamed windows carry
    /// the fit's IM calibration applied to every row (null where the row has no 1/K0), and
    /// `cal.json` records it. The same inputs without ion mobility write null IM columns and a
    /// `cal.json` without IM keys. A seed with 1/K0 against a library without it records that
    /// there is no calibration. The IM anchors follow the anchor cut the RT anchors use.
    #[test]
    fn im_windows_are_the_calibration_applied_to_every_row() {
        use mumdia_core::constants::{ccs_to_im, im_to_ccs};
        let n = 600usize;
        let z: Vec<i32> = (0..n).map(|i| if i % 3 == 0 { 3 } else { 2 }).collect();
        let mz: Vec<f64> = (0..n).map(|i| 400.0 + i as f64 * 0.9).collect();
        let pred: Vec<Option<f64>> = (0..n)
            .map(|i| (i % 50 != 7).then_some(0.7 + i as f64 * 0.0008))
            .collect();
        let obs: Vec<Option<f64>> = (0..n)
            .map(|i| {
                let p = pred[i]?;
                let (a, b) = if z[i] == 2 {
                    (10.0, 1.02)
                } else {
                    (-5.0, 0.98)
                };
                let jitter = if i % 2 == 0 { 0.004 } else { -0.004 };
                Some(ccs_to_im(a + b * im_to_ccs(p, mz[i], z[i]), mz[i], z[i]) + jitter)
            })
            .collect();
        // Every tenth row is confident only at the relaxed cut.
        let q: Vec<f64> = (0..n)
            .map(|i| if i % 10 == 1 { 0.03 } else { 0.001 })
            .collect();
        let irt: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let cfg = RtImTrainConfig::default();
        let stage_with = |tag: &str,
                          lib_im: bool,
                          seed_im: bool,
                          cfg: &RtImTrainConfig,
                          floor: Option<ImWindowFloor>| {
            let prec = scratch(&format!("im_prec_{tag}.parquet"));
            let seed = scratch(&format!("im_seed_{tag}.parquet"));
            let mut lib_cols = vec![
                Col::U32("candidate_id".into(), (0..n as u32).collect()),
                Col::F32("predicted_irt".into(), irt.clone()),
                Col::F64("precursor_mz".into(), mz.clone()),
                Col::I32("charge".into(), z.clone()),
            ];
            let mut seed_cols = vec![
                Col::U32("candidate_id".into(), (0..n as u32).collect()),
                Col::U32("base_peptide_id".into(), (0..n as u32).collect()),
                Col::F64("spectrum_q".into(), q.clone()),
                Col::F64("score".into(), (0..n).map(|i| 1.0 + i as f64).collect()),
                Col::F64(
                    "observed_rt".into(),
                    irt.iter().map(|&x| 60.0 + 30.0 * x as f64).collect(),
                ),
                Col::Str("label".into(), vec!["target".to_string(); n]),
                Col::F64("precursor_mz".into(), mz.clone()),
                Col::I32("charge".into(), z.clone()),
            ];
            if lib_im {
                lib_cols.push(Col::OptF64("predicted_im".into(), pred.clone()));
            }
            if seed_im {
                seed_cols.push(Col::OptF64("observed_im".into(), obs.clone()));
            }
            write_table(&prec, lib_cols).unwrap();
            write_table(&seed, seed_cols).unwrap();
            let windows = scratch(&format!("im_windows_{tag}.parquet"));
            let cal = scratch(&format!("im_cal_{tag}.json"));
            let (written, kept) = run_in_memory(RtImTrainParams {
                precursor_span: None,
                seed_psms: &seed,
                library_precursors: &prec,
                out_windows: &windows,
                out_cal: &cal,
                cfg,
                config_hash: "test",
                im_window_floor: floor,
                im_anchor_q: None,
                anchor_irt_from_seed: false,
            })
            .unwrap();
            assert_eq!(written.rows, n as u64);
            let kept = kept.expect("the RT windows are handed over");
            // The 4D arm of the in-memory/file equivalence: the 1/K0 windows handed to
            // extract are the ones it reads back from the file just written.
            let from_file = crate::stages::extract::read_run_windows(&windows, n).unwrap();
            assert_eq!(bits(&kept), bits(&from_file), "{tag}");
            assert_eq!(im_bits(&kept), im_bits(&from_file), "{tag}");
            assert_eq!(kept.has_im(), lib_im && seed_im, "{tag}");
            let cal_json: serde_json::Value = mumdia_io::json::read_json(&cal).unwrap();
            let w = TableFile::open(&windows).unwrap();
            let cols = (
                w.opt_f64("im_pred_cal").unwrap(),
                w.opt_f64("im_lo").unwrap(),
                w.opt_f64("im_hi").unwrap(),
            );
            (cal_json, cols)
        };
        let stage = |tag: &str, lib_im: bool, seed_im: bool, cfg: &RtImTrainConfig| {
            stage_with(tag, lib_im, seed_im, cfg, None)
        };
        type ImCols = (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>);
        let all_null = |c: &ImCols| c.0.iter().chain(&c.1).chain(&c.2).all(|v| v.is_none());

        // 3D: no IM keys anywhere, null columns, and the library's 1/K0 is never read.
        for (tag, lib_im) in [("3d", false), ("3d_lib_im", true)] {
            let (cal_json, cols) = stage(tag, lib_im, false, &cfg);
            let keys: Vec<&String> = cal_json.as_object().unwrap().keys().collect();
            assert!(
                keys.iter()
                    .all(|k| !k.starts_with("im_") && *k != "w_im" && *k != "p_im"),
                "{tag}: a 3D cal.json has no IM keys: {keys:?}"
            );
            assert!(all_null(&cols), "{tag}");
        }
        // A seed with 1/K0 against a library without it: the block says why there is none.
        let (cal_json, cols) = stage("no_lib_im", false, true, &cfg);
        assert_eq!(cal_json["im_method"], "unavailable");
        assert!(cal_json["w_im"].is_null() && cal_json["w_im_sizing"].is_null());
        assert!(all_null(&cols));

        // 4D: the calibration applied to every row.
        let (cal_json, (im_c, im_lo, im_hi)) = stage("im", true, true, &cfg);
        assert_eq!(cal_json["im_method"], "per_charge_linear_ccs");
        assert_eq!(cal_json["w_im_sizing"], "holdout");
        assert_eq!(cal_json["im_holdout_frac"], IM_HOLDOUT_FRAC);
        // The anchors `im_from_seed` builds here: the confident rows with a 1/K0 on both
        // sides, in candidate order.
        let anchors_at = |q_train: f64| -> Vec<ImAnchor> {
            (0..n)
                .filter(|&i| q[i] < q_train && pred[i].is_some())
                .map(|i| ImAnchor {
                    cid: i as u32,
                    base_peptide_id: i as u32,
                    charge: z[i],
                    mz: mz[i],
                    pred_im: pred[i].unwrap(),
                    obs_im: obs[i].unwrap(),
                })
                .collect()
        };
        let anchors = anchors_at(cfg.q_train);
        let want = fit_im(&anchors, &cfg).expect("anchors calibrate");
        assert_eq!(cal_json["im_n_train"], anchors.len());
        // A number read back from `cal.json` can be one ulp off (serde_json's default float
        // parser), so it is compared to 1e-12; the windows themselves are compared bit for bit.
        let close = |v: &serde_json::Value, want: f64| {
            v.as_f64()
                .is_some_and(|x| (x - want).abs() <= 1e-12 * want.abs())
        };
        assert!(close(&cal_json["w_im"], want.w_im), "{}", cal_json["w_im"]);
        for i in 0..n {
            let Some(p) = pred[i] else {
                assert!(im_c[i].is_none() && im_lo[i].is_none() && im_hi[i].is_none());
                continue;
            };
            let c = want.predict(p, mz[i], z[i]);
            assert_eq!(im_c[i].map(f64::to_bits), Some(c.to_bits()), "row {i}");
            assert_eq!(
                im_lo[i].map(f64::to_bits),
                Some((c - want.w_im).to_bits()),
                "row {i}"
            );
            assert_eq!(
                im_hi[i].map(f64::to_bits),
                Some((c + want.w_im).to_bits()),
                "row {i}"
            );
        }
        assert!(cal_json.get("im_window_floor").is_none());
        // The second pass's floor: `im_window_min` raised above the fitted width, and where it
        // came from recorded beside it.
        let floored = RtImTrainConfig {
            im_window_min: 4.0 * want.w_im,
            ..cfg.clone()
        };
        let floor = ImWindowFloor {
            configured: cfg.im_window_min,
            first_pass_w_im: 4.0 * want.w_im,
            source: "first/cal.json",
        };
        let (cal_floor, (_, f_lo, f_hi)) =
            stage_with("im_floor", true, true, &floored, Some(floor));
        assert!(
            close(&cal_floor["w_im"], 4.0 * want.w_im),
            "{}",
            cal_floor["w_im"]
        );
        assert_eq!(cal_floor["im_window_floor"]["source"], "first/cal.json");
        assert!(close(
            &cal_floor["im_window_floor"]["floor"],
            4.0 * want.w_im
        ));
        assert_eq!(
            cal_floor["im_window_floor"]["configured"].as_f64(),
            Some(cfg.im_window_min)
        );
        let i = (0..n).find(|&i| pred[i].is_some()).unwrap();
        assert!((f_hi[i].unwrap() - f_lo[i].unwrap() - 8.0 * want.w_im).abs() < 1e-12);
        // Under a relaxed anchor cut (`with_anchor_cut`) the IM anchors are the relaxed set.
        let relaxed = with_anchor_cut(
            &cfg,
            &AnchorPlan {
                q_train: 0.05,
                n_anchors: n,
                sufficient: true,
            },
        );
        let (cal_relaxed, _) = stage("im_relaxed", true, true, &relaxed);
        let more = anchors_at(0.05);
        assert!(more.len() > anchors.len());
        assert_eq!(cal_relaxed["im_n_train"], more.len());
    }

    /// `seed_decoy_corrected` sizing on a seed whose weaker rows (targets and decoys up to
    /// `im_window_sizing_q`) carry `observed_im`: `w_im` is the decoy-corrected `p_im`
    /// residual of the true hits times the multiplier, wider than the anchors' window, and
    /// `cal.json` says so, while `anchors` keeps the anchors' window. With the seed's own cut
    /// (`im_anchor_q`) a relaxed anchor cut adds no IM anchors from those rows.
    #[test]
    fn seed_decoy_corrected_sizing_widens_w_im_from_the_true_hits() {
        use mumdia_core::config::ImWindowSizing;
        use mumdia_core::constants::{ccs_to_im, im_to_ccs};
        // 600 confident anchors 0.004 off their calibrated 1/K0; at q 0.05, 300 true targets
        // 0.02 off, and 100 false targets spread as the 100 decoys are.
        let (n_anchor, n_wide, n_decoy) = (600usize, 300usize, 100usize);
        let n = n_anchor + n_wide + 100 + n_decoy;
        let z: Vec<i32> = (0..n).map(|i| if i % 3 == 0 { 3 } else { 2 }).collect();
        let mz: Vec<f64> = (0..n).map(|i| 400.0 + i as f64 * 0.5).collect();
        let pred: Vec<f64> = (0..n).map(|i| 0.7 + i as f64 * 0.0005).collect();
        let offset = |i: usize| -> f64 {
            let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
            if i < n_anchor {
                sign * 0.004
            } else if i < n_anchor + n_wide {
                sign * 0.02
            } else {
                ((i * 37) % 201) as f64 / 1000.0 - 0.1
            }
        };
        let obs: Vec<Option<f64>> = (0..n)
            .map(|i| {
                let (a, b) = if z[i] == 2 {
                    (10.0, 1.02)
                } else {
                    (-5.0, 0.98)
                };
                let truth = ccs_to_im(a + b * im_to_ccs(pred[i], mz[i], z[i]), mz[i], z[i]);
                Some(truth + offset(i))
            })
            .collect();
        let q: Vec<f64> = (0..n)
            .map(|i| if i < n_anchor { 0.001 } else { 0.05 })
            .collect();
        let label: Vec<String> = (0..n)
            .map(|i| if i >= n - n_decoy { "decoy" } else { "target" }.to_string())
            .collect();
        let irt: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let prec = scratch("sdc_prec.parquet");
        let seed = scratch("sdc_seed.parquet");
        write_table(
            &prec,
            vec![
                Col::U32("candidate_id".into(), (0..n as u32).collect()),
                Col::F32("predicted_irt".into(), irt.clone()),
                Col::F64("precursor_mz".into(), mz.clone()),
                Col::I32("charge".into(), z.clone()),
                Col::OptF64(
                    "predicted_im".into(),
                    pred.iter().map(|&p| Some(p)).collect(),
                ),
            ],
        )
        .unwrap();
        write_table(
            &seed,
            vec![
                Col::U32("candidate_id".into(), (0..n as u32).collect()),
                Col::U32("base_peptide_id".into(), (0..n as u32).collect()),
                Col::F64("spectrum_q".into(), q),
                Col::F64("score".into(), (0..n).map(|i| 1.0 + i as f64).collect()),
                Col::F64(
                    "observed_rt".into(),
                    irt.iter().map(|&x| 60.0 + 30.0 * x as f64).collect(),
                ),
                Col::Str("label".into(), label),
                Col::F64("precursor_mz".into(), mz),
                Col::I32("charge".into(), z),
                Col::OptF64("observed_im".into(), obs),
            ],
        )
        .unwrap();
        let run = |tag: &str, cfg: &RtImTrainConfig, im_anchor_q: Option<f64>| {
            let cal = scratch(&format!("sdc_cal_{tag}.json"));
            run_in_memory(RtImTrainParams {
                precursor_span: None,
                seed_psms: &seed,
                library_precursors: &prec,
                out_windows: &scratch(&format!("sdc_windows_{tag}.parquet")),
                out_cal: &cal,
                cfg,
                config_hash: "test",
                im_window_floor: None,
                im_anchor_q,
                anchor_irt_from_seed: false,
            })
            .unwrap();
            let v: serde_json::Value = mumdia_io::json::read_json(&cal).unwrap();
            v
        };
        let cfg = RtImTrainConfig::default();
        let anchors_cfg = RtImTrainConfig {
            im_window_sizing: ImWindowSizing::Anchors,
            ..cfg.clone()
        };
        let sized = run("seed", &cfg, Some(0.01));
        let plain = run("anchors", &anchors_cfg, Some(0.01));
        assert_eq!(plain["w_im_sizing"], "holdout");
        assert!(plain.get("im_window_sizing").is_none(), "{plain}");
        assert_eq!(sized["w_im_sizing"], "seed_decoy_corrected");
        let block = &sized["im_window_sizing"];
        assert_eq!(block["applied"], true);
        assert_eq!(block["seed_targets"], n - n_decoy);
        assert_eq!(block["seed_decoys"], n_decoy);
        // The true hits' p95 lies in the 0.02 group; the plain p95 of every target, false
        // ones included, would lie among the false targets' spread (about 0.05).
        let est = block["decoy_corrected_p_im"].as_f64().unwrap();
        assert!((0.015..0.025).contains(&est), "{est}");
        let w = sized["w_im"].as_f64().unwrap();
        assert!(
            (w - est * cfg.im_window_multiplier).abs() <= 1e-9 * w,
            "{w} {est}"
        );
        assert!(w > plain["w_im"].as_f64().unwrap());
        // The anchors are the 600 confident targets either way, and a relaxed cut adds the
        // weaker targets only without the seed's own cut.
        assert_eq!(sized["im_n_train"], n_anchor);
        assert_eq!(plain["im_n_train"], n_anchor);
        let relaxed = with_anchor_cut(
            &cfg,
            &AnchorPlan {
                q_train: 0.1,
                n_anchors: n,
                sufficient: true,
            },
        );
        assert_eq!(run("pinned", &relaxed, Some(0.01))["im_n_train"], n_anchor);
        assert_eq!(run("free", &relaxed, None)["im_n_train"], n - n_decoy);
    }

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
    /// row-aligned range must still get the hash join, with its last-row-wins meaning. The
    /// 1/K0 is read through the same row, so it joins as the iRT does.
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
        let im: Vec<f64> = vec![0.9, f64::NAN, 1.1, 0.7, 1.3];
        for cid in &cases {
            let join = CidJoin::new(cid);
            let aligned = cid.iter().enumerate().all(|(i, &c)| c as usize == i);
            assert_eq!(matches!(join, CidJoin::Dense(_)), aligned, "{cid:?}");
            let want = reference(cid, &irt);
            let want_im = reference(cid, &im.iter().map(|&v| v as f32).collect::<Vec<_>>());
            for c in 0..20u32 {
                let got = join.irt(c, &irt).map(f64::to_bits);
                assert_eq!(got, want.get(&c).map(|v| v.to_bits()), "{cid:?} id {c}");
                let got_im = join.row(c).map(|r| (im[r] as f32).to_bits());
                assert_eq!(
                    got_im,
                    want_im.get(&c).map(|v| (*v as f32).to_bits()),
                    "{cid:?} id {c}"
                );
            }
        }
        assert_eq!(CidJoin::Unused.irt(0, &irt), None);
        // An empty library joins nothing.
        assert_eq!(CidJoin::new(&[]).irt(0, &[]), None);
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

    /// Bit patterns of a window set's 1/K0 fields; `None` when it has none.
    #[allow(clippy::type_complexity)]
    fn im_bits(w: &RtWindows) -> Option<Vec<(u32, u32, Option<u64>)>> {
        let (lo, hi, cal) = (w.im_lo.as_ref()?, w.im_hi.as_ref()?, w.im_cal.as_ref()?);
        Some(
            (0..w.len())
                .map(|c| (lo[c].to_bits(), hi[c].to_bits(), cal[c].map(f64::to_bits)))
                .collect(),
        )
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
                im_window_floor: None,
                im_anchor_q: None,
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
            // A 3D run holds no 1/K0 window either way.
            assert_eq!(im_bits(&kept), None, "{tag}");
            assert_eq!(im_bits(&from_file), None, "{tag}");
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
        let mut b = RtWindowsBuilder::new(4, false);
        for &(c, cal, lo, hi) in &rows {
            b.row(c, cal, lo, hi);
        }
        let kept = b.finish("lib", "rw").expect("no NaN bound");
        let read = crate::stages::extract::read_run_windows(&path, 4).unwrap();
        assert_eq!(bits(&kept), bits(&read));
        assert_eq!(kept.rt_cal[2], 11.0);
        assert!(kept.rt_cal[1].is_nan() && kept.rt_lo[3] == f64::NEG_INFINITY);

        write(&path, &[(1u32, 3.0, f64::NAN, 4.0)]);
        let mut b = RtWindowsBuilder::new(4, false);
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
        let mut b = RtWindowsBuilder::new(1, false);
        b.row(1, 3.0, f64::NAN, 4.0);
        assert!(b.finish("lib", "rw").is_some());
    }

    /// The 1/K0 half of the scatter: a null window stays unbounded, a repeated id takes the
    /// last row, a NaN 1/K0 bound withholds the in-memory form and is refused from the file
    /// with the row named, and a table whose 1/K0 columns are all null holds no 1/K0 window,
    /// in memory or read back.
    #[test]
    fn the_windows_builder_scatters_the_im_columns_as_extract_reads_them() {
        type Row = (u32, Option<f64>, Option<f64>, Option<f64>);
        let write = |path: &str, rows: &[Row]| {
            write_table(
                path,
                vec![
                    Col::U32("candidate_id".into(), rows.iter().map(|r| r.0).collect()),
                    Col::F64("rt_pred_cal".into(), vec![10.0; rows.len()]),
                    Col::F64("rt_lo".into(), vec![5.0; rows.len()]),
                    Col::F64("rt_hi".into(), vec![15.0; rows.len()]),
                    Col::OptF64("im_pred_cal".into(), rows.iter().map(|r| r.1).collect()),
                    Col::OptF64("im_lo".into(), rows.iter().map(|r| r.2).collect()),
                    Col::OptF64("im_hi".into(), rows.iter().map(|r| r.3).collect()),
                ],
            )
            .unwrap();
        };
        let build = |rows: &[Row]| {
            let mut b = RtWindowsBuilder::new(4, true);
            for &(c, cal, lo, hi) in rows {
                b.row(c, 10.0, 5.0, 15.0);
                b.im_row(c, cal, lo, hi);
            }
            b.finish("lib", "rw")
        };
        let rows: Vec<Row> = vec![
            (0, Some(0.9), Some(0.88), Some(0.92)),
            (2, None, None, None),
            (3, Some(1.0), Some(0.97), Some(1.03)),
            (3, Some(1.1), Some(1.07), Some(1.13)), // repeated: this one wins
            (8, Some(1.2), Some(1.1), Some(1.3)),   // past the library
        ];
        let path = scratch("builder_im_windows.parquet");
        write(&path, &rows);
        let kept = build(&rows).expect("no NaN bound");
        let read = crate::stages::extract::read_run_windows(&path, 4).unwrap();
        assert_eq!(bits(&kept), bits(&read));
        assert_eq!(im_bits(&kept), im_bits(&read));
        let lo = kept.im_lo.as_ref().unwrap();
        assert_eq!(lo[3], 1.07f32);
        assert_eq!(lo[1], f32::NEG_INFINITY, "a candidate without a row");
        assert_eq!(lo[2], f32::NEG_INFINITY, "a row without a 1/K0 window");
        assert!(kept.has_im());
        // The gate: inside the window, outside it, and a candidate without one.
        assert!(kept.admits(0, 10.0, Some(0.9)));
        assert!(!kept.admits(0, 10.0, Some(0.95)));
        assert!(kept.admits(0, 10.0, None), "no 1/K0 test asked for");
        assert!(
            !kept.admits(0, 20.0, Some(0.9)),
            "the RT test still applies"
        );
        assert!(kept.admits(2, 10.0, Some(5.0)), "unbounded in 1/K0");

        // All null: no 1/K0 window in memory or read back.
        let nulls: Vec<Row> = vec![(0, None, None, None), (1, None, None, None)];
        write(&path, &nulls);
        let kept = build(&nulls).expect("no NaN bound");
        let read = crate::stages::extract::read_run_windows(&path, 4).unwrap();
        assert_eq!(im_bits(&kept), None);
        assert_eq!(im_bits(&read), None);
        assert!(!read.has_im());
        assert!(read.admits(0, 10.0, Some(9.0)), "no window, no gate");

        // A NaN bound.
        let bad: Vec<Row> = vec![(1, Some(1.0), Some(f64::NAN), Some(1.1))];
        write(&path, &bad);
        assert!(build(&bad).is_none(), "a NaN bound must not be handed over");
        let err = crate::stages::extract::read_run_windows(&path, 4)
            .err()
            .unwrap();
        assert!(err.to_string().contains("NaN 1/K0 bound"), "{err}");
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

    /// `windows_unbounded` is the rule `window_plan` applies, for every anchor count and
    /// every `min_anchors`, so the grouped run's refusal (`run_groups`) refuses exactly the
    /// fits that `rt-im-train` would give unbounded windows.
    #[test]
    fn windows_unbounded_is_the_window_plan_rule() {
        assert!(windows_unbounded(0));
        assert!(windows_unbounded(1));
        assert!(!windows_unbounded(2));
        assert!(!windows_unbounded(10_000));
        for min_anchors in [2usize, 3, 20, 50] {
            for n_train in 0..=60 {
                assert_eq!(
                    windows_unbounded(n_train),
                    window_plan(n_train, min_anchors, 120.0) == WindowPlan::Unbounded,
                    "n_train {n_train}, min_anchors {min_anchors}"
                );
            }
        }
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
    fn composition_features_and_huber_fit() {
        let f = composition_features("AAGLATM[Oxidation]ISTHKR", 3);
        assert_eq!(f, [1.0, 1.0, 0.0, 13.0, 1.0, 1.0, 1.0]);
        let g = composition_features("C(UniMod:4)PEPTIDEK", 4);
        assert_eq!(g, [1.0, 0.0, 1.0, 9.0, 0.0, 0.0, 0.0]);
        // y = -3 * charge3 - 6 * charge4 + 0.2 * length, with two gross outliers
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..90usize {
            let z = 2 + (i % 3) as i32;
            let len = 7 + (i % 13);
            let row = [
                1.0,
                f64::from(u8::from(z == 3)),
                f64::from(u8::from(z >= 4)),
                len as f64,
                (i % 2) as f64,
                ((i / 2) % 2) as f64,
                ((i / 4) % 2) as f64,
            ];
            let mut v = -3.0 * row[1] - 6.0 * row[2] + 0.2 * row[3];
            if i == 10 || i == 50 {
                v += 60.0;
            }
            x.push(row);
            y.push(v);
        }
        let b = huber_fit(&x, &y).unwrap();
        // a column carried by too few anchors is not fitted
        let mut xs = x.clone();
        for (i, r) in xs.iter_mut().enumerate() {
            r[2] = f64::from(u8::from(i < 3)); // only 3 "charge 4" anchors
        }
        let bs = huber_fit(&drop_sparse_columns(xs, MIN_ANCHORS_PER_TERM), &y).unwrap();
        assert!(bs[2].abs() < 1e-3, "{bs:?}");
        assert!(
            (b[1] + 3.0).abs() < 0.1 && (b[2] + 6.0).abs() < 0.1 && (b[3] - 0.2).abs() < 0.02,
            "{b:?}"
        );
    }

    #[test]
    fn the_local_window_widens_only_the_side_and_region_whose_anchors_need_it() {
        // 400 anchors over calibrated RT 0..400 with symmetric +-3 s residuals, except the
        // last 50 (RT > 350), which elute 10-30 s late: the order-losing gradient end.
        let n = 400;
        let cal: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let resid: Vec<f64> = (0..n)
            .map(|i| {
                if i >= 350 {
                    10.0 + ((i * 7) % 21) as f64
                } else {
                    ((i * 13) % 7) as f64 - 3.0
                }
            })
            .collect();
        let w_rt = 4.5;
        let lw = LocalWindow::fit(&cal, &resid, 40, 0.9, 1.0, w_rt, 120.0).unwrap();
        // mid-gradient: both sides stay the global width (widen-only floor)
        let (l, r) = lw.at(150.0);
        assert_eq!((l, r), (w_rt, w_rt));
        // gradient end: the late side widens to the anchors' upper quantile, the early side not
        let (l, r) = lw.at(399.0);
        assert_eq!(l, w_rt);
        assert!(r > 20.0, "late side {r}");
        // beyond the grid the end values are held
        assert_eq!(lw.at(1e6), lw.at(399.0));
        // the cap binds
        let capped = LocalWindow::fit(&cal, &resid, 40, 0.9, 1.0, w_rt, 12.0).unwrap();
        assert_eq!(capped.at(399.0).1, 12.0);
    }

    #[test]
    fn a_seed_short_of_anchors_relaxes_the_cut_and_then_reports_short() {
        // 60 target base peptides: 10 at q 0.005, 25 more at 0.015, 25 more at 0.04, plus
        // decoys at every cut, which never count. Duplicate rows of one peptide count once.
        let seed = scratch("plan_anchors_seed.parquet");
        let mut base = Vec::new();
        let mut q = Vec::new();
        let mut label = Vec::new();
        for i in 0..60u32 {
            let qi = if i < 10 {
                0.005
            } else if i < 35 {
                0.015
            } else {
                0.04
            };
            for (l, b) in [("target", i), ("target", i), ("decoy", 1000 + i)] {
                base.push(b);
                q.push(qi);
                label.push(l.to_string());
            }
        }
        let n = base.len();
        write_table(
            &seed,
            vec![
                Col::U32("base_peptide_id".into(), base),
                Col::F64("spectrum_q".into(), q),
                Col::F64("score".into(), vec![1.0; n]),
                Col::F64("observed_rt".into(), vec![100.0; n]),
                Col::Str("label".into(), label),
            ],
        )
        .unwrap();
        let plan = |need: usize, ladder: Vec<f64>| {
            let cfg = RtImTrainConfig {
                min_seed_for_calibration: need,
                anchor_q_ladder: ladder,
                ..Default::default()
            };
            plan_anchors(&seed, &cfg).unwrap()
        };
        let p = plan(10, vec![0.02, 0.05]);
        assert_eq!((p.q_train, p.n_anchors, p.sufficient), (0.01, 10, true));
        let p = plan(30, vec![0.02, 0.05]);
        assert_eq!((p.q_train, p.n_anchors, p.sufficient), (0.02, 35, true));
        let p = plan(50, vec![0.02, 0.05]);
        assert_eq!((p.q_train, p.n_anchors, p.sufficient), (0.05, 60, true));
        let p = plan(61, vec![0.02, 0.05]);
        assert_eq!((p.q_train, p.n_anchors, p.sufficient), (0.05, 60, false));
        // An empty ladder keeps the strict cut; a rung below q_train is ignored.
        let p = plan(30, vec![]);
        assert_eq!((p.q_train, p.n_anchors, p.sufficient), (0.01, 10, false));
        let p = plan(30, vec![0.005, 0.02]);
        assert_eq!((p.q_train, p.n_anchors, p.sufficient), (0.02, 35, true));
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
        let local = RtImTrainConfig {
            local_window_anchors: 60,
            ..Default::default()
        };
        let x_filter = RtImTrainConfig {
            anchor_x_filter: true,
            ..Default::default()
        };
        for (tag, cfg, n) in [
            ("default", RtImTrainConfig::default(), 400usize),
            ("holdout", holdout, 400),
            ("adaptive", adaptive, 400),
            ("local", local, 400),
            ("linear", linear, 400),
            ("x_filter", x_filter, 400),
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
                im_window_floor: None,
                im_anchor_q: None,
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
                    im_window_floor: None,
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

    /// `run_in_memory_checked` hands the fit to its check before anything is written. A check
    /// that refuses leaves neither the windows nor `cal.json`, and one that accepts writes what
    /// `run` writes. The per-group refusal of unbounded windows in `run_groups` rests on this.
    #[test]
    fn a_refused_fit_check_writes_nothing_and_an_accepted_one_writes_what_run_writes() {
        fn params<'a>(
            seed: &'a str,
            lib: &'a str,
            windows: &'a str,
            cal: &'a str,
            cfg: &'a RtImTrainConfig,
        ) -> RtImTrainParams<'a> {
            RtImTrainParams {
                precursor_span: None,
                seed_psms: seed,
                library_precursors: lib,
                out_windows: windows,
                out_cal: cal,
                cfg,
                config_hash: "h",
                im_window_floor: None,
                im_anchor_q: None,
                anchor_irt_from_seed: false,
            }
        }
        let dir = std::env::temp_dir().join(format!("mumdia_rt_checked_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // One target anchor: the decoy and the unconfident row of the helper do not count.
        let (seed, lib) = anchors_and_library(&dir, 1);
        let out = |name: &str| dir.join(name).to_str().unwrap().to_string();
        let cfg = RtImTrainConfig::default();

        let (w_ref, c_ref) = (out("w_refused.parquet"), out("cal_refused.json"));
        let refused = run_in_memory_checked(params(&seed, &lib, &w_ref, &c_ref, &cfg), |fit| {
            assert_eq!(fit.n_train(), 1);
            if windows_unbounded(fit.n_train()) {
                anyhow::bail!("refused: {} anchor", fit.n_train());
            }
            Ok(())
        });
        let Err(err) = refused else {
            panic!("the check's error was not returned");
        };
        assert_eq!(err.to_string(), "refused: 1 anchor");
        assert!(!std::path::Path::new(&w_ref).exists(), "windows written");
        assert!(!std::path::Path::new(&c_ref).exists(), "cal.json written");

        let (w_ok, c_ok) = (out("w_accepted.parquet"), out("cal_accepted.json"));
        let (written, _) =
            run_in_memory_checked(params(&seed, &lib, &w_ok, &c_ok, &cfg), |_| Ok(())).unwrap();
        let (w_run, c_run) = (out("w_run.parquet"), out("cal_run.json"));
        let rows = run(params(&seed, &lib, &w_run, &c_run, &cfg)).unwrap();
        assert_eq!(written.rows, rows);
        assert_eq!(
            std::fs::read(&w_ok).unwrap(),
            std::fs::read(&w_run).unwrap()
        );
        assert_eq!(
            std::fs::read(&c_ok).unwrap(),
            std::fs::read(&c_run).unwrap()
        );
        let cal: serde_json::Value = mumdia_io::json::read_json(&c_ok).unwrap();
        assert_eq!(cal["calibration_status"], INSUFFICIENT_ANCHORS_STATUS);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `seed_anchor_bound` counts what `plan_anchors` counts at `q_train`, and the fit never
    /// keeps more: it also drops anchors without a finite iRT. The grouped run's early
    /// refusal of unbounded windows rests on this bound, so it can never refuse a seed whose
    /// fit would have bounded windows.
    #[test]
    fn the_seed_anchor_bound_counts_what_plan_anchors_counts_and_bounds_the_fit() {
        let dir = std::env::temp_dir().join(format!("mumdia_rt_bound_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Six target anchors at q 0.001; the decoy and the row at q 0.5 do not count.
        let (seed, lib) = anchors_and_library(&dir, 6);
        let cfg = RtImTrainConfig {
            anchor_q_ladder: Vec::new(),
            ..RtImTrainConfig::default()
        };
        let bound = seed_anchor_bound(&seed, cfg.q_train).unwrap();
        assert_eq!(bound, 6);
        assert_eq!(bound, plan_anchors(&seed, &cfg).unwrap().n_anchors);
        // A cut at or below every anchor's q leaves none: the comparison is strict, as in
        // `fit_anchors`.
        assert_eq!(seed_anchor_bound(&seed, 0.001).unwrap(), 0);
        assert_eq!(seed_anchor_bound(&seed, 1.0).unwrap(), 7);
        // The seed's own iRT is finite for every anchor: the shared fit keeps all six.
        assert_eq!(fit_from_seed(&seed, &cfg).unwrap().n_train(), 6);

        let out = |name: &str| dir.join(name).to_str().unwrap().to_string();
        let fit_on = |lib: &str, tag: &str| -> usize {
            let (w, c) = (
                out(&format!("w_{tag}.parquet")),
                out(&format!("c_{tag}.json")),
            );
            let mut n_train = usize::MAX;
            run_in_memory_checked(
                RtImTrainParams {
                    precursor_span: None,
                    seed_psms: &seed,
                    library_precursors: lib,
                    out_windows: &w,
                    out_cal: &c,
                    cfg: &cfg,
                    config_hash: "h",
                    im_window_floor: None,
                    im_anchor_q: None,
                    anchor_irt_from_seed: false,
                },
                |fit| {
                    n_train = fit.n_train();
                    Ok(())
                },
            )
            .unwrap();
            n_train
        };
        // The helper's library has no finite iRT for candidate 3: the fit keeps five.
        assert_eq!(fit_on(&lib, "lib"), 5);
        // A library with a finite iRT for one anchor only: the bound is six, the fit has one,
        // which is the case the grouped run's second, exact check exists for.
        let sparse = out("lib_sparse.parquet");
        write_table(
            &sparse,
            vec![
                Col::U32("candidate_id".into(), (0..11u32).collect()),
                Col::F32(
                    "predicted_irt".into(),
                    (0..11)
                        .map(|i| if i == 0 { 1.0 } else { f32::NAN })
                        .collect(),
                ),
            ],
        )
        .unwrap();
        let n_sparse = fit_on(&sparse, "sparse");
        assert_eq!(n_sparse, 1);
        assert!(windows_unbounded(n_sparse) && !windows_unbounded(bound));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `anchor_x_filter` drops the anchors with an absurd predicted iRT before the fit, the
    /// window sizing and the residuals use them, so the windows are those of the clean
    /// anchors alone and cal.json records how many it dropped. Off, every anchor is fitted
    /// and cal.json has no such key. A grouped run's shared fit is filtered the same way.
    #[test]
    fn the_anchor_x_filter_fits_the_clean_anchors_and_records_the_count() {
        let dir = std::env::temp_dir().join(format!("mumdia_rt_xfilter_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = |name: &str| dir.join(name).to_str().unwrap().to_string();
        // A curved 2000-3500 s gradient (a straight line survives any grid) plus three
        // anchors whose predicted iRT is absurd, as a multi-head refit produced on HYE.
        let n = 400usize;
        let f = |x: f64| 2000.0 + (x - 2000.0).powi(2) / 1500.0;
        let mut irt: Vec<f32> = (0..n).map(|i| 2000.0 + 3.75 * i as f32).collect();
        let mut rt: Vec<f64> = irt
            .iter()
            .enumerate()
            .map(|(i, x)| f(*x as f64) + ((i * 7919) % 11) as f64 - 5.0)
            .collect();
        irt.extend([-170_000.0, 375_000.0, 90_000.0]);
        rt.extend([2500.0, 3400.0, 3000.0]);
        let seed = |name: &str, m: usize| {
            let p = path(name);
            write_table(
                &p,
                vec![
                    Col::U32("candidate_id".into(), (0..m as u32).collect()),
                    Col::U32("base_peptide_id".into(), (0..m as u32).collect()),
                    Col::F64("spectrum_q".into(), vec![0.001; m]),
                    Col::F64("score".into(), (0..m).map(|i| 10.0 + i as f64).collect()),
                    Col::F64("observed_rt".into(), rt[..m].to_vec()),
                    Col::Str("label".into(), vec!["target".to_string(); m]),
                    Col::F32("predicted_irt".into(), irt[..m].to_vec()),
                ],
            )
            .unwrap();
            p
        };
        let dirty = seed("dirty.parquet", n + 3);
        let clean = seed("clean.parquet", n);
        let lib = path("lib.parquet");
        write_table(
            &lib,
            vec![
                Col::U32("candidate_id".into(), (0..600u32).collect()),
                Col::F32(
                    "predicted_irt".into(),
                    (0..600).map(|i| 1900.0 + 3.0 * i as f32).collect(),
                ),
            ],
        )
        .unwrap();
        let filtered = RtImTrainConfig {
            anchor_x_filter: true,
            ..Default::default()
        };
        let stage = |seed: &str, cfg: &RtImTrainConfig, tag: &str| {
            let (windows, cal) = (
                path(&format!("w_{tag}.parquet")),
                path(&format!("cal_{tag}.json")),
            );
            run(RtImTrainParams {
                precursor_span: None,
                seed_psms: seed,
                library_precursors: &lib,
                out_windows: &windows,
                out_cal: &cal,
                cfg,
                config_hash: "h",
                im_window_floor: None,
                im_anchor_q: None,
                anchor_irt_from_seed: true,
            })
            .unwrap();
            let cal: serde_json::Value = mumdia_io::json::read_json(&cal).unwrap();
            (std::fs::read(&windows).unwrap(), cal)
        };
        let (w_clean, cal_clean) = stage(&clean, &RtImTrainConfig::default(), "clean");
        let (w_plain, cal_plain) = stage(&dirty, &RtImTrainConfig::default(), "plain");
        let (w_filtered, cal_filtered) = stage(&dirty, &filtered, "filtered");
        let (w_clean_on, cal_clean_on) = stage(&clean, &filtered, "clean_on");
        assert_eq!(cal_plain["n_train"], n + 3);
        assert!(cal_plain.get("n_rt_outliers_removed").is_none());
        assert!(cal_clean.get("n_rt_outliers_removed").is_none());
        assert_ne!(w_plain, w_clean, "the absurd anchors distort the plain fit");
        assert_eq!(cal_filtered["n_rt_outliers_removed"], 3);
        assert_eq!(cal_filtered["n_train"], n);
        assert_eq!(cal_filtered["w_rt"], cal_clean["w_rt"]);
        assert_eq!(
            w_filtered, w_clean,
            "the filtered fit is the fit on the clean anchors"
        );
        // On clean anchors the filter removes nothing and changes nothing, cal.json included.
        assert_eq!(w_clean_on, w_clean);
        assert_eq!(cal_clean_on, cal_clean);
        let fit = fit_from_seed(&dirty, &filtered).unwrap();
        assert_eq!(fit.n_train(), n);
        assert_eq!(fit.n_rt_outliers_removed, 3);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
