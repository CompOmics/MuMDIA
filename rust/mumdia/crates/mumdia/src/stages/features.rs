//! Stage E `mumdia features` (docs/10_features.md).
//! Reads psms_extracted + chromatograms (+ MS1 apex isotopes carried on the
//! PSM rows) and computes a fixed, named, versioned feature vector per PSM.
//! The active feature set is config-driven (`minimal` or `rich`); its ordered
//! list is hashed into a `classifier_feature_schema_id` and written to a
//! companion `<features>.schema.json` so the classifier input is reproducible
//! and never applied under a mismatched set (docs/02_config_and_data_model.md).

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{anyhow, Context as _, Result};
use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, StringArray, UInt32Array};
use arrow::record_batch::RecordBatch;
use mumdia_core::config::{FeatureSet, FeaturesConfig};
use mumdia_core::constants::{ppm_diff, PROTON};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{Col, ListF32, TableFile, TableWriter};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, warn};

use crate::calibrate::percentile;
use crate::stats::{cosine, pearson, pearson_pairs, pearson_vs, spectral_angle, Centered};
use rayon::prelude::*;

// Extended feature battery (FeatureSet::Extended). One module per family; each
// exposes `NAMES: &[&str]` and `values(&Evidence) -> Vec<f64>` in matching order
// and length. The families are DIA-NN / OpenSWATH / AlphaDIA / MS2Rescore /
// OktoberFest analogs plus novel families. Kept separate so they can be built
// and reviewed independently; the registry below concatenates them in a fixed
// order that defines the extended schema.
mod apex_dispersion;
mod chromatographic;
mod coelution;
mod demix;
pub(crate) mod entropy;
mod interference;
mod ion_series;
mod mass_accuracy;
mod mass_uncertainty;
mod ms1;
mod nonzero;
mod novel;
mod order_consistency;
mod peak_scans;
mod rt;
mod similarity;

type FamilyFn = fn(&Evidence) -> Vec<f64>;

/// Ordered family registry. Order is part of the frozen feature schema; append
/// only. Each entry is (feature names, value function).
const FAMILIES: &[(&[&str], FamilyFn)] = &[
    (similarity::NAMES, similarity::values),
    (entropy::NAMES, entropy::values),
    (coelution::NAMES, coelution::values),
    (interference::NAMES, interference::values),
    (chromatographic::NAMES, chromatographic::values),
    (mass_accuracy::NAMES, mass_accuracy::values),
    (ion_series::NAMES, ion_series::values),
    (ms1::NAMES, ms1::values),
    (rt::NAMES, rt::values),
    (novel::NAMES, novel::values),
    (nonzero::NAMES, nonzero::values),
    (order_consistency::NAMES, order_consistency::values),
    (peak_scans::NAMES, peak_scans::values),
    (apex_dispersion::NAMES, apex_dispersion::values),
    (mass_uncertainty::NAMES, mass_uncertainty::values),
    (demix::NAMES, demix::values),
];

/// Names already used by the Minimal/Rich sets, which the extended battery must
/// not shadow (a colliding extended feature is dropped, keeping the legacy one).
fn reserved_names() -> std::collections::HashSet<&'static str> {
    MINIMAL_FEATURES
        .iter()
        .chain(RICH_EXTRA.iter())
        .copied()
        .collect()
}

/// Extended-battery feature names as `&'static str`, registry order, globally
/// deduplicated: a name that repeats across families, or collides with a
/// Minimal/Rich name, is kept only on first appearance. This makes the schema
/// robust to independently-authored family modules reusing a name.
fn extended_name_refs() -> Vec<&'static str> {
    let reserved = reserved_names();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (names, _) in FAMILIES {
        for &n in *names {
            if reserved.contains(n) {
                continue;
            }
            if seen.insert(n) {
                out.push(n);
            }
        }
    }
    out
}

/// Extended-battery feature names in registry order (deduplicated).
pub fn extended_names() -> Vec<String> {
    extended_name_refs().iter().map(|s| s.to_string()).collect()
}

/// Precomputed dedup plan for the extended battery: for each family, in registry
/// order, the local value indices that survive the reserved-name and global
/// first-appearance filter. This mask is a compile-time-static invariant, so it
/// is computed once (not rebuilt per PSM). It reproduces exactly the same
/// survivors, in the same order, as the reserved + `seen` filter below and as
/// [`extended_name_refs`], keeping names and values in lockstep.
fn extended_value_plan() -> &'static Vec<Vec<usize>> {
    static PLAN: std::sync::OnceLock<Vec<Vec<usize>>> = std::sync::OnceLock::new();
    PLAN.get_or_init(|| {
        let reserved = reserved_names();
        let mut seen = std::collections::HashSet::new();
        let mut plan = Vec::with_capacity(FAMILIES.len());
        for (names, _) in FAMILIES {
            let mut keep = Vec::new();
            for (i, &n) in names.iter().enumerate() {
                if reserved.contains(n) {
                    continue;
                }
                if seen.insert(n) {
                    keep.push(i);
                }
            }
            plan.push(keep);
        }
        plan
    })
}

/// Compute the full extended battery for one PSM, in the same deduplicated order
/// as [`extended_name_refs`]. Each family returns exactly `NAMES.len()` values;
/// the registry applies the identical dedup predicate to values so names and
/// values stay in lockstep.
fn extended_values(e: &Evidence) -> Vec<f64> {
    let plan = extended_value_plan();
    let mut out = Vec::with_capacity(plan.iter().map(|keep| keep.len()).sum());
    for ((names, f), keep) in FAMILIES.iter().zip(plan) {
        let vals = f(e);
        debug_assert_eq!(
            vals.len(),
            names.len(),
            "extended feature family returned {} values for {} names",
            vals.len(),
            names.len()
        );
        for &i in keep {
            out.push(if vals[i].is_finite() { vals[i] } else { 0.0 });
        }
    }
    out
}

/// The minimal feature set (docs/10_features.md).
pub const MINIMAL_FEATURES: &[&str] = &[
    "rt_error_abs",
    "rt_error_rel",
    "n_matched_fragments",
    "coelution_run",
    "log_apex_intensity",
    "frag_corr",
    "frag_cosine",
    "spectral_angle",
    "coelution_mean",
    "coelution_best",
    "n_coelution_above",
    "charge",
    "peptide_length",
    "n_proteins",
];

/// Additional features for the `rich`/`standard` set (docs/10_features.md).
pub const RICH_EXTRA: &[&str] = &[
    "library_norm_manhattan",
    "library_rmsd",
    "xcorr_coelution",
    "xcorr_shape",
    "sum_b_intensity",
    "sum_y_intensity",
    "diff_by_intensity",
    "n_b_ions",
    "n_y_ions",
    "weighted_mass_error",
    "mean_mass_error",
    "isotope_corr",
    "ms1_isom1_ratio",
    "log_mono_ms1",
    "has_ms1",
    "log_sn",
    "n_observations",
    "base_width_rt",
    "seed_score",
    "seed_identified",
    "matched_fraction",
    "profile_cos",
    "ref_corr",
    "best_ref_corr",
    "low_frag_coel",
    "evidence",
    "contrast_min",
    "resid_corr",
    "coel_clean",
    "shadow_frac",
];

/// The ordered active feature list for the configured set.
pub fn active_features(set: FeatureSet) -> Vec<String> {
    let mut v: Vec<String> = MINIMAL_FEATURES.iter().map(|s| s.to_string()).collect();
    if matches!(set, FeatureSet::Rich | FeatureSet::Extended) {
        v.extend(RICH_EXTRA.iter().map(|s| s.to_string()));
    }
    if matches!(set, FeatureSet::Extended) {
        v.extend(extended_names());
        // psms-derived (not an Evidence family): the co-elution peak-contest metrics.
        // A peak-borrowing decoy loses most contested intensity/fragments to the real
        // co-eluting peptide, so these three separate borrowers from genuine IDs.
        v.push("peak_contested_frac".to_string());
        v.push("peak_contested_count_frac".to_string());
        v.push("peak_apportioned_frac".to_string());
        // Cross-candidate charge-state corroboration (aggregated across the charge
        // states of one peptidoform, not visible to the per-PSM Evidence families):
        // a real peptide co-occurs at multiple charges more than a shift decoy.
        v.push("n_charge_states".to_string());
        v.push("charge_multi_flag".to_string());
        v.push("cross_charge_intensity_log".to_string());
    }
    v
}

pub fn feature_schema_id(cols: &[String]) -> String {
    mumdia_io::hash::blake3_str(&cols.join(","))
}

/// Companion schema record written next to features.parquet and carried forward.
#[derive(Serialize, Deserialize)]
pub struct FeatureSchema {
    pub feature_columns: Vec<String>,
    pub schema_id: String,
}

impl FeatureSchema {
    pub fn read(artifact_path: &str) -> Result<FeatureSchema> {
        let companion = format!("{artifact_path}.schema.json");
        match mumdia_io::json::read_json::<FeatureSchema>(&companion) {
            Ok(s) => Ok(s),
            Err(e) => {
                // The companion is a convenience, not the source of truth: the feature
                // column list is recoverable from the parquet's own schema (every column
                // that is not one of the fixed metadata columns). A missing/corrupt
                // companion used to abort the run outright -- observed this session when a
                // competed table was rewritten by an external tool that did not know to
                // copy the sidecar. Reconstruct instead, and say so.
                // Footer only: the column list lives in the parquet metadata, so there is
                // no reason to decode ~390 columns of data to read their names.
                let names = mumdia_io::table::column_names(artifact_path).with_context(|| {
                    format!(
                        "reading {companion} failed ({e}) and the artifact itself could                          not be read to reconstruct the feature schema"
                    )
                })?;
                let feature_columns: Vec<String> = names
                    .into_iter()
                    .filter(|c| !NON_FEATURE_COLUMNS.contains(&c.as_str()))
                    .collect();
                if feature_columns.is_empty() {
                    anyhow::bail!(
                        "reading {companion} failed ({e}) and {artifact_path} contains no \
                         feature columns to reconstruct it from"
                    );
                }
                tracing::warn!(
                    companion = %companion,
                    n_features = feature_columns.len(),
                    "feature schema companion unreadable; reconstructed the feature list \
                     from the artifact's own parquet schema"
                );
                // schema_id is provenance only; mark it as reconstructed rather than
                // inventing a hash that would collide with a real one.
                Ok(FeatureSchema {
                    feature_columns,
                    schema_id: "reconstructed-from-parquet".to_string(),
                })
            }
        }
    }
}

/// Columns of a competed/features artifact that are metadata, not rescoring features.
/// Used to reconstruct a feature list when the `.schema.json` companion is missing.
///
/// Verified against a real artifact: excluding exactly these reproduces the recorded
/// `feature_columns` list byte-for-byte. Two traps this encodes: `charge` IS a feature
/// (carried as an f64), while `elution_lo`/`elution_hi` are peak-bound metadata carried
/// for quantification, not features. Getting either wrong changes the trained population.
pub const NON_FEATURE_COLUMNS: &[&str] = &[
    "candidate_id",
    "peptidoform_id",
    "base_peptide_id",
    "peptidoform",
    "protein",
    "label",
    "precursor_mz",
    "prelim_score",
    "apex_rt",
    "elution_lo",
    "elution_hi",
    "peak_rank",
    "source",
    "unique_evidence",
];

fn peptide_length(peptidoform: &str) -> i32 {
    // Strip the decoy marker so its letters (D,E,C,O,Y) are not counted as residues.
    // seq_len feeds the peptide_length feature and length-normalized features (e.g.
    // mean_matched_ordinal_norm), so a decoy-only +5 offset is a target/decoy label leak.
    let peptidoform = peptidoform.strip_prefix("DECOY_").unwrap_or(peptidoform);
    let mut n = 0;
    let mut in_brackets = false;
    for c in peptidoform.chars() {
        match c {
            '[' => in_brackets = true,
            ']' => in_brackets = false,
            c if c.is_ascii_alphabetic() && !in_brackets => n += 1,
            _ => {}
        }
    }
    n
}

/// Stage B marks unavailable RT calibration as NaN when fewer than two anchors
/// exist. Treat that sentinel as no RT evidence instead of allowing NaN to
/// contaminate the feature matrix or preliminary competition score.
fn calibrated_rt_error(apex_rt: f64, rt_pred_cal: f64) -> f64 {
    if apex_rt.is_finite() && rt_pred_cal.is_finite() {
        (apex_rt - rt_pred_cal).abs()
    } else {
        0.0
    }
}

/// One chromatogram row as the feature code sees it. The trace slices and the fragment
/// name are borrowed from the chunk store ([`ChromChunk`]), which owns one flat buffer
/// per array instead of a `Vec` per row, and shares one RT axis across the rows of a
/// candidate (extract samples every fragment of a candidate on the same window grid).
struct ChromRow<'a> {
    frag_name: &'a str,
    frag_mz: f64,
    frag_obs_mz: f64,
    pred_int: f32,
    rt: &'a [f32],
    inten: &'a [f32],
}

/// One PSM's fragment traces on a single union RT axis: `axis_full` ascending and
/// deduplicated, `traces_full[i]` the intensities of row `i` sampled on it (zero where
/// that row has no point).
///
/// Both feature paths need exactly this, and both used to rebuild it independently, each
/// row through its own `HashMap<u32, f32>` keyed on the RT bit pattern -- two maps per
/// fragment per PSM. It is built once per PSM now and handed to both.
struct TraceAlign {
    axis_full: Vec<f32>,
    traces_full: Vec<Vec<f64>>,
}

/// Build the union alignment of a PSM's rows, by the shared-axis fast path when it
/// applies and by the union-and-map build otherwise.
fn align_traces(rows: &[ChromRow]) -> TraceAlign {
    align_shared_axis(rows).unwrap_or_else(|| align_union(rows))
}

/// Fast path: every row with a trace samples the SAME stored axis, and that axis is
/// strictly ascending. This is the normal case -- [`ChromChunk::axis_for`] stores one
/// axis per candidate because extract samples every fragment of a candidate on the same
/// window grid -- and then the union axis IS that slice and every per-fragment map is
/// the identity, so the sort, the dedup and the maps are all pure overhead.
///
/// Returns `None` unless the precondition holds, and the two builds then agree bit for
/// bit: sorting and deduplicating a strictly ascending axis returns it unchanged, and
/// looking a row's own RT value up in its own map returns that row's intensity at the
/// same position. A row with no trace at all gets zeros either way (its map is empty).
///
/// "Same axis" is decided by VALUE, not by provenance. The pointer check is kept only as
/// the cheap accept for the case the store actually produces; a `ptr::eq` + length test
/// would otherwise make the fast path's correctness depend on an invariant of
/// [`ChromChunk::axis_for`] -- that a shared slice is exactly the row's own trace grid --
/// which nothing here can see, and any future store that handed out a longer shared buffer
/// would move every extended feature of every PSM at once with no test failing. Comparing
/// bit patterns rather than `==` keeps `-0.0` and `+0.0` distinct, so the accepted set is
/// the one the union build reproduces exactly.
fn align_shared_axis(rows: &[ChromRow]) -> Option<TraceAlign> {
    let mut shared: Option<&[f32]> = None;
    for r in rows {
        if r.rt.is_empty() {
            continue;
        }
        // A row whose value count does not match its axis is aligned by the map build,
        // which truncates or zero-fills; that is rare enough not to be worth mirroring.
        if r.rt.len() != r.inten.len() {
            return None;
        }
        match shared {
            None => shared = Some(r.rt),
            Some(a) => {
                if a.len() != r.rt.len() {
                    return None;
                }
                if !std::ptr::eq(a.as_ptr(), r.rt.as_ptr())
                    && !a.iter().zip(r.rt).all(|(x, y)| x.to_bits() == y.to_bits())
                {
                    return None;
                }
            }
        }
    }
    let axis = shared?;
    // Strictly ascending rules out the two cases the sort-and-dedup build would change:
    // an out-of-order axis, and repeated RT values (where the map keeps the last).
    if !axis.windows(2).all(|w| w[0] < w[1]) {
        return None;
    }
    let traces_full = rows
        .iter()
        .map(|r| {
            if r.rt.is_empty() {
                vec![0.0; axis.len()]
            } else {
                r.inten.iter().map(|&v| v as f64).collect()
            }
        })
        .collect();
    Some(TraceAlign {
        axis_full: axis.to_vec(),
        traces_full,
    })
}

/// General path: the sorted, deduplicated union of every row's RT values, with each row
/// resampled onto it through a bit-pattern-keyed map.
fn align_union(rows: &[ChromRow]) -> TraceAlign {
    let mut axis_full: Vec<f32> = rows.iter().flat_map(|r| r.rt.iter().cloned()).collect();
    axis_full.sort_by(|a, b| a.total_cmp(b));
    axis_full.dedup();
    let traces_full: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
            let map: HashMap<u32, f32> =
                r.rt.iter()
                    .zip(r.inten.iter())
                    .map(|(&t, &v)| (t.to_bits(), v))
                    .collect();
            axis_full
                .iter()
                .map(|t| *map.get(&t.to_bits()).unwrap_or(&0.0) as f64)
                .collect()
        })
        .collect();
    TraceAlign {
        axis_full,
        traces_full,
    }
}

/// Per-PSM evidence handed to the extended feature families. All arrays are
/// f64. Fragment-indexed arrays share one order; time-series share `axis`
/// (elution-peak-bounded) or `axis_full` (whole extracted window). Built once
/// per PSM by `build_evidence`, then scalar fields are filled by the caller.
/// The family modules in `stages/features/` read this and return feature values.
pub struct Evidence {
    /// RT axis (seconds) restricted to the detected elution peak.
    pub axis: Vec<f64>,
    /// Per-fragment intensity over `axis` (zero-filled), fragment order.
    pub traces: Vec<Vec<f64>>,
    /// Full extracted-window RT axis (seconds).
    pub axis_full: Vec<f64>,
    /// Per-fragment intensity over `axis_full`.
    pub traces_full: Vec<Vec<f64>>,
    /// Predicted (library) intensity per fragment.
    pub pred: Vec<f64>,
    /// Observed intensity at the apex scan per fragment.
    pub obs_apex: Vec<f64>,
    /// b-ion (true) vs y-ion (false) per fragment.
    pub is_b: Vec<bool>,
    /// Ion ordinal per fragment.
    pub ordinal: Vec<u32>,
    /// Fragment charge per fragment.
    pub frag_charge: Vec<u32>,
    /// Theoretical fragment m/z per fragment.
    pub frag_mz: Vec<f64>,
    /// Intensity-weighted observed fragment m/z per fragment.
    pub frag_obs_mz: Vec<f64>,
    /// Signed mass error (ppm) per fragment.
    pub mass_err_ppm: Vec<f64>,
    /// Index of the apex within `axis`.
    pub apex_idx: usize,
    /// Predicted-intensity-weighted reference elution profile over `axis`.
    pub ref_profile: Vec<f64>,
    /// The same profile over `axis_full`, and therefore `axis_full.len()` long: both
    /// readers index it with a position on that axis. Built once here instead of once in
    /// `coelution` and again in `interference`. Those two builds were bit-identical --
    /// both weight by the raw `pred[f]`, both accumulate fragment-outer and time-inner
    /// over `traces_full` -- so this is `weighted_reference_full` called once.
    /// `chromatographic` is NOT folded in: it clamps the weights at zero and falls back
    /// to an unweighted sum when they are all zero, which differs whenever a predicted
    /// intensity is negative (`index.rs` rejects only non-finite ones), so it keeps its
    /// own build.
    ///
    /// The length is a contract of `build_evidence`, not of the type, so both readers
    /// CHECK it and rebuild rather than trust it: this struct is `pub` with `pub` fields,
    /// several test fixtures fill it with `vec![]`, and a short profile would degrade
    /// silently rather than fail (see the comment in `interference::values`).
    pub ref_profile_full: Vec<f64>,
    /// The pairwise and reference correlations several families and `fragment_features`
    /// all computed from `traces`, `ref_profile`, `traces_full` and `ref_profile_full`,
    /// computed once per PSM by `PairStats::new` from exactly those fields. `Some` only
    /// from `evidence_from`; a reader uses it when its shape matches the evidence it is
    /// reading (`PairStats::fits`) and computes the value itself otherwise, which is
    /// what the `vec![]`-filled test fixtures (with `None`) exercise.
    pub pair_stats: Option<PairStats>,
    // --- scalars (filled by the caller after build) ---
    pub apex_rt: f64,
    pub rt_pred_cal: f64,
    pub rt_err: f64,
    pub gradient: f64,
    pub precursor_mz: f64,
    pub charge: i32,
    pub seq_len: i32,
    pub n_matched: i32,
    pub n_predicted: i32,
    pub seed_score: f64,
    pub seed_identified: f64,
    pub apex_intensity: f64,
    // --- MS1 apex isotopes (None when no MS1 provided) ---
    pub ms1_mono: Option<f64>,
    pub ms1_iso1: Option<f64>,
    pub ms1_iso2: Option<f64>,
    pub ms1_isom1: Option<f64>,
    /// MS1 isotope XICs [mono, +1, +2] resampled onto `axis`. Populated when the
    /// extract stage emits MS1 window-grid chromatograms (default on with MS1).
    pub ms1_xic: Vec<Vec<f64>>,
    /// Opt-in `ms1_precursor_features` gate (config). When false the ms1 family's
    /// `ms1_isotope_height_corr` returns 0.0 (default), keeping the vector effect
    /// unchanged; when true it computes the apex-isotope Pearson.
    pub ms1_precursor_features: bool,
    /// Spectrum-centric demix features (D2), from the extract stage. All 0 unless
    /// `extract.emit_demix_features` populated the columns.
    pub deconv_explained: f64,
    pub deconv_active: f64,
    pub deconv_share: f64,
    pub deconv_max_collin: f64,
    pub deconv_shadow: f64,
}

/// Parse a fragment name like `b3`, `y7`, `b3^2` into (is_b, ordinal, charge).
fn parse_ion(name: &str) -> (bool, u32, u32) {
    let is_b = name.starts_with('b');
    let rest = name.get(1..).unwrap_or("");
    let (ord_str, chg) = match rest.split_once('^') {
        Some((o, c)) => (o, c.parse::<u32>().unwrap_or(1)),
        None => (rest, 1),
    };
    (is_b, ord_str.parse::<u32>().unwrap_or(0), chg)
}

/// Predicted-intensity-weighted reference profile over the FULL extraction window, with
/// the raw (unclamped) weights.
///
/// [`Evidence::ref_profile_full`] is this; the function exists so the one build is
/// written once and the equality of the two it replaced can be read off it. It reproduces
/// `interference`'s inline loop exactly (`for f in 0..k` under an `f < traces_full.len()`
/// guard, `n = x.len().min(t)`), and `coelution`'s `weighted_reference` whenever
/// `traces_full.len() == pred.len()` -- which `build_evidence` guarantees, since both are
/// built per row of the same `rows`, and which coelution's `has_full` guard requires
/// anyway. Fragment-outer, time-inner, so the f64 accumulation order is the old one.
fn weighted_reference_full(traces_full: &[Vec<f64>], pred: &[f64], t: usize) -> Vec<f64> {
    let mut r = vec![0.0f64; t];
    for (f, x) in traces_full.iter().enumerate().take(pred.len()) {
        let w = pred[f];
        for (dst, v) in r.iter_mut().zip(x.iter()) {
            *dst += w * v;
        }
    }
    r
}

/// Observed intensity at the scan nearest the apex, per row (the first of two equally near
/// scans, by the strict `<`), widened to f64. `fragment_features` and `build_evidence` each
/// ran this K x T search over the same rows; the caller now runs it once per PSM and hands
/// the result to both.
fn apex_intensities(rows: &[ChromRow], apex_rt: f64) -> Vec<f64> {
    rows.iter()
        .map(|r| {
            let mut best = 0.0f32;
            let mut bestd = f64::MAX;
            for (k, &rt) in r.rt.iter().enumerate() {
                let d = (rt as f64 - apex_rt).abs();
                if d < bestd {
                    bestd = d;
                    best = r.inten[k];
                }
            }
            best as f64
        })
        .collect()
}

/// The elution-peak window `lo..=hi` on the union axis: the global half-widths when a
/// confident set gave them, otherwise the walk down the smoothed top-3-predicted profile
/// from the apex-nearest scan; the whole axis below three points. `fragment_features`
/// (under `bound_features`) and `build_evidence` each carried this block, identical term
/// for term, and computed it twice per PSM.
fn peak_window(
    axis_full: &[f32],
    traces_full: &[Vec<f64>],
    pred: &[f64],
    apex_rt: f64,
    frac: f64,
    grace: usize,
    global_bounds: Option<(f64, f64)>,
) -> (usize, usize) {
    if axis_full.len() < 3 {
        return (0, axis_full.len().saturating_sub(1));
    }
    let ai = axis_full
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (**a as f64 - apex_rt)
                .abs()
                .total_cmp(&(**b as f64 - apex_rt).abs())
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    match global_bounds {
        Some((l, r)) => global_bound_indices(axis_full, apex_rt, ai, l, r),
        None => {
            let mut ord: Vec<usize> = (0..pred.len()).collect();
            ord.sort_by(|&a, &b| pred[b].total_cmp(&pred[a]));
            let k3: Vec<usize> = ord.into_iter().take(3).collect();
            let prof_raw: Vec<f64> = (0..axis_full.len())
                .map(|k| k3.iter().map(|&i| traces_full[i][k]).sum::<f64>())
                .collect();
            let prof = smooth3(&prof_raw);
            peak_bounds(&prof, ai, frac, grace)
        }
    }
}

/// The pairwise and fragment-vs-reference statistics of one PSM's peak-window traces,
/// computed once and read by `fragment_features` and the `coelution`, `interference`,
/// `ion_series` and `nonzero` families.
///
/// Each of them computed some of these itself, from the same traces: the pair Pearson
/// matrix and the lag-optimised cross-correlation of every pair (in `fragment_features`
/// and again in `coelution`, which docs/28 section 3.1 found bit-identical as columns),
/// subsets of the same pair matrix (`ion_series` per series and for complementary pairs,
/// `nonzero` when every scan carries signal), every fragment against the reference
/// profile (four times per fragment) and every full-window trace against the full-window
/// reference (twice). Every entry here is produced by the same kernel call the reader made
/// (`pearson_pairs`, `best_xcorr_normed`, `pearson_vs`), on the same arguments in the same
/// order, so a read is bit-identical to the computation it replaces. Pearson is exactly
/// symmetric, so `corr(b, a)` serves a pair asked for the other way round; the
/// cross-correlation is not, and is stored for `a < b` only, the one order every reader uses.
pub struct PairStats {
    k: usize,
    np: usize,
    /// `k x k` row-major, `pearson(traces[a], traces[b])` off the diagonal, 1.0 on it.
    corr: Vec<f64>,
    /// `k x k` row-major, `best_xcorr_normed(traces[a], traces[b], 5, ..)` for `a < b`.
    xcorr: Vec<(i32, f64)>,
    /// `xcorr_norm(traces[f])`.
    norms: Vec<f64>,
    /// `pearson(traces[f], ref_profile)`.
    ref_corr: Vec<f64>,
    /// `pearson(traces_full[f], ref_profile_full)`, or empty when `traces_full` does not
    /// have one row per fragment.
    ref_corr_full: Vec<f64>,
}

impl PairStats {
    fn new(
        traces: &[Vec<f64>],
        np: usize,
        ref_profile: &[f64],
        traces_full: &[Vec<f64>],
        ref_profile_full: &[f64],
    ) -> PairStats {
        let k = traces.len();
        let mut pairs: Vec<f64> = Vec::with_capacity(k * k.saturating_sub(1) / 2);
        pearson_pairs(traces, &mut pairs);
        let mut pairs = pairs.into_iter();
        let norms: Vec<f64> = traces.iter().map(|t| xcorr_norm(t)).collect();
        let mut corr = vec![0.0f64; k * k];
        let mut xcorr = vec![(0i32, 0.0f64); k * k];
        for a in 0..k {
            corr[a * k + a] = 1.0;
            for b in (a + 1)..k {
                let p = pairs.next().expect("one correlation per pair");
                corr[a * k + b] = p;
                corr[b * k + a] = p;
                xcorr[a * k + b] = best_xcorr_normed(
                    &traces[a],
                    &traces[b],
                    XCORR_MAXLAG as i32,
                    norms[a],
                    norms[b],
                );
            }
        }
        let rc = Centered::new(ref_profile);
        let ref_corr = traces
            .iter()
            .map(|t| pearson_vs(t, ref_profile, &rc))
            .collect();
        let ref_corr_full = if traces_full.len() == k {
            let rfc = Centered::new(ref_profile_full);
            traces_full
                .iter()
                .map(|t| pearson_vs(t, ref_profile_full, &rfc))
                .collect()
        } else {
            Vec::new()
        };
        PairStats {
            k,
            np,
            corr,
            xcorr,
            norms,
            ref_corr,
            ref_corr_full,
        }
    }

    /// Whether these statistics describe `k` fragments on an `np`-point peak window. A
    /// reader checks it before trusting a read: `Evidence` has `pub` fields and could be
    /// assembled around statistics of other traces.
    pub(crate) fn fits(&self, k: usize, np: usize) -> bool {
        self.k == k && self.np == np
    }

    /// `pearson(traces[a], traces[b])`, either order; 1.0 when `a == b`.
    pub(crate) fn corr(&self, a: usize, b: usize) -> f64 {
        self.corr[a * self.k + b]
    }

    /// The `k x k` pair matrix, row-major, 1.0 on the diagonal.
    pub(crate) fn corr_matrix(&self) -> &[f64] {
        &self.corr
    }

    /// `best_xcorr(traces[a], traces[b], 5)` for `a < b`.
    pub(crate) fn xcorr(&self, a: usize, b: usize) -> (i32, f64) {
        debug_assert!(a < b, "the cross-correlation is stored for a < b only");
        self.xcorr[a * self.k + b]
    }

    pub(crate) fn norms(&self) -> &[f64] {
        &self.norms
    }

    pub(crate) fn ref_corr(&self) -> &[f64] {
        &self.ref_corr
    }

    /// `pearson(traces_full[f], ref_profile_full)` per fragment, when it was computed.
    pub(crate) fn ref_corr_full(&self) -> Option<&[f64]> {
        (self.ref_corr_full.len() == self.k).then_some(&self.ref_corr_full[..])
    }
}

/// One PSM's elution-peak window and everything `fragment_features` and
/// [`evidence_from`] both need from it: the window, the traces sliced to it, the
/// predicted-intensity-weighted reference profile over it (clamped weights) and over the
/// whole window (raw weights), and the [`PairStats`]. Built once per PSM from the
/// alignment; `fragment_features` borrows it when `bound_features` makes its window the
/// same one, and the Evidence then takes it over.
struct PeakTraces {
    lo: usize,
    hi: usize,
    traces: Vec<Vec<f64>>,
    ref_profile: Vec<f64>,
    ref_profile_full: Vec<f64>,
    stats: PairStats,
}

impl PeakTraces {
    fn new(al: &TraceAlign, pred: &[f64], (lo, hi): (usize, usize)) -> PeakTraces {
        let traces: Vec<Vec<f64>> = al.traces_full.iter().map(|t| t[lo..=hi].to_vec()).collect();
        let np = al.axis_full[lo..=hi].len();
        let mut ref_profile = vec![0.0f64; np];
        for (fi, tr) in traces.iter().enumerate() {
            let w = pred[fi].max(0.0);
            for k in 0..np {
                ref_profile[k] += w * tr[k];
            }
        }
        let ref_profile_full = weighted_reference_full(&al.traces_full, pred, al.axis_full.len());
        let stats = PairStats::new(
            &traces,
            np,
            &ref_profile,
            &al.traces_full,
            &ref_profile_full,
        );
        PeakTraces {
            lo,
            hi,
            traces,
            ref_profile,
            ref_profile_full,
            stats,
        }
    }
}

/// Build the trace-derived fields of [`Evidence`] from a PSM's chromatogram
/// rows (scalar fields default; the caller fills them), in one call. The chunked pass
/// builds the same parts itself ([`apex_intensities`], [`peak_window`], [`PeakTraces`])
/// so that `fragment_features` can share them, then calls [`evidence_from`]; this is the
/// one-call form the tests use.
#[cfg(test)]
fn build_evidence(
    rows: &[ChromRow],
    al: TraceAlign,
    ms1_rows: &[ChromRow],
    apex_rt: f64,
    frac: f64,
    grace: usize,
    global_bounds: Option<(f64, f64)>,
) -> Evidence {
    let obs = apex_intensities(rows, apex_rt);
    let pred: Vec<f64> = rows.iter().map(|r| r.pred_int as f64).collect();
    let win = peak_window(
        &al.axis_full,
        &al.traces_full,
        &pred,
        apex_rt,
        frac,
        grace,
        global_bounds,
    );
    let peak = PeakTraces::new(&al, &pred, win);
    evidence_from(rows, al, obs, pred, peak, ms1_rows, apex_rt)
}

/// The trace-derived fields of [`Evidence`] from parts built once per PSM: the apex
/// intensities, the predicted intensities, the alignment (moved in: the Evidence owns it)
/// and the peak window with its statistics. The families therefore see the same elution
/// peak the legacy features use.
fn evidence_from(
    rows: &[ChromRow],
    al: TraceAlign,
    obs_apex: Vec<f64>,
    pred: Vec<f64>,
    peak: PeakTraces,
    ms1_rows: &[ChromRow],
    apex_rt: f64,
) -> Evidence {
    let m = rows.len();
    let mut is_b = Vec::with_capacity(m);
    let mut ordinal = Vec::with_capacity(m);
    let mut frag_charge = Vec::with_capacity(m);
    let mut frag_mz = Vec::with_capacity(m);
    let mut frag_obs_mz = Vec::with_capacity(m);
    let mut mass_err_ppm = Vec::with_capacity(m);
    for r in rows {
        let (b, o, c) = parse_ion(r.frag_name);
        is_b.push(b);
        ordinal.push(o);
        frag_charge.push(c);
        frag_mz.push(r.frag_mz);
        frag_obs_mz.push(r.frag_obs_mz);
        mass_err_ppm.push(ppm_diff(r.frag_obs_mz, r.frag_mz));
    }

    // The alignment is built once per PSM by the caller and read by `fragment_features`
    // first; [`Evidence`] then owns it, so it is moved in rather than rebuilt here.
    let TraceAlign {
        axis_full,
        traces_full,
    } = al;
    let PeakTraces {
        lo: lo_i,
        hi: hi_i,
        traces,
        ref_profile,
        ref_profile_full,
        stats,
    } = peak;
    let axis: Vec<f64> = axis_full[lo_i..=hi_i].iter().map(|&x| x as f64).collect();
    let axis_full_f: Vec<f64> = axis_full.iter().map(|&x| x as f64).collect();

    let apex_idx = axis
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - apex_rt).abs().total_cmp(&(*b - apex_rt).abs()))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // MS1 isotope XICs [mono, +1, +2] sampled on the same grid as the fragments,
    // mapped onto axis_full then sliced to the elution peak. Present only when the
    // extract stage emitted them (grid mode + MS1 data); else empty.
    //
    // Extract writes an MS1 row on the fragments' own window grid (the same `grid_rt` every
    // fragment row carries), so the usual row samples `axis_full` itself and the map below
    // is the identity. `ms1_on_axis` recognises that case and slices the row directly: the
    // map built three SipHash tables and a full-window vector per PSM to return, at every
    // position, the value already stored there. Bit-identical: on a strictly ascending axis
    // every bit pattern is distinct, so `map[axis_full[k]]` is exactly `r.inten[k]`, and
    // both paths widen the same f32 to f64. Any other row -- another grid, a length
    // mismatch, an axis that is not strictly ascending -- still goes through the map.
    let axis_ascending = axis_full.windows(2).all(|w| w[0] < w[1]);
    let ms1_on_axis = |r: &ChromRow| -> bool {
        axis_ascending
            && r.rt.len() == axis_full.len()
            && r.inten.len() == r.rt.len()
            && r.rt
                .iter()
                .zip(&axis_full)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    };
    let ms1_xic: Vec<Vec<f64>> = {
        let mut out = Vec::new();
        for name in ["ms1_mono", "ms1_iso1", "ms1_iso2"] {
            if let Some(r) = ms1_rows.iter().find(|r| r.frag_name == name) {
                if ms1_on_axis(r) {
                    out.push(r.inten[lo_i..=hi_i].iter().map(|&v| v as f64).collect());
                    continue;
                }
                let map: HashMap<u32, f32> =
                    r.rt.iter()
                        .zip(r.inten.iter())
                        .map(|(&t, &v)| (t.to_bits(), v))
                        .collect();
                let full: Vec<f64> = axis_full
                    .iter()
                    .map(|t| *map.get(&t.to_bits()).unwrap_or(&0.0) as f64)
                    .collect();
                out.push(full[lo_i..=hi_i].to_vec());
            }
        }
        if out.len() == 3 {
            out
        } else {
            Vec::new()
        }
    };

    Evidence {
        axis,
        traces,
        axis_full: axis_full_f,
        traces_full,
        pred,
        obs_apex,
        is_b,
        ordinal,
        frag_charge,
        frag_mz,
        frag_obs_mz,
        mass_err_ppm,
        apex_idx,
        ref_profile,
        ref_profile_full,
        pair_stats: Some(stats),
        apex_rt,
        rt_pred_cal: 0.0,
        rt_err: 0.0,
        gradient: 1.0,
        precursor_mz: 0.0,
        charge: 0,
        seq_len: 0,
        n_matched: 0,
        n_predicted: 0,
        seed_score: 0.0,
        seed_identified: 0.0,
        apex_intensity: 0.0,
        ms1_mono: None,
        ms1_iso1: None,
        ms1_iso2: None,
        ms1_isom1: None,
        ms1_xic,
        ms1_precursor_features: false,
        deconv_explained: 0.0,
        deconv_active: 0.0,
        deconv_share: 0.0,
        deconv_max_collin: 0.0,
        deconv_shadow: 0.0,
    }
}

pub struct FeaturesParams<'a> {
    pub psms: &'a str,
    pub chromatograms: &'a str,
    /// Optional seed_psms for search-engine corroboration features
    /// (docs/10_features.md).
    pub seed: Option<&'a str>,
    pub out: &'a str,
    pub out_pin: &'a str,
    pub cfg: &'a FeaturesConfig,
    pub config_hash: &'a str,
}

/// Chromatogram rows resident at once in the chunked feature pass. At the ~200-point
/// traces of a 2 h gradient a chunk of 2^20 rows is about 1 GiB of trace payload; the
/// whole-run store this replaced held 62.7 GiB at 31.1M rows on the HYE benchmark.
const CHUNK_CHROM_ROWS: usize = 1 << 20;

/// PSM rows resident at once in the same pass. A chunk closes on whichever limit binds
/// first, because the two quantities are only loosely related and the value buffers are
/// sized in PSM rows, not chromatogram rows.
///
/// The chunk was closed on chromatogram rows alone, while `ValueMatrix`, `ext_vals`,
/// `frag_feats`, `prelim`, `elu_lo` and `elu_hi` are all `rows_in_chunk = psm_hi - psm_lo`
/// long. The ratio between the two is data-dependent and unbounded: docs/27 measured
/// 2,603,894 PSM rows in 38 chunks, so 68,523 PSM rows per chunk at 15.3 chromatogram rows
/// each, and 387 x 68,523 x 8 = 212 MB of value matrix (the 0.21 GiB the stage reported).
/// At the engine default `top_n_fragments = 6` plus three MS1 rows the ratio is 9 and the
/// same constant gives 116,508 PSM rows per chunk; with `retain_top_peaks = 5`, which
/// multiplies PSM rows but not chromatogram rows (several PSM rows share one candidate's
/// traces), it gives 582,542 and about 5.2 GB in flight -- matrix, `ext_vals` and the
/// previous chunk still owned by the writer thread. 2^16 bounds that at ~203 MB per
/// matrix whatever the shape, and matches `FEATURE_ROW_GROUP_ROWS`, the row group the
/// writer buffers anyway.
///
/// It binds at the docs/27 shape (68,523 PSM rows per chunk against this 65,536, so 39
/// chunks rather than 38) and therefore moves production chunk boundaries. What that
/// preserves is the VALUES, and only the values: the third arm of
/// `extended_features_are_chunk_invariant` closes chunks on this limit and compares every
/// f64 column bit for bit. The features.parquet BYTES are NOT preserved, which is measured
/// rather than conceded -- a chunk is one `TableWriter::write_cols` call, and
/// `moving_a_chunk_boundary_moves_parquet_bytes_above_one_row_group` shows a 300,000-row
/// table written in 68,523-row calls differs from the same table written in one. Nothing
/// downstream of features reads those bytes; the artifact hash in the manifest is
/// provenance, and it moves. `ci/smoke.sh` cannot speak to this either way: it is orders
/// of magnitude below 65,536 PSM rows, so it runs one chunk with the limit and without it.
const CHUNK_PSM_ROWS: usize = 1 << 16;

/// Rows per parquet row group of the features table: the encoder buffers
/// `rows x n_features x 8` bytes, so 2^16 rows of the 387-feature Extended set is
/// ~200 MB in flight instead of the writer default's ~3 GB.
const FEATURE_ROW_GROUP_ROWS: usize = 1 << 16;

/// Rows per decoded chromatogram batch. Matches the list-column batch size of the IO
/// layer, so a batch is tens of MB whatever the chunk size is.
const CHROM_BATCH_ROWS: usize = 1 << 12;

/// Interned fragment names. A chromatogram table has tens of millions of rows but only
/// a few dozen distinct names (`y1`..`y30`, `b1`.., `ms1_mono`..), so rows carry a name
/// id and each string is stored once.
#[derive(Clone, Default)]
struct NameTab {
    ids: HashMap<String, u32>,
    names: Vec<String>,
}

impl NameTab {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.ids.get(s) {
            return i;
        }
        let i = self.names.len() as u32;
        self.names.push(s.to_string());
        self.ids.insert(s.to_string(), i);
        i
    }

    fn get(&self, id: u32) -> &str {
        &self.names[id as usize]
    }
}

/// Marker for an empty trace: the row carries no RT axis at all (extract emits an empty
/// trace, not a zero-filled one, for a predicted fragment that was never observed).
const NO_AXIS: u32 = u32::MAX;

/// One flat set of chromatogram rows (fragments, or the MS1 isotope XICs), grouped by
/// candidate. One allocation per array instead of two `Vec`s per row.
#[derive(Default)]
struct RowSet {
    name_id: Vec<u32>,
    frag_mz: Vec<f64>,
    frag_obs_mz: Vec<f64>,
    pred_int: Vec<f32>,
    /// Index into [`ChromChunk::axis_off`], or [`NO_AXIS`] for an empty trace.
    axis_id: Vec<u32>,
    /// Row `r`'s intensities are `int_vals[int_off[r]..int_off[r + 1]]`.
    int_off: Vec<usize>,
    int_vals: Vec<f32>,
    /// Candidate `c`'s rows are `cand_off[c]..cand_off[c + 1]`.
    cand_off: Vec<usize>,
}

impl RowSet {
    fn new() -> RowSet {
        RowSet {
            int_off: vec![0],
            cand_off: vec![0],
            ..Default::default()
        }
    }

    fn nrows(&self) -> usize {
        self.name_id.len()
    }

    fn payload_bytes(&self) -> usize {
        crate::memlog::bytes_of(&self.name_id)
            + crate::memlog::bytes_of(&self.frag_mz)
            + crate::memlog::bytes_of(&self.frag_obs_mz)
            + crate::memlog::bytes_of(&self.pred_int)
            + crate::memlog::bytes_of(&self.axis_id)
            + crate::memlog::bytes_of(&self.int_off)
            + crate::memlog::bytes_of(&self.int_vals)
            + crate::memlog::bytes_of(&self.cand_off)
    }
}

/// One chunk of the chromatogram table: a contiguous run of candidates, with the RT axis
/// shared by every row of a candidate that samples the same grid (extract's window-grid
/// mode gives every fragment of a candidate the identical axis, so this halves the store)
/// and one flat buffer per array.
#[derive(Default)]
struct ChromChunk {
    /// Candidate ids in table order.
    cids: Vec<u32>,
    index: HashMap<u32, usize>,
    frag: RowSet,
    ms1: RowSet,
    /// Axis `a` is `axis_vals[axis_off[a]..axis_off[a + 1]]`.
    axis_off: Vec<usize>,
    axis_vals: Vec<f32>,
    /// First axis of the candidate being filled, so dedup only compares within it.
    open_axis_lo: usize,
}

impl ChromChunk {
    fn new() -> ChromChunk {
        ChromChunk {
            frag: RowSet::new(),
            ms1: RowSet::new(),
            axis_off: vec![0],
            ..Default::default()
        }
    }

    fn open_candidate(&mut self, cid: u32) {
        self.index.insert(cid, self.cids.len());
        self.cids.push(cid);
        self.open_axis_lo = self.axis_off.len() - 1;
    }

    fn close_candidate(&mut self) {
        self.frag.cand_off.push(self.frag.nrows());
        self.ms1.cand_off.push(self.ms1.nrows());
    }

    /// Store `rt` as an axis id, reusing an axis already stored for the open candidate
    /// when the values are identical (the common case: one grid per candidate).
    fn axis_for(&mut self, rt: &[f32]) -> u32 {
        if rt.is_empty() {
            return NO_AXIS;
        }
        for a in self.open_axis_lo..self.axis_off.len() - 1 {
            let (lo, hi) = (self.axis_off[a], self.axis_off[a + 1]);
            if hi - lo == rt.len() && self.axis_vals[lo..hi] == *rt {
                return a as u32;
            }
        }
        self.axis_vals.extend_from_slice(rt);
        self.axis_off.push(self.axis_vals.len());
        (self.axis_off.len() - 2) as u32
    }

    #[allow(clippy::too_many_arguments)]
    fn push_row(
        &mut self,
        is_ms1: bool,
        name_id: u32,
        frag_mz: f64,
        frag_obs_mz: f64,
        pred_int: f32,
        rt: &[f32],
        inten: &[f32],
    ) {
        let axis_id = self.axis_for(rt);
        let set = if is_ms1 {
            &mut self.ms1
        } else {
            &mut self.frag
        };
        set.name_id.push(name_id);
        set.frag_mz.push(frag_mz);
        set.frag_obs_mz.push(frag_obs_mz);
        set.pred_int.push(pred_int);
        set.axis_id.push(axis_id);
        set.int_vals.extend_from_slice(inten);
        set.int_off.push(set.int_vals.len());
    }

    fn axis(&self, id: u32) -> &[f32] {
        if id == NO_AXIS {
            return &[];
        }
        let a = id as usize;
        &self.axis_vals[self.axis_off[a]..self.axis_off[a + 1]]
    }

    /// Candidate `ci`'s rows of one set as the borrowed view the feature code takes.
    fn rows<'a>(&'a self, set: &'a RowSet, ci: usize, names: &'a NameTab) -> Vec<ChromRow<'a>> {
        (set.cand_off[ci]..set.cand_off[ci + 1])
            .map(|r| ChromRow {
                frag_name: names.get(set.name_id[r]),
                frag_mz: set.frag_mz[r],
                frag_obs_mz: set.frag_obs_mz[r],
                pred_int: set.pred_int[r],
                rt: self.axis(set.axis_id[r]),
                inten: &set.int_vals[set.int_off[r]..set.int_off[r + 1]],
            })
            .collect()
    }

    fn payload_bytes(&self) -> (usize, usize) {
        let axes =
            crate::memlog::bytes_of(&self.axis_vals) + crate::memlog::bytes_of(&self.axis_off);
        (self.frag.payload_bytes() + axes, self.ms1.payload_bytes())
    }
}

/// Sequential reader over the chromatogram table that hands out one [`ChromChunk`] of a
/// requested row count at a time. One decoded batch is resident beyond the chunk; a batch
/// straddling a chunk boundary is sliced and its remainder kept for the next chunk.
struct ChromStream {
    inner: mumdia_io::table::BatchReader,
    pending: Option<RecordBatch>,
    has_obs_mz: bool,
}

impl ChromStream {
    /// `ch` may be a whole-file handle or a [`TableFile::span`] of one; a span carries the
    /// whole file's schema, so the optional column is decided from the handle either way.
    ///
    /// This used to call `mumdia_io::table::column_names(path)`, which opens the file and
    /// parses its footer a SECOND time for a question the handle can already answer. The
    /// confident-bounds pass opens one stream per span, so that was two full footer parses
    /// per span on a table whose footer is not small.
    fn open(ch: &TableFile) -> Result<ChromStream> {
        let has_obs_mz = ch.has_column("frag_obs_mz");
        let mut cols = vec![
            "candidate_id",
            "frag_name",
            "frag_mz",
            "predicted_intensity",
            "rt",
            "intensity",
        ];
        if has_obs_mz {
            cols.push("frag_obs_mz");
        }
        Ok(ChromStream {
            inner: ch.batches(Some(&cols), CHROM_BATCH_ROWS)?,
            pending: None,
            has_obs_mz,
        })
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if let Some(b) = self.pending.take() {
            return Ok(Some(b));
        }
        match self.inner.next() {
            None => Ok(None),
            Some(b) => Ok(Some(b?)),
        }
    }

    /// Read exactly `want` rows into a chunk. `names` is shared across chunks so the same
    /// fragment name interns to the same id for the whole run.
    fn read_chunk(&mut self, want: usize, names: &mut NameTab) -> Result<ChromChunk> {
        self.read_chunk_filtered(want, names, None)
    }

    /// [`ChromStream::read_chunk`], keeping only the candidates in `keep`. The rows of a
    /// dropped candidate are still decoded (parquet gives no cheaper way to skip a row
    /// inside a page) but never copied, which is what makes the global-bounds pass over
    /// the whole table affordable: it needs a few thousand confident candidates.
    fn read_chunk_filtered(
        &mut self,
        want: usize,
        names: &mut NameTab,
        keep: Option<&std::collections::HashSet<u32>>,
    ) -> Result<ChromChunk> {
        let mut chunk = ChromChunk::new();
        let mut open: Option<u32> = None;
        let mut taken = 0usize;
        while taken < want {
            let b = match self.next_batch()? {
                Some(b) => b,
                None => break,
            };
            let n = b.num_rows().min(want - taken);
            if n < b.num_rows() {
                self.pending = Some(b.slice(n, b.num_rows() - n));
            }
            let s = b.schema();
            let col = |name: &str| -> Result<&ArrayRef> {
                let i = s
                    .index_of(name)
                    .map_err(|_| anyhow!("chromatograms batch has no column '{name}'"))?;
                Ok(b.column(i))
            };
            let cid = col("candidate_id")?
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| anyhow!("chromatograms column 'candidate_id' is not u32"))?;
            let name = col("frag_name")?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("chromatograms column 'frag_name' is not utf8"))?;
            let fmz = col("frag_mz")?
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| anyhow!("chromatograms column 'frag_mz' is not f64"))?;
            let obsmz = if self.has_obs_mz {
                Some(
                    col("frag_obs_mz")?
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| anyhow!("chromatograms column 'frag_obs_mz' is not f64"))?,
                )
            } else {
                None
            };
            let pint = col("predicted_intensity")?
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow!("chromatograms column 'predicted_intensity' is not f32"))?;
            let rt = ListF32::of(col("rt")?, "rt")?;
            let inten = ListF32::of(col("intensity")?, "intensity")?;
            // Trace values are read as slices of the decoded batch and copied ONCE, by
            // `push_row` (the axis is deduplicated against the candidate's earlier rows
            // before it is stored, and the intensities are appended to the flat buffer).
            // They used to go through a per-row scratch buffer, which was a second full
            // copy of the payload plus an `Arc` allocation per row per column; see
            // [`list_row`].
            for k in 0..n {
                let c = cid.value(k);
                if keep.is_some_and(|s| !s.contains(&c)) {
                    continue;
                }
                if open != Some(c) {
                    if open.is_some() {
                        chunk.close_candidate();
                    }
                    chunk.open_candidate(c);
                    open = Some(c);
                }
                let rt_row = rt.row_slice(k, "rt")?;
                let int_row = inten.row_slice(k, "intensity")?;
                let nm = name.value(k);
                let id = names.intern(nm);
                chunk.push_row(
                    nm.starts_with("ms1_"),
                    id,
                    fmz.value(k),
                    obsmz.map(|a| a.value(k)).unwrap_or_else(|| fmz.value(k)),
                    pint.value(k),
                    rt_row,
                    int_row,
                );
            }
            taken += n;
        }
        if open.is_some() {
            chunk.close_candidate();
        }
        Ok(chunk)
    }
}

/// One unit of work: a contiguous run of PSM rows and the chromatogram rows they own.
#[derive(Debug)]
struct Chunk {
    psm_lo: usize,
    psm_hi: usize,
    chrom_rows: usize,
}

/// Split the run into chunks that never cut a candidate.
///
/// Extract emits a PSM row and that row's chromatogram rows in one pass, so both tables
/// carry the same candidates in the same order and each candidate's rows are contiguous
/// (`extract.rs`, the per-candidate emission loop). The chunked pass depends on that, so
/// it is verified here rather than assumed: a violation is a hard error naming the
/// artifact, not a silently mis-joined feature table.
///
/// A chunk closes when EITHER limit is reached: `chunk_rows` bounds the traces resident
/// (what the loader holds) and `max_psm_rows` bounds the value buffers (what the compute
/// and the writer hold). See `CHUNK_PSM_ROWS` for why one limit is not enough. Both are
/// closed at the end of the current candidate's PSM rows, so neither ever cuts a
/// candidate.
fn plan_chunks(
    psm_cid: &[u32],
    ch_cid: &[u32],
    chrom_path: &str,
    chunk_rows: usize,
    max_psm_rows: usize,
) -> Result<Vec<Chunk>> {
    // Run-length groups of both tables, with a contiguity check on each.
    let groups = |v: &[u32], what: &str| -> Result<Vec<(u32, usize, usize)>> {
        let mut out: Vec<(u32, usize, usize)> = Vec::new();
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut i = 0usize;
        while i < v.len() {
            let c = v[i];
            let lo = i;
            while i < v.len() && v[i] == c {
                i += 1;
            }
            if !seen.insert(c) {
                return Err(anyhow!(
                    "{what}: candidate {c} appears in more than one run; the chunked \
                     feature pass needs each candidate's rows contiguous. Re-run extract \
                     to regenerate {chrom_path} and psms_extracted.parquet."
                ));
            }
            out.push((c, lo, i - lo));
        }
        Ok(out)
    };
    let pg = groups(psm_cid, "psms_extracted.parquet")?;
    let cg = groups(ch_cid, chrom_path)?;

    // The chromatogram candidates must be a subsequence of the PSM candidates, in order
    // (a PSM row can have no chromatogram rows; the reverse cannot happen).
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut ci = 0usize;
    let (mut lo, mut acc) = (0usize, 0usize);
    for (gi, &(pc, plo, plen)) in pg.iter().enumerate() {
        if ci < cg.len() && cg[ci].0 == pc {
            acc += cg[ci].2;
            ci += 1;
        }
        let last = gi + 1 == pg.len();
        if acc >= chunk_rows || plo + plen - lo >= max_psm_rows || last {
            chunks.push(Chunk {
                psm_lo: lo,
                psm_hi: plo + plen,
                chrom_rows: acc,
            });
            lo = plo + plen;
            acc = 0;
        }
    }
    if ci != cg.len() {
        return Err(anyhow!(
            "{chrom_path} holds candidate {} which is not in psms_extracted.parquet in \
             table order; the chunked feature pass needs both tables emitted by the same \
             extract run.",
            cg[ci].0
        ));
    }
    if chunks.is_empty() {
        chunks.push(Chunk {
            psm_lo: 0,
            psm_hi: psm_cid.len(),
            chrom_rows: 0,
        });
    }
    Ok(chunks)
}

/// Feature values of one chunk, one owned column per feature. Replaces the
/// `HashMap<&str, Vec<f64>>` that held one `Vec` per feature for the WHOLE RUN (7.5 GiB
/// at 2.6M rows x 387 features); a chunk's columns are a few hundred blocks, not a few
/// hundred thousand.
///
/// The columns are owned rather than slices of one flat buffer so that
/// [`ValueMatrix::take_column`] can hand each one to the parquet writer by move. The
/// flat layout it replaced had to copy every column (`column(c).to_vec()`), which meant
/// the whole matrix existed twice at the moment of the write -- the memory report said
/// `largest_chunk` and meant half of the real peak.
///
/// One allocation per column rather than one for the matrix does cut against this stage's
/// stated problem (the NUMBER of live medium blocks, not bytes), so the arithmetic is
/// worth stating: 387 columns of one chunk are 387 blocks and at most two matrices are in
/// flight, so 774 -- 0.07% of the kernel's 1,048,576 mappings per process, against the
/// 92,030 + 36,368 medium mappings measured on the process that hit the limit. What it
/// buys is a whole matrix (hundreds of MB) not copied at every write. The trade is
/// lopsided in favour of the columns; `value_matrix_columns_are_one_block_each` pins the
/// count so a future widening cannot creep past it unnoticed.
struct ValueMatrix {
    cols: Vec<Vec<f64>>,
}

impl ValueMatrix {
    fn new(n_cols: usize, rows: usize) -> ValueMatrix {
        ValueMatrix {
            cols: (0..n_cols).map(|_| vec![0.0; rows]).collect(),
        }
    }

    /// Set the column `c` (as resolved once by [`ColIx`]) for chunk-local row `r`. A
    /// feature outside the active set resolves to `None` and is dropped, exactly as the
    /// old map was filtered by `cols_active` at write time.
    fn set(&mut self, c: Option<usize>, r: usize, v: f64) {
        if let Some(c) = c {
            self.cols[c][r] = v;
        }
    }

    fn column(&self, c: usize) -> &[f64] {
        &self.cols[c]
    }

    /// Move column `c` out, leaving an empty Vec behind. The matrix is dropped straight
    /// after, so the emptied columns are never read again.
    fn take_column(&mut self, c: usize) -> Vec<f64> {
        std::mem::take(&mut self.cols[c])
    }

    /// What the columns actually hold, summed over them rather than computed from the
    /// shape, so a column already moved out by [`ValueMatrix::take_column`] counts as the
    /// zero it is. `vec![0.0; rows]` allocates capacity exactly `rows`, so on a full matrix
    /// this equals `cols * rows * 8`.
    fn payload_bytes(&self) -> usize {
        self.cols
            .iter()
            .map(|c| c.capacity() * std::mem::size_of::<f64>())
            .sum()
    }
}

/// Column index of every feature the serial assembly writes, resolved ONCE from the
/// active feature list.
///
/// The assembly used to name each feature as a string and hash it per row: ~390 lookups
/// for every PSM in the Extended set, all resolving to the same fixed permutation. The
/// permutation is known before the loop, so it is taken once here; `None` is a feature
/// the configured set does not carry, which the assembly then drops exactly as the
/// name-keyed setter did.
///
/// The field name IS the feature name, so the list below cannot drift from the names it
/// resolves, and `colix_covers_every_active_column` checks that the fields of one set
/// account for every column of that set exactly once.
macro_rules! col_ix {
    ($($field:ident),* $(,)?) => {
        struct ColIx {
            $($field: Option<usize>,)*
            /// One entry per extended-battery feature, in registry order.
            ext: Vec<Option<usize>>,
        }

        impl ColIx {
            fn new(cols_active: &[String], ext_names: &[&'static str]) -> ColIx {
                let idx: std::collections::HashMap<&str, usize> = cols_active
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (c.as_str(), i))
                    .collect();
                ColIx {
                    $($field: idx.get(stringify!($field)).copied(),)*
                    ext: ext_names.iter().map(|n| idx.get(n).copied()).collect(),
                }
            }

            /// Every non-extended column this resolved, for the coverage test.
            #[cfg(test)]
            fn named(&self) -> Vec<Option<usize>> {
                vec![$(self.$field,)*]
            }

            /// Each field paired with the feature name it is spelled as, so a test can
            /// check that a field resolves to the column its OWN name denotes rather than
            /// only that the resolved columns form a permutation. `stringify!` takes the
            /// names from the field list itself, so this cannot drift from it.
            #[cfg(test)]
            fn named_pairs(&self) -> Vec<(&'static str, Option<usize>)> {
                vec![$((stringify!($field), self.$field),)*]
            }
        }
    };
}

col_ix!(
    rt_error_abs,
    rt_error_rel,
    n_matched_fragments,
    coelution_run,
    log_apex_intensity,
    frag_corr,
    frag_cosine,
    spectral_angle,
    coelution_mean,
    coelution_best,
    n_coelution_above,
    charge,
    peptide_length,
    n_proteins,
    library_norm_manhattan,
    library_rmsd,
    xcorr_coelution,
    xcorr_shape,
    sum_b_intensity,
    sum_y_intensity,
    diff_by_intensity,
    n_b_ions,
    n_y_ions,
    weighted_mass_error,
    mean_mass_error,
    isotope_corr,
    ms1_isom1_ratio,
    log_mono_ms1,
    has_ms1,
    log_sn,
    n_observations,
    base_width_rt,
    seed_score,
    seed_identified,
    matched_fraction,
    profile_cos,
    ref_corr,
    best_ref_corr,
    low_frag_coel,
    evidence,
    contrast_min,
    resid_corr,
    coel_clean,
    shadow_frac,
    peak_contested_frac,
    peak_contested_count_frac,
    peak_apportioned_frac,
    n_charge_states,
    charge_multi_flag,
    cross_charge_intensity_log,
);

/// Percolator-style PIN written row by row as the chunks are computed. Nothing in the
/// pipeline reads it (rescore builds its own), so it is gated by `features.emit_pin`;
/// the byte output is unchanged from the single-shot writer it replaced.
struct PinWriter {
    w: std::io::BufWriter<std::fs::File>,
}

impl PinWriter {
    fn create(path: &str, feature_cols: &[String]) -> Result<PinWriter> {
        use std::io::Write as _;
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        w.write_all(b"SpecId\tLabel\tScanNr\tExpMass\tCalcMass\t")?;
        w.write_all(feature_cols.join("\t").as_bytes())?;
        w.write_all(b"\tPeptide\tProteins\n")?;
        Ok(PinWriter { w })
    }

    #[allow(clippy::too_many_arguments)]
    fn write_chunk(
        &mut self,
        n_cols: usize,
        m: &ValueMatrix,
        cid: &[u32],
        label: &[String],
        pform: &[String],
        protein: &[String],
        mz: &[f64],
    ) -> Result<()> {
        use std::io::Write as _;
        for i in 0..cid.len() {
            let lab = if label[i] == "decoy" { -1 } else { 1 };
            write!(
                self.w,
                "cand_{}\t{}\t{}\t{:.5}\t{:.5}\t",
                cid[i], lab, cid[i], mz[i], mz[i]
            )?;
            for c in 0..n_cols {
                let v = m.column(c)[i];
                write!(self.w, "{v:.6}\t")?;
            }
            writeln!(self.w, "-.{}.-\t{}", pform[i], protein[i])?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        use std::io::Write as _;
        self.w.flush()?;
        Ok(())
    }
}

/// Number of distinct charge states of each row's peptidoform, indexed by the dense
/// peptidoform id of that row.
///
/// Prefers the bitmask build and falls back to the set-per-peptidoform build, which is
/// what this used to do for every run: one `HashSet<i32>` heap block per distinct
/// peptidoform, about a million live for the whole run on a real library.
fn charge_states_per_row(pf_of_row: &[u32], n_pf: usize, charge: &[i32]) -> Vec<f64> {
    // Both builds zip the two columns, which TRUNCATES where the old `charge[i]` indexing
    // panicked. A short charge column is an artifact-shape error, so it stays loud rather
    // than becoming a silent partial reduction over the rows that happened to line up.
    assert_eq!(
        pf_of_row.len(),
        charge.len(),
        "features: the charge column does not cover every PSM row"
    );
    charge_states_by_mask(pf_of_row, n_pf, charge)
        .unwrap_or_else(|| charge_states_by_set(pf_of_row, n_pf, charge))
}

/// Bitmask build: exact for charges in `0..32`, which covers every real precursor charge.
/// `None` when any charge falls outside that range, so the caller takes the set build and
/// the count is provably the same either way.
fn charge_states_by_mask(pf_of_row: &[u32], n_pf: usize, charge: &[i32]) -> Option<Vec<f64>> {
    if !charge.iter().all(|&z| (0..32).contains(&z)) {
        return None;
    }
    let mut mask: Vec<u32> = vec![0; n_pf];
    for (&pf, &z) in pf_of_row.iter().zip(charge) {
        mask[pf as usize] |= 1u32 << z;
    }
    Some(
        pf_of_row
            .iter()
            .map(|&pf| mask[pf as usize].count_ones() as f64)
            .collect(),
    )
}

/// Reference build: one set of charges per peptidoform.
fn charge_states_by_set(pf_of_row: &[u32], n_pf: usize, charge: &[i32]) -> Vec<f64> {
    let mut sets: Vec<std::collections::HashSet<i32>> = vec![Default::default(); n_pf];
    for (&pf, &z) in pf_of_row.iter().zip(charge) {
        sets[pf as usize].insert(z);
    }
    pf_of_row
        .iter()
        .map(|&pf| sets[pf as usize].len() as f64)
        .collect()
}

/// Contiguous row spans of the chromatogram table that CAN hold a confident candidate,
/// from the parquet row-group statistics on `candidate_id` alone. `confident` must be
/// sorted ascending.
///
/// A group whose `[min, max]` contains no confident id contains no confident ROW either,
/// because min/max bound every value in the group, so skipping it changes nothing the
/// caller would have kept. The converse is only conservative: a kept group may hold no
/// confident row after all, and the row filter drops those as it always did.
///
/// A candidate's rows are contiguous (`plan_chunks` verifies exactly that), so a confident
/// candidate that spans two row groups puts its id inside BOTH groups' ranges and both are
/// kept; a confident candidate is therefore never split across a span boundary.
/// `pruned_confident_bounds_equal_the_whole_table_scan` exercises exactly that on a fixture
/// whose candidates do straddle group boundaries.
///
/// HOW MUCH THIS SAVES ON REAL DATA: close to nothing, and the comment says so rather than
/// carrying the fixture's figure. `extract` writes the chromatogram table in
/// `CHROM_ROW_GROUP_ROWS = 1 << 16` row groups, so one group spans thousands of candidates,
/// while the confident set is every seed PSM at `spectrum_q <= 0.01` -- on the AIF benchmark
/// about 21,856 candidates spread over the whole id range. A group is kept if it holds ONE
/// of them, so at a few per cent confident and thousands of candidates per group essentially
/// every group is kept and this returns one span covering the table.
/// `row_group_pruning_saves_nothing_on_a_production_shaped_table` measures that case and
/// pins it. The pruning is kept anyway because it is exact, because it costs one pass over
/// a footer the caller has already parsed (and, since `confident_global_bounds` takes
/// spans from the open handle, no extra file open), and because it does pay on the shapes
/// where the confident set is sparse or clustered in id -- a small seed set, a per-group
/// search, a re-run over a subset. It is not a reason to expect the 88 GB back.
///
/// Returns `None` when the statistics cannot be used -- a writer that recorded none, a file
/// with no row groups, or a range that cannot describe a `u32` column -- which means "read
/// the whole table", the previous behaviour.
fn confident_row_spans(
    stats: &[mumdia_io::table::RowGroupStats],
    confident: &[u32],
) -> Option<Vec<(usize, usize)>> {
    if stats.is_empty() || stats.iter().any(|s| s.min.is_none() || s.max.is_none()) {
        return None;
    }
    // `candidate_id` is u32, so a negative bound or an inverted range means the statistics
    // were not read as the unsigned values they are. Skipping a group on a range like that
    // would silently drop a confident anchor and shift the global half-widths that then
    // bound EVERY candidate, so refuse the pruning instead and read the whole table.
    if stats
        .iter()
        .any(|s| s.min < Some(0.0) || s.min > s.max || s.max > Some(u32::MAX as f64))
    {
        return None;
    }
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    for s in stats {
        let (lo, hi) = (s.min?, s.max?);
        // First confident id at or above the group's minimum; the group is wanted when
        // that id is also at or below its maximum. u32 is exact in f64.
        let i = confident.partition_point(|&c| (c as f64) < lo);
        if confident.get(i).is_some_and(|&c| (c as f64) <= hi) {
            match spans.last_mut() {
                Some(last) if last.0 + last.1 == start => last.1 += s.rows,
                _ => spans.push((start, s.rows)),
            }
        }
        start += s.rows;
    }
    Some(spans)
}

/// Concurrent decoders in the confident-bounds pass, PROCESS-WIDE. Each holds one decoded
/// batch plus the parquet reader's row group, so this is a memory bound as much as a
/// thread bound: the pass is pure decode and would otherwise scale to every core, at one
/// ~65,536-row row group of traces each (~115 MB at the HYE shape). Eight keeps the pass
/// under about a gigabyte, well below the chunk loop's peak. What that buys at the stage
/// level is NOT measured: the only number is `bench_confident_bounds_serial_against_parallel`,
/// a page-cache-warm 105 MB fixture whose caveats are written on it, so the docs/27
/// section 3.4 stage timing would have to be re-measured before this is quoted as a
/// stage-level saving.
///
/// Process-wide rather than per call because [`run`] is itself invoked from inside a
/// rayon `par_iter` on two supported paths -- `run_groups` when `groups.parallel > 1`, and
/// `run_experiment` when `experiment.parallel_runs > 1` -- and
/// `rayon::current_num_threads` reports the whole pool regardless of nesting, so a
/// per-call cap of eight would have been eight decoders PER BAND. At `groups.parallel = 4`
/// that is 32 concurrent row groups, ~3.7 GB, on top of whatever the bands already hold.
/// [`DecoderLease`] is the budget that makes the figure above true off the default as
/// well; see it for the exact bound, which is not quite eight.
const BOUNDS_DECODERS: usize = 8;

/// Decoders of the [`BOUNDS_DECODERS`] budget not currently leased.
static BOUNDS_DECODER_BUDGET: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(BOUNDS_DECODERS);

/// A reservation against [`BOUNDS_DECODER_BUDGET`], returned to it on drop.
///
/// Take-what-is-left rather than a blocking semaphore: a lease is acquired from inside a
/// rayon worker, and a worker that blocks cannot steal, so waiting for a permit would
/// stall the enclosing band-level or run-level parallelism (and would make progress depend
/// on the pool being large enough to run every permit holder). A call that finds the
/// budget empty gets zero and decodes on one thread, which is the serial pass.
///
/// The bound this gives is `BOUNDS_DECODERS + (C - 1)` concurrently open decoders, for `C`
/// concurrent calls: the budget covers the first eight, and every call that got nothing
/// still runs its own single decoder. At `groups.parallel = 4` that is 11 rather than 32,
/// and at the default `C = 1` it is exactly eight.
struct DecoderLease {
    budget: &'static std::sync::atomic::AtomicUsize,
    n: usize,
}

impl DecoderLease {
    /// Take up to `want` decoders, however many the budget still holds. Never blocks.
    fn take(want: usize) -> Self {
        Self::take_from(&BOUNDS_DECODER_BUDGET, want)
    }

    /// [`DecoderLease::take`] against an explicit budget, so the arithmetic can be tested
    /// on a budget of its own rather than on the shared one every other test competes for.
    fn take_from(budget: &'static std::sync::atomic::AtomicUsize, want: usize) -> Self {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        let mut cur = budget.load(Acquire);
        loop {
            let n = want.min(cur);
            match budget.compare_exchange_weak(cur, cur - n, AcqRel, Acquire) {
                Ok(_) => return DecoderLease { budget, n },
                Err(seen) => cur = seen,
            }
        }
    }

    /// Threads to decode on: the lease, or one when the budget was empty.
    fn decoders(&self) -> usize {
        self.n.max(1)
    }

    /// What the budget actually granted, which may be zero.
    fn granted(&self) -> usize {
        self.n
    }
}

impl Drop for DecoderLease {
    fn drop(&mut self) {
        self.budget
            .fetch_add(self.n, std::sync::atomic::Ordering::Release);
    }
}

/// The `(first_row, n_rows)` sub-chunks the confident-bounds pass reads, on the ABSOLUTE
/// `chunk_rows` grid rather than one restarted per span.
///
/// The grid matters for equality, not for cost: a confident candidate straddling a
/// boundary is bounded from each part separately, so the half-width samples depend on
/// where the boundaries fall. Keeping them on the absolute grid is what makes the pruned
/// pass, the whole-table pass and the parallel pass produce the same multiset.
fn confident_subchunks(spans: &[(usize, usize)], chunk_rows: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for &(first, n_rows) in spans {
        let (mut abs, end) = (first, first + n_rows);
        while abs < end {
            let want = (chunk_rows - abs % chunk_rows).min(end - abs);
            out.push((abs, want));
            abs += want;
        }
    }
    out
}

/// Half-width samples from one sub-chunk, appended to `lefts` / `rights`.
///
/// `ch.span` rather than `TableFile::open_rows`: the caller already holds an open handle
/// with the footer parsed, and `span` is an `Arc` clone of it. `open_rows` is `open` plus
/// `span`, so it re-opened the file and re-parsed the whole footer once per span -- and
/// the pruning's whole point is to produce MANY spans, so the cost grew with the saving.
#[allow(clippy::too_many_arguments)]
fn subchunk_half_widths(
    ch: &TableFile,
    first: usize,
    n_rows: usize,
    keep: &std::collections::HashSet<u32>,
    confident_rows: &HashMap<u32, Vec<usize>>,
    apex_rt: &[f64],
    cfg: &FeaturesConfig,
    names: &mut NameTab,
    lefts: &mut Vec<f64>,
    rights: &mut Vec<f64>,
) -> Result<()> {
    let span = ch.span(first, n_rows)?;
    let mut stream = ChromStream::open(&span)?;
    // Chunk reading already groups rows by candidate, so this reuses it and keeps only
    // the confident candidates' rows long enough to bound their peak.
    let chunk = stream.read_chunk_filtered(n_rows, names, Some(keep))?;
    for (ci, &c) in chunk.cids.iter().enumerate() {
        let Some(psm_rows) = confident_rows.get(&c) else {
            continue;
        };
        let rows = chunk.rows(&chunk.frag, ci, names);
        if rows.is_empty() {
            continue;
        }
        for &i in psm_rows {
            if let Some((lo, hi)) = elution_peak_rt_bounds(
                &rows,
                apex_rt[i],
                cfg.bound_peak_fraction,
                cfg.bound_peak_grace,
            ) {
                let l = apex_rt[i] - lo as f64;
                let r = hi as f64 - apex_rt[i];
                if l >= 0.0 && r >= 0.0 {
                    lefts.push(l);
                    rights.push(r);
                }
            }
        }
    }
    Ok(())
}

/// Every confident candidate's half-width samples, in table order.
///
/// This pass is a full decode of the chromatogram table (~68 GB of traces at the HYE
/// benchmark shape) to learn TWO scalars from the ~0.84% of rows that belong to a
/// confident candidate, and it ran on one thread: docs/27 section 3.4 measured the
/// features stage at 7:15 with it against 5:05 without, so 30% of the stage. Row-group
/// pruning does not help on a production-shaped table (`confident_row_spans` explains
/// why, and `row_group_pruning_saves_nothing_on_a_production_shaped_table` pins it), so
/// the decode is split across the threads a [`DecoderLease`] grants instead, at most
/// [`BOUNDS_DECODERS`]. Each group of sub-chunks is decoded in order by one thread with
/// its own [`NameTab`], and the parts are concatenated in order, so this produces the same
/// samples in the same ORDER the serial loop produced -- not merely the same multiset that
/// `percentile` would need. The decoder count is therefore a memory knob only: it changes
/// how the sub-chunks are grouped, never which sub-chunks are read nor in what order their
/// samples land, which is what
/// `parallel_confident_bounds_match_the_serial_pass_sample_for_sample` executes at four
/// pool sizes.
fn confident_half_widths(
    ch: &TableFile,
    spans: &[(usize, usize)],
    confident_rows: &HashMap<u32, Vec<usize>>,
    apex_rt: &[f64],
    cfg: &FeaturesConfig,
    chunk_rows: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let keep: std::collections::HashSet<u32> = confident_rows.keys().copied().collect();
    let subs = confident_subchunks(spans, chunk_rows);
    // `par_chunks` rather than a per-sub-chunk `par_iter`: the group count IS the decoder
    // count, so this call never has more row groups resident than it leased, whatever the
    // pool size. The groups are equal-sized runs of equal-sized reads, so the imbalance is
    // at most one sub-chunk.
    let lease = DecoderLease::take(rayon::current_num_threads().min(BOUNDS_DECODERS));
    let per = subs.len().div_ceil(lease.decoders()).max(1);
    // `collect::<Vec<Result<_>>>` rather than `collect::<Result<Vec<_>>>`: the latter keeps
    // whichever error a worker recorded first, so a run with two failing sub-chunk groups
    // reported an arbitrary one of them. Taking the first `Err` in group order restores the
    // serial pass's diagnostic. The success path's order was already preserved.
    let parts: Vec<Result<(Vec<f64>, Vec<f64>)>> = subs
        .par_chunks(per)
        .map(|group| {
            let mut names = NameTab::default();
            let (mut lefts, mut rights) = (Vec::new(), Vec::new());
            for &(first, n_rows) in group {
                subchunk_half_widths(
                    ch,
                    first,
                    n_rows,
                    &keep,
                    confident_rows,
                    apex_rt,
                    cfg,
                    &mut names,
                    &mut lefts,
                    &mut rights,
                )?;
            }
            Ok((lefts, rights))
        })
        .collect();
    drop(lease);
    let mut lefts: Vec<f64> = Vec::new();
    let mut rights: Vec<f64> = Vec::new();
    for part in parts {
        let (mut l, mut r) = part?;
        lefts.append(&mut l);
        rights.append(&mut r);
    }
    Ok((lefts, rights))
}

/// The pass exactly as it ran before it was parallelised: one stream per span, chunks
/// read from it in sequence, one [`NameTab`] for the whole pass. Kept as the reference
/// `parallel_confident_bounds_match_the_serial_pass_sample_for_sample` compares against,
/// so the equality claim is pinned against the OLD code rather than against a second
/// description of the new one.
#[cfg(test)]
fn confident_half_widths_serial(
    ch: &TableFile,
    spans: &[(usize, usize)],
    confident_rows: &HashMap<u32, Vec<usize>>,
    apex_rt: &[f64],
    cfg: &FeaturesConfig,
    chunk_rows: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut lefts: Vec<f64> = Vec::new();
    let mut rights: Vec<f64> = Vec::new();
    let mut names = NameTab::default();
    let keep: std::collections::HashSet<u32> = confident_rows.keys().copied().collect();
    for &(first, n_rows) in spans {
        let span = ch.span(first, n_rows)?;
        let mut stream = ChromStream::open(&span)?;
        let (mut abs, end) = (first, first + n_rows);
        while abs < end {
            let want = (chunk_rows - abs % chunk_rows).min(end - abs);
            let chunk = stream.read_chunk_filtered(want, &mut names, Some(&keep))?;
            abs += want;
            for (ci, &c) in chunk.cids.iter().enumerate() {
                let Some(psm_rows) = confident_rows.get(&c) else {
                    continue;
                };
                let rows = chunk.rows(&chunk.frag, ci, &names);
                if rows.is_empty() {
                    continue;
                }
                for &i in psm_rows {
                    if let Some((lo, hi)) = elution_peak_rt_bounds(
                        &rows,
                        apex_rt[i],
                        cfg.bound_peak_fraction,
                        cfg.bound_peak_grace,
                    ) {
                        let l = apex_rt[i] - lo as f64;
                        let r = hi as f64 - apex_rt[i];
                        if l >= 0.0 && r >= 0.0 {
                            lefts.push(l);
                            rights.push(r);
                        }
                    }
                }
            }
        }
    }
    Ok((lefts, rights))
}

/// Global elution half-widths from the confident set, computed in one streaming pass that
/// holds a single candidate's rows at a time. Returns None when fewer than 20 confident
/// anchors have a resolvable peak (the caller then keeps per-candidate detection).
///
/// `spans` are the row spans to read, from [`confident_row_spans`]. The whole table used
/// to be decoded here -- up to 88 GB of traces to learn two scalars from a few thousand
/// candidates -- with every row of a non-confident candidate decoded and thrown away.
/// Chunk boundaries inside a span stay on the same ABSOLUTE row grid the whole-table pass
/// used, so a candidate that straddles one is split exactly where it was split before and
/// the half-width samples, and therefore the percentiles, are unchanged.
fn confident_global_bounds(
    ch: &TableFile,
    spans: &[(usize, usize)],
    confident_rows: &HashMap<u32, Vec<usize>>,
    apex_rt: &[f64],
    cfg: &FeaturesConfig,
    chunk_rows: usize,
) -> Result<Option<(f64, f64)>> {
    let (lefts, rights) =
        confident_half_widths(ch, spans, confident_rows, apex_rt, cfg, chunk_rows)?;
    Ok(bounds_from_samples(&BoundSamples { lefts, rights }, cfg))
}

/// One caller's confident-anchor elution half-widths, in seconds, before the percentile.
///
/// A banded search holds these per band, so the run pools them and fits once; see
/// [`bounds_from_samples`].
#[derive(Default, Clone)]
pub struct BoundSamples {
    pub lefts: Vec<f64>,
    pub rights: Vec<f64>,
}

impl BoundSamples {
    /// Absorb another band's anchors. Order does not matter: both vectors are reduced by
    /// a percentile, which sorts.
    pub fn absorb(&mut self, other: BoundSamples) {
        self.lefts.extend(other.lefts);
        self.rights.extend(other.rights);
    }

    pub fn len(&self) -> usize {
        self.lefts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lefts.is_empty()
    }
}

/// The global half-widths, or `None` below 20 anchors, in which case the caller keeps the
/// per-candidate boundary detection.
///
/// The 20 is a floor on the anchor count of the RUN, not of one band. A banded search
/// splits the run's confident set across `groups.window_groups` chromatogram tables, and
/// measured on a seven-file immunopeptidomics search 735 pooled anchors arrived as 0 or 1
/// per band, so every band fell back to per-candidate detection while the unbanded arm
/// used a global window.
pub fn bounds_from_samples(s: &BoundSamples, cfg: &FeaturesConfig) -> Option<(f64, f64)> {
    if s.lefts.len() >= 20 {
        let q = (cfg.bound_confident_pct / 100.0).clamp(0.0, 1.0);
        let (l, r) = (percentile(&s.lefts, q), percentile(&s.rights, q));
        info!(
            n_confident = s.lefts.len(),
            left_hw_s = l,
            right_hw_s = r,
            pct = cfg.bound_confident_pct,
            "features: global elution half-widths from confident set"
        );
        Some((l, r))
    } else {
        warn!(
            n_confident = s.lefts.len(),
            "features: bound_from_confident set but < 20 confident anchors; falling back              to per-candidate boundary"
        );
        None
    }
}

pub fn run(p: FeaturesParams) -> Result<u64> {
    // Neither output may be one of the inputs (docs/31 F6).
    let mut inputs = vec![("--psms", p.psms), ("--chromatograms", p.chromatograms)];
    if let Some(seed) = p.seed {
        inputs.push(("--seed-psms", seed));
    }
    mumdia_io::refuse_output_over_input(p.out, &inputs)?;
    if !p.out_pin.is_empty() {
        mumdia_io::refuse_output_over_input(p.out_pin, &inputs)?;
    }
    run_with_chunk_rows(p, CHUNK_CHROM_ROWS)
}

/// [`run_with_chunk_rows`] with the PSM-row bound exposed as well; see `CHUNK_PSM_ROWS`.
/// Both limits only move chunk boundaries, which move no value.
pub fn run_with_chunk_limits(
    p: FeaturesParams,
    chunk_rows: usize,
    max_psm_rows: usize,
) -> Result<u64> {
    run_chunked(
        p,
        chunk_rows,
        max_psm_rows,
        PinFinish::Normal,
        BoundsSource::Learn,
    )
}

/// How the PIN is closed. `Fail` exists only under `cfg(test)` and injects a failure at
/// the flush, which is the one step of the stage that can fail after every chunk has been
/// computed but before the features table is published. It is a parameter rather than a
/// global so that `a_failed_pin_finish_does_not_publish_the_features_table` cannot
/// interfere with any other test in this binary.
#[derive(Clone, Copy)]
enum PinFinish {
    Normal,
    #[cfg(test)]
    Fail,
}

impl PinFinish {
    fn apply(self, w: PinWriter) -> Result<()> {
        match self {
            PinFinish::Normal => w.finish(),
            #[cfg(test)]
            PinFinish::Fail => {
                drop(w);
                Err(anyhow!("features: injected PIN flush failure"))
            }
        }
    }
}

/// [`run`] with an explicit chunk size. Only the chunk boundaries change with it: the
/// feature VALUES, the row order and the PIN bytes do not, which is what the
/// `features_chunking_is_value_preserving` test asserts -- the PIN by hash, the table
/// column by column, because a chunk is one `TableWriter::write_cols` call and the parquet
/// encoder places its data-page boundaries by accumulated bytes within each call. A run
/// whose chunk boundaries move therefore writes a features.parquet with the same values
/// and different BYTES once the table exceeds one row group
/// (`moving_a_chunk_boundary_moves_parquet_bytes_above_one_row_group` measures it), and
/// both chunk limits move boundaries. Nothing downstream reads those bytes.
pub fn run_with_chunk_rows(p: FeaturesParams, chunk_rows: usize) -> Result<u64> {
    run_chunked(
        p,
        chunk_rows,
        CHUNK_PSM_ROWS,
        PinFinish::Normal,
        BoundsSource::Learn,
    )
}

/// [`run`] with the run's pooled confident elution half-widths supplied.
///
/// A banded search calls [`confident_bound_samples`] on every band, pools the samples and
/// fits [`bounds_from_samples`] once, then passes the pair here. Without it each band fits
/// its own and falls below the 20-anchor floor: measured on a seven-file
/// immunopeptidomics search, 735 pooled anchors arrived as 0 or 1 per band.
pub fn run_with_bounds(p: FeaturesParams, bounds: Option<(f64, f64)>) -> Result<u64> {
    let mut inputs = vec![("--psms", p.psms), ("--chromatograms", p.chromatograms)];
    if let Some(seed) = p.seed {
        inputs.push(("--seed-psms", seed));
    }
    mumdia_io::refuse_output_over_input(p.out, &inputs)?;
    if !p.out_pin.is_empty() {
        mumdia_io::refuse_output_over_input(p.out_pin, &inputs)?;
    }
    run_chunked(
        p,
        CHUNK_CHROM_ROWS,
        CHUNK_PSM_ROWS,
        PinFinish::Normal,
        BoundsSource::Given(bounds),
    )
}

/// This table's confident anchors' elution half-widths, without computing any feature.
///
/// The band half of the pooled bounds: it runs exactly the pass [`run`] would run
/// internally -- same confident set, same row-group pruning, same half-width detector --
/// and returns the samples instead of the percentile.
pub fn confident_bound_samples(p: &FeaturesParams) -> Result<BoundSamples> {
    if !p.cfg.bound_from_confident {
        return Ok(BoundSamples::default());
    }
    let Some(seed) = p.seed else {
        return Ok(BoundSamples::default());
    };
    let ps = TableFile::open(p.psms)?;
    let cid = ps.u32("candidate_id")?;
    let apex_rt = ps.f64("apex_rt")?;

    let s = TableFile::open(seed)?;
    let scid = s.u32("candidate_id")?;
    let sq = s.f64("spectrum_q")?;
    let slabel = s.str("label")?;
    let mut confident: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for i in 0..s.nrows {
        if sq[i] <= 0.01 && slabel[i] == "target" {
            confident.insert(scid[i]);
        }
    }

    let mut confident_rows: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &c) in cid.iter().enumerate() {
        if confident.contains(&c) {
            confident_rows.entry(c).or_default().push(i);
        }
    }
    let ch = TableFile::open(p.chromatograms)?;
    let mut sorted: Vec<u32> = confident_rows.keys().copied().collect();
    sorted.sort_unstable();
    let spans = ch
        .row_group_stats("candidate_id")
        .ok()
        .and_then(|st| confident_row_spans(&st, &sorted))
        .unwrap_or_else(|| vec![(0, ch.nrows)]);
    let (lefts, rights) = confident_half_widths(
        &ch,
        &spans,
        &confident_rows,
        &apex_rt,
        p.cfg,
        CHUNK_CHROM_ROWS,
    )?;
    Ok(BoundSamples { lefts, rights })
}

/// Where a call gets the global elution half-widths from.
#[derive(Clone, Copy)]
enum BoundsSource {
    /// Learn them from this table's own confident anchors, which is right when the table
    /// is the whole run.
    Learn,
    /// Use the run's pooled pair. `None` means the pooled set was below the anchor floor,
    /// so every band keeps per-candidate detection, together rather than band by band.
    Given(Option<(f64, f64)>),
}

/// Main-pass loaders beyond each pass's first, PROCESS-WIDE, for the reason
/// [`BOUNDS_DECODERS`] is: [`run`] is called from inside a rayon `par_iter` when
/// `groups.parallel > 1` or `experiment.parallel_runs > 1`, and each loader holds one
/// decoded chunk (0.92 GiB of traces at the HYE benchmark shape), so a per-pass allowance
/// would multiply with the concurrency. Every pass keeps the one loader it always had and
/// takes up to `features.chrom_loaders - 1` more from this pool, whatever is left of it, so
/// `C` concurrent passes run at most `C + MAIN_LOADER_EXTRAS` loaders.
const MAIN_LOADER_EXTRAS: usize = 4;

/// Extra main-pass loaders of the [`MAIN_LOADER_EXTRAS`] pool not currently leased.
static MAIN_LOADER_BUDGET: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(MAIN_LOADER_EXTRAS);

/// Decode one planned chunk -- `n_rows` chromatogram rows from `first` -- on its own, from
/// its row span, with a fragment-name table of its own.
///
/// The chunk is the same whichever thread reads it and whatever was read before it: its
/// rows are fixed by the plan, [`ChromChunk`] is built from them in table order, and the
/// candidate grouping, the axis dedup and every stored value depend on those rows alone.
/// The name ids do differ from a table shared across chunks, but nothing reads an id
/// except to look its string up in the table the chunk travels with
/// ([`ChromChunk::rows`]). A zero-row chunk (PSM rows with no chromatogram rows) is an
/// empty chunk, as the sequential stream returned for it.
///
/// A span decodes whole row groups, so a chunk that starts or ends inside one decodes
/// and discards the rest of it: at most one 65,536-row group at each end of a ~1M-row
/// chunk, against the single sequential stream that decoded every row once.
fn load_chunk(ch: &TableFile, first: usize, n_rows: usize) -> Result<(ChromChunk, NameTab)> {
    let mut names = NameTab::default();
    if n_rows == 0 {
        return Ok((ChromChunk::new(), names));
    }
    let span = ch.span(first, n_rows)?;
    let mut stream = ChromStream::open(&span)?;
    let chunk = stream.read_chunk(n_rows, &mut names)?;
    Ok((chunk, names))
}

/// The main pass's chromatogram loaders and the ordered hand-over to the computation.
///
/// `loaders` threads each claim the next unclaimed chunk, decode it with [`load_chunk`] and
/// park it; the computation takes chunk `j` only after chunk `j - 1`, so the features are
/// computed, assembled and written in table order exactly as with one sequential loader.
/// A loader may claim chunk `j` only while `j < taken + loaders`, where `taken` is the
/// number of chunks the computation has taken, so at most `loaders` chunks are decoded or
/// being decoded ahead of it, and `loaders + 1` are resident with the one it is computing.
/// With one loader that is the previous bound exactly: one chunk computed, one decoded.
///
/// Errors keep the order too. A failed chunk is parked like any other and no further chunk
/// is claimed; the chunks before it were claimed earlier (claims are in order), so the
/// computation receives them first and then the error, at the same chunk as before. A
/// loader that dies without parking its chunk (a panic) is counted out, so the computation
/// reports that the loaders stopped instead of waiting for a chunk that will never come.
struct ChunkLoader<'a> {
    ch: &'a TableFile,
    first: &'a [usize],
    rows: &'a [usize],
    loaders: usize,
    state: std::sync::Mutex<LoadState>,
    cv: std::sync::Condvar,
}

/// [`ChunkLoader`]'s shared state; see there.
struct LoadState {
    /// The next chunk no loader has claimed.
    next: usize,
    /// Chunks the computation has taken.
    taken: usize,
    /// Decoded chunks the computation has not taken yet, by chunk index. An ordered map so
    /// that nothing about the hand-over depends on hash iteration.
    ready: std::collections::BTreeMap<usize, Result<(ChromChunk, NameTab)>>,
    /// Claim nothing more: a chunk failed, or the computation has stopped.
    stop: bool,
    /// Loaders still running.
    live: usize,
}

impl<'a> ChunkLoader<'a> {
    fn new(
        ch: &'a TableFile,
        first: &'a [usize],
        rows: &'a [usize],
        loaders: usize,
    ) -> ChunkLoader<'a> {
        ChunkLoader {
            ch,
            first,
            rows,
            loaders: loaders.max(1),
            state: std::sync::Mutex::new(LoadState {
                next: 0,
                taken: 0,
                ready: std::collections::BTreeMap::new(),
                stop: false,
                live: loaders.max(1),
            }),
            cv: std::sync::Condvar::new(),
        }
    }

    /// The state, whether or not another thread panicked holding it: every update below
    /// leaves the state consistent before it can panic, and the only panics possible while
    /// the lock is held are allocation failures.
    fn lock(&self) -> std::sync::MutexGuard<'_, LoadState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn wait<'g>(
        &self,
        g: std::sync::MutexGuard<'g, LoadState>,
    ) -> std::sync::MutexGuard<'g, LoadState> {
        self.cv
            .wait(g)
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// One loader thread's loop: claim, decode, park, until every chunk is claimed or the
    /// pass stops.
    fn run(&self, timers: &PassTimers) {
        // Counted out on every exit, a panic included, so `take` cannot wait forever. A
        // panic also stops the pass: the chunk it was decoding is never parked, so the
        // other loaders would otherwise fill the window past it and wait for a computation
        // that is itself waiting for the missing chunk.
        struct Exit<'l, 'a>(&'l ChunkLoader<'a>);
        impl Drop for Exit<'_, '_> {
            fn drop(&mut self) {
                let mut st = self.0.lock();
                st.live -= 1;
                if std::thread::panicking() {
                    st.stop = true;
                }
                drop(st);
                self.0.cv.notify_all();
            }
        }
        let _exit = Exit(self);
        loop {
            let j = {
                let mut st = self.lock();
                loop {
                    if st.stop || st.next >= self.rows.len() {
                        return;
                    }
                    if st.next < st.taken + self.loaders {
                        break;
                    }
                    let blocked = Instant::now();
                    st = self.wait(st);
                    timers.add(&timers.loader_blocked_ns, blocked);
                }
                st.next += 1;
                // The window: never more than `loaders` chunks claimed ahead of the
                // computation, which is what bounds the resident chunks at `loaders + 1`.
                debug_assert!(st.next - st.taken <= self.loaders);
                st.next - 1
            };
            let busy = Instant::now();
            let r = load_chunk(self.ch, self.first[j], self.rows[j]);
            timers.add(&timers.loader_busy_ns, busy);
            let mut st = self.lock();
            if r.is_err() {
                st.stop = true;
            }
            st.ready.insert(j, r);
            drop(st);
            self.cv.notify_all();
        }
    }

    /// Chunk `j`, blocking until a loader has parked it. `None` when every loader has
    /// exited without parking it, which only a loader panic can cause.
    fn take(&self, j: usize) -> Option<Result<(ChromChunk, NameTab)>> {
        let mut st = self.lock();
        loop {
            if let Some(r) = st.ready.remove(&j) {
                st.taken += 1;
                drop(st);
                self.cv.notify_all();
                return Some(r);
            }
            if st.live == 0 {
                return None;
            }
            st = self.wait(st);
        }
    }

    /// Stop the loaders when the returned guard drops: nothing more is claimed and every
    /// loader waiting for the window wakes and exits. A loader in the middle of a decode
    /// finishes it first, and its chunk is dropped with the loader.
    fn release_on_drop(&self) -> impl Drop + '_ {
        struct Release<'l, 'a>(&'l ChunkLoader<'a>);
        impl Drop for Release<'_, '_> {
            fn drop(&mut self) {
                self.0.lock().stop = true;
                self.0.cv.notify_all();
            }
        }
        Release(self)
    }
}

/// Where the chunked pass spends its time (perf survey P0), so the next optimisation can
/// be sized from a log line instead of from a cost model.
///
/// The pass is a three-stage pipeline -- loader thread(s) decoding chromatogram chunks,
/// the calling thread computing and assembling the features of one chunk, a writer thread
/// encoding the previous chunk's columns -- and its wall time is set by whichever stage is
/// busiest. Each stage records its BUSY time and the time it spent BLOCKED on a
/// neighbour, so the binding stage is the one that is busy while the others wait on it: a
/// pass whose `wait_for_loader_ms` is large and whose `loader_blocked_ms` is small is
/// decode-bound, and one whose `loader_blocked_ms` and `writer_idle_ms` are both large
/// while `compute_ms` dominates is compute-bound. Busy time is summed over threads, so
/// with several loaders it can exceed the wall time.
///
/// Relaxed atomics: the counters are read once, after every thread has been joined.
/// Nothing here feeds a value, a file or a hash; the only output is the log line.
#[derive(Default)]
struct PassTimers {
    /// Loader threads decoding and storing chunks, summed over loaders.
    loader_busy_ns: std::sync::atomic::AtomicU64,
    /// Loader threads waiting for the computation to take a chunk, because the decode
    /// window ([`ChunkLoader`]) is full; summed over loaders.
    loader_blocked_ns: std::sync::atomic::AtomicU64,
    /// The computation waiting for the next chunk to be decoded.
    wait_loader_ns: std::sync::atomic::AtomicU64,
    /// The parallel per-PSM kernels (`fragment_features`, `build_evidence`, families).
    compute_ns: std::sync::atomic::AtomicU64,
    /// The serial assembly of the value matrix, the PIN and the output columns.
    assemble_ns: std::sync::atomic::AtomicU64,
    /// The computation waiting for the writer to take the finished columns.
    wait_writer_ns: std::sync::atomic::AtomicU64,
    /// The writer encoding columns (and closing the file).
    writer_busy_ns: std::sync::atomic::AtomicU64,
    /// The writer waiting for the next chunk's columns.
    writer_idle_ns: std::sync::atomic::AtomicU64,
}

impl PassTimers {
    fn add(&self, slot: &std::sync::atomic::AtomicU64, since: Instant) {
        let ns = u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX);
        slot.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    }

    fn ms(slot: &std::sync::atomic::AtomicU64) -> u64 {
        slot.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000
    }

    /// The stage that was busiest per thread: the loaders' busy time is shared across
    /// `loaders` threads, the computation's and the writer's belong to one thread each.
    fn binding(&self, loaders: usize) -> &'static str {
        let loader = Self::ms(&self.loader_busy_ns) / loaders.max(1) as u64;
        let compute = Self::ms(&self.compute_ns) + Self::ms(&self.assemble_ns);
        let writer = Self::ms(&self.writer_busy_ns);
        if loader >= compute && loader >= writer {
            "loader"
        } else if writer >= compute {
            "writer"
        } else {
            "compute"
        }
    }

    fn log(&self, wall: std::time::Duration, loaders: usize, chunks: usize) {
        info!(
            wall_ms = u64::try_from(wall.as_millis()).unwrap_or(u64::MAX),
            chunks,
            loaders,
            loader_busy_ms = Self::ms(&self.loader_busy_ns),
            loader_blocked_ms = Self::ms(&self.loader_blocked_ns),
            wait_for_loader_ms = Self::ms(&self.wait_loader_ns),
            compute_ms = Self::ms(&self.compute_ns),
            assemble_ms = Self::ms(&self.assemble_ns),
            wait_for_writer_ms = Self::ms(&self.wait_writer_ns),
            writer_busy_ms = Self::ms(&self.writer_busy_ns),
            writer_idle_ms = Self::ms(&self.writer_idle_ns),
            binding = self.binding(loaders),
            "features: pass timers"
        );
    }
}

fn run_chunked(
    p: FeaturesParams,
    chunk_rows: usize,
    max_psm_rows: usize,
    pin_finish: PinFinish,
    bounds: BoundsSource,
) -> Result<u64> {
    let t0 = Instant::now();
    let ps = TableFile::open(p.psms)?;
    let cid = ps.u32("candidate_id")?;
    // Top-K peak rank (#7), passed through untouched. Missing in pre-v2 extracted
    // artifacts -> 0 (the selected apex), so old inputs behave exactly as before.
    let peak_rank = ps.i32("peak_rank").unwrap_or_else(|_| vec![0; ps.nrows]);
    // Demix features (D2), absent unless extract.emit_demix_features -> 0.
    let deconv_expl = ps
        .f32("deconv_explained_frac")
        .unwrap_or_else(|_| vec![0.0; ps.nrows]);
    let deconv_act = ps
        .f32("deconv_active")
        .unwrap_or_else(|_| vec![0.0; ps.nrows]);
    let deconv_shr = ps
        .f32("deconv_share")
        .unwrap_or_else(|_| vec![0.0; ps.nrows]);
    let deconv_col = ps
        .f32("deconv_max_collinearity")
        .unwrap_or_else(|_| vec![0.0; ps.nrows]);
    let deconv_sha = ps
        .f32("shadow_kept_frac")
        .unwrap_or_else(|_| vec![0.0; ps.nrows]);
    let apex_rt = ps.f64("apex_rt")?;
    let apex_int = ps.f32("apex_intensity")?;
    let n_matched = ps.i32("n_matched_fragments")?;
    let n_pred = ps
        .i32("n_predicted_fragments")
        .unwrap_or_else(|_| vec![6; ps.nrows]);
    let corun = ps.i32("coelution_run")?;
    let rt_cal = ps.f64("rt_pred_cal")?;
    let charge = ps.i32("charge")?;
    let label = ps.str("label")?;
    let base = ps.u32("base_peptide_id")?;
    let pform = ps.str("peptidoform")?;
    let protein = ps.str("protein")?;
    let mz = ps.f64("precursor_mz")?;
    let contested = ps
        .f64("contested_frac")
        .unwrap_or_else(|_| vec![0.0; ps.nrows]);
    // Richer soft-competition columns (present only with emit_contested_features;
    // default to 0 so the feature vector length is stable when absent).
    let contested_count = ps
        .f64("contested_count_frac")
        .unwrap_or_else(|_| vec![0.0; ps.nrows]);
    let apportioned = ps
        .f64("apportioned_frac")
        .unwrap_or_else(|_| vec![0.0; ps.nrows]);
    let ms1_m1 = ps
        .opt_f64("ms1_isom1")
        .unwrap_or_else(|_| vec![None; ps.nrows]);
    let ms1_mono = ps
        .opt_f64("ms1_mono")
        .unwrap_or_else(|_| vec![None; ps.nrows]);
    let ms1_i1 = ps
        .opt_f64("ms1_iso1")
        .unwrap_or_else(|_| vec![None; ps.nrows]);
    let ms1_i2 = ps
        .opt_f64("ms1_iso2")
        .unwrap_or_else(|_| vec![None; ps.nrows]);

    // The chromatogram table is the largest artifact of the run: 62.7 GiB of traces at
    // 2.6M candidates on the HYE benchmark, where the whole store used to be materialised
    // before the first feature was computed. It is now processed in chunks of
    // CHUNK_CHROM_ROWS rows, so what is resident is one chunk of traces (about 1 GiB) plus
    // that chunk's feature values. The plan below fixes the chunk boundaries from the
    // candidate_id column alone, which is the only part of the table read up front.
    let ch = TableFile::open(p.chromatograms)?;
    let ch_cid = ch.u32("candidate_id")?;
    let chrom_rows_total = ch_cid.len();
    let chunks = plan_chunks(
        &cid,
        &ch_cid,
        p.chromatograms,
        chunk_rows.max(1),
        max_psm_rows.max(1),
    )?;
    drop(ch_cid);

    // Seed corroboration maps (candidate_id -> seed score / identified flag) plus the
    // confident-target candidate set (spectrum_q <= 0.01, label == target) used to
    // learn a global elution half-width when `bound_from_confident` is set. This
    // mirrors the RT-calibration / DeepLC-fine-tune anchor set (rt_im_train.rs).
    let (seed_score_map, seed_id_map, confident_cids): (
        HashMap<u32, f64>,
        HashMap<u32, f64>,
        std::collections::HashSet<u32>,
    ) = match p.seed {
        Some(path) => {
            let s = TableFile::open(path)?;
            let scid = s.u32("candidate_id")?;
            let ssc = s.f64("score")?;
            let sq = s.f64("spectrum_q")?;
            let slabel = s.str("label")?;
            let mut sm = HashMap::new();
            let mut im = HashMap::new();
            let mut conf = std::collections::HashSet::new();
            for i in 0..s.nrows {
                sm.insert(scid[i], ssc[i]);
                im.insert(scid[i], if sq[i] <= 0.01 { 1.0 } else { 0.0 });
                if sq[i] <= 0.01 && slabel[i] == "target" {
                    conf.insert(scid[i]);
                }
            }
            (sm, im, conf)
        }
        None => (
            HashMap::new(),
            HashMap::new(),
            std::collections::HashSet::new(),
        ),
    };

    // Global elution half-widths learned once from the confident set. Some((L, R)) in
    // seconds when `bound_from_confident` and >= 20 confident anchors have a resolvable
    // peak; then every candidate's feature region is [apex - L, apex + R]. None keeps
    // the per-candidate boundary detection (default). This is a global quantity, so it
    // costs one extra streaming pass over the chromatogram table before the chunked pass
    // below; only the confident candidates' rows are ever held.
    let global_bounds: Option<(f64, f64)> = match bounds {
        // A banded run fits the half-widths once over every band's anchors and hands
        // the pair down, so a band with one anchor of its own still gets the run's
        // window. The pooled fit also makes this pass unnecessary here.
        BoundsSource::Given(b) => b,
        BoundsSource::Learn if p.cfg.bound_from_confident => {
            let mut confident_rows: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, &c) in cid.iter().enumerate() {
                if confident_cids.contains(&c) {
                    confident_rows.entry(c).or_default().push(i);
                }
            }
            // Read only the row groups whose candidate_id range can hold a confident
            // candidate; the rest contain none by construction.
            let mut sorted: Vec<u32> = confident_rows.keys().copied().collect();
            sorted.sort_unstable();
            let spans = ch
                .row_group_stats("candidate_id")
                .ok()
                .and_then(|s| confident_row_spans(&s, &sorted))
                .unwrap_or_else(|| vec![(0, ch.nrows)]);
            let span_rows: usize = spans.iter().map(|s| s.1).sum();
            info!(
                spans = spans.len(),
                rows = span_rows,
                of_rows = ch.nrows,
                confident = sorted.len(),
                "features: confident-bounds pass reads this much of the chromatogram table"
            );
            confident_global_bounds(
                &ch,
                &spans,
                &confident_rows,
                &apex_rt,
                p.cfg,
                chunk_rows.max(1),
            )?
        }
        BoundsSource::Learn => None,
    };

    let gradient = apex_rt.iter().cloned().fold(0.0f64, f64::max).max(1.0);
    let cols_active = active_features(p.cfg.set);
    let n = ps.nrows;

    // --- Cross-candidate charge-state corroboration (Extended set) ---
    // Group the extracted PSMs by peptidoform (the ProForma string is charge-
    // independent; DECOY_ peptidoforms group among themselves, so this is not a
    // target/decoy label leak). A real peptide co-occurs at multiple charge states
    // more than a shift decoy, and this evidence axis is invisible to the per-PSM
    // Evidence families since each charge is a separate candidate. It is a whole-run
    // reduction over PSM columns only, so it is computed before the chunk loop.
    //
    // The grouping used to be a `HashMap<&str, HashSet<i32>>` plus a `HashMap<&str, f64>`:
    // ONE HashSet heap block per distinct peptidoform, about a million of them, live for
    // the whole run. Peptidoforms get a dense id here instead -- the same thing compete.rs
    // does with `pform_id` -- and the two reductions become a `Vec<u32>` charge bitmask and
    // a `Vec<f64>` of sums, three allocations in total. The sums are accumulated in the same
    // row order as before, so the f64 addition order, and therefore the value, is unchanged.
    let mut pf_ids: HashMap<&str, u32> = HashMap::with_capacity(n);
    let mut pf_of_row: Vec<u32> = Vec::with_capacity(n);
    for p in pform.iter().take(n) {
        let next = pf_ids.len() as u32;
        pf_of_row.push(*pf_ids.entry(p.as_str()).or_insert(next));
    }
    let n_pf = pf_ids.len();
    drop(pf_ids);
    let mut pf_int: Vec<f64> = vec![0.0; n_pf];
    for i in 0..n {
        pf_int[pf_of_row[i] as usize] += apex_int[i] as f64;
    }
    let f_n_charge = charge_states_per_row(&pf_of_row, n_pf, &charge);
    let f_charge_multi: Vec<f64> = f_n_charge
        .iter()
        .map(|&c| if c >= 2.0 { 1.0 } else { 0.0 })
        .collect();
    // ln(1 + summed apex intensity of the OTHER charge states of this peptidoform):
    // how much independent charge-state evidence reinforces this PSM (unbounded).
    let f_cross_charge_int: Vec<f64> = (0..n)
        .map(|i| (1.0 + (pf_int[pf_of_row[i] as usize] - apex_int[i] as f64).max(0.0)).ln())
        .collect();
    drop((pf_of_row, pf_int));

    let extended = matches!(p.cfg.set, FeatureSet::Extended);
    let ext_names = extended_name_refs();

    // The two expensive per-PSM computations (`fragment_features` and, when the
    // extended set is active, `build_evidence` + `extended_values`) are pure
    // functions of that PSM's own inputs, so they are computed in parallel over the
    // chunk and indexed by row. The serial assembly below reads them back by row and
    // is otherwise unchanged, so the feature values are identical to the whole-run
    // version this replaced.
    //
    // The extended values used to come back as one `Vec<f64>` PER PSM inside a
    // `Vec<PerPsm>`: 337 f64 (2,696 B) in its own heap block, 70k-175k of them live
    // from the `collect()` to the `drop` at the end of the chunk. That is the single
    // largest producer of medium-sized live mappings in the stage, and mimalloc maps
    // blocks of that size individually, so it counts against the kernel's per-process
    // mapping limit (1,048,576) rather than against bytes. They are one flat
    // `rows x n_ext` buffer now, filled by `par_chunks_mut` and addressed by offset;
    // the values, their order and their row assignment are unchanged.
    let n_ext = if extended { ext_names.len() } else { 0 };
    // The feature -> column permutation, taken once instead of once per value per row.
    let ix = ColIx::new(&cols_active, &ext_names);

    let writer = TableWriter::new(p.out).with_row_group_rows(FEATURE_ROW_GROUP_ROWS);
    let mut pin = if p.cfg.emit_pin {
        Some(PinWriter::create(p.out_pin, &cols_active)?)
    } else {
        tracing::debug!(
            path = %p.out_pin,
            "features: PIN emission disabled (features.emit_pin = false)"
        );
        None
    };
    // Accounting for the audit (docs/27): the largest chunk in flight against the run
    // total streamed is exactly the quantity chunking changes.
    let (mut max_frag_bytes, mut max_ms1_bytes, mut max_matrix_bytes) = (0usize, 0usize, 0usize);
    // Value-buffer high-water mark across the overlap, and the bytes the writer thread is
    // holding right now (the previous chunk's moved columns; 0 before the first send).
    let (mut max_inflight_bytes, mut in_writer) = (0usize, 0usize);
    let (mut tot_frag_bytes, mut tot_ms1_bytes) = (0usize, 0usize);
    let (mut n_cand, mut n_frag_rows, mut n_ms1_rows) = (0usize, 0usize, 0usize);

    // The chunk decode is single-threaded parquet work and it ran in series with the
    // feature computation, which is what left a 24-thread features process at 1.6-2.8 cores
    // (measured on the 8-12-mer immunopeptidomics run: 3.7 of a 4-minute stage were the
    // load). It then moved onto one loader thread, one chunk ahead of the computation, so
    // decode and compute overlapped but the decode itself stayed on one core. Now
    // `chrom_loaders` threads each decode a whole chunk from that chunk's row span (see
    // [`ChunkLoader`]), and the computation takes the chunks in table order.
    let chunk_rows: Vec<usize> = chunks.iter().map(|c| c.chrom_rows).collect();
    let chunk_first: Vec<usize> = chunk_rows
        .iter()
        .scan(0usize, |acc, &n| {
            let first = *acc;
            *acc += n;
            Some(first)
        })
        .collect();
    let want_loaders = p.cfg.chrom_loaders.max(1).min(chunks.len().max(1));
    // Loaders beyond the first come from the process-wide pool; this pass's own first loader
    // is never leased, so a pass always runs even when the pool is empty.
    let extra_lease = DecoderLease::take_from(&MAIN_LOADER_BUDGET, want_loaders - 1);
    let n_loaders = 1 + extra_lease.granted();
    let loader = ChunkLoader::new(&ch, &chunk_first, &chunk_rows, n_loaders);
    let timers = PassTimers::default();
    let pass_start = Instant::now();
    let rows = std::thread::scope(|sc| -> Result<u64> {
        // The queue is created outside the scope (the loaders borrow it), so its stop
        // flag, not the drop of a channel, is what releases them when this closure returns
        // early. The guard is declared FIRST so it drops LAST among this closure's locals,
        // and it drops before the scope joins the loaders: an error return here therefore
        // wakes every loader waiting for the window to open instead of leaving the scope
        // to join threads that would never wake -- the hang the rendezvous channel was
        // once fixed for, in its new form.
        let _release = loader.release_on_drop();
        // The parquet encode is single-threaded; on its own thread it overlaps with the
        // next chunk's computation instead of stalling it. Rendezvous again, so the
        // encoder never queues a backlog of value matrices.
        //
        // `None` is the commit marker: the writer publishes the artifact only after the
        // chunk loop has sent every chunk. Without it, an error anywhere in this closure
        // would drop `wtx`, end the writer's loop normally and publish a TRUNCATED table
        // over the previous good one, where the old in-line write simply returned before
        // `close` and let `AtomicPath` remove its temp file.
        let (wtx, wrx) = std::sync::mpsc::sync_channel::<Option<Vec<Col>>>(0);
        let timers_w = &timers;
        let wh = sc.spawn(move || -> Result<u64> {
            let mut writer = writer;
            let mut commit = false;
            loop {
                let idle = Instant::now();
                let Ok(msg) = wrx.recv() else { break };
                timers_w.add(&timers_w.writer_idle_ns, idle);
                match msg {
                    Some(cols) => {
                        let busy = Instant::now();
                        writer.write_cols(cols)?;
                        timers_w.add(&timers_w.writer_busy_ns, busy);
                    }
                    None => {
                        commit = true;
                        break;
                    }
                }
            }
            if !commit {
                // Dropping the writer unpublished takes its temp file with it.
                drop(writer);
                return Err(anyhow!(
                    "features: the chunk loop stopped before the last chunk; \
                     the features table was not written"
                ));
            }
            let busy = Instant::now();
            let rows = writer.close();
            timers_w.add(&timers_w.writer_busy_ns, busy);
            rows
        });
        for _ in 0..n_loaders {
            let (loader, timers) = (&loader, &timers);
            sc.spawn(move || loader.run(timers));
        }
        for (j, chunk) in chunks.iter().enumerate() {
            let waited = Instant::now();
            let (store, names) = loader.take(j).ok_or_else(|| {
                anyhow!(
                    "features: the chromatogram loaders stopped before chunk {}..{}",
                    chunk.psm_lo,
                    chunk.psm_hi
                )
            })??;
            timers.add(&timers.wait_loader_ns, waited);
            let (lo, hi) = (chunk.psm_lo, chunk.psm_hi);
            let rows_in_chunk = hi - lo;
            let (fb, mb) = store.payload_bytes();
            max_frag_bytes = max_frag_bytes.max(fb);
            max_ms1_bytes = max_ms1_bytes.max(mb);
            tot_frag_bytes += fb;
            tot_ms1_bytes += mb;
            n_cand += store.cids.len();
            n_frag_rows += store.frag.nrows();
            n_ms1_rows += store.ms1.nrows();

            // One PSM's work, writing its extended values into `ext` (already zeroed and
            // exactly `n_ext` wide) rather than returning a fresh Vec.
            let per_psm = |i: usize, ext: &mut [f64]| -> FragFeatures {
                let ci = store.index.get(&cid[i]).copied();
                let rows = ci
                    .map(|c| store.rows(&store.frag, c, &names))
                    .unwrap_or_default();
                if rows.is_empty() {
                    // No chromatogram rows: default features and all-zero extended
                    // values, as before.
                    return FragFeatures::default();
                }
                // One alignment and one apex-intensity search per PSM, read by
                // `fragment_features` and then moved into the Evidence: they used to be
                // rebuilt inside each of them.
                let al = align_traces(&rows);
                let obs = apex_intensities(&rows, apex_rt[i]);
                if ext.is_empty() {
                    return fragment_features(
                        &rows,
                        &al,
                        &obs,
                        None,
                        apex_rt[i],
                        p.cfg.coelution_corr_threshold,
                        p.cfg.bound_features,
                        p.cfg.bound_peak_fraction,
                        p.cfg.bound_peak_grace,
                        global_bounds,
                    );
                }
                // The Extended set: the peak window, its traces, the reference profiles
                // and the pair statistics once. `fragment_features` shares them when
                // `bound_features` makes its window this one; without it, it scores the
                // whole extracted window and computes its own.
                let pred: Vec<f64> = rows.iter().map(|r| r.pred_int as f64).collect();
                let win = peak_window(
                    &al.axis_full,
                    &al.traces_full,
                    &pred,
                    apex_rt[i],
                    p.cfg.bound_peak_fraction,
                    p.cfg.bound_peak_grace,
                    global_bounds,
                );
                let peak = PeakTraces::new(&al, &pred, win);
                let ff = fragment_features(
                    &rows,
                    &al,
                    &obs,
                    p.cfg.bound_features.then_some(&peak),
                    apex_rt[i],
                    p.cfg.coelution_corr_threshold,
                    p.cfg.bound_features,
                    p.cfg.bound_peak_fraction,
                    p.cfg.bound_peak_grace,
                    global_bounds,
                );
                {
                    let ms1_rows = ci
                        .map(|c| store.rows(&store.ms1, c, &names))
                        .unwrap_or_default();
                    let mut ev = evidence_from(&rows, al, obs, pred, peak, &ms1_rows, apex_rt[i]);
                    ev.rt_pred_cal = rt_cal[i];
                    ev.rt_err = calibrated_rt_error(apex_rt[i], rt_cal[i]);
                    ev.gradient = gradient;
                    ev.precursor_mz = mz[i];
                    ev.charge = charge[i];
                    ev.seq_len = peptide_length(&pform[i]);
                    ev.n_matched = n_matched[i];
                    ev.n_predicted = n_pred[i];
                    ev.seed_score = *seed_score_map.get(&cid[i]).unwrap_or(&0.0);
                    ev.seed_identified = *seed_id_map.get(&cid[i]).unwrap_or(&0.0);
                    ev.apex_intensity = apex_int[i] as f64;
                    ev.ms1_mono = ms1_mono[i];
                    ev.ms1_iso1 = ms1_i1[i];
                    ev.ms1_iso2 = ms1_i2[i];
                    ev.ms1_isom1 = ms1_m1[i];
                    ev.ms1_precursor_features = p.cfg.ms1_precursor_features;
                    ev.deconv_explained = deconv_expl[i] as f64;
                    ev.deconv_active = deconv_act[i] as f64;
                    ev.deconv_share = deconv_shr[i] as f64;
                    ev.deconv_max_collin = deconv_col[i] as f64;
                    ev.deconv_shadow = deconv_sha[i] as f64;
                    // Truncating zip, exactly as the name/value zip in the assembly
                    // did: a family that returned fewer values leaves the tail at 0.0.
                    for (dst, v) in ext.iter_mut().zip(extended_values(&ev)) {
                        *dst = v;
                    }
                }
                ff
            };

            let computing = Instant::now();
            let mut frag_feats: Vec<FragFeatures> = vec![FragFeatures::default(); rows_in_chunk];
            let mut ext_vals: Vec<f64> = vec![0.0; rows_in_chunk * n_ext];
            if n_ext > 0 {
                frag_feats
                    .par_iter_mut()
                    .zip(ext_vals.par_chunks_mut(n_ext))
                    .enumerate()
                    .for_each(|(r, (ff, ext))| *ff = per_psm(lo + r, ext));
            } else {
                frag_feats
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(r, ff)| *ff = per_psm(lo + r, &mut []));
            }
            timers.add(&timers.compute_ns, computing);

            let assembling = Instant::now();
            let mut m = ValueMatrix::new(cols_active.len(), rows_in_chunk);
            let mut prelim = vec![0.0f64; rows_in_chunk];
            let mut elu_lo = vec![0.0f64; rows_in_chunk];
            let mut elu_hi = vec![0.0f64; rows_in_chunk];
            for (r, i) in (lo..hi).enumerate() {
                let ff = &frag_feats[r];
                elu_lo[r] = ff.elution_lo;
                elu_hi[r] = ff.elution_hi;
                let rt_err = calibrated_rt_error(apex_rt[i], rt_cal[i]);
                // MS1 isotope features.
                let neutral = mz[i] * charge[i] as f64 - charge[i] as f64 * PROTON;
                let (iso_corr, isom1_ratio, log_mono, has_ms1) =
                    isotope_features(ms1_m1[i], ms1_mono[i], ms1_i1[i], ms1_i2[i], neutral);

                m.set(ix.rt_error_abs, r, rt_err);
                m.set(ix.rt_error_rel, r, rt_err / gradient);
                m.set(ix.n_matched_fragments, r, n_matched[i] as f64);
                m.set(ix.coelution_run, r, corun[i] as f64);
                m.set(ix.log_apex_intensity, r, (1.0 + apex_int[i] as f64).ln());
                m.set(ix.frag_corr, r, ff.frag_corr);
                m.set(ix.frag_cosine, r, ff.frag_cosine);
                m.set(ix.spectral_angle, r, ff.spectral_angle);
                m.set(ix.coelution_mean, r, ff.coelution_mean);
                m.set(ix.coelution_best, r, ff.coelution_best);
                m.set(ix.n_coelution_above, r, ff.n_coelution_above);
                m.set(ix.charge, r, charge[i] as f64);
                m.set(ix.peptide_length, r, peptide_length(&pform[i]) as f64);
                m.set(
                    ix.n_proteins,
                    r,
                    (protein[i].matches(';').count() + 1) as f64,
                );
                m.set(ix.library_norm_manhattan, r, ff.norm_manhattan);
                m.set(ix.library_rmsd, r, ff.rmsd);
                m.set(ix.xcorr_coelution, r, ff.xcorr_coelution);
                m.set(ix.xcorr_shape, r, ff.xcorr_shape);
                m.set(ix.sum_b_intensity, r, ff.sum_b);
                m.set(ix.sum_y_intensity, r, ff.sum_y);
                m.set(ix.diff_by_intensity, r, ff.sum_b - ff.sum_y);
                m.set(ix.n_b_ions, r, ff.n_b);
                m.set(ix.n_y_ions, r, ff.n_y);
                m.set(ix.weighted_mass_error, r, ff.weighted_mass_error);
                m.set(ix.mean_mass_error, r, ff.mean_mass_error);
                m.set(ix.isotope_corr, r, iso_corr);
                m.set(ix.ms1_isom1_ratio, r, isom1_ratio);
                m.set(ix.log_mono_ms1, r, log_mono);
                m.set(ix.has_ms1, r, has_ms1);
                m.set(ix.log_sn, r, ff.log_sn);
                m.set(ix.n_observations, r, ff.n_observations);
                m.set(ix.base_width_rt, r, ff.base_width_rt);
                m.set(
                    ix.seed_score,
                    r,
                    *seed_score_map.get(&cid[i]).unwrap_or(&0.0),
                );
                m.set(
                    ix.seed_identified,
                    r,
                    *seed_id_map.get(&cid[i]).unwrap_or(&0.0),
                );
                m.set(
                    ix.matched_fraction,
                    r,
                    n_matched[i] as f64 / (n_pred[i].max(1) as f64),
                );
                m.set(ix.profile_cos, r, ff.profile_cos);
                m.set(ix.ref_corr, r, ff.ref_corr);
                m.set(ix.best_ref_corr, r, ff.best_ref_corr);
                m.set(ix.low_frag_coel, r, ff.low_frag_coel);
                m.set(ix.evidence, r, ff.evidence);
                m.set(ix.contrast_min, r, ff.contrast_min);
                m.set(ix.resid_corr, r, ff.resid_corr);
                m.set(ix.coel_clean, r, ff.coel_clean);
                m.set(ix.shadow_frac, r, ff.shadow_frac);
                m.set(ix.peak_contested_frac, r, contested[i]);
                m.set(ix.peak_contested_count_frac, r, contested_count[i]);
                m.set(ix.peak_apportioned_frac, r, apportioned[i]);

                // Extended battery (opt-in). Built once per PSM above and fanned out to the
                // family modules; written here under the fixed registry-order columns.
                if n_ext > 0 {
                    let row = &ext_vals[r * n_ext..(r + 1) * n_ext];
                    for (c, v) in ix.ext.iter().zip(row) {
                        m.set(*c, r, *v);
                    }
                }

                // Cross-charge corroboration features (whole-run reductions, indexed by row).
                m.set(ix.n_charge_states, r, f_n_charge[i]);
                m.set(ix.charge_multi_flag, r, f_charge_multi[i]);
                m.set(ix.cross_charge_intensity_log, r, f_cross_charge_int[i]);

                prelim[r] = n_matched[i] as f64 * (0.5 + ff.frag_corr.max(0.0))
                    + ff.coelution_mean.max(0.0)
                    + (1.0 + apex_int[i] as f64).ln() * 0.1
                    - rt_err / gradient;
            }
            drop(frag_feats);
            // `ext_vals` and the matrix are both live for the whole assembly above, so the
            // stage's real high-water mark is not one matrix. Measure the three value-shaped
            // buffers that overlap rather than asserting a multiple of one of them: this
            // chunk's matrix, this chunk's extended buffer, and the PREVIOUS chunk's columns,
            // which the writer thread still owns until it has encoded them.
            let ext_bytes = crate::memlog::bytes_of(&ext_vals);
            drop(ext_vals);
            let matrix_bytes = m.payload_bytes();
            max_matrix_bytes = max_matrix_bytes.max(matrix_bytes);
            max_inflight_bytes = max_inflight_bytes.max(matrix_bytes + ext_bytes + in_writer);

            // PIN first: it reads the same values the columns below move into Arrow.
            if let Some(w) = pin.as_mut() {
                w.write_chunk(
                    cols_active.len(),
                    &m,
                    &cid[lo..hi],
                    &label[lo..hi],
                    &pform[lo..hi],
                    &protein[lo..hi],
                    &mz[lo..hi],
                )?;
            }

            let mut cols: Vec<Col> = vec![
                Col::U32("candidate_id".into(), cid[lo..hi].to_vec()),
                Col::I32("peak_rank".into(), peak_rank[lo..hi].to_vec()),
                Col::Str("label".into(), label[lo..hi].to_vec()),
                Col::U32("base_peptide_id".into(), base[lo..hi].to_vec()),
                Col::Str("peptidoform".into(), pform[lo..hi].to_vec()),
                Col::Str("protein".into(), protein[lo..hi].to_vec()),
                Col::F64("apex_rt".into(), apex_rt[lo..hi].to_vec()),
                Col::F64("elution_lo".into(), elu_lo),
                Col::F64("elution_hi".into(), elu_hi),
                Col::F64("precursor_mz".into(), mz[lo..hi].to_vec()),
                Col::F64("prelim_score".into(), prelim),
            ];
            // Moved, not copied: the matrix is dropped on the next line, and copying
            // every column is what made the matrix exist twice at this point.
            for (c, name) in cols_active.iter().enumerate() {
                cols.push(Col::F64(name.clone(), m.take_column(c)));
            }
            drop(m);
            timers.add(&timers.assemble_ns, assembling);
            // The parquet encode is single-threaded and ran here, between one chunk's
            // computation and the next chunk's: the writer thread below takes it off the
            // critical path, the way the loader thread took the decode off it. The
            // channel is a rendezvous, so at most one chunk's columns wait behind the
            // one being written.
            let handing = Instant::now();
            let sent = wtx.send(Some(cols)).is_ok();
            timers.add(&timers.wait_writer_ns, handing);
            if !sent {
                // The writer failed; its error is the real one, so surface that.
                break;
            }
            // The send has returned, so the writer now owns this chunk's columns and holds
            // them while the next chunk is assembled. They are the matrix's columns, moved.
            in_writer = matrix_bytes;
        }
        // The PIN is flushed BEFORE the commit marker, because `writer.close()` publishes
        // `features.parquet` over the previous good one and the PIN write is the last thing
        // that can still fail. The in-line version ran `pin.finish()` and then
        // `writer.close()`, so a PIN that failed at flush (a full disk, a lost permission on
        // `out_pin`) left the previous features table untouched. Moving `close` onto the
        // writer thread silently reversed that: the table was published and the stage then
        // errored, leaving a published features table with no matching PIN. Failing here
        // returns without sending the marker, so the writer drops unpublished exactly as it
        // does for any other error in this closure.
        if let Some(w) = pin.take() {
            pin_finish.apply(w)?;
        }
        // Commit only after every chunk went out AND the PIN is on disk; a `break` above
        // leaves this send failing, and the writer's own error is what `join` then returns.
        let _ = wtx.send(None);
        drop(wtx);
        wh.join()
            .map_err(|_| anyhow!("features: the parquet writer thread panicked"))?
    })?;
    drop(extra_lease);
    timers.log(pass_start.elapsed(), n_loaders, chunks.len());

    crate::memlog::report(
        "features chromatogram store",
        &[
            ("largest_chunk_fragment_traces", max_frag_bytes),
            ("largest_chunk_ms1_xic_traces", max_ms1_bytes),
            ("run_total_streamed", tot_frag_bytes + tot_ms1_bytes),
        ],
    );
    info!(
        chunks = chunks.len(),
        candidates = n_cand,
        fragment_rows = n_frag_rows,
        ms1_rows = n_ms1_rows,
        chrom_rows = chrom_rows_total,
        "mem: features chromatogram store shape"
    );
    // `in_flight_peak` is MEASURED, not a multiple of `largest_chunk`. The in-line write
    // this replaced held one matrix plus the `column(c).to_vec()` copy of it, so 2x was the
    // right figure for it; the writer thread holds no copy but does hold the previous
    // chunk's columns while this chunk is assembled, and the assembly also has the extended
    // buffer live (337 of 387 columns wide at Extended, so ~0.87x a matrix). The overlap
    // therefore peaks near 2.9x a matrix, ABOVE the 2x it replaced -- the encode came off
    // the critical path at the cost of bytes, and the report says so rather than asserting
    // the old figure. `max_inflight_bytes` covers only these three buffers; the chromatogram
    // store is reported separately above and the Arrow encode's own buffers are the writer's.
    crate::memlog::report(
        "features value matrix",
        &[
            ("largest_chunk", max_matrix_bytes),
            ("in_flight_peak", max_inflight_bytes),
        ],
    );

    // Feature schema companion.
    let schema_id = feature_schema_id(&cols_active);
    mumdia_io::json::write_json(
        &format!("{}.schema.json", p.out),
        &FeatureSchema {
            feature_columns: cols_active.clone(),
            schema_id: schema_id.clone(),
        },
    )?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("feature_schema_id".to_string(), json!(schema_id));
    stats.insert("n_features".to_string(), json!(cols_active.len()));
    stats.insert("set".to_string(), json!(format!("{:?}", p.cfg.set)));
    ArtifactReport {
        logical_name: artifact::FEATURES.0.to_string(),
        schema_name: artifact::FEATURES.0.to_string(),
        schema_version: artifact::FEATURES.1,
        stage: "features".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({"set": format!("{:?}", p.cfg.set), "coelution_corr_threshold": p.cfg.coelution_corr_threshold}),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;

    info!(rows, features = cols_active.len(), set = ?p.cfg.set, elapsed_ms = elapsed, "features: done");
    Ok(rows)
}

#[derive(Clone, Default)]
struct FragFeatures {
    frag_corr: f64,
    frag_cosine: f64,
    spectral_angle: f64,
    coelution_mean: f64,
    coelution_best: f64,
    n_coelution_above: f64,
    norm_manhattan: f64,
    rmsd: f64,
    xcorr_coelution: f64,
    xcorr_shape: f64,
    sum_b: f64,
    sum_y: f64,
    n_b: f64,
    n_y: f64,
    weighted_mass_error: f64,
    mean_mass_error: f64,
    log_sn: f64,
    n_observations: f64,
    base_width_rt: f64,
    // DIA-NN-style profile features (computed over the elution window, not apex).
    profile_cos: f64,   // pCos: elution^2-weighted spectral cosine over the profile
    ref_corr: f64,      // pTimeCorr: mean fragment-vs-reference-profile correlation
    best_ref_corr: f64, // strongest fragment-vs-reference correlation
    low_frag_coel: f64, // pResCorr proxy: co-elution of the low-intensity fragments
    // DIA-NN interference-correction-style features (OpenSWATH/mProphet analogs).
    evidence: f64,     // summed fragment-vs-reference correlations (DIA-NN `Evidence`)
    contrast_min: f64, // min fragment-vs-(sum of others) correlation; low = interfered fragment
    resid_corr: f64, // mean pairwise corr of residuals f_k - proj_k*ref; high = shared interferent
    coel_clean: f64, // pairwise co-elution after interference capping at 1.5*r*ref
    shadow_frac: f64, // fraction of intensity above the 1.5*r*ref cap (interference shadow)
    // Elution-peak boundaries the engine computed and used to bound the features
    // above; emitted so downstream (and plotting) read them rather than re-derive.
    elution_lo: f64,
    elution_hi: f64,
}

/// Fragment-intensity agreement, co-elution, ion-series, and mass-accuracy
/// features for one PSM from its chromatogram rows.
/// 3-point smooth (matches the extract/DIA-NN kernel) for boundary finding.
fn smooth3(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    if n < 3 {
        return v.to_vec();
    }
    let mut o = v.to_vec();
    for i in 1..n - 1 {
        o[i] = 0.5 * v[i] + 0.25 * v[i - 1] + 0.25 * v[i + 1];
    }
    o[0] = 2.0 / 3.0 * v[0] + 1.0 / 3.0 * v[1];
    o[n - 1] = 2.0 / 3.0 * v[n - 1] + 1.0 / 3.0 * v[n - 2];
    o
}

/// Elution-peak boundary indices: descend from the apex until the profile drops
/// below `frac` * apex height (DIA-NN-style peak/3; benchmarked best vs DIA-NN RT).
///
/// `grace` bridges zig-zag: up to `grace` consecutive sub-threshold scans are
/// stepped over, so the boundary triggers only on the `grace + 1`th consecutive
/// sub-threshold scan. `grace = 0` reproduces the plain descend-to-first-miss
/// walk (the feature-set default); `grace = 1` stops on 2 consecutive misses.
pub(crate) fn peak_bounds(prof: &[f64], ai: usize, frac: f64, grace: usize) -> (usize, usize) {
    let n = prof.len();
    if n < 3 {
        return (0, n.saturating_sub(1));
    }
    // If the supplied apex sits at zero profile height, relocate it to the global
    // maximum. Using the max only for the threshold while walking from the zero
    // `ai` collapses both walks to a zero-width window around the wrong scan.
    let mut ai = ai;
    if prof[ai] <= 0.0 {
        ai = prof
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
    }
    let peak = prof[ai];
    if peak <= 0.0 {
        return (0, n - 1);
    }
    let thr = frac * peak;
    // Walk left: `lo` tracks the furthest above-threshold scan; sub-threshold scans
    // are counted and bridged while the run of consecutive misses stays <= grace.
    let mut lo = ai;
    let mut miss = 0usize;
    let mut i = ai;
    while i > 0 {
        i -= 1;
        if prof[i] >= thr {
            lo = i;
            miss = 0;
        } else {
            miss += 1;
            if miss > grace {
                break;
            }
        }
    }
    let mut hi = ai;
    miss = 0;
    i = ai;
    while i + 1 < n {
        i += 1;
        if prof[i] >= thr {
            hi = i;
            miss = 0;
        } else {
            miss += 1;
            if miss > grace {
                break;
            }
        }
    }
    (lo, hi)
}

/// Map a global (left, right) elution half-width (seconds) around `apex_rt` onto
/// index bounds of `axis_full`, falling back to the apex-nearest scan `ai` if the
/// window collapses between scans (sparse grid, or half-width below one cycle).
fn global_bound_indices(
    axis_full: &[f32],
    apex_rt: f64,
    ai: usize,
    l: f64,
    r: f64,
) -> (usize, usize) {
    let lo_rt = (apex_rt - l) as f32;
    let hi_rt = (apex_rt + r) as f32;
    let li = axis_full.iter().position(|&t| t >= lo_rt).unwrap_or(0);
    let hi = axis_full
        .iter()
        .rposition(|&t| t <= hi_rt)
        .unwrap_or(axis_full.len().saturating_sub(1));
    if li <= hi {
        (li, hi)
    } else {
        (ai, ai)
    }
}

/// Per-candidate elution-peak RT bounds (seconds): reference = smoothed sum of the
/// top-3 predicted-intensity fragments, walked from the apex-nearest scan while
/// >= `frac` x apex height, bridging <= `grace` sub-threshold scans. Returns
/// > (lo_rt, hi_rt), or None when fewer than 3 distinct scans. Mirrors the boundary
/// > logic inside `fragment_features`/`build_evidence` so the confident-set half-widths
/// > match the per-candidate detector they replace when `bound_from_confident` is set.
fn elution_peak_rt_bounds(
    rows: &[ChromRow],
    apex_rt: f64,
    frac: f64,
    grace: usize,
) -> Option<(f32, f32)> {
    let TraceAlign {
        axis_full: axis,
        traces_full: traces,
    } = align_traces(rows);
    if axis.len() < 3 {
        return None;
    }
    let mut ord: Vec<usize> = (0..rows.len()).collect();
    ord.sort_by(|&a, &b| rows[b].pred_int.total_cmp(&rows[a].pred_int));
    let k3: Vec<usize> = ord.into_iter().take(3).collect();
    let prof_raw: Vec<f64> = (0..axis.len())
        .map(|k| k3.iter().map(|&i| traces[i][k]).sum::<f64>())
        .collect();
    let prof = smooth3(&prof_raw);
    let ai = axis
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (**a as f64 - apex_rt)
                .abs()
                .total_cmp(&(**b as f64 - apex_rt).abs())
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let (lo, hi) = peak_bounds(&prof, ai, frac, grace);
    Some((axis[lo], axis[hi]))
}

// The peak-bounding knobs (`bound`, `frac`, `grace`, `global_bounds`) are passed through
// from the config as they always were; the alignment, the apex intensities and the shared
// peak window are the others.
//
// `shared` is the PSM's [`PeakTraces`] when the caller built one AND `bound` is set, which
// is exactly when this function's own peak window is that one: both walk the same profile
// with the same knobs (`peak_window`). Without `bound` it scores the whole extracted window
// and must not share.
#[allow(clippy::too_many_arguments)]
fn fragment_features(
    rows: &[ChromRow],
    al: &TraceAlign,
    obs: &[f64],
    shared: Option<&PeakTraces>,
    apex_rt: f64,
    coel_thresh: f64,
    bound: bool,
    frac: f64,
    grace: usize,
    global_bounds: Option<(f64, f64)>,
) -> FragFeatures {
    debug_assert!(
        bound || shared.is_none(),
        "the whole-window path must not share"
    );
    let mut f = FragFeatures::default();
    // Observed apex intensity per fragment (nearest scan to apex): `obs`, from
    // `apex_intensities`, the f64 widening of the value this loop used to find itself.
    let mut pred = Vec::with_capacity(rows.len());
    let mut mass_err = Vec::new();
    let mut mass_w = Vec::new();
    for (r, &best) in rows.iter().zip(obs) {
        pred.push(r.pred_int as f64);
        let pe = ppm_diff(r.frag_obs_mz, r.frag_mz).abs();
        mass_err.push(pe);
        mass_w.push(best);
        let is_b = r.frag_name.starts_with('b');
        if is_b {
            f.sum_b += best;
            f.n_b += 1.0;
        } else {
            f.sum_y += best;
            f.n_y += 1.0;
        }
    }
    f.frag_corr = pearson(obs, &pred);
    f.frag_cosine = cosine(obs, &pred);
    f.spectral_angle = spectral_angle(obs, &pred);

    // normalized manhattan + rmsd on sum-normalized vectors
    let (on, pn) = (normalize_sum(obs), normalize_sum(&pred));
    f.norm_manhattan = on.iter().zip(&pn).map(|(a, b)| (a - b).abs()).sum();
    let m = on.len().max(1) as f64;
    f.rmsd = (on
        .iter()
        .zip(&pn)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / m)
        .sqrt();

    // mass accuracy (intensity-weighted and unweighted mean |ppm|)
    let wsum: f64 = mass_w.iter().sum();
    f.weighted_mass_error = if wsum > 0.0 {
        mass_err
            .iter()
            .zip(&mass_w)
            .map(|(e, w)| e * w)
            .sum::<f64>()
            / wsum
    } else {
        0.0
    };
    f.mean_mass_error = if !mass_err.is_empty() {
        mass_err.iter().sum::<f64>() / mass_err.len() as f64
    } else {
        0.0
    };

    // Traces aligned on the union RT axis (built once per PSM by the caller), then
    // restricted to the elution PEAK so the trace-based features below are computed over
    // the peak, not the whole extracted RT window (which spans +/- w_rt and would dilute
    // co-elution/profile scores).
    let axis_full = &al.axis_full;
    let traces_full = &al.traces_full;
    // boundary on the smoothed summed top-3-predicted-fragment profile, around apex
    let (lo_i, hi_i) = match shared {
        Some(pk) => (pk.lo, pk.hi),
        None if bound => peak_window(
            axis_full,
            traces_full,
            &pred,
            apex_rt,
            frac,
            grace,
            global_bounds,
        ),
        None => (0, axis_full.len().saturating_sub(1)),
    };
    let axis: &[f32] = &axis_full[lo_i..=hi_i];
    let traces_owned: Vec<Vec<f64>>;
    let traces: &[Vec<f64>] = match shared {
        Some(pk) => &pk.traces,
        None => {
            traces_owned = traces_full
                .iter()
                .map(|t| t[lo_i..=hi_i].to_vec())
                .collect();
            &traces_owned
        }
    };
    f.n_observations = axis.len() as f64;
    f.elution_lo = axis.first().map(|&x| x as f64).unwrap_or(0.0);
    f.elution_hi = axis.last().map(|&x| x as f64).unwrap_or(0.0);
    f.base_width_rt = if axis.len() >= 2 {
        (axis[axis.len() - 1] - axis[0]) as f64
    } else {
        0.0
    };
    let mut corrs = Vec::new();
    let mut lags = Vec::new();
    let mut shapes = Vec::new();
    if axis.len() >= 2 {
        if let Some(pk) = shared {
            // The PSM's pair statistics, computed by the same calls on the same traces.
            let st = &pk.stats;
            for a in 0..traces.len() {
                for b in (a + 1)..traces.len() {
                    corrs.push(st.corr(a, b));
                    let (lag, shape) = st.xcorr(a, b);
                    lags.push(lag.abs() as f64);
                    shapes.push(shape);
                }
            }
        } else {
            // Every pair's correlation from traces centred once (`pearson_pairs`,
            // bit-identical to `pearson` pair by pair), in the (a, b) order below.
            pearson_pairs(traces, &mut corrs);
            // Each trace's norm once, not once per pair it is in.
            let norms: Vec<f64> = traces.iter().map(|t| xcorr_norm(t)).collect();
            for a in 0..traces.len() {
                for b in (a + 1)..traces.len() {
                    let (lag, shape) =
                        best_xcorr_normed(&traces[a], &traces[b], 5, norms[a], norms[b]);
                    lags.push(lag.abs() as f64);
                    shapes.push(shape);
                }
            }
        }
    }
    f.coelution_mean = mean(&corrs);
    f.coelution_best = corrs.iter().cloned().fold(f64::MIN, f64::max).max(0.0);
    if corrs.is_empty() {
        f.coelution_best = 0.0;
    }
    f.n_coelution_above = corrs.iter().filter(|c| **c >= coel_thresh).count() as f64;
    f.xcorr_coelution = mean(&lags); // ideal 0
    f.xcorr_shape = mean(&shapes); // ideal 1

    // --- DIA-NN-style profile features (pCos / pTimeCorr / pResCorr analogs) ---
    // Reference elution profile = predicted-intensity-weighted sum of fragment XICs.
    if !traces.is_empty() && axis.len() >= 2 {
        let np = axis.len();
        // The shared window's reference profile is this one, term for term.
        let refp_owned: Vec<f64>;
        let refp: &[f64] = match shared {
            Some(pk) => &pk.ref_profile,
            None => {
                let mut r = vec![0.0f64; np];
                for (fi, tr) in traces.iter().enumerate() {
                    let w = pred[fi].max(0.0);
                    for k in 0..np {
                        r[k] += w * tr[k];
                    }
                }
                refp_owned = r;
                &refp_owned
            }
        };
        // pCos: at each scan, cosine(observed fragment vector, predicted vector),
        // weighted by reference-profile^2 (concentrates on the elution peak).
        let (mut num, mut den) = (0.0, 0.0);
        // One scratch vector for the whole scan loop; it used to be a fresh allocation
        // per scan, tens of millions of them over a run.
        let mut obs_k: Vec<f64> = Vec::with_capacity(traces.len());
        for k in 0..np {
            let w = refp[k] * refp[k];
            if w <= 0.0 {
                continue;
            }
            obs_k.clear();
            obs_k.extend(traces.iter().map(|tr| tr[k]));
            num += cosine(&obs_k, &pred) * w;
            den += w;
        }
        f.profile_cos = if den > 0.0 { num / den } else { 0.0 };
        // pTimeCorr: each fragment XIC correlated with the reference profile, which is
        // centred once for all of them (or read from the shared statistics, which made
        // the same calls).
        let rc: Vec<f64> = match shared {
            Some(pk) => pk.stats.ref_corr().to_vec(),
            None => {
                let refp_c = Centered::new(refp);
                traces
                    .iter()
                    .map(|tr| pearson_vs(tr, refp, &refp_c))
                    .collect()
            }
        };
        f.ref_corr = mean(&rc);
        f.best_ref_corr = rc.iter().cloned().fold(f64::MIN, f64::max).max(0.0);
        // pResCorr proxy: co-elution of the low-predicted-intensity fragments.
        // Real peptides show their minor fragments co-eluting; chimeras do not.
        let mut order: Vec<usize> = (0..pred.len()).collect();
        // `unwrap_or(Equal)`, matching the two other sorts of this same array in this file.
        // The bare `unwrap()` here panicked on a non-finite predicted intensity, inside a
        // rayon closure and after extract, so one NULL cell in a library discarded the most
        // expensive stage in the pipeline with a message that named neither the column nor
        // the row. Library load now rejects non-finite intensities, which is the real fix;
        // this keeps the three sorts consistent so the next reader does not have to work
        // out why one of them differed.
        order.sort_by(|&a, &b| pred[a].total_cmp(&pred[b]));
        let take = (order.len() / 2).max(1);
        let low: Vec<f64> = order.iter().take(take).map(|&i| rc[i]).collect();
        f.low_frag_coel = mean(&low);

        // --- DIA-NN interference-correction-style features ---
        // Evidence: summed fragment-vs-reference correlations (aggregate confidence).
        f.evidence = rc.iter().sum();
        // Contrast: each fragment vs the summed profile of the other fragments.
        let total: Vec<f64> = (0..np)
            .map(|k| traces.iter().map(|tr| tr[k]).sum::<f64>())
            .collect();
        let mut contrasts = Vec::with_capacity(traces.len());
        // Scratch reused across fragments; the per-fragment allocation it replaces was
        // one heap block per fragment per PSM.
        let mut others: Vec<f64> = Vec::with_capacity(np);
        for tr in traces {
            others.clear();
            others.extend((0..np).map(|k| total[k] - tr[k]));
            contrasts.push(pearson(tr, &others));
        }
        f.contrast_min = contrasts.iter().cloned().fold(f64::MAX, f64::min);
        // Empty contrasts fold to f64::MAX; treat "no contrast computed" as 0.0.
        if f.contrast_min == f64::MAX {
            f.contrast_min = 0.0;
        }
        // Interference capping (shadow removal): r_k = <f_k,ref>/<ref,ref>, cap at 1.5*r*ref.
        let rr: f64 = refp.iter().map(|x| x * x).sum::<f64>().max(1e-9);
        let mut cleaned: Vec<Vec<f64>> = Vec::with_capacity(traces.len());
        let mut residuals: Vec<Vec<f64>> = Vec::with_capacity(traces.len());
        let mut shadow_num = 0.0;
        let mut total_int = 0.0;
        for tr in traces {
            let rk: f64 = tr.iter().zip(refp).map(|(a, b)| a * b).sum::<f64>() / rr;
            let cl: Vec<f64> = (0..np)
                .map(|k| {
                    let cap = (1.5 * rk * refp[k]).max(0.0);
                    tr[k].min(cap)
                })
                .collect();
            let res: Vec<f64> = (0..np).map(|k| tr[k] - rk * refp[k]).collect();
            for k in 0..np {
                shadow_num += (tr[k] - (1.5 * rk * refp[k]).max(0.0)).max(0.0);
                total_int += tr[k];
            }
            cleaned.push(cl);
            residuals.push(res);
        }
        f.shadow_frac = if total_int > 0.0 {
            shadow_num / total_int
        } else {
            0.0
        };
        // co-elution of cleaned traces and correlation of residuals (shared interferent),
        // each a pair matrix over rows centred once, in the old (a, b) order.
        let mut clean_corrs = Vec::new();
        let mut res_corrs = Vec::new();
        pearson_pairs(&cleaned, &mut clean_corrs);
        pearson_pairs(&residuals, &mut res_corrs);
        f.coel_clean = mean(&clean_corrs);
        f.resid_corr = mean(&res_corrs);
    }

    // chromatographic log S/N: apex vs median trace point.
    let apex_val = obs.iter().cloned().fold(0.0, f64::max);
    let mut all_points: Vec<f64> = traces
        .iter()
        .flatten()
        .cloned()
        .filter(|v| *v > 0.0)
        .collect();
    let noise = if all_points.is_empty() {
        1.0
    } else {
        // One order statistic, by selection: bit-identical to indexing the sorted copy.
        let mid = all_points.len() / 2;
        order_stat(&mut all_points, mid).max(1.0)
    };
    f.log_sn = ((apex_val + 1.0) / (noise + 1.0)).ln();
    f
}

fn normalize_sum(v: &[f64]) -> Vec<f64> {
    let s: f64 = v.iter().sum();
    if s > 0.0 {
        v.iter().map(|x| x / s).collect()
    } else {
        v.to_vec()
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// The `k`-th smallest value of `v` under `f64::total_cmp`, by selection (O(n)) rather than
/// by sorting a copy (O(n log n)). `v` is permuted.
///
/// Bit-identical to `sorted[k]` of the sort it replaces: `total_cmp` is a total order in
/// which two values compare equal only when their bit patterns are equal, so the sorted
/// sequence of a multiset is unique bit for bit, whatever algorithm produced it (stable or
/// not), and every order statistic is determined by the multiset alone.
fn order_stat(v: &mut [f64], k: usize) -> f64 {
    *v.select_nth_unstable_by(k, f64::total_cmp).1
}

/// The median of a NON-EMPTY `v` exactly as the families computed it from a sorted copy:
/// the middle value, or for an even length `0.5 * (s[n / 2 - 1] + s[n / 2])`. After
/// selecting position `n / 2`, the lower half holds the `n / 2` smallest values, so its
/// maximum under `total_cmp` is `s[n / 2 - 1]`. `v` is permuted.
fn median_select(v: &mut [f64]) -> f64 {
    let n = v.len();
    let hi = order_stat(v, n / 2);
    if n % 2 == 1 {
        hi
    } else {
        let lo = v[..n / 2]
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .expect("an even-length median has a lower half");
        0.5 * (lo + hi)
    }
}

/// The linearly interpolated `q`-quantile exactly as a sorted copy gave it (position
/// `q * (n - 1)`, `s[lo] * (1 - frac) + s[hi] * frac` between the two neighbouring order
/// statistics; 0 for an empty `v`, the value itself for one). After selecting `lo`, every
/// value to its right is at least `s[lo]`, so their minimum is `s[lo + 1]`. `v` is
/// permuted, and two calls on the same buffer are as good as two sorted copies, because
/// selection depends on the multiset only.
fn quantile_select(v: &mut [f64], q: f64) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return v[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let s_lo = order_stat(v, lo);
    if lo == hi {
        s_lo
    } else {
        let s_hi = v[lo + 1..]
            .iter()
            .copied()
            .min_by(f64::total_cmp)
            .expect("a value above the lower neighbour");
        let frac = pos - lo as f64;
        s_lo * (1.0 - frac) + s_hi * frac
    }
}

/// Best cross-correlation of two traces over integer lags in [-maxlag, maxlag].
/// Returns (lag_of_max, normalized_max_value).
///
/// The one-call form, for a family that correlates a pair once. The shipped callers
/// correlate every pair of a PSM and pass cached norms to [`best_xcorr_normed`] instead
/// (through [`PairStats`] or directly), so outside the tests nothing calls this today.
#[cfg_attr(not(test), allow(dead_code))]
fn best_xcorr(a: &[f64], b: &[f64], maxlag: i32) -> (i32, f64) {
    if a.len() < 2 {
        return (0, 0.0);
    }
    best_xcorr_normed(a, b, maxlag, xcorr_norm(a), xcorr_norm(b))
}

/// The Euclidean norm of one trace, exactly as [`best_xcorr`] computes it for each of its
/// two arguments. A pair matrix calls `best_xcorr` K - 1 times per trace, so the norms are
/// computed once per trace with this and handed to [`best_xcorr_normed`].
fn xcorr_norm(a: &[f64]) -> f64 {
    (a.iter().map(|x| x * x).sum::<f64>()).sqrt()
}

/// The lag at which every call site cross-correlates, and the lag-window width.
const XCORR_MAXLAG: usize = 5;
const XCORR_LAGS: usize = 2 * XCORR_MAXLAG + 1;

/// [`best_xcorr`] with both norms supplied; `na` and `nb` must be [`xcorr_norm`] of `a`
/// and `b`, which makes the two functions agree bit for bit.
///
/// The correlation used to be one serial pass per lag: each lag's dot product a single
/// dependent chain of adds with a bounds branch inside, 11 chains one after another.
/// [`lag_dots`] runs the 11 accumulators side by side instead. Every accumulator still
/// receives exactly its old terms in ascending `i`, starting from `0.0`, each a separate
/// multiply and add (Rust never contracts them into a fused multiply-add), so every dot
/// product, every `dot / (na * nb)` and the strict `>` first-maximum selection are
/// unchanged. Any other lag window, or a `b` shorter than `a` (which indexed out of
/// bounds before and still does), takes the per-lag loop as it was.
fn best_xcorr_normed(a: &[f64], b: &[f64], maxlag: i32, na: f64, nb: f64) -> (i32, f64) {
    let n = a.len();
    if n < 2 {
        return (0, 0.0);
    }
    if na <= 0.0 || nb <= 0.0 {
        return (0, 0.0);
    }
    let (mut best_lag, mut best_val) = (0i32, f64::MIN);
    if maxlag == XCORR_MAXLAG as i32 && b.len() >= n {
        // `b[..n]`: the old loop never read past `n`, whatever `b.len()` was.
        let dots = lag_dots(a, &b[..n]);
        for (li, &dot) in dots.iter().enumerate() {
            let v = dot / (na * nb);
            if v > best_val {
                best_val = v;
                best_lag = li as i32 - maxlag;
            }
        }
        return (best_lag, best_val.max(0.0));
    }
    for lag in -maxlag..=maxlag {
        let mut dot = 0.0;
        #[allow(clippy::needless_range_loop)] // i also drives j = i + lag
        for i in 0..n {
            let j = i as i32 + lag;
            if j >= 0 && (j as usize) < n {
                dot += a[i] * b[j as usize];
            }
        }
        let v = dot / (na * nb);
        if v > best_val {
            best_val = v;
            best_lag = lag;
        }
    }
    (best_lag, best_val.max(0.0))
}

/// The dot products `sum_i a[i] * b[i + lag]` for every lag in
/// `-XCORR_MAXLAG..=XCORR_MAXLAG`, in that order, over the `i` where `i + lag` is inside
/// `0..n`. `b` must be exactly as long as `a`.
///
/// `i` is the outer loop, so each accumulator gets its terms in ascending `i` exactly as
/// the one-lag-at-a-time loop gave them. Away from the two ends every lag is valid and
/// the body is branch-free over an 11-wide window of `b`; within `XCORR_MAXLAG` of either
/// end each lag keeps its bounds test.
fn lag_dots(a: &[f64], b: &[f64]) -> [f64; XCORR_LAGS] {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    let l = XCORR_MAXLAG;
    let mut acc = [0.0f64; XCORR_LAGS];
    for (i, &ai) in a.iter().enumerate() {
        if i >= l && i + l < n {
            let w: &[f64; XCORR_LAGS] =
                b[i - l..=i + l].try_into().expect("an 11-wide window of b");
            for (dst, &bj) in acc.iter_mut().zip(w) {
                *dst += ai * bj;
            }
        } else {
            for (li, dst) in acc.iter_mut().enumerate() {
                // lag = li - l, so j = i + li - l.
                if i + li >= l && i + li - l < n {
                    *dst += ai * b[i + li - l];
                }
            }
        }
    }
    acc
}

/// Averagine isotope-envelope agreement from MS1 apex intensities.
/// Returns (isotope_corr, isom1_ratio, log_mono, has_ms1).
fn isotope_features(
    m1: Option<f64>,
    mono: Option<f64>,
    i1: Option<f64>,
    i2: Option<f64>,
    neutral_mass: f64,
) -> (f64, f64, f64, f64) {
    match (mono, i1, i2) {
        (Some(mono), Some(i1), Some(i2)) => {
            // Poisson averagine: lambda ~ 0.00052 * mass (expected extra neutrons).
            let lambda = 0.000_52 * neutral_mass;
            let theo = [1.0, lambda, lambda * lambda / 2.0];
            let obs = [mono, i1, i2];
            let corr = pearson(&obs, &theo);
            let ratio = m1.unwrap_or(0.0) / (mono + 1.0);
            (corr, ratio, (1.0 + mono).ln(), 1.0)
        }
        _ => (0.0, 0.0, 0.0, 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_below_the_anchor_floor_clear_it_once_pooled() {
        // The banded defect, as arithmetic. Each band's confident set is a slice of the
        // run's, so fitting per band puts every band under the 20-anchor floor and each
        // one silently falls back to per-candidate boundary detection, while the same run
        // searched unbanded fits a global window. Measured on a seven-file
        // immunopeptidomics search: 735 pooled anchors, 0 or 1 per band.
        let cfg = FeaturesConfig {
            bound_from_confident: true,
            bound_confident_pct: 50.0,
            ..Default::default()
        };
        let band = |n: usize, w: f64| BoundSamples {
            lefts: vec![w; n],
            rights: vec![w * 2.0; n],
        };

        // Sixty-three bands of one anchor each: every one of them declines on its own.
        let bands: Vec<BoundSamples> = (0..63).map(|_| band(1, 4.0)).collect();
        for b in &bands {
            assert_eq!(
                bounds_from_samples(b, &cfg),
                None,
                "a single-anchor band must not fit a window of its own"
            );
        }

        // Pooled, the same anchors clear the floor and every band gets one window.
        let mut pooled = BoundSamples::default();
        for b in bands {
            pooled.absorb(b);
        }
        assert_eq!(pooled.len(), 63);
        assert_eq!(pooled.rights.len(), 63);
        assert_eq!(bounds_from_samples(&pooled, &cfg), Some((4.0, 8.0)));

        // The floor still bites when the RUN really is short of anchors, which is the
        // case the fallback exists for.
        let mut thin = BoundSamples::default();
        for _ in 0..19 {
            thin.absorb(band(1, 4.0));
        }
        assert_eq!(bounds_from_samples(&thin, &cfg), None);
        assert!(!thin.is_empty());
    }

    #[test]
    fn plan_chunks_cuts_only_at_candidate_boundaries() {
        // Three PSM rows for candidate 1 (top-K), one each for 2 and 3; candidate 2 has
        // no chromatogram rows at all.
        let psm = [1, 1, 1, 2, 3];
        let chrom = [1, 1, 1, 1, 3, 3];
        // One row per chunk requested: the planner must still keep a candidate whole.
        let cs = plan_chunks(&psm, &chrom, "c.parquet", 1, usize::MAX).unwrap();
        assert_eq!(
            cs.len(),
            2,
            "one chunk per candidate that has chromatogram rows"
        );
        assert_eq!((cs[0].psm_lo, cs[0].psm_hi, cs[0].chrom_rows), (0, 3, 4));
        assert_eq!((cs[1].psm_lo, cs[1].psm_hi, cs[1].chrom_rows), (3, 5, 2));
        // Every PSM row lands in exactly one chunk, in order.
        assert_eq!(cs[0].psm_lo, 0);
        assert_eq!(cs.last().unwrap().psm_hi, psm.len());
        // A chunk larger than the table is one chunk over everything.
        let one = plan_chunks(&psm, &chrom, "c.parquet", 1 << 20, usize::MAX).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!((one[0].psm_lo, one[0].psm_hi, one[0].chrom_rows), (0, 5, 6));
    }

    #[test]
    fn plan_chunks_bounds_psm_rows_as_well_as_chromatogram_rows() {
        // Four top-K PSM rows per candidate, one chromatogram row each: the ratio the
        // chromatogram limit cannot see. With only `chunk_rows` the whole run is one
        // chunk, and the value buffers (sized in PSM rows) grow with `retain_top_peaks`
        // without any constant bounding them.
        let psm: Vec<u32> = (0..6u32).flat_map(|c| [c; 4]).collect();
        let chrom: Vec<u32> = (0..6u32).collect();
        let unbounded = plan_chunks(&psm, &chrom, "c.parquet", 1 << 20, usize::MAX).unwrap();
        assert_eq!(unbounded.len(), 1);
        assert_eq!(unbounded[0].psm_hi - unbounded[0].psm_lo, 24);

        // The PSM limit closes the chunk instead, at the end of the candidate that
        // reached it -- never inside one, so the chunk can exceed the limit by at most
        // the last candidate's PSM rows.
        let cs = plan_chunks(&psm, &chrom, "c.parquet", 1 << 20, 6).unwrap();
        assert_eq!(cs.len(), 3);
        for c in &cs {
            assert_eq!(c.psm_hi - c.psm_lo, 8, "two whole candidates per chunk");
            assert_eq!(c.chrom_rows, 2);
        }
        // Every PSM row lands in exactly one chunk, in order, whichever limit binds.
        assert_eq!(cs[0].psm_lo, 0);
        assert!(cs.windows(2).all(|w| w[0].psm_hi == w[1].psm_lo));
        assert_eq!(cs.last().unwrap().psm_hi, psm.len());
        // A limit below one candidate's PSM rows gives one candidate per chunk, not a
        // split candidate.
        let cs = plan_chunks(&psm, &chrom, "c.parquet", 1 << 20, 1).unwrap();
        assert_eq!(cs.len(), 6);
        assert!(cs.iter().all(|c| c.psm_hi - c.psm_lo == 4));
    }

    #[test]
    fn plan_chunks_rejects_artifacts_it_cannot_join() {
        // A chromatogram candidate the PSM table does not have.
        let e = plan_chunks(&[1, 2], &[1, 9], "c.parquet", 1 << 20, usize::MAX).unwrap_err();
        assert!(format!("{e}").contains("candidate 9"), "{e}");
        // A candidate whose rows are not contiguous.
        let e = plan_chunks(&[1, 2, 1], &[1], "c.parquet", 1 << 20, usize::MAX).unwrap_err();
        assert!(format!("{e}").contains("more than one run"), "{e}");
    }

    #[test]
    fn peptide_length_ignores_mods() {
        assert_eq!(peptide_length("PEPC[Carbamidomethyl]M[Oxidation]IDE"), 8);
        assert_eq!(peptide_length("PEPTIDE"), 7);
        // DECOY_ marker letters must not count as residues (label-leak guard).
        assert_eq!(peptide_length("DECOY_PEPTIDE"), 7);
        assert_eq!(peptide_length("DECOY_PEPC[Carbamidomethyl]IDE"), 7);
    }

    #[test]
    fn feature_sets_sized() {
        assert_eq!(active_features(FeatureSet::Minimal).len(), 14);
        assert_eq!(active_features(FeatureSet::Rich).len(), 14 + 30);
        // Extended = minimal + rich + the family battery, and its names are unique.
        let ext = active_features(FeatureSet::Extended);
        // +6 psms-derived extras: 3 co-elution peak-contest metrics
        // (peak_contested_frac + peak_contested_count_frac + peak_apportioned_frac)
        // + 3 charge-corroboration features.
        assert_eq!(ext.len(), 14 + 30 + extended_names().len() + 6);
        let uniq: std::collections::HashSet<&String> = ext.iter().collect();
        assert_eq!(
            uniq.len(),
            ext.len(),
            "duplicate feature name in Extended set"
        );
    }

    #[test]
    fn unavailable_rt_calibration_contributes_zero_error() {
        assert_eq!(calibrated_rt_error(600.0, f64::NAN), 0.0);
        assert_eq!(calibrated_rt_error(600.0, f64::INFINITY), 0.0);
        assert_eq!(calibrated_rt_error(600.0, 580.0), 20.0);
    }

    #[test]
    fn xcorr_aligned_traces() {
        let a = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let (lag, shape) = best_xcorr(&a, &a, 3);
        assert_eq!(lag, 0);
        assert!(shape > 0.99);
    }

    // --- trace alignment (the fast path must reproduce the union-and-map build) ---

    fn row<'a>(name: &'a str, pred: f32, rt: &'a [f32], inten: &'a [f32]) -> ChromRow<'a> {
        ChromRow {
            frag_name: name,
            frag_mz: 100.0,
            frag_obs_mz: 100.0,
            pred_int: pred,
            rt,
            inten,
        }
    }

    fn same(a: &TraceAlign, b: &TraceAlign) {
        assert_eq!(
            a.axis_full.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.axis_full.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "axis differs"
        );
        assert_eq!(a.traces_full.len(), b.traces_full.len(), "fragment count");
        for (i, (x, y)) in a.traces_full.iter().zip(&b.traces_full).enumerate() {
            assert_eq!(
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "fragment {i} trace differs"
            );
        }
    }

    /// The MS1 XIC resampling exactly as it stood before the identity fast path: one map
    /// per isotope keyed on the RT bit pattern, a full-window vector, then the peak slice.
    /// A transcription of the replaced code, kept as the reference.
    fn old_ms1_xic(
        axis_full: &[f32],
        ms1_rows: &[ChromRow],
        lo: usize,
        hi: usize,
    ) -> Vec<Vec<f64>> {
        let mut out = Vec::new();
        for name in ["ms1_mono", "ms1_iso1", "ms1_iso2"] {
            if let Some(r) = ms1_rows.iter().find(|r| r.frag_name == name) {
                let map: HashMap<u32, f32> =
                    r.rt.iter()
                        .zip(r.inten.iter())
                        .map(|(&t, &v)| (t.to_bits(), v))
                        .collect();
                let full: Vec<f64> = axis_full
                    .iter()
                    .map(|t| *map.get(&t.to_bits()).unwrap_or(&0.0) as f64)
                    .collect();
                out.push(full[lo..=hi].to_vec());
            }
        }
        if out.len() == 3 {
            out
        } else {
            Vec::new()
        }
    }

    #[test]
    fn ms1_rows_on_the_fragment_axis_skip_the_map_bit_for_bit() {
        // The fast path applies when the MS1 row samples the fragments' own axis, which is
        // what extract writes; every other shape must still reach the map. Both must equal
        // the map build it replaced, bit for bit, including signed zeros and subnormals.
        let grid: Vec<f32> = (0..12).map(|k| 100.0 + k as f32 * 1.5).collect();
        let shifted: Vec<f32> = grid.iter().map(|t| t + 0.25).collect();
        let every_other: Vec<f32> = grid.iter().step_by(2).copied().collect();
        let tr_a: Vec<f32> = (0..12)
            .map(|k| ((k as f32 - 6.0).abs() * -3.0 + 20.0).max(0.0))
            .collect();
        let tr_b: Vec<f32> = (0..12).map(|k| (k * 7 % 5) as f32).collect();
        let iso = |k: usize| -> Vec<f32> {
            let mut v: Vec<f32> = (0..12).map(|i| (i * (k + 3) % 11) as f32 * 10.0).collect();
            v[1] = -0.0;
            v[2] = f32::MIN_POSITIVE / 4.0; // subnormal
            v
        };
        let (i0, i1, i2) = (iso(0), iso(1), iso(2));
        let long: Vec<f32> = i1.iter().chain([5.0f32, 6.0].iter()).copied().collect();
        let short_vals: Vec<f32> = i2.iter().step_by(2).copied().collect();
        let frags = vec![row("y1", 1.0, &grid, &tr_a), row("b2", 0.5, &grid, &tr_b)];
        let cases: Vec<(&str, Vec<ChromRow>)> = vec![
            (
                "on axis",
                vec![
                    row("ms1_mono", 0.0, &grid, &i0),
                    row("ms1_iso1", 0.0, &grid, &i1),
                    row("ms1_iso2", 0.0, &grid, &i2),
                ],
            ),
            (
                "shifted grid",
                vec![
                    row("ms1_mono", 0.0, &shifted, &i0),
                    row("ms1_iso1", 0.0, &grid, &i1),
                    row("ms1_iso2", 0.0, &grid, &i2),
                ],
            ),
            (
                "sub-grid",
                vec![
                    row("ms1_mono", 0.0, &grid, &i0),
                    row("ms1_iso1", 0.0, &grid, &i1),
                    row("ms1_iso2", 0.0, &every_other, &short_vals),
                ],
            ),
            (
                "longer intensity",
                vec![
                    row("ms1_mono", 0.0, &grid, &i0),
                    row("ms1_iso1", 0.0, &grid, &long),
                    row("ms1_iso2", 0.0, &grid, &i2),
                ],
            ),
            (
                "empty trace",
                vec![
                    row("ms1_mono", 0.0, &grid, &i0),
                    row("ms1_iso1", 0.0, &[], &[]),
                    row("ms1_iso2", 0.0, &grid, &i2),
                ],
            ),
            (
                "two isotopes",
                vec![
                    row("ms1_mono", 0.0, &grid, &i0),
                    row("ms1_iso1", 0.0, &grid, &i1),
                ],
            ),
        ];
        for (tag, ms1) in &cases {
            for (apex, bounds) in [(109.0, None), (104.5, Some((3.0, 4.0))), (90.0, None)] {
                let al = align_traces(&frags);
                let axis_full = al.axis_full.clone();
                let ev = build_evidence(&frags, al, ms1, apex, 1.0 / 3.0, 0, bounds);
                // The peak window as positions on the full axis.
                let lo = axis_full
                    .iter()
                    .position(|&t| t as f64 == ev.axis[0])
                    .unwrap();
                let hi = lo + ev.axis.len() - 1;
                let want = old_ms1_xic(&axis_full, ms1, lo, hi);
                let bits = |v: &Vec<Vec<f64>>| -> Vec<Vec<u64>> {
                    v.iter()
                        .map(|t| t.iter().map(|x| x.to_bits()).collect())
                        .collect()
                };
                assert_eq!(bits(&ev.ms1_xic), bits(&want), "{tag}, apex {apex}");
            }
        }
    }

    /// `best_xcorr` exactly as it stood before the lag-parallel kernel: one serial pass
    /// per lag, norms recomputed per call. A transcription kept as the reference.
    fn old_best_xcorr(a: &[f64], b: &[f64], maxlag: i32) -> (i32, f64) {
        let n = a.len();
        if n < 2 {
            return (0, 0.0);
        }
        let na = (a.iter().map(|x| x * x).sum::<f64>()).sqrt();
        let nb = (b.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if na <= 0.0 || nb <= 0.0 {
            return (0, 0.0);
        }
        let (mut best_lag, mut best_val) = (0i32, f64::MIN);
        for lag in -maxlag..=maxlag {
            let mut dot = 0.0;
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let j = i as i32 + lag;
                if j >= 0 && (j as usize) < n {
                    dot += a[i] * b[j as usize];
                }
            }
            let v = dot / (na * nb);
            if v > best_val {
                best_val = v;
                best_lag = lag;
            }
        }
        (best_lag, best_val.max(0.0))
    }

    #[test]
    fn lag_parallel_xcorr_matches_the_per_lag_loop_bit_for_bit() {
        // Random, sparse, constant, all-zero, single-spike, signed and tied traces at
        // every length through the 11-lag window and past it, including the lengths below
        // 2 * maxlag + 1 where no position has every lag valid, and a `b` longer than `a`
        // (the old loop read only its first `n` values but normed all of it).
        let mut rng = Lcg(0xc0ff_ee00_1234_5678);
        let mut checked = 0usize;
        for n in 0..48usize {
            for shape in 0..8u32 {
                let gen = |rng: &mut Lcg, len: usize| -> Vec<f64> {
                    (0..len)
                        .map(|i| match shape {
                            0 => rng.unit() * 1e4,
                            1 => {
                                if rng.below(3) == 0 {
                                    rng.unit() * 50.0
                                } else {
                                    0.0
                                }
                            }
                            2 => 7.0,
                            3 => 0.0,
                            4 => {
                                if i == len / 2 {
                                    3.5
                                } else {
                                    0.0
                                }
                            }
                            5 => rng.unit() - 0.5,
                            6 => (rng.below(4) as f64) * 0.25,
                            _ => {
                                if i % 2 == 0 {
                                    -0.0
                                } else {
                                    f64::MIN_POSITIVE * rng.unit()
                                }
                            }
                        })
                        .collect()
                };
                let a = gen(&mut rng, n);
                for extra in [0usize, 3] {
                    let b = gen(&mut rng, n + extra);
                    for maxlag in [5i32, 3] {
                        let want = old_best_xcorr(&a, &b, maxlag);
                        let got = best_xcorr(&a, &b, maxlag);
                        assert_eq!(
                            (got.0, got.1.to_bits()),
                            (want.0, want.1.to_bits()),
                            "n {n}, shape {shape}, extra {extra}, maxlag {maxlag}"
                        );
                        let normed =
                            best_xcorr_normed(&a, &b, maxlag, xcorr_norm(&a), xcorr_norm(&b));
                        assert_eq!((normed.0, normed.1.to_bits()), (want.0, want.1.to_bits()));
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 48 * 8 * 2 * 2);
    }

    #[test]
    fn order_statistics_by_selection_equal_the_sorted_copy_bit_for_bit() {
        // The sort-based median and quantiles exactly as the families wrote them, against
        // selection, on vectors with ties, zeros of both signs, subnormals, infinities and
        // NaN (which `total_cmp` orders like any other value).
        fn old_median(v: &[f64]) -> f64 {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.total_cmp(b));
            let n = s.len();
            if n % 2 == 1 {
                s[n / 2]
            } else {
                0.5 * (s[n / 2 - 1] + s[n / 2])
            }
        }
        fn old_quantile(v: &[f64], q: f64) -> f64 {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.total_cmp(b));
            let n = s.len();
            if n == 0 {
                return 0.0;
            }
            if n == 1 {
                return s[0];
            }
            let pos = q * (n - 1) as f64;
            let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
            if lo == hi {
                s[lo]
            } else {
                let frac = pos - lo as f64;
                s[lo] * (1.0 - frac) + s[hi] * frac
            }
        }
        let mut rng = Lcg(0x0bad_cafe_d00d_f00d);
        for n in 1..70usize {
            for kind in 0..4u32 {
                let v: Vec<f64> = (0..n)
                    .map(|_| match (kind, rng.below(12)) {
                        (0, _) => rng.unit() * 100.0,
                        (1, r) => (r % 4) as f64, // heavy ties
                        (2, 0) => -0.0,
                        (2, 1) => 0.0,
                        (2, 2) => f64::MIN_POSITIVE / 8.0,
                        (2, 3) => f64::INFINITY,
                        (2, _) => rng.unit() - 0.5,
                        (_, 0) => f64::NAN,
                        (_, _) => rng.unit(),
                    })
                    .collect();
                let bits = |x: f64| x.to_bits();
                assert_eq!(
                    bits(median_select(&mut v.clone())),
                    bits(old_median(&v)),
                    "median, n {n}, kind {kind}"
                );
                let mut buf = v.clone();
                for q in [0.75, 0.25, 0.0, 1.0, 0.5, 1.0 / 3.0] {
                    assert_eq!(
                        bits(quantile_select(&mut buf, q)),
                        bits(old_quantile(&v, q)),
                        "quantile {q}, n {n}, kind {kind}"
                    );
                }
                let mut sorted = v.clone();
                sorted.sort_by(|a, b| a.total_cmp(b));
                for k in [0, n / 2, n - 1] {
                    assert_eq!(bits(order_stat(&mut v.clone(), k)), bits(sorted[k]));
                }
            }
        }
        assert_eq!(quantile_select(&mut [], 0.5), 0.0);
    }

    /// Every field of a `FragFeatures` as bit patterns, for exact comparison.
    fn ff_bits(f: &FragFeatures) -> Vec<u64> {
        [
            f.frag_corr,
            f.frag_cosine,
            f.spectral_angle,
            f.coelution_mean,
            f.coelution_best,
            f.n_coelution_above,
            f.norm_manhattan,
            f.rmsd,
            f.xcorr_coelution,
            f.xcorr_shape,
            f.sum_b,
            f.sum_y,
            f.n_b,
            f.n_y,
            f.weighted_mass_error,
            f.mean_mass_error,
            f.log_sn,
            f.n_observations,
            f.base_width_rt,
            f.profile_cos,
            f.ref_corr,
            f.best_ref_corr,
            f.low_frag_coel,
            f.evidence,
            f.contrast_min,
            f.resid_corr,
            f.coel_clean,
            f.shadow_frac,
            f.elution_lo,
            f.elution_hi,
        ]
        .iter()
        .map(|v| v.to_bits())
        .collect()
    }

    #[test]
    fn shared_pair_statistics_equal_the_per_reader_computations_bit_for_bit() {
        // Per PSM, the shared path against each reader's own computation on the SAME
        // evidence: `fragment_features` with and without the shared peak window, and the
        // Extended families with the `PairStats` cache and with it removed (every reader
        // then takes its fallback). Random traces with dropouts and ties, 1-12 fragments,
        // 3-60 points, per-candidate and global bounds, an empty trace now and then.
        let mut rng = Lcg(0x5a5a_1234_0f0f_9876);
        let mut checked = 0usize;
        for case in 0..400u32 {
            let k = 1 + rng.below(12) as usize;
            let npts = 3 + rng.below(58) as usize;
            let grid: Vec<f32> = (0..npts).map(|i| 300.0 + i as f32 * 1.25).collect();
            let centre = rng.below(npts as u32) as f64;
            let traces: Vec<Vec<f32>> = (0..k)
                .map(|f| {
                    if f > 0 && rng.below(15) == 0 {
                        return Vec::new();
                    }
                    let h = 10f64.powf(1.0 + 3.0 * rng.unit());
                    let w = 1.0 + 4.0 * rng.unit();
                    (0..npts)
                        .map(|i| {
                            if rng.below(4) == 0 {
                                return 0.0;
                            }
                            let x = (i as f64 - centre - (f % 3) as f64) / w;
                            let v = h * (-0.5 * x * x).exp() + h * 0.05 * rng.unit();
                            if case % 5 == 0 {
                                ((v / 50.0).round() * 50.0) as f32
                            } else {
                                v as f32
                            }
                        })
                        .collect()
                })
                .collect();
            let names: Vec<String> = (0..k)
                .map(|_| {
                    let s = if rng.below(2) == 0 { 'b' } else { 'y' };
                    let z = if rng.below(4) == 0 { "^2" } else { "" };
                    format!("{s}{}{z}", 1 + rng.below(14))
                })
                .collect();
            let preds: Vec<f32> = (0..k).map(|_| rng.unit() as f32).collect();
            let rows: Vec<ChromRow> = (0..k)
                .map(|f| {
                    let rt: &[f32] = if traces[f].is_empty() { &[] } else { &grid };
                    ChromRow {
                        frag_name: &names[f],
                        frag_mz: 200.0 + f as f64 * 50.0,
                        frag_obs_mz: 200.0 + f as f64 * 50.0 + 0.001,
                        pred_int: preds[f],
                        rt,
                        inten: &traces[f],
                    }
                })
                .collect();
            let apex = grid[centre as usize] as f64 + 0.3;
            let bounds = if case % 2 == 0 {
                None
            } else {
                Some((4.0, 6.5))
            };
            let (frac, grace) = (1.0 / 3.0, (case % 3 == 0) as usize);

            let al = align_traces(&rows);
            let obs = apex_intensities(&rows, apex);
            let pred: Vec<f64> = rows.iter().map(|r| r.pred_int as f64).collect();
            let win = peak_window(
                &al.axis_full,
                &al.traces_full,
                &pred,
                apex,
                frac,
                grace,
                bounds,
            );
            let peak = PeakTraces::new(&al, &pred, win);
            let shared = fragment_features(
                &rows,
                &al,
                &obs,
                Some(&peak),
                apex,
                0.9,
                true,
                frac,
                grace,
                bounds,
            );
            let own =
                fragment_features(&rows, &al, &obs, None, apex, 0.9, true, frac, grace, bounds);
            assert_eq!(
                ff_bits(&shared),
                ff_bits(&own),
                "fragment_features, case {case}"
            );

            let mut ev = evidence_from(&rows, al, obs, pred, peak, &[], apex);
            ev.n_matched = k as i32;
            ev.n_predicted = k as i32;
            let with_cache = extended_values(&ev);
            ev.pair_stats = None;
            let without = extended_values(&ev);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&with_cache), bits(&without), "families, case {case}");
            checked += 1;
        }
        assert_eq!(checked, 400);
    }

    #[test]
    fn shared_axis_alignment_reproduces_the_union_build() {
        // The normal case: every fragment of a candidate samples ONE stored axis, so the
        // fast path applies and must be bit-identical to the map build it skips.
        let grid = [10.0f32, 11.0, 12.0, 13.0];
        let a = [0.0f32, 4.0, 9.0, 2.0];
        let b = [1.0f32, 5.0, 8.0, 0.0];
        let rows = vec![row("y1", 1.0, &grid, &a), row("y2", 0.5, &grid, &b)];
        assert!(
            align_shared_axis(&rows).is_some(),
            "rows sharing one stored axis must take the fast path"
        );
        same(&align_traces(&rows), &align_union(&rows));

        // A predicted-but-unobserved fragment carries no axis at all; it is zero-filled
        // by both builds and must not disable the fast path for the others.
        let rows = vec![
            row("y1", 1.0, &grid, &a),
            row("y2", 0.5, &[], &[]),
            row("y3", 0.2, &grid, &b),
        ];
        assert!(align_shared_axis(&rows).is_some());
        let al = align_traces(&rows);
        same(&al, &align_union(&rows));
        assert_eq!(
            al.traces_full[1],
            vec![0.0; 4],
            "empty trace is zero-filled"
        );
    }

    #[test]
    fn alignment_falls_back_when_the_axes_are_not_one_shared_grid() {
        let g1 = [10.0f32, 11.0, 12.0];
        let g2 = [10.5f32, 11.0, 12.5];
        let v1 = [1.0f32, 2.0, 3.0];
        let v2 = [4.0f32, 5.0, 6.0];
        // Two different grids: the union has five points and each row is zero elsewhere.
        let rows = vec![row("y1", 1.0, &g1, &v1), row("y2", 1.0, &g2, &v2)];
        assert!(align_shared_axis(&rows).is_none(), "distinct axes");
        let al = align_traces(&rows);
        assert_eq!(al.axis_full, vec![10.0, 10.5, 11.0, 12.0, 12.5]);
        assert_eq!(al.traces_full[0], vec![1.0, 0.0, 2.0, 3.0, 0.0]);
        assert_eq!(al.traces_full[1], vec![0.0, 4.0, 5.0, 0.0, 6.0]);

        // Equal VALUES stored separately ARE the same axis. This used to be rejected,
        // because the precondition was `ptr::eq` and the fast path's correctness rested on
        // the chunk store's provenance rather than on the values; the answer was the same
        // then and is the same now, which is what `same` checks.
        let copy = g1;
        let rows = vec![row("y1", 1.0, &g1, &v1), row("y2", 1.0, &copy, &v2)];
        assert!(align_shared_axis(&rows).is_some());
        same(&align_traces(&rows), &align_union(&rows));

        // A non-ascending axis would be reordered by the sort, and a repeated RT value
        // would be collapsed by the dedup; both must refuse the fast path.
        let unsorted = [12.0f32, 10.0, 11.0];
        let rows = vec![row("y1", 1.0, &unsorted, &v1)];
        assert!(align_shared_axis(&rows).is_none(), "unsorted axis");
        same(&align_traces(&rows), &align_union(&rows));
        let dup = [10.0f32, 10.0, 11.0];
        let rows = vec![row("y1", 1.0, &dup, &v1)];
        assert!(align_shared_axis(&rows).is_none(), "repeated RT");
        same(&align_traces(&rows), &align_union(&rows));
    }

    // --- cross-charge corroboration (dense ids must reproduce the set build) ---

    #[test]
    fn charge_state_bitmask_matches_the_set_build() {
        // Five rows over three peptidoforms, charges repeated within a peptidoform.
        let pf = [0u32, 0, 1, 1, 2, 0];
        let z = [2i32, 3, 2, 2, 4, 2];
        let by_mask = charge_states_by_mask(&pf, 3, &z).expect("charges are in 0..32");
        assert_eq!(by_mask, charge_states_by_set(&pf, 3, &z));
        assert_eq!(by_mask, vec![2.0, 2.0, 1.0, 1.0, 1.0, 2.0]);
        assert_eq!(charge_states_per_row(&pf, 3, &z), by_mask);

        // Charge 0 and charge 31 are inside the mask; 32 and a negative are not, and the
        // fallback then carries the count.
        let edge = [0i32, 31, 0, 31, 0, 0];
        assert_eq!(
            charge_states_by_mask(&pf, 3, &edge).expect("0 and 31 fit"),
            charge_states_by_set(&pf, 3, &edge)
        );
        let out = [32i32, 3, 2, 2, 4, 2];
        assert!(charge_states_by_mask(&pf, 3, &out).is_none());
        assert_eq!(
            charge_states_per_row(&pf, 3, &out),
            charge_states_by_set(&pf, 3, &out)
        );
        let neg = [-1i32, 3, 2, 2, 4, 2];
        assert!(charge_states_by_mask(&pf, 3, &neg).is_none());
        assert_eq!(
            charge_states_per_row(&pf, 3, &neg),
            charge_states_by_set(&pf, 3, &neg)
        );
    }

    #[test]
    #[should_panic(expected = "does not cover every PSM row")]
    fn a_short_charge_column_is_an_error_not_a_partial_reduction() {
        // Both builds zip the peptidoform ids with the charges, and a zip truncates where
        // the `charge[i]` indexing it replaced panicked. A charge column that does not cover
        // every PSM row is an artifact-shape error, so it must stay loud rather than
        // silently reducing over a prefix.
        let pf = [0u32, 0, 1, 1];
        charge_states_per_row(&pf, 2, &[2i32, 3]);
    }

    // --- the feature -> column permutation ---

    #[test]
    fn colix_covers_every_active_column() {
        for set in [FeatureSet::Minimal, FeatureSet::Rich, FeatureSet::Extended] {
            let cols = active_features(set);
            let ext = extended_name_refs();
            let ix = ColIx::new(&cols, &ext);
            // Every column the assembly resolves must agree with the name lookup it
            // replaced, and together they must account for each active column exactly
            // once: a mistyped field would resolve to None and leave a column unwritten.
            let mut seen: Vec<usize> = ix
                .named()
                .into_iter()
                .chain(ix.ext.iter().copied())
                .flatten()
                .collect();
            seen.sort_unstable();
            assert_eq!(
                seen,
                (0..cols.len()).collect::<Vec<_>>(),
                "{set:?}: the assembly does not write each active column exactly once"
            );
            // The extended names resolve only under the Extended set.
            if !matches!(set, FeatureSet::Extended) {
                assert!(ix.ext.iter().all(|c| c.is_none()), "{set:?}");
            }
        }
    }

    #[test]
    fn colix_resolves_each_field_to_the_column_its_own_name_denotes() {
        // Coverage alone only proves the permutation is TOTAL. It cannot see a field
        // resolving to the wrong column, because a permutation stays a permutation under a
        // swap. This pins each field to the index of the column spelled like it, which is
        // what the string literal it replaced did, taking the names from the field list via
        // `stringify!` rather than repeating them.
        for set in [FeatureSet::Minimal, FeatureSet::Rich, FeatureSet::Extended] {
            let cols = active_features(set);
            let ext = extended_name_refs();
            let ix = ColIx::new(&cols, &ext);
            for (name, got) in ix.named_pairs() {
                let want = cols.iter().position(|c| c == name);
                assert_eq!(got, want, "{set:?}: field '{name}' resolved to {got:?}");
            }
            for (name, got) in ext.iter().zip(&ix.ext) {
                let want = cols.iter().position(|c| c == name);
                assert_eq!(*got, want, "{set:?}: extended '{name}' resolved to {got:?}");
            }
        }
        // This still cannot catch a transposition in the ASSEMBLY itself -- writing
        // `ff.frag_corr` into the `frag_cosine` column and vice versa resolves both fields
        // correctly and covers every column once. No structural check can: a swap leaves a
        // permutation a permutation. `extended_features_match_the_pre_permutation_build`
        // is the test that sees it, by comparing the values against a digest captured from
        // the name-keyed assembly.
    }

    #[test]
    fn value_matrix_hands_its_columns_over_by_move() {
        let mut m = ValueMatrix::new(3, 2);
        m.set(Some(1), 0, 7.5);
        m.set(Some(1), 1, -1.0);
        m.set(None, 0, 99.0); // a feature outside the active set is dropped
        assert_eq!(m.column(1), &[7.5, -1.0]);
        // A taken column leaves nothing behind: the write moves the values out rather than
        // copying them, which is what stopped the matrix existing twice at the write.
        let before = m.payload_bytes();
        assert_eq!(m.take_column(1), vec![7.5, -1.0]);
        assert_eq!(m.column(0), &[0.0, 0.0]);
        assert_eq!(
            m.payload_bytes(),
            before - 2 * std::mem::size_of::<f64>(),
            "payload_bytes must measure what is still held, not the original shape"
        );
    }

    #[test]
    fn value_matrix_columns_are_one_block_each() {
        // The per-column layout is what lets the writer take the columns by move, and it is
        // also the one place this change adds heap blocks rather than removing them. The
        // arithmetic that makes it acceptable: one block per column, at most two matrices in
        // flight, against the 1,048,576 mappings a process gets.
        let cols = active_features(FeatureSet::Extended).len();
        let m = ValueMatrix::new(cols, 4096);
        assert_eq!(m.cols.len(), cols);
        assert!(
            2 * cols < 1_048_576 / 1000,
            "two matrices in flight must stay far under the mapping limit: {cols} columns"
        );
        assert_eq!(m.payload_bytes(), cols * 4096 * std::mem::size_of::<f64>());
    }

    #[test]
    fn the_in_flight_peak_is_not_two_matrices() {
        // What the report used to assert (`2 * largest_chunk`) was the in-line write's peak:
        // one matrix plus the copy of it. With the writer thread there is no copy, but the
        // previous chunk's columns are still held while this chunk's matrix AND its extended
        // buffer are live, so the overlap is wider than 2x. The report measures the three;
        // this pins the arithmetic that makes the old constant wrong.
        let (rows, n_cols) = (4096usize, active_features(FeatureSet::Extended).len());
        let n_ext = extended_name_refs().len();
        let matrix = n_cols * rows * std::mem::size_of::<f64>();
        let ext = n_ext * rows * std::mem::size_of::<f64>();
        let in_flight = matrix + ext + matrix;
        assert!(
            in_flight > 2 * matrix,
            "the overlap must be reported as more than two matrices: {in_flight} vs {}",
            2 * matrix
        );
        assert!(
            n_ext * 100 / n_cols >= 80,
            "the extended buffer is ~0.87x a matrix"
        );
    }

    // --- confident-bounds row-group pruning ---

    fn stats(groups: &[(usize, u32, u32)]) -> Vec<mumdia_io::table::RowGroupStats> {
        groups
            .iter()
            .map(|&(rows, lo, hi)| mumdia_io::table::RowGroupStats {
                rows,
                min: Some(lo as f64),
                max: Some(hi as f64),
            })
            .collect()
    }

    #[test]
    fn confident_row_spans_keep_every_group_that_can_hold_an_anchor() {
        // Four groups of 10 rows covering candidate ids 0-9, 10-19, 20-29, 30-39.
        let s = stats(&[(10, 0, 9), (10, 10, 19), (10, 20, 29), (10, 30, 39)]);
        // Anchors in groups 0 and 2 only.
        assert_eq!(
            confident_row_spans(&s, &[3, 25]).unwrap(),
            vec![(0, 10), (20, 10)]
        );
        // Adjacent kept groups merge into one span.
        assert_eq!(
            confident_row_spans(&s, &[3, 15]).unwrap(),
            vec![(0, 20)],
            "adjacent groups must merge"
        );
        // An anchor in a gap between two groups' ranges keeps neither.
        let gapped = stats(&[(10, 0, 9), (10, 20, 29)]);
        assert!(confident_row_spans(&gapped, &[15]).unwrap().is_empty());
        // No anchors at all: nothing to read.
        assert!(confident_row_spans(&s, &[]).unwrap().is_empty());
        // Every group holding an anchor is kept, and the kept spans cover them.
        let all: Vec<u32> = (0..40).collect();
        assert_eq!(confident_row_spans(&s, &all).unwrap(), vec![(0, 40)]);
        // Missing statistics mean "read everything", the behaviour before the pruning.
        let blind = vec![mumdia_io::table::RowGroupStats {
            rows: 10,
            min: None,
            max: None,
        }];
        assert!(confident_row_spans(&blind, &[3]).is_none());
        assert!(confident_row_spans(&[], &[3]).is_none());
    }

    /// A chromatogram table with several row groups, its confident candidates, and the
    /// PSM apex RTs, for the pruning-equality test below.
    ///
    /// The fragment count per candidate VARIES (4 to 9), so candidates straddle the 12-row
    /// group boundaries. That is the configuration the pruning's safety argument is about
    /// and the previous fixture never produced: at a fixed 6 rows per candidate every
    /// boundary fell exactly between two candidates, so "a candidate spanning two row
    /// groups puts its id inside BOTH ranges and both are kept" was never exercised.
    /// The anchor pattern also leaves skipped groups between kept ones.
    fn craft_chrom_for_bounds(path: &str) -> (HashMap<u32, Vec<usize>>, Vec<f64>) {
        let (mut cid, mut name): (Vec<u32>, Vec<String>) = (Vec::new(), Vec::new());
        let (mut fmz, mut pint): (Vec<f64>, Vec<f32>) = (Vec::new(), Vec::new());
        let (mut rt, mut inten): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
        let mut apex_rt: Vec<f64> = Vec::new();
        let mut confident: HashMap<u32, Vec<usize>> = HashMap::new();
        for c in 0..200u32 {
            let apex = 100.0 + c as f64 * 5.0;
            apex_rt.push(apex);
            // Every fifth candidate is a confident anchor, so most row groups hold none.
            if c % 5 == 0 {
                confident.insert(c, vec![c as usize]);
            }
            let grid: Vec<f32> = (0..9).map(|k| (apex - 4.0 + k as f64) as f32).collect();
            // 4..=9 fragments, so the running row total is not a multiple of the 12-row
            // group size and candidates land across boundaries.
            let nfrag = 4 + c % 6;
            for f in 0..nfrag {
                cid.push(c);
                name.push(format!("y{}", f + 1));
                fmz.push(200.0 + f as f64 * 30.0);
                pint.push(1.0 / (f + 1) as f32);
                rt.push(grid.clone());
                inten.push(
                    (0..9)
                        .map(|k| {
                            let x = k as f32 - 4.0;
                            (100.0 - f as f32 * 10.0) / (1.0 + x * x)
                        })
                        .collect(),
                );
            }
        }
        // Small row groups (two candidates each), so the pruning has something to prune.
        let mut w = TableWriter::new(path).with_row_group_rows(12);
        w.write_cols(vec![
            Col::U32("candidate_id".into(), cid),
            Col::Str("frag_name".into(), name),
            Col::F64("frag_mz".into(), fmz),
            Col::F32("predicted_intensity".into(), pint),
            Col::ListF32("rt".into(), rt),
            Col::ListF32("intensity".into(), inten),
        ])
        .unwrap();
        w.close().unwrap();
        (confident, apex_rt)
    }

    #[test]
    fn pruned_confident_bounds_equal_the_whole_table_scan() {
        let dir = std::env::temp_dir().join("mumdia_features_bounds");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir
            .join("chrom_bounds.parquet")
            .to_string_lossy()
            .to_string();
        let (confident, apex_rt) = craft_chrom_for_bounds(&path);
        let ch = TableFile::open(&path).unwrap();
        let cfg = FeaturesConfig::default();

        let mut sorted: Vec<u32> = confident.keys().copied().collect();
        sorted.sort_unstable();
        let rg = ch.row_group_stats("candidate_id").unwrap();

        // The whole safety argument is about a candidate whose rows cross a row-group
        // boundary: its id is then inside BOTH groups' [min, max] and neither can be
        // skipped. Assert the fixture actually produces that, because the previous one --
        // a fixed 6 rows per candidate into 12-row groups -- never did, and the argument
        // went untested while the test passed.
        let straddlers = rg.windows(2).filter(|w| w[0].max == w[1].min).count();
        assert!(
            straddlers > 5,
            "the fixture must make candidates straddle group boundaries; only {straddlers} do"
        );
        // And a SKIPPED group must sit between two kept ones, which is what forces the
        // chunk grid inside a span to stay on absolute rows rather than restart per span.
        let kept: Vec<bool> = rg
            .iter()
            .map(|s| {
                let (lo, hi) = (s.min.unwrap(), s.max.unwrap());
                sorted.iter().any(|&c| (c as f64) >= lo && (c as f64) <= hi)
            })
            .collect();
        assert!(
            kept.windows(3).any(|w| w[0] && !w[1] && w[2]),
            "the fixture must skip a group between two kept ones"
        );

        let spans = confident_row_spans(&rg, &sorted).expect("the writer records statistics");
        let read: usize = spans.iter().map(|s| s.1).sum();
        assert!(
            read < ch.nrows,
            "the pruning must skip something: read {read} of {} rows",
            ch.nrows
        );
        assert!(read > 0);
        assert!(spans.len() > 1, "a single span would not exercise the grid");

        // Same half-widths at several chunk sizes, pruned against the whole-table scan.
        // The chunk size matters: a candidate that straddles a chunk boundary is bounded
        // from each part separately, and the pruned pass has to split it in exactly the
        // same places for the percentiles to come out the same. Chunk sizes both below and
        // above the 12-row group size, so a span covers several groups in one chunk.
        for chunk in [7usize, 16, 64, 100, 1 << 20] {
            let pruned =
                confident_global_bounds(&ch, &spans, &confident, &apex_rt, &cfg, chunk).unwrap();
            let whole =
                confident_global_bounds(&ch, &[(0, ch.nrows)], &confident, &apex_rt, &cfg, chunk)
                    .unwrap();
            let bits = |v: Option<(f64, f64)>| v.map(|(l, r)| (l.to_bits(), r.to_bits()));
            assert_eq!(
                bits(pruned),
                bits(whole),
                "chunk {chunk}: pruned half-widths differ from the whole-table scan"
            );
            assert!(pruned.is_some(), "chunk {chunk}: 40 anchors is enough");
        }
    }

    #[test]
    fn parallel_confident_bounds_match_the_serial_pass_sample_for_sample() {
        // The parallel pass must not be judged on the two percentiles alone: a percentile
        // survives a reordering, and would survive a sample being read twice as long as
        // its rank landed elsewhere. This compares the SAMPLES, bit for bit and in order,
        // against the pass as it ran before it was parallelised.
        let dir = std::env::temp_dir().join("mumdia_features_bounds_par");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chrom_par.parquet").to_string_lossy().to_string();
        let (confident, apex_rt) = craft_chrom_for_bounds(&path);
        let ch = TableFile::open(&path).unwrap();
        let cfg = FeaturesConfig::default();

        let mut sorted: Vec<u32> = confident.keys().copied().collect();
        sorted.sort_unstable();
        let rg = ch.row_group_stats("candidate_id").unwrap();
        let spans = confident_row_spans(&rg, &sorted).expect("the writer records statistics");
        assert!(spans.len() > 1, "a single span would not exercise the grid");

        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        // Chunk sizes below and above the 12-row group size, and one that swallows the
        // whole table (a single sub-chunk, so the parallel driver degenerates to one
        // decoder and must still agree).
        //
        // And at four POOL SIZES, because `rayon::current_num_threads()` is the one input
        // that changes how `confident_half_widths` groups the sub-chunks: the group size
        // is `subs.len().div_ceil(decoders)`, so 1, 3, 8 and the ambient pool give four
        // different partitions of the same sub-chunk list. The argument that the partition
        // cannot matter is in the function's doc comment; this executes it rather than
        // leaving it argued. The serial arm is unaffected by the pool and is recomputed
        // inside each pool anyway, so every arm is compared against the same reference.
        let ambient = rayon::current_num_threads();
        let mut pools: Vec<usize> = vec![1, 3, BOUNDS_DECODERS, ambient];
        pools.sort_unstable();
        pools.dedup();
        for threads in pools {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for chunk in [7usize, 16, 64, 100, 1 << 20] {
                    for sp in [spans.clone(), vec![(0, ch.nrows)]] {
                        let (pl, pr) =
                            confident_half_widths(&ch, &sp, &confident, &apex_rt, &cfg, chunk)
                                .unwrap();
                        let (sl, sr) = confident_half_widths_serial(
                            &ch, &sp, &confident, &apex_rt, &cfg, chunk,
                        )
                        .unwrap();
                        assert!(
                            !sl.is_empty(),
                            "chunk {chunk}: the fixture must yield anchors"
                        );
                        assert_eq!(
                            bits(&pl),
                            bits(&sl),
                            "{threads} threads, chunk {chunk}: left half-widths differ"
                        );
                        assert_eq!(
                            bits(&pr),
                            bits(&sr),
                            "{threads} threads, chunk {chunk}: right half-widths differ"
                        );
                    }
                }
            });
        }

        // Nested inside a `par_iter`, which is how `run_groups` and `run_experiment` call
        // the stage. Both calls contend for the same process-wide decoder budget, so one of
        // them runs on fewer decoders than it asked for -- and must still produce the serial
        // pass's samples.
        let (sl, sr) =
            confident_half_widths_serial(&ch, &spans, &confident, &apex_rt, &cfg, 16).unwrap();
        let both: Vec<(Vec<f64>, Vec<f64>)> = (0..4u32)
            .into_par_iter()
            .map(|_| confident_half_widths(&ch, &spans, &confident, &apex_rt, &cfg, 16).unwrap())
            .collect();
        for (i, (l, r)) in both.iter().enumerate() {
            assert_eq!(
                bits(l),
                bits(&sl),
                "nested call {i}: left half-widths differ"
            );
            assert_eq!(
                bits(r),
                bits(&sr),
                "nested call {i}: right half-widths differ"
            );
        }

        // The sub-chunk grid is the thing that makes the above hold, so pin it directly:
        // boundaries sit on absolute multiples of `chunk_rows`, never restarted per span.
        let subs = confident_subchunks(&[(5, 20), (40, 9)], 8);
        assert_eq!(
            subs,
            vec![(5, 3), (8, 8), (16, 8), (24, 1), (40, 8), (48, 1)]
        );
        assert_eq!(confident_subchunks(&[], 8), vec![]);
    }

    /// A production-SHAPED chromatogram table: ~220-point traces (a 2 h gradient at the
    /// HYE benchmark's cycle time), 20 fragment rows per candidate, one shared grid per
    /// candidate. `n_cand` candidates, every tenth confident. Returns the confident row
    /// map and the apex RTs. Used only by the ignored benchmark below, which is why it is
    /// allowed to write ~100 MB.
    fn craft_big_chrom(path: &str, n_cand: u32) -> (HashMap<u32, Vec<usize>>, Vec<f64>) {
        const POINTS: usize = 220;
        const FRAGS: u32 = 20;
        let (mut cid, mut name): (Vec<u32>, Vec<String>) = (Vec::new(), Vec::new());
        let (mut fmz, mut pint): (Vec<f64>, Vec<f32>) = (Vec::new(), Vec::new());
        let (mut rt, mut inten): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
        let mut apex_rt: Vec<f64> = Vec::new();
        let mut confident: HashMap<u32, Vec<usize>> = HashMap::new();
        for c in 0..n_cand {
            let apex = 300.0 + (c % 4000) as f64 * 0.7;
            apex_rt.push(apex);
            if c % 10 == 0 {
                confident.insert(c, vec![c as usize]);
            }
            let grid: Vec<f32> = (0..POINTS)
                .map(|k| (apex - 55.0 + k as f64 * 0.5) as f32)
                .collect();
            for f in 0..FRAGS {
                cid.push(c);
                name.push(format!("y{}", f + 1));
                fmz.push(200.0 + f as f64 * 30.0);
                pint.push(1.0 / (f + 1) as f32);
                rt.push(grid.clone());
                inten.push(
                    (0..POINTS)
                        .map(|k| {
                            let x = k as f32 - 110.0;
                            (1000.0 - f as f32 * 20.0) / (1.0 + 0.05 * x * x)
                        })
                        .collect(),
                );
            }
        }
        // Row groups that DIVIDE the benchmark's sub-chunk size, as extract's 65,536-row
        // groups divide the production 2^20-row sub-chunk. Misaligned groups would make
        // every decoder decompress pages belonging to another's span and measure that
        // instead of the parallelism.
        let mut w = TableWriter::new(path).with_row_group_rows(2_048);
        w.write_cols(vec![
            Col::U32("candidate_id".into(), cid),
            Col::Str("frag_name".into(), name),
            Col::F64("frag_mz".into(), fmz),
            Col::F32("predicted_intensity".into(), pint),
            Col::ListF32("rt".into(), rt),
            Col::ListF32("intensity".into(), inten),
        ])
        .unwrap();
        w.close().unwrap();
        (confident, apex_rt)
    }

    #[test]
    #[ignore = "benchmark: writes ~100 MB and decodes it four times"]
    fn bench_confident_bounds_serial_against_parallel() {
        // Both arms decode the WHOLE table from the same file with the same sub-chunk
        // grid and the same filter; the only difference is how many threads do it. The
        // serial arm is the code as it shipped (`confident_half_widths_serial`), not a
        // re-description of it, and its samples are compared with the parallel arm's, so
        // a faster arm that read less would fail rather than win.
        //
        // What this measures, and what it does NOT. The two arms are fair to each other,
        // but the ratio between them is an UPPER BOUND on the production saving, not an
        // estimate of it, on four counts:
        //   - ~105 MB, deliberately warmed, so both arms read from the page cache and the
        //     whole benchmark is CPU-bound decode. A production pass streams ~68 GB, where
        //     eight concurrent readers can instead become storage-bandwidth-bound;
        //   - every candidate's trace is `grid.clone()` plus a smooth Lorentzian, so the
        //     file is far more snappy-compressible than real traces and decompression is a
        //     larger share of the work here than it is there;
        //   - 300 of 3,000 candidates are confident (10%), against the ~0.84% the stage
        //     sees in production, so the post-filter bounding work is ~12x over-weighted;
        //   - one host, one pool size.
        // So do not turn this ratio into a stage-level projection. The stage timing that
        // would settle it is docs/27 section 3.4 re-measured, which this has not been.
        let dir = std::env::temp_dir().join("mumdia_features_bench");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chrom_big.parquet").to_string_lossy().to_string();
        let t = Instant::now();
        let (confident, apex_rt) = craft_big_chrom(&path, 3_000);
        let write = t.elapsed();
        let ch = TableFile::open(&path).unwrap();
        let cfg = FeaturesConfig::default();
        let spans = vec![(0usize, ch.nrows)];
        // 15 sub-chunks over 60,000 rows, which is the ratio the production constant
        // gives on a real table (2^20 rows per sub-chunk over ~38.8M rows = 38).
        let chunk = 4_096usize;
        assert_eq!(confident_subchunks(&spans, chunk).len(), 15);

        // One untimed pass of each arm, so neither pays the page cache's first read.
        let warm_s =
            confident_half_widths_serial(&ch, &spans, &confident, &apex_rt, &cfg, chunk).unwrap();
        let warm_p = confident_half_widths(&ch, &spans, &confident, &apex_rt, &cfg, chunk).unwrap();
        assert_eq!(warm_s.0.len(), warm_p.0.len());

        let t = Instant::now();
        let (sl, sr) =
            confident_half_widths_serial(&ch, &spans, &confident, &apex_rt, &cfg, chunk).unwrap();
        let serial = t.elapsed();
        let t = Instant::now();
        let (pl, pr) =
            confident_half_widths(&ch, &spans, &confident, &apex_rt, &cfg, chunk).unwrap();
        let parallel = t.elapsed();

        assert_eq!(sl, pl);
        assert_eq!(sr, pr);
        println!(
            "confident-bounds pass over {} rows ({} anchors), {} threads capped at {}: \
             serial {:?}, parallel {:?}, speedup {:.2}x (fixture write {:?})",
            ch.nrows,
            sl.len(),
            rayon::current_num_threads(),
            BOUNDS_DECODERS,
            serial,
            parallel,
            serial.as_secs_f64() / parallel.as_secs_f64().max(1e-9),
            write
        );
    }

    #[test]
    #[ignore = "benchmark: decodes one batch of production-shaped traces many times"]
    fn bench_list_row_against_append_row() {
        // Both arms do the SAME final work -- copy the row into the flat buffer a chunk
        // keeps -- so the only difference is the scratch buffer and the `Arc` that
        // `append_row` needs to reach the values. Arm A allocates its scratch once,
        // outside the timed loop but inside the arm, which is the best case for the old
        // path rather than a straw man.
        let dir = std::env::temp_dir().join("mumdia_features_bench");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chrom_rows.parquet").to_string_lossy().to_string();
        craft_big_chrom(&path, 200); // 4,000 rows of 220 points
        let ch = TableFile::open(&path).unwrap();
        let mut reader = ch.batches(Some(&["rt", "intensity"]), 1 << 12).unwrap();
        let b = reader.next().unwrap().unwrap();
        let n = b.num_rows();
        let rt = ListF32::of(b.column(0), "rt").unwrap();
        let reps = 400;

        let mut sink: Vec<f32> = Vec::with_capacity(n * 220);
        let t = Instant::now();
        let mut scratch: Vec<f32> = Vec::new();
        for _ in 0..reps {
            sink.clear();
            for k in 0..n {
                scratch.clear();
                rt.append_row(k, &mut scratch, "rt").unwrap();
                sink.extend_from_slice(&scratch);
            }
        }
        let old = t.elapsed();
        let checksum_old = sink.len();

        let t = Instant::now();
        for _ in 0..reps {
            sink.clear();
            for k in 0..n {
                sink.extend_from_slice(rt.row_slice(k, "rt").unwrap());
            }
        }
        let new = t.elapsed();
        assert_eq!(checksum_old, sink.len());
        println!(
            "{} rows x {} reps: append_row + scratch {:?}, list_row {:?}, {:.2}x",
            n,
            reps,
            old,
            new,
            old.as_secs_f64() / new.as_secs_f64().max(1e-9)
        );
    }

    /// `coelution::weighted_reference` exactly as it stood before the fold: EVERY trace,
    /// weight `pred.get(i).unwrap_or(0.0)`, `n = t.min(tr.len())`. A transcription of
    /// deleted code, kept so the equality claim is pinned against the old build rather
    /// than against a second description of the new one -- do not tidy it into the new
    /// shape.
    #[allow(clippy::needless_range_loop)]
    fn old_coelution_weighted_reference(traces: &[Vec<f64>], pred: &[f64], t: usize) -> Vec<f64> {
        let mut r = vec![0.0f64; t];
        for (i, tr) in traces.iter().enumerate() {
            let w = pred.get(i).cloned().unwrap_or(0.0);
            let n = t.min(tr.len());
            for j in 0..n {
                r[j] += w * tr[j];
            }
        }
        r
    }

    /// `interference`'s inline `rfull` exactly as it stood before the fold: `for f in 0..k`
    /// under an `f < traces_full.len()` guard, `n = x.len().min(t)`. Same rule as above:
    /// a transcription, not a rewrite.
    #[allow(clippy::needless_range_loop)]
    fn old_interference_rfull(traces_full: &[Vec<f64>], pred: &[f64], t: usize) -> Vec<f64> {
        let mut rfull = vec![0.0f64; t];
        for f in 0..pred.len() {
            if f < traces_full.len() {
                let x = &traces_full[f];
                let w = pred[f];
                let n = x.len().min(t);
                for j in 0..n {
                    rfull[j] += w * x[j];
                }
            }
        }
        rfull
    }

    #[test]
    fn the_cached_full_window_reference_is_the_profile_both_families_built() {
        // Pins the OLD builds, which live in `old_coelution_weighted_reference` and
        // `old_interference_rfull` above rather than inline here. One negative predicted
        // intensity, because that is exactly where the third build
        // (`chromatographic::weighted_profile`, which clamps at zero) parts company and
        // why it is not folded in.
        let axis: Vec<f32> = (0..7).map(|k| 100.0 + k as f32).collect();
        let traces: Vec<Vec<f32>> = vec![
            vec![1.0, 3.0, 9.0, 20.0, 8.0, 2.0, 0.5],
            vec![0.5, 2.0, 7.0, 15.0, 6.0, 1.0, 0.25],
            vec![0.0, 1.0, 2.0, 4.0, 2.0, 0.5, 0.0],
        ];
        let preds = [0.8f32, -0.3, 0.5];
        let rows: Vec<ChromRow> = (0..3)
            .map(|i| ChromRow {
                frag_name: ["y3", "y4", "b2"][i],
                frag_mz: 300.0 + i as f64,
                frag_obs_mz: 300.0 + i as f64,
                pred_int: preds[i],
                rt: &axis,
                inten: &traces[i],
            })
            .collect();
        let al = align_traces(&rows);
        let e = build_evidence(&rows, al, &[], 103.0, 0.5, 1, None);

        let tf = e.axis_full.len();
        let old_coelution = old_coelution_weighted_reference(&e.traces_full, &e.pred, tf);
        let old_interference = old_interference_rfull(&e.traces_full, &e.pred, tf);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        assert_eq!(bits(&e.ref_profile_full), bits(&old_coelution));
        assert_eq!(bits(&e.ref_profile_full), bits(&old_interference));
        assert_eq!(e.ref_profile_full.len(), e.axis_full.len());
        // The negative weight is really in play, so the equality above is not the trivial
        // one, and the clamped third build really does differ.
        assert!(e.pred.iter().any(|&w| w < 0.0));
        let clamped: Vec<f64> = (0..tf)
            .map(|t| {
                e.traces_full
                    .iter()
                    .enumerate()
                    .map(|(f, x)| e.pred[f].max(0.0) * x[t])
                    .sum()
            })
            .collect();
        assert_ne!(
            bits(&e.ref_profile_full),
            bits(&clamped),
            "a clamped profile must not be substitutable for the raw-weighted one"
        );

        // The Evidence above has `traces_full.len() == pred.len()` and uniform trace
        // lengths, which is all `build_evidence` can produce, so on its own it never
        // exercises the two conditions under which the three builds could differ at all.
        // Call the function directly on them: traces both SHORTER and LONGER than the
        // window, and a trace with no predicted intensity behind it.
        let ragged: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0],                // shorter than the window
            vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5], // longer than the window
            vec![9.0, 9.0, 9.0, 9.0, 9.0],      // beyond `pred`: weighted 0.0, or skipped
        ];
        let rp = [0.8f64, -0.3];
        let t = 5usize;
        let new = weighted_reference_full(&ragged, &rp, t);
        assert_eq!(new.len(), t);
        assert_eq!(bits(&new), bits(&old_interference_rfull(&ragged, &rp, t)));
        assert_eq!(
            bits(&new),
            bits(&old_coelution_weighted_reference(&ragged, &rp, t))
        );
        assert!(new.iter().any(|&v| v != 0.0), "not the trivial equality");

        // Where the new build and the OLD COELUTION build are not identical, and why it
        // cannot be reached. The old build multiplied every trace past `pred` by a literal
        // 0.0, and `0.0 * NaN` is NaN, where `.take(pred.len())` skips it. coelution reads
        // the profile only under `has_full`, which requires
        // `traces_full.len() == pred.len()`, so a trace without a predicted intensity never
        // reaches it; interference's loop was already `min(k, traces_full.len())` and so
        // agrees with the new build even here.
        let mut poisoned = ragged.clone();
        poisoned[2][0] = f64::NAN;
        let new_p = weighted_reference_full(&poisoned, &rp, t);
        assert!(
            new_p[0].is_finite(),
            "the extra trace is skipped, not weighted"
        );
        assert!(
            old_coelution_weighted_reference(&poisoned, &rp, t)[0].is_nan(),
            "0.0 * NaN is the divergence the `has_full` guard makes unreachable"
        );
        assert_eq!(
            bits(&new_p),
            bits(&old_interference_rfull(&poisoned, &rp, t))
        );
    }

    #[test]
    fn a_wrong_length_full_window_reference_is_rebuilt_rather_than_read_short() {
        // `Evidence` is `pub` with `pub` fields and three test fixtures already fill
        // `ref_profile_full` with `vec![]`. Under a `debug_assert` a wrong length degraded
        // SILENTLY in release -- `sum_full_profile` 0, `peak_bounds` and the second-peak
        // scan skipped, `pearson` over the shorter overlap. Both readers now check the
        // length and rebuild, so every one of these must reproduce the values the correctly
        // built Evidence gives.
        let axis: Vec<f32> = (0..9).map(|k| 100.0 + k as f32).collect();
        let traces: Vec<Vec<f32>> = vec![
            vec![1.0, 3.0, 9.0, 20.0, 30.0, 18.0, 6.0, 2.0, 0.5],
            vec![0.5, 2.0, 7.0, 15.0, 24.0, 13.0, 4.0, 1.0, 0.25],
            vec![0.2, 1.0, 2.0, 4.0, 7.0, 3.0, 1.0, 0.5, 0.1],
        ];
        let preds = [0.8f32, 0.5, 0.3];
        let build = || {
            let rows: Vec<ChromRow> = (0..3)
                .map(|i| ChromRow {
                    frag_name: ["y3", "y4", "b2"][i],
                    frag_mz: 300.0 + i as f64,
                    frag_obs_mz: 300.0 + i as f64,
                    pred_int: preds[i],
                    rt: &axis,
                    inten: &traces[i],
                })
                .collect();
            let al = align_traces(&rows);
            build_evidence(&rows, al, &[], 104.0, 0.5, 1, None)
        };
        let good = build();
        let tf = good.axis_full.len();
        assert!(tf >= 3 && good.traces_full.len() == good.pred.len());
        assert_eq!(good.ref_profile_full.len(), tf);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        let want_coel = bits(&coelution::values(&good));
        let want_intf = bits(&interference::values(&good));
        // A field that is empty, one short, and one long: none of them may change a value.
        for bad in [vec![], vec![0.0; tf - 1], vec![1.0e9; tf + 3]] {
            let mut broken = build();
            broken.ref_profile_full = bad;
            assert_eq!(bits(&coelution::values(&broken)), want_coel);
            assert_eq!(bits(&interference::values(&broken)), want_intf);
        }
        // And the check really is a check: a wrong profile of the RIGHT length is read as
        // given, so the guard is a length contract and nothing more.
        let mut wrong = build();
        wrong.ref_profile_full = vec![1.0e9; tf];
        assert_ne!(bits(&interference::values(&wrong)), want_intf);
    }

    #[test]
    fn list_row_borrows_exactly_what_append_row_copied() {
        use arrow::array::{Float32Builder, LargeListBuilder, ListBuilder};
        // Rows of several lengths, an empty row and a NULL row: `append_row` leaves the
        // cleared scratch buffer empty for a null, and `list_row` must return an empty
        // slice for the same rows.
        //
        // BOTH offset widths. `ListF32::Large` is a separate arm of `list_row` with its own
        // `value_offsets()` (i64 rather than i32), and it is the arm a polars-written
        // chromatogram table takes -- the large_* foot-gun CLAUDE.md records for string
        // columns applies to list columns too. Testing only `ListBuilder` would leave the
        // arm the engine can actually meet on foreign input unexecuted.
        let mut b = ListBuilder::new(Float32Builder::new());
        let mut lb = LargeListBuilder::new(Float32Builder::new());
        for k in 0..12usize {
            match k % 4 {
                0 => {
                    b.append(false); // null row
                    lb.append(false);
                }
                1 => {
                    b.append(true); // empty row
                    lb.append(true);
                }
                _ => {
                    for j in 0..(k % 5 + 1) {
                        b.values().append_value(k as f32 * 10.0 + j as f32);
                        lb.values().append_value(k as f32 * 10.0 + j as f32);
                    }
                    b.append(true);
                    lb.append(true);
                }
            }
        }
        let small: ArrayRef = std::sync::Arc::new(b.finish());
        let large: ArrayRef = std::sync::Arc::new(lb.finish());
        // A LargeList is accepted and read through the same path as a List. The offset
        // width is resolved inside `ListF32::of`, so there is no variant left to match on;
        // what matters is that the wide form reads back, which the loop below checks value
        // by value against `append_row`.
        assert_eq!(ListF32::of(&large, "trace").unwrap().len(), 12);
        // Whole array, and the slices the loader takes at a chunk boundary: `value_offsets`
        // is adjusted for a sliced list, and this is where a hand-rolled offset would be
        // wrong.
        for arr in [&small, &large] {
            for (off, len) in [(0usize, 12usize), (3, 5), (7, 5), (11, 1)] {
                let sliced = arr.slice(off, len);
                let l = ListF32::of(&sliced, "trace").unwrap();
                for k in 0..len {
                    let mut want: Vec<f32> = vec![7.0; 3]; // non-empty: append_row appends
                    want.clear();
                    l.append_row(k, &mut want, "trace").unwrap();
                    assert_eq!(
                        l.row_slice(k, "trace").unwrap(),
                        want.as_slice(),
                        "slice {off}..{} row {k}",
                        off + len
                    );
                }
            }
        }
        // A non-f32 list is still an error rather than a silent reinterpretation, at both
        // offset widths. The error now surfaces when the column is OPENED rather than when
        // a row is read: `ListF32::of` resolves the child once instead of downcasting per
        // row, so a wrong inner type is refused before any row is handed out.
        let mut ib = ListBuilder::new(arrow::array::Int32Builder::new());
        ib.values().append_value(1);
        ib.append(true);
        let iarr: ArrayRef = std::sync::Arc::new(ib.finish());
        assert!(ListF32::of(&iarr, "trace").is_err());
        let mut ilb = LargeListBuilder::new(arrow::array::Int32Builder::new());
        ilb.values().append_value(1);
        ilb.append(true);
        let ilarr: ArrayRef = std::sync::Arc::new(ilb.finish());
        assert!(ListF32::of(&ilarr, "trace").is_err());
    }

    #[test]
    fn moving_a_chunk_boundary_moves_parquet_bytes_above_one_row_group() {
        // `CHUNK_PSM_ROWS` moves production chunk boundaries (39 chunks rather than 38 at
        // the docs/27 shape), and a chunk is one `TableWriter::write_cols` call. Whether
        // that moves the FILE was asserted rather than shown, and the smoke cannot show it
        // either way because it runs one chunk. So measure the writer directly, at the two
        // scales that matter.
        //
        // The answer is that it does, and the claim of a byte-identical features.parquet
        // under a binding PSM cap is therefore wrong: parquet splits a column chunk into
        // data pages on accumulated bytes, and the page-boundary check happens at the end
        // of each `write_batch_size` slice of each write CALL, so a different call pattern
        // lands the checks in different places. Only the feature VALUES are invariant, and
        // those are what `extended_features_are_chunk_invariant` pins. Nothing downstream
        // of features reads parquet bytes; the artifact hash in the manifest is provenance.
        let dir = std::env::temp_dir().join("mumdia_features_writecalls");
        std::fs::create_dir_all(&dir).unwrap();
        let write = |tag: &str, n: usize, step: usize| -> String {
            let p = dir
                .join(format!("wcb_{tag}.parquet"))
                .to_string_lossy()
                .to_string();
            let mut w = TableWriter::new(&p).with_row_group_rows(FEATURE_ROW_GROUP_ROWS);
            let mut lo = 0usize;
            while lo < n {
                let hi = (lo + step).min(n);
                let vals: Vec<f64> = (lo..hi).map(|i| (i as f64) * 1.000_001).collect();
                w.write_cols(vec![Col::F64("v".into(), vals)]).unwrap();
                lo = hi;
            }
            w.close().unwrap();
            p
        };
        let hash = |p: &str| mumdia_io::hash::blake3_file(p).unwrap();

        // Below one row group the whole table is encoded in a single flush, so the call
        // pattern cannot reach the page boundaries: byte-identical, which is why the 61-row
        // `extended_features_are_chunk_invariant` fixture is byte-identical in all three of
        // its arms and why that tells us nothing about a production table.
        let small = 4_000usize;
        assert_eq!(
            hash(&write("small_one", small, small)),
            hash(&write("small_many", small, 37)),
            "one row group: the call pattern cannot move a page"
        );

        // Above it, it can and does. 68,523 is the docs/27 PSM rows per chunk, i.e. exactly
        // the boundary this constant moves.
        let big = 300_000usize;
        let one = hash(&write("big_one", big, big));
        for step in [1_000usize, 40_000, 68_523] {
            assert_ne!(
                one,
                hash(&write(&format!("big_{step}"), big, step)),
                "step {step}: a moved write-call boundary is expected to move the bytes"
            );
        }
        // Not every pattern differs: a step equal to the row group size reproduces the
        // single-call file, which is why this has to be measured rather than reasoned.
        assert_eq!(
            one,
            hash(&write("big_rg", big, FEATURE_ROW_GROUP_ROWS)),
            "a step equal to the row group size lands every page check where one call did"
        );
    }

    #[test]
    fn the_bounds_decoder_budget_is_process_wide() {
        // `BOUNDS_DECODERS` is a MEMORY bound, and `features::run` is itself called from
        // inside a rayon `par_iter` when `groups.parallel > 1` or
        // `experiment.parallel_runs > 1`, where `rayon::current_num_threads()` still
        // reports the whole pool. A per-call cap would therefore have been eight decoders
        // per band. This pins that the budget is shared, that it is returned, and the bound
        // it actually gives.
        // Its own budget, not the shared one: other tests in this binary run concurrently
        // and legitimately hold leases against `BOUNDS_DECODER_BUDGET`.
        static BUDGET: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(BOUNDS_DECODERS);
        let free = || BUDGET.load(std::sync::atomic::Ordering::Acquire);
        assert_eq!(free(), BOUNDS_DECODERS, "the budget starts whole");
        {
            let a = DecoderLease::take_from(&BUDGET, BOUNDS_DECODERS);
            assert_eq!(a.decoders(), BOUNDS_DECODERS);
            assert_eq!(free(), 0);
            // A second concurrent call gets nothing and falls back to one decoder of its
            // own: never zero (it must make progress without waiting), never eight.
            let b = DecoderLease::take_from(&BUDGET, BOUNDS_DECODERS);
            assert_eq!(b.n, 0);
            assert_eq!(b.decoders(), 1, "a starved call decodes serially");
            // So C concurrent calls open at most BOUNDS_DECODERS + (C - 1) decoders.
            let c = DecoderLease::take_from(&BUDGET, BOUNDS_DECODERS);
            assert_eq!(
                a.decoders() + b.decoders() + c.decoders(),
                BOUNDS_DECODERS + 2
            );
        }
        assert_eq!(free(), BOUNDS_DECODERS, "every lease is returned on drop");
        // A partial take leaves the remainder for the next caller rather than rounding.
        {
            let a = DecoderLease::take_from(&BUDGET, 3);
            let b = DecoderLease::take_from(&BUDGET, BOUNDS_DECODERS);
            assert_eq!((a.n, b.n), (3, BOUNDS_DECODERS - 3));
            assert_eq!(free(), 0);
        }
        assert_eq!(free(), BOUNDS_DECODERS);
        // And the shared budget is the one the driver actually uses, whole when idle here.
        assert!(
            BOUNDS_DECODER_BUDGET.load(std::sync::atomic::Ordering::Acquire) <= BOUNDS_DECODERS,
            "the shared budget is never over-returned"
        );
    }

    #[test]
    fn row_group_pruning_saves_nothing_on_a_production_shaped_table() {
        // The doc comment on `confident_row_spans` claims the pruning is worth close to
        // nothing on real data. This measures that claim instead of asserting it, from the
        // shape extract actually writes: `CHROM_ROW_GROUP_ROWS = 1 << 16` rows per group and
        // about 10 chromatogram rows per candidate, so ~6,550 candidates per group, with the
        // confident set (seed PSMs at spectrum_q <= 0.01) spread over the id range.
        let rows_per_group = 1usize << 16;
        let rows_per_candidate = 10usize;
        let per_group = rows_per_group / rows_per_candidate;
        let n_groups = 200usize;
        let n_cand = (per_group * n_groups) as u32;
        let s: Vec<mumdia_io::table::RowGroupStats> = (0..n_groups)
            .map(|g| mumdia_io::table::RowGroupStats {
                rows: rows_per_group,
                min: Some((g * per_group) as f64),
                max: Some(((g + 1) * per_group - 1) as f64),
            })
            .collect();

        // 1% of candidates confident, evenly spread: every group holds ~65 of them.
        let spread: Vec<u32> = (0..n_cand).step_by(100).collect();
        let spans = confident_row_spans(&s, &spread).unwrap();
        let read: usize = spans.iter().map(|x| x.1).sum();
        let total = rows_per_group * n_groups;
        assert_eq!(
            (spans.len(), read),
            (1, total),
            "on a production-shaped table the pruning reads the whole file in one span"
        );

        // Where it does pay: a confident set clustered in a slice of the id range, which is
        // the shape a per-group search or a re-run over a subset produces.
        let clustered: Vec<u32> = (0..per_group as u32 * 3).collect();
        let read: usize = confident_row_spans(&s, &clustered)
            .unwrap()
            .iter()
            .map(|x| x.1)
            .sum();
        assert_eq!(
            read,
            3 * rows_per_group,
            "a clustered set reads 3 of 200 groups"
        );
    }

    #[test]
    fn row_group_pruning_refuses_statistics_that_cannot_describe_a_u32_column() {
        // A writer that ordered an unsigned column's min/max under SIGNED comparison hands
        // back a negative bound. Skipping a group on a range like that would drop a
        // confident anchor and shift the global half-widths that bound every candidate, so
        // the pruning must refuse and read the whole table instead.
        let signed = vec![
            mumdia_io::table::RowGroupStats {
                rows: 10,
                min: Some(-4.0),
                max: Some(9.0),
            },
            mumdia_io::table::RowGroupStats {
                rows: 10,
                min: Some(10.0),
                max: Some(19.0),
            },
        ];
        assert!(confident_row_spans(&signed, &[15]).is_none());
        // An inverted range is equally unusable.
        let inverted = stats(&[(10, 9, 0)]);
        assert!(confident_row_spans(&inverted, &[5]).is_none());
        // And the same groups with honest bounds do prune.
        let ok = stats(&[(10, 0, 9), (10, 10, 19)]);
        assert_eq!(confident_row_spans(&ok, &[15]).unwrap(), vec![(10, 10)]);
    }

    // --- the whole chunked pass, end to end ---

    /// Crafted psms_extracted + chromatograms for the Extended feature set: shared window
    /// grids, a predicted-but-unobserved fragment, candidates with no chromatogram rows at
    /// all, MS1 isotope XICs, b/y and multiply-charged fragment names, and several charge
    /// states per peptidoform.
    fn craft_extended_inputs(dir: &std::path::Path, n_cand: u32) -> (String, String) {
        let psms = dir.join("psms_ext.parquet").to_string_lossy().to_string();
        let chrom = dir.join("chrom_ext.parquet").to_string_lossy().to_string();
        let (mut cid, mut base): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
        let (mut apex_rt, mut cal, mut mz): (Vec<f64>, Vec<f64>, Vec<f64>) =
            (Vec::new(), Vec::new(), Vec::new());
        let mut apex_int: Vec<f32> = Vec::new();
        let (mut nmatch, mut corun, mut z): (Vec<i32>, Vec<i32>, Vec<i32>) =
            (Vec::new(), Vec::new(), Vec::new());
        let (mut label, mut pform, mut prot): (Vec<String>, Vec<String>, Vec<String>) =
            (Vec::new(), Vec::new(), Vec::new());
        let (mut ccid, mut cname): (Vec<u32>, Vec<String>) = (Vec::new(), Vec::new());
        let (mut cfmz, mut cobsmz): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let mut cpint: Vec<f32> = Vec::new();
        let (mut crt, mut cint): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
        for c in 0..n_cand {
            let apex = 100.0 + c as f64 * 7.0;
            cid.push(c);
            base.push(c / 3);
            apex_rt.push(apex);
            cal.push(apex - 1.5);
            mz.push(500.0 + c as f64);
            apex_int.push(1000.0 + c as f32);
            nmatch.push(3);
            corun.push(2);
            z.push(2 + (c % 3) as i32);
            label.push(if c % 3 == 0 { "decoy" } else { "target" }.to_string());
            // Three charge states share a peptidoform, so the cross-charge reduction has
            // something to group.
            pform.push(format!("PEPTIDEK{}", c / 3));
            prot.push(format!("P{c};Q{c}"));
            if c % 7 == 6 {
                continue; // a PSM row with no chromatogram rows
            }
            let npts = 5 + (c % 6) as usize;
            let grid: Vec<f32> = (0..npts).map(|k| (apex - 4.0 + k as f64) as f32).collect();
            let nfrag = 2 + (c % 4);
            for f in 0..nfrag {
                ccid.push(c);
                cname.push(if f % 2 == 0 {
                    format!("y{}", f + 1)
                } else {
                    format!("b{}^2", f + 1)
                });
                cfmz.push(200.0 + f as f64 * 30.0);
                cobsmz.push(200.0 + f as f64 * 30.0 + 0.0001 * (c % 5) as f64);
                cpint.push(1.0 / (f + 1) as f32);
                if f == nfrag - 1 && c % 4 == 0 {
                    crt.push(Vec::new()); // predicted but never observed
                    cint.push(Vec::new());
                } else {
                    crt.push(grid.clone());
                    cint.push(
                        (0..npts)
                            .map(|k| {
                                let x = k as f32 - (npts as f32) / 2.0;
                                ((-x * x / 3.0).exp() * (100.0 - f as f32 * 9.0)).max(0.0)
                            })
                            .collect(),
                    );
                }
            }
            for (iso, nm) in ["ms1_mono", "ms1_iso1", "ms1_iso2"].iter().enumerate() {
                ccid.push(c);
                cname.push(nm.to_string());
                cfmz.push(500.0 + c as f64 + iso as f64 * 0.5);
                cobsmz.push(500.0 + c as f64 + iso as f64 * 0.5);
                cpint.push(0.0);
                crt.push(grid.clone());
                cint.push((0..npts).map(|k| (k as f32 + 1.0) * 50.0).collect());
            }
        }
        let n = cid.len();
        mumdia_io::table::write_table(
            &psms,
            vec![
                Col::U32("candidate_id".into(), cid),
                Col::F64("apex_rt".into(), apex_rt),
                Col::F32("apex_intensity".into(), apex_int),
                Col::I32("n_matched_fragments".into(), nmatch),
                Col::I32("coelution_run".into(), corun),
                Col::F64("rt_pred_cal".into(), cal),
                Col::I32("charge".into(), z),
                Col::Str("label".into(), label),
                Col::U32("base_peptide_id".into(), base),
                Col::Str("peptidoform".into(), pform),
                Col::Str("protein".into(), prot),
                Col::F64("precursor_mz".into(), mz),
                Col::OptF64("ms1_mono".into(), vec![Some(900.0); n]),
                Col::OptF64("ms1_iso1".into(), vec![Some(450.0); n]),
                Col::OptF64("ms1_iso2".into(), vec![Some(120.0); n]),
                Col::OptF64("ms1_isom1".into(), vec![Some(30.0); n]),
            ],
        )
        .unwrap();
        mumdia_io::table::write_table(
            &chrom,
            vec![
                Col::U32("candidate_id".into(), ccid),
                Col::Str("frag_name".into(), cname),
                Col::F64("frag_mz".into(), cfmz),
                Col::F64("frag_obs_mz".into(), cobsmz),
                Col::F32("predicted_intensity".into(), cpint),
                Col::ListF32("rt".into(), crt),
                Col::ListF32("intensity".into(), cint),
            ],
        )
        .unwrap();
        (psms, chrom)
    }

    fn fnv(h: &mut u64, b: &[u8]) {
        for &x in b {
            *h ^= x as u64;
            *h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }

    /// Order-sensitive digest of a features table: every column name in file order and
    /// then its values as raw bit patterns. Writing a value into a different column than
    /// before changes it, which is the property
    /// `extended_features_match_the_pre_permutation_build` needs.
    fn features_digest(path: &str) -> (usize, usize, u64) {
        let t = mumdia_io::table::Table::read(path).unwrap();
        let names = t.column_names();
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for name in &names {
            fnv(&mut h, name.as_bytes());
            if let Ok(v) = t.f64(name) {
                fnv(&mut h, b"f64");
                for x in &v {
                    fnv(&mut h, &x.to_bits().to_le_bytes());
                }
            } else if let Ok(v) = t.u32(name) {
                fnv(&mut h, b"u32");
                for x in &v {
                    fnv(&mut h, &x.to_le_bytes());
                }
            } else if let Ok(v) = t.i32(name) {
                fnv(&mut h, b"i32");
                for x in &v {
                    fnv(&mut h, &x.to_le_bytes());
                }
            } else if let Ok(v) = t.str(name) {
                fnv(&mut h, b"str");
                for s in &v {
                    fnv(&mut h, s.as_bytes());
                }
            }
        }
        (t.nrows, names.len(), h)
    }

    #[test]
    fn extended_features_match_the_pre_permutation_build() {
        // A GOLDEN, captured by running THIS fixture against commit 1ac1b44^ -- the last
        // commit whose serial assembly still keyed each value by a feature-name string
        // literal (`m.set("frag_corr", r, ff.frag_corr)`), before `ColIx` replaced the
        // names with a precomputed column permutation.
        //
        // This is the only test that can see a transposed pair. `colix_covers_every_active
        // _column` proves the permutation is total and
        // `colix_resolves_each_field_to_the_column_its_own_name_denotes` proves each field
        // resolves to its own name, but neither can see `m.set(ix.frag_cosine, r,
        // ff.frag_corr)` written alongside `m.set(ix.frag_corr, r, ff.frag_cosine)`: the
        // swap is still a permutation and each field still resolves correctly. It also
        // catches what chunk invariance cannot -- a regression that shifts every row
        // identically -- and the equality evidence the change was promoted on was a
        // temporary capture module the commit deleted, so nothing in the repository
        // reproduced it until now.
        //
        // REGENERATING: these two constants are a claim about the values of ANOTHER commit.
        // If a deliberate change to the feature arithmetic breaks them, re-derive them by
        // running this fixture at 1ac1b44^ (or at the last commit whose values are trusted)
        // and changing them in one commit that says which values moved and why. Taking the
        // new numbers from the current build makes the test pin nothing.
        //
        // THE DIGEST IS PER PLATFORM, and that is a finding rather than a nuisance: the
        // same fixture gives 0x4e43..a1cc on Windows and 0x6137..c42a on Linux, stably on
        // both (two runs each). The extended battery reaches `ln`, `exp` and `powf`, which
        // are the platform's libm and agree only to within the last bit, so the engine's
        // f64 feature VALUES are not bit-identical across operating systems. The PIN is: it
        // formats to fixed decimals, which is why the same PIN hash holds on both. A
        // platform with no entry here checks the shape and the PIN and says it has no
        // digest, rather than failing on a constant captured somewhere else.
        const GOLDEN_DIGEST: Option<u64> = if cfg!(target_os = "windows") {
            Some(0x4e43_7960_2b2a_a1cc)
        } else if cfg!(target_os = "linux") {
            Some(0x6137_b855_a2da_c42a)
        } else {
            None
        };
        const GOLDEN_PIN: &str = "806a126473eafce7a1567ac71b5253e8605af0062864a02f2ab5f0b77e851817";

        let dir = std::env::temp_dir().join("mumdia_features_golden");
        std::fs::create_dir_all(&dir).unwrap();
        let (psms, chrom) = craft_extended_inputs(&dir, 61);
        // Both chunk sizes, because the golden was captured at both and they agreed: the
        // pin therefore covers the chunked path as well as the single-chunk one.
        for (tag, chunk) in [("one", 1usize << 20), ("many", 1usize)] {
            let out = dir
                .join(format!("golden_{tag}.parquet"))
                .to_string_lossy()
                .to_string();
            let pin = dir
                .join(format!("golden_{tag}.pin"))
                .to_string_lossy()
                .to_string();
            let mut cfg = FeaturesConfig {
                set: FeatureSet::Extended,
                emit_pin: true,
                bound_from_confident: false,
                ..Default::default()
            };
            cfg.ms1_precursor_features = true;
            run_with_chunk_rows(
                FeaturesParams {
                    psms: &psms,
                    chromatograms: &chrom,
                    seed: None,
                    out: &out,
                    out_pin: &pin,
                    cfg: &cfg,
                    config_hash: "test",
                },
                chunk,
            )
            .unwrap();
            let (rows, ncols, digest) = features_digest(&out);
            assert_eq!((rows, ncols), (61, 398), "chunk {tag}: table shape moved");
            match GOLDEN_DIGEST {
                Some(golden) => assert_eq!(
                    digest, golden,
                    "chunk {tag}: a feature value or its column differs from the                      name-keyed assembly at 1ac1b44^ (this platform computes                      0x{digest:016x})"
                ),
                None => eprintln!(
                    "no feature digest is recorded for this platform; chunk {tag} computes                      0x{digest:016x}. Add it above once it has been checked against a                      platform that has one."
                ),
            }
            assert_eq!(
                mumdia_io::hash::blake3_file(&pin).unwrap(),
                GOLDEN_PIN,
                "chunk {tag}: the PIN bytes differ from the name-keyed assembly"
            );
        }
    }

    /// Deterministic pseudo-random stream for the production-shaped fixture below. A
    /// 64-bit LCG (Knuth's MMIX constants), high bits only: no dependency, and the same
    /// sequence on every platform, which a golden digest needs.
    struct Lcg(u64);

    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }

        /// Uniform in `[0, 1)`.
        fn unit(&mut self) -> f64 {
            self.next_u32() as f64 / (1u64 << 31) as f64
        }

        /// Uniform in `0..n`.
        fn below(&mut self, n: u32) -> u32 {
            self.next_u32() % n.max(1)
        }
    }

    /// A PRODUCTION-SHAPED Extended fixture for the per-PSM kernels: mostly 12 fragments
    /// on ~150-240-point shared grids (the HYE shape), with every degenerate case the
    /// kernels branch on mixed in -- 0 to 11 fragments, 1- to 12-point grids, predicted but
    /// unobserved fragments, all-zero traces, quantised intensities (ties), zero and
    /// negative predicted intensities, rows on a shifted grid (the union alignment), a row
    /// whose intensity vector is longer than its axis, MS1 isotope XICs on the fragment
    /// grid, on a different grid, incomplete and absent, and apexes off the axis.
    ///
    /// `craft_extended_inputs` has at most five fragments and ten points, so it never
    /// reaches the body of an 11-lag cross-correlation nor a pair matrix of the size the
    /// kernels are tuned for; this one does, which is what the kernel golden below needs.
    fn craft_kernel_inputs(dir: &std::path::Path, n_cand: u32) -> (String, String) {
        let psms = dir
            .join("psms_kernel.parquet")
            .to_string_lossy()
            .to_string();
        let chrom = dir
            .join("chrom_kernel.parquet")
            .to_string_lossy()
            .to_string();
        let mut rng = Lcg(0x5eed_1234_abcd_0001);
        let (mut cid, mut base): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
        let (mut apex_rt, mut cal, mut mz): (Vec<f64>, Vec<f64>, Vec<f64>) =
            (Vec::new(), Vec::new(), Vec::new());
        let mut apex_int: Vec<f32> = Vec::new();
        let (mut nmatch, mut npred, mut corun, mut z): (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut label, mut pform, mut prot): (Vec<String>, Vec<String>, Vec<String>) =
            (Vec::new(), Vec::new(), Vec::new());
        let mut ms1: [Vec<Option<f64>>; 4] = Default::default();
        let (mut ccid, mut cname): (Vec<u32>, Vec<String>) = (Vec::new(), Vec::new());
        let (mut cfmz, mut cobsmz): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let mut cpint: Vec<f32> = Vec::new();
        let (mut crt, mut cint): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
        // One Gaussian elution peak on `n` points, centred at `centre`, with dropouts,
        // noise, an optional interferent and optional quantisation.
        let peak = |rng: &mut Lcg, n: usize, centre: f64| -> Vec<f32> {
            let height = 10f64.powf(1.0 + 3.0 * rng.unit());
            let width = 1.5 + 4.0 * rng.unit();
            let interferent = if rng.below(5) == 0 {
                Some((
                    rng.below(n.max(1) as u32) as f64,
                    height * 2.0 * rng.unit(),
                    1.0 + 3.0 * rng.unit(),
                ))
            } else {
                None
            };
            let quantise = rng.below(10) == 0;
            (0..n)
                .map(|i| {
                    let x = (i as f64 - centre) / width;
                    let mut v = height * (-0.5 * x * x).exp();
                    if let Some((c2, h2, w2)) = interferent {
                        let x2 = (i as f64 - c2) / w2;
                        v += h2 * (-0.5 * x2 * x2).exp();
                    }
                    if rng.below(4) == 0 {
                        return 0.0;
                    }
                    v += height * 0.05 * rng.unit();
                    if quantise {
                        let q = height / 8.0;
                        v = (v / q).round() * q;
                    }
                    v as f32
                })
                .collect()
        };
        for c in 0..n_cand {
            let roll = rng.unit();
            let k: u32 = if roll < 0.55 {
                12
            } else if roll < 0.6 {
                0
            } else {
                1 + rng.below(11)
            };
            let npts: usize = if rng.below(10) == 0 {
                1 + rng.below(12) as usize
            } else {
                150 + rng.below(91) as usize
            };
            let dt = 0.5 + 2.5 * rng.unit();
            let start = 200.0 + c as f64 * 3.0;
            let grid: Vec<f32> = (0..npts).map(|i| (start + i as f64 * dt) as f32).collect();
            let centre = (npts / 2) as f64 + rng.below(9) as f64 - 4.0;
            let centre_i = (centre.max(0.0) as usize).min(npts - 1);
            let apex = if rng.below(20) == 0 {
                grid[0] as f64 - 10.0 // apex off the left end of the axis
            } else {
                grid[centre_i] as f64 + (rng.unit() - 0.5) * 0.6 * dt
            };
            cid.push(c);
            base.push(c / 3);
            apex_rt.push(apex);
            cal.push(apex + (rng.unit() - 0.5) * 20.0);
            mz.push(400.0 + 600.0 * rng.unit());
            apex_int.push((10f64.powf(2.0 + 3.0 * rng.unit())) as f32);
            npred.push(k as i32);
            nmatch.push(rng.below(k + 1) as i32);
            corun.push(rng.below(k + 1) as i32);
            z.push(2 + (c % 3) as i32);
            label.push(if c % 3 == 0 { "decoy" } else { "target" }.to_string());
            pform.push(format!("PEPTIDEK{}", c / 3));
            prot.push(format!("P{c};Q{}", c / 2));
            for (slot, v) in ms1.iter_mut().enumerate() {
                v.push(if rng.below(8) == 0 {
                    None
                } else {
                    Some(10f64.powf(2.0 + 2.0 * rng.unit()) / (slot + 1) as f64)
                });
            }
            for f in 0..k {
                let series = if rng.below(2) == 0 { 'y' } else { 'b' };
                let ordinal = 1 + rng.below(14);
                let name = if rng.below(5) == 0 {
                    format!("{series}{ordinal}^2")
                } else {
                    format!("{series}{ordinal}")
                };
                let fmz = 150.0 + 1200.0 * rng.unit();
                ccid.push(c);
                cname.push(name);
                cfmz.push(fmz);
                cobsmz.push(fmz * (1.0 + (rng.unit() - 0.5) * 20e-6));
                cpint.push(match rng.below(20) {
                    0 => 0.0,
                    1 => -0.05,
                    2 => 0.5,
                    _ => rng.unit() as f32,
                });
                let jitter = rng.below(5) as f64 - 2.0;
                // The first fragment always carries a trace: a candidate whose every row is
                // empty has an empty union axis, which extract never emits and which the
                // peak-window slice in `fragment_features` does not accept.
                let mut case = rng.below(20);
                if f == 0 && case == 0 {
                    case = 4;
                }
                match case {
                    0 => {
                        crt.push(Vec::new()); // predicted but never observed
                        cint.push(Vec::new());
                    }
                    1 => {
                        crt.push(grid.clone());
                        cint.push(vec![0.0; npts]);
                    }
                    2 => {
                        // A row on a shifted grid: no shared axis, so the union build runs.
                        crt.push(grid.iter().map(|&t| t + (0.25 * dt) as f32).collect());
                        cint.push(peak(&mut rng, npts, centre + jitter));
                    }
                    3 => {
                        // More values than axis points: the union build truncates.
                        crt.push(grid.clone());
                        cint.push(peak(&mut rng, npts + 2, centre + jitter));
                    }
                    _ => {
                        crt.push(grid.clone());
                        cint.push(peak(&mut rng, npts, centre + jitter));
                    }
                }
            }
            if k == 0 {
                continue; // a PSM row with no chromatogram rows at all
            }
            let ms1_case = rng.below(10);
            let isotopes: &[&str] = match ms1_case {
                8 => &["ms1_mono", "ms1_iso1"],
                9 => &[],
                _ => &["ms1_mono", "ms1_iso1", "ms1_iso2"],
            };
            for (iso, nm) in isotopes.iter().enumerate() {
                ccid.push(c);
                cname.push(nm.to_string());
                cfmz.push(500.0 + iso as f64 * 0.5);
                cobsmz.push(500.0 + iso as f64 * 0.5);
                cpint.push(0.0);
                if ms1_case == 7 {
                    // Every second fragment scan: an MS1 grid that is not the fragments'.
                    let sub: Vec<f32> = grid.iter().step_by(2).copied().collect();
                    let n = sub.len();
                    crt.push(sub);
                    cint.push(peak(&mut rng, n, centre / 2.0));
                } else {
                    crt.push(grid.clone());
                    cint.push(peak(&mut rng, npts, centre));
                }
            }
        }
        let [m_mono, m_i1, m_i2, m_m1] = ms1;
        mumdia_io::table::write_table(
            &psms,
            vec![
                Col::U32("candidate_id".into(), cid),
                Col::F64("apex_rt".into(), apex_rt),
                Col::F32("apex_intensity".into(), apex_int),
                Col::I32("n_matched_fragments".into(), nmatch),
                Col::I32("n_predicted_fragments".into(), npred),
                Col::I32("coelution_run".into(), corun),
                Col::F64("rt_pred_cal".into(), cal),
                Col::I32("charge".into(), z),
                Col::Str("label".into(), label),
                Col::U32("base_peptide_id".into(), base),
                Col::Str("peptidoform".into(), pform),
                Col::Str("protein".into(), prot),
                Col::F64("precursor_mz".into(), mz),
                Col::OptF64("ms1_mono".into(), m_mono),
                Col::OptF64("ms1_iso1".into(), m_i1),
                Col::OptF64("ms1_iso2".into(), m_i2),
                Col::OptF64("ms1_isom1".into(), m_m1),
            ],
        )
        .unwrap();
        // Extract's row-group size, so a benchmark on a large instance of this fixture
        // decodes the row groups a production table has.
        let mut w = TableWriter::new(&chrom).with_row_group_rows(1 << 16);
        w.write_cols(vec![
            Col::U32("candidate_id".into(), ccid),
            Col::Str("frag_name".into(), cname),
            Col::F64("frag_mz".into(), cfmz),
            Col::F64("frag_obs_mz".into(), cobsmz),
            Col::F32("predicted_intensity".into(), cpint),
            Col::ListF32("rt".into(), crt),
            Col::ListF32("intensity".into(), cint),
        ])
        .unwrap();
        w.close().unwrap();
        (psms, chrom)
    }

    #[test]
    #[ignore = "benchmark fixture: set MUMDIA_BENCH_OUT (and MUMDIA_BENCH_CANDS); writes GBs"]
    fn write_kernel_bench_fixture() {
        // Writes a large instance of `craft_kernel_inputs` for an A/B of two `mumdia
        // features` binaries on HYE-shaped traces (mostly 12 fragments on 150-240-point
        // grids), which the local real runs (a 15-minute Astral gradient, ~8 points per
        // trace) do not have. Run with `--ignored --nocapture`, then point `mumdia
        // features --psms-extracted <out>/psms_kernel.parquet --chromatograms
        // <out>/chrom_kernel.parquet` at it with the Extended set.
        let Ok(out) = std::env::var("MUMDIA_BENCH_OUT") else {
            eprintln!("MUMDIA_BENCH_OUT is not set; nothing written");
            return;
        };
        let n: u32 = std::env::var("MUMDIA_BENCH_CANDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(40_000);
        let dir = std::path::PathBuf::from(out);
        std::fs::create_dir_all(&dir).unwrap();
        let (psms, chrom) = craft_kernel_inputs(&dir, n);
        eprintln!("wrote {psms} and {chrom} ({n} candidates)");
    }

    #[test]
    fn production_shaped_features_match_the_pre_kernel_build() {
        // A GOLDEN over the production-shaped fixture above, captured at main 6887c41
        // BEFORE the per-PSM kernel work (the shared pair statistics, the MS1 identity
        // path, the lag-parallel cross-correlation, the centred Pearson, the order
        // statistics by selection, the power iteration and the cached Spearman ranks).
        // Every one of those changes claims bit-identical feature values; this is the test
        // that holds them to it on inputs large enough to reach the code they changed.
        //
        // Three configurations, because the kernels branch on them: per-candidate peak
        // bounds, the global half-widths a confident set would give, and
        // `bound_features = false` (where `fragment_features` scores the whole window and
        // must not share the Extended path's peak statistics). Each runs in small chunks,
        // so the multi-chunk loader is on the path too.
        //
        // PER PLATFORM for the reason `extended_features_match_the_pre_permutation_build`
        // gives (the platform libm). REGENERATING: re-derive at the last commit whose
        // values are trusted, never from the build under test.
        const GOLDEN_DIGEST: Option<u64> = if cfg!(target_os = "windows") {
            Some(0x7eaa_3aa3_e1bc_0502)
        } else {
            None
        };
        let dir = std::env::temp_dir().join("mumdia_features_kernel_golden");
        std::fs::create_dir_all(&dir).unwrap();
        let (psms, chrom) = craft_kernel_inputs(&dir, 400);
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let arms: [(&str, bool, BoundsSource); 3] = [
            ("percand", true, BoundsSource::Learn),
            ("global", true, BoundsSource::Given(Some((6.0, 9.0)))),
            ("unbounded", false, BoundsSource::Learn),
        ];
        for (tag, bound_features, bounds) in arms {
            let out = dir
                .join(format!("kernel_{tag}.parquet"))
                .to_string_lossy()
                .to_string();
            let mut cfg = FeaturesConfig {
                set: FeatureSet::Extended,
                emit_pin: false,
                bound_from_confident: false,
                bound_features,
                ..Default::default()
            };
            cfg.ms1_precursor_features = true;
            run_chunked(
                FeaturesParams {
                    psms: &psms,
                    chromatograms: &chrom,
                    seed: None,
                    out: &out,
                    out_pin: "",
                    cfg: &cfg,
                    config_hash: "test",
                },
                300,
                usize::MAX,
                PinFinish::Normal,
                bounds,
            )
            .unwrap();
            let (rows, ncols, digest) = features_digest(&out);
            assert_eq!((rows, ncols), (400, 398), "arm {tag}: table shape moved");
            eprintln!("arm {tag}: 0x{digest:016x}");
            fnv(&mut h, &digest.to_le_bytes());
        }
        match GOLDEN_DIGEST {
            Some(golden) => assert_eq!(
                h, golden,
                "a feature value differs from the pre-kernel build (this platform computes \
                 0x{h:016x})"
            ),
            None => {
                eprintln!("no kernel digest is recorded for this platform; it computes 0x{h:016x}")
            }
        }
    }

    #[test]
    fn extended_features_are_chunk_invariant() {
        // The `features_chunking_is_value_preserving` integration test runs the DEFAULT
        // (Minimal) set with no seed, so it never reaches the extended battery, the
        // per-PSM extended buffer, the extended half of the column permutation, or the
        // trace alignment's fast path. This one does, over the full chunked pass:
        // loader thread, parallel per-PSM map, serial assembly, writer thread.
        let dir = std::env::temp_dir().join("mumdia_features_extended");
        std::fs::create_dir_all(&dir).unwrap();
        let (psms, chrom) = craft_extended_inputs(&dir, 61);
        let run = |tag: &str, chunk_rows: usize, max_psm_rows: usize| -> String {
            let out = dir
                .join(format!("features_{tag}.parquet"))
                .to_string_lossy()
                .to_string();
            let pin = dir
                .join(format!("features_{tag}.pin"))
                .to_string_lossy()
                .to_string();
            let mut cfg = FeaturesConfig {
                set: FeatureSet::Extended,
                emit_pin: true,
                // The confident-bounds pass depends on the chunk size by construction
                // (it bounds a candidate from each side of a chunk boundary separately),
                // so it is off here; `pruned_confident_bounds_equal_the_whole_table_scan`
                // covers that pass instead.
                bound_from_confident: false,
                ..Default::default()
            };
            cfg.ms1_precursor_features = true;
            run_with_chunk_limits(
                FeaturesParams {
                    psms: &psms,
                    chromatograms: &chrom,
                    seed: None,
                    out: &out,
                    out_pin: &pin,
                    cfg: &cfg,
                    config_hash: "test",
                },
                chunk_rows,
                max_psm_rows,
            )
            .unwrap();
            out
        };
        let one = run("one", 1 << 20, usize::MAX);
        let many = run("many", 1, usize::MAX);
        // And chunks closed by the PSM-row limit rather than the chromatogram one: a
        // different set of boundaries again, over the same candidates.
        let psm_capped = run("psmcap", 1 << 20, 3);

        // Values only, and deliberately so: 61 rows is one row group, so these three files
        // happen to be byte-identical as well, but that is an artefact of the fixture's
        // size. `moving_a_chunk_boundary_moves_parquet_bytes_above_one_row_group` measures
        // what happens at a production size.
        let a = mumdia_io::table::Table::read(&one).unwrap();
        let b = mumdia_io::table::Table::read(&many).unwrap();
        let c = mumdia_io::table::Table::read(&psm_capped).unwrap();
        assert_eq!(a.nrows, 61);
        assert_eq!(a.nrows, c.nrows);
        assert_eq!(a.nrows, b.nrows);
        assert_eq!(a.column_names(), b.column_names());
        assert_eq!(
            a.column_names().len(),
            active_features(FeatureSet::Extended).len() + NON_FEATURE_COLUMNS.len() - 3,
            "metadata columns plus every active feature"
        );
        // Every f64 column bit for bit, so a feature that differs in the last ulp fails.
        let mut checked = 0usize;
        for name in a.column_names() {
            if let (Ok(x), Ok(y), Ok(z)) = (a.f64(&name), b.f64(&name), c.f64(&name)) {
                let bits = |v: &[f64]| v.iter().map(|q| q.to_bits()).collect::<Vec<u64>>();
                assert_eq!(
                    bits(&x),
                    bits(&y),
                    "column '{name}' differs between chunk sizes"
                );
                assert_eq!(
                    bits(&x),
                    bits(&z),
                    "column '{name}' differs under the PSM cap"
                );
                checked += 1;
            }
        }
        assert!(checked > 300, "only {checked} f64 columns compared");
        // The cross-charge reduction is a whole-run quantity: three charge states per
        // peptidoform, so it must read 3 for the peptidoforms that have all three.
        let n_charge = a.f64("n_charge_states").unwrap();
        assert!(
            n_charge.contains(&3.0),
            "the charge-state count never reached 3"
        );
        assert!(n_charge.iter().all(|&v| (1.0..=3.0).contains(&v)));
    }

    #[test]
    fn the_loader_count_moves_no_byte_of_the_features_table() {
        // `features.chrom_loaders` decodes chunks on several threads from their own row
        // spans. The claim is stronger than equal values: the chunk boundaries are fixed by
        // the plan and the computation takes the chunks in table order, so every
        // `write_cols` call receives the same columns and the file is byte-identical. Two
        // chunkings: chromatogram-row chunks, and one PSM per chunk, which makes the
        // zero-row chunks of the PSMs that have no chromatogram rows (`load_chunk` returns
        // an empty chunk for those without opening a span).
        let dir = std::env::temp_dir().join("mumdia_features_loaders");
        std::fs::create_dir_all(&dir).unwrap();
        let (psms, chrom) = craft_kernel_inputs(&dir, 120);
        for (tag, chunk_rows, max_psm_rows) in [("rows", 150usize, usize::MAX), ("psm", 1 << 20, 1)]
        {
            let mut hashes: Vec<(usize, String)> = Vec::new();
            for loaders in [1usize, 2, 3, 7] {
                let out = dir
                    .join(format!("features_{tag}_{loaders}.parquet"))
                    .to_string_lossy()
                    .to_string();
                let cfg = FeaturesConfig {
                    set: FeatureSet::Extended,
                    bound_from_confident: false,
                    chrom_loaders: loaders,
                    ..Default::default()
                };
                run_with_chunk_limits(
                    FeaturesParams {
                        psms: &psms,
                        chromatograms: &chrom,
                        seed: None,
                        out: &out,
                        out_pin: "",
                        cfg: &cfg,
                        config_hash: "test",
                    },
                    chunk_rows,
                    max_psm_rows,
                )
                .unwrap();
                hashes.push((loaders, mumdia_io::hash::blake3_file(&out).unwrap()));
            }
            for (loaders, h) in &hashes[1..] {
                assert_eq!(
                    h, &hashes[0].1,
                    "chunking {tag}: {loaders} loaders wrote different bytes than one"
                );
            }
        }
    }

    #[test]
    fn a_failed_chunk_pass_neither_hangs_nor_publishes_a_partial_table() {
        // Both halves of this are about the stage's write and load having moved onto
        // their own threads: an error must still come back (the loader must not be left
        // blocked on a rendezvous send that nobody will receive), and the writer must not
        // publish what it has over the previous good artifact.
        let dir = std::env::temp_dir().join("mumdia_features_failure");
        std::fs::create_dir_all(&dir).unwrap();
        let (psms, chrom) = craft_extended_inputs(&dir, 13);
        let out = dir
            .join("features_fail.parquet")
            .to_string_lossy()
            .to_string();
        let pin = dir.join("features_fail.pin").to_string_lossy().to_string();
        let cfg = FeaturesConfig {
            set: FeatureSet::Rich,
            bound_from_confident: false,
            ..Default::default()
        };
        let go = |chrom_path: &str| {
            run_with_chunk_rows(
                FeaturesParams {
                    psms: &psms,
                    chromatograms: chrom_path,
                    seed: None,
                    out: &out,
                    out_pin: &pin,
                    cfg: &cfg,
                    config_hash: "test",
                },
                4,
            )
        };
        go(&chrom).expect("the good run must succeed");
        let good = mumdia_io::hash::blake3_file(&out).unwrap();

        // The same candidates, but the chromatogram table is missing a column the stream
        // requires, so the loader thread fails after the plan has been made.
        let broken = dir
            .join("chrom_broken.parquet")
            .to_string_lossy()
            .to_string();
        let src = mumdia_io::table::Table::read(&chrom).unwrap();
        mumdia_io::table::write_table(
            &broken,
            vec![
                Col::U32("candidate_id".into(), src.u32("candidate_id").unwrap()),
                Col::Str("frag_name".into(), src.str("frag_name").unwrap()),
                Col::F64("frag_mz".into(), src.f64("frag_mz").unwrap()),
                Col::ListF32("rt".into(), src.list_f32("rt").unwrap()),
                Col::ListF32("intensity".into(), src.list_f32("intensity").unwrap()),
            ],
        )
        .unwrap();

        // On a worker thread with a deadline, so a regression that reintroduces the hang
        // fails this test instead of wedging the suite.
        let (tx, rx) = std::sync::mpsc::channel();
        let (b, p2, o2, pin2) = (broken.clone(), psms.clone(), out.clone(), pin.clone());
        std::thread::spawn(move || {
            let cfg = FeaturesConfig {
                set: FeatureSet::Rich,
                bound_from_confident: false,
                ..Default::default()
            };
            let r = run_with_chunk_rows(
                FeaturesParams {
                    psms: &p2,
                    chromatograms: &b,
                    seed: None,
                    out: &o2,
                    out_pin: &pin2,
                    cfg: &cfg,
                    config_hash: "test",
                },
                4,
            );
            let _ = tx.send(r.is_err());
        });
        let failed = rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("the stage must return, not hang, when the loader fails");
        assert!(
            failed,
            "a chromatogram table without predicted_intensity must be an error"
        );
        assert_eq!(
            mumdia_io::hash::blake3_file(&out).unwrap(),
            good,
            "a failed run must leave the previous features table exactly as it was"
        );
        // And no temp file left behind.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("features_fail.parquet.tmp-"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp files left behind: {leftovers:?}"
        );
    }

    #[test]
    fn a_failed_pin_finish_does_not_publish_the_features_table() {
        // The in-line write ran `pin.finish()` and THEN `writer.close()`, so a PIN that
        // failed at the flush left the previous features table exactly as it was. Moving
        // `close` onto the writer thread reversed the order silently: the table was
        // published and the stage then errored, leaving a published features table with no
        // matching PIN. This pins the OLD ordering.
        let dir = std::env::temp_dir().join("mumdia_features_pin_order");
        std::fs::create_dir_all(&dir).unwrap();
        let (psms, chrom) = craft_extended_inputs(&dir, 11);
        let out = dir
            .join("features_pin.parquet")
            .to_string_lossy()
            .to_string();
        let pin = dir.join("features_pin.pin").to_string_lossy().to_string();
        let cfg = FeaturesConfig {
            set: FeatureSet::Rich,
            emit_pin: true,
            bound_from_confident: false,
            ..Default::default()
        };
        let params = || FeaturesParams {
            psms: &psms,
            chromatograms: &chrom,
            seed: None,
            out: &out,
            out_pin: &pin,
            cfg: &cfg,
            config_hash: "test",
        };

        run_chunked(
            params(),
            4,
            CHUNK_PSM_ROWS,
            PinFinish::Normal,
            BoundsSource::Learn,
        )
        .expect("the good run must succeed");
        let good = mumdia_io::hash::blake3_file(&out).unwrap();

        let err = run_chunked(
            params(),
            4,
            CHUNK_PSM_ROWS,
            PinFinish::Fail,
            BoundsSource::Learn,
        )
        .expect_err("a PIN that cannot be flushed must fail the stage");
        assert!(
            err.to_string().contains("PIN"),
            "the PIN failure must be the reported error, not the writer's: {err}"
        );
        assert_eq!(
            mumdia_io::hash::blake3_file(&out).unwrap(),
            good,
            "a PIN failure must not publish the features table over the previous one"
        );
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("features_pin.parquet.tmp-"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp files left behind: {leftovers:?}"
        );
    }

    #[test]
    fn the_chrom_stream_reads_its_optional_column_from_the_handle_it_is_given() {
        // `ChromStream::open` used to answer "does this table carry frag_obs_mz?" by
        // opening the file a second time and parsing its whole footer
        // (`mumdia_io::table::column_names(path)`), once per stream -- and the
        // confident-bounds pass opens one stream per span. It reads the handle's schema
        // now, including on a `TableFile::span`, which carries the WHOLE file's schema.
        // Same answer either way, which is what this pins: the observed m/z must still
        // come from `frag_obs_mz` where the column exists and fall back to `frag_mz` where
        // it does not.
        let dir = std::env::temp_dir().join("mumdia_features_stream_schema");
        std::fs::create_dir_all(&dir).unwrap();
        let with = dir.join("with_obs.parquet").to_string_lossy().to_string();
        let without = dir.join("no_obs.parquet").to_string_lossy().to_string();
        let (cid, name): (Vec<u32>, Vec<String>) = (
            (0..8u32).flat_map(|c| [c, c]).collect(),
            (0..16).map(|i| format!("y{}", i % 2 + 1)).collect(),
        );
        let fmz: Vec<f64> = (0..16).map(|i| 200.0 + i as f64).collect();
        let obs: Vec<f64> = fmz.iter().map(|v| v + 0.25).collect();
        let pint: Vec<f32> = vec![1.0; 16];
        let rt: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]; 16];
        let inten: Vec<Vec<f32>> = vec![vec![4.0, 5.0, 6.0]; 16];
        let base = |extra: bool| {
            let mut cols = vec![
                Col::U32("candidate_id".into(), cid.clone()),
                Col::Str("frag_name".into(), name.clone()),
                Col::F64("frag_mz".into(), fmz.clone()),
                Col::F32("predicted_intensity".into(), pint.clone()),
                Col::ListF32("rt".into(), rt.clone()),
                Col::ListF32("intensity".into(), inten.clone()),
            ];
            if extra {
                cols.push(Col::F64("frag_obs_mz".into(), obs.clone()));
            }
            cols
        };
        let mut w = TableWriter::new(&with).with_row_group_rows(4);
        w.write_cols(base(true)).unwrap();
        w.close().unwrap();
        let mut w = TableWriter::new(&without).with_row_group_rows(4);
        w.write_cols(base(false)).unwrap();
        w.close().unwrap();

        // Every combination of "has the column" x "whole file / span of it".
        for (path, offset) in [(&with, 0usize), (&with, 8), (&without, 0), (&without, 8)] {
            let ch = TableFile::open(path).unwrap();
            let handle = if offset == 0 {
                ch.span(0, 16).unwrap()
            } else {
                ch.span(offset, 8).unwrap()
            };
            let mut stream = ChromStream::open(&handle).unwrap();
            let mut names = NameTab::default();
            let chunk = stream.read_chunk(handle.nrows, &mut names).unwrap();
            assert_eq!(chunk.cids.len(), handle.nrows / 2);
            let rows = chunk.rows(&chunk.frag, 0, &names);
            let (m, o) = (rows[0].frag_mz, rows[0].frag_obs_mz);
            if path == &with {
                assert_eq!(o, m + 0.25, "frag_obs_mz must be read where it exists");
            } else {
                assert_eq!(
                    o, m,
                    "frag_obs_mz must fall back to frag_mz where it does not"
                );
            }
        }
    }

    #[test]
    fn the_shared_axis_fast_path_does_not_depend_on_where_the_axis_came_from() {
        // The precondition used to be `ptr::eq` plus a length check, so the fast path was
        // correct only because of an invariant of the chunk store that nothing here could
        // see. It compares bit patterns now, so two rows carrying equal-valued axes from
        // different buffers take the same path -- and, as always, must agree with the union
        // build. A `-0.0` against a `+0.0` stays on the union path, because the two builds
        // do not agree there.
        let axis_a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let axis_b: Vec<f32> = vec![1.0, 2.0, 3.0]; // equal values, different allocation
        let (ia, ib) = (vec![10.0f32, 20.0, 30.0], vec![1.0f32, 2.0, 3.0]);
        let rows = vec![row("y1", 1.0, &axis_a, &ia), row("y2", 0.5, &axis_b, &ib)];
        let fast = align_shared_axis(&rows).expect("equal axes must take the fast path");
        same(&fast, &align_union(&rows));

        let signed_zero: Vec<f32> = vec![-0.0, 2.0, 3.0];
        let plus_zero: Vec<f32> = vec![0.0, 2.0, 3.0];
        let rows = vec![
            row("y1", 1.0, &signed_zero, &ia),
            row("y2", 0.5, &plus_zero, &ib),
        ];
        assert!(
            align_shared_axis(&rows).is_none(),
            "-0.0 and +0.0 are not the same axis for the union build, so not for this one"
        );
        same(&align_traces(&rows), &align_union(&rows));
    }
}
