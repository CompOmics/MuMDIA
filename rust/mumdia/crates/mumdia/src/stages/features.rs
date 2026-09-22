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
use crate::stats::{cosine, pearson, spectral_angle};
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
                if a.len() != r.rt.len() || !std::ptr::eq(a.as_ptr(), r.rt.as_ptr()) {
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

/// Build the trace-derived fields of [`Evidence`] from a PSM's chromatogram
/// rows (scalar fields default; the caller fills them). Mirrors the alignment
/// and peak-bounding of [`fragment_features`] so the extended families see the
/// same elution peak the legacy features use.
fn build_evidence(
    rows: &[ChromRow],
    al: TraceAlign,
    ms1_rows: &[ChromRow],
    apex_rt: f64,
    frac: f64,
    grace: usize,
    global_bounds: Option<(f64, f64)>,
) -> Evidence {
    let m = rows.len();
    let mut obs_apex = Vec::with_capacity(m);
    let mut pred = Vec::with_capacity(m);
    let mut is_b = Vec::with_capacity(m);
    let mut ordinal = Vec::with_capacity(m);
    let mut frag_charge = Vec::with_capacity(m);
    let mut frag_mz = Vec::with_capacity(m);
    let mut frag_obs_mz = Vec::with_capacity(m);
    let mut mass_err_ppm = Vec::with_capacity(m);
    for r in rows {
        let mut best = 0.0f32;
        let mut bestd = f64::MAX;
        for (k, &rt) in r.rt.iter().enumerate() {
            let d = (rt as f64 - apex_rt).abs();
            if d < bestd {
                bestd = d;
                best = r.inten[k];
            }
        }
        obs_apex.push(best as f64);
        pred.push(r.pred_int as f64);
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

    let (lo_i, hi_i) = if axis_full.len() >= 3 {
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
            Some((l, r)) => global_bound_indices(&axis_full, apex_rt, ai, l, r),
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
    } else {
        (0, axis_full.len().saturating_sub(1))
    };
    let axis: Vec<f64> = axis_full[lo_i..=hi_i].iter().map(|&x| x as f64).collect();
    let traces: Vec<Vec<f64>> = traces_full
        .iter()
        .map(|t| t[lo_i..=hi_i].to_vec())
        .collect();
    let axis_full_f: Vec<f64> = axis_full.iter().map(|&x| x as f64).collect();

    let apex_idx = axis
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - apex_rt).abs().total_cmp(&(*b - apex_rt).abs()))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let np = axis.len();
    let mut ref_profile = vec![0.0f64; np];
    for (fi, tr) in traces.iter().enumerate() {
        let w = pred[fi].max(0.0);
        for k in 0..np {
            ref_profile[k] += w * tr[k];
        }
    }

    // MS1 isotope XICs [mono, +1, +2] sampled on the same grid as the fragments,
    // mapped onto axis_full then sliced to the elution peak. Present only when the
    // extract stage emitted them (grid mode + MS1 data); else empty.
    let ms1_xic: Vec<Vec<f64>> = {
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
    fn open(ch: &TableFile, path: &str) -> Result<ChromStream> {
        let has_obs_mz = mumdia_io::table::column_names(path)?
            .iter()
            .any(|c| c == "frag_obs_mz");
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
            // Trace values are appended into scratch buffers and then copied into the
            // chunk, because the axis is deduplicated against the candidate's earlier
            // rows before it is stored.
            let mut rt_buf: Vec<f32> = Vec::new();
            let mut int_buf: Vec<f32> = Vec::new();
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
                rt_buf.clear();
                int_buf.clear();
                rt.append_row(k, &mut rt_buf, "rt")?;
                inten.append_row(k, &mut int_buf, "intensity")?;
                let nm = name.value(k);
                let id = names.intern(nm);
                chunk.push_row(
                    nm.starts_with("ms1_"),
                    id,
                    fmz.value(k),
                    obsmz.map(|a| a.value(k)).unwrap_or_else(|| fmz.value(k)),
                    pint.value(k),
                    &rt_buf,
                    &int_buf,
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
fn plan_chunks(
    psm_cid: &[u32],
    ch_cid: &[u32],
    chrom_path: &str,
    chunk_rows: usize,
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
        if acc >= chunk_rows || last {
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
struct ValueMatrix {
    rows: usize,
    cols: Vec<Vec<f64>>,
}

impl ValueMatrix {
    fn new(n_cols: usize, rows: usize) -> ValueMatrix {
        ValueMatrix {
            rows,
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

    fn payload_bytes(&self) -> usize {
        self.cols.len() * self.rows * std::mem::size_of::<f64>()
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
///
/// Returns `None` when the statistics cannot be used -- a writer that recorded none, or a
/// file with no row groups -- which means "read the whole table", the previous behaviour.
fn confident_row_spans(
    stats: &[mumdia_io::table::RowGroupStats],
    confident: &[u32],
) -> Option<Vec<(usize, usize)>> {
    if stats.is_empty() || stats.iter().any(|s| s.min.is_none() || s.max.is_none()) {
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
    chrom_path: &str,
    spans: &[(usize, usize)],
    confident_rows: &HashMap<u32, Vec<usize>>,
    apex_rt: &[f64],
    cfg: &FeaturesConfig,
    chunk_rows: usize,
) -> Result<Option<(f64, f64)>> {
    let mut lefts: Vec<f64> = Vec::new();
    let mut rights: Vec<f64> = Vec::new();
    let mut names = NameTab::default();
    let keep: std::collections::HashSet<u32> = confident_rows.keys().copied().collect();
    // Chunk reading already groups rows by candidate, so this reuses it and keeps only
    // the confident candidates' rows long enough to bound their peak.
    for &(first, n_rows) in spans {
        let span = TableFile::open_rows(chrom_path, first, n_rows)?;
        let mut stream = ChromStream::open(&span, chrom_path)?;
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
    if lefts.len() >= 20 {
        let q = (cfg.bound_confident_pct / 100.0).clamp(0.0, 1.0);
        let (l, r) = (percentile(&lefts, q), percentile(&rights, q));
        info!(
            n_confident = lefts.len(),
            left_hw_s = l,
            right_hw_s = r,
            pct = cfg.bound_confident_pct,
            "features: global elution half-widths from confident set"
        );
        Ok(Some((l, r)))
    } else {
        warn!(
            n_confident = lefts.len(),
            "features: bound_from_confident set but < 20 confident anchors; \
             falling back to per-candidate boundary"
        );
        Ok(None)
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

/// [`run`] with an explicit chunk size. Only the chunk boundaries change with it: the
/// feature values, the row order and the PIN bytes do not, which is what the
/// `features_chunking_is_value_preserving` test asserts by hashing both artifacts.
pub fn run_with_chunk_rows(p: FeaturesParams, chunk_rows: usize) -> Result<u64> {
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
    let chunks = plan_chunks(&cid, &ch_cid, p.chromatograms, chunk_rows.max(1))?;
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
    let global_bounds: Option<(f64, f64)> = if p.cfg.bound_from_confident {
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
            p.chromatograms,
            &spans,
            &confident_rows,
            &apex_rt,
            p.cfg,
            chunk_rows.max(1),
        )?
    } else {
        None
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
    let (mut tot_frag_bytes, mut tot_ms1_bytes) = (0usize, 0usize);
    let (mut n_cand, mut n_frag_rows, mut n_ms1_rows) = (0usize, 0usize, 0usize);

    // The chunk decode is single-threaded parquet work and it ran in series with the
    // feature computation, which is what left a 24-thread features process at 1.6-2.8 cores
    // (measured on the 8-12-mer immunopeptidomics run: 3.7 of a 4-minute stage were the
    // load). The loader now runs on its own thread, one chunk ahead of the computation, so
    // decode and compute overlap; at most two chunks are resident. Each chunk travels with a
    // snapshot of the fragment-name table as it stood when the chunk was read, which is
    // exactly what the serial code saw at that point.
    let chunk_rows: Vec<usize> = chunks.iter().map(|c| c.chrom_rows).collect();
    let ch_ref = &ch;
    let chrom_path = p.chromatograms;
    let rows = std::thread::scope(|sc| -> Result<u64> {
        // A RENDEZVOUS channel, not a one-deep buffer. `sync_channel(1)` let the loader
        // finish a chunk, park it in the buffer and start a third, so three chunks of
        // traces were resident where the comment claimed two. At depth 0 the loader's
        // `send` blocks until this loop takes the chunk, which still overlaps decode with
        // compute fully (the loader is free again the instant the chunk is handed over)
        // and holds exactly two: the one being computed and the one being decoded.
        //
        // Both channels are created HERE rather than outside the scope so that both
        // receivers drop when this closure returns. An error return used to leave `rx`
        // alive in the enclosing frame while the scope waited to join the loader, and the
        // loader was blocked in `send` on a channel that would never be received from
        // again: the stage hung instead of reporting the error. Depth 0 makes that more
        // reachable, not less, so it is fixed rather than papered over.
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<(ChromChunk, NameTab)>>(0);
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
        let wh = sc.spawn(move || -> Result<u64> {
            let mut writer = writer;
            let mut commit = false;
            for msg in wrx {
                match msg {
                    Some(cols) => writer.write_cols(cols)?,
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
            writer.close()
        });
        sc.spawn(move || {
            let mut stream = match ChromStream::open(ch_ref, chrom_path) {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };
            let mut names = NameTab::default();
            for want in chunk_rows {
                let r = stream
                    .read_chunk(want, &mut names)
                    .map(|store| (store, names.clone()));
                let failed = r.is_err();
                if tx.send(r).is_err() || failed {
                    return;
                }
            }
        });
        for chunk in &chunks {
            let (store, names) = rx.recv().map_err(|_| {
                anyhow!(
                    "features: the chromatogram loader stopped before chunk {}..{}",
                    chunk.psm_lo,
                    chunk.psm_hi
                )
            })??;
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
                // One alignment per PSM, read by `fragment_features` and then moved
                // into the Evidence: it used to be rebuilt inside each of them.
                let al = align_traces(&rows);
                let ff = fragment_features(
                    &rows,
                    &al,
                    apex_rt[i],
                    p.cfg.coelution_corr_threshold,
                    p.cfg.bound_features,
                    p.cfg.bound_peak_fraction,
                    p.cfg.bound_peak_grace,
                    global_bounds,
                );
                if !ext.is_empty() {
                    let ms1_rows = ci
                        .map(|c| store.rows(&store.ms1, c, &names))
                        .unwrap_or_default();
                    let mut ev = build_evidence(
                        &rows,
                        al,
                        &ms1_rows,
                        apex_rt[i],
                        p.cfg.bound_peak_fraction,
                        p.cfg.bound_peak_grace,
                        global_bounds,
                    );
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
            drop(ext_vals);
            max_matrix_bytes = max_matrix_bytes.max(m.payload_bytes());

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
            // The parquet encode is single-threaded and ran here, between one chunk's
            // computation and the next chunk's: the writer thread below takes it off the
            // critical path, the way the loader thread took the decode off it. The
            // channel is a rendezvous, so at most one chunk's columns wait behind the
            // one being written.
            if wtx.send(Some(cols)).is_err() {
                // The writer failed; its error is the real one, so surface that.
                break;
            }
        }
        // Commit only after every chunk went out; a `break` above leaves this send
        // failing, and the writer's own error is what `join` then returns.
        let _ = wtx.send(None);
        drop(wtx);
        wh.join()
            .map_err(|_| anyhow!("features: the parquet writer thread panicked"))?
    })?;
    if let Some(w) = pin {
        w.finish()?;
    }

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
    // Two matrices can be alive at once and the report used to name only one, so it
    // understated the peak by exactly 2x. It did so for the wrong reason as well: the
    // second copy was `column(c).to_vec()` into the writer's columns, pure waste. The
    // columns are moved now, and the second matrix that remains is the next chunk's,
    // built while the writer thread encodes this one -- overlap, not a copy.
    crate::memlog::report(
        "features value matrix",
        &[
            ("largest_chunk", max_matrix_bytes),
            ("in_flight_peak", 2 * max_matrix_bytes),
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
// from the config as they always were; the alignment is the eighth.
#[allow(clippy::too_many_arguments)]
fn fragment_features(
    rows: &[ChromRow],
    al: &TraceAlign,
    apex_rt: f64,
    coel_thresh: f64,
    bound: bool,
    frac: f64,
    grace: usize,
    global_bounds: Option<(f64, f64)>,
) -> FragFeatures {
    let mut f = FragFeatures::default();
    // Observed apex intensity per fragment (nearest scan to apex).
    let mut obs = Vec::with_capacity(rows.len());
    let mut pred = Vec::with_capacity(rows.len());
    let mut mass_err = Vec::new();
    let mut mass_w = Vec::new();
    for r in rows {
        let mut best = 0.0f32;
        let mut bestd = f64::MAX;
        for (k, &rt) in r.rt.iter().enumerate() {
            let d = (rt as f64 - apex_rt).abs();
            if d < bestd {
                bestd = d;
                best = r.inten[k];
            }
        }
        obs.push(best as f64);
        pred.push(r.pred_int as f64);
        let pe = ppm_diff(r.frag_obs_mz, r.frag_mz).abs();
        mass_err.push(pe);
        mass_w.push(best as f64);
        let is_b = r.frag_name.starts_with('b');
        if is_b {
            f.sum_b += best as f64;
            f.n_b += 1.0;
        } else {
            f.sum_y += best as f64;
            f.n_y += 1.0;
        }
    }
    f.frag_corr = pearson(&obs, &pred);
    f.frag_cosine = cosine(&obs, &pred);
    f.spectral_angle = spectral_angle(&obs, &pred);

    // normalized manhattan + rmsd on sum-normalized vectors
    let (on, pn) = (normalize_sum(&obs), normalize_sum(&pred));
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
    let (lo_i, hi_i) = if bound && axis_full.len() >= 3 {
        // boundary on the smoothed summed top-3-predicted-fragment profile, around apex
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
    } else {
        (0, axis_full.len().saturating_sub(1))
    };
    let axis: Vec<f32> = axis_full[lo_i..=hi_i].to_vec();
    let traces: Vec<Vec<f64>> = traces_full
        .iter()
        .map(|t| t[lo_i..=hi_i].to_vec())
        .collect();
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
    for a in 0..traces.len() {
        for b in (a + 1)..traces.len() {
            if axis.len() >= 2 {
                corrs.push(pearson(&traces[a], &traces[b]));
                let (lag, shape) = best_xcorr(&traces[a], &traces[b], 5);
                lags.push(lag.abs() as f64);
                shapes.push(shape);
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
        let mut refp = vec![0.0f64; np];
        for (fi, tr) in traces.iter().enumerate() {
            let w = pred[fi].max(0.0);
            for k in 0..np {
                refp[k] += w * tr[k];
            }
        }
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
        // pTimeCorr: each fragment XIC correlated with the reference profile.
        let rc: Vec<f64> = traces.iter().map(|tr| pearson(tr, &refp)).collect();
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
        for tr in &traces {
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
        for tr in &traces {
            let rk: f64 = tr.iter().zip(&refp).map(|(a, b)| a * b).sum::<f64>() / rr;
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
        // co-elution of cleaned traces and correlation of residuals (shared interferent).
        let mut clean_corrs = Vec::new();
        let mut res_corrs = Vec::new();
        for a in 0..traces.len() {
            for b in (a + 1)..traces.len() {
                clean_corrs.push(pearson(&cleaned[a], &cleaned[b]));
                res_corrs.push(pearson(&residuals[a], &residuals[b]));
            }
        }
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
        all_points.sort_by(|a, b| a.total_cmp(b));
        all_points[all_points.len() / 2].max(1.0)
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

/// Best cross-correlation of two traces over integer lags in [-maxlag, maxlag].
/// Returns (lag_of_max, normalized_max_value).
fn best_xcorr(a: &[f64], b: &[f64], maxlag: i32) -> (i32, f64) {
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
    fn plan_chunks_cuts_only_at_candidate_boundaries() {
        // Three PSM rows for candidate 1 (top-K), one each for 2 and 3; candidate 2 has
        // no chromatogram rows at all.
        let psm = [1, 1, 1, 2, 3];
        let chrom = [1, 1, 1, 1, 3, 3];
        // One row per chunk requested: the planner must still keep a candidate whole.
        let cs = plan_chunks(&psm, &chrom, "c.parquet", 1).unwrap();
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
        let one = plan_chunks(&psm, &chrom, "c.parquet", 1 << 20).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!((one[0].psm_lo, one[0].psm_hi, one[0].chrom_rows), (0, 5, 6));
    }

    #[test]
    fn plan_chunks_rejects_artifacts_it_cannot_join() {
        // A chromatogram candidate the PSM table does not have.
        let e = plan_chunks(&[1, 2], &[1, 9], "c.parquet", 1 << 20).unwrap_err();
        assert!(format!("{e}").contains("candidate 9"), "{e}");
        // A candidate whose rows are not contiguous.
        let e = plan_chunks(&[1, 2, 1], &[1], "c.parquet", 1 << 20).unwrap_err();
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

        // Equal VALUES but stored separately are not the same slice: still correct, just
        // through the general path.
        let copy = g1;
        let rows = vec![row("y1", 1.0, &g1, &v1), row("y2", 1.0, &copy, &v2)];
        assert!(align_shared_axis(&rows).is_none());
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
    fn value_matrix_hands_its_columns_over_by_move() {
        let mut m = ValueMatrix::new(3, 2);
        m.set(Some(1), 0, 7.5);
        m.set(Some(1), 1, -1.0);
        m.set(None, 0, 99.0); // a feature outside the active set is dropped
        assert_eq!(m.column(1), &[7.5, -1.0]);
        assert_eq!(m.take_column(1), vec![7.5, -1.0]);
        assert_eq!(m.column(0), &[0.0, 0.0]);
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
            for f in 0..6u32 {
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
        let spans = confident_row_spans(&rg, &sorted).expect("the writer records statistics");
        let read: usize = spans.iter().map(|s| s.1).sum();
        assert!(
            read < ch.nrows,
            "the pruning must skip something: read {read} of {} rows",
            ch.nrows
        );
        assert!(read > 0);

        // Same half-widths at several chunk sizes, pruned against the whole-table scan.
        // The chunk size matters: a candidate that straddles a chunk boundary is bounded
        // from each part separately, and the pruned pass has to split it in exactly the
        // same places for the percentiles to come out the same.
        for chunk in [7usize, 16, 64, 1 << 20] {
            let pruned =
                confident_global_bounds(&path, &spans, &confident, &apex_rt, &cfg, chunk).unwrap();
            let whole =
                confident_global_bounds(&path, &[(0, ch.nrows)], &confident, &apex_rt, &cfg, chunk)
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
        let run = |tag: &str, chunk_rows: usize| -> String {
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
                chunk_rows,
            )
            .unwrap();
            out
        };
        let one = run("one", 1 << 20);
        let many = run("many", 1);

        let a = mumdia_io::table::Table::read(&one).unwrap();
        let b = mumdia_io::table::Table::read(&many).unwrap();
        assert_eq!(a.nrows, 61);
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
            if let (Ok(x), Ok(y)) = (a.f64(&name), b.f64(&name)) {
                let xb: Vec<u64> = x.iter().map(|v| v.to_bits()).collect();
                let yb: Vec<u64> = y.iter().map(|v| v.to_bits()).collect();
                assert_eq!(xb, yb, "column '{name}' differs between chunk sizes");
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
}
