//! Stage E `mumdia features` (PLAN.md Stage E, Section 8.3 enriched catalogue).
//! Reads psms_extracted + chromatograms (+ MS1 apex isotopes carried on the
//! PSM rows) and computes a fixed, named, versioned feature vector per PSM.
//! The active feature set is config-driven (`minimal` or `rich`); its ordered
//! list is hashed into a `classifier_feature_schema_id` and written to a
//! companion `<features>.schema.json` so the classifier input is reproducible
//! and never applied under a mismatched set (PLAN.md Section 2, 5, 9.1).

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context as _, Result};
use mumdia_core::config::{FeatureSet, FeaturesConfig};
use mumdia_core::constants::{ppm_diff, PROTON};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, Table};
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

/// The minimal feature set (PLAN.md Section 10).
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

/// Additional features for the `rich`/`standard` set (PLAN.md Section 8.3).
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

struct ChromRow {
    frag_name: String,
    frag_mz: f64,
    frag_obs_mz: f64,
    pred_int: f32,
    rt: Vec<f32>,
    inten: Vec<f32>,
}

/// Per-PSM evidence handed to the extended feature families. All arrays are
/// f64. Fragment-indexed arrays share one order; time-series share `axis`
/// (elution-peak-bounded) or `axis_full` (whole extracted window). Built once
/// per PSM by [`build_evidence`], then scalar fields are filled by the caller.
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
        let (b, o, c) = parse_ion(&r.frag_name);
        is_b.push(b);
        ordinal.push(o);
        frag_charge.push(c);
        frag_mz.push(r.frag_mz);
        frag_obs_mz.push(r.frag_obs_mz);
        mass_err_ppm.push(ppm_diff(r.frag_obs_mz, r.frag_mz));
    }

    let mut axis_full: Vec<f32> = rows.iter().flat_map(|r| r.rt.iter().cloned()).collect();
    axis_full.sort_by(|a, b| a.partial_cmp(b).unwrap());
    axis_full.dedup();
    let traces_full: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
            let map: HashMap<u32, f32> =
                r.rt.iter()
                    .zip(&r.inten)
                    .map(|(&t, &v)| (t.to_bits(), v))
                    .collect();
            axis_full
                .iter()
                .map(|t| *map.get(&t.to_bits()).unwrap_or(&0.0) as f64)
                .collect()
        })
        .collect();

    let (lo_i, hi_i) = if axis_full.len() >= 3 {
        let ai = axis_full
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (**a as f64 - apex_rt)
                    .abs()
                    .partial_cmp(&((**b as f64 - apex_rt).abs()))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        match global_bounds {
            Some((l, r)) => global_bound_indices(&axis_full, apex_rt, ai, l, r),
            None => {
                let mut ord: Vec<usize> = (0..pred.len()).collect();
                ord.sort_by(|&a, &b| {
                    pred[b]
                        .partial_cmp(&pred[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
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
        .min_by(|(_, a), (_, b)| {
            (*a - apex_rt)
                .abs()
                .partial_cmp(&(*b - apex_rt).abs())
                .unwrap()
        })
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
                        .zip(&r.inten)
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
    /// Optional seed_psms for search-engine corroboration features (PLAN.md 8.3).
    pub seed: Option<&'a str>,
    pub out: &'a str,
    pub out_pin: &'a str,
    pub cfg: &'a FeaturesConfig,
    pub config_hash: &'a str,
}

pub fn run(p: FeaturesParams) -> Result<u64> {
    let t0 = Instant::now();
    let ps = Table::read(p.psms)?;
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

    // Group chromatograms by candidate_id.
    // Project explicitly. features uses all of these; naming them means a future extra
    // column in the artifact is not silently decoded on this hot path.
    let ch = Table::read_cols(
        p.chromatograms,
        &[
            "candidate_id",
            "frag_name",
            "frag_mz",
            "frag_obs_mz",
            "predicted_intensity",
            "rt",
            "intensity",
        ],
    )?;
    let ch_cid = ch.u32("candidate_id")?;
    let ch_name = ch.str("frag_name")?;
    let ch_fmz = ch.f64("frag_mz")?;
    let ch_obsmz = ch.f64("frag_obs_mz").unwrap_or_else(|_| ch_fmz.clone());
    let ch_pint = ch.f32("predicted_intensity")?;
    let ch_rt = ch.list_f32("rt")?;
    let ch_int = ch.list_f32("intensity")?;
    let mut chrom: HashMap<u32, Vec<ChromRow>> = HashMap::new();
    // MS1 isotope XIC rows (frag_name "ms1_mono"/"ms1_iso1"/"ms1_iso2") are kept
    // separate so they never enter the fragment-based features; they feed the
    // extended MS1-profile family via Evidence.ms1_xic. Absent in older artifacts.
    let mut ms1x: HashMap<u32, Vec<ChromRow>> = HashMap::new();
    for i in 0..ch.nrows {
        let row = ChromRow {
            frag_name: ch_name[i].clone(),
            frag_mz: ch_fmz[i],
            frag_obs_mz: ch_obsmz[i],
            pred_int: ch_pint[i],
            rt: ch_rt[i].clone(),
            inten: ch_int[i].clone(),
        };
        if ch_name[i].starts_with("ms1_") {
            ms1x.entry(ch_cid[i]).or_default().push(row);
        } else {
            chrom.entry(ch_cid[i]).or_default().push(row);
        }
    }

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
            let s = Table::read(path)?;
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
    // the per-candidate boundary detection (default).
    let global_bounds: Option<(f64, f64)> = if p.cfg.bound_from_confident {
        let mut lefts: Vec<f64> = Vec::new();
        let mut rights: Vec<f64> = Vec::new();
        for i in 0..ps.nrows {
            if !confident_cids.contains(&cid[i]) {
                continue;
            }
            if let Some(rows) = chrom.get(&cid[i]) {
                if let Some((lo, hi)) = elution_peak_rt_bounds(
                    rows,
                    apex_rt[i],
                    p.cfg.bound_peak_fraction,
                    p.cfg.bound_peak_grace,
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
        if lefts.len() >= 20 {
            let q = (p.cfg.bound_confident_pct / 100.0).clamp(0.0, 1.0);
            let (l, r) = (percentile(&lefts, q), percentile(&rights, q));
            info!(
                n_confident = lefts.len(),
                left_hw_s = l,
                right_hw_s = r,
                pct = p.cfg.bound_confident_pct,
                "features: global elution half-widths from confident set"
            );
            Some((l, r))
        } else {
            warn!(
                n_confident = lefts.len(),
                "features: bound_from_confident set but < 20 confident anchors; \
                 falling back to per-candidate boundary"
            );
            None
        }
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
    // Evidence families since each charge is a separate candidate.
    let mut pf_charges: HashMap<&str, std::collections::HashSet<i32>> = HashMap::new();
    let mut pf_int: HashMap<&str, f64> = HashMap::new();
    for i in 0..n {
        pf_charges
            .entry(pform[i].as_str())
            .or_default()
            .insert(charge[i]);
        *pf_int.entry(pform[i].as_str()).or_insert(0.0) += apex_int[i] as f64;
    }
    let f_n_charge: Vec<f64> = (0..n)
        .map(|i| pf_charges[pform[i].as_str()].len() as f64)
        .collect();
    let f_charge_multi: Vec<f64> = f_n_charge
        .iter()
        .map(|&c| if c >= 2.0 { 1.0 } else { 0.0 })
        .collect();
    // ln(1 + summed apex intensity of the OTHER charge states of this peptidoform):
    // how much independent charge-state evidence reinforces this PSM (unbounded).
    let f_cross_charge_int: Vec<f64> = (0..n)
        .map(|i| (1.0 + (pf_int[pform[i].as_str()] - apex_int[i] as f64).max(0.0)).ln())
        .collect();

    // Compute the full feature superset into a name->values map.
    let mut fmap: HashMap<&str, Vec<f64>> = HashMap::new();
    let push = |m: &mut HashMap<&str, Vec<f64>>, k: &'static str, v: f64| {
        m.entry(k).or_insert_with(|| Vec::with_capacity(n)).push(v);
    };
    let mut prelim = vec![0.0f64; n];
    let mut elu_lo = vec![0.0f64; n];
    let mut elu_hi = vec![0.0f64; n];
    let extended = matches!(p.cfg.set, FeatureSet::Extended);
    let ext_names = extended_name_refs();

    // The two expensive per-PSM computations (`fragment_features` and, when the
    // extended set is active, `build_evidence` + `extended_values`) are pure
    // functions of that PSM's own inputs, so precompute them in parallel and
    // index by row. Collecting into a Vec preserves row order; the serial
    // assembly below reads `per[i]` and is otherwise byte-for-byte unchanged, so
    // fmap / prelim / output columns are identical to the serial version.
    struct PerPsm {
        ff: FragFeatures,
        ext: Vec<f64>,
    }
    let per: Vec<PerPsm> = (0..n)
        .into_par_iter()
        .map(|i| {
            let ff = match chrom.get(&cid[i]) {
                Some(rows) if !rows.is_empty() => fragment_features(
                    rows,
                    apex_rt[i],
                    p.cfg.coelution_corr_threshold,
                    p.cfg.bound_features,
                    p.cfg.bound_peak_fraction,
                    p.cfg.bound_peak_grace,
                    global_bounds,
                ),
                _ => FragFeatures::default(),
            };
            let ext = if extended {
                match chrom.get(&cid[i]) {
                    Some(rows) if !rows.is_empty() => {
                        let ms1_rows = ms1x.get(&cid[i]).map(|v| v.as_slice()).unwrap_or(&[]);
                        let mut ev = build_evidence(
                            rows,
                            ms1_rows,
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
                        extended_values(&ev)
                    }
                    _ => vec![0.0; ext_names.len()],
                }
            } else {
                Vec::new()
            };
            PerPsm { ff, ext }
        })
        .collect();

    for i in 0..n {
        let ff = &per[i].ff;
        elu_lo[i] = ff.elution_lo;
        elu_hi[i] = ff.elution_hi;
        let rt_err = calibrated_rt_error(apex_rt[i], rt_cal[i]);
        // MS1 isotope features.
        let neutral = mz[i] * charge[i] as f64 - charge[i] as f64 * PROTON;
        let (iso_corr, isom1_ratio, log_mono, has_ms1) =
            isotope_features(ms1_m1[i], ms1_mono[i], ms1_i1[i], ms1_i2[i], neutral);

        push(&mut fmap, "rt_error_abs", rt_err);
        push(&mut fmap, "rt_error_rel", rt_err / gradient);
        push(&mut fmap, "n_matched_fragments", n_matched[i] as f64);
        push(&mut fmap, "coelution_run", corun[i] as f64);
        push(
            &mut fmap,
            "log_apex_intensity",
            (1.0 + apex_int[i] as f64).ln(),
        );
        push(&mut fmap, "frag_corr", ff.frag_corr);
        push(&mut fmap, "frag_cosine", ff.frag_cosine);
        push(&mut fmap, "spectral_angle", ff.spectral_angle);
        push(&mut fmap, "coelution_mean", ff.coelution_mean);
        push(&mut fmap, "coelution_best", ff.coelution_best);
        push(&mut fmap, "n_coelution_above", ff.n_coelution_above);
        push(&mut fmap, "charge", charge[i] as f64);
        push(
            &mut fmap,
            "peptide_length",
            peptide_length(&pform[i]) as f64,
        );
        push(
            &mut fmap,
            "n_proteins",
            (protein[i].matches(';').count() + 1) as f64,
        );
        push(&mut fmap, "library_norm_manhattan", ff.norm_manhattan);
        push(&mut fmap, "library_rmsd", ff.rmsd);
        push(&mut fmap, "xcorr_coelution", ff.xcorr_coelution);
        push(&mut fmap, "xcorr_shape", ff.xcorr_shape);
        push(&mut fmap, "sum_b_intensity", ff.sum_b);
        push(&mut fmap, "sum_y_intensity", ff.sum_y);
        push(&mut fmap, "diff_by_intensity", ff.sum_b - ff.sum_y);
        push(&mut fmap, "n_b_ions", ff.n_b);
        push(&mut fmap, "n_y_ions", ff.n_y);
        push(&mut fmap, "weighted_mass_error", ff.weighted_mass_error);
        push(&mut fmap, "mean_mass_error", ff.mean_mass_error);
        push(&mut fmap, "isotope_corr", iso_corr);
        push(&mut fmap, "ms1_isom1_ratio", isom1_ratio);
        push(&mut fmap, "log_mono_ms1", log_mono);
        push(&mut fmap, "has_ms1", has_ms1);
        push(&mut fmap, "log_sn", ff.log_sn);
        push(&mut fmap, "n_observations", ff.n_observations);
        push(&mut fmap, "base_width_rt", ff.base_width_rt);
        push(
            &mut fmap,
            "seed_score",
            *seed_score_map.get(&cid[i]).unwrap_or(&0.0),
        );
        push(
            &mut fmap,
            "seed_identified",
            *seed_id_map.get(&cid[i]).unwrap_or(&0.0),
        );
        push(
            &mut fmap,
            "matched_fraction",
            n_matched[i] as f64 / (n_pred[i].max(1) as f64),
        );
        push(&mut fmap, "profile_cos", ff.profile_cos);
        push(&mut fmap, "ref_corr", ff.ref_corr);
        push(&mut fmap, "best_ref_corr", ff.best_ref_corr);
        push(&mut fmap, "low_frag_coel", ff.low_frag_coel);
        push(&mut fmap, "evidence", ff.evidence);
        push(&mut fmap, "contrast_min", ff.contrast_min);
        push(&mut fmap, "resid_corr", ff.resid_corr);
        push(&mut fmap, "coel_clean", ff.coel_clean);
        push(&mut fmap, "shadow_frac", ff.shadow_frac);
        push(&mut fmap, "peak_contested_frac", contested[i]);
        push(&mut fmap, "peak_contested_count_frac", contested_count[i]);
        push(&mut fmap, "peak_apportioned_frac", apportioned[i]);

        // Extended battery (opt-in). Build the shared Evidence once per PSM and
        // fan it out to the family modules; push their values under the fixed
        // registry-order names.
        if extended {
            for (k, v) in ext_names.iter().zip(&per[i].ext) {
                push(&mut fmap, k, *v);
            }
        }

        prelim[i] = n_matched[i] as f64 * (0.5 + ff.frag_corr.max(0.0))
            + ff.coelution_mean.max(0.0)
            + (1.0 + apex_int[i] as f64).ln() * 0.1
            - rt_err / gradient;
    }

    // Cross-charge corroboration features (populated directly; not per-PSM Evidence).
    fmap.insert("n_charge_states", f_n_charge);
    fmap.insert("charge_multi_flag", f_charge_multi);
    fmap.insert("cross_charge_intensity_log", f_cross_charge_int);

    // Build output columns: bookkeeping + active feature list.
    let mut cols: Vec<Col> = vec![
        Col::U32("candidate_id".into(), cid.clone()),
        Col::I32("peak_rank".into(), peak_rank.clone()),
        Col::Str("label".into(), label.clone()),
        Col::U32("base_peptide_id".into(), base.clone()),
        Col::Str("peptidoform".into(), pform.clone()),
        Col::Str("protein".into(), protein.clone()),
        Col::F64("apex_rt".into(), apex_rt.clone()),
        Col::F64("elution_lo".into(), elu_lo.clone()),
        Col::F64("elution_hi".into(), elu_hi.clone()),
        Col::F64("precursor_mz".into(), mz.clone()),
        Col::F64("prelim_score".into(), prelim.clone()),
    ];
    for name in &cols_active {
        let key: &str = name.as_str();
        cols.push(Col::F64(
            name.clone(),
            fmap.get(key).cloned().unwrap_or_else(|| vec![0.0; n]),
        ));
    }
    let rows = write_table(p.out, cols)?;

    // Feature schema companion.
    let schema_id = feature_schema_id(&cols_active);
    mumdia_io::json::write_json(
        &format!("{}.schema.json", p.out),
        &FeatureSchema {
            feature_columns: cols_active.clone(),
            schema_id: schema_id.clone(),
        },
    )?;

    // PIN. Nothing in the pipeline reads this artifact (`rescore` builds its own PIN for
    // the sidecars); it exists for external Percolator-style tooling, so it is gated.
    // When it IS written we resolve each feature column ONCE into a slice reference and
    // stream rows from those, instead of transposing the whole matrix into a
    // Vec<Vec<f64>> first: at 1.5M rows x 387 features that transpose allocated ~4.6 GB
    // and performed ~580M string-keyed HashMap lookups. Byte output is unchanged.
    if p.cfg.emit_pin {
        let empty: Vec<f64> = Vec::new();
        let fcols: Vec<&Vec<f64>> = cols_active
            .iter()
            .map(|c| fmap.get(c.as_str()).unwrap_or(&empty))
            .collect();
        write_pin(
            p.out_pin,
            &cols_active,
            &cid,
            &label,
            &pform,
            &protein,
            &mz,
            &fcols,
        )?;
    } else {
        tracing::debug!(
            path = %p.out_pin,
            "features: PIN emission disabled (features.emit_pin = false)"
        );
    }

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

#[derive(Default)]
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
    let mut axis: Vec<f32> = rows.iter().flat_map(|r| r.rt.iter().cloned()).collect();
    axis.sort_by(|a, b| a.partial_cmp(b).unwrap());
    axis.dedup();
    if axis.len() < 3 {
        return None;
    }
    let traces: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
            let map: HashMap<u32, f32> =
                r.rt.iter()
                    .zip(&r.inten)
                    .map(|(&t, &v)| (t.to_bits(), v))
                    .collect();
            axis.iter()
                .map(|t| *map.get(&t.to_bits()).unwrap_or(&0.0) as f64)
                .collect()
        })
        .collect();
    let mut ord: Vec<usize> = (0..rows.len()).collect();
    ord.sort_by(|&a, &b| {
        rows[b]
            .pred_int
            .partial_cmp(&rows[a].pred_int)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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
                .partial_cmp(&((**b as f64 - apex_rt).abs()))
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let (lo, hi) = peak_bounds(&prof, ai, frac, grace);
    Some((axis[lo], axis[hi]))
}

fn fragment_features(
    rows: &[ChromRow],
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

    // Align traces on the union RT axis, then restrict to the elution PEAK so the
    // trace-based features below are computed over the peak, not the whole extracted
    // RT window (which spans +/- w_rt and would dilute co-elution/profile scores).
    let mut axis_full: Vec<f32> = rows.iter().flat_map(|r| r.rt.iter().cloned()).collect();
    axis_full.sort_by(|a, b| a.partial_cmp(b).unwrap());
    axis_full.dedup();
    let traces_full: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
            let map: HashMap<u32, f32> =
                r.rt.iter()
                    .zip(&r.inten)
                    .map(|(&t, &v)| (t.to_bits(), v))
                    .collect();
            axis_full
                .iter()
                .map(|t| *map.get(&t.to_bits()).unwrap_or(&0.0) as f64)
                .collect()
        })
        .collect();
    let (lo_i, hi_i) = if bound && axis_full.len() >= 3 {
        // boundary on the smoothed summed top-3-predicted-fragment profile, around apex
        let ai = axis_full
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (**a as f64 - apex_rt)
                    .abs()
                    .partial_cmp(&((**b as f64 - apex_rt).abs()))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        match global_bounds {
            Some((l, r)) => global_bound_indices(&axis_full, apex_rt, ai, l, r),
            None => {
                let mut ord: Vec<usize> = (0..pred.len()).collect();
                ord.sort_by(|&a, &b| {
                    pred[b]
                        .partial_cmp(&pred[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
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
        for k in 0..np {
            let w = refp[k] * refp[k];
            if w <= 0.0 {
                continue;
            }
            let obs_k: Vec<f64> = traces.iter().map(|tr| tr[k]).collect();
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
        order.sort_by(|&a, &b| pred[a].partial_cmp(&pred[b]).unwrap());
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
        for tr in &traces {
            let others: Vec<f64> = (0..np).map(|k| total[k] - tr[k]).collect();
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
        all_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

#[allow(clippy::too_many_arguments)]
fn write_pin(
    path: &str,
    feature_cols: &[String],
    cid: &[u32],
    label: &[String],
    pform: &[String],
    protein: &[String],
    mz: &[f64],
    feats: &[&Vec<f64>],
) -> Result<()> {
    use std::io::Write as _;
    // Stream row by row through a BufWriter instead of materializing the whole
    // ~574k-row PIN as one String. Byte output is unchanged: same header, same
    // per-row field order, same {:.5}/{:.6} precision and tab/newline separators.
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    w.write_all(b"SpecId\tLabel\tScanNr\tExpMass\tCalcMass\t")?;
    w.write_all(feature_cols.join("\t").as_bytes())?;
    w.write_all(b"\tPeptide\tProteins\n")?;
    for i in 0..cid.len() {
        let lab = if label[i] == "decoy" { -1 } else { 1 };
        write!(
            w,
            "cand_{}\t{}\t{}\t{:.5}\t{:.5}\t",
            cid[i], lab, cid[i], mz[i], mz[i]
        )?;
        // `feats` is now COLUMN-major (one slice per feature, resolved once by the
        // caller), so the row value is feats[fi][i]. Absent columns are empty slices and
        // print 0.000000, matching the previous `.unwrap_or(0.0)` behaviour exactly.
        #[allow(clippy::needless_range_loop)]
        for fi in 0..feature_cols.len() {
            let v = feats[fi].get(i).copied().unwrap_or(0.0);
            write!(w, "{v:.6}\t")?;
        }
        writeln!(w, "-.{}.-\t{}", pform[i], protein[i])?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
