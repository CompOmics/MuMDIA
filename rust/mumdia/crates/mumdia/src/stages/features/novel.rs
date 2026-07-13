//! Extended feature family: novel.
//!
//! Seed corroboration, precursor/charge metadata, and count/intensity
//! summaries. Contract: `NAMES` and `values(&Evidence)` return the same number
//! of items in the same order. Names are globally unique, snake_case, and
//! append-only (part of the frozen extended feature schema). See the `Evidence`
//! struct in `stages/features.rs` for the available per-PSM evidence.
//!
//! Sequence-dependent features from the plan (`n_missed_cleavages`,
//! `frac_missed_cleavages`, `cterm_is_lys`, `cterm_is_arg`, `n_modifications`)
//! are skipped: the `Evidence` struct carries only `seq_len`, not the stripped
//! sequence or the ProForma modification list, so their evidence is unavailable.
use super::Evidence;

const PROTON_MASS: f64 = 1.007276466812;

pub const NAMES: &[&str] = &[
    "log_seed_hyperscore",
    "seed_hyperscore_per_matched",
    "seed_identified",
    "peptide_length",
    "precursor_charge",
    "charge_is_2",
    "charge_is_3",
    "charge_is_4plus",
    "precursor_mass",
    "log_total_matched_intensity",
    "n_matched_frags",
    "n_predicted_frags",
];

/// Replace any non-finite (NaN/Inf) value with 0.0.
#[inline]
fn finite(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

pub fn values(e: &Evidence) -> Vec<f64> {
    // Seed corroboration.
    let log_seed_hyperscore = (1.0 + e.seed_score.max(0.0)).ln();
    let seed_hyperscore_per_matched = e.seed_score / (e.n_matched.max(1) as f64);
    let seed_identified = e.seed_identified;

    // Peptide / precursor metadata.
    let peptide_length = e.seq_len as f64;
    let precursor_charge = e.charge as f64;
    let charge_is_2 = if e.charge == 2 { 1.0 } else { 0.0 };
    let charge_is_3 = if e.charge == 3 { 1.0 } else { 0.0 };
    let charge_is_4plus = if e.charge >= 4 { 1.0 } else { 0.0 };
    // Neutral monoisotopic precursor mass. Guard nonsensical charge (<=0).
    let precursor_mass = if e.charge > 0 {
        (e.precursor_mz - PROTON_MASS) * (e.charge as f64)
    } else {
        0.0
    };

    // Matched apex intensity sum (only fragments observed at the apex).
    let matched_sum: f64 = e
        .obs_apex
        .iter()
        .filter(|&&a| a > 0.0)
        .sum();
    let log_total_matched_intensity = (1.0 + matched_sum.max(0.0)).ln();

    // Count passthroughs.
    let n_matched_frags = e.n_matched as f64;
    let n_predicted_frags = e.n_predicted as f64;

    let out = vec![
        log_seed_hyperscore,
        seed_hyperscore_per_matched,
        seed_identified,
        peptide_length,
        precursor_charge,
        charge_is_2,
        charge_is_3,
        charge_is_4plus,
        precursor_mass,
        log_total_matched_intensity,
        n_matched_frags,
        n_predicted_frags,
    ];
    out.into_iter().map(finite).collect()
}
