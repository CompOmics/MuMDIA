//! Extended feature family: spectrum-centric demixing (D2, fragment_competition
//! strategies report / family I3).
//!
//! These carry the per-candidate result of the non-negative least-squares demix of the
//! co-isolated candidate x fragment design matrix at the candidate's apex scan
//! (`extract::demix_at_apex`, solver `crate::solve::nnls`). They are 0 unless
//! `extract.emit_demix_features` populated them, so the vector effect is unchanged when
//! demixing is off. Label-blind; the demix reads only observed/predicted intensity.
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items in the same
//! order; every value is finite.
use super::Evidence;

pub const NAMES: &[&str] = &[
    // Fraction of the apex spectrum's energy the joint fit explains (0-1).
    "deconv_explained_frac",
    // 1 if this candidate survived the NNLS active set (coefficient > 0), else 0.
    "deconv_active",
    // This candidate's fraction of the total demixed abundance at its apex (0-1); a
    // borrower explained away by better-supported co-eluters gets a low share.
    "deconv_share",
    // Identifiability (D3): max cosine similarity of this candidate's design column with
    // any co-isolated rival (0-1). Near 1 = near-degenerate, so the demix split is
    // arbitrary and its coefficient should be distrusted. No incumbent engine emits this.
    "deconv_max_collinearity",
    // D1 shadow: fraction of the candidate's observed apex intensity that survives after
    // subtracting each co-eluter's unique-ion-estimated contribution (0-1). A real second
    // peptide keeps most of its signal; a pure borrower is subtracted toward 0.
    "shadow_kept_frac",
];

#[inline]
fn fin(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

pub fn values(e: &Evidence) -> Vec<f64> {
    vec![
        fin(e.deconv_explained).clamp(0.0, 1.0),
        fin(e.deconv_active).clamp(0.0, 1.0),
        fin(e.deconv_share).clamp(0.0, 1.0),
        fin(e.deconv_max_collin).clamp(0.0, 1.0),
        fin(e.deconv_shadow).clamp(0.0, 1.0),
    ]
}
