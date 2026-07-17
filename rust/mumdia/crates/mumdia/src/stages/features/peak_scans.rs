//! Extended feature family: peak-scan count / window-degeneracy indicator.
//!
//! Label-blind and emitted for EVERY PSM (never early-returns to zeros), so the
//! rescorer can distinguish an *undefined* zero from a *measured* zero. When the
//! extraction window is mis-centered (e.g. an RT-calibration error puts the true
//! apex at the window edge), the peak collapses to 1-2 non-empty scans and the
//! window-based families (order_consistency, peak_completeness, self-cosine)
//! degenerate to 0.0. Those zeros are indistinguishable from a genuine decoy-like
//! zero unless the model also sees how many peak scans actually existed. This
//! family exposes exactly that.
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items in
//! the same order; every value is finite; the length is stable.
use super::Evidence;

pub const NAMES: &[&str] = &["n_peak_scans", "peak_window_degenerate"];

/// Scan count below which the window-based families early-return all-zeros
/// (mirrors order_consistency::MIN_SCANS).
const MIN_SCANS: usize = 3;

pub fn values(e: &Evidence) -> Vec<f64> {
    // Number of scan positions where any fragment carries observed intensity.
    let np = e.traces.iter().map(|t| t.len()).min().unwrap_or(0);
    let n_scans = (0..np)
        .filter(|&j| e.traces.iter().any(|t| t[j] > 0.0))
        .count();
    let degenerate = if n_scans < MIN_SCANS { 1.0 } else { 0.0 };
    vec![n_scans as f64, degenerate]
}
