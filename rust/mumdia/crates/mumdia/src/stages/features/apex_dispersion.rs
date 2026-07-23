//! Extended feature family: apex dispersion + peak shape (sensitivity_plan
//! spec 03 §8.3-8.4, backlog P5.3).
//!
//! A real peptide's fragments share one apex and one symmetric elution peak; a
//! chimeric/interfered match has fragments apexing at scattered retention times
//! and an irregular consensus peak (shoulders, tailing, truncation). These
//! features quantify that agreement from the peak-bounded traces, independent of
//! absolute intensity (which is chimeric in DIA), so they capture BREADTH-of-
//! coelution evidence rather than height.
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items in
//! the same order; every value is finite; label-blind; emitted for every PSM.
use super::Evidence;

pub const NAMES: &[&str] = &[
    "frag_apex_rt_std",         // scatter of per-fragment apex RTs (s)
    "frag_apex_rt_mad",         // robust scatter (median abs dev, s)
    "frag_apex_max_dev",        // max |fragment apex - consensus apex| (s)
    "frag_apex_mean_dev",       // mean |fragment apex - consensus apex| (s)
    "frag_apex_agree_frac",     // fraction of signal fragments apexing within 1 scan of consensus
    "precursor_frag_apex_delta",// |MS1 mono apex - consensus apex| (s); 0 if no MS1
    "peak_symmetry",            // right-area / (left-area+right-area) of consensus peak (~0.5 ideal)
    "peak_tailing",             // right half-width / left half-width at half height
    "peak_n_local_maxima",      // local maxima in consensus profile (>=1)
    "peak_shoulder_score",      // 2nd-highest local max / apex height
    "peak_fwhm_scans",          // full width at half maximum (scans)
    "peak_truncation",          // 1 if apex at window edge or boundary height > 0.5*apex
    "apex_frac_of_window",      // apex position within the window [0,1] (edge = mis-centered)
];

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn std(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

fn median_abs_dev(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let m = median(v);
    let dev: Vec<f64> = v.iter().map(|x| (x - m).abs()).collect();
    median(&dev)
}

/// Index of the maximum of a slice (first on ties); None if empty or all <= 0.
fn argmax_pos(v: &[f64]) -> Option<usize> {
    let mut best = 0usize;
    let mut bv = f64::NEG_INFINITY;
    let mut any = false;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
            any = true;
        }
    }
    if any && bv > 0.0 {
        Some(best)
    } else {
        None
    }
}

pub fn values(e: &Evidence) -> Vec<f64> {
    let axis = &e.axis;
    let n = axis.len();
    let consensus_rt = if e.apex_idx < n { axis[e.apex_idx] } else { 0.0 };
    let mean_scan_dt = if n >= 2 {
        (axis[n - 1] - axis[0]) / (n as f64 - 1.0)
    } else {
        0.0
    };

    // Per-fragment apex RTs (only fragments that carry signal on the axis).
    let mut apex_rts: Vec<f64> = Vec::new();
    for tr in &e.traces {
        if let Some(pos) = argmax_pos(tr) {
            if pos < n {
                apex_rts.push(axis[pos]);
            }
        }
    }
    let devs: Vec<f64> = apex_rts.iter().map(|r| (r - consensus_rt).abs()).collect();
    let frag_apex_rt_std = std(&apex_rts);
    let frag_apex_rt_mad = median_abs_dev(&apex_rts);
    let frag_apex_max_dev = devs.iter().cloned().fold(0.0, f64::max);
    let frag_apex_mean_dev = mean(&devs);
    let frag_apex_agree_frac = if devs.is_empty() {
        0.0
    } else {
        let tol = mean_scan_dt.max(1e-9);
        devs.iter().filter(|d| **d <= tol).count() as f64 / devs.len() as f64
    };

    // Precursor (MS1 mono) vs fragment consensus apex.
    let precursor_frag_apex_delta = e
        .ms1_xic
        .first()
        .and_then(|mono| argmax_pos(mono).map(|p| (axis.get(p).copied().unwrap_or(consensus_rt) - consensus_rt).abs()))
        .unwrap_or(0.0);

    // Consensus profile shape (predicted-intensity-weighted reference profile).
    let prof = &e.ref_profile;
    let (peak_symmetry, peak_tailing, peak_fwhm_scans, peak_truncation, peak_n_local_maxima, peak_shoulder_score) =
        profile_shape(prof, e.apex_idx);

    let apex_frac_of_window = if n >= 2 {
        e.apex_idx as f64 / (n as f64 - 1.0)
    } else {
        0.0
    };

    let out = vec![
        frag_apex_rt_std,
        frag_apex_rt_mad,
        frag_apex_max_dev,
        frag_apex_mean_dev,
        frag_apex_agree_frac,
        precursor_frag_apex_delta,
        peak_symmetry,
        peak_tailing,
        peak_n_local_maxima,
        peak_shoulder_score,
        peak_fwhm_scans,
        peak_truncation,
        apex_frac_of_window,
    ];
    // Guarantee finiteness.
    out.into_iter()
        .map(|x| if x.is_finite() { x } else { 0.0 })
        .collect()
}

/// Shape descriptors of a consensus elution profile around `apex_idx`.
/// Returns (symmetry, tailing, fwhm_scans, truncation, n_local_maxima, shoulder).
fn profile_shape(prof: &[f64], apex_idx: usize) -> (f64, f64, f64, f64, f64, f64) {
    let n = prof.len();
    if n == 0 {
        return (0.5, 1.0, 0.0, 0.0, 0.0, 0.0);
    }
    let apex = apex_idx.min(n - 1);
    let apex_h = prof[apex];
    if apex_h <= 0.0 {
        return (0.5, 1.0, 0.0, 1.0, 0.0, 0.0);
    }
    // Left/right areas about the apex (symmetry).
    let left_area: f64 = prof[..=apex].iter().sum();
    let right_area: f64 = prof[apex..].iter().sum();
    let total = left_area + right_area;
    let symmetry = if total > 0.0 { right_area / total } else { 0.5 };

    // Half-max widths (in scans) each side of the apex.
    let half = 0.5 * apex_h;
    let mut lw = 0usize;
    let mut i = apex;
    while i > 0 && prof[i - 1] >= half {
        lw += 1;
        i -= 1;
    }
    let mut rw = 0usize;
    let mut j = apex;
    while j + 1 < n && prof[j + 1] >= half {
        rw += 1;
        j += 1;
    }
    let tailing = if lw > 0 {
        rw as f64 / lw as f64
    } else if rw > 0 {
        2.0 // right-only shoulder: strongly tailing
    } else {
        1.0
    };
    let fwhm = (lw + rw + 1) as f64;

    // Truncation: apex at edge, or either boundary still above half the apex.
    let truncation = if apex == 0 || apex == n - 1 || prof[0] > half || prof[n - 1] > half {
        1.0
    } else {
        0.0
    };

    // Local maxima + shoulder (2nd-highest local max relative to apex).
    let mut maxima: Vec<f64> = Vec::new();
    for k in 0..n {
        let l = if k == 0 { 0.0 } else { prof[k - 1] };
        let r = if k + 1 == n { 0.0 } else { prof[k + 1] };
        if prof[k] > 0.0 && prof[k] >= l && prof[k] >= r && prof[k] > l {
            maxima.push(prof[k]);
        }
    }
    let n_local = maxima.len().max(1) as f64;
    maxima.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let shoulder = if maxima.len() >= 2 {
        maxima[1] / apex_h
    } else {
        0.0
    };
    (symmetry, tailing, fwhm, truncation, n_local, shoulder)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(axis: Vec<f64>, traces: Vec<Vec<f64>>, ref_profile: Vec<f64>, apex_idx: usize) -> Evidence {
        Evidence {
            axis,
            traces,
            axis_full: vec![],
            traces_full: vec![],
            pred: vec![],
            obs_apex: vec![],
            is_b: vec![],
            ordinal: vec![],
            frag_charge: vec![],
            frag_mz: vec![],
            frag_obs_mz: vec![],
            mass_err_ppm: vec![],
            apex_idx,
            ref_profile,
            apex_rt: 0.0,
            rt_pred_cal: 0.0,
            rt_err: 0.0,
            gradient: 0.0,
            precursor_mz: 0.0,
            charge: 2,
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
            ms1_xic: vec![],
        }
    }

    #[test]
    fn names_match_values_len_and_finite() {
        let axis = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tr = vec![vec![0.0, 1.0, 4.0, 1.0, 0.0], vec![0.0, 1.0, 3.0, 1.0, 0.0]];
        let prof = vec![0.0, 1.0, 4.0, 1.0, 0.0];
        let v = values(&ev(axis, tr, prof, 2));
        assert_eq!(v.len(), NAMES.len());
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn coeluting_fragments_have_low_apex_dispersion() {
        // both fragments apex at index 2
        let axis = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tr = vec![vec![0.0, 2.0, 9.0, 2.0, 0.0], vec![0.0, 1.0, 8.0, 1.0, 0.0]];
        let prof = vec![0.0, 1.5, 8.5, 1.5, 0.0];
        let v = values(&ev(axis, tr, prof, 2));
        let std_i = NAMES.iter().position(|n| *n == "frag_apex_rt_std").unwrap();
        let agree_i = NAMES.iter().position(|n| *n == "frag_apex_agree_frac").unwrap();
        assert_eq!(v[std_i], 0.0); // both apex at same scan
        assert_eq!(v[agree_i], 1.0);
    }

    #[test]
    fn scattered_fragments_have_high_apex_dispersion() {
        // fragments apex at index 1 and index 3 (2 scans apart)
        let axis = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tr = vec![vec![0.0, 9.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 9.0, 0.0]];
        let prof = vec![0.0, 4.5, 1.0, 4.5, 0.0];
        let v = values(&ev(axis, tr, prof, 1));
        let std_i = NAMES.iter().position(|n| *n == "frag_apex_rt_std").unwrap();
        let maxdev_i = NAMES.iter().position(|n| *n == "frag_apex_max_dev").unwrap();
        assert!(v[std_i] > 0.0);
        assert!(v[maxdev_i] >= 2.0);
    }

    #[test]
    fn truncation_flagged_when_apex_at_edge() {
        let axis = vec![0.0, 1.0, 2.0];
        let tr = vec![vec![9.0, 4.0, 1.0]];
        let prof = vec![9.0, 4.0, 1.0];
        let v = values(&ev(axis, tr, prof, 0));
        let trunc_i = NAMES.iter().position(|n| *n == "peak_truncation").unwrap();
        assert_eq!(v[trunc_i], 1.0);
    }
}
