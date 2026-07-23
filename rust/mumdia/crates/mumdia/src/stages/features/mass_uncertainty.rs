//! Extended feature family: fragment mass-error dispersion + evidence breadth
//! (sensitivity_plan spec 03 §8.1-8.2, backlog P5.1/P5.2).
//!
//! A correct identification matches its fragments with small AND mutually
//! consistent mass errors, and spreads its observed intensity across many of its
//! predicted transitions (breadth of evidence). A chimeric/interfered match tends
//! to have scattered mass errors and its signal concentrated in one or two
//! coincidental channels. These features summarize the DISTRIBUTION of per-
//! fragment mass errors and how the observed evidence is spread, complementing the
//! single-number mass-accuracy family. Intensity here is used only for shape
//! (participation ratio, concentration), not as an absolute magnitude, so it is
//! robust to the chimeric-intensity problem in DIA.
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items in
//! the same order; every value is finite; label-blind; emitted for every PSM.
use super::Evidence;

pub const NAMES: &[&str] = &[
    "frag_mass_err_median",     // median signed ppm over matched fragments
    "frag_mass_err_abs_median", // median |ppm|
    "frag_mass_err_std",        // dispersion of ppm (consistency)
    "frag_mass_err_iqr",        // robust dispersion of ppm
    "frag_mass_err_max_abs",    // worst |ppm|
    "frag_mass_err_range",      // max - min ppm
    "effective_frag_count",     // inverse participation ratio of observed intensity
    "evidence_concentration",   // fraction of observed intensity in the strongest fragment
    "frac_top3_pred_observed",  // fraction of the top-3 predicted ions actually observed
    "frac_top5_pred_observed",  // fraction of the top-5 predicted ions actually observed
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

/// Linear-interpolated percentile (q in [0,1]) of an unsorted slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Fraction of the top-`k` predicted fragments (by predicted intensity) that are
/// actually observed at the apex (obs_apex > 0). Breadth of the STRONG ions.
fn frac_top_pred_observed(pred: &[f64], obs: &[f64], k: usize) -> f64 {
    let n = pred.len().min(obs.len());
    if n == 0 {
        return 0.0;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        pred[b]
            .partial_cmp(&pred[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let take = k.min(n);
    if take == 0 {
        return 0.0;
    }
    let hit = idx[..take].iter().filter(|&&i| obs[i] > 0.0).count();
    hit as f64 / take as f64
}

pub fn values(e: &Evidence) -> Vec<f64> {
    // Mass errors are meaningful only for observed (matched) fragments.
    let errs: Vec<f64> = e
        .mass_err_ppm
        .iter()
        .zip(&e.obs_apex)
        .filter(|(_, &o)| o > 0.0)
        .map(|(&m, _)| m)
        .collect();

    let (median, abs_median, iqr, max_abs, range) = if errs.is_empty() {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        let mut s = errs.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = percentile(&s, 0.5);
        let mut a: Vec<f64> = errs.iter().map(|x| x.abs()).collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let abs_med = percentile(&a, 0.5);
        let iqr = percentile(&s, 0.75) - percentile(&s, 0.25);
        let max_abs = a[a.len() - 1];
        let range = s[s.len() - 1] - s[0];
        (med, abs_med, iqr, max_abs, range)
    };
    let err_std = std(&errs);

    // Evidence spread over observed intensity.
    let obs_pos: Vec<f64> = e.obs_apex.iter().cloned().filter(|&x| x > 0.0).collect();
    let sum: f64 = obs_pos.iter().sum();
    let sum_sq: f64 = obs_pos.iter().map(|x| x * x).sum();
    let effective_frag_count = if sum_sq > 0.0 {
        sum * sum / sum_sq
    } else {
        0.0
    };
    let evidence_concentration = if sum > 0.0 {
        obs_pos.iter().cloned().fold(0.0, f64::max) / sum
    } else {
        0.0
    };

    let out = vec![
        median,
        abs_median,
        err_std,
        iqr,
        max_abs,
        range,
        effective_frag_count,
        evidence_concentration,
        frac_top_pred_observed(&e.pred, &e.obs_apex, 3),
        frac_top_pred_observed(&e.pred, &e.obs_apex, 5),
    ];
    out.into_iter()
        .map(|x| if x.is_finite() { x } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(pred: Vec<f64>, obs: Vec<f64>, err: Vec<f64>) -> Evidence {
        Evidence {
            axis: vec![],
            traces: vec![],
            axis_full: vec![],
            traces_full: vec![],
            pred,
            obs_apex: obs,
            is_b: vec![],
            ordinal: vec![],
            frag_charge: vec![],
            frag_mz: vec![],
            frag_obs_mz: vec![],
            mass_err_ppm: err,
            apex_idx: 0,
            ref_profile: vec![],
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
        let v = values(&ev(
            vec![9.0, 5.0, 1.0],
            vec![8.0, 4.0, 0.0],
            vec![1.0, -2.0, 50.0],
        ));
        assert_eq!(v.len(), NAMES.len());
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn only_observed_fragments_count_toward_mass_error() {
        // third fragment is unobserved (obs 0) with a huge error -> excluded
        let v = values(&ev(
            vec![9.0, 5.0, 1.0],
            vec![8.0, 4.0, 0.0],
            vec![1.0, -1.0, 999.0],
        ));
        let maxabs_i = NAMES
            .iter()
            .position(|n| *n == "frag_mass_err_max_abs")
            .unwrap();
        assert!(v[maxabs_i] <= 1.0 + 1e-9); // 999 excluded, only |1|,|-1|
    }

    #[test]
    fn breadth_of_top_predicted_ions() {
        // top-3 predicted = frags 0,1,2 (pred 9,5,3); obs>0 for 0 and 1 only -> 2/3
        let v = values(&ev(
            vec![9.0, 5.0, 3.0, 1.0],
            vec![8.0, 4.0, 0.0, 2.0],
            vec![0.0; 4],
        ));
        let t3 = NAMES
            .iter()
            .position(|n| *n == "frac_top3_pred_observed")
            .unwrap();
        assert!((v[t3] - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn concentration_high_when_one_dominant_fragment() {
        let v = values(&ev(vec![1.0, 1.0], vec![100.0, 1.0], vec![0.0, 0.0]));
        let conc = NAMES
            .iter()
            .position(|n| *n == "evidence_concentration")
            .unwrap();
        assert!(v[conc] > 0.9);
    }
}
