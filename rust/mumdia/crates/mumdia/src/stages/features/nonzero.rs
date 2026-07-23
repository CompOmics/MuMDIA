//! Extended feature family: nonzero / zero-ignoring variants.
//!
//! The apex spectral features sample each fragment at the single nearest scan to
//! the apex, so a fragment that peaks one grid scan off reads 0.0 there even when
//! it is present in the peak (~8% of predicted fragments). The trace features
//! include the intra-peak grid zeros. These variants recompute the key features
//! while IGNORING those zeros: apex spectral agreement over the per-fragment
//! peak-MAX (a present fragment is never spuriously 0), and co-elution over only
//! the scans where signal is actually present. Added ALONGSIDE the originals so
//! the classifier can use whichever is more discriminative.
//!
//! Contract: NAMES and values(&Evidence) in matching order/length. See the
//! Evidence struct in stages/features.rs.
use super::Evidence;
use crate::stats::{cosine, pearson, spectral_angle};

pub const NAMES: &[&str] = &[
    "frag_corr_peakmax",
    "frag_cosine_peakmax",
    "spectral_angle_peakmax",
    "frag_corr_matched_nz",
    "frag_cosine_matched_nz",
    "peakmax_apex_gain",
    "n_frag_present_inpeak",
    "frac_frag_present_inpeak",
    "coelution_mean_bothpos",
    "coelution_mean_summpos",
    "ref_corr_nz",
    "profile_cos_nz",
];

pub fn values(e: &Evidence) -> Vec<f64> {
    let k = e.pred.len();
    let np = e.axis.len();

    // Per-fragment peak-max over the elution-peak trace: a fragment present
    // anywhere in the peak is nonzero, so the apex-scan-alignment zero is removed.
    let omax: Vec<f64> = e
        .traces
        .iter()
        .map(|t| t.iter().cloned().fold(0.0f64, f64::max))
        .collect();

    let frag_corr_peakmax = pearson(&omax, &e.pred);
    let frag_cosine_peakmax = cosine(&omax, &e.pred);
    let spectral_angle_peakmax = spectral_angle(&omax, &e.pred);

    // Matched-only (peak-max > 0) apex spectral agreement.
    let (mut om, mut pm) = (Vec::new(), Vec::new());
    for (i, &peakmax) in omax.iter().enumerate().take(k) {
        if peakmax > 0.0 {
            om.push(peakmax);
            pm.push(e.pred[i]);
        }
    }
    let frag_corr_matched_nz = pearson(&om, &pm);
    let frag_cosine_matched_nz = cosine(&om, &pm);

    // How much the single apex scan understates the fragment peak (mean over
    // present fragments). High = fragments peak off the apex scan (grid artifact,
    // or a chimera whose borrowed fragments peak at different times).
    let mut gains = Vec::new();
    for (i, &peakmax) in omax.iter().enumerate().take(k) {
        if peakmax > 0.0 {
            gains.push(((peakmax - e.obs_apex[i]).max(0.0)) / peakmax);
        }
    }
    let peakmax_apex_gain = super::mean(&gains);

    let n_present = omax.iter().filter(|&&x| x > 0.0).count();
    let n_frag_present_inpeak = n_present as f64;
    let frac_frag_present_inpeak = if k > 0 {
        n_present as f64 / k as f64
    } else {
        0.0
    };

    // Summed profile and its nonzero-scan mask.
    let summ: Vec<f64> = (0..np)
        .map(|t| e.traces.iter().map(|tr| tr[t]).sum())
        .collect();
    let sm: Vec<usize> = (0..np).filter(|&t| summ[t] > 0.0).collect();

    // Pairwise co-elution ignoring zeros: over scans where both fragments are
    // present (both-positive), and over scans where any signal is present.
    let mut both = Vec::new();
    let mut summpos = Vec::new();
    for a in 0..e.traces.len() {
        for b in (a + 1)..e.traces.len() {
            let idx: Vec<usize> = (0..np)
                .filter(|&t| e.traces[a][t] > 0.0 && e.traces[b][t] > 0.0)
                .collect();
            if idx.len() >= 2 {
                let xa: Vec<f64> = idx.iter().map(|&t| e.traces[a][t]).collect();
                let xb: Vec<f64> = idx.iter().map(|&t| e.traces[b][t]).collect();
                both.push(pearson(&xa, &xb));
            }
            if sm.len() >= 2 {
                let xa: Vec<f64> = sm.iter().map(|&t| e.traces[a][t]).collect();
                let xb: Vec<f64> = sm.iter().map(|&t| e.traces[b][t]).collect();
                summpos.push(pearson(&xa, &xb));
            }
        }
    }
    let coelution_mean_bothpos = super::mean(&both);
    let coelution_mean_summpos = super::mean(&summpos);

    // Fragment-vs-reference correlation over each fragment's own nonzero scans.
    let mut rc = Vec::new();
    for tr in &e.traces {
        let idx: Vec<usize> = (0..np).filter(|&t| tr[t] > 0.0).collect();
        if idx.len() >= 2 {
            let x: Vec<f64> = idx.iter().map(|&t| tr[t]).collect();
            let r: Vec<f64> = idx.iter().map(|&t| e.ref_profile[t]).collect();
            rc.push(pearson(&x, &r));
        }
    }
    let ref_corr_nz = super::mean(&rc);

    // profile cosine restricted to scans with signal present.
    let (mut num, mut den) = (0.0, 0.0);
    for &t in &sm {
        let w = e.ref_profile[t] * e.ref_profile[t];
        if w <= 0.0 {
            continue;
        }
        let obs_t: Vec<f64> = e.traces.iter().map(|tr| tr[t]).collect();
        num += cosine(&obs_t, &e.pred) * w;
        den += w;
    }
    let profile_cos_nz = if den > 0.0 { num / den } else { 0.0 };

    vec![
        frag_corr_peakmax,
        frag_cosine_peakmax,
        spectral_angle_peakmax,
        frag_corr_matched_nz,
        frag_cosine_matched_nz,
        peakmax_apex_gain,
        n_frag_present_inpeak,
        frac_frag_present_inpeak,
        coelution_mean_bothpos,
        coelution_mean_summpos,
        ref_corr_nz,
        profile_cos_nz,
    ]
}
