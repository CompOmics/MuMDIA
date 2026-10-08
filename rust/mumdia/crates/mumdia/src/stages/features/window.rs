//! Extended feature family: fixed-window integrated evidence.
//!
//! At low ion counts (single-cell Astral, tens of counts per fragment peak) a fragment
//! is sampled in one scan and missing from the next, so the single apex scan and an
//! elution peak bounded on that noisy trace (often 1-2 scans) both read Poisson noise.
//! These features sum every fragment over a FIXED window of the apex +-k scans of the
//! full extracted axis (k = 1, 2, 3), independent of the detected peak bounds, and score
//! the summed spectrum against the library: agreement, fragment coverage, intensity, and
//! how many scans of the window carry several fragments at once.
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items in the same
//! order. Every value is finite; degenerate cases return 0.0.
use super::entropy::spectral_entropy_similarity_sqrt;
use super::Evidence;
use crate::stats::{cosine, pearson};

const KS: [usize; 3] = [1, 2, 3];

pub const NAMES: &[&str] = &[
    "wf1_cos_sqrt",
    "wf1_pearson",
    "wf1_pearson_sqrt",
    "wf1_nfrag",
    "wf1_predw_cov",
    "wf1_top3_cov",
    "wf1_logsum",
    "wf1_scans_ge2",
    "wf1_scans_ge3",
    "wf1_occ_mean",
    "wf1_entropy_sim",
    "wf2_cos_sqrt",
    "wf2_pearson",
    "wf2_pearson_sqrt",
    "wf2_nfrag",
    "wf2_predw_cov",
    "wf2_top3_cov",
    "wf2_logsum",
    "wf2_scans_ge2",
    "wf2_scans_ge3",
    "wf2_occ_mean",
    "wf2_entropy_sim",
    "wf3_cos_sqrt",
    "wf3_pearson",
    "wf3_pearson_sqrt",
    "wf3_nfrag",
    "wf3_predw_cov",
    "wf3_top3_cov",
    "wf3_logsum",
    "wf3_scans_ge2",
    "wf3_scans_ge3",
    "wf3_occ_mean",
    "wf3_entropy_sim",
    "wf_contrast",
];

const PER_K: usize = 11;

#[inline]
fn fin(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Index of the scan of `axis` nearest `rt` (first on a tie).
fn nearest(axis: &[f64], rt: f64) -> usize {
    let mut best = 0;
    let mut d = f64::INFINITY;
    for (i, &a) in axis.iter().enumerate() {
        let di = (a - rt).abs();
        if di < d {
            d = di;
            best = i;
        }
    }
    best
}

pub fn values(e: &Evidence) -> Vec<f64> {
    let n_f = e.pred.len().min(e.traces_full.len());
    let n_t = e.axis_full.len();
    if n_f == 0 || n_t == 0 || e.traces_full[..n_f].iter().any(|t| t.len() != n_t) {
        return vec![0.0; NAMES.len()];
    }
    let ia = nearest(&e.axis_full, e.apex_rt);
    // Fragments by descending predicted intensity (stable on ties), so "top 3" is the
    // library's three most intense.
    let mut order: Vec<usize> = (0..n_f).collect();
    order.sort_by(|&a, &b| e.pred[b].total_cmp(&e.pred[a]));
    let pred: Vec<f64> = order.iter().map(|&f| e.pred[f].max(0.0)).collect();
    let psum: f64 = pred.iter().sum();
    let pred_sqrt: Vec<f64> = pred.iter().map(|x| x.sqrt()).collect();

    let mut out = Vec::with_capacity(NAMES.len());
    for k in KS {
        let i0 = ia.saturating_sub(k);
        let i1 = (ia + k + 1).min(n_t);
        let summed: Vec<f64> = order
            .iter()
            .map(|&f| e.traces_full[f][i0..i1].iter().map(|x| x.max(0.0)).sum())
            .collect();
        let sq: Vec<f64> = summed.iter().map(|x| x.sqrt()).collect();
        let present: Vec<bool> = summed.iter().map(|&x| x > 0.0).collect();
        let n_present = present.iter().filter(|&&p| p).count();
        let predw_cov = if psum > 0.0 {
            pred.iter()
                .zip(&present)
                .filter(|(_, &p)| p)
                .map(|(x, _)| x)
                .sum::<f64>()
                / psum
        } else {
            0.0
        };
        let top3 = present.iter().take(3).filter(|&&p| p).count();
        let total: f64 = summed.iter().sum();
        let mut ge2 = 0usize;
        let mut ge3 = 0usize;
        for t in i0..i1 {
            let c = order.iter().filter(|&&f| e.traces_full[f][t] > 0.0).count();
            ge2 += usize::from(c >= 2);
            ge3 += usize::from(c >= 3);
        }
        let occ: Vec<f64> = order
            .iter()
            .zip(&present)
            .filter(|(_, &p)| p)
            .map(|(&f, _)| {
                e.traces_full[f][i0..i1]
                    .iter()
                    .filter(|&&x| x > 0.0)
                    .count() as f64
            })
            .collect();
        let occ_mean = if occ.is_empty() {
            0.0
        } else {
            occ.iter().sum::<f64>() / occ.len() as f64
        };
        let ent = if total > 0.0 {
            spectral_entropy_similarity_sqrt(&summed, &pred)
        } else {
            0.0
        };
        let vals = [
            cosine(&sq, &pred_sqrt),
            pearson(&summed, &pred),
            pearson(&sq, &pred_sqrt),
            n_present as f64,
            predw_cov,
            top3 as f64,
            total.ln_1p(),
            ge2 as f64,
            ge3 as f64,
            occ_mean,
            ent,
        ];
        debug_assert_eq!(vals.len(), PER_K);
        out.extend(vals.iter().map(|&v| fin(v)));
    }
    // Contrast: summed signal in apex +-2 against the flanks +-3..+-6.
    let tot_at = |t: usize| -> f64 { (0..n_f).map(|f| e.traces_full[f][t].max(0.0)).sum() };
    let inner: f64 = (ia.saturating_sub(2)..(ia + 3).min(n_t)).map(tot_at).sum();
    let left: f64 = (ia.saturating_sub(6)..ia.saturating_sub(2))
        .map(tot_at)
        .sum();
    let right: f64 = ((ia + 3).min(n_t)..(ia + 7).min(n_t)).map(tot_at).sum();
    out.push(fin(inner.ln_1p() - (left + right).ln_1p()));
    debug_assert_eq!(out.len(), NAMES.len());
    out
}
