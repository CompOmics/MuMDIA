//! Extended feature family: entropy.
//!
//! Spectral-entropy / information-divergence between the sum-normalized
//! observed-apex distribution `o` and the library (predicted) distribution `l`,
//! plus standalone complexity measures. All computations use the convention
//! `0 * ln 0 := 0` and guard every division, `ln`, and reduction so every
//! returned value is finite (0.0 in degenerate cases).
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items in
//! the same order. See the `Evidence` struct in `stages/features.rs`.
use super::Evidence;

pub const NAMES: &[&str] = &[
    "spectral_entropy_similarity",
    "weighted_spectral_entropy_similarity",
    "spectral_entropy_similarity_sqrt",
    "spectral_entropy_similarity_topk",
    "spectral_entropy_similarity_area",
    "jensen_shannon_divergence",
    "jeffreys_divergence",
    "kl_obs_pred",
    "kl_pred_obs",
    "cross_entropy_obs_pred",
    "obs_spectrum_entropy",
    "pred_spectrum_entropy",
    "entropy_diff",
    "entropy_ratio",
    "obs_normalized_entropy",
    "normalized_entropy_diff",
    "residual_spectrum_entropy",
    "entropy_weight_obs",
];

const EPS: f64 = 1e-10;

/// Finite guard: replace NaN/Inf with 0.0.
#[inline]
fn fin(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

/// Clamp a similarity value to [0, 1] after a finite guard.
#[inline]
fn clamp01(x: f64) -> f64 {
    let x = fin(x);
    x.clamp(0.0, 1.0)
}

/// Sum-normalize a non-negative intensity vector. Returns `None` if the sum of
/// finite positive entries is not strictly positive. Negative and non-finite
/// entries are treated as 0.
fn sumnorm(v: &[f64]) -> Option<Vec<f64>> {
    let s: f64 = v.iter().filter(|x| x.is_finite() && **x > 0.0).sum();
    if s > 0.0 {
        Some(
            v.iter()
                .map(|x| if x.is_finite() && *x > 0.0 { x / s } else { 0.0 })
                .collect(),
        )
    } else {
        None
    }
}

/// Shannon entropy (natural log) of a probability vector. Uses `0 ln 0 := 0`.
fn shannon(p: &[f64]) -> f64 {
    let mut h = 0.0;
    for &x in p {
        if x > 0.0 {
            h -= x * x.ln();
        }
    }
    fin(h)
}

/// Spectral-entropy similarity of the sqrt-transformed observed vs predicted
/// intensity vectors. This is exactly the `spectral_entropy_similarity_sqrt`
/// feature (see `values`, item 3), exposed so the extraction acceptance gate
/// (`GateMode::SpectralEntropy`) can threshold the single best target/decoy
/// discriminator found in the full-feature gate search, without duplicating the
/// entropy kernel. `obs` and `pred` are co-indexed by predicted fragment.
pub fn spectral_entropy_similarity_sqrt(obs: &[f64], pred: &[f64]) -> f64 {
    let o: Vec<f64> = obs.iter().map(|x| if *x > 0.0 { x.sqrt() } else { 0.0 }).collect();
    let l: Vec<f64> = pred.iter().map(|x| if *x > 0.0 { x.sqrt() } else { 0.0 }).collect();
    entropy_sim(&o, &l)
}

/// Li spectral-entropy similarity: `1 - (2 H(m) - H(o) - H(l)) / ln 4`, with
/// `m = (o + l) / 2` over sum-normalized `o` and `l`. Returns 0.0 if either
/// input has non-positive mass or the lengths differ. Clamped to [0, 1].
fn entropy_sim(o_raw: &[f64], l_raw: &[f64]) -> f64 {
    if o_raw.len() != l_raw.len() || o_raw.is_empty() {
        return 0.0;
    }
    let (o, l) = match (sumnorm(o_raw), sumnorm(l_raw)) {
        (Some(o), Some(l)) => (o, l),
        _ => return 0.0,
    };
    let ln4 = 4.0_f64.ln();
    if ln4 <= 0.0 {
        return 0.0;
    }
    let m: Vec<f64> = o.iter().zip(&l).map(|(a, b)| 0.5 * (a + b)).collect();
    let sim = 1.0 - (2.0 * shannon(&m) - shannon(&o) - shannon(&l)) / ln4;
    clamp01(sim)
}

/// Li per-spectrum entropy weighting: if the spectrum's Shannon entropy `S < 3`,
/// raise the (normalized) intensities to the power `0.25 + 0.25 S` (else leave
/// them). The result is left un-renormalized; the caller renormalizes.
fn li_weight(v_raw: &[f64]) -> Vec<f64> {
    match sumnorm(v_raw) {
        Some(p) => {
            let s = shannon(&p);
            let w = if s >= 3.0 { 1.0 } else { 0.25 + 0.25 * s };
            p.iter().map(|x| fin(x.powf(w))).collect()
        }
        None => vec![0.0; v_raw.len()],
    }
}

/// Shannon entropy of a raw intensity vector after sum-normalization (0.0 if no
/// mass).
fn spectrum_entropy(v_raw: &[f64]) -> f64 {
    match sumnorm(v_raw) {
        Some(p) => shannon(&p),
        None => 0.0,
    }
}

pub fn values(e: &Evidence) -> Vec<f64> {
    let k = e.pred.len();

    // Guard the shared fragment order: obs_apex must be co-indexed with pred.
    // If lengths disagree or there are no fragments, the battery is degenerate.
    if k == 0 || e.obs_apex.len() != k {
        return vec![0.0; NAMES.len()];
    }

    let o: Vec<f64> = e.obs_apex.clone();
    let l: Vec<f64> = e.pred.clone();

    // 1. spectral_entropy_similarity
    let sim = entropy_sim(&o, &l);

    // 2. weighted_spectral_entropy_similarity
    let wsim = entropy_sim(&li_weight(&o), &li_weight(&l));

    // 3. spectral_entropy_similarity_sqrt
    let o_sqrt: Vec<f64> = o.iter().map(|x| if *x > 0.0 { x.sqrt() } else { 0.0 }).collect();
    let l_sqrt: Vec<f64> = l.iter().map(|x| if *x > 0.0 { x.sqrt() } else { 0.0 }).collect();
    let sim_sqrt = entropy_sim(&o_sqrt, &l_sqrt);

    // 4. spectral_entropy_similarity_topk (top-6 fragments by predicted intensity)
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&a, &b| {
        l[b]
            .partial_cmp(&l[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let topn = idx.len().min(6);
    let o_top: Vec<f64> = idx[..topn].iter().map(|&i| o[i]).collect();
    let l_top: Vec<f64> = idx[..topn].iter().map(|&i| l[i]).collect();
    let sim_topk = entropy_sim(&o_top, &l_top);

    // 5. spectral_entropy_similarity_area (peak-XIC area as observed vector)
    let area: Vec<f64> = (0..k)
        .map(|i| {
            if i < e.traces.len() {
                e.traces[i].iter().filter(|x| x.is_finite()).sum()
            } else {
                0.0
            }
        })
        .collect();
    let sim_area = entropy_sim(&area, &l);

    // Sum-normalized o and l for the divergence family.
    let on = sumnorm(&o);
    let ln = sumnorm(&l);

    // 6. jensen_shannon_divergence
    // 7. jeffreys_divergence (symmetric KL)
    // 8. kl_obs_pred = KL(o || l)
    // 9. kl_pred_obs = KL(l || o)
    // 10. cross_entropy_obs_pred = -sum o ln(l+eps)
    let (mut jsd, mut jeff, mut kl_op, mut kl_po, mut ce) = (0.0, 0.0, 0.0, 0.0, 0.0);
    if let (Some(op), Some(lp)) = (&on, &ln) {
        let (mut sjsd, mut sjeff, mut skl_op, mut skl_po, mut sce) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..k {
            let oi = op[i];
            let li = lp[i];
            let mi = 0.5 * (oi + li);
            if oi > 0.0 && mi > 0.0 {
                sjsd += 0.5 * oi * (oi / mi).ln();
            }
            if li > 0.0 && mi > 0.0 {
                sjsd += 0.5 * li * (li / mi).ln();
            }
            sjeff += (li - oi) * ((li + EPS) / (oi + EPS)).ln();
            if oi > 0.0 {
                skl_op += oi * ((oi + EPS) / (li + EPS)).ln();
                sce += -oi * (li + EPS).ln();
            }
            if li > 0.0 {
                skl_po += li * ((li + EPS) / (oi + EPS)).ln();
            }
        }
        jsd = fin(sjsd);
        jeff = fin(sjeff);
        kl_op = fin(skl_op);
        kl_po = fin(skl_po);
        ce = fin(sce);
    }

    // 11. obs_spectrum_entropy
    let obs_ent = spectrum_entropy(&o);
    // 12. pred_spectrum_entropy
    let pred_ent = spectrum_entropy(&l);
    // 13. entropy_diff
    let ent_diff = fin(obs_ent - pred_ent);
    // 14. entropy_ratio
    let ent_ratio = fin(obs_ent / (pred_ent + EPS));

    // 15. obs_normalized_entropy (Pielou evenness); 0 if n_matched < 2
    let n_matched = o.iter().filter(|x| x.is_finite() && **x > 0.0).count();
    let obs_norm_ent = if n_matched >= 2 {
        fin(obs_ent / (n_matched as f64).ln())
    } else {
        0.0
    };
    // pred normalized entropy (for the diff below)
    let n_pred = l.iter().filter(|x| x.is_finite() && **x > 0.0).count();
    let pred_norm_ent = if n_pred >= 2 {
        fin(pred_ent / (n_pred as f64).ln())
    } else {
        0.0
    };
    // 16. normalized_entropy_diff
    let norm_ent_diff = fin(obs_norm_ent - pred_norm_ent);

    // 17. residual_spectrum_entropy
    let resid_ent = match (&on, &ln) {
        (Some(op), Some(lp)) => {
            let r: Vec<f64> = (0..k).map(|i| (op[i] - lp[i]).abs()).collect();
            spectrum_entropy(&r)
        }
        _ => 0.0,
    };

    // 18. entropy_weight_obs (Li exponent from observed spectral entropy)
    let ent_weight = if obs_ent >= 3.0 { 1.0 } else { fin(0.25 + 0.25 * obs_ent) };

    vec![
        sim,
        wsim,
        sim_sqrt,
        sim_topk,
        sim_area,
        jsd,
        jeff,
        kl_op,
        kl_po,
        ce,
        obs_ent,
        pred_ent,
        ent_diff,
        ent_ratio,
        obs_norm_ent,
        norm_ent_diff,
        resid_ent,
        ent_weight,
    ]
}
