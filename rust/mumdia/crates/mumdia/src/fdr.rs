//! Native target-decoy FDR / q-values (PLAN.md Section 4 Stage F, Section 8:
//! DIA-NN no-pi0 estimator `q = n_decoys / max(1,n_targets)`, monotonized).
//! Shared by search-seed and rescore.

/// Compute per-record q-values from (score, is_decoy). Higher score is better.
/// Returns q aligned to the input order.
pub fn target_decoy_q(scores: &[(f64, bool)]) -> Vec<f64> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .0
            .partial_cmp(&scores[a].0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (mut td, mut tt) = (0usize, 0usize);
    let mut fdr_at = vec![1.0f64; n];
    for (rank, &i) in order.iter().enumerate() {
        if scores[i].1 {
            td += 1;
        } else {
            tt += 1;
        }
        fdr_at[rank] = (td.max(0) as f64) / (tt.max(1) as f64);
    }
    // Monotonize from worst-scoring to best so q is non-increasing with score.
    let mut q = vec![1.0f64; n];
    let mut qmin = 1.0f64;
    for rank in (0..n).rev() {
        qmin = qmin.min(fdr_at[rank]);
        q[order[rank]] = qmin;
    }
    q
}

/// Entrapment-calibrated q-values. Higher score is better. `is_entrapment`
/// marks spike-in foreign-proteome PSMs (false by construction); `is_real`
/// marks the sample's own target PSMs. Rows that are neither (decoys) are ranked
/// but enter no count. FDR(t) = `ratio` * n_entrap(>=t) / max(1, n_real(>=t)),
/// where `ratio` = N_real_lib / N_entrap_lib corrects for unequal library sizes.
/// Monotonized from worst to best so q is non-increasing with score. This is the
/// empirical-null analog of `target_decoy_q`: the entrapment population, unlike
/// in-silico decoys, experiences the same chimeric DIA interference as real
/// targets, so the estimate is not optimistic. Returns q aligned to input order.
pub fn entrapment_q(scores: &[f64], is_entrapment: &[bool], is_real: &[bool], ratio: f64) -> Vec<f64> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    // Stable sort: ties keep input order, so q values are deterministic.
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (mut ne, mut nr) = (0usize, 0usize);
    let mut fdr_at = vec![1.0f64; n];
    for (rank, &i) in order.iter().enumerate() {
        if is_entrapment[i] {
            ne += 1;
        } else if is_real[i] {
            nr += 1;
        }
        fdr_at[rank] = (ratio * ne as f64) / (nr.max(1) as f64);
    }
    let mut q = vec![1.0f64; n];
    let mut qmin = 1.0f64;
    for rank in (0..n).rev() {
        qmin = qmin.min(fdr_at[rank]);
        q[order[rank]] = qmin;
    }
    q
}

/// Number of targets at or below the given q threshold.
pub fn count_targets_at_q(q: &[f64], is_decoy: &[bool], threshold: f64) -> usize {
    q.iter()
        .zip(is_decoy)
        .filter(|(qq, d)| !**d && **qq <= threshold)
        .count()
}

/// ln(n!) via summed logs (n small in matched-fragment counts).
pub fn ln_factorial(n: u32) -> f64 {
    let mut s = 0.0;
    for k in 2..=n {
        s += (k as f64).ln();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_separation_gives_zero_q_for_targets() {
        // all targets score above all decoys
        let s = vec![(10.0, false), (9.0, false), (1.0, true), (0.5, true)];
        let q = target_decoy_q(&s);
        // both targets outrank all decoys -> q = 0 for both
        assert!(q[0] < 1e-9 && q[1] < 1e-9);
        assert_eq!(count_targets_at_q(&q, &[false, false, true, true], 0.01), 2);
        // With interleave the top target has q=0
        let s2 = vec![(10.0, false), (9.0, false), (8.0, false), (1.0, true)];
        let q2 = target_decoy_q(&s2);
        assert_eq!(count_targets_at_q(&q2, &[false, false, false, true], 0.34), 3);
    }

    #[test]
    fn entrapment_q_ranks_real_above_spike_in() {
        // Two real targets score highest, then an entrapment, then a real, then
        // entrapment. Real=is_real, entrapment=is_entrapment; a decoy row is
        // ranked but counts toward neither.
        let scores = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0];
        let is_entrap = vec![false, false, true, false, true, false];
        let is_real = vec![true, true, false, true, false, false]; // last row = decoy
        let q = entrapment_q(&scores, &is_entrap, &is_real, 1.0);
        // Top two real targets precede any entrapment -> q = 0.
        assert!(q[0] < 1e-9 && q[1] < 1e-9);
        // At the 3rd-ranked real target (rank 3): 1 entrap / 3 real = 0.333.
        assert!((q[3] - 1.0 / 3.0).abs() < 1e-9);
        // Library-size ratio scales the estimate linearly.
        let q2 = entrapment_q(&scores, &is_entrap, &is_real, 2.0);
        assert!((q2[3] - 2.0 / 3.0).abs() < 1e-9);
        // Determinism: identical inputs give identical output.
        assert_eq!(q, entrapment_q(&scores, &is_entrap, &is_real, 1.0));
    }
}
