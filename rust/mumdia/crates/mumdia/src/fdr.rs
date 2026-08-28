//! Native target-decoy FDR / q-values (docs/11_compete_rescore_fdr.md): no-pi0
//! estimator `q = (n_decoys + 1) / max(1, n_targets)`, monotonized, with tied
//! scores collapsed to a single block q. Shared by search-seed and rescore.

/// Rank key for the q-value kernels: finite scores pass through, non-finite ones
/// (NaN, +/-inf) become the worst possible key.
///
/// Two reasons this exists rather than trusting the caller. First, a NaN score used to
/// make the tied-block walk below spin forever, because the walk advances on
/// `score == s` and `NaN == NaN` is false, so a garbage feature value in an
/// unvalidated caller (`rescoring.rs` train scores, `search_seed.rs` hyperscore) turned
/// into a silent hang in a long batch job rather than an error. Second, mapping to
/// `NEG_INFINITY` rather than sorting NaN natively is the conservative direction: under
/// `total_cmp` a positive NaN sorts *above* every real score and would rank first.
/// After this mapping every key is finite or `NEG_INFINITY`, so `==` is well-behaved and
/// `total_cmp` is a genuine total order, which also removes the `sort_by`
/// "comparison function does not correctly implement a total order" panic hazard that
/// `unwrap_or(Equal)` on NaN created.
#[inline]
fn rank_key(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        f64::NEG_INFINITY
    }
}

/// Compute per-record q-values from (score, is_decoy). Higher score is better.
/// Returns q aligned to the input order.
pub fn target_decoy_q(scores: &[(f64, bool)]) -> Vec<f64> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let key: Vec<f64> = scores.iter().map(|&(s, _)| rank_key(s)).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| key[b].total_cmp(&key[a]));
    let (mut td, mut tt) = (0usize, 0usize);
    let mut fdr_at = vec![1.0f64; n];
    // Walk in score order, processing tied-score blocks together so every PSM in
    // a block gets the same FDR (its within-tie order is arbitrary and must not
    // change the q). Numerator uses `n_decoys + 1` (the standard conservative
    // target-decoy estimate); the bare `n_decoys / n_targets` is optimistic in
    // the low-count regime.
    let mut rank = 0usize;
    while rank < n {
        let s = key[order[rank]];
        let mut end = rank;
        while end < n && key[order[end]] == s {
            if scores[order[end]].1 {
                td += 1;
            } else {
                tt += 1;
            }
            end += 1;
        }
        let f = (td as f64 + 1.0) / (tt.max(1) as f64);
        for value in fdr_at.iter_mut().take(end).skip(rank) {
            *value = f;
        }
        rank = end;
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
/// but enter no count. FDR(t) = (`ratio` * n_entrap(>=t) + 1) / max(1, n_real(>=t)),
/// where `ratio` = N_real_lib / N_entrap_lib corrects for unequal library sizes and
/// the `+1` is the conservative finite-sample pseudocount (as in `target_decoy_q`).
/// Monotonized from worst to best so q is non-increasing with score. This is the
/// empirical-null analog of `target_decoy_q`: the entrapment population, unlike
/// in-silico decoys, experiences the same chimeric DIA interference as real
/// targets, so the estimate is not optimistic. Returns q aligned to input order.
pub fn entrapment_q(
    scores: &[f64],
    is_entrapment: &[bool],
    is_real: &[bool],
    ratio: f64,
) -> Vec<f64> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let key: Vec<f64> = scores.iter().map(|&s| rank_key(s)).collect();
    let mut order: Vec<usize> = (0..n).collect();
    // Stable sort: ties keep input order, so q values are deterministic. Keyed on
    // `rank_key` so a non-finite score cannot hang the tied-block walk below.
    order.sort_by(|&a, &b| key[b].total_cmp(&key[a]));
    let (mut ne, mut nr) = (0usize, 0usize);
    let mut fdr_at = vec![1.0f64; n];
    // Process tied-score blocks together so every row in a block gets the same
    // FDP regardless of its arbitrary within-tie order (determinism,
    // docs/14_build_test_deploy_gotchas.md). Mirrors the tied-block walk in
    // `target_decoy_q`.
    let mut rank = 0usize;
    while rank < n {
        let s = key[order[rank]];
        let mut end = rank;
        while end < n && key[order[end]] == s {
            let i = order[end];
            if is_entrapment[i] {
                ne += 1;
            } else if is_real[i] {
                nr += 1;
            }
            end += 1;
        }
        let f = (ratio * ne as f64 + 1.0) / (nr.max(1) as f64);
        for value in fdr_at.iter_mut().take(end).skip(rank) {
            *value = f;
        }
        rank = end;
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

/// Validate that every PSM label is a known class. An unknown or malformed
/// label must not silently count as a target (docs/18_findings_and_decisions.md):
/// the target-decoy null depends on exact labeling. Entrapment status is derived
/// from the protein accession (see `classify_entrapment`), not the label, so the
/// only valid label values here are "target" and "decoy".
pub fn validate_labels(labels: &[String]) -> anyhow::Result<()> {
    for l in labels {
        if l != "target" && l != "decoy" {
            anyhow::bail!("unknown PSM label {l:?}; expected \"target\" or \"decoy\"");
        }
    }
    Ok(())
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
    fn perfect_separation_q_is_conservative_plus_one() {
        // all targets score above all decoys; with the (n_decoys+1)/n_targets
        // estimator the best targets get q = 1/n_targets (not 0).
        let s = vec![(10.0, false), (9.0, false), (1.0, true), (0.5, true)];
        let q = target_decoy_q(&s);
        // 2 targets, 0 decoys ranked above them -> q = (0+1)/2 = 0.5 for both
        assert!((q[0] - 0.5).abs() < 1e-9 && (q[1] - 0.5).abs() < 1e-9);
        assert_eq!(count_targets_at_q(&q, &[false, false, true, true], 0.5), 2);
        // 3 targets above 1 decoy -> best target q = (0+1)/3 = 1/3
        let s2 = vec![(10.0, false), (9.0, false), (8.0, false), (1.0, true)];
        let q2 = target_decoy_q(&s2);
        assert_eq!(
            count_targets_at_q(&q2, &[false, false, false, true], 0.34),
            3
        );
    }

    #[test]
    fn tied_scores_share_one_q() {
        // three PSMs at the same score must all receive the same q regardless of
        // their arbitrary within-tie order (target/decoy interleave in a block).
        let s = vec![(5.0, false), (5.0, true), (5.0, false)];
        let q = target_decoy_q(&s);
        assert!((q[0] - q[1]).abs() < 1e-12 && (q[1] - q[2]).abs() < 1e-12);
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
        // +1 finite-sample pseudocount: raw FDP walk (ratio=1) is
        // [1, .5, 1, 2/3, 1, 1], monotonized worst->best to [.5, .5, 2/3, 2/3, 1, 1].
        // Even the top real targets are not q=0.
        assert!((q[0] - 0.5).abs() < 1e-9 && (q[1] - 0.5).abs() < 1e-9);
        // At the 3rd-ranked real target: (1 entrap + 1) / 3 real = 2/3.
        assert!((q[3] - 2.0 / 3.0).abs() < 1e-9);
        // A larger library-size ratio inflates the estimate (more conservative).
        let q2 = entrapment_q(&scores, &is_entrap, &is_real, 2.0);
        assert!(q2[3] >= q[3]);
        // Determinism: identical inputs give identical output.
        assert_eq!(q, entrapment_q(&scores, &is_entrap, &is_real, 1.0));
    }

    #[test]
    fn entrapment_q_tied_scores_share_one_q() {
        // Tied scores must all receive the same q regardless of within-tie order.
        let scores = vec![5.0, 5.0, 5.0];
        let is_entrap = vec![false, true, false];
        let is_real = vec![true, false, true];
        let q = entrapment_q(&scores, &is_entrap, &is_real, 1.0);
        assert!((q[0] - q[1]).abs() < 1e-12 && (q[1] - q[2]).abs() < 1e-12);
    }

    #[test]
    fn non_finite_scores_terminate_and_rank_last() {
        // Regression: the tied-block walk advances on `score == s`, and `NaN == NaN` is
        // false, so a NaN score made both kernels spin forever. `rescore.rs` validates
        // finiteness before calling in, but `rescoring.rs` (train scores) and
        // `search_seed.rs` (raw hyperscore) do not, so the failure mode was a silent
        // hang in a long batch job. This test would not terminate before the fix.
        // Four finite targets, then a +inf decoy and a NaN target. Enough finite targets
        // that q is informative: with only one the (n_decoys+1)/n_targets pseudocount
        // pins it at 1.0 and the test could not discriminate.
        let s = vec![
            (10.0, false),
            (9.0, false),
            (8.0, false),
            (7.0, false),
            (f64::INFINITY, true),
            (f64::NAN, false),
        ];
        let q = target_decoy_q(&s);
        assert_eq!(q.len(), 6);
        assert!(q.iter().all(|v| v.is_finite()), "q must be finite: {q:?}");
        // Every non-finite score maps to the worst key, so the +inf decoy is ranked last
        // rather than first. That is the conservative direction and the reason for not
        // sorting NaN natively: under `total_cmp` a positive NaN outranks every real
        // score, which would have put a garbage row at the top of the list.
        for (i, v) in q.iter().take(4).enumerate() {
            assert!(
                *v <= 0.25 + 1e-12,
                "finite target {i} should keep q = 1/4, got {v} (a non-finite score                  outranking it would raise this): {q:?}"
            );
        }
    }

    #[test]
    fn entrapment_q_terminates_on_non_finite_scores() {
        let scores = vec![f64::NAN, 5.0, 3.0, f64::INFINITY];
        let is_entrap = vec![false, false, true, false];
        let is_real = vec![true, true, false, true];
        let q = entrapment_q(&scores, &is_entrap, &is_real, 1.0);
        assert_eq!(q.len(), 4);
        assert!(
            q.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0),
            "{q:?}"
        );
    }
}
