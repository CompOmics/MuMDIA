//! Top-K chromatographic peak enumeration (sensitivity program,
//! docs/11_compete_rescore_fdr.md, backlog P1).
//!
//! The central sensitivity hypothesis is that MuMDIA selects one chromatographic
//! apex too early, so the correct peak is discarded before the scorer ever sees it
//! (docs/20_sensitivity_and_quantification_playbook.md). This module is the
//! non-destructive alternative: given a candidate's consensus elution profile
//! (summed observed fragment intensity per acquisition-scan group, aligned to a
//! monotonic RT axis), it enumerates up to `K` local-maximum peak groups, each with
//! an apex, peak boundaries, an integrated evidence score, and a rank. Downstream
//! code can then compute features for every retained peak and let an out-of-fold
//! peak-selection model choose, instead of committing to one apex up front.
//!
//! This module is intentionally pure and side-effect free so it is cheap to unit
//! test with synthetic chromatograms (see the tests) and carries no dependency on
//! the extraction hot path. `enumerate_peaks(.., k = 1, ..)` returns the single
//! strongest peak group (the global-argmax apex with the same fractional-height
//! boundary walk the features stage uses), so callers can adopt it incrementally.

/// One retained chromatographic peak group for a candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct PeakGroup {
    /// Index of the apex within the input profile.
    pub apex_idx: usize,
    /// Inclusive left boundary index.
    pub start_idx: usize,
    /// Inclusive right boundary index.
    pub end_idx: usize,
    /// Intensity at the apex.
    pub apex_intensity: f32,
    /// Integrated intensity within `[start_idx, end_idx]` (the evidence score).
    pub area: f32,
    /// Rank by evidence (`area`), 0 = strongest. Assigned after sorting.
    pub rank: usize,
}

/// Enumerate up to `k` peak groups from a chromatographic `profile`.
///
/// * `profile` — non-negative intensities per scan group along a monotonic RT axis.
/// * `k` — maximum peaks to return (>= 1; `k == 0` yields an empty vector).
/// * `bound_fraction` — peak-boundary threshold as a fraction of the local apex
///   height (matches `features.bound_peak_fraction`, default 1/3): the walk stops
///   when the profile drops below `bound_fraction * apex` or turns back upward
///   (a valley), whichever comes first.
/// * `min_prominence_frac` — a local maximum is ignored unless its height is at
///   least this fraction of the global maximum, suppressing noise flicker. Use
///   `0.0` to keep every local maximum.
///
/// Peaks are returned strongest-first by integrated `area`, deduplicated so two
/// maxima inside one peak envelope collapse to the stronger one. Determinism: ties
/// break by earlier `apex_idx`.
pub fn enumerate_peaks(
    profile: &[f32],
    k: usize,
    bound_fraction: f32,
    min_prominence_frac: f32,
) -> Vec<PeakGroup> {
    if k == 0 || profile.is_empty() {
        return Vec::new();
    }
    let n = profile.len();
    let global_max = profile.iter().cloned().fold(0.0f32, f32::max);
    if global_max <= 0.0 {
        return Vec::new();
    }
    let prom_floor = min_prominence_frac.max(0.0) * global_max;

    // 1) Local maxima. `i` is a maximum when it is >= both neighbours and strictly
    //    greater than the left neighbour (so a flat plateau registers once, at its
    //    left edge). Edges count as maxima against their single neighbour.
    let mut maxima: Vec<usize> = Vec::new();
    for i in 0..n {
        let v = profile[i];
        if v <= 0.0 || v < prom_floor {
            continue;
        }
        let left_ok = i == 0 || v > profile[i - 1];
        let right_ok = i + 1 == n || v >= profile[i + 1];
        if left_ok && right_ok {
            maxima.push(i);
        }
    }
    if maxima.is_empty() {
        return Vec::new();
    }

    // 2) Boundaries for each maximum by the fractional-height descent walk.
    let mut peaks: Vec<PeakGroup> = maxima
        .iter()
        .map(|&apex| {
            let apex_v = profile[apex];
            let thr = bound_fraction.max(0.0) * apex_v;
            // walk left: stop below threshold or when the profile turns upward
            let mut start = apex;
            while start > 0 {
                let prev = profile[start - 1];
                if prev < thr || prev > profile[start] {
                    break;
                }
                start -= 1;
            }
            // walk right
            let mut end = apex;
            while end + 1 < n {
                let next = profile[end + 1];
                if next < thr || next > profile[end] {
                    break;
                }
                end += 1;
            }
            let area: f32 = profile[start..=end].iter().sum();
            PeakGroup {
                apex_idx: apex,
                start_idx: start,
                end_idx: end,
                apex_intensity: apex_v,
                area,
                rank: 0,
            }
        })
        .collect();

    // 3) Deduplicate maxima that fall inside another (stronger) peak's envelope.
    //    Sort strongest-first by area, then apex intensity, then earliest apex.
    peaks.sort_by(|a, b| {
        b.area
            .partial_cmp(&a.area)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.apex_intensity
                    .partial_cmp(&a.apex_intensity)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.apex_idx.cmp(&b.apex_idx))
    });
    let mut kept: Vec<PeakGroup> = Vec::new();
    for p in peaks {
        let overlaps = kept
            .iter()
            .any(|q| p.apex_idx >= q.start_idx && p.apex_idx <= q.end_idx);
        if !overlaps {
            kept.push(p);
        }
        if kept.len() == k {
            break;
        }
    }
    for (r, p) in kept.iter_mut().enumerate() {
        p.rank = r;
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    const FR: f32 = 1.0 / 3.0;

    #[test]
    fn empty_or_zero_profile_yields_no_peaks() {
        assert!(enumerate_peaks(&[], 5, FR, 0.1).is_empty());
        assert!(enumerate_peaks(&[0.0, 0.0, 0.0], 5, FR, 0.1).is_empty());
    }

    #[test]
    fn single_true_peak() {
        // one clean triangular peak apexing at index 3
        let p = [0.0, 1.0, 4.0, 9.0, 4.0, 1.0, 0.0];
        let peaks = enumerate_peaks(&p, 5, FR, 0.1);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].apex_idx, 3);
        assert_eq!(peaks[0].rank, 0);
        // boundaries descend to >= 1/3 * 9 = 3.0: indices 2..=4 (value 4) are in,
        // 1 and 5 (value 1) are below threshold.
        assert_eq!(peaks[0].start_idx, 2);
        assert_eq!(peaks[0].end_idx, 4);
    }

    #[test]
    fn k1_returns_only_the_strongest() {
        // two peaks; K=1 keeps the strongest (area) one only
        let p = [0.0, 9.0, 0.0, 0.0, 5.0, 0.0];
        let peaks = enumerate_peaks(&p, 1, FR, 0.1);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].apex_idx, 1);
    }

    #[test]
    fn interference_dominant_but_true_peak_retained_with_topk() {
        // A dominant interference peak (broad+tall plateau, apex idx 1, largest
        // integrated area) and a genuine but weaker true peak (apex idx 8). K=1
        // keeps only the dominant interferent and DISCARDS the true peak; K>=2
        // RETAINS the true peak so the scorer can still choose it. This is the core
        // sensitivity behaviour: preserve the correct peak instead of discarding it
        // early (spec 01 §3.1).
        let p = [
            0.0, 20.0, 20.0, 20.0, 0.0, 0.0, 6.0, 8.0, 9.0, 8.0, 6.0, 0.0,
        ];
        let k1 = enumerate_peaks(&p, 1, FR, 0.05);
        assert_eq!(k1.len(), 1);
        assert_eq!(k1[0].apex_idx, 1, "K=1 keeps the dominant interferent");
        assert!(
            !k1.iter().any(|pk| pk.apex_idx == 8),
            "K=1 discards the true peak"
        );
        let k3 = enumerate_peaks(&p, 3, FR, 0.05);
        assert!(
            k3.iter().any(|pk| pk.apex_idx == 8),
            "K>1 must retain the true peak at idx 8"
        );
    }

    #[test]
    fn two_local_maxima_ranked_by_area() {
        // broad peak (apex 2, larger area) vs sharp peak (apex 6, smaller area)
        let p = [2.0, 5.0, 6.0, 5.0, 2.0, 3.0, 7.0, 3.0, 0.0];
        let peaks = enumerate_peaks(&p, 5, FR, 0.1);
        assert!(peaks.len() >= 2);
        // rank 0 is the larger-area (broad) peak around idx 2
        assert_eq!(peaks[0].rank, 0);
        assert_eq!(peaks[0].apex_idx, 2);
        assert!(peaks.iter().any(|pk| pk.apex_idx == 6));
    }

    #[test]
    fn truncated_peak_at_left_edge() {
        // apex at index 0 (peak truncated by the window start)
        let p = [9.0, 6.0, 3.0, 1.0, 0.0];
        let peaks = enumerate_peaks(&p, 5, FR, 0.1);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].apex_idx, 0);
        assert_eq!(peaks[0].start_idx, 0);
    }

    #[test]
    fn prominence_filter_suppresses_noise_flicker() {
        // one real peak (apex 3, height 10) plus a tiny noise bump (height 1)
        let p = [0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 1.0, 0.0];
        // min_prominence_frac 0.2 -> floor 2.0 drops the height-1 bump
        let peaks = enumerate_peaks(&p, 5, FR, 0.2);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].apex_idx, 3);
        // with no prominence filter the noise bump is also returned
        let peaks_all = enumerate_peaks(&p, 5, FR, 0.0);
        assert!(peaks_all.iter().any(|pk| pk.apex_idx == 6));
    }

    #[test]
    fn overlapping_maxima_collapse_to_stronger() {
        // a shoulder (idx 2) on the side of a main peak (idx 4): the shoulder apex
        // falls inside the main peak envelope and must not become a second peak
        let p = [0.0, 3.0, 5.0, 7.0, 9.0, 6.0, 3.0, 0.0];
        let peaks = enumerate_peaks(&p, 5, FR, 0.1);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].apex_idx, 4);
    }

    #[test]
    fn determinism_ties_break_by_earlier_apex() {
        // two identical peaks; equal area -> earlier apex ranks first, stable
        let p = [0.0, 5.0, 0.0, 5.0, 0.0];
        let a = enumerate_peaks(&p, 5, FR, 0.1);
        let b = enumerate_peaks(&p, 5, FR, 0.1);
        assert_eq!(a, b);
        assert_eq!(a[0].apex_idx, 1);
    }
}
