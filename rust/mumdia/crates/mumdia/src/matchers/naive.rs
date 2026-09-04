//! Naive band-join reference (fragindex_spec Section 4.1): obviously-correct
//! ground truth for the equivalence gate. O(C * frags * peaks); slow, not used in
//! production. For each candidate in the window and each of its fragments, count
//! every peak within tolerance and accumulate Count and Dot per matched pair (no
//! de-duplication, fragindex_spec Section 1.4).
//!
//! Predicted m/z is rounded to f32 before the predicate, matching `FragIndex`'s
//! f32 posting storage, so the gate compares the two matchers under an identical
//! predicate and can require exact Count equality.

use crate::index::Library;
use mumdia_core::constants::within_ppm;

/// `(candidate_id, count, dot)` for every candidate in `[cand_lo, cand_hi)` with at
/// least one matched pair, sorted by candidate id.
pub fn score_scan_count_dot(
    lib: &Library,
    peaks: &[(f64, f32)],
    cand_lo: u32,
    cand_hi: u32,
    tol_ppm: f64,
) -> Vec<(u32, u32, f64)> {
    let mut out = Vec::new();
    for c in cand_lo..cand_hi {
        let cand = &lib.cands[c as usize];
        let mut count = 0u32;
        let mut dot = 0.0f64;
        for k in 0..cand.n_frag {
            let gi = cand.frag_start + k;
            let fmz = lib.frag_mz[gi] as f64; // library stores f32, as FragIndex matches on
            let fint = lib.frag_int[gi];
            for &(pmz, pint) in peaks {
                if within_ppm(fmz, pmz, tol_ppm) {
                    count += 1;
                    dot += fint as f64 * pint as f64;
                }
            }
        }
        if count > 0 {
            out.push((c, count, dot));
        }
    }
    out
}
