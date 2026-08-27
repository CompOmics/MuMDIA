//! `fragindex`: log-space-binned CSR inverted fragment index with an epoch-stamped
//! dense accumulator (docs/06_predict_frag_index_matchers.md). Clean-room reimplementation
//! from the spec; no code or constants copied from Sage/MSFragger.
//!
//! Posting m/z is stored f32 (adequate for MuMDIA's 20-50 ppm regime: f32 ULP is
//! ~0.12 ppm, 200-400x below the tolerance window) and widened to f64 for the
//! canonical [`within_ppm`] verify. The equivalence gate compares against the
//! [`super::naive`] band-join under the SAME f32-stored / f64-verify predicate, so
//! the storage precision is not a source of gate disagreement.
//!
//! Single global index over all candidates. The precursor-window narrowing that a
//! per-block index would give for free is recovered here by binary-searching the
//! candidate-id sub-range WITHIN each probed bin: build scatters candidates in id
//! order, so `post_cand` is ascending within every bin.

use crate::index::Library;
use crate::matchers::binning::LogBins;
use mumdia_core::constants::within_ppm;

/// The CSR inverted fragment index. Structure-of-arrays so the verify hot loop
/// streams only `post_mz`; `post_cand`/`post_int`/`post_frag` are read on a
/// verified hit. Built once per tolerance (seed at `fragment_tol_ppm`, extract at
/// the learned masscal tolerance).
pub struct FragIndex {
    bins: LogBins,
    /// CSR row offsets, length `n_bins + 1`.
    bin_start: Vec<u32>,
    /// Owning candidate id per posting (== dense candidate index, precondition).
    post_cand: Vec<u32>,
    /// Predicted m/z per posting, f32.
    post_mz: Vec<f32>,
    /// Predicted intensity per posting.
    post_int: Vec<f32>,
    /// Candidate-local fragment ordinal per posting.
    post_frag: Vec<u16>,
    /// Precursor m/z indexed by candidate id (ascending); for `candidate_range`.
    prec_mz: Vec<f64>,
    n_cand: usize,
    tol_ppm: f64,
}

impl FragIndex {
    /// Build the CSR index from a loaded library at a fixed tolerance
    /// (docs/06_predict_frag_index_matchers.md, two-pass counting sort).
    /// Deterministic: candidate-order scatter, no parallel sort, no hashing.
    pub fn build(lib: &Library, tol_ppm: f64) -> FragIndex {
        let n_cand = lib.cands.len();
        // Precondition (docs/06_predict_frag_index_matchers.md, CLAUDE.md
        // index.rs:73): candidate_id is dense 0..n_cand so it indexes the
        // accumulator and post_cand directly.
        for (c, cand) in lib.cands.iter().enumerate() {
            assert_eq!(
                cand.candidate_id as usize, c,
                "fragindex requires contiguous candidate_id (id {} at index {c})",
                cand.candidate_id
            );
        }
        let total = lib.frag_mz.len();
        assert!(total <= u32::MAX as usize, "total_frags exceeds u32");

        // m/z range from the library fragments (guard > 0), spec geometry in f64.
        let mut mz_min = f64::INFINITY;
        let mut mz_max = f64::NEG_INFINITY;
        for &mz in &lib.frag_mz {
            if mz < mz_min {
                mz_min = mz;
            }
            if mz > mz_max {
                mz_max = mz;
            }
        }
        if !mz_min.is_finite() || !mz_max.is_finite() {
            mz_min = 1.0;
            mz_max = 2.0;
        }
        let bins = LogBins::new(tol_ppm, mz_min.max(1.0), mz_max.max(mz_min.max(1.0) + 1e-6));

        // pass 1: per-bin occupancy (+1 offset for the counting-sort idiom).
        // Bin each posting by the SAME f32-rounded m/z that the verify uses
        // (post_mz is f32), so the +/-1 probe's one-bin-width proof holds exactly
        // for the stored value and a boundary-straddling within-tol pair is never
        // missed. Binning by the raw f64 while verifying the f32 could place the
        // posting two bins from the peak.
        let mut bin_start = vec![0u32; bins.n_bins + 1];
        for &mz in &lib.frag_mz {
            bin_start[bins.bin(mz as f32 as f64) + 1] += 1;
        }
        // prefix sum -> CSR start offsets.
        for b in 0..bins.n_bins {
            bin_start[b + 1] += bin_start[b];
        }

        // pass 2: scatter, in candidate-id order so within-bin post_cand is ascending.
        let mut post_cand = vec![0u32; total];
        let mut post_mz = vec![0f32; total];
        let mut post_int = vec![0f32; total];
        let mut post_frag = vec![0u16; total];
        let mut cursor: Vec<u32> = bin_start[..bins.n_bins].to_vec();
        for (c, cand) in lib.cands.iter().enumerate() {
            for k in 0..cand.n_frag {
                let gi = cand.frag_start + k;
                let mz = lib.frag_mz[gi];
                let b = bins.bin(mz as f32 as f64); // bin by the stored (f32) value
                let slot = cursor[b] as usize;
                post_cand[slot] = c as u32;
                post_mz[slot] = mz as f32;
                post_int[slot] = lib.frag_int[gi];
                post_frag[slot] = k as u16;
                cursor[b] += 1;
            }
        }

        FragIndex {
            bins,
            bin_start,
            post_cand,
            post_mz,
            post_int,
            post_frag,
            prec_mz: lib.prec_mz.clone(),
            n_cand,
            tol_ppm,
        }
    }

    pub fn n_cand(&self) -> usize {
        self.n_cand
    }

    pub fn tol_ppm(&self) -> f64 {
        self.tol_ppm
    }

    /// Candidate-id half-open range `[lo, hi)` whose precursor m/z lies in the
    /// isolation window `[win_lo, win_hi]` (prec_mz ascending). Matches the
    /// bucketed `Library::candidate_range` semantics.
    #[inline]
    pub fn candidate_range(&self, win_lo: f64, win_hi: f64) -> (u32, u32) {
        let lo = self.prec_mz.partition_point(|&p| p < win_lo) as u32;
        let hi = self.prec_mz.partition_point(|&p| p <= win_hi) as u32;
        (lo, hi)
    }

    /// Probe one experimental peak: for every predicted posting within tolerance
    /// whose candidate lies in `[cand_lo, cand_hi)`, call `f(cid, post_mz_f64,
    /// post_int, post_frag)`. This is the primitive both the seed accumulator and
    /// the extract per-peak claimant loop use (a drop-in for `Library::page_search`).
    ///
    /// Probes bins `bin(peak)-1 ..= bin(peak)+1` (clamped) and verifies each posting
    /// with the exact f64 predicate. Within each bin, `post_cand` is ascending, so
    /// the precursor-window sub-range is found by binary search rather than scanned.
    #[inline]
    pub fn probe_peak<F: FnMut(u32, f64, f32, u16)>(
        &self,
        peak_mz: f64,
        cand_lo: u32,
        cand_hi: u32,
        mut f: F,
    ) {
        if cand_hi <= cand_lo {
            return;
        }
        let b = self.bins.bin(peak_mz);
        let lo_bin = b.saturating_sub(1);
        let hi_bin = (b + 1).min(self.bins.n_bins - 1);
        for nb in lo_bin..=hi_bin {
            let (a, z) = self.narrow_bin(nb, cand_lo, cand_hi);
            self.emit_range(a, z, peak_mz, &mut f);
        }
    }

    /// Probe one peak using a per-window narrowing cache. Identical semantics and
    /// identical callback order to [`FragIndex::probe_peak`] for the `(cand_lo,
    /// cand_hi)` the cache was built with; the only difference is that the two
    /// binary searches per bin are amortized (see [`WindowNarrow`]).
    #[inline]
    pub fn probe_peak_win<F: FnMut(u32, f64, f32, u16)>(
        &self,
        nw: &mut WindowNarrow,
        peak_mz: f64,
        mut f: F,
    ) {
        if nw.cand_hi <= nw.cand_lo {
            return;
        }
        let b = self.bins.bin(peak_mz);
        let lo_bin = b.saturating_sub(1);
        let hi_bin = (b + 1).min(self.bins.n_bins - 1);
        for nb in lo_bin..=hi_bin {
            let (a, z) = match nw.range[nb] {
                (u32::MAX, _) => {
                    let r = self.narrow_bin(nb, nw.cand_lo, nw.cand_hi);
                    nw.range[nb] = (r.0 as u32, r.1 as u32);
                    r
                }
                (a, z) => (a as usize, z as usize),
            };
            self.emit_range(a, z, peak_mz, &mut f);
        }
    }

    /// Sub-range `[a, z)` of bin `nb`'s postings whose candidate lies in
    /// `[cand_lo, cand_hi)`. Within a bin `post_cand` is ascending, so this is a
    /// binary search rather than a scan.
    #[inline]
    fn narrow_bin(&self, nb: usize, cand_lo: u32, cand_hi: u32) -> (usize, usize) {
        let s = self.bin_start[nb] as usize;
        let e = self.bin_start[nb + 1] as usize;
        if e <= s {
            return (s, s);
        }
        let slice = &self.post_cand[s..e];
        let a = s + slice.partition_point(|&c| c < cand_lo);
        let z = s + slice.partition_point(|&c| c < cand_hi);
        (a, z)
    }

    /// Verify each posting in `[a, z)` against the exact f64 tolerance predicate and
    /// emit the survivors.
    #[inline]
    fn emit_range<F: FnMut(u32, f64, f32, u16)>(
        &self,
        a: usize,
        z: usize,
        peak_mz: f64,
        f: &mut F,
    ) {
        for p in a..z {
            let pmz = self.post_mz[p] as f64;
            if within_ppm(pmz, peak_mz, self.tol_ppm) {
                f(self.post_cand[p], pmz, self.post_int[p], self.post_frag[p]);
            }
        }
    }

    /// Build an empty narrowing cache for the candidate window `[cand_lo, cand_hi)`.
    pub fn window_narrow(&self, cand_lo: u32, cand_hi: u32) -> WindowNarrow {
        WindowNarrow {
            cand_lo,
            cand_hi,
            range: vec![(u32::MAX, u32::MAX); self.bins.n_bins],
        }
    }
}

/// Per-isolation-window cache of each fragment bin's `[cand_lo, cand_hi)` posting
/// sub-range.
///
/// `probe_peak` spends two binary searches per probed bin (six per peak) narrowing
/// a bin's postings to the precursor window. Those searches dominate the useful
/// work: an isolation window holds well under 1% of the library's candidates, so a
/// bin of a few dozen postings typically narrows to none or one. Since `cand_lo`
/// and `cand_hi` are fixed for a whole isolation window and every scan of that
/// window revisits the same bins, the searches only need to happen once per
/// `(window, bin)` instead of once per peak.
///
/// Entries are filled lazily, so a window only pays for the bins its peaks actually
/// reach. `u32::MAX` marks "not yet computed"; it cannot collide with a real posting
/// index because `bin_start` is itself `u32`, so an index that large could not be
/// represented in the first place.
///
/// Cost is 8 bytes per bin (order of 1 MB for a 20 ppm index over the usual
/// fragment m/z range), held by one worker for the duration of one window.
pub struct WindowNarrow {
    cand_lo: u32,
    cand_hi: u32,
    range: Vec<(u32, u32)>,
}

/// Epoch-stamped dense accumulator for the seed's fused `(count, obs_sum)` semiring
/// (docs/06_predict_frag_index_matchers.md). `obs_sum` sums the OBSERVED peak
/// intensity per matched posting (predicted intensity deliberately dropped,
/// reproducing the seed's existing `_pi` discard); `count` is the per-posting match
/// count. Reused across all scans of a block; the accumulator is reset lazily via
/// `epoch`, so only touched candidates are ever written or read.
pub struct SeedScratch {
    count: Vec<u32>,
    obs_sum: Vec<f64>,
    stamp: Vec<u32>,
    touched: Vec<u32>,
    epoch: u32,
    /// `candidate_id` that maps to slot 0. The arrays are indexed WINDOW-RELATIVE, so
    /// they only need to span the widest isolation window rather than the whole library:
    /// sized by `n_cand` they cost 16 B x n_cand PER rayon worker (877 MB per worker on
    /// the profiled 54.8M-candidate library), almost all of it never touched because a
    /// worker only ever sees candidates inside one window.
    base: u32,
}

impl SeedScratch {
    /// `cap` is the expected maximum candidate-window width, not the library size.
    /// Passing a smaller value is safe: the arrays grow on demand.
    pub fn new(cap: usize) -> SeedScratch {
        SeedScratch {
            count: vec![0; cap],
            obs_sum: vec![0.0; cap],
            stamp: vec![0; cap], // 0 is never a live epoch (epoch increments before scan 1)
            touched: Vec::new(),
            epoch: 0,
            base: 0,
        }
    }

    /// Ensure the window-relative arrays span `width` slots.
    fn ensure(&mut self, width: usize) {
        if self.count.len() < width {
            self.count.resize(width, 0);
            self.obs_sum.resize(width, 0.0);
            // New slots must not appear stamped for the current epoch.
            self.stamp.resize(width, 0);
        }
    }

    /// Accumulate one scan's peaks over the candidate window. Peaks must already be
    /// in the caller's fixed order (e.g. m/z ascending, or the top-N re-sorted
    /// order) so `obs_sum` is summed deterministically. After the call, `touched()`
    /// lists the hit candidates and `count`/`obs_sum` hold their values.
    pub fn accumulate(
        &mut self,
        idx: &FragIndex,
        peaks: &[(f64, f32)],
        cand_lo: u32,
        cand_hi: u32,
    ) {
        self.epoch += 1;
        self.touched.clear();
        // Index relative to this window's first candidate.
        self.base = cand_lo;
        self.ensure((cand_hi.saturating_sub(cand_lo)) as usize + 1);
        let epoch = self.epoch;
        let base = self.base;
        for &(mz, inten) in peaks {
            idx.probe_peak(mz, cand_lo, cand_hi, |cid, _pmz, _pint, _pfrag| {
                let cc = (cid - base) as usize;
                if self.stamp[cc] != epoch {
                    self.stamp[cc] = epoch;
                    self.count[cc] = 0;
                    self.obs_sum[cc] = 0.0;
                    self.touched.push(cid);
                }
                self.count[cc] += 1;
                self.obs_sum[cc] += inten as f64;
            });
        }
    }

    /// Candidates hit in the last `accumulate`, in first-touch (probe) order.
    /// Callers that need determinism across a float reduction sort this first.
    pub fn touched(&self) -> &[u32] {
        &self.touched
    }

    /// Valid only for candidate ids from the most recent [`SeedScratch::accumulate`]
    /// window (which is what [`SeedScratch::touched`] returns).
    #[inline]
    pub fn count(&self, cid: u32) -> u32 {
        self.count[(cid - self.base) as usize]
    }

    /// See [`SeedScratch::count`] for the validity window.
    #[inline]
    pub fn obs_sum(&self, cid: u32) -> f64 {
        self.obs_sum[(cid - self.base) as usize]
    }
}

/// Score one scan under both the Count and Dot semirings
/// (docs/06_predict_frag_index_matchers.md), returning `(candidate_id, count, dot)`
/// for every touched candidate. Used by the equivalence gate against
/// [`super::naive`]. Dot = sum over matched postings of
/// `predicted_intensity * peak_intensity` (both widened to f64).
pub fn score_scan_count_dot(
    idx: &FragIndex,
    peaks: &[(f64, f32)],
    cand_lo: u32,
    cand_hi: u32,
) -> Vec<(u32, u32, f64)> {
    use std::collections::HashMap;
    let mut acc: HashMap<u32, (u32, f64)> = HashMap::new();
    for &(mz, inten) in peaks {
        idx.probe_peak(mz, cand_lo, cand_hi, |cid, _pmz, pint, _pfrag| {
            let e = acc.entry(cid).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += pint as f64 * inten as f64;
        });
    }
    let mut out: Vec<(u32, u32, f64)> = acc.into_iter().map(|(c, (n, d))| (c, n, d)).collect();
    out.sort_by_key(|r| r.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Candidate, Library};
    use crate::matchers::naive;

    /// Build a tiny in-memory Library from per-candidate (fragments, precursor_mz).
    /// precursor_mz values must be passed ascending (candidate_range precondition).
    fn lib_from(cands: &[(Vec<(f64, f32)>, f64)]) -> Library {
        let mut frag_mz = Vec::new();
        let mut frag_int = Vec::new();
        let mut frag_name_id: Vec<u16> = Vec::new();
        let mut prec_mz = Vec::new();
        let mut cs = Vec::new();
        for (i, (frags, pmz)) in cands.iter().enumerate() {
            let start = frag_mz.len();
            for &(mz, int) in frags {
                frag_mz.push(mz);
                frag_int.push(int);
                frag_name_id.push(0);
            }
            cs.push(Candidate {
                candidate_id: i as u32,
                peptidoform_id: i as u32,
                base_peptide_id: i as u32,
                peptidoform: String::new(),
                charge: 2,
                precursor_mz: *pmz,
                predicted_irt: 0.0,
                is_decoy: false,
                protein: String::new(),
                frag_start: start,
                n_frag: frags.len(),
            });
            prec_mz.push(*pmz);
        }
        Library {
            cands: cs,
            frag_mz,
            frag_int,
            frag_name_id,
            frag_name_dict: vec!["f".to_string()],
            idx_mz: Vec::new(),
            idx_cid: Vec::new(),
            idx_int: Vec::new(),
            bucket_min: Vec::new(),
            bucket_size: 1,
            prec_mz,
        }
    }

    #[test]
    fn probe_peak_win_matches_probe_peak_callback_for_callback() {
        // The cached probe must be a drop-in: same postings, same order, same values,
        // including on repeat probes (which is where the cache is actually exercised)
        // and on empty / out-of-window bins.
        let tol = 20.0;
        let mut cands: Vec<(Vec<(f64, f32)>, f64)> = Vec::new();
        let mut pmz = 400.0f64;
        for i in 0..40 {
            let base = 300.0 + (i as f64) * 17.3;
            cands.push((
                vec![
                    (base, 1.0 + i as f32),
                    (base + 0.0004, 0.5),  // same bin as `base`
                    (base * 1.00001, 0.7), // adjacent bin
                    (900.0 + i as f64, 0.3),
                ],
                pmz,
            ));
            pmz += 3.0;
        }
        let lib = lib_from(&cands);
        let idx = FragIndex::build(&lib, tol);

        for &(win_lo, win_hi) in &[(400.0, 520.0), (0.0, 1e9), (401.5, 402.5), (1e9, 2e9)] {
            let (lo, hi) = idx.candidate_range(win_lo, win_hi);
            let mut nw = idx.window_narrow(lo, hi);
            // Two passes over the same peaks: pass 2 reads the filled cache.
            for _pass in 0..2 {
                let mut probes: Vec<f64> = Vec::new();
                for i in 0..40 {
                    let base = 300.0 + (i as f64) * 17.3;
                    probes.push(base);
                    probes.push(base * (1.0 + tol * 1e-6 * 0.98)); // just inside tol
                    probes.push(base * (1.0 + tol * 1e-6 * 4.0)); // outside tol
                    probes.push(900.0 + i as f64);
                }
                probes.push(50.0); // below the indexed range
                probes.push(5000.0); // above it
                for &q in &probes {
                    let mut a: Vec<(u32, u64, u32, u16)> = Vec::new();
                    idx.probe_peak(q, lo, hi, |c, m, it, fr| {
                        a.push((c, m.to_bits(), it.to_bits(), fr))
                    });
                    let mut b: Vec<(u32, u64, u32, u16)> = Vec::new();
                    idx.probe_peak_win(&mut nw, q, |c, m, it, fr| {
                        b.push((c, m.to_bits(), it.to_bits(), fr))
                    });
                    assert_eq!(
                        a, b,
                        "cached probe diverged at q={q} window=({win_lo},{win_hi})"
                    );
                }
            }
        }
    }

    // docs/06_predict_frag_index_matchers.md: fragindex == naive at K=C, same
    // predicate.
    #[test]
    fn equivalence_gate_vs_naive() {
        let tol = 20.0;
        // a spread of candidates with overlapping and distinct fragment m/z
        let lib = lib_from(&[
            (vec![(200.10, 1.0), (500.20, 2.0), (800.30, 0.5)], 400.0),
            (vec![(500.205, 3.0), (900.40, 1.0)], 405.0), // 500.20 vs 500.205 within 20ppm
            (vec![(1200.50, 2.0), (1200.51, 1.5)], 410.0), // two near-identical frags
            (vec![(1500.60, 1.0)], 800.0),
        ]);
        let idx = FragIndex::build(&lib, tol);
        // peaks that hit several candidates, some within tol of multiple frags
        let peaks = vec![
            (500.202f64, 10.0f32),
            (1200.505f64, 5.0f32), // within tol of BOTH 1200.50 and 1200.51 -> count 2 for cand 2
            (900.40f64, 4.0f32),
            (200.10f64, 1.0f32),
            (1500.605f64, 3.0f32),
        ];
        let (lo, hi) = (0u32, lib.cands.len() as u32);
        let fi = score_scan_count_dot(&idx, &peaks, lo, hi);
        let nv = naive::score_scan_count_dot(&lib, &peaks, lo, hi, tol);
        assert_eq!(fi.len(), nv.len(), "matched-candidate set size differs");
        for ((fc, fn_, fd), (nc, nn, nd)) in fi.iter().zip(nv.iter()) {
            assert_eq!(fc, nc, "candidate id mismatch");
            assert_eq!(
                fn_, nn,
                "count (matched-posting multiplicity) mismatch for cand {fc}"
            );
            assert!(
                (fd - nd).abs() <= 1e-6 * (1.0 + fd.abs().max(nd.abs())),
                "dot mismatch for cand {fc}: {fd} vs {nd}"
            );
        }
        // and the double-frag candidate really counted both postings
        let c2 = fi.iter().find(|r| r.0 == 2).expect("cand 2 hit");
        assert_eq!(c2.1, 2, "cand 2 must count both near-identical fragments");
    }

    #[test]
    fn two_frags_one_peak_counts_both() {
        let tol = 20.0;
        let lib = lib_from(&[(vec![(700.00, 2.0), (700.005, 3.0)], 400.0)]);
        let idx = FragIndex::build(&lib, tol);
        let peaks = vec![(700.002f64, 10.0f32)]; // within tol of both fragments
        let r = score_scan_count_dot(&idx, &peaks, 0, 1);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].1, 2, "both fragment-peak pairs counted (no dedup)");
        // dot = (2.0 + 3.0) * 10.0
        assert!((r[0].2 - 50.0).abs() < 1e-9);
    }

    #[test]
    fn epoch_reset_no_carry_across_scans() {
        let tol = 20.0;
        let lib = lib_from(&[(vec![(600.00, 1.0)], 400.0), (vec![(800.00, 1.0)], 405.0)]);
        let idx = FragIndex::build(&lib, tol);
        let mut sc = SeedScratch::new(idx.n_cand());
        // scan 1 hits cand 0 only
        sc.accumulate(&idx, &[(600.00, 5.0)], 0, 2);
        assert_eq!(sc.touched().len(), 1);
        assert_eq!(sc.count(0), 1);
        // scan 2 hits cand 1 only; cand 0 must NOT carry a score
        sc.accumulate(&idx, &[(800.00, 7.0)], 0, 2);
        assert_eq!(sc.touched(), &[1u32]);
        assert_eq!(sc.count(1), 1);
        assert!((sc.obs_sum(1) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn tolerance_edge_inside_matches_outside_does_not() {
        let tol = 10.0;
        let lib = lib_from(&[(vec![(1000.0, 1.0)], 400.0)]);
        let idx = FragIndex::build(&lib, tol);
        let inside = 1000.0 + 10.0 * 1e-6 * 1000.0 * 0.99; // just inside 10 ppm
        let outside = 1000.0 + 10.0 * 1e-6 * 1000.0 * 1.01; // just outside
        assert_eq!(score_scan_count_dot(&idx, &[(inside, 1.0)], 0, 1).len(), 1);
        assert_eq!(score_scan_count_dot(&idx, &[(outside, 1.0)], 0, 1).len(), 0);
    }

    #[test]
    fn precursor_window_gate_excludes_out_of_range() {
        let tol = 20.0;
        // three candidates at prec_mz 400/500/600; a peak matches a frag in each
        let lib = lib_from(&[
            (vec![(700.0, 1.0)], 400.0),
            (vec![(700.0, 1.0)], 500.0),
            (vec![(700.0, 1.0)], 600.0),
        ]);
        let idx = FragIndex::build(&lib, tol);
        let (lo, hi) = idx.candidate_range(450.0, 550.0); // only cand 1 (prec 500)
        assert_eq!((lo, hi), (1, 2));
        let r = score_scan_count_dot(&idx, &[(700.0, 1.0)], lo, hi);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 1);
    }

    #[test]
    fn probe_finds_within_tol_across_bin_boundaries() {
        // Sweep fragment m/z across the range; for each, a peak just inside tol on
        // both the low and high side must be found by probe_peak. This exercises many
        // bin positions including boundary straddles and guards the build-bins-f32 /
        // verify-f32 consistency (a raw-f64 bin would miss a boundary-straddling pair).
        let tol = 20.0;
        let mut mz = 250.0f64;
        while mz < 1900.0 {
            let lib = lib_from(&[(vec![(mz, 1.0)], 400.0)]);
            let idx = FragIndex::build(&lib, tol);
            for sign in [-1.0f64, 1.0] {
                let peak = mz + sign * tol * 1e-6 * mz * 0.98; // just inside tolerance
                let r = score_scan_count_dot(&idx, &[(peak, 1.0)], 0, 1);
                assert_eq!(
                    r.len(),
                    1,
                    "missed within-tol peak: frag mz={mz} sign={sign}"
                );
            }
            mz *= 1.0009;
        }
    }
}
