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
use rayon::prelude::*;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};

/// Postings per chunk below which a build does not bother splitting: each chunk carries a
/// histogram of every bin, so a chunk must be worth more than its histogram.
const MIN_CHUNK_POSTINGS: usize = 1 << 16;

/// `n` zeroed atomics, allocated and touched in parallel (the pages are first written here,
/// so a serial fill would be a serial page-fault pass over the whole posting array).
fn atomic_u32_zeroed(n: usize) -> Vec<AtomicU32> {
    (0..n).into_par_iter().map(|_| AtomicU32::new(0)).collect()
}

/// The CSR inverted fragment index. Structure-of-arrays so the verify hot loop
/// decides on `post_mz` alone and `post_cand`/`post_int`/`post_frag` are wanted only
/// where a posting verifies (their loads sit under the predicate's branch, so the
/// compiler is free to sink them there). Built once per tolerance (seed at
/// `fragment_tol_ppm`, extract at the learned masscal tolerance).
pub struct FragIndex {
    bins: LogBins,
    /// CSR row offsets, length `n_bins + 1`.
    bin_start: Vec<u32>,
    /// Owning candidate id per posting (== dense candidate index, precondition).
    post_cand: Vec<u32>,
    /// Predicted m/z per posting, f32.
    post_mz: Vec<f32>,
    /// Predicted intensity per posting. Empty on an m/z-only index
    /// ([`FragIndex::build_mz_only`]).
    post_int: Vec<f32>,
    /// Candidate-local fragment ordinal per posting. Empty on an m/z-only index.
    post_frag: Vec<u16>,
    /// Precursor m/z indexed by candidate id (ascending); for `candidate_range`. The
    /// library's own array, shared rather than copied.
    prec_mz: std::sync::Arc<[f64]>,
    n_cand: usize,
    tol_ppm: f64,
}

impl FragIndex {
    /// Build the CSR index from a loaded library at a fixed tolerance
    /// (docs/06_predict_frag_index_matchers.md, two-pass counting sort).
    ///
    /// Deterministic and parallel, with arrays bit-identical to the serial counting sort
    /// ([`FragIndex::build_serial`], kept under `cfg(test)` as the reference):
    ///
    /// - the m/z range is a parallel min/max over the finite fragment m/z, which is exact;
    /// - the fragments are cut into chunks at candidate boundaries (the library is CSR in
    ///   candidate order, so a chunk is a contiguous run of candidates), and pass 1 counts
    ///   one bin histogram per chunk, concurrently;
    /// - chunk `j`'s cursor in bin `b` starts at `bin_start[b]` plus the counts of chunks
    ///   `0..j` in that bin. Every chunk therefore owns disjoint slots, and within a bin
    ///   the chunks follow each other in candidate order, so `post_cand` is ascending
    ///   within every bin exactly as the serial candidate-order scatter leaves it;
    /// - pass 2 scatters every chunk concurrently. Safe Rust: the posting arrays are
    ///   atomics during the scatter, written with relaxed stores (plain moves on x86 and
    ///   ARM; every slot is written exactly once, by one chunk), and are turned into plain
    ///   arrays in place afterwards. Binning calls the same [`LogBins::bin`] as the serial
    ///   build, so m/z on a bin edge, below the range, above it (the top bin) and NaN
    ///   (bin 0) land where they always did.
    pub fn build(lib: &Library, tol_ppm: f64) -> FragIndex {
        assert!(
            !lib.fragment_payload_released(),
            "FragIndex::build needs the library's fragment payload (predicted intensities); \
             an m/z-only library indexes with FragIndex::build_mz_only"
        );
        Self::build_chunked(lib, tol_ppm, None, true)
    }

    /// The seed's index: postings of candidate and m/z only, no predicted intensity and no
    /// fragment ordinal (6 bytes per posting less). The seed never reads either -- its
    /// accumulator counts matches and sums OBSERVED intensity
    /// ([`SeedScratch::accumulate`], through [`FragIndex::probe_peak_cand`]) -- and this is
    /// what an m/z-only library ([`Library::load_mz_only`]) can be indexed with. The
    /// geometry, `bin_start`, `post_cand` and `post_mz` are the full index's bit for bit.
    ///
    /// The entry points that hand out a posting's intensity or ordinal
    /// ([`FragIndex::probe_peak`], [`FragIndex::window_narrow`]) refuse an index built this
    /// way rather than returning zeros.
    ///
    /// [`Library::load_mz_only`]: crate::index::Library::load_mz_only
    pub fn build_mz_only(lib: &Library, tol_ppm: f64) -> FragIndex {
        Self::build_chunked(lib, tol_ppm, None, false)
    }

    /// Whether the postings carry their intensity and ordinal (a [`FragIndex::build`]
    /// index) or not ([`FragIndex::build_mz_only`]).
    pub fn has_payload(&self) -> bool {
        self.post_int.len() == self.post_mz.len()
    }

    /// [`FragIndex::build`] with an explicit chunk count (tests drive many chunks over a
    /// small library through this); `None` picks it. `payload` false builds the m/z-only
    /// index.
    fn build_chunked(
        lib: &Library,
        tol_ppm: f64,
        n_chunks: Option<usize>,
        payload: bool,
    ) -> FragIndex {
        let t0 = std::time::Instant::now();
        let n_cand = lib.n_candidates();
        // Precondition (docs/06_predict_frag_index_matchers.md): candidate_id is dense
        // 0..n_cand so it indexes the accumulator and post_cand directly. The library
        // stores its candidates as columns indexed by that id, so this holds by
        // construction (and `Library::from_candidates` asserts it for a hand-built one).
        let total = lib.frag_mz.len();
        assert!(total <= u32::MAX as usize, "total_frags exceeds u32");
        // Every fragment belongs to exactly one candidate, in candidate order: the
        // chunking below cuts the fragment array at candidate boundaries.
        assert_eq!(
            lib.frag_offsets.last().copied().unwrap_or(0) as usize,
            total,
            "the library's fragment offsets must tile its fragment arrays"
        );
        let bins = Self::geometry(lib, tol_ppm);
        let n_bins = bins.n_bins;

        // Chunks: contiguous candidate runs of about equal posting counts. The result does
        // not depend on the count. What the count costs is one histogram of every bin per
        // chunk, so the automatic choice is one chunk per thread, capped so that no chunk
        // is under `MIN_CHUNK_POSTINGS` postings and the histograms together stay under
        // half of one posting array (a tight tolerance has many bins: ~400k at 7.5 ppm).
        let n_chunks = n_chunks
            .unwrap_or_else(|| {
                rayon::current_num_threads()
                    .min(total / MIN_CHUNK_POSTINGS)
                    .min(total / (2 * n_bins).max(1))
            })
            .clamp(1, n_cand.max(1));
        let mut bounds: Vec<usize> = (0..=n_chunks)
            .map(|j| {
                let target = (total as u64 * j as u64 / n_chunks as u64) as u32;
                lib.frag_offsets
                    .partition_point(|&o| o < target)
                    .min(n_cand)
            })
            .collect();
        bounds[0] = 0;
        bounds[n_chunks] = n_cand;
        bounds.dedup();
        let chunks: Vec<(usize, usize)> = bounds.windows(2).map(|w| (w[0], w[1])).collect();

        // Pass 1: per-chunk bin occupancy. Bin each posting by the SAME f32-rounded m/z
        // that the verify uses (post_mz is f32), so the +/-1 probe's one-bin-width proof
        // holds exactly for the stored value and a boundary-straddling within-tol pair is
        // never missed. Binning by the raw f64 while verifying the f32 could place the
        // posting two bins from the peak.
        let mut cursors: Vec<Vec<u32>> = chunks
            .par_iter()
            .map(|&(c0, c1)| {
                let mut h = vec![0u32; n_bins];
                let (a, z) = (lib.frag_offsets[c0] as usize, lib.frag_offsets[c1] as usize);
                for &mz in &lib.frag_mz[a..z] {
                    h[bins.bin(mz as f64)] += 1;
                }
                h
            })
            .collect();
        // Per-bin totals -> CSR start offsets, then each chunk's histogram becomes its
        // starting cursor in every bin: the bin's start plus the earlier chunks' counts.
        let mut bin_start = vec![0u32; n_bins + 1];
        for b in 0..n_bins {
            let n: u32 = cursors.iter().map(|h| h[b]).sum();
            bin_start[b + 1] = bin_start[b] + n;
        }
        for b in 0..n_bins {
            let mut run = bin_start[b];
            for h in cursors.iter_mut() {
                let n = h[b];
                h[b] = run;
                run += n;
            }
        }

        // Pass 2: scatter every chunk in candidate order into its own slots.
        let post_cand: Vec<AtomicU32> = atomic_u32_zeroed(total);
        let post_mz: Vec<AtomicU32> = atomic_u32_zeroed(total);
        let n_payload = if payload { total } else { 0 };
        let post_int: Vec<AtomicU32> = atomic_u32_zeroed(n_payload);
        let post_frag: Vec<AtomicU16> = (0..n_payload)
            .into_par_iter()
            .map(|_| AtomicU16::new(0))
            .collect();
        cursors
            .par_iter_mut()
            .zip(chunks.par_iter())
            .for_each(|(cur, &(c0, c1))| {
                for c in c0..c1 {
                    for (k, gi) in lib.frag_range(c as u32).enumerate() {
                        let mz = lib.frag_mz[gi];
                        let b = bins.bin(mz as f64); // bin by the stored (f32) value
                        let slot = cur[b] as usize;
                        cur[b] += 1;
                        post_cand[slot].store(c as u32, Ordering::Relaxed);
                        post_mz[slot].store(mz.to_bits(), Ordering::Relaxed);
                        if payload {
                            post_int[slot].store(lib.frag_int[gi].to_bits(), Ordering::Relaxed);
                            post_frag[slot].store(k as u16, Ordering::Relaxed);
                        }
                    }
                }
            });
        drop(cursors);
        // In place: an atomic has the size and alignment of its integer, and `f32` those
        // of `u32`, so each of these collects reuses its allocation
        // (`the_posting_arrays_are_converted_in_place` pins it).
        let post_cand: Vec<u32> = post_cand.into_iter().map(AtomicU32::into_inner).collect();
        let post_mz: Vec<f32> = post_mz
            .into_iter()
            .map(|a| f32::from_bits(a.into_inner()))
            .collect();
        let post_int: Vec<f32> = post_int
            .into_iter()
            .map(|a| f32::from_bits(a.into_inner()))
            .collect();
        let post_frag: Vec<u16> = post_frag.into_iter().map(AtomicU16::into_inner).collect();

        tracing::info!(
            postings = total,
            bins = n_bins,
            chunks = chunks.len(),
            payload,
            tol_ppm,
            elapsed_ms = t0.elapsed().as_millis() as u64,
            "fragindex: built"
        );
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

    /// The bin geometry for `lib` at `tol_ppm`: log-space bins over the finite fragment
    /// m/z range.
    fn geometry(lib: &Library, tol_ppm: f64) -> LogBins {
        // m/z range from the library fragments (guard > 0), spec geometry in f64.
        //
        // Skip non-finite values rather than letting one of them decide the range. A single
        // +inf fragment m/z used to make `mz_max` infinite, which tripped the fallback
        // below and collapsed the range for the WHOLE library to [1.0, 2.0]: `LogBins`
        // then clamps every real fragment into one bin and the +/-1 probe degenerates into
        // a linear scan of the entire posting list per peak. No panic and no wrong answer,
        // just an unbounded hang. Library load rejects non-finite m/z now, so this is the
        // second line of defence; dropping the offending value is the right shape either
        // way, because the range is a property of the real fragments.
        //
        // A parallel min/max. The minimum and maximum of a set do not depend on the order
        // they are taken in; the one tie that could (0.0 against -0.0) is clamped to 1.0
        // by the `max(1.0)` below either way.
        let (mut mz_min, mut mz_max) = lib
            .frag_mz
            .par_iter()
            .fold(
                || (f64::INFINITY, f64::NEG_INFINITY),
                |(lo, hi), &mz| {
                    let mz = mz as f64;
                    if !mz.is_finite() {
                        return (lo, hi);
                    }
                    (if mz < lo { mz } else { lo }, if mz > hi { mz } else { hi })
                },
            )
            .reduce(
                || (f64::INFINITY, f64::NEG_INFINITY),
                |a, b| {
                    (
                        if b.0 < a.0 { b.0 } else { a.0 },
                        if b.1 > a.1 { b.1 } else { a.1 },
                    )
                },
            );
        // Reached only when the library has no finite fragment m/z at all (in practice: no
        // fragments). A placeholder range keeps `LogBins::new` well-defined; `page_search`
        // early-returns on the resulting empty index.
        if !mz_min.is_finite() || !mz_max.is_finite() {
            mz_min = 1.0;
            mz_max = 2.0;
        }
        LogBins::new(tol_ppm, mz_min.max(1.0), mz_max.max(mz_min.max(1.0) + 1e-6))
    }

    /// The serial two-pass counting sort the parallel [`FragIndex::build`] replaced, kept
    /// verbatim as the reference its arrays are compared against bit for bit.
    #[cfg(test)]
    fn build_serial(lib: &Library, tol_ppm: f64) -> FragIndex {
        let n_cand = lib.n_candidates();
        let total = lib.frag_mz.len();
        let mut mz_min = f64::INFINITY;
        let mut mz_max = f64::NEG_INFINITY;
        for &mz in &lib.frag_mz {
            let mz = mz as f64;
            if !mz.is_finite() {
                continue;
            }
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
        let mut bin_start = vec![0u32; bins.n_bins + 1];
        for &mz in &lib.frag_mz {
            bin_start[bins.bin(mz as f64) + 1] += 1;
        }
        for b in 0..bins.n_bins {
            bin_start[b + 1] += bin_start[b];
        }
        let mut post_cand = vec![0u32; total];
        let mut post_mz = vec![0f32; total];
        let mut post_int = vec![0f32; total];
        let mut post_frag = vec![0u16; total];
        let mut cursor: Vec<u32> = bin_start[..bins.n_bins].to_vec();
        for c in 0..n_cand {
            for (k, gi) in lib.frag_range(c as u32).enumerate() {
                let mz = lib.frag_mz[gi];
                let b = bins.bin(mz as f64);
                let slot = cursor[b] as usize;
                post_cand[slot] = c as u32;
                post_mz[slot] = mz;
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
        assert!(
            self.has_payload(),
            "probe_peak hands out posting intensities, which an m/z-only index does not hold"
        );
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

    /// [`FragIndex::probe_peak`] for a caller that needs only the matching CANDIDATE of each
    /// posting: the same bins, the same narrowing, the same exact f64 predicate and the
    /// same posting order, calling `f(cid)` where `probe_peak` calls
    /// `f(cid, pmz, pint, pfrag)`. Works on an m/z-only index ([`FragIndex::build_mz_only`])
    /// as well as a full one, because it never touches the payload arrays.
    #[inline]
    pub fn probe_peak_cand<F: FnMut(u32)>(
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
            if z <= a {
                continue;
            }
            for (&mz, &cid) in self.post_mz[a..z].iter().zip(&self.post_cand[a..z]) {
                if within_ppm(mz as f64, peak_mz, self.tol_ppm) {
                    f(cid);
                }
            }
        }
    }

    /// The fragment bin a query m/z probes, as [`FragIndex::probe_peak_win`] would
    /// compute it. Exposed so a caller can compute the bin in a pass of its own and
    /// hand it to [`FragIndex::probe_peak_win_binned`]; the computation is a `ln()`,
    /// and taking it off the probe's dependency chain is worth 7-17% of the probe
    /// depending on the shape, even when the count of `ln()` calls is unchanged (see
    /// extract's per-scan setup scratch). It is NOT here to be cached across calls:
    /// buffering bins per window so
    /// several tasks can share one fill was tried and reverted, because the fill is
    /// serial and the work it removes was parallel.
    #[inline]
    pub fn bin_of(&self, mz: f64) -> u32 {
        self.bins.bin(mz) as u32
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
        f: F,
    ) {
        self.probe_peak_win_binned(nw, peak_mz, self.bin_of(peak_mz), f);
    }

    /// [`FragIndex::probe_peak_win`] with the bin already computed.
    ///
    /// `bin` must be `self.bin_of(peak_mz)`. It is clamped into range, so a wrong
    /// value cannot index out of bounds, but it WILL probe the wrong bins and drop
    /// matches, silently. That is why the only caller computes `peak_mz` and `bin`
    /// from one expression into one scratch entry and reads both back from it
    /// (`extract.rs`, `setup`): the pairing is structural there, not asserted, so
    /// there is no second evaluation that could drift. A caller that stores the two
    /// apart must assert it itself.
    #[inline]
    pub fn probe_peak_win_binned<F: FnMut(u32, f64, f32, u16)>(
        &self,
        nw: &mut WindowNarrow,
        peak_mz: f64,
        bin: u32,
        mut f: F,
    ) {
        if nw.cand_hi <= nw.cand_lo {
            return;
        }
        let b = (bin as usize).min(self.bins.n_bins - 1);
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
    ///
    /// The four parallel arrays are sliced once and zipped, so the loop pays four
    /// range checks for the whole range instead of four per posting; the early return
    /// keeps the empty range, which is what a narrow candidate window mostly produces,
    /// from paying for the four slicings at all.
    ///
    /// Worth -5.7 to -9.3% OF THIS LOOP, at 1, 8, 64 and 512 postings in the range,
    /// measured against the indexed loop it replaced in the same binary and over the
    /// same index (`bench_emit_range` below; 1.28 against 1.37 ns/posting at 512, both
    /// cache-hot). That is not -5.7 to -9.3% of the probe. At the harness's widest arm
    /// (65 emitted postings per peak, 798 ns/peak) the whole verify loop is about 11%
    /// of the probe, so the ceiling here is about -0.8% of a probe and less on a narrow
    /// candidate window; the earlier "-4 to -6% of the probe" in this comment came from
    /// an unreproducible pair of binaries and was wrong. It is kept because it is
    /// measurably faster, never slower, and costs no memory -- not because it moves the
    /// stage. The end-to-end AIF extract A/B below cannot resolve it.
    ///
    /// The PREDICATE is untouched. Hoisting it to precomputed m/z bounds would be the
    /// obvious next step and is not equivalent: `within_ppm` compares
    /// `hi - lo <= tol * 1e-6 * lo` in f64 per pair, and a bound computed once from
    /// `peak_mz` rounds differently from that at the tolerance edge. The edge decides
    /// real matches, so the loop keeps calling the canonical predicate.
    #[inline]
    fn emit_range<F: FnMut(u32, f64, f32, u16)>(
        &self,
        a: usize,
        z: usize,
        peak_mz: f64,
        f: &mut F,
    ) {
        if z <= a {
            return;
        }
        let mzs = &self.post_mz[a..z];
        let cands = &self.post_cand[a..z];
        let ints = &self.post_int[a..z];
        let frags = &self.post_frag[a..z];
        for (((&mz, &cid), &pint), &pfrag) in mzs.iter().zip(cands).zip(ints).zip(frags) {
            let pmz = mz as f64;
            if within_ppm(pmz, peak_mz, self.tol_ppm) {
                f(cid, pmz, pint, pfrag);
            }
        }
    }

    /// Build an empty narrowing cache for the candidate window `[cand_lo, cand_hi)`.
    pub fn window_narrow(&self, cand_lo: u32, cand_hi: u32) -> WindowNarrow {
        // The windowed probes hand out posting intensities and ordinals.
        assert!(
            self.has_payload(),
            "the windowed probes hand out posting intensities, which an m/z-only index \
             does not hold"
        );
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
///
/// Array-of-structs: a candidate's stamp, count and observed sum sit in one 16-byte slot,
/// so a matched posting updates one cache line instead of three (the three arrays used to
/// be separate, and in a dense window the posting stream reaches slots at random). Also
/// kept, beside the first-touch list: the QUALIFIED list, the candidates whose count has
/// reached the caller's `min_count`, appended at the moment they reach it, so a scan's
/// scoring walks only the candidates it can report instead of filtering every candidate
/// the scan touched.
pub struct SeedScratch {
    slots: Vec<SeedSlot>,
    touched: Vec<u32>,
    qualified: Vec<u32>,
    /// Count at which a candidate enters `qualified` (at least 1).
    min_count: u32,
    epoch: u32,
    /// `candidate_id` that maps to slot 0. The arrays are indexed WINDOW-RELATIVE, so
    /// they only need to span the widest isolation window rather than the whole library:
    /// sized by `n_cand` they cost 16 B x n_cand PER rayon worker (877 MB per worker on
    /// the profiled 54.8M-candidate library), almost all of it never touched because a
    /// worker only ever sees candidates inside one window.
    base: u32,
}

/// One candidate's accumulator slot (16 bytes).
#[derive(Clone, Copy, Default)]
struct SeedSlot {
    /// Epoch of the scan that last touched it; 0 is never a live epoch.
    stamp: u32,
    count: u32,
    obs_sum: f64,
}

impl SeedScratch {
    /// `cap` is the expected maximum candidate-window width, not the library size.
    /// Passing a smaller value is safe: the arrays grow on demand. The qualified list
    /// holds every touched candidate (a `min_count` of 1).
    pub fn new(cap: usize) -> SeedScratch {
        SeedScratch::with_min_count(cap, 1)
    }

    /// As [`SeedScratch::new`], with the count at which a candidate is QUALIFIED
    /// ([`SeedScratch::qualified`]): the seed's `min_matched_peaks`. Zero behaves as one,
    /// since every touched candidate has matched at least once.
    pub fn with_min_count(cap: usize, min_count: usize) -> SeedScratch {
        SeedScratch {
            slots: vec![SeedSlot::default(); cap],
            touched: Vec::new(),
            qualified: Vec::new(),
            min_count: u32::try_from(min_count).unwrap_or(u32::MAX).max(1),
            epoch: 0,
            base: 0,
        }
    }

    /// Ensure the window-relative slots span `width`.
    fn ensure(&mut self, width: usize) {
        if self.slots.len() < width {
            // New slots must not appear stamped for the current epoch.
            self.slots.resize(width, SeedSlot::default());
        }
    }

    /// Accumulate one scan's peaks over the candidate window. Peaks must already be
    /// in the caller's fixed order (e.g. m/z ascending, or the top-N re-sorted
    /// order) so `obs_sum` is summed deterministically. After the call, `touched()`
    /// lists the hit candidates, `qualified()` those among them that reached the minimum
    /// count, and `count`/`obs_sum` hold their values.
    pub fn accumulate(
        &mut self,
        idx: &FragIndex,
        peaks: &[(f64, f32)],
        cand_lo: u32,
        cand_hi: u32,
    ) {
        self.epoch += 1;
        self.touched.clear();
        self.qualified.clear();
        // Index relative to this window's first candidate.
        self.base = cand_lo;
        self.ensure((cand_hi.saturating_sub(cand_lo)) as usize + 1);
        let (epoch, base, min_count) = (self.epoch, self.base, self.min_count);
        let SeedScratch {
            slots,
            touched,
            qualified,
            ..
        } = self;
        for &(mz, inten) in peaks {
            idx.probe_peak_cand(mz, cand_lo, cand_hi, |cid| {
                let s = &mut slots[(cid - base) as usize];
                if s.stamp != epoch {
                    *s = SeedSlot {
                        stamp: epoch,
                        count: 0,
                        obs_sum: 0.0,
                    };
                    touched.push(cid);
                }
                s.count += 1;
                s.obs_sum += inten as f64;
                if s.count == min_count {
                    qualified.push(cid);
                }
            });
        }
    }

    /// Candidates hit in the last `accumulate`, in first-touch (probe) order.
    /// Callers that need determinism across a float reduction sort this first.
    pub fn touched(&self) -> &[u32] {
        &self.touched
    }

    /// Candidates of the last `accumulate` whose count reached the minimum count, in the
    /// order they reached it: exactly the touched candidates with `count >= min_count`,
    /// in a different order, so a caller that sorts them (the seed sorts by score, then
    /// candidate id) gets the list it got by filtering `touched`.
    pub fn qualified(&self) -> &[u32] {
        &self.qualified
    }

    /// Valid only for candidate ids from the most recent [`SeedScratch::accumulate`]
    /// window (which is what [`SeedScratch::touched`] returns).
    #[inline]
    pub fn count(&self, cid: u32) -> u32 {
        self.slots[(cid - self.base) as usize].count
    }

    /// See [`SeedScratch::count`] for the validity window.
    #[inline]
    pub fn obs_sum(&self, cid: u32) -> f64 {
        self.slots[(cid - self.base) as usize].obs_sum
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
        let mut cs = Vec::new();
        for (i, (frags, pmz)) in cands.iter().enumerate() {
            let start = frag_mz.len();
            for &(mz, int) in frags {
                // the library stores fragment m/z as f32
                frag_mz.push(mz as f32);
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
        }
        Library::from_candidates(cs, frag_mz, frag_int, frag_name_id, vec!["f".to_string()])
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
                    // The precomputed-bin entry point must be the same drop-in: same
                    // postings, same order, same bits, whenever `bin` came from `bin_of`.
                    let mut c: Vec<(u32, u64, u32, u16)> = Vec::new();
                    idx.probe_peak_win_binned(&mut nw, q, idx.bin_of(q), |c_, m, it, fr| {
                        c.push((c_, m.to_bits(), it.to_bits(), fr))
                    });
                    assert_eq!(
                        a, c,
                        "binned probe diverged at q={q} window=({win_lo},{win_hi})"
                    );
                }
            }
        }
    }

    /// `bin_of` is the bin the probe would have computed, for every m/z including the
    /// ones the geometry clamps (at or below `mz_min`, above `mz_max`, zero, negative).
    ///
    /// A modest test, and it was once described here as the one the whole contract
    /// rests on, which it is not: `bin_of` is `self.bins.bin(mz) as u32`, so this can
    /// only fail on a `u32` truncation, which needs more than 2^32 bins (there are
    /// about 115k at 20 ppm over 200-2000). What it does pin is that the cast is the
    /// only thing between the two and that the clamped edges agree. The equality that
    /// matters -- that the bin handed to `probe_peak_win_binned` belongs to the m/z
    /// handed with it -- is structural at the only call site: extract computes both in
    /// one expression into one scratch entry and reads both out of it.
    #[test]
    fn bin_of_is_the_bin_the_probe_uses() {
        let tol = 20.0;
        let lib = lib_from(&[
            (vec![(300.0, 1.0), (900.0, 1.0)], 400.0),
            (vec![(1800.0, 1.0)], 500.0),
        ]);
        let idx = FragIndex::build(&lib, tol);
        // A peak the probe finds nothing for still has to agree, because extract buffers
        // the bin for every peak of the scan, hit or not.
        let mut mz = 1.0f64;
        while mz < 5000.0 {
            assert_eq!(
                idx.bin_of(mz) as usize,
                idx.bins.bin(mz),
                "bin_of disagrees with the probe's own binning at {mz}"
            );
            mz *= 1.013;
        }
        for &q in &[0.0f64, -1.0, f64::MIN_POSITIVE, 1e12] {
            assert_eq!(idx.bin_of(q) as usize, idx.bins.bin(q), "edge m/z {q}");
        }
    }

    /// A bin past the top of the geometry is clamped rather than indexing out of
    /// bounds. `bin_of` can never produce one, so this pins the defensive clamp that
    /// keeps the public entry point total for a caller that hands over a stale bin.
    #[test]
    fn a_bin_past_the_top_is_clamped_not_a_panic() {
        let tol = 20.0;
        // The indexed range has to be wide, or the clamped top bin is still the peak's.
        let lib = lib_from(&[(vec![(700.0, 1.0)], 400.0), (vec![(1800.0, 1.0)], 500.0)]);
        let idx = FragIndex::build(&lib, tol);
        let mut nw = idx.window_narrow(0, 2);
        let mut n = 0usize;
        idx.probe_peak_win_binned(&mut nw, 700.0, u32::MAX, |_, _, _, _| n += 1);
        // Wrong bin, so the match is missed -- but nothing panics, and the correct bin
        // still finds it.
        assert_eq!(
            n, 0,
            "a bogus bin probes the top of the index, not the peak"
        );
        let mut nw = idx.window_narrow(0, 2);
        let mut m = 0usize;
        idx.probe_peak_win_binned(&mut nw, 700.0, idx.bin_of(700.0), |_, _, _, _| m += 1);
        assert_eq!(m, 1);
    }

    /// `emit_range` walks four parallel arrays. Whatever way it is written, every
    /// posting within tolerance must be emitted exactly once and carry ITS OWN
    /// candidate, intensity and fragment ordinal: a mis-sliced or mis-zipped array
    /// would pair a posting's m/z with a neighbour's candidate, which no small fixture
    /// and no count-only assertion would notice. The fixture puts hundreds of postings
    /// in the probed bins so the verify loop actually runs long.
    #[test]
    fn emit_range_pairs_every_posting_with_its_own_columns() {
        let tol = 20.0;
        let n_cand = 400usize;
        let n_frag = 4usize;
        // All fragments within a few ppm of 600, so they crowd into a handful of bins;
        // intensity and ordinal identify the posting uniquely.
        let mut cands: Vec<(Vec<(f64, f32)>, f64)> = Vec::new();
        for i in 0..n_cand {
            let frags: Vec<(f64, f32)> = (0..n_frag)
                .map(|k| {
                    let ppm = (i * n_frag + k) as f64 * 0.11 - 40.0; // -40..+136 ppm
                    (600.0 * (1.0 + ppm * 1e-6), (i * n_frag + k) as f32)
                })
                .collect();
            cands.push((frags, 400.0 + i as f64 * 1e-3));
        }
        let lib = lib_from(&cands);
        let idx = FragIndex::build(&lib, tol);
        let (lo, hi) = idx.candidate_range(400.0, 400.0 + (n_cand as f64 - 1.0) * 1e-3);
        assert_eq!((lo, hi), (0, n_cand as u32));

        let q = 600.0f64;
        let mut got: Vec<(u32, u64, u32, u16)> = Vec::new();
        idx.probe_peak(q, lo, hi, |c, m, it, fr| {
            got.push((c, m.to_bits(), it.to_bits(), fr))
        });
        // Reference: the same predicate, over the library itself.
        let mut want: Vec<(u32, u64, u32, u16)> = Vec::new();
        for c in 0..lib.n_candidates() {
            for (k, gi) in lib.frag_range(c as u32).enumerate() {
                let pmz = lib.frag_mz[gi] as f64;
                if within_ppm(pmz, q, tol) {
                    want.push((
                        c as u32,
                        pmz.to_bits(),
                        lib.frag_int[gi].to_bits(),
                        k as u16,
                    ));
                }
            }
        }
        assert!(
            want.len() > 100,
            "the fixture must put a long run of postings in the probed bins, got {}",
            want.len()
        );
        let mut a = got.clone();
        let mut b = want.clone();
        a.sort_unstable();
        b.sort_unstable();
        let mut once = a.clone();
        once.dedup();
        assert_eq!(once.len(), got.len(), "no posting may be emitted twice");
        assert_eq!(a, b, "emitted postings, or their columns, differ");
        // And the intensity really identifies the posting: cand*n_frag + frag.
        for &(cid, _, int_bits, frag) in &got {
            assert_eq!(
                f32::from_bits(int_bits),
                (cid as usize * n_frag + frag as usize) as f32,
                "posting columns are out of step for cand {cid} frag {frag}"
            );
        }
        // The cached and binned entry points see the same thing.
        let mut nw = idx.window_narrow(lo, hi);
        let mut cached: Vec<(u32, u64, u32, u16)> = Vec::new();
        idx.probe_peak_win_binned(&mut nw, q, idx.bin_of(q), |c, m, it, fr| {
            cached.push((c, m.to_bits(), it.to_bits(), fr))
        });
        assert_eq!(got, cached);
    }

    /// The indexed verify loop `emit_range` replaced, kept verbatim so the two can be
    /// timed against each other in the same binary. `tests/bench_fragindex.rs` cannot
    /// do it: `emit_range` is private and every arm of that harness calls the same one,
    /// so no arm varies it.
    fn emit_range_indexed<F: FnMut(u32, f64, f32, u16)>(
        idx: &FragIndex,
        a: usize,
        z: usize,
        peak_mz: f64,
        f: &mut F,
    ) {
        for p in a..z {
            let pmz = idx.post_mz[p] as f64;
            if within_ppm(pmz, peak_mz, idx.tol_ppm) {
                f(idx.post_cand[p], pmz, idx.post_int[p], idx.post_frag[p]);
            }
        }
    }

    /// Microbenchmark: `emit_range`'s slice-and-zip against the indexed loop above, at
    /// several range lengths, over the SAME index and the same peak m/z sequence, with
    /// the checksum asserted equal. `#[ignore]`d.
    ///
    ///   cargo test --release -p mumdia bench_emit_range -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_emit_range() {
        let tol = 20.0;
        println!("emit_range: sliced-and-zipped (shipped) vs the indexed loop it replaced");
        // Fix the geometry with two anchor fragments present in every fixture, so the
        // bin around 600 is the same bin whatever else the library holds.
        let anchors = vec![(200.0f64, 1.0f32), (1500.0, 1.0)];
        let probe_lib = lib_from(&[(
            {
                let mut v = anchors.clone();
                v.push((600.0, 1.0));
                v
            },
            900.0,
        )]);
        let geom = FragIndex::build(&probe_lib, tol);
        let b0 = geom.bins.bin(600.0);
        // Every m/z within 9 ppm of 600 (so every posting verifies at 20 ppm) that the
        // geometry puts in that one bin, as the index will see it after the f32 store.
        let in_bin: Vec<f64> = (-45..=45)
            .map(|k| ((600.0 * (1.0 + k as f64 * 0.2e-6)) as f32) as f64)
            .filter(|&mz| geom.bins.bin(mz) == b0)
            .collect();
        assert!(in_bin.len() >= 8, "the bin holds too few distinct m/z");
        for &n_post in &[1usize, 8, 64, 512] {
            // `n_post` postings inside that one bin, each on its own candidate, plus the
            // anchors so the index is not degenerate. m/z repeat once the bin runs out
            // of distinct values, which is what a crowded bin looks like anyway.
            let mut cands: Vec<(Vec<(f64, f32)>, f64)> = (0..n_post)
                .map(|i| {
                    (
                        vec![(in_bin[i % in_bin.len()], i as f32)],
                        400.0 + i as f64 * 1e-3,
                    )
                })
                .collect();
            cands.push((anchors.clone(), 900.0));
            let lib = lib_from(&cands);
            let idx = FragIndex::build(&lib, tol);
            let b = idx.bins.bin(600.0);
            let (a, z) = (idx.bin_start[b] as usize, idx.bin_start[b + 1] as usize);
            assert_eq!(z - a, n_post, "fixture did not put the run in one bin");
            let iters = 4_000_000 / (n_post + 2);
            let mut best = (f64::INFINITY, f64::INFINITY);
            let (mut hs, mut hi_) = (0u64, 0u64);
            for r in 0..14 {
                // rotate, so warm-cache position cannot favour one side
                for s in [r % 2, 1 - r % 2] {
                    let mut acc = 0u64;
                    let t = std::time::Instant::now();
                    for it in 0..iters {
                        // perturbation far below tolerance, so no posting changes side
                        let q = 600.0 * (1.0 + (it % 4) as f64 * 1e-12);
                        if s == 0 {
                            idx.emit_range(a, z, q, &mut |c, m, i, f| {
                                acc = acc
                                    .wrapping_add(c as u64)
                                    .wrapping_add(m.to_bits())
                                    .wrapping_add(i.to_bits() as u64)
                                    .wrapping_add(f as u64);
                            });
                        } else {
                            emit_range_indexed(&idx, a, z, q, &mut |c, m, i, f| {
                                acc = acc
                                    .wrapping_add(c as u64)
                                    .wrapping_add(m.to_bits())
                                    .wrapping_add(i.to_bits() as u64)
                                    .wrapping_add(f as u64);
                            });
                        }
                    }
                    let el = t.elapsed().as_secs_f64() * 1e9 / (iters as f64);
                    std::hint::black_box(acc);
                    if s == 0 {
                        best.0 = best.0.min(el);
                        hs = acc;
                    } else {
                        best.1 = best.1.min(el);
                        hi_ = acc;
                    }
                }
            }
            assert_eq!(hs, hi_, "the two verify loops emit different postings");
            println!(
                "  range {n_post:4} postings: sliced {:8.1} ns/range, indexed {:8.1} ns/range  \
                 delta {:+6.1}%  ({:.2} vs {:.2} ns/posting)",
                best.0,
                best.1,
                100.0 * (best.0 - best.1) / best.1,
                best.0 / n_post as f64,
                best.1 / n_post as f64
            );
        }
    }

    /// Every array of a parallel build equals the serial build's, bit for bit.
    fn assert_same_index(a: &FragIndex, b: &FragIndex, what: &str) {
        assert_eq!(
            format!("{:?}", a.bins),
            format!("{:?}", b.bins),
            "{what}: geometry"
        );
        assert_eq!(a.bin_start, b.bin_start, "{what}: bin_start");
        assert_eq!(a.post_cand, b.post_cand, "{what}: post_cand");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.post_mz), bits(&b.post_mz), "{what}: post_mz");
        assert_eq!(bits(&a.post_int), bits(&b.post_int), "{what}: post_int");
        assert_eq!(a.post_frag, b.post_frag, "{what}: post_frag");
        assert_eq!(a.prec_mz, b.prec_mz, "{what}: prec_mz");
        assert_eq!(a.n_cand, b.n_cand, "{what}: n_cand");
    }

    /// A library of `n` candidates with 0..=11 fragments each, spread over 100-2000 m/z
    /// so every bin range is populated and many candidates share bins.
    fn random_lib(n: usize, seed: u64) -> Library {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let cands: Vec<(Vec<(f64, f32)>, f64)> = (0..n)
            .map(|i| {
                let k = (next() * 12.0) as usize;
                let frags = (0..k)
                    .map(|_| (100.0 + 1900.0 * next(), next() as f32))
                    .collect();
                (frags, 400.0 + i as f64 * 0.01)
            })
            .collect();
        lib_from(&cands)
    }

    /// The parallel build (chunked histograms, per-chunk cursors, atomic scatter) against
    /// the serial counting sort it replaced, over chunk counts from one to one per
    /// candidate, at two tolerances.
    #[test]
    fn the_parallel_build_reproduces_the_serial_index_bit_for_bit() {
        let lib = random_lib(3_000, 0x5eed);
        for tol in [20.0, 7.5] {
            let serial = FragIndex::build_serial(&lib, tol);
            for chunks in [1usize, 2, 3, 7, 64] {
                let par = FragIndex::build_chunked(&lib, tol, Some(chunks), true);
                assert_same_index(&par, &serial, &format!("tol {tol} chunks {chunks}"));
            }
            assert_same_index(&FragIndex::build(&lib, tol), &serial, "build");
        }
        // Empty candidates at the ends and in the middle, so chunk boundaries fall on runs
        // of candidates without fragments.
        let mut cands: Vec<(Vec<(f64, f32)>, f64)> = Vec::new();
        for i in 0..200 {
            let frags = if i % 3 == 0 || !(5..=190).contains(&i) {
                Vec::new()
            } else {
                vec![(300.0 + i as f64, 1.0), (900.0 - i as f64, 0.5)]
            };
            cands.push((frags, 400.0 + i as f64));
        }
        let lib = lib_from(&cands);
        let serial = FragIndex::build_serial(&lib, 20.0);
        for chunks in [1usize, 4, 50, 200] {
            let par = FragIndex::build_chunked(&lib, 20.0, Some(chunks), true);
            assert_same_index(&par, &serial, &format!("gappy chunks {chunks}"));
        }
    }

    /// The m/z that the bin geometry treats specially must land in the same bins through
    /// the parallel build: values exactly on bin edges and one f32 ULP either side, values
    /// below the range (zero, negative, subnormal, below 1.0 where the range is clamped),
    /// the top of the range, and the non-finite values the range skips (NaN to bin 0,
    /// +inf clamped to the top bin, -inf to bin 0).
    #[test]
    fn edge_mz_bins_identically_in_the_parallel_build() {
        let tol = 20.0;
        // Geometry of a plain 150-1800 library, to place values on its bin edges.
        let probe = lib_from(&[(vec![(150.0, 1.0), (1800.0, 1.0)], 400.0)]);
        let g = FragIndex::build_serial(&probe, tol).bins;
        let mut special: Vec<f64> = Vec::new();
        for k in [0usize, 1, 2, 1_000, 50_000, g.n_bins - 3, g.n_bins - 2] {
            // The m/z of edge k: exp(ln_min + k w), with ln_min = ln(150).
            let e = ((150.0f64).ln() + k as f64 * g.w).exp() as f32;
            for d in [-1i32, 0, 1] {
                special.push(f32::from_bits((e.to_bits() as i32 + d) as u32) as f64);
            }
        }
        special.extend([
            150.0,
            1800.0,
            0.0,
            -0.0,
            -5.0,
            f32::MIN_POSITIVE as f64,
            0.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ]);
        let mut cands: Vec<(Vec<(f64, f32)>, f64)> = Vec::new();
        for (i, chunk) in special.chunks(3).enumerate() {
            cands.push((
                chunk.iter().map(|&m| (m, i as f32)).collect(),
                400.0 + i as f64,
            ));
        }
        let lib = lib_from(&cands);
        let serial = FragIndex::build_serial(&lib, tol);
        for chunks in [1usize, 2, 5, cands.len()] {
            let par = FragIndex::build_chunked(&lib, tol, Some(chunks), true);
            assert_same_index(&par, &serial, &format!("edges chunks {chunks}"));
        }
        // And the clamping is what it claims: NaN and -inf in bin 0, +inf in the top bin.
        assert_eq!(serial.bins.bin(f64::NAN), 0);
        assert_eq!(serial.bins.bin(f64::NEG_INFINITY), 0);
        assert_eq!(serial.bins.bin(f64::INFINITY), serial.bins.n_bins - 1);
    }

    /// The seed's m/z-only index is the full index without its payload: the same geometry,
    /// CSR offsets, candidates and m/z, bit for bit, and `probe_peak_cand` reports exactly
    /// the candidates `probe_peak` reports, in the same order, on either index.
    #[test]
    fn the_mz_only_index_is_the_full_index_without_its_payload() {
        let lib = random_lib(2_000, 0xfeed);
        for tol in [20.0, 7.5] {
            let full = FragIndex::build(&lib, tol);
            for chunks in [None, Some(1usize), Some(5)] {
                let mz = FragIndex::build_chunked(&lib, tol, chunks, false);
                assert!(full.has_payload() && !mz.has_payload());
                assert!(mz.post_int.is_empty() && mz.post_frag.is_empty());
                assert_eq!(format!("{:?}", mz.bins), format!("{:?}", full.bins));
                assert_eq!(mz.bin_start, full.bin_start);
                assert_eq!(mz.post_cand, full.post_cand);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&mz.post_mz), bits(&full.post_mz));
                assert_eq!(mz.prec_mz, full.prec_mz);
                // Probe every fragment m/z, nudged inside and outside the tolerance, over a
                // narrow and the full candidate window.
                for &(lo, hi) in &[(0u32, 2_000u32), (300, 900), (1_999, 2_000), (5, 5)] {
                    for (i, &m) in lib.frag_mz.iter().enumerate().step_by(7) {
                        let q = m as f64 * (1.0 + [0.0, 0.9, -0.9, 1.3][i % 4] * tol * 1e-6);
                        let mut want = Vec::new();
                        full.probe_peak(q, lo, hi, |c, _, _, _| want.push(c));
                        let mut got_full = Vec::new();
                        full.probe_peak_cand(q, lo, hi, |c| got_full.push(c));
                        let mut got_mz = Vec::new();
                        mz.probe_peak_cand(q, lo, hi, |c| got_mz.push(c));
                        assert_eq!(got_full, want, "q={q} window=({lo},{hi})");
                        assert_eq!(got_mz, want, "q={q} window=({lo},{hi})");
                    }
                }
            }
        }
    }

    /// The payload entry points refuse an m/z-only index rather than handing out zeros.
    #[test]
    #[should_panic(expected = "m/z-only index")]
    fn an_mz_only_index_refuses_the_windowed_probe() {
        let lib = random_lib(50, 3);
        let _ = FragIndex::build_mz_only(&lib, 20.0).window_narrow(0, 50);
    }

    #[test]
    #[should_panic(expected = "m/z-only index")]
    fn an_mz_only_index_refuses_probe_peak() {
        let lib = random_lib(50, 3);
        FragIndex::build_mz_only(&lib, 20.0).probe_peak(500.0, 0, 50, |_, _, _, _| {});
    }

    /// The seed accumulator reads through `probe_peak_cand`, so it must see the same counts
    /// and observed sums over an m/z-only index as over a full one.
    #[test]
    fn the_seed_accumulator_is_the_same_over_the_mz_only_index() {
        let lib = random_lib(1_500, 0xabc);
        let full = FragIndex::build(&lib, 20.0);
        let mz = FragIndex::build_mz_only(&lib, 20.0);
        let peaks: Vec<(f64, f32)> = lib
            .frag_mz
            .iter()
            .step_by(3)
            .enumerate()
            .map(|(i, &m)| (m as f64 * (1.0 + 3e-6), 1.0 + i as f32 * 0.5))
            .collect();
        let (mut a, mut b) = (SeedScratch::new(8), SeedScratch::new(8));
        for &(lo, hi) in &[(0u32, 1_500u32), (200, 700)] {
            a.accumulate(&full, &peaks, lo, hi);
            b.accumulate(&mz, &peaks, lo, hi);
            assert_eq!(a.touched(), b.touched());
            for &c in a.touched() {
                assert_eq!(a.count(c), b.count(c));
                assert_eq!(a.obs_sum(c).to_bits(), b.obs_sum(c).to_bits());
            }
        }
    }

    /// The three-array accumulator the slot accumulator replaced, verbatim, as the
    /// reference it is compared against and timed against.
    struct SoaScratch {
        count: Vec<u32>,
        obs_sum: Vec<f64>,
        stamp: Vec<u32>,
        touched: Vec<u32>,
        epoch: u32,
        base: u32,
    }

    impl SoaScratch {
        fn new(cap: usize) -> SoaScratch {
            SoaScratch {
                count: vec![0; cap],
                obs_sum: vec![0.0; cap],
                stamp: vec![0; cap],
                touched: Vec::new(),
                epoch: 0,
                base: 0,
            }
        }
        fn accumulate(&mut self, idx: &FragIndex, peaks: &[(f64, f32)], lo: u32, hi: u32) {
            self.epoch += 1;
            self.touched.clear();
            self.base = lo;
            let width = (hi.saturating_sub(lo)) as usize + 1;
            if self.count.len() < width {
                self.count.resize(width, 0);
                self.obs_sum.resize(width, 0.0);
                self.stamp.resize(width, 0);
            }
            let (epoch, base) = (self.epoch, self.base);
            for &(mz, inten) in peaks {
                idx.probe_peak_cand(mz, lo, hi, |cid| {
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
    }

    /// Dense scans over a crowded window: peaks drawn from the library's own fragment m/z,
    /// so most candidates are touched and many several times.
    fn dense_scans(lib: &Library, n_scans: usize, per_scan: usize) -> Vec<Vec<(f64, f32)>> {
        let mut state = 0x51ce_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        (0..n_scans)
            .map(|_| {
                let mut v: Vec<(f64, f32)> = (0..per_scan)
                    .map(|_| {
                        let m = lib.frag_mz[next() % lib.frag_mz.len()] as f64;
                        (m * (1.0 + 2e-6), (next() % 1000) as f32 / 7.0)
                    })
                    .collect();
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
                v
            })
            .collect()
    }

    /// The slot accumulator touches the same candidates in the same order, with the same
    /// counts and the same observed sums bit for bit, as the three-array one, scan after
    /// scan over one reused scratch (so the epoch reset is exercised); and its qualified
    /// list is exactly the touched candidates at or above the minimum count.
    #[test]
    fn the_slot_accumulator_matches_the_three_array_one() {
        let lib = random_lib(3_000, 0x9e37);
        let idx = FragIndex::build_mz_only(&lib, 20.0);
        let scans = dense_scans(&lib, 40, 400);
        for min in [0usize, 1, 2, 3, 6] {
            let mut new = SeedScratch::with_min_count(4, min);
            let mut old = SoaScratch::new(4);
            for (k, peaks) in scans.iter().enumerate() {
                let (lo, hi) = if k % 3 == 0 { (0, 3_000) } else { (500, 2_100) };
                new.accumulate(&idx, peaks, lo, hi);
                old.accumulate(&idx, peaks, lo, hi);
                assert_eq!(new.touched(), &old.touched[..], "scan {k}");
                for &c in new.touched() {
                    let cc = (c - old.base) as usize;
                    assert_eq!(new.count(c), old.count[cc]);
                    assert_eq!(new.obs_sum(c).to_bits(), old.obs_sum[cc].to_bits());
                }
                let mut want: Vec<u32> = old
                    .touched
                    .iter()
                    .copied()
                    .filter(|&c| old.count[(c - old.base) as usize] as usize >= min)
                    .collect();
                let mut got = new.qualified().to_vec();
                assert_eq!(
                    got.len(),
                    want.len(),
                    "scan {k} min {min}: a candidate twice?"
                );
                want.sort_unstable();
                got.sort_unstable();
                assert_eq!(got, want, "scan {k} min {min}");
            }
        }
    }

    /// Microbenchmark: the slot accumulator plus its qualified list against the three-array
    /// accumulator plus the filter over `touched` the seed used to run, over dense scans of
    /// one wide window. `#[ignore]`d.
    ///
    ///   cargo test --release -p mumdia bench_seed_accumulator -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_seed_accumulator() {
        // A window that fits in cache (60k candidates, ~1 MB of slots) and one that does not
        // (2M candidates, ~32 MB), where the slot's single cache line per posting counts.
        for (n_cand, n_scans, per_scan) in
            [(60_000usize, 200usize, 2_000usize), (2_000_000, 60, 4_000)]
        {
            let lib = random_lib(n_cand, 0x7);
            let idx = FragIndex::build_mz_only(&lib, 20.0);
            let scans = dense_scans(&lib, n_scans, per_scan);
            let hi = n_cand as u32;
            let min = 4usize;
            let mut best = (f64::INFINITY, f64::INFINITY);
            let (mut ha, mut hb) = (0u64, 0u64);
            for r in 0..6 {
                for arm in [r % 2, 1 - r % 2] {
                    let t = std::time::Instant::now();
                    let mut acc = 0u64;
                    if arm == 0 {
                        let mut sc = SeedScratch::with_min_count(n_cand + 1, min);
                        for peaks in &scans {
                            sc.accumulate(&idx, peaks, 0, hi);
                            for &c in sc.qualified() {
                                acc = acc
                                    .wrapping_add(c as u64)
                                    .wrapping_add(sc.obs_sum(c).to_bits());
                            }
                        }
                    } else {
                        let mut sc = SoaScratch::new(n_cand + 1);
                        for peaks in &scans {
                            sc.accumulate(&idx, peaks, 0, hi);
                            for &c in sc.touched.iter() {
                                let cc = (c - sc.base) as usize;
                                if sc.count[cc] as usize >= min {
                                    acc = acc
                                        .wrapping_add(c as u64)
                                        .wrapping_add(sc.obs_sum[cc].to_bits());
                                }
                            }
                        }
                    }
                    let el = t.elapsed().as_secs_f64();
                    std::hint::black_box(acc);
                    if arm == 0 {
                        best.0 = best.0.min(el);
                        ha = acc;
                    } else {
                        best.1 = best.1.min(el);
                        hb = acc;
                    }
                }
            }
            assert_eq!(ha, hb, "the two accumulators disagree");
            println!(
                "seed accumulator, {n_scans} scans x {per_scan} peaks over {n_cand} candidates: \
                 slots + qualified {:.1} ms, three arrays + filter {:.1} ms ({:+.1}%)",
                best.0 * 1e3,
                best.1 * 1e3,
                100.0 * (best.0 - best.1) / best.1
            );
        }
    }

    /// The atomic posting arrays become plain arrays IN PLACE: an atomic has its integer's
    /// size and alignment, so `into_iter().map(..).collect()` reuses the allocation rather
    /// than holding two copies of a posting array at the build's peak.
    #[test]
    fn the_posting_arrays_are_converted_in_place() {
        let a = atomic_u32_zeroed(10_000);
        let p = a.as_ptr() as usize;
        let b: Vec<u32> = a.into_iter().map(AtomicU32::into_inner).collect();
        assert_eq!(b.as_ptr() as usize, p, "u32 conversion reallocated");
        let a = atomic_u32_zeroed(10_000);
        let p = a.as_ptr() as usize;
        let f: Vec<f32> = a
            .into_iter()
            .map(|x| f32::from_bits(x.into_inner()))
            .collect();
        assert_eq!(f.as_ptr() as usize, p, "f32 conversion reallocated");
        let h: Vec<AtomicU16> = (0..10_000).map(|_| AtomicU16::new(0)).collect();
        let p = h.as_ptr() as usize;
        let h: Vec<u16> = h.into_iter().map(AtomicU16::into_inner).collect();
        assert_eq!(h.as_ptr() as usize, p, "u16 conversion reallocated");
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
        let (lo, hi) = (0u32, lib.n_candidates() as u32);
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
