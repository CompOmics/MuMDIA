//! Stage D `mumdia extract`: targeted 3D extraction (docs/09_extract.md).
//!
//! Data-driven and peak-major: observed peaks probe the inverted fragment index,
//! and a candidate hypothesis is materialized only where fragment evidence
//! exists (a sparse accumulator keyed by `candidate_id`, entries created on first
//! collision). Work scales with peak-candidate collisions, not library size.
//! RT is applied as a per-candidate window post-filter (the documented
//! fallback); MVP is 3D so IM is absent.
//!
//! The cascade: (a) isolation-window candidate range + RT window membership,
//! (b) cheap matched-fragment presence gate, (c) matched-fragment count + a
//! consecutive-scan co-elution run, (d) apex detection. Exact intensity scores
//! are computed in the features stage from the emitted chromatograms.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{ExtractConfig, GateMode, PeakClaim};
use mumdia_core::schema::artifact;
use mumdia_io::report::{ArtifactReport, Written};
use mumdia_io::table::{
    write_table, write_table_hashed, Col, TableFile, TableWriter, WRITE_TABLE_CHUNK_ROWS,
};
use serde_json::json;
use tracing::{info, warn};

use mumdia_core::constants::{ppm_bounds, ISOTOPE_SPACING};

use crate::index::Library;
use crate::matchers::binning::LogBins;
#[cfg(test)]
use crate::matchers::fragindex::NarrowedProbe;
use crate::matchers::fragindex::{BinnedProbe, FragIndex, LocalIndex, WindowNarrow};
use crate::spectra::{load_ms1, load_ms2, Ms1Scan};
use crate::stages::rt_im_train::RtWindows;
use mumdia_core::config::MatcherKind;
use mumdia_core::types::Ms2Scan;
use rayon::prelude::*;

/// The matcher backend plus the values that never change across a probe: which
/// index is in use, the library, and the fragment tolerance. Bundling them keeps the
/// per-peak call down to what actually varies.
struct Prober<'a> {
    fidx: Option<&'a FragIndex>,
    lib: &'a Library,
    frag_tol: f64,
}

impl Prober<'_> {
    /// Probe one (offset-corrected) query m/z, invoking `f(candidate_id,
    /// candidate_local_fragment_ordinal, predicted_intensity)` for every verified
    /// match in the candidate window `[lo, hi)`. Bucketed resolves the fragment
    /// ordinal via `Library::local_frag_index` (nearest stored m/z); fragindex
    /// carries the true generating ordinal in `post_frag` (a semantic change for
    /// candidates with fragments at sub-f32-identical m/z, per the plan). Both apply
    /// the same tolerance; the fragindex index is already built at `frag_tol`.
    ///
    /// `nw` is an optional per-window narrowing cache for the fragindex path; it must
    /// have been built for this same `(lo, hi)`. Passing `None` is always correct and
    /// gives the uncached probe.
    #[inline]
    fn probe(
        &self,
        nw: Option<&mut WindowNarrow>,
        q_mz: f64,
        lo: u32,
        hi: u32,
        f: &mut dyn FnMut(u32, u16, f32),
    ) {
        match (self.fidx, nw) {
            (Some(idx), Some(nw)) => {
                idx.probe_peak_win(nw, q_mz, |cid, _pmz, pint, frag| f(cid, frag, pint))
            }
            (Some(idx), None) => {
                idx.probe_peak(q_mz, lo, hi, |cid, _pmz, pint, frag| f(cid, frag, pint))
            }
            (None, _) => {
                let lib = self.lib;
                lib.page_search(q_mz, self.frag_tol, lo, hi, |cid, frag_mz, pi| {
                    let frag = lib.local_frag_index(cid, frag_mz) as u16;
                    f(cid, frag, pi);
                })
            }
        }
    }
}

pub struct ExtractParams<'a> {
    pub ms2: &'a str,
    pub library_precursors: &'a str,
    pub library_fragments: &'a str,
    pub run_windows: &'a str,
    /// Optional MS1 spectra for precursor isotope-envelope features
    /// (docs/10_features.md). When absent, MS1 columns are null.
    pub ms1: Option<&'a str>,
    /// Optional per-run mass recalibration (search-seed `<seed>.masscal.json`):
    /// systematic fragment ppm offset + learned tolerance.
    pub mass_cal: Option<&'a str>,
    pub out_psms: &'a str,
    pub out_chrom: &'a str,
    /// Optional candidate allowlist (a prior run's `psms.parquet`): restrict
    /// extraction to these `candidate_id`s. Used for "gate first, then compete":
    /// run a cheap gate-on pass, then re-extract with a peak-claim strategy over
    /// only the accepted survivors, so the expensive two-pass profile map is built
    /// over ~10^5 candidates instead of ~10^7.
    pub restrict_candidates: Option<&'a str>,
    pub cfg: &'a ExtractConfig,
    pub config_hash: &'a str,
    /// The precursor table is one isolation-window group's band with local ids `0..n`, and
    /// its fragments sit in the library-wide fragment table at ids `offset..offset + n`.
    /// `run_windows` and the outputs are then in band-local ids; the spectra of every window
    /// are still read, and scans of other windows find no candidate. `None` is the ordinary
    /// whole-library extract.
    pub fragment_offset: Option<u32>,
    /// `library_precursors` is the WHOLE library and this stage searches its rows
    /// `[first, first + n)`, loaded directly by row span (`Library::load_row_span_with`):
    /// local ids `0..n`, fragments at library-wide ids `first..first + n`, outputs in local
    /// ids exactly as for a band file. A grouped run loads a band this way wherever nothing
    /// rewrites the band's precursor table, instead of writing the band out first.
    /// `fragment_offset` is then `None` (or `Some(first)`). `None` is the ordinary load.
    pub precursor_span: Option<(usize, usize)>,
    /// How many bands of the same run are being searched beside this one
    /// (`groups.parallel`). The probing fan-out is this band's share of the thread pool,
    /// not the whole pool, because every band in flight computes it independently.
    pub sibling_bands: usize,
    /// This run's MS2 and MS1 scans, already decoded. A grouped search
    /// (`groups.window_groups > 1`) decodes the run once in `run_groups` and lends the
    /// same buffers to every band, because every band re-reads the whole run and only the
    /// library differs. `None` loads them from `ms2` / `ms1`, which is what a standalone
    /// `mumdia extract` and an ungrouped `run` do.
    ///
    /// Borrowed, never owned, and the stage only reads them. What makes that safe is not
    /// that nothing here corrects the observed m/z -- the per-peak mass recalibration
    /// corrects exactly that, `peak.mz / mass_off.factor_at(peak.mz)`, and each band
    /// applies its own factor under `groups.calibration = per_group`. It is safe because
    /// the corrected value is computed into a LOCAL (`q_mz`) at every one of those call
    /// sites and never written back, the scans arrive as a shared slice that no stage can
    /// write through, and the loaders have already sorted by retention time so no stage
    /// re-sorts. The destructive peak-claim strategies likewise rewrite this band's own
    /// `Hit` intensities, never `scans`. So no band can leave a trace in the scans the
    /// next band sees.
    ///
    /// If you ever hoist `q_mz` out of the per-peak loop, hoist it into a side buffer.
    /// Writing it back into the scans was harmless when each band decoded its own copy
    /// and silently corrupts every later band now. See the note above the probe loop.
    ///
    /// A grouped search lends both, because every band would otherwise decode the MS1
    /// again. The ungrouped `run` lends only the MS2 its seed already decoded
    /// (`SharedScans::ms1 = None`): it has no MS1 decode to share, and letting extract
    /// decode it keeps that decode concurrent with the library load and after the
    /// library's errors, where a standalone extract has it.
    pub scans: Option<SharedScans<'a>>,
    /// The windows `rt-im-train` just fitted and wrote to `run_windows`, handed over in
    /// memory by an orchestrator (`rt_im_train::run_in_memory`) so the table is not
    /// decoded again. Used only when they were fitted on this `library_precursors` table,
    /// written to this `run_windows` path, and cover exactly this library's candidates
    /// (`RtWindows::mismatch`), and then they are what the file would have produced;
    /// otherwise, and whenever this is `None`, the `run_windows` file is read. A standalone
    /// `mumdia extract` always reads the file.
    pub rt_windows: Option<crate::stages::rt_im_train::RtWindows>,
}

/// One run's decoded spectra, lent to a stage instead of being re-decoded by it.
///
/// An EMPTY slice here never means "this run has no such spectra". It means the caller
/// has nothing to lend, and the stage decodes the corresponding path itself. Saying "no
/// MS1" is `ExtractParams::ms1 = None`, as it always was. The two are not interchangeable:
/// an empty `ms1` believed over a named `ms1` path would drop every MS1 feature and every
/// `ms1_mono` / `ms1_iso1` / `ms1_iso2` chromatogram row, and write a plausible table
/// while doing it.
#[derive(Clone, Copy)]
pub struct SharedScans<'a> {
    pub ms2: &'a [Ms2Scan],
    /// `None`: the caller lends the MS2 only, and extract decodes the MS1 from
    /// `ExtractParams::ms1` itself, as it would with nothing lent. Not a warning case,
    /// unlike an empty slice, which says the caller meant to lend and had nothing.
    pub ms1: Option<&'a [Ms1Scan]>,
}

/// One observed hit: the scan it was observed in, candidate-local fragment index,
/// observed intensity and observed m/z (for mass-accuracy features).
///
/// 16 bytes. It used to be 24: the scan's RT as an f64 and the observed m/z widened to an
/// f64. Both were redundant. Every hit comes from one scan of the stage's `scans` slice, so
/// the scan index recovers the RT exactly (`scans[scan].rt_seconds`, the same f64 the hit
/// used to copy), and peaks are stored at f32 width, so the f32 m/z IS the value the f64
/// held, widened again where it is used. The hit payload is the stage's largest structure
/// (1.6 billion hits measured on one HYE run), so a third off it is a third off the
/// accumulator.
///
/// Order-sensitive code keeps reading the RT, not the index: the per-candidate pass sorts
/// and groups hits on the looked-up RT, and the two-pass elution profile keys on the RT's
/// bits, exactly as before. Scans are RT-sorted by the loaders, but a sort on the index
/// would still differ from one on the RT wherever two scans share an RT.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct Hit {
    /// Index into the stage's `scans` slice.
    scan: u32,
    frag: u16,
    inten: f32,
    obs_mz: f32,
}

const _: () = assert!(std::mem::size_of::<Hit>() == 16);

/// The retention time of `h`: the RT of the scan it was observed in.
#[inline]
fn hit_rt(scans: &[Ms2Scan], h: &Hit) -> f64 {
    scans[h.scan as usize].rt_seconds
}

/// Hits for many candidates in one flat buffer with per-candidate offsets (CSR).
///
/// The accumulator used to be a `HashMap<u32, Vec<Hit>>`: one heap block per candidate
/// with evidence, created on first collision and grown by doubling. A candidate with no
/// `run_windows` row gets infinite RT bounds and therefore collects hits across the whole
/// gradient -- 10^4 to 10^5 hits, i.e. 240 KB to 2.4 MB in a single block -- and mimalloc
/// maps blocks that size individually. Measured on a live grouped search: 129,393 mappings
/// at 180 GB resident, 92,030 of them 64-256 KB and 36,368 of them 256 KB-1 MB, with the
/// process dying at ~290 GB on the per-process mapping limit (`vm.max_map_count` =
/// 1,048,576) while 1.7 TB of RAM was free. The CSR layout holds the same hits, for the
/// same candidates, in the same per-candidate order, in three allocations per store.
struct HitStore {
    /// Candidate ids owning a slice of `hits`, strictly ascending.
    cids: Vec<u32>,
    /// `offs[i]..offs[i + 1]` is `cids[i]`'s slice of `hits`; length `cids.len() + 1` with
    /// `offs[0] == 0`. `usize` rather than `u32`: one run has been measured at 1.6 billion
    /// hits, past what a 32-bit offset can address.
    offs: Vec<usize>,
    hits: Vec<Hit>,
}

impl Default for HitStore {
    fn default() -> Self {
        HitStore {
            cids: Vec::new(),
            offs: vec![0],
            hits: Vec::new(),
        }
    }
}

impl HitStore {
    fn len(&self) -> usize {
        self.cids.len()
    }

    fn is_empty(&self) -> bool {
        self.cids.is_empty()
    }

    fn clear(&mut self) {
        self.cids.clear();
        self.offs.clear();
        self.offs.push(0);
        self.hits.clear();
    }

    /// Candidate `i`'s hits.
    fn slice(&self, i: usize) -> &[Hit] {
        &self.hits[self.offs[i]..self.offs[i + 1]]
    }

    /// Append `hits` to candidate `cid`, which must be >= the last id pushed. When it is
    /// equal the hits extend that candidate's existing segment, which is how one
    /// candidate's hits from several windows end up contiguous and in window order.
    fn push_segment(&mut self, cid: u32, hits: &[Hit]) {
        if hits.is_empty() {
            return;
        }
        debug_assert!(self.cids.last().map(|&l| l <= cid).unwrap_or(true));
        if self.cids.last() != Some(&cid) {
            self.cids.push(cid);
            self.offs.push(0);
        }
        self.hits.extend_from_slice(hits);
        *self.offs.last_mut().expect("offs is never empty") = self.hits.len();
    }

    /// One disjoint `&mut [Hit]` per candidate, ascending by id: what the per-candidate
    /// pass consumes in place of the `Vec<Hit>` it used to be handed. The sort it runs is
    /// a slice sort either way, so the hits it sees and the order it sees them in are
    /// exactly those of the owned vector.
    fn slices_mut(&mut self) -> Vec<(u32, &mut [Hit])> {
        let n = self.cids.len();
        self.slices_mut_range(0, n)
    }

    /// [`HitStore::slices_mut`] for candidates `a..b` of the store only.
    fn slices_mut_range(&mut self, a: usize, b: usize) -> Vec<(u32, &mut [Hit])> {
        let HitStore { cids, offs, hits } = self;
        let mut out = Vec::with_capacity(b - a);
        let mut base = offs[a];
        let mut rest: &mut [Hit] = &mut hits[base..offs[b]];
        for i in a..b {
            let end = offs[i + 1];
            let (head, tail) = rest.split_at_mut(end - base);
            out.push((cids[i], head));
            rest = tail;
            base = end;
        }
        out
    }
}

/// Group `(cid, hit)` pairs by candidate with a STABLE counting sort applied in place,
/// and return the CSR spine (`cids`, `offs`) for the grouped `hits`.
///
/// Stability is the whole contract: each candidate's hits must come out in exactly the
/// order the probe produced them (ascending scan order, and within a scan the order the
/// posting list emitted), because that is the order the per-candidate `Vec<Hit>` held them
/// in and every float reduction downstream is order-sensitive.
///
/// The permutation is applied by cycle following rather than scattering into a second
/// buffer: a second buffer would double the stage's largest structure. `cid` is consumed
/// as scratch (it becomes the destination slot of each hit). Every `cid` must lie in
/// `[lo, hi)`.
fn group_hits_by_candidate(
    lo: u32,
    hi: u32,
    cid: &mut [u32],
    hits: &mut [Hit],
) -> (Vec<u32>, Vec<usize>) {
    debug_assert_eq!(cid.len(), hits.len());
    let n = hits.len();
    if n == 0 {
        return (Vec::new(), vec![0]);
    }
    assert!(
        n <= u32::MAX as usize,
        "a single probing task produced {n} hits, past the u32 slot index"
    );
    let span = (hi - lo) as usize;
    // `cum[c]` is first candidate `lo + c`'s hit count, then (after the prefix sum) its
    // first slot, and finally -- the scatter having advanced it once per hit -- one past
    // its last, i.e. the cumulative count through `c`. One span-sized u32 array, 4 bytes
    // per candidate of the sub-range, serves all three roles.
    let mut cum: Vec<u32> = vec![0; span];
    for &c in cid.iter() {
        debug_assert!(c >= lo && c < hi);
        cum[(c - lo) as usize] += 1;
    }
    let mut running = 0u32;
    for v in cum.iter_mut() {
        let k = *v;
        *v = running;
        running += k;
    }
    debug_assert_eq!(running as usize, n);
    // Walking in order is what makes the sort stable: `cid[i]` becomes the slot hit `i`
    // belongs in, and equal candidates take consecutive slots in arrival order.
    for c in cid.iter_mut() {
        let b = (*c - lo) as usize;
        *c = cum[b];
        cum[b] += 1;
    }
    for i in 0..n {
        while cid[i] as usize != i {
            let j = cid[i] as usize;
            hits.swap(i, j);
            cid.swap(i, j);
        }
    }
    // `cum[c]` now holds the cumulative count through candidate `c`, so a candidate is
    // occupied exactly where that count grew.
    let mut cids: Vec<u32> = Vec::new();
    let mut offs: Vec<usize> = vec![0];
    let mut prev = 0u32;
    for (c, &end) in cum.iter().enumerate() {
        if end > prev {
            cids.push(lo + c as u32);
            offs.push(end as usize);
        }
        prev = end;
    }
    (cids, offs)
}

/// One probed window's stores read as a single ascending candidate sequence. The
/// sub-range tasks of a window cover disjoint, ascending candidate spans, so their stores
/// concatenate without a merge.
struct HitRun {
    parts: Vec<HitStore>,
    pi: usize,
    ci: usize,
}

impl HitRun {
    fn new(parts: Vec<HitStore>) -> HitRun {
        let mut r = HitRun {
            parts,
            pi: 0,
            ci: 0,
        };
        r.settle();
        r
    }

    fn settle(&mut self) {
        while self.pi < self.parts.len() && self.ci >= self.parts[self.pi].len() {
            self.pi += 1;
            self.ci = 0;
        }
    }

    fn peek(&self) -> Option<u32> {
        self.parts.get(self.pi).map(|p| p.cids[self.ci])
    }

    fn current(&self) -> &[Hit] {
        self.parts[self.pi].slice(self.ci)
    }

    fn advance(&mut self) {
        self.ci += 1;
        self.settle();
    }

    /// Step past `m` candidates of the current part.
    fn advance_by(&mut self, m: usize) {
        self.ci += m;
        self.settle();
    }

    /// The next `m` candidates, all in the current part, as `(cid, hits)` slices into the
    /// run's own store: nothing is copied.
    fn span_mut(&mut self, m: usize) -> Vec<(u32, &mut [Hit])> {
        let ci = self.ci;
        self.parts[self.pi].slices_mut_range(ci, ci + m)
    }

    /// Hits not yet gathered out of this run.
    fn hits_left(&self) -> usize {
        match self.parts.get(self.pi) {
            None => 0,
            Some(p) => {
                (p.hits.len() - p.offs[self.ci])
                    + self.parts[self.pi + 1..]
                        .iter()
                        .map(|q| q.hits.len())
                        .sum::<usize>()
            }
        }
    }

    /// Free the stores already fully gathered.
    fn release_consumed(&mut self) {
        if self.pi > 0 {
            self.parts.drain(..self.pi);
            self.pi = 0;
        }
    }
}

/// Receive one probing result, executing pending pool work while waiting instead of
/// parking.
///
/// The consumer runs on the thread that called `accumulate_groups`, and under
/// `groups.parallel` that thread is itself a rayon worker: a plain blocking `recv` parks
/// one worker per band in flight, so 24 bands parked 24 of 32 threads and the pool ran on
/// what was left. `yield_now` executes one queued task instead, which is usually one of
/// this batch's own probes. When the pool has nothing queued it returns `Idle` (or `None`
/// off a worker thread), and a short blocking wait then avoids a spin.
///
/// Scheduling only: the results are still consumed in arrival order and reordered by
/// sub-range, so nothing about the output depends on this.
fn recv_participating<T>(rx: &std::sync::mpsc::Receiver<T>) -> T {
    use std::sync::mpsc::{RecvTimeoutError, TryRecvError};
    loop {
        match rx.try_recv() {
            Ok(v) => return v,
            Err(TryRecvError::Disconnected) => {
                panic!("every probing task sends exactly one store")
            }
            Err(TryRecvError::Empty) => {}
        }
        if let Some(rayon::Yield::Executed) = rayon::yield_now() {
            continue;
        }
        match rx.recv_timeout(std::time::Duration::from_millis(1)) {
            Ok(v) => return v,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                panic!("every probing task sends exactly one store")
            }
        }
    }
}

/// Gather up to `max_cands` candidates below `bound` out of `runs`, ascending by candidate
/// id, appending each candidate's hits run by run. Runs are held in window order, so a
/// candidate seen in several windows keeps exactly the hit sequence the serial path
/// produced: ascending scan order within a window, windows concatenated.
///
/// `bound` is the first candidate a later window can still add hits to, and `u32::MAX`
/// when nothing is pending, which no real candidate id reaches (ids are dense `0..ncand`).
fn gather_chunk(runs: &mut [HitRun], bound: u32, max_cands: usize, out: &mut HitStore) {
    out.clear();
    while out.len() < max_cands {
        let mut min_cid: Option<u32> = None;
        for r in runs.iter() {
            if let Some(c) = r.peek() {
                if min_cid.map(|m| c < m).unwrap_or(true) {
                    min_cid = Some(c);
                }
            }
        }
        let cid = match min_cid {
            Some(c) if c < bound => c,
            _ => break,
        };
        for r in runs.iter_mut() {
            if r.peek() == Some(cid) {
                out.push_segment(cid, r.current());
                r.advance();
            }
        }
    }
}

/// When the next [`gather_chunk`] of `(bound, max_cands)` would copy the hits of candidates
/// that ALL sit in one run's current part, each as that candidate's only segment, return
/// `(run, m)`: that run and how many of its candidates the gather would take. Flushing
/// them straight out of the run's store then hands the flush exactly the batch the gather
/// would have built, candidate for candidate and hit for hit, without the copy.
///
/// The batch has to be the SAME batch, not merely the same candidates, because each flush
/// call becomes one chromatogram chunk and the parquet page framing follows the chunk
/// sizes. So a span is taken only when the gather could not have gone past it: it filled
/// `max_cands`, or nothing else below `bound` remains, in any run or in a later part of this
/// one. Otherwise `None`, and the caller gathers.
fn single_run_span(runs: &[HitRun], bound: u32, max_cands: usize) -> Option<(usize, usize)> {
    // The run holding the smallest pending candidate, and the smallest of every other run.
    let mut best: Option<(usize, u32)> = None;
    let mut other_min = u32::MAX;
    for (i, r) in runs.iter().enumerate() {
        let Some(c) = r.peek() else { continue };
        match best {
            None => best = Some((i, c)),
            Some((_, bc)) if c < bc => {
                other_min = other_min.min(bc);
                best = Some((i, c));
            }
            Some(_) => other_min = other_min.min(c),
        }
    }
    let (r, first) = best?;
    // Nothing flushable, or the first candidate has a segment in another run too.
    if first >= bound || other_min == first || max_cands == 0 {
        return None;
    }
    let run = &runs[r];
    let part = &run.parts[run.pi];
    let limit = bound.min(other_min);
    let m = part.cids[run.ci..]
        .partition_point(|&c| c < limit)
        .min(max_cands);
    if m == max_cands {
        return Some((r, m));
    }
    // A short span: the gather would go on to whatever else lies below `bound`.
    if other_min < bound {
        return None;
    }
    if run.ci + m == part.len() {
        let next = run.parts[run.pi + 1..]
            .iter()
            .find(|q| !q.is_empty())
            .map(|q| q.cids[0]);
        if next.is_some_and(|c| c < bound) {
            return None;
        }
    }
    Some((r, m))
}

/// A flush of finished candidates: `(cid, hits)` in ascending id, at most `CAND_CHUNK` of
/// them. Returns false when the consumer went away.
type FlushFn<'f> = dyn for<'h> FnMut(Vec<(u32, &'h mut [Hit])>) -> bool + Send + 'f;

/// Flush every candidate of `runs` below `bound`, `max_cands` at a time, in ascending id.
/// Returns true when `flush` asked to stop.
///
/// A batch whose candidates all come from one run's store, each as its only segment, is
/// flushed straight out of that store ([`single_run_span`]); every other batch is gathered
/// into `chunk` first. That is the common case by far: a sub-range reached by one window
/// and holding no leftovers from the batch before. `zero_copy = false` always gathers,
/// which is the reference the zero-copy batches are tested against.
fn flush_below(
    runs: &mut [HitRun],
    bound: u32,
    max_cands: usize,
    zero_copy: bool,
    chunk: &mut HitStore,
    flush: &mut FlushFn<'_>,
) -> bool {
    loop {
        if zero_copy {
            if let Some((r, m)) = single_run_span(runs, bound, max_cands) {
                let ok = flush(runs[r].span_mut(m));
                runs[r].advance_by(m);
                if !ok {
                    return true;
                }
                continue;
            }
        }
        gather_chunk(runs, bound, max_cands, chunk);
        if chunk.is_empty() {
            return false;
        }
        if !flush(chunk.slices_mut()) {
            return true;
        }
    }
}

/// The streamed accumulator: one run per probed window, oldest first.
#[derive(Default)]
struct HitAcc {
    runs: Vec<HitRun>,
}

impl HitAcc {
    fn n_hits(&self) -> usize {
        self.runs.iter().map(|r| r.hits_left()).sum()
    }

    /// Free what has been gathered and fold whatever is left into a single run, so the
    /// number of live runs stays bounded however much the windows overlap.
    fn compact(&mut self) {
        for r in self.runs.iter_mut() {
            r.release_consumed();
        }
        self.runs.retain(|r| r.peek().is_some());
        if self.runs.len() > 1 {
            let mut merged = HitStore::default();
            gather_chunk(&mut self.runs, u32::MAX, usize::MAX, &mut merged);
            self.runs = vec![HitRun::new(vec![merged])];
        }
    }
}

type ChromOutputRow = (u32, String, f64, f64, f32, Vec<f32>, Vec<f32>);

/// Candidates per parallel chunk of the per-candidate pass: large enough to keep every
/// core busy within a chunk, small enough that a chunk's chromatogram rows are tens of MB.
const CAND_CHUNK: usize = 8192;
/// Rows per parquet row group of the chromatogram table (~64k rows of two ~60-point traces
/// is ~30 MB uncompressed), which bounds the encoder's in-progress buffer. Defined with the
/// layouts in [`crate::chromatograms`], because the v2 axis rule restarts at these
/// boundaries; the writer takes [`crate::chromatograms::row_group_rows`], which is this
/// unless the test knob moves it.
const CHROM_ROW_GROUP_ROWS: usize = crate::chromatograms::ROW_GROUP_ROWS;

/// One chunk of chromatogram rows, drained into the column set of the configured layout
/// (`extract.chromatogram_schema`, [`crate::chromatograms::Layout`]).
#[derive(Default)]
struct ChromChunk {
    rows: crate::chromatograms::Rows,
}

impl ChromChunk {
    /// `offset` is the library row of this band's local id 0 (`Library::global_offset`),
    /// so the table carries library-wide ids even when the stage searched one band.
    fn cols(mut self, offset: u32, layout: crate::chromatograms::Layout) -> Vec<Col> {
        if offset != 0 {
            for c in &mut self.rows.cid {
                *c += offset;
            }
        }
        self.rows.into_cols(layout, true)
    }
}

/// One observed peak for the per-scan demix: (observed intensity, observed m/z at the
/// artifact's f32 width, claimants as (candidate_id, fragment_ordinal,
/// predicted_intensity)).
type DemixRow = (f32, f32, Vec<(u32, u16, f32)>);

/// Per-candidate contested-peak statistics from the co-elution arbitration
/// (two-pass path). `won`/`lost` are the summed observed intensity of shared peaks
/// this candidate won (was the most-eluting claimant) or lost to a better
/// co-eluter; `n_won`/`n_lost` are the corresponding peak-instance counts; and
/// `apportioned` is the candidate's co-elution-weighted proportional share of the
/// contested intensity (what it would keep under `CoelutionProportional`). These
/// feed the soft competition features (`contested_frac`, `contested_count_frac`,
/// `apportioned_frac`) without removing any candidate.
#[derive(Default, Clone, Copy)]
struct Contested {
    won: f64,
    lost: f64,
    n_won: u32,
    n_lost: u32,
    apportioned: f64,
}

/// The `--restrict-candidates` allowlist, as a bitset over the dense candidate id range.
///
/// The allowlist used to be a `HashSet<u32>` probed once per VERIFIED POSTING, inside the
/// probe callback: a hash and a bucket load for every fragment match of every peak of
/// every scan. Candidate ids are dense `0..ncand` (`index.rs`), so one bit per candidate
/// answers the same question with a shift and a load, and it is one allocation of
/// `ncand / 8` bytes instead of a hash table of the allowlist -- smaller as soon as the
/// allowlist holds more than about a sixty-fourth of the library, which is the case the
/// allowlist exists for (a prior gate-on pass over a 35M-83M candidate library).
struct CandMask {
    bits: Vec<u64>,
    n: usize,
}

impl CandMask {
    fn new(n: usize) -> CandMask {
        CandMask {
            bits: vec![0; n.div_ceil(64)],
            n,
        }
    }

    fn insert(&mut self, c: u32) {
        let i = c as usize;
        if i < self.n {
            self.bits[i >> 6] |= 1u64 << (i & 63);
        }
    }

    /// Ids outside `0..n` are absent. The probe only ever asks about ids it produced, so
    /// they are in range by construction; an out-of-range id in the allowlist file could
    /// not have matched the hash set either, because no probe would ask for it.
    #[inline]
    fn contains(&self, c: u32) -> bool {
        let i = c as usize;
        i < self.n && self.bits[i >> 6] & (1u64 << (i & 63)) != 0
    }

    fn len(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }
}

/// Ordinals a [`FragSet`] holds in its inline bitmask; above this it spills to a sorted
/// vector. A candidate carries a few dozen predicted fragments (6 by default, 12 in the
/// shipped DIA-NN library), so the spill is unreachable in practice and costs nothing when
/// it is not used (an empty `Vec` does not allocate).
const FRAG_SET_BITS: usize = 512;

/// A set of a candidate's local fragment ordinals.
///
/// This replaces `hits.iter().map(|h| h.frag).collect::<Vec<u16>>()` followed by a sort and
/// a dedup, which allocated one `u16` per HIT and sorted it: a candidate with no
/// `run_windows` row collects 10^4-10^5 hits, so that was a 20-200 KB allocation and an
/// `n log n` sort to recover at most a few dozen distinct values.
#[derive(Default)]
struct FragSet {
    bits: [u64; FRAG_SET_BITS / 64],
    spill: Vec<u16>,
}

impl FragSet {
    #[inline]
    fn insert(&mut self, f: u16) {
        let i = f as usize;
        if i < FRAG_SET_BITS {
            self.bits[i >> 6] |= 1u64 << (i & 63);
        } else if let Err(p) = self.spill.binary_search(&f) {
            self.spill.insert(p, f);
        }
    }

    #[inline]
    fn contains(&self, f: u16) -> bool {
        let i = f as usize;
        if i < FRAG_SET_BITS {
            self.bits[i >> 6] & (1u64 << (i & 63)) != 0
        } else {
            self.spill.binary_search(&f).is_ok()
        }
    }

    fn len(&self) -> usize {
        self.bits
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum::<usize>()
            + self.spill.len()
    }

    /// The ordinals, ascending: exactly what the sorted, deduplicated vector held.
    fn to_vec(&self) -> Vec<u16> {
        let mut v = Vec::with_capacity(self.len());
        for (w, &word) in self.bits.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let b = word.trailing_zeros() as usize;
                v.push((w * 64 + b) as u16);
                word &= word - 1;
            }
        }
        v.extend_from_slice(&self.spill);
        v
    }
}

/// One candidate's scan groups: per group its RT and, for each fragment ordinal
/// `0..width`, the observed intensity when the fragment was seen in that group.
///
/// This replaces `Vec<(f64, BTreeMap<u16, f32>)>`, one tree per scan group, which on the
/// window grid meant one tree per grid scan and a node allocation for every group that
/// held a fragment. Here the whole candidate is three flat buffers: the RTs, a dense
/// `groups x width` value array and a presence bitmask of the same shape. `width` is one
/// past the largest ordinal among the candidate's hits, not the number of fragments it
/// observed, so the buffers are `groups x (max observed ordinal + 1)` f32 values plus
/// `groups x ceil(width / 64)` presence words. Ordinals are candidate-local, so that is
/// at most `groups x n_predicted_fragments` values. It can exceed what the candidate
/// emits: one that observed only ordinal 11 holds 12 values per group against one trace,
/// and in sparse (non-grid) mode a trace covers only the groups where its fragment
/// occurs.
///
/// It answers every question the trees answered with the same values in the same order:
/// `count` is the tree's `len`, `frags` its keys ascending, `sum` its values summed in key
/// order through the same `Iterator::sum`, `get` its lookup (`None` for an absent
/// fragment, including one past `width`). Presence is a bit, not a sentinel value, because
/// an observed intensity can be 0.0 (or negative in a hand-made artifact) and must still
/// count as present.
struct ScanGroups {
    rt: Vec<f64>,
    width: usize,
    /// `u64` words of presence per group.
    words: usize,
    val: Vec<f32>,
    bits: Vec<u64>,
}

impl ScanGroups {
    /// No groups yet, room for ordinals `0..width`.
    fn new(width: usize) -> ScanGroups {
        ScanGroups {
            rt: Vec::new(),
            width,
            words: width.div_ceil(64),
            val: Vec::new(),
            bits: Vec::new(),
        }
    }

    /// One empty group per RT of `rts`.
    fn empty_on(rts: &[f64], width: usize) -> ScanGroups {
        let words = width.div_ceil(64);
        ScanGroups {
            rt: rts.to_vec(),
            width,
            words,
            val: vec![0.0; rts.len() * width],
            bits: vec![0; rts.len() * words],
        }
    }

    fn len(&self) -> usize {
        self.rt.len()
    }

    fn is_empty(&self) -> bool {
        self.rt.is_empty()
    }

    #[inline]
    fn rt(&self, i: usize) -> f64 {
        self.rt[i]
    }

    /// Open a new, empty group at `rt`.
    fn push_group(&mut self, rt: f64) {
        self.rt.push(rt);
        self.val.resize(self.val.len() + self.width, 0.0);
        self.bits.resize(self.bits.len() + self.words, 0);
    }

    #[inline]
    fn present(&self, i: usize, f: usize) -> bool {
        f < self.width && self.bits[i * self.words + f / 64] >> (f % 64) & 1 == 1
    }

    /// Fragment `f`'s intensity in group `i`, if it was observed there.
    #[inline]
    fn get(&self, i: usize, f: u16) -> Option<f32> {
        let f = f as usize;
        self.present(i, f).then(|| self.val[i * self.width + f])
    }

    /// [`ScanGroups::get`], 0.0 when absent: the tree's `get(..).unwrap_or(0.0)`.
    #[inline]
    fn or_zero(&self, i: usize, f: u16) -> f32 {
        self.get(i, f).unwrap_or(0.0)
    }

    /// Distinct fragments observed in group `i`.
    fn count(&self, i: usize) -> usize {
        self.bits[i * self.words..(i + 1) * self.words]
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    /// The fragments observed in group `i`, ascending.
    fn frags(&self, i: usize) -> impl Iterator<Item = u16> + '_ {
        let row = &self.bits[i * self.words..(i + 1) * self.words];
        row.iter().enumerate().flat_map(|(w, &word)| {
            let mut word = word;
            std::iter::from_fn(move || {
                (word != 0).then(|| {
                    let b = word.trailing_zeros() as usize;
                    word &= word - 1;
                    (w * 64 + b) as u16
                })
            })
        })
    }

    /// The observed intensities of group `i` summed in ascending fragment order: the tree's
    /// `values().sum()`.
    fn sum(&self, i: usize) -> f32 {
        let base = i * self.width;
        self.frags(i)
            .map(|f| self.val[base + f as usize])
            .sum::<f32>()
    }

    /// The first hit of a new group: the tree's `insert(frag, inten)`.
    #[inline]
    fn insert(&mut self, i: usize, f: u16, v: f32) {
        let f = f as usize;
        self.bits[i * self.words + f / 64] |= 1u64 << (f % 64);
        self.val[i * self.width + f] = v;
    }

    /// A later hit of the same group: the tree's `entry(frag).or_insert(0.0)` followed by
    /// `if v > *e { *e = v }`. A fragment first seen here therefore starts from 0.0, not
    /// from `v`, exactly as the tree did.
    #[inline]
    fn merge_max(&mut self, i: usize, f: u16, v: f32) {
        let fu = f as usize;
        if !self.present(i, fu) {
            self.insert(i, f, 0.0);
        }
        let e = &mut self.val[i * self.width + fu];
        if v > *e {
            *e = v;
        }
    }

    /// Replace group `j` with group `i` of `src` (same width).
    fn copy_group(&mut self, j: usize, src: &ScanGroups, i: usize) {
        debug_assert_eq!(self.width, src.width);
        let (w, k) = (self.width, self.words);
        self.val[j * w..(j + 1) * w].copy_from_slice(&src.val[i * w..(i + 1) * w]);
        self.bits[j * k..(j + 1) * k].copy_from_slice(&src.bits[i * k..(i + 1) * k]);
    }

    /// Distinct fragments observed anywhere in groups `lo..=hi`.
    fn count_union(&self, lo: usize, hi: usize) -> usize {
        (0..self.words)
            .map(|w| {
                (lo..=hi)
                    .map(|i| self.bits[i * self.words + w])
                    .fold(0u64, |a, b| a | b)
                    .count_ones() as usize
            })
            .sum()
    }

    /// Build from `(rt, [(frag, intensity)])` rows, each fragment inserted once: the
    /// fixtures of the gate-score tests, which were written against the tree form.
    #[cfg(test)]
    fn from_rows(rows: &[(f64, &[(u16, f32)])]) -> ScanGroups {
        let width = rows
            .iter()
            .flat_map(|(_, fs)| fs.iter().map(|(f, _)| *f as usize + 1))
            .max()
            .unwrap_or(0);
        let mut g = ScanGroups::new(width);
        for (rt, fs) in rows {
            g.push_group(*rt);
            let i = g.len() - 1;
            for &(f, v) in fs.iter() {
                g.insert(i, f, v);
            }
        }
        g
    }
}

/// Index of the value in ascending `rts` nearest to `t` (binary search).
fn nearest_index(rts: &[f64], t: f64) -> usize {
    if rts.is_empty() {
        return 0;
    }
    let p = rts.partition_point(|&r| r < t);
    if p == 0 {
        0
    } else if p >= rts.len() {
        rts.len() - 1
    } else if (t - rts[p - 1]).abs() <= (rts[p] - t).abs() {
        p - 1
    } else {
        p
    }
}

/// Composite per-claimant weight-cue multiplier for `PeakClaim::CoelutionMultiCue`
/// (modular fragment-competition framework). Each enabled [`ClaimCues`] cue contributes
/// a factor in [0,1]; disabled cues contribute 1.0, so the product is 1.0 when no cue
/// is on and the arbitration reduces exactly to the elution-profile-height weight.
/// Label-blind (reads only observed/predicted fragment m/z), so target/decoy
/// exchangeability is preserved.
#[inline]
#[allow(clippy::too_many_arguments)]
fn claim_cue_multiplier(
    cfg: &ExtractConfig,
    lib: &Library,
    cid: u32,
    frag: u16,
    obs_mz: f64,
    rt: f64,
    rt_cal: &[f64],
    ms1_scans: &[Ms1Scan],
    ms1_rts: &[f64],
) -> f32 {
    let mut w = 1.0f32;
    if cfg.claim_cues.mz_close {
        // Sub-tolerance m/z proximity (S3): the observed peak sits at the true owner's
        // m/z, so a claimant whose predicted fragment m/z is closer wins more weight.
        let (mzs, _, _) = lib.cand_frags(cid);
        let pred_mz = mzs.get(frag as usize).map(|m| *m as f64).unwrap_or(obs_mz);
        let ppm = mumdia_core::constants::ppm_diff(obs_mz, pred_mz) as f32;
        let sigma = (cfg.claim_cues.mz_close_sigma_ppm as f32).max(1e-6);
        w *= (-(ppm / sigma).powi(2)).exp();
    }
    if cfg.claim_cues.rt_prior {
        // DeepLC RT prior (S3): down-weight a claimant whose calibrated predicted RT is
        // far from the current scan (a briefly-co-eluting interferent). No-op when the
        // predicted RT is unset (0).
        let rt_pred = rt_cal.get(cid as usize).copied().unwrap_or(0.0);
        if rt_pred > 0.0 {
            let tau = (cfg.claim_cues.rt_prior_tau_s as f32).max(1e-3);
            let d = (rt - rt_pred) as f32;
            w *= (-(d * d) / (2.0 * tau * tau)).exp();
        }
    }
    if cfg.claim_cues.ms1_support && !ms1_scans.is_empty() {
        // MS1 precursor-envelope support (S4): the claimant's OWN precursor isotope
        // envelope at the nearest MS1 scan. Absent mono precursor -> down-weight; a
        // present mono with an implausible +1/mono ratio -> mild down-weight. A decoy's
        // precursor m/z is well-defined but has no real co-eluting MS1 signal.
        let cand = lib.cand(cid);
        let j = nearest_index(ms1_rts, rt);
        let s = &ms1_scans[j];
        let z = (cand.charge.max(1)) as f64;
        let sp = ISOTOPE_SPACING / z;
        let tol = cfg.prec_tol_ppm;
        let mono = sum_near(&s.mz, &s.intensity, cand.precursor_mz, tol);
        if mono <= 0.0 {
            w *= 0.5;
        } else {
            let i1 = sum_near(&s.mz, &s.intensity, cand.precursor_mz + sp, tol);
            let ratio = i1 / mono;
            if !(0.05..=2.0).contains(&ratio) {
                w *= 0.75;
            }
        }
    }
    w
}

/// Everything the spectrum-centric NNLS demix (D2, fragment-competition report) derives
/// from an apex SCAN alone: the co-isolated candidate x fragment-channel design matrix
/// (rows = the scan's observed peaks, columns = candidates that claim a peak,
/// `A[peak,cand]` = the candidate's predicted intensity for its matching fragment), the
/// observed vector, the NNLS solution of `min_{beta>=0} ||A beta - y||^2`, and the
/// quantities computed from those that do not depend on which candidate is asked about.
///
/// This used to be recomputed per CANDIDATE even though the candidate id only selects a
/// column and gates an early return, so every candidate re-probed every peak of its apex
/// scan and re-ran the NNLS. Solving once per scan and reading each candidate's column out
/// is exactly equivalent: nothing above the column read depends on the candidate. The
/// apex-scan lookup keys on the candidate's `apex_rt`, and the scan it resolves to has
/// `scan.rt_seconds == apex_rt` by construction, so the RT admission test is
/// scan-determined too.
///
/// Deterministic: peaks in scan order, candidate columns in sorted `cid` order, ordered
/// reductions.
struct DemixScan {
    /// candidate id -> design-matrix column, ascending by id.
    col_of: std::collections::BTreeMap<u32, usize>,
    /// Row-major `m x n` design matrix.
    a: Vec<f64>,
    /// Observed intensity per row.
    y: Vec<f64>,
    m: usize,
    n: usize,
    /// NNLS solution, one coefficient per column.
    beta: Vec<f64>,
    sum_beta: f64,
    /// Joint residual-explained fraction `1 - ||y - A beta||^2 / ||y||^2`.
    explained: f64,
    /// Per-column abundance seeded from the channels that column claims ALONE (its unique
    /// ions), NaN where it has none. Feeds the D1 shadow subtraction.
    a_p: Vec<f64>,
}

/// The acquisition scan at `apex_rt` whose isolation window covers `prec_mz`, if any.
/// Two candidates sharing an apex RT but sitting in different windows resolve to
/// different scans, so this index -- not the RT -- is what a solved problem is keyed by.
fn demix_apex_scan(
    scans: &[Ms2Scan],
    rt_scan: &HashMap<u64, Vec<u32>>,
    apex_rt: f64,
    prec_mz: f64,
) -> Option<usize> {
    rt_scan.get(&apex_rt.to_bits()).and_then(|v| {
        v.iter()
            .copied()
            .find(|&s| {
                let w = &scans[s as usize].window;
                w.lower_mz <= prec_mz && prec_mz <= w.upper_mz
            })
            .map(|s| s as usize)
    })
}

/// Assemble and solve the demix problem for one apex scan. `None` when the scan yields no
/// usable channels or columns.
#[allow(clippy::too_many_arguments)]
fn demix_solve_scan(
    idx: Option<&FragIndex>,
    lib: &Library,
    scan: &Ms2Scan,
    mass_off: &MassOffset,
    frag_tol: f64,
    apex_rt: f64,
    rt_lo: &[f64],
    rt_hi: &[f64],
    cfg: &ExtractConfig,
) -> Option<DemixScan> {
    let pr = Prober {
        fidx: idx,
        lib,
        frag_tol,
    };
    let (lo, hi) = lib.candidate_range(scan.window.lower_mz, scan.window.upper_mz);
    // Column set (candidates), deterministic by sorted cid. Rows carry (obs, claimants).
    let mut col_of: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    let mut rows: Vec<(f64, Vec<(u32, f32)>)> = Vec::new();
    let mut claimants: Vec<(u32, u16, f32)> = Vec::new();
    for peak in &scan.peaks {
        // Peaks are stored at the artifact's f32 width; widen once per peak. Exact, so
        // `mz` is the value the peak used to carry in an f64 field.
        let mz = peak.mz as f64;
        let q_mz = mz / mass_off.factor_at(mz);
        claimants.clear();
        {
            let mut push = |c: u32, frag: u16, pi: f32| {
                let cc = c as usize;
                if apex_rt < rt_lo[cc] || apex_rt > rt_hi[cc] {
                    return;
                }
                claimants.push((c, frag, pi));
            };
            pr.probe(None, q_mz, lo, hi, &mut push);
        }
        if claimants.is_empty() {
            continue;
        }
        let mut entry: Vec<(u32, f32)> = Vec::with_capacity(claimants.len());
        for &(c, _f, pi) in &claimants {
            if col_of.len() < cfg.demix_max_candidates || col_of.contains_key(&c) {
                col_of.entry(c).or_insert(0);
                entry.push((c, pi));
            }
        }
        if !entry.is_empty() {
            rows.push((peak.intensity as f64, entry));
        }
    }
    let n = col_of.len();
    let m = rows.len();
    if n == 0 || m == 0 {
        return None;
    }
    for (k, v) in col_of.values_mut().enumerate() {
        *v = k;
    }
    let mut a = vec![0.0f64; m * n];
    let mut y = vec![0.0f64; m];
    for (r, (obs, ents)) in rows.iter().enumerate() {
        y[r] = *obs;
        for &(c, pi) in ents {
            if let Some(&col) = col_of.get(&c) {
                // A candidate matching a peak via >1 fragment keeps its largest predicted
                // intensity for that channel (deterministic, order-independent).
                let cell = &mut a[r * n + col];
                if (pi as f64) > *cell {
                    *cell = pi as f64;
                }
            }
        }
    }
    let lambda = cfg.demix_lambda.max(1e-9);
    let beta = crate::solve::nnls(&a, m, n, &y, lambda, 200 * n.max(1));
    let sum_beta: f64 = beta.iter().sum();
    // Explained fraction = 1 - ||y - A beta||^2 / ||y||^2.
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (r, &yr) in y.iter().enumerate() {
        let mut pred = 0.0;
        for col in 0..n {
            pred += a[r * n + col] * beta[col];
        }
        let d = yr - pred;
        num += d * d;
        den += yr * yr;
    }
    let explained = if den > 0.0 {
        (1.0 - num / den).clamp(0.0, 1.0)
    } else {
        0.0
    };
    // Per-column abundance from the rows it claims ALONE: collect the unique-row ratios per
    // column, then take the median deterministically.
    let mut a_p = vec![f64::NAN; n];
    {
        let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); n];
        for r in 0..m {
            let mut nz = 0usize;
            let mut col = 0usize;
            for j in 0..n {
                if a[r * n + j] > 0.0 {
                    nz += 1;
                    col = j;
                }
            }
            if nz == 1 && a[r * n + col] > 0.0 {
                ratios[col].push(y[r] / a[r * n + col]);
            }
        }
        for (j, rs) in ratios.iter_mut().enumerate() {
            if !rs.is_empty() {
                rs.sort_by(|x, z| x.total_cmp(z));
                a_p[j] = rs[rs.len() / 2];
            }
        }
    }
    Some(DemixScan {
        col_of,
        a,
        y,
        m,
        n,
        beta,
        sum_beta,
        explained,
        a_p,
    })
}

/// `(deconv_explained_frac, deconv_active, deconv_share, deconv_max_collinearity,
/// shadow_kept_frac)` for one candidate.
type DemixFeatures = (f64, f64, f64, f64, f64);

/// Read candidate `cid`'s demix features out of its apex scan's solved problem:
/// `(deconv_explained_frac, deconv_active, deconv_share, deconv_max_collinearity,
/// shadow_kept_frac)`. Zeros when the candidate is not one of the scan's columns.
fn demix_features_for(d: &DemixScan, cid: u32) -> DemixFeatures {
    let c_col = match d.col_of.get(&cid) {
        Some(&c) => c,
        None => return (0.0, 0.0, 0.0, 0.0, 0.0),
    };
    let (a, y, m, n) = (&d.a, &d.y, d.m, d.n);
    let coef = d.beta[c_col].max(0.0);
    let share = if d.sum_beta > 0.0 {
        coef / d.sum_beta
    } else {
        0.0
    };
    let active = if coef > 1e-6 { 1.0 } else { 0.0 };
    // D1 shadow-spectrum: subtract every OTHER candidate's unique-ion-seeded contribution
    // from candidate c's channels and measure how much of c's observed intensity survives.
    // A real second peptide keeps most of its signal; a pure borrower is subtracted away.
    // This sidesteps the shared-peak circularity by seeding abundances from unique ions
    // only. 1.0 (kept all) when c has no interfering neighbor with unique ions.
    let shadow_kept = {
        let (mut kept, mut total) = (0.0f64, 0.0f64);
        for r in 0..m {
            let dc = a[r * n + c_col];
            if dc <= 0.0 {
                continue;
            }
            total += y[r];
            let mut sub = 0.0;
            for j in 0..n {
                if j != c_col && d.a_p[j].is_finite() {
                    sub += d.a_p[j] * a[r * n + j];
                }
            }
            kept += (y[r] - sub).max(0.0);
        }
        if total > 0.0 {
            (kept / total).clamp(0.0, 1.0)
        } else {
            1.0
        }
    };
    // Identifiability (D3): the maximum cosine similarity of candidate c's design column
    // with any other column. Near 1 = c is near-degenerate with a rival, so its
    // coefficient is an essentially arbitrary split (distrust the demix). No incumbent
    // engine emits this. O(m n), reuses the assembled matrix.
    let norm_c = {
        let mut s = 0.0;
        for r in 0..m {
            let v = a[r * n + c_col];
            s += v * v;
        }
        s.sqrt()
    };
    let mut max_collin = 0.0f64;
    if norm_c > 0.0 {
        for j in 0..n {
            if j == c_col {
                continue;
            }
            let (mut dot, mut nj) = (0.0f64, 0.0f64);
            for r in 0..m {
                let vc = a[r * n + c_col];
                let vj = a[r * n + j];
                dot += vc * vj;
                nj += vj * vj;
            }
            let nj = nj.sqrt();
            if nj > 0.0 {
                max_collin = max_collin.max(dot / (norm_c * nj));
            }
        }
    }
    (
        d.explained,
        active,
        share,
        max_collin.clamp(0.0, 1.0),
        shadow_kept,
    )
}

/// Sum intensities of peaks within `tol_ppm` of `target` (m/z-sorted arrays).
fn sum_near(mz: &[f32], inten: &[f32], target: f64, tol_ppm: f64) -> f32 {
    if mz.is_empty() {
        return 0.0;
    }
    let (lo, hi) = ppm_bounds(target, tol_ppm);
    // MS1 m/z is stored as f32 (the artifact's precision); widening at the comparison
    // yields exactly the values the previous `Vec<f64>` copy held.
    let s = mz.partition_point(|&m| (m as f64) < lo);
    let mut acc = 0.0f32;
    let mut i = s;
    while i < mz.len() && (mz[i] as f64) <= hi {
        acc += inten[i];
        i += 1;
    }
    acc
}

/// Co-elution acceptance score (sensitivity program): predicted-intensity-weighted
/// mean Pearson correlation of each matched fragment's XIC to the signature-ion
/// reference profile, over the elution scan groups. High when the peptide's own
/// fragments co-elute (real); low when a matched fragment is a non-co-eluting
/// interferent that only coincides at the apex. More robust to chimeric DIA
/// interference than the single-scan apex intensity Pearson. Returns 1.0 (do not
/// reject) when there are too few scan groups or no reference signal.
/// Contiguous elution-peak scan indices `[lo, hi]` around the signature-ion apex
/// (scans above 10% of the reference apex height) plus the reference profile.
/// `None` when there are too few scans, no reference signal, or a < 3-scan peak.
/// Over the full (wide) extraction window the traces are mostly zeros and any
/// correlation is noise; the spectral/co-elution gates are only meaningful across
/// the elution peak itself.
fn peak_window(groups: &ScanGroups, sig: &[u16]) -> Option<(usize, usize, Vec<f64>)> {
    if groups.len() < 3 {
        return None;
    }
    let refp: Vec<f64> = (0..groups.len())
        .map(|i| {
            sig.iter()
                .map(|&o| groups.or_zero(i, o) as f64)
                .sum::<f64>()
        })
        .collect();
    let (apex, apex_v) =
        refp.iter().enumerate().fold(
            (0usize, 0.0f64),
            |(bi, bv), (i, v)| if *v > bv { (i, *v) } else { (bi, bv) },
        );
    if apex_v <= 0.0 {
        return None;
    }
    let thr = 0.1 * apex_v;
    let (mut lo, mut hi) = (apex, apex);
    while lo > 0 && refp[lo - 1] >= thr {
        lo -= 1;
    }
    while hi + 1 < refp.len() && refp[hi + 1] >= thr {
        hi += 1;
    }
    if hi - lo + 1 < 3 {
        return None;
    }
    Some((lo, hi, refp))
}

/// Peak-integrated spectral Pearson: correlate the PEAK-SUMMED observed spectrum
/// (each predicted fragment integrated over the elution-peak scans) with the
/// predicted intensities. Averaging over the peak removes the single-interfered-
/// scan fragility of the apex-only Pearson. Returns 1.0 when no peak is resolved.
fn peak_spectral_score(groups: &ScanGroups, sig: &[u16], fints0: &[f32]) -> f64 {
    let (lo, hi, _refp) = match peak_window(groups, sig) {
        Some(w) => w,
        None => return 1.0,
    };
    let obs: Vec<f64> = (0..fints0.len())
        .map(|f| {
            (lo..=hi)
                .map(|i| groups.or_zero(i, f as u16) as f64)
                .sum::<f64>()
        })
        .collect();
    let pred: Vec<f64> = fints0.iter().map(|x| *x as f64).collect();
    crate::stats::pearson(&obs, &pred)
}

/// Co-elution acceptance score (temporal): predicted-intensity-weighted mean
/// Pearson correlation of each matched fragment's XIC to the signature-ion
/// reference profile, over the elution peak. High when the peptide's own fragments
/// co-elute; low when a matched fragment only coincides at the apex. Orthogonal to
/// the intensity-agreement of `peak_spectral_score`.
fn coelution_gate_score(groups: &ScanGroups, distinct: &[u16], sig: &[u16], fints0: &[f32]) -> f64 {
    let (lo, hi, refp) = match peak_window(groups, sig) {
        Some(w) => w,
        None => return 1.0,
    };
    let refw = &refp[lo..=hi];
    let (mut wsum, mut wtot) = (0.0f64, 0.0f64);
    for &f in distinct {
        let tr: Vec<f64> = (lo..=hi).map(|i| groups.or_zero(i, f) as f64).collect();
        if tr.iter().any(|x| *x > 0.0) {
            let c = crate::stats::pearson(&tr, refw).max(0.0);
            let w = *fints0.get(f as usize).unwrap_or(&0.0) as f64 + 1e-9;
            wsum += c * w;
            wtot += w;
        }
    }
    if wtot > 0.0 {
        wsum / wtot
    } else {
        1.0
    }
}

/// fragindex non-two-pass accumulation over isolation-window groups, in parallel.
/// Each scan belongs to exactly one window, so the groups are independent; the
/// per-candidate hit lists are merged by concatenating in (window-sorted) group
/// order. Bit-identical to serial accumulation: the per-candidate cascade rt-sorts
/// hits before the apex sum, and same-rt hits for a candidate all come from one
/// window, so the concatenation order does not affect the rt-sorted result. Only
/// the PeakClaim::None / Winner / Proportional (non-two-pass) strategies use this;
/// the co-elution two-pass path stays serial.
/// Per-peak mass-offset correction applied to an observed peak m/z before library
/// matching. Either a single scalar ppm offset (`grid_*` empty, the default) or an
/// m/z-dependent grid (sorted ascending) that is linearly interpolated and clamped
/// at the ends. `factor_at(mz)` returns the divisor `1 + ppm(mz) * 1e-6`.
struct MassOffset {
    scalar_ppm: f64,
    grid_mz: Vec<f64>,
    grid_ppm: Vec<f64>,
}
impl MassOffset {
    #[inline]
    fn factor_at(&self, mz: f64) -> f64 {
        let ppm = if self.grid_mz.len() >= 2 {
            match self.grid_mz.binary_search_by(|g| g.total_cmp(&mz)) {
                Ok(i) => self.grid_ppm[i],
                Err(0) => self.grid_ppm[0],
                Err(i) if i >= self.grid_mz.len() => self.grid_ppm[self.grid_mz.len() - 1],
                Err(i) => {
                    let (x0, x1) = (self.grid_mz[i - 1], self.grid_mz[i]);
                    let (y0, y1) = (self.grid_ppm[i - 1], self.grid_ppm[i]);
                    y0 + (y1 - y0) * (mz - x0) / (x1 - x0)
                }
            }
        } else {
            self.scalar_ppm
        };
        1.0 + ppm * 1e-6
    }
}

/// One isolation window and the scans acquired in it, with the candidate range that
/// window can match. Ascending by window m/z, which (precursors being sorted by m/z)
/// makes the candidate ranges ascending too: that is what lets the driver decide a
/// candidate is final and flush it (see `GROUPS_IN_FLIGHT`).
struct WinGroup {
    lo_cid: u32,
    hi_cid: u32,
    scans: Vec<usize>,
}

/// Group the run's scans by isolation window, ascending.
fn window_groups(lib: &Library, scans: &[Ms2Scan]) -> Vec<WinGroup> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<(u64, u64), Vec<usize>> = BTreeMap::new();
    for (si, scan) in scans.iter().enumerate() {
        groups
            .entry((
                scan.window.lower_mz.to_bits(),
                scan.window.upper_mz.to_bits(),
            ))
            .or_default()
            .push(si);
    }
    groups
        .into_values()
        .filter_map(|ids| {
            let w = &scans[*ids.first()?].window;
            let (lo, hi) = lib.candidate_range(w.lower_mz, w.upper_mz);
            (hi > lo).then_some(WinGroup {
                lo_cid: lo,
                hi_cid: hi,
                scans: ids,
            })
        })
        .collect()
}

/// One probing task of the streamed accumulation: the scans of one window against the
/// candidates of one sub-range `[lo, hi)`.
#[derive(Clone, Copy)]
struct ProbeTask<'a> {
    ids: &'a [usize],
    scans: &'a [Ms2Scan],
    rt_lo: &'a [f64],
    rt_hi: &'a [f64],
    mass_off: &'a MassOffset,
    cfg: &'a ExtractConfig,
    restrict: Option<&'a CandMask>,
    /// The whole-library bin geometry the index `run` is handed was built with.
    bins: &'a LogBins,
    lo: u32,
    hi: u32,
}

impl ProbeTask<'_> {
    /// Probe every peak of the task's scans through `ix` and return the task's hits grouped
    /// by candidate. Generic over the index so the production [`LocalIndex`] and the
    /// reference [`NarrowedProbe`] run the same code, monomorphised, with nothing but the
    /// index between them.
    fn run<I: BinnedProbe>(&self, ix: &mut I) -> HitStore {
        let ProbeTask {
            ids,
            scans,
            rt_lo,
            rt_hi,
            mass_off,
            cfg,
            restrict,
            bins,
            lo,
            hi,
        } = *self;
        // Flat `(cid, hit)` pairs in probe order, grouped by candidate at the end of
        // the task with a stable counting sort. The task-local `HashMap<u32,
        // Vec<Hit>>` this replaces was one of the two populations of medium heap
        // blocks that exhausted the mapping table.
        let mut flat_cid: Vec<u32> = Vec::new();
        let mut flat_hit: Vec<Hit> = Vec::new();
        let mut claimants: Vec<(u32, u16, f32)> = Vec::new();
        // One scan's `(q_mz, bin)`, refilled per scan and reused. A few KB at 300-2000
        // peaks a scan, task-local, so it is not the per-window buffer that the comment in
        // `accumulate_groups` records as tried and reverted: nothing
        // is shared, nothing is filled before the pool starts, and no thread waits.
        //
        // It still pays, with exactly the same number of `ln()` calls the probe made
        // per peak, because a separate pass takes the `ln()` off the dependency chain
        // that the bin-cache load and then the posting loads hang from. Measured over
        // rayon at this task shape (`tests/bench_fragindex.rs` `bench_wall`, 32
        // threads, min of 9, two independent passes): -11.7 / -6.4% on the default
        // shape, -15.1 / -17.9% with smaller windows, -17.4 / -18.6% with one wide
        // window in the batch. Single thread, sorted production-shaped peaks
        // (`bench_probe`): -9.1 / -12.7 / -6.7% for a narrow / medium / wide
        // candidate window.
        //
        // It carries `q_mz` and not just the bin so `factor_at` still runs ONCE per
        // peak. That is free here and is not free for the caller: under
        // `search_seed.mass_cal_loess` the offset is a ~74-point grid and `factor_at`
        // is a binary search plus an interpolation, measured 8.3 ns/peak against
        // 0.725 for the scalar default, and a bin-only scratch (which recomputes
        // `q_mz` in the peak loop) turned that arm from -1.7% into +5.6%.
        let mut setup: Vec<(f64, u32)> = Vec::new();
        for &si in ids {
            let scan = &scans[si];
            let rt = scan.rt_seconds;
            let hs = si as u32;
            setup.clear();
            setup.extend(scan.peaks.iter().map(|p| {
                let mz = p.mz as f64;
                let q = mz / mass_off.factor_at(mz);
                (q, bins.bin(q) as u32)
            }));
            for (peak, &(q_mz, bin)) in scan.peaks.iter().zip(&setup) {
                let inten = peak.intensity;
                let obs_mz = peak.mz;
                claimants.clear();
                ix.probe_binned(q_mz, bin, |cid, _pmz, pint, frag| {
                    let c = cid as usize;
                    if rt < rt_lo[c] || rt > rt_hi[c] {
                        return;
                    }
                    // The allowlist is applied here, before the claim, exactly where the
                    // serial path applies it: a candidate outside the list neither
                    // collects hits nor competes for a shared peak.
                    if let Some(s) = restrict {
                        if !s.contains(cid) {
                            return;
                        }
                    }
                    claimants.push((cid, frag, pint));
                });
                if claimants.is_empty() {
                    continue;
                }
                match cfg.peak_claim {
                    PeakClaim::WinnerPredictedIntensity => {
                        let mut best = 0usize;
                        for i in 1..claimants.len() {
                            let a = claimants[i];
                            let b = claimants[best];
                            if a.2 > b.2 || (a.2 == b.2 && a.0 < b.0) {
                                best = i;
                            }
                        }
                        let (cid, frag, _) = claimants[best];
                        flat_cid.push(cid);
                        flat_hit.push(Hit {
                            scan: hs,
                            frag,
                            inten,
                            obs_mz,
                        });
                    }
                    PeakClaim::Proportional => {
                        let sump: f32 = claimants.iter().map(|c| c.2.max(0.0)).sum();
                        for &(cid, frag, pi) in &claimants {
                            let share = if sump > 0.0 {
                                inten * (pi.max(0.0) / sump)
                            } else {
                                inten / claimants.len() as f32
                            };
                            flat_cid.push(cid);
                            flat_hit.push(Hit {
                                scan: hs,
                                frag,
                                inten: share,
                                obs_mz,
                            });
                        }
                    }
                    _ => {
                        for &(cid, frag, _) in &claimants {
                            flat_cid.push(cid);
                            flat_hit.push(Hit {
                                scan: hs,
                                frag,
                                inten,
                                obs_mz,
                            });
                        }
                    }
                }
            }
        }
        let (cids, offs) = group_hits_by_candidate(lo, hi, &mut flat_cid, &mut flat_hit);
        drop(flat_cid);
        // Not shrunk: the buffer's doubling slack is the same slack the per-candidate
        // vectors carried, and `shrink_to_fit` here measured 2,520-2,560 MB of peak
        // RSS against 2,456-2,505 MB without it on the AIF fixture, i.e. its copy
        // costs more than the slack it returns.
        HitStore {
            cids,
            offs,
            hits: flat_hit,
        }
    }
}

/// How the probing tasks of the streamed accumulation reach the fragment index.
#[derive(Clone, Copy)]
enum TaskProbe<'a> {
    /// Each task builds a [`LocalIndex`] over its own candidate sub-range, binned with the
    /// whole library's geometry. The default, and the only production path.
    Local {
        lib: &'a Library,
        bins: &'a LogBins,
        tol_ppm: f64,
        stats: &'a LocalIndexStats,
    },
    /// Each task probes the global [`FragIndex`] through a [`WindowNarrow`]: the path this
    /// replaced, kept as the reference the local index is compared against.
    #[cfg(test)]
    Global(&'a FragIndex),
}

impl TaskProbe<'_> {
    fn bins(&self) -> &LogBins {
        match self {
            TaskProbe::Local { bins, .. } => bins,
            #[cfg(test)]
            TaskProbe::Global(idx) => idx.bins(),
        }
    }
}

/// What the task-local indexes of one extract cost, summed over every task: logged once
/// at the end of the accumulation.
#[derive(Default)]
struct LocalIndexStats {
    tasks: std::sync::atomic::AtomicU64,
    postings: std::sync::atomic::AtomicU64,
    largest_bytes: std::sync::atomic::AtomicUsize,
}

impl LocalIndexStats {
    fn record(&self, ix: &LocalIndex) {
        use std::sync::atomic::Ordering::Relaxed;
        self.tasks.fetch_add(1, Relaxed);
        self.postings.fetch_add(ix.len() as u64, Relaxed);
        self.largest_bytes.fetch_max(ix.heap_bytes(), Relaxed);
    }
}

/// Probe one batch of isolation windows and flush what each candidate sub-range finalises,
/// in ascending candidate order.
///
/// Two axes are cut. Across WINDOWS, because each scan belongs to exactly one of them. And
/// across a shared, GLOBAL grid of candidate sub-ranges, because a batch of `step` windows
/// on `t` threads leaves `t - step` threads idle whenever the batch is smaller than the
/// pool (measured at 4 in flight on 24 threads: 3.2-3.7 cores busy for the whole stage),
/// and because a candidate is final once every window of the batch has passed it, which is
/// a statement about a boundary all the windows share. Per-window sub-ranges would
/// parallelise just as well and could not be flushed.
///
/// That matters most exactly where the stage hurts. `window_groups` drops the windows that
/// select no candidate of the band, so a band of a 63-band grouped plan overlaps only 1-3
/// windows, the whole band is one batch, the flush bound is `u32::MAX` and nothing was ever
/// flushed early: the accumulator held the entire band. Cutting on the candidate axis makes
/// the flush unit a sub-range of it instead, at no extra probing, because the sub-ranges
/// are the tasks that already existed.
///
/// Order is the contract. A candidate is flushed only once every window that can reach it
/// has reported, its hits are gathered run by run in window order, and sub-ranges are
/// flushed in ascending index, so the per-candidate hit sequence and the ascending PSM row
/// order are exactly what the serial path produced. Chromatogram parquet row-group
/// boundaries can move, because the flush is now cut on sub-ranges rather than on the
/// batch; no row and no value moves with them.
///
/// Returns true when `flush` asked to stop (the chromatogram writer went away).
#[allow(clippy::too_many_arguments)]
fn accumulate_groups(
    probe: TaskProbe<'_>,
    sibling_bands: usize,
    groups: &[WinGroup],
    scans: &[Ms2Scan],
    rt_lo: &[f64],
    rt_hi: &[f64],
    mass_off: &MassOffset,
    cfg: &ExtractConfig,
    restrict: Option<&CandMask>,
    // First candidate a window of a LATER batch can still add hits to; nothing at or above
    // it may be flushed here.
    bound: u32,
    acc: &mut HitAcc,
    chunk: &mut HitStore,
    flush: &mut FlushFn<'_>,
) -> bool {
    if groups.is_empty() {
        return false;
    }
    // `current_num_threads()` is the whole pool, and under `groups.parallel` every band in
    // flight computes this independently: 24 bands each fanning out to twice the pool gave
    // thousands of live narrowed-bin caches (1-2 MB each) allocated in lockstep, which is
    // what a task's local index is now (4 bytes per occupied bin plus its postings). Divide
    // by the bands beside this one so the fan-out describes this band's share.
    let threads = (rayon::current_num_threads() / sibling_bands.max(1)).max(1);
    let tasks_per_window = (threads * 2).div_ceil(groups.len()).max(1);
    // The sub-range WIDTH is taken from the mean window span divided by
    // `tasks_per_window`, not from the global span divided by a task count, so each window
    // still splits into about `tasks_per_window` pieces and the batch still runs about
    // 2x threads of them -- the same parallelism as before, on a shared grid.
    let g_lo = groups.iter().map(|g| g.lo_cid).min().expect("non-empty");
    let g_hi = groups.iter().map(|g| g.hi_cid).max().expect("non-empty");
    let mean_span = groups
        .iter()
        .map(|g| (g.hi_cid - g.lo_cid) as usize)
        .sum::<usize>()
        / groups.len();
    let width = mean_span
        .div_ceil(tasks_per_window)
        .max(MIN_CANDIDATES_PER_TASK)
        .min(u32::MAX as usize) as u64;
    let n_sub = ((g_hi - g_lo) as u64).div_ceil(width).max(1) as usize;
    let sub_bounds = |k: usize| -> (u32, u32) {
        let hi = g_hi as u64;
        let s = (g_lo as u64 + k as u64 * width).min(hi);
        ((s) as u32, (s + width).min(hi) as u32)
    };
    // One task per (sub-range, window) pair that actually intersects, k-major so the pool
    // starts on the sub-ranges that will be flushed first. `expected[k]` is how many
    // windows must report before sub-range `k` is final.
    let mut tasks: Vec<(usize, usize, u32, u32)> = Vec::new();
    let mut expected: Vec<usize> = vec![0; n_sub];
    for (k, exp) in expected.iter_mut().enumerate() {
        let (s, e) = sub_bounds(k);
        for (gi, g) in groups.iter().enumerate() {
            let (lo, hi) = (g.lo_cid.max(s), g.hi_cid.min(e));
            if hi > lo {
                tasks.push((k, gi, lo, hi));
                *exp += 1;
            }
        }
    }
    // Note on the per-peak setup `(peak.mz / mass_off.factor_at(peak.mz), bin_of(q_mz))`:
    // the tasks of one window each recompute it for every peak of the window, so a window
    // split into `t` tasks computes it `t` times. Hoisting it into a PER-WINDOW buffer,
    // shared by those tasks, has now been tried twice and reverted twice.
    //
    // The m/z half was reverted first: 8 B/peak, +200 MB of peak RSS on the AIF fixture
    // (317 MB with the whole run in one batch), no measurable time.
    //
    // The `ln()` bin half was reverted in the same place for a different reason. The
    // buffer has to be filled before the pool starts, so it converts per-task work that
    // was SPREAD ACROSS THE POOL into serial work on one thread, and the default shape
    // does not have enough tasks per window to pay that back: `tasks_per_window` is
    // `(2 * threads).div_ceil(groups.len())`, which is 4 at 32 threads and the default
    // 16 windows in flight, so the fill costs one pass per peak to save three thirty-
    // seconds of one; break-even needs `tasks_per_window >= threads`, i.e. a batch of one
    // or two windows. Measured on `tests/bench_fragindex.rs` `bench_wall`, which runs this
    // task shape over rayon (32 threads, min of 9, two independent passes): against the
    // in-probe baseline the serial fill is +8.5 / +22.1% on the default shape and
    // +38.9 / +30.0% with smaller windows (1-2 tasks each), where the scratch below is
    // -11.7 / -6.4% and -15.1 / -17.9%. Only a batch of one wide window pays the fill
    // back (-23.7 / -19.0%, against the scratch's -17.4 / -18.6%). A fill spread over the
    // POOL removes the regression and is still not worth it: against the scratch it is
    // -5.6 / -2.3% on the default shape and +5.7 / +3.2% -- a loss -- on the second, for
    // 4 B/peak of every window in flight, about half of what got the m/z hoist reverted,
    // in extract, the tallest stage.
    //
    // End to end, `mumdia extract` on the AIF run (152 windows, 1.69M candidates,
    // 465,806 scans, 32 threads, three binaries interleaved over ~24 reps each, timing
    // the accumulation phase alone): against the in-probe baseline's 3,274 ms min /
    // 3,360 mean-of-3-fastest / 4,233 median, the serial-fill buffer is +14.0 / +13.2 /
    // +9.3% and the scratch below -0.3 / -1.9 / -4.5%. The regression is the size the
    // model predicts: one serial pass over the in-flight peaks, 39.6M x 5.4 ns = 430 ms
    // of a 3.3 s phase. Byte-identical artifacts in all 69 runs. See docs/09_extract.md.

    //
    // The memory-cheap version of that hoist -- writing the corrected value back into
    // `scan.peaks` instead of into a side buffer -- is now FORBIDDEN, not merely
    // unmeasured. Under `groups.window_groups > 1` the scans are one buffer lent to every
    // band (`ExtractParams::scans`), and under `groups.calibration = per_group` each band
    // applies its own `MassOffset` to it, so band g00's factor would be baked into the
    // peaks band g01 reads. The borrow checker stops it today, because `scans` arrives as
    // a shared slice; do not reach for `&mut` to get around that.
    let (tx, rx) = std::sync::mpsc::channel::<(usize, usize, HitStore)>();
    {
        let probe_range = |gi: usize, lo: u32, hi: u32| -> HitStore {
            let task = ProbeTask {
                ids: &groups[gi].scans,
                scans,
                rt_lo,
                rt_hi,
                mass_off,
                cfg,
                restrict,
                bins: probe.bins(),
                lo,
                hi,
            };
            match probe {
                // The task indexes its own sub-range: the postings the narrowed global
                // index would have handed it, in the same order, and nothing else.
                TaskProbe::Local {
                    lib,
                    bins,
                    tol_ppm,
                    stats,
                } => {
                    let mut ix = LocalIndex::build(lib, bins, tol_ppm, lo, hi);
                    stats.record(&ix);
                    task.run(&mut ix)
                }
                #[cfg(test)]
                TaskProbe::Global(idx) => task.run(&mut NarrowedProbe {
                    idx,
                    nw: idx.window_narrow(lo, hi),
                }),
            }
        };
        let n = tasks.len();
        // `move`: the receiver is Send but not Sync, so it has to be owned by the scope
        // closure (the consumer) rather than borrowed into it.
        rayon::scope(move |sc| {
            // The workers run under rayon; this thread is the single consumer. `spawn` so
            // the scope does not block the consumer loop below until every task is done.
            sc.spawn(move |_| {
                tasks.par_iter().for_each_with(tx, |tx, &(k, gi, lo, hi)| {
                    // A closed receiver only happens if the consumer panicked.
                    let _ = tx.send((k, gi, probe_range(gi, lo, hi)));
                });
            });
            // Stores wait here, keyed by `(sub-range, window)`, until their whole
            // sub-range has reported. `next_k` is the lowest sub-range not yet flushed, so
            // the map's first keys are always that sub-range's.
            let mut pending: std::collections::BTreeMap<(usize, usize), HitStore> =
                std::collections::BTreeMap::new();
            let mut arrived: Vec<usize> = vec![0; n_sub];
            let mut next_k = 0usize;
            let mut stopped = false;
            for _ in 0..n {
                let (k, gi, store) = recv_participating(&rx);
                arrived[k] += 1;
                pending.insert((k, gi), store);
                while next_k < n_sub && arrived[next_k] == expected[next_k] {
                    let mut runs = std::mem::take(&mut acc.runs);
                    for _ in 0..expected[next_k] {
                        let key = *pending.keys().next().expect("the sub-range is complete");
                        debug_assert_eq!(key.0, next_k);
                        let store = pending.remove(&key).expect("key came from the map");
                        // One run per window, appended in window order, after whatever
                        // earlier windows already left in the accumulator.
                        runs.push(HitRun::new(vec![store]));
                    }
                    // Everything this sub-range owns is final, except what a later batch
                    // can still reach.
                    let sub_bound = sub_bounds(next_k).1.min(bound);
                    if !stopped && flush_below(&mut runs, sub_bound, CAND_CHUNK, true, chunk, flush)
                    {
                        stopped = true;
                    }
                    acc.runs = runs;
                    acc.compact();
                    next_k += 1;
                }
            }
            debug_assert!(pending.is_empty() && next_k == n_sub);
            stopped
        })
    }
}

/// Isolation windows probed before the driver flushes the candidates that are now final.
/// It is neither the unit of parallelism nor, since the sub-range flush, the unit of
/// memory: each window is probed in parallel over a shared grid of candidate sub-ranges,
/// and `accumulate_groups` flushes a sub-range as soon as every window of the batch has
/// passed it, so the accumulator holds a sub-range of the batch rather than the batch.
/// The batch still bounds how much of the precursor axis is open at once, because a
/// candidate is not final until the last window that can reach it has run. Measured on the
/// HYE benchmark at 32 threads before either change (docs/27 section 3.10): 32 in flight
/// 24.65 GiB / 5:00, 16 in flight 16.57 GiB / 5:04, 8 in flight 12.31 GiB / 5:26,
/// identical output throughout.
const DEFAULT_MAX_WINDOWS_IN_FLIGHT: usize = 16;

/// Fewest candidates a probing task takes: below this the per-task narrowed bin cache (one
/// entry per fragment bin) costs more than the posting-list work it divides.
const MIN_CANDIDATES_PER_TASK: usize = 4096;

fn groups_in_flight(cfg: &ExtractConfig) -> usize {
    cfg.windows_in_flight
        .filter(|&n| n > 0)
        .unwrap_or_else(|| rayon::current_num_threads().min(DEFAULT_MAX_WINDOWS_IN_FLIGHT))
        .max(1)
}

/// Parallel two-pass co-elution peak-claim. Each isolation-window group is
/// processed independently (a candidate's precursor m/z places it in one window,
/// and a peak's claimants come only from that window via `candidate_range`), so
/// both the base accumulation (pass 1, for elution profiles) and the arbitration
/// (pass 2) fan out across the ~150 windows. Returns the (possibly reassigned)
/// accumulation and per-candidate (won, lost) contested intensity. Mirrors the
/// serial two-pass exactly, window-partitioned; merge is disjoint across windows
/// (extend/sum is overlap-safe if windows ever overlap in m/z).
#[allow(clippy::too_many_arguments)]
fn extract_twopass_windows(
    idx: Option<&FragIndex>,
    lib: &Library,
    scans: &[Ms2Scan],
    rt_lo: &[f64],
    rt_hi: &[f64],
    rt_cal: &[f64],
    ms1_scans: &[Ms1Scan],
    ms1_rts: &[f64],
    mass_off: &MassOffset,
    frag_tol: f64,
    cfg: &ExtractConfig,
    restrict: Option<&CandMask>,
    reassign: bool,
    claim_margin: f32,
) -> (HashMap<u32, Vec<Hit>>, HashMap<u32, Contested>) {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<(u64, u64), Vec<usize>> = BTreeMap::new();
    for (si, scan) in scans.iter().enumerate() {
        groups
            .entry((
                scan.window.lower_mz.to_bits(),
                scan.window.upper_mz.to_bits(),
            ))
            .or_default()
            .push(si);
    }
    let group_vec: Vec<Vec<usize>> = groups.into_values().collect();
    let pr = Prober {
        fidx: idx,
        lib,
        frag_tol,
    };

    type Part = (Vec<(u32, Vec<Hit>)>, Vec<(u32, Contested)>);
    let partials: Vec<Part> = group_vec
        .par_iter()
        .map(|ids| {
            if ids.is_empty() {
                return (Vec::new(), Vec::new());
            }
            let w = &scans[ids[0]].window;
            let (lo, hi) = lib.candidate_range(w.lower_mz, w.upper_mz);
            if hi <= lo {
                return (Vec::new(), Vec::new());
            }
            let mut claimants: Vec<(u32, u16, f32)> = Vec::new();
            // `(lo, hi)` is fixed for this whole isolation window and both passes
            // reprobe the same bins across every scan of it, so narrow each bin once.
            let mut nw = idx.map(|i| i.window_narrow(lo, hi));
            // PASS 1: base accumulation (full peak intensity) for elution profiles.
            let mut acc1: HashMap<u32, Vec<Hit>> = HashMap::new();
            for &si in ids {
                let scan = &scans[si];
                let rt = scan.rt_seconds;
                let hs = si as u32;
                for peak in &scan.peaks {
                    let inten = peak.intensity;
                    let obs_mz = peak.mz;
                    let q_mz = obs_mz as f64 / mass_off.factor_at(obs_mz as f64);
                    claimants.clear();
                    {
                        let mut push = |cid: u32, frag: u16, pi: f32| {
                            let c = cid as usize;
                            if rt < rt_lo[c] || rt > rt_hi[c] {
                                return;
                            }
                            if let Some(s) = restrict {
                                if !s.contains(cid) {
                                    return;
                                }
                            }
                            claimants.push((cid, frag, pi));
                        };
                        pr.probe(nw.as_mut(), q_mz, lo, hi, &mut push);
                    }
                    for &(cid, frag, _) in &claimants {
                        acc1.entry(cid).or_default().push(Hit {
                            scan: hs,
                            frag,
                            inten,
                            obs_mz,
                        });
                    }
                }
            }
            let mut profile: HashMap<u32, HashMap<u64, f32>> = HashMap::new();
            for (cid, hits) in &acc1 {
                let m = profile.entry(*cid).or_default();
                for h in hits {
                    *m.entry(hit_rt(scans, h).to_bits()).or_insert(0.0) += h.inten;
                }
            }
            // S2 uniqueness-seeded EM: re-seed each candidate's elution profile from its
            // cue-weighted APPORTIONED intensity (not the full peak) for a fixed number of
            // iterations, so a borrowing candidate's profile is no longer inflated by the
            // peaks it borrows. A single-claimant (uncontested) peak contributes its full
            // intensity every iteration -> an immovable anchor. Deterministic (fixed N,
            // ordered f32 reductions). Only under CoelutionMultiCue; 0 iters = no-op.
            let em_iters = if matches!(cfg.peak_claim, PeakClaim::CoelutionMultiCue) {
                cfg.claim_cues.apportion_em_iters
            } else {
                0
            };
            for _ in 0..em_iters {
                let mut next: HashMap<u32, HashMap<u64, f32>> = HashMap::new();
                for &si in ids {
                    let scan = &scans[si];
                    let rt = scan.rt_seconds;
                    let rtb = rt.to_bits();
                    for peak in &scan.peaks {
                        let inten = peak.intensity;
                        let obs_mz = peak.mz as f64;
                        let q_mz = obs_mz / mass_off.factor_at(obs_mz);
                        claimants.clear();
                        {
                            let mut push = |cid: u32, frag: u16, pi: f32| {
                                let c = cid as usize;
                                if rt < rt_lo[c] || rt > rt_hi[c] {
                                    return;
                                }
                                if let Some(s) = restrict {
                                    if !s.contains(cid) {
                                        return;
                                    }
                                }
                                claimants.push((cid, frag, pi));
                            };
                            pr.probe(nw.as_mut(), q_mz, lo, hi, &mut push);
                        }
                        if claimants.is_empty() {
                            continue;
                        }
                        let weights: Vec<f32> = claimants
                            .iter()
                            .map(|&(cid, frag, _)| {
                                let h = profile
                                    .get(&cid)
                                    .and_then(|m| m.get(&rtb))
                                    .copied()
                                    .unwrap_or(0.0);
                                if h > 0.0 {
                                    h * claim_cue_multiplier(
                                        cfg, lib, cid, frag, obs_mz, rt, rt_cal, ms1_scans, ms1_rts,
                                    )
                                } else {
                                    h
                                }
                            })
                            .collect();
                        let sum_w: f32 = weights.iter().copied().sum();
                        for (i, &(cid, _, _)) in claimants.iter().enumerate() {
                            let share = if sum_w > 0.0 {
                                inten * (weights[i] / sum_w)
                            } else {
                                inten / claimants.len() as f32
                            };
                            *next.entry(cid).or_default().entry(rtb).or_insert(0.0) += share;
                        }
                    }
                }
                profile = next;
            }
            // PASS 2: arbitrate each shared peak by which claimant is most eluting.
            let mut acc2: HashMap<u32, Vec<Hit>> = HashMap::new();
            let mut contested: HashMap<u32, Contested> = HashMap::new();
            let demix_mode = matches!(cfg.peak_claim, PeakClaim::CoelutionDemix);
            let shadow_mode = matches!(cfg.peak_claim, PeakClaim::CoelutionShadow);
            let demix_stride = cfg.demix_scan_stride.max(1);
            let mut demix_abund: BTreeMap<u32, f64> = BTreeMap::new();
            let mut demix_ctr = 0usize;
            for &si in ids {
                let scan = &scans[si];
                let rt = scan.rt_seconds;
                let rtb = rt.to_bits();
                let hs = si as u32;
                // Spectrum-centric demix redistribution (CoelutionDemix): solve one NNLS
                // over this scan's co-isolated candidate x fragment matrix and split each
                // shared peak by beta_c * D[peak,c] (smooth joint deconvolution) rather than
                // the per-peak profile-height arbitration below.
                if demix_mode {
                    let mut cand: std::collections::BTreeSet<u32> =
                        std::collections::BTreeSet::new();
                    let mut prows: Vec<DemixRow> = Vec::new();
                    for peak in &scan.peaks {
                        let obs_mz = peak.mz as f64;
                        let q_mz = obs_mz / mass_off.factor_at(obs_mz);
                        claimants.clear();
                        {
                            let mut push = |cid: u32, frag: u16, pi: f32| {
                                let c = cid as usize;
                                if rt < rt_lo[c] || rt > rt_hi[c] {
                                    return;
                                }
                                if let Some(s) = restrict {
                                    if !s.contains(cid) {
                                        return;
                                    }
                                }
                                claimants.push((cid, frag, pi));
                            };
                            pr.probe(nw.as_mut(), q_mz, lo, hi, &mut push);
                        }
                        if claimants.is_empty() {
                            continue;
                        }
                        for &(cid, _, _) in &claimants {
                            cand.insert(cid);
                        }
                        prows.push((peak.intensity, peak.mz, claimants.clone()));
                    }
                    if prows.is_empty() {
                        continue;
                    }
                    // Solve the NNLS every `demix_stride` scans or whenever a new candidate
                    // enters the co-isolated set; reuse the stored abundances otherwise. The
                    // abundances change slowly across a few scans (elution is gradual), while
                    // each scan's own peaks + predicted intensities still drive the split.
                    let new_cand = cand.iter().any(|c| !demix_abund.contains_key(c));
                    if demix_abund.is_empty() || demix_ctr.is_multiple_of(demix_stride) || new_cand
                    {
                        let cols: Vec<u32> = cand.iter().copied().collect();
                        let n = cols.len();
                        let col_of: BTreeMap<u32, usize> =
                            cols.iter().enumerate().map(|(i, &c)| (c, i)).collect();
                        let m = prows.len();
                        let mut amat = vec![0.0f64; m * n];
                        let mut yv = vec![0.0f64; m];
                        for (r, (obs, _, cl)) in prows.iter().enumerate() {
                            yv[r] = *obs as f64;
                            for &(cid, _, pi) in cl {
                                let cell = &mut amat[r * n + col_of[&cid]];
                                if (pi as f64) > *cell {
                                    *cell = pi as f64;
                                }
                            }
                        }
                        let beta = crate::solve::nnls(
                            &amat,
                            m,
                            n,
                            &yv,
                            cfg.demix_lambda.max(1e-9),
                            200 * n.max(1),
                        );
                        demix_abund.clear();
                        for (&c, &col) in &col_of {
                            demix_abund.insert(c, beta[col]);
                        }
                    }
                    demix_ctr += 1;
                    // Apportion each peak by abundance_c * predicted_c (the joint split).
                    for (obs, obs_mz, cl) in &prows {
                        let mut denom = 0.0f64;
                        let mut winner = cl[0].0;
                        let mut best_bd = f64::NEG_INFINITY;
                        for &(cid, _, pi) in cl {
                            let bd = demix_abund.get(&cid).copied().unwrap_or(0.0) * pi as f64;
                            denom += bd;
                            if bd > best_bd || (bd == best_bd && cid < winner) {
                                best_bd = bd;
                                winner = cid;
                            }
                        }
                        for &(cid, frag, pi) in cl {
                            let bd = demix_abund.get(&cid).copied().unwrap_or(0.0) * pi as f64;
                            let share = if denom > 0.0 {
                                *obs as f64 * (bd / denom)
                            } else {
                                *obs as f64 / cl.len() as f64
                            };
                            let e = contested.entry(cid).or_default();
                            e.apportioned += share;
                            if cid == winner {
                                e.won += *obs as f64;
                                e.n_won += 1;
                            } else {
                                e.lost += *obs as f64;
                                e.n_lost += 1;
                            }
                            acc2.entry(cid).or_default().push(Hit {
                                scan: hs,
                                frag,
                                inten: share as f32,
                                obs_mz: *obs_mz,
                            });
                        }
                    }
                    continue;
                }
                if shadow_mode {
                    // Shadow subtraction: estimate each candidate's abundance from its UNIQUE
                    // channels, then clean every candidate's channels by subtracting the other
                    // claimants' estimated contributions. No solve; several real co-eluters can
                    // both keep signal at a shared peak.
                    let mut prows: Vec<DemixRow> = Vec::new();
                    for peak in &scan.peaks {
                        let obs_mz = peak.mz as f64;
                        let q_mz = obs_mz / mass_off.factor_at(obs_mz);
                        claimants.clear();
                        {
                            let mut push = |cid: u32, frag: u16, pi: f32| {
                                let c = cid as usize;
                                if rt < rt_lo[c] || rt > rt_hi[c] {
                                    return;
                                }
                                if let Some(s) = restrict {
                                    if !s.contains(cid) {
                                        return;
                                    }
                                }
                                claimants.push((cid, frag, pi));
                            };
                            pr.probe(nw.as_mut(), q_mz, lo, hi, &mut push);
                        }
                        if claimants.is_empty() {
                            continue;
                        }
                        prows.push((peak.intensity, peak.mz, claimants.clone()));
                    }
                    if prows.is_empty() {
                        continue;
                    }
                    // Per-candidate abundance from unique (single-claimant) channels, median of
                    // observed/predicted. Deterministic (sorted cid, median of sorted ratios).
                    let mut uniq: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
                    for (obs, _, cl) in &prows {
                        if cl.len() == 1 {
                            let (cid, _, pi) = cl[0];
                            if pi > 0.0 {
                                uniq.entry(cid).or_default().push(*obs as f64 / pi as f64);
                            }
                        }
                    }
                    let mut a_p: BTreeMap<u32, f64> = BTreeMap::new();
                    for (cid, mut v) in uniq {
                        v.sort_by(|x, z| x.total_cmp(z));
                        a_p.insert(cid, v[v.len() / 2]);
                    }
                    for (obs, obs_mz, cl) in &prows {
                        for &(cid, frag, _) in cl {
                            let mut sub = 0.0f64;
                            for &(pj, _, pij) in cl {
                                if pj != cid {
                                    sub += a_p.get(&pj).copied().unwrap_or(0.0) * pij as f64;
                                }
                            }
                            let cleaned = (*obs as f64 - sub).max(0.0);
                            let e = contested.entry(cid).or_default();
                            e.apportioned += cleaned;
                            if cleaned >= 0.5 * *obs as f64 {
                                e.won += *obs as f64;
                                e.n_won += 1;
                            } else {
                                e.lost += *obs as f64;
                                e.n_lost += 1;
                            }
                            acc2.entry(cid).or_default().push(Hit {
                                scan: hs,
                                frag,
                                inten: cleaned as f32,
                                obs_mz: *obs_mz,
                            });
                        }
                    }
                    continue;
                }
                for peak in &scan.peaks {
                    let inten = peak.intensity;
                    let obs_mz = peak.mz as f64;
                    let hit_mz = peak.mz;
                    let q_mz = obs_mz / mass_off.factor_at(obs_mz);
                    claimants.clear();
                    {
                        let mut push = |cid: u32, frag: u16, pi: f32| {
                            let c = cid as usize;
                            if rt < rt_lo[c] || rt > rt_hi[c] {
                                return;
                            }
                            if let Some(s) = restrict {
                                if !s.contains(cid) {
                                    return;
                                }
                            }
                            claimants.push((cid, frag, pi));
                        };
                        pr.probe(nw.as_mut(), q_mz, lo, hi, &mut push);
                    }
                    if claimants.is_empty() {
                        continue;
                    }
                    let ph = |cid: u32| -> f32 {
                        profile
                            .get(&cid)
                            .and_then(|m| m.get(&rtb))
                            .copied()
                            .unwrap_or(0.0)
                    };
                    // Modular per-claimant competition weight (fragment-competition
                    // framework): the elution profile height, optionally multiplied by
                    // the composable ClaimCues under CoelutionMultiCue. For every other
                    // method the cue is 1.0, so `weights` reduces EXACTLY to `ph` and the
                    // arbitration below is bit-identical to the profile-height version.
                    let multicue = matches!(cfg.peak_claim, PeakClaim::CoelutionMultiCue);
                    let weights: Vec<f32> = claimants
                        .iter()
                        .map(|&(cid, frag, _pi)| {
                            let h = ph(cid);
                            if multicue && h > 0.0 {
                                h * claim_cue_multiplier(
                                    cfg, lib, cid, frag, obs_mz, rt, rt_cal, ms1_scans, ms1_rts,
                                )
                            } else {
                                h
                            }
                        })
                        .collect();
                    let mut best = 0usize;
                    for i in 1..claimants.len() {
                        let (ci, _, pii) = claimants[i];
                        let (cb, _, pib) = claimants[best];
                        let (wi, wb) = (weights[i], weights[best]);
                        if wi > wb || (wi == wb && (pii > pib || (pii == pib && ci < cb))) {
                            best = i;
                        }
                    }
                    let win = claimants[best].0;
                    let sum_w: f32 = weights.iter().copied().sum();
                    let top_w = weights[best];
                    let second_w = claimants
                        .iter()
                        .enumerate()
                        .filter(|&(_, c)| c.0 != win)
                        .map(|(i, _)| weights[i])
                        .fold(0.0f32, f32::max);
                    let dominant =
                        top_w > 0.0 && (second_w <= 0.0 || top_w >= claim_margin * second_w);
                    for (i, &(cid, frag, _pi)) in claimants.iter().enumerate() {
                        let e = contested.entry(cid).or_default();
                        // Weight-proportional share (retained intensity under a
                        // proportional split), tracked for every claimant.
                        let share = if sum_w > 0.0 {
                            inten * (weights[i] / sum_w)
                        } else {
                            inten / claimants.len() as f32
                        };
                        e.apportioned += share as f64;
                        if cid == win {
                            e.won += inten as f64;
                            e.n_won += 1;
                        } else {
                            e.lost += inten as f64;
                            e.n_lost += 1;
                        }
                        if reassign {
                            match cfg.peak_claim {
                                // Destructive MultiCue: winner-take-all on the composite
                                // cue weight (only the best-cue-weighted claimant keeps
                                // the peak). CoelutionWinner is the same rule on the plain
                                // profile height.
                                PeakClaim::CoelutionWinner | PeakClaim::CoelutionMultiCue => {
                                    if cid == win {
                                        acc2.entry(cid).or_default().push(Hit {
                                            scan: hs,
                                            frag,
                                            inten,
                                            obs_mz: hit_mz,
                                        });
                                    }
                                }
                                PeakClaim::CoelutionProportional => {
                                    acc2.entry(cid).or_default().push(Hit {
                                        scan: hs,
                                        frag,
                                        inten: share,
                                        obs_mz: hit_mz,
                                    });
                                }
                                PeakClaim::CoelutionWinnerMargin if !dominant || cid == win => {
                                    acc2.entry(cid).or_default().push(Hit {
                                        scan: hs,
                                        frag,
                                        inten,
                                        obs_mz: hit_mz,
                                    });
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            let out_acc = if reassign { acc2 } else { acc1 };
            (
                out_acc.into_iter().collect(),
                contested.into_iter().collect(),
            )
        })
        .collect();

    let mut acc: HashMap<u32, Vec<Hit>> = HashMap::new();
    let mut contested: HashMap<u32, Contested> = HashMap::new();
    for (a, c) in partials {
        for (cid, hits) in a {
            acc.entry(cid).or_default().extend(hits);
        }
        for (cid, s) in c {
            let e = contested.entry(cid).or_default();
            e.won += s.won;
            e.lost += s.lost;
            e.n_won += s.n_won;
            e.n_lost += s.n_lost;
            e.apportioned += s.apportioned;
        }
    }
    (acc, contested)
}

/// The mass calibration extract applies, as read from `ExtractParams::mass_cal`.
struct MassCalRead {
    frag_offset: f64,
    frag_tol: f64,
    grid_mz: Vec<f64>,
    grid_ppm: Vec<f64>,
    /// The file it came from; `None` when no file was passed or it does not exist, and the
    /// values are the configured fallback (no offset, `extract.frag_tol_ppm`).
    from_file: Option<String>,
}

/// Read the per-run mass recalibration: the scalar offset + learned tolerance, plus an
/// optional m/z-dependent correction grid (mass_cal_loess). No logging here; the caller
/// reports it once the spectra are in, where the stage always reported it.
fn read_mass_cal(p: &ExtractParams) -> Result<MassCalRead> {
    let read_grid = |v: &serde_json::Value, key: &str| -> Vec<f64> {
        v.get(key)
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|e| e.as_f64()).collect())
            .unwrap_or_default()
    };
    match p.mass_cal {
        Some(path) if std::path::Path::new(path).exists() => {
            let v: serde_json::Value = mumdia_io::json::read_json(path)?;
            Ok(MassCalRead {
                frag_offset: v
                    .get("frag_ppm_offset")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0),
                frag_tol: v
                    .get("frag_tol_ppm")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(p.cfg.frag_tol_ppm),
                grid_mz: read_grid(&v, "mz_cal_grid_mz"),
                grid_ppm: read_grid(&v, "mz_cal_grid_ppm"),
                from_file: Some(path.to_string()),
            })
        }
        _ => Ok(MassCalRead {
            frag_offset: 0.0,
            frag_tol: p.cfg.frag_tol_ppm,
            grid_mz: Vec::new(),
            grid_ppm: Vec::new(),
            from_file: None,
        }),
    }
}

/// Decode the spectra the caller did not lend: `(ms2, ms1)`, each `None` when the lent
/// buffer is used instead.
///
/// An empty lent slice is never believed over a named path. A caller that lends
/// `SharedScans { ms2, ms1: Some(&[]) }` while still passing `ms1: Some(path)` would
/// otherwise lose every MS1 feature and every MS1 chromatogram row with no error and no
/// warning, because both are guarded on `!ms1_scans.is_empty()` and simply write nothing.
/// Decoding the named artifact instead costs nothing when the run really has no MS1 rows
/// (the decode is then empty too) and makes the silent version impossible. A lent MS1 of
/// `None` is the caller saying it lends the MS2 only; the MS1 is then decoded without a
/// warning.
#[allow(clippy::type_complexity)]
fn decode_unlent(p: &ExtractParams) -> Result<(Option<Vec<Ms2Scan>>, Option<Vec<Ms1Scan>>)> {
    match p.scans {
        Some(shared) => {
            let ms2 = if shared.ms2.is_empty() {
                let owned = load_ms2(p.ms2)?;
                if !owned.is_empty() {
                    warn!(
                        ms2 = p.ms2,
                        scans = owned.len(),
                        "extract: the caller lent an empty MS2 buffer for a run that has \
                         scans; decoding the artifact instead of searching nothing"
                    );
                }
                Some(owned)
            } else {
                None
            };
            let ms1 = match (p.ms1, shared.ms1) {
                (Some(path), None) => Some(load_ms1(path)?),
                (Some(path), Some([])) => {
                    let owned = load_ms1(path)?;
                    if !owned.is_empty() {
                        warn!(
                            ms1 = path,
                            scans = owned.len(),
                            "extract: the caller lent an empty MS1 buffer while naming an \
                             MS1 artifact; decoding it instead of dropping every MS1 feature"
                        );
                    }
                    Some(owned)
                }
                _ => None,
            };
            Ok((ms2, ms1))
        }
        None => {
            // The two artifacts are independent; the MS2 error, if any, is reported first.
            let (ms2, ms1) = rayon::join(
                || load_ms2(p.ms2),
                || match p.ms1 {
                    Some(path) => load_ms1(path),
                    None => Ok(Vec::new()),
                },
            );
            let ms2 = ms2?;
            Ok((Some(ms2), Some(ms1?)))
        }
    }
}

/// Read `run_windows` and scatter it into the dense per-candidate arrays extract uses,
/// `rt_cal[c]`, `rt_lo[c]` and `rt_hi[c]`, for a library of `ncand` candidates.
///
/// The decoded columns (`candidate_id`, `rt_pred_cal`, `rt_lo`, `rt_hi`: 28 bytes per
/// row) are dropped when this returns, rather than living until the end of the stage
/// beside the 24-byte-per-candidate dense copy they were scattered into. On an unbounded
/// 203M-candidate library that is 5.7 GB that used to sit through the whole extraction.
pub(crate) fn read_run_windows(path: &str, ncand: usize) -> Result<RtWindows> {
    let rw = TableFile::open(path)?;
    let rw_cid = rw.u32("candidate_id")?;
    let rw_cal = rw.f64("rt_pred_cal")?;
    let rw_lo = rw.f64("rt_lo")?;
    let rw_hi = rw.f64("rt_hi")?;
    let mut rt_lo = vec![f64::NEG_INFINITY; ncand];
    let mut rt_hi = vec![f64::INFINITY; ncand];
    // NaN, not 0.0, for a candidate with no `run_windows` row. Stage B already uses NaN
    // as the explicit "calibration unavailable" sentinel (`candidate_window`), and
    // `calibrated_rt_error` maps a non-finite value to 0.0, i.e. no RT evidence. A 0.0
    // here is *finite*, so it used to produce `rt_error_abs = apex_rt`, the worst possible
    // value, for exactly the candidates the other path gives the best value to. That made
    // the feature a proxy for "was this candidate in the window table", which is not a
    // property of the spectrum. One sentinel, one meaning.
    let mut rt_cal = vec![f64::NAN; ncand];
    for i in 0..rw.nrows {
        let c = rw_cid[i] as usize;
        if c < ncand {
            // The eight RT-window guards downstream are all `rt < rt_lo || rt > rt_hi`,
            // which is false for NaN, so a NaN bound does not reject the scan, it accepts
            // *every* scan: the candidate is searched across the whole gradient with no RT
            // prior and no warning. The legitimate unbounded case is written as explicit
            // -inf/+inf by `candidate_window`, so a NaN here can only mean a corrupt or
            // externally-written window table. Reject it while the row can be named.
            if rw_lo[i].is_nan() || rw_hi[i].is_nan() {
                anyhow::bail!(
                    "run_windows row {i} (candidate_id {c}) has a NaN RT bound \
                     (rt_lo={}, rt_hi={}); an unbounded window must be written as \
                     -inf/+inf, because a NaN bound silently matches every scan instead \
                     of being rejected. Re-run rt-im-train to regenerate {}",
                    rw_lo[i],
                    rw_hi[i],
                    path
                );
            }
            rt_lo[c] = rw_lo[i];
            rt_hi[c] = rw_hi[i];
            rt_cal[c] = rw_cal[i];
        }
    }
    Ok(RtWindows {
        rt_cal,
        rt_lo,
        rt_hi,
        fitted_for: None,
    })
}

/// One accepted candidate peak of `extract::run`: its `psms_extracted` row, its
/// chromatogram rows and its retained alternative peaks.
struct CandOut {
    cid: u32,
    /// Chromatographic peak rank (0 = selected apex). Top-K promotion (#7).
    peak_rank: u8,
    apex_rt: f64,
    apex_int: f32,
    n_match: i32,
    corun: i32,
    npred: i32,
    calrt: f64,
    mz: f64,
    contested: f64,
    contested_count_frac: f64,
    apportioned_frac: f64,
    z: i32,
    label: String,
    base: u32,
    pform: String,
    prot: String,
    irt: f32,
    ms1_m1: Option<f64>,
    ms1_mono: Option<f64>,
    ms1_i1: Option<f64>,
    ms1_i2: Option<f64>,
    /// Gate diagnostic scores, computed for EVERY accepted candidate regardless
    /// of `gate_mode` (sensitivity program): the single-apex-scan intensity
    /// Pearson, the peak-integrated spectral Pearson, and the temporal co-elution
    /// score. Emitted so an offline analysis can compare gate metrics (and their
    /// combination) at matched pool size, without re-extraction.
    gate_apex: f32,
    gate_peak_spectral: f32,
    gate_coelution: f32,
    gate_spectral_entropy: f32,
    /// Spectrum-centric demix features (D2), all 0 unless `emit_demix_features`:
    /// residual-explained fraction, active-set survival flag, and this candidate's
    /// fraction of the total demixed abundance at its apex.
    deconv_explained: f32,
    deconv_active: f32,
    deconv_share: f32,
    deconv_collin: f32,
    deconv_shadow: f32,
    /// (cid, frag_name, frag_mz, frag_obs_mz, predicted_intensity, rt, intensity)
    chrom: Vec<ChromOutputRow>,
    /// Top-K retained peak groups (sensitivity_plan P1.1/P1.2), populated only
    /// when `retain_top_peaks > 1`. Each: (rank, apex_rt, start_rt, end_rt,
    /// evidence_count, area). Ranked by co-eluting fragment breadth (not
    /// intensity). The main PSM above still reports the single selected apex,
    /// so FDR is unaffected; these are candidate peaks for an offline peak-
    /// selection model. Empty for K=1.
    peaks: Vec<(u8, f64, f64, f64, f64, f64)>,
}

/// The `psms_extracted` rows not yet written, as the columns the table has always had.
///
/// The table used to be held whole, 21 to 32 column vectors for every accepted row, and
/// written once at the end through `write_table`, which cut it into 65,536-row chunks.
/// Now `PsmStream` writes a full chunk as soon as it fills, through the same writer, so
/// the writer sees the same chunks and the file is byte-identical (`write_table_chunked`
/// states why the chunk sequence is what matters). Only the demix features need the
/// whole table, because they are patched in after the loop; with
/// `emit_demix_features` the rows are kept and written at the end as before.
#[derive(Default)]
struct PsmRows {
    cid: Vec<u32>,
    peak_rank: Vec<i32>,
    apex_rt: Vec<f64>,
    apex_int: Vec<f32>,
    n_match: Vec<i32>,
    npred: Vec<i32>,
    corun: Vec<i32>,
    calrt: Vec<f64>,
    mz: Vec<f64>,
    z: Vec<i32>,
    label: Vec<String>,
    base: Vec<u32>,
    pform: Vec<String>,
    prot: Vec<String>,
    irt: Vec<f32>,
    contested: Vec<f64>,
    ms1_m1: Vec<Option<f64>>,
    ms1_mono: Vec<Option<f64>>,
    ms1_i1: Vec<Option<f64>>,
    ms1_i2: Vec<Option<f64>>,
    contested_count: Vec<f64>,
    apportioned: Vec<f64>,
    gate_apex: Vec<f32>,
    gate_peakspec: Vec<f32>,
    gate_coel: Vec<f32>,
    gate_se: Vec<f32>,
    deconv_expl: Vec<f32>,
    deconv_act: Vec<f32>,
    deconv_share: Vec<f32>,
    deconv_collin: Vec<f32>,
    deconv_shadow: Vec<f32>,
}

impl PsmRows {
    fn len(&self) -> usize {
        self.cid.len()
    }

    /// Append one row; its chromatogram rows are the caller's.
    fn push(&mut self, r: CandOut, cfg: &ExtractConfig) {
        self.cid.push(r.cid);
        self.peak_rank.push(r.peak_rank as i32);
        self.apex_rt.push(r.apex_rt);
        self.apex_int.push(r.apex_int);
        self.n_match.push(r.n_match);
        self.corun.push(r.corun);
        self.npred.push(r.npred);
        self.calrt.push(r.calrt);
        self.mz.push(r.mz);
        self.contested.push(r.contested);
        if cfg.emit_contested_features {
            self.contested_count.push(r.contested_count_frac);
            self.apportioned.push(r.apportioned_frac);
        }
        self.z.push(r.z);
        self.label.push(r.label);
        self.base.push(r.base);
        self.pform.push(r.pform);
        self.prot.push(r.prot);
        self.irt.push(r.irt);
        self.ms1_m1.push(r.ms1_m1);
        self.ms1_mono.push(r.ms1_mono);
        self.ms1_i1.push(r.ms1_i1);
        self.ms1_i2.push(r.ms1_i2);
        if cfg.emit_gate_diagnostics {
            self.gate_apex.push(r.gate_apex);
            self.gate_peakspec.push(r.gate_peak_spectral);
            self.gate_coel.push(r.gate_coelution);
            self.gate_se.push(r.gate_spectral_entropy);
        }
        if cfg.emit_demix_features {
            self.deconv_expl.push(r.deconv_explained);
            self.deconv_act.push(r.deconv_active);
            self.deconv_share.push(r.deconv_share);
            self.deconv_collin.push(r.deconv_collin);
            self.deconv_shadow.push(r.deconv_shadow);
        }
    }

    /// Move the pending rows out as the table's columns, in the table's column order,
    /// with library-wide candidate ids (`offset` is `Library::global_offset`).
    fn take_cols(&mut self, offset: u32, cfg: &ExtractConfig) -> Vec<Col> {
        use std::mem::take;
        let n = self.len();
        let mut cols = vec![
            Col::U32(
                "candidate_id".into(),
                self.cid.iter().map(|c| c + offset).collect(),
            ),
            Col::I32("peak_rank".into(), take(&mut self.peak_rank)),
            Col::F64("apex_rt".into(), take(&mut self.apex_rt)),
            Col::OptF64("apex_im".into(), vec![None; n]),
            Col::F32("apex_intensity".into(), take(&mut self.apex_int)),
            Col::I32("n_matched_fragments".into(), take(&mut self.n_match)),
            Col::I32("n_predicted_fragments".into(), take(&mut self.npred)),
            Col::I32("coelution_run".into(), take(&mut self.corun)),
            Col::F64("rt_pred_cal".into(), take(&mut self.calrt)),
            Col::F64("precursor_mz".into(), take(&mut self.mz)),
            Col::I32("charge".into(), take(&mut self.z)),
            Col::Str("label".into(), take(&mut self.label)),
            Col::U32("base_peptide_id".into(), take(&mut self.base)),
            Col::Str("peptidoform".into(), take(&mut self.pform)),
            Col::Str("protein".into(), take(&mut self.prot)),
            Col::F32("predicted_irt".into(), take(&mut self.irt)),
            Col::F64("contested_frac".into(), take(&mut self.contested)),
            Col::OptF64("ms1_isom1".into(), take(&mut self.ms1_m1)),
            Col::OptF64("ms1_mono".into(), take(&mut self.ms1_mono)),
            Col::OptF64("ms1_iso1".into(), take(&mut self.ms1_i1)),
            Col::OptF64("ms1_iso2".into(), take(&mut self.ms1_i2)),
        ];
        self.cid.clear();
        // Richer soft-competition columns only when emit_contested_features (default-off
        // keeps the schema byte-identical; contested_frac above is the pre-existing one).
        if cfg.emit_contested_features {
            cols.push(Col::F64(
                "contested_count_frac".into(),
                take(&mut self.contested_count),
            ));
            cols.push(Col::F64(
                "apportioned_frac".into(),
                take(&mut self.apportioned),
            ));
        }
        // Diagnostic gate-score columns only when enabled (default-off keeps the schema
        // byte-identical to the production chain).
        if cfg.emit_gate_diagnostics {
            cols.push(Col::F32("gate_apex".into(), take(&mut self.gate_apex)));
            cols.push(Col::F32(
                "gate_peak_spectral".into(),
                take(&mut self.gate_peakspec),
            ));
            cols.push(Col::F32("gate_coelution".into(), take(&mut self.gate_coel)));
            cols.push(Col::F32(
                "gate_spectral_entropy".into(),
                take(&mut self.gate_se),
            ));
        }
        if cfg.emit_demix_features {
            cols.push(Col::F32(
                "deconv_explained_frac".into(),
                take(&mut self.deconv_expl),
            ));
            cols.push(Col::F32("deconv_active".into(), take(&mut self.deconv_act)));
            cols.push(Col::F32(
                "deconv_share".into(),
                take(&mut self.deconv_share),
            ));
            cols.push(Col::F32(
                "deconv_max_collinearity".into(),
                take(&mut self.deconv_collin),
            ));
            cols.push(Col::F32(
                "shadow_kept_frac".into(),
                take(&mut self.deconv_shadow),
            ));
        }
        cols
    }
}

/// `psms_extracted` as the candidate loop produces it (X7).
///
/// Streamed, every `WRITE_TABLE_CHUNK_ROWS` pushed rows are written as one chunk through a
/// `TableWriter` opened as `write_table_hashed` opens it, and `finish` writes the short
/// tail, or the one empty chunk that fixes the schema of an empty table, then the footer.
/// That is the chunk sequence `write_table` cuts from the whole table, so the file is the
/// `write_table` file (`the_streamed_psms_table_is_the_write_table_file`). Unstreamed, the
/// rows are kept whole and `finish` hands them to `write_table_hashed`; the demix pass
/// needs that, because it patches rows after the loop. Either way the file is hashed as it
/// is written, so its report needs no read-back (docs/03_io_layer.md, "Hash on write").
struct PsmStream {
    /// The rows not yet written; unstreamed, every row.
    rows: PsmRows,
    /// `None` on the unstreamed path.
    writer: Option<TableWriter>,
    path: String,
    /// `Library::global_offset`, added to every written candidate id.
    offset: u32,
    /// Time spent writing, for the `extract: psms_extracted writer` log line.
    busy: std::time::Duration,
}

impl PsmStream {
    fn new(path: &str, streamed: bool, offset: u32) -> PsmStream {
        PsmStream {
            rows: PsmRows::default(),
            writer: streamed.then(|| TableWriter::new(path).with_content_hash()),
            path: path.to_string(),
            offset,
            busy: std::time::Duration::ZERO,
        }
    }

    /// Append one row, and write the pending rows when they make a full chunk. An error is
    /// the chunk's write failing; the rows it held are gone, so the caller stops.
    fn push(&mut self, r: CandOut, cfg: &ExtractConfig) -> Result<()> {
        self.rows.push(r, cfg);
        if let Some(w) = self.writer.as_mut() {
            if self.rows.len() == WRITE_TABLE_CHUNK_ROWS {
                let t = Instant::now();
                let written = w.write_cols(self.rows.take_cols(self.offset, cfg));
                self.busy += t.elapsed();
                written?;
            }
        }
        Ok(())
    }

    /// Write what is pending and publish the table: its rows and content hash, and the
    /// total write time.
    fn finish(mut self, cfg: &ExtractConfig) -> Result<(Written, std::time::Duration)> {
        let t = Instant::now();
        let n = match self.writer.take() {
            Some(mut w) => {
                // The tail, unless the rows were an exact multiple of the chunk, whose last
                // full chunk was written by `push`; an empty table still writes its one
                // empty chunk.
                if self.rows.len() > 0 || w.rows() == 0 {
                    w.write_cols(self.rows.take_cols(self.offset, cfg))?;
                }
                w.close_hashed()?
            }
            None => write_table_hashed(&self.path, self.rows.take_cols(self.offset, cfg))?,
        };
        self.busy += t.elapsed();
        Ok((n, self.busy))
    }
}

/// Finish the chromatogram table the writer thread was fed, or abandon it.
///
/// `psms_failed` is a `psms_extracted` chunk write that failed inside the candidate loop.
/// The loop stopped there, so the chromatogram table holds only the candidates before
/// that point. It is then not published: the writer is dropped unclosed, which removes its
/// temporary file (`AtomicPath`), and a chromatograms table already at the path stays as
/// it was, as the `psms_extracted` one does. Publishing it would put a truncated table
/// beside an older `psms_extracted`, a pair no extract wrote together. Published, the
/// table's rows and content hash are returned; the writer must have been built
/// [`TableWriter::with_content_hash`].
fn close_chromatograms(w: TableWriter, psms_failed: Option<anyhow::Error>) -> Result<Written> {
    match psms_failed {
        Some(e) => {
            drop(w);
            Err(e)
        }
        None => w.close_hashed(),
    }
}

pub fn run(p: ExtractParams) -> Result<(u64, u64)> {
    run_hashed(p).map(|(psms, chrom)| (psms.rows, chrom.rows))
}

/// [`run`], returning each output's row count and the content hash its report records
/// (`psms_extracted`, then `chromatograms`), so an orchestrator can record both artifacts
/// without reading and hashing them again.
pub fn run_hashed(mut p: ExtractParams) -> Result<(Written, Written)> {
    let t0 = Instant::now();
    // Taken out before the closures below borrow `p`; consumed where the file is read.
    let handed_windows = p.rt_windows.take();
    // Neither output may be one of the inputs (docs/31 F6).
    let inputs = [
        ("--ms2", p.ms2),
        ("--lib-precursors", p.library_precursors),
        ("--lib-fragments", p.library_fragments),
    ];
    mumdia_io::refuse_output_over_input(p.out_psms, &inputs)?;
    mumdia_io::refuse_output_over_input(p.out_chrom, &inputs)?;
    // Skip the bucketed page_search index when the fragindex backend is selected (the
    // default): it is never read on that path and costs a full sort plus several full
    // copies of every library fragment.
    let fragindex = matches!(p.cfg.matcher, MatcherKind::Fragindex);
    let build_bucketed = !fragindex;
    // The two co-elution strategies and the contested feature need a first pass to
    // build per-candidate elution profiles before shared peaks can be arbitrated.
    let two_pass = matches!(
        p.cfg.peak_claim,
        PeakClaim::CoelutionWinner
            | PeakClaim::CoelutionProportional
            | PeakClaim::CoelutionWinnerMargin
            | PeakClaim::CoelutionMultiCue
            | PeakClaim::CoelutionDemix
            | PeakClaim::CoelutionShadow
    ) || p.cfg.emit_contested_features;
    // The GLOBAL fragment index is needed only by the paths that probe arbitrary candidate
    // windows through it: the two-pass arbitration and the per-apex-scan demix. The
    // default streamed accumulation gives each probing task a `LocalIndex` of its own
    // sub-range instead, and needs only the whole library's bin geometry.
    let global_index = fragindex && (two_pass || p.cfg.emit_demix_features);
    // The mass calibration is read FIRST: it is one small JSON file, and its learned
    // tolerance is what the fragment index is built at, so building the index while the
    // spectra decode needs it up front. Its errors and its log lines stay where they were,
    // after the spectra (`mass?` below).
    let mass = read_mass_cal(&p);
    let load_indexed = || -> Result<(Library, Option<FragIndex>, Option<LogBins>)> {
        let lib = Library::load_for_stage(
            p.library_precursors,
            p.library_fragments,
            p.fragment_offset,
            p.precursor_span,
            p.cfg.bucket_size,
            build_bucketed,
        )?;
        // fragindex backend, built once at the learned fragment tolerance when a path
        // needs the global index; otherwise the bucketed `Library::page_search` path is
        // used. `Prober::probe` dispatches on this per peak. The geometry is the global
        // index's own when it is built, and computed on its own when it is not; either
        // way it is `FragIndex::geometry` at the learned tolerance.
        let (fidx, bins) = match (&mass, fragindex, global_index) {
            (Ok(m), true, true) => {
                let f = FragIndex::build(&lib, m.frag_tol);
                let b = f.bins().clone();
                (Some(f), Some(b))
            }
            (Ok(m), true, false) => (None, Some(FragIndex::geometry(&lib, m.frag_tol))),
            _ => (None, None),
        };
        Ok((lib, fidx, bins))
    };
    // On the fragindex path the spectra decode runs concurrently with the library load and
    // the index build: they are independent, and the scans are resident during the index
    // build either way, so the overlap does not raise the peak. The bucketed path keeps them
    // in sequence, because its library load builds a full sorted copy of every fragment and
    // holding the scans through that transient would. Errors are reported in the old order:
    // the library's, then the allowlist's and the RT windows', then the spectra's.
    let (loaded, decoded) = if fragindex {
        let (l, d) = rayon::join(load_indexed, || decode_unlent(&p));
        (l, Some(d))
    } else {
        (load_indexed(), None)
    };
    let (lib, prebuilt_fidx, prebuilt_bins) = loaded?;

    // Optional candidate allowlist (gate-first-then-compete): restrict extraction to
    // the accepted survivors of a prior gate-on run so the two-pass peak-claim profile
    // map stays small.
    let restrict: Option<CandMask> = match p.restrict_candidates {
        Some(path) => {
            let t = TableFile::open(path)?;
            let mut s = CandMask::new(lib.n_candidates());
            for c in t.u32("candidate_id")? {
                s.insert(c);
            }
            info!(
                restrict_candidates = s.len(),
                "extract: restricting to candidate allowlist"
            );
            Some(s)
        }
        None => None,
    };

    // run windows indexed by candidate_id: the orchestrator's in-memory copy when it was
    // fitted for this library and this run_windows file (`rt_im_train::RtWindows` says why
    // that is the same arrays), the file otherwise.
    let ncand = lib.n_candidates();
    let handed_windows = match handed_windows {
        Some(w) => match w.mismatch(p.library_precursors, p.run_windows, ncand) {
            None => Some(w),
            Some(why) => {
                warn!(
                    handed = w.len(),
                    candidates = ncand,
                    run_windows = p.run_windows,
                    "extract: the RT windows handed over are not this extract's ({why}); \
                     reading the run_windows file instead"
                );
                None
            }
        },
        None => None,
    };
    let RtWindows {
        rt_cal,
        rt_lo,
        rt_hi,
        ..
    } = match handed_windows {
        Some(w) => {
            info!(
                candidates = ncand,
                "extract: RT windows handed over in memory; run_windows is not re-read"
            );
            w
        }
        None => read_run_windows(p.run_windows, ncand)?,
    };

    // Decoded here unless the caller lent its own copies (see `ExtractParams::scans` and
    // `decode_unlent`); on the fragindex path the decode already ran, concurrently with the
    // library load.
    let (owned_ms2, owned_ms1) = match decoded {
        Some(d) => d?,
        None => decode_unlent(&p)?,
    };
    let (scans, ms1_scans): (&[Ms2Scan], &[Ms1Scan]) = (
        owned_ms2
            .as_deref()
            .unwrap_or_else(|| p.scans.map(|s| s.ms2).unwrap_or(&[])),
        owned_ms1
            .as_deref()
            .unwrap_or_else(|| p.scans.and_then(|s| s.ms1).unwrap_or(&[])),
    );
    let ms1_rts: Vec<f64> = ms1_scans.iter().map(|s| s.rt_seconds).collect();
    info!(
        candidates = ncand,
        scans = scans.len(),
        ms1 = ms1_scans.len(),
        "extract: loaded; probing peaks"
    );

    // Isolation-window -> sorted scan RTs, for zero-filled chromatogram grids.
    let windows: Vec<(f64, f64, Vec<f64>)> = if p.cfg.emit_window_grid {
        let mut tmp: HashMap<(u64, u64), Vec<f64>> = HashMap::new();
        for s in scans {
            tmp.entry((s.window.lower_mz.to_bits(), s.window.upper_mz.to_bits()))
                .or_default()
                .push(s.rt_seconds);
        }
        let mut w: Vec<(f64, f64, Vec<f64>)> = tmp
            .into_iter()
            .map(|((lb, ub), mut v)| {
                v.sort_by(|a, b| a.total_cmp(b));
                (f64::from_bits(lb), f64::from_bits(ub), v)
            })
            .collect();
        w.sort_by(|a, b| a.0.total_cmp(&b.0));
        w
    } else {
        Vec::new()
    };
    // Widest isolation window, so the per-candidate scan over `windows` (sorted by lower
    // m/z) can binary-search the span that can possibly cover a precursor instead of
    // walking all of them: a window covering `pm` has `lower_mz <= pm` and
    // `lower_mz >= upper_mz - max_width >= pm - max_width`.
    let window_max_width =
        windows
            .iter()
            .map(|w| w.1 - w.0)
            .fold(0.0f64, |a, b| if b > a { b } else { a });

    // Per-run mass recalibration (optional): the scalar offset + learned tolerance, plus
    // an optional m/z-dependent correction grid (mass_cal_loess), read at the top of the
    // stage (`read_mass_cal`) and reported here, where it always was.
    let MassCalRead {
        frag_offset,
        frag_tol,
        grid_mz,
        grid_ppm,
        from_file,
    } = mass?;
    if let Some(path) = from_file.as_deref() {
        info!(
            frag_ppm_offset = frag_offset,
            frag_tol_ppm = frag_tol,
            mz_cal_grid = grid_mz.len(),
            "extract: using mass recalibration"
        );
        // `extract.frag_tol_ppm` is a FALLBACK, not a setting, in any orchestrated
        // run: search-seed always writes `frag_tol_ppm` into masscal.json --
        // including in its calibration-failure branch, where it writes
        // `search_seed.fragment_tol_ppm` -- and both orchestrators always pass
        // `--mass-cal`. So a config carrying `extract.frag_tol_ppm = 40` extracted at
        // the learned value with nothing said about it. Say it, because a config key
        // that is read and then ignored is worse than one that is absent.
        if (frag_tol - p.cfg.frag_tol_ppm).abs() > 1e-9 {
            warn!(
                configured_frag_tol_ppm = p.cfg.frag_tol_ppm,
                learned_frag_tol_ppm = frag_tol,
                mass_cal = path,
                "extract: extract.frag_tol_ppm is overridden by the learned tolerance                      from mass calibration. It applies only when no --mass-cal is passed;                      to widen the search tolerance, set search_seed.fragment_tol_ppm"
            );
        }
    }
    // The grid is used only if both arrays agree in length and have >= 2 points.
    let mass_off = if grid_mz.len() >= 2 && grid_mz.len() == grid_ppm.len() {
        MassOffset {
            scalar_ppm: frag_offset,
            grid_mz,
            grid_ppm,
        }
    } else {
        MassOffset {
            scalar_ppm: frag_offset,
            grid_mz: Vec::new(),
            grid_ppm: Vec::new(),
        }
    };

    // The fragment index, built with the library above at this same learned tolerance,
    // when a path needs the global one; and the geometry the streamed path's task-local
    // indexes bin with.
    let fidx = match (global_index, prebuilt_fidx) {
        (true, Some(f)) => Some(f),
        (true, None) => Some(FragIndex::build(&lib, frag_tol)),
        (false, _) => None,
    };
    let stream_bins: Option<LogBins> = match (fragindex, prebuilt_bins) {
        (true, Some(b)) => Some(b),
        (true, None) => Some(match &fidx {
            Some(f) => f.bins().clone(),
            None => FragIndex::geometry(&lib, frag_tol),
        }),
        (false, _) => None,
    };
    let local_stats = LocalIndexStats::default();

    // Peak-major accumulation. The fast path (single pass, fragment index, no candidate
    // allowlist) leaves `acc` empty and fills `stream_groups` instead; every other path
    // materialises the whole run here and is handed to the driver as one batch.
    let mut acc: HashMap<u32, Vec<Hit>> = HashMap::new();
    let mut stream_groups: Option<Vec<WinGroup>> = None;
    // Reused per-peak buffer of (candidate_id, local_frag_index, predicted_intensity).
    let mut claimants: Vec<(u32, u16, f32)> = Vec::new();
    // Per-candidate contested-peak stats under the co-elution arbitration, for the
    // non-destructive soft competition features. Populated only on the two-pass path.
    let mut contested: HashMap<u32, Contested> = HashMap::new();
    let claim_margin = p.cfg.peak_claim_margin as f32;

    if !two_pass {
        if stream_bins.is_some() {
            // Parallel across isolation-window groups (bit-identical to serial: the
            // cascade rt-sorts each candidate's hits before summing), with or without a
            // candidate allowlist: the allowlist is applied inside the probe, before the
            // peak claim, as on the serial path. Until 2026-09-18 an allowlist routed to
            // the serial path below, whose whole-run accumulator ignores
            // `windows_in_flight`; on a 35M-candidate immunopeptidomics library that was
            // a 325 GB, 2-hour extract, and on 83M candidates it aborted at 471 GB.
            // Streamed: the driver below probes the windows in batches and writes each
            // candidate out as soon as no later window can add a hit to it, so the whole
            // run's hits are never resident. Measured at 1.6 billion hits (35.9 GiB of
            // payload) on the HYE benchmark, which was 60% of extract's 61.3 GiB peak.
            stream_groups = Some(window_groups(&lib, scans));
        } else {
            let pr = Prober {
                fidx: fidx.as_ref(),
                lib: &lib,
                frag_tol,
            };
            for (si, scan) in scans.iter().enumerate() {
                let (lo, hi) = lib.candidate_range(scan.window.lower_mz, scan.window.upper_mz);
                if hi <= lo {
                    continue;
                }
                let rt = scan.rt_seconds;
                let hs = si as u32;
                for peak in &scan.peaks {
                    let inten = peak.intensity;
                    let obs_mz = peak.mz;
                    let q_mz = obs_mz as f64 / mass_off.factor_at(obs_mz as f64);
                    // Collect every co-isolated, in-RT-window candidate matching this
                    // peak, then apportion per the claim strategy. In wide DIA one peak
                    // matches many candidates (~98% of fragments collide).
                    claimants.clear();
                    {
                        let mut push = |cid: u32, frag: u16, pi: f32| {
                            let c = cid as usize;
                            if rt < rt_lo[c] || rt > rt_hi[c] {
                                return;
                            }
                            if let Some(s) = &restrict {
                                if !s.contains(cid) {
                                    return;
                                }
                            }
                            claimants.push((cid, frag, pi));
                        };
                        pr.probe(None, q_mz, lo, hi, &mut push);
                    }
                    if claimants.is_empty() {
                        continue;
                    }
                    match p.cfg.peak_claim {
                        PeakClaim::WinnerPredictedIntensity => {
                            let mut best = 0usize;
                            for i in 1..claimants.len() {
                                let a = claimants[i];
                                let b = claimants[best];
                                if a.2 > b.2 || (a.2 == b.2 && a.0 < b.0) {
                                    best = i;
                                }
                            }
                            let (cid, frag, _) = claimants[best];
                            acc.entry(cid).or_default().push(Hit {
                                scan: hs,
                                frag,
                                inten,
                                obs_mz,
                            });
                        }
                        PeakClaim::Proportional => {
                            let sump: f32 = claimants.iter().map(|c| c.2.max(0.0)).sum();
                            for &(cid, frag, pi) in &claimants {
                                let share = if sump > 0.0 {
                                    inten * (pi.max(0.0) / sump)
                                } else {
                                    inten / claimants.len() as f32
                                };
                                acc.entry(cid).or_default().push(Hit {
                                    scan: hs,
                                    frag,
                                    inten: share,
                                    obs_mz,
                                });
                            }
                        }
                        // None (and the co-elution variants, which never reach here).
                        _ => {
                            for &(cid, frag, _) in &claimants {
                                acc.entry(cid).or_default().push(Hit {
                                    scan: hs,
                                    frag,
                                    inten,
                                    obs_mz,
                                });
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Two-pass co-elution peak-claim, parallelized across isolation windows
        // (each window's candidates interact only within it, so the two expensive
        // probing passes fan out over the ~150 windows).
        // CoelutionMultiCue ships non-destructive by default (cue-weighted split flows
        // into the contested/apportioned FEATURES only). It becomes destructive, rewriting
        // the extracted intensities so ALL downstream features recompute on the competed
        // evidence, only when `claim_cues.reassign` is set (entrapment-gated).
        let reassign = matches!(
            p.cfg.peak_claim,
            PeakClaim::CoelutionWinner
                | PeakClaim::CoelutionProportional
                | PeakClaim::CoelutionWinnerMargin
                | PeakClaim::CoelutionDemix
                | PeakClaim::CoelutionShadow
        ) || (matches!(p.cfg.peak_claim, PeakClaim::CoelutionMultiCue)
            && p.cfg.claim_cues.reassign);
        let (a, c) = extract_twopass_windows(
            fidx.as_ref(),
            &lib,
            scans,
            &rt_lo,
            &rt_hi,
            &rt_cal,
            ms1_scans,
            &ms1_rts,
            &mass_off,
            frag_tol,
            p.cfg,
            restrict.as_ref(),
            reassign,
            claim_margin,
        );
        acc = a;
        contested = c;
    }
    if stream_groups.is_none() {
        info!(
            materialized = acc.len(),
            "extract: candidates with evidence"
        );
    }
    if stream_groups.is_none() {
        // The whole-run hit accumulator is the largest unattributed structure of the
        // stage: extract's sampled peak was 61.38 GiB against 3.8 GiB of named buffers
        // (docs/27 section 0). Payload, growth slack and the per-candidate spine are
        // reported apart because every hit arrives through `entry().or_default().push()`,
        // which grows geometrically, so the slack is not a rounding error.
        let (mut hits, mut cap) = (0usize, 0usize);
        for v in acc.values() {
            hits += v.len();
            cap += v.capacity();
        }
        let sz = std::mem::size_of::<Hit>();
        crate::memlog::report(
            "extract hit accumulator",
            &[
                ("hits_payload", hits * sz),
                ("growth_slack", cap.saturating_sub(hits) * sz),
                ("vec_spine", acc.len() * std::mem::size_of::<Vec<Hit>>()),
                (
                    "hashmap_table",
                    acc.capacity() * (std::mem::size_of::<(u32, Vec<Hit>)>() + 1),
                ),
            ],
        );
        info!(
            hits,
            candidates = acc.len(),
            hit_bytes = sz,
            hits_per_candidate = hits as f64 / acc.len().max(1) as f64,
            "mem: extract hit accumulator shape"
        );
    }

    // Cascade + apex per candidate.
    let scan_window = p.cfg.fixed_scan_window.max(1);

    // Chromatogram rows stream to parquet chunk by chunk (see the candidate loop below).
    // Hashed as it is written: the report's content hash then needs no read-back of the
    // run's largest artifact (docs/03_io_layer.md, "Hash on write"). The layout and the
    // encodings are the shared ones ([`crate::chromatograms::writer`]: the `rt` axis PLAIN).
    // Under v2 every row passes through one `Encoder` in table order, which counts the rows
    // to know where each row group starts, so it and the writer take the same row-group
    // size.
    let chrom_layout =
        crate::chromatograms::Layout::from_schema_version(p.cfg.chromatogram_schema)?;
    let chrom_rg_rows = crate::chromatograms::row_group_rows();
    if chrom_rg_rows != CHROM_ROW_GROUP_ROWS {
        info!(
            rows = chrom_rg_rows,
            "extract: chromatogram row groups resized by {}",
            crate::chromatograms::ROW_GROUP_ROWS_ENV
        );
    }
    let chrom_writer = crate::chromatograms::writer(p.out_chrom, chrom_rg_rows).with_content_hash();
    let mut chrom_encoder = crate::chromatograms::Encoder::new(chrom_rg_rows);

    // Deterministic output order (a HashMap's iteration order is randomized,
    // and downstream floating-point sums in the rescorer are order-sensitive).
    // Empty on the streamed path, where the driver gathers each flush's candidates out of
    // the CSR accumulator in ascending order instead.
    let mut cand_ids: Vec<u32> = acc.keys().cloned().collect();
    cand_ids.sort_unstable();

    // The eager accumulator is drained into CSR stores of `CAND_CHUNK` candidates at a
    // time, in the deterministic sorted order, and candidates are then processed in
    // parallel. Each candidate's work depends only on its own hits plus read-only
    // library/window/MS1 data, and every float reduction (apex sum, wsum, chromatogram
    // grids) is self-contained, so collecting the results in cand_ids order and appending
    // them serially below yields output (PSM rows and chromatogram rows) byte-identical to
    // the serial loop. The optional Pearson gate now allocates its two scratch vectors per
    // candidate (was a hoisted reused buffer) because buffers cannot be shared across
    // parallel candidates.

    // psms_extracted rows, streamed in `write_table`'s own chunks unless the demix pass
    // needs the whole table (see `PsmStream`). A chunk write that fails inside the loop
    // stops it; the error is kept here and returned once the chromatogram writer has been
    // stopped without publishing (`close_chromatograms`).
    let stream_psms = !p.cfg.emit_demix_features;
    let mut psms = PsmStream::new(p.out_psms, stream_psms, lib.global_offset);
    let mut psms_err: Option<anyhow::Error> = None;

    // Apex-scan lookup for spectrum-centric demixing (D2): rt_bits -> scan indices.
    // Built only when demixing is requested, so the default path pays nothing.
    let rt_scan: HashMap<u64, Vec<u32>> = if p.cfg.emit_demix_features {
        let mut m: HashMap<u64, Vec<u32>> = HashMap::new();
        for (si, s) in scans.iter().enumerate() {
            m.entry(s.rt_seconds.to_bits()).or_default().push(si as u32);
        }
        m
    } else {
        HashMap::new()
    };

    let per_candidate = |cid: u32, hits: &mut [Hit]| -> Vec<CandOut> {
        // distinct matched fragments (tier b), as a bitmask over the candidate's local
        // fragment ordinals rather than a sorted, deduplicated `u16` per hit.
        let mut distinct = FragSet::default();
        for h in hits.iter() {
            distinct.insert(h.frag);
        }
        let n_distinct = distinct.len();
        if n_distinct < p.cfg.presence_min_matched.max(1) {
            return Vec::new();
        }

        // Group hits into scan groups by RT (dedupe same fragment in a scan by max).
        // Probing walks a window's scans in ascending index, so the hits of a candidate are
        // already rt-ascending in the ordinary case; the sort then allocates scratch as large
        // as the vector itself for nothing. Equal-rt hits collapse to `max` per (rt, frag)
        // below either way, so skipping a sort that would not move anything is exact.
        if !hits
            .windows(2)
            .all(|w| hit_rt(scans, &w[0]) <= hit_rt(scans, &w[1]))
        {
            hits.sort_by(|a, b| hit_rt(scans, a).total_cmp(&hit_rt(scans, b)));
        }
        // Scan groups, dense (`ScanGroups`): per group its RT and each fragment's max
        // observed intensity. The per-group fragment order is the ordinal order, as the
        // `BTreeMap` per group this replaces fixed it, so the f32 apex sum is
        // deterministic and unchanged.
        let width = hits.iter().map(|h| h.frag as usize + 1).max().unwrap_or(0);
        let mut groups = ScanGroups::new(width);
        for h in hits.iter() {
            let h_rt = hit_rt(scans, h);
            match groups.rt.last() {
                Some(&rt) if (rt - h_rt).abs() < 1e-9 => {
                    let i = groups.len() - 1;
                    groups.merge_max(i, h.frag, h.inten);
                }
                _ => {
                    groups.push_group(h_rt);
                    let i = groups.len() - 1;
                    groups.insert(i, h.frag, h.inten);
                }
            }
        }

        let (fmzs0, fints0, _) = lib.cand_frags(cid);

        // Acquisition scan grid: the covering isolation-window scans within the
        // RT window. Project the sparse hit-groups onto it so apex counting and
        // the co-elution run see MISSING acquisition scans (count 0, and they
        // break a run) rather than only scans that happened to carry a hit. When
        // no covering-window grid is available, fall back to the sparse groups.
        let grid: Vec<f64> = if !windows.is_empty() {
            let pm = lib.prec_mz[cid as usize];
            let (lo, hi) = (rt_lo[cid as usize], rt_hi[cid as usize]);
            // `windows` is sorted by lower m/z, so a covering window has
            // `lower_mz <= pm` AND, since its width is at most `window_max_width`,
            // `lower_mz >= pm - window_max_width`. Binary-search that span instead of
            // scanning all ~150 windows per candidate; the membership test inside the
            // span is unchanged, so the same windows contribute in the same order.
            let s = windows.partition_point(|w| w.0 < pm - window_max_width);
            let e = windows.partition_point(|w| w.0 <= pm);
            let mut g: Vec<f64> = Vec::new();
            let mut covering = 0usize;
            for (wl, wu, rts) in &windows[s..e] {
                if *wl <= pm && pm <= *wu {
                    let a = rts.partition_point(|&r| r < lo);
                    let b = rts.partition_point(|&r| r <= hi);
                    g.extend_from_slice(&rts[a..b]);
                    covering += 1;
                }
            }
            // One covering window contributes one already-ascending slice of its own
            // sorted scan RTs, which is the overwhelmingly common case; only an overlap
            // of two windows needs the merge sorting.
            if covering > 1 {
                g.sort_by(|a, b| a.total_cmp(b));
            }
            g.dedup();
            g
        } else {
            Vec::new()
        };
        if !grid.is_empty() {
            let mut aligned = ScanGroups::empty_on(&grid, width);
            // Both sides are ascending (`grid` is sorted and deduplicated; the scan groups
            // were built from rt-sorted hits), so this is a merge rather than a per-
            // candidate `HashMap` of the grid. The match is still on the exact bit
            // pattern, so a group whose RT is not a grid RT is dropped exactly as before.
            let mut j = 0usize;
            for i in 0..groups.len() {
                let rt = groups.rt(i);
                while j < grid.len() && grid[j] < rt {
                    j += 1;
                }
                if j < grid.len() && grid[j].to_bits() == rt.to_bits() {
                    aligned.copy_group(j, &groups, i);
                    j += 1;
                }
            }
            groups = aligned;
        }

        // Apex: the scan group with the most distinct matched fragments, allowing
        // scans within `apex_count_tol` of that maximum (so a slightly-lower-count
        // but much more intense scan can still win), then the one maximizing the
        // summed intensity of its 3 most intense fragments. This is the diagnostic-
        // plot apex; robust to a bright single-fragment interferent that would win a
        // pure max-summed-intensity apex in chimeric DIA.
        // Distinct-fragment count per scan group, optionally smoothed by a centered
        // rolling SUM (`apex_count_window`). Low-intensity fragments flicker in/out
        // scan-to-scan; a single-scan count then spikes at noise scans and misplaces
        // the apex. The rolling sum makes the apex land in the region of *sustained*
        // fragment presence. It is deliberately a sum, not a mean: the window is
        // truncated at the profile edges, so interior positions accumulate more than
        // edge positions, which center-weights the apex toward the RT-window centre
        // (~= the predicted RT). That mild RT-prior steers off off-centre interfering
        // peaks; measured, sum beats mean by ~+300 IDs on the AIF file. Window 1
        // reproduces the exact per-scan-count behavior.
        let counts: Vec<usize> = (0..groups.len()).map(|i| groups.count(i)).collect();
        let w = p.cfg.apex_count_window.max(1);
        let r = w / 2;
        let sigma = p.cfg.apex_gaussian_sigma_scans;
        // Smoothed per-scan fragment-count score. Default is the truncated
        // rolling SUM (`apex_count_window`); with `apex_gaussian_sigma_scans` > 0
        // a Gaussian matched filter (radius 3*sigma) is used instead. Both are
        // deterministic and reduce to the raw per-scan count when disabled.
        let smoothed: Vec<f64> = if sigma > 0.0 {
            let radius = (sigma * 3.0).ceil() as usize;
            let kernel: Vec<f64> = (0..=2 * radius)
                .map(|k| {
                    let d = k as f64 - radius as f64;
                    (-0.5 * (d / sigma).powi(2)).exp()
                })
                .collect();
            (0..counts.len())
                .map(|i| {
                    let mut acc = 0.0;
                    for (k, &wt) in kernel.iter().enumerate() {
                        let idx = i as isize + k as isize - radius as isize;
                        if idx >= 0 && (idx as usize) < counts.len() {
                            acc += counts[idx as usize] as f64 * wt;
                        }
                    }
                    acc
                })
                .collect()
        } else if w <= 1 {
            counts.iter().map(|&c| c as f64).collect()
        } else {
            (0..counts.len())
                .map(|i| {
                    let lo = i.saturating_sub(r);
                    let hi = (i + r).min(counts.len() - 1);
                    counts[lo..=hi].iter().sum::<usize>() as f64
                })
                .collect()
        };
        let maxc = smoothed.iter().copied().fold(0.0f64, f64::max);
        let thresh = (maxc - p.cfg.apex_count_tol as f64).max(0.0);
        // Optional Gaussian RT prior on the apex tiebreak: among count-qualified
        // scans, multiply the top-3 intensity by exp(-0.5*((rt - rt_pred_cal)/sigma)^2)
        // so a distant-from-prediction interferent inside a wide RT window cannot
        // define the apex. sigma = `apex_rt_prior_s`; 0 (or an unset rt_cal) disables it.
        let rt_prior_sigma = p.cfg.apex_rt_prior_s;
        let rt_cal_c = rt_cal[cid as usize];
        let use_prior = rt_prior_sigma > 0.0 && rt_cal_c > 0.0;
        // Signature-ion apex tiebreak: sum the OBSERVED intensity of the top-K
        // PREDICTED fragments (`apex_top_fragments`; 0 -> a default of 3) at each
        // qualifying scan, instead of the 3 brightest observed peaks. A bright
        // interferent on a non-signature ion can then no longer define the apex.
        let k_sig = if p.cfg.apex_top_fragments > 0 {
            p.cfg.apex_top_fragments
        } else {
            3
        };
        let sig: Vec<u16> = {
            let mut ord: Vec<usize> = (0..fints0.len()).collect();
            ord.sort_by(|&a, &b| fints0[b].total_cmp(&fints0[a]));
            ord.into_iter().take(k_sig).map(|o| o as u16).collect()
        };
        let mut apex_rt = groups.rt(0);
        let mut apex_sum = 0.0f32;
        let mut best_sig = f32::NEG_INFINITY;
        for (i, &n_here) in counts.iter().enumerate() {
            if n_here == 0 || smoothed[i] < thresh {
                continue;
            }
            let rt = groups.rt(i);
            let sig_sum: f32 = sig.iter().map(|&o| groups.or_zero(i, o)).sum();
            let prior = if use_prior {
                (-0.5 * ((rt - rt_cal_c) / rt_prior_sigma).powi(2)).exp() as f32
            } else {
                1.0
            };
            let score = if p.cfg.apex_evidence_rank {
                // Breadth-of-evidence apex: the count of distinct co-eluting
                // predicted fragments at this scan dominates; observed signature
                // intensity only breaks ties within [0,1). Interference-resistant
                // in wide-window DIA (a chimeric-intensity spike cannot outvote a
                // scan where more of the peptide's own transitions co-elute).
                let n_frag = n_here as f32;
                let tie = sig_sum / (sig_sum + 1.0);
                (n_frag + tie) * prior
            } else {
                // Legacy: signature-ion observed intensity (x RT prior). Bit-identical
                // to the previous behaviour (prior = 1.0 when the RT prior is off).
                sig_sum * prior
            };
            if score > best_sig {
                best_sig = score;
                apex_rt = rt;
                apex_sum = groups.sum(i); // report full apex intensity
            }
        }

        // Co-elution run: max consecutive scan groups with >= min_coelution frags.
        let mut best_run = 0usize;
        let mut cur = 0usize;
        for &n_here in &counts {
            if n_here >= p.cfg.presence_min_coelution.max(1) {
                cur += 1;
                best_run = best_run.max(cur);
            } else {
                cur = 0;
            }
        }

        // Acceptance (tier c): presence, consecutive-scan run, and matched
        // fraction of the predicted fragments (symmetric discriminator).
        let matched_fraction = n_distinct as f64 / (fmzs0.len().max(1) as f64);
        if n_distinct < p.cfg.presence_min_fragments.max(1)
            || best_run < scan_window
            || best_run < p.cfg.min_coelution_run
            || matched_fraction < p.cfg.min_matched_fraction
        {
            return Vec::new();
        }

        let c = lib.cand(cid);

        // MS1 apex isotope intensities at a given RT (nearest MS1 scan). Factored
        // so both the selected apex (rank 0) and any promoted alternate peak (#7)
        // compute their own MS1 evidence at their own apex RT. Computed BEFORE the
        // acceptance gate so MS1 evidence can rescue a candidate the single-scan
        // fragment-Pearson gate would otherwise reject.
        let ms1_at = |rt: f64| -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
            if ms1_scans.is_empty() {
                return (None, None, None, None);
            }
            let j = nearest_index(&ms1_rts, rt);
            let s = &ms1_scans[j];
            let z = c.charge as f64;
            let sp = ISOTOPE_SPACING / z;
            let tol = p.cfg.prec_tol_ppm;
            (
                Some(sum_near(&s.mz, &s.intensity, c.precursor_mz - sp, tol) as f64),
                Some(sum_near(&s.mz, &s.intensity, c.precursor_mz, tol) as f64),
                Some(sum_near(&s.mz, &s.intensity, c.precursor_mz + sp, tol) as f64),
                Some(sum_near(&s.mz, &s.intensity, c.precursor_mz + 2.0 * sp, tol) as f64),
            )
        };
        let (o_ms1_m1, o_ms1_mono, o_ms1_i1, o_ms1_i2) = ms1_at(apex_rt);
        // Cheap MS1 support: mono present and the +1/mono ratio in a plausible
        // averagine band. Used only as the rescue signal for the Pearson gate.
        let ms1_support = {
            let mono = o_ms1_mono.unwrap_or(0.0);
            let i1 = o_ms1_i1.unwrap_or(0.0);
            mono > 0.0 && i1 > 0.0 && {
                let r = i1 / mono;
                (0.1..=1.5).contains(&r)
            }
        };

        // Optional tier-d Pearson gate (kept for configurability; matched fraction
        // above is the primary symmetric discriminator). With `ms1_rescue`, a
        // candidate that fails the single-scan fragment Pearson is kept when it has
        // adequate matched fragments AND MS1 isotope-pattern support.
        // The first group within 1e-9 s of the apex RT: the group the apex was read from.
        let apex_gi: Option<usize> =
            (0..groups.len()).find(|&i| (groups.rt(i) - apex_rt).abs() < 1e-9);
        // Spectral-agreement score closures, evaluated lazily: the acceptance gate
        // needs only the ACTIVE `gate_mode`'s score, and the four diagnostic scores
        // are computed only when `emit_gate_diagnostics` is set (see below), so the
        // default chain pays the same per-candidate cost as before this feature.
        let apex_obs: Option<Vec<f64>> = apex_gi.map(|gi| {
            (0..fmzs0.len())
                .map(|k| groups.or_zero(gi, k as u16) as f64)
                .collect()
        });
        let pred_f64: Vec<f64> = fints0.iter().map(|x| *x as f64).collect();
        // Single-apex-scan intensity Pearson (1.0 when no apex scan resolved -> do
        // not reject on spectral agreement).
        let apex_pearson = || match &apex_obs {
            Some(obs) => crate::stats::pearson(obs, &pred_f64),
            None => 1.0,
        };
        // spectral_entropy_similarity_sqrt of the apex spectrum (shared kernel in
        // features::entropy; best single target/decoy gate discriminator).
        let apex_entropy = || match &apex_obs {
            Some(obs) => {
                crate::stages::features::entropy::spectral_entropy_similarity_sqrt(obs, &pred_f64)
            }
            None => 1.0,
        };
        let peak_spec = || peak_spectral_score(&groups, &sig, fints0);
        // `to_vec` only where the co-elution score is actually asked for (a non-default
        // gate mode, or the diagnostics), so the default chain never materialises it.
        let coel = || coelution_gate_score(&groups, &distinct.to_vec(), &sig, fints0);

        if p.cfg.gate_min_score > 0.0 {
            // Acceptance gate. `gate_min_score` thresholds the ACTIVE gate_mode's
            // spectral-agreement score (plan Section 9): the legacy single-apex-scan
            // Pearson (one chimeric scan can dominate), the peak-integrated spectral
            // Pearson, the apex spectral-entropy similarity, the temporal co-elution
            // score, or Combined (both, more specific). Only the active score computes.
            let rejected = match p.cfg.gate_mode {
                GateMode::ApexPearson => apex_pearson() < p.cfg.gate_min_score,
                GateMode::PeakSpectral => peak_spec() < p.cfg.gate_min_score,
                GateMode::SpectralEntropy => apex_entropy() < p.cfg.gate_min_score,
                GateMode::Coelution => coel() < p.cfg.gate_min_score,
                GateMode::Combined => {
                    peak_spec() < p.cfg.gate_min_score || coel() < p.cfg.gate_coelution_min
                }
            };
            if rejected {
                let rescued = p.cfg.ms1_rescue
                    && ms1_support
                    && n_distinct >= p.cfg.presence_min_fragments.max(1);
                if !rescued {
                    return Vec::new();
                }
            }
        }

        // Diagnostic scores (all four metrics, for the offline gate-metric
        // comparison). Computed and emitted ONLY when `emit_gate_diagnostics` is set,
        // so the default psms.parquet schema and per-candidate compute are unchanged
        // (sensitivity-program: default-off, byte-identical). Zero when off (the four
        // columns are not written).
        let (gate_apex, gate_peak_spectral, gate_coelution, gate_spectral_entropy) =
            if p.cfg.emit_gate_diagnostics {
                (apex_pearson(), peak_spec(), coel(), apex_entropy())
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

        // Soft competition features from the co-elution arbitration (all 0 when the
        // two-pass path did not run). contested_frac: fraction of contested INTENSITY
        // lost to better co-eluters. contested_count_frac: fraction of contested
        // fragment-PEAKS lost. apportioned_frac: fraction of contested intensity the
        // candidate retains under proportional apportionment (1 = keeps all, ~0 = a
        // peak-borrower stripped by its co-eluting competitors).
        let cst = contested.get(&cid).copied().unwrap_or_default();
        let contested_val = {
            let t = cst.won + cst.lost;
            if t > 0.0 {
                cst.lost / t
            } else {
                0.0
            }
        };
        let contested_count_frac = {
            let n = cst.n_won + cst.n_lost;
            if n > 0 {
                cst.n_lost as f64 / n as f64
            } else {
                0.0
            }
        };
        let apportioned_frac = {
            let t = cst.won + cst.lost;
            if t > 0.0 {
                cst.apportioned / t
            } else {
                0.0
            }
        };

        let (fmzs, fints, fnames) = lib.cand_frags(cid);

        // Per-fragment intensity-weighted observed m/z (for mass accuracy). Indexed by
        // the candidate-local ordinal rather than hashed: a fragment ordinal is
        // `0..n_frag` by construction, so the map was a hash table over a dense range.
        // An untouched entry keeps `sum_w == 0`, which takes the same theoretical-m/z
        // fallback the absent map entry took.
        let mut wsum: Vec<(f64, f64)> = vec![(0.0, 0.0); fmzs.len()]; // (sum w*mz, sum w)
        for h in hits.iter() {
            if let Some(e) = wsum.get_mut(h.frag as usize) {
                e.0 += h.obs_mz as f64 * h.inten as f64;
                e.1 += h.inten as f64;
            }
        }

        let mut chrom_rows: Vec<ChromOutputRow> = Vec::new();

        // Emit chromatograms. When emit_window_grid is set, each fragment is sampled
        // on the full isolation-window scan grid (all scans of the covering window(s)
        // within the RT window), with 0.0 where the fragment is absent, so the elution
        // profile drops to zero between peaks (correct boundary calling downstream).
        //
        // The traces are read straight off the scan groups. In grid mode `groups` IS the
        // grid (it was rebuilt on it above, one entry per grid RT in the same order), so a
        // fragment's grid-sampled trace is its value in each group, and the RT axis is the
        // grid itself and the same for every fragment and for the MS1 XICs below: one
        // conversion instead of one per fragment. That replaces a `HashMap<u16,
        // Vec<(f64, f32)>>` over the whole candidate plus a `HashMap<u64, f32>` per
        // fragment, and produces the same values in the same order.
        let mut observed = FragSet::default();
        for i in 0..groups.len() {
            for f in groups.frags(i) {
                observed.insert(f);
            }
        }
        let grid_rt: Vec<f32> = grid.iter().map(|&r| r as f32).collect();
        // Emit a row for EVERY predicted transition so the feature families see the
        // full predicted set (a missing strong ion is penalized). An OBSERVED
        // fragment carries its grid-sampled (or sorted) trace; a NEVER-OBSERVED one
        // carries an EMPTY trace, NOT a grid-length zero vector. The empty trace
        // still yields obs_apex = 0 downstream, and keeps the total chromatogram
        // list-value count down (a grid-length zero per absent fragment would
        // bloat it needlessly; the column itself is now a 64-bit LargeList, so the
        // old ~2.1B 32-bit offset ceiling no longer applies).
        // obs m/z falls back to theoretical; harmless since mass-accuracy counts
        // only fragments with obs_apex > 0.
        for fi in 0..fmzs.len() {
            let frag = fi as u16;
            let (sm, sw) = wsum[fi];
            let obs_mz = if sw > 0.0 { sm / sw } else { fmzs[fi] as f64 };
            let (rts, ints): (Vec<f32>, Vec<f32>) = if !observed.contains(frag) {
                (Vec::new(), Vec::new()) // absent predicted transition
            } else if !grid.is_empty() {
                (
                    grid_rt.clone(),
                    (0..groups.len()).map(|i| groups.or_zero(i, frag)).collect(),
                )
            } else {
                // Sparse mode: the groups are already ascending in RT, so the trace is
                // the fragment's entries read in group order -- what the per-fragment
                // vector held before its (already-sorted) stable sort.
                let mut rts: Vec<f32> = Vec::new();
                let mut ints: Vec<f32> = Vec::new();
                for i in 0..groups.len() {
                    if let Some(v) = groups.get(i, frag) {
                        rts.push(groups.rt(i) as f32);
                        ints.push(v);
                    }
                }
                (rts, ints)
            };
            chrom_rows.push((
                cid,
                // Fragment names are interned in the library (a u16 dictionary id per
                // fragment instead of a String); materialise the String only here, for
                // the emitted chromatogram row.
                lib.frag_name_str(fnames[fi]).to_string(),
                fmzs[fi] as f64,
                obs_mz,
                fints[fi],
                rts,
                ints,
            ));
        }

        // MS1 isotope XICs (mono/+1/+2) sampled on the same scan grid as the
        // fragments, so the features stage can correlate the MS1 precursor
        // envelope against the MS2 fragments over the elution peak (DIA-NN
        // Ms1.Profile.Corr class). Grid mode only; nearest MS1 scan per grid RT.
        if !ms1_scans.is_empty() && !grid.is_empty() {
            let sp = ISOTOPE_SPACING / c.charge as f64;
            let tol = p.cfg.prec_tol_ppm;
            for (nm, dmz) in [("ms1_mono", 0.0), ("ms1_iso1", sp), ("ms1_iso2", 2.0 * sp)] {
                let mz = c.precursor_mz + dmz;
                let ints: Vec<f32> = grid
                    .iter()
                    .map(|&r| {
                        let j = nearest_index(&ms1_rts, r);
                        // No `as f32`: `sum_near` returns f32 and the target is `Vec<f32>`,
                        // so the cast was an identity. It was tolerated while `ms1_scans`
                        // was a `Vec<Ms1Scan>` and `clippy::unnecessary_cast` fires on it
                        // now that it is a `&[Ms1Scan]` -- verified in both directions on
                        // this tree with clippy 1.96.0, silent before and an error after,
                        // though which of the lint's heuristics distinguishes `Vec`
                        // indexing from slice indexing was not established.
                        sum_near(&ms1_scans[j].mz, &ms1_scans[j].intensity, mz, tol)
                    })
                    .collect();
                chrom_rows.push((cid, nm.to_string(), mz, mz, 0.0, grid_rt.clone(), ints));
            }
        }

        // Top-K peak retention (opt-in; sensitivity_plan P1.1/P1.2). Enumerate peak
        // groups over the per-scan distinct-fragment COUNT profile (co-eluting
        // breadth, interference-resistant per the intensity-is-chimeric argument),
        // ranked by breadth-area. The PSM above still carries the selected apex, so
        // FDR is unchanged; these are extra candidate peaks for an offline peak-
        // selection model. Empty for K=1 (the default).
        let peaks: Vec<(u8, f64, f64, f64, f64, f64)> =
            if p.cfg.retain_top_peaks > 1 && !groups.is_empty() {
                let count_prof: Vec<f32> = counts.iter().map(|&c| c as f32).collect();
                crate::peaks::enumerate_peaks(&count_prof, p.cfg.retain_top_peaks, 1.0 / 3.0, 0.1)
                    .into_iter()
                    .map(|pk| {
                        (
                            pk.rank as u8,
                            groups.rt(pk.apex_idx),
                            groups.rt(pk.start_idx),
                            groups.rt(pk.end_idx),
                            counts[pk.apex_idx] as f64,
                            pk.area as f64,
                        )
                    })
                    .collect()
            } else {
                Vec::new()
            };

        // Spectrum-centric NNLS demixing at the selected apex (D2). Non-destructive:
        // emits interference-corrected features only. Gated; zero when off.
        //
        // Filled in a second pass below, not here: the whole problem is a function of
        // the apex SCAN, so solving it inside this per-candidate loop re-probed every
        // peak of that scan and re-ran the NNLS once per candidate sharing it.
        let (deconv_explained, deconv_active, deconv_share, deconv_collin, deconv_shadow) =
            (0.0, 0.0, 0.0, 0.0, 0.0);

        let rank0 = CandOut {
            cid,
            peak_rank: 0, // selected apex; ranks >= 1 added when promote_top_peaks > 1
            apex_rt,
            apex_int: apex_sum,
            n_match: n_distinct as i32,
            corun: best_run as i32,
            npred: fmzs0.len() as i32,
            calrt: rt_cal[cid as usize],
            mz: c.precursor_mz,
            contested: contested_val,
            contested_count_frac,
            apportioned_frac,
            z: c.charge,
            label: if c.is_decoy { "decoy" } else { "target" }.to_string(),
            base: c.base_peptide_id,
            pform: c.peptidoform.to_string(),
            prot: c.protein.to_string(),
            irt: c.predicted_irt,
            ms1_m1: o_ms1_m1,
            ms1_mono: o_ms1_mono,
            ms1_i1: o_ms1_i1,
            ms1_i2: o_ms1_i2,
            gate_apex: gate_apex as f32,
            gate_peak_spectral: gate_peak_spectral as f32,
            gate_coelution: gate_coelution as f32,
            gate_spectral_entropy: gate_spectral_entropy as f32,
            deconv_explained: deconv_explained as f32,
            deconv_active: deconv_active as f32,
            deconv_share: deconv_share as f32,
            deconv_collin: deconv_collin as f32,
            deconv_shadow: deconv_shadow as f32,
            chrom: chrom_rows,
            peaks,
        };
        if p.cfg.promote_top_peaks <= 1 || groups.is_empty() {
            return vec![rank0];
        }
        // Promote alternate chromatographic peaks (#7). Enumerate on the same
        // distinct-fragment COUNT profile as the diagnostic peaks (breadth of
        // co-elution, interference-resistant), exclude the envelope holding the
        // selected apex, gate each candidate peak by area fraction + apex-RT
        // separation + matched-fragment floor, and emit up to promote_top_peaks - 1
        // extra records. Each shares the candidate's chromatograms (emitted once on
        // rank 0, looked up by candidate_id downstream) and re-slices only its own
        // apex-dependent scalars + MS1; the features stage recomputes peak-shape and
        // co-elution features from the shared chrom windowed to each row's own apex.
        let count_prof: Vec<f32> = counts.iter().map(|&c| c as f32).collect();
        let alt_peaks =
            crate::peaks::enumerate_peaks(&count_prof, p.cfg.promote_top_peaks, 1.0 / 3.0, 0.1);
        // Reference area for the area gate: the enumerated envelope holding the
        // selected apex (0 disables the area gate if the apex is not a counted peak).
        let rank0_area = apex_gi
            .and_then(|gi| {
                alt_peaks
                    .iter()
                    .find(|pk| pk.start_idx <= gi && gi <= pk.end_idx)
            })
            .map(|pk| pk.area as f64)
            .unwrap_or(0.0);
        let mut out = vec![rank0];
        let mut rank: u8 = 1;
        for pk in &alt_peaks {
            if out.len() >= p.cfg.promote_top_peaks {
                break;
            }
            // Exclude the envelope containing the selected apex (rank 0).
            if let Some(gi) = apex_gi {
                if pk.start_idx <= gi && gi <= pk.end_idx {
                    continue;
                }
            }
            let alt_apex_rt = groups.rt(pk.apex_idx);
            if (alt_apex_rt - apex_rt).abs() < p.cfg.alt_peak_min_separation_s {
                continue;
            }
            if rank0_area > 0.0 && (pk.area as f64) < p.cfg.alt_peak_min_area_frac * rank0_area {
                continue;
            }
            // Distinct fragments across the alternate envelope: the size of the union of
            // its groups' fragment sets.
            let alt_distinct = groups.count_union(pk.start_idx, pk.end_idx);
            if alt_distinct < p.cfg.presence_min_matched.max(1) {
                continue;
            }
            let alt_apex_int: f32 = groups.sum(pk.apex_idx);
            let (a_m1, a_mono, a_i1, a_i2) = ms1_at(alt_apex_rt);
            out.push(CandOut {
                cid,
                peak_rank: rank,
                apex_rt: alt_apex_rt,
                apex_int: alt_apex_int,
                n_match: alt_distinct as i32,
                corun: best_run as i32,
                npred: fmzs0.len() as i32,
                calrt: rt_cal[cid as usize],
                mz: c.precursor_mz,
                contested: contested_val,
                contested_count_frac,
                apportioned_frac,
                z: c.charge,
                label: if c.is_decoy { "decoy" } else { "target" }.to_string(),
                base: c.base_peptide_id,
                pform: c.peptidoform.to_string(),
                prot: c.protein.to_string(),
                irt: c.predicted_irt,
                ms1_m1: a_m1,
                ms1_mono: a_mono,
                ms1_i1: a_i1,
                ms1_i2: a_i2,
                gate_apex: gate_apex as f32,
                gate_peak_spectral: gate_peak_spectral as f32,
                gate_coelution: gate_coelution as f32,
                gate_spectral_entropy: gate_spectral_entropy as f32,
                // Demix is a rank-0 apex feature; alternate peaks carry 0.
                deconv_explained: 0.0,
                deconv_active: 0.0,
                deconv_share: 0.0,
                deconv_collin: 0.0,
                deconv_shadow: 0.0,
                chrom: Vec::new(), // shared per-candidate via the rank-0 row
                peaks: Vec::new(),
            });
            rank += 1;
        }
        out
    };

    // Process candidates in cid-ordered chunks: each chunk is parallel over its candidates
    // and, once collected (order-preserving), appended serially -- PSM scalars to the column
    // Vecs, chromatogram rows to a parquet writer running on its own thread. Output order is
    // exactly the old "collect everything, then flatten" order, but a chunk's trace payload
    // is freed as soon as it is encoded, instead of every trace of the run (plus a second
    // copy in the Arrow builders) being resident at one final write. The bounded channel
    // keeps at most a few chunks in flight between the extraction threads and the encoder.
    let mut n_accepted = 0u64;
    // Candidates that had at least one hit; counted as they are flushed on the streamed
    // path, where no whole-run accumulator exists to size.
    let mut n_materialized = 0u64;
    // Trace payload accounting: what one chunk holds between extraction and encoding is
    // the whole point of the incremental writer, so record the largest chunk alongside
    // the run total that the old "accumulate everything, then write" path held at once.
    let mut chrom_bytes_total = 0usize;
    let mut chrom_bytes_max_chunk = 0usize;
    // Top-K retained peaks (opt-in; empty for K=1).
    let (mut pk_cid, mut pk_rank): (Vec<u32>, Vec<i32>) = (Vec::new(), Vec::new());
    let (mut pk_apex, mut pk_start, mut pk_end): (Vec<f64>, Vec<f64>, Vec<f64>) =
        (Vec::new(), Vec::new(), Vec::new());
    let (mut pk_ev, mut pk_area): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    // The chromatogram writer runs on its own thread and cannot borrow the library, so the
    // band's offset travels with the chunks.
    let chrom_offset = lib.global_offset;
    // Writer-side timing (P0 instrumentation, log lines only): how long the extraction
    // side sat blocked in `tx.send` because both channel slots were full, against how long
    // the writer thread spent encoding. A large blocked time with a writer busy for most of
    // the stage means the serial encoder bounds extract; a small one means it does not.
    let mut chrom_send_blocked = std::time::Duration::ZERO;
    let mut chrom_chunks_sent = 0u64;
    let chrom_written = std::thread::scope(|sc| -> Result<Written> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<Col>>(2);
        // The writer is handed back unclosed: whether the table is published depends on
        // how the loop ended (`close_chromatograms`), which only this side knows.
        let writer = sc.spawn(move || -> Result<(TableWriter, std::time::Duration)> {
            let mut w = chrom_writer;
            let mut busy = std::time::Duration::ZERO;
            for cols in rx {
                let t = Instant::now();
                w.write_cols(cols)?;
                busy += t.elapsed();
            }
            Ok((w, busy))
        });
        // One flush of finished candidates: score them in parallel, append their PSM
        // rows, and hand their chromatogram rows to the writer thread. Called once per
        // window batch on the streamed path and once on the eager paths, so the code
        // that produces a row is the same either way. Returns false when the writer has
        // gone away; its error surfaces at the join below.
        let mut emit_batch = |mut cand_hits: Vec<(u32, &mut [Hit])>| -> bool {
            for chunk in cand_hits.chunks_mut(CAND_CHUNK) {
                let outs: Vec<Vec<CandOut>> = chunk
                    .par_iter_mut()
                    .map(|(cid, hits)| per_candidate(*cid, hits))
                    .collect();
                let mut ch = ChromChunk::default();
                for mut r in outs.into_iter().flatten() {
                    n_accepted += 1;
                    let rcid = r.cid;
                    for (rank, apex, start, end, ev, area) in &r.peaks {
                        pk_cid.push(rcid);
                        pk_rank.push(*rank as i32);
                        pk_apex.push(*apex);
                        pk_start.push(*start);
                        pk_end.push(*end);
                        pk_ev.push(*ev);
                        pk_area.push(*area);
                    }
                    let chrom = std::mem::take(&mut r.chrom);
                    if let Err(e) = psms.push(r, p.cfg) {
                        psms_err = Some(e);
                        return false;
                    }
                    let rows = &mut ch.rows;
                    for (cc, nm, fmz, omz, pint, rt, it) in chrom {
                        rows.cid.push(cc);
                        rows.name.push(nm);
                        rows.frag_mz.push(fmz);
                        rows.frag_obs_mz.push(omz);
                        rows.predicted_intensity.push(pint);
                        if chrom_layout == crate::chromatograms::Layout::V2 {
                            // The band-local id: the offset is added to every row alike,
                            // so it cannot change which rows share a candidate.
                            match chrom_encoder.encode(cc, rt, it) {
                                Ok(e) => {
                                    rows.rt.push(e.rt);
                                    rows.intensity.push(e.intensity);
                                    rows.trace_offset.push(e.trace_offset);
                                    rows.trace_len.push(e.trace_len);
                                }
                                Err(e) => {
                                    psms_err = Some(e);
                                    return false;
                                }
                            }
                        } else {
                            rows.rt.push(rt);
                            rows.intensity.push(it);
                        }
                    }
                }
                let r = &ch.rows;
                let chunk_bytes = r.trace_bytes()
                    + crate::memlog::bytes_of(&r.cid)
                    + crate::memlog::bytes_of(&r.frag_mz)
                    + crate::memlog::bytes_of(&r.frag_obs_mz)
                    + crate::memlog::bytes_of(&r.predicted_intensity)
                    + crate::memlog::bytes_of(&r.trace_offset)
                    + crate::memlog::bytes_of(&r.trace_len)
                    + r.name.iter().map(|s| s.len()).sum::<usize>();
                chrom_bytes_total += chunk_bytes;
                chrom_bytes_max_chunk = chrom_bytes_max_chunk.max(chunk_bytes);
                // Hand the chunk's chromatogram rows to the writer thread. A send error means
                // the writer failed; its error surfaces at the join below. The columns are
                // built before the clock starts, so the timer holds only the wait for a
                // free channel slot.
                let cols = ch.cols(chrom_offset, chrom_layout);
                let t_send = Instant::now();
                let sent = tx.send(cols);
                chrom_send_blocked += t_send.elapsed();
                chrom_chunks_sent += 1;
                if sent.is_err() {
                    return false;
                }
            }
            true
        };

        // Streamed path: probe a batch of windows, then flush everything the next batch
        // cannot touch. `acc_stream` therefore holds the hits of the windows in flight
        // rather than the hits of the run. `chunk` is the reusable gather buffer the
        // candidates are flushed through; it grows to one chunk's hits and is then reused,
        // so the flat accumulator is never copied whole.
        let mut acc_stream = HitAcc::default();
        let mut chunk = HitStore::default();
        let mut open_hits_max = 0usize;
        if let Some(groups) = stream_groups.as_ref() {
            // Counting the flushed candidates here rather than inside `emit_batch` keeps
            // the eager path below able to borrow `emit_batch` on its own.
            let mut flush = |c: Vec<(u32, &mut [Hit])>| -> bool {
                n_materialized += c.len() as u64;
                emit_batch(c)
            };
            let step = groups_in_flight(p.cfg);
            let mut gi = 0usize;
            while gi < groups.len() {
                let upto = (gi + step).min(groups.len());
                // Everything below the next batch's first candidate is final: windows are
                // processed in ascending m/z and precursors are sorted by m/z, so no later
                // window can add a hit below `bound`. Within the batch, `accumulate_groups`
                // flushes each candidate sub-range as soon as every window has passed it,
                // so the accumulator holds a sub-range of the band rather than the band.
                let bound = groups.get(upto).map(|g| g.lo_cid).unwrap_or(u32::MAX);
                let mut stopped = accumulate_groups(
                    TaskProbe::Local {
                        lib: &lib,
                        bins: stream_bins
                            .as_ref()
                            .expect("the streamed path implies the fragindex geometry"),
                        tol_ppm: frag_tol,
                        stats: &local_stats,
                    },
                    p.sibling_bands,
                    &groups[gi..upto],
                    scans,
                    &rt_lo,
                    &rt_hi,
                    &mass_off,
                    p.cfg,
                    restrict.as_ref(),
                    bound,
                    &mut acc_stream,
                    &mut chunk,
                    &mut flush,
                );
                gi = upto;
                open_hits_max = open_hits_max.max(acc_stream.n_hits());
                // Candidates past the batch's own sub-range grid (a window of this batch
                // can reach past the last window's range when the windows differ in width)
                // are final too, and the sub-range loop cannot have reached them.
                if !stopped
                    && flush_below(
                        &mut acc_stream.runs,
                        bound,
                        CAND_CHUNK,
                        true,
                        &mut chunk,
                        &mut flush,
                    )
                {
                    stopped = true;
                }
                acc_stream.compact();
                if stopped {
                    break;
                }
            }
            crate::memlog::report(
                "extract hit accumulator (streamed)",
                &[(
                    "largest_open_payload",
                    open_hits_max * std::mem::size_of::<Hit>(),
                )],
            );
            info!(
                materialized = n_materialized,
                windows = groups.len(),
                windows_in_flight = step,
                "extract: candidates with evidence"
            );
            {
                use std::sync::atomic::Ordering::Relaxed;
                info!(
                    tasks = local_stats.tasks.load(Relaxed),
                    postings = local_stats.postings.load(Relaxed),
                    library_fragments = lib.frag_mz.len(),
                    largest_task_index_bytes = local_stats.largest_bytes.load(Relaxed),
                    bins = stream_bins.as_ref().map(|b| b.n_bins).unwrap_or(0),
                    "extract: task-local fragment indexes"
                );
            }
        } else {
            // Eager paths: move each chunk of candidates out of the whole-run
            // `HashMap<u32, Vec<Hit>>` into the CSR buffer in ascending id order, freeing
            // the map's per-candidate vectors as they are consumed, so the flat copy never
            // coexists with the whole map.
            let mut i = 0usize;
            while i < cand_ids.len() {
                let end = (i + CAND_CHUNK).min(cand_ids.len());
                chunk.clear();
                for &cid in &cand_ids[i..end] {
                    let hits = acc.remove(&cid).expect("id came from the map");
                    chunk.push_segment(cid, &hits);
                }
                i = end;
                if !emit_batch(chunk.slices_mut()) {
                    break;
                }
            }
        }
        // A final empty chunk fixes the schema when no candidate was accepted at all. Not
        // after a failed psms_extracted write, whose table is abandoned below.
        if psms_err.is_none() {
            let _ = tx.send(ChromChunk::default().cols(0, chrom_layout));
        }
        drop(tx);
        let (w, mut writer_busy) = writer
            .join()
            .map_err(|_| anyhow::anyhow!("chromatogram writer thread panicked"))??;
        let t_close = Instant::now();
        let n = close_chromatograms(w, psms_err.take())?;
        writer_busy += t_close.elapsed();
        crate::memlog::report(
            "extract chromatogram traces",
            &[
                ("largest_chunk_in_flight", chrom_bytes_max_chunk),
                ("run_total_streamed", chrom_bytes_total),
            ],
        );
        info!(
            chunks = chrom_chunks_sent,
            send_blocked_ms = chrom_send_blocked.as_millis() as u64,
            writer_busy_ms = writer_busy.as_millis() as u64,
            "extract: chromatogram writer"
        );
        Ok(n)
    })?;

    // Spectrum-centric NNLS demixing (D2), second pass: solve ONCE PER APEX SCAN.
    //
    // The design matrix, the observed vector and the NNLS solution depend only on the apex
    // scan; the candidate id merely selects a column. Solving inside the per-candidate loop
    // above therefore re-probed every peak of a scan, and re-ran the NNLS, once for every
    // candidate that apexed in it -- on a wide-window run that is a second full pass over
    // the spectra. Grouping by resolved scan index collapses it to one solve per scan.
    //
    // Exactly the same numbers as the per-candidate version: `demix_solve_scan` reproduces
    // the assembly verbatim, `demix_features_for` reproduces the per-candidate reads, and
    // the group key is the same scan the old code resolved from `(apex_rt, precursor_mz)`.
    // Rows are patched by candidate id in the output columns, so row order is untouched.
    if p.cfg.emit_demix_features {
        // Which candidates need a demix, grouped by the scan that serves them. BTreeMap so
        // the scan iteration order is deterministic.
        let mut by_scan: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for i in 0..psms.rows.len() {
            if psms.rows.peak_rank[i] != 0 {
                continue;
            }
            if let Some(si) =
                demix_apex_scan(scans, &rt_scan, psms.rows.apex_rt[i], psms.rows.mz[i])
            {
                by_scan.entry(si).or_default().push(psms.rows.cid[i]);
            }
        }
        let n_scans = by_scan.len();
        let n_cands: usize = by_scan.values().map(|v| v.len()).sum();
        let scan_jobs: Vec<(usize, Vec<u32>)> = by_scan.into_iter().collect();
        let solved: Vec<Vec<(u32, DemixFeatures)>> = scan_jobs
            .par_iter()
            .map(|(si, cids)| {
                let scan = &scans[*si];
                match demix_solve_scan(
                    fidx.as_ref(),
                    &lib,
                    scan,
                    &mass_off,
                    frag_tol,
                    scan.rt_seconds,
                    &rt_lo,
                    &rt_hi,
                    p.cfg,
                ) {
                    Some(d) => cids
                        .iter()
                        .map(|&cid| (cid, demix_features_for(&d, cid)))
                        .collect(),
                    None => Vec::new(),
                }
            })
            .collect();
        let feats: HashMap<u32, (f64, f64, f64, f64, f64)> = solved.into_iter().flatten().collect();
        info!(
            scans_solved = n_scans,
            candidates = n_cands,
            "extract: demix features (one NNLS per apex scan)"
        );
        // Patch the rank-0 rows in place; the columns were filled in the chunk loop above.
        for i in 0..psms.rows.len() {
            if psms.rows.peak_rank[i] != 0 {
                continue;
            }
            if let Some(&(expl, act, share, collin, shadow)) = feats.get(&psms.rows.cid[i]) {
                psms.rows.deconv_expl[i] = expl as f32;
                psms.rows.deconv_act[i] = act as f32;
                psms.rows.deconv_share[i] = share as f32;
                psms.rows.deconv_collin[i] = collin as f32;
                psms.rows.deconv_shadow[i] = shadow as f32;
            }
        }
    }

    // The last, short chunk (or the one empty chunk that fixes the schema of an empty
    // table), then the footer. Unstreamed, the whole table goes through `write_table`,
    // which cuts the same chunks.
    let (psms_written, psms_write_busy) = psms.finish(p.cfg)?;
    info!(
        rows = psms_written.rows,
        streamed = stream_psms,
        writer_busy_ms = psms_write_busy.as_millis() as u64,
        "extract: psms_extracted writer"
    );

    // (chromatograms were streamed to `p.out_chrom` during the candidate loop above)

    // Top-K retained peaks (opt-in, sensitivity_plan P1.1/P1.2). Written next to
    // the psms table only when retain_top_peaks > 1; one row per (candidate, peak).
    if !pk_cid.is_empty() {
        let pk_path = format!("{}.peaks.parquet", p.out_psms);
        let n_peaks = write_table(
            &pk_path,
            vec![
                Col::U32(
                    "candidate_id".into(),
                    pk_cid.iter().map(|c| c + lib.global_offset).collect(),
                ),
                Col::I32("peak_rank".into(), pk_rank),
                Col::F64("apex_rt".into(), pk_apex),
                Col::F64("start_rt".into(), pk_start),
                Col::F64("end_rt".into(), pk_end),
                Col::F64("evidence_count".into(), pk_ev),
                Col::F64("area".into(), pk_area),
            ],
        )?;
        info!(peaks = n_peaks, path = %pk_path, "extract: wrote top-K peak table");
    }

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("accepted".to_string(), json!(n_accepted));
    stats.insert("scan_window".to_string(), json!(scan_window));
    let n_chrom = chrom_written.rows;
    let mut written: Vec<Written> = Vec::with_capacity(2);
    for (path, schema, file) in [
        (p.out_psms, artifact::PSMS_EXTRACTED, psms_written),
        (
            p.out_chrom,
            artifact::chromatograms(chrom_layout.version()),
            chrom_written,
        ),
    ] {
        let report = ArtifactReport {
            logical_name: schema.0.to_string(),
            schema_name: schema.0.to_string(),
            schema_version: schema.1,
            stage: "extract".to_string(),
            rows: file.rows,
            // Both files were hashed while they were written.
            content_hash: file.content_hash,
            params: json!({
                "frag_tol_ppm": p.cfg.frag_tol_ppm,
                "effective_frag_tol_ppm": frag_tol,
                "frag_ppm_offset": frag_offset,
                "presence_min_fragments": p.cfg.presence_min_fragments,
                "presence_min_coelution": p.cfg.presence_min_coelution,
                "gate_min_score": p.cfg.gate_min_score,
                "gate_mode": p.cfg.gate_mode,
                "gate_coelution_min": p.cfg.gate_coelution_min,
                "scan_window": scan_window,
            }),
            stats: stats.clone(),
            model_identity: None,
            elapsed_ms: elapsed,
        };
        report.write_for(path)?;
        written.push(report.written());
    }

    info!(
        accepted = n_accepted,
        chromatograms = n_chrom,
        elapsed_ms = elapsed,
        "extract: done"
    );
    let chrom = written.pop().expect("two reports written");
    let psms = written.pop().expect("two reports written");
    Ok((psms, chrom))
}

#[cfg(test)]
mod cand_mask_tests {
    use super::CandMask;

    /// The allowlist bitset must answer exactly what the `HashSet<u32>` answered for every
    /// id the probe can produce, which is every id in `0..ncand`.
    #[test]
    fn answers_the_same_as_the_hash_set() {
        let n = 1000usize;
        let ids: Vec<u32> = vec![0, 1, 63, 64, 65, 127, 128, 500, 999, 500];
        let reference: std::collections::HashSet<u32> = ids.iter().copied().collect();
        let mut mask = CandMask::new(n);
        for &c in &ids {
            mask.insert(c);
        }
        assert_eq!(
            mask.len(),
            reference.len(),
            "duplicates must not be counted"
        );
        for c in 0..n as u32 {
            assert_eq!(mask.contains(c), reference.contains(&c), "id {c}");
        }
    }

    /// An id past the library's candidate count cannot be asked about by the probe, so it
    /// must not panic and must not be stored.
    #[test]
    fn ids_outside_the_library_are_absent() {
        let mut mask = CandMask::new(64);
        mask.insert(5);
        mask.insert(64);
        mask.insert(u32::MAX);
        assert_eq!(mask.len(), 1);
        assert!(mask.contains(5));
        assert!(!mask.contains(64) && !mask.contains(u32::MAX));
    }

    /// An empty library gives an empty mask rather than an out-of-bounds write.
    #[test]
    fn an_empty_library_gives_an_empty_mask() {
        let mut mask = CandMask::new(0);
        mask.insert(0);
        assert_eq!(mask.len(), 0);
        assert!(!mask.contains(0));
    }
}

#[cfg(test)]
mod frag_set_tests {
    use super::{FragSet, FRAG_SET_BITS};

    /// The bitmask replaces "collect every hit's ordinal, sort, dedup", so it has to
    /// produce that vector, and its length, for any input.
    #[test]
    fn reproduces_sort_and_dedup() {
        for &case in &[
            &[][..],
            &[0][..],
            &[7, 7, 7][..],
            &[5, 1, 9, 1, 63, 64, 0, 9][..],
            &[511, 510, 0, 1, 64, 65, 128, FRAG_SET_BITS as u16 - 1][..],
        ] {
            let mut set = FragSet::default();
            for &f in case {
                set.insert(f);
            }
            let mut reference: Vec<u16> = case.to_vec();
            reference.sort_unstable();
            reference.dedup();
            assert_eq!(set.len(), reference.len(), "{case:?}");
            assert_eq!(set.to_vec(), reference, "{case:?}");
            for f in 0..600u16 {
                assert_eq!(set.contains(f), reference.contains(&f), "{case:?} frag {f}");
            }
        }
    }

    /// Ordinals past the inline mask spill to a sorted vector rather than panicking, and
    /// the two halves still read back as one ascending set.
    #[test]
    fn ordinals_past_the_mask_spill_in_order() {
        let case: Vec<u16> = vec![900, 3, 512, 900, 1200, 511, 512];
        let mut set = FragSet::default();
        for &f in &case {
            set.insert(f);
        }
        let mut reference = case.clone();
        reference.sort_unstable();
        reference.dedup();
        assert_eq!(set.to_vec(), reference);
        assert_eq!(set.len(), reference.len());
        assert!(set.contains(1200) && set.contains(3) && !set.contains(4));
    }
}

#[cfg(test)]
mod covering_window_tests {
    /// The per-candidate grid used to scan every isolation window. The binary-searched
    /// span must select exactly the windows the scan did, for uneven widths and for
    /// precursors falling in a gap, or a candidate silently loses part of its scan grid.
    #[test]
    fn the_searched_span_holds_every_covering_window() {
        // Deliberately uneven: overlapping wide and narrow windows, plus a gap.
        let mut windows: Vec<(f64, f64)> = vec![
            (400.0, 404.0),
            (402.0, 426.0),
            (404.0, 408.0),
            (408.0, 412.0),
            (412.0, 413.0),
            (420.0, 430.0),
            (430.0, 470.0),
            (460.0, 461.0),
        ];
        windows.sort_by(|a, b| a.0.total_cmp(&b.0));
        let max_width = windows
            .iter()
            .map(|w| w.1 - w.0)
            .fold(0.0f64, |a, b| if b > a { b } else { a });
        let mut pms: Vec<f64> = Vec::new();
        for i in 0..2000 {
            pms.push(395.0 + i as f64 * 0.05);
        }
        for &w in &windows {
            pms.push(w.0);
            pms.push(w.1);
        }
        for pm in pms {
            let full: Vec<usize> = (0..windows.len())
                .filter(|&i| windows[i].0 <= pm && pm <= windows[i].1)
                .collect();
            let s = windows.partition_point(|w| w.0 < pm - max_width);
            let e = windows.partition_point(|w| w.0 <= pm);
            let span: Vec<usize> = (s..e)
                .filter(|&i| windows[i].0 <= pm && pm <= windows[i].1)
                .collect();
            assert_eq!(span, full, "pm {pm}");
        }
    }
}

#[cfg(test)]
mod mass_offset_tests {
    use super::MassOffset;

    #[test]
    fn scalar_and_grid_interpolation() {
        // Scalar offset: constant factor regardless of m/z.
        let s = MassOffset {
            scalar_ppm: 5.0,
            grid_mz: vec![],
            grid_ppm: vec![],
        };
        assert!((s.factor_at(500.0) - (1.0 + 5e-6)).abs() < 1e-12);
        // Grid: linear interpolation between points, clamped past the ends.
        let g = MassOffset {
            scalar_ppm: 0.0,
            grid_mz: vec![200.0, 400.0, 600.0],
            grid_ppm: vec![2.0, 4.0, 0.0],
        };
        assert!((g.factor_at(300.0) - (1.0 + 3e-6)).abs() < 1e-12); // 200->400: 2->4, mid = 3
        assert!((g.factor_at(400.0) - (1.0 + 4e-6)).abs() < 1e-12); // exact grid point
        assert!((g.factor_at(100.0) - (1.0 + 2e-6)).abs() < 1e-12); // clamp low
        assert!((g.factor_at(900.0) - 1.0).abs() < 1e-12); // clamp high (ppm 0)
    }
}

#[cfg(test)]
mod scan_group_tests {
    use super::ScanGroups;
    use std::collections::BTreeMap;

    type TreeGroups = Vec<(f64, BTreeMap<u16, f32>)>;

    /// The per-group trees `ScanGroups` replaced, built verbatim from `(rt, frag, inten)`
    /// hits in the order the per-candidate pass builds them.
    fn tree_groups(hits: &[(f64, u16, f32)]) -> TreeGroups {
        let mut groups: TreeGroups = Vec::new();
        for &(h_rt, frag, inten) in hits {
            match groups.last_mut() {
                Some((rt, map)) if (*rt - h_rt).abs() < 1e-9 => {
                    let e = map.entry(frag).or_insert(0.0);
                    if inten > *e {
                        *e = inten;
                    }
                }
                _ => {
                    let mut m = BTreeMap::new();
                    m.insert(frag, inten);
                    groups.push((h_rt, m));
                }
            }
        }
        groups
    }

    fn dense_groups(hits: &[(f64, u16, f32)]) -> ScanGroups {
        let width = hits.iter().map(|h| h.1 as usize + 1).max().unwrap_or(0);
        let mut groups = ScanGroups::new(width);
        for &(h_rt, frag, inten) in hits {
            match groups.rt.last() {
                Some(&rt) if (rt - h_rt).abs() < 1e-9 => {
                    let i = groups.len() - 1;
                    groups.merge_max(i, frag, inten);
                }
                _ => {
                    groups.push_group(h_rt);
                    let i = groups.len() - 1;
                    groups.insert(i, frag, inten);
                }
            }
        }
        groups
    }

    /// Every question the per-candidate pass asks of a scan group, asked of both forms.
    fn assert_same(tree: &TreeGroups, dense: &ScanGroups, what: &str) {
        assert_eq!(tree.len(), dense.len(), "{what}: group count");
        for (i, (rt, map)) in tree.iter().enumerate() {
            assert_eq!(rt.to_bits(), dense.rt(i).to_bits(), "{what}: rt {i}");
            assert_eq!(map.len(), dense.count(i), "{what}: count {i}");
            let keys: Vec<u16> = map.keys().copied().collect();
            assert_eq!(
                keys,
                dense.frags(i).collect::<Vec<_>>(),
                "{what}: frags {i}"
            );
            let tree_sum: f32 = map.values().sum();
            assert_eq!(
                tree_sum.to_bits(),
                dense.sum(i).to_bits(),
                "{what}: sum {i}"
            );
            for f in 0..(dense.width as u16 + 70) {
                assert_eq!(
                    map.get(&f).map(|v| v.to_bits()),
                    dense.get(i, f).map(f32::to_bits),
                    "{what}: group {i} frag {f}"
                );
            }
        }
    }

    /// Randomised hit lists with repeated RTs, RTs within 1e-9 s of each other (which the
    /// grouping merges), repeated fragments in a group, and intensities that are zero,
    /// negative, negative zero and NaN: the cases where "first hit inserts, later hits take
    /// the max starting from 0.0" differs from "take the max".
    #[test]
    fn dense_groups_answer_what_the_trees_answered() {
        let mut state = 0x5ca9_u64;
        let mut next = move |m: u64| -> u64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m.max(1)
        };
        let specials = [0.0f32, -0.0, -3.5, f32::NAN, 1e-40, 7.25];
        for case in 0..300 {
            let width = 1 + next(if case % 5 == 0 { 140 } else { 14 }) as u16;
            let mut rt = 100.0f64;
            let mut hits: Vec<(f64, u16, f32)> = Vec::new();
            for _ in 0..next(80) {
                match next(4) {
                    0 => rt += 1.0 + next(3) as f64,
                    1 => rt += 1e-10, // same group
                    _ => {}
                }
                let inten = if next(6) == 0 {
                    specials[next(specials.len() as u64) as usize]
                } else {
                    next(1000) as f32 * 0.5
                };
                hits.push((rt, next(width as u64) as u16, inten));
            }
            let tree = tree_groups(&hits);
            let dense = dense_groups(&hits);
            assert_same(&tree, &dense, &format!("case {case}"));

            // The grid projection: a grid holding some of the group RTs and some others.
            let mut grid: Vec<f64> = tree
                .iter()
                .map(|(r, _)| *r)
                .filter(|_| next(3) != 0)
                .collect();
            for _ in 0..next(5) {
                grid.push(95.0 + next(200) as f64 * 0.5);
            }
            grid.sort_by(|a, b| a.total_cmp(b));
            grid.dedup();
            let mut aligned: TreeGroups = grid.iter().map(|&r| (r, BTreeMap::new())).collect();
            let mut j = 0usize;
            for (r, map) in tree.clone() {
                while j < grid.len() && grid[j] < r {
                    j += 1;
                }
                if j < grid.len() && grid[j].to_bits() == r.to_bits() {
                    aligned[j].1 = map;
                    j += 1;
                }
            }
            let mut dense_aligned = ScanGroups::empty_on(&grid, dense.width);
            let mut j = 0usize;
            for i in 0..dense.len() {
                let r = dense.rt(i);
                while j < grid.len() && grid[j] < r {
                    j += 1;
                }
                if j < grid.len() && grid[j].to_bits() == r.to_bits() {
                    dense_aligned.copy_group(j, &dense, i);
                    j += 1;
                }
            }
            assert_same(
                &aligned,
                &dense_aligned,
                &format!("case {case} on the grid"),
            );
            // The union count the promoted-peak gate takes over an envelope.
            if !aligned.is_empty() {
                let lo = next(aligned.len() as u64) as usize;
                let hi = lo + next((aligned.len() - lo) as u64) as usize;
                let mut set = std::collections::HashSet::new();
                for (_, m) in &aligned[lo..=hi] {
                    set.extend(m.keys().copied());
                }
                assert_eq!(set.len(), dense_aligned.count_union(lo, hi), "case {case}");
            }
        }
    }
}

#[cfg(test)]
mod coelution_tests {
    use super::{coelution_gate_score, peak_spectral_score, ScanGroups};

    fn g(rows: &[(f64, &[(u16, f32)])]) -> ScanGroups {
        ScanGroups::from_rows(rows)
    }

    #[test]
    fn coeluting_fragments_score_high() {
        // frags 0,1,2 all peak together at group index 2
        let groups = g(&[
            (0.0, &[(0, 1.0), (1, 1.0), (2, 1.0)]),
            (1.0, &[(0, 4.0), (1, 3.0), (2, 2.0)]),
            (2.0, &[(0, 9.0), (1, 8.0), (2, 5.0)]),
            (3.0, &[(0, 4.0), (1, 3.0), (2, 2.0)]),
            (4.0, &[(0, 1.0), (1, 1.0), (2, 1.0)]),
        ]);
        let s = coelution_gate_score(&groups, &[0, 1, 2], &[0, 1], &[10.0, 8.0, 5.0]);
        assert!(s > 0.95, "co-eluting fragments should score high, got {s}");
    }

    #[test]
    fn non_coeluting_interferent_drops_the_score() {
        // frags 0,1 co-elute; frag 2 (a strong-predicted interferent) sits off-peak
        let groups = g(&[
            (0.0, &[(0, 1.0), (1, 1.0), (2, 9.0)]),
            (1.0, &[(0, 4.0), (1, 3.0), (2, 0.0)]),
            (2.0, &[(0, 9.0), (1, 8.0), (2, 0.0)]),
            (3.0, &[(0, 4.0), (1, 3.0), (2, 0.0)]),
            (4.0, &[(0, 1.0), (1, 1.0), (2, 0.0)]),
        ]);
        let s = coelution_gate_score(&groups, &[0, 1, 2], &[0, 1], &[10.0, 8.0, 9.0]);
        assert!(
            s < 0.8,
            "a strong non-co-eluting interferent should lower the score, got {s}"
        );
    }

    #[test]
    fn too_few_scans_does_not_reject() {
        let groups = g(&[(0.0, &[(0, 5.0)]), (1.0, &[(0, 9.0)])]);
        assert_eq!(coelution_gate_score(&groups, &[0], &[0], &[10.0]), 1.0);
    }

    #[test]
    fn peak_spectral_high_when_integrated_pattern_matches() {
        // observed peak-summed spectrum (9:8:5 at apex, tails scale) matches predicted
        let groups = g(&[
            (0.0, &[(0, 1.0), (1, 1.0), (2, 1.0)]),
            (1.0, &[(0, 4.0), (1, 3.0), (2, 2.0)]),
            (2.0, &[(0, 9.0), (1, 8.0), (2, 5.0)]),
            (3.0, &[(0, 4.0), (1, 3.0), (2, 2.0)]),
            (4.0, &[(0, 1.0), (1, 1.0), (2, 1.0)]),
        ]);
        let s = peak_spectral_score(&groups, &[0, 1], &[19.0, 16.0, 11.0]);
        assert!(s > 0.99, "integrated pattern matches predicted, got {s}");
    }

    #[test]
    fn peak_spectral_recovers_fragment_absent_at_apex_scan() {
        // A real strong-predicted fragment (2) is momentarily unsampled at the apex
        // scan (DIA scan gap) but present across the rest of the peak. The single-scan
        // apex Pearson would see obs=0 for it and collapse; integrating over the peak
        // recovers its true contribution and matches the predicted 19:16:16.
        let groups = g(&[
            (0.0, &[(0, 1.0), (1, 1.0), (2, 2.0)]),
            (1.0, &[(0, 4.0), (1, 3.0), (2, 6.0)]),
            (2.0, &[(0, 9.0), (1, 8.0), (2, 0.0)]), // frag 2 unsampled at apex scan
            (3.0, &[(0, 4.0), (1, 3.0), (2, 6.0)]),
            (4.0, &[(0, 1.0), (1, 1.0), (2, 2.0)]),
        ]);
        let s = peak_spectral_score(&groups, &[0, 1], &[19.0, 16.0, 16.0]);
        assert!(
            s > 0.99,
            "peak integration should recover the off-apex fragment, got {s}"
        );
    }
}

#[cfg(test)]
mod accumulate_tests {
    use super::*;
    use crate::index::{Candidate, Library};
    use mumdia_core::types::{IsolationWindow, Ms2Scan, Peak};

    /// `n` candidates in one isolation window, each with two fragments, m/z ascending.
    fn lib(n: usize) -> Library {
        let mut frag_mz = Vec::with_capacity(2 * n);
        let mut frag_int = Vec::with_capacity(2 * n);
        let mut frag_name_id = Vec::with_capacity(2 * n);
        let mut cands = Vec::with_capacity(n);
        for i in 0..n {
            let start = frag_mz.len();
            // Fragments are shared across candidates on purpose (300.0 + i % 97 * 1.37),
            // so a peak has many claimants and the sub-ranges all see work.
            for k in 0..2 {
                frag_mz.push((300.0 + ((i + 13 * k) % 97) as f64 * 1.37) as f32);
                frag_int.push(0.5 + 0.1 * k as f32);
                frag_name_id.push(k as u16);
            }
            let pmz = 400.0 + i as f64 * 1e-3;
            cands.push(Candidate {
                candidate_id: i as u32,
                peptidoform_id: i as u32,
                base_peptide_id: i as u32,
                peptidoform: String::new(),
                charge: 2,
                precursor_mz: pmz,
                predicted_irt: 0.0,
                is_decoy: false,
                protein: String::new(),
                frag_start: start,
                n_frag: 2,
            });
        }
        Library::from_candidates(
            cands,
            frag_mz,
            frag_int,
            frag_name_id,
            vec!["b".to_string(), "y".to_string()],
        )
    }

    fn scans(n_scans: usize, window: IsolationWindow, base: usize) -> Vec<Ms2Scan> {
        (0..n_scans)
            .map(|si| Ms2Scan {
                scan_index: (base + si) as u32,
                rt_seconds: 100.0 + (base + si) as f64,
                window,
                peaks: (0..97)
                    .map(|k| Peak {
                        mz: 300.0 + k as f32 * 1.37,
                        intensity: 10.0 + (si + k) as f32,
                    })
                    .collect(),
            })
            .collect()
    }

    /// The candidate-range split inside a window is a partition, so the accumulator it
    /// produces must not depend on how many sub-ranges ran: same candidates, same hits, and
    /// the same order within each candidate, which is what keeps every downstream float
    /// reduction identical.
    ///
    /// The number of sub-ranges per window is `2 * threads / windows`, capped by the band's
    /// candidate count, so four windows on a two-thread pool probe each window whole (the
    /// pre-change behaviour) and on a sixteen-thread pool split each into three. Note the
    /// pool must have at least two threads here: `accumulate_groups` spawns its producer
    /// into the pool and consumes on the calling thread, and under `install` that calling
    /// thread is one of the pool's own. The engine calls it from the main thread, which is
    /// never a pool worker (`build_global`), so a one-thread engine is unaffected.
    #[test]
    fn candidate_range_split_reproduces_the_unsplit_accumulation() {
        let per_window = 3 * MIN_CANDIDATES_PER_TASK;
        let n = 4 * per_window;
        let lib = lib(n);
        let idx = FragIndex::build(&lib, 20.0);
        let windows: Vec<IsolationWindow> = (0..4)
            .map(|w| {
                let lo = 400.0 + (w * per_window) as f64 * 1e-3;
                IsolationWindow {
                    target_mz: lo,
                    lower_mz: lo - 1e-9,
                    upper_mz: lo + (per_window - 1) as f64 * 1e-3 + 1e-9,
                    im_lower: None,
                    im_upper: None,
                }
            })
            .collect();
        let mut sc: Vec<Ms2Scan> = Vec::new();
        for w in &windows {
            sc.extend(scans(3, *w, sc.len()));
        }
        let groups: Vec<WinGroup> = windows
            .iter()
            .enumerate()
            .map(|(wi, w)| {
                let (lo, hi) = idx.candidate_range(w.lower_mz, w.upper_mz);
                assert_eq!(
                    (hi - lo) as usize,
                    per_window,
                    "window {wi} must select its own candidates only"
                );
                WinGroup {
                    lo_cid: lo,
                    hi_cid: hi,
                    scans: (wi * 3..wi * 3 + 3).collect(),
                }
            })
            .collect();
        let rt_lo = vec![0.0; n];
        let rt_hi = vec![1e9; n];
        let mass_off = MassOffset {
            scalar_ppm: 0.0,
            grid_mz: Vec::new(),
            grid_ppm: Vec::new(),
        };
        let cfg = ExtractConfig::default();
        let bins = FragIndex::geometry(&lib, 20.0);
        let stats = LocalIndexStats::default();
        let local = TaskProbe::Local {
            lib: &lib,
            bins: &bins,
            tol_ppm: 20.0,
            stats: &stats,
        };
        let global = TaskProbe::Global(&idx);

        // Callback for callback: every task shape this fixture produces (each window whole,
        // and cut into 2, 3 and 7 sub-ranges, which covers the grids of the pool sizes
        // below), every scan of the window, every peak. The task-local index must hand the
        // probe exactly the postings the narrowed global index handed it, in the same order,
        // with the same m/z, intensity and ordinal bits.
        let mut n_callbacks = 0usize;
        for g in &groups {
            for pieces in [1u32, 2, 3, 7] {
                let span = g.hi_cid - g.lo_cid;
                for k in 0..pieces {
                    let lo = g.lo_cid + span * k / pieces;
                    let hi = g.lo_cid + span * (k + 1) / pieces;
                    let mut li = LocalIndex::build(&lib, &bins, 20.0, lo, hi);
                    let mut np = NarrowedProbe {
                        idx: &idx,
                        nw: idx.window_narrow(lo, hi),
                    };
                    for &si in &g.scans {
                        for peak in &sc[si].peaks {
                            let mz = peak.mz as f64;
                            let q = mz / mass_off.factor_at(mz);
                            let bin = bins.bin(q) as u32;
                            assert_eq!(bin, idx.bin_of(q), "one geometry");
                            let (mut a, mut b) = (Vec::new(), Vec::new());
                            li.probe_binned(q, bin, |c, m, it, f| {
                                a.push((c, m.to_bits(), it.to_bits(), f))
                            });
                            np.probe_binned(q, bin, |c, m, it, f| {
                                b.push((c, m.to_bits(), it.to_bits(), f))
                            });
                            assert_eq!(a, b, "sub-range {lo}..{hi} scan {si} peak m/z {q}");
                            n_callbacks += a.len();
                        }
                    }
                }
            }
        }
        assert!(n_callbacks > 10_000, "the comparison must see real traffic");

        // `bound = 0` flushes nothing, so the whole batch stays in the accumulator and the
        // comparison is over the accumulation itself. `flush_all` below runs the same
        // fixture with the flush live.
        let run =
            |probe: TaskProbe<'_>, threads: usize, bound: u32| -> (HitAcc, Vec<(u32, Vec<Hit>)>) {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("thread pool");
                let mut acc = HitAcc::default();
                let mut chunk = HitStore::default();
                let mut flushed: Vec<(u32, Vec<Hit>)> = Vec::new();
                pool.install(|| {
                    let mut sink = |c: Vec<(u32, &mut [Hit])>| -> bool {
                        for (cid, hits) in c {
                            flushed.push((cid, hits.to_vec()));
                        }
                        true
                    };
                    accumulate_groups(
                        // One band in this test, so the probing fan-out is the whole pool.
                        probe, 1, &groups, &sc, &rt_lo, &rt_hi, &mass_off, &cfg, None, bound,
                        &mut acc, &mut chunk, &mut sink,
                    );
                });
                (acc, flushed)
            };
        // Two threads: 2 * 2 / 4 windows = one sub-range per window, the unsplit path, over
        // the global index: the reference both probes are held to.
        let (acc, empty) = run(global, 2, 0);
        assert!(empty.is_empty(), "bound 0 must flush nothing");
        let unsplit = materialize(acc);
        assert!(!unsplit.is_empty(), "the fixture must produce hits");
        assert!(
            unsplit.values().any(|v| v.len() > 1),
            "candidates must collect several hits, or hit order proves nothing"
        );
        for (probe, what) in [(global, "global"), (local, "local")] {
            for threads in [2, 8, 16] {
                let split = materialize(run(probe, threads, 0).0);
                assert_eq!(
                    split.len(),
                    unsplit.len(),
                    "{what}, {threads} threads: candidate count"
                );
                for (cid, hits) in &unsplit {
                    assert_eq!(
                        split.get(cid),
                        Some(hits),
                        "{what}, {threads} threads: candidate {cid} hits differ"
                    );
                }
            }
        }
        // Flushing per candidate sub-range instead of per batch must deliver exactly the
        // same candidates, ascending, with the same hits in the same order: the sub-range
        // grid is where a candidate becomes final, not where its evidence changes.
        for (probe, threads) in [
            (global, 2),
            (global, 16),
            (local, 2),
            (local, 8),
            (local, 16),
        ] {
            let (acc, flushed) = run(probe, threads, u32::MAX);
            assert_eq!(acc.n_hits(), 0, "{threads} threads: nothing may stay open");
            assert!(
                flushed.windows(2).all(|w| w[0].0 < w[1].0),
                "{threads} threads: flushed candidates must be strictly ascending"
            );
            assert_eq!(
                flushed.len(),
                unsplit.len(),
                "{threads} threads: flushed candidate count"
            );
            for (cid, hits) in &flushed {
                assert_eq!(
                    unsplit.get(cid),
                    Some(hits),
                    "{threads} threads: flushed candidate {cid} hits differ"
                );
            }
        }
    }

    /// Flush the whole accumulator through the chunked gather the driver uses, and read the
    /// CSR back as the `HashMap<u32, Vec<Hit>>` the accumulator used to be. This is the
    /// assertion that the flat layout hands the per-candidate pass the same sequences: the
    /// gather, the CSR spine and `slices_mut` all sit between `accumulate_groups` and
    /// `per_candidate` in production.
    fn materialize(mut acc: HitAcc) -> BTreeMap<u32, Vec<Hit>> {
        let mut out: BTreeMap<u32, Vec<Hit>> = BTreeMap::new();
        let mut chunk = HitStore::default();
        let mut last: Option<u32> = None;
        loop {
            // A small chunk on purpose, so several gathers run and their boundaries are
            // exercised against a fixture with far more candidates than that.
            gather_chunk(&mut acc.runs, u32::MAX, 512, &mut chunk);
            if chunk.is_empty() {
                break;
            }
            assert_eq!(
                chunk.offs.len(),
                chunk.cids.len() + 1,
                "the CSR spine must have one offset per candidate plus the terminator"
            );
            assert_eq!(
                *chunk.offs.last().expect("offs is never empty"),
                chunk.hits.len(),
                "the last offset must be the hit count"
            );
            for (cid, hits) in chunk.slices_mut() {
                assert!(
                    last.map(|l| l < cid).unwrap_or(true),
                    "candidates must be gathered strictly ascending"
                );
                last = Some(cid);
                assert!(!hits.is_empty(), "an empty segment must not be emitted");
                out.insert(cid, hits.to_vec());
            }
            acc.compact();
        }
        out
    }

    /// The counting sort is the step that replaces "push onto this candidate's Vec", so it
    /// must reproduce that exactly: same candidates, and each candidate's hits in arrival
    /// order. Built here against the naive per-candidate accumulation it replaced.
    #[test]
    fn grouping_reproduces_per_candidate_push_order() {
        let (lo, hi) = (10u32, 23u32);
        let mut cid: Vec<u32> = Vec::new();
        let mut hits: Vec<Hit> = Vec::new();
        let mut reference: BTreeMap<u32, Vec<Hit>> = BTreeMap::new();
        // Interleaved candidates, several repeats, and some candidates of the span never
        // touched (11, 12 and 22 stay empty), which is what the occupancy scan must skip.
        for i in 0..200u32 {
            let c = lo + ((i * 7) % 11) + if i % 5 == 0 { 2 } else { 0 };
            let c = c.min(hi - 1);
            let h = Hit {
                scan: i,
                frag: (i % 6) as u16,
                inten: 1.0 + i as f32,
                obs_mz: 300.0 + i as f32 * 0.01,
            };
            cid.push(c);
            hits.push(h);
            reference.entry(c).or_default().push(h);
        }
        let (cids, offs) = group_hits_by_candidate(lo, hi, &mut cid, &mut hits);
        let store = HitStore { cids, offs, hits };
        assert_eq!(
            store.len(),
            reference.len(),
            "every candidate with a hit must own a segment, and no other"
        );
        for (i, (&rc, rhits)) in reference.iter().enumerate() {
            assert_eq!(store.cids[i], rc, "segment {i} candidate");
            assert_eq!(store.slice(i), rhits.as_slice(), "candidate {rc} hit order");
        }
    }

    /// A candidate seen in several windows keeps window order: the windows' runs are
    /// appended to the accumulator in window order, and the gather concatenates a
    /// candidate's segments run by run.
    #[test]
    fn gather_concatenates_a_candidate_across_runs_in_window_order() {
        let h = |scan: u32| Hit {
            scan,
            frag: 0,
            inten: 1.0,
            obs_mz: 300.0,
        };
        let mut a = HitStore::default();
        a.push_segment(4, &[h(1), h(2)]);
        a.push_segment(9, &[h(3)]);
        let mut b = HitStore::default();
        b.push_segment(4, &[h(4)]);
        b.push_segment(7, &[h(5)]);
        let mut acc = HitAcc {
            runs: vec![HitRun::new(vec![a]), HitRun::new(vec![b])],
        };
        let mut out = HitStore::default();
        gather_chunk(&mut acc.runs, u32::MAX, usize::MAX, &mut out);
        assert_eq!(out.cids, vec![4, 7, 9]);
        assert_eq!(
            out.slice(0).iter().map(|x| x.scan).collect::<Vec<_>>(),
            vec![1, 2, 4],
            "the earlier window's hits must come first"
        );
        assert_eq!(out.slice(1).iter().map(|x| x.scan).collect::<Vec<_>>(), [5]);
        assert_eq!(out.slice(2).iter().map(|x| x.scan).collect::<Vec<_>>(), [3]);
    }

    /// `bound` is the only thing holding a candidate back, and what it holds back must
    /// still be there, in order, for the next flush.
    #[test]
    fn gather_stops_at_the_bound_and_keeps_the_rest() {
        let h = |scan: u32| Hit {
            scan,
            frag: 0,
            inten: 1.0,
            obs_mz: 300.0,
        };
        let mut a = HitStore::default();
        for c in [2u32, 5, 8, 11] {
            a.push_segment(c, &[h(c)]);
        }
        let mut acc = HitAcc {
            runs: vec![HitRun::new(vec![a])],
        };
        let mut out = HitStore::default();
        gather_chunk(&mut acc.runs, 8, usize::MAX, &mut out);
        assert_eq!(out.cids, vec![2, 5]);
        assert_eq!(acc.n_hits(), 2, "8 and 11 stay open");
        acc.compact();
        gather_chunk(&mut acc.runs, u32::MAX, usize::MAX, &mut out);
        assert_eq!(out.cids, vec![8, 11]);
        assert_eq!(acc.n_hits(), 0);
    }

    /// The zero-copy flush hands over exactly the batches the gathering flush builds: the
    /// same candidates in the same calls, the same hits in the same order, and it leaves the
    /// runs holding the same remainder. Randomised over runs with disjoint and overlapping
    /// candidates, several parts, empty stores, bounds inside and past the runs and small
    /// chunk caps, so every branch of `single_run_span` is taken.
    #[test]
    fn the_zero_copy_flush_hands_over_the_batches_the_gather_builds() {
        let mut state = 0x2e40_u64;
        let mut next = move |m: u64| -> u64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m.max(1)
        };
        let mut n_zero_copy_batches = 0usize;
        for case in 0..400 {
            // One recipe, built twice: `HitStore` is not `Clone`, on purpose.
            let n_runs = 1 + next(4) as usize;
            let mut recipe: Vec<Vec<Vec<(u32, u32)>>> = Vec::new(); // run -> part -> (cid, n)
            for _ in 0..n_runs {
                let n_parts = 1 + next(3) as usize;
                let mut c = next(20) as u32;
                let mut parts = Vec::new();
                for _ in 0..n_parts {
                    let mut part = Vec::new();
                    for _ in 0..next(12) {
                        part.push((c, 1 + next(3) as u32));
                        c += 1 + next(3) as u32;
                    }
                    parts.push(part);
                }
                recipe.push(parts);
            }
            let build = || -> Vec<HitRun> {
                let mut scan = 0u32;
                recipe
                    .iter()
                    .map(|parts| {
                        let stores = parts
                            .iter()
                            .map(|part| {
                                let mut st = HitStore::default();
                                for &(cid, n) in part {
                                    let hits: Vec<Hit> = (0..n)
                                        .map(|_| {
                                            scan += 1;
                                            Hit {
                                                scan,
                                                frag: (scan % 7) as u16,
                                                inten: scan as f32,
                                                obs_mz: 300.0,
                                            }
                                        })
                                        .collect();
                                    st.push_segment(cid, &hits);
                                }
                                st
                            })
                            .collect();
                        HitRun::new(stores)
                    })
                    .collect()
            };
            let bound = [u32::MAX, next(60) as u32, 0][case % 3];
            let cap = 1 + next(6) as usize;
            let outcome = |zero_copy: bool| {
                let mut runs = build();
                let mut chunk = HitStore::default();
                let mut batches: Vec<Vec<(u32, Vec<Hit>)>> = Vec::new();
                let mut sink = |c: Vec<(u32, &mut [Hit])>| -> bool {
                    batches.push(c.into_iter().map(|(cid, h)| (cid, h.to_vec())).collect());
                    true
                };
                let stopped = flush_below(&mut runs, bound, cap, zero_copy, &mut chunk, &mut sink);
                assert!(!stopped);
                let mut rest = HitStore::default();
                gather_chunk(&mut runs, u32::MAX, usize::MAX, &mut rest);
                let rest: Vec<(u32, Vec<Hit>)> = rest
                    .slices_mut()
                    .into_iter()
                    .map(|(c, h)| (c, h.to_vec()))
                    .collect();
                (batches, rest)
            };
            let reference = outcome(false);
            let candidate = outcome(true);
            assert_eq!(
                candidate, reference,
                "case {case}: bound {bound}, cap {cap}"
            );
            // Whether the first batch of this case is one the zero-copy path takes.
            if single_run_span(&build(), bound, cap).is_some() {
                n_zero_copy_batches += 1;
            }
        }
        assert!(
            n_zero_copy_batches > 50,
            "the zero-copy branch must be exercised ({n_zero_copy_batches} cases)"
        );
    }

    /// The chunk cap is a flush-size limit, not a filter: chunking must not lose or
    /// reorder a candidate.
    #[test]
    fn gather_chunking_partitions_the_accumulator() {
        let h = |scan: u32| Hit {
            scan,
            frag: 0,
            inten: 1.0,
            obs_mz: 300.0,
        };
        let mut a = HitStore::default();
        for c in 0..50u32 {
            a.push_segment(c, &[h(2 * c), h(2 * c + 1)]);
        }
        let mut acc = HitAcc {
            runs: vec![HitRun::new(vec![a])],
        };
        let mut seen: Vec<u32> = Vec::new();
        let mut out = HitStore::default();
        loop {
            gather_chunk(&mut acc.runs, u32::MAX, 7, &mut out);
            if out.is_empty() {
                break;
            }
            assert!(out.len() <= 7, "the chunk cap must bind");
            seen.extend_from_slice(&out.cids);
            acc.compact();
        }
        assert_eq!(seen, (0..50u32).collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod psms_stream_tests {
    use super::*;

    /// A per-process scratch path; tests in this module run concurrently.
    fn scratch(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("mumdia_psms_stream_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_str().unwrap().to_string()
    }

    /// One accepted row whose every column varies with `i`, with nulls in the MS1 columns.
    fn row(i: usize) -> CandOut {
        let f = i as f64;
        CandOut {
            cid: i as u32,
            peak_rank: (i % 3) as u8,
            apex_rt: 100.0 + f * 0.25,
            apex_int: 1.0 + (i % 97) as f32,
            n_match: (i % 7) as i32,
            corun: (i % 5) as i32,
            npred: 6,
            calrt: if i.is_multiple_of(11) {
                f64::NAN
            } else {
                f * 0.5
            },
            mz: 400.0 + (i % 1000) as f64 * 0.01,
            contested: (i % 4) as f64 * 0.25,
            contested_count_frac: (i % 3) as f64 / 3.0,
            apportioned_frac: (i % 2) as f64,
            z: 2 + (i % 2) as i32,
            label: if i.is_multiple_of(2) {
                "target"
            } else {
                "decoy"
            }
            .to_string(),
            base: (i / 2) as u32,
            pform: format!("PEPT{}IDEK", i % 50),
            prot: format!("P{}", i % 13),
            irt: (i % 200) as f32 * 0.1,
            ms1_m1: i.is_multiple_of(3).then_some(f),
            ms1_mono: (!i.is_multiple_of(5)).then_some(f * 2.0),
            ms1_i1: None,
            ms1_i2: (i % 7 == 1).then_some(-f),
            gate_apex: (i % 10) as f32 * 0.1,
            gate_peak_spectral: 0.5,
            gate_coelution: -0.25,
            gate_spectral_entropy: (i % 9) as f32,
            deconv_explained: 0.0,
            deconv_active: 1.0,
            deconv_share: 0.5,
            deconv_collin: 0.125,
            deconv_shadow: 0.0,
            chrom: Vec::new(),
            peaks: Vec::new(),
        }
    }

    /// The streamed table is the `write_table` file: at 0 rows (the one empty chunk), one
    /// row, exactly one chunk (whose last chunk `push` wrote, so `finish` writes no tail),
    /// one past it, and two chunks plus a short tail. The unstreamed arm is `write_table`
    /// over the whole table, which is how the table was written before it was streamed.
    /// Every optional column group is covered by the second configuration.
    #[test]
    fn the_streamed_psms_table_is_the_write_table_file() {
        let c = WRITE_TABLE_CHUNK_ROWS;
        let wide = ExtractConfig {
            emit_contested_features: true,
            emit_gate_diagnostics: true,
            emit_demix_features: true,
            ..ExtractConfig::default()
        };
        let cases: [(&str, &ExtractConfig, usize); 7] = [
            ("default", &ExtractConfig::default(), 0),
            ("default", &ExtractConfig::default(), 1),
            ("default", &ExtractConfig::default(), c),
            ("default", &ExtractConfig::default(), c + 1),
            ("default", &ExtractConfig::default(), 2 * c + 3),
            ("wide", &wide, 0),
            ("wide", &wide, c + 1),
        ];
        for (tag, cfg, n) in cases {
            let streamed = scratch(&format!("streamed_{tag}_{n}.parquet"));
            let whole = scratch(&format!("whole_{tag}_{n}.parquet"));
            let (mut s, mut w) = (
                PsmStream::new(&streamed, true, 7),
                PsmStream::new(&whole, false, 7),
            );
            for i in 0..n {
                s.push(row(i), cfg).unwrap();
                w.push(row(i), cfg).unwrap();
                // Streamed, a full chunk leaves nothing pending; whole, nothing is written.
                assert_eq!(s.rows.len(), (i + 1) % c, "{tag} {n}: pending rows");
                assert_eq!(w.rows.len(), i + 1);
            }
            let (s_written, _) = s.finish(cfg).unwrap();
            let (w_written, _) = w.finish(cfg).unwrap();
            assert_eq!(s_written.rows, n as u64, "{tag} {n}");
            assert_eq!(w_written.rows, n as u64, "{tag} {n}");
            assert_eq!(
                std::fs::read(&streamed).unwrap(),
                std::fs::read(&whole).unwrap(),
                "{tag}, {n} rows: streamed psms_extracted differs from write_table's"
            );
            // The hash taken while writing is the hash of the published file.
            assert_eq!(
                s_written.content_hash,
                mumdia_io::hash::blake3_file(&streamed).unwrap(),
                "{tag} {n}: streamed hash"
            );
            assert_eq!(s_written.content_hash, w_written.content_hash, "{tag} {n}");
            let t = TableFile::open(&streamed).unwrap();
            assert_eq!(t.nrows, n);
            if n > 0 {
                // The global offset is applied to every chunk, not only the first.
                assert_eq!(t.u32("candidate_id").unwrap()[n - 1], (n - 1) as u32 + 7);
            }
        }
    }

    /// A chunk that cannot be written is an error from the push that fills it, not before
    /// and not later, and nothing is published at the path.
    #[test]
    fn a_failed_chunk_write_is_the_error_of_the_push_that_filled_it() {
        // The parent of the output is a FILE, so the writer cannot create its temp file.
        let parent = scratch("not_a_directory");
        std::fs::write(&parent, b"a file").unwrap();
        let out = format!("{parent}/psms_extracted.parquet");
        let cfg = ExtractConfig::default();
        let mut s = PsmStream::new(&out, true, 0);
        for i in 0..WRITE_TABLE_CHUNK_ROWS - 1 {
            s.push(row(i), &cfg).unwrap();
        }
        assert!(
            s.push(row(0), &cfg).is_err(),
            "the full chunk's write must fail"
        );
        drop(s);
        assert!(!std::path::Path::new(&out).exists());
    }

    /// Temp files a writer left next to `path` (`AtomicPath` names them `<path>.tmp-*`).
    fn leftovers(path: &str) -> Vec<String> {
        let p = std::path::Path::new(path);
        let stem = format!("{}.tmp-", p.file_name().unwrap().to_str().unwrap());
        std::fs::read_dir(p.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_str().unwrap().to_string())
            .filter(|n| n.starts_with(&stem))
            .collect()
    }

    /// After a failed psms_extracted write the chromatogram table the writer thread was
    /// fed is not published: the call returns that error, the chromatograms table already
    /// at the path is untouched and no temp file is left. Without a failure it is published.
    #[test]
    fn a_failed_psms_write_leaves_the_previous_chromatograms_in_place() {
        let path = scratch("chromatograms.parquet");
        let table = |v: Vec<u32>| vec![Col::U32("candidate_id".into(), v)];
        write_table(&path, table(vec![1, 2, 3])).unwrap();
        let before = std::fs::read(&path).unwrap();

        let mut w = TableWriter::new(&path).with_content_hash();
        w.write_cols(table(vec![9])).unwrap();
        let err =
            close_chromatograms(w, Some(anyhow::anyhow!("psms chunk write failed"))).unwrap_err();
        assert!(err.to_string().contains("psms chunk write failed"), "{err}");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "replaced after a failure"
        );
        assert!(leftovers(&path).is_empty(), "{:?}", leftovers(&path));

        let mut w = TableWriter::new(&path).with_content_hash();
        w.write_cols(table(vec![9])).unwrap();
        let published = close_chromatograms(w, None).unwrap();
        assert_eq!(published.rows, 1);
        assert_eq!(
            published.content_hash,
            mumdia_io::hash::blake3_file(&path).unwrap()
        );
        assert_eq!(
            TableFile::open(&path).unwrap().u32("candidate_id").unwrap(),
            vec![9]
        );
        assert!(leftovers(&path).is_empty());
    }
}
