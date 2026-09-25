//! Stage G `mumdia quant` (docs/12_quant_lfq_align_mbr_report_audit.md): quantify
//! identified peptidoforms and roll up to protein groups. Integrate each fragment
//! chromatogram over the apex region by the trapezoidal rule, sum the top-N
//! fragments into a per-run peptidoform quantity, then roll up to protein groups.
//! MVP is single-run, so cross-run normalization and MaxLFQ/directLFQ (which need
//! multiple runs) reduce to a top-N sum; the method is a config strategy for
//! later.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::{anyhow, Result};
use arrow::array::{Array, Float32Array, StringArray, UInt32Array};
use mumdia_core::config::{
    FragmentSelection, NormalizeMethod, PeakWindowMode, QuantConfig, QuantQColumn, RollupMethod,
};
use mumdia_core::schema::artifact;
use mumdia_io::report::{ArtifactReport, Written};
use mumdia_io::table::{column_names, write_table, Col, ListF32, TableFile};
use rayon::prelude::*;
use serde_json::json;
use tracing::{info, warn};

pub struct QuantParams<'a> {
    pub psms_scored: &'a str,
    pub chromatograms: &'a str,
    pub out_peptide: &'a str,
    pub out_protein: &'a str,
    /// Optional per-fragment area export (for ion-level directLFQ across runs).
    pub out_fragment: Option<&'a str>,
    /// Optional per-candidate peak-window diagnostic (candidate_id, lo_rt, hi_rt,
    /// width_s). Emitted only when `bound_peak` is on; a diagnostic of the
    /// integration windows, not part of the quant contract.
    pub out_peak_bounds: Option<&'a str>,
    pub cfg: &'a QuantConfig,
    pub config_hash: &'a str,
}

/// Interned fragment names. A chromatogram table has tens of millions of rows and only a
/// few dozen distinct fragment names (`y1`..`y30`, `b1`..), so a row carries a name id and
/// each string is stored once instead of the `String` per row this stage used to keep.
#[derive(Default)]
struct NameTab {
    ids: HashMap<String, u32>,
    names: Vec<String>,
}

impl NameTab {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.ids.get(s) {
            return i;
        }
        let i = self.names.len() as u32;
        self.names.push(s.to_string());
        self.ids.insert(s.to_string(), i);
        i
    }

    fn get(&self, id: u32) -> &str {
        &self.names[id as usize]
    }
}

/// Marker for a row with no RT axis at all: extract emits an empty trace, not a
/// zero-filled one, for a predicted fragment that was never observed.
const NO_AXIS: u32 = u32::MAX;

/// A string column at some rows as one buffer plus per-value offsets, from
/// [`TableFile::str_flat_rows`].
///
/// `TableFile::str` returns a `String` per row: a 24-byte spine plus its own heap block.
/// On the six-run HYE scored table (879,027 rows) `peptidoform` and `protein_group`
/// together are 1.76 million live allocations and 42 MB of spine to carry 31 MB of text,
/// held for the whole stage, and both are read per row but cloned only for the few
/// percent of rows that become output. This is the payload of those rows alone, in two
/// allocations.
struct FlatStr {
    off: Vec<usize>,
    txt: String,
}

impl FlatStr {
    /// The column at `rows` (strictly ascending): value `j` is row `rows[j]`.
    fn read_rows(t: &TableFile, name: &str, rows: &[usize]) -> Result<FlatStr> {
        let (off, txt) = t.str_flat_rows(name, rows)?;
        Ok(FlatStr { off, txt })
    }

    #[inline]
    fn get(&self, row: usize) -> &str {
        &self.txt[self.off[row]..self.off[row + 1]]
    }
}

/// Largest bitset [`CidSet`] will build, in bytes. 64 MiB of bits covers a candidate-id
/// range of 537 million, which is 2.6x the largest library measured (the 203M-precursor
/// immuno library); past that the set falls back to hashing rather than sizing an
/// allocation off an untrusted id.
const CID_BITSET_MAX_BYTES: usize = 64 << 20;

/// The candidate ids whose chromatogram rows quant keeps.
///
/// This is probed once per chromatogram row -- 14,306,517 times on the six-run HYE
/// artifact -- to answer a question about a contiguous library row index, so a
/// `HashSet<u32>` pays SipHash for every row of the largest artifact of the run. The
/// accepted ids span at most the library's precursor count, so one bit each is 1.4 MB at
/// 10.9M precursors and 25 MB at 203M, and the probe becomes a shift and a load.
///
/// The base comes from the set rather than being assumed to be 0: a banded search offsets
/// candidate ids by the band's `lib.global_offset`, so a band's accepted ids can start
/// anywhere in the library.
enum CidSet {
    /// `bits` covers `base..=top`; bit `c - base` is set when `c` is wanted.
    Bits { base: u32, top: u32, bits: Vec<u64> },
    /// The id range is too wide to be worth a bit each (see [`CID_BITSET_MAX_BYTES`]).
    Hashed(std::collections::HashSet<u32>),
}

impl CidSet {
    fn empty() -> CidSet {
        CidSet::Bits {
            base: 0,
            top: 0,
            bits: Vec::new(),
        }
    }

    /// `ids` need not be sorted or distinct: a repeated id sets the same bit twice.
    fn from_ids(ids: &[u32]) -> CidSet {
        let (Some(&base), Some(&top)) = (ids.iter().min(), ids.iter().max()) else {
            return CidSet::empty();
        };
        // In u64: `top - base` reaches `u32::MAX`, and `+ 1` on a 32-bit `usize` would wrap
        // to a zero-length allocation that the fill below then indexes out of.
        let span = u64::from(top - base) + 1;
        if span.div_ceil(64) * 8 > CID_BITSET_MAX_BYTES as u64 {
            return CidSet::Hashed(ids.iter().copied().collect());
        }
        let span = span as usize;
        let mut bits = vec![0u64; span.div_ceil(64)];
        for &c in ids {
            let k = (c - base) as usize;
            bits[k >> 6] |= 1u64 << (k & 63);
        }
        CidSet::Bits { base, top, bits }
    }

    #[inline]
    fn contains(&self, c: u32) -> bool {
        match self {
            CidSet::Bits { base, bits, .. } => match c.checked_sub(*base) {
                Some(k) => {
                    let k = k as usize;
                    bits.get(k >> 6).is_some_and(|w| (w >> (k & 63)) & 1 == 1)
                }
                None => false,
            },
            CidSet::Hashed(h) => h.contains(&c),
        }
    }

    /// Inclusive `(min, max)` of the ids in the set, or `None` when it is empty. A row
    /// group whose `candidate_id` statistics lie outside this range holds no wanted row.
    fn range(&self) -> Option<(u32, u32)> {
        match self {
            CidSet::Bits { bits, .. } if bits.is_empty() => None,
            CidSet::Bits { base, top, .. } => Some((*base, *top)),
            CidSet::Hashed(h) => match (h.iter().min(), h.iter().max()) {
                (Some(&lo), Some(&hi)) => Some((lo, hi)),
                _ => None,
            },
        }
    }
}

/// Distinct candidate ids marked so far, one bit per id over a range fixed up front (the
/// hashed fallback past [`CID_BITSET_MAX_BYTES`], as for [`CidSet`]). For a count over a
/// whole table that must not cost a map entry per candidate.
struct CidMarks {
    base: u32,
    bits: Vec<u64>,
    hashed: Option<std::collections::HashSet<u32>>,
    count: usize,
}

impl CidMarks {
    /// Marks for ids in `min(ids)..=max(ids)`; every id later inserted must lie there.
    fn over(ids: &[u32]) -> CidMarks {
        let empty = CidMarks {
            base: 0,
            bits: Vec::new(),
            hashed: None,
            count: 0,
        };
        let (Some(&base), Some(&top)) = (ids.iter().min(), ids.iter().max()) else {
            return empty;
        };
        let span = u64::from(top - base) + 1;
        if span.div_ceil(64) * 8 > CID_BITSET_MAX_BYTES as u64 {
            return CidMarks {
                hashed: Some(std::collections::HashSet::new()),
                ..empty
            };
        }
        CidMarks {
            base,
            bits: vec![0u64; (span as usize).div_ceil(64)],
            ..empty
        }
    }

    /// Mark `c`, returning whether it was unmarked before.
    #[inline]
    fn insert(&mut self, c: u32) -> bool {
        let new = match &mut self.hashed {
            Some(h) => h.insert(c),
            None => {
                let k = (c - self.base) as usize;
                let (w, b) = (k >> 6, 1u64 << (k & 63));
                let new = self.bits[w] & b == 0;
                self.bits[w] |= b;
                new
            }
        };
        self.count += usize::from(new);
        new
    }

    fn count(&self) -> usize {
        self.count
    }
}

/// The fragment chromatogram rows quant keeps, in table order, in flat buffers.
///
/// This is the same treatment `features` gives the same table ([`super::features`]'s
/// `ChromChunk`): one heap block per array instead of a `Vec<f32>` for the RT axis, a
/// `Vec<f32>` for the intensities and a `String` for the fragment name PER ROW. The
/// motivation is the number of live heap blocks rather than the bytes: a grouped search
/// died at ~290 GB resident with 1.7 TB free because the process had hit the kernel's
/// per-process memory-mapping limit (1,048,576) with one mapping per medium block, and
/// `--out-peak-bounds` bypasses the accepted-candidate filter and holds the whole table
/// this way. The RT axis is additionally shared between the rows of one candidate that
/// sample the same grid, which is every row of a candidate under extract's window-grid
/// mode, so the axis is stored once per candidate instead of once per fragment.
#[derive(Default)]
struct ChromStore {
    names: NameTab,
    /// Candidate id per row, in table order.
    cid: Vec<u32>,
    name_id: Vec<u32>,
    pred: Vec<f32>,
    /// Index into `axis_off`, or [`NO_AXIS`] for an empty trace.
    axis_id: Vec<u32>,
    /// Axis `a` is `axis_vals[axis_off[a]..axis_off[a + 1]]`.
    axis_off: Vec<usize>,
    axis_vals: Vec<f32>,
    /// Axis `a` is strictly increasing and non-negative, so it carries no duplicate RT
    /// and no NaN. [`peak_window`] needs that before it may treat the axis as the union
    /// axis directly instead of merging the samples.
    axis_strict: Vec<bool>,
    /// Row `r`'s intensities are `int_vals[int_off[r]..int_off[r + 1]]`.
    int_off: Vec<usize>,
    int_vals: Vec<f32>,
    /// First axis of the candidate being filled, so dedup only compares within it.
    open_axis_lo: usize,
    open_cid: Option<u32>,
    /// Every stored axis is non-decreasing, non-negative and NaN-free, so the nearest
    /// sample to an apex may be found by binary search rather than by scanning the whole
    /// trace. False makes every search fall back to the scan, which is what this stage
    /// always did.
    rt_sorted: bool,
}

impl ChromStore {
    fn new() -> ChromStore {
        ChromStore {
            axis_off: vec![0],
            int_off: vec![0],
            rt_sorted: true,
            ..Default::default()
        }
    }

    fn nrows(&self) -> usize {
        self.cid.len()
    }

    /// The first axis of the OPEN candidate whose values are identical to `rt`, in
    /// ascending id order. [`ChromStore::axis_for`] interns a row's trace through this, and
    /// [`ChromStore::append`] re-runs it across a row-group seam so a split table dedups
    /// exactly as one pass would have.
    ///
    /// BITWISE, not `==`. Interning makes every consumer read `store.rt(row)` instead of
    /// the row's own values, so it may only substitute an axis that is identical to the
    /// last bit. `-0.0 == 0.0` holds under f32's `PartialEq` while [`peak_window`]'s merge
    /// keys the union on `to_bits`, so an `==` match would fold two union points into one
    /// and move the profile, the apex, the walked window and the quantity. Extract's grid
    /// is positive mzML scan times, but `mumdia quant --chromatograms` takes a table
    /// written by anything, which is the same reason the rt/intensity length check in
    /// [`load_chrom_span`] exists. (A NaN-carrying axis dedups against an identical one,
    /// which `==` never did; the values stored are the same bits either way.)
    fn open_axis_matching(&self, rt: &[f32]) -> Option<u32> {
        (self.open_axis_lo..self.axis_off.len() - 1)
            .find(|&a| {
                let (lo, hi) = (self.axis_off[a], self.axis_off[a + 1]);
                hi - lo == rt.len()
                    && self.axis_vals[lo..hi]
                        .iter()
                        .zip(rt)
                        .all(|(x, y)| x.to_bits() == y.to_bits())
            })
            .map(|a| a as u32)
    }

    /// Store `rt` as an axis id, reusing an axis already stored for the open candidate
    /// when the values are identical (the common case: one grid per candidate).
    fn axis_for(&mut self, cid: u32, rt: &[f32]) -> Result<u32> {
        if self.open_cid != Some(cid) {
            self.open_cid = Some(cid);
            self.open_axis_lo = self.axis_off.len() - 1;
        }
        if rt.is_empty() {
            return Ok(NO_AXIS);
        }
        if let Some(a) = self.open_axis_matching(rt) {
            return Ok(a);
        }
        let nondecreasing = rt_is_sorted(rt);
        self.rt_sorted &= nondecreasing;
        self.axis_strict
            .push(nondecreasing && rt.windows(2).all(|w| w[0] < w[1]));
        self.axis_vals.extend_from_slice(rt);
        self.axis_off.push(self.axis_vals.len());
        let id = (self.axis_off.len() - 2) as u32;
        // [`NO_AXIS`] means "this row has no RT axis": [`ChromStore::axis`] hands back an
        // empty slice for it, [`peak_window`]'s uniformity scan skips the row and
        // [`fixed_window_indices`] returns `None`, so a real axis minted under that id
        // would integrate to 0 without a word. One comparison per DISTINCT axis, at the
        // point the id is minted, is cheaper than the argument that the count cannot get
        // there -- and `--out-peak-bounds` is exactly the path that holds the whole table.
        if id == NO_AXIS {
            anyhow::bail!(
                "chromatogram table has more than {NO_AXIS} distinct retention-time axes; \
                 the axis id would collide with the empty-trace marker"
            );
        }
        Ok(id)
    }

    fn push(&mut self, cid: u32, name: &str, pred: f32, rt: &[f32], inten: &[f32]) -> Result<()> {
        // The shared-axis profile in [`peak_window`] is sized from the AXIS and indexed by
        // the INTENSITY position, and `fixed_window_indices` slices the intensities with
        // indices taken from the RT trace. `run` refuses a mismatched row while it can
        // still name the candidate and the file; this catches a fixture or a future loader
        // that does not.
        debug_assert_eq!(
            rt.len(),
            inten.len(),
            "chromatogram row for candidate_id {cid} has mismatched rt/intensity lengths"
        );
        let axis_id = self.axis_for(cid, rt)?;
        let name_id = self.names.intern(name);
        self.cid.push(cid);
        self.name_id.push(name_id);
        self.pred.push(pred);
        self.axis_id.push(axis_id);
        self.int_vals.extend_from_slice(inten);
        self.int_off.push(self.int_vals.len());
        Ok(())
    }

    /// Append `other`'s rows after this store's, producing exactly the store one pass over
    /// the table in this order would have produced -- axis ids included. This is what lets
    /// the chromatogram table be read row group by row group in parallel
    /// ([`load_chromatograms`]): row groups are disjoint, contiguous row spans, so
    /// concatenating their stores in file order reproduces table order exactly.
    ///
    /// The one thing concatenation can get wrong is axis IDENTITY across the seam.
    /// [`ChromStore::axis_for`] dedups a trace only against the axes of the candidate it
    /// currently has open, and `other` was built with nothing open, so a candidate whose
    /// rows straddle the boundary would mint a second axis holding the same values. That is
    /// not an internal detail: [`peak_window`] takes the shared-axis accumulation for one
    /// axis id and the merged-sample union for two, and those paths do NOT agree for every
    /// axis this table may carry. An axis `[-0.0, 1.0, 2.0, 3.0]` is marked strict
    /// ([`rt_is_sorted`] accepts `-0.0`, which compares `>= 0.0`), so the shared path walks
    /// it in value order, while the merge keys the union on `to_bits`, where `-0.0`'s bits
    /// sort after every positive f32: the merged path walks `[1.0, 2.0, 3.0, -0.0]` with
    /// the profile permuted, `axis_sorted` false, and reaches a different apex, window and
    /// quantity. Extract writes positive mzML scan times, but `mumdia quant
    /// --chromatograms` takes a table written by anything, and a parquet row-group layout
    /// is the writer's choice, so the seam must not be able to decide this.
    ///
    /// So the leading rows of `other` that continue this store's open candidate are deduped
    /// against the open window here, by the same [`ChromStore::open_axis_matching`] search
    /// in the same ascending-id order, and the open candidate is carried across the seam
    /// rather than closed. Every field then matches the single pass bit for bit, which is
    /// what `the_row_group_plan_builds_the_single_passes_store_exactly` asserts over a
    /// fixture split at every row.
    fn append(&mut self, other: ChromStore) -> Result<()> {
        if other.nrows() == 0 {
            // A span that kept no row is a span the single pass walked straight past: it
            // must not close the open candidate, or the next span's rows would stop
            // deduping against it. No rows also means no axes, which only `push` mints.
            debug_assert_eq!(other.axis_off.len(), 1);
            self.rt_sorted &= other.rt_sorted;
            return Ok(());
        }
        let axis_count = self.axis_off.len() - 1;
        let n_axes = other.axis_off.len() - 1;
        // The same ceiling [`ChromStore::axis_for`] enforces at mint time, re-checked here
        // because concatenation is the other way the count can reach the empty-trace
        // marker: ids run `0..n`, so `n` axes need `n - 1 < NO_AXIS`. Taken before the
        // dedup below, which can only lower the count.
        if axis_count as u64 + n_axes as u64 > NO_AXIS as u64 {
            anyhow::bail!(
                "chromatogram table has more than {NO_AXIS} distinct retention-time axes; \
                 the axis id would collide with the empty-trace marker"
            );
        }
        // The leading rows of `other` that continue this store's open candidate. Axes are
        // minted in row order, so the axes those rows own are exactly `other`'s ids
        // `0..prefix_axes`, and no later row of `other` can reuse one of this store's axes:
        // a candidate change resets the dedup window in the single pass too.
        let (mut prefix_rows, mut prefix_axes) = (0usize, 0usize);
        if let Some(open) = self.open_cid {
            while prefix_rows < other.nrows() && other.cid[prefix_rows] == open {
                let a = other.axis_id[prefix_rows];
                if a != NO_AXIS {
                    prefix_axes = prefix_axes.max(a as usize + 1);
                }
                prefix_rows += 1;
            }
        }
        // `other`'s axis ids, in this store's numbering. A prefix axis that the open window
        // already holds maps onto it and is not copied; everything else is minted in
        // `other`'s order, which is the order the single pass mints it in.
        let mut minted_prefix = 0usize;
        let mut map: Vec<u32> = Vec::with_capacity(n_axes);
        for a in 0..n_axes {
            let vals = &other.axis_vals[other.axis_off[a]..other.axis_off[a + 1]];
            let reuse = if a < prefix_axes {
                self.open_axis_matching(vals)
            } else {
                None
            };
            match reuse {
                Some(existing) => map.push(existing),
                None => {
                    map.push((self.axis_off.len() - 1) as u32);
                    self.axis_vals.extend_from_slice(vals);
                    self.axis_off.push(self.axis_vals.len());
                    self.axis_strict.push(other.axis_strict[a]);
                    minted_prefix += usize::from(a < prefix_axes);
                }
            }
        }
        let name_map: Vec<u32> = other
            .names
            .names
            .iter()
            .map(|n| self.names.intern(n))
            .collect();
        let int_base = self.int_vals.len();
        self.int_off
            .extend(other.int_off[1..].iter().map(|o| int_base + o));
        self.int_vals.extend_from_slice(&other.int_vals);
        self.cid.extend_from_slice(&other.cid);
        self.pred.extend_from_slice(&other.pred);
        self.name_id
            .extend(other.name_id.iter().map(|&id| name_map[id as usize]));
        self.axis_id.extend(other.axis_id.iter().map(|&id| {
            if id == NO_AXIS {
                NO_AXIS
            } else {
                map[id as usize]
            }
        }));
        // A deduped axis was already counted by whichever store holds the copy, and an
        // identical trace answers [`rt_is_sorted`] identically, so the conjunction stands.
        self.rt_sorted &= other.rt_sorted;
        if prefix_rows < other.nrows() {
            // `other`'s last candidate opened its window at or after the end of the prefix,
            // so its start maps by the offset the prefix's dedup left behind. When the
            // whole span continued the open candidate instead, the window it deduped
            // against is still the right one and both fields stay as they are.
            debug_assert!(other.open_axis_lo >= prefix_axes);
            self.open_cid = other.open_cid;
            self.open_axis_lo = axis_count + minted_prefix + (other.open_axis_lo - prefix_axes);
        }
        Ok(())
    }

    fn axis(&self, id: u32) -> &[f32] {
        if id == NO_AXIS {
            return &[];
        }
        let a = id as usize;
        &self.axis_vals[self.axis_off[a]..self.axis_off[a + 1]]
    }

    fn rt(&self, row: usize) -> &[f32] {
        self.axis(self.axis_id[row])
    }

    fn inten(&self, row: usize) -> &[f32] {
        &self.int_vals[self.int_off[row]..self.int_off[row + 1]]
    }

    fn name(&self, row: usize) -> &str {
        self.names.get(self.name_id[row])
    }

    /// Payload bytes by part, for [`crate::memlog::report`].
    fn mem_parts(&self) -> Vec<(&'static str, usize)> {
        use crate::memlog::bytes_of;
        vec![
            ("intensities", bytes_of(&self.int_vals)),
            ("rt_axes", bytes_of(&self.axis_vals)),
            (
                "row_index",
                bytes_of(&self.cid)
                    + bytes_of(&self.name_id)
                    + bytes_of(&self.pred)
                    + bytes_of(&self.axis_id)
                    + bytes_of(&self.int_off),
            ),
            (
                "axis_index",
                bytes_of(&self.axis_off) + bytes_of(&self.axis_strict),
            ),
        ]
    }
}

/// The fragment chromatogram rows grouped by candidate: candidate `ci` owns the rows
/// `rows[cand_off[ci]..cand_off[ci + 1]]` of [`ChromStore`], ascending by candidate id and
/// in table order within a candidate. That is exactly the iteration the
/// `BTreeMap<u32, Vec<usize>>` this replaces produced, without a map node and a `Vec` per
/// candidate; the per-candidate results below (windows, areas) are flat arrays indexed by
/// the same `ci`.
struct CandIndex {
    /// Distinct candidate ids, ascending.
    cids: Vec<u32>,
    cand_off: Vec<usize>,
    rows: Vec<usize>,
    /// `pred[slot]` for the row at `rows[slot]`, so the fragment-selection ranking reads
    /// one contiguous slice per candidate instead of gathering pairs.
    slot_pred: Vec<f32>,
}

impl CandIndex {
    fn build(store: &ChromStore) -> CandIndex {
        // The table is written grouped by candidate, so a one-entry run cache resolves
        // almost every row; an ungrouped table falls back to the binary search and is
        // grouped just the same.
        let mut distinct: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut last: Option<u32> = None;
        for &c in &store.cid {
            if last != Some(c) {
                distinct.insert(c);
                last = Some(c);
            }
        }
        let mut cids: Vec<u32> = distinct.into_iter().collect();
        cids.sort_unstable();
        let mut cand_off = vec![0usize; cids.len() + 1];
        let mut cache: Option<(u32, usize)> = None;
        let lookup = |c: u32, cache: &mut Option<(u32, usize)>| -> usize {
            if let Some((lc, li)) = *cache {
                if lc == c {
                    return li;
                }
            }
            let i = cids
                .binary_search(&c)
                .expect("candidate id was collected above");
            *cache = Some((c, i));
            i
        };
        for &c in &store.cid {
            cand_off[lookup(c, &mut cache) + 1] += 1;
        }
        for k in 1..=cids.len() {
            cand_off[k] += cand_off[k - 1];
        }
        let mut cursor = cand_off[..cids.len()].to_vec();
        let mut rows = vec![0usize; store.nrows()];
        let mut slot_pred = vec![0.0f32; store.nrows()];
        cache = None;
        for (row, &c) in store.cid.iter().enumerate() {
            let ci = lookup(c, &mut cache);
            rows[cursor[ci]] = row;
            slot_pred[cursor[ci]] = store.pred[row];
            cursor[ci] += 1;
        }
        CandIndex {
            cids,
            cand_off,
            rows,
            slot_pred,
        }
    }

    fn len(&self) -> usize {
        self.cids.len()
    }

    fn slots(&self, ci: usize) -> std::ops::Range<usize> {
        self.cand_off[ci]..self.cand_off[ci + 1]
    }

    fn rows_of(&self, ci: usize) -> &[usize] {
        &self.rows[self.slots(ci)]
    }

    fn find(&self, cid: u32) -> Option<usize> {
        self.cids.binary_search(&cid).ok()
    }
}

/// Index of the sample nearest to `target`, reproducing the first-minimum tie-break of a
/// forward linear scan (`d < best`). `sorted` asserts the trace is non-decreasing and
/// NaN-free ([`ChromStore::rt_sorted`], or the union axis built by [`peak_window`]), and
/// only then is the answer found by binary search; otherwise the scan runs as before.
///
/// The scan was the dominant per-fragment cost of a fixed-window integration: it is
/// `O(trace)` where the integration itself is `O(window)`, and on a window-grid
/// chromatogram the trace is a few thousand samples against a window of ten.
fn nearest_index(rt: &[f32], target: f64, sorted: bool) -> usize {
    if !sorted {
        let mut k = 0usize;
        let mut best = f64::INFINITY;
        for (i, &r) in rt.iter().enumerate() {
            let d = (r as f64 - target).abs();
            if d < best {
                best = d;
                k = i;
            }
        }
        return k;
    }
    // The binary search below needs a FINITE target, and the earlier claim that any
    // non-finite one "makes every predicate false and lands on 0" is wrong for `+inf`:
    // `r < inf` holds at every sample, so `partition_point` returns `rt.len()` and the
    // last index comes back, where the scan returns 0 (its `d` is `inf` everywhere and
    // `inf < inf` is false, so `k` is never assigned). NaN and `-inf` do land on 0 by
    // themselves; the precondition that actually holds is "finite", so it is stated once
    // here instead of argued per case. Retention times reach this stage finite (convert
    // drops a spectrum whose scan start time is not, `convert.rs`), and both call sites
    // filter the apex hint on `is_finite`, so this is the guard for the third caller.
    // Returning 0 is what the scan returns for every non-finite target.
    if !target.is_finite() {
        return 0;
    }
    // First sample at or after `target`.
    let j = rt.partition_point(|&r| (r as f64) < target);
    // Backing up over an equal-valued run keeps the FIRST of several identical samples,
    // which is the one the scan's strict `<` kept.
    let back = |mut m: usize| {
        while m > 0 && rt[m - 1] == rt[m] {
            m -= 1;
        }
        m
    };
    if j == 0 {
        return 0;
    }
    if j >= rt.len() {
        return back(rt.len() - 1);
    }
    // Sorted, so the distance is V-shaped and the minimum is at `j - 1` or `j`; an exact
    // tie is the equidistant case, where the scan kept the earlier index.
    if (target - rt[j - 1] as f64).abs() <= (rt[j] as f64 - target).abs() {
        back(j - 1)
    } else {
        j
    }
}

/// Trapezoidal integral of an intensity trace over RT (seconds). A single point
/// yields its raw intensity.
fn trapezoid(rt: &[f32], inten: &[f32]) -> f64 {
    if rt.len() < 2 {
        return inten.first().copied().unwrap_or(0.0) as f64;
    }
    let mut area = 0.0f64;
    for i in 0..rt.len() - 1 {
        let dt = (rt[i + 1] - rt[i]) as f64;
        area += dt * (inten[i] + inten[i + 1]) as f64 * 0.5;
    }
    area
}

/// Apex-outward interference-correction envelope: walking outward from the apex
/// (the max sample) in each direction, cap every sample at the running minimum so
/// far. A co-eluting interferent that lifts the peak wings back up is clipped to
/// the trough between it and the true peak, so it no longer inflates the
/// integrated area. Returns the corrected intensities aligned to the input.
/// Behavior-preserving when there is no wing interference (a clean monotone peak
/// is unchanged). Deterministic.
fn center_envelope_1d(inten: &[f32]) -> Vec<f32> {
    let n = inten.len();
    if n < 3 {
        return inten.to_vec();
    }
    let apex = (0..n).fold(0usize, |b, i| if inten[i] > inten[b] { i } else { b });
    let mut out = inten.to_vec();
    let mut m = inten[apex];
    for i in (0..apex).rev() {
        m = m.min(inten[i]);
        out[i] = m;
    }
    let mut m = inten[apex];
    for i in (apex + 1)..n {
        m = m.min(inten[i]);
        out[i] = m;
    }
    out
}

/// Trapezoidal integral of one fragment trace restricted to RT in `[lo, hi]`.
/// Reuses [`trapezoid`] on the in-window samples so the single-sample rule is
/// identical; an empty window integrates to 0. When `envelope` is set, the
/// in-window intensities are passed through [`center_envelope_1d`] first to strip
/// co-eluting interference in the peak wings before integration.
fn trapezoid_window(rt: &[f32], inten: &[f32], lo: f64, hi: f64, envelope: bool) -> f64 {
    let mut wr: Vec<f32> = Vec::new();
    let mut wi: Vec<f32> = Vec::new();
    for k in 0..rt.len() {
        let r = rt[k] as f64;
        if r >= lo && r <= hi {
            wr.push(rt[k]);
            wi.push(inten[k]);
        }
    }
    if wr.is_empty() {
        return 0.0;
    }
    if envelope {
        let ev = center_envelope_1d(&wi);
        trapezoid(&wr, &ev)
    } else {
        trapezoid(&wr, &wi)
    }
}

/// Background level for a fixed-scan window `[lo, hi)`: the `quantile` quantile of
/// the intensities in the flanks (`flank` samples on each side, clipped to the
/// trace). Returns 0 when no flank sample exists.
fn flank_baseline(inten: &[f32], lo: usize, hi: usize, flank: usize, quantile: f64) -> f32 {
    let mut v: Vec<f32> = Vec::with_capacity(2 * flank);
    let fl = lo.saturating_sub(flank);
    v.extend_from_slice(&inten[fl..lo]);
    let fh = (hi + flank).min(inten.len());
    if hi < fh {
        v.extend_from_slice(&inten[hi..fh]);
    }
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let pos = ((v.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    v[pos.min(v.len() - 1)]
}

/// Sample index range `[lo, hi)` covered by a fixed window centred on the sample
/// nearest to `apex`. `half_s > 0` selects the samples within `half_s` seconds of
/// the apex (always at least that nearest sample) and overrides `half`, which
/// otherwise takes `half` scans on each side. `None` for an empty trace.
///
/// Shared by the integration below and by the applied-window contract in [`run`],
/// so the bounds reported for a quantity are the bounds it was integrated over. [`run`]
/// computes this ONCE per fragment row and uses it for both, where it used to compute the
/// same indices a second time to report the bounds.
///
/// `sorted` says the trace is non-decreasing and NaN-free (see [`nearest_index`]).
fn fixed_window_indices(
    rt: &[f32],
    apex: f64,
    half: usize,
    half_s: f64,
    sorted: bool,
) -> Option<(usize, usize)> {
    if rt.is_empty() {
        return None;
    }
    let k = nearest_index(rt, apex, sorted);
    // The scan this replaced left `best` at infinity when every distance was NaN (a
    // non-finite apex, which makes the guard below reject the window); reproduce that
    // rather than letting a NaN slip past the comparison as false.
    let d = (rt[k] as f64 - apex).abs();
    let best = if d.is_nan() { f64::INFINITY } else { d };
    let (mut lo, mut hi) = if half_s > 0.0 {
        // Distance guard: the nearest sample must itself lie inside the window. Without
        // it the nearest sample was always included however far away it was, so a weak
        // fragment whose only samples sit far off the apex contributed its off-peak
        // intensity as this candidate's "area" -- where the equivalent RT-bounded path
        // (`trapezoid_window`) correctly integrates an empty window to 0.
        if best > half_s {
            return None;
        }
        let mut lo = k;
        while lo > 0 && (apex - rt[lo - 1] as f64) <= half_s {
            lo -= 1;
        }
        let mut hi = k + 1;
        while hi < rt.len() && (rt[hi] as f64 - apex) <= half_s {
            hi += 1;
        }
        (lo, hi)
    } else {
        (k.saturating_sub(half), (k + half + 1).min(rt.len()))
    };
    // Guarantee at least two samples when the trace can supply them, exactly as
    // `peak_window` does for the walked bounds: `trapezoid` falls back to returning the
    // first intensity for a shorter slice, so a one-sample window reports a HEIGHT where
    // every other candidate reports an area (intensity x seconds). Mixing those units in
    // one run corrupts relative and LFQ quantities, which is a worse error than
    // integrating one scan more than the nominal window asked for. Reachable when the
    // cycle time approaches the window width, or on a sparse observed-scan trace
    // (`extract.emit_window_grid = false`).
    if hi - lo < 2 && rt.len() >= 2 {
        if hi < rt.len() {
            hi += 1;
        } else {
            lo = lo.saturating_sub(1);
        }
    }
    Some((lo, hi))
}

/// True when `rt` is non-decreasing, non-negative and NaN-free, the property
/// [`nearest_index`] needs before it may binary-search. [`run`] takes it once per store
/// ([`ChromStore::rt_sorted`]) rather than per call.
fn rt_is_sorted(rt: &[f32]) -> bool {
    rt.is_empty() || (rt[0] >= 0.0 && rt.windows(2).all(|w| w[0] <= w[1]))
}

/// Fixed-window integration over the samples chosen by [`fixed_window_indices`],
/// with optional apex-outward envelope and optional flank-baseline subtraction
/// (`baseline = Some((flank, quantile))`). Empty trace integrates to 0.
///
/// Test-facing composition of [`fixed_window_indices`] and [`trapezoid_fixed_at`]; [`run`]
/// calls the two separately so the window indices are computed once and serve both the
/// area and the reported bounds.
#[cfg(test)]
fn trapezoid_fixed_opts(
    rt: &[f32],
    inten: &[f32],
    apex: f64,
    half: usize,
    half_s: f64,
    envelope: bool,
    baseline: Option<(usize, f64)>,
) -> f64 {
    let Some((lo, hi)) = fixed_window_indices(rt, apex, half, half_s, rt_is_sorted(rt)) else {
        return 0.0;
    };
    trapezoid_fixed_at(rt, inten, lo, hi, envelope, baseline)
}

/// Fixed-window integration over the sample range `[lo, hi)` already chosen by
/// [`fixed_window_indices`], with optional apex-outward envelope and optional
/// flank-baseline subtraction (`baseline = Some((flank, quantile))`).
fn trapezoid_fixed_at(
    rt: &[f32],
    inten: &[f32],
    lo: usize,
    hi: usize,
    envelope: bool,
    baseline: Option<(usize, f64)>,
) -> f64 {
    let mut w: Vec<f32> = inten[lo..hi].to_vec();
    if let Some((flank, quantile)) = baseline {
        let b = flank_baseline(inten, lo, hi, flank, quantile);
        for x in w.iter_mut() {
            *x = (*x - b).max(0.0);
        }
    }
    if envelope {
        w = center_envelope_1d(&w);
    }
    trapezoid(&rt[lo..hi], &w)
}

/// Top-N sum with the fragment ranking chosen by `selection`. `observed_area`
/// delegates to [`summarize_fragment_areas`] (legacy, byte-identical); `predicted`
/// ranks the positive finite areas by library intensity and sums the top N.
///
/// The areas and their library intensities arrive as two parallel slices of one
/// candidate's fragment rows rather than a `Vec<(f64, f32)>`, because [`run`] holds them
/// as flat per-candidate slices of one buffer each. `None` is a candidate with no
/// chromatogram row at all.
fn select_fragment_areas(
    areas: Option<(&[f64], &[f32])>,
    top_n: usize,
    selection: FragmentSelection,
) -> (Option<f64>, usize, &'static str) {
    match selection {
        FragmentSelection::ObservedArea => summarize_fragment_areas(areas.map(|a| a.0), top_n),
        FragmentSelection::Predicted => {
            let Some((areas, preds)) = areas else {
                return (None, 0, "no_fragment_traces");
            };
            let mut positive: Vec<(f64, f32)> = areas
                .iter()
                .copied()
                .zip(preds.iter().copied())
                // The ranking key must be finite too. `total_cmp` orders NaN ABOVE every
                // real value, so in the descending sort below a NaN predicted intensity
                // reached the front and was preferentially selected into the top N. The
                // absent-column case is unaffected: it substitutes 0.0, which is finite,
                // so every key ties and the area tie-break decides, as documented.
                .filter(|(area, pred)| area.is_finite() && *area > 0.0 && pred.is_finite())
                .collect();
            if positive.is_empty() {
                return (None, 0, "no_positive_fragment_area");
            }
            if top_n == 0 {
                return (None, 0, "no_fragments_selected");
            }
            positive.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.0.total_cmp(&a.0)));
            let used = positive.len().min(top_n);
            let quantity: f64 = positive.iter().take(used).map(|x| x.0).sum();
            if quantity.is_finite() && quantity > 0.0 {
                (Some(quantity), used, "quantified")
            } else {
                (None, used, "nonfinite_quantity")
            }
        }
    }
}

/// Elution-peak RT window `[lo, hi]` for one candidate, from the summed XIC across
/// all its fragment chromatograms. Fragments are aligned on the union of their RT
/// samples ordered by the f32 RT bit pattern: for the non-negative
/// RTs here the bit order matches the value order, so both the union axis and the
/// f64 summation order are fixed (determinism,
/// docs/14_build_test_deploy_gotchas.md). When a finite identification apex is
/// available, the nearest sampled RT anchors the outward
/// [`super::features::peak_bounds`] walk. This prevents a brighter off-apex
/// interferent from moving quantification to a different peak. Older scored
/// artifacts and missing/non-finite hints retain the legacy co-elution apex
/// detector. Returns an unbounded window when there are fewer than two distinct
/// RT samples (nothing to bound).
///
/// The union used to be built by inserting every sample into TWO `BTreeMap`s keyed
/// identically, one tree walk per sample each. It is now either a direct accumulation
/// (every row of the candidate shares one strictly increasing axis, which is what
/// extract's window grid writes, so the union IS that axis) or one stable sort of the
/// samples by RT bits. Both reduce each RT's f64 sum in the same order the tree did --
/// row order, then sample order -- so the profile, the apex and the bounds are
/// bit-identical.
fn peak_window(
    rows: &[usize],
    store: &ChromStore,
    frac: f64,
    grace: usize,
    apex_hint: Option<f64>,
) -> (f64, f64, f64) {
    // Does every non-empty row of this candidate sample the same strictly increasing
    // axis? An empty trace contributes no sample and so cannot widen the union.
    let mut shared: Option<u32> = None;
    let mut uniform = true;
    for &i in rows {
        let a = store.axis_id[i];
        if a == NO_AXIS {
            continue;
        }
        match shared {
            None => shared = Some(a),
            Some(s) if s == a => {}
            _ => {
                uniform = false;
                break;
            }
        }
    }
    let shared = shared.filter(|&a| uniform && store.axis_strict[a as usize]);

    let mut owned_axis: Vec<f32> = Vec::new();
    let (prof, cnt) = if let Some(a) = shared {
        let n = store.axis(a).len();
        let mut prof = vec![0.0f64; n];
        let mut cnt = vec![0u32; n];
        for &i in rows {
            for (k, &v) in store.inten(i).iter().enumerate() {
                prof[k] += v as f64;
                if v > 0.0 {
                    cnt[k] += 1;
                }
            }
        }
        (prof, cnt)
    } else {
        let total: usize = rows.iter().map(|&i| store.inten(i).len()).sum();
        let mut samples: Vec<(u32, f32)> = Vec::with_capacity(total);
        for &i in rows {
            let rts = store.rt(i);
            let ins = store.inten(i);
            for k in 0..rts.len() {
                samples.push((rts[k].to_bits(), ins[k]));
            }
        }
        // Stable, so within one RT the contributions keep their (row, sample) order --
        // the order the map accumulated them in.
        samples.sort_by_key(|s| s.0);
        let mut prof: Vec<f64> = Vec::new();
        let mut cnt: Vec<u32> = Vec::new();
        let mut k = 0usize;
        while k < samples.len() {
            let bits = samples[k].0;
            let mut sum = 0.0f64;
            let mut c = 0u32;
            while k < samples.len() && samples[k].0 == bits {
                sum += samples[k].1 as f64;
                if samples[k].1 > 0.0 {
                    c += 1;
                }
                k += 1;
            }
            owned_axis.push(f32::from_bits(bits));
            prof.push(sum);
            cnt.push(c);
        }
        (prof, cnt)
    };
    let axis: &[f32] = match shared {
        Some(a) => store.axis(a),
        None => &owned_axis,
    };
    if axis.len() < 2 {
        // Nothing to bound; apex is the lone RT if present, else NaN.
        let apex = axis.first().map_or(f64::NAN, |&r| r as f64);
        return (f64::NEG_INFINITY, f64::INFINITY, apex);
    }
    // [`nearest_index`] may binary-search only an ascending axis. The shared one was
    // checked strictly increasing when it was stored ([`ChromStore::axis_strict`]). The
    // merged one is ordered by BIT PATTERN, which is value order only while no RT is
    // negative or NaN -- and the previous test, "the last sample is `>= 0.0`", passes for
    // `-0.0`, whose bits sort after every positive f32. An axis ending in `-0.0` was
    // therefore declared sorted while descending at its last step, and the search and the
    // scan then disagreed on `ai`, which is the apex index that sizes the integration
    // window. Check the order itself instead: one pass over an axis the sort above already
    // paid O(n log n) for, and it needs no argument about the bit layout. Strict `<` is
    // right because the merge already deduped by bits, so the only way two neighbours can
    // compare equal is `0.0` against `-0.0`.
    let axis_sorted = shared.is_some() || axis.windows(2).all(|w| w[0] < w[1]);
    let ai = if let Some(hint) = apex_hint.filter(|v| v.is_finite()) {
        // The identified apex need not exactly equal a chromatogram sample (for
        // example after serialization/calibration), so anchor to the nearest RT.
        nearest_index(axis, hint, axis_sorted)
    } else {
        // Legacy robust apex: among scans whose co-eluting-fragment count is
        // within 1 of the maximum ("-1 for robustness"), take the one with the
        // highest summed intensity. This rejects a lone tall interferent fragment
        // in favor of a region where many fragments co-elute. Falls back to the
        // summed argmax only if no scan has a fragment.
        let max_cnt = cnt.iter().copied().max().unwrap_or(0);
        let thresh = max_cnt.saturating_sub(1).max(1);
        let mut ai = 0usize;
        let mut best = f64::NEG_INFINITY;
        let mut found = false;
        for (i, &v) in prof.iter().enumerate() {
            if cnt[i] >= thresh && v > best {
                best = v;
                ai = i;
                found = true;
            }
        }
        if !found {
            best = f64::NEG_INFINITY;
            for (i, &v) in prof.iter().enumerate() {
                if v > best {
                    best = v;
                    ai = i;
                }
            }
        }
        ai
    };
    let (mut lo, mut hi) = super::features::peak_bounds(&prof, ai, frac, grace);
    // Guarantee a nonzero-width window. A near-1-scan summed XIC (both apex
    // shoulders below the threshold) collapses to lo==hi; trapezoid_window would
    // then hit the single-sample rule and return the apex HEIGHT, not an area
    // (intensity x seconds), mixing units against broad-peak peptides in the same
    // run and corrupting relative/LFQ quantities. Widen to the adjacent grid scans
    // so at least two samples are always integrated (prof.len() >= 2 here).
    if lo == hi {
        if hi + 1 < prof.len() {
            hi += 1;
        }
        lo = lo.saturating_sub(1);
    }
    (axis[lo] as f64, axis[hi] as f64, axis[ai] as f64)
}

fn finite_option(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

///
/// `is_decoy` is the one bit of the `label` column this stage ever tested (`label ==
/// "decoy"`, and anything that is neither spelling is treated as a target exactly as
/// before); it arrives from `TableFile::str_eq` rather than from a `String` per row.
fn passes_quant_filter(is_decoy: bool, q_value: f64, threshold: f64, transferred: bool) -> bool {
    if is_decoy {
        return false;
    }
    // A match-between-runs transfer is quantifiable on its own evidence: the MBR worker
    // has already applied `mbr.q_transfer` to accept it, and the q column selected here
    // is one MBR does not lower (see the call site). A decoy is still never quantified,
    // whatever its transfer status.
    transferred || (q_value.is_finite() && q_value <= threshold)
}

/// Select positive, finite fragment areas and sum the top N. Missing traces and
/// all-zero/non-finite traces are deliberately nullable, not quantitative zero:
/// zero would be indistinguishable from a measured biological absence and would
/// bias protein rollups and downstream ratios.
fn summarize_fragment_areas(
    areas: Option<&[f64]>,
    top_n: usize,
) -> (Option<f64>, usize, &'static str) {
    let Some(areas) = areas else {
        return (None, 0, "no_fragment_traces");
    };
    let mut positive: Vec<f64> = areas
        .iter()
        .copied()
        .filter(|area| area.is_finite() && *area > 0.0)
        .collect();
    if positive.is_empty() {
        return (None, 0, "no_positive_fragment_area");
    }
    if top_n == 0 {
        return (None, 0, "no_fragments_selected");
    }
    positive.sort_by(|a, b| b.total_cmp(a));
    let used = positive.len().min(top_n);
    let quantity: f64 = positive.iter().take(used).sum();
    if quantity.is_finite() && quantity > 0.0 {
        (Some(quantity), used, "quantified")
    } else {
        (None, used, "nonfinite_quantity")
    }
}

type ProteinBaseQuant = BTreeMap<String, BTreeMap<u32, f64>>;

/// Record one identified row for protein rollup. Multiple charge/mod precursor
/// rows belonging to the same base peptide contribute only their maximum
/// quantity, preventing repeated identifications from inflating Top-N. This max
/// is a single-run representative only; proper cross-run abundance estimation
/// must combine per-run quant tables rather than roll pooled scored rows here.
fn add_protein_base_quantity(
    groups: &mut ProteinBaseQuant,
    protein_group: &str,
    base_peptide_id: u32,
    quantity: Option<f64>,
) {
    // `entry` needs an owned key, so this allocated a `String` for every scored row even
    // though the group almost always exists already. Look it up first and allocate only
    // for a group that is new. The group is still created for an unquantifiable row: a
    // protein whose every peptide is unquantifiable is REPORTED, with status
    // `no_quantifiable_peptide`, not omitted.
    if !groups.contains_key(protein_group) {
        groups.insert(protein_group.to_string(), BTreeMap::new());
    }
    let bases = groups
        .get_mut(protein_group)
        .expect("the group was just inserted");
    if let Some(quantity) = quantity.filter(|v| v.is_finite() && *v > 0.0) {
        bases
            .entry(base_peptide_id)
            .and_modify(|current| *current = current.max(quantity))
            .or_insert(quantity);
    }
}

/// Roll up unique quantifiable base peptides. `n_peptides` is the number of
/// unique positive bases before Top-N truncation, not the number of precursor
/// rows and not the number selected into the sum.
fn rollup_protein_bases(
    bases: &BTreeMap<u32, f64>,
    rollup: RollupMethod,
    top_n: usize,
) -> (Option<f64>, usize, &'static str) {
    if bases.is_empty() {
        return (None, 0, "no_quantifiable_peptide");
    }
    let mut values: Vec<f64> = bases.values().copied().collect();
    values.sort_by(|a, b| b.total_cmp(a));
    let quantity: f64 = match rollup {
        RollupMethod::TopNSum => values.iter().take(top_n).sum(),
        RollupMethod::Sum => values.iter().sum(),
    };
    if quantity.is_finite() && quantity > 0.0 {
        (Some(quantity), bases.len(), "quantified")
    } else {
        (None, bases.len(), "no_quantifiable_peptide")
    }
}

/// Rows of the chromatogram table allowed to be in flight at once across the parallel
/// row-group readers, a cap on decoder work rather than on memory.
///
/// A row group is decoded into arrow arrays IN FULL before this stage's accepted-candidate
/// filter looks at a single row, so N readers hold N row groups of `rt` and `intensity`
/// whatever fraction of them is kept, and the per-span stores of one chunk wait alongside
/// them until they are merged.
///
/// A ROW COUNT DOES NOT BOUND BYTES. The row carries two list columns whose length is the
/// number of scans of the covering isolation window inside the candidate's RT window: 55.5
/// values, 444 bytes of f32, on the six-run HYE artifact, and tens of times that on a run
/// whose `w_rt` is wide or unbounded. 1 << 19 rows is 235 MB at the first and 8 GB at the
/// second, so the byte budget below is what actually bounds the buffers and this is only
/// the ceiling on how many groups may be decoded at once.
///
/// It is also why a table written before extract's `CHROM_ROW_GROUP_ROWS = 1 << 16`
/// (2026-09-04), whose row groups hold 1,048,576 rows, reads with a single reader: 32 of
/// those would be 14.9 GB, which trades a few seconds for a memory regression. One reader
/// is the floor in every case, and one reader is still one whole row group -- as the single
/// pass it falls back to also is.
const CHROM_ROWS_IN_FLIGHT: usize = 1 << 19;

/// Decoded arrow bytes the parallel readers may hold at once, estimated per row from the
/// file's own compressed size, which tracks the trace length as [`CHROM_ROWS_IN_FLIGHT`]
/// cannot. 256 MB is what 1 << 19 rows of the six-run HYE artifact came to, so the shipped
/// shape (65,536-row groups, 444 B per row) still plans the same eight readers; a run with
/// 30x longer traces falls to one reader instead of asking for 8 GB of buffers.
///
/// This budget is SPENT, not saved: the single reader it replaces held one 65,536-row group,
/// about 29 MB, so the parallel read costs about 204 MB of transient buffers on the HYE
/// shape, against the 60-105 MB of `String` spine the flat scored-column reads gave back.
/// The direction of quant's peak is therefore up, by roughly 100-145 MB on a stage that
/// already holds a ~355 MB store, in exchange for the 3-4x load measured by
/// `chromatogram_read_arms`. The per-span stores waiting to be merged are a third term:
/// negligible on the default path, where they hold the accepted few percent, and a second
/// copy of the chunk's rows under `--out-peak-bounds`, which keeps every row.
const CHROM_BYTES_IN_FLIGHT: u64 = 256 << 20;

/// Assumed decompression factor of the chromatogram parquet, for sizing readers off the
/// file's compressed size. Measured 4.5x on the six-run HYE artifact (802 MB of snappy to
/// 3.62 GB of f32); 6 is used so the estimate errs towards fewer readers.
const CHROM_DECOMPRESSION_FACTOR: u64 = 6;

/// Readers the byte budget allows for row groups of `widest` rows, from the file's
/// compressed bytes per row. `usize::MAX` when the size cannot be taken, which leaves
/// [`CHROM_ROWS_IN_FLIGHT`] in charge rather than refusing to plan.
fn chrom_byte_readers(path: &str, nrows: usize, widest: usize) -> usize {
    let Ok(meta) = std::fs::metadata(path) else {
        return usize::MAX;
    };
    if nrows == 0 {
        return usize::MAX;
    }
    let per_row = (meta.len() / nrows as u64).max(1) * CHROM_DECOMPRESSION_FACTOR;
    let per_group = per_row.saturating_mul(widest as u64).max(1);
    (CHROM_BYTES_IN_FLIGHT / per_group).max(1) as usize
}

/// Read the chromatogram table into one [`ChromStore`], row group by row group.
///
/// Row groups are disjoint, contiguous row spans of the file, so each is read into its own
/// store and the stores are concatenated in file order ([`ChromStore::append`]): the same
/// rows, in the same order, as the single pass this replaces. Two things follow, and they
/// are the whole point.
///
/// * The read runs in parallel, while both phases that consume the store already do
///   (`par_iter` over candidates). The load is most of quant's wall: on the six-file HYE
///   benchmark the stage is 9 s per file and the table is 802 MB of snappy decompressing
///   to 3.62 GB, all of it independent per row group.
/// * A row group whose `candidate_id` statistics lie outside the accepted range is never
///   opened. On an unbanded run every group holds an accepted candidate, so this does not
///   fire; on a banded run, where a band's accepted ids are a narrow slice of the library,
///   it skips whole groups from the footer alone.
///
/// It falls back to a single pass when the footer offers no usable layout, when there is
/// one row group, and when the groups are large enough that neither parallelism nor
/// pruning would apply -- in which case the per-span copy would be pure loss. Which of the
/// two runs is therefore a function of `threads` and of the writer's row-group size, and
/// neither may reach the output: [`ChromStore::append`] rebuilds the single pass's store
/// exactly, axis ids included, which is what makes that safe.
fn load_chromatograms(
    ch: &TableFile,
    has_pred: bool,
    keep_all: bool,
    wanted: &CidSet,
    path: &str,
    threads: usize,
) -> Result<ChromStore> {
    let cols: Vec<&str> = if has_pred {
        vec![
            "candidate_id",
            "frag_name",
            "predicted_intensity",
            "rt",
            "intensity",
        ]
    } else {
        vec!["candidate_id", "frag_name", "rt", "intensity"]
    };
    let Some((spans, readers)) = chrom_spans(ch, keep_all, wanted, threads, path) else {
        return load_chrom_span(ch, &cols, has_pred, keep_all, wanted, path);
    };
    let mut store = ChromStore::new();
    // Chunked rather than one `par_iter` over every span, because that bounds two things
    // at once: the arrow buffers in flight, and the number of finished per-span stores
    // waiting to be merged, which held all at once would be a second copy of the store.
    for chunk in spans.chunks(readers) {
        let parts: Vec<Result<ChromStore>> = chunk
            .par_iter()
            .map(|span| load_chrom_span(span, &cols, has_pred, keep_all, wanted, path))
            .collect();
        // Consumed in span order, so a malformed row is reported by the same error the
        // single pass reports and not by whichever thread happened to finish first.
        for part in parts {
            store.append(part?)?;
        }
    }
    Ok(store)
}

/// Plan the chromatogram read as row-group spans, with the number to read concurrently, or
/// `None` to read the file in one pass. `threads` is the caller's rayon pool width; it is
/// a parameter rather than a `rayon::current_num_threads()` call inside so the plan can be
/// asserted on any host.
///
/// Pruning trusts the writer's `candidate_id` statistics, which by the parquet
/// specification describe the group's NON-NULL values. A null `candidate_id` is outside
/// this table's contract -- extract writes the column from a `Vec<u32>`, which has no null
/// -- and the single pass does not read it as a candidate either: it takes
/// `a_cid.value(k)`, the raw slot behind the validity bitmap, whose content the decoder
/// does not define. Closing that last gap needs a null count from `row_group_stats`, which
/// is `mumdia-io`'s to add.
fn chrom_spans(
    ch: &TableFile,
    keep_all: bool,
    wanted: &CidSet,
    threads: usize,
    path: &str,
) -> Option<(Vec<TableFile>, usize)> {
    // Footer only: no column data is touched here.
    let stats = ch.row_group_stats("candidate_id").ok()?;
    if stats.len() < 2 {
        return None;
    }
    let widest = stats.iter().map(|s| s.rows).max().unwrap_or(0).max(1);
    let readers = (CHROM_ROWS_IN_FLIGHT / widest)
        .min(chrom_byte_readers(path, ch.nrows, widest))
        .clamp(1, threads.max(1));
    // `keep_all` (`--out-peak-bounds`) wants every candidate, so nothing may be skipped.
    let accepted = if keep_all { None } else { wanted.range() };
    let mut spans = Vec::with_capacity(stats.len());
    let mut pruned = 0usize;
    let mut first = 0usize;
    let mut probe: Option<(usize, usize)> = None;
    for s in &stats {
        let start = first;
        first += s.rows;
        if s.rows == 0 {
            continue;
        }
        probe = probe.or(Some((start, s.rows)));
        // A group outside the accepted id range holds no row this stage would store. Note
        // that the per-row rt/intensity length guard already runs only on kept rows (the
        // filter `continue`s before it), so skipping such a group withdraws no check.
        if let (Some((lo, hi)), Some(gmin), Some(gmax)) = (accepted, s.min, s.max) {
            if gmax < f64::from(lo) || gmin > f64::from(hi) {
                pruned += 1;
                continue;
            }
        }
        spans.push(ch.span(start, s.rows).ok()?);
    }
    // One reader and nothing skipped is the single pass plus a copy per span.
    if readers < 2 && pruned == 0 {
        return None;
    }
    // Every group skipped means the file holds nothing this stage would keep. Read ONE
    // group rather than the whole file: what a full read still does here is check the
    // projection and the column types, which one group checks just as well, while the
    // per-row rt/intensity guard runs only on KEPT rows and so was never going to fire.
    // The store is empty either way, and reading 802 MB to produce an empty store is the
    // case pruning exists for.
    if spans.is_empty() {
        let (start, rows) = probe?;
        spans.push(ch.span(start, rows).ok()?);
    }
    Some((spans, readers))
}

/// One pass over `tf` -- the whole file, or one row-group span of it -- into a new store.
///
/// The MS1 isotope XIC pseudo-traces (`frag_name` "ms1_*") are precursor channels, not
/// fragment ions, and neither the peak-window detection nor the top-N sum ever reads one,
/// so they are not stored at all rather than loaded and skipped later.
///
/// Zero, not NaN, for an absent `predicted_intensity`: the value is only a ranking key,
/// and `total_cmp` orders NaN above every real intensity, which would silently invert the
/// `predicted` ranking rather than fail.
fn load_chrom_span(
    tf: &TableFile,
    cols: &[&str],
    has_pred: bool,
    keep_all: bool,
    wanted: &CidSet,
    path: &str,
) -> Result<ChromStore> {
    let mut store = ChromStore::new();
    let reader = tf.batches(Some(cols), 4096)?;
    let sch = reader.schema();
    let ix = |n: &str| {
        sch.index_of(n)
            .map_err(|_| anyhow!("chromatogram table has no column '{n}'"))
    };
    let (i_cid, i_name, i_rt, i_int) = (
        ix("candidate_id")?,
        ix("frag_name")?,
        ix("rt")?,
        ix("intensity")?,
    );
    let i_pred = if has_pred {
        Some(ix("predicted_intensity")?)
    } else {
        None
    };
    // Trace values are appended into scratch buffers and then copied into the store,
    // because the RT axis is deduplicated against the candidate's earlier rows before
    // it is stored.
    let mut rt_buf: Vec<f32> = Vec::new();
    let mut int_buf: Vec<f32> = Vec::new();
    for b in reader {
        let b = b?;
        let a_cid = b
            .column(i_cid)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| anyhow!("column 'candidate_id' is not u32"))?;
        let a_name = b
            .column(i_name)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("column 'frag_name' is not utf8"))?;
        let a_rt = ListF32::of(b.column(i_rt), "rt")?;
        let a_int = ListF32::of(b.column(i_int), "intensity")?;
        let a_pred = match i_pred {
            Some(i) => Some(
                b.column(i)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("column 'predicted_intensity' is not f32"))?,
            ),
            None => None,
        };
        for k in 0..b.num_rows() {
            let c = a_cid.value(k);
            if !keep_all && !wanted.contains(c) {
                continue;
            }
            let nm = if a_name.is_null(k) {
                ""
            } else {
                a_name.value(k)
            };
            rt_buf.clear();
            int_buf.clear();
            a_rt.append_row(k, &mut rt_buf, "rt")?;
            a_int.append_row(k, &mut int_buf, "intensity")?;
            let (rt, it) = (&rt_buf, &int_buf);
            // `rt` and `intensity` are two independent list columns, and every
            // integration below slices `intensity` with indices computed from the LENGTH
            // OF `rt`. Extract writes them from paired vectors so they always match, but
            // a chromatograms table is path-addressable: `mumdia quant --chromatograms`
            // accepts one written by anything, and a shorter intensity trace would panic
            // with a slice-index message naming no candidate. Checked here, per row,
            // while the row can still be named.
            if rt.len() != it.len() {
                anyhow::bail!(
                    "chromatogram row for candidate_id {c} has {} retention-time points \
                     but {} intensity points; every integration window is derived from \
                     the retention-time trace and applied to the intensity trace, so the \
                     two must be the same length in {}",
                    rt.len(),
                    it.len(),
                    path
                );
            }
            // The MS1 isotope XIC pseudo-traces are precursor channels, not fragment
            // ions, and no phase below reads one, so they are dropped rather than
            // stored and skipped later. Dropped AFTER the length check, which is a
            // guard on the table rather than on what this stage happens to consume.
            if nm.starts_with("ms1_") {
                continue;
            }
            let pred = match a_pred {
                Some(a) => a.value(k),
                None => 0.0,
            };
            store.push(c, nm, pred, rt, it)?;
        }
    }
    Ok(store)
}

pub fn run(p: QuantParams) -> Result<(u64, u64)> {
    run_hashed(p).map(|w| (w.peptide.rows, w.protein.rows))
}

/// The quant outputs as written and reported: row count and report content hash of each.
#[derive(Clone, Debug)]
pub struct QuantWritten {
    pub peptide: Written,
    pub protein: Written,
    /// `Some` exactly when `QuantParams::out_fragment` was set.
    pub fragment: Option<Written>,
}

/// [`run`], returning each output's row count and the content hash its report records, so
/// an orchestrator can record the artifacts without reading and hashing them again.
pub fn run_hashed(p: QuantParams) -> Result<QuantWritten> {
    let t0 = Instant::now();
    // No output may be one of the inputs (docs/31 F6).
    let inputs = [
        ("--psms-scored", p.psms_scored),
        ("--chromatograms", p.chromatograms),
    ];
    for out in [Some(p.out_peptide), Some(p.out_protein), p.out_fragment]
        .into_iter()
        .flatten()
    {
        mumdia_io::refuse_output_over_input(out, &inputs)?;
    }

    // Identified target PSMs below the peptide q threshold.
    //
    // Only the columns that decide which rows pass the filter are read for every row: the
    // candidate id, `label` (as the one bit its readers test), the q column and the MBR
    // flag. The identity columns that reach an output (`peptidoform`, `charge`,
    // `protein_group`, `base_peptide_id`) are read at the accepted rows alone, below, once
    // those rows are known. They used to be read whole and held for the whole stage, which
    // on a per-run split of the immunopeptidomics experiment (tens of millions of scored
    // rows, a few percent accepted) was most of quant's resident set before a
    // chromatogram was read. Same values at the rows that are used, same null and type
    // policy (`TableFile::str_flat_rows` and friends refuse a NULL anywhere in the column,
    // as the whole-column getters do).
    let ps = TableFile::open(p.psms_scored)?;
    let cid = ps.u32("candidate_id")?;
    let is_decoy = ps.str_eq("label", "decoy")?;
    // Q-value column to filter on. Peptide/precursor q is per-run only when the
    // rescore itself is single-run; grouped q-values are experiment-wide otherwise.
    // For per-run slices of a pooled rescore, run_psm_q is the run-local FDR gate.
    let pep_q = match p.cfg.q_filter {
        QuantQColumn::PeptideQ => ps.f64("peptide_q_value")?,
        QuantQColumn::PrecursorQ => ps.f64("precursor_q")?,
        QuantQColumn::PsmQ => ps.f64("q_value")?,
        QuantQColumn::RunPsmQ => ps.f64("run_psm_q")?,
    };
    // Match-between-runs acceptance, if this scored table has been through `mumdia mbr`.
    //
    // The MBR worker lowers the three PSM-level q columns on an accepted transfer, but
    // `q_filter` defaults to `peptide_q` and the report writers filter on
    // `peptide_q_value` / `pg_q_value`, none of which MBR touches. So `mumdia mbr`
    // followed by a manual quant with a default config reported MBR as having done
    // nothing at all. The orchestrator only avoided this by force-setting
    // `q_filter = psm_q`.
    //
    // Lowering the grouped columns instead would be the wrong fix: they are written to a
    // group's single winning row on purpose, so writing one on a transferred loser would
    // make the peptide and precursor counts double-count that group. A transferred row
    // has already passed `mbr.q_transfer` inside the worker, so the honest statement is
    // that acceptance is a second, independent route through the filter.
    let is_transferred: Vec<bool> = match ps.bool("is_transferred") {
        Ok(v) => v,
        Err(_) => vec![false; ps.nrows],
    };
    let n_transferred = is_transferred.iter().filter(|&&b| b).count();
    if n_transferred > 0 {
        info!(
            transferred_psms = n_transferred,
            "quant: including match-between-runs transfers, which the selected q column              does not itself reflect"
        );
    }
    // Every map in this stage is keyed on `candidate_id` alone, and `candidate_id` is a
    // library row index, so it repeats across runs. Handed a POOLED scored table
    // (`rescore --competed a b c ...`, which stamps `source` with the input table each
    // PSM came from) together with one run's chromatograms, quant would emit n_runs rows
    // per precursor, every one of them carrying that single run's quantity and apex: no
    // error, no warning, and every cross-run fold change exactly 1.00.
    //
    // `run-experiment` splits by `source` before calling quant, and the docs say to do
    // the same by hand, but nothing enforced it -- and the pooled table is precisely what
    // the recorded multi-run recipe produces. Refuse instead, naming the fix.
    if let Some(n_sources) = pooled_source_count(&ps, p.psms_scored)? {
        if n_sources > 1 {
            anyhow::bail!(
                "quant: {} names a pooled scored table covering {n_sources} runs \
                 (column `source` has {n_sources} distinct values), but quantification \
                 keys on candidate_id alone and takes a single run's chromatograms. \
                 Quantifying it as-is would emit one identical row per run and make every \
                 cross-run ratio 1.00. Split the table by `source` first (this is what \
                 `mumdia run-experiment` does internally), then run quant once per run \
                 with that run's own chromatograms.",
                p.psms_scored
            );
        }
    }

    // The rows that pass the quant filter, ascending: the only rows whose identity columns
    // reach an output, each visited once by the peptide and fragment tables below.
    let accepted: Vec<usize> = (0..ps.nrows)
        .filter(|&i| {
            passes_quant_filter(is_decoy[i], pep_q[i], p.cfg.q_threshold, is_transferred[i])
        })
        .collect();
    // The identity columns at those rows only (see the note at the top of the scored
    // read). `peptidoform` and `protein_group` stay flat (see [`FlatStr`]).
    let pform = FlatStr::read_rows(&ps, "peptidoform", &accepted)?;
    let charge = ps.i32_rows("charge", &accepted)?;
    let pg = FlatStr::read_rows(&ps, "protein_group", &accepted)?;
    let base = ps.u32_rows("base_peptide_id", &accepted)?;
    let acc_cid: Vec<u32> = accepted.iter().map(|&i| cid[i]).collect();
    drop(accepted);

    // Chromatograms, grouped by candidate, for the candidates quant actually uses.
    //
    // The chromatogram table is the largest artifact of a run (every extracted candidate,
    // two traces per fragment), but the outputs below consult only (a) the rows that pass
    // the quant filter and (b), in consensus mode, the reliable anchors whose windows set
    // the median half-widths; every other candidate's window and areas were computed and
    // discarded. So collect that candidate set first and keep only its rows while streaming
    // the table batch by batch. Each candidate's window and areas depend only on its own
    // rows (in stored order) and the consensus median only on the anchor set, so the outputs
    // are identical to reading everything, while the resident set is proportional to the
    // accepted rows (a few percent of the table). The peak-window diagnostic export is the
    // one consumer that wants every candidate, so it keeps the full read.
    let consensus_mode = p.cfg.bound_peak && p.cfg.peak_window_mode == PeakWindowMode::Consensus;
    // `label == "target"` is read only under consensus mode (the anchor set here and the
    // median half-widths below), so the second pass over the column is taken only there.
    // Empty otherwise, and indexed only behind `consensus_mode`.
    //
    // It IS a second decode of the column, which a single `Vec<String>` would not have
    // been. `label` holds two distinct values over the whole table, so it is dictionary
    // encoded and this costs one more pass over the smallest column in the artifact, for
    // `nrows` bools -- 879 KB on the six-run HYE table, against the ~21 MB of `String`
    // spine plus payload that one `str` read would have held for the whole stage. Three
    // spellings cannot be folded into one bit, and neither bit may be derived from the
    // other: a label that is neither spelling is a target to the filter and not an anchor.
    let is_target: Vec<bool> = if consensus_mode {
        ps.str_eq("label", "target")?
    } else {
        Vec::new()
    };
    let keep_all = p.out_peak_bounds.is_some() && p.cfg.bound_peak;
    let wanted: CidSet = if keep_all {
        CidSet::empty()
    } else {
        // Accepted ids, with duplicates: the scored table carries one row per candidate, so
        // this is the accepted row count (53,863 on the six-run HYE table, 215 KB) and a
        // repeat would only set the same bit twice. `CidSet::from_ids` walks it three times
        // -- min, max, fill -- which is three passes over that, not over the table.
        let mut ids: Vec<u32> = Vec::new();
        for i in 0..ps.nrows {
            if passes_quant_filter(is_decoy[i], pep_q[i], p.cfg.q_threshold, is_transferred[i])
                || (consensus_mode && is_target[i] && pep_q[i] <= p.cfg.reliable_q)
            {
                ids.push(cid[i]);
            }
        }
        CidSet::from_ids(&ids)
    };
    // The best target q per candidate, for the consensus anchors. Only the candidates whose
    // chromatograms are loaded are ever looked up (`index.cids` below), which outside
    // `keep_all` is a subset of `wanted`, so the map holds those alone: the same values at
    // every key that is read, without an entry per target row of the whole table.
    let q_by_cid: HashMap<u32, f64> = if consensus_mode {
        let mut m: HashMap<u32, f64> = HashMap::new();
        for i in 0..ps.nrows {
            if is_target[i] && (keep_all || wanted.contains(cid[i])) {
                let e = m.entry(cid[i]).or_insert(f64::INFINITY);
                if pep_q[i] < *e {
                    *e = pep_q[i];
                }
            }
        }
        m
    } else {
        HashMap::new()
    };
    // `psms_scored` carries the exact identification apex (schema psms_scored v4;
    // the column has been present since v3). Older artifacts, or rows whose apex is
    // null (read as NaN) or non-finite, carry no hint, and quant then re-detects the
    // apex from the chromatogram itself.
    //
    // That fallback is not equivalent: re-detection reproduces the identification's
    // apex only about half the time (CLAUDE.md, "the selected apex was historically
    // correct/strongest only about 48-52% of the time"), so a quantity integrated
    // around a re-detected apex can belong to a different peak than the one that was
    // identified. It must therefore be visible rather than silent: warn, and record
    // the coverage in the artifact report so a downstream reader can tell which apex
    // source a quantity actually used.
    //
    // The identification apex per candidate is the first finite `apex_rt` in row order.
    // It is kept only for the candidates it can be looked up for (the loaded ones, as for
    // `q_by_cid`; every candidate under `keep_all`). It used to be a map entry for every
    // scored row, which on a large per-run table was the stage's largest structure. The
    // report's `candidates_with_scored_apex` still counts the WHOLE table's candidates with
    // a finite apex, as it always did, through one bit per candidate id.
    let mut apex_by_cid: HashMap<u32, f64> = HashMap::new();
    let mut scored_apex = CidMarks::over(&cid);
    let apex_column_present = ps
        .visit_f64("apex_rt", |first, vals| {
            for (k, &a) in vals.iter().enumerate() {
                let c = cid[first + k];
                if a.is_finite() && scored_apex.insert(c) && (keep_all || wanted.contains(c)) {
                    apex_by_cid.insert(c, a);
                }
            }
            Ok(())
        })
        .is_ok();
    if !apex_column_present {
        // A column that is absent or not f64 carries no hint, exactly as before; nothing a
        // failed read left behind may count.
        apex_by_cid.clear();
        scored_apex = CidMarks::over(&[]);
        warn!(
            psms_scored = p.psms_scored,
            "quant: scored table has no apex_rt column (pre-v3 artifact); every              quantity will be integrated around a RE-DETECTED apex, which reproduces              the identification apex only about half the time"
        );
    }
    let candidates_with_scored_apex = scored_apex.count();
    drop(scored_apex);
    // Nothing below reads a column of the scored table per row any more: the outputs walk
    // the accepted rows, whose values were gathered above.
    drop((cid, is_decoy, pep_q, is_transferred, is_target));
    // `predicted_intensity` is OPTIONAL. Chromatogram artifacts written before that column
    // existed do not carry it, and the default `observed_area` ranking never reads it, so
    // probe the footer (which decodes no data) and project only what is present. Demanding
    // the column unconditionally made every older artifact unquantifiable.
    let has_pred = column_names(p.chromatograms)?
        .iter()
        .any(|c| c == "predicted_intensity");
    if !has_pred && p.cfg.fragment_selection == FragmentSelection::Predicted {
        anyhow::bail!(
            "quant.fragment_selection = predicted ranks fragments by the \
             `predicted_intensity` column, which {} does not carry. Re-run `extract` to \
             write a current chromatogram artifact, or set \
             quant.fragment_selection = observed_area.",
            p.chromatograms
        );
    }
    let ch = TableFile::open(p.chromatograms)?;
    // Flat, grouped-by-candidate store (see [`ChromStore`]), read row group by row group
    // in parallel (see [`load_chromatograms`]).
    let store = load_chromatograms(
        &ch,
        has_pred,
        keep_all,
        &wanted,
        p.chromatograms,
        rayon::current_num_threads(),
    )?;
    drop(wanted);
    // Group the fragment chromatogram rows by candidate, ascending by candidate id and in
    // table order within a candidate: the order the `BTreeMap<u32, Vec<usize>>` this
    // replaces iterated in, so every per-candidate reduction below runs over the same rows
    // in the same order.
    let index = CandIndex::build(&store);
    {
        let mut parts = store.mem_parts();
        parts.push((
            "cand_index",
            crate::memlog::bytes_of(&index.rows)
                + crate::memlog::bytes_of(&index.slot_pred)
                + crate::memlog::bytes_of(&index.cids)
                + crate::memlog::bytes_of(&index.cand_off),
        ));
        crate::memlog::report("quant chromatogram store", &parts);
    }
    info!(
        chromatogram_rows = store.nrows(),
        candidates = index.len(),
        "quant: chromatograms loaded"
    );
    // Optional peak-window diagnostic: (candidate_id, lo_rt, hi_rt) for finite windows.
    let emit_bounds = p.out_peak_bounds.is_some() && p.cfg.bound_peak;
    let (mut pb_cid, mut pb_lo, mut pb_hi) = (Vec::new(), Vec::new(), Vec::new());

    // Phase 1: per-candidate summed-XIC window (lo_rt, hi_rt, apex_rt), anchored at
    // the identification apex when available and otherwise using the legacy robust
    // co-elution detector. Indexed by candidate position in `index` for the consensus
    // estimate and for phase 2, which is the same ascending-candidate_id order the
    // `BTreeMap<u32, _>` this replaces iterated in, without its node per candidate.
    //
    // Each candidate's window depends only on its own chromatogram rows, so this is
    // embarrassingly parallel; a range is an indexed parallel iterator, so the collected
    // Vec is in candidate order and every float inside `peak_window` is still reduced per
    // candidate in the same order as before.
    let win: Vec<(f64, f64, f64)> = if p.cfg.bound_peak {
        (0..index.len())
            .into_par_iter()
            .map(|ci| {
                peak_window(
                    index.rows_of(ci),
                    &store,
                    p.cfg.peak_fraction,
                    p.cfg.peak_grace,
                    apex_by_cid.get(&index.cids[ci]).copied(),
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    // Consensus mode: peak width is a near-constant instrument/gradient property, so
    // take the median left/right half-width over CONFIDENT peptides (q <= reliable_q)
    // and apply it around each candidate's apex. The median ignores the interference-
    // stretched and collapsed per-candidate windows. It is estimated independently
    // for each quant invocation/run; cross-run-identical widths require an external
    // shared policy. Falls back to per-candidate if too few confident anchors.
    // Spelled `consensus_mode` rather than repeating the two-term condition, because that
    // is also the flag `is_target` was read under: they must not be able to drift apart.
    let consensus: Option<(f64, f64)> = if consensus_mode {
        // `q_by_cid` was taken with the scored columns, for the loaded candidates.
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for (ci, &(lo, hi, apex)) in win.iter().enumerate() {
            if lo.is_finite()
                && hi.is_finite()
                && apex.is_finite()
                && q_by_cid
                    .get(&index.cids[ci])
                    .is_some_and(|&q| q <= p.cfg.reliable_q)
            {
                left.push(apex - lo);
                right.push(hi - apex);
            }
        }
        if left.len() >= 20 {
            let ml = median_sorted(&mut left);
            let mr = median_sorted(&mut right);
            info!(
                anchors = left.len(),
                med_left_s = ml,
                med_right_s = mr,
                "quant: consensus peak window"
            );
            Some((ml, mr))
        } else {
            info!(
                anchors = left.len(),
                "quant: too few confident anchors, using per-candidate windows"
            );
            None
        }
    } else {
        None
    };

    // Phase 2: integrate each fragment over the chosen window and retain the
    // actually applied apex/bounds for the peptide-quant contract.
    //
    // Integrate each candidate's fragment traces. Parallel across candidates for the same
    // reason the peak-window phase above is: a candidate reads only its own chromatogram
    // rows and every float reduction happens inside one candidate's `trapezoid*` call.
    // The areas go into ONE flat buffer laid out exactly like `index.rows`, so candidate
    // `ci`'s areas are `area_by_slot[index.slots(ci)]` and nothing is allocated per
    // candidate. Rayon writes into disjoint sub-slices of it, carved once below, which is
    // also why the positional pairing the serial fold used to assert is gone: each
    // candidate writes the slice that is its own by construction.
    let want_fixed = p.cfg.fixed_scan_halfwidth > 0 || p.cfg.fixed_window_s > 0.0;
    let baseline = if p.cfg.baseline_subtract {
        Some((p.cfg.baseline_flank_scans, p.cfg.baseline_quantile))
    } else {
        None
    };
    let mut area_by_slot: Vec<f64> = vec![0.0; index.rows.len()];
    let mut slices: Vec<&mut [f64]> = Vec::with_capacity(index.len());
    {
        let mut rest: &mut [f64] = &mut area_by_slot;
        for ci in 0..index.len() {
            let n = index.cand_off[ci + 1] - index.cand_off[ci];
            let (head, tail) = rest.split_at_mut(n);
            slices.push(head);
            rest = tail;
        }
    }
    let applied_win: Vec<(f64, f64, f64)> = slices
        .par_iter_mut()
        .enumerate()
        .map(|(ci, out)| {
            let c = index.cids[ci];
            let rows = index.rows_of(ci);
            let (lo_rt, hi_rt, integration_apex) = if !p.cfg.bound_peak {
                // A fixed window needs only an apex to centre on, not the descent walk,
                // so take the identification apex directly. Previously this branch
                // returned NaN unconditionally, which made `fixed` below false and
                // silently integrated the WHOLE trace: `{"bound_peak": false,
                // "fixed_window_s": 5.0}` -- the natural way to ask for a pure fixed
                // window with no walk -- returned whole-chromatogram areas with no
                // warning. `Config::validate` warns about two neighbouring combinations
                // and not about this one.
                //
                // Still NaN when no fixed window is configured, because then the
                // integration really is unbounded and the reported
                // `integration_apex_rt` should say so rather than name a centre that
                // nothing was centred on.
                let apex = if want_fixed {
                    apex_by_cid.get(&c).copied().unwrap_or(f64::NAN)
                } else {
                    f64::NAN
                };
                (f64::NEG_INFINITY, f64::INFINITY, apex)
            } else {
                let (lo, hi, apex) = win[ci];
                match consensus {
                    Some((ml, mr)) if apex.is_finite() => (apex - ml, apex + mr, apex),
                    _ => (lo, hi, apex),
                }
            };
            // A fixed window replaces the walked bounds entirely; it needs a finite apex
            // to centre on, so an unknown apex falls back to the configured window.
            let fixed = want_fixed && integration_apex.is_finite();
            // Applied-window contract: under a fixed window the walked bounds are NOT the
            // integration range, so report the RT extent actually covered (union over this
            // candidate's traces, whose sample grids may differ). Otherwise
            // `integration_lo_rt`/`integration_hi_rt` and the peak-bounds diagnostic would
            // describe a window that produced no part of `quantity`. The window indices are
            // computed ONCE per row and serve both the area and this union; the union used
            // to recompute the identical indices in a second pass over the same rows.
            let mut flo = f64::INFINITY;
            let mut fhi = f64::NEG_INFINITY;
            for (slot, &i) in rows.iter().enumerate() {
                let rt = store.rt(i);
                let it = store.inten(i);
                out[slot] = if fixed {
                    match fixed_window_indices(
                        rt,
                        integration_apex,
                        p.cfg.fixed_scan_halfwidth,
                        p.cfg.fixed_window_s,
                        store.rt_sorted,
                    ) {
                        Some((lo, hi)) => {
                            flo = flo.min(rt[lo] as f64);
                            fhi = fhi.max(rt[hi - 1] as f64);
                            trapezoid_fixed_at(
                                rt,
                                it,
                                lo,
                                hi,
                                p.cfg.interference_envelope,
                                baseline,
                            )
                        }
                        None => 0.0,
                    }
                } else if p.cfg.bound_peak {
                    trapezoid_window(rt, it, lo_rt, hi_rt, p.cfg.interference_envelope)
                } else {
                    trapezoid(rt, it)
                };
            }
            let (lo_rt, hi_rt) = if fixed && flo.is_finite() && fhi.is_finite() {
                (flo, fhi)
            } else {
                (lo_rt, hi_rt)
            };
            (lo_rt, hi_rt, integration_apex)
        })
        .collect();
    drop(slices);
    let area_by_slot = area_by_slot;
    if emit_bounds {
        for (ci, &(lo_rt, hi_rt, _)) in applied_win.iter().enumerate() {
            if lo_rt.is_finite() && hi_rt.is_finite() {
                pb_cid.push(index.cids[ci]);
                pb_lo.push(lo_rt);
                pb_hi.push(hi_rt);
            }
        }
    }

    // Optional peak-window diagnostic export.
    if let Some(pbpath) = p.out_peak_bounds {
        let width: Vec<f64> = pb_lo.iter().zip(&pb_hi).map(|(l, h)| h - l).collect();
        write_table(
            pbpath,
            vec![
                Col::U32("candidate_id".into(), pb_cid),
                Col::F64("lo_rt".into(), pb_lo),
                Col::F64("hi_rt".into(), pb_hi),
                Col::F64("width_s".into(), width),
            ],
        )?;
    }

    // Per-peptidoform quantity = sum of the top-N positive fragment areas.
    let (
        mut q_cid,
        mut q_base,
        mut q_pform,
        mut q_z,
        mut q_pg,
        mut q_val,
        mut q_status,
        mut q_nfrag,
        mut q_apex,
        mut q_lo,
        mut q_hi,
    ) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut per_group: ProteinBaseQuant = BTreeMap::new();
    let mut n_quantified_peptides = 0u64;
    // The top-N selection depends on the candidate alone, so it is computed once per
    // candidate rather than once per scored row: it collects and SORTS the candidate's
    // areas, and every extra row mapping to the same candidate repeated that.
    let mut selected: HashMap<usize, usize> = HashMap::new();
    let mut selections: Vec<(Option<f64>, usize, &'static str)> = Vec::new();
    // One pass over the accepted rows, in table order, which is the order the filter over
    // every row visited them in.
    for (k, &c) in acc_cid.iter().enumerate() {
        let ci = index.find(c);
        let (quantity, used, status) = match ci {
            None => select_fragment_areas(None, p.cfg.top_n_fragments, p.cfg.fragment_selection),
            Some(ci) => {
                let slot = *selected.entry(ci).or_insert_with(|| {
                    let r = index.slots(ci);
                    selections.push(select_fragment_areas(
                        Some((&area_by_slot[r.clone()], &index.slot_pred[r])),
                        p.cfg.top_n_fragments,
                        p.cfg.fragment_selection,
                    ));
                    selections.len() - 1
                });
                selections[slot]
            }
        };
        if quantity.is_some() {
            n_quantified_peptides += 1;
        }
        let (integration_lo, integration_hi, integration_apex) = match ci.map(|ci| applied_win[ci])
        {
            Some((lo, hi, apex)) => (finite_option(lo), finite_option(hi), finite_option(apex)),
            None => (None, None, None),
        };
        q_cid.push(c);
        q_base.push(base[k]);
        q_pform.push(pform.get(k).to_string());
        q_z.push(charge[k]);
        q_pg.push(pg.get(k).to_string());
        q_val.push(quantity);
        q_status.push(status.to_string());
        q_nfrag.push(used as i32);
        q_apex.push(integration_apex);
        q_lo.push(integration_lo);
        q_hi.push(integration_hi);
        add_protein_base_quantity(&mut per_group, pg.get(k), base[k], quantity);
    }

    let n_pep = write_table(
        p.out_peptide,
        vec![
            Col::U32("candidate_id".into(), q_cid),
            Col::U32("base_peptide_id".into(), q_base),
            Col::Str("peptidoform".into(), q_pform),
            Col::I32("charge".into(), q_z),
            Col::Str("protein_group".into(), q_pg),
            Col::OptF64("quantity".into(), q_val),
            Col::Str("quant_status".into(), q_status),
            Col::I32("n_fragments_used".into(), q_nfrag),
            Col::OptF64("integration_apex_rt".into(), q_apex),
            Col::OptF64("integration_lo_rt".into(), q_lo),
            Col::OptF64("integration_hi_rt".into(), q_hi),
        ],
    )?;

    // Protein-group rollup over unique, quantifiable base peptides only.
    let (mut g_name, mut g_val, mut g_status, mut g_npep) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut n_quantified_protein_groups = 0u64;
    for (group, bases) in &per_group {
        let (quantity, n_bases, status) =
            rollup_protein_bases(bases, p.cfg.rollup, p.cfg.top_n_peptides);
        if quantity.is_some() {
            n_quantified_protein_groups += 1;
        }
        g_name.push(group.clone());
        g_val.push(quantity);
        g_status.push(status.to_string());
        g_npep.push(n_bases as i32);
    }
    let n_pg = write_table(
        p.out_protein,
        vec![
            Col::Str("protein_group".into(), g_name),
            Col::OptF64("quantity".into(), g_val),
            Col::Str("quant_status".into(), g_status),
            Col::I32("n_peptides".into(), g_npep),
        ],
    )?;

    // Optional per-fragment area export for ion-level directLFQ across runs.
    let mut fragment_output: Option<(&str, u64)> = None;
    if let Some(fpath) = p.out_fragment {
        let (mut f_cid, mut f_pf, mut f_z, mut f_pg, mut f_name, mut f_area) = (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        for (k, &c) in acc_cid.iter().enumerate() {
            // The fragment name comes from the interned table by row rather than from a
            // per-candidate `Vec<(&str, f64)>` built alongside the areas; the rows are the
            // same rows in the same order.
            if let Some(ci) = index.find(c) {
                for slot in index.slots(ci) {
                    let a = area_by_slot[slot];
                    if !a.is_finite() || a <= 0.0 {
                        continue;
                    }
                    f_cid.push(c);
                    f_pf.push(pform.get(k).to_string());
                    f_z.push(charge[k]);
                    f_pg.push(pg.get(k).to_string());
                    f_name.push(store.name(index.rows[slot]).to_string());
                    f_area.push(a);
                }
            }
        }
        let fragment_rows = write_table(
            fpath,
            vec![
                Col::U32("candidate_id".into(), f_cid),
                Col::Str("peptidoform".into(), f_pf),
                Col::I32("charge".into(), f_z),
                Col::Str("protein_group".into(), f_pg),
                Col::Str("fragment_name".into(), f_name),
                Col::F64("quantity".into(), f_area),
            ],
        )?;
        fragment_output = Some((fpath, fragment_rows));
    }

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("peptide_rows".to_string(), json!(n_pep));
    stats.insert(
        "quantified_peptides".to_string(),
        json!(n_quantified_peptides),
    );
    stats.insert(
        "nonquantifiable_peptides".to_string(),
        json!(n_pep.saturating_sub(n_quantified_peptides)),
    );
    stats.insert("protein_group_rows".to_string(), json!(n_pg));
    stats.insert(
        "quantified_protein_groups".to_string(),
        json!(n_quantified_protein_groups),
    );
    let report_params = json!({
        "q_threshold": p.cfg.q_threshold,
        "top_n_fragments": p.cfg.top_n_fragments,
        "fragment_selection": format!("{:?}", p.cfg.fragment_selection),
        "fixed_scan_halfwidth": p.cfg.fixed_scan_halfwidth,
        "fixed_window_s": p.cfg.fixed_window_s,
        "baseline_subtract": p.cfg.baseline_subtract,
        "baseline_flank_scans": p.cfg.baseline_flank_scans,
        "baseline_quantile": p.cfg.baseline_quantile,
        "top_n_peptides": p.cfg.top_n_peptides,
        "rollup": format!("{:?}", p.cfg.rollup),
        "bound_peak": p.cfg.bound_peak,
        "peak_fraction": p.cfg.peak_fraction,
        "peak_grace": p.cfg.peak_grace,
        "peak_window_mode": format!("{:?}", p.cfg.peak_window_mode),
        "reliable_q": p.cfg.reliable_q,
        "q_filter": format!("{:?}", p.cfg.q_filter),
        "config_hash": p.config_hash,
        "psms_scored": p.psms_scored,
        "chromatograms": p.chromatograms,
        // Which apex each quantity was integrated around: `scored_apex` rows reuse the
        // identification apex, `redetected` rows fell back to quant's own peak pick.
        "apex_rt_column_present": apex_column_present,
        "candidates_with_scored_apex": candidates_with_scored_apex,
    });
    let mut written: Vec<Written> = Vec::with_capacity(2);
    for (path, schema, rows) in [
        (p.out_peptide, artifact::PEPTIDE_QUANT, n_pep),
        (p.out_protein, artifact::PROTEIN_GROUP_QUANT, n_pg),
    ] {
        let report = ArtifactReport {
            logical_name: schema.0.to_string(),
            schema_name: schema.0.to_string(),
            schema_version: schema.1,
            stage: "quant".to_string(),
            rows,
            content_hash: mumdia_io::hash::blake3_file(path)?,
            params: report_params.clone(),
            stats: stats.clone(),
            model_identity: None,
            elapsed_ms: elapsed,
        };
        report.write_for(path)?;
        written.push(report.written());
    }
    let mut fragment_written: Option<Written> = None;
    if let Some((path, rows)) = fragment_output {
        let report = ArtifactReport {
            logical_name: artifact::FRAGMENT_QUANT.0.to_string(),
            schema_name: artifact::FRAGMENT_QUANT.0.to_string(),
            schema_version: artifact::FRAGMENT_QUANT.1,
            stage: "quant".to_string(),
            rows,
            content_hash: mumdia_io::hash::blake3_file(path)?,
            params: report_params,
            stats: stats.clone(),
            model_identity: None,
            elapsed_ms: elapsed,
        };
        report.write_for(path)?;
        fragment_written = Some(report.written());
    }

    info!(
        peptide_rows = n_pep,
        quantified_peptides = n_quantified_peptides,
        protein_group_rows = n_pg,
        quantified_protein_groups = n_quantified_protein_groups,
        elapsed_ms = elapsed,
        "quant: done"
    );
    let protein = written.pop().expect("two reports written");
    let peptide = written.pop().expect("two reports written");
    Ok(QuantWritten {
        peptide,
        protein,
        fragment: fragment_written,
    })
}

/// Combine several per-run quant tables into a protein-by-run abundance matrix
/// with MaxLFQ (peptide-level: `by_fragment=false`, reads `peptide_quant`) or
/// directLFQ (ion-level: `by_fragment=true`, reads `fragment_quant`). Each
/// protein's feature-by-run intensity matrix is passed to the ratio-alignment
/// core in [`crate::quant_lfq`]. Output is long form: protein_group, run,
/// quantity, n_features. With one input this reduces to the per-run sum.
///
/// `normalize` applies a cross-run size factor to the feature-by-run matrix
/// before rollup (see `size_factors`).
pub fn run_lfq_combine(
    inputs: &[String],
    by_fragment: bool,
    normalize: NormalizeMethod,
    out: &str,
) -> Result<u64> {
    use std::collections::BTreeMap;
    let n = inputs.len();
    // protein_group -> feature key -> per-run intensity
    let mut data: BTreeMap<String, BTreeMap<String, Vec<Option<f64>>>> = BTreeMap::new();
    for (ri, path) in inputs.iter().enumerate() {
        let t = TableFile::open(path)?;
        let pform = t.str("peptidoform")?;
        let z = t.i32("charge")?;
        let pgc = t.str("protein_group")?;
        let q = t.opt_f64("quantity")?;
        let fname = if by_fragment {
            Some(t.str("fragment_name")?)
        } else {
            None
        };
        for i in 0..t.nrows {
            let Some(quantity) = q[i].filter(|v| v.is_finite() && *v > 0.0) else {
                continue;
            };
            let key = match &fname {
                Some(fnm) => format!("{}|{}|{}", pform[i], z[i], fnm[i]),
                None => format!("{}|{}", pform[i], z[i]),
            };
            // Same reason as `add_protein_base_quantity`: `entry` would clone the protein
            // group name on every row, and a run's quant table repeats each group many
            // times over.
            if !data.contains_key(&pgc[i]) {
                data.insert(pgc[i].clone(), BTreeMap::new());
            }
            let slot = &mut data
                .get_mut(&pgc[i])
                .expect("the group was just inserted")
                .entry(key)
                .or_insert_with(|| vec![None; n])[ri];
            *slot = Some(slot.map_or(quantity, |previous| previous.max(quantity)));
        }
    }
    // Cross-run normalization: one global size factor per run, estimated from the
    // whole feature-by-run matrix and applied before protein rollup so both the
    // MaxLFQ profile and any downstream ratio inherit the corrected scale.
    let factors = size_factors(&data, n, normalize);
    if normalize != NormalizeMethod::None {
        for feats in data.values_mut() {
            for vec in feats.values_mut() {
                for r in 0..n {
                    if let Some(v) = vec[r] {
                        vec[r] = Some(v / factors[r]);
                    }
                }
            }
        }
    }
    let (mut c_pg, mut c_run, mut c_q, mut c_nf) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    // The rollup levels below read the SAME feature vectors, so each level borrows them
    // from `data` rather than cloning them: with the protein level and the two sibling
    // levels this held three or four copies of every feature vector at once.
    let mut mat: Vec<&Vec<Option<f64>>> = Vec::new();
    for (pgname, feats) in &data {
        mat.clear();
        mat.extend(feats.values());
        let prof = crate::quant_lfq::lfq_profile(&mat, n);
        for (r, &v) in prof.iter().enumerate() {
            c_pg.push(pgname.clone());
            c_run.push(r as i32);
            c_q.push(v);
            c_nf.push(feats.len() as i32);
        }
    }
    let rows = write_table(
        out,
        vec![
            Col::Str("protein_group".into(), c_pg),
            Col::I32("run".into(), c_run),
            Col::F64("quantity".into(), c_q),
            Col::I32("n_features".into(), c_nf),
        ],
    )?;

    // Peptide- and precursor-level matrices from the SAME normalized features,
    // written as sibling files next to the protein matrix (the protein output is
    // unchanged). Precursor = one (peptidoform, charge); peptide = one stripped
    // base sequence. Both roll their member features up with the same LFQ engine.
    // Purely additive analysis granularity; strictly post-FDR, no identification
    // or FDR change.
    let mut prec: BTreeMap<(String, i32), Vec<&Vec<Option<f64>>>> = BTreeMap::new();
    let mut pep: BTreeMap<String, Vec<&Vec<Option<f64>>>> = BTreeMap::new();
    for feats in data.values() {
        for (key, vec) in feats {
            // key = "peptidoform|charge" (maxlfq) or "peptidoform|charge|fragment"
            // (directlfq); peptidoform strings never contain '|'.
            let mut it = key.splitn(3, '|');
            let pform = it.next().unwrap_or("");
            let charge: i32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            prec.entry((pform.to_string(), charge))
                .or_default()
                .push(vec);
            pep.entry(base_sequence(pform)).or_default().push(vec);
        }
    }
    // (group key, charge, feature-by-run matrix) for one sibling-matrix level. The matrix
    // borrows its rows from `data`.
    type LevelGroup<'a> = (String, i32, Vec<&'a Vec<Option<f64>>>);
    let write_level = |path: String, groups: Vec<LevelGroup>| -> Result<()> {
        let (mut g_key, mut g_z, mut g_run, mut g_q, mut g_nf) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (key, z, mat) in &groups {
            let prof = crate::quant_lfq::lfq_profile(mat, n);
            for (r, &v) in prof.iter().enumerate() {
                g_key.push(key.clone());
                g_z.push(*z);
                g_run.push(r as i32);
                g_q.push(v);
                g_nf.push(mat.len() as i32);
            }
        }
        write_table(
            &path,
            vec![
                Col::Str("group".into(), g_key),
                Col::I32("charge".into(), g_z),
                Col::I32("run".into(), g_run),
                Col::F64("quantity".into(), g_q),
                Col::I32("n_features".into(), g_nf),
            ],
        )?;
        Ok(())
    };
    write_level(
        format!("{out}.precursor.parquet"),
        prec.into_iter().map(|((p, z), m)| (p, z, m)).collect(),
    )?;
    write_level(
        format!("{out}.peptide.parquet"),
        pep.into_iter().map(|(p, m)| (p, -1, m)).collect(),
    )?;

    info!(
        proteins = data.len(),
        runs = n,
        method = if by_fragment { "directlfq" } else { "maxlfq" },
        normalize = ?normalize,
        size_factors = ?factors,
        "quant-lfq: done (+ .peptide/.precursor sibling matrices)"
    );
    Ok(rows)
}

/// Stripped base amino-acid sequence of a peptidoform: drop bracketed or
/// parenthesized modification blocks and any DECOY_ prefix, keep the residues.
/// Used to roll precursors up to peptide-level LFQ groups.
fn base_sequence(peptidoform: &str) -> String {
    let s = peptidoform.strip_prefix("DECOY_").unwrap_or(peptidoform);
    let mut out = String::new();
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = (depth - 1).max(0),
            c if depth == 0 && c.is_ascii_alphabetic() => out.push(c),
            _ => {}
        }
    }
    out
}

/// Per-run size factors for cross-run normalization of the feature-by-run matrix.
///
/// - `MedianRatio` (DESeq-style): for every complete-case feature (present and
///   positive in all runs) take its ratio to a geometric-mean pseudo-reference;
///   the run factor is the median of those ratios. Robust to a minority of
///   genuinely changing features, so a spike-in design's real fold changes are
///   preserved, not flattened.
/// - `Median`: align each run's median log2 intensity to the median of the
///   per-run medians.
/// - `None`: all factors 1.0.
///
/// Medians are taken over sorted values and the matrix is iterated in `BTreeMap`
/// key order, so the result is deterministic (docs/14_build_test_deploy_gotchas.md).
fn size_factors(
    data: &std::collections::BTreeMap<String, std::collections::BTreeMap<String, Vec<Option<f64>>>>,
    n: usize,
    method: NormalizeMethod,
) -> Vec<f64> {
    match method {
        NormalizeMethod::None => vec![1.0; n],
        NormalizeMethod::MedianRatio => {
            let mut lr: Vec<Vec<f64>> = vec![Vec::new(); n];
            for feats in data.values() {
                for vec in feats.values() {
                    if vec.iter().all(|x| x.is_some_and(|v| v > 0.0)) {
                        let logs: Vec<f64> = vec.iter().map(|x| x.unwrap().log2()).collect();
                        let refm = logs.iter().sum::<f64>() / n as f64;
                        for r in 0..n {
                            lr[r].push(logs[r] - refm);
                        }
                    }
                }
            }
            (0..n)
                .map(|r| {
                    if lr[r].is_empty() {
                        1.0
                    } else {
                        2f64.powf(median_sorted(&mut lr[r]))
                    }
                })
                .collect()
        }
        NormalizeMethod::Median => {
            let mut logs: Vec<Vec<f64>> = vec![Vec::new(); n];
            for feats in data.values() {
                for vec in feats.values() {
                    for r in 0..n {
                        if let Some(v) = vec[r] {
                            if v > 0.0 {
                                logs[r].push(v.log2());
                            }
                        }
                    }
                }
            }
            let med: Vec<f64> = (0..n)
                .map(|r| {
                    if logs[r].is_empty() {
                        0.0
                    } else {
                        median_sorted(&mut logs[r])
                    }
                })
                .collect();
            let mut m2 = med.clone();
            let target = if m2.is_empty() {
                0.0
            } else {
                median_sorted(&mut m2)
            };
            (0..n).map(|r| 2f64.powf(med[r] - target)).collect()
        }
    }
}

/// Median of a slice, sorting in place (ascending). Even lengths average the two
/// middle values. Empty slice returns 0.0. Values are assumed finite (no NaN).
fn median_sorted(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len();
    if m % 2 == 1 {
        v[m / 2]
    } else {
        (v[m / 2 - 1] + v[m / 2]) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::Table;

    /// A store built from plain per-row traces, all rows belonging to one candidate, so
    /// the axes deduplicate exactly as extract's window grid makes them.
    fn store_of(rt: &[Vec<f32>], inten: &[Vec<f32>]) -> ChromStore {
        let mut s = ChromStore::new();
        for (i, (r, it)) in rt.iter().zip(inten).enumerate() {
            s.push(0, &format!("y{i}"), 0.0, r, it).unwrap();
        }
        s
    }

    /// The same rows, forced through [`peak_window`]'s MERGED union construction.
    ///
    /// Distinct candidate ids alone do not force it. They stop the axis dedup, so each row
    /// gets its own axis id, but the uniformity scan then finds exactly ONE axis id
    /// whenever `rows.len() == 1` and takes the shared-axis fast path anyway -- which made
    /// every single-row `peak_window_both` case compare the shared path against itself.
    /// Clearing `axis_strict` withdraws the fast path's permission without touching a
    /// stored value, so the merge runs over exactly the same samples for any row count.
    fn store_of_unshared(rt: &[Vec<f32>], inten: &[Vec<f32>]) -> ChromStore {
        let mut s = ChromStore::new();
        for (i, (r, it)) in rt.iter().zip(inten).enumerate() {
            s.push(i as u32, &format!("y{i}"), 0.0, r, it).unwrap();
        }
        for strict in s.axis_strict.iter_mut() {
            *strict = false;
        }
        s
    }

    /// [`peak_window`] through BOTH union constructions, asserting they are bit-identical.
    /// The shared-axis path is an accumulation into one profile array; the merged path is
    /// the sort over every sample that replaced the two BTreeMaps. Every `peak_window`
    /// test below goes through this, so each is also an equality test between the paths.
    ///
    /// The routing is asserted rather than assumed: see [`store_of_unshared`] for why
    /// distinct candidate ids were not enough on their own.
    fn peak_window_both(
        rows: &[usize],
        rt: &[Vec<f32>],
        inten: &[Vec<f32>],
        frac: f64,
        grace: usize,
        hint: Option<f64>,
    ) -> (f64, f64, f64) {
        let merged_store = store_of_unshared(rt, inten);
        assert!(
            merged_store.axis_strict.iter().all(|&s| !s),
            "the merged fixture must refuse `peak_window`'s shared-axis fast path"
        );
        let shared = peak_window(rows, &store_of(rt, inten), frac, grace, hint);
        let merged = peak_window(rows, &merged_store, frac, grace, hint);
        let bits = |w: (f64, f64, f64)| (w.0.to_bits(), w.1.to_bits(), w.2.to_bits());
        assert_eq!(
            bits(shared),
            bits(merged),
            "shared-axis and merged-sample unions disagree: {shared:?} vs {merged:?}"
        );
        shared
    }

    /// [`fixed_window_indices`] through both nearest-sample searches, asserting they
    /// agree: `false` forces the linear first-minimum scan the binary search replaced.
    fn fixed_window_indices_both(
        rt: &[f32],
        apex: f64,
        half: usize,
        half_s: f64,
    ) -> Option<(usize, usize)> {
        let scanned = fixed_window_indices(rt, apex, half, half_s, false);
        let searched = fixed_window_indices(rt, apex, half, half_s, rt_is_sorted(rt));
        assert_eq!(
            scanned, searched,
            "scan and binary search disagree on {rt:?} at apex {apex}"
        );
        scanned
    }

    /// One candidate's fragment areas as the two parallel slices `select_fragment_areas`
    /// reads (it used to take a `Vec<(f64, f32)>`).
    fn split_areas(pairs: &[(f64, f32)]) -> (Vec<f64>, Vec<f32>) {
        (
            pairs.iter().map(|p| p.0).collect(),
            pairs.iter().map(|p| p.1).collect(),
        )
    }

    fn quant_test_path(name: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!("mumdia_quant_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        dir.join(format!("{n}_{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn trapezoid_area() {
        // triangle: rt 0,1,2 ; int 0,2,0 -> area = 2
        assert!((trapezoid(&[0.0, 1.0, 2.0], &[0.0, 2.0, 0.0]) - 2.0).abs() < 1e-9);
        // single point -> raw intensity
        assert_eq!(trapezoid(&[5.0], &[7.0]), 7.0);
    }

    #[test]
    fn trapezoid_window_clips_to_range() {
        let rt = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        let it = [0.0f32, 5.0, 10.0, 5.0, 0.0];
        // Whole trace: symmetric triangle, area = 20.
        assert!((trapezoid(&rt, &it) - 20.0).abs() < 1e-9);
        // Restricted to [1,3]: samples (1,5),(2,10),(3,5) -> 7.5 + 7.5 = 15.
        assert!((trapezoid_window(&rt, &it, 1.0, 3.0, false) - 15.0).abs() < 1e-9);
        // Window with a single in-range sample returns that raw intensity.
        assert_eq!(trapezoid_window(&rt, &it, 2.0, 2.0, false), 10.0);
        // Empty window integrates to 0.
        assert_eq!(trapezoid_window(&rt, &it, 10.0, 20.0, false), 0.0);
    }

    #[test]
    fn base_sequence_strips_mods_and_decoy() {
        assert_eq!(base_sequence("PEPTIDEK"), "PEPTIDEK");
        assert_eq!(base_sequence("M[Oxidation]PEC[Carbamidomethyl]K"), "MPECK");
        assert_eq!(base_sequence("DECOY_VAVGDGVAK"), "VAVGDGVAK");
    }

    #[test]
    fn center_envelope_clips_wing_interference() {
        // A clean rise-then-fall peak is left unchanged.
        let clean = [1.0f32, 3.0, 6.0, 3.0, 1.0];
        assert_eq!(center_envelope_1d(&clean), clean.to_vec());
        // Interference bump in the right wing (idx4 rises back to 5.0 after the
        // trough at idx3=2.0): apex idx2, the outward running-min caps it to 2.0.
        let interf = [1.0f32, 4.0, 10.0, 2.0, 5.0, 1.0];
        assert_eq!(
            center_envelope_1d(&interf),
            vec![1.0, 4.0, 10.0, 2.0, 2.0, 1.0]
        );
        // Enabling the envelope removes the bump, so the integrated area shrinks.
        let rt = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let off = trapezoid_window(&rt, &interf, 0.0, 5.0, false);
        let on = trapezoid_window(&rt, &interf, 0.0, 5.0, true);
        assert!(
            on < off,
            "envelope should not increase the area (on={on}, off={off})"
        );
    }

    #[test]
    fn peak_window_bounds_summed_xic_and_rejects_lone_interferent() {
        // Two fragments share an RT grid. Fragment 0 is the real co-eluting peptide
        // peaking at rt=6; fragment 1 is a lone interferent spiking at rt=1, well
        // separated (>= 2 zero scans) so the grace walk cannot bridge to it. The
        // SUMMED XIC apex lands on rt=6 and the window brackets the real peak only,
        // even though the interferent is tall.
        let grid: Vec<f32> = (0..12).map(|k| k as f32).collect();
        let real = vec![
            0.0f32, 0.0, 0.0, 0.0, 2.0, 6.0, 10.0, 6.0, 2.0, 0.0, 0.0, 0.0,
        ];
        let interf = vec![
            0.0f32, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let ch_rt = vec![grid.clone(), grid.clone()];
        let ch_int = vec![real, interf];
        let (lo, hi, _) = peak_window_both(&[0, 1], &ch_rt, &ch_int, 1.0 / 6.0, 1, None);
        // Apex rt=6 (sum=10); 1/6 threshold ~1.67. Left: idx4(2)>=thr, idx3/idx2=0
        // -> 2 consecutive misses stop at rt=4. Right: symmetric stop at rt=8.
        assert_eq!(lo, 4.0);
        assert_eq!(hi, 8.0);
        // The interferent spike at rt=1 is outside [lo,hi], so its windowed area is 0.
        assert_eq!(trapezoid_window(&ch_rt[1], &ch_int[1], lo, hi, false), 0.0);
    }

    #[test]
    fn peak_window_grace_bridges_single_dip() {
        // Summed profile with a single-scan dip below threshold on the right shoulder,
        // then recovery. grace=1 must bridge the dip; grace=0 must stop at it.
        // apex=10 at idx4; threshold at 1/3 -> 3.33. Right side: 5,1(dip),5,0.
        let grid: Vec<f32> = (0..9).map(|k| k as f32).collect();
        let prof = vec![0.0f32, 0.0, 1.0, 5.0, 10.0, 5.0, 1.0, 5.0, 0.0];
        let ch_rt = vec![grid.clone()];
        let ch_int = vec![prof];
        let (_, hi1, _) = peak_window_both(&[0], &ch_rt, &ch_int, 1.0 / 3.0, 1, None);
        let (_, hi0, _) = peak_window_both(&[0], &ch_rt, &ch_int, 1.0 / 3.0, 0, None);
        // grace=1 bridges the idx6 dip (1.0 < 3.33) and includes idx7 (5.0) -> rt 7.
        assert_eq!(hi1, 7.0);
        // grace=0 stops at the first sub-threshold scan -> last above-threshold rt 5.
        assert_eq!(hi0, 5.0);
    }

    #[test]
    fn median_sorted_odd_even_empty() {
        assert_eq!(median_sorted(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_sorted(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median_sorted(&mut []), 0.0);
    }

    #[test]
    fn median_ratio_recovers_global_scale_not_real_changes() {
        use std::collections::BTreeMap;
        // Two runs. Run 1 is a global 2x of run 0 for the bulk (unchanged) features,
        // plus one genuinely-up and one genuinely-down feature. Median-of-ratios must
        // recover the 2x global scale (f[1]/f[0] ~ 2) without being pulled by the two
        // real changes, and after dividing by the factors the bulk ratio -> 1 while
        // the real changes survive.
        let mut feats: BTreeMap<String, Vec<Option<f64>>> = BTreeMap::new();
        for i in 0..8 {
            let a = 100.0 * (i as f64 + 1.0);
            feats.insert(format!("bulk{i:02}"), vec![Some(a), Some(2.0 * a)]);
        }
        feats.insert("up".into(), vec![Some(100.0), Some(800.0)]); // +2 log2 vs global
        feats.insert("down".into(), vec![Some(400.0), Some(200.0)]); // -2 log2 vs global
        let mut data: BTreeMap<String, BTreeMap<String, Vec<Option<f64>>>> = BTreeMap::new();
        data.insert("PG".into(), feats);

        let f = size_factors(&data, 2, NormalizeMethod::MedianRatio);
        assert!(
            (f[1] / f[0] - 2.0).abs() < 0.02,
            "expected ~2x scale, got {f:?}"
        );
        // Bulk normalizes to ratio 1; the up/down real changes are preserved.
        let bulk = (100.0 / f[0], 200.0 / f[1]);
        assert!(
            (bulk.0 / bulk.1 - 1.0).abs() < 1e-9,
            "bulk should flatten to 1"
        );
        let up = (100.0 / f[0]) / (800.0 / f[1]);
        assert!(
            (up - 0.25).abs() < 1e-9,
            "up feature run0/run1 should stay 1:4"
        );
    }

    #[test]
    fn none_leaves_matrix_unnormalized() {
        use std::collections::BTreeMap;
        let mut feats: BTreeMap<String, Vec<Option<f64>>> = BTreeMap::new();
        feats.insert("f".into(), vec![Some(10.0), Some(40.0)]);
        let mut data: BTreeMap<String, BTreeMap<String, Vec<Option<f64>>>> = BTreeMap::new();
        data.insert("PG".into(), feats);
        assert_eq!(
            size_factors(&data, 2, NormalizeMethod::None),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn peak_window_apex_prefers_coelution_over_lone_interferent() {
        // Four real fragments co-elute at idx5 (each modest); one interferent fragment
        // has a lone tall spike at idx1. A plain summed-intensity argmax picks the
        // interferent (20 > 12); the co-elution rule ("-1 for robustness" on fragment
        // count) must pick idx5 (rt=5) where 4 fragments co-elute. This is the B-apex-
        // on-noise failure (candidate 7064964).
        let grid: Vec<f32> = (0..9).map(|k| k as f32).collect();
        let real = vec![0.0f32, 0.0, 0.0, 0.0, 2.0, 3.0, 2.0, 0.0, 0.0];
        let interf = vec![0.0f32, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ch_rt = vec![
            grid.clone(),
            grid.clone(),
            grid.clone(),
            grid.clone(),
            grid.clone(),
        ];
        let ch_int = vec![
            real.clone(),
            real.clone(),
            real.clone(),
            real.clone(),
            interf,
        ];
        let (_, _, apex) = peak_window_both(&[0, 1, 2, 3, 4], &ch_rt, &ch_int, 1.0 / 6.0, 1, None);
        assert_eq!(
            apex, 5.0,
            "apex must be the 4-fragment co-elution scan, not the lone interferent spike"
        );
    }

    #[test]
    fn identified_apex_anchors_window_against_brighter_off_apex_peak() {
        // All fragments share a bright interference peak at rt=1, so even the
        // legacy co-elution detector selects it. The identification apex at rt=5
        // must instead anchor the descent walk around the identified peak.
        let grid: Vec<f32> = (0..10).map(|k| k as f32).collect();
        let trace = vec![0.0f32, 50.0, 0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 0.0, 0.0];
        let ch_rt = vec![grid.clone(), grid.clone(), grid.clone()];
        let ch_int = vec![trace.clone(), trace.clone(), trace];

        let (_, _, legacy_apex) = peak_window_both(&[0, 1, 2], &ch_rt, &ch_int, 1.0 / 6.0, 1, None);
        assert_eq!(legacy_apex, 1.0);

        let (lo, hi, anchored_apex) =
            peak_window_both(&[0, 1, 2], &ch_rt, &ch_int, 1.0 / 6.0, 1, Some(5.1));
        assert_eq!(anchored_apex, 5.0);
        assert!(lo > 1.0 && hi >= 5.0, "anchored window was [{lo}, {hi}]");

        let (_, _, nonfinite_fallback) =
            peak_window_both(&[0, 1, 2], &ch_rt, &ch_int, 1.0 / 6.0, 1, Some(f64::NAN));
        assert_eq!(nonfinite_fallback, legacy_apex);
    }

    #[test]
    fn fragment_summary_distinguishes_missing_zero_and_positive_traces() {
        assert_eq!(
            summarize_fragment_areas(None, 3),
            (None, 0, "no_fragment_traces")
        );
        assert_eq!(
            summarize_fragment_areas(Some(&[0.0, -1.0, f64::NAN]), 3),
            (None, 0, "no_positive_fragment_area")
        );
        assert_eq!(
            summarize_fragment_areas(Some(&[0.0, 2.0, 5.0, f64::INFINITY]), 3),
            (Some(7.0), 2, "quantified")
        );
    }

    #[test]
    fn protein_rollup_uses_one_maximum_per_base_peptide() {
        let mut groups = ProteinBaseQuant::new();
        add_protein_base_quantity(&mut groups, "PG", 10, Some(12.0));
        add_protein_base_quantity(&mut groups, "PG", 10, Some(20.0));
        add_protein_base_quantity(&mut groups, "PG", 11, Some(5.0));
        add_protein_base_quantity(&mut groups, "PG", 12, None);

        let bases = &groups["PG"];
        assert_eq!(bases.len(), 2, "only quantifiable unique bases count");
        assert_eq!(
            rollup_protein_bases(bases, RollupMethod::Sum, 3),
            (Some(25.0), 2, "quantified")
        );
        assert_eq!(
            rollup_protein_bases(bases, RollupMethod::TopNSum, 1),
            (Some(20.0), 2, "quantified")
        );

        add_protein_base_quantity(&mut groups, "NO_QUANT", 99, None);
        assert_eq!(
            rollup_protein_bases(&groups["NO_QUANT"], RollupMethod::Sum, 3),
            (None, 0, "no_quantifiable_peptide")
        );
    }

    // Also the legacy-artifact regression: this chromatogram table carries no
    // `predicted_intensity` column, as every artifact written before that column existed
    // does. Quant must still read it and quantify from it unchanged.
    #[test]
    fn quant_run_preserves_unquantifiable_ids_and_applied_window_contract() {
        let scored = quant_test_path("scored.parquet");
        let chrom = quant_test_path("chrom.parquet");
        let peptide = quant_test_path("peptide_quant.parquet");
        let protein = quant_test_path("protein_quant.parquet");
        let fragment = quant_test_path("fragment_quant.parquet");
        let bounds = quant_test_path("peak_bounds.parquet");

        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2, 3, 4]),
                Col::U32("base_peptide_id".into(), vec![10, 10, 11, 12]),
                Col::Str(
                    "peptidoform".into(),
                    vec!["PEP1".into(), "PEP2".into(), "PEP3".into(), "PEP4".into()],
                ),
                Col::I32("charge".into(), vec![2, 3, 2, 2]),
                Col::Str("label".into(), vec!["target".into(); 4]),
                Col::Str(
                    "protein_group".into(),
                    vec!["PG".into(), "PG".into(), "EMPTY".into(), "ZERO".into()],
                ),
                Col::F64("peptide_q_value".into(), vec![0.0; 4]),
                Col::F64("apex_rt".into(), vec![5.0; 4]),
                Col::F64("elution_lo".into(), vec![4.0; 4]),
                Col::F64("elution_hi".into(), vec![6.0; 4]),
            ],
        )
        .unwrap();

        let grid: Vec<f32> = (0..10).map(|rt| rt as f32).collect();
        let c1a = vec![0.0f32, 50.0, 0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 0.0, 0.0];
        let c1b = vec![0.0f32, 25.0, 0.0, 0.0, 2.5, 5.0, 2.5, 0.0, 0.0, 0.0];
        let c2 = vec![0.0f32, 0.0, 0.0, 0.0, 10.0, 20.0, 10.0, 0.0, 0.0, 0.0];
        let zero = vec![0.0f32; 10];
        write_table(
            &chrom,
            vec![
                Col::U32("candidate_id".into(), vec![1, 1, 2, 4]),
                Col::Str(
                    "frag_name".into(),
                    vec!["b2".into(), "y3".into(), "b4".into(), "y5".into()],
                ),
                Col::ListF32(
                    "rt".into(),
                    vec![grid.clone(), grid.clone(), grid.clone(), grid],
                ),
                Col::ListF32("intensity".into(), vec![c1a, c1b, c2, zero]),
            ],
        )
        .unwrap();

        let cfg = QuantConfig::default();
        let rows = run(QuantParams {
            psms_scored: &scored,
            chromatograms: &chrom,
            out_peptide: &peptide,
            out_protein: &protein,
            out_fragment: Some(&fragment),
            out_peak_bounds: Some(&bounds),
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap();
        assert_eq!(rows, (4, 3));

        let pq = Table::read(&peptide).unwrap();
        assert_eq!(pq.u32("base_peptide_id").unwrap(), vec![10, 10, 11, 12]);
        assert_eq!(
            pq.str("quant_status").unwrap(),
            vec![
                "quantified",
                "quantified",
                "no_fragment_traces",
                "no_positive_fragment_area"
            ]
        );
        assert_eq!(pq.i32("n_fragments_used").unwrap(), vec![2, 1, 0, 0]);
        let quantities = pq.opt_f64("quantity").unwrap();
        assert_eq!(quantities[0], Some(22.5));
        assert_eq!(quantities[1], Some(30.0));
        assert_eq!(quantities[2], None);
        assert_eq!(quantities[3], None);
        assert_eq!(
            pq.opt_f64("integration_apex_rt").unwrap(),
            vec![Some(5.0), Some(5.0), None, Some(5.0)]
        );
        assert_eq!(pq.opt_f64("integration_lo_rt").unwrap()[0], Some(4.0));
        assert_eq!(pq.opt_f64("integration_hi_rt").unwrap()[0], Some(6.0));

        let gq = Table::read(&protein).unwrap();
        let names = gq.str("protein_group").unwrap();
        let values = gq.opt_f64("quantity").unwrap();
        let statuses = gq.str("quant_status").unwrap();
        let counts = gq.i32("n_peptides").unwrap();
        let by_group: HashMap<_, _> = names
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, (values[i], statuses[i].clone(), counts[i])))
            .collect();
        assert_eq!(by_group["PG"], (Some(30.0), "quantified".to_string(), 1));
        assert_eq!(
            by_group["EMPTY"],
            (None, "no_quantifiable_peptide".to_string(), 0)
        );
        assert_eq!(
            by_group["ZERO"],
            (None, "no_quantifiable_peptide".to_string(), 0)
        );

        let fq = Table::read(&fragment).unwrap();
        assert_eq!(fq.nrows, 3, "the all-zero ion must not be exported");
        assert!(fq.f64("quantity").unwrap().iter().all(|area| *area > 0.0));
    }

    #[test]
    fn peak_window_never_collapses_to_single_sample() {
        // A sharp ~1-scan summed XIC (both apex shoulders below the 1/6 threshold)
        // would collapse the peak_bounds window to lo==hi; trapezoid_window would then
        // return the raw apex HEIGHT (10) rather than a time-integrated area. Grid step
        // is 2 s so the true triangle area (20) differs from the height (10), proving
        // the widening yields intensity*seconds units consistent with broad peaks.
        let grid = vec![0.0f32, 2.0, 4.0, 6.0, 8.0];
        let spike = vec![0.0f32, 0.0, 10.0, 0.0, 0.0];
        let ch_rt = vec![grid.clone()];
        let ch_int = vec![spike];
        let (lo, hi, apex) = peak_window_both(&[0], &ch_rt, &ch_int, 1.0 / 6.0, 1, None);
        assert_eq!(apex, 4.0, "apex should be the summed-XIC max rt");
        assert!(hi > lo, "window must have nonzero width: lo={lo} hi={hi}");
        let a = trapezoid_window(&ch_rt[0], &ch_int[0], lo, hi, false);
        assert!(
            (a - 20.0).abs() < 1e-9,
            "expected triangle area 20, not height 10, got {a}"
        );
    }

    #[test]
    fn fixed_window_scan_and_second_forms_select_the_apex_subwindow() {
        // 1 s grid, triangle apex at 5 s.
        let rt: Vec<f32> = (0..11).map(|i| i as f32).collect();
        let it = vec![0.0f32, 0.0, 0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        // +/-1 scan around the apex sample: (4,5),(5,10),(6,5) -> 7.5 + 7.5 = 15.
        assert!((trapezoid_fixed_opts(&rt, &it, 5.0, 1, 0.0, false, None) - 15.0).abs() < 1e-9);
        // +/-1 s is the same three samples on this grid.
        assert!((trapezoid_fixed_opts(&rt, &it, 5.0, 0, 1.0, false, None) - 15.0).abs() < 1e-9);
        // The seconds form overrides the scan count, as its doc comment claims.
        assert!((trapezoid_fixed_opts(&rt, &it, 5.0, 5, 1.0, false, None) - 15.0).abs() < 1e-9);
        // A degenerate window (both half-widths 0) is widened to two samples rather
        // than returning the apex HEIGHT as if it were an area: (5,10),(6,5) -> 7.5.
        // Reporting an intensity where every other candidate reports intensity x seconds
        // mixes units within one run and corrupts relative and LFQ quantities, which is
        // the same reason `peak_window` widens its own collapsed window. Unreachable from
        // a config -- both zero means no fixed window at all -- but the helper is called
        // directly here and the units must hold at the boundary.
        assert_eq!(
            trapezoid_fixed_opts(&rt, &it, 5.0, 0, 0.0, false, None),
            7.5
        );
        assert_eq!(fixed_window_indices_both(&rt, 5.0, 0, 0.0), Some((5, 7)));
        // A window wider than the trace integrates the whole trace (area 20).
        assert!((trapezoid_fixed_opts(&rt, &it, 5.0, 0, 100.0, false, None) - 20.0).abs() < 1e-9);
        // An apex off the sampled range still integrates around the nearest sample.
        assert_eq!(
            trapezoid_fixed_opts(&rt, &it, -50.0, 0, 0.0, false, None),
            0.0
        );
        // Empty trace integrates to 0 rather than panicking on the index math.
        assert_eq!(
            trapezoid_fixed_opts(&[], &[], 5.0, 3, 0.0, false, None),
            0.0
        );
        assert_eq!(fixed_window_indices_both(&[], 5.0, 3, 0.0), None);
        assert_eq!(fixed_window_indices_both(&rt, 5.0, 1, 0.0), Some((4, 7)));
        assert_eq!(fixed_window_indices_both(&rt, 5.0, 0, 1.0), Some((4, 7)));

        // Distance guard on the seconds form: an apex far outside the sampled range has
        // NO sample inside +/- half_s, so the window is empty and the area is 0. Before
        // the guard the nearest sample was included however far away it was, so a weak
        // fragment sampled 45 s from the apex contributed its off-peak intensity as this
        // candidate's area -- while `trapezoid_window`, given the same RT bounds,
        // correctly returns 0.
        assert_eq!(fixed_window_indices_both(&rt, -50.0, 0, 5.0), None);
        assert_eq!(
            trapezoid_fixed_opts(&rt, &it, -50.0, 0, 5.0, false, None),
            0.0
        );
        // Just inside the guard, the sample is kept (and widened to two).
        assert_eq!(fixed_window_indices_both(&rt, -4.0, 0, 5.0), Some((0, 2)));
        // A single-sample trace cannot be widened, and must not panic.
        assert_eq!(
            fixed_window_indices_both(&rt[0..1], 0.0, 0, 1.0),
            Some((0, 1))
        );
    }

    #[test]
    fn flank_baseline_uses_the_flank_quantile() {
        let it = vec![1.0f32, 3.0, 100.0, 100.0, 100.0, 5.0, 7.0];
        // Window [2,5) with 2-sample flanks -> flank pool [1,3,5,7].
        // Median: position round(3 * 0.5) = 2 -> 5.
        assert_eq!(flank_baseline(&it, 2, 5, 2, 0.5), 5.0);
        // Lower quartile: position round(3 * 0.25) = 1 -> 3.
        assert_eq!(flank_baseline(&it, 2, 5, 2, 0.25), 3.0);
        // No flank sample exists (the window covers the trace) -> no background.
        assert_eq!(flank_baseline(&it, 0, 7, 3, 0.5), 0.0);

        // End to end: subtracting the median flank level lowers the area by
        // baseline * width and clips at zero.
        let rt: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let plain = trapezoid_fixed_opts(&rt, &it, 3.0, 1, 0.0, false, None);
        let debased = trapezoid_fixed_opts(&rt, &it, 3.0, 1, 0.0, false, Some((2, 0.5)));
        assert!((plain - 200.0).abs() < 1e-9, "got {plain}");
        assert!((debased - 190.0).abs() < 1e-9, "got {debased}");
    }

    #[test]
    fn select_fragment_areas_ranks_by_predicted_intensity() {
        // Fragment 0 has the largest observed area but the smallest library intensity:
        // the interference case `fragment_selection = predicted` exists to avoid.
        let pairs = [(100.0f64, 0.1f32), (40.0, 1.0), (30.0, 0.8)];
        let (a, pr) = split_areas(&pairs);
        let areas = Some((a.as_slice(), pr.as_slice()));
        assert_eq!(
            select_fragment_areas(areas, 2, FragmentSelection::ObservedArea),
            (Some(140.0), 2, "quantified")
        );
        assert_eq!(
            select_fragment_areas(areas, 2, FragmentSelection::Predicted),
            (Some(70.0), 2, "quantified")
        );
        // `observed_area` must stay byte-identical to the legacy summariser.
        assert_eq!(
            select_fragment_areas(areas, 2, FragmentSelection::ObservedArea),
            summarize_fragment_areas(Some(&a), 2)
        );
        // Both rankings report the same statuses on the degenerate inputs.
        let (zero_a, zero_p) = split_areas(&[(0.0, 1.0)]);
        let (nan_a, nan_p) = split_areas(&[(f64::NAN, 1.0), (10.0, 0.5)]);
        for sel in [
            FragmentSelection::ObservedArea,
            FragmentSelection::Predicted,
        ] {
            assert_eq!(
                select_fragment_areas(None, 3, sel),
                (None, 0, "no_fragment_traces")
            );
            assert_eq!(
                select_fragment_areas(Some((&zero_a, &zero_p)), 3, sel),
                (None, 0, "no_positive_fragment_area")
            );
            assert_eq!(
                select_fragment_areas(areas, 0, sel),
                (None, 0, "no_fragments_selected")
            );
            // Non-finite areas are dropped, not summed into a NaN quantity.
            assert_eq!(
                select_fragment_areas(Some((&nan_a, &nan_p)), 3, sel),
                (Some(10.0), 1, "quantified")
            );
        }
    }

    /// Scored + chromatogram pair for the fragment-selection tests: one candidate, three
    /// fragments on a 1 s grid with a 5 s apex and a wide 0-10 s elution hint. The
    /// brightest fragment by observed area (`b2`, which also carries a late interferent)
    /// is the dimmest by library intensity.
    fn selection_fixture(with_predicted: bool) -> (String, String) {
        let scored = quant_test_path("sel_scored.parquet");
        let chrom = quant_test_path("sel_chrom.parquet");
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1]),
                Col::U32("base_peptide_id".into(), vec![10]),
                Col::Str("peptidoform".into(), vec!["PEP1".into()]),
                Col::I32("charge".into(), vec![2]),
                Col::Str("label".into(), vec!["target".into()]),
                Col::Str("protein_group".into(), vec!["PG".into()]),
                Col::F64("peptide_q_value".into(), vec![0.0]),
                Col::F64("apex_rt".into(), vec![5.0]),
                Col::F64("elution_lo".into(), vec![0.0]),
                Col::F64("elution_hi".into(), vec![10.0]),
            ],
        )
        .unwrap();
        let grid: Vec<f32> = (0..11).map(|rt| rt as f32).collect();
        let b2 = vec![
            0.0f32, 0.0, 0.0, 0.0, 50.0, 100.0, 50.0, 0.0, 300.0, 600.0, 300.0,
        ];
        let y3 = vec![0.0f32, 0.0, 0.0, 0.0, 20.0, 40.0, 20.0, 0.0, 0.0, 0.0, 0.0];
        let y5 = vec![0.0f32, 0.0, 0.0, 0.0, 15.0, 30.0, 15.0, 0.0, 0.0, 0.0, 0.0];
        let mut cols = vec![
            Col::U32("candidate_id".into(), vec![1, 1, 1]),
            Col::Str(
                "frag_name".into(),
                vec!["b2".into(), "y3".into(), "y5".into()],
            ),
            Col::ListF32("rt".into(), vec![grid.clone(), grid.clone(), grid]),
            Col::ListF32("intensity".into(), vec![b2, y3, y5]),
        ];
        if with_predicted {
            cols.push(Col::F32("predicted_intensity".into(), vec![0.1, 1.0, 0.8]));
        }
        write_table(&chrom, cols).unwrap();
        (scored, chrom)
    }

    #[test]
    fn fixed_window_and_predicted_selection_use_the_library_fragments_at_the_apex() {
        let (scored, chrom) = selection_fixture(true);
        let peptide = quant_test_path("sel_peptide.parquet");
        let protein = quant_test_path("sel_protein.parquet");
        let cfg = QuantConfig {
            top_n_fragments: 2,
            fragment_selection: FragmentSelection::Predicted,
            fixed_window_s: 1.0,
            ..QuantConfig::default()
        };
        let rows = run(QuantParams {
            psms_scored: &scored,
            chromatograms: &chrom,
            out_peptide: &peptide,
            out_protein: &protein,
            out_fragment: None,
            out_peak_bounds: None,
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap();
        assert_eq!(rows, (1, 1));

        // +/-1 s of the 5 s apex integrates y3 to 60 and y5 to 45; b2 integrates to 150
        // there but ranks last by library intensity, so `predicted` must exclude it.
        let pq = Table::read(&peptide).unwrap();
        assert_eq!(pq.opt_f64("quantity").unwrap(), vec![Some(105.0)]);
        assert_eq!(pq.i32("n_fragments_used").unwrap(), vec![2]);
        // The reported window must be the one that was integrated, not the 0-10 s hint.
        assert_eq!(pq.opt_f64("integration_apex_rt").unwrap(), vec![Some(5.0)]);
        assert_eq!(pq.opt_f64("integration_lo_rt").unwrap(), vec![Some(4.0)]);
        assert_eq!(pq.opt_f64("integration_hi_rt").unwrap(), vec![Some(6.0)]);
    }

    #[test]
    fn adding_predicted_intensity_does_not_move_a_default_quantity() {
        // Compatibility contract: the column's presence must not change a legacy result,
        // and its absence must not stop one. Same traces, same default config, both ways.
        let cfg = QuantConfig::default();
        let mut out = Vec::new();
        for with_predicted in [false, true] {
            let (scored, chrom) = selection_fixture(with_predicted);
            let peptide = quant_test_path("cmp_peptide.parquet");
            let protein = quant_test_path("cmp_protein.parquet");
            let rows = run(QuantParams {
                psms_scored: &scored,
                chromatograms: &chrom,
                out_peptide: &peptide,
                out_protein: &protein,
                out_fragment: None,
                out_peak_bounds: None,
                cfg: &cfg,
                config_hash: "test",
            })
            .unwrap();
            let pq = Table::read(&peptide).unwrap();
            out.push((
                rows,
                pq.opt_f64("quantity").unwrap(),
                pq.str("quant_status").unwrap(),
                pq.i32("n_fragments_used").unwrap(),
                pq.opt_f64("integration_lo_rt").unwrap(),
                pq.opt_f64("integration_hi_rt").unwrap(),
            ));
        }
        assert_eq!(out[0], out[1]);
        assert!(out[0].1[0].is_some(), "the fixture must be quantifiable");
    }

    #[test]
    fn an_empty_chromatogram_table_preserves_every_identification() {
        // Extraction can accept a candidate and still write no chromatogram for
        // it, and a run can legitimately produce an empty chromatogram table. The
        // identification must survive with an explicit unquantifiable status: an
        // accepted PSM silently vanishing from the peptide table would be a
        // reported identification lost to a quantification detail, which is the
        // opposite of the contract that identification and quantifiability are
        // separate.
        let scored = quant_test_path("empty_scored.parquet");
        let chrom = quant_test_path("empty_chrom.parquet");
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2]),
                Col::U32("base_peptide_id".into(), vec![10, 11]),
                Col::Str("peptidoform".into(), vec!["PEP1".into(), "PEP2".into()]),
                Col::I32("charge".into(), vec![2, 2]),
                Col::Str("label".into(), vec!["target".into(); 2]),
                Col::Str("protein_group".into(), vec!["PG".into(), "PG".into()]),
                Col::F64("peptide_q_value".into(), vec![0.0, 0.0]),
                Col::F64("apex_rt".into(), vec![5.0, 5.0]),
                Col::F64("elution_lo".into(), vec![4.0, 4.0]),
                Col::F64("elution_hi".into(), vec![6.0, 6.0]),
            ],
        )
        .unwrap();
        write_table(
            &chrom,
            vec![
                Col::U32("candidate_id".into(), Vec::new()),
                Col::Str("frag_name".into(), Vec::new()),
                Col::ListF32("rt".into(), Vec::new()),
                Col::ListF32("intensity".into(), Vec::new()),
            ],
        )
        .unwrap();

        let peptide = quant_test_path("empty_peptide.parquet");
        let protein = quant_test_path("empty_protein.parquet");
        let cfg = QuantConfig::default();
        let rows = run(QuantParams {
            psms_scored: &scored,
            chromatograms: &chrom,
            out_peptide: &peptide,
            out_protein: &protein,
            out_fragment: None,
            out_peak_bounds: None,
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap();
        assert_eq!(rows.0, 2, "both identifications must be reported");

        let pq = Table::read(&peptide).unwrap();
        assert_eq!(
            pq.str("quant_status").unwrap(),
            vec!["no_fragment_traces"; 2]
        );
        assert_eq!(pq.opt_f64("quantity").unwrap(), vec![None, None]);
        assert_eq!(pq.i32("n_fragments_used").unwrap(), vec![0, 0]);
        // The protein group has no quantifiable peptide, and must say so rather
        // than reporting a quantity of zero.
        let gq = Table::read(&protein).unwrap();
        assert_eq!(
            gq.str("quant_status").unwrap(),
            vec!["no_quantifiable_peptide"]
        );
        assert_eq!(gq.opt_f64("quantity").unwrap(), vec![None]);
    }

    #[test]
    fn predicted_selection_without_the_column_is_a_clear_error() {
        let (scored, chrom) = selection_fixture(false);
        let cfg = QuantConfig {
            fragment_selection: FragmentSelection::Predicted,
            ..QuantConfig::default()
        };
        let err = run(QuantParams {
            psms_scored: &scored,
            chromatograms: &chrom,
            out_peptide: &quant_test_path("err_peptide.parquet"),
            out_protein: &quant_test_path("err_protein.parquet"),
            out_fragment: None,
            out_peak_bounds: None,
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap_err()
        .to_string();
        // The message must name the missing column and the way out, because the artifact
        // is silently older rather than malformed.
        assert!(err.contains("predicted_intensity"), "{err}");
        assert!(err.contains("observed_area"), "{err}");
    }

    #[test]
    fn quant_filter_admits_an_mbr_transfer_but_never_a_decoy() {
        // MBR lowers only the three PSM-level q columns, while `q_filter` defaults to
        // `peptide_q` and the report writers filter on the grouped columns, so a
        // transferred row failed every default filter and `mumdia mbr` followed by a
        // manual quant reported MBR as having done nothing.
        // `TARGET` / `DECOY` stand for the label column's two spellings; the filter now
        // takes the decoy bit `TableFile::str_eq` produces rather than the string.
        const TARGET: bool = false;
        const DECOY: bool = true;
        assert!(passes_quant_filter(TARGET, 0.5, 0.01, true));
        assert!(!passes_quant_filter(TARGET, 0.5, 0.01, false));
        // Acceptance is not a licence to quantify a decoy.
        assert!(!passes_quant_filter(DECOY, 0.001, 0.01, true));
        // A non-finite q is still not a pass on its own.
        assert!(!passes_quant_filter(TARGET, f64::NAN, 0.01, false));
        assert!(passes_quant_filter(TARGET, f64::NAN, 0.01, true));
        // Ordinary acceptance is unchanged.
        assert!(passes_quant_filter(TARGET, 0.005, 0.01, false));
    }

    #[test]
    fn predicted_selection_does_not_rank_a_nan_intensity_first() {
        // `total_cmp` orders NaN above every real value, so in the descending sort a NaN
        // predicted intensity reached the front of the ranking and was preferentially
        // summed into the top N.
        let pairs: [(f64, f32); 3] = [(10.0, f32::NAN), (20.0, 0.9), (30.0, 0.8)];
        let (a, pr) = split_areas(&pairs);
        let (q, used, status) =
            select_fragment_areas(Some((&a, &pr)), 1, FragmentSelection::Predicted);
        assert_eq!(status, "quantified");
        assert_eq!(used, 1);
        // The 0.9-intensity fragment wins, not the NaN one.
        assert_eq!(q, Some(20.0));
    }

    #[test]
    fn nearest_index_binary_search_matches_the_scan() {
        // The search replaces a forward linear scan whose strict `<` kept the FIRST
        // minimum, so ties, runs of identical RTs and targets off either end all have to
        // land on the same index.
        let traces: [&[f32]; 7] = [
            &[],
            &[5.0],
            &[1.0, 1.0, 5.0],
            &[1.0, 3.0, 3.0, 10.0],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            &[2.0, 2.0, 2.0],
            &[0.0, 0.0, 1.0, 1.0, 1.0, 9.0],
        ];
        for rt in traces {
            for &t in &[-10.0f64, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 9.0, 100.0] {
                assert_eq!(
                    nearest_index(rt, t, false),
                    nearest_index(rt, t, true),
                    "rt={rt:?} target={t}"
                );
            }
        }
    }

    #[test]
    fn nearest_index_returns_the_scans_answer_for_a_non_finite_target() {
        // The comment that used to stand in for a guard claimed any non-finite target
        // "makes every predicate false and lands on 0". That holds for NaN and for -inf
        // and is FALSE for +inf: `r < inf` is true at every sample, so `partition_point`
        // returned `rt.len()` and the search handed back the LAST index where the scan
        // hands back 0 (its `d` is `inf` everywhere and `inf < inf` never fires, so `k` is
        // never assigned). Both call sites filter the apex hint on `is_finite` today, so
        // no artifact moved; the precondition is now enforced rather than asserted.
        let traces: [&[f32]; 5] = [
            &[],
            &[5.0],
            &[1.0, 2.0],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            &[2.0, 2.0, 2.0],
        ];
        for rt in traces {
            for t in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
                assert_eq!(
                    nearest_index(rt, t, true),
                    nearest_index(rt, t, false),
                    "rt={rt:?} target={t}"
                );
                assert_eq!(
                    nearest_index(rt, t, true),
                    0,
                    "the scan leaves `k` at 0 for every non-finite target: rt={rt:?} target={t}"
                );
            }
        }
    }

    #[test]
    fn an_axis_differing_only_in_the_sign_of_a_zero_is_a_distinct_axis() {
        // Interning made every consumer read `store.rt(row)` instead of the row's own
        // values, and `-0.0 == 0.0` under f32's `PartialEq`, so a slice comparison handed
        // the second row the FIRST row's axis. `peak_window`'s merge keys the union on
        // `to_bits`, where `-0.0` and `0.0` are two points, so the substitution folded the
        // union: profile [4.0, 6.0] over axis [0.0, 1.0] instead of [1.0, 6.0, 3.0] over
        // [0.0, 1.0, -0.0]. Extract writes positive mzML scan times, but
        // `mumdia quant --chromatograms` takes a table written by anything.
        let mut s = ChromStore::new();
        s.push(0, "y1", 0.0, &[0.0, 1.0], &[1.0, 2.0]).unwrap();
        s.push(0, "y2", 0.0, &[-0.0, 1.0], &[3.0, 4.0]).unwrap();
        assert_ne!(s.axis_id[0], s.axis_id[1]);
        assert_eq!(
            s.rt(1)[0].to_bits(),
            (-0.0f32).to_bits(),
            "the row must read back the bits the table carried"
        );
        // An axis that really is identical still interns to one copy, which is what the
        // store exists for.
        s.push(0, "y3", 0.0, &[0.0, 1.0], &[5.0, 6.0]).unwrap();
        assert_eq!(s.axis_id[0], s.axis_id[2]);

        // And the fold moved the apex, not just the profile: with the `-0.0` sample the
        // brightest, the substitution summed it onto the `+0.0` point and put the apex on
        // the wrong sample.
        let mut moved = ChromStore::new();
        moved.push(0, "y1", 0.0, &[0.0, 1.0], &[1.0, 2.0]).unwrap();
        moved.push(0, "y2", 0.0, &[-0.0, 1.0], &[9.0, 0.0]).unwrap();
        let (_, _, apex) = peak_window(&[0, 1], &moved, 0.5, 0, None);
        assert_eq!(
            apex.to_bits(),
            (-0.0f64).to_bits(),
            "apex must land on the -0.0 sample (interning gave +0.0), got {apex}"
        );
    }

    #[test]
    fn a_merged_axis_ending_in_negative_zero_is_not_binary_searched() {
        // The merged union is ordered by BIT PATTERN, and `-0.0`'s bits sort after every
        // positive f32, so an axis that ends in `-0.0` descends at its last step. The old
        // guard asked only whether the last sample was `>= 0.0`, which `-0.0` satisfies,
        // so the axis was declared sorted and `nearest_index` binary-searched a descending
        // trace. `ai` is the apex index that anchors `peak_bounds` and the returned
        // window, so the disagreement is a different quantity.
        let rt = [vec![1.0f32, 2.0], vec![-0.0f32, 3.0]];
        let inten = [vec![1.0f32, 1.0], vec![1.0f32, 1.0]];
        let store = store_of(&rt, &inten);
        // Union axis, in bit order: [1.0, 2.0, 3.0, -0.0].
        assert_ne!(store.axis_id[0], store.axis_id[1]);

        // Below every sample the scan picks `-0.0` at index 3; the binary search returned
        // index 0 because `partition_point` found nothing smaller than the target.
        let (_, _, apex_low) = peak_window(&[0, 1], &store, 0.5, 0, Some(-1.0));
        assert_eq!(
            apex_low.to_bits(),
            (-0.0f64).to_bits(),
            "expected the -0.0 sample, got {apex_low}"
        );
        // Above every sample the scan picks 3.0 at index 2; the binary search ran off the
        // end and returned index 3, which is the `-0.0`.
        let (_, _, apex_high) = peak_window(&[0, 1], &store, 0.5, 0, Some(100.0));
        assert_eq!(apex_high, 3.0, "expected the 3.0 sample, got {apex_high}");
    }

    #[test]
    fn the_merged_fixture_refuses_the_shared_axis_fast_path_even_for_one_row() {
        // `peak_window` takes the shared-axis fast path whenever the rows resolve to ONE
        // axis id that is marked strictly increasing, and a single-row call always does,
        // whatever candidate id the row was pushed under. Distinct candidate ids therefore
        // did NOT make the merged fixture merge, and the three single-row
        // `peak_window_both` cases were comparing the shared path against itself.
        let rt = [vec![0.0f32, 1.0, 2.0]];
        let inten = [vec![1.0f32, 5.0, 1.0]];
        let plain = store_of(&rt, &inten);
        assert!(
            plain.axis_strict[plain.axis_id[0] as usize],
            "the fixture's axis is strictly increasing, so the fast path is available"
        );
        let merged = store_of_unshared(&rt, &inten);
        assert!(
            merged.axis_strict.iter().all(|&s| !s),
            "clearing `axis_strict` is what actually withdraws the fast path"
        );
        // Which is the equality `peak_window_both` claims to be testing.
        let bits = |w: (f64, f64, f64)| (w.0.to_bits(), w.1.to_bits(), w.2.to_bits());
        assert_eq!(
            bits(peak_window(&[0], &plain, 0.5, 0, None)),
            bits(peak_window(&[0], &merged, 0.5, 0, None))
        );
    }

    #[test]
    fn the_chromatogram_store_returns_the_rows_it_was_given() {
        let grid: Vec<f32> = (0..5).map(|k| k as f32).collect();
        let offset: Vec<f32> = (0..5).map(|k| k as f32 + 0.5).collect();
        let mut s = ChromStore::new();
        s.push(7, "y1", 0.25, &grid, &[1.0, 2.0, 3.0, 2.0, 1.0])
            .unwrap();
        s.push(7, "y2", 0.75, &grid, &[0.0, 1.0, 2.0, 1.0, 0.0])
            .unwrap();
        s.push(7, "b3", 0.5, &offset, &[9.0; 5]).unwrap();
        s.push(7, "b4", 0.1, &[], &[]).unwrap();
        s.push(8, "y1", 0.6, &grid, &[4.0; 5]).unwrap();
        assert_eq!(s.nrows(), 5);
        assert_eq!(s.rt(0), grid.as_slice());
        assert_eq!(s.inten(1), &[0.0, 1.0, 2.0, 1.0, 0.0]);
        assert_eq!(s.name(2), "b3");
        assert_eq!(s.pred[4], 0.6);
        assert_eq!(s.rt(3), &[] as &[f32]);
        assert_eq!(s.inten(3), &[] as &[f32]);
        // One stored axis per distinct grid within a candidate, and the interned name
        // table holds one string per distinct name.
        assert_eq!(s.axis_id[0], s.axis_id[1]);
        assert_ne!(s.axis_id[0], s.axis_id[2]);
        assert_eq!(s.axis_id[3], NO_AXIS);
        assert_eq!(s.names.names.len(), 4);
        // A new candidate starts a new axis even for identical values: that costs a copy,
        // never a value.
        assert_ne!(s.axis_id[0], s.axis_id[4]);
        assert_eq!(s.rt(4), grid.as_slice());
        assert!(s.rt_sorted);
        assert!(s.axis_strict[s.axis_id[0] as usize]);
    }

    #[test]
    fn an_unsorted_or_duplicated_axis_refuses_the_fast_paths() {
        let mut descending = ChromStore::new();
        descending
            .push(1, "y1", 0.0, &[3.0, 1.0, 2.0], &[1.0, 2.0, 3.0])
            .unwrap();
        assert!(
            !descending.rt_sorted,
            "a descending step must send every nearest-sample search back to the scan"
        );
        assert!(!descending.axis_strict[0]);

        let mut duplicated = ChromStore::new();
        duplicated
            .push(1, "y1", 0.0, &[1.0, 1.0, 2.0], &[1.0, 2.0, 3.0])
            .unwrap();
        assert!(
            duplicated.rt_sorted,
            "non-decreasing is all the nearest-sample search needs"
        );
        assert!(
            !duplicated.axis_strict[0],
            "a repeated RT must merge into one profile point, so the axis cannot be \
             used as the union axis"
        );

        let mut negative = ChromStore::new();
        negative
            .push(1, "y1", 0.0, &[-1.0, 2.0], &[1.0, 2.0])
            .unwrap();
        assert!(
            !negative.rt_sorted,
            "bit order is value order only above zero"
        );
    }

    #[test]
    fn a_repeated_rt_sample_merges_into_one_profile_point() {
        // The union used to be a BTreeMap keyed on the RT bit pattern, so two samples at
        // the same RT became ONE profile point carrying their sum. Expressed as two rows
        // the same two samples must give exactly the same window.
        let duplicated = peak_window(
            &[0],
            &store_of(
                &[vec![0.0f32, 1.0, 1.0, 2.0, 3.0]],
                &[vec![0.0f32, 4.0, 6.0, 0.0, 0.0]],
            ),
            1.0 / 6.0,
            1,
            None,
        );
        let split = peak_window(
            &[0, 1],
            &store_of(
                &[vec![0.0f32, 1.0, 2.0, 3.0], vec![1.0f32]],
                &[vec![0.0f32, 4.0, 0.0, 0.0], vec![6.0f32]],
            ),
            1.0 / 6.0,
            1,
            None,
        );
        assert_eq!(duplicated, split);
        assert_eq!(duplicated.2, 1.0, "the merged point is the apex");
    }

    #[test]
    fn the_candidate_index_groups_exactly_as_the_map_it_replaces() {
        for cids in [
            vec![3u32, 3, 3, 1, 1, 9], // grouped by candidate, as extract writes it
            vec![3u32, 1, 3, 9, 1, 3], // interleaved: the run cache cannot shortcut this
            vec![5u32],
        ] {
            let mut s = ChromStore::new();
            for (r, &c) in cids.iter().enumerate() {
                s.push(c, &format!("y{r}"), r as f32, &[r as f32], &[1.0])
                    .unwrap();
            }
            let index = CandIndex::build(&s);
            let mut want: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
            for (r, &c) in cids.iter().enumerate() {
                want.entry(c).or_default().push(r);
            }
            let got: BTreeMap<u32, Vec<usize>> = (0..index.len())
                .map(|ci| (index.cids[ci], index.rows_of(ci).to_vec()))
                .collect();
            assert_eq!(got, want, "grouping of {cids:?}");
            for ci in 0..index.len() {
                assert_eq!(index.find(index.cids[ci]), Some(ci));
                for slot in index.slots(ci) {
                    assert_eq!(index.slot_pred[slot], s.pred[index.rows[slot]]);
                }
            }
            assert_eq!(index.find(u32::MAX), None);
        }
    }

    #[test]
    fn several_scored_rows_of_one_candidate_agree_and_ms1_traces_stay_out() {
        // The top-N selection is memoised per candidate, so two scored rows on candidate 1
        // must carry exactly what the per-row computation produced. The `ms1_*` traces are
        // precursor channels, not fragment ions: here they are ten times brighter than any
        // fragment, so including one would move both the window and the sum. Candidate 3
        // has no scored row at all and reaches only the peak-bounds diagnostic, which is
        // the export that keeps every candidate's chromatogram rows.
        let scored = quant_test_path("memo_scored.parquet");
        let chrom = quant_test_path("memo_chrom.parquet");
        let peptide = quant_test_path("memo_peptide.parquet");
        let protein = quant_test_path("memo_protein.parquet");
        let fragment = quant_test_path("memo_fragment.parquet");
        let bounds = quant_test_path("memo_bounds.parquet");

        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1, 1, 2]),
                Col::U32("base_peptide_id".into(), vec![10, 11, 12]),
                Col::Str(
                    "peptidoform".into(),
                    vec!["PEPA".into(), "PEPB".into(), "PEPC".into()],
                ),
                Col::I32("charge".into(), vec![2, 3, 2]),
                Col::Str("label".into(), vec!["target".into(); 3]),
                Col::Str(
                    "protein_group".into(),
                    vec!["PG".into(), "PG".into(), "PG2".into()],
                ),
                Col::F64("peptide_q_value".into(), vec![0.0; 3]),
                Col::F64("apex_rt".into(), vec![5.0; 3]),
            ],
        )
        .unwrap();

        let grid: Vec<f32> = (0..10).map(|rt| rt as f32).collect();
        let c1a = vec![0.0f32, 50.0, 0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 0.0, 0.0];
        let c1b = vec![0.0f32, 25.0, 0.0, 0.0, 2.5, 5.0, 2.5, 0.0, 0.0, 0.0];
        let c2 = vec![0.0f32, 0.0, 0.0, 0.0, 10.0, 20.0, 10.0, 0.0, 0.0, 0.0];
        let ms1 = vec![100.0f32; 10];
        write_table(
            &chrom,
            vec![
                Col::U32("candidate_id".into(), vec![1, 1, 1, 2, 3]),
                Col::Str(
                    "frag_name".into(),
                    vec![
                        "b2".into(),
                        "ms1_mono".into(),
                        "y3".into(),
                        "b4".into(),
                        "y5".into(),
                    ],
                ),
                Col::ListF32(
                    "rt".into(),
                    vec![
                        grid.clone(),
                        grid.clone(),
                        grid.clone(),
                        grid.clone(),
                        grid.clone(),
                    ],
                ),
                Col::ListF32("intensity".into(), vec![c1a, ms1, c1b, c2.clone(), c2]),
            ],
        )
        .unwrap();

        let cfg = QuantConfig::default();
        let rows = run(QuantParams {
            psms_scored: &scored,
            chromatograms: &chrom,
            out_peptide: &peptide,
            out_protein: &protein,
            out_fragment: Some(&fragment),
            out_peak_bounds: Some(&bounds),
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap();
        assert_eq!(rows, (3, 2));

        let pq = Table::read(&peptide).unwrap();
        let quantities = pq.opt_f64("quantity").unwrap();
        // b2 + y3 over the walked window [4, 6]: (5+10)/2 + (10+5)/2 = 15, and half that.
        assert_eq!(quantities[0], Some(22.5));
        assert_eq!(
            quantities[0], quantities[1],
            "the two rows of candidate 1 must quantify identically"
        );
        assert_eq!(quantities[2], Some(30.0));
        assert_eq!(pq.i32("n_fragments_used").unwrap(), vec![2, 2, 1]);
        assert_eq!(
            pq.opt_f64("integration_lo_rt").unwrap(),
            vec![Some(4.0); 3],
            "the ms1 channel must not widen the window"
        );

        let fq = Table::read(&fragment).unwrap();
        let names = fq.str("fragment_name").unwrap();
        assert_eq!(names.len(), 5, "two fragments x two rows, plus candidate 2");
        assert!(
            !names.iter().any(|n| n.starts_with("ms1_")),
            "ms1 pseudo-traces are not fragment quantities: {names:?}"
        );

        let bq = Table::read(&bounds).unwrap();
        assert_eq!(
            bq.u32("candidate_id").unwrap(),
            vec![1, 2, 3],
            "the diagnostic keeps candidates no scored row selected"
        );
    }

    #[test]
    fn the_apex_of_an_accepted_candidate_may_come_from_a_row_the_filter_drops() {
        // Quant reads the identity columns at the accepted rows only and keeps the apex of
        // the loaded candidates only. Neither may change what a row gets: the apex of a
        // candidate is its FIRST finite `apex_rt` in row order over the whole table, which
        // for candidate 3 below is row 1 (4.0, q 0.5, filtered out) and not its accepted row
        // 5 (9.0); candidate 5's first row carries NaN, so row 2's 6.0 is its apex. The
        // report still counts every candidate of the table with a finite apex (3, 5, 7, 9),
        // the decoy's and the filtered ones included.
        let scored = quant_test_path("narrow_scored.parquet");
        let chrom = quant_test_path("narrow_chrom.parquet");
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![5, 3, 5, 7, 9, 3, 11]),
                Col::U32("base_peptide_id".into(), vec![50, 30, 50, 70, 90, 30, 110]),
                Col::Str(
                    "peptidoform".into(),
                    ["P5", "P3", "P5b", "P7", "P9", "P3b", "P11"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                ),
                Col::I32("charge".into(), vec![2, 3, 3, 2, 2, 2, 2]),
                Col::Str(
                    "label".into(),
                    [
                        "target", "target", "target", "decoy", "target", "target", "target",
                    ]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                ),
                Col::Str(
                    "protein_group".into(),
                    ["A", "B", "A", "C", "D", "B", "E"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                ),
                Col::F64(
                    "peptide_q_value".into(),
                    vec![0.001, 0.5, 0.001, 0.0, 0.001, 0.001, 0.9],
                ),
                Col::F64(
                    "apex_rt".into(),
                    vec![f64::NAN, 4.0, 6.0, 4.0, 5.0, 9.0, f64::NAN],
                ),
            ],
        )
        .unwrap();
        let grid: Vec<f32> = (0..12).map(|rt| rt as f32).collect();
        let trace: Vec<f32> = (0..12).map(|k| 1.0 + (k as f32 - 5.0).abs()).collect();
        let cids = [3u32, 5, 7, 9];
        write_table(
            &chrom,
            vec![
                Col::U32("candidate_id".into(), cids.to_vec()),
                Col::Str("frag_name".into(), vec!["y2".into(); cids.len()]),
                Col::ListF32("rt".into(), vec![grid.clone(); cids.len()]),
                Col::ListF32("intensity".into(), vec![trace.clone(); cids.len()]),
            ],
        )
        .unwrap();
        let fixed = QuantConfig {
            bound_peak: false,
            fixed_window_s: 2.0,
            ..QuantConfig::default()
        };
        // `bound_peak` with the diagnostic export is the `keep_all` read, which keeps every
        // candidate's apex, so both paths through the apex map are held to the same rows.
        for (tag, cfg, bounds) in [
            ("fixed", fixed, None),
            (
                "keep_all",
                QuantConfig::default(),
                Some("narrow_bounds.parquet"),
            ),
        ] {
            let peptide = quant_test_path(&format!("narrow_{tag}_peptide.parquet"));
            let protein = quant_test_path(&format!("narrow_{tag}_protein.parquet"));
            let bounds = bounds.map(quant_test_path);
            run(QuantParams {
                psms_scored: &scored,
                chromatograms: &chrom,
                out_peptide: &peptide,
                out_protein: &protein,
                out_fragment: None,
                out_peak_bounds: bounds.as_deref(),
                cfg: &cfg,
                config_hash: "test",
            })
            .unwrap();
            let pq = Table::read(&peptide).unwrap();
            assert_eq!(pq.u32("candidate_id").unwrap(), vec![5, 5, 9, 3], "{tag}");
            assert_eq!(
                pq.str("peptidoform").unwrap(),
                vec!["P5", "P5b", "P9", "P3b"],
                "{tag}"
            );
            assert_eq!(pq.i32("charge").unwrap(), vec![2, 3, 2, 2], "{tag}");
            assert_eq!(pq.u32("base_peptide_id").unwrap(), vec![50, 50, 90, 30]);
            assert_eq!(
                pq.opt_f64("integration_apex_rt").unwrap(),
                vec![Some(6.0), Some(6.0), Some(5.0), Some(4.0)],
                "{tag}: candidate 3's apex is the one on its filtered row"
            );
            let report: serde_json::Value =
                mumdia_io::json::read_json(&format!("{peptide}.report.json")).unwrap();
            assert_eq!(report["params"]["candidates_with_scored_apex"], json!(4));
            assert_eq!(report["params"]["apex_rt_column_present"], json!(true));
            let pg = Table::read(&protein).unwrap();
            assert_eq!(pg.str("protein_group").unwrap(), vec!["A", "B", "D"]);
        }
    }
    // ---- chromatogram load: accepted-candidate bitset, row-group spans ----

    /// Every row of a store as the values its consumers read:
    /// `(candidate_id, fragment name, predicted intensity, rt trace, intensity trace)`,
    /// floats as bit patterns so the comparison is exact.
    type StoredRow = (u32, String, u32, Vec<u32>, Vec<u32>);

    fn store_rows(s: &ChromStore) -> Vec<StoredRow> {
        (0..s.nrows())
            .map(|r| {
                (
                    s.cid[r],
                    s.name(r).to_string(),
                    s.pred[r].to_bits(),
                    s.rt(r).iter().map(|v| v.to_bits()).collect(),
                    s.inten(r).iter().map(|v| v.to_bits()).collect(),
                )
            })
            .collect()
    }

    /// Every field of a store, the axis TABLE and the axis IDS included, so a store built
    /// by the row-group plan can be held to BEING the single pass's store instead of merely
    /// agreeing with it row by row. The ids are the part that matters: [`peak_window`]
    /// routes on how many distinct axis ids a candidate's rows carry, and the shared-axis
    /// and merged-sample unions do not agree for every axis the table may hold (an axis
    /// beginning with `-0.0` is the counterexample [`ChromStore::append`] documents).
    #[derive(Debug, PartialEq)]
    struct StoreSnapshot {
        rows: Vec<StoredRow>,
        axis_id: Vec<u32>,
        axis_off: Vec<usize>,
        axis_vals: Vec<u32>,
        axis_strict: Vec<bool>,
        int_off: Vec<usize>,
        names: Vec<String>,
        name_id: Vec<u32>,
        rt_sorted: bool,
        open_cid: Option<u32>,
        open_axis_lo: usize,
    }

    fn snapshot(s: &ChromStore) -> StoreSnapshot {
        StoreSnapshot {
            rows: store_rows(s),
            axis_id: s.axis_id.clone(),
            axis_off: s.axis_off.clone(),
            axis_vals: s.axis_vals.iter().map(|v| v.to_bits()).collect(),
            axis_strict: s.axis_strict.clone(),
            int_off: s.int_off.clone(),
            names: s.names.names.clone(),
            name_id: s.name_id.clone(),
            rt_sorted: s.rt_sorted,
            open_cid: s.open_cid,
            open_axis_lo: s.open_axis_lo,
        }
    }

    /// A chromatogram table of `n_cand` candidates x `per_cand` fragments on a shared
    /// 8-point grid, written with `row_group_rows` rows per parquet row group.
    fn chrom_fixture(name: &str, n_cand: u32, per_cand: usize, row_group_rows: usize) -> String {
        use mumdia_io::table::TableWriter;
        let path = quant_test_path(name);
        let grid: Vec<f32> = (0..8).map(|k| k as f32).collect();
        let (mut cid, mut nm, mut rt, mut it, mut pred) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for c in 0..n_cand {
            for f in 0..per_cand {
                cid.push(c);
                nm.push(format!("y{f}"));
                rt.push(grid.clone());
                it.push(
                    (0..8)
                        .map(|k| (c as f32 + 1.0) * (f as f32 + 1.0) * k as f32)
                        .collect::<Vec<f32>>(),
                );
                pred.push(f as f32);
            }
        }
        let mut w = TableWriter::new(&path).with_row_group_rows(row_group_rows);
        w.write_cols(vec![
            Col::U32("candidate_id".into(), cid),
            Col::Str("frag_name".into(), nm),
            Col::F32("predicted_intensity".into(), pred),
            Col::ListF32("rt".into(), rt),
            Col::ListF32("intensity".into(), it),
        ])
        .unwrap();
        w.close().unwrap();
        path
    }

    const CHROM_COLS: [&str; 5] = [
        "candidate_id",
        "frag_name",
        "predicted_intensity",
        "rt",
        "intensity",
    ];

    #[test]
    fn the_accepted_candidate_bitset_answers_exactly_what_the_hash_set_did() {
        // A banded search offsets candidate ids by the band's `lib.global_offset`, so the
        // base is not 0 and the probe must say `false` BELOW it as well as above.
        let ids: Vec<u32> = vec![1_000_000, 1_000_001, 1_000_063, 1_000_064, 1_002_500];
        let reference: std::collections::HashSet<u32> = ids.iter().copied().collect();
        let set = CidSet::from_ids(&ids);
        assert!(matches!(set, CidSet::Bits { .. }));
        assert_eq!(set.range(), Some((1_000_000, 1_002_500)));
        for c in [
            0u32,
            1,
            999_999,
            1_000_000,
            1_000_001,
            1_000_002,
            1_000_062,
            1_000_063,
            1_000_064,
            1_000_065,
            1_002_499,
            1_002_500,
            1_002_501,
            u32::MAX,
        ] {
            assert_eq!(
                set.contains(c),
                reference.contains(&c),
                "bitset and hash set disagree on {c}"
            );
        }
        // The empty set admits nothing, which is what `keep_all` relies on when the
        // caller short-circuits it away.
        let empty = CidSet::empty();
        assert!(!empty.contains(0));
        assert!(!empty.contains(u32::MAX));
        assert_eq!(empty.range(), None);
        // An id range too wide to be worth a bit each falls back to hashing rather than
        // sizing an allocation off the id (see `CID_BITSET_MAX_BYTES`).
        let wide = CidSet::from_ids(&[0, u32::MAX]);
        assert!(matches!(wide, CidSet::Hashed(_)));
        assert!(wide.contains(0) && wide.contains(u32::MAX) && !wide.contains(1));
        assert_eq!(wide.range(), Some((0, u32::MAX)));
    }

    #[test]
    fn the_flat_string_columns_read_what_the_string_per_row_read() {
        // The scored table's three string columns are no longer materialised as
        // `Vec<String>`. Pin the substitution against the read it replaces, including an
        // empty value, a repeated value and a label spelling that is neither "target" nor
        // "decoy" (which the filter has always treated as a target).
        let p = quant_test_path("flat.parquet");
        let pforms = vec![
            "PEPTIDEK/2".to_string(),
            String::new(),
            "PEPTIDEK/2".to_string(),
            "AC[Carbamidomethyl]DEK/3".to_string(),
        ];
        let labels = vec![
            "target".to_string(),
            "decoy".to_string(),
            "target".to_string(),
            "entrapment".to_string(),
        ];
        write_table(
            &p,
            vec![
                Col::Str("peptidoform".into(), pforms.clone()),
                Col::Str("label".into(), labels.clone()),
            ],
        )
        .unwrap();
        let t = TableFile::open(&p).unwrap();
        let flat = FlatStr::read_rows(&t, "peptidoform", &[0, 1, 2, 3]).unwrap();
        let by_row = t.str("peptidoform").unwrap();
        assert_eq!(by_row, pforms);
        for (i, want) in by_row.iter().enumerate() {
            assert_eq!(flat.get(i), want);
        }
        // At the accepted rows only, value `j` is row `rows[j]`, the empty value included.
        let some = FlatStr::read_rows(&t, "peptidoform", &[1, 3]).unwrap();
        assert_eq!((some.get(0), some.get(1)), ("", by_row[3].as_str()));
        let is_decoy = t.str_eq("label", "decoy").unwrap();
        let is_target = t.str_eq("label", "target").unwrap();
        for (i, l) in labels.iter().enumerate() {
            assert_eq!(is_decoy[i], l == "decoy");
            assert_eq!(is_target[i], l == "target");
            // An unrecognised label is a target to the filter, as it was to `label ==
            // "decoy"`, and is NOT a consensus anchor, as it was not to `label ==
            // "target"`.
            assert_eq!(
                passes_quant_filter(is_decoy[i], 0.0, 0.01, false),
                l != "decoy"
            );
        }
        assert_eq!(is_target, vec![true, false, true, false]);
    }

    #[test]
    fn reading_row_group_by_row_group_reproduces_the_single_pass() {
        // 6 candidates x 3 fragments = 18 rows in row groups of 4, so candidates straddle
        // seams. Only some candidates are accepted, so the filter runs inside each span.
        let path = chrom_fixture("spans.parquet", 6, 3, 4);
        let ch = TableFile::open(&path).unwrap();
        let wanted = CidSet::from_ids(&[1, 2, 4, 5]);

        let one_pass = load_chrom_span(&ch, &CHROM_COLS, true, false, &wanted, &path).unwrap();

        // Per-span stores concatenated in file order, built explicitly so the assertion
        // does not depend on how many threads this machine has.
        let stats = ch.row_group_stats("candidate_id").unwrap();
        assert!(stats.len() > 1, "the fixture must have several row groups");
        let mut by_span = ChromStore::new();
        let mut first = 0usize;
        for s in &stats {
            let span = ch.span(first, s.rows).unwrap();
            by_span
                .append(load_chrom_span(&span, &CHROM_COLS, true, false, &wanted, &path).unwrap())
                .unwrap();
            first += s.rows;
        }
        assert_eq!(snapshot(&by_span), snapshot(&one_pass));
        // And the planner's own result, whichever path it chose on this machine.
        let planned = load_chromatograms(&ch, true, false, &wanted, &path, 4).unwrap();
        assert_eq!(snapshot(&planned), snapshot(&one_pass));

        // The comparison only proves something if a candidate really does straddle a seam,
        // which is the case `append` has to reconstruct. Candidate 1 owns file rows 3..6
        // and the groups hold 4 rows, so row 3 is in the first group and rows 4 and 5 in
        // the second.
        assert_eq!(
            stats[0].rows, 4,
            "the fixture's row groups must hold 4 rows"
        );
        let rows_of_1: Vec<usize> = (0..one_pass.nrows())
            .filter(|&r| one_pass.cid[r] == 1)
            .collect();
        assert_eq!(rows_of_1.len(), 3);
        // One axis id for it in BOTH stores: `append` deduped its second part against the
        // window the first part left open, exactly as the single pass would have.
        assert_eq!(
            rows_of_1
                .iter()
                .map(|&r| by_span.axis_id[r])
                .collect::<std::collections::HashSet<u32>>()
                .len(),
            1,
            "the seam must not mint a second axis for a straddling candidate"
        );
    }

    #[test]
    fn a_seam_inside_an_axis_beginning_with_negative_zero_does_not_move_the_window() {
        // Why `append` has to dedup across the seam rather than leave two identical axes.
        // `rt_is_sorted` accepts a leading `-0.0` (it compares `>= 0.0`) and the axis is
        // strictly increasing after it, so ONE axis id takes `peak_window`'s shared path
        // and walks the profile in value order. TWO ids take the merged union, which keys
        // on `to_bits`, where `-0.0` sorts after every positive f32: the axis becomes
        // `[1.0, 2.0, 3.0, -0.0]`, `axis_sorted` flips to false and the profile is
        // permuted. Same rows, same file, different quantity -- decided by nothing but the
        // writer's row-group size. Extract writes positive scan times; `mumdia quant
        // --chromatograms` takes a table written by anything.
        let grid = [-0.0f32, 1.0, 2.0, 3.0];
        let traces = [
            vec![9.0f32, 1.0, 0.0, 0.0],
            vec![7.0f32, 2.0, 0.0, 0.0],
            vec![5.0f32, 0.0, 1.0, 0.0],
        ];
        let mut whole = ChromStore::new();
        for (f, t) in traces.iter().enumerate() {
            whole.push(7, &format!("y{f}"), 0.0, &grid, t).unwrap();
        }
        assert!(
            whole.axis_strict[0],
            "a leading -0.0 is accepted as strictly increasing, which is the trap"
        );
        // The same rows, cut at every position a row-group boundary could fall at.
        for cut in 1..traces.len() {
            let mut split = ChromStore::new();
            for (lo, hi) in [(0, cut), (cut, traces.len())] {
                let mut part = ChromStore::new();
                for (f, t) in traces.iter().enumerate().take(hi).skip(lo) {
                    part.push(7, &format!("y{f}"), 0.0, &grid, t).unwrap();
                }
                split.append(part).unwrap();
            }
            assert_eq!(snapshot(&split), snapshot(&whole), "cut at {cut}");
            let rows = [0usize, 1, 2];
            let bits = |w: (f64, f64, f64)| (w.0.to_bits(), w.1.to_bits(), w.2.to_bits());
            for hint in [None, Some(0.0), Some(-1.0), Some(2.5), Some(f64::NAN)] {
                assert_eq!(
                    bits(peak_window(&rows, &whole, 0.5, 0, hint)),
                    bits(peak_window(&rows, &split, 0.5, 0, hint)),
                    "cut at {cut}, hint {hint:?}"
                );
            }
        }
    }

    #[test]
    fn the_row_group_plan_builds_the_single_passes_store_exactly() {
        // Concatenation has to reproduce the single pass FIELD BY FIELD, not just row by
        // row, because `peak_window` routes on the axis ids. Cut the same rows at every
        // position, including cuts inside a candidate, cuts on a candidate boundary and
        // three-way cuts that leave a whole candidate alone in the middle part.
        let grid = [0.0f32, 1.0, 2.0];
        let offset = [0.5f32, 1.5, 2.5];
        // (candidate, name, rt) -- candidate 2 appears twice, once either side of 3, and
        // candidate 4's trace is empty (NO_AXIS), which is the other thing `append` remaps.
        let rows: Vec<(u32, &str, &[f32])> = vec![
            (1, "y1", &grid),
            (1, "y2", &grid),
            (1, "b3", &offset),
            (1, "b4", &grid),
            (2, "y1", &offset),
            (3, "y1", &grid),
            (2, "y1", &offset),
            (4, "y1", &[]),
            (4, "y2", &grid),
        ];
        let build = |cuts: &[usize]| -> ChromStore {
            let mut out = ChromStore::new();
            let mut lo = 0usize;
            for &hi in cuts.iter().chain(std::iter::once(&rows.len())) {
                let mut part = ChromStore::new();
                for &(c, n, rt) in &rows[lo..hi] {
                    let inten: Vec<f32> = rt.iter().map(|v| v + 1.0).collect();
                    part.push(c, n, c as f32, rt, &inten).unwrap();
                }
                out.append(part).unwrap();
                lo = hi;
            }
            out
        };
        let whole = build(&[]);
        for cut in 1..rows.len() {
            assert_eq!(snapshot(&build(&[cut])), snapshot(&whole), "cut at {cut}");
        }
        for a in 1..rows.len() {
            for b in a + 1..rows.len() {
                assert_eq!(
                    snapshot(&build(&[a, b])),
                    snapshot(&whole),
                    "cuts at {a} and {b}"
                );
            }
        }
        // A span that kept no row -- every row filtered out, or an empty row group -- must
        // not close the open candidate either.
        let mut with_empty = ChromStore::new();
        let mut head = ChromStore::new();
        for &(c, n, rt) in &rows[..2] {
            let inten: Vec<f32> = rt.iter().map(|v| v + 1.0).collect();
            head.push(c, n, c as f32, rt, &inten).unwrap();
        }
        with_empty.append(head).unwrap();
        with_empty.append(ChromStore::new()).unwrap();
        let mut tail = ChromStore::new();
        for &(c, n, rt) in &rows[2..] {
            let inten: Vec<f32> = rt.iter().map(|v| v + 1.0).collect();
            tail.push(c, n, c as f32, rt, &inten).unwrap();
        }
        with_empty.append(tail).unwrap();
        assert_eq!(snapshot(&with_empty), snapshot(&whole));
        // And a `push` after an append continues the open candidate's window.
        let mut pushed = build(&[4]);
        let mut direct = build(&[]);
        for s in [&mut pushed, &mut direct] {
            s.push(4, "y3", 4.0, &grid, &[1.0, 2.0, 3.0]).unwrap();
        }
        assert_eq!(snapshot(&pushed), snapshot(&direct));
    }

    #[test]
    fn a_row_group_outside_the_accepted_ids_is_never_opened() {
        // 8 candidates x 2 fragments in row groups of 4, so each group holds exactly two
        // candidates. Accepting only candidate 0 leaves one group to read.
        let path = chrom_fixture("prune.parquet", 8, 2, 4);
        let ch = TableFile::open(&path).unwrap();
        let stats = ch.row_group_stats("candidate_id").unwrap();
        assert_eq!(stats.len(), 4);

        let wanted = CidSet::from_ids(&[0]);
        let (spans, _) =
            chrom_spans(&ch, false, &wanted, 4, &path).expect("pruning must plan spans");
        assert_eq!(spans.len(), 1, "only the first group can hold candidate 0");

        // Pruning is a read plan, not a filter: the store is what the full read would
        // have produced.
        let pruned = load_chromatograms(&ch, true, false, &wanted, &path, 4).unwrap();
        let full = load_chrom_span(&ch, &CHROM_COLS, true, false, &wanted, &path).unwrap();
        assert_eq!(snapshot(&pruned), snapshot(&full));
        assert_eq!(pruned.nrows(), 2);

        // `keep_all` (`--out-peak-bounds`) wants every candidate, so nothing is skipped.
        // The thread count is passed in rather than read from the rayon pool, so this says
        // the same thing on a 64-core host and on a 1-vCPU runner.
        let (all_spans, _) = chrom_spans(&ch, true, &wanted, 4, &path)
            .expect("a multi-group file plans spans for 4 threads");
        assert_eq!(all_spans.len(), 4);
        // One thread and nothing pruned is the single pass plus a copy per span, so the
        // plan declines rather than paying for it.
        assert!(chrom_spans(&ch, true, &wanted, 1, &path).is_none());

        // An accepted set disjoint from the file prunes every group. That must not turn
        // into a full read of an 802 MB table to produce an empty store: one group is read,
        // which is what validates the projection and the column types, and the per-row
        // rt/intensity guard runs only on kept rows, so it was never going to fire here.
        let none = CidSet::from_ids(&[10_000]);
        let (probe, _) = chrom_spans(&ch, false, &none, 4, &path)
            .expect("an empty accepted set still validates the table");
        assert_eq!(
            probe.len(),
            1,
            "one group, not four, and not the whole file"
        );
        assert_eq!(probe[0].nrows, 4);
        let empty = load_chromatograms(&ch, true, false, &none, &path, 4).unwrap();
        assert_eq!(
            snapshot(&empty),
            snapshot(&load_chrom_span(&ch, &CHROM_COLS, true, false, &none, &path).unwrap())
        );
        assert_eq!(empty.nrows(), 0);
        // And the projection is still checked: a multi-group table missing `rt` fails even
        // though every one of its groups is pruned.
        let bad = quant_test_path("prune_noshape.parquet");
        let mut w = mumdia_io::table::TableWriter::new(&bad).with_row_group_rows(4);
        w.write_cols(vec![
            Col::U32("candidate_id".into(), (0..8u32).collect::<Vec<u32>>()),
            Col::Str("frag_name".into(), vec!["y1".to_string(); 8]),
        ])
        .unwrap();
        w.close().unwrap();
        let bt = TableFile::open(&bad).unwrap();
        assert_eq!(bt.row_group_stats("candidate_id").unwrap().len(), 2);
        assert!(load_chromatograms(&bt, false, false, &none, &bad, 4).is_err());
    }

    #[test]
    fn a_candidate_split_across_row_groups_quantifies_identically() {
        // The same scored table against two chromatogram artifacts that differ only in
        // parquet row-group size, one of which puts every fragment of a candidate in its
        // own row group. The artifacts carry 8 candidates and the scored table accepts 6,
        // so the split artifact's last groups are PRUNED: the row-group plan is then taken
        // whatever the rayon pool width is, and this says the same thing on a 1-vCPU runner
        // as on a 64-core host. The comparison is of the written FILES, byte for byte.
        let scored = quant_test_path("split_scored.parquet");
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1, 2, 3, 4, 5]),
                Col::U32("base_peptide_id".into(), (0..6).collect::<Vec<u32>>()),
                Col::Str(
                    "peptidoform".into(),
                    (0..6).map(|i| format!("PEP{i}")).collect::<Vec<String>>(),
                ),
                Col::I32("charge".into(), vec![2; 6]),
                Col::Str("label".into(), vec!["target".into(); 6]),
                Col::Str("protein_group".into(), vec!["PG".into(); 6]),
                Col::F64("peptide_q_value".into(), vec![0.0; 6]),
                Col::F64("apex_rt".into(), vec![4.0; 6]),
            ],
        )
        .unwrap();

        let quantify = |chrom: &str, tag: &str| -> Vec<Vec<u8>> {
            let peptide = quant_test_path(&format!("{tag}_peptide.parquet"));
            let protein = quant_test_path(&format!("{tag}_protein.parquet"));
            let fragment = quant_test_path(&format!("{tag}_fragment.parquet"));
            let cfg = QuantConfig::default();
            run(QuantParams {
                psms_scored: &scored,
                chromatograms: chrom,
                out_peptide: &peptide,
                out_protein: &protein,
                out_fragment: Some(&fragment),
                out_peak_bounds: None,
                cfg: &cfg,
                config_hash: "test",
            })
            .unwrap();
            // The quantities as well, so a failure says which column moved rather than
            // only that two files differ.
            let pq = Table::read(&peptide).unwrap();
            assert_eq!(pq.opt_f64("quantity").unwrap().len(), 6);
            [&peptide, &protein, &fragment]
                .iter()
                .map(|p| std::fs::read(p).unwrap())
                .collect()
        };

        let one_group = chrom_fixture("one_group.parquet", 8, 4, 1_000);
        let per_row = chrom_fixture("per_row.parquet", 8, 4, 1);
        assert_eq!(
            TableFile::open(&per_row)
                .unwrap()
                .row_group_stats("candidate_id")
                .unwrap()
                .len(),
            32,
            "every row its own group, so every candidate straddles three seams"
        );
        assert_eq!(
            TableFile::open(&one_group)
                .unwrap()
                .row_group_stats("candidate_id")
                .unwrap()
                .len(),
            1,
            "and the reference artifact is the single pass"
        );
        assert_eq!(quantify(&one_group, "whole"), quantify(&per_row, "split"));
    }

    /// Single pass against the row-group plan, IN THE SHIPPED REGIME.
    ///
    /// The row-group size is the benchmark's main parameter and the earlier fixture had it
    /// wrong: 8,192-row groups plan `524288/8192 = 64` readers, clamped to the pool, so
    /// every span landed in ONE `spans.chunks(readers)` chunk with no merge barrier at all.
    /// Extract writes 65,536-row groups (`CHROM_ROW_GROUP_ROWS`), which plan 8 readers and
    /// therefore one serial `append` barrier per 8 groups. This fixture uses the shipped
    /// size and enough groups for several chunks.
    ///
    /// Both arms read the same file and build their own store, and the arms run twice in
    /// each order, because the first read warms the page cache for whichever runs second.
    /// A second pair keeps only 5.6% of the rows, the fraction a real run keeps, since the
    /// accepted-candidate filter decides how much of each decoded group is copied into the
    /// span store. Run with
    /// `cargo test -p mumdia --release -- --ignored --nocapture chromatogram_read_arms`.
    ///
    /// Still 8-point traces on a page-cached file, against 55.5 values per row and 802 MB
    /// of snappy on the artifact: take the ratio, not the milliseconds.
    #[test]
    #[ignore = "microbenchmark"]
    fn chromatogram_read_arms() {
        // 16 groups of 65,536 rows: 2 chunks at the 8 readers this plans.
        let path = chrom_fixture("bench.parquet", 262_144, 4, 1 << 16);
        let ch = TableFile::open(&path).unwrap();
        let threads = rayon::current_num_threads();
        let groups = ch.row_group_stats("candidate_id").unwrap().len();
        let all = CidSet::from_ids(&(0..262_144u32).collect::<Vec<u32>>());
        // Every 18th candidate, which is the 5.6% a real run keeps.
        let some = CidSet::from_ids(&(0..262_144u32).step_by(18).collect::<Vec<u32>>());

        for (label, wanted) in [("all rows", &all), ("5.6% of rows", &some)] {
            let mut serial_ms = Vec::new();
            let mut planned_ms = Vec::new();
            let mut rows = 0usize;
            for round in 0..2 {
                // Swapped, so neither arm is always the one that finds the file cold.
                let mut run_serial = || {
                    let t = Instant::now();
                    let s = load_chrom_span(&ch, &CHROM_COLS, true, false, wanted, &path).unwrap();
                    serial_ms.push(t.elapsed().as_secs_f64() * 1e3);
                    s
                };
                let mut run_planned = || {
                    let t = Instant::now();
                    let s = load_chromatograms(&ch, true, false, wanted, &path, threads).unwrap();
                    planned_ms.push(t.elapsed().as_secs_f64() * 1e3);
                    s
                };
                let (a, b) = if round == 0 {
                    let a = run_serial();
                    let b = run_planned();
                    (a, b)
                } else {
                    let b = run_planned();
                    let a = run_serial();
                    (a, b)
                };
                assert_eq!(snapshot(&a), snapshot(&b));
                rows = a.nrows();
            }
            println!(
                "{label}: stored {rows} of {} rows, {groups} groups of 65536, {threads} threads \
                 | one pass {serial_ms:?} ms | row-group plan {planned_ms:?} ms",
                ch.nrows,
            );
        }
    }

    /// The accepted-candidate probe, at the shape it runs at on the six-run HYE artifact:
    /// 53,863 accepted ids out of a 10.9M-precursor library, probed once per chromatogram
    /// row, 14,306,517 times.
    ///
    /// Each arm builds its own set inside its own timer, because the build is part of what
    /// the stage pays: the bitset is 1.4 MB of zeroing that the hash set does not do, and
    /// the comparison is only honest if it carries that. Run with
    /// `cargo test -p mumdia --release -- --ignored --nocapture accepted_candidate_probe_arms`.
    ///
    /// Measured on an i9-13900KS: HashSet 160.0-160.6 ms, bitset 10.5-10.9 ms. The probe
    /// order below is scattered over the library, which is the bitset's WORST case (one
    /// cache line per probe over 1.4 MB); the real table is grouped by candidate and
    /// ascending, so the stage sees at least this much.
    #[test]
    #[ignore = "microbenchmark"]
    fn accepted_candidate_probe_arms() {
        let n_lib = 10_900_000u32;
        let ids: Vec<u32> = (0..53_863u32)
            .map(|i| i.wrapping_mul(202) % n_lib)
            .collect();
        // Scattered over the library rather than grouped by candidate, so neither arm gets
        // the run locality the real table has, and the misses dominate as they do in the
        // stage (94% of rows belong to a candidate quant does not keep).
        let probes: Vec<u32> = (0..14_306_517u64)
            .map(|k| ((k * 1_000_003) % u64::from(n_lib)) as u32)
            .collect();

        let t = Instant::now();
        let hashed: std::collections::HashSet<u32> = ids.iter().copied().collect();
        let mut hits_hashed = 0usize;
        for &c in &probes {
            hits_hashed += usize::from(hashed.contains(&c));
        }
        let hash_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let bits = CidSet::from_ids(&ids);
        let mut hits_bits = 0usize;
        for &c in &probes {
            hits_bits += usize::from(bits.contains(c));
        }
        let bits_ms = t.elapsed().as_secs_f64() * 1e3;

        assert_eq!(
            hits_hashed, hits_bits,
            "the two sets must agree on every probe"
        );
        println!(
            "{} probes, {} accepted of {n_lib} | HashSet {hash_ms:.1} ms | bitset {bits_ms:.1} ms \
             | {hits_bits} hits",
            probes.len(),
            ids.len(),
        );
    }
}

/// Distinct values of the scored table's `source` column, or `None` when the column is
/// absent (a single-run table from before pooled rescoring stamped it).
///
/// `rescore` writes `source` as u32. The previous guard read it as i32 and treated the
/// resulting type error like an absent column, so the pooled-table refusal above never
/// ran on the engine's own output: a pooled table quantified against one run's
/// chromatograms produced one identical row per run without a word (docs/29 #1). A
/// present column that is neither u32 nor i32 is an error now, not a shrug.
pub(crate) fn pooled_source_count(ps: &TableFile, path: &str) -> Result<Option<usize>> {
    if !ps.has_column("source") {
        return Ok(None);
    }
    let distinct = match ps.u32("source") {
        Ok(v) => v
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        Err(u32_err) => match ps.i32("source") {
            Ok(v) => v
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            Err(_) => anyhow::bail!(
                "column `source` in {path} is present but neither u32 (what rescore writes) \
                 nor i32: {u32_err:#}"
            ),
        },
    };
    Ok(Some(distinct))
}

#[cfg(test)]
mod source_guard_tests {
    use super::*;

    fn table(name: &str, cols: Vec<Col>) -> String {
        let dir = std::env::temp_dir().join(format!("mumdia_quant_source_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir
            .join(format!("{name}.parquet"))
            .to_str()
            .unwrap()
            .to_string();
        write_table(&p, cols).unwrap();
        p
    }

    #[test]
    fn the_pooled_guard_reads_the_u32_source_rescore_writes() {
        // The exact type rescore emits (`Col::U32("source", ...)`, rescore.rs): two runs.
        let p = table(
            "u32",
            vec![
                Col::U32("candidate_id".into(), vec![0, 0]),
                Col::U32("source".into(), vec![0, 1]),
            ],
        );
        let t = TableFile::open(&p).unwrap();
        assert_eq!(pooled_source_count(&t, &p).unwrap(), Some(2));

        // The signed spelling still counts, an absent column is None, and a present
        // column of the wrong type is an error rather than "absent".
        let p = table(
            "i32",
            vec![
                Col::U32("candidate_id".into(), vec![0, 0]),
                Col::I32("source".into(), vec![3, 3]),
            ],
        );
        let t = TableFile::open(&p).unwrap();
        assert_eq!(pooled_source_count(&t, &p).unwrap(), Some(1));

        let p = table("none", vec![Col::U32("candidate_id".into(), vec![0, 1])]);
        let t = TableFile::open(&p).unwrap();
        assert_eq!(pooled_source_count(&t, &p).unwrap(), None);

        let p = table(
            "f64",
            vec![
                Col::U32("candidate_id".into(), vec![0, 1]),
                Col::F64("source".into(), vec![0.0, 1.0]),
            ],
        );
        let t = TableFile::open(&p).unwrap();
        assert!(pooled_source_count(&t, &p).is_err());
    }
}
