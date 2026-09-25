//! The inverted fragment index (docs/06_predict_frag_index_matchers.md), shared
//! by search-seed and extract.
//!
//! Structure-of-Arrays, flat and contiguous: three parallel arrays
//! (`idx_mz`, `idx_cid`, `idx_int`) globally sorted by fragment m/z, then
//! chunked into fixed buckets; within a bucket the order is precursor m/z
//! (== `candidate_id`, since candidates are ordered by precursor m/z at library
//! build). One binary search over `bucket_min` selects buckets overlapping a
//! query, a second over `candidate_id` narrows to the isolation-window slice,
//! and a linear tail applies the exact ppm bound (mirrors Sage `page_search`).
//!
//! MVP stores the index m/z as f32 and does all ppm math in f64
//! (docs/06_predict_frag_index_matchers.md). RT is applied as a per-candidate
//! window post-filter at probe time rather than a pre-partition (the documented
//! fallback, docs/09_extract.md), which keeps the index run-independent.

use anyhow::Result;
use arrow::array::{Array, Float32Array, Float64Array, StringArray, UInt32Array};
use mumdia_core::constants::{ppm_bounds, PROTON};
use mumdia_io::table::{require_no_nulls, TableFile};
use rayon::prelude::*;

/// Fragment rows per decoded batch while streaming the fragment table (a few MB).
const FRAG_BATCH_ROWS: usize = 1 << 16;

/// Precursor rows per decoded batch for the two columns that are validated in a streaming
/// pass and never kept (`label`, `candidate_id`).
const PREC_BATCH_ROWS: usize = 1 << 16;

#[derive(Clone, Debug)]
pub struct Candidate {
    pub candidate_id: u32,
    pub peptidoform_id: u32,
    pub base_peptide_id: u32,
    pub peptidoform: String,
    pub charge: i32,
    pub precursor_mz: f64,
    pub predicted_irt: f32,
    pub is_decoy: bool,
    pub protein: String,
    pub frag_start: usize,
    pub n_frag: usize,
}

pub struct Library {
    pub cands: Vec<Candidate>,
    /// Per-candidate fragment arrays, contiguous, grouped by candidate.
    /// Fragment m/z, f32. Both matchers already quantise to f32 before use (the
    /// fragindex bins and stores `mz as f32`, the naive matcher compares
    /// `mz as f32 as f64`), so storing f32 is exact for matching and saves 4 B of the
    /// 26 B per fragment row. f32 resolves 0.06 ppm at m/z 1000, two orders below the
    /// 5-20 ppm matching tolerances.
    pub frag_mz: Vec<f32>,
    pub frag_int: Vec<f32>,
    /// Fragment names, INTERNED: one dictionary index per fragment rather than one
    /// `String` per fragment. Library fragment names are drawn from a tiny vocabulary
    /// (`b1`, `y7`, `y12^2`, ...) that repeats across every candidate, while a
    /// `Vec<String>` costs ~24 bytes of `String` struct per fragment before any text --
    /// about 16 GB per copy at 657M fragments. Resolve with [`Library::frag_name_str`].
    pub frag_name_id: Vec<u16>,
    /// Distinct fragment names, indexed by [`Library::frag_name_id`].
    pub frag_name_dict: Vec<String>,
    /// Flat inverted index sorted by fragment m/z, bucketed.
    pub idx_mz: Vec<f32>,
    pub idx_cid: Vec<u32>,
    pub idx_int: Vec<f32>,
    pub bucket_min: Vec<f32>,
    pub bucket_size: usize,
    /// precursor m/z indexed by candidate_id (ascending).
    pub prec_mz: Vec<f64>,
    /// Row of the precursor table that local `candidate_id` 0 corresponds to: 0 for a full
    /// load, the first selected row for [`Library::load_range_with`]. Every id the library
    /// hands out and every artifact a stage writes from it is local; a pooling step that
    /// combines groups adds this back to obtain the library-wide id.
    pub global_offset: u32,
}

/// Reject a non-finite value in a numeric library column, naming the row, the column
/// and the file so the user can find the cell.
///
/// The engine validates finiteness at every internal and sidecar boundary
/// (`predict_frag.rs` on sidecar output, `rescore.rs` on the feature matrix and on
/// returned scores) and, before this, at neither external-input boundary. A NULL in a
/// user-supplied Parquet becomes NaN, and NaN is *accepted* by the comparison-based
/// guards downstream rather than rejected by them, so the failure is silent and
/// directional. These two helpers are the missing half of that contract.
fn require_finite_f64(v: &[f64], column: &str, path: &str) -> Result<()> {
    if let Some(row) = v.iter().position(|x| !x.is_finite()) {
        anyhow::bail!(
            "library column '{column}' has a non-finite value ({}) at row {row} in {path}; \
             a Parquet NULL decodes to NaN, and a NaN here silently means \"unbounded\" or \
             \"matches everything\" downstream rather than an error. Fix or drop the row",
            v[row]
        );
    }
    Ok(())
}

fn require_finite_f32(v: &[f32], column: &str, path: &str) -> Result<()> {
    if let Some(row) = v.iter().position(|x| !x.is_finite()) {
        anyhow::bail!(
            "library column '{column}' has a non-finite value ({}) at row {row} in {path}; \
             a Parquet NULL decodes to NaN, and a NaN here silently means \"unbounded\" or \
             \"matches everything\" downstream rather than an error. Fix or drop the row",
            v[row]
        );
    }
    Ok(())
}

/// Read the `label` column as one boolean per row, streaming, without materialising a
/// `String` per row.
///
/// The column is two-valued and its only consumer is [`Candidate::is_decoy`], so reading it
/// as a `Vec<String>` cost one heap block per precursor -- 203M blocks and about 6.5 GB on
/// the full library -- to produce one bit each. The validity rule is unchanged and stays in
/// one place: an unexpected value is handed to [`crate::fdr::validate_labels`], which is
/// also the only value ever turned into a `String`.
fn read_is_decoy(pt: &TableFile, path: &str) -> Result<Vec<bool>> {
    let mut out: Vec<bool> = Vec::with_capacity(pt.nrows);
    for b in pt.batches(Some(&["label"]), PREC_BATCH_ROWS)? {
        let b = b?;
        let a = b
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("column 'label' is not utf8"))?;
        // `value()` ignores the validity bitmap, so a NULL would read as "" and become a
        // target. Same contract as the fragment columns below.
        require_no_nulls(a, "label", path, out.len())?;
        for k in 0..a.len() {
            match a.value(k) {
                "decoy" => out.push(true),
                "target" => out.push(false),
                other => {
                    // `validate_labels` refuses anything but target/decoy, so this arm
                    // returns. Written as an explicit error rather than a fall-through:
                    // if that rule ever widens, not pushing here would leave the column
                    // shorter than the table and every later row would read its
                    // neighbour's label.
                    crate::fdr::validate_labels(&[other.to_string()])?;
                    anyhow::bail!(
                        "library precursor row {} of {path} has label '{other}', which is                          neither 'target' nor 'decoy'",
                        out.len()
                    );
                }
            }
        }
    }
    Ok(out)
}

/// Check, streaming, that `candidate_id` is the contiguous row-aligned range
/// `offset..offset + ncand`.
///
/// The ids are a precondition, never data: `index.rs` uses the row position everywhere and
/// the column exists only to be verified. Holding it cost 4 bytes per precursor (812 MB on
/// the full library) for the whole of the fragment load and index build.
fn check_candidate_ids(pt: &TableFile, path: &str, offset: usize, ncand: usize) -> Result<()> {
    let mut row = 0usize;
    for b in pt.batches(Some(&["candidate_id"]), PREC_BATCH_ROWS)? {
        let b = b?;
        let a = b
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| anyhow::anyhow!("column 'candidate_id' is not u32"))?;
        require_no_nulls(a, "candidate_id", path, row)?;
        for &candidate_id in a.values().iter().take(ncand.saturating_sub(row)) {
            if candidate_id as usize != row + offset {
                anyhow::bail!(
                    "library precursor row {} has candidate_id {} but candidate_id must \
                     be the contiguous range 0..n in row order; reindex the library \
                     (e.g. via the decoy-builder scripts)",
                    row + offset,
                    candidate_id
                );
            }
            row += 1;
        }
    }
    Ok(())
}

impl Library {
    /// Load a library and build the bucketed inverted index.
    ///
    /// Prefer [`Library::load_with`] and pass `build_bucketed = false` when the caller
    /// uses the `fragindex` backend (the default), since that backend never reads the
    /// bucketed arrays and building them costs a full sort of every library fragment.
    pub fn load(precursors: &str, fragments: &str, bucket_size: usize) -> Result<Library> {
        Self::load_with(precursors, fragments, bucket_size, true)
    }

    /// As [`Library::load`], but skips the bucketed `page_search` index when
    /// `build_bucketed` is false. `page_search` already early-returns on an empty index,
    /// so skipping is safe for callers that only use the fragindex matcher.
    pub fn load_with(
        precursors: &str,
        fragments: &str,
        bucket_size: usize,
        build_bucketed: bool,
    ) -> Result<Library> {
        Self::load_impl(
            precursors,
            fragments,
            bucket_size,
            build_bucketed,
            None,
            None,
        )
    }

    /// Load only the precursors whose `precursor_mz` lies in `[mz_lo, mz_hi]`, with their
    /// fragments, as a library of its own: local `candidate_id`s `0..n`, and
    /// [`Library::global_offset`] recording where the slice starts in the file.
    ///
    /// The precursor table is m/z-sorted with row-aligned ids (both are load-time
    /// invariants), so the range is one row span, found from the row-group statistics and
    /// one decode of `precursor_mz` over the boundary groups; the rest of the table is never
    /// read. Fragments are read the same way when their table is sorted by `candidate_id`
    /// at row-group granularity (what the library writers produce); an older, unsorted table
    /// still loads correctly through a filtered full scan, with a warning naming the cost.
    ///
    /// This is the per-isolation-window-group search: a group of windows can only select
    /// precursors in its m/z band, so a run searched group by group holds one band's library
    /// at a time. An empty range is an error, since no stage can run on it.
    pub fn load_range_with(
        precursors: &str,
        fragments: &str,
        mz_lo: f64,
        mz_hi: f64,
        bucket_size: usize,
        build_bucketed: bool,
    ) -> Result<Library> {
        if !(mz_lo.is_finite() && mz_hi.is_finite()) || mz_lo > mz_hi {
            anyhow::bail!("precursor m/z range [{mz_lo}, {mz_hi}] is empty or not finite");
        }
        let (first_row, n_rows) = Self::precursor_row_span(precursors, mz_lo, mz_hi)?;
        if n_rows == 0 {
            anyhow::bail!(
                "no precursor of {precursors} has precursor_mz in [{mz_lo}, {mz_hi}]; the \
                 isolation-window group selects nothing from this library"
            );
        }
        Self::load_impl(
            precursors,
            fragments,
            bucket_size,
            build_bucketed,
            Some((first_row, n_rows)),
            Some(first_row),
        )
    }

    /// Load a band that was written out as a precursor table of its own (local ids `0..n`,
    /// see `groups::write_band_slice`) together with its fragments from the library-wide
    /// fragment table, whose rows for this band carry the ids from `fragment_offset` to
    /// `fragment_offset + n`. This is what every stage of an isolation-window group loads:
    /// the band file is what the DeepLC sidecars rewrite, and the fragment table is shared
    /// and read by id range (selectively when it is sorted by candidate, else through a
    /// filtered scan).
    pub fn load_with_fragment_offset(
        precursors: &str,
        fragments: &str,
        fragment_offset: u32,
        bucket_size: usize,
        build_bucketed: bool,
    ) -> Result<Library> {
        Self::load_impl(
            precursors,
            fragments,
            bucket_size,
            build_bucketed,
            None,
            Some(fragment_offset as usize),
        )
    }

    /// Load the precursor rows `[first_row, first_row + n_rows)` of the whole library, with
    /// their fragments, as a library of its own: local ids `0..n_rows` and
    /// [`Library::global_offset`] `first_row`. The same library, value for value, as
    /// [`Library::load_with_fragment_offset`] over a band file that
    /// `groups::write_band_slice` wrote from those rows, without the file: a grouped run
    /// loads a band this way wherever nothing rewrites the band's precursor table.
    pub fn load_row_span_with(
        precursors: &str,
        fragments: &str,
        first_row: usize,
        n_rows: usize,
        bucket_size: usize,
        build_bucketed: bool,
    ) -> Result<Library> {
        if n_rows == 0 {
            anyhow::bail!("an empty row span of {precursors} is not a library");
        }
        Self::load_impl(
            precursors,
            fragments,
            bucket_size,
            build_bucketed,
            Some((first_row, n_rows)),
            Some(first_row),
        )
    }

    /// The load a stage asks for: the whole table, a band file (`fragment_offset`), or a
    /// row span of the whole table (`precursor_span`, with `fragment_offset` either absent
    /// or equal to the span's first row).
    pub fn load_for_stage(
        precursors: &str,
        fragments: &str,
        fragment_offset: Option<u32>,
        precursor_span: Option<(usize, usize)>,
        bucket_size: usize,
        build_bucketed: bool,
    ) -> Result<Library> {
        match (precursor_span, fragment_offset) {
            (None, None) => Self::load_with(precursors, fragments, bucket_size, build_bucketed),
            (None, Some(offset)) => Self::load_with_fragment_offset(
                precursors,
                fragments,
                offset,
                bucket_size,
                build_bucketed,
            ),
            (Some((first, n)), off) => {
                if off.is_some_and(|o| o as usize != first) {
                    anyhow::bail!(
                        "a row span starting at {first} was asked for with fragment offset \
                         {off:?}; the span's first row is the offset"
                    );
                }
                Self::load_row_span_with(
                    precursors,
                    fragments,
                    first,
                    n,
                    bucket_size,
                    build_bucketed,
                )
            }
        }
    }

    /// The row span `[first_row, first_row + n)` of the m/z-sorted precursor table whose
    /// `precursor_mz` lies in `[mz_lo, mz_hi]`. Row groups that cannot contain the range are
    /// skipped from their statistics; the ones that can are decoded (one column) and the
    /// exact bounds found by partition point. Without statistics the whole column is decoded,
    /// which is 8 bytes per precursor and still no library.
    pub fn precursor_row_span(precursors: &str, mz_lo: f64, mz_hi: f64) -> Result<(usize, usize)> {
        let pt = TableFile::open(precursors)?;
        let stats = pt.row_group_stats("precursor_mz")?;
        // Candidate row groups: those whose [min, max] overlaps the range. With statistics
        // this is a contiguous run for a sorted table; without, every group.
        let mut start = 0usize;
        let mut span_start = None;
        let mut span_end = 0usize;
        for s in &stats {
            let overlaps = match (s.min, s.max) {
                (Some(min), Some(max)) => max >= mz_lo && min <= mz_hi,
                _ => true,
            };
            if overlaps {
                if span_start.is_none() {
                    span_start = Some(start);
                }
                span_end = start + s.rows;
            }
            start += s.rows;
        }
        let Some(span_start) = span_start else {
            return Ok((0, 0));
        };
        let scan = TableFile::open_rows(precursors, span_start, span_end - span_start)?;
        let mz = scan.f64("precursor_mz")?;
        if mz.windows(2).any(|w| w[1] < w[0]) {
            anyhow::bail!(
                "library precursors in {precursors} are not ascending by precursor_mz; a \
                 range load needs the sorted order the decoy builders produce"
            );
        }
        let lo = mz.partition_point(|&v| v < mz_lo);
        let hi = mz.partition_point(|&v| v <= mz_hi);
        Ok((span_start + lo, hi.saturating_sub(lo)))
    }

    /// Open the fragment table for the local candidates `[offset, offset + ncand)` of the
    /// precursor table. Sorted by `candidate_id` at row-group granularity, only the groups
    /// that can hold those ids are decoded; otherwise the whole table is, filtered by id.
    fn open_fragments_for(fragments: &str, offset: usize, ncand: usize) -> Result<TableFile> {
        let ft = TableFile::open(fragments)?;
        let stats = ft.row_group_stats("candidate_id")?;
        if !mumdia_io::table::RowGroupStats::sorted_by_column(&stats) {
            tracing::warn!(
                fragments,
                "library: the fragment table is not sorted by candidate_id at row-group \
                 granularity, so a range load scans it whole and keeps the rows in range; \
                 rewrite it with scripts/sort_fragments.py to make group loads selective"
            );
            return Ok(ft);
        }
        let (lo, hi) = (offset as f64, (offset + ncand) as f64 - 1.0);
        let mut start = 0usize;
        let mut span_start = None;
        let mut span_end = 0usize;
        for s in &stats {
            let (Some(min), Some(max)) = (s.min, s.max) else {
                unreachable!("sorted_by_column requires statistics")
            };
            if max >= lo && min <= hi {
                if span_start.is_none() {
                    span_start = Some(start);
                }
                span_end = start + s.rows;
            }
            start += s.rows;
        }
        match span_start {
            Some(s) => TableFile::open_rows(fragments, s, span_end - s),
            None => TableFile::open_rows(fragments, 0, 0),
        }
    }

    /// `span`: the precursor rows to read (`None` = the whole file) and, with it, the file
    /// row of local id 0 that the ids in the file are checked against. `frag_offset`: the
    /// library-wide id of local candidate 0 in the fragment table, `None` when the precursor
    /// table is the whole library. They coincide for a range load of one library file; a band
    /// written to its own file has ids `0..n` (span `None`) while its fragments still carry
    /// library-wide ids. `Some(0)` is a band that happens to start at library row 0 and is
    /// still a band: the fragment table then holds other candidates' rows to skip, which is
    /// exactly what a plain `0` could not express.
    fn load_impl(
        precursors: &str,
        fragments: &str,
        bucket_size: usize,
        build_bucketed: bool,
        span: Option<(usize, usize)>,
        frag_offset: Option<usize>,
    ) -> Result<Library> {
        let partial = span.is_some() || frag_offset.is_some();
        let frag_offset = frag_offset.unwrap_or(0);
        let offset = span.map(|(first, _)| first).unwrap_or(0);
        let pt = match span {
            None => TableFile::open(precursors)?,
            Some((first, n)) => TableFile::open_rows(precursors, first, n)?,
        };
        let pfid = pt.u32("peptidoform_id")?;
        let baseid = pt.u32("base_peptide_id")?;
        let mut pform = pt.str("peptidoform")?;
        let charge = pt.i32("charge")?;
        let pmz = pt.f64("precursor_mz")?;
        let irt = pt.f32("predicted_irt")?;
        let mut protein = pt.str("protein")?;
        // `label` is validated and reduced to a bit in one streaming pass rather than being
        // held as a `Vec<String>`; `candidate_id` is checked the same way further down.
        let is_decoy = read_is_decoy(&pt, precursors)?;
        // A Parquet NULL decodes to NaN (mumdia-io `Table::f64`/`f32`), and NaN is
        // accepted rather than rejected by every downstream guard that should catch it:
        // the ascending-m/z check below is `<`, the extract RT-window guards are
        // `rt < lo || rt > hi`, and `within_ppm` compares against `min`/`max`, all of
        // which are false for NaN. So a single empty cell in a converted third-party
        // library does not produce a NaN result, it produces a candidate that is absent
        // from its own isolation window or one that matches every peak. Reject at load,
        // where the offending row can still be named.
        require_finite_f64(&pmz, "precursor_mz", precursors)?;
        require_finite_f32(&irt, "predicted_irt", precursors)?;
        // Charge must be positive. `ISOTOPE_SPACING / z` divides by it in three places in
        // extract, so a 0 gives an infinite spacing and every MS1 isotope channel lands
        // at the same m/z; a negative charge mirrors the envelope. Since a NULL in an
        // integer column used to decode as the raw buffer value -- in practice 0 -- an
        // imported library with a missing charge produced exactly that, silently. The
        // NULL is rejected by the accessor now, but an explicit 0 or -1 in the file still
        // has to be caught here.
        if let Some(row) = charge.iter().position(|&z| z <= 0) {
            anyhow::bail!(
                "library column 'charge' is {} at row {row} in {precursors}, but charge \
                 must be >= 1: extract divides the isotope spacing by it, so a 0 collapses \
                 every isotope channel onto one m/z and a negative value mirrors the \
                 envelope",
                charge[row]
            );
        }
        // Required strings must carry something. The accessor rejects a NULL, but an
        // explicitly empty string is a different thing and still reaches here: an empty
        // `peptidoform` cannot be parsed into residues, and an empty `protein` silently
        // joins every such candidate into one protein group.
        if let Some(row) = pform.iter().position(|v| v.trim().is_empty()) {
            anyhow::bail!(
                "library column 'peptidoform' is empty at row {row} in {precursors}; it is \
                 required, and an empty value cannot be parsed into residues"
            );
        }
        // An empty protein is a fact about the library, not a reason to refuse it: DIA-NN
        // leaves the protein empty for peptides it did not map to the FASTA, the iRT-kit
        // standards above all, and every DIA-NN library with the standards in it carries a
        // few dozen. Left empty, those peptides would silently share one anonymous protein
        // group; refused, no such library loads. So they are named: the same UNASSIGNED
        // group scripts/import_diann_lib.py writes at import, said out loud with a count
        // and examples, so the group is visible in proteins.tsv and in this log.
        let unassigned: Vec<usize> = protein
            .iter()
            .enumerate()
            .filter(|(_, v)| v.trim().is_empty())
            .map(|(i, _)| i)
            .collect();
        if !unassigned.is_empty() {
            let examples: Vec<&str> = unassigned
                .iter()
                .take(3)
                .map(|&i| pform[i].as_str())
                .collect();
            tracing::warn!(
                rows = unassigned.len(),
                examples = ?examples,
                library = precursors,
                "library: rows with an empty protein are grouped as UNASSIGNED (typically \
                 the iRT-kit standards); re-import with scripts/import_diann_lib.py to \
                 make the group explicit in the file"
            );
            for i in unassigned {
                protein[i] = "UNASSIGNED".to_string();
            }
        }

        let ncand = pt.nrows;
        // Precondition: candidate_id is the contiguous, row-aligned range 0..ncand
        // (the library + decoy builders guarantee this). An external library that
        // violates it would misgroup fragments or panic on the index below, so
        // check explicitly and fail with a clear error instead.
        // For a range load the same invariant holds against the file row: local id c is
        // file row c + offset, so the slice's ids must be exactly offset..offset + ncand.
        check_candidate_ids(&pt, precursors, offset, ncand)?;
        drop(pt);

        // Fragment table: two streaming passes over the four columns the library needs (the
        // artifact also carries `ion_type`, `ordinal`, `frag_charge` and `cardinality`, which
        // are never fetched). Pass 1 decodes only `candidate_id` and counts fragments per
        // candidate; pass 2 decodes the four columns batch by batch and scatters each row
        // straight into its final grouped slot, interning the fragment name on the way.
        //
        // This is the same counting sort as before -- rows scattered in ascending file order
        // keep each candidate's fragments in stored order, so the resulting layout is
        // identical -- but with the file as the source instead of owned copies: no
        // whole-table Arrow batches (23 GB at 657M rows), no owned copy of the four columns,
        // no `frag_order` permutation and no `Vec<String>` with one heap allocation per
        // fragment. The resident peak is the final arrays plus one batch, which is what lets
        // a modification-expanded library load on a 32 GB machine.
        // A partial load (a range of one file, or a band file against the shared fragment
        // table) sees fragments of other candidates and skips them.
        let ft = if partial {
            Self::open_fragments_for(fragments, frag_offset, ncand)?
        } else {
            TableFile::open(fragments)?
        };
        if ft.nrows > u32::MAX as usize {
            anyhow::bail!(
                "fragment library has {} rows; per-candidate fragment offsets are u32",
                ft.nrows
            );
        }
        let mut frag_offsets: Vec<u32> = vec![0; ncand + 1];
        {
            let mut row = 0usize;
            for b in ft.batches(Some(&["candidate_id"]), FRAG_BATCH_ROWS)? {
                let b = b?;
                let a = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| anyhow::anyhow!("fragment column 'candidate_id' is not u32"))?;
                // `values()` is the physical buffer and ignores the validity bitmap: a NULL
                // candidate_id would read as 0 and attach the fragment to candidate 0
                // (docs/29 #2).
                require_no_nulls(a, "candidate_id", fragments, row)?;
                for &candidate_id in a.values().iter() {
                    let c = candidate_id as usize;
                    row += 1;
                    // A range load sees the fragments of neighbouring candidates in the
                    // boundary row groups (or the whole table when it is unsorted); they
                    // belong to precursors this library does not hold and are skipped. A
                    // full load has no such rows, so an id past the end is a broken file.
                    if c < frag_offset || c >= frag_offset + ncand {
                        if partial {
                            continue;
                        }
                        anyhow::bail!(
                            "fragment row {} references candidate_id {c} >= precursor count {ncand}",
                            row - 1
                        );
                    }
                    frag_offsets[c - frag_offset + 1] += 1;
                }
            }
        }
        for c in 0..ncand {
            frag_offsets[c + 1] += frag_offsets[c];
        }
        // Rows in range, which for a range load is fewer than the rows decoded.
        let n_frag_rows = frag_offsets[ncand] as usize;

        let mut frag_mz: Vec<f32> = vec![0.0; n_frag_rows];
        let mut frag_int: Vec<f32> = vec![0.0; n_frag_rows];
        // Fragment names are INTERNED (see the struct field docs): a u16 dictionary id per
        // fragment, assigned by first appearance in file order.
        let mut frag_name_id: Vec<u16> = vec![0; n_frag_rows];
        let mut frag_name_dict: Vec<String> = Vec::new();
        let mut name_lookup: std::collections::HashMap<String, u16> =
            std::collections::HashMap::new();
        {
            let mut cursor = frag_offsets.clone();
            let reader = ft.batches(
                Some(&["candidate_id", "mz", "predicted_intensity", "name"]),
                FRAG_BATCH_ROWS,
            )?;
            let schema = reader.schema();
            let ix = |n: &str| {
                schema
                    .index_of(n)
                    .map_err(|_| anyhow::anyhow!("fragment library has no column '{n}'"))
            };
            let (i_cid, i_mz, i_int, i_name) = (
                ix("candidate_id")?,
                ix("mz")?,
                ix("predicted_intensity")?,
                ix("name")?,
            );
            let mut row_base = 0usize;
            for b in reader {
                let b = b?;
                let a_cid = b
                    .column(i_cid)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| anyhow::anyhow!("fragment column 'candidate_id' is not u32"))?;
                let a_mz = b
                    .column(i_mz)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| anyhow::anyhow!("fragment column 'mz' is not f64"))?;
                let a_int = b
                    .column(i_int)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        anyhow::anyhow!("fragment column 'predicted_intensity' is not f32")
                    })?;
                let a_name = b
                    .column(i_name)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| anyhow::anyhow!("fragment column 'name' is not utf8"))?;
                // Every required column, before any value is read: the finiteness checks
                // above ran on physical buffers, where a NULL is a perfectly finite 0.0, and
                // the fill below used to turn NULLs into NaN and "" instead of refusing
                // them (docs/29 #2).
                require_no_nulls(a_cid, "candidate_id", fragments, row_base)?;
                require_no_nulls(a_mz, "mz", fragments, row_base)?;
                require_no_nulls(a_int, "predicted_intensity", fragments, row_base)?;
                require_no_nulls(a_name, "name", fragments, row_base)?;
                // Same contract as the precursor columns above, applied batch by batch. A
                // non-finite fragment m/z is worse than a wrong value: `FragIndex::build`
                // collapses its whole m/z range when the observed min or max is not finite,
                // which clamps every real fragment into one bin and turns the probe into a
                // linear scan of the entire posting list. A non-finite predicted_intensity
                // sorts ahead of every real value under `total_cmp`, so it is preferentially
                // selected for quantification.
                if let Some(k) = a_mz.values().iter().position(|x| !x.is_finite()) {
                    anyhow::bail!(
                        "library column 'mz' has a non-finite value ({}) for candidate_id {} in \
                         {fragments}; a Parquet NULL decodes to NaN, and a NaN here silently \
                         means \"matches everything\" downstream rather than an error. Fix or \
                         drop the row",
                        a_mz.value(k),
                        a_cid.value(k)
                    );
                }
                if let Some(k) = a_int.values().iter().position(|x| !x.is_finite()) {
                    anyhow::bail!(
                        "library column 'predicted_intensity' has a non-finite value ({}) for \
                         candidate_id {} in {fragments}; a Parquet NULL decodes to NaN, and a \
                         NaN here sorts ahead of every real intensity. Fix or drop the row",
                        a_int.value(k),
                        a_cid.value(k)
                    );
                }
                for k in 0..b.num_rows() {
                    let c = a_cid.value(k) as usize;
                    if c < frag_offset || c >= frag_offset + ncand {
                        if partial {
                            continue;
                        }
                        anyhow::bail!(
                            "fragment table changed between passes: candidate_id {c} >= {ncand}"
                        );
                    }
                    let c = c - frag_offset;
                    let pos = cursor[c] as usize;
                    cursor[c] += 1;
                    // NULLs were rejected above, so the physical values are the values.
                    frag_mz[pos] = a_mz.value(k) as f32;
                    frag_int[pos] = a_int.value(k);
                    let name = a_name.value(k);
                    let id = match name_lookup.get(name) {
                        Some(&id) => id,
                        None => {
                            let id = u16::try_from(frag_name_dict.len()).map_err(|_| {
                                anyhow::anyhow!(
                                    "library has more than {} distinct fragment names; the \
                                     interned name id is a u16",
                                    u16::MAX
                                )
                            })?;
                            name_lookup.insert(name.to_string(), id);
                            frag_name_dict.push(name.to_string());
                            id
                        }
                    };
                    frag_name_id[pos] = id;
                }
                row_base += b.num_rows();
            }
        }
        drop(name_lookup);

        // `prec_mz` IS the decoded `precursor_mz` column: row c is candidate c, in the same
        // order, so the second array was a copy of the first. Move it instead of pushing a
        // duplicate, which removes 8 bytes per candidate (1.6 GB on the full library) and
        // one large heap block from the load's peak.
        let prec_mz = pmz;
        let mut cands = Vec::with_capacity(ncand);
        for c in 0..ncand {
            let start = frag_offsets[c] as usize;
            let n = frag_offsets[c + 1] as usize - start;
            cands.push(Candidate {
                // Local id: file row minus the slice's offset (verified equal above).
                candidate_id: c as u32,
                peptidoform_id: pfid[c],
                base_peptide_id: baseid[c],
                // Move the strings out of the column Vecs instead of cloning them.
                peptidoform: std::mem::take(&mut pform[c]),
                charge: charge[c],
                precursor_mz: prec_mz[c],
                predicted_irt: irt[c],
                is_decoy: is_decoy[c],
                protein: std::mem::take(&mut protein[c]),
                frag_start: start,
                n_frag: n,
            });
        }
        drop(frag_offsets);

        // Precondition for `candidate_range`: precursors ascending by m/z. The
        // fragment-index `partition_point` search over `prec_mz` assumes this;
        // an unsorted import (e.g. `import_diann_lib.py` output fed directly,
        // skipping the sorting decoy builder) would silently return wrong
        // candidate windows. Check explicitly and fail loudly.
        // The `is_nan()` arm is not redundant with the finiteness pass at the top of this
        // function: a bare `<` is false for NaN, so without it a NaN would pass the very
        // check whose purpose is to stop `partition_point` running on a slice that is not
        // partitioned by its predicate. Keeping both means neither guard is load-bearing
        // alone.
        for c in 1..ncand {
            if prec_mz[c] < prec_mz[c - 1] || prec_mz[c].is_nan() {
                anyhow::bail!(
                    "library precursors must be ascending by precursor_mz (row {c} m/z \
                     {} < row {} m/z {}); sort/reindex the library (the decoy-builder \
                     scripts do this)",
                    prec_mz[c],
                    c - 1,
                    prec_mz[c - 1]
                );
            }
        }
        // A missing class makes downstream target-decoy q-values meaningless.
        // Fail at library load rather than completing a long search with a
        // plausible-looking but invalid FDR estimate.
        let n_target = cands.iter().filter(|c| !c.is_decoy).count();
        let n_decoy = cands.iter().filter(|c| c.is_decoy).count();
        if n_target == 0 || n_decoy == 0 {
            anyhow::bail!(
                "library must contain both target and decoy candidates for valid FDR \
                 (targets={n_target}, decoys={n_decoy}); add paired decoys before search \
                 (e.g. via make_reverse_decoys.py)"
            );
        }

        // Build the bucketed inverted index, unless the caller only uses the fragindex
        // backend. This is a full copy of every library fragment as a (f32, u32, f32)
        // triple plus a global sort plus three more full arrays -- at 657M fragments that
        // is tens of GB and a large fraction of library-load time, all of it dead when
        // `page_search` is never called. `page_search` early-returns on an empty index.
        let bs = bucket_size.max(1);
        let (mut idx_mz, mut idx_cid, mut idx_int, mut bucket_min) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        if build_bucketed {
            let mut entries: Vec<(f32, u32, f32)> = Vec::with_capacity(n_frag_rows);
            for cd in cands.iter().take(ncand) {
                for k in 0..cd.n_frag {
                    let gi = cd.frag_start + k;
                    entries.push((frag_mz[gi], cd.candidate_id, frag_int[gi]));
                }
            }
            // Global sort by fragment m/z. Parallel stable sort: identical result to
            // the serial stable `sort_by` (same comparator, ties keep input order).
            entries.par_sort_by(|a, b| a.0.total_cmp(&b.0));

            // Chunk into buckets; within a bucket sort by candidate_id.
            for chunk_start in (0..entries.len()).step_by(bs) {
                let end = (chunk_start + bs).min(entries.len());
                bucket_min.push(entries[chunk_start].0);
                entries[chunk_start..end].sort_by_key(|e| e.1);
            }

            idx_mz.reserve(entries.len());
            idx_cid.reserve(entries.len());
            idx_int.reserve(entries.len());
            for (m, c, i) in entries {
                idx_mz.push(m);
                idx_cid.push(c);
                idx_int.push(i);
            }
        }

        crate::memlog::report(
            "library steady state",
            &[
                ("frag_mz", crate::memlog::bytes_of(&frag_mz)),
                ("frag_int", crate::memlog::bytes_of(&frag_int)),
                ("frag_name_id", crate::memlog::bytes_of(&frag_name_id)),
                ("idx_mz", crate::memlog::bytes_of(&idx_mz)),
                ("idx_cid", crate::memlog::bytes_of(&idx_cid)),
                ("idx_int", crate::memlog::bytes_of(&idx_int)),
                ("cands", crate::memlog::bytes_of(&cands)),
                ("prec_mz", crate::memlog::bytes_of(&prec_mz)),
            ],
        );
        Ok(Library {
            cands,
            frag_mz,
            frag_int,
            frag_name_id,
            frag_name_dict,
            idx_mz,
            idx_cid,
            idx_int,
            bucket_min,
            bucket_size: bs,
            prec_mz,
            // Where local id 0 sits in the library: the precursor span's first row, or the
            // fragment offset of a band file (its precursor rows are already local).
            global_offset: u32::try_from(if span.is_some() { offset } else { frag_offset })
                .map_err(|_| anyhow::anyhow!("candidate offset does not fit u32"))?,
        })
    }

    pub fn n_candidates(&self) -> usize {
        self.cands.len()
    }

    /// Fragments of a candidate as (m/z, predicted intensity, name) slices.
    /// Resolve an interned fragment-name id from [`Library::cand_frags`].
    pub fn frag_name_str(&self, id: u16) -> &str {
        self.frag_name_dict
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Per-candidate fragment m/z, predicted intensity, and INTERNED name ids (resolve
    /// with [`Library::frag_name_str`]).
    pub fn cand_frags(&self, cid: u32) -> (&[f32], &[f32], &[u16]) {
        assert!(
            !self.fragment_payload_released(),
            "the library's fragment payload was released (Library::release_fragment_payload); \
             only cand_frag_mz is available after that"
        );
        let c = &self.cands[cid as usize];
        let s = c.frag_start;
        let e = s + c.n_frag;
        (
            &self.frag_mz[s..e],
            &self.frag_int[s..e],
            &self.frag_name_id[s..e],
        )
    }

    /// Per-candidate fragment m/z alone. The only fragment column that survives
    /// [`Library::release_fragment_payload`], so this is what a stage that has already built
    /// its index must use.
    pub fn cand_frag_mz(&self, cid: u32) -> &[f32] {
        let c = &self.cands[cid as usize];
        &self.frag_mz[c.frag_start..c.frag_start + c.n_frag]
    }

    /// Free the fragment columns that only an index build reads: `frag_int`,
    /// `frag_name_id` and the name dictionary. `frag_mz` is kept, because the seed's mass
    /// recalibration still matches every library fragment against the scan.
    ///
    /// Predicted intensity and fragment name are copied into the index (or, for the
    /// bucketed arrays, into `idx_int`) at build time and never read from the library
    /// afterwards by a stage that has an index, so holding them is 6 bytes per fragment of
    /// dead weight for the whole search -- about 7 GB on a 1.2G-fragment library, and two
    /// of the large mappings the process is rationed on. Call it only after the index the
    /// stage will actually probe exists; [`Library::cand_frags`] panics afterwards rather
    /// than returning a short or stale slice.
    pub fn release_fragment_payload(&mut self) {
        self.frag_int = Vec::new();
        self.frag_name_id = Vec::new();
        self.frag_name_dict = Vec::new();
    }

    /// Whether [`Library::release_fragment_payload`] has run. Derived rather than stored:
    /// the three fragment arrays are parallel, so a `frag_int` shorter than `frag_mz` can
    /// only mean the payload was dropped.
    pub fn fragment_payload_released(&self) -> bool {
        self.frag_int.len() != self.frag_mz.len()
    }

    /// Local fragment index (0..n_frag) of the candidate whose stored m/z is
    /// closest to `frag_mz_f32` (returned by `page_search`).
    pub fn local_frag_index(&self, cid: u32, frag_mz_f32: f32) -> usize {
        let (mzs, _, _) = self.cand_frags(cid);
        let mut best = 0usize;
        let mut bestd = f32::MAX;
        for (i, &m) in mzs.iter().enumerate() {
            let d = (m - frag_mz_f32).abs();
            if d < bestd {
                bestd = d;
                best = i;
            }
        }
        best
    }

    /// Candidate-id half-open range [lo, hi) whose precursor m/z falls in the
    /// isolation window [win_lo, win_hi].
    pub fn candidate_range(&self, win_lo: f64, win_hi: f64) -> (u32, u32) {
        let lo = self.prec_mz.partition_point(|&m| m < win_lo) as u32;
        let hi = self.prec_mz.partition_point(|&m| m <= win_hi) as u32;
        (lo, hi)
    }

    /// Probe the index for observed neutral m/z `q` within `tol_ppm`, restricted
    /// to candidate ids in [cand_lo, cand_hi). Calls `f(candidate_id, frag_mz,
    /// predicted_intensity)` for each match (docs/06_predict_frag_index_matchers.md).
    pub fn page_search<F: FnMut(u32, f32, f32)>(
        &self,
        q: f64,
        tol_ppm: f64,
        cand_lo: u32,
        cand_hi: u32,
        mut f: F,
    ) {
        if cand_hi <= cand_lo || self.idx_mz.is_empty() {
            return;
        }
        let (lo, hi) = ppm_bounds(q, tol_ppm);
        let (lo32, hi32) = (lo as f32, hi as f32);
        let nb = self.bucket_min.len();
        // First bucket whose min could hold an entry >= lo: the bucket before
        // the first min > lo.
        let first = self
            .bucket_min
            .partition_point(|&m| m <= lo32)
            .saturating_sub(1);
        // Last bucket whose min <= hi.
        let last = self.bucket_min.partition_point(|&m| m <= hi32);
        let bs = self.bucket_size;
        for b in first..last.min(nb) {
            let start = b * bs;
            let end = (start + bs).min(self.idx_cid.len());
            let cids = &self.idx_cid[start..end];
            // candidate_id sub-slice via binary search (sorted ascending).
            let s = cids.partition_point(|&c| c < cand_lo);
            let e = cids.partition_point(|&c| c < cand_hi);
            for k in s..e {
                let gi = start + k;
                let m = self.idx_mz[gi];
                if m >= lo32 && m <= hi32 {
                    f(self.idx_cid[gi], m, self.idx_int[gi]);
                }
            }
        }
    }
}

/// Deconvolve an observed z-charged peak m/z to neutral m/z
/// (docs/06_predict_frag_index_matchers.md). Done in f64.
#[inline]
pub fn deconvolve(peak_mz: f64, z: i32) -> f64 {
    peak_mz * z as f64 - (z as f64 - 1.0) * PROTON
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::{write_table, Col, TableWriter};

    /// The label and candidate_id passes read in 65,536-row batches, and their row
    /// accounting across a batch boundary -- the absolute row in an error, and the
    /// `ncand` cap on the last batch -- is the only genuinely new logic in them. Nothing
    /// covered it: every other fixture in this file is six rows.
    #[test]
    fn labels_and_ids_are_read_across_a_batch_boundary() {
        let dir = std::env::temp_dir().join(format!("mumdia_idx_batch_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = PREC_BATCH_ROWS + 5;
        let path = |t: &str| dir.join(t).to_str().unwrap().to_string();

        let good = path("good.parquet");
        write_table(
            &good,
            vec![
                Col::U32("candidate_id".into(), (0..n as u32).collect()),
                Col::Str(
                    "label".into(),
                    (0..n)
                        .map(|i| if i % 2 == 0 { "target" } else { "decoy" }.to_string())
                        .collect(),
                ),
            ],
        )
        .unwrap();
        let t = TableFile::open(&good).unwrap();
        let decoy = read_is_decoy(&t, &good).unwrap();
        assert_eq!(decoy.len(), n, "one value per row across both batches");
        assert!(!decoy[0] && decoy[1]);
        // The rows either side of the boundary, and the last row of the short batch.
        assert_eq!(decoy[PREC_BATCH_ROWS - 1], (PREC_BATCH_ROWS - 1) % 2 == 1);
        assert_eq!(decoy[PREC_BATCH_ROWS], PREC_BATCH_ROWS % 2 == 1);
        assert_eq!(decoy[n - 1], (n - 1) % 2 == 1);
        check_candidate_ids(&t, &good, 0, n).unwrap();
        // A band: the ids start at an offset and only `ncand` of them are checked.
        check_candidate_ids(&t, &good, 0, PREC_BATCH_ROWS).unwrap();

        // A wrong id in the SECOND batch must be named by its absolute row.
        let bad = path("bad_id.parquet");
        let mut ids: Vec<u32> = (0..n as u32).collect();
        ids[PREC_BATCH_ROWS + 2] = 7;
        write_table(
            &bad,
            vec![
                Col::U32("candidate_id".into(), ids),
                Col::Str("label".into(), vec!["target".to_string(); n]),
            ],
        )
        .unwrap();
        let t = TableFile::open(&bad).unwrap();
        let err = check_candidate_ids(&t, &bad, 0, n).unwrap_err().to_string();
        assert!(
            err.contains(&format!("row {}", PREC_BATCH_ROWS + 2)) && err.contains("candidate_id 7"),
            "{err}"
        );

        // A NULL label in the second batch, likewise.
        let nulls = path("null_label.parquet");
        let mut labels: Vec<Option<String>> = vec![Some("target".to_string()); n];
        labels[PREC_BATCH_ROWS + 1] = None;
        write_table(
            &nulls,
            vec![
                Col::U32("candidate_id".into(), (0..n as u32).collect()),
                Col::OptStr("label".into(), labels),
            ],
        )
        .unwrap();
        let t = TableFile::open(&nulls).unwrap();
        // The row is in the error's source and the path in its context, so the whole
        // chain has to be formatted (`{:#}`), as mumdia-io's own test of this does.
        let err = format!("{:#}", read_is_decoy(&t, &nulls).unwrap_err());
        assert!(
            err.contains(&(PREC_BATCH_ROWS + 1).to_string()),
            "the null must be named by its absolute row: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Six candidates at m/z 400, 450, 500, 520, 600, 650 with two fragments each, written
    /// in row groups of two rows so a range crosses row-group boundaries on both tables.
    /// `frag_order` is the candidate order of the fragment table: `[0, 1, 2, 3, 4, 5]` is
    /// sorted by candidate_id (the writers' contract); anything else exercises the scan
    /// fallback of a range load.
    fn build_six_lib(dir: &std::path::Path, tag: &str, frag_order: &[usize]) -> (String, String) {
        let p = dir
            .join(format!("prec_{tag}.parquet"))
            .to_str()
            .unwrap()
            .to_string();
        let f = dir
            .join(format!("frag_{tag}.parquet"))
            .to_str()
            .unwrap()
            .to_string();
        let mz = [400.0, 450.0, 500.0, 520.0, 600.0, 650.0];
        let mut w = TableWriter::new(&p).with_row_group_rows(2);
        w.write_cols(vec![
            Col::U32("candidate_id".into(), (0..6).collect()),
            Col::U32("peptidoform_id".into(), (10..16).collect()),
            Col::U32("base_peptide_id".into(), (20..26).collect()),
            Col::Str(
                "peptidoform".into(),
                (0..6).map(|i| format!("PEPTIDE{i}K")).collect(),
            ),
            Col::I32("charge".into(), vec![2; 6]),
            Col::F64("precursor_mz".into(), mz.to_vec()),
            Col::F32(
                "predicted_irt".into(),
                (0..6).map(|i| i as f32 * 10.0).collect(),
            ),
            Col::Str(
                "label".into(),
                (0..6)
                    .map(|i| if i % 2 == 0 { "target" } else { "decoy" }.to_string())
                    .collect(),
            ),
            Col::Str("protein".into(), (0..6).map(|i| format!("P{i}")).collect()),
        ])
        .unwrap();
        w.close().unwrap();
        let mut cid = Vec::new();
        let mut fmz = Vec::new();
        let mut fint = Vec::new();
        let mut name = Vec::new();
        for &c in frag_order {
            for (k, off) in [100.0, 200.0].iter().enumerate() {
                cid.push(c as u32);
                fmz.push(off + 10.0 * c as f64);
                fint.push(1.0 - 0.1 * k as f32);
                name.push(if k == 0 {
                    "b2".to_string()
                } else {
                    "y3".to_string()
                });
            }
        }
        let mut w = TableWriter::new(&f).with_row_group_rows(2);
        w.write_cols(vec![
            Col::U32("candidate_id".into(), cid),
            Col::F64("mz".into(), fmz),
            Col::F32("predicted_intensity".into(), fint),
            Col::Str("name".into(), name),
        ])
        .unwrap();
        w.close().unwrap();
        (p, f)
    }

    fn frag_slice(lib: &Library, c: usize) -> (Vec<f32>, Vec<f32>, Vec<String>) {
        let cand = &lib.cands[c];
        let r = cand.frag_start..cand.frag_start + cand.n_frag;
        (
            lib.frag_mz[r.clone()].to_vec(),
            lib.frag_int[r.clone()].to_vec(),
            lib.frag_name_id[r]
                .iter()
                .map(|&id| lib.frag_name_dict[id as usize].clone())
                .collect(),
        )
    }

    #[test]
    fn range_load_is_the_slice_of_the_full_load() {
        let dir = std::env::temp_dir().join(format!("mumdia_index_range_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, f) = build_six_lib(&dir, "sorted", &[0, 1, 2, 3, 4, 5]);
        let full = Library::load_with(&p, &f, 8, false).unwrap();
        assert_eq!(full.global_offset, 0);
        // [440, 530] covers 450, 500, 520: file rows 1..4, crossing two row groups.
        let part = Library::load_range_with(&p, &f, 440.0, 530.0, 8, false).unwrap();
        assert_eq!(part.n_candidates(), 3);
        assert_eq!(part.global_offset, 1);
        for c in 0..3 {
            let (a, b) = (&part.cands[c], &full.cands[c + 1]);
            assert_eq!(a.candidate_id, c as u32);
            assert_eq!(a.peptidoform, b.peptidoform);
            assert_eq!(a.peptidoform_id, b.peptidoform_id);
            assert_eq!(a.base_peptide_id, b.base_peptide_id);
            assert_eq!(a.precursor_mz, b.precursor_mz);
            assert_eq!(a.is_decoy, b.is_decoy);
            assert_eq!(frag_slice(&part, c), frag_slice(&full, c + 1));
        }
        assert_eq!(part.frag_mz.len(), 6);
        assert_eq!(part.candidate_range(499.0, 501.0), (1, 2));
        // The two halves of the library cover it exactly once.
        let lo = Library::load_range_with(&p, &f, 0.0, 470.0, 8, false).unwrap();
        let hi = Library::load_range_with(&p, &f, 470.0, 1000.0, 8, false).unwrap();
        assert_eq!((lo.n_candidates(), lo.global_offset), (2, 0));
        assert_eq!((hi.n_candidates(), hi.global_offset), (4, 2));
        assert!(Library::load_range_with(&p, &f, 700.0, 800.0, 8, false).is_err());
    }

    #[test]
    fn band_slice_file_loads_with_the_fragment_offset_and_matches_the_range_load() {
        let dir = std::env::temp_dir().join(format!("mumdia_index_band_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, f) = build_six_lib(&dir, "band", &[0, 1, 2, 3, 4, 5]);
        let full = Library::load_with(&p, &f, 8, false).unwrap();
        // The band of rows 1..4 written as its own table, ids 0..3.
        let band = dir.join("band_prec.parquet").to_str().unwrap().to_string();
        let n = crate::groups::write_band_slice(&p, 1, 3, &band).unwrap();
        assert_eq!(n, 3);
        let ids = mumdia_io::table::TableFile::open(&band)
            .unwrap()
            .u32("candidate_id")
            .unwrap();
        assert_eq!(ids, vec![0, 1, 2]);
        let lib = Library::load_with_fragment_offset(&band, &f, 1, 8, false).unwrap();
        assert_eq!(lib.n_candidates(), 3);
        assert_eq!(lib.global_offset, 1);
        for c in 0..3 {
            assert_eq!(lib.cands[c].candidate_id, c as u32);
            assert_eq!(lib.cands[c].peptidoform, full.cands[c + 1].peptidoform);
            assert_eq!(frag_slice(&lib, c), frag_slice(&full, c + 1));
        }
        // Same content as loading the band by m/z from the whole file.
        let by_mz = Library::load_range_with(&p, &f, 440.0, 530.0, 8, false).unwrap();
        for c in 0..3 {
            assert_eq!(frag_slice(&lib, c), frag_slice(&by_mz, c));
            assert_eq!(lib.cands[c].precursor_mz, by_mz.cands[c].precursor_mz);
        }
        // And as loading the same rows by span, with no band file at all: what a grouped
        // run does wherever nothing rewrites the band's precursor table.
        let by_span = Library::load_for_stage(&p, &f, None, Some((1, 3)), 8, false).unwrap();
        assert_eq!(by_span.global_offset, lib.global_offset);
        assert_eq!(by_span.n_candidates(), lib.n_candidates());
        assert_eq!(by_span.frag_mz, lib.frag_mz);
        assert_eq!(by_span.frag_int, lib.frag_int);
        assert_eq!(by_span.frag_name_id, lib.frag_name_id);
        assert_eq!(by_span.frag_name_dict, lib.frag_name_dict);
        assert_eq!(by_span.prec_mz, lib.prec_mz);
        for c in 0..3 {
            let (a, b) = (&by_span.cands[c], &lib.cands[c]);
            assert_eq!(
                (
                    a.candidate_id,
                    a.peptidoform_id,
                    a.base_peptide_id,
                    a.charge
                ),
                (
                    b.candidate_id,
                    b.peptidoform_id,
                    b.base_peptide_id,
                    b.charge
                )
            );
            assert_eq!((a.frag_start, a.n_frag), (b.frag_start, b.n_frag));
            assert_eq!((&a.peptidoform, &a.protein), (&b.peptidoform, &b.protein));
            assert_eq!(
                (
                    a.precursor_mz.to_bits(),
                    a.predicted_irt.to_bits(),
                    a.is_decoy
                ),
                (
                    b.precursor_mz.to_bits(),
                    b.predicted_irt.to_bits(),
                    b.is_decoy
                )
            );
        }
        // A span with its own first row as the fragment offset is the same request; any
        // other offset is refused.
        assert!(Library::load_for_stage(&p, &f, Some(1), Some((1, 3)), 8, false).is_ok());
        assert!(Library::load_for_stage(&p, &f, Some(2), Some((1, 3)), 8, false).is_err());
        assert!(Library::load_for_stage(&p, &f, None, Some((1, 0)), 8, false).is_err());
    }

    /// The FIRST band starts at library row 0, and it is still a band: the shared fragment
    /// table holds every other candidate's rows, which the load must skip. Inferring
    /// "this is a band" from a non-zero offset made this case read the whole table and fail
    /// on the first foreign id, which is what every group-0 of a real run hit.
    #[test]
    fn the_band_at_library_row_zero_is_still_a_band() {
        let dir = std::env::temp_dir().join(format!("mumdia_index_band0_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, f) = build_six_lib(&dir, "band0", &[0, 1, 2, 3, 4, 5]);
        let full = Library::load_with(&p, &f, 8, false).unwrap();
        let band = dir.join("band0_prec.parquet").to_str().unwrap().to_string();
        assert_eq!(crate::groups::write_band_slice(&p, 0, 2, &band).unwrap(), 2);
        let lib = Library::load_with_fragment_offset(&band, &f, 0, 8, false).unwrap();
        assert_eq!(lib.n_candidates(), 2);
        assert_eq!(lib.global_offset, 0);
        for c in 0..2 {
            assert_eq!(lib.cands[c].peptidoform, full.cands[c].peptidoform);
            assert_eq!(frag_slice(&lib, c), frag_slice(&full, c));
        }
    }

    #[test]
    fn range_load_of_an_unsorted_fragment_table_scans_and_still_matches() {
        let dir =
            std::env::temp_dir().join(format!("mumdia_index_unsorted_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, f) = build_six_lib(&dir, "shuffled", &[3, 0, 5, 1, 4, 2]);
        let full = Library::load_with(&p, &f, 8, false).unwrap();
        let part = Library::load_range_with(&p, &f, 440.0, 530.0, 8, false).unwrap();
        assert_eq!(part.n_candidates(), 3);
        for c in 0..3 {
            assert_eq!(frag_slice(&part, c), frag_slice(&full, c + 1));
        }
    }

    fn build_tiny_lib(dir: &std::path::Path) -> (String, String) {
        let p = dir.join("prec.parquet").to_str().unwrap().to_string();
        let f = dir.join("frag.parquet").to_str().unwrap().to_string();
        // two candidates, sorted by precursor m/z
        write_table(
            &p,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1]),
                Col::U32("peptidoform_id".into(), vec![0, 1]),
                Col::U32("base_peptide_id".into(), vec![0, 1]),
                Col::Str(
                    "peptidoform".into(),
                    vec!["PEPTIDEK".into(), "SAMPLER".into()],
                ),
                Col::I32("charge".into(), vec![2, 2]),
                Col::F64("precursor_mz".into(), vec![400.0, 500.0]),
                Col::F32("predicted_irt".into(), vec![10.0, 20.0]),
                Col::Str("label".into(), vec!["target".into(), "decoy".into()]),
                Col::Str("protein".into(), vec!["P1".into(), "P2".into()]),
                Col::I32("n_fragments".into(), vec![2, 2]),
            ],
        )
        .unwrap();
        write_table(
            &f,
            vec![
                Col::U32("candidate_id".into(), vec![0, 0, 1, 1]),
                Col::F64("mz".into(), vec![200.1, 300.2, 250.5, 350.6]),
                Col::F32("predicted_intensity".into(), vec![1.0, 0.8, 0.9, 0.7]),
                Col::Str(
                    "name".into(),
                    vec!["b2".into(), "y3".into(), "b2".into(), "y3".into()],
                ),
                Col::Str(
                    "ion_type".into(),
                    vec!["b".into(), "y".into(), "b".into(), "y".into()],
                ),
                Col::I32("ordinal".into(), vec![2, 3, 2, 3]),
                Col::I32("frag_charge".into(), vec![1, 1, 1, 1]),
            ],
        )
        .unwrap();
        (p, f)
    }

    #[test]
    fn page_search_finds_only_in_window_and_tol() {
        // Unique per process: a fixed name races when two `cargo test` runs share a
        // machine, which is the convention docs/14 states and four other test modules
        // already follow.
        let dir = std::env::temp_dir().join(format!("mumdia_index_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, f) = build_tiny_lib(&dir);
        let lib = Library::load(&p, &f, 8).unwrap();
        assert_eq!(lib.n_candidates(), 2);

        // window covering only candidate 0 (precursor 400)
        let (lo, hi) = lib.candidate_range(399.0, 401.0);
        assert_eq!((lo, hi), (0, 1));

        // probe fragment 200.1 within 20 ppm -> should hit candidate 0 only.
        let mut hits = Vec::new();
        lib.page_search(200.1, 20.0, lo, hi, |cid, mz, _| hits.push((cid, mz)));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0);

        // fragment of candidate 1 (250.5) must NOT be found in candidate 0's window
        let mut hits2 = Vec::new();
        lib.page_search(250.5, 20.0, lo, hi, |cid, _, _| hits2.push(cid));
        assert!(hits2.is_empty());

        // in candidate 1's window it is found
        let (lo1, hi1) = lib.candidate_range(499.0, 501.0);
        let mut hits3 = Vec::new();
        lib.page_search(250.5, 20.0, lo1, hi1, |cid, _, _| hits3.push(cid));
        assert_eq!(hits3, vec![1]);
    }

    /// Write a library whose numeric columns can be poisoned, into a caller-unique
    /// directory. Mirrors `build_tiny_lib` but parameterised on the two values whose
    /// non-finite forms the loader must reject.
    fn build_lib_values(
        dir: &std::path::Path,
        pmz: Vec<f64>,
        frag_mz: Vec<f64>,
    ) -> (String, String) {
        std::fs::create_dir_all(dir).unwrap();
        let p = dir.join("prec.parquet").to_str().unwrap().to_string();
        let f = dir.join("frag.parquet").to_str().unwrap().to_string();
        write_table(
            &p,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1]),
                Col::U32("peptidoform_id".into(), vec![0, 1]),
                Col::U32("base_peptide_id".into(), vec![0, 1]),
                Col::Str(
                    "peptidoform".into(),
                    vec!["PEPTIDEK".into(), "SAMPLER".into()],
                ),
                Col::I32("charge".into(), vec![2, 2]),
                Col::F64("precursor_mz".into(), pmz),
                Col::F32("predicted_irt".into(), vec![10.0, 20.0]),
                Col::Str("label".into(), vec!["target".into(), "decoy".into()]),
                Col::Str("protein".into(), vec!["P1".into(), "P2".into()]),
                Col::I32("n_fragments".into(), vec![1, 1]),
            ],
        )
        .unwrap();
        write_table(
            &f,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1]),
                Col::F64("mz".into(), frag_mz),
                Col::F32("predicted_intensity".into(), vec![1.0, 0.9]),
                Col::Str("name".into(), vec!["b2".into(), "y3".into()]),
                Col::Str("ion_type".into(), vec!["b".into(), "y".into()]),
                Col::I32("ordinal".into(), vec![2, 3]),
                Col::I32("frag_charge".into(), vec![1, 1]),
            ],
        )
        .unwrap();
        (p, f)
    }

    fn unique_dir(tag: &str) -> std::path::PathBuf {
        // Unique per process so two concurrent `cargo test` runs cannot race, per the
        // convention in docs/14.
        std::env::temp_dir().join(format!("mumdia_index_{tag}_{}", std::process::id()))
    }

    #[test]
    fn non_finite_precursor_mz_is_rejected_with_the_row() {
        // A Parquet NULL decodes to NaN. Before the finiteness pass, NaN passed the
        // ascending-m/z check (a bare `<` is false for NaN), and `candidate_range`'s
        // `partition_point` then ran on a slice that was not partitioned by its
        // predicate: with [400, NaN] a window around 400 could return an empty range,
        // so a precursor genuinely inside the isolation window was never extracted in
        // any scan of that window for the whole run.
        let dir = unique_dir("nan_pmz");
        let (p, f) = build_lib_values(&dir, vec![400.0, f64::NAN], vec![200.1, 250.5]);
        // `unwrap_err` would require Debug on Library, which is a large SoA type.
        let err = match Library::load(&p, &f, 8) {
            Ok(_) => panic!("a non-finite library value must be rejected at load"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("precursor_mz"), "{err}");
        assert!(err.contains("non-finite"), "{err}");
        assert!(
            err.contains("row 1"),
            "should name the offending row: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn non_finite_fragment_mz_is_rejected() {
        // An infinite fragment m/z used to collapse the fragment index's whole m/z range
        // to a two-Da placeholder, clamping every real fragment into one bin and turning
        // the +/-1 probe into a linear scan of the entire posting list per peak: no error,
        // no wrong answer, an unbounded hang.
        let dir = unique_dir("inf_frag");
        let (p, f) = build_lib_values(&dir, vec![400.0, 500.0], vec![200.1, f64::INFINITY]);
        // `unwrap_err` would require Debug on Library, which is a large SoA type.
        let err = match Library::load(&p, &f, 8) {
            Ok(_) => panic!("a non-finite library value must be rejected at load"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("'mz'"), "{err}");
        assert!(err.contains("non-finite"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn finite_library_still_loads() {
        // Guard against the finiteness pass rejecting a valid library.
        let dir = unique_dir("ok");
        let (p, f) = build_lib_values(&dir, vec![400.0, 500.0], vec![200.1, 250.5]);
        let lib = Library::load(&p, &f, 8).unwrap();
        assert_eq!(lib.n_candidates(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn non_positive_charge_is_rejected() {
        // `ISOTOPE_SPACING / z` divides by charge in three places in extract, so a 0
        // collapses every isotope channel onto one m/z. A NULL in an integer column used
        // to decode as the raw buffer value -- in practice 0 -- so an imported library
        // with a missing charge produced exactly that, silently.
        let dir = unique_dir("bad_charge");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("prec.parquet").to_str().unwrap().to_string();
        let f = dir.join("frag.parquet").to_str().unwrap().to_string();
        write_table(
            &p,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1]),
                Col::U32("peptidoform_id".into(), vec![0, 1]),
                Col::U32("base_peptide_id".into(), vec![0, 1]),
                Col::Str(
                    "peptidoform".into(),
                    vec!["PEPTIDEK".into(), "SAMPLER".into()],
                ),
                Col::I32("charge".into(), vec![2, 0]),
                Col::F64("precursor_mz".into(), vec![400.0, 500.0]),
                Col::F32("predicted_irt".into(), vec![10.0, 20.0]),
                Col::Str("label".into(), vec!["target".into(), "decoy".into()]),
                Col::Str("protein".into(), vec!["P1".into(), "P2".into()]),
                Col::I32("n_fragments".into(), vec![1, 1]),
            ],
        )
        .unwrap();
        write_table(
            &f,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1]),
                Col::F64("mz".into(), vec![200.1, 250.5]),
                Col::F32("predicted_intensity".into(), vec![1.0, 0.9]),
                Col::Str("name".into(), vec!["b2".into(), "y3".into()]),
                Col::Str("ion_type".into(), vec!["b".into(), "y".into()]),
                Col::I32("ordinal".into(), vec![2, 3]),
                Col::I32("frag_charge".into(), vec![1, 1]),
            ],
        )
        .unwrap();
        let err = match Library::load(&p, &f, 8) {
            Ok(_) => panic!("charge 0 must be rejected at load"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("'charge'"), "{err}");
        assert!(err.contains("row 1"), "should name the row: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A two-candidate library whose string columns are the caller's, for the two
    /// empty-string cases below.
    fn library_with(
        dir: &std::path::Path,
        peptidoforms: [&str; 2],
        proteins: [&str; 2],
    ) -> (String, String) {
        std::fs::create_dir_all(dir).unwrap();
        let p = dir.join("prec.parquet").to_str().unwrap().to_string();
        let f = dir.join("frag.parquet").to_str().unwrap().to_string();
        write_table(
            &p,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1]),
                Col::U32("peptidoform_id".into(), vec![0, 1]),
                Col::U32("base_peptide_id".into(), vec![0, 1]),
                Col::Str(
                    "peptidoform".into(),
                    vec![peptidoforms[0].into(), peptidoforms[1].into()],
                ),
                Col::I32("charge".into(), vec![2, 2]),
                Col::F64("precursor_mz".into(), vec![400.0, 500.0]),
                Col::F32("predicted_irt".into(), vec![10.0, 20.0]),
                Col::Str("label".into(), vec!["target".into(), "decoy".into()]),
                Col::Str(
                    "protein".into(),
                    vec![proteins[0].into(), proteins[1].into()],
                ),
                Col::I32("n_fragments".into(), vec![1, 1]),
            ],
        )
        .unwrap();
        write_table(
            &f,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1]),
                Col::F64("mz".into(), vec![200.1, 250.5]),
                Col::F32("predicted_intensity".into(), vec![1.0, 0.9]),
                Col::Str("name".into(), vec!["b2".into(), "y3".into()]),
                Col::Str("ion_type".into(), vec!["b".into(), "y".into()]),
                Col::I32("ordinal".into(), vec![2, 3]),
                Col::I32("frag_charge".into(), vec![1, 1]),
            ],
        )
        .unwrap();
        (p, f)
    }

    /// A two-candidate library whose `candidate_id` and `label` columns are the caller's,
    /// for the two streaming-validated columns below.
    fn library_ids_labels(
        dir: &std::path::Path,
        ids: [u32; 2],
        labels: [&str; 2],
    ) -> (String, String) {
        std::fs::create_dir_all(dir).unwrap();
        let p = dir.join("prec.parquet").to_str().unwrap().to_string();
        let f = dir.join("frag.parquet").to_str().unwrap().to_string();
        write_table(
            &p,
            vec![
                Col::U32("candidate_id".into(), ids.to_vec()),
                Col::U32("peptidoform_id".into(), vec![0, 1]),
                Col::U32("base_peptide_id".into(), vec![0, 1]),
                Col::Str(
                    "peptidoform".into(),
                    vec!["PEPTIDEK".into(), "SAMPLER".into()],
                ),
                Col::I32("charge".into(), vec![2, 2]),
                Col::F64("precursor_mz".into(), vec![400.0, 500.0]),
                Col::F32("predicted_irt".into(), vec![10.0, 20.0]),
                Col::Str(
                    "label".into(),
                    vec![labels[0].to_string(), labels[1].to_string()],
                ),
                Col::Str("protein".into(), vec!["P1".into(), "P2".into()]),
            ],
        )
        .unwrap();
        write_table(
            &f,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1]),
                Col::F64("mz".into(), vec![200.1, 250.5]),
                Col::F32("predicted_intensity".into(), vec![1.0, 0.9]),
                Col::Str("name".into(), vec!["b2".into(), "y3".into()]),
            ],
        )
        .unwrap();
        (p, f)
    }

    /// The `label` column is read as a boolean per row instead of a `String` per row (203M
    /// heap blocks and ~6.5 GB on the full library, to set one bit each). The bit must be
    /// exactly what `label[c] == "decoy"` produced, and an unexpected value must still be
    /// refused by `fdr::validate_labels`'s rule and message.
    #[test]
    fn the_label_column_becomes_the_boolean_it_always_meant() {
        let dir = unique_dir("label_bool");
        let (p, f) = library_ids_labels(&dir, [0, 1], ["target", "decoy"]);
        let lib = Library::load_with(&p, &f, 8, false).unwrap();
        assert_eq!(
            lib.cands.iter().map(|c| c.is_decoy).collect::<Vec<_>>(),
            vec![false, true]
        );
        // Reversed, so a constant-false or index-shifted read cannot pass both cases.
        let (p2, f2) = library_ids_labels(&unique_dir("label_bool2"), [0, 1], ["decoy", "target"]);
        let lib2 = Library::load_with(&p2, &f2, 8, false).unwrap();
        assert_eq!(
            lib2.cands.iter().map(|c| c.is_decoy).collect::<Vec<_>>(),
            vec![true, false]
        );
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(unique_dir("label_bool2")).ok();
    }

    #[test]
    fn an_unknown_label_is_still_refused_by_name() {
        let dir = unique_dir("label_bad");
        let (p, f) = library_ids_labels(&dir, [0, 1], ["target", "REVERSE"]);
        let err = match Library::load_with(&p, &f, 8, false) {
            Ok(_) => panic!("an unknown label must be rejected at load"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("unknown PSM label"), "{err}");
        assert!(err.contains("REVERSE"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `candidate_id` is verified in a streaming pass and never held (4 B per precursor,
    /// 812 MB on the full library, live for the whole fragment load). The check itself is
    /// unchanged: row-aligned or refused, naming the row.
    #[test]
    fn a_non_contiguous_candidate_id_is_rejected_with_the_row() {
        let dir = unique_dir("cid_gap");
        let (p, f) = library_ids_labels(&dir, [0, 2], ["target", "decoy"]);
        let err = match Library::load_with(&p, &f, 8, false) {
            Ok(_) => panic!("a non-contiguous candidate_id must be rejected at load"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("contiguous range"), "{err}");
        assert!(
            err.contains("row 1"),
            "should name the offending row: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A range load checks the ids against the FILE row, not the local one, and the
    /// streaming check must keep that: local id c is file row c + offset.
    #[test]
    fn the_streamed_id_check_uses_the_file_row_for_a_range_load() {
        let dir = std::env::temp_dir().join(format!("mumdia_index_cidspan_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, f) = build_six_lib(&dir, "cidspan", &[0, 1, 2, 3, 4, 5]);
        // Rows 1..4 carry file ids 1, 2, 3; the load accepts them against offset 1.
        let part = Library::load_range_with(&p, &f, 440.0, 530.0, 8, false).unwrap();
        assert_eq!(part.n_candidates(), 3);
        assert_eq!(
            part.cands
                .iter()
                .map(|c| c.candidate_id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    /// `prec_mz` is now the moved `precursor_mz` column rather than a second copy of it.
    /// It must still be exactly the per-candidate value, since `candidate_range` and the
    /// fragment index both index it by candidate id.
    #[test]
    fn prec_mz_is_the_per_candidate_precursor_mz() {
        let dir = std::env::temp_dir().join(format!("mumdia_index_precmz_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, f) = build_six_lib(&dir, "precmz", &[0, 1, 2, 3, 4, 5]);
        let lib = Library::load_with(&p, &f, 8, false).unwrap();
        assert_eq!(lib.prec_mz.len(), lib.n_candidates());
        for c in 0..lib.n_candidates() {
            assert_eq!(lib.prec_mz[c], lib.cands[c].precursor_mz);
        }
        assert_eq!(lib.prec_mz, vec![400.0, 450.0, 500.0, 520.0, 600.0, 650.0]);
        // The same holds for a band, whose `prec_mz` is the band's slice of the column.
        let part = Library::load_range_with(&p, &f, 440.0, 530.0, 8, false).unwrap();
        assert_eq!(part.prec_mz, vec![450.0, 500.0, 520.0]);
        for c in 0..part.n_candidates() {
            assert_eq!(part.prec_mz[c], part.cands[c].precursor_mz);
        }
    }

    /// Releasing the fragment payload frees `frag_int`/`frag_name_id` and keeps `frag_mz`
    /// byte for byte, because the seed's mass recalibration still walks every library
    /// fragment m/z after its index exists.
    #[test]
    fn releasing_the_fragment_payload_keeps_every_fragment_mz() {
        let dir = std::env::temp_dir().join(format!("mumdia_index_release_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, f) = build_six_lib(&dir, "release", &[0, 1, 2, 3, 4, 5]);
        let mut lib = Library::load_with(&p, &f, 8, false).unwrap();
        assert!(!lib.fragment_payload_released());
        let before: Vec<Vec<f32>> = (0..lib.n_candidates())
            .map(|c| lib.cand_frag_mz(c as u32).to_vec())
            .collect();
        // Before the release the two accessors agree, which is what makes the swap safe.
        for c in 0..lib.n_candidates() {
            assert_eq!(lib.cand_frags(c as u32).0, lib.cand_frag_mz(c as u32));
        }
        lib.release_fragment_payload();
        assert!(lib.fragment_payload_released());
        assert!(lib.frag_int.is_empty() && lib.frag_name_id.is_empty());
        let after: Vec<Vec<f32>> = (0..lib.n_candidates())
            .map(|c| lib.cand_frag_mz(c as u32).to_vec())
            .collect();
        assert_eq!(before, after);
    }

    /// The released payload must fail loudly rather than hand back a short or stale slice:
    /// `cand_frags` is what extract reads predicted intensities through.
    #[test]
    #[should_panic(expected = "fragment payload was released")]
    fn cand_frags_after_a_release_panics_instead_of_lying() {
        let dir =
            std::env::temp_dir().join(format!("mumdia_index_relpanic_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, f) = build_six_lib(&dir, "relpanic", &[0, 1, 2, 3, 4, 5]);
        let mut lib = Library::load_with(&p, &f, 8, false).unwrap();
        lib.release_fragment_payload();
        let _ = lib.cand_frags(0);
    }

    #[test]
    fn an_empty_protein_loads_as_the_unassigned_group() {
        // DIA-NN leaves the protein empty for peptides it did not map (the iRT-kit
        // standards), so every real DIA-NN library carries a few dozen such rows. They
        // load, named rather than anonymous: an empty protein would have silently joined
        // every such candidate into one group, and refusing the library served nobody.
        let dir = unique_dir("empty_protein");
        let (p, f) = library_with(&dir, ["PEPTIDEK", "SAMPLER"], ["P1", ""]);
        let lib = Library::load(&p, &f, 8).expect("an empty protein is not a load error");
        assert_eq!(lib.cands[0].protein, "P1");
        assert_eq!(lib.cands[1].protein, "UNASSIGNED");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_empty_peptidoform_is_rejected() {
        // Unlike the protein, an empty peptidoform has no meaning at all: it cannot be
        // parsed into residues, so the accessor's NULL rejection is extended to the
        // explicitly empty string.
        let dir = unique_dir("empty_peptidoform");
        let (p, f) = library_with(&dir, ["PEPTIDEK", ""], ["P1", "P2"]);
        let err = match Library::load(&p, &f, 8) {
            Ok(_) => panic!("an empty peptidoform must be rejected at load"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("'peptidoform'"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod null_fixture_tests {
    use super::*;
    use arrow::array::{Float32Array, Float64Array, Int32Array, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use mumdia_io::table::{write_batches, write_table, Col};
    use std::sync::Arc;

    fn precursors(dir: &std::path::Path) -> String {
        let p = dir.join("prec.parquet").to_str().unwrap().to_string();
        write_table(
            &p,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1]),
                Col::U32("peptidoform_id".into(), vec![0, 1]),
                Col::U32("base_peptide_id".into(), vec![0, 1]),
                Col::Str(
                    "peptidoform".into(),
                    vec!["PEPTIDEK".into(), "SAMPLER".into()],
                ),
                Col::I32("charge".into(), vec![2, 2]),
                Col::F64("precursor_mz".into(), vec![400.0, 500.0]),
                Col::F32("predicted_irt".into(), vec![10.0, 20.0]),
                Col::Str("label".into(), vec!["target".into(), "decoy".into()]),
                Col::Str("protein".into(), vec!["P1".into(), "P2".into()]),
                Col::I32("n_fragments".into(), vec![1, 1]),
            ],
        )
        .unwrap();
        p
    }

    /// A two-row fragment table with one NULL in `null_in`.
    fn fragments_with_null(dir: &std::path::Path, null_in: &str) -> String {
        let f = dir
            .join(format!("frag_{null_in}.parquet"))
            .to_str()
            .unwrap()
            .to_string();
        let cid = UInt32Array::from(if null_in == "candidate_id" {
            vec![Some(0u32), None]
        } else {
            vec![Some(0u32), Some(1)]
        });
        let mz = Float64Array::from(if null_in == "mz" {
            vec![Some(200.1), None]
        } else {
            vec![Some(200.1), Some(250.5)]
        });
        let inten = Float32Array::from(if null_in == "predicted_intensity" {
            vec![None, Some(0.9f32)]
        } else {
            vec![Some(1.0f32), Some(0.9)]
        });
        let name = StringArray::from(if null_in == "name" {
            vec![Some("b2"), None]
        } else {
            vec![Some("b2"), Some("y3")]
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("candidate_id", DataType::UInt32, true),
            Field::new("mz", DataType::Float64, true),
            Field::new("predicted_intensity", DataType::Float32, true),
            Field::new("name", DataType::Utf8, true),
            Field::new("ion_type", DataType::Utf8, false),
            Field::new("ordinal", DataType::Int32, false),
            Field::new("frag_charge", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(cid),
                Arc::new(mz),
                Arc::new(inten),
                Arc::new(name),
                Arc::new(StringArray::from(vec!["b", "y"])),
                Arc::new(Int32Array::from(vec![2, 3])),
                Arc::new(Int32Array::from(vec![1, 1])),
            ],
        )
        .unwrap();
        write_batches(&f, schema, &[batch]).unwrap();
        f
    }

    #[test]
    fn a_null_in_any_required_fragment_column_is_refused_by_name() {
        // Before docs/29 #2 a NULL m/z or intensity loaded as NaN (the finiteness check
        // looked at physical buffers) and a NULL candidate_id attached the fragment to
        // candidate 0.
        let dir = std::env::temp_dir().join(format!("mumdia_index_nulls_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = precursors(&dir);
        for col in ["candidate_id", "mz", "predicted_intensity", "name"] {
            let f = fragments_with_null(&dir, col);
            let err = match Library::load(&p, &f, 8) {
                Ok(_) => panic!("a NULL {col} must not load"),
                Err(e) => format!("{e:#}"),
            };
            assert!(err.contains(&format!("'{col}'")), "{col}: {err}");
            assert!(err.contains("NULL"), "{col}: {err}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
