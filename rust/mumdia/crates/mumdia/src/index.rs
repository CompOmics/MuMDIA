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
use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, StringArray, UInt32Array};
use mumdia_core::constants::{ppm_bounds, PROTON};
use mumdia_io::table::{require_no_nulls, TableFile};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Fragment rows per decoded batch while streaming the fragment table (a few MB).
const FRAG_BATCH_ROWS: usize = 1 << 16;

/// Most parts a fragment table is decoded in at once. A part decoder holds its own page
/// buffers and batches (about 4-5 MB each on the AIF library in 1M-row groups: 16 parts
/// put 70 MB on the seed peak), so the count is bounded rather than left to the thread
/// count of a large host.
const LOAD_PARTS_MAX: usize = 16;

/// Precursor rows per decoded batch for the two columns that are validated in a streaming
/// pass and never kept (`label`, `candidate_id`).
const PREC_BATCH_ROWS: usize = 1 << 16;

/// One candidate as an owned record: the input of [`Library::from_candidates`], which is how
/// tests and benchmarks build a library in memory. A loaded library does not hold these; it
/// holds the same fields as columns (see [`Library`]), and [`Library::cand`] reads one back
/// as a borrowed [`CandInfo`].
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

/// One candidate's precursor fields, borrowed from the library's columns.
#[derive(Clone, Copy, Debug)]
pub struct CandInfo<'a> {
    pub peptidoform_id: u32,
    pub base_peptide_id: u32,
    pub peptidoform: &'a str,
    pub charge: i32,
    pub precursor_mz: f64,
    pub predicted_irt: f32,
    pub is_decoy: bool,
    pub protein: &'a str,
}

/// A spectral library in memory, structure-of-arrays, indexed by LOCAL candidate id
/// (`0..n_candidates()`, the row of the precursor table this library was loaded from).
///
/// Per candidate it is plain columns rather than one struct per candidate. The struct it
/// replaced carried two owned `String`s (a heap block each), a copy of `precursor_mz` next
/// to `prec_mz`, its own position as `candidate_id`, and two `usize` fragment bounds: about
/// 96 bytes of struct plus two allocations per precursor, ~180 bytes in all, against about
/// 60 here. `peptidoform` is one arena ([`Library::peptidoform`]), `protein` is interned
/// ([`Library::protein`]; a library has far fewer protein groups than precursors), the
/// fragment bounds are one `u32` CSR array ([`Library::frag_offsets`]), and `prec_mz` is
/// shared with the fragment index instead of copied into it.
pub struct Library {
    pub peptidoform_id: Vec<u32>,
    pub base_peptide_id: Vec<u32>,
    pub charge: Vec<i32>,
    pub predicted_irt: Vec<f32>,
    pub is_decoy: Vec<bool>,
    /// Peptidoform text, concatenated: candidate `c` is
    /// `pform_data[pform_offsets[c]..pform_offsets[c + 1]]`.
    pform_offsets: Vec<usize>,
    pform_data: String,
    /// Protein (group) per candidate, as an id into `protein_dict`.
    protein_id: Vec<u32>,
    protein_dict: Vec<String>,
    /// CSR fragment offsets, `n_candidates() + 1` entries: candidate `c`'s fragments are
    /// `frag_offsets[c]..frag_offsets[c + 1]` of the fragment arrays below.
    pub frag_offsets: Vec<u32>,
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
    /// precursor m/z indexed by candidate_id (ascending). Shared, not copied, with every
    /// [`crate::matchers::fragindex::FragIndex`] built from this library.
    pub prec_mz: Arc<[f64]>,
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

/// An interned fragment-name id as the `u16` the library stores. The first name past
/// `u16::MAX` distinct ones is refused, as the per-row map refused it.
fn frag_name_id_u16(id: u32) -> Result<u16> {
    u16::try_from(id).map_err(|_| {
        anyhow::anyhow!(
            "library has more than {} distinct fragment names; the interned name id is a u16",
            u16::MAX
        )
    })
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

/// The per-candidate columns of a precursor table, validated.
struct PrecursorColumns {
    peptidoform_id: Vec<u32>,
    base_peptide_id: Vec<u32>,
    charge: Vec<i32>,
    precursor_mz: Vec<f64>,
    predicted_irt: Vec<f32>,
    is_decoy: Vec<bool>,
    pform_offsets: Vec<usize>,
    pform_data: String,
    protein_id: Vec<u32>,
    protein_dict: Vec<String>,
}

/// Read and validate the precursor table's columns: the whole file, or the row span of a
/// range load, whose ids are checked against the FILE row (`offset`).
///
/// The eight column reads and the `candidate_id` check are independent, so they run
/// concurrently (`rayon::scope`, one task per column, each with its own reader); a
/// one-thread pool runs them in turn. Their results are then taken in the order the
/// serial load read them, and every check runs in its old order, so a table with more than
/// one problem is refused with the same message as before.
fn load_precursors(
    precursors: &str,
    span: Option<(usize, usize)>,
    offset: usize,
) -> Result<PrecursorColumns> {
    let pt = match span {
        None => TableFile::open(precursors)?,
        Some((first, n)) => TableFile::open_rows(precursors, first, n)?,
    };
    let ncand = pt.nrows;
    let (mut r_pfid, mut r_base, mut r_pform, mut r_charge) = (None, None, None, None);
    let (mut r_pmz, mut r_irt, mut r_prot, mut r_dec, mut r_ids) = (None, None, None, None, None);
    {
        let pt = &pt;
        rayon::scope(|s| {
            s.spawn(|_| r_pfid = Some(pt.u32("peptidoform_id")));
            s.spawn(|_| r_base = Some(pt.u32("base_peptide_id")));
            // One arena for the peptidoform text instead of one `String` per precursor.
            s.spawn(|_| r_pform = Some(pt.str_flat("peptidoform")));
            s.spawn(|_| r_charge = Some(pt.i32("charge")));
            s.spawn(|_| r_pmz = Some(pt.f64("precursor_mz")));
            s.spawn(|_| r_irt = Some(pt.f32("predicted_irt")));
            // Interned: a library has far fewer protein groups than precursors.
            s.spawn(|_| r_prot = Some(pt.str_interned("protein")));
            // `label` is validated and reduced to a bit in one streaming pass rather than
            // being held as a `Vec<String>`.
            s.spawn(|_| r_dec = Some(read_is_decoy(pt, precursors)));
            // Precondition: candidate_id is the contiguous, row-aligned range 0..ncand
            // (the library + decoy builders guarantee this). An external library that
            // violates it would misgroup fragments or panic on the index below, so it is
            // checked explicitly -- its verdict is reported after the column checks, where
            // the serial load reported it. For a range load the same invariant holds
            // against the file row: local id c is file row c + offset.
            s.spawn(|_| r_ids = Some(check_candidate_ids(pt, precursors, offset, ncand)));
        });
    }
    let taken = "the scope ran every read";
    let peptidoform_id = r_pfid.expect(taken)?;
    let base_peptide_id = r_base.expect(taken)?;
    let (pform_offsets, mut pform_data) = r_pform.expect(taken)?;
    let charge = r_charge.expect(taken)?;
    let precursor_mz = r_pmz.expect(taken)?;
    let predicted_irt = r_irt.expect(taken)?;
    let (protein_id, mut protein_dict) = r_prot.expect(taken)?;
    let is_decoy = r_dec.expect(taken)?;
    // `str_flat` grows its buffer by doubling; the arena lives for the whole search.
    pform_data.shrink_to_fit();
    let pform = |i: usize| &pform_data[pform_offsets[i]..pform_offsets[i + 1]];
    // A Parquet NULL decodes to NaN (mumdia-io `Table::f64`/`f32`), and NaN is
    // accepted rather than rejected by every downstream guard that should catch it:
    // the ascending-m/z check is `<`, the extract RT-window guards are
    // `rt < lo || rt > hi`, and `within_ppm` compares against `min`/`max`, all of
    // which are false for NaN. So a single empty cell in a converted third-party
    // library does not produce a NaN result, it produces a candidate that is absent
    // from its own isolation window or one that matches every peak. Reject at load,
    // where the offending row can still be named.
    require_finite_f64(&precursor_mz, "precursor_mz", precursors)?;
    require_finite_f32(&predicted_irt, "predicted_irt", precursors)?;
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
    if let Some(row) = (0..ncand).position(|i| pform(i).trim().is_empty()) {
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
    //
    // The protein column is interned, so the rename happens on the dictionary: every
    // blank VALUE becomes `UNASSIGNED`, and every row holding one reads it.
    let blank: Vec<bool> = protein_dict.iter().map(|v| v.trim().is_empty()).collect();
    if blank.iter().any(|&b| b) {
        let unassigned: Vec<usize> = protein_id
            .iter()
            .enumerate()
            .filter(|(_, &p)| blank[p as usize])
            .map(|(i, _)| i)
            .collect();
        let examples: Vec<&str> = unassigned.iter().take(3).map(|&i| pform(i)).collect();
        tracing::warn!(
            rows = unassigned.len(),
            examples = ?examples,
            library = precursors,
            "library: rows with an empty protein are grouped as UNASSIGNED (typically \
             the iRT-kit standards); re-import with scripts/import_diann_lib.py to \
             make the group explicit in the file"
        );
        for (v, &b) in protein_dict.iter_mut().zip(&blank) {
            if b {
                *v = "UNASSIGNED".to_string();
            }
        }
    }
    r_ids.expect(taken)?;
    Ok(PrecursorColumns {
        peptidoform_id,
        base_peptide_id,
        charge,
        precursor_mz,
        predicted_irt,
        is_decoy,
        pform_offsets,
        pform_data,
        protein_id,
        protein_dict,
    })
}

/// The fragment columns of a library, grouped by candidate.
#[derive(Debug)]
struct FragmentColumns {
    frag_offsets: Vec<u32>,
    frag_mz: Vec<f32>,
    frag_int: Vec<f32>,
    frag_name_id: Vec<u16>,
    frag_name_dict: Vec<String>,
}

/// Read the fragment rows of the candidates `[frag_offset, frag_offset + ncand)` and group
/// them by candidate.
///
/// Two streaming passes over the four columns the library needs (the artifact also carries
/// `ion_type`, `ordinal`, `frag_charge` and `cardinality`, which are never fetched). Pass 1
/// decodes only `candidate_id` and counts fragments per candidate; pass 2 decodes the four
/// columns batch by batch and places each row in its final grouped slot, interning the
/// fragment name on the way.
///
/// This is a counting sort -- rows placed in ascending file order keep each candidate's
/// fragments in stored order -- with the file as the source instead of owned copies: no
/// whole-table Arrow batches (23 GB at 657M rows), no owned copy of the four columns, no
/// permutation and no `Vec<String>` with one heap allocation per fragment. The resident
/// peak is the final arrays plus the batches in flight, which is what lets a
/// modification-expanded library load on a 32 GB machine. A partial load (a range of one
/// file, or a band file against the shared fragment table) sees fragments of other
/// candidates and skips them.
///
/// Parallel, in two ways, with the arrays bit-identical to the serial pass:
///
/// - The table is cut into row-contiguous parts ([`TableFile::row_parts`]: row groups,
///   and page-aligned ranges inside a group that has an offset index). Pass 1 counts the
///   parts concurrently into shared atomic counters (integer sums, so the order does not
///   matter) and records whether each part's in-range ids are ascending.
/// - When they are ascending across the whole table -- the layout every library writer
///   produces, and what `scripts/sort_fragments.py` restores -- the counting sort is the
///   identity: a kept row's slot is its rank among the kept rows. Each part then owns one
///   contiguous slice of the output (`split_at_mut`, no shared writes), fills it, and
///   interns names locally; the local dictionaries are merged in part order, which is
///   first appearance in file order, and the ids remapped. Otherwise (an unsorted table,
///   or one part) pass 2 is the serial scatter.
///
/// Within a part, and on the serial path, the columns are decoded with one reader each,
/// in parallel and one batch ahead of the placement ([`crate::colread::for_each_zipped`]).
/// That is the only parallelism a one-row-group table without an offset index has, which
/// is what pyarrow writes by default.
///
/// Errors do not depend on the part layout. Pass 1 refuses `candidate_id` defects (NULL,
/// out of range on a full load), pass 2 the others (NULLs, non-finite m/z or intensity).
/// Within a pass the table is refused on its FIRST defective row in file order, whatever
/// the kind of defect: every batch reports its first bad row across all its checks, and
/// the parts' results are taken in part order. The name limit (more distinct names than
/// the u16 id holds) is the one check that belongs to no row. The serial pass meets it at
/// the row that introduces the 65,536th name, the parted pass at the merge of the parts'
/// dictionaries, so in a table that also has a row defect, which of the two is reported
/// can depend on the layout. Both are refused either way.
fn load_fragments(
    fragments: &str,
    frag_offset: usize,
    ncand: usize,
    partial: bool,
    max_parts: usize,
    payload: bool,
) -> Result<FragmentColumns> {
    let ft = if partial {
        Library::open_fragments_for(fragments, frag_offset, ncand)?
    } else {
        TableFile::open(fragments)?
    };
    if ft.nrows > u32::MAX as usize {
        anyhow::bail!(
            "fragment library has {} rows; per-candidate fragment offsets are u32",
            ft.nrows
        );
    }
    let range = FragRange {
        fragments,
        frag_offset,
        ncand,
        partial,
    };
    let parts = ft.row_parts(max_parts)?;
    // First handle row of each part, for the error messages.
    let part_row0: Vec<usize> = parts
        .iter()
        .scan(0usize, |acc, p| {
            let r = *acc;
            *acc += p.nrows;
            Some(r)
        })
        .collect();

    // Pass 1.
    let (mut frag_offsets, infos) = if parts.len() > 1 {
        let counts: Vec<AtomicU32> = (0..=ncand).map(|_| AtomicU32::new(0)).collect();
        let infos = crate::colread::first_err(
            parts
                .par_iter()
                .zip(part_row0.par_iter())
                .map(|(part, &row0)| {
                    count_part(part, row0, &range, |c| {
                        counts[c + 1].fetch_add(1, Ordering::Relaxed);
                    })
                })
                .collect(),
        )?;
        // In place: `AtomicU32` and `u32` share size and alignment.
        let counts: Vec<u32> = counts.into_iter().map(AtomicU32::into_inner).collect();
        (counts, infos)
    } else {
        let mut counts: Vec<u32> = vec![0; ncand + 1];
        let info = count_part(&ft, 0, &range, |c| counts[c + 1] += 1)?;
        (counts, vec![info])
    };
    for c in 0..ncand {
        frag_offsets[c + 1] += frag_offsets[c];
    }
    // Rows in range, which for a range load is fewer than the rows decoded.
    let n_frag_rows = frag_offsets[ncand] as usize;

    // The m/z-only load still DECODES and CHECKS all four columns, through the same pass
    // and the same checks as the full load, and then discards the two payload columns
    // instead of storing them. What it saves is the 6 bytes per fragment those two arrays
    // would hold for the life of the seed (and 6 more in the index), not their decode: a
    // library whose intensities or names are NULL or non-finite, or that has more distinct
    // fragment names than the u16 id holds, is refused at the seed in seconds with the full
    // load's message, rather than by extract after the DeepLC calibration or fine-tune
    // that runs between the two.
    let mut frag_mz: Vec<f32> = vec![0.0; n_frag_rows];
    let mut frag_int: Vec<f32> = if payload {
        vec![0.0; n_frag_rows]
    } else {
        Vec::new()
    };
    // Fragment names are INTERNED (see the struct field docs): a u16 dictionary id per
    // fragment, assigned by first appearance in file order.
    let mut frag_name_id: Vec<u16> = if payload {
        vec![0; n_frag_rows]
    } else {
        Vec::new()
    };
    let out = FragOut {
        mz: &mut frag_mz,
        int: payload.then_some(&mut frag_int[..]),
        name_id: payload.then_some(&mut frag_name_id[..]),
    };
    let frag_name_dict = if parts.len() > 1 && ascending_across(&infos) {
        place_sorted_parts(&parts, &part_row0, &infos, &range, out)?
    } else {
        scatter_serial(&ft, &range, &frag_offsets, out)?
    };
    Ok(FragmentColumns {
        frag_offsets,
        frag_mz,
        frag_int,
        frag_name_id,
        frag_name_dict,
    })
}

/// The columns pass 2 reads, on every load: the m/z-only load reads the payload too, to
/// check it, and does not store it.
const PASS2_COLUMNS: &[&str] = &["candidate_id", "mz", "predicted_intensity", "name"];

/// Pass 2's output arrays; the payload ones are `None` on an m/z-only load.
struct FragOut<'a> {
    mz: &'a mut [f32],
    int: Option<&'a mut [f32]>,
    name_id: Option<&'a mut [u16]>,
}

/// Which candidates a fragment load keeps, and how it names its input.
struct FragRange<'a> {
    fragments: &'a str,
    frag_offset: usize,
    ncand: usize,
    partial: bool,
}

impl FragRange<'_> {
    /// Local candidate of a fragment row's id, `None` for a row a partial load skips, and
    /// the error a full load raises for an id past the end (`row` is the handle row).
    #[inline]
    fn local(&self, c: usize, row: usize) -> Result<Option<usize>> {
        if c < self.frag_offset || c >= self.frag_offset + self.ncand {
            if self.partial {
                return Ok(None);
            }
            anyhow::bail!(
                "fragment row {row} references candidate_id {c} >= precursor count {}",
                self.ncand
            );
        }
        Ok(Some(c - self.frag_offset))
    }
}

/// What pass 1 learned about one part of the fragment table.
#[derive(Clone, Copy, Debug)]
struct PartInfo {
    /// Rows of the part the load keeps.
    kept: usize,
    /// Local candidate of the first and last kept row.
    first: Option<u32>,
    last: Option<u32>,
    /// Whether the kept rows' candidates never decrease within the part.
    ascending: bool,
}

/// Pass 1 over one part: count each kept row's candidate through `bump`.
fn count_part(
    part: &TableFile,
    row0: usize,
    range: &FragRange,
    mut bump: impl FnMut(usize) + Send,
) -> Result<PartInfo> {
    let mut info = PartInfo {
        kept: 0,
        first: None,
        last: None,
        ascending: true,
    };
    crate::colread::for_each_zipped(
        part,
        &["candidate_id"],
        &[],
        FRAG_BATCH_ROWS,
        |base, cols| {
            let a = cols[0]
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| anyhow::anyhow!("fragment column 'candidate_id' is not u32"))?;
            // `values()` is the physical buffer and ignores the validity bitmap: a NULL
            // candidate_id would read as 0 and attach the fragment to candidate 0
            // (docs/29 #2). So only the rows before the first NULL are read, and the batch
            // is refused on whichever defect comes first in the file: an id out of range
            // in those rows, or the NULL. Checking every NULL of the batch first would name
            // a later row whenever a batch holds both, and which one would depend on where
            // the batch boundaries fall, which differs between the serial and the parted
            // pass.
            let clean = first_null(a).unwrap_or(a.len());
            for (k, &candidate_id) in a.values()[..clean].iter().enumerate() {
                // A range load sees the fragments of neighbouring candidates in the
                // boundary row groups (or the whole table when it is unsorted); they
                // belong to precursors this library does not hold and are skipped. A
                // full load has no such rows, so an id past the end is a broken file.
                let Some(c) = range.local(candidate_id as usize, row0 + base + k)? else {
                    continue;
                };
                bump(c);
                let c = c as u32;
                if info.last.is_some_and(|l| c < l) {
                    info.ascending = false;
                }
                info.first.get_or_insert(c);
                info.last = Some(c);
                info.kept += 1;
            }
            // The NULL at `clean`, if there is one.
            require_no_nulls(a, "candidate_id", range.fragments, row0 + base)
        },
    )?;
    Ok(info)
}

/// First NULL row of an array (its validity bitmap), `None` when it has none.
fn first_null(a: &dyn Array) -> Option<usize> {
    if a.null_count() == 0 {
        return None;
    }
    (0..a.len()).find(|&i| a.is_null(i))
}

/// First row of a string batch whose dictionary VALUE is NULL behind a valid key: the one
/// NULL a validity check on the column cannot see. A parquet reader never produces one (a
/// parquet dictionary holds no NULLs; a missing value is a NULL key), but an Arrow producer
/// can, and the row would otherwise read as whatever the value buffer holds.
fn first_null_dict_value(col: &dyn Array, name: &mumdia_io::table::StrBatch) -> Option<usize> {
    match name {
        mumdia_io::table::StrBatch::Plain(_) => None,
        mumdia_io::table::StrBatch::Dict { keys, values } => {
            if values.null_count() == 0 {
                return None;
            }
            (0..keys.len()).find(|&k| !col.is_null(k) && values.is_null(keys[k] as usize))
        }
    }
}

/// Whether the kept rows' candidates never decrease over the whole table, parts in order.
fn ascending_across(infos: &[PartInfo]) -> bool {
    let mut prev: Option<u32> = None;
    for i in infos {
        if !i.ascending {
            return false;
        }
        if let (Some(p), Some(f)) = (prev, i.first) {
            if f < p {
                return false;
            }
        }
        if i.last.is_some() {
            prev = i.last;
        }
    }
    true
}

/// `v` cut into consecutive slices of the given lengths (which must sum to `v.len()`).
fn split_lens<'a, T>(mut v: &'a mut [T], lens: &[usize]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(lens.len());
    for &n in lens {
        let (head, tail) = std::mem::take(&mut v).split_at_mut(n);
        out.push(head);
        v = tail;
    }
    out
}

/// `split_lens` of an optional array: one `None` per part when it is absent.
fn split_opt<'a, T>(v: Option<&'a mut [T]>, lens: &[usize]) -> Vec<Option<&'a mut [T]>> {
    match v {
        Some(v) => split_lens(v, lens).into_iter().map(Some).collect(),
        None => lens.iter().map(|_| None).collect(),
    }
}

/// Pass 2 on an ascending table: every part fills its own slice, in parallel.
fn place_sorted_parts(
    parts: &[TableFile],
    part_row0: &[usize],
    infos: &[PartInfo],
    range: &FragRange,
    out: FragOut,
) -> Result<Vec<String>> {
    let lens: Vec<usize> = infos.iter().map(|i| i.kept).collect();
    let payload = out.int.is_some();
    let FragOut { mz, int, name_id } = out;
    let jobs: Vec<_> = parts
        .iter()
        .zip(part_row0)
        .zip(split_lens(mz, &lens))
        .zip(split_opt(int, &lens))
        .zip(split_opt(name_id, &lens))
        .map(|((((part, &row0), mz), int), ids)| (part, row0, mz, int, ids))
        .collect();
    let placed: Vec<(Vec<String>, Option<&mut [u16]>)> = crate::colread::first_err(
        jobs.into_par_iter()
            .map(|(part, row0, mz, int, ids)| place_part(part, row0, range, mz, int, ids))
            .collect(),
    )?;
    // Merge the parts' dictionaries in part order: that is first appearance over the kept
    // rows in file order, exactly the order the serial pass assigns. An m/z-only load
    // merges too, because the merge is where the table-wide name limit is checked, and
    // then drops the result.
    let mut names = mumdia_io::table::StrInterner::new();
    let mut remap: Vec<(Vec<u16>, &mut [u16])> = Vec::with_capacity(placed.len());
    for (local, ids) in placed {
        let map = local
            .iter()
            .map(|v| frag_name_id_u16(names.intern(v)))
            .collect::<Result<Vec<u16>>>()?;
        if let Some(ids) = ids {
            remap.push((map, ids));
        }
    }
    if !payload {
        return Ok(Vec::new());
    }
    remap.into_par_iter().for_each(|(map, ids)| {
        for id in ids.iter_mut() {
            *id = map[*id as usize];
        }
    });
    Ok(names.into_values())
}

/// One part of an ascending table, into its own output slices; returns its local name
/// dictionary (the slice holds local ids) and hands the id slice back for the remap.
fn place_part<'a>(
    part: &TableFile,
    row0: usize,
    range: &FragRange,
    mz_out: &mut [f32],
    mut int_out: Option<&mut [f32]>,
    mut id_out: Option<&'a mut [u16]>,
) -> Result<(Vec<String>, Option<&'a mut [u16]>)> {
    let mut names = mumdia_io::table::StrInterner::new();
    let mut j = 0usize;
    let changed = || anyhow::anyhow!("fragment table {} changed between passes", range.fragments);
    crate::colread::for_each_zipped(
        part,
        PASS2_COLUMNS,
        &["name"],
        FRAG_BATCH_ROWS,
        |base, cols| {
            let batch = FragBatch::of(cols, range.fragments, row0 + base)?;
            names.begin(&batch.name);
            for k in 0..batch.len() {
                // Pass 1 already refused an out-of-range id on a full load, so one here
                // means the file changed under the load.
                if range
                    .local(batch.cid.value(k) as usize, row0 + base + k)
                    .map_err(|_| changed())?
                    .is_none()
                {
                    continue;
                }
                if j >= mz_out.len() {
                    return Err(changed());
                }
                // NULLs were rejected above, so the physical values are the values.
                mz_out[j] = batch.mz.value(k) as f32;
                if let Some(int_out) = int_out.as_deref_mut() {
                    int_out[j] = batch.int.value(k);
                }
                // Interned on every load, stored on a payload load: the name limit is a
                // property of the table, not of what this load keeps.
                let id = batch.name_id(&mut names, k, range.fragments, row0 + base)?;
                let id = frag_name_id_u16(id)?;
                if let Some(id_out) = id_out.as_deref_mut() {
                    id_out[j] = id;
                }
                j += 1;
            }
            Ok(())
        },
    )?;
    if j != mz_out.len() {
        return Err(changed());
    }
    Ok((names.into_values(), id_out))
}

/// Pass 2 by scatter: each kept row to the next free slot of its candidate.
fn scatter_serial(
    ft: &TableFile,
    range: &FragRange,
    frag_offsets: &[u32],
    out: FragOut,
) -> Result<Vec<String>> {
    let FragOut {
        mz: frag_mz,
        mut int,
        mut name_id,
    } = out;
    let payload = int.is_some();
    // Interned through the column's dictionary (`batches_dict`): the name column is a few
    // hundred distinct values over every fragment row, so a row costs an i32 key lookup in
    // a per-batch memo rather than a SipHash of its text, and the ids are still assigned
    // in first-appearance order over the KEPT rows, exactly as the per-row map assigned
    // them.
    let mut names = mumdia_io::table::StrInterner::new();
    let mut cursor: Vec<u32> = frag_offsets.to_vec();
    crate::colread::for_each_zipped(
        ft,
        PASS2_COLUMNS,
        &["name"],
        FRAG_BATCH_ROWS,
        |base, cols| {
            let batch = FragBatch::of(cols, range.fragments, base)?;
            names.begin(&batch.name);
            for k in 0..batch.len() {
                let c = batch.cid.value(k) as usize;
                let Some(c) = range.local(c, base + k).map_err(|_| {
                    anyhow::anyhow!(
                        "fragment table changed between passes: candidate_id {c} >= {}",
                        range.ncand
                    )
                })?
                else {
                    continue;
                };
                let pos = cursor[c] as usize;
                cursor[c] += 1;
                // NULLs were rejected above, so the physical values are the values.
                frag_mz[pos] = batch.mz.value(k) as f32;
                if let Some(frag_int) = int.as_deref_mut() {
                    frag_int[pos] = batch.int.value(k);
                }
                // Interned on every load, stored on a payload load (see `place_part`).
                let id = batch.name_id(&mut names, k, range.fragments, base)?;
                let id = frag_name_id_u16(id)?;
                if let Some(frag_name_id) = name_id.as_deref_mut() {
                    frag_name_id[pos] = id;
                }
            }
            Ok(())
        },
    )?;
    Ok(if payload {
        names.into_values()
    } else {
        Vec::new()
    })
}

/// One decoded batch of the four fragment columns, type-checked, NULL-checked and
/// finiteness-checked. Every load reads and checks all four; an m/z-only load then stores
/// only the m/z.
struct FragBatch<'a> {
    cid: &'a UInt32Array,
    mz: &'a Float64Array,
    int: &'a Float32Array,
    name: mumdia_io::table::StrBatch<'a>,
}

impl<'a> FragBatch<'a> {
    /// `cols` is `[candidate_id, mz, predicted_intensity, name]` ([`PASS2_COLUMNS`]);
    /// `row_base` is the handle row of the batch's first row, for the messages.
    fn of(cols: &'a [ArrayRef], fragments: &str, row_base: usize) -> Result<FragBatch<'a>> {
        let a_cid = cols[0]
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| anyhow::anyhow!("fragment column 'candidate_id' is not u32"))?;
        let a_mz = cols[1]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow::anyhow!("fragment column 'mz' is not f64"))?;
        let a_int = cols[2]
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow::anyhow!("fragment column 'predicted_intensity' is not f32"))?;
        let name = mumdia_io::table::StrBatch::of(&cols[3])
            .ok_or_else(|| anyhow::anyhow!("fragment column 'name' is not utf8"))?;
        // NULLs in every required column: the finiteness checks below run on physical
        // buffers, where a NULL is a perfectly finite 0.0, and the fill used to turn NULLs
        // into NaN and "" instead of refusing them (docs/29 #2).
        //
        // Finiteness is the same contract as the precursor columns, applied batch by
        // batch. A non-finite fragment m/z is worse than a wrong value: `FragIndex::build`
        // collapses its whole m/z range when the observed min or max is not finite, which
        // clamps every real fragment into one bin and turns the probe into a linear scan
        // of the entire posting list. A non-finite predicted_intensity sorts ahead of every
        // real value under `total_cmp`, so it is preferentially selected for
        // quantification.
        //
        // Every check finds the first row it fails on, and the batch is refused on the
        // SMALLEST of those rows, a tie going to the check listed first (a NULL before a
        // non-finite value at the same row, whose physical value means nothing). Refusing
        // check by check instead -- every NULL of the batch before any non-finite value --
        // names a later row whenever a batch holds two kinds of defect, and which row it
        // names depends on where the batch boundaries fall, which differs between the
        // serial pass and the parted one. This way both report the first defective row of
        // the file.
        let first = [
            first_null(a_cid),
            first_null(a_mz),
            first_null(a_int),
            first_null(cols[3].as_ref()),
            first_null_dict_value(cols[3].as_ref(), &name),
            a_mz.values().iter().position(|x| !x.is_finite()),
            a_int.values().iter().position(|x| !x.is_finite()),
        ]
        .into_iter()
        .enumerate()
        .filter_map(|(check, row)| row.map(|row| (row, check)))
        .min();
        if let Some((k, check)) = first {
            match check {
                0 => require_no_nulls(a_cid, "candidate_id", fragments, row_base)?,
                1 => require_no_nulls(a_mz, "mz", fragments, row_base)?,
                2 => require_no_nulls(a_int, "predicted_intensity", fragments, row_base)?,
                3 => require_no_nulls(cols[3].as_ref(), "name", fragments, row_base)?,
                4 => anyhow::bail!(
                    "fragment column 'name' has a NULL dictionary value at row {} in {fragments}",
                    row_base + k
                ),
                5 => anyhow::bail!(
                    "library column 'mz' has a non-finite value ({}) for candidate_id {} in \
                     {fragments}; a Parquet NULL decodes to NaN, and a NaN here silently \
                     means \"matches everything\" downstream rather than an error. Fix or \
                     drop the row",
                    a_mz.value(k),
                    a_cid.value(k)
                ),
                _ => anyhow::bail!(
                    "library column 'predicted_intensity' has a non-finite value ({}) for \
                     candidate_id {} in {fragments}; a Parquet NULL decodes to NaN, and a \
                     NaN here sorts ahead of every real intensity. Fix or drop the row",
                    a_int.value(k),
                    a_cid.value(k)
                ),
            }
        }
        Ok(FragBatch {
            cid: a_cid,
            mz: a_mz,
            int: a_int,
            name,
        })
    }

    fn len(&self) -> usize {
        self.cid.len()
    }

    /// Interned id of row `k`'s fragment name.
    fn name_id(
        &self,
        names: &mut mumdia_io::table::StrInterner,
        k: usize,
        fragments: &str,
        row_base: usize,
    ) -> Result<u32> {
        names.row(&self.name, k).ok_or_else(|| {
            anyhow::anyhow!(
                "fragment column 'name' has a NULL dictionary value at row {} in {fragments}",
                row_base + k
            )
        })
    }
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
            true,
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
            true,
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
            true,
        )
    }

    /// The seed's library: the precursor columns and the fragment m/z, with NO fragment
    /// payload (`frag_int`, `frag_name_id`, `frag_name_dict` stay empty, so
    /// [`Library::fragment_payload_released`] is true and [`Library::cand_frags`] refuses to
    /// hand out intensities). `fragment_offset` is that of
    /// [`Library::load_with_fragment_offset`] for a band file, `None` for a whole library.
    ///
    /// The seed never reads a predicted intensity or a fragment name: its hyperscore is a
    /// match count plus observed intensity, and its mass recalibration walks `frag_mz`. So
    /// holding the two columns until the index was built -- where the fragindex path then
    /// released them -- cost 6 bytes per fragment at the seed's build peak, plus 6 more in
    /// the index ([`FragIndex::build_mz_only`]).
    ///
    /// Validation is the full load's, by construction: the two payload columns are still
    /// decoded batch by batch through the same pass and checked with the same checks
    /// (presence and type, NULLs, non-finite intensities, the u16 name limit), and each
    /// batch is then discarded instead of stored. A library that extract's full load would
    /// refuse is refused here, with the same message, before any DeepLC step runs on the
    /// seed's output.
    ///
    /// [`FragIndex::build_mz_only`]: crate::matchers::fragindex::FragIndex::build_mz_only
    pub fn load_mz_only(
        precursors: &str,
        fragments: &str,
        fragment_offset: Option<u32>,
    ) -> Result<Library> {
        Self::load_impl(
            precursors,
            fragments,
            1,
            false,
            None,
            fragment_offset.map(|o| o as usize),
            false,
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
            true,
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
        let (span, frag_offset) = Self::stage_rows(precursors, fragment_offset, precursor_span)?;
        Self::load_impl(
            precursors,
            fragments,
            bucket_size,
            build_bucketed,
            span,
            frag_offset,
            true,
        )
    }

    /// [`Library::load_mz_only`] for any load [`Library::load_for_stage`] accepts, so a
    /// seed can search a band by row span without its fragment payload.
    pub fn load_mz_only_for_stage(
        precursors: &str,
        fragments: &str,
        fragment_offset: Option<u32>,
        precursor_span: Option<(usize, usize)>,
    ) -> Result<Library> {
        let (span, frag_offset) = Self::stage_rows(precursors, fragment_offset, precursor_span)?;
        Self::load_impl(precursors, fragments, 1, false, span, frag_offset, false)
    }

    /// The `(span, frag_offset)` of [`Library::load_impl`] for a stage's load: the whole
    /// table, a band file, or a row span of the whole table, whose first row is also its
    /// fragment offset (see [`Library::load_row_span_with`]).
    fn stage_rows(
        precursors: &str,
        fragment_offset: Option<u32>,
        precursor_span: Option<(usize, usize)>,
    ) -> Result<(Option<(usize, usize)>, Option<usize>)> {
        match (precursor_span, fragment_offset) {
            (None, off) => Ok((None, off.map(|o| o as usize))),
            (Some((first, n)), off) => {
                if off.is_some_and(|o| o as usize != first) {
                    anyhow::bail!(
                        "a row span starting at {first} was asked for with fragment offset                          {off:?}; the span's first row is the offset"
                    );
                }
                if n == 0 {
                    anyhow::bail!("an empty row span of {precursors} is not a library");
                }
                Ok((Some((first, n)), Some(first)))
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
        payload: bool,
    ) -> Result<Library> {
        // The bucketed index is built from the predicted intensities.
        assert!(
            payload || !build_bucketed,
            "the bucketed page_search index needs the fragment payload"
        );
        // Phase timers for the load-path work (`library: loaded` below). The stage logs
        // bracket the library load together with the spectra decode and the index build, so
        // before these lines no log could say which of the three a seed or extract spent its
        // load phase on.
        let t_load = std::time::Instant::now();
        let partial = span.is_some() || frag_offset.is_some();
        let frag_offset = frag_offset.unwrap_or(0);
        let offset = span.map(|(first, _)| first).unwrap_or(0);
        let pc = load_precursors(precursors, span, offset)?;
        let ncand = pc.charge.len();
        let precursor_ms = t_load.elapsed().as_millis() as u64;

        let fc = load_fragments(
            fragments,
            frag_offset,
            ncand,
            partial,
            rayon::current_num_threads().min(LOAD_PARTS_MAX),
            payload,
        )?;
        let fragment_ms = t_load.elapsed().as_millis() as u64 - precursor_ms;
        let n_frag_rows = fc.frag_mz.len();

        // `prec_mz` IS the decoded `precursor_mz` column: row c is candidate c, in the same
        // order. It is shared with the fragment index through the `Arc` rather than copied
        // into it.
        let prec_mz: Arc<[f64]> = Arc::from(pc.precursor_mz);

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
        let n_decoy = pc.is_decoy.iter().filter(|&&d| d).count();
        let n_target = ncand - n_decoy;
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
            for c in 0..ncand {
                let (s, e) = (fc.frag_offsets[c] as usize, fc.frag_offsets[c + 1] as usize);
                for gi in s..e {
                    entries.push((fc.frag_mz[gi], c as u32, fc.frag_int[gi]));
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

        let lib = Library {
            peptidoform_id: pc.peptidoform_id,
            base_peptide_id: pc.base_peptide_id,
            charge: pc.charge,
            predicted_irt: pc.predicted_irt,
            is_decoy: pc.is_decoy,
            pform_offsets: pc.pform_offsets,
            pform_data: pc.pform_data,
            protein_id: pc.protein_id,
            protein_dict: pc.protein_dict,
            frag_offsets: fc.frag_offsets,
            frag_mz: fc.frag_mz,
            frag_int: fc.frag_int,
            frag_name_id: fc.frag_name_id,
            frag_name_dict: fc.frag_name_dict,
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
        };
        crate::memlog::report(
            "library steady state",
            &[
                ("frag_mz", crate::memlog::bytes_of(&lib.frag_mz)),
                ("frag_int", crate::memlog::bytes_of(&lib.frag_int)),
                ("frag_name_id", crate::memlog::bytes_of(&lib.frag_name_id)),
                ("idx_mz", crate::memlog::bytes_of(&lib.idx_mz)),
                ("idx_cid", crate::memlog::bytes_of(&lib.idx_cid)),
                ("idx_int", crate::memlog::bytes_of(&lib.idx_int)),
                ("precursor_columns", lib.precursor_bytes()),
                ("frag_offsets", crate::memlog::bytes_of(&lib.frag_offsets)),
            ],
        );
        tracing::info!(
            library = precursors,
            candidates = ncand,
            fragments = n_frag_rows,
            precursor_ms,
            fragment_ms,
            elapsed_ms = t_load.elapsed().as_millis() as u64,
            "library: loaded"
        );
        Ok(lib)
    }

    /// Build a library in memory from owned candidate records and their fragment arrays,
    /// which is how tests and benchmarks make one. `cands[c].candidate_id` must be `c` and
    /// the fragment ranges must tile `frag_mz` in candidate order; both are asserted, since
    /// a loaded library has them by construction and the matchers rely on it.
    pub fn from_candidates(
        cands: Vec<Candidate>,
        frag_mz: Vec<f32>,
        frag_int: Vec<f32>,
        frag_name_id: Vec<u16>,
        frag_name_dict: Vec<String>,
    ) -> Library {
        let n = cands.len();
        let mut pform_offsets = Vec::with_capacity(n + 1);
        pform_offsets.push(0usize);
        let mut pform_data = String::new();
        let mut proteins = mumdia_io::table::StrInterner::new();
        let mut protein_id = Vec::with_capacity(n);
        let mut frag_offsets = Vec::with_capacity(n + 1);
        frag_offsets.push(0u32);
        let mut prec_mz = Vec::with_capacity(n);
        let (mut pfid, mut base, mut charge, mut irt, mut dec) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for (c, cd) in cands.into_iter().enumerate() {
            assert_eq!(
                cd.candidate_id as usize, c,
                "from_candidates: candidate_id must be the position (id {} at {c})",
                cd.candidate_id
            );
            assert_eq!(
                cd.frag_start,
                *frag_offsets.last().unwrap() as usize,
                "from_candidates: fragments must tile the arrays in candidate order"
            );
            pfid.push(cd.peptidoform_id);
            base.push(cd.base_peptide_id);
            charge.push(cd.charge);
            irt.push(cd.predicted_irt);
            dec.push(cd.is_decoy);
            prec_mz.push(cd.precursor_mz);
            pform_data.push_str(&cd.peptidoform);
            pform_offsets.push(pform_data.len());
            protein_id.push(proteins.intern(&cd.protein));
            frag_offsets.push(u32::try_from(cd.frag_start + cd.n_frag).expect("u32 offsets"));
        }
        assert_eq!(
            *frag_offsets.last().unwrap() as usize,
            frag_mz.len(),
            "from_candidates: fragments must tile the arrays in candidate order"
        );
        Library {
            peptidoform_id: pfid,
            base_peptide_id: base,
            charge,
            predicted_irt: irt,
            is_decoy: dec,
            pform_offsets,
            pform_data,
            protein_id,
            protein_dict: proteins.into_values(),
            frag_offsets,
            frag_mz,
            frag_int,
            frag_name_id,
            frag_name_dict,
            idx_mz: Vec::new(),
            idx_cid: Vec::new(),
            idx_int: Vec::new(),
            bucket_min: Vec::new(),
            bucket_size: 1,
            prec_mz: Arc::from(prec_mz),
            global_offset: 0,
        }
    }

    pub fn n_candidates(&self) -> usize {
        self.prec_mz.len()
    }

    /// Candidate `cid`'s peptidoform, as the precursor table spells it.
    #[inline]
    pub fn peptidoform(&self, cid: u32) -> &str {
        let c = cid as usize;
        &self.pform_data[self.pform_offsets[c]..self.pform_offsets[c + 1]]
    }

    /// Candidate `cid`'s protein (group); an empty value in the file reads as `UNASSIGNED`.
    #[inline]
    pub fn protein(&self, cid: u32) -> &str {
        &self.protein_dict[self.protein_id[cid as usize] as usize]
    }

    /// Candidate `cid`'s fragment rows, as a range of the fragment arrays.
    #[inline]
    pub fn frag_range(&self, cid: u32) -> std::ops::Range<usize> {
        let c = cid as usize;
        self.frag_offsets[c] as usize..self.frag_offsets[c + 1] as usize
    }

    /// Candidate `cid`'s precursor fields, borrowed.
    #[inline]
    pub fn cand(&self, cid: u32) -> CandInfo<'_> {
        let c = cid as usize;
        CandInfo {
            peptidoform_id: self.peptidoform_id[c],
            base_peptide_id: self.base_peptide_id[c],
            peptidoform: self.peptidoform(cid),
            charge: self.charge[c],
            precursor_mz: self.prec_mz[c],
            predicted_irt: self.predicted_irt[c],
            is_decoy: self.is_decoy[c],
            protein: self.protein(cid),
        }
    }

    /// Bytes held by the per-candidate columns (for the memory log).
    fn precursor_bytes(&self) -> usize {
        crate::memlog::bytes_of(&self.peptidoform_id)
            + crate::memlog::bytes_of(&self.base_peptide_id)
            + crate::memlog::bytes_of(&self.charge)
            + crate::memlog::bytes_of(&self.predicted_irt)
            + crate::memlog::bytes_of(&self.is_decoy)
            + crate::memlog::bytes_of(&self.pform_offsets)
            + self.pform_data.capacity()
            + crate::memlog::bytes_of(&self.protein_id)
            + self
                .protein_dict
                .iter()
                .map(|s| s.capacity() + 24)
                .sum::<usize>()
            + std::mem::size_of_val(&*self.prec_mz)
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
        let r = self.frag_range(cid);
        (
            &self.frag_mz[r.clone()],
            &self.frag_int[r.clone()],
            &self.frag_name_id[r],
        )
    }

    /// Per-candidate fragment m/z alone. The only fragment column that survives
    /// [`Library::release_fragment_payload`], so this is what a stage that has already built
    /// its index must use.
    pub fn cand_frag_mz(&self, cid: u32) -> &[f32] {
        &self.frag_mz[self.frag_range(cid)]
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
        let r = lib.frag_range(c as u32);
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
            let (a, b) = (part.cand(c as u32), full.cand(c as u32 + 1));
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
            assert_eq!(lib.peptidoform(c as u32), full.peptidoform(c as u32 + 1));
            assert_eq!(frag_slice(&lib, c), frag_slice(&full, c + 1));
        }
        // Same content as loading the band by m/z from the whole file.
        let by_mz = Library::load_range_with(&p, &f, 440.0, 530.0, 8, false).unwrap();
        for c in 0..3 {
            assert_eq!(frag_slice(&lib, c), frag_slice(&by_mz, c));
            assert_eq!(lib.prec_mz[c], by_mz.prec_mz[c]);
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
        assert_eq!(by_span.frag_offsets, lib.frag_offsets);
        for c in 0..3u32 {
            let (a, b) = (by_span.cand(c), lib.cand(c));
            assert_eq!(
                (a.peptidoform_id, a.base_peptide_id, a.charge),
                (b.peptidoform_id, b.base_peptide_id, b.charge)
            );
            assert_eq!(by_span.frag_range(c), lib.frag_range(c));
            assert_eq!((a.peptidoform, a.protein), (b.peptidoform, b.protein));
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
        // The seed's payload-free load of the same span is the same library without the
        // fragment payload.
        let mz_only = Library::load_mz_only_for_stage(&p, &f, None, Some((1, 3))).unwrap();
        assert!(mz_only.fragment_payload_released());
        assert_eq!(mz_only.global_offset, lib.global_offset);
        assert_eq!(mz_only.frag_mz, lib.frag_mz);
        assert_eq!(mz_only.frag_offsets, lib.frag_offsets);
        assert_eq!(mz_only.prec_mz, lib.prec_mz);
        assert!(Library::load_mz_only_for_stage(&p, &f, Some(2), Some((1, 3))).is_err());
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
            assert_eq!(lib.peptidoform(c as u32), full.peptidoform(c as u32));
            assert_eq!(frag_slice(&lib, c), frag_slice(&full, c));
        }
    }

    /// Write a fragment table whose `name` column is encoded as the caller asks: parquet's
    /// dictionary encoding (the writers' default), plain (dictionary off), or a dictionary
    /// that overflows its page-size limit and falls back to plain part way through the
    /// column chunk. The reader must produce the same library from all three.
    fn write_named_fragments(
        path: &str,
        cid: &[u32],
        names: &[&str],
        encoding: &str,
        row_group_rows: usize,
    ) {
        use arrow::array::{Float32Array, Float64Array, StringArray, UInt32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        let n = cid.len();
        let schema = std::sync::Arc::new(Schema::new(vec![
            Field::new("candidate_id", DataType::UInt32, false),
            Field::new("mz", DataType::Float64, false),
            Field::new("predicted_intensity", DataType::Float32, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![
                std::sync::Arc::new(UInt32Array::from(cid.to_vec())),
                std::sync::Arc::new(Float64Array::from(
                    (0..n).map(|i| 150.0 + i as f64).collect::<Vec<_>>(),
                )),
                std::sync::Arc::new(Float32Array::from(
                    (0..n).map(|i| 1.0 / (1 + i) as f32).collect::<Vec<_>>(),
                )),
                std::sync::Arc::new(StringArray::from(names.to_vec())),
            ],
        )
        .unwrap();
        let mut props =
            WriterProperties::builder().set_max_row_group_row_count(Some(row_group_rows));
        props = match encoding {
            "dictionary" => props,
            "plain" => props.set_dictionary_enabled(false),
            // A 16-byte dictionary page limit overflows after a handful of distinct names,
            // so the chunk switches to plain pages part way through.
            "fallback" => props
                .set_dictionary_page_size_limit(16)
                .set_data_page_row_count_limit(2)
                .set_write_batch_size(2),
            other => panic!("unknown encoding {other}"),
        };
        let file = std::fs::File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(file, schema, Some(props.build())).unwrap();
        w.write(&batch).unwrap();
        let meta = w.close().unwrap();
        // The fixture must be what it says: otherwise the three arms test one encoding.
        use parquet::basic::Encoding;
        let enc_of = |rg: usize| {
            let col = meta.row_group(rg).column(3);
            (
                col.dictionary_page_offset().is_some(),
                col.encodings().any(|e| e == Encoding::PLAIN),
            )
        };
        let (has_dict, has_plain) = enc_of(0);
        match encoding {
            "dictionary" => assert!(has_dict, "dictionary fixture has no dictionary page"),
            "plain" => assert!(!has_dict, "plain fixture has a dictionary page"),
            _ => {
                let any_fallback = (0..meta.num_row_groups()).any(|r| {
                    let (d, pl) = enc_of(r);
                    d && pl
                });
                assert!(
                    any_fallback || (has_dict && has_plain),
                    "fallback fixture never fell back to plain pages"
                );
            }
        }
    }

    /// The fragment names are interned per dictionary key now rather than per row, and the
    /// dictionary must be exactly what the per-row map built: ids in FIRST-APPEARANCE order
    /// over the rows the load keeps, never over the rows a partial load skips. The
    /// reference below is that per-row map, run over the table.
    #[test]
    fn fragment_names_are_interned_in_first_appearance_order_over_in_range_rows() {
        let dir = unique_dir("name_intern");
        std::fs::create_dir_all(&dir).unwrap();
        let (p, _) = build_six_lib(&dir, "names", &[0, 1, 2, 3, 4, 5]);
        // Three fragments per candidate, sorted by candidate. Candidates 0, 4 and 5 carry
        // names nobody in the band 1..=3 uses, and the band's own first appearances are in
        // an order unlike the full table's.
        let cid: Vec<u32> = (0..6u32).flat_map(|c| [c, c, c]).collect();
        let names: Vec<&str> = vec![
            "a0", "b2", "y1", // 0
            "y7", "b2", "y7", // 1
            "b3", "y1", "y7", // 2
            "y9^2", "b3", "b2", // 3
            "zz4", "a0", "y1", // 4
            "zz5", "zz5", "y9^2", // 5
        ];
        let reference = |lo: u32, hi: u32| -> (Vec<String>, Vec<u16>) {
            let mut dict: Vec<String> = Vec::new();
            let mut map: std::collections::HashMap<&str, u16> = Default::default();
            let mut ids = Vec::new();
            for (c, n) in cid.iter().zip(&names) {
                if *c < lo || *c >= hi {
                    continue;
                }
                let id = *map.entry(n).or_insert_with(|| {
                    dict.push(n.to_string());
                    (dict.len() - 1) as u16
                });
                ids.push(id);
            }
            (dict, ids)
        };
        // (full dictionary, full ids, range dictionary, range ids) of the first arm.
        type NameArrays = (Vec<String>, Vec<u16>, Vec<String>, Vec<u16>);
        let mut seen: Option<NameArrays> = None;
        for enc in ["dictionary", "plain", "fallback"] {
            for rg in [2usize, 1024] {
                let f = dir
                    .join(format!("frag_{enc}_{rg}.parquet"))
                    .to_str()
                    .unwrap()
                    .to_string();
                write_named_fragments(&f, &cid, &names, enc, rg);
                let full = Library::load_with(&p, &f, 8, false).unwrap();
                let (dict, ids) = reference(0, 6);
                assert_eq!(
                    full.frag_name_dict, dict,
                    "{enc}/{rg}: full-load dictionary"
                );
                assert_eq!(full.frag_name_id, ids, "{enc}/{rg}: full-load ids");
                // [440, 530] is candidates 1..=3.
                let part = Library::load_range_with(&p, &f, 440.0, 530.0, 8, false).unwrap();
                let (pdict, pids) = reference(1, 4);
                assert_eq!(part.frag_name_dict, pdict, "{enc}/{rg}: range dictionary");
                assert_eq!(part.frag_name_id, pids, "{enc}/{rg}: range ids");
                assert!(
                    !part
                        .frag_name_dict
                        .iter()
                        .any(|n| n.starts_with("zz") || n == "a0"),
                    "a skipped row's name reached the band's dictionary"
                );
                let got = (
                    full.frag_name_dict.clone(),
                    full.frag_name_id.clone(),
                    part.frag_name_dict.clone(),
                    part.frag_name_id.clone(),
                );
                match &seen {
                    None => seen = Some(got),
                    Some(s) => assert_eq!(s, &got, "{enc}/{rg} differs from the first encoding"),
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
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

    /// A fragment table for `n_cand` candidates with 0..=6 fragments each, names drawn from
    /// a small vocabulary in a scrambled order, written in `row_group_rows`-row groups.
    /// `order` is the candidate order of the rows: ascending (the writers' layout),
    /// shuffled (an unsorted table), or blocks of ascending ids out of order.
    fn write_parts_fragments(path: &str, n_cand: u32, order: &str, row_group_rows: usize) {
        let mut rows: Vec<(u32, f64, f32, String)> = Vec::new();
        let mut state = 0x9e37_79b9_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let vocab = ["b2", "y1", "y7", "b3", "y9^2", "b10", "y3", "a2"];
        for c in 0..n_cand {
            for k in 0..(next() % 7) {
                let name = vocab[(next() as usize + c as usize * 3) % vocab.len()];
                rows.push((
                    c,
                    150.0 + (next() % 180_000) as f64 * 0.01 + k as f64 * 1e-4,
                    (next() % 1000) as f32 / 1000.0,
                    name.to_string(),
                ));
            }
        }
        match order {
            "ascending" => {}
            "shuffled" => {
                for i in (1..rows.len()).rev() {
                    let j = next() as usize % (i + 1);
                    rows.swap(i, j);
                }
            }
            "blocks" => {
                // Ascending inside each block of ~50 rows, blocks reversed: every part
                // is ascending on its own, the table is not.
                let blocks: Vec<Vec<_>> = rows.chunks(50).map(|b| b.to_vec()).collect();
                rows = blocks.into_iter().rev().flatten().collect();
            }
            other => panic!("unknown order {other}"),
        }
        let mut w = TableWriter::new(path).with_row_group_rows(row_group_rows);
        w.write_cols(vec![
            Col::U32("candidate_id".into(), rows.iter().map(|r| r.0).collect()),
            Col::F64("mz".into(), rows.iter().map(|r| r.1).collect()),
            Col::F32(
                "predicted_intensity".into(),
                rows.iter().map(|r| r.2).collect(),
            ),
            Col::Str("name".into(), rows.iter().map(|r| r.3.clone()).collect()),
        ])
        .unwrap();
        w.close().unwrap();
    }

    /// The parallel fragment passes (row-group and page-range parts, atomic counts, the
    /// identity placement of an ascending table, per-part name dictionaries merged in part
    /// order) must produce the arrays of the serial pass bit for bit, on every layout and
    /// every kind of load: whole, a range of the table, and a band against the shared
    /// table.
    #[test]
    fn parallel_fragment_passes_reproduce_the_serial_arrays() {
        let dir = unique_dir("par_frag");
        std::fs::create_dir_all(&dir).unwrap();
        let n_cand = 400u32;
        for order in ["ascending", "shuffled", "blocks"] {
            for rg in [37usize, 1 << 20] {
                let f = dir
                    .join(format!("frag_{order}_{rg}.parquet"))
                    .to_str()
                    .unwrap()
                    .to_string();
                write_parts_fragments(&f, n_cand, order, rg);
                // (frag_offset, ncand, partial): whole, a middle band, a band at row 0, the
                // tail band.
                for (off, n, partial) in [
                    (0usize, n_cand as usize, false),
                    (101, 157, true),
                    (0, 64, true),
                    (300, 100, true),
                ] {
                    let serial = load_fragments(&f, off, n, partial, 1, true).unwrap();
                    for parts in [2usize, 5, 64] {
                        let par = load_fragments(&f, off, n, partial, parts, true).unwrap();
                        let what = format!("{order} rg={rg} band=({off},{n}) parts={parts}");
                        assert_eq!(par.frag_offsets, serial.frag_offsets, "{what}: offsets");
                        assert_eq!(
                            par.frag_mz.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            serial
                                .frag_mz
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            "{what}: frag_mz"
                        );
                        assert_eq!(
                            par.frag_int.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            serial
                                .frag_int
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            "{what}: frag_int"
                        );
                        assert_eq!(par.frag_name_id, serial.frag_name_id, "{what}: name ids");
                        assert_eq!(par.frag_name_dict, serial.frag_name_dict, "{what}: dict");
                    }
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two defects of DIFFERENT kinds inside one 65,536-row serial batch, with a part
    /// boundary between them: the serial pass sees both in one batch and the parted pass
    /// in two parts. Both must refuse the table on the first bad row (row 10), not on the
    /// kind of defect a batch happens to check first.
    #[test]
    fn a_batch_is_refused_on_its_first_bad_row_whatever_the_kind_of_defect() {
        use arrow::array::{Float32Array, Float64Array, StringArray, UInt32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        let dir = unique_dir("first_bad_row");
        std::fs::create_dir_all(&dir).unwrap();
        let n = 62_000usize;
        let (early, late) = (10usize, 60_000usize);
        assert!(late < FRAG_BATCH_ROWS, "both defects in one serial batch");
        // 30,000-row groups: the parted pass cuts between the two defects.
        let write = |tag: &str, fault: &str| -> String {
            let f = dir
                .join(format!("{tag}.parquet"))
                .to_str()
                .unwrap()
                .to_string();
            let mut cid: Vec<Option<u32>> = (0..n).map(|i| Some((i / 10) as u32)).collect();
            let mut mz: Vec<Option<f64>> = (0..n).map(|i| Some(200.0 + i as f64 * 1e-3)).collect();
            let mut int: Vec<Option<f32>> = vec![Some(1.0); n];
            let mut name: Vec<Option<&str>> = vec![Some("y1"); n];
            match fault {
                // Pass 1: an out-of-range id early, a NULL id late.
                "oor_then_null_cid" => {
                    cid[early] = Some(999_999);
                    cid[late] = None;
                }
                // Pass 1, the other way round.
                "null_cid_then_oor" => {
                    cid[early] = None;
                    cid[late] = Some(999_999);
                }
                // Pass 2: a non-finite intensity early, a NULL m/z late.
                "nan_int_then_null_mz" => {
                    int[early] = Some(f32::NAN);
                    mz[late] = None;
                }
                // Pass 2: a non-finite m/z early, a NULL name late.
                "inf_mz_then_null_name" => {
                    mz[early] = Some(f64::INFINITY);
                    name[late] = None;
                }
                // Pass 2: a NULL intensity early, a non-finite m/z late.
                "null_int_then_nan_mz" => {
                    int[early] = None;
                    mz[late] = Some(f64::NAN);
                }
                other => panic!("unknown fault {other}"),
            }
            let schema = std::sync::Arc::new(Schema::new(vec![
                Field::new("candidate_id", DataType::UInt32, true),
                Field::new("mz", DataType::Float64, true),
                Field::new("predicted_intensity", DataType::Float32, true),
                Field::new("name", DataType::Utf8, true),
            ]));
            let batch = arrow::record_batch::RecordBatch::try_new(
                schema.clone(),
                vec![
                    std::sync::Arc::new(UInt32Array::from(cid)),
                    std::sync::Arc::new(Float64Array::from(mz)),
                    std::sync::Arc::new(Float32Array::from(int)),
                    std::sync::Arc::new(StringArray::from(name)),
                ],
            )
            .unwrap();
            let mut w =
                mumdia_io::table::BatchWriter::with_row_group_rows(&f, schema, 30_000).unwrap();
            w.write(&batch).unwrap();
            w.close().unwrap();
            f
        };
        let ncand = n / 10;
        for (fault, expect) in [
            (
                "oor_then_null_cid",
                "fragment row 10 references candidate_id 999999",
            ),
            ("null_cid_then_oor", "'candidate_id' has a NULL at row 10"),
            (
                "nan_int_then_null_mz",
                "'predicted_intensity' has a non-finite value",
            ),
            ("inf_mz_then_null_name", "'mz' has a non-finite value"),
            (
                "null_int_then_nan_mz",
                "'predicted_intensity' has a NULL at row 10",
            ),
        ] {
            let f = write(fault, fault);
            for payload in [true, false] {
                let msg = |parts: usize| match load_fragments(&f, 0, ncand, false, parts, payload) {
                    Ok(_) => panic!("{fault}: the bad rows must be refused"),
                    Err(e) => format!("{e:#}"),
                };
                let serial = msg(1);
                assert!(
                    serial.contains(expect),
                    "{fault} payload={payload}: the serial pass must name the first bad row, \
                     got {serial}"
                );
                for parts in [2usize, 3] {
                    assert_eq!(
                        msg(parts),
                        serial,
                        "{fault} payload={payload} parts={parts}"
                    );
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A broken row in a LATER part must be reported exactly as the serial pass reports
    /// it: the first bad row in file order, whichever part finished first.
    #[test]
    fn a_parallel_load_reports_the_first_bad_row_as_the_serial_load_does() {
        let dir = unique_dir("par_err");
        std::fs::create_dir_all(&dir).unwrap();
        let n = 500usize;
        let mk = |tag: &str, cid: Vec<u32>, mz: Vec<f64>| {
            let f = dir
                .join(format!("{tag}.parquet"))
                .to_str()
                .unwrap()
                .to_string();
            let mut w = TableWriter::new(&f).with_row_group_rows(40);
            w.write_cols(vec![
                Col::U32("candidate_id".into(), cid),
                Col::F64("mz".into(), mz),
                Col::F32("predicted_intensity".into(), vec![1.0; n]),
                Col::Str("name".into(), vec!["y1".to_string(); n]),
            ])
            .unwrap();
            w.close().unwrap();
            f
        };
        let ids: Vec<u32> = (0..n as u32).map(|i| i / 5).collect();
        let mzs: Vec<f64> = (0..n).map(|i| 200.0 + i as f64).collect();
        // Two out-of-range ids, in the fourth and the tenth row group.
        let mut bad_ids = ids.clone();
        bad_ids[150] = 9_999;
        bad_ids[390] = 9_998;
        let f1 = mk("bad_id", bad_ids, mzs.clone());
        // Two non-finite m/z, in the third and the ninth row group.
        let mut bad_mz = mzs.clone();
        bad_mz[90] = f64::INFINITY;
        bad_mz[350] = f64::NAN;
        let f2 = mk("bad_mz", ids.clone(), bad_mz);
        for f in [&f1, &f2] {
            let msg = |parts: usize| match load_fragments(f, 0, 100, false, parts, true) {
                Ok(_) => panic!("{f}: the bad row must be refused"),
                Err(e) => format!("{e:#}"),
            };
            let serial = msg(1);
            for parts in [2, 5, 13] {
                assert_eq!(msg(parts), serial, "{f} with {parts} parts");
            }
        }
        assert!(format!(
            "{:#}",
            load_fragments(&f1, 0, 100, false, 5, true).err().unwrap()
        )
        .contains("fragment row 150 "));
        std::fs::remove_dir_all(&dir).ok();
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
        assert_eq!(lib.is_decoy.clone(), vec![false, true]);
        // Reversed, so a constant-false or index-shifted read cannot pass both cases.
        let (p2, f2) = library_ids_labels(&unique_dir("label_bool2"), [0, 1], ["decoy", "target"]);
        let lib2 = Library::load_with(&p2, &f2, 8, false).unwrap();
        assert_eq!(lib2.is_decoy.clone(), vec![true, false]);
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
        // Local ids are positions, so what pins the span is that local 0..3 hold file rows
        // 1..4: their peptidoform ids are 11, 12, 13, and the offset records row 1.
        assert_eq!(part.peptidoform_id, vec![11, 12, 13]);
        assert_eq!(part.global_offset, 1);
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
            assert_eq!(lib.prec_mz[c], lib.cand(c as u32).precursor_mz);
        }
        assert_eq!(
            lib.prec_mz.to_vec(),
            vec![400.0, 450.0, 500.0, 520.0, 600.0, 650.0]
        );
        // The same holds for a band, whose `prec_mz` is the band's slice of the column.
        let part = Library::load_range_with(&p, &f, 440.0, 530.0, 8, false).unwrap();
        assert_eq!(part.prec_mz.to_vec(), vec![450.0, 500.0, 520.0]);
        for c in 0..part.n_candidates() {
            assert_eq!(part.prec_mz[c], part.cand(c as u32).precursor_mz);
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
        assert_eq!(lib.protein(0), "P1");
        assert_eq!(lib.protein(1), "UNASSIGNED");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The library holds its precursor fields as columns (peptidoform arena, interned
    /// protein, CSR fragment offsets) instead of one struct per candidate. Every field of
    /// every candidate must still be the table's value for that row, including a protein
    /// that repeats (interned once), a blank protein (renamed on the dictionary) and a
    /// literal `UNASSIGNED` next to it.
    #[test]
    fn the_columnar_library_holds_every_precursor_field_of_the_table() {
        let dir = unique_dir("columnar");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("prec.parquet").to_str().unwrap().to_string();
        let f = dir.join("frag.parquet").to_str().unwrap().to_string();
        let prots = ["P9", "P1", "P9", " ", "UNASSIGNED", "P1"];
        let pforms = ["AAK", "C[Carbamidomethyl]DK", "EEK", "FFK", "GGK", "HHHHK"];
        write_table(
            &p,
            vec![
                Col::U32("candidate_id".into(), (0..6).collect()),
                Col::U32("peptidoform_id".into(), vec![7, 7, 8, 8, 9, 9]),
                Col::U32("base_peptide_id".into(), vec![3, 3, 4, 4, 5, 5]),
                Col::Str(
                    "peptidoform".into(),
                    pforms.iter().map(|s| s.to_string()).collect(),
                ),
                Col::I32("charge".into(), vec![2, 3, 2, 4, 1, 2]),
                Col::F64(
                    "precursor_mz".into(),
                    vec![400.0, 410.5, 420.25, 430.0, 440.0, 450.0],
                ),
                Col::F32(
                    "predicted_irt".into(),
                    vec![1.5, -2.0, 3.25, 0.0, 9.0, 12.0],
                ),
                Col::Str(
                    "label".into(),
                    (0..6)
                        .map(|i| if i % 2 == 0 { "target" } else { "decoy" }.to_string())
                        .collect(),
                ),
                Col::Str(
                    "protein".into(),
                    prots.iter().map(|s| s.to_string()).collect(),
                ),
            ],
        )
        .unwrap();
        // Uneven fragment counts, so the CSR offsets are not a constant stride.
        let counts = [1usize, 3, 0, 2, 4, 1];
        let cid: Vec<u32> = (0..6u32)
            .flat_map(|c| std::iter::repeat_n(c, counts[c as usize]))
            .collect();
        let n = cid.len();
        write_table(
            &f,
            vec![
                Col::U32("candidate_id".into(), cid),
                Col::F64("mz".into(), (0..n).map(|i| 100.0 + i as f64).collect()),
                Col::F32("predicted_intensity".into(), vec![1.0; n]),
                Col::Str("name".into(), vec!["y1".to_string(); n]),
            ],
        )
        .unwrap();
        let lib = Library::load_with(&p, &f, 8, false).unwrap();
        assert_eq!(lib.n_candidates(), 6);
        let mut start = 0usize;
        for c in 0..6u32 {
            let i = c as usize;
            let v = lib.cand(c);
            assert_eq!(v.peptidoform, pforms[i]);
            assert_eq!(lib.peptidoform(c), pforms[i]);
            let want_prot = if prots[i].trim().is_empty() {
                "UNASSIGNED"
            } else {
                prots[i]
            };
            assert_eq!(v.protein, want_prot);
            assert_eq!(v.peptidoform_id, [7, 7, 8, 8, 9, 9][i]);
            assert_eq!(v.base_peptide_id, [3, 3, 4, 4, 5, 5][i]);
            assert_eq!(v.charge, [2, 3, 2, 4, 1, 2][i]);
            assert_eq!(
                v.precursor_mz,
                [400.0, 410.5, 420.25, 430.0, 440.0, 450.0][i]
            );
            assert_eq!(
                v.predicted_irt.to_bits(),
                [1.5f32, -2.0, 3.25, 0.0, 9.0, 12.0][i].to_bits()
            );
            assert_eq!(v.is_decoy, i % 2 == 1);
            assert_eq!(lib.frag_range(c), start..start + counts[i]);
            let want_mz: Vec<f32> = (start..start + counts[i])
                .map(|k| (100.0 + k as f64) as f32)
                .collect();
            assert_eq!(lib.cand_frag_mz(c), &want_mz[..]);
            start += counts[i];
        }
        assert_eq!(lib.frag_offsets, vec![0, 1, 4, 4, 6, 10, 11]);
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

    /// The seed's m/z-only library holds exactly the full load's precursor columns, CSR
    /// offsets and fragment m/z, and no payload, for whole, range and band loads over
    /// ascending and shuffled tables in one and in many parts.
    #[test]
    fn the_mz_only_library_is_the_full_library_without_its_payload() {
        let dir = unique_dir("mz_only");
        std::fs::create_dir_all(&dir).unwrap();
        let (p, _) = build_six_lib(&dir, "mzonly", &[0, 1, 2, 3, 4, 5]);
        for order in ["ascending", "shuffled"] {
            for rg in [3usize, 1 << 20] {
                // The six-candidate precursor table, so the fragment ids stay below 6.
                let f = dir
                    .join(format!("frag_mzonly_{order}_{rg}.parquet"))
                    .to_str()
                    .unwrap()
                    .to_string();
                write_parts_fragments(&f, 6, order, rg);
                let pairs: Vec<(Library, Library)> = vec![
                    (
                        Library::load_with(&p, &f, 8, false).unwrap(),
                        Library::load_mz_only(&p, &f, None).unwrap(),
                    ),
                    (
                        {
                            let band = dir.join("mzonly_band.parquet");
                            let band = band.to_str().unwrap().to_string();
                            crate::groups::write_band_slice(&p, 2, 3, &band).unwrap();
                            Library::load_with_fragment_offset(&band, &f, 2, 8, false).unwrap()
                        },
                        {
                            let band = dir.join("mzonly_band.parquet");
                            let band = band.to_str().unwrap().to_string();
                            Library::load_mz_only(&band, &f, Some(2)).unwrap()
                        },
                    ),
                ];
                for (full, mz) in &pairs {
                    let what = format!("{order} rg={rg}");
                    assert_eq!(mz.n_candidates(), full.n_candidates(), "{what}");
                    assert_eq!(mz.frag_offsets, full.frag_offsets, "{what}: offsets");
                    assert_eq!(
                        mz.frag_mz.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        full.frag_mz.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{what}: frag_mz"
                    );
                    assert_eq!(mz.prec_mz, full.prec_mz, "{what}: prec_mz");
                    assert_eq!(mz.global_offset, full.global_offset, "{what}: offset");
                    for c in 0..full.n_candidates() as u32 {
                        let (a, b) = (mz.cand(c), full.cand(c));
                        assert_eq!(
                            (a.peptidoform, a.protein, a.charge, a.is_decoy),
                            (b.peptidoform, b.protein, b.charge, b.is_decoy),
                            "{what}: candidate {c}"
                        );
                        assert_eq!(a.predicted_irt.to_bits(), b.predicted_irt.to_bits());
                        assert_eq!(
                            (a.peptidoform_id, a.base_peptide_id),
                            (b.peptidoform_id, b.base_peptide_id)
                        );
                    }
                    assert!(mz.frag_int.is_empty() && mz.frag_name_id.is_empty());
                    assert!(mz.frag_name_dict.is_empty());
                    if !mz.frag_mz.is_empty() {
                        assert!(mz.fragment_payload_released(), "{what}");
                    }
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Validation parity of the m/z-only load: it reads and checks the payload columns as
    /// the full load does and only discards them, so every library the full load (extract's)
    /// refuses, the seed refuses with the same message. That includes the per-VALUE faults
    /// of the payload (a NULL or NaN intensity, a NULL name, more distinct names than the
    /// u16 id holds), which the seed used to leave to extract, after the DeepLC step that
    /// runs between the two.
    #[test]
    fn the_mz_only_load_refuses_every_library_the_full_load_refuses() {
        use arrow::array::{Float32Array, Float64Array, Int32Array, StringArray, UInt32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        let dir = unique_dir("mz_only_validation");
        std::fs::create_dir_all(&dir).unwrap();
        let (p, _) = library_with(&dir, ["PEPTIDEK", "SAMPLER"], ["P1", "P2"]);
        // One fragment table per fault, written column by column so each can be missing,
        // mistyped or poisoned.
        let write = |tag: &str, fault: &str| -> String {
            let f = dir
                .join(format!("{tag}.parquet"))
                .to_str()
                .unwrap()
                .to_string();
            let mut fields = Vec::new();
            let mut cols: Vec<ArrayRef> = Vec::new();
            fields.push(Field::new("candidate_id", DataType::UInt32, true));
            cols.push(std::sync::Arc::new(UInt32Array::from(
                if fault == "null_cid" {
                    vec![Some(0u32), None]
                } else {
                    vec![Some(0u32), Some(1)]
                },
            )));
            fields.push(Field::new("mz", DataType::Float64, true));
            cols.push(std::sync::Arc::new(Float64Array::from(
                if fault == "nan_mz" {
                    vec![200.1, f64::NAN]
                } else {
                    vec![200.1, 250.5]
                },
            )));
            match fault {
                "no_int" => {}
                "int_f64" => {
                    fields.push(Field::new("predicted_intensity", DataType::Float64, true));
                    cols.push(std::sync::Arc::new(Float64Array::from(vec![1.0, 0.9])));
                }
                _ => {
                    fields.push(Field::new("predicted_intensity", DataType::Float32, true));
                    cols.push(std::sync::Arc::new(Float32Array::from(match fault {
                        "null_int" => vec![Some(1.0f32), None],
                        "nan_int" => vec![Some(1.0f32), Some(f32::NAN)],
                        _ => vec![Some(1.0f32), Some(0.9)],
                    })));
                }
            }
            match fault {
                "no_name" => {}
                "name_i32" => {
                    fields.push(Field::new("name", DataType::Int32, true));
                    cols.push(std::sync::Arc::new(Int32Array::from(vec![2, 3])));
                }
                _ => {
                    fields.push(Field::new("name", DataType::Utf8, true));
                    cols.push(std::sync::Arc::new(StringArray::from(
                        if fault == "null_name" {
                            vec![Some("b2"), None]
                        } else {
                            vec![Some("b2"), Some("y3")]
                        },
                    )));
                }
            }
            let schema = std::sync::Arc::new(Schema::new(fields));
            let batch = arrow::record_batch::RecordBatch::try_new(schema.clone(), cols).unwrap();
            mumdia_io::table::write_batches(&f, schema, &[batch]).unwrap();
            f
        };
        let full = |f: &str| {
            Library::load_with(&p, f, 8, false)
                .err()
                .map(|e| format!("{e:#}"))
        };
        let mz = |f: &str| {
            Library::load_mz_only(&p, f, None)
                .err()
                .map(|e| format!("{e:#}"))
        };
        // The table without a fault loads both ways.
        let ok = write("ok", "none");
        assert!(full(&ok).is_none() && mz(&ok).is_none());
        // Schema faults, faults in the columns the seed stores, and faults in the payload
        // it discards: the same refusal for every one.
        for fault in [
            "no_int",
            "no_name",
            "int_f64",
            "name_i32",
            "null_cid",
            "nan_mz",
            "null_int",
            "nan_int",
            "null_name",
        ] {
            let f = write(fault, fault);
            let (a, b) = (full(&f), mz(&f));
            assert!(a.is_some(), "{fault}: the full load must refuse it");
            assert_eq!(
                a, b,
                "{fault}: the m/z-only load must refuse it the same way"
            );
        }
        // The name limit, on the serial and the parted pass: one name more than the u16 id
        // holds, over the two candidates, in row groups small enough to be cut into parts.
        let many = dir.join("many_names.parquet").to_str().unwrap().to_string();
        let n = u16::MAX as usize + 2;
        let mut w = TableWriter::new(&many).with_row_group_rows(4096);
        w.write_cols(vec![
            Col::U32(
                "candidate_id".into(),
                (0..n).map(|i| (i * 2 / n) as u32).collect(),
            ),
            Col::F64(
                "mz".into(),
                (0..n).map(|i| 150.0 + i as f64 * 1e-3).collect(),
            ),
            Col::F32("predicted_intensity".into(), vec![1.0; n]),
            Col::Str("name".into(), (0..n).map(|i| format!("n{i}")).collect()),
        ])
        .unwrap();
        w.close().unwrap();
        let (a, b) = (full(&many), mz(&many));
        assert!(
            a.as_deref()
                .is_some_and(|e| e.contains("distinct fragment names")),
            "the full load must refuse the name limit: {a:?}"
        );
        assert_eq!(
            a, b,
            "the m/z-only load must refuse the name limit the same way"
        );
        for parts in [1usize, 4] {
            let e = load_fragments(&many, 0, 2, false, parts, false)
                .err()
                .map(|e| format!("{e:#}"));
            assert!(
                e.as_deref()
                    .is_some_and(|e| e.contains("distinct fragment names")),
                "{parts} parts, m/z only: {e:?}"
            );
        }
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
