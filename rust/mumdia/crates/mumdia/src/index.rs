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
use mumdia_core::constants::{ppm_bounds, PROTON};
use mumdia_io::table::Table;
use rayon::prelude::*;

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
    pub frag_mz: Vec<f64>,
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
        let pt = Table::read(precursors)?;
        let cid = pt.u32("candidate_id")?;
        let pfid = pt.u32("peptidoform_id")?;
        let baseid = pt.u32("base_peptide_id")?;
        let pform = pt.str("peptidoform")?;
        let charge = pt.i32("charge")?;
        let pmz = pt.f64("precursor_mz")?;
        let irt = pt.f32("predicted_irt")?;
        let label = pt.str("label")?;
        let protein = pt.str("protein")?;
        crate::fdr::validate_labels(&label)?;

        let ncand = pt.nrows;
        // The typed getters above returned owned Vecs, so the decoded Arrow batches are
        // dead weight from here on. Release them before the fragment table is read:
        // holding both tables plus every derived Vec is what makes library load the peak-
        // RSS wall (the fragment table alone is ~23 GB of Arrow batches at 657M rows).
        drop(pt);

        // Projected: the fragment artifact also carries `ion_type`, `ordinal`,
        // `frag_charge` and `cardinality`, none of which the library reads. Decoding them
        // costs a full pass and a full copy of each at fragment-library scale (hundreds of
        // millions of rows), and `ion_type` is a string column, so it also allocates.
        let ft = Table::read_cols(
            fragments,
            &["candidate_id", "mz", "predicted_intensity", "name"],
        )?;
        let f_cid = ft.u32("candidate_id")?;
        let f_mz = ft.f64("mz")?;
        let f_int = ft.f32("predicted_intensity")?;
        let mut f_name = ft.str("name")?;
        let n_frag_rows = ft.nrows;
        drop(ft);
        // Precondition: candidate_id is the contiguous, row-aligned range 0..ncand
        // (the library + decoy builders guarantee this). An external library that
        // violates it would misgroup fragments or panic on the index below, so
        // check explicitly and fail with a clear error instead.
        for (c, &candidate_id) in cid.iter().enumerate().take(ncand) {
            if candidate_id as usize != c {
                anyhow::bail!(
                    "library precursor row {c} has candidate_id {} but candidate_id must \
                     be the contiguous range 0..{ncand} in row order; reindex the library \
                     (e.g. via the decoy-builder scripts)",
                    candidate_id
                );
            }
        }
        // Group fragments by candidate_id, preserving stored order, via a counting sort
        // into two flat arrays. The previous `Vec<Vec<usize>>` performed one heap
        // allocation per candidate (54.8M of them on the profiled library) to express a
        // grouping that a counting sort does with exactly two allocations. Scattering in
        // ascending row order keeps each candidate's fragments in stored order, so the
        // resulting layout is identical to before.
        let mut frag_offsets: Vec<u32> = vec![0; ncand + 1];
        for (i, &candidate_id) in f_cid.iter().enumerate().take(n_frag_rows) {
            let c = candidate_id as usize;
            if c >= ncand {
                anyhow::bail!(
                    "fragment row {i} references candidate_id {c} >= precursor count {ncand}"
                );
            }
            frag_offsets[c + 1] += 1;
        }
        for c in 0..ncand {
            frag_offsets[c + 1] += frag_offsets[c];
        }
        let mut frag_order: Vec<u32> = vec![0; n_frag_rows];
        {
            let mut cursor = frag_offsets.clone();
            for (i, &candidate_id) in f_cid.iter().enumerate().take(n_frag_rows) {
                let c = candidate_id as usize;
                frag_order[cursor[c] as usize] = i as u32;
                cursor[c] += 1;
            }
        }

        let mut cands = Vec::with_capacity(ncand);
        let mut frag_mz = Vec::with_capacity(n_frag_rows);
        let mut frag_int = Vec::with_capacity(n_frag_rows);
        let mut frag_name_id: Vec<u16> = Vec::with_capacity(n_frag_rows);
        let mut frag_name_dict: Vec<String> = Vec::new();
        let mut name_lookup: std::collections::HashMap<String, u16> =
            std::collections::HashMap::new();
        let mut prec_mz = Vec::with_capacity(ncand);

        for c in 0..ncand {
            let start = frag_mz.len();
            for &fi32 in &frag_order[frag_offsets[c] as usize..frag_offsets[c + 1] as usize] {
                let fi = fi32 as usize;
                frag_mz.push(f_mz[fi]);
                frag_int.push(f_int[fi]);
                // Intern the name: fragment names come from a tiny repeating vocabulary
                // (b1, y7, y12^2, ...), so store a u16 dictionary index per fragment
                // instead of a String. A `Vec<String>` costs ~24 B of struct per fragment
                // before any text -- ~16 GB per copy at 657M fragments -- and the old code
                // additionally held a second copy while moving names across.
                let name = std::mem::take(&mut f_name[fi]);
                let id = match name_lookup.get(&name) {
                    Some(&id) => id,
                    None => {
                        let id = u16::try_from(frag_name_dict.len()).map_err(|_| {
                            anyhow::anyhow!(
                                "library has more than {} distinct fragment names; the                                  interned name id is a u16",
                                u16::MAX
                            )
                        })?;
                        name_lookup.insert(name.clone(), id);
                        frag_name_dict.push(name);
                        id
                    }
                };
                frag_name_id.push(id);
            }
            let n = frag_mz.len() - start;
            cands.push(Candidate {
                candidate_id: cid[c],
                peptidoform_id: pfid[c],
                base_peptide_id: baseid[c],
                peptidoform: pform[c].clone(),
                charge: charge[c],
                precursor_mz: pmz[c],
                predicted_irt: irt[c],
                is_decoy: label[c] == "decoy",
                protein: protein[c].clone(),
                frag_start: start,
                n_frag: n,
            });
            prec_mz.push(pmz[c]);
        }
        // Names have all been moved out; free the (now empty) source vector and the
        // grouping arrays before the index build allocates `entries`.
        drop(f_name);
        drop(name_lookup);
        drop(frag_order);
        drop(frag_offsets);

        // Precondition for `candidate_range`: precursors ascending by m/z. The
        // fragment-index `partition_point` search over `prec_mz` assumes this;
        // an unsorted import (e.g. `import_diann_lib.py` output fed directly,
        // skipping the sorting decoy builder) would silently return wrong
        // candidate windows. Check explicitly and fail loudly.
        for c in 1..ncand {
            if prec_mz[c] < prec_mz[c - 1] {
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
                    entries.push((frag_mz[gi] as f32, cd.candidate_id, frag_int[gi]));
                }
            }
            // Global sort by fragment m/z. Parallel stable sort: identical result to
            // the serial stable `sort_by` (same comparator, ties keep input order).
            entries.par_sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

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
    pub fn cand_frags(&self, cid: u32) -> (&[f64], &[f32], &[u16]) {
        let c = &self.cands[cid as usize];
        let s = c.frag_start;
        let e = s + c.n_frag;
        (
            &self.frag_mz[s..e],
            &self.frag_int[s..e],
            &self.frag_name_id[s..e],
        )
    }

    /// Local fragment index (0..n_frag) of the candidate whose stored m/z is
    /// closest to `frag_mz_f32` (returned by `page_search`).
    pub fn local_frag_index(&self, cid: u32, frag_mz_f32: f32) -> usize {
        let (mzs, _, _) = self.cand_frags(cid);
        let mut best = 0usize;
        let mut bestd = f32::MAX;
        for (i, &m) in mzs.iter().enumerate() {
            let d = ((m as f32) - frag_mz_f32).abs();
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
    use mumdia_io::table::{write_table, Col};

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
        let dir = std::env::temp_dir().join("mumdia_index_test");
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
}
