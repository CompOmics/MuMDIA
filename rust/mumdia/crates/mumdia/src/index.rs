//! The inverted fragment index (PLAN.md Stage D part 1) shared by search-seed
//! and extract.
//!
//! Structure-of-Arrays, flat and contiguous: three parallel arrays
//! (`idx_mz`, `idx_cid`, `idx_int`) globally sorted by fragment m/z, then
//! chunked into fixed buckets; within a bucket the order is precursor m/z
//! (== `candidate_id`, since candidates are ordered by precursor m/z at library
//! build). One binary search over `bucket_min` selects buckets overlapping a
//! query, a second over `candidate_id` narrows to the isolation-window slice,
//! and a linear tail applies the exact ppm bound (mirrors Sage `page_search`).
//!
//! MVP stores the index m/z as f32 and does all ppm math in f64 (PLAN.md
//! Stage D precision note). RT is applied as a per-candidate window post-filter
//! at probe time rather than a pre-partition (the documented fallback, PLAN.md
//! Stage D part 2), which keeps the index run-independent.

use anyhow::Result;
use mumdia_core::constants::{ppm_bounds, PROTON};
use mumdia_io::table::Table;
use rayon::prelude::*;
use tracing::warn;

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
    pub frag_name: Vec<String>,
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
    pub fn load(precursors: &str, fragments: &str, bucket_size: usize) -> Result<Library> {
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

        let ft = Table::read(fragments)?;
        let f_cid = ft.u32("candidate_id")?;
        let f_mz = ft.f64("mz")?;
        let f_int = ft.f32("predicted_intensity")?;
        let f_name = ft.str("name")?;

        let ncand = pt.nrows;
        // Precondition: candidate_id is the contiguous, row-aligned range 0..ncand
        // (the library + decoy builders guarantee this). An external library that
        // violates it would misgroup fragments or panic on the index below, so
        // check explicitly and fail with a clear error instead.
        for c in 0..ncand {
            if cid[c] as usize != c {
                anyhow::bail!(
                    "library precursor row {c} has candidate_id {} but candidate_id must \
                     be the contiguous range 0..{ncand} in row order; reindex the library \
                     (e.g. via the decoy-builder scripts)",
                    cid[c]
                );
            }
        }
        // Group fragments by candidate_id, preserving stored order.
        let mut per_cand_frags: Vec<Vec<usize>> = vec![Vec::new(); ncand];
        for i in 0..ft.nrows {
            let c = f_cid[i] as usize;
            if c >= ncand {
                anyhow::bail!(
                    "fragment row {i} references candidate_id {c} >= precursor count {ncand}"
                );
            }
            per_cand_frags[c].push(i);
        }

        let mut cands = Vec::with_capacity(ncand);
        let mut frag_mz = Vec::with_capacity(ft.nrows);
        let mut frag_int = Vec::with_capacity(ft.nrows);
        let mut frag_name = Vec::with_capacity(ft.nrows);
        let mut prec_mz = Vec::with_capacity(ncand);

        for c in 0..ncand {
            let start = frag_mz.len();
            for &fi in &per_cand_frags[c] {
                frag_mz.push(f_mz[fi]);
                frag_int.push(f_int[fi]);
                frag_name.push(f_name[fi].clone());
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
        // A decoy-free library gives q = (0+1)/n_targets, i.e. silently-invalid
        // near-zero FDR downstream. Warn loudly (bailing could break intentional
        // decoy-free diagnostics, so this is a warning, not an error).
        let n_decoy = cands.iter().filter(|c| c.is_decoy).count();
        if n_decoy == 0 {
            warn!(
                "library has 0 decoy candidates; target-decoy q-values will be invalid \
                 (add decoys, e.g. via make_reverse_decoys.py)"
            );
        }

        // Build flat index entries.
        let mut entries: Vec<(f32, u32, f32)> = Vec::with_capacity(ft.nrows);
        for c in 0..ncand {
            let cd = &cands[c];
            for k in 0..cd.n_frag {
                let gi = cd.frag_start + k;
                entries.push((frag_mz[gi] as f32, cd.candidate_id, frag_int[gi]));
            }
        }
        // Global sort by fragment m/z. Parallel stable sort: identical result to
        // the serial stable `sort_by` (same comparator, ties keep input order).
        entries.par_sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Chunk into buckets; within a bucket sort by candidate_id.
        let bs = bucket_size.max(1);
        let mut bucket_min = Vec::new();
        for chunk_start in (0..entries.len()).step_by(bs) {
            let end = (chunk_start + bs).min(entries.len());
            bucket_min.push(entries[chunk_start].0);
            entries[chunk_start..end].sort_by_key(|e| e.1);
        }

        let mut idx_mz = Vec::with_capacity(entries.len());
        let mut idx_cid = Vec::with_capacity(entries.len());
        let mut idx_int = Vec::with_capacity(entries.len());
        for (m, c, i) in entries {
            idx_mz.push(m);
            idx_cid.push(c);
            idx_int.push(i);
        }

        Ok(Library {
            cands,
            frag_mz,
            frag_int,
            frag_name,
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
    pub fn cand_frags(&self, cid: u32) -> (&[f64], &[f32], &[String]) {
        let c = &self.cands[cid as usize];
        let s = c.frag_start;
        let e = s + c.n_frag;
        (&self.frag_mz[s..e], &self.frag_int[s..e], &self.frag_name[s..e])
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
    /// predicted_intensity)` for each match (PLAN.md Stage D `page_search`).
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
        let first = self.bucket_min.partition_point(|&m| m <= lo32).saturating_sub(1);
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

/// Deconvolve an observed z-charged peak m/z to neutral m/z (PLAN.md Stage D
/// point 3). Done in f64.
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
                Col::Str("peptidoform".into(), vec!["PEPTIDEK".into(), "SAMPLER".into()]),
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
                Col::Str("name".into(), vec!["b2".into(), "y3".into(), "b2".into(), "y3".into()]),
                Col::Str("ion_type".into(), vec!["b".into(), "y".into(), "b".into(), "y".into()]),
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
