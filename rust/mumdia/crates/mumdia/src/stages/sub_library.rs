//! Build a sub-library from a set of candidates: the precursors a screen kept, renumbered
//! to the contiguous `0..n` range the fragment index requires, with their fragment rows
//! remapped to match (`mumdia sub-library`).
//!
//! This is what a second pass searches: the survivors of a first pass, or of `prescan`, as
//! a library of their own. A contiguous m/z *band* of a library needs none of this, because
//! a band is a row range and the engine reads one directly
//! (`groups::write_band_slice` plus `Library::load_with_fragment_offset`); this stage is for
//! an arbitrary set of ids scattered through the library.
//!
//! Pair-linked by construction: a target and its decoy share `peptidoform_id` (the decoy
//! builders copy the target row), so the decision is taken once per pair and no member of a
//! pair can be kept while the other is dropped. Target/decoy exchangeability downstream is
//! then exactly what it was in the full library. `base_peptide_id` would additionally union
//! the charge states, which on a 9-mer immunopeptidomics library turned 23% survivors into
//! 60% kept, so it is not the key.
//!
//! Streaming: both tables are read one batch at a time and written the same way. What is
//! resident is one batch of each, two `u32` per library precursor (the pair key and the
//! old-to-new map), one `bool` per precursor, and under `--pair-link` one more `bool` per
//! peptidoform id up to the largest one the library carries. That is why a 203M-precursor
//! library subsets in about 1.6 GB: the fixed cost is the three per-precursor vectors, not
//! the tables.

use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{Array, BooleanArray, StringArray, UInt32Array};
use arrow::compute::filter_record_batch;
use arrow::record_batch::RecordBatch;
use mumdia_io::table::{BatchWriter, TableFile};
use std::sync::Arc;
use tracing::{info, warn};

const BATCH_ROWS: usize = 1 << 16;
const ROW_GROUP_ROWS: usize = 1 << 17;
/// Marks a library precursor that the sub-library does not keep.
const DROPPED: u32 = u32::MAX;

pub struct SubLibraryParams<'a> {
    pub precursors: &'a str,
    pub fragments: &'a str,
    /// Parquet with a `candidate_id` column: the candidates a screen kept. Order and
    /// duplicates do not matter.
    pub survivors: &'a str,
    pub out_precursors: &'a str,
    pub out_fragments: &'a str,
    /// Union the decision over `peptidoform_id`, so a target and its decoy travel together.
    /// Off keeps exactly the listed candidates, which breaks pairing; only for a list that
    /// is already pair-complete.
    pub pair_linked: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SubLibraryStats {
    pub library_precursors: u64,
    pub survivors: u64,
    pub precursors: u64,
    pub fragments: u64,
    pub targets: u64,
    pub decoys: u64,
}

/// The `candidate_id` column of a batch, checked.
fn ids_of(batch: &RecordBatch, ix: usize, path: &str) -> Result<UInt32Array> {
    let a = batch
        .column(ix)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| anyhow!("{path}: candidate_id is not u32"))?;
    if a.null_count() > 0 {
        bail!("{path}: candidate_id has nulls");
    }
    Ok(a.clone())
}

fn column_index(schema: &arrow::datatypes::Schema, name: &str, path: &str) -> Result<usize> {
    schema
        .index_of(name)
        .map_err(|_| anyhow!("{path} has no {name} column"))
}

pub fn run(p: SubLibraryParams) -> Result<SubLibraryStats> {
    let t0 = Instant::now();
    let pt = TableFile::open(p.precursors)?;
    let n = pt.nrows;
    if n == 0 {
        bail!("{}: empty library", p.precursors);
    }

    // Pass 1: the pair key per row, and the row-aligned id invariant the fragment index
    // depends on (`index.rs`).
    let mut pair: Vec<u32> = Vec::with_capacity(n);
    {
        let reader = pt.batches(Some(&["candidate_id", "peptidoform_id"]), BATCH_ROWS)?;
        let schema = reader.schema();
        let cid_ix = column_index(&schema, "candidate_id", p.precursors)?;
        let pf_ix = column_index(&schema, "peptidoform_id", p.precursors)?;
        for b in reader {
            let b = b?;
            let cid = ids_of(&b, cid_ix, p.precursors)?;
            let pf = ids_of(&b, pf_ix, p.precursors)?;
            for i in 0..b.num_rows() {
                if cid.value(i) as usize != pair.len() {
                    bail!(
                        "{}: candidate_id must be the row index 0..n-1; row {} carries {}",
                        p.precursors,
                        pair.len(),
                        cid.value(i)
                    );
                }
                pair.push(pf.value(i));
            }
        }
    }

    // The survivors, and the keep decision unioned over the pair key.
    let surv = TableFile::open(p.survivors)?.u32("candidate_id")?;
    let n_surv = surv.len() as u64;
    let mut keep: Vec<bool> = vec![false; n];
    if p.pair_linked {
        let max_pair = pair.iter().copied().max().unwrap_or(0) as usize;
        let mut group_kept = vec![false; max_pair + 1];
        for &c in &surv {
            let c = c as usize;
            if c >= n {
                bail!(
                    "{}: candidate_id {c} is outside the library's 0..{n}",
                    p.survivors
                );
            }
            group_kept[pair[c] as usize] = true;
        }
        for (i, k) in keep.iter_mut().enumerate() {
            *k = group_kept[pair[i] as usize];
        }
    } else {
        for &c in &surv {
            let c = c as usize;
            if c >= n {
                bail!(
                    "{}: candidate_id {c} is outside the library's 0..{n}",
                    p.survivors
                );
            }
            keep[c] = true;
        }
    }
    drop(pair);

    // Old id -> new id, ascending, so the output keeps the library's m/z order.
    let mut new_id: Vec<u32> = vec![DROPPED; n];
    let mut n_keep: u32 = 0;
    for (i, &k) in keep.iter().enumerate() {
        if k {
            new_id[i] = n_keep;
            n_keep += 1;
        }
    }
    if n_keep == 0 {
        bail!("sub-library: the survivor set keeps no precursor");
    }

    // Pass 2: the kept precursor rows, renumbered.
    let (mut targets, mut decoys) = (0u64, 0u64);
    let mut rows_out = 0u64;
    {
        let reader = pt.batches(None, BATCH_ROWS)?;
        let schema = reader.schema();
        let cid_ix = column_index(&schema, "candidate_id", p.precursors)?;
        let label_ix = column_index(&schema, "label", p.precursors)?;
        let mut w =
            BatchWriter::with_row_group_rows(p.out_precursors, schema.clone(), ROW_GROUP_ROWS)?;
        let mut first_row = 0usize;
        for b in reader {
            let b = b?;
            let rows = b.num_rows();
            let mask: Vec<bool> = (0..rows).map(|i| keep[first_row + i]).collect();
            let mapped: Vec<u32> = (0..rows)
                .map(|i| new_id[first_row + i])
                .filter(|&v| v != DROPPED)
                .collect();
            first_row += rows;
            if mapped.is_empty() {
                continue;
            }
            let kept = filter_record_batch(&b, &BooleanArray::from(mask))?;
            let label = kept
                .column(label_ix)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("{}: label is not utf8", p.precursors))?;
            for i in 0..kept.num_rows() {
                if label.value(i) == "decoy" {
                    decoys += 1;
                } else {
                    targets += 1;
                }
            }
            let mut cols = kept.columns().to_vec();
            cols[cid_ix] = Arc::new(UInt32Array::from(mapped));
            let out = RecordBatch::try_new(schema.clone(), cols)
                .with_context(|| format!("rebuilding a batch of {}", p.precursors))?;
            rows_out += out.num_rows() as u64;
            w.write(&out)?;
        }
        w.close()?;
    }
    drop(keep);

    // Pass 3: the fragment rows of those candidates, ids remapped. Row order is preserved,
    // so a table sorted by `candidate_id` stays sorted.
    let ft = TableFile::open(p.fragments)?;
    let mut frag_rows = 0u64;
    {
        let reader = ft.batches(None, BATCH_ROWS)?;
        let schema = reader.schema();
        let cid_ix = column_index(&schema, "candidate_id", p.fragments)?;
        let mut w =
            BatchWriter::with_row_group_rows(p.out_fragments, schema.clone(), ROW_GROUP_ROWS)?;
        for b in reader {
            let b = b?;
            let cid = ids_of(&b, cid_ix, p.fragments)?;
            let mut mask = Vec::with_capacity(b.num_rows());
            let mut mapped = Vec::with_capacity(b.num_rows());
            for i in 0..b.num_rows() {
                let c = cid.value(i) as usize;
                if c >= n {
                    bail!(
                        "{}: fragment row references candidate_id {c}, outside the library's 0..{n}",
                        p.fragments
                    );
                }
                let m = new_id[c];
                mask.push(m != DROPPED);
                if m != DROPPED {
                    mapped.push(m);
                }
            }
            if mapped.is_empty() {
                continue;
            }
            let kept = filter_record_batch(&b, &BooleanArray::from(mask))?;
            let mut cols = kept.columns().to_vec();
            cols[cid_ix] = Arc::new(UInt32Array::from(mapped));
            let out = RecordBatch::try_new(schema.clone(), cols)
                .with_context(|| format!("rebuilding a batch of {}", p.fragments))?;
            frag_rows += out.num_rows() as u64;
            w.write(&out)?;
        }
        w.close()?;
    }
    if frag_rows == 0 {
        warn!(
            fragments = %p.fragments,
            "sub-library: no fragment row belongs to a kept candidate; the search will find nothing"
        );
    }

    let stats = SubLibraryStats {
        library_precursors: n as u64,
        survivors: n_surv,
        precursors: rows_out,
        fragments: frag_rows,
        targets,
        decoys,
    };
    info!(
        library_precursors = stats.library_precursors,
        survivors = stats.survivors,
        precursors = stats.precursors,
        fragments = stats.fragments,
        targets = stats.targets,
        decoys = stats.decoys,
        pair_linked = p.pair_linked,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "sub-library: done"
    );
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::{write_table, Col};

    /// Six precursors: three peptidoforms, each a target and its decoy sharing
    /// `peptidoform_id`, with two fragments each.
    fn lib(dir: &std::path::Path) -> (String, String) {
        let p = dir.join("prec.parquet").to_str().unwrap().to_string();
        let f = dir.join("frag.parquet").to_str().unwrap().to_string();
        write_table(
            &p,
            vec![
                Col::U32("candidate_id".into(), (0..6).collect()),
                Col::U32("peptidoform_id".into(), vec![0, 0, 1, 1, 2, 2]),
                Col::U32("base_peptide_id".into(), vec![0, 0, 1, 1, 2, 2]),
                Col::Str(
                    "peptidoform".into(),
                    ["A", "A", "B", "B", "C", "C"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                ),
                Col::I32("charge".into(), vec![2; 6]),
                Col::F64(
                    "precursor_mz".into(),
                    vec![400.0, 400.0, 500.0, 500.0, 600.0, 600.0],
                ),
                Col::F32("predicted_irt".into(), vec![1.0; 6]),
                Col::Str(
                    "label".into(),
                    ["target", "decoy", "target", "decoy", "target", "decoy"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                ),
                Col::Str("protein".into(), vec!["P".to_string(); 6]),
            ],
        )
        .unwrap();
        write_table(
            &f,
            vec![
                Col::U32(
                    "candidate_id".into(),
                    vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                ),
                Col::F32(
                    "frag_mz".into(),
                    (0..12).map(|i| 100.0 + i as f32).collect(),
                ),
                Col::F32("predicted_intensity".into(), vec![0.5; 12]),
                Col::Str("frag_name".into(), vec!["y1".to_string(); 12]),
            ],
        )
        .unwrap();
        (p, f)
    }

    #[test]
    fn the_sub_library_is_renumbered_pair_linked_and_its_fragments_follow() {
        let dir = std::env::temp_dir().join(format!("mumdia_sublib_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, f) = lib(&dir);
        // One survivor from pair 0 (the target) and one from pair 2 (the decoy): both whole
        // pairs must come through, pair 1 must not.
        let s = dir.join("surv.parquet").to_str().unwrap().to_string();
        write_table(&s, vec![Col::U32("candidate_id".into(), vec![0, 5])]).unwrap();
        let op = dir.join("out_prec.parquet").to_str().unwrap().to_string();
        let of = dir.join("out_frag.parquet").to_str().unwrap().to_string();
        let st = run(SubLibraryParams {
            precursors: &p,
            fragments: &f,
            survivors: &s,
            out_precursors: &op,
            out_fragments: &of,
            pair_linked: true,
        })
        .unwrap();
        assert_eq!(st.precursors, 4, "both members of pairs 0 and 2");
        assert_eq!((st.targets, st.decoys), (2, 2), "pairing intact");
        assert_eq!(st.fragments, 8);
        let out = TableFile::open(&op).unwrap();
        assert_eq!(out.u32("candidate_id").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(
            out.f64("precursor_mz").unwrap(),
            vec![400.0, 400.0, 600.0, 600.0],
            "library m/z order preserved"
        );
        let fo = TableFile::open(&of).unwrap();
        assert_eq!(
            fo.u32("candidate_id").unwrap(),
            vec![0, 0, 1, 1, 2, 2, 3, 3],
            "fragments remapped onto the new ids, order preserved"
        );
        assert_eq!(
            fo.f32("frag_mz").unwrap(),
            vec![100.0, 101.0, 102.0, 103.0, 108.0, 109.0, 110.0, 111.0],
            "pair 1's fragments dropped, the others' kept verbatim"
        );

        // Without pair linking the listed candidates are kept as given.
        let op2 = dir.join("out_prec2.parquet").to_str().unwrap().to_string();
        let of2 = dir.join("out_frag2.parquet").to_str().unwrap().to_string();
        let st2 = run(SubLibraryParams {
            precursors: &p,
            fragments: &f,
            survivors: &s,
            out_precursors: &op2,
            out_fragments: &of2,
            pair_linked: false,
        })
        .unwrap();
        assert_eq!(st2.precursors, 2);
        assert_eq!((st2.targets, st2.decoys), (1, 1));
        assert_eq!(
            TableFile::open(&of2).unwrap().u32("candidate_id").unwrap(),
            vec![0, 0, 1, 1]
        );
    }

    #[test]
    fn a_library_whose_ids_are_not_row_aligned_is_refused() {
        let dir = std::env::temp_dir().join(format!("mumdia_sublib_bad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p, f) = lib(&dir);
        let bad = dir.join("bad_prec.parquet").to_str().unwrap().to_string();
        let t = TableFile::open(&p).unwrap();
        write_table(
            &bad,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1, 2, 3, 4, 9]),
                Col::U32("peptidoform_id".into(), t.u32("peptidoform_id").unwrap()),
                Col::Str("label".into(), t.str("label").unwrap()),
            ],
        )
        .unwrap();
        let s = dir.join("surv2.parquet").to_str().unwrap().to_string();
        write_table(&s, vec![Col::U32("candidate_id".into(), vec![0])]).unwrap();
        let err = run(SubLibraryParams {
            precursors: &bad,
            fragments: &f,
            survivors: &s,
            out_precursors: dir.join("o.parquet").to_str().unwrap(),
            out_fragments: dir.join("of.parquet").to_str().unwrap(),
            pair_linked: true,
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("row index"), "{err}");
    }
}
