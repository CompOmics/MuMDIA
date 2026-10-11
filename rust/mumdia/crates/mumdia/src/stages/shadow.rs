//! Shadow demotion (`rescore.shadow_min_shared`).
//!
//! In DIA a co-eluting, co-isolated peptide that the first pass already accepts lends its
//! fragments to every other candidate whose library fragments fall on the same m/z: a
//! "shadow" scores on borrowed evidence. On the single-cell and HeLa Astral runs 25-37% of
//! the decoys above the 1% cut were such shadows, and they set the cut. A candidate of
//! either label is demoted below every other score when at least `min_shared` of its
//! library fragments lie within `ppm` of a fragment of a higher-scoring, first-pass-accepted
//! target in the same run and isolation window whose apex is within `rt_s` of its own; the
//! candidate's own base peptide (itself, its target/decoy partner, its sibling charge
//! states and modforms) never counts as the lender. Only the candidates in the top
//! `region x accepted` rows of their run by first-pass score are examined (label-blind),
//! because a candidate further down cannot cross the cut anyway.
//!
//! Deterministic: rows are examined independently, lenders are pooled per row and sorted,
//! and no floating-point value is reduced across rows.
use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};
use arrow::array::{Array, Float32Array, Float64Array, UInt32Array};
use mumdia_io::table::TableFile;
use rayon::prelude::*;

/// What the demotion reads besides the scored rows: the library's fragments and, per run
/// (`source`), its isolation windows.
pub struct ShadowInputs<'a> {
    pub lib_fragments: &'a str,
    /// `isolation_windows.parquet` of each run, indexed by `source`.
    pub isolation_windows: &'a [String],
}

impl<'a> ShadowInputs<'a> {
    pub fn new(lib_fragments: &'a str, isolation_windows: &'a [String]) -> Self {
        ShadowInputs {
            lib_fragments,
            isolation_windows,
        }
    }
}

/// A lender: `(apex_rt, score, candidate_id, base_peptide_id)`.
type Lender = (f64, f64, u32, u32);

/// The scored rows the demotion examines, all of length `n`.
pub struct ShadowRows<'a> {
    pub source: &'a [u32],
    pub cid: &'a [u32],
    pub base: &'a [u32],
    pub apex_rt: &'a [f64],
    pub precursor_mz: &'a [f64],
    pub score: &'a [f64],
    pub is_decoy: &'a [bool],
    /// First-pass accepted targets (the possible lenders).
    pub accepted: &'a [bool],
}

#[derive(Clone, Copy, Debug)]
pub struct ShadowParams {
    pub min_shared: usize,
    pub ppm: f64,
    pub rt_s: f64,
    pub region: usize,
}

/// Isolation windows of one run as ascending `(lower, upper)`.
fn read_windows(path: &str) -> Result<Vec<(f64, f64)>> {
    let t = TableFile::open(path).with_context(|| format!("shadow: isolation windows {path}"))?;
    let lo = t.f64("lower")?;
    let hi = t.f64("upper")?;
    let mut w: Vec<(f64, f64)> = lo.into_iter().zip(hi).collect();
    w.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    Ok(w)
}

/// The window index holding `mz` (the last window starting at or below it), if any.
fn window_of(w: &[(f64, f64)], mz: f64) -> Option<u32> {
    let j = w.partition_point(|x| x.0 <= mz);
    if j == 0 {
        return None;
    }
    let (lo, hi) = w[j - 1];
    (mz >= lo && mz < hi).then_some((j - 1) as u32)
}

/// Sorted library fragment m/z of every candidate in `need`.
fn read_fragments(path: &str, need: &HashSet<u32>) -> Result<HashMap<u32, Vec<f64>>> {
    let t = TableFile::open(path).with_context(|| format!("shadow: library fragments {path}"))?;
    let mut out: HashMap<u32, Vec<f64>> = HashMap::with_capacity(need.len());
    t.for_each_batch(Some(&["candidate_id", "mz"]), 1 << 20, |b| {
        let ci = b.schema().index_of("candidate_id")?;
        let mi = b.schema().index_of("mz")?;
        let cid = b
            .column(ci)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("shadow: lib_fragments.candidate_id is not u32")?;
        let mzc = b.column(mi);
        let mz: Vec<f64> = if let Some(a) = mzc.as_any().downcast_ref::<Float64Array>() {
            a.values().to_vec()
        } else if let Some(a) = mzc.as_any().downcast_ref::<Float32Array>() {
            a.values().iter().map(|&x| x as f64).collect()
        } else {
            bail!("shadow: lib_fragments.mz is neither f64 nor f32");
        };
        for (i, &c) in cid.values().iter().enumerate() {
            if need.contains(&c) {
                out.entry(c).or_default().push(mz[i]);
            }
        }
        Ok(())
    })?;
    for v in out.values_mut() {
        v.sort_by(|a, b| a.total_cmp(b));
    }
    Ok(out)
}

/// Fragments of `frag` within `ppm` of some value in the sorted `pool`.
fn n_shared(frag: &[f64], pool: &[f64], ppm: f64) -> usize {
    if pool.is_empty() {
        return 0;
    }
    frag.iter()
        .filter(|&&f| {
            let p = pool.partition_point(|&x| x < f);
            let tol = f * ppm * 1e-6;
            (p < pool.len() && (pool[p] - f).abs() <= tol)
                || (p > 0 && (f - pool[p - 1]).abs() <= tol)
        })
        .count()
}

/// The rows to demote. `rows.*` must all have the same length.
pub fn flag(rows: &ShadowRows, inputs: &ShadowInputs, prm: ShadowParams) -> Result<Vec<bool>> {
    let n = rows.cid.len();
    let n_sources = rows
        .source
        .iter()
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    if inputs.isolation_windows.len() < n_sources {
        bail!(
            "shadow demotion needs the isolation windows of every run: {} given, {} runs",
            inputs.isolation_windows.len(),
            n_sources
        );
    }
    let windows: Vec<Vec<(f64, f64)>> = inputs.isolation_windows[..n_sources]
        .iter()
        .map(|p| read_windows(p))
        .collect::<Result<_>>()?;
    let win: Vec<Option<u32>> = (0..n)
        .map(|i| window_of(&windows[rows.source[i] as usize], rows.precursor_mz[i]))
        .collect();

    // The label-blind region of each run: its top `region x accepted` rows by score.
    let mut by_source: Vec<Vec<usize>> = vec![Vec::new(); n_sources];
    for i in 0..n {
        by_source[rows.source[i] as usize].push(i);
    }
    let mut in_region = vec![false; n];
    for idx in by_source.iter_mut() {
        let nacc = idx.iter().filter(|&&i| rows.accepted[i]).count();
        idx.sort_by(|&a, &b| rows.score[b].total_cmp(&rows.score[a]).then(a.cmp(&b)));
        for &i in idx.iter().take(prm.region.saturating_mul(nacc)) {
            in_region[i] = true;
        }
    }

    // Lenders per (run, window): (apex_rt, score, cid, base), ascending by apex_rt.
    let mut lenders: HashMap<(u32, u32), Vec<Lender>> = HashMap::new();
    for (i, wi) in win.iter().enumerate() {
        if rows.accepted[i] {
            if let Some(w) = *wi {
                lenders.entry((rows.source[i], w)).or_default().push((
                    rows.apex_rt[i],
                    rows.score[i],
                    rows.cid[i],
                    rows.base[i],
                ));
            }
        }
    }
    for v in lenders.values_mut() {
        v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
    }

    let mut need: HashSet<u32> = HashSet::new();
    for (i, &reg) in in_region.iter().enumerate() {
        if reg || rows.accepted[i] {
            need.insert(rows.cid[i]);
        }
    }
    let frags = read_fragments(inputs.lib_fragments, &need)?;

    let demote: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !in_region[i] {
                return false;
            }
            let (Some(w), Some(own)) = (win[i], frags.get(&rows.cid[i])) else {
                return false;
            };
            let Some(cands) = lenders.get(&(rows.source[i], w)) else {
                return false;
            };
            let rt = rows.apex_rt[i];
            let lo = cands.partition_point(|c| c.0 < rt - prm.rt_s);
            let mut pool: Vec<f64> = Vec::new();
            for c in &cands[lo..] {
                if c.0 > rt + prm.rt_s {
                    break;
                }
                if c.2 == rows.cid[i] || c.3 == rows.base[i] || c.1 <= rows.score[i] {
                    continue;
                }
                if let Some(f) = frags.get(&c.2) {
                    pool.extend_from_slice(f);
                }
            }
            if pool.is_empty() {
                return false;
            }
            pool.sort_by(|a, b| a.total_cmp(b));
            n_shared(own, &pool, prm.ppm) >= prm.min_shared
        })
        .collect();
    Ok(demote)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_and_shared_fragments() {
        let w = vec![(400.0, 420.0), (420.0, 440.0)];
        assert_eq!(window_of(&w, 399.9), None);
        assert_eq!(window_of(&w, 400.0), Some(0));
        assert_eq!(window_of(&w, 420.0), Some(1));
        assert_eq!(window_of(&w, 440.0), None);
        let pool = vec![300.0, 500.0, 700.0];
        // 10 ppm of 500 is 0.005
        assert_eq!(n_shared(&[300.002, 500.004, 600.0, 700.01], &pool, 10.0), 2);
        assert_eq!(n_shared(&[300.0], &[], 10.0), 0);
    }

    fn scratch(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("mumdia_shadow_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_str().unwrap().to_string()
    }

    #[test]
    fn only_a_co_eluting_co_isolated_borrower_of_a_better_accepted_target_is_demoted() {
        use mumdia_io::table::{write_table, Col};
        let win = scratch("iw.parquet");
        write_table(
            &win,
            vec![
                Col::U32("window_id".into(), vec![0, 1]),
                Col::F64("target".into(), vec![410.0, 430.0]),
                Col::F64("lower".into(), vec![400.0, 420.0]),
                Col::F64("upper".into(), vec![420.0, 440.0]),
            ],
        )
        .unwrap();
        // cid 1: the accepted lender. cid 2..6 each share 3 of its 4 fragments.
        let lender = [200.0, 300.0, 400.0, 500.0];
        let borrower = [200.001, 300.001, 400.001, 650.0];
        let mut fc = Vec::new();
        let mut fm = Vec::new();
        for c in 1u32..=6 {
            let f: &[f64] = if c == 1 { &lender } else { &borrower };
            for &m in f {
                fc.push(c);
                fm.push(m);
            }
        }
        let frag = scratch("frag.parquet");
        write_table(
            &frag,
            vec![
                Col::U32("candidate_id".into(), fc),
                Col::F64("mz".into(), fm),
            ],
        )
        .unwrap();
        // rows: lender; co-eluting decoy (demote); target 10 s away (keep); the lender's own
        // decoy partner (same base, keep); a candidate in the other window (keep); a
        // co-eluting target scoring ABOVE the lender (keep)
        let source = vec![0u32; 6];
        let cid = vec![1u32, 2, 3, 4, 5, 6];
        let base = vec![10u32, 20, 30, 10, 50, 60];
        let apex = vec![100.0, 101.0, 110.0, 100.5, 100.0, 100.2];
        let mz = vec![410.0, 405.0, 405.0, 410.0, 430.0, 415.0];
        let score = vec![0.99, 0.95, 0.95, 0.95, 0.95, 0.999];
        let is_decoy = vec![false, true, false, true, false, false];
        let accepted = vec![true, false, false, false, false, false];
        let windows = vec![win];
        let d = flag(
            &ShadowRows {
                source: &source,
                cid: &cid,
                base: &base,
                apex_rt: &apex,
                precursor_mz: &mz,
                score: &score,
                is_decoy: &is_decoy,
                accepted: &accepted,
            },
            &ShadowInputs {
                lib_fragments: &frag,
                isolation_windows: &windows,
            },
            ShadowParams {
                min_shared: 3,
                ppm: 10.0,
                rt_s: 3.0,
                region: 8,
            },
        )
        .unwrap();
        assert_eq!(d, vec![false, true, false, false, false, false]);
    }
}
