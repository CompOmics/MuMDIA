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
//! On a run whose spectra carry 1/K0 the test can take a 1/K0 term
//! (`rescore.shadow_im_tolerance`, off by default): a lender then counts only when its apex
//! 1/K0 lies within the tolerance of the candidate's, because two precursors co-isolated in
//! one diaPASEF slot but separated in mobility do not share fragment peaks. The apex 1/K0 of
//! each row is joined from its run's `psms_extracted` on `(candidate_id, peak_rank)`
//! (`extract::apex_im_by_peak`); a row without one keeps the m/z-and-RT test alone, so with
//! the term off, and on every 3D run, the demotion is what it was.
//!
//! On scanning-quadrupole spectra the test can take a Q1 term (`rescore.shadow_q1`, off by
//! default), from each run's per-fragment Q1 decisions (`extract.q1_fragments`, the
//! `q1f_frag_class` and `q1f_frag_offset` lists of `psms_extracted`, and the window's reach,
//! step and offset tolerance from its report). The overlapping windows of a sweep put every
//! precursor in a window of its own, so co-isolation is read from the m/z instead: a lender
//! counts when the two precursors are closer than twice the calibrated window's reach (they
//! share a Q1 step). Each of the candidate's library fragments that a lender shares then
//! counts by where the candidate's patch saw its signal in Q1: 0 when the patch has none for
//! it (nothing was borrowed); 0 when its Q1 centre matches the candidate's expected one and
//! 1 when it matches the lender's, within the decision's offset tolerance; otherwise the
//! two-component split of its centre, `mu / d` (its offset from the candidate's centre over
//! the lender's, both in steps, clamped to 0..1); and 1 when the precursors are closer than
//! the offset tolerance, where Q1 cannot tell them apart, or the fragment has no Q1 profile;
//! the largest over the lenders sharing it.
//! The candidate is demoted when these add up to `min_shared`. A run without the decisions
//! keeps the window test, with a warning.
//!
//! Deterministic: rows are examined independently, lenders are pooled per row and sorted,
//! and no floating-point value is reduced across rows.
use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};
use arrow::array::{Array, Float32Array, Float64Array, UInt32Array};
use mumdia_io::table::TableFile;
use rayon::prelude::*;

/// What the demotion reads besides the scored rows: the library's fragments and, per run
/// (`source`), its isolation windows and, for the 1/K0 term, its `psms_extracted`.
pub struct ShadowInputs<'a> {
    pub lib_fragments: &'a str,
    /// `isolation_windows.parquet` of each run, indexed by `source`.
    pub isolation_windows: &'a [String],
    /// `psms_extracted.parquet` of each run, indexed by `source`, read only for the 1/K0
    /// term (`rescore.shadow_im_tolerance`). Empty when the caller has none.
    pub psms_extracted: &'a [String],
}

impl<'a> ShadowInputs<'a> {
    pub fn new(lib_fragments: &'a str, isolation_windows: &'a [String]) -> Self {
        ShadowInputs {
            lib_fragments,
            isolation_windows,
            psms_extracted: &[],
        }
    }

    /// With each run's `psms_extracted`, for the 1/K0 term.
    pub fn with_psms_extracted(mut self, psms_extracted: &'a [String]) -> Self {
        self.psms_extracted = psms_extracted;
        self
    }
}

/// A lender: `(apex_rt, score, candidate_id, base_peptide_id, apex 1/K0)`.
type Lender = (f64, f64, u32, u32, Option<f64>);

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
    /// The peak each row selected (`selected_peak_rank`), which with `cid` finds the row's
    /// extracted row for the 1/K0 term.
    pub peak_rank: &'a [i32],
}

#[derive(Clone, Copy, Debug)]
pub struct ShadowParams {
    pub min_shared: usize,
    pub ppm: f64,
    pub rt_s: f64,
    pub region: usize,
    /// `rescore.shadow_im_tolerance`: the apex 1/K0 distance within which a lender counts;
    /// 0 is no 1/K0 term.
    pub im_tol: f64,
    /// `rescore.shadow_q1`: the Q1 term of scanning-quadrupole runs.
    pub q1: bool,
}

/// The Q1 term of one run (`rescore.shadow_q1`): from its extract report, the distance
/// within which two precursors are co-isolated (twice the window's reach, m/z), the step
/// (m/z) and the decision's offset tolerance (steps); and per examined row, keyed by
/// `(candidate_id, peak_rank)`, its fragments' classes and Q1 offsets in the library's
/// order.
struct Q1Run {
    reach: f64,
    step: f64,
    tol: f64,
    rows: HashMap<(u32, i32), (Vec<f32>, Vec<f32>)>,
}

/// What a shared fragment counts for under the Q1 term (the module doc): its class and Q1
/// offset `mu` (steps) in the candidate's decision, and `d`, the lender's precursor m/z
/// minus the candidate's, in steps. Assigned whole to the precursor whose centre it matches
/// within the decision's offset tolerance `tol`, split `mu / d` when it matches neither
/// (or both), and counted whole when the two centres are within `tol` of each other.
fn q1_borrowed(class: f32, mu: f32, d: f64, tol: f64) -> f64 {
    if class == 0.0 {
        return 0.0;
    }
    if !mu.is_finite() || d.abs() <= tol {
        return 1.0;
    }
    let mu = mu as f64;
    match ((mu - d).abs() <= tol, mu.abs() <= tol) {
        (true, false) => 1.0,
        (false, true) => 0.0,
        _ => (mu / d).clamp(0.0, 1.0),
    }
}

/// The Q1 term of every run, `None` for a run without the decisions (with a warning). Reads
/// the lists only for the rows in `examine`.
fn q1_runs(
    rows: &ShadowRows,
    inputs: &ShadowInputs,
    n_sources: usize,
    examine: &[bool],
) -> Result<Vec<Option<Q1Run>>> {
    let mut want: Vec<std::collections::HashSet<(u32, i32)>> = vec![Default::default(); n_sources];
    for (i, &e) in examine.iter().enumerate() {
        if e {
            want[rows.source[i] as usize].insert((rows.cid[i], rows.peak_rank[i]));
        }
    }
    (0..n_sources)
        .map(|s| {
            let Some(path) = inputs.psms_extracted.get(s) else {
                tracing::warn!(
                    run = s,
                    "rescore: shadow_q1 is set, but this rescore has no psms_extracted for the \
                     run; its shadow test keeps the isolation-window term"
                );
                return Ok(None);
            };
            let report = format!("{path}.report.json");
            let q = std::path::Path::new(&report)
                .exists()
                .then(|| mumdia_io::json::read_json(&report).ok())
                .flatten()
                .map(|r: serde_json::Value| r["params"]["q1_fragments"].clone())
                .filter(|q| !q.is_null());
            let t = if std::path::Path::new(path).exists() {
                Some(TableFile::open(path)?)
            } else {
                None
            };
            let (Some(q), Some(t)) = (q, t.filter(|t| t.has_column("q1f_frag_class"))) else {
                tracing::warn!(
                    psms_extracted = %path,
                    "rescore: shadow_q1 is set, but this run's psms_extracted carries no \
                     per-fragment Q1 decisions (extract.q1_fragments); its shadow test keeps \
                     the isolation-window term"
                );
                return Ok(None);
            };
            let num = |v: &serde_json::Value| v.as_f64().filter(|x| x.is_finite() && *x > 0.0);
            let (Some(reach), Some(step), Some(tol)) = (
                num(&q["reach_mz"]),
                num(&q["step_mz"]),
                num(&q["tolerances"]["offset_steps"]),
            ) else {
                bail!("shadow: {report}: q1_fragments has no reach, step or offset tolerance");
            };
            let mut out: HashMap<(u32, i32), (Vec<f32>, Vec<f32>)> = HashMap::new();
            let want = &want[s];
            t.for_each_batch(
                Some(&[
                    "candidate_id",
                    "peak_rank",
                    "q1f_frag_class",
                    "q1f_frag_offset",
                ]),
                1 << 16,
                |b| {
                    let col = |n: &str| -> Result<&arrow::array::ArrayRef> {
                        Ok(b.column(b.schema().index_of(n)?))
                    };
                    let cid = col("candidate_id")?
                        .as_any()
                        .downcast_ref::<UInt32Array>()
                        .context("shadow: psms_extracted.candidate_id is not u32")?;
                    let rank = col("peak_rank")?
                        .as_any()
                        .downcast_ref::<arrow::array::Int32Array>()
                        .context("shadow: psms_extracted.peak_rank is not i32")?;
                    let cls =
                        mumdia_io::table::ListF32::of(col("q1f_frag_class")?, "q1f_frag_class")?;
                    let off =
                        mumdia_io::table::ListF32::of(col("q1f_frag_offset")?, "q1f_frag_offset")?;
                    for k in 0..b.num_rows() {
                        let key = (cid.value(k), rank.value(k));
                        if want.contains(&key) {
                            out.insert(
                                key,
                                (
                                    cls.row_slice(k, "q1f_frag_class")?.to_vec(),
                                    off.row_slice(k, "q1f_frag_offset")?.to_vec(),
                                ),
                            );
                        }
                    }
                    Ok(())
                },
            )?;
            Ok(Some(Q1Run {
                reach: 2.0 * reach,
                step,
                tol,
                rows: out,
            }))
        })
        .collect()
}

/// The apex 1/K0 of every row, joined from its run's `psms_extracted` on
/// `(candidate_id, peak_rank)`; `None` for a row without one. All `None` when the 1/K0 term
/// is off, and, with a warning, when the caller passed no `psms_extracted` for the runs.
fn apex_ims(
    rows: &ShadowRows,
    inputs: &ShadowInputs,
    prm: ShadowParams,
    n_sources: usize,
) -> Result<Vec<Option<f64>>> {
    let n = rows.cid.len();
    if prm.im_tol <= 0.0 {
        return Ok(vec![None; n]);
    }
    if inputs.psms_extracted.len() < n_sources {
        tracing::warn!(
            given = inputs.psms_extracted.len(),
            runs = n_sources,
            "rescore: shadow_im_tolerance is set, but this rescore has no psms_extracted for \
             every run; the shadow test runs without its 1/K0 term"
        );
        return Ok(vec![None; n]);
    }
    let by_source: Vec<std::collections::BTreeMap<(u32, i32), f64>> = inputs.psms_extracted
        [..n_sources]
        .iter()
        .map(|p| {
            // A grouped run that did not pool its bands has no run-level table; its rows
            // keep the m/z-and-RT test.
            if !std::path::Path::new(p).exists() {
                tracing::warn!(
                    psms_extracted = %p,
                    "rescore: no psms_extracted for the shadow test's 1/K0 term of this run"
                );
                return Ok(std::collections::BTreeMap::new());
            }
            crate::stages::extract::apex_im_by_peak(p)
        })
        .collect::<Result<_>>()?;
    Ok((0..n)
        .map(|i| {
            by_source[rows.source[i] as usize]
                .get(&(rows.cid[i], rows.peak_rank[i]))
                .copied()
        })
        .collect())
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

/// Library fragment m/z of every candidate in `need`, in the library's order (the order of
/// the per-fragment Q1 decision lists). The pools are sorted where they are built.
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

/// The Q1 test of row `i` (the module doc): its co-isolated lenders by precursor m/z, and
/// its shared fragments counted by where their signal sits in Q1.
fn q1_shadow(
    i: usize,
    rows: &ShadowRows,
    q: &Q1Run,
    lenders: &[usize],
    frags: &HashMap<u32, Vec<f64>>,
    im: &[Option<f64>],
    prm: ShadowParams,
) -> bool {
    let Some(own) = frags.get(&rows.cid[i]) else {
        return false;
    };
    let (pm, rt) = (rows.precursor_mz[i], rows.apex_rt[i]);
    let lo = lenders.partition_point(|&j| rows.precursor_mz[j] < pm - q.reach);
    // (fragment m/z, lender's m/z offset in steps), every co-isolated lender's fragments.
    let mut pool: Vec<(f64, f64)> = Vec::new();
    for &j in &lenders[lo..] {
        if rows.precursor_mz[j] > pm + q.reach {
            break;
        }
        if (rows.apex_rt[j] - rt).abs() > prm.rt_s
            || rows.cid[j] == rows.cid[i]
            || rows.base[j] == rows.base[i]
            || rows.score[j] <= rows.score[i]
        {
            continue;
        }
        if let (Some(a), Some(b)) = (im[i], im[j]) {
            if (a - b).abs() > prm.im_tol {
                continue;
            }
        }
        if let Some(f) = frags.get(&rows.cid[j]) {
            let d = (rows.precursor_mz[j] - pm) / q.step;
            pool.extend(f.iter().map(|&m| (m, d)));
        }
    }
    if pool.is_empty() {
        return false;
    }
    pool.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let lists = q
        .rows
        .get(&(rows.cid[i], rows.peak_rank[i]))
        .filter(|(c, o)| c.len() == own.len() && o.len() == own.len());
    let mut total = 0.0f64;
    for (k, &f) in own.iter().enumerate() {
        let tol = f * prm.ppm * 1e-6;
        let a = pool.partition_point(|e| e.0 < f - tol);
        let mut best: Option<f64> = None;
        for e in &pool[a..] {
            if e.0 > f + tol {
                break;
            }
            let b = match lists {
                Some((c, o)) => q1_borrowed(c[k], o[k], e.1, q.tol),
                None => 1.0,
            };
            best = Some(best.map_or(b, |x: f64| x.max(b)));
        }
        if let Some(b) = best {
            total += b;
        }
    }
    total >= prm.min_shared as f64
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
    let im = apex_ims(rows, inputs, prm, n_sources)?;

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
                    im[i],
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

    // The Q1 term (`rescore.shadow_q1`): per run its reach, step, tolerance and the examined
    // rows' decisions, and its lenders ascending by precursor m/z.
    let q1: Vec<Option<Q1Run>> = if prm.q1 {
        q1_runs(rows, inputs, n_sources, &in_region)?
    } else {
        (0..n_sources).map(|_| None).collect()
    };
    let mut q1_lenders: Vec<Vec<usize>> = vec![Vec::new(); n_sources];
    if prm.q1 {
        for i in 0..n {
            if rows.accepted[i] {
                q1_lenders[rows.source[i] as usize].push(i);
            }
        }
        for v in q1_lenders.iter_mut() {
            v.sort_by(|&a, &b| {
                rows.precursor_mz[a]
                    .total_cmp(&rows.precursor_mz[b])
                    .then(a.cmp(&b))
            });
        }
    }

    let demote: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !in_region[i] {
                return false;
            }
            if let Some(q) = &q1[rows.source[i] as usize] {
                return q1_shadow(
                    i,
                    rows,
                    q,
                    &q1_lenders[rows.source[i] as usize],
                    &frags,
                    &im,
                    prm,
                );
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
                // The 1/K0 term: a lender separated in mobility shares no fragment peak.
                if let (Some(a), Some(b)) = (im[i], c.4) {
                    if (a - b).abs() > prm.im_tol {
                        continue;
                    }
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
    fn a_shared_fragment_counts_by_where_its_q1_signal_sits() {
        // Absent: nothing borrowed. Centred on the candidate: nothing; on the lender (3 steps
        // away): all of it; matching neither: the split. Closer than the tolerance, or no
        // profile: all.
        assert_eq!(q1_borrowed(0.0, 3.0, 3.0, 0.8), 0.0);
        assert_eq!(q1_borrowed(1.0, 0.0, 3.0, 0.8), 0.0);
        assert_eq!(q1_borrowed(2.0, 3.0, 3.0, 0.8), 1.0);
        assert!((q1_borrowed(2.0, 1.5, 3.0, 0.8) - 0.5).abs() < 1e-12);
        assert_eq!(q1_borrowed(2.0, -1.0, 3.0, 0.8), 0.0);
        assert_eq!(q1_borrowed(2.0, 0.3, -3.0, 0.8), 0.0);
        assert_eq!(q1_borrowed(2.0, 2.4, 3.0, 0.8), 1.0);
        assert_eq!(q1_borrowed(2.0, 0.6, 3.0, 0.8), 0.0);
        assert_eq!(q1_borrowed(1.0, 0.0, 0.5, 0.8), 1.0);
        assert_eq!(q1_borrowed(1.0, f32::NAN, 3.0, 0.8), 1.0);
    }

    /// The Q1 term on overlapping windows: the lender 2.3 m/z (3 steps) away co-isolates by
    /// m/z although the two precursors' windows differ; a candidate whose three shared
    /// fragments sit at the lender's Q1 centre is demoted, one whose shared fragments sit at
    /// its own is not, and without the decisions the run keeps the window test.
    #[test]
    fn the_q1_term_demotes_by_where_the_shared_signal_sits() {
        use mumdia_io::table::{write_table, Col};
        let step = 0.7614;
        // Overlapping ZT Scan windows, one per step.
        let n_w = 40;
        let t: Vec<f64> = (0..n_w).map(|k| 400.0 + step * k as f64).collect();
        let win = scratch("q1_iw.parquet");
        write_table(
            &win,
            vec![
                Col::U32("window_id".into(), (0..n_w as u32).collect()),
                Col::F64("target".into(), t.clone()),
                Col::F64("lower".into(), t.iter().map(|x| x - 1.9).collect()),
                Col::F64("upper".into(), t.iter().map(|x| x + 1.9).collect()),
            ],
        )
        .unwrap();
        // cid 1 the lender at 410; cid 2 and 3 at 412.3, each sharing 3 of the lender's 4.
        let lender = [200.0, 300.0, 400.0, 500.0];
        let borrower = [200.001, 300.001, 400.001, 650.0];
        let (mut fc, mut fm) = (Vec::new(), Vec::new());
        for c in 1u32..=3 {
            for &m in if c == 1 { &lender } else { &borrower } {
                fc.push(c);
                fm.push(m);
            }
        }
        let frag = scratch("q1_frag.parquet");
        write_table(
            &frag,
            vec![
                Col::U32("candidate_id".into(), fc),
                Col::F64("mz".into(), fm),
            ],
        )
        .unwrap();
        // The decisions: cid 2's shared fragments at the lender (+3 steps), cid 3's at its own.
        let psms = scratch("q1_psms_extracted.parquet");
        write_table(
            &psms,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2, 3]),
                Col::I32("peak_rank".into(), vec![0; 3]),
                Col::ListF32(
                    "q1f_frag_class".into(),
                    vec![vec![1.0; 4], vec![2.0, 2.0, 2.0, 1.0], vec![1.0; 4]],
                ),
                Col::ListF32(
                    "q1f_frag_offset".into(),
                    vec![vec![0.0; 4], vec![3.0, 3.0, 3.0, 0.0], vec![0.1; 4]],
                ),
            ],
        )
        .unwrap();
        mumdia_io::json::write_json(
            &format!("{psms}.report.json"),
            &serde_json::json!({"params": {"q1_fragments": {
                "reach_mz": 4.57, "step_mz": step, "tolerances": {"offset_steps": 0.8}
            }}}),
        )
        .unwrap();
        let source = vec![0u32; 3];
        let cid = vec![1u32, 2, 3];
        let base = vec![10u32, 20, 30];
        let apex = vec![100.0, 100.5, 100.5];
        let mz = vec![412.3, 410.0, 410.0];
        let score = vec![0.99, 0.9, 0.9];
        let is_decoy = vec![false, true, false];
        let accepted = vec![true, false, false];
        let peak_rank = vec![0i32; 3];
        let rows = ShadowRows {
            source: &source,
            cid: &cid,
            base: &base,
            apex_rt: &apex,
            precursor_mz: &mz,
            score: &score,
            is_decoy: &is_decoy,
            accepted: &accepted,
            peak_rank: &peak_rank,
        };
        let windows = vec![win];
        let tables = vec![psms];
        let prm = |q1: bool| ShadowParams {
            min_shared: 3,
            ppm: 10.0,
            rt_s: 3.0,
            region: 8,
            im_tol: 0.0,
            q1,
        };
        let inputs = ShadowInputs::new(&frag, &windows).with_psms_extracted(&tables);
        assert_eq!(
            flag(&rows, &inputs, prm(true)).unwrap(),
            vec![false, true, false]
        );
        // The window test: the precursors 2.3 m/z apart are in different windows of the
        // sweep, so nothing is demoted.
        assert_eq!(flag(&rows, &inputs, prm(false)).unwrap(), vec![false; 3]);
        // Without the decisions the run keeps the window test.
        let bare = ShadowInputs::new(&frag, &windows);
        assert_eq!(flag(&rows, &bare, prm(true)).unwrap(), vec![false; 3]);
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
        let peak_rank = vec![0i32; 6];
        let windows = vec![win];
        let rows = ShadowRows {
            source: &source,
            cid: &cid,
            base: &base,
            apex_rt: &apex,
            precursor_mz: &mz,
            score: &score,
            is_decoy: &is_decoy,
            accepted: &accepted,
            peak_rank: &peak_rank,
        };
        let prm = |im_tol: f64| ShadowParams {
            min_shared: 3,
            ppm: 10.0,
            rt_s: 3.0,
            region: 8,
            im_tol,
            q1: false,
        };
        let d = flag(&rows, &ShadowInputs::new(&frag, &windows), prm(0.0)).unwrap();
        assert_eq!(d, vec![false, true, false, false, false, false]);

        // The 1/K0 term. The lender (cid 1) and the borrower (cid 2) are 0.05 apart in
        // apex 1/K0: within 0.1 the borrower is still demoted, within 0.02 it is not.
        let psms = scratch("psms_extracted.parquet");
        write_table(
            &psms,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2, 3, 4, 5, 6]),
                Col::I32("peak_rank".into(), vec![0; 6]),
                Col::OptF64(
                    "apex_im".into(),
                    vec![
                        Some(0.90),
                        Some(0.95),
                        Some(0.9),
                        None,
                        Some(0.9),
                        Some(0.9),
                    ],
                ),
            ],
        )
        .unwrap();
        let tables = vec![psms];
        let with_im = ShadowInputs::new(&frag, &windows).with_psms_extracted(&tables);
        assert_eq!(flag(&rows, &with_im, prm(0.1)).unwrap(), d);
        assert_eq!(
            flag(&rows, &with_im, prm(0.02)).unwrap(),
            vec![false; 6],
            "a lender separated in 1/K0 lends nothing"
        );
        // The term off reads no table: the same flags with or without one.
        assert_eq!(flag(&rows, &with_im, prm(0.0)).unwrap(), d);
        // A 3D table (no apex_im values) keeps the m/z-and-RT test.
        let psms3 = scratch("psms_extracted_3d.parquet");
        write_table(
            &psms3,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2]),
                Col::I32("peak_rank".into(), vec![0; 2]),
                Col::OptF64("apex_im".into(), vec![None; 2]),
            ],
        )
        .unwrap();
        let tables3 = vec![psms3];
        let flat = ShadowInputs::new(&frag, &windows).with_psms_extracted(&tables3);
        assert_eq!(flag(&rows, &flat, prm(0.02)).unwrap(), d);
        // Without the tables the term is skipped with a warning.
        assert_eq!(
            flag(&rows, &ShadowInputs::new(&frag, &windows), prm(0.02)).unwrap(),
            d
        );
    }
}
