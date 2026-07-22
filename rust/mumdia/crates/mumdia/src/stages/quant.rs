//! Stage G `mumdia quant` (PLAN.md Stage G): quantify identified peptidoforms
//! and roll up to protein groups. Integrate each fragment chromatogram over the
//! apex region by the trapezoidal rule, sum the top-N fragments into a per-run
//! peptidoform quantity, then roll up to protein groups. MVP is single-run, so
//! cross-run normalization and MaxLFQ/directLFQ (which need multiple runs)
//! reduce to a top-N sum; the method is a config strategy for later.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{NormalizeMethod, PeakWindowMode, QuantConfig, QuantQColumn, RollupMethod};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, Table};
use serde_json::json;
use tracing::info;

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

/// Trapezoidal integral of one fragment trace restricted to RT in `[lo, hi]`.
/// Reuses [`trapezoid`] on the in-window samples so the single-sample rule is
/// identical; an empty window integrates to 0.
fn trapezoid_window(rt: &[f32], inten: &[f32], lo: f64, hi: f64) -> f64 {
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
    trapezoid(&wr, &wi)
}

/// Elution-peak RT window `[lo, hi]` for one candidate, from the summed XIC across
/// all its fragment chromatograms. Fragments are aligned on the union of their RT
/// samples via a BTreeMap keyed by the f32 RT bit pattern: for the non-negative
/// RTs here the bit order matches the value order, so both the union axis and the
/// f64 summation order are fixed (determinism, plan.md Section 7). The apex is the
/// first RT of maximum summed intensity; [`super::features::peak_bounds`] then
/// walks out with the given `frac`/`grace`. Returns an unbounded window when there
/// are fewer than two distinct RT samples (nothing to bound).
fn peak_window(
    rows: &[usize],
    ch_rt: &[Vec<f32>],
    ch_int: &[Vec<f32>],
    frac: f64,
    grace: usize,
) -> (f64, f64, f64) {
    let mut prof_map: BTreeMap<u32, f64> = BTreeMap::new();
    // Per-scan count of co-eluting (nonzero) fragments, aligned to prof_map keys.
    let mut cnt_map: BTreeMap<u32, u32> = BTreeMap::new();
    for &i in rows {
        let rts = &ch_rt[i];
        let ins = &ch_int[i];
        for k in 0..rts.len() {
            *prof_map.entry(rts[k].to_bits()).or_insert(0.0) += ins[k] as f64;
            if ins[k] > 0.0 {
                *cnt_map.entry(rts[k].to_bits()).or_insert(0) += 1;
            }
        }
    }
    if prof_map.len() < 2 {
        // Nothing to bound; apex is the lone RT if present, else NaN.
        let apex = prof_map.keys().next().map_or(f64::NAN, |b| f32::from_bits(*b) as f64);
        return (f64::NEG_INFINITY, f64::INFINITY, apex);
    }
    let axis: Vec<f64> = prof_map.keys().map(|b| f32::from_bits(*b) as f64).collect();
    let prof: Vec<f64> = prof_map.values().cloned().collect();
    let cnt: Vec<u32> = prof_map.keys().map(|b| *cnt_map.get(b).unwrap_or(&0)).collect();
    // Robust apex: among scans whose co-eluting-fragment count is within 1 of the
    // maximum ("-1 for robustness"), take the one with the highest summed intensity.
    // This rejects a lone tall interferent fragment (which a plain summed-intensity
    // argmax would pick in a low/absent run) in favor of a region where many
    // fragments co-elute. Falls back to summed argmax only if no scan has a fragment.
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
        if lo > 0 {
            lo -= 1;
        }
    }
    (axis[lo], axis[hi], axis[ai])
}

pub fn run(p: QuantParams) -> Result<(u64, u64)> {
    let t0 = Instant::now();

    // Identified target PSMs below the peptide q threshold.
    let ps = Table::read(p.psms_scored)?;
    let cid = ps.u32("candidate_id")?;
    let pform = ps.str("peptidoform")?;
    let charge = ps.i32("charge")?;
    let label = ps.str("label")?;
    let pg = ps.str("protein_group")?;
    // q-value column to filter on. Peptide q is per-run in a single-run rescore, but
    // GLOBAL (best PSM per peptide across all runs) under an experiment-wide rescore,
    // where the per-PSM q_value is the correct per-run choice for cross-run quant.
    let pep_q = match p.cfg.q_filter {
        QuantQColumn::PeptideQ => ps.f64("peptide_q_value")?,
        QuantQColumn::PsmQ => ps.f64("q_value")?,
    };

    // Chromatograms grouped by candidate.
    let ch = Table::read(p.chromatograms)?;
    let ch_cid = ch.u32("candidate_id")?;
    let ch_name = ch.str("frag_name")?;
    let ch_rt = ch.list_f32("rt")?;
    let ch_int = ch.list_f32("intensity")?;
    // Group the b/y fragment chromatogram rows by candidate. The MS1 isotope XIC
    // pseudo-traces (frag_name "ms1_*") are precursor channels, not fragment ions,
    // and are excluded from both the peak-window detection and the top-N sum. The
    // BTreeMap keeps candidate iteration order deterministic.
    let mut cand_rows: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for i in 0..ch.nrows {
        if ch_name[i].starts_with("ms1_") {
            continue;
        }
        cand_rows.entry(ch_cid[i]).or_default().push(i);
    }
    let mut areas: HashMap<u32, Vec<f64>> = HashMap::new();
    // Store the fragment name by reference (borrowed from `ch_name`, which outlives
    // this map) to avoid a per-row String clone; it is materialized once at export.
    let mut frag_areas: HashMap<u32, Vec<(&str, f64)>> = HashMap::new();
    // Optional peak-window diagnostic: (candidate_id, lo_rt, hi_rt) for finite windows.
    let emit_bounds = p.out_peak_bounds.is_some() && p.cfg.bound_peak;
    let (mut pb_cid, mut pb_lo, mut pb_hi) = (Vec::new(), Vec::new(), Vec::new());

    // Phase 1: per-candidate summed-XIC window (lo_rt, hi_rt, apex_rt). The window
    // comes from the summed XIC across all fragments, so a lone off-apex interferent
    // cannot define the bound. Kept keyed by candidate for the consensus estimate.
    let mut win: BTreeMap<u32, (f64, f64, f64)> = BTreeMap::new();
    if p.cfg.bound_peak {
        for (&c, rows) in &cand_rows {
            win.insert(
                c,
                peak_window(rows, &ch_rt, &ch_int, p.cfg.peak_fraction, p.cfg.peak_grace),
            );
        }
    }

    // Consensus mode: peak width is a near-constant instrument/gradient property, so
    // take the median left/right half-width over CONFIDENT peptides (q <= reliable_q)
    // and apply it around each candidate's apex. The median ignores the interference-
    // stretched and collapsed per-candidate windows, and being global it is identical
    // across runs, preserving fold changes. Falls back to per-candidate if too few
    // confident anchors for a stable median.
    let consensus: Option<(f64, f64)> =
        if p.cfg.bound_peak && p.cfg.peak_window_mode == PeakWindowMode::Consensus {
            let mut q_by_cid: HashMap<u32, f64> = HashMap::new();
            for i in 0..ps.nrows {
                if label[i] == "target" {
                    let e = q_by_cid.entry(cid[i]).or_insert(f64::INFINITY);
                    if pep_q[i] < *e {
                        *e = pep_q[i];
                    }
                }
            }
            let (mut left, mut right) = (Vec::new(), Vec::new());
            for (c, &(lo, hi, apex)) in &win {
                if lo.is_finite()
                    && hi.is_finite()
                    && apex.is_finite()
                    && q_by_cid.get(c).is_some_and(|&q| q <= p.cfg.reliable_q)
                {
                    left.push(apex - lo);
                    right.push(hi - apex);
                }
            }
            if left.len() >= 20 {
                let ml = median_sorted(&mut left);
                let mr = median_sorted(&mut right);
                info!(anchors = left.len(), med_left_s = ml, med_right_s = mr, "quant: consensus peak window");
                Some((ml, mr))
            } else {
                info!(anchors = left.len(), "quant: too few confident anchors, using per-candidate windows");
                None
            }
        } else {
            None
        };

    // Phase 2: integrate each fragment over the chosen window.
    for (&c, rows) in &cand_rows {
        let (lo_rt, hi_rt) = if !p.cfg.bound_peak {
            (f64::NEG_INFINITY, f64::INFINITY)
        } else {
            let (lo, hi, apex) = win[&c];
            match consensus {
                Some((ml, mr)) if apex.is_finite() => (apex - ml, apex + mr),
                _ => (lo, hi),
            }
        };
        if emit_bounds && lo_rt.is_finite() && hi_rt.is_finite() {
            pb_cid.push(c);
            pb_lo.push(lo_rt);
            pb_hi.push(hi_rt);
        }
        for &i in rows {
            let a = if p.cfg.bound_peak {
                trapezoid_window(&ch_rt[i], &ch_int[i], lo_rt, hi_rt)
            } else {
                trapezoid(&ch_rt[i], &ch_int[i])
            };
            areas.entry(c).or_default().push(a);
            frag_areas.entry(c).or_default().push((ch_name[i].as_str(), a));
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

    // Per-peptidoform quantity = sum of the top-N fragment areas.
    let (mut q_cid, mut q_pform, mut q_z, mut q_pg, mut q_val, mut q_nfrag) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut per_group: HashMap<String, Vec<f64>> = HashMap::new();
    for i in 0..ps.nrows {
        if label[i] == "decoy" || pep_q[i] > p.cfg.q_threshold {
            continue;
        }
        // Sort the per-candidate area vector in place (by mutable reference) rather
        // than cloning it per PSM; `areas` is not read again after this loop and a
        // repeated candidate_id re-sorts an already-sorted vector (idempotent).
        let (quantity, used): (f64, usize) = match areas.get_mut(&cid[i]) {
            Some(a) => {
                a.sort_by(|x, y| y.partial_cmp(x).unwrap());
                let used = a.len().min(p.cfg.top_n_fragments);
                (a.iter().take(used).sum(), used)
            }
            None => (0.0, 0),
        };
        q_cid.push(cid[i]);
        q_pform.push(pform[i].clone());
        q_z.push(charge[i]);
        q_pg.push(pg[i].clone());
        q_val.push(quantity);
        q_nfrag.push(used as i32);
        per_group.entry(pg[i].clone()).or_default().push(quantity);
    }

    let n_pep = write_table(
        p.out_peptide,
        vec![
            Col::U32("candidate_id".into(), q_cid),
            Col::Str("peptidoform".into(), q_pform),
            Col::I32("charge".into(), q_z),
            Col::Str("protein_group".into(), q_pg),
            Col::F64("quantity".into(), q_val),
            Col::I32("n_fragments_used".into(), q_nfrag),
        ],
    )?;

    // Protein-group rollup.
    let mut groups: Vec<String> = per_group.keys().cloned().collect();
    groups.sort();
    let (mut g_name, mut g_val, mut g_npep) = (Vec::new(), Vec::new(), Vec::new());
    for g in &groups {
        let mut v = per_group[g].clone();
        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let quantity = match p.cfg.rollup {
            RollupMethod::TopNSum => v.iter().take(p.cfg.top_n_peptides).sum(),
            RollupMethod::Sum => v.iter().sum(),
        };
        g_name.push(g.clone());
        g_val.push(quantity);
        g_npep.push(v.len() as i32);
    }
    let n_pg = write_table(
        p.out_protein,
        vec![
            Col::Str("protein_group".into(), g_name),
            Col::F64("quantity".into(), g_val),
            Col::I32("n_peptides".into(), g_npep),
        ],
    )?;

    // Optional per-fragment area export for ion-level directLFQ across runs.
    if let Some(fpath) = p.out_fragment {
        let (mut f_cid, mut f_pf, mut f_z, mut f_pg, mut f_name, mut f_area) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..ps.nrows {
            if label[i] == "decoy" || pep_q[i] > p.cfg.q_threshold {
                continue;
            }
            if let Some(fa) = frag_areas.get(&cid[i]) {
                for (nm, a) in fa {
                    f_cid.push(cid[i]);
                    f_pf.push(pform[i].clone());
                    f_z.push(charge[i]);
                    f_pg.push(pg[i].clone());
                    f_name.push(nm.to_string());
                    f_area.push(*a);
                }
            }
        }
        write_table(
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
    }

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("quantified_peptides".to_string(), json!(n_pep));
    stats.insert("quantified_protein_groups".to_string(), json!(n_pg));
    for (path, schema, rows) in [
        (p.out_peptide, artifact::PEPTIDE_QUANT, n_pep),
        (p.out_protein, artifact::PROTEIN_GROUP_QUANT, n_pg),
    ] {
        ArtifactReport {
            logical_name: schema.0.to_string(),
            schema_name: schema.0.to_string(),
            schema_version: schema.1,
            stage: "quant".to_string(),
            rows,
            content_hash: mumdia_io::hash::blake3_file(path)?,
            params: json!({"q_threshold": p.cfg.q_threshold, "top_n_fragments": p.cfg.top_n_fragments,
                           "rollup": format!("{:?}", p.cfg.rollup), "bound_peak": p.cfg.bound_peak,
                           "peak_fraction": p.cfg.peak_fraction, "peak_grace": p.cfg.peak_grace,
                           "q_filter": format!("{:?}", p.cfg.q_filter)}),
            stats: stats.clone(),
            model_identity: None,
            elapsed_ms: elapsed,
        }
        .write_for(path)?;
    }

    info!(peptides = n_pep, protein_groups = n_pg, elapsed_ms = elapsed, "quant: done");
    Ok((n_pep, n_pg))
}

/// Combine several per-run quant tables into a protein-by-run abundance matrix
/// with MaxLFQ (peptide-level: `by_fragment=false`, reads `peptide_quant`) or
/// directLFQ (ion-level: `by_fragment=true`, reads `fragment_quant`). Each
/// protein's feature-by-run intensity matrix is passed to the ratio-alignment
/// core in [`crate::quant_lfq`]. Output is long form: protein_group, run,
/// quantity, n_features. With one input this reduces to the per-run sum.
///
/// `normalize` applies a cross-run size factor to the feature-by-run matrix
/// before rollup (see [`size_factors`]).
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
        let t = Table::read(path)?;
        let pform = t.str("peptidoform")?;
        let z = t.i32("charge")?;
        let pgc = t.str("protein_group")?;
        let q = t.f64("quantity")?;
        let fname = if by_fragment { Some(t.str("fragment_name")?) } else { None };
        for i in 0..t.nrows {
            let key = match &fname {
                Some(fnm) => format!("{}|{}|{}", pform[i], z[i], fnm[i]),
                None => format!("{}|{}", pform[i], z[i]),
            };
            data.entry(pgc[i].clone())
                .or_default()
                .entry(key)
                .or_insert_with(|| vec![None; n])[ri] = Some(q[i]);
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
    let (mut c_pg, mut c_run, mut c_q, mut c_nf) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (pgname, feats) in &data {
        let mat: Vec<Vec<Option<f64>>> = feats.values().cloned().collect();
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
    info!(
        proteins = data.len(),
        runs = n,
        method = if by_fragment { "directlfq" } else { "maxlfq" },
        normalize = ?normalize,
        size_factors = ?factors,
        "quant-lfq: done"
    );
    Ok(rows)
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
/// key order, so the result is deterministic (plan.md Section 7).
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
                    if vec.iter().all(|x| x.map_or(false, |v| v > 0.0)) {
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
                .map(|r| if logs[r].is_empty() { 0.0 } else { median_sorted(&mut logs[r]) })
                .collect();
            let mut m2 = med.clone();
            let target = if m2.is_empty() { 0.0 } else { median_sorted(&mut m2) };
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
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
        assert!((trapezoid_window(&rt, &it, 1.0, 3.0) - 15.0).abs() < 1e-9);
        // Window with a single in-range sample returns that raw intensity.
        assert_eq!(trapezoid_window(&rt, &it, 2.0, 2.0), 10.0);
        // Empty window integrates to 0.
        assert_eq!(trapezoid_window(&rt, &it, 10.0, 20.0), 0.0);
    }

    #[test]
    fn peak_window_bounds_summed_xic_and_rejects_lone_interferent() {
        // Two fragments share an RT grid. Fragment 0 is the real co-eluting peptide
        // peaking at rt=6; fragment 1 is a lone interferent spiking at rt=1, well
        // separated (>= 2 zero scans) so the grace walk cannot bridge to it. The
        // SUMMED XIC apex lands on rt=6 and the window brackets the real peak only,
        // even though the interferent is tall.
        let grid: Vec<f32> = (0..12).map(|k| k as f32).collect();
        let real = vec![0.0f32, 0.0, 0.0, 0.0, 2.0, 6.0, 10.0, 6.0, 2.0, 0.0, 0.0, 0.0];
        let interf = vec![0.0f32, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ch_rt = vec![grid.clone(), grid.clone()];
        let ch_int = vec![real, interf];
        let (lo, hi, _) = peak_window(&[0, 1], &ch_rt, &ch_int, 1.0 / 6.0, 1);
        // Apex rt=6 (sum=10); 1/6 threshold ~1.67. Left: idx4(2)>=thr, idx3/idx2=0
        // -> 2 consecutive misses stop at rt=4. Right: symmetric stop at rt=8.
        assert_eq!(lo, 4.0);
        assert_eq!(hi, 8.0);
        // The interferent spike at rt=1 is outside [lo,hi], so its windowed area is 0.
        assert_eq!(trapezoid_window(&ch_rt[1], &ch_int[1], lo, hi), 0.0);
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
        let (_, hi1, _) = peak_window(&[0], &ch_rt, &ch_int, 1.0 / 3.0, 1);
        let (_, hi0, _) = peak_window(&[0], &ch_rt, &ch_int, 1.0 / 3.0, 0);
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
        assert!((f[1] / f[0] - 2.0).abs() < 0.02, "expected ~2x scale, got {f:?}");
        // Bulk normalizes to ratio 1; the up/down real changes are preserved.
        let bulk = (100.0 / f[0], 200.0 / f[1]);
        assert!((bulk.0 / bulk.1 - 1.0).abs() < 1e-9, "bulk should flatten to 1");
        let up = (100.0 / f[0]) / (800.0 / f[1]);
        assert!((up - 0.25).abs() < 1e-9, "up feature run0/run1 should stay 1:4");
    }

    #[test]
    fn none_leaves_matrix_unnormalized() {
        use std::collections::BTreeMap;
        let mut feats: BTreeMap<String, Vec<Option<f64>>> = BTreeMap::new();
        feats.insert("f".into(), vec![Some(10.0), Some(40.0)]);
        let mut data: BTreeMap<String, BTreeMap<String, Vec<Option<f64>>>> = BTreeMap::new();
        data.insert("PG".into(), feats);
        assert_eq!(size_factors(&data, 2, NormalizeMethod::None), vec![1.0, 1.0]);
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
        let ch_rt = vec![grid.clone(), grid.clone(), grid.clone(), grid.clone(), grid.clone()];
        let ch_int = vec![real.clone(), real.clone(), real.clone(), real.clone(), interf];
        let (_, _, apex) = peak_window(&[0, 1, 2, 3, 4], &ch_rt, &ch_int, 1.0 / 6.0, 1);
        assert_eq!(apex, 5.0, "apex must be the 4-fragment co-elution scan, not the lone interferent spike");
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
        let (lo, hi, apex) = peak_window(&[0], &ch_rt, &ch_int, 1.0 / 6.0, 1);
        assert_eq!(apex, 4.0, "apex should be the summed-XIC max rt");
        assert!(hi > lo, "window must have nonzero width: lo={lo} hi={hi}");
        let a = trapezoid_window(&ch_rt[0], &ch_int[0], lo, hi);
        assert!((a - 20.0).abs() < 1e-9, "expected triangle area 20, not height 10, got {a}");
    }
}
