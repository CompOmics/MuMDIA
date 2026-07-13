//! Stage S `mumdia search-seed`: a native broad DIA-aware seed search over the
//! inverted fragment index (PLAN.md Stage S). Its purpose is calibration, not
//! final identification. It sits behind the file contract, so a Sage adapter can
//! replace it later (the plan's default); MVP uses a native Sage-lite hyperscore.
//!
//! Library-level decoys are the single source of truth (no separate engine
//! decoy generation), so target-decoy counting is never mixed-method.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{MatcherKind, SearchSeedConfig};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col};
use serde_json::json;
use tracing::info;

use crate::fdr::{count_targets_at_q, ln_factorial, target_decoy_q};
use crate::matchers::fragindex::{FragIndex, SeedScratch};
use mumdia_core::types::Ms2Scan;
use rayon::prelude::*;
use crate::index::Library;
use crate::spectra::load_ms2;

pub struct SearchSeedParams<'a> {
    pub ms2: &'a str,
    pub library_precursors: &'a str,
    pub library_fragments: &'a str,
    pub out: &'a str,
    pub cfg: &'a SearchSeedConfig,
    pub bucket_size: usize,
    pub config_hash: &'a str,
}

#[derive(Clone)]
struct Best {
    score: f64,
    rt: f64,
    matched: u32,
    scan_index: u32,
}

pub fn run(p: SearchSeedParams) -> Result<u64> {
    let t0 = Instant::now();
    let lib = Library::load(p.library_precursors, p.library_fragments, p.bucket_size)?;
    let scans = load_ms2(p.ms2)?;
    info!(candidates = lib.n_candidates(), scans = scans.len(), "search-seed: loaded");

    // fragindex backend, built once at the seed's fragment tolerance when selected.
    let fidx = matches!(p.cfg.matcher, MatcherKind::Fragindex)
        .then(|| FragIndex::build(&lib, p.cfg.fragment_tol_ppm));

    // Best-per-candidate PSM. The fragindex path parallelizes across isolation-window
    // groups (each scan belongs to exactly one window, so groups are independent) and
    // is bit-identical to the serial path via a deterministic per-candidate merge; the
    // bucketed path stays serial.
    let best: HashMap<u32, Best> = if let Some(idx) = fidx.as_ref() {
        seed_fragindex_windows(idx, &scans, p.cfg)
    } else {
        let mut best: HashMap<u32, Best> = HashMap::new();
        for scan in &scans {
            let (lo, hi) = lib.candidate_range(scan.window.lower_mz, scan.window.upper_mz);
            if hi <= lo {
                continue;
            }
            let peak_idx = select_peaks(scan, p.cfg.top_n_peaks);
            let mut acc: HashMap<u32, (u32, f64)> = HashMap::new();
            for &pidx in &peak_idx {
                let peak = &scan.peaks[pidx];
                let inten = peak.intensity as f64;
                lib.page_search(peak.mz, p.cfg.fragment_tol_ppm, lo, hi, |cid, _mz, _pi| {
                    let e = acc.entry(cid).or_insert((0, 0.0));
                    e.0 += 1;
                    e.1 += inten;
                });
            }
            let mut scored: Vec<(u32, f64, u32)> = acc
                .into_iter()
                .filter(|(_, v)| v.0 as usize >= p.cfg.min_matched_peaks)
                .map(|(cid, v)| (cid, hyperscore(v.0, v.1), v.0))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            scored.truncate(p.cfg.report_psms);
            for (cid, score, matched) in scored {
                let entry = best.entry(cid).or_insert(Best {
                    score: f64::NEG_INFINITY,
                    rt: 0.0,
                    matched: 0,
                    scan_index: 0,
                });
                if score > entry.score {
                    *entry = Best { score, rt: scan.rt_seconds, matched, scan_index: scan.scan_index };
                }
            }
        }
        best
    };

    // Assemble PSM rows (best per candidate) and compute q-values.
    let mut rows: Vec<(u32, &Best)> = best.iter().map(|(k, v)| (*k, v)).collect();
    rows.sort_by_key(|(cid, _)| *cid);
    let sd: Vec<(f64, bool)> = rows
        .iter()
        .map(|(cid, b)| (b.score, lib.cands[*cid as usize].is_decoy))
        .collect();
    let q = target_decoy_q(&sd);

    let (mut cid_c, mut pform_c, mut charge_c, mut mz_c) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut base_c, mut prot_c, mut label_c) = (Vec::new(), Vec::new(), Vec::new());
    let (mut score_c, mut q_c, mut rt_c, mut matched_c, mut scan_c, mut irt_c) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for (i, (cid, b)) in rows.iter().enumerate() {
        let c = &lib.cands[*cid as usize];
        cid_c.push(*cid);
        pform_c.push(c.peptidoform.clone());
        charge_c.push(c.charge);
        mz_c.push(c.precursor_mz);
        base_c.push(c.base_peptide_id);
        prot_c.push(c.protein.clone());
        label_c.push(if c.is_decoy { "decoy" } else { "target" }.to_string());
        score_c.push(b.score);
        q_c.push(q[i]);
        rt_c.push(b.rt);
        matched_c.push(b.matched as i32);
        scan_c.push(b.scan_index);
        irt_c.push(c.predicted_irt);
    }

    let is_dec: Vec<bool> = label_c.iter().map(|l| l == "decoy").collect();
    let n_at_1pct = count_targets_at_q(&q_c, &is_dec, p.cfg.fdr_seed);

    // Per-run fragment mass recalibration + learned tolerance (PLAN.md Section 5
    // improvement 3, Section 8.4). Collect matched-fragment ppm deviations from
    // confident target PSMs; the median is the systematic offset, a high
    // percentile of the centered deviations sets the tolerance. Written to
    // <seed>.masscal.json and consumed by extract.
    let mut scan_by_index: HashMap<u32, &mumdia_core::types::Ms2Scan> = HashMap::new();
    for s in &scans {
        scan_by_index.insert(s.scan_index, s);
    }
    let mut devs: Vec<f64> = Vec::new();
    for (i, (cid, b)) in rows.iter().enumerate() {
        if is_dec[i] || q[i] > p.cfg.fdr_seed {
            continue;
        }
        if let Some(scan) = scan_by_index.get(&b.scan_index) {
            let (mzs, _, _) = lib.cand_frags(*cid);
            for &fmz in mzs {
                let (lo, hi) = mumdia_core::constants::ppm_bounds(fmz, 50.0);
                let s = scan.peaks.partition_point(|pk| pk.mz < lo);
                let (mut bestd, mut bestppm) = (f64::MAX, None);
                let mut j = s;
                while j < scan.peaks.len() && scan.peaks[j].mz <= hi {
                    let d = (scan.peaks[j].mz - fmz).abs();
                    if d < bestd {
                        bestd = d;
                        bestppm = Some(mumdia_core::constants::ppm_diff(scan.peaks[j].mz, fmz));
                    }
                    j += 1;
                }
                if let Some(pp) = bestppm {
                    devs.push(pp);
                }
            }
        }
    }
    let (frag_ppm_offset, frag_tol_learned) = if devs.len() >= 20 {
        let mut sorted = devs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let offset = sorted[sorted.len() / 2];
        let centered: Vec<f64> = devs.iter().map(|d| (d - offset).abs()).collect();
        let tol = (crate::calibrate::percentile(&centered, 0.95) * 1.5).max(5.0);
        (offset, tol)
    } else {
        (0.0, p.cfg.fragment_tol_ppm)
    };
    mumdia_io::json::write_json(
        &format!("{}.masscal.json", p.out),
        &json!({
            "frag_ppm_offset": frag_ppm_offset,
            "frag_tol_ppm": frag_tol_learned,
            "n_dev": devs.len(),
        }),
    )?;
    info!(frag_ppm_offset, frag_tol_learned, "search-seed: mass recalibration");

    let n = write_table(
        p.out,
        vec![
            Col::U32("candidate_id".into(), cid_c),
            Col::Str("peptidoform".into(), pform_c),
            Col::I32("charge".into(), charge_c),
            Col::F64("precursor_mz".into(), mz_c),
            Col::U32("base_peptide_id".into(), base_c),
            Col::Str("protein".into(), prot_c),
            Col::Str("label".into(), label_c),
            Col::F64("score".into(), score_c),
            Col::F64("spectrum_q".into(), q_c),
            Col::F64("observed_rt".into(), rt_c),
            Col::F32("predicted_irt".into(), irt_c),
            Col::I32("matched_peaks".into(), matched_c),
            Col::U32("scan_index".into(), scan_c),
        ],
    )?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("psms".to_string(), json!(n));
    stats.insert(format!("targets_at_q{}", p.cfg.fdr_seed), json!(n_at_1pct));
    ArtifactReport {
        logical_name: artifact::SEED_PSMS.0.to_string(),
        schema_name: artifact::SEED_PSMS.0.to_string(),
        schema_version: artifact::SEED_PSMS.1,
        stage: "search-seed".to_string(),
        rows: n,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({
            "fragment_tol_ppm": p.cfg.fragment_tol_ppm,
            "report_psms": p.cfg.report_psms,
            "min_matched_peaks": p.cfg.min_matched_peaks,
            "fdr_seed": p.cfg.fdr_seed,
        }),
        stats,
        model_identity: Some("native-seed-hyperscore-v1".to_string()),
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;

    info!(
        psms = n,
        confident = n_at_1pct,
        elapsed_ms = elapsed,
        "search-seed: done"
    );
    Ok(n)
}

/// Peak indices to probe for a scan: the `top_n` most intense (index-ascending
/// re-sort keeps the obs_sum accumulation order deterministic), or all peaks when
/// `top_n == 0` or the scan is small.
fn select_peaks(scan: &Ms2Scan, top_n: usize) -> Vec<usize> {
    if top_n > 0 && scan.peaks.len() > top_n {
        let mut idx: Vec<usize> = (0..scan.peaks.len()).collect();
        idx.sort_by(|&a, &b| {
            scan.peaks[b]
                .intensity
                .partial_cmp(&scan.peaks[a].intensity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(top_n);
        idx.sort_unstable();
        idx
    } else {
        (0..scan.peaks.len()).collect()
    }
}

/// fragindex seed over isolation-window groups, in parallel. Each scan belongs to
/// exactly one isolation window, so grouping scans by window gives independent
/// parallel units (the candidate axis overlaps between adjacent windows, handled by
/// the deterministic per-candidate merge below). Bit-identical to the serial
/// best-per-candidate: within a group scans run RT-ascending with strictly-greater
/// update (earliest-RT max), and the cross-group merge keeps `max hyperscore, tie ->
/// earliest RT, tie -> min scan_index`, which equals the serial global earliest-RT max.
fn seed_fragindex_windows(
    idx: &FragIndex,
    scans: &[Ms2Scan],
    cfg: &SearchSeedConfig,
) -> HashMap<u32, Best> {
    use std::collections::BTreeMap;
    // Group scan indices by window; BTreeMap keys give a deterministic group order
    // (the merge is order-independent anyway, being a total-order max).
    let mut groups: BTreeMap<(u64, u64), Vec<usize>> = BTreeMap::new();
    for (si, scan) in scans.iter().enumerate() {
        groups
            .entry((scan.window.lower_mz.to_bits(), scan.window.upper_mz.to_bits()))
            .or_default()
            .push(si);
    }
    let group_vec: Vec<Vec<usize>> = groups.into_values().collect();
    let n_cand = idx.n_cand();

    let partials: Vec<Vec<(u32, Best)>> = group_vec
        .par_iter()
        .map_init(
            || SeedScratch::new(n_cand),
            |scratch, ids| {
                if ids.is_empty() {
                    return Vec::new();
                }
                let w = &scans[ids[0]].window;
                let (lo, hi) = idx.candidate_range(w.lower_mz, w.upper_mz);
                if hi <= lo {
                    return Vec::new();
                }
                let mut local: HashMap<u32, Best> = HashMap::new();
                for &si in ids {
                    let scan = &scans[si];
                    let peak_idx = select_peaks(scan, cfg.top_n_peaks);
                    let peaks: Vec<(f64, f32)> = peak_idx
                        .iter()
                        .map(|&pi| (scan.peaks[pi].mz, scan.peaks[pi].intensity))
                        .collect();
                    scratch.accumulate(idx, &peaks, lo, hi);
                    let touched: Vec<u32> = scratch.touched().to_vec();
                    let mut scored: Vec<(u32, f64, u32)> = touched
                        .iter()
                        .filter(|&&cid| scratch.count(cid) as usize >= cfg.min_matched_peaks)
                        .map(|&cid| {
                            (cid, hyperscore(scratch.count(cid), scratch.obs_sum(cid)), scratch.count(cid))
                        })
                        .collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
                    scored.truncate(cfg.report_psms);
                    for (cid, score, matched) in scored {
                        let e = local.entry(cid).or_insert(Best {
                            score: f64::NEG_INFINITY,
                            rt: 0.0,
                            matched: 0,
                            scan_index: 0,
                        });
                        if score > e.score {
                            *e = Best { score, rt: scan.rt_seconds, matched, scan_index: scan.scan_index };
                        }
                    }
                }
                local.into_iter().collect()
            },
        )
        .collect();

    // Deterministic cross-group merge (total order, so independent of group/thread order).
    let mut best: HashMap<u32, Best> = HashMap::new();
    for part in &partials {
        for (cid, b) in part {
            let better = match best.get(cid) {
                None => true,
                Some(e) => {
                    b.score > e.score
                        || (b.score == e.score && b.rt < e.rt)
                        || (b.score == e.score && b.rt == e.rt && b.scan_index < e.scan_index)
                }
            };
            if better {
                best.insert(*cid, b.clone());
            }
        }
    }
    best
}

/// Sage-style hyperscore: ln(matched!) + ln(1 + summed matched intensity).
fn hyperscore(matched: u32, sum_obs: f64) -> f64 {
    ln_factorial(matched) + (1.0 + sum_obs).ln()
}
