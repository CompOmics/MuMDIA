//! Stage S `mumdia search-seed`: a native broad DIA-aware seed search over the
//! inverted fragment index (docs/07_search_seed.md). Its purpose is calibration,
//! not final identification. It sits behind the file contract, so a Sage adapter
//! can replace it later; MVP uses a native Sage-lite hyperscore.
//!
//! Library-level decoys are the single source of truth (no separate engine
//! decoy generation), so target-decoy counting is never mixed-method.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use mumdia_core::config::{MatcherKind, SearchSeedConfig};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col};
use serde_json::json;
use tracing::{info, warn};

use crate::fdr::{count_targets_at_q, ln_factorial, target_decoy_q};
use crate::index::Library;
use crate::matchers::fragindex::{FragIndex, SeedScratch};
use crate::spectra::load_ms2;
use mumdia_core::types::Ms2Scan;
use rayon::prelude::*;

pub struct SearchSeedParams<'a> {
    pub ms2: &'a str,
    pub library_precursors: &'a str,
    pub library_fragments: &'a str,
    pub out: &'a str,
    pub cfg: &'a SearchSeedConfig,
    pub bucket_size: usize,
    pub config_hash: &'a str,
    /// The precursor table is one isolation-window group's band, written with local ids
    /// `0..n` (`groups.window_groups`), and its fragments sit in the library-wide fragment
    /// table at ids `offset..offset + n`. The seed table is then in band-local ids. `None`
    /// is the ordinary whole-library search.
    pub fragment_offset: Option<u32>,
    /// This run's MS2 scans, already decoded. A grouped search
    /// (`groups.window_groups > 1`) decodes the run once in `run_groups` and lends the
    /// same buffer to every band, because every band re-reads the whole run and only the
    /// library differs. `None` loads them from `ms2`, which is what a standalone
    /// `mumdia search-seed` and an ungrouped `run` do.
    ///
    /// Borrowed, never owned, and the stage only reads them: it takes a shared slice that
    /// it cannot write through, `load_ms2` has already sorted by retention time so it
    /// does not re-sort, `select_peaks` returns peak INDICES rather than truncating
    /// `scan.peaks`, and the mass recalibration this stage fits is written to
    /// `<out>.masscal.json` rather than applied to the peaks. So no band can leave a
    /// trace in the scans the next band sees.
    ///
    /// `select_peaks` is the one to watch. Truncating `scan.peaks` in place is the obvious
    /// way to stop rebuilding an index vector per scan per band, and it would hand the
    /// following band -- and extract, which shares the same buffer -- capped spectra. A
    /// 300-peak cap costs 60% of the peptides on a 50-window Orbitrap DIA run
    /// (docs/04_convert.md).
    ///
    /// An EMPTY slice does not mean "this run has no MS2". It means the caller has
    /// nothing to lend, and the stage decodes `ms2` itself.
    pub ms2_scans: Option<&'a [Ms2Scan]>,
    /// Also write `<out>.masscal.parquet`: the ppm deviation of every matched fragment of
    /// every calibrant PSM, keyed by LIBRARY-WIDE candidate id (see
    /// [`crate::masscal::Calibrants`]).
    ///
    /// Only a grouped run asks for it. A band's own fit is over its own ~2,000 deviations
    /// and its own `spectrum_q`, and combining the bands' fitted scalars is a wider
    /// tolerance than the estimator applied to the union (35% wider, measured; see
    /// [`crate::masscal`]), so `seed-pool` refits over the sidecars instead. Because the
    /// pooled q is not this band's q in either direction, the sidecar carries the band's
    /// best-scoring targets ([`crate::masscal::CALIBRANT_OFFER_PSMS`]) as well as the ones
    /// this band's own q accepts, and `seed-pool` decides. The band's own
    /// `masscal.json` is unaffected: it is still fitted on its confident targets alone.
    ///
    /// An ungrouped run already fits on every deviation it has, writes no sidecar, and
    /// selects exactly the rows it always did.
    pub emit_calibrants: bool,
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
    // `--out` must not be one of this stage's own inputs: every input is read
    // before the output is published, so writing over one replaces it and exits 0
    // (docs/31 F6). The shared guard existed and was wired into two stages.
    mumdia_io::refuse_output_over_input(
        p.out,
        &[
            ("--ms2", p.ms2),
            ("--lib-precursors", p.library_precursors),
            ("--lib-fragments", p.library_fragments),
        ],
    )?;
    // See extract: the bucketed index is dead weight on the fragindex path.
    let build_bucketed = !matches!(p.cfg.matcher, MatcherKind::Fragindex);
    let mut lib = match p.fragment_offset {
        None => Library::load_with(
            p.library_precursors,
            p.library_fragments,
            p.bucket_size,
            build_bucketed,
        )?,
        Some(offset) => Library::load_with_fragment_offset(
            p.library_precursors,
            p.library_fragments,
            offset,
            p.bucket_size,
            build_bucketed,
        )?,
    };
    // Decoded here unless the caller lent its own copy (see `ms2_scans`). The owned
    // buffer is declared first so it outlives the borrow. An empty lent slice is not
    // believed over the path: it means the caller had nothing to lend.
    let owned_scans: Vec<Ms2Scan>;
    let scans: &[Ms2Scan] = match p.ms2_scans {
        Some(s) if !s.is_empty() => s,
        _ => {
            owned_scans = load_ms2(p.ms2)?;
            if p.ms2_scans.is_some() && !owned_scans.is_empty() {
                warn!(
                    ms2 = p.ms2,
                    scans = owned_scans.len(),
                    "search-seed: the caller lent an empty MS2 buffer for a run that has                      scans; decoding the artifact instead of searching nothing"
                );
            }
            &owned_scans
        }
    };
    info!(
        candidates = lib.n_candidates(),
        scans = scans.len(),
        "search-seed: loaded"
    );

    // fragindex backend, built once at the seed's fragment tolerance when selected.
    let fidx = matches!(p.cfg.matcher, MatcherKind::Fragindex)
        .then(|| FragIndex::build(&lib, p.cfg.fragment_tol_ppm));
    if fidx.is_some() {
        // The index owns its own copy of every posting, and the seed reads neither the
        // predicted intensity nor the fragment name from either side: the hyperscore is
        // count + observed intensity, and the mass recalibration below needs only
        // `frag_mz`. So the library's `frag_int` and `frag_name_id` are dead from here on
        // -- 6 bytes per library fragment, held for the whole search. The bucketed path
        // keeps them, because `page_search` serves `idx_int` out of arrays built from them.
        lib.release_fragment_payload();
    }

    // Best-per-candidate PSM. The fragindex path parallelizes across isolation-window
    // groups (each scan belongs to exactly one window, so groups are independent) and
    // is bit-identical to the serial path via a deterministic per-candidate merge; the
    // bucketed path stays serial.
    let best: HashMap<u32, Best> = if let Some(idx) = fidx.as_ref() {
        seed_fragindex_windows(idx, scans, p.cfg)
    } else {
        let mut best: HashMap<u32, Best> = HashMap::new();
        for scan in scans {
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
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            scored.truncate(p.cfg.report_psms);
            for (cid, score, matched) in scored {
                let entry = best.entry(cid).or_insert(Best {
                    score: f64::NEG_INFINITY,
                    rt: 0.0,
                    matched: 0,
                    scan_index: 0,
                });
                if score > entry.score {
                    *entry = Best {
                        score,
                        rt: scan.rt_seconds,
                        matched,
                        scan_index: scan.scan_index,
                    };
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

    // One allocation per column instead of a growth sequence per column: the row count is
    // known, and these are the tables that make the seed's peak.
    let n_rows = rows.len();
    let mut cid_c = Vec::with_capacity(n_rows);
    let mut pform_c = Vec::with_capacity(n_rows);
    let mut charge_c = Vec::with_capacity(n_rows);
    let mut mz_c = Vec::with_capacity(n_rows);
    let mut base_c = Vec::with_capacity(n_rows);
    let mut prot_c = Vec::with_capacity(n_rows);
    // The label is the boolean, not a string that a later pass has to parse back into one.
    let mut is_dec: Vec<bool> = Vec::with_capacity(n_rows);
    let mut score_c = Vec::with_capacity(n_rows);
    let mut q_c = Vec::with_capacity(n_rows);
    let mut rt_c = Vec::with_capacity(n_rows);
    let mut matched_c = Vec::with_capacity(n_rows);
    let mut scan_c = Vec::with_capacity(n_rows);
    let mut irt_c = Vec::with_capacity(n_rows);
    for (i, (cid, b)) in rows.iter().enumerate() {
        let c = &lib.cands[*cid as usize];
        cid_c.push(*cid);
        pform_c.push(c.peptidoform.clone());
        charge_c.push(c.charge);
        mz_c.push(c.precursor_mz);
        base_c.push(c.base_peptide_id);
        prot_c.push(c.protein.clone());
        is_dec.push(c.is_decoy);
        score_c.push(b.score);
        q_c.push(q[i]);
        rt_c.push(b.rt);
        matched_c.push(b.matched as i32);
        scan_c.push(b.scan_index);
        irt_c.push(c.predicted_irt);
    }

    let n_at_1pct = count_targets_at_q(&q_c, &is_dec, p.cfg.fdr_seed);

    // Per-run fragment mass recalibration + learned tolerance
    // (docs/07_search_seed.md). Collect matched-fragment ppm deviations from
    // confident target PSMs; the median is the systematic offset, a high
    // percentile of the centered deviations sets the tolerance. Written to
    // <seed>.masscal.json and consumed by extract.
    let mut scan_by_index: HashMap<u32, &mumdia_core::types::Ms2Scan> = HashMap::new();
    for s in scans {
        scan_by_index.insert(s.scan_index, s);
    }
    let mut devs: Vec<f64> = Vec::new();
    // Fragment m/z paired to each ppm deviation, for the optional m/z-dependent
    // (LOESS) mass calibration. Used only when `mass_cal_loess` is set.
    let mut dev_mz: Vec<f64> = Vec::new();
    // The same deviations keyed by library-wide candidate id and scan, written beside the
    // masscal for a grouped run to pool. Stays empty otherwise.
    let gid_base = p.fragment_offset.unwrap_or(0);
    let mut calibrants = crate::masscal::Calibrants::default();
    let offer_floor = if p.emit_calibrants {
        calibrant_offer_floor(&score_c, &is_dec)
    } else {
        // No sidecar, so nothing is offered and the loop below selects exactly what it
        // always did: this band's confident targets.
        f64::INFINITY
    };
    for (i, (cid, b)) in rows.iter().enumerate() {
        if is_dec[i] {
            continue;
        }
        // This band's own calibrants, which are what its own `masscal.json` is fitted from.
        let confident = q[i] <= p.cfg.fdr_seed;
        // Offered to the pool on top of them, and never to this band's own fit.
        let offered = score_c[i] >= offer_floor;
        if !confident && !offered {
            continue;
        }
        if let Some(scan) = scan_by_index.get(&b.scan_index) {
            let gid = cid.checked_add(gid_base).with_context(|| {
                format!("search-seed: candidate id {cid} overflows the band offset {gid_base}")
            })?;
            // m/z only: the predicted intensity and the fragment name are not part of the
            // mass calibration, and on the fragindex path they no longer exist.
            let mzs = lib.cand_frag_mz(*cid);
            // The calibrant collection window has to be at least as wide as the search
            // tolerance, or the deviation percentile that SETS the learned tolerance is
            // truncated by the collection window itself. A literal 50 ppm silently did
            // that on any wide-tolerance configuration: a TOF run searched at 100 ppm
            // could not observe a deviation beyond 50, so its p95 was bounded at 50 and
            // the learned tolerance came out too tight. The 50 ppm floor is kept for the
            // narrow-tolerance case, where a wider window would just admit noise.
            let collect_ppm = 50.0_f64.max(p.cfg.fragment_tol_ppm);
            for &fmz in mzs {
                // library m/z is stored f32; widen once, the value is unchanged
                let fmz = fmz as f64;
                let (lo, hi) = mumdia_core::constants::ppm_bounds(fmz, collect_ppm);
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
                    if confident {
                        devs.push(pp);
                        dev_mz.push(fmz);
                    }
                    if p.emit_calibrants {
                        calibrants.candidate_id.push(gid);
                        calibrants.scan_index.push(b.scan_index);
                        // f32 is the library's own storage width for a fragment m/z, and a
                        // ppm deviation is a few parts in 1e7 at f32 -- far below the
                        // resolution any percentile of these points has.
                        calibrants.frag_mz.push(fmz as f32);
                        calibrants.ppm.push(pp as f32);
                    }
                }
            }
        }
    }
    // Median offset + 95th-percentile-of-centered tolerance, with the optional robust
    // second pass and the optional m/z-dependent grid. The estimator lives in
    // `crate::masscal` because `seed-pool` fits the very same one over the pooled
    // deviations of a grouped run.
    let cal = crate::masscal::MassCal::fit_from(&devs, &dev_mz, p.cfg);
    mumdia_io::json::write_json(&crate::masscal::json_path(p.out), &cal.to_json())?;
    if p.emit_calibrants {
        let n_cal = crate::masscal::write_calibrants(p.out, &calibrants)?;
        info!(
            calibrant_deviations = n_cal,
            "search-seed: wrote the calibrant deviations for the pooled mass calibration"
        );
    }
    info!(
        frag_ppm_offset = cal.frag_ppm_offset,
        frag_tol_learned = cal.frag_tol_ppm,
        cal_passes = cal.cal_passes,
        "search-seed: mass recalibration"
    );

    let n = write_table(
        p.out,
        vec![
            Col::U32("candidate_id".into(), cid_c),
            Col::Str("peptidoform".into(), pform_c),
            Col::I32("charge".into(), charge_c),
            Col::F64("precursor_mz".into(), mz_c),
            Col::U32("base_peptide_id".into(), base_c),
            Col::Str("protein".into(), prot_c),
            // Materialised here and nowhere else: the artifact's column is text, but the
            // engine never needed the strings for anything but the write itself. (A `Col`
            // variant over a two-valued dictionary would remove these too; that is a
            // mumdia-io change.)
            Col::Str("label".into(), label_column(&is_dec)),
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
            "top_n_peaks": p.cfg.top_n_peaks,
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

/// Score of the [`crate::masscal::CALIBRANT_OFFER_PSMS`]-th best TARGET PSM: every target at
/// or above it is offered to the pooled mass calibration whatever this band's own q says.
///
/// `-inf` when the band has fewer targets than that, so it offers all of them. The pooled
/// selection inside one band is a score-ranked prefix of the band's targets, so a prefix is
/// the shape of superset that makes `seed-pool`'s pooled-q selection exact.
fn calibrant_offer_floor(scores: &[f64], is_decoy: &[bool]) -> f64 {
    let mut targets: Vec<f64> = scores
        .iter()
        .zip(is_decoy)
        .filter(|(_, d)| !**d)
        .map(|(s, _)| *s)
        .collect();
    targets.sort_by(|a, b| b.total_cmp(a));
    targets
        .get(crate::masscal::CALIBRANT_OFFER_PSMS - 1)
        .copied()
        .unwrap_or(f64::NEG_INFINITY)
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
                .total_cmp(&scan.peaks[a].intensity)
                .then(a.cmp(&b))
        });
        idx.truncate(top_n);
        idx.sort_unstable();
        idx
    } else {
        (0..scan.peaks.len()).collect()
    }
}

/// The seed artifact's `label` column, built from the boolean the library already carries.
///
/// It used to be the other way round: the column was assembled as `Vec<String>` while the
/// rows were walked and the `is_decoy` vector was then rebuilt by comparing each of those
/// strings to `"decoy"`, which is a full extra pass over the rows to recover a bit that
/// `Candidate::is_decoy` had all along.
fn label_column(is_decoy: &[bool]) -> Vec<String> {
    is_decoy
        .iter()
        .map(|&d| if d { "decoy" } else { "target" }.to_string())
        .collect()
}

/// The `(lower_mz, upper_mz)` `convert` writes for an MS2 scan whose reported isolation
/// window is zero-width or whose precursor is missing (`convert.rs`, "AIF / all-ion"). It is
/// a property of the SCAN, so it reads the same on every band of a grouped search.
const FULL_RANGE_WINDOW: (f64, f64) = (0.0, 1.0e6);

/// True when the run mixes the full-range isolation window with narrow ones, which is the
/// signature of a scan that lost its isolation window: [`FULL_RANGE_WINDOW`] is what
/// `convert` substitutes. Every window full-range is a legitimate all-ion acquisition and
/// none is an ordinary run, so only the mixture is worth a warning.
///
/// This test, unlike the candidate-width ratio below, reads the same under a grouped search.
/// A full-range window resolves to the WHOLE loaded band, and so does the band's own
/// isolation window, so on a band the two widths are equal and no ratio can separate them.
fn has_stray_full_range_window(windows: &[(f64, f64)]) -> bool {
    let n = windows.iter().filter(|&&w| w == FULL_RANGE_WINDOW).count();
    n > 0 && n < windows.len()
}

/// What the per-worker scratch sizing and the width warning read off the window groups.
struct WindowSurvey {
    /// Median candidate-window width over the SERVED groups; 1 when none is served.
    median: usize,
    /// Widest served candidate window, or `None` when no group is served.
    widest: Option<usize>,
    /// Groups this search serves, of `total`.
    served: usize,
    total: usize,
}

/// Candidate-window widths over the window groups THIS search actually serves, from each
/// group's `(lo, hi)` candidate range.
///
/// The MEDIAN is the sizing statistic, not the widest and certainly not the library. The
/// scratch is indexed window-relative and grows on demand, so an underestimate costs a
/// reallocation while an overestimate costs 16 B x width per rayon worker up front.
///
/// The max is the wrong statistic because one window can be the whole library:
/// [`FULL_RANGE_WINDOW`] resolves to every candidate. So a single malformed or all-ion scan
/// in an otherwise 50-window run sized every worker's scratch to the library -- 877 MB per
/// worker on the profiled 54.8M-candidate library, about 28 GB of commit charge on 32 cores,
/// for arrays a worker inside one narrow window never touches.
///
/// Groups that select nothing (`hi <= lo`) are EXCLUDED from the widths, which is the same
/// test the worker uses to skip a group, and counted instead: `served` against `total`.
/// Under an isolation-window-group search the library is one m/z band while the run's
/// windows are all of them, so with 63 bands 62 of every 63 groups select nothing and
/// contributed a width of 1. The median was then 1 on every band and `widest > 8 * median`
/// fired on every band, printing a warning about a zero-width or missing isolation window
/// once per band for a run that has no such scan.
///
/// `served`/`total` is what keeps the correction from becoming a memory regression: see
/// [`initial_scratch_width`].
///
/// Widths are collected in group order and then sorted, so the value is deterministic.
fn window_survey(ranges: &[(u32, u32)]) -> WindowSurvey {
    let mut widths: Vec<usize> = ranges
        .iter()
        .filter(|(lo, hi)| hi > lo)
        .map(|(lo, hi)| hi.saturating_sub(*lo) as usize + 1)
        .collect();
    widths.sort_unstable();
    let median = widths.get(widths.len() / 2).copied().unwrap_or(1).max(1);
    WindowSurvey {
        median,
        widest: widths.last().copied(),
        served: widths.len(),
        total: ranges.len(),
    }
}

/// Width each rayon leaf's [`SeedScratch`] is allocated with before it knows which groups it
/// drew.
///
/// `map_init` runs its initialiser once per LEAF, not once per thread, and it runs before
/// the leaf sees a group, so a leaf that draws only unserved groups pays the allocation and
/// then returns empty. Where every group is served -- an ungrouped search -- the eager size
/// is what the leaf's first `accumulate` would have grown it to anyway, so sizing it up
/// front costs nothing and saves the reallocation. Where most groups are NOT served it is
/// pure waste, and that is exactly the banded search this survey was corrected for: the
/// library is one m/z band while the scans are all of them, so 62 of 63 groups are unserved
/// and eager sizing would commit 16 B x band width on every leaf (about 14 MB per worker on
/// the profiled 54.8M-candidate library over 63 bands, roughly 445 MB across 32 workers)
/// for arrays nearly all of them never touch.
///
/// So: the served median when a majority of groups is served, one slot otherwise. The leaf
/// that does draw work still reaches the same size through `SeedScratch::ensure`, in one
/// amortised reallocation.
fn initial_scratch_width(survey: &WindowSurvey) -> usize {
    if survey.served * 2 > survey.total {
        survey.median
    } else {
        1
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
            .entry((
                scan.window.lower_mz.to_bits(),
                scan.window.upper_mz.to_bits(),
            ))
            .or_default()
            .push(si);
    }
    let group_vec: Vec<Vec<usize>> = groups.into_values().collect();
    let group_windows: Vec<(f64, f64)> = group_vec
        .iter()
        .filter_map(|ids| ids.first())
        .map(|&si| {
            let w = &scans[si].window;
            (w.lower_mz, w.upper_mz)
        })
        .collect();
    let ranges: Vec<(u32, u32)> = group_windows
        .iter()
        .map(|&(lo, hi)| idx.candidate_range(lo, hi))
        .collect();
    let survey = window_survey(&ranges);
    // Two diagnostics, in order of how much each actually establishes.
    //
    // 1. A scan that lost its isolation window is recognisable from the WINDOW: convert
    //    substitutes the full m/z range, so a run mixing full-range groups with narrow ones
    //    contains such a scan whatever library happens to be loaded. This is the test that
    //    survives a grouped search, where the ratio below cannot fire at all -- a
    //    full-range window resolves to the whole loaded band, and so does the band's own
    //    isolation window, so on a band the two widths are equal.
    // 2. Otherwise, a window covering far more candidates than the median is still worth a
    //    word even when the window itself is well formed (an unusually wide quadrupole
    //    setting, or a library whose m/z range sits inside one window), before anyone
    //    wonders why extraction is slow. Under a grouped search this second test sees only
    //    the in-band groups, so it reports on this band, which is all it can honestly say.
    if has_stray_full_range_window(&group_windows) {
        warn!(
            full_range_windows = group_windows
                .iter()
                .filter(|&&w| w == FULL_RANGE_WINDOW)
                .count(),
            window_groups = group_windows.len(),
            "search-seed: some isolation windows are the full m/z range that convert \
             writes for a scan with a zero-width or missing isolation window, while others \
             are narrow. An all-ion acquisition makes EVERY window full-range; a mixture \
             means those scans lost their isolation window"
        );
    } else if let Some(widest) = survey.widest {
        if widest > 8 * survey.median {
            warn!(
                median_window_candidates = survey.median,
                widest_window_candidates = widest,
                library_candidates = idx.n_cand(),
                "search-seed: one isolation window covers far more candidates than the \
                 median of the window groups this search serves. An all-ion acquisition \
                 does this legitimately"
            );
        }
    }

    let scratch_width = initial_scratch_width(&survey);
    let partials: Vec<Vec<(u32, Best)>> = group_vec
        .par_iter()
        .map_init(
            || SeedScratch::new(scratch_width),
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
                    // Borrowed, not copied: `touched` can be as long as the candidate
                    // window, so the copy was one allocation of up to a window's width per
                    // scan. Same slice, same order, so the scored list is unchanged.
                    let touched = scratch.touched();
                    let mut scored: Vec<(u32, f64, u32)> = touched
                        .iter()
                        .filter(|&&cid| scratch.count(cid) as usize >= cfg.min_matched_peaks)
                        .map(|&cid| {
                            (
                                cid,
                                hyperscore(scratch.count(cid), scratch.obs_sum(cid)),
                                scratch.count(cid),
                            )
                        })
                        .collect();
                    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    scored.truncate(cfg.report_psms);
                    for (cid, score, matched) in scored {
                        let e = local.entry(cid).or_insert(Best {
                            score: f64::NEG_INFINITY,
                            rt: 0.0,
                            matched: 0,
                            scan_index: 0,
                        });
                        if score > e.score {
                            *e = Best {
                                score,
                                rt: scan.rt_seconds,
                                matched,
                                scan_index: scan.scan_index,
                            };
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

#[cfg(test)]
mod survey_tests {
    use super::{
        has_stray_full_range_window, initial_scratch_width, label_column, window_survey,
        FULL_RANGE_WINDOW,
    };

    fn window_width_stats(ranges: &[(u32, u32)]) -> (usize, Option<usize>) {
        let s = window_survey(ranges);
        (s.median, s.widest)
    }

    #[test]
    fn window_width_stats_ignores_the_groups_this_band_does_not_serve() {
        // A grouped search loads ONE m/z band of the library and sees the run's whole
        // window list, so every group outside the band resolves to an empty candidate
        // range. With 63 bands that is 62 empty groups against one real one; counting the
        // empty groups as width 1 put the median at 1 (scratch sized to a single slot,
        // resized on first use) and made `widest > 8 * median` true on every band, so the
        // zero-width-isolation-window warning fired 63 times per run for a run with no such
        // scan.
        let mut ranges = vec![(0u32, 0u32); 62];
        ranges.push((1000, 6000));
        let (median, widest) = window_width_stats(&ranges);
        assert_eq!((median, widest), (5001, Some(5001)));
        // ... which is what the guard needs in order not to fire.
        assert!(widest.unwrap() <= 8 * median);
    }

    #[test]
    fn window_width_stats_is_the_median_of_the_served_groups_and_still_flags_an_outlier() {
        // Widths 11, 11, 11, and one group covering the library: the median is still 11 and
        // the outlier is still reported. Order-independent, and the empty groups mixed in
        // change neither answer.
        let ranges = [(0, 10), (100, 110), (0, 0), (50, 60), (0, 5_000_000)];
        let (median, widest) = window_width_stats(&ranges);
        assert_eq!((median, widest), (11, Some(5_000_001)));
        assert!(widest.unwrap() > 8 * median);
        // No group selects anything: nothing to size from, nothing to warn about.
        assert_eq!(window_width_stats(&[(7, 7), (0, 0)]), (1, None));
        assert_eq!(window_width_stats(&[]), (1, None));
    }

    #[test]
    fn a_band_that_serves_one_group_of_many_is_not_sized_from_the_band_width() {
        // `map_init` runs its initialiser once per rayon LEAF and before the leaf knows
        // which groups it drew, so sizing the scratch from the served median would commit
        // 16 B x band width on every leaf while only the leaf holding the one in-band group
        // ever touches it. The correction to the median must not turn into that.
        let mut banded = vec![(0u32, 0u32); 62];
        banded.push((1000, 871_000));
        let survey = window_survey(&banded);
        assert_eq!((survey.served, survey.total), (1, 63));
        assert_eq!(
            survey.median, 870_001,
            "the warning still reads the band width"
        );
        assert_eq!(
            initial_scratch_width(&survey),
            1,
            "62 of 63 leaves would allocate a band they never probe"
        );

        // An ungrouped search serves every group, and there the eager size is what the
        // leaf's first `accumulate` would have grown it to anyway.
        let plain = window_survey(&[(0, 10), (100, 110), (50, 60)]);
        assert_eq!((plain.served, plain.total), (3, 3));
        assert_eq!(initial_scratch_width(&plain), 11);
        // One dead window among live ones does not tip the majority.
        let mostly = window_survey(&[(0, 10), (100, 110), (7, 7), (50, 60)]);
        assert_eq!(initial_scratch_width(&mostly), 11);
        // Nothing served at all: one slot, and `SeedScratch::ensure` never runs.
        assert_eq!(initial_scratch_width(&window_survey(&[])), 1);
    }

    #[test]
    fn a_stray_full_range_window_is_flagged_from_the_window_and_not_from_the_width() {
        // convert maps a zero-width or missing isolation window to the full m/z range, so
        // the malformed scan is visible in the window itself. The candidate-width ratio
        // that used to carry this warning cannot see it under a grouped search: the
        // full-range window resolves to the WHOLE loaded band, and so does the band's own
        // isolation window, so the two served widths are equal and `widest > 8 * median`
        // is false. The band below is 5,001 candidates wide and serves exactly those two
        // groups.
        let banded_ranges = [(1000u32, 6000u32), (1000, 6000)];
        let banded = window_survey(&banded_ranges);
        assert_eq!(banded.widest, Some(5001));
        assert!(
            banded.widest.unwrap() <= 8 * banded.median,
            "the width ratio is silent on a band, which is the false negative"
        );
        // The window test is not, and it does not consult the library at all.
        let mut banded_windows = vec![(500.0f64, 510.0f64); 62];
        banded_windows.push(FULL_RANGE_WINDOW);
        assert!(has_stray_full_range_window(&banded_windows));

        // A genuine all-ion acquisition is every window full-range: legitimate, silent.
        assert!(!has_stray_full_range_window(&[
            FULL_RANGE_WINDOW,
            FULL_RANGE_WINDOW
        ]));
        // An ordinary run has none.
        assert!(!has_stray_full_range_window(&[
            (500.0, 510.0),
            (510.0, 520.0)
        ]));
        assert!(!has_stray_full_range_window(&[]));
        // A wide-but-not-full window is not this warning's business; the width ratio keeps
        // it.
        assert!(!has_stray_full_range_window(&[
            (500.0, 510.0),
            (0.0, 2000.0)
        ]));
    }

    #[test]
    fn a_band_offers_a_score_prefix_of_its_targets_to_the_pooled_mass_calibration() {
        use super::calibrant_offer_floor;
        use crate::masscal::CALIBRANT_OFFER_PSMS;
        // Fewer targets than the cap: every target is offered, whatever the band's own q
        // makes of them. This is the CI fixture's case, where each band's q rejects all of
        // its targets and a q-selected sidecar would be empty in every band.
        let scores = [9.0, 8.0, 7.0, 6.0];
        let decoys = [false, true, false, false];
        assert_eq!(
            calibrant_offer_floor(&scores, &decoys),
            f64::NEG_INFINITY,
            "3 targets, so all three are at or above the floor"
        );
        assert_eq!(calibrant_offer_floor(&[], &[]), f64::NEG_INFINITY);
        // More targets than the cap: the floor is the cap-th best TARGET score, and the
        // decoys interleaved among them do not consume a slot.
        let n = 2 * (CALIBRANT_OFFER_PSMS + 500);
        let scores: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();
        let decoys: Vec<bool> = (0..n).map(|i| i % 2 == 1).collect();
        let floor = calibrant_offer_floor(&scores, &decoys);
        let offered = scores
            .iter()
            .zip(&decoys)
            .filter(|(s, d)| !**d && **s >= floor)
            .count();
        assert_eq!(offered, CALIBRANT_OFFER_PSMS);
    }

    #[test]
    fn label_column_is_the_boolean_the_old_code_parsed_back_out_of_it() {
        // The artifact text is unchanged, and the boolean the stage uses is now its source
        // rather than its product: `is_decoy` -> column -> `l == "decoy"` is the identity.
        let is_dec = [false, true, true, false];
        let col = label_column(&is_dec);
        assert_eq!(col, vec!["target", "decoy", "decoy", "target"]);
        let round_trip: Vec<bool> = col.iter().map(|l| l == "decoy").collect();
        assert_eq!(round_trip, is_dec);
        assert!(label_column(&[]).is_empty());
    }
}

#[cfg(test)]
mod peak_selection_tests {
    use super::select_peaks;
    use mumdia_core::types::{IsolationWindow, Ms2Scan, Peak};

    fn scan(n: usize) -> Ms2Scan {
        Ms2Scan {
            scan_index: 0,
            id: "scan=0".into(),
            rt_seconds: 0.0,
            window: IsolationWindow {
                target_mz: 0.0,
                lower_mz: 0.0,
                upper_mz: 2_000.0,
                im_lower: None,
                im_upper: None,
            },
            peaks: (0..n)
                .map(|i| Peak {
                    mz: 100.0 + i as f64,
                    intensity: i as f32,
                    ion_mobility: None,
                })
                .collect(),
        }
    }

    #[test]
    fn zero_selects_all_and_seed_cap_keeps_only_top_intensity_peaks() {
        let s = scan(305);
        assert_eq!(select_peaks(&s, 0), (0..305).collect::<Vec<_>>());
        assert_eq!(select_peaks(&s, 300), (5..305).collect::<Vec<_>>());
    }
}
