//! Pool the seed searches of a run's isolation-window groups into one seed table
//! (`groups.window_groups`, `groups.calibration = global`).
//!
//! Each group seeds its own library band with band-local `candidate_id`s and its own
//! `spectrum_q`, estimated on that band's PSMs alone. The retention-time calibration wants
//! the whole run's anchors on one q scale, so the pooled table carries library-wide ids
//! (local id plus the band's first row), a `spectrum_q` re-estimated over the union with the
//! same target-decoy kernel the seed uses, and one mass calibration fitted over the bands'
//! calibrants. Rows stay one best PSM per candidate, as the seed writes them:
//! bands are disjoint in candidates except where windows overlap across a cut, and there the
//! higher score wins.
//!
//! After the library band iRTs have been re-predicted against the pooled anchors
//! (multi-head calibration per band), [`refresh_irt`] copies each anchor's current iRT from
//! its band's table into the pooled seed, so `rt-im-train` can take anchors' iRT from the
//! seed (`anchor_irt_from_seed`) while looking at one band's library at a time.
//!
//! The mass calibration is pooled the same way and for the same reason: the bands write
//! their calibrant ppm deviations beside their masscal, and this stage fits
//! [`crate::masscal::MassCal`] ONCE over the union, keeping the deviations whose PSM passes
//! the POOLED q. Combining the bands' fitted scalars instead came out 35% wider on the
//! six-file HYE Astral benchmark and cost 3.4% of the peptides; [`crate::masscal`] has the
//! numbers and the mechanism. A band directory written before the sidecar existed still
//! pools, by the scalar combination, with a warning.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use mumdia_core::config::SearchSeedConfig;
use mumdia_io::table::{write_table, Col, TableFile};
use serde_json::json;
use tracing::{info, warn};

use crate::fdr::target_decoy_q;
use crate::masscal::MassCal;

/// One group's seed table and the library band it searched.
pub struct BandSeed {
    pub path: String,
    /// The band's first library row: local id plus this is the library-wide id.
    pub offset: u32,
    /// Rows of the band; the band's candidates are `offset..offset + rows`.
    pub rows: u32,
}

pub struct SeedPoolParams<'a> {
    pub seeds: &'a [BandSeed],
    /// The groups' `<seed>.masscal.json`, same order as `seeds`.
    pub masscals: &'a [String],
    /// The groups' `<seed>.masscal.parquet` calibrant deviations, same order as `seeds`.
    /// A path that does not exist is a band seeded before the sidecar existed; the pool
    /// then falls back to combining the bands' scalars.
    pub calibrants: &'a [String],
    /// The seed configuration the bands searched under. The pooled mass calibration is
    /// the same estimator under the same `fdr_seed`, `fragment_tol_ppm`,
    /// `two_pass_mass_cal` and `mass_cal_loess` that a single-library search would use.
    pub cfg: &'a SearchSeedConfig,
    /// Output seed table; `<out>.masscal.json` is written beside it.
    pub out: &'a str,
}

/// One seed row with library-wide id.
#[derive(Clone)]
struct Row {
    cid: u32,
    pform: String,
    charge: i32,
    mz: f64,
    base: u32,
    protein: String,
    label: String,
    score: f64,
    rt: f64,
    irt: f32,
    matched: i32,
    scan: u32,
}

fn read_rows(path: &str, offset: u32) -> Result<Vec<Row>> {
    let t = TableFile::open(path)?;
    let cid = t.u32("candidate_id")?;
    let pform = t.str("peptidoform")?;
    let charge = t.i32("charge")?;
    let mz = t.f64("precursor_mz")?;
    let base = t.u32("base_peptide_id")?;
    let protein = t.str("protein")?;
    let label = t.str("label")?;
    let score = t.f64("score")?;
    let rt = t.f64("observed_rt")?;
    let irt = t.f32("predicted_irt")?;
    let matched = t.i32("matched_peaks")?;
    let scan = t.u32("scan_index")?;
    let mut rows = Vec::with_capacity(t.nrows);
    for i in 0..t.nrows {
        rows.push(Row {
            cid: cid[i]
                .checked_add(offset)
                .with_context(|| format!("candidate id overflow pooling {path}"))?,
            pform: pform[i].clone(),
            charge: charge[i],
            mz: mz[i],
            base: base[i],
            protein: protein[i].clone(),
            label: label[i].clone(),
            score: score[i],
            rt: rt[i],
            irt: irt[i],
            matched: matched[i],
            scan: scan[i],
        });
    }
    Ok(rows)
}

fn write_rows(path: &str, rows: &[Row], q: Vec<f64>) -> Result<u64> {
    write_table(
        path,
        vec![
            Col::U32("candidate_id".into(), rows.iter().map(|r| r.cid).collect()),
            Col::Str(
                "peptidoform".into(),
                rows.iter().map(|r| r.pform.clone()).collect(),
            ),
            Col::I32("charge".into(), rows.iter().map(|r| r.charge).collect()),
            Col::F64("precursor_mz".into(), rows.iter().map(|r| r.mz).collect()),
            Col::U32(
                "base_peptide_id".into(),
                rows.iter().map(|r| r.base).collect(),
            ),
            Col::Str(
                "protein".into(),
                rows.iter().map(|r| r.protein.clone()).collect(),
            ),
            Col::Str(
                "label".into(),
                rows.iter().map(|r| r.label.clone()).collect(),
            ),
            Col::F64("score".into(), rows.iter().map(|r| r.score).collect()),
            Col::F64("spectrum_q".into(), q),
            Col::F64("observed_rt".into(), rows.iter().map(|r| r.rt).collect()),
            Col::F32("predicted_irt".into(), rows.iter().map(|r| r.irt).collect()),
            Col::I32(
                "matched_peaks".into(),
                rows.iter().map(|r| r.matched).collect(),
            ),
            Col::U32("scan_index".into(), rows.iter().map(|r| r.scan).collect()),
        ],
    )
}

/// Pool the bands' seeds; returns the pooled row count.
pub fn run(p: SeedPoolParams) -> Result<u64> {
    let t0 = Instant::now();
    if p.seeds.is_empty() {
        bail!("seed-pool: no group seed tables");
    }
    if p.masscals.len() != p.seeds.len() {
        bail!(
            "seed-pool: {} seed tables but {} mass calibrations",
            p.seeds.len(),
            p.masscals.len()
        );
    }
    if p.calibrants.len() != p.seeds.len() {
        bail!(
            "seed-pool: {} seed tables but {} calibrant sidecar paths",
            p.seeds.len(),
            p.calibrants.len()
        );
    }
    // One row per library-wide candidate: where two bands share a candidate (window
    // overlap across a cut) the higher score stays, and on a tie the earlier band, so the
    // pooled q sees each once. The band the kept row came from is remembered for the mass
    // calibration below.
    let mut best: HashMap<u32, (usize, Row)> = HashMap::new();
    let mut n_in = 0usize;
    for (bi, band) in p.seeds.iter().enumerate() {
        for r in read_rows(&band.path, band.offset)? {
            n_in += 1;
            match best.get(&r.cid) {
                Some((_, b)) if b.score >= r.score => {}
                _ => {
                    best.insert(r.cid, (bi, r));
                }
            }
        }
    }
    let mut kept: Vec<(usize, Row)> = best.into_values().collect();
    kept.sort_by_key(|(_, r)| r.cid);
    let (winner_band, rows): (Vec<usize>, Vec<Row>) = kept.into_iter().unzip();
    crate::fdr::validate_labels(&rows.iter().map(|r| r.label.clone()).collect::<Vec<_>>())?;
    let pairs: Vec<(f64, bool)> = rows.iter().map(|r| (r.score, r.label == "decoy")).collect();
    let q = target_decoy_q(&pairs);
    // The pooled calibrant selector: for every target candidate the POOLED q accepts at
    // the seed threshold, the scan its winning PSM was matched on and the band that PSM
    // came from. A band's own q accepts a different, looser set -- that is half of why the
    // banded tolerance came out wide. The scan and the band pin the union to one PSM per
    // candidate, which is the population an ungrouped seed fits on: a precursor in the
    // overlap of two windows is loaded by both bands and each band serves both windows for
    // it, so both usually hold the SAME (candidate, scan) PSM with the same deviations, and
    // the scan alone would count them once per band.
    let accepted: HashMap<u32, (u32, usize)> = rows
        .iter()
        .zip(&winner_band)
        .zip(&q)
        .filter(|((r, _), qq)| **qq <= p.cfg.fdr_seed && r.label != "decoy")
        .map(|((r, &bi), _)| (r.cid, (r.scan, bi)))
        .collect();
    let n_out = rows.len();
    let n = write_rows(p.out, &rows, q)?;

    // One fit over the run's deviations when the bands wrote them; otherwise the old
    // combination of their scalars.
    let (cal, source) = match pooled_masscal(&p, &accepted)? {
        Some(cal) => (cal, "pooled_deviations"),
        None => (combine_band_scalars(&p)?, "band_scalars"),
    };
    let mut body = cal.to_json();
    if let Some(obj) = body.as_object_mut() {
        obj.insert("pooled_from_groups".into(), json!(p.seeds.len()));
        obj.insert("masscal_source".into(), json!(source));
    }
    mumdia_io::json::write_json(&crate::masscal::json_path(p.out), &body)?;
    info!(
        groups = p.seeds.len(),
        rows_in = n_in,
        rows_out = n_out,
        frag_ppm_offset = cal.frag_ppm_offset,
        frag_tol_ppm = cal.frag_tol_ppm,
        n_dev = cal.n_dev,
        masscal_source = source,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "seed-pool: done"
    );
    Ok(n)
}

/// Fit the mass calibration once over the bands' calibrant deviations, keeping the ones
/// whose PSM the POOLED q accepts (`accepted`: candidate -> (winning scan, the band the
/// winning row came from)). Only that band's rows of the candidate are taken, so a
/// candidate two overlapping bands both matched contributes its deviations once.
///
/// `Ok(None)` when a band has no sidecar, which is a band directory seeded before the
/// sidecar existed: the caller then combines the bands' scalars as it always did.
fn pooled_masscal(
    p: &SeedPoolParams,
    accepted: &HashMap<u32, (u32, usize)>,
) -> Result<Option<MassCal>> {
    let missing = p
        .calibrants
        .iter()
        .filter(|c| !Path::new(c.as_str()).exists())
        .count();
    if missing > 0 {
        warn!(
            missing,
            groups = p.calibrants.len(),
            "seed-pool: a group has no calibrant deviation sidecar, which is a band                  directory seeded before the pooled mass calibration existed. Combining the                  bands' fitted scalars instead, which is measurably WIDER than one fit over the                  pooled deviations (11.40 against 8.45 ppm on the six-file HYE Astral                  benchmark). Re-run search-seed on the bands to get the pooled fit"
        );
        if p.cfg.mass_cal_loess {
            warn!(
                "seed-pool: search_seed.mass_cal_loess is set but cannot be honoured from                      the bands' scalars -- an m/z-dependent grid needs the deviations. extract                      will apply the scalar offset only"
            );
        }
        return Ok(None);
    }
    // Bands in order, rows in file order: the fit sorts internally, so the result does not
    // depend on this, but the LOESS grid's tie order does.
    let mut devs: Vec<f64> = Vec::new();
    let mut dev_mz: Vec<f64> = Vec::new();
    let mut n_band = 0usize;
    // The largest band's sidecar is what this stage holds decoded at once, beside the
    // accepted deviations.
    let mut largest_band_bytes = 0u64;
    for (bi, path) in p.calibrants.iter().enumerate() {
        let c = crate::masscal::read_calibrants(path)?;
        n_band += c.len();
        largest_band_bytes = largest_band_bytes.max(c.bytes());
        for i in 0..c.len() {
            if accepted.get(&c.candidate_id[i]) == Some(&(c.scan_index[i], bi)) {
                devs.push(c.ppm[i] as f64);
                dev_mz.push(c.frag_mz[i] as f64);
            }
        }
    }
    let cal = MassCal::fit_from(&devs, &dev_mz, p.cfg);
    if cal.cal_passes == 0 {
        warn!(
            calibrants = devs.len(),
            frag_tol_ppm = cal.frag_tol_ppm,
            "seed-pool: too few pooled calibrants to fit a fragment tolerance; extract runs                  at the configured tolerance with no offset"
        );
    }
    info!(
        band_deviations = n_band,
        band_deviation_bytes = (n_band * crate::masscal::Calibrants::BYTES_PER_DEVIATION) as u64,
        largest_band_bytes,
        pooled_deviations = devs.len(),
        frag_ppm_offset = cal.frag_ppm_offset,
        frag_tol_ppm = cal.frag_tol_ppm,
        cal_passes = cal.cal_passes,
        "seed-pool: fragment mass calibration fitted once over the pooled deviations"
    );
    Ok(Some(cal))
}

/// The pre-sidecar combination: the bands' scalar offsets and learned tolerances weighted
/// by calibrant count. Kept so a band directory written by an older build still pools.
///
/// It is not the estimator: a mean of per-band p95s is not the p95 of the union, and the
/// bands selected their calibrants on their own q. The m/z grids are not combined either,
/// because they would need the deviations.
fn combine_band_scalars(p: &SeedPoolParams) -> Result<MassCal> {
    let mut w_sum = 0.0;
    let (mut off, mut tol, mut med, mut mad) = (0.0, 0.0, 0.0, 0.0);
    let (mut n_dev, mut passes) = (0u64, 0u64);
    // A band with no calibrants wrote the configured tolerance in place of a learned one
    // (search-seed's failure branch); if no band calibrated, the pool keeps that
    // tolerance rather than averaging nothing into a zero.
    let mut uncalibrated_tol = 0.0f64;
    for path in p.masscals {
        let v: serde_json::Value = mumdia_io::json::read_json(path)?;
        let w = v["n_dev"].as_u64().unwrap_or(0);
        n_dev += w;
        passes = passes.max(v["cal_passes"].as_u64().unwrap_or(0));
        if w == 0 {
            uncalibrated_tol = uncalibrated_tol.max(v["frag_tol_ppm"].as_f64().unwrap_or(0.0));
            continue;
        }
        let wf = w as f64;
        w_sum += wf;
        off += wf * v["frag_ppm_offset"].as_f64().unwrap_or(0.0);
        tol += wf * v["frag_tol_ppm"].as_f64().unwrap_or(0.0);
        med += wf * v["ppm_residual_median"].as_f64().unwrap_or(0.0);
        mad += wf * v["ppm_residual_mad"].as_f64().unwrap_or(0.0);
    }
    let (off, tol, med, mad) = if w_sum > 0.0 {
        (off / w_sum, tol / w_sum, med / w_sum, mad / w_sum)
    } else {
        warn!(
            frag_tol_ppm = uncalibrated_tol,
            "seed-pool: no group had confident seeds to calibrate mass; extract runs at the              configured fragment tolerance with no offset"
        );
        (0.0, uncalibrated_tol, 0.0, 0.0)
    };
    Ok(MassCal {
        frag_ppm_offset: off,
        frag_tol_ppm: tol,
        n_dev,
        cal_passes: passes,
        ppm_residual_median: med,
        ppm_residual_mad: mad,
        mz_cal_grid_mz: Vec::new(),
        mz_cal_grid_ppm: Vec::new(),
    })
}

/// Rewrite `seed_in` to `seed_out` with each row's `predicted_irt` taken from the band
/// table that holds its candidate (`bands`: precursor table with band-local ids, and the
/// band's first library row). Rows outside every band keep their value.
///
/// Bands are applied in order, so where two bands hold one candidate (windows overlapping
/// a cut) the later band's value stays, as it always did.
///
/// Two shortcuts, each with the general path behind it:
/// - A band file's ids are its rows `0..n` (`groups::write_band_slice` writes them so),
///   and then the iRT of local id `l` is simply row `l`. A band table whose ids are not
///   dense and ascending is looked up through a map, last row winning, as before.
/// - The pooled seed is written sorted by candidate id (`seed_pool::run`), so the rows a
///   band can refresh, ids `offset..offset + n`, are one slice found by binary search,
///   instead of a pass over every row per band: 63 bands made that 63 passes over the whole
///   pooled seed. An unsorted input is scanned whole.
pub fn refresh_irt(seed_in: &str, seed_out: &str, bands: &[(String, u32)]) -> Result<u64> {
    let t0 = Instant::now();
    let mut rows = read_rows(seed_in, 0)?;
    let t = TableFile::open(seed_in)?;
    let q = t.f64("spectrum_q")?;
    let mut refreshed = 0usize;
    let sorted = rows.windows(2).all(|w| w[0].cid <= w[1].cid);
    for (path, offset) in bands {
        let b = TableFile::open(path)?;
        let cid = b.u32("candidate_id")?;
        let irt = b.f32("predicted_irt")?;
        let n = b.nrows;
        let dense = cid.iter().enumerate().all(|(i, &c)| c as usize == i);
        let by_local: Option<HashMap<u32, f32>> =
            (!dense).then(|| cid.iter().copied().zip(irt.iter().copied()).collect());
        let lookup = |local: u32| -> Option<f32> {
            match &by_local {
                Some(m) => m.get(&local).copied(),
                None => irt.get(local as usize).copied(),
            }
        };
        let span = if sorted {
            let (lo, hi) = (*offset as u64, *offset as u64 + n as u64);
            rows.partition_point(|r| (r.cid as u64) < lo)
                ..rows.partition_point(|r| (r.cid as u64) < hi)
        } else {
            0..rows.len()
        };
        for r in rows[span].iter_mut() {
            let Some(local) = r.cid.checked_sub(*offset) else {
                continue;
            };
            if (local as usize) < n {
                if let Some(v) = lookup(local) {
                    r.irt = v;
                    refreshed += 1;
                }
            }
        }
    }
    let n = write_rows(seed_out, &rows, q)?;
    info!(
        rows = rows.len(),
        refreshed,
        bands = bands.len(),
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "seed-pool: refreshed anchor iRT from the band libraries"
    );
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed_cfg(fdr_seed: f64) -> SearchSeedConfig {
        SearchSeedConfig {
            fdr_seed,
            ..Default::default()
        }
    }

    fn write_seed(path: &str, cids: &[u32], scores: &[f64], labels: &[&str], irt: &[f32]) {
        let n = cids.len();
        write_table(
            path,
            vec![
                Col::U32("candidate_id".into(), cids.to_vec()),
                Col::Str(
                    "peptidoform".into(),
                    (0..n).map(|i| format!("PEP{i}K")).collect(),
                ),
                Col::I32("charge".into(), vec![2; n]),
                Col::F64("precursor_mz".into(), vec![500.0; n]),
                Col::U32("base_peptide_id".into(), cids.to_vec()),
                Col::Str("protein".into(), vec!["P".to_string(); n]),
                Col::Str(
                    "label".into(),
                    labels.iter().map(|s| s.to_string()).collect(),
                ),
                Col::F64("score".into(), scores.to_vec()),
                Col::F64("spectrum_q".into(), vec![1.0; n]),
                Col::F64(
                    "observed_rt".into(),
                    (0..n).map(|i| i as f64 * 10.0).collect(),
                ),
                Col::F32("predicted_irt".into(), irt.to_vec()),
                Col::I32("matched_peaks".into(), vec![5; n]),
                Col::U32("scan_index".into(), (0..n as u32).collect()),
            ],
        )
        .unwrap();
    }

    #[test]
    // No band writes a calibrant sidecar here, which is the pre-sidecar band directory:
    // the pool must still combine the bands' scalars and say so.
    fn pooled_seed_has_global_ids_one_q_scale_and_a_combined_calibration() {
        let dir = std::env::temp_dir().join(format!("mumdia_seed_pool_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.parquet").to_str().unwrap().to_string();
        let b = dir.join("b.parquet").to_str().unwrap().to_string();
        // Band a: local ids 0..3 at offset 0; band b: local ids 0..2 at offset 3, whose local
        // 0 overlaps band a's global 3 (window overlap) with a lower score.
        write_seed(
            &a,
            &[0, 1, 2, 3],
            &[30.0, 25.0, 20.0, 15.0],
            &["target", "decoy", "target", "target"],
            &[1.0, 2.0, 3.0, 4.0],
        );
        write_seed(
            &b,
            &[0, 1, 2],
            &[10.0, 28.0, 5.0],
            &["target", "target", "decoy"],
            &[9.0, 5.0, 6.0],
        );
        for (path, n_dev, off, tol) in [(&a, 30u64, 1.0, 8.0), (&b, 10u64, 3.0, 12.0)] {
            mumdia_io::json::write_json(
                &format!("{path}.masscal.json"),
                &json!({"frag_ppm_offset": off, "frag_tol_ppm": tol, "frag_ppm_sigma": tol, "n_dev": n_dev,
                        "cal_passes": 1, "ppm_residual_median": 0.0, "ppm_residual_mad": 1.0,
                        "mz_cal_grid_mz": [], "mz_cal_grid_ppm": []}),
            )
            .unwrap();
        }
        let out = dir.join("pooled.parquet").to_str().unwrap().to_string();
        let n = run(SeedPoolParams {
            seeds: &[
                BandSeed {
                    path: a.clone(),
                    offset: 0,
                    rows: 4,
                },
                BandSeed {
                    path: b.clone(),
                    offset: 3,
                    rows: 3,
                },
            ],
            masscals: &[format!("{a}.masscal.json"), format!("{b}.masscal.json")],
            calibrants: &[
                crate::masscal::calibrants_path(&a),
                crate::masscal::calibrants_path(&b),
            ],
            cfg: &seed_cfg(0.01),
            out: &out,
        })
        .unwrap();
        assert_eq!(n, 6, "seven rows in, one overlap collapsed");
        let t = TableFile::open(&out).unwrap();
        assert_eq!(t.u32("candidate_id").unwrap(), vec![0, 1, 2, 3, 4, 5]);
        let score = t.f64("score").unwrap();
        assert_eq!(
            score[3], 15.0,
            "band a's higher score kept for the shared candidate"
        );
        // q re-estimated on the union: two decoys among six, the top three targets are clean.
        let q = t.f64("spectrum_q").unwrap();
        let labels = t.str("label").unwrap();
        let pairs: Vec<(f64, bool)> = score
            .iter()
            .zip(&labels)
            .map(|(s, l)| (*s, l == "decoy"))
            .collect();
        assert_eq!(q, target_decoy_q(&pairs));
        let cal: serde_json::Value =
            mumdia_io::json::read_json(&format!("{out}.masscal.json")).unwrap();
        assert!(
            (cal["frag_ppm_offset"].as_f64().unwrap() - 1.5).abs() < 1e-9,
            "weighted by calibrants: (30*1 + 10*3)/40"
        );
        assert!((cal["frag_tol_ppm"].as_f64().unwrap() - 9.0).abs() < 1e-9);
        assert_eq!(cal["n_dev"].as_u64().unwrap(), 40);
        assert_eq!(cal["masscal_source"], "band_scalars");

        // Refresh: band b's re-predicted table says local 1 (global 4) now has iRT 50.
        let lib_b = dir.join("lib_b.parquet").to_str().unwrap().to_string();
        write_table(
            &lib_b,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1, 2]),
                Col::F32("predicted_irt".into(), vec![40.0, 50.0, 60.0]),
            ],
        )
        .unwrap();
        let refreshed = dir.join("refreshed.parquet").to_str().unwrap().to_string();
        refresh_irt(&out, &refreshed, &[(lib_b, 3)]).unwrap();
        let irt = TableFile::open(&refreshed)
            .unwrap()
            .f32("predicted_irt")
            .unwrap();
        assert_eq!(irt, vec![1.0, 2.0, 3.0, 40.0, 50.0, 60.0]);
    }

    /// The per-row map over every row for every band, verbatim: the reference the sliced,
    /// direct-indexed `refresh_irt` is compared against byte for byte.
    fn refresh_irt_reference(seed_in: &str, seed_out: &str, bands: &[(String, u32)]) {
        let mut rows = read_rows(seed_in, 0).unwrap();
        let q = TableFile::open(seed_in).unwrap().f64("spectrum_q").unwrap();
        for (path, offset) in bands {
            let b = TableFile::open(path).unwrap();
            let cid = b.u32("candidate_id").unwrap();
            let irt = b.f32("predicted_irt").unwrap();
            let n = b.nrows;
            let by_local: HashMap<u32, f32> = cid.into_iter().zip(irt).collect();
            for r in rows.iter_mut() {
                let Some(local) = r.cid.checked_sub(*offset) else {
                    continue;
                };
                if (local as usize) < n {
                    if let Some(v) = by_local.get(&local) {
                        r.irt = *v;
                    }
                }
            }
        }
        write_rows(seed_out, &rows, q).unwrap();
    }

    /// `refresh_irt` over a seed sorted by candidate (the pooled seed) and over an
    /// unsorted one, with overlapping bands (the later wins), a band ending past the last
    /// row, a band starting at 0, and a band table whose ids are not its rows (shuffled
    /// and with a repeat): the bytes of the old whole-row, map-lookup version.
    #[test]
    fn the_sliced_refresh_writes_what_the_per_row_map_wrote() {
        let dir =
            std::env::temp_dir().join(format!("mumdia_seed_pool_refresh_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = |t: &str| dir.join(t).to_str().unwrap().to_string();
        let n = 40usize;
        let sorted: Vec<u32> = (0..n as u32).map(|i| i * 2 + 1).collect();
        let mut unsorted = sorted.clone();
        unsorted.reverse();
        unsorted.swap(3, 17);
        let band = |tag: &str, ids: Vec<u32>, irt0: f32| -> String {
            let p = path(tag);
            let k = ids.len();
            write_table(
                &p,
                vec![
                    Col::U32("candidate_id".into(), ids),
                    Col::F32(
                        "predicted_irt".into(),
                        (0..k).map(|i| irt0 + i as f32).collect(),
                    ),
                ],
            )
            .unwrap();
            p
        };
        let bands = vec![
            (band("b0.parquet", (0..30).collect(), 100.0), 0u32),
            // Overlaps the first band from global 20: the later band wins there.
            (band("b1.parquet", (0..25).collect(), 500.0), 20),
            // Runs past the last seed row.
            (band("b2.parquet", (0..60).collect(), 900.0), 50),
            // Ids that are not the rows: shuffled, with local 3 twice (last wins).
            (band("b3.parquet", vec![5, 3, 0, 3, 9, 1], 2000.0), 60),
        ];
        for (tag, cids) in [("sorted", &sorted), ("unsorted", &unsorted)] {
            let seed_in = path(&format!("seed_{tag}.parquet"));
            let labels: Vec<&str> = (0..n)
                .map(|i| if i % 3 == 0 { "decoy" } else { "target" })
                .collect();
            let scores: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();
            let irt: Vec<f32> = (0..n).map(|i| i as f32).collect();
            write_seed(&seed_in, cids, &scores, &labels, &irt);
            let (a, b) = (
                path(&format!("new_{tag}.parquet")),
                path(&format!("ref_{tag}.parquet")),
            );
            refresh_irt(&seed_in, &a, &bands).unwrap();
            refresh_irt_reference(&seed_in, &b, &bands);
            assert_eq!(
                std::fs::read(&a).unwrap(),
                std::fs::read(&b).unwrap(),
                "{tag}: the refreshed seed differs from the per-row map's"
            );
            let changed = TableFile::open(&a).unwrap().f32("predicted_irt").unwrap();
            assert!(
                changed.iter().zip(&irt).any(|(x, y)| x != y),
                "{tag}: no row was refreshed, so the comparison proves nothing"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uncalibrated_bands_pool_to_the_configured_tolerance_not_zero() {
        let dir =
            std::env::temp_dir().join(format!("mumdia_seed_pool_uncal_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.parquet").to_str().unwrap().to_string();
        write_seed(&a, &[0, 1], &[3.0, 2.0], &["target", "target"], &[1.0, 2.0]);
        // search-seed's failure branch: no calibrants, the configured 20 ppm written as is.
        mumdia_io::json::write_json(
            &format!("{a}.masscal.json"),
            &json!({"frag_ppm_offset": 0.0, "frag_tol_ppm": 20.0, "frag_ppm_sigma": 20.0,
                    "n_dev": 0, "cal_passes": 0, "ppm_residual_median": 0.0,
                    "ppm_residual_mad": 0.0, "mz_cal_grid_mz": [], "mz_cal_grid_ppm": []}),
        )
        .unwrap();
        let out = dir.join("pooled.parquet").to_str().unwrap().to_string();
        run(SeedPoolParams {
            seeds: &[BandSeed {
                path: a.clone(),
                offset: 0,
                rows: 2,
            }],
            masscals: &[format!("{a}.masscal.json")],
            calibrants: &[crate::masscal::calibrants_path(&a)],
            cfg: &seed_cfg(0.01),
            out: &out,
        })
        .unwrap();
        let cal: serde_json::Value =
            mumdia_io::json::read_json(&format!("{out}.masscal.json")).unwrap();
        assert_eq!(cal["frag_tol_ppm"].as_f64().unwrap(), 20.0);
        assert_eq!(cal["frag_ppm_sigma"].as_f64().unwrap(), 20.0);
        assert_eq!(cal["n_dev"].as_u64().unwrap(), 0);
    }

    /// Three bands with their calibrant sidecars, built so the two q scales disagree.
    ///
    /// Each band holds 100 candidates on its own score range (band A the highest, band C
    /// the lowest) and bands A and B carry two decoys at their bottom. On the POOLED
    /// scale that puts band A's 98 targets at q = 1/98, band B's at 3/196 and band C's at
    /// 5/296; on each band's OWN scale every target sits at 1/98 or 1/100. So a threshold
    /// of 0.012 accepts every band's calibrants under the band q and only band A's under
    /// the pooled q, while 0.017 accepts all three under both.
    struct Fixture {
        seeds: Vec<BandSeed>,
        masscals: Vec<String>,
        calibrants: Vec<String>,
        /// Per band, in sidecar row order, the deviations and fragment m/z it carries,
        /// already rounded through `f32` as the sidecar stores them.
        devs: Vec<Vec<f64>>,
        mz: Vec<Vec<f64>>,
        out: String,
    }

    fn three_bands(tag: &str) -> Fixture {
        let dir =
            std::env::temp_dir().join(format!("mumdia_seed_pool_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut f = Fixture {
            seeds: Vec::new(),
            masscals: Vec::new(),
            calibrants: Vec::new(),
            devs: Vec::new(),
            mz: Vec::new(),
            out: dir.join("pooled.parquet").to_str().unwrap().to_string(),
        };
        // (rows, decoys at the bottom, top score, deviation centre, deviation step): the
        // three deviation populations differ, so a fit over one band is not a fit over
        // the union and the assertions below can tell them apart.
        let spec = [
            (100usize, 2usize, 1000.0f64, -2.0f64, 1.0f64),
            (100, 2, 900.0, -2.0, 2.0),
            (100, 0, 800.0, 8.0, 3.0),
        ];
        for (bi, &(n, n_dec, top, centre, step)) in spec.iter().enumerate() {
            let offset = (bi * 100) as u32;
            let path = dir
                .join(format!("g{bi}.parquet"))
                .to_str()
                .unwrap()
                .to_string();
            let cids: Vec<u32> = (0..n as u32).collect();
            let scores: Vec<f64> = (0..n).map(|i| top - i as f64).collect();
            let labels: Vec<&str> = (0..n)
                .map(|i| if i >= n - n_dec { "decoy" } else { "target" })
                .collect();
            let irt: Vec<f32> = (0..n).map(|i| i as f32).collect();
            write_seed(&path, &cids, &scores, &labels, &irt);
            // Two calibrant deviations per TARGET row, which is what this band's own q
            // accepts, keyed by library-wide id and by the row's scan.
            let mut c = crate::masscal::Calibrants::default();
            let (mut d, mut m) = (Vec::new(), Vec::new());
            for i in 0..n - n_dec {
                for k in 0..2usize {
                    let ppm = (centre + ((((i * 2 + k) % 11) as f64) - 5.0) * step) as f32;
                    let fmz = (300.0 + (((i + k) % 11) as f64) * 50.0) as f32;
                    c.candidate_id.push(offset + i as u32);
                    c.scan_index.push(i as u32);
                    c.frag_mz.push(fmz);
                    c.ppm.push(ppm);
                    d.push(ppm as f64);
                    m.push(fmz as f64);
                }
            }
            crate::masscal::write_calibrants(&path, c).unwrap();
            // The band's own scalar fit, so the fallback path has something to combine
            // and the pooled path can be shown not to be it.
            let band = MassCal::fit_from(&d, &m, &seed_cfg(0.012));
            mumdia_io::json::write_json(&crate::masscal::json_path(&path), &band.to_json())
                .unwrap();
            f.masscals.push(crate::masscal::json_path(&path));
            f.calibrants.push(crate::masscal::calibrants_path(&path));
            f.seeds.push(BandSeed {
                path,
                offset,
                rows: n as u32,
            });
            f.devs.push(d);
            f.mz.push(m);
        }
        f
    }

    #[test]
    fn the_pooled_tolerance_is_the_estimator_applied_to_the_concatenated_deviations() {
        // The fix in one assertion: three bands' deviations are fitted ONCE, and the
        // result is what `search-seed`'s own estimator gives on the concatenation. A band
        // does not get to contribute its p95, only its points.
        let f = three_bands("concat");
        let cfg = seed_cfg(0.017);
        run(SeedPoolParams {
            seeds: &f.seeds,
            masscals: &f.masscals,
            calibrants: &f.calibrants,
            cfg: &cfg,
            out: &f.out,
        })
        .unwrap();
        let got: serde_json::Value =
            mumdia_io::json::read_json(&crate::masscal::json_path(&f.out)).unwrap();
        let all_devs: Vec<f64> = f.devs.iter().flatten().cloned().collect();
        let all_mz: Vec<f64> = f.mz.iter().flatten().cloned().collect();
        let want = MassCal::fit_from(&all_devs, &all_mz, &cfg);
        assert_eq!(got["masscal_source"], "pooled_deviations");
        assert_eq!(got["n_dev"].as_u64().unwrap(), all_devs.len() as u64);
        assert_eq!(got["frag_tol_ppm"].as_f64().unwrap(), want.frag_tol_ppm);
        assert_eq!(got["frag_ppm_sigma"].as_f64().unwrap(), want.frag_tol_ppm);
        assert_eq!(
            got["frag_ppm_offset"].as_f64().unwrap(),
            want.frag_ppm_offset
        );
        assert_eq!(
            got["ppm_residual_mad"].as_f64().unwrap(),
            want.ppm_residual_mad
        );
        assert_eq!(got["cal_passes"].as_u64().unwrap(), 1);
        assert_eq!(got["pooled_from_groups"].as_u64().unwrap(), 3);
        // And it is not the calibrant-weighted mean of the bands' own tolerances, which
        // is what the stage used to write.
        let (mut w, mut wt) = (0.0f64, 0.0f64);
        for (d, m) in f.devs.iter().zip(&f.mz) {
            let band = MassCal::fit_from(d, m, &cfg);
            w += d.len() as f64;
            wt += d.len() as f64 * band.frag_tol_ppm;
        }
        let averaged = wt / w;
        assert!(
            (averaged - want.frag_tol_ppm).abs() > 1e-6,
            "the scalar combination ({averaged}) is a different number from the pooled fit \
             ({}), which is the defect",
            want.frag_tol_ppm
        );

        // The optional m/z-dependent grid is fitted from the pooled deviations too, so it
        // keeps working on this path rather than being silently dropped.
        let mut loess = seed_cfg(0.017);
        loess.mass_cal_loess = true;
        run(SeedPoolParams {
            seeds: &f.seeds,
            masscals: &f.masscals,
            calibrants: &f.calibrants,
            cfg: &loess,
            out: &f.out,
        })
        .unwrap();
        let grid: serde_json::Value =
            mumdia_io::json::read_json(&crate::masscal::json_path(&f.out)).unwrap();
        let gm = grid["mz_cal_grid_mz"].as_array().unwrap();
        assert!(gm.len() >= 2, "the pooled path fits the LOESS grid");
        assert_eq!(gm.len(), grid["mz_cal_grid_ppm"].as_array().unwrap().len());
        // And the robust second pass, which also needs the deviations.
        let mut two = seed_cfg(0.017);
        two.two_pass_mass_cal = true;
        run(SeedPoolParams {
            seeds: &f.seeds,
            masscals: &f.masscals,
            calibrants: &f.calibrants,
            cfg: &two,
            out: &f.out,
        })
        .unwrap();
        let second: serde_json::Value =
            mumdia_io::json::read_json(&crate::masscal::json_path(&f.out)).unwrap();
        assert_eq!(second["cal_passes"].as_u64().unwrap(), 2);
        assert_eq!(
            second["frag_tol_ppm"].as_f64().unwrap(),
            MassCal::fit_from(&all_devs, &all_mz, &two).frag_tol_ppm
        );
    }

    #[test]
    fn the_pooled_q_and_not_the_band_q_selects_the_calibrants() {
        let f = three_bands("poolq");
        let cfg = seed_cfg(0.012);
        // Every band's OWN q accepts every one of its targets at this threshold, so the
        // sidecars carry all three bands' deviations.
        for s in &f.seeds {
            let t = TableFile::open(&s.path).unwrap();
            let sc = t.f64("score").unwrap();
            let lb = t.str("label").unwrap();
            let pairs: Vec<(f64, bool)> = sc
                .iter()
                .zip(&lb)
                .map(|(x, l)| (*x, l == "decoy"))
                .collect();
            let bq = target_decoy_q(&pairs);
            let worst = bq
                .iter()
                .zip(&lb)
                .filter(|(_, l)| *l == "target")
                .fold(0.0f64, |a, (q, _)| a.max(*q));
            assert!(
                worst <= cfg.fdr_seed,
                "band {} puts its worst target at {worst}, which its own q accepts",
                s.offset
            );
        }
        run(SeedPoolParams {
            seeds: &f.seeds,
            masscals: &f.masscals,
            calibrants: &f.calibrants,
            cfg: &cfg,
            out: &f.out,
        })
        .unwrap();
        // The pooled q does not: only band A's targets clear 0.012 on the union.
        let pooled = TableFile::open(&f.out).unwrap();
        let pq = pooled.f64("spectrum_q").unwrap();
        let plab = pooled.str("label").unwrap();
        let accepted: Vec<u32> = pooled
            .u32("candidate_id")
            .unwrap()
            .into_iter()
            .zip(pq.iter().zip(&plab))
            .filter(|(_, (q, l))| **q <= cfg.fdr_seed && *l == "target")
            .map(|(c, _)| c)
            .collect();
        assert_eq!(accepted.len(), 98, "band A's targets and nothing else");
        assert!(accepted.iter().all(|&c| c < 100));

        let got: serde_json::Value =
            mumdia_io::json::read_json(&crate::masscal::json_path(&f.out)).unwrap();
        let want = MassCal::fit_from(&f.devs[0], &f.mz[0], &cfg);
        assert_eq!(got["masscal_source"], "pooled_deviations");
        assert_eq!(
            got["n_dev"].as_u64().unwrap(),
            f.devs[0].len() as u64,
            "only the calibrants of the PSMs the pooled q accepts"
        );
        assert_eq!(got["frag_tol_ppm"].as_f64().unwrap(), want.frag_tol_ppm);
        assert_eq!(
            got["frag_ppm_offset"].as_f64().unwrap(),
            want.frag_ppm_offset
        );
        // Selecting on each band's own q instead would have taken every sidecar row and
        // fitted a different, wider tolerance.
        let all_devs: Vec<f64> = f.devs.iter().flatten().cloned().collect();
        let all_mz: Vec<f64> = f.mz.iter().flatten().cloned().collect();
        let band_q_selection = MassCal::fit_from(&all_devs, &all_mz, &cfg);
        assert!(
            band_q_selection.frag_tol_ppm > want.frag_tol_ppm,
            "the band-q selection is the wider fit: {} against {}",
            band_q_selection.frag_tol_ppm,
            want.frag_tol_ppm
        );
    }

    #[test]
    fn a_band_without_a_sidecar_still_pools_on_the_scalars() {
        // An older band directory has no calibrant sidecar. That must not fail the run:
        // the scalars combine as they always did, and the log says so.
        let f = three_bands("legacy");
        std::fs::remove_file(&f.calibrants[1]).unwrap();
        let cfg = seed_cfg(0.017);
        run(SeedPoolParams {
            seeds: &f.seeds,
            masscals: &f.masscals,
            calibrants: &f.calibrants,
            cfg: &cfg,
            out: &f.out,
        })
        .unwrap();
        let got: serde_json::Value =
            mumdia_io::json::read_json(&crate::masscal::json_path(&f.out)).unwrap();
        assert_eq!(got["masscal_source"], "band_scalars");
        let expected = combine_band_scalars(&SeedPoolParams {
            seeds: &f.seeds,
            masscals: &f.masscals,
            calibrants: &f.calibrants,
            cfg: &cfg,
            out: &f.out,
        })
        .unwrap();
        assert_eq!(
            got["frag_tol_ppm"].as_f64().unwrap(),
            expected.frag_tol_ppm,
            "the calibrant-weighted mean of the bands' own tolerances"
        );
        assert_eq!(got["n_dev"].as_u64().unwrap(), expected.n_dev);
    }
}
