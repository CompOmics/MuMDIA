//! Pool the seed searches of a run's isolation-window groups into one seed table
//! (`groups.window_groups`, `groups.calibration = global`).
//!
//! Each group seeds its own library band with band-local `candidate_id`s and its own
//! `spectrum_q`, estimated on that band's PSMs alone. The retention-time calibration wants
//! the whole run's anchors on one q scale, so the pooled table carries library-wide ids
//! (local id plus the band's first row), a `spectrum_q` re-estimated over the union with the
//! same target-decoy kernel the seed uses, and one mass calibration combined from the bands'
//! by their calibrant counts. Rows stay one best PSM per candidate, as the seed writes them:
//! bands are disjoint in candidates except where windows overlap across a cut, and there the
//! higher score wins.
//!
//! After the library band iRTs have been re-predicted against the pooled anchors
//! (multi-head calibration per band), [`refresh_irt`] copies each anchor's current iRT from
//! its band's table into the pooled seed, so `rt-im-train` can take anchors' iRT from the
//! seed (`anchor_irt_from_seed`) while looking at one band's library at a time.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use mumdia_io::table::{write_table, Col, TableFile};
use serde_json::json;
use tracing::{info, warn};

use crate::fdr::target_decoy_q;

/// One group's seed table and the library band it searched.
pub struct BandSeed {
    pub path: String,
    /// The band's first library row: local id plus this is the library-wide id.
    pub offset: u32,
    /// Rows of the band; the band's candidates are `offset..offset + rows`.
    pub rows: u32,
    /// Where to write the band's view of the pooled seed: its own rows with band-local
    /// ids and the pooled `spectrum_q`, for the band's stages that key on the seed by
    /// candidate id (features' corroboration and elution boundary).
    pub view: String,
}

pub struct SeedPoolParams<'a> {
    pub seeds: &'a [BandSeed],
    /// The groups' `<seed>.masscal.json`, same order as `seeds`.
    pub masscals: &'a [String],
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
    // One row per library-wide candidate: where two bands share a candidate (window
    // overlap across a cut) the higher score stays, so the pooled q sees each once.
    let mut best: HashMap<u32, Row> = HashMap::new();
    let mut n_in = 0usize;
    for band in p.seeds {
        for r in read_rows(&band.path, band.offset)? {
            n_in += 1;
            match best.get(&r.cid) {
                Some(b) if b.score >= r.score => {}
                _ => {
                    best.insert(r.cid, r);
                }
            }
        }
    }
    let mut rows: Vec<Row> = best.into_values().collect();
    rows.sort_by_key(|r| r.cid);
    crate::fdr::validate_labels(&rows.iter().map(|r| r.label.clone()).collect::<Vec<_>>())?;
    let pairs: Vec<(f64, bool)> = rows.iter().map(|r| (r.score, r.label == "decoy")).collect();
    let q = target_decoy_q(&pairs);
    let n_out = rows.len();
    let n = write_rows(p.out, &rows, q.clone())?;
    // Each band's rows again, with local ids and the pooled q.
    for band in p.seeds {
        let (view_rows, view_q): (Vec<Row>, Vec<f64>) = rows
            .iter()
            .zip(&q)
            .filter(|(r, _)| r.cid >= band.offset && r.cid - band.offset < band.rows)
            .map(|(r, q)| {
                (
                    Row {
                        cid: r.cid - band.offset,
                        ..r.clone()
                    },
                    *q,
                )
            })
            .unzip();
        write_rows(&band.view, &view_rows, view_q)?;
    }

    // Mass calibration: the bands' scalar offsets and learned tolerances combined by
    // calibrant count. The optional m/z grids are not combined (they would need the
    // deviations, which the seed does not keep); extract then applies the scalar offset.
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
    mumdia_io::json::write_json(
        &format!("{}.masscal.json", p.out),
        &json!({
            "frag_ppm_offset": off,
            "frag_tol_ppm": tol,
            "frag_ppm_sigma": tol,
            "n_dev": n_dev,
            "cal_passes": passes,
            "ppm_residual_median": med,
            "ppm_residual_mad": mad,
            "mz_cal_grid_mz": Vec::<f64>::new(),
            "mz_cal_grid_ppm": Vec::<f64>::new(),
            "pooled_from_groups": p.seeds.len(),
        }),
    )?;
    info!(
        groups = p.seeds.len(),
        rows_in = n_in,
        rows_out = n_out,
        frag_ppm_offset = off,
        frag_tol_ppm = tol,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "seed-pool: done"
    );
    Ok(n)
}

/// Rewrite `seed_in` to `seed_out` with each row's `predicted_irt` taken from the band
/// table that holds its candidate (`bands`: precursor table with band-local ids, and the
/// band's first library row). Rows outside every band keep their value.
pub fn refresh_irt(seed_in: &str, seed_out: &str, bands: &[(String, u32)]) -> Result<u64> {
    let t0 = Instant::now();
    let mut rows = read_rows(seed_in, 0)?;
    let t = TableFile::open(seed_in)?;
    let q = t.f64("spectrum_q")?;
    let mut refreshed = 0usize;
    for (path, offset) in bands {
        let b = TableFile::open(path)?;
        let cid = b.u32("candidate_id")?;
        let irt = b.f32("predicted_irt")?;
        let n = b.nrows;
        let by_local: HashMap<u32, f32> = cid.into_iter().zip(irt).collect();
        for r in rows.iter_mut() {
            let Some(local) = r.cid.checked_sub(*offset) else {
                continue;
            };
            if (local as usize) < n {
                if let Some(v) = by_local.get(&local) {
                    r.irt = *v;
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
        let view = |tag: &str| {
            dir.join(format!("{tag}_view.parquet"))
                .to_str()
                .unwrap()
                .to_string()
        };
        let n = run(SeedPoolParams {
            seeds: &[
                BandSeed {
                    path: a.clone(),
                    offset: 0,
                    rows: 4,
                    view: view("a"),
                },
                BandSeed {
                    path: b.clone(),
                    offset: 3,
                    rows: 3,
                    view: view("b"),
                },
            ],
            masscals: &[format!("{a}.masscal.json"), format!("{b}.masscal.json")],
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

        // Band views: the band's own rows, local ids, the pooled q (band b's local 0 is
        // the shared candidate, which the pool holds with band a's score).
        let va = TableFile::open(&view("a")).unwrap();
        assert_eq!(va.u32("candidate_id").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(va.f64("spectrum_q").unwrap(), q[0..4].to_vec());
        let vb = TableFile::open(&view("b")).unwrap();
        assert_eq!(vb.u32("candidate_id").unwrap(), vec![0, 1, 2]);
        assert_eq!(vb.f64("spectrum_q").unwrap(), q[3..6].to_vec());
        assert_eq!(vb.f64("score").unwrap()[0], 15.0);

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
        let view = dir.join("a_view.parquet").to_str().unwrap().to_string();
        run(SeedPoolParams {
            seeds: &[BandSeed {
                path: a.clone(),
                offset: 0,
                rows: 2,
                view,
            }],
            masscals: &[format!("{a}.masscal.json")],
            out: &out,
        })
        .unwrap();
        let cal: serde_json::Value =
            mumdia_io::json::read_json(&format!("{out}.masscal.json")).unwrap();
        assert_eq!(cal["frag_tol_ppm"].as_f64().unwrap(), 20.0);
        assert_eq!(cal["frag_ppm_sigma"].as_f64().unwrap(), 20.0);
        assert_eq!(cal["n_dev"].as_u64().unwrap(), 0);
    }
}
