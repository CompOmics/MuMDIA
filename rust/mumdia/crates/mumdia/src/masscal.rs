//! Fragment mass calibration: the estimator, the `<seed>.masscal.json` artifact, and the
//! calibrant sidecar that lets a grouped run fit the estimator ONCE over the whole run.
//!
//! `search-seed` learns a fragment tolerance from the ppm deviations of the matched
//! fragments of its confident target PSMs: the median is the systematic offset and
//! `1.5 * p95(|dev - median|)` the tolerance. That estimator lives here because it is
//! fitted in two places and must be the same fit in both.
//!
//! Under an isolation-window-group search (`groups.window_groups`) each band seeds its own
//! slice of the library, and combining the bands' fitted SCALARS is not the same estimator
//! as fitting the union. It is systematically wider, for two reasons that compound:
//!
//! 1. a p95 estimated on a band's ~2,000 deviations has a heavier tail than the p95 of the
//!    union, and averaging those per-band p95s does not recover the union's;
//! 2. each band selects its calibrants on its OWN `spectrum_q`, whose finite-sample floor
//!    is looser than the pooled one, so bands admit PSMs the pooled q rejects.
//!
//! Measured on the six-file HYE Astral benchmark, 100 bands against none: the offsets agree
//! (-1.8834 against -1.8486 ppm) while the tolerance comes out 35% wider (11.400 against
//! 8.452 ppm), extract then accepts 33% more candidates and the run returns 3.4% fewer
//! peptides at 1%. So the bands write their calibrant deviations beside the masscal
//! ([`write_calibrants`]) and `seed-pool` fits [`MassCal::fit_from`] once over the union,
//! filtered on the POOLED q, exactly as the retention-time calibration already uses the
//! pooled anchors under `groups.calibration = global`.

use anyhow::Result;
use mumdia_core::config::SearchSeedConfig;
use mumdia_io::table::{write_table, Col, TableFile};
use serde_json::json;

/// Fewest calibrant deviations a tolerance is fitted from. Below this the configured
/// search tolerance is kept rather than a percentile of a handful of points.
pub const MIN_CALIBRANTS: usize = 20;

/// Fewest calibrant deviations the optional m/z-dependent (LOESS) grid is fitted from.
pub const MIN_LOESS_CALIBRANTS: usize = 50;

// Every target is offered.
//
// A band estimates `spectrum_q` on its own PSMs, and that estimate is not the pooled one
// in either direction. On the HYE Astral benchmark at 100 bands it is looser (106,088
// band-confident against 97,584 pooled), which is half of why the banded tolerance came
// out wide, and `seed-pool` re-selecting on the pooled q removes that. On a band with few
// targets, or a band whose low-scoring targets sit among decoys, it is STRICTER instead:
// on the CI fixture at three bands every band's own q rejects every one of its targets.
//
// So the sidecar has to carry a superset of what the pooled q will accept, and no rule a
// band can apply on its own data gives one. The pooled threshold moves with the other
// bands: a band of clean, high-scoring targets lowers it, and then the pool accepts
// targets of a noisier band at a score where that band's own q is several times the
// threshold. There was such a rule until 2026-09-25, a fixed prefix of each band's 2,000
// best targets, sized from the 100-band benchmark, where about 1,000 per band were
// pooled-accepted. It was exact from 8 bands up and short below: on the six-file HYE
// Astral benchmark the pooled fit saw 167,418 deviations at 2 bands and 177,380 at 4,
// against the unbanded 181,196, and fitted 7.76 and 8.24 ppm against 8.45, because a
// band of a 2-band plan has tens of thousands of pooled-accepted targets. Every target
// PSM of the band is therefore offered, and `seed-pool` keeps the ones the pooled q
// accepts, so the pooled fit is the unbanded fit at any band count
// (`tests/pipeline.rs`, `a_two_band_pooled_mass_calibration_equals_the_unbanded_fit`).
// The sidecar stays 16 B per deviation, now one row per matched fragment of every target
// PSM of the band rather than of its best 2,000; `seed-pool` reads the sidecars one band
// at a time and keeps only the accepted deviations.

/// Median offset and `1.5 * p95(|dev - median|)` tolerance, floored at 5 ppm.
///
/// Panics on an empty slice; every caller goes through [`MassCal::fit_from`], which
/// guards on [`MIN_CALIBRANTS`].
pub fn fit(devs: &[f64]) -> (f64, f64) {
    let mut sorted = devs.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let offset = sorted[sorted.len() / 2];
    let centered: Vec<f64> = devs.iter().map(|x| (x - offset).abs()).collect();
    let tol = (crate::calibrate::percentile(&centered, 0.95) * 1.5).max(5.0);
    (offset, tol)
}

/// One fragment mass calibration: what `<seed>.masscal.json` carries and what `extract`
/// reads back.
#[derive(Clone, Debug)]
pub struct MassCal {
    pub frag_ppm_offset: f64,
    pub frag_tol_ppm: f64,
    pub n_dev: u64,
    pub cal_passes: u64,
    /// Post-correction residual median: how far the single offset left the mass axis
    /// off zero. Diagnostic only.
    pub ppm_residual_median: f64,
    /// Post-correction residual MAD: the achieved precision. Diagnostic only.
    pub ppm_residual_mad: f64,
    /// Optional m/z-dependent correction grid (empty = scalar offset only).
    pub mz_cal_grid_mz: Vec<f64>,
    pub mz_cal_grid_ppm: Vec<f64>,
}

impl MassCal {
    /// Fit offset, tolerance and diagnostics from calibrant deviations.
    ///
    /// `devs` are ppm deviations of matched fragments of confident target PSMs and
    /// `dev_mz` the fragment m/z paired to each, in the same order. The configured
    /// `cfg.fragment_tol_ppm` is what a run with too few calibrants keeps.
    ///
    /// `cfg.two_pass_mass_cal` re-fits on the deviations inside the first-pass window, so
    /// random-match outliers cannot bias the offset. `cfg.mass_cal_loess` additionally
    /// samples a LOESS of deviation against fragment m/z on a fixed 25-Th grid, which
    /// `extract` interpolates per peak. Both read the deviations, so both work wherever
    /// the deviations are, which is the point of the calibrant sidecar.
    pub fn fit_from(devs: &[f64], dev_mz: &[f64], cfg: &SearchSeedConfig) -> MassCal {
        debug_assert_eq!(devs.len(), dev_mz.len());
        let (frag_ppm_offset, frag_tol_ppm, cal_passes) = if devs.len() >= MIN_CALIBRANTS {
            let (o1, t1) = fit(devs);
            if cfg.two_pass_mass_cal {
                // Second pass: keep only deviations inside the first-pass window, so
                // random-match outliers cannot bias the offset, then re-fit.
                let inl: Vec<f64> = devs
                    .iter()
                    .cloned()
                    .filter(|d| (d - o1).abs() <= t1)
                    .collect();
                if inl.len() >= MIN_CALIBRANTS {
                    let (o2, t2) = fit(&inl);
                    (o2, t2, 2)
                } else {
                    (o1, t1, 1)
                }
            } else {
                (o1, t1, 1)
            }
        } else {
            (0.0, cfg.fragment_tol_ppm, 0)
        };
        // Calibration-quality residual stats for the mass dimension: the median and MAD of
        // the calibrant deviations AFTER the offset correction. A residual median far from
        // zero means the single offset did not fully de-bias the mass axis (a case for an
        // m/z-dependent calibration); the MAD is the achieved precision. Purely
        // diagnostic; consumed by the per-run calibration-quality report, never by
        // extraction.
        let (ppm_residual_median, ppm_residual_mad) = if devs.is_empty() {
            (0.0, 0.0)
        } else {
            let centered: Vec<f64> = devs.iter().map(|d| d - frag_ppm_offset).collect();
            let med = crate::calibrate::percentile(&centered, 0.5);
            let absdev: Vec<f64> = centered.iter().map(|c| (c - med).abs()).collect();
            (med, crate::calibrate::percentile(&absdev, 0.5))
        };
        // Optional m/z-dependent (LOESS) mass calibration grid: fit the calibrant ppm
        // deviation versus fragment m/z and sample it on a fixed 25-Th grid that extract
        // interpolates per peak. Empty unless `mass_cal_loess` is set and enough
        // calibrants exist; extract then falls back to the scalar offset. Deterministic:
        // pairs are sorted by m/z before the fit.
        let (mz_cal_grid_mz, mz_cal_grid_ppm): (Vec<f64>, Vec<f64>) =
            if cfg.mass_cal_loess && dev_mz.len() >= MIN_LOESS_CALIBRANTS {
                let mut pairs: Vec<(f64, f64)> =
                    dev_mz.iter().cloned().zip(devs.iter().cloned()).collect();
                pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
                let xs: Vec<f64> = pairs.iter().map(|(m, _)| *m).collect();
                let ys: Vec<f64> = pairs.iter().map(|(_, pp)| *pp).collect();
                let loess = crate::calibrate::Loess::fit(&xs, &ys, 0.3, 200);
                let lo = xs.first().copied().unwrap_or(150.0);
                let hi = xs.last().copied().unwrap_or(2000.0);
                let step = 25.0;
                let n = (((hi - lo) / step).floor() as usize).max(1);
                let gm: Vec<f64> = (0..=n).map(|k| lo + k as f64 * step).collect();
                let gp: Vec<f64> = gm.iter().map(|&m| loess.predict(m)).collect();
                (gm, gp)
            } else {
                (Vec::new(), Vec::new())
            };
        MassCal {
            frag_ppm_offset,
            frag_tol_ppm,
            n_dev: devs.len() as u64,
            cal_passes,
            ppm_residual_median,
            ppm_residual_mad,
            mz_cal_grid_mz,
            mz_cal_grid_ppm,
        }
    }

    /// The `<seed>.masscal.json` body. `extract` reads these keys.
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "frag_ppm_offset": self.frag_ppm_offset,
            "frag_tol_ppm": self.frag_tol_ppm,
            // The learned tolerance is the local mass-uncertainty estimate.
            "frag_ppm_sigma": self.frag_tol_ppm,
            "n_dev": self.n_dev,
            "cal_passes": self.cal_passes,
            // Calibration-quality diagnostics (post-correction residuals).
            "ppm_residual_median": self.ppm_residual_median,
            "ppm_residual_mad": self.ppm_residual_mad,
            // Optional m/z-dependent correction grid (empty = scalar offset only).
            "mz_cal_grid_mz": self.mz_cal_grid_mz,
            "mz_cal_grid_ppm": self.mz_cal_grid_ppm,
        })
    }
}

/// `<seed>.masscal.json`, beside the seed table.
pub fn json_path(seed_out: &str) -> String {
    format!("{seed_out}.masscal.json")
}

/// `<seed>.masscal.parquet`, the calibrant deviation sidecar beside the masscal.
///
/// Absent from an ungrouped run, which fits on the deviations it still holds in memory,
/// and absent from a band directory written before this file existed. `seed-pool` treats
/// its absence as "combine the bands' scalars", which is what it always did.
pub fn calibrants_path(seed_out: &str) -> String {
    format!("{seed_out}.masscal.parquet")
}

/// One band's calibrant deviations: one row per matched fragment of a confident target
/// PSM. Small and typed -- 16 B per deviation, about 3 MB for a whole run's 200k.
#[derive(Default, Clone, Debug)]
pub struct Calibrants {
    /// LIBRARY-WIDE candidate id (a band's local id plus its fragment offset), so the
    /// pooled seed can look the PSM up on its pooled q.
    pub candidate_id: Vec<u32>,
    /// The scan the PSM was matched on. Where two bands share a candidate (window overlap
    /// across a band cut) only the PSM the pool KEPT contributes its fragments, so a
    /// pooled fit sees one PSM per candidate exactly as an ungrouped fit does.
    pub scan_index: Vec<u32>,
    /// Fragment m/z, for the optional m/z-dependent grid. `f32` is the library's own
    /// storage width, so this round-trips exactly.
    pub frag_mz: Vec<f32>,
    /// Signed ppm deviation of the observed peak from the library fragment.
    pub ppm: Vec<f32>,
}

impl Calibrants {
    pub fn len(&self) -> usize {
        self.ppm.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ppm.is_empty()
    }
}

/// Write the calibrant sidecar for `seed_out`. Always written when a grouped band asks for
/// it, empty included: an existing empty sidecar says "this band found no calibrants",
/// which is a different fact from a missing one.
pub fn write_calibrants(seed_out: &str, c: &Calibrants) -> Result<u64> {
    write_table(
        &calibrants_path(seed_out),
        vec![
            Col::U32("candidate_id".into(), c.candidate_id.clone()),
            Col::U32("scan_index".into(), c.scan_index.clone()),
            Col::F32("frag_mz".into(), c.frag_mz.clone()),
            Col::F32("ppm".into(), c.ppm.clone()),
        ],
    )
}

/// Read a calibrant sidecar written by [`write_calibrants`].
pub fn read_calibrants(path: &str) -> Result<Calibrants> {
    let t = TableFile::open(path)?;
    Ok(Calibrants {
        candidate_id: t.u32("candidate_id")?,
        scan_index: t.u32("scan_index")?,
        frag_mz: t.f32("frag_mz")?,
        ppm: t.f32("ppm")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SearchSeedConfig {
        SearchSeedConfig::default()
    }

    #[test]
    fn the_tolerance_is_a_percentile_of_the_union_and_not_an_average_of_percentiles() {
        // The defect in one assertion. One large band with clean deviations and two small
        // ones each carrying a couple of random matches: in the small bands those few
        // outliers ARE the p95, so their fitted tolerances are enormous, and a
        // calibrant-weighted mean of the three is far wider than the p95 of the union,
        // where four outliers among 240 points sit above the 95th percentile and are not
        // it. Only the second is the estimator `search-seed` defines.
        let clean = |n: usize, step: f64| -> Vec<f64> {
            (0..n)
                .map(|i| ((i % 41) as f64 - 20.0) * step)
                .collect::<Vec<f64>>()
        };
        let mut small_a = clean(18, 0.5);
        small_a.extend([150.0, 160.0]);
        let mut small_b = clean(18, 0.5);
        small_b.extend([152.0, 162.0]);
        let bands = [clean(200, 0.5), small_a, small_b];
        let union: Vec<f64> = bands.iter().flatten().cloned().collect();
        let (mut w, mut wt) = (0.0f64, 0.0f64);
        for b in &bands {
            let (_, t) = fit(b);
            assert!(t > 5.0, "not floored, or the comparison is vacuous");
            w += b.len() as f64;
            wt += b.len() as f64 * t;
        }
        let averaged = wt / w;
        let (_, pooled_tol) = fit(&union);
        assert!(
            averaged > 2.0 * pooled_tol,
            "averaging per-band p95s ({averaged}) is far wider than the union's \
             ({pooled_tol})"
        );
        // And the pooled fit is what `fit_from` produces on the union.
        let mz = vec![500.0; union.len()];
        let cal = MassCal::fit_from(&union, &mz, &cfg());
        assert_eq!(cal.frag_tol_ppm, pooled_tol);
        assert_eq!(cal.n_dev, union.len() as u64);
        assert_eq!(cal.cal_passes, 1);
    }

    #[test]
    fn too_few_calibrants_keeps_the_configured_tolerance() {
        let devs = vec![1.0; MIN_CALIBRANTS - 1];
        let mz = vec![500.0; devs.len()];
        let cal = MassCal::fit_from(&devs, &mz, &cfg());
        assert_eq!(cal.frag_tol_ppm, cfg().fragment_tol_ppm);
        assert_eq!(cal.frag_ppm_offset, 0.0);
        assert_eq!(cal.cal_passes, 0);
        // Empty is the same branch and must not divide by anything.
        let empty = MassCal::fit_from(&[], &[], &cfg());
        assert_eq!(empty.n_dev, 0);
        assert_eq!(empty.ppm_residual_mad, 0.0);
    }

    #[test]
    fn the_second_pass_and_the_loess_grid_are_fitted_from_the_deviations() {
        // Both optional levers read the deviations rather than a scalar, so both work
        // wherever the deviations are. The pooled path has them because the sidecar
        // carries them.
        let mut devs: Vec<f64> = (0..200).map(|i| ((i % 41) as f64 - 20.0) * 0.5).collect();
        let mut mz: Vec<f64> = (0..200).map(|i| 300.0 + i as f64 * 5.0).collect();
        // Ten gross outliers the first pass must reject.
        for k in 0..10 {
            devs.push(140.0 + k as f64);
            mz.push(400.0 + k as f64);
        }
        let mut c = cfg();
        c.two_pass_mass_cal = true;
        let one = MassCal::fit_from(&devs, &mz, &cfg());
        let two = MassCal::fit_from(&devs, &mz, &c);
        assert_eq!(one.cal_passes, 1);
        assert_eq!(two.cal_passes, 2);
        assert!(
            two.frag_tol_ppm < one.frag_tol_ppm,
            "the outlier-trimmed re-fit is tighter: {} against {}",
            two.frag_tol_ppm,
            one.frag_tol_ppm
        );
        c.mass_cal_loess = true;
        let grid = MassCal::fit_from(&devs, &mz, &c);
        assert!(!grid.mz_cal_grid_mz.is_empty());
        assert_eq!(grid.mz_cal_grid_mz.len(), grid.mz_cal_grid_ppm.len());
        // Off by default, and the json says so with two empty arrays.
        assert!(two.mz_cal_grid_mz.is_empty());
        assert_eq!(
            two.to_json()["mz_cal_grid_mz"],
            serde_json::Value::Array(Vec::new())
        );
        assert_eq!(
            two.to_json()["frag_ppm_sigma"],
            two.to_json()["frag_tol_ppm"]
        );
    }

    #[test]
    fn the_calibrant_sidecar_round_trips() {
        let dir = std::env::temp_dir().join(format!("mumdia_masscal_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let seed = dir.join("seed.parquet").to_str().unwrap().to_string();
        let c = Calibrants {
            candidate_id: vec![7, 7, 9],
            scan_index: vec![3, 3, 11],
            frag_mz: vec![301.5, 402.25, 503.125],
            ppm: vec![-1.5, 2.25, 0.5],
        };
        assert_eq!(write_calibrants(&seed, &c).unwrap(), 3);
        let back = read_calibrants(&calibrants_path(&seed)).unwrap();
        assert_eq!(back.candidate_id, c.candidate_id);
        assert_eq!(back.scan_index, c.scan_index);
        assert_eq!(back.frag_mz, c.frag_mz);
        assert_eq!(back.ppm, c.ppm);
        // An empty sidecar is a fact ("this band calibrated nothing"), not an absence.
        let empty = dir.join("empty.parquet").to_str().unwrap().to_string();
        assert_eq!(write_calibrants(&empty, &Calibrants::default()).unwrap(), 0);
        assert!(read_calibrants(&calibrants_path(&empty))
            .unwrap()
            .is_empty());
    }
}
