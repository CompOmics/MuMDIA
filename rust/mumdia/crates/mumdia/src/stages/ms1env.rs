//! The MS1 isotope-envelope family (`features.ms1_envelope`).
//!
//! The features the engine already has read the precursor's MS1 at the apex alone, three
//! isotopes (M0, M+1, M+2) and M-1 summed within `extract.prec_tol_ppm` in the nearest MS1
//! scan (the `ms1` family), plus the isotope XICs on the candidate's grid. None of them reads
//! an m/z: an interfering peak inside the tolerance counts as the precursor's. This family
//! reads, in the MS1 scans around the apex ([`ENV_SCANS`] either side of the nearest), the
//! peak nearest each of the isotopes M-1 to M+4 and the half-spacing positions above M0 and
//! M+1, at the calibrated MS1 m/z (`extract.ms1_calibrate`, the surface under `"surface"`),
//! within a search tolerance twice `extract.prec_tol_ppm` and at least [`ENV_MIN_TOL_PPM`],
//! so that the m/z error itself is measured rather than cut at the tolerance.
//!
//! Extract writes the evidence ([`Ms1Env`], psms_extracted v7): per isotope the summed
//! intensity and the intensity-weighted ppm error of the peaks found, the intensity at the
//! half-spacing positions, the scans read, and under `extract.q1_fragments` the agreement of
//! the fitted Q1 centre with the centre the observed envelope predicts. The features stage
//! derives [`FEATURES`] from it ([`values`]):
//!
//! - `ms1e_env_r2`: the share of the observed M0..M+4 envelope (summed over the scans) that
//!   the averagine envelope explains, `1 - |o - a e|^2 / |o|^2` with the least-squares scale
//!   `a >= 0`; 0 without signal.
//! - `ms1e_env_n`: the isotopes found in a row from M0 (0 to 5).
//! - `ms1e_m1_share`: M-1 against M-1 plus M0. A peak one spacing below the monoisotopic m/z
//!   belongs to another envelope or says the monoisotopic peak was assigned too high.
//! - `ms1e_half_share`: the half-spacing positions against them plus M0 and M+1: a species
//!   of twice the charge (or a wrongly assigned charge) puts peaks there.
//! - `ms1e_spacing_ppm`: the intensity-weighted mean of `|ppm(k + 1) - ppm(k)|` over the
//!   consecutive isotopes found among M0..M+4, which is the deviation of their m/z spacing
//!   from `1.00335 / z` in ppm of the precursor m/z; [`MISSING_PPM`] without a pair.
//! - `ms1e_mono_ppm` and `ms1e_mono_ppm_abs`: the M0 error after the calibration, signed (0
//!   when M0 is not found) and absolute ([`MISSING_PPM`] when it is not).
//! - `ms1e_ibar_dev`: the observed envelope's mean isotope index over M0..M+4 against the
//!   averagine one; 0 without signal.
//! - `ms1e_q1_agree`: under `extract.q1_fragments`, the distance (Q1 steps) between the
//!   fitted Q1 centre of the candidate's fragments and the centre its observed MS1 envelope
//!   predicts (`mu + sp (ibar_averagine - ibar_observed) / step`); 0 without a fit.
//!
//! Every value is finite. Deterministic: scans in retention-time order, isotopes in order,
//! the nearest peak with ties to the lower m/z.

use mumdia_core::constants::{averagine_isotopes, ISOTOPE_SPACING, PROTON};

use crate::spectra::Ms1Scan;

/// MS1 scans read either side of the one nearest the apex.
pub const ENV_SCANS: usize = 2;
/// The search tolerance is twice `extract.prec_tol_ppm`, and at least this.
pub const ENV_MIN_TOL_PPM: f64 = 10.0;
/// The ppm value a feature takes where the evidence has none (no isotope pair, M0 not found).
pub const MISSING_PPM: f64 = 10.0;
/// Isotopes read: M-1, M0 .. M+4.
pub const N_ISO: usize = 6;

/// The evidence columns of psms_extracted v7, in table order.
pub const EVIDENCE_COLUMNS: [&str; 16] = [
    "ms1e_i_m1",
    "ms1e_i_0",
    "ms1e_i_1",
    "ms1e_i_2",
    "ms1e_i_3",
    "ms1e_i_4",
    "ms1e_ppm_m1",
    "ms1e_ppm_0",
    "ms1e_ppm_1",
    "ms1e_ppm_2",
    "ms1e_ppm_3",
    "ms1e_ppm_4",
    "ms1e_i_half",
    "ms1e_scans",
    "ms1e_tol_ppm",
    "ms1e_q1_agree",
];

/// The feature family, appended after every other column of the `extended` set.
pub const FEATURES: &[&str] = &[
    "ms1e_env_r2",
    "ms1e_env_n",
    "ms1e_m1_share",
    "ms1e_half_share",
    "ms1e_spacing_ppm",
    "ms1e_mono_ppm",
    "ms1e_mono_ppm_abs",
    "ms1e_ibar_dev",
    "ms1e_q1_agree",
];

/// One row's MS1 isotope-envelope evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ms1Env {
    /// Summed intensity of the nearest peak of each isotope (M-1, M0 .. M+4) over the scans.
    pub i: [f64; N_ISO],
    /// Intensity-weighted mean ppm error of those peaks (NaN where none was found).
    pub ppm: [f64; N_ISO],
    /// Summed intensity at the half-spacing positions above M0 and M+1.
    pub half: f64,
    /// MS1 scans read.
    pub scans: f64,
    /// The search tolerance (ppm).
    pub tol_ppm: f64,
    /// `extract.q1_fragments`: the fitted Q1 centre against the observed envelope's (steps);
    /// NaN without a fit.
    pub q1_agree: f64,
}

impl Default for Ms1Env {
    fn default() -> Self {
        Ms1Env {
            i: [0.0; N_ISO],
            ppm: [f64::NAN; N_ISO],
            half: 0.0,
            scans: 0.0,
            tol_ppm: 0.0,
            q1_agree: f64::NAN,
        }
    }
}

impl Ms1Env {
    /// The values in [`EVIDENCE_COLUMNS`] order.
    pub fn columns(&self) -> [f64; 16] {
        let mut v = [0.0; 16];
        v[..N_ISO].copy_from_slice(&self.i);
        v[N_ISO..2 * N_ISO].copy_from_slice(&self.ppm);
        v[12] = self.half;
        v[13] = self.scans;
        v[14] = self.tol_ppm;
        v[15] = self.q1_agree;
        v
    }
}

/// The search tolerance for an `extract.prec_tol_ppm` of `prec_tol_ppm`.
pub fn search_tol(prec_tol_ppm: f64) -> f64 {
    (2.0 * prec_tol_ppm).max(ENV_MIN_TOL_PPM)
}

/// The peak of `mz` (ascending) nearest `target` within `tol_ppm`, ties to the lower m/z.
fn nearest(mz: &[f32], target: f64, tol_ppm: f64) -> Option<usize> {
    let (lo, hi) = mumdia_core::constants::ppm_bounds(target, tol_ppm);
    let mut j = mz.partition_point(|&m| (m as f64) < lo);
    let mut best: Option<(f64, usize)> = None;
    while j < mz.len() && (mz[j] as f64) <= hi {
        let d = (mz[j] as f64 - target).abs();
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, j));
        }
        j += 1;
    }
    best.map(|(_, j)| j)
}

/// The evidence of a precursor of m/z `pm` and charge `z` in the MS1 scans around scan `j`
/// (the one nearest the apex): `shift(mz, rt)` is the calibrated read factor
/// (`ExtractConfig::ms1_shift`), `tol_ppm` the search tolerance ([`search_tol`]).
pub fn read(
    scans: &[Ms1Scan],
    j: usize,
    pm: f64,
    z: i32,
    tol_ppm: f64,
    shift: &dyn Fn(f64, f64) -> f64,
) -> Ms1Env {
    let mut e = Ms1Env {
        tol_ppm,
        ..Ms1Env::default()
    };
    if scans.is_empty() || z <= 0 {
        return e;
    }
    let sp = ISOTOPE_SPACING / z as f64;
    let first = j.saturating_sub(ENV_SCANS);
    let last = (j + ENV_SCANS).min(scans.len() - 1);
    let mut wppm = [0.0f64; N_ISO];
    for s in &scans[first..=last] {
        e.scans += 1.0;
        for (k, (ik, wk)) in e.i.iter_mut().zip(wppm.iter_mut()).enumerate() {
            let lib = pm + (k as f64 - 1.0) * sp;
            let at = lib * shift(lib, s.rt_seconds);
            if let Some(p) = nearest(&s.mz, at, tol_ppm) {
                let v = s.intensity[p] as f64;
                if v > 0.0 {
                    *ik += v;
                    *wk += v * (s.mz[p] as f64 / at - 1.0) * 1e6;
                }
            }
        }
        for h in [0.5, 1.5] {
            let lib = pm + h * sp;
            let at = lib * shift(lib, s.rt_seconds);
            if let Some(p) = nearest(&s.mz, at, tol_ppm) {
                e.half += (s.intensity[p] as f64).max(0.0);
            }
        }
    }
    for ((pk, &ik), &wk) in e.ppm.iter_mut().zip(&e.i).zip(&wppm) {
        if ik > 0.0 {
            *pk = wk / ik;
        }
    }
    e
}

/// The observed envelope's mean isotope index over M0..M+4 (`i` in [`Ms1Env`] order), NaN
/// without signal.
fn ibar_observed(i: &[f64; N_ISO]) -> f64 {
    let s: f64 = i[1..].iter().sum();
    if s > 0.0 {
        i[1..]
            .iter()
            .enumerate()
            .map(|(k, v)| k as f64 * v)
            .sum::<f64>()
            / s
    } else {
        f64::NAN
    }
}

/// The averagine envelope M0..M+4 of a precursor of neutral mass `mass`, and its mean index.
fn averagine5(mass: f64) -> ([f64; 5], f64) {
    let v = averagine_isotopes(mass, 5);
    let mut a = [0.0; 5];
    a.copy_from_slice(&v);
    let ibar = a.iter().enumerate().map(|(k, x)| k as f64 * x).sum();
    (a, ibar)
}

/// The neutral mass of an ion of m/z `mz` and charge `z` (0 for `z <= 0`).
fn neutral(mz: f64, z: i32) -> f64 {
    if z <= 0 {
        return 0.0;
    }
    (mz - PROTON) * z as f64
}

/// `ms1e_q1_agree` before its absolute value: the fitted Q1 centre `mu` (steps, against the
/// expected centre of the averagine envelope) less the centre the observed envelope predicts,
/// in steps of `step_mz`. NaN without signal or a step.
pub fn q1_agree(e: &Ms1Env, pm: f64, z: i32, mu: f64, step_mz: f64) -> f64 {
    if step_mz.is_nan() || step_mz <= 0.0 || z <= 0 {
        return f64::NAN;
    }
    let ib = ibar_observed(&e.i);
    if !ib.is_finite() {
        return f64::NAN;
    }
    let (_, ia) = averagine5(neutral(pm, z));
    let sp = ISOTOPE_SPACING / z as f64;
    mu + sp * (ia - ib) / step_mz
}

fn fin(x: f64, or: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        or
    }
}

/// The [`FEATURES`] of one row from its evidence (`ev` in [`EVIDENCE_COLUMNS`] order) and
/// its precursor m/z and charge.
pub fn values(ev: &[f64; 16], pm: f64, z: i32) -> [f64; 9] {
    let mut i = [0.0f64; N_ISO];
    for (k, x) in i.iter_mut().enumerate() {
        *x = fin(ev[k], 0.0).max(0.0);
    }
    let ppm = &ev[N_ISO..2 * N_ISO];
    let half = fin(ev[12], 0.0).max(0.0);
    let (a, ia) = averagine5(neutral(pm, z));
    let o = &i[1..];
    let (mut soe, mut see, mut soo) = (0.0f64, 0.0f64, 0.0f64);
    for (ok, ek) in o.iter().zip(&a) {
        soe += ok * ek;
        see += ek * ek;
        soo += ok * ok;
    }
    let r2 = if soo > 0.0 && see > 0.0 {
        let sc = (soe / see).max(0.0);
        let res: f64 = o
            .iter()
            .zip(&a)
            .map(|(ok, ek)| (ok - sc * ek).powi(2))
            .sum();
        (1.0 - res / soo).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let n_run = o.iter().take_while(|&&v| v > 0.0).count() as f64;
    let m1 = if i[0] + i[1] > 0.0 {
        i[0] / (i[0] + i[1])
    } else {
        0.0
    };
    let hs = if half + i[1] + i[2] > 0.0 {
        half / (half + i[1] + i[2])
    } else {
        0.0
    };
    let (mut sd, mut sw) = (0.0f64, 0.0f64);
    for k in 1..N_ISO - 1 {
        let (p0, p1) = (ppm[k], ppm[k + 1]);
        if p0.is_finite() && p1.is_finite() && i[k] > 0.0 && i[k + 1] > 0.0 {
            let w = i[k].min(i[k + 1]);
            sd += w * (p1 - p0).abs();
            sw += w;
        }
    }
    let spacing = if sw > 0.0 { sd / sw } else { MISSING_PPM };
    let mono = ppm[1];
    let (mono_s, mono_a) = if mono.is_finite() && i[1] > 0.0 {
        (mono, mono.abs())
    } else {
        (0.0, MISSING_PPM)
    };
    let ib = ibar_observed(&i);
    let ibar_dev = if ib.is_finite() { ib - ia } else { 0.0 };
    let q1 = fin(ev[15].abs(), 0.0);
    [
        r2,
        n_run,
        m1,
        hs,
        fin(spacing, MISSING_PPM),
        fin(mono_s, 0.0),
        fin(mono_a, MISSING_PPM),
        fin(ibar_dev, 0.0),
        q1,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(rt: f64, peaks: &[(f64, f32)]) -> Ms1Scan {
        let mut p = peaks.to_vec();
        p.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ms1Scan {
            scan_index: 0,
            rt_seconds: rt,
            mz: p.iter().map(|x| x.0 as f32).collect(),
            intensity: p.iter().map(|x| x.1).collect(),
            im: None,
        }
    }

    /// An averagine envelope of a 2+ at 600 m/z, read 2 ppm high in five scans.
    fn envelope_scans(extra: &[(f64, f32)]) -> (Vec<Ms1Scan>, f64) {
        let pm = 600.0;
        let mass = neutral(pm, 2);
        let a = averagine_isotopes(mass, 5);
        let sp = ISOTOPE_SPACING / 2.0;
        let scans = (0..7)
            .map(|s| {
                let h = [0.2f32, 0.6, 1.0, 0.6, 0.2, 0.1, 0.05][s];
                let mut p: Vec<(f64, f32)> = (0..5)
                    .map(|k| ((pm + k as f64 * sp) * (1.0 + 2e-6), 1e6 * h * a[k] as f32))
                    .collect();
                p.extend_from_slice(extra);
                scan(100.0 + s as f64, &p)
            })
            .collect();
        (scans, pm)
    }

    #[test]
    fn a_clean_envelope_reads_as_averagine_with_its_error() {
        let (scans, pm) = envelope_scans(&[]);
        let e = read(&scans, 2, pm, 2, 10.0, &|_, _| 1.0);
        assert_eq!(e.scans, 5.0);
        assert_eq!(e.i[0], 0.0);
        assert!(e.ppm[0].is_nan());
        for k in 1..6 {
            assert!(e.i[k] > 0.0);
            assert!((e.ppm[k] - 2.0).abs() < 0.2, "{k}: {}", e.ppm[k]);
        }
        let v = values(&e.columns(), pm, 2);
        assert!(v[0] > 0.999, "r2 {}", v[0]);
        assert_eq!(v[1], 5.0);
        assert_eq!(v[2], 0.0);
        assert_eq!(v[3], 0.0);
        assert!(v[4] < 0.2, "spacing {}", v[4]);
        assert!((v[5] - 2.0).abs() < 0.2 && (v[6] - 2.0).abs() < 0.2);
        assert!(v[7].abs() < 1e-3, "ibar {}", v[7]);
        assert_eq!(v[8], 0.0);
        // The calibration's factor moves the expected m/z: read 2 ppm high, nothing is off.
        let c = read(&scans, 2, pm, 2, 10.0, &|_, _| 1.0 + 2e-6);
        assert!(c.ppm[1].abs() < 0.2);
    }

    #[test]
    fn an_m1_peak_and_half_spacing_peaks_are_seen() {
        let sp = ISOTOPE_SPACING / 2.0;
        let (scans, pm) = envelope_scans(&[(600.0 - sp, 4e5), (600.0 + 0.5 * sp, 3e5)]);
        let e = read(&scans, 2, pm, 2, 10.0, &|_, _| 1.0);
        let v = values(&e.columns(), pm, 2);
        assert!(v[2] > 0.2, "m1 share {}", v[2]);
        assert!(v[3] > 0.1, "half share {}", v[3]);
    }

    #[test]
    fn no_signal_gives_the_missing_values() {
        let e = read(&[], 0, 600.0, 2, 10.0, &|_, _| 1.0);
        let v = values(&e.columns(), 600.0, 2);
        assert_eq!(
            v,
            [0.0, 0.0, 0.0, 0.0, MISSING_PPM, 0.0, MISSING_PPM, 0.0, 0.0]
        );
        assert!(v.iter().all(|x| x.is_finite()));
        assert!(q1_agree(&e, 600.0, 2, 0.3, 0.76).is_nan());
    }

    #[test]
    fn the_q1_agreement_is_zero_for_an_averagine_envelope_at_the_expected_centre() {
        let (scans, pm) = envelope_scans(&[]);
        let e = read(&scans, 2, pm, 2, 10.0, &|_, _| 1.0);
        assert!(q1_agree(&e, pm, 2, 0.0, 0.7614).abs() < 1e-3);
        assert!((q1_agree(&e, pm, 2, 0.5, 0.7614) - 0.5).abs() < 1e-3);
        // A heavier observed envelope predicts a higher centre: a fitted 0 then disagrees.
        let mut heavy = e;
        heavy.i[1] *= 0.3;
        assert!(q1_agree(&heavy, pm, 2, 0.0, 0.7614) < -0.05);
    }

    #[test]
    fn the_search_tolerance_is_twice_the_precursor_tolerance_and_at_least_ten() {
        assert_eq!(search_tol(3.0), 10.0);
        assert_eq!(search_tol(7.17), 14.34);
    }
}
