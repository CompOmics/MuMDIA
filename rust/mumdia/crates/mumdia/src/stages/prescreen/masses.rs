//! Blind peptide neutral-mass hypotheses (the prototype's `complement_mass.infer_spectrum`).
//!
//! For every tag path of a spectrum and every other peak as a complementary anchor, the peptide
//! mass `z (mz_i - H+) + z' (mz_o - H+)` is proposed when it fits the spectrum's isolation
//! window at some precursor charge; the remaining path peaks then look for their own
//! complements, each on a distinct peak off the path. A hypothesis needs at least two distinct
//! complementary peaks. Per 0.01 Da mass bin the best (most complements, then best fit) is
//! kept. No sequence is read: the hypotheses are written before any candidate is consulted.
//!
//! Optional MS1 support: in the scans within `ms1_rt_s` before and after the spectrum, a
//! monoisotopic peak within max(0.005 Da, `ms1_ppm`) gives link 1, plus the +1 isotope at
//! 1.00335483507 / z link 2. A permissive supporting observation, not an isotope-envelope fit.

use mumdia_core::constants::PROTON;

use super::tags::{fit_weight, nearest, path_rms, Alphabet, Graph};

pub const ISOTOPE_SPACING: f64 = 1.003_354_835_07;

/// One MS1 scan: retention time, m/z ascending, intensities.
pub type Ms1Scan = (f64, Vec<f64>, Vec<f32>);

#[derive(Clone, Debug, PartialEq)]
pub struct Hypothesis {
    pub neutral: f64,
    pub complements: u8,
    pub quality: f64,
    pub key: u32,
    pub precursor_charge: i32,
}

/// Hypotheses of one spectrum with isolation window `[low, high)`; `mz` positive peaks ascending.
pub fn infer_spectrum(
    mz: &[f64],
    low: f64,
    high: f64,
    alpha: &Alphabet,
    tol: f64,
    sigma: f64,
) -> Vec<Hypothesis> {
    let mut best: std::collections::BTreeMap<i64, Hypothesis> = Default::default();
    let m = |s: u32| alpha.masses[s as usize];
    for z in 1..=2 {
        let zf = z as f64;
        let g = Graph::build(mz, z, tol, alpha, None);
        for i in 0..mz.len() {
            for e in g.edges(i) {
                let (j, a) = (g.target[e] as usize, g.label[e]);
                for f in g.edges(j) {
                    let (k, b) = (g.target[f] as usize, g.label[f]);
                    for h in g.edges(k) {
                        let (l, c) = (g.target[h] as usize, g.label[h]);
                        let nodes = [i, j, k, l];
                        let cum = [m(a), m(a) + m(b), m(a) + m(b) + m(c)];
                        let fit = fit_weight(path_rms(mz, &nodes, &cum, z), sigma);
                        for other in 0..mz.len() {
                            if nodes.contains(&other) {
                                continue;
                            }
                            for oz in 1..=2 {
                                let peptide =
                                    zf * (mz[i] - PROTON) + oz as f64 * (mz[other] - PROTON);
                                let pz = (z.max(oz)..6).find(|&pz| {
                                    let x = peptide / pz as f64 + PROTON;
                                    low <= x && x < high
                                });
                                let Some(pz) = pz else { continue };
                                let mut used = vec![other];
                                for &pn in &nodes[1..] {
                                    let partner = peptide - zf * (mz[pn] - PROTON);
                                    for q in 1..=pz.min(2) {
                                        let qf = q as f64;
                                        let Some(at) =
                                            nearest(mz, partner / qf + PROTON, 0.005 / qf)
                                        else {
                                            continue;
                                        };
                                        if nodes.contains(&at) || used.contains(&at) {
                                            continue;
                                        }
                                        used.push(at);
                                        break;
                                    }
                                }
                                if used.len() < 2 {
                                    continue;
                                }
                                let bin = (peptide * 100.0).round_ties_even() as i64;
                                let n = used.len() as u8;
                                let cand = Hypothesis {
                                    neutral: peptide,
                                    complements: n,
                                    quality: fit,
                                    key: alpha.key(
                                        alpha.canonical(a as u16, b as u16, c as u16),
                                        z,
                                        false,
                                    ),
                                    precursor_charge: pz,
                                };
                                match best.get(&bin) {
                                    Some(h)
                                        if !(n > h.complements
                                            || (n == h.complements && fit > h.quality)) => {}
                                    _ => {
                                        best.insert(bin, cand);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    best.into_values().collect()
}

/// MS1 link of a hypothesis: 0 none, 1 monoisotopic peak, 2 monoisotopic plus +1 isotope.
/// `ms1` is `(rt, mz ascending, intensity)` per scan, ascending RT.
pub fn ms1_link(h: &Hypothesis, rt: f64, ms1: &[Ms1Scan], rt_tol: f64, ppm: f64) -> u8 {
    let z = h.precursor_charge as f64;
    let prec = h.neutral / z + PROTON;
    let tol = (prec * ppm * 1e-6).max(0.005);
    let at = ms1.partition_point(|s| s.0 < rt);
    let mut link = 0u8;
    for s in ms1[at.saturating_sub(1)..(at + 1).min(ms1.len())].iter() {
        if (s.0 - rt).abs() > rt_tol {
            continue;
        }
        let mono = nearest(&s.1, prec, tol).filter(|&p| s.2[p] > 0.0);
        let iso = nearest(&s.1, prec + ISOTOPE_SPACING / z, tol).filter(|&p| s.2[p] > 0.0);
        if mono.is_some() {
            link = link.max(1);
            if iso.is_some() {
                link = 2;
            }
        }
    }
    link
}

/// Candidate mass support from hypotheses of one spectrum: the best `quality x (complements -
/// 1) / 3` among hypotheses within one 0.01 Da bin of the candidate's neutral mass.
pub fn support(neutral: f64, hyps: &[Hypothesis]) -> f64 {
    let b = (neutral * 100.0).round_ties_even() as i64;
    hyps.iter()
        .filter(|h| ((h.neutral * 100.0).round_ties_even() as i64 - b).abs() <= 1)
        .map(|h| h.quality * (h.complements as f64 - 1.0) / 3.0)
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_core::constants::WATER;

    fn alpha() -> Alphabet {
        Alphabet::build(&[(b'C', "Carbamidomethyl".into())], &[]).unwrap()
    }

    /// b and y ladders of one peptide: the inferred hypotheses include its neutral mass, with
    /// at least two distinct complements, inferred without the sequence.
    #[test]
    fn complementary_ladders_infer_the_peptide_mass() {
        let a = alpha();
        let seq = b"GASPVTLK";
        let m: Vec<f64> = seq
            .iter()
            .map(|&r| a.masses[a.state(r, 0.0).unwrap() as usize])
            .collect();
        let neutral: f64 = m.iter().sum::<f64>() + WATER;
        let mut mz = Vec::new();
        let mut b = 0.0;
        for x in &m[..m.len() - 1] {
            b += x;
            mz.push(b + PROTON);
            mz.push(neutral - b + PROTON);
        }
        mz.sort_by(f64::total_cmp);
        let prec = neutral / 2.0 + PROTON;
        let h = infer_spectrum(&mz, prec - 1.0, prec + 1.0, &a, 0.005, 0.003);
        let hit = h
            .iter()
            .find(|x| (x.neutral - neutral).abs() < 0.01)
            .expect("peptide mass");
        assert!(hit.complements >= 2);
        assert_eq!(hit.precursor_charge, 2);
        assert!(support(neutral, &h) > 0.0);
        assert_eq!(support(neutral + 1.0, &h), 0.0);
    }

    #[test]
    fn ms1_links_respect_time_and_tolerance_and_name_the_isotope() {
        let h = Hypothesis {
            neutral: 1000.0,
            complements: 2,
            quality: 1.0,
            key: 0,
            precursor_charge: 2,
        };
        let mono = 1000.0 / 2.0 + PROTON;
        let iso = mono + ISOTOPE_SPACING / 2.0;
        let scan = |rt: f64, peaks: Vec<f64>| (rt, peaks.clone(), vec![1.0f32; peaks.len()]);
        let ms1 = vec![scan(10.0, vec![mono]), scan(20.0, vec![mono, iso])];
        assert_eq!(ms1_link(&h, 10.5, &ms1, 2.0, 10.0), 1);
        assert_eq!(ms1_link(&h, 20.0, &ms1, 2.0, 10.0), 2);
        assert_eq!(ms1_link(&h, 15.0, &ms1, 2.0, 10.0), 0, "outside 2 s");
        let off = vec![scan(10.0, vec![mono + 0.006])];
        assert_eq!(
            ms1_link(&h, 10.0, &off, 2.0, 10.0),
            0,
            "max(0.005 Da, 10 ppm) = 0.005"
        );
    }
}
