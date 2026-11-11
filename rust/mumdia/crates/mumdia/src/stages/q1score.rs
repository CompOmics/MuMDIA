//! Unfragmented-precursor and multi-scale Q1 scores of scanning-quadrupole spectra
//! (`features.q1_precursor`, `features.q1_multiscale`).
//!
//! Both are read from the per-fragment decision's RT x step patch ([`super::q1frag`]) of a row
//! of `extract.q1_fragments`, at its 2D apex, and written to psms_extracted v8.
//!
//! **The unfragmented precursor** ([`precursor_scores`]). Part of a precursor survives the
//! collision cell, and in a ZT Scan MS2 spectrum it shows up at its own m/z in the steps whose
//! window transmits it. Each isotope (M0, M+1, M+2) is one ion, so its Q1 profile is the
//! calibrated window centred on that isotope's m/z, and the three are read like fragments of
//! their own ([`super::q1frag::CandQ1::precursor`]) on the cells of the fragment patch. From
//! them: the isotopes present by the decision's coherent-cell rule, the precursor's share of
//! the precursor plus fragment intensity at the 2D apex, one shift of the isotopes' Q1
//! profiles from where the calibrated window puts them (the least-squares grid and parabola of
//! the fragments' shared centre, width as calibrated) and their residual under it, the
//! correlation of each present fragment's Q1 profile with the precursor's (its isotopes
//! summed), the fragments whose own Q1 centre lies within the decision's offset tolerance of
//! the precursor's shift (and whose 2D correlation reaches the floor), and the distance
//! between that shift and the fragments' shared centre. A fragment of this precursor sits
//! where the precursor sits; one borrowed from a co-isolated precursor does not.
//!
//! **Multi-scale scores** ([`multiscale`]). Over the 3, 7 and 11 steps centred on the step
//! nearest the fragments' fitted Q1 centre ([`SCALES`]), with the Q1 profiles of the patch at
//! the 2D apex: the mean correlation of every present fragment's profile with the most intense
//! fragment's, the residual (`1 - R^2`, with a scale and a floor) of the fragments' consensus
//! profile against their mean expected transmission at the fitted centre and width, the most
//! intense fragment's intensity-weighted Q1 centroid against its expected centre (steps), and
//! with the precursor read, the mean correlation of the fragments' profiles with the
//! precursor's. The narrow scale reads the transmission's flat top, where a fragment of the
//! precursor and one of a neighbour one or two steps away look alike only in intensity; the
//! wide one reads its edges, where they part.
//!
//! Deterministic: every sum runs in fragment and step order, ties keep the first.

use mumdia_core::config::Q1FragModel;

use super::q1frag::{
    mu_of, presence, profile_sse, profiles, refine, CandQ1, Decision, Patch, ABSENT, MU_MAX, MU_N,
};
use crate::stats::pearson;

/// The precursor isotopes read in the MS2 patch: M0, M+1 and M+2.
pub const PREC_ISOTOPES: usize = 3;
/// The Q1 windows of the multi-scale scores, in steps centred on the Q1 apex step.
pub const SCALES: [usize; 3] = [3, 7, 11];

/// The `q1p_*` columns of psms_extracted v8 (`features.q1_precursor`), in the order
/// [`precursor_scores`] returns them; also the feature names.
pub const Q1P_COLUMNS: &[&str] = &[
    "q1p_n_iso",
    "q1p_share",
    "q1p_offset",
    "q1p_resid",
    "q1p_frag_corr_mean",
    "q1p_frag_corr_max",
    "q1p_n_belong",
    "q1p_centre_agree",
];

/// The `q1m*` columns of every scale (`features.q1_multiscale`), in the order
/// [`multiscale`] returns them.
pub const Q1M_COLUMNS: &[&str] = &[
    "q1m3_corr_best",
    "q1m3_shape",
    "q1m3_centroid",
    "q1m7_corr_best",
    "q1m7_shape",
    "q1m7_centroid",
    "q1m11_corr_best",
    "q1m11_shape",
    "q1m11_centroid",
];

/// The precursor-referenced `q1m*` columns, written after [`Q1M_COLUMNS`] when
/// `features.q1_precursor` is on as well.
pub const Q1M_PREC_COLUMNS: &[&str] = &["q1m3_corr_prec", "q1m7_corr_prec", "q1m11_corr_prec"];

/// The `q1m*` columns written: [`Q1M_COLUMNS`], then [`Q1M_PREC_COLUMNS`] when the precursor is
/// read (`prec`).
pub fn q1m_columns(prec: bool) -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Q1M_COLUMNS.to_vec();
    if prec {
        v.extend_from_slice(Q1M_PREC_COLUMNS);
    }
    v
}

/// The `q1p_*` values of a row without a patch (its apex not in the sweep): nothing present,
/// no shift (0), residual 1, no agreement ([`MU_MAX`]).
pub fn empty_precursor() -> Vec<f64> {
    vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, MU_MAX]
}

/// The `q1m*` values of a row without a patch: no correlation, residual 1, centroid 0.
pub fn empty_multiscale(prec: bool) -> Vec<f64> {
    let mut v = Vec::with_capacity(4 * SCALES.len());
    for _ in SCALES {
        v.extend_from_slice(&[0.0, 1.0, 0.0]);
    }
    if prec {
        v.extend(std::iter::repeat_n(0.0, SCALES.len()));
    }
    v
}

/// `1 - R^2` of profile `y` against `e` with a scale and a floor (the decision's rule): its
/// [`profile_sse`] over the total sum of squares of `y / sum(y)`, clamped to [0, 1]; 0 for a
/// flat profile, 1 for one without signal.
fn resid_of(y: &[f64], e: &[f64]) -> f64 {
    let tot: f64 = y.iter().sum();
    if tot.is_nan() || tot <= 0.0 || y.len() < 2 {
        return 1.0;
    }
    let n = y.len() as f64;
    let (mut sy, mut syy) = (0.0, 0.0);
    for &v in y {
        let yn = v / tot;
        sy += yn;
        syy += yn * yn;
    }
    let sst = syy - sy * sy / n;
    if sst > 0.0 {
        (profile_sse(y, e) / sst).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// The step spacing the decision `d` worked at, or the model's.
fn step_of(d: &Decision, p: &Patch, m: &Q1FragModel) -> f64 {
    if d.step > 0.0 {
        d.step
    } else {
        p.step(m.step_mz)
    }
}

/// The unfragmented-precursor scores of a row: the fragment patch `fp` with its decision
/// `d`, and the precursor's patch `pp` of `pc` ([`CandQ1::precursor`]) on the same cells.
/// Returns the [`Q1P_COLUMNS`] values and the precursor's Q1 profile at the 2D apex (its
/// isotopes summed), which [`multiscale`] correlates the fragments with.
pub fn precursor_scores(
    fp: &Patch,
    d: &Decision,
    pp: &Patch,
    pc: &CandQ1,
    m: &Q1FragModel,
) -> (Vec<f64>, Vec<f64>) {
    let ns = fp.ns();
    if fp.cycles.is_empty() || ns == 0 || pp.ns() != ns || pp.cycles.len() != fp.cycles.len() {
        return (empty_precursor(), vec![0.0; ns]);
    }
    let (_, present) = presence(pp, pc, m);
    let n_iso = present.iter().filter(|&&x| x).count() as f64;
    let qp = profiles(pp, d.apex_ci);
    let qf = profiles(fp, d.apex_ci);
    let ip: f64 = qp.iter().map(|v| v.iter().sum::<f64>()).sum();
    let i_f: f64 = qf.iter().map(|v| v.iter().sum::<f64>()).sum();
    let share = if ip + i_f > 0.0 { ip / (ip + i_f) } else { 0.0 };
    let step = step_of(d, fp, m);
    // One shift shared by the present isotopes with a Q1 profile (at least two steps with
    // signal), on the decision's centre grid at the calibrated width.
    let fitted: Vec<usize> = (0..pc.frags.len())
        .filter(|&k| present[k] && qp[k].iter().filter(|&&v| v > 0.0).count() >= 2)
        .collect();
    let expected = |k: usize, mu: f64| -> Vec<f64> {
        fp.t.iter()
            .map(|&t| pc.expected_at(m, k, t, mu, step, 1.0))
            .collect()
    };
    let (mut mu_p, mut resid, mut fit) = (0.0, 1.0, false);
    if !fitted.is_empty() {
        let totals: Vec<f64> = (0..MU_N)
            .map(|g| {
                let mu = mu_of(g as f64);
                fitted
                    .iter()
                    .map(|&k| profile_sse(&qp[k], &expected(k, mu)))
                    .sum()
            })
            .collect();
        let mut kb = 0usize;
        for g in 1..MU_N {
            if totals[g] < totals[kb] {
                kb = g;
            }
        }
        mu_p = mu_of(refine(&totals, kb));
        resid = fitted
            .iter()
            .map(|&k| resid_of(&qp[k], &expected(k, mu_p)))
            .sum::<f64>()
            / fitted.len() as f64;
        fit = true;
    }
    // The precursor's profile, its isotopes summed, and the fragments' correlation with it.
    let env: Vec<f64> = (0..ns).map(|si| qp.iter().map(|v| v[si]).sum()).collect();
    let corrs: Vec<f64> = if env.iter().any(|&v| v > 0.0) {
        (0..d.class.len())
            .filter(|&f| d.class[f] != ABSENT)
            .map(|f| pearson(&qf[f], &env))
            .collect()
    } else {
        Vec::new()
    };
    let corr_mean = if corrs.is_empty() {
        0.0
    } else {
        corrs.iter().sum::<f64>() / corrs.len() as f64
    };
    let corr_max = corrs.iter().copied().fold(0.0f64, f64::max);
    let tol = &m.tolerances;
    let n_belong = if fit {
        (0..d.class.len())
            .filter(|&f| {
                d.has_q[f]
                    && (d.offset[f] - mu_p).abs() <= tol.offset_steps
                    && d.corr2d[f] >= tol.corr2d_min
            })
            .count() as f64
    } else {
        0.0
    };
    let agree = if fit && d.has_fit {
        (mu_p - d.centre).abs()
    } else {
        MU_MAX
    };
    (
        vec![
            n_iso, share, mu_p, resid, corr_mean, corr_max, n_belong, agree,
        ],
        env,
    )
}

/// The multi-scale Q1 scores of a row: the fragment patch `fp` of `cand` with its decision
/// `d`, and the precursor's Q1 profile at the 2D apex when it was read (`env`,
/// [`precursor_scores`]). Returns the [`q1m_columns`] values.
pub fn multiscale(
    fp: &Patch,
    cand: &CandQ1,
    d: &Decision,
    m: &Q1FragModel,
    env: Option<&[f64]>,
) -> Vec<f64> {
    let ns = fp.ns();
    let nf = fp.nf.min(d.class.len());
    if fp.cycles.is_empty() || ns == 0 || nf == 0 {
        return empty_multiscale(env.is_some());
    }
    let qf = profiles(fp, d.apex_ci);
    let step = step_of(d, fp, m);
    // The Q1 apex step: nearest the fitted shared centre (the expected one without a fit),
    // the lower on a tie.
    let x = cand.centre + d.centre * step;
    let mut a = 0usize;
    for si in 1..ns {
        if (fp.t[si] - x).abs() < (fp.t[a] - x).abs() {
            a = si;
        }
    }
    let present: Vec<usize> = (0..nf).filter(|&f| d.class[f] != ABSENT).collect();
    let tot: Vec<f64> = qf.iter().map(|v| v.iter().sum::<f64>()).collect();
    // The fragments' mean expected transmission at the fitted centre and width.
    let mut emean = vec![0.0f64; ns];
    if !present.is_empty() {
        for &f in &present {
            for (si, e) in emean.iter_mut().enumerate() {
                *e += cand.expected_at(m, f, fp.t[si], d.centre, step, d.width);
            }
        }
        for e in emean.iter_mut() {
            *e /= present.len() as f64;
        }
    }
    let mut out = Vec::with_capacity(4 * SCALES.len());
    let mut corr_prec = Vec::with_capacity(SCALES.len());
    for k in SCALES {
        let half = k / 2;
        let lo = a.saturating_sub(half);
        let hi = (a + half).min(ns - 1);
        let w = lo..=hi;
        let slice = |v: &[f64]| -> Vec<f64> { v[w.clone()].to_vec() };
        // The most intense present fragment over the window (the first on a tie).
        let mut best: Option<(usize, f64)> = None;
        for &f in &present {
            let s: f64 = qf[f][w.clone()].iter().sum();
            if s > 0.0 && best.is_none_or(|(_, b)| s > b) {
                best = Some((f, s));
            }
        }
        let (corr_best, centroid) = match best {
            Some((b, sb)) => {
                let qb = slice(&qf[b]);
                let others: Vec<f64> = present
                    .iter()
                    .filter(|&&f| f != b)
                    .map(|&f| pearson(&slice(&qf[f]), &qb))
                    .collect();
                let cb = if others.is_empty() {
                    0.0
                } else {
                    others.iter().sum::<f64>() / others.len() as f64
                };
                let c: f64 = w.clone().map(|si| qf[b][si] * fp.t[si]).sum::<f64>() / sb;
                (cb, (c - cand.frags[b].centre) / step)
            }
            None => (0.0, 0.0),
        };
        // The consensus profile: each present fragment normalised to its own total.
        let cons: Vec<f64> = w
            .clone()
            .map(|si| {
                present
                    .iter()
                    .filter(|&&f| tot[f] > 0.0)
                    .map(|&f| qf[f][si] / tot[f])
                    .sum()
            })
            .collect();
        let shape = resid_of(&cons, &slice(&emean));
        out.extend_from_slice(&[corr_best, shape, centroid]);
        if let Some(env) = env {
            let ew = slice(env);
            let cs: Vec<f64> = if ew.iter().any(|&v| v > 0.0) {
                present
                    .iter()
                    .map(|&f| pearson(&slice(&qf[f]), &ew))
                    .collect()
            } else {
                Vec::new()
            };
            corr_prec.push(if cs.is_empty() {
                0.0
            } else {
                cs.iter().sum::<f64>() / cs.len() as f64
            });
        }
    }
    out.extend(corr_prec);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::q1frag::{decide, transmission};
    use mumdia_core::config::Q1FragTolerances;

    fn model() -> Q1FragModel {
        Q1FragModel {
            step_mz: 0.76,
            half_width_mz: 1.9,
            edge_width_mz: 1.5,
            offset_terms: Vec::new(),
            tolerances: Q1FragTolerances {
                offset_steps: 1.15,
                centre_steps: 0.7,
                corr2d_min: 0.5,
            },
        }
    }

    /// A patch of `nf` fragments over 7 cycles and the steps within reach + 2 of `centre`,
    /// each fragment `f` transmitted around `centres[f]` (target units, `None`: absent) with
    /// an elution peaking at the middle cycle.
    fn synth(m: &Q1FragModel, cand: &CandQ1, centres: &[Option<f64>], amp: &[f64]) -> Patch {
        let reach = m.half_width_mz + 0.5 * m.edge_width_mz + 2.0 * m.step_mz;
        let n = (2.0 * reach / m.step_mz).floor() as usize + 1;
        let t0 = cand.centre - reach;
        let t: Vec<f64> = (0..n).map(|i| t0 + i as f64 * m.step_mz).collect();
        let mut p = Patch::new((0..7).collect(), 0, t.clone(), centres.len());
        let elution = [0.05, 0.2, 0.6, 1.0, 0.6, 0.2, 0.05];
        for (f, c) in centres.iter().enumerate() {
            let Some(c) = c else { continue };
            for (ci, e) in elution.iter().enumerate() {
                for (si, &tt) in t.iter().enumerate() {
                    let v = amp[f] * e * transmission(tt - c, m.half_width_mz, m.edge_width_mz);
                    p.set(f, ci, si, v as f32);
                }
            }
        }
        p
    }

    #[test]
    fn the_precursor_profile_is_found_where_the_fragments_are_and_a_shadow_is_not() {
        let m = model();
        let pm = 600.0;
        let frag_mz = [300.0, 400.0, 500.0, 700.0];
        let cand = CandQ1::new(&m, pm, 2, 100.0, &frag_mz, &[1; 4], &[1.0; 4], &|_| 1.0);
        let pc = CandQ1::precursor(&m, pm, 2, 100.0, PREC_ISOTOPES, &|_| 1.0);
        // Three fragments of this precursor and one of a neighbour three steps above.
        let centres: Vec<Option<f64>> = (0..4)
            .map(|f| {
                let c = cand.frags[f].centre;
                Some(if f == 3 { c + 3.0 * m.step_mz } else { c })
            })
            .collect();
        let fp = synth(&m, &cand, &centres, &[100.0, 80.0, 60.0, 90.0]);
        let d = decide(&fp, &cand, &m, Some(&m.tolerances), 3);
        // The precursor's isotopes, each at its own m/z and Q1 position.
        let pcen: Vec<Option<f64>> = pc.frags.iter().map(|f| Some(f.centre)).collect();
        let mut pp = synth(&m, &cand, &pcen, &[50.0, 30.0, 10.0]);
        pp.s0 = fp.s0;
        let (v, env) = precursor_scores(&fp, &d, &pp, &pc, &m);
        assert_eq!(v.len(), Q1P_COLUMNS.len());
        assert_eq!(v[0], 3.0, "every isotope present");
        assert!(v[1] > 0.0 && v[1] < 1.0);
        assert!(
            v[2].abs() <= 0.25,
            "the precursor sits where expected: {}",
            v[2]
        );
        assert!(v[3] < 0.05, "it fits the window: {}", v[3]);
        assert!(v[4] > 0.5 && v[5] > 0.9);
        assert_eq!(v[6], 3.0, "the three own fragments, not the neighbour's");
        assert!(v[7] < 0.5);
        assert!(env.iter().any(|&x| x > 0.0));
        // No precursor in the patch: nothing present, no fit.
        let empty = Patch::new(fp.cycles.clone(), fp.s0, fp.t.clone(), PREC_ISOTOPES);
        let (v0, env0) = precursor_scores(&fp, &d, &empty, &pc, &m);
        assert_eq!(v0[0], 0.0);
        assert_eq!(v0[2], 0.0);
        assert_eq!(v0[3], 1.0);
        assert_eq!(v0[4], 0.0);
        assert_eq!(v0[6], 0.0);
        assert_eq!(v0[7], MU_MAX);
        assert!(env0.iter().all(|&x| x == 0.0));
        // A precursor three steps off its expected position (a co-isolated one only).
        let off: Vec<Option<f64>> = pc
            .frags
            .iter()
            .map(|f| Some(f.centre + 3.0 * m.step_mz))
            .collect();
        let pp2 = synth(&m, &cand, &off, &[50.0, 30.0, 10.0]);
        let (v2, _) = precursor_scores(&fp, &d, &pp2, &pc, &m);
        assert!((v2[2] - 3.0).abs() <= 0.25, "shift recovered: {}", v2[2]);
        assert!(
            v2[6] <= 1.0,
            "own fragments do not sit at the shifted precursor"
        );
        assert!(v2[7] > 2.5);
        // Deterministic.
        assert_eq!(precursor_scores(&fp, &d, &pp, &pc, &m).0, v);
    }

    #[test]
    fn multiscale_scores_separate_own_fragments_from_a_shadow() {
        let m = model();
        let pm = 600.0;
        let frag_mz = [300.0, 400.0, 500.0, 700.0];
        let cand = CandQ1::new(&m, pm, 2, 100.0, &frag_mz, &[1; 4], &[1.0; 4], &|_| 1.0);
        let own: Vec<Option<f64>> = cand.frags.iter().map(|f| Some(f.centre)).collect();
        let fp = synth(&m, &cand, &own, &[100.0, 80.0, 60.0, 90.0]);
        let d = decide(&fp, &cand, &m, Some(&m.tolerances), 3);
        let v = multiscale(&fp, &cand, &d, &m, None);
        assert_eq!(v.len(), Q1M_COLUMNS.len());
        for (k, _) in SCALES.iter().enumerate() {
            let (corr, shape, centroid) = (v[3 * k], v[3 * k + 1], v[3 * k + 2]);
            // Three steps sit on the window's flat top: their shape says little, and their
            // centroid is the middle step's target, up to half a step from the centre.
            if k > 0 {
                assert!(corr > 0.99, "scale {k}: {corr}");
                assert!(shape < 0.05, "scale {k}: {shape}");
                assert!(centroid.abs() < 0.3, "scale {k}: {centroid}");
            } else {
                assert!(centroid.abs() <= 0.75, "scale {k}: {centroid}");
            }
        }
        // Every fragment of a neighbour three steps up: the wide scales see the shift.
        let shadow: Vec<Option<f64>> = cand
            .frags
            .iter()
            .map(|f| Some(f.centre + 3.0 * m.step_mz))
            .collect();
        let fs = synth(&m, &cand, &shadow, &[100.0, 80.0, 60.0, 90.0]);
        let ds = decide(&fs, &cand, &m, Some(&m.tolerances), 3);
        let vs = multiscale(&fs, &cand, &ds, &m, None);
        assert!(vs[8].abs() > 1.0, "11-step centroid off: {}", vs[8]);
        // With the precursor's profile: the precursor columns follow.
        let env: Vec<f64> = (0..fp.ns())
            .map(|si| transmission(fp.t[si] - cand.centre, 1.9, 1.5))
            .collect();
        let vp = multiscale(&fp, &cand, &d, &m, Some(&env));
        assert_eq!(vp.len(), q1m_columns(true).len());
        assert_eq!(&vp[..9], &v[..]);
        assert!(vp[10] > 0.9 && vp[11] > 0.9);
        // A patch without signal.
        let none = Patch::new(fp.cycles.clone(), fp.s0, fp.t.clone(), 4);
        let dn = decide(&none, &cand, &m, Some(&m.tolerances), 3);
        assert_eq!(
            multiscale(&none, &cand, &dn, &m, None),
            empty_multiscale(false)
        );
        assert_eq!(multiscale(&fp, &cand, &d, &m, None), v);
    }

    #[test]
    fn the_columns_match_the_values() {
        assert_eq!(empty_precursor().len(), Q1P_COLUMNS.len());
        assert_eq!(empty_multiscale(false).len(), q1m_columns(false).len());
        assert_eq!(empty_multiscale(true).len(), q1m_columns(true).len());
        assert_eq!(q1m_columns(true)[9..], *Q1M_PREC_COLUMNS);
    }
}
