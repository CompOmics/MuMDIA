//! Database-free sequence tags for the prescreen's optional components.
//!
//! Inputs are the spectra and the residue-state alphabet only: no library sequence, FASTA
//! peptide or identification proposes, ranks or prunes a tag. A tag is a path of peaks in one
//! spectrum whose successive neutral mass differences match residue states within
//! `edge_tol_da` (0.005 Da neutral, so `0.005 / z` in m/z at fragment charge `z`). Every
//! permitted edge is allocated exactly (`Graph`, compressed sparse rows): there is no peak,
//! degree or path-count cap, and every alternative assignment of a peak pair is kept.
//!
//! A conventional trimer is three residue differences over four peaks. With `gap_edges` a step
//! may instead be a two-residue mass, each compatible ordered pair kept as its own alternative;
//! the resulting three-peak ladder is recorded under a separate key bit, so a shorter ladder is
//! never confused with a four-peak one.
//!
//! Per spectrum the discovery keeps, for each reversal-canonical trimer, fragment charge and
//! ladder kind, the minimum whole-path RMS: the residual spread after fitting one common ladder
//! offset (`path_rms`). Repeated occurrences are summarised, never counted as independent
//! evidence. Endpoint queries (`matched_path`, `complementary_path`) re-traverse the original
//! peaks, so alternatives lost by keeping only the best fit stay reachable.

use std::collections::HashMap;

use anyhow::Result;
use mumdia_core::config::Config;
use mumdia_core::constants::{residue_mass, PROTON};
use mumdia_core::mass::{parse_peptidoform, unimod_mass};

/// Mass agreement used to map a peptidoform's per-residue delta onto an alphabet state.
const STATE_MASS_TOL: f64 = 1e-3;

/// Residue states the tags are written in: the 20 standard residues with I and L merged, each
/// fixed modification replacing its residue's plain state, each variable modification an extra
/// state whose backbone (`normal`) is the residue's plain state.
#[derive(Clone, Debug)]
pub struct Alphabet {
    pub masses: Vec<f64>,
    pub names: Vec<String>,
    /// State -> its unmodified backbone state (itself for a plain or fixed state).
    pub normal: Vec<u16>,
    /// `(residue, delta, state)`; the plain/fixed state of a residue has its fixed delta or 0.
    states: Vec<(u8, f64, u16)>,
}

impl Alphabet {
    /// From the chemistry configuration (`peptidoforms.fixed_mods` / `variable_mods`) plus
    /// `prescreen.tags.extra_mods` (`RESIDUE:Name`).
    pub fn from_config(cfg: &Config) -> Result<Alphabet> {
        let fixed: Vec<(u8, String)> = cfg
            .peptidoforms
            .fixed_mods
            .iter()
            .map(|m| (m.residue as u8, m.name.clone()))
            .collect();
        let mut variable: Vec<(u8, String)> = cfg
            .peptidoforms
            .variable_mods
            .iter()
            .map(|m| (m.residue as u8, m.name.clone()))
            .collect();
        for s in &cfg.prescreen.tags.extra_mods {
            let (r, n) = s.split_once(':').ok_or_else(|| {
                anyhow::anyhow!("prescreen.tags.extra_mods entry '{s}' must be RESIDUE:Name")
            })?;
            variable.push((r.as_bytes()[0].to_ascii_uppercase(), n.to_string()));
        }
        Alphabet::build(&fixed, &variable)
    }

    pub fn build(fixed: &[(u8, String)], variable: &[(u8, String)]) -> Result<Alphabet> {
        const PLAIN: &[u8] = b"GASPVTCLNDQKEMHFRYW";
        let delta = |name: &str| {
            unimod_mass(name).ok_or_else(|| {
                anyhow::anyhow!("prescreen tag alphabet: unknown modification '{name}'")
            })
        };
        let mut a = Alphabet {
            masses: Vec::new(),
            names: Vec::new(),
            normal: Vec::new(),
            states: Vec::new(),
        };
        for &r in PLAIN {
            let base = residue_mass(r).expect("standard residue");
            let fx = fixed.iter().find(|(fr, _)| fr.to_ascii_uppercase() == r);
            let (d, name) = match fx {
                Some((_, n)) => (delta(n)?, format!("{}[{n}]", r as char)),
                None => (0.0, (r as char).to_string()),
            };
            let s = a.masses.len() as u16;
            a.masses.push(base + d);
            a.names.push(name);
            a.normal.push(s);
            a.states.push((r, d, s));
            if r == b'L' {
                a.states.push((b'I', d, s));
            }
        }
        for (r, n) in variable {
            let r = r.to_ascii_uppercase();
            let plain = a
                .states
                .iter()
                .find(|(x, _, _)| *x == r)
                .map(|&(_, d, s)| (d, s))
                .ok_or_else(|| {
                    anyhow::anyhow!("prescreen tag alphabet: no residue '{}'", r as char)
                })?;
            let d = plain.0 + delta(n)?;
            if a.states
                .iter()
                .any(|&(x, y, _)| x == r && (y - d).abs() < STATE_MASS_TOL)
            {
                continue;
            }
            let s = a.masses.len() as u16;
            a.masses
                .push(residue_mass(r).expect("standard residue") + d);
            a.names.push(format!("{}[{n}]", r as char));
            a.normal.push(plain.1);
            a.states.push((r, d, s));
            if r == b'L' {
                a.states.push((b'I', d, s));
            }
        }
        Ok(a)
    }

    pub fn n(&self) -> u32 {
        self.masses.len() as u32
    }

    /// The state of residue `r` carrying total delta `d`, if the alphabet has one.
    pub fn state(&self, r: u8, d: f64) -> Option<u16> {
        self.states
            .iter()
            .find(|&&(x, y, _)| x == r && (y - d).abs() < STATE_MASS_TOL)
            .map(|&(_, _, s)| s)
    }

    /// A peptidoform as states, `None` at a position the alphabet cannot express (an
    /// unconfigured modification, or a terminal modification on that end residue). Such
    /// positions break the trimers covering them; they never get a substitute state.
    pub fn tokenise(&self, pf: &str) -> Option<Vec<Option<u16>>> {
        let p = parse_peptidoform(pf).ok()?;
        let l = p.residues.len();
        let mut out: Vec<Option<u16>> = p
            .residues
            .iter()
            .zip(&p.mods)
            .map(|(&r, &d)| self.state(r, d))
            .collect();
        if p.n_term_mod != 0.0 && l > 0 {
            out[0] = None;
        }
        if p.c_term_mod != 0.0 && l > 0 {
            out[l - 1] = None;
        }
        Some(out)
    }

    /// Reversal-canonical trimer code `min(abc, cba)` in base `n`.
    #[inline]
    pub fn canonical(&self, a: u16, b: u16, c: u16) -> u32 {
        let n = self.n();
        let f = (a as u32 * n + b as u32) * n + c as u32;
        let r = (c as u32 * n + b as u32) * n + a as u32;
        f.min(r)
    }

    /// The trimer code with every state replaced by its backbone state.
    pub fn backbone(&self, code: u32) -> u32 {
        let n = self.n();
        let t = code % (n * n * n);
        let (a, b, c) = (t / (n * n), (t / n) % n, t % n);
        self.canonical(
            self.normal[a as usize],
            self.normal[b as usize],
            self.normal[c as usize],
        )
    }

    /// Full key: trimer code, fragment charge and ladder kind (four-peak or gapped three-peak).
    #[inline]
    pub fn key(&self, code: u32, z: i32, gapped: bool) -> u32 {
        let n3 = self.n() * self.n() * self.n();
        code + n3 * ((z as u32 - 1) + 2 * gapped as u32)
    }

    /// `(trimer code, charge, gapped)` of a key.
    pub fn unkey(&self, key: u32) -> (u32, i32, bool) {
        let n3 = self.n() * self.n() * self.n();
        let k = key / n3;
        (key % n3, (k % 2) as i32 + 1, k >= 2)
    }
}

/// Peak graph for one fragment charge: edge `i -> j` labelled with a residue state (or, with gap
/// edges, a two-residue pair) when `mz[j] - mz[i]` lies within the step's mass +/- tol over z.
pub struct Graph {
    pub off: Vec<usize>,
    pub target: Vec<u32>,
    /// Single-residue edge: the state. Gap edge: `GAP_FLAG | pair index`.
    pub label: Vec<u32>,
}

pub const GAP_FLAG: u32 = 1 << 31;

/// Two-residue steps for gap edges: every ordered state pair, kept as separate alternatives.
pub fn pair_steps(alpha: &Alphabet) -> Vec<(f64, u16, u16)> {
    let n = alpha.n() as u16;
    let mut v = Vec::with_capacity((n as usize) * (n as usize));
    for a in 0..n {
        for b in 0..n {
            v.push((alpha.masses[a as usize] + alpha.masses[b as usize], a, b));
        }
    }
    v
}

impl Graph {
    /// Exact allocation: one pass counts, one fills. `mz` ascending.
    pub fn build(
        mz: &[f64],
        z: i32,
        tol: f64,
        alpha: &Alphabet,
        pairs: Option<&[(f64, u16, u16)]>,
    ) -> Graph {
        let zf = z as f64;
        let mut steps: Vec<(f64, u32)> = alpha
            .masses
            .iter()
            .enumerate()
            .map(|(s, &m)| (m, s as u32))
            .collect();
        if let Some(p) = pairs {
            steps.extend(
                p.iter()
                    .enumerate()
                    .map(|(i, &(m, _, _))| (m, GAP_FLAG | i as u32)),
            );
        }
        let mut off = vec![0usize; mz.len() + 1];
        let mut target = Vec::new();
        let mut label = Vec::new();
        for i in 0..mz.len() {
            for &(m, lab) in &steps {
                let lo = mz[i] + (m - tol) / zf;
                let hi = mz[i] + (m + tol) / zf;
                let mut j = mz.partition_point(|&x| x < lo);
                while j < mz.len() && mz[j] <= hi {
                    target.push(j as u32);
                    label.push(lab);
                    j += 1;
                }
            }
            off[i + 1] = target.len();
        }
        Graph { off, target, label }
    }

    pub fn edges(&self, i: usize) -> std::ops::Range<usize> {
        self.off[i]..self.off[i + 1]
    }
}

/// RMS of the ladder residuals after fitting one common offset: `steps[k]` is the cumulative
/// expected neutral mass from the first peak to peak `k + 1`. The first peak's residual is 0.
pub fn path_rms(mz: &[f64], nodes: &[usize], cumulative: &[f64], z: i32) -> f64 {
    let zf = z as f64;
    let n = nodes.len() as f64;
    let mut sum = 0.0;
    let mut sq = 0.0;
    for (k, &cm) in cumulative.iter().enumerate() {
        let r = zf * (mz[nodes[k + 1]] - mz[nodes[0]]) - cm;
        sum += r;
        sq += r * r;
    }
    let mean = sum / n;
    (sq / n - mean * mean).max(0.0).sqrt()
}

/// Soft whole-path weight `exp(-0.5 (rms / sigma)^2)`.
#[inline]
pub fn fit_weight(rms: f64, sigma: f64) -> f64 {
    (-0.5 * (rms / sigma).powi(2)).exp()
}

/// One spectrum's tags: `(key, minimum rms)` ascending by key, and the number of paths walked.
pub struct SpectrumTags {
    pub keys: Vec<u32>,
    pub rms: Vec<f32>,
    pub paths: u64,
}

impl SpectrumTags {
    pub fn get(&self, key: u32) -> Option<f32> {
        self.keys.binary_search(&key).ok().map(|i| self.rms[i])
    }
}

/// Every tag path of one spectrum at fragment charges `1..=max_charge`.
pub fn discover(
    mz: &[f64],
    alpha: &Alphabet,
    tol: f64,
    max_charge: i32,
    pairs: Option<&[(f64, u16, u16)]>,
) -> SpectrumTags {
    let mut best: HashMap<u32, f64> = HashMap::new();
    let mut paths = 0u64;
    let mut put = |key: u32, rms: f64| {
        let e = best.entry(key).or_insert(f64::INFINITY);
        if rms < *e {
            *e = rms;
        }
    };
    let m = |s: u32| alpha.masses[s as usize];
    for z in 1..=max_charge {
        let g = Graph::build(mz, z, tol, alpha, pairs);
        for i in 0..mz.len() {
            for e in g.edges(i) {
                let (j, la) = (g.target[e] as usize, g.label[e]);
                for f in g.edges(j) {
                    let (k, lb) = (g.target[f] as usize, g.label[f]);
                    match (la & GAP_FLAG != 0, lb & GAP_FLAG != 0) {
                        (false, false) => {
                            for h in g.edges(k) {
                                let (l, lc) = (g.target[h] as usize, g.label[h]);
                                if lc & GAP_FLAG != 0 {
                                    continue;
                                }
                                let (a, b, c) = (la as u16, lb as u16, lc as u16);
                                let cum = [m(la), m(la) + m(lb), m(la) + m(lb) + m(lc)];
                                let rms = path_rms(mz, &[i, j, k, l], &cum, z);
                                paths += 1;
                                put(alpha.key(alpha.canonical(a, b, c), z, false), rms);
                            }
                        }
                        // A gap step and a single step: a three-peak ladder of three residues.
                        (true, false) => {
                            let p =
                                pairs.expect("gap label implies pairs")[(la & !GAP_FLAG) as usize];
                            let cum = [p.0, p.0 + m(lb)];
                            let rms = path_rms(mz, &[i, j, k], &cum, z);
                            paths += 1;
                            put(
                                alpha.key(alpha.canonical(p.1, p.2, lb as u16), z, true),
                                rms,
                            );
                        }
                        (false, true) => {
                            let p =
                                pairs.expect("gap label implies pairs")[(lb & !GAP_FLAG) as usize];
                            let cum = [m(la), m(la) + p.0];
                            let rms = path_rms(mz, &[i, j, k], &cum, z);
                            paths += 1;
                            put(
                                alpha.key(alpha.canonical(la as u16, p.1, p.2), z, true),
                                rms,
                            );
                        }
                        (true, true) => {}
                    }
                }
            }
        }
    }
    let mut v: Vec<(u32, f64)> = best.into_iter().collect();
    v.sort_unstable_by_key(|x| x.0);
    SpectrumTags {
        keys: v.iter().map(|x| x.0).collect(),
        rms: v.iter().map(|x| x.1 as f32).collect(),
        paths,
    }
}

/// Best soft fit of a path from a peak near `start` through states `a, b, c` to a peak near
/// `end` (both within `0.01 / z`), re-traversing the original peaks: the prototype's
/// `matched_path`, used for positioned flank support.
#[allow(clippy::too_many_arguments)]
pub fn matched_path(
    mz: &[f64],
    start: f64,
    end: f64,
    s: [u16; 3],
    z: i32,
    alpha: &Alphabet,
    tol: f64,
    sigma: f64,
) -> f64 {
    let zf = z as f64;
    let ends = 0.01 / zf;
    let m = |k: usize| alpha.masses[s[k] as usize];
    let range = |from: f64, step: f64| {
        let lo = from + (step - tol) / zf;
        let hi = from + (step + tol) / zf;
        let a = mz.partition_point(|&x| x < lo);
        let b = mz.partition_point(|&x| x <= hi);
        a..b
    };
    let mut best = 0.0f64;
    let i0 = mz.partition_point(|&x| x < start - ends);
    let i1 = mz.partition_point(|&x| x <= start + ends);
    for i in i0..i1 {
        for j in range(mz[i], m(0)) {
            for k in range(mz[j], m(1)) {
                for l in range(mz[k], m(2)) {
                    if (mz[l] - end).abs() <= ends {
                        let cum = [m(0), m(0) + m(1), m(0) + m(1) + m(2)];
                        let rms = path_rms(mz, &[i, j, k, l], &cum, z);
                        best = best.max(fit_weight(rms, sigma));
                    }
                }
            }
        }
    }
    best
}

/// Index of the nearest peak to `x` within `tol`, first on a tie.
pub fn nearest(mz: &[f64], x: f64, tol: f64) -> Option<usize> {
    let mut j = mz.partition_point(|&m| m < x - tol);
    let mut best = None;
    let mut err = tol + 1.0;
    while j < mz.len() && mz[j] <= x + tol {
        let d = (mz[j] - x).abs();
        if d < err {
            best = Some(j);
            err = d;
        }
        j += 1;
    }
    best
}

/// Complementary support of a tag `a, b, c` at fragment charge `z` for a candidate of neutral
/// mass `peptide_mass`: over every observed path of the tag, the complements `M - z (mz - H+)`
/// of its four peaks are looked up at charges `1..=max_charge`; a complement counts only on a
/// peak that is not on the path and not already used. Two, three and four distinct complements
/// give support 1/3, 2/3 and 1, times the path's soft fit. The prototype's
/// `complementary_path`.
#[allow(clippy::too_many_arguments)]
pub fn complementary_path(
    mz: &[f64],
    s: [u16; 3],
    z: i32,
    peptide_mass: f64,
    max_charge: i32,
    alpha: &Alphabet,
    tol: f64,
    sigma: f64,
) -> f64 {
    let zf = z as f64;
    let m = |k: usize| alpha.masses[s[k] as usize];
    let maximum = (peptide_mass - m(0) - m(1) - m(2)) / zf + PROTON;
    let stop = mz.partition_point(|&x| x < maximum);
    let range = |from: f64, step: f64| {
        let lo = from + (step - tol) / zf;
        let hi = from + (step + tol) / zf;
        mz.partition_point(|&x| x < lo)..mz.partition_point(|&x| x <= hi)
    };
    let mut best = 0.0f64;
    for i in 0..stop {
        for j in range(mz[i], m(0)) {
            for k in range(mz[j], m(1)) {
                for l in range(mz[k], m(2)) {
                    let nodes = [i, j, k, l];
                    let mut used: Vec<usize> = Vec::with_capacity(4);
                    for &p in &nodes {
                        let complement = peptide_mass - zf * (mz[p] - PROTON);
                        for oz in 1..=max_charge {
                            let ozf = oz as f64;
                            let Some(o) = nearest(mz, complement / ozf + PROTON, 0.005 / ozf)
                            else {
                                continue;
                            };
                            if nodes.contains(&o) || used.contains(&o) {
                                continue;
                            }
                            used.push(o);
                            break;
                        }
                    }
                    if used.len() >= 2 {
                        let cum = [m(0), m(0) + m(1), m(0) + m(1) + m(2)];
                        let fit = fit_weight(path_rms(mz, &nodes, &cum, z), sigma);
                        best = best.max(fit * (used.len() - 1) as f64 / 3.0);
                    }
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alpha() -> Alphabet {
        Alphabet::build(
            &[(b'C', "Carbamidomethyl".into())],
            &[(b'M', "Oxidation".into())],
        )
        .unwrap()
    }

    fn st(a: &Alphabet, r: u8) -> u16 {
        a.state(r, if r == b'C' { 57.021_463_735 } else { 0.0 })
            .unwrap()
    }

    #[test]
    fn the_alphabet_merges_i_and_l_and_follows_the_chemistry_config() {
        let a = alpha();
        assert_eq!(a.n(), 20, "19 plain states (I = L) + oxidised M");
        assert_eq!(a.state(b'I', 0.0), a.state(b'L', 0.0));
        assert!(a.state(b'C', 0.0).is_none(), "fixed CAM replaces plain C");
        let ox = a.state(b'M', 15.994_914_62).unwrap();
        assert_eq!(a.normal[ox as usize], a.state(b'M', 0.0).unwrap());
        // An unconfigured modification is unexpressible, never substituted.
        let t = a.tokenise("PEPS[Phospho]TIDEK").unwrap();
        assert_eq!(t.iter().filter(|s| s.is_none()).count(), 1);
        let t = a.tokenise("[Acetyl]-PEPTIDEK").unwrap();
        assert!(
            t[0].is_none(),
            "terminal modification breaks its end residue"
        );
    }

    #[test]
    fn canonical_codes_are_reversal_invariant_and_keys_round_trip() {
        let a = alpha();
        let (g, s, p) = (st(&a, b'G'), st(&a, b'S'), st(&a, b'P'));
        assert_eq!(a.canonical(g, s, p), a.canonical(p, s, g));
        for z in [1, 2] {
            for gapped in [false, true] {
                let k = a.key(a.canonical(g, s, p), z, gapped);
                assert_eq!(a.unkey(k), (a.canonical(g, s, p), z, gapped));
            }
        }
    }

    /// A ladder G-A-S from 200.0 at charge 1.
    fn ladder(a: &Alphabet, start: f64, res: &[u8], z: f64) -> Vec<f64> {
        let mut v = vec![start];
        for &r in res {
            let last = *v.last().unwrap();
            v.push(last + a.masses[st(a, r) as usize] / z);
        }
        v
    }

    #[test]
    fn a_peak_keeps_more_than_64_alternative_edges() {
        let a = alpha();
        let g = a.masses[st(&a, b'G') as usize];
        let mut mz = vec![200.0];
        // 100 alternative second peaks, all inside the 0.005 Da edge tolerance of +G.
        for k in 0..100 {
            mz.push(200.0 + g - 0.002 + k as f64 * 0.00004);
        }
        let third = 200.0 + g + a.masses[st(&a, b'A') as usize];
        mz.push(third);
        mz.push(third + a.masses[st(&a, b'S') as usize]);
        mz.sort_by(f64::total_cmp);
        let gr = Graph::build(&mz, 1, 0.005, &a, None);
        let from0 = gr
            .edges(0)
            .filter(|&e| gr.label[e] == st(&a, b'G') as u32)
            .count();
        assert_eq!(from0, 100, "no degree cap");
        let t = discover(&mz, &a, 0.005, 1, None);
        assert!(t.paths >= 100, "every alternative path walked: {}", t.paths);
        let key = a.key(
            a.canonical(st(&a, b'G'), st(&a, b'A'), st(&a, b'S')),
            1,
            false,
        );
        let rms = t.get(key).unwrap();
        assert!(rms < 1e-4, "the minimum-RMS alternative is kept: {rms}");
    }

    /// The edge tolerance is a neutral mass: 0.005 Da at charge 1, 0.0025 m/z at charge 2.
    #[test]
    fn edge_tolerance_is_charge_aware() {
        let a = alpha();
        let g = a.masses[st(&a, b'G') as usize];
        let edge = |d: f64, z: i32| {
            let mz = vec![300.0, 300.0 + g / z as f64 + d];
            Graph::build(&mz, z, 0.005, &a, None).target.contains(&1)
        };
        assert!(edge(0.0049, 1));
        assert!(!edge(0.0051, 1));
        assert!(edge(0.0024, 2));
        assert!(!edge(0.0026, 2));
    }

    #[test]
    fn whole_path_rms_ignores_a_common_offset_and_sees_accumulated_error() {
        let a = alpha();
        let res = [b'G', b'A', b'S'];
        let cum = {
            let m: Vec<f64> = res.iter().map(|&r| a.masses[st(&a, r) as usize]).collect();
            [m[0], m[0] + m[1], m[0] + m[1] + m[2]]
        };
        let exact = ladder(&a, 400.0, &res, 1.0);
        assert!(path_rms(&exact, &[0, 1, 2, 3], &cum, 1) < 1e-9);
        let shifted: Vec<f64> = exact.iter().map(|x| x + 0.003).collect();
        assert!(
            path_rms(&shifted, &[0, 1, 2, 3], &cum, 1) < 1e-9,
            "offset invariant"
        );
        let mut drift = exact.clone();
        drift[2] += 0.004;
        drift[3] += 0.004;
        let r = path_rms(&drift, &[0, 1, 2, 3], &cum, 1);
        assert!(r > 0.001, "{r}");
        assert!(fit_weight(r, 0.003) < 1.0 && fit_weight(0.0, 0.003) == 1.0);
    }

    /// A two-residue gap keeps every compatible assignment: A+V and G+L have the same mass, so
    /// one three-peak ladder yields gapped keys for AVS, VAS, GLS and LGS, and no four-peak key.
    #[test]
    fn gap_edges_keep_every_two_residue_assignment() {
        let a = alpha();
        let m = |r: u8| a.masses[st(&a, r) as usize];
        let gap = m(b'A') + m(b'V');
        assert!((gap - (m(b'G') + m(b'L'))).abs() < 0.005);
        let mz = vec![300.0, 300.0 + gap, 300.0 + gap + m(b'S')];
        let pairs = pair_steps(&a);
        let t = discover(&mz, &a, 0.005, 1, Some(&pairs));
        for (x, y) in [(b'A', b'V'), (b'V', b'A'), (b'G', b'L'), (b'L', b'G')] {
            let code = a.canonical(st(&a, x), st(&a, y), st(&a, b'S'));
            assert!(
                t.get(a.key(code, 1, true)).is_some(),
                "{}{}S",
                x as char,
                y as char
            );
            assert!(
                t.get(a.key(code, 1, false)).is_none(),
                "not a four-peak ladder"
            );
        }
        let without = discover(&mz, &a, 0.005, 1, None);
        assert!(without.keys.is_empty(), "two steps are not a trimer");
    }

    #[test]
    fn tags_from_peaks_ranked_below_300_need_the_uncapped_input() {
        let a = alpha();
        let tag = ladder(&a, 500.0, b"GAS", 1.0);
        let mut peaks: Vec<(f64, f64)> = (0..400)
            .map(|i| (1000.0 + i as f64 * 0.37, 1000.0))
            .collect();
        peaks.extend(tag.iter().map(|&m| (m, 1.0)));
        let key = a.key(
            a.canonical(st(&a, b'G'), st(&a, b'A'), st(&a, b'S')),
            1,
            false,
        );
        let mut all: Vec<f64> = peaks.iter().map(|p| p.0).collect();
        all.sort_by(f64::total_cmp);
        assert!(discover(&all, &a, 0.005, 1, None).get(key).is_some());
        let mut top = peaks.clone();
        top.sort_by(|x, y| y.1.total_cmp(&x.1));
        let mut capped: Vec<f64> = top[..300].iter().map(|p| p.0).collect();
        capped.sort_by(f64::total_cmp);
        assert!(discover(&capped, &a, 0.005, 1, None).get(key).is_none());
    }

    /// A complement counts only on a distinct peak off the path: when the candidate mass makes
    /// node 0 and node 3 each other's complements, neither counts.
    #[test]
    fn complementary_support_rejects_path_peaks_and_reuse() {
        let a = alpha();
        let s = [st(&a, b'G'), st(&a, b'A'), st(&a, b'S')];
        let path = ladder(&a, 400.0, b"GAS", 1.0);
        let m = (path[0] - PROTON) + (path[3] - PROTON);
        assert_eq!(complementary_path(&path, s, 1, m, 1, &a, 0.005, 0.003), 0.0);
        let mut mz = path.clone();
        mz.push(m - (path[1] - PROTON) + PROTON);
        mz.push(m - (path[2] - PROTON) + PROTON);
        mz.sort_by(f64::total_cmp);
        let one_third = complementary_path(&mz, s, 1, m, 1, &a, 0.005, 0.003);
        assert!((one_third - 1.0 / 3.0).abs() < 1e-6, "{one_third}");
    }

    #[test]
    fn matched_path_finds_a_positioned_ladder_only_at_its_endpoints() {
        let a = alpha();
        let s = [st(&a, b'G'), st(&a, b'A'), st(&a, b'S')];
        let mz = ladder(&a, 400.0, b"GAS", 1.0);
        assert!(matched_path(&mz, mz[0], mz[3], s, 1, &a, 0.005, 0.003) > 0.99);
        assert_eq!(
            matched_path(&mz, mz[0] + 0.05, mz[3], s, 1, &a, 0.005, 0.003),
            0.0
        );
    }
}
