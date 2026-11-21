//! Database-free sequence tags for the tag prefilter.
//!
//! Inputs are the spectra and the residue-state alphabet only: no library sequence, FASTA
//! peptide or identification proposes, ranks or prunes a tag. A tag is a path of peaks in one
//! spectrum whose successive neutral mass differences match residue states within
//! `edge_tol_da` (0.005 Da neutral, so `0.005 / z` in m/z at fragment charge `z`). Every
//! permitted edge is allocated exactly (`Graph`, compressed sparse rows): there is no peak,
//! degree or path-count cap, and every alternative assignment of a peak pair is kept.
//!
//! A trimer is three residue differences over four peaks. Per spectrum the discovery keeps each
//! reversal-canonical trimer and fragment charge once; repeated occurrences are never counted
//! as independent evidence.

use anyhow::Result;
use mumdia_core::config::{Config, ResidueMod, UnknownModPolicy};
use mumdia_core::constants::residue_mass;
use mumdia_core::mass::{parse_peptidoform, unimod_mass};

/// Mass agreement used to map a peptidoform's per-residue delta onto an alphabet state.
const STATE_MASS_TOL: f64 = 1e-3;

/// Residue states the tags are written in: the 20 standard residues with I and L merged, each
/// fixed modification replacing its residue's plain state, each variable modification an extra
/// state.
#[derive(Clone, Debug)]
pub struct Alphabet {
    pub masses: Vec<f64>,
    pub names: Vec<String>,
    /// `(residue, delta, state)`; the plain/fixed state of a residue has its fixed delta or 0.
    states: Vec<(u8, f64, u16)>,
}

impl Alphabet {
    /// From the chemistry configuration (`peptidoforms.fixed_mods` / `variable_mods`) plus
    /// `prescreen.tags.extra_mods` (`RESIDUE:Name`). Only residue rules become states, read as
    /// the digest reads them: a terminal rule (residue `*`) is no state, because `tokenise`
    /// already breaks the modified end residue, and under `unknown_modification = skip` an
    /// unknown name is dropped.
    pub fn from_config(cfg: &Config) -> Result<Alphabet> {
        let p = &cfg.peptidoforms;
        let residue_rule = |m: &&ResidueMod| {
            m.position.is_anywhere()
                && (p.unknown_modification == UnknownModPolicy::Error
                    || unimod_mass(&m.name).is_some())
        };
        let fixed: Vec<(u8, String)> = p
            .fixed_mods
            .iter()
            .filter(residue_rule)
            .map(|m| (m.residue as u8, m.name.clone()))
            .collect();
        let mut variable: Vec<(u8, String)> = p
            .variable_mods
            .iter()
            .filter(residue_rule)
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

    /// Full key: trimer code and fragment charge.
    #[inline]
    pub fn key(&self, code: u32, z: i32) -> u32 {
        code + self.n() * self.n() * self.n() * (z as u32 - 1)
    }
}

/// Peak graph for one fragment charge: edge `i -> j` labelled with a residue state when
/// `mz[j] - mz[i]` lies within the state's mass +/- tol over z.
pub struct Graph {
    pub off: Vec<usize>,
    pub target: Vec<u32>,
    /// The edge's residue state.
    pub label: Vec<u32>,
}

impl Graph {
    /// Exact allocation: one pass counts, one fills. `mz` ascending.
    pub fn build(mz: &[f64], z: i32, tol: f64, alpha: &Alphabet) -> Graph {
        let zf = z as f64;
        let steps: Vec<(f64, u32)> = alpha
            .masses
            .iter()
            .enumerate()
            .map(|(s, &m)| (m, s as u32))
            .collect();
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

/// One spectrum's tags: the keys ascending without repeats, and the number of paths walked.
pub struct SpectrumTags {
    pub keys: Vec<u32>,
    pub paths: u64,
}

/// Every tag path of one spectrum at fragment charges `1..=max_charge`.
pub fn discover(mz: &[f64], alpha: &Alphabet, tol: f64, max_charge: i32) -> SpectrumTags {
    let mut keys: Vec<u32> = Vec::new();
    let mut paths = 0u64;
    for z in 1..=max_charge {
        let g = Graph::build(mz, z, tol, alpha);
        for i in 0..mz.len() {
            for e in g.edges(i) {
                let (j, la) = (g.target[e] as usize, g.label[e] as u16);
                for f in g.edges(j) {
                    let (k, lb) = (g.target[f] as usize, g.label[f] as u16);
                    for h in g.edges(k) {
                        paths += 1;
                        keys.push(alpha.key(alpha.canonical(la, lb, g.label[h] as u16), z));
                    }
                }
            }
        }
    }
    keys.sort_unstable();
    keys.dedup();
    SpectrumTags { keys, paths }
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
        assert!(a.state(b'M', 15.994_914_62).is_some());
        // An unconfigured modification is unexpressible, never substituted.
        let t = a.tokenise("PEPS[Phospho]TIDEK").unwrap();
        assert_eq!(t.iter().filter(|s| s.is_none()).count(), 1);
        let t = a.tokenise("[Acetyl]-PEPTIDEK").unwrap();
        assert!(
            t[0].is_none(),
            "terminal modification breaks its end residue"
        );
    }

    /// A terminal rule, and an unknown name under `unknown_modification = skip`, leave the
    /// alphabet as the residue rules make it (a terminal rule used to stop the tag prefilter).
    #[test]
    fn terminal_and_skipped_rules_leave_the_alphabet_unchanged() {
        use mumdia_core::config::ModPosition;
        let mut c = Config::default();
        let base = Alphabet::from_config(&c).unwrap().n();
        c.peptidoforms.variable_mods.push(ResidueMod {
            residue: '*',
            name: "Acetyl".into(),
            position: ModPosition::ProteinNTerm,
        });
        c.peptidoforms.unknown_modification = UnknownModPolicy::Skip;
        c.peptidoforms.variable_mods.push(ResidueMod {
            residue: 'K',
            name: "NoSuchModification".into(),
            position: ModPosition::Anywhere,
        });
        assert_eq!((base, Alphabet::from_config(&c).unwrap().n()), (20, 20));
    }

    #[test]
    fn canonical_codes_are_reversal_invariant_and_keys_keep_the_charge() {
        let a = alpha();
        let (g, s, p) = (st(&a, b'G'), st(&a, b'S'), st(&a, b'P'));
        assert_eq!(a.canonical(g, s, p), a.canonical(p, s, g));
        let code = a.canonical(g, s, p);
        assert_ne!(a.key(code, 1), a.key(code, 2));
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
        let gr = Graph::build(&mz, 1, 0.005, &a);
        let from0 = gr
            .edges(0)
            .filter(|&e| gr.label[e] == st(&a, b'G') as u32)
            .count();
        assert_eq!(from0, 100, "no degree cap");
        let t = discover(&mz, &a, 0.005, 1);
        assert!(t.paths >= 100, "every alternative path walked: {}", t.paths);
        let key = a.key(a.canonical(st(&a, b'G'), st(&a, b'A'), st(&a, b'S')), 1);
        assert_eq!(t.keys, vec![key], "one key however many paths");
    }

    /// The edge tolerance is a neutral mass: 0.005 Da at charge 1, 0.0025 m/z at charge 2.
    #[test]
    fn edge_tolerance_is_charge_aware() {
        let a = alpha();
        let g = a.masses[st(&a, b'G') as usize];
        let edge = |d: f64, z: i32| {
            let mz = vec![300.0, 300.0 + g / z as f64 + d];
            Graph::build(&mz, z, 0.005, &a).target.contains(&1)
        };
        assert!(edge(0.0049, 1));
        assert!(!edge(0.0051, 1));
        assert!(edge(0.0024, 2));
        assert!(!edge(0.0026, 2));
    }
}
