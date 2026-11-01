//! The fragment ions of a set of placement hypotheses (docs/36_ptm_localisation.md).
//!
//! Every hypothesis gives the same b and y positions (ion type, ordinal, charge); they
//! differ only in how many localised modifications each fragment carries. An ion is one
//! distinct m/z over all hypotheses, with the set of hypotheses that predict it. An ion
//! every hypothesis predicts is common: it carries no localisation information and serves
//! to measure how often a real fragment of this peptide is detected. An ion only some
//! hypotheses predict is site-determining. A modification count no hypothesis gives a
//! position (for example b2 carrying a site when neither of its residues is a candidate)
//! is an impossible variant: never a fragment of this peptide under any placement, it
//! measures how often a site-determining m/z is seen when it is not there.

use mumdia_core::constants::residue_mass;
use mumdia_core::constants::{PROTON, WATER};
use std::collections::BTreeMap;

use super::hypotheses::Split;

/// The four ion classes the detection and false-presence rates are kept for:
/// `(b, 1+)`, `(b, 2+)`, `(y, 1+)`, `(y, 2+)`.
pub(crate) const N_CLASSES: usize = 4;

/// Class index of an ion type (`false` = b, `true` = y) at a fragment charge (1 or 2).
#[inline]
pub(crate) fn class_of(is_y: bool, z: u8) -> usize {
    usize::from(is_y) * 2 + usize::from(z >= 2)
}

/// One distinct fragment m/z over a peptide's hypotheses.
#[derive(Clone, Debug)]
pub(crate) struct Ion {
    pub mz: f64,
    pub class: usize,
    /// Which hypotheses predict it, by hypothesis index.
    pub member: Vec<bool>,
}

/// The ions of a hypothesis set and the impossible variants.
#[derive(Clone, Debug)]
pub(crate) struct IonSet {
    /// Every distinct m/z, ascending.
    pub ions: Vec<Ion>,
    /// `(m/z, class)` of every impossible variant, ascending in m/z.
    pub impossible: Vec<(f64, usize)>,
}

impl IonSet {
    /// Whether ion `j` is predicted by every hypothesis.
    pub fn is_common(&self, j: usize) -> bool {
        self.ions[j].member.iter().all(|&m| m)
    }
}

/// m/z key: 1e-4 Th, so two routes to the same mass are one ion.
#[inline]
fn key(mz: f64) -> i64 {
    (mz * 1e4).round() as i64
}

/// b1..b(n-1) and y1..y(n-1) at charges 1..=`zmax` for `s` under each of `hyps`, and the
/// impossible variants.
pub(crate) fn build(s: &Split, hyps: &[Vec<usize>], zmax: u8) -> IonSet {
    let n = s.residues.len();
    let h = hyps.len();
    // Base prefix masses (no localised modification): pre[o] = mass of the first o residues.
    let mut pre = vec![s.n_term; n + 1];
    for i in 0..n {
        pre[i + 1] =
            pre[i] + residue_mass(s.residues[i]).expect("validated residue") + s.base_mods[i];
    }
    let total = pre[n] + s.c_term;
    let k = hyps.first().map_or(0, |x| x.len());
    // ions keyed by m/z; BTreeMap so the ion order is ascending and deterministic
    let mut by_key: BTreeMap<i64, Ion> = BTreeMap::new();
    // per position: the modification counts hypotheses give it
    let mut positions: Vec<(bool, usize, u8, f64, Vec<bool>)> = Vec::new();
    for is_y in [false, true] {
        for o in 1..n {
            let base = if is_y {
                total - pre[n - o] + WATER
            } else {
                pre[o]
            };
            for z in 1..=zmax {
                let mut taken = vec![false; k + 1];
                for (hi, sites) in hyps.iter().enumerate() {
                    let c = if is_y {
                        sites.iter().filter(|&&i| i >= n - o).count()
                    } else {
                        sites.iter().filter(|&&i| i < o).count()
                    };
                    taken[c] = true;
                    let mz = (base + c as f64 * s.delta + z as f64 * PROTON) / z as f64;
                    let e = by_key.entry(key(mz)).or_insert_with(|| Ion {
                        mz,
                        class: class_of(is_y, z),
                        member: vec![false; h],
                    });
                    e.member[hi] = true;
                }
                positions.push((is_y, o, z, base, taken));
            }
        }
    }
    let mut impossible = Vec::new();
    for (is_y, _o, z, base, taken) in &positions {
        for (c, &t) in taken.iter().enumerate() {
            if t {
                continue;
            }
            let mz = (base + c as f64 * s.delta + *z as f64 * PROTON) / *z as f64;
            if !by_key.contains_key(&key(mz)) {
                impossible.push((mz, class_of(*is_y, *z)));
            }
        }
    }
    impossible.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    IonSet {
        ions: by_key.into_values().collect(),
        impossible,
    }
}

#[cfg(test)]
mod tests {
    use super::super::hypotheses::{enumerate, split};
    use super::*;
    use mumdia_core::mass::parse_peptidoform;

    /// Under each hypothesis the ions are exactly the b/y ladder of that peptidoform.
    #[test]
    fn ions_are_the_ladders_of_the_placements() {
        let s = split("PS[Phospho]ETK", "Phospho").unwrap();
        let hyps = vec![vec![1], vec![3]];
        let set = build(&s, &hyps, 1);
        for (hi, pf) in ["PS[Phospho]ETK", "PSET[Phospho]K"].iter().enumerate() {
            let p = parse_peptidoform(pf).unwrap();
            let mut want: Vec<i64> = Vec::new();
            // the engine's ladder drops b1, y1 and y2; this one keeps them, so add them
            let n = p.residues.len();
            let mut pre = 0.0;
            for i in 0..n - 1 {
                pre += residue_mass(p.residues[i]).unwrap() + p.mods[i];
                want.push(key(pre + PROTON));
            }
            let mut suf = WATER;
            for i in (1..n).rev() {
                suf += residue_mass(p.residues[i]).unwrap() + p.mods[i];
                want.push(key(suf + PROTON));
            }
            want.sort_unstable();
            want.dedup();
            let mut got: Vec<i64> = set
                .ions
                .iter()
                .filter(|ion| ion.member[hi])
                .map(|ion| key(ion.mz))
                .collect();
            got.sort_unstable();
            assert_eq!(got, want, "{pf}");
        }
    }

    /// Site-determining ions are exactly those spanning the residues between the two
    /// candidate sites: b2, b3 (and y2, y3 from the other end) for sites 1 and 3 of PSETK.
    #[test]
    fn site_determining_ions_lie_between_the_sites() {
        let s = split("PS[Phospho]ETK", "Phospho").unwrap();
        let hyps = enumerate(&[1, 3], 1, 10).unwrap();
        let set = build(&s, &hyps, 1);
        let sdi: Vec<usize> = (0..set.ions.len()).filter(|&j| !set.is_common(j)).collect();
        // b2, b3 and y2, y3, each in two variants (with and without the site)
        assert_eq!(sdi.len(), 8);
        let common = set.ions.len() - sdi.len();
        // b1, b4, y1, y4
        assert_eq!(common, 4);
        // impossible: b1 with a site, b4 without, y1 with, y4 without
        assert_eq!(set.impossible.len(), 4);
        for &(mz, _) in &set.impossible {
            assert!(set.ions.iter().all(|i| key(i.mz) != key(mz)));
        }
    }

    #[test]
    fn classes_split_type_and_charge() {
        assert_eq!(class_of(false, 1), 0);
        assert_eq!(class_of(false, 2), 1);
        assert_eq!(class_of(true, 1), 2);
        assert_eq!(class_of(true, 2), 3);
        let s = split("PS[Phospho]ETK", "Phospho").unwrap();
        let set = build(&s, &[vec![1], vec![3]], 2);
        assert!(set.ions.iter().any(|i| i.class == 1));
        assert!(set.ions.iter().any(|i| i.class == 3));
    }
}
