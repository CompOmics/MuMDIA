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

/// The ion classes the detection and false-presence rates are kept for: `(b, 1+)`,
/// `(b, 2+)`, `(y, 1+)`, `(y, 2+)`, then the same four for neutral-loss ions. The loss
/// classes stay empty since `localise.neutral_loss` was removed in 0.15.0; they are kept so
/// the report keeps its shape.
pub(crate) const N_CLASSES: usize = 8;

/// Class index of an ion type (`false` = b, `true` = y) at a fragment charge (1 or 2), as a
/// neutral-loss ion or not.
#[inline]
pub(crate) fn class_of(is_y: bool, z: u8, loss: bool) -> usize {
    usize::from(loss) * 4 + usize::from(is_y) * 2 + usize::from(z >= 2)
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

/// One fragment position: ion type, charge, base mass, and the modification counts the
/// hypotheses give it.
type Position = (bool, u8, f64, Vec<bool>);

/// m/z key: 1e-4 Th, so two routes to the same mass are one ion.
#[inline]
fn key(mz: f64) -> i64 {
    (mz * 1e4).round() as i64
}

/// b1..b(n-1) and y1..y(n-1) at charges 1..=`zmax` for `s` under each of `hyps`, their
/// and the impossible variants.
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
    let mz_of = |base: f64, c: usize, lost: f64, z: u8| {
        (base + c as f64 * s.delta - lost + f64::from(z) * PROTON) / f64::from(z)
    };
    // ions keyed by m/z; BTreeMap so the ion order is ascending and deterministic
    let mut by_key: BTreeMap<i64, Ion> = BTreeMap::new();
    let mut add = |mz: f64, class: usize, hi: usize| {
        let e = by_key.entry(key(mz)).or_insert_with(|| Ion {
            mz,
            class,
            member: vec![false; h],
        });
        e.member[hi] = true;
    };
    // per position: the modification counts hypotheses give it
    let mut positions: Vec<Position> = Vec::new();
    for is_y in [false, true] {
        for o in 1..n {
            let base = if is_y {
                total - pre[n - o] + WATER
            } else {
                pre[o]
            };
            let in_frag = |i: usize| {
                if is_y {
                    s.in_suffix(i, o)
                } else {
                    s.in_prefix(i, o)
                }
            };
            for z in 1..=zmax {
                let mut taken = vec![false; k + 1];
                let counts: Vec<usize> = hyps
                    .iter()
                    .map(|sites| sites.iter().filter(|&&i| in_frag(i)).count())
                    .collect();
                for &c in &counts {
                    taken[c] = true;
                }
                for (hi, &c) in counts.iter().enumerate() {
                    add(mz_of(base, c, 0.0, z), class_of(is_y, z, false), hi);
                }
                positions.push((is_y, z, base, taken));
            }
        }
    }
    let mut impossible = Vec::new();
    for (is_y, z, base, taken) in &positions {
        for (c, &t) in taken.iter().enumerate() {
            if t {
                continue;
            }
            let mz = mz_of(*base, c, 0.0, *z);
            if !by_key.contains_key(&key(mz)) {
                impossible.push((mz, class_of(*is_y, *z, false)));
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

    /// The z = 1 b1..b(n-1) and y1..y(n-1) keys of a peptidoform, from the mass model.
    fn ladder_keys(pf: &str) -> Vec<i64> {
        let p = parse_peptidoform(pf).unwrap();
        let n = p.residues.len();
        let mut want: Vec<i64> = Vec::new();
        let mut pre = p.n_term_mod;
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
        want
    }

    fn member_keys(set: &IonSet, hi: usize) -> Vec<i64> {
        let mut got: Vec<i64> = set
            .ions
            .iter()
            .filter(|ion| ion.member[hi])
            .map(|ion| key(ion.mz))
            .collect();
        got.sort_unstable();
        got
    }

    /// Under each hypothesis the ions are exactly the b/y ladder of that peptidoform.
    #[test]
    fn ions_are_the_ladders_of_the_placements() {
        let s = split("PS[Phospho]ETK", "Phospho").unwrap();
        let hyps = vec![vec![1], vec![3]];
        let set = build(&s, &hyps, 1);
        for (hi, pf) in ["PS[Phospho]ETK", "PSET[Phospho]K"].iter().enumerate() {
            assert_eq!(member_keys(&set, hi), ladder_keys(pf), "{pf}");
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

    /// A placement on the N-terminal slot gives the ladder of the N-terminally modified
    /// peptidoform: every b ion carries it, no y ion does.
    #[test]
    fn the_n_terminal_slot_is_in_every_b_ion() {
        let s = split("[Acetyl]-AKPEPK", "Acetyl").unwrap();
        let hyps = vec![vec![1], vec![6]];
        let set = build(&s, &hyps, 1);
        for (hi, pf) in ["AK[Acetyl]PEPK", "[Acetyl]-AKPEPK"].iter().enumerate() {
            assert_eq!(member_keys(&set, hi), ladder_keys(pf), "{pf}");
        }
        // b1 and y5 tell the two apart, each with and without the acetyl
        let sdi: Vec<&Ion> = set
            .ions
            .iter()
            .filter(|i| !i.member.iter().all(|&m| m))
            .collect();
        assert_eq!(sdi.len(), 4);
        assert_eq!(sdi.iter().filter(|i| i.class == 0).count(), 2);
        assert_eq!(sdi.iter().filter(|i| i.class == 2).count(), 2);
    }

    #[test]
    fn classes_split_type_charge_and_loss() {
        assert_eq!(class_of(false, 1, false), 0);
        assert_eq!(class_of(false, 2, false), 1);
        assert_eq!(class_of(true, 1, false), 2);
        assert_eq!(class_of(true, 2, false), 3);
        assert_eq!(class_of(false, 1, true), 4);
        assert_eq!(class_of(true, 2, true), 7);
        let s = split("PS[Phospho]ETK", "Phospho").unwrap();
        let set = build(&s, &[vec![1], vec![3]], 2);
        assert!(set.ions.iter().any(|i| i.class == 1));
        assert!(set.ions.iter().any(|i| i.class == 3));
        assert!(set.ions.iter().all(|i| i.class < 4));
    }
}
