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
//!
//! With a neutral loss ([`Loss`], phosphoric acid from a phosphorylated serine or threonine)
//! a fragment carrying at least one site that loses it also gives a loss ion, the fragment's
//! m/z less the loss. The loss of phosphoric acid from a fragment with `c` sites lands exactly
//! on the water loss of the same fragment with `c - 1`, so where some placement gives the
//! fragment one site fewer, the loss ion of the others would be that placement's water loss
//! counted against it; those loss ions are left out. What remains are the positions where
//! every placement puts the same number of sites in the fragment, which tell the residues
//! that lose phosphoric acid from those that do not. Loss ions are four classes of their own,
//! with their own detection rates (from the common loss ions); they have no impossible
//! variants, and take the false-presence rate of the plain class of the same ion type and
//! charge. One m/z is one observation, so a loss ion that falls on another ion's m/z joins
//! that ion.

use mumdia_core::constants::residue_mass;
use mumdia_core::constants::{PROTON, WATER};
use std::collections::BTreeMap;

use super::hypotheses::Split;

/// The ion classes the detection and false-presence rates are kept for: `(b, 1+)`,
/// `(b, 2+)`, `(y, 1+)`, `(y, 2+)`, then the same four for neutral-loss ions.
pub(crate) const N_CLASSES: usize = 8;

/// Class index of an ion type (`false` = b, `true` = y) at a fragment charge (1 or 2), as a
/// neutral-loss ion or not.
#[inline]
pub(crate) fn class_of(is_y: bool, z: u8, loss: bool) -> usize {
    usize::from(loss) * 4 + usize::from(is_y) * 2 + usize::from(z >= 2)
}

/// A neutral loss from fragments that carry the localised modification on some slots.
#[derive(Clone, Debug)]
pub(crate) struct Loss {
    /// The mass lost.
    pub mass: f64,
    /// Per slot of the peptide (residues, then the N-terminus): whether a site there loses
    /// it.
    pub lossy: Vec<bool>,
}

/// The loss of phosphoric acid (the phosphate group plus water, 97.9769 Da) from a
/// phosphorylated serine or threonine, and from the decoy residue so that its placements stay
/// exchangeable with the real ones; a phosphotyrosine and the N-terminus do not lose it.
pub(crate) fn phosphate_loss(s: &Split, decoy: Option<u8>) -> Loss {
    let lossy = (0..s.n_slots())
        .map(|i| match s.residue_at(i) {
            Some(b'S' | b'T') => true,
            Some(r) => decoy == Some(r),
            None => false,
        })
        .collect();
    Loss {
        mass: s.delta + WATER,
        lossy,
    }
}

/// Loss ions have no impossible variants (see [`build`]), so a loss class takes the
/// false-presence rate of the plain class of the same ion type and charge.
pub(crate) fn loss_false_presence_from_plain(f: &mut [f64; N_CLASSES]) {
    for c in 0..4 {
        f[c + 4] = f[c];
    }
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
/// neutral-loss ions when `loss` is given, and the impossible variants.
pub(crate) fn build(s: &Split, hyps: &[Vec<usize>], zmax: u8, loss: Option<&Loss>) -> IonSet {
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
                for (hi, sites) in hyps.iter().enumerate() {
                    let c = counts[hi];
                    add(mz_of(base, c, 0.0, z), class_of(is_y, z, false), hi);
                    if let Some(l) = loss {
                        // the loss of a fragment with c sites falls on the water loss of the
                        // same fragment with c - 1, which another placement may predict
                        let aliased = c >= 1 && taken[c - 1];
                        if !aliased && sites.iter().any(|&i| in_frag(i) && l.lossy[i]) {
                            add(mz_of(base, c, l.mass, z), class_of(is_y, z, true), hi);
                        }
                    }
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
        let set = build(&s, &hyps, 1, None);
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
        let set = build(&s, &hyps, 1, None);
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
        let set = build(&s, &hyps, 1, None);
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
        let set = build(&s, &[vec![1], vec![3]], 2, None);
        assert!(set.ions.iter().any(|i| i.class == 1));
        assert!(set.ions.iter().any(|i| i.class == 3));
        assert!(set.ions.iter().all(|i| i.class < 4));
    }

    /// Phosphate loss: a fragment carrying a phosphoserine or -threonine gives a loss ion
    /// 97.9769 Da lighter, one carrying only a phosphotyrosine does not. In PSEYK with sites
    /// S2 or Y4, b2 and b3 carry the site under S2 only: their loss falls on the water loss of
    /// the same fragment without it, which Y4 predicts, so they are left out; b4 and y4 carry
    /// the site under both placements, and only S2 loses it there.
    #[test]
    fn phosphate_loss_ions_come_from_s_and_t_only() {
        let s = split("PS[Phospho]EYK", "Phospho").unwrap();
        let loss = phosphate_loss(&s, Some(b'A'));
        assert!((loss.mass - 97.976_896).abs() < 1e-5);
        assert_eq!(loss.lossy, vec![false, true, false, false, false, false]);
        let hyps = vec![vec![1], vec![3]];
        let plain = build(&s, &hyps, 1, None);
        let with = build(&s, &hyps, 1, Some(&loss));
        let lossy: Vec<&Ion> = with.ions.iter().filter(|i| i.class >= 4).collect();
        assert_eq!(lossy.len(), 2);
        assert!(lossy.iter().all(|i| i.member == vec![true, false]));
        // the plain ions and their impossible variants are unchanged
        assert_eq!(
            with.ions.iter().filter(|i| i.class < 4).count(),
            plain.ions.len()
        );
        assert_eq!(with.impossible.len(), plain.impossible.len());
        // b4 - H3PO4 of S2 is the b4 m/z of the unmodified peptide less water
        let b4: f64 = b"PSEY".iter().map(|&r| residue_mass(r).unwrap()).sum();
        assert!(lossy.iter().any(|i| key(i.mz) == key(b4 - WATER + PROTON)));
        // the decoy residue loses as S and T do, so its placements stay exchangeable
        let d = split("PAS[Phospho]K", "Phospho").unwrap();
        assert_eq!(
            phosphate_loss(&d, Some(b'A')).lossy,
            vec![false, true, true, false, false]
        );
        let mut f = [0.1, 0.2, 0.3, 0.4, 0.9, 0.9, 0.9, 0.9];
        loss_false_presence_from_plain(&mut f);
        assert_eq!(f, [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]);
    }
}
