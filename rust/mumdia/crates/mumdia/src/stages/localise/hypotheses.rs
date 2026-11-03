//! Isomer groups and placement hypotheses (docs/36_ptm_localisation.md).
//!
//! A peptidoform is read as its residues, the per-residue mass deltas of every modification
//! that is NOT localised (held where the identification put them), the termini, and the
//! sites that carry the localised modification. A site is a slot: `0..n` are the residues
//! and slot `n` (one past the last residue, `Split::n_term_slot`) is the N-terminal amine,
//! a placement distinct from the first residue's side chain. A hypothesis is one placement of
//! the localised modification's count on the candidate slots; the hypotheses of a peptide are
//! every such placement, in lexicographic order of their sorted slots.

use mumdia_core::mass::{parse_peptidoform, unimod_mass};

/// Modification siblings: same residues, charge, label and modification composition (the
/// sorted deltas, terminal ones included), so they differ only in localisation. Shared by
/// the prescreen's localisation policy and this stage.
pub(crate) fn sibling_key(pf: &str, charge: i32, label: u32) -> Option<String> {
    let p = parse_peptidoform(pf).ok()?;
    let mut d: Vec<i64> = p
        .mods
        .iter()
        .chain([&p.n_term_mod, &p.c_term_mod])
        .filter(|&&x| x != 0.0)
        .map(|&x| (x * 1e4).round() as i64)
        .collect();
    d.sort_unstable();
    Some(format!(
        "{}/{charge}/{label}/{d:?}",
        String::from_utf8_lossy(&p.residues)
    ))
}

/// The modification composition of a peptidoform, without its residues: the sorted deltas
/// in 1e-4 Da, terminal ones included. Positional isomers share it, and so do a target and
/// its paired decoy, which is what the picked isomer-group competition keys on.
pub(crate) fn composition_key(pf: &str) -> Option<Vec<i64>> {
    let p = parse_peptidoform(pf).ok()?;
    let mut d: Vec<i64> = p
        .mods
        .iter()
        .chain([&p.n_term_mod, &p.c_term_mod])
        .filter(|&&x| x != 0.0)
        .map(|&x| (x * 1e4).round() as i64)
        .collect();
    d.sort_unstable();
    Some(d)
}

/// A peptidoform split for localisation of one modification.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Split {
    /// Residue letters, upper case.
    pub residues: Vec<u8>,
    /// Per-residue mass delta of every modification except the localised one.
    pub base_mods: Vec<f64>,
    /// N-terminal mass delta of every modification except the localised one.
    pub n_term: f64,
    pub c_term: f64,
    /// Slots carrying the localised modification, ascending: 0-based residue positions, and
    /// [`Split::n_term_slot`] for the N-terminal amine.
    pub sites: Vec<usize>,
    /// The localised modification's mass delta.
    pub delta: f64,
}

impl Split {
    /// The slot of the N-terminal amine: one past the last residue.
    pub fn n_term_slot(&self) -> usize {
        self.residues.len()
    }

    /// Residue slots and the N-terminal slot.
    pub fn n_slots(&self) -> usize {
        self.residues.len() + 1
    }

    /// The residue at slot `i`; `None` for the N-terminal slot.
    pub fn residue_at(&self, i: usize) -> Option<u8> {
        self.residues.get(i).copied()
    }

    /// Whether slot `i` is a placement on the residue `decoy`.
    pub fn is_decoy_slot(&self, i: usize, decoy: Option<u8>) -> bool {
        decoy.is_some() && self.residue_at(i) == decoy
    }

    /// Whether the fragment holding the first `o` residues (a b ion) carries slot `i`. The
    /// N-terminal slot is in every b ion.
    pub fn in_prefix(&self, i: usize, o: usize) -> bool {
        i < o || i == self.n_term_slot()
    }

    /// Whether the fragment holding the last `o` residues (a y ion) carries slot `i`. The
    /// N-terminal slot is in no y ion.
    pub fn in_suffix(&self, i: usize, o: usize) -> bool {
        let n = self.residues.len();
        i < n && i + o >= n
    }

    /// `S3` (residue and 1-based position) or `N-term`, as `site_probabilities` writes it.
    pub fn slot_label(&self, i: usize) -> String {
        match self.residue_at(i) {
            Some(r) => format!("{}{}", r as char, i + 1),
            None => N_TERM_LABEL.to_string(),
        }
    }
}

/// How `site_probabilities` and `sites.tsv` name the N-terminal slot.
pub(crate) const N_TERM_LABEL: &str = "N-term";

/// The slot a `site_probabilities` label names (`S3` or `N-term`) in a peptide of
/// `n_residues` residues: the inverse of [`Split::slot_label`].
pub(crate) fn parse_slot_label(label: &str, n_residues: usize) -> Option<usize> {
    if label == N_TERM_LABEL {
        return Some(n_residues);
    }
    let pos: usize = label.get(1..)?.parse().ok()?;
    (1..=n_residues).contains(&pos).then(|| pos - 1)
}

/// Whether the bracket text `t` names the modification `name` of mass `delta`: its name, or
/// a signed mass within 1e-3 Da of it.
fn names_mod(t: &str, name: &str, delta: f64) -> bool {
    t == name
        || t.trim_start_matches('+')
            .parse::<f64>()
            .is_ok_and(|v| (v - delta).abs() < 1e-3)
}

/// The bracket text of the N-terminal group `[X]-` of `pf`, if it has one.
fn n_term_text(pf: &str) -> Option<&str> {
    if !pf.starts_with('[') {
        return None;
    }
    let close = pf.find(']')?;
    (pf.as_bytes().get(close + 1) == Some(&b'-')).then(|| &pf[1..close])
}

/// The bracketed modification texts of each residue of a ProForma-lite peptidoform, in
/// residue order (`None` for an unmodified residue). Terminal groups are skipped.
fn residue_brackets(pf: &str) -> Vec<Option<String>> {
    let b = pf.as_bytes();
    let mut out: Vec<Option<String>> = Vec::new();
    let mut i = 0;
    // An N-terminal group `[X]-`.
    if b.first() == Some(&b'[') {
        if let Some(close) = pf.find(']') {
            if b.get(close + 1) == Some(&b'-') {
                i = close + 2;
            }
        }
    }
    while i < b.len() {
        let c = b[i];
        if c == b'-' {
            break; // the C-terminal group
        }
        if c == b'[' {
            let Some(off) = pf[i..].find(']') else { break };
            let text = &pf[i + 1..i + off];
            if let Some(last) = out.last_mut() {
                *last = Some(match last.take() {
                    Some(prev) => format!("{prev}+{text}"),
                    None => text.to_string(),
                });
            }
            i += off + 1;
            continue;
        }
        if c.is_ascii_alphabetic() {
            out.push(None);
        }
        i += 1;
    }
    out
}

/// Split `pf` for the localisation of `name` (a catalogue modification). `None` when the
/// peptidoform does not parse, does not carry the modification, or a residue or the
/// N-terminus carries it together with another modification (stacked modifications are not
/// localised). The modification in the N-terminal group (`[name]-`) is a site on the
/// N-terminal slot.
pub(crate) fn split(pf: &str, name: &str) -> Option<Split> {
    let delta = unimod_mass(name)?;
    let p = parse_peptidoform(pf).ok()?;
    let brackets = residue_brackets(pf);
    if brackets.len() != p.residues.len() {
        return None;
    }
    let is_mod = |t: &str| names_mod(t, name, delta);
    let mut sites = Vec::new();
    let mut base_mods = p.mods.clone();
    for (i, b) in brackets.iter().enumerate() {
        let Some(text) = b else { continue };
        if is_mod(text) {
            sites.push(i);
            base_mods[i] -= delta;
            if base_mods[i].abs() > 1e-6 {
                return None;
            }
            base_mods[i] = 0.0;
        } else if text.split('+').any(is_mod) {
            return None;
        }
    }
    let mut n_term = p.n_term_mod;
    if let Some(text) = n_term_text(pf) {
        if is_mod(text) {
            sites.push(p.residues.len());
            n_term -= delta;
            if n_term.abs() > 1e-6 {
                return None;
            }
            n_term = 0.0;
        } else if text.split('+').any(is_mod) {
            return None;
        }
    }
    if sites.is_empty() {
        return None;
    }
    Some(Split {
        residues: p.residues,
        base_mods,
        n_term,
        c_term: p.c_term_mod,
        sites,
        delta,
    })
}

/// The slots a placement may use: residues in `candidates` (the modification's residue
/// rules) or equal to `decoy`, that carry no other modification, and the N-terminal slot when
/// `n_term` allows the modification there and the N-terminus carries no other modification.
///
/// A modification on the first residue and one on the N-terminus give the same b and y
/// ladder, so with the N-terminal slot open the decoy residue in first position is dropped: a
/// decoy that no fragment can tell from a real placement would win half of their ties and
/// measure nothing. A real residue there stays, and the tie is reported as the ambiguity it is.
pub(crate) fn candidate_positions(
    s: &Split,
    candidates: &[u8],
    decoy: Option<u8>,
    n_term: bool,
) -> Vec<usize> {
    let n_term = n_term && s.n_term == 0.0;
    let mut out: Vec<usize> = (0..s.residues.len())
        .filter(|&i| {
            let r = s.residues[i];
            if s.base_mods[i] != 0.0 {
                return false;
            }
            candidates.contains(&r) || (decoy == Some(r) && !(n_term && i == 0))
        })
        .collect();
    if n_term {
        out.push(s.n_term_slot());
    }
    out
}

/// n choose k, saturating.
fn choose(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut c: u128 = 1;
    for i in 0..k {
        c = c.saturating_mul((n - i) as u128) / (i + 1) as u128;
    }
    c
}

/// Every placement of `k` modifications on `positions`, each ascending, in lexicographic
/// order. `None` when there are more than `cap`.
pub(crate) fn enumerate(positions: &[usize], k: usize, cap: usize) -> Option<Vec<Vec<usize>>> {
    if choose(positions.len(), k) > cap as u128 {
        return None;
    }
    let n = positions.len();
    let mut out = Vec::new();
    if k == 0 || k > n {
        return Some(out);
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.iter().map(|&i| positions[i]).collect());
        // Advance the rightmost index that can still move; none left means done.
        let mut j = k;
        loop {
            if j == 0 {
                return Some(out);
            }
            j -= 1;
            if idx[j] < n - k + j {
                break;
            }
        }
        idx[j] += 1;
        for t in j + 1..k {
            idx[t] = idx[t - 1] + 1;
        }
    }
}

/// The peptidoform text of a placement: `s` with the localised modification `name` on
/// `sites` and every other modification where it was, written in the engine's
/// ProForma-lite spelling (names where the catalogue knows the delta, signed masses
/// otherwise).
pub(crate) fn render(original: &str, s: &Split, name: &str, sites: &[usize]) -> String {
    let brackets = residue_brackets(original);
    let mut out = String::new();
    // N-terminal group: another modification's is copied as written, the localised one is
    // written where the placement puts it.
    match n_term_text(original) {
        Some(t) if !names_mod(t, name, s.delta) => {
            out.push('[');
            out.push_str(t);
            out.push_str("]-");
        }
        _ if sites.contains(&s.n_term_slot()) => {
            out.push('[');
            out.push_str(name);
            out.push_str("]-");
        }
        _ => {}
    }
    for (i, &r) in s.residues.iter().enumerate() {
        out.push(r as char);
        let own = brackets.get(i).cloned().flatten();
        let keeps_other = own.as_deref().is_some_and(|t| !names_mod(t, name, s.delta));
        if keeps_other {
            out.push('[');
            out.push_str(own.as_deref().unwrap_or(""));
            out.push(']');
        } else if sites.contains(&i) {
            out.push('[');
            out.push_str(name);
            out.push(']');
        }
    }
    // C-terminal group, copied as written.
    if let Some(pos) = original.rfind("-[") {
        out.push_str(&original[pos..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_finds_the_localised_sites_and_keeps_the_rest() {
        let s = split(
            "AS[Phospho]M[Oxidation]C[Carbamidomethyl]T[Phospho]K",
            "Phospho",
        )
        .unwrap();
        assert_eq!(s.residues, b"ASMCTK");
        assert_eq!(s.sites, vec![1, 4]);
        assert_eq!(s.base_mods[1], 0.0);
        assert!((s.base_mods[2] - 15.994_915).abs() < 1e-3);
        assert!((s.base_mods[3] - 57.021_464).abs() < 1e-3);
        assert!(split("PEPTIDEK", "Phospho").is_none());
        // a signed mass is recognised as the modification
        let m = split("PEPS[+79.966331]K", "Phospho").unwrap();
        assert_eq!(m.sites, vec![3]);
    }

    #[test]
    fn enumeration_is_lexicographic_and_capped() {
        let h = enumerate(&[1, 3, 5, 7], 2, 100).unwrap();
        assert_eq!(
            h,
            vec![
                vec![1, 3],
                vec![1, 5],
                vec![1, 7],
                vec![3, 5],
                vec![3, 7],
                vec![5, 7]
            ]
        );
        assert_eq!(enumerate(&[1, 3, 5, 7], 1, 100).unwrap().len(), 4);
        assert!(enumerate(&[1, 3, 5, 7], 2, 5).is_none());
        assert_eq!(enumerate(&[1, 3, 5, 7], 2, 6).unwrap().len(), 6);
        // 30 choose 3 = 4060
        let p: Vec<usize> = (0..30).collect();
        assert!(enumerate(&p, 3, 4059).is_none());
        assert_eq!(enumerate(&p, 3, 4060).unwrap().len(), 4060);
    }

    #[test]
    fn candidate_positions_skip_other_modifications_and_add_the_decoy_residue() {
        let s = split("S[Phospho]AM[Oxidation]TYK", "Phospho").unwrap();
        assert_eq!(candidate_positions(&s, b"STY", None, false), vec![0, 3, 4]);
        assert_eq!(
            candidate_positions(&s, b"STY", Some(b'A'), false),
            vec![0, 1, 3, 4]
        );
        // the N-terminal slot only where allowed, and it is never the decoy residue
        assert_eq!(
            candidate_positions(&s, b"STY", Some(b'A'), true),
            vec![0, 1, 3, 4, 6]
        );
        assert!(!s.is_decoy_slot(6, Some(b'A')));
        assert!(!s.is_decoy_slot(1, None));
        assert!(s.is_decoy_slot(1, Some(b'A')));
    }

    /// A terminal modification is a site on the N-terminal slot, distinct from the first
    /// residue; a residue carrying the modification keeps its own slot.
    #[test]
    fn the_n_terminus_is_its_own_slot() {
        let t = split("[Acetyl]-AKPEPK", "Acetyl").unwrap();
        assert_eq!(t.residues, b"AKPEPK");
        assert_eq!(t.sites, vec![6]);
        assert_eq!(t.n_term_slot(), 6);
        assert_eq!(t.n_term, 0.0);
        assert_eq!(t.slot_label(6), "N-term");
        assert_eq!(t.slot_label(1), "K2");
        assert_eq!(parse_slot_label("N-term", 6), Some(6));
        assert_eq!(parse_slot_label("K2", 6), Some(1));
        assert_eq!(parse_slot_label("K9", 6), None);
        let k = split("AK[Acetyl]PEPK", "Acetyl").unwrap();
        assert_eq!(k.sites, vec![1]);
        // both forms give the same candidate slots when the N-terminus is allowed
        assert_eq!(candidate_positions(&t, b"K", None, true), vec![1, 5, 6]);
        assert_eq!(candidate_positions(&k, b"K", None, true), vec![1, 5, 6]);
        assert_eq!(candidate_positions(&k, b"K", None, false), vec![1, 5]);
        // and render into each other
        assert_eq!(
            render("[Acetyl]-AKPEPK", &t, "Acetyl", &[1]),
            "AK[Acetyl]PEPK"
        );
        assert_eq!(
            render("AK[Acetyl]PEPK", &k, "Acetyl", &[6]),
            "[Acetyl]-AKPEPK"
        );
        // a b ion carries the N-terminal slot, a y ion never does
        assert!(t.in_prefix(6, 1) && !t.in_suffix(6, 5));
        assert!(t.in_suffix(5, 1) && !t.in_prefix(5, 5));
        // another N-terminal modification is held, and closes the N-terminal slot
        let o = split("[Acetyl]-AS[Phospho]PEPK", "Phospho").unwrap();
        assert_eq!(o.sites, vec![1]);
        assert!((o.n_term - 42.010_565).abs() < 1e-3);
        assert_eq!(candidate_positions(&o, b"ST", None, true), vec![1]);
        assert_eq!(
            render("[Acetyl]-AS[Phospho]PEPK", &o, "Phospho", &[1]),
            "[Acetyl]-AS[Phospho]PEPK"
        );
        // the decoy residue in first position cannot be told from the N-terminus: dropped
        // when the N-terminal slot is open, kept otherwise
        assert_eq!(
            candidate_positions(&t, b"K", Some(b'A'), true),
            vec![1, 5, 6]
        );
        assert_eq!(
            candidate_positions(&t, b"K", Some(b'A'), false),
            vec![0, 1, 5]
        );
        // a real residue there stays: the ambiguity is real
        assert_eq!(
            candidate_positions(&t, b"AK", Some(b'G'), true),
            vec![0, 1, 5, 6]
        );
        // the modification on both the N-terminus and a residue
        let b = split("[Acetyl]-AK[Acetyl]PEPK", "Acetyl").unwrap();
        assert_eq!(b.sites, vec![1, 6]);
        assert_eq!(
            render("[Acetyl]-AK[Acetyl]PEPK", &b, "Acetyl", &[5, 6]),
            "[Acetyl]-AKPEPK[Acetyl]"
        );
    }

    #[test]
    fn render_moves_only_the_localised_modification() {
        let pf = "[Acetyl]-S[Phospho]M[Oxidation]TK";
        let s = split(pf, "Phospho").unwrap();
        assert_eq!(
            render(pf, &s, "Phospho", &[2]),
            "[Acetyl]-SM[Oxidation]T[Phospho]K"
        );
        assert_eq!(render(pf, &s, "Phospho", &[0]), pf);
    }

    #[test]
    fn sibling_and_composition_keys_ignore_the_position() {
        assert_eq!(
            sibling_key("S[Phospho]TK", 2, 0),
            sibling_key("ST[Phospho]K", 2, 0)
        );
        assert_ne!(
            sibling_key("S[Phospho]TK", 2, 0),
            sibling_key("S[Phospho]TK", 3, 0)
        );
        assert_eq!(
            composition_key("S[Phospho]TK"),
            composition_key("KT[Phospho]S")
        );
    }
}
