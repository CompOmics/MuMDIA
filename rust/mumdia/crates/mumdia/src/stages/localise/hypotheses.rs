//! Isomer groups and placement hypotheses (docs/36_ptm_localisation.md).
//!
//! A peptidoform is read as its residues, the per-residue mass deltas of every modification
//! that is NOT localised (held where the identification put them), the termini, and the
//! positions that carry the localised modification. A hypothesis is one placement of the
//! localised modification's count on the candidate residues; the hypotheses of a peptide are
//! every such placement, in lexicographic order of their sorted positions.

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
    pub n_term: f64,
    pub c_term: f64,
    /// 0-based positions carrying the localised modification, ascending.
    pub sites: Vec<usize>,
    /// The localised modification's mass delta.
    pub delta: f64,
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
/// peptidoform does not parse, does not carry the modification, or a residue carries it
/// together with another modification (stacked modifications are not localised).
pub(crate) fn split(pf: &str, name: &str) -> Option<Split> {
    let delta = unimod_mass(name)?;
    let p = parse_peptidoform(pf).ok()?;
    let brackets = residue_brackets(pf);
    if brackets.len() != p.residues.len() {
        return None;
    }
    let mut sites = Vec::new();
    let mut base_mods = p.mods.clone();
    for (i, b) in brackets.iter().enumerate() {
        let Some(text) = b else { continue };
        let is_mod = |t: &str| {
            t == name
                || t.trim_start_matches('+')
                    .parse::<f64>()
                    .is_ok_and(|v| (v - delta).abs() < 1e-3)
        };
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
    if sites.is_empty() {
        return None;
    }
    Some(Split {
        residues: p.residues,
        base_mods,
        n_term: p.n_term_mod,
        c_term: p.c_term_mod,
        sites,
        delta,
    })
}

/// The positions a placement may use: residues in `candidates` (the modification's
/// residues) or equal to `decoy`, that carry no other modification.
pub(crate) fn candidate_positions(s: &Split, candidates: &[u8], decoy: Option<u8>) -> Vec<usize> {
    (0..s.residues.len())
        .filter(|&i| {
            let r = s.residues[i];
            s.base_mods[i] == 0.0 && (candidates.contains(&r) || decoy == Some(r))
        })
        .collect()
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
    // N-terminal group, copied as written.
    if original.starts_with('[') {
        if let Some(close) = original.find(']') {
            if original.as_bytes().get(close + 1) == Some(&b'-') {
                out.push_str(&original[..close + 2]);
            }
        }
    }
    for (i, &r) in s.residues.iter().enumerate() {
        out.push(r as char);
        let own = brackets.get(i).cloned().flatten();
        let keeps_other = own.as_deref().is_some_and(|t| {
            !(t == name
                || t.trim_start_matches('+')
                    .parse::<f64>()
                    .is_ok_and(|v| (v - s.delta).abs() < 1e-3))
        });
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
        assert_eq!(candidate_positions(&s, b"STY", None), vec![0, 3, 4]);
        assert_eq!(
            candidate_positions(&s, b"STY", Some(b'A')),
            vec![0, 1, 3, 4]
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
