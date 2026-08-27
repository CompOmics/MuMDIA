//! ProForma-lite peptidoform parsing and a UniMod-backed mass model
//! (docs/02_config_and_data_model.md).
//!
//! All fragment m/z derive from one shared mass model
//! (docs/14_build_test_deploy_gotchas.md). Modifications are UniMod names or
//! explicit signed mass deltas; an unknown name is a typed error, never a silent
//! drop.

use crate::constants::{mass_to_mz, residue_mass, PROTON, WATER};
use crate::error::MassError;

/// A modification MuMDIA understands by name. MVP subset; names must be
/// UniMod/PSI-MS so the sidecar adapters map them (docs/05_digest_peptidoforms.md).
pub fn unimod_mass(name: &str) -> Option<f64> {
    let m = match name {
        "Carbamidomethyl" => 57.021_463_735,
        "Oxidation" => 15.994_914_620,
        "Acetyl" => 42.010_564_684,
        "Phospho" => 79.966_331_090,
        "Deamidated" => 0.984_016_106,
        "Methyl" => 14.015_650_064,
        "Dimethyl" => 28.031_300_128,
        "Carbamyl" => 43.005_813_726,
        // Cysteine prenylation (UniMod 44/48/376). Deltas are the monoisotopic
        // composition masses: Farnesyl C15H24, GeranylGeranyl C20H32,
        // Hydroxyfarnesyl C15H24O. Enables a FASTA/imported prenylation search.
        "Farnesyl" => 204.187_801_1,
        "GeranylGeranyl" => 272.250_401_2,
        "Hydroxyfarnesyl" => 220.182_715_7,
        _ => return None,
    };
    Some(m)
}

/// Ion series MuMDIA scores in the MVP (b and y, docs/18_findings_and_decisions.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IonType {
    B,
    Y,
}

impl IonType {
    pub fn symbol(&self) -> char {
        match self {
            IonType::B => 'b',
            IonType::Y => 'y',
        }
    }
}

/// One theoretical fragment ion.
#[derive(Clone, Debug)]
pub struct Fragment {
    pub ion_type: IonType,
    pub ordinal: usize,
    pub charge: i32,
    pub mz: f64,
}

impl Fragment {
    /// Stable name such as `b3` (charge 1) or `y5^2` (charge 2).
    ///
    /// Derived on demand rather than stored: it is a pure function of the three fields
    /// above, and fragment generation runs over every theoretical ion of every
    /// peptidoform in the library, of which the top-N truncation then discards the large
    /// majority. Storing it allocated one `String` per generated ion -- hundreds of
    /// millions of them on a real library -- almost all only to be dropped. Callers that
    /// need the text (the library writer) materialise it for the rows they keep.
    pub fn name(&self) -> String {
        frag_name(self.ion_type, self.ordinal, self.charge)
    }
}

/// A parsed peptidoform: residues plus per-residue and terminal mass deltas.
#[derive(Clone, Debug)]
pub struct ParsedPeptidoform {
    pub residues: Vec<u8>,
    pub mods: Vec<f64>,
    pub n_term_mod: f64,
    pub c_term_mod: f64,
}

impl ParsedPeptidoform {
    /// Neutral monoisotopic mass of the whole peptidoform.
    pub fn neutral_mass(&self) -> f64 {
        let mut m = WATER + self.n_term_mod + self.c_term_mod;
        for (r, md) in self.residues.iter().zip(&self.mods) {
            // residue mass validated at parse time
            m += residue_mass(*r).expect("validated residue") + md;
        }
        m
    }

    /// Precursor m/z at the given charge.
    pub fn precursor_mz(&self, charge: i32) -> f64 {
        mass_to_mz(self.neutral_mass(), charge)
    }

    /// All b and y fragments at the requested fragment charges, dropping the
    /// low-information b1/y1/y2 (docs/06_predict_frag_index_matchers.md).
    pub fn fragments(&self, frag_charges: &[i32]) -> Vec<Fragment> {
        let n = self.residues.len();
        let mut out = Vec::new();
        if n < 2 {
            return out;
        }
        // Prefix residue sums for b ions.
        let mut prefix = self.n_term_mod;
        for i in 0..n - 1 {
            prefix += residue_mass(self.residues[i]).expect("validated") + self.mods[i];
            let ordinal = i + 1; // b1..b(n-1)
            if ordinal <= 1 {
                continue; // drop b1
            }
            for &z in frag_charges {
                let mz = (prefix + z as f64 * PROTON) / z as f64;
                out.push(Fragment {
                    ion_type: IonType::B,
                    ordinal,
                    charge: z,
                    mz,
                });
            }
        }
        // Suffix residue sums for y ions.
        let mut suffix = self.c_term_mod + WATER;
        for i in (1..n).rev() {
            suffix += residue_mass(self.residues[i]).expect("validated") + self.mods[i];
            let ordinal = n - i; // y1..y(n-1)
            if ordinal <= 2 {
                continue; // drop y1, y2
            }
            for &z in frag_charges {
                let mz = (suffix + z as f64 * PROTON) / z as f64;
                out.push(Fragment {
                    ion_type: IonType::Y,
                    ordinal,
                    charge: z,
                    mz,
                });
            }
        }
        out
    }

    /// Number of proton-carrying basic residues (Arg, His, Lys) in the whole
    /// peptide. Used for the composition-based charge cap: the maximum sensible
    /// precursor charge is `1 (N-terminus) + basic_residue_count()`.
    pub fn basic_residue_count(&self) -> usize {
        self.residues
            .iter()
            .filter(|&&r| matches!(r, b'R' | b'H' | b'K'))
            .count()
    }

    /// Number of basic residues (Arg, His, Lys) contained in the sub-sequence of
    /// a b/y fragment of the given ordinal. A b-ion of ordinal `k` spans the
    /// first `k` residues; a y-ion of ordinal `k` spans the last `k`. The
    /// maximum sensible charge of that fragment is `1 (its N-terminal amine) +
    /// this count`.
    pub fn fragment_basic_sites(&self, ion: IonType, ordinal: usize) -> usize {
        let n = self.residues.len();
        let k = ordinal.min(n);
        let slice = match ion {
            IonType::B => &self.residues[0..k],
            IonType::Y => &self.residues[n - k..n],
        };
        slice
            .iter()
            .filter(|&&r| matches!(r, b'R' | b'H' | b'K'))
            .count()
    }
}

fn frag_name(t: IonType, ordinal: usize, charge: i32) -> String {
    if charge == 1 {
        format!("{}{}", t.symbol(), ordinal)
    } else {
        format!("{}{}^{}", t.symbol(), ordinal, charge)
    }
}

/// Parse a ProForma-lite peptidoform string.
///
/// Grammar (subset): optional `[Mod]-` N-terminal group, residues each
/// optionally followed by `[Mod]`, optional trailing `-[Mod]` C-terminal group.
/// A `[Mod]` is a UniMod name or a signed float such as `[+15.9949]`.
pub fn parse_peptidoform(s: &str) -> Result<ParsedPeptidoform, MassError> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let n = bytes.len();
    let mut residues = Vec::new();
    let mut mods: Vec<f64> = Vec::new();
    let mut n_term_mod = 0.0;
    let mut c_term_mod = 0.0;

    // Optional N-terminal group: `[Mod]-`
    if i < n && bytes[i] == b'[' {
        let (m, next) = parse_bracket(s, i)?;
        if next < n && bytes[next] == b'-' {
            n_term_mod = m;
            i = next + 1;
        }
        // else: it was a residue-attached mod before any residue -> error below
    }

    while i < n {
        let c = bytes[i];
        if c == b'-' {
            // C-terminal group: `-[Mod]` at the very end
            if i + 1 < n && bytes[i + 1] == b'[' {
                let (m, next) = parse_bracket(s, i + 1)?;
                c_term_mod = m;
                i = next;
                continue;
            }
            return Err(MassError::Parse(format!(
                "stray '-' at position {i} in '{s}'"
            )));
        }
        if !c.is_ascii_alphabetic() {
            return Err(MassError::Parse(format!(
                "unexpected char '{}' at position {i} in '{s}'",
                c as char
            )));
        }
        if residue_mass(c).is_none() {
            return Err(MassError::AmbiguousResidue(c as char));
        }
        residues.push(c.to_ascii_uppercase());
        mods.push(0.0);
        i += 1;
        // Optional residue-attached mod
        if i < n && bytes[i] == b'[' {
            let (m, next) = parse_bracket(s, i)?;
            *mods.last_mut().unwrap() += m;
            i = next;
        }
    }

    if residues.is_empty() {
        return Err(MassError::Parse(format!("no residues in '{s}'")));
    }
    Ok(ParsedPeptidoform {
        residues,
        mods,
        n_term_mod,
        c_term_mod,
    })
}

/// Parse a `[...]` group starting at `open` (index of `[`). Returns the mass
/// delta and the index just past the closing `]`.
fn parse_bracket(s: &str, open: usize) -> Result<(f64, usize), MassError> {
    let bytes = s.as_bytes();
    let close = s[open..]
        .find(']')
        .map(|off| open + off)
        .ok_or_else(|| MassError::Parse(format!("unclosed '[' in '{s}'")))?;
    let inner = &s[open + 1..close];
    let mass = if let Some(m) = unimod_mass(inner) {
        m
    } else if let Ok(v) = inner.trim_start_matches('+').parse::<f64>() {
        v
    } else {
        return Err(MassError::UnknownModification(inner.to_string()));
    };
    let _ = bytes; // silence unused in some cfgs
    Ok((mass, close + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peptide_mass_plain() {
        // PEPTIDE neutral monoisotopic mass = 799.35997 Da
        let p = parse_peptidoform("PEPTIDE").unwrap();
        assert!(
            (p.neutral_mass() - 799.359_965).abs() < 1e-3,
            "{}",
            p.neutral_mass()
        );
    }

    #[test]
    fn carbamidomethyl_by_name_and_mass_agree() {
        let a = parse_peptidoform("PEC[Carbamidomethyl]TIDE").unwrap();
        let b = parse_peptidoform("PEC[+57.021464]TIDE").unwrap();
        assert!((a.neutral_mass() - b.neutral_mass()).abs() < 1e-4);
    }

    #[test]
    fn nterm_and_cterm_mods() {
        let p = parse_peptidoform("[Acetyl]-PEPK").unwrap();
        let plain = parse_peptidoform("PEPK").unwrap();
        assert!((p.neutral_mass() - plain.neutral_mass() - 42.010_565).abs() < 1e-4);
    }

    #[test]
    fn fragments_drop_low_info() {
        let p = parse_peptidoform("PEPTIDE").unwrap();
        let frags = p.fragments(&[1]);
        // no b1, y1, y2
        assert!(frags.iter().all(|f| f.name() != "b1"));
        assert!(frags.iter().all(|f| f.name() != "y1"));
        assert!(frags.iter().all(|f| f.name() != "y2"));
        // y3 should exist for a 7-mer
        assert!(frags.iter().any(|f| f.name() == "y3"));
    }

    #[test]
    fn basic_residue_count_counts_rhk_only() {
        assert_eq!(
            parse_peptidoform("PEPTIDE").unwrap().basic_residue_count(),
            0
        );
        assert_eq!(
            parse_peptidoform("PEPTIDER").unwrap().basic_residue_count(),
            1
        );
        // R, H, K each count once; D/E (acidic) do not.
        assert_eq!(parse_peptidoform("HKRDE").unwrap().basic_residue_count(), 3);
    }

    #[test]
    fn fragment_basic_sites_by_slice() {
        // AAAKAAR: K at index 3, R at index 6 (n = 7).
        let p = parse_peptidoform("AAAKAAR").unwrap();
        // b-ions span the first `ordinal` residues.
        assert_eq!(p.fragment_basic_sites(IonType::B, 3), 0); // AAA
        assert_eq!(p.fragment_basic_sites(IonType::B, 4), 1); // AAAK
        assert_eq!(p.fragment_basic_sites(IonType::B, 6), 1); // AAAKAA
                                                              // y-ions span the last `ordinal` residues.
        assert_eq!(p.fragment_basic_sites(IonType::Y, 1), 1); // R
        assert_eq!(p.fragment_basic_sites(IonType::Y, 3), 1); // AAR
        assert_eq!(p.fragment_basic_sites(IonType::Y, 4), 2); // KAAR -> K + R
        assert_eq!(p.fragment_basic_sites(IonType::Y, 7), 2); // whole peptide: K + R
    }

    #[test]
    fn unknown_mod_errors() {
        assert!(matches!(
            parse_peptidoform("PEP[Nonsense]TIDE"),
            Err(MassError::UnknownModification(_))
        ));
    }

    #[test]
    fn ambiguous_residue_errors() {
        assert!(matches!(
            parse_peptidoform("PEPBIDE"),
            Err(MassError::AmbiguousResidue('B'))
        ));
    }
}
