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

    // ─────────────────────────────────────────────────────────────────────────────
    // Reference values, pinned against independently published constants.
    //
    // Before this, the only external anchor in the whole repository was one test
    // pinning PEPTIDE's NEUTRAL monoisotopic mass at 1e-3 Da tolerance. Neutral, so
    // `PROTON` never entered it; and 1e-3 Da cannot distinguish the proton mass
    // (1.007276466812) from the hydrogen-ATOM mass (1.007825035), a 0.55 mDa difference
    // that is the classic error in this field and one the constants file explicitly calls
    // out. No test referenced `residue_mass`, none pinned an individual residue, none
    // asserted a fragment m/z at any charge, and `WATER`, `AMMONIA` and
    // `ISOTOPE_SPACING` were unpinned. The two other mass tests are self-consistency
    // checks that pass unchanged if every residue mass is wrong by the same amount.
    //
    // This class of defect is INVISIBLE to the end-to-end smoke test by construction:
    // `ci/make_fixture_mzml.py` plants its peaks by reading the engine's own library, so
    // the fixture agrees with the mass model however wrong the model is. It is not
    // hypothetical either -- `ISOTOPE_SPACING` once shipped 485 ppm wrong and was caught
    // by a human reading code.
    //
    // Sources: CODATA 2018 for the proton; AME2020 / standard monoisotopic residue masses
    // as tabulated by the Unimod and PSI-MS references. Tolerances are absolute Da and
    // deliberately tight enough to catch a substituted constant, not merely a typo.

    use crate::constants::{AMMONIA, ISOTOPE_SPACING};

    /// Tight enough to separate the proton from the hydrogen atom (0.55 mDa apart).
    const MDA: f64 = 1e-5;

    #[test]
    fn particle_and_molecule_constants_match_published_values() {
        assert!(
            (PROTON - 1.007_276_466_812).abs() < 1e-11,
            "PROTON = {PROTON}, expected the proton mass 1.007276466812 (CODATA), NOT the \
             hydrogen-atom mass 1.007825035"
        );
        // The hydrogen atom is one electron mass heavier. Asserted explicitly so
        // substituting one for the other fails loudly. The tolerance is 1e-7, not 1e-9,
        // because the conventional H-atom figure 1.007825035 is itself rounded and sits
        // 1.2e-8 from PROTON + m_e; 1e-7 is still 4500x tighter than the 0.55 mDa gap
        // this is guarding.
        assert!(
            (1.007_825_035 - PROTON - 0.000_548_579_909).abs() < 1e-7,
            "the proton/H-atom difference is one electron mass; check PROTON"
        );
        assert!((WATER - 18.010_564_684).abs() < MDA, "WATER = {WATER}");
        assert!((AMMONIA - 17.026_549_1).abs() < MDA, "AMMONIA = {AMMONIA}");
        // 13C - 12C = 13.003354835 - 12. Once shipped 485 ppm wrong.
        assert!(
            (ISOTOPE_SPACING - 1.003_354_835).abs() < 1e-9,
            "ISOTOPE_SPACING = {ISOTOPE_SPACING}, expected the 13C-12C difference"
        );
    }

    #[test]
    fn every_standard_residue_mass_matches_published_values() {
        // All twenty, so a single substituted or transposed value cannot hide behind a
        // self-consistency check.
        let expected: [(u8, f64); 20] = [
            (b'G', 57.021_463_735),
            (b'A', 71.037_113_805),
            (b'S', 87.032_028_435),
            (b'P', 97.052_763_875),
            (b'V', 99.068_413_945),
            (b'T', 101.047_678_505),
            (b'C', 103.009_184_505),
            (b'L', 113.084_064_015),
            (b'I', 113.084_064_015),
            (b'N', 114.042_927_470),
            (b'D', 115.026_943_065),
            (b'Q', 128.058_577_540),
            (b'K', 128.094_963_050),
            (b'E', 129.042_593_135),
            (b'M', 131.040_484_645),
            (b'H', 137.058_911_875),
            (b'F', 147.068_413_945),
            (b'R', 156.101_111_050),
            (b'Y', 163.063_328_575),
            (b'W', 186.079_312_980),
        ];
        for (aa, want) in expected {
            let got = residue_mass(aa).unwrap_or_else(|| panic!("{} has no mass", aa as char));
            assert!(
                (got - want).abs() < MDA,
                "residue {} = {got}, expected {want}",
                aa as char
            );
        }
        // The pairs that are genuinely equal, and the pair that is nearly equal: K and Q
        // differ by 36.4 mDa, which a coarse tolerance would let collapse.
        assert_eq!(residue_mass(b'L'), residue_mass(b'I'));
        let (k, q) = (residue_mass(b'K').unwrap(), residue_mass(b'Q').unwrap());
        assert!(
            (k - q - 0.036_385_51).abs() < MDA,
            "K - Q should be 36.386 mDa, got {}",
            k - q
        );
        // Ambiguous residues stay unmassed rather than silently taking a neighbour's mass.
        for aa in [b'B', b'J', b'O', b'U', b'X', b'Z'] {
            assert!(
                residue_mass(aa).is_none(),
                "{} must be ambiguous",
                aa as char
            );
        }
        assert_eq!(residue_mass(b'g'), residue_mass(b'G'), "case-insensitive");
    }

    #[test]
    fn precursor_mz_is_charge_correct_to_sub_ppm() {
        // PEPTIDE: neutral 799.359964, so [M+H]+ = 800.367241 and [M+2H]2+ = 400.687259.
        // This is where PROTON enters, and where the previous neutral-only anchor did not
        // reach: with the hydrogen-atom mass instead, the 2+ value moves 0.27 mDa, which
        // is 0.7 ppm at this m/z -- inside a 20 ppm window, so it would not break the
        // search, it would bias every mass error.
        let p = parse_peptidoform("PEPTIDE").unwrap();
        let neutral = p.neutral_mass();
        assert!(
            (neutral - 799.359_964_289).abs() < 1e-6,
            "neutral = {neutral}"
        );

        let m1 = p.precursor_mz(1);
        let m2 = p.precursor_mz(2);
        let m3 = p.precursor_mz(3);
        assert!((m1 - (neutral + PROTON)).abs() < 1e-12);
        assert!((m2 - (neutral + 2.0 * PROTON) / 2.0).abs() < 1e-12);
        // Absolute reference values, so the relation above cannot be satisfied by a
        // wrong neutral mass.
        assert!((m1 - 800.367_240_756).abs() < 1e-6, "1+ = {m1}");
        assert!((m2 - 400.687_258_611).abs() < 1e-6, "2+ = {m2}");
        assert!((m3 - 267.460_597_896).abs() < 1e-6, "3+ = {m3}");
    }

    #[test]
    fn by_fragment_mz_matches_published_values_at_charge_one_and_two() {
        // PEPTIDE singly-charged b and y series. b1/y1/y2 are dropped by `fragments` as
        // low-information, so the series starts at b2 and y3.
        //
        // The values follow from the residue masses, WATER and PROTON that the two tests
        // above pin against published tables, so this test is about the SERIES ALGEBRA:
        // which residues enter each ordinal, where the terminal water goes, and how
        // charge enters. Taken together the three tests anchor both halves -- a wrong
        // residue mass fails the table test, a wrong series definition fails this one.
        // They agree with standard published PEPTIDE b/y tables to well inside 1e-6.
        let p = parse_peptidoform("PEPTIDE").unwrap();
        let frags = p.fragments(&[1, 2]);

        let get = |ion: IonType, ordinal: usize, charge: i32| -> f64 {
            frags
                .iter()
                .find(|f| f.ion_type == ion && f.ordinal == ordinal && f.charge == charge)
                .unwrap_or_else(|| panic!("missing {ion:?}{ordinal} at charge {charge}"))
                .mz
        };

        // b ions, charge 1: b_n = sum(residues 1..n) + PROTON.
        for (ordinal, want) in [
            (2usize, 227.102_633_477),
            (3, 324.155_397_352),
            (4, 425.203_075_857),
            (5, 538.287_139_872),
            (6, 653.314_082_937),
        ] {
            let got = get(IonType::B, ordinal, 1);
            assert!(
                (got - want).abs() < 1e-6,
                "b{ordinal} = {got}, expected {want}"
            );
        }
        // y ions, charge 1: y_n = sum(residues) + WATER + PROTON.
        for (ordinal, want) in [
            (3usize, 376.171_441_366),
            (4, 477.219_119_871),
            (5, 574.271_883_746),
            (6, 703.314_476_881),
        ] {
            let got = get(IonType::Y, ordinal, 1);
            assert!(
                (got - want).abs() < 1e-6,
                "y{ordinal} = {got}, expected {want}"
            );
        }
        // Charge 2 is the relation that pins PROTON in the DENOMINATOR as well as the
        // numerator: mz2 = (mz1 - PROTON + 2*PROTON) / 2 = (mz1 + PROTON) / 2. Getting
        // the charge algebra wrong by using the neutral mass gives a value 0.5 Da away,
        // which is why this is asserted against the 1+ value rather than only a table.
        for ordinal in [3usize, 4, 5, 6] {
            let one = get(IonType::Y, ordinal, 1);
            let two = get(IonType::Y, ordinal, 2);
            assert!(
                (two - (one + PROTON) / 2.0).abs() < 1e-9,
                "y{ordinal}^2 = {two}, inconsistent with y{ordinal} = {one}"
            );
        }
        // And one absolute 2+ value, to sub-ppm.
        let y6_2 = get(IonType::Y, 6, 2);
        assert!((y6_2 - 352.160_876_674).abs() < 1e-6, "y6^2 = {y6_2}");
    }

    #[test]
    fn a_modified_peptidoform_shifts_by_exactly_the_modification_mass() {
        // Carbamidomethyl on C is +57.021464. Pinned as an absolute delta rather than
        // against `unimod_mass` alone, so a wrong table entry cannot agree with itself.
        let plain = parse_peptidoform("PEPTIDEC").unwrap().neutral_mass();
        let modded = parse_peptidoform("PEPTIDEC[Carbamidomethyl]")
            .unwrap()
            .neutral_mass();
        assert!(
            (modded - plain - 57.021_464).abs() < MDA,
            "carbamidomethyl delta = {}",
            modded - plain
        );
        // Oxidation on M is +15.994915.
        let plain_m = parse_peptidoform("PEPTIDEM").unwrap().neutral_mass();
        let ox = parse_peptidoform("PEPTIDEM[Oxidation]")
            .unwrap()
            .neutral_mass();
        assert!(
            (ox - plain_m - 15.994_915).abs() < MDA,
            "oxidation delta = {}",
            ox - plain_m
        );
    }
}
