//! Physical constants, defined once (PLAN.md Section 7 "Numerics").
//!
//! Values are standard monoisotopic atomic/particle masses (public-domain
//! physical facts derived from CODATA / AME atomic masses). No coefficient
//! vector or table is copied from another proteomics implementation
//! (clean-room boundary, PLAN.md Section 11).

/// Proton mass in Da. PLAN.md Section 8 fixes MuMDIA on the physically correct
/// proton mass (not DIA-NN's H-atom value 1.007825035).
pub const PROTON: f64 = 1.007_276_466_812;

/// Monoisotopic mass of a neutral water molecule (H2O), Da.
pub const WATER: f64 = 18.010_564_684;

/// Monoisotopic mass of ammonia (NH3), Da.
pub const AMMONIA: f64 = 17.026_549_1;

/// Mass difference between two adjacent isotope peaks of a peptide (Da): the
/// true 13C - 12C mass difference (13.003_354_835 - 12 = 1.003_354_835), used as
/// the isotope-peak spacing for MS1 envelope extraction. Public physical fact
/// (AME2020 atomic masses); not copied from any proteomics implementation.
pub const ISOTOPE_SPACING: f64 = 1.003_354_835;

/// Monoisotopic residue mass in Da for a standard amino acid, or `None` for
/// residues MuMDIA treats as ambiguous (B, J, O, U, X, Z).
pub fn residue_mass(aa: u8) -> Option<f64> {
    let m = match aa.to_ascii_uppercase() {
        b'G' => 57.021_463_735,
        b'A' => 71.037_113_805,
        b'S' => 87.032_028_435,
        b'P' => 97.052_763_875,
        b'V' => 99.068_413_945,
        b'T' => 101.047_678_505,
        b'C' => 103.009_184_505,
        b'L' => 113.084_064_015,
        b'I' => 113.084_064_015,
        b'N' => 114.042_927_470,
        b'D' => 115.026_943_065,
        b'Q' => 128.058_577_540,
        b'K' => 128.094_963_050,
        b'E' => 129.042_593_135,
        b'M' => 131.040_484_645,
        b'H' => 137.058_911_875,
        b'F' => 147.068_413_945,
        b'R' => 156.101_111_050,
        b'Y' => 163.063_328_575,
        b'W' => 186.079_312_980,
        _ => return None,
    };
    Some(m)
}

/// True for the 20 standard amino acids MuMDIA enumerates.
pub fn is_standard_residue(aa: u8) -> bool {
    residue_mass(aa).is_some()
}

/// Convert a neutral monoisotopic mass to m/z at the given charge.
#[inline]
pub fn mass_to_mz(neutral_mass: f64, charge: i32) -> f64 {
    (neutral_mass + charge as f64 * PROTON) / charge as f64
}

/// ppm difference of `observed` relative to `theoretical`.
#[inline]
pub fn ppm_diff(observed: f64, theoretical: f64) -> f64 {
    1e6 * (observed - theoretical) / theoretical
}

/// True when `observed` is within `tol` ppm of `theoretical` (PLAN.md Section 4).
#[inline]
pub fn ppm_match(observed: f64, theoretical: f64, tol_ppm: f64) -> bool {
    (ppm_diff(observed, theoretical)).abs() <= tol_ppm
}

/// Lower/upper m/z bounds of a `tol` ppm window around `mz`, in f64.
#[inline]
pub fn ppm_bounds(mz: f64, tol_ppm: f64) -> (f64, f64) {
    let d = mz * tol_ppm * 1e-6;
    (mz - d, mz + d)
}

/// Canonical within-tolerance predicate for the fragment index (fragindex_spec
/// Section 2.1): multiplicative, symmetric relative to the SMALLER mass. Two m/z
/// values are within tolerance iff `hi - lo <= tol_ppm*1e-6 * lo`, equivalently
/// `hi/lo <= 1 + delta`, equivalently `ln(hi) - ln(lo) <= ln(1 + delta)` (the last
/// form is what makes log-space binning exact). Computed in f64; f32-stored m/z is
/// widened by the caller. This is a min-relative form and differs at the tolerance
/// edge from the query-relative [`ppm_bounds`] and the theoretical-relative
/// [`ppm_diff`]; it is the predicate the log-bin +/-1 probe is proven exact against.
#[inline]
pub fn within_ppm(a: f64, b: f64, tol_ppm: f64) -> bool {
    let lo = a.min(b);
    let hi = a.max(b);
    hi - lo <= tol_ppm * 1e-6 * lo
}

#[cfg(test)]
mod ppm_tests {
    use super::within_ppm;

    #[test]
    fn within_ppm_three_forms_agree() {
        let delta = 20.0f64 * 1e-6;
        // points chosen away from the exact 20 ppm boundary (8, 10, 50, 28.6 ppm)
        // so the three algebraic forms round identically
        let cases: [(f64, f64); 4] = [
            (500.0, 500.004),
            (1000.0, 1000.01),
            (1999.9, 2000.0),
            (700.0, 700.02),
        ];
        for &(a, b) in &cases {
            let lo = a.min(b);
            let hi = a.max(b);
            let f_sub = hi - lo <= delta * lo;
            let f_ratio = hi / lo <= 1.0 + delta;
            let f_log = hi.ln() - lo.ln() <= (1.0 + delta).ln();
            assert_eq!(f_sub, within_ppm(a, b, 20.0));
            // ratio and sub forms are algebraically identical
            assert_eq!(f_sub, f_ratio, "sub vs ratio at ({a},{b})");
            // log form agrees except within fp epsilon of the exact edge; our
            // sample points are chosen away from the exact boundary
            assert_eq!(f_sub, f_log, "sub vs log at ({a},{b})");
        }
    }

    #[test]
    fn within_ppm_edges() {
        // exactly at the edge (hi-lo == delta*lo) is inclusive
        let lo = 1000.0;
        let hi = lo + 10.0 * 1e-6 * lo; // 10 ppm above
        assert!(within_ppm(lo, hi, 10.0));
        assert!(!within_ppm(lo, hi, 9.9));
        assert!(within_ppm(hi, lo, 10.0)); // symmetric in argument order
    }
}
