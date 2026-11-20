//! The native predictors (docs/13_sidecars.md): a linear retention model and a heuristic
//! fragment-intensity model, so the engine also runs with no Python. The ML predictors are
//! Python sidecars called through `sidecar::run_*` over the file contract.

use mumdia_core::mass::{Fragment, IonType, ParsedPeptidoform};

/// The native retention model's identity, as a library's model identity names it.
pub const NATIVE_RT: &str = "native-rt-v1";

/// The native fragment model's identity, as a library's model identity names it.
pub const NATIVE_FRAG: &str = "native-frag-v1";

fn rt_coeff(aa: u8) -> f64 {
    // A simple, self-derived hydrophobicity-like scale (clean-room; not copied
    // from any published coefficient vector). Monotone enough to calibrate.
    match aa {
        b'A' => 1.1,
        b'R' => -1.0,
        b'N' => -0.6,
        b'D' => -0.5,
        b'C' => 0.5,
        b'E' => 0.0,
        b'Q' => -0.3,
        b'G' => 0.2,
        b'H' => -0.5,
        b'I' => 4.0,
        b'L' => 4.0,
        b'K' => -1.5,
        b'M' => 2.5,
        b'F' => 5.0,
        b'P' => 1.0,
        b'S' => -0.2,
        b'T' => 0.3,
        b'V' => 2.5,
        b'W' => 5.5,
        b'Y' => 3.0,
        _ => 0.0,
    }
}

/// The native additive retention-coefficient model's indexed retention time (arbitrary
/// units). Deterministic, no Python.
pub fn native_irt(pep: &ParsedPeptidoform) -> f32 {
    let mut s = 0.0;
    for (r, m) in pep.residues.iter().zip(&pep.mods) {
        s += rt_coeff(*r);
        // small hydrophobic shift for a mass adduct
        s += 0.01 * m;
    }
    // A terminal group is an adduct like any other. Both are 0.0 for every peptidoform
    // without one, so every earlier prediction is unchanged.
    s += 0.01 * (pep.n_term_mod + pep.c_term_mod);
    // Length term dampens very long peptides eluting infinitely late.
    s += (pep.residues.len() as f64).sqrt();
    s as f32
}

/// The native heuristic fragment-intensity model: a relative intensity per fragment, the
/// largest 1.0. Deterministic, no Python.
pub fn native_intensities(pep: &ParsedPeptidoform, frags: &[Fragment]) -> Vec<f32> {
    let l = pep.residues.len().max(1) as f64;
    let half = l / 2.0;
    let mut out: Vec<f32> = frags
        .iter()
        .map(|f| {
            let ion = match f.ion_type {
                IonType::Y => 1.0,
                IonType::B => 0.75,
            };
            // Mid-sequence fragments tend to be more intense.
            let d = ((f.ordinal as f64) - half).abs() / half.max(1.0);
            let pos = 1.0 - 0.5 * d;
            // charge-2 fragments generally weaker.
            let chg = if f.charge >= 2 { 0.5 } else { 1.0 };
            (ion * pos * chg) as f32
        })
        .collect();
    let max = out.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for v in &mut out {
            *v /= max;
        }
    }
    out
}
