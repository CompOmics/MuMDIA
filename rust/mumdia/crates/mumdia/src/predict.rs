//! Predictor traits + native fallback implementations (PLAN.md Section 0, 3.2).
//!
//! The three ML predictors are Python sidecars behind stable interfaces. So the
//! engine also runs with zero external runtime dependencies, MVP ships native
//! fallbacks (a linear retention model and a heuristic fragment-intensity
//! model) selected by config. The file contract (request JSON + Parquet ->
//! response) is the boundary, so a sidecar can replace the native path without
//! changing callers.

use mumdia_core::mass::{Fragment, IonType, ParsedPeptidoform};

/// Predicts an indexed retention time (arbitrary units) per peptidoform.
pub trait RtPredictor {
    fn predict_irt(&self, pep: &ParsedPeptidoform) -> f32;
    fn identity(&self) -> String;
}

/// Predicts a relative intensity per theoretical fragment.
pub trait FragmentPredictor {
    fn predict_intensities(&self, pep: &ParsedPeptidoform, frags: &[Fragment]) -> Vec<f32>;
    fn identity(&self) -> String;
}

/// Native additive retention-coefficient model. Deterministic, no Python.
pub struct NativeRt;

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

impl RtPredictor for NativeRt {
    fn predict_irt(&self, pep: &ParsedPeptidoform) -> f32 {
        let mut s = 0.0;
        for (r, m) in pep.residues.iter().zip(&pep.mods) {
            s += rt_coeff(*r);
            // small hydrophobic shift for a mass adduct
            s += 0.01 * m;
        }
        // Length term dampens very long peptides eluting infinitely late.
        s += (pep.residues.len() as f64).sqrt();
        s as f32
    }
    fn identity(&self) -> String {
        "native-rt-v1".to_string()
    }
}

/// Native heuristic fragment-intensity model. Deterministic, no Python.
pub struct NativeFrag;

impl FragmentPredictor for NativeFrag {
    fn predict_intensities(&self, pep: &ParsedPeptidoform, frags: &[Fragment]) -> Vec<f32> {
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
    fn identity(&self) -> String {
        "native-frag-v1".to_string()
    }
}
