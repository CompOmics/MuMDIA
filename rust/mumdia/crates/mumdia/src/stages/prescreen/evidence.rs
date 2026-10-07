//! Optional tag-based evidence components, computed per candidate and eligible spectrum and
//! maximised over spectra, each component independently (the prototype's `score_candidates`):
//!
//! - `tag`: spectrum-side information of each distinct candidate trimer observed in the
//!   spectrum, `-ln((n_w(key) + 0.5) / (n_w + 1))`, times the whole-path soft fit;
//! - `fasta`: the library-side information of the same trimers, `-ln((docs + 0.5) / (families
//!   + 1))`, times the fit, kept as its own column rather than multiplied into the spectrum one;
//! - `complement`: spectrum information times complementary support (`tags::complementary_path`,
//!   distinct peaks off the path, no reuse), divided by `sqrt(L)` as the prototype's filter did;
//! - `flank`: spectrum information times the best positioned ladder (`tags::matched_path`)
//!   whose endpoints sit at the candidate's own b or y masses.
//!
//! A trimer repeated in the candidate counts once (deduplicated by reversal-canonical code), so
//! repeated motifs and overlapping occurrences do not receive unrestricted duplicate weight.

use std::collections::HashMap;

use mumdia_core::constants::{PROTON, WATER};

use super::retrieval::BackboneIndex;
use super::tags::{complementary_path, fit_weight, matched_path, Alphabet, SpectrumTags};

pub const N_COMPONENTS: usize = 4;
pub const TAG: usize = 0;
pub const FASTA: usize = 1;
pub const COMPLEMENT: usize = 2;
pub const FLANK: usize = 3;

/// Which components to compute.
#[derive(Clone, Copy, Debug, Default)]
pub struct Wanted {
    pub tag: bool,
    pub fasta: bool,
    pub complement: bool,
    pub flank: bool,
}

impl Wanted {
    pub fn any(&self) -> bool {
        self.tag || self.fasta || self.complement || self.flank
    }
}

/// One window's spectrum-side tag frequencies: key -> number of spectra holding it.
pub struct WindowFreq {
    pub count: HashMap<u32, u32>,
    pub n: u32,
}

impl WindowFreq {
    pub fn build(tags: &[SpectrumTags]) -> WindowFreq {
        let mut count: HashMap<u32, u32> = HashMap::new();
        for t in tags {
            for &k in &t.keys {
                *count.entry(k).or_default() += 1;
            }
        }
        WindowFreq {
            count,
            n: tags.len() as u32,
        }
    }

    pub fn information(&self, key: u32) -> f64 {
        let c = self.count.get(&key).copied().unwrap_or(0);
        -((c as f64 + 0.5) / (self.n as f64 + 1.0)).ln()
    }
}

/// Everything about a candidate the components need.
pub struct CandidateTags<'a> {
    /// Residue states, every position expressible (a candidate with an unexpressible position
    /// still scores the trimers that avoid it).
    pub states: &'a [Option<u16>],
    /// Residue masses as the core score uses them.
    pub masses: &'a [f64],
    pub zmax: i32,
}

pub struct Params<'a> {
    pub alpha: &'a Alphabet,
    pub tol: f64,
    pub sigma: f64,
    pub index: Option<&'a BackboneIndex>,
}

/// The components for one spectrum. `mz`: the spectrum's positive peaks, ascending.
pub fn spectrum_components(
    c: &CandidateTags,
    tags: &SpectrumTags,
    freq: &WindowFreq,
    mz: &[f64],
    want: Wanted,
    p: &Params,
) -> [f64; N_COMPONENTS] {
    let alpha = p.alpha;
    let l = c.states.len();
    let neutral: f64 = c.masses.iter().sum::<f64>() + WATER;
    let mut out = [0.0; N_COMPONENTS];
    let mut seen: Vec<u32> = Vec::new();
    for q in 0..l.saturating_sub(2) {
        let (Some(a), Some(b), Some(d)) = (c.states[q], c.states[q + 1], c.states[q + 2]) else {
            continue;
        };
        let code = alpha.canonical(a, b, d);
        if seen.contains(&code) {
            continue;
        }
        seen.push(code);
        let mut ev = [0.0f64; N_COMPONENTS];
        for z in 1..=c.zmax {
            for gapped in [false, true] {
                let key = alpha.key(code, z, gapped);
                let Some(rms) = tags.get(key) else {
                    continue;
                };
                let rarity = freq.information(key);
                let soft = fit_weight(rms as f64, p.sigma);
                ev[TAG] = ev[TAG].max(rarity * soft);
                if want.fasta {
                    if let Some(ix) = p.index {
                        ev[FASTA] = ev[FASTA].max(ix.information(alpha.backbone(code)) * soft);
                    }
                }
                if gapped {
                    // Complement and flank traverse four-peak ladders only.
                    continue;
                }
                if want.complement {
                    let comp = complementary_path(
                        mz,
                        [a, b, d],
                        z,
                        neutral,
                        c.zmax,
                        alpha,
                        p.tol,
                        p.sigma,
                    )
                    .max(complementary_path(
                        mz,
                        [d, b, a],
                        z,
                        neutral,
                        c.zmax,
                        alpha,
                        p.tol,
                        p.sigma,
                    ));
                    ev[COMPLEMENT] = ev[COMPLEMENT].max(rarity * comp);
                }
                if want.flank {
                    ev[FLANK] = ev[FLANK].max(rarity * flank(c, code, z, mz, p));
                }
            }
        }
        for k in 0..N_COMPONENTS {
            out[k] += ev[k];
        }
    }
    out[COMPLEMENT] /= (l as f64).sqrt();
    if !want.tag {
        out[TAG] = 0.0;
    }
    out
}

/// Best positioned ladder of trimer `code` at charge `z`: for every occurrence in either full
/// orientation, a b-direction path from `prefix(q)` to `prefix(q + 3)` and a y-direction path
/// over the complementary suffix masses.
fn flank(c: &CandidateTags, code: u32, z: i32, mz: &[f64], p: &Params) -> f64 {
    let alpha = p.alpha;
    let l = c.states.len();
    let zf = z as f64;
    let mut best = 0.0f64;
    for rev in [false, true] {
        let at = |i: usize| if rev { l - 1 - i } else { i };
        let mut prefix = vec![0.0; l + 1];
        for i in 0..l {
            prefix[i + 1] = prefix[i] + c.masses[at(i)];
        }
        for q in 0..l.saturating_sub(2) {
            let (Some(x), Some(y), Some(v)) =
                (c.states[at(q)], c.states[at(q + 1)], c.states[at(q + 2)])
            else {
                continue;
            };
            if alpha.canonical(x, y, v) != code {
                continue;
            }
            let start = prefix[q] / zf + PROTON;
            let end = prefix[q + 3] / zf + PROTON;
            best = best.max(matched_path(
                mz,
                start,
                end,
                [x, y, v],
                z,
                alpha,
                p.tol,
                p.sigma,
            ));
            let start = (prefix[l] - prefix[q + 3] + WATER) / zf + PROTON;
            let end = (prefix[l] - prefix[q] + WATER) / zf + PROTON;
            best = best.max(matched_path(
                mz,
                start,
                end,
                [v, y, x],
                z,
                alpha,
                p.tol,
                p.sigma,
            ));
        }
    }
    best
}
