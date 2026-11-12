//! SCIEX ZT Scan DIA input, recognised before a search (docs/37_ztscan.md, "Automatic for ZT
//! Scan files").
//!
//! This is the single gate of the ZT Scan analysis. An input is a ZT Scan run when the
//! isolation census `convert` refuses by (`stages::convert::IsolationCensus`, the `isolation`
//! block of `mumdia peak-census`) says that its MS2 spectra each name an isolation target and
//! no window, in the layout of a scanning quadrupole: many distinct targets, repeating cycle
//! after cycle, in steps far finer than any isolation window. Such a file is what `convert`
//! reads into spectra_ms2 v3, the version that records the Q1 sweep (`q1_cycle`, `q1_step`)
//! every key of the analysis reads; no other input is ever given the analysis.
//!
//! `run` and `run-experiment` probe every input once, before the configuration is hashed, and
//! resolve the automatic keys from the answer ([`Config::resolve_ztscan`]): on when every
//! input is a ZT Scan run, off otherwise, so every other search runs exactly as before.
//! `mumdia convert` probes its input for the half-width alone.

use anyhow::Result;
use mumdia_core::config::{Config, ModPosition, ZtScanInputs};
use rayon::prelude::*;
use tracing::{debug, info, warn};

use crate::stages::convert::{self, IsolationCensus};

/// Spectra read from the head of each input: the sample `convert` decides its refusal on
/// (10,000 spectra, 15 cycles of the 663-step K562 method), as the desktop application's
/// census reads it.
pub const PROBE_SPECTRA: usize = 10_000;

/// MS2 spectra that settle a file whose MS2 spectra state their windows (or isolate nothing):
/// when none of the first this many names a target alone, the file is not a ZT Scan run.
const PLAIN_MS2: usize = 64;

/// What one input is, for the ZT Scan analysis.
#[derive(Clone, Debug, PartialEq)]
pub enum InputKind {
    /// A SCIEX ZT Scan DIA run (a scanning quadrupole whose spectra name their targets only).
    ZtScan(ZtLayout),
    /// MS2 spectra that name differing targets and no offsets, in a layout that is not a
    /// scanning quadrupole: `convert` refuses such a file without a half-width.
    TargetOnly,
    /// Every other input: windows with offsets, all-ion spectra, a diaPASEF `.d`, a Thermo
    /// `.raw` read natively, a file that cannot be read.
    Other,
}

/// The sweep of a ZT Scan run, as the census read it.
#[derive(Clone, Debug, PartialEq)]
pub struct ZtLayout {
    /// The half-width the census suggests (five steps wide, rounded to 0.05 m/z).
    pub half_width: f64,
    /// Distinct quadrupole positions.
    pub positions: usize,
    /// The lowest and highest position and the median step, m/z.
    pub lowest_target: f64,
    pub highest_target: f64,
    pub step: f64,
}

/// One probed input.
pub struct Probe {
    pub path: String,
    pub kind: InputKind,
    census: Option<IsolationCensus>,
}

impl Probe {
    /// The census's refusal of a [`InputKind::TargetOnly`] file, as `convert` words it.
    fn refusal(&self) -> anyhow::Error {
        match &self.census {
            Some(c) => c.refusal(&self.path),
            None => anyhow::anyhow!("{}: not a ZT Scan run", self.path),
        }
    }
}

/// Probe one input. Only an mzML is read (its first [`PROBE_SPECTRA`] spectra, metadata only;
/// a file that states its windows is settled after its first 64 MS2 spectra); a diaPASEF
/// `.d` and a natively read Thermo `.raw` are never ZT Scan runs, and a vendor file reaches a
/// search only converted to mzML. A file that cannot be read is [`InputKind::Other`]: the
/// search's own checks report it.
pub fn probe(path: &str) -> Probe {
    let other = |census| Probe {
        path: path.to_string(),
        kind: InputKind::Other,
        census,
    };
    if crate::instrument::input_has_im(path)
        || crate::raw::detect(path) != crate::raw::Vendor::MzMl
        || !std::path::Path::new(path).is_file()
    {
        return other(None);
    }
    let census = convert::census_of_head(path, PROBE_SPECTRA, PLAIN_MS2);
    let d = census.decision();
    let kind = match (d.convert_refuses, d.suggested_half_width) {
        (true, Some(half_width)) => InputKind::ZtScan(ZtLayout {
            half_width,
            positions: d.positions,
            lowest_target: d.lowest_target.unwrap_or(0.0),
            highest_target: d.highest_target.unwrap_or(0.0),
            step: d.median_step.unwrap_or(0.0),
        }),
        (true, None) => InputKind::TargetOnly,
        (false, _) => return other(Some(census)),
    };
    Probe {
        path: path.to_string(),
        kind,
        census: Some(census),
    }
}

/// What the inputs of a search are.
pub struct Detection {
    /// What [`Config::resolve_ztscan`] resolves from.
    pub inputs: ZtScanInputs,
    /// Every input, in order.
    pub probes: Vec<Probe>,
}

impl Detection {
    /// The sweeps of the ZT Scan inputs.
    pub fn layouts(&self) -> Vec<&ZtLayout> {
        self.probes
            .iter()
            .filter_map(|p| match &p.kind {
                InputKind::ZtScan(l) => Some(l),
                _ => None,
            })
            .collect()
    }
}

/// Probe every input (concurrently) and classify the search. `half_width_set`: the
/// configuration sets `convert.isolation_half_width` itself. Without it, an input whose spectra
/// name differing targets and no offsets in a layout that is not a scanning quadrupole is
/// refused here, as `convert` would refuse it, and ZT Scan inputs whose censuses suggest
/// different half-widths are refused, because one configuration reads them all.
pub fn detect(paths: &[String], half_width_set: bool) -> Result<Detection> {
    let probes: Vec<Probe> = paths.par_iter().map(|p| probe(p)).collect();
    if !half_width_set {
        if let Some(p) = probes.iter().find(|p| p.kind == InputKind::TargetOnly) {
            return Err(p.refusal());
        }
    }
    let zt: Vec<(&str, f64)> = probes
        .iter()
        .filter_map(|p| match &p.kind {
            InputKind::ZtScan(l) => Some((p.path.as_str(), l.half_width)),
            _ => None,
        })
        .collect();
    let inputs = match zt.first() {
        None => ZtScanInputs::None,
        Some(&(_, half_width)) => {
            if !half_width_set && zt.iter().any(|&(_, h)| h != half_width) {
                let each: Vec<String> = zt.iter().map(|(p, h)| format!("{p}: {h} m/z")).collect();
                anyhow::bail!(
                    "the SCIEX ZT Scan DIA inputs of this search suggest different Q1 \
                     half-widths ({}): their methods differ, and one configuration reads them \
                     all. Set convert.isolation_half_width to half of the Q1 window, or search \
                     the files of each method separately (docs/37_ztscan.md, section 2).",
                    each.join("; ")
                );
            }
            if zt.len() == probes.len() {
                ZtScanInputs::All { half_width }
            } else {
                ZtScanInputs::Mixed {
                    half_width,
                    zt: zt.len(),
                    total: probes.len(),
                }
            }
        }
    };
    Ok(Detection { inputs, probes })
}

/// Detect what the inputs are, resolve the automatic keys of the ZT Scan analysis in `cfg`
/// ([`Config::resolve_ztscan`]) and say what was decided. `run` and `run-experiment` call it
/// once, before the configuration is hashed. `library_input`: the search reads an imported
/// library, whose search space was decided when it was built; for a library built from the
/// FASTA the settings measured on ZT Scan are recommended where the configuration differs
/// from them (never applied: [`fasta_library_advice`]).
pub fn resolve(
    cfg: &mut Config,
    paths: &[String],
    library_input: bool,
    caller: &str,
) -> Result<Detection> {
    let d = detect(paths, cfg.convert.isolation_half_width.is_some())?;
    let notes = cfg
        .resolve_ztscan(d.inputs)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    match d.inputs {
        ZtScanInputs::None => debug!("{caller}: no input is a SCIEX ZT Scan DIA run"),
        ZtScanInputs::All { .. } => {
            let l = d.layouts()[0];
            info!(
                "{caller}: SCIEX ZT Scan DIA input ({} quadrupole positions from {:.1} to {:.1} \
                 m/z, {:.4} m/z apart): read with the Q1 half-width {} m/z, and the ZT Scan \
                 analysis is on: {} (docs/37_ztscan.md, \"Automatic for ZT Scan files\"; set a \
                 key to false to leave it out)",
                l.positions,
                l.lowest_target,
                l.highest_target,
                l.step,
                cfg.convert.isolation_half_width.unwrap_or(l.half_width),
                {
                    let on = cfg.ztscan_keys_on();
                    if on.is_empty() {
                        "no key".to_string()
                    } else {
                        on.join(", ")
                    }
                }
            );
        }
        ZtScanInputs::Mixed { .. } => {}
    }
    for n in &notes {
        warn!("{caller}: {n}");
    }
    if !library_input && !matches!(d.inputs, ZtScanInputs::None) {
        if let Some(advice) = fasta_library_advice(cfg, &d.layouts()) {
            warn!("{caller}: {advice}");
        }
    }
    Ok(d)
}

/// The library settings measured on ZT Scan (docs/37_ztscan.md, section 11), against a
/// configuration that builds its library from the FASTA: one line naming each setting that
/// differs, or `None` when none does. A recommendation, not a change: the settings were
/// measured with an imported library predicted by another predictor, not with the engine's
/// own digest and predictors, and a library built from the FASTA is the search's search
/// space and its cache key, which the configuration states. Deamidation of N or Q is named
/// as well (docs/37_ztscan.md, section 13).
pub fn fasta_library_advice(cfg: &Config, layouts: &[&ZtLayout]) -> Option<String> {
    let (dg, pf) = (&cfg.digest, &cfg.peptidoforms);
    let mut differ: Vec<String> = Vec::new();
    if dg.missed_cleavages != 1 {
        differ.push(format!(
            "digest.missed_cleavages 1 (here {})",
            dg.missed_cleavages
        ));
    }
    if (dg.min_len, dg.max_len) != (6, 40) {
        differ.push(format!(
            "digest.min_len 6 and max_len 40 (here {} and {})",
            dg.min_len, dg.max_len
        ));
    }
    if pf.charge_by_basic_residues || (pf.charge_min, pf.charge_max) != (1, 4) {
        differ.push(format!(
            "peptidoforms.charge_min 1 and charge_max 4 (here {} and {}{})",
            pf.charge_min,
            pf.charge_max,
            if pf.charge_by_basic_residues {
                ", by basic residues"
            } else {
                ""
            }
        ));
    }
    let has = |name: &str, residue: char, position: ModPosition| {
        pf.variable_mods
            .iter()
            .any(|m| m.name == name && m.residue == residue && m.position == position)
    };
    if !has("Oxidation", 'M', ModPosition::Anywhere) {
        differ.push("Oxidation on M among peptidoforms.variable_mods".to_string());
    }
    if !has("Acetyl", '*', ModPosition::ProteinNTerm) {
        differ.push(
            "Acetyl at the protein N-terminus (residue \"*\", position \"protein_n_term\") among \
             peptidoforms.variable_mods"
                .to_string(),
        );
    }
    if pf.max_variable_mods != 2 {
        differ.push(format!(
            "peptidoforms.max_variable_mods 2 (here {})",
            pf.max_variable_mods
        ));
    }
    let hw = cfg.convert.isolation_half_width;
    let lo = layouts
        .iter()
        .map(|l| l.lowest_target)
        .fold(f64::INFINITY, f64::min);
    let hi = layouts
        .iter()
        .map(|l| l.highest_target)
        .fold(f64::NEG_INFINITY, f64::max);
    if let (Some(hw), true) = (hw, lo.is_finite() && hi.is_finite()) {
        let (min, max) = ((lo - hw).floor(), (hi + hw).ceil());
        if pf.precursor_mz_min != Some(min) || pf.precursor_mz_max != Some(max) {
            differ.push(format!(
                "peptidoforms.precursor_mz_min {min} and precursor_mz_max {max}, the range the \
                 quadrupole sweeps"
            ));
        }
    }
    let deamidated = pf
        .variable_mods
        .iter()
        .chain(&pf.fixed_mods)
        .any(|m| m.name == "Deamidated" && (m.residue == 'N' || m.residue == 'Q'));
    if differ.is_empty() && !deamidated {
        return None;
    }
    let mut s = String::from("SCIEX ZT Scan DIA input with a library built from the FASTA.");
    if !differ.is_empty() {
        s.push_str(&format!(
            " The library settings measured on ZT Scan (docs/37_ztscan.md, section 11: +3.5% \
             precursors with an imported library) differ from this configuration: {}. The \
             engine does not change a FASTA library's search space by itself (it was measured \
             with another predictor's library, and the configuration states the search space \
             and its library cache key); set them in the configuration, as the desktop \
             application does with the ZT Scan analysis.",
            differ.join("; ")
        ));
    }
    if deamidated {
        s.push_str(
            " Deamidation of N or Q is in the search space: on ZT Scan the Q1 window passes the \
             whole isotope envelope, so a deamidated form (+0.984 Da) is matched by the 13C \
             isotope of the unmodified peptide eluting with it, which neither the decoys nor an \
             entrapment library detects; leave it out (docs/37_ztscan.md, section 13).",
        );
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> ZtLayout {
        ZtLayout {
            half_width: 1.9,
            positions: 663,
            lowest_target: 394.2558,
            highest_target: 898.2707,
            step: 0.7614,
        }
    }

    /// The library advice names every setting a FASTA configuration differs in, and nothing
    /// for one built as measured.
    #[test]
    fn the_fasta_advice_names_what_differs_from_the_measured_library() {
        let mut cfg = Config::default();
        cfg.convert.isolation_half_width = Some(1.9);
        let l = layout();
        let a = fasta_library_advice(&cfg, &[&l]).expect("the defaults differ");
        for k in [
            "digest.missed_cleavages 1 (here 2)",
            "digest.min_len 6 and max_len 40 (here 5 and 50)",
            "charge_min 1 and charge_max 4 (here 2 and 3)",
            "Acetyl at the protein N-terminus",
            "max_variable_mods 2 (here 1)",
            "precursor_mz_min 392 and precursor_mz_max 901",
        ] {
            assert!(a.contains(k), "{k}: {a}");
        }
        assert!(!a.contains("Oxidation on M"), "{a}");
        assert!(!a.contains("Deamidation"), "{a}");
        let measured = Config::from_json(
            r#"{"digest":{"missed_cleavages":1,"min_len":6,"max_len":40},
                "peptidoforms":{"charge_min":1,"charge_max":4,"max_variable_mods":2,
                    "precursor_mz_min":392,"precursor_mz_max":901,
                    "variable_mods":[{"residue":"M","name":"Oxidation"},
                        {"residue":"*","name":"Acetyl","position":"protein_n_term"}]},
                "convert":{"isolation_half_width":1.9}}"#,
        )
        .unwrap();
        assert_eq!(fasta_library_advice(&measured, &[&l]), None);
        let mut deam = measured.clone();
        deam.peptidoforms
            .variable_mods
            .push(mumdia_core::config::ResidueMod {
                residue: 'N',
                name: "Deamidated".into(),
                position: ModPosition::Anywhere,
            });
        let a = fasta_library_advice(&deam, &[&l]).unwrap();
        assert!(a.contains("Deamidation of N or Q"), "{a}");
    }

    /// A missing file, a vendor path and a diaPASEF `.d` are never ZT Scan runs, and are not
    /// read.
    #[test]
    fn only_a_readable_mzml_is_probed() {
        for p in ["/no/such/file.mzML", "run.raw", "run.d", "run.wiff"] {
            assert_eq!(probe(p).kind, InputKind::Other, "{p}");
        }
        let d = detect(&["/no/such/file.mzML".to_string()], false).unwrap();
        assert_eq!(d.inputs, ZtScanInputs::None);
    }
}
