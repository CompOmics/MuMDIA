//! Stage A2 `mumdia peptidoforms`: expand stripped peptides into peptidoforms
//! (PLAN.md Stage A2). Fixed + variable modification enumeration and charge
//! states, emitted as ProForma strings with UniMod names. Experiment-wide.

use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::PeptidoformsConfig;
use mumdia_core::mass::unimod_mass;
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, Table};
use serde_json::json;
use tracing::info;

/// Build a ProForma-lite string from a stripped peptide and residue-position ->
/// mod-name assignments.
fn proforma(pep: &str, mods: &[(usize, &str)]) -> String {
    let mut s = String::with_capacity(pep.len() + mods.len() * 12);
    for (i, c) in pep.chars().enumerate() {
        s.push(c);
        if let Some((_, name)) = mods.iter().find(|(p, _)| *p == i) {
            s.push('[');
            s.push_str(name);
            s.push(']');
        }
    }
    s
}

/// Enumerate variable-mod combinations (subsets of candidate positions) up to
/// `max_var`. Returns each as a list of (position, mod_name).
fn variable_combos<'a>(
    positions: &[(usize, &'a str)],
    max_var: usize,
) -> Vec<Vec<(usize, &'a str)>> {
    let mut out = vec![Vec::new()];
    let n = positions.len();
    // subsets of size 1..=max_var
    for k in 1..=max_var.min(n) {
        let mut idx: Vec<usize> = (0..k).collect();
        loop {
            out.push(idx.iter().map(|&i| positions[i]).collect());
            // advance combination
            let mut i = k as isize - 1;
            while i >= 0 && idx[i as usize] == n - k + i as usize {
                i -= 1;
            }
            if i < 0 {
                break;
            }
            idx[i as usize] += 1;
            for j in (i as usize + 1)..k {
                idx[j] = idx[j - 1] + 1;
            }
        }
    }
    out
}

pub struct PeptidoformsParams<'a> {
    pub peptides: &'a str,
    pub out: &'a str,
    pub cfg: &'a PeptidoformsConfig,
    pub config_hash: &'a str,
}

pub fn run(p: PeptidoformsParams) -> Result<u64> {
    let t0 = Instant::now();
    let t = Table::read(p.peptides)?;
    let pep_id = t.u32("id")?;
    let peptide = t.str("peptide")?;
    let protein = t.str("protein")?;
    let label = t.str("label")?;
    let target_id = t.i32("target_id")?;

    // Validate mod names up front (unknown modification = error, PLAN.md A2).
    for m in p.cfg.fixed_mods.iter().chain(p.cfg.variable_mods.iter()) {
        if unimod_mass(&m.name).is_none() {
            anyhow::bail!("unknown modification '{}' in config (fixed/variable)", m.name);
        }
    }

    let (mut id_c, mut pepid_c, mut base_c) = (Vec::new(), Vec::new(), Vec::new());
    let (mut pep_c, mut pform_c, mut charge_c) = (Vec::new(), Vec::new(), Vec::new());
    let (mut label_c, mut prot_c) = (Vec::new(), Vec::new());
    let mut next: u32 = 0;

    for row in 0..t.nrows {
        let pep = &peptide[row];
        let base_pep_id = if target_id[row] >= 0 {
            target_id[row] as u32
        } else {
            pep_id[row]
        };

        // Fixed mods: apply to every matching residue.
        let mut fixed: Vec<(usize, &str)> = Vec::new();
        for (i, c) in pep.chars().enumerate() {
            for fm in &p.cfg.fixed_mods {
                if c == fm.residue {
                    fixed.push((i, fm.name.as_str()));
                }
            }
        }
        // Variable-mod candidate positions.
        let mut var_positions: Vec<(usize, &str)> = Vec::new();
        for (i, c) in pep.chars().enumerate() {
            for vm in &p.cfg.variable_mods {
                if c == vm.residue {
                    var_positions.push((i, vm.name.as_str()));
                }
            }
        }

        for combo in variable_combos(&var_positions, p.cfg.max_variable_mods) {
            let mut mods = fixed.clone();
            mods.extend(combo);
            mods.sort_by_key(|(pos, _)| *pos);
            let form = proforma(pep, &mods);
            for z in p.cfg.charge_min..=p.cfg.charge_max {
                id_c.push(next);
                next += 1;
                pepid_c.push(pep_id[row]);
                base_c.push(base_pep_id);
                pep_c.push(pep.clone());
                pform_c.push(form.clone());
                charge_c.push(z);
                label_c.push(label[row].clone());
                prot_c.push(protein[row].clone());
            }
        }
    }

    let rows = write_table(
        p.out,
        vec![
            Col::U32("id".into(), id_c),
            Col::U32("peptide_id".into(), pepid_c),
            Col::U32("base_peptide_id".into(), base_c),
            Col::Str("peptide".into(), pep_c),
            Col::Str("peptidoform".into(), pform_c),
            Col::I32("charge".into(), charge_c),
            Col::Str("label".into(), label_c),
            Col::Str("protein".into(), prot_c),
        ],
    )?;

    let elapsed = t0.elapsed().as_millis();
    ArtifactReport {
        logical_name: artifact::PEPTIDOFORMS.0.to_string(),
        schema_name: artifact::PEPTIDOFORMS.0.to_string(),
        schema_version: artifact::PEPTIDOFORMS.1,
        stage: "peptidoforms".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({
            "fixed_mods": p.cfg.fixed_mods.iter().map(|m| format!("{}:{}", m.residue, m.name)).collect::<Vec<_>>(),
            "variable_mods": p.cfg.variable_mods.iter().map(|m| format!("{}:{}", m.residue, m.name)).collect::<Vec<_>>(),
            "max_variable_mods": p.cfg.max_variable_mods,
            "charges": format!("{}..{}", p.cfg.charge_min, p.cfg.charge_max),
        }),
        stats: Default::default(),
        model_identity: None,
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;

    info!(rows, elapsed_ms = elapsed, "peptidoforms: done");
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proforma_places_mods() {
        let s = proforma("PEPCM", &[(3, "Carbamidomethyl"), (4, "Oxidation")]);
        assert_eq!(s, "PEPC[Carbamidomethyl]M[Oxidation]");
    }

    #[test]
    fn combos_bounded() {
        let pos = vec![(1usize, "Oxidation"), (4, "Oxidation"), (7, "Oxidation")];
        // max 1 variable mod -> empty + 3 singles = 4
        assert_eq!(variable_combos(&pos, 1).len(), 4);
        // max 2 -> empty + 3 singles + 3 pairs = 7
        assert_eq!(variable_combos(&pos, 2).len(), 7);
    }
}
