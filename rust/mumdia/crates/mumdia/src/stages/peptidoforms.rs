//! Stage A2 `mumdia peptidoforms`: expand stripped peptides into peptidoforms
//! (PLAN.md Stage A2). Fixed + variable modification enumeration and charge
//! states, emitted as ProForma strings with UniMod names. Experiment-wide.

use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{PeptidoformsConfig, ResidueMod, UnknownModPolicy};
use mumdia_core::mass::unimod_mass;
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, TableFile};
use serde_json::json;
use tracing::{info, warn};

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

/// Expand the alternative modification choices for one selected set of residue
/// sites. `selected` contains indices into `sites` in ascending position order.
fn expand_site_choices<'a>(
    sites: &[(usize, Vec<&'a str>)],
    selected: &[usize],
    depth: usize,
    current: &mut Vec<(usize, &'a str)>,
    out: &mut Vec<Vec<(usize, &'a str)>>,
) {
    if depth == selected.len() {
        out.push(current.clone());
        return;
    }
    let (position, alternatives) = &sites[selected[depth]];
    for &name in alternatives {
        current.push((*position, name));
        expand_site_choices(sites, selected, depth + 1, current, out);
        current.pop();
    }
}

/// Enumerate variable-mod combinations up to `max_var` modified residue sites.
/// Every site contributes zero or one of its configured alternatives; two
/// modifications can therefore never be stacked on the same residue. Forms are
/// emitted by modified-site count and then by residue/config order, preserving
/// the legacy order when each residue has one possible variable modification.
fn variable_combos<'a>(
    sites: &[(usize, Vec<&'a str>)],
    max_var: usize,
) -> Vec<Vec<(usize, &'a str)>> {
    let mut out = vec![Vec::new()];
    let n = sites.len();
    for k in 1..=max_var.min(n) {
        let mut idx: Vec<usize> = (0..k).collect();
        loop {
            expand_site_choices(sites, &idx, 0, &mut Vec::with_capacity(k), &mut out);
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

fn known_mods<'a>(
    mods: &'a [ResidueMod],
    kind: &str,
    policy: UnknownModPolicy,
) -> Result<Vec<(char, &'a str)>> {
    let mut out = Vec::with_capacity(mods.len());
    for m in mods {
        if m.residue == '*' {
            anyhow::bail!(
                "unsupported '*' residue for {kind} modification '{}'; wildcard/terminal \
                 modifications are not implemented",
                m.name
            );
        }
        if unimod_mass(&m.name).is_none() {
            match policy {
                UnknownModPolicy::Error => {
                    anyhow::bail!("unknown {kind} modification '{}' in config", m.name)
                }
                UnknownModPolicy::Skip => {
                    warn!(
                        modification = %m.name,
                        residue = %m.residue,
                        mod_kind = kind,
                        "peptidoforms: skipping unknown modification"
                    );
                    continue;
                }
            }
        }
        out.push((m.residue, m.name.as_str()));
    }
    Ok(out)
}

/// Validate and normalize modification/charge rules before expanding any
/// peptides. Exact duplicate variable alternatives are removed in config order;
/// distinct alternatives on one residue remain valid choices.
type ModRule<'a> = (char, &'a str);

fn validated_rules(cfg: &PeptidoformsConfig) -> Result<(Vec<ModRule<'_>>, Vec<ModRule<'_>>)> {
    if cfg.charge_min < 1 {
        anyhow::bail!(
            "peptidoforms.charge_min must be >= 1 (got {})",
            cfg.charge_min
        );
    }
    if cfg.charge_min > cfg.charge_max {
        anyhow::bail!(
            "peptidoforms charge range is empty: charge_min {} > charge_max {}",
            cfg.charge_min,
            cfg.charge_max
        );
    }

    let fixed = known_mods(&cfg.fixed_mods, "fixed", cfg.unknown_modification)?;
    let variable = known_mods(&cfg.variable_mods, "variable", cfg.unknown_modification)?;

    // At most one fixed modification may target a residue: otherwise every
    // matching peptide site would require an unsupported stacked assignment.
    let mut fixed_by_residue: BTreeMap<char, &str> = BTreeMap::new();
    for &(residue, name) in &fixed {
        if let Some(previous) = fixed_by_residue.insert(residue, name) {
            anyhow::bail!(
                "multiple fixed modifications target residue '{residue}' \
                 ('{previous}' and '{name}'); stacked fixed modifications are unsupported"
            );
        }
    }

    // Duplicate variable config entries must not duplicate candidate rows.
    let mut seen_variable: HashSet<(char, &str)> = HashSet::new();
    let mut variable_dedup = Vec::with_capacity(variable.len());
    for (residue, name) in variable {
        if !seen_variable.insert((residue, name)) {
            warn!(
                modification = name,
                residue = %residue,
                "peptidoforms: ignoring duplicate variable modification rule"
            );
            continue;
        }
        if let Some(fixed_name) = fixed_by_residue.get(&residue) {
            anyhow::bail!(
                "residue '{residue}' has fixed modification '{fixed_name}' and variable \
                 modification '{name}'; fixed-variable stacking on one site is unsupported"
            );
        }
        variable_dedup.push((residue, name));
    }

    Ok((fixed, variable_dedup))
}

pub struct PeptidoformsParams<'a> {
    pub peptides: &'a str,
    pub out: &'a str,
    pub cfg: &'a PeptidoformsConfig,
    pub config_hash: &'a str,
}

pub fn run(p: PeptidoformsParams) -> Result<u64> {
    let t0 = Instant::now();
    let (fixed_rules, variable_rules) = validated_rules(p.cfg)?;
    let t = TableFile::open(p.peptides)?;
    let pep_id = t.u32("id")?;
    let peptide = t.str("peptide")?;
    let protein = t.str("protein")?;
    let label = t.str("label")?;
    let target_id = t.i32("target_id")?;

    let (mut id_c, mut pepid_c, mut base_c) = (Vec::new(), Vec::new(), Vec::new());
    let (mut pep_c, mut pform_c, mut charge_c) = (Vec::new(), Vec::new(), Vec::new());
    let (mut label_c, mut prot_c) = (Vec::new(), Vec::new());
    let mut next: u32 = 0;

    for row in 0..t.nrows {
        let pep = &peptide[row];
        // Composition-based precursor charge range (opt-in). The maximum charge a
        // peptide can physically hold is one proton on the N-terminus plus one per
        // basic residue (Arg, His, Lys); when enabled, emit charges 1..=cap and
        // ignore the fixed charge_min/charge_max. Otherwise use the fixed range.
        let (z_lo, z_hi) = if p.cfg.charge_by_basic_residues {
            let n_basic = pep
                .bytes()
                .filter(|b| matches!(b, b'R' | b'H' | b'K'))
                .count() as i32;
            (1, 1 + n_basic)
        } else {
            (p.cfg.charge_min, p.cfg.charge_max)
        };
        let base_pep_id = if target_id[row] >= 0 {
            target_id[row] as u32
        } else {
            pep_id[row]
        };

        // Fixed mods: apply to every matching residue.
        let mut fixed: Vec<(usize, &str)> = Vec::new();
        for (i, c) in pep.chars().enumerate() {
            for &(residue, name) in &fixed_rules {
                if c == residue {
                    fixed.push((i, name));
                }
            }
        }
        // Variable-mod candidate sites, each with zero-or-one alternatives.
        let mut var_sites: Vec<(usize, Vec<&str>)> = Vec::new();
        for (i, c) in pep.chars().enumerate() {
            let mut alternatives: Vec<&str> = variable_rules
                .iter()
                .filter_map(|&(residue, name)| (c == residue).then_some(name))
                .collect();
            // Deduplicate the per-site alternatives (order-preserving). A config that
            // lists the same variable modification twice for one residue would otherwise
            // generate the identical peptidoform more than once. This used to be caught
            // downstream by hashing every emitted ProForma string into a `seen_forms`
            // HashSet -- tens of millions of String clones and hashes to guard against a
            // degenerate config. Deduping the handful of names per site is equivalent and
            // free: `variable_combos` picks distinct position subsets and one alternative
            // per chosen position, so once the alternatives are unique every emitted form
            // is unique by construction.
            let mut seen_alt = 0usize;
            while seen_alt < alternatives.len() {
                if alternatives[..seen_alt].contains(&alternatives[seen_alt]) {
                    alternatives.remove(seen_alt);
                } else {
                    seen_alt += 1;
                }
            }
            if !alternatives.is_empty() {
                var_sites.push((i, alternatives));
            }
        }

        for combo in variable_combos(&var_sites, p.cfg.max_variable_mods) {
            let mut mods = fixed.clone();
            mods.extend(combo);
            mods.sort_by_key(|(pos, _)| *pos);
            let form = proforma(pep, &mods);
            for z in z_lo..=z_hi {
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
        let sites = vec![
            (1usize, vec!["Oxidation"]),
            (4, vec!["Oxidation"]),
            (7, vec!["Oxidation"]),
        ];
        // max 1 variable mod -> empty + 3 singles = 4
        assert_eq!(variable_combos(&sites, 1).len(), 4);
        // max 2 -> empty + 3 singles + 3 pairs = 7
        assert_eq!(variable_combos(&sites, 2).len(), 7);
    }

    #[test]
    fn default_rules_and_form_order_are_preserved() {
        let cfg = PeptidoformsConfig::default();
        let (fixed, variable) = validated_rules(&cfg).unwrap();
        assert_eq!(fixed, vec![('C', "Carbamidomethyl")]);
        assert_eq!(variable, vec![('M', "Oxidation")]);

        let sites = vec![(1usize, vec!["Oxidation"]), (4usize, vec!["Oxidation"])];
        assert_eq!(
            variable_combos(&sites, cfg.max_variable_mods),
            vec![vec![], vec![(1, "Oxidation")], vec![(4, "Oxidation")],]
        );
    }

    #[test]
    fn same_site_alternatives_never_stack_and_are_deterministic() {
        let sites = vec![(1usize, vec!["Oxidation", "Acetyl"]), (4, vec!["Phospho"])];
        let combos = variable_combos(&sites, 2);
        assert_eq!(
            combos,
            vec![
                vec![],
                vec![(1, "Oxidation")],
                vec![(1, "Acetyl")],
                vec![(4, "Phospho")],
                vec![(1, "Oxidation"), (4, "Phospho")],
                vec![(1, "Acetyl"), (4, "Phospho")],
            ]
        );
        assert!(combos.iter().all(|combo| {
            let positions: HashSet<usize> = combo.iter().map(|(position, _)| *position).collect();
            positions.len() == combo.len()
        }));
    }

    #[test]
    fn skip_policy_removes_unknown_modification() {
        let mut cfg = PeptidoformsConfig::default();
        cfg.variable_mods.push(ResidueMod {
            residue: 'M',
            name: "DefinitelyNotAUnimodName".to_string(),
        });
        cfg.unknown_modification = UnknownModPolicy::Skip;
        let (_fixed, variable) = validated_rules(&cfg).unwrap();
        assert_eq!(variable, vec![('M', "Oxidation")]);
    }

    #[test]
    fn fixed_stacking_and_fixed_variable_overlap_are_rejected() {
        let mut stacked = PeptidoformsConfig::default();
        stacked.fixed_mods.push(ResidueMod {
            residue: 'C',
            name: "Oxidation".to_string(),
        });
        let err = validated_rules(&stacked).unwrap_err().to_string();
        assert!(err.contains("multiple fixed modifications"));

        let mut overlap = PeptidoformsConfig::default();
        overlap.variable_mods.push(ResidueMod {
            residue: 'C',
            name: "Oxidation".to_string(),
        });
        let err = validated_rules(&overlap).unwrap_err().to_string();
        assert!(err.contains("fixed-variable stacking"));
    }

    #[test]
    fn invalid_charges_and_wildcard_modifications_are_rejected() {
        let zero_charge = PeptidoformsConfig {
            charge_min: 0,
            ..PeptidoformsConfig::default()
        };
        assert!(validated_rules(&zero_charge)
            .unwrap_err()
            .to_string()
            .contains("charge_min must be >= 1"));

        let reversed = PeptidoformsConfig {
            charge_min: 4,
            charge_max: 3,
            ..PeptidoformsConfig::default()
        };
        assert!(validated_rules(&reversed)
            .unwrap_err()
            .to_string()
            .contains("charge range is empty"));

        let mut wildcard = PeptidoformsConfig::default();
        wildcard.variable_mods.push(ResidueMod {
            residue: '*',
            name: "Oxidation".to_string(),
        });
        assert!(validated_rules(&wildcard)
            .unwrap_err()
            .to_string()
            .contains("unsupported '*' residue"));
    }
}
