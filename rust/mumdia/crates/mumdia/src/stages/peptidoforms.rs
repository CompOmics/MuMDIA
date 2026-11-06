//! Stage A2 `mumdia peptidoforms`: expand stripped peptides into peptidoforms
//! (docs/05_digest_peptidoforms.md). Fixed + variable modification enumeration
//! and charge states, emitted as ProForma strings with UniMod names.
//! Experiment-wide.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{Config, ModPosition, PeptidoformsConfig, ResidueMod, UnknownModPolicy};
use mumdia_core::constants::{mass_to_mz, residue_mass, WATER};
use mumdia_core::mass::unimod_mass;
use mumdia_core::schema::artifact;
use mumdia_io::report::{ArtifactReport, Written};
use mumdia_io::table::{write_table, Col, TableFile};
use serde_json::json;
use tracing::{info, warn};

/// The slot of the N-terminal group in a peptide's site numbering: slot 0 is the N-terminus
/// and slot `i + 1` is residue `i`, so a terminal site sorts before every residue site and
/// the residue sites keep their relative order.
const N_TERM_SLOT: usize = 0;

/// The slot of residue `i`.
fn residue_slot(i: usize) -> usize {
    i + 1
}

/// Build a ProForma-lite string from a stripped peptide and slot -> mod-name assignments
/// ([`N_TERM_SLOT`] for the N-terminal group, written `[Name]-` before the first residue,
/// and [`residue_slot`] for a residue).
fn proforma(pep: &str, mods: &[(usize, &str)]) -> String {
    let mut s = String::with_capacity(pep.len() + mods.len() * 12);
    if let Some((_, name)) = mods.iter().find(|(p, _)| *p == N_TERM_SLOT) {
        s.push('[');
        s.push_str(name);
        s.push_str("]-");
    }
    for (i, c) in pep.chars().enumerate() {
        s.push(c);
        if let Some((_, name)) = mods.iter().find(|(p, _)| *p == residue_slot(i)) {
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
) -> Result<Vec<&'a ResidueMod>> {
    let mut out = Vec::with_capacity(mods.len());
    for m in mods {
        if m.residue == '*' && m.position.is_anywhere() {
            anyhow::bail!(
                "unsupported '*' residue for {kind} modification '{}'; a wildcard residue \
                 is valid only for a terminal rule (position peptide_n_term or \
                 protein_n_term)",
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
        out.push(m);
    }
    Ok(out)
}

/// Validate and normalize modification/charge rules before expanding any
/// peptides. Exact duplicate variable alternatives are removed in config order;
/// distinct alternatives on one residue remain valid choices.
type ModRule<'a> = (char, &'a str);

/// A rule on the N-terminal amine: where it applies, and the modification name.
type TermRule<'a> = (ModPosition, &'a str);

/// The validated rules: residue rules as before, plus the N-terminal ones.
#[derive(Debug)]
struct Rules<'a> {
    fixed: Vec<ModRule<'a>>,
    variable: Vec<ModRule<'a>>,
    /// At most one fixed N-terminal rule ([`PeptidoformsConfig::terminal_rule_problem`]).
    fixed_n_term: Option<TermRule<'a>>,
    /// The variable N-terminal alternatives, in configuration order, duplicates removed.
    variable_n_term: Vec<TermRule<'a>>,
}

impl Rules<'_> {
    /// Whether a rule applies at the protein N-terminus only, so the stage needs the
    /// digest's `protein_n_term` flag.
    fn needs_protein_flag(&self) -> bool {
        self.fixed_n_term
            .iter()
            .chain(&self.variable_n_term)
            .any(|(p, _)| *p == ModPosition::ProteinNTerm)
    }

    fn has_n_term(&self) -> bool {
        self.fixed_n_term.is_some() || !self.variable_n_term.is_empty()
    }
}

/// Whether an N-terminal rule at `position` applies to a peptide whose protein-N-terminus
/// flag is `starts_protein`.
fn term_rule_applies(position: ModPosition, starts_protein: bool) -> bool {
    match position {
        ModPosition::PeptideNTerm => true,
        ModPosition::ProteinNTerm => starts_protein,
        _ => false,
    }
}

fn validated_rules(cfg: &PeptidoformsConfig) -> Result<Rules<'_>> {
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

    if let Some(problem) = cfg.terminal_rule_problem() {
        anyhow::bail!(problem);
    }
    let fixed_all = known_mods(&cfg.fixed_mods, "fixed", cfg.unknown_modification)?;
    let variable_all = known_mods(&cfg.variable_mods, "variable", cfg.unknown_modification)?;
    let fixed: Vec<ModRule> = fixed_all
        .iter()
        .filter(|m| m.position.is_anywhere())
        .map(|m| (m.residue, m.name.as_str()))
        .collect();
    let variable: Vec<ModRule> = variable_all
        .iter()
        .filter(|m| m.position.is_anywhere())
        .map(|m| (m.residue, m.name.as_str()))
        .collect();
    let fixed_n_term = fixed_all
        .iter()
        .find(|m| m.position.is_n_term())
        .map(|m| (m.position, m.name.as_str()));
    let mut variable_n_term: Vec<TermRule> = Vec::new();
    for m in variable_all.iter().filter(|m| m.position.is_n_term()) {
        let rule = (m.position, m.name.as_str());
        if variable_n_term.contains(&rule) {
            warn!(
                modification = %m.name,
                position = m.position.name(),
                "peptidoforms: ignoring duplicate variable N-terminal modification rule"
            );
            continue;
        }
        variable_n_term.push(rule);
    }

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

    Ok(Rules {
        fixed,
        variable: variable_dedup,
        fixed_n_term,
        variable_n_term,
    })
}

/// The precursor m/z limits (`peptidoforms.precursor_mz_min` / `precursor_mz_max`), both
/// inclusive; an unset end is unbounded.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MzRange {
    lo: f64,
    hi: f64,
}

impl MzRange {
    /// The configured range, or `None` when neither end is set (the default, where the
    /// stage emits every charge state exactly as before).
    fn of(cfg: &PeptidoformsConfig) -> Option<MzRange> {
        if cfg.precursor_mz_min.is_none() && cfg.precursor_mz_max.is_none() {
            return None;
        }
        Some(MzRange {
            lo: cfg.precursor_mz_min.unwrap_or(f64::NEG_INFINITY),
            hi: cfg.precursor_mz_max.unwrap_or(f64::INFINITY),
        })
    }

    fn contains(&self, mz: f64) -> bool {
        mz >= self.lo && mz <= self.hi
    }
}

/// The summed residue mass of a stripped peptide, independent of the residue order (each
/// residue's mass times its count, in a fixed residue order), so a target and its
/// composition-preserving decoy give the same number to the last bit. `None` for a residue
/// without a mass (B, J, O, U, X, Z), whose forms the m/z filter then keeps and leaves to
/// predict-frag.
fn composition_mass(peptide: &str) -> Option<f64> {
    let mut counts = [0u32; 26];
    for b in peptide.bytes() {
        let u = b.to_ascii_uppercase();
        if !u.is_ascii_uppercase() {
            return None;
        }
        counts[(u - b'A') as usize] += 1;
    }
    let mut m = 0.0;
    for (i, &n) in counts.iter().enumerate() {
        if n > 0 {
            m += residue_mass(b'A' + i as u8)? * f64::from(n);
        }
    }
    Some(m)
}

/// The precursor m/z the range is decided on: the composition mass of the pair's TARGET,
/// plus water and this form's modification masses summed in ascending order, at charge
/// `z`. Every row of a target/decoy pair (same base peptide, charge and modification
/// multiset, the key predict-frag pairs on) gets the same value, so the pair is kept or
/// dropped together even under a decoy scheme that changed the residue composition.
fn decision_mz(target_composition: f64, mod_masses: &mut [f64], z: i32) -> f64 {
    mod_masses.sort_by(f64::total_cmp);
    let neutral = WATER + target_composition + mod_masses.iter().sum::<f64>();
    mass_to_mz(neutral, z)
}

pub struct PeptidoformsParams<'a> {
    pub peptides: &'a str,
    pub out: &'a str,
    pub cfg: &'a PeptidoformsConfig,
    pub config_hash: &'a str,
}

/// The warning for a search on an imported library whose configuration sets library m/z
/// limits: they act while MuMDIA builds a library (peptidoforms, predict-frag), and an
/// imported library is used as it is. `None` when no limit is set or the library is built.
pub fn mz_limits_ignored_warning(cfg: &Config, library_input: bool) -> Option<String> {
    let set: Vec<&str> = [
        (
            "peptidoforms.precursor_mz_min",
            cfg.peptidoforms.precursor_mz_min,
        ),
        (
            "peptidoforms.precursor_mz_max",
            cfg.peptidoforms.precursor_mz_max,
        ),
        (
            "predict_frag.fragment_mz_min",
            cfg.predict_frag.fragment_mz_min,
        ),
        (
            "predict_frag.fragment_mz_max",
            cfg.predict_frag.fragment_mz_max,
        ),
    ]
    .into_iter()
    .filter_map(|(name, v)| v.map(|_| name))
    .collect();
    (library_input && !set.is_empty()).then(|| {
        format!(
            "{} only apply while MuMDIA builds a library from a FASTA; the imported library \
             is searched as it is (filter it with `mumdia sub-library` first)",
            set.join(", ")
        )
    })
}

/// Warn when the configured precursor m/z range leaves part of a run's isolation windows
/// uncovered: a library built with it has no precursor there, so whatever elutes in that
/// part of the windows cannot be identified. Reads `isolation_windows.parquet` (`lower`,
/// `upper`); no warning without a limit, or when the range covers every window.
pub fn warn_if_windows_outside_range(cfg: &PeptidoformsConfig, isolation_windows: &str) {
    let Some(range) = MzRange::of(cfg) else {
        return;
    };
    let span = TableFile::open(isolation_windows).and_then(|t| {
        let (lo, hi) = (t.f64("lower")?, t.f64("upper")?);
        Ok((
            lo.iter().copied().fold(f64::INFINITY, f64::min),
            hi.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ))
    });
    match span {
        Ok((lo, hi)) if lo.is_finite() && hi.is_finite() => {
            if lo < range.lo || hi > range.hi {
                warn!(
                    precursor_mz_min = ?cfg.precursor_mz_min,
                    precursor_mz_max = ?cfg.precursor_mz_max,
                    windows_lower = lo,
                    windows_upper = hi,
                    "the library's precursor m/z range does not cover this run's isolation \
                     windows; precursors in the uncovered part are not in a library built \
                     with it and cannot be identified"
                );
            }
        }
        Ok(_) => {}
        Err(e) => warn!(
            error = %e,
            path = isolation_windows,
            "could not read the isolation windows to check the precursor m/z range"
        ),
    }
}

pub fn run(p: PeptidoformsParams) -> Result<u64> {
    run_hashed(p).map(|w| w.rows)
}

/// [`run`], returning the output's row count and the content hash its report records, so
/// an orchestrator can record the artifact without reading and hashing it again.
pub fn run_hashed(p: PeptidoformsParams) -> Result<Written> {
    let t0 = Instant::now();
    let rules = validated_rules(p.cfg)?;
    let (fixed_rules, variable_rules) = (&rules.fixed, &rules.variable);
    let t = TableFile::open(p.peptides)?;
    let pep_id = t.u32("id")?;
    let peptide = t.str("peptide")?;
    let protein = t.str("protein")?;
    let label = t.str("label")?;
    let target_id = t.i32("target_id")?;
    // The digest's protein N-terminus flag, needed only by a protein-N-terminal rule. A
    // table without it was digested without such a rule, and reading every peptide as
    // internal would silently search no terminal form at all.
    let starts_protein: Option<Vec<bool>> = if rules.needs_protein_flag() {
        if !t.has_column("protein_n_term") {
            anyhow::bail!(
                "{}: no protein_n_term column, so it was digested without a protein \
                 N-terminal modification rule. Run digest again with this configuration \
                 (a rule with position protein_n_term makes the digest record the flag)",
                p.peptides
            );
        }
        Some(t.bool("protein_n_term")?)
    } else {
        None
    };
    let mut n_term_rows = 0u64;

    // The precursor m/z limits, when set. The decision uses each pair's target
    // composition, so the targets' composition masses are collected first, by id.
    let mz_range = MzRange::of(p.cfg);
    let target_composition: HashMap<u32, Option<f64>> = if mz_range.is_some() {
        (0..t.nrows)
            .filter(|&row| target_id[row] < 0)
            .map(|row| (pep_id[row], composition_mass(&peptide[row])))
            .collect()
    } else {
        HashMap::new()
    };
    let mod_mass: HashMap<&str, f64> = if mz_range.is_some() {
        rules
            .fixed
            .iter()
            .chain(&rules.variable)
            .map(|(_, name)| *name)
            .chain(rules.fixed_n_term.iter().map(|(_, name)| *name))
            .chain(rules.variable_n_term.iter().map(|(_, name)| *name))
            .filter_map(|name| unimod_mass(name).map(|m| (name, m)))
            .collect()
    } else {
        HashMap::new()
    };
    let mut n_out_of_range = 0u64;
    let mut n_enumerated = 0u64;

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

        // The composition the m/z decision uses: the target's, found by `target_id` for a
        // decoy (its own when the target is not in the table). `None` without a range, or
        // for a residue without a mass (the form is then kept).
        let pair_composition: Option<f64> = match mz_range {
            None => None,
            Some(_) if target_id[row] >= 0 => target_composition
                .get(&(target_id[row] as u32))
                .copied()
                .unwrap_or_else(|| composition_mass(pep)),
            Some(_) => composition_mass(pep),
        };

        let row_starts_protein = starts_protein.as_ref().is_some_and(|f| f[row]);
        // Fixed mods: apply to every matching residue, and a fixed N-terminal group to
        // every peptide its position covers. Sites are slots ([`residue_slot`]).
        let mut fixed: Vec<(usize, &str)> = Vec::new();
        if let Some((position, name)) = rules.fixed_n_term {
            if term_rule_applies(position, row_starts_protein) {
                fixed.push((N_TERM_SLOT, name));
            }
        }
        for (i, c) in pep.chars().enumerate() {
            for &(residue, name) in fixed_rules {
                if c == residue {
                    fixed.push((residue_slot(i), name));
                }
            }
        }
        // Variable-mod candidate sites, each with zero-or-one alternatives. The N-terminal
        // group is one more site, before the residues, and counts toward
        // `max_variable_mods` like any other.
        let mut var_sites: Vec<(usize, Vec<&str>)> = Vec::new();
        let n_term_alternatives: Vec<&str> = rules
            .variable_n_term
            .iter()
            .filter(|(position, _)| term_rule_applies(*position, row_starts_protein))
            .map(|(_, name)| *name)
            .collect();
        if !n_term_alternatives.is_empty() {
            var_sites.push((N_TERM_SLOT, n_term_alternatives));
        }
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
                var_sites.push((residue_slot(i), alternatives));
            }
        }

        for combo in variable_combos(&var_sites, p.cfg.max_variable_mods) {
            let mut mods = fixed.clone();
            mods.extend(combo);
            mods.sort_by_key(|(pos, _)| *pos);
            let form = proforma(pep, &mods);
            let has_n_term = mods.first().is_some_and(|(slot, _)| *slot == N_TERM_SLOT);
            let mut mod_masses: Vec<f64> = if pair_composition.is_some() {
                mods.iter()
                    .map(|(_, name)| mod_mass.get(name).copied().unwrap_or(0.0))
                    .collect()
            } else {
                Vec::new()
            };
            for z in z_lo..=z_hi {
                if let Some(range) = mz_range {
                    n_enumerated += 1;
                    if let Some(comp) = pair_composition {
                        if !range.contains(decision_mz(comp, &mut mod_masses, z)) {
                            n_out_of_range += 1;
                            continue;
                        }
                    }
                }
                id_c.push(next);
                next += 1;
                pepid_c.push(pep_id[row]);
                base_c.push(base_pep_id);
                pep_c.push(pep.clone());
                pform_c.push(form.clone());
                charge_c.push(z);
                label_c.push(label[row].clone());
                prot_c.push(protein[row].clone());
                n_term_rows += u64::from(has_n_term);
            }
        }
    }

    if let Some(range) = mz_range {
        if id_c.is_empty() && n_enumerated > 0 {
            anyhow::bail!(
                "peptidoforms: the precursor m/z range [{}, {}] excludes all {n_enumerated} \
                 charge states of these peptides (charges {}..={}); widen \
                 peptidoforms.precursor_mz_min / precursor_mz_max, the charge range or the \
                 digest length range",
                range.lo,
                range.hi,
                p.cfg.charge_min,
                p.cfg.charge_max
            );
        }
        let share = if n_enumerated > 0 {
            n_out_of_range as f64 / n_enumerated as f64
        } else {
            0.0
        };
        info!(
            precursor_mz_min = ?p.cfg.precursor_mz_min,
            precursor_mz_max = ?p.cfg.precursor_mz_max,
            out_of_range = n_out_of_range,
            enumerated = n_enumerated,
            share_dropped = %format!("{:.1}%", share * 100.0),
            "peptidoforms: charge states outside the precursor m/z range were not emitted"
        );
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
    let show = |m: &ResidueMod| {
        if m.position.is_anywhere() {
            format!("{}:{}", m.residue, m.name)
        } else {
            format!("{}:{}", m.position.name(), m.name)
        }
    };
    let mut stats = std::collections::BTreeMap::new();
    if rules.has_n_term() {
        // Recorded only with a terminal rule, so every other report is what it was.
        stats.insert("n_term_modified_rows".to_string(), json!(n_term_rows));
    }
    let mut params = json!({
        "fixed_mods": p.cfg.fixed_mods.iter().map(show).collect::<Vec<_>>(),
        "variable_mods": p.cfg.variable_mods.iter().map(show).collect::<Vec<_>>(),
        "max_variable_mods": p.cfg.max_variable_mods,
        "charges": format!("{}..{}", p.cfg.charge_min, p.cfg.charge_max),
    });
    if mz_range.is_some() {
        // Recorded only with a limit, so every other report is what it was.
        params["precursor_mz_min"] = json!(p.cfg.precursor_mz_min);
        params["precursor_mz_max"] = json!(p.cfg.precursor_mz_max);
        stats.insert("out_of_mz_range_rows".to_string(), json!(n_out_of_range));
    }
    let report = ArtifactReport {
        logical_name: artifact::PEPTIDOFORMS.0.to_string(),
        schema_name: artifact::PEPTIDOFORMS.0.to_string(),
        schema_version: artifact::PEPTIDOFORMS.1,
        stage: "peptidoforms".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params,
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    };
    report.write_for(p.out)?;

    info!(rows, elapsed_ms = elapsed, "peptidoforms: done");
    Ok(report.written())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proforma_places_mods() {
        let s = proforma(
            "PEPCM",
            &[
                (residue_slot(3), "Carbamidomethyl"),
                (residue_slot(4), "Oxidation"),
            ],
        );
        assert_eq!(s, "PEPC[Carbamidomethyl]M[Oxidation]");
    }

    /// The N-terminal group is written before the first residue with a dash, and a
    /// modification on that first residue still follows it.
    #[test]
    fn proforma_writes_the_n_terminal_group() {
        assert_eq!(proforma("SEK", &[(N_TERM_SLOT, "Acetyl")]), "[Acetyl]-SEK");
        assert_eq!(
            proforma(
                "SEK",
                &[(N_TERM_SLOT, "Acetyl"), (residue_slot(0), "Phospho")]
            ),
            "[Acetyl]-S[Phospho]EK"
        );
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
        let rules = validated_rules(&cfg).unwrap();
        assert_eq!(rules.fixed, vec![('C', "Carbamidomethyl")]);
        assert_eq!(rules.variable, vec![('M', "Oxidation")]);
        assert!(!rules.has_n_term());
        assert!(!cfg.has_protein_n_term_rule());

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
    fn homocysteine_thiolactone_on_lysine_is_a_known_variable_modification() {
        // Issue #234: N-homocysteinylation (UniMod 1270) offered as a variable K site.
        let mut cfg = PeptidoformsConfig::default();
        cfg.variable_mods.push(ResidueMod {
            residue: 'K',
            name: "HCysThiolactone".to_string(),
            position: ModPosition::Anywhere,
        });
        let rules = validated_rules(&cfg).unwrap();
        assert_eq!(
            rules.variable,
            vec![('M', "Oxidation"), ('K', "HCysThiolactone")]
        );
        let sites = vec![(residue_slot(4), vec!["HCysThiolactone"])];
        let forms: Vec<String> = variable_combos(&sites, 1)
            .iter()
            .map(|c| proforma("PEPTKIDER", c))
            .collect();
        assert_eq!(forms, vec!["PEPTKIDER", "PEPTK[HCysThiolactone]IDER"]);
        let delta = mumdia_core::mass::parse_peptidoform(&forms[1])
            .unwrap()
            .neutral_mass()
            - mumdia_core::mass::parse_peptidoform(&forms[0])
                .unwrap()
                .neutral_mass();
        assert!((delta - 117.024_835).abs() < 1e-5, "delta = {delta}");
    }

    #[test]
    fn skip_policy_removes_unknown_modification() {
        let mut cfg = PeptidoformsConfig::default();
        cfg.variable_mods.push(ResidueMod {
            residue: 'M',
            name: "DefinitelyNotAUnimodName".to_string(),
            position: ModPosition::Anywhere,
        });
        cfg.unknown_modification = UnknownModPolicy::Skip;
        let rules = validated_rules(&cfg).unwrap();
        assert_eq!(rules.variable, vec![('M', "Oxidation")]);
    }

    #[test]
    fn fixed_stacking_and_fixed_variable_overlap_are_rejected() {
        let mut stacked = PeptidoformsConfig::default();
        stacked.fixed_mods.push(ResidueMod {
            residue: 'C',
            name: "Oxidation".to_string(),
            position: ModPosition::Anywhere,
        });
        let err = validated_rules(&stacked).unwrap_err().to_string();
        assert!(err.contains("multiple fixed modifications"));

        let mut overlap = PeptidoformsConfig::default();
        overlap.variable_mods.push(ResidueMod {
            residue: 'C',
            name: "Oxidation".to_string(),
            position: ModPosition::Anywhere,
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
            position: ModPosition::Anywhere,
        });
        assert!(validated_rules(&wildcard)
            .unwrap_err()
            .to_string()
            .contains("unsupported '*' residue"));
    }

    fn term(name: &str, residue: char, position: ModPosition) -> ResidueMod {
        ResidueMod {
            residue,
            name: name.to_string(),
            position,
        }
    }

    /// Acetyl at the protein N-terminus as a variable rule on top of the defaults.
    fn acetyl_cfg(max_variable_mods: usize) -> PeptidoformsConfig {
        let mut cfg = PeptidoformsConfig {
            max_variable_mods,
            charge_min: 2,
            charge_max: 2,
            ..PeptidoformsConfig::default()
        };
        cfg.variable_mods
            .push(term("Acetyl", '*', ModPosition::ProteinNTerm));
        cfg
    }

    #[test]
    fn terminal_rule_conflicts_are_refused() {
        let msg = |cfg: &PeptidoformsConfig| validated_rules(cfg).unwrap_err().to_string();
        // A fixed and a variable rule on the one N-terminus.
        let mut both = acetyl_cfg(1);
        both.fixed_mods
            .push(term("Propionyl", '*', ModPosition::PeptideNTerm));
        assert!(msg(&both).contains("cannot be combined"), "{}", msg(&both));
        // Two fixed N-terminal groups.
        let mut two = PeptidoformsConfig::default();
        two.fixed_mods
            .push(term("Acetyl", '*', ModPosition::ProteinNTerm));
        two.fixed_mods
            .push(term("Propionyl", '*', ModPosition::PeptideNTerm));
        assert!(msg(&two).contains("two fixed N-terminal"), "{}", msg(&two));
        // A terminal rule restricted to one residue.
        let mut restricted = PeptidoformsConfig::default();
        restricted
            .variable_mods
            .push(term("Acetyl", 'S', ModPosition::ProteinNTerm));
        assert!(
            msg(&restricted).contains("takes residue '*'"),
            "{}",
            msg(&restricted)
        );
        // The C-terminal positions.
        for position in [ModPosition::PeptideCTerm, ModPosition::ProteinCTerm] {
            let mut c_term = PeptidoformsConfig::default();
            c_term.variable_mods.push(term("Amidated", '*', position));
            assert!(msg(&c_term).contains("C-terminal"), "{}", msg(&c_term));
        }
        // The same checks at configuration load.
        let json = r#"{"peptidoforms":{"variable_mods":[
            {"residue":"*","name":"Acetyl","position":"protein_c_term"}]}}"#;
        assert!(mumdia_core::config::Config::from_json(json).is_err());
        let json = r#"{"peptidoforms":{"variable_mods":[
            {"residue":"*","name":"Acetyl","position":"protein_n_term"}]}}"#;
        let cfg = mumdia_core::config::Config::from_json(json).unwrap();
        assert!(cfg.peptidoforms.has_protein_n_term_rule());
        // A variable residue Acetyl on S and an N-terminal one are different sites.
        let mut residue_and_term = acetyl_cfg(2);
        residue_and_term
            .variable_mods
            .push(term("Phospho", 'S', ModPosition::Anywhere));
        let rules = validated_rules(&residue_and_term).unwrap();
        assert_eq!(
            rules.variable_n_term,
            vec![(ModPosition::ProteinNTerm, "Acetyl")]
        );
        assert_eq!(rules.variable, vec![('M', "Oxidation"), ('S', "Phospho")]);
    }

    /// Expand a hand-written peptides table (`peptide`, `label`, flag) and return
    /// (peptidoform, label) per row.
    fn expand(
        tag: &str,
        rows: &[(&str, &str, bool)],
        cfg: &PeptidoformsConfig,
        with_flag: bool,
    ) -> Result<Vec<(String, String)>> {
        expand_charged(tag, rows, cfg, with_flag)
            .map(|v| v.into_iter().map(|(f, l, _, _)| (f, l)).collect())
    }

    /// [`expand`], returning (peptidoform, label, charge, base_peptide_id) per row.
    fn expand_charged(
        tag: &str,
        rows: &[(&str, &str, bool)],
        cfg: &PeptidoformsConfig,
        with_flag: bool,
    ) -> Result<Vec<(String, String, i32, u32)>> {
        let dir = std::env::temp_dir().join(format!("mumdia_pfx_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = |n: &str| dir.join(n).to_string_lossy().into_owned();
        let n = rows.len();
        let mut target_id = Vec::with_capacity(n);
        let mut last_target = -1i32;
        for (i, r) in rows.iter().enumerate() {
            if r.1 == "target" {
                last_target = i as i32;
                target_id.push(-1);
            } else {
                target_id.push(last_target);
            }
        }
        let mut cols = vec![
            Col::U32("id".into(), (0..n as u32).collect()),
            Col::Str(
                "peptide".into(),
                rows.iter().map(|r| r.0.to_string()).collect(),
            ),
            Col::Str(
                "protein".into(),
                rows.iter().map(|_| "P1".to_string()).collect(),
            ),
            Col::I32("start".into(), vec![0; n]),
            Col::I32("end".into(), vec![0; n]),
            Col::Str(
                "label".into(),
                rows.iter().map(|r| r.1.to_string()).collect(),
            ),
            Col::I32("target_id".into(), target_id),
            Col::Str("decoy_strategy".into(), vec!["reverse".to_string(); n]),
        ];
        if with_flag {
            cols.push(Col::Bool(
                "protein_n_term".into(),
                rows.iter().map(|r| r.2).collect(),
            ));
        }
        write_table(&path("peptides.parquet"), cols).unwrap();
        let out = path("pf.parquet");
        let result = run(PeptidoformsParams {
            peptides: &path("peptides.parquet"),
            out: &out,
            cfg,
            config_hash: "h",
        })
        .map(|_| {
            let t = TableFile::open(&out).unwrap();
            let (pf, label) = (t.str("peptidoform").unwrap(), t.str("label").unwrap());
            let (z, base) = (t.i32("charge").unwrap(), t.u32("base_peptide_id").unwrap());
            pf.into_iter()
                .zip(label)
                .zip(z.into_iter().zip(base))
                .map(|((f, l), (z, b))| (f, l, z, b))
                .collect()
        });
        let _ = std::fs::remove_dir_all(&dir);
        result
    }

    /// One eligible target (and its reversed decoy, which inherits the flag) and one
    /// internal target. Under max 1 the acetylated form and the oxidised form are
    /// alternatives, never combined; under max 2 they combine. Target and decoy expand
    /// into the same number of forms in the same order of modification content.
    #[test]
    fn protein_n_term_acetyl_expands_eligible_peptides_only() {
        let rows = [
            ("SMEQK", "target", true),
            ("QEMSK", "decoy", true),
            ("LMNPK", "target", false),
            ("PNMLK", "decoy", false),
        ];
        let forms = expand("max1", &rows, &acetyl_cfg(1), true).unwrap();
        let of = |label: &str, prefix: &str| -> Vec<String> {
            forms
                .iter()
                .filter(|(f, l)| {
                    l == label
                        && f.replace("[Acetyl]-", "")
                            .replace("[Oxidation]", "")
                            .starts_with(prefix)
                })
                .map(|(f, _)| f.clone())
                .collect()
        };
        assert_eq!(
            of("target", "SMEQK"),
            ["SMEQK", "[Acetyl]-SMEQK", "SM[Oxidation]EQK"]
        );
        assert_eq!(
            of("decoy", "QEMSK"),
            ["QEMSK", "[Acetyl]-QEMSK", "QEM[Oxidation]SK"]
        );
        assert_eq!(of("target", "LMNPK"), ["LMNPK", "LM[Oxidation]NPK"]);
        assert_eq!(of("decoy", "PNMLK"), ["PNMLK", "PNM[Oxidation]LK"]);
        let acetyl = |label: &str| {
            forms
                .iter()
                .filter(|(f, l)| l == label && f.starts_with("[Acetyl]-"))
                .count()
        };
        assert_eq!(acetyl("target"), 1);
        assert_eq!(acetyl("decoy"), 1);

        let forms2 = expand("max2", &rows, &acetyl_cfg(2), true).unwrap();
        let target2: Vec<&str> = forms2
            .iter()
            .filter(|(f, l)| l == "target" && f.contains("SM"))
            .map(|(f, _)| f.as_str())
            .collect();
        assert_eq!(
            target2,
            [
                "SMEQK",
                "[Acetyl]-SMEQK",
                "SM[Oxidation]EQK",
                "[Acetyl]-SM[Oxidation]EQK"
            ]
        );
        let n = |label: &str| forms2.iter().filter(|(_, l)| l == label).count();
        assert_eq!(n("target"), n("decoy"));
        // The acetylated forms weigh one acetyl more, on the precursor and on every b ion.
        let plain = mumdia_core::mass::parse_peptidoform("SMEQK").unwrap();
        let ac = mumdia_core::mass::parse_peptidoform("[Acetyl]-SMEQK").unwrap();
        assert!((ac.neutral_mass() - plain.neutral_mass() - 42.010_565).abs() < 1e-5);
        let (fp, fa) = (plain.fragments(&[1]), ac.fragments(&[1]));
        for (a, b) in fa.iter().zip(&fp) {
            let shift = if a.ion_type == mumdia_core::mass::IonType::B {
                42.010_565
            } else {
                0.0
            };
            assert!(
                (a.mz - b.mz - shift).abs() < 1e-5,
                "{} {}",
                a.name(),
                b.name()
            );
        }
    }

    /// A peptide-N-terminal rule applies to every peptide and needs no digest flag; a
    /// fixed one is placed on every form.
    #[test]
    fn peptide_n_term_rules_apply_to_every_peptide() {
        let rows = [("SMEQK", "target", false), ("QEMSK", "decoy", false)];
        let mut fixed = PeptidoformsConfig::default();
        fixed
            .fixed_mods
            .push(term("Propionyl", '*', ModPosition::PeptideNTerm));
        let forms = expand("pepfixed", &rows, &fixed, false).unwrap();
        assert!(
            forms.iter().all(|(f, _)| f.starts_with("[Propionyl]-")),
            "{forms:?}"
        );
        assert_eq!(forms.len(), 2 * 2 * 2);
    }

    #[test]
    fn a_protein_n_term_rule_needs_the_digest_flag() {
        let rows = [("SMEQK", "target", true), ("QEMSK", "decoy", true)];
        let err = expand("noflag", &rows, &acetyl_cfg(1), false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no protein_n_term column"), "{err}");
    }

    /// The default configuration writes no `position`, so the canonical configuration,
    /// the configuration hash and the library-cache key of every existing configuration
    /// are unchanged; a terminal rule writes its position.
    #[test]
    fn the_default_position_stays_out_of_the_serialised_configuration() {
        let json = serde_json::to_string(&PeptidoformsConfig::default()).unwrap();
        assert!(!json.contains("position"), "{json}");
        assert!(
            json.contains(r#"{"residue":"C","name":"Carbamidomethyl"}"#),
            "{json}"
        );
        let json = serde_json::to_string(&acetyl_cfg(1)).unwrap();
        assert!(
            json.contains(r#"{"residue":"*","name":"Acetyl","position":"protein_n_term"}"#),
            "{json}"
        );
    }

    /// The precursor m/z of one emitted row, as predict-frag computes it.
    fn mz_of(form: &str, z: i32) -> f64 {
        mumdia_core::mass::parse_peptidoform(form)
            .unwrap()
            .precursor_mz(z)
    }

    /// With limits, every emitted charge state lies inside them (inclusive), the ones
    /// outside are not emitted, and a target and its decoy keep the same charge states
    /// and forms.
    #[test]
    fn precursor_mz_limits_drop_charge_states_outside_and_keep_pairs_together() {
        let rows = [
            ("SAMPLEMK", "target", false),
            ("EMLPMASK", "decoy", false),
            ("PEPTIDEWWWWR", "target", false),
            ("WWWWEDITPEPR", "decoy", false),
        ];
        let mut cfg = PeptidoformsConfig {
            charge_min: 1,
            charge_max: 4,
            ..PeptidoformsConfig::default()
        };
        let all = expand_charged("mzall", &rows, &cfg, false).unwrap();
        // Bounds chosen from the unfiltered table: the middle of the m/z values.
        let mut mzs: Vec<f64> = all.iter().map(|(f, _, z, _)| mz_of(f, *z)).collect();
        mzs.sort_by(f64::total_cmp);
        let (lo, hi) = (mzs[mzs.len() / 4], mzs[3 * mzs.len() / 4]);
        cfg.precursor_mz_min = Some(lo - 1e-6);
        cfg.precursor_mz_max = Some(hi + 1e-6);
        let kept = expand_charged("mzlim", &rows, &cfg, false).unwrap();
        assert!(!kept.is_empty() && kept.len() < all.len());
        for (f, _, z, _) in &kept {
            let mz = mz_of(f, *z);
            assert!(mz >= lo - 1e-6 && mz <= hi + 1e-6, "{f} {z}+ at {mz}");
        }
        // Exactly the rows of the unfiltered table that lie inside, in the same order.
        let inside: Vec<_> = all
            .iter()
            .filter(|(f, _, z, _)| {
                let mz = mz_of(f, *z);
                mz >= lo - 1e-6 && mz <= hi + 1e-6
            })
            .cloned()
            .collect();
        assert_eq!(kept, inside);
        // Pairs: per base peptide and charge, as many decoy rows as target rows.
        let mut count: BTreeMap<(u32, i32, String), i64> = BTreeMap::new();
        for (_, l, z, b) in &kept {
            *count.entry((*b, *z, String::new())).or_default() +=
                if l == "target" { 1 } else { -1 };
        }
        assert!(count.values().all(|&c| c == 0), "{count:?}");
    }

    /// The bounds are inclusive, an open end is unbounded, and a range that excludes every
    /// charge state is an error rather than an empty library.
    #[test]
    fn precursor_mz_limits_are_inclusive_and_an_empty_result_is_refused() {
        let rows = [("PEPTIDEK", "target", false), ("EDITPEPK", "decoy", false)];
        let base = PeptidoformsConfig {
            variable_mods: vec![],
            ..PeptidoformsConfig::default()
        };
        let mz2 = mz_of("PEPTIDEK", 2);
        let mz3 = mz_of("PEPTIDEK", 3);
        let at = |lo: Option<f64>, hi: Option<f64>, tag: &str| {
            let cfg = PeptidoformsConfig {
                precursor_mz_min: lo,
                precursor_mz_max: hi,
                ..base.clone()
            };
            expand_charged(tag, &rows, &cfg, false)
        };
        let charges = |v: Vec<(String, String, i32, u32)>| -> Vec<i32> {
            v.into_iter().map(|(_, _, z, _)| z).collect()
        };
        // The decision value differs from predict-frag's in the last bits at most.
        assert_eq!(
            charges(at(Some(mz2 - 1e-9), None, "lo").unwrap()),
            vec![2, 2]
        );
        assert_eq!(
            charges(at(None, Some(mz3 + 1e-9), "hi").unwrap()),
            vec![3, 3]
        );
        assert_eq!(
            charges(at(Some(mz3 - 1e-9), Some(mz2 + 1e-9), "both").unwrap()),
            vec![2, 3, 2, 3]
        );
        let err = at(Some(mz2 + 1.0), None, "none").unwrap_err().to_string();
        assert!(err.contains("excludes all"), "{err}");
        // Without limits the table is the one the stage always wrote.
        assert_eq!(
            charges(at(None, None, "default").unwrap()),
            vec![2, 3, 2, 3]
        );
    }

    /// The decision is made on the target's composition, so a decoy whose residues were
    /// changed (a scheme that does not preserve the composition) still follows its target.
    #[test]
    fn the_precursor_mz_decision_follows_the_target() {
        // The "decoy" here is far heavier than its target; it must still be kept exactly
        // when its target is.
        let rows = [("PEPTIDEK", "target", false), ("WWWWWWWK", "decoy", false)];
        let mz2 = mz_of("PEPTIDEK", 2);
        let cfg = PeptidoformsConfig {
            variable_mods: vec![],
            precursor_mz_min: Some(mz2 - 0.01),
            precursor_mz_max: Some(mz2 + 0.01),
            ..PeptidoformsConfig::default()
        };
        let kept = expand_charged("follow", &rows, &cfg, false).unwrap();
        assert_eq!(
            kept.iter()
                .map(|(f, l, z, _)| (f.as_str(), l.as_str(), *z))
                .collect::<Vec<_>>(),
            vec![("PEPTIDEK", "target", 2), ("WWWWWWWK", "decoy", 2)]
        );
    }

    #[test]
    fn composition_mass_ignores_order_and_refuses_unknown_residues() {
        assert_eq!(composition_mass("PEPTIDEK"), composition_mass("KEDITPEP"));
        assert_eq!(composition_mass("PEPTIDEK"), composition_mass("peptidek"));
        assert_eq!(composition_mass("PEPXIDEK"), None);
        assert_eq!(composition_mass("PEP1"), None);
    }
}
