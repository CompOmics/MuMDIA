//! Stage A `mumdia digest`: enzymatic (tryptic) or non-specific enumeration + decoy
//! pairing (docs/05_digest_peptidoforms.md). Experiment-wide, computed once. MVP uses a
//! documented, clean-room decoy scheme (reverse or seeded scramble); the DIA-NN
//! terminal shift map is a later license-checked addition
//! (docs/14_build_test_deploy_gotchas.md).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{Context, Result};
use mumdia_core::cleavage::CleavageRule;
use mumdia_core::config::{DecoyStrategy, DigestConfig};
use mumdia_core::constants::is_standard_residue;
use mumdia_core::schema::artifact;
use mumdia_io::report::{ArtifactReport, Written};
use mumdia_io::table::{write_table, Col};
use serde_json::json;
use tracing::{info, warn};

/// A bounded retry count keeps decoy construction deterministic and prevents a
/// pathological low-complexity peptide from looping forever. If no collision-free
/// permutation exists, the target/decoy pair is dropped together so the emitted
/// library retains a one-to-one target-decoy population.
const MAX_DECOY_ATTEMPTS: usize = 64;

/// Parse a FASTA file into (accession, sequence) pairs.
pub fn read_fasta(path: &str) -> Result<Vec<(String, String)>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading fasta {path}"))?;
    let mut out = Vec::new();
    let mut acc = String::new();
    let mut seq = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix('>') {
            if !acc.is_empty() {
                out.push((acc.clone(), seq.clone()));
            }
            acc = rest.split_whitespace().next().unwrap_or("").to_string();
            seq.clear();
        } else {
            // FASTA sequence case is not biologically meaningful. Normalizing here
            // avoids silently discarding otherwise valid lowercase proteins in
            // `digest_protein`'s standard-residue check.
            seq.push_str(&line.trim().to_ascii_uppercase());
        }
    }
    if !acc.is_empty() {
        out.push((acc, seq));
    }
    Ok(out)
}

/// Enumerate the peptides of one protein under `cfg`'s rule; see [`digest_with`].
#[cfg(test)]
fn digest_protein(seq: &[u8], cfg: &DigestConfig) -> Vec<(usize, usize, String)> {
    digest_with(seq, cfg, &cfg.rule().unwrap())
}

/// Enumerate the peptides of one protein: the spans between cleavage sites of `rule` with
/// up to `missed_cleavages` missed cleavages, or every peptide of an allowed length when
/// the rule cuts every bond; length bounds respected, ambiguous residues dropped.
fn digest_with(seq: &[u8], cfg: &DigestConfig, rule: &CleavageRule) -> Vec<(usize, usize, String)> {
    if rule.is_non_specific() {
        return digest_non_specific(seq, cfg);
    }
    let sites = rule.sites(seq);
    let mut out = Vec::new();
    for i in 0..sites.len().saturating_sub(1) {
        for j in (i + 1)..sites.len() {
            let missed = j - i - 1;
            if missed > cfg.missed_cleavages as usize {
                break;
            }
            let (start, end) = (sites[i], sites[j]);
            let sub = &seq[start..end];
            let len = end - start;
            // The ordinary form and the Met-excised form are judged independently. The
            // ordinary form used to be rejected with `continue`, which also skipped the
            // excision below, so an N-terminal peptide of `max_len + 1` residues produced
            // nothing although its excised form was in range (docs/29 #9).
            if len >= cfg.min_len
                && len <= cfg.max_len
                && sub.iter().all(|&c| is_standard_residue(c))
            {
                out.push((start, end, String::from_utf8_lossy(sub).to_string()));
            }

            // N-terminal methionine excision: for a peptide anchored at the
            // protein N-terminus whose first residue is the initiator Met, also
            // emit the Met-removed form (start shifted to 1). The excised peptide
            // is checked against the length bounds and standard-residue rule on its
            // own. This mirrors DIA-NN's `--met-excision`; without it the search
            // database cannot contain these (biologically dominant) peptides.
            if cfg.n_term_met_excision && start == 0 && seq.first() == Some(&b'M') {
                let ex = &seq[1..end];
                let ex_len = ex.len();
                if ex_len >= cfg.min_len.max(1)
                    && ex_len <= cfg.max_len
                    && ex.iter().all(|&c| is_standard_residue(c))
                {
                    out.push((1, end, String::from_utf8_lossy(ex).to_string()));
                }
            }
        }
    }
    out
}

/// Every peptide of `min_len..=max_len` residues, by start then length, that contains
/// only standard residues. `missed_cleavages` does not apply, and N-terminal Met excision
/// adds nothing because the excised form is the peptide starting at position 1. A run of
/// standard residues is found once per start, so the scan is O(residues x lengths).
fn digest_non_specific(seq: &[u8], cfg: &DigestConfig) -> Vec<(usize, usize, String)> {
    let n = seq.len();
    // next_bad[i]: the first position >= i holding a non-standard residue (n if none).
    let mut next_bad = vec![n; n + 1];
    for i in (0..n).rev() {
        next_bad[i] = if is_standard_residue(seq[i]) {
            next_bad[i + 1]
        } else {
            i
        };
    }
    let mut out = Vec::new();
    for start in 0..n {
        let stop = next_bad[start].min(start + cfg.max_len).min(n);
        for end in (start + cfg.min_len)..=stop {
            out.push((
                start,
                end,
                String::from_utf8_lossy(&seq[start..end]).to_string(),
            ));
        }
    }
    out
}

/// Peptides (with repeats) a non-specific digest of these proteins enumerates before
/// de-duplication: the count to warn about before holding them all in memory.
fn non_specific_upper_bound(proteins: &[(String, String)], cfg: &DigestConfig) -> u64 {
    let mut total = 0u64;
    for (_, seq) in proteins {
        let n = seq.len();
        for len in cfg.min_len..=cfg.max_len.min(n) {
            total += (n - len + 1) as u64;
        }
    }
    total
}

/// Above this many enumerated peptides a non-specific digest warns: every distinct
/// peptide, its decoy and the collision sets are held in memory (roughly 100 bytes each).
const NON_SPECIFIC_WARN_PEPTIDES: u64 = 300_000_000;

/// Make a decoy sequence from a target (docs/05_digest_peptidoforms.md). Reverse
/// keeps the C-terminal residue fixed; scramble seeded-shuffles the interior.
fn make_decoy(pep: &str, strategy: DecoyStrategy, seed: u64) -> Option<String> {
    let b = pep.as_bytes();
    let n = b.len();
    if n < 3 {
        return None;
    }
    match strategy {
        DecoyStrategy::Reverse => {
            let mut interior: Vec<u8> = b[..n - 1].to_vec();
            interior.reverse();
            interior.push(b[n - 1]);
            Some(String::from_utf8(interior).unwrap())
        }
        DecoyStrategy::Scramble => {
            // Deterministic Fisher-Yates with a splitmix64 PRNG seeded per peptide.
            let mut interior: Vec<u8> = b[..n - 1].to_vec();
            let mut state = seed ^ fnv1a(pep);
            for i in (1..interior.len()).rev() {
                state = splitmix64(&mut state);
                let j = (state % (i as u64 + 1)) as usize;
                interior.swap(i, j);
            }
            interior.push(b[n - 1]);
            Some(String::from_utf8(interior).unwrap())
        }
        DecoyStrategy::DiannShift => None, // realized at predict-frag, not here
        DecoyStrategy::None => None,
    }
}

/// A sequence as the instrument sees it. I and L are isobaric (same residue mass, so
/// the same precursor and fragment m/z), so two sequences that differ only by I/L are
/// one peptide: the reversal of IEVNVELR, LEVNVEIR, is that target to the instrument.
fn il_key(sequence: &str) -> String {
    sequence.replace('I', "L")
}

/// Construct a deterministic decoy that is different from its paired target,
/// does not collide with any target peptide, and is unique among emitted decoys,
/// all compared up to I/L ([`il_key`]). `targets` and `used_decoys` hold I/L keys.
///
/// The configured transform is tried first. A collision is retried with
/// independently seeded interior scrambles while preserving the C terminus.
/// Returns the sequence and number of collision retries.
fn collision_safe_decoy(
    pep: &str,
    strategy: DecoyStrategy,
    seed: u64,
    targets: &HashSet<String>,
    used_decoys: &HashSet<String>,
) -> Option<(String, usize)> {
    for attempt in 0..MAX_DECOY_ATTEMPTS {
        let attempt_seed = seed ^ (attempt as u64).wrapping_mul(0xD1B5_4A32_D192_ED03) ^ fnv1a(pep);
        let transform = if attempt == 0 {
            strategy
        } else {
            DecoyStrategy::Scramble
        };
        let decoy = make_decoy(pep, transform, attempt_seed)?;
        let key = il_key(&decoy);
        if key != il_key(pep) && !targets.contains(&key) && !used_decoys.contains(&key) {
            return Some((decoy, attempt));
        }
    }
    None
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fnv1a(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

pub struct DigestParams<'a> {
    pub fasta: &'a str,
    pub out: &'a str,
    pub cfg: &'a DigestConfig,
    pub rng_seed: u64,
    pub config_hash: &'a str,
    /// Record which peptides start a protein, as the Boolean column `protein_n_term`
    /// (peptides v2). Set when a modification rule is placed at the protein N-terminus
    /// ([`mumdia_core::config::PeptidoformsConfig::has_protein_n_term_rule`]); without it
    /// the table is v1, byte for byte.
    pub protein_n_term: bool,
}

/// Whether a peptide that starts at `start` in `seq` starts the protein: at position 0, or
/// at position 1 as the initiator-Met-excised form when `digest.n_term_met_excision` is on.
/// The peptide at position 1 of a protein that does not begin with `M` (one following an
/// N-terminal K, say) is an ordinary cleavage product and does not.
fn starts_protein(seq: &[u8], start: usize, cfg: &DigestConfig) -> bool {
    start == 0 || (start == 1 && cfg.n_term_met_excision && seq.first() == Some(&b'M'))
}

pub fn run(p: DigestParams) -> Result<u64> {
    run_hashed(p).map(|w| w.rows)
}

/// [`run`], returning the output's row count and the content hash its report records, so
/// an orchestrator can record the artifact without reading and hashing it again.
pub fn run_hashed(p: DigestParams) -> Result<Written> {
    let t0 = Instant::now();
    let proteins = read_fasta(p.fasta)?;
    info!(proteins = proteins.len(), "digest: read fasta");
    let rule = p.cfg.rule().map_err(anyhow::Error::msg)?;
    if rule.is_non_specific() {
        let bound = non_specific_upper_bound(&proteins, p.cfg);
        info!(
            enumerated_upper_bound = bound,
            min_len = p.cfg.min_len,
            max_len = p.cfg.max_len,
            "digest: non-specific cleavage (every peptide bond; digest.missed_cleavages and              N-terminal Met excision do not apply)"
        );
        if bound > NON_SPECIFIC_WARN_PEPTIDES {
            warn!(
                enumerated_upper_bound = bound,
                "digest: a non-specific digest of this FASTA and length range enumerates more                  than {NON_SPECIFIC_WARN_PEPTIDES} peptides before de-duplication; memory and                  the predictors scale with it. Narrow digest.min_len..max_len or the FASTA if                  that is more than intended"
            );
        }
    }

    // Dedup target peptides by sequence, aggregating protein accessions.
    let mut pep_proteins: HashMap<String, Vec<String>> = HashMap::new();
    let mut pep_pos: HashMap<String, (i32, i32)> = HashMap::new();
    // Peptides that start a protein in at least one of the proteins they occur in. The
    // position kept in `pep_pos` is the first occurrence's only, so the flag is an OR over
    // every occurrence rather than a property of that one.
    let mut starts_a_protein: HashSet<String> = HashSet::new();
    let mut order: Vec<String> = Vec::new();
    for (acc, seq) in &proteins {
        let bytes = seq.as_bytes();
        for (start, end, pep) in digest_with(bytes, p.cfg, &rule) {
            if p.protein_n_term
                && starts_protein(bytes, start, p.cfg)
                && !starts_a_protein.contains(&pep)
            {
                starts_a_protein.insert(pep.clone());
            }
            // Look up with the borrowed key so the common already-seen case does
            // not clone `pep`; clone only on the first (Vacant) encounter. The last
            // use moves `pep` into `order`, keeping insertion order unchanged.
            if let Some(v) = pep_proteins.get_mut(&pep) {
                if !v.contains(acc) {
                    v.push(acc.clone());
                }
            } else {
                pep_proteins.insert(pep.clone(), vec![acc.clone()]);
                pep_pos.insert(pep.clone(), (start as i32, end as i32));
                order.push(pep);
            }
        }
    }

    let (mut id_c, mut pep_c, mut prot_c, mut start_c, mut end_c) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut label_c, mut target_c, mut strat_c) = (Vec::new(), Vec::new(), Vec::new());
    let mut nterm_c: Vec<bool> = Vec::new();

    let target_sequences: HashSet<String> = order.iter().map(|s| il_key(s)).collect();
    let mut used_decoys: HashSet<String> = HashSet::new();
    let mut decoy_collision_retries = 0usize;
    let mut dropped_pairs = 0usize;
    let mut next_id: u32 = 0;
    for pep in &order {
        // Resolve the decoy before writing the target. When a low-complexity
        // sequence has no collision-free permutation, dropping both rows keeps
        // the emitted target/decoy population paired instead of biasing FDR.
        let decoy = if matches!(
            p.cfg.decoy.strategy,
            DecoyStrategy::Reverse | DecoyStrategy::Scramble
        ) {
            match collision_safe_decoy(
                pep,
                p.cfg.decoy.strategy,
                p.rng_seed,
                &target_sequences,
                &used_decoys,
            ) {
                Some((decoy, retries)) => {
                    decoy_collision_retries += retries;
                    used_decoys.insert(il_key(&decoy));
                    Some(decoy)
                }
                None => {
                    dropped_pairs += 1;
                    continue;
                }
            }
        } else {
            None
        };

        let tid = next_id;
        next_id += 1;
        let proteins_joined = pep_proteins[pep].join(";");
        let (st, en) = pep_pos[pep];
        id_c.push(tid);
        pep_c.push(pep.clone());
        prot_c.push(proteins_joined.clone());
        start_c.push(st);
        end_c.push(en);
        label_c.push("target".to_string());
        target_c.push(-1i32);
        strat_c.push(format!("{:?}", p.cfg.decoy.strategy).to_lowercase());
        let starts = starts_a_protein.contains(pep);
        nterm_c.push(starts);

        // Materialize sequence-rewrite decoys here (docs/05_digest_peptidoforms.md).
        if let Some(dec) = decoy {
            let did = next_id;
            next_id += 1;
            id_c.push(did);
            pep_c.push(dec);
            prot_c.push(format!("DECOY_{proteins_joined}"));
            start_c.push(st);
            end_c.push(en);
            label_c.push("decoy".to_string());
            target_c.push(tid as i32);
            strat_c.push(format!("{:?}", p.cfg.decoy.strategy).to_lowercase());
            // A decoy takes its target's eligibility, never its own sequence's: the pair
            // then expands into the same terminal forms and stays exchangeable.
            nterm_c.push(starts);
        }
    }

    let n_targets = label_c.iter().filter(|l| *l == "target").count();
    let n_decoys = label_c.len() - n_targets;
    if dropped_pairs > 0 {
        warn!(
            dropped_pairs,
            max_attempts = MAX_DECOY_ATTEMPTS,
            "digest: dropped target/decoy pairs with no collision-free decoy permutation"
        );
    }
    let n_protein_n_term_targets = nterm_c
        .iter()
        .zip(&label_c)
        .filter(|(f, l)| **f && *l == "target")
        .count();
    let mut columns = vec![
        Col::U32("id".into(), id_c),
        Col::Str("peptide".into(), pep_c),
        Col::Str("protein".into(), prot_c),
        Col::I32("start".into(), start_c),
        Col::I32("end".into(), end_c),
        Col::Str("label".into(), label_c),
        Col::I32("target_id".into(), target_c),
        Col::Str("decoy_strategy".into(), strat_c),
    ];
    if p.protein_n_term {
        columns.push(Col::Bool("protein_n_term".into(), nterm_c));
    }
    let schema = artifact::peptides(p.protein_n_term);
    let rows = write_table(p.out, columns)?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("n_targets".to_string(), json!(n_targets));
    stats.insert("n_decoys".to_string(), json!(n_decoys));
    stats.insert(
        "decoy_collision_retries".to_string(),
        json!(decoy_collision_retries),
    );
    stats.insert(
        "dropped_target_decoy_pairs".to_string(),
        json!(dropped_pairs),
    );
    if p.protein_n_term {
        stats.insert(
            "protein_n_term_targets".to_string(),
            json!(n_protein_n_term_targets),
        );
    }
    let report = ArtifactReport {
        logical_name: schema.0.to_string(),
        schema_name: schema.0.to_string(),
        schema_version: schema.1,
        stage: "digest".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({
            "enzyme": format!("{:?}", p.cfg.enzyme),
            // Not applied under a non-specific rule, so not recorded as if it were.
            "missed_cleavages": (!rule.is_non_specific()).then_some(p.cfg.missed_cleavages),
            "min_len": p.cfg.min_len, "max_len": p.cfg.max_len,
            "decoy_strategy": format!("{:?}", p.cfg.decoy.strategy),
            "rng_seed": p.rng_seed,
            "max_decoy_attempts": MAX_DECOY_ATTEMPTS,
        }),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    };
    let mut report = report;
    if let Some(rule) = &p.cfg.cleavage_rule {
        // Recorded only for a custom rule, so a preset's report is what it always was.
        report.params["cleavage_rule"] = json!(rule);
    }
    report.write_for(p.out)?;

    info!(
        targets = n_targets,
        decoys = n_decoys,
        rows,
        elapsed_ms = elapsed,
        "digest: done"
    );
    Ok(report.written())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_core::config::Enzyme;

    #[test]
    fn trypsin_p_cleaves_after_kr() {
        let cfg = DigestConfig {
            missed_cleavages: 0,
            min_len: 1,
            max_len: 50,
            ..Default::default()
        };
        // MKRPADEK: cut after K(1), R(2), K(7). TrypsinP cuts before P too.
        let peps: Vec<String> = digest_protein(b"MKRPADEK", &cfg)
            .into_iter()
            .map(|(_, _, p)| p)
            .collect();
        // no-missed-cleavage peptides: MK, R, PADEK
        assert!(peps.contains(&"MK".to_string()), "{peps:?}");
        assert!(peps.contains(&"R".to_string()), "{peps:?}");
        assert!(peps.contains(&"PADEK".to_string()), "{peps:?}");
    }

    /// Non-specific: every window of an allowed length, whatever the residues and the
    /// missed-cleavage setting; a non-standard residue splits the sequence; Met excision
    /// adds no extra rows (the excised form is the window starting at 1).
    /// The trypsin site finder this module had before cleavage rules, kept as the
    /// reference the rule-driven one must reproduce exactly.
    fn legacy_tryptic_sites(seq: &[u8], trypsin_p: bool) -> Vec<usize> {
        let mut sites = vec![0usize];
        for i in 0..seq.len() {
            let c = seq[i];
            if c == b'K' || c == b'R' {
                let before_p = i + 1 < seq.len() && seq[i + 1] == b'P';
                if trypsin_p || !before_p {
                    sites.push(i + 1);
                }
            }
        }
        if *sites.last().unwrap() != seq.len() {
            sites.push(seq.len());
        }
        sites
    }

    /// The `trypsin_p` and `trypsin` presets cut exactly where the hard-coded trypsin did,
    /// on random sequences rich in K, R and P, so a tryptic library is unchanged.
    #[test]
    fn trypsin_presets_reproduce_the_legacy_sites() {
        let alphabet = b"KRPKRPACDEFGHILMNQSTVWY";
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for (enzyme, tp) in [(Enzyme::TrypsinP, true), (Enzyme::Trypsin, false)] {
            let rule = DigestConfig {
                enzyme,
                ..Default::default()
            }
            .rule()
            .unwrap();
            for _ in 0..2000 {
                state = splitmix64(&mut state);
                let n = 1 + (state % 60) as usize;
                let seq: Vec<u8> = (0..n)
                    .map(|_| {
                        state = splitmix64(&mut state);
                        alphabet[(state % alphabet.len() as u64) as usize]
                    })
                    .collect();
                assert_eq!(
                    rule.sites(&seq),
                    legacy_tryptic_sites(&seq, tp),
                    "{enzyme:?} {}",
                    String::from_utf8_lossy(&seq)
                );
            }
        }
    }

    /// A preset and a custom rule with the same text digest alike; Asp-N cuts before D.
    #[test]
    fn presets_and_custom_rules_digest_alike() {
        let asp = DigestConfig {
            enzyme: Enzyme::AspN,
            missed_cleavages: 0,
            min_len: 1,
            ..Default::default()
        };
        let custom = DigestConfig {
            enzyme: Enzyme::Custom,
            cleavage_rule: Some("*D".into()),
            ..asp.clone()
        };
        let peps = |c: &DigestConfig| -> Vec<String> {
            digest_protein(b"AADKKDAA", c)
                .into_iter()
                .map(|(_, _, p)| p)
                .collect()
        };
        assert_eq!(peps(&asp), vec!["AA", "DKK", "DAA"]);
        assert_eq!(peps(&asp), peps(&custom));
    }

    #[test]
    fn non_specific_enumerates_every_window_of_the_allowed_lengths() {
        let cfg = DigestConfig {
            enzyme: Enzyme::NonSpecific,
            missed_cleavages: 0,
            min_len: 10,
            max_len: 10,
            ..Default::default()
        };
        let seq = b"MAEGKRSTVLDPQWYHNFIC";
        let got = digest_protein(seq, &cfg);
        assert_eq!(got.len(), seq.len() - 10 + 1, "{got:?}");
        for (k, (s, e, p)) in got.iter().enumerate() {
            assert_eq!((*s, *e), (k, k + 10));
            assert_eq!(p.as_bytes(), &seq[k..k + 10]);
        }

        let cfg = DigestConfig {
            min_len: 3,
            max_len: 4,
            ..cfg
        };
        // X splits PEPT|IDE: windows of 3-4 inside each run only.
        let peps: Vec<String> = digest_protein(b"PEPTXIDE", &cfg)
            .into_iter()
            .map(|(_, _, p)| p)
            .collect();
        assert_eq!(peps, vec!["PEP", "PEPT", "EPT", "IDE"], "{peps:?}");
    }

    /// The upper bound the non-specific warning reads is the window count before
    /// de-duplication.
    #[test]
    fn non_specific_upper_bound_counts_windows() {
        let cfg = DigestConfig {
            enzyme: Enzyme::NonSpecific,
            min_len: 8,
            max_len: 12,
            ..Default::default()
        };
        let prots = vec![
            ("a".to_string(), "A".repeat(20)),
            ("b".to_string(), "A".repeat(9)),
        ];
        // 20 residues: 13+12+11+10+9 = 55; 9 residues: lengths 8, 9 -> 2+1 = 3.
        assert_eq!(non_specific_upper_bound(&prots, &cfg), 58);
    }

    #[test]
    fn met_excision_emits_both_n_term_forms() {
        // MAKPEPTIDEK: cut after K(2) -> N-term peptide "MAK" (pos 0..3).
        // With excision on, "AK" (pos 1..3) is also emitted; internal "PEPTIDEK"
        // is unaffected. Off, only the Met-retained "MAK" appears.
        let base = DigestConfig {
            missed_cleavages: 0,
            min_len: 1,
            max_len: 50,
            ..Default::default()
        };
        let on = DigestConfig {
            n_term_met_excision: true,
            ..base.clone()
        };
        let off = DigestConfig {
            n_term_met_excision: false,
            ..base
        };
        let peps = |c: &DigestConfig| -> Vec<String> {
            digest_protein(b"MAKPEPTIDEK", c)
                .into_iter()
                .map(|(_, _, p)| p)
                .collect()
        };
        let with = peps(&on);
        assert!(with.contains(&"MAK".to_string()), "{with:?}");
        assert!(with.contains(&"AK".to_string()), "{with:?}");
        assert!(with.contains(&"PEPTIDEK".to_string()), "{with:?}");
        let without = peps(&off);
        assert!(without.contains(&"MAK".to_string()), "{without:?}");
        assert!(!without.contains(&"AK".to_string()), "{without:?}");
    }

    #[test]
    fn met_excision_only_at_protein_n_terminus() {
        // The interior "M" in "AKMDER" (a fragment starting mid-protein) must not
        // be excised: excision keys on protein position 0, not any leading M.
        let cfg = DigestConfig {
            missed_cleavages: 0,
            min_len: 1,
            max_len: 50,
            n_term_met_excision: true,
            ..Default::default()
        };
        // AKMDER: cut after K(2) -> "AK" (0..2) and "MDER" (2..6). "MDER" starts
        // with M but at protein position 2, so no "DER" excision variant.
        let peps: Vec<String> = digest_protein(b"AKMDER", &cfg)
            .into_iter()
            .map(|(_, _, p)| p)
            .collect();
        assert!(peps.contains(&"MDER".to_string()), "{peps:?}");
        assert!(!peps.contains(&"DER".to_string()), "{peps:?}");
    }

    #[test]
    fn met_excision_applies_when_only_the_excised_form_is_in_range() {
        // MPEPTIDK is eight residues; with a length window of exactly seven the
        // Met-retained form is too long and the excised PEPTIDK is the only valid
        // peptide. The length rejection used to `continue` past the excision (docs/29
        // #9), so the digest returned nothing.
        let peps = |min_len: usize, max_len: usize| -> Vec<String> {
            let cfg = DigestConfig {
                missed_cleavages: 0,
                min_len,
                max_len,
                n_term_met_excision: true,
                ..Default::default()
            };
            digest_protein(b"MPEPTIDK", &cfg)
                .into_iter()
                .map(|(_, _, p)| p)
                .collect()
        };
        assert_eq!(peps(7, 7), vec!["PEPTIDK".to_string()]);
        // With room for both, both appear, Met-retained first.
        assert_eq!(
            peps(7, 8),
            vec!["MPEPTIDK".to_string(), "PEPTIDK".to_string()]
        );
        // The excised form is judged on its own length too: at min_len 8 only the
        // ordinary form is in range.
        assert_eq!(peps(8, 8), vec!["MPEPTIDK".to_string()]);
    }

    #[test]
    fn reverse_decoy_keeps_cterm() {
        let d = make_decoy("PEPTIDER", DecoyStrategy::Reverse, 0).unwrap();
        assert_eq!(d.as_bytes()[d.len() - 1], b'R');
        assert_eq!(d, "EDITPEPR"); // reverse of PEPTIDE (->EDITPEP) + R
    }

    #[test]
    fn scramble_is_deterministic() {
        let a = make_decoy("PEPTIDEK", DecoyStrategy::Scramble, 42).unwrap();
        let b = make_decoy("PEPTIDEK", DecoyStrategy::Scramble, 42).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.as_bytes()[a.len() - 1], b'K');
    }

    #[test]
    fn collision_safe_decoy_avoids_targets_and_other_decoys() {
        // The direct reverse of PEPTIDEK is EDITPEPK, which is deliberately
        // present as another target. Construction must retry with a scramble.
        let targets = HashSet::from([il_key("PEPTIDEK"), il_key("EDITPEPK")]);
        let used = HashSet::new();
        let (decoy, retries) =
            collision_safe_decoy("PEPTIDEK", DecoyStrategy::Reverse, 42, &targets, &used)
                .expect("a collision-free scramble exists");
        assert!(retries > 0);
        assert!(!targets.contains(&il_key(&decoy)));
        assert_ne!(decoy, "PEPTIDEK");

        let used = HashSet::from([il_key(&decoy)]);
        let (second, _) =
            collision_safe_decoy("EDITPEPK", DecoyStrategy::Reverse, 42, &targets, &used)
                .expect("a distinct collision-free scramble exists");
        assert!(!targets.contains(&il_key(&second)));
        assert!(!used.contains(&il_key(&second)));
    }

    /// The reversal of IEVNVELR is LEVNVEIR, which differs from it only by I/L and is
    /// therefore the same peptide to the instrument (found as a high-scoring "decoy" on
    /// an Astral HYE run). It must be rejected like an exact palindrome, and so must a
    /// decoy that equals ANOTHER target up to I/L.
    #[test]
    fn a_decoy_equal_to_a_target_up_to_i_l_is_a_collision() {
        assert_eq!(
            make_decoy("IEVNVELR", DecoyStrategy::Reverse, 0).unwrap(),
            "LEVNVEIR"
        );
        let targets = HashSet::from([il_key("IEVNVELR")]);
        let (decoy, retries) = collision_safe_decoy(
            "IEVNVELR",
            DecoyStrategy::Reverse,
            42,
            &targets,
            &HashSet::new(),
        )
        .expect("a collision-free scramble exists");
        assert!(retries > 0, "the plain reversal must be refused");
        assert_ne!(il_key(&decoy), il_key("IEVNVELR"));

        // The reversal of PEPTLDEK is EDLTPEPK, which equals ANOTHER target, EDITPEPK,
        // up to I/L.
        let targets = HashSet::from([il_key("PEPTLDEK"), il_key("EDITPEPK")]);
        let (decoy, retries) = collision_safe_decoy(
            "PEPTLDEK",
            DecoyStrategy::Reverse,
            42,
            &targets,
            &HashSet::new(),
        )
        .expect("a collision-free scramble exists");
        assert!(retries > 0);
        assert!(!targets.contains(&il_key(&decoy)));
    }

    /// Digest `fasta` into a scratch directory with `protein_n_term` set as given, and read
    /// back (peptide, label, target_id, flag) per row plus the report's schema version.
    #[allow(clippy::type_complexity)]
    fn digest_fasta(
        tag: &str,
        fasta: &str,
        protein_n_term: bool,
    ) -> (Vec<(String, String, i32, Option<bool>)>, Vec<String>, u64) {
        let dir = std::env::temp_dir().join(format!("mumdia_digest_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let fa = dir.join("p.fasta").to_string_lossy().into_owned();
        let out = dir.join("peptides.parquet").to_string_lossy().into_owned();
        std::fs::write(&fa, fasta).unwrap();
        let cfg = DigestConfig {
            missed_cleavages: 0,
            min_len: 5,
            ..Default::default()
        };
        run(DigestParams {
            fasta: &fa,
            out: &out,
            cfg: &cfg,
            rng_seed: 7,
            config_hash: "h",
            protein_n_term,
        })
        .unwrap();
        let t = mumdia_io::table::TableFile::open(&out).unwrap();
        let names = t.column_names();
        let flags = t
            .has_column("protein_n_term")
            .then(|| t.bool("protein_n_term").unwrap());
        let (pep, label, tid) = (
            t.str("peptide").unwrap(),
            t.str("label").unwrap(),
            t.i32("target_id").unwrap(),
        );
        let rows = (0..t.nrows)
            .map(|i| {
                (
                    pep[i].clone(),
                    label[i].clone(),
                    tid[i],
                    flags.as_ref().map(|f| f[i]),
                )
            })
            .collect();
        let report: serde_json::Value =
            mumdia_io::json::read_json(&format!("{out}.report.json")).unwrap();
        let version = report["schema_version"].as_u64().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        (rows, names, version)
    }

    /// P1 starts with M (its first peptide and the Met-excised form start the protein); the
    /// peptide WNDLLSTEFR is internal in P2, which is read first, and N-terminal in P3, so
    /// the flag is an OR over occurrences; VLDEFGHIK starts at position 1 after a K, which
    /// is a cleavage product and not a protein N-terminus.
    const NTERM_FASTA: &str = ">P1\nMSEQPEPTIDEKAAAGGGLLLR\n>P2\nGGGAAAKWNDLLSTEFRK\n\
                               >P3\nWNDLLSTEFRAGHIPEDK\n>P4\nKVLDEFGHIKR\n";

    #[test]
    fn protein_n_term_flag_marks_every_peptide_that_starts_a_protein() {
        let (rows, names, version) = digest_fasta("flag", NTERM_FASTA, true);
        assert_eq!(version, 2);
        assert_eq!(names.last().map(String::as_str), Some("protein_n_term"));
        let flag = |p: &str| {
            rows.iter()
                .find(|r| r.0 == p && r.1 == "target")
                .unwrap_or_else(|| panic!("{p} not digested: {rows:?}"))
                .3
                .unwrap()
        };
        for p in ["MSEQPEPTIDEK", "SEQPEPTIDEK", "GGGAAAK", "WNDLLSTEFR"] {
            assert!(flag(p), "{p} starts a protein");
        }
        for p in ["AAAGGGLLLR", "AGHIPEDK", "VLDEFGHIK"] {
            assert!(!flag(p), "{p} does not start a protein");
        }
        // Every decoy carries its target's flag, whatever its own sequence.
        let mut n_decoys = 0;
        for (i, r) in rows.iter().enumerate() {
            if r.1 == "decoy" {
                n_decoys += 1;
                let target = rows.iter().take(i).rev().find(|t| t.1 == "target").unwrap();
                assert_eq!(r.3, target.3, "decoy {} of {}", r.0, target.0);
            }
        }
        assert!(n_decoys >= 7);
        let flagged = |label: &str| {
            rows.iter()
                .filter(|r| r.1 == label && r.3 == Some(true))
                .count()
        };
        assert_eq!(flagged("target"), flagged("decoy"));
    }

    #[test]
    fn without_a_protein_n_term_rule_the_table_is_version_1() {
        let (rows, names, version) = digest_fasta("noflag", NTERM_FASTA, false);
        assert_eq!(version, 1);
        assert_eq!(
            names,
            [
                "id",
                "peptide",
                "protein",
                "start",
                "end",
                "label",
                "target_id",
                "decoy_strategy"
            ]
        );
        assert!(rows.iter().all(|r| r.3.is_none()));
        // The rows themselves are those of the flagged digest.
        let (flagged, _, _) = digest_fasta("noflag2", NTERM_FASTA, true);
        let strip = |v: &[(String, String, i32, Option<bool>)]| {
            v.iter()
                .map(|r| (r.0.clone(), r.1.clone(), r.2))
                .collect::<Vec<_>>()
        };
        assert_eq!(strip(&rows), strip(&flagged));
    }

    #[test]
    fn met_excision_off_leaves_position_1_unflagged() {
        let cfg = DigestConfig {
            n_term_met_excision: false,
            ..Default::default()
        };
        assert!(starts_protein(b"MSEQK", 0, &cfg));
        assert!(!starts_protein(b"MSEQK", 1, &cfg));
        let on = DigestConfig::default();
        assert!(starts_protein(b"MSEQK", 1, &on));
        assert!(!starts_protein(b"KSEQK", 1, &on));
        assert!(!starts_protein(b"MSEQK", 2, &on));
    }

    #[test]
    fn impossible_low_complexity_decoy_drops_pair() {
        let targets = HashSet::from([il_key("AAAAAK")]);
        assert!(collision_safe_decoy(
            "AAAAAK",
            DecoyStrategy::Reverse,
            42,
            &targets,
            &HashSet::new()
        )
        .is_none());
    }
}
