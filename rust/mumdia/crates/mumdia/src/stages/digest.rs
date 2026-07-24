//! Stage A `mumdia digest`: fully-tryptic enumeration + decoy pairing
//! (PLAN.md Stage A). Experiment-wide, computed once. MVP uses a documented,
//! clean-room decoy scheme (reverse or seeded scramble); the DIA-NN terminal
//! shift map is a later license-checked addition (PLAN.md Section 11).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{Context, Result};
use mumdia_core::config::{DecoyStrategy, DigestConfig, Enzyme};
use mumdia_core::constants::is_standard_residue;
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
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

/// Tryptic cleavage sites: index positions (0-based, after this residue) where
/// the enzyme cuts.
fn cleavage_sites(seq: &[u8], enzyme: Enzyme) -> Vec<usize> {
    let mut sites = vec![0usize];
    for i in 0..seq.len() {
        let c = seq[i];
        if c == b'K' || c == b'R' {
            let before_p = i + 1 < seq.len() && seq[i + 1] == b'P';
            let cut = match enzyme {
                Enzyme::TrypsinP => true,
                Enzyme::Trypsin => !before_p,
            };
            if cut {
                sites.push(i + 1);
            }
        }
    }
    if *sites.last().unwrap() != seq.len() {
        sites.push(seq.len());
    }
    sites
}

/// Enumerate tryptic peptides for one protein with up to `missed` missed
/// cleavages, respecting length bounds and dropping ambiguous residues.
fn digest_protein(seq: &[u8], cfg: &DigestConfig) -> Vec<(usize, usize, String)> {
    let sites = cleavage_sites(seq, cfg.enzyme);
    let mut out = Vec::new();
    for i in 0..sites.len().saturating_sub(1) {
        for j in (i + 1)..sites.len() {
            let missed = j - i - 1;
            if missed > cfg.missed_cleavages as usize {
                break;
            }
            let (start, end) = (sites[i], sites[j]);
            let len = end - start;
            if len < cfg.min_len || len > cfg.max_len {
                continue;
            }
            let sub = &seq[start..end];
            if !sub.iter().all(|&c| is_standard_residue(c)) {
                continue;
            }
            out.push((start, end, String::from_utf8_lossy(sub).to_string()));

            // N-terminal methionine excision: for a peptide anchored at the
            // protein N-terminus whose first residue is the initiator Met, also
            // emit the Met-removed form (start shifted to 1). The excised peptide
            // is re-checked against the length bounds and standard-residue rule.
            // This mirrors DIA-NN's `--met-excision`; without it the search
            // database cannot contain these (biologically dominant) peptides.
            if cfg.n_term_met_excision && start == 0 && seq.first() == Some(&b'M') {
                let ex = &seq[1..end];
                let ex_len = ex.len();
                if ex_len >= cfg.min_len
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

/// Make a decoy sequence from a target (PLAN.md Section 9.2). Reverse keeps the
/// C-terminal residue fixed; scramble seeded-shuffles the interior.
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

/// Construct a deterministic decoy that is different from its paired target,
/// does not collide with any target peptide, and is unique among emitted decoys.
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
        if decoy != pep && !targets.contains(&decoy) && !used_decoys.contains(&decoy) {
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
}

pub fn run(p: DigestParams) -> Result<u64> {
    let t0 = Instant::now();
    let proteins = read_fasta(p.fasta)?;
    info!(proteins = proteins.len(), "digest: read fasta");

    // Dedup target peptides by sequence, aggregating protein accessions.
    let mut pep_proteins: HashMap<String, Vec<String>> = HashMap::new();
    let mut pep_pos: HashMap<String, (i32, i32)> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for (acc, seq) in &proteins {
        let bytes = seq.as_bytes();
        for (start, end, pep) in digest_protein(bytes, p.cfg) {
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

    let target_sequences: HashSet<String> = order.iter().cloned().collect();
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
                    used_decoys.insert(decoy.clone());
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

        // Materialize sequence-rewrite decoys here (PLAN.md Stage A).
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
    let rows = write_table(
        p.out,
        vec![
            Col::U32("id".into(), id_c),
            Col::Str("peptide".into(), pep_c),
            Col::Str("protein".into(), prot_c),
            Col::I32("start".into(), start_c),
            Col::I32("end".into(), end_c),
            Col::Str("label".into(), label_c),
            Col::I32("target_id".into(), target_c),
            Col::Str("decoy_strategy".into(), strat_c),
        ],
    )?;

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
    ArtifactReport {
        logical_name: artifact::PEPTIDES.0.to_string(),
        schema_name: artifact::PEPTIDES.0.to_string(),
        schema_version: artifact::PEPTIDES.1,
        stage: "digest".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({
            "enzyme": format!("{:?}", p.cfg.enzyme),
            "missed_cleavages": p.cfg.missed_cleavages,
            "min_len": p.cfg.min_len, "max_len": p.cfg.max_len,
            "decoy_strategy": format!("{:?}", p.cfg.decoy.strategy),
            "rng_seed": p.rng_seed,
            "max_decoy_attempts": MAX_DECOY_ATTEMPTS,
        }),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;

    info!(
        targets = n_targets,
        decoys = n_decoys,
        rows,
        elapsed_ms = elapsed,
        "digest: done"
    );
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let targets = HashSet::from(["PEPTIDEK".to_string(), "EDITPEPK".to_string()]);
        let used = HashSet::new();
        let (decoy, retries) =
            collision_safe_decoy("PEPTIDEK", DecoyStrategy::Reverse, 42, &targets, &used)
                .expect("a collision-free scramble exists");
        assert!(retries > 0);
        assert!(!targets.contains(&decoy));
        assert_ne!(decoy, "PEPTIDEK");

        let used = HashSet::from([decoy]);
        let (second, _) =
            collision_safe_decoy("EDITPEPK", DecoyStrategy::Reverse, 42, &targets, &used)
                .expect("a distinct collision-free scramble exists");
        assert!(!targets.contains(&second));
        assert!(!used.contains(&second));
    }

    #[test]
    fn impossible_low_complexity_decoy_drops_pair() {
        let targets = HashSet::from(["AAAAAK".to_string()]);
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
