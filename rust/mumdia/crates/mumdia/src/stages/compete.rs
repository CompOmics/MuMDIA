//! Compete step `mumdia compete` (docs/11_compete_rescore_fdr.md): within each
//! competition group keep only the best-scoring candidate before target-decoy
//! counting, so multiple plausible candidates for one elution peak cannot inflate
//! discoveries. MVP groups by base peptide (target + its decoy + charge/mod
//! variants); the grouping is configurable.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use mumdia_core::config::{CompeteConfig, CompeteGroupBy, CompetitionMode};
use mumdia_core::rejection::RejectionReason;
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, Table};
use serde_json::json;
use tracing::{info, warn};

use crate::stages::features::FeatureSchema;

pub struct CompeteParams<'a> {
    pub features: &'a str,
    pub out: &'a str,
    pub cfg: &'a CompeteConfig,
    pub config_hash: &'a str,
}

pub fn run(p: CompeteParams) -> Result<u64> {
    let t0 = Instant::now();
    let t = Table::read(p.features)?;
    let cid = t.u32("candidate_id")?;
    let label = t.str("label")?;
    let base = t.u32("base_peptide_id")?;
    let pform = t.str("peptidoform")?;
    let protein = t.str("protein")?;
    let apex_rt = t.f64("apex_rt")?;
    let elution_lo = t.f64("elution_lo")?;
    let elution_hi = t.f64("elution_hi")?;
    let mz = t.f64("precursor_mz")?;
    let prelim = t.f64("prelim_score")?;
    // Top-K peak rank (#7). Part of the competition key so peaks of one candidate
    // compete only within their own rank (a lower-scoring peak of a candidate must
    // not eliminate a sibling's better peak on prelim score before rescore picks).
    // Missing -> 0 (single-apex), so the grouping is unchanged when promotion is off.
    let peak_rank = t.i32("peak_rank").unwrap_or_else(|_| vec![0; t.nrows]);
    let schema = FeatureSchema::read(p.features)?;
    let feat_names = &schema.feature_columns;
    // `?`, not `.unwrap()`. The names come from the `.schema.json` companion and the
    // values from the parquet, which are two separately addressable files: a stale
    // companion left beside a rewritten table names a column the parquet no longer
    // has, and that used to abort with `called `Option::unwrap()` on a `None` value`
    // naming neither the column nor the file. `FeatureSchema::read` already falls
    // back to the parquet's own columns when the companion is MISSING; this covers
    // the case where it is present and wrong.
    let feat_cols: Vec<Vec<f64>> = feat_names
        .iter()
        .map(|c| {
            t.f64(c).with_context(|| {
                format!(
                    concat!(
                        "feature column {c:?} is named by {f}.schema.json but is not in ",
                        "{f}; the companion is stale, delete it to reconstruct the column ",
                        "list from the parquet itself"
                    ),
                    c = c,
                    f = p.features
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // `charge` is a minimal feature column (present in every set), stored as f64.
    // Only the peptidoform-charge grouping needs it.
    let charge = t.f64("charge").ok();
    if matches!(p.cfg.group_by, CompeteGroupBy::PeptidoformCharge) && charge.is_none() {
        anyhow::bail!("compete group_by=peptidoform_charge requires a 'charge' feature column");
    }
    // Dense peptidoform id by first appearance (deterministic) so the fixed-size
    // tuple key can separate modforms without allocating a String per PSM. Built
    // only for the peptidoform-charge grouping; empty otherwise.
    let pform_id: Vec<u32> = if matches!(p.cfg.group_by, CompeteGroupBy::PeptidoformCharge) {
        let mut ids = Vec::with_capacity(t.nrows);
        let mut seen: HashMap<&str, u32> = HashMap::new();
        for peptidoform in pform.iter().take(t.nrows) {
            let next = seen.len() as u32;
            ids.push(*seen.entry(peptidoform.as_str()).or_insert(next));
        }
        ids
    } else {
        Vec::new()
    };

    // Competition group members. The label is part of the key so a target is NOT
    // competed against its own decoy: the decoy population must survive for the
    // rescorer/FDR to have a valid null (otherwise decoys are depleted and FDR is
    // badly underestimated). Competition only arbitrates redundant charge/mod
    // variants within targets and within decoys.
    // Key is a fixed-size tuple (base-or-pform id, label_code, bucket) instead of a
    // freshly-allocated String per PSM. Precursor grouping uses a constant bucket
    // (0) so its equivalence classes are unchanged.
    let mut groups: HashMap<(u32, u8, i64, i32), Vec<usize>> = HashMap::new();
    for i in 0..t.nrows {
        let label_code = match label[i].as_str() {
            "target" => 0u8,
            "decoy" => 1u8,
            _ => 2u8,
        };
        let pk = peak_rank[i];
        let key = match p.cfg.group_by {
            CompeteGroupBy::BasePeptide => (base[i], label_code, 0i64, pk),
            CompeteGroupBy::Apex => {
                let bucket = (apex_rt[i] / p.cfg.apex_rt_tolerance_s).round() as i64;
                (base[i], label_code, bucket, pk)
            }
            CompeteGroupBy::PeptidoformCharge => {
                // pform_id separates modforms; charge in the bucket separates
                // charges -> one group per peptidoform+charge (precursor-level).
                let c = charge.as_ref().unwrap()[i].round() as i64;
                (pform_id[i], label_code, c, pk)
            }
        };
        groups.entry(key).or_default().push(i);
    }

    // Per-candidate unique-fragment evidence for the `unique_evidence` mode. Prefers
    // an explicit `unique_fragment_count` feature; otherwise approximates it as
    // matched-fragment count discounted by the contested fraction; None if neither
    // is available (mode then falls back to winner-take-all).
    let unique_ev_src = unique_evidence_with_source(&t);
    if matches!(p.cfg.mode, CompetitionMode::UniqueEvidence) {
        // The mode keeps any non-winner whose unique evidence >= unique_evidence_min_fragments.
        // Warn when NOTHING in this run can fall below that threshold, because then the mode is
        // silently identical to CompetitionMode::None.
        //
        // Tested on the VALUES, not on which column the estimate came from. Keying on the
        // column name missed the common case: `peak_contested_frac` is part of the Extended
        // feature set unconditionally, so the estimate always reports itself as
        // "contested-discounted" even when that column is all zeros (competition features off),
        // which discounts nothing and leaves the raw matched count -- exactly the no-op the
        // warning exists to announce.
        if let Some((ev, src)) = unique_ev_src.as_ref() {
            let thr = p.cfg.unique_evidence_min_fragments as f64;
            let below = ev.iter().filter(|v| **v < thr).count();
            if below == 0 {
                warn!(
                    source = src,
                    threshold = thr,
                    candidates = ev.len(),
                    "compete mode=unique_evidence: no candidate's unique evidence falls below                      compete.unique_evidence_min_fragments, so NOTHING will be removed and this                      run is equivalent to mode=none. Enable the contested/competition features                      so the evidence is actually discounted, or raise the threshold."
                );
            }
        }
    }
    let unique_ev = unique_ev_src.map(|(v, _)| v);
    if matches!(p.cfg.mode, CompetitionMode::UniqueEvidence) && unique_ev.is_none() {
        warn!(
            "compete mode=unique_evidence: no unique_fragment_count / \
             (n_matched_fragments, peak_contested_frac/contested_frac) columns; \
             falling back to winner-take-all"
        );
    }

    // Resolve each group per competition mode (pure function; unit tested below).
    let (keep, removed) = resolve_competition(
        &groups,
        &prelim,
        p.cfg.mode,
        p.cfg.margin,
        p.cfg.unique_evidence_min_fragments,
        unique_ev.as_deref(),
    );

    let sel = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<_>>();
    let mut cols: Vec<Col> = vec![
        Col::U32(
            "candidate_id".into(),
            keep.iter().map(|&i| cid[i]).collect(),
        ),
        Col::I32(
            "peak_rank".into(),
            keep.iter().map(|&i| peak_rank[i]).collect(),
        ),
        Col::Str(
            "label".into(),
            keep.iter().map(|&i| label[i].clone()).collect(),
        ),
        Col::U32(
            "base_peptide_id".into(),
            keep.iter().map(|&i| base[i]).collect(),
        ),
        Col::Str(
            "peptidoform".into(),
            keep.iter().map(|&i| pform[i].clone()).collect(),
        ),
        Col::Str(
            "protein".into(),
            keep.iter().map(|&i| protein[i].clone()).collect(),
        ),
        Col::F64("apex_rt".into(), sel(&apex_rt)),
        Col::F64("elution_lo".into(), sel(&elution_lo)),
        Col::F64("elution_hi".into(), sel(&elution_hi)),
        Col::F64("precursor_mz".into(), sel(&mz)),
        Col::F64("prelim_score".into(), sel(&prelim)),
    ];
    for (fi, name) in feat_names.iter().enumerate() {
        cols.push(Col::F64(name.clone(), sel(&feat_cols[fi])));
    }
    let rows = write_table(p.out, cols)?;
    // Carry the feature schema forward for rescore.
    mumdia_io::json::write_json(&format!("{}.schema.json", p.out), &schema)?;

    // Optional per-removal competition audit (spec 04 §2): every candidate removed
    // by within-group competition, with its winner and removal reason. Within-label
    // competition, so the sibling that outcompeted a loser shares its label.
    if p.cfg.emit_competition_audit {
        let reason_of = |i: usize| {
            if label[i] == "decoy" {
                RejectionReason::OutcompetedByDecoy
            } else {
                RejectionReason::OutcompetedByTarget
            }
        };
        let audit = format!("{}.compete_audit.parquet", p.out);
        write_table(
            &audit,
            vec![
                Col::U32(
                    "candidate_id".into(),
                    removed.iter().map(|&(m, _)| cid[m]).collect(),
                ),
                Col::Str(
                    "label".into(),
                    removed.iter().map(|&(m, _)| label[m].clone()).collect(),
                ),
                Col::Str(
                    "peptidoform".into(),
                    removed.iter().map(|&(m, _)| pform[m].clone()).collect(),
                ),
                Col::U32(
                    "winner_candidate_id".into(),
                    removed.iter().map(|&(_, w)| cid[w]).collect(),
                ),
                Col::F64(
                    "loser_prelim".into(),
                    removed.iter().map(|&(m, _)| prelim[m]).collect(),
                ),
                Col::F64(
                    "winner_prelim".into(),
                    removed.iter().map(|&(_, w)| prelim[w]).collect(),
                ),
                Col::Str(
                    "rejection_reason".into(),
                    removed
                        .iter()
                        .map(|&(m, _)| reason_of(m).code().to_string())
                        .collect(),
                ),
            ],
        )?;
    }

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("input_rows".to_string(), json!(t.nrows));
    stats.insert("kept".to_string(), json!(rows));
    stats.insert("removed".to_string(), json!(removed.len()));
    ArtifactReport {
        logical_name: artifact::PSMS_COMPETED.0.to_string(),
        schema_name: artifact::PSMS_COMPETED.0.to_string(),
        schema_version: artifact::PSMS_COMPETED.1,
        stage: "compete".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({
            "group_by": format!("{:?}", p.cfg.group_by),
            "mode": format!("{:?}", p.cfg.mode),
        }),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;

    info!(
        input = t.nrows,
        kept = rows,
        removed = removed.len(),
        mode = ?p.cfg.mode,
        "compete: done"
    );
    Ok(rows)
}

/// Per-candidate unique-fragment evidence for `CompetitionMode::UniqueEvidence`.
/// Prefers an explicit `unique_fragment_count` column; otherwise approximates it as
/// `n_matched_fragments * (1 - peak_contested_frac)` (contested-discounted matched
/// count). The legacy `contested_frac` spelling remains a fallback; raw matched
/// count is used if neither fraction exists, and `None` if no matched count exists.
/// Also reports which column the estimate came from, so the caller can warn when the
/// weakest fallback would make the mode a silent no-op.
fn unique_evidence_with_source(t: &Table) -> Option<(Vec<f64>, &'static str)> {
    if let Some(u) = col_f64(t, "unique_fragment_count") {
        return Some((u, "unique_fragment_count"));
    }
    let nm = col_f64(t, "n_matched_fragments")?;
    let contested = prefer_peak_contested_fraction(
        col_f64(t, "peak_contested_frac"),
        col_f64(t, "contested_frac"),
    );
    match contested {
        Some(cf) => Some((
            nm.iter()
                .zip(cf)
                .map(|(n, c)| n * (1.0 - c).clamp(0.0, 1.0))
                .collect(),
            "contested-discounted n_matched_fragments",
        )),
        None => Some((nm, "n_matched_fragments")),
    }
}

/// Prefer the Extended-feature spelling while accepting older feature tables.
fn prefer_peak_contested_fraction(
    peak_contested: Option<Vec<f64>>,
    legacy_contested: Option<Vec<f64>>,
) -> Option<Vec<f64>> {
    peak_contested.or(legacy_contested)
}

/// Read a numeric column as f64, accepting an f64 or i32 encoding.
fn col_f64(t: &Table, name: &str) -> Option<Vec<f64>> {
    t.f64(name).ok().or_else(|| {
        t.i32(name)
            .ok()
            .map(|v| v.into_iter().map(|x| x as f64).collect())
    })
}

/// Resolve within-group competition per [`CompetitionMode`]. Returns the
/// sorted-unique kept row indices and the `(loser, winner)` removal pairs.
/// Deterministic: groups are visited in sorted key order; the winner is the
/// highest `prelim` (ties broken by smallest index).
fn resolve_competition(
    groups: &HashMap<(u32, u8, i64, i32), Vec<usize>>,
    prelim: &[f64],
    mode: CompetitionMode,
    margin: f64,
    unique_min: usize,
    unique_ev: Option<&[f64]>,
) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut group_keys: Vec<&(u32, u8, i64, i32)> = groups.keys().collect();
    group_keys.sort_unstable();
    let mut keep: Vec<usize> = Vec::new();
    let mut removed: Vec<(usize, usize)> = Vec::new();
    for gk in group_keys {
        let members = &groups[gk];
        let win = *members
            .iter()
            // `total_cmp`, not `partial_cmp(..).unwrap_or(Equal)`: this picks the
            // single row that survives competition, and treating every NaN
            // prelim_score as equal to every other score made that choice depend on
            // iteration order. `total_cmp` is a genuine total order, so the winner
            // is well defined even then, and the `.then(a.cmp(&b))` index tiebreak
            // keeps it deterministic.
            .min_by(|&&a, &&b| prelim[b].total_cmp(&prelim[a]).then(a.cmp(&b)))
            .unwrap();
        match mode {
            CompetitionMode::None | CompetitionMode::FeaturesOnly => {
                keep.extend(members.iter().copied());
            }
            CompetitionMode::WinnerTakeAll => {
                keep.push(win);
                removed.extend(
                    members
                        .iter()
                        .copied()
                        .filter(|&m| m != win)
                        .map(|m| (m, win)),
                );
            }
            CompetitionMode::UniqueEvidence => {
                keep.push(win);
                let thr = unique_min as f64;
                for &m in members {
                    if m == win {
                        continue;
                    }
                    if unique_ev.map(|u| u[m] >= thr).unwrap_or(false) {
                        keep.push(m);
                    } else {
                        removed.push((m, win));
                    }
                }
            }
            CompetitionMode::MarginGated => {
                keep.push(win);
                for &m in members {
                    if m == win {
                        continue;
                    }
                    if prelim[win] - prelim[m] >= margin {
                        removed.push((m, win));
                    } else {
                        keep.push(m);
                    }
                }
            }
        }
    }
    keep.sort_unstable();
    keep.dedup();
    (keep, removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_group(members: Vec<usize>) -> HashMap<(u32, u8, i64, i32), Vec<usize>> {
        let mut g = HashMap::new();
        g.insert((0u32, 0u8, 0i64, 0i32), members);
        g
    }

    #[test]
    fn winner_take_all_keeps_only_winner() {
        let g = one_group(vec![0, 1, 2]);
        let prelim = [0.1, 0.9, 0.5];
        let (keep, removed) =
            resolve_competition(&g, &prelim, CompetitionMode::WinnerTakeAll, 0.0, 2, None);
        assert_eq!(keep, vec![1]);
        assert_eq!(removed.len(), 2);
    }

    #[test]
    fn none_and_features_only_keep_all() {
        let g = one_group(vec![0, 1, 2]);
        let prelim = [0.1, 0.9, 0.5];
        for mode in [CompetitionMode::None, CompetitionMode::FeaturesOnly] {
            let (keep, removed) = resolve_competition(&g, &prelim, mode, 0.0, 2, None);
            assert_eq!(keep, vec![0, 1, 2]);
            assert!(removed.is_empty());
        }
    }

    #[test]
    fn margin_gated_keeps_close_losers_removes_distant() {
        // winner idx1 (0.9); idx2 (0.85) within margin 0.1 -> kept; idx0 removed
        let g = one_group(vec![0, 1, 2]);
        let prelim = [0.1, 0.9, 0.85];
        let (keep, removed) =
            resolve_competition(&g, &prelim, CompetitionMode::MarginGated, 0.1, 2, None);
        assert_eq!(keep, vec![1, 2]);
        assert_eq!(removed, vec![(0, 1)]);
    }

    #[test]
    fn unique_evidence_keeps_losers_with_enough_evidence() {
        // winner idx1; idx0 unique 3 (>=2) kept; idx2 unique 1 removed
        let g = one_group(vec![0, 1, 2]);
        let prelim = [0.1, 0.9, 0.5];
        let ev = [3.0, 5.0, 1.0];
        let (keep, removed) = resolve_competition(
            &g,
            &prelim,
            CompetitionMode::UniqueEvidence,
            0.0,
            2,
            Some(&ev),
        );
        assert_eq!(keep, vec![0, 1]);
        assert_eq!(removed, vec![(2, 1)]);
    }

    #[test]
    fn unique_evidence_without_data_falls_back_to_winner_take_all() {
        let g = one_group(vec![0, 1, 2]);
        let prelim = [0.1, 0.9, 0.5];
        let (keep, _) =
            resolve_competition(&g, &prelim, CompetitionMode::UniqueEvidence, 0.0, 2, None);
        assert_eq!(keep, vec![1]);
    }

    #[test]
    fn unique_evidence_prefers_extended_peak_contested_fraction() {
        let selected = prefer_peak_contested_fraction(Some(vec![0.25]), Some(vec![0.75])).unwrap();
        assert_eq!(selected, vec![0.25]);
        let legacy = prefer_peak_contested_fraction(None, Some(vec![0.75])).unwrap();
        assert_eq!(legacy, vec![0.75]);
    }

    #[test]
    fn winner_take_all_is_deterministic_across_groups() {
        let mut g = HashMap::new();
        g.insert((0u32, 0u8, 0i64, 0i32), vec![0, 1]);
        g.insert((1u32, 1u8, 0i64, 0i32), vec![2, 3]);
        let prelim = [0.2, 0.8, 0.9, 0.3];
        let (keep, _) =
            resolve_competition(&g, &prelim, CompetitionMode::WinnerTakeAll, 0.0, 2, None);
        assert_eq!(keep, vec![1, 2]); // winners of each group, sorted
    }

    #[test]
    fn winner_tie_breaks_to_smallest_index() {
        let g = one_group(vec![0, 1, 2]);
        let prelim = [0.9, 0.9, 0.1]; // tie between idx0 and idx1
        let (keep, _) =
            resolve_competition(&g, &prelim, CompetitionMode::WinnerTakeAll, 0.0, 2, None);
        assert_eq!(keep, vec![0]);
    }
}
