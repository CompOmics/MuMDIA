//! Candidate audit `mumdia audit` (sensitivity program, spec
//! `01_workflow_and_gap_analysis.md` §4, `02_sensitivity_diagnostic_plan.md` §5,
//! backlog P0.3 / P0.4).
//!
//! Non-destructive, post-hoc observability: reconstruct, for every candidate in
//! the search space (the library precursors), the pipeline stage flags and the
//! EARLIEST rejection reason, by tracking which `candidate_id`s survive across the
//! artifact chain library -> psms(extract) -> competed(compete) -> scored(rescore).
//! Writes `candidate_audit.parquet` and prints the identification-loss waterfall.
//!
//! This stage never re-runs compute and never changes any pipeline output, so it
//! is safe to run after any search. It answers "where was each candidate first
//! lost?" at the resolution the artifacts allow. The extraction stage collapses
//! "no fragment traces" and "traces but no accepted peak" into a single observable
//! event (a candidate is in `psms` or it is not); when a future in-extract audit
//! sidecar `<psms>.audit.parquet` is present, its precise per-candidate reason
//! refines the extract-stage bucket (see [`load_extract_reasons`]).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{Context, Result};
use mumdia_core::rejection::RejectionReason;
use mumdia_io::table::{write_table, Col, Table};
use serde_json::json;
use tracing::info;

pub struct AuditParams<'a> {
    /// Library precursors parquet: the full candidate search space.
    pub library_precursors: &'a str,
    /// psms parquet written by `extract` (candidates that produced an accepted peak).
    pub psms: &'a str,
    /// competed parquet written by `compete` (survivors of within-group competition).
    pub competed: &'a str,
    /// scored parquet written by `rescore` (candidate_id + q_value).
    pub scored: &'a str,
    /// Output `candidate_audit.parquet`.
    pub out: &'a str,
    /// Precursor q-value threshold for `passed_precursor_fdr` / `reported`.
    pub q_threshold: f64,
    /// Run identifier stamped on every row.
    pub run_id: &'a str,
    /// Optional protein-substring marking entrapment candidates (e.g. `_HUMAN` for
    /// an E. coli sample vs an HYE library). Empty = no entrapment labelling.
    pub entrapment_substr: &'a str,
}

/// Optional per-candidate extract-stage reason refinement written by a future
/// in-extract audit (`extract.emit_candidate_audit`). Returns a map
/// candidate_id -> reason code string. Absent file -> empty map (no refinement).
fn load_extract_reasons(psms_path: &str) -> HashMap<u32, String> {
    let sidecar = format!("{psms_path}.audit.parquet");
    let mut out = HashMap::new();
    if let Ok(t) = Table::read(&sidecar) {
        if let (Ok(cid), Ok(reason)) = (t.u32("candidate_id"), t.str("rejection_reason")) {
            for (c, r) in cid.into_iter().zip(reason) {
                out.insert(c, r);
            }
        }
    }
    out
}

pub fn run(p: AuditParams) -> Result<u64> {
    let t0 = Instant::now();

    // Search space = all library precursors.
    let lib = Table::read(p.library_precursors)
        .with_context(|| format!("audit: reading library precursors {}", p.library_precursors))?;
    let cid = lib.u32("candidate_id")?;
    let pform = lib.str("peptidoform")?;
    let charge = lib.i32("charge")?;
    let label = lib.str("label")?;
    let protein = lib.str("protein")?;
    let n = cid.len();

    // Survivor sets keyed by candidate_id from each downstream artifact.
    let extracted: HashSet<u32> = Table::read(p.psms)?
        .u32("candidate_id")?
        .into_iter()
        .collect();
    let competed: HashSet<u32> = Table::read(p.competed)?
        .u32("candidate_id")?
        .into_iter()
        .collect();
    let scored_t = Table::read(p.scored)?;
    let scored_cid = scored_t.u32("candidate_id")?;
    let scored_q = scored_t.f64("q_value")?;
    // peptide-level q is optional (only present in some scored schemas).
    let scored_pep_q = scored_t.f64("peptide_q_value").ok();
    let mut q_by_cid: HashMap<u32, f64> = HashMap::with_capacity(scored_cid.len());
    let mut pepq_by_cid: HashMap<u32, f64> = HashMap::new();
    for (i, c) in scored_cid.iter().enumerate() {
        q_by_cid.insert(*c, scored_q[i]);
        if let Some(pq) = &scored_pep_q {
            pepq_by_cid.insert(*c, pq[i]);
        }
    }
    let extract_reasons = load_extract_reasons(p.psms);

    // Output columns.
    let mut run_c: Vec<String> = Vec::with_capacity(n);
    let mut prec_c: Vec<u32> = Vec::with_capacity(n);
    let mut seq_c: Vec<String> = Vec::with_capacity(n);
    let mut chg_c: Vec<i32> = Vec::with_capacity(n);
    let mut td_c: Vec<String> = Vec::with_capacity(n);
    let mut entrap_c: Vec<bool> = Vec::with_capacity(n);
    let mut f_generated: Vec<bool> = Vec::with_capacity(n);
    let mut f_traces: Vec<bool> = Vec::with_capacity(n);
    let mut f_peak: Vec<bool> = Vec::with_capacity(n);
    let mut f_peak_sel: Vec<bool> = Vec::with_capacity(n);
    let mut f_variant: Vec<bool> = Vec::with_capacity(n);
    let mut f_td_winner: Vec<bool> = Vec::with_capacity(n);
    let mut f_prec_fdr: Vec<bool> = Vec::with_capacity(n);
    let mut f_pep_fdr: Vec<bool> = Vec::with_capacity(n);
    let mut f_reported: Vec<bool> = Vec::with_capacity(n);
    let mut reason_c: Vec<String> = Vec::with_capacity(n);

    // Waterfall counters.
    let mut waterfall: HashMap<&'static str, u64> = HashMap::new();

    for i in 0..n {
        let c = cid[i];
        let is_decoy = label[i] == "decoy";
        let traces = extracted.contains(&c);
        let variant = competed.contains(&c);
        let q = q_by_cid.get(&c).copied();
        let in_scored = q.is_some();
        let passed_prec = q.map(|v| v <= p.q_threshold).unwrap_or(false);
        let passed_pep = pepq_by_cid
            .get(&c)
            .map(|v| *v <= p.q_threshold)
            .unwrap_or(passed_prec); // fall back to precursor gate when no peptide-q

        // Earliest rejection reason along the ladder.
        let reason: RejectionReason = if !traces {
            // Extraction produced no accepted peak for this candidate. Refine with
            // the in-extract audit sidecar if present; otherwise the generic bucket.
            match extract_reasons.get(&c).map(String::as_str) {
                Some("NO_FRAGMENT_TRACES") => RejectionReason::NoFragmentTraces,
                Some("NO_VALID_FRAGMENTS") => RejectionReason::NoValidFragments,
                Some("PEAK_NOT_SELECTED") => RejectionReason::PeakNotSelected,
                Some("RT_PRUNED") => RejectionReason::RtPruned,
                Some("WRONG_ISOLATION_WINDOW") => RejectionReason::WrongIsolationWindow,
                _ => RejectionReason::NoPeakGroup,
            }
        } else if !variant {
            if is_decoy {
                RejectionReason::OutcompetedByDecoy
            } else {
                RejectionReason::OutcompetedByTarget
            }
        } else if !passed_prec {
            RejectionReason::FailedPrecursorFdr
        } else if !passed_pep {
            RejectionReason::FailedPeptideFdr
        } else {
            RejectionReason::Reported
        };
        *waterfall.entry(reason.code()).or_insert(0) += 1;

        run_c.push(p.run_id.to_string());
        prec_c.push(c);
        seq_c.push(pform[i].clone());
        chg_c.push(charge[i]);
        td_c.push(label[i].clone());
        entrap_c.push(!p.entrapment_substr.is_empty() && protein[i].contains(p.entrapment_substr));
        f_generated.push(true); // in the search space by construction
        f_traces.push(traces);
        f_peak.push(traces); // artifact resolution: an accepted peak == present in psms
        f_peak_sel.push(traces);
        f_variant.push(variant);
        f_td_winner.push(in_scored);
        f_prec_fdr.push(passed_prec);
        f_pep_fdr.push(passed_pep && passed_prec);
        f_reported.push(passed_prec);
        reason_c.push(reason.code().to_string());
    }

    let rows = write_table(
        p.out,
        vec![
            Col::Str("run_id".into(), run_c),
            Col::U32("precursor_id".into(), prec_c),
            Col::Str("modified_sequence".into(), seq_c),
            Col::I32("charge".into(), chg_c),
            Col::Str("target_decoy_label".into(), td_c),
            Col::Bool("entrapment_label".into(), entrap_c),
            Col::Bool("candidate_generated".into(), f_generated),
            Col::Bool("traces_extracted".into(), f_traces),
            Col::Bool("peak_generated".into(), f_peak),
            Col::Bool("peak_selected".into(), f_peak_sel),
            Col::Bool("variant_selected".into(), f_variant),
            Col::Bool("target_decoy_winner".into(), f_td_winner),
            Col::Bool("passed_precursor_fdr".into(), f_prec_fdr),
            Col::Bool("passed_peptide_fdr".into(), f_pep_fdr),
            Col::Bool("reported".into(), f_reported),
            Col::Str("rejection_reason".into(), reason_c),
        ],
    )?;

    // Stage-level metrics + waterfall (P0.4), written next to the audit table.
    let n_extracted = extracted.len() as u64;
    let n_competed = competed.len() as u64;
    let n_reported = *waterfall.get("REPORTED").unwrap_or(&0);
    let metrics = json!({
        "run_id": p.run_id,
        "q_threshold": p.q_threshold,
        "search_space": n,
        "extracted": n_extracted,
        "competed": n_competed,
        "reported": n_reported,
        "trace_recall": n_extracted as f64 / (n.max(1) as f64),
        "waterfall": waterfall.iter().map(|(k, v)| (k.to_string(), *v)).collect::<std::collections::BTreeMap<_, _>>(),
    });
    mumdia_io::json::write_json(&format!("{}.metrics.json", p.out), &metrics)?;

    let elapsed = t0.elapsed().as_millis();
    let mut wf: Vec<(&&str, &u64)> = waterfall.iter().collect();
    wf.sort_by_key(|(_, v)| std::cmp::Reverse(**v));
    let wf_str: String = wf
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ");
    info!(
        search_space = n,
        extracted = n_extracted,
        competed = n_competed,
        reported = n_reported,
        elapsed_ms = elapsed,
        "audit: done"
    );
    info!("audit waterfall: {wf_str}");
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> String {
        let dir = std::env::temp_dir().join("mumdia_audit_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_str().unwrap().to_string()
    }

    fn write_lib(path: &str, cids: &[u32], labels: &[&str]) {
        write_table(
            path,
            vec![
                Col::U32("candidate_id".into(), cids.to_vec()),
                Col::Str(
                    "peptidoform".into(),
                    cids.iter().map(|c| format!("PEP{c}")).collect(),
                ),
                Col::I32("charge".into(), cids.iter().map(|_| 2i32).collect()),
                Col::Str(
                    "label".into(),
                    labels.iter().map(|s| s.to_string()).collect(),
                ),
                Col::Str(
                    "protein".into(),
                    cids.iter().map(|_| "sp|X|ECOLI".to_string()).collect(),
                ),
            ],
        )
        .unwrap();
    }
    fn write_cid_only(path: &str, cids: &[u32]) {
        write_table(path, vec![Col::U32("candidate_id".into(), cids.to_vec())]).unwrap();
    }

    #[test]
    fn waterfall_assigns_earliest_loss_per_candidate() {
        // 6 candidates in the library. Fates:
        //  1 target  -> reported          (extract+compete+scored q<=0.01)
        //  2 target  -> failed precursor  (scored q=0.5)
        //  3 target  -> outcompeted       (extract yes, compete no)
        //  4 target  -> no peak group     (not extracted)
        //  5 decoy   -> outcompeted decoy (extract yes, compete no)
        //  6 decoy   -> no peak group     (not extracted)
        let lib = tmp("lib.parquet");
        let psms = tmp("psms.parquet");
        let comp = tmp("comp.parquet");
        let scored = tmp("scored.parquet");
        let out = tmp("candidate_audit.parquet");
        write_lib(
            &lib,
            &[1, 2, 3, 4, 5, 6],
            &["target", "target", "target", "target", "decoy", "decoy"],
        );
        write_cid_only(&psms, &[1, 2, 3, 5]); // 4,6 never extracted
        write_cid_only(&comp, &[1, 2]); // 3,5 outcompeted
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2]),
                Col::F64("q_value".into(), vec![0.001, 0.5]),
            ],
        )
        .unwrap();

        let rows = run(AuditParams {
            library_precursors: &lib,
            psms: &psms,
            competed: &comp,
            scored: &scored,
            out: &out,
            q_threshold: 0.01,
            run_id: "t",
            entrapment_substr: "",
        })
        .unwrap();
        assert_eq!(rows, 6);

        let a = Table::read(&out).unwrap();
        let cid = a.u32("precursor_id").unwrap();
        let reason = a.str("rejection_reason").unwrap();
        let reported = a.bool("reported").unwrap();
        let by: std::collections::HashMap<u32, (String, bool)> = cid
            .iter()
            .cloned()
            .zip(reason.into_iter().zip(reported))
            .collect();
        assert_eq!(by[&1].0, "REPORTED");
        assert!(by[&1].1);
        assert_eq!(by[&2].0, "FAILED_PRECURSOR_FDR");
        assert!(!by[&2].1);
        assert_eq!(by[&3].0, "OUTCOMPETED_BY_TARGET");
        assert_eq!(by[&4].0, "NO_PEAK_GROUP");
        assert_eq!(by[&5].0, "OUTCOMPETED_BY_DECOY");
        assert_eq!(by[&6].0, "NO_PEAK_GROUP");
    }

    #[test]
    fn entrapment_label_from_protein_substring() {
        let lib = tmp("lib2.parquet");
        // one ECOLI, one HUMAN protein
        write_table(
            &lib,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2]),
                Col::Str("peptidoform".into(), vec!["A".into(), "B".into()]),
                Col::I32("charge".into(), vec![2, 3]),
                Col::Str("label".into(), vec!["target".into(), "target".into()]),
                Col::Str(
                    "protein".into(),
                    vec!["sp|X|EFTU_ECOLI".into(), "sp|Y|ALBU_HUMAN".into()],
                ),
            ],
        )
        .unwrap();
        let psms = tmp("psms2.parquet");
        let comp = tmp("comp2.parquet");
        let scored = tmp("scored2.parquet");
        let out = tmp("audit2.parquet");
        write_cid_only(&psms, &[1, 2]);
        write_cid_only(&comp, &[1, 2]);
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2]),
                Col::F64("q_value".into(), vec![0.001, 0.001]),
            ],
        )
        .unwrap();
        run(AuditParams {
            library_precursors: &lib,
            psms: &psms,
            competed: &comp,
            scored: &scored,
            out: &out,
            q_threshold: 0.01,
            run_id: "t",
            entrapment_substr: "_HUMAN",
        })
        .unwrap();
        let a = Table::read(&out).unwrap();
        let entrap = a.bool("entrapment_label").unwrap();
        assert_eq!(entrap, vec![false, true]);
    }
}
