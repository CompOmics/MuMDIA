//! Candidate audit `mumdia audit` (sensitivity program,
//! docs/20_sensitivity_and_quantification_playbook.md, backlog P0.3 / P0.4).
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
//! refines the extract-stage bucket (see `load_extract_reasons`).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{Context, Result};
use mumdia_core::rejection::RejectionReason;
use mumdia_io::table::{write_table, Col, TableFile};
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
    /// Precursor q-value threshold for `passed_precursor_fdr` / `reported`, applied to
    /// the scored table's `precursor_q` (the PSM `q_value` only on an older table
    /// without that column; `<out>.metrics.json` records which, as `q_unit`).
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
    if let Ok(t) = TableFile::open(&sidecar) {
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
    let lib = TableFile::open(p.library_precursors)
        .with_context(|| format!("audit: reading library precursors {}", p.library_precursors))?;
    let cid = lib.u32("candidate_id")?;
    let pform = lib.str("peptidoform")?;
    let charge = lib.i32("charge")?;
    let label = lib.str("label")?;
    let protein = lib.str("protein")?;
    let n = cid.len();

    // Survivor sets keyed by candidate_id from each downstream artifact. Projected: the
    // competed and scored artifacts carry ~390 feature columns and the audit wants three of
    // them, so an unprojected read decodes (and holds) the whole feature matrix for nothing.
    let extracted: HashSet<u32> = TableFile::open(p.psms)?
        .u32("candidate_id")?
        .into_iter()
        .collect();
    let competed: HashSet<u32> = TableFile::open(p.competed)?
        .u32("candidate_id")?
        .into_iter()
        .collect();
    let scored_t = TableFile::open(p.scored)?;
    // One run only. Every survivor set and q lookup here is keyed by candidate_id, so a
    // pooled table (several `source` values) would overwrite one run's q with another's
    // and attribute the last run's fate to every run (docs/29 #16). `run-experiment`
    // writes per-run split tables; audit those.
    if let Some(n_sources) = crate::stages::quant::pooled_source_count(&scored_t, p.scored)? {
        if n_sources > 1 {
            anyhow::bail!(
                "audit: {} is a pooled scored table with {n_sources} sources; the audit is \
                 per run and keys on candidate_id, so pass one run's split table \
                 (<experiment>/<run>/scored.parquet)",
                p.scored
            );
        }
    }
    let scored_cid = scored_t.u32("candidate_id")?;
    // The gate is named `passed_precursor_fdr`, so it reads the precursor q. It read
    // the PSM `q_value` (docs/29 #16), a different unit: a PSM can pass at 1% while
    // its precursor group does not, and the other way round. Older scored tables have
    // no `precursor_q`; there the PSM q is used and the metrics say so.
    let (scored_q, q_unit) = if scored_t.has_column("precursor_q") {
        // Present means present: a column of the wrong type is an error, not a reason to
        // read another unit in its place (docs/30 R7, the absent-versus-malformed rule
        // quant applies to `source`).
        let v = scored_t
            .f64("precursor_q")
            .with_context(|| format!("audit: reading precursor_q from {}", p.scored))?;
        (v, "precursor_q")
    } else {
        tracing::warn!(
            scored = p.scored,
            "audit: no `precursor_q` column; the precursor gate falls back to the PSM \
             q_value, which is not the same unit"
        );
        (scored_t.f64("q_value")?, "q_value")
    };
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
            // Extraction produced no accepted row for this candidate. Refine with the
            // in-extract audit sidecar if present; otherwise the generic bucket, which
            // says only that the candidate did not survive extraction.
            match extract_reasons.get(&c).map(String::as_str) {
                Some("NO_FRAGMENT_TRACES") => RejectionReason::NoFragmentTraces,
                Some("NO_VALID_FRAGMENTS") => RejectionReason::NoValidFragments,
                Some("PEAK_NOT_SELECTED") => RejectionReason::PeakNotSelected,
                Some("RT_PRUNED") => RejectionReason::RtPruned,
                Some("WRONG_ISOLATION_WINDOW") => RejectionReason::WrongIsolationWindow,
                _ => RejectionReason::DidNotSurviveExtraction,
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
        } else if is_decoy {
            // Passed every gate, and the report never writes a decoy (docs/30 R7).
            RejectionReason::RemovedDuringReporting
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
        // One definition of "reported": the rejection reason. The flag used to repeat
        // the precursor gate alone, so a row could read `reported = true` next to
        // `FAILED_PEPTIDE_FDR`, and a decoy could be reported (docs/30 R7). The gate
        // diagnostics keep their own columns above.
        f_reported.push(reason == RejectionReason::Reported);
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
        "q_unit": q_unit,
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
        // Unique per process: a fixed name races when two `cargo test` runs share a
        // machine, which is the convention docs/14 states and four other test modules
        // already follow.
        let dir = std::env::temp_dir().join(format!("mumdia_audit_test_{}", std::process::id()));
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
        //  4 target  -> did not survive extraction (not extracted)
        //  5 decoy   -> outcompeted decoy (extract yes, compete no)
        //  6 decoy   -> did not survive extraction (not extracted)
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

        let a = TableFile::open(&out).unwrap();
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
        assert_eq!(by[&4].0, "DID_NOT_SURVIVE_EXTRACTION");
        assert_eq!(by[&5].0, "OUTCOMPETED_BY_DECOY");
        assert_eq!(by[&6].0, "DID_NOT_SURVIVE_EXTRACTION");
    }

    #[test]
    fn the_precursor_gate_reads_precursor_q_not_psm_q() {
        // Two candidates whose PSM and precursor q lie on opposite sides of 1%. The
        // label `passed_precursor_fdr` has to follow `precursor_q` (docs/29 #16).
        let lib = tmp("lib_q.parquet");
        let psms = tmp("psms_q.parquet");
        let comp = tmp("comp_q.parquet");
        let scored = tmp("scored_q.parquet");
        let out = tmp("audit_q.parquet");
        write_lib(&lib, &[1, 2], &["target", "target"]);
        write_cid_only(&psms, &[1, 2]);
        write_cid_only(&comp, &[1, 2]);
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2]),
                Col::F64("q_value".into(), vec![0.001, 0.5]),
                Col::F64("precursor_q".into(), vec![0.5, 0.001]),
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
            entrapment_substr: "",
        })
        .unwrap();
        let a = TableFile::open(&out).unwrap();
        let cid = a.u32("precursor_id").unwrap();
        let reason = a.str("rejection_reason").unwrap();
        let passed = a.bool("passed_precursor_fdr").unwrap();
        let by: std::collections::HashMap<u32, (String, bool)> = cid
            .iter()
            .cloned()
            .zip(reason.into_iter().zip(passed))
            .collect();
        // PSM q 0.001 but precursor q 0.5: fails the precursor gate.
        assert_eq!(by[&1], ("FAILED_PRECURSOR_FDR".to_string(), false));
        // PSM q 0.5 but precursor q 0.001: passes it.
        assert_eq!(by[&2], ("REPORTED".to_string(), true));
        let m: serde_json::Value =
            mumdia_io::json::read_json(&format!("{out}.metrics.json")).unwrap();
        assert_eq!(m["q_unit"], "precursor_q");
    }

    #[test]
    fn the_reported_flag_follows_the_reason_and_the_report_rules() {
        // docs/30 R7: a target passing the precursor gate but not the peptide gate, a
        // target passing both, and a decoy passing both. Only the second is reported,
        // the flag says so, and the metrics count the same row.
        let lib = tmp("lib_rep.parquet");
        let psms = tmp("psms_rep.parquet");
        let comp = tmp("comp_rep.parquet");
        let scored = tmp("scored_rep.parquet");
        let out = tmp("audit_rep.parquet");
        write_lib(&lib, &[1, 2, 3], &["target", "target", "decoy"]);
        write_cid_only(&psms, &[1, 2, 3]);
        write_cid_only(&comp, &[1, 2, 3]);
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2, 3]),
                Col::F64("q_value".into(), vec![0.001, 0.001, 0.001]),
                Col::F64("precursor_q".into(), vec![0.001, 0.001, 0.001]),
                Col::F64("peptide_q_value".into(), vec![0.5, 0.001, 0.001]),
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
            entrapment_substr: "",
        })
        .unwrap();
        let a = TableFile::open(&out).unwrap();
        let cid = a.u32("precursor_id").unwrap();
        let reason = a.str("rejection_reason").unwrap();
        let reported = a.bool("reported").unwrap();
        let by: std::collections::HashMap<u32, (String, bool)> = cid
            .iter()
            .cloned()
            .zip(reason.into_iter().zip(reported))
            .collect();
        assert_eq!(by[&1], ("FAILED_PEPTIDE_FDR".to_string(), false));
        assert_eq!(by[&2], ("REPORTED".to_string(), true));
        assert_eq!(by[&3], ("REMOVED_DURING_REPORTING".to_string(), false));
        let m: serde_json::Value =
            mumdia_io::json::read_json(&format!("{out}.metrics.json")).unwrap();
        assert_eq!(m["reported"], 1);
    }

    #[test]
    fn a_present_but_malformed_precursor_q_column_is_an_error_not_a_fallback() {
        let lib = tmp("lib_bad.parquet");
        let psms = tmp("psms_bad.parquet");
        let comp = tmp("comp_bad.parquet");
        let scored = tmp("scored_bad.parquet");
        let out = tmp("audit_bad.parquet");
        write_lib(&lib, &[1], &["target"]);
        write_cid_only(&psms, &[1]);
        write_cid_only(&comp, &[1]);
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1]),
                Col::F64("q_value".into(), vec![0.001]),
                Col::I32("precursor_q".into(), vec![0]),
            ],
        )
        .unwrap();
        let e = run(AuditParams {
            library_precursors: &lib,
            psms: &psms,
            competed: &comp,
            scored: &scored,
            out: &out,
            q_threshold: 0.01,
            run_id: "t",
            entrapment_substr: "",
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("precursor_q"), "{e}");
    }

    #[test]
    fn a_pooled_scored_table_is_refused() {
        // Keyed by candidate_id alone, a two-source table would let the second run's q
        // overwrite the first's. The experiment writes split tables; audit those.
        let lib = tmp("lib_pool.parquet");
        let psms = tmp("psms_pool.parquet");
        let comp = tmp("comp_pool.parquet");
        let scored = tmp("scored_pool.parquet");
        let out = tmp("audit_pool.parquet");
        write_lib(&lib, &[1], &["target"]);
        write_cid_only(&psms, &[1]);
        write_cid_only(&comp, &[1]);
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1, 1]),
                Col::U32("source".into(), vec![0, 1]),
                Col::F64("q_value".into(), vec![0.001, 0.5]),
                Col::F64("precursor_q".into(), vec![0.001, 0.5]),
            ],
        )
        .unwrap();
        let e = run(AuditParams {
            library_precursors: &lib,
            psms: &psms,
            competed: &comp,
            scored: &scored,
            out: &out,
            q_threshold: 0.01,
            run_id: "t",
            entrapment_substr: "",
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("pooled") && e.contains("2 sources"), "{e}");
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
        let a = TableFile::open(&out).unwrap();
        let entrap = a.bool("entrapment_label").unwrap();
        assert_eq!(entrap, vec![false, true]);
    }
}
