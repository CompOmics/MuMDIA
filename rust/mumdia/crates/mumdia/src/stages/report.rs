//! Human-readable results report: `peptides.tsv` + `proteins.tsv` from the
//! scored PSMs, for users who expect a `report.tsv` analog rather than Parquet.
//!
//! peptides.tsv: one row per confident (target, peptide q <= threshold)
//! peptidoform+charge, best q kept, joined to its quantity. proteins.tsv: one
//! row per confident protein group (pg q <= threshold), joined to its quantity.
//! `run_experiment` is the pooled-rescore counterpart: the same two files for a whole
//! `run-experiment`, selected on the experiment-wide q columns, with one quantity
//! column per run.
use std::collections::{HashMap, HashSet};
use std::io::Write;

use anyhow::{Context, Result};
use mumdia_io::table::TableFile;

pub struct ReportParams<'a> {
    pub scored: &'a str,
    pub peptide_quant: Option<&'a str>,
    pub protein_quant: Option<&'a str>,
    pub out_peptides: &'a str,
    pub out_proteins: &'a str,
    pub q_threshold: f64,
}

/// Stripped amino-acid sequence: drop a DECOY_ prefix and any bracketed/
/// parenthesized modification blocks, keep the residue letters.
fn strip(pf: &str) -> String {
    let s = pf.strip_prefix("DECOY_").unwrap_or(pf);
    let mut out = String::new();
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = (depth - 1).max(0),
            c if depth == 0 && c.is_ascii_alphabetic() => out.push(c),
            _ => {}
        }
    }
    out
}

fn qcell(q: f64) -> String {
    if q.is_nan() {
        String::new()
    } else {
        format!("{q:.1}")
    }
}

/// The match-between-runs acceptance basis of every row: the flag and the q it was
/// accepted at, or all-false and all-NaN when the table has never been through
/// `mumdia mbr`.
///
/// Absent means absent; present means readable. A present column of the wrong type used
/// to be swallowed by an `Err(_) => vec![false; n]` arm, so a boolean written as a
/// nullable dtype or as int8 (which `mbr_worker.py` writes through pandas) silently
/// removed EVERY transfer from both TSVs at exit 0, indistinguishable from an MBR run
/// that transferred nothing while the parquet still showed them (docs/31 F2). This is the
/// absent-versus-malformed rule `quant` and `audit` already follow.
fn transfer_columns(t: &TableFile, n: usize) -> Result<(Vec<bool>, Vec<f64>)> {
    let flag = if t.has_column("is_transferred") {
        t.bool("is_transferred").with_context(|| {
            "report: `is_transferred` is present but not a boolean column; it is written by \
             `mumdia mbr` and decides which rows are reported"
        })?
    } else {
        vec![false; n]
    };
    let q = if t.has_column("transfer_q") {
        t.f64("transfer_q")
            .with_context(|| "report: `transfer_q` is present but not a float column")?
    } else {
        vec![f64::NAN; n]
    };
    if flag.len() != n || q.len() != n {
        anyhow::bail!(
            "report: the transfer columns have {} and {} rows against {n} scored rows",
            flag.len(),
            q.len()
        );
    }
    Ok((flag, q))
}

/// Write peptides.tsv + proteins.tsv from a scored PSM table. Returns
/// (n_peptides, n_protein_groups) at the FDR threshold.
pub fn run(p: ReportParams) -> Result<(u64, u64)> {
    let t = TableFile::open(p.scored)?;
    let pform = t.str("peptidoform")?;
    let charge = t.i32("charge")?;
    let protein = t.str("protein")?;
    let label = t.str("label")?;
    let pep_q = t.f64("peptide_q_value")?;
    let pg = t.str("protein_group")?;
    let pg_q = t.f64("pg_q_value")?;
    let score = t.f64("score")?;
    let n = t.nrows;
    // Match-between-runs acceptance, when this table has been through `mumdia mbr`.
    // MBR lowers the three PSM-level q columns and neither of the two grouped columns
    // this stage filters on, so without this a `mumdia mbr` followed by `mumdia report`
    // showed no transfers at all. Same contract as `quant`: a transfer has already
    // passed `mbr.q_transfer`, and a decoy is still never reported.
    let (is_transferred, transfer_q) = transfer_columns(&t, n)?;
    let accepted =
        |i: usize, q: &[f64]| label[i] == "target" && (is_transferred[i] || q[i] <= p.q_threshold);
    // The acceptance basis is exported with every row (`is_transferred`, `transfer_q`),
    // because the rule above is otherwise invisible in the TSV: a transferred row keeps
    // its grouped q, usually 1.0, and a tighter threshold does not revoke a transfer that
    // already passed `mbr.q_transfer` (docs/29 #19). A transfer is a peptide-level
    // acceptance and not protein-group confidence; a group admitted through one carries
    // the flag so a reader can tell.
    let transfer_cells = |i: usize| -> String {
        if is_transferred[i] {
            let q = if transfer_q[i].is_nan() {
                String::new()
            } else {
                format!("{:.6}", transfer_q[i])
            };
            format!("\ttrue\t{q}")
        } else {
            "\tfalse\t".to_string()
        }
    };

    let pep_quant: HashMap<(String, i32), f64> = match p.peptide_quant {
        Some(path) => {
            let q = TableFile::open(path)?;
            let qp = q.str("peptidoform")?;
            let qc = q.i32("charge")?;
            let qq = q.f64("quantity")?;
            (0..q.nrows)
                .map(|i| ((qp[i].clone(), qc[i]), qq[i]))
                .collect()
        }
        None => HashMap::new(),
    };
    let prot_quant: HashMap<String, f64> = match p.protein_quant {
        Some(path) => {
            let q = TableFile::open(path)?;
            let qg = q.str("protein_group")?;
            let qq = q.f64("quantity")?;
            (0..q.nrows).map(|i| (qg[i].clone(), qq[i])).collect()
        }
        None => HashMap::new(),
    };

    // Peptides, best q first, unique by (peptidoform, charge).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| pep_q[a].total_cmp(&pep_q[b]));
    let mut seen: HashSet<(String, i32)> = HashSet::new();
    let mut seen_strip: HashSet<String> = HashSet::new();
    // Atomic, like every parquet and json artifact. These two TSVs were the only
    // outputs still written straight to their final path, so an interruption left a
    // truncated `peptides.tsv` under the canonical name -- and unlike a truncated
    // parquet, which fails to open because its footer is missing, a short TSV parses
    // perfectly and simply has fewer peptides in it. That is the worst failure shape
    // available: a plausible wrong answer.
    let pep_target = mumdia_io::table::AtomicPath::new(p.out_peptides)?;
    let mut w = std::io::BufWriter::new(std::fs::File::create(pep_target.tmp())?);
    // The row unit here is the precursor (peptidoform + charge), NOT the stripped
    // sequence; the header and the returned count reflect that.
    writeln!(
        w,
        "precursor\tstripped_sequence\tcharge\tprotein\tq_value\tscore\tquantity\t\
         is_transferred\ttransfer_q"
    )?;
    let mut npep = 0u64;
    for &i in &order {
        if !accepted(i, &pep_q) {
            continue;
        }
        seen_strip.insert(strip(&pform[i]));
        let key = (pform[i].clone(), charge[i]);
        if !seen.insert(key.clone()) {
            continue;
        }
        let qv = pep_quant.get(&key).copied().unwrap_or(f64::NAN);
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{:.6}\t{:.4}\t{}{}",
            pform[i],
            strip(&pform[i]),
            charge[i],
            protein[i],
            pep_q[i],
            score[i],
            qcell(qv),
            transfer_cells(i)
        )?;
        npep += 1;
    }
    w.flush()?;
    drop(w);
    pep_target.publish()?;

    // Protein groups, best q first, unique.
    let mut porder: Vec<usize> = (0..n).collect();
    porder.sort_by(|&a, &b| pg_q[a].total_cmp(&pg_q[b]));
    let mut pseen: HashSet<String> = HashSet::new();
    let prot_target = mumdia_io::table::AtomicPath::new(p.out_proteins)?;
    let mut w2 = std::io::BufWriter::new(std::fs::File::create(prot_target.tmp())?);
    writeln!(
        w2,
        "protein_group\tq_value\tquantity\tis_transferred\ttransfer_q"
    )?;
    let mut nprot = 0u64;
    for &i in &porder {
        if !accepted(i, &pg_q) || pg[i].is_empty() {
            continue;
        }
        if !pseen.insert(pg[i].clone()) {
            continue;
        }
        let qv = prot_quant.get(&pg[i]).copied().unwrap_or(f64::NAN);
        writeln!(
            w2,
            "{}\t{:.6}\t{}{}",
            pg[i],
            pg_q[i],
            qcell(qv),
            transfer_cells(i)
        )?;
        nprot += 1;
    }
    w2.flush()?;
    drop(w2);
    prot_target.publish()?;

    tracing::info!(
        precursors = npep,
        stripped_sequences = seen_strip.len() as u64,
        protein_groups = nprot,
        "report: done (peptides.tsv rows are precursors, not stripped sequences)"
    );
    Ok((npep, nprot))
}

/// Inputs for the experiment-wide report `run-experiment` writes at its root.
pub struct ExperimentReportParams<'a> {
    /// The pooled scored table: every run's PSMs with `source` = run index and the
    /// grouped q columns assigned experiment-wide (the MBR-rescued table when MBR ran).
    pub scored: &'a str,
    /// Run names in `source` order; they name the per-run quantity columns.
    pub run_names: &'a [String],
    /// Per-run `peptide_quant.parquet`, in `source` order: the same per-run precursor
    /// quantities the single-run report shows. Empty writes no quantity columns.
    pub peptide_quants: &'a [String],
    /// The cross-run protein matrix (`lfq_maxlfq.parquet`: protein_group, run, quantity),
    /// whose normalised MaxLFQ quantities fill the per-run protein columns.
    pub protein_lfq: Option<&'a str>,
    pub out_peptides: &'a str,
    pub out_proteins: &'a str,
    pub q_threshold: f64,
}

/// The experiment-wide report for a pooled rescore: one `peptides.tsv` and one
/// `proteins.tsv` for the whole experiment.
///
/// Rows are selected on the experiment-wide grouped q columns, the one unit that is
/// valid across a pooled rescore: `rescore` writes `peptide_q_value` and `pg_q_value`
/// to each group's single experiment-wide winner and 1.0 to the rest, so a per-run
/// reading of those columns is diluted by about 1/n_runs and a per-run report is not
/// the right shape (the per-run unit is `run_psm_q`). Each row carries `n_runs`, the
/// number of runs in which the precursor (protein group) has an accepted target PSM
/// on its own `run_psm_q`, and one quantity column per run: `quantity_<run>` for
/// precursors (each run's own quant), `lfq_<run>` for protein groups (the cross-run
/// MaxLFQ matrix). Returns (n_precursors, n_protein_groups) at the threshold.
pub fn run_experiment(p: ExperimentReportParams) -> Result<(u64, u64)> {
    let n_runs = p.run_names.len();
    if !p.peptide_quants.is_empty() && p.peptide_quants.len() != n_runs {
        anyhow::bail!(
            "experiment report: {n_runs} run names but {} per-run quant tables",
            p.peptide_quants.len()
        );
    }
    let t = TableFile::open(p.scored)?;
    let pform = t.str("peptidoform")?;
    let charge = t.i32("charge")?;
    let protein = t.str("protein")?;
    let label = t.str("label")?;
    let pep_q = t.f64("peptide_q_value")?;
    let pg = t.str("protein_group")?;
    let pg_q = t.f64("pg_q_value")?;
    let score = t.f64("score")?;
    let source = t.u32("source")?;
    let run_q = t.f64("run_psm_q")?;
    let n = t.nrows;
    let (is_transferred, transfer_q) = transfer_columns(&t, n)?;
    let accepted =
        |i: usize, q: &[f64]| label[i] == "target" && (is_transferred[i] || q[i] <= p.q_threshold);
    // The acceptance basis is exported with every row (`is_transferred`, `transfer_q`),
    // because the rule above is otherwise invisible in the TSV: a transferred row keeps
    // its grouped q, usually 1.0, and a tighter threshold does not revoke a transfer that
    // already passed `mbr.q_transfer` (docs/29 #19). A transfer is a peptide-level
    // acceptance and not protein-group confidence; a group admitted through one carries
    // the flag so a reader can tell.
    let transfer_cells = |i: usize| -> String {
        if is_transferred[i] {
            let q = if transfer_q[i].is_nan() {
                String::new()
            } else {
                format!("{:.6}", transfer_q[i])
            };
            format!("\ttrue\t{q}")
        } else {
            "\tfalse\t".to_string()
        }
    };

    // Runs in which each precursor / protein group was identified on its own per-run FDR.
    let mut pep_runs: HashMap<(String, i32), Vec<bool>> = HashMap::new();
    let mut pg_runs: HashMap<String, Vec<bool>> = HashMap::new();
    for i in 0..n {
        if !accepted(i, &run_q) {
            continue;
        }
        let s = source[i] as usize;
        if s >= n_runs {
            anyhow::bail!(
                "experiment report: row {i} of {} has source {s} but only {n_runs} run names \
                 were given",
                p.scored
            );
        }
        pep_runs
            .entry((pform[i].clone(), charge[i]))
            .or_insert_with(|| vec![false; n_runs])[s] = true;
        if !pg[i].is_empty() {
            pg_runs
                .entry(pg[i].clone())
                .or_insert_with(|| vec![false; n_runs])[s] = true;
        }
    }
    let count = |v: Option<&Vec<bool>>| v.map(|b| b.iter().filter(|&&x| x).count()).unwrap_or(0);

    // Per-run precursor quantities, one map per run, NaN for an unquantified precursor.
    let mut pep_quant: Vec<HashMap<(String, i32), f64>> =
        Vec::with_capacity(p.peptide_quants.len());
    for path in p.peptide_quants {
        let q = TableFile::open(path)?;
        let qp = q.str("peptidoform")?;
        let qc = q.i32("charge")?;
        let qq = q.f64("quantity")?;
        pep_quant.push(
            (0..q.nrows)
                .map(|i| ((qp[i].clone(), qc[i]), qq[i]))
                .collect(),
        );
    }
    // Cross-run protein quantities: protein_group -> per-run MaxLFQ value, NaN for none.
    let mut prot_lfq: HashMap<String, Vec<f64>> = HashMap::new();
    if let Some(path) = p.protein_lfq {
        let l = TableFile::open(path)?;
        let lg = l.str("protein_group")?;
        let lr = l.i32("run")?;
        let lq = l.f64("quantity")?;
        for i in 0..l.nrows {
            if lr[i] < 0 || lr[i] as usize >= n_runs {
                continue;
            }
            prot_lfq
                .entry(lg[i].clone())
                .or_insert_with(|| vec![f64::NAN; n_runs])[lr[i] as usize] = lq[i];
        }
    }

    // Precursors: best experiment-wide q first, unique by (peptidoform, charge). The
    // winner carries the grouped q; its copies in other runs carry 1.0 and are skipped
    // here but counted in `n_runs` above.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| pep_q[a].total_cmp(&pep_q[b]));
    let mut seen: HashSet<(String, i32)> = HashSet::new();
    let mut seen_strip: HashSet<String> = HashSet::new();
    let pep_target = mumdia_io::table::AtomicPath::new(p.out_peptides)?;
    let mut w = std::io::BufWriter::new(std::fs::File::create(pep_target.tmp())?);
    write!(
        w,
        "precursor\tstripped_sequence\tcharge\tprotein\tq_value\tscore\tn_runs"
    )?;
    if !pep_quant.is_empty() {
        for name in p.run_names {
            write!(w, "\tquantity_{name}")?;
        }
    }
    write!(w, "\tis_transferred\ttransfer_q")?;
    writeln!(w)?;
    let mut npep = 0u64;
    for &i in &order {
        if !accepted(i, &pep_q) {
            continue;
        }
        seen_strip.insert(strip(&pform[i]));
        let key = (pform[i].clone(), charge[i]);
        if !seen.insert(key.clone()) {
            continue;
        }
        write!(
            w,
            "{}\t{}\t{}\t{}\t{:.6}\t{:.4}\t{}",
            pform[i],
            strip(&pform[i]),
            charge[i],
            protein[i],
            pep_q[i],
            score[i],
            count(pep_runs.get(&key))
        )?;
        for m in &pep_quant {
            write!(w, "\t{}", qcell(m.get(&key).copied().unwrap_or(f64::NAN)))?;
        }
        write!(w, "{}", transfer_cells(i))?;
        writeln!(w)?;
        npep += 1;
    }
    w.flush()?;
    drop(w);
    pep_target.publish()?;

    // Protein groups: best experiment-wide q first, unique.
    let mut porder: Vec<usize> = (0..n).collect();
    porder.sort_by(|&a, &b| pg_q[a].total_cmp(&pg_q[b]));
    let mut pseen: HashSet<String> = HashSet::new();
    let prot_target = mumdia_io::table::AtomicPath::new(p.out_proteins)?;
    let mut w2 = std::io::BufWriter::new(std::fs::File::create(prot_target.tmp())?);
    write!(w2, "protein_group\tq_value\tn_runs")?;
    if p.protein_lfq.is_some() {
        for name in p.run_names {
            write!(w2, "\tlfq_{name}")?;
        }
    }
    write!(w2, "\tis_transferred\ttransfer_q")?;
    writeln!(w2)?;
    let mut nprot = 0u64;
    for &i in &porder {
        if !accepted(i, &pg_q) || pg[i].is_empty() {
            continue;
        }
        if !pseen.insert(pg[i].clone()) {
            continue;
        }
        write!(
            w2,
            "{}\t{:.6}\t{}",
            pg[i],
            pg_q[i],
            count(pg_runs.get(&pg[i]))
        )?;
        if p.protein_lfq.is_some() {
            let none = vec![f64::NAN; n_runs];
            for x in prot_lfq.get(&pg[i]).unwrap_or(&none) {
                write!(w2, "\t{}", qcell(*x))?;
            }
        }
        write!(w2, "{}", transfer_cells(i))?;
        writeln!(w2)?;
        nprot += 1;
    }
    w2.flush()?;
    drop(w2);
    prot_target.publish()?;

    tracing::info!(
        precursors = npep,
        stripped_sequences = seen_strip.len() as u64,
        protein_groups = nprot,
        runs = n_runs,
        "report: experiment-wide done (rows are precursors selected on the experiment-wide \
         peptide_q_value; n_runs counts per-run acceptances on run_psm_q)"
    );
    Ok((npep, nprot))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::{write_table, Col};

    #[test]
    fn experiment_report_selects_experiment_wide_and_writes_one_column_per_run() {
        // Two runs. PEPTIDEK/2 is the experiment-wide winner in run 0; its run-1 copy
        // carries the loser's 1.0 but passes on its own run_psm_q, so n_runs = 2 and the
        // precursor is written once. SECONDK/3 exists in run 1 only. The decoy has the
        // best q and score everywhere and must not appear. LATEK is above threshold.
        let scored = tmp("scored_combined.parquet");
        write_table(
            &scored,
            vec![
                Col::Str(
                    "peptidoform".into(),
                    vec![
                        "PEPTIDEK".into(),
                        "PEPTIDEK".into(),
                        "SECONDK".into(),
                        "DECOY_KEDITPEP".into(),
                        "LATEK".into(),
                    ],
                ),
                Col::I32("charge".into(), vec![2, 2, 3, 2, 2]),
                Col::Str(
                    "protein".into(),
                    vec![
                        "P1".into(),
                        "P1".into(),
                        "P2".into(),
                        "DECOY_P1".into(),
                        "P1".into(),
                    ],
                ),
                Col::Str(
                    "label".into(),
                    vec![
                        "target".into(),
                        "target".into(),
                        "target".into(),
                        "decoy".into(),
                        "target".into(),
                    ],
                ),
                Col::F64(
                    "peptide_q_value".into(),
                    vec![0.001, 1.0, 0.005, 0.0001, 0.5],
                ),
                Col::Str(
                    "protein_group".into(),
                    vec![
                        "PG1".into(),
                        "PG1".into(),
                        "PG2".into(),
                        "DECOY_PG1".into(),
                        "PG1".into(),
                    ],
                ),
                Col::F64("pg_q_value".into(), vec![0.002, 1.0, 0.5, 0.0001, 1.0]),
                Col::F64("score".into(), vec![3.5, 3.0, 2.0, 9.9, 0.1]),
                Col::U32("source".into(), vec![0, 1, 1, 0, 0]),
                Col::F64("run_psm_q".into(), vec![0.001, 0.004, 0.005, 0.0001, 0.5]),
            ],
        )
        .unwrap();
        let qa = tmp("a_peptide_quant.parquet");
        write_table(
            &qa,
            vec![
                Col::Str("peptidoform".into(), vec!["PEPTIDEK".into()]),
                Col::I32("charge".into(), vec![2]),
                Col::F64("quantity".into(), vec![100.0]),
            ],
        )
        .unwrap();
        let qb = tmp("b_peptide_quant.parquet");
        write_table(
            &qb,
            vec![
                Col::Str(
                    "peptidoform".into(),
                    vec!["PEPTIDEK".into(), "SECONDK".into()],
                ),
                Col::I32("charge".into(), vec![2, 3]),
                Col::F64("quantity".into(), vec![120.0, 50.0]),
            ],
        )
        .unwrap();
        let lfq = tmp("lfq_maxlfq.parquet");
        write_table(
            &lfq,
            vec![
                Col::Str("protein_group".into(), vec!["PG1".into(), "PG1".into()]),
                Col::I32("run".into(), vec![0, 1]),
                Col::F64("quantity".into(), vec![1000.0, 1200.0]),
            ],
        )
        .unwrap();
        let out = tmp("exp_report");
        std::fs::create_dir_all(&out).unwrap();
        let peptides = format!("{out}/peptides.tsv");
        let proteins = format!("{out}/proteins.tsv");
        let names = vec!["a".to_string(), "b".to_string()];
        let (n_pep, n_prot) = run_experiment(ExperimentReportParams {
            scored: &scored,
            run_names: &names,
            peptide_quants: &[qa, qb],
            protein_lfq: Some(&lfq),
            out_peptides: &peptides,
            out_proteins: &proteins,
            q_threshold: 0.01,
        })
        .unwrap();
        assert_eq!((n_pep, n_prot), (2, 1));

        let text = std::fs::read_to_string(&peptides).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[0],
            "precursor\tstripped_sequence\tcharge\tprotein\tq_value\tscore\tn_runs\tquantity_a\tquantity_b\tis_transferred\ttransfer_q"
        );
        assert_eq!(lines.len(), 3, "header plus two precursors:\n{text}");
        assert!(
            !text.contains("DECOY_") && !text.contains("LATEK"),
            "{text}"
        );
        let pep: Vec<&str> = lines[1].split('\t').collect();
        assert_eq!(
            (pep[0], pep[2], pep[6], pep[7], pep[8]),
            ("PEPTIDEK", "2", "2", "100.0", "120.0")
        );
        let sec: Vec<&str> = lines[2].split('\t').collect();
        // Identified in one run, quantified there only: the run-a cell is empty, not 0.
        assert_eq!(
            (sec[0], sec[6], sec[7], sec[8]),
            ("SECONDK", "1", "", "50.0")
        );

        let prot = std::fs::read_to_string(&proteins).unwrap();
        let plines: Vec<&str> = prot.lines().collect();
        assert_eq!(
            plines[0],
            "protein_group\tq_value\tn_runs\tlfq_a\tlfq_b\tis_transferred\ttransfer_q"
        );
        assert_eq!(
            plines.len(),
            2,
            "PG2 is above threshold, the decoy group is a decoy:\n{prot}"
        );
        let pg: Vec<&str> = plines[1].split('\t').collect();
        assert_eq!(
            (pg[0], pg[2], pg[3], pg[4]),
            ("PG1", "2", "1000.0", "1200.0")
        );
    }

    #[test]
    fn strip_mods_and_decoy() {
        assert_eq!(strip("PEPTIDEK"), "PEPTIDEK");
        assert_eq!(strip("M[Oxidation]EGC[Carbamidomethyl]VDGHK"), "MEGCVDGHK");
        assert_eq!(strip("DECOY_VAVGDGVAK"), "VAVGDGVAK");
    }

    fn tmp(name: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!("mumdia_report_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        dir.join(format!("{n}_{name}"))
            .to_string_lossy()
            .into_owned()
    }

    /// A scored table with one confident target, one target above threshold, and
    /// one decoy that would win on score if the label were ignored.
    fn scored_table() -> String {
        let path = tmp("scored.parquet");
        write_table(
            &path,
            vec![
                Col::Str(
                    "peptidoform".into(),
                    vec![
                        "PEPTIDEK".into(),
                        "M[Oxidation]EGVDGHK".into(),
                        "DECOY_KEDITPEP".into(),
                        "LATEPEPTIDEK".into(),
                    ],
                ),
                Col::I32("charge".into(), vec![2, 3, 2, 2]),
                Col::Str(
                    "protein".into(),
                    vec!["P1".into(), "P2".into(), "DECOY_P1".into(), "P1".into()],
                ),
                Col::Str(
                    "label".into(),
                    vec![
                        "target".into(),
                        "target".into(),
                        "decoy".into(),
                        "target".into(),
                    ],
                ),
                Col::F64("peptide_q_value".into(), vec![0.001, 0.005, 0.0001, 0.5]),
                Col::Str(
                    "protein_group".into(),
                    vec!["PG1".into(), "PG2".into(), "DECOY_PG1".into(), "".into()],
                ),
                Col::F64("pg_q_value".into(), vec![0.002, 0.9, 0.0001, 1.0]),
                Col::F64("score".into(), vec![3.5, 2.5, 9.9, 0.1]),
            ],
        )
        .unwrap();
        path
    }

    #[test]
    fn report_filters_by_label_and_threshold_and_never_writes_a_decoy() {
        let scored = scored_table();
        let out = tmp("report_dir");
        std::fs::create_dir_all(&out).unwrap();
        let peptides = format!("{out}/peptides.tsv");
        let proteins = format!("{out}/proteins.tsv");
        let (n_pep, n_prot) = run(ReportParams {
            scored: &scored,
            peptide_quant: None,
            protein_quant: None,
            out_peptides: &peptides,
            out_proteins: &proteins,
            q_threshold: 0.01,
        })
        .unwrap();

        // Two targets pass; the decoy has the best q and the best score and must
        // still be excluded. A decoy in a user-facing report is not a cosmetic
        // problem: it is reported as an identification.
        assert_eq!((n_pep, n_prot), (2, 1));
        let text = std::fs::read_to_string(&peptides).unwrap();
        assert!(
            !text.contains("DECOY_"),
            "decoy leaked into peptides.tsv:\n{text}"
        );
        assert!(text.contains("PEPTIDEK"));
        assert!(text.contains("M[Oxidation]EGVDGHK"));
        // Above-threshold target excluded.
        assert!(!text.contains("LATEPEPTIDEK"));
        // The stripped column must be the modification-free sequence, since that
        // is the unit `peptide_q_value` controls.
        assert!(
            text.contains("MEGVDGHK"),
            "stripped sequence missing:\n{text}"
        );
        // Quantity is empty when no quant table was supplied, not zero: absence of
        // a measurement is not a measurement of zero. Nothing here is a transfer.
        let data_line = text.lines().nth(1).unwrap();
        let cells: Vec<&str> = data_line.split('\t').collect();
        assert_eq!(
            (cells[6], cells[7], cells[8]),
            ("", "false", ""),
            "expected an empty quantity cell and no transfer: {data_line:?}"
        );

        let prot = std::fs::read_to_string(&proteins).unwrap();
        assert!(prot.contains("PG1"));
        assert!(!prot.contains("DECOY_PG1"));
        // An empty protein_group must not become a row.
        assert_eq!(prot.lines().count(), 2, "header plus one group:\n{prot}");
    }

    /// The same table after `mumdia mbr`: the above-threshold target is an accepted
    /// transfer (`is_transferred`, with the transfer q the worker accepted it at).
    fn scored_table_with_transfer() -> String {
        let path = tmp("scored_mbr.parquet");
        write_table(
            &path,
            vec![
                Col::Str(
                    "peptidoform".into(),
                    vec![
                        "PEPTIDEK".into(),
                        "M[Oxidation]EGVDGHK".into(),
                        "DECOY_KEDITPEP".into(),
                        "LATEPEPTIDEK".into(),
                    ],
                ),
                Col::I32("charge".into(), vec![2, 3, 2, 2]),
                Col::Str(
                    "protein".into(),
                    vec!["P1".into(), "P2".into(), "DECOY_P1".into(), "P1".into()],
                ),
                Col::Str(
                    "label".into(),
                    vec![
                        "target".into(),
                        "target".into(),
                        "decoy".into(),
                        "target".into(),
                    ],
                ),
                Col::F64("peptide_q_value".into(), vec![0.001, 0.005, 0.0001, 0.5]),
                Col::Str(
                    "protein_group".into(),
                    vec!["PG1".into(), "PG2".into(), "DECOY_PG1".into(), "".into()],
                ),
                Col::F64("pg_q_value".into(), vec![0.002, 0.9, 0.0001, 1.0]),
                Col::F64("score".into(), vec![3.5, 2.5, 9.9, 0.1]),
                Col::Bool("is_transferred".into(), vec![false, false, false, true]),
                Col::F64(
                    "transfer_q".into(),
                    vec![f64::NAN, f64::NAN, f64::NAN, 0.004],
                ),
            ],
        )
        .unwrap();
        path
    }

    #[test]
    fn a_malformed_transfer_column_is_an_error_not_a_silent_loss_of_every_transfer() {
        // docs/31 F2: `mbr_worker.py` writes `is_transferred` through pandas, so a
        // nullable dtype or int8 0/1 reaches this stage. The old `Err(_) => vec![false; n]`
        // arm turned that into "no transfers", removing every match-between-runs
        // identification from both TSVs at exit 0 while the parquet still showed them.
        let path = tmp("scored_int_flag.parquet");
        write_table(
            &path,
            vec![
                Col::Str("peptidoform".into(), vec!["PEPTIDEK".into()]),
                Col::I32("charge".into(), vec![2]),
                Col::Str("protein".into(), vec!["P1".into()]),
                Col::Str("label".into(), vec!["target".into()]),
                Col::F64("peptide_q_value".into(), vec![0.5]),
                Col::Str("protein_group".into(), vec!["PG1".into()]),
                Col::F64("pg_q_value".into(), vec![0.5]),
                Col::F64("score".into(), vec![1.0]),
                // The flag as an integer, which is what a pandas round-trip can produce.
                Col::I32("is_transferred".into(), vec![1]),
                Col::F64("transfer_q".into(), vec![0.004]),
            ],
        )
        .unwrap();
        let out = tmp("report_badflag");
        std::fs::create_dir_all(&out).unwrap();
        let e = run(ReportParams {
            scored: &path,
            peptide_quant: None,
            protein_quant: None,
            out_peptides: &format!("{out}/peptides.tsv"),
            out_proteins: &format!("{out}/proteins.tsv"),
            q_threshold: 0.01,
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("is_transferred"), "{e}");
    }

    #[test]
    fn a_table_that_never_saw_mbr_reports_no_transfers_without_complaint() {
        // The absent case stays absent: `mumdia report` on a plain scored table must not
        // start demanding columns only `mumdia mbr` writes.
        let scored = scored_table();
        let out = tmp("report_nombr");
        std::fs::create_dir_all(&out).unwrap();
        let (n_pep, _) = run(ReportParams {
            scored: &scored,
            peptide_quant: None,
            protein_quant: None,
            out_peptides: &format!("{out}/peptides.tsv"),
            out_proteins: &format!("{out}/proteins.tsv"),
            q_threshold: 0.01,
        })
        .unwrap();
        assert_eq!(n_pep, 2);
        let text = std::fs::read_to_string(format!("{out}/peptides.tsv")).unwrap();
        for line in text.lines().skip(1) {
            let c: Vec<&str> = line.split('\t').collect();
            assert_eq!((c[7], c[8]), ("false", ""));
        }
    }

    #[test]
    fn report_exports_the_transfer_basis_and_a_tighter_threshold_does_not_revoke_it() {
        // docs/29 #19: the acceptance rule admits any target flagged `is_transferred`
        // whatever its grouped q, and the TSV has to show that basis rather than a
        // `q_value` of 0.5 next to a 1% threshold with no explanation.
        let scored = scored_table_with_transfer();
        let out = tmp("report_mbr");
        std::fs::create_dir_all(&out).unwrap();
        let peptides = format!("{out}/peptides.tsv");
        let proteins = format!("{out}/proteins.tsv");
        let report = |q: f64| {
            run(ReportParams {
                scored: &scored,
                peptide_quant: None,
                protein_quant: None,
                out_peptides: &peptides,
                out_proteins: &proteins,
                q_threshold: q,
            })
            .unwrap()
        };

        let (n_pep, _) = report(0.01);
        assert_eq!(n_pep, 3, "two confident targets plus the transfer");
        let text = std::fs::read_to_string(&peptides).unwrap();
        let header = text.lines().next().unwrap();
        assert!(
            header.ends_with("\tquantity\tis_transferred\ttransfer_q"),
            "{header}"
        );
        let late = text
            .lines()
            .find(|l| l.starts_with("LATEPEPTIDEK"))
            .expect("the transfer is reported");
        let cells: Vec<&str> = late.split('\t').collect();
        // The grouped q is preserved as it is (0.5), and the basis is next to it.
        assert_eq!(
            (cells[4], cells[7], cells[8]),
            ("0.500000", "true", "0.004000")
        );
        let first = text.lines().nth(1).unwrap();
        let c: Vec<&str> = first.split('\t').collect();
        assert_eq!(
            (c[7], c[8]),
            ("false", ""),
            "a scored row is not a transfer: {first}"
        );

        // Protein rows carry the same two columns; PG1 was admitted on its own q.
        let prot = std::fs::read_to_string(&proteins).unwrap();
        let plines: Vec<&str> = prot.lines().collect();
        assert_eq!(
            plines[0],
            "protein_group\tq_value\tquantity\tis_transferred\ttransfer_q"
        );
        let pg: Vec<&str> = plines[1].split('\t').collect();
        assert_eq!((pg[0], pg[3], pg[4]), ("PG1", "false", ""));

        // Tighter threshold: the 0.005 target drops, the transfer stays. This is the
        // rule the columns exist to make visible, not a new one.
        let (n_pep, _) = report(0.001);
        assert_eq!(n_pep, 2);
        let text = std::fs::read_to_string(&peptides).unwrap();
        assert!(text.contains("LATEPEPTIDEK\t"), "{text}");
        assert!(!text.contains("M[Oxidation]EGVDGHK"), "{text}");
    }

    #[test]
    fn report_writes_header_only_when_nothing_passes() {
        // A run that identifies nothing at the threshold still has to produce the
        // files, with headers, so a downstream reader fails on empty data rather
        // than on a missing file. This is the state the fixture run hits for
        // proteins.tsv, where 16 groups cannot reach 1 percent FDR.
        let scored = scored_table();
        let out = tmp("report_empty");
        std::fs::create_dir_all(&out).unwrap();
        let peptides = format!("{out}/peptides.tsv");
        let proteins = format!("{out}/proteins.tsv");
        let (n_pep, n_prot) = run(ReportParams {
            scored: &scored,
            peptide_quant: None,
            protein_quant: None,
            out_peptides: &peptides,
            out_proteins: &proteins,
            q_threshold: 0.0,
        })
        .unwrap();
        assert_eq!((n_pep, n_prot), (0, 0));
        for path in [&peptides, &proteins] {
            let text = std::fs::read_to_string(path).unwrap();
            assert_eq!(text.lines().count(), 1, "expected header only in {path}");
            assert!(text.starts_with("precursor\t") || text.starts_with("protein_group\t"));
        }
    }

    #[test]
    fn report_joins_quantities_and_leaves_unquantified_rows_empty() {
        let scored = scored_table();
        let quant = tmp("peptide_quant.parquet");
        // Only the first passing precursor is quantified.
        write_table(
            &quant,
            vec![
                Col::Str("peptidoform".into(), vec!["PEPTIDEK".into()]),
                Col::I32("charge".into(), vec![2]),
                Col::F64("quantity".into(), vec![1234.5]),
            ],
        )
        .unwrap();
        let out = tmp("report_quant");
        std::fs::create_dir_all(&out).unwrap();
        let peptides = format!("{out}/peptides.tsv");
        let proteins = format!("{out}/proteins.tsv");
        run(ReportParams {
            scored: &scored,
            peptide_quant: Some(&quant),
            protein_quant: None,
            out_peptides: &peptides,
            out_proteins: &proteins,
            q_threshold: 0.01,
        })
        .unwrap();
        let text = std::fs::read_to_string(&peptides).unwrap();
        let mut quantified = 0;
        let mut empty = 0;
        for line in text.lines().skip(1) {
            // The quantity column, by position: the transfer columns follow it.
            let cell = line.split('\t').nth(6).unwrap();
            if cell.is_empty() {
                empty += 1;
            } else {
                assert_eq!(cell, "1234.5");
                quantified += 1;
            }
        }
        assert_eq!((quantified, empty), (1, 1));
    }
}
