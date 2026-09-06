//! Human-readable results report: `peptides.tsv` + `proteins.tsv` from the
//! scored PSMs, for users who expect a `report.tsv` analog rather than Parquet.
//!
//! peptides.tsv: one row per confident (target, peptide q <= threshold)
//! peptidoform+charge, best q kept, joined to its quantity. proteins.tsv: one
//! row per confident protein group (pg q <= threshold), joined to its quantity.
use std::collections::{HashMap, HashSet};
use std::io::Write;

use anyhow::Result;
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
    let is_transferred: Vec<bool> = match t.bool("is_transferred") {
        Ok(v) => v,
        Err(_) => vec![false; n],
    };
    let accepted =
        |i: usize, q: &[f64]| label[i] == "target" && (is_transferred[i] || q[i] <= p.q_threshold);

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
        "precursor\tstripped_sequence\tcharge\tprotein\tq_value\tscore\tquantity"
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
            "{}\t{}\t{}\t{}\t{:.6}\t{:.4}\t{}",
            pform[i],
            strip(&pform[i]),
            charge[i],
            protein[i],
            pep_q[i],
            score[i],
            qcell(qv)
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
    writeln!(w2, "protein_group\tq_value\tquantity")?;
    let mut nprot = 0u64;
    for &i in &porder {
        if !accepted(i, &pg_q) || pg[i].is_empty() {
            continue;
        }
        if !pseen.insert(pg[i].clone()) {
            continue;
        }
        let qv = prot_quant.get(&pg[i]).copied().unwrap_or(f64::NAN);
        writeln!(w2, "{}\t{:.6}\t{}", pg[i], pg_q[i], qcell(qv))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::{write_table, Col};

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
        // a measurement is not a measurement of zero.
        let data_line = text.lines().nth(1).unwrap();
        assert!(
            data_line.ends_with('\t'),
            "expected an empty quantity cell: {data_line:?}"
        );

        let prot = std::fs::read_to_string(&proteins).unwrap();
        assert!(prot.contains("PG1"));
        assert!(!prot.contains("DECOY_PG1"));
        // An empty protein_group must not become a row.
        assert_eq!(prot.lines().count(), 2, "header plus one group:\n{prot}");
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
            let cell = line.rsplit('\t').next().unwrap();
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
