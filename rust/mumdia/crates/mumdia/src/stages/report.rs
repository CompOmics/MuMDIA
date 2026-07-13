//! Human-readable results report: `peptides.tsv` + `proteins.tsv` from the
//! scored PSMs, for users who expect a `report.tsv` analog rather than Parquet.
//!
//! peptides.tsv: one row per confident (target, peptide q <= threshold)
//! peptidoform+charge, best q kept, joined to its quantity. proteins.tsv: one
//! row per confident protein group (pg q <= threshold), joined to its quantity.
use std::collections::{HashMap, HashSet};
use std::io::Write;

use anyhow::Result;
use mumdia_io::table::Table;

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
    let t = Table::read(p.scored)?;
    let pform = t.str("peptidoform")?;
    let charge = t.i32("charge")?;
    let protein = t.str("protein")?;
    let label = t.str("label")?;
    let pep_q = t.f64("peptide_q_value")?;
    let pg = t.str("protein_group")?;
    let pg_q = t.f64("pg_q_value")?;
    let score = t.f64("score")?;
    let n = t.nrows;

    let pep_quant: HashMap<(String, i32), f64> = match p.peptide_quant {
        Some(path) => {
            let q = Table::read(path)?;
            let qp = q.str("peptidoform")?;
            let qc = q.i32("charge")?;
            let qq = q.f64("quantity")?;
            (0..q.nrows).map(|i| ((qp[i].clone(), qc[i]), qq[i])).collect()
        }
        None => HashMap::new(),
    };
    let prot_quant: HashMap<String, f64> = match p.protein_quant {
        Some(path) => {
            let q = Table::read(path)?;
            let qg = q.str("protein_group")?;
            let qq = q.f64("quantity")?;
            (0..q.nrows).map(|i| (qg[i].clone(), qq[i])).collect()
        }
        None => HashMap::new(),
    };

    // Peptides, best q first, unique by (peptidoform, charge).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| pep_q[a].partial_cmp(&pep_q[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut seen: HashSet<(String, i32)> = HashSet::new();
    let mut w = std::io::BufWriter::new(std::fs::File::create(p.out_peptides)?);
    writeln!(w, "peptide\tstripped_sequence\tcharge\tprotein\tq_value\tscore\tquantity")?;
    let mut npep = 0u64;
    for &i in &order {
        if label[i] != "target" || !(pep_q[i] <= p.q_threshold) {
            continue;
        }
        let key = (pform[i].clone(), charge[i]);
        if !seen.insert(key.clone()) {
            continue;
        }
        let qv = pep_quant.get(&key).copied().unwrap_or(f64::NAN);
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{:.6}\t{:.4}\t{}",
            pform[i], strip(&pform[i]), charge[i], protein[i], pep_q[i], score[i], qcell(qv)
        )?;
        npep += 1;
    }
    w.flush()?;

    // Protein groups, best q first, unique.
    let mut porder: Vec<usize> = (0..n).collect();
    porder.sort_by(|&a, &b| pg_q[a].partial_cmp(&pg_q[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut pseen: HashSet<String> = HashSet::new();
    let mut w2 = std::io::BufWriter::new(std::fs::File::create(p.out_proteins)?);
    writeln!(w2, "protein_group\tq_value\tquantity")?;
    let mut nprot = 0u64;
    for &i in &porder {
        if label[i] != "target" || pg[i].is_empty() || !(pg_q[i] <= p.q_threshold) {
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

    Ok((npep, nprot))
}

#[cfg(test)]
mod tests {
    use super::strip;
    #[test]
    fn strip_mods_and_decoy() {
        assert_eq!(strip("PEPTIDEK"), "PEPTIDEK");
        assert_eq!(strip("M[Oxidation]EGC[Carbamidomethyl]VDGHK"), "MEGCVDGHK");
        assert_eq!(strip("DECOY_VAVGDGVAK"), "VAVGDGVAK");
    }
}
