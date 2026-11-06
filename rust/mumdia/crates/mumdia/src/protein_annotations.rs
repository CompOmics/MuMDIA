//! The protein annotation side table (`protein_annotations.parquet`, docs/15).
//!
//! A library's `protein` column is the grouping key every later table inherits: the scored
//! tables' `protein` and `protein_group`, quant, the cross-run LFQ and the reports. For an
//! imported library it holds UniProt entry names (`ALBU_HUMAN`), for a FASTA-built one the
//! header's first word (`sp|P02768|ALBU_HUMAN`), so the accessions and gene names users map
//! results with were not in any output. This table keeps them beside the key without
//! changing it: one row per `protein` key, giving its accessions, entry names, genes and
//! description. `report` joins it to append `protein_ids`, `protein_names` and `genes` to
//! `peptides.tsv`, `precursors.tsv` and `proteins.tsv`; nothing else reads it, so every
//! identification, q value and quantity is the same with or without it, and without it the
//! TSVs keep their bytes.
//!
//! Where it comes from: `scripts/import_diann_lib.py` writes one row per distinct library
//! `protein` string, and `digest` one row per FASTA protein ([`write_from_fasta`]), each
//! beside its output. The engine finds it next to the precursor table it is given, or
//! through `--lib-annotations`, and copies it into the output directory so a later
//! `mumdia report` finds it there ([`publish`]).
//!
//! The join: a key that is in the table gives its row; a key that is not (a protein string
//! of several FASTA proteins, one per shared peptide) is split on `;` and the rows of its
//! members are concatenated, each value once, in member order. A key with no row at all
//! (the `UNASSIGNED` group of an imported library) has empty cells.

use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use mumdia_io::table::{write_table, Col, TableFile};

/// The table's file name, beside a library's precursor table and in an output directory.
pub const FILE_NAME: &str = "protein_annotations.parquet";

/// The TSV columns the join appends, after every other column so a reader that addresses
/// the existing columns by position is unaffected.
pub const TSV_HEADER: &str = "\tprotein_ids\tprotein_names\tgenes";

/// The columns a table must carry; `description` is written but not required.
const REQUIRED: [&str; 4] = ["protein", "protein_ids", "protein_names", "genes"];

/// One protein key's annotation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Annotation {
    /// Accessions, `;`-separated.
    pub protein_ids: String,
    /// UniProt entry names, `;`-separated.
    pub protein_names: String,
    /// Gene names, `;`-separated.
    pub genes: String,
    /// Free text (a FASTA header's description); not reported.
    pub description: String,
}

/// (accession, entry name) of one FASTA header word or protein identifier.
///
/// `db|ACCESSION|ENTRY_NAME` (UniProt `sp|` / `tr|`, and the same with a prefix such as
/// `CON_sp|` or `ENTRAP_sp|`) gives its two fields; any other word is its own accession,
/// with no entry name. `scripts/_lib_io.py` (`uniprot_fields`) applies the same rule.
pub fn uniprot_fields(token: &str) -> (String, String) {
    let token = token.trim();
    let parts: Vec<&str> = token.split('|').collect();
    if parts.len() >= 3 && !parts[1].is_empty() {
        (parts[1].to_string(), parts[2].to_string())
    } else {
        (token.to_string(), String::new())
    }
}

/// The key and annotation of one FASTA header (the line without its `>`).
///
/// The key is the first word, which is what `digest` writes as the protein. The gene is the
/// UniProt `GN=` field, the description the text before ` OS=` (all of it when there is
/// none). `None` for a header with no word.
pub fn from_fasta_header(header: &str) -> Option<(String, Annotation)> {
    let header = header.trim();
    let key = header.split_whitespace().next()?.to_string();
    let rest = header[key.len()..].trim();
    let (protein_ids, protein_names) = uniprot_fields(&key);
    let genes = rest
        .split_whitespace()
        .find_map(|w| w.strip_prefix("GN="))
        .unwrap_or("")
        .to_string();
    let description = if rest.starts_with("OS=") {
        ""
    } else {
        rest.find(" OS=").map_or(rest, |i| rest[..i].trim())
    };
    Some((
        key,
        Annotation {
            protein_ids,
            protein_names,
            genes,
            description: description.to_string(),
        },
    ))
}

/// Write the annotation table of every protein of `fasta` to `out`, one row per header key
/// (the first header of a repeated key wins), ascending by key. Returns the row count.
pub fn write_from_fasta(fasta: &str, out: &str) -> Result<u64> {
    let f = std::fs::File::open(fasta).with_context(|| format!("reading fasta {fasta}"))?;
    let mut rows: BTreeMap<String, Annotation> = BTreeMap::new();
    for line in std::io::BufReader::new(f).lines() {
        let line = line.with_context(|| format!("reading fasta {fasta}"))?;
        if let Some(h) = line.strip_prefix('>') {
            if let Some((key, a)) = from_fasta_header(h) {
                rows.entry(key).or_insert(a);
            }
        }
    }
    write(&rows, out)
}

/// Write `rows` as the annotation table at `out` (keys ascend, as a `BTreeMap` iterates).
pub fn write(rows: &BTreeMap<String, Annotation>, out: &str) -> Result<u64> {
    let mut cols: [Vec<String>; 5] = Default::default();
    for (k, a) in rows {
        cols[0].push(k.clone());
        cols[1].push(a.protein_ids.clone());
        cols[2].push(a.protein_names.clone());
        cols[3].push(a.genes.clone());
        cols[4].push(a.description.clone());
    }
    let [p, i, n, g, d] = cols;
    write_table(
        out,
        vec![
            Col::Str("protein".into(), p),
            Col::Str("protein_ids".into(), i),
            Col::Str("protein_names".into(), n),
            Col::Str("genes".into(), g),
            Col::Str("description".into(), d),
        ],
    )
}

/// `items`, each non-empty value once, in first-occurrence order, `;`-joined.
fn join_unique<'a>(items: impl IntoIterator<Item = &'a str>) -> String {
    let mut seen: Vec<&str> = Vec::new();
    for v in items.into_iter().flat_map(|s| s.split(';')) {
        let v = v.trim();
        if !v.is_empty() && !seen.contains(&v) {
            seen.push(v);
        }
    }
    seen.join(";")
}

/// A cell for a TSV: tabs and line breaks would split the row, so they become spaces.
fn tsv_safe(s: &str) -> String {
    s.chars()
        .map(|c| {
            if matches!(c, '\t' | '\n' | '\r') {
                ' '
            } else {
                c
            }
        })
        .collect()
}

/// The table, loaded for the report's join.
#[derive(Clone, Debug, Default)]
pub struct ProteinAnnotations {
    rows: HashMap<String, Annotation>,
}

impl ProteinAnnotations {
    /// Read the table at `path`, refusing one without the `protein`, `protein_ids`,
    /// `protein_names` and `genes` columns. A key that appears twice has the values of both
    /// rows, each once.
    pub fn load(path: &str) -> Result<ProteinAnnotations> {
        let t = TableFile::open(path)
            .with_context(|| format!("reading the protein annotations {path}"))?;
        for c in REQUIRED {
            if !t.has_column(c) {
                bail!(
                    "{path}: not a protein annotation table (no `{c}` column; it needs {})",
                    REQUIRED.join(", ")
                );
            }
        }
        let key = t.str("protein")?;
        let ids = t.str("protein_ids")?;
        let names = t.str("protein_names")?;
        let genes = t.str("genes")?;
        let mut rows: HashMap<String, Annotation> = HashMap::with_capacity(key.len());
        for (((k, i), n), g) in key.into_iter().zip(ids).zip(names).zip(genes) {
            match rows.get_mut(&k) {
                Some(a) => {
                    a.protein_ids = join_unique([a.protein_ids.as_str(), i.as_str()]);
                    a.protein_names = join_unique([a.protein_names.as_str(), n.as_str()]);
                    a.genes = join_unique([a.genes.as_str(), g.as_str()]);
                }
                None => {
                    rows.insert(
                        k,
                        Annotation {
                            protein_ids: i,
                            protein_names: n,
                            genes: g,
                            description: String::new(),
                        },
                    );
                }
            }
        }
        Ok(ProteinAnnotations { rows })
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The annotation of a `protein` (or `protein_group`) cell: its own row, else the rows
    /// of its `;`-separated members, concatenated in member order.
    pub fn lookup(&self, protein: &str) -> Annotation {
        if let Some(a) = self.rows.get(protein) {
            return a.clone();
        }
        let members: Vec<&Annotation> = protein
            .split(';')
            .filter_map(|m| self.rows.get(m.trim()))
            .collect();
        Annotation {
            protein_ids: join_unique(members.iter().map(|a| a.protein_ids.as_str())),
            protein_names: join_unique(members.iter().map(|a| a.protein_names.as_str())),
            genes: join_unique(members.iter().map(|a| a.genes.as_str())),
            description: String::new(),
        }
    }

    /// The three cells [`TSV_HEADER`] names, each with its leading tab.
    pub fn cells(&self, protein: &str) -> String {
        let a = self.lookup(protein);
        format!(
            "\t{}\t{}\t{}",
            tsv_safe(&a.protein_ids),
            tsv_safe(&a.protein_names),
            tsv_safe(&a.genes)
        )
    }
}

/// `<directory of path>/protein_annotations.parquet`.
pub fn beside(path: &str) -> PathBuf {
    Path::new(path)
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join(FILE_NAME)
}

/// The annotation table of a search: `explicit` (`--lib-annotations`), which must exist,
/// else the one beside `lib_precursors` when there is one. `None` when there is neither.
pub fn find(explicit: Option<&str>, lib_precursors: Option<&str>) -> Result<Option<String>> {
    if let Some(e) = explicit {
        if !Path::new(e).is_file() {
            bail!("--lib-annotations {e}: no such file");
        }
        return Ok(Some(e.to_string()));
    }
    Ok(lib_precursors
        .map(beside)
        .filter(|b| b.is_file())
        .map(|b| b.display().to_string()))
}

/// [`find`], validated: a table that is there but malformed is refused before the search
/// starts rather than at its report.
pub fn resolve_input(
    explicit: Option<&str>,
    lib_precursors: Option<&str>,
) -> Result<Option<String>> {
    let found = find(explicit, lib_precursors)?;
    if let Some(path) = &found {
        let n = ProteinAnnotations::load(path)?.len();
        tracing::info!(
            path = %path,
            proteins = n,
            "protein annotations: the TSVs will carry protein_ids, protein_names and genes"
        );
    }
    Ok(found)
}

/// The table a search's report joins: [`resolve_input`]'s, else, for a FASTA search (no
/// `lib_precursors`), the one `digest` wrote into `out_dir`.
pub fn search_source(
    resolved: Option<&str>,
    lib_precursors: Option<&str>,
    out_dir: &str,
) -> Option<String> {
    match resolved {
        Some(s) => Some(s.to_string()),
        None if lib_precursors.is_none() => {
            Some(Path::new(out_dir).join(FILE_NAME).display().to_string())
        }
        None => None,
    }
}

/// Publish a search's table into its output directory ([`publish`]) and record it in the
/// manifest, as written by `digest` (a FASTA search) or taken from the library input.
pub fn publish_recorded(
    src: &str,
    out_dir: &str,
    library_input: bool,
    config_hash: &str,
    man: Option<&mut mumdia_core::manifest::Manifest>,
) -> Result<String> {
    let (dst, n) = publish(src, out_dir)?;
    if let Some(m) = man {
        m.record(mumdia_io::record_artifact(
            mumdia_core::schema::artifact::PROTEIN_ANNOTATIONS.0,
            mumdia_core::schema::artifact::PROTEIN_ANNOTATIONS,
            &dst,
            n,
            if library_input {
                "library-input"
            } else {
                "digest"
            },
            config_hash,
        )?);
    }
    Ok(dst)
}

/// Whether two paths name the same file (both must exist to compare by identity).
fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// Validate the table at `src` and copy it, byte for byte, to `<out_dir>/protein_annotations.parquet`
/// (nothing to copy when it is that file already). Returns the copy's path and row count.
pub fn publish(src: &str, out_dir: &str) -> Result<(String, u64)> {
    let n = ProteinAnnotations::load(src)?.len() as u64;
    let dst = Path::new(out_dir).join(FILE_NAME);
    let dst_s = dst.display().to_string();
    if !same_file(Path::new(src), &dst) {
        let target = mumdia_io::table::AtomicPath::new(&dst_s)?;
        std::fs::copy(src, target.tmp())
            .with_context(|| format!("copying the protein annotations {src} to {dst_s}"))?;
        target.publish()?;
    }
    Ok((dst_s, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("mumdia_pa_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.display().to_string()
    }

    #[test]
    fn uniprot_headers_give_accession_entry_name_gene_and_description() {
        let (k, a) = from_fasta_header(
            "sp|P02768|ALBU_HUMAN Serum albumin OS=Homo sapiens OX=9606 GN=ALB PE=1 SV=2",
        )
        .unwrap();
        assert_eq!(k, "sp|P02768|ALBU_HUMAN");
        assert_eq!(a.protein_ids, "P02768");
        assert_eq!(a.protein_names, "ALBU_HUMAN");
        assert_eq!(a.genes, "ALB");
        assert_eq!(a.description, "Serum albumin");
        let (_, a) = from_fasta_header(
            "tr|A0A024R161|A0A024R161_HUMAN Guanine OS=Homo sapiens GN=DNAJC25-GNG10",
        )
        .unwrap();
        assert_eq!(
            (
                a.protein_ids.as_str(),
                a.protein_names.as_str(),
                a.genes.as_str()
            ),
            ("A0A024R161", "A0A024R161_HUMAN", "DNAJC25-GNG10")
        );
    }

    #[test]
    fn prefixed_and_plain_headers() {
        // A contaminant or entrapment prefix keeps the UniProt fields; the key keeps the prefix.
        let (k, a) =
            from_fasta_header("CON_sp|P02769|ALBU_BOVIN Albumin OS=Bos taurus GN=ALB").unwrap();
        assert_eq!(k, "CON_sp|P02769|ALBU_BOVIN");
        assert_eq!(
            (
                a.protein_ids.as_str(),
                a.protein_names.as_str(),
                a.genes.as_str()
            ),
            ("P02769", "ALBU_BOVIN", "ALB")
        );
        // Any other header: the first word is the accession, nothing else is known.
        let (k, a) = from_fasta_header("CON__P13645 Keratin, type I cytoskeletal 10").unwrap();
        assert_eq!(k, "CON__P13645");
        assert_eq!(
            (
                a.protein_ids.as_str(),
                a.protein_names.as_str(),
                a.genes.as_str()
            ),
            ("CON__P13645", "", "")
        );
        assert_eq!(a.description, "Keratin, type I cytoskeletal 10");
        let (k, a) = from_fasta_header("  myprotein").unwrap();
        assert_eq!(
            (k.as_str(), a.protein_ids.as_str(), a.description.as_str()),
            ("myprotein", "myprotein", "")
        );
        assert!(from_fasta_header("   ").is_none());
    }

    #[test]
    fn a_fasta_table_round_trips_and_joins_shared_peptides_in_member_order() {
        let d = tmp("fasta");
        let fasta = format!("{d}/p.fasta");
        std::fs::write(
            &fasta,
            ">sp|P2|B_HUMAN Beta OS=Homo sapiens GN=GB\nPEPTIDEK\n\
             >sp|P1|A_HUMAN Alpha OS=Homo sapiens GN=GA\nPEPTIDER\n\
             >plain description\nAAAK\n>sp|P1|A_HUMAN duplicate GN=XX\nK\n",
        )
        .unwrap();
        let out = format!("{d}/{FILE_NAME}");
        assert_eq!(write_from_fasta(&fasta, &out).unwrap(), 3);
        let t = TableFile::open(&out).unwrap();
        assert_eq!(
            t.str("protein").unwrap(),
            vec!["plain", "sp|P1|A_HUMAN", "sp|P2|B_HUMAN"],
            "rows ascend by key"
        );
        let ann = ProteinAnnotations::load(&out).unwrap();
        assert_eq!(
            ann.cells("sp|P1|A_HUMAN"),
            "\tP1\tA_HUMAN\tGA",
            "the first header wins"
        );
        // A shared peptide's protein string: members in their order, values once each.
        assert_eq!(
            ann.cells("sp|P2|B_HUMAN;sp|P1|A_HUMAN;plain"),
            "\tP2;P1;plain\tB_HUMAN;A_HUMAN\tGB;GA"
        );
        assert_eq!(ann.cells("UNASSIGNED"), "\t\t\t");
        assert_eq!(ann.cells("sp|P1|A_HUMAN;missing"), "\tP1\tA_HUMAN\tGA");
    }

    #[test]
    fn an_imported_key_is_matched_whole_and_duplicates_merge() {
        let d = tmp("imported");
        let out = format!("{d}/{FILE_NAME}");
        write_table(
            &out,
            vec![
                Col::Str(
                    "protein".into(),
                    vec!["A_HUMAN;B_HUMAN".into(), "C_HUMAN".into(), "C_HUMAN".into()],
                ),
                Col::Str(
                    "protein_ids".into(),
                    vec!["P1;P2".into(), "P3".into(), "P3;P3-2".into()],
                ),
                Col::Str(
                    "protein_names".into(),
                    vec!["A_HUMAN;B_HUMAN".into(), "C_HUMAN".into(), "C_HUMAN".into()],
                ),
                Col::Str("genes".into(), vec!["GA".into(), "GC".into(), "".into()]),
            ],
        )
        .unwrap();
        let ann = ProteinAnnotations::load(&out).unwrap();
        assert_eq!(ann.len(), 2);
        assert_eq!(ann.cells("A_HUMAN;B_HUMAN"), "\tP1;P2\tA_HUMAN;B_HUMAN\tGA");
        assert_eq!(ann.cells("C_HUMAN"), "\tP3;P3-2\tC_HUMAN\tGC");
    }

    #[test]
    fn a_table_without_the_columns_is_refused_and_find_and_publish_follow_the_convention() {
        let d = tmp("find");
        let bad = format!("{d}/bad.parquet");
        write_table(&bad, vec![Col::Str("protein".into(), vec!["x".into()])]).unwrap();
        let e = ProteinAnnotations::load(&bad).unwrap_err();
        assert!(format!("{e:#}").contains("not a protein annotation table"));

        let lib = format!("{d}/lib_precursors.parquet");
        assert_eq!(
            find(None, Some(&lib)).unwrap(),
            None,
            "nothing beside the library yet"
        );
        assert!(find(Some(&format!("{d}/nope.parquet")), Some(&lib)).is_err());
        let fasta = format!("{d}/p.fasta");
        std::fs::write(&fasta, ">sp|P1|A_HUMAN Alpha GN=GA\nK\n").unwrap();
        write_from_fasta(&fasta, &format!("{d}/{FILE_NAME}")).unwrap();
        let found = find(None, Some(&lib)).unwrap().unwrap();
        assert!(found.ends_with(FILE_NAME));

        let out = tmp("find_out");
        let (copy, n) = publish(&found, &out).unwrap();
        assert_eq!(n, 1);
        assert_eq!(
            std::fs::read(&copy).unwrap(),
            std::fs::read(&found).unwrap()
        );
        // Publishing the copy into its own directory is a no-op.
        assert_eq!(publish(&copy, &out).unwrap().0, copy);
        assert!(
            publish(&bad, &out).is_err(),
            "a malformed table is refused before it is copied"
        );
    }
}
