//! Protein annotations: the accessions, entry names and genes of a library's proteins
//! (`protein_annotations.parquet`, docs/15).
//!
//! A library's `protein` column is the grouping key every later table inherits: the scored
//! tables' `protein` and `protein_group`, quant, the cross-run LFQ and the reports. For an
//! imported library it holds UniProt entry names (`ALBU_HUMAN`), for a FASTA-built one the
//! header's first word (`sp|P02768|ALBU_HUMAN`), so the accessions and gene names users map
//! results with were not in any output. The annotations keep them beside the key without
//! changing it: per `protein` key, its accessions, entry names, genes and description.
//! `report` joins them to fill `protein_ids`, `protein_names` and `genes`, the last three
//! columns of `peptides.tsv`, `precursors.tsv` and `proteins.tsv`; nothing else reads them,
//! so every identification, q value and quantity is the same with or without them.
//!
//! Where a search takes them from, in this order ([`resolve_input`]):
//!
//! 1. `--lib-annotations`: a side table, or a library precursor table that carries the
//!    columns below.
//! 2. The library itself: three optional precursor-table columns, `protein_ids`,
//!    `protein_names` and `genes` ([`LIBRARY_COLUMNS`]), which give each row the annotation
//!    of its `protein`. The library importer and the decoy builders write them
//!    (`scripts/_lib_io.py`), so an imported library carries its annotations wherever its
//!    two tables are copied (#299). Only the target rows are read ([`from_library`]).
//! 3. `protein_annotations.parquet` beside `--lib-precursors`: the side table the importer
//!    wrote beside a library before the columns existed.
//! 4. None: a warning when the search starts and again at its report, a record in the
//!    manifest ([`missing_warning`]), and the three TSV columns present and empty.
//!
//! A FASTA search writes its own table from the headers (`digest`, [`write_from_fasta`]), and
//! a FASTA-built library keeps it beside its precursor table, as before.
//!
//! Whatever the source, the search leaves a side table in its output directory so a later
//! `mumdia report` finds it there ([`publish`]): a side table byte for byte, a library's
//! columns as the side table they amount to.
//!
//! The join: a key that is in the table gives its row; a key that is not (a protein string
//! of several FASTA proteins, one per shared peptide) is split on `;` and the rows of its
//! members are concatenated, each value once, in member order. A key with no row at all
//! (the `UNASSIGNED` group of an imported library) has empty cells.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use arrow::datatypes::DataType;
use mumdia_io::table::{write_table, Col, StrBatch, StrInterner, TableFile};

/// The table's file name, beside a library's precursor table and in an output directory.
pub const FILE_NAME: &str = "protein_annotations.parquet";

/// The TSV columns the join appends, after every other column so a reader that addresses
/// the existing columns by position is unaffected.
pub const TSV_HEADER: &str = "\tprotein_ids\tprotein_names\tgenes";

/// The columns a table must carry; `description` is written but not required.
const REQUIRED: [&str; 4] = ["protein", "protein_ids", "protein_names", "genes"];

/// The optional precursor-table columns a library carries its annotations in: on every row,
/// the annotation of that row's `protein` (utf8, `;`-separated, empty when unknown). All
/// three or none; a decoy row carries its target's values, and only target rows are read.
pub const LIBRARY_COLUMNS: [&str; 3] = ["protein_ids", "protein_names", "genes"];

/// Rows per decoded batch when a library's columns are read.
const LIBRARY_BATCH_ROWS: usize = 1 << 16;

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

/// Whether the precursor table `t` carries the library annotation columns
/// ([`LIBRARY_COLUMNS`]): `true` with all three as utf8, `false` with none of them, and an
/// error for a table with only some of them or one that is not utf8, which no writer of the
/// columns produces. Reads the footer only.
pub fn library_has_columns(t: &TableFile) -> Result<bool> {
    let present: Vec<&str> = LIBRARY_COLUMNS
        .iter()
        .copied()
        .filter(|c| t.has_column(c))
        .collect();
    if present.is_empty() {
        return Ok(false);
    }
    if present.len() < LIBRARY_COLUMNS.len() {
        let missing: Vec<&str> = LIBRARY_COLUMNS
            .iter()
            .copied()
            .filter(|c| !present.contains(c))
            .collect();
        bail!(
            "{}: the library carries the protein annotation column(s) {} but not {}; the \
             columns {} are written together (scripts/_lib_io.py), so the table was written \
             by something that dropped some of them",
            t.path(),
            present.join(", "),
            missing.join(", "),
            LIBRARY_COLUMNS.join(", ")
        );
    }
    // The reader's schema is the file's, and building it decodes nothing.
    let schema = t.batches(Some(&LIBRARY_COLUMNS), 1)?.schema();
    for c in LIBRARY_COLUMNS {
        let ty = schema.field_with_name(c)?.data_type();
        if ty != &DataType::Utf8 {
            bail!(
                "{}: the protein annotation column `{c}` is {ty:?}, not utf8, which is the only \
                 string type the engine reads (scripts/_lib_io.py writes it so)",
                t.path()
            );
        }
    }
    Ok(true)
}

/// The annotations a library precursor table carries in its [`LIBRARY_COLUMNS`], one entry
/// per distinct target `protein`: the values of every target row of that key, each once, in
/// the order the file first gives them (rows that agree, which is every row a writer of the
/// columns produces, give that row's values). A NULL cell reads as empty; a key whose three
/// values are all empty gets no entry, as `UNASSIGNED` gets none in a side table.
///
/// Streamed through the columns' dictionaries, so the cost is one hash per distinct value
/// per batch and one per distinct (protein, annotation) combination, not one per row, and
/// the resident set is those combinations, whatever the library's row count.
pub fn from_library(t: &TableFile) -> Result<BTreeMap<String, Annotation>> {
    let cols = ["label", "protein", "protein_ids", "protein_names", "genes"];
    let reader = t.batches_dict(Some(&cols), LIBRARY_BATCH_ROWS, &cols)?;
    let schema = reader.schema();
    let ix: Vec<usize> = cols
        .iter()
        .map(|c| schema.index_of(c))
        .collect::<std::result::Result<_, _>>()?;
    let mut interners: Vec<StrInterner> = (0..cols.len()).map(|_| StrInterner::new()).collect();
    // Label id -> whether it is `target`, filled as labels appear.
    let mut target_label: Vec<bool> = Vec::new();
    let mut seen: HashSet<[u32; 4]> = HashSet::new();
    let mut order: Vec<[u32; 4]> = Vec::new();
    let mut last: Option<[u32; 4]> = None;
    for b in reader {
        let b = b?;
        let views: Vec<StrBatch> = ix
            .iter()
            .zip(cols)
            .map(|(&i, c)| {
                StrBatch::of(b.column(i))
                    .ok_or_else(|| anyhow!("{}: column '{c}' is not utf8", t.path()))
            })
            .collect::<Result<_>>()?;
        for (k, v) in interners.iter_mut().zip(&views) {
            k.begin(v);
        }
        // A dictionary batch's key under a NULL row is arbitrary, so a row's own validity
        // is read before its value (`StrInterner::row`).
        let arrays: Vec<&arrow::array::ArrayRef> = ix.iter().map(|&i| b.column(i)).collect();
        let cell = |interners: &mut [StrInterner], j: usize, r: usize| -> Option<u32> {
            if arrays[j].is_null(r) {
                None
            } else {
                interners[j].row(&views[j], r)
            }
        };
        for r in 0..b.num_rows() {
            let Some(label) = cell(&mut interners, 0, r) else {
                continue;
            };
            while target_label.len() <= label as usize {
                let v = &interners[0].values()[target_label.len()];
                target_label.push(v == "target");
            }
            if !target_label[label as usize] {
                continue;
            }
            let Some(protein) = cell(&mut interners, 1, r) else {
                continue;
            };
            let mut key = [protein, 0, 0, 0];
            for j in 2..5 {
                key[j - 1] = match cell(&mut interners, j, r) {
                    Some(id) => id,
                    None => interners[j].intern(""),
                };
            }
            if last == Some(key) {
                continue;
            }
            last = Some(key);
            if seen.insert(key) {
                order.push(key);
            }
        }
    }
    let mut rows: BTreeMap<String, Annotation> = BTreeMap::new();
    for [p, i, n, g] in order {
        let protein = &interners[1].values()[p as usize];
        let (ids, names, genes) = (
            interners[2].values()[i as usize].as_str(),
            interners[3].values()[n as usize].as_str(),
            interners[4].values()[g as usize].as_str(),
        );
        if protein.is_empty() || (ids.is_empty() && names.is_empty() && genes.is_empty()) {
            continue;
        }
        match rows.get_mut(protein) {
            Some(a) => {
                a.protein_ids = join_unique([a.protein_ids.as_str(), ids]);
                a.protein_names = join_unique([a.protein_names.as_str(), names]);
                a.genes = join_unique([a.genes.as_str(), genes]);
            }
            None => {
                rows.insert(
                    protein.clone(),
                    Annotation {
                        protein_ids: join_unique([ids]),
                        protein_names: join_unique([names]),
                        genes: join_unique([genes]),
                        description: String::new(),
                    },
                );
            }
        }
    }
    Ok(rows)
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
    /// The annotations at `path`: a side table, or a library precursor table (one with a
    /// `label` column) that carries the [`LIBRARY_COLUMNS`] ([`from_library`]). A side table
    /// without the `protein`, `protein_ids`, `protein_names` and `genes` columns is refused,
    /// as is a library table without the annotation columns. A side-table key that appears
    /// twice has the values of both rows, each once.
    pub fn load(path: &str) -> Result<ProteinAnnotations> {
        let t = TableFile::open(path)
            .with_context(|| format!("reading the protein annotations {path}"))?;
        if t.has_column("label") {
            if !library_has_columns(&t)? {
                bail!(
                    "{path}: a library precursor table without the protein annotation columns \
                     ({}), so it names no protein",
                    LIBRARY_COLUMNS.join(", ")
                );
            }
            return Ok(ProteinAnnotations::from_rows(from_library(&t)?));
        }
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

    /// The annotations of `rows`, keyed as given.
    pub fn from_rows(rows: BTreeMap<String, Annotation>) -> ProteinAnnotations {
        ProteinAnnotations {
            rows: rows.into_iter().collect(),
        }
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

/// Where a search's protein annotations come from ([`resolve_input`], [`search_source`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// `--lib-annotations`: a side table, or a library precursor table that carries the
    /// [`LIBRARY_COLUMNS`].
    Explicit(String),
    /// The library's own [`LIBRARY_COLUMNS`], in the `--lib-precursors` table.
    Library(String),
    /// `protein_annotations.parquet` beside `--lib-precursors`: a library imported before
    /// the columns existed.
    Beside(String),
    /// The table a FASTA search's `digest` wrote into its output directory.
    Digest(String),
}

impl Source {
    /// The file the annotations are read from.
    pub fn path(&self) -> &str {
        match self {
            Source::Explicit(p) | Source::Library(p) | Source::Beside(p) | Source::Digest(p) => p,
        }
    }

    /// The file a search hashes as its `lib_annotations` input: a side table given or found
    /// beside the library. The library's own columns are covered by the `lib_precursors`
    /// hash, and a FASTA search's table is an output, so neither is an input of its own.
    pub fn hashed_input(&self) -> Option<&str> {
        match self {
            Source::Explicit(p) | Source::Beside(p) => Some(p),
            Source::Library(_) | Source::Digest(_) => None,
        }
    }
}

/// What a search without protein annotations says, in its log when it starts and at its
/// report, and in its manifest (`protein_annotations_missing`, [`missing_gap`]).
pub fn missing_warning(lib_precursors: &str) -> String {
    format!(
        "no protein annotations: the library {lib_precursors} carries no {} columns and there \
         is no {FILE_NAME} beside it, so the protein_ids, protein_names and genes columns of \
         peptides.tsv, precursors.tsv and proteins.tsv are empty (identifications, q values \
         and quantities are unaffected). The library importer (scripts/import_diann_lib.py, \
         and the desktop application's library import) now writes them into the library: \
         import it again, or pass --lib-annotations with a protein annotation table or an \
         annotated library precursor table",
        LIBRARY_COLUMNS.join(", ")
    )
}

/// The manifest record of a search without protein annotations ([`missing_warning`]).
pub fn missing_gap(lib_precursors: &str) -> mumdia_core::manifest::ProteinAnnotationsGap {
    mumdia_core::manifest::ProteinAnnotationsGap {
        lib_precursors: lib_precursors.to_string(),
        warning: missing_warning(lib_precursors),
    }
}

/// Where a search takes its protein annotations from, decided and validated before it
/// starts, so a malformed table is refused then rather than at the report: `explicit`
/// (`--lib-annotations`: a side table or an annotated library precursor table, which must
/// exist), else the [`LIBRARY_COLUMNS`] of `lib_precursors` (only its footer is read here),
/// else the side table beside it. `None` for a FASTA search (no `lib_precursors`: `digest`
/// writes its own, [`search_source`]) and, with a warning ([`missing_warning`]), for a
/// library with none.
pub fn resolve_input(
    explicit: Option<&str>,
    lib_precursors: Option<&str>,
) -> Result<Option<Source>> {
    if let Some(e) = explicit {
        if !Path::new(e).is_file() {
            bail!("--lib-annotations {e}: no such file");
        }
        let n = ProteinAnnotations::load(e)?.len();
        tracing::info!(
            path = %e,
            proteins = n,
            "protein annotations: from --lib-annotations; the TSVs will carry protein_ids, \
             protein_names and genes"
        );
        return Ok(Some(Source::Explicit(e.to_string())));
    }
    let Some(lib) = lib_precursors else {
        return Ok(None);
    };
    let t = TableFile::open(lib)
        .with_context(|| format!("reading the library {lib} for its protein annotations"))?;
    if library_has_columns(&t)? {
        tracing::info!(
            path = %lib,
            "protein annotations: from the library's own protein_ids, protein_names and genes \
             columns"
        );
        return Ok(Some(Source::Library(lib.to_string())));
    }
    let b = beside(lib);
    if b.is_file() {
        let b = b.display().to_string();
        let n = ProteinAnnotations::load(&b)?.len();
        tracing::info!(
            path = %b,
            proteins = n,
            "protein annotations: from the table beside the library; the TSVs will carry \
             protein_ids, protein_names and genes"
        );
        return Ok(Some(Source::Beside(b)));
    }
    tracing::warn!("{}", missing_warning(lib));
    Ok(None)
}

/// The annotations a search's report joins: [`resolve_input`]'s, else, for a FASTA search
/// (no `lib_precursors`), the table `digest` wrote into `out_dir`. `None` for a library
/// search without any.
pub fn search_source(
    resolved: Option<&Source>,
    lib_precursors: Option<&str>,
    out_dir: &str,
) -> Option<Source> {
    match resolved {
        Some(s) => Some(s.clone()),
        None if lib_precursors.is_none() => Some(Source::Digest(
            Path::new(out_dir).join(FILE_NAME).display().to_string(),
        )),
        None => None,
    }
}

/// Publish a search's annotations into its output directory ([`publish`]) and record them
/// in the manifest. Returns the table's path and its producer: `digest` (a FASTA search),
/// `library-input` (copied from a side table) or `library-columns` (read from a library's
/// columns).
pub fn publish_recorded(
    src: &Source,
    out_dir: &str,
    config_hash: &str,
    man: Option<&mut mumdia_core::manifest::Manifest>,
) -> Result<(String, &'static str)> {
    let (dst, n, from_library) = publish_from(src.path(), out_dir)?;
    let producer = match (src, from_library) {
        (Source::Digest(_), _) => "digest",
        (_, true) => "library-columns",
        (_, false) => "library-input",
    };
    if let Some(m) = man {
        m.record(mumdia_io::record_artifact(
            mumdia_core::schema::artifact::PROTEIN_ANNOTATIONS.0,
            mumdia_core::schema::artifact::PROTEIN_ANNOTATIONS,
            &dst,
            n,
            producer,
            config_hash,
        )?);
    }
    Ok((dst, producer))
}

/// Whether two paths name the same file (both must exist to compare by identity).
fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// Write the annotations at `src` to `<out_dir>/protein_annotations.parquet`: a side table
/// is validated and copied byte for byte (nothing to copy when it is that file already); a
/// library precursor table's [`LIBRARY_COLUMNS`] are written as the side table they amount
/// to ([`from_library`], [`write`]). Returns the table's path and row count.
pub fn publish(src: &str, out_dir: &str) -> Result<(String, u64)> {
    let (dst, n, _) = publish_from(src, out_dir)?;
    Ok((dst, n))
}

/// [`publish`], and whether `src` was a library precursor table.
fn publish_from(src: &str, out_dir: &str) -> Result<(String, u64, bool)> {
    let dst = Path::new(out_dir).join(FILE_NAME);
    let dst_s = dst.display().to_string();
    let t =
        TableFile::open(src).with_context(|| format!("reading the protein annotations {src}"))?;
    if t.has_column("label") {
        if !library_has_columns(&t)? {
            bail!(
                "{src}: a library precursor table without the protein annotation columns ({})",
                LIBRARY_COLUMNS.join(", ")
            );
        }
        let rows = from_library(&t)?;
        let n = write(&rows, &dst_s)?;
        return Ok((dst_s, n, true));
    }
    drop(t);
    let n = ProteinAnnotations::load(src)?.len() as u64;
    if !same_file(Path::new(src), &dst) {
        let target = mumdia_io::table::AtomicPath::new(&dst_s)?;
        std::fs::copy(src, target.tmp())
            .with_context(|| format!("copying the protein annotations {src} to {dst_s}"))?;
        target.publish()?;
    }
    Ok((dst_s, n, false))
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

        let lib = library(&d, "lib_precursors.parquet", &[ALBU], false);
        assert_eq!(
            resolve_input(None, Some(&lib)).unwrap(),
            None,
            "nothing beside the library yet"
        );
        assert!(resolve_input(Some(&format!("{d}/nope.parquet")), Some(&lib)).is_err());
        let fasta = format!("{d}/p.fasta");
        std::fs::write(&fasta, ">sp|P1|A_HUMAN Alpha GN=GA\nK\n").unwrap();
        write_from_fasta(&fasta, &format!("{d}/{FILE_NAME}")).unwrap();
        let found = resolve_input(None, Some(&lib)).unwrap().unwrap();
        assert!(matches!(&found, Source::Beside(p) if p.ends_with(FILE_NAME)));
        assert_eq!(found.hashed_input(), Some(found.path()));

        let out = tmp("find_out");
        let (copy, n) = publish(found.path(), &out).unwrap();
        assert_eq!(n, 1);
        assert_eq!(
            std::fs::read(&copy).unwrap(),
            std::fs::read(found.path()).unwrap()
        );
        // Publishing the copy into its own directory is a no-op.
        assert_eq!(publish(&copy, &out).unwrap().0, copy);
        assert!(
            publish(&bad, &out).is_err(),
            "a malformed table is refused before it is copied"
        );
    }

    type Row = [&'static str; 5];
    // (protein, label, protein_ids, protein_names, genes)
    const ALBU: Row = ["ALBU_HUMAN", "target", "P02768", "ALBU_HUMAN", "ALB"];

    /// A library precursor table of `rows`, with the three annotation columns only when
    /// `annotated`.
    fn library(dir: &str, name: &str, rows: &[Row], annotated: bool) -> String {
        let path = format!("{dir}/{name}");
        let col = |j: usize| -> Vec<String> { rows.iter().map(|r| r[j].to_string()).collect() };
        let mut cols = vec![
            Col::U32("candidate_id".into(), (0..rows.len() as u32).collect()),
            Col::Str("label".into(), col(1)),
            Col::Str("protein".into(), col(0)),
        ];
        if annotated {
            for (j, c) in LIBRARY_COLUMNS.iter().enumerate() {
                cols.push(Col::Str(c.to_string(), col(j + 2)));
            }
        }
        write_table(&path, cols).unwrap();
        path
    }

    /// The library's columns, read from its target rows: one entry per protein key, rows
    /// that agree once, rows of one key that differ united in file order; decoys, whose
    /// `DECOY_` keys are never reported, and keys without any value give no entry.
    #[test]
    fn the_library_columns_give_one_entry_per_target_protein() {
        let d = tmp("library");
        let lib = library(
            &d,
            "lib_precursors.parquet",
            &[
                ALBU,
                ["DECOY_ALBU_HUMAN", "decoy", "P02768", "ALBU_HUMAN", "ALB"],
                ALBU,
                ["TMED2_HUMAN", "target", "Q15363", "TMED2_HUMAN", "TMED2"],
                [
                    "TMED2_HUMAN",
                    "target",
                    "Q15363;Q15363-2",
                    "TMED2_HUMAN",
                    "TMED2",
                ],
                ["UNASSIGNED", "target", "", "", ""],
                [
                    "A_HUMAN;B_HUMAN",
                    "target",
                    "P1;P2",
                    "A_HUMAN;B_HUMAN",
                    "GA",
                ],
                [
                    "DECOY_TMED2_HUMAN",
                    "decoy",
                    "Q15363",
                    "TMED2_HUMAN",
                    "TMED2",
                ],
            ],
            true,
        );
        let t = TableFile::open(&lib).unwrap();
        assert!(library_has_columns(&t).unwrap());
        let rows = from_library(&t).unwrap();
        assert_eq!(
            rows.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["ALBU_HUMAN", "A_HUMAN;B_HUMAN", "TMED2_HUMAN"],
            "targets only, ascending, no entry for a key without values"
        );
        assert_eq!(rows["TMED2_HUMAN"].protein_ids, "Q15363;Q15363-2");
        let ann = ProteinAnnotations::load(&lib).unwrap();
        assert_eq!(ann.len(), 3);
        assert_eq!(ann.cells("ALBU_HUMAN"), "\tP02768\tALBU_HUMAN\tALB");
        assert_eq!(ann.cells("A_HUMAN;B_HUMAN"), "\tP1;P2\tA_HUMAN;B_HUMAN\tGA");
        assert_eq!(ann.cells("UNASSIGNED"), "\t\t\t");

        // Published as the side table they amount to, which loads to the same join.
        let out = tmp("library_out");
        let (table, n) = publish(&lib, &out).unwrap();
        assert_eq!(n, 3);
        let side = ProteinAnnotations::load(&table).unwrap();
        for key in ["ALBU_HUMAN", "A_HUMAN;B_HUMAN", "TMED2_HUMAN", "UNASSIGNED"] {
            assert_eq!(side.cells(key), ann.cells(key), "{key}");
        }
        let t = TableFile::open(&table).unwrap();
        assert!(!t.has_column("label"), "a side table, not a library");
        assert_eq!(
            t.str("protein").unwrap(),
            vec!["ALBU_HUMAN", "A_HUMAN;B_HUMAN", "TMED2_HUMAN"]
        );
    }

    /// A NULL annotation cell reads as empty, and the columns of a library without them, or
    /// with only some of them, are told apart.
    #[test]
    fn null_cells_are_empty_and_a_partial_column_set_is_refused() {
        let d = tmp("nulls");
        let lib = format!("{d}/nulls.parquet");
        write_table(
            &lib,
            vec![
                Col::Str("label".into(), vec!["target".into(), "target".into()]),
                Col::Str("protein".into(), vec!["A_HUMAN".into(), "B_HUMAN".into()]),
                Col::OptStr("protein_ids".into(), vec![Some("P1".into()), None]),
                Col::OptStr("protein_names".into(), vec![None, Some("B_HUMAN".into())]),
                Col::OptStr("genes".into(), vec![Some("GA".into()), None]),
            ],
        )
        .unwrap();
        let ann = ProteinAnnotations::load(&lib).unwrap();
        assert_eq!(ann.cells("A_HUMAN"), "\tP1\t\tGA");
        assert_eq!(ann.cells("B_HUMAN"), "\t\tB_HUMAN\t");

        let plain = library(&d, "plain.parquet", &[ALBU], false);
        assert!(!library_has_columns(&TableFile::open(&plain).unwrap()).unwrap());
        let e = ProteinAnnotations::load(&plain).unwrap_err();
        assert!(
            format!("{e:#}").contains("without the protein annotation columns"),
            "{e:#}"
        );
        let partial = format!("{d}/partial.parquet");
        write_table(
            &partial,
            vec![
                Col::Str("label".into(), vec!["target".into()]),
                Col::Str("protein".into(), vec!["A_HUMAN".into()]),
                Col::Str("genes".into(), vec!["GA".into()]),
            ],
        )
        .unwrap();
        let e = library_has_columns(&TableFile::open(&partial).unwrap()).unwrap_err();
        let msg = format!("{e:#}");
        assert!(
            msg.contains("genes") && msg.contains("protein_ids, protein_names"),
            "{msg}"
        );
        assert!(
            resolve_input(None, Some(&partial)).is_err(),
            "refused before the search"
        );
    }

    /// The order a search looks in: `--lib-annotations`, then the library's columns, then
    /// the table beside the library, then none; and a FASTA search has none to resolve.
    #[test]
    fn the_sources_are_taken_in_order_of_precedence() {
        let d = tmp("precedence");
        let annotated = library(&d, "annotated.parquet", &[ALBU], true);
        let plain = library(&d, "plain.parquet", &[ALBU], false);
        // A side table beside both, naming the protein differently.
        let beside_table = beside(&plain).display().to_string();
        let mut rows = BTreeMap::new();
        rows.insert(
            "ALBU_HUMAN".to_string(),
            Annotation {
                protein_ids: "SIDE".into(),
                ..Annotation::default()
            },
        );
        write(&rows, &beside_table).unwrap();
        let other = tmp("precedence_other");
        let explicit = format!("{other}/explicit.parquet");
        rows.get_mut("ALBU_HUMAN").unwrap().protein_ids = "EXPLICIT".into();
        write(&rows, &explicit).unwrap();

        let got = |e: Option<&str>, lib: Option<&str>| resolve_input(e, lib).unwrap();
        assert_eq!(
            got(Some(&explicit), Some(&annotated)),
            Some(Source::Explicit(explicit.clone()))
        );
        assert_eq!(
            got(None, Some(&annotated)),
            Some(Source::Library(annotated.clone())),
            "the library's own columns win over the table beside it"
        );
        assert_eq!(
            got(None, Some(&plain)),
            Some(Source::Beside(beside_table.clone()))
        );
        // An annotated library precursor table can itself be the explicit source.
        assert_eq!(
            got(Some(&annotated), Some(&plain)),
            Some(Source::Explicit(annotated.clone()))
        );
        assert_eq!(got(None, None), None, "a FASTA search writes its own");
        assert_eq!(
            search_source(None, None, &other),
            Some(Source::Digest(format!(
                "{}",
                Path::new(&other).join(FILE_NAME).display()
            )))
        );
        assert_eq!(search_source(None, Some(&plain), &other), None);
        assert_eq!(Source::Library(annotated.clone()).hashed_input(), None);

        // What each source publishes, and the producer the manifest records.
        let ids = |p: &str| {
            ProteinAnnotations::load(p)
                .unwrap()
                .lookup("ALBU_HUMAN")
                .protein_ids
        };
        for (k, (src, want, producer)) in [
            (
                Source::Explicit(explicit.clone()),
                "EXPLICIT",
                "library-input",
            ),
            (
                Source::Library(annotated.clone()),
                "P02768",
                "library-columns",
            ),
            (
                Source::Beside(beside_table.clone()),
                "SIDE",
                "library-input",
            ),
            (
                Source::Explicit(annotated.clone()),
                "P02768",
                "library-columns",
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let out = tmp(&format!("precedence_out_{k}"));
            let mut m = mumdia_core::manifest::Manifest::new("{}".into(), "c".repeat(64));
            let (path, got_producer) = publish_recorded(&src, &out, "h", Some(&mut m)).unwrap();
            assert_eq!(ids(&path), want, "{src:?}");
            assert_eq!(got_producer, producer, "{src:?}");
            assert_eq!(m.artifacts["protein_annotations"].producing_stage, producer);
        }
    }

    /// A library search without annotations warns, naming the library, what is empty and
    /// how to fill it; the manifest record carries the same text.
    #[test]
    fn a_library_without_annotations_is_warned_about() {
        use std::sync::{Arc, Mutex};

        #[derive(Clone, Default)]
        struct Buf(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Buf {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().write(b)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let d = tmp("missing");
        let plain = library(&d, "plain.parquet", &[ALBU], false);
        let buf = Buf::default();
        let writer = buf.clone();
        let sub = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        let got = tracing::subscriber::with_default(sub, || resolve_input(None, Some(&plain)));
        assert_eq!(got.unwrap(), None);
        let log = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
        assert!(log.contains("WARN"), "{log}");
        assert!(log.contains("no protein annotations"), "{log}");
        assert!(log.contains(&plain), "{log}");

        let w = missing_warning(&plain);
        for needle in [
            "protein_ids, protein_names, genes",
            FILE_NAME,
            "peptides.tsv, precursors.tsv and proteins.tsv are empty",
            "import_diann_lib.py",
            "--lib-annotations",
        ] {
            assert!(w.contains(needle), "{needle}: {w}");
        }
        let gap = missing_gap(&plain);
        assert_eq!(gap.lib_precursors, plain);
        assert_eq!(gap.warning, w);
    }
}
