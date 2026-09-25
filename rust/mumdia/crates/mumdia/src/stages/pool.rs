//! Pool the per-group artifacts of a grouped run into the single-run artifacts the pooled
//! stages read (`groups.window_groups`).
//!
//! Every group's extract, features and compete wrote band-local `candidate_id`s. Here each
//! table is rewritten with library-wide ids (local id plus the band's first library row)
//! and appended, batch by batch, into one `psms_extracted.parquet`, `chromatograms.parquet`,
//! `features.parquet` and `psms_competed.parquet`, so rescore, quant and report run exactly
//! as on an ungrouped run: one table, one `source`, ids that mean the same thing in every
//! artifact. Where two bands overlap (a window overlap across a cut) a candidate was searched
//! twice; the competed row with the higher `prelim_score` wins, and the loser's rows are
//! dropped from all four tables, so the pooled tables hold each candidate once. The
//! extracted and chromatogram tables hold every candidate extract accepted, not only the
//! ones compete kept, so their losers are found in those tables themselves
//! (`table_losers`): a candidate compete deleted in one of two overlapping bands is a
//! duplicate there although it is in one competed table only.
//!
//! Streaming: one batch is resident at a time per table, and the writer's row groups are
//! capped, so the pooling costs one read and one write of the group artifacts and no more
//! memory than any single stage.
//!
//! `psms_extracted` is pooled only when something reads it (the candidate audit). The
//! extracted rows are already on disk per band; copying them into a run-level table that no
//! stage opens was a full read and a full write of the run's widest non-chromatogram
//! artifact.
//!
//! `chromatograms` is pooled unless the run leaves it per band (`groups.pool_chromatograms =
//! false`): quant then reads the bands' tables in band order and drops the overlap losers
//! itself, from the loser sets this stage writes to `overlap_losers.parquet`.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{Array, BooleanArray, UInt32Array};
use arrow::compute::filter_record_batch;
use mumdia_io::table::{BatchWriter, Col, SpliceWriter, TableFile, TableWriter, Written};
use serde::{Deserialize, Serialize};
use tracing::info;

const BATCH_ROWS: usize = 1 << 16;
const ROW_GROUP_ROWS: usize = 1 << 17;

/// One group's artifacts. The ids inside are already library-wide.
#[derive(Clone, Debug)]
pub struct BandArtifacts {
    pub psms: String,
    pub chromatograms: String,
    pub competed: String,
}

pub struct PoolParams<'a> {
    pub bands: &'a [BandArtifacts],
    /// Where to pool the extracted rows, or `None` to leave them per band. Nothing in the
    /// pipeline reads this table -- features ran per band, and rescore reads the competed
    /// one -- so the caller passes `None` unless the candidate audit is on, which is the
    /// one consumer. The per-band tables are written either way.
    pub out_psms: Option<&'a str>,
    /// Where to pool the chromatograms, or `None` when quant reads the bands' tables
    /// directly (`groups.pool_chromatograms = false`), which `out_losers` then makes
    /// possible where the bands overlap.
    pub out_chromatograms: Option<&'a str>,
    /// Where to write the overlap losers, one row per `(band, candidate_id)` the pooled
    /// chromatograms do not take from that band ([`write_losers`], `table_losers`), or
    /// `None`. They are a function of the competed and chromatogram tables, so a later
    /// reader of the band chromatogram tables can drop exactly the rows the pooled table
    /// does not hold without pooling anything.
    pub out_losers: Option<&'a str>,
    /// Where to pool the competed rows, or `None` when rescore reads the bands' competed
    /// tables directly (`groups.pool_competed = false`, allowed only with `bands_disjoint`).
    pub out_competed: Option<&'a str>,
    /// The bands' library row spans do not overlap, so no candidate was searched in two
    /// bands and there is no overlap duplicate to find. The caller knows it from the plan's
    /// row spans (`run_groups`); `false` makes the pool look, as the standalone `mumdia pool`
    /// does. Set wrongly, a candidate searched twice would be pooled twice.
    pub bands_disjoint: bool,
}

/// Row counts of the pooled tables and how many overlap duplicates were removed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoolStats {
    pub psms: u64,
    /// 0 when the chromatograms were not pooled.
    pub chromatograms: u64,
    pub competed: u64,
    /// Content hashes of the pooled tables, computed while they were spliced, in the order
    /// psms (when pooled), chromatograms, competed.
    pub psms_hash: Option<String>,
    /// `None` when the chromatograms were not pooled.
    pub chromatograms_hash: Option<String>,
    /// `None` when the competed rows were not pooled.
    pub competed_hash: Option<String>,
    /// Candidates that appeared in two bands' competed tables and were kept from one.
    pub duplicates: u64,
    /// How many `(band, candidate)` drops the chromatogram tables add to the competed
    /// dedup's losers: overlap candidates compete deleted in all but one band, which hold
    /// rows in two bands' chromatogram tables and one competed table (`table_losers`), or in
    /// none (then kept from the first band that holds them). 0 under the default
    /// `compete.group_by = peptidoform_charge`, which gives each candidate a group of its
    /// own and deletes none.
    pub chromatogram_only_duplicates: u64,
    /// Per band, ascending, the candidates the pooled chromatograms do not take from that
    /// band's chromatogram table, because another band won them (`table_losers`). Empty
    /// sets for disjoint bands.
    pub losers: Vec<Vec<u32>>,
    /// `overlap_losers.parquet` as written, when `out_losers` asked for it.
    pub losers_written: Option<Written>,
}

/// The footer key under which [`write_losers`] records the band chromatogram tables its
/// `band` column indexes, as a JSON list of [`BandTableId`].
pub const LOSERS_BAND_TABLES_KEY: &str = "mumdia.overlap_losers.band_tables";

/// One band chromatogram table as the loser file names it, so that a reader can tell the
/// tables it was given are the ones the losers were found for, in the same order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BandTableId {
    /// The table's directory and file name: `gNN/chromatograms.parquet` for a grouped run.
    pub name: String,
    /// Its row count, from its footer.
    pub rows: u64,
    /// The content hash its report (`<table>.report.json`) records, when it has one.
    pub content_hash: Option<String>,
}

impl BandTableId {
    /// The name, row count and recorded content hash of the table at `path`, from its
    /// footer and its report; no data page is read.
    pub fn of(path: &str) -> Result<BandTableId> {
        let p = std::path::Path::new(path);
        let file = p
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        let name = match p.parent().and_then(|d| d.file_name()) {
            Some(dir) => format!("{}/{file}", dir.to_string_lossy()),
            None => file,
        };
        let rows = TableFile::open(path)?.nrows as u64;
        let report = format!("{path}.report.json");
        // A report that is missing or unreadable records no hash, and the check then rests
        // on the name and the row count; neither is a reason to fail the pool.
        let content_hash = mumdia_io::json::read_json::<serde_json::Value>(&report)
            .ok()
            .and_then(|v| v.get("content_hash")?.as_str().map(str::to_string));
        Ok(BandTableId {
            name,
            rows,
            content_hash,
        })
    }

    fn describe(&self) -> String {
        match &self.content_hash {
            Some(h) => format!("{} ({} rows, content hash {h})", self.name, self.rows),
            None => format!("{} ({} rows)", self.name, self.rows),
        }
    }
}

/// Write the overlap losers as `band` (the band's position in pool order) and
/// `candidate_id`, one row per dropped candidate, ascending by band then candidate, with
/// `tables` (one per band, in pool order, bands without a loser included) in the footer
/// under [`LOSERS_BAND_TABLES_KEY`]. A run whose bands are disjoint writes the table with
/// no rows, which says as much.
pub fn write_losers(path: &str, losers: &[Vec<u32>], tables: &[BandTableId]) -> Result<Written> {
    if losers.len() != tables.len() {
        bail!(
            "pool: {} loser sets for {} band tables",
            losers.len(),
            tables.len()
        );
    }
    let mut band: Vec<u32> = Vec::new();
    let mut cid: Vec<u32> = Vec::new();
    for (b, set) in losers.iter().enumerate() {
        for &c in set {
            band.push(b as u32);
            cid.push(c);
        }
    }
    let mut w = TableWriter::new(path)
        .with_content_hash()
        .with_metadata(LOSERS_BAND_TABLES_KEY, &serde_json::to_string(tables)?);
    w.write_cols(vec![
        Col::U32("band".into(), band),
        Col::U32("candidate_id".into(), cid),
    ])?;
    w.close_hashed()
}

/// Read [`write_losers`]'s table back as one ascending, distinct set per table of
/// `tables`, the band chromatogram tables the caller will read. They must be the tables
/// the file names, in its order: the same count, and in every position the same directory
/// and file name, the same row count and, where both the file and the table's report
/// record one, the same content hash. Anything else (a table left out, two swapped, a
/// band directory of an earlier plan, another run's loser file) would apply a band's
/// loser set to the wrong table, so it is refused, naming the expected order.
pub fn read_losers(path: &str, tables: &[String]) -> Result<Vec<Vec<u32>>> {
    let t = TableFile::open(path)?;
    let Some(recorded) = t.metadata_value(LOSERS_BAND_TABLES_KEY) else {
        bail!(
            "{path} does not name the band chromatogram tables its losers belong to (no \
             {LOSERS_BAND_TABLES_KEY} footer entry). Re-run the grouped run, or quantify the \
             pooled chromatograms.parquet that `mumdia pool --groups-dir` rebuilds."
        );
    };
    let recorded: Vec<BandTableId> = serde_json::from_str(&recorded)
        .with_context(|| format!("{path}: parsing the {LOSERS_BAND_TABLES_KEY} footer entry"))?;
    let order = || {
        recorded
            .iter()
            .map(|r| r.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    if recorded.len() != tables.len() {
        bail!(
            "{path} holds the overlap losers of {} band chromatogram table(s), but {} were \
             given; give every band's table, in this order: {}",
            recorded.len(),
            tables.len(),
            order()
        );
    }
    for (b, (rec, given)) in recorded.iter().zip(tables).enumerate() {
        let id = BandTableId::of(given)?;
        let hash_differs = matches!(
            (&rec.content_hash, &id.content_hash),
            (Some(a), Some(g)) if a != g
        );
        if id.name != rec.name || id.rows != rec.rows || hash_differs {
            bail!(
                "{path}: band {b} is {}, but the table given in that position, {given}, is \
                 {}; give the band tables in this order: {}",
                rec.describe(),
                id.describe(),
                order()
            );
        }
    }
    let band = t.u32("band")?;
    let cid = t.u32("candidate_id")?;
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); tables.len()];
    for (&b, &c) in band.iter().zip(&cid) {
        let Some(set) = out.get_mut(b as usize) else {
            bail!(
                "{path} drops candidate {c} from band {b}, but it names {} band table(s)",
                tables.len()
            );
        };
        set.push(c);
    }
    for set in &mut out {
        set.sort_unstable();
        set.dedup();
    }
    Ok(out)
}

/// The best competed row of every candidate: library-wide id -> (band index, best
/// `prelim_score`).
type CompetedWinners = HashMap<u32, (usize, f64)>;

/// The competed dedup: per band, the ids to drop from its competed table because another
/// band's row won the candidate; the winner of every competed candidate; and how many
/// candidates two bands' competed tables share.
fn overlap_losers(bands: &[BandArtifacts]) -> Result<(Vec<HashSet<u32>>, CompetedWinners, u64)> {
    // Library-wide id -> (band index, best prelim_score) from the competed tables.
    let mut best: CompetedWinners = HashMap::new();
    let mut losers: Vec<HashSet<u32>> = vec![HashSet::new(); bands.len()];
    let mut duplicates = 0u64;
    for (b, band) in bands.iter().enumerate() {
        let t = TableFile::open(&band.competed)?;
        if !t.has_column("prelim_score") {
            bail!(
                "{} has no prelim_score column; the pooled dedup keys on it",
                band.competed
            );
        }
        let cid = t.u32("candidate_id")?;
        let score = t.f64("prelim_score")?;
        // One entry per candidate per band (compete may keep several peak ranks).
        let mut per_band: HashMap<u32, f64> = HashMap::new();
        for (c, s) in cid.iter().zip(&score) {
            let e = per_band.entry(*c).or_insert(f64::NEG_INFINITY);
            if *s > *e {
                *e = *s;
            }
        }
        for (global, s) in per_band {
            match best.get(&global) {
                None => {
                    best.insert(global, (b, s));
                }
                Some(&(prev_b, prev_s)) => {
                    duplicates += 1;
                    if s > prev_s {
                        losers[prev_b].insert(global);
                        best.insert(global, (b, s));
                    } else {
                        losers[b].insert(global);
                    }
                }
            }
        }
    }
    Ok((losers, best, duplicates))
}

/// Per band, the ids the pooled copy of one kind of band table (`paths`, one per band, in
/// pool order) must not take from that band: `competed` (the competed dedup's losers), and
/// every candidate two or more bands' tables hold, from every band but one.
///
/// A band's extracted and chromatogram tables hold every candidate extract accepted there,
/// its competed table only the ones compete kept. Where compete deleted a candidate in one
/// of two overlapping bands and kept it in the other (under `compete.group_by =
/// base_peptide` or `apex`, a sibling charge state or modform that only the first band
/// holds beat it), the candidate is a duplicate of these tables but not of the competed
/// ones, and the competed dedup does not see it: both bands' rows would be pooled, and
/// quant would sum two bands' traces into one quantity. So a candidate that several bands'
/// tables hold is kept from the band whose competed row won it and dropped from the
/// others; one that no competed row kept (read only by `keep_all` quant) is kept from the
/// first band in pool order that holds it. Where every competed table holds exactly the
/// candidates of its band's table, as under the default grouping, this adds nothing to
/// `competed`.
///
/// Only the ids two bands can share are read: those inside the overlap of a band's
/// `candidate_id` range with another band's, from the row groups whose statistics meet
/// that overlap, which window overlap puts at the two ends of a band.
fn table_losers(
    paths: &[&str],
    best: &CompetedWinners,
    competed: &[HashSet<u32>],
) -> Result<Vec<HashSet<u32>>> {
    let mut files: Vec<(TableFile, Vec<mumdia_io::table::RowGroupStats>)> = Vec::new();
    let mut ranges: Vec<Option<(u32, u32)>> = Vec::new();
    for path in paths {
        let t = TableFile::open(path)?;
        let stats = t.row_group_stats("candidate_id")?;
        let mut range: Option<(u32, u32)> = None;
        for s in stats.iter().filter(|s| s.rows > 0) {
            let (lo, hi) = match (s.min, s.max) {
                (Some(lo), Some(hi)) => (lo.floor() as u32, hi.ceil() as u32),
                // No statistics: the group could hold any id.
                _ => (0, u32::MAX),
            };
            range = Some(match range {
                None => (lo, hi),
                Some((a, z)) => (a.min(lo), z.max(hi)),
            });
        }
        files.push((t, stats));
        ranges.push(range);
    }
    // (candidate, band) for every id a band holds inside its overlap with another band.
    let mut held: Vec<(u32, u32)> = Vec::new();
    for (b, (t, stats)) in files.iter().enumerate() {
        let Some((lo, hi)) = ranges[b] else {
            continue;
        };
        let windows: Vec<(u32, u32)> = ranges
            .iter()
            .enumerate()
            .filter(|&(o, _)| o != b)
            .filter_map(|(_, r)| {
                let (a, z) = (*r)?;
                let (s, e) = (a.max(lo), z.min(hi));
                (s <= e).then_some((s, e))
            })
            .collect();
        if windows.is_empty() {
            continue;
        }
        let mut first = 0usize;
        for s in stats {
            let start = first;
            first += s.rows;
            if s.rows == 0 {
                continue;
            }
            let meets = match (s.min, s.max) {
                (Some(gmin), Some(gmax)) => windows
                    .iter()
                    .any(|&(a, z)| f64::from(a) <= gmax && gmin <= f64::from(z)),
                _ => true,
            };
            if !meets {
                continue;
            }
            for c in t.span(start, s.rows)?.u32("candidate_id")? {
                if windows.iter().any(|&(a, z)| a <= c && c <= z) {
                    held.push((c, b as u32));
                }
            }
        }
    }
    held.sort_unstable();
    held.dedup();
    let mut losers: Vec<HashSet<u32>> = competed.to_vec();
    let mut i = 0usize;
    while i < held.len() {
        let c = held[i].0;
        let mut j = i;
        while j < held.len() && held[j].0 == c {
            j += 1;
        }
        if j - i > 1 {
            let keep = match best.get(&c) {
                Some(&(w, _)) => w as u32,
                None => held[i].1,
            };
            for &(_, h) in &held[i..j] {
                if h != keep {
                    losers[h as usize].insert(c);
                }
            }
        }
        i = j;
    }
    Ok(losers)
}

/// Append every band's table into `out`, losers dropped. Returns the row count and the
/// content hash of `out`, computed while it was written (the per-row-group rewrite temp
/// files are not hashed).
///
/// The bands already wrote library-wide ids (`Library::global_offset` is added on the way
/// out of extract), so pooling is a concatenation and almost every row group can be spliced
/// into the output as bytes, without being decoded. That matters at scale: decoding and
/// re-encoding one run's 68 GB of chromatograms ran at 3.5 MB/s on one core, five hours,
/// against a disk that does 221 MB/s. Only the row groups that actually hold a dropped
/// candidate are decoded, filtered and re-encoded, and window overlap puts those at the two
/// ends of a band.
fn pool_table(paths: impl Iterator<Item = (String, HashSet<u32>)>, out: &str) -> Result<Written> {
    let paths: Vec<(String, HashSet<u32>)> = paths.collect();
    let Some((template, _)) = paths.first() else {
        bail!("pool: no group tables to pool into {out}");
    };
    let mut w = SpliceWriter::create_hashed(out, template)?;
    let mut rows = 0u64;
    for (path, drop) in &paths {
        if drop.is_empty() {
            rows += w.append_row_groups(path, |_| true)?;
            continue;
        }
        // Which row groups can hold a dropped id, from the statistics on candidate_id.
        let t = TableFile::open(path)?;
        let stats = t.row_group_stats("candidate_id")?;
        let spans = SpliceWriter::row_group_spans(path)?;
        if stats.len() != spans.len() {
            bail!(
                "{path}: {} row groups but {} statistics",
                spans.len(),
                stats.len()
            );
        }
        let mut sorted: Vec<u32> = drop.iter().copied().collect();
        sorted.sort_unstable();
        let dirty: Vec<bool> = stats
            .iter()
            .map(|s| match (s.min, s.max) {
                (Some(lo), Some(hi)) => {
                    let at = sorted.partition_point(|&d| f64::from(d) < lo);
                    at < sorted.len() && f64::from(sorted[at]) <= hi
                }
                // No statistics: assume it holds one and take the slow path.
                _ => true,
            })
            .collect();
        // File order is the pooled row order, so clean runs and dirty groups are appended as
        // they come, not clean ones first.
        let mut i = 0usize;
        while i < spans.len() {
            if dirty[i] {
                rows += rewrite_row_group(&mut w, path, spans[i], drop)?;
                i += 1;
                continue;
            }
            let start = i;
            while i < spans.len() && !dirty[i] {
                i += 1;
            }
            rows += w.append_row_groups(path, |k| k >= start && k < i)?;
        }
    }
    let spliced = w.close_hashed()?;
    if spliced.rows != rows {
        bail!(
            "pool: counted {rows} rows into {out} but the file holds {}",
            spliced.rows
        );
    }
    Ok(spliced)
}

/// Decode one row group, drop the losers, and splice the result back in. Used only for the
/// row groups whose candidate range holds a dropped id.
fn rewrite_row_group(
    w: &mut SpliceWriter,
    path: &str,
    span: (usize, usize),
    drop: &HashSet<u32>,
) -> Result<u64> {
    let (first_row, n_rows) = span;
    let t = TableFile::open_rows(path, first_row, n_rows)?;
    let reader = t.batches(None, BATCH_ROWS)?;
    let schema = reader.schema();
    let cid_ix = schema
        .index_of("candidate_id")
        .map_err(|_| anyhow!("{path} has no candidate_id column"))?;
    let tmp = format!("{path}.pool-rg{first_row}.parquet");
    let mut bw = BatchWriter::with_row_group_rows(&tmp, schema.clone(), ROW_GROUP_ROWS)?;
    let mut kept = 0u64;
    for b in reader {
        let b = b?;
        let cid = b
            .column(cid_ix)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| anyhow!("{path}: candidate_id is not u32"))?;
        if cid.null_count() > 0 {
            bail!("{path}: candidate_id has nulls");
        }
        let keep: Vec<bool> = cid.values().iter().map(|c| !drop.contains(c)).collect();
        let batch = filter_record_batch(&b, &BooleanArray::from(keep))?;
        if batch.num_rows() > 0 {
            bw.write(&batch)?;
            kept += batch.num_rows() as u64;
        }
    }
    bw.close()?;
    let spliced = w.append_row_groups(&tmp, |_| true)?;
    std::fs::remove_file(&tmp).ok();
    if spliced != kept {
        bail!("{path}: rewrote {kept} rows but spliced {spliced}");
    }
    Ok(kept)
}

pub fn run(p: PoolParams) -> Result<PoolStats> {
    let t0 = Instant::now();
    if p.bands.is_empty() {
        bail!("pool: no groups");
    }
    // Band spans that do not overlap cannot share a candidate: the ids are library rows
    // (band-local id plus the band's first row). The dedup then has nothing to find, and
    // it decoded `candidate_id` and `prelim_score` of every band's competed table and built
    // a map over every candidate of the run to find it.
    let none = || vec![HashSet::new(); p.bands.len()];
    let (losers, chrom_losers, psms_losers, duplicates) = if p.bands_disjoint {
        info!(
            groups = p.bands.len(),
            "pool: the bands' library row spans are disjoint; no overlap duplicate to find"
        );
        (none(), none(), none(), 0)
    } else {
        let (losers, best, duplicates) = overlap_losers(p.bands)?;
        let of = |pick: fn(&BandArtifacts) -> &String| {
            let paths: Vec<&str> = p.bands.iter().map(|b| pick(b).as_str()).collect();
            table_losers(&paths, &best, &losers)
        };
        let chrom_losers = of(|b| &b.chromatograms)?;
        let psms_losers = match p.out_psms {
            Some(_) => of(|b| &b.psms)?,
            None => none(),
        };
        (losers, chrom_losers, psms_losers, duplicates)
    };
    let count = |sets: &[HashSet<u32>]| sets.iter().map(|s| s.len() as u64).sum::<u64>();
    let chromatogram_only_duplicates = count(&chrom_losers) - count(&losers);
    if chromatogram_only_duplicates > 0 {
        info!(
            chromatogram_only_duplicates,
            "pool: overlap candidates that compete deleted in all but one band are kept from \
             one band's chromatograms"
        );
    }
    let with = |pick: fn(&BandArtifacts) -> &String, sets: &[HashSet<u32>]| {
        p.bands
            .iter()
            .zip(sets.iter())
            .map(move |(b, l)| (pick(b).clone(), l.clone()))
            .collect::<Vec<_>>()
    };
    let psms = match p.out_psms {
        Some(out) => Some(pool_table(
            with(|b| &b.psms, &psms_losers).into_iter(),
            out,
        )?),
        None => None,
    };
    let chromatograms = match p.out_chromatograms {
        Some(out) => Some(pool_table(
            with(|b| &b.chromatograms, &chrom_losers).into_iter(),
            out,
        )?),
        None => None,
    };
    let loser_lists: Vec<Vec<u32>> = chrom_losers
        .iter()
        .map(|set| {
            let mut v: Vec<u32> = set.iter().copied().collect();
            v.sort_unstable();
            v
        })
        .collect();
    let losers_written = match p.out_losers {
        Some(out) => {
            let tables = p
                .bands
                .iter()
                .map(|b| BandTableId::of(&b.chromatograms))
                .collect::<Result<Vec<_>>>()?;
            Some(write_losers(out, &loser_lists, &tables)?)
        }
        None => None,
    };
    let competed = match p.out_competed {
        Some(out) => Some(pool_table(with(|b| &b.competed, &losers).into_iter(), out)?),
        None => {
            if !p.bands_disjoint {
                bail!(
                    "pool: the competed rows can be left per band only when the bands' \
                     library row spans are disjoint"
                );
            }
            None
        }
    };
    let stats = PoolStats {
        psms: psms.as_ref().map_or(0, |w| w.rows),
        chromatograms: chromatograms.as_ref().map_or(0, |w| w.rows),
        competed: competed.as_ref().map_or(0, |w| w.rows),
        psms_hash: psms.map(|w| w.content_hash),
        chromatograms_hash: chromatograms.map(|w| w.content_hash),
        competed_hash: competed.map(|w| w.content_hash),
        duplicates,
        chromatogram_only_duplicates,
        losers: loser_lists,
        losers_written,
    };
    info!(
        groups = p.bands.len(),
        psms = stats.psms,
        psms_pooled = p.out_psms.is_some(),
        chromatograms_pooled = p.out_chromatograms.is_some(),
        competed_pooled = p.out_competed.is_some(),
        chromatograms = stats.chromatograms,
        competed = stats.competed,
        duplicates,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "pool: done"
    );
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::{write_table, Col};

    fn band(
        dir: &std::path::Path,
        tag: &str,
        marker: f64,
        cids: &[u32],
        scores: &[f64],
    ) -> BandArtifacts {
        let mk = |name: &str, extra: bool| -> String {
            let p = dir
                .join(format!("{tag}_{name}.parquet"))
                .to_str()
                .unwrap()
                .to_string();
            let mut cols = vec![
                Col::U32("candidate_id".into(), cids.to_vec()),
                Col::F64(
                    "x".into(),
                    cids.iter().map(|c| *c as f64 + marker).collect(),
                ),
                // A LargeList column, as the chromatogram table has: the distinction from
                // `List` lives only in the file's Arrow metadata, which a spliced output has
                // to carry over from the tables it was spliced from.
                Col::LargeListF32(
                    "trace".into(),
                    cids.iter()
                        .map(|c| vec![*c as f32, *c as f32 + 0.5])
                        .collect(),
                ),
            ];
            if extra {
                cols.push(Col::F64("prelim_score".into(), scores.to_vec()));
            }
            write_table(&p, cols).unwrap();
            p
        };
        BandArtifacts {
            psms: mk("psms", false),
            chromatograms: mk("chrom", false),
            competed: mk("comp", true),
        }
    }

    /// A band whose extracted and chromatogram tables hold candidates its competed table
    /// does not: compete deleted them there.
    fn band_with_deleted(
        dir: &std::path::Path,
        tag: &str,
        marker: f64,
        extracted: &[u32],
        competed: &[(u32, f64)],
    ) -> BandArtifacts {
        let mk = |name: &str, cids: &[u32], scores: Option<Vec<f64>>| -> String {
            let p = dir
                .join(format!("{tag}_{name}.parquet"))
                .to_str()
                .unwrap()
                .to_string();
            let mut cols = vec![
                Col::U32("candidate_id".into(), cids.to_vec()),
                Col::F64(
                    "x".into(),
                    cids.iter().map(|c| *c as f64 + marker).collect(),
                ),
            ];
            if let Some(s) = scores {
                cols.push(Col::F64("prelim_score".into(), s));
            }
            write_table(&p, cols).unwrap();
            p
        };
        let comp_ids: Vec<u32> = competed.iter().map(|c| c.0).collect();
        BandArtifacts {
            psms: mk("psms", extracted, None),
            chromatograms: mk("chrom", extracted, None),
            competed: mk(
                "comp",
                &comp_ids,
                Some(competed.iter().map(|c| c.1).collect()),
            ),
        }
    }

    /// Two overlapping bands whose competed tables disagree with their chromatogram tables:
    /// 3 was kept in both bands (band 0 wins it), 4 was deleted by compete in band 0 and
    /// kept in band 1, and 5 was deleted in both. The competed dedup sees only 3; the
    /// extracted and chromatogram tables must still hold each of 3, 4 and 5 once: 4 from
    /// band 1, whose competed row won it, and 5 from band 0, the first band that holds it.
    #[test]
    fn a_candidate_compete_deleted_in_one_band_is_pooled_from_one_band() {
        let dir = std::env::temp_dir().join(format!("mumdia_pool_deleted_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let b0 = band_with_deleted(
            &dir,
            "e0",
            0.0,
            &[0, 1, 2, 3, 4, 5],
            &[(0, 9.0), (1, 9.0), (2, 9.0), (3, 7.0)],
        );
        let b1 = band_with_deleted(
            &dir,
            "e1",
            0.003,
            &[3, 4, 5, 6],
            &[(3, 5.0), (4, 6.0), (6, 9.0)],
        );
        let out = |n: &str| {
            dir.join(format!("deleted_{n}.parquet"))
                .to_str()
                .unwrap()
                .to_string()
        };
        let (op, oc, ol, ok) = (out("psms"), out("chrom"), out("losers"), out("comp"));
        let bands = [b0, b1];
        let stats = run(PoolParams {
            bands: &bands,
            out_psms: Some(op.as_str()),
            out_chromatograms: Some(&oc),
            out_losers: Some(&ol),
            out_competed: Some(ok.as_str()),
            bands_disjoint: false,
        })
        .unwrap();
        assert_eq!(stats.duplicates, 1);
        assert_eq!(stats.chromatogram_only_duplicates, 2);
        assert_eq!(stats.losers, vec![vec![4], vec![3, 5]]);
        let paths: Vec<String> = bands.iter().map(|b| b.chromatograms.clone()).collect();
        assert_eq!(read_losers(&ol, &paths).unwrap(), stats.losers);
        for path in [&op, &oc] {
            let t = TableFile::open(path).unwrap();
            assert_eq!(
                t.u32("candidate_id").unwrap(),
                vec![0, 1, 2, 3, 5, 4, 6],
                "{path}"
            );
            let x = t.f64("x").unwrap();
            // 3 and 5 from band 0 (x = 3.000, 5.000), 4 from band 1 (x = 4.003).
            assert!(
                (x[3] - 3.0).abs() < 1e-9
                    && (x[4] - 5.0).abs() < 1e-9
                    && (x[5] - 4.003).abs() < 1e-9,
                "{path}: {x:?}"
            );
        }
        let t = TableFile::open(&ok).unwrap();
        assert_eq!(t.u32("candidate_id").unwrap(), vec![0, 1, 2, 3, 4, 6]);
    }

    #[test]
    fn pooled_tables_carry_global_ids_and_one_row_per_overlap_candidate() {
        let dir = std::env::temp_dir().join(format!("mumdia_pool_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // The bands already carry library-wide ids: band 0 holds 0..4, band 1 holds 3..6,
        // so 3 and 4 are in both. Band 0 wins 3 (higher score), band 1 wins 4. The `x`
        // marker says which band a row came from.
        let b0 = band(
            &dir,
            "b0",
            0.0,
            &[0, 1, 2, 3, 4],
            &[9.0, 9.0, 9.0, 7.0, 2.0],
        );
        let b1 = band(&dir, "b1", 0.003, &[3, 4, 5, 6], &[5.0, 6.0, 9.0, 9.0]);
        let out = |n: &str| {
            dir.join(format!("out_{n}.parquet"))
                .to_str()
                .unwrap()
                .to_string()
        };
        let (op, oc, ok) = (out("psms"), out("chrom"), out("comp"));
        let stats = run(PoolParams {
            bands: &[b0, b1],
            out_psms: Some(op.as_str()),
            out_chromatograms: Some(&oc),
            out_losers: None,
            out_competed: Some(ok.as_str()),
            bands_disjoint: false,
        })
        .unwrap();
        assert_eq!(stats.duplicates, 2);
        assert_eq!(stats.competed, 7);
        for path in [&op, &oc, &ok] {
            let t = TableFile::open(path).unwrap();
            let cid = t.u32("candidate_id").unwrap();
            assert_eq!(cid, vec![0, 1, 2, 3, 4, 5, 6], "{path}");
            assert!(
                matches!(
                    t.schema.field_with_name("trace").unwrap().data_type(),
                    arrow::datatypes::DataType::LargeList(_)
                ),
                "{path}: the spliced table lost the LargeList type"
            );
            let x = t.f64("x").unwrap();
            // Id 3 came from band 0 (x = 3.000), id 4 from band 1 (x = 4.003).
            assert!(
                (x[3] - 3.0).abs() < 1e-9 && (x[4] - 4.003).abs() < 1e-9,
                "{path}: {x:?}"
            );
        }
    }

    /// A band whose losers sit in one row group: the groups around it are spliced as bytes
    /// and that one is decoded, filtered and re-encoded. The pooled table must still be the
    /// bands' rows, in band order, in row order, with the losers gone -- the splice must not
    /// reorder anything, and the mixed path must not drop or duplicate a group.
    #[test]
    fn a_band_of_many_row_groups_keeps_its_order_when_only_one_holds_a_loser() {
        use mumdia_io::table::BatchWriter;
        let dir = std::env::temp_dir().join(format!("mumdia_pool_rg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // 500 rows in row groups of 50, so 10 groups; the losers are ids 220..230, which is
        // group 4 alone.
        let mk = |tag: &str, ids: std::ops::Range<u32>, score: f64| -> String {
            let p = dir
                .join(format!("{tag}.parquet"))
                .to_str()
                .unwrap()
                .to_string();
            let cids: Vec<u32> = ids.collect();
            let n = cids.len();
            let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new(
                    "candidate_id",
                    arrow::datatypes::DataType::UInt32,
                    false,
                ),
                arrow::datatypes::Field::new(
                    "prelim_score",
                    arrow::datatypes::DataType::Float64,
                    false,
                ),
            ]));
            let mut w = BatchWriter::with_row_group_rows(&p, schema.clone(), 50).unwrap();
            for chunk in cids.chunks(25) {
                let batch = arrow::record_batch::RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        std::sync::Arc::new(UInt32Array::from(chunk.to_vec())),
                        std::sync::Arc::new(arrow::array::Float64Array::from(vec![
                            score;
                            chunk.len()
                        ])),
                    ],
                )
                .unwrap();
                w.write(&batch).unwrap();
            }
            w.close().unwrap();
            assert_eq!(TableFile::open(&p).unwrap().nrows, n);
            p
        };
        // Band 0 holds 0..500 and band 1 holds 220..230 with a higher score, so band 0
        // loses exactly those ten, which live in one of its ten row groups.
        let b0 = BandArtifacts {
            psms: mk("rg_b0_psms", 0..500, 1.0),
            chromatograms: mk("rg_b0_chrom", 0..500, 1.0),
            competed: mk("rg_b0_comp", 0..500, 1.0),
        };
        let b1 = BandArtifacts {
            psms: mk("rg_b1_psms", 220..230, 9.0),
            chromatograms: mk("rg_b1_chrom", 220..230, 9.0),
            competed: mk("rg_b1_comp", 220..230, 9.0),
        };
        let out = |n: &str| {
            dir.join(format!("rg_out_{n}.parquet"))
                .to_str()
                .unwrap()
                .to_string()
        };
        let (op, oc, ok) = (out("psms"), out("chrom"), out("comp"));
        let stats = run(PoolParams {
            bands: &[b0, b1],
            out_psms: Some(op.as_str()),
            out_chromatograms: Some(&oc),
            out_losers: None,
            out_competed: Some(ok.as_str()),
            bands_disjoint: false,
        })
        .unwrap();
        assert_eq!(stats.duplicates, 10);
        assert_eq!(stats.competed, 500);
        let mut expect: Vec<u32> = (0..500).filter(|c| !(220..230).contains(c)).collect();
        expect.extend(220..230);
        for path in [&op, &oc, &ok] {
            let t = TableFile::open(path).unwrap();
            assert_eq!(t.u32("candidate_id").unwrap(), expect, "{path}");
        }
    }

    #[test]
    fn without_a_psms_path_the_other_tables_are_pooled_unchanged() {
        let dir = std::env::temp_dir().join(format!("mumdia_pool_nopsms_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let b0 = band(
            &dir,
            "n0",
            0.0,
            &[0, 1, 2, 3, 4],
            &[9.0, 9.0, 9.0, 7.0, 2.0],
        );
        let b1 = band(&dir, "n1", 0.003, &[3, 4, 5, 6], &[5.0, 6.0, 9.0, 9.0]);
        let out = |n: &str| {
            dir.join(format!("nopsms_{n}.parquet"))
                .to_str()
                .unwrap()
                .to_string()
        };
        let (op, oc, ok) = (out("psms"), out("chrom"), out("comp"));
        let stats = run(PoolParams {
            bands: &[b0, b1],
            out_psms: None,
            out_chromatograms: Some(&oc),
            out_losers: None,
            out_competed: Some(ok.as_str()),
            bands_disjoint: false,
        })
        .unwrap();
        // The skipped table is not written, and the pooled tables that are read downstream
        // are exactly what they are when it is.
        assert_eq!(stats.psms, 0);
        assert!(!std::path::Path::new(&op).exists());
        assert_eq!(stats.competed, 7);
        assert_eq!(stats.duplicates, 2);
        for path in [&oc, &ok] {
            let t = TableFile::open(path).unwrap();
            assert_eq!(t.u32("candidate_id").unwrap(), vec![0, 1, 2, 3, 4, 5, 6]);
        }
    }

    /// Disjoint bands: the dedup is skipped and the pooled tables are the ones it would
    /// have written, byte for byte; and the competed rows can be left per band.
    #[test]
    fn disjoint_bands_skip_the_dedup_and_pool_the_same_bytes() {
        let dir = std::env::temp_dir().join(format!("mumdia_pool_disjoint_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let b0 = band(&dir, "d0", 0.0, &[0, 1, 2, 3], &[9.0, 1.0, 9.0, 7.0]);
        let b1 = band(&dir, "d1", 0.003, &[4, 5, 6], &[5.0, 6.0, 9.0]);
        let out = |n: &str| {
            dir.join(format!("disjoint_{n}.parquet"))
                .to_str()
                .unwrap()
                .to_string()
        };
        let bands = [b0, b1];
        let pool_as = |tag: &str, disjoint: bool| {
            let (oc, ok) = (out(&format!("{tag}_chrom")), out(&format!("{tag}_comp")));
            let stats = run(PoolParams {
                bands: &bands,
                out_psms: None,
                out_chromatograms: Some(&oc),
                out_losers: None,
                out_competed: Some(ok.as_str()),
                bands_disjoint: disjoint,
            })
            .unwrap();
            (
                stats,
                std::fs::read(&oc).unwrap(),
                std::fs::read(&ok).unwrap(),
            )
        };
        let (looked, lc, lk) = pool_as("looked", false);
        let (skipped, sc, sk) = pool_as("skipped", true);
        assert_eq!(looked.duplicates, 0);
        assert_eq!(looked, skipped);
        assert!(
            lc == sc && lk == sk,
            "the skip must not change a pooled byte"
        );
        // Left per band: no competed table, no hash, the chromatograms unchanged.
        let oc = out("perband_chrom");
        let stats = run(PoolParams {
            bands: &bands,
            out_psms: None,
            out_chromatograms: Some(&oc),
            out_losers: None,
            out_competed: None,
            bands_disjoint: true,
        })
        .unwrap();
        assert_eq!((stats.competed, stats.competed_hash), (0, None));
        assert!(std::fs::read(&oc).unwrap() == lc);
        // And never without the disjointness that makes it safe.
        let e = run(PoolParams {
            bands: &bands,
            out_psms: None,
            out_chromatograms: Some(&out("refused_chrom")),
            out_losers: None,
            out_competed: None,
            bands_disjoint: false,
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("disjoint"), "{e}");
    }
}
