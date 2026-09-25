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
//! dropped from all four tables, so the pooled tables hold each candidate once.
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

use anyhow::{anyhow, bail, Result};
use arrow::array::{Array, BooleanArray, UInt32Array};
use arrow::compute::filter_record_batch;
use mumdia_io::table::{write_table_hashed, BatchWriter, Col, SpliceWriter, TableFile, Written};
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
    /// Where to write the overlap losers, one row per `(band, candidate_id)` the dedup
    /// dropped from that band ([`write_losers`]), or `None`. They are a function of the
    /// competed tables, so a later reader of the band tables can drop exactly the rows the
    /// pooled tables do not hold without pooling anything.
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
    /// Candidates that appeared in two bands and were kept from one.
    pub duplicates: u64,
    /// Per band, ascending, the candidates the dedup dropped from it because another band's
    /// row won them. Empty sets for disjoint bands.
    pub losers: Vec<Vec<u32>>,
    /// `overlap_losers.parquet` as written, when `out_losers` asked for it.
    pub losers_written: Option<Written>,
}

/// Write the overlap losers as `band` (the band's position in pool order) and
/// `candidate_id`, one row per dropped candidate, ascending by band then candidate. A run
/// whose bands are disjoint writes the table with no rows, which says as much.
pub fn write_losers(path: &str, losers: &[Vec<u32>]) -> Result<Written> {
    let mut band: Vec<u32> = Vec::new();
    let mut cid: Vec<u32> = Vec::new();
    for (b, set) in losers.iter().enumerate() {
        for &c in set {
            band.push(b as u32);
            cid.push(c);
        }
    }
    write_table_hashed(
        path,
        vec![
            Col::U32("band".into(), band),
            Col::U32("candidate_id".into(), cid),
        ],
    )
}

/// Read [`write_losers`]'s table back as one ascending, distinct set per band, for `n_bands`
/// bands. A band index outside the list is refused: it names a table the caller did not
/// give.
pub fn read_losers(path: &str, n_bands: usize) -> Result<Vec<Vec<u32>>> {
    let t = TableFile::open(path)?;
    let band = t.u32("band")?;
    let cid = t.u32("candidate_id")?;
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); n_bands];
    for (&b, &c) in band.iter().zip(&cid) {
        let Some(set) = out.get_mut(b as usize) else {
            bail!(
                "{path} drops candidate {c} from band {b}, but {n_bands} chromatogram \
                 table(s) were given; give every band's table, in band order"
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

/// Per band, the local ids to drop because another band's row won the candidate.
fn overlap_losers(bands: &[BandArtifacts]) -> Result<(Vec<HashSet<u32>>, u64)> {
    // Library-wide id -> (band index, best prelim_score) from the competed tables.
    let mut best: HashMap<u32, (usize, f64)> = HashMap::new();
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
    Ok((losers, duplicates))
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
    let (losers, duplicates) = if p.bands_disjoint {
        info!(
            groups = p.bands.len(),
            "pool: the bands' library row spans are disjoint; no overlap duplicate to find"
        );
        (vec![HashSet::new(); p.bands.len()], 0)
    } else {
        overlap_losers(p.bands)?
    };
    let with = |pick: fn(&BandArtifacts) -> &String| {
        p.bands
            .iter()
            .zip(losers.iter())
            .map(move |(b, l)| (pick(b).clone(), l.clone()))
            .collect::<Vec<_>>()
    };
    let psms = match p.out_psms {
        Some(out) => Some(pool_table(with(|b| &b.psms).into_iter(), out)?),
        None => None,
    };
    let chromatograms = match p.out_chromatograms {
        Some(out) => Some(pool_table(with(|b| &b.chromatograms).into_iter(), out)?),
        None => None,
    };
    let loser_lists: Vec<Vec<u32>> = losers
        .iter()
        .map(|set| {
            let mut v: Vec<u32> = set.iter().copied().collect();
            v.sort_unstable();
            v
        })
        .collect();
    let losers_written = match p.out_losers {
        Some(out) => Some(write_losers(out, &loser_lists)?),
        None => None,
    };
    let competed = match p.out_competed {
        Some(out) => Some(pool_table(with(|b| &b.competed).into_iter(), out)?),
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
