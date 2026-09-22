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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use arrow::array::{Array, BooleanArray, UInt32Array};
use arrow::compute::filter_record_batch;
use mumdia_io::table::{BatchWriter, TableFile};
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
    pub out_psms: &'a str,
    pub out_chromatograms: &'a str,
    pub out_competed: &'a str,
}

/// Row counts of the pooled tables and how many overlap duplicates were removed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoolStats {
    pub psms: u64,
    pub chromatograms: u64,
    pub competed: u64,
    /// Candidates that appeared in two bands and were kept from one.
    pub duplicates: u64,
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

/// Append every band's table into `out`, ids offset, losers dropped. Returns the row count.
fn pool_table(paths: impl Iterator<Item = (String, HashSet<u32>)>, out: &str) -> Result<u64> {
    let mut writer: Option<BatchWriter> = None;
    let mut schema: Option<Arc<arrow::datatypes::Schema>> = None;
    let mut rows = 0u64;
    for (path, drop) in paths {
        let t = TableFile::open(&path)?;
        let reader = t.batches(None, BATCH_ROWS)?;
        let this_schema = reader.schema();
        match &schema {
            None => {
                schema = Some(this_schema.clone());
                writer = Some(BatchWriter::with_row_group_rows(
                    out,
                    this_schema.clone(),
                    ROW_GROUP_ROWS,
                )?);
            }
            Some(first) => {
                if first.fields() != this_schema.fields() {
                    bail!(
                        "{path} has a different schema from the first pooled table; the group \
                         artifacts must come from the same configuration"
                    );
                }
            }
        }
        let cid_ix = this_schema
            .index_of("candidate_id")
            .map_err(|_| anyhow!("{path} has no candidate_id column"))?;
        let w = writer.as_mut().expect("opened above");
        for b in reader {
            let b = b?;
            let keep: Option<Vec<bool>> = {
                let cid = b
                    .column(cid_ix)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| anyhow!("{path}: candidate_id is not u32"))?;
                if cid.null_count() > 0 {
                    bail!("{path}: candidate_id has nulls");
                }
                (!drop.is_empty()).then(|| cid.values().iter().map(|c| !drop.contains(c)).collect())
            };
            // The band stages already wrote library-wide ids (`Library::global_offset` is
            // added on the way out of extract), so pooling appends rather than rewrites.
            let batch = match keep {
                Some(k) => filter_record_batch(&b, &BooleanArray::from(k))?,
                None => b,
            };
            if batch.num_rows() > 0 {
                w.write(&batch)?;
                rows += batch.num_rows() as u64;
            }
        }
    }
    match writer {
        Some(w) => {
            w.close()?;
            Ok(rows)
        }
        None => bail!("pool: no group tables to pool into {out}"),
    }
}

pub fn run(p: PoolParams) -> Result<PoolStats> {
    let t0 = Instant::now();
    if p.bands.is_empty() {
        bail!("pool: no groups");
    }
    let (losers, duplicates) = overlap_losers(p.bands)?;
    let with = |pick: fn(&BandArtifacts) -> &String| {
        p.bands
            .iter()
            .zip(losers.iter())
            .map(move |(b, l)| (pick(b).clone(), l.clone()))
            .collect::<Vec<_>>()
    };
    let stats = PoolStats {
        psms: pool_table(with(|b| &b.psms).into_iter(), p.out_psms)?,
        chromatograms: pool_table(with(|b| &b.chromatograms).into_iter(), p.out_chromatograms)?,
        competed: pool_table(with(|b| &b.competed).into_iter(), p.out_competed)?,
        duplicates,
    };
    info!(
        groups = p.bands.len(),
        psms = stats.psms,
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
            out_psms: &op,
            out_chromatograms: &oc,
            out_competed: &ok,
        })
        .unwrap();
        assert_eq!(stats.duplicates, 2);
        assert_eq!(stats.competed, 7);
        for path in [&op, &oc, &ok] {
            let t = TableFile::open(path).unwrap();
            let cid = t.u32("candidate_id").unwrap();
            assert_eq!(cid, vec![0, 1, 2, 3, 4, 5, 6], "{path}");
            let x = t.f64("x").unwrap();
            // Id 3 came from band 0 (x = 3.000), id 4 from band 1 (x = 4.003).
            assert!(
                (x[3] - 3.0).abs() < 1e-9 && (x[4] - 4.003).abs() < 1e-9,
                "{path}: {x:?}"
            );
        }
    }
}
