//! Compete step `mumdia compete` (PLAN.md Section 4 Stage F, the compete step):
//! within each competition group keep only the best-scoring candidate before
//! target-decoy counting, so multiple plausible candidates for one elution peak
//! cannot inflate discoveries. MVP groups by base peptide (target + its decoy +
//! charge/mod variants); the grouping is configurable.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{CompeteConfig, CompeteGroupBy};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, Table};
use serde_json::json;
use tracing::info;

use crate::stages::features::FeatureSchema;

pub struct CompeteParams<'a> {
    pub features: &'a str,
    pub out: &'a str,
    pub cfg: &'a CompeteConfig,
    pub config_hash: &'a str,
}

pub fn run(p: CompeteParams) -> Result<u64> {
    let t0 = Instant::now();
    let t = Table::read(p.features)?;
    let cid = t.u32("candidate_id")?;
    let label = t.str("label")?;
    let base = t.u32("base_peptide_id")?;
    let pform = t.str("peptidoform")?;
    let protein = t.str("protein")?;
    let apex_rt = t.f64("apex_rt")?;
    let mz = t.f64("precursor_mz")?;
    let prelim = t.f64("prelim_score")?;
    let schema = FeatureSchema::read(p.features)?;
    let feat_names = &schema.feature_columns;
    let feat_cols: Vec<Vec<f64>> = feat_names.iter().map(|c| t.f64(c).unwrap()).collect();

    // Winner per competition group. The label is part of the key so a target is
    // NOT competed against its own decoy: the decoy population must survive for
    // the rescorer/FDR to have a valid null (otherwise decoys are depleted and
    // FDR is badly underestimated). Competition only removes redundant charge/
    // modification variants within targets and within decoys.
    // Key is a fixed-size tuple (base_peptide_id, label_code, bucket) instead of a
    // freshly-allocated String per PSM. The label is mapped to a small integer
    // code so it stays part of the key exactly as before; Precursor grouping uses
    // a constant bucket (0) so its equivalence classes are unchanged.
    let mut winner: HashMap<(u32, u8, i64), usize> = HashMap::new();
    for i in 0..t.nrows {
        let label_code = match label[i].as_str() {
            "target" => 0u8,
            "decoy" => 1u8,
            _ => 2u8,
        };
        let key = match p.cfg.group_by {
            CompeteGroupBy::Precursor => (base[i], label_code, 0i64),
            CompeteGroupBy::Apex => {
                let bucket = (apex_rt[i] / p.cfg.apex_rt_tolerance_s).round() as i64;
                (base[i], label_code, bucket)
            }
        };
        winner
            .entry(key)
            .and_modify(|w| {
                if prelim[i] > prelim[*w] {
                    *w = i;
                }
            })
            .or_insert(i);
    }
    let mut keep: Vec<usize> = winner.into_values().collect();
    keep.sort_unstable();

    let sel = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<_>>();
    let mut cols: Vec<Col> = vec![
        Col::U32("candidate_id".into(), keep.iter().map(|&i| cid[i]).collect()),
        Col::Str("label".into(), keep.iter().map(|&i| label[i].clone()).collect()),
        Col::U32("base_peptide_id".into(), keep.iter().map(|&i| base[i]).collect()),
        Col::Str("peptidoform".into(), keep.iter().map(|&i| pform[i].clone()).collect()),
        Col::Str("protein".into(), keep.iter().map(|&i| protein[i].clone()).collect()),
        Col::F64("apex_rt".into(), sel(&apex_rt)),
        Col::F64("precursor_mz".into(), sel(&mz)),
        Col::F64("prelim_score".into(), sel(&prelim)),
    ];
    for (fi, name) in feat_names.iter().enumerate() {
        cols.push(Col::F64(name.clone(), sel(&feat_cols[fi])));
    }
    let rows = write_table(p.out, cols)?;
    // Carry the feature schema forward for rescore.
    mumdia_io::json::write_json(&format!("{}.schema.json", p.out), &schema)?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("input_rows".to_string(), json!(t.nrows));
    stats.insert("kept".to_string(), json!(rows));
    ArtifactReport {
        logical_name: artifact::PSMS_COMPETED.0.to_string(),
        schema_name: artifact::PSMS_COMPETED.0.to_string(),
        schema_version: artifact::PSMS_COMPETED.1,
        stage: "compete".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({"group_by": format!("{:?}", p.cfg.group_by)}),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;

    info!(input = t.nrows, kept = rows, elapsed_ms = elapsed, "compete: done");
    Ok(rows)
}
