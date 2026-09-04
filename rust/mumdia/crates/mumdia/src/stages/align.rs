//! Stage D2 `mumdia align` (PLAN.md Stage D2): put all runs on a common RT
//! coordinate. Choose a reference run; for peptides confidently identified in
//! both a run and the reference, fit a smooth monotone RT mapping (LOESS) from
//! the run's observed RT to the reference RT, and record the residual spread
//! (which sets how tight an MBR window can be). MVP is 3D, so IM is omitted.
//!
//! This is an experiment-level stage. With a single run it degenerates to the
//! identity mapping; it is exercised on multiple runs (unit-tested on crafted
//! two-run input; real multi-run validation needs a multi-file experiment).

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, TableFile};
use serde_json::json;
use tracing::info;

use crate::calibrate::{percentile, Loess};

pub struct AlignParams<'a> {
    /// One seed_psms.parquet per run; the first is the reference.
    pub seeds: &'a [String],
    pub out: &'a str,
    pub q_train: f64,
    pub grid_n: usize,
    pub config_hash: &'a str,
}

/// (base_peptide_id -> observed RT) for confident target seed PSMs, best per peptide.
fn confident_rts(path: &str, q_train: f64) -> Result<HashMap<u32, f64>> {
    let t = TableFile::open(path)?;
    let base = t.u32("base_peptide_id")?;
    let q = t.f64("spectrum_q")?;
    let rt = t.f64("observed_rt")?;
    let score = t.f64("score")?;
    let label = t.str("label")?;
    crate::fdr::validate_labels(&label)?;
    let mut best: HashMap<u32, (f64, f64)> = HashMap::new(); // base -> (score, rt)
    for i in 0..t.nrows {
        if label[i] == "decoy" || q[i] > q_train {
            continue;
        }
        let e = best.entry(base[i]).or_insert((f64::NEG_INFINITY, 0.0));
        if score[i] > e.0 {
            *e = (score[i], rt[i]);
        }
    }
    Ok(best.into_iter().map(|(k, (_, rt))| (k, rt)).collect())
}

pub fn run(p: AlignParams) -> Result<u64> {
    let t0 = Instant::now();
    assert!(!p.seeds.is_empty(), "align needs at least one run");
    let ref_rts = confident_rts(&p.seeds[0], p.q_train)?;
    let ref_rt_values: Vec<f64> = ref_rts.values().cloned().collect();
    let (grid_lo, grid_hi) = (
        ref_rt_values
            .iter()
            .cloned()
            .fold(f64::MAX, f64::min)
            .min(0.0),
        ref_rt_values
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max)
            .max(1.0),
    );

    let (mut run_c, mut src_c, mut ref_c, mut resid_c) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());

    for (run_id, seed) in p.seeds.iter().enumerate() {
        let this = confident_rts(seed, p.q_train)?;
        // shared peptides with the reference
        let mut xs = Vec::new(); // this run RT
        let mut ys = Vec::new(); // reference RT
        for (base, rt) in &this {
            if let Some(rref) = ref_rts.get(base) {
                xs.push(*rt);
                ys.push(*rref);
            }
        }
        let (lo, hi) = if run_id == 0 {
            (grid_lo, grid_hi)
        } else {
            (
                xs.iter().cloned().fold(f64::MAX, f64::min).min(grid_lo),
                xs.iter().cloned().fold(f64::MIN, f64::max).max(grid_hi),
            )
        };
        let loess = if xs.len() >= 4 {
            Some(Loess::fit(&xs, &ys, 0.4, p.grid_n.max(2)))
        } else {
            None
        };
        // residual spread on shared peptides
        let resid = if let Some(l) = &loess {
            let r: Vec<f64> = xs
                .iter()
                .zip(&ys)
                .map(|(x, y)| (y - l.predict(*x)).abs())
                .collect();
            percentile(&r, 0.95)
        } else {
            0.0
        };
        // emit the mapping on a grid (identity for the reference or too few shared)
        let gn = p.grid_n.max(2);
        for g in 0..gn {
            let src = lo + (hi - lo) * g as f64 / (gn - 1) as f64;
            let refrt = match &loess {
                Some(l) if run_id != 0 => l.predict(src),
                _ => src, // identity
            };
            run_c.push(run_id as u32);
            src_c.push(src);
            ref_c.push(refrt);
            resid_c.push(resid);
        }
        info!(
            run = run_id,
            shared = xs.len(),
            residual_p95 = resid,
            "align: run mapped"
        );
    }

    let rows = write_table(
        p.out,
        vec![
            Col::U32("run_id".into(), run_c),
            Col::F64("source_rt".into(), src_c),
            Col::F64("reference_rt".into(), ref_c),
            Col::F64("residual_spread".into(), resid_c),
        ],
    )?;

    let elapsed = t0.elapsed().as_millis();
    ArtifactReport {
        logical_name: "alignment".to_string(),
        schema_name: "alignment".to_string(),
        schema_version: 1,
        stage: "align".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({"runs": p.seeds.len(), "q_train": p.q_train}),
        stats: Default::default(),
        model_identity: None,
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;
    info!(
        runs = p.seeds.len(),
        rows,
        elapsed_ms = elapsed,
        "align: done"
    );
    Ok(rows)
}
