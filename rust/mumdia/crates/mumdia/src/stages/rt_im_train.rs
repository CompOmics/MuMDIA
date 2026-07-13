//! Stage B `mumdia rt-im-train`: per-run RT calibration and windows (PLAN.md
//! Stage B). Calibrates the run-independent predicted iRT to observed RT from
//! confident seed PSMs, then sets a per-candidate RT window from the residuals.
//! MVP is 3D, so IM columns are null. The sidecars are not re-run here.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{CalibrationMethod, RtImTrainConfig};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, Table};
use serde_json::json;
use tracing::{info, warn};

use crate::calibrate::{linear_fit, percentile, Loess};

pub struct RtImTrainParams<'a> {
    pub seed_psms: &'a str,
    pub library_precursors: &'a str,
    pub out_windows: &'a str,
    pub out_cal: &'a str,
    pub cfg: &'a RtImTrainConfig,
    pub config_hash: &'a str,
}

pub fn run(p: RtImTrainParams) -> Result<u64> {
    let t0 = Instant::now();

    // Library predicted iRT, keyed by candidate_id (single source of truth, so
    // a patched/updated library iRT is used for both training and application).
    let lib = Table::read(p.library_precursors)?;
    let lib_cid = lib.u32("candidate_id")?;
    let lib_irt = lib.f32("predicted_irt")?;
    let mut irt_by_cid: HashMap<u32, f64> = HashMap::with_capacity(lib.nrows);
    for i in 0..lib.nrows {
        irt_by_cid.insert(lib_cid[i], lib_irt[i] as f64);
    }

    // Training rows: confident seed PSMs, one apex (best score) per peptide;
    // predicted iRT is joined from the library by candidate_id.
    let seed = Table::read(p.seed_psms)?;
    let s_cid = seed.u32("candidate_id")?;
    let s_base = seed.u32("base_peptide_id")?;
    let s_q = seed.f64("spectrum_q")?;
    let s_score = seed.f64("score")?;
    let s_rt = seed.f64("observed_rt")?;

    let mut best_per_pep: HashMap<u32, (f64, f64, f64)> = HashMap::new(); // base -> (score, irt, rt)
    for i in 0..seed.nrows {
        if s_q[i] >= p.cfg.q_train {
            continue;
        }
        let irt = match irt_by_cid.get(&s_cid[i]) {
            Some(v) => *v,
            None => continue,
        };
        let e = best_per_pep.entry(s_base[i]).or_insert((f64::NEG_INFINITY, 0.0, 0.0));
        if s_score[i] > e.0 {
            *e = (s_score[i], irt, s_rt[i]);
        }
    }
    let train_irt: Vec<f64> = best_per_pep.values().map(|v| v.1).collect();
    let train_rt: Vec<f64> = best_per_pep.values().map(|v| v.2).collect();
    let n_train = train_irt.len();
    info!(n_train, "rt-im-train: training points");

    // Fit calibration predicted_irt -> observed RT (seconds).
    let (slope, intercept) = linear_fit(&train_irt, &train_rt);
    let use_loess = matches!(p.cfg.calibration_method, CalibrationMethod::Loess)
        && n_train >= p.cfg.min_seed_for_calibration;
    let loess = if use_loess {
        Some(Loess::fit(&train_irt, &train_rt, p.cfg.loess_span, 200))
    } else {
        None
    };

    let predict = |irt: f64| -> f64 {
        match &loess {
            Some(l) => l.predict(irt),
            None => slope * irt + intercept,
        }
    };

    // Residuals and RT window.
    let (w_rt, status) = if n_train >= 2 {
        let resid: Vec<f64> = train_irt
            .iter()
            .zip(&train_rt)
            .map(|(x, y)| (y - predict(*x)).abs())
            .collect();
        let w = percentile(&resid, p.cfg.p_rt) * p.cfg.rt_window_multiplier;
        let w = w.max(1.0);
        let status = if use_loess { "loess" } else { "linear" };
        (w, status.to_string())
    } else {
        warn!("rt-im-train: too few seeds; using fallback fixed RT window");
        (p.cfg.fallback_rt_window_s, "fallback_fixed".to_string())
    };

    // Apply to every library candidate.
    let cid = lib_cid;
    let irt = lib_irt;
    let n = lib.nrows;
    let (mut cid_c, mut cal_c, mut lo_c, mut hi_c) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut im_c, mut imlo_c, mut imhi_c): (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let cal = predict(irt[i] as f64);
        cid_c.push(cid[i]);
        cal_c.push(cal);
        lo_c.push(cal - w_rt);
        hi_c.push(cal + w_rt);
        im_c.push(None);
        imlo_c.push(None);
        imhi_c.push(None);
    }

    let rows = write_table(
        p.out_windows,
        vec![
            Col::U32("candidate_id".into(), cid_c),
            Col::F64("rt_pred_cal".into(), cal_c),
            Col::F64("rt_lo".into(), lo_c),
            Col::F64("rt_hi".into(), hi_c),
            Col::OptF64("im_pred_cal".into(), im_c),
            Col::OptF64("im_lo".into(), imlo_c),
            Col::OptF64("im_hi".into(), imhi_c),
        ],
    )?;

    // cal.json
    mumdia_io::json::write_json(
        p.out_cal,
        &json!({
            "method": if use_loess {"loess"} else {"linear"},
            "slope": slope,
            "intercept": intercept,
            "w_rt": w_rt,
            "p_rt": p.cfg.p_rt,
            "multiplier": p.cfg.rt_window_multiplier,
            "n_train": n_train,
            "calibration_status": status,
        }),
    )?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("n_train".to_string(), json!(n_train));
    stats.insert("w_rt".to_string(), json!(w_rt));
    stats.insert("calibration_status".to_string(), json!(status));
    ArtifactReport {
        logical_name: artifact::RUN_WINDOWS.0.to_string(),
        schema_name: artifact::RUN_WINDOWS.0.to_string(),
        schema_version: artifact::RUN_WINDOWS.1,
        stage: "rt-im-train".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out_windows)?,
        params: json!({"q_train": p.cfg.q_train, "p_rt": p.cfg.p_rt, "method": format!("{:?}", p.cfg.calibration_method)}),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    }
    .write_for(p.out_windows)?;

    info!(rows, w_rt, status, elapsed_ms = elapsed, "rt-im-train: done");
    Ok(rows)
}
