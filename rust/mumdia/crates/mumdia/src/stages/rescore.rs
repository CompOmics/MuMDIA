//! Stage F `mumdia rescore` (PLAN.md Stage F): rescore competed PSMs across the
//! experiment and compute native target-decoy q-values at PSM and peptide level.
//! MVP default is the native semi-supervised rescorer; Mokapot / percolator.exe
//! are optional strategies over the same PIN/feature contract.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{RescoreConfig, RescorerKind};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, Table};
use serde_json::json;
use tracing::{info, warn};

use crate::fdr::{entrapment_q, target_decoy_q};
use crate::rescoring::{percolator_lite, RescoreInput};
use crate::stages::features::FeatureSchema;

/// Which empirical null the q-values are computed against.
#[derive(Clone, Copy, PartialEq)]
enum QMode {
    /// Target-decoy competition (native / mokapot / percolator paths).
    Decoy,
    /// Spike-in entrapment population (the `Entrapment` classifier).
    Entrapment,
}

pub struct RescoreParams<'a> {
    /// One or more competed feature tables (experiment-wide concat).
    pub competed: &'a [String],
    pub out: &'a str,
    /// Working directory + script dir for the Mokapot sidecar (when selected).
    pub work_dir: &'a str,
    pub script_dir: &'a str,
    pub cfg: &'a RescoreConfig,
    pub config_hash: &'a str,
}

pub fn run(p: RescoreParams) -> Result<u64> {
    let t0 = Instant::now();

    // Concatenate competed inputs.
    let (mut cid, mut label, mut base, mut pform, mut protein, mut charge, mut prelim) = (
        Vec::new(),
        Vec::<String>::new(),
        Vec::new(),
        Vec::<String>::new(),
        Vec::<String>::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut feats: Vec<Vec<f64>> = Vec::new();
    let mut mz: Vec<f64> = Vec::new();
    // Feature list is taken from the schema companion of the first input so the
    // classifier input matches the set the features stage produced.
    let feat_names = FeatureSchema::read(&p.competed[0])?.feature_columns;
    for path in p.competed {
        let t = Table::read(path)?;
        let c = t.u32("candidate_id")?;
        let l = t.str("label")?;
        let b = t.u32("base_peptide_id")?;
        let pf = t.str("peptidoform")?;
        let pr = t.str("protein")?;
        let z = t.f64("charge")?; // carried as an f64 feature
        let pl = t.f64("prelim_score")?;
        let pm = t.f64("precursor_mz")?;
        let fcols: Vec<Vec<f64>> = feat_names.iter().map(|n| t.f64(n).unwrap()).collect();
        for i in 0..t.nrows {
            cid.push(c[i]);
            label.push(l[i].clone());
            base.push(b[i]);
            pform.push(pf[i].clone());
            protein.push(pr[i].clone());
            charge.push(z[i] as i32);
            prelim.push(pl[i]);
            mz.push(pm[i]);
            feats.push((0..feat_names.len()).map(|k| fcols[k][i]).collect());
        }
    }
    let is_decoy: Vec<bool> = label.iter().map(|l| l == "decoy").collect();
    let (is_entrapment, is_real_target) = classify_entrapment(p.cfg, &protein, &is_decoy);
    let n = cid.len();
    info!(psms = n, "rescore: loaded competed PSMs");

    // Track the path actually taken so the report reflects reality rather than a
    // hardcoded label, and pick the null the q-values are computed against.
    let mut classifier_used = "native_tda";
    let mut model_identity = "native-percolator-lite-v1".to_string();
    let mut qmode = QMode::Decoy;

    let scores = if n == 0 {
        Vec::new()
    } else {
        match p.cfg.classifier {
            RescorerKind::Mokapot => match run_mokapot(&p, &feat_names, &cid, &label, &pform, &protein, &mz, &feats) {
                Ok(s) => {
                    info!("rescore: using Mokapot scores");
                    classifier_used = "mokapot";
                    model_identity = "mokapot".to_string();
                    s
                }
                Err(e) => {
                    if p.cfg.strict {
                        anyhow::bail!("rescore: Mokapot sidecar failed ({e}) and rescore.strict=true");
                    }
                    warn!("rescore: Mokapot failed ({e}); falling back to native_tda");
                    native_scores(&p, &feats, &is_decoy, &cid, &prelim)
                }
            },
            RescorerKind::Percolator => {
                if p.cfg.strict {
                    anyhow::bail!("rescore: classifier=percolator but percolator.exe is not wired, and rescore.strict=true");
                }
                warn!("rescore: percolator.exe path not wired; using native_tda");
                native_scores(&p, &feats, &is_decoy, &cid, &prelim)
            }
            RescorerKind::NativeTda => native_scores(&p, &feats, &is_decoy, &cid, &prelim),
            RescorerKind::Entrapment => {
                let n_ent = is_entrapment.iter().filter(|&&b| b).count();
                if p.cfg.entrapment_marker.is_none() || n_ent == 0 {
                    if p.cfg.strict {
                        anyhow::bail!(
                            "rescore: classifier=entrapment but no entrapment PSMs matched \
                             (marker={:?}, n_ent={n_ent}) and rescore.strict=true; set \
                             rescore.entrapment_marker to the spike-in accession substring",
                            p.cfg.entrapment_marker
                        );
                    }
                    warn!(
                        entrapment_psms = n_ent,
                        "rescore: classifier=entrapment but no entrapment PSMs \
                         (set rescore.entrapment_marker to the spike-in accession \
                         substring); falling back to native_tda"
                    );
                    native_scores(&p, &feats, &is_decoy, &cid, &prelim)
                } else if p.cfg.python.is_some() {
                    match run_entrapment_gbm(&p, &feat_names, &cid, &base, &is_entrapment, &is_decoy, &feats) {
                        Ok(s) => {
                            info!(entrapment_psms = n_ent, "rescore: using entrapment GBM sidecar scores");
                            classifier_used = "entrapment_gbm";
                            model_identity = "entrapment-gbm-sidecar-v1".to_string();
                            qmode = QMode::Entrapment;
                            s
                        }
                        Err(e) => {
                            if p.cfg.strict {
                                anyhow::bail!("rescore: entrapment GBM sidecar failed ({e}) and rescore.strict=true");
                            }
                            warn!("rescore: entrapment GBM sidecar failed ({e}); using native linear entrapment fallback");
                            classifier_used = "entrapment_native";
                            model_identity = "native-percolator-lite-entrapment-v1".to_string();
                            qmode = QMode::Entrapment;
                            native_scores(&p, &feats, &is_entrapment, &cid, &prelim)
                        }
                    }
                } else {
                    info!(
                        entrapment_psms = n_ent,
                        "rescore: classifier=entrapment, no rescore.python; using native linear entrapment rescorer"
                    );
                    classifier_used = "entrapment_native";
                    model_identity = "native-percolator-lite-entrapment-v1".to_string();
                    qmode = QMode::Entrapment;
                    native_scores(&p, &feats, &is_entrapment, &cid, &prelim)
                }
            }
        }
    };

    // PSM-level q-values against the selected null.
    let psm_q = match qmode {
        QMode::Decoy => {
            let sd: Vec<(f64, bool)> = scores.iter().cloned().zip(is_decoy.iter().cloned()).collect();
            target_decoy_q(&sd)
        }
        QMode::Entrapment => {
            entrapment_q(&scores, &is_entrapment, &is_real_target, p.cfg.entrapment_ratio)
        }
    };

    // Peptide-level q: reduce to best PSM per base peptide, q on that set, map
    // back. Protein-group q: same over the protein-accession-set string (the MVP
    // grouping; decoys carry a DECOY_ prefix). Full parsimony/razor is a later
    // option (PLAN.md Stage G). Group score = best member PSM score.
    let peptide_q = grouped_q(&base, &scores, &is_decoy, &is_entrapment, &is_real_target, qmode, p.cfg.entrapment_ratio);
    // Intern the protein-accession-set strings to dense u32 ids once (first-seen
    // order) so protein-group grouping runs over integers exactly like the peptide
    // path, avoiding hashing/cloning ~574k strings per grouped_q lookup. The map is
    // bijective, so grouping and the resulting per-PSM q-values are unchanged.
    let protein_id: Vec<u32> = {
        let mut interner: HashMap<&str, u32> = HashMap::new();
        let mut ids = Vec::with_capacity(protein.len());
        for s in &protein {
            let next = interner.len() as u32;
            ids.push(*interner.entry(s.as_str()).or_insert(next));
        }
        ids
    };
    let pg_q = grouped_q(&protein_id, &scores, &is_decoy, &is_entrapment, &is_real_target, qmode, p.cfg.entrapment_ratio);
    // Multi-context q (PLAN.md Section 8 rescore): single run, so run-specific ==
    // experiment-wide == global. Distinct column kept for multi-run forward-compat.
    let global_q = psm_q.clone();

    // Reported IDs: real targets in entrapment mode (spike-in excluded), else all
    // non-decoy targets.
    let is_reported: Vec<bool> = match qmode {
        QMode::Decoy => is_decoy.iter().map(|d| !d).collect(),
        QMode::Entrapment => is_real_target.clone(),
    };

    let n_psm_1 = (0..n).filter(|&i| is_reported[i] && psm_q[i] <= 0.01).count();
    let n_pep_1 = {
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            if is_reported[i] && peptide_q[i] <= 0.01 {
                seen.insert(base[i]);
            }
        }
        seen.len()
    };
    let n_pg_1 = {
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            if is_reported[i] && pg_q[i] <= 0.01 {
                seen.insert(protein_id[i]);
            }
        }
        seen.len()
    };
    // Entrapment leak: spike-in peptides passing the 1% gate. A running check on
    // FDR validity (should track the reported q if the null is well-modelled).
    let n_entrap_1 = {
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            if is_entrapment[i] && peptide_q[i] <= 0.01 {
                seen.insert(base[i]);
            }
        }
        seen.len()
    };

    let rows = write_table(
        p.out,
        vec![
            Col::U32("candidate_id".into(), cid),
            Col::Str("peptidoform".into(), pform),
            Col::I32("charge".into(), charge),
            Col::Str("label".into(), label),
            // `protein` feeds two columns; clone once here and move the original below.
            Col::Str("protein".into(), protein.clone()),
            Col::U32("base_peptide_id".into(), base),
            Col::F64("score".into(), scores),
            Col::F64("q_value".into(), psm_q),
            Col::F64("peptide_q_value".into(), peptide_q),
            Col::Str("protein_group".into(), protein),
            Col::F64("pg_q_value".into(), pg_q),
            Col::F64("global_q_value".into(), global_q),
            Col::F64("prelim_score".into(), prelim),
        ],
    )?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("psms".to_string(), json!(n));
    stats.insert("classifier".to_string(), json!(classifier_used));
    stats.insert("target_psms_at_1pct".to_string(), json!(n_psm_1));
    stats.insert("target_peptides_at_1pct".to_string(), json!(n_pep_1));
    stats.insert("target_protein_groups_at_1pct".to_string(), json!(n_pg_1));
    if qmode == QMode::Entrapment {
        stats.insert("entrapment_ratio".to_string(), json!(p.cfg.entrapment_ratio));
        stats.insert("entrapment_peptides_at_1pct".to_string(), json!(n_entrap_1));
    }
    ArtifactReport {
        logical_name: artifact::PSMS_SCORED.0.to_string(),
        schema_name: artifact::PSMS_SCORED.0.to_string(),
        schema_version: artifact::PSMS_SCORED.1,
        stage: "rescore".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({"classifier": classifier_used, "folds": p.cfg.folds, "num_iter": p.cfg.num_iter, "train_fdr": p.cfg.train_fdr}),
        stats,
        model_identity: Some(model_identity),
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;

    info!(
        psms = n,
        target_psms_at_1pct = n_psm_1,
        target_peptides_at_1pct = n_pep_1,
        elapsed_ms = elapsed,
        "rescore: done"
    );
    Ok(rows)
}

/// Native semi-supervised rescorer scores.
fn native_scores(
    p: &RescoreParams,
    feats: &[Vec<f64>],
    is_decoy: &[bool],
    cid: &[u32],
    prelim: &[f64],
) -> Vec<f64> {
    percolator_lite(RescoreInput {
        features: feats,
        is_decoy,
        candidate_id: cid,
        init_score: prelim,
        folds: p.cfg.folds,
        num_iter: p.cfg.num_iter,
        train_fdr: p.cfg.train_fdr,
    })
}

/// Per-PSM entrapment classification from the protein-accession string. A target
/// is entrapment when its protein contains `entrapment_marker`, does not contain
/// `entrapment_exclude` (if set), and does not match any
/// `entrapment_contaminant_markers` (genuine contaminants inside the spike-in
/// proteome, e.g. keratins, which are real and so must not be used as negatives).
/// Everything else non-decoy is a real target. Decoys are neither. Returns
/// `(is_entrapment, is_real_target)`.
fn classify_entrapment(
    cfg: &RescoreConfig,
    protein: &[String],
    is_decoy: &[bool],
) -> (Vec<bool>, Vec<bool>) {
    let marker = cfg.entrapment_marker.as_deref();
    let exclude = cfg.entrapment_exclude.as_deref();
    let contaminants = &cfg.entrapment_contaminant_markers;
    let n = protein.len();
    let mut ent = vec![false; n];
    let mut real = vec![false; n];
    for i in 0..n {
        if is_decoy[i] {
            continue;
        }
        let is_ent = match marker {
            Some(m) => {
                protein[i].contains(m)
                    && exclude.map_or(true, |e| !protein[i].contains(e))
                    && !contaminants.iter().any(|c| protein[i].contains(c.as_str()))
            }
            None => false,
        };
        ent[i] = is_ent;
        real[i] = !is_ent;
    }
    (ent, real)
}

/// Reduce PSMs to the best score per group key, compute group q-values against
/// the selected null, and map back to a per-PSM vector. Mirrors the PSM-level
/// logic for peptide- and protein-group-level q.
fn grouped_q<K: std::hash::Hash + Eq + Clone>(
    keys: &[K],
    scores: &[f64],
    is_decoy: &[bool],
    is_entrapment: &[bool],
    is_real: &[bool],
    qmode: QMode,
    ratio: f64,
) -> Vec<f64> {
    let n = scores.len();
    let mut best: HashMap<K, (f64, bool, bool, bool)> = HashMap::new();
    for i in 0..n {
        let e = best
            .entry(keys[i].clone())
            .or_insert((f64::NEG_INFINITY, is_decoy[i], is_entrapment[i], is_real[i]));
        if scores[i] > e.0 {
            *e = (scores[i], is_decoy[i], is_entrapment[i], is_real[i]);
        }
    }
    let ks: Vec<K> = best.keys().cloned().collect();
    let qv = match qmode {
        QMode::Decoy => {
            let sd: Vec<(f64, bool)> = ks.iter().map(|k| (best[k].0, best[k].1)).collect();
            target_decoy_q(&sd)
        }
        QMode::Entrapment => {
            let sc: Vec<f64> = ks.iter().map(|k| best[k].0).collect();
            let e: Vec<bool> = ks.iter().map(|k| best[k].2).collect();
            let r: Vec<bool> = ks.iter().map(|k| best[k].3).collect();
            entrapment_q(&sc, &e, &r, ratio)
        }
    };
    let qmap: HashMap<K, f64> = ks.into_iter().zip(qv).collect();
    (0..n).map(|i| qmap[&keys[i]]).collect()
}

/// Run the entrapment GBM sidecar: write a Parquet of features + meta columns,
/// invoke `entrapment_worker.py`, read back candidate_id + score. Positives are
/// real targets, negatives are spike-in (entrapment) targets; the worker fits a
/// gradient-boosted classifier out-of-fold by base peptide (PLAN.md Section 3.2
/// positional-CLI file contract, as for the MS2PIP/DeepLC/Mokapot sidecars).
#[allow(clippy::too_many_arguments)]
fn run_entrapment_gbm(
    p: &RescoreParams,
    feat_names: &[String],
    cid: &[u32],
    base: &[u32],
    is_entrapment: &[bool],
    is_decoy: &[bool],
    feats: &[Vec<f64>],
) -> Result<Vec<f64>> {
    let python = p
        .cfg
        .python
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("classifier=entrapment GBM requires rescore.python"))?;
    std::fs::create_dir_all(p.work_dir).ok();
    let inp = format!("{}/entrapment_in.parquet", p.work_dir);
    let outp = format!("{}/entrapment_out.parquet", p.work_dir);

    let mut cols = vec![
        Col::U32("candidate_id".into(), cid.to_vec()),
        Col::U32("base_peptide_id".into(), base.to_vec()),
        Col::I32("is_entrapment".into(), is_entrapment.iter().map(|&b| b as i32).collect()),
        Col::I32("is_decoy".into(), is_decoy.iter().map(|&b| b as i32).collect()),
    ];
    for (fi, name) in feat_names.iter().enumerate() {
        cols.push(Col::F64(name.clone(), (0..cid.len()).map(|i| feats[i][fi]).collect()));
    }
    write_table(&inp, cols)?;

    let script = crate::sidecar::resolve_script(p.script_dir, "entrapment_worker.py");
    let status = std::process::Command::new(python)
        .arg(&script)
        .arg(&inp)
        .arg(&outp)
        .arg(p.cfg.folds.to_string())
        .env("PYTHONUTF8", "1")
        .status()?;
    if !status.success() {
        anyhow::bail!("entrapment worker exited with {status}");
    }

    let t = Table::read(&outp)?;
    let ocid = t.u32("candidate_id")?;
    let osc = t.f64("score")?;
    let map: HashMap<u32, f64> = ocid.into_iter().zip(osc).collect();
    let min = map.values().cloned().fold(f64::INFINITY, f64::min);
    Ok(cid.iter().map(|c| *map.get(c).unwrap_or(&(min - 1.0))).collect())
}

/// Run Mokapot over a PIN written from the competed set; return scores aligned
/// to the input candidate order (PLAN.md Section 3.2 file contract).
#[allow(clippy::too_many_arguments)]
fn run_mokapot(
    p: &RescoreParams,
    feat_names: &[String],
    cid: &[u32],
    label: &[String],
    pform: &[String],
    protein: &[String],
    mz: &[f64],
    feats: &[Vec<f64>],
) -> Result<Vec<f64>> {
    use std::fmt::Write as _;
    let python = p
        .cfg
        .python
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("classifier=mokapot requires rescore.python"))?;
    std::fs::create_dir_all(p.work_dir).ok();
    let pin = format!("{}/rescore.pin", p.work_dir);
    let outp = format!("{}/mokapot_out.parquet", p.work_dir);

    let mut s = String::new();
    s.push_str("SpecId\tLabel\tScanNr\tExpMass\tCalcMass\t");
    s.push_str(&feat_names.join("\t"));
    s.push_str("\tPeptide\tProteins\n");
    for i in 0..cid.len() {
        let lab = if label[i] == "decoy" { -1 } else { 1 };
        write!(s, "cand_{}\t{}\t{}\t{:.5}\t{:.5}\t", cid[i], lab, cid[i], mz[i], mz[i]).ok();
        for fi in 0..feat_names.len() {
            write!(s, "{:.6}\t", feats[i][fi]).ok();
        }
        writeln!(s, "-.{}.-\t{}", pform[i], protein[i]).ok();
    }
    std::fs::write(&pin, s)?;

    let script = crate::sidecar::resolve_script(p.script_dir, "mokapot_worker.py");
    let status = std::process::Command::new(python)
        .arg(&script)
        .arg(&pin)
        .arg(&outp)
        .env("PYTHONUTF8", "1")
        .status()?;
    if !status.success() {
        anyhow::bail!("mokapot worker exited with {status}");
    }

    let t = Table::read(&outp)?;
    let ocid = t.u32("candidate_id")?;
    let osc = t.f64("score")?;
    let map: HashMap<u32, f64> = ocid.into_iter().zip(osc).collect();
    let min = map.values().cloned().fold(f64::INFINITY, f64::min);
    Ok(cid.iter().map(|c| *map.get(c).unwrap_or(&(min - 1.0))).collect())
}
