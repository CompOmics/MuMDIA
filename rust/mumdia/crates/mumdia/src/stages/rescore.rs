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
    if p.competed.is_empty() {
        anyhow::bail!("rescore requires at least one competed input");
    }
    if p.cfg.folds < 2 {
        anyhow::bail!("rescore.folds must be >= 2 for out-of-fold scoring");
    }

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
    let mut apex_rt: Vec<f64> = Vec::new();
    let mut elution_lo: Vec<f64> = Vec::new();
    let mut elution_hi: Vec<f64> = Vec::new();
    // `source` = index of the competed input each PSM came from (0..N). For a
    // single-run rescore this is all-zero; for an experiment-wide rescore over
    // several files it lets quant map each scored PSM back to its run, and it is
    // why the Mokapot PIN below keys on a unique row index rather than
    // candidate_id (which is the library index and repeats across runs).
    let mut source: Vec<u32> = Vec::new();
    // Top-K peak rank per row (#7), for the per-candidate best-peak collapse below.
    let mut peak_rank: Vec<i32> = Vec::new();
    // Feature list is taken from the schema companion of the first input. Every
    // subsequent companion must match exactly: silently concatenating differing
    // feature order/sets would train and score on semantically misaligned columns.
    let expected_schema = FeatureSchema::read(&p.competed[0])?;
    let feat_names = expected_schema.feature_columns.clone();
    for (src, path) in p.competed.iter().enumerate() {
        let actual_schema = FeatureSchema::read(path)?;
        validate_feature_schema(&expected_schema, &actual_schema, path)?;
        let t = Table::read(path)?;
        let c = t.u32("candidate_id")?;
        let l = t.str("label")?;
        let b = t.u32("base_peptide_id")?;
        let pf = t.str("peptidoform")?;
        let pr = t.str("protein")?;
        let z = t.f64("charge")?; // carried as an f64 feature
        let pl = t.f64("prelim_score")?;
        let pm = t.f64("precursor_mz")?;
        let ar = t.f64("apex_rt")?;
        let elo = t.f64("elution_lo")?;
        let ehi = t.f64("elution_hi")?;
        let pkr = t.i32("peak_rank").unwrap_or_else(|_| vec![0; t.nrows]);
        let fcols: Vec<Vec<f64>> = feat_names.iter().map(|n| t.f64(n)).collect::<Result<_>>()?;
        for i in 0..t.nrows {
            cid.push(c[i]);
            peak_rank.push(pkr[i]);
            label.push(l[i].clone());
            base.push(b[i]);
            pform.push(pf[i].clone());
            protein.push(pr[i].clone());
            charge.push(z[i] as i32);
            prelim.push(pl[i]);
            mz.push(pm[i]);
            apex_rt.push(ar[i]);
            elution_lo.push(elo[i]);
            elution_hi.push(ehi[i]);
            source.push(src as u32);
            feats.push((0..feat_names.len()).map(|k| fcols[k][i]).collect());
        }
    }
    crate::fdr::validate_labels(&label)?;
    let is_decoy: Vec<bool> = label.iter().map(|l| l == "decoy").collect();
    let (mut is_entrapment, mut is_real_target) = classify_entrapment(p.cfg, &protein, &is_decoy);
    let mut is_decoy = is_decoy;
    let mut n = cid.len();
    if n > 0 {
        let n_decoys = is_decoy.iter().filter(|&&v| v).count();
        let n_targets = n - n_decoys;
        if n_targets == 0 || n_decoys == 0 {
            anyhow::bail!(
                "rescore requires both target and decoy PSMs for valid FDR \
                 (targets={n_targets}, decoys={n_decoys})"
            );
        }
        for (row, values) in feats.iter().enumerate() {
            if let Some((feature, value)) = values
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                anyhow::bail!(
                    "rescore input contains non-finite feature '{}' at flat row {row}: {value}",
                    feat_names[feature]
                );
            }
            if !prelim[row].is_finite() || !mz[row].is_finite() {
                anyhow::bail!(
                    "rescore input contains non-finite prelim_score/precursor_mz at flat row {row}"
                );
            }
        }
    }
    info!(psms = n, "rescore: loaded competed PSMs");

    // Track the path actually taken so the report reflects reality rather than a
    // hardcoded label, and pick the null the q-values are computed against.
    let mut classifier_used = "native_tda";
    let mut model_identity = "native-percolator-lite-v1".to_string();
    let mut qmode = QMode::Decoy;

    let mut scores = if n == 0 {
        classifier_used = "not_run_empty";
        model_identity = "none-empty-input".to_string();
        Vec::new()
    } else {
        match p.cfg.classifier {
            RescorerKind::Mokapot => match run_pin_sidecar(
                &p,
                "mokapot_worker.py",
                &feat_names,
                &cid,
                &label,
                &pform,
                &protein,
                &mz,
                &feats,
            ) {
                Ok(s) => {
                    info!("rescore: using Mokapot scores");
                    classifier_used = "mokapot";
                    let estimator =
                        std::env::var("MUMDIA_RESCORE_MODEL").unwrap_or_else(|_| "nn".to_string());
                    model_identity = format!("mokapot-{estimator}");
                    s
                }
                Err(e) => {
                    if p.cfg.strict {
                        anyhow::bail!(
                            "rescore: Mokapot sidecar failed ({e}) and rescore.strict=true"
                        );
                    }
                    warn!("rescore: Mokapot failed ({e}); falling back to native_tda");
                    native_scores(&p, &feats, &is_decoy, &base, &prelim)
                }
            },
            RescorerKind::NnTorch => match run_pin_sidecar(
                &p,
                "nn_rescore_worker.py",
                &feat_names,
                &cid,
                &label,
                &pform,
                &protein,
                &mz,
                &feats,
            ) {
                Ok(s) => {
                    info!("rescore: using PyTorch NN sidecar scores");
                    classifier_used = "nn_torch";
                    model_identity = "nn-torch-semisup-sidecar-v1".to_string();
                    s
                }
                Err(e) => {
                    if p.cfg.strict {
                        anyhow::bail!(
                            "rescore: NnTorch sidecar failed ({e}) and rescore.strict=true"
                        );
                    }
                    warn!("rescore: NnTorch failed ({e}); falling back to native_tda");
                    native_scores(&p, &feats, &is_decoy, &base, &prelim)
                }
            },
            RescorerKind::Percolator => {
                if p.cfg.strict {
                    anyhow::bail!("rescore: classifier=percolator but percolator.exe is not wired, and rescore.strict=true");
                }
                warn!("rescore: percolator.exe path not wired; using native_tda");
                native_scores(&p, &feats, &is_decoy, &base, &prelim)
            }
            RescorerKind::NativeTda => native_scores(&p, &feats, &is_decoy, &base, &prelim),
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
                    native_scores(&p, &feats, &is_decoy, &base, &prelim)
                } else if p.cfg.python.is_some() {
                    match run_entrapment_gbm(
                        &p,
                        &feat_names,
                        &cid,
                        &base,
                        &is_entrapment,
                        &is_decoy,
                        &feats,
                    ) {
                        Ok(s) => {
                            info!(
                                entrapment_psms = n_ent,
                                "rescore: using entrapment GBM sidecar scores"
                            );
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
                            native_scores(&p, &feats, &is_entrapment, &base, &prelim)
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
                    native_scores(&p, &feats, &is_entrapment, &base, &prelim)
                }
            }
        }
    };
    if let Some((row, score)) = scores
        .iter()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        anyhow::bail!("rescore produced non-finite score at flat row {row}: {score}");
    }

    // Top-K per-candidate collapse (#7): keep only the best-scoring peak per
    // (source, candidate_id), so the rescorer (not the up-front apex pick) selects
    // the peak and the terminal q-null is exactly one row per candidate. Promoting K
    // peaks therefore does not K-inflate the decoy null. Guarded: at
    // promote_top_peaks = 1 every candidate has a single peak, `best.len() == n`, and
    // the whole block is a no-op (byte-identical). Decoys collapse by the identical
    // rule, so target/decoy exchangeability is preserved. Tie-break: lower peak_rank
    // then lower row index (deterministic).
    if n > 0 {
        let mut best: HashMap<(u32, u32), usize> = HashMap::with_capacity(n);
        for i in 0..n {
            let key = (source[i], cid[i]);
            match best.get(&key) {
                Some(&j) => {
                    let better = scores[i] > scores[j]
                        || (scores[i] == scores[j] && (peak_rank[i], i) < (peak_rank[j], j));
                    if better {
                        best.insert(key, i);
                    }
                }
                None => {
                    best.insert(key, i);
                }
            }
        }
        if best.len() < n {
            let mut keep: Vec<usize> = best.into_values().collect();
            keep.sort_unstable();
            macro_rules! keep_rows {
                ($($v:ident),+ $(,)?) => {$(
                    { let tmp: Vec<_> = keep.iter().map(|&i| $v[i].clone()).collect(); $v = tmp; }
                )+};
            }
            keep_rows!(
                cid,
                label,
                base,
                pform,
                protein,
                charge,
                prelim,
                apex_rt,
                elution_lo,
                elution_hi,
                source,
                scores,
                peak_rank,
                is_decoy,
                is_entrapment,
                is_real_target
            );
            n = keep.len();
            info!(kept = n, "rescore: top-K per-candidate best-peak collapse");
        }
    }

    // PSM-level q-values against the selected null.
    let psm_q = match qmode {
        QMode::Decoy => {
            let sd: Vec<(f64, bool)> = scores
                .iter()
                .cloned()
                .zip(is_decoy.iter().cloned())
                .collect();
            target_decoy_q(&sd)
        }
        QMode::Entrapment => entrapment_q(
            &scores,
            &is_entrapment,
            &is_real_target,
            p.cfg.entrapment_ratio,
        ),
    };

    // Peptide-level q: reduce to best PSM per base peptide, q on that set, map
    // back. Protein-group q: same over the protein-accession-set string (the MVP
    // grouping; decoys carry a DECOY_ prefix). Full parsimony/razor is a later
    // option (PLAN.md Stage G). Group score = best member PSM score.
    let peptide_q = grouped_q(
        &base,
        &scores,
        &is_decoy,
        &is_entrapment,
        &is_real_target,
        qmode,
        p.cfg.entrapment_ratio,
    );
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
    let pg_q = grouped_q(
        &protein_id,
        &scores,
        &is_decoy,
        &is_entrapment,
        &is_real_target,
        qmode,
        p.cfg.entrapment_ratio,
    );
    // Multi-context q-values (PLAN.md Section 8 rescore; comment.md C1/C3). The
    // pooled per-PSM q is `experiment_psm_q`; `run_psm_q` re-runs TDA within each
    // source (run) so a per-run quant/report gets a real per-run FDR rather than the
    // pooled value; `precursor_q` groups on peptidoform+charge. `global_q` is kept
    // as a byte-identical alias of the pooled q for backward-compat.
    let global_q = psm_q.clone();
    let experiment_psm_q = psm_q.clone();
    // Per-run PSM q: TDA within each source separately, scattered back by row index.
    // Single-run (source all-zero) => equals `q_value`. Sorted (BTree) source
    // iteration keeps it deterministic; no floats are summed.
    let run_psm_q = {
        let mut by_src: std::collections::BTreeMap<u32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, &s) in source.iter().enumerate() {
            by_src.entry(s).or_default().push(i);
        }
        let mut rq = vec![1.0f64; n];
        for (_s, idxs) in by_src {
            let q = match qmode {
                QMode::Decoy => {
                    let sd: Vec<(f64, bool)> =
                        idxs.iter().map(|&i| (scores[i], is_decoy[i])).collect();
                    target_decoy_q(&sd)
                }
                QMode::Entrapment => {
                    let sc: Vec<f64> = idxs.iter().map(|&i| scores[i]).collect();
                    let en: Vec<bool> = idxs.iter().map(|&i| is_entrapment[i]).collect();
                    let re: Vec<bool> = idxs.iter().map(|&i| is_real_target[i]).collect();
                    entrapment_q(&sc, &en, &re, p.cfg.entrapment_ratio)
                }
            };
            for (k, &i) in idxs.iter().enumerate() {
                rq[i] = q[k];
            }
        }
        rq
    };
    // Precursor-level q: group on peptidoform+charge (interned to dense u32 like the
    // protein path) and run TDA over the best PSM per precursor.
    let precursor_id: Vec<u32> = {
        let mut interner: HashMap<(&str, i32), u32> = HashMap::new();
        let mut ids = Vec::with_capacity(pform.len());
        for (pf, &z) in pform.iter().zip(charge.iter()) {
            let next = interner.len() as u32;
            ids.push(*interner.entry((pf.as_str(), z)).or_insert(next));
        }
        ids
    };
    let precursor_q = grouped_q(
        &precursor_id,
        &scores,
        &is_decoy,
        &is_entrapment,
        &is_real_target,
        qmode,
        p.cfg.entrapment_ratio,
    );

    // Reported IDs: real targets in entrapment mode (spike-in excluded), else all
    // non-decoy targets.
    let is_reported: Vec<bool> = match qmode {
        QMode::Decoy => is_decoy.iter().map(|d| !d).collect(),
        QMode::Entrapment => is_real_target.clone(),
    };

    let n_psm_1 = (0..n)
        .filter(|&i| is_reported[i] && psm_q[i] <= 0.01)
        .count();
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
    let n_prec_1 = {
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            if is_reported[i] && precursor_q[i] <= 0.01 {
                seen.insert(precursor_id[i]);
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
            Col::F64("apex_rt".into(), apex_rt),
            Col::F64("elution_lo".into(), elution_lo),
            Col::F64("elution_hi".into(), elution_hi),
            Col::F64("score".into(), scores),
            Col::F64("q_value".into(), psm_q),
            Col::F64("peptide_q_value".into(), peptide_q),
            Col::Str("protein_group".into(), protein),
            Col::F64("pg_q_value".into(), pg_q),
            Col::F64("global_q_value".into(), global_q),
            Col::F64("prelim_score".into(), prelim),
            // Run identity for experiment-wide rescore (index into --competed);
            // all-zero for a single-run rescore. Lets quant map scores per file.
            Col::U32("source".into(), source),
            // Multi-context q columns (comment.md C1/C3). run_psm_q = per-run PSM
            // FDR; experiment_psm_q = pooled PSM FDR (== q_value/global_q_value);
            // precursor_q = per (peptidoform+charge) FDR.
            Col::F64("run_psm_q".into(), run_psm_q),
            Col::F64("experiment_psm_q".into(), experiment_psm_q),
            Col::F64("precursor_q".into(), precursor_q),
            // Which chromatographic peak the rescorer selected for this candidate
            // (#7). 0 = the up-front apex; > 0 = a promoted alternate peak won.
            Col::I32("selected_peak_rank".into(), peak_rank),
        ],
    )?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("psms".to_string(), json!(n));
    stats.insert("classifier".to_string(), json!(classifier_used));
    stats.insert("target_psms_at_1pct".to_string(), json!(n_psm_1));
    stats.insert("target_peptides_at_1pct".to_string(), json!(n_pep_1));
    stats.insert("target_protein_groups_at_1pct".to_string(), json!(n_pg_1));
    stats.insert("target_precursors_at_1pct".to_string(), json!(n_prec_1));
    if qmode == QMode::Entrapment {
        stats.insert(
            "entrapment_ratio".to_string(),
            json!(p.cfg.entrapment_ratio),
        );
        stats.insert("entrapment_peptides_at_1pct".to_string(), json!(n_entrap_1));
    }
    ArtifactReport {
        logical_name: artifact::PSMS_SCORED.0.to_string(),
        schema_name: artifact::PSMS_SCORED.0.to_string(),
        schema_version: artifact::PSMS_SCORED.1,
        stage: "rescore".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({
            "classifier": classifier_used,
            "classifier_requested": format!("{:?}", p.cfg.classifier),
            "strict": p.cfg.strict,
            "folds": p.cfg.folds,
            "num_iter": p.cfg.num_iter,
            "train_fdr": p.cfg.train_fdr,
            "feature_schema_id": expected_schema.schema_id,
            "competed_inputs": p.competed,
            "config_hash": p.config_hash,
        }),
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

/// Reject a concatenation whose feature companions differ in either identity or
/// ordered feature columns. Both checks are intentional: the ID catches a
/// declared contract mismatch, while the explicit column comparison protects
/// against a malformed or manually edited companion.
fn validate_feature_schema(
    expected: &FeatureSchema,
    actual: &FeatureSchema,
    path: &str,
) -> Result<()> {
    if expected.schema_id != actual.schema_id || expected.feature_columns != actual.feature_columns
    {
        anyhow::bail!(
            "rescore feature schema mismatch for '{path}': expected id '{}' columns {:?}, \
             found id '{}' columns {:?}",
            expected.schema_id,
            expected.feature_columns,
            actual.schema_id,
            actual.feature_columns
        );
    }
    Ok(())
}

/// Native semi-supervised rescorer scores.
fn native_scores(
    p: &RescoreParams,
    feats: &[Vec<f64>],
    is_decoy: &[bool],
    fold_key: &[u32],
    prelim: &[f64],
) -> Vec<f64> {
    percolator_lite(RescoreInput {
        features: feats,
        is_decoy,
        fold_key,
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
                    && exclude.is_none_or(|e| !protein[i].contains(e))
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
    // Keep the winning row index with each picked group. Exact target/null score
    // ties go to the active null (decoy or entrapment) so input row order cannot
    // make the accepted set anti-conservative.
    let mut best: HashMap<K, (f64, bool, bool, bool, usize)> = HashMap::new();
    for i in 0..n {
        let e = best.entry(keys[i].clone()).or_insert((
            f64::NEG_INFINITY,
            is_decoy[i],
            is_entrapment[i],
            is_real[i],
            i,
        ));
        let incoming_null = match qmode {
            QMode::Decoy => is_decoy[i],
            QMode::Entrapment => is_entrapment[i],
        };
        let current_null = match qmode {
            QMode::Decoy => e.1,
            QMode::Entrapment => e.2,
        };
        if scores[i] > e.0 || (scores[i] == e.0 && incoming_null && !current_null) {
            *e = (scores[i], is_decoy[i], is_entrapment[i], is_real[i], i);
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
    // Assign the group q ONLY to the picked winning row of each group. A
    // losing sibling (a lower-scoring charge/mod variant, which may itself be a
    // false target) must not inherit the winner's low q; it gets 1.0. The
    // report/counts dedup by key on the winner, so peptide/PG counts are
    // unchanged, but per-PSM peptide_q/pg_q no longer propagate to losers.
    let mut out = vec![1.0f64; n];
    for (key, (_, _, _, _, row)) in best {
        out[row] = qmap[&key];
    }
    out
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
        // Unique per-row id for score readback: candidate_id repeats across
        // competed runs, so mapping scores back by candidate_id collides (later
        // runs overwrite earlier). Map by this flat row index instead.
        Col::U32("row_id".into(), (0..cid.len()).map(|i| i as u32).collect()),
        Col::U32("candidate_id".into(), cid.to_vec()),
        Col::U32("base_peptide_id".into(), base.to_vec()),
        Col::I32(
            "is_entrapment".into(),
            is_entrapment.iter().map(|&b| b as i32).collect(),
        ),
        Col::I32(
            "is_decoy".into(),
            is_decoy.iter().map(|&b| b as i32).collect(),
        ),
    ];
    for (fi, name) in feat_names.iter().enumerate() {
        cols.push(Col::F64(
            name.clone(),
            (0..cid.len()).map(|i| feats[i][fi]).collect(),
        ));
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
    let orid = t.u32("row_id")?;
    let osc = t.f64("score")?;
    align_sidecar_scores(&orid, &osc, cid.len(), "entrapment_worker")
}

/// Run a PIN-contract Python rescorer sidecar (`mokapot_worker.py` or
/// `nn_rescore_worker.py`) over a PIN written from the competed set; return scores
/// aligned to the input candidate order (PLAN.md Section 3.2 file contract). Both
/// sidecars share this exact contract: PIN in, `candidate_id`+`score` parquet out.
#[allow(clippy::too_many_arguments)]
fn run_pin_sidecar(
    p: &RescoreParams,
    script_name: &str,
    feat_names: &[String],
    cid: &[u32],
    label: &[String],
    pform: &[String],
    protein: &[String],
    mz: &[f64],
    feats: &[Vec<f64>],
) -> Result<Vec<f64>> {
    use std::fmt::Write as _;
    let python = p.cfg.python.as_deref().ok_or_else(|| {
        anyhow::anyhow!("classifier sidecar {script_name} requires rescore.python")
    })?;
    std::fs::create_dir_all(p.work_dir).ok();
    let pin = format!("{}/rescore.pin", p.work_dir);
    let outp = format!("{}/rescore_sidecar_out.parquet", p.work_dir);

    let mut s = String::new();
    s.push_str("SpecId\tLabel\tScanNr\tExpMass\tCalcMass\t");
    s.push_str(&feat_names.join("\t"));
    s.push_str("\tPeptide\tProteins\n");
    // Key the PIN on the unique row index i (SpecId=psm_i, ScanNr=i), NOT
    // candidate_id: candidate_id is the library index and repeats across runs, so
    // an experiment-wide (multi-file) PIN would collide on ScanNr and mokapot's
    // per-spectrum competition would collapse the runs. The row index is unique
    // across the whole concatenation. Single-run behaviour is unchanged (the
    // mapping is bijective and mokapot does not use SpecId/ScanNr as features).
    for i in 0..cid.len() {
        let lab = if label[i] == "decoy" { -1 } else { 1 };
        write!(s, "psm_{}\t{}\t{}\t{:.5}\t{:.5}\t", i, lab, i, mz[i], mz[i]).ok();
        #[allow(clippy::needless_range_loop)] // parallel index into feats[i] bounded by feat_names
        for fi in 0..feat_names.len() {
            write!(s, "{:.6}\t", feats[i][fi]).ok();
        }
        writeln!(s, "-.{}.-\t{}", pform[i], protein[i]).ok();
    }
    std::fs::write(&pin, s)?;

    let script = crate::sidecar::resolve_script(p.script_dir, script_name);
    let status = std::process::Command::new(python)
        .arg(&script)
        .arg(&pin)
        .arg(&outp)
        .env("PYTHONUTF8", "1")
        // Pass the configured NN hyperparameters so the worker uses them instead
        // of its own defaults, and so the folds/num_iter/train_fdr recorded in the
        // report reflect the values actually used (comment.md C4). Ignored by
        // mokapot_worker.py, which shares this PIN contract.
        .env("MUMDIA_NN_FOLDS", p.cfg.folds.to_string())
        .env("MUMDIA_NN_ITERS", p.cfg.num_iter.to_string())
        .env("MUMDIA_NN_TRAIN_FDR", p.cfg.train_fdr.to_string())
        .status()?;
    if !status.success() {
        anyhow::bail!("{script_name} exited with {status}");
    }

    // The worker echoes the PIN's SpecId tail as `candidate_id`, which here is the
    // flat row index. Exact, unique, finite coverage is part of the classifier
    // contract: silently assigning a worst score to missing rows changes the
    // trained population and can invalidate sensitivity/FDR comparisons.
    let t = Table::read(&outp)?;
    let orow = t.u32("candidate_id")?;
    let osc = t.f64("score")?;
    align_sidecar_scores(&orow, &osc, cid.len(), script_name)
}

/// Validate and align a sidecar's `(flat_row_id, score)` response. Every input
/// row must occur exactly once, there may be no extras, and all scores must be
/// finite. This is shared by PIN and entrapment sidecars.
fn align_sidecar_scores(
    row_ids: &[u32],
    scores: &[f64],
    expected_rows: usize,
    sidecar: &str,
) -> Result<Vec<f64>> {
    if row_ids.len() != scores.len() || row_ids.len() != expected_rows {
        anyhow::bail!(
            "{sidecar} output coverage mismatch: expected {expected_rows} rows, \
             got {} ids and {} scores",
            row_ids.len(),
            scores.len()
        );
    }
    let mut aligned = vec![0.0f64; expected_rows];
    let mut seen = vec![false; expected_rows];
    for (&row_id, &score) in row_ids.iter().zip(scores) {
        let row = row_id as usize;
        if row >= expected_rows {
            anyhow::bail!(
                "{sidecar} returned out-of-range row id {row_id}; expected 0..{expected_rows}"
            );
        }
        if seen[row] {
            anyhow::bail!("{sidecar} returned duplicate row id {row_id}");
        }
        if !score.is_finite() {
            anyhow::bail!("{sidecar} returned non-finite score for row id {row_id}: {score}");
        }
        seen[row] = true;
        aligned[row] = score;
    }
    if let Some(missing) = seen.iter().position(|present| !present) {
        anyhow::bail!("{sidecar} did not return a score for row id {missing}");
    }
    Ok(aligned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn competed_feature_schema_must_match_id_and_ordered_columns() {
        let expected = FeatureSchema {
            feature_columns: vec!["a".into(), "b".into()],
            schema_id: "schema-a".into(),
        };
        let same = FeatureSchema {
            feature_columns: vec!["a".into(), "b".into()],
            schema_id: "schema-a".into(),
        };
        assert!(validate_feature_schema(&expected, &same, "same.parquet").is_ok());

        let reordered = FeatureSchema {
            feature_columns: vec!["b".into(), "a".into()],
            schema_id: "schema-a".into(),
        };
        assert!(validate_feature_schema(&expected, &reordered, "reordered.parquet").is_err());

        let different_id = FeatureSchema {
            feature_columns: vec!["a".into(), "b".into()],
            schema_id: "schema-b".into(),
        };
        assert!(validate_feature_schema(&expected, &different_id, "id.parquet").is_err());
    }

    #[test]
    fn sidecar_scores_require_exact_unique_finite_coverage() {
        assert_eq!(
            align_sidecar_scores(&[1, 0], &[2.0, 1.0], 2, "test").unwrap(),
            vec![1.0, 2.0]
        );
        assert!(align_sidecar_scores(&[0], &[1.0], 2, "test").is_err());
        assert!(align_sidecar_scores(&[0, 0], &[1.0, 2.0], 2, "test").is_err());
        assert!(align_sidecar_scores(&[0, 2], &[1.0, 2.0], 2, "test").is_err());
        assert!(align_sidecar_scores(&[0, 1], &[1.0, f64::NAN], 2, "test").is_err());
    }

    #[test]
    fn picked_peptide_exact_tie_is_won_by_decoy() {
        // The target is deliberately first: row order must not win an exact
        // paired target/decoy tie.
        let mut keys = vec![0u32, 0u32];
        let mut scores = vec![5.0, 5.0];
        let mut decoys = vec![false, true];
        let mut entrapments = vec![false, false];
        let mut real = vec![true, false];
        for key in 1..=100 {
            keys.push(key);
            scores.push(4.0 - key as f64 * 0.01);
            decoys.push(false);
            entrapments.push(false);
            real.push(true);
        }

        let q = grouped_q(
            &keys,
            &scores,
            &decoys,
            &entrapments,
            &real,
            QMode::Decoy,
            1.0,
        );
        assert_eq!(q[0], 1.0, "tied target must be the losing sibling");
        assert!(q[1] < 0.05, "tied decoy should own the picked-group q");
    }
}
