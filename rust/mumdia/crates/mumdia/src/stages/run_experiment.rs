//! Experiment-wide orchestrator `mumdia run-experiment`: run the per-file search
//! chain over N runs, then a single experiment-wide rescore, optional rescuable
//! match-between-runs transfer, a by-run split, per-run quantification, and
//! cross-run LFQ. This reifies the multi-run flow that was previously a manual
//! script sequence, keeping the rescuable-tier MBR first-class.
//!
//! The library is built or imported once and shared. Each run gets its own
//! optional DeepLC fine-tune (RT is per-run), its own extraction (chromatograms
//! kept for quantification), and its own competed table. The competed tables of
//! all runs feed one rescore so the classifier and FDR are experiment-wide.

use std::time::Instant;

use anyhow::Result;
use arrow::array::{Array, BooleanArray, UInt32Array};
use arrow::compute::filter_record_batch;
use mumdia_core::config::{Config, QuantQColumn};
use serde_json::json;
use tracing::info;

use crate::stages::*;

pub struct RunExperimentParams<'a> {
    pub config: &'a Config,
    pub fasta: Option<&'a str>,
    pub mzmls: &'a [String],
    /// Optional per-run labels (default `r0..rN-1`); also the per-run subdir names.
    pub run_names: Option<&'a [String]>,
    pub out_dir: &'a str,
    pub lib_precursors: Option<&'a str>,
    pub lib_fragments: Option<&'a str>,
    pub max_spectra: usize,
    pub top_peaks_ms2: usize,
}

fn preflight(p: &RunExperimentParams) -> Result<()> {
    use mumdia_core::config::{MbrStrategy, RescorerKind};
    if p.mzmls.len() < 2 {
        anyhow::bail!(
            "run-experiment needs >= 2 --mzml files (got {})",
            p.mzmls.len()
        );
    }
    for m in p.mzmls {
        if !std::path::Path::new(m).exists() {
            anyhow::bail!("--mzml not found: {m}");
        }
    }
    match (p.lib_precursors, p.lib_fragments) {
        (Some(lp), Some(lf)) => {
            for path in [lp, lf] {
                if !std::path::Path::new(path).exists() {
                    anyhow::bail!("library not found: {path}");
                }
            }
        }
        (None, None) => match p.fasta {
            Some(f) if std::path::Path::new(f).exists() => {}
            Some(f) => anyhow::bail!("--fasta not found: {f}"),
            None => {
                anyhow::bail!("provide either --fasta or both --lib-precursors and --lib-fragments")
            }
        },
        _ => anyhow::bail!("library-input mode requires both --lib-precursors and --lib-fragments"),
    }
    let cfg = p.config;
    if cfg.rt_im_train.finetune_deeplc && cfg.predict_frag.deeplc_python.is_none() {
        anyhow::bail!("rt_im_train.finetune_deeplc requires predict_frag.deeplc_python");
    }
    if matches!(
        cfg.rescore.classifier,
        RescorerKind::Mokapot | RescorerKind::NnTorch
    ) && cfg.rescore.python.is_none()
    {
        anyhow::bail!(
            "rescore.classifier={:?} requires rescore.python",
            cfg.rescore.classifier
        );
    }
    if cfg.mbr.strategy != MbrStrategy::None && cfg.mbr.python.is_none() {
        anyhow::bail!("mbr.strategy != none requires mbr.python");
    }
    Ok(())
}

/// The per-run search chain: convert -> seed -> optional fine-tune -> rt-cal ->
/// extract -> features -> compete. Returns (competed_path, chrom_path). The
/// chromatograms are kept for the later per-run quantification.
#[allow(clippy::too_many_arguments)]
fn process_run(
    cfg: &Config,
    ch: &str,
    lib_p_base: &str,
    lib_f: &str,
    mzml: &str,
    out: &str,
    top_peaks_ms2: usize,
    max_spectra: usize,
) -> Result<(String, String)> {
    let d = |name: &str| format!("{out}/{name}");
    std::fs::create_dir_all(out).ok();
    let co = convert::run(convert::ConvertParams {
        mzml,
        out_dir: &d("spectra"),
        max_spectra,
        top_peaks_ms2,
        top_peaks_ms1: 0,
        config_hash: ch,
    })?;
    let seed = d("seed_psms.parquet");
    search_seed::run(search_seed::SearchSeedParams {
        ms2: &co.ms2,
        library_precursors: lib_p_base,
        library_fragments: lib_f,
        out: &seed,
        cfg: &cfg.search_seed,
        bucket_size: cfg.extract.bucket_size,
        config_hash: ch,
    })?;
    // Per-run DeepLC fine-tune (RT is per-run); reuse the base library otherwise.
    let lib_p = if cfg.rt_im_train.finetune_deeplc {
        let python = cfg
            .predict_frag
            .deeplc_python
            .as_deref()
            .expect("preflight");
        let script = crate::sidecar::resolve_script(
            &cfg.predict_frag.sidecar_script_dir,
            "deeplc_finetune.py",
        );
        let lib_p_ft = d("fragment_library_precursors_ft.parquet");
        crate::sidecar::run_deeplc_finetune(
            python,
            &script,
            lib_p_base,
            &seed,
            &lib_p_ft,
            cfg.rt_im_train.finetune_epochs,
            cfg.rt_im_train.finetune_patience,
            cfg.rt_im_train.q_train,
            cfg.rt_im_train.finetune_batch,
        )?;
        lib_p_ft
    } else {
        lib_p_base.to_string()
    };
    let windows = d("run_windows.parquet");
    rt_im_train::run(rt_im_train::RtImTrainParams {
        seed_psms: &seed,
        library_precursors: &lib_p,
        out_windows: &windows,
        out_cal: &d("cal.json"),
        cfg: &cfg.rt_im_train,
        config_hash: ch,
    })?;
    let psms = d("psms_extracted.parquet");
    let chrom = d("chromatograms.parquet");
    extract::run(extract::ExtractParams {
        ms2: &co.ms2,
        library_precursors: &lib_p,
        library_fragments: lib_f,
        run_windows: &windows,
        ms1: Some(&co.ms1),
        mass_cal: Some(&format!("{seed}.masscal.json")),
        out_psms: &psms,
        out_chrom: &chrom,
        restrict_candidates: None,
        cfg: &cfg.extract,
        config_hash: ch,
    })?;
    let feats = d("features.parquet");
    features::run(features::FeaturesParams {
        psms: &psms,
        chromatograms: &chrom,
        seed: Some(&seed),
        out: &feats,
        out_pin: &d("run.pin"),
        cfg: &cfg.features,
        config_hash: ch,
    })?;
    let competed = d("psms_competed.parquet");
    compete::run(compete::CompeteParams {
        features: &feats,
        out: &competed,
        cfg: &cfg.compete,
        config_hash: ch,
    })?;
    Ok((competed, chrom))
}

/// Split an experiment-wide scored table into per-run tables by the `source`
/// column (0..n-1), preserving the schema exactly (arrow row filter). Quant then
/// runs per run with `q_filter = psm_q`, keeping each run's own confident PSMs.
fn split_by_source(scored: &str, out_paths: &[String]) -> Result<()> {
    let t = mumdia_io::table::Table::read(scored)?;
    let src_idx = t
        .schema
        .index_of("source")
        .map_err(|_| anyhow::anyhow!("scored table has no `source` column for split"))?;
    for (i, out) in out_paths.iter().enumerate() {
        let mut filtered = Vec::with_capacity(t.batches.len());
        for b in &t.batches {
            let src = b
                .column(src_idx)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| anyhow::anyhow!("`source` column is not u32"))?;
            let mask: BooleanArray = (0..src.len()).map(|k| src.value(k) == i as u32).collect();
            filtered.push(filter_record_batch(b, &mask)?);
        }
        mumdia_io::table::write_batches(out, t.schema.clone(), &filtered)?;
    }
    Ok(())
}

pub fn run(p: RunExperimentParams) -> Result<()> {
    let t0 = Instant::now();
    let cfg = p.config;
    preflight(&p)?;
    let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
    std::fs::create_dir_all(p.out_dir).ok();
    let d = |name: &str| format!("{}/{}", p.out_dir, name);
    let n_runs = p.mzmls.len();
    let names: Vec<String> = match p.run_names {
        Some(ns) if ns.len() == n_runs => ns.to_vec(),
        _ => (0..n_runs).map(|i| format!("r{i}")).collect(),
    };

    // --- shared library (imported or digested once) ---
    let (lib_p_base, lib_f) = match (p.lib_precursors, p.lib_fragments) {
        (Some(lp), Some(lf)) => {
            info!(lib_precursors = lp, "run-experiment: library-input mode");
            (lp.to_string(), lf.to_string())
        }
        _ => {
            let fasta = p.fasta.expect("preflight guarantees --fasta in build mode");
            let dig = d("peptides.parquet");
            digest::run(digest::DigestParams {
                fasta,
                out: &dig,
                cfg: &cfg.digest,
                rng_seed: cfg.rng_seed,
                config_hash: &ch,
            })?;
            let pf = d("peptidoforms.parquet");
            peptidoforms::run(peptidoforms::PeptidoformsParams {
                peptides: &dig,
                out: &pf,
                cfg: &cfg.peptidoforms,
                config_hash: &ch,
            })?;
            let lib_p = d("fragment_library_precursors.parquet");
            let lib_f = d("fragment_library_fragments.parquet");
            predict_frag::run(predict_frag::PredictFragParams {
                peptidoforms: &pf,
                out_precursors: &lib_p,
                out_fragments: &lib_f,
                work_dir: &d("sidecar_work"),
                cfg: &cfg.predict_frag,
                config_hash: &ch,
            })?;
            (lib_p, lib_f)
        }
    };

    // --- per-run search chains ---
    let mut competed: Vec<String> = Vec::with_capacity(n_runs);
    let mut chroms: Vec<String> = Vec::with_capacity(n_runs);
    for (i, mzml) in p.mzmls.iter().enumerate() {
        info!(run = %names[i], i = i + 1, n = n_runs, "run-experiment: per-run chain");
        let (comp, chrom) = process_run(
            cfg,
            &ch,
            &lib_p_base,
            &lib_f,
            mzml,
            &d(&names[i]),
            p.top_peaks_ms2,
            p.max_spectra,
        )?;
        competed.push(comp);
        chroms.push(chrom);
    }

    // --- one experiment-wide rescore over all competed tables ---
    let scored_combined = d("scored_combined.parquet");
    rescore::run(rescore::RescoreParams {
        competed: &competed,
        out: &scored_combined,
        work_dir: &d("sidecar_work"),
        script_dir: &cfg.predict_frag.sidecar_script_dir,
        cfg: &cfg.rescore,
        config_hash: &ch,
    })?;

    // --- optional rescuable-tier MBR transfer ---
    let scored_for_quant = if cfg.mbr.strategy != mumdia_core::config::MbrStrategy::None {
        let python = cfg.mbr.python.as_deref().expect("preflight");
        let script =
            crate::sidecar::resolve_script(&cfg.predict_frag.sidecar_script_dir, "mbr_worker.py");
        let transferred = d("mbr_transferred.parquet");
        let scored_mbr = d("scored_mbr.parquet");
        // The competed tables carry candidate_id + apex_rt in `source` order.
        crate::sidecar::run_mbr(
            python,
            &script,
            &scored_combined,
            &competed,
            &transferred,
            Some(&scored_mbr),
            &[],
            cfg.mbr.q_anchor,
            cfg.mbr.min_anchor_runs,
            cfg.mbr.q_transfer,
            cfg.mbr.consensus_corr_min,
            cfg.rng_seed,
        )?;
        // The worker writes the augmented scored table only when it accepts at
        // least one transfer; with none, fall back to the un-augmented combined
        // table so quantification still runs.
        if std::path::Path::new(&scored_mbr).exists() {
            info!("run-experiment: MBR transfers applied");
            scored_mbr
        } else {
            info!("run-experiment: MBR accepted no transfers; using combined scored");
            scored_combined.clone()
        }
    } else {
        scored_combined.clone()
    };

    // --- split by source, per-run quant (psm_q filter), cross-run LFQ ---
    let split_paths: Vec<String> = (0..n_runs)
        .map(|i| d(&format!("{}/scored.parquet", names[i])))
        .collect();
    split_by_source(&scored_for_quant, &split_paths)?;

    let mut qcfg = cfg.quant.clone();
    qcfg.q_filter = QuantQColumn::PsmQ; // per-run confident PSMs from the pooled q_value
    let mut peptide_quants: Vec<String> = Vec::with_capacity(n_runs);
    for i in 0..n_runs {
        let pq = d(&format!("{}/peptide_quant.parquet", names[i]));
        quant::run(quant::QuantParams {
            psms_scored: &split_paths[i],
            chromatograms: &chroms[i],
            out_peptide: &pq,
            out_protein: &d(&format!("{}/protein_group_quant.parquet", names[i])),
            out_fragment: None,
            out_peak_bounds: None,
            cfg: &qcfg,
            config_hash: &ch,
        })?;
        peptide_quants.push(pq);
    }
    let lfq = d("lfq_maxlfq.parquet");
    quant::run_lfq_combine(
        &peptide_quants,
        false,
        mumdia_core::config::NormalizeMethod::MedianRatio,
        &lfq,
    )?;

    // --- experiment manifest ---
    let manifest = json!({
        "config_hash": ch,
        "n_runs": n_runs,
        "runs": names,
        "scored_combined": scored_combined,
        "scored_for_quant": scored_for_quant,
        "mbr": format!("{:?}", cfg.mbr.strategy),
        "lfq": lfq,
        "peptide_quants": peptide_quants,
    });
    mumdia_io::json::write_json(&d("experiment_manifest.json"), &manifest)?;
    info!(
        elapsed_ms = t0.elapsed().as_millis(),
        n_runs, "run-experiment: complete"
    );
    Ok(())
}
