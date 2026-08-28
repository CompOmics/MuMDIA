//! Experiment-wide orchestrator `mumdia run-experiment`: run the per-file search
//! chain over N runs, then a single experiment-wide rescore, optional rescuable
//! match-between-runs transfer, a by-run split, per-run quantification, and
//! cross-run LFQ. This reifies the multi-run flow that was previously a manual
//! script sequence, keeping the rescuable-tier MBR first-class.
//!
//! The library is built or imported once and shared. Each run gets its own retention-time
//! calibration, its own extraction (chromatograms kept for quantification), and its own
//! competed table. The competed tables of all runs feed one rescore so the classifier and
//! FDR are experiment-wide.
//!
//! The DeepLC fine-tune defaults to running ONCE, on the first run, with every other run
//! reusing that library and fitting its own calibration against it
//! (`experiment.finetune_scope`). It is the most expensive step of a large experiment, and
//! per-run weight adaptation buys little once each run is calibrated separately.

use std::time::Instant;

use anyhow::{Context, Result};
use arrow::array::{Array, BooleanArray, UInt32Array};
use arrow::compute::filter_record_batch;
use mumdia_core::config::{Config, FinetuneScope, QuantQColumn};
use mumdia_core::manifest::Manifest;
use rayon::prelude::*;
use serde_json::json;
use tracing::{info, warn};

use crate::stages::*;

pub struct RunExperimentParams<'a> {
    pub config: &'a Config,
    /// Path the config was loaded from; see `RunParams::config_path`.
    pub config_path: Option<&'a str>,
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
    // Check the interpreters EXIST, not merely that the fields are set. The rescore runs
    // after every per-run chain, so on an 83-file batch a mistyped `rescore.python` used to
    // discard days of compute at the final stage. A wrong path is also the most likely error
    // when moving a config between machines.
    for (field, path) in [
        (
            "predict_frag.deeplc_python",
            cfg.predict_frag.deeplc_python.as_deref(),
        ),
        ("rescore.python", cfg.rescore.python.as_deref()),
        ("mbr.python", cfg.mbr.python.as_deref()),
    ] {
        if let Some(exe) = path {
            if !std::path::Path::new(exe).exists() {
                anyhow::bail!(
                    concat!(
                        "{} points at an interpreter that does not exist: {}. A config ",
                        "written for another machine is the usual cause; run ",
                        "`mumdia doctor --config <cfg>` to probe every configured sidecar ",
                        "environment before starting a long batch."
                    ),
                    field,
                    exe
                );
            }
        }
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
/// extract -> features -> compete. The chromatograms are kept for the later per-run
/// quantification.
/// `shared_ft` is a precursor library that has ALREADY been
/// DeepLC-fine-tuned (by an earlier run of this same experiment); when present this run
/// uses it as-is instead of fine-tuning again, and still fits its own retention-time
/// calibration on top. Returns `(competed, chromatograms, fine_tuned_library_if_produced)`.
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
    shared_ft: Option<&str>,
) -> Result<(String, String, Option<String>)> {
    let d = |name: &str| format!("{out}/{name}");
    std::fs::create_dir_all(out).ok();
    // Fold the conversion caps into the convert artifacts' provenance key, exactly as
    // both the standalone `convert` subcommand and single-run `run` do. The caps change
    // the spectra output but are not part of the config, so the bare config hash made
    // two experiments with different caps record identical convert provenance.
    let convert_hash = mumdia_io::hash::blake3_str(&format!(
        "{}\u{1f}max_spectra={}\u{1f}top_peaks_ms2={}\u{1f}top_peaks_ms1={}",
        cfg.canonical_json(),
        max_spectra,
        top_peaks_ms2,
        0
    ));
    let co = convert::run(convert::ConvertParams {
        mzml,
        out_dir: &d("spectra"),
        max_spectra,
        top_peaks_ms2,
        top_peaks_ms1: 0,
        config_hash: &convert_hash,
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
    // DeepLC fine-tune. Under `FinetuneScope::FirstRunOnly` the caller hands every run
    // after the first the library the first run produced, so the fine-tune -- the most
    // expensive step in the whole experiment -- is paid once. Run-to-run chromatographic
    // drift is then absorbed by `rt_im_train`'s per-run calibration below, which is fitted
    // separately for every run regardless.
    let mut produced_ft: Option<String> = None;
    let lib_p = if let Some(ft) = shared_ft {
        ft.to_string()
    } else if cfg.rt_im_train.finetune_deeplc {
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
            // Keep the fine-tune exclusion aligned with rt-im-train's holdout
            // split (see run.rs); 0.0 (default) changes nothing.
            cfg.rt_im_train.window_holdout_frac,
            cfg.rng_seed,
        )?;
        produced_ft = Some(lib_p_ft.clone());
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
    Ok((competed, chrom, produced_ft))
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
    // Same contract as the single-run orchestrator, and it matters more here: an
    // 83-file batch must not fail on a missing interpreter after the first run has
    // already been searched.
    let mut resolved = p.config.clone();
    resolved.predict_frag.sidecar_script_dir =
        crate::python::resolve_script_dir(&resolved.predict_frag.sidecar_script_dir, p.config_path);
    crate::python::resolve(&mut resolved)?;
    let p = RunExperimentParams {
        config: &resolved,
        ..p
    };
    let cfg = p.config;
    preflight(&p)?;
    let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
    std::fs::create_dir_all(p.out_dir).ok();
    let d = |name: &str| format!("{}/{}", p.out_dir, name);
    let n_runs = p.mzmls.len();
    // Reject a bad --run-names rather than silently substituting r0..rN-1.
    //
    // The old `_ =>` arm swallowed any count mismatch with no warning, and accepted
    // duplicates outright. Duplicates are the dangerous half: two runs then share
    // `d(&names[i])`, so above one parallel run two `process_run` calls concurrently
    // write the same spectra_ms2 / seed_psms / psms_extracted / chromatograms into one
    // directory, and the split and quant paths collide too. The output is an interleaving
    // of two runs with no error anywhere. Reachable by naming runs after their basenames
    // when an experiment spans two directories holding same-named files.
    let names: Vec<String> = match p.run_names {
        Some(ns) => {
            if ns.len() != n_runs {
                anyhow::bail!(
                    "--run-names has {} entries but there are {n_runs} runs; pass one name \
                     per --mzml, in the same order, or omit it to get r0..r{}",
                    ns.len(),
                    n_runs - 1
                );
            }
            let mut sorted = ns.to_vec();
            sorted.sort();
            sorted.dedup();
            if sorted.len() != ns.len() {
                anyhow::bail!(
                    "--run-names must be unique: each name is a per-run output \
                     subdirectory, so a repeat makes two runs write the same artifacts \
                     into one directory and interleave their results with no error"
                );
            }
            if let Some(bad) = ns.iter().find(|n| {
                n.is_empty() || n.contains('/') || n.contains('\\') || *n == "." || *n == ".."
            }) {
                anyhow::bail!(
                    "--run-names entry {bad:?} is not usable as a directory name; each \
                     becomes a subdirectory of --out-dir"
                );
            }
            ns.to_vec()
        }
        None => (0..n_runs).map(|i| format!("r{i}")).collect(),
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
    // Runs are independent: `process_run` derives every path from its own per-run output
    // directory, so nothing is shared between them. Concurrency is OPT-IN and bounded
    // because each run's extract can hold tens of GB -- running all 83 files of an
    // experiment at once would exhaust memory long before it saturated the CPU.
    // `parallel_runs = 1` (the default) is the previous sequential behaviour exactly.
    //
    // Ordering: `competed`/`chroms` must stay in run index order, since the downstream
    // combined rescore keys rows by `source` index. Chunks are processed in order and
    // rayon's indexed `collect` preserves within-chunk order, so the result is identical
    // to the sequential build regardless of completion order.
    let par = cfg.experiment.parallel_runs.max(1);
    let mut competed: Vec<String> = Vec::with_capacity(n_runs);
    let mut chroms: Vec<String> = Vec::with_capacity(n_runs);

    // Under `FinetuneScope::FirstRunOnly` (the default) the first run is processed alone so
    // its DeepLC fine-tune can be handed to all the others. That fine-tune is the most
    // expensive step in a large experiment -- tens of minutes per run -- so paying it N
    // times instead of once is the difference between hours and days on an 80-run batch.
    // Every run still fits its OWN retention-time calibration (LOESS by default) against
    // that shared library, and that per-run fit is what absorbs chromatographic drift.
    let share_ft = cfg.rt_im_train.finetune_deeplc
        && matches!(cfg.experiment.finetune_scope, FinetuneScope::FirstRunOnly)
        && n_runs > 1;
    let mut shared_ft: Option<String> = None;
    let mut first: usize = 0;
    if share_ft {
        info!(
            run = %names[0],
            n = n_runs,
            "run-experiment: fine-tuning DeepLC on the first run only; the remaining runs              reuse that library and fit their own RT calibration on it              (experiment.finetune_scope = per_run to fine-tune every run instead)"
        );
        let (comp, chrom, ft) = process_run(
            cfg,
            &ch,
            &lib_p_base,
            &lib_f,
            &p.mzmls[0],
            &d(&names[0]),
            p.top_peaks_ms2,
            p.max_spectra,
            None,
        )?;
        competed.push(comp);
        chroms.push(chrom);
        match ft {
            Some(path) => {
                info!(library = %path, "run-experiment: reusing this fine-tuned library for the remaining runs");
                shared_ft = Some(path);
            }
            // Defensive: `share_ft` implies the first run fine-tunes, so this should not
            // happen. Falling through with `None` means the rest fine-tune themselves --
            // slower, but never silently wrong.
            None => warn!(
                "run-experiment: the first run produced no fine-tuned library; the                  remaining runs will each fine-tune"
            ),
        }
        first = 1;
    }

    let rest: Vec<usize> = (first..n_runs).collect();
    if par == 1 {
        for &i in &rest {
            info!(run = %names[i], i = i + 1, n = n_runs, "run-experiment: per-run chain");
            let (comp, chrom, _) = process_run(
                cfg,
                &ch,
                &lib_p_base,
                &lib_f,
                &p.mzmls[i],
                &d(&names[i]),
                p.top_peaks_ms2,
                p.max_spectra,
                shared_ft.as_deref(),
            )?;
            competed.push(comp);
            chroms.push(chrom);
        }
    } else {
        info!(
            parallel_runs = par,
            n = n_runs,
            "run-experiment: per-run chains in parallel (each run can hold tens of GB;              lower experiment.parallel_runs if memory is tight)"
        );
        for chunk in rest.chunks(par) {
            let done: Vec<(String, String, Option<String>)> = chunk
                .par_iter()
                .map(|&i| {
                    info!(run = %names[i], i = i + 1, n = n_runs, "run-experiment: per-run chain");
                    process_run(
                        cfg,
                        &ch,
                        &lib_p_base,
                        &lib_f,
                        &p.mzmls[i],
                        &d(&names[i]),
                        p.top_peaks_ms2,
                        p.max_spectra,
                        shared_ft.as_deref(),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            for (comp, chrom, _) in done {
                competed.push(comp);
                chroms.push(chrom);
            }
        }
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
        // Remove any stale copy BEFORE the worker runs, so that the `exists()` check
        // afterwards is a statement about THIS run. The worker writes this file only when
        // it accepts at least one transfer, so without the removal a rerun into the same
        // --out-dir that accepted transfers last time and none this time would silently
        // feed the previous experiment's scored table to the split and to every per-run
        // quant. It joins cleanly, because candidate_id and source are stable across
        // reruns of the same library, so the failure is plausible numbers from the wrong
        // data, logged as "MBR transfers applied".
        if std::path::Path::new(&scored_mbr).exists() {
            std::fs::remove_file(&scored_mbr).with_context(|| {
                format!("removing a previous run's {scored_mbr} before running MBR")
            })?;
        }
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
        //
        // `exists()` is only a signal because the stale file was removed before the
        // worker ran (see above). Without that, it could not tell "this run wrote it"
        // from "a previous run left it": rerunning the same --out-dir with different
        // mzMLs or a tighter mbr.q_transfer, and finding no transfers this time, made
        // the split and every per-run quant consume the OLD scored table. It joins
        // successfully, because candidate_id and source are stable across reruns of the
        // same library, so the result was plausible quantities from the wrong data while
        // the log said "MBR transfers applied".
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
    // Per-run quant gates on the pooled `q_value`, not on whatever `quant.q_filter` says. The
    // grouped q columns (peptide_q_value / precursor_q / pg_q_value) are assigned only to the
    // single experiment-wide winning row of each group (see `grouped_q` in rescore.rs), so a
    // per-run table can only gate on a per-PSM column. `run_psm_q` would be the natural choice,
    // but the pooled column is what the cross-run LFQ step downstream assumes.
    //
    // Warn rather than override in silence: a user who explicitly configured a different
    // `q_filter` otherwise gets quantities gated on a column they did not select, and there is
    // no record of the substitution in any artifact. Changing `q_filter` also does not select a
    // source, which is a separate documented trap.
    if !matches!(qcfg.q_filter, QuantQColumn::PsmQ) {
        warn!(
            configured = ?qcfg.q_filter,
            used = ?QuantQColumn::PsmQ,
            "run-experiment: per-run quant ignores the configured quant.q_filter and gates on the \
             pooled q_value; grouped q columns exist only on each group's experiment-wide winner"
        );
    }
    qcfg.q_filter = QuantQColumn::PsmQ;
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
    // Provenance parity with the single-run manifest. The experiment manifest is
    // still thinner: it carries no per-artifact records, because the per-run
    // chains do not thread a shared manifest. What it must not lack is the
    // identity of the code and the inputs, which is what makes a result
    // attributable at all.
    let mut prov = Manifest::new(cfg.canonical_json(), ch.clone());
    for (i, m) in p.mzmls.iter().enumerate() {
        if let (Ok(bytes), Ok(hash)) = (
            std::fs::metadata(m).map(|x| x.len()),
            mumdia_io::hash::blake3_file(m),
        ) {
            prov.record_input(&format!("mzml[{i}]"), m, bytes, hash);
        }
    }
    for (role, path) in [
        ("fasta", p.fasta),
        ("lib_precursors", p.lib_precursors),
        ("lib_fragments", p.lib_fragments),
    ] {
        let Some(path) = path else { continue };
        if let (Ok(bytes), Ok(hash)) = (
            std::fs::metadata(path).map(|x| x.len()),
            mumdia_io::hash::blake3_file(path),
        ) {
            prov.record_input(role, path, bytes, hash);
        }
    }
    let manifest = serde_json::json!({
        "mumdia_version": prov.mumdia_version,
        "git_sha": prov.git_sha,
        "commit_date": prov.commit_date,
        "cli_args": prov.cli_args,
        "inputs": prov.inputs,
        "experiment": manifest,
    });
    mumdia_io::json::write_json(&d("experiment_manifest.json"), &manifest)?;
    info!(
        elapsed_ms = t0.elapsed().as_millis(),
        n_runs, "run-experiment: complete"
    );
    Ok(())
}
