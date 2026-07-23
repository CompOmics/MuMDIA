//! Orchestrator `mumdia run` (PLAN.md Section 4 Orchestration): sequence the MVP
//! stage chain on one run and write a JSON run manifest. The orchestrator only
//! threads file paths; all computation lives in the stage commands, so each
//! remains independently runnable (PLAN.md Section 3.5).

use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::Config;
use mumdia_core::manifest::Manifest;
use mumdia_core::schema::artifact;
use mumdia_io::record_artifact;
use mumdia_io::report::ArtifactReport;
use tracing::info;

use crate::stages::*;

pub struct RunParams<'a> {
    pub config: &'a Config,
    /// FASTA to digest into the spectral library. Required unless a prebuilt
    /// library is supplied via `lib_precursors` + `lib_fragments`.
    pub fasta: Option<&'a str>,
    pub mzml: &'a str,
    pub out_dir: &'a str,
    /// Library-input mode: consume this prebuilt precursor library (e.g. an
    /// imported DIA-NN speclib) instead of digesting the FASTA. Requires
    /// `lib_fragments`; when both are set, digest/peptidoforms/predict-frag are
    /// skipped and the FASTA is not read.
    pub lib_precursors: Option<&'a str>,
    pub lib_fragments: Option<&'a str>,
    pub max_spectra: usize,
    pub top_peaks_ms2: usize,
}

/// Validate inputs and sidecar configuration before any multi-minute compute,
/// so a missing file or a misconfigured rescorer fails immediately with an
/// actionable message.
fn preflight(p: &RunParams) -> Result<()> {
    use mumdia_core::config::RescorerKind;
    // Inputs depend on the mode: library-input supplies a prebuilt library and
    // skips the FASTA; otherwise the FASTA is digested into the library.
    let mut required: Vec<(&str, &str)> = vec![("--mzml", p.mzml)];
    match (p.lib_precursors, p.lib_fragments) {
        (Some(lp), Some(lf)) => {
            required.push(("--lib-precursors", lp));
            required.push(("--lib-fragments", lf));
        }
        (None, None) => match p.fasta {
            Some(f) => required.push(("--fasta", f)),
            None => anyhow::bail!(
                "provide either --fasta (to digest a library) or both \
                 --lib-precursors and --lib-fragments (library-input mode)"
            ),
        },
        _ => anyhow::bail!("library-input mode requires both --lib-precursors and --lib-fragments"),
    }
    for (flag, path) in required {
        if !std::path::Path::new(path).exists() {
            anyhow::bail!("{flag} not found or unreadable: {path}");
        }
    }
    if p.config.rt_im_train.finetune_deeplc && p.config.predict_frag.deeplc_python.is_none() {
        anyhow::bail!(
            "rt_im_train.finetune_deeplc requires predict_frag.deeplc_python (a Python \
             interpreter with DeepLC 4.0 multitask installed)"
        );
    }
    match p.config.rescore.classifier {
        RescorerKind::Mokapot | RescorerKind::NnTorch if p.config.rescore.python.is_none() => {
            anyhow::bail!(
                "rescore.classifier={:?} requires rescore.python (a Python interpreter \
                 with the selected rescorer's dependencies), or use classifier=native_tda",
                p.config.rescore.classifier
            )
        }
        RescorerKind::Entrapment if p.config.rescore.entrapment_marker.is_none() => anyhow::bail!(
            "rescore.classifier=entrapment requires rescore.entrapment_marker (the spike-in \
             accession substring, e.g. \"_HUMAN\")"
        ),
        _ => {}
    }
    Ok(())
}

pub fn run(p: RunParams) -> Result<()> {
    let t0 = Instant::now();
    let cfg = p.config;
    preflight(&p)?;
    let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
    std::fs::create_dir_all(p.out_dir).ok();
    let d = |name: &str| format!("{}/{}", p.out_dir, name);

    let mut man = Manifest::new(cfg.canonical_json(), ch.clone());

    // --- experiment-wide artifacts: the spectral library ---
    // Either digest the FASTA (default) or consume a prebuilt library
    // (library-input mode), then feed the same lib_p/lib_f downstream.
    let (lib_p, lib_f) = match (p.lib_precursors, p.lib_fragments) {
        (Some(lp), Some(lf)) => {
            // Library-input mode: skip digest -> peptidoforms -> predict-frag and
            // consume the supplied library (e.g. an imported DIA-NN speclib).
            info!(
                lib_precursors = lp,
                lib_fragments = lf,
                "run: library-input mode (skipping digest/peptidoforms/predict-frag)"
            );
            let np = mumdia_io::table::Table::read(lp)?.nrows as u64;
            let nf = mumdia_io::table::Table::read(lf)?.nrows as u64;
            man.record(record_artifact(
                artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
                artifact::FRAGMENT_LIBRARY_PRECURSORS,
                lp,
                np,
                "library-input",
                &ch,
            )?);
            man.record(record_artifact(
                artifact::FRAGMENT_LIBRARY_FRAGMENTS.0,
                artifact::FRAGMENT_LIBRARY_FRAGMENTS,
                lf,
                nf,
                "library-input",
                &ch,
            )?);
            (lp.to_string(), lf.to_string())
        }
        _ => {
            // Build the library from the FASTA digest. preflight guarantees the
            // FASTA is present in this branch.
            let fasta = p.fasta.expect("preflight guarantees --fasta in build mode");
            let dig = d("peptides.parquet");
            let n = digest::run(digest::DigestParams {
                fasta,
                out: &dig,
                cfg: &cfg.digest,
                rng_seed: cfg.rng_seed,
                config_hash: &ch,
            })?;
            man.record(record_artifact(
                artifact::PEPTIDES.0,
                artifact::PEPTIDES,
                &dig,
                n,
                "digest",
                &ch,
            )?);

            let pf = d("peptidoforms.parquet");
            let n = peptidoforms::run(peptidoforms::PeptidoformsParams {
                peptides: &dig,
                out: &pf,
                cfg: &cfg.peptidoforms,
                config_hash: &ch,
            })?;
            man.record(record_artifact(
                artifact::PEPTIDOFORMS.0,
                artifact::PEPTIDOFORMS,
                &pf,
                n,
                "peptidoforms",
                &ch,
            )?);

            let lib_p = d("fragment_library_precursors.parquet");
            let lib_f = d("fragment_library_fragments.parquet");
            let (np, nf) = predict_frag::run(predict_frag::PredictFragParams {
                peptidoforms: &pf,
                out_precursors: &lib_p,
                out_fragments: &lib_f,
                work_dir: &d("sidecar_work"),
                cfg: &cfg.predict_frag,
                config_hash: &ch,
            })?;
            man.record(record_artifact(
                artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
                artifact::FRAGMENT_LIBRARY_PRECURSORS,
                &lib_p,
                np,
                "predict-frag",
                &ch,
            )?);
            man.record(record_artifact(
                artifact::FRAGMENT_LIBRARY_FRAGMENTS.0,
                artifact::FRAGMENT_LIBRARY_FRAGMENTS,
                &lib_f,
                nf,
                "predict-frag",
                &ch,
            )?);
            (lib_p, lib_f)
        }
    };

    // --- per-run artifacts ---
    let spectra_dir = d("spectra");
    let co = convert::run(convert::ConvertParams {
        mzml: p.mzml,
        out_dir: &spectra_dir,
        max_spectra: p.max_spectra,
        top_peaks_ms2: p.top_peaks_ms2,
        top_peaks_ms1: 0,
        config_hash: &ch,
    })?;
    for (name, schema, path) in [
        ("spectra_ms1", artifact::SPECTRA_MS1, &co.ms1),
        ("spectra_ms2", artifact::SPECTRA_MS2, &co.ms2),
        (
            "isolation_windows",
            artifact::ISOLATION_WINDOWS,
            &co.isolation_windows,
        ),
        ("ms2_to_ms1", artifact::MS2_TO_MS1, &co.ms2_to_ms1),
    ] {
        let rows = mumdia_io::table::Table::read(path)?.nrows as u64;
        man.record(record_artifact(name, schema, path, rows, "convert", &ch)?);
    }

    let seed = d("seed_psms.parquet");
    let n = search_seed::run(search_seed::SearchSeedParams {
        ms2: &co.ms2,
        library_precursors: &lib_p,
        library_fragments: &lib_f,
        out: &seed,
        cfg: &cfg.search_seed,
        bucket_size: cfg.extract.bucket_size,
        config_hash: &ch,
    })?;
    man.record(record_artifact(
        artifact::SEED_PSMS.0,
        artifact::SEED_PSMS,
        &seed,
        n,
        "search-seed",
        &ch,
    )?);

    // Optional DeepLC multitask fine-tune: adapt the RT model to this run's
    // confident seed PSMs and rewrite the library's predicted_irt before RT
    // calibration. The seed is iRT-independent, so it was computed above on the
    // base library and is reused here. rt-im-train and extract then read the
    // fine-tuned library.
    let lib_p = if cfg.rt_im_train.finetune_deeplc {
        let python = cfg
            .predict_frag
            .deeplc_python
            .as_deref()
            .expect("preflight guarantees deeplc_python when finetune_deeplc is set");
        let script = crate::sidecar::resolve_script(
            &cfg.predict_frag.sidecar_script_dir,
            "deeplc_finetune.py",
        );
        let lib_p_ft = d("fragment_library_precursors_ft.parquet");
        crate::sidecar::run_deeplc_finetune(
            python,
            &script,
            &lib_p,
            &seed,
            &lib_p_ft,
            cfg.rt_im_train.finetune_epochs,
            cfg.rt_im_train.finetune_patience,
            cfg.rt_im_train.q_train,
            cfg.rt_im_train.finetune_batch,
        )?;
        // The fine-tuned precursor table is the artifact actually consumed by
        // RT calibration and extraction. Replace the base-library manifest entry
        // so provenance points at the downstream input instead of only at the
        // pre-fine-tune table.
        let n_ft = mumdia_io::table::Table::read(&lib_p_ft)?.nrows as u64;
        man.record(record_artifact(
            artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
            artifact::FRAGMENT_LIBRARY_PRECURSORS,
            &lib_p_ft,
            n_ft,
            "deeplc-finetune",
            &ch,
        )?);
        lib_p_ft
    } else {
        lib_p
    };

    let windows = d("run_windows.parquet");
    let cal = d("cal.json");
    let n = rt_im_train::run(rt_im_train::RtImTrainParams {
        seed_psms: &seed,
        library_precursors: &lib_p,
        out_windows: &windows,
        out_cal: &cal,
        cfg: &cfg.rt_im_train,
        config_hash: &ch,
    })?;
    man.record(record_artifact(
        artifact::RUN_WINDOWS.0,
        artifact::RUN_WINDOWS,
        &windows,
        n,
        "rt-im-train",
        &ch,
    )?);

    let psms = d("psms_extracted.parquet");
    let chrom = d("chromatograms.parquet");
    let (npsm, nchr) = extract::run(extract::ExtractParams {
        ms2: &co.ms2,
        library_precursors: &lib_p,
        library_fragments: &lib_f,
        run_windows: &windows,
        ms1: Some(&co.ms1),
        mass_cal: Some(&format!("{seed}.masscal.json")),
        out_psms: &psms,
        out_chrom: &chrom,
        restrict_candidates: None,
        cfg: &cfg.extract,
        config_hash: &ch,
    })?;
    man.record(record_artifact(
        artifact::PSMS_EXTRACTED.0,
        artifact::PSMS_EXTRACTED,
        &psms,
        npsm,
        "extract",
        &ch,
    )?);
    man.record(record_artifact(
        artifact::CHROMATOGRAMS.0,
        artifact::CHROMATOGRAMS,
        &chrom,
        nchr,
        "extract",
        &ch,
    )?);

    let feats = d("features.parquet");
    let pin = d("run.pin");
    let n = features::run(features::FeaturesParams {
        psms: &psms,
        chromatograms: &chrom,
        seed: Some(&seed),
        out: &feats,
        out_pin: &pin,
        cfg: &cfg.features,
        config_hash: &ch,
    })?;
    man.record(record_artifact(
        artifact::FEATURES.0,
        artifact::FEATURES,
        &feats,
        n,
        "features",
        &ch,
    )?);

    let competed = d("psms_competed.parquet");
    let n = compete::run(compete::CompeteParams {
        features: &feats,
        out: &competed,
        cfg: &cfg.compete,
        config_hash: &ch,
    })?;
    man.record(record_artifact(
        artifact::PSMS_COMPETED.0,
        artifact::PSMS_COMPETED,
        &competed,
        n,
        "compete",
        &ch,
    )?);

    let scored = d("psms_scored.parquet");
    let n = rescore::run(rescore::RescoreParams {
        competed: std::slice::from_ref(&competed),
        out: &scored,
        work_dir: &d("sidecar_work"),
        script_dir: &cfg.predict_frag.sidecar_script_dir,
        cfg: &cfg.rescore,
        config_hash: &ch,
    })?;
    man.record(record_artifact(
        artifact::PSMS_SCORED.0,
        artifact::PSMS_SCORED,
        &scored,
        n,
        "rescore",
        &ch,
    )?);
    // Use the rescore artifact report as the source of truth. The configured
    // sidecar may differ from the model that actually ran in compatibility mode,
    // and the report records that distinction.
    let rescore_report: ArtifactReport =
        mumdia_io::json::read_json(&format!("{scored}.report.json"))?;
    let actual_rescorer = rescore_report
        .params
        .get("classifier")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let actual_rescorer_model = rescore_report
        .model_identity
        .clone()
        .unwrap_or_else(|| actual_rescorer.clone());

    // Optional candidate audit (sensitivity program, P0.3): reconstruct the
    // per-candidate identification-loss ladder from the artifact chain. Off by
    // default (gated on extract.emit_candidate_audit); adds one cheap join pass.
    if cfg.extract.emit_candidate_audit {
        let audit_out = d("candidate_audit.parquet");
        audit::run(audit::AuditParams {
            library_precursors: &lib_p,
            psms: &psms,
            competed: &competed,
            scored: &scored,
            out: &audit_out,
            q_threshold: 0.01,
            run_id: p.out_dir,
            entrapment_substr: "",
        })?;
    }

    let pep_q = d("peptide_quant.parquet");
    let pg_q = d("protein_group_quant.parquet");
    let frag_q = d("fragment_quant.parquet");
    let (nq1, nq2) = quant::run(quant::QuantParams {
        psms_scored: &scored,
        chromatograms: &chrom,
        out_peptide: &pep_q,
        out_protein: &pg_q,
        out_fragment: Some(&frag_q),
        out_peak_bounds: None,
        cfg: &cfg.quant,
        config_hash: &ch,
    })?;
    man.record(record_artifact(
        artifact::PEPTIDE_QUANT.0,
        artifact::PEPTIDE_QUANT,
        &pep_q,
        nq1,
        "quant",
        &ch,
    )?);
    man.record(record_artifact(
        artifact::PROTEIN_GROUP_QUANT.0,
        artifact::PROTEIN_GROUP_QUANT,
        &pg_q,
        nq2,
        "quant",
        &ch,
    )?);
    let n_frag_quant = mumdia_io::table::Table::read(&frag_q)?.nrows as u64;
    man.record(record_artifact(
        artifact::FRAGMENT_QUANT.0,
        artifact::FRAGMENT_QUANT,
        &frag_q,
        n_frag_quant,
        "quant",
        &ch,
    )?);

    // Human-readable report (peptides.tsv + proteins.tsv) + stdout summary.
    let pep_tsv = d("peptides.tsv");
    let prot_tsv = d("proteins.tsv");
    let (n_pep, n_prot) = report::run(report::ReportParams {
        scored: &scored,
        peptide_quant: Some(&pep_q),
        protein_quant: Some(&pg_q),
        out_peptides: &pep_tsv,
        out_proteins: &prot_tsv,
        q_threshold: cfg.quant.q_threshold,
    })?;
    println!(
        "MuMDIA: {n_pep} precursor rows, {n_prot} protein groups at peptide/PG q <= {} (rescorer used: {})\n  {}\n  {}",
        cfg.quant.q_threshold, actual_rescorer, pep_tsv, prot_tsv
    );

    // Model identities reflect the path that produced the downstream artifacts,
    // including imported libraries and per-run RT fine-tuning.
    let library_input = p.lib_precursors.is_some();
    let rt_identity = if cfg.rt_im_train.finetune_deeplc {
        "deeplc-finetuned".to_string()
    } else if library_input {
        "imported-library".to_string()
    } else {
        format!("{:?}", cfg.predict_frag.rt_predictor)
    };
    let fragment_identity = if library_input {
        "imported-library".to_string()
    } else {
        format!("{:?}", cfg.predict_frag.predictor)
    };
    man.model_identities
        .insert("rt_predictor".into(), rt_identity);
    man.model_identities
        .insert("fragment_predictor".into(), fragment_identity);
    man.model_identities
        .insert("rescorer".into(), actual_rescorer_model);
    man.model_identities.insert(
        "feature_schema_id".into(),
        features::feature_schema_id(&features::active_features(cfg.features.set)),
    );

    let manifest_path = d("manifest.json");
    mumdia_io::json::write_json(&manifest_path, &man)?;

    info!(
        elapsed_ms = t0.elapsed().as_millis(),
        manifest = manifest_path,
        "run: pipeline complete"
    );
    Ok(())
}
