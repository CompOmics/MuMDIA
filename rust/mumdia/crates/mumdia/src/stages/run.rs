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
use tracing::info;

use crate::stages::*;

pub struct RunParams<'a> {
    pub config: &'a Config,
    pub fasta: &'a str,
    pub mzml: &'a str,
    pub out_dir: &'a str,
    pub max_spectra: usize,
    pub top_peaks_ms2: usize,
}

/// Validate inputs and sidecar configuration before any multi-minute compute,
/// so a missing file or a misconfigured rescorer fails immediately with an
/// actionable message.
fn preflight(p: &RunParams) -> Result<()> {
    use mumdia_core::config::RescorerKind;
    for (flag, path) in [("--fasta", p.fasta), ("--mzml", p.mzml)] {
        if !std::path::Path::new(path).exists() {
            anyhow::bail!("{flag} not found or unreadable: {path}");
        }
    }
    match p.config.rescore.classifier {
        RescorerKind::Mokapot if p.config.rescore.python.is_none() => anyhow::bail!(
            "rescore.classifier=mokapot requires rescore.python (a Python interpreter with \
             mokapot installed; see env/mumdia-rescore.yml), or use classifier=native_tda"
        ),
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

    // --- experiment-wide artifacts ---
    let dig = d("peptides.parquet");
    let n = digest::run(digest::DigestParams {
        fasta: p.fasta,
        out: &dig,
        cfg: &cfg.digest,
        rng_seed: cfg.rng_seed,
        config_hash: &ch,
    })?;
    man.record(record_artifact(artifact::PEPTIDES.0, artifact::PEPTIDES, &dig, n, "digest", &ch)?);

    let pf = d("peptidoforms.parquet");
    let n = peptidoforms::run(peptidoforms::PeptidoformsParams {
        peptides: &dig,
        out: &pf,
        cfg: &cfg.peptidoforms,
        config_hash: &ch,
    })?;
    man.record(record_artifact(artifact::PEPTIDOFORMS.0, artifact::PEPTIDOFORMS, &pf, n, "peptidoforms", &ch)?);

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
    man.record(record_artifact(artifact::FRAGMENT_LIBRARY_PRECURSORS.0, artifact::FRAGMENT_LIBRARY_PRECURSORS, &lib_p, np, "predict-frag", &ch)?);
    man.record(record_artifact(artifact::FRAGMENT_LIBRARY_FRAGMENTS.0, artifact::FRAGMENT_LIBRARY_FRAGMENTS, &lib_f, nf, "predict-frag", &ch)?);

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
        ("isolation_windows", artifact::ISOLATION_WINDOWS, &co.isolation_windows),
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
    man.record(record_artifact(artifact::SEED_PSMS.0, artifact::SEED_PSMS, &seed, n, "search-seed", &ch)?);

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
    man.record(record_artifact(artifact::RUN_WINDOWS.0, artifact::RUN_WINDOWS, &windows, n, "rt-im-train", &ch)?);

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
        cfg: &cfg.extract,
        config_hash: &ch,
    })?;
    man.record(record_artifact(artifact::PSMS_EXTRACTED.0, artifact::PSMS_EXTRACTED, &psms, npsm, "extract", &ch)?);
    man.record(record_artifact(artifact::CHROMATOGRAMS.0, artifact::CHROMATOGRAMS, &chrom, nchr, "extract", &ch)?);

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
    man.record(record_artifact(artifact::FEATURES.0, artifact::FEATURES, &feats, n, "features", &ch)?);

    let competed = d("psms_competed.parquet");
    let n = compete::run(compete::CompeteParams {
        features: &feats,
        out: &competed,
        cfg: &cfg.compete,
        config_hash: &ch,
    })?;
    man.record(record_artifact(artifact::PSMS_COMPETED.0, artifact::PSMS_COMPETED, &competed, n, "compete", &ch)?);

    let scored = d("psms_scored.parquet");
    let n = rescore::run(rescore::RescoreParams {
        competed: &[competed.clone()],
        out: &scored,
        work_dir: &d("sidecar_work"),
        script_dir: &cfg.predict_frag.sidecar_script_dir,
        cfg: &cfg.rescore,
        config_hash: &ch,
    })?;
    man.record(record_artifact(artifact::PSMS_SCORED.0, artifact::PSMS_SCORED, &scored, n, "rescore", &ch)?);

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
    man.record(record_artifact(artifact::PEPTIDE_QUANT.0, artifact::PEPTIDE_QUANT, &pep_q, nq1, "quant", &ch)?);
    man.record(record_artifact(artifact::PROTEIN_GROUP_QUANT.0, artifact::PROTEIN_GROUP_QUANT, &pg_q, nq2, "quant", &ch)?);

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
        "MuMDIA: {n_pep} peptides, {n_prot} protein groups at q <= {} (rescorer: {:?})\n  {}\n  {}",
        cfg.quant.q_threshold, cfg.rescore.classifier, pep_tsv, prot_tsv
    );

    // Model identities.
    man.model_identities
        .insert("rt_predictor".into(), format!("{:?}", cfg.predict_frag.rt_predictor));
    man.model_identities
        .insert("fragment_predictor".into(), format!("{:?}", cfg.predict_frag.predictor));
    man.model_identities
        .insert("rescorer".into(), format!("{:?}", cfg.rescore.classifier));
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
