//! Orchestrator `mumdia run` (docs/01_overview_and_dataflow.md): sequence the MVP
//! stage chain on one run and write a JSON run manifest. The orchestrator only
//! threads file paths; all computation lives in the stage commands, so each
//! remains independently runnable.

use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::Config;
use mumdia_core::manifest::Manifest;
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::{record_artifact, record_artifact_with_hash};
use tracing::{info, warn};

use crate::stages::*;

pub struct RunParams<'a> {
    pub config: &'a Config,
    /// Path the config was loaded from, if any. Used only to resolve a relative
    /// `predict_frag.sidecar_script_dir` against the config's own directory
    /// instead of against the current working directory, which silently changed
    /// which worker scripts ran depending on where the command was invoked.
    pub config_path: Option<&'a str>,
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

/// A library-input artifact record waiting for its input hash (see `run`).
struct LibraryInputRecord {
    /// The artifact schema; its name is also the manifest's logical name.
    schema: (&'static str, u32),
    path: String,
    rows: u64,
    /// The manifest input role whose hash is this file's content hash.
    input_role: &'static str,
}

/// Validate inputs and sidecar configuration before any multi-minute compute,
/// so a missing file or a misconfigured rescorer fails immediately with an
/// actionable message.
fn preflight(p: &RunParams, cfg: &Config) -> Result<()> {
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
    // The interpreter fields were filled in by `python::resolve` before this point,
    // so a field still empty here means the role is unused or discovery failed.
    // The messages name the field and the alternative, which is what a user acts on.
    // An EXPLICIT head count is a hard requirement; the automatic default is not, or a
    // native Python-free run -- a supported configuration -- would fail at startup.
    if cfg.rt_im_train.multihead_calibration.is_some_and(|n| n > 0)
        && cfg.predict_frag.deeplc_python.is_none()
    {
        anyhow::bail!(
            "rt_im_train.multihead_calibration requires predict_frag.deeplc_python (a Python \
             interpreter with DeepLC >= 4.4.0, or \"auto\" to discover one); it calibrates \
             the DeepLC base model against this run's confident seed PSMs"
        );
    }
    if cfg.rt_im_train.multihead_calibration.is_none()
        && !cfg.rt_im_train.finetune_deeplc
        && cfg.predict_frag.deeplc_python.is_none()
    {
        warn!(
            "no DeepLC interpreter resolved, so retention-time multi-head calibration is \
             not running. It is the default and is worth 4.8% of peptides on AIF and 14.3% \
             on Astral; set predict_frag.deeplc_python (\"auto\" discovers one). The \
             manifest records the retention-time model that actually ran."
        );
    }
    if cfg.rt_im_train.finetune_deeplc && cfg.predict_frag.deeplc_python.is_none() {
        anyhow::bail!(
            "rt_im_train.finetune_deeplc requires predict_frag.deeplc_python (a Python \
             interpreter with DeepLC installed; see env/mumdia-deeplc.yml), or set that \
             field to \"auto\" to discover one"
        );
    }
    if p.lib_precursors.is_some()
        && matches!(
            cfg.rt_im_train.library_irt,
            mumdia_core::config::LibraryIrt::Deeplc
        )
        && cfg.predict_frag.deeplc_python.is_none()
    {
        anyhow::bail!(
            "rt_im_train.library_irt = deeplc requires predict_frag.deeplc_python (a Python \
             interpreter with DeepLC >= 4.4.0, or \"auto\" to discover one); set \
             library_irt = library to keep the imported iRT"
        );
    }
    match cfg.rescore.classifier {
        RescorerKind::Mokapot | RescorerKind::NnTorch if cfg.rescore.python.is_none() => {
            anyhow::bail!(
                "rescore.classifier={:?} requires rescore.python (a Python interpreter \
                 with the selected rescorer's dependencies), or \"auto\" to discover one, \
                 or use classifier=native_tda",
                cfg.rescore.classifier
            )
        }
        RescorerKind::Entrapment if cfg.rescore.entrapment_marker.is_none() => anyhow::bail!(
            "rescore.classifier=entrapment requires rescore.entrapment_marker (the spike-in \
             accession substring, e.g. \"_HUMAN\")"
        ),
        _ => {}
    }
    Ok(())
}

pub fn run(p: RunParams) -> Result<()> {
    let t0 = Instant::now();
    // The time from here to the first stage, by step (`prestage::PreStageTimer`).
    let mut pre = crate::prestage::PreStageTimer::start("run");
    // Fill in the sidecar interpreters and the worker directory before anything is
    // validated or hashed, so every stage sees a concrete path and the manifest
    // records the interpreter that actually ran rather than the word "auto".
    let mut resolved = p.config.clone();
    resolved.predict_frag.sidecar_script_dir =
        crate::python::resolve_script_dir(&resolved.predict_frag.sidecar_script_dir, p.config_path);
    crate::python::resolve(&mut resolved)?;
    pre.step("resolve_interpreters");
    let cfg = &resolved;
    preflight(&p, cfg)?;
    pre.step("preflight");
    let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
    std::fs::create_dir_all(p.out_dir).ok();
    let d = |name: &str| format!("{}/{}", p.out_dir, name);

    let mut man = Manifest::new(cfg.canonical_json(), ch.clone());
    info!(provenance = %man.provenance(), "run: build provenance");
    // Hash the inputs, starting now, before any stage reads them. This is what ties a
    // result to the exact bytes it came from; recording only the path does not, because a
    // path is reused. The hash is the FIRST read of each input, so it is the cold one: on
    // a large library it is minutes of sequential I/O, and it used to sit on the critical
    // path before the first stage, one input after another. It runs on a background thread
    // instead, in the order the stages below read the files, and is joined only when the
    // manifest is written, so the manifest records exactly what the serial loop recorded
    // (`prestage::InputHashes`). A missing input is a preflight error; one that becomes
    // unreadable since then is left out of the manifest with a warning, as before.
    let input_hashes = crate::prestage::InputHashes::spawn(
        "run",
        [
            ("mzml", Some(p.mzml)),
            ("fasta", p.fasta),
            ("lib_precursors", p.lib_precursors),
            ("lib_fragments", p.lib_fragments),
        ]
        .into_iter()
        .filter_map(|(role, path)| path.map(|x| (role.to_string(), x.to_string())))
        .collect(),
    );
    // The two library-input records, completed once the input hashes are joined: they are
    // the same files, so their content hash IS the input hash, and hashing them here as
    // well read a multi-GB library a second time before the first stage. A later stage
    // that records an adapted precursor table under the same logical name wins, exactly as
    // it did when this record was inserted first and then overwritten (end of `run`).
    let mut library_input_records: Vec<LibraryInputRecord> = Vec::new();
    pre.step("provenance");

    // A FASTA build may leave DeepLC to the multi-head calibration, which re-predicts every
    // row before anything reads the iRT (`predict_frag.defer_deeplc_to_multihead`).
    let rt_placeholder = cfg.defers_library_deeplc(
        p.lib_precursors.is_some(),
        cfg.predict_frag.deeplc_python.is_some(),
    );

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
            let np = mumdia_io::table::nrows(lp)?;
            let nf = mumdia_io::table::nrows(lf)?;
            library_input_records.push(LibraryInputRecord {
                schema: artifact::FRAGMENT_LIBRARY_PRECURSORS,
                path: lp.to_string(),
                rows: np,
                input_role: "lib_precursors",
            });
            library_input_records.push(LibraryInputRecord {
                schema: artifact::FRAGMENT_LIBRARY_FRAGMENTS,
                path: lf.to_string(),
                rows: nf,
                input_role: "lib_fragments",
            });
            (lp.to_string(), lf.to_string())
        }
        _ => {
            // Build the library from the FASTA digest. preflight guarantees the
            // FASTA is present in this branch.
            let fasta = p.fasta.expect("preflight guarantees --fasta in build mode");
            let lib_p = d("fragment_library_precursors.parquet");
            let lib_f = d("fragment_library_fragments.parquet");
            // `predict_frag.library_cache`: a library stored by an earlier run with the same
            // FASTA, build settings, predictor versions and engine is published here instead
            // of being built (`library_cache`).
            let cache = crate::library_cache::LibraryCache::for_config(
                cfg,
                fasta,
                rt_placeholder,
                (&man.mumdia_version, &man.git_sha),
            );
            if let Some((wp, wf)) = cache.as_ref().and_then(|c| c.restore(&lib_p, &lib_f)) {
                pre.first_stage("library-cache");
                // A reused output directory may still hold an earlier build's digest and
                // peptidoforms, which did not produce this library.
                crate::library_cache::remove_build_intermediates(p.out_dir);
                man.record(wp.record(
                    artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
                    artifact::FRAGMENT_LIBRARY_PRECURSORS,
                    &lib_p,
                    "library-cache",
                    &ch,
                ));
                man.record(wf.record(
                    artifact::FRAGMENT_LIBRARY_FRAGMENTS.0,
                    artifact::FRAGMENT_LIBRARY_FRAGMENTS,
                    &lib_f,
                    "library-cache",
                    &ch,
                ));
            } else {
                let dig = d("peptides.parquet");
                pre.first_stage("digest");
                let w = digest::run_hashed(digest::DigestParams {
                    fasta,
                    out: &dig,
                    cfg: &cfg.digest,
                    rng_seed: cfg.rng_seed,
                    config_hash: &ch,
                })?;
                man.record(w.record(
                    artifact::PEPTIDES.0,
                    artifact::PEPTIDES,
                    &dig,
                    "digest",
                    &ch,
                ));

                let pf = d("peptidoforms.parquet");
                let w = peptidoforms::run_hashed(peptidoforms::PeptidoformsParams {
                    peptides: &dig,
                    out: &pf,
                    cfg: &cfg.peptidoforms,
                    config_hash: &ch,
                })?;
                man.record(w.record(
                    artifact::PEPTIDOFORMS.0,
                    artifact::PEPTIDOFORMS,
                    &pf,
                    "peptidoforms",
                    &ch,
                ));

                if cfg.predict_frag.defer_deeplc_to_multihead && !rt_placeholder {
                    info!(
                        "run: predict_frag.defer_deeplc_to_multihead is set, but no multi-head \
                         calibration re-predicts this library (it needs rt_predictor = deeplc \
                         and a DeepLC interpreter); predicting it with DeepLC as usual"
                    );
                }
                let (wp, wf) = predict_frag::run_hashed(predict_frag::PredictFragParams {
                    rt_placeholder,
                    peptidoforms: &pf,
                    out_precursors: &lib_p,
                    out_fragments: &lib_f,
                    work_dir: &d("sidecar_work"),
                    cfg: &cfg.predict_frag,
                    config_hash: &ch,
                })?;
                man.record(wp.record(
                    artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
                    artifact::FRAGMENT_LIBRARY_PRECURSORS,
                    &lib_p,
                    "predict-frag",
                    &ch,
                ));
                man.record(wf.record(
                    artifact::FRAGMENT_LIBRARY_FRAGMENTS.0,
                    artifact::FRAGMENT_LIBRARY_FRAGMENTS,
                    &lib_f,
                    "predict-frag",
                    &ch,
                ));
                match &cache {
                    Some(c) => c.store(&lib_p, &lib_f),
                    None => {
                        if let Some(hint) =
                            crate::library_cache::reuse_hint(cfg, &lib_p, &lib_f, rt_placeholder)
                        {
                            info!(
                                "run: to search another file against this library without \
                                 building it again, pass {hint}, or set \
                                 predict_frag.library_cache to a directory and keep --fasta"
                            );
                        }
                    }
                }
            }
            (lib_p, lib_f)
        }
    };

    // --- per-run artifacts ---
    let spectra_dir = d("spectra");
    // Fold the conversion caps into the convert artifacts' provenance key, exactly as the
    // standalone `convert` subcommand does. They change the spectra output but are not part
    // of the config, so passing the bare config hash here made two runs with different caps
    // record an identical config_hash for their convert artifacts -- provenance that
    // disagreed with the standalone entry point for the same inputs.
    let convert_hash = mumdia_io::hash::blake3_str(&format!(
        "{}\u{1f}max_spectra={}\u{1f}top_peaks_ms2={}\u{1f}top_peaks_ms1={}",
        cfg.canonical_json(),
        p.max_spectra,
        p.top_peaks_ms2,
        0
    ));
    pre.first_stage("convert");
    info!(stage = %"convert", "run: stage start");
    let co = convert::run(convert::ConvertParams {
        mzml: p.mzml,
        out_dir: &spectra_dir,
        max_spectra: p.max_spectra,
        top_peaks_ms2: p.top_peaks_ms2,
        top_peaks_ms1: 0,
        config_hash: &convert_hash,
    })?;
    for (name, schema, path, hash) in [
        (
            "spectra_ms1",
            artifact::SPECTRA_MS1,
            &co.ms1,
            &co.hashes.ms1,
        ),
        (
            "spectra_ms2",
            artifact::SPECTRA_MS2,
            &co.ms2,
            &co.hashes.ms2,
        ),
        (
            "isolation_windows",
            artifact::ISOLATION_WINDOWS,
            &co.isolation_windows,
            &co.hashes.isolation_windows,
        ),
        (
            "ms2_to_ms1",
            artifact::MS2_TO_MS1,
            &co.ms2_to_ms1,
            &co.hashes.ms2_to_ms1,
        ),
    ] {
        let rows = mumdia_io::table::nrows(path)?;
        // `convert_hash`, not the bare config hash: the manifest is the provenance record,
        // so it must carry the same cap-folded key the artifact's own report does. Stamping
        // `ch` here made two runs differing only in `--top-peaks-ms2` record identical
        // provenance for their spectra, and disagreed with the report written beside them.
        // The content hash is the one convert computed for that report.
        man.record(record_artifact_with_hash(
            name,
            schema,
            path,
            rows,
            "convert",
            &convert_hash,
            hash.clone(),
        ));
    }

    // The RT model is decided here for both paths: it names the manifest identity, and the
    // grouped path runs it per band.
    let has_deeplc = cfg.predict_frag.deeplc_python.is_some();
    let mh_heads = cfg.rt_im_train.multihead_heads(
        has_deeplc,
        cfg.deeplc_rt_source(p.lib_precursors.is_some(), has_deeplc),
    );
    // Grouped: everything from the seed to the competed table happens one isolation-window
    // group at a time, and the pooled artifacts come back under the same names the
    // ungrouped path writes, so rescore, quant and report below are shared.
    let (seed, lib_p, psms, chrom, feats, competed, grouped_rt_model) =
        if cfg.groups.window_groups > 1 {
            let pooled = run_groups::run(run_groups::GroupRun {
                cfg,
                config_hash: &ch,
                converted: &co,
                lib_precursors: &lib_p,
                lib_fragments: &lib_f,
                out_dir: p.out_dir,
                man: Some(&mut man),
                shared_bands: None,
                mh_heads,
                library_input: p.lib_precursors.is_some(),
                // A single run re-predicts nothing before banding.
                library_irt_repredicted: false,
                slices_from: None,
                irt_placeholder: rt_placeholder,
            })?;
            (
                pooled.seed,
                lib_p.clone(),
                pooled.psms,
                pooled.chromatograms,
                pooled.features,
                pooled.competed,
                Some(pooled.rt_model),
            )
        } else {
            let seed = d("seed_psms.parquet");
            // Ungrouped: the seed and the extract below are the only readers of the
            // spectra. Whether the seed's MS2 decode is lent on to extract depends on what
            // runs between them. A DeepLC step (multi-head calibration, fine-tune or
            // re-prediction) is where the tallest sidecar of a single run sits, and holding
            // the scans (~1 GB) across it would add them to the process-tree peak, so then
            // each stage decodes its own and drops it. Without one, only rt-im-train runs in
            // between, which holds far less than extract does with the scans resident
            // anyway, so the decode is kept and lent: one MS2 decode per run instead of two.
            let rt_sidecar = mh_heads > 0
                || cfg.rt_im_train.finetune_deeplc
                || cfg.rt_im_train.repredicts_library_irt(
                    p.lib_precursors.is_some(),
                    cfg.predict_frag.deeplc_python.is_some(),
                );
            info!(stage = %"search-seed", "run: stage start");
            let (w, seed_ms2) = search_seed::run_returning_scans(search_seed::SearchSeedParams {
                precursor_span: None,
                fragment_offset: None,
                ms2_scans: None,
                emit_calibrants: false,
                library: None,
                ms2: &co.ms2,
                library_precursors: &lib_p,
                library_fragments: &lib_f,
                out: &seed,
                cfg: &cfg.search_seed,
                bucket_size: cfg.extract.bucket_size,
                config_hash: &ch,
            })?;
            man.record(w.record(
                artifact::SEED_PSMS.0,
                artifact::SEED_PSMS,
                &seed,
                "search-seed",
                &ch,
            ));
            // Consumed here either way: kept for extract, or freed now, before any sidecar.
            // (A conditional move would leave the scans alive to the end of this block.)
            // Only on extract's fragindex matcher: the bucketed one builds a sorted copy of
            // every library fragment in its load, and extract keeps its decode out of that
            // transient for the same reason, so scans held from the seed would put it back.
            let lend = !rt_sidecar
                && matches!(
                    cfg.extract.matcher,
                    mumdia_core::config::MatcherKind::Fragindex
                );
            let lent_ms2: Option<Vec<mumdia_core::types::Ms2Scan>> = seed_ms2.filter(|_| lend);

            // Optional DeepLC multitask fine-tune: adapt the RT model to this run's
            // confident seed PSMs and rewrite the library's predicted_irt before RT
            // calibration. The seed is iRT-independent, so it was computed above on the
            // base library and is reused here. rt-im-train and extract then read the
            // fine-tuned library.
            let lib_p = if mh_heads > 0 {
                // Multi-head calibration occupies the fine-tune's slot: it needs this run's
                // confident seed PSMs, which exist only now, and it rewrites the same library the
                // fine-tune would. Validation refuses both at once.
                let python = cfg
                    .predict_frag
                    .deeplc_python
                    .as_deref()
                    .expect("mh_heads is 0 unless an interpreter resolved");
                let script = crate::sidecar::resolve_script(
                    &cfg.predict_frag.sidecar_script_dir,
                    "deeplc_finetune.py",
                );
                let lib_p_mh = d("fragment_library_precursors_multihead.parquet");
                info!(stage = %"deeplc-multihead", "run: stage start");
                crate::sidecar::run_deeplc_multihead(
                    python,
                    &script,
                    &lib_p,
                    &seed,
                    &lib_p_mh,
                    mh_heads,
                    cfg.rt_im_train.q_train,
                    cfg.rt_im_train.window_holdout_frac,
                    rayon::current_num_threads(),
                    cfg.rt_im_train.deeplc_predict_shards,
                    cfg.rt_im_train.deeplc_projection_cache.as_deref(),
                )?;
                if rt_placeholder {
                    crate::sidecar::require_every_row_repredicted(&lib_p_mh)?;
                }
                let n_mh = mumdia_io::table::nrows(&lib_p_mh)?;
                man.record(record_artifact(
                    artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
                    artifact::FRAGMENT_LIBRARY_PRECURSORS,
                    &lib_p_mh,
                    n_mh,
                    "deeplc-multihead",
                    &ch,
                )?);
                lib_p_mh
            } else if cfg.rt_im_train.finetune_deeplc {
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
                info!(stage = %"deeplc-finetune", "run: stage start");
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
                    // Held-out window sizing: the sidecar must exclude the same peptides
                    // from the fine-tune reference that rt-im-train later scores as
                    // held-out, else adapter memorization leaks into the "held-out"
                    // residuals and the window shrinks back toward in-sample optimism.
                    cfg.rt_im_train.window_holdout_frac,
                    cfg.rng_seed,
                    rayon::current_num_threads(),
                    cfg.rt_im_train.deeplc_predict_shards,
                )?;
                // The fine-tuned precursor table is the artifact actually consumed by
                // RT calibration and extraction. Replace the base-library manifest entry
                // so provenance points at the downstream input instead of only at the
                // pre-fine-tune table.
                let n_ft = mumdia_io::table::nrows(&lib_p_ft)?;
                man.record(record_artifact(
                    artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
                    artifact::FRAGMENT_LIBRARY_PRECURSORS,
                    &lib_p_ft,
                    n_ft,
                    "deeplc-finetune",
                    &ch,
                )?);
                lib_p_ft
            } else if cfg.rt_im_train.repredicts_library_irt(
                p.lib_precursors.is_some(),
                cfg.predict_frag.deeplc_python.is_some(),
            ) {
                // Library-input mode without a fine-tune: replace the imported iRT with DeepLC
                // base-model predictions before calibration. The prediction does not depend on
                // the run, so `run-experiment` computes it once for all runs instead.
                let python = cfg
                    .predict_frag
                    .deeplc_python
                    .as_deref()
                    .expect("repredicts_library_irt implies deeplc_python");
                let script = crate::sidecar::resolve_script(
                    &cfg.predict_frag.sidecar_script_dir,
                    "deeplc_finetune.py",
                );
                let lib_p_dl = d("fragment_library_precursors_deeplc.parquet");
                info!(stage = %"deeplc-repredict", "run: stage start");
                crate::sidecar::run_deeplc_repredict(
                    python,
                    &script,
                    &lib_p,
                    &lib_p_dl,
                    rayon::current_num_threads(),
                    cfg.rt_im_train.deeplc_predict_shards,
                    cfg.rt_im_train.deeplc_projection_cache.as_deref(),
                )?;
                let n_dl = mumdia_io::table::nrows(&lib_p_dl)?;
                man.record(record_artifact(
                    artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
                    artifact::FRAGMENT_LIBRARY_PRECURSORS,
                    &lib_p_dl,
                    n_dl,
                    "deeplc-repredict",
                    &ch,
                )?);
                lib_p_dl
            } else {
                // Two reasons land here, and the log has to name the right one: without a DeepLC
                // interpreter the imported iRT is kept as is (worth a warning); with one, the
                // multi-head calibration re-predicts the library itself against this run's anchors
                // and a base-model re-prediction first would only be overwritten
                // (`repredicts_library_irt`), which is a plan, not a shortfall.
                if p.lib_precursors.is_some()
                    && matches!(
                        cfg.rt_im_train.library_irt,
                        mumdia_core::config::LibraryIrt::Auto
                    )
                {
                    if cfg.predict_frag.deeplc_python.is_none() {
                        tracing::warn!(
                    "run: keeping the imported library iRT because no predict_frag.deeplc_python \
                     is configured; configure one to re-predict with DeepLC, or set \
                     rt_im_train.library_irt = library to silence this"
                );
                    } else {
                        tracing::info!(
                        "run: the multi-head calibration re-predicts the library iRT against this \
                     run's anchors; skipping the base-model re-prediction it would overwrite"
                    );
                    }
                }
                lib_p
            };

            let windows = d("run_windows.parquet");
            let cal = d("cal.json");
            info!(stage = %"rt-im-train", "run: stage start");
            // In memory as well as on disk: extract reads the same library, so it takes the
            // fitted windows as they are instead of decoding the table written here
            // (`rt_im_train::RtWindows`). The file stays the artifact.
            let (w, fitted_windows) = rt_im_train::run_in_memory(rt_im_train::RtImTrainParams {
                precursor_span: None,
                anchor_irt_from_seed: false,
                seed_psms: &seed,
                library_precursors: &lib_p,
                out_windows: &windows,
                out_cal: &cal,
                cfg: &cfg.rt_im_train,
                config_hash: &ch,
            })?;
            man.record(w.record(
                artifact::RUN_WINDOWS.0,
                artifact::RUN_WINDOWS,
                &windows,
                "rt-im-train",
                &ch,
            ));

            let psms = d("psms_extracted.parquet");
            let chrom = d("chromatograms.parquet");
            info!(stage = %"extract", "run: stage start");
            // The MS2 only (`SharedScans::ms1 = None`): extract decodes the MS1 itself,
            // concurrently with its library load and after the library's errors, as it does
            // with nothing lent.
            let (wpsm, wchr) = extract::run_hashed(extract::ExtractParams {
                precursor_span: None,
                fragment_offset: None,
                sibling_bands: 1,
                rt_windows: fitted_windows,
                scans: lent_ms2
                    .as_deref()
                    .map(|ms2| extract::SharedScans { ms2, ms1: None }),
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
            drop(lent_ms2);
            man.record(wpsm.record(
                artifact::PSMS_EXTRACTED.0,
                artifact::PSMS_EXTRACTED,
                &psms,
                "extract",
                &ch,
            ));
            man.record(wchr.record(
                artifact::CHROMATOGRAMS.0,
                artifact::chromatograms(cfg.extract.chromatogram_schema),
                &chrom,
                "extract",
                &ch,
            ));

            let feats = d("features.parquet");
            let pin = d("run.pin");
            info!(stage = %"features", "run: stage start");
            let wf = features::run_hashed(features::FeaturesParams {
                psms: &psms,
                chromatograms: &chrom,
                seed: Some(&seed),
                out: &feats,
                out_pin: &pin,
                cfg: &cfg.features,
                config_hash: &ch,
            })?;
            man.record(wf.record(
                artifact::FEATURES.0,
                artifact::FEATURES,
                &feats,
                "features",
                &ch,
            ));

            let competed = d("psms_competed.parquet");
            info!(stage = %"compete", "run: stage start");
            let w = compete::run_hashed(compete::CompeteParams {
                features: &feats,
                out: &competed,
                cfg: &cfg.compete,
                config_hash: &ch,
                features_hash: Some(&wf.content_hash),
            })?;
            man.record(w.record(
                artifact::PSMS_COMPETED.0,
                artifact::PSMS_COMPETED,
                &competed,
                "compete",
                &ch,
            ));
            let chrom = vec![quant::ChromTable::whole(&chrom)];
            (seed, lib_p, psms, chrom, feats, vec![competed], None)
        };
    let _ = &feats;
    let _ = &seed;

    let scored = d("psms_scored.parquet");
    info!(stage = %"rescore", "run: stage start");
    // One table, or a grouped run's band tables (`groups.pool_competed = false`), which are
    // all this run's rows: source 0 for every one.
    let sources = vec![0u32; competed.len()];
    let w = rescore::run_hashed(rescore::RescoreParams {
        competed: &competed,
        sources: (competed.len() > 1).then_some(sources.as_slice()),
        out: &scored,
        work_dir: &rescore::sidecar_work_dir(&d("sidecar_work")),
        script_dir: &cfg.predict_frag.sidecar_script_dir,
        cfg: &cfg.rescore,
        config_hash: &ch,
    })?;
    man.record(w.record(
        artifact::PSMS_SCORED.0,
        artifact::PSMS_SCORED,
        &scored,
        "rescore",
        &ch,
    ));
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
        if competed.len() != 1 {
            anyhow::bail!(
                "the candidate audit reads one competed table, and this run has {}",
                competed.len()
            );
        }
        info!(stage = %"audit", "run: stage start");
        audit::run(audit::AuditParams {
            library_precursors: &lib_p,
            psms: &psms,
            // The audit keeps the grouped run's competed table pooled (`run_groups`).
            competed: &competed[0],
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
    info!(stage = %"quant", "run: stage start");
    let wq = quant::run_hashed(quant::QuantParams {
        psms_scored: &scored,
        chromatograms: &chrom,
        out_peptide: &pep_q,
        out_protein: &pg_q,
        out_fragment: Some(&frag_q),
        out_peak_bounds: None,
        cfg: &cfg.quant,
        config_hash: &ch,
    })?;
    man.record(wq.peptide.record(
        artifact::PEPTIDE_QUANT.0,
        artifact::PEPTIDE_QUANT,
        &pep_q,
        "quant",
        &ch,
    ));
    man.record(wq.protein.record(
        artifact::PROTEIN_GROUP_QUANT.0,
        artifact::PROTEIN_GROUP_QUANT,
        &pg_q,
        "quant",
        &ch,
    ));
    // The row count from the footer, as before; the hash from quant's own report. Quant
    // writes the fragment table whenever `out_fragment` is set, which it is here.
    let n_frag_quant = mumdia_io::table::nrows(&frag_q)?;
    man.record(match wq.fragment {
        Some(wfq) => record_artifact_with_hash(
            artifact::FRAGMENT_QUANT.0,
            artifact::FRAGMENT_QUANT,
            &frag_q,
            n_frag_quant,
            "quant",
            &ch,
            wfq.content_hash,
        ),
        None => record_artifact(
            artifact::FRAGMENT_QUANT.0,
            artifact::FRAGMENT_QUANT,
            &frag_q,
            n_frag_quant,
            "quant",
            &ch,
        )?,
    });

    // Human-readable report (peptides.tsv + proteins.tsv) + stdout summary.
    let pep_tsv = d("peptides.tsv");
    let prot_tsv = d("proteins.tsv");
    info!(stage = %"report", "run: stage start");
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
    let deeplc_py = cfg.predict_frag.deeplc_python.as_deref();
    let rt_identity = if let Some(model) = grouped_rt_model.as_deref() {
        match model {
            "library" => "imported-library".to_string(),
            m => crate::sidecar::deeplc_identity(deeplc_py, &format!("{m} (per window group)")),
        }
    } else if mh_heads > 0 {
        crate::sidecar::deeplc_identity(deeplc_py, &format!("multihead-{mh_heads}"))
    } else if cfg.rt_im_train.finetune_deeplc {
        crate::sidecar::deeplc_identity(deeplc_py, "finetuned")
    } else if cfg
        .rt_im_train
        .repredicts_library_irt(library_input, deeplc_py.is_some())
    {
        crate::sidecar::deeplc_identity(deeplc_py, "base")
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

    // The input hashes, taken on the background thread started at the top, and the
    // library-input records that share them. A record is inserted only where no later
    // stage recorded that logical name, which is the map the old order produced: the
    // library-input record first, then any adapted precursor table overwriting it.
    let hashes = input_hashes.record(&mut man);
    for r in library_input_records {
        let logical = r.schema.0;
        if man.artifacts.contains_key(logical) {
            continue;
        }
        let rec = match hashes.get(r.input_role) {
            Some(h) => Ok(record_artifact_with_hash(
                logical,
                r.schema,
                &r.path,
                r.rows,
                "library-input",
                &ch,
                h.clone(),
            )),
            // The input could not be hashed on the thread: try once more here. Every stage
            // has read the library by now, so a file that is still unreadable became so
            // during the run, and failing on it would end a finished run without its
            // manifest. It is left out with a warning instead, as `InputHashes::record`
            // leaves out the input itself.
            None => record_artifact(logical, r.schema, &r.path, r.rows, "library-input", &ch),
        };
        match rec {
            Ok(rec) => man.record(rec),
            Err(e) => warn!(
                artifact = logical,
                path = %r.path,
                error = %format!("{e:#}"),
                "run: the library input could not be hashed; its artifact record is left out \
                 of the manifest"
            ),
        }
    }

    let manifest_path = d("manifest.json");
    mumdia_io::json::write_json(&manifest_path, &man)?;

    info!(
        elapsed_ms = t0.elapsed().as_millis(),
        manifest = manifest_path,
        "run: pipeline complete"
    );
    Ok(())
}
