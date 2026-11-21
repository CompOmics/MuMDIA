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
    /// The library's protein annotation table (`--lib-annotations`); `None` takes the one
    /// beside `lib_precursors` when there is one (`protein_annotations::find`).
    pub lib_annotations: Option<&'a str>,
    pub max_spectra: usize,
    pub top_peaks_ms2: usize,
    /// `[mzml]` when `raw::ensure_mzml` converted it from a vendor input, else empty;
    /// deleted once the spectra are written when `convert.keep_converted_mzml` is false.
    pub converted_mzml: &'a [String],
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

/// Refuse the settings that do not yet support spectra with 1/K0, before any compute.
///
/// The input kind is the raw dispatch's: a diaPASEF `.d` (`raw::Vendor::BrukerTdf`) is the
/// one input that `convert` reads with its ion mobility, and the decision made here is the
/// one conversion will make. Each refused mechanism keys its work on m/z isolation windows
/// only, so on diaPASEF slots that share an m/z range it would merge what extraction keeps
/// apart (docs/35_diapasef.md). The experiment's second pass is supported: it carries a
/// consensus 1/K0 and a 1/K0 window floor of its own (docs/12, "Second pass").
pub(crate) fn refuse_unsupported_on_4d(cfg: &Config, inputs: &[String]) -> Result<()> {
    let Some(tdf) = inputs.iter().find(|m| crate::instrument::input_has_im(m)) else {
        return Ok(());
    };
    if cfg.groups.window_groups > 1 {
        anyhow::bail!(
            "groups.window_groups = {} is not supported on diaPASEF input yet ({tdf}): the \
             isolation-window groups are cut in m/z and do not carry the 1/K0 slots of a \
             diaPASEF run. Set groups.window_groups = 1.",
            cfg.groups.window_groups
        );
    }
    let p = &cfg.prescreen;
    if p.enabled || p.tag_prefilter || p.score_before_prediction {
        anyhow::bail!(
            "the prescreen (prescreen.enabled, tag_prefilter, score_before_prediction) is \
             not supported on diaPASEF input yet ({tdf}): it groups spectra by m/z window \
             only. Turn all three off."
        );
    }
    Ok(())
}

/// Validate inputs and sidecar configuration before any multi-minute compute,
/// so a missing file or a misconfigured rescorer fails immediately with an
/// actionable message.
fn preflight(p: &RunParams, cfg: &Config) -> Result<()> {
    // Inputs depend on the mode: library-input supplies a prebuilt library and
    // skips the FASTA; otherwise the FASTA is digested into the library.
    let mut required: Vec<(&str, &str)> = vec![("--mzml", p.mzml)];
    match (p.lib_precursors, p.lib_fragments) {
        (Some(lp), Some(lf)) => {
            required.push(("--lib-precursors", lp));
            required.push(("--lib-fragments", lf));
        }
        (None, None) => match p.fasta {
            Some(f) => {
                refuse_decoy_free_digest(cfg)?;
                required.push(("--fasta", f));
            }
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
    refuse_unsupported_on_4d(cfg, &[p.mzml.to_string()])?;
    // `python::resolve` ran first and refused every role this configuration requires
    // without a usable interpreter, explicit or discovered (`Role::required_by`); what is
    // left to check here is the roles it only wants.
    warn_without_deeplc(cfg);
    // An imported library under `library_irt = auto` runs DeepLC whenever an interpreter
    // is configured (the multi-head calibration, or the base-model re-prediction), and
    // `resolve` takes an explicit path for a wanted role as given. A wrong one would
    // otherwise fail after convert and the seed.
    if p.lib_precursors.is_some()
        && cfg.rt_im_train.library_irt == mumdia_core::config::LibraryIrt::Auto
    {
        if let Some(exe) = cfg.predict_frag.deeplc_python.as_deref() {
            if crate::python::interpreter_missing(exe) {
                anyhow::bail!(
                    "predict_frag.deeplc_python points at an interpreter that does not \
                     exist: {exe}. An imported library is re-predicted with DeepLC under \
                     rt_im_train.library_irt = auto; fix the path, set it to \"auto\" to \
                     discover an interpreter, or set library_irt = library to keep the \
                     imported iRT."
                );
            }
        }
    }
    Ok(())
}

/// The automatic multi-head calibration with no DeepLC interpreter found: a run that works,
/// at a measured cost, which both orchestrators name before they start.
pub(crate) fn warn_without_deeplc(cfg: &Config) {
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
}

/// `digest.decoy.strategy = none` digests targets only, and a library without decoys is
/// refused when the seed loads it, after the whole prediction. A FASTA search is refused
/// before the digest instead; `digest` and `build-library` still accept it, for a
/// target-only library (docs/02_config_and_data_model.md, "Invariants").
pub(crate) fn refuse_decoy_free_digest(cfg: &Config) -> Result<()> {
    if cfg.digest.decoy.strategy == mumdia_core::config::DecoyStrategy::None {
        anyhow::bail!(
            "digest.decoy.strategy = none builds a library without decoys, and a search \
             needs them for its target-decoy FDR; use \"reverse\" (the default) or \
             \"scramble\""
        );
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
    // `features.im_features = auto` resolves on the one TIMS predicate, decided from the
    // raw dispatch before anything is converted, as the library-build keys are. Every stage
    // then sees `on` or `off`, and the manifest records what ran.
    resolved.resolve_im_features(crate::instrument::input_has_im(p.mzml));
    // The ZT Scan analysis resolves on the input as well: on for a SCIEX ZT Scan DIA run,
    // off for every other run, which then runs exactly as before (`ztscan`, docs/37_ztscan.md,
    // "Automatic for ZT Scan files").
    crate::ztscan::resolve(
        &mut resolved,
        &[p.mzml.to_string()],
        p.lib_precursors.is_some(),
        "run",
    )?;
    crate::stages::retrace::refuse_inputs_without_raw_events(&resolved, &[p.mzml])?;
    pre.step("resolve_interpreters");
    let cfg = &resolved;
    preflight(&p, cfg)?;
    // A FASTA library on the native fallback models is refused before anything runs,
    // unless `predict_frag.allow_native_predictors` says otherwise (`rt_health`).
    crate::stages::rt_health::refuse_native_predictors(cfg, p.lib_precursors.is_none(), "run")?;
    // What the library is predicted with, for the native-model warning now and the causes
    // the end summary names (`rt_health`).
    let native = crate::stages::rt_health::NativePredictions::of_config(
        cfg,
        p.lib_precursors.is_none(),
        crate::instrument::input_has_im(p.mzml),
    );
    if let Some(w) = crate::stages::rt_health::native_predictor_warning(&native) {
        warn!("{w}");
    }
    if let Some(w) =
        crate::stages::peptidoforms::mz_limits_ignored_warning(cfg, p.lib_precursors.is_some())
    {
        warn!("{w}");
    }
    pre.step("preflight");
    let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
    // The output directory must take a write and a rename before any compute: the
    // previous `create_dir_all(..).ok()` discarded the error, and an unwritable folder
    // surfaced minutes later as a stage failure naming one artifact (`storage`).
    crate::storage::prepare_output(p.out_dir, cfg)?;
    let d = |name: &str| format!("{}/{}", p.out_dir, name);
    // The library's protein annotations, which only the report reads (`protein_annotations`):
    // `--lib-annotations`, else the library's own columns, else the table beside
    // `--lib-precursors`, validated here rather than at the end; a library with none is
    // warned about here and at the report. A FASTA search writes its own (`digest`).
    let lib_annotations =
        crate::protein_annotations::resolve_input(p.lib_annotations, p.lib_precursors)?;

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
            (
                "lib_annotations",
                lib_annotations.as_ref().and_then(|s| s.hashed_input()),
            ),
        ]
        .into_iter()
        .filter_map(|(role, path)| path.map(|x| (role.to_string(), x.to_string())))
        .collect(),
    );
    if !p.config.convert.keep_converted_mzml {
        crate::raw::discard_after_convert(p.converted_mzml, input_hashes.progress());
    }
    // The two library-input records, completed once the input hashes are joined: they are
    // the same files, so their content hash IS the input hash, and hashing them here as
    // well read a multi-GB library a second time before the first stage. A later stage
    // that records an adapted precursor table under the same logical name wins, exactly as
    // it did when this record was inserted first and then overwritten (end of `run`).
    let mut library_input_records: Vec<LibraryInputRecord> = Vec::new();
    pre.step("provenance");

    // `prescreen.tag_prefilter`: the spectra are converted first so the database-free tag
    // prefilter can run on the peptidoforms (FASTA mode) or the imported library before any
    // prediction or calibration. It uses no retention time; every later stage does.
    let before_prediction = cfg.prescreen.tag_prefilter || cfg.prescreen.score_before_prediction;
    let spectra_dir = d("spectra");
    let mut early_co: Option<convert::ConvertOutputs> = None;

    // A FASTA build may leave DeepLC to the multi-head calibration, which re-predicts every
    // row before anything reads the iRT (`predict_frag.defer_deeplc_to_multihead`), unless
    // the library cache holds the library with its predictions (`library_build::defers_deeplc`).
    // Whether the spectra carry 1/K0 is decided from the raw dispatch, before conversion.
    let run_has_im = crate::instrument::input_has_im(p.mzml);
    let rt_placeholder = library_build::defers_deeplc(
        cfg,
        p.lib_precursors.is_some(),
        p.fasta,
        run_has_im,
        (&man.mumdia_version, &man.git_sha),
        !before_prediction,
        "run",
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
                schema: crate::stages::predict_frag::precursors_schema(lp)?,
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
            // Build the library from the FASTA digest, or restore it from the library
            // cache (`library_build`, shared with `run-experiment` and `build-library`).
            // preflight guarantees the FASTA is present in this branch.
            let fasta = p.fasta.expect("preflight guarantees --fasta in build mode");
            // Tag prefilter: only the peptidoforms it keeps reach MS2PIP and DeepLC. The
            // spectra are converted for it here, before any prediction.
            let mut prefilter = |pf: &str,
                                 man: &mut Manifest,
                                 pre: &mut crate::prestage::PreStageTimer|
             -> Result<String> {
                pre.first_stage("convert");
                info!(stage = %"convert", "run: stage start");
                let co = convert_and_record(&p, cfg, &spectra_dir, man)?;
                let surv = prescreen::run_before_prediction(cfg, &ch, &co.ms2, pf, p.out_dir)?
                    .expect("a prediction prefilter is set");
                man.record(record_artifact(
                    "prediction_prefilter_survivors",
                    artifact::PRESCAN_SURVIVORS,
                    &surv,
                    mumdia_io::table::nrows(&surv)?,
                    "prescreen",
                    &ch,
                )?);
                let kept = d("peptidoforms_tag_prefiltered.parquet");
                let n = prescreen::filter_rows(pf, "id", &surv, &kept)?;
                info!(
                    kept = n,
                    of = mumdia_io::table::nrows(pf)?,
                    "run: peptidoforms passed to prediction after the prediction prefilter"
                );
                man.record(record_artifact(
                    artifact::PEPTIDOFORMS.0,
                    artifact::PEPTIDOFORMS,
                    &kept,
                    n,
                    "prescreen",
                    &ch,
                )?);
                early_co = Some(co);
                Ok(kept)
            };
            let engine = (man.mumdia_version.clone(), man.git_sha.clone());
            let lib = library_build::build(
                &library_build::BuildParams {
                    cfg,
                    config_hash: &ch,
                    fasta,
                    out_dir: p.out_dir,
                    rt_placeholder,
                    // Whether this run's spectra will carry 1/K0, from the raw dispatch,
                    // before anything is converted: the library-build keys resolve on it
                    // (`predict_frag.im_predictor = auto`), and conversion decides the same
                    // way.
                    run_has_im,
                    engine: (&engine.0, &engine.1),
                    // A tag-prefiltered library depends on this run's spectra, so it is
                    // neither restored from nor stored in the library cache.
                    use_cache: !before_prediction,
                    caller: "run",
                },
                Some(&mut man),
                &mut pre,
                if before_prediction {
                    Some(&mut prefilter as library_build::PeptidoformFilter)
                } else {
                    None
                },
            )?;
            (lib.precursors, lib.fragments)
        }
    };

    // --- per-run artifacts ---
    let co = match early_co {
        Some(co) => co,
        None => {
            pre.first_stage("convert");
            info!(stage = %"convert", "run: stage start");
            convert_and_record(&p, cfg, &spectra_dir, &mut man)?
        }
    };
    if p.lib_precursors.is_none() {
        crate::stages::peptidoforms::warn_if_windows_outside_range(
            &cfg.peptidoforms,
            &co.isolation_windows,
        );
    }

    // Imported library with the tag prefilter: it runs here, after convert and before the seed
    // search and the multi-head calibration, and its survivors become the library every later
    // stage reads (including the fragment-rarity prescreen after calibration, when enabled).
    let (lib_p, lib_f) = match (before_prediction, p.lib_precursors.is_some()) {
        (true, true) => {
            let surv = prescreen::run_before_prediction(cfg, &ch, &co.ms2, &lib_p, p.out_dir)?
                .expect("a prediction prefilter is set");
            man.record(record_artifact(
                "prediction_prefilter_survivors",
                artifact::PRESCAN_SURVIVORS,
                &surv,
                mumdia_io::table::nrows(&surv)?,
                "prescreen",
                &ch,
            )?);
            let sp = d("library_tag_prefiltered_precursors.parquet");
            let sf = d("library_tag_prefiltered_fragments.parquet");
            info!(stage = %"sub-library", "run: stage start");
            // Not pair-linked: the prefilter is label-blind and decides each candidate on its own
            // evidence.
            let st = sub_library::run(sub_library::SubLibraryParams {
                precursors: &lib_p,
                fragments: &lib_f,
                survivors: &surv,
                out_precursors: &sp,
                out_fragments: &sf,
                pair_linked: false,
                overrides: None,
            })?;
            info!(
                library = st.library_precursors,
                kept = st.precursors,
                targets = st.targets,
                decoys = st.decoys,
                "run: the prefiltered library replaces the imported one"
            );
            man.record(record_artifact(
                artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
                crate::stages::predict_frag::precursors_schema(&sp)?,
                &sp,
                st.precursors,
                "prescreen",
                &ch,
            )?);
            man.record(record_artifact(
                artifact::FRAGMENT_LIBRARY_FRAGMENTS.0,
                artifact::FRAGMENT_LIBRARY_FRAGMENTS,
                &sf,
                st.fragments,
                "prescreen",
                &ch,
            )?);
            (sp, sf)
        }
        _ => (lib_p, lib_f),
    };

    // The RT model is decided here for both paths: it names the manifest identity, and the
    // grouped path runs it per band.
    let has_deeplc = cfg.predict_frag.deeplc_python.is_some();
    let mh_heads = cfg.rt_im_train.multihead_heads(
        has_deeplc,
        cfg.deeplc_rt_source(p.lib_precursors.is_some(), has_deeplc),
    );
    rt_bootstrap::warn_if_inert(cfg, mh_heads, "run");
    // What the ungrouped run's RT adaptation ran (`chain::adapt_rt_library`), and whether
    // `rt_bootstrap` refitted the library on the anchors of a bootstrap search, for the
    // manifest's RT identity.
    let mut rt_step: Option<chain::RtStep> = None;
    let mut search_refit = false;
    // Grouped: everything from the seed to the competed table happens one isolation-window
    // group at a time, and the pooled artifacts come back under the same names the
    // ungrouped path writes, so rescore, quant and report below are shared.
    let (seed, lib_p, psms, chrom, competed, grouped_rt_model) = if cfg.groups.window_groups > 1 {
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
            observed_im_q: cfg.rt_im_train.seed_observed_im_q(),
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
        })?;
        man.record(w.record(
            artifact::SEED_PSMS.0,
            search_seed::seed_schema(&seed)?,
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

        // The RT library: a multi-head calibration or fine-tune of the base library on
        // this run's seed, or the base-model re-prediction of an imported library
        // (`chain::adapt_rt_library`, shared with `run-experiment`). The seed is
        // iRT-independent, so it was computed above on the base library.
        let base_irt = p
            .lib_precursors
            .is_some()
            .then(|| chain::BaseIrt::new(d("fragment_library_precursors_deeplc.parquet")));
        let adapted = chain::adapt_rt_library(chain::AdaptParams {
            cfg,
            config_hash: &ch,
            lib_p_base: &lib_p,
            lib_f: &lib_f,
            converted: &co,
            seed: &seed,
            out_dir: p.out_dir,
            mh_heads,
            shared_rt_lib: None,
            irt_placeholder: rt_placeholder,
            threads: rayon::current_num_threads(),
            base_irt: base_irt.as_ref(),
        })?;
        if let Some((rows, path)) = &adapted.refit_anchors {
            search_refit = true;
            man.record(record_artifact(
                artifact::RT_BOOTSTRAP_ANCHORS.0,
                artifact::RT_BOOTSTRAP_ANCHORS,
                path,
                *rows,
                "rt-bootstrap",
                &ch,
            )?);
        }
        // The adapted precursor table is what RT calibration and extraction read, so it
        // replaces the library's manifest entry.
        if let Some(step) = adapted.step {
            man.record(record_artifact(
                artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
                crate::stages::predict_frag::precursors_schema(&adapted.lib)?,
                &adapted.lib,
                mumdia_io::table::nrows(&adapted.lib)?,
                step.stage(),
                &ch,
            )?);
        }
        rt_step = adapted.step;
        if p.lib_precursors.is_some()
            && cfg.rt_im_train.library_irt == mumdia_core::config::LibraryIrt::Auto
            && cfg.predict_frag.deeplc_python.is_none()
        {
            warn!(
                "run: keeping the imported library iRT because no predict_frag.deeplc_python \
                     is configured; configure one to re-predict with DeepLC, or set \
                     rt_im_train.library_irt = library to silence this"
            );
        }
        let lib_p = adapted.lib;

        let out = chain::extract_to_compete(chain::ChainParams {
            cfg,
            config_hash: &ch,
            lib_p: &lib_p,
            lib_f: &lib_f,
            converted: &co,
            seed: &seed,
            out_dir: p.out_dir,
            lent_ms2,
            manifest: Some(&mut man),
            retrace: true,
            extras: Default::default(),
        })?;
        let chrom = vec![quant::ChromTable::whole(&out.chromatograms)];
        (
            seed,
            lib_p,
            out.psms_extracted,
            chrom,
            vec![out.competed],
            None,
        )
    };

    let scored = d("psms_scored.parquet");
    info!(stage = %"rescore", "run: stage start");
    // One table, or a grouped run's band tables (`groups.pool_competed = false`), which are
    // all this run's rows: source 0 for every one.
    let sources = vec![0u32; competed.len()];
    let shadow_windows = vec![d("spectra/isolation_windows.parquet")];
    // The run's extracted rows, for the shadow test's 1/K0 term (`rescore.shadow_im_tolerance`).
    let shadow_psms = vec![psms.clone()];
    let w = rescore::run_hashed(rescore::RescoreParams {
        competed: &competed,
        sources: (competed.len() > 1).then_some(sources.as_slice()),
        out: &scored,
        work_dir: &rescore::sidecar_work_dir(&d("sidecar_work")),
        script_dir: &cfg.predict_frag.sidecar_script_dir,
        cfg: &cfg.rescore,
        config_hash: &ch,
        shadow: Some(
            crate::stages::shadow::ShadowInputs::new(&lib_f, &shadow_windows)
                .with_psms_extracted(&shadow_psms),
        ),
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
            // The report's threshold, so `reported` agrees with the TSVs of the same run.
            q_threshold: cfg.quant.q_threshold,
            run_id: p.out_dir,
            entrapment_substr: "",
        })?;
    }

    // Modification site localisation (`localise.enabled`, experimental): reporting only,
    // read by nothing upstream, so identifications and quantities are the same either way.
    let localisation = if cfg.localise.enabled {
        info!(stage = %"localise", "run: stage start");
        let out = d("localisation.parquet");
        let mass_cal = crate::masscal::json_path(&seed);
        let s = crate::stages::localise::run(crate::stages::localise::LocaliseParams {
            scored: &scored,
            spectra_ms2: &co.ms2,
            mass_cal: Some(&mass_cal),
            psms_extracted: Some(&psms),
            source: None,
            out: &out,
            cfg: &cfg.localise,
            variable_mods: &cfg.peptidoforms.variable_mods,
            fasta: p.fasta,
            fallback_tol_ppm: cfg.extract.frag_tol_ppm,
            config_hash: &ch,
        })?;
        man.record(s.written.record(
            artifact::LOCALISATION.0,
            artifact::LOCALISATION,
            &out,
            "localise",
            &ch,
        ));
        Some(out)
    } else {
        None
    };

    let pep_q = d("peptide_quant.parquet");
    let pg_q = d("protein_group_quant.parquet");
    let frag_q = d("fragment_quant.parquet");
    // `quant.fragment_detail`: the fragments behind each quantity, for a viewer.
    let quant_frags = cfg
        .quant
        .fragment_detail
        .then(|| d("quant_fragments.parquet"));
    if !cfg.quant.normalisation.is_off() {
        warn!(
            "run: quant.normalisation is a cross-run step; a single run has no reference, so \
             no normalised table is written (give several --mzml, or use quant-normalise)"
        );
    }
    info!(stage = %"quant", "run: stage start");
    let wq = quant::run_hashed(quant::QuantParams {
        psms_scored: &scored,
        chromatograms: &chrom,
        out_peptide: Some(&pep_q),
        out_protein: Some(&pg_q),
        out_fragment: Some(&frag_q),
        out_peak_bounds: None,
        out_quant_fragments: quant_frags.as_deref(),
        cfg: &cfg.quant,
        config_hash: &ch,
        cross_run: None,
        export_cross_run: false,
        first_pass_ids: None,
        // `quant.q1_fragments` / `quant.q1_cycle_halfwidth`: the extracted rows' decisions
        // and the sweep of the spectra.
        q1: cfg
            .quant
            .reads_q1_sweep()
            .then_some(crate::stages::q1quant::Q1QuantInputs {
                psms_extracted: Some(&psms),
                spectra_ms2: &co.ms2,
            }),
    })?;
    // The schema quant wrote: v4 under `quant.q1_fragments` (the `q1_quant` column).
    let pep_schema = artifact::peptide_quant_q1(false, cfg.quant.q1_fragments);
    man.record(wq.peptide.expect("out_peptide was set").record(
        pep_schema.0,
        pep_schema,
        &pep_q,
        "quant",
        &ch,
    ));
    man.record(wq.protein.expect("out_protein was set").record(
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
    // Recorded only when written, so a run without `quant.fragment_detail` keeps its
    // manifest.
    if let (Some(path), Some(w)) = (&quant_frags, &wq.quant_fragments) {
        man.record(w.record(
            artifact::QUANT_FRAGMENTS.0,
            artifact::QUANT_FRAGMENTS,
            path,
            "quant",
            &ch,
        ));
    }

    // Human-readable report (peptides.tsv + proteins.tsv) + stdout summary.
    let pep_tsv = d("peptides.tsv");
    let prot_tsv = d("proteins.tsv");
    let prec_tsv = d("precursors.tsv");
    info!(stage = %"report", "run: stage start");
    let loc_tables: Vec<String> = localisation.iter().cloned().collect();
    let sites_tsv = d("sites.tsv");
    // The annotations as a table in the output directory, so a later `mumdia report` finds
    // it beside the scored table, recorded only when there is one. Without one the TSVs'
    // three annotation columns are empty, which is said again here and recorded in the
    // manifest (`protein_annotations_missing`).
    let annotations = match crate::protein_annotations::search_source(
        lib_annotations.as_ref(),
        p.lib_precursors,
        p.out_dir,
    ) {
        Some(src) => Some(
            crate::protein_annotations::publish_recorded(&src, p.out_dir, &ch, Some(&mut man))?.0,
        ),
        None => {
            let lib = p.lib_precursors.unwrap_or_default();
            warn!(
                "report: {}",
                crate::protein_annotations::missing_warning(lib)
            );
            man.protein_annotations_missing = Some(crate::protein_annotations::missing_gap(lib));
            None
        }
    };
    let loc = localisation.as_ref().map(|_| report::Localisation {
        tables: &loc_tables,
        experiment: None,
        fasta: p.fasta,
        out_sites: &sites_tsv,
    });
    let counts = report::run(
        report::ReportParams {
            scored: &scored,
            peptide_quant: Some(&pep_q),
            protein_quant: Some(&pg_q),
            out_peptides: &pep_tsv,
            out_proteins: &prot_tsv,
            out_precursors: Some(&prec_tsv),
            q_threshold: cfg.quant.q_threshold,
            protein_annotations: annotations.as_deref(),
        },
        loc.as_ref(),
    )?;
    let (n_pep, n_prot) = (counts.peptide_rows, counts.protein_groups);
    println!(
        "MuMDIA: {n_pep} precursor rows, {n_prot} protein groups at peptide/PG q <= {} (rescorer used: {})\n  {}\n  {}",
        cfg.quant.q_threshold, actual_rescorer, pep_tsv, prot_tsv
    );
    if let Some(n) = counts.precursor_rows {
        println!(
            "MuMDIA: {n} precursors at precursor_q <= {}\n  {prec_tsv}",
            cfg.quant.q_threshold
        );
    }

    // Model identities reflect the path that produced the downstream artifacts: the RT
    // model is the step that actually ran, not the one the configuration planned (a
    // multi-head calibration skipped for want of anchors is not one).
    let library_input = p.lib_precursors.is_some();
    let deeplc_py = cfg.predict_frag.deeplc_python.as_deref();
    let library_rt = || {
        if library_input {
            "imported-library".to_string()
        } else {
            format!("{:?}", cfg.predict_frag.rt_predictor)
        }
    };
    let rt_identity = match (grouped_rt_model.as_deref(), rt_step) {
        (Some("library"), _) => library_rt(),
        (Some(m), _) => {
            crate::sidecar::deeplc_identity(deeplc_py, &format!("{m} (per window group)"))
        }
        (None, Some(chain::RtStep::Multihead)) if search_refit => crate::sidecar::deeplc_identity(
            deeplc_py,
            &format!("multihead-{mh_heads}-search-anchors"),
        ),
        (None, Some(chain::RtStep::Multihead)) => {
            crate::sidecar::deeplc_identity(deeplc_py, &format!("multihead-{mh_heads}"))
        }
        (None, Some(chain::RtStep::Finetune)) => {
            crate::sidecar::deeplc_identity(deeplc_py, "finetuned")
        }
        (None, Some(chain::RtStep::Repredict)) => {
            crate::sidecar::deeplc_identity(deeplc_py, "base")
        }
        (None, None) => library_rt(),
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
        features::feature_schema_id(&features::active_features_of(cfg, false)),
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

    // A run without a retention-time calibration, recorded only when it happened so a
    // calibrated run's manifest keeps its bytes, and said once more at the end.
    let run_name = std::path::Path::new(p.mzml)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.mzml.to_string());
    if let Some(gap) = crate::stages::rt_health::gap_of(&d("cal.json"), &run_name, "run") {
        man.rt_calibration_missing.push(gap);
    }
    crate::stages::rt_health::warn_summary(&man.rt_calibration_missing, 1, &native);

    // `output.retain`: what the tier removes, recorded in the manifest before anything goes,
    // so the manifest names every file the directory had.
    let retention = crate::retain::plan(
        std::path::Path::new(p.out_dir),
        cfg.output.retain,
        &crate::retain::Layout::default(),
        &crate::retain::hashes_by_rel(
            p.out_dir,
            man.artifacts
                .values()
                .map(|a| (a.path.as_str(), a.content_hash.as_str())),
        ),
    )?;
    man.retention = retention.clone();
    let manifest_path = d("manifest.json");
    mumdia_io::json::write_json(&manifest_path, &man)?;
    if let Some(r) = &retention {
        crate::retain::remove(std::path::Path::new(p.out_dir), r)?;
    }

    info!(
        elapsed_ms = t0.elapsed().as_millis(),
        manifest = manifest_path,
        "run: pipeline complete"
    );
    // The DeepLC projection cache may have grown during the run.
    crate::cache::enforce_for(cfg);
    Ok(())
}

/// Convert the run's spectra and record the four artifacts in the manifest.
fn convert_and_record(
    p: &RunParams,
    cfg: &Config,
    spectra_dir: &str,
    man: &mut Manifest,
) -> Result<convert::ConvertOutputs> {
    // The caps in the provenance key, as the standalone `convert` subcommand has them: the
    // bare config hash once made two runs with different caps record identical provenance.
    let convert_hash =
        convert::provenance_hash(&cfg.canonical_json(), p.max_spectra, p.top_peaks_ms2, 0);
    let co = convert::run(convert::ConvertParams {
        mzml: p.mzml,
        out_dir: spectra_dir,
        max_spectra: p.max_spectra,
        top_peaks_ms2: p.top_peaks_ms2,
        top_peaks_ms1: 0,
        config_hash: &convert_hash,
        tdf: convert::TdfParams::from_config(&cfg.convert),
        isolation_half_width: cfg.convert.isolation_half_width,
    })?;
    crate::raw::discard_converted(p.mzml);
    for (name, schema, path, hash) in [
        ("spectra_ms1", co.schemas.ms1, &co.ms1, &co.hashes.ms1),
        ("spectra_ms2", co.schemas.ms2, &co.ms2, &co.hashes.ms2),
        (
            "isolation_windows",
            co.schemas.isolation_windows,
            &co.isolation_windows,
            &co.hashes.isolation_windows,
        ),
        (
            "ms2_to_ms1",
            co.schemas.ms2_to_ms1,
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

    Ok(co)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grouped runs and the prescreen are refused on a diaPASEF `.d`, each with a message
    /// naming the setting to change; an mzML is never refused. The defaults pass, the
    /// experiment's second pass included.
    #[test]
    fn diapasef_input_refuses_what_is_keyed_on_mz_windows_only() {
        let d = std::env::temp_dir().join(format!("mumdia-run-4d-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let dia = d.join("dia.d");
        crate::raw::tdf_fixture(&dia, &[0, 9], "2", true);
        let dia = vec![dia.to_string_lossy().into_owned()];
        let mzml = vec!["run.mzML".to_string()];
        let base = Config::default();

        // The defaults pass, `mbr.strategy = second_pass` (an experiment's default) included.
        assert_eq!(
            base.mbr.strategy,
            mumdia_core::config::MbrStrategy::SecondPass
        );
        assert!(refuse_unsupported_on_4d(&base, &dia).is_ok());
        let none = base.clone();

        let mut grouped = none.clone();
        grouped.groups.window_groups = 4;
        let e = refuse_unsupported_on_4d(&grouped, &dia)
            .unwrap_err()
            .to_string();
        assert!(e.contains("groups.window_groups = 1"), "{e}");

        for set in [
            |c: &mut Config| c.prescreen.enabled = true,
            |c: &mut Config| c.prescreen.tag_prefilter = true,
            |c: &mut Config| c.prescreen.score_before_prediction = true,
        ] {
            let mut c = none.clone();
            set(&mut c);
            let e = refuse_unsupported_on_4d(&c, &dia).unwrap_err().to_string();
            assert!(e.contains("prescreen"), "{e}");
            assert!(refuse_unsupported_on_4d(&c, &mzml).is_ok());
        }
        assert!(refuse_unsupported_on_4d(&grouped, &mzml).is_ok());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Two configurations that used to fail only mid-run are refused by the preflight: a
    /// FASTA search with `digest.decoy.strategy = none`, and an imported-library search under
    /// `library_irt = auto` whose DeepLC interpreter path does not exist. Each is refused only
    /// where its search would fail.
    #[test]
    fn preflight_refuses_what_would_fail_mid_run() {
        let d = std::env::temp_dir().join(format!("mumdia-run-preflight-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let file = |name: &str| {
            let p = d.join(name);
            std::fs::write(&p, b"").unwrap();
            p.to_string_lossy().into_owned()
        };
        let (mzml, fasta) = (file("run.mzML"), file("db.fasta"));
        let (lp, lf) = (file("lib_p.parquet"), file("lib_f.parquet"));
        let out = d.to_string_lossy().into_owned();
        let check = |cfg: &Config, library: bool| {
            preflight(
                &RunParams {
                    config: cfg,
                    config_path: None,
                    fasta: (!library).then_some(fasta.as_str()),
                    mzml: &mzml,
                    out_dir: &out,
                    lib_precursors: library.then_some(lp.as_str()),
                    lib_fragments: library.then_some(lf.as_str()),
                    lib_annotations: None,
                    max_spectra: 0,
                    top_peaks_ms2: 0,
                    converted_mzml: &[],
                },
                cfg,
            )
        };
        let base = Config::default();
        assert!(check(&base, false).is_ok());
        assert!(check(&base, true).is_ok());

        let mut no_decoys = base.clone();
        no_decoys.digest.decoy.strategy = mumdia_core::config::DecoyStrategy::None;
        let e = check(&no_decoys, false).unwrap_err().to_string();
        assert!(e.contains("digest.decoy.strategy"), "{e}");
        assert!(
            check(&no_decoys, true).is_ok(),
            "an imported library has its own decoys"
        );

        let mut stale = base.clone();
        stale.predict_frag.deeplc_python = Some(d.join("no-such-python").to_string_lossy().into());
        let e = check(&stale, true).unwrap_err().to_string();
        assert!(e.contains("predict_frag.deeplc_python"), "{e}");
        assert!(
            check(&stale, false).is_ok(),
            "a FASTA run does not re-predict a library"
        );
        let mut on_path = base.clone();
        on_path.predict_frag.deeplc_python = Some("python3".into());
        assert!(check(&on_path, true).is_ok(), "found on PATH by discovery");
        stale.rt_im_train.library_irt = mumdia_core::config::LibraryIrt::Library;
        assert!(
            check(&stale, true).is_ok(),
            "library_irt = library starts no DeepLC"
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}
