//! The spectral library of a FASTA search, built or restored from the library cache, and
//! `mumdia build-library`, which builds one without spectra.
//!
//! `run` and `run-experiment` used to carry their own copies of the FASTA branch: digest,
//! peptidoforms and predict-frag, looked up in and stored into `predict_frag.library_cache`
//! under a key of everything that determines the library (`library_cache`). [`build`] is
//! that branch, once, so the two orchestrators and `build-library` key a library
//! identically and a library built here is the one a later search finds. Building a
//! library used to need a search, and so a spectra file; the desktop's workaround was a
//! dummy file (issue #233).
//!
//! `build-library` writes the two library tables and their reports, a `manifest.json` and
//! a `library.json` into its output directory, and stores the library in the cache like a
//! FASTA search does. `library.json` records the row counts, the predictor models, the
//! cache key and the two ways to search with the library: through the cache from the same
//! FASTA and configuration, or with `--lib-precursors` / `--lib-fragments` under the
//! `rt_im_train.library_irt` value that reproduces the FASTA search.

use std::time::Instant;

use anyhow::{Context, Result};
use mumdia_core::config::{Config, FragPredictorKind, ImPredictorKind, RtPredictorKind};
use mumdia_core::manifest::Manifest;
use mumdia_core::schema::artifact;
use serde::Serialize;
use tracing::{info, warn};

use crate::library_cache::LibraryCache;
use crate::prestage::PreStageTimer;
use crate::stages::{digest, peptidoforms, predict_frag};

/// The file `build-library` describes its library in.
pub const LIBRARY_JSON: &str = "library.json";

/// Version of the `library.json` layout. Raised when a field changes meaning or goes; a
/// new field does not raise it.
pub const LIBRARY_JSON_FORMAT: u32 = 1;

/// What a FASTA library build needs to know.
pub struct BuildParams<'a> {
    pub cfg: &'a Config,
    pub config_hash: &'a str,
    pub fasta: &'a str,
    /// The library tables are written here, with the digest and peptidoform tables of a
    /// build and the predictors' work directory `sidecar_work/`.
    pub out_dir: &'a str,
    /// The iRT is a placeholder for the multi-head calibration to replace
    /// (`predict_frag.defer_deeplc_to_multihead`, `Config::defers_library_deeplc`).
    pub rt_placeholder: bool,
    /// Whether the spectra the library is for carry 1/K0 (`instrument::input_has_im`),
    /// which decides `predict_frag.im_predictor = auto`.
    pub run_has_im: bool,
    /// `(version, git_sha)` of this engine, for the cache entry.
    pub engine: (&'a str, &'a str),
    /// Look the library up in, and store it into, `predict_frag.library_cache`. False when
    /// the library depends on the spectra (a prefilter before prediction).
    pub use_cache: bool,
    /// The command, for the log: `run`, `run-experiment` or `build-library`.
    pub caller: &'static str,
}

/// Where a library came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Origin {
    /// Digested, expanded and predicted by this command.
    Built,
    /// Restored from `predict_frag.library_cache`.
    LibraryCache,
}

/// A library at its two paths.
pub struct Library {
    pub precursors: String,
    pub fragments: String,
    pub origin: Origin,
    /// The cache entry the library was looked up under, when a cache is configured.
    pub cache: Option<LibraryCache>,
}

/// Rewrites the peptidoform table between the digest and the prediction, recording what it
/// writes in the manifest, and returns the table to predict: `run`'s prefilter before
/// prediction, which converts the spectra and keeps the peptidoforms they support.
pub type PeptidoformFilter<'f> =
    &'f mut dyn FnMut(&str, &mut Manifest, &mut PreStageTimer) -> Result<String>;

/// Build the library of a FASTA search, or restore it from the library cache.
///
/// With `man`, every table is recorded in it as `run` records them; `run-experiment`
/// records none. `filter` needs `man`.
pub fn build(
    p: &BuildParams,
    mut man: Option<&mut Manifest>,
    pre: &mut PreStageTimer,
    filter: Option<PeptidoformFilter<'_>>,
) -> Result<Library> {
    let cfg = p.cfg;
    let ch = p.config_hash;
    let d = |name: &str| format!("{}/{}", p.out_dir, name);
    let lib_p = d("fragment_library_precursors.parquet");
    let lib_f = d("fragment_library_fragments.parquet");
    // `predict_frag.library_cache`: a library stored by an earlier command with the same
    // FASTA, build settings, predictor versions and engine is published here instead of
    // being built (`library_cache`).
    let cache = if p.use_cache {
        LibraryCache::for_config(cfg, p.fasta, p.rt_placeholder, p.run_has_im, p.engine)
    } else {
        None
    };
    if let Some((wp, wf)) = cache.as_ref().and_then(|c| c.restore(&lib_p, &lib_f)) {
        pre.first_stage("library-cache");
        // A reused output directory may still hold an earlier build's digest and
        // peptidoforms, which did not produce this library.
        crate::library_cache::remove_build_intermediates(p.out_dir);
        if let Some(m) = man.as_deref_mut() {
            m.record(wp.record(
                artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
                predict_frag::precursors_schema(&lib_p)?,
                &lib_p,
                "library-cache",
                ch,
            ));
            m.record(wf.record(
                artifact::FRAGMENT_LIBRARY_FRAGMENTS.0,
                artifact::FRAGMENT_LIBRARY_FRAGMENTS,
                &lib_f,
                "library-cache",
                ch,
            ));
        }
        return Ok(Library {
            precursors: lib_p,
            fragments: lib_f,
            origin: Origin::LibraryCache,
            cache,
        });
    }

    let dig = d("peptides.parquet");
    pre.first_stage("digest");
    let w = digest::run_hashed(digest::DigestParams {
        fasta: p.fasta,
        out: &dig,
        cfg: &cfg.digest,
        rng_seed: cfg.rng_seed,
        config_hash: ch,
    })?;
    if let Some(m) = man.as_deref_mut() {
        m.record(w.record(artifact::PEPTIDES.0, artifact::PEPTIDES, &dig, "digest", ch));
    }

    let pf = d("peptidoforms.parquet");
    let w = peptidoforms::run_hashed(peptidoforms::PeptidoformsParams {
        peptides: &dig,
        out: &pf,
        cfg: &cfg.peptidoforms,
        config_hash: ch,
    })?;
    if let Some(m) = man.as_deref_mut() {
        m.record(w.record(
            artifact::PEPTIDOFORMS.0,
            artifact::PEPTIDOFORMS,
            &pf,
            "peptidoforms",
            ch,
        ));
    }

    let pf = match filter {
        Some(f) => {
            let m = man
                .as_deref_mut()
                .context("a peptidoform filter records into the manifest, and none was given")?;
            f(&pf, m, pre)?
        }
        None => pf,
    };

    if cfg.predict_frag.defer_deeplc_to_multihead
        && !p.rt_placeholder
        && p.caller != "build-library"
    {
        info!(
            "{}: predict_frag.defer_deeplc_to_multihead is set, but no multi-head calibration \
             re-predicts this library (it needs rt_predictor = deeplc and a DeepLC \
             interpreter); predicting it with DeepLC as usual",
            p.caller
        );
    }
    let (wp, wf) = predict_frag::run_hashed(predict_frag::PredictFragParams {
        rt_placeholder: p.rt_placeholder,
        peptidoforms: &pf,
        out_precursors: &lib_p,
        out_fragments: &lib_f,
        work_dir: &d("sidecar_work"),
        cfg: &cfg.predict_frag,
        config_hash: ch,
        run_has_im: p.run_has_im,
    })?;
    if let Some(m) = man {
        m.record(wp.record(
            artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
            predict_frag::precursors_schema(&lib_p)?,
            &lib_p,
            "predict-frag",
            ch,
        ));
        m.record(wf.record(
            artifact::FRAGMENT_LIBRARY_FRAGMENTS.0,
            artifact::FRAGMENT_LIBRARY_FRAGMENTS,
            &lib_f,
            "predict-frag",
            ch,
        ));
    }
    match &cache {
        Some(c) => {
            c.store(&lib_p, &lib_f);
            // A stored library can be several GB: bring the caches back within
            // MUMDIA_CACHE_MAX_GB now rather than only at the end of the command.
            crate::cache::enforce_for(cfg);
        }
        None => {
            if let Some(hint) =
                crate::library_cache::reuse_hint(cfg, &lib_p, &lib_f, p.rt_placeholder)
            {
                let what = if p.caller == "run" {
                    "another file"
                } else {
                    "other files"
                };
                info!(
                    "{}: to search {what} against this library without building it again, \
                     pass {hint}, or turn predict_frag.library_cache on (\"auto\" or a \
                     directory) and keep --fasta",
                    p.caller
                );
            }
        }
    }
    Ok(Library {
        precursors: lib_p,
        fragments: lib_f,
        origin: Origin::Built,
        cache,
    })
}

/// `mumdia build-library`.
pub struct BuildLibraryParams<'a> {
    pub config: &'a Config,
    /// Path the config was loaded from, to resolve a relative
    /// `predict_frag.sidecar_script_dir` as `run` does.
    pub config_path: Option<&'a str>,
    pub fasta: &'a str,
    pub out_dir: &'a str,
    /// Build the library for spectra that carry 1/K0 (a timsTOF diaPASEF `.d`): what a
    /// search of such spectra passes as `run_has_im`, so it finds this library in the cache.
    pub ion_mobility: bool,
}

/// The settings that make the library depend on spectra, which `build-library` has none
/// of. `None` when the configuration has none.
pub fn spectra_dependent_setting(cfg: &Config) -> Option<&'static str> {
    if cfg.prescreen.tag_prefilter {
        Some("prescreen.tag_prefilter")
    } else if cfg.prescreen.score_before_prediction {
        Some("prescreen.score_before_prediction")
    } else {
        None
    }
}

/// Refuse a build whose predictor has no interpreter, before the digest runs. The roles
/// were resolved best effort, since the search's rescorer is not this command's concern.
fn require_build_interpreters(cfg: &Config, run_has_im: bool) -> Result<()> {
    let pf = &cfg.predict_frag;
    let mut missing: Vec<&str> = Vec::new();
    match pf.predictor {
        FragPredictorKind::Ms2pip if pf.ms2pip_python.is_none() => {
            missing.push("predict_frag.ms2pip_python (predictor = ms2pip)")
        }
        FragPredictorKind::Peptdeep if pf.peptdeep_python.is_none() => {
            missing.push("predict_frag.peptdeep_python (predictor = peptdeep)")
        }
        _ => {}
    }
    if pf.rt_predictor == RtPredictorKind::Deeplc && pf.deeplc_python.is_none() {
        missing.push("predict_frag.deeplc_python (rt_predictor = deeplc)");
    }
    if pf.im_predictor_for(run_has_im) == ImPredictorKind::Im2deep && pf.im2deep_python.is_none() {
        missing.push("predict_frag.im2deep_python (im_predictor = im2deep)");
    }
    if !missing.is_empty() {
        anyhow::bail!(
            "build-library: no interpreter resolved for {}. Name one in the configuration, set \
             it to \"auto\" with an environment that has the predictor installed, or check \
             with `mumdia doctor --config`",
            missing.join(", ")
        );
    }
    Ok(())
}

/// Build the library of `fasta` without spectra, as a FASTA search would, into `out_dir`.
pub fn run(p: BuildLibraryParams) -> Result<serde_json::Value> {
    let t0 = Instant::now();
    let mut pre = PreStageTimer::start("build-library");
    // Resolved as `run` resolves its configuration, so the predictor fields, which are
    // part of the cache key, hold the same interpreters. Best effort: a rescorer or
    // match-between-runs interpreter is the search's concern, not this command's.
    let mut resolved = p.config.clone();
    resolved.predict_frag.sidecar_script_dir =
        crate::python::resolve_script_dir(&resolved.predict_frag.sidecar_script_dir, p.config_path);
    crate::python::resolve_with(&mut resolved, crate::python::Strictness::BestEffort)?;
    // As a search of spectra with (or without) 1/K0 resolves it, so the configuration hash
    // in the library's reports is that search's.
    resolved.resolve_im_features(p.ion_mobility);
    pre.step("resolve_interpreters");
    let cfg = &resolved;

    if !std::path::Path::new(p.fasta).is_file() {
        anyhow::bail!("--fasta not found or unreadable: {}", p.fasta);
    }
    if let Some(setting) = spectra_dependent_setting(cfg) {
        anyhow::bail!(
            "{setting} filters the peptidoforms on a run's spectra before prediction, so the \
             library it gives depends on those spectra and cannot be built without them. Turn \
             it off to build the library, and search with it set: a search with a prefilter \
             before prediction builds its own library, and does not use the library cache."
        );
    }
    require_build_interpreters(cfg, p.ion_mobility)?;
    // The placeholder iRT of `defer_deeplc_to_multihead` is filled in by a search's
    // multi-head calibration, so a library built here always carries its predictions.
    let has_deeplc = cfg.predict_frag.deeplc_python.is_some();
    if cfg.defers_library_deeplc(false, has_deeplc) {
        warn!(
            "build-library: predict_frag.defer_deeplc_to_multihead is set and ignored here: \
             this library carries DeepLC retention times, not the placeholders a search with \
             that setting builds, so such a search does not find it in the library cache. \
             Search it with --lib-precursors / --lib-fragments, or turn the setting off for \
             the search."
        );
    }
    let im = cfg.predict_frag.im_predictor_for(p.ion_mobility);
    if p.ion_mobility && im == ImPredictorKind::None {
        warn!(
            "build-library: --ion-mobility on, but this library carries no predicted 1/K0: \
             predict_frag.im_predictor is none, or auto without an IM2Deep interpreter. The \
             diaPASEF recipe (configs/examples/diapasef.json) predicts it with IM2Deep. A \
             library without 1/K0 is the same with --ion-mobility on or off."
        );
    } else if !p.ion_mobility && im != ImPredictorKind::None {
        info!(
            "build-library: predict_frag.im_predictor = im2deep predicts 1/K0 whatever \
             --ion-mobility says"
        );
    }
    pre.step("preflight");

    let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
    std::fs::create_dir_all(p.out_dir)
        .with_context(|| format!("creating the output directory {}", p.out_dir))?;
    let mut man = Manifest::new(cfg.canonical_json(), ch.clone());
    info!(provenance = %man.provenance(), "build-library: build provenance");
    let input_hashes = crate::prestage::InputHashes::spawn(
        "build-library",
        vec![("fasta".to_string(), p.fasta.to_string())],
    );
    pre.step("provenance");

    let engine = (man.mumdia_version.clone(), man.git_sha.clone());
    let lib = build(
        &BuildParams {
            cfg,
            config_hash: &ch,
            fasta: p.fasta,
            out_dir: p.out_dir,
            rt_placeholder: false,
            run_has_im: p.ion_mobility,
            engine: (&engine.0, &engine.1),
            use_cache: true,
            caller: "build-library",
        },
        Some(&mut man),
        &mut pre,
        None,
    )?;

    let n_prec = mumdia_io::table::nrows(&lib.precursors)?;
    let n_frag = mumdia_io::table::nrows(&lib.fragments)?;
    let targets = mumdia_io::table::Table::read_cols(&lib.precursors, &["label"])?
        .str_eq("label", "target")?
        .iter()
        .filter(|t| **t)
        .count() as u64;
    let has_im = mumdia_io::table::column_names(&lib.precursors)?
        .iter()
        .any(|c| c == "predicted_im");
    let models = match lib.cache.as_ref() {
        Some(c) => c.models().clone(),
        None => crate::library_cache::model_identities(cfg, false, p.ion_mobility)
            .unwrap_or(serde_json::Value::Null),
    };
    man.model_identities.insert(
        "fragment_predictor".into(),
        format!("{:?}", cfg.predict_frag.predictor),
    );
    man.model_identities.insert(
        "rt_predictor".into(),
        format!("{:?}", cfg.predict_frag.rt_predictor),
    );
    man.model_identities
        .insert("im_predictor".into(), format!("{im:?}"));
    input_hashes.record(&mut man);
    let manifest_path = format!("{}/manifest.json", p.out_dir);
    mumdia_io::json::write_json(&manifest_path, &man)?;

    let library_irt = crate::library_cache::reuse_library_irt(cfg);
    let config_arg = p
        .config_path
        .map(|c| format!(" --config {c}"))
        .unwrap_or_default();
    let cache = lib.cache.as_ref().map(|c| {
        serde_json::json!({
            "dir": c.dir().display().to_string(),
            "key": c.key(),
            "entry": c.entry_dir().display().to_string(),
            "stored": c.entry_dir().join("entry.json").is_file(),
        })
    });
    let info = serde_json::json!({
        "format": LIBRARY_JSON_FORMAT,
        "mumdia_version": man.mumdia_version,
        "git_sha": man.git_sha,
        "fasta": p.fasta,
        "fasta_blake3": man.inputs.get("fasta").map(|i| i.content_hash.clone()),
        "config": p.config_path,
        "config_hash": ch,
        "ion_mobility": p.ion_mobility,
        "predicts_ion_mobility": has_im,
        "origin": lib.origin,
        "precursors": {
            "path": lib.precursors,
            "rows": n_prec,
            "targets": targets,
            "decoys": n_prec - targets,
        },
        "fragments": { "path": lib.fragments, "rows": n_frag },
        "models": models,
        "cache": cache,
        "reuse": {
            "fasta": format!(
                "mumdia run --fasta {fasta} --mzml <spectra> --out-dir <results>{config_arg}: \
                 the same FASTA, configuration and engine find this library in the library \
                 cache{im} and skip digest, peptidoforms and predict-frag",
                fasta = p.fasta,
                im = if p.ion_mobility {
                    " when the spectra are a timsTOF diaPASEF .d"
                } else {
                    " when the spectra carry no ion mobility"
                },
            ),
            "lib_precursors": lib.precursors,
            "lib_fragments": lib.fragments,
            "library_irt": library_irt,
            "library": format!(
                "mumdia run --lib-precursors {} --lib-fragments {} --mzml <spectra> \
                 --out-dir <results>{config_arg}, with rt_im_train.library_irt = \
                 \"{library_irt}\" in the configuration",
                lib.precursors, lib.fragments
            ),
        },
    });
    let info_path = format!("{}/{LIBRARY_JSON}", p.out_dir);
    mumdia_io::json::write_json(&info_path, &info)?;
    info!(
        elapsed_ms = t0.elapsed().as_millis(),
        precursors = n_prec,
        fragments = n_frag,
        origin = ?lib.origin,
        library = %info_path,
        "build-library: complete"
    );
    println!(
        "MuMDIA: library of {n_prec} precursors ({targets} targets, {} decoys) and {n_frag} \
         fragments ({})\n  {}\n  {}\n  search it with --lib-precursors/--lib-fragments and \
         rt_im_train.library_irt = \"{library_irt}\", or with --fasta and the same \
         configuration through the library cache",
        n_prec - targets,
        match lib.origin {
            Origin::Built => "built",
            Origin::LibraryCache => "restored from the library cache",
        },
        lib.precursors,
        lib.fragments,
    );
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_prefilter_before_prediction_makes_the_library_depend_on_spectra() {
        let mut c = Config::default();
        assert_eq!(spectra_dependent_setting(&c), None);
        // The prescreen after calibration filters the library a search reads, not the one
        // it builds.
        c.prescreen.enabled = true;
        assert_eq!(spectra_dependent_setting(&c), None);
        c.prescreen.score_before_prediction = true;
        assert_eq!(
            spectra_dependent_setting(&c),
            Some("prescreen.score_before_prediction")
        );
        c.prescreen.tag_prefilter = true;
        assert_eq!(
            spectra_dependent_setting(&c),
            Some("prescreen.tag_prefilter")
        );
    }

    #[test]
    fn a_predictor_without_an_interpreter_is_refused_before_the_digest() {
        let mut c = Config::default();
        assert!(require_build_interpreters(&c, false).is_ok());
        c.predict_frag.predictor = FragPredictorKind::Ms2pip;
        c.predict_frag.rt_predictor = RtPredictorKind::Deeplc;
        let e = require_build_interpreters(&c, false)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("ms2pip_python") && e.contains("deeplc_python"),
            "{e}"
        );
        // `auto` asks for IM2Deep only for spectra with 1/K0.
        let mut c = Config::default();
        c.predict_frag.im_predictor = ImPredictorKind::Im2deep;
        assert!(require_build_interpreters(&c, false).is_err());
        c.predict_frag.im_predictor = ImPredictorKind::Auto;
        assert!(require_build_interpreters(&c, true).is_ok());
    }
}
