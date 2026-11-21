//! The chain every ungrouped search runs after its seed: the retention-time library
//! ([`adapt_rt_library`]), then rt-im-train, the optional prescreen, extract, the optional
//! retrace, features and compete.
//!
//! One implementation for every caller. Single-run `run` passes its manifest (so each
//! artifact is recorded) and the MS2 decode its seed lends on; `run-experiment`, the
//! bootstrap pass and the second pass (`run_experiment::finish_run`) pass neither. Before
//! this module the two orchestrators each carried their own copy of the chain, and a stage
//! added to one had to be added to the other by hand.

use anyhow::{bail, Result};
use mumdia_core::config::Config;
use mumdia_core::manifest::Manifest;
use mumdia_core::schema::artifact;
use mumdia_core::types::Ms2Scan;
use mumdia_io::record_artifact;
use tracing::info;

use crate::stages::{
    compete, convert, extract, features, prescreen, retrace, rt_bootstrap, rt_im_train,
};

/// The inputs of one run's extract-to-compete chain.
pub struct ChainParams<'a> {
    pub cfg: &'a Config,
    pub config_hash: &'a str,
    /// The precursor table the run searches: the base library or its RT-adapted copy.
    pub lib_p: &'a str,
    pub lib_f: &'a str,
    pub converted: &'a convert::ConvertOutputs,
    /// This run's seed (`seed_psms.parquet`); its `.masscal.json` sits beside it.
    pub seed: &'a str,
    pub out_dir: &'a str,
    /// The seed's MS2 decode, lent on to extract so the run decodes its MS2 once. Owned, so
    /// it is freed as soon as extract returns rather than held through features.
    pub lent_ms2: Option<Vec<Ms2Scan>>,
    /// Records every artifact of the chain when given.
    pub manifest: Option<&'a mut Manifest>,
    /// Whether this pass may retrace (`retrace.enabled` decides whether it does): on for
    /// `run`, `run-experiment` and the second pass, off for the RT bootstrap pass, which
    /// only collects anchors and would pay retrace's 100-150 s and 36-39 GB per HYE run for
    /// nothing it uses.
    pub retrace: bool,
    /// What only the second pass of match-between-runs sets; `Default` everywhere else.
    pub extras: PassExtras<'a>,
}

/// The settings the second pass of match-between-runs gives a run's chain
/// (`run_experiment::search_second_pass`). The default, which every other caller passes,
/// changes nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct PassExtras<'a> {
    /// The 1/K0 window floor the caller wrote into `cfg.rt_im_train.im_window_min`, recorded
    /// in `cal.json`. Set on a run with ion mobility only.
    pub im_window_floor: Option<rt_im_train::ImWindowFloor<'a>>,
    /// Traces recorded beyond the RT window for quantification. Set on every second-pass
    /// run, 3D or 4D, under a fixed quant integration window and `extract.emit_rt_window`.
    pub trace_pad: Option<extract::TracePad>,
}

/// The chain's outputs, by path.
pub struct ChainOutputs {
    pub psms_extracted: String,
    pub chromatograms: String,
    pub competed: String,
}

/// Record `r` in the manifest when the caller passed one.
fn keep(manifest: &mut Option<&mut Manifest>, r: mumdia_core::manifest::ArtifactRecord) {
    if let Some(m) = manifest.as_deref_mut() {
        m.record(r);
    }
}

/// The DeepLC step that rewrote a run's precursor table ([`adapt_rt_library`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RtStep {
    Multihead,
    Finetune,
    /// The base-model re-prediction of an imported library ([`BaseIrt`]).
    Repredict,
}

impl RtStep {
    /// The stage name a manifest records the adapted table under.
    pub fn stage(self) -> &'static str {
        match self {
            RtStep::Multihead => "deeplc-multihead",
            RtStep::Finetune => "deeplc-finetune",
            RtStep::Repredict => "deeplc-repredict",
        }
    }
}

/// An imported library re-predicted with the DeepLC base model, written by the first caller
/// of [`BaseIrt::get`] and reused by every later one, parallel chains included: `run-experiment`
/// makes it once for all its runs.
pub struct BaseIrt {
    path: String,
    made: std::sync::Mutex<bool>,
}

impl BaseIrt {
    pub fn new(path: String) -> Self {
        Self {
            path,
            made: std::sync::Mutex::new(false),
        }
    }

    /// The re-predicted table, written from `lib_in` on the first call (`threads` torch
    /// threads). Every caller passes the same `lib_in`.
    pub fn get(&self, cfg: &Config, lib_in: &str, threads: usize) -> Result<&str> {
        let mut made = self
            .made
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !*made {
            let python = cfg
                .predict_frag
                .deeplc_python
                .as_deref()
                .expect("a base-model re-prediction implies deeplc_python");
            let script = crate::sidecar::resolve_script(
                &cfg.predict_frag.sidecar_script_dir,
                "deeplc_finetune.py",
            );
            info!(stage = %"deeplc-repredict", "run: stage start");
            crate::sidecar::run_deeplc_repredict(
                python,
                &script,
                lib_in,
                &self.path,
                threads,
                cfg.rt_im_train.deeplc_predict_shards,
                crate::cache::projection_dir(cfg).as_ref(),
            )?;
            *made = true;
        }
        Ok(&self.path)
    }
}

/// The inputs of one run's retention-time library adaptation ([`adapt_rt_library`]).
pub struct AdaptParams<'a> {
    pub cfg: &'a Config,
    pub config_hash: &'a str,
    /// The precursor table the run was seeded on: the built or imported library, or the
    /// experiment's base-model re-prediction of it.
    pub lib_p_base: &'a str,
    pub lib_f: &'a str,
    pub converted: &'a convert::ConvertOutputs,
    pub seed: &'a str,
    pub out_dir: &'a str,
    /// Heads of the multi-head calibration (`RtImTrainConfig::multihead_heads`), 0 for none.
    pub mh_heads: usize,
    /// An earlier run's adapted library to reuse (`experiment.rt_library_scope`).
    pub shared_rt_lib: Option<&'a str>,
    /// `lib_p_base` carries placeholder iRT the multi-head calibration must replace in full
    /// (`predict_frag.defer_deeplc_to_multihead`).
    pub irt_placeholder: bool,
    /// Torch threads of the DeepLC workers.
    pub threads: usize,
    /// The base-model re-prediction of an imported library; `None` for a FASTA library.
    pub base_irt: Option<&'a BaseIrt>,
}

/// What [`adapt_rt_library`] chose for one run.
pub struct Adapted {
    /// The precursor table the rest of the run reads.
    pub lib: String,
    /// The table this run produced, which the runs after it may reuse.
    pub produced: Option<String>,
    /// Whether the adaptation searched the run in this process (the bootstrap pass of
    /// `rt_bootstrap`: extract, features and compete on the seed-anchored library), so a
    /// memory window around it measured a search of this run.
    pub searched: bool,
    /// The DeepLC step that wrote `lib`; `None` when the run reads `lib_p_base` as it is or
    /// a shared library.
    pub step: Option<RtStep>,
    /// The search anchors the multi-head calibration was refitted on: rows and path.
    pub refit_anchors: Option<(u64, String)>,
}

/// Choose the precursor table the rest of an ungrouped run reads: an earlier run's adapted
/// library, this run's multi-head calibration or fine-tune of `lib_p_base`, the base-model
/// re-prediction of an imported library, or `lib_p_base` itself. Under
/// `rt_im_train.multihead_anchors = search`, or `auto` on 1/K0 spectra or a seed short of
/// anchors, the multi-head calibration is refitted on the confident targets of a bootstrap
/// search of this run (`rt_bootstrap::refit`). The one copy for `run` and `run-experiment`.
pub fn adapt_rt_library(p: AdaptParams) -> Result<Adapted> {
    let AdaptParams {
        cfg,
        config_hash: ch,
        lib_p_base,
        lib_f,
        converted: co,
        seed,
        out_dir: out,
        mh_heads,
        shared_rt_lib,
        irt_placeholder,
        threads,
        base_irt,
    } = p;
    let as_is = |lib: &str, step: Option<RtStep>| Adapted {
        lib: lib.to_string(),
        produced: None,
        searched: false,
        step,
        refit_anchors: None,
    };
    if let Some(shared) = shared_rt_lib {
        // An earlier run adapted the library and `experiment.rt_library_scope` says to reuse
        // it. The fine-tune and the multi-head calibration both hand back a table with
        // rewritten retention times, and this run fits its own LOESS on top either way.
        return Ok(as_is(shared, None));
    }
    let python = || {
        cfg.predict_frag
            .deeplc_python
            .as_deref()
            .expect("a DeepLC step implies deeplc_python")
    };
    let script = || {
        crate::sidecar::resolve_script(&cfg.predict_frag.sidecar_script_dir, "deeplc_finetune.py")
    };
    // The anchor cut of the multi-head calibration and the fine-tune: `q_train`, relaxed
    // through `anchor_q_ladder` on a seed short of anchors (`rt_im_train::plan_anchors`).
    // Still short at the last rung, the multi-head calibration is skipped rather than handed
    // an empty reference.
    let plan = if mh_heads > 0 || cfg.rt_im_train.finetune_deeplc {
        Some(rt_im_train::plan_anchors(seed, &cfg.rt_im_train)?)
    } else {
        None
    };
    let skipped = plan.is_some_and(|a| mh_heads > 0 && !a.sufficient);
    if let Some(a) = plan.filter(|_| skipped && irt_placeholder) {
        bail!(
            "the library carries placeholder retention times for the multi-head calibration \
             to replace, but the seed search of {out} found only {} target anchors (need {}) \
             even at q {}; re-run without predict_frag.defer_deeplc_to_multihead",
            a.n_anchors,
            cfg.rt_im_train.min_seed_for_calibration,
            a.q_train
        );
    }
    if let Some(plan) = plan.filter(|_| mh_heads > 0 && !skipped) {
        // Under `multihead_anchors = search`, or `auto` on a seed short of anchors, the
        // seed-anchored fit goes to `rt_bootstrap/` and the refitted library takes the usual
        // name; otherwise this is the usual name.
        let has_im = crate::instrument::spectra_has_im(&co.ms2);
        let search = rt_bootstrap::decide(cfg, mh_heads, &plan, has_im, out);
        let seed_lib = rt_bootstrap::seed_library_path(search, out);
        info!(stage = %"deeplc-multihead", "run: stage start");
        crate::sidecar::run_deeplc_multihead(
            python(),
            &script(),
            lib_p_base,
            seed,
            &seed_lib,
            mh_heads,
            plan.q_train,
            cfg.rt_im_train.window_holdout_frac,
            threads,
            cfg.rt_im_train.deeplc_predict_shards,
            crate::cache::projection_dir(cfg).as_ref(),
        )?;
        if irt_placeholder {
            crate::sidecar::require_every_row_repredicted(&seed_lib)?;
        }
        let (lib, refit_anchors) = if search {
            info!(stage = %"rt-bootstrap", "run: stage start");
            let o = rt_bootstrap::refit(&rt_bootstrap::Refit {
                cfg,
                config_hash: ch,
                converted: co,
                lib_p_base,
                lib_f,
                seed,
                out,
                seed_library: &seed_lib,
                mh_heads,
                threads,
                irt_placeholder,
                seed_anchors: plan.n_anchors,
            })?;
            let anchors = match (o.anchors, o.anchors_path) {
                (Some(t), Some(path)) => Some((t.rows as u64, path)),
                _ => None,
            };
            (o.library, anchors)
        } else {
            (seed_lib, None)
        };
        rt_bootstrap::record_auto(cfg, &plan, has_im, search, &lib);
        return Ok(Adapted {
            produced: Some(lib.clone()),
            lib,
            searched: search,
            step: Some(RtStep::Multihead),
            refit_anchors,
        });
    }
    if let Some(plan) = plan.filter(|_| cfg.rt_im_train.finetune_deeplc) {
        let lib = format!("{out}/fragment_library_precursors_ft.parquet");
        info!(stage = %"deeplc-finetune", "run: stage start");
        crate::sidecar::run_deeplc_finetune(
            python(),
            &script(),
            lib_p_base,
            seed,
            &lib,
            cfg.rt_im_train.finetune_epochs,
            cfg.rt_im_train.finetune_patience,
            plan.q_train,
            cfg.rt_im_train.finetune_batch,
            // Held-out window sizing: the fine-tune must exclude the peptides rt-im-train
            // later scores as held-out, else adapter memorization leaks into the "held-out"
            // residuals and the window shrinks back toward in-sample optimism.
            cfg.rt_im_train.window_holdout_frac,
            cfg.rng_seed,
            threads,
            cfg.rt_im_train.deeplc_predict_shards,
        )?;
        return Ok(Adapted {
            produced: Some(lib.clone()),
            lib,
            searched: false,
            step: Some(RtStep::Finetune),
            refit_anchors: None,
        });
    }
    // An imported library takes the base-model retention times when `library_irt` asks for
    // them without a multi-head calibration, and when the calibration was skipped for want
    // of anchors, unless `library_irt = library` keeps the imported ones.
    let has_deeplc = cfg.predict_frag.deeplc_python.is_some();
    let base = (skipped && cfg.deeplc_rt_source(true, has_deeplc))
        || cfg.rt_im_train.repredicts_library_irt(true, has_deeplc);
    if let Some(b) = base_irt.filter(|_| base) {
        return Ok(as_is(
            b.get(cfg, lib_p_base, threads)?,
            Some(RtStep::Repredict),
        ));
    }
    Ok(as_is(lib_p_base, None))
}

/// Run rt-im-train, the optional prescreen, extract, features and compete for one run.
pub fn extract_to_compete(p: ChainParams) -> Result<ChainOutputs> {
    let ChainParams {
        cfg,
        config_hash: ch,
        lib_p,
        lib_f,
        converted: co,
        seed,
        out_dir,
        lent_ms2,
        mut manifest,
        retrace: retrace_allowed,
        extras,
    } = p;
    let d = |name: &str| format!("{out_dir}/{name}");

    // The anchor cut every anchor consumer uses: `q_train`, relaxed through
    // `anchor_q_ladder` on a seed short of anchors (`rt_im_train::plan_anchors`).
    let rt_cfg = rt_im_train::with_anchor_cut(
        &cfg.rt_im_train,
        &rt_im_train::plan_anchors(seed, &cfg.rt_im_train)?,
    );
    let windows = d("run_windows.parquet");
    info!(stage = %"rt-im-train", "run: stage start");
    // In memory as well as on disk: extract reads the same library, so it takes the fitted
    // windows as they are instead of decoding the table written here
    // (`rt_im_train::RtWindows`). The file stays the artifact.
    let (w, fitted_windows) = rt_im_train::run_in_memory(rt_im_train::RtImTrainParams {
        precursor_span: None,
        anchor_irt_from_seed: false,
        seed_psms: seed,
        library_precursors: lib_p,
        out_windows: &windows,
        out_cal: &d("cal.json"),
        cfg: &rt_cfg,
        config_hash: ch,
        im_window_floor: extras.im_window_floor,
        im_anchor_q: Some(cfg.search_seed.fdr_seed),
    })?;
    keep(
        &mut manifest,
        w.record(
            artifact::RUN_WINDOWS.0,
            artifact::run_windows(rt_cfg.run_windows_f32),
            &windows,
            "rt-im-train",
            ch,
        ),
    );
    // A run left without a calibration is extracted over the whole gradient; say so now,
    // with the causes the library points to, rather than only in the end summary.
    crate::stages::rt_health::warn_after_fit(&d("cal.json"), out_dir, lib_f);
    // The Q1 calibration of a scanning-quadrupole run (`rt_im_train.q1_calibrate`), added to
    // the `cal.json` just written. Like retrace, the RT bootstrap pass, which only collects
    // anchors, skips it, unless that pass's extract decides per fragment
    // (`extract.q1_fragments`), which reads the calibration and the tolerances it learns.
    let q1_dec = cfg.features.q1_deconvolved.on();
    if cfg.rt_im_train.q1_calibrate.on()
        && (retrace_allowed || cfg.extract.q1_fragments.on() || q1_dec)
    {
        info!(stage = %"q1-calibrate", "run: stage start");
        let mass_cal = format!("{seed}.masscal.json");
        crate::stages::q1cal::run(crate::stages::q1cal::Q1CalParams {
            seed_psms: seed,
            ms2: &co.ms2,
            library_fragments: lib_f,
            mass_cal: Some(&mass_cal),
            q_max: rt_cfg.q_train,
            frag_tol_fallback_ppm: cfg.extract.frag_tol_ppm,
            out_cal: &d("cal.json"),
            fragment_decision: cfg.extract.q1_fragments.on(),
            deconvolution: q1_dec,
        })?;
    }

    let survivors =
        prescreen::run_if_enabled(cfg, ch, &co.ms2, Some(&co.ms1), lib_p, &windows, out_dir)?;
    // Counted and hashed only when there is a manifest to record them in.
    if let (Some(s), Some(m)) = (&survivors, manifest.as_deref_mut()) {
        m.record(record_artifact(
            "prescreen_survivors",
            artifact::PRESCAN_SURVIVORS,
            s,
            mumdia_io::table::nrows(s)?,
            "prescreen",
            ch,
        )?);
    }

    let psms = d("psms_extracted.parquet");
    let chrom = d("chromatograms.parquet");
    let mass_cal = format!("{seed}.masscal.json");
    // Under `retrace.enabled`, extract writes its centroid traces aside, with their
    // per-point 1/K0, and retrace writes `chromatograms.parquet` from the raw events, so
    // every later stage reads the usual path.
    let raw = retrace::raw_if_enabled(cfg, retrace_allowed, &co.ms2)?;
    let chrom_extract = if raw.is_some() {
        d(retrace::CENTROID_NAME)
    } else {
        chrom.clone()
    };
    info!(stage = %"extract", "run: stage start");
    let mut ext_cfg =
        crate::stages::ms1cal::extract_cfg(&cfg.extract, seed, &co.ms1, cfg.rt_im_train.q_train)?;
    // The ZT Scan keys: the per-fragment Q1 decision and the Q1-deconvolved evidence read the
    // calibration just written to cal.json.
    crate::stages::q1frag::configure_extract(cfg, Some(&d("cal.json")), &mut ext_cfg)?;
    let (wpsm, wchr) = extract::run_hashed(extract::ExtractParams {
        precursor_span: None,
        fragment_offset: None,
        sibling_bands: 1,
        rt_windows: fitted_windows,
        // The MS2 only (`SharedScans::ms1 = None`): extract decodes the MS1 itself,
        // concurrently with its library load and after the library's errors, as it does
        // with nothing lent.
        scans: lent_ms2
            .as_deref()
            .map(|ms2| extract::SharedScans { ms2, ms1: None }),
        ms2: &co.ms2,
        library_precursors: lib_p,
        library_fragments: lib_f,
        run_windows: &windows,
        ms1: Some(&co.ms1),
        mass_cal: Some(&mass_cal),
        out_psms: &psms,
        out_chrom: &chrom_extract,
        restrict_candidates: survivors.as_deref(),
        cfg: &ext_cfg,
        config_hash: ch,
        // The per-point 1/K0 of the traces, for the ion-mobility feature family and for
        // retrace. The callers resolved `features.im_features` once for the run or the
        // experiment (`Config::resolve_im_features`); the spectra's own record answers an
        // unresolved `auto`.
        chrom_im: raw.is_some()
            || cfg
                .features
                .im_features
                .on(crate::instrument::spectra_has_im(&co.ms2)),
        trace_pad: extras.trace_pad,
        // Retrace reads only the axis and keys of the rows it rebuilds, so extract writes
        // those as zeros (`retrace::Skeleton`).
        retrace_skeleton: raw.as_ref().map(|_| retrace::Skeleton {
            ms1: cfg.retrace.ms1,
        }),
    })?;
    drop(lent_ms2);
    // The rebuilt chromatograms and the patched psms_extracted, when retrace runs; the
    // centroid table is deleted once it is read.
    let retraced = retrace::run_if_enabled(
        raw.as_deref(),
        cfg,
        &ext_cfg,
        ch,
        retrace::ChainFiles {
            centroid: &chrom_extract,
            psms_extracted: &psms,
            run_windows: &windows,
            library_precursors: lib_p,
            mass_cal: &mass_cal,
            out: &chrom,
        },
    )?;
    // `output.retain = results` keeps no run_windows in any outcome, and nothing after
    // retrace reads it (extract took the fitted windows in memory), so it goes now rather
    // than at the end of the run: on a large library it is the widest table of the run.
    if cfg.output.retain == mumdia_core::config::Retain::Results {
        crate::retain::remove_early(
            &windows,
            crate::retain::Class::Viewer,
            "run_windows after extract",
        );
    }
    // Each table is recorded once, as the later stages read it.
    let (wpsm, wchr, chrom_stage) = match retraced {
        Some(r) => (r.psms_extracted, r.chromatograms, "retrace"),
        None => (wpsm, wchr, "extract"),
    };
    keep(
        &mut manifest,
        wpsm.record(
            artifact::PSMS_EXTRACTED.0,
            // The version extract wrote (v4 on spectra with 1/K0), read from the table.
            extract::psms_schema(&psms)?,
            &psms,
            "extract",
            ch,
        ),
    );
    keep(
        &mut manifest,
        wchr.record(
            artifact::CHROMATOGRAMS.0,
            // The layout and the 1/K0 list the table holds, read from the table.
            crate::chromatograms::recorded_schema(&chrom)?,
            &chrom,
            chrom_stage,
            ch,
        ),
    );

    // `extract.q1_traces`: the cycle-trace table beside the chromatograms, when this extract
    // wrote one (a table an earlier run left in the directory is not this run's).
    if ext_cfg.q1_traces.on() {
        if let Some(r) = extract::q1_trace_record(&chrom, ch)? {
            keep(&mut manifest, r);
        }
    }

    let feats = d("features.parquet");
    info!(stage = %"features", "run: stage start");
    let wf = features::run_hashed(features::FeaturesParams {
        psms: &psms,
        chromatograms: &chrom,
        seed: Some(seed),
        out: &feats,
        out_pin: &d("run.pin"),
        cfg: &cfg.features,
        // Retrace's `imc_ref_w`, when this pass retraced under `retrace.apex_features`.
        imc_ref_w: raw.is_some() && cfg.retrace.apex_features,
        // The tolerances extract decided with, for `features.q1_consistency`.
        q1_tolerances: ext_cfg.q1_model.as_ref().map(|m| m.tolerances),
    })?;
    keep(
        &mut manifest,
        wf.record(
            artifact::FEATURES.0,
            artifact::FEATURES,
            &feats,
            "features",
            ch,
        ),
    );

    let competed = d("psms_competed.parquet");
    info!(stage = %"compete", "run: stage start");
    let w = compete::run_hashed(compete::CompeteParams {
        features: &feats,
        out: &competed,
        cfg: &cfg.compete,
        features_hash: Some(&wf.content_hash),
    })?;
    keep(
        &mut manifest,
        w.record(
            artifact::PSMS_COMPETED.0,
            artifact::PSMS_COMPETED,
            &competed,
            "compete",
            ch,
        ),
    );

    Ok(ChainOutputs {
        psms_extracted: psms,
        chromatograms: chrom,
        competed,
    })
}
