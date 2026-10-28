//! The chain every ungrouped search runs after its seed and its retention-time library:
//! rt-im-train, the optional prescreen, extract, the optional retrace, features and compete.
//!
//! One implementation for every caller. Single-run `run` passes its manifest (so each
//! artifact is recorded) and the MS2 decode its seed lends on; `run-experiment`, the
//! bootstrap pass and the second pass (`run_experiment::finish_run`) pass neither. Before
//! this module the two orchestrators each carried their own copy of the chain, and a stage
//! added to one had to be added to the other by hand.

use anyhow::Result;
use mumdia_core::config::Config;
use mumdia_core::manifest::Manifest;
use mumdia_core::schema::artifact;
use mumdia_core::types::Ms2Scan;
use mumdia_io::record_artifact;
use tracing::info;

use crate::stages::{compete, convert, extract, features, prescreen, retrace, rt_im_train};

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
    pub run_windows: String,
    pub psms_extracted: String,
    pub chromatograms: String,
    pub features: String,
    pub competed: String,
}

/// Record `r` in the manifest when the caller passed one.
fn keep(manifest: &mut Option<&mut Manifest>, r: mumdia_core::manifest::ArtifactRecord) {
    if let Some(m) = manifest.as_deref_mut() {
        m.record(r);
    }
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
            artifact::RUN_WINDOWS,
            &windows,
            "rt-im-train",
            ch,
        ),
    );

    let survivors =
        prescreen::run_if_enabled(cfg, ch, &co.ms2, Some(&co.ms1), lib_p, &windows, out_dir)?;
    if let Some(s) = &survivors {
        keep(
            &mut manifest,
            record_artifact(
                "prescreen_survivors",
                artifact::PRESCAN_SURVIVORS,
                s,
                mumdia_io::table::nrows(s)?,
                "prescreen",
                ch,
            )?,
        );
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
    let ext_cfg =
        crate::stages::ms1cal::extract_cfg(&cfg.extract, seed, &co.ms1, cfg.rt_im_train.q_train)?;
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

    let feats = d("features.parquet");
    info!(stage = %"features", "run: stage start");
    let wf = features::run_hashed(features::FeaturesParams {
        psms: &psms,
        chromatograms: &chrom,
        seed: Some(seed),
        out: &feats,
        out_pin: &d("run.pin"),
        cfg: &cfg.features,
        config_hash: ch,
        // Retrace's `imc_ref_w`, when this pass retraced under `retrace.apex_features`.
        imc_ref_w: raw.is_some() && cfg.retrace.apex_features,
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
        config_hash: ch,
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
        run_windows: windows,
        psms_extracted: psms,
        chromatograms: chrom,
        features: feats,
        competed,
    })
}
