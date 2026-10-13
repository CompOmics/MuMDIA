//! The chain every ungrouped search runs after its seed and its retention-time library:
//! rt-im-train, the optional prescreen, extract, features and compete.
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

use crate::stages::{compete, convert, extract, features, prescreen, rt_im_train};

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
}

/// The chain's outputs, by path.
pub struct ChainOutputs {
    pub run_windows: String,
    pub psms_extracted: String,
    pub chromatograms: String,
    pub features: String,
    pub competed: String,
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
    } = p;
    let d = |name: &str| format!("{out_dir}/{name}");
    let mut record = |r: mumdia_core::manifest::ArtifactRecord| {
        if let Some(m) = manifest.as_deref_mut() {
            m.record(r);
        }
    };

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
    })?;
    record(w.record(
        artifact::RUN_WINDOWS.0,
        artifact::RUN_WINDOWS,
        &windows,
        "rt-im-train",
        ch,
    ));

    let survivors =
        prescreen::run_if_enabled(cfg, ch, &co.ms2, Some(&co.ms1), lib_p, &windows, out_dir)?;
    if let Some(s) = &survivors {
        record(record_artifact(
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
        mass_cal: Some(&format!("{seed}.masscal.json")),
        out_psms: &psms,
        out_chrom: &chrom,
        restrict_candidates: survivors.as_deref(),
        cfg: &ext_cfg,
        config_hash: ch,
    })?;
    drop(lent_ms2);
    record(wpsm.record(
        artifact::PSMS_EXTRACTED.0,
        artifact::PSMS_EXTRACTED,
        &psms,
        "extract",
        ch,
    ));
    record(wchr.record(
        artifact::CHROMATOGRAMS.0,
        artifact::chromatograms(cfg.extract.chromatogram_schema),
        &chrom,
        "extract",
        ch,
    ));

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
    })?;
    record(wf.record(
        artifact::FEATURES.0,
        artifact::FEATURES,
        &feats,
        "features",
        ch,
    ));

    let competed = d("psms_competed.parquet");
    info!(stage = %"compete", "run: stage start");
    let w = compete::run_hashed(compete::CompeteParams {
        features: &feats,
        out: &competed,
        cfg: &cfg.compete,
        config_hash: ch,
        features_hash: Some(&wf.content_hash),
    })?;
    record(w.record(
        artifact::PSMS_COMPETED.0,
        artifact::PSMS_COMPETED,
        &competed,
        "compete",
        ch,
    ));

    Ok(ChainOutputs {
        run_windows: windows,
        psms_extracted: psms,
        chromatograms: chrom,
        features: feats,
        competed,
    })
}
