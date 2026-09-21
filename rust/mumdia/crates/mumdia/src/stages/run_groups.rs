//! The grouped middle of a run (`groups.window_groups > 1`): from the converted spectra to
//! the pooled competed table, one isolation-window group at a time.
//!
//! `run` converts the file, then hands over here when groups are configured, and takes the
//! pooled artifacts back for rescore, quant and report, which see exactly what an ungrouped
//! run produces. Per group: the band of the library is written as a precursor table of its
//! own (local ids), seeded, given a retention-time model and windows, extracted, featured
//! and competed, in a directory `groups/gNN/` under the run. Between the seeds and the RT
//! model the seeds are pooled (`groups.calibration = global`), so every group's calibration
//! is fitted on the whole run's anchors; `per_group` calibrates each group on its own.
//! Groups run one after the other in this process (`groups.parallel` is reserved), so the
//! resident set is one band's library, one band's hits and one band's accepted rows.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use mumdia_core::config::{Config, GroupCalibration};
use mumdia_core::manifest::Manifest;
use mumdia_core::schema::artifact;
use mumdia_io::record_artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::TableFile;
use serde_json::json;
use tracing::info;

use super::{compete, convert, extract, features, pool, rt_im_train, search_seed, seed_pool};
use crate::groups;
use crate::index::Library;

pub struct GroupRun<'a> {
    pub cfg: &'a Config,
    pub config_hash: &'a str,
    pub converted: &'a convert::ConvertOutputs,
    /// The library as given (or built): the whole precursor and fragment tables.
    pub lib_precursors: &'a str,
    pub lib_fragments: &'a str,
    pub out_dir: &'a str,
    /// Where to record the band and pooled artifacts. `None` under `run-experiment`,
    /// which keeps one experiment-level manifest instead of one per run.
    pub man: Option<&'a mut Manifest>,
    /// A previous run's `groups/` directory, whose per-band precursor tables already carry
    /// adapted retention times (`experiment.rt_library_scope = first_run_only`). Set, no
    /// band re-predicts: each takes that run's table for its band and fits its own per-run
    /// LOESS on top, which is what the ungrouped path does with a shared library.
    pub shared_bands: Option<&'a str>,
    /// Multi-head calibration heads resolved by the caller (0 = off).
    pub mh_heads: usize,
    pub library_input: bool,
}

/// Paths the pooled stages continue with.
pub struct Pooled {
    pub seed: String,
    pub psms: String,
    pub chromatograms: String,
    pub features: String,
    pub competed: String,
    /// One RT-model identity string for the manifest, e.g. `multihead-80` per group.
    pub rt_model: String,
}

fn windows_of(path: &str) -> Result<Vec<(f64, f64)>> {
    let t = TableFile::open(path)?;
    let lo = t.f64("lower")?;
    let hi = t.f64("upper")?;
    Ok(lo.into_iter().zip(hi).collect())
}

/// The file name a band's adapted precursor table takes, so a later run can find it.
fn band_lib_name(rt_model: &str) -> &'static str {
    if rt_model.starts_with("multihead") {
        "lib_precursors_multihead.parquet"
    } else if rt_model == "finetuned" {
        "lib_precursors_ft.parquet"
    } else {
        "lib_precursors_deeplc.parquet"
    }
}

/// Record an artifact when this run keeps a manifest of its own.
fn record_opt(man: Option<&mut Manifest>, rec: mumdia_core::manifest::ArtifactRecord) {
    if let Some(m) = man {
        m.record(rec);
    }
}

pub fn run(mut g: GroupRun) -> Result<Pooled> {
    let t0 = Instant::now();
    let cfg = g.cfg;
    let ch = g.config_hash;
    // The multi-head ridge and the base-model re-prediction are deterministic in the
    // anchors, so fitting them per band on the pooled anchors gives every band the same
    // model. A fine-tune is not: trained per band it would give each band a different
    // model, and cost the training once per band. Until the fine-tune can be trained once
    // and applied per band, a grouped run refuses it; fine-tune the library once beforehand
    // (docs/08, once-per-library) and search that table instead.
    if cfg.rt_im_train.finetune_deeplc {
        bail!(
            "rt_im_train.finetune_deeplc is not supported with groups.window_groups > 1: the              fine-tune would be trained separately in every group (non-deterministic, and the              training cost once per group). Fine-tune the library once beforehand and search              the fine-tuned table, or leave the default multi-head calibration on"
        );
    }
    let d = |name: &str| format!("{}/{}", g.out_dir, name);
    let gd = |i: usize, name: &str| format!("{}/groups/g{i:02}/{name}", g.out_dir);

    // --- plan
    let windows = windows_of(&g.converted.isolation_windows)?;
    let stats = TableFile::open(g.lib_precursors)?.row_group_stats("precursor_mz")?;
    let plan = groups::plan(&windows, &stats, cfg.groups.window_groups)?;
    info!(
        groups = plan.bands.len(),
        windows = plan.windows.len(),
        est_unselectable = plan.est_unselectable as u64,
        est_duplicated = plan.est_duplicated as u64,
        calibration = ?cfg.groups.calibration,
        "groups: plan"
    );
    for b in &plan.bands {
        info!(
            group = b.index,
            mz_lo = b.mz_lo,
            mz_hi = b.mz_hi,
            windows = b.windows.len(),
            est_precursors = b.est_precursors as u64,
            "groups: band"
        );
    }
    std::fs::create_dir_all(d("groups"))?;
    mumdia_io::json::write_json(
        &d("groups/plan.json"),
        &serde_json::json!({
            "window_groups": cfg.groups.window_groups,
            "calibration": format!("{:?}", cfg.groups.calibration),
            "est_unselectable": plan.est_unselectable,
            "est_duplicated": plan.est_duplicated,
            "bands": plan.bands.iter().map(|b| serde_json::json!({
                "index": b.index, "mz_lo": b.mz_lo, "mz_hi": b.mz_hi,
                "windows": b.windows, "est_precursors": b.est_precursors,
            })).collect::<Vec<_>>(),
        }),
    )?;

    // --- band files and seeds
    struct Band {
        index: usize,
        offset: u32,
        n: usize,
        prec: String,
        seed: String,
        /// The band's rows of the pooled seed (local ids, pooled q).
        seed_view: String,
    }
    let mut bands: Vec<Band> = Vec::new();
    for b in &plan.bands {
        let (first, n) = Library::precursor_row_span(g.lib_precursors, b.mz_lo, b.mz_hi)?;
        if n == 0 {
            info!(
                group = b.index,
                "groups: band selects no precursor; skipped"
            );
            continue;
        }
        std::fs::create_dir_all(gd(b.index, ""))?;
        let prec = gd(b.index, "lib_precursors.parquet");
        info!(stage = %"band-slice", group = b.index, rows = n, "run: stage start");
        groups::write_band_slice(g.lib_precursors, first, n, &prec)?;
        let seed = gd(b.index, "seed_psms.parquet");
        info!(stage = %"search-seed", group = b.index, "run: stage start");
        let rows = search_seed::run(search_seed::SearchSeedParams {
            ms2: &g.converted.ms2,
            library_precursors: &prec,
            library_fragments: g.lib_fragments,
            out: &seed,
            cfg: &cfg.search_seed,
            bucket_size: cfg.extract.bucket_size,
            config_hash: ch,
            fragment_offset: Some(first as u32),
        })?;
        record_opt(
            g.man.as_deref_mut(),
            record_artifact(
                &format!("{}[g{:02}]", artifact::SEED_PSMS.0, b.index),
                artifact::SEED_PSMS,
                &seed,
                rows,
                "search-seed",
                ch,
            )?,
        );
        bands.push(Band {
            index: b.index,
            offset: first as u32,
            n,
            prec,
            seed,
            seed_view: gd(b.index, "seed_psms_pooled.parquet"),
        });
    }
    if bands.is_empty() {
        bail!("groups: no band selects any precursor of the library");
    }

    // --- pooled seed: library-wide ids, one q scale, one mass calibration
    let pooled_seed = d("seed_psms.parquet");
    let seeds: Vec<seed_pool::BandSeed> = bands
        .iter()
        .map(|b| seed_pool::BandSeed {
            path: b.seed.clone(),
            offset: b.offset,
            rows: b.n as u32,
            view: b.seed_view.clone(),
        })
        .collect();
    let masscals: Vec<String> = bands
        .iter()
        .map(|b| format!("{}.masscal.json", b.seed))
        .collect();
    info!(stage = %"seed-pool", "run: stage start");
    let n = seed_pool::run(seed_pool::SeedPoolParams {
        seeds: &seeds,
        masscals: &masscals,
        out: &pooled_seed,
    })?;
    record_opt(
        g.man.as_deref_mut(),
        record_artifact(
            artifact::SEED_PSMS.0,
            artifact::SEED_PSMS,
            &pooled_seed,
            n,
            "seed-pool",
            ch,
        )?,
    );
    let global = cfg.groups.calibration == GroupCalibration::Global;

    // --- RT model per band, against the pooled or the band's own anchors
    let has_deeplc = cfg.predict_frag.deeplc_python.is_some();
    let python = cfg.predict_frag.deeplc_python.as_deref();
    let script =
        crate::sidecar::resolve_script(&cfg.predict_frag.sidecar_script_dir, "deeplc_finetune.py");
    let repredict = cfg
        .rt_im_train
        .repredicts_library_irt(g.library_input, has_deeplc);
    let rt_model = if g.mh_heads > 0 {
        format!("multihead-{}", g.mh_heads)
    } else if cfg.rt_im_train.finetune_deeplc {
        "finetuned".to_string()
    } else if repredict {
        "base".to_string()
    } else {
        "library".to_string()
    };
    let mut repredicted: Vec<(String, u32)> = Vec::new();
    if let Some(shared) = g.shared_bands {
        // Take a previous run's adapted bands wholesale: same ids, same row order, which is
        // what `fragment_offset` and the per-band seed views key on. Only the retention
        // times inside differ from the raw library, and this run still fits its own LOESS.
        let name = band_lib_name(&rt_model);
        for b in &mut bands {
            let from = format!("{shared}/g{:02}/{name}", b.index);
            if !std::path::Path::new(&from).exists() {
                bail!(
                    "groups: {from} is missing, so the shared bands do not match this run: \
                     reusing them needs the same group plan and the same retention-time model"
                );
            }
            repredicted.push((from.clone(), b.offset));
            b.prec = from;
        }
        info!(
            groups = bands.len(),
            model = %rt_model,
            source = %shared,
            "groups: reusing a previous run's adapted bands"
        );
    }
    for b in bands.iter_mut().filter(|_| g.shared_bands.is_none()) {
        let anchors = if global { &pooled_seed } else { &b.seed };
        let out = if g.mh_heads > 0 {
            let out = gd(b.index, "lib_precursors_multihead.parquet");
            info!(stage = %"deeplc-multihead", group = b.index, "run: stage start");
            crate::sidecar::run_deeplc_multihead(
                python.expect("mh_heads is 0 unless an interpreter resolved"),
                &script,
                &b.prec,
                anchors,
                &out,
                g.mh_heads,
                cfg.rt_im_train.q_train,
                cfg.rt_im_train.window_holdout_frac,
                rayon::current_num_threads(),
            )?;
            Some(out)
        } else if cfg.rt_im_train.finetune_deeplc {
            let out = gd(b.index, "lib_precursors_ft.parquet");
            info!(stage = %"deeplc-finetune", group = b.index, "run: stage start");
            crate::sidecar::run_deeplc_finetune(
                python.expect("preflight guarantees deeplc_python when finetune_deeplc is set"),
                &script,
                &b.prec,
                anchors,
                &out,
                cfg.rt_im_train.finetune_epochs,
                cfg.rt_im_train.finetune_patience,
                cfg.rt_im_train.q_train,
                cfg.rt_im_train.finetune_batch,
                cfg.rt_im_train.window_holdout_frac,
                cfg.rng_seed,
            )?;
            Some(out)
        } else if repredict {
            let out = gd(b.index, "lib_precursors_deeplc.parquet");
            info!(stage = %"deeplc-repredict", group = b.index, "run: stage start");
            crate::sidecar::run_deeplc_repredict(
                python.expect("repredicts_library_irt implies deeplc_python"),
                &script,
                &b.prec,
                &out,
                rayon::current_num_threads(),
            )?;
            Some(out)
        } else {
            None
        };
        if let Some(out) = out {
            let rows = mumdia_io::table::nrows(&out)?;
            record_opt(
                g.man.as_deref_mut(),
                record_artifact(
                    &format!(
                        "{}[g{:02}]",
                        artifact::FRAGMENT_LIBRARY_PRECURSORS.0,
                        b.index
                    ),
                    artifact::FRAGMENT_LIBRARY_PRECURSORS,
                    &out,
                    rows,
                    &rt_model,
                    ch,
                )?,
            );
            repredicted.push((out.clone(), b.offset));
            b.prec = out;
        }
    }
    // The pooled anchors carry the library's iRT as it was when seeded; after a
    // re-prediction each anchor takes its band's new value, so rt-im-train can read the
    // anchors' iRT from the seed while holding one band's table.
    let anchors_for_windows = if global && !repredicted.is_empty() {
        let refreshed = d("seed_psms_calibrated.parquet");
        info!(stage = %"seed-refresh", "run: stage start");
        seed_pool::refresh_irt(&pooled_seed, &refreshed, &repredicted)?;
        Some(refreshed)
    } else {
        None
    };

    // --- windows, extract, features, compete per band
    let mut arts: Vec<pool::BandArtifacts> = Vec::new();
    let mut cals: Vec<(usize, String)> = Vec::new();
    for b in &bands {
        let windows = gd(b.index, "run_windows.parquet");
        let cal = gd(b.index, "cal.json");
        cals.push((b.index, cal.clone()));
        let (seed_for_windows, from_seed) = match (&anchors_for_windows, global) {
            (Some(refreshed), _) => (refreshed.clone(), true),
            (None, true) => (pooled_seed.clone(), true),
            (None, false) => (b.seed.clone(), false),
        };
        info!(stage = %"rt-im-train", group = b.index, "run: stage start");
        let rows = rt_im_train::run(rt_im_train::RtImTrainParams {
            seed_psms: &seed_for_windows,
            library_precursors: &b.prec,
            out_windows: &windows,
            out_cal: &cal,
            cfg: &cfg.rt_im_train,
            config_hash: ch,
            anchor_irt_from_seed: from_seed,
        })?;
        record_opt(
            g.man.as_deref_mut(),
            record_artifact(
                &format!("{}[g{:02}]", artifact::RUN_WINDOWS.0, b.index),
                artifact::RUN_WINDOWS,
                &windows,
                rows,
                "rt-im-train",
                ch,
            )?,
        );
        let mass_cal = if global {
            format!("{pooled_seed}.masscal.json")
        } else {
            format!("{}.masscal.json", b.seed)
        };
        let psms = gd(b.index, "psms_extracted.parquet");
        let chrom = gd(b.index, "chromatograms.parquet");
        info!(stage = %"extract", group = b.index, candidates = b.n, "run: stage start");
        let (npsm, nchr) = extract::run(extract::ExtractParams {
            ms2: &g.converted.ms2,
            library_precursors: &b.prec,
            library_fragments: g.lib_fragments,
            run_windows: &windows,
            ms1: Some(&g.converted.ms1),
            mass_cal: Some(&mass_cal),
            out_psms: &psms,
            out_chrom: &chrom,
            restrict_candidates: None,
            cfg: &cfg.extract,
            config_hash: ch,
            fragment_offset: Some(b.offset),
        })?;
        record_opt(
            g.man.as_deref_mut(),
            record_artifact(
                &format!("{}[g{:02}]", artifact::PSMS_EXTRACTED.0, b.index),
                artifact::PSMS_EXTRACTED,
                &psms,
                npsm,
                "extract",
                ch,
            )?,
        );
        record_opt(
            g.man.as_deref_mut(),
            record_artifact(
                &format!("{}[g{:02}]", artifact::CHROMATOGRAMS.0, b.index),
                artifact::CHROMATOGRAMS,
                &chrom,
                nchr,
                "extract",
                ch,
            )?,
        );
        let feats = gd(b.index, "features.parquet");
        let pin = gd(b.index, "run.pin");
        info!(stage = %"features", group = b.index, "run: stage start");
        let nf = features::run(features::FeaturesParams {
            psms: &psms,
            chromatograms: &chrom,
            // Corroboration and the confident elution boundary key on the seed by local
            // id; under global calibration "confident" means the pooled q.
            seed: Some(if global { &b.seed_view } else { &b.seed }),
            out: &feats,
            out_pin: &pin,
            cfg: &cfg.features,
            config_hash: ch,
        })?;
        record_opt(
            g.man.as_deref_mut(),
            record_artifact(
                &format!("{}[g{:02}]", artifact::FEATURES.0, b.index),
                artifact::FEATURES,
                &feats,
                nf,
                "features",
                ch,
            )?,
        );
        let competed = gd(b.index, "psms_competed.parquet");
        info!(stage = %"compete", group = b.index, "run: stage start");
        let nc = compete::run(compete::CompeteParams {
            features: &feats,
            out: &competed,
            cfg: &cfg.compete,
            config_hash: ch,
        })?;
        record_opt(
            g.man.as_deref_mut(),
            record_artifact(
                &format!("{}[g{:02}]", artifact::PSMS_COMPETED.0, b.index),
                artifact::PSMS_COMPETED,
                &competed,
                nc,
                "compete",
                ch,
            )?,
        );
        arts.push(pool::BandArtifacts {
            offset: b.offset,
            psms,
            chromatograms: chrom,
            features: feats,
            competed,
        });
    }

    // The run-level cal.json, as an ungrouped run writes it.
    groups::summarise_cal(&cals, global, &d("cal.json"))?;

    // --- pool into the single-run artifacts
    let out = Pooled {
        seed: pooled_seed,
        psms: d("psms_extracted.parquet"),
        chromatograms: d("chromatograms.parquet"),
        features: d("features.parquet"),
        competed: d("psms_competed.parquet"),
        rt_model,
    };
    info!(stage = %"pool", groups = arts.len(), "run: stage start");
    let stats = pool::run(pool::PoolParams {
        bands: &arts,
        out_psms: &out.psms,
        out_chromatograms: &out.chromatograms,
        out_features: &out.features,
        out_competed: &out.competed,
    })
    .context("pooling the window groups")?;
    for (name, schema, path, rows) in [
        (
            artifact::PSMS_EXTRACTED.0,
            artifact::PSMS_EXTRACTED,
            &out.psms,
            stats.psms,
        ),
        (
            artifact::CHROMATOGRAMS.0,
            artifact::CHROMATOGRAMS,
            &out.chromatograms,
            stats.chromatograms,
        ),
        (
            artifact::FEATURES.0,
            artifact::FEATURES,
            &out.features,
            stats.features,
        ),
        (
            artifact::PSMS_COMPETED.0,
            artifact::PSMS_COMPETED,
            &out.competed,
            stats.competed,
        ),
    ] {
        record_opt(
            g.man.as_deref_mut(),
            record_artifact(name, schema, path, rows, "pool", ch)?,
        );
        ArtifactReport {
            logical_name: name.to_string(),
            schema_name: name.to_string(),
            schema_version: schema.1,
            stage: "pool".to_string(),
            rows,
            content_hash: mumdia_io::hash::blake3_file(path)?,
            params: json!({
                "window_groups": arts.len(),
                "calibration": if global { "global" } else { "per_group" },
            }),
            stats: BTreeMap::from([
                ("groups".to_string(), json!(arts.len())),
                (
                    "overlap_duplicates_removed".to_string(),
                    json!(stats.duplicates),
                ),
            ]),
            model_identity: None,
            elapsed_ms: t0.elapsed().as_millis(),
        }
        .write_for(path)?;
    }
    // The feature and competed schema companions (`<table>.schema.json`) describe the
    // classifier's columns; every band wrote the same one, so the first band's is the pool's.
    for (band, pooled) in [
        (&arts[0].features, &out.features),
        (&arts[0].competed, &out.competed),
    ] {
        let src = format!("{band}.schema.json");
        if std::path::Path::new(&src).exists() {
            std::fs::copy(&src, format!("{pooled}.schema.json"))
                .with_context(|| format!("copying {src} beside the pooled table"))?;
        }
    }
    info!(
        groups = arts.len(),
        duplicates = stats.duplicates,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "groups: done"
    );
    Ok(out)
}
