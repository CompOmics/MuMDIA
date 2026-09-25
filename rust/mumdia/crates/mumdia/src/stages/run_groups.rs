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
//! `groups.parallel` bands run at a time in this process, so the resident set is that many
//! bands' libraries, hits and accepted rows -- plus ONE copy of the run's spectra, which
//! this module decodes once per PHASE and lends to every band's seed and extract. Before
//! that each band decoded them itself, twice: 63 bands were 126 decodes of a ~1 GB MS2
//! artifact and as many resident copies as there were bands in flight. Once per phase and
//! not once per run, because nothing between the two phases reads the spectra and the
//! DeepLC sidecars sit there: see `scan_fingerprint` and the two decode sites.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use mumdia_core::config::{Config, GroupBalance, GroupCalibration, GroupRtAdaptation};
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
use crate::spectra::{load_ms1, load_ms2, Ms1Scan};
use mumdia_core::types::Ms2Scan;

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
    /// The caller has already re-predicted `lib_precursors` with the DeepLC base model
    /// (`run-experiment` does so once per experiment when the multi-head calibration is
    /// off). Under `groups.rt_adaptation = once_per_run` the bands then keep those values
    /// rather than re-predicting their slices of the same table; under `per_band` they
    /// re-predict as before. Only a caller that did the re-prediction may set it.
    pub library_irt_repredicted: bool,
    /// A previous run's `groups/` directory whose band slices (`gNN/lib_precursors.parquet`)
    /// this run may take instead of writing its own, when its plan is this run's plan: the
    /// runs of one experiment search one library, so equal plans give equal slices. Only a
    /// band a DeepLC sidecar rewrites is written out at all.
    pub slices_from: Option<&'a str>,
    /// `lib_precursors` carries the native model's iRT as a placeholder for a DeepLC pass
    /// the multi-head calibration replaces (`predict_frag.defer_deeplc_to_multihead`): every
    /// band's calibrated table must then have re-predicted every row.
    pub irt_placeholder: bool,
}

/// Paths the pooled stages continue with.
/// What phase 1 of the band loop hands to phase 2.
///
/// The two phases exist because the confident elution half-widths are a property of the
/// run, not of a band: every band has to have extracted before the first one can compute
/// features on the pooled window.
struct BandExtract {
    index: usize,
    /// Rows this band's extract accepted: the cost the features phase is dispatched on.
    npsm: u64,
    psms: String,
    chrom: String,
    /// `(index, cal.json)`, as `summarise_cal` wants it.
    cal: (usize, String),
    /// This band's confident anchors, moved out into the pool between the phases.
    samples: features::BoundSamples,
    /// The extract-stage artifact records, recorded once phase 2 has run so the manifest
    /// keeps the order a single-phase loop wrote.
    recs: Vec<mumdia_core::manifest::ArtifactRecord>,
}

pub struct Pooled {
    pub seed: String,
    pub psms: String,
    /// The chromatogram tables quant reads for this run, in row order: the pooled
    /// `chromatograms.parquet`, or, when it was not written (`groups.pool_chromatograms =
    /// false`), the bands' own tables in band order, each with the overlap losers it does
    /// not contribute.
    pub chromatograms: Vec<super::quant::ChromTable>,
    pub features: String,
    /// The competed tables rescore reads for this run, in row order: the pooled
    /// `psms_competed.parquet`, or, when it was not written (`groups.pool_competed =
    /// false`), the bands' own tables in band order. Always one table, the pooled one,
    /// when the candidate audit or match-between-runs is on.
    pub competed: Vec<String>,
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

/// Whether two `groups/plan.json` files plan the same bands: same windows, same m/z bounds.
/// A slice is a function of the library and the band's m/z range, so equal plans over one
/// library give byte-identical slices. Unreadable or absent is "not the same".
fn same_plan(a: &str, b: &str) -> bool {
    let read = |p: &str| mumdia_io::json::read_json::<serde_json::Value>(p).ok();
    match (read(a), read(b)) {
        (Some(x), Some(y)) => x.get("bands").is_some() && x.get("bands") == y.get("bands"),
        _ => false,
    }
}

/// Record an artifact when this run keeps a manifest of its own.
fn record_opt(man: Option<&mut Manifest>, rec: mumdia_core::manifest::ArtifactRecord) {
    if let Some(m) = man {
        m.record(rec);
    }
}

/// Record a file that no stage report has hashed (a sidecar's output, the pooled seed),
/// hashing it only when there is a manifest to record it in. `run-experiment` passes no
/// manifest, and `record_opt(man, record_artifact(..)?)` read and hashed the whole file
/// first and then dropped the record.
fn record_hashing(
    man: Option<&mut Manifest>,
    logical_name: &str,
    schema: (&str, u32),
    path: &str,
    rows: u64,
    stage: &str,
    config_hash: &str,
) -> Result<()> {
    if let Some(m) = man {
        m.record(record_artifact(
            logical_name,
            schema,
            path,
            rows,
            stage,
            config_hash,
        )?);
    }
    Ok(())
}

/// Content fingerprint of the scans lent to the bands. FNV-1a over every field the
/// stages read, so any change to any peak changes it.
///
/// Sharing one buffer across bands rests on "no band writes to it", and most of that is
/// discharged by the type system rather than by this: the stages take `&[Ms2Scan]` and
/// `&[Ms1Scan]`, so a band that wanted to write would have to change the signature to
/// `&mut`, which does not compile. What the borrow checker does NOT cover is a write
/// through interior mutability or `unsafe`, and it says nothing at all about the second
/// decode of the same artifact returning something different from the first, which this
/// module now relies on. Both are checked here, in debug builds only: the assertions
/// below are `debug_assert!`, and in a release build this function returns a constant
/// that the optimiser deletes along with its call sites.
#[cfg(debug_assertions)]
fn scan_fingerprint(ms2: &[Ms2Scan], ms1: &[Ms1Scan]) -> u64 {
    #[inline]
    fn mix(h: u64, x: u64) -> u64 {
        (h ^ x).wrapping_mul(0x0000_0100_0000_01b3)
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    h = mix(h, ms2.len() as u64);
    for s in ms2 {
        h = mix(h, s.scan_index as u64);
        h = mix(h, s.rt_seconds.to_bits());
        h = mix(h, s.window.lower_mz.to_bits());
        h = mix(h, s.window.upper_mz.to_bits());
        h = mix(h, s.peaks.len() as u64);
        for p in &s.peaks {
            h = mix(h, p.mz.to_bits() as u64);
            h = mix(h, p.intensity.to_bits() as u64);
        }
    }
    h = mix(h, ms1.len() as u64);
    for s in ms1 {
        h = mix(h, s.scan_index as u64);
        h = mix(h, s.rt_seconds.to_bits());
        h = mix(h, s.mz.len() as u64);
        for (m, i) in s.mz.iter().zip(&s.intensity) {
            h = mix(h, m.to_bits() as u64);
            h = mix(h, i.to_bits() as u64);
        }
    }
    h
}

#[cfg(not(debug_assertions))]
fn scan_fingerprint(_ms2: &[Ms2Scan], _ms1: &[Ms1Scan]) -> u64 {
    0
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

    // --- the run's MS2, decoded once for the whole seeding phase
    //
    // Each band searches the whole run and differs only in its slice of the library, so
    // the scans are the same bytes for all of them, and one decode serves the lot. The
    // buffer is dropped at the end of the seeding phase rather than held to the end of the
    // run. Nothing between here and extract reads the spectra, and what sits in between is
    // the retention-time phase, where DeepLC sidecar processes run -- one per band under
    // `groups.rt_adaptation = per_band`, 63 of them on the 63-band run. Holding ~1 GB of
    // decoded MS2 across that is exactly the resident set the banding exists to bound, so
    // the extraction phase below decodes it a second time instead. Two decodes per run
    // against the 2m this replaces (m = bands), and `run.rs` refuses to share for the same
    // reason in the ungrouped single-run case. It is decoded before the plan because the
    // band costs, and under `groups.balance = cost` the cuts, count its peaks per window.
    info!(stage = %"load-ms2", "run: stage start");
    let ms2_scans: Vec<Ms2Scan> = load_ms2(&g.converted.ms2)?;
    let peaks = groups::peaks_per_window(&ms2_scans);

    // --- plan
    let windows = windows_of(&g.converted.isolation_windows)?;
    let stats = TableFile::open(g.lib_precursors)?.row_group_stats("precursor_mz")?;
    let cost_weight = |lo: f64, hi: f64| -> f64 {
        groups::est_precursors(&stats, lo, hi)
            * peaks
                .get(&(lo.to_bits(), hi.to_bits()))
                .copied()
                .unwrap_or(0) as f64
    };
    let plan = match cfg.groups.balance {
        GroupBalance::Precursors => groups::plan(&windows, &stats, cfg.groups.window_groups)?,
        GroupBalance::Cost => groups::plan_weighted(
            &windows,
            &stats,
            cfg.groups.window_groups,
            Some(&cost_weight),
        )?,
    };
    info!(
        groups = plan.bands.len(),
        windows = plan.windows.len(),
        est_unselectable = plan.est_unselectable as u64,
        est_duplicated = plan.est_duplicated as u64,
        calibration = ?cfg.groups.calibration,
        balance = ?cfg.groups.balance,
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
    let mut plan_json = serde_json::json!({
        "window_groups": cfg.groups.window_groups,
        "calibration": format!("{:?}", cfg.groups.calibration),
        "est_unselectable": plan.est_unselectable,
        "est_duplicated": plan.est_duplicated,
        "bands": plan.bands.iter().map(|b| serde_json::json!({
            "index": b.index, "mz_lo": b.mz_lo, "mz_hi": b.mz_hi,
            "windows": b.windows, "est_precursors": b.est_precursors,
        })).collect::<Vec<_>>(),
    });
    // Recorded only when it is not the default, so a default plan.json is what it was.
    if cfg.groups.balance != GroupBalance::Precursors {
        plan_json["balance"] = serde_json::json!(format!("{:?}", cfg.groups.balance));
    }
    mumdia_io::json::write_json(&d("groups/plan.json"), &plan_json)?;

    // --- band seeds
    struct Band {
        index: usize,
        offset: u32,
        n: usize,
        /// The band's precursor table: the whole library read by row span while `span` is
        /// set, else a band file with local ids (its slice, or the adapted table a DeepLC
        /// sidecar wrote).
        prec: String,
        span: Option<(usize, usize)>,
        seed: String,
    }
    let mut bands: Vec<Band> = Vec::new();
    // Estimated cost per band index, filled once the run's MS2 is decoded below.
    let mut band_cost: BTreeMap<usize, f64> = BTreeMap::new();
    // Bands in flight are driven from the rayon pool, and each one's extraction blocks its
    // own thread on the accumulation channel while the probing tasks run on the others. A
    // band in flight therefore occupies a worker that cannot do the work it is waiting for,
    // and `groups.parallel >= threads` deadlocks: every worker parks and no task is left to
    // feed them. Reproduced on the fixture with `parallel = 2, --threads 2` (the process sat
    // at 0.1 s of CPU indefinitely). Leave at least one worker free.
    let threads = rayon::current_num_threads();
    let par = {
        let want = cfg.groups.parallel.max(1);
        let most = threads.saturating_sub(1).max(1);
        if want > most {
            tracing::warn!(
                requested = want,
                used = most,
                threads,
                "groups.parallel is at least the thread count, which would deadlock: every                  band in flight parks a worker on its accumulation channel. Using one fewer                  band than there are threads; raise --threads to run more at once"
            );
            most
        } else {
            want
        }
    };
    // Band artifact records are kept only when this run has a manifest of its own. The
    // stages hash every output for their reports anyway, so a kept record reuses that hash
    // (`Written::record`); an unkept one is not built at all. Before, every band closure
    // called `record_artifact`, which read and hashed each band artifact a second time,
    // and under `run-experiment` (no manifest) the record was then dropped.
    let keep_records = g.man.is_some();
    let ms2_fingerprint = {
        info!(
            scans = ms2_scans.len(),
            bands = plan.bands.len(),
            "groups: MS2 decoded once for the seeding phase and shared"
        );
        let fp = scan_fingerprint(&ms2_scans, &[]);
        // Each band's estimated cost, its windows' precursors times their MS2 peaks, so the
        // band queues below start the longest bands first (`groups::window_costs`).
        let window_cost = groups::window_costs(&plan.windows, &stats, &peaks);
        for b in &plan.bands {
            band_cost.insert(b.index, b.windows.iter().map(|&w| window_cost[w]).sum());
        }

        // Bands are independent, so `groups.parallel` of them are sliced and seeded at once,
        // through a bounded queue: at most that many in flight, which bounds how many
        // working sets are resident (the whole point of banding), and a free slot takes the
        // next band at once instead of waiting for a chunk's slowest band. Results do not
        // depend on the schedule, and the records below are merged in band order.
        let slice_one =
        |b: &groups::Band| -> Result<Option<(Band, Vec<mumdia_core::manifest::ArtifactRecord>)>> {
            let (first, n) = Library::precursor_row_span(g.lib_precursors, b.mz_lo, b.mz_hi)?;
            if n == 0 {
                info!(
                    group = b.index,
                    "groups: band selects no precursor; skipped"
                );
                return Ok(None);
            }
            std::fs::create_dir_all(gd(b.index, ""))?;
            // The seed reads the band straight from the library by row span: the same
            // library, value for value, as a band file of those rows
            // (`Library::load_row_span_with`), without writing one. Until 2026-09-25 every
            // band was written out first, whether or not anything rewrote it, which on a
            // 203.5M-precursor library without an RT model was the whole precursor table
            // written again for every run. A band file is now written only where a DeepLC
            // sidecar rewrites the band (below).
            let seed = gd(b.index, "seed_psms.parquet");
            info!(stage = %"search-seed", group = b.index, "run: stage start");
            let written = search_seed::run_hashed(search_seed::SearchSeedParams {
                precursor_span: Some((first, n)),
                ms2: &g.converted.ms2,
                library_precursors: g.lib_precursors,
                library_fragments: g.lib_fragments,
                out: &seed,
                cfg: &cfg.search_seed,
                bucket_size: cfg.extract.bucket_size,
                config_hash: ch,
                fragment_offset: None,
                ms2_scans: Some(&ms2_scans),
                // The band writes its calibrant deviations so `seed-pool` can fit the
                // mass calibration once over the whole run rather than average the
                // bands' fitted scalars (see `crate::masscal`).
                emit_calibrants: true,
                library: None,
            })?;
            let rec: Vec<mumdia_core::manifest::ArtifactRecord> = if keep_records {
                vec![written.record(
                    &format!("{}[g{:02}]", artifact::SEED_PSMS.0, b.index),
                    artifact::SEED_PSMS,
                    &seed,
                    "search-seed",
                    ch,
                )]
            } else {
                Vec::new()
            };
            Ok(Some((
                Band {
                    index: b.index,
                    offset: first as u32,
                    n,
                    prec: g.lib_precursors.to_string(),
                    span: Some((first, n)),
                    seed,
                },
                rec,
            )))
        };
        let seed_order = groups::longest_first(
            &plan
                .bands
                .iter()
                .map(|b| band_cost[&b.index])
                .collect::<Vec<f64>>(),
        );
        let done = groups::run_bounded(&plan.bands, par, &seed_order, slice_one)?;
        for (band, recs) in done.into_iter().flatten() {
            for r in recs {
                record_opt(g.man.as_deref_mut(), r);
            }
            bands.push(band);
        }
        debug_assert_eq!(
            fp,
            scan_fingerprint(&ms2_scans, &[]),
            "a band's seed search modified the shared MS2 buffer"
        );
        fp
    };
    drop(ms2_scans);
    drop(peaks);
    if bands.is_empty() {
        bail!("groups: no band selects any precursor of the library");
    }
    // Whether any library row is in two bands. A band is a row span of the m/z-sorted
    // precursor table, and a candidate id is the band-local id plus the band's first row,
    // so spans that do not overlap cannot share a candidate and the pool has no overlap
    // duplicate to find. Exact where the plan's m/z test is not: two bands that touch at one
    // m/z value both hold a precursor at exactly that value.
    let bands_disjoint = {
        let mut spans: Vec<(usize, usize)> =
            bands.iter().map(|b| (b.offset as usize, b.n)).collect();
        spans.sort_unstable();
        spans.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0)
    };

    // --- pooled seed: library-wide ids, one q scale, one mass calibration
    let pooled_seed = d("seed_psms.parquet");
    let seeds: Vec<seed_pool::BandSeed> = bands
        .iter()
        .map(|b| seed_pool::BandSeed {
            path: b.seed.clone(),
            offset: b.offset,
            rows: b.n as u32,
        })
        .collect();
    let masscals: Vec<String> = bands
        .iter()
        .map(|b| crate::masscal::json_path(&b.seed))
        .collect();
    // The bands' calibrant deviations, so the fragment tolerance is fitted once over the
    // whole run rather than averaged from the bands' own p95s on their own q.
    let calibrants: Vec<String> = bands
        .iter()
        .map(|b| crate::masscal::calibrants_path(&b.seed))
        .collect();
    info!(stage = %"seed-pool", "run: stage start");
    let n = seed_pool::run(seed_pool::SeedPoolParams {
        seeds: &seeds,
        masscals: &masscals,
        calibrants: &calibrants,
        cfg: &cfg.search_seed,
        out: &pooled_seed,
    })?;
    record_hashing(
        g.man.as_deref_mut(),
        artifact::SEED_PSMS.0,
        artifact::SEED_PSMS,
        &pooled_seed,
        n,
        "seed-pool",
        ch,
    )?;
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
    // `groups.rt_adaptation = once_per_run`: one DeepLC worker for all the bands under
    // global calibration, rather than one per band (see `GroupRtAdaptation`). `per_group`
    // fits each band on its own anchors, so it keeps the per-band loop.
    let once_per_run = cfg.groups.rt_adaptation == GroupRtAdaptation::OncePerRun && global;
    // The caller re-predicted the library with the base model already, and the bands are
    // slices of that table: re-predicting them again only recomputes the same model's
    // values in different company. Kept per band under `per_band`, which is today's
    // behaviour.
    let keep_caller_irt = once_per_run && repredict && g.library_irt_repredicted;
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
            b.span = None;
        }
        info!(
            groups = bands.len(),
            model = %rt_model,
            source = %shared,
            "groups: reusing a previous run's adapted bands"
        );
    }
    let union_mode = g.shared_bands.is_none()
        && once_per_run
        && !keep_caller_irt
        && (g.mh_heads > 0 || repredict);
    if g.shared_bands.is_none() && keep_caller_irt {
        info!(
            groups = bands.len(),
            "groups: the library's iRT was re-predicted with the DeepLC base model for the \
             whole experiment; the bands keep those values (groups.rt_adaptation = \
             once_per_run)"
        );
    }
    // A DeepLC sidecar rewrites a band's precursor FILE, so the bands it adapts are written
    // out here, and only then: without an adaptation every stage reads the band by row span.
    // A previous run of the same experiment that planned the same bands under the same
    // library wrote the same bytes (a slice is a deterministic function of the library and
    // the row span), so its slices are taken instead where `slices_from` offers them.
    let per_band_sidecar = g.shared_bands.is_none()
        && !keep_caller_irt
        && (g.mh_heads > 0 || cfg.rt_im_train.finetune_deeplc || repredict);
    if per_band_sidecar {
        let reuse_dir = g
            .slices_from
            .filter(|dir| same_plan(&format!("{dir}/plan.json"), &d("groups/plan.json")));
        let write_one = |b: &Band| -> Result<String> {
            let (first, n) = b.span.expect("bands are read by span until adapted");
            if let Some(dir) = reuse_dir {
                let from = format!("{dir}/g{:02}/lib_precursors.parquet", b.index);
                if std::path::Path::new(&from).exists()
                    && mumdia_io::table::nrows(&from).is_ok_and(|r| r == n as u64)
                {
                    info!(
                        stage = %"band-slice",
                        group = b.index,
                        rows = n,
                        reused = %from,
                        "run: stage skipped"
                    );
                    return Ok(from);
                }
            }
            let out = gd(b.index, "lib_precursors.parquet");
            info!(stage = %"band-slice", group = b.index, rows = n, "run: stage start");
            groups::write_band_slice(g.lib_precursors, first, n, &out)?;
            Ok(out)
        };
        let order: Vec<usize> = (0..bands.len()).collect();
        let files = groups::run_bounded(&bands, par, &order, write_one)?;
        for (b, file) in bands.iter_mut().zip(files) {
            b.prec = file;
            b.span = None;
        }
    }
    if union_mode {
        let name = band_lib_name(&rt_model);
        let pairs: Vec<(String, String)> = bands
            .iter()
            .map(|b| (b.prec.clone(), gd(b.index, name)))
            .collect();
        let python = python.expect("mh_heads and repredict imply deeplc_python");
        let mode = if g.mh_heads > 0 {
            info!(stage = %"deeplc-multihead", groups = bands.len(), "run: stage start");
            crate::sidecar::BandAdaptation::Multihead {
                seed: &pooled_seed,
                n_heads: g.mh_heads,
                q_train: cfg.rt_im_train.q_train,
                window_holdout_frac: cfg.rt_im_train.window_holdout_frac,
            }
        } else {
            info!(stage = %"deeplc-repredict", groups = bands.len(), "run: stage start");
            crate::sidecar::BandAdaptation::Repredict
        };
        crate::sidecar::run_deeplc_bands(
            python,
            &script,
            &pairs,
            &d("groups/rt_bands.tsv"),
            mode,
            rayon::current_num_threads(),
            cfg.rt_im_train.deeplc_predict_shards,
            cfg.rt_im_train.deeplc_projection_cache.as_deref(),
        )?;
        for (b, (_, out)) in bands.iter_mut().zip(pairs) {
            if g.irt_placeholder && g.mh_heads > 0 {
                crate::sidecar::require_every_row_repredicted(&out)?;
            }
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
            b.span = None;
        }
    }
    for b in bands
        .iter_mut()
        .filter(|_| g.shared_bands.is_none() && !union_mode && !keep_caller_irt)
    {
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
                cfg.rt_im_train.deeplc_predict_shards,
                cfg.rt_im_train.deeplc_projection_cache.as_deref(),
            )?;
            if g.irt_placeholder {
                crate::sidecar::require_every_row_repredicted(&out)?;
            }
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
                rayon::current_num_threads(),
                cfg.rt_im_train.deeplc_predict_shards,
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
                cfg.rt_im_train.deeplc_predict_shards,
                cfg.rt_im_train.deeplc_projection_cache.as_deref(),
            )?;
            Some(out)
        } else {
            None
        };
        if let Some(out) = out {
            if keep_records {
                let rows = mumdia_io::table::nrows(&out)?;
                record_hashing(
                    g.man.as_deref_mut(),
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
                )?;
            }
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
    // Under global calibration every band fits the same pooled anchors with their iRT read
    // from the seed, so the retention-time fit is a property of the run: fitted here once,
    // and each band only applies it to its own table (docs/33 section 4). Before, each band
    // decoded the pooled seed and refitted the same curve. `per_group` fits per band.
    let shared_rt_fit = if global {
        let anchors = anchors_for_windows.as_deref().unwrap_or(&pooled_seed);
        info!(stage = %"rt-fit", anchors = %anchors, "run: stage start");
        let fit = rt_im_train::fit_from_seed(anchors, &cfg.rt_im_train)?;
        info!(
            n_train = fit.n_train(),
            bands = bands.len(),
            "groups: retention-time calibration fitted once for every band"
        );
        Some(fit)
    } else {
        None
    };

    // --- windows, extract, features, compete per band
    let mut arts: Vec<pool::BandArtifacts> = Vec::new();
    let mut cals: Vec<(usize, String)> = Vec::new();
    // --- the run's spectra, decoded once for the whole extraction phase
    //
    // Both are decoded here, after the retention-time phase and its per-band DeepLC
    // sidecars, and both are dropped at the end of this block, before the pooling that
    // reads and rewrites the run's largest artifacts. Neither is resident anywhere else.
    // The MS2 is a second decode of the artifact the seeding phase already read; see the
    // note there for why that is cheaper than holding it. `extract` itself loaded both
    // from these paths, so the failure on a missing or unreadable artifact is the same
    // one, raised a moment earlier.
    {
        info!(stage = %"load-ms2", "run: stage start");
        let ms2_scans: Vec<Ms2Scan> = load_ms2(&g.converted.ms2)?;
        info!(stage = %"load-ms1", "run: stage start");
        let ms1_scans: Vec<Ms1Scan> = load_ms1(&g.converted.ms1)?;
        // The seeded and the extracted band tables are joined by candidate id and scored on
        // one q scale, so the two phases have to have searched the same spectra. They are the
        // same file read twice by a deterministic loader, which is why this is a
        // debug_assert and not a run-time check.
        debug_assert_eq!(
            ms2_fingerprint,
            scan_fingerprint(&ms2_scans, &[]),
            "the extraction phase decoded a different MS2 than the seeding phase"
        );
        let extract_fingerprint = scan_fingerprint(&ms2_scans, &ms1_scans);
        info!(
            ms2 = ms2_scans.len(),
            ms1 = ms1_scans.len(),
            bands = bands.len(),
            "groups: spectra decoded once for the extraction phase and shared"
        );
        // `groups.parallel` bands at a time, through the same bounded queue as the seeding
        // phase, most expensive first. Each band in flight holds its own extraction working
        // set, so that bound is what the stage's memory scales with; the artifacts are pooled
        // in band order regardless of which band finishes first.
        // Phase 1 of two: retention-time windows and extraction for every band. The
        // confident elution half-widths are a property of the RUN, so features cannot
        // start until every band has contributed its anchors (see the pooling below).
        let band_extract = |b: &Band| -> Result<BandExtract> {
            let mut recs: Vec<mumdia_core::manifest::ArtifactRecord> = Vec::new();
            let windows = gd(b.index, "run_windows.parquet");
            let cal = gd(b.index, "cal.json");
            info!(stage = %"rt-im-train", group = b.index, "run: stage start");
            // Handed to this band's extract in memory; see `rt_im_train::RtWindows`.
            let (windows_written, fitted_windows) = match &shared_rt_fit {
                // Global: the run's one fit, applied to this band's table.
                Some(fit) => rt_im_train::apply_in_memory(
                    fit,
                    &rt_im_train::ApplyParams {
                        precursor_span: b.span,
                        library_precursors: &b.prec,
                        out_windows: &windows,
                        out_cal: &cal,
                        cfg: &cfg.rt_im_train,
                        config_hash: ch,
                    },
                )?,
                // Per group: this band's own anchors, iRT joined from its own table.
                None => rt_im_train::run_in_memory(rt_im_train::RtImTrainParams {
                    precursor_span: b.span,
                    seed_psms: &b.seed,
                    library_precursors: &b.prec,
                    out_windows: &windows,
                    out_cal: &cal,
                    cfg: &cfg.rt_im_train,
                    config_hash: ch,
                    anchor_irt_from_seed: false,
                })?,
            };
            if keep_records {
                recs.push(windows_written.record(
                    &format!("{}[g{:02}]", artifact::RUN_WINDOWS.0, b.index),
                    artifact::RUN_WINDOWS,
                    &windows,
                    "rt-im-train",
                    ch,
                ));
            }
            let mass_cal = if global {
                format!("{pooled_seed}.masscal.json")
            } else {
                format!("{}.masscal.json", b.seed)
            };
            let psms = gd(b.index, "psms_extracted.parquet");
            let chrom = gd(b.index, "chromatograms.parquet");
            info!(stage = %"extract", group = b.index, candidates = b.n, "run: stage start");
            let (psms_written, chrom_written) = extract::run_hashed(extract::ExtractParams {
                precursor_span: b.span,
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
                // A band file carries local ids with its fragments at the band's offset; a
                // span implies it.
                fragment_offset: b.span.is_none().then_some(b.offset),
                sibling_bands: par,
                rt_windows: fitted_windows,
                scans: Some(extract::SharedScans {
                    ms2: &ms2_scans,
                    ms1: Some(&ms1_scans),
                }),
            })?;
            if keep_records {
                recs.push(psms_written.record(
                    &format!("{}[g{:02}]", artifact::PSMS_EXTRACTED.0, b.index),
                    artifact::PSMS_EXTRACTED,
                    &psms,
                    "extract",
                    ch,
                ));
                recs.push(chrom_written.record(
                    &format!("{}[g{:02}]", artifact::CHROMATOGRAMS.0, b.index),
                    artifact::CHROMATOGRAMS,
                    &chrom,
                    "extract",
                    ch,
                ));
            }
            // The band half of the pooled bounds: the same pass `features` would run
            // internally, returning its anchors instead of a percentile of them.
            let samples = features::confident_bound_samples(&features::FeaturesParams {
                psms: &psms,
                chromatograms: &chrom,
                seed: Some(&pooled_seed),
                out: "",
                out_pin: "",
                cfg: &cfg.features,
                config_hash: ch,
            })?;
            Ok(BandExtract {
                index: b.index,
                npsm: psms_written.rows,
                psms,
                chrom,
                cal: (b.index, gd(b.index, "cal.json")),
                samples,
                recs,
            })
        };
        let extract_order = groups::longest_first(
            &bands
                .iter()
                .map(|b| band_cost.get(&b.index).copied().unwrap_or(0.0))
                .collect::<Vec<f64>>(),
        );
        let mut extracted: Vec<BandExtract> =
            groups::run_bounded(&bands, par, &extract_order, band_extract)?;
        debug_assert_eq!(
            extract_fingerprint,
            scan_fingerprint(&ms2_scans, &ms1_scans),
            "a band's extract modified the shared scan buffers"
        );
        drop(ms2_scans);
        drop(ms1_scans);

        // --- the run's confident elution half-widths, fitted once over every band
        //
        // An ungrouped run fits these from the whole run's confident anchors. A band holds
        // a slice of the m/z range and therefore a slice of the anchors, so fitting per
        // band puts most bands under the 20-anchor floor and they silently fall back to
        // per-candidate boundary detection: measured on a seven-file immunopeptidomics
        // search, 735 pooled anchors arrived as 0 or 1 per band and EVERY band fell back.
        // Pooling the samples is the same move `seed-pool` already makes for the q scale
        // and the mass calibration, and for the same reason.
        let mut pooled_bounds = features::BoundSamples::default();
        for e in &mut extracted {
            pooled_bounds.absorb(std::mem::take(&mut e.samples));
        }
        let bounds = features::bounds_from_samples(&pooled_bounds, &cfg.features);
        info!(
            anchors = pooled_bounds.len(),
            bands = extracted.len(),
            left_hw_s = bounds.map(|b| b.0),
            right_hw_s = bounds.map(|b| b.1),
            "groups: confident elution half-widths pooled over the bands"
        );

        // Phase 2 of two: features and competition, every band on the pooled window.
        let band_features = |e: &BandExtract| -> Result<(
            pool::BandArtifacts,
            (usize, String),
            Vec<mumdia_core::manifest::ArtifactRecord>,
        )> {
            let mut recs = Vec::new();
            let feats = gd(e.index, "features.parquet");
            let pin = gd(e.index, "run.pin");
            info!(stage = %"features", group = e.index, "run: stage start");
            let features_written = features::run_with_bounds_hashed(
                features::FeaturesParams {
                    psms: &e.psms,
                    chromatograms: &e.chrom,
                    // Corroboration keys on the seed by candidate id, and the band's tables
                    // carry library-wide ids, so the pooled seed is the one that matches.
                    // Its q is the pooled one either way, which is what "confident" has to
                    // mean once the bands are scored together.
                    seed: Some(&pooled_seed),
                    out: &feats,
                    out_pin: &pin,
                    cfg: &cfg.features,
                    config_hash: ch,
                },
                bounds,
            )?;
            if keep_records {
                recs.push(features_written.record(
                    &format!("{}[g{:02}]", artifact::FEATURES.0, e.index),
                    artifact::FEATURES,
                    &feats,
                    "features",
                    ch,
                ));
            }
            let competed = gd(e.index, "psms_competed.parquet");
            info!(stage = %"compete", group = e.index, "run: stage start");
            let competed_written = compete::run_hashed(compete::CompeteParams {
                features: &feats,
                out: &competed,
                cfg: &cfg.compete,
                config_hash: ch,
                features_hash: Some(&features_written.content_hash),
            })?;
            if keep_records {
                recs.push(competed_written.record(
                    &format!("{}[g{:02}]", artifact::PSMS_COMPETED.0, e.index),
                    artifact::PSMS_COMPETED,
                    &competed,
                    "compete",
                    ch,
                ));
            }
            Ok((
                pool::BandArtifacts {
                    psms: e.psms.clone(),
                    chromatograms: e.chrom.clone(),
                    competed,
                },
                e.cal.clone(),
                recs,
            ))
        };
        // Features and compete follow the band's accepted rows, which extract has just
        // counted, so that is the cost this phase is dispatched on.
        let features_order = groups::longest_first(
            &extracted
                .iter()
                .map(|e| e.npsm as f64)
                .collect::<Vec<f64>>(),
        );
        let done = groups::run_bounded(&extracted, par, &features_order, band_features)?;
        for (art, cal, recs) in done {
            for r in recs {
                record_opt(g.man.as_deref_mut(), r);
            }
            cals.push(cal);
            arts.push(art);
        }
        for e in extracted {
            for r in e.recs {
                record_opt(g.man.as_deref_mut(), r);
            }
        }
    }
    // Both buffers are dropped here, at the end of the block above, so neither is
    // resident across `pool::run` below.

    // The run-level cal.json, as an ungrouped run writes it.
    groups::summarise_cal(&cals, global, &d("cal.json"))?;

    // --- pool into the single-run artifacts
    //
    // The competed rows are left per band only where rescore then reads exactly the rows
    // the pooled table would hold, in its order, and nothing else reads that table: see
    // `groups.pool_competed`.
    let pooled_competed_path = d("psms_competed.parquet");
    let competed_readers = cfg.extract.emit_candidate_audit
        || cfg.mbr.strategy != mumdia_core::config::MbrStrategy::None;
    let pool_competed = cfg.groups.pool_competed || !bands_disjoint || competed_readers;
    if !cfg.groups.pool_competed {
        if pool_competed {
            info!(
                bands_disjoint,
                audit = cfg.extract.emit_candidate_audit,
                mbr = ?cfg.mbr.strategy,
                "groups: groups.pool_competed is off, but the competed table is pooled: the \
                 bands overlap, or the candidate audit or match-between-runs reads it"
            );
        } else {
            info!(
                groups = arts.len(),
                "groups: the competed rows stay per band and rescore reads the band tables \
                 (groups.pool_competed = false)"
            );
            // A pooled table left by an earlier run into this directory is not this run's,
            // and nothing would say so: take it away with its companions.
            for f in [
                pooled_competed_path.clone(),
                format!("{pooled_competed_path}.report.json"),
                format!("{pooled_competed_path}.schema.json"),
            ] {
                if std::path::Path::new(&f).exists() {
                    std::fs::remove_file(&f)
                        .with_context(|| format!("removing an earlier run's {f}"))?;
                }
            }
        }
    }
    // The chromatograms have one reader, quant, which can read the band tables itself with
    // the overlap losers (`groups.pool_chromatograms`).
    let pooled_chrom_path = d("chromatograms.parquet");
    let losers_path = d("groups/overlap_losers.parquet");
    let pool_chromatograms = cfg.groups.pool_chromatograms;
    if !pool_chromatograms {
        info!(
            groups = arts.len(),
            bands_disjoint,
            "groups: the chromatograms stay per band and quant reads the band tables with the \
             overlap losers (groups.pool_chromatograms = false)"
        );
    }
    // A pooled table (or loser sets) an earlier run left here under the other setting is
    // not this run's, and nothing would say so: take it away with its report, so that
    // neither a later standalone quant nor a reader of the directory mistakes it for this
    // run's.
    let stale = if pool_chromatograms {
        &losers_path
    } else {
        &pooled_chrom_path
    };
    for f in [stale.clone(), format!("{stale}.report.json")] {
        if std::path::Path::new(&f).exists() {
            std::fs::remove_file(&f).with_context(|| format!("removing an earlier run's {f}"))?;
        }
    }
    let mut out = Pooled {
        seed: pooled_seed,
        psms: d("psms_extracted.parquet"),
        // Filled in below, once the pool has found the overlap losers.
        chromatograms: Vec::new(),
        // Not pooled: nothing reads a run-level features table (compete's output carries
        // the feature columns), and on a real run it is 55 GB of writes per run.
        features: String::new(),
        competed: if pool_competed {
            vec![pooled_competed_path.clone()]
        } else {
            arts.iter().map(|a| a.competed.clone()).collect()
        },
        rt_model,
    };
    // The pooled extracted table has exactly one reader, the candidate audit, and that is
    // off by default. Pooling it anyway read and rewrote every band's extracted rows for a
    // file nothing opens; the per-band tables stay where they are either way.
    let pool_psms = cfg.extract.emit_candidate_audit;
    if !pool_psms {
        info!(
            groups = arts.len(),
            "groups: psms_extracted stays per band (extract.emit_candidate_audit is off, and              nothing else reads the pooled table)"
        );
    }
    info!(stage = %"pool", groups = arts.len(), "run: stage start");
    let stats = pool::run(pool::PoolParams {
        bands: &arts,
        out_psms: pool_psms.then_some(out.psms.as_str()),
        out_chromatograms: pool_chromatograms.then_some(pooled_chrom_path.as_str()),
        out_losers: (!pool_chromatograms).then_some(losers_path.as_str()),
        out_competed: pool_competed.then_some(pooled_competed_path.as_str()),
        bands_disjoint,
    })
    .context("pooling the window groups")?;
    out.chromatograms = if pool_chromatograms {
        vec![super::quant::ChromTable::whole(&pooled_chrom_path)]
    } else {
        arts.iter()
            .zip(&stats.losers)
            .map(|(a, drop)| super::quant::ChromTable {
                path: a.chromatograms.clone(),
                drop: drop.clone(),
            })
            .collect()
    };
    let pooled_artifacts = stats
        .psms_hash
        .clone()
        .filter(|_| pool_psms)
        .map(|h| {
            (
                artifact::PSMS_EXTRACTED.0,
                artifact::PSMS_EXTRACTED,
                &out.psms,
                stats.psms,
                h,
            )
        })
        .into_iter()
        .chain(stats.chromatograms_hash.clone().map(|h| {
            (
                artifact::CHROMATOGRAMS.0,
                artifact::CHROMATOGRAMS,
                &pooled_chrom_path,
                stats.chromatograms,
                h,
            )
        }))
        .chain(stats.losers_written.clone().map(|w| {
            (
                artifact::OVERLAP_LOSERS.0,
                artifact::OVERLAP_LOSERS,
                &losers_path,
                w.rows,
                w.content_hash,
            )
        }))
        .chain(stats.competed_hash.clone().map(|h| {
            (
                artifact::PSMS_COMPETED.0,
                artifact::PSMS_COMPETED,
                &pooled_competed_path,
                stats.competed,
                h,
            )
        }));
    for (name, schema, path, rows, content_hash) in pooled_artifacts {
        // One hash for both the manifest record and the report beside the file, computed
        // by the pool while it spliced the table: these are the run's largest artifacts,
        // and a read-back would read all of it again.
        record_opt(
            g.man.as_deref_mut(),
            mumdia_io::record_artifact_with_hash(
                name,
                schema,
                path,
                rows,
                "pool",
                ch,
                content_hash.clone(),
            ),
        );
        ArtifactReport {
            logical_name: name.to_string(),
            schema_name: name.to_string(),
            schema_version: schema.1,
            stage: "pool".to_string(),
            rows,
            content_hash,
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
    // The competed table's schema companion (`<table>.schema.json`) names the classifier's
    // columns; every band wrote the same one, so the first band's is the pool's.
    if pool_competed {
        let src = format!("{}.schema.json", arts[0].competed);
        if std::path::Path::new(&src).exists() {
            std::fs::copy(&src, format!("{pooled_competed_path}.schema.json"))
                .with_context(|| format!("copying {src} beside the pooled table"))?;
        }
    }
    // Opt-in: the bands' extracted and feature tables are read by nothing once the pool is
    // written (features are carried by the competed table; the extracted table's one
    // reader, the candidate audit, reads the pooled copy), and on a large run they are the
    // bulk of the band directories. Chromatograms and competed tables stay: the pooled
    // copies are what later stages read, but the band copies are what `mumdia pool
    // --groups-dir` re-pools from.
    if cfg.groups.delete_band_intermediates {
        let (mut files, mut bytes) = (0u64, 0u64);
        for (index, _) in &cals {
            for name in ["psms_extracted.parquet", "features.parquet", "run.pin"] {
                let base = gd(*index, name);
                for f in [
                    base.clone(),
                    format!("{base}.report.json"),
                    format!("{base}.schema.json"),
                ] {
                    if let Ok(meta) = std::fs::metadata(&f) {
                        std::fs::remove_file(&f)
                            .with_context(|| format!("deleting the band intermediate {f}"))?;
                        files += 1;
                        bytes += meta.len();
                    }
                }
            }
        }
        info!(
            groups = cals.len(),
            files,
            bytes,
            "groups: deleted the bands' psms_extracted and features tables after pooling \
             (groups.delete_band_intermediates); the manifest still lists them"
        );
    }
    info!(
        groups = arts.len(),
        duplicates = stats.duplicates,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "groups: done"
    );
    Ok(out)
}
