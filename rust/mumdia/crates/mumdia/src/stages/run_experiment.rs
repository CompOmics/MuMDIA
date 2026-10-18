//! Experiment-wide orchestrator `mumdia run-experiment`: run the per-file search
//! chain over N runs, then a single experiment-wide rescore, optional rescuable
//! match-between-runs transfer, a by-run split, per-run quantification, and
//! cross-run LFQ. This reifies the multi-run flow that was previously a manual
//! script sequence, keeping the rescuable-tier MBR first-class.
//!
//! The library is built or imported once and shared. Each run gets its own retention-time
//! calibration, its own extraction (chromatograms kept for quantification), and its own
//! competed table. The competed tables of all runs feed one rescore so the classifier and
//! FDR are experiment-wide.
//!
//! The DeepLC fine-tune defaults to running ONCE, on the first run, with every other run
//! reusing that library and fitting its own calibration against it
//! (`experiment.finetune_scope`). It is the most expensive step of a large experiment, and
//! per-run weight adaptation buys little once each run is calibrated separately.

use std::time::Instant;

use anyhow::{Context, Result};
use arrow::array::{Array, BooleanArray, UInt32Array};
use arrow::compute::filter_record_batch;
use mumdia_core::config::{Config, MultiheadAnchors, QuantQColumn, RtLibraryScope};
use mumdia_core::manifest::Manifest;
use mumdia_core::schema::artifact;
use mumdia_io::table::Written;
use rayon::prelude::*;
use serde_json::json;
use tracing::{info, warn};

use crate::stages::quant::ChromTable;
use crate::stages::*;

pub struct RunExperimentParams<'a> {
    pub config: &'a Config,
    /// Path the config was loaded from; see `RunParams::config_path`.
    pub config_path: Option<&'a str>,
    pub fasta: Option<&'a str>,
    pub mzmls: &'a [String],
    /// Optional per-run labels (default `r0..rN-1`); also the per-run subdir names.
    pub run_names: Option<&'a [String]>,
    pub out_dir: &'a str,
    pub lib_precursors: Option<&'a str>,
    pub lib_fragments: Option<&'a str>,
    pub max_spectra: usize,
    pub top_peaks_ms2: usize,
}

fn preflight(p: &RunExperimentParams) -> Result<()> {
    use mumdia_core::config::RescorerKind;
    if p.mzmls.len() < 2 {
        anyhow::bail!(
            "run-experiment needs >= 2 --mzml files (got {})",
            p.mzmls.len()
        );
    }
    for m in p.mzmls {
        if !std::path::Path::new(m).exists() {
            anyhow::bail!("--mzml not found: {m}");
        }
    }
    crate::stages::run::refuse_unsupported_on_4d(p.config, p.mzmls)?;
    match (p.lib_precursors, p.lib_fragments) {
        (Some(lp), Some(lf)) => {
            for path in [lp, lf] {
                if !std::path::Path::new(path).exists() {
                    anyhow::bail!("library not found: {path}");
                }
            }
        }
        (None, None) => match p.fasta {
            Some(f) if std::path::Path::new(f).exists() => {}
            Some(f) => anyhow::bail!("--fasta not found: {f}"),
            None => {
                anyhow::bail!("provide either --fasta or both --lib-precursors and --lib-fragments")
            }
        },
        _ => anyhow::bail!("library-input mode requires both --lib-precursors and --lib-fragments"),
    }
    let cfg = p.config;
    if cfg.rt_im_train.finetune_deeplc && cfg.predict_frag.deeplc_python.is_none() {
        anyhow::bail!("rt_im_train.finetune_deeplc requires predict_frag.deeplc_python");
    }
    // Explicit count: hard requirement. Automatic default: degrade with a warning, so a
    // native Python-free experiment stays runnable (`run.rs` says the same).
    if cfg.rt_im_train.multihead_calibration.is_some_and(|n| n > 0)
        && cfg.predict_frag.deeplc_python.is_none()
    {
        anyhow::bail!(
            "rt_im_train.multihead_calibration requires predict_frag.deeplc_python (DeepLC \
             >= 4.5.0): it calibrates the base model against each run's confident seed PSMs"
        );
    }
    if cfg.rt_im_train.multihead_calibration.is_none()
        && !cfg.rt_im_train.finetune_deeplc
        && cfg.predict_frag.deeplc_python.is_none()
    {
        warn!(
            "no DeepLC interpreter resolved, so retention-time multi-head calibration is \
             not running; it is the default and was worth 4.8% of peptides on AIF and \
             14.3% on Astral. Set predict_frag.deeplc_python (\"auto\" discovers one)."
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
            "rt_im_train.library_irt = deeplc requires predict_frag.deeplc_python; set \
             library_irt = library to keep the imported iRT"
        );
    }
    // Check the interpreters EXIST, not merely that the fields are set. The rescore runs
    // after every per-run chain, so on an 83-file batch a mistyped `rescore.python` used to
    // discard days of compute at the final stage. A wrong path is also the most likely error
    // when moving a config between machines.
    for (field, path) in [
        (
            "predict_frag.deeplc_python",
            cfg.predict_frag.deeplc_python.as_deref(),
        ),
        ("rescore.python", cfg.rescore.python.as_deref()),
        ("mbr.python", cfg.mbr.python.as_deref()),
    ] {
        if let Some(exe) = path {
            if !std::path::Path::new(exe).exists() {
                anyhow::bail!(
                    concat!(
                        "{} points at an interpreter that does not exist: {}. A config ",
                        "written for another machine is the usual cause; run ",
                        "`mumdia doctor --config <cfg>` to probe every configured sidecar ",
                        "environment before starting a long batch."
                    ),
                    field,
                    exe
                );
            }
        }
    }
    if matches!(
        cfg.rescore.classifier,
        RescorerKind::Mokapot | RescorerKind::NnTorch
    ) && cfg.rescore.python.is_none()
    {
        anyhow::bail!(
            "rescore.classifier={:?} requires rescore.python",
            cfg.rescore.classifier
        );
    }
    if cfg.mbr.strategy.transfers() && cfg.mbr.python.is_none() {
        anyhow::bail!(
            "mbr.strategy = {:?} (the rescuable-tier transfer) requires mbr.python",
            cfg.mbr.strategy
        );
    }
    refuse_unchecked_cross_run_step(cfg)?;
    Ok(())
}

/// Refuse the cross-run quant step ([`QuantConfig::needs_cross_run_fit`]) where its fit is
/// not yet known to be sound. The fit keys every reduction on `candidate_id` across runs, so
/// a candidate id must name the same precursor in every run. That holds for one shared
/// library and for the second pass's empirical library. It has not been checked for a
/// grouped run, whose bands write their own tables, nor for per-run library adaptation.
///
/// [`QuantConfig::needs_cross_run_fit`]: mumdia_core::config::QuantConfig::needs_cross_run_fit
fn refuse_unchecked_cross_run_step(cfg: &Config) -> Result<()> {
    if !cfg.quant.needs_cross_run_fit() {
        return Ok(());
    }
    if cfg.groups.window_groups > 1 {
        anyhow::bail!(
            "the cross-run quant step (quant.fragment_selection = cross_run, \
             quant.baseline_source = experiment or quant.fixed_scan_halfwidth_scale > 0) is \
             not supported with groups.window_groups > 1: candidate_id stability across a \
             grouped run's bands has not been checked"
        );
    }
    if cfg.experiment.rt_library_scope == RtLibraryScope::PerRun {
        anyhow::bail!(
            "the cross-run quant step (quant.fragment_selection = cross_run, \
             quant.baseline_source = experiment or quant.fixed_scan_halfwidth_scale > 0) is \
             not supported with experiment.rt_library_scope = per_run: candidate_id \
             stability across per-run libraries has not been checked. Use first_run_only."
        );
    }
    Ok(())
}

/// What the second pass of match-between-runs produced: each run's competed table and
/// chromatograms on the empirical library, in run order, and the library.
struct SecondPassOut {
    competed: Vec<String>,
    chroms: Vec<ChromTable>,
    library: second_pass::EmpiricalLibrary,
    /// Every run's `psms_extracted` of the second pass, by run, for the shadow 1/K0 term.
    psms_extracted: Vec<String>,
    /// Some run's spectra carry 1/K0, so the diagnostics are written to
    /// `empirical_library.json` (a 3D experiment's summary is what it always was).
    has_im: bool,
}

/// The first pass's `w_im` of a run, from its `cal.json`: `None` for a run without one (every
/// 3D run) or with a non-finite one.
fn first_pass_w_im(cal: &str) -> Option<f64> {
    let v: serde_json::Value = mumdia_io::json::read_json(cal).ok()?;
    v.get("w_im")?.as_f64().filter(|w| w.is_finite())
}

/// The second pass (`mbr.strategy = second_pass`): the empirical library of the first pass's
/// pooled scored table (`second_pass::build_library`) into `<out>/library`, then every run
/// searched again on it from the spectra its first pass converted (`run_dirs[i]/spectra`),
/// into `<out>/<name>`: a seed against the empirical library, which every run shares, its own
/// retention-time calibration on that seed, extract, features and compete. `None` when the
/// first pass accepted nothing, so there is nothing to search again.
///
/// The fragment-rarity prescreen is off here: the empirical library is already the screened
/// set, and the second pass must search all of it. Under `parallel_runs = auto` the runs are
/// pulled from the plan's queue, otherwise they run in chunks of `par` as the first pass did:
/// the plan made for the first pass is reused as it is.
///
/// On a run whose spectra carry 1/K0 the chain gets two settings of its own
/// (`chain::PassExtras`), and a 3D run neither:
///
/// - a 1/K0 window floor: `im_window_min` is raised to the run's first-pass `w_im`, because a
///   consensus 1/K0 calibrates to a narrower window than the gate's evidence supports
///   (recorded in `cal.json` as `im_window_floor`);
/// - a quantification trace pad (`extract::TracePad::for_quant`): the narrow second-pass RT
///   window would clip the fixed integration window and its baseline flanks of an apex near
///   its edge. It needs `extract.emit_rt_window`, which tells the features stage where the
///   window ends; without it there is no pad, with a warning.
///
/// Retrace runs here as in the first pass (`finish_run` with `retrace` on).
#[allow(clippy::too_many_arguments)]
fn search_second_pass(
    cfg: &Config,
    ch: &str,
    first_scored: &str,
    lib_p: &str,
    lib_f: &str,
    names: &[String],
    run_dirs: &[String],
    out: &str,
    plan: &crate::sched::RunConcurrency,
    par: usize,
) -> Result<Option<SecondPassOut>> {
    // The first pass's extracted rows, by source, for the consensus 1/K0.
    let first_psms: Vec<String> = run_dirs
        .iter()
        .map(|d| format!("{d}/psms_extracted.parquet"))
        .collect();
    let library = second_pass::build_library(&second_pass::BuildParams {
        scored: first_scored,
        lib_precursors: lib_p,
        lib_fragments: lib_f,
        psms_extracted: &first_psms,
        out_dir: &format!("{out}/library"),
        q: cfg.mbr.q_anchor,
    })?;
    if library.targets == 0 {
        return Ok(None);
    }
    let mut cfg2 = cfg.clone();
    cfg2.prescreen.enabled = false;
    let cfg2 = &cfg2;
    let t_lib = Instant::now();
    let seed_lib = search_seed::SeedLibrary::load(
        &library.precursors,
        &library.fragments,
        None,
        None,
        &cfg.search_seed,
        cfg.extract.bucket_size,
    )?;
    info!(
        candidates = seed_lib.n_candidates(),
        elapsed_ms = t_lib.elapsed().as_millis() as u64,
        "run-experiment: second pass, one seed library for every run"
    );
    let n = names.len();
    let one = |&i: &usize| -> Result<(String, String)> {
        info!(run = %names[i], i = i + 1, n, "run-experiment: second pass");
        let run_out = format!("{out}/{}", names[i]);
        std::fs::create_dir_all(&run_out).with_context(|| format!("creating {run_out}"))?;
        let co = convert::ConvertOutputs::existing(&format!("{}/spectra", run_dirs[i]))?;
        let seed = seed_run(
            cfg2,
            ch,
            &library.precursors,
            &library.fragments,
            &co,
            &run_out,
            Some(&seed_lib),
        )?;
        // The run's own settings on 4D data; on a 3D run `cfg_run` is `cfg2` and the extras
        // are the default, so its chain is the one it always was.
        let mut cfg_run = cfg2.clone();
        let mut extras = crate::stages::chain::PassExtras::default();
        let first_cal = format!("{}/cal.json", run_dirs[i]);
        if let Some(w) = first_pass_w_im(&first_cal) {
            let configured = cfg_run.rt_im_train.im_window_min;
            cfg_run.rt_im_train.im_window_min = configured.max(w);
            extras.im_window_floor = Some(crate::stages::rt_im_train::ImWindowFloor {
                configured,
                first_pass_w_im: w,
                source: &first_cal,
            });
            info!(
                run = %names[i],
                configured,
                first_pass_w_im = w,
                floor = cfg_run.rt_im_train.im_window_min,
                "run-experiment: second pass, 1/K0 window floor from the first pass"
            );
        }
        if crate::instrument::spectra_has_im(&co.ms2) {
            match extract::TracePad::for_quant(&cfg.quant) {
                Some(pad) if cfg.extract.emit_rt_window => extras.trace_pad = Some(pad),
                Some(_) => warn!(
                    run = %names[i],
                    "run-experiment: second pass without extract.emit_rt_window, so the \
                     chromatograms are not padded for quantification and a fixed integration \
                     window near the RT window's edge is clipped"
                ),
                None => {}
            }
        }
        finish_run(
            &cfg_run,
            ch,
            &library.precursors,
            &library.fragments,
            &co,
            &seed,
            &run_out,
            true,
            extras,
        )
    };
    let all: Vec<usize> = (0..n).collect();
    let done: Vec<(String, String)> = if plan.is_auto() {
        plan.map_pooled(&all, one)?
    } else {
        let mut done = Vec::with_capacity(n);
        for chunk in all.chunks(par.max(1)) {
            done.extend(crate::colread::first_err(
                chunk.par_iter().map(one).collect(),
            )?);
        }
        done
    };
    drop(seed_lib);
    let (competed, chroms): (Vec<String>, Vec<ChromTable>) = done
        .into_iter()
        .map(|(c, h)| (c, ChromTable::whole(&h)))
        .unzip();
    // The pooled rescore needs decoys to estimate an FDR. Real data always leaves some; a
    // synthetic or near-empty experiment can extract none in the second pass's narrow windows,
    // and then the first pass stands.
    if count_decoys(&competed)? == 0 {
        warn!(
            "run-experiment: the second pass extracted no decoy PSM, so it cannot estimate an \
             FDR of its own"
        );
        return Ok(None);
    }
    let has_im = run_dirs.iter().any(|d| {
        convert::ConvertOutputs::existing(&format!("{d}/spectra"))
            .is_ok_and(|co| crate::instrument::spectra_has_im(&co.ms2))
    });
    let psms_extracted = names
        .iter()
        .map(|nm| format!("{out}/{nm}/psms_extracted.parquet"))
        .collect();
    Ok(Some(SecondPassOut {
        competed,
        chroms,
        library,
        psms_extracted,
        has_im,
    }))
}

/// Decoy rows over `tables`, from their `label` column.
fn count_decoys(tables: &[String]) -> Result<u64> {
    let mut n = 0u64;
    for t in tables {
        let f = mumdia_io::table::TableFile::open(t)?;
        for b in f.batches(Some(&["label"]), 1 << 16)? {
            let b = b?;
            let label = arrow::compute::cast(b.column(0), &arrow::datatypes::DataType::Utf8)?;
            let label = label
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .expect("cast to utf8");
            n += (0..label.len())
                .filter(|&i| label.is_valid(i) && label.value(i) == "decoy")
                .count() as u64;
        }
    }
    Ok(n)
}

/// The per-run search chain: convert -> seed -> optional fine-tune -> rt-cal ->
/// extract -> features -> compete. The chromatograms are kept for the later per-run
/// quantification.
/// `shared_ft` is a precursor library that has ALREADY been
/// DeepLC-fine-tuned (by an earlier run of this same experiment); when present this run
/// uses it as-is instead of fine-tuning again, and still fits its own retention-time
/// calibration on top. Returns `(competed, chromatograms, fine_tuned_library_if_produced)`,
/// where `competed` is the run's competed tables in row order: one, or a grouped run's band
/// tables when its pooled table was not written (`groups.pool_competed = false`).
#[allow(clippy::too_many_arguments)]
fn process_run(
    cfg: &Config,
    ch: &str,
    lib_p_base: &str,
    lib_f: &str,
    // Whether the experiment was given a library rather than a FASTA. Only the caller
    // knows, and the multi-head default is scoped on it.
    library_input: bool,
    mzml: &str,
    out: &str,
    top_peaks_ms2: usize,
    max_spectra: usize,
    shared_rt_lib: Option<&str>,
    // `lib_p_base` is the experiment-level base-model re-prediction of the imported iRT.
    library_irt_repredicted: bool,
    // An earlier grouped run's `groups/` directory whose band slices this run may reuse.
    slices_from: Option<&str>,
    // `lib_p_base` carries placeholder iRT the multi-head calibration must replace in full
    // (`predict_frag.defer_deeplc_to_multihead`).
    irt_placeholder: bool,
) -> Result<(Vec<String>, Vec<ChromTable>, Option<String>)> {
    let co = convert_run(cfg, mzml, out, top_peaks_ms2, max_spectra)?;
    let has_deeplc = cfg.predict_frag.deeplc_python.is_some();
    let mh_heads = cfg
        .rt_im_train
        .multihead_heads(has_deeplc, cfg.deeplc_rt_source(library_input, has_deeplc));
    // Grouped: seed, calibration, extract, features and compete happen one isolation-window
    // group at a time, and the pooled competed table and chromatograms come back under the
    // names the pooled stages below read, exactly as in the single-run orchestrator. The
    // third return value is then this run's `groups/` directory rather than an adapted
    // library, which is what `experiment.rt_library_scope = first_run_only` hands to the
    // runs that follow (`shared_rt_lib` here).
    if cfg.groups.window_groups > 1 {
        let pooled = crate::stages::run_groups::run(crate::stages::run_groups::GroupRun {
            cfg,
            config_hash: ch,
            converted: &co,
            lib_precursors: lib_p_base,
            lib_fragments: lib_f,
            out_dir: out,
            man: None,
            shared_bands: shared_rt_lib,
            mh_heads,
            library_input,
            library_irt_repredicted,
            slices_from,
            irt_placeholder,
        })?;
        return Ok((
            pooled.competed,
            pooled.chromatograms,
            Some(format!("{out}/groups")),
        ));
    }
    let seed = seed_run(cfg, ch, lib_p_base, lib_f, &co, out, None)?;
    chain_after_seed(
        cfg,
        ch,
        lib_p_base,
        lib_f,
        library_input,
        &co,
        out,
        &seed,
        shared_rt_lib,
        irt_placeholder,
    )
}

/// Convert one run's mzML (or vendor file) into `<out>/spectra`.
fn convert_run(
    cfg: &Config,
    mzml: &str,
    out: &str,
    top_peaks_ms2: usize,
    max_spectra: usize,
) -> Result<convert::ConvertOutputs> {
    let d = |name: &str| format!("{out}/{name}");
    std::fs::create_dir_all(out).ok();
    // Fold the conversion caps into the convert artifacts' provenance key, exactly as
    // both the standalone `convert` subcommand and single-run `run` do. The caps change
    // the spectra output but are not part of the config, so the bare config hash made
    // two experiments with different caps record identical convert provenance.
    let convert_hash = mumdia_io::hash::blake3_str(&format!(
        "{}\u{1f}max_spectra={}\u{1f}top_peaks_ms2={}\u{1f}top_peaks_ms1={}",
        cfg.canonical_json(),
        max_spectra,
        top_peaks_ms2,
        0
    ));
    convert::run(convert::ConvertParams {
        mzml,
        out_dir: &d("spectra"),
        max_spectra,
        top_peaks_ms2,
        top_peaks_ms1: 0,
        config_hash: &convert_hash,
        tdf: convert::TdfParams::from_config(&cfg.convert),
    })
}

/// Seed one ungrouped run against the base library (the seed is iRT-independent) into
/// `<out>/seed_psms.parquet`, with the experiment's shared seed library when one is lent.
/// Returns the seed path.
fn seed_run(
    cfg: &Config,
    ch: &str,
    lib_p_base: &str,
    lib_f: &str,
    co: &convert::ConvertOutputs,
    out: &str,
    library: Option<&search_seed::SeedLibrary>,
) -> Result<String> {
    let seed = format!("{out}/seed_psms.parquet");
    search_seed::run(search_seed::SearchSeedParams {
        precursor_span: None,
        fragment_offset: None,
        // One reader at a time, as in the ungrouped `run`.
        ms2_scans: None,
        emit_calibrants: false,
        library,
        ms2: &co.ms2,
        library_precursors: lib_p_base,
        library_fragments: lib_f,
        out: &seed,
        cfg: &cfg.search_seed,
        bucket_size: cfg.extract.bucket_size,
        config_hash: ch,
    })?;
    Ok(seed)
}

/// Everything of an ungrouped run's chain after its seed: the optional RT-library
/// adaptation, rt-im-train, extract, features and compete. Returns
/// `(competed, chromatograms, adapted_library_if_produced)`.
#[allow(clippy::too_many_arguments)]
fn chain_after_seed(
    cfg: &Config,
    ch: &str,
    lib_p_base: &str,
    lib_f: &str,
    library_input: bool,
    co: &convert::ConvertOutputs,
    out: &str,
    seed: &str,
    shared_rt_lib: Option<&str>,
    irt_placeholder: bool,
) -> Result<(Vec<String>, Vec<ChromTable>, Option<String>)> {
    let has_deeplc = cfg.predict_frag.deeplc_python.is_some();
    let mh_heads = cfg
        .rt_im_train
        .multihead_heads(has_deeplc, cfg.deeplc_rt_source(library_input, has_deeplc));
    let adapted = adapt_rt_library(
        cfg,
        lib_p_base,
        seed,
        out,
        mh_heads,
        shared_rt_lib,
        irt_placeholder,
        rayon::current_num_threads(),
        BootInputs { ch, lib_f, co },
    )?;
    let (competed, chrom) = finish_run(
        cfg,
        ch,
        &adapted.lib,
        lib_f,
        co,
        seed,
        out,
        true,
        Default::default(),
    )?;
    Ok((
        vec![competed],
        vec![ChromTable::whole(&chrom)],
        adapted.produced,
    ))
}

/// What a run's search-anchored refit needs beyond the adaptation's own arguments
/// (`rt_im_train.multihead_anchors = search`, `rt_bootstrap`): the bootstrap pass searches
/// the run once on the seed-anchored library.
#[derive(Clone, Copy)]
struct BootInputs<'a> {
    ch: &'a str,
    lib_f: &'a str,
    co: &'a convert::ConvertOutputs,
}

/// What [`adapt_rt_library`] chose for one run.
struct Adapted {
    /// The precursor table the rest of the run reads.
    lib: String,
    /// The table this run produced, which the runs after it may reuse.
    produced: Option<String>,
    /// Whether the adaptation searched the run in this process (the bootstrap pass of
    /// `rt_bootstrap`: extract, features and compete on the seed-anchored library), so a
    /// memory window around it measured a search of this run.
    searched: bool,
}

/// Choose the precursor table the rest of an ungrouped run reads: a previous run's adapted
/// library, this run's multi-head calibration or fine-tune of the base library (`threads`
/// torch threads), or the base library. Under `rt_im_train.multihead_anchors = search`, or
/// `auto` on 1/K0 spectra or a seed short of anchors, the multi-head calibration is refitted
/// on the confident targets of a bootstrap search of this run (`rt_bootstrap::refit`).
#[allow(clippy::too_many_arguments)]
fn adapt_rt_library(
    cfg: &Config,
    lib_p_base: &str,
    seed: &str,
    out: &str,
    mh_heads: usize,
    shared_rt_lib: Option<&str>,
    irt_placeholder: bool,
    threads: usize,
    boot: BootInputs,
) -> Result<Adapted> {
    let d = |name: &str| format!("{out}/{name}");
    // DeepLC fine-tune. Under `RtLibraryScope::FirstRunOnly` the caller hands every run
    // after the first the library the first run produced, so the fine-tune -- the most
    // expensive step in the whole experiment -- is paid once. Run-to-run chromatographic
    // drift is then absorbed by `rt_im_train`'s per-run calibration below, which is fitted
    // separately for every run regardless.
    let mut produced_rt_lib: Option<String> = None;
    let mut searched = false;
    // The anchor cut (`rt_im_train::plan_anchors`): relaxed on a seed short of anchors, and
    // when still short the multi-head calibration is skipped for this run, which then keeps
    // `lib_p_base` and adapts nothing for the runs after it to reuse.
    let anchor_plan =
        if shared_rt_lib.is_none() && (mh_heads > 0 || cfg.rt_im_train.finetune_deeplc) {
            Some(rt_im_train::plan_anchors(seed, &cfg.rt_im_train)?)
        } else {
            None
        };
    let anchor_q = anchor_plan.map_or(cfg.rt_im_train.q_train, |a| a.q_train);
    let mh_heads = match anchor_plan {
        Some(a) if mh_heads > 0 && !a.sufficient => {
            if irt_placeholder {
                anyhow::bail!(
                    "the library carries placeholder retention times for the multi-head \
                     calibration to replace, but the seed search of {out} found only {} target \
                     anchors even at q {}; re-run without predict_frag.defer_deeplc_to_multihead",
                    a.n_anchors,
                    a.q_train
                );
            }
            0
        }
        _ => mh_heads,
    };
    let lib_p = if let Some(shared) = shared_rt_lib {
        // A previous run already adapted the library and `experiment.rt_library_scope`
        // says to reuse it. Which mechanism produced it does not matter here: the
        // fine-tune and multi-head both hand back a precursor table with rewritten
        // retention times, and this run fits its own LOESS on top either way.
        shared.to_string()
    } else if mh_heads > 0 {
        let python = cfg
            .predict_frag
            .deeplc_python
            .as_deref()
            .expect("mh_heads is 0 unless an interpreter resolved");
        let script = crate::sidecar::resolve_script(
            &cfg.predict_frag.sidecar_script_dir,
            "deeplc_finetune.py",
        );
        // Planned above whenever a multi-head calibration runs on a run with no shared
        // library, which is this branch.
        let plan = anchor_plan.expect("the anchor cut is planned for a multi-head calibration");
        // Under `multihead_anchors = search`, or `auto` on a seed short of anchors, the
        // seed-anchored fit goes to `rt_bootstrap/` and the refitted library takes the usual
        // name; otherwise this is the usual name.
        let has_im = crate::instrument::spectra_has_im(&boot.co.ms2);
        let search = rt_bootstrap::decide(cfg, mh_heads, &plan, has_im, out);
        let lib_p_mh = rt_bootstrap::seed_library_path(search, out);
        crate::sidecar::run_deeplc_multihead(
            python,
            &script,
            lib_p_base,
            seed,
            &lib_p_mh,
            mh_heads,
            anchor_q,
            cfg.rt_im_train.window_holdout_frac,
            threads,
            cfg.rt_im_train.deeplc_predict_shards,
            crate::cache::projection_dir(cfg).as_ref(),
        )?;
        if irt_placeholder {
            crate::sidecar::require_every_row_repredicted(&lib_p_mh)?;
        }
        let lib_p_mh = if search {
            rt_bootstrap::refit(&rt_bootstrap::Refit {
                cfg,
                config_hash: boot.ch,
                converted: boot.co,
                lib_p_base,
                lib_f: boot.lib_f,
                seed,
                out,
                seed_library: &lib_p_mh,
                mh_heads,
                threads,
                irt_placeholder,
                seed_anchors: plan.n_anchors,
            })?
            .library
        } else {
            lib_p_mh
        };
        rt_bootstrap::record_auto(cfg, &plan, has_im, search, &lib_p_mh);
        produced_rt_lib = Some(lib_p_mh.clone());
        searched = search;
        lib_p_mh
    } else if cfg.rt_im_train.finetune_deeplc {
        let python = cfg
            .predict_frag
            .deeplc_python
            .as_deref()
            .expect("preflight");
        let script = crate::sidecar::resolve_script(
            &cfg.predict_frag.sidecar_script_dir,
            "deeplc_finetune.py",
        );
        let lib_p_ft = d("fragment_library_precursors_ft.parquet");
        crate::sidecar::run_deeplc_finetune(
            python,
            &script,
            lib_p_base,
            seed,
            &lib_p_ft,
            cfg.rt_im_train.finetune_epochs,
            cfg.rt_im_train.finetune_patience,
            anchor_q,
            cfg.rt_im_train.finetune_batch,
            // Keep the fine-tune exclusion aligned with rt-im-train's holdout
            // split (see run.rs); 0.0 (default) changes nothing.
            cfg.rt_im_train.window_holdout_frac,
            cfg.rng_seed,
            threads,
            cfg.rt_im_train.deeplc_predict_shards,
        )?;
        produced_rt_lib = Some(lib_p_ft.clone());
        lib_p_ft
    } else {
        lib_p_base.to_string()
    };
    Ok(Adapted {
        lib: lib_p,
        produced: produced_rt_lib,
        searched,
    })
}

/// Retention-time windows, extract, the optional retrace, features and compete of one
/// ungrouped run on `lib_p`. `retrace` says whether this pass may retrace
/// (`chain::ChainParams::retrace`): off for the RT bootstrap pass only. Returns
/// `(competed, chromatograms)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_run(
    cfg: &Config,
    ch: &str,
    lib_p: &str,
    lib_f: &str,
    co: &convert::ConvertOutputs,
    seed: &str,
    out: &str,
    retrace: bool,
    extras: crate::stages::chain::PassExtras,
) -> Result<(String, String)> {
    let o = crate::stages::chain::extract_to_compete(crate::stages::chain::ChainParams {
        cfg,
        config_hash: ch,
        lib_p,
        lib_f,
        converted: co,
        seed,
        out_dir: out,
        lent_ms2: None,
        manifest: None,
        retrace,
        extras,
    })?;
    Ok((o.competed, o.chromatograms))
}

/// Split an experiment-wide scored table into per-run tables by the `source`
/// column (0..n-1), preserving the schema and the rows exactly. Quant then runs per run
/// with `q_filter = psm_q`, keeping each run's own confident PSMs. Returns each run's table
/// with its content hash, computed while it was written.
///
/// The rows of one run are contiguous: rescore appends each competed table's rows in input
/// order and stamps `source` with the input's index, the top-K collapse keeps that order,
/// and MBR's round trip preserves it. So every row group except at most `n_runs - 1`
/// boundary groups holds one source, and its statistics say so (`min == max`). Those
/// groups are spliced into their run's table as bytes (`SpliceWriter`), and only the
/// boundary groups are decoded, filtered and re-encoded. Decoding and re-encoding all of
/// it, which this did before, is one thread's work over every row of every column: an
/// estimated 3.5-7 min on the 258.75M-row immunopeptidomics pool, against a copy at disk
/// speed.
///
/// The per-run tables hold the same rows, in the same order, with the same values either
/// way, so quant and the report read the same data. Their bytes differ from a re-encoded
/// split: a spliced group keeps the scored table's own layout (1,048,576-row groups), so
/// the content hashes the experiment manifest records for them change. The re-encoding
/// path remains for a table the splice cannot take: one whose `source` is nullable or has
/// no statistics (the MBR worker's pyarrow output), or whose re-encoded boundary rows the
/// splice refuses.
fn split_by_source(scored: &str, out_paths: &[String]) -> Result<Vec<Written>> {
    let t = mumdia_io::table::TableFile::open(scored)?;
    let src_idx = t
        .schema
        .index_of("source")
        .map_err(|_| anyhow::anyhow!("scored table has no `source` column for split"))?;
    if t.schema.field(src_idx).data_type() != &arrow::datatypes::DataType::UInt32 {
        anyhow::bail!("`source` column is not u32");
    }
    let spliced = if t.schema.field(src_idx).is_nullable() {
        None
    } else {
        match split_spliced(&t, scored, out_paths) {
            Ok(done) => done,
            Err(e) => {
                warn!(
                    error = %format!("{e:#}"),
                    "run-experiment: splicing the scored table by source failed; re-encoding \
                     every row instead"
                );
                None
            }
        }
    };
    let (written_tables, written) = match spliced {
        Some(done) => done,
        None => split_rewritten(&t, src_idx, out_paths)?,
    };
    // The split is a partition, so it must account for every input row. Nothing
    // enforced that: a `source` value outside `0..out_paths.len()` -- which a
    // hand-assembled or externally rescored table can carry, and which the MBR
    // worker could reintroduce -- dropped those PSMs into no output at all. Every
    // downstream number is then computed from a silently smaller population, with
    // no error and no warning. A count is the whole check.
    let total: usize = t.nrows;
    if written != total {
        anyhow::bail!(
            "splitting {scored} by `source` placed {written} of {total} rows into \
             {} per-run tables; the rest carry a source index outside 0..{}, so they \
             would be dropped from every per-run quantity",
            out_paths.len(),
            out_paths.len()
        );
    }
    Ok(written_tables)
}

/// Rows per row group of a re-encoded per-run table, and of the re-encoded boundary rows
/// the splice takes. The scored table is 22 columns wide (two more after MBR), about 170
/// bytes a row decoded, so a group is about 22 MB.
const SPLIT_ROW_GROUP_ROWS: usize = 1 << 17;

/// The splice form of [`split_by_source`]: `None` when the statistics cannot drive it.
/// Returns the tables and the rows placed.
fn split_spliced(
    t: &mumdia_io::table::TableFile,
    scored: &str,
    out_paths: &[String],
) -> Result<Option<(Vec<Written>, usize)>> {
    use mumdia_io::table::SpliceWriter;
    let stats = t.row_group_stats("source")?;
    let spans = SpliceWriter::row_group_spans(scored)?;
    if stats.len() != spans.len() {
        return Ok(None);
    }
    let n = out_paths.len();
    // Per row group, the run whose rows it holds, when its statistics say it holds one
    // run's rows only. A group whose one source is outside 0..n is left to the boundary
    // rewrite, which places none of its rows, so the partition check reports them.
    let single: Vec<Option<usize>> = stats
        .iter()
        .map(|s| match (s.min, s.max) {
            (Some(lo), Some(hi)) if lo == hi && lo >= 0.0 && lo < n as f64 => Some(lo as usize),
            _ => None,
        })
        .collect();
    let mut writers: Vec<SpliceWriter> = out_paths
        .iter()
        .map(|out| SpliceWriter::create_hashed(out, scored))
        .collect::<Result<_>>()?;
    let mut written = 0usize;
    let mut g = 0usize;
    while g < spans.len() {
        match single[g] {
            Some(run) => {
                // Consecutive groups of this one source, spliced in one call.
                let start = g;
                while g < spans.len() && single[g] == Some(run) {
                    g += 1;
                }
                let end = g;
                written +=
                    writers[run].append_row_groups(scored, |k| k >= start && k < end)? as usize;
            }
            None => {
                written += rewrite_boundary_group(t, spans[g], out_paths, &mut writers)?;
                g += 1;
            }
        }
    }
    let tables = writers
        .into_iter()
        .map(SpliceWriter::close_hashed)
        .collect::<Result<Vec<_>>>()?;
    Ok(Some((tables, written)))
}

/// Decode one row group that holds more than one source, and splice each run's rows of it
/// into that run's table through a temporary file. Returns the rows placed.
fn rewrite_boundary_group(
    t: &mumdia_io::table::TableFile,
    span: (usize, usize),
    out_paths: &[String],
    writers: &mut [mumdia_io::table::SpliceWriter],
) -> Result<usize> {
    let (first_row, n_rows) = span;
    let part = t.span(first_row, n_rows)?;
    let src_idx = part
        .schema
        .index_of("source")
        .map_err(|_| anyhow::anyhow!("scored table has no `source` column for split"))?;
    // One temporary table per run with rows in this group, created with its first row.
    let mut temps: Vec<Option<(String, mumdia_io::table::BatchWriter)>> =
        (0..out_paths.len()).map(|_| None).collect();
    let mut placed = 0usize;
    part.for_each_batch(None, 1 << 14, |b| {
        let src = b
            .column(src_idx)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| anyhow::anyhow!("`source` column is not u32"))?;
        for (i, slot) in temps.iter_mut().enumerate() {
            let mask: BooleanArray = (0..src.len()).map(|k| src.value(k) == i as u32).collect();
            let filtered = filter_record_batch(b, &mask)?;
            if filtered.num_rows() == 0 {
                continue;
            }
            if slot.is_none() {
                let tmp = format!("{}.split-rg{first_row}.parquet", out_paths[i]);
                let w = mumdia_io::table::BatchWriter::with_row_group_rows(
                    &tmp,
                    part.schema.clone(),
                    SPLIT_ROW_GROUP_ROWS,
                )?;
                *slot = Some((tmp, w));
            }
            let (_, w) = slot.as_mut().expect("created above");
            w.write(&filtered)?;
            placed += filtered.num_rows();
        }
        Ok(())
    })?;
    let mut spliced = 0usize;
    for (i, slot) in temps.into_iter().enumerate() {
        let Some((tmp, w)) = slot else {
            continue;
        };
        w.close()?;
        let appended = writers[i].append_row_groups(&tmp, |_| true);
        std::fs::remove_file(&tmp).ok();
        spliced += appended? as usize;
    }
    if spliced != placed {
        anyhow::bail!("split: re-encoded {placed} boundary rows but spliced {spliced}");
    }
    Ok(placed)
}

/// The re-encoding form of [`split_by_source`]: one streaming pass, every output with its
/// writer open, each input batch filtered once per run and appended, so the resident set is
/// one batch rather than the whole experiment-wide scored table. Returns the tables and the
/// rows placed.
fn split_rewritten(
    t: &mumdia_io::table::TableFile,
    src_idx: usize,
    out_paths: &[String],
) -> Result<(Vec<Written>, usize)> {
    // Row groups capped at 131,072 rows. Only the row-group boundaries change; the rows,
    // their order and their values do not.
    let mut writers: Vec<mumdia_io::table::BatchWriter> = out_paths
        .iter()
        .map(|out| {
            mumdia_io::table::BatchWriter::with_options(
                out,
                t.schema.clone(),
                mumdia_io::table::WriteOptions::new()
                    .row_group_rows(SPLIT_ROW_GROUP_ROWS)
                    .content_hash(),
            )
        })
        .collect::<Result<_>>()?;
    let mut written = 0usize;
    t.for_each_batch(None, 1 << 14, |b| {
        let src = b
            .column(src_idx)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| anyhow::anyhow!("`source` column is not u32"))?;
        for (i, w) in writers.iter_mut().enumerate() {
            let mask: BooleanArray = (0..src.len()).map(|k| src.value(k) == i as u32).collect();
            let filtered = filter_record_batch(b, &mask)?;
            written += filtered.num_rows();
            if filtered.num_rows() > 0 {
                w.write(&filtered)?;
            }
        }
        Ok(())
    })?;
    let tables = writers
        .into_iter()
        .map(mumdia_io::table::BatchWriter::close_hashed)
        .collect::<Result<Vec<_>>>()?;
    Ok((tables, written))
}

/// Reject run names that would share a per-run output directory.
///
/// Compared without regard to case on every platform, not only where the filesystem
/// is known to fold case: `RunA` and `runa` are two directories on ext4 and one
/// directory on NTFS, APFS and most network shares, and an experiment's output may be
/// written to any of them. Sequential runs overwrite each other's artifacts there and
/// parallel ones interleave, with no error from either (docs/29 #5). Refusing the pair
/// everywhere costs nothing anyone would want.
fn check_run_names_distinct(ns: &[String]) -> Result<()> {
    let mut seen: std::collections::BTreeMap<String, &str> = std::collections::BTreeMap::new();
    for n in ns {
        if let Some(prev) = seen.insert(n.to_lowercase(), n.as_str()) {
            if prev == n {
                anyhow::bail!(
                    "--run-names must be unique: {n:?} is given twice; each name is a per-run \
                     output subdirectory, so a repeat makes two runs write the same artifacts \
                     into one directory and interleave their results with no error"
                );
            }
            anyhow::bail!(
                "--run-names {prev:?} and {n:?} differ only in case; on a case-insensitive \
                 filesystem (Windows, macOS, most network shares) they are one per-run output \
                 directory, so the names must be distinct without regard to case"
            );
        }
    }
    Ok(())
}

/// Why `name` cannot be a per-run directory name on every platform, or `None`.
///
/// Syntactic, not probed: Windows's rules are applied everywhere, because an experiment's
/// output may be written to any filesystem and a name that is one directory on NTFS must
/// not be two on ext4. `a` and `a.` passed the old check (empty, separators, `.`, `..`) and
/// were one directory on Windows: the second run overwrote the first and the experiment
/// exited 0 with both split tables holding `source = 1` (docs/30 R2).
fn portable_dir_name_problem(name: &str) -> Option<&'static str> {
    const RESERVED: [&str; 22] = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
        "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    if name.is_empty() {
        return Some("it is empty");
    }
    if name == "." || name == ".." {
        return Some("`.` and `..` are not names");
    }
    if name.contains('/') || name.contains('\\') {
        return Some("it contains a path separator");
    }
    if name
        .chars()
        .any(|c| matches!(c, '<' | '>' | ':' | '"' | '|' | '?' | '*') || (c as u32) < 0x20)
    {
        return Some("it contains a character Windows forbids in a file name (<>:\"|?* or a control character)");
    }
    if name.ends_with('.') || name.ends_with(' ') {
        return Some("it ends with a dot or a space, which Windows strips, so it names the same directory as the trimmed form");
    }
    let stem = name.split('.').next().unwrap_or(name).to_ascii_uppercase();
    if RESERVED.contains(&stem.as_str()) {
        return Some("it is a Windows reserved device name (CON, PRN, AUX, NUL, COM1-9, LPT1-9), with or without an extension");
    }
    None
}

pub fn run(p: RunExperimentParams) -> Result<()> {
    let t0 = Instant::now();
    // The time from here to the first stage, by step (`prestage::PreStageTimer`).
    let mut pre = crate::prestage::PreStageTimer::start("run-experiment");
    // Same contract as the single-run orchestrator, and it matters more here: an
    // 83-file batch must not fail on a missing interpreter after the first run has
    // already been searched.
    let mut resolved = p.config.clone();
    resolved.predict_frag.sidecar_script_dir =
        crate::python::resolve_script_dir(&resolved.predict_frag.sidecar_script_dir, p.config_path);
    crate::python::resolve(&mut resolved)?;
    // `features.im_features = auto` resolves once for the experiment, on whether any run's
    // spectra will carry 1/K0 (`instrument::input_has_im`, as the library build decides),
    // so every run is scored with one feature schema and the pooled rescore accepts them.
    resolved.resolve_im_features(p.mzmls.iter().any(|m| crate::instrument::input_has_im(m)));
    crate::stages::retrace::refuse_inputs_without_raw_events(&resolved, p.mzmls)?;
    pre.step("resolve_interpreters");
    let p = RunExperimentParams {
        config: &resolved,
        ..p
    };
    let cfg = p.config;
    if cfg.prescreen.tag_prefilter || cfg.prescreen.score_before_prediction {
        anyhow::bail!(
            "{}",
            concat!(
                "prescreen.tag_prefilter and score_before_prediction are supported by a single-file ",
                "`run` only: an experiment ",
                "shares one library across its runs, so the survivors of every run would have to ",
                "be united first. Search the files one per `run`."
            )
        );
    }
    preflight(&p)?;
    pre.step("preflight");
    let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
    std::fs::create_dir_all(p.out_dir).ok();
    let d = |name: &str| format!("{}/{}", p.out_dir, name);
    let n_runs = p.mzmls.len();

    // Provenance: the identity of the code, of the configuration and of the INPUTS,
    // hashed from the start, before any stage reads them (docs/29 #15). Hashing at the
    // end recorded whatever bytes were on disk after a multi-hour experiment, which is
    // not necessarily what the search read.
    //
    // The hashes are taken on a background thread (`prestage::InputHashes`) and joined
    // when the manifest is written. Taken serially here they were the first, cold read of
    // every mzML and of the library, minutes of I/O with nothing else running on a large
    // experiment. The order is the order the chain reads the files: the first run's mzML,
    // then the library its seed loads, then the other runs. The manifest keys inputs by
    // role, so the order changes nothing in it.
    let mut prov = Manifest::new(cfg.canonical_json(), ch.clone());
    let mut hash_order: Vec<(String, String)> = Vec::with_capacity(n_runs + 3);
    hash_order.push(("mzml[0]".to_string(), p.mzmls[0].clone()));
    for (role, path) in [
        ("fasta", p.fasta),
        ("lib_precursors", p.lib_precursors),
        ("lib_fragments", p.lib_fragments),
    ] {
        if let Some(path) = path {
            hash_order.push((role.to_string(), path.to_string()));
        }
    }
    for (i, m) in p.mzmls.iter().enumerate().skip(1) {
        hash_order.push((format!("mzml[{i}]"), m.clone()));
    }
    let input_hashes = crate::prestage::InputHashes::spawn("run-experiment", hash_order);
    pre.step("provenance");
    // Reject a bad --run-names rather than silently substituting r0..rN-1.
    //
    // The old `_ =>` arm swallowed any count mismatch with no warning, and accepted
    // duplicates outright. Duplicates are the dangerous half: two runs then share
    // `d(&names[i])`, so above one parallel run two `process_run` calls concurrently
    // write the same spectra_ms2 / seed_psms / psms_extracted / chromatograms into one
    // directory, and the split and quant paths collide too. The output is an interleaving
    // of two runs with no error anywhere. Reachable by naming runs after their basenames
    // when an experiment spans two directories holding same-named files.
    let names: Vec<String> = match p.run_names {
        Some(ns) => {
            if ns.len() != n_runs {
                anyhow::bail!(
                    "--run-names has {} entries but there are {n_runs} runs; pass one name \
                     per --mzml, in the same order, or omit it to get r0..r{}",
                    ns.len(),
                    n_runs - 1
                );
            }
            check_run_names_distinct(ns)?;
            if let Some((bad, why)) = ns
                .iter()
                .find_map(|n| portable_dir_name_problem(n).map(|why| (n, why)))
            {
                anyhow::bail!(
                    "--run-names entry {bad:?} is not usable as a directory name: {why}; each \
                     name becomes a subdirectory of --out-dir"
                );
            }
            ns.to_vec()
        }
        None => (0..n_runs).map(|i| format!("r{i}")).collect(),
    };

    // A FASTA build may leave DeepLC to the multi-head calibration of each run, which
    // re-predicts every row before anything reads the iRT
    // (`predict_frag.defer_deeplc_to_multihead`).
    let irt_placeholder = cfg.defers_library_deeplc(
        p.lib_precursors.is_some(),
        cfg.predict_frag.deeplc_python.is_some(),
    );

    // --- shared library (imported or digested once) ---
    let (lib_p_base, lib_f) = match (p.lib_precursors, p.lib_fragments) {
        (Some(lp), Some(lf)) => {
            info!(lib_precursors = lp, "run-experiment: library-input mode");
            (lp.to_string(), lf.to_string())
        }
        _ => {
            let fasta = p.fasta.expect("preflight guarantees --fasta in build mode");
            let lib_p = d("fragment_library_precursors.parquet");
            let lib_f = d("fragment_library_fragments.parquet");
            // `predict_frag.library_cache`, as in `run`: a stored library with the same key
            // is published here instead of being built.
            // One library serves every run: it carries 1/K0 when any run's spectra will
            // (`instrument::input_has_im`, decided before conversion as in `run`).
            let run_has_im = p.mzmls.iter().any(|m| crate::instrument::input_has_im(m));
            let cache = crate::library_cache::LibraryCache::for_config(
                cfg,
                fasta,
                irt_placeholder,
                run_has_im,
                (&prov.mumdia_version, &prov.git_sha),
            );
            if cache
                .as_ref()
                .and_then(|c| c.restore(&lib_p, &lib_f))
                .is_some()
            {
                pre.first_stage("library-cache");
                // A reused output directory may still hold an earlier build's digest and
                // peptidoforms, which did not produce this library.
                crate::library_cache::remove_build_intermediates(p.out_dir);
            } else {
                let dig = d("peptides.parquet");
                pre.first_stage("digest");
                digest::run(digest::DigestParams {
                    fasta,
                    out: &dig,
                    cfg: &cfg.digest,
                    rng_seed: cfg.rng_seed,
                    config_hash: &ch,
                })?;
                let pf = d("peptidoforms.parquet");
                peptidoforms::run(peptidoforms::PeptidoformsParams {
                    peptides: &dig,
                    out: &pf,
                    cfg: &cfg.peptidoforms,
                    config_hash: &ch,
                })?;
                predict_frag::run(predict_frag::PredictFragParams {
                    rt_placeholder: irt_placeholder,
                    peptidoforms: &pf,
                    out_precursors: &lib_p,
                    out_fragments: &lib_f,
                    work_dir: &d("sidecar_work"),
                    cfg: &cfg.predict_frag,
                    config_hash: &ch,
                    run_has_im,
                })?;
                match &cache {
                    Some(c) => {
                        c.store(&lib_p, &lib_f);
                        crate::cache::enforce_for(cfg);
                    }
                    None => {
                        if let Some(hint) =
                            crate::library_cache::reuse_hint(cfg, &lib_p, &lib_f, irt_placeholder)
                        {
                            info!(
                                "run-experiment: to search other files against this library \
                                 without building it again, pass {hint}, or turn \
                                 predict_frag.library_cache on (\"auto\" or a directory) and \
                                 keep --fasta"
                            );
                        }
                    }
                }
            }
            (lib_p, lib_f)
        }
    };

    // Library-input mode without a fine-tune: the DeepLC base-model re-prediction of the
    // imported iRT does not depend on any run, so it is computed once here and every
    // per-run chain fits its own RT calibration against the same table.
    let library_irt_repredicted = cfg.rt_im_train.repredicts_library_irt(
        p.lib_precursors.is_some(),
        cfg.predict_frag.deeplc_python.is_some(),
    );
    // Library-input mode: the next stage is the base-model re-prediction or the first
    // run's conversion. A FASTA build logged at its digest already, and this is a no-op.
    pre.first_stage(if library_irt_repredicted {
        "deeplc-repredict"
    } else {
        "convert"
    });
    let lib_p_base = if library_irt_repredicted {
        let python = cfg
            .predict_frag
            .deeplc_python
            .as_deref()
            .expect("repredicts_library_irt implies deeplc_python");
        let script = crate::sidecar::resolve_script(
            &cfg.predict_frag.sidecar_script_dir,
            "deeplc_finetune.py",
        );
        let out = d("fragment_library_precursors_deeplc.parquet");
        info!(
            library = %lib_p_base,
            "run-experiment: re-predicting the library iRT once with the DeepLC base model"
        );
        crate::sidecar::run_deeplc_repredict(
            python,
            &script,
            &lib_p_base,
            &out,
            rayon::current_num_threads(),
            cfg.rt_im_train.deeplc_predict_shards,
            crate::cache::projection_dir(cfg).as_ref(),
        )?;
        out
    } else {
        // Same two reasons as in `run`: no interpreter (warn) or the multi-head calibration
        // taking over the re-prediction on the first run (a plan, logged as such).
        if p.lib_precursors.is_some()
            && !cfg.rt_im_train.finetune_deeplc
            && matches!(
                cfg.rt_im_train.library_irt,
                mumdia_core::config::LibraryIrt::Auto
            )
        {
            if cfg.predict_frag.deeplc_python.is_none() {
                warn!(
                    "run-experiment: keeping the imported library iRT because no \
                     predict_frag.deeplc_python is configured; configure one to re-predict with \
                     DeepLC, or set rt_im_train.library_irt = library to silence this"
                );
            } else {
                info!(
                    "run-experiment: the multi-head calibration re-predicts the library iRT on \
                     the first run; skipping the base-model re-prediction it would overwrite"
                );
            }
        }
        lib_p_base
    };

    // --- per-run search chains ---
    // Runs are independent: `process_run` derives every path from its own per-run output
    // directory, so nothing is shared between them. Concurrency is OPT-IN and bounded
    // because each run's extract can hold tens of GB -- running all 83 files of an
    // experiment at once would exhaust memory long before it saturated the CPU.
    // `parallel_runs = 1` (the default) is the previous sequential behaviour exactly.
    //
    // Ordering: `competed`/`chroms` must stay in run index order, since the downstream
    // combined rescore keys rows by `source` index. Chunks are processed in order and
    // rayon's indexed `collect` preserves within-chunk order, so the result is identical
    // to the sequential build regardless of completion order.
    //
    // `parallel_runs = "auto"` (stored as 0) is the other scheduler (`sched::RunConcurrency`):
    // the count comes from the thread budget, each chain runs in a rayon pool of its own
    // share, and a chain starts when a slot frees rather than at a chunk boundary. The
    // explicit count keeps the chunk loops below exactly as they were.
    let threads_budget = rayon::current_num_threads();
    // `resolve_for_host`: an automatic plan with no memory reading to bound it (any
    // platform but Linux) runs one chain at a time.
    let mut plan =
        crate::sched::resolve_for_host(cfg.experiment.parallel_runs, n_runs, threads_budget);
    let par = plan.par;
    if plan.is_auto() {
        info!(
            parallel_runs = plan.par,
            pool_threads = plan.pool_threads.unwrap_or(0),
            threads = threads_budget,
            n = n_runs,
            "run-experiment: parallel_runs = auto, sized from the thread budget (one run per \
             16 threads at most, each in a pool of its own)"
        );
    }
    // Each run's competed tables, in row order (see `process_run`).
    let mut competed: Vec<Vec<String>> = Vec::with_capacity(n_runs);
    // Each run's chromatogram tables, in row order: one, or a grouped run's band tables
    // (`groups.pool_chromatograms = false`).
    let mut chroms: Vec<Vec<ChromTable>> = Vec::with_capacity(n_runs);

    // Under `RtLibraryScope::FirstRunOnly` (the default) the first run is processed alone
    // so the library it adapted can be handed to all the others. That adaptation -- the
    // DeepLC fine-tune, or multi-head calibration -- is the most expensive step in a large
    // experiment: one full re-prediction of the library per run, tens of minutes on a
    // 9.4M-row library, so paying it N times instead of once is the difference between
    // hours and days on a large batch. Every run still fits its OWN retention-time
    // calibration (LOESS by default) against that shared library, and that per-run fit is
    // what absorbs chromatographic drift.
    //
    // Multi-head was per-run unconditionally until 2026-09-14, on the reasoning that it is
    // fitted against THIS run's chromatography. That is true, and it is also true of the
    // fine-tune, which has been shareable all along: what a shared library fixes is elution
    // ORDER, which replicate injections on one method share. A batch that genuinely
    // reorders wants `per_run`.
    // Resolved exactly as `process_run` resolves it, so the decision to process the first
    // run alone matches what that run will actually do.
    let has_deeplc_for_scope = cfg.predict_frag.deeplc_python.is_some();
    let deeplc_rt_source_for_scope =
        cfg.deeplc_rt_source(p.lib_precursors.is_some(), has_deeplc_for_scope);
    let adapts_rt_library = cfg.rt_im_train.finetune_deeplc
        || cfg
            .rt_im_train
            .multihead_heads(has_deeplc_for_scope, deeplc_rt_source_for_scope)
            > 0;
    // A grouped run adapts its bands rather than one library table, and shares them the
    // same way: the first run's `groups/` directory is what the rest reuse.
    let share_ft = adapts_rt_library
        && matches!(
            cfg.experiment.rt_library_scope,
            RtLibraryScope::FirstRunOnly
        )
        && n_runs > 1;
    // `experiment.overlap_front_threads`: under first_run_only, the converts and seeds of
    // runs 2..N run beside run 1's RT adaptation instead of before it (the branch below).
    let grouped_runs = cfg.groups.window_groups > 1;
    // `"auto"` (the default): the threads the DeepLC thread cap leaves idle, so run 1's
    // adaptation keeps its thread count and the adapted library its bits.
    let overlap = match cfg.experiment.overlap_front_threads {
        Some(n) => n,
        None => crate::sched::auto_overlap_threads(threads_budget),
    };
    if overlap > 0 && (grouped_runs || !share_ft) && cfg.experiment.overlap_front_threads.is_some()
    {
        info!(
            overlap_front_threads = overlap,
            "run-experiment: experiment.overlap_front_threads applies to ungrouped runs whose \
             first run adapts the library for the rest (rt_library_scope = first_run_only); \
             running the runs one after the other"
        );
    }
    // `rt_im_train.multihead_anchors = search` makes run 1's adaptation a whole search
    // (extract to rescore) rather than one DeepLC worker, which would compete with the fronts
    // for the same threads, so the first run then adapts alone, as without the overlap.
    // `auto` decides only after run 1's seed, so it keeps the overlap: when it chooses the
    // bootstrap, that pass runs on the adaptation's thread share beside the fronts. It
    // chooses it on runs short of seed anchors, which are the small ones.
    let mh_heads_for_scope = cfg
        .rt_im_train
        .multihead_heads(has_deeplc_for_scope, deeplc_rt_source_for_scope);
    let search_anchors = rt_bootstrap::always(cfg, mh_heads_for_scope);
    rt_bootstrap::warn_if_inert(cfg, mh_heads_for_scope, "run-experiment");
    if share_ft && overlap > 0 && !grouped_runs && search_anchors {
        info!(
            "run-experiment: rt_im_train.multihead_anchors = search; the first run adapts the \
             library alone (no experiment.overlap_front_threads), because its bootstrap search \
             would share the threads of the overlapped fronts"
        );
    }
    let overlapped = share_ft && overlap > 0 && !grouped_runs && !search_anchors;
    if overlapped && cfg.experiment.overlap_front_threads.is_none() {
        info!(
            overlap_front_threads = overlap,
            threads = threads_budget,
            "run-experiment: overlap_front_threads = auto, the threads beyond the physical \
             cores, which run 1's DeepLC thread cap leaves idle"
        );
    }
    // Ungrouped runs seed against ONE library: every run's seed searches the base library
    // at the seed tolerance, so the library and its fragment index are the same arrays for
    // every run, and each seed used to load and build them again. So the chain runs in
    // three phases: every run is converted first (the conversion is the memory-heavy step
    // of its own and needs no library), then the seed library is loaded once and every run
    // seeded against it, then the library is dropped and each run's chain continues from
    // its seed. The outputs are the ones the per-run chain wrote: every stage is
    // deterministic and reads only its own run's inputs, so only the order in which the
    // stages of different runs execute changes. A grouped run seeds band by band against
    // band libraries and keeps its own chain. The overlapped experiment orders its own
    // phases (below), so it skips these.
    let library_input = p.lib_precursors.is_some();
    let prepared: Vec<Option<(convert::ConvertOutputs, String)>> = if grouped_runs || overlapped {
        (0..n_runs).map(|_| None).collect()
    } else {
        let mut conv: Vec<convert::ConvertOutputs> = Vec::with_capacity(n_runs);
        let all: Vec<usize> = (0..n_runs).collect();
        let convert_one = |i: usize| {
            info!(run = %names[i], i = i + 1, n = n_runs, "run-experiment: convert");
            convert_run(
                cfg,
                &p.mzmls[i],
                &d(&names[i]),
                p.top_peaks_ms2,
                p.max_spectra,
            )
        };
        if plan.is_auto() {
            // Not in per-run pools: a conversion already takes its share of the engine's
            // pool by dividing it among the live conversions (`convert::LIVE_CONVERTS`),
            // and inside a pool of its own that division would count the others twice.
            // With a memory reading the first conversion runs alone and its peak bounds
            // how many of the rest run at once, as the chains are bounded later: the
            // thread-sized count alone is no memory bound.
            conv = crate::sched::run_first_alone(
                plan,
                threads_budget,
                "run-experiment",
                "the first conversion",
                &all,
                |&i| convert_one(i),
                |p, rest| crate::sched::map_bounded(rest, p.par, |&i| convert_one(i)),
            )?;
        } else {
            for chunk in all.chunks(par) {
                let done: Vec<convert::ConvertOutputs> =
                    crate::colread::first_err(chunk.par_iter().map(|&i| convert_one(i)).collect())?;
                conv.extend(done);
            }
        }
        let t_lib = Instant::now();
        let seed_lib = search_seed::SeedLibrary::load(
            &lib_p_base,
            &lib_f,
            None,
            None,
            &cfg.search_seed,
            cfg.extract.bucket_size,
        )?;
        info!(
            candidates = seed_lib.n_candidates(),
            runs = n_runs,
            elapsed_ms = t_lib.elapsed().as_millis() as u64,
            "run-experiment: one seed library and fragment index for every run's seed"
        );
        let mut seeds: Vec<String> = Vec::with_capacity(n_runs);
        let seed_one = |i: usize| {
            info!(run = %names[i], i = i + 1, n = n_runs, "run-experiment: seed");
            seed_run(
                cfg,
                &ch,
                &lib_p_base,
                &lib_f,
                &conv[i],
                &d(&names[i]),
                Some(&seed_lib),
            )
        };
        if plan.is_auto() {
            // Bounded the same way: the first seed alone, from a reset high-water mark, so
            // its peak is one seed over the resident seed library.
            seeds = crate::sched::run_first_alone(
                plan,
                threads_budget,
                "run-experiment",
                "the first seed",
                &all,
                |&i| seed_one(i),
                |p, rest| p.map_pooled(rest, |&i| seed_one(i)),
            )?;
        } else {
            for chunk in all.chunks(par) {
                let done: Vec<String> =
                    crate::colread::first_err(chunk.par_iter().map(|&i| seed_one(i)).collect())?;
                seeds.extend(done);
            }
        }
        drop(seed_lib);
        conv.into_iter().zip(seeds).map(Some).collect()
    };
    // One run's chain: from its seed when phases 1-2 ran, else (grouped) whole, with the
    // band slices of an earlier grouped run to reuse (`slices_from`).
    let run_one = |i: usize,
                   shared: Option<&str>,
                   slices_from: Option<&str>|
     -> Result<(Vec<String>, Vec<ChromTable>, Option<String>)> {
        match &prepared[i] {
            Some((co, seed)) => chain_after_seed(
                cfg,
                &ch,
                &lib_p_base,
                &lib_f,
                library_input,
                co,
                &d(&names[i]),
                seed,
                shared,
                irt_placeholder,
            ),
            None => process_run(
                cfg,
                &ch,
                &lib_p_base,
                &lib_f,
                library_input,
                &p.mzmls[i],
                &d(&names[i]),
                p.top_peaks_ms2,
                p.max_spectra,
                shared,
                library_irt_repredicted,
                slices_from,
                irt_placeholder,
            ),
        }
    };

    let mut shared_ft: Option<String> = None;
    let mut first: usize = 0;
    // Opened just before the first run's chain when it runs alone under
    // `parallel_runs = auto`, so its peak sizes the chains after it.
    let mut first_window: Option<crate::sched::PeakWindow> = None;
    // What that window measured, for the log.
    let mut first_step = "the first run's chain";
    // The first run's adapted library when only its adaptation ran alone and its finish
    // joins the other runs' chains (below).
    let mut first_lib: Option<String> = None;
    if overlapped {
        // Run 1's front, then its RT adaptation beside the fronts of runs 2..N on disjoint
        // thread budgets, then every run's rest on the whole pool.
        let total = rayon::current_num_threads();
        let front_threads = overlap.min(total.saturating_sub(1)).max(1);
        let adapt_threads = total.saturating_sub(front_threads).max(1);
        let has_deeplc = cfg.predict_frag.deeplc_python.is_some();
        let mh_heads = cfg
            .rt_im_train
            .multihead_heads(has_deeplc, cfg.deeplc_rt_source(library_input, has_deeplc));
        info!(
            run = %names[0],
            n = n_runs,
            front_threads,
            adapt_threads,
            "run-experiment: converting and seeding the other runs while the first adapts \
             the library's retention times (experiment.overlap_front_threads)"
        );
        let out0 = d(&names[0]);
        let co0 = convert_run(cfg, &p.mzmls[0], &out0, p.top_peaks_ms2, p.max_spectra)?;
        let seed0 = seed_run(cfg, &ch, &lib_p_base, &lib_f, &co0, &out0, None)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(front_threads)
            .build()
            .context("building the thread pool for the overlapped runs")?;
        // Set when run 1's RT adaptation fails: the fronts of runs 2..N then stop before
        // their next convert or seed instead of doing N-1 full passes for a run that is
        // going to report the adaptation's error anyway. The reverse direction (a front
        // failing while the DeepLC worker runs) cannot interrupt the worker; the front's
        // error is logged at once, and returned after the worker exits.
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let cancelled = || {
            anyhow::anyhow!(
                "run-experiment: not converting and seeding the remaining runs because the \
                 first run's retention-time adaptation failed"
            )
        };
        let (adapted, fronts) = std::thread::scope(|scope| {
            let fronts = scope.spawn(|| {
                // The fronts keep the phase order of the ungrouped experiment above: every
                // conversion first, with no library resident, then one seed library for all
                // of their seeds (run 1 has already seeded on its own load).
                let r = pool.install(|| -> Result<Vec<(convert::ConvertOutputs, String)>> {
                    use std::sync::atomic::Ordering;
                    let mut conv: Vec<convert::ConvertOutputs> = Vec::with_capacity(n_runs - 1);
                    for (i, (name, mzml)) in names.iter().zip(p.mzmls).enumerate().skip(1) {
                        if cancel.load(Ordering::SeqCst) {
                            return Err(cancelled());
                        }
                        info!(run = %name, i = i + 1, n = n_runs, "run-experiment: convert, overlapped");
                        conv.push(convert_run(
                            cfg,
                            mzml,
                            &d(name),
                            p.top_peaks_ms2,
                            p.max_spectra,
                        )?);
                    }
                    if cancel.load(Ordering::SeqCst) {
                        return Err(cancelled());
                    }
                    let seed_lib = search_seed::SeedLibrary::load(
                        &lib_p_base,
                        &lib_f,
                        None,
                        None,
                        &cfg.search_seed,
                        cfg.extract.bucket_size,
                    )?;
                    let mut seeds: Vec<String> = Vec::with_capacity(n_runs - 1);
                    for (co, i) in conv.iter().zip(1..n_runs) {
                        if cancel.load(Ordering::SeqCst) {
                            return Err(cancelled());
                        }
                        info!(run = %names[i], i = i + 1, n = n_runs, "run-experiment: seed, overlapped");
                        seeds.push(seed_run(
                            cfg,
                            &ch,
                            &lib_p_base,
                            &lib_f,
                            co,
                            &d(&names[i]),
                            Some(&seed_lib),
                        )?);
                    }
                    drop(seed_lib);
                    Ok(conv.into_iter().zip(seeds).collect())
                });
                if let Err(e) = &r {
                    if !cancel.load(std::sync::atomic::Ordering::SeqCst) {
                        tracing::error!(
                            error = %format!("{e:#}"),
                            "run-experiment: an overlapped run's convert or seed failed; the \
                             first run's retention-time adaptation is still running and the \
                             experiment stops when it returns"
                        );
                    }
                }
                r
            });
            let adapted = adapt_rt_library(
                cfg,
                &lib_p_base,
                &seed0,
                &out0,
                mh_heads,
                None,
                irt_placeholder,
                adapt_threads,
                BootInputs {
                    ch: &ch,
                    lib_f: &lib_f,
                    co: &co0,
                },
            );
            if adapted.is_err() {
                cancel.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            let fronts = fronts
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("the overlapped runs' thread panicked")));
            (adapted, fronts)
        });
        let Adapted {
            lib: lib0,
            produced,
            ..
        } = adapted?;
        let fronts = fronts?;
        // Under `parallel_runs = auto`, the first run's finish is what every later run
        // repeats after its reused adaptation, so its peak alone sizes them.
        let window = plan.is_auto().then(crate::sched::PeakWindow::open);
        let (comp0, chrom0) = finish_run(
            cfg,
            &ch,
            &lib0,
            &lib_f,
            &co0,
            &seed0,
            &out0,
            true,
            Default::default(),
        )?;
        competed.push(vec![comp0]);
        chroms.push(vec![ChromTable::whole(&chrom0)]);
        if produced.is_none() {
            warn!(
                "run-experiment: the first run produced no adapted library; the remaining \
                 runs will each adapt their own"
            );
        }
        let back = |i: usize, (co, seed): &(convert::ConvertOutputs, String)| {
            let out = d(&names[i]);
            info!(run = %names[i], i = i + 1, n = n_runs, "run-experiment: per-run chain after the overlapped front");
            let lib_p = adapt_rt_library(
                cfg,
                &lib_p_base,
                seed,
                &out,
                mh_heads,
                produced.as_deref(),
                irt_placeholder,
                rayon::current_num_threads(),
                BootInputs {
                    ch: &ch,
                    lib_f: &lib_f,
                    co,
                },
            )?
            .lib;
            finish_run(
                cfg,
                &ch,
                &lib_p,
                &lib_f,
                co,
                seed,
                &out,
                true,
                Default::default(),
            )
        };
        let items: Vec<(usize, &(convert::ConvertOutputs, String))> =
            (1..n_runs).zip(fronts.iter()).collect();
        if plan.is_auto() {
            // Run 1 finished alone above, so its peak is the measurement the rest is sized
            // on (`sched::bound_by_measured_peak`), unless each of them adapts the library
            // itself in a DeepLC worker the measurement cannot see.
            plan = match (produced.is_none(), window) {
                (true, _) => {
                    crate::sched::one_at_a_time_for_sidecars(plan, threads_budget, "run-experiment")
                }
                (false, Some(w)) => crate::sched::bound_by_measured_peak(
                    plan,
                    threads_budget,
                    "run-experiment",
                    "the first run's chain",
                    w,
                ),
                (false, None) => plan,
            };
            let done = plan.map_pooled(&items, |&(i, f)| back(i, f))?;
            for (comp, chrom) in done {
                competed.push(vec![comp]);
                chroms.push(vec![ChromTable::whole(&chrom)]);
            }
        } else {
            for chunk in items.chunks(par) {
                let done: Vec<(String, String)> = if par == 1 {
                    chunk
                        .iter()
                        .map(|&(i, f)| back(i, f))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    chunk
                        .par_iter()
                        .map(|&(i, f)| back(i, f))
                        .collect::<Result<Vec<_>>>()?
                };
                for (comp, chrom) in done {
                    competed.push(vec![comp]);
                    chroms.push(vec![ChromTable::whole(&chrom)]);
                }
            }
        }
        first = n_runs;
    } else if share_ft {
        info!(
            run = %names[0],
            n = n_runs,
            "run-experiment: adapting the library's retention times on the first run only; \
             the remaining runs reuse that library and fit their own RT calibration on it \
             (experiment.rt_library_scope = per_run to adapt for every run instead)"
        );
        // Under `parallel_runs = auto` the first run's peak sizes the rest (below).
        let ft = match (&prepared[0], plan.is_auto()) {
            // A converted and seeded run adapts first. When that adaptation searched the
            // run (`rt_bootstrap`: extract, features and compete on the seed-anchored
            // library, all of the finish but retrace), the window around it measured a
            // search of this run, which sizes the chains as the whole first chain did, and
            // the first run's finish joins the other runs' chains below instead of running
            // alone. On the six HYE diaPASEF runs that alone time was about 5 of 48 min,
            // with one of the six per-run pools idle in the parallel phase after it.
            (Some((co0, seed0)), true) => {
                let out0 = d(&names[0]);
                let window = crate::sched::PeakWindow::open();
                let adapted = adapt_rt_library(
                    cfg,
                    &lib_p_base,
                    seed0,
                    &out0,
                    mh_heads_for_scope,
                    None,
                    irt_placeholder,
                    rayon::current_num_threads(),
                    BootInputs {
                        ch: &ch,
                        lib_f: &lib_f,
                        co: co0,
                    },
                )?;
                first_window = Some(window);
                if adapted.searched {
                    first_step = "the first run's retention-time adaptation (a search of it)";
                    first_lib = Some(adapted.lib);
                } else {
                    // Nothing of a search ran in this process yet, so the finish runs
                    // alone and is measured, as before.
                    let (comp, chrom) = finish_run(
                        cfg,
                        &ch,
                        &adapted.lib,
                        &lib_f,
                        co0,
                        seed0,
                        &out0,
                        true,
                        Default::default(),
                    )?;
                    competed.push(vec![comp]);
                    chroms.push(vec![ChromTable::whole(&chrom)]);
                    first = 1;
                }
                adapted.produced
            }
            _ => {
                first_window = plan.is_auto().then(crate::sched::PeakWindow::open);
                let (comp, chrom, ft) = run_one(0, None, None)?;
                competed.push(comp);
                chroms.push(chrom);
                first = 1;
                ft
            }
        };
        match ft {
            Some(path) => {
                info!(library = %path, "run-experiment: reusing this adapted library for the remaining runs");
                shared_ft = Some(path);
            }
            // Defensive: `share_ft` implies the first run fine-tunes, so this should not
            // happen. Falling through with `None` means the rest fine-tune themselves --
            // slower, but never silently wrong.
            None => warn!(
                "run-experiment: the first run produced no fine-tuned library; the \
                 remaining runs will each fine-tune"
            ),
        }
    }

    // A grouped run writes a band out only where a DeepLC sidecar rewrites it, and the runs
    // of one experiment plan the same bands over the same library, so a run after the
    // first takes the first grouped run's slices where the plans agree (`run_groups`).
    let grouped = cfg.groups.window_groups > 1;
    let mut slice_source: Option<String> = if grouped { shared_ft.clone() } else { None };
    let rest: Vec<usize> = (first..n_runs).collect();
    if plan.is_auto() && first < n_runs {
        // Under `parallel_runs = auto` the chains are pulled from a queue, each in its own
        // pool. A chain that adapts the library itself (no adapted library to share) runs
        // a DeepLC worker no memory reading of this process includes, so those chains run
        // one at a time. Otherwise the first run, when it or its adaptation ran alone
        // above, sizes the rest; and when no chain has run alone yet and there is a memory
        // reading to take, the first one runs alone now so the rest are sized on its
        // measured peak. Without a reading that would only cost concurrency.
        let mut queue: &[usize] = &rest;
        if adapts_rt_library && shared_ft.is_none() {
            plan = crate::sched::one_at_a_time_for_sidecars(plan, threads_budget, "run-experiment");
        } else if let Some(w) = first_window {
            plan = crate::sched::bound_by_measured_peak(
                plan,
                threads_budget,
                "run-experiment",
                first_step,
                w,
            );
        } else if first == 0
            && plan.par > 1
            && rest.len() > 1
            && crate::sched::memory_reading().is_some()
        {
            let i = rest[0];
            info!(run = %names[i], i = i + 1, n = n_runs, "run-experiment: per-run chain, alone to size the rest (parallel_runs = auto)");
            // On the engine's whole pool, exactly as the sequential loop runs it, from a
            // reset high-water mark so the conversions and seeds before it are not counted.
            let window = crate::sched::PeakWindow::open();
            let (comp, chrom, groups_dir) =
                run_one(i, shared_ft.as_deref(), slice_source.as_deref())?;
            competed.push(comp);
            chroms.push(chrom);
            if grouped && slice_source.is_none() {
                slice_source = groups_dir;
            }
            plan = crate::sched::bound_by_measured_peak(
                plan,
                threads_budget,
                "run-experiment",
                "the first chain",
                window,
            );
            queue = &rest[1..];
        }
        info!(
            parallel_runs = plan.par,
            pool_threads = plan.pool_threads.unwrap_or(0),
            n = n_runs,
            "run-experiment: per-run chains, each in a pool of its own (parallel_runs = auto)"
        );
        // The chains were sized on the peak of a chain, or of a search of the first run,
        // that ran alone and took all of its feature loaders (and, a whole chain, its
        // retrace readers, about 0.1 GB each). The process-wide pools of both hold enough
        // for a few concurrent calls only, so they grow by the chains in flight while the
        // chains run (`sched::BudgetGrowth`). Neither changes a row: the loaders and readers
        // only decode, and the tables keep their order.
        let _loaders = (plan.par > 1)
            .then(|| {
                crate::stages::features::grow_loaders_for_chains(
                    cfg.features.chrom_loaders,
                    plan.par,
                )
            })
            .flatten();
        let _readers = (plan.par > 1)
            .then(|| crate::stages::retrace::grow_readers_for_chains(plan.par))
            .flatten();
        // A grouped run's band slices are reused by the runs that START after it finished;
        // which run that is depends on timing, as it depended on the chunk size before,
        // and a reused slice is the same bytes a run would have written (`run_groups`).
        let slices = std::sync::Mutex::new(slice_source.clone());
        let done = plan.map_pooled(queue, |&i| {
            // The first run's finish, on the library its adaptation produced above.
            if let (0, Some(lib0), Some((co0, seed0))) = (i, first_lib.as_deref(), &prepared[0]) {
                info!(run = %names[0], i = 1, n = n_runs, "run-experiment: per-run chain, the first run's finish on its adapted library");
                let (comp, chrom) = finish_run(
                    cfg,
                    &ch,
                    lib0,
                    &lib_f,
                    co0,
                    seed0,
                    &d(&names[0]),
                    true,
                    Default::default(),
                )?;
                return Ok((vec![comp], vec![ChromTable::whole(&chrom)], None));
            }
            info!(run = %names[i], i = i + 1, n = n_runs, "run-experiment: per-run chain");
            let from = slices.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let r = run_one(i, shared_ft.as_deref(), from.as_deref())?;
            if grouped {
                let mut slot = slices.lock().unwrap_or_else(|e| e.into_inner());
                if slot.is_none() {
                    slot.clone_from(&r.2);
                }
            }
            Ok(r)
        })?;
        for (comp, chrom, _) in done {
            competed.push(comp);
            chroms.push(chrom);
        }
    } else if par == 1 {
        for &i in &rest {
            info!(run = %names[i], i = i + 1, n = n_runs, "run-experiment: per-run chain");
            let (comp, chrom, groups_dir) =
                run_one(i, shared_ft.as_deref(), slice_source.as_deref())?;
            competed.push(comp);
            chroms.push(chrom);
            if grouped && slice_source.is_none() {
                slice_source = groups_dir;
            }
        }
    } else {
        info!(
            parallel_runs = par,
            n = n_runs,
            "run-experiment: per-run chains in parallel (each run can hold tens of GB; \
             lower experiment.parallel_runs if memory is tight)"
        );
        for chunk in rest.chunks(par) {
            let done: Vec<(Vec<String>, Vec<ChromTable>, Option<String>)> = chunk
                .par_iter()
                .map(|&i| {
                    info!(run = %names[i], i = i + 1, n = n_runs, "run-experiment: per-run chain");
                    run_one(i, shared_ft.as_deref(), slice_source.as_deref())
                })
                .collect::<Result<Vec<_>>>()?;
            for (comp, chrom, groups_dir) in done {
                competed.push(comp);
                chroms.push(chrom);
                if grouped && slice_source.is_none() {
                    slice_source = groups_dir;
                }
            }
        }
    }

    // --- one experiment-wide rescore over all competed tables ---
    //
    // Every table of run i carries source i. When each run is one table (every run
    // ungrouped, or grouped with its competed table pooled) the source is the table's own
    // index, as it always was, and the rescore report is what it was.
    let rescore_inputs: Vec<String> = competed.iter().flatten().cloned().collect();
    let rescore_sources: Vec<u32> = competed
        .iter()
        .enumerate()
        .flat_map(|(i, tables)| std::iter::repeat_n(i as u32, tables.len()))
        .collect();
    let one_table_per_run = competed.iter().all(|tables| tables.len() == 1);
    let shadow_windows: Vec<String> = names
        .iter()
        .map(|nm| format!("{}/spectra/isolation_windows.parquet", d(nm)))
        .collect();
    // Each run's extracted rows, for the shadow test's 1/K0 term
    // (`rescore.shadow_im_tolerance`).
    let shadow_psms: Vec<String> = names
        .iter()
        .map(|nm| format!("{}/psms_extracted.parquet", d(nm)))
        .collect();
    // Under `mbr.strategy = second_pass` (the default) this rescore is the first pass, kept
    // under `first_pass/`, and the experiment's `scored_combined.parquet` is the second pass's.
    let second_pass_on = cfg.mbr.strategy == mumdia_core::config::MbrStrategy::SecondPass;
    let (first_scored, first_work) = if second_pass_on {
        let dir = d("first_pass");
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {dir}"))?;
        (
            format!("{dir}/scored_combined.parquet"),
            format!("{dir}/sidecar_work"),
        )
    } else {
        (d("scored_combined.parquet"), d("sidecar_work"))
    };
    let first_written = rescore::run_hashed(rescore::RescoreParams {
        competed: &rescore_inputs,
        sources: (!one_table_per_run).then_some(rescore_sources.as_slice()),
        out: &first_scored,
        work_dir: &rescore::sidecar_work_dir(&first_work),
        script_dir: &cfg.predict_frag.sidecar_script_dir,
        cfg: &cfg.rescore,
        config_hash: &ch,
        shadow: Some(
            crate::stages::shadow::ShadowInputs::new(&lib_f, &shadow_windows)
                .with_psms_extracted(&shadow_psms),
        ),
    })?;

    // --- match-between-runs: the second pass ---
    let mut second = if second_pass_on {
        let run_dirs: Vec<String> = names.iter().map(|nm| d(nm)).collect();
        search_second_pass(
            cfg,
            &ch,
            &first_scored,
            &lib_p_base,
            &lib_f,
            &names,
            &run_dirs,
            &d("second_pass"),
            &plan,
            par,
        )?
    } else {
        None
    };
    if second_pass_on && second.is_none() {
        warn!(
            "run-experiment: the second pass was not applied (see above); the first pass is the \
             result"
        );
    }
    let (scored_combined, scored_written) = match &mut second {
        Some(sp) => {
            let scored = d("scored_combined.parquet");
            let written = rescore::run_hashed(rescore::RescoreParams {
                competed: &sp.competed,
                sources: None,
                out: &scored,
                work_dir: &rescore::sidecar_work_dir(&d("sidecar_work")),
                script_dir: &cfg.predict_frag.sidecar_script_dir,
                cfg: &cfg.rescore,
                config_hash: &ch,
                // The second pass's own extracted rows: its candidate ids are the empirical
                // library's, not the first pass's.
                shadow: Some(
                    crate::stages::shadow::ShadowInputs::new(
                        &sp.library.fragments,
                        &shadow_windows,
                    )
                    .with_psms_extracted(&sp.psms_extracted),
                ),
            })?;
            // The second pass against the first: logged always, and written to
            // `empirical_library.json` on an experiment with ion mobility.
            let diag = second_pass::diagnose(&sp.library, &scored, names.len())?;
            if sp.has_im {
                second_pass::record_diagnostics(&mut sp.library, diag)?;
            }
            chroms = sp.chroms.iter().map(|c| vec![c.clone()]).collect();
            (scored, written)
        }
        // The experiment's `scored_combined.parquet` is the result either way: the first pass,
        // copied up from `first_pass/` with its report when the second pass did not apply.
        None if second_pass_on => {
            let scored = d("scored_combined.parquet");
            for (src, dst) in [
                (first_scored.clone(), scored.clone()),
                (
                    format!("{first_scored}.report.json"),
                    format!("{scored}.report.json"),
                ),
            ] {
                std::fs::copy(&src, &dst)
                    .with_context(|| format!("copying the first pass {src} to {dst}"))?;
            }
            (scored, first_written)
        }
        None => (first_scored.clone(), first_written),
    };
    // The classifier that actually ran, from the rescore artifact report: the source of
    // truth, since the configured enum can differ from it under a compatibility path.
    let rescore_report: mumdia_io::report::ArtifactReport =
        mumdia_io::json::read_json(&format!("{scored_combined}.report.json"))?;
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

    // --- optional rescuable-tier MBR transfer ---
    let scored_for_quant = if cfg.mbr.strategy.transfers() {
        let python = cfg.mbr.python.as_deref().expect("preflight");
        let script =
            crate::sidecar::resolve_script(&cfg.predict_frag.sidecar_script_dir, "mbr_worker.py");
        let transferred = d("mbr_transferred.parquet");
        let scored_mbr = d("scored_mbr.parquet");
        // Remove any stale copy BEFORE the worker runs, so that the `exists()` check
        // afterwards is a statement about THIS run. The worker writes this file only when
        // it accepts at least one transfer, so without the removal a rerun into the same
        // --out-dir that accepted transfers last time and none this time would silently
        // feed the previous experiment's scored table to the split and to every per-run
        // quant. It joins cleanly, because candidate_id and source are stable across
        // reruns of the same library, so the failure is plausible numbers from the wrong
        // data, logged as "MBR transfers applied".
        if std::path::Path::new(&scored_mbr).exists() {
            std::fs::remove_file(&scored_mbr).with_context(|| {
                format!("removing a previous run's {scored_mbr} before running MBR")
            })?;
        }
        // The competed tables carry candidate_id + apex_rt in `source` order, one per run:
        // match-between-runs keeps a grouped run's competed table pooled (`run_groups`).
        if !one_table_per_run {
            anyhow::bail!(
                "match-between-runs reads one competed table per run, and a grouped run left \
                 its competed rows per band"
            );
        }
        let mbr_competed: Vec<String> = competed.iter().map(|t| t[0].clone()).collect();
        crate::sidecar::run_mbr(
            python,
            &script,
            &scored_combined,
            &mbr_competed,
            &transferred,
            Some(&scored_mbr),
            &[],
            cfg.mbr.q_anchor,
            cfg.mbr.min_anchor_runs,
            cfg.mbr.q_transfer,
            cfg.mbr.consensus_corr_min,
            cfg.rng_seed,
        )?;
        // The worker writes the augmented scored table only when it accepts at
        // least one transfer; with none, fall back to the un-augmented combined
        // table so quantification still runs.
        //
        // `exists()` is only a signal because the stale file was removed before the
        // worker ran (see above). Without that, it could not tell "this run wrote it"
        // from "a previous run left it": rerunning the same --out-dir with different
        // mzMLs or a tighter mbr.q_transfer, and finding no transfers this time, made
        // the split and every per-run quant consume the OLD scored table. It joins
        // successfully, because candidate_id and source are stable across reruns of the
        // same library, so the result was plausible quantities from the wrong data while
        // the log said "MBR transfers applied".
        if std::path::Path::new(&scored_mbr).exists() {
            info!("run-experiment: MBR transfers applied");
            scored_mbr
        } else {
            info!("run-experiment: MBR accepted no transfers; using combined scored");
            scored_combined.clone()
        }
    } else {
        scored_combined.clone()
    };

    // --- split by source, per-run quant (psm_q filter), cross-run LFQ ---
    let split_paths: Vec<String> = (0..n_runs)
        .map(|i| d(&format!("{}/scored.parquet", names[i])))
        .collect();
    let split_written = split_by_source(&scored_for_quant, &split_paths)?;

    let mut qcfg = cfg.quant.clone();
    // Per-run quant gates on the pooled `q_value`, not on whatever `quant.q_filter` says. The
    // grouped q columns (peptide_q_value / precursor_q / pg_q_value) are assigned only to the
    // single experiment-wide winning row of each group (see `grouped_q` in rescore.rs), so a
    // per-run table can only gate on a per-PSM column. `run_psm_q` would be the natural choice,
    // but the pooled column is what the cross-run LFQ step downstream assumes.
    //
    // Warn rather than override in silence: a user who explicitly configured a different
    // `q_filter` otherwise gets quantities gated on a column they did not select, and there is
    // no record of the substitution in any artifact. Changing `q_filter` also does not select a
    // source, which is a separate documented trap.
    if !matches!(qcfg.q_filter, QuantQColumn::PsmQ) {
        warn!(
            configured = ?qcfg.q_filter,
            used = ?QuantQColumn::PsmQ,
            "run-experiment: per-run quant ignores the configured quant.q_filter and gates on the \
             pooled q_value; grouped q columns exist only on each group's experiment-wide winner"
        );
    }
    qcfg.q_filter = QuantQColumn::PsmQ;
    // Both quant loops run under the experiment's memory-bounded plan, each run in its own
    // pool share, with the results in run order. An explicit `parallel_runs` keeps quant
    // serial, as it was. Each loop reads `chroms`, which after a second pass are the second
    // pass's chromatograms: the ones the reported quantities come from.
    let run_idx: Vec<usize> = (0..n_runs).collect();
    let quant_plan = if plan.is_auto() {
        plan
    } else {
        crate::sched::RunConcurrency {
            par: 1,
            pool_threads: None,
        }
    };
    // --- the cross-run step (`quant.needs_cross_run_fit`) ---
    // Pass 1 quantifies every run with its per-run settings and writes its
    // `fragment_quant.parquet` as the cross-run export; the fit over those tables is applied
    // by pass 2, below, in every run. Pass 1's peptide and protein tables are temporary.
    let cross_run = if qcfg.needs_cross_run_fit() {
        let frag_tables = quant_plan.map_pooled(&run_idx, |&i| {
            let ft = d(&format!("{}/fragment_quant.parquet", names[i]));
            let tmp = |n: &str| d(&format!("{}/{n}.pass1.parquet", names[i]));
            let written = quant::run_hashed(quant::QuantParams {
                psms_scored: &split_paths[i],
                chromatograms: &chroms[i],
                out_peptide: &tmp("peptide_quant"),
                out_protein: &tmp("protein_group_quant"),
                out_fragment: Some(&ft),
                out_peak_bounds: None,
                cfg: &qcfg,
                config_hash: &ch,
                cross_run: None,
                export_cross_run: true,
            })?;
            for n in ["peptide_quant", "protein_group_quant"] {
                for path in [tmp(n), format!("{}.report.json", tmp(n))] {
                    std::fs::remove_file(&path)
                        .with_context(|| format!("removing the first-pass table {path}"))?;
                }
            }
            Ok((ft, written.fragment.expect("out_fragment was set")))
        })?;
        let paths: Vec<String> = frag_tables.iter().map(|(p, _)| p.clone()).collect();
        let fit = quant::fit_cross_run(&paths, &qcfg)?;
        info!(
            weights = fit.weights.as_ref().map(|w| w.len()),
            background = fit.background.as_ref().map(|b| b.len()),
            halfwidth = fit.halfwidth.as_ref().map(|h| h.len()),
            "run-experiment: cross-run quant fit (candidates)"
        );
        Some((fit, frag_tables))
    } else {
        None
    };
    // Each run's quant hashes its two tables for their reports; the manifest below reuses
    // those hashes instead of reading the tables again.
    let finals = quant_plan.map_pooled(&run_idx, |&i| {
        let pq = d(&format!("{}/peptide_quant.parquet", names[i]));
        let gq = d(&format!("{}/protein_group_quant.parquet", names[i]));
        let written = quant::run_hashed(quant::QuantParams {
            psms_scored: &split_paths[i],
            chromatograms: &chroms[i],
            out_peptide: &pq,
            out_protein: &gq,
            out_fragment: None,
            out_peak_bounds: None,
            cfg: &qcfg,
            config_hash: &ch,
            cross_run: cross_run.as_ref().map(|(fit, _)| fit),
            export_cross_run: false,
        })?;
        Ok((pq, gq, written))
    })?;
    let mut peptide_quants: Vec<String> = Vec::with_capacity(n_runs);
    let mut protein_quants: Vec<String> = Vec::with_capacity(n_runs);
    let mut quant_written: Vec<quant::QuantWritten> = Vec::with_capacity(n_runs);
    for (pq, gq, written) in finals {
        peptide_quants.push(pq);
        protein_quants.push(gq);
        quant_written.push(written);
    }
    let lfq = d("lfq_maxlfq.parquet");
    let n_lfq = quant::run_lfq_combine(
        &peptide_quants,
        false,
        mumdia_core::config::NormalizeMethod::MedianRatio,
        &lfq,
    )?;

    // --- experiment-wide report ---
    // peptides.tsv and proteins.tsv at the experiment root, selected on the
    // experiment-wide grouped q columns (the one unit that is valid across a pooled
    // rescore), with one quantity column per run: each run's own precursor quantity, and
    // the cross-run MaxLFQ protein quantity. There is deliberately no per-run TSV: the
    // grouped q columns are written to each group's experiment-wide winner only, so a
    // per-run report would be diluted by about 1/n_runs; the per-run unit is `run_psm_q`
    // in the split tables.
    let pep_tsv = d("peptides.tsv");
    let prot_tsv = d("proteins.tsv");
    let (n_rep_pep, n_rep_prot) = report::run_experiment(report::ExperimentReportParams {
        scored: &scored_for_quant,
        run_names: &names,
        peptide_quants: &peptide_quants,
        protein_lfq: Some(&lfq),
        out_peptides: &pep_tsv,
        out_proteins: &prot_tsv,
        q_threshold: cfg.quant.q_threshold,
    })?;

    // --- experiment manifest ---
    let mut manifest = json!({
        "config_hash": ch,
        "n_runs": n_runs,
        "runs": names,
        "scored_combined": scored_combined,
        "scored_for_quant": scored_for_quant,
        "rescorer": actual_rescorer,
        "mbr": format!("{:?}", cfg.mbr.strategy),
        // The second pass of match-between-runs: the empirical library it searched and the
        // first pass it was built from. `null` when it did not run.
        "second_pass": second.as_ref().map(|sp| json!({
            "first_pass_scored": first_scored,
            "library": sp.library,
            "runs": names.iter().map(|nm| d(&format!("second_pass/{nm}"))).collect::<Vec<_>>(),
        })),
        // Per-run quant gates on the pooled q_value whatever the configuration says
        // (see the warning above). The configuration hash is of the configuration as
        // given, so the substitution has to be recorded here or it is recorded nowhere.
        "quant_q_filter": {
            "configured": format!("{:?}", cfg.quant.q_filter),
            "effective": format!("{:?}", qcfg.q_filter),
        },
        "lfq": lfq,
        "peptide_quants": peptide_quants,
        "report": {
            "peptides": pep_tsv,
            "proteins": prot_tsv,
            "n_precursors": n_rep_pep,
            "n_protein_groups": n_rep_prot,
            "q_threshold": cfg.quant.q_threshold,
            "unit": "experiment-wide peptide_q_value / pg_q_value; n_runs on run_psm_q",
        },
    });
    // The cross-run quant step and its first-pass fragment tables, only when it ran, so
    // every other manifest keeps its bytes.
    if let (Some((_, tables)), Some(m)) = (&cross_run, manifest.as_object_mut()) {
        m.insert(
            "cross_run_quant".to_string(),
            json!({
                "fragment_quants": tables.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>(),
            }),
        );
    }
    // Provenance parity with the single-run manifest: the identity of the code, of
    // the inputs (hashed at the start of the run, see above), and of every artifact
    // this stage produced.
    //
    // The per-artifact records were the gap. The experiment manifest listed output
    // PATHS in its `experiment` block and nothing else, so an experiment result had
    // no content hash, no row count and no schema version anywhere -- exactly the
    // three things the single-run manifest exists to provide, and the three things
    // needed to tell whether two experiment outputs are the same data. Files written
    // by the per-run chains are not covered here (those chains do not thread a shared
    // manifest); what is covered is everything `run-experiment` itself writes.
    //
    // Model identities reflect the path that produced the downstream artifacts, as
    // in the single-run manifest (docs/29 #15): which RT source the library carried,
    // which fragment predictor, and the classifier that actually ran.
    let library_input = p.lib_precursors.is_some();
    let deeplc_py = cfg.predict_frag.deeplc_python.as_deref();
    // Resolved the same way the stage itself resolved it, so the manifest records what
    // ran rather than what was configured.
    let mh_heads = cfg.rt_im_train.multihead_heads(
        deeplc_py.is_some(),
        cfg.deeplc_rt_source(library_input, deeplc_py.is_some()),
    );
    // `auto` decides per adapting run. Under `first_run_only` that is run 1, whose library
    // records the choice; under `per_run` the runs may differ, and each library says which.
    let auto_search = mh_heads > 0
        && cfg.rt_im_train.multihead_anchors == MultiheadAnchors::Auto
        && cfg.groups.window_groups <= 1
        && match cfg.experiment.rt_library_scope {
            RtLibraryScope::FirstRunOnly => {
                rt_bootstrap::anchor_source(&format!(
                    "{}/{}",
                    d(&names[0]),
                    rt_bootstrap::MULTIHEAD_LIBRARY
                ))
                .as_deref()
                    == Some("search")
            }
            _ => false,
        };
    let rt_identity = if rt_bootstrap::always(cfg, mh_heads) || auto_search {
        // Requested (`search`) or chosen for run 1 (`auto`); whether a run refitted or fell
        // back is in its adapted library's `.summary.json` (`anchor_source`).
        crate::sidecar::deeplc_identity(deeplc_py, &format!("multihead-{mh_heads}-search-anchors"))
    } else if mh_heads > 0
        && cfg.rt_im_train.multihead_anchors == MultiheadAnchors::Auto
        && cfg.groups.window_groups <= 1
        && !matches!(
            cfg.experiment.rt_library_scope,
            RtLibraryScope::FirstRunOnly
        )
    {
        crate::sidecar::deeplc_identity(deeplc_py, &format!("multihead-{mh_heads}-auto-anchors"))
    } else if mh_heads > 0 {
        crate::sidecar::deeplc_identity(deeplc_py, &format!("multihead-{mh_heads}"))
    } else if cfg.rt_im_train.finetune_deeplc {
        if matches!(
            cfg.experiment.rt_library_scope,
            RtLibraryScope::FirstRunOnly
        ) {
            crate::sidecar::deeplc_identity(deeplc_py, "finetuned-first-run")
        } else {
            crate::sidecar::deeplc_identity(deeplc_py, "finetuned-per-run")
        }
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
    prov.model_identities
        .insert("rt_predictor".into(), rt_identity);
    prov.model_identities
        .insert("fragment_predictor".into(), fragment_identity);
    prov.model_identities
        .insert("rescorer".into(), actual_rescorer_model);
    prov.model_identities.insert(
        "feature_schema_id".into(),
        features::feature_schema_id(&features::active_features_of(cfg, false)),
    );
    prov.model_identities.insert(
        "mbr".into(),
        if second_pass_on && second.is_none() {
            "SecondPass-not-applied".to_string()
        } else {
            format!("{:?}", cfg.mbr.strategy)
        },
    );
    // Recorded in a fixed order. `Manifest` stores them in a BTreeMap, so the serialised
    // order is by logical name and does not depend on this sequence. The last field is the
    // content hash when the producing stage already computed it (rescore and quant for their
    // reports, the by-source split while it wrote its tables); the MBR worker's table is
    // hashed here.
    // (logical name, schema, path, producing stage, content hash when already known)
    type Recorded<'s> = (String, (&'s str, u32), String, &'s str, Option<String>);
    let mut artifacts: Vec<Recorded> = vec![(
        "scored_combined".to_string(),
        artifact::PSMS_SCORED,
        scored_combined.clone(),
        "rescore",
        Some(scored_written.content_hash.clone()),
    )];
    if let Some(sp) = &second {
        artifacts.push((
            "scored_first_pass".to_string(),
            artifact::PSMS_SCORED,
            first_scored.clone(),
            "rescore",
            None,
        ));
        artifacts.push((
            "empirical_library_precursors".to_string(),
            crate::stages::predict_frag::precursors_schema(&sp.library.precursors)?,
            sp.library.precursors.clone(),
            "second-pass",
            None,
        ));
        artifacts.push((
            "empirical_library_fragments".to_string(),
            artifact::FRAGMENT_LIBRARY_FRAGMENTS,
            sp.library.fragments.clone(),
            "second-pass",
            None,
        ));
    }
    // Only when MBR actually produced a different table; otherwise
    // `scored_for_quant` IS `scored_combined` and recording it twice would claim two
    // artifacts where one file exists.
    if scored_for_quant != scored_combined {
        artifacts.push((
            "scored_for_quant".to_string(),
            artifact::PSMS_SCORED,
            scored_for_quant.clone(),
            "mbr",
            None,
        ));
    }
    for (i, name) in names.iter().enumerate() {
        artifacts.push((
            format!("scored[{name}]"),
            artifact::PSMS_SCORED,
            split_paths[i].clone(),
            "split-by-source",
            Some(split_written[i].content_hash.clone()),
        ));
        artifacts.push((
            format!("peptide_quant[{name}]"),
            artifact::PEPTIDE_QUANT,
            peptide_quants[i].clone(),
            "quant",
            Some(quant_written[i].peptide.content_hash.clone()),
        ));
        artifacts.push((
            format!("protein_group_quant[{name}]"),
            artifact::PROTEIN_GROUP_QUANT,
            protein_quants[i].clone(),
            "quant",
            Some(quant_written[i].protein.content_hash.clone()),
        ));
        // The cross-run export of pass 1, written only when the step ran.
        if let Some((_, tables)) = &cross_run {
            artifacts.push((
                format!("fragment_quant[{name}]"),
                artifact::fragment_quant(true),
                tables[i].0.clone(),
                "quant",
                Some(tables[i].1.content_hash.clone()),
            ));
        }
    }
    for (logical, schema, path, stage, known_hash) in artifacts {
        let rows = mumdia_io::table::nrows(&path)
            .with_context(|| format!("counting rows of {path} for the experiment manifest"))?;
        prov.record(match known_hash {
            Some(hash) => mumdia_io::record_artifact_with_hash(
                &logical, schema, &path, rows, stage, &ch, hash,
            ),
            None => mumdia_io::record_artifact(&logical, schema, &path, rows, stage, &ch)?,
        });
    }
    prov.record(mumdia_io::record_artifact(
        artifact::LFQ_MAXLFQ.0,
        artifact::LFQ_MAXLFQ,
        &lfq,
        n_lfq,
        "quant-lfq",
        &ch,
    )?);

    // The input hashes started at the top of the experiment.
    input_hashes.record(&mut prov);

    // The resolved configuration itself, not only its hash: a hash identifies a
    // configuration but cannot replay one (docs/29 #15).
    let manifest = serde_json::json!({
        "mumdia_version": prov.mumdia_version,
        "git_sha": prov.git_sha,
        "commit_date": prov.commit_date,
        "cli_args": prov.cli_args,
        "config_hash": prov.config_hash,
        "config_json": prov.config_json,
        "model_identities": prov.model_identities,
        "inputs": prov.inputs,
        "inputs_hashed_at": "start",
        "artifacts": prov.artifacts,
        "experiment": manifest,
    });
    mumdia_io::json::write_json(&d("experiment_manifest.json"), &manifest)?;
    info!(
        elapsed_ms = t0.elapsed().as_millis(),
        n_runs,
        report_precursors = n_rep_pep,
        report_protein_groups = n_rep_prot,
        "run-experiment: complete"
    );
    // The DeepLC projection cache may have grown during the experiment.
    crate::cache::enforce_for(cfg);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::{write_table, Col, Table};

    /// The cross-run quant step is refused where candidate_id stability across runs has not
    /// been checked, and only when one of its keys asks for it.
    #[test]
    fn the_cross_run_step_is_refused_where_candidate_ids_are_unchecked() {
        let on = r#""quant":{"fragment_selection":"cross_run"}"#;
        for (json, ok) in [
            ("{}".to_string(), true),
            (format!("{{{on}}}"), true),
            (r#"{"groups":{"window_groups":2}}"#.to_string(), true),
            (
                r#"{"experiment":{"rt_library_scope":"per_run"}}"#.to_string(),
                true,
            ),
            (format!(r#"{{{on},"groups":{{"window_groups":2}}}}"#), false),
            (
                format!(r#"{{{on},"experiment":{{"rt_library_scope":"per_run"}}}}"#),
                false,
            ),
        ] {
            let cfg = Config::from_json(&json).unwrap();
            assert_eq!(refuse_unchecked_cross_run_step(&cfg).is_ok(), ok, "{json}");
        }
    }

    fn names(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn run_names_that_differ_only_in_case_are_rejected() {
        // One directory on Windows, macOS and most network shares (docs/29 #5): the
        // check refuses it everywhere rather than probing the destination filesystem.
        let e = check_run_names_distinct(&names(&["RunA", "runa"]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("differ only in case"), "{e}");
        assert!(e.contains("RunA") && e.contains("runa"), "{e}");
    }

    #[test]
    fn run_names_that_alias_on_windows_are_rejected_everywhere() {
        // docs/30 R2: `a` and `a.` are one directory on Windows and the experiment ran
        // to completion with one run overwriting the other. The syntactic rules apply on
        // every platform so the output is portable.
        for (bad, why) in [
            ("a.", "dot or a space"),
            ("a ", "dot or a space"),
            ("NUL", "reserved"),
            ("com1.log", "reserved"),
            ("run:1", "forbids"),
            ("run?", "forbids"),
            ("a/b", "separator"),
            ("", "empty"),
            ("..", "not names"),
        ] {
            let why_got = portable_dir_name_problem(bad)
                .unwrap_or_else(|| panic!("{bad:?} must be rejected"));
            assert!(why_got.contains(why), "{bad:?}: {why_got}");
        }
        for ok in ["r0", "run.1", "A-b_c", "B_01", "sample 3", "com10", "conx"] {
            assert_eq!(
                portable_dir_name_problem(ok),
                None,
                "{ok:?} must be accepted"
            );
        }
    }

    #[test]
    fn repeated_run_names_are_rejected_and_distinct_ones_pass() {
        let e = check_run_names_distinct(&names(&["a", "b", "a"]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("given twice"), "{e}");
        check_run_names_distinct(&names(&["a", "b", "c_1"])).unwrap();
    }

    fn tmp(name: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!("mumdia_runexp_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        dir.join(format!("{n}_{name}"))
            .to_str()
            .unwrap()
            .to_string()
    }

    fn scored_with_sources(sources: Vec<u32>) -> String {
        let n = sources.len();
        let path = tmp("scored.parquet");
        write_table(
            &path,
            vec![
                Col::U32("source".into(), sources),
                Col::U32("candidate_id".into(), (0..n as u32).collect()),
                Col::F64("q_value".into(), vec![0.001; n]),
            ],
        )
        .unwrap();
        path
    }

    /// A scored-like table in row groups of `rg_rows`, with a string and a float column
    /// beside `source`, so a splice has more than one leaf type to carry.
    fn scored_in_groups(sources: &[u32], rg_rows: usize) -> String {
        let n = sources.len();
        let path = tmp("scored_rg.parquet");
        let mut w = mumdia_io::table::TableWriter::new(&path).with_row_group_rows(rg_rows);
        w.write_cols(vec![
            Col::U32("source".into(), sources.to_vec()),
            Col::U32("candidate_id".into(), (0..n as u32).collect()),
            Col::Str(
                "peptidoform".into(),
                (0..n).map(|i| format!("PEP{i}K")).collect(),
            ),
            Col::F64("q_value".into(), (0..n).map(|i| i as f64 * 0.001).collect()),
        ])
        .unwrap();
        w.close().unwrap();
        path
    }

    /// Every row of `path`, as comparable tuples.
    fn rows_of(path: &str) -> Vec<(u32, u32, String, u64)> {
        let t = Table::read(path).unwrap();
        let src = t.u32("source").unwrap();
        let cid = t.u32("candidate_id").unwrap();
        let pf = t.str("peptidoform").unwrap();
        let q = t.f64("q_value").unwrap();
        (0..t.nrows)
            .map(|i| (src[i], cid[i], pf[i].clone(), q[i].to_bits()))
            .collect()
    }

    #[test]
    fn the_spliced_split_holds_the_rewritten_splits_rows() {
        // Runs of 0, 1 and 2 in row groups of 4: groups 0-1 are run 0 only, group 2 holds
        // the 0/1 boundary, group 3 is run 1 only, group 4 holds the 1/2 boundary, the rest
        // is run 2. Run 3 has no rows.
        let mut sources = vec![0u32; 10];
        sources.extend(vec![1u32; 7]);
        sources.extend(vec![2u32; 9]);
        let scored = scored_in_groups(&sources, 4);
        let spliced: Vec<String> = (0..4).map(|i| tmp(&format!("sp{i}.parquet"))).collect();
        let written = split_by_source(&scored, &spliced).unwrap();
        let t = mumdia_io::table::TableFile::open(&scored).unwrap();
        let rewritten: Vec<String> = (0..4).map(|i| tmp(&format!("rw{i}.parquet"))).collect();
        let (_, placed) =
            split_rewritten(&t, t.schema.index_of("source").unwrap(), &rewritten).unwrap();
        assert_eq!(placed, sources.len());
        for i in 0..4 {
            assert_eq!(rows_of(&spliced[i]), rows_of(&rewritten[i]), "run {i}");
            // The hash is the file's, computed while it was written.
            assert_eq!(
                written[i].content_hash,
                mumdia_io::hash::blake3_file(&spliced[i]).unwrap(),
                "run {i}"
            );
            assert_eq!(written[i].rows as usize, rows_of(&spliced[i]).len());
        }
        // The single-source groups were spliced, not re-encoded: run 0's table keeps the
        // input's two whole groups of 4 and then its 2 boundary rows.
        let groups = |p: &str| {
            mumdia_io::table::SpliceWriter::row_group_spans(p)
                .unwrap()
                .iter()
                .map(|s| s.1)
                .collect::<Vec<usize>>()
        };
        assert_eq!(groups(&spliced[0]), vec![4, 4, 2]);
        assert_eq!(groups(&spliced[1]), vec![2, 4, 1]);
        assert_eq!(groups(&spliced[2]), vec![3, 4, 2]);
        assert!(groups(&spliced[3]).is_empty());
        // No temporary boundary file of THIS split is left behind. The directory is shared
        // with the module's other tests, which run concurrently and write their own
        // `.split-rg` files while they split, so only names that start with one of this
        // test's outputs are this split's.
        for o in &spliced {
            let path = std::path::Path::new(o);
            let dir = path.parent().unwrap();
            let own = path.file_name().unwrap().to_string_lossy().into_owned();
            for e in std::fs::read_dir(dir).unwrap() {
                let name = e.unwrap().file_name().to_string_lossy().into_owned();
                assert!(
                    !(name.starts_with(&own) && name.contains(".split-rg")),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn a_whole_row_group_of_an_unknown_source_is_still_an_error() {
        // A group whose statistics say one source, outside 0..n, is spliced nowhere: the
        // partition check must still report the rows it did not place.
        let mut sources = vec![0u32; 4];
        sources.extend(vec![5u32; 4]);
        let scored = scored_in_groups(&sources, 4);
        let outs: Vec<String> = (0..2).map(|i| tmp(&format!("bad{i}.parquet"))).collect();
        let err = split_by_source(&scored, &outs).unwrap_err().to_string();
        assert!(err.contains("4 of 8 rows"), "{err}");
    }

    #[test]
    fn split_by_source_partitions_every_row() {
        let scored = scored_with_sources(vec![0, 1, 0, 2, 1, 0]);
        let outs: Vec<String> = (0..3).map(|i| tmp(&format!("run{i}.parquet"))).collect();
        split_by_source(&scored, &outs).unwrap();
        let counts: Vec<usize> = outs.iter().map(|o| Table::read(o).unwrap().nrows).collect();
        assert_eq!(counts, vec![3, 2, 1]);
        // Each output must hold only its own run, or a per-run quantity is computed
        // from another run's PSMs.
        for (i, o) in outs.iter().enumerate() {
            let t = Table::read(o).unwrap();
            let idx = t.schema.index_of("source").unwrap();
            for b in &t.batches {
                let src = b
                    .column(idx)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap();
                for k in 0..src.len() {
                    assert_eq!(src.value(k), i as u32);
                }
            }
        }
    }

    #[test]
    fn a_source_index_with_no_output_table_is_an_error_not_a_silent_drop() {
        // Source 2 with only two runs: the old code wrote two tables holding five of
        // the six rows and returned Ok, so every per-run quantity and the cross-run
        // LFQ were computed from a population one row short, with nothing logged.
        let scored = scored_with_sources(vec![0, 1, 0, 2, 1, 0]);
        let outs: Vec<String> = (0..2).map(|i| tmp(&format!("run{i}.parquet"))).collect();
        let err = split_by_source(&scored, &outs).unwrap_err().to_string();
        assert!(err.contains("5 of 6 rows"), "{err}");
    }

    #[test]
    fn a_run_that_identified_nothing_yields_an_empty_table_not_an_error() {
        // Legitimate and must stay legitimate: run 1 contributed no PSMs. The
        // downstream quant reads an empty table; it must not read run 0's rows.
        let scored = scored_with_sources(vec![0, 0]);
        let outs: Vec<String> = (0..2).map(|i| tmp(&format!("run{i}.parquet"))).collect();
        split_by_source(&scored, &outs).unwrap();
        assert_eq!(Table::read(&outs[0]).unwrap().nrows, 2);
        assert_eq!(Table::read(&outs[1]).unwrap().nrows, 0);
    }
}
