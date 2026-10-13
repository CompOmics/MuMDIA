//! Search-anchored multi-head calibration (`rt_im_train.multihead_anchors = search`, or
//! `auto` on a run whose seed search is short of anchors; docs/08_rt_im_train.md section 4f).
//!
//! The multi-head calibration of the run that adapts the library is normally fitted on that
//! run's confident seed PSMs: a few hundred anchors from a broad, RT-free search. Under
//! `search` the run first goes through the whole chain once on the seed-anchored library
//! (rt-im-train, extract, features, compete and a single-run rescore), the confident targets
//! of that search become the anchors, and the calibration is refitted on them. Everything
//! the bootstrap pass writes lives in `<run>/rt_bootstrap/`, so the final pass writes its
//! artifacts under their usual names and the refitted library takes the name the
//! seed-anchored one has without this option (`fragment_library_precursors_multihead.parquet`).
//!
//! Any failure of the bootstrap pass, or fewer search anchors than the minimum the seed
//! anchors must reach (`min_seed_for_calibration` distinct target base peptides), keeps the
//! seed-anchored library with a warning. The run never fails for it.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result};
use mumdia_core::config::{Config, MultiheadAnchors};
use mumdia_io::table::{Col, Table};
use tracing::{info, warn};

use crate::stages::rt_im_train::AnchorPlan;
use crate::stages::{convert, rescore, run_experiment};

/// The bootstrap pass's directory under a run's output directory.
pub const DIR: &str = "rt_bootstrap";
/// The anchor table inside [`DIR`].
pub const ANCHORS: &str = "search_anchors.parquet";
/// The multi-head library's file name, seed-anchored or refitted.
pub const MULTIHEAD_LIBRARY: &str = "fragment_library_precursors_multihead.parquet";

/// `<out>/rt_bootstrap`.
pub fn dir(out: &str) -> String {
    format!("{out}/{DIR}")
}

/// Whether `multihead_anchors = search` was asked for explicitly: the adapting run then
/// always takes the bootstrap pass, which `run-experiment` must know before that run's seed
/// search exists (it then adapts the first run alone, `experiment.overlap_front_threads`).
pub fn always(cfg: &Config, mh_heads: usize) -> bool {
    mh_heads > 0 && cfg.rt_im_train.multihead_anchors == MultiheadAnchors::Search
}

/// Whether this run refits its multi-head calibration on search anchors, given the anchor
/// cut its seed search reached (`rt_im_train::plan_anchors`). `search`: whenever a
/// multi-head calibration runs (`mh_heads > 0`). `auto`: when one runs on an ungrouped run
/// and the seed is short of anchors, fewer than `multihead_search_below` at `q_train` or
/// only enough after relaxing `q_train` through `anchor_q_ladder`. `seed`: never.
pub fn active(cfg: &Config, mh_heads: usize, plan: &AnchorPlan) -> bool {
    if mh_heads == 0 {
        return false;
    }
    match cfg.rt_im_train.multihead_anchors {
        MultiheadAnchors::Seed => false,
        MultiheadAnchors::Search => true,
        MultiheadAnchors::Auto => cfg.groups.window_groups <= 1 && short_of_anchors(cfg, plan),
    }
}

/// `auto`'s test: the seed found fewer than `multihead_search_below` anchors at `q_train`,
/// or reached `min_seed_for_calibration` only at a relaxed cut.
fn short_of_anchors(cfg: &Config, plan: &AnchorPlan) -> bool {
    plan.q_train > cfg.rt_im_train.q_train
        || plan.n_anchors < cfg.rt_im_train.multihead_search_below
}

/// [`active`], with one INFO line saying what `auto` chose and why. Called once per
/// adapting run, before its multi-head calibration.
pub fn decide(cfg: &Config, mh_heads: usize, plan: &AnchorPlan, out: &str) -> bool {
    let search = active(cfg, mh_heads, plan);
    if mh_heads > 0
        && cfg.rt_im_train.multihead_anchors == MultiheadAnchors::Auto
        && cfg.groups.window_groups <= 1
    {
        let below = cfg.rt_im_train.multihead_search_below;
        if search {
            info!(
                run = %out,
                seed_anchors = plan.n_anchors,
                anchor_q = plan.q_train,
                multihead_search_below = below,
                "rt-bootstrap: multihead_anchors = auto chose search anchors: the seed search \
                 found {} anchors at q {} (fewer than {below}, or only at a relaxed q), so the \
                 run is searched once to refit its multi-head calibration",
                plan.n_anchors,
                plan.q_train
            );
        } else {
            info!(
                run = %out,
                seed_anchors = plan.n_anchors,
                multihead_search_below = below,
                "rt-bootstrap: multihead_anchors = auto kept the seed anchors: {} at q_train, \
                 at least {below}",
                plan.n_anchors
            );
        }
    }
    search
}

/// Warn once when `multihead_anchors = search` was asked for but no multi-head calibration
/// runs, which is how the automatic head count degrades without a DeepLC interpreter or on
/// a native retention-time source. The explicit contradictions are refused by validation.
/// `auto` degrades to `seed` there silently, as the automatic head count itself does.
pub fn warn_if_inert(cfg: &Config, mh_heads: usize, who: &str) {
    if cfg.rt_im_train.multihead_anchors == MultiheadAnchors::Search && mh_heads == 0 {
        warn!(
            "{who}: rt_im_train.multihead_anchors = search has no effect because no multi-head \
             calibration runs (it needs a DeepLC interpreter and a DeepLC retention-time \
             source); the library is not refitted on search anchors"
        );
    }
}

/// Where the seed-anchored multi-head library of a run in `out` is written: inside the
/// bootstrap directory when it is refitted later (`search`, from [`decide`]), else under
/// its usual name.
pub fn seed_library_path(search: bool, out: &str) -> String {
    if search {
        let d = dir(out);
        std::fs::create_dir_all(&d).ok();
        format!("{d}/{MULTIHEAD_LIBRARY}")
    } else {
        format!("{out}/{MULTIHEAD_LIBRARY}")
    }
}

/// Under `multihead_anchors = auto`, record in `<library>.summary.json` what was chosen and
/// on which count (`anchors_auto`: seed anchors, the anchor cut, the threshold, the choice).
/// Nothing under `seed` or `search`. The anchor source itself is recorded by [`refit`], or
/// here as `seed` when `auto` kept the seed anchors.
pub fn record_auto(cfg: &Config, plan: &AnchorPlan, search: bool, library: &str) {
    if cfg.rt_im_train.multihead_anchors != MultiheadAnchors::Auto {
        return;
    }
    let mut fields = serde_json::json!({
        "anchors_auto": {
            "seed_anchors": plan.n_anchors,
            "anchor_q": plan.q_train,
            "multihead_search_below": cfg.rt_im_train.multihead_search_below,
            "chose": if search { "search" } else { "seed" },
        },
    });
    if !search {
        fields["anchor_source"] = serde_json::json!("seed");
    }
    record_anchor_source(library, fields);
}

/// The anchor source a multi-head library's `.summary.json` records (`search` or `seed`);
/// `None` when it records none.
pub fn anchor_source(library: &str) -> Option<String> {
    let v: serde_json::Value =
        mumdia_io::json::read_json(&format!("{library}.summary.json")).ok()?;
    v.get("anchor_source")?.as_str().map(str::to_string)
}

/// What [`write_search_anchors`] wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnchorTable {
    /// Rows: one per candidate.
    pub rows: usize,
    /// Distinct target base peptides among them, the unit `rt_im_train::plan_anchors`
    /// counts the seed anchors in.
    pub base_peptides: usize,
}

/// Write the search anchors of a single-run scored table to `out`, in the seed's column
/// shape so `deeplc_finetune.py` reads them exactly as it reads `seed_psms.parquet`.
///
/// Selected: targets with a finite `precursor_q <= q`, a finite `apex_rt` and a finite
/// `score`. `precursor_q` is written to its group's winning row only, so this is already one
/// row per precursor; a candidate that still has several rows keeps its best-scoring one
/// (the first on a tie). Rows are written in ascending `candidate_id`, which makes the
/// table, and the worker's reference built from it, independent of the scored table's row
/// order.
///
/// Columns: `candidate_id` (u32), `peptidoform` (utf8), `charge` (i32), `base_peptide_id`
/// (u32), `label` (utf8, always `target`), `score` (f64), `spectrum_q` (f64, the
/// `precursor_q`), `observed_rt` (f64, the `apex_rt` in seconds).
pub fn write_search_anchors(scored: &str, q: f64, out: &str) -> Result<AnchorTable> {
    let t = Table::read_cols(
        scored,
        &[
            "candidate_id",
            "peptidoform",
            "charge",
            "base_peptide_id",
            "label",
            "score",
            "precursor_q",
            "apex_rt",
        ],
    )
    .with_context(|| format!("reading the bootstrap scored table {scored}"))?;
    let cid = t.u32("candidate_id")?;
    let is_target = t.str_eq("label", "target")?;
    let score = t.f64("score")?;
    let pq = t.f64("precursor_q")?;
    let apex = t.f64("apex_rt")?;
    // candidate_id -> row of its best-scoring confident target row. Ordered, so the output
    // order is the candidate order whatever order the scored table holds.
    let mut best: BTreeMap<u32, usize> = BTreeMap::new();
    for i in 0..t.nrows {
        if !is_target[i]
            || !pq[i].is_finite()
            || pq[i] > q
            || !apex[i].is_finite()
            || !score[i].is_finite()
        {
            continue;
        }
        match best.get(&cid[i]) {
            Some(&j) if score[j] >= score[i] => {}
            _ => {
                best.insert(cid[i], i);
            }
        }
    }
    let rows: Vec<usize> = best.values().copied().collect();
    let peptidoform = t.str("peptidoform")?;
    let charge = t.i32("charge")?;
    let base = t.u32("base_peptide_id")?;
    let base_peptides = rows
        .iter()
        .map(|&i| base[i])
        .collect::<BTreeSet<u32>>()
        .len();
    mumdia_io::table::write_table(
        out,
        vec![
            Col::U32(
                "candidate_id".into(),
                rows.iter().map(|&i| cid[i]).collect(),
            ),
            Col::Str(
                "peptidoform".into(),
                rows.iter().map(|&i| peptidoform[i].clone()).collect(),
            ),
            Col::I32("charge".into(), rows.iter().map(|&i| charge[i]).collect()),
            Col::U32(
                "base_peptide_id".into(),
                rows.iter().map(|&i| base[i]).collect(),
            ),
            Col::Str("label".into(), vec!["target".to_string(); rows.len()]),
            Col::F64("score".into(), rows.iter().map(|&i| score[i]).collect()),
            Col::F64("spectrum_q".into(), rows.iter().map(|&i| pq[i]).collect()),
            Col::F64(
                "observed_rt".into(),
                rows.iter().map(|&i| apex[i]).collect(),
            ),
        ],
    )
    .with_context(|| format!("writing the search anchors {out}"))?;
    Ok(AnchorTable {
        rows: rows.len(),
        base_peptides,
    })
}

/// The inputs of one run's bootstrap pass and refit.
pub struct Refit<'a> {
    pub cfg: &'a Config,
    pub config_hash: &'a str,
    pub converted: &'a convert::ConvertOutputs,
    /// The table the multi-head calibration re-predicts (the base library).
    pub lib_p_base: &'a str,
    pub lib_f: &'a str,
    pub seed: &'a str,
    /// The run's output directory.
    pub out: &'a str,
    /// The seed-anchored multi-head library, from [`seed_library_path`].
    pub seed_library: &'a str,
    pub mh_heads: usize,
    /// Torch threads of the refit's DeepLC worker.
    pub threads: usize,
    /// The base library carries placeholder iRT that the calibration must replace in full.
    pub irt_placeholder: bool,
    /// Target anchors the seed calibration used, for the log line.
    pub seed_anchors: usize,
}

/// What the refit produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// The library the rest of the run reads: `<out>/fragment_library_precursors_multihead.parquet`.
    pub library: String,
    /// `Some` when the library was refitted on search anchors; `None` when the bootstrap
    /// fell back to the seed-anchored library.
    pub anchors: Option<AnchorTable>,
    /// The anchor table, when one was written.
    pub anchors_path: Option<String>,
}

/// Run the bootstrap pass and refit the multi-head calibration on its anchors, or keep the
/// seed-anchored library when that is not possible. Either way the library ends up at
/// `<out>/fragment_library_precursors_multihead.parquet` and its `.summary.json` records the
/// anchor source. Only a failure to move the seed-anchored library into place is an error.
pub fn refit(p: &Refit) -> Result<Outcome> {
    let library = format!("{}/{MULTIHEAD_LIBRARY}", p.out);
    let anchors_path = format!("{}/{ANCHORS}", dir(p.out));
    // The bootstrap search runs on the adaptation's thread share: under `run-experiment`'s
    // overlap (`auto` choosing `search` on the first run) the fronts of the other runs hold
    // the rest of the pool. No stage's output depends on its thread count.
    let pass = || bootstrap_and_refit(p, &library, &anchors_path);
    let result = if p.threads > 0 && p.threads < rayon::current_num_threads() {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(p.threads)
            .build()
        {
            Ok(pool) => pool.install(pass),
            Err(e) => Err(anyhow::anyhow!(
                "building the bootstrap pass's thread pool: {e}"
            )),
        }
    } else {
        pass()
    };
    match result {
        Ok(table) => {
            record_anchor_source(
                &library,
                serde_json::json!({
                    "anchor_source": "search",
                    "search_anchors": {
                        "table": anchors_path,
                        "rows": table.rows,
                        "base_peptides": table.base_peptides,
                        "q_train": p.cfg.rt_im_train.q_train,
                        "seed_anchors": p.seed_anchors,
                        "seed_library": p.seed_library,
                    },
                }),
            );
            Ok(Outcome {
                library,
                anchors: Some(table),
                anchors_path: Some(anchors_path),
            })
        }
        Err(e) => {
            let reason = format!("{e:#}");
            warn!(
                run = %p.out,
                reason = %reason,
                "rt-bootstrap: keeping the seed-anchored multi-head library \
                 (rt_im_train.multihead_anchors = search or auto could not refit it)"
            );
            move_into_place(p.seed_library, &library)?;
            record_anchor_source(
                &library,
                serde_json::json!({
                    "anchor_source": "seed",
                    "search_anchors_fallback": reason,
                }),
            );
            let written = std::path::Path::new(&anchors_path).exists();
            Ok(Outcome {
                library,
                anchors: None,
                anchors_path: written.then_some(anchors_path),
            })
        }
    }
}

/// The bootstrap pass and the refit; any error means "keep the seed-anchored library".
fn bootstrap_and_refit(p: &Refit, library: &str, anchors_path: &str) -> Result<AnchorTable> {
    let cfg = p.cfg;
    let bdir = dir(p.out);
    std::fs::create_dir_all(&bdir).with_context(|| format!("creating {bdir}"))?;
    // A reused output directory may hold an earlier run's anchors; they must not be taken
    // for this run's when the pass fails before writing its own.
    std::fs::remove_file(anchors_path).ok();
    info!(
        run = %p.out,
        "rt-bootstrap: searching the run on the seed-anchored library to collect search anchors"
    );
    let (competed, _chrom) = run_experiment::finish_run(
        cfg,
        p.config_hash,
        p.seed_library,
        p.lib_f,
        p.converted,
        p.seed,
        &bdir,
    )
    .context("the bootstrap search (rt-im-train to compete) failed")?;
    let scored = format!("{bdir}/psms_scored.parquet");
    let windows = vec![p.converted.isolation_windows.clone()];
    rescore::run_hashed(rescore::RescoreParams {
        competed: std::slice::from_ref(&competed),
        sources: None,
        out: &scored,
        work_dir: &rescore::sidecar_work_dir(&format!("{bdir}/sidecar_work")),
        script_dir: &cfg.predict_frag.sidecar_script_dir,
        cfg: &cfg.rescore,
        config_hash: p.config_hash,
        shadow: Some(crate::stages::shadow::ShadowInputs::new(p.lib_f, &windows)),
    })
    .context("the bootstrap single-run rescore failed")?;
    // The bootstrap's widest tables are only an input of the rescore that just read them.
    // The scored table, the windows and the anchors stay as the pass's record.
    for name in [
        "psms_extracted.parquet",
        "psms_extracted.parquet.peaks.parquet",
        "chromatograms.parquet",
        "features.parquet",
        "run.pin",
        "psms_competed.parquet",
    ] {
        std::fs::remove_file(format!("{bdir}/{name}")).ok();
    }

    let q = cfg.rt_im_train.q_train;
    let table = write_search_anchors(&scored, q, anchors_path)?;
    let need = cfg.rt_im_train.min_seed_for_calibration.max(2);
    if table.base_peptides < need {
        anyhow::bail!(
            "the bootstrap search found {} target base peptides at precursor_q <= {q} \
             (need rt_im_train.min_seed_for_calibration = {need})",
            table.base_peptides
        );
    }
    info!(
        run = %p.out,
        search_anchors = table.rows,
        base_peptides = table.base_peptides,
        seed_anchors = p.seed_anchors,
        q_train = q,
        "rt-bootstrap: refitting the multi-head calibration on {} search anchors \
         (confident targets of the bootstrap search, observed RT = apex_rt) instead of {} \
         seed anchors",
        table.rows,
        p.seed_anchors
    );
    let python = cfg
        .predict_frag
        .deeplc_python
        .as_deref()
        .context("no DeepLC interpreter for the refit")?;
    let script =
        crate::sidecar::resolve_script(&cfg.predict_frag.sidecar_script_dir, "deeplc_finetune.py");
    crate::sidecar::run_deeplc_multihead(
        python,
        &script,
        p.lib_p_base,
        anchors_path,
        library,
        p.mh_heads,
        q,
        cfg.rt_im_train.window_holdout_frac,
        p.threads,
        cfg.rt_im_train.deeplc_predict_shards,
        crate::cache::projection_dir(cfg).as_ref(),
    )
    .context("the multi-head refit on the search anchors failed")?;
    if p.irt_placeholder {
        crate::sidecar::require_every_row_repredicted(library)?;
    }
    Ok(table)
}

/// Move the seed-anchored library (and its summary) to the usual name.
fn move_into_place(from: &str, to: &str) -> Result<()> {
    for (src, dst) in [
        (from.to_string(), to.to_string()),
        (format!("{from}.summary.json"), format!("{to}.summary.json")),
    ] {
        if !std::path::Path::new(&src).exists() {
            // The summary is the worker's; a library without one is still a library.
            if src.ends_with(".summary.json") {
                continue;
            }
            anyhow::bail!("the seed-anchored multi-head library {src} does not exist");
        }
        if std::fs::rename(&src, &dst).is_err() {
            std::fs::copy(&src, &dst)
                .with_context(|| format!("moving {src} to {dst}"))
                .map(|_| ())?;
        }
    }
    Ok(())
}

/// Merge `fields` into `<library>.summary.json`, creating it when the worker wrote none.
/// Provenance only: a failure here is logged, never fatal.
pub(crate) fn record_anchor_source(library: &str, fields: serde_json::Value) {
    let path = format!("{library}.summary.json");
    let mut v: serde_json::Value = mumdia_io::json::read_json(&path)
        .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
    if let (Some(obj), Some(add)) = (v.as_object_mut(), fields.as_object()) {
        for (k, val) in add {
            obj.insert(k.clone(), val.clone());
        }
    }
    if let Err(e) = mumdia_io::json::write_json(&path, &v) {
        warn!(path = %path, error = %format!("{e:#}"), "rt-bootstrap: could not record the anchor source");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::Col;

    fn tmp(name: &str) -> String {
        let d = std::env::temp_dir().join(format!(
            "mumdia_rt_bootstrap_{}_{}",
            std::process::id(),
            name
        ));
        std::fs::create_dir_all(&d).unwrap();
        d.to_string_lossy().replace('\\', "/")
    }

    /// A scored table in rescore's column shape, with the columns the writer reads.
    #[allow(clippy::type_complexity)]
    fn scored(path: &str, rows: &[(u32, &str, i32, u32, &str, f64, f64, f64)]) {
        mumdia_io::table::write_table(
            path,
            vec![
                Col::U32("candidate_id".into(), rows.iter().map(|r| r.0).collect()),
                Col::Str(
                    "peptidoform".into(),
                    rows.iter().map(|r| r.1.to_string()).collect(),
                ),
                Col::I32("charge".into(), rows.iter().map(|r| r.2).collect()),
                Col::U32("base_peptide_id".into(), rows.iter().map(|r| r.3).collect()),
                Col::Str(
                    "label".into(),
                    rows.iter().map(|r| r.4.to_string()).collect(),
                ),
                Col::F64("score".into(), rows.iter().map(|r| r.5).collect()),
                Col::F64("precursor_q".into(), rows.iter().map(|r| r.6).collect()),
                Col::F64("apex_rt".into(), rows.iter().map(|r| r.7).collect()),
                // A column the writer must ignore.
                Col::F64("q_value".into(), vec![0.0; rows.len()]),
            ],
        )
        .unwrap();
    }

    #[test]
    fn the_anchor_table_keeps_confident_targets_one_row_per_candidate_in_candidate_order() {
        let d = tmp("select");
        let s = format!("{d}/scored.parquet");
        scored(
            &s,
            &[
                // cid, peptidoform, z, base, label, score, precursor_q, apex_rt
                (7, "PEPTIDEK", 2, 70, "target", 5.0, 0.001, 600.0),
                (3, "ELVISK", 2, 30, "target", 4.0, 0.010, 300.0), // at q: kept
                (4, "ELVISK", 3, 30, "target", 3.0, 0.005, 301.0), // same base, other charge
                (9, "DECOYK", 2, 90, "decoy", 9.0, 0.000, 900.0),  // decoy: dropped
                (5, "LOSERK", 2, 50, "target", 2.0, 1.000, 500.0), // grouped loser: dropped
                (6, "WIDEK", 2, 60, "target", 2.5, 0.011, 610.0),  // above q: dropped
                (8, "NANRTK", 2, 80, "target", 6.0, 0.001, f64::NAN), // no apex: dropped
                (7, "PEPTIDEK", 2, 70, "target", 6.0, 0.002, 602.0), // better row of cid 7
                (7, "PEPTIDEK", 2, 70, "target", 6.0, 0.002, 604.0), // tie: first kept
            ],
        );
        let out = format!("{d}/anchors.parquet");
        let t = write_search_anchors(&s, 0.01, &out).unwrap();
        assert_eq!(
            t,
            AnchorTable {
                rows: 3,
                base_peptides: 2
            }
        );
        let a = Table::read(&out).unwrap();
        assert_eq!(a.u32("candidate_id").unwrap(), vec![3, 4, 7]);
        assert_eq!(
            a.str("peptidoform").unwrap(),
            vec!["ELVISK", "ELVISK", "PEPTIDEK"]
        );
        assert_eq!(a.i32("charge").unwrap(), vec![2, 3, 2]);
        assert_eq!(a.u32("base_peptide_id").unwrap(), vec![30, 30, 70]);
        assert_eq!(a.str("label").unwrap(), vec!["target"; 3]);
        assert_eq!(a.f64("score").unwrap(), vec![4.0, 3.0, 6.0]);
        assert_eq!(a.f64("spectrum_q").unwrap(), vec![0.010, 0.005, 0.002]);
        assert_eq!(a.f64("observed_rt").unwrap(), vec![300.0, 301.0, 602.0]);
        // The exact column set and types `deeplc_finetune.py::build_reference` reads, with
        // the engine's string type (utf8, not large_utf8).
        use arrow::datatypes::DataType;
        let want = [
            ("candidate_id", DataType::UInt32),
            ("peptidoform", DataType::Utf8),
            ("charge", DataType::Int32),
            ("base_peptide_id", DataType::UInt32),
            ("label", DataType::Utf8),
            ("score", DataType::Float64),
            ("spectrum_q", DataType::Float64),
            ("observed_rt", DataType::Float64),
        ];
        let got: Vec<(String, DataType)> = a
            .schema
            .fields()
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone()))
            .collect();
        assert_eq!(
            got,
            want.iter()
                .map(|(n, t)| (n.to_string(), t.clone()))
                .collect::<Vec<_>>()
        );
        // Deterministic: the same input writes the same bytes.
        let out2 = format!("{d}/anchors2.parquet");
        write_search_anchors(&s, 0.01, &out2).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), std::fs::read(&out2).unwrap());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn an_empty_selection_writes_an_empty_table_with_the_columns() {
        let d = tmp("empty");
        let s = format!("{d}/scored.parquet");
        scored(&s, &[(1, "AK", 2, 1, "decoy", 1.0, 0.0, 10.0)]);
        let out = format!("{d}/anchors.parquet");
        let t = write_search_anchors(&s, 0.01, &out).unwrap();
        assert_eq!(
            t,
            AnchorTable {
                rows: 0,
                base_peptides: 0
            }
        );
        let a = Table::read(&out).unwrap();
        assert_eq!(a.nrows, 0);
        assert!(a.column_names().contains(&"observed_rt".to_string()));
        std::fs::remove_dir_all(&d).ok();
    }

    fn plan(q_train: f64, n_anchors: usize) -> AnchorPlan {
        AnchorPlan {
            q_train,
            n_anchors,
            sufficient: true,
        }
    }

    #[test]
    fn the_seed_library_moves_into_the_bootstrap_directory_only_when_searching() {
        let d = tmp("paths");
        assert_eq!(
            seed_library_path(false, &d),
            format!("{d}/{MULTIHEAD_LIBRARY}"),
            "without the bootstrap the seed-anchored library has the usual name"
        );
        assert_eq!(
            seed_library_path(true, &d),
            format!("{d}/{DIR}/{MULTIHEAD_LIBRARY}")
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn seed_and_search_ignore_the_anchor_count() {
        let mut seed = Config::default();
        seed.rt_im_train.multihead_anchors = MultiheadAnchors::Seed;
        let mut search = Config::default();
        search.rt_im_train.multihead_anchors = MultiheadAnchors::Search;
        for p in [plan(0.01, 10), plan(0.01, 100_000), plan(0.05, 60)] {
            assert!(!active(&seed, 80, &p));
            assert!(active(&search, 80, &p));
            assert!(
                !active(&search, 0, &p),
                "inert without a multi-head calibration"
            );
        }
        assert!(always(&search, 80));
        assert!(!always(&search, 0));
        assert!(!always(&seed, 80));
        assert!(
            !always(&Config::default(), 80),
            "auto is decided per run, never in advance"
        );
    }

    #[test]
    fn auto_searches_only_a_run_short_of_seed_anchors() {
        let cfg = Config::default();
        assert_eq!(cfg.rt_im_train.multihead_anchors, MultiheadAnchors::Auto);
        let below = cfg.rt_im_train.multihead_search_below;
        let q = cfg.rt_im_train.q_train;
        // Measured points: single cells (100-250) and a single-cell HeLa run (1,249) search;
        // Orbitrap AIF (5,025) and bulk Astral (13,786) keep the seed anchors.
        for n in [97, 250, 1_249, below - 1] {
            assert!(active(&cfg, 80, &plan(q, n)), "{n} seed anchors");
        }
        for n in [below, 5_025, 13_786] {
            assert!(!active(&cfg, 80, &plan(q, n)), "{n} seed anchors");
        }
        // Enough anchors only after relaxing q_train: short of anchors by definition.
        assert!(active(&cfg, 80, &plan(0.05, 10_000)));
        // No multi-head calibration, or a grouped run: auto is seed.
        assert!(!active(&cfg, 0, &plan(q, 10)));
        let mut grouped = Config::default();
        grouped.groups.window_groups = 4;
        assert!(!active(&grouped, 80, &plan(q, 10)));
        // The threshold is the configured one; 0 searches only a relaxed run.
        let mut zero = Config::default();
        zero.rt_im_train.multihead_search_below = 0;
        assert!(!active(&zero, 80, &plan(q, 10)));
        assert!(active(&zero, 80, &plan(0.02, 60)));
        // decide() is active() plus a log line.
        assert!(decide(&cfg, 80, &plan(q, 10), "r0"));
        assert!(!decide(&cfg, 80, &plan(q, 13_786), "r0"));
    }

    #[test]
    fn auto_records_its_choice_beside_the_library() {
        let d = tmp("auto_record");
        let lib = format!("{d}/{MULTIHEAD_LIBRARY}");
        std::fs::write(format!("{lib}.summary.json"), br#"{"rows": 3}"#).unwrap();
        let cfg = Config::default();
        record_auto(&cfg, &plan(0.01, 13_786), false, &lib);
        let v: serde_json::Value =
            mumdia_io::json::read_json(&format!("{lib}.summary.json")).unwrap();
        assert_eq!(v["rows"], 3, "the worker's summary is kept");
        assert_eq!(v["anchor_source"], "seed");
        assert_eq!(v["anchors_auto"]["seed_anchors"], 13_786);
        assert_eq!(v["anchors_auto"]["chose"], "seed");
        assert_eq!(
            v["anchors_auto"]["multihead_search_below"],
            cfg.rt_im_train.multihead_search_below
        );
        assert_eq!(anchor_source(&lib).as_deref(), Some("seed"));

        // A search choice leaves the anchor source to the refit that recorded it.
        record_anchor_source(&lib, serde_json::json!({"anchor_source": "search"}));
        record_auto(&cfg, &plan(0.01, 200), true, &lib);
        assert_eq!(anchor_source(&lib).as_deref(), Some("search"));
        let v: serde_json::Value =
            mumdia_io::json::read_json(&format!("{lib}.summary.json")).unwrap();
        assert_eq!(v["anchors_auto"]["chose"], "search");

        // Under an explicit seed or search nothing is added.
        let other = format!("{d}/other.parquet");
        let mut seed = Config::default();
        seed.rt_im_train.multihead_anchors = MultiheadAnchors::Seed;
        record_auto(&seed, &plan(0.01, 10), false, &other);
        assert!(!std::path::Path::new(&format!("{other}.summary.json")).exists());
        assert_eq!(anchor_source(&other), None);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_failed_bootstrap_keeps_the_seed_library_under_the_usual_name_and_says_so() {
        // The bootstrap fails at once (no spectra to search); the seed-anchored library and
        // its summary must end up under the usual name with the anchor source recorded.
        let d = tmp("fallback");
        let bdir = dir(&d);
        std::fs::create_dir_all(&bdir).unwrap();
        let seed_lib = format!("{bdir}/{MULTIHEAD_LIBRARY}");
        std::fs::write(&seed_lib, b"not really a parquet").unwrap();
        std::fs::write(format!("{seed_lib}.summary.json"), br#"{"rows": 3}"#).unwrap();
        let mut cfg = Config::default();
        cfg.rt_im_train.multihead_anchors = MultiheadAnchors::Search;
        let co = convert::ConvertOutputs {
            ms1: format!("{d}/missing_ms1.parquet"),
            ms2: format!("{d}/missing_ms2.parquet"),
            isolation_windows: format!("{d}/missing_windows.parquet"),
            ms2_to_ms1: format!("{d}/missing_map.parquet"),
            hashes: convert::ConvertHashes {
                ms1: String::new(),
                ms2: String::new(),
                isolation_windows: String::new(),
                ms2_to_ms1: String::new(),
            },
            schemas: convert::ConvertSchemas::for_source(false),
        };
        let o = refit(&Refit {
            cfg: &cfg,
            config_hash: "h",
            converted: &co,
            lib_p_base: &format!("{d}/missing_lib.parquet"),
            lib_f: &format!("{d}/missing_frag.parquet"),
            seed: &format!("{d}/missing_seed.parquet"),
            out: &d,
            seed_library: &seed_lib,
            mh_heads: 80,
            threads: 1,
            irt_placeholder: false,
            seed_anchors: 120,
        })
        .expect("a failed bootstrap is a fallback, not an error");
        let usual = format!("{d}/{MULTIHEAD_LIBRARY}");
        assert_eq!(o.library, usual);
        assert_eq!(o.anchors, None);
        assert_eq!(std::fs::read(&usual).unwrap(), b"not really a parquet");
        let summary: serde_json::Value =
            mumdia_io::json::read_json(&format!("{usual}.summary.json")).unwrap();
        assert_eq!(summary["rows"], 3, "the worker's summary is kept");
        assert_eq!(summary["anchor_source"], "seed");
        assert!(summary["search_anchors_fallback"].is_string());
        std::fs::remove_dir_all(&d).ok();
    }
}
