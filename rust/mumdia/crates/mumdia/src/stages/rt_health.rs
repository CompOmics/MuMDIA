//! Making a search without a retention-time calibration impossible to miss.
//!
//! A run whose seed ends with fewer than two confident target anchors has no iRT -> RT
//! mapping, so `rt-im-train` gives it unbounded windows and every candidate of the
//! library is extracted over the whole gradient (`rt_im_train::window_plan`). The search
//! still finishes, which is the problem: it is many times slower than a calibrated one,
//! hands the classifier tens of millions of candidates, and reports a small fraction of
//! what the data hold, with nothing in the result that says so. The usual cause in a
//! FASTA search is the native fragment model, whose intensities rarely give the seed a
//! confident target (the HCDch2 entry of CLAUDE.md describes the same mechanism for
//! heuristic charge-2 fragments); an instrument m/z offset beyond the seed tolerance and
//! a library that does not fit the sample are the others.
//!
//! One check stops a search, and it runs before the search starts:
//! [`refuse_native_predictors`] refuses a FASTA library built on the native fallback
//! models unless `predict_frag.allow_native_predictors` says to build it anyway. The rest
//! of this module only reads and reports: the native-predictor warning at the start of a
//! FASTA search that was allowed to use them, a prominent warning after `rt-im-train`
//! names the consequence and the likely causes, and the end of `run` and `run-experiment`
//! records each such run in the manifest ([`RtCalibrationGap`]) and logs one summary
//! line. Nothing is written for a calibrated run, so its artifacts, reports and manifest
//! keep their bytes.
//!
//! It does not stop a search whose run ends uncalibrated. That stop would change the
//! default behaviour of every search with an uncalibrated run, and some of those are
//! legitimate: a blank well in a single-cell plate has no anchors by construction, and the
//! second pass of an experiment searches such runs again on the consensus retention times
//! of the others. Grouped searches are the exception and stop already
//! (`run_groups::refuse_unbounded_rt`), because there unbounded windows multiply the memory
//! by the number of bands.

use anyhow::Result;
use mumdia_core::config::{Config, FragPredictorKind, RtPredictorKind};
use mumdia_core::manifest::RtCalibrationGap;
use tracing::warn;

use crate::stages::rt_im_train::{FALLBACK_FIXED_STATUS, INSUFFICIENT_ANCHORS_STATUS};

/// Which predictions of a library are the native fallback models.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativePredictions {
    /// Fragment intensities from the native heuristic (`predict_frag.predictor = native`).
    pub fragments: bool,
    /// Retention times from the native additive model (`predict_frag.rt_predictor = native`).
    pub retention_time: bool,
    /// The spectra carry 1/K0 and the library has no predicted 1/K0
    /// (`predict_frag.im_predictor` unset).
    pub ion_mobility_missing: bool,
}

impl NativePredictions {
    /// What a search builds its library with: nothing native in library-input mode, where
    /// the imported library supplies every prediction and `predict_frag` is not read.
    pub fn of_config(cfg: &Config, fasta_mode: bool, spectra_have_im: bool) -> Self {
        if !fasta_mode {
            return Self::default();
        }
        let pf = &cfg.predict_frag;
        Self {
            fragments: pf.predictor == FragPredictorKind::Native,
            retention_time: pf.rt_predictor == RtPredictorKind::Native,
            ion_mobility_missing: spectra_have_im && pf.im_predictor.is_none(),
        }
    }

    /// What the library a run searched was built with, from the report `predict-frag`
    /// writes beside its fragment table. An imported library, an empirical one and an
    /// unreadable report say nothing native.
    pub fn of_library(lib_fragments: &str) -> Self {
        let Ok(text) = std::fs::read_to_string(format!("{lib_fragments}.report.json")) else {
            return Self::default();
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            return Self::default();
        };
        if v.get("stage").and_then(|s| s.as_str()) != Some("predict-frag") {
            return Self::default();
        }
        let param = |k: &str| {
            v.get("params")
                .and_then(|p| p.get(k))
                .and_then(|x| x.as_str())
                .map(str::to_string)
        };
        Self {
            fragments: param("fragment_predictor").as_deref() == Some("Native"),
            retention_time: param("rt_predictor").as_deref() == Some("Native"),
            ion_mobility_missing: false,
        }
    }
}

/// Whether something after the library build replaces a native library retention time
/// with DeepLC's: a fine-tune (`rt_im_train.finetune_deeplc`) or an explicit multi-head
/// calibration (`rt_im_train.multihead_calibration` > 0). The automatic multi-head default
/// is scoped to libraries whose retention times are DeepLC's already, so it never runs on
/// a native one (`RtImTrainConfig::multihead_heads`).
pub fn native_rt_is_replaced(cfg: &Config) -> bool {
    cfg.rt_im_train.finetune_deeplc || cfg.rt_im_train.multihead_calibration.is_some_and(|n| n > 0)
}

/// Refuse a FASTA library build on the native fallback models, before anything runs.
///
/// `fasta_mode` is whether the command predicts a library from a FASTA (`run` and
/// `run-experiment` with `--fasta`, `build-library`, `predict-frag`); an imported library
/// reads no `predict_frag` setting and passes. Refused: native fragment intensities, and
/// native retention times that nothing replaces ([`native_rt_is_replaced`]).
/// `predict_frag.allow_native_predictors` lets the build run on them anyway. The
/// engine's default configuration names the native models, so a FASTA search without
/// `--config` is refused with the same message.
pub fn refuse_native_predictors(cfg: &Config, fasta_mode: bool, command: &str) -> Result<()> {
    if !fasta_mode || cfg.predict_frag.allow_native_predictors {
        return Ok(());
    }
    let pf = &cfg.predict_frag;
    let mut what: Vec<&str> = Vec::new();
    if pf.predictor == FragPredictorKind::Native {
        what.push("native fragment intensities (predict_frag.predictor = native)");
    }
    if pf.rt_predictor == RtPredictorKind::Native && !native_rt_is_replaced(cfg) {
        what.push("native retention times (predict_frag.rt_predictor = native)");
    }
    if what.is_empty() {
        return Ok(());
    }
    let defaults = if pf.predictor == FragPredictorKind::Native
        && pf.rt_predictor == RtPredictorKind::Native
    {
        " These are the engine defaults, so a FASTA search without --config, or with a \
         configuration that names no predictor, is refused too."
    } else {
        ""
    };
    anyhow::bail!(
        "{command}: refusing to build the library of this FASTA search with {}. The native \
         models are a fallback for a machine without the predictor environments: they \
         usually leave the seed search without confident anchors, so the retention-time \
         window is unbounded and the search is slow and finds far fewer peptides.{defaults} \
         To fix it, either predict the library with MS2PIP or AlphaPeptDeep for the \
         fragments and DeepLC for the retention times (predict_frag.predictor = ms2pip or \
         peptdeep, predict_frag.rt_predictor = deeplc; configs/examples/fasta-sidecars.json, \
         configs/examples/fasta-peptdeep.json, and configs/examples/diapasef.json for \
         diaPASEF; `mumdia doctor --config <file>` checks the interpreters), or search a \
         library predicted by DIA-NN and imported with scripts/import_diann_lib.py \
         (--lib-precursors / --lib-fragments). To run the native models anyway, set \
         predict_frag.allow_native_predictors = true.",
        what.join(" and ")
    )
}

/// The warning logged at the start of a FASTA search on native fragment intensities, or
/// `None`. One line, before any stage runs: the library build alone can take an hour.
pub fn native_predictor_warning(n: &NativePredictions) -> Option<String> {
    if !n.fragments {
        return None;
    }
    let mut also = String::new();
    if n.retention_time {
        also.push_str(", native retention times (predict_frag.rt_predictor = native)");
    }
    if n.ion_mobility_missing {
        also.push_str(
            ", and no predicted 1/K0 for spectra that carry ion mobility \
             (predict_frag.im_predictor unset)",
        );
    }
    Some(format!(
        "predict-frag: this FASTA search predicts its library with native fragment \
         intensities (predict_frag.predictor = native){also}. The native models are a \
         fallback for a machine without the predictor environments: they often leave the \
         seed search without confident anchors, and then the retention-time window is \
         unbounded and the search is slow and finds far fewer peptides. Configure MS2PIP or \
         AlphaPeptDeep and DeepLC (with IM2Deep for diaPASEF; configs/examples/\
         fasta-sidecars.json, configs/examples/diapasef.json), or search an imported library."
    ))
}

/// The run's calibration gap, read from its `cal.json`, or `None` when the run was
/// calibrated or wrote no `cal.json`.
pub fn gap_of(cal_json: &str, run: &str, pass: &str) -> Option<RtCalibrationGap> {
    let text = std::fs::read_to_string(cal_json).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let status = v.get("calibration_status")?.as_str()?;
    if status != INSUFFICIENT_ANCHORS_STATUS && status != FALLBACK_FIXED_STATUS {
        return None;
    }
    Some(RtCalibrationGap {
        run: run.to_string(),
        pass: pass.to_string(),
        calibration_status: status.to_string(),
        n_train: v.get("n_train").and_then(|n| n.as_u64()).unwrap_or(0),
        cal_json: cal_json.to_string(),
    })
}

/// The likely causes of a seed without anchors, the native models first when the library
/// used them.
pub fn likely_causes(n: &NativePredictions) -> String {
    let mut causes: Vec<&str> = Vec::new();
    if n.fragments {
        causes.push(
            "native fragment intensities (predict_frag.predictor = native): configure MS2PIP \
             or AlphaPeptDeep, or search an imported library",
        );
    }
    if n.retention_time {
        causes
            .push("native retention times (predict_frag.rt_predictor = native): configure DeepLC");
    }
    causes.push(
        "a fragment m/z offset beyond search_seed.fragment_tol_ppm, which the seed's tolerance \
         retry did not reach (see the seed's .masscal.json)",
    );
    causes.push(
        "a library or search settings that do not fit the acquisition or the sample \
         (instrument, enzyme, modifications, species)",
    );
    causes.push("a run with almost no peptide signal, such as a blank");
    causes.join("; ")
}

fn status_phrase(gap: &RtCalibrationGap) -> &'static str {
    if gap.calibration_status == INSUFFICIENT_ANCHORS_STATUS {
        "no retention-time calibration: every candidate is extracted over the whole gradient"
    } else {
        "too few anchors for a calibrated window: candidates are extracted in the broad \
         fallback window (rt_im_train.fallback_rt_window_s)"
    }
}

/// The prominent warning for one uncalibrated run, logged right after `rt-im-train`.
pub fn gap_warning(gap: &RtCalibrationGap, n: &NativePredictions) -> String {
    format!(
        "RETENTION-TIME CALIBRATION FAILED for {run} ({anchors} target anchors, status {status}): \
         {phrase}. The search continues, but it is much slower and identifies far fewer \
         peptides than a calibrated one. Likely causes: {causes}. docs/17_troubleshooting.md, \
         \"Seed: no confident targets\".",
        run = gap.run,
        anchors = gap.n_train,
        status = gap.calibration_status,
        phrase = status_phrase(gap),
        causes = likely_causes(n),
    )
}

/// After `rt-im-train`: read the run's `cal.json` and, when the run has no calibration,
/// log [`gap_warning`] with the causes the library it searched points to.
pub fn warn_after_fit(cal_json: &str, run: &str, lib_fragments: &str) {
    if let Some(gap) = gap_of(cal_json, run, "") {
        warn!(
            "{}",
            gap_warning(&gap, &NativePredictions::of_library(lib_fragments))
        );
    }
}

/// The end-of-search summary, or `None` when every run was calibrated. `n_runs` counts
/// the runs of one pass; the gaps of a second pass are counted on their own.
pub fn summary(gaps: &[RtCalibrationGap], n_runs: usize, n: &NativePredictions) -> Option<String> {
    if gaps.is_empty() {
        return None;
    }
    let count = |pass: &str, status: &str| {
        gaps.iter()
            .filter(|g| g.pass == pass && g.calibration_status == status)
            .count()
    };
    let mut parts: Vec<String> = Vec::new();
    for (pass, label) in [
        ("run", "this search"),
        ("first_pass", "the first pass"),
        ("second_pass", "the second pass"),
    ] {
        let unbounded = count(pass, INSUFFICIENT_ANCHORS_STATUS);
        let fixed = count(pass, FALLBACK_FIXED_STATUS);
        if unbounded + fixed == 0 {
            continue;
        }
        let mut s = format!(
            "{} of {n_runs} run(s) in {label} had no retention-time calibration",
            unbounded + fixed
        );
        if unbounded > 0 && fixed > 0 {
            s.push_str(&format!(
                " ({unbounded} with unbounded windows, {fixed} on the fixed fallback window)"
            ));
        } else if fixed > 0 {
            s.push_str(" (the fixed fallback window)");
        } else {
            s.push_str(" (unbounded windows)");
        }
        parts.push(s);
    }
    // Each run once, in the order the gaps were recorded.
    let mut runs: Vec<&str> = Vec::new();
    for g in gaps {
        if !runs.contains(&g.run.as_str()) {
            runs.push(&g.run);
        }
    }
    Some(format!(
        "SEARCH DEGRADED: {}. Their identifications are far fewer than the data hold; \
         read the result as incomplete. Runs: {}. Likely causes: {}. The manifest lists \
         them under rt_calibration_missing; docs/17_troubleshooting.md, \"Seed: no confident \
         targets\".",
        parts.join("; "),
        runs.join(", "),
        likely_causes(n),
    ))
}

/// Log [`summary`] when there is one.
pub fn warn_summary(gaps: &[RtCalibrationGap], n_runs: usize, n: &NativePredictions) {
    if let Some(s) = summary(gaps, n_runs, n) {
        warn!("{s}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("mumdia-rt-health-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write(path: &std::path::Path, v: serde_json::Value) -> String {
        std::fs::write(path, serde_json::to_string(&v).unwrap()).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// The start-of-search warning fires on native fragment intensities in a FASTA search
    /// only; an imported library and a sidecar predictor stay silent.
    #[test]
    fn the_native_warning_is_for_a_fasta_search_on_native_fragments() {
        let cfg = Config::default();
        assert_eq!(cfg.predict_frag.predictor, FragPredictorKind::Native);
        let fasta = NativePredictions::of_config(&cfg, true, false);
        assert!(fasta.fragments && fasta.retention_time && !fasta.ion_mobility_missing);
        let w = native_predictor_warning(&fasta).expect("native fragments warn");
        assert!(w.contains("predict_frag.predictor = native"), "{w}");
        assert!(w.contains("rt_predictor = native"), "{w}");
        assert!(w.contains("unbounded"), "{w}");
        assert!(
            !w.contains("1/K0"),
            "no ion-mobility clause on 3D input: {w}"
        );

        // diaPASEF spectra without an IM predictor name the missing 1/K0 as well.
        let tims = NativePredictions::of_config(&cfg, true, true);
        assert!(tims.ion_mobility_missing);
        assert!(native_predictor_warning(&tims).unwrap().contains("1/K0"));

        // Library-input mode reads no predict_frag setting.
        let lib = NativePredictions::of_config(&cfg, false, true);
        assert_eq!(lib, NativePredictions::default());
        assert!(native_predictor_warning(&lib).is_none());

        // A sidecar fragment predictor does not warn, even with native retention times.
        let mut ms2pip = cfg.clone();
        ms2pip.predict_frag.predictor = FragPredictorKind::Ms2pip;
        let n = NativePredictions::of_config(&ms2pip, true, false);
        assert!(!n.fragments && n.retention_time);
        assert!(native_predictor_warning(&n).is_none());
    }

    /// A FASTA library on native fragments, or on native retention times that nothing
    /// replaces, is refused with the reason and the fixes; the engine defaults are such a
    /// library. An imported library, the opt-in, and a fine-tune or an explicit multi-head
    /// calibration over native retention times pass.
    #[test]
    fn a_fasta_library_on_the_native_models_is_refused() {
        let cfg = Config::default();
        let e = refuse_native_predictors(&cfg, true, "run")
            .unwrap_err()
            .to_string();
        for needle in [
            "run: refusing",
            "predict_frag.predictor = native",
            "predict_frag.rt_predictor = native",
            "engine defaults",
            "without --config",
            "configs/examples/fasta-sidecars.json",
            "configs/examples/diapasef.json",
            "mumdia doctor",
            "import_diann_lib.py",
            "predict_frag.allow_native_predictors = true",
        ] {
            assert!(e.contains(needle), "missing {needle:?}: {e}");
        }
        // Library-input mode reads no predict_frag setting.
        refuse_native_predictors(&cfg, false, "run").unwrap();
        // The explicit opt-in.
        let mut allowed = cfg.clone();
        allowed.predict_frag.allow_native_predictors = true;
        refuse_native_predictors(&allowed, true, "run").unwrap();

        // Native fragments alone are refused, without the defaults clause.
        let mut frag = cfg.clone();
        frag.predict_frag.rt_predictor = RtPredictorKind::Deeplc;
        let e = refuse_native_predictors(&frag, true, "build-library")
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("build-library: refusing"), "{e}");
        assert!(e.contains("fragment intensities"), "{e}");
        assert!(!e.contains("rt_predictor = native"), "{e}");
        assert!(!e.contains("engine defaults"), "{e}");

        // Native retention times under a sidecar fragment predictor are refused, unless a
        // fine-tune or an explicit multi-head calibration replaces them.
        let mut rt = cfg.clone();
        rt.predict_frag.predictor = FragPredictorKind::Ms2pip;
        let e = refuse_native_predictors(&rt, true, "predict-frag")
            .unwrap_err()
            .to_string();
        assert!(e.contains("rt_predictor = native"), "{e}");
        assert!(!e.contains("fragment intensities"), "{e}");
        assert!(!native_rt_is_replaced(&rt));
        let mut ft = rt.clone();
        ft.rt_im_train.finetune_deeplc = true;
        assert!(native_rt_is_replaced(&ft));
        refuse_native_predictors(&ft, true, "run").unwrap();
        let mut mh = rt.clone();
        mh.rt_im_train.multihead_calibration = Some(80);
        refuse_native_predictors(&mh, true, "run").unwrap();
        // `0` turns the calibration off and the automatic default does not run on a native
        // retention time, so neither replaces it.
        for heads in [Some(0), None] {
            let mut off = rt.clone();
            off.rt_im_train.multihead_calibration = heads;
            assert!(
                refuse_native_predictors(&off, true, "run").is_err(),
                "{heads:?}"
            );
        }
        // A replaced retention time does not excuse native fragments.
        let mut both = cfg.clone();
        both.rt_im_train.finetune_deeplc = true;
        let e = refuse_native_predictors(&both, true, "run")
            .unwrap_err()
            .to_string();
        assert!(e.contains("fragment intensities"), "{e}");
        assert!(!e.contains("rt_predictor = native"), "{e}");
    }

    /// The opt-in never reaches the canonical configuration, so a search allowed to use
    /// the native models keeps the configuration hash and the artifacts it had before the
    /// refusal existed.
    #[test]
    fn the_opt_in_is_not_part_of_the_canonical_configuration() {
        let cfg = Config::default();
        let allowed =
            Config::from_json(r#"{"predict_frag":{"allow_native_predictors":true}}"#).unwrap();
        assert!(allowed.predict_frag.allow_native_predictors);
        assert_eq!(allowed.canonical_json(), cfg.canonical_json());
        assert!(!cfg.canonical_json().contains("allow_native_predictors"));
    }

    /// The library's own report says how it was predicted; an imported or unreadable one
    /// says nothing native.
    #[test]
    fn the_library_report_names_the_native_models() {
        let d = scratch("lib");
        let f = d.join("frags.parquet");
        write(
            &d.join("frags.parquet.report.json"),
            serde_json::json!({"stage": "predict-frag",
                "params": {"fragment_predictor": "Native", "rt_predictor": "Deeplc"}}),
        );
        let n = NativePredictions::of_library(&f.to_string_lossy());
        assert!(n.fragments && !n.retention_time);
        assert!(likely_causes(&n).starts_with("native fragment intensities"));

        write(
            &d.join("frags.parquet.report.json"),
            serde_json::json!({"stage": "import", "params": {"fragment_predictor": "Native"}}),
        );
        assert_eq!(
            NativePredictions::of_library(&f.to_string_lossy()),
            NativePredictions::default()
        );
        assert_eq!(
            NativePredictions::of_library(&d.join("absent.parquet").to_string_lossy()),
            NativePredictions::default()
        );
        assert!(!likely_causes(&NativePredictions::default()).contains("native"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A gap is read for the two uncalibrated statuses only; a calibrated or missing
    /// `cal.json` gives none.
    #[test]
    fn a_gap_is_read_for_the_uncalibrated_statuses_only() {
        let d = scratch("gap");
        for (status, n_train, gap) in [
            (INSUFFICIENT_ANCHORS_STATUS, 0, true),
            (FALLBACK_FIXED_STATUS, 12, true),
            ("loess", 3000, false),
            ("linear", 80, false),
        ] {
            let cal = write(
                &d.join(format!("{status}.json")),
                serde_json::json!({"calibration_status": status, "n_train": n_train}),
            );
            let g = gap_of(&cal, "r1", "run");
            assert_eq!(g.is_some(), gap, "{status}");
            if let Some(g) = g {
                assert_eq!(g.calibration_status, status);
                assert_eq!(g.n_train, n_train);
                assert_eq!(g.run, "r1");
                assert_eq!(g.pass, "run");
                assert!(gap_warning(&g, &NativePredictions::default()).contains("r1"));
            }
        }
        assert!(gap_of(&d.join("absent.json").to_string_lossy(), "r", "run").is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    fn gap(run: &str, pass: &str, status: &str) -> RtCalibrationGap {
        RtCalibrationGap {
            run: run.into(),
            pass: pass.into(),
            calibration_status: status.into(),
            n_train: 0,
            cal_json: format!("{run}/cal.json"),
        }
    }

    /// The summary counts each pass on its own and is absent when every run calibrated.
    #[test]
    fn the_summary_counts_the_uncalibrated_runs_per_pass() {
        let n = NativePredictions {
            fragments: true,
            ..Default::default()
        };
        assert!(summary(&[], 12, &n).is_none());

        let mut gaps: Vec<RtCalibrationGap> = (0..12)
            .map(|i| {
                gap(
                    &format!("run{i}"),
                    "first_pass",
                    INSUFFICIENT_ANCHORS_STATUS,
                )
            })
            .collect();
        let s = summary(&gaps, 12, &n).unwrap();
        assert!(
            s.contains("12 of 12 run(s) in the first pass had no retention-time calibration (unbounded windows)"),
            "{s}"
        );
        assert!(s.contains("native fragment intensities"), "{s}");
        assert!(!s.contains("second pass"), "{s}");

        gaps.push(gap("run3", "second_pass", FALLBACK_FIXED_STATUS));
        let s = summary(&gaps, 12, &n).unwrap();
        assert!(
            s.contains("1 of 12 run(s) in the second pass had no retention-time calibration (the fixed fallback window)"),
            "{s}"
        );
        assert_eq!(s.matches("run3").count(), 1, "each run is named once: {s}");

        let single = [gap("a", "run", INSUFFICIENT_ANCHORS_STATUS)];
        let s = summary(&single, 1, &NativePredictions::default()).unwrap();
        assert!(s.contains("1 of 1 run(s) in this search"), "{s}");
        assert!(s.contains("Runs: a."), "{s}");
    }
}
