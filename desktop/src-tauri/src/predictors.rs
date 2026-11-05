//! The predictors a FASTA search on the built-in path builds its library with, chosen and
//! checked before the search starts.
//!
//! # Why this exists
//!
//! The Search screen derives a FASTA search's configuration from the selected preset, and
//! the preset selected by default is the one for an imported library, which names no
//! fragment predictor. While DIA-NN was unavailable the built-in path was taken with that
//! preset, so the library was predicted with the engine's native models. A desktop user
//! searched twelve diaPASEF runs that way: the native fragment intensities left the seed
//! of every run without a confident anchor, every run was extracted with an unbounded
//! retention-time window, and the search took about ten hours and reported a small
//! fraction of what the data hold, with nothing on the screen saying why.
//!
//! So the built-in path no longer inherits a preset that names no fragment predictor. It
//! takes the first recipe whose interpreters the engine finds (`mumdia doctor`): the
//! diaPASEF recipe for a timsTOF diaPASEF `.d`, then MS2PIP with DeepLC, then AlphaPeptDeep
//! with DeepLC. A preset that names a fragment predictor is the user's choice and is kept.
//! When the search would still run on native predictions, [`Plan::blocked`] says why and
//! what to do instead, and the interface does not start it: the engine refuses a FASTA
//! library on the native models (`rt_health::refuse_native_predictors`), and the
//! application never passes the engine's opt-in. On a diaPASEF `.d` without the diaPASEF
//! settings, [`Plan::warning`] says what that costs, and the interface asks before it
//! starts.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

/// The diaPASEF recipe (`configs/examples/diapasef.json`): AlphaPeptDeep, DeepLC, IM2Deep.
pub const DIAPASEF_RECIPE: &str = "diapasef.json";

/// The FASTA recipes for every other acquisition, in order of preference.
pub const FASTA_RECIPES: &[&str] = &["fasta-sidecars.json", "fasta-peptdeep.json"];

/// What a configuration predicts its library with, as the engine would read it: a key
/// that is absent takes the engine default (`native`, `native`, `none`).
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub struct Predictors {
    pub fragments: String,
    pub retention_time: String,
    pub ion_mobility: String,
    /// Whether a native library retention time is replaced after the build, as the engine
    /// decides it: a fine-tune (`rt_im_train.finetune_deeplc`) or an explicit multi-head
    /// calibration (`rt_im_train.multihead_calibration` > 0).
    pub retention_time_replaced: bool,
}

impl Predictors {
    pub fn of(cfg: &Value) -> Self {
        let get = |k: &str, default: &str| {
            cfg.get("predict_frag")
                .and_then(|p| p.get(k))
                .and_then(|v| v.as_str())
                .unwrap_or(default)
                .to_string()
        };
        Self {
            fragments: get("predictor", "native"),
            retention_time: get("rt_predictor", "native"),
            ion_mobility: get("im_predictor", "none"),
            retention_time_replaced: rt_replaced(cfg),
        }
    }

    pub fn native_fragments(&self) -> bool {
        self.fragments == "native"
    }

    pub fn native_retention_time(&self) -> bool {
        self.retention_time == "native"
    }

    pub fn predicts_ion_mobility(&self) -> bool {
        self.ion_mobility != "none"
    }

    /// Whether the engine would refuse to build the library with these predictors: native
    /// fragment intensities, or native retention times that nothing replaces.
    pub fn refused(&self) -> bool {
        self.native_fragments() || (self.native_retention_time() && !self.retention_time_replaced)
    }
}

/// `rt_im_train.finetune_deeplc = true` or `rt_im_train.multihead_calibration` > 0.
fn rt_replaced(cfg: &Value) -> bool {
    let rt = cfg.get("rt_im_train");
    let finetune = rt
        .and_then(|r| r.get("finetune_deeplc"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let heads = rt
        .and_then(|r| r.get("multihead_calibration"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    finetune || heads > 0
}

/// Whether a configuration names its fragment predictor, which makes the predictors the
/// user's choice rather than the engine default.
pub fn names_fragment_predictor(cfg: &Value) -> bool {
    cfg.get("predict_frag")
        .and_then(|p| p.get("predictor"))
        .is_some()
}

/// The plan for one FASTA search on the built-in path.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct Plan {
    /// The configuration the run's settings are derived from: the recipe chosen here, or
    /// the selected preset unchanged (`None` for the engine defaults).
    pub base: Option<String>,
    /// The recipe chosen in place of the selected preset, by file name.
    pub recipe: Option<String>,
    /// One or two sentences for the Search screen: what will predict the library.
    pub note: String,
    /// Whether any selected spectra file is a timsTOF diaPASEF `.d`.
    pub diapasef: bool,
    pub predictors: Predictors,
    /// Set when the search would build its library with the native models, which the
    /// engine refuses. The interface shows it and does not start the search or the library
    /// build: there is no override in the application.
    pub blocked: Option<String>,
    /// Set when the search would run on a diaPASEF `.d` without predicted 1/K0. The
    /// interface shows it and asks before starting.
    pub warning: Option<String>,
}

/// Why a FASTA search on the built-in path cannot start with these predictors, and what
/// to do instead, or `None` when the engine would build the library.
pub fn blocked(p: &Predictors, diapasef: bool) -> Option<String> {
    if !p.refused() {
        return None;
    }
    let mut what = Vec::new();
    if p.native_fragments() {
        what.push("native fragment intensities");
    }
    if p.native_retention_time() && !p.retention_time_replaced {
        what.push("native retention times");
    }
    let components = if diapasef {
        "AlphaPeptDeep (or MS2PIP) with the analysis components, so the diaPASEF recipe \
         (AlphaPeptDeep, DeepLC and IM2Deep) can be used"
    } else {
        "MS2PIP or AlphaPeptDeep with the analysis components (DeepLC)"
    };
    Some(format!(
        "This search cannot start: it would predict its library with the engine's {}. The \
         engine no longer builds a library with its native models, because they usually \
         leave the first-pass search without confident anchors, so the search is many \
         times slower and finds far fewer peptides. To search this FASTA, install \
         {components} on the Setup screen, choose \"Predict it with DIA-NN\" above to build \
         the library with DIA-NN, or convert a DIA-NN library on the From a spectral \
         library tab.",
        what.join(" and ")
    ))
}

/// The warning for a search on diaPASEF spectra without predicted 1/K0, or `None`. A
/// search the engine would refuse is [`blocked`] instead.
pub fn warning(p: &Predictors, diapasef: bool) -> Option<String> {
    if p.refused() {
        return None;
    }
    if diapasef && !p.predicts_ion_mobility() {
        return Some(
            "These are diaPASEF spectra, and this search predicts no ion mobility (1/K0) for \
             its library and does not use the diaPASEF settings, so it uses less of the data \
             than the diaPASEF recipe. Install the analysis components and AlphaPeptDeep (or \
             MS2PIP) on the Setup screen, so the diaPASEF recipe (AlphaPeptDeep, DeepLC and \
             IM2Deep) can be used, or search a library predicted by DIA-NN or imported in the \
             Library tab."
                .to_string(),
        );
    }
    None
}

/// Choose the configuration a FASTA search on the built-in path derives from.
///
/// `selected` is the preset chosen on the Search screen and `selected_cfg` its content (an
/// empty object for the engine defaults). `replaceable` says whether that choice is one the
/// application shipped (or the engine defaults), which may be replaced by a recipe; a
/// settings file the user saved is never replaced, because replacing it would drop the
/// user's edits, and it is warned about instead. `usable` answers whether the engine finds
/// every interpreter a recipe requires; it is asked only when a recipe may be chosen.
pub fn plan(
    selected: Option<&str>,
    selected_cfg: &Value,
    replaceable: bool,
    diapasef: bool,
    recipes_dir: &Path,
    usable: &dyn Fn(&Path) -> bool,
) -> Plan {
    let keep = |note: String| {
        let predictors = Predictors::of(selected_cfg);
        Plan {
            base: selected.map(str::to_string),
            recipe: None,
            note,
            diapasef,
            blocked: blocked(&predictors, diapasef),
            warning: warning(&predictors, diapasef),
            predictors,
        }
    };
    if names_fragment_predictor(selected_cfg) {
        let p = Predictors::of(selected_cfg);
        return keep(format!(
            "The selected preset predicts the library with {} (retention time: {}).",
            p.fragments, p.retention_time
        ));
    }
    if !replaceable {
        return keep(
            "The selected settings name no fragment predictor, so the library would be \
             predicted with the engine's native models, which it refuses. Set \
             predict_frag.predictor and predict_frag.rt_predictor in Settings, or choose the \
             fasta-sidecars or diapasef preset."
                .to_string(),
        );
    }
    let mut candidates: Vec<&str> = Vec::new();
    if diapasef {
        candidates.push(DIAPASEF_RECIPE);
    }
    candidates.extend_from_slice(FASTA_RECIPES);
    for (i, name) in candidates.iter().enumerate() {
        let path = recipes_dir.join(name);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(cfg) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if !usable(&path) {
            continue;
        }
        let predictors = Predictors::of(&cfg);
        let stem = name.trim_end_matches(".json");
        let mut note = format!(
            "The library is predicted with the {stem} recipe: {} fragments, {} retention times",
            predictors.fragments, predictors.retention_time
        );
        if predictors.predicts_ion_mobility() {
            note.push_str(&format!(", {} ion mobility", predictors.ion_mobility));
        }
        note.push('.');
        if diapasef && i > 0 {
            note.push_str(
                " The diaPASEF recipe was not used: the engine did not find every \
                 interpreter it needs (AlphaPeptDeep, DeepLC and IM2Deep).",
            );
        }
        return Plan {
            base: Some(path.display().to_string()),
            recipe: Some(name.to_string()),
            note,
            diapasef,
            blocked: blocked(&predictors, diapasef),
            warning: warning(&predictors, diapasef),
            predictors,
        };
    }
    keep(
        "No fragment predictor was found (MS2PIP or AlphaPeptDeep, with DeepLC), so the \
         built-in predictors cannot build this library."
            .to_string(),
    )
}

/// Whether the engine finds and can import every interpreter `config` requires: `mumdia
/// doctor --json`'s own verdict, in the environment a run gets.
pub fn doctor_ok(engine: &Path, config: &Path) -> bool {
    let mut cmd = crate::engine::command(engine);
    crate::components::stamp_env(&mut cmd);
    let Ok(out) = cmd
        .arg("doctor")
        .arg("--json")
        .arg("--config")
        .arg(config)
        .output()
    else {
        return false;
    };
    serde_json::from_slice::<Value>(&out.stdout)
        .ok()
        .and_then(|v| v.get("ok").and_then(|b| b.as_bool()))
        .unwrap_or(false)
}

/// Read a preset's content, or an empty object for the engine defaults.
pub fn read_preset(path: Option<&str>) -> Result<Value, String> {
    match path.filter(|p| !p.trim().is_empty()) {
        None => Ok(Value::Object(Default::default())),
        Some(p) => {
            let text = std::fs::read_to_string(p)
                .map_err(|e| format!("could not read the preset {p}: {e}"))?;
            serde_json::from_str(&text)
                .map_err(|e| format!("the preset {p} is not valid JSON: {e}"))
        }
    }
}

/// Whether any of the selected spectra files is a timsTOF diaPASEF `.d`.
pub fn any_diapasef(mzml: &[String]) -> bool {
    mzml.iter()
        .any(|m| crate::thermo::is_native_diapasef(&PathBuf::from(m)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn recipes(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "mumdia-console-recipes-{}-{tag}",
            std::process::id()
        ));
        std::fs::create_dir_all(&d).unwrap();
        for (name, body) in [
            (
                DIAPASEF_RECIPE,
                json!({"predict_frag": {"predictor": "peptdeep", "rt_predictor": "deeplc", "im_predictor": "im2deep"}}),
            ),
            (
                "fasta-sidecars.json",
                json!({"predict_frag": {"predictor": "ms2pip", "rt_predictor": "deeplc"}}),
            ),
            (
                "fasta-peptdeep.json",
                json!({"predict_frag": {"predictor": "peptdeep", "rt_predictor": "deeplc"}}),
            ),
        ] {
            std::fs::write(d.join(name), body.to_string()).unwrap();
        }
        d
    }

    fn named(p: &Path) -> String {
        p.file_name().unwrap().to_string_lossy().into_owned()
    }

    /// The default preset (an imported-library one, naming no fragment predictor) is no
    /// longer inherited: the first usable recipe replaces it, the diaPASEF recipe first
    /// for a diaPASEF `.d`.
    #[test]
    fn a_preset_without_a_fragment_predictor_takes_the_first_usable_recipe() {
        let dir = recipes("first");
        let library_preset = json!({"rescore": {"classifier": "nn_torch"}});
        let all = |_: &Path| true;

        let p = plan(
            Some("diann-library.json"),
            &library_preset,
            true,
            true,
            &dir,
            &all,
        );
        assert_eq!(p.recipe.as_deref(), Some(DIAPASEF_RECIPE));
        assert_eq!(p.predictors.ion_mobility, "im2deep");
        assert!(p.warning.is_none(), "{:?}", p.warning);
        assert!(p.blocked.is_none(), "{:?}", p.blocked);

        let p = plan(
            Some("diann-library.json"),
            &library_preset,
            true,
            false,
            &dir,
            &all,
        );
        assert_eq!(p.recipe.as_deref(), Some("fasta-sidecars.json"));
        assert!(p.warning.is_none());
        assert!(p.blocked.is_none());

        // No AlphaPeptDeep: the diaPASEF recipe is skipped, MS2PIP is used, and the missing
        // 1/K0 is a warning.
        let no_peptdeep = |path: &Path| named(path) == "fasta-sidecars.json";
        let p = plan(None, &json!({}), true, true, &dir, &no_peptdeep);
        assert_eq!(p.recipe.as_deref(), Some("fasta-sidecars.json"));
        assert!(
            p.note.contains("diaPASEF recipe was not used"),
            "{}",
            p.note
        );
        assert!(p.warning.as_deref().unwrap().contains("1/K0"));
        assert!(
            p.blocked.is_none(),
            "a missing 1/K0 is asked about, not refused"
        );

        // Only AlphaPeptDeep on 3D input: the second FASTA recipe.
        let only_peptdeep = |path: &Path| named(path) == "fasta-peptdeep.json";
        let p = plan(None, &json!({}), true, false, &dir, &only_peptdeep);
        assert_eq!(p.recipe.as_deref(), Some("fasta-peptdeep.json"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Nothing usable: the preset is kept, it would build the library on the native models,
    /// and the search is blocked with the reason and the fixes, not asked about.
    #[test]
    fn nothing_usable_keeps_the_preset_and_blocks_the_search() {
        let dir = recipes("none");
        let none = |_: &Path| false;
        let p = plan(
            Some("diann-library.json"),
            &json!({}),
            true,
            true,
            &dir,
            &none,
        );
        assert_eq!(p.base.as_deref(), Some("diann-library.json"));
        assert_eq!(p.recipe, None);
        assert!(p.predictors.native_fragments());
        let b = p.blocked.expect("native predictions block the search");
        for needle in [
            "cannot start",
            "native fragment intensities and native retention times",
            "Setup screen",
            "Predict it with DIA-NN",
            "From a spectral library",
            "diaPASEF recipe",
        ] {
            assert!(b.contains(needle), "missing {needle:?}: {b}");
        }
        assert!(
            p.warning.is_none(),
            "a blocked search is not also asked about: {:?}",
            p.warning
        );

        let p = plan(None, &json!({}), true, false, &dir, &none);
        assert_eq!(p.base, None);
        let b = p.blocked.unwrap();
        assert!(!b.contains("diaPASEF"), "{b}");
        assert!(p.warning.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The application blocks what the engine refuses, and nothing else: native fragments,
    /// or native retention times that no fine-tune or explicit multi-head calibration
    /// replaces. The engine's opt-in does not unblock a search in the application.
    #[test]
    fn the_block_follows_the_engine_refusal() {
        let blocked_of = |cfg: Value| blocked(&Predictors::of(&cfg), false);
        assert!(blocked_of(json!({})).is_some());
        let rt_native = json!({"predict_frag": {"predictor": "ms2pip"}});
        let b = blocked_of(rt_native).unwrap();
        assert!(b.contains("native retention times"), "{b}");
        assert!(!b.contains("fragment intensities"), "{b}");
        assert!(blocked_of(json!({"predict_frag": {"predictor": "ms2pip"},
                                  "rt_im_train": {"finetune_deeplc": true}}))
        .is_none());
        assert!(blocked_of(json!({"predict_frag": {"predictor": "peptdeep"},
                                  "rt_im_train": {"multihead_calibration": 80}}))
        .is_none());
        for heads in [json!(0), json!(null)] {
            assert!(blocked_of(json!({"predict_frag": {"predictor": "ms2pip"},
                                      "rt_im_train": {"multihead_calibration": heads}}))
            .is_some());
        }
        let b = blocked_of(json!({"predict_frag": {"rt_predictor": "deeplc"}})).unwrap();
        assert!(b.contains("native fragment intensities"), "{b}");
        assert!(!b.contains("retention times."), "{b}");
        assert!(blocked_of(json!({"predict_frag": {"predictor": "native",
                                  "allow_native_predictors": true}}))
        .is_some());
        assert!(blocked_of(json!({"predict_frag": {"predictor": "ms2pip",
                                  "rt_predictor": "deeplc"}}))
        .is_none());
    }

    /// A preset that names its fragment predictor is the user's choice: kept, never probed,
    /// and blocked when it is native.
    #[test]
    fn a_preset_that_names_its_predictor_is_kept() {
        let dir = recipes("kept");
        let never = |_: &Path| -> bool { panic!("an explicit choice is not probed") };
        let ms2pip = json!({"predict_frag": {"predictor": "ms2pip", "rt_predictor": "deeplc"}});
        let p = plan(Some("mine.json"), &ms2pip, false, false, &dir, &never);
        assert_eq!(p.base.as_deref(), Some("mine.json"));
        assert_eq!(p.recipe, None);
        assert!(p.warning.is_none());

        // The same preset on a diaPASEF .d predicts no 1/K0.
        let p = plan(Some("mine.json"), &ms2pip, false, true, &dir, &never);
        assert!(p.warning.unwrap().contains("1/K0"));

        let native = json!({"predict_frag": {"predictor": "native"}});
        let p = plan(Some("native.json"), &native, true, false, &dir, &never);
        assert!(p.blocked.unwrap().contains("native fragment intensities"));
        assert!(p.warning.is_none());

        // A saved settings file that names no predictor is kept, not replaced, and blocked:
        // replacing it would drop the user's edits.
        let saved = json!({"extract": {"gate_min_score": 0.5}});
        let p = plan(Some("saved.json"), &saved, false, false, &dir, &never);
        assert_eq!(p.base.as_deref(), Some("saved.json"));
        assert_eq!(p.recipe, None);
        assert!(p.note.contains("Set predict_frag.predictor"), "{}", p.note);
        assert!(p.blocked.unwrap().contains("native fragment intensities"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The compiled-in recipes the plan reads are the shipped example configurations, and
    /// each names the predictors this module expects of it.
    #[test]
    fn the_shipped_recipes_name_their_predictors() {
        let diapasef: Value =
            serde_json::from_str(include_str!("../../../configs/examples/diapasef.json")).unwrap();
        let p = Predictors::of(&diapasef);
        assert!(!p.native_fragments() && !p.native_retention_time() && p.predicts_ion_mobility());
        assert!(!p.refused());
        for body in [
            include_str!("../../../configs/examples/fasta-sidecars.json"),
            include_str!("../../../configs/examples/fasta-peptdeep.json"),
        ] {
            let v: Value = serde_json::from_str(body).unwrap();
            assert!(names_fragment_predictor(&v));
            let p = Predictors::of(&v);
            assert!(!p.native_fragments() && !p.native_retention_time());
            assert!(!p.refused());
        }
        // The default preset names none, which is why the plan replaces it.
        let lib: Value =
            serde_json::from_str(include_str!("../../../configs/examples/diann-library.json"))
                .unwrap();
        assert!(!names_fragment_predictor(&lib));
    }
}
