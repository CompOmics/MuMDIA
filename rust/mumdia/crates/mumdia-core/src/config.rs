//! Typed configuration (docs/02_config_and_data_model.md).
//!
//! One serde structure with per-stage sections, `#[serde(default)]` on every
//! field, and `deny_unknown_fields` so misconfiguration fails loudly. Every
//! choice point is an enum backed by a strategy; MVP ships only the strategies
//! MVP needs, with MVP-conservative defaults: fixed tolerances, one documented
//! decoy scheme, the `minimal` feature set.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Strategy enums
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoyStrategy {
    /// Reverse the sequence keeping the C-terminal residue fixed. Documented,
    /// clean-room default for MVP (docs/14_build_test_deploy_gotchas.md). No
    /// borrowed map.
    #[default]
    Reverse,
    /// Deterministic seeded shuffle of the interior residues.
    Scramble,
    /// DIA-NN terminal-residue fragment m/z shift. Deferred: license-checked
    /// addition (docs/14_build_test_deploy_gotchas.md), not part of MVP.
    DiannShift,
    None,
}

/// Fragment-matcher backend for search-seed and extract (docs/06_predict_frag_index_matchers.md).
/// Default `Fragindex` (log-bin CSR matcher): on narrow-window DIA it is ~1.95x
/// faster in search-seed and ~1.26x in extract with essentially unchanged IDs
/// (HYE B_01: peptides -0.1%); `Bucketed` is the previous `Library::page_search`
/// path (retained for A/B and for the AIF full-range-window case, where the
/// predicate difference shifts IDs more).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatcherKind {
    Bucketed,
    #[default]
    Fragindex,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Enzyme {
    /// Trypsin/P: cut after K or R (including before P).
    #[default]
    TrypsinP,
    /// Classic trypsin: cut after K or R but not before P.
    Trypsin,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationMethod {
    #[default]
    Loess,
    Linear,
    None,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureSet {
    /// MVP feature set (docs/10_features.md).
    #[default]
    Minimal,
    Rich,
    /// Minimal + Rich + the extended battery (DIA-NN / OpenSWATH / AlphaDIA /
    /// MS2Rescore / OktoberFest analogs + novel families) from the per-family
    /// modules in `stages/features/`. Superset, opt-in; the classifier picks the
    /// signal it can use (esp. under the nonlinear `Entrapment` rescorer).
    Extended,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RtPredictorKind {
    /// Native additive retention-coefficient model (no Python). MVP default so
    /// the engine runs with zero external runtime dependencies.
    #[default]
    Native,
    /// DeepLC Python sidecar (docs/13_sidecars.md).
    Deeplc,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FragPredictorKind {
    /// Native heuristic intensity model (no Python). MVP default.
    #[default]
    Native,
    /// MS2PIP Python sidecar (docs/13_sidecars.md).
    Ms2pip,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RescorerKind {
    /// Native semi-supervised linear rescorer + target-decoy q-values. MVP
    /// default (always available).
    #[default]
    NativeTda,
    /// Mokapot Python sidecar (docs/13_sidecars.md).
    Mokapot,
    /// PyTorch semi-supervised MLP sidecar (`nn_rescore_worker.py`): a nonlinear
    /// Percolator/mokapot-style rescorer (CV folds + iterative positive
    /// re-selection). On the E.coli benchmark it beats the linear mokapot model on
    /// the same PIN, and — being robust to an unfiltered pool — gains further when
    /// the extraction gate is opened. Same positional-CLI PIN contract as Mokapot;
    /// requires `rescore.python` to point at an interpreter with torch.
    NnTorch,
    /// External percolator.exe over the PIN file.
    Percolator,
    /// Spike-in (entrapment) negative rescorer: treat foreign-proteome PSMs
    /// (identified by `entrapment_marker`) as real negatives, train a nonlinear
    /// GBM sidecar (out-of-fold by base peptide) or a native linear fallback,
    /// and report entrapment-calibrated q-values. The chimeric false matches
    /// that in-silico decoys under-model appear as real negatives here, so it
    /// closes the FDR-validity gap the decoy schemes cannot.
    Entrapment,
}

/// Fragment-peak apportionment when one observed MS2 peak matches the fragments
/// of several co-isolated, co-eluting candidates (near-universal in wide-window
/// DIA: ~98% of fragment m/z collide within tolerance). Decides how the peak's
/// intensity is shared, to stop a chimeric candidate borrowing a real peptide's
/// peak wholesale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PeakClaim {
    /// Every matching candidate gets the full peak intensity (legacy default).
    #[default]
    None,
    /// Winner-take-all: only the candidate with the highest predicted intensity
    /// for its matching fragment gets the peak; the rest get nothing.
    WinnerPredictedIntensity,
    /// Soft apportionment: split the peak intensity across claimants in
    /// proportion to their predicted intensity for the matching fragment.
    Proportional,
    /// Presence-aware winner-take-all (two-pass): a first pass builds each
    /// candidate's per-scan elution profile (summed matched intensity); the peak
    /// then goes to the claimant most eluting at that scan (highest profile
    /// height, i.e. best corroborated by its OTHER fragments), not the one that
    /// merely predicts the brightest ion there.
    CoelutionWinner,
    /// Presence-aware soft apportionment (two-pass): split the peak across
    /// claimants in proportion to their per-scan elution-profile height.
    CoelutionProportional,
    /// Margin-gated co-elution winner (two-pass): winner-take-all ONLY when the
    /// top eluter's profile height dominates the runner-up by `peak_claim_margin`
    /// (else the peak stays shared among all claimants, as in `None`). Avoids
    /// stripping real peptides at ambiguous peaks where no candidate clearly owns
    /// the elution.
    CoelutionWinnerMargin,
    /// Multi-cue co-elution winner (two-pass, modular fragment-competition framework).
    /// The per-claimant competition weight is the elution profile height multiplied
    /// by the composable cues enabled in [`ClaimCues`] (sub-tolerance m/z proximity,
    /// RT prior, isotope coherence, MS1 precursor support, ...), each defaulting to
    /// 1.0 so this reduces to `CoelutionWinner` when no cue is enabled. Winner-take-all
    /// on the composite weight when `reassign` is set.
    CoelutionMultiCue,
    /// Spectrum-centric demix redistribution (two-pass, destructive). At each scan the
    /// co-isolated candidate x fragment design matrix is assembled and solved by
    /// non-negative least squares; each shared peak's intensity is then split among its
    /// claimants in proportion to `beta_c * D[peak,c]` (the joint deconvolution) instead
    /// of stripped winner-take-all. The smooth, principled destructive mode - the CHIMERYS
    /// coefficient split, made chromatographic and clean-room. Always redistributes; the
    /// demix FEATURES are the separate `emit_demix_features` path. Deterministic (sorted
    /// candidate columns, ridge NNLS).
    CoelutionDemix,
    /// Shadow-subtraction redistribution (two-pass, destructive, no solver). At each scan,
    /// each co-eluter's abundance is estimated from the channels it ALONE claims (its unique
    /// ions, `a_p = median y/D` over those); every candidate then keeps, at each of its
    /// channels, `max(0, y - sum_{p != c} a_p * D[peak,p])` - its intensity minus the
    /// interferers' estimated contributions. Unlike winner-take-all, several real co-eluters
    /// can both retain signal at a shared peak; unlike the NNLS demix it needs no solve, so
    /// it is cheap. A candidate with no unique ion cannot be estimated and contributes no
    /// subtraction. The gentle destructive mode. Deterministic; default off.
    CoelutionShadow,
}

/// Composable per-claimant weight cues for [`PeakClaim::CoelutionMultiCue`] (the
/// modular fragment-competition framework). Each cue is label-blind (reads only
/// observed/predicted m/z + intensity, RT, MS1) so target/decoy exchangeability is
/// preserved, and each defaults OFF (weight 1.0) so the composite weight reduces to
/// the plain elution-profile height. Enable cues incrementally and validate as
/// non-destructive features before any destructive/default use.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClaimCues {
    /// Sub-tolerance m/z proximity (S3): weight a claimant by
    /// `exp(-(ppm_err/sigma)^2)`, where `ppm_err` is the signed ppm offset of the
    /// observed peak from this claimant's predicted fragment m/z. Two collided
    /// fragments share a peak only because both fall within `frag_tol`, but the
    /// observed peak sits at the true owner's m/z; the sub-tolerance offset is a
    /// novel apportionment weight (engines use ppm only as a binary gate).
    pub mz_close: bool,
    /// Gaussian sigma (ppm) for the `mz_close` cue. Default 5 ppm.
    pub mz_close_sigma_ppm: f64,
    /// DeepLC retention-time prior (S3): weight a claimant by
    /// `exp(-(rt - rt_pred)^2 / 2 tau^2)`, where `rt_pred` is the candidate's
    /// calibrated predicted RT. A co-isolated interferent whose predicted RT is far
    /// from the current scan gets a low weight even if it briefly co-elutes, so a
    /// shared peak is apportioned toward the candidate the RT model actually places
    /// there. No-op where the predicted RT is unset (0).
    pub rt_prior: bool,
    /// Gaussian sigma (seconds) for the `rt_prior` cue. Default 30 s.
    pub rt_prior_tau_s: f64,
    /// MS1 precursor-envelope support (S4, cross-dimension): weight a claimant by
    /// whether its own precursor isotope envelope (mono + a plausible +1/mono ratio)
    /// is actually present in the nearest MS1 scan. A shift/reverse decoy has a
    /// well-defined precursor m/z but no real co-eluting MS1 precursor, so its support
    /// is noise, starving its MS2 claim via an orthogonal dimension that is nearly
    /// impossible to fake. No-op when no MS1 is provided. Down-weights (never zeroes)
    /// so a genuinely MS1-poor real peptide is not eliminated.
    pub ms1_support: bool,
    /// DESTRUCTIVE redistribution for `CoelutionMultiCue`. When true, the cue-weighted
    /// arbitration rewrites the extracted peak intensities (winner-take-all on the
    /// composite weight), instead of only emitting the apportioned/contested features.
    /// The competed evidence then feeds EVERY downstream feature (co-elution, spectral,
    /// mass-accuracy, ...), so this is the impactful form. Off by default; changes the
    /// search/FDR evidence, so it is entrapment-gated per CLAUDE.md.
    pub reassign: bool,
    /// Uniqueness-seeded EM apportionment (S2): number of fixed-point iterations that
    /// re-seed each candidate's per-scan elution profile from its APPORTIONED (not full)
    /// intensity before the final arbitration. The plain profile is built from full
    /// intensities, so a borrowing candidate's profile is inflated by the very peaks it
    /// borrows; re-seeding from the cue-weighted share removes that feedback, while
    /// uncontested (single-claimant) peaks contribute full intensity every iteration as
    /// an immovable anchor. 0 (default) disables EM (single-pass profile). Deterministic
    /// (fixed N); applies under `CoelutionMultiCue`.
    pub apportion_em_iters: u32,
}
impl Default for ClaimCues {
    fn default() -> Self {
        Self {
            mz_close: false,
            mz_close_sigma_ppm: 5.0,
            rt_prior: false,
            rt_prior_tau_s: 30.0,
            ms1_support: false,
            reassign: false,
            apportion_em_iters: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-stage config
// ---------------------------------------------------------------------------

fn t<T>() -> T
where
    T: Default,
{
    T::default()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DecoyConfig {
    pub strategy: DecoyStrategy,
}
impl Default for DecoyConfig {
    fn default() -> Self {
        Self { strategy: t() }
    }
}

/// Vendor-format conversion, read by every subcommand that takes a spectra path
/// (`raw.rs`; docs/04_convert.md, "Vendor formats").
///
/// The engine itself reads mzML only, deliberately: `mzdata` is pinned to its
/// pure-Rust `mzml` + `miniz_oxide` features so the build needs no C or .NET
/// toolchain, and its vendor readers would reintroduce both. A vendor file is
/// therefore converted to mzML first by an external converter run as a child
/// process, in the same way the Python sidecars are: ThermoRawFileParser for Thermo
/// `.raw`, ProteoWizard `msconvert` for everything else and as the Thermo fallback.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConvertConfig {
    /// Path to the ThermoRawFileParser executable, or `"auto"` to search.
    ///
    /// `"auto"` looks at `MUMDIA_THERMO_PARSER`, then beside the engine binary,
    /// then on `PATH`. Empty means the same as `"auto"`; a real path is used
    /// verbatim and its absence is an error rather than a silent fallback, because
    /// a fallback would convert with a different program than the one asked for and
    /// vendor conversion is not reproducible across converters.
    pub thermo_raw_parser: String,
    /// Path to ProteoWizard `msconvert`, or `"auto"` to search.
    ///
    /// Used for every vendor format except Thermo, which prefers
    /// ThermoRawFileParser: Bruker `.d`, SCIEX `.wiff`, Agilent `.d` and Waters
    /// `.raw`. It is also the Thermo fallback when no ThermoRawFileParser is found.
    ///
    /// `"auto"` searches `MUMDIA_MSCONVERT`, beside the engine binary, the
    /// version-stamped ProteoWizard directories under Program Files on Windows
    /// (newest first), then `PATH`.
    ///
    /// MuMDIA never ships or downloads ProteoWizard. Its vendor readers bundle the
    /// instrument vendors' own libraries under the vendors' licence terms, which the
    /// user accepts when obtaining it, and automating that acceptance is not
    /// MuMDIA's to do.
    pub msconvert: String,
    /// Extra arguments appended to every `msconvert` invocation.
    ///
    /// An escape hatch, not a tuning surface. The per-vendor defaults already
    /// request indexed 64-bit zlib mzML, vendor peak picking where it exists, and
    /// `--combineIonMobilitySpectra` for Bruker. Use this for something the defaults
    /// cannot express, such as an `--filter` that trims an acquisition. Arguments
    /// are passed through verbatim and are not validated.
    pub msconvert_args: Vec<String>,
    /// Reuse an mzML that already sits beside the `.raw` and is newer than it.
    ///
    /// On by default: conversion is minutes per file and its output is
    /// deterministic given the same converter, so re-running a search should not
    /// pay for it twice. Turn it off when the neighbouring mzML may have come from
    /// a different converter or a different `.raw` of the same name.
    pub reuse_converted: bool,
}
impl Default for ConvertConfig {
    fn default() -> Self {
        Self {
            thermo_raw_parser: "auto".to_string(),
            msconvert: "auto".to_string(),
            msconvert_args: Vec::new(),
            reuse_converted: true,
        }
    }
}

/// Sequence-tag prescan (`mumdia prescan`). Prunes modification-bearing candidates that have no
/// anchored tag support in a given run, before the per-run library is assembled.
///
/// The screen is deliberately blind to target/decoy label: tags are emitted in both orientations
/// and a reverse decoy preserves composition and precursor m/z, so a decoy survives exactly when
/// its target does. That keeps exchangeability, and therefore downstream FDR, intact.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PrescanConfig {
    /// Peak-delta match tolerance in Da. Permissive on purpose: a false tag only fails to prune,
    /// while a missed tag discards a real candidate with no way to recover it downstream.
    pub tol_da: f64,
    /// Widen each candidate's RT window by this many seconds before binning. The window comes
    /// from a calibration fitted on a different run, and `cal.json` residuals are in-sample and
    /// roughly 3x optimistic, so size this from out-of-sample RT error, not from the reported fit.
    pub rt_slack_s: f64,
    /// RT bin width for the observed-tag index.
    pub rt_bin_s: f64,
    /// Most intense peaks per MS2 used to build tags (0 = all). This bounds the O(peaks^2) delta
    /// search and is NOT destructive: it only affects tag construction, never the spectra artifact
    /// that extraction later reads.
    pub top_peaks: usize,
    /// Residue:UniModName entries that may appear in a screened peptidoform, e.g. `C:Carbamidomethyl`.
    /// A peptidoform carrying anything outside this set plus `anchor_mods` is dropped rather than
    /// screened on a partially understood sequence.
    pub mods: Vec<String>,
    /// Residue:UniModName entries the screen anchors ON. Only trimers covering one of these
    /// positions count as evidence, so backbone signal cannot keep a modified hypothesis alive.
    pub anchor_mods: Vec<String>,
}
impl Default for PrescanConfig {
    fn default() -> Self {
        Self {
            tol_da: 0.005,
            rt_slack_s: 150.0,
            rt_bin_s: 25.0,
            top_peaks: 150,
            mods: vec!["C:Carbamidomethyl".to_string(), "M:Oxidation".to_string()],
            anchor_mods: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DigestConfig {
    pub enzyme: Enzyme,
    pub missed_cleavages: u32,
    pub min_len: usize,
    pub max_len: usize,
    pub decoy: DecoyConfig,
    /// N-terminal methionine excision: when a protein begins with `M`, also emit
    /// the initiator-Met-removed form of its N-terminal peptides. The initiator
    /// methionine is cleaved in vivo for most proteins, so search engines
    /// (including DIA-NN via `--met-excision`) enumerate both forms. Omitting it
    /// makes the search database structurally miss those excised peptides.
    pub n_term_met_excision: bool,
}
impl Default for DigestConfig {
    fn default() -> Self {
        Self {
            enzyme: t(),
            missed_cleavages: 2,
            min_len: 5,
            max_len: 50,
            decoy: t(),
            n_term_met_excision: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PeptidoformsConfig {
    /// UniMod names applied to every matching residue (residue -> mod name).
    pub fixed_mods: Vec<ResidueMod>,
    pub variable_mods: Vec<ResidueMod>,
    pub max_variable_mods: usize,
    pub charge_min: i32,
    pub charge_max: i32,
    /// Composition-based precursor charge range. When true, ignore
    /// `charge_min`/`charge_max` and emit every charge from 1 up to
    /// `1 (N-terminus) + (#R + #H + #K)`, the proton-carrying capacity of the
    /// peptide. Peptides therefore never receive a charge state they cannot
    /// physically hold, and each peptide's range depends on its own basic-residue
    /// count. Default false (fixed `charge_min..=charge_max` for every peptide).
    /// Pairs with `predict_frag.charge_by_basic_residues` for fragments. Changing
    /// the enumerated charge states changes the search/training/FDR population, so
    /// this remains benchmark-gated.
    pub charge_by_basic_residues: bool,
    /// `error` (default) or `skip` for unknown modifications.
    pub unknown_modification: UnknownModPolicy,
}
impl Default for PeptidoformsConfig {
    fn default() -> Self {
        Self {
            fixed_mods: vec![ResidueMod {
                residue: 'C',
                name: "Carbamidomethyl".to_string(),
            }],
            variable_mods: vec![ResidueMod {
                residue: 'M',
                name: "Oxidation".to_string(),
            }],
            max_variable_mods: 1,
            charge_min: 2,
            charge_max: 3,
            charge_by_basic_residues: false,
            unknown_modification: UnknownModPolicy::Error,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidueMod {
    /// Target residue; `*` for any / terminal handled separately in MVP.
    pub residue: char,
    /// UniMod name.
    pub name: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownModPolicy {
    #[default]
    Error,
    Skip,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PredictFragConfig {
    pub predictor: FragPredictorKind,
    pub rt_predictor: RtPredictorKind,
    /// Fragment charges rule: charge 1 always; charge 2 added for precursor
    /// charge >= this threshold (docs/18_findings_and_decisions.md). Default 2:
    /// DIA-NN uses doubly-charged fragments for ~16% of charge-2 precursors'
    /// transitions, so blocking them (the old default of 3) discarded real
    /// signal.
    pub charge2_from_precursor_charge: i32,
    /// Composition-based fragment charge cap. When true, a b/y fragment is kept
    /// at charge z only if `z <= 1 (its N-terminal amine) + (#R + #H + #K within
    /// that fragment)`, and never above the precursor charge. This supersedes the
    /// `charge2_from_precursor_charge` rule when set. Default false. Pairs with
    /// `peptidoforms.charge_by_basic_residues` for precursors; benchmark-gated
    /// because it changes the scored transition set.
    pub charge_by_basic_residues: bool,
    pub top_n_fragments: usize,
    pub ms2pip_model: String,
    /// Python executable for the MS2PIP sidecar (env with ms2pip + pyarrow).
    pub ms2pip_python: Option<String>,
    /// Python executable for the DeepLC sidecar (env with deeplc + pyarrow).
    pub deeplc_python: Option<String>,
    /// Directory holding the sidecar worker scripts.
    pub sidecar_script_dir: String,
}
impl Default for PredictFragConfig {
    fn default() -> Self {
        Self {
            predictor: t(),
            rt_predictor: t(),
            charge2_from_precursor_charge: 2,
            charge_by_basic_residues: false,
            top_n_fragments: 6,
            ms2pip_model: "HCD".to_string(),
            ms2pip_python: None,
            deeplc_python: None,
            sidecar_script_dir: "scripts".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SearchSeedConfig {
    pub fdr_seed: f64,
    pub fragment_tol_ppm: f64,
    /// Max reported PSMs per spectrum (wide-window DIA, docs/07_search_seed.md).
    pub report_psms: usize,
    /// Minimum matched fragments for a seed PSM.
    pub min_matched_peaks: usize,
    /// If > 0, probe only the `top_n_peaks` most intense peaks per MS2 scan
    /// (0 = all peaks). The seed only produces calibration anchors (RT/mass/IM),
    /// which come from abundant peptides, so this cuts the dominant per-peak index
    /// probing cost without discarding peaks from the downstream extraction
    /// artifact. Default 300; set to 0 to probe every converted peak.
    pub top_n_peaks: usize,
    /// Fragment-matcher backend (docs/06_predict_frag_index_matchers.md).
    /// Default `Fragindex`.
    pub matcher: MatcherKind,
    /// Robust two-pass fragment mass calibration (sensitivity_plan P3.1). After the
    /// first median-offset + tolerance fit, re-fit on only the deviations inside the
    /// first-pass tolerance window (rejecting outliers), giving a tighter, more
    /// robust offset + local uncertainty. Falls back to the single-pass result when
    /// too few in-window calibrants remain. Default false (single pass unchanged).
    pub two_pass_mass_cal: bool,
    /// m/z-dependent fragment mass calibration. When true, fit a LOESS of the
    /// calibrant ppm deviation versus fragment m/z and emit a sampled correction
    /// grid to `<seed>.masscal.json`; extract then applies an m/z-interpolated
    /// offset per peak instead of the single scalar `frag_ppm_offset`. This
    /// removes any m/z-correlated curvature the flat offset leaves. Default false
    /// (scalar offset unchanged), opt-in and benchmark-gated.
    pub mass_cal_loess: bool,
}
impl Default for SearchSeedConfig {
    fn default() -> Self {
        Self {
            fdr_seed: 0.01,
            fragment_tol_ppm: 20.0,
            report_psms: 5,
            min_matched_peaks: 4,
            top_n_peaks: 300,
            matcher: MatcherKind::Fragindex,
            two_pass_mass_cal: false,
            mass_cal_loess: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RtImTrainConfig {
    pub calibration_method: CalibrationMethod,
    pub q_train: f64,
    /// Percentile of |obs - calibrated_pred| residuals for the RT window.
    pub p_rt: f64,
    pub rt_window_multiplier: f64,
    pub min_seed_for_calibration: usize,
    /// LOESS span (fraction of points in each local fit).
    pub loess_span: f64,
    /// Fallback fixed RT window in seconds when calibration cannot be fit.
    pub fallback_rt_window_s: f64,
    /// Fine-tune the DeepLC multitask model on this run's confident seed PSMs
    /// and rewrite the library's `predicted_irt` before RT calibration. Requires
    /// `predict_frag.deeplc_python` (the DeepLC interpreter). Off by default; the
    /// main use is library-input mode, where the base iRT comes from the imported
    /// library rather than a DeepLC prediction.
    pub finetune_deeplc: bool,
    /// DeepLC fine-tune training epochs (passed to `deeplc_finetune.py --epochs`).
    /// Early stopping with `finetune_patience` usually halts before this cap, so it
    /// is an upper bound rather than a fixed count. Only used when `finetune_deeplc`.
    pub finetune_epochs: usize,
    /// DeepLC fine-tune early-stopping patience (`--patience`): epochs without
    /// validation-loss improvement before stopping. Only used when `finetune_deeplc`.
    pub finetune_patience: usize,
    /// DeepLC fine-tune batch size (`--batch`). 0 (default) auto-scales to the confident
    /// seed size so each epoch has >= ~30 gradient steps; a fixed large batch underfits
    /// small seeds (a ~4k-peptide reference at batch 512 is ~8 steps/epoch and never
    /// converges). Only used when `finetune_deeplc`.
    pub finetune_batch: usize,
    /// Adaptive RT window (sensitivity_plan spec 03 §3.5, backlog P3.2/P3.3):
    /// instead of one global residual-percentile half-width for every candidate,
    /// bin the calibration anchors by calibrated RT and give each candidate the
    /// LOCAL residual percentile of its RT region, clamped to
    /// `[rt_window_min_s, fallback_rt_window_s]` and scaled by
    /// `rt_window_multiplier`. A fixed window is simultaneously too wide for
    /// well-calibrated regions and too narrow for poorly-calibrated ones; this
    /// tightens clean regions (less interference) and widens noisy ones (more
    /// recall). Empty/sparse bins fall back to the global width. Default false.
    pub adaptive_rt_window: bool,
    /// Number of equal-width calibrated-RT bins for the adaptive window.
    pub adaptive_rt_bins: usize,
    /// Lower clamp (seconds) for any RT half-window (the existing 1 s floor).
    pub rt_window_min_s: f64,
    /// Size `w_rt` from HELD-OUT residuals instead of in-sample ones. A fraction of
    /// anchor peptides (`base_peptide_id % 1000 < round(frac*1000)`, so the split is
    /// deterministic and shared with `deeplc_finetune.py`) is excluded from the sizing
    /// fit and, when `finetune_deeplc` runs, from the fine-tune reference; `w_rt` is
    /// then the residual percentile of those held-out anchors against the fit they
    /// never entered. The final calibration curve still uses every anchor. In-sample
    /// sizing underestimates the tail and rewards a memorizing RT model with a window
    /// it does not deserve (measured: it inverted the 4.0.0a2/4.1.0 ranking); held-out
    /// sizing measured +0.9% peptides with DeepLC 4.1.0 and -1.5% with 4.0.0a2 on the
    /// AIF benchmark, both at 0.98% decoy, so enable it only with a generalizing RT
    /// model. 0.0 (default) keeps in-sample sizing. Mutually exclusive with
    /// `adaptive_rt_window`. Benchmark-gated; do not default on.
    pub window_holdout_frac: f64,
    /// Where an imported library's `predicted_irt` comes from. `auto` (the default)
    /// re-predicts every peptidoform with the DeepLC base model when
    /// `predict_frag.deeplc_python` is configured and keeps the imported values, with a
    /// warning, when it is not; `deeplc` requires the interpreter; `library` keeps the
    /// imported values. Ignored under `finetune_deeplc` (the fine-tune re-predicts every
    /// peptidoform itself) and in FASTA mode (predict-frag already produces DeepLC
    /// predictions). Measured on the AIF benchmark with calibration only and native_tda:
    /// 10,416 peptides at 1% from DeepLC 4.1.1 base predictions against 10,015 from the
    /// DIA-NN library iRT and 10,181 from a per-run fine-tune, with `w_rt` 343 s against
    /// 632 s and 472 s (docs/08 section 4c). `run-experiment` predicts once per experiment.
    pub library_irt: LibraryIrt,
}

/// Source of `predicted_irt` for an imported library; see `RtImTrainConfig::library_irt`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LibraryIrt {
    #[default]
    Auto,
    Library,
    Deeplc,
}

impl RtImTrainConfig {
    /// Whether an imported library's `predicted_irt` is re-predicted with the DeepLC base
    /// model before RT calibration. False in FASTA mode, under a fine-tune (which
    /// re-predicts by itself), under `library_irt = library`, and under `auto` without a
    /// DeepLC interpreter (the orchestrator warns in that case). `deeplc` without an
    /// interpreter is a preflight error, so it resolves to true here.
    pub fn repredicts_library_irt(&self, library_input: bool, has_deeplc: bool) -> bool {
        if !library_input || self.finetune_deeplc {
            return false;
        }
        match self.library_irt {
            LibraryIrt::Library => false,
            LibraryIrt::Deeplc => true,
            LibraryIrt::Auto => has_deeplc,
        }
    }
}
impl Default for RtImTrainConfig {
    fn default() -> Self {
        Self {
            calibration_method: t(),
            q_train: 0.01,
            p_rt: 0.95,
            rt_window_multiplier: 1.0,
            min_seed_for_calibration: 50,
            loess_span: 0.3,
            fallback_rt_window_s: 120.0,
            finetune_deeplc: false,
            finetune_epochs: 25,   // deeplc_finetune.py default
            finetune_patience: 10, // deeplc_finetune.py default
            finetune_batch: 0,     // 0 = auto-scale to seed size
            adaptive_rt_window: false,
            adaptive_rt_bins: 12,
            rt_window_min_s: 1.0,
            window_holdout_frac: 0.0,
            library_irt: LibraryIrt::Auto,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExtractConfig {
    /// Isolation windows probed per batch before the candidates no later window can touch
    /// are scored and written. `None` (the default) uses the rayon thread count capped at
    /// 16. The hit accumulator holds the windows in flight, so this sets the stage's peak
    /// almost linearly, while the accumulation phase it parallelises is a small part of the
    /// wall clock: on the HYE benchmark at 32 threads, 32 in flight is 24.65 GiB / 5:00, 16
    /// is 16.57 GiB / 5:04 and 8 is 12.31 GiB / 5:26, with identical output (docs/27 section
    /// 3.10). Set 8 or 4 on a memory-bound machine. Not a sensitivity knob.
    #[serde(default)]
    pub windows_in_flight: Option<usize>,
    pub fixed_scan_window: usize,
    pub frag_tol_ppm: f64,
    pub prec_tol_ppm: f64,
    /// tier-(b) minimum matched fragment count.
    pub presence_min_matched: usize,
    /// minimum distinct fragments for acceptance.
    pub presence_min_fragments: usize,
    /// minimum simultaneously-present fragments over the consecutive-scan run.
    pub presence_min_coelution: usize,
    /// tier-(d) spectral-agreement gate: reject a candidate whose observed fragment
    /// intensities agree with the predicted pattern below this score.
    ///
    /// Renamed from `min_frag_corr`, which was accurate for none of the four
    /// `gate_mode` values: under the default `apex_pearson` it is an intensity
    /// correlation at ONE apex scan rather than a chromatographic co-elution
    /// correlation, and under `spectral_entropy` it is not a correlation at all. The
    /// old name is not accepted (`deny_unknown_fields`), so an old config fails loudly
    /// with the offending key named rather than silently reverting to a default.
    /// Applied symmetrically to targets and decoys, but that alone does not prove
    /// null exchangeability in chimeric DIA; validate every threshold with an
    /// independent entrapment. 0 disables.
    pub gate_min_score: f64,
    /// tier-(c) minimum fraction of the candidate's predicted fragments that
    /// must be observed. With enough predicted fragments (top_n>=~10) this is a
    /// strong, symmetric discriminator: real peptides match a large fraction,
    /// chimeric false matches and decoys match a small fraction alike, so the
    /// target-decoy null stays valid.
    pub min_matched_fraction: f64,
    /// Shape-aware apex selection: choose the apex scan group by the summed
    /// observed intensity of only the top-K predicted (signature) fragments,
    /// rather than all matched fragments. In chimeric DIA a bright co-eluting
    /// interferent contributing to arbitrary channels wins a max-over-all-fragments
    /// apex; restricting to the peptide's strongest predicted ions locks onto its
    /// true elution instead. 0 selects the implementation default of the top 3
    /// predicted fragments.
    pub apex_top_fragments: usize,
    /// Optional Gaussian RT prior on apex selection: weight each scan group by
    /// exp(-0.5*((rt - rt_cal)/sigma)^2) with sigma = this value in seconds, so a
    /// distant interferent inside a wide RT window cannot define the apex. 0 = off.
    pub apex_rt_prior_s: f64,
    /// Fragment-count apex: pick the scan with the most distinct matched fragments,
    /// allowing scans within `apex_count_tol` of that maximum (so a slightly-lower-
    /// count but much more intense scan can still win), then the max summed-top-3
    /// intensity among them. Supersedes the summed-intensity apex when set.
    pub apex_count_tol: usize,
    /// Rolling-window width (in scan groups, centered, odd) for the distinct-
    /// fragment count that drives apex selection. Low-intensity fragments flicker
    /// in and out scan-to-scan; a single-scan count then spikes at noise scans and
    /// misplaces the apex. This sums the per-scan distinct-fragment count over a
    /// centered window so the apex lands in the region of *sustained* fragment
    /// presence, not an isolated flicker. A sum (not a mean) is used deliberately:
    /// edge truncation makes interior positions accumulate more, center-weighting
    /// the apex toward the RT-window centre (~= predicted RT) as a mild RT-prior;
    /// measured to beat a mean by ~+300 IDs on AIF. 1 = no smoothing (per-scan).
    pub apex_count_window: usize,
    /// Gaussian matched-filter smoothing of the per-scan fragment-count series
    /// before apex selection, as a sigma in scan units. 0.0 (default) keeps the
    /// `apex_count_window` rolling-sum smoother unchanged. When > 0, the count
    /// series is convolved with a Gaussian kernel (radius = 3*sigma) instead,
    /// which localizes the apex more robustly than a uniform window against
    /// scan-to-scan flicker. Opt-in and benchmark-gated: it changes apex
    /// selection and therefore identifications.
    pub apex_gaussian_sigma_scans: f64,
    /// Emit per-fragment chromatograms on the FULL isolation-window scan grid with
    /// 0.0 where a fragment is absent (aggregating scans of the same isolation
    /// window), so the elution profile drops to zero between peaks and the
    /// features-stage boundary calling is not misled by interpolated gaps.
    pub emit_window_grid: bool,
    /// m/z bucket size (power of two).
    pub bucket_size: usize,
    /// How a shared observed peak's intensity is apportioned among co-isolated,
    /// co-eluting candidates that all match it (see [`PeakClaim`]).
    pub peak_claim: PeakClaim,
    /// Composable claim-weight cues for `PeakClaim::CoelutionMultiCue` (modular
    /// fragment-competition framework). All default off (weight 1.0).
    pub claim_cues: ClaimCues,
    /// Spectrum-centric NNLS demixing (D2, fragment-competition report). When true,
    /// at each accepted candidate's apex scan, assemble the co-isolated candidate x
    /// fragment design matrix, solve non-negative least squares (deterministic
    /// ridge-regularized), and emit non-destructive demix features (deconv_explained_frac,
    /// deconv_active, deconv_share) so the rescorer sees each candidate's
    /// interference-corrected abundance. Default false; changes no extracted intensity.
    pub emit_demix_features: bool,
    /// Ridge for the demix NNLS passive solve (keeps it PD/deterministic under the
    /// ~98% wide-window column collinearity). Default 1.0.
    pub demix_lambda: f64,
    /// Cap on the number of co-isolated candidates (design-matrix columns) in a single
    /// demix solve, to bound compute on crowded windows. Default 64.
    pub demix_max_candidates: usize,
    /// Scan stride for the DESTRUCTIVE `CoelutionDemix` redistribution: solve the
    /// per-scan NNLS every Nth scan and reuse the resulting candidate abundances to
    /// apportion the intervening scans (a re-solve is forced whenever a new candidate
    /// enters the co-isolated set, so accuracy is preserved where the population
    /// changes). This is the practicality lever - a full per-scan solve over the ~465k
    /// scans of a wide-window run is impractical. 1 (default) solves at every scan.
    /// Only affects `CoelutionDemix`; the non-destructive demix FEATURES are unaffected.
    pub demix_scan_stride: usize,
    /// Emit a non-destructive `contested_frac` per PSM: the fraction of a
    /// candidate's matched intensity that a co-eluting competitor claims more
    /// strongly (by the two-pass elution-profile arbitration). Does not alter the
    /// extracted intensities; feeds a rescorer feature. Forces the two-pass path.
    pub emit_contested_features: bool,
    /// Dominance factor for `CoelutionWinnerMargin`: a shared peak is claimed
    /// winner-take-all only if the top eluter's profile height is at least this
    /// multiple of the runner-up's; otherwise the peak stays shared.
    pub peak_claim_margin: f64,
    /// Fragment-matcher backend (docs/06_predict_frag_index_matchers.md).
    /// Default `Fragindex`.
    pub matcher: MatcherKind,
    /// Minimum-PSMs-per-peptide evidence filter: reject a candidate whose fragments
    /// co-elute over fewer than this many consecutive scan groups (`coelution_run`).
    /// A single/double-scan spike is a transient (likely-interferent) match; a real
    /// peptide persists across its elution. 0 disables (the `scan_window` floor still
    /// applies). This is the DIA analog of a "seen in >= N PSMs" requirement.
    pub min_coelution_run: usize,
    /// Rescue a candidate that fails the single-scan fragment-Pearson gate when it
    /// has adequate matched fragments AND MS1 isotope-pattern support (mono + a
    /// plausible +1/mono ratio). Off by default: it relaxes acceptance, so enable
    /// it only with target-decoy/entrapment FDR validation. MS1 evidence is now
    /// computed before the gate so this can take effect.
    pub ms1_rescue: bool,
    /// Number of chromatographic peak hypotheses to enumerate per candidate.
    /// `K>1` writes up to K local maxima to the diagnostic
    /// `<out-psms>.peaks.parquet` sidecar. The primary PSM still contains only the
    /// selected apex, so these extra hypotheses are not currently rescored or used
    /// to improve identifications. K=1 preserves the single-apex behaviour.
    pub retain_top_peaks: usize,
    /// Number of chromatographic peaks PROMOTED to real feature/rescore rows per
    /// candidate (AlphaDIA plan #7, top-K). `1` (default) emits only the selected
    /// apex, so the pipeline is byte-identical. `>1` additionally emits the next
    /// strongest non-overlapping `enumerate_peaks` groups (each a full re-sliced PSM
    /// record carrying `peak_rank`), so the rescorer can pick the correct-but-not-apex
    /// peak; the selected apex stays `peak_rank = 0`. Must be `<= retain_top_peaks`.
    /// Behaviour-changing and benchmark/entrapment-gated: it changes the extracted
    /// row population, and compete/rescore must collapse per candidate so the decoy
    /// null is not K-inflated.
    pub promote_top_peaks: usize,
    /// Minimum integrated area of a promoted alternate peak (rank >= 1) as a fraction
    /// of the rank-0 peak's area. Suppresses noise-level alternates. Only used when
    /// `promote_top_peaks > 1`.
    pub alt_peak_min_area_frac: f64,
    /// Minimum apex-RT separation (seconds) between a promoted alternate peak and the
    /// rank-0 apex, so a near-duplicate of the selected peak is not re-emitted. Only
    /// used when `promote_top_peaks > 1`.
    pub alt_peak_min_separation_s: f64,
    /// Diagnostic candidate-audit: when true, extraction records, for every probed
    /// candidate, either the survivor stage-flags or the earliest `RejectionReason`,
    /// and writes `<out-psms>.audit.parquet` (spec 01 §4 / P0.3). Near-zero cost
    /// when false (no per-candidate audit allocation). Default false (production).
    pub emit_candidate_audit: bool,
    /// Evidence-count apex selection: choose the apex scan by the NUMBER of distinct
    /// co-eluting predicted fragments present (breadth of evidence), using observed
    /// signature-ion intensity only as a sub-integer tiebreak. In wide-window DIA a
    /// single fragment m/z channel is chimeric, so the tallest scan is often a
    /// co-isolated interferent; the scan where the most of the peptide's own
    /// predicted transitions co-elute is a more reliable apex. Default `true`, on
    /// correctness grounds rather than a count: `false` keeps the legacy
    /// signature-intensity apex, whose score is 0.0 at every qualifying scan when none
    /// of the top-K predicted fragments is observed, so the strict `>` never replaces
    /// the first candidate and the apex silently becomes the LOWEST-RT qualifying scan.
    /// The rolling distinct-fragment count (`apex_count_window`) still gates which scans
    /// qualify in both modes.
    pub apex_evidence_rank: bool,
    /// Emit the four gate-diagnostic scores (`gate_apex`, `gate_peak_spectral`,
    /// `gate_coelution`, `gate_spectral_entropy`) as extra `psms.parquet` columns,
    /// for the offline gate-metric comparison. Default `false` (diagnostic sidecar,
    /// like `emit_candidate_audit`): when off, neither the columns nor the extra
    /// per-candidate score computation happen, so the default chain is byte-identical.
    pub emit_gate_diagnostics: bool,
    /// Which spectral-agreement score the `gate_min_score` gate thresholds
    /// (sensitivity program). The legacy gate uses a single apex-scan intensity
    /// Pearson, which one chimeric scan can dominate. See [`GateMode`].
    pub gate_mode: GateMode,
    /// Second threshold for `GateMode::Combined`: the co-elution score must exceed
    /// this while the peak-integrated spectral score exceeds `gate_min_score`.
    /// Requiring BOTH is more specific (rejects interferents that pass one axis).
    pub gate_coelution_min: f64,
}
impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            windows_in_flight: None,
            fixed_scan_window: 3,
            frag_tol_ppm: 20.0,
            prec_tol_ppm: 20.0,
            presence_min_matched: 3,
            presence_min_fragments: 3,
            presence_min_coelution: 2,
            // Loosening raises recall and candidate volume; external entrapment
            // validation is required before treating any threshold as FDR-safe.
            // Relaxed from the historical 0.5 to 0.2 to recover low-abundance
            // candidates the hard single-scan Pearson gate was dropping
            // (docs/18_findings_and_decisions.md); still a hard gate, not the
            // soft/budgeted redesign.
            // 0.2. Briefly set to 0.6 on the strength of the gate sweep in docs/18,
            // where `native_tda` peaked at 0.6 (9,503 peptides) while `nn_torch` peaked
            // at the loosest gate tested. Since `native_tda` is the default classifier,
            // 0.6 looked like the matching default.
            //
            // Measured, and it is not. Same AIF file, current defaults (augmented
            // library, `apex_evidence_rank = true`, paired CV folds), 2x2 of gate x
            // rescorer at an unchanged empirical decoy fraction of 0.0097-0.0098:
            //
            //     gate 0.2   native_tda 10,847   nn_torch 10,914
            //     gate 0.6   native_tda 10,369   nn_torch 10,399
            //
            // 0.6 costs 4.4% of peptides for `native_tda` and 4.7% for `nn_torch`, and
            // halves what extract accepts (45,338 -> 21,979). `native_tda`'s inverted-U
            // flattened from below -- its count rose from 9,503 to 10,847 and the optimum
            // moved to the loose end -- so the sweep that motivated 0.6 no longer
            // describes this configuration, and loose is now better for BOTH rescorers.
            gate_min_score: 0.2,
            min_matched_fraction: 0.0,
            apex_top_fragments: 0, // superseded by apex_count_tol; kept for compat
            apex_rt_prior_s: 0.0,  // RT prior off by default
            apex_count_tol: 1,     // fragment-count apex with 1-fragment slack
            apex_gaussian_sigma_scans: 0.0, // Gaussian apex smoother off by default (opt-in)
            apex_count_window: 1,  // no rolling smoothing by default (opt-in; window 5
            // cuts AIF apex misassignment, median |dRT| 131s->9s)
            emit_window_grid: true, // zero-filled window-grid chromatograms
            bucket_size: 8192,
            peak_claim: PeakClaim::None,
            claim_cues: ClaimCues::default(),
            emit_demix_features: false,
            demix_lambda: 1.0,
            demix_max_candidates: 64,
            demix_scan_stride: 1,
            emit_contested_features: false,
            peak_claim_margin: 2.0,
            matcher: MatcherKind::Fragindex,
            min_coelution_run: 0, // disabled; scan_window floor still applies
            ms1_rescue: false,    // opt-in; relaxes acceptance, validate FDR first
            retain_top_peaks: 1,  // legacy single-apex behaviour (K=1)
            promote_top_peaks: 1, // top-K promotion off (only the selected apex is a row)
            alt_peak_min_area_frac: 0.10, // alternate peak >= 10% of rank-0 area
            alt_peak_min_separation_s: 5.0, // alternate apex >= 5 s from rank-0 apex
            emit_candidate_audit: false, // diagnostic; off in production
            // On. The legacy signature-intensity apex scores a scan group by the summed
            // OBSERVED intensity of only the top-K PREDICTED fragments, so when none of
            // those K is observed at any qualifying scan the score is 0.0 everywhere, the
            // strict `>` never replaces the first candidate, and the apex silently becomes
            // the LOWEST-RT qualifying scan -- up to a full RT window away, or anywhere in
            // the gradient for a candidate with no window row. The RT prior cannot rescue
            // it, because the combination is multiplicative and a zero annihilates the
            // prior in exactly the case the prior exists for. Evidence rank scores
            // `(n_distinct_fragments + tie) * prior`, which is always positive, so the
            // fallback is unreachable. The wrong apex propagates into `prelim_score`
            // (which decides the pre-FDR competition winner), `rt_error_abs`,
            // `log_apex_intensity`, and quant's integration centre.
            apex_evidence_rank: true,
            emit_gate_diagnostics: false, // diagnostic gate-score columns; off in production
            gate_mode: GateMode::ApexPearson, // legacy single-scan intensity Pearson
            gate_coelution_min: 0.5,      // used only by GateMode::Combined
        }
    }
}

/// Spectral-agreement score the extraction acceptance gate (`gate_min_score`)
/// thresholds. All are computed at the gate from data already in hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GateMode {
    /// Legacy: Pearson of observed-vs-predicted fragment intensities at the single
    /// apex scan. One chimeric scan can dominate it.
    #[default]
    ApexPearson,
    /// Pearson of the PEAK-INTEGRATED observed spectrum (each fragment summed over
    /// the elution-peak scans) vs predicted intensities. Averages out a single
    /// interfered scan; the standard library-dot-product measure.
    PeakSpectral,
    /// Li spectral-entropy similarity of the sqrt-transformed apex-scan observed vs
    /// predicted intensities (`spectral_entropy_similarity_sqrt`). The full-feature
    /// gate search (all ~379 features, target-vs-decoy) found this the single best
    /// gate discriminator: AUC 0.826 / matched-pool recall 69.8%, versus apex
    /// Pearson's 0.781 / 64.5%. Same inputs as `ApexPearson`, better separation.
    SpectralEntropy,
    /// Predicted-intensity-weighted mean CO-ELUTION correlation of each matched
    /// fragment's XIC to the signature reference over the elution peak (temporal
    /// agreement, orthogonal to intensity agreement).
    Coelution,
    /// Require BOTH: peak-integrated spectral Pearson >= `gate_min_score` AND the
    /// co-elution score >= `gate_coelution_min`. More specific (an interferent
    /// passing one axis is still rejected), for a cleaner FDR pool.
    Combined,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FeaturesConfig {
    pub set: FeatureSet,
    /// Write the Percolator-style `.pin` text file requested by `--out-pin`. No MuMDIA
    /// stage consumes it (`rescore` builds its own PIN for the sidecars); it exists for
    /// external tooling. At 1.5M rows x 387 features it is a ~5.4 GB text write.
    /// Default false: nothing in MuMDIA reads it, which makes the write pure cost unless
    /// an external tool wants the file. Set true to get the artifact back.
    pub emit_pin: bool,
    pub coelution_corr_threshold: f64,
    pub prec_tol_ppm: f64,
    /// Restrict trace-based features (co-elution, profile, xcorr, interference,
    /// base width) to the elution peak around the apex rather than the whole
    /// extracted RT window, so they are not diluted over large RT stretches.
    pub bound_features: bool,
    /// Peak-boundary threshold as a fraction of apex height (DIA-NN-style: descend
    /// to peak*fraction, or stop earlier at a valley below it). 1/3 matched DIA-NN's
    /// RT bounds best in the diagnostic-plot benchmark.
    pub bound_peak_fraction: f64,
    /// Grace when walking the elution-peak boundary: number of consecutive
    /// sub-threshold scans to BRIDGE before stopping. 0 (default) stops at the first
    /// scan below `bound_peak_fraction` (brittle on jagged/gappy profiles); 1 bridges
    /// a single-scan dip (DIA sampling gap / noise), giving steadier boundaries.
    pub bound_peak_grace: usize,
    /// Elution-peak boundary source. When true (default) a single set of left/right
    /// half-widths (seconds) is learned once from the confident seed PSMs
    /// (`spectrum_q <= 0.01`, target-only, the same set that anchors RT calibration /
    /// DeepLC fine-tune) and applied to EVERY candidate around its own apex. This
    /// removes per-candidate boundary manipulation so a decoy is scored over a real-
    /// peptide-width window centred on its apex. When false, each candidate detects its
    /// own peak boundary from its top-3-predicted-fragment profile (per-candidate,
    /// but noisy/manipulable for chimeric decoys; the legacy behaviour). If the seed
    /// yields < 20 confident anchors the stage logs a warning and falls back to
    /// per-candidate detection for that run.
    pub bound_from_confident: bool,
    /// Percentile (0-100) of the confident-set half-widths taken as the global left/
    /// right elution half-width when `bound_from_confident` is true. 50 = median
    /// (typical real peak width); higher percentiles widen the shared window.
    pub bound_confident_pct: f64,
    /// Emit the MS1 apex-isotope precursor feature `ms1_isotope_height_corr`
    /// (Pearson of the observed apex isotope heights `[i0,i1,i2]` against the
    /// Poisson-averagine model). Default false (the feature is present in the
    /// battery but returns 0.0, so the vector length is unchanged in effect). It
    /// overlaps the existing `ms1_isotope_cosine_apex`, so it is opt-in and
    /// benchmark-gated rather than default-on (AlphaDIA-plan item 12).
    pub ms1_precursor_features: bool,
}
impl Default for FeaturesConfig {
    fn default() -> Self {
        Self {
            set: t(),
            // Off. No MuMDIA stage reads this file: `rescore` builds its own PIN
            // (features.rs, "the PIN is not consumed by any stage"). It is a ~5.4 GB text
            // write per run on a real library, so a 40-run experiment wrote hundreds of GB
            // that nothing read back, by default. Set true when a PIN is wanted for an
            // external tool.
            emit_pin: false,
            coelution_corr_threshold: 0.9,
            prec_tol_ppm: 20.0,
            bound_features: true,
            bound_peak_fraction: 1.0 / 3.0,
            bound_peak_grace: 0, // stop at first sub-threshold scan (legacy)
            bound_from_confident: true, // fixed feature window from confident-seed norm
            bound_confident_pct: 50.0, // median confident half-width
            ms1_precursor_features: false, // opt-in; overlaps ms1_isotope_cosine_apex
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CompeteConfig {
    /// Competition grouping: `precursor` collapses charge/modification siblings
    /// separately within each target/decoy label; targets and decoys therefore do
    /// not compete directly. `apex` also groups by rounded apex RT;
    /// `peptidoform_charge` keeps each peptidoform+charge as its own group
    /// (precursor-level, as DIA-NN/Spectronaut report), so sibling charges of one
    /// peptide are not collapsed.
    pub group_by: CompeteGroupBy,
    pub apex_rt_tolerance_s: f64,
    /// How within-group competition resolves (sensitivity program, spec 04 §6 /
    /// P2.4). `winner_take_all` = legacy (keep only the top `prelim_score` per
    /// group). The other modes preserve more candidate evidence for the rescorer/
    /// FDR to arbitrate. Default `winner_take_all` (unchanged behaviour).
    pub mode: CompetitionMode,
    /// Score margin (in `prelim_score` units) required to remove a loser under
    /// `margin_gated`. A loser closer than this to the winner is kept.
    pub margin: f64,
    /// Minimum distinct unique-fragment count a loser must have to survive under
    /// `unique_evidence` (needs the `unique_fragment_count` feature; falls back to
    /// winner-take-all when the column is absent).
    pub unique_evidence_min_fragments: usize,
    /// Diagnostic: when true, write `<out>.compete_audit.parquet` recording every
    /// removed candidate with its group, winner, scores, and removal reason.
    pub emit_competition_audit: bool,
}
impl Default for CompeteConfig {
    fn default() -> Self {
        Self {
            group_by: CompeteGroupBy::BasePeptide,
            apex_rt_tolerance_s: 5.0,
            mode: CompetitionMode::WinnerTakeAll,
            margin: 0.0,
            unique_evidence_min_fragments: 2,
            emit_competition_audit: false,
        }
    }
}

/// Within-group competition resolution (spec 04 §6). Only `WinnerTakeAll` removes
/// candidates unconditionally; the others preserve candidates the rescorer can
/// still discriminate, which is the sensitivity program's central principle
/// ("preserve candidate evidence until the workflow can make a calibrated
/// decision"). Target/decoy labels remain part of the competition key in every
/// mode, so a target never competes against its own decoy (the null is preserved).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompetitionMode {
    /// Legacy: keep only the highest `prelim_score` candidate per group.
    #[default]
    WinnerTakeAll,
    /// Keep every candidate (no within-group removal); FDR handles ambiguity.
    None,
    /// Keep every candidate; conflict/contested features (added upstream) carry the
    /// interference signal into rescoring. Same retained set as `None`; the name
    /// documents intent for the experiment matrix.
    FeaturesOnly,
    /// Keep a loser when it has enough independent evidence
    /// (`unique_fragment_count >= unique_evidence_min_fragments`); otherwise remove
    /// it (winner-take-all fallback).
    UniqueEvidence,
    /// Remove a loser only when `winner_score - loser_score >= margin`; otherwise
    /// keep it. Conservative removal for the low-FDR region.
    MarginGated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompeteGroupBy {
    /// One winner per stripped base peptide, per label. Renamed from `precursor`,
    /// which it is not: `compete.rs` keys the group on `base_peptide_id`, which comes
    /// from the stripped sequence, so every charge state AND every modification
    /// variant of one peptide collapses to a single winner before FDR. Use
    /// `peptidoform_charge` for a genuine precursor unit, and note that it is
    /// REQUIRED for a PTM search. The old name is not accepted, so an old config
    /// fails loudly rather than silently changing the competition unit.
    BasePeptide,
    Apex,
    /// Precursor-level: separate every distinct peptidoform+charge. Recovers
    /// sibling charges the peptide-level `Precursor` grouping collapses; the
    /// label stays in the key so a target never competes against its own decoy.
    PeptidoformCharge,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RollupMethod {
    /// Sum of the top-N most abundant peptides (single-run default).
    #[default]
    TopNSum,
    /// Sum of all group peptides.
    Sum,
}

/// How the elution-peak integration window is chosen per candidate in quant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PeakWindowMode {
    /// Each candidate's window comes from its own summed-XIC descent walk. Exact
    /// per peak but sensitive to interference (stretched) and sparse peaks
    /// (collapsed).
    #[default]
    PerCandidate,
    /// Consensus window: the median left and right half-widths of confident
    /// peptides (a near-constant instrument/gradient property) applied around
    /// each candidate's apex. Robust to a single window being distorted. The
    /// widths are estimated per quant invocation, not shared automatically
    /// across runs.
    Consensus,
}

/// Cross-run normalization applied to the feature-by-run matrix in the LFQ combine
/// step (`quant-lfq`) before protein rollup. A single global size factor per run
/// corrects the global signal-level difference between runs (e.g. total ion
/// current), so unchanged peptides center on a log-ratio of 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NormalizeMethod {
    /// No normalization: raw per-run areas.
    None,
    /// Median-of-ratios (DESeq-style): per-run size factor = median over
    /// complete-case features (present in every run) of the ratio to a
    /// geometric-mean pseudo-reference. Robust to a minority of changing
    /// features, so it does not flatten a spike-in design's real fold changes.
    #[default]
    MedianRatio,
    /// Global median: per-run size factor aligns each run's median intensity to
    /// the common (median-of-per-run-medians) target. Simpler, less robust to
    /// composition shifts than median-of-ratios.
    Median,
}

impl NormalizeMethod {
    /// Parse a CLI/token spelling (`none`, `median_ratio`/`median-ratio`, `median`).
    pub fn from_token(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "none" => Some(Self::None),
            "median_ratio" => Some(Self::MedianRatio),
            "median" => Some(Self::Median),
            _ => None,
        }
    }
}

/// Which q-value column quant filters candidates on. Peptide- or precursor-level
/// q is appropriate for a single-run rescore. Under experiment-wide rescoring,
/// those grouped q-values are pooled and carried only on the best PSM across all
/// runs, so filtering per-run slices on them creates disjoint quant sets.
/// `RunPsmQ` is the run-local FDR gate for that cross-run workflow; `PsmQ` keeps
/// the pooled per-PSM gate available when that is explicitly intended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum QuantQColumn {
    /// Filter on `peptide_q_value` (per-run peptide FDR). Default.
    #[default]
    PeptideQ,
    /// Filter on `precursor_q`. This is valid only for a single-run rescore:
    /// experiment-wide rescoring currently computes precursor q-values over the
    /// pooled experiment and assigns each precursor's grouped q-value to its
    /// best PSM, so it is not a per-run cross-run-quant gate.
    PrecursorQ,
    /// Filter on the per-PSM `q_value`. In an experiment-wide rescore this is a
    /// pooled-experiment PSM q-value, not a run-local FDR estimate.
    PsmQ,
    /// Filter on `run_psm_q` (per-run PSM FDR). The correct choice for cross-run
    /// quant off an experiment-wide rescore: each run's PSMs are FDR-controlled
    /// within their own run, so quant keeps the right per-run precursors without
    /// the external `split_scored.py` peptide-q overwrite.
    RunPsmQ,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QuantConfig {
    /// Peptide-level q-value cutoff for inclusion.
    pub q_threshold: f64,
    /// Number of top fragments summed per peptidoform.
    pub top_n_fragments: usize,
    /// Number of top peptides summed per protein group (TopNSum).
    pub top_n_peptides: usize,
    pub rollup: RollupMethod,
    /// Integrate each fragment only over the detected elution-peak window rather
    /// than the whole chromatogram. The window is found from the summed XIC apex.
    pub bound_peak: bool,
    /// Descent threshold for the peak-window walk: stop where the summed XIC drops
    /// below `peak_fraction` * apex height (1/6 expanded from the 1/3 feature bound).
    pub peak_fraction: f64,
    /// Zig-zag grace: bridge up to this many consecutive sub-threshold scans during
    /// the peak-window walk; the boundary triggers on `peak_grace + 1` consecutive
    /// sub-threshold scans (1 = stop on 2 consecutive misses).
    pub peak_grace: usize,
    /// Per-candidate window vs a consensus width derived from confident peptides.
    pub peak_window_mode: PeakWindowMode,
    /// Peptide q-value cutoff defining the "confident" set that calibrates the
    /// consensus half-widths (Consensus mode only). Tighter than `q_threshold`.
    pub reliable_q: f64,
    /// Which q-value column to filter candidates on (`peptide_q` default;
    /// `precursor_q` is single-run only; use `run_psm_q` for per-run slices of an
    /// experiment-wide rescore). See [`QuantQColumn`].
    pub q_filter: QuantQColumn,
    /// Apply an apex-outward interference-correction envelope to each fragment
    /// trace before integrating its area, stripping co-eluting interference in the
    /// peak wings. Off by default (identity on a clean peak). Opt-in and
    /// benchmark-gated: it changes reported quantities.
    pub interference_envelope: bool,
    /// Which fragments enter the top-N sum. `observed_area` (default, legacy) ranks
    /// by the integrated area itself, which preferentially selects interfered
    /// fragments (their areas are inflated) and so varies run to run.
    /// `predicted` ranks by the library (predicted or empirical) fragment intensity,
    /// a per-precursor constant, so every run sums the same fragments. Astral HYE
    /// 2026-08-26: CV 0.163 -> 0.112 on 6/6 ions at top-3. Benchmark-gated.
    pub fragment_selection: FragmentSelection,
    /// When > 0, integrate each fragment over the `2k+1` scans centred on the
    /// identification apex instead of the descent-walk window (`bound_peak`
    /// window ignored; falls back to it when the apex is unknown). A fixed narrow
    /// window is far less sensitive to interference in the peak wings than the
    /// walked bounds. 0 (default) = off.
    pub fixed_scan_halfwidth: usize,
    /// Subtract a per-fragment local background before integrating (fixed-scan
    /// window only). The background is the `baseline_quantile` quantile of the
    /// intensities in the two flanks (`baseline_flank_scans` samples on each side
    /// of the integration window); window intensities are clipped at zero after
    /// subtraction. Targets the additive floor that compresses ratios in the
    /// low-abundance condition. Off by default; benchmark-gated.
    pub baseline_subtract: bool,
    /// Flank length (samples per side) used to estimate the background.
    pub baseline_flank_scans: usize,
    /// Quantile of the flank intensities taken as the background level.
    pub baseline_quantile: f64,
    /// When > 0, integrate each fragment over the samples within `fixed_window_s`
    /// seconds of the identification apex (instrument-independent alternative to
    /// `fixed_scan_halfwidth`, which it overrides). 0 (default) = off.
    pub fixed_window_s: f64,
}

/// Fragment ranking for the quant top-N sum. See [`QuantConfig::fragment_selection`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FragmentSelection {
    /// Rank fragments by their own integrated area (legacy).
    #[default]
    ObservedArea,
    /// Rank fragments by library intensity (`predicted_intensity` in the chromatogram table).
    Predicted,
}
impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            q_threshold: 0.01,
            top_n_fragments: 3,
            top_n_peptides: 3,
            rollup: RollupMethod::TopNSum,
            bound_peak: true,
            peak_fraction: 1.0 / 6.0,
            peak_grace: 1,
            peak_window_mode: PeakWindowMode::PerCandidate,
            reliable_q: 0.001,
            q_filter: QuantQColumn::PeptideQ,
            interference_envelope: false, // apex-outward interference envelope off by default
            fragment_selection: FragmentSelection::ObservedArea,
            fixed_scan_halfwidth: 0,
            baseline_subtract: false,
            baseline_flank_scans: 12,
            baseline_quantile: 0.25,
            fixed_window_s: 0.0,
        }
    }
}

/// Match-between-runs strategy (Stage D3, docs/12_quant_lfq_align_mbr_report_audit.md).
/// Default `None` reproduces the current chain byte-for-byte.
///
/// ONLY `None` VS NOT-`None` IS IMPLEMENTED. The three non-`None` variants are described
/// below as the intended staging, but no code distinguishes them: every test in the tree is
/// `strategy != None`, so selecting `RtTransfer` or `Full` today behaves exactly like
/// `EmpiricalLibrary`. They are kept as the recorded design ladder rather than deleted
/// because the MBR tier is planned and benchmark-gated (CLAUDE.md); `validate()` warns when
/// a non-`None` variant is selected so a config cannot quietly expect more than it gets.
///
/// Intended staging: `EmpiricalLibrary` builds the consensus anchor library only;
/// `RtTransfer` adds cross-run expected-RT transfer extraction; `Full` adds
/// requantification. All require >= 2 runs and a decoy-transfer FDR (see the plan).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MbrStrategy {
    /// No match-between-runs (default).
    #[default]
    None,
    /// Build the cross-run consensus anchor library (M1) only; no transfer.
    EmpiricalLibrary,
    /// EmpiricalLibrary + cross-run expected-RT transfer extraction (M2/M3).
    RtTransfer,
    /// RtTransfer + requantification of accepted transfers (M5).
    Full,
}

/// Decoy-transfer null for the MBR false-transfer FDR (M4). `ReverseSequence`
/// transfers reverse/scramble decoys at the same expected RT; `PermutedRt` transfers
/// real precursors to a decoupled (wrong) expected RT; `Both` combines them. The
/// prototype's shuffled-RT null gave a ~0.6% in-window false rate vs 66.6% true
/// (113x separation), so the transfer q-value is well-calibrated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DecoyTransfer {
    #[default]
    PermutedRt,
    ReverseSequence,
    Both,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MbrConfig {
    pub strategy: MbrStrategy,
    /// q-value for a precursor to become a cross-run anchor (validated at 0.01).
    pub q_anchor: f64,
    /// Minimum number of OTHER runs a precursor must be confident in to transfer.
    pub min_anchor_runs: usize,
    /// Accept threshold for a transferred identification's transfer q-value.
    pub q_transfer: f64,
    /// NOT YET WIRED. Transfer RT half-window (seconds) around the cross-run-predicted
    /// RT. The M2 leave-target-out residual was ~17 s at p95, ~15x tighter than the search
    /// window, which is where this default comes from -- but no code reads this field yet,
    /// so setting it has no effect. Kept as the recorded design value for the MBR transfer
    /// tier; `validate()` warns if it is changed from the default. See CLAUDE.md, "MBR
    /// transfer/re-extraction remains benchmark-gated".
    pub rt_window_s: f64,
    /// NOT YET WIRED. Which decoy-transfer null would estimate the false-transfer rate
    /// (M4). No code reads this field yet; `validate()` warns if it is changed.
    pub decoy_transfer: DecoyTransfer,
    /// Minimum correlation of the observed fragment pattern to the empirical
    /// consensus for a transfer to be accepted (interference guard; 0 disables).
    pub consensus_corr_min: f64,
    /// NOT YET WIRED. Would requantify already-identified precursors too (fill the
    /// matrix), not only transferred ones, under `strategy = Full`. No code reads this
    /// field yet; `validate()` warns if it is changed.
    pub requant_all: bool,
    /// Python interpreter for the `mbr_worker.py` sidecar (pandas/pyarrow/numpy;
    /// e.g. the `py312_mumdia` env). Required when `strategy != None`.
    pub python: Option<String>,
}
impl Default for MbrConfig {
    fn default() -> Self {
        Self {
            strategy: MbrStrategy::None,
            q_anchor: 0.01,
            min_anchor_runs: 2,
            q_transfer: 0.01,
            rt_window_s: 20.0, // >= the p95 M2 residual (~17 s)
            decoy_transfer: DecoyTransfer::PermutedRt,
            consensus_corr_min: 0.0,
            requant_all: false,
            python: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RescoreConfig {
    pub classifier: RescorerKind,
    pub folds: usize,
    pub train_fdr: f64,
    /// number of semi-supervised iterations for the native rescorer.
    pub num_iter: usize,
    /// Refuse a rescore whose in-memory feature matrix would exceed this many GiB.
    /// 0 (default) means no ceiling, which is the previous behaviour.
    ///
    /// The matrix is `Vec<Vec<f64>>`: eight bytes per value, plus a heap allocation and a
    /// 24-byte spine entry per PSM. Nothing in the workspace estimates or checks available
    /// memory (there is deliberately no `sysinfo` dependency), so an experiment-wide
    /// rescore over enough runs was simply killed by the OS after however long it took to
    /// get there. `native_tda` additionally runs all folds in parallel, each holding an
    /// owned standardised copy of its training slice, so the true peak is roughly
    /// `(1 + folds) x` this figure.
    ///
    /// Setting a ceiling converts that into an error at startup, naming the estimate and
    /// the two ways out. It is not a batching implementation: sub-batching changes which
    /// PSMs share a pooled `q_value`, so it is the operator's decision, not a silent one.
    pub max_feature_matrix_gib: f64,
    pub python: Option<String>,
    /// Path to an external `percolator` executable.
    ///
    /// Parsed and never read: no stage launches percolator, and `RescorerKind` has no
    /// variant that would. It is the only silently inert config field in the tree, since
    /// the three MBR ones warn (see `validate`). Kept rather than deleted because the
    /// external-percolator path is still intended; `validate` now warns when it is set.
    pub percolator_bin: Option<String>,
    /// Protein-accession substring marking spike-in (entrapment) negatives, e.g.
    /// "_HUMAN". Required when `classifier = entrapment`; PSMs whose protein
    /// contains it are the empirical false population.
    pub entrapment_marker: Option<String>,
    /// If a protein also contains this substring it is NOT counted as
    /// entrapment (the sample's own species, e.g. "_ECOLI"): shared peptides
    /// then count as real targets. `None` = the marker alone decides.
    pub entrapment_exclude: Option<String>,
    /// Protein substrings marking genuine contaminants inside the spike-in
    /// proteome (e.g. "KRT", "ALBU", keratin/albumin entry-name tokens). A PSM
    /// matching `entrapment_marker` but also one of these is treated as a REAL
    /// target, not an entrapment negative: such peptides are truly present
    /// (handling contaminants) so using them as negatives mislabels real signal
    /// and inflates the estimated FDR. Empty = every spike-in hit is a negative.
    pub entrapment_contaminant_markers: Vec<String>,
    /// N_real_lib / N_entrap_lib. Scales the entrapment FDR estimate so it is
    /// unbiased when the spike-in library differs in size from the real one.
    pub entrapment_ratio: f64,
    /// When true, any sidecar/classifier failure or misconfiguration (Mokapot or
    /// entrapment sidecar error, unwired percolator, entrapment mode with no
    /// entrapment PSMs) is a hard error instead of a silent fall back to the
    /// native rescorer. Default true so a named scientific workflow cannot
    /// silently execute a different model; set false only for explicit legacy
    /// compatibility.
    pub strict: bool,
    /// How the feature matrix reaches a sidecar rescorer. See [`Handoff`]. Defaults to
    /// `parquet`, which applies to nn_torch only; mokapot and entrapment sidecars always
    /// receive the tab-separated PIN.
    #[serde(default)]
    pub handoff: Handoff,
    /// Restrict the classifier's input to these feature columns, by name. Absent (the
    /// default) falls through to `feature_preset`.
    ///
    /// The restriction is a projection, not a reordering: the columns keep the order of
    /// the feature schema, only those named are read out of the competed table, and the
    /// matrix, the sidecar handoff and the training all shrink with the list. Feature
    /// selection is a memory and I/O lever, not a speed one (docs/28 section 7), and any
    /// list must clear the sensitivity gate before it becomes a default.
    ///
    /// Mutually exclusive with [`RescoreConfig::features_file`]. Every name must exist in
    /// the competed table's schema; a missing one is an error, never a silent drop.
    #[serde(default)]
    pub features: Option<Vec<String>>,
    /// The same restriction, read from a file with one feature name per line (blank lines
    /// and `#` comments ignored), which is how a 100+ name list stays readable.
    #[serde(default)]
    pub features_file: Option<String>,
    /// Named feature list used when neither `features` nor `features_file` is set. `all`
    /// is every feature the competed table carries. `compact` is the 114-name list of
    /// docs/28 section 12 (`bench/feature_selection/fs_union75_dedup.txt`, embedded in the
    /// binary), which with the hard-negative training recipe reproduced the full Extended
    /// set within seed noise on three pools (HYE A01 +1.2%, AIF -0.2%, entrapment +4.9%,
    /// spike-in FDP unchanged) at 3.4x less rescore memory. Preset names the table lacks
    /// are skipped with a log line rather than an error, so a preset tolerates a smaller
    /// `features.set`; the intersection must not be empty. Explicit lists stay strict.
    /// Default `all`: the projection is a memory lever (3.4x smaller rescore matrix), not a
    /// sensitivity one, and it cost 1.2% on the held-out HYE B01 pool under the default
    /// training (+0.2% / -0.1% / +1.5% on A01 / AIF / entrapment), so it is the option for
    /// pooled rescoring on small machines (docs/28 section 21), not the default.
    pub feature_preset: FeaturePreset,
    /// Cap the decoys the sidecar TRAINS on at this multiple of the targets it selected
    /// that iteration; 0 (the default) trains on every decoy, which is about 19:1 on a
    /// DIA pool and is where the rescore spends its time.
    ///
    /// This thins gradient steps only. Selection, scoring, target-decoy competition and
    /// q-values still run over the full pool, so the cap cannot loosen the q threshold;
    /// what it can move is the learned boundary, hence a knob and not a default.
    pub train_neg_ratio: f64,
    /// Which decoys survive [`RescoreConfig::train_neg_ratio`]. See [`NegSelect`].
    pub train_neg_select: NegSelect,
    /// Stratified thinning of whatever survived the cap: a fraction in (0, 1], or a row
    /// cap when > 1. Positives and negatives are thinned by the same factor, so the class
    /// balance is unchanged. 0 (the default) keeps every row.
    #[serde(default)]
    pub train_subsample: f64,
    /// Reuse the previous iteration's weights and optimiser state, running this many
    /// epochs from the second self-training iteration on instead of a full fresh fit.
    /// 0 (the default) refits from scratch every iteration, which is 25 epochs x 10
    /// iterations x 3 folds of the whole training set.
    pub train_warm_epochs: usize,
    /// Under `train_neg_select = hybrid`, the share of the negative budget taken from the
    /// margin (highest-scoring decoys); the rest is sampled at random. Default 0.5.
    #[serde(default = "default_margin_frac")]
    pub train_margin_frac: f64,
    /// Independent self-training passes whose out-of-fold scores are rank-averaged. 1 (the
    /// default) is a single pass. 3 was the one knob positive on every pool of the seeded
    /// sweep (docs/28 section 17), at three times the training cost.
    #[serde(default = "default_seeds")]
    pub seeds: usize,
}

fn default_margin_frac() -> f64 {
    0.5
}

fn default_seeds() -> usize {
    1
}

/// Named feature list for `RescoreConfig::feature_preset`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaturePreset {
    /// Every feature column of the competed table.
    #[default]
    All,
    /// The 114-feature list of docs/28 section 12, embedded in the engine.
    Compact,
}

/// Which decoys survive the training-set negative cap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NegSelect {
    /// A uniform random sample of the fold's decoys. The population the model sees keeps
    /// the shape of the real decoy distribution, only thinner.
    #[default]
    Random,
    /// The highest-scoring decoys under the current model: the part of the decoy
    /// distribution that still competes with accepted targets, and the only part the
    /// decision boundary depends on. Trains on hard negatives only, so the model never
    /// sees the easy bulk it must also keep rejecting.
    Margin,
    /// Half the budget from the margin, half sampled at random from the rest, so the
    /// boundary is informed by the hard cases without losing the shape of the bulk.
    Hybrid,
}
impl Default for RescoreConfig {
    fn default() -> Self {
        Self {
            classifier: t(),
            folds: 3,
            train_fdr: 0.01,
            num_iter: 10,
            max_feature_matrix_gib: 0.0, // no ceiling; previous behaviour
            python: None,
            percolator_bin: None,
            entrapment_marker: None,
            entrapment_exclude: None,
            entrapment_contaminant_markers: Vec::new(),
            entrapment_ratio: 1.0,
            strict: true,
            handoff: Handoff::Parquet,
            features: None,
            features_file: None,
            // The training recipe of docs/28 section 15, default since 2026-09-05: measured
            // with seeds on four pools against the previous defaults (every decoy, cold
            // refits): HYE A01 +1.0%, HYE B01 +2.2%, AIF -0.1%, entrapment +3.3% with the
            // spike-in FDP unchanged, at 9-19x less training time. `train_neg_ratio = 0`,
            // `train_neg_select = random`, `train_warm_epochs = 0` restore the previous
            // behaviour exactly. The compact feature preset stays opt-in (see its field).
            feature_preset: FeaturePreset::All,
            train_neg_ratio: 3.0,
            train_neg_select: NegSelect::Hybrid,
            train_subsample: 0.0,
            train_warm_epochs: 5,
            train_margin_frac: 0.5,
            seeds: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// How many DeepLC fine-tunes an experiment pays for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinetuneScope {
    /// Fine-tune DeepLC once, on the FIRST run's confident seeds, and reuse that library
    /// for every run. Each run still fits its OWN retention-time calibration (LOESS by
    /// default) on top of it.
    ///
    /// MEASURED COST (6-run ProteoBench HYE AIF set, 2026-07-28). Reuse is NOT free: the
    /// run that owned the fine-tune reached a median |RT residual| of 15.2 s, while the
    /// five reusing runs reached 20.3, 20.5, 20.9, 24.9 and 25.4 s -- +7.2 s, +47% on
    /// average -- and their calibrated RT windows widened from 145 s to 179-227 s. The
    /// degradation is MONOTONIC in acquisition order, i.e. real chromatographic drift that
    /// a single fine-tune cannot track; per-run LOESS corrects the slope (0.96-0.99) but
    /// not the scatter. Wider windows also cost compute downstream: extract roughly
    /// doubled (126 s -> 203-242 s) and features up to tripled (116 s -> 215-388 s),
    /// which claws back part of the saving.
    ///
    /// It is still the default because the fine-tune dominates a large experiment: one
    /// 36.5 min fine-tune instead of N. On an 80-run batch that is ~48 h saved against
    /// ~6.5 h of extra extract/features. But on a long batch the drift keeps growing, so
    /// prefer `PerRun` when the extra hours are affordable, and treat periodic
    /// re-fine-tuning (not yet implemented) as the better answer for very large batches.
    #[default]
    FirstRunOnly,
    /// Fine-tune separately for every run. Adapts the model weights to each run's own
    /// chromatography instead of only calibrating a shared model, which measurably
    /// tightens retention time: see the numbers on `FirstRunOnly`. Costs one full DeepLC
    /// fine-tune per run (36.5 min on the HYE library: 5.7 min training plus 30.8 min
    /// predicting 4.9M peptidoforms).
    PerRun,
}

/// How the feature matrix crosses the Rust -> Python boundary for a sidecar rescorer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Handoff {
    /// Tab-separated PIN. Percolator's format, and what `mokapot.read_pin` requires, so it
    /// is what a mokapot or entrapment sidecar receives whatever this is set to.
    Tsv,
    /// Parquet feature table with f32 features, and the default since 2026-09-05.
    ///
    /// The TSV path makes the worker parse every column into a float64 pandas frame before
    /// it builds its float32 matrix, so the text file, the frame and the matrix are alive
    /// together. Measured on the HYE competed table (2,603,894 PSMs x 387 features, one
    /// self-training iteration, 32 threads), parquet against tsv: rescore peak 29.96 ->
    /// 8.95 GB, wall 8:35 -> 6:33, sidecar file 9.53 -> 3.28 GB, the worker's read and
    /// standardise phase 111.7 -> 16.9 s, and 47,752 against 47,762 peptides at 1% with the
    /// decoy fraction 1.00% either way (docs/28 section 11). An earlier 8,858,206-PSM
    /// experiment-wide rescore went from 671.6 min to 12 min, because there the 30.18 GB
    /// TSV crossed the worker's streaming threshold and every iteration re-read a 12.77 GB
    /// memmap.
    ///
    /// Features are f32 because the TSV was already lossy (`{:.6}`) and the worker casts to
    /// f32 regardless; the two paths therefore feed marginally different values into a
    /// chaotic self-training loop, which is where that 10-peptide difference comes from.
    ///
    /// nn_torch only: `mokapot_worker.py` calls `mokapot.read_pin()` and cannot read
    /// Parquet, so a mokapot run falls back to `Tsv` with a warning instead of failing.
    #[default]
    Parquet,
}

/// Options for the experiment-wide orchestrator (`mumdia run-experiment`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExperimentConfig {
    /// How many per-run search chains to execute concurrently.
    ///
    /// 1 (default) is strictly sequential, i.e. the historical behaviour. Runs are
    /// independent, so raising this scales nearly linearly in wall time, but EACH
    /// concurrent run holds its own extraction working set (tens of GB on a large
    /// library), so the practical ceiling is memory, not cores. Raise it deliberately
    /// after checking peak RSS for a single run; 2-4 is a reasonable start on a
    /// large-memory machine. Results are unaffected: chunks are processed in index
    /// order and completion order never reaches the output.
    pub parallel_runs: usize,
    /// Whether the DeepLC fine-tune runs once for the experiment or once per run.
    /// Only consulted when `rt_im_train.finetune_deeplc` is set.
    pub finetune_scope: FinetuneScope,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            parallel_runs: 1,
            finetune_scope: FinetuneScope::FirstRunOnly,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub convert: ConvertConfig,
    pub prescan: PrescanConfig,
    pub rng_seed: u64,
    pub digest: DigestConfig,
    pub peptidoforms: PeptidoformsConfig,
    pub predict_frag: PredictFragConfig,
    pub search_seed: SearchSeedConfig,
    pub rt_im_train: RtImTrainConfig,
    pub extract: ExtractConfig,
    pub features: FeaturesConfig,
    pub compete: CompeteConfig,
    pub rescore: RescoreConfig,
    pub quant: QuantConfig,
    #[serde(default)]
    pub mbr: MbrConfig,
    #[serde(default)]
    pub experiment: ExperimentConfig,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            rng_seed: 0,
            convert: t(),
            prescan: t(),
            digest: t(),
            peptidoforms: t(),
            predict_frag: t(),
            search_seed: t(),
            rt_im_train: t(),
            extract: t(),
            features: t(),
            compete: t(),
            rescore: t(),
            quant: t(),
            mbr: t(),
            experiment: t(),
        }
    }
}

impl Config {
    /// Parse from a JSON string, rejecting unknown keys, then validate.
    pub fn from_json(s: &str) -> Result<Self, crate::error::ConfigError> {
        let c: Config =
            serde_json::from_str(s).map_err(|e| crate::error::ConfigError::Parse(e.to_string()))?;
        c.validate()?;
        Ok(c)
    }

    /// Reject config combinations that would silently produce wrong or invalid
    /// results, so a misconfiguration fails loudly at load instead of yielding a
    /// bogus run. Defaults always pass.
    pub fn validate(&self) -> Result<(), crate::error::ConfigError> {
        use crate::error::ConfigError::Invalid;
        if self.digest.decoy.strategy == DecoyStrategy::DiannShift {
            return Err(Invalid(
                "digest.decoy.strategy=diann_shift is not implemented in the engine \
                 digest (it produces zero decoys and an invalid target-decoy FDR). Use \
                 \"reverse\" or \"scramble\", or supply a prebuilt library whose decoys \
                 are already generated."
                    .into(),
            ));
        }
        if self.rt_im_train.calibration_method == CalibrationMethod::None {
            return Err(Invalid(
                "rt_im_train.calibration_method=none is not valid (it silently falls \
                 through to the linear fit). Use \"linear\" or \"loess\"."
                    .into(),
            ));
        }
        if self.extract.retain_top_peaks == 0 {
            return Err(Invalid(
                "extract.retain_top_peaks must be >= 1 (1 = legacy single-apex \
                 behaviour; K>1 retains up to K peak groups per candidate)."
                    .into(),
            ));
        }
        if !self.extract.gate_min_score.is_finite()
            || !(0.0..=1.0).contains(&self.extract.gate_min_score)
        {
            return Err(Invalid(
                "extract.gate_min_score must be finite and in [0, 1] (0 disables \
                the gate)."
                    .into(),
            ));
        }
        if !self.quant.fixed_window_s.is_finite() || self.quant.fixed_window_s < 0.0 {
            return Err(Invalid(
                "quant.fixed_window_s must be finite and >= 0 (0 disables the fixed \
                 integration window)."
                    .into(),
            ));
        }
        if !self.quant.baseline_quantile.is_finite()
            || !(0.0..=1.0).contains(&self.quant.baseline_quantile)
        {
            return Err(Invalid(
                "quant.baseline_quantile must be finite and in [0, 1].".into(),
            ));
        }
        // The entrapment estimate is
        // `(ratio * n_entrap + 1) / max(1, n_real)`, so `ratio` scales the entire
        // numerator. At 0 it collapses to `1/n_real`: with more than 100 real targets
        // every row passes at 1% FDR with no null contributing at all. Negative values
        // make q negative, and both `count_targets_at_q` and `passes_quant_filter` accept
        // a negative q as passing. This is the failure mode of computing the ratio the
        // wrong way round (N_entrap/N_real, a small number), which silently reports a
        // near-zero FDR on the tool whose whole job is to validate the FDR.
        // ── divisors and window widths ──────────────────────────────────────────
        //
        // Every field below is either divided by or used as a bin width, so a zero or a
        // negative value does not degrade the result, it destroys it: a zero divisor
        // gives an infinite or NaN bucket index that then goes through `as i64`, which in
        // Rust is a saturating cast, so every row lands in one bucket silently. There is
        // no error and no warning, just a stage that quietly stops distinguishing
        // anything.
        //
        // `Config::validate` grew field by field around whatever had most recently gone
        // wrong (`min_frag_corr`, `fixed_window_s`, `baseline_quantile`, `train_fdr`,
        // `entrapment_ratio`), so the rest of the numeric surface was never covered. This
        // is the divisor set, audited together.
        for (name, value) in [
            // `prescan.rs`: `(rt - lo) / bin` selects the retention-time cell.
            ("prescan.rt_bin_s", self.prescan.rt_bin_s),
            // `compete.rs`: `(apex_rt / tolerance).round()` is the apex bucket under
            // `group_by = apex`.
            (
                "compete.apex_rt_tolerance_s",
                self.compete.apex_rt_tolerance_s,
            ),
            // `prescan.rs`: fragment match half-width in Da.
            ("prescan.tol_da", self.prescan.tol_da),
            // Fragment/precursor tolerances. Zero admits nothing and overflows the
            // log-bin count in `binning.rs`; negative completes the run and reports zero
            // identifications with no error.
            (
                "search_seed.fragment_tol_ppm",
                self.search_seed.fragment_tol_ppm,
            ),
            ("extract.frag_tol_ppm", self.extract.frag_tol_ppm),
            ("extract.prec_tol_ppm", self.extract.prec_tol_ppm),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Invalid(format!(
                    "{name} must be finite and > 0 (got {value}); it is used as a divisor \
                     or a bin width, and a zero or negative value collapses every row \
                     into one bucket instead of failing"
                )));
            }
        }

        if !self.rescore.entrapment_ratio.is_finite() || self.rescore.entrapment_ratio <= 0.0 {
            return Err(Invalid(
                "rescore.entrapment_ratio must be finite and > 0. It is \
                 N_real_library / N_entrapment_library, so it is >= 1 for the usual \
                 spike-in proportions; 0 or a negative value makes every PSM pass at any \
                 q threshold. Check you have not inverted the ratio."
                    .into(),
            ));
        }
        if self.rescore.folds < 2 {
            return Err(Invalid(
                "rescore.folds must be >= 2 so every PSM can receive an \
                 out-of-fold score."
                    .into(),
            ));
        }
        if self.rescore.num_iter == 0 {
            return Err(Invalid(
                "rescore.num_iter must be >= 1 for iterative model training.".into(),
            ));
        }
        if !self.rescore.train_fdr.is_finite()
            || self.rescore.train_fdr <= 0.0
            || self.rescore.train_fdr > 1.0
        {
            return Err(Invalid(
                "rescore.train_fdr must be finite and in (0, 1].".into(),
            ));
        }
        if matches!(
            self.rescore.classifier,
            RescorerKind::Mokapot | RescorerKind::NnTorch
        ) && self.rescore.python.is_none()
        {
            return Err(Invalid(format!(
                "rescore.classifier={:?} requires rescore.python",
                self.rescore.classifier
            )));
        }
        if self.rescore.classifier == RescorerKind::Percolator {
            return Err(Invalid(
                "rescore.classifier=percolator is not wired; use native_tda, \
                 mokapot, nn_torch, or entrapment."
                    .into(),
            ));
        }
        if self.rescore.classifier == RescorerKind::Entrapment
            && self.rescore.entrapment_marker.is_none()
        {
            return Err(Invalid(
                "rescore.classifier=entrapment requires \
                 rescore.entrapment_marker."
                    .into(),
            ));
        }
        // MBR fields that are accepted and documented but not yet read by any stage. Setting
        // them has NO effect, which is worse than rejecting them: the run looks configured.
        // Warn rather than error so existing configs keep parsing (serde uses
        // deny_unknown_fields, so deleting the fields outright would break them).
        {
            let d = MbrConfig::default();
            let mut inert: Vec<&str> = Vec::new();
            if (self.mbr.rt_window_s - d.rt_window_s).abs() > f64::EPSILON {
                inert.push("mbr.rt_window_s");
            }
            if self.mbr.decoy_transfer != d.decoy_transfer {
                inert.push("mbr.decoy_transfer");
            }
            if self.mbr.requant_all != d.requant_all {
                inert.push("mbr.requant_all");
            }
            if !inert.is_empty() {
                tracing::warn!(
                    fields = ?inert,
                    "these MBR config fields are parsed but not read by any stage yet, so \
                     setting them has no effect on this run"
                );
            }
        }
        // The fourth inert field, and the only one that used to warn about nothing. Under
        // `deny_unknown_fields` a user reasonably reads an accepted key as an honoured
        // one, so silence here means a run that looks configured for external percolator
        // and is not.
        if self.rescore.percolator_bin.is_some() {
            tracing::warn!(
                "rescore.percolator_bin is parsed but no stage launches percolator, so \
                 setting it has no effect. Use rescore.classifier = mokapot or nn_torch \
                 for an external classifier."
            );
        }
        {
            // Only `strategy == None` vs `!= None` is ever tested, so the non-None variants
            // are indistinguishable in behaviour.
            if self.mbr.strategy != MbrStrategy::None {
                tracing::warn!(
                    strategy = ?self.mbr.strategy,
                    "mbr.strategy currently only distinguishes none vs not-none; the \
                     individual non-none variants behave identically"
                );
            }
        }
        if self.quant.fixed_window_s > 0.0 && self.quant.fixed_scan_halfwidth > 0 {
            tracing::warn!(
                fixed_window_s = self.quant.fixed_window_s,
                fixed_scan_halfwidth = self.quant.fixed_scan_halfwidth,
                "quant: both fixed-window forms are set; the seconds form wins and \
                 fixed_scan_halfwidth is ignored"
            );
        }
        if self.quant.baseline_subtract
            && self.quant.fixed_scan_halfwidth == 0
            && self.quant.fixed_window_s == 0.0
        {
            tracing::warn!(
                "quant.baseline_subtract applies only to the fixed-window integration, \
                 which is off (fixed_scan_halfwidth = 0 and fixed_window_s = 0), so no \
                 baseline is subtracted"
            );
        }
        Ok(())
    }

    /// Apply a named tuning profile on top of the current config. `dia` is the
    /// validated DIA preset (Extended features + rolling-window apex + RT prior);
    /// the other extraction defaults (emit_window_grid, reverse decoys,
    /// gate_min_score) remain conservative baselines. Lets one command reach a
    /// respectable result without hand-authoring the full config JSON.
    pub fn apply_profile(&mut self, name: &str) -> Result<(), crate::error::ConfigError> {
        match name {
            "dia" => {
                self.features.set = FeatureSet::Extended;
                self.extract.apex_count_window = 5;
                self.extract.apex_rt_prior_s = 120.0;
            }
            other => {
                return Err(crate::error::ConfigError::Invalid(format!(
                    "unknown --profile '{other}' (known profiles: dia)"
                )));
            }
        }
        Ok(())
    }

    /// Canonical JSON of the fully-resolved config, for hashing into the
    /// manifest (docs/02_config_and_data_model.md).
    pub fn canonical_json(&self) -> String {
        serde_json::to_string(self).expect("config serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_fixed_window_fields_round_trip_and_are_validated() {
        let c = Config::from_json(
            r#"{"quant":{"fragment_selection":"predicted","top_n_fragments":12,
                 "fixed_window_s":5.0,"fixed_scan_halfwidth":3,"baseline_subtract":true,
                 "baseline_flank_scans":8,"baseline_quantile":0.5}}"#,
        )
        .unwrap();
        assert_eq!(c.quant.fragment_selection, FragmentSelection::Predicted);
        assert_eq!(c.quant.top_n_fragments, 12);
        assert_eq!(c.quant.fixed_window_s, 5.0);
        assert_eq!(c.quant.fixed_scan_halfwidth, 3);
        assert!(c.quant.baseline_subtract);
        assert_eq!(c.quant.baseline_flank_scans, 8);
        assert_eq!(c.quant.baseline_quantile, 0.5);

        // The defaults must reproduce the pre-2026-08 integration exactly: neither fixed
        // form on, ranking by observed area, no baseline. A config written before these
        // fields existed therefore quantifies as it did before.
        let d = QuantConfig::default();
        assert_eq!(d.fragment_selection, FragmentSelection::ObservedArea);
        assert_eq!(d.fixed_scan_halfwidth, 0);
        assert_eq!(d.fixed_window_s, 0.0);
        assert!(!d.baseline_subtract);
        assert_eq!(
            Config::default().quant.fragment_selection,
            d.fragment_selection
        );

        assert!(Config::from_json(r#"{"quant":{"fixed_window_s":-1.0}}"#).is_err());
        assert!(Config::from_json(r#"{"quant":{"baseline_quantile":1.5}}"#).is_err());
        assert!(Config::from_json(r#"{"quant":{"baseline_quantile":-0.5}}"#).is_err());
        // Unknown enum variants must fail rather than fall back to the default ranking.
        assert!(Config::from_json(r#"{"quant":{"fragment_selection":"library"}}"#).is_err());
    }

    #[test]
    fn shipped_configs_parse() {
        // `Config` is `deny_unknown_fields`, so a documentation key such as `_comment` in a
        // shipped config is a hard PARSE error -- the whole config is rejected before any
        // value is read. That shipped once and was only caught by running `doctor` on the
        // deployment target. The workspace suite passing while a shipped config was unusable
        // is the gap this closes: a config is an artifact the engine must accept, not merely
        // a file that happens to be valid JSON.
        //
        // Only TRACKED configs are listed. Machine- or collaborator-specific configs stay
        // untracked (see .gitignore), so they are checked by `mumdia doctor`, not here.
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
        for name in [
            "configs/examples/native.json",
            "configs/examples/fasta-sidecars.json",
            "configs/examples/diann-library.json",
            "docker/config.dia.json",
            "docker/config.diann-lib.json",
        ] {
            let path = format!("{root}/{name}");
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue; // absent in some checkout layouts
            };
            Config::from_json(&text)
                .unwrap_or_else(|e| panic!("shipped config {name} does not parse: {e}"));
        }
    }

    #[test]
    fn rescore_defaults_are_the_training_recipe_and_json_omission_keeps_them() {
        let d = RescoreConfig::default();
        assert_eq!(d.feature_preset, FeaturePreset::All);
        assert_eq!(d.train_neg_ratio, 3.0);
        assert_eq!(d.train_neg_select, NegSelect::Hybrid);
        assert_eq!(d.train_warm_epochs, 5);
        // A config that does not mention them gets the same values (no field-level
        // serde default shadowing the struct default), and each can be switched off.
        let c = Config::from_json(r#"{"rescore":{"classifier":"native_tda"}}"#).expect("parses");
        assert_eq!(c.rescore.feature_preset, FeaturePreset::All);
        assert_eq!(c.rescore.train_neg_ratio, 3.0);
        assert_eq!(c.rescore.train_neg_select, NegSelect::Hybrid);
        assert_eq!(c.rescore.train_warm_epochs, 5);
        let c = Config::from_json(
            r#"{"rescore":{"feature_preset":"compact","train_neg_ratio":0.0,"train_neg_select":"random","train_warm_epochs":0}}"#,
        )
        .expect("parses");
        assert_eq!(c.rescore.feature_preset, FeaturePreset::Compact);
        assert_eq!(c.rescore.train_neg_ratio, 0.0);
        assert_eq!(c.rescore.train_neg_select, NegSelect::Random);
        assert_eq!(c.rescore.train_warm_epochs, 0);
    }

    #[test]
    fn library_irt_resolves_per_mode_and_interpreter() {
        let mut rt = RtImTrainConfig::default();
        assert_eq!(rt.library_irt, LibraryIrt::Auto);
        assert!(rt.repredicts_library_irt(true, true));
        assert!(
            !rt.repredicts_library_irt(true, false),
            "auto without an interpreter keeps the library iRT"
        );
        assert!(
            !rt.repredicts_library_irt(false, true),
            "FASTA mode never re-predicts"
        );
        rt.finetune_deeplc = true;
        assert!(
            !rt.repredicts_library_irt(true, true),
            "the fine-tune re-predicts by itself"
        );
        rt.finetune_deeplc = false;
        rt.library_irt = LibraryIrt::Library;
        assert!(!rt.repredicts_library_irt(true, true));
        rt.library_irt = LibraryIrt::Deeplc;
        assert!(
            rt.repredicts_library_irt(true, false),
            "explicit deeplc is a preflight matter, not a fallback"
        );
        let c = Config::from_json(r#"{"rt_im_train":{"library_irt":"deeplc"}}"#).expect("parses");
        assert_eq!(c.rt_im_train.library_irt, LibraryIrt::Deeplc);
        let c = Config::from_json("{}").expect("parses");
        assert_eq!(c.rt_im_train.library_irt, LibraryIrt::Auto);
    }

    #[test]
    fn experiment_finetune_scope_defaults_to_first_run_only() {
        // The expensive default must be the cheap one: paying a DeepLC fine-tune per run
        // is hours-to-days on a large experiment, and each run calibrates itself anyway.
        assert_eq!(
            Config::default().experiment.finetune_scope,
            FinetuneScope::FirstRunOnly
        );
        let c = Config::from_json("{}").expect("empty config parses");
        assert_eq!(c.experiment.finetune_scope, FinetuneScope::FirstRunOnly);
        let c = Config::from_json(r#"{"experiment":{"finetune_scope":"per_run"}}"#)
            .expect("per_run parses");
        assert_eq!(c.experiment.finetune_scope, FinetuneScope::PerRun);
        // Old configs that predate the section keep working.
        let c = Config::from_json(r#"{"experiment":{"parallel_runs":3}}"#).expect("parses");
        assert_eq!(c.experiment.parallel_runs, 3);
        assert_eq!(c.experiment.finetune_scope, FinetuneScope::FirstRunOnly);
    }

    #[test]
    fn experiment_parallel_runs_defaults_to_sequential() {
        // The default must stay 1: concurrent per-run chains multiply peak memory,
        // so opting in is the caller's decision, and 1 reproduces the historical
        // sequential orchestrator exactly.
        assert_eq!(Config::default().experiment.parallel_runs, 1);
        // Old configs written before the section existed must still parse.
        let c = Config::from_json("{}").expect("empty config parses");
        assert_eq!(c.experiment.parallel_runs, 1);
        let c = Config::from_json(r#"{"experiment":{"parallel_runs":4}}"#).expect("parses");
        assert_eq!(c.experiment.parallel_runs, 4);
    }

    #[test]
    fn default_roundtrips() {
        let c = Config::default();
        let j = serde_json::to_string(&c).unwrap();
        let back: Config = serde_json::from_str(&j).unwrap();
        assert_eq!(back.digest.min_len, 5);
        assert_eq!(back.features.set, FeatureSet::Minimal);
        assert_eq!(back.search_seed.top_n_peaks, 300);
        assert!(back.rescore.strict);
    }

    #[test]
    fn unknown_key_rejected() {
        let j = r#"{"digest":{"min_len":7,"bogus":1}}"#;
        assert!(Config::from_json(j).is_err());
    }

    #[test]
    fn partial_override_keeps_defaults() {
        let j = r#"{"digest":{"min_len":7}}"#;
        let c = Config::from_json(j).unwrap();
        assert_eq!(c.digest.min_len, 7);
        assert_eq!(c.digest.max_len, 50);
        assert_eq!(c.peptidoforms.charge_max, 3);
        assert_eq!(c.search_seed.top_n_peaks, 300);
    }

    #[test]
    fn precursor_q_quant_filter_parses_and_serializes() {
        let c = Config::from_json(r#"{"quant":{"q_filter":"precursor_q"}}"#).unwrap();
        assert_eq!(c.quant.q_filter, QuantQColumn::PrecursorQ);
        assert_eq!(
            serde_json::to_string(&c.quant.q_filter).unwrap(),
            r#""precursor_q""#
        );
    }

    #[test]
    fn invalid_rescore_contracts_are_rejected() {
        assert!(Config::from_json(r#"{"rescore":{"folds":1}}"#).is_err());
        assert!(Config::from_json(r#"{"rescore":{"num_iter":0}}"#).is_err());
        assert!(Config::from_json(r#"{"rescore":{"classifier":"nn_torch"}}"#).is_err());
        assert!(Config::from_json(r#"{"rescore":{"classifier":"percolator"}}"#).is_err());
        assert!(Config::from_json(r#"{"rescore":{"classifier":"entrapment"}}"#).is_err());
    }

    #[test]
    fn explicit_uncapped_seed_and_invalid_gate_are_distinguished() {
        let c = Config::from_json(r#"{"search_seed":{"top_n_peaks":0}}"#).unwrap();
        assert_eq!(c.search_seed.top_n_peaks, 0);

        assert!(Config::from_json(r#"{"extract":{"gate_min_score":-0.1}}"#).is_err());
        assert!(Config::from_json(r#"{"extract":{"gate_min_score":1.1}}"#).is_err());
    }

    #[test]
    fn entrapment_ratio_must_be_positive() {
        // `(ratio * n_entrap + 1) / max(1, n_real)`: at ratio 0 this collapses to
        // 1/n_real, so above ~100 real targets every row passes at 1% with no null
        // contributing. A negative ratio makes q negative, which both
        // `count_targets_at_q` and `passes_quant_filter` accept as passing. That is the
        // failure mode of inverting the ratio, on the tool whose job is validating FDR.
        let mut cfg = Config::default();
        assert!(cfg.validate().is_ok(), "default must be valid");

        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            cfg.rescore.entrapment_ratio = bad;
            let err = cfg
                .validate()
                .expect_err(&format!("{bad} must be rejected"));
            assert!(
                format!("{err}").contains("entrapment_ratio"),
                "error should name the field: {err}"
            );
        }
        cfg.rescore.entrapment_ratio = 3.5;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn divisors_must_be_finite_and_positive() {
        // Each of these is divided by or used as a bin width. A zero divisor produces an
        // infinite or NaN bucket index, and `as i64` in Rust saturates rather than
        // trapping, so every row silently lands in one bucket: the stage stops
        // distinguishing anything and reports no error at all.
        let mut cfg = Config::default();
        assert!(cfg.validate().is_ok(), "default must be valid");

        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut c = Config::default();
            c.prescan.rt_bin_s = bad;
            let e = c
                .validate()
                .expect_err("prescan.rt_bin_s must reject {bad}");
            assert!(format!("{e}").contains("prescan.rt_bin_s"), "{e}");

            let mut c = Config::default();
            c.compete.apex_rt_tolerance_s = bad;
            let e = c
                .validate()
                .expect_err("compete.apex_rt_tolerance_s must reject");
            assert!(format!("{e}").contains("apex_rt_tolerance_s"), "{e}");

            let mut c = Config::default();
            c.search_seed.fragment_tol_ppm = bad;
            let e = c.validate().expect_err("fragment_tol_ppm must reject");
            assert!(format!("{e}").contains("fragment_tol_ppm"), "{e}");
        }

        // A legitimate change still passes.
        cfg.prescan.rt_bin_s = 10.0;
        cfg.compete.apex_rt_tolerance_s = 2.5;
        assert!(cfg.validate().is_ok());
    }
}
