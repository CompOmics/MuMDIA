//! Typed configuration (PLAN.md Section 7, Section 9).
//!
//! One serde structure with per-stage sections, `#[serde(default)]` on every
//! field, and `deny_unknown_fields` so misconfiguration fails loudly. Every
//! choice point is an enum backed by a strategy (Section 9.1); MVP ships only
//! the strategies MVP needs, with MVP-conservative defaults (Section 10):
//! fixed tolerances, one documented decoy scheme, the `minimal` feature set.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Strategy enums (Section 9)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoyStrategy {
    /// Reverse the sequence keeping the C-terminal residue fixed. Documented,
    /// clean-room default for MVP (PLAN.md Section 11). No borrowed map.
    #[default]
    Reverse,
    /// Deterministic seeded shuffle of the interior residues.
    Scramble,
    /// DIA-NN terminal-residue fragment m/z shift. Deferred: license-checked
    /// addition (PLAN.md Section 11), not part of MVP.
    DiannShift,
    None,
}

/// Fragment-matcher backend for search-seed and extract (fragindex_spec).
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
    /// MVP feature set (PLAN.md Section 10).
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
    /// DeepLC Python sidecar (PLAN.md Section 0, Section 3.2).
    Deeplc,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FragPredictorKind {
    /// Native heuristic intensity model (no Python). MVP default.
    #[default]
    Native,
    /// MS2PIP Python sidecar (PLAN.md Section 0, Section 3.2).
    Ms2pip,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RescorerKind {
    /// Native semi-supervised linear rescorer + target-decoy q-values. MVP
    /// default (always available).
    #[default]
    NativeTda,
    /// Mokapot Python sidecar (PLAN.md Section 0).
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
    /// charge >= this threshold (PLAN.md Decision 3). Default 2: DIA-NN uses
    /// doubly-charged fragments for ~16% of charge-2 precursors' transitions, so
    /// blocking them (the old default of 3) discarded real signal.
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
    /// Max reported PSMs per spectrum (wide-window DIA, PLAN.md Stage S).
    pub report_psms: usize,
    /// Minimum matched fragments for a seed PSM.
    pub min_matched_peaks: usize,
    /// If > 0, probe only the `top_n_peaks` most intense peaks per MS2 scan
    /// (0 = all peaks). The seed only produces calibration anchors (RT/mass/IM),
    /// which come from abundant peptides, so this cuts the dominant per-peak index
    /// probing cost without discarding peaks from the downstream extraction
    /// artifact. Default 300; set to 0 to probe every converted peak.
    pub top_n_peaks: usize,
    /// Fragment-matcher backend (fragindex_spec). Default `Fragindex`.
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
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExtractConfig {
    pub fixed_scan_window: usize,
    pub frag_tol_ppm: f64,
    pub prec_tol_ppm: f64,
    /// tier-(b) minimum matched fragment count.
    pub presence_min_matched: usize,
    /// minimum distinct fragments for acceptance.
    pub presence_min_fragments: usize,
    /// minimum simultaneously-present fragments over the consecutive-scan run.
    pub presence_min_coelution: usize,
    /// tier-(d) spectral-agreement gate: reject a candidate whose apex observed
    /// fragment intensities correlate with the predicted pattern below this.
    /// Applied symmetrically to targets and decoys, but that alone does not prove
    /// null exchangeability in chimeric DIA; validate every threshold with an
    /// independent entrapment. 0 disables.
    pub min_frag_corr: f64,
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
    /// Fragment-matcher backend (fragindex_spec). Default `Fragindex`.
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
    /// predicted transitions co-elute is a more reliable apex. `false` (default)
    /// keeps the legacy signature-intensity apex. The rolling distinct-fragment
    /// count (`apex_count_window`) still gates which scans qualify in both modes.
    pub apex_evidence_rank: bool,
    /// Emit the four gate-diagnostic scores (`gate_apex`, `gate_peak_spectral`,
    /// `gate_coelution`, `gate_spectral_entropy`) as extra `psms.parquet` columns,
    /// for the offline gate-metric comparison. Default `false` (diagnostic sidecar,
    /// like `emit_candidate_audit`): when off, neither the columns nor the extra
    /// per-candidate score computation happen, so the default chain is byte-identical.
    pub emit_gate_diagnostics: bool,
    /// Which spectral-agreement score the `min_frag_corr` gate thresholds
    /// (sensitivity program). The legacy gate uses a single apex-scan intensity
    /// Pearson, which one chimeric scan can dominate. See [`GateMode`].
    pub gate_mode: GateMode,
    /// Second threshold for `GateMode::Combined`: the co-elution score must exceed
    /// this while the peak-integrated spectral score exceeds `min_frag_corr`.
    /// Requiring BOTH is more specific (rejects interferents that pass one axis).
    pub gate_coelution_min: f64,
}
impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
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
            // (comment.md S1); still a hard gate, not the soft/budgeted redesign.
            min_frag_corr: 0.2,
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
            apex_evidence_rank: false, // legacy signature-intensity apex
            emit_gate_diagnostics: false, // diagnostic gate-score columns; off in production
            gate_mode: GateMode::ApexPearson, // legacy single-scan intensity Pearson
            gate_coelution_min: 0.5, // used only by GateMode::Combined
        }
    }
}

/// Spectral-agreement score the extraction acceptance gate (`min_frag_corr`)
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
    /// Require BOTH: peak-integrated spectral Pearson >= `min_frag_corr` AND the
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
    /// external tooling. At 1.5M rows x 387 features it is a ~5.4 GB text write, so set
    /// this to false to skip it when nothing downstream needs it. Default true, so the
    /// artifact keeps appearing unless it is explicitly turned off.
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
    /// (Pearson of the observed apex isotope heights [i0,i1,i2] against the
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
            emit_pin: true, // preserve the artifact; set false to skip a ~5.4 GB text write
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
            group_by: CompeteGroupBy::Precursor,
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
    Precursor,
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
        }
    }
}

/// Match-between-runs strategy (Stage D3, `mbr_plan.md`). Default `None` reproduces
/// the current chain byte-for-byte. Later variants transfer identification evidence
/// across a run set: `EmpiricalLibrary` builds the consensus anchor library only;
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
    /// Transfer RT half-window (seconds) around the cross-run-predicted RT. The M2
    /// leave-target-out residual was ~17 s at p95, ~15x tighter than the search
    /// window; this is the default so the false-transfer search space stays small.
    pub rt_window_s: f64,
    /// Which decoy-transfer null estimates the false-transfer rate (M4).
    pub decoy_transfer: DecoyTransfer,
    /// Minimum correlation of the observed fragment pattern to the empirical
    /// consensus for a transfer to be accepted (interference guard; 0 disables).
    pub consensus_corr_min: f64,
    /// Requantify already-identified precursors too (fill the matrix), not only
    /// transferred ones. Only used when `strategy = Full`.
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
    pub python: Option<String>,
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
}
impl Default for RescoreConfig {
    fn default() -> Self {
        Self {
            classifier: t(),
            folds: 3,
            train_fdr: 0.01,
            num_iter: 10,
            python: None,
            percolator_bin: None,
            entrapment_marker: None,
            entrapment_exclude: None,
            entrapment_contaminant_markers: Vec::new(),
            entrapment_ratio: 1.0,
            strict: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
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
}
impl Default for Config {
    fn default() -> Self {
        Self {
            rng_seed: 0,
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
        if !self.extract.min_frag_corr.is_finite()
            || !(0.0..=1.0).contains(&self.extract.min_frag_corr)
        {
            return Err(Invalid(
                "extract.min_frag_corr must be finite and in [0, 1] (0 disables \
                the gate)."
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
        Ok(())
    }

    /// Apply a named tuning profile on top of the current config. `dia` is the
    /// validated DIA preset (Extended features + rolling-window apex + RT prior);
    /// the other extraction defaults (emit_window_grid, reverse decoys,
    /// min_frag_corr) remain conservative baselines. Lets one command reach a
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
    /// manifest (PLAN.md Section 9.1).
    pub fn canonical_json(&self) -> String {
        serde_json::to_string(self).expect("config serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        assert!(Config::from_json(r#"{"extract":{"min_frag_corr":-0.1}}"#).is_err());
        assert!(Config::from_json(r#"{"extract":{"min_frag_corr":1.1}}"#).is_err());
    }
}
