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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoyStrategy {
    /// Reverse the sequence keeping the C-terminal residue fixed. Documented,
    /// clean-room default for MVP (PLAN.md Section 11). No borrowed map.
    Reverse,
    /// Deterministic seeded shuffle of the interior residues.
    Scramble,
    /// DIA-NN terminal-residue fragment m/z shift. Deferred: license-checked
    /// addition (PLAN.md Section 11), not part of MVP.
    DiannShift,
    None,
}
impl Default for DecoyStrategy {
    fn default() -> Self {
        DecoyStrategy::Reverse
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoySource {
    /// MuMDIA generates decoys; the seed engine is told not to.
    Library,
    SearchEngine,
}
impl Default for DecoySource {
    fn default() -> Self {
        DecoySource::Library
    }
}

/// Fragment-matcher backend for search-seed and extract (fragindex_spec).
/// Default `Fragindex` (log-bin CSR matcher): on narrow-window DIA it is ~1.95x
/// faster in search-seed and ~1.26x in extract with essentially unchanged IDs
/// (HYE B_01: peptides -0.1%); `Bucketed` is the previous `Library::page_search`
/// path (retained for A/B and for the AIF full-range-window case, where the
/// predicate difference shifts IDs more); `Naive` is the band-join reference
/// (equivalence tests only).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatcherKind {
    Bucketed,
    Fragindex,
    Naive,
}
impl Default for MatcherKind {
    fn default() -> Self {
        MatcherKind::Fragindex
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Enzyme {
    /// Trypsin/P: cut after K or R (including before P).
    TrypsinP,
    /// Classic trypsin: cut after K or R but not before P.
    Trypsin,
}
impl Default for Enzyme {
    fn default() -> Self {
        Enzyme::TrypsinP
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToleranceRegime {
    /// MVP-conservative: fixed ppm tolerances (PLAN.md Section 10).
    Fixed,
    /// v1: learned per-run percentile-driven optimizer (PLAN.md Section 8.4).
    LearnedPercentile,
}
impl Default for ToleranceRegime {
    fn default() -> Self {
        ToleranceRegime::Fixed
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationMethod {
    Loess,
    Linear,
    None,
}
impl Default for CalibrationMethod {
    fn default() -> Self {
        CalibrationMethod::Loess
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureSet {
    /// MVP feature set (PLAN.md Section 10).
    Minimal,
    Rich,
    Custom,
    /// Minimal + Rich + the extended battery (DIA-NN / OpenSWATH / AlphaDIA /
    /// MS2Rescore / OktoberFest analogs + novel families) from the per-family
    /// modules in `stages/features/`. Superset, opt-in; the classifier picks the
    /// signal it can use (esp. under the nonlinear `Entrapment` rescorer).
    Extended,
}
impl Default for FeatureSet {
    fn default() -> Self {
        FeatureSet::Minimal
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RtPredictorKind {
    /// Native additive retention-coefficient model (no Python). MVP default so
    /// the engine runs with zero external runtime dependencies.
    Native,
    /// DeepLC Python sidecar (PLAN.md Section 0, Section 3.2).
    Deeplc,
}
impl Default for RtPredictorKind {
    fn default() -> Self {
        RtPredictorKind::Native
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FragPredictorKind {
    /// Native heuristic intensity model (no Python). MVP default.
    Native,
    /// MS2PIP Python sidecar (PLAN.md Section 0, Section 3.2).
    Ms2pip,
}
impl Default for FragPredictorKind {
    fn default() -> Self {
        FragPredictorKind::Native
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RescorerKind {
    /// Native semi-supervised linear rescorer + target-decoy q-values. MVP
    /// default (always available).
    NativeTda,
    /// Mokapot Python sidecar (PLAN.md Section 0).
    Mokapot,
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
impl Default for RescorerKind {
    fn default() -> Self {
        RescorerKind::NativeTda
    }
}

/// Fragment-peak apportionment when one observed MS2 peak matches the fragments
/// of several co-isolated, co-eluting candidates (near-universal in wide-window
/// DIA: ~98% of fragment m/z collide within tolerance). Decides how the peak's
/// intensity is shared, to stop a chimeric candidate borrowing a real peptide's
/// peak wholesale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PeakClaim {
    /// Every matching candidate gets the full peak intensity (legacy default).
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
}
impl Default for PeakClaim {
    fn default() -> Self {
        PeakClaim::None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanWindowMode {
    PeakWidthDerived,
    Fixed,
}
impl Default for ScanWindowMode {
    fn default() -> Self {
        ScanWindowMode::PeakWidthDerived
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
    pub source: DecoySource,
    pub ratio: u32,
}
impl Default for DecoyConfig {
    fn default() -> Self {
        Self {
            strategy: t(),
            source: t(),
            ratio: 1,
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
}
impl Default for DigestConfig {
    fn default() -> Self {
        Self {
            enzyme: t(),
            missed_cleavages: 2,
            min_len: 5,
            max_len: 50,
            decoy: t(),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownModPolicy {
    Error,
    Skip,
}
impl Default for UnknownModPolicy {
    fn default() -> Self {
        UnknownModPolicy::Error
    }
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
    pub precursor_tol_ppm: f64,
    pub fragment_tol_ppm: f64,
    /// Max reported PSMs per spectrum (wide-window DIA, PLAN.md Stage S).
    pub report_psms: usize,
    /// Minimum matched fragments for a seed PSM.
    pub min_matched_peaks: usize,
    /// If > 0, probe only the `top_n_peaks` most intense peaks per MS2 scan
    /// (0 = all peaks). The seed only produces calibration anchors (RT/mass/IM),
    /// which come from abundant peptides, so this cuts the dominant per-peak index
    /// probing cost with negligible anchor loss.
    pub top_n_peaks: usize,
    /// Fragment-matcher backend (fragindex_spec). Default `Fragindex`.
    pub matcher: MatcherKind,
}
impl Default for SearchSeedConfig {
    fn default() -> Self {
        Self {
            fdr_seed: 0.01,
            precursor_tol_ppm: 20.0,
            fragment_tol_ppm: 20.0,
            report_psms: 5,
            min_matched_peaks: 4,
            top_n_peaks: 0,
            matcher: MatcherKind::Fragindex,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RtImTrainConfig {
    pub tolerance_regime: ToleranceRegime,
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
}
impl Default for RtImTrainConfig {
    fn default() -> Self {
        Self {
            tolerance_regime: t(),
            calibration_method: t(),
            q_train: 0.01,
            p_rt: 0.95,
            rt_window_multiplier: 1.0,
            min_seed_for_calibration: 50,
            loess_span: 0.3,
            fallback_rt_window_s: 120.0,
            finetune_deeplc: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExtractConfig {
    pub scan_window_mode: ScanWindowMode,
    pub scan_scale: f64,
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
    /// fragment intensities correlate with the predicted pattern below this
    /// (Pearson). Applied symmetrically to targets and decoys, it removes
    /// chimeric false matches so the target-decoy null is valid. 0 disables.
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
    /// true elution instead. 0 = use all matched fragments (legacy behavior).
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
    /// Emit per-fragment chromatograms on the FULL isolation-window scan grid with
    /// 0.0 where a fragment is absent (aggregating scans of the same isolation
    /// window), so the elution profile drops to zero between peaks and the
    /// features-stage boundary calling is not misled by interpolated gaps.
    pub emit_window_grid: bool,
    /// tier-(c) k-select cap: exact scores run on at most this many per cell.
    pub k_select: usize,
    /// m/z bucket size (power of two).
    pub bucket_size: usize,
    /// Max fragment charge to probe (deconvolution loop bound).
    pub max_fragment_charge: i32,
    /// How a shared observed peak's intensity is apportioned among co-isolated,
    /// co-eluting candidates that all match it (see [`PeakClaim`]).
    pub peak_claim: PeakClaim,
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
}
impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            scan_window_mode: ScanWindowMode::Fixed, // MVP-conservative
            scan_scale: 2.2,
            fixed_scan_window: 3,
            frag_tol_ppm: 20.0,
            prec_tol_ppm: 20.0,
            presence_min_matched: 3,
            presence_min_fragments: 3,
            presence_min_coelution: 2,
            // Validated defensible regime (holds a ~valid target-decoy FDR):
            // strict spectral gate on frag_corr. Loosening these raises raw
            // counts but inflates FDR on chimeric DIA data (see COMPARISON.md).
            min_frag_corr: 0.5,
            min_matched_fraction: 0.0,
            apex_top_fragments: 0, // superseded by apex_count_tol; kept for compat
            apex_rt_prior_s: 0.0,  // RT prior off by default
            apex_count_tol: 1,     // fragment-count apex with 1-fragment slack
            apex_count_window: 1,  // no rolling smoothing by default (opt-in; window 5
                                   // cuts AIF apex misassignment, median |dRT| 131s->9s)
            emit_window_grid: true, // zero-filled window-grid chromatograms
            k_select: 50,
            bucket_size: 8192,
            max_fragment_charge: 1,
            peak_claim: PeakClaim::None,
            emit_contested_features: false,
            peak_claim_margin: 2.0,
            matcher: MatcherKind::Fragindex,
            min_coelution_run: 0, // disabled; scan_window floor still applies
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FeaturesConfig {
    pub set: FeatureSet,
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
}
impl Default for FeaturesConfig {
    fn default() -> Self {
        Self {
            set: t(),
            coelution_corr_threshold: 0.9,
            prec_tol_ppm: 20.0,
            bound_features: true,
            bound_peak_fraction: 1.0 / 3.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CompeteConfig {
    /// competition grouping: `precursor` groups target/decoy pairs and charge
    /// variants of one peptide; `apex` also groups by rounded apex RT;
    /// `peptidoform_charge` keeps each peptidoform+charge as its own group
    /// (precursor-level, as DIA-NN/Spectronaut report), so sibling charges of one
    /// peptide are not collapsed.
    pub group_by: CompeteGroupBy,
    pub apex_rt_tolerance_s: f64,
}
impl Default for CompeteConfig {
    fn default() -> Self {
        Self {
            group_by: CompeteGroupBy::Precursor,
            apex_rt_tolerance_s: 5.0,
        }
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RollupMethod {
    /// Sum of the top-N most abundant peptides (single-run default).
    TopNSum,
    /// Sum of all group peptides.
    Sum,
}
impl Default for RollupMethod {
    fn default() -> Self {
        RollupMethod::TopNSum
    }
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
    /// each candidate's apex. Robust to a single window being distorted, and
    /// identical across runs so it preserves fold changes.
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
    /// native rescorer. Use in production so a broken rescorer never passes
    /// silently as native scores.
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
            strict: false,
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
    pub threads: Option<usize>,
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
}
impl Default for Config {
    fn default() -> Self {
        Self {
            rng_seed: 0,
            threads: None,
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
        }
    }
}

impl Config {
    /// Parse from a JSON string, rejecting unknown keys, then validate.
    pub fn from_json(s: &str) -> Result<Self, crate::error::ConfigError> {
        let c: Config = serde_json::from_str(s)
            .map_err(|e| crate::error::ConfigError::Parse(e.to_string()))?;
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
        // Warn (not fail) when a declared-but-unimplemented knob is set away from
        // its default: it silently has no effect, which otherwise misleads tuning.
        let d = Self::default();
        let mut dead = Vec::new();
        if self.search_seed.precursor_tol_ppm != d.search_seed.precursor_tol_ppm {
            dead.push("search_seed.precursor_tol_ppm");
        }
        if self.rt_im_train.tolerance_regime != d.rt_im_train.tolerance_regime {
            dead.push("rt_im_train.tolerance_regime");
        }
        if self.extract.k_select != d.extract.k_select {
            dead.push("extract.k_select");
        }
        if self.extract.max_fragment_charge != d.extract.max_fragment_charge {
            dead.push("extract.max_fragment_charge");
        }
        if self.extract.scan_scale != d.extract.scan_scale {
            dead.push("extract.scan_scale");
        }
        if self.digest.decoy.source != d.digest.decoy.source {
            dead.push("digest.decoy.source");
        }
        if self.digest.decoy.ratio != d.digest.decoy.ratio {
            dead.push("digest.decoy.ratio");
        }
        for k in &dead {
            eprintln!(
                "config warning: `{k}` is set but not implemented in the engine; it has no effect"
            );
        }
        Ok(())
    }

    /// Apply a named tuning profile on top of the current config. `dia` is the
    /// validated DIA preset (Extended features + rolling-window apex + RT prior);
    /// the other extraction defaults (emit_window_grid, reverse decoys,
    /// min_frag_corr) are already the good values. Lets one command reach a
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
        assert_eq!(back.rt_im_train.tolerance_regime, ToleranceRegime::Fixed);
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
    }
}
