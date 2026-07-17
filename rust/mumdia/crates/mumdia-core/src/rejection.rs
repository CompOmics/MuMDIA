//! Candidate rejection reason codes for the candidate-audit table.
//!
//! Sensitivity program (spec `01_workflow_and_gap_analysis.md` §4,
//! `02_sensitivity_diagnostic_plan.md` §5, backlog P0.3). Each variant names the
//! pipeline stage at which a candidate precursor is lost. A candidate's audit row
//! records the EARLIEST such stage, so the aggregate answers "where was each
//! DIA-NN-only precursor first lost?" without conflating later stages.
//!
//! The serialized spelling is SCREAMING_SNAKE_CASE and matches the reason strings
//! in the specification exactly (e.g. `NO_PEAK_GROUP`). Use [`RejectionReason::code`]
//! for the stable string written to Parquet/JSON (no serde round-trip cost).

use serde::{Deserialize, Serialize};

/// Earliest-loss category for a candidate. Ordered along the pipeline so
/// [`RejectionReason::stage_order`] can pick the earliest when several apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RejectionReason {
    // --- search space (Stage A) ---
    PeptideNotGenerated,
    ModificationNotAllowed,
    ChargeOutOfRange,
    PrecursorMzOutOfRange,
    NoValidFragments,
    WrongIsolationWindow,
    // --- candidate generation / pruning (Stage B) ---
    RtPruned,
    CandidateCapReached,
    // --- extraction + peak formation (Stages C, D) ---
    NoFragmentTraces,
    NoPeakGroup,
    // --- peak / peptide ranking (Stage E) ---
    PeakNotSelected,
    // --- competition (Stage G) ---
    OutcompetedByTarget,
    OutcompetedByDecoy,
    // --- FDR + reporting (Stage H) ---
    FailedPrecursorFdr,
    FailedPeptideFdr,
    RemovedDuringReporting,
    /// Sentinel: not rejected; the candidate reached the final report. Present so
    /// every candidate has exactly one audit row.
    Reported,
}

impl RejectionReason {
    /// Stable string code (matches the specification's reason strings). Used for
    /// Parquet columns and JSON without a serde round-trip.
    pub fn code(&self) -> &'static str {
        use RejectionReason::*;
        match self {
            PeptideNotGenerated => "PEPTIDE_NOT_GENERATED",
            ModificationNotAllowed => "MODIFICATION_NOT_ALLOWED",
            ChargeOutOfRange => "CHARGE_OUT_OF_RANGE",
            PrecursorMzOutOfRange => "PRECURSOR_MZ_OUT_OF_RANGE",
            NoValidFragments => "NO_VALID_FRAGMENTS",
            WrongIsolationWindow => "WRONG_ISOLATION_WINDOW",
            RtPruned => "RT_PRUNED",
            CandidateCapReached => "CANDIDATE_CAP_REACHED",
            NoFragmentTraces => "NO_FRAGMENT_TRACES",
            NoPeakGroup => "NO_PEAK_GROUP",
            PeakNotSelected => "PEAK_NOT_SELECTED",
            OutcompetedByTarget => "OUTCOMPETED_BY_TARGET",
            OutcompetedByDecoy => "OUTCOMPETED_BY_DECOY",
            FailedPrecursorFdr => "FAILED_PRECURSOR_FDR",
            FailedPeptideFdr => "FAILED_PEPTIDE_FDR",
            RemovedDuringReporting => "REMOVED_DURING_REPORTING",
            Reported => "REPORTED",
        }
    }

    /// Position on the identification-loss ladder (0 = earliest stage). The
    /// `Reported` sentinel sorts last. When a candidate could be assigned more than
    /// one reason across stages, keep the one with the smallest `stage_order`.
    pub fn stage_order(&self) -> u8 {
        use RejectionReason::*;
        match self {
            PeptideNotGenerated => 0,
            ModificationNotAllowed => 1,
            ChargeOutOfRange => 2,
            PrecursorMzOutOfRange => 3,
            NoValidFragments => 4,
            WrongIsolationWindow => 5,
            RtPruned => 6,
            CandidateCapReached => 7,
            NoFragmentTraces => 8,
            NoPeakGroup => 9,
            PeakNotSelected => 10,
            OutcompetedByTarget => 11,
            OutcompetedByDecoy => 12,
            FailedPrecursorFdr => 13,
            FailedPeptideFdr => 14,
            RemovedDuringReporting => 15,
            Reported => 255,
        }
    }

    /// True if the candidate was lost (any non-`Reported` reason).
    pub fn is_rejection(&self) -> bool {
        !matches!(self, RejectionReason::Reported)
    }

    /// Keep the earlier of two losses (smaller `stage_order`). `Reported` never
    /// overrides a real rejection.
    pub fn earliest(self, other: RejectionReason) -> RejectionReason {
        if other.stage_order() < self.stage_order() {
            other
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RejectionReason::*;
    use super::*;

    #[test]
    fn codes_match_spec_strings() {
        assert_eq!(NoPeakGroup.code(), "NO_PEAK_GROUP");
        assert_eq!(PeakNotSelected.code(), "PEAK_NOT_SELECTED");
        assert_eq!(OutcompetedByDecoy.code(), "OUTCOMPETED_BY_DECOY");
        assert_eq!(Reported.code(), "REPORTED");
    }

    #[test]
    fn serde_roundtrip_uses_spec_spelling() {
        let j = serde_json::to_string(&FailedPrecursorFdr).unwrap();
        assert_eq!(j, "\"FAILED_PRECURSOR_FDR\"");
        let back: RejectionReason = serde_json::from_str(&j).unwrap();
        assert_eq!(back, FailedPrecursorFdr);
    }

    #[test]
    fn earliest_keeps_smaller_stage() {
        // an extraction loss precedes an FDR loss
        assert_eq!(NoPeakGroup.earliest(FailedPrecursorFdr), NoPeakGroup);
        assert_eq!(FailedPrecursorFdr.earliest(NoPeakGroup), NoPeakGroup);
        // Reported never wins against a real rejection
        assert_eq!(Reported.earliest(NoFragmentTraces), NoFragmentTraces);
        assert_eq!(NoFragmentTraces.earliest(Reported), NoFragmentTraces);
    }

    #[test]
    fn is_rejection_flags_only_losses() {
        assert!(NoPeakGroup.is_rejection());
        assert!(!Reported.is_rejection());
    }

    #[test]
    fn stage_order_is_monotone_ladder() {
        // ladder ordering across the major stages
        assert!(PeptideNotGenerated.stage_order() < NoFragmentTraces.stage_order());
        assert!(NoFragmentTraces.stage_order() < NoPeakGroup.stage_order());
        assert!(NoPeakGroup.stage_order() < PeakNotSelected.stage_order());
        assert!(PeakNotSelected.stage_order() < OutcompetedByTarget.stage_order());
        assert!(OutcompetedByTarget.stage_order() < FailedPrecursorFdr.stage_order());
        assert!(FailedPrecursorFdr.stage_order() < RemovedDuringReporting.stage_order());
        assert!(RemovedDuringReporting.stage_order() < Reported.stage_order());
    }
}
