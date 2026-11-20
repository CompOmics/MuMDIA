//! Candidate rejection reason codes for the candidate-audit table.
//!
//! Sensitivity program (docs/20_sensitivity_and_quantification_playbook.md,
//! backlog P0.3). Each variant names the pipeline stage at which a candidate
//! precursor is lost. The audit (`stages/audit.rs`) tests the stages in pipeline
//! order and records the first that applies, so the aggregate answers "where was each
//! missing precursor first lost?" without conflating later stages; `compete` writes
//! the two `OUTCOMPETED_*` codes to its own audit table.
//!
//! [`RejectionReason::code`] is the stable SCREAMING_SNAKE_CASE string written to
//! Parquet/JSON and matches the reason strings in the specification exactly (e.g.
//! `NO_FRAGMENT_TRACES`).
//!
//! `DID_NOT_SURVIVE_EXTRACTION` was `NO_PEAK_GROUP` until docs/29 #16. The audit
//! assigns it to every candidate that has traces in no extracted row, and `extract`
//! does not write the per-candidate table that would separate presence failures,
//! matched-fraction failures and gate rejections, so the old name claimed a cause the
//! audit cannot see. The name now says what is known.

/// Earliest-loss category for a candidate, in pipeline order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RejectionReason {
    // --- search space (Stage A) ---
    NoValidFragments,
    WrongIsolationWindow,
    // --- candidate generation / pruning (Stage B) ---
    RtPruned,
    // --- extraction + peak formation (Stages C, D) ---
    NoFragmentTraces,
    /// Extracted no accepted row, for any extraction-side reason; see the module
    /// documentation for why this is not called a peak-group failure.
    DidNotSurviveExtraction,
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
    /// Stable string code (matches the specification's reason strings), written to
    /// Parquet columns and JSON.
    pub fn code(&self) -> &'static str {
        use RejectionReason::*;
        match self {
            NoValidFragments => "NO_VALID_FRAGMENTS",
            WrongIsolationWindow => "WRONG_ISOLATION_WINDOW",
            RtPruned => "RT_PRUNED",
            NoFragmentTraces => "NO_FRAGMENT_TRACES",
            DidNotSurviveExtraction => "DID_NOT_SURVIVE_EXTRACTION",
            PeakNotSelected => "PEAK_NOT_SELECTED",
            OutcompetedByTarget => "OUTCOMPETED_BY_TARGET",
            OutcompetedByDecoy => "OUTCOMPETED_BY_DECOY",
            FailedPrecursorFdr => "FAILED_PRECURSOR_FDR",
            FailedPeptideFdr => "FAILED_PEPTIDE_FDR",
            RemovedDuringReporting => "REMOVED_DURING_REPORTING",
            Reported => "REPORTED",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RejectionReason::*;

    #[test]
    fn codes_match_spec_strings() {
        assert_eq!(DidNotSurviveExtraction.code(), "DID_NOT_SURVIVE_EXTRACTION");
        assert_eq!(PeakNotSelected.code(), "PEAK_NOT_SELECTED");
        assert_eq!(OutcompetedByDecoy.code(), "OUTCOMPETED_BY_DECOY");
        assert_eq!(Reported.code(), "REPORTED");
    }
}
