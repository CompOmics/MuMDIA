//! Frozen artifact schema identifiers (PLAN.md Section 3.3, 3.5). Each artifact
//! carries a logical schema name and version so a stage can validate its inputs
//! and a model is never applied under a mismatched schema.

/// (logical name, schema version) for every MVP artifact.
pub mod artifact {
    pub const SPECTRA_MS1: (&str, u32) = ("spectra_ms1", 1);
    pub const SPECTRA_MS2: (&str, u32) = ("spectra_ms2", 1);
    pub const ISOLATION_WINDOWS: (&str, u32) = ("isolation_windows", 1);
    pub const MS2_TO_MS1: (&str, u32) = ("ms2_to_ms1", 1);
    pub const PEPTIDES: (&str, u32) = ("peptides", 1);
    pub const PEPTIDOFORMS: (&str, u32) = ("peptidoforms", 1);
    pub const FRAGMENT_LIBRARY_PRECURSORS: (&str, u32) = ("fragment_library_precursors", 1);
    pub const FRAGMENT_LIBRARY_FRAGMENTS: (&str, u32) = ("fragment_library_fragments", 1);
    pub const SEED_PSMS: (&str, u32) = ("seed_psms", 1);
    pub const RUN_WINDOWS: (&str, u32) = ("run_windows", 1);
    pub const PSMS_EXTRACTED: (&str, u32) = ("psms_extracted", 1);
    pub const CHROMATOGRAMS: (&str, u32) = ("chromatograms", 1);
    pub const FEATURES: (&str, u32) = ("features", 1);
    pub const PSMS_COMPETED: (&str, u32) = ("psms_competed", 1);
    pub const PSMS_SCORED: (&str, u32) = ("psms_scored", 1);
    pub const PEPTIDE_QUANT: (&str, u32) = ("peptide_quant", 1);
    pub const PROTEIN_GROUP_QUANT: (&str, u32) = ("protein_group_quant", 1);
}
