//! Frozen artifact schema identifiers (docs/03_io_layer.md). Each artifact
//! carries a logical schema name and version, recorded in the artifact's
//! `report.json` and in the run manifest.
//!
//! What this is NOT, despite the previous wording here: input validation. No code
//! anywhere reads a `schema_version` back and compares it -- all uses are writes. The
//! version is provenance, so a reader of a finished artifact can tell which shape it
//! is; a stage handed an incompatible artifact still fails on the missing or retyped
//! column, not on the version. Claiming otherwise mattered because `CONTRIBUTING.md`
//! stated a compatibility policy that nothing enforced. Making the check real is
//! worthwhile and is a separate change; until then the honest statement is this one
//! and a model is never applied under a mismatched schema.

/// (logical name, schema version) for every MVP artifact.
pub mod artifact {
    // v2: ion mobility. `spectra_ms2` gains `window_im_lower/upper` and
    // `isolation_windows` gains `im_lower/upper` (nullable 1/K0, null for 3D input);
    // `spectra_ms1/ms2` gain a per-peak `im` list only when the source has mobility.
    // v1 artifacts still load: readers treat an absent column as "no mobility".
    // v3: a per-peak mobility width list `im_width` (1/K0), only under
    // `convert.tdf_im_width`; v1/v2 artifacts read as "no width".
    pub const SPECTRA_MS1: (&str, u32) = ("spectra_ms1", 3);
    pub const SPECTRA_MS2: (&str, u32) = ("spectra_ms2", 3);
    pub const ISOLATION_WINDOWS: (&str, u32) = ("isolation_windows", 2);
    pub const MS2_TO_MS1: (&str, u32) = ("ms2_to_ms1", 1);
    pub const PEPTIDES: (&str, u32) = ("peptides", 1);
    pub const PEPTIDOFORMS: (&str, u32) = ("peptidoforms", 1);
    // v2: nullable `predicted_im` (1/K0, V s cm^-2), always written by predict-frag and
    // the DIA-NN importer. v1 libraries still load: every reader selects columns by name,
    // and rt-im-train treats an absent column as "no library IM".
    pub const FRAGMENT_LIBRARY_PRECURSORS: (&str, u32) = ("fragment_library_precursors", 2);
    pub const FRAGMENT_LIBRARY_FRAGMENTS: (&str, u32) = ("fragment_library_fragments", 1);
    pub const PRESCAN_SURVIVORS: (&str, u32) = ("prescan_survivors", 1);
    // v2: nullable `observed_im`, the intensity-weighted median 1/K0 of a confident target's
    // matched fragment peaks (null elsewhere and on 3D data).
    pub const SEED_PSMS: (&str, u32) = ("seed_psms", 2);
    pub const RUN_WINDOWS: (&str, u32) = ("run_windows", 1);
    // v3: `apex_im` is filled on 4D data (the intensity-weighted median 1/K0 of the
    // candidate's fragment peaks in its apex scan); null on 3D data, as before.
    // v4: nullable `apex_im_mad`, `ms1_apex_im` and `im_pred_cal` (null on 3D data or
    // without IM calibration).
    // v5: nullable IM peak-shape columns `apex_im_width`, `apex_im_width_mad`,
    // `apex_im_overlap`, `ms1_im_width`, `ms1_frag_overlap` (null without spectra v3
    // widths).
    pub const PSMS_EXTRACTED: (&str, u32) = ("psms_extracted", 5);
    // v2: on 4D data a per-point 1/K0 list `im`, parallel to `intensity`; a 3D run writes
    // exactly the v1 columns.
    pub const CHROMATOGRAMS: (&str, u32) = ("chromatograms", 2);
    pub const FEATURES: (&str, u32) = ("features", 1);
    pub const PSMS_COMPETED: (&str, u32) = ("psms_competed", 3);
    pub const PSMS_SCORED: (&str, u32) = ("psms_scored", 4);
    pub const PEPTIDE_QUANT: (&str, u32) = ("peptide_quant", 2);
    pub const PROTEIN_GROUP_QUANT: (&str, u32) = ("protein_group_quant", 2);
    pub const FRAGMENT_QUANT: (&str, u32) = ("fragment_quant", 3);
    /// Cross-run MaxLFQ table, written only by `run-experiment` and `quant-lfq`.
    pub const LFQ_MAXLFQ: (&str, u32) = ("lfq_maxlfq", 1);
}
