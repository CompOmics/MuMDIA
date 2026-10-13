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
    pub const SPECTRA_MS1: (&str, u32) = ("spectra_ms1", 1);
    pub const SPECTRA_MS2: (&str, u32) = ("spectra_ms2", 1);
    /// v2, written only for a source that carries ion mobility (the native diaPASEF
    /// reader): a per-peak 1/K0 list `im` parallel to `mz`. A 3D source writes v1.
    pub const SPECTRA_MS1_V2: (&str, u32) = ("spectra_ms1", 2);
    /// v2, as [`SPECTRA_MS1_V2`], plus the window's 1/K0 bounds `window_im_lower` and
    /// `window_im_upper`. A 3D source writes v1. Readers select columns by name, so both
    /// versions load everywhere.
    pub const SPECTRA_MS2_V2: (&str, u32) = ("spectra_ms2", 2);
    pub const ISOLATION_WINDOWS: (&str, u32) = ("isolation_windows", 1);
    /// v2, written only for a source with ion mobility: each window's 1/K0 bounds
    /// `im_lower` and `im_upper`, so slots sharing an m/z range are separate rows. A 3D
    /// source writes v1.
    pub const ISOLATION_WINDOWS_V2: (&str, u32) = ("isolation_windows", 2);
    pub const MS2_TO_MS1: (&str, u32) = ("ms2_to_ms1", 1);
    pub const PEPTIDES: (&str, u32) = ("peptides", 1);
    pub const PEPTIDOFORMS: (&str, u32) = ("peptidoforms", 1);
    pub const FRAGMENT_LIBRARY_PRECURSORS: (&str, u32) = ("fragment_library_precursors", 1);
    /// v2: v1 plus a nullable Float64 `predicted_im` (1/K0, V s cm^-2). Written only when the
    /// library has an ion-mobility source: `predict_frag.im_predictor` predicted it, or an
    /// imported library carried at least one finite, positive value. The column's presence
    /// therefore means "this library has 1/K0"; every other library is v1, byte for byte.
    /// Readers select columns by name, so both versions load everywhere.
    pub const FRAGMENT_LIBRARY_PRECURSORS_V2: (&str, u32) = ("fragment_library_precursors", 2);
    pub const FRAGMENT_LIBRARY_FRAGMENTS: (&str, u32) = ("fragment_library_fragments", 1);
    pub const PRESCAN_SURVIVORS: (&str, u32) = ("prescan_survivors", 1);
    pub const PRESCREEN_SCORES: (&str, u32) = ("prescreen_scores", 1);
    pub const SEED_PSMS: (&str, u32) = ("seed_psms", 1);
    /// v2: v1 plus a nullable Float64 `observed_im`, the intensity-weighted median 1/K0 of a
    /// confident target's matched fragment peaks. Written only when the searched spectra
    /// carry per-peak 1/K0; every other seed is v1, byte for byte.
    pub const SEED_PSMS_V2: (&str, u32) = ("seed_psms", 2);
    /// The confident targets of a bootstrap search, in the seed's column shape, that the
    /// multi-head calibration is refitted on under `rt_im_train.multihead_anchors = search`
    /// (`<run>/rt_bootstrap/search_anchors.parquet`, `stages::rt_bootstrap`).
    pub const RT_BOOTSTRAP_ANCHORS: (&str, u32) = ("rt_bootstrap_anchors", 1);
    pub const RUN_WINDOWS: (&str, u32) = ("run_windows", 1);
    pub const PSMS_EXTRACTED: (&str, u32) = ("psms_extracted", 2);
    /// v1: every row stores its whole `rt` axis and its whole `intensity` trace.
    pub const CHROMATOGRAMS: (&str, u32) = ("chromatograms", 1);
    /// v2, the default (`extract.chromatogram_schema = 2`): the axis once per candidate
    /// per row group (`rt_axis`), each trace trimmed to its nonzero run (`intensity_trimmed`),
    /// and `trace_offset` / `trace_len` to rebuild it (`mumdia::chromatograms`). Every reader
    /// in the engine accepts both; the renamed lists make a v1-only reader fail on v2.
    pub const CHROMATOGRAMS_V2: (&str, u32) = ("chromatograms", 2);
    /// v2: the feature columns are Float32 except the few `F64_FEATURE_COLUMNS` of
    /// `stages/features.rs`; v1 stored every feature as Float64. Every reader accepts both.
    pub const FEATURES: (&str, u32) = ("features", 2);
    /// v4: the feature columns are stored as in `features` v2; v3 stored every feature as
    /// Float64. Every reader accepts both, and compete given a v1 features table writes v4.
    pub const PSMS_COMPETED: (&str, u32) = ("psms_competed", 4);
    pub const PSMS_SCORED: (&str, u32) = ("psms_scored", 4);
    pub const PEPTIDE_QUANT: (&str, u32) = ("peptide_quant", 2);
    pub const PROTEIN_GROUP_QUANT: (&str, u32) = ("protein_group_quant", 2);
    pub const FRAGMENT_QUANT: (&str, u32) = ("fragment_quant", 1);
    /// Cross-run MaxLFQ table, written only by `run-experiment` and `quant-lfq`.
    pub const LFQ_MAXLFQ: (&str, u32) = ("lfq_maxlfq", 1);
    /// The candidates a grouped run's pool dropped from each band (`band`,
    /// `candidate_id`), written by a grouped run under `groups.pool_chromatograms = false`,
    /// the default.
    pub const OVERLAP_LOSERS: (&str, u32) = ("overlap_losers", 1);

    /// The `spectra_ms1` schema a conversion writes: [`SPECTRA_MS1_V2`] when the source
    /// carries ion mobility, else [`SPECTRA_MS1`].
    pub fn spectra_ms1(has_im: bool) -> (&'static str, u32) {
        if has_im {
            SPECTRA_MS1_V2
        } else {
            SPECTRA_MS1
        }
    }

    /// The `spectra_ms2` schema a conversion writes: [`SPECTRA_MS2_V2`] when the source
    /// carries ion mobility, else [`SPECTRA_MS2`].
    pub fn spectra_ms2(has_im: bool) -> (&'static str, u32) {
        if has_im {
            SPECTRA_MS2_V2
        } else {
            SPECTRA_MS2
        }
    }

    /// The `isolation_windows` schema a conversion writes: [`ISOLATION_WINDOWS_V2`] when
    /// the source carries ion mobility, else [`ISOLATION_WINDOWS`].
    pub fn isolation_windows(has_im: bool) -> (&'static str, u32) {
        if has_im {
            ISOLATION_WINDOWS_V2
        } else {
            ISOLATION_WINDOWS
        }
    }

    /// The chromatogram schema for `extract.chromatogram_schema`: [`CHROMATOGRAMS_V2`] for 2,
    /// else [`CHROMATOGRAMS`] (config validation admits only 1 and 2).
    pub fn chromatograms(schema: u32) -> (&'static str, u32) {
        if schema == CHROMATOGRAMS_V2.1 {
            CHROMATOGRAMS_V2
        } else {
            CHROMATOGRAMS
        }
    }

    /// The seed schema a search writes: [`SEED_PSMS_V2`] when the searched spectra carry
    /// per-peak 1/K0, else [`SEED_PSMS`].
    pub fn seed_psms(has_im: bool) -> (&'static str, u32) {
        if has_im {
            SEED_PSMS_V2
        } else {
            SEED_PSMS
        }
    }

    /// The precursor-library schema a table records: [`FRAGMENT_LIBRARY_PRECURSORS_V2`] when
    /// it carries `predicted_im`, else [`FRAGMENT_LIBRARY_PRECURSORS`].
    pub fn fragment_library_precursors(has_im: bool) -> (&'static str, u32) {
        if has_im {
            FRAGMENT_LIBRARY_PRECURSORS_V2
        } else {
            FRAGMENT_LIBRARY_PRECURSORS
        }
    }
}

#[cfg(test)]
mod tests {
    use super::artifact::*;

    /// A 3D conversion records the versions it always has; only a source with ion
    /// mobility moves the three selectors, and the name never changes.
    #[test]
    fn spectra_selectors_keep_3d_at_v1() {
        assert_eq!(spectra_ms1(false), ("spectra_ms1", 1));
        assert_eq!(spectra_ms2(false), ("spectra_ms2", 1));
        assert_eq!(isolation_windows(false), ("isolation_windows", 1));
        assert_eq!(spectra_ms1(true), ("spectra_ms1", 2));
        assert_eq!(spectra_ms2(true), ("spectra_ms2", 2));
        assert_eq!(isolation_windows(true), ("isolation_windows", 2));
        assert_eq!(spectra_ms1(false), SPECTRA_MS1);
        assert_eq!(spectra_ms2(false), SPECTRA_MS2);
        assert_eq!(isolation_windows(false), ISOLATION_WINDOWS);
        assert_eq!(seed_psms(false), SEED_PSMS);
        assert_eq!(seed_psms(false), ("seed_psms", 1));
        assert_eq!(seed_psms(true), ("seed_psms", 2));
    }
}
