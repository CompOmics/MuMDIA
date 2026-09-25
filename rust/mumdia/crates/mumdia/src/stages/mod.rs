//! Pipeline stages. Each is an independent subcommand reading path-addressable
//! inputs and writing declared Parquet + a report (docs/03_io_layer.md).

pub mod align;
pub mod audit;
pub mod compete;
pub mod convert;
pub mod digest;
pub mod extract;
pub mod features;
pub mod peptidoforms;
pub mod pool;
pub mod predict_frag;
pub mod prescan;
pub mod quant;
pub mod report;
pub mod rescore;
pub mod rt_im_train;
pub mod run;
pub mod run_experiment;
pub mod run_groups;
pub mod search_seed;
pub mod seed_pool;
pub mod sub_library;

/// How the full scans of the widest artifacts are read: rescore's feature stream
/// (`rescore::for_each_feature_batch`) and compete's pass-through copy
/// (`compete::copy_kept_rows`), which both read ~390 columns of every row of a competed or
/// features table. `MUMDIA_WIDE_SCAN` chooses; the batches, and so every output, are the
/// same under all three (`every_read_mode_streams_the_same_feature_rows`,
/// `coalesced_reads_yield_the_plain_readers_batches`).
///
/// The plain reader is the default because it is the only one measured not to lose: from
/// the page cache on the HYE competed table (879,018 x 387, 131,072-row groups) the feature
/// stream took 1.54 s plain at 16,384-row batches, 1.58 s plain at one row group a batch,
/// 2.50 s coalesced. The other two are for the seek-bound case, a spinning array where one
/// reader of the immunopeptidomics competed tables measured 44 MB/s against a 133 MB/s
/// sequential ceiling, and neither has been measured there yet (the 2026-09-25 survey's
/// iostat check). Promote one once it has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WideScan {
    /// The plain reader at the scans' own batch sizes, with its automatic parallel decode
    /// over column groups (docs/03 "Parallel decode"). Unset, `plain`, or any other value.
    Plain,
    /// `rowgroup`: the plain reader, with rescore's feature stream decoding one row group
    /// a batch, so the arrow reader sweeps each column chunk of the group before the next
    /// (step 1 of R1 in the survey). Compete's copy keeps its batches, which decide the
    /// competed table's page boundaries.
    RowGroup,
    /// `coalesced`: [`mumdia_io::table::ScanOptions::coalesced`] (docs/03 "Sequential
    /// row-group reads"): each row group's projected column chunks are read with one
    /// sequential read and the pages are served from memory, with the next row group
    /// prefetched. One reader decodes the scan, and the cache holds two or three row groups
    /// of the projection (about 0.9 GB on a 131,072-row competed group).
    Coalesced,
}

impl WideScan {
    /// The mode `MUMDIA_WIDE_SCAN` names.
    pub(crate) fn from_env() -> WideScan {
        WideScan::parse(std::env::var("MUMDIA_WIDE_SCAN").ok().as_deref())
    }

    fn parse(v: Option<&str>) -> WideScan {
        match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("coalesced") => WideScan::Coalesced,
            Some("rowgroup" | "row_group" | "row-group") => WideScan::RowGroup,
            _ => WideScan::Plain,
        }
    }

    /// The read options of the mode.
    pub(crate) fn options(self) -> mumdia_io::table::ScanOptions {
        match self {
            WideScan::Plain | WideScan::RowGroup => mumdia_io::table::ScanOptions::default(),
            WideScan::Coalesced => mumdia_io::table::ScanOptions::coalesced(),
        }
    }
}

/// Read options for the wide scans under `MUMDIA_WIDE_SCAN` ([`WideScan`]).
pub(crate) fn wide_scan_options() -> mumdia_io::table::ScanOptions {
    WideScan::from_env().options()
}

#[cfg(test)]
mod tests {
    use super::WideScan;

    #[test]
    fn the_wide_scan_is_plain_unless_asked_otherwise() {
        for (v, want) in [
            (None, WideScan::Plain),
            (Some(""), WideScan::Plain),
            (Some("plain"), WideScan::Plain),
            (Some("something else"), WideScan::Plain),
            (Some("rowgroup"), WideScan::RowGroup),
            (Some(" Row_Group "), WideScan::RowGroup),
            (Some("coalesced"), WideScan::Coalesced),
            (Some("COALESCED"), WideScan::Coalesced),
        ] {
            assert_eq!(WideScan::parse(v), want, "{v:?}");
        }
        assert!(WideScan::Plain.options().coalesce.is_none());
        assert!(WideScan::RowGroup.options().coalesce.is_none());
        assert!(WideScan::Coalesced.options().coalesce.is_some());
    }
}
