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

/// Read options for the full scans of the widest artifacts: rescore's feature stream
/// (`rescore::for_each_feature_batch`) and compete's pass-through copy
/// (`compete::copy_kept_rows`), which both read ~390 columns of every row of a competed or
/// features table.
///
/// Coalesced row-group reads ([`mumdia_io::table::ScanOptions::coalesced`], docs/03
/// "Sequential row-group reads") by default: each row group's projected column chunks are
/// read with one sequential read and the pages are served from memory, where the plain
/// reader seeks once per page and covers a row group in several strided passes. One reader
/// decodes the scan, so the disk sees one forward sweep with the next row group prefetched.
/// The batches are the plain reader's batches exactly, so nothing downstream changes.
///
/// `MUMDIA_WIDE_SCAN=plain` restores the plain reader and with it the automatic parallel
/// decode over column groups (docs/03 "Parallel decode"). That is the faster choice when the
/// table is in the page cache or on an SSD and the decode, not the read, is the limit.
pub(crate) fn wide_scan_options() -> mumdia_io::table::ScanOptions {
    match std::env::var("MUMDIA_WIDE_SCAN") {
        Ok(v) if v.trim().eq_ignore_ascii_case("plain") => mumdia_io::table::ScanOptions::default(),
        _ => mumdia_io::table::ScanOptions::coalesced(),
    }
}
