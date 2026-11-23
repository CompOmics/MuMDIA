//! Pipeline stages. Each is an independent subcommand reading path-addressable
//! inputs and writing declared Parquet + a report (docs/03_io_layer.md).

pub mod audit;
pub mod chain;
pub mod compete;
pub mod convert;
pub mod digest;
pub mod extract;
pub mod features;
pub mod library_build;
pub mod localise;
pub mod mobilogram;
pub mod ms1cal;
pub mod peptidoforms;
pub mod pool;
pub mod predict_frag;
pub mod prescan;
pub mod prescreen;
pub mod q1cal;
pub mod q1consist;
pub mod q1dec;
pub mod q1frag;
pub mod q1quant;
pub mod q1score;
pub mod q1trace;
pub mod quant;
pub mod quant_norm;
pub mod quant_pattern;
pub mod quant_stable;
pub mod report;
pub mod rescore;
pub mod retrace;
pub mod rt_bootstrap;
pub mod rt_health;
pub mod rt_im_train;
pub mod run;
pub mod run_experiment;
pub mod run_groups;
pub mod search_seed;
pub mod second_pass;
pub mod seed_pool;
pub mod shadow;
pub mod sub_library;

/// Delete an intermediate table and its companions (`<path>.report.json`,
/// `<path>.schema.json`), those that exist, and return the files and bytes removed. A file
/// that exists and cannot be removed is an error naming it ("deleting the `what` <file>"),
/// so a run never reports space it did not free. The one helper for intermediates a run
/// deletes once their only reader is done: a grouped run's band tables
/// (`groups.delete_band_intermediates`) and retrace's centroid chromatograms. A grouped run
/// also takes away, with it, the pooled tables an earlier run left that this one does not
/// write.
pub(crate) fn delete_intermediate(path: &str, what: &str) -> anyhow::Result<(u64, u64)> {
    use anyhow::Context as _;
    let (mut files, mut bytes) = (0u64, 0u64);
    for f in [
        path.to_string(),
        format!("{path}.report.json"),
        format!("{path}.schema.json"),
    ] {
        if let Ok(meta) = std::fs::metadata(&f) {
            std::fs::remove_file(&f).with_context(|| format!("deleting the {what} {f}"))?;
            files += 1;
            bytes += meta.len();
        }
    }
    Ok((files, bytes))
}
