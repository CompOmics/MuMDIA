//! mumdia-io: on-disk contracts (Parquet, JSON), content hashing, logging, and
//! the `inspect` helper (docs/03_io_layer.md).

pub mod hash;
pub mod json;
pub mod report;
pub mod table;

use anyhow::Result;
use mumdia_core::manifest::ArtifactRecord;

/// Initialize tracing once, honoring `RUST_LOG` (default `info`).
pub fn init_logging() {
    init_logging_level(None)
}

/// Initialize tracing with an explicit level, falling back to `RUST_LOG` and then
/// to `info`.
///
/// `RUST_LOG` was the only way to change verbosity, which is not discoverable
/// from `--help` and is awkward on Windows. An explicit level wins over it, so
/// `--log-level` and `-v/-q` behave as the flags a user expects; `RUST_LOG` still
/// works, and still offers per-module filtering that a single level cannot.
/// Diagnostics go to STDERR, and ANSI colour is emitted only when stderr is a
/// terminal.
///
/// Both were previously wrong in the same direction: every log line went to stdout
/// with escape sequences attached whatever the destination. So `mumdia run > out.txt`
/// interleaved the diagnostic log with the result summary in one file that no
/// redirection could separate, the file contained
/// `^[[2m2026-08-28T07:17:23Z^[[0m ^[[32m INFO^[[0m ...`, and a parser could not use a
/// `key=value` regex because tracing wraps each field name and value individually.
/// `mumdia doctor 2>/dev/null` still printed its whole report.
///
/// Results stay on stdout (the `println!` summaries, `inspect`, the report paths), so
/// `2>/dev/null` now yields exactly the answer and `1>/dev/null` exactly the log.
pub fn init_logging_level(level: Option<&str>) {
    use std::io::IsTerminal as _;
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = match level {
        Some(l) => EnvFilter::new(l),
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
    };
    let _ = fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .try_init();
}

/// Refuse to write an output over one of this stage's own inputs.
///
/// There was no path-equality check anywhere in the workspace, so
/// `mumdia prescan --lib-precursors lib/x.parquet --out lib/x.parquet` replaced a full
/// precursor library with a two-column survivors table and exited 0 -- the library had
/// already been read into memory, so nothing failed. `align` has the same shape, and in
/// FASTA mode `run --out-dir lib` truncated the precursor library before the digest had
/// produced a row. Comparing canonicalised paths catches the aliases (`./lib/x.parquet`,
/// a symlink, a different drive-letter case on Windows) that a string compare misses; a
/// path that cannot be canonicalised does not exist yet, so it cannot be an input.
pub fn refuse_output_over_input(output: &str, inputs: &[(&str, &str)]) -> Result<()> {
    let Ok(out) = std::fs::canonicalize(output) else {
        return Ok(());
    };
    for (flag, path) in inputs {
        if let Ok(inp) = std::fs::canonicalize(path) {
            if inp == out {
                anyhow::bail!(
                    "refusing to write the output over its own input: {output} is also \
                     {flag}. The input is read before the output is written, so this would \
                     silently replace it with a different table. Write to a new path."
                );
            }
        }
    }
    Ok(())
}

/// Build an [`ArtifactRecord`] by hashing the written file (docs/03_io_layer.md).
pub fn record_artifact(
    logical_name: &str,
    schema: (&str, u32),
    path: &str,
    rows: u64,
    stage: &str,
    config_hash: &str,
) -> Result<ArtifactRecord> {
    Ok(ArtifactRecord {
        logical_name: logical_name.to_string(),
        path: path.to_string(),
        format: "parquet".to_string(),
        schema_name: schema.0.to_string(),
        schema_version: schema.1,
        rows,
        content_hash: hash::blake3_file(path)?,
        producing_stage: stage.to_string(),
        config_hash: config_hash.to_string(),
    })
}

/// Human-readable schema + head sample + row count for any Parquet artifact
/// (`mumdia inspect`, docs/03_io_layer.md).
pub fn inspect(path: &str) -> Result<String> {
    let t = table::Table::read(path)?;
    let mut s = String::new();
    s.push_str(&format!("artifact: {path}\n"));
    s.push_str(&format!("rows: {}\n", t.nrows));
    s.push_str("schema:\n");
    for f in t.schema.fields() {
        s.push_str(&format!(
            "  {}: {:?}{}\n",
            f.name(),
            f.data_type(),
            if f.is_nullable() { " (nullable)" } else { "" }
        ));
    }
    if let Some(first) = t.batches.first() {
        let head_rows = first.num_rows().min(10);
        let head = first.slice(0, head_rows);
        if let Ok(p) = arrow::util::pretty::pretty_format_batches(&[head]) {
            s.push_str("head:\n");
            s.push_str(&p.to_string());
            s.push('\n');
        }
    }
    Ok(s)
}
