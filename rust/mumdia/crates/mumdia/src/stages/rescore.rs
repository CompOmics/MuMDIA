//! Stage F `mumdia rescore` (docs/11_compete_rescore_fdr.md): rescore competed
//! PSMs across the experiment and compute native target-decoy q-values at PSM and
//! peptide level. MVP default is the native semi-supervised rescorer; Mokapot /
//! percolator.exe are optional strategies over the same PIN/feature contract.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{anyhow, Context as _, Result};
use arrow::array::{Array, Float64Array};
use mumdia_core::config::{FeaturePreset, RescoreConfig, RescorerKind};
use mumdia_core::schema::artifact;
use mumdia_io::report::{ArtifactReport, Written};
use mumdia_io::table::{write_table, Col, TableFile};
use rayon::prelude::*;
use serde_json::json;
use tracing::{info, warn};

use crate::fdr::{entrapment_q, target_decoy_q_by, target_decoy_q_split};
use crate::rescoring::{percolator_lite, FeatureMatrix, RescoreInput};
use crate::stages::features::FeatureSchema;

/// Fewest rows per decoded batch while streaming the ~390 feature columns of a competed
/// table (~16k rows x 387 f64 is about 50 MB per batch).
const FEATURE_BATCH_ROWS: usize = 1 << 14;

/// Most rows per decoded batch of that stream: the competed row-group cap
/// (`compete::COMPETED_ROW_GROUP_ROWS`), ~400 MB of decoded f64 at 387 features.
const FEATURE_BATCH_ROWS_MAX: usize = 1 << 17;

/// Rows per decoded batch of the feature stream over `t` under the read options `scan`.
///
/// Under the plain reader (`MUMDIA_WIDE_SCAN=plain`) a batch is one row group: the largest
/// row group of `t`, clamped to `[FEATURE_BATCH_ROWS, FEATURE_BATCH_ROWS_MAX]`. The arrow
/// reader fills a batch column by column, so a batch that covers a whole row group reads
/// every page of one column chunk before it moves to the next, which is a near-forward
/// sweep through the row group; a 16,384-row batch visits every column chunk of the group
/// once per batch instead (step 1 of R1 in the 2026-09-25 survey). The competed table
/// carries the features file's 65,536-row groups or compete's own 131,072.
///
/// Under the coalesced reader (the default) a row group is read with one sequential read
/// whatever the batch size, so the batch stays at the 16,384-row floor and ~50 MB. Measured
/// from the page cache on the HYE competed table (879,018 rows, 131,072-row groups): 1.54 s
/// plain at 16,384 rows, 1.58 s plain at a row group, 2.50 s coalesced at 16,384 rows and
/// 2.76 s coalesced at a row group, so the larger batch buys no time where the read is not
/// the limit and costs ~350 MB of decoded f64.
///
/// The rows reach the closure one at a time in file order whatever the batch size, so the
/// handoff and the matrix are unchanged
/// (`every_read_mode_streams_the_same_feature_rows` covers both readers and several sizes).
fn feature_batch_rows(t: &TableFile, scan: &mumdia_io::table::ScanOptions) -> usize {
    if scan.coalesce.is_some() {
        return FEATURE_BATCH_ROWS;
    }
    let largest = t.row_group_rows().into_iter().max().unwrap_or(0);
    largest.clamp(FEATURE_BATCH_ROWS, FEATURE_BATCH_ROWS_MAX)
}

/// Payload bytes of the per-PSM metadata columns that stay resident beside the feature
/// matrix for the whole stage.
fn meta_bytes(cid: &[u32], is_decoy: &[bool], pform: &FlatStr, protein: &FlatStr) -> usize {
    std::mem::size_of_val(cid) + std::mem::size_of_val(is_decoy) + pform.bytes() + protein.bytes()
}

/// A string column held as one text buffer plus one byte offset per row, rather than one
/// `String` per row.
///
/// [`TableFile::str`] is `out.push(a.value(k).to_string())`: a 24-byte spine AND its own
/// heap block for every row, for the whole stage, including the 16-22 minutes the sidecar
/// spends training. Measured on `out_hye/psms_competed.parquet` (879,027 rows), mean byte
/// lengths are 21.66 for `peptidoform` and 13.84 for `protein`, which under mimalloc's
/// 8-byte-granular bins is 48 and 40 bytes resident per row. Flat is 8 bytes of offset
/// plus the text itself -- 29.7 and 21.8 bytes per row here -- in two allocations for the
/// whole column instead of one per row.
///
/// The allocation COUNT is the other half: the three string columns through `str` are
/// three live heap BLOCKS per PSM, 2.6 million at the measured single run and 9.4 million
/// at the six-run Astral pool, against five for the whole stage here. Blocks, not kernel
/// mappings -- mimalloc serves 32-48 byte blocks out of its segment pages, and the
/// per-process mapping limit this repository has actually measured a death on
/// (`stages/extract.rs`, 129,393 mappings at 180 GB) was 240 KB-2.4 MB blocks, which
/// mimalloc does map individually. Nothing here establishes that a 40-byte `String` block
/// consumes a mapping, so this is an allocator-pressure argument, not a `vm.max_map_count`
/// one.
///
/// `bytes()` reports capacity rather than length on purpose. `TableFile::str_flat` builds
/// `data` from `String::new()` and appends one row at a time, so it ends at a capacity in
/// `[len, 2 x len)` -- measured 1.19x on 879,027 rows of 21-byte values and 1.34x on
/// 3,133,636 -- and that slack is resident for the whole stage. `shrink` gives it back
/// once the concatenation is complete, which is what makes the per-row figure above a
/// resident figure rather than a lower bound.
#[derive(Default)]
struct FlatStr {
    /// `rows + 1` byte offsets into `data`; row `r` is `data[offsets[r]..offsets[r + 1]]`.
    /// Always a char boundary, because whole values are concatenated. Empty only before
    /// the first input has been merged in.
    offsets: Vec<usize>,
    data: String,
}

impl FlatStr {
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn get(&self, i: usize) -> &str {
        &self.data[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Rows in order. `ExactSizeIterator` is load-bearing, not decoration:
    /// `StringArray::from_iter_values` is `data_len.expect("Iterator must be sized")` on
    /// the iterator's upper size bound, so it PANICS on an unbounded one. Anything that
    /// wraps this in a `filter` before handing it to arrow turns a write into an abort;
    /// materialise the selection (see `gather`) instead.
    fn iter(&self) -> impl ExactSizeIterator<Item = &str> {
        (0..self.len()).map(|i| self.get(i))
    }

    /// Rows `lo..hi`, with the same `ExactSizeIterator` contract as [`FlatStr::iter`].
    ///
    /// The writer needs this because an arrow `StringArray` addresses its values buffer
    /// with i32 offsets: a column built from every row at once panics with `offset
    /// overflow` once the concatenated text passes 2 GiB, which a pooled scored table
    /// reaches at a few hundred million PSMs.
    fn range(&self, lo: usize, hi: usize) -> impl ExactSizeIterator<Item = &str> {
        (lo..hi).map(|i| self.get(i))
    }

    /// Resident bytes: the offsets and the text buffer as ALLOCATED, not as filled. The
    /// text arrives from `str_flat`'s amortised doubling with up to 2x slack in it, and a
    /// figure that ignored that would under-report exactly the thing `memlog` is for.
    fn bytes(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<usize>() + self.data.capacity()
    }

    /// Return the growth slack once the whole column is in, so the resident cost for the
    /// rest of the stage is the text and nothing else. One realloc-copy of the text buffer
    /// (~68 MB at the Astral pool) at a point where the metadata columns are all that is
    /// live, against 1.19-1.34x of that text held for the 16-22 minutes the sidecar trains.
    fn shrink(&mut self) {
        self.offsets.shrink_to_fit();
        self.data.shrink_to_fit();
    }

    /// Append one competed input's column, the flat counterpart of [`merge_col`].
    ///
    /// The first input is MOVED in and the offsets are then reserved to the total row
    /// count from the parquet footers; every later input has its offsets rebased onto the
    /// end of this buffer and its text appended. This is the part of the flat layout that
    /// is easiest to get subtly wrong, so `flat_columns_concatenate_like_merge_col` pins
    /// it against `merge_col` over the same inputs.
    fn merge(&mut self, (offsets, data): (Vec<usize>, String), total_rows: usize) {
        if self.offsets.is_empty() {
            self.offsets = offsets;
            self.data = data;
            if self.offsets.len() < total_rows + 1 {
                self.offsets.reserve(total_rows + 1 - self.offsets.len());
            }
            return;
        }
        let base = self.data.len();
        self.data.push_str(&data);
        // `offsets[0]` is this input's leading 0 and is already covered by the previous
        // input's last offset.
        self.offsets.extend(offsets[1..].iter().map(|o| base + o));
    }

    /// The named rows, in the given order: the flat counterpart of the `keep_rows!`
    /// gather in the top-K collapse.
    ///
    /// The text buffer is sized from the kept rows' own lengths rather than grown by
    /// doubling, so the gathered column does not carry the same slack the read does while
    /// it coexists with the source it was gathered from.
    fn gather(&self, keep: &[usize]) -> FlatStr {
        let need: usize = keep
            .iter()
            .map(|&i| self.offsets[i + 1] - self.offsets[i])
            .sum();
        let mut out = FlatStr {
            offsets: Vec::with_capacity(keep.len() + 1),
            data: String::with_capacity(need),
        };
        out.offsets.push(0);
        for &i in keep {
            out.data.push_str(self.get(i));
            out.offsets.push(out.data.len());
        }
        out
    }
}

/// Which empirical null the q-values are computed against.
#[derive(Clone, Copy, PartialEq)]
enum QMode {
    /// Target-decoy competition (native / mokapot / percolator paths).
    Decoy,
    /// Spike-in entrapment population (the `Entrapment` classifier).
    Entrapment,
}

pub struct RescoreParams<'a> {
    /// One or more competed feature tables (experiment-wide concat).
    pub competed: &'a [String],
    /// The `source` of each competed table's rows, one entry per table, non-decreasing;
    /// `None` is the table's own index. A grouped run whose pooled competed table was not
    /// written (`groups.pool_competed = false`) hands its bands' tables here in band order,
    /// every one with that run's source, so the rows arrive exactly as the pooled table
    /// would have held them.
    pub sources: Option<&'a [u32]>,
    pub out: &'a str,
    /// Working directory for the sidecar files (the feature handoff, the fold keys, the
    /// worker's output and its memmap), and the script dir for the sidecar (when
    /// selected). Callers take it from [`sidecar_work_dir`], so `MUMDIA_SIDECAR_DIR`
    /// moves it. The files are removed once the scores are aligned
    /// (`MUMDIA_KEEP_HANDOFF=1` keeps them).
    pub work_dir: &'a str,
    pub script_dir: &'a str,
    pub cfg: &'a RescoreConfig,
    pub config_hash: &'a str,
}

/// Wall time of each phase of [`run`], logged once at the end as `rescore: phase timings`
/// (docs/11 "rescore: cost"). Log only: nothing reads them back, and the scored table and
/// its report do not depend on them.
///
/// - `pass1_ms`: the metadata columns of every input, the label scan and the scalar check;
/// - `features_ms`: the feature columns, streamed into the sidecar handoff under
///   `rescore.strict` with a sidecar classifier, or into the engine's matrix otherwise;
///   `handoff_encode_ms` is the part of a streamed pass spent transposing and encoding the
///   handoff, so `features_ms - handoff_encode_ms` is the read, decode and narrowing;
/// - `classifier_ms`: the classifier, including a sidecar's handoff write when the matrix
///   was built first, the worker itself, and reading its scores back;
/// - `collapse_ms`, `q_ms`, `write_ms`: the post-classifier tail, i.e. the top-K collapse,
///   every q column and the counts at 1%, and the scored table.
#[derive(Default)]
struct PhaseTimings {
    pass1_ms: u128,
    features_ms: u128,
    handoff_encode_ms: u128,
    classifier_ms: u128,
    collapse_ms: u128,
    q_ms: u128,
    write_ms: u128,
}

impl PhaseTimings {
    fn log(&self, elapsed_ms: u128, classifier: &str) {
        info!(
            pass1_ms = self.pass1_ms as u64,
            features_ms = self.features_ms as u64,
            handoff_encode_ms = self.handoff_encode_ms as u64,
            classifier_ms = self.classifier_ms as u64,
            collapse_ms = self.collapse_ms as u64,
            q_ms = self.q_ms as u64,
            write_ms = self.write_ms as u64,
            elapsed_ms = elapsed_ms as u64,
            classifier,
            "rescore: phase timings"
        );
    }
}

/// A byte count in the largest unit that keeps it readable.
///
/// The matrix spans six orders of magnitude between the smoke fixture and a 40-run
/// experiment, so a fixed unit is unhelpful at one end or the other: `0.00 GiB` says
/// nothing, and `270336.0 MiB` says it badly.
/// Bytes of the flat f32 feature matrix for `rows` PSMs and `features` columns, or
/// `None` when the product overflows.
fn feature_matrix_bytes(rows: usize, features: usize) -> Option<u64> {
    (rows as u64)
        .checked_mul(features as u64)?
        .checked_mul(std::mem::size_of::<f32>() as u64)
}

/// `Some(ceiling)` when a configured `rescore.max_feature_matrix_gib` (0 = off) is
/// exceeded by `matrix_bytes`.
fn matrix_ceiling_exceeded(matrix_bytes: u64, max_gib: f64) -> Option<f64> {
    if max_gib <= 0.0 {
        return None;
    }
    let gib = matrix_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    (gib > max_gib).then_some(max_gib)
}

fn human_bytes(bytes: f64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes / MIB)
    } else {
        format!("{:.0} KiB", bytes / KIB)
    }
}

/// Append one competed input's column to the concatenated column.
///
/// The first input is MOVED in, not copied, and the destination is then reserved to the
/// total row count known from the parquet footers, so a multi-input concatenation
/// reallocates once and every later input appends into spare capacity. For the common
/// single-input rescore nothing is copied at all.
fn merge_col<T>(dst: &mut Vec<T>, mut src: Vec<T>, total_rows: usize) {
    if dst.is_empty() {
        *dst = src;
        if dst.len() < total_rows {
            dst.reserve(total_rows - dst.len());
        }
        return;
    }
    dst.append(&mut src);
}

/// Refuse a read of the competed inputs whose row count disagrees with the row count every
/// other read of them produced.
///
/// The metadata columns, the streamed handoff and the feature matrix are now three separate
/// passes over the same files, where they used to be one loop that could not disagree with
/// itself. Any disagreement row-misaligns the features against `label`, `source` and every
/// other metadata column, and nothing downstream can detect that, so every pass is measured
/// against the row count the parquet footers declared.
fn refuse_row_disagreement(what: &str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        anyhow::bail!(
            "rescore: {what} has {got} rows where {expected} were expected; the reads of the \
             competed inputs disagree about the input, and a disagreement row-misaligns the \
             feature rows against every metadata column"
        );
    }
    Ok(())
}

/// Refuse a rescore whose population cannot support target-decoy FDR.
///
/// A one-sided population is not a hard rescore failure in any technical sense:
/// the classifier would train and q-values would come out. They would simply be
/// meaningless, because the estimator counts one class against the other. Failing
/// here is the difference between an error and a plausible-looking result that
/// nothing downstream can detect. The message names both counts, since which side
/// is missing points at a different cause: no decoys usually means the library
/// lost its decoy half, no targets means the labels are inverted.
fn require_both_labels(n_targets: usize, n_decoys: usize) -> Result<()> {
    if n_targets == 0 || n_decoys == 0 {
        anyhow::bail!(
            "rescore requires both target and decoy PSMs for valid FDR \
             (targets={n_targets}, decoys={n_decoys})"
        );
    }
    Ok(())
}

pub fn run(p: RescoreParams) -> Result<u64> {
    run_hashed(p).map(|w| w.rows)
}

/// [`run`], returning the output's row count and the content hash its report records, so
/// an orchestrator can record the artifact without reading and hashing it again.
pub fn run_hashed(p: RescoreParams) -> Result<Written> {
    let t0 = Instant::now();
    if p.competed.is_empty() {
        anyhow::bail!("rescore requires at least one competed input");
    }
    // `--out` must not be one of the competed tables: they are read before the scored
    // table is published, so writing over one replaces it and exits 0 (docs/31 F6).
    let competed_inputs: Vec<(&str, &str)> = p
        .competed
        .iter()
        .map(|c| ("--competed", c.as_str()))
        .collect();
    mumdia_io::refuse_output_over_input(p.out, &competed_inputs)?;
    if p.cfg.folds < 2 {
        anyhow::bail!("rescore.folds must be >= 2 for out-of-fold scoring");
    }
    if let Some(s) = p.sources {
        if s.len() != p.competed.len() {
            anyhow::bail!(
                "rescore: {} sources for {} competed tables",
                s.len(),
                p.competed.len()
            );
        }
        // Non-decreasing, so each source's rows stay contiguous, which the per-source q and
        // the by-source split both rely on.
        if s.windows(2).any(|w| w[0] > w[1]) {
            anyhow::bail!("rescore: the competed tables' sources must be non-decreasing");
        }
    }
    let source_of = |k: usize| p.sources.map_or(k as u32, |s| s[k]);

    // Concatenate competed inputs.
    //
    // The three string columns are NOT `Vec<String>`. `label` is two-valued and every read
    // of it in this stage is `== "decoy"`, so it is reduced to one bit per row as it is
    // read; `peptidoform` and `protein` are flat (see `FlatStr`). Measured on the competed
    // table this takes the resident metadata from 120 bytes and 3 live heap blocks per PSM
    // to 1 + 29.7 + 21.8 = 52.5 bytes per row and 5 blocks for the whole stage (the bits,
    // and offsets + text for each of the two flat columns), once `shrink` below has taken
    // the read's growth slack back off the text buffers.
    let (mut cid, mut is_decoy, mut base, mut charge, mut prelim) = (
        Vec::new(),
        Vec::<bool>::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut pform = FlatStr::default();
    let mut protein = FlatStr::default();
    // The first label that is neither "target" nor "decoy", with its VALUE, so the message
    // `fdr::validate_labels` produced can be reproduced verbatim at the point it was
    // produced (below) rather than replaced by a bool that cannot name what it saw.
    let mut bad_label: Option<String> = None;
    let mut mz: Vec<f64> = Vec::new();
    let mut apex_rt: Vec<f64> = Vec::new();
    let mut elution_lo: Vec<f64> = Vec::new();
    let mut elution_hi: Vec<f64> = Vec::new();
    // `source` = index of the competed input each PSM came from (0..N). For a
    // single-run rescore this is all-zero; for an experiment-wide rescore over
    // several files it lets quant map each scored PSM back to its run, and it is
    // why the Mokapot PIN below keys on a unique row index rather than
    // candidate_id (which is the library index and repeats across runs).
    let mut source: Vec<u32> = Vec::new();
    // Top-K peak rank per row (#7), for the per-candidate best-peak collapse below.
    let mut peak_rank: Vec<i32> = Vec::new();
    // Feature list is taken from the schema companion of the first input. Every
    // subsequent companion must match exactly: silently concatenating differing
    // feature order/sets would train and score on semantically misaligned columns.
    let expected_schema = FeatureSchema::read(&p.competed[0])?;
    // The classifier's input columns. Without `rescore.features`/`features_file` this is
    // every feature the competed table carries; with them it is a projection of that list,
    // in schema order, and the matrix, the sidecar handoff and the training all shrink
    // with it. The list actually used is recorded in the artifact report below, which is
    // the source of truth for what the classifier saw.
    let feat_names = resolve_feature_subset(p.cfg, &expected_schema.feature_columns)?;
    // An empty feature set is refused HERE rather than carried into the passes below.
    //
    // `resolve_feature_subset` already refuses an explicitly empty selection, and
    // `FeatureSchema::read` already refuses to reconstruct an empty list from the parquet,
    // so the one way left in is a `.schema.json` companion whose `feature_columns` is `[]`.
    // What that used to do was an accident of the validation loop's shape, not a decision:
    // it ran `for (row, values) in feats.iter_rows().enumerate()`, `iter_rows` is
    // `chunks_exact(n_features.max(1))`, and with no feature columns the matrix buffer is
    // empty, so the loop never ran a single iteration -- which silently skipped the
    // prelim_score/precursor_mz finiteness check for EVERY row as well, and then trained a
    // classifier on zero inputs. Neither is worth preserving; say so instead.
    if feat_names.is_empty() {
        anyhow::bail!(
            "rescore: the feature schema of {} declares no feature columns, so the \
             classifier would have no input. Check the `.schema.json` companion next to \
             the competed table.",
            p.competed[0]
        );
    }
    if feat_names.len() != expected_schema.feature_columns.len() {
        info!(
            selected = feat_names.len(),
            available = expected_schema.feature_columns.len(),
            "rescore: feature subset active"
        );
    }
    // Total rows across the inputs, from the parquet footers, so the flat matrix is
    // allocated once at its final size.
    let mut total_rows = 0usize;
    for path in p.competed.iter() {
        total_rows += TableFile::open(path)?.nrows;
    }
    // The matrix is one contiguous f32 buffer (`rescoring::FeatureMatrix`), so its size
    // follows from the parquet footers and the selected feature count before a byte is
    // allocated. The ceiling used to be applied after the matrix had been filled, against
    // an estimate of the old `Vec<Vec<f64>>` layout (8 bytes per value plus a 24-byte
    // spine per PSM), so it could neither prevent the allocation it described nor
    // describe the one that happened (docs/29 #11). It is a limit on the matrix alone.
    let matrix_bytes = feature_matrix_bytes(total_rows, feat_names.len()).ok_or_else(|| {
        anyhow::anyhow!(
            "rescore feature matrix size overflows: {total_rows} PSMs x {} features",
            feat_names.len()
        )
    })?;
    info!(
        psms = total_rows,
        features = feat_names.len(),
        feature_matrix = %human_bytes(matrix_bytes as f64),
        folds = p.cfg.folds,
        "rescore: feature matrix size before allocation"
    );
    if let Some(ceiling) = matrix_ceiling_exceeded(matrix_bytes, p.cfg.max_feature_matrix_gib) {
        anyhow::bail!(
            "rescore feature matrix would be {} ({total_rows} PSMs x {} features x 4 bytes, \
             f32), over the configured rescore.max_feature_matrix_gib of {ceiling:.2}. This \
             is the matrix alone: per-PSM metadata, the one standardised training copy \
             native_tda holds (its peak is 1 + (folds - 1) / folds times this, 1.67x at \
             the default 3 folds and never above 2x) and the Python worker's own copy come on top. \
             Either raise the ceiling, or rescore fewer runs \
             per invocation -- `run_psm_q` is computed per source, so sub-batching costs no \
             per-run FDR, though it does change which PSMs share the pooled q_value.",
            human_bytes(matrix_bytes as f64),
            feat_names.len(),
        );
    }
    // Pass 1: the per-PSM metadata columns of every input. Reading them in their own pass,
    // separately from the ~390 feature columns, is what lets the feature pass below be
    // skipped entirely, and it preserves the order of the checks: the labels and the
    // target/decoy population are validated before anything reads a feature value, as they
    // were when both were read together. The cost is one extra parquet footer read per
    // input.
    let t_pass1 = Instant::now();
    for (src, path) in p.competed.iter().enumerate() {
        let actual_schema = FeatureSchema::read(path)?;
        validate_feature_schema(&expected_schema, &actual_schema, path)?;
        let t = TableFile::open(path)?;
        let c = t.u32("candidate_id")?;
        // Read flat and reduced to one bit here. The full label text of an input is alive
        // only for the length of this reduction (5.55 bytes per row, measured), and the
        // scan is in flat row order across inputs, so `bad_label` holds the same first
        // offending value the serial `validate_labels` below would have found.
        let l = {
            let (off, data) = t.str_flat("label")?;
            let mut d = Vec::with_capacity(t.nrows);
            for r in 0..off.len().saturating_sub(1) {
                match &data[off[r]..off[r + 1]] {
                    "decoy" => d.push(true),
                    "target" => d.push(false),
                    other => {
                        d.push(false);
                        if bad_label.is_none() {
                            bad_label = Some(other.to_string());
                        }
                    }
                }
            }
            d
        };
        let b = t.u32("base_peptide_id")?;
        let pf = t.str_flat("peptidoform")?;
        let pr = t.str_flat("protein")?;
        let z = t.f64("charge")?; // carried as an f64 feature
        let pl = t.f64("prelim_score")?;
        let pm = t.f64("precursor_mz")?;
        let ar = t.f64("apex_rt")?;
        let elo = t.f64("elution_lo")?;
        let ehi = t.f64("elution_hi")?;
        let pkr = t.i32("peak_rank").unwrap_or_else(|_| vec![0; t.nrows]);
        // Take the reader's columns rather than copying them row by row into a second set.
        // The row loop this replaces cloned every metadata STRING -- label, peptidoform,
        // protein -- into a fresh allocation while the source vector stayed alive until the
        // end of the iteration, so at peak both copies existed: twice the string bytes and,
        // more importantly, twice the live allocation count, three per PSM. `merge_col`
        // moves the whole vector for the first (usually only) input and appends the rest,
        // which copies `String` headers but never the heap blocks they own.
        merge_col(&mut cid, c, total_rows);
        merge_col(&mut peak_rank, pkr, total_rows);
        merge_col(&mut is_decoy, l, total_rows);
        merge_col(&mut base, b, total_rows);
        pform.merge(pf, total_rows);
        protein.merge(pr, total_rows);
        merge_col(
            &mut charge,
            z.into_iter().map(|v| v as i32).collect(),
            total_rows,
        );
        merge_col(&mut prelim, pl, total_rows);
        merge_col(&mut mz, pm, total_rows);
        merge_col(&mut apex_rt, ar, total_rows);
        merge_col(&mut elution_lo, elo, total_rows);
        merge_col(&mut elution_hi, ehi, total_rows);
        merge_col(&mut source, vec![source_of(src); t.nrows], total_rows);
    }
    // Both text buffers were grown by `String::push_str` one value at a time, inside
    // `str_flat` and again in `merge`, so each ends with up to 2x its own length in unused
    // capacity (measured 1.19x at 879,027 rows, 1.34x at 3.1M). Nothing appends to them
    // after this point and they are resident until the stage ends, so give it back.
    pform.shrink();
    protein.shrink();
    // The row loop this pass replaces was `for i in 0..t.nrows { cid.push(c[i]); ... }`,
    // which could not produce columns of different lengths: it indexed every column at the
    // footer's row count and panicked if one was short. `merge_col` appends whatever the
    // reader returned, and `source` is still built from `t.nrows`, so a reader column that
    // disagreed with the footer would row-misalign `source` -- the column quant splits the
    // scored table on -- against `cid`, `label` and the feature rows, with nothing
    // downstream able to detect it (`n` is taken as `cid.len()` below). `TableFile`
    // concatenates all batches and so returns exactly `nrows` today; this keeps the
    // invariant checked at the point that relies on it rather than assumed.
    for (name, len) in [
        ("candidate_id", cid.len()),
        ("peak_rank", peak_rank.len()),
        ("label", is_decoy.len()),
        ("base_peptide_id", base.len()),
        ("peptidoform", pform.len()),
        ("protein", protein.len()),
        ("charge", charge.len()),
        ("prelim_score", prelim.len()),
        ("precursor_mz", mz.len()),
        ("apex_rt", apex_rt.len()),
        ("elution_lo", elution_lo.len()),
        ("elution_hi", elution_hi.len()),
        ("source", source.len()),
    ] {
        refuse_row_disagreement(&format!("the competed column '{name}'"), len, total_rows)?;
    }
    // What `fdr::validate_labels(&label)` reported, from the scan that ran during the read
    // above. Same rule, same message, same position in the sequence of checks: an unknown
    // or malformed label must not silently count as a target, because the target-decoy
    // null depends on exact labelling (docs/18_findings_and_decisions.md).
    if let Some(l) = bad_label {
        anyhow::bail!("unknown PSM label {l:?}; expected \"target\" or \"decoy\"");
    }
    let (mut is_entrapment, mut is_real_target) = classify_entrapment(p.cfg, &protein, &is_decoy);
    let mut n = cid.len();
    // First row with a non-finite prelim_score/precursor_mz, if any. Found here but NOT
    // reported here: the serial validation this replaces checked a row's features before
    // its scalars and stopped at the first offending ROW, so the two answers are combined
    // once the feature scan has run (`bail_non_finite`).
    let mut bad_scalar = None;
    if n > 0 {
        let n_decoys = is_decoy.iter().filter(|&&v| v).count();
        let n_targets = n - n_decoys;
        require_both_labels(n_targets, n_decoys)?;
        bad_scalar = (0..n)
            .into_par_iter()
            .find_first(|&row| !prelim[row].is_finite() || !mz[row].is_finite());
    }
    let mut timings = PhaseTimings {
        pass1_ms: t_pass1.elapsed().as_millis(),
        ..PhaseTimings::default()
    };
    let t_features = Instant::now();

    // Pass 2: the feature values, either into the engine's own matrix or straight through
    // to the sidecar's handoff file.
    //
    // Under `rescore.strict` with a PIN sidecar, nothing in this process ever reads that
    // matrix. The handoff file is written FROM it and it is released immediately after
    // (that release is itself measured: docs/28), and strict turns a sidecar failure into
    // an error rather than a fall back to `native_tda`, so the native path that would read
    // it is unreachable. It was still materialised in full first -- one allocation of
    // `rows x features x 4` bytes, ~250 GB at experiment scale -- purely so that it could
    // be copied to disk. Streaming the competed feature columns straight into the handoff
    // removes it: what stays resident is one staging block of `HANDOFF_BATCH_ROWS` rows
    // (387 MB of staged f32 at 387 features, plus what `flush_block` transposes out of it
    // while a block is encoded, so call the transient roughly twice that), and the
    // non-finite validation moves into the same pass, inline with the narrowing, while the
    // row is still in cache.
    let sidecar_script = match p.cfg.classifier {
        RescorerKind::Mokapot => Some("mokapot_worker.py"),
        RescorerKind::NnTorch => Some("nn_rescore_worker.py"),
        _ => None,
    };
    // `python.is_some()` belongs in this condition: without an interpreter the sidecar
    // fails before it reads anything, and a streamed handoff would be work thrown away.
    let stream_to_handoff =
        n > 0 && p.cfg.strict && p.cfg.python.is_some() && sidecar_script.is_some();
    let mut feats_slot: Option<FeatureMatrix> = None;
    let mut prewritten: Option<SidecarPaths> = None;
    if stream_to_handoff {
        let paths = sidecar_paths(&p, sidecar_script.expect("checked in the condition"));
        check_sidecar_space(
            p.cfg.python.as_deref(),
            &paths.dir(),
            n,
            feat_names.len(),
            paths.format(),
        )?;
        // The validation aborts the stream on the first offending row rather than after the
        // file is complete. A malformed competed table used to cost nothing: the serial
        // scan ran before `run_pin_sidecar` was ever entered and nothing had been written.
        // Deferring it until after `finish()` made the same failure cost a full handoff
        // write, which at experiment scale is hundreds of GB of IO for a run that was
        // always going to abort. The message is unchanged, because `bad_scalar` is already
        // known here and `bail_non_finite` still makes the same choice between the two on
        // the row index.
        let streamed = {
            let mut w = HandoffWriter::new(&paths, &feat_names, &is_decoy, &pform, &protein, &mz)?;
            let scan = for_each_feature_batch(p.competed, &feat_names, |b| {
                // The row-at-a-time scan stopped at the first row that either held a
                // non-finite feature or was the offending scalar row, and
                // `bail_non_finite` chose between the two on the row index. A batch that
                // holds the first bad feature, or reaches the scalar row, is where that
                // row lies; the same call makes the same choice with the same message.
                let bad_feature = b.first_non_finite();
                if bad_feature.is_some()
                    || bad_scalar.is_some_and(|scalar_row| scalar_row < b.first_row + b.rows)
                {
                    bail_non_finite(bad_feature, bad_scalar, &feat_names)?;
                }
                w.push_batch(b)
            });
            // `finish` consumes the writer, and the `Err` arm drops it, so the file is
            // closed either way before the cleanup below.
            scan.and_then(|()| w.finish_timed())
        };
        let rows = match streamed {
            Ok((rows, encode)) => {
                timings.handoff_encode_ms = encode.as_millis();
                rows
            }
            Err(e) => {
                // A parquet handoff is written through `AtomicPath` and never appears at
                // its final name, but the PIN encoding writes the final path directly, so
                // an aborted stream would leave a truncated PIN for the next reader to
                // find. Take the rubble with us.
                let _ = std::fs::remove_file(&paths.handoff);
                return Err(e);
            }
        };
        // The metadata and the feature values come from two separate passes over the same
        // files. Nothing should be able to make them disagree, and if anything does, the
        // handoff is row-misaligned with every metadata column: refuse rather than train
        // on it.
        refuse_row_disagreement("the streamed handoff", rows as usize, n)?;
        info!(
            path = %paths.handoff,
            rows,
            features = feat_names.len(),
            elapsed_ms = t_features.elapsed().as_millis() as u64,
            encode_ms = timings.handoff_encode_ms as u64,
            "rescore: streamed the competed features into the sidecar handoff \
             (no engine-side feature matrix)"
        );
        prewritten = Some(paths);
    } else {
        let feats = load_feature_matrix(p.competed, &feat_names, total_rows)?;
        // Same guard as the streamed path's: the feature pass and the metadata pass are
        // separate reads of the same files and only this comparison ties them together.
        refuse_row_disagreement("the feature matrix", feats.rows(), n)?;
        let bad_feature = feats
            .find_non_finite()
            .map(|(row, col)| (row, col, feats.row(row)[col]));
        bail_non_finite(bad_feature, bad_scalar, &feat_names)?;
        feats_slot = Some(feats);
    }
    timings.features_ms = t_features.elapsed().as_millis();
    let t_classifier = Instant::now();
    crate::memlog::report(
        "rescore feature matrix",
        &[
            ("feats", feats_slot.as_ref().map_or(0, |f| f.bytes())),
            (
                "metadata_columns",
                meta_bytes(&cid, &is_decoy, &pform, &protein),
            ),
        ],
    );
    info!(
        psms = n,
        features = feat_names.len(),
        folds = p.cfg.folds,
        "rescore: loaded competed PSMs"
    );

    // Track the path actually taken so the report reflects reality rather than a
    // hardcoded label, and pick the null the q-values are computed against.
    let mut classifier_used = "native_tda";
    let mut model_identity = "native-percolator-lite-v1".to_string();
    let mut qmode = QMode::Decoy;
    // The `MUMDIA_NN_*` knobs the NN worker inherited, when it ran (see `inherited_nn_env`).
    let mut nn_env: Option<std::collections::BTreeMap<String, String>> = None;

    let mut scores = if n == 0 {
        classifier_used = "not_run_empty";
        model_identity = "none-empty-input".to_string();
        Vec::new()
    } else {
        match p.cfg.classifier {
            RescorerKind::Mokapot => match run_pin_sidecar(
                &p,
                "mokapot_worker.py",
                &feat_names,
                &cid,
                &is_decoy,
                &pform,
                &protein,
                &mz,
                &mut feats_slot,
                &base,
                prewritten.take(),
            ) {
                Ok(s) => {
                    info!("rescore: using Mokapot scores");
                    classifier_used = "mokapot";
                    let estimator =
                        std::env::var("MUMDIA_RESCORE_MODEL").unwrap_or_else(|_| "nn".to_string());
                    model_identity = format!("mokapot-{estimator}");
                    s
                }
                Err(e) => {
                    if p.cfg.strict {
                        anyhow::bail!(
                            "rescore: Mokapot sidecar failed ({e}) and rescore.strict=true"
                        );
                    }
                    warn!("rescore: Mokapot failed ({e}); falling back to native_tda");
                    native_scores(&p, kept_matrix(&feats_slot), &is_decoy, &base, &prelim)
                }
            },
            RescorerKind::NnTorch => match run_pin_sidecar(
                &p,
                "nn_rescore_worker.py",
                &feat_names,
                &cid,
                &is_decoy,
                &pform,
                &protein,
                &mz,
                &mut feats_slot,
                &base,
                prewritten.take(),
            ) {
                Ok(s) => {
                    info!("rescore: using PyTorch NN sidecar scores");
                    classifier_used = "nn_torch";
                    model_identity = "nn-torch-semisup-sidecar-v1".to_string();
                    let env = inherited_nn_env(std::env::vars_os());
                    if !env.is_empty() {
                        info!(
                            ?env,
                            "rescore: the NN worker inherited these MUMDIA_NN_* variables"
                        );
                    }
                    nn_env = Some(env);
                    s
                }
                Err(e) => {
                    if p.cfg.strict {
                        anyhow::bail!(
                            "rescore: NnTorch sidecar failed ({e}) and rescore.strict=true"
                        );
                    }
                    warn!("rescore: NnTorch failed ({e}); falling back to native_tda");
                    native_scores(&p, kept_matrix(&feats_slot), &is_decoy, &base, &prelim)
                }
            },
            RescorerKind::Percolator => {
                if p.cfg.strict {
                    anyhow::bail!("rescore: classifier=percolator but percolator.exe is not wired, and rescore.strict=true");
                }
                warn!("rescore: percolator.exe path not wired; using native_tda");
                native_scores(&p, kept_matrix(&feats_slot), &is_decoy, &base, &prelim)
            }
            RescorerKind::NativeTda => {
                native_scores(&p, kept_matrix(&feats_slot), &is_decoy, &base, &prelim)
            }
            RescorerKind::Entrapment => {
                let n_ent = is_entrapment.iter().filter(|&&b| b).count();
                if p.cfg.entrapment_marker.is_none() || n_ent == 0 {
                    if p.cfg.strict {
                        anyhow::bail!(
                            "rescore: classifier=entrapment but no entrapment PSMs matched \
                             (marker={:?}, n_ent={n_ent}) and rescore.strict=true; set \
                             rescore.entrapment_marker to the spike-in accession substring",
                            p.cfg.entrapment_marker
                        );
                    }
                    warn!(
                        entrapment_psms = n_ent,
                        "rescore: classifier=entrapment but no entrapment PSMs \
                         (set rescore.entrapment_marker to the spike-in accession \
                         substring); falling back to native_tda"
                    );
                    native_scores(&p, kept_matrix(&feats_slot), &is_decoy, &base, &prelim)
                } else if p.cfg.python.is_some() {
                    match run_entrapment_gbm(
                        &p,
                        &feat_names,
                        &cid,
                        &base,
                        &is_entrapment,
                        &is_decoy,
                        kept_matrix(&feats_slot),
                    ) {
                        Ok(s) => {
                            info!(
                                entrapment_psms = n_ent,
                                "rescore: using entrapment GBM sidecar scores"
                            );
                            classifier_used = "entrapment_gbm";
                            model_identity = "entrapment-gbm-sidecar-v1".to_string();
                            qmode = QMode::Entrapment;
                            s
                        }
                        Err(e) => {
                            if p.cfg.strict {
                                anyhow::bail!("rescore: entrapment GBM sidecar failed ({e}) and rescore.strict=true");
                            }
                            warn!("rescore: entrapment GBM sidecar failed ({e}); using native linear entrapment fallback");
                            classifier_used = "entrapment_native";
                            // `is_decoy`, not `is_entrapment`: the negative class for
                            // TRAINING is the in-silico decoy population, exactly as on
                            // every other classifier path. Entrapment is the evaluation
                            // null and must stay held out -- training on it makes the
                            // leak estimate a measure of the fit rather than of the FDR.
                            // Passing `is_entrapment` here also inverted the semantics of
                            // `percolator_lite`'s positive set, which selects on
                            // `is_decoy == false`: every entrapment row with a low
                            // internal q was recruited as a POSITIVE training example and
                            // counted as a target inside the loop's own q estimate.
                            model_identity = "native-percolator-lite-entrapment-v1".to_string();
                            qmode = QMode::Entrapment;
                            native_scores(&p, kept_matrix(&feats_slot), &is_decoy, &base, &prelim)
                        }
                    }
                } else {
                    info!(
                        entrapment_psms = n_ent,
                        "rescore: classifier=entrapment, no rescore.python; using native linear entrapment rescorer"
                    );
                    classifier_used = "entrapment_native";
                    model_identity = "native-percolator-lite-entrapment-v1".to_string();
                    qmode = QMode::Entrapment;
                    native_scores(&p, kept_matrix(&feats_slot), &is_decoy, &base, &prelim)
                }
            }
        }
    };
    if let Some((row, score)) = scores
        .iter()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        anyhow::bail!("rescore produced non-finite score at flat row {row}: {score}");
    }
    timings.classifier_ms = t_classifier.elapsed().as_millis();
    let t_collapse = Instant::now();

    // Top-K per-candidate collapse (#7): keep only the best-scoring peak per
    // (source, candidate_id), so the rescorer (not the up-front apex pick) selects
    // the peak and the terminal q-null is exactly one row per candidate. Promoting K
    // peaks therefore does not K-inflate the decoy null. Guarded: at
    // promote_top_peaks = 1 every candidate has a single peak, `best.len() == n`, and
    // the whole block is a no-op (byte-identical). Decoys collapse by the identical
    // rule, so target/decoy exchangeability is preserved. Tie-break: lower peak_rank
    // then lower row index (deterministic).
    //
    // The no-op is now detected before the map is built (`every_candidate_is_unique`):
    // the map was `with_capacity(n)` over every row, ~9 GB transient at the 258.75M-row
    // immunopeptidomics pool, to find that no key repeats.
    if n > 0 && !every_candidate_is_unique(&source, &cid) {
        let mut best: HashMap<(u32, u32), usize> = HashMap::with_capacity(n);
        for i in 0..n {
            let key = (source[i], cid[i]);
            match best.get(&key) {
                Some(&j) => {
                    let better = scores[i] > scores[j]
                        || (scores[i] == scores[j] && (peak_rank[i], i) < (peak_rank[j], j));
                    if better {
                        best.insert(key, i);
                    }
                }
                None => {
                    best.insert(key, i);
                }
            }
        }
        if best.len() < n {
            let mut keep: Vec<usize> = best.into_values().collect();
            keep.sort_unstable();
            macro_rules! keep_rows {
                ($($v:ident),+ $(,)?) => {$(
                    { let tmp: Vec<_> = keep.iter().map(|&i| $v[i].clone()).collect(); $v = tmp; }
                )+};
            }
            // The flat string columns gather their own text rather than cloning one
            // `String` per kept row.
            pform = pform.gather(&keep);
            protein = protein.gather(&keep);
            keep_rows!(
                cid,
                base,
                charge,
                prelim,
                apex_rt,
                elution_lo,
                elution_hi,
                source,
                scores,
                peak_rank,
                is_decoy,
                is_entrapment,
                is_real_target
            );
            n = keep.len();
            info!(kept = n, "rescore: top-K per-candidate best-peak collapse");
        }
    }
    timings.collapse_ms = t_collapse.elapsed().as_millis();
    let t_q = Instant::now();

    // PSM-level q-values against the selected null. The split form reads the two columns
    // in place: the pair form made the stage copy them into an `n * 16` byte buffer first,
    // 4.1 GB at the 258.75M-row immunopeptidomics pool, for one walk into the kernel.
    let psm_q = match qmode {
        QMode::Decoy => target_decoy_q_split(&scores, &is_decoy),
        QMode::Entrapment => entrapment_q(
            &scores,
            &is_entrapment,
            &is_real_target,
            p.cfg.entrapment_ratio,
        ),
    };

    // Peptide-level q: reduce to best PSM per base peptide, q on that set, map
    // back. Protein-group q: same over the protein-accession-set string (the MVP
    // grouping; decoys carry a DECOY_ prefix). Full parsimony/razor is a later
    // option (docs/12_quant_lfq_align_mbr_report_audit.md). Group score = best
    // member PSM score.
    let peptide_q = grouped_q(
        &base,
        &scores,
        &is_decoy,
        &is_entrapment,
        &is_real_target,
        qmode,
        p.cfg.entrapment_ratio,
    );
    // Intern the protein-accession-set strings to dense u32 ids once (first-seen
    // order) so protein-group grouping runs over integers exactly like the peptide
    // path, avoiding hashing/cloning ~574k strings per grouped_q lookup. The map is
    // bijective, so grouping and the resulting per-PSM q-values are unchanged.
    //
    // The interner grows with the DISTINCT keys and is not presized: `with_capacity(n)`
    // would size it by rows, 17.7 GB at the immunopeptidomics pool, for a key set that
    // is a small fraction of them (69,958 proteins over 879,027 rows on HYE).
    let protein_id: Vec<u32> = {
        let mut interner: HashMap<&str, u32> = HashMap::new();
        let mut ids = Vec::with_capacity(protein.len());
        for s in protein.iter() {
            let next = interner.len() as u32;
            ids.push(*interner.entry(s).or_insert(next));
        }
        ids
    };
    let pg_q = grouped_q(
        &protein_id,
        &scores,
        &is_decoy,
        &is_entrapment,
        &is_real_target,
        qmode,
        p.cfg.entrapment_ratio,
    );
    // Multi-context q-values (docs/11_compete_rescore_fdr.md). The pooled per-PSM q
    // is `experiment_psm_q`; `run_psm_q` re-runs TDA within each source (run) so a
    // per-run quant/report gets a real per-run FDR rather than the pooled value;
    // `precursor_q` groups on peptidoform+charge. `global_q_value` and
    // `experiment_psm_q` are kept as byte-identical aliases of the pooled q for
    // backward-compat, and are written from the SAME Arrow array below rather than from
    // two more copies of the column (see `write_scored_table`).
    // Per-run PSM q: TDA within each source separately (`per_source_q`). Single-run
    // (source all-zero) => equals `q_value`. No floats are summed.
    let run_psm_q = per_source_q(
        &source,
        &scores,
        &is_decoy,
        &is_entrapment,
        &is_real_target,
        qmode,
        p.cfg.entrapment_ratio,
        &psm_q,
    );
    // Precursor-level q: group on peptidoform+charge (interned to dense u32 like the
    // protein path) and run TDA over the best PSM per precursor.
    let precursor_id: Vec<u32> = {
        let mut interner: HashMap<(&str, i32), u32> = HashMap::new();
        let mut ids = Vec::with_capacity(pform.len());
        for (pf, &z) in pform.iter().zip(charge.iter()) {
            let next = interner.len() as u32;
            ids.push(*interner.entry((pf, z)).or_insert(next));
        }
        ids
    };
    let precursor_q = grouped_q(
        &precursor_id,
        &scores,
        &is_decoy,
        &is_entrapment,
        &is_real_target,
        qmode,
        p.cfg.entrapment_ratio,
    );

    // Reported IDs: real targets in entrapment mode (spike-in excluded), else all
    // non-decoy targets.
    let is_reported: Vec<bool> = match qmode {
        QMode::Decoy => is_decoy.iter().map(|d| !d).collect(),
        QMode::Entrapment => is_real_target.clone(),
    };

    let n_psm_1 = (0..n)
        .filter(|&i| is_reported[i] && psm_q[i] <= 0.01)
        .count();
    let n_pep_1 = {
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            if is_reported[i] && peptide_q[i] <= 0.01 {
                seen.insert(base[i]);
            }
        }
        seen.len()
    };
    let n_pg_1 = {
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            if is_reported[i] && pg_q[i] <= 0.01 {
                seen.insert(protein_id[i]);
            }
        }
        seen.len()
    };
    let n_prec_1 = {
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            if is_reported[i] && precursor_q[i] <= 0.01 {
                seen.insert(precursor_id[i]);
            }
        }
        seen.len()
    };
    // Entrapment leak: spike-in peptides passing the 1% gate. A running check on
    // FDR validity (should track the reported q if the null is well-modelled).
    let n_entrap_1 = {
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            if is_entrapment[i] && peptide_q[i] <= 0.01 {
                seen.insert(base[i]);
            }
        }
        seen.len()
    };
    timings.q_ms = t_q.elapsed().as_millis();
    let t_write = Instant::now();

    let (rows, scored_hash) = write_scored(
        p.out,
        true,
        ScoredColumns {
            cid,
            pform,
            charge,
            is_decoy,
            protein,
            base,
            apex_rt,
            elution_lo,
            elution_hi,
            scores,
            psm_q,
            peptide_q,
            pg_q,
            prelim,
            source,
            run_psm_q,
            precursor_q,
            peak_rank,
        },
    )?;
    timings.write_ms = t_write.elapsed().as_millis();

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("psms".to_string(), json!(n));
    stats.insert("classifier".to_string(), json!(classifier_used));
    stats.insert("target_psms_at_1pct".to_string(), json!(n_psm_1));
    stats.insert("target_peptides_at_1pct".to_string(), json!(n_pep_1));
    stats.insert("target_protein_groups_at_1pct".to_string(), json!(n_pg_1));
    stats.insert("target_precursors_at_1pct".to_string(), json!(n_prec_1));
    if qmode == QMode::Entrapment {
        stats.insert(
            "entrapment_ratio".to_string(),
            json!(p.cfg.entrapment_ratio),
        );
        stats.insert("entrapment_peptides_at_1pct".to_string(), json!(n_entrap_1));
    }
    let mut params = json!({
        "classifier": classifier_used,
        "classifier_requested": format!("{:?}", p.cfg.classifier),
        "strict": p.cfg.strict,
        "folds": p.cfg.folds,
        "num_iter": p.cfg.num_iter,
        "train_fdr": p.cfg.train_fdr,
        "feature_schema_id": expected_schema.schema_id,
        "train_neg_ratio": p.cfg.train_neg_ratio,
        "train_neg_select": format!("{:?}", p.cfg.train_neg_select).to_lowercase(),
        "train_subsample": p.cfg.train_subsample,
        "train_warm_epochs": p.cfg.train_warm_epochs,
        "train_margin_frac": p.cfg.train_margin_frac,
        "seeds": p.cfg.seeds.max(1),
        "n_features_used": feat_names.len(),
        "n_features_available": expected_schema.feature_columns.len(),
        "feature_preset": if p.cfg.features.is_some() || p.cfg.features_file.is_some() {
            "explicit".to_string()
        } else {
            format!("{:?}", p.cfg.feature_preset).to_lowercase()
        },
        "feature_selection_id": crate::stages::features::feature_schema_id(&feat_names),
        "features_used": if feat_names.len() == expected_schema.feature_columns.len() {
            serde_json::Value::Null
        } else {
            json!(feat_names)
        },
        "competed_inputs": p.competed,
        "config_hash": p.config_hash,
    });
    // Only when given, so a report of the usual one-table-per-source call is unchanged.
    if let Some(sources) = p.sources {
        params["competed_sources"] = json!(sources);
    }
    if let Some(env) = &nn_env {
        params["nn_env"] = json!(env);
    }
    let report = ArtifactReport {
        logical_name: artifact::PSMS_SCORED.0.to_string(),
        schema_name: artifact::PSMS_SCORED.0.to_string(),
        schema_version: artifact::PSMS_SCORED.1,
        stage: "rescore".to_string(),
        rows,
        // Computed while the table was written.
        content_hash: scored_hash.expect("write_scored hashes when asked"),
        params,
        stats,
        model_identity: Some(model_identity),
        elapsed_ms: elapsed,
    };
    report.write_for(p.out)?;

    timings.log(elapsed, classifier_used);
    info!(
        psms = n,
        target_psms_at_1pct = n_psm_1,
        target_peptides_at_1pct = n_pep_1,
        elapsed_ms = elapsed,
        "rescore: done"
    );
    Ok(report.written())
}

/// The `psms_scored` columns, in schema order.
struct ScoredColumns {
    cid: Vec<u32>,
    pform: FlatStr,
    charge: Vec<i32>,
    /// `label`, as the bit every read of it in this stage takes. `validate_labels` (the
    /// check reproduced in `run`) guarantees the column is exactly {"target", "decoy"},
    /// so writing `if decoy {"decoy"} else {"target"}` reproduces the input bytes.
    is_decoy: Vec<bool>,
    protein: FlatStr,
    base: Vec<u32>,
    apex_rt: Vec<f64>,
    elution_lo: Vec<f64>,
    elution_hi: Vec<f64>,
    scores: Vec<f64>,
    psm_q: Vec<f64>,
    peptide_q: Vec<f64>,
    pg_q: Vec<f64>,
    prelim: Vec<f64>,
    source: Vec<u32>,
    run_psm_q: Vec<f64>,
    precursor_q: Vec<f64>,
    peak_rank: Vec<i32>,
}

/// Write `psms_scored`, sharing the Arrow array behind every column that repeats.
///
/// Three of the 21 columns are duplicates by construction: `protein_group` is `protein`,
/// and `global_q_value` and `experiment_psm_q` are both the pooled `q_value` (the comments
/// at their definitions say so). Through `write_table` each duplicate had to arrive as its
/// own `Vec` -- `protein.clone()` and two `psm_q.clone()`s at the call site -- so the stage
/// carried a second full copy of the protein strings, one heap block per PSM, and two more
/// copies of the pooled q column, all three alive at the same moment as the originals. An
/// `ArrayRef` is refcounted, so passing the same array in several slots of the batch writes
/// the same bytes from one buffer.
///
/// What this does NOT claim is a strict reduction in peak: `write_table` encodes in
/// `WRITE_TABLE_CHUNK_ROWS` chunks and moves each chunk's rows out of the source vectors,
/// so its own Arrow copy is one chunk wide, while one batch means the `StringArray`
/// conversion re-packs each of `peptidoform`, `label` and `protein` into one contiguous
/// offsets+values buffer at once. The three clones go away; a full-width Arrow copy of each
/// string column arrives. Which side is larger depends on the string widths of the run, so
/// the honest claim is the one the test below pins: the file is identical.
///
/// The parquet is byte-identical to the previous `write_table` call: the same schema
/// (names, types, all non-nullable, same order), one record batch, and `BatchWriter` opens
/// the identical SNAPPY writer on the identical `AtomicPath` temp-then-rename. The
/// duplicate columns are encoded independently, exactly as they were when they were
/// separate vectors.
fn scored_schema() -> std::sync::Arc<arrow::datatypes::Schema> {
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    let f = |name: &str, t: DataType| Field::new(name, t, false);
    Arc::new(Schema::new(vec![
        f("candidate_id", DataType::UInt32),
        f("peptidoform", DataType::Utf8),
        f("charge", DataType::Int32),
        f("label", DataType::Utf8),
        f("protein", DataType::Utf8),
        f("base_peptide_id", DataType::UInt32),
        f("apex_rt", DataType::Float64),
        f("elution_lo", DataType::Float64),
        f("elution_hi", DataType::Float64),
        f("score", DataType::Float64),
        f("q_value", DataType::Float64),
        f("peptide_q_value", DataType::Float64),
        f("protein_group", DataType::Utf8),
        f("pg_q_value", DataType::Float64),
        f("global_q_value", DataType::Float64),
        f("prelim_score", DataType::Float64),
        // Run identity for experiment-wide rescore (index into --competed); all-zero for
        // a single-run rescore. Lets quant map scores per file.
        f("source", DataType::UInt32),
        // Multi-context q columns (docs/11_compete_rescore_fdr.md). run_psm_q = per-run
        // PSM FDR; experiment_psm_q = pooled PSM FDR (== q_value/global_q_value);
        // precursor_q = per (peptidoform+charge) FDR.
        f("run_psm_q", DataType::Float64),
        f("experiment_psm_q", DataType::Float64),
        f("precursor_q", DataType::Float64),
        // Which chromatographic peak the rescorer selected for this candidate (#7).
        // 0 = the up-front apex; > 0 = a promoted alternate peak won.
        f("selected_peak_rank", DataType::Int32),
    ]))
}

/// Rows per record batch in [`write_scored`].
///
/// The same 65,536 that `mumdia_io::table::write_table` feeds its writer, so the parquet
/// is unchanged; what the chunk bounds is the width of one arrow `StringArray`.
const SCORED_CHUNK_ROWS: usize = 1 << 16;

/// [`write_scored`] without the digest; the tests compare its bytes across constructions.
#[cfg(test)]
fn write_scored_table(path: &str, c: ScoredColumns) -> Result<u64> {
    Ok(write_scored(path, false, c)?.0)
}

/// [`write_scored_table`], hashing the table while it is written when `hash` is set; the
/// bytes are the same either way.
fn write_scored(path: &str, hash: bool, c: ScoredColumns) -> Result<(u64, Option<String>)> {
    use arrow::array::{ArrayRef, Int32Array, StringArray, UInt32Array};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let schema = scored_schema();
    let nrows = c.cid.len();
    let mut opts = mumdia_io::table::WriteOptions::new();
    if hash {
        opts = opts.content_hash();
    }
    let mut w = mumdia_io::table::BatchWriter::with_options(path, schema.clone(), opts)?;
    // At least one batch, so a scored table with no rows still writes its schema.
    let mut lo = 0usize;
    loop {
        let hi = (lo + SCORED_CHUNK_ROWS).min(nrows);
        // `StringArray::from(Vec<String>)` IS `StringArray::from_iter_values` in arrow 59
        // (`string_array.rs`), so building the same values from the flat columns and from
        // the label bits produces the identical offsets+values buffers, and therefore the
        // identical parquet. `the_flat_metadata_columns_write_the_same_scored_parquet`
        // pins that against the previous `Vec<String>` construction.
        let protein: ArrayRef = Arc::new(StringArray::from_iter_values(c.protein.range(lo, hi)));
        let f = |v: &[f64]| -> ArrayRef {
            Arc::new(Float64Array::from_iter_values(v[lo..hi].iter().copied()))
        };
        let q: ArrayRef = f(&c.psm_q);
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from_iter_values(c.cid[lo..hi].iter().copied())),
            Arc::new(StringArray::from_iter_values(c.pform.range(lo, hi))),
            Arc::new(Int32Array::from_iter_values(
                c.charge[lo..hi].iter().copied(),
            )),
            Arc::new(StringArray::from_iter_values(
                c.is_decoy[lo..hi]
                    .iter()
                    .map(|&d| if d { "decoy" } else { "target" }),
            )),
            protein.clone(),
            Arc::new(UInt32Array::from_iter_values(
                c.base[lo..hi].iter().copied(),
            )),
            f(&c.apex_rt),
            f(&c.elution_lo),
            f(&c.elution_hi),
            f(&c.scores),
            q.clone(),
            f(&c.peptide_q),
            protein,
            f(&c.pg_q),
            q.clone(),
            f(&c.prelim),
            Arc::new(UInt32Array::from_iter_values(
                c.source[lo..hi].iter().copied(),
            )),
            f(&c.run_psm_q),
            q,
            f(&c.precursor_q),
            Arc::new(Int32Array::from_iter_values(
                c.peak_rank[lo..hi].iter().copied(),
            )),
        ];
        let batch = RecordBatch::try_new(schema.clone(), arrays)
            .with_context(|| format!("building the scored record batch for {path}"))?;
        w.write(&batch)?;
        lo = hi;
        if lo >= nrows {
            break;
        }
    }
    w.close_with_digest()
}

/// Reject a concatenation whose feature companions differ in either identity or
/// ordered feature columns. Both checks are intentional: the ID catches a
/// declared contract mismatch, while the explicit column comparison protects
/// against a malformed or manually edited companion.
/// The classifier's feature columns: every column of the schema, or the projection named
/// by `rescore.features` / `rescore.features_file`.
///
/// The result keeps SCHEMA order, not the order the caller listed, so a selection can
/// never silently reorder the matrix (the sidecar contract is positional). A name that is
/// not in the schema is an error rather than a silent drop: a typo in a 100-name list
/// would otherwise train a different model than the one asked for.
/// The `compact` preset, one name per line; see `FeaturePreset::Compact`.
const COMPACT_FEATURES: &str = include_str!("feature_presets/compact.txt");

/// One feature name per line, blank lines and `#` comments ignored.
fn parse_feature_list(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn preset_names(preset: FeaturePreset) -> Option<Vec<String>> {
    match preset {
        FeaturePreset::All => None,
        FeaturePreset::Compact => Some(parse_feature_list(COMPACT_FEATURES)),
    }
}

fn resolve_feature_subset(cfg: &RescoreConfig, available: &[String]) -> Result<Vec<String>> {
    let wanted: Vec<String> = match (&cfg.features, &cfg.features_file) {
        (Some(_), Some(_)) => anyhow::bail!(
            "rescore.features and rescore.features_file are mutually exclusive; set one"
        ),
        (None, None) => {
            let Some(names) = preset_names(cfg.feature_preset) else {
                return Ok(available.to_vec());
            };
            // A preset is a default, not a contract with one feature set: names the
            // table lacks (a smaller `features.set`) are skipped, visibly.
            let have: std::collections::HashSet<&str> =
                available.iter().map(String::as_str).collect();
            let keep: std::collections::HashSet<&str> = names
                .iter()
                .map(String::as_str)
                .filter(|n| have.contains(n))
                .collect();
            if keep.is_empty() {
                anyhow::bail!(
                    "rescore.feature_preset {:?} shares no column with the competed table's \
                     feature schema",
                    cfg.feature_preset
                );
            }
            if keep.len() < names.len() {
                info!(
                    preset = ?cfg.feature_preset,
                    skipped = names.len() - keep.len(),
                    kept = keep.len(),
                    "rescore: preset names absent from this feature set are skipped"
                );
            }
            return Ok(available
                .iter()
                .filter(|a| keep.contains(a.as_str()))
                .cloned()
                .collect());
        }
        (Some(list), None) => list.clone(),
        (None, Some(path)) => parse_feature_list(
            &std::fs::read_to_string(path)
                .with_context(|| format!("reading rescore.features_file {path}"))?,
        ),
    };
    if wanted.is_empty() {
        anyhow::bail!("rescore feature selection resolved to an empty list");
    }
    let have: std::collections::HashSet<&str> = available.iter().map(String::as_str).collect();
    let missing: Vec<&String> = wanted
        .iter()
        .filter(|w| !have.contains(w.as_str()))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "rescore feature selection names {} column(s) absent from the competed table's \
             feature schema, first few: {:?}",
            missing.len(),
            missing.iter().take(5).collect::<Vec<_>>()
        );
    }
    let keep: std::collections::HashSet<&str> = wanted.iter().map(String::as_str).collect();
    Ok(available
        .iter()
        .filter(|a| keep.contains(a.as_str()))
        .cloned()
        .collect())
}

fn validate_feature_schema(
    expected: &FeatureSchema,
    actual: &FeatureSchema,
    path: &str,
) -> Result<()> {
    if expected.schema_id != actual.schema_id || expected.feature_columns != actual.feature_columns
    {
        anyhow::bail!(
            "rescore feature schema mismatch for '{path}': expected id '{}' columns {:?}, \
             found id '{}' columns {:?}",
            expected.schema_id,
            expected.feature_columns,
            actual.schema_id,
            actual.feature_columns
        );
    }
    Ok(())
}

/// Native semi-supervised rescorer scores.
fn native_scores(
    p: &RescoreParams,
    feats: &FeatureMatrix,
    is_decoy: &[bool],
    fold_key: &[u32],
    prelim: &[f64],
) -> Vec<f64> {
    percolator_lite(RescoreInput {
        features: feats,
        is_decoy,
        fold_key,
        init_score: prelim,
        folds: p.cfg.folds,
        num_iter: p.cfg.num_iter,
        train_fdr: p.cfg.train_fdr,
    })
}

/// Per-PSM entrapment classification from the protein-accession string. A target
/// is entrapment when its protein contains `entrapment_marker`, does not contain
/// `entrapment_exclude` (if set), and does not match any
/// `entrapment_contaminant_markers` (genuine contaminants inside the spike-in
/// proteome, e.g. keratins, which are real and so must not be used as negatives).
/// Everything else non-decoy is a real target. Decoys are neither. Returns
/// `(is_entrapment, is_real_target)`.
fn classify_entrapment(
    cfg: &RescoreConfig,
    protein: &FlatStr,
    is_decoy: &[bool],
) -> (Vec<bool>, Vec<bool>) {
    let marker = cfg.entrapment_marker.as_deref();
    let exclude = cfg.entrapment_exclude.as_deref();
    let contaminants = &cfg.entrapment_contaminant_markers;
    let n = protein.len();
    let mut ent = vec![false; n];
    let mut real = vec![false; n];
    for i in 0..n {
        if is_decoy[i] {
            continue;
        }
        let acc = protein.get(i);
        let is_ent = match marker {
            Some(m) => {
                acc.contains(m)
                    && exclude.is_none_or(|e| !acc.contains(e))
                    && !contaminants.iter().any(|c| acc.contains(c.as_str()))
            }
            None => false,
        };
        ent[i] = is_ent;
        real[i] = !is_ent;
    }
    (ent, real)
}

/// The per-source PSM q column: the pooled kernel (`qmode`) re-run within each source,
/// scattered back to the rows.
///
/// The rows of one source are one contiguous run whenever `source` does not decrease,
/// which is how the stage concatenates its inputs, so each source's rows are a slice and
/// the kernel reads them in place. The previous form collected one row-index vector per
/// source (8 bytes per row, all alive at once) and copied the scores and labels out
/// through them; the slice holds the same rows in the same order, so the kernel sees the
/// same input and returns the same q. With a single source the input IS the pooled one,
/// and `pooled` (the kernel's output on it) is returned as is. A `source` that decreases
/// somewhere takes the index form, which is exact in every case.
#[allow(clippy::too_many_arguments)]
fn per_source_q(
    source: &[u32],
    scores: &[f64],
    is_decoy: &[bool],
    is_entrapment: &[bool],
    is_real: &[bool],
    qmode: QMode,
    ratio: f64,
    pooled: &[f64],
) -> Vec<f64> {
    let n = scores.len();
    let kernel = |lo: usize, hi: usize| match qmode {
        QMode::Decoy => target_decoy_q_split(&scores[lo..hi], &is_decoy[lo..hi]),
        QMode::Entrapment => entrapment_q(
            &scores[lo..hi],
            &is_entrapment[lo..hi],
            &is_real[lo..hi],
            ratio,
        ),
    };
    if source.windows(2).all(|w| w[0] == w[1]) {
        return pooled.to_vec();
    }
    if source.windows(2).all(|w| w[0] <= w[1]) {
        let mut rq = vec![1.0f64; n];
        let mut lo = 0usize;
        while lo < n {
            let s = source[lo];
            let hi = lo + source[lo..].iter().take_while(|&&x| x == s).count();
            rq[lo..hi].copy_from_slice(&kernel(lo, hi));
            lo = hi;
        }
        return rq;
    }
    per_source_q_by_index(
        source,
        scores,
        is_decoy,
        is_entrapment,
        is_real,
        qmode,
        ratio,
    )
}

/// [`per_source_q`] for a `source` column in any order: one row-index vector per source,
/// in ascending source order (a `BTreeMap`, so the result does not depend on hashing).
fn per_source_q_by_index(
    source: &[u32],
    scores: &[f64],
    is_decoy: &[bool],
    is_entrapment: &[bool],
    is_real: &[bool],
    qmode: QMode,
    ratio: f64,
) -> Vec<f64> {
    let mut by_src: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, &s) in source.iter().enumerate() {
        by_src.entry(s).or_default().push(i);
    }
    let mut rq = vec![1.0f64; scores.len()];
    for (_s, idxs) in by_src {
        let q = match qmode {
            QMode::Decoy => target_decoy_q_by(idxs.len(), |k| (scores[idxs[k]], is_decoy[idxs[k]])),
            QMode::Entrapment => {
                let sc: Vec<f64> = idxs.iter().map(|&i| scores[i]).collect();
                let en: Vec<bool> = idxs.iter().map(|&i| is_entrapment[i]).collect();
                let re: Vec<bool> = idxs.iter().map(|&i| is_real[i]).collect();
                entrapment_q(&sc, &en, &re, ratio)
            }
        };
        for (k, &i) in idxs.iter().enumerate() {
            rq[i] = q[k];
        }
    }
    rq
}

/// Whether every `(source, candidate_id)` pair occurs once, which is when the top-K
/// collapse keeps every row and can be skipped.
///
/// The rows of one competed input are contiguous and `source` is the input's index, so
/// `source` does not decrease along the rows. Each source's candidate ids are then checked
/// against one bitset over `0..=max(candidate_id)`, cleared between sources: an eighth of
/// a byte per library candidate (25 MB at a 203M-precursor library) and one pass, where
/// the map the collapse builds takes ~35 bytes per row. `false` whenever that cannot
/// decide: a source that decreases, or a repeated pair, sends the caller to the map, which
/// is exact in every case.
fn every_candidate_is_unique(source: &[u32], cid: &[u32]) -> bool {
    if source.windows(2).any(|w| w[1] < w[0]) {
        return false;
    }
    let Some(&max) = cid.iter().max() else {
        return true;
    };
    let mut seen = vec![0u64; max as usize / 64 + 1];
    let mut current = source.first().copied();
    for (&s, &c) in source.iter().zip(cid) {
        if Some(s) != current {
            seen.fill(0);
            current = Some(s);
        }
        let (word, bit) = (c as usize / 64, 1u64 << (c % 64));
        if seen[word] & bit != 0 {
            return false;
        }
        seen[word] |= bit;
    }
    true
}

/// A group's winning row: its score, the three label bits it carries into the q kernel,
/// and the flat row index the q is written back to.
type GroupBest = (f64, bool, bool, bool, usize);

/// Group ids as a dense `0..span` range, so the reduction below can index an array
/// instead of hashing.
///
/// `protein_id` and `precursor_id` arrive dense already (both were interned in `run`
/// immediately before the call), so their span IS their group count and they always take
/// the direct arm. `base_peptide_id` is dense over the LIBRARY, not over the competed
/// rows: at 203M precursors an array indexed by it would be gigabytes for a few million
/// rows. So index directly while the span is no larger than the row count, and otherwise
/// intern, which costs one hash per row but bounds the array by the number of groups. The
/// interning is a permutation of the group labels, and the reduction's result does not
/// depend on which integer names a group, so the two arms agree row for row
/// (`grouped_q_is_the_same_on_a_dense_and_a_sparse_key_space`).
///
/// The gate is `span <= n`, not a multiple of it, because the array is 24 bytes per SLOT
/// and the alternative is not free but is bounded by the ROW count. At n rows the direct
/// arm costs `24 * span`; the interning arm costs `4n` for the ids plus `24k` for the k
/// distinct groups (k <= n), with its own `HashMap<u32, u32>` (9 bytes per bucket over
/// next_pow2(8n/7) buckets, about 18n) live only while it runs -- so at most about 28n
/// bytes, and at the six-run Astral pool (n = 3,133,636) 76 MB against the direct arm's
/// 24 * span. The two cross at span ~ 1.2n. An earlier `4n` gate admitted span up to
/// 12.5M there, i.e. 301 MB, which is worse than both the interning arm AND the
/// `HashMap<u32, GroupBest>` this rewrite replaced (4,194,304 buckets x 33 bytes = 138 MB)
/// -- and `base_peptide_id` against a 6-12M base-peptide library lands precisely in that
/// band. Under `span <= n` the direct arm is at most 24n, which is inside the interning
/// arm's own bound and saves it one hash per row.
fn dense_group_ids(keys: &[u32]) -> (std::borrow::Cow<'_, [u32]>, usize) {
    let Some(&max) = keys.iter().max() else {
        return (std::borrow::Cow::Borrowed(keys), 0);
    };
    let span = max as usize + 1;
    // `.max(1024)` so a handful of rows with a scattered key space does not build a hash
    // table to save 24 KB.
    if span <= keys.len().max(1024) {
        return (std::borrow::Cow::Borrowed(keys), span);
    }
    let mut interner: HashMap<u32, u32> = HashMap::with_capacity(keys.len().min(span));
    let ids: Vec<u32> = keys
        .iter()
        .map(|&k| {
            let next = interner.len() as u32;
            *interner.entry(k).or_insert(next)
        })
        .collect();
    let span = interner.len();
    (std::borrow::Cow::Owned(ids), span)
}

/// Reduce PSMs to the best score per group key, compute group q-values against
/// the selected null, and map back to a per-PSM vector. Mirrors the PSM-level
/// logic for peptide- and protein-group-level q.
///
/// The reduction indexes a `Vec<Option<GroupBest>>` rather than hashing, because two of
/// the three calls have a group count close to the row count and were building the
/// largest hash table in the stage to reproduce their own input: measured on
/// out_hye/psms_competed.parquet, 746,772 distinct `base_peptide_id` and 879,018 distinct
/// (peptidoform, charge) over 879,027 rows, against 69,958 distinct proteins.
fn grouped_q(
    keys: &[u32],
    scores: &[f64],
    is_decoy: &[bool],
    is_entrapment: &[bool],
    is_real: &[bool],
    qmode: QMode,
    ratio: f64,
) -> Vec<f64> {
    let n = scores.len();
    let (ids, span) = dense_group_ids(keys);
    // Keep the winning row index with each picked group. Exact target/null score
    // ties go to the active null (decoy or entrapment) so input row order cannot
    // make the accepted set anti-conservative.
    let mut best: Vec<Option<GroupBest>> = vec![None; span];
    // Counted here rather than recovered from the collect below, so `picked` can be sized
    // exactly: see the comment on its allocation.
    let mut n_groups = 0usize;
    for i in 0..n {
        // In entrapment mode the in-silico decoys are not the null, so they must not
        // compete for the group. A target and its paired decoy SHARE `base_peptide_id`
        // by construction (`make_shift_decoys.py`, `make_reverse_decoys.py` and the
        // native `peptidoforms.rs` all copy it), so a decoy that outscored its target
        // won the group -- and its tuple is `is_entrapment = false, is_real = false`, so
        // it counted toward neither the entrapment nor the real population and the group
        // vanished from the analysis entirely, while the real target was assigned 1.0.
        // Both `target_peptides_at_1pct` and the entrapment leak metric are computed
        // from this q, so the FDR-validity instrument was measured on a population that
        // decoys had partially deleted, and the leak count was under-reported.
        // Skipped rows keep the default 1.0 below, which is right: a decoy has no
        // meaningful entrapment-calibrated group q.
        if matches!(qmode, QMode::Entrapment) && is_decoy[i] {
            continue;
        }
        // `get_or_insert` is `HashMap::entry(..).or_insert(..)`: the placeholder score is
        // NEG_INFINITY, so the comparison below promotes the first row of a group unless
        // its own score is NEG_INFINITY or NaN, in which case the tuple already holds
        // that row's flags and index and is left alone. Same arithmetic, same outcome.
        let slot = &mut best[ids[i] as usize];
        n_groups += usize::from(slot.is_none());
        let e = slot.get_or_insert((
            f64::NEG_INFINITY,
            is_decoy[i],
            is_entrapment[i],
            is_real[i],
            i,
        ));
        let incoming_null = match qmode {
            QMode::Decoy => is_decoy[i],
            QMode::Entrapment => is_entrapment[i],
        };
        let current_null = match qmode {
            QMode::Decoy => e.1,
            QMode::Entrapment => e.2,
        };
        if scores[i] > e.0 || (scores[i] == e.0 && incoming_null && !current_null) {
            *e = (scores[i], is_decoy[i], is_entrapment[i], is_real[i], i);
        }
    }
    // The picked groups, in ascending key order. The hash version read them out of
    // `HashMap::keys()` and then probed the map four more times per group; this walk is
    // one pass, and it also removes a dependency on `HashMap` iteration order, which the
    // project bans even where it is harmless (it was harmless: a group's q depends only
    // on the multiset of (score, is_null), because every member of a tied block is
    // assigned the same qmin and the totals are order-free).
    //
    // `with_capacity(n_groups)` + `extend`, NOT `collect`. `Flatten`'s `size_hint` lower
    // bound is 0 -- `Option<T>` is not a `ConstSizeIntoIterator`, only arrays are -- so
    // `collect` grows from nothing by doubling and lands on next_pow2(n_groups) slots:
    // verified with rustc -O, `len = 3,133,636` gave `capacity = 4,194,304`, 100.7 MB of
    // allocation for 75.2 MB of groups, and the final doubling holds the 50.3 MB
    // predecessor as well while `best` is still owned by the `IntoIter`. That made the
    // COLLECT the peak of this function (226 MB at the Astral pool, 882 MB at 11.6M rows)
    // rather than the kernel call below. Sized exactly it is 75.2 + 75.2 = 150 MB there,
    // and the peak moves back to the kernel.
    let mut picked: Vec<GroupBest> = Vec::with_capacity(n_groups);
    picked.extend(best.into_iter().flatten());
    debug_assert_eq!(picked.len(), n_groups);
    let qv = match qmode {
        // Read in place, not through an `n_groups * 16` byte pair buffer.
        QMode::Decoy => target_decoy_q_by(picked.len(), |k| (picked[k].0, picked[k].1)),
        QMode::Entrapment => {
            let sc: Vec<f64> = picked.iter().map(|g| g.0).collect();
            let e: Vec<bool> = picked.iter().map(|g| g.2).collect();
            let r: Vec<bool> = picked.iter().map(|g| g.3).collect();
            entrapment_q(&sc, &e, &r, ratio)
        }
    };
    // Assign the group q ONLY to the picked winning row of each group. A
    // losing sibling (a lower-scoring charge/mod variant, which may itself be a
    // false target) must not inherit the winner's low q; it gets 1.0. The
    // report/counts dedup by key on the winner, so peptide/PG counts are
    // unchanged, but per-PSM peptide_q/pg_q no longer propagate to losers.
    let mut out = vec![1.0f64; n];
    for (g, q) in picked.iter().zip(qv) {
        out[g.4] = q;
    }
    out
}

/// Run the entrapment GBM sidecar: write a Parquet of features + meta columns,
/// invoke `entrapment_worker.py`, read back candidate_id + score. Positives are
/// real targets, negatives are spike-in (entrapment) targets; the worker fits a
/// gradient-boosted classifier out-of-fold by base peptide (the positional-CLI
/// file contract in docs/13_sidecars.md, as for the MS2PIP/DeepLC/Mokapot
/// sidecars).
#[allow(clippy::too_many_arguments)]
fn run_entrapment_gbm(
    p: &RescoreParams,
    feat_names: &[String],
    cid: &[u32],
    base: &[u32],
    is_entrapment: &[bool],
    is_decoy: &[bool],
    feats: &FeatureMatrix,
) -> Result<Vec<f64>> {
    let python = p
        .cfg
        .python
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("classifier=entrapment GBM requires rescore.python"))?;
    std::fs::create_dir_all(p.work_dir).ok();
    // Per-invocation names; see the note in `sidecar::run_ms2pip`.
    let pid = std::process::id();
    let inp = format!("{}/entrapment_in_{pid}.parquet", p.work_dir);
    let outp = format!("{}/entrapment_out_{pid}.parquet", p.work_dir);

    let mut cols = vec![
        // Unique per-row id for score readback: candidate_id repeats across
        // competed runs, so mapping scores back by candidate_id collides (later
        // runs overwrite earlier). Map by this flat row index instead.
        Col::U32("row_id".into(), (0..cid.len()).map(|i| i as u32).collect()),
        Col::U32("candidate_id".into(), cid.to_vec()),
        Col::U32("base_peptide_id".into(), base.to_vec()),
        Col::I32(
            "is_entrapment".into(),
            is_entrapment.iter().map(|&b| b as i32).collect(),
        ),
        Col::I32(
            "is_decoy".into(),
            is_decoy.iter().map(|&b| b as i32).collect(),
        ),
    ];
    for (fi, name) in feat_names.iter().enumerate() {
        cols.push(Col::F64(
            name.clone(),
            (0..cid.len()).map(|i| feats.row(i)[fi] as f64).collect(),
        ));
    }
    check_sidecar_space(
        Some(python),
        p.work_dir,
        cid.len(),
        feat_names.len(),
        HandoffFormat::ParquetF64,
    )?;
    write_table(&inp, cols)?;

    let script = crate::sidecar::resolve_script(p.script_dir, "entrapment_worker.py");
    let status = std::process::Command::new(python)
        .arg(&script)
        .arg(&inp)
        .arg(&outp)
        .arg(p.cfg.folds.to_string())
        .env("PYTHONUTF8", "1")
        .status()?;
    if !status.success() {
        anyhow::bail!("entrapment worker exited with {status}");
    }

    let t = TableFile::open(&outp)?;
    let orid = t.u32("row_id")?;
    let osc = t.f64("score")?;
    let aligned = align_sidecar_scores(&orid, &osc, cid.len(), "entrapment_worker")?;
    drop(t);
    remove_sidecar_files(&[&inp, &outp], keep_handoff());
    Ok(aligned)
}

/// Owns a spawned sidecar child and kills it on drop unless it was already awaited.
/// Without this, a `mumdia` process that dies mid-rescore (Ctrl-C, kill, panic) leaves
/// the Python worker alive holding its feature memmap, and every later rescore then fails
/// on a file it cannot delete.
struct ChildGuard(Option<std::process::Child>);

impl ChildGuard {
    fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        let mut c = self.0.take().expect("child awaited twice");
        c.wait()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Write the sidecar feature table as Parquet, streaming a batch at a time.
///
/// Same logical contract as the tab-separated PIN - `SpecId`, `Label`, `ScanNr`, `ExpMass`,
/// `CalcMass`, the feature columns, `Peptide`, `Proteins` - so the worker reads either format
/// from the same column names.
///
/// Features are `f32`: the TSV wrote `{:.6}` and the worker casts to float32 anyway, so f32
/// matches what is actually used, halves the file, and does not silently increase precision
/// relative to the validated TSV reference.
///
/// Batched because `feats` is already resident (one flat row-major matrix, 8 bytes per
/// value); materialising 387 full columns as well would add ~12.8 GB for nothing.
/// The feature matrix, for the paths that read it.
///
/// It is absent in exactly two situations, both of them `rescore.strict` with a PIN
/// sidecar: it was streamed past (never built) or released once the worker had its own copy
/// on disk. Strict has no native fallback, so neither situation leaves a reader; every
/// caller of this function is on a non-strict or native path, where the matrix was built.
fn kept_matrix(slot: &Option<FeatureMatrix>) -> &FeatureMatrix {
    slot.as_ref().expect(
        "the feature matrix is absent only under rescore.strict with a sidecar classifier, \
         which has no native fallback",
    )
}

/// ~250k rows x 387 f32 is about 390 MB per batch, which keeps the encoder's working set
/// modest.
///
/// The staging layout says a smaller block should be faster and the end-to-end measurement
/// says nothing at all, so the constant stays where it is for want of a reason to move it.
///
/// `flush_block` stages rows row-major and then gathers 387 columns back out at a
/// 1,548-byte stride, so every element read costs a 64-byte line, and making the block
/// L2-resident really does make THAT loop cheaper. `handoff_block_size_end_to_end`
/// (262,144 rows x 387 f32, release, s/Mrow, MINIMUM over three repetitions in each of
/// three whole runs, on a machine that was carrying other work):
///
/// ```text
///   block     end to end   transpose only
///   250,000        5.99         2.32
///   131,072        6.03         1.51
///    65,536        5.06          -
///    16,384        5.03         1.32
///     4,096        5.51         0.35
///     1,024        6.50         0.18
/// ```
///
/// The transpose in isolation is strongly block-dependent, reproduces in every run, and is
/// 13x cheaper at 1,024 than at 250,000. End to end it does not survive: the six arms span
/// 5.03-6.50 with no monotone trend, and the spread of one arm across whole runs (up to
/// 1.45x; 131,072 measured 6.03, 7.33 and 8.72) is larger than the spread between arms
/// (1.29x). The transpose is real and it is not the binding cost; the parquet encoder is.
///
/// This does NOT reproduce an earlier reading of the same benchmark that had 250,000 at
/// 5.3 s/Mrow and 4,096 at 9.2, i.e. 1.7x slower, and the difference is not the benchmark's
/// arm bias: the pre-fix benchmark (every arm paying a 250,000-row staging reservation
/// before `with_block_rows` replaced it) run on this machine in the same session gives
/// 250,000 at 7.06 and 4,096 at 7.05. Treat the block size as unmeasured rather than
/// settled, and do not quote a ratio from one run of this benchmark.
///
/// The memory is 387 MB of staged f32 PLUS the columnar copy `flush_block` gathers out of
/// it -- all 387 `Vec<f32>` are live until `RecordBatch::try_new` -- so the transient is
/// ~774 MB, not 387, against ~12.7 MB at 4,096. That is a real difference and the reason
/// to revisit this; it is not at the measured peak, which is why nobody has. Under
/// `rescore.strict` with `nn_torch` there is no feature matrix at handoff time and the
/// engine sits near 1.4 GB, while the 9.3 GB process-tree peak happens later, inside the
/// worker. Moving the constant also has to answer an output question this note cannot:
/// the batch size is not the row-group size, but nothing here pins that the handoff's
/// BYTES are independent of how rows are fed to the arrow writer.
const HANDOFF_BATCH_ROWS: usize = 250_000;
/// Row groups are capped well below the batch: the worker reads this file back with
/// `ParquetFile.iter_batches`, which decodes a whole row group before it slices batches out
/// of it, so at parquet's default 1,048,576-row groups the load transient was 1.6 GB of
/// Arrow buffers per group on top of the matrix. 131,072 rows x 387 f32 is 200 MB.
const HANDOFF_ROW_GROUP_ROWS: usize = 131_072;

/// The parquet encoding's state. Boxed in `HandoffSink` because the arrow writer it owns
/// dwarfs a `BufWriter`.
struct ParquetHandoff {
    schema: std::sync::Arc<arrow::datatypes::Schema>,
    writer: mumdia_io::table::BatchWriter,
    /// Row-major staging buffer for the current block, transposed to columns on flush
    /// ([`HandoffWriter::push_row`], the matrix path).
    stage: Vec<f32>,
    /// Column-major staging for the current block, one vector per feature, moved into the
    /// batch on flush without a transpose ([`HandoffWriter::push_batch`], the stream).
    /// A block is staged one way or the other, never both.
    cols: Vec<Vec<f32>>,
    /// Rows staged in `cols`.
    col_rows: usize,
    block_start: usize,
}

enum HandoffSink {
    Parquet(Box<ParquetHandoff>),
    Pin(std::io::BufWriter<std::fs::File>),
}

/// The sidecar handoff file, written one row at a time.
///
/// Same logical contract in both encodings - `SpecId`, `Label`, `ScanNr`, `ExpMass`,
/// `CalcMass`, the feature columns, `Peptide`, `Proteins` - so the worker reads either
/// format from the same column names. Parquet applies to `nn_torch` only; `mokapot_worker`
/// goes through `mokapot.read_pin()`, which requires the tab-separated form.
///
/// Features are `f32` in the parquet: the TSV writes `{:.6}` and the worker casts to
/// float32 anyway, so f32 matches what is actually used, halves the file, and does not
/// silently increase precision relative to the validated TSV reference.
///
/// Row-at-a-time, rather than a function over a resident matrix, because the source is
/// either that matrix or the competed parquet itself (`for_each_feature_row`) and neither
/// caller should hold a second copy. The parquet blocks are the same
/// `HANDOFF_BATCH_ROWS`-row blocks written before, with the same values in the same order,
/// so the file is unchanged.
struct HandoffWriter<'a> {
    sink: HandoffSink,
    /// Rows per parquet batch. `HANDOFF_BATCH_ROWS` outside tests.
    block_rows: usize,
    /// Rows accepted so far, so `finish` reports what was written rather than what the
    /// caller's metadata columns happen to be long.
    rows: u64,
    /// Time spent in `flush_block` (transpose, batch build, encode), for the phase log.
    encode: std::time::Duration,
    feat_names: &'a [String],
    /// `label`, reduced to the bit the PIN's `Label` column and the parquet's need.
    is_decoy: &'a [bool],
    pform: &'a FlatStr,
    protein: &'a FlatStr,
    mz: &'a [f64],
}

impl<'a> HandoffWriter<'a> {
    fn new(
        paths: &SidecarPaths,
        feat_names: &'a [String],
        is_decoy: &'a [bool],
        pform: &'a FlatStr,
        protein: &'a FlatStr,
        mz: &'a [f64],
    ) -> Result<HandoffWriter<'a>> {
        Self::with_block_size(
            paths,
            feat_names,
            is_decoy,
            pform,
            protein,
            mz,
            HANDOFF_BATCH_ROWS,
        )
    }

    /// [`HandoffWriter::new`] with the parquet block size named, so the staging buffer is
    /// reserved at that size ONCE.
    ///
    /// Production always goes through `new` at `HANDOFF_BATCH_ROWS`. This exists for
    /// `handoff_block_size_end_to_end`: a benchmark arm at N has to allocate what a build
    /// with `HANDOFF_BATCH_ROWS = N` would allocate. The staging buffers are reserved when
    /// the first block is staged, at the block size of the writer.
    #[allow(clippy::too_many_arguments)]
    fn with_block_size(
        paths: &SidecarPaths,
        feat_names: &'a [String],
        is_decoy: &'a [bool],
        pform: &'a FlatStr,
        protein: &'a FlatStr,
        mz: &'a [f64],
        block_rows: usize,
    ) -> Result<HandoffWriter<'a>> {
        use arrow::datatypes::{DataType, Field, Schema};
        use std::io::Write as _;
        use std::sync::Arc;

        let sink = if paths.use_pq {
            let mut fields: Vec<Field> = vec![
                Field::new("SpecId", DataType::Utf8, false),
                Field::new("Label", DataType::Int32, false),
                Field::new("ScanNr", DataType::Int32, false),
                Field::new("ExpMass", DataType::Float64, false),
                Field::new("CalcMass", DataType::Float64, false),
            ];
            for n in feat_names {
                fields.push(Field::new(n, DataType::Float32, false));
            }
            fields.push(Field::new("Peptide", DataType::Utf8, false));
            fields.push(Field::new("Proteins", DataType::Utf8, false));
            let schema = Arc::new(Schema::new(fields));
            let writer = mumdia_io::table::BatchWriter::with_row_group_rows(
                &paths.handoff,
                schema.clone(),
                HANDOFF_ROW_GROUP_ROWS,
            )?;
            HandoffSink::Parquet(Box::new(ParquetHandoff {
                schema,
                writer,
                // Reserved lazily, by whichever of `push_row` and `push_batch` stages the
                // first block, so the path that is not used allocates nothing.
                stage: Vec::new(),
                cols: vec![Vec::new(); feat_names.len()],
                col_rows: 0,
                block_start: 0,
            }))
        } else {
            // Streamed through a BufWriter. It was previously accumulated in ONE
            // un-reserved String: at ~1M rows x 387 features that is a >5 GB allocation
            // (plus realloc churn) held entirely in RAM before the first byte reaches disk.
            let mut w =
                std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&paths.handoff)?);
            w.write_all(b"SpecId\tLabel\tScanNr\tExpMass\tCalcMass\t")?;
            w.write_all(feat_names.join("\t").as_bytes())?;
            w.write_all(b"\tPeptide\tProteins\n")?;
            HandoffSink::Pin(w)
        };
        Ok(HandoffWriter {
            sink,
            block_rows: block_rows.max(1),
            rows: 0,
            encode: std::time::Duration::ZERO,
            feat_names,
            is_decoy,
            pform,
            protein,
            mz,
        })
    }

    /// Shrink the parquet batch so a test can exercise several blocks. Production always
    /// uses `HANDOFF_BATCH_ROWS`.
    ///
    /// Call it before the first row: the staging buffers are reserved at the block size in
    /// force when the first block is staged. The benchmark arms still go through
    /// [`HandoffWriter::with_block_size`], which is what a build with another
    /// `HANDOFF_BATCH_ROWS` would construct.
    #[cfg(test)]
    fn with_block_rows(mut self, rows: usize) -> Self {
        self.block_rows = rows.max(1);
        self
    }

    /// Append flat row `i`. Callers push rows in ascending order from 0.
    ///
    /// SpecId / ScanNr key on the row index, NOT candidate_id: candidate_id is the library
    /// index and repeats across runs, so an experiment-wide table would collide.
    fn push_row(&mut self, i: usize, values: &[f32]) -> Result<()> {
        use std::io::Write as _;
        let nf = self.feat_names.len();
        self.rows += 1;
        match &mut self.sink {
            HandoffSink::Parquet(pq) => {
                debug_assert_eq!(pq.col_rows, 0, "a block is staged by rows or by columns");
                if pq.stage.capacity() == 0 {
                    pq.stage
                        .reserve_exact(self.block_rows.saturating_mul(nf.max(1)));
                }
                pq.stage.extend_from_slice(&values[..nf]);
                if pq.stage.len() >= self.block_rows.saturating_mul(nf.max(1)) {
                    self.flush_block()?;
                }
                Ok(())
            }
            HandoffSink::Pin(w) => {
                let lab = if self.is_decoy[i] { -1 } else { 1 };
                write!(
                    w,
                    "psm_{}\t{}\t{}\t{:.5}\t{:.5}\t",
                    i, lab, i, self.mz[i], self.mz[i]
                )?;
                for v in values.iter().take(nf) {
                    write!(w, "{:.6}\t", v)?;
                }
                writeln!(w, "-.{}.-\t{}", self.pform.get(i), self.protein.get(i))?;
                Ok(())
            }
        }
    }

    /// Append a decoded batch of the feature stream, whose first row is the next flat row.
    ///
    /// The parquet encoding stages it column by column: each feature's values are narrowed
    /// straight out of the decoded Arrow column into that feature's block vector, in
    /// parallel over features, and the block is handed to the writer without the row-major
    /// round trip `push_row` makes (decoded columns to rows, rows back to columns on
    /// flush). Blocks still hold `block_rows` rows and are flushed at the same rows, so the
    /// writer receives the same record batches and the file is byte-identical
    /// (`the_streamed_handoff_is_byte_identical_to_the_handoff_from_the_matrix`). The PIN is
    /// written a row at a time, as before.
    fn push_batch(&mut self, b: &FeatureBatch<'_>) -> Result<()> {
        if !matches!(self.sink, HandoffSink::Parquet(_)) {
            let mut row = vec![0.0f32; self.feat_names.len()];
            for k in 0..b.rows {
                b.row_into(k, &mut row);
                self.push_row(b.first_row + k, &row)?;
            }
            return Ok(());
        }
        let block_rows = self.block_rows;
        let mut lo = 0usize;
        while lo < b.rows {
            let HandoffSink::Parquet(pq) = &mut self.sink else {
                unreachable!("checked above");
            };
            debug_assert!(
                pq.stage.is_empty(),
                "a block is staged by rows or by columns"
            );
            let take = (block_rows - pq.col_rows).min(b.rows - lo);
            let hi = lo + take;
            pq.cols.par_iter_mut().enumerate().for_each(|(j, col)| {
                if col.capacity() == 0 {
                    col.reserve_exact(block_rows);
                }
                b.extend_column(j, lo, hi, col);
            });
            pq.col_rows += take;
            self.rows += take as u64;
            lo = hi;
            if pq.col_rows >= block_rows {
                self.flush_block()?;
            }
        }
        Ok(())
    }

    /// Write the staged block as one record batch: the columns staged by `push_batch` are
    /// moved in as they are, rows staged by `push_row` are transposed first.
    fn flush_block(&mut self) -> Result<()> {
        use arrow::array::{ArrayRef, Float32Array, Int32Array};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let nf = self.feat_names.len();
        let (is_decoy, pform, protein, mz) = (self.is_decoy, self.pform, self.protein, self.mz);
        let HandoffSink::Parquet(pq) = &mut self.sink else {
            return Ok(());
        };
        if pq.stage.is_empty() && pq.col_rows == 0 {
            return Ok(());
        }
        let t = std::time::Instant::now();
        let start = pq.block_start;
        let k = if pq.col_rows > 0 {
            pq.col_rows
        } else {
            pq.stage.len() / nf.max(1)
        };
        let end = start + k;
        let features: Vec<Vec<f32>> = if pq.col_rows > 0 {
            // Moved out, and re-reserved lazily by the next block.
            pq.cols.iter_mut().map(std::mem::take).collect()
        } else {
            transpose_block(&pq.stage, nf, k)
        };
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(nf + 7);
        // One text buffer per block for each of the two formatted columns rather than one
        // `String` per row. The values are the same, so the array and the file are too.
        arrays.push(Arc::new(string_column(k, 12, |r, s| {
            use std::fmt::Write as _;
            let _ = write!(s, "psm_{}", start + r);
        })?));
        arrays.push(Arc::new(Int32Array::from(
            (start..end)
                .map(|i| if is_decoy[i] { -1 } else { 1 })
                .collect::<Vec<_>>(),
        )));
        arrays.push(Arc::new(Int32Array::from(
            (start..end).map(|i| i as i32).collect::<Vec<_>>(),
        )));
        // `ExpMass` and `CalcMass` are the same values: one array in both slots, which
        // writes the same bytes as two copies (`a_shared_column_array_writes_the_same_
        // parquet_as_two_copies`).
        let mzv: ArrayRef = Arc::new(Float64Array::from(mz[start..end].to_vec()));
        arrays.push(mzv.clone());
        arrays.push(mzv);
        for col in features {
            arrays.push(Arc::new(Float32Array::from(col)));
        }
        arrays.push(Arc::new(string_column(k, 28, |r, s| {
            s.push_str("-.");
            s.push_str(pform.get(start + r));
            s.push_str(".-");
        })?));
        // `StringArray::from(Vec<String>)` is `from_iter_values`, so the same bytes.
        arrays.push(Arc::new(arrow::array::StringArray::from_iter_values(
            (start..end).map(|i| protein.get(i)),
        )));
        pq.writer
            .write(&RecordBatch::try_new(pq.schema.clone(), arrays)?)?;
        pq.stage.clear();
        pq.col_rows = 0;
        pq.block_start = end;
        self.encode += t.elapsed();
        Ok(())
    }

    /// Finish the file and return the rows written.
    fn finish(self) -> Result<u64> {
        Ok(self.finish_timed()?.0)
    }

    /// [`HandoffWriter::finish`], also returning the time spent transposing and encoding
    /// the parquet blocks, footer included.
    fn finish_timed(mut self) -> Result<(u64, std::time::Duration)> {
        use std::io::Write as _;
        self.flush_block()?;
        let t = std::time::Instant::now();
        let rows = match self.sink {
            HandoffSink::Parquet(pq) => pq.writer.close()?,
            HandoffSink::Pin(mut w) => {
                w.flush()?;
                self.rows
            }
        };
        Ok((rows, self.encode + t.elapsed()))
    }
}

/// The `k` rows of a row-major block of `nf` features as one vector per feature.
///
/// In parallel over tiles of 16 features: a tile's task reads each staged row once, as 64
/// contiguous bytes, and appends one value to each of its 16 columns, where one task per
/// feature re-read every row's cache line for every feature. Each value is copied to its
/// own column in row order, so the columns are the serial loop's exactly.
fn transpose_block(stage: &[f32], nf: usize, k: usize) -> Vec<Vec<f32>> {
    const TILE: usize = 16;
    let tiles: Vec<Vec<Vec<f32>>> = (0..nf.div_ceil(TILE))
        .into_par_iter()
        .map(|t| {
            let c0 = t * TILE;
            let w = TILE.min(nf - c0);
            let mut out: Vec<Vec<f32>> = (0..w).map(|_| Vec::with_capacity(k)).collect();
            for r in 0..k {
                let row = &stage[r * nf + c0..r * nf + c0 + w];
                for (col, &v) in out.iter_mut().zip(row) {
                    col.push(v);
                }
            }
            out
        })
        .collect();
    tiles.into_iter().flatten().collect()
}

/// A `k`-row string column whose row `r` is what `write(r, buf)` appends, built in one
/// text buffer (about `bytes_per_row` bytes a row reserved up front) and one offsets
/// vector, rather than one `String` per row.
fn string_column(
    k: usize,
    bytes_per_row: usize,
    mut write: impl FnMut(usize, &mut String),
) -> Result<arrow::array::StringArray> {
    use arrow::buffer::{Buffer, OffsetBuffer, ScalarBuffer};
    let mut values = String::with_capacity(k.saturating_mul(bytes_per_row));
    let mut offsets: Vec<i32> = Vec::with_capacity(k + 1);
    offsets.push(0);
    for r in 0..k {
        write(r, &mut values);
        offsets.push(
            i32::try_from(values.len())
                .map_err(|_| anyhow!("a handoff string column passed 2 GiB in one block"))?,
        );
    }
    Ok(arrow::array::StringArray::try_new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Buffer::from(values.into_bytes()),
        None,
    )?)
}

/// Stream the selected feature columns of every competed input, one row at a time, in flat
/// row order, narrowed to f32 exactly as `FeatureMatrix::push` narrows them.
///
/// One pass over just the feature columns. The null policy is unchanged: a null f64 reads
/// as NaN, which the validation then rejects.
///
/// The stage itself reads through [`for_each_feature_batch`], which hands whole decoded
/// batches to the handoff and the matrix; this row form is what the tests and the
/// benchmark compare every consumer against.
#[cfg(test)]
fn for_each_feature_row(
    competed: &[String],
    feat_names: &[String],
    f: impl FnMut(usize, &[f32]) -> Result<()>,
) -> Result<()> {
    // Every feature column of every row: the widest read of the stage, so one forward read
    // per row group (`wide_scan_options`); the batches are the plain reader's.
    let scan = super::wide_scan_options();
    for_each_feature_row_with(
        competed,
        feat_names,
        &scan,
        |t| feature_batch_rows(t, &scan),
        f,
    )
}

/// [`for_each_feature_row`] under explicit read options and batch sizing, for the
/// benchmark that compares them.
#[cfg(test)]
fn for_each_feature_row_with(
    competed: &[String],
    feat_names: &[String],
    scan: &mumdia_io::table::ScanOptions,
    batch_rows: impl Fn(&TableFile) -> usize,
    mut f: impl FnMut(usize, &[f32]) -> Result<()>,
) -> Result<()> {
    let mut row: Vec<f32> = vec![0.0; feat_names.len()];
    for_each_feature_batch_with(competed, feat_names, scan, batch_rows, |b| {
        for k in 0..b.rows {
            b.row_into(k, &mut row);
            f(b.first_row + k, &row)?;
        }
        Ok(())
    })
}

/// One decoded batch of the feature stream: the selected feature columns, in selection
/// order, for the flat rows `first_row..first_row + rows`.
///
/// Every value leaves it narrowed exactly as `FeatureMatrix::push` narrows it, `v as f32`,
/// with a null read as `f64::NAN as f32` (the same expression the matrix path narrowed
/// through, so a null cell keeps its bits). The validation then rejects it as non-finite.
struct FeatureBatch<'a> {
    first_row: usize,
    rows: usize,
    cols: Vec<&'a Float64Array>,
}

impl FeatureBatch<'_> {
    /// Value `k` (batch-local row) of selected column `j`, narrowed.
    #[inline]
    fn value(&self, j: usize, k: usize) -> f32 {
        let c = self.cols[j];
        if c.is_null(k) {
            f64::NAN as f32
        } else {
            c.value(k) as f32
        }
    }

    /// Batch-local row `k` into `row` (one value per selected column).
    fn row_into(&self, k: usize, row: &mut [f32]) {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = self.value(j, k);
        }
    }

    /// Batch-local rows `lo..lo + dst.len() / nf` into `dst`, row-major, in parallel over
    /// row chunks. The values are the ones `row_into` produces; only the order in which
    /// they are written differs, and each lands in its own slot.
    fn rows_into(&self, lo: usize, dst: &mut [f32]) {
        let nf = self.cols.len();
        if nf == 0 {
            return;
        }
        const CHUNK_ROWS: usize = 1024;
        dst.par_chunks_mut(CHUNK_ROWS * nf)
            .enumerate()
            .for_each(|(c, chunk)| {
                let r0 = lo + c * CHUNK_ROWS;
                for (r, row) in chunk.chunks_exact_mut(nf).enumerate() {
                    self.row_into(r0 + r, row);
                }
            });
    }

    /// Batch-local rows `lo..hi` of column `j`, narrowed, appended to `out`.
    fn extend_column(&self, j: usize, lo: usize, hi: usize, out: &mut Vec<f32>) {
        let c = self.cols[j];
        if c.null_count() == 0 {
            out.extend(c.values()[lo..hi].iter().map(|&v| v as f32));
        } else {
            out.extend((lo..hi).map(|k| self.value(j, k)));
        }
    }

    /// `(flat row, column, value)` of the first non-finite narrowed value in row-major order:
    /// the lowest row, and in it the lowest column, which is what the row-at-a-time scan
    /// found with `values.iter().position(..)` on the first offending row. Columns are
    /// searched in parallel; the minimum over `(row, column)` does not depend on which
    /// worker finds what.
    fn first_non_finite(&self) -> Option<(usize, usize, f32)> {
        (0..self.cols.len())
            .into_par_iter()
            .filter_map(|j| {
                (0..self.rows)
                    .find(|&k| !self.value(j, k).is_finite())
                    .map(|k| (k, j))
            })
            .min()
            .map(|(k, j)| (self.first_row + k, j, self.value(j, k)))
    }
}

/// Stream the selected feature columns of every competed input one decoded batch at a
/// time, in flat row order: the column form of [`for_each_feature_row`], for consumers
/// that can take a batch whole (the parquet handoff stages it column by column, the matrix
/// fills it row-major in parallel).
fn for_each_feature_batch(
    competed: &[String],
    feat_names: &[String],
    f: impl FnMut(&FeatureBatch<'_>) -> Result<()>,
) -> Result<()> {
    let scan = super::wide_scan_options();
    for_each_feature_batch_with(
        competed,
        feat_names,
        &scan,
        |t| feature_batch_rows(t, &scan),
        f,
    )
}

fn for_each_feature_batch_with(
    competed: &[String],
    feat_names: &[String],
    scan: &mumdia_io::table::ScanOptions,
    batch_rows: impl Fn(&TableFile) -> usize,
    mut f: impl FnMut(&FeatureBatch<'_>) -> Result<()>,
) -> Result<()> {
    let names: Vec<&str> = feat_names.iter().map(String::as_str).collect();
    let mut flat = 0usize;
    for path in competed {
        let t = TableFile::open(path)?;
        let reader = t.scan(Some(&names), batch_rows(&t), scan)?;
        let sch = reader.schema();
        let order: Vec<usize> = feat_names
            .iter()
            .map(|n| {
                sch.index_of(n)
                    .map_err(|_| anyhow!("competed table {path} has no feature column '{n}'"))
            })
            .collect::<Result<_>>()?;
        for b in reader {
            let b = b?;
            let cols: Vec<&Float64Array> = order
                .iter()
                .map(|&i| {
                    b.column(i)
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| {
                            anyhow!("feature column '{}' is not f64", sch.field(i).name())
                        })
                })
                .collect::<Result<_>>()?;
            let batch = FeatureBatch {
                first_row: flat,
                rows: b.num_rows(),
                cols,
            };
            f(&batch)?;
            flat += batch.rows;
        }
    }
    Ok(())
}

/// The engine's own copy of the feature values: one contiguous f32 buffer.
fn load_feature_matrix(
    competed: &[String],
    feat_names: &[String],
    total_rows: usize,
) -> Result<FeatureMatrix> {
    let nf = feat_names.len();
    let mut m = FeatureMatrix::with_capacity(total_rows, nf);
    // One row-major block per batch, filled in parallel and appended. The block is reused,
    // so after the first batch it costs no allocation; the values are the ones
    // `for_each_feature_row` hands out, in the same rows.
    let mut block: Vec<f32> = Vec::new();
    for_each_feature_batch(competed, feat_names, |b| {
        block.clear();
        block.resize(b.rows * nf, 0.0);
        b.rows_into(0, &mut block);
        m.push_row(&block);
        Ok(())
    })?;
    m.finish()
}

/// Report the first non-finite input value, whichever kind of column it is in.
///
/// The validation these two answers come from used to be one serial row loop that checked a
/// row's features before its scalars and stopped at the first offending ROW. Both scans run
/// separately now (one of them inside the handoff stream), so the choice between them is
/// made here, on the row index, to keep the message identical.
fn bail_non_finite(
    bad_feature: Option<(usize, usize, f32)>,
    bad_scalar: Option<usize>,
    feat_names: &[String],
) -> Result<()> {
    if let Some((row, feature, value)) = bad_feature {
        if bad_scalar.is_none_or(|scalar_row| row <= scalar_row) {
            anyhow::bail!(
                "rescore input contains non-finite feature '{}' at flat row {row}: {value}",
                feat_names[feature]
            );
        }
    }
    if let Some(row) = bad_scalar {
        anyhow::bail!(
            "rescore input contains non-finite prelim_score/precursor_mz at flat row {row}"
        );
    }
    Ok(())
}

/// The directory the rescore sidecar files go to: `MUMDIA_SIDECAR_DIR` when it is set and
/// not empty, else `default` (the orchestrators pass `<out-dir>/sidecar_work`, the
/// standalone `mumdia rescore` passes `sidecar_work` unless `--work-dir` names one).
///
/// An environment variable rather than a configuration field, so moving the files, for
/// instance onto a RAM-backed directory or a disk with room, does not change the
/// configuration hash every artifact records. The NN worker puts its streaming memmap next
/// to its output file, so the memmap moves too.
pub fn sidecar_work_dir(default: &str) -> String {
    match std::env::var("MUMDIA_SIDECAR_DIR") {
        Ok(d) if !d.trim().is_empty() => d,
        _ => default.to_string(),
    }
}

/// Whether `MUMDIA_KEEP_HANDOFF` asks to keep the sidecar files once the scores are read
/// back (`1`, `true`, `yes` or `on`).
fn keep_handoff() -> bool {
    env_flag(std::env::var("MUMDIA_KEEP_HANDOFF").ok().as_deref())
}

/// `1`, `true`, `yes` or `on`, in any case, is set; anything else, or unset, is not.
fn env_flag(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Remove the files of one sidecar invocation once its scores are aligned, unless
/// `keep` is set. Returns the bytes removed.
///
/// Every invocation names its files after its output and its PID (`sidecar_paths`), so
/// nothing ever reused or removed them: they piled up in the work directory, 7.7 GB per
/// HYE rescore and 359 GB per immunopeptidomics pool for the handoff alone. They are
/// removed only after `align_sidecar_scores` has accepted the worker's output, so a failed
/// worker still leaves its input behind for a rerun or a look at it; `MUMDIA_KEEP_HANDOFF=1`
/// keeps them on success too. A file that cannot be removed is reported and does not fail
/// the stage, which has its scores.
fn remove_sidecar_files(files: &[&str], keep: bool) -> u64 {
    if keep {
        info!(
            files = ?files,
            "rescore: kept the sidecar files (MUMDIA_KEEP_HANDOFF)"
        );
        return 0;
    }
    let mut freed = 0u64;
    for f in files {
        let bytes = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
        match std::fs::remove_file(f) {
            Ok(()) => freed += bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => warn!(file = %f, error = %e, "rescore: could not remove a sidecar file"),
        }
    }
    info!(
        freed = %human_bytes(freed as f64),
        files = files.len(),
        "rescore: removed the sidecar files (set MUMDIA_KEEP_HANDOFF=1 to keep them)"
    );
    freed
}

/// How the feature handoff is encoded, for the space estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
enum HandoffFormat {
    Pin,
    Parquet,
    /// The entrapment worker's input: f64 features in a parquet table.
    ParquetF64,
}

/// `(floor, estimate)` bytes of a sidecar invocation's files for `rows` PSMs and `nf`
/// features: the handoff, the fold keys (4 bytes a row) and the worker's output (~20).
///
/// The floor is what the handoff cannot be smaller than, so less free space than that is
/// certain to fail; the estimate is its usual size. The PIN writes every value as `{:.6}`
/// plus a tab, never fewer than 9 bytes. The f32 parquet handoff measured 0.72 of the raw
/// f32 size on HYE (1.01 GB for 879,018 x 387) and 0.87 on the immunopeptidomics pool
/// (359 GB for 258.75M rows), so half the raw size is taken as its floor and the raw size
/// as its estimate.
fn handoff_space(rows: u64, nf: u64, format: HandoffFormat) -> (u64, u64) {
    let cells = rows.saturating_mul(nf);
    let (floor, estimate) = match format {
        HandoffFormat::Pin => (cells.saturating_mul(9), cells.saturating_mul(11)),
        HandoffFormat::Parquet => (cells.saturating_mul(2), cells.saturating_mul(4)),
        HandoffFormat::ParquetF64 => (cells.saturating_mul(4), cells.saturating_mul(8)),
    };
    // SpecId, Label, ScanNr, ExpMass, CalcMass, Peptide, Proteins; then keys and output.
    let per_row = rows.saturating_mul(48 + 4 + 20);
    (
        floor.saturating_add(per_row),
        estimate.saturating_add(per_row),
    )
}

/// Free bytes on the filesystem that holds `dir`, asked of the sidecar's own interpreter
/// (`shutil.disk_usage`): the standard library has no portable free-space call, and the
/// interpreter is the one thing every sidecar run already requires. `None` when it cannot
/// be asked, which skips the check rather than failing a run the sidecar would then fail
/// anyway, with its own clearer error.
fn free_bytes(python: &str, dir: &str) -> Option<u64> {
    let out = std::process::Command::new(python)
        .args([
            "-c",
            "import shutil, sys; print(shutil.disk_usage(sys.argv[1]).free)",
            dir,
        ])
        .env("PYTHONUTF8", "1")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// Refuse a sidecar run whose work directory cannot hold its files, before a byte of them
/// is written (`MUMDIA_SIDECAR_SPACE_CHECK=0` skips the check).
///
/// Without it a pooled rescore wrote the handoff until the disk filled, hundreds of GB and
/// hours into the stage, and failed there with a bare `No space left on device`.
fn check_sidecar_space(
    python: Option<&str>,
    dir: &str,
    rows: usize,
    nf: usize,
    format: HandoffFormat,
) -> Result<()> {
    if std::env::var("MUMDIA_SIDECAR_SPACE_CHECK")
        .is_ok_and(|v| matches!(v.trim(), "0" | "off" | "false" | "no"))
    {
        return Ok(());
    }
    let (floor, estimate) = handoff_space(rows as u64, nf as u64, format);
    let Some(free) = python.and_then(|py| free_bytes(py, dir)) else {
        tracing::debug!(
            dir,
            "rescore: free space unknown; skipping the sidecar space check"
        );
        return Ok(());
    };
    space_verdict(free, floor, estimate, dir, rows, nf, format)
}

/// The decision of [`check_sidecar_space`] once the free space is known.
fn space_verdict(
    free: u64,
    floor: u64,
    estimate: u64,
    dir: &str,
    rows: usize,
    nf: usize,
    format: HandoffFormat,
) -> Result<()> {
    if free < floor {
        anyhow::bail!(
            "rescore: the sidecar work directory {dir} has {} free, and the {format:?} \
             handoff of {rows} PSMs x {nf} features needs at least {} ({} expected) with \
             the fold keys and the worker's output. Point MUMDIA_SIDECAR_DIR (or `mumdia \
             rescore --work-dir`) at a directory with room, rescore fewer runs per \
             invocation, or set MUMDIA_SIDECAR_SPACE_CHECK=0 to skip this check.",
            human_bytes(free as f64),
            human_bytes(floor as f64),
            human_bytes(estimate as f64)
        );
    }
    if free < estimate {
        warn!(
            dir,
            free = %human_bytes(free as f64),
            expected = %human_bytes(estimate as f64),
            "rescore: the sidecar work directory may be too small for the handoff \
             (MUMDIA_SIDECAR_DIR moves it)"
        );
    }
    Ok(())
}

/// The per-invocation sidecar file paths, and which handoff encoding they name.
struct SidecarPaths {
    /// The feature handoff: a parquet for `nn_torch`, the tab-separated PIN otherwise.
    handoff: String,
    out: String,
    foldkeys: String,
    use_pq: bool,
}

impl SidecarPaths {
    /// The directory the files are in.
    fn dir(&self) -> String {
        std::path::Path::new(&self.handoff)
            .parent()
            .map(|d| d.to_string_lossy().into_owned())
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| ".".to_string())
    }

    fn format(&self) -> HandoffFormat {
        if self.use_pq {
            HandoffFormat::Parquet
        } else {
            HandoffFormat::Pin
        }
    }

    /// Every file of the invocation, for the cleanup.
    fn files(&self) -> Vec<&str> {
        vec![&self.handoff, &self.foldkeys, &self.out]
    }
}

/// Per-invocation sidecar filenames. Fixed names (`rescore.pin`,
/// `rescore_sidecar_out.parquet`) made two concurrent rescores clobber each other and let a
/// killed run's orphaned Python worker hold `*.feat.mm` open forever, so every later
/// rescore failed with `OSError: [Errno 22]` on a path it did not own. Keying on the output
/// artifact plus the PID makes collisions impossible.
fn sidecar_paths(p: &RescoreParams, script_name: &str) -> SidecarPaths {
    std::fs::create_dir_all(p.work_dir).ok();
    let tag = format!(
        "{}_{}",
        std::path::Path::new(p.out)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("rescore"),
        std::process::id()
    );
    // Parquet applies to nn_torch only: mokapot_worker.py goes through
    // `mokapot.read_pin()`, which requires the tab-separated form. Falling back with a
    // warning beats failing a configured run.
    let want_pq = matches!(p.cfg.handoff, mumdia_core::config::Handoff::Parquet);
    let use_pq = want_pq && script_name.contains("nn_rescore");
    if want_pq && !use_pq {
        tracing::warn!(
            script = script_name,
            "rescore.handoff=parquet is supported only by the nn_torch sidecar (mokapot reads \
             a PIN through mokapot.read_pin); writing the tab-separated PIN instead"
        );
    }
    let handoff = if use_pq {
        format!("{}/rescore_{tag}.features.parquet", p.work_dir)
    } else {
        format!("{}/rescore_{tag}.pin", p.work_dir)
    };
    SidecarPaths {
        handoff,
        out: format!("{}/rescore_{tag}_out.parquet", p.work_dir),
        foldkeys: format!("{}/rescore_{tag}.foldkeys.parquet", p.work_dir),
        use_pq,
    }
}

/// Run a PIN-contract Python rescorer sidecar (`mokapot_worker.py` or
/// `nn_rescore_worker.py`) over a PIN written from the competed set; return scores
/// aligned to the input candidate order (the file contract in
/// docs/13_sidecars.md). Both sidecars share this exact contract: PIN in,
/// `candidate_id`+`score` parquet out.
///
/// `prewritten` is `Some` when the handoff was already streamed straight from the competed
/// parquet, which is what happens under `rescore.strict` (see `run`); then there is no
/// engine-side matrix and `feats` is `None`. Otherwise the handoff is written here, from
/// the matrix, and released afterwards if strict allows it.
#[allow(clippy::too_many_arguments)]
fn run_pin_sidecar(
    p: &RescoreParams,
    script_name: &str,
    feat_names: &[String],
    cid: &[u32],
    is_decoy: &[bool],
    pform: &FlatStr,
    protein: &FlatStr,
    mz: &[f64],
    feats: &mut Option<FeatureMatrix>,
    base: &[u32],
    prewritten: Option<SidecarPaths>,
) -> Result<Vec<f64>> {
    let python = p.cfg.python.as_deref().ok_or_else(|| {
        anyhow::anyhow!("classifier sidecar {script_name} requires rescore.python")
    })?;
    let t_handoff = Instant::now();
    let paths = match prewritten {
        Some(paths) => paths,
        None => {
            let paths = sidecar_paths(p, script_name);
            check_sidecar_space(
                Some(python),
                &paths.dir(),
                cid.len(),
                feat_names.len(),
                paths.format(),
            )?;
            let matrix = feats.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "rescore: the feature matrix was released before the {script_name} handoff"
                )
            })?;
            // Key the handoff on the unique row index i (SpecId=psm_i, ScanNr=i), NOT
            // candidate_id: candidate_id is the library index and repeats across runs, so
            // an experiment-wide (multi-file) PIN would collide on ScanNr and mokapot's
            // per-spectrum competition would collapse the runs. The row index is unique
            // across the whole concatenation. Single-run behaviour is unchanged (the
            // mapping is bijective and mokapot does not use SpecId/ScanNr as features).
            let mut w = HandoffWriter::new(&paths, feat_names, is_decoy, pform, protein, mz)?;
            for i in 0..cid.len() {
                w.push_row(i, matrix.row(i))?;
            }
            let rows = w.finish()?;
            tracing::info!(
                path = %paths.handoff,
                rows,
                features = feat_names.len(),
                parquet = paths.use_pq,
                "rescore: wrote the sidecar feature table"
            );
            paths
        }
    };
    let (pin, outp) = (&paths.handoff, &paths.out);

    // Cross-validation fold keys, row-aligned to the PIN.
    //
    // The NN worker derived its fold from `md5(strip_pep(Peptide))`, and `strip_pep`
    // removes bracketed mods and flanking residues but not the `DECOY_` marker, so a
    // target and its paired decoy landed in DIFFERENT folds -- while `percolator_lite`
    // keys on `base_peptide_id` and pairs them, and docs/11 claimed the two used "the
    // same CV-fold scheme". Stripping the prefix would only fix the shift-decoy library:
    // a reverse decoy's peptidoform is the reversed sequence, so no string derived from
    // it can reach its target. `base_peptide_id` is the pairing both builders preserve
    // (`dprec = tprec.copy()`), so pass it explicitly.
    //
    // By environment variable rather than argv: the sidecar contract is positional and
    // shared with `mokapot_worker.py`, and the NN hyperparameters already travel this way.
    // A worker that does not read it simply keeps its previous behaviour.
    let foldkeys = &paths.foldkeys;
    write_table(foldkeys, vec![Col::U32("fold_key".into(), base.to_vec())])?;
    let handoff_ms = t_handoff.elapsed().as_millis();

    // Everything the worker reads is on disk now. Under `strict` a sidecar failure is an
    // error rather than a fall back to native_tda, so nothing downstream reads the engine's
    // copy of the matrix again: release it before the child starts instead of holding it
    // idle beside the worker's own for the whole training run. Measured on the six-run
    // Astral pool (3.13M x 387): the engine sat at 5.3 GB of a 17.9 GB process-tree peak.
    // Under strict the matrix is usually never built at all (the handoff was streamed),
    // and then there is nothing to release.
    if p.cfg.strict {
        if let Some(bytes) = feats.as_ref().map(|m| m.bytes()) {
            *feats = None;
            tracing::info!(
                released = %human_bytes(bytes as f64),
                "rescore: released the engine's feature matrix for the sidecar run"
            );
        }
    }

    let script = crate::sidecar::resolve_script(p.script_dir, script_name);
    // Spawn (not `status()`) so the child handle is owned by a guard that kills it if we
    // unwind or are dropped: a killed `mumdia` used to leave the Python worker running,
    // holding its multi-GB memmap open, which made every subsequent rescore fail on a
    // stale lock with no hint about which PID held it.
    let child = std::process::Command::new(python)
        .arg(&script)
        .arg(pin)
        .arg(outp)
        .env("PYTHONUTF8", "1")
        // Pass the configured NN hyperparameters so the worker uses them instead
        // of its own defaults, and so the folds/num_iter/train_fdr recorded in the
        // report reflect the values actually used
        // (docs/18_findings_and_decisions.md). Ignored by mokapot_worker.py,
        // which shares this PIN contract. `NN_ENV_SET_BY_ENGINE` lists these names.
        .env("MUMDIA_NN_FOLDS", p.cfg.folds.to_string())
        .env("MUMDIA_NN_ITERS", p.cfg.num_iter.to_string())
        .env("MUMDIA_NN_TRAIN_FDR", p.cfg.train_fdr.to_string())
        // Training-set reduction. Passed unconditionally so the report and the worker
        // agree on what ran; the defaults (0 / random) are the worker's own, so an
        // unconfigured run behaves exactly as before.
        .env("MUMDIA_NN_NEG_RATIO", p.cfg.train_neg_ratio.to_string())
        .env(
            "MUMDIA_NN_NEG_SELECT",
            match p.cfg.train_neg_select {
                mumdia_core::config::NegSelect::Random => "random",
                mumdia_core::config::NegSelect::Margin => "margin",
                mumdia_core::config::NegSelect::Hybrid => "hybrid",
            },
        )
        .env("MUMDIA_NN_TRAIN_SUB", p.cfg.train_subsample.to_string())
        .env(
            "MUMDIA_NN_WARM_START",
            if p.cfg.train_warm_epochs > 0 {
                "1"
            } else {
                "0"
            },
        )
        .env("MUMDIA_NN_WARM_EPOCHS", p.cfg.train_warm_epochs.to_string())
        .env("MUMDIA_NN_MARGIN_FRAC", p.cfg.train_margin_frac.to_string())
        .env("MUMDIA_NN_SEEDS", p.cfg.seeds.max(1).to_string())
        .env("MUMDIA_NN_FOLD_KEYS", foldkeys)
        .spawn()
        .map_err(|e| {
            // A bare `.spawn()?` reported only "No such file or directory (os error 2)" with
            // no indication of WHICH path was missing - the usual failure when a config moves
            // between machines.
            anyhow::anyhow!("spawning sidecar failed: {python} {script}: {e}")
        })?;
    let t_worker = Instant::now();
    let mut guard = ChildGuard(Some(child));
    let status = guard.wait()?;
    if !status.success() {
        anyhow::bail!("{script_name} exited with {status}");
    }
    let worker_ms = t_worker.elapsed().as_millis();
    let t_read = Instant::now();

    // The worker echoes the PIN's SpecId tail as `candidate_id`, which here is the
    // flat row index. Exact, unique, finite coverage is part of the classifier
    // contract: silently assigning a worst score to missing rows changes the
    // trained population and can invalidate sensitivity/FDR comparisons.
    let t = TableFile::open(outp)?;
    let orow = t.u32("candidate_id")?;
    let osc = t.f64("score")?;
    let aligned = align_sidecar_scores(&orow, &osc, cid.len(), script_name)?;
    drop(t);
    info!(
        script = script_name,
        handoff_and_fold_keys_ms = handoff_ms as u64,
        worker_ms = worker_ms as u64,
        read_scores_ms = t_read.elapsed().as_millis() as u64,
        "rescore: sidecar timings"
    );
    // The scores are in memory and validated; the files are not read again.
    remove_sidecar_files(&paths.files(), keep_handoff());
    Ok(aligned)
}

/// The `MUMDIA_NN_*` variables `run_pin_sidecar` sets on the worker itself. Their values
/// are the configured ones, which the report's `params` already records. Keep this list in
/// step with the `.env(...)` calls there.
const NN_ENV_SET_BY_ENGINE: [&str; 11] = [
    "MUMDIA_NN_FOLDS",
    "MUMDIA_NN_ITERS",
    "MUMDIA_NN_TRAIN_FDR",
    "MUMDIA_NN_NEG_RATIO",
    "MUMDIA_NN_NEG_SELECT",
    "MUMDIA_NN_TRAIN_SUB",
    "MUMDIA_NN_WARM_START",
    "MUMDIA_NN_WARM_EPOCHS",
    "MUMDIA_NN_MARGIN_FRAC",
    "MUMDIA_NN_SEEDS",
    "MUMDIA_NN_FOLD_KEYS",
];

/// The `MUMDIA_NN_*` variables in `vars` that the NN worker inherits from this process on
/// top of the ones `run_pin_sidecar` sets, sorted by name.
///
/// Several of them change the scores: `MUMDIA_NN_SEED`, `MUMDIA_NN_THREADS` (which
/// `--threads` sets), `MUMDIA_NN_PARALLEL` (the keyed epoch shuffle). They reach the
/// worker only through the environment, so without this record two runs of one config
/// could report different identifications with nothing in `psms_scored.parquet.report.json`
/// to say why. A name or value that is not valid Unicode is kept lossily; the worker
/// could not read such a value as a number either.
fn inherited_nn_env<I>(vars: I) -> std::collections::BTreeMap<String, String>
where
    I: IntoIterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
{
    vars.into_iter()
        .map(|(k, v)| {
            // Windows names are case-insensitive, and Python's `os.environ` upper-cases
            // them, so the worker reads `mumdia_nn_seed` as `MUMDIA_NN_SEED`.
            let k = k.to_string_lossy();
            let k = if cfg!(windows) {
                k.to_ascii_uppercase()
            } else {
                k.into_owned()
            };
            (k, v.to_string_lossy().into_owned())
        })
        .filter(|(k, _)| k.starts_with("MUMDIA_NN_") && !NN_ENV_SET_BY_ENGINE.contains(&k.as_str()))
        .collect()
}

/// Validate and align a sidecar's `(flat_row_id, score)` response. Every input
/// row must occur exactly once, there may be no extras, and all scores must be
/// finite. This is shared by PIN and entrapment sidecars.
fn align_sidecar_scores(
    row_ids: &[u32],
    scores: &[f64],
    expected_rows: usize,
    sidecar: &str,
) -> Result<Vec<f64>> {
    if row_ids.len() != scores.len() || row_ids.len() != expected_rows {
        anyhow::bail!(
            "{sidecar} output coverage mismatch: expected {expected_rows} rows, \
             got {} ids and {} scores",
            row_ids.len(),
            scores.len()
        );
    }
    let mut aligned = vec![0.0f64; expected_rows];
    let mut seen = vec![false; expected_rows];
    for (&row_id, &score) in row_ids.iter().zip(scores) {
        let row = row_id as usize;
        if row >= expected_rows {
            anyhow::bail!(
                "{sidecar} returned out-of-range row id {row_id}; expected 0..{expected_rows}"
            );
        }
        if seen[row] {
            anyhow::bail!("{sidecar} returned duplicate row id {row_id}");
        }
        if !score.is_finite() {
            anyhow::bail!("{sidecar} returned non-finite score for row id {row_id}: {score}");
        }
        seen[row] = true;
        aligned[row] = score;
    }
    if let Some(missing) = seen.iter().position(|present| !present) {
        anyhow::bail!("{sidecar} did not return a score for row id {missing}");
    }
    Ok(aligned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdr::target_decoy_q;

    #[test]
    fn inherited_nn_env_records_only_the_knobs_the_engine_does_not_set() {
        use std::ffi::OsString;
        let vars = [
            ("MUMDIA_NN_SEED", "7"),
            ("PATH", "/usr/bin"),
            ("MUMDIA_NN_FOLDS", "5"),
            ("MUMDIA_NN_PARALLEL", "3"),
            ("MUMDIA_RESCORE_MODEL", "nn"),
            ("MUMDIA_NN_FOLD_KEYS", "/tmp/keys.parquet"),
        ]
        .map(|(k, v)| (OsString::from(k), OsString::from(v)));
        let got = inherited_nn_env(vars);
        let want: Vec<(&str, &str)> = vec![("MUMDIA_NN_PARALLEL", "3"), ("MUMDIA_NN_SEED", "7")];
        assert_eq!(
            got.iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect::<Vec<_>>(),
            want,
            "sorted by name, the engine's own variables left out"
        );
        assert!(inherited_nn_env(Vec::new()).is_empty());
    }

    #[test]
    fn nn_env_set_by_engine_lists_every_variable_run_pin_sidecar_sets() {
        // The list decides what the report calls inherited, so it must name exactly the
        // `MUMDIA_NN_*` literals of `run_pin_sidecar`'s `.env(...)` calls.
        let src = include_str!("rescore.rs");
        let start = src.find("\nfn run_pin_sidecar(").expect("run_pin_sidecar");
        let body = &src[start..];
        // Its closing brace is the first one in column 0 (the checkout may use CRLF).
        // `\x7d` is that brace, spelled as an escape: `ci/gen_config_reference.py` counts
        // the braces of a `#[cfg(test)]` module without masking string literals, and a
        // bare one here would end its blanking of this module early.
        let body = &body[..body.find("\n\x7d").expect("end of run_pin_sidecar")];
        let mut set: Vec<&str> = body
            .match_indices("\"MUMDIA_NN_")
            .map(|(i, _)| {
                let lit = &body[i + 1..];
                &lit[..lit.find('"').expect("closing quote")]
            })
            .collect();
        set.sort_unstable();
        set.dedup();
        let mut listed = NN_ENV_SET_BY_ENGINE.to_vec();
        listed.sort_unstable();
        assert_eq!(set, listed);
    }

    fn cfg_with(features: Option<Vec<&str>>, file: Option<&str>) -> RescoreConfig {
        RescoreConfig {
            features: features.map(|v| v.into_iter().map(String::from).collect()),
            features_file: file.map(String::from),
            feature_preset: FeaturePreset::All,
            ..Default::default()
        }
    }

    #[test]
    fn compact_preset_is_114_unique_names_and_intersects_the_schema() {
        let names = preset_names(FeaturePreset::Compact).unwrap();
        assert_eq!(names.len(), 114);
        let uniq: std::collections::HashSet<&String> = names.iter().collect();
        assert_eq!(
            uniq.len(),
            114,
            "duplicate name in the embedded compact list"
        );
        let cfg = RescoreConfig {
            feature_preset: FeaturePreset::Compact,
            ..cfg_with(None, None)
        };
        // Schema order is kept and names the table lacks are skipped, not fatal.
        let avail: Vec<String> = ["zz", "rt_error_abs", "coelution_run", "yy"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            resolve_feature_subset(&cfg, &avail).unwrap(),
            vec!["rt_error_abs".to_string(), "coelution_run".to_string()]
        );
        let none: Vec<String> = vec!["zz".to_string()];
        let e = resolve_feature_subset(&cfg, &none).unwrap_err();
        assert!(e.to_string().contains("shares no column"), "{e}");
        // An explicit list wins over the preset.
        let explicit = RescoreConfig {
            feature_preset: FeaturePreset::Compact,
            ..cfg_with(Some(vec!["zz"]), None)
        };
        assert_eq!(
            resolve_feature_subset(&explicit, &avail).unwrap(),
            vec!["zz".to_string()]
        );
    }

    #[test]
    fn feature_subset_projects_in_schema_order_and_defaults_to_all() {
        let avail: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        // No selection: every column, untouched.
        assert_eq!(
            resolve_feature_subset(&cfg_with(None, None), &avail).unwrap(),
            avail
        );
        // A selection is a projection: schema order wins over the order asked for, and
        // duplicates collapse, because the sidecar contract is positional.
        let got =
            resolve_feature_subset(&cfg_with(Some(vec!["d", "a", "d"]), None), &avail).unwrap();
        assert_eq!(got, vec!["a".to_string(), "d".to_string()]);
    }

    #[test]
    fn feature_subset_rejects_unknown_names_and_bad_configuration() {
        let avail: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let e = resolve_feature_subset(&cfg_with(Some(vec!["a", "zz"]), None), &avail).unwrap_err();
        assert!(format!("{e}").contains("zz"), "{e}");
        let e = resolve_feature_subset(&cfg_with(Some(vec![]), None), &avail).unwrap_err();
        assert!(format!("{e}").contains("empty"), "{e}");
        let e =
            resolve_feature_subset(&cfg_with(Some(vec!["a"]), Some("f.txt")), &avail).unwrap_err();
        assert!(format!("{e}").contains("mutually exclusive"), "{e}");
    }

    #[test]
    fn feature_subset_reads_a_list_file_ignoring_blanks_and_comments() {
        let dir = std::env::temp_dir().join("mumdia_fs_subset_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("feats_{}.txt", std::process::id()));
        std::fs::write(
            &path,
            "# picked by docs/28

b
  c  # trailing note
",
        )
        .unwrap();
        let avail: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let got =
            resolve_feature_subset(&cfg_with(None, Some(path.to_str().unwrap())), &avail).unwrap();
        assert_eq!(got, vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn merge_col_moves_the_first_input_and_appends_the_rest() {
        // One input: the destination IS the reader's vector, not a copy of it. Checked on
        // the pointer, because "moved rather than copied" is the whole claim.
        let src = vec![String::from("a"), String::from("b")];
        let ptr = src.as_ptr();
        let mut dst: Vec<String> = Vec::new();
        merge_col(&mut dst, src, 2);
        assert_eq!(dst.as_ptr(), ptr);
        assert_eq!(dst, vec!["a".to_string(), "b".to_string()]);
        // Several inputs: the concatenation is in input order, and the reserve from the
        // footer row count means the appends do not realloc.
        let mut dst: Vec<u32> = Vec::new();
        merge_col(&mut dst, vec![1, 2], 5);
        let after_first = dst.as_ptr();
        merge_col(&mut dst, vec![3], 5);
        merge_col(&mut dst, vec![4, 5], 5);
        assert_eq!(dst, vec![1, 2, 3, 4, 5]);
        assert_eq!(
            dst.as_ptr(),
            after_first,
            "the reserve should have sufficed"
        );
        // An empty first input must not lose the rows of the second.
        let mut dst: Vec<u32> = Vec::new();
        merge_col(&mut dst, Vec::new(), 2);
        merge_col(&mut dst, vec![7, 8], 2);
        assert_eq!(dst, vec![7, 8]);
    }

    /// A competed-shaped parquet: the named feature columns as nullable f64, values keyed
    /// on the row so the concatenation order is checkable, and (when `nulls`) one null per
    /// five rows, which the reader must present as NaN.
    fn crafted_competed(path: &str, rows: usize, tag: f64, feat_names: &[String], nulls: bool) {
        let cols: Vec<Col> = feat_names
            .iter()
            .enumerate()
            .map(|(j, name)| {
                let v: Vec<Option<f64>> = (0..rows)
                    .map(|i| {
                        if nulls && j == 1 && i % 5 == 0 {
                            None
                        } else {
                            Some(tag + i as f64 + j as f64 * 0.125)
                        }
                    })
                    .collect();
                Col::OptF64(name.clone(), v)
            })
            .collect();
        write_table(path, cols).unwrap();
    }

    /// One input's worth of `TableFile::str_flat` output, for the tests that feed
    /// `FlatStr` directly.
    fn flat_parts<S: AsRef<str>>(v: &[S]) -> (Vec<usize>, String) {
        let mut offsets = vec![0usize];
        let mut data = String::new();
        for s in v {
            data.push_str(s.as_ref());
            offsets.push(data.len());
        }
        (offsets, data)
    }

    fn flat<S: AsRef<str>>(v: &[S]) -> FlatStr {
        let mut f = FlatStr::default();
        f.merge(flat_parts(v), v.len());
        f
    }

    #[test]
    fn flat_columns_concatenate_and_gather_like_a_vec_of_strings() {
        // `FlatStr::merge` is the flat counterpart of `merge_col` and is the part of the
        // layout easiest to get subtly wrong: the second input's offsets have to be
        // rebased onto the end of the first input's text. Pinned against what the
        // `Vec<String>` columns did, over the same inputs, including the empty-first-input
        // case `merge_col_moves_the_first_input_and_appends_the_rest` covers.
        let a: Vec<String> = vec!["PEPTIDEK".into(), "".into(), "MKK[+42]R".into()];
        let b: Vec<String> = vec!["ELVIS".into(), "LIVES".into()];
        let total = a.len() + b.len();

        let mut want: Vec<String> = Vec::new();
        merge_col(&mut want, a.clone(), total);
        merge_col(&mut want, b.clone(), total);

        let mut got = FlatStr::default();
        got.merge(flat_parts(&a), total);
        got.merge(flat_parts(&b), total);
        assert_eq!(got.len(), want.len());
        assert_eq!(got.iter().collect::<Vec<_>>(), want, "concatenated rows");
        assert_eq!(got.iter().len(), want.len(), "the iterator is exact-sized");

        // An empty first input must not lose the rows of the second, and must not leave
        // the offsets without their leading zero.
        let mut empty_first = FlatStr::default();
        empty_first.merge(flat_parts::<String>(&[]), 2);
        empty_first.merge(flat_parts(&b), 2);
        assert_eq!(empty_first.iter().collect::<Vec<_>>(), b);

        // `gather` is the top-K collapse's `keep_rows!`, which cloned one `String` per
        // kept row.
        let keep = vec![4usize, 2, 0];
        let gathered = got.gather(&keep);
        let cloned: Vec<String> = keep.iter().map(|&i| want[i].clone()).collect();
        assert_eq!(gathered.iter().collect::<Vec<_>>(), cloned);
        assert_eq!(gathered.len(), 3);
        assert_eq!(
            gathered.data.capacity(),
            gathered.data.len(),
            "gather sizes its text buffer from the kept rows rather than doubling into it"
        );
    }

    #[test]
    fn the_flat_column_reports_capacity_and_shrink_returns_it() {
        // `bytes()` feeds `meta_bytes` and therefore `memlog`, so it has to report what is
        // ALLOCATED. `TableFile::str_flat` builds the text buffer from `String::new()` and
        // appends one value at a time, so it arrives with up to 2x the text it holds in
        // unused capacity; a `bytes()` that reported `len` would under-report exactly the
        // resident cost the flat layout is justified on, and `shrink` is what makes the
        // documented per-row figure a resident figure instead of a lower bound.
        let v: Vec<String> = (0..10_000).map(|i| format!("PEPTIDEK[+16]{i}")).collect();
        let text: usize = v.iter().map(|s| s.len()).sum();
        let mut f = flat(&v);
        assert!(
            f.data.capacity() > text,
            "the row-at-a-time read should leave growth slack: capacity {} for {text} bytes",
            f.data.capacity()
        );
        assert_eq!(
            f.bytes(),
            f.offsets.capacity() * std::mem::size_of::<usize>() + f.data.capacity(),
            "bytes() is capacity, not length"
        );
        let before = f.bytes();
        f.shrink();
        assert!(f.data.capacity() >= text && f.data.capacity() < before);
        assert!(f.bytes() < before);
        assert_eq!(f.iter().collect::<Vec<_>>(), v, "shrink moved no value");
    }

    #[test]
    fn the_flat_metadata_columns_write_the_same_scored_parquet() {
        // The output-equality claim of the flat/bit metadata columns: `psms_scored` must
        // be byte-for-byte the file the `Vec<String>` columns produced. Both arms are
        // written by `write_scored_table` itself, one from the flat columns and one from
        // a `ScoredColumns` whose string arrays are rebuilt the old way, so what is
        // compared is the array construction and nothing else.
        //
        // `label` is the part that could not be reproduced from a bool if the input were
        // not exactly {"target", "decoy"}; the check that guarantees it is pinned
        // separately by `an_unknown_label_is_still_refused_by_its_value`.
        let n = 300usize;
        let dir = std::env::temp_dir().join("mumdia_scored_flat_test");
        std::fs::create_dir_all(&dir).unwrap();
        let pid = std::process::id();
        let pform: Vec<String> = (0..n).map(|i| format!("PEPTIDEK[+16]{i}")).collect();
        let protein: Vec<String> = (0..n)
            .map(|i| format!("sp|P{:05}|PROT_HUMAN;sp|Q{:05}|ALT_HUMAN", i % 37, i % 11))
            .collect();
        let is_decoy: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();

        let columns = |pform: FlatStr, protein: FlatStr| ScoredColumns {
            cid: (0..n as u32).collect(),
            pform,
            charge: (0..n).map(|i| 2 + (i % 3) as i32).collect(),
            is_decoy: is_decoy.clone(),
            protein,
            base: (0..n as u32).map(|i| i / 2).collect(),
            apex_rt: (0..n).map(|i| i as f64 * 0.5).collect(),
            elution_lo: (0..n).map(|i| i as f64 * 0.5 - 1.0).collect(),
            elution_hi: (0..n).map(|i| i as f64 * 0.5 + 1.0).collect(),
            scores: (0..n).map(|i| (i % 29) as f64 / 29.0).collect(),
            psm_q: (0..n).map(|i| (i % 97) as f64 / 97.0).collect(),
            peptide_q: (0..n).map(|i| (i % 53) as f64 / 53.0).collect(),
            pg_q: (0..n).map(|i| (i % 41) as f64 / 41.0).collect(),
            prelim: (0..n).map(|i| i as f64).collect(),
            source: vec![0; n],
            run_psm_q: (0..n).map(|i| (i % 89) as f64 / 89.0).collect(),
            precursor_q: (0..n).map(|i| (i % 61) as f64 / 61.0).collect(),
            peak_rank: vec![0; n],
        };

        let from_flat = dir.join(format!("flat_{pid}.parquet"));
        let rows = write_scored_table(
            from_flat.to_str().unwrap(),
            columns(flat(&pform), flat(&protein)),
        )
        .unwrap();
        assert_eq!(rows, n as u64);

        // The previous construction: `StringArray::from(Vec<String>)` over the same
        // values, including the label text the bit stands for.
        let label: Vec<String> = is_decoy
            .iter()
            .map(|&d| if d { "decoy" } else { "target" }.to_string())
            .collect();
        let from_vecs = dir.join(format!("vecs_{pid}.parquet"));
        {
            use arrow::array::{ArrayRef, Int32Array, StringArray, UInt32Array};
            use std::sync::Arc;
            let c = columns(flat(&pform), flat(&protein));
            let schema = scored_schema();
            let protein_a: ArrayRef = Arc::new(StringArray::from(protein.clone()));
            let q: ArrayRef = Arc::new(Float64Array::from(c.psm_q));
            let arrays: Vec<ArrayRef> = vec![
                Arc::new(UInt32Array::from(c.cid)),
                Arc::new(StringArray::from(pform.clone())),
                Arc::new(Int32Array::from(c.charge)),
                Arc::new(StringArray::from(label)),
                protein_a.clone(),
                Arc::new(UInt32Array::from(c.base)),
                Arc::new(Float64Array::from(c.apex_rt)),
                Arc::new(Float64Array::from(c.elution_lo)),
                Arc::new(Float64Array::from(c.elution_hi)),
                Arc::new(Float64Array::from(c.scores)),
                q.clone(),
                Arc::new(Float64Array::from(c.peptide_q)),
                protein_a,
                Arc::new(Float64Array::from(c.pg_q)),
                q.clone(),
                Arc::new(Float64Array::from(c.prelim)),
                Arc::new(UInt32Array::from(c.source)),
                Arc::new(Float64Array::from(c.run_psm_q)),
                q,
                Arc::new(Float64Array::from(c.precursor_q)),
                Arc::new(Int32Array::from(c.peak_rank)),
            ];
            let batch = arrow::record_batch::RecordBatch::try_new(schema.clone(), arrays).unwrap();
            let mut w =
                mumdia_io::table::BatchWriter::new(from_vecs.to_str().unwrap(), schema).unwrap();
            w.write(&batch).unwrap();
            w.close().unwrap();
        }
        assert_eq!(
            std::fs::read(&from_flat).unwrap(),
            std::fs::read(&from_vecs).unwrap(),
            "the flat metadata columns must write the identical psms_scored"
        );
    }

    #[test]
    fn a_scored_table_wider_than_one_batch_keeps_every_row_in_order() {
        // `write_scored_table` used to build one record batch over every row. An arrow
        // `StringArray` addresses its values with i32 offsets, so on a pooled table whose
        // concatenated `peptidoform` or `protein` text passes 2 GiB that construction
        // panics inside arrow with `offset overflow` -- which is how a 258,753,296-PSM
        // seven-run rescore died after its worker had finished, at the last write of a
        // 6.7-hour stage. The writer now emits `SCORED_CHUNK_ROWS` rows per batch, the
        // same 65,536 `write_table` uses, so the widest string buffer it ever builds is
        // one chunk of rows rather than the whole table.
        //
        // Two-GiB text is not testable here, so what this pins is the mechanism: more
        // than one batch, every row present, in order, across the boundary. Byte equality
        // with the single-batch file for a table SMALLER than a chunk is pinned by
        // `the_flat_metadata_columns_write_the_same_scored_parquet` above.
        let n = SCORED_CHUNK_ROWS + 7;
        let pform: Vec<String> = (0..n).map(|i| format!("PEPTIDE{i}")).collect();
        let protein: Vec<String> = (0..n).map(|i| format!("sp|P{i:07}|PROT")).collect();
        let c = ScoredColumns {
            cid: (0..n as u32).collect(),
            pform: flat(&pform),
            charge: vec![2; n],
            is_decoy: (0..n).map(|i| i % 2 == 0).collect(),
            protein: flat(&protein),
            base: (0..n as u32).collect(),
            apex_rt: (0..n).map(|i| i as f64).collect(),
            elution_lo: vec![0.0; n],
            elution_hi: vec![1.0; n],
            scores: (0..n).map(|i| i as f64).collect(),
            psm_q: vec![0.5; n],
            peptide_q: vec![0.5; n],
            pg_q: vec![0.5; n],
            prelim: vec![0.0; n],
            source: vec![0; n],
            run_psm_q: vec![0.5; n],
            precursor_q: vec![0.5; n],
            peak_rank: vec![0; n],
        };
        let path = scratch("chunked_scored.parquet");
        let rows = write_scored_table(&path, c).unwrap();
        assert_eq!(rows, n as u64);

        let t = mumdia_io::table::TableFile::open(&path).unwrap();
        let (off, txt) = t.str_flat("peptidoform").unwrap();
        assert_eq!(off.len(), n + 1, "one offset per row plus the terminator");
        let row = |i: usize| &txt[off[i]..off[i + 1]];
        // The chunk boundary itself, and the short final batch after it.
        assert_eq!(row(0), "PEPTIDE0");
        assert_eq!(
            row(SCORED_CHUNK_ROWS - 1),
            format!("PEPTIDE{}", SCORED_CHUNK_ROWS - 1)
        );
        assert_eq!(
            row(SCORED_CHUNK_ROWS),
            format!("PEPTIDE{SCORED_CHUNK_ROWS}")
        );
        assert_eq!(row(n - 1), format!("PEPTIDE{}", n - 1));
        let (poff, ptxt) = t.str_flat("protein").unwrap();
        assert_eq!(
            &ptxt[poff[n - 1]..poff[n]],
            format!("sp|P{:07}|PROT", n - 1)
        );
    }

    fn scratch(name: &str) -> String {
        let dir = std::env::temp_dir().join("mumdia_rescore_stream_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{}_{name}", std::process::id()))
            .to_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn streamed_feature_rows_are_the_matrix_rows() {
        // The streamed handoff never builds a `FeatureMatrix`, so the equality that has to
        // hold is between what `for_each_feature_row` yields and what the matrix would have
        // held: same values, same flat order across several inputs, same null policy.
        let names: Vec<String> = ["f0", "f1", "f2"].iter().map(|s| s.to_string()).collect();
        let a = scratch("cmp_a.parquet");
        let b = scratch("cmp_b.parquet");
        crafted_competed(&a, 7, 100.0, &names, false);
        crafted_competed(&b, 5, 900.0, &names, true);
        let competed = vec![a, b];

        let m = load_feature_matrix(&competed, &names, 12).unwrap();
        assert_eq!(m.rows(), 12);
        let mut seen: Vec<(usize, Vec<f32>)> = Vec::new();
        for_each_feature_row(&competed, &names, |i, v| {
            seen.push((i, v.to_vec()));
            Ok(())
        })
        .unwrap();
        assert_eq!(seen.len(), 12);
        for (i, values) in &seen {
            assert_eq!(*i, seen[*i].0, "flat row index must be dense and ascending");
            assert_eq!(
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                m.row(*i).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row {i}"
            );
        }
        // Row 7 is the first row of the second input, and its f1 is the null: NaN in both
        // paths, and the validation that runs inside the stream finds the same row the
        // matrix scan does.
        assert!(seen[7].1[1].is_nan());
        assert_eq!(m.find_non_finite(), Some((7, 1)));
        // The batch form's own scan finds it too, in the batch that holds it and in no
        // earlier one.
        let mut found = Vec::new();
        for_each_feature_batch(&competed, &names, |b| {
            found.push((b.first_row, b.first_non_finite()));
            Ok(())
        })
        .unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0], (0, None));
        assert_eq!(found[1].0, 7);
        let (row, col, v) = found[1].1.expect("the null is found");
        assert_eq!((row, col), (7, 1));
        assert!(v.is_nan());
    }

    #[test]
    fn the_first_non_finite_value_of_a_batch_is_the_row_major_first() {
        // Two offending cells in one batch: the earlier row wins whatever its column, and
        // within a row the lower column, as `values.iter().position(..)` on the first
        // offending row picked it. An f64 that overflows f32 counts, because the check is
        // on the narrowed value.
        let path = scratch("fnf.parquet");
        let big = f64::MAX;
        write_table(
            &path,
            vec![
                Col::F64("a".into(), vec![0.0, 1.0, 2.0, f64::NAN, 4.0]),
                Col::F64("b".into(), vec![0.0, 1.0, big, 3.0, 4.0]),
                Col::F64("c".into(), vec![0.0, 1.0, f64::INFINITY, 3.0, 4.0]),
            ],
        )
        .unwrap();
        let names: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let mut got = None;
        for_each_feature_batch(std::slice::from_ref(&path), &names, |b| {
            got = b.first_non_finite();
            Ok(())
        })
        .unwrap();
        let (row, col, v) = got.expect("found");
        assert_eq!((row, col), (2, 1));
        assert!(v.is_infinite(), "f64::MAX narrows to infinity: {v}");
        // And the row stream agrees.
        let mut first = None;
        for_each_feature_row(&[path], &names, |i, values| {
            if first.is_none() {
                if let Some(c) = values.iter().position(|v| !v.is_finite()) {
                    first = Some((i, c));
                }
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(first, Some((2, 1)));
    }

    #[test]
    fn the_matrix_filled_from_batches_is_the_matrix_filled_from_rows() {
        let names: Vec<String> = ["f0", "f1", "f2"].iter().map(|s| s.to_string()).collect();
        let a = scratch("mfb_a.parquet");
        let b = scratch("mfb_b.parquet");
        crafted_competed(&a, 2_500, 100.0, &names, false);
        crafted_competed(&b, 1_300, 900.0, &names, true);
        let competed = vec![a, b];
        let m = load_feature_matrix(&competed, &names, 3_800).unwrap();
        let mut rows = FeatureMatrix::with_capacity(3_800, names.len());
        for_each_feature_row(&competed, &names, |_, v| {
            rows.push_row(v);
            Ok(())
        })
        .unwrap();
        let rows = rows.finish().unwrap();
        assert_eq!(m.rows(), rows.rows());
        for i in 0..m.rows() {
            assert_eq!(
                m.row(i).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                rows.row(i).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row {i}"
            );
        }
    }

    #[test]
    fn the_tiled_transpose_is_the_serial_one() {
        for (nf, k) in [
            (1usize, 1usize),
            (3, 7),
            (16, 5),
            (17, 9),
            (40, 33),
            (387, 11),
        ] {
            let stage: Vec<f32> = (0..nf * k).map(|x| x as f32 * 0.5 - 7.0).collect();
            let want: Vec<Vec<f32>> = (0..nf)
                .map(|fi| (0..k).map(|r| stage[r * nf + fi]).collect())
                .collect();
            assert_eq!(transpose_block(&stage, nf, k), want, "{nf} x {k}");
        }
    }

    #[test]
    fn the_string_column_is_the_vec_of_strings_column() {
        use arrow::array::StringArray;
        let want = StringArray::from(
            (0..5)
                .map(|i| format!("psm_{}", 40 + i))
                .collect::<Vec<_>>(),
        );
        let got = string_column(5, 1, |r, s| {
            use std::fmt::Write as _;
            let _ = write!(s, "psm_{}", 40 + r);
        })
        .unwrap();
        assert_eq!(got, want);
        let empty = string_column(0, 12, |_, _| {}).unwrap();
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn the_feature_batch_follows_the_row_groups_within_its_bounds() {
        let names: Vec<String> = ["f0", "f1", "f2"].iter().map(|s| s.to_string()).collect();
        let plain = mumdia_io::table::ScanOptions::default();
        let coalesced = mumdia_io::table::ScanOptions::coalesced();
        let small = scratch("fbr_small.parquet");
        crafted_competed(&small, 7, 100.0, &names, false);
        // A tiny table still reads in batches of the floor.
        assert_eq!(
            feature_batch_rows(&TableFile::open(&small).unwrap(), &plain),
            FEATURE_BATCH_ROWS
        );
        // One batch per row group between the floor and the cap.
        let mid = scratch("fbr_mid.parquet");
        let mut w = mumdia_io::table::TableWriter::new(&mid).with_row_group_rows(40_000);
        w.write_cols(vec![Col::F64("f0".into(), vec![1.0; 100_000])])
            .unwrap();
        w.close().unwrap();
        let mid = TableFile::open(&mid).unwrap();
        assert_eq!(feature_batch_rows(&mid, &plain), 40_000);
        // The coalesced reader reads the group whole whatever the batch, so it keeps the
        // floor.
        assert_eq!(feature_batch_rows(&mid, &coalesced), FEATURE_BATCH_ROWS);
        // A table written as one huge group is read in batches of the cap.
        let big = scratch("fbr_big.parquet");
        write_table(&big, vec![Col::F64("f0".into(), vec![1.0; 200_000])]).unwrap();
        assert_eq!(
            feature_batch_rows(&TableFile::open(&big).unwrap(), &plain),
            FEATURE_BATCH_ROWS_MAX
        );
    }

    #[test]
    fn every_read_mode_streams_the_same_feature_rows() {
        // The coalesced reader, the plain reader with its parallel column groups, and any
        // batch size have to hand the closure the same rows: the handoff and the matrix are
        // built from them one at a time.
        let names: Vec<String> = ["f0", "f1", "f2"].iter().map(|s| s.to_string()).collect();
        let a = scratch("modes_a.parquet");
        let b = scratch("modes_b.parquet");
        crafted_competed(&a, 7, 100.0, &names, false);
        crafted_competed(&b, 5, 900.0, &names, true);
        let competed = vec![a, b];
        let collect = |scan: &mumdia_io::table::ScanOptions, rows: usize| {
            let mut seen: Vec<(usize, Vec<u32>)> = Vec::new();
            for_each_feature_row_with(
                &competed,
                &names,
                scan,
                |_| rows,
                |i, v| {
                    seen.push((i, v.iter().map(|x| x.to_bits()).collect()));
                    Ok(())
                },
            )
            .unwrap();
            seen
        };
        let reference = collect(&mumdia_io::table::ScanOptions::default(), 16_384);
        assert_eq!(reference.len(), 12);
        for scan in [
            mumdia_io::table::ScanOptions::default().with_decode_threads(2),
            mumdia_io::table::ScanOptions::coalesced(),
            mumdia_io::table::ScanOptions::coalesced().with_decode_threads(3),
        ] {
            for rows in [1, 3, 16_384] {
                assert_eq!(collect(&scan, rows), reference, "{scan:?}, {rows} rows");
            }
        }
    }

    /// The feature stream over a real competed table, by read mode and batch size.
    ///
    /// `MUMDIA_BENCH_PARQUET=<psms_competed.parquet> cargo test -p mumdia --release
    /// bench_feature_stream -- --ignored --nocapture`. Every arm narrows every value and
    /// folds its bits into a checksum, which the arms must agree on; the minimum of
    /// `MUMDIA_BENCH_REPS` (3) interleaved rounds is printed.
    #[test]
    #[ignore = "benchmark; needs MUMDIA_BENCH_PARQUET"]
    fn bench_feature_stream_a_real_artifact() {
        let Ok(src) = std::env::var("MUMDIA_BENCH_PARQUET") else {
            println!("set MUMDIA_BENCH_PARQUET to a real competed table to run this");
            return;
        };
        let reps: usize = std::env::var("MUMDIA_BENCH_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let names = FeatureSchema::read(&src).unwrap().feature_columns;
        let competed = vec![src.clone()];
        let t = TableFile::open(&src).unwrap();
        println!(
            "{src}: {} rows, {} features, row groups {:?}",
            t.nrows,
            names.len(),
            t.row_group_rows().iter().take(4).collect::<Vec<_>>()
        );
        type Arm = (&'static str, mumdia_io::table::ScanOptions, bool);
        let arms: Vec<Arm> = vec![
            (
                "plain, 16,384 rows",
                mumdia_io::table::ScanOptions::default(),
                false,
            ),
            (
                "plain, row group",
                mumdia_io::table::ScanOptions::default(),
                true,
            ),
            (
                "coalesced, 16,384 rows",
                mumdia_io::table::ScanOptions::coalesced(),
                false,
            ),
            (
                "coalesced, row group",
                mumdia_io::table::ScanOptions::coalesced(),
                true,
            ),
        ];
        let mut best = vec![f64::INFINITY; arms.len()];
        let mut sums: Vec<u64> = vec![0; arms.len()];
        for _ in 0..reps {
            for (k, (_, scan, aligned)) in arms.iter().enumerate() {
                let t0 = std::time::Instant::now();
                let mut sum = 0u64;
                let sizing = |t: &TableFile| {
                    if *aligned {
                        feature_batch_rows(t, &mumdia_io::table::ScanOptions::default())
                    } else {
                        FEATURE_BATCH_ROWS
                    }
                };
                for_each_feature_row_with(&competed, &names, scan, sizing, |_, v| {
                    for x in v {
                        sum = sum.wrapping_add(u64::from(x.to_bits()));
                    }
                    Ok(())
                })
                .unwrap();
                best[k] = best[k].min(t0.elapsed().as_secs_f64());
                sums[k] = sum;
            }
        }
        for (k, (name, _, _)) in arms.iter().enumerate() {
            println!("{name:>24}: {:7.2} s", best[k]);
        }
        assert!(
            sums.windows(2).all(|w| w[0] == w[1]),
            "arms disagree: {sums:?}"
        );
    }

    #[test]
    fn the_streamed_handoff_is_byte_identical_to_the_handoff_from_the_matrix() {
        // The claim of the streaming change: the sidecar receives exactly the file it
        // received when the engine materialised the whole feature matrix first. Both
        // encodings, several inputs, and a batch small enough to force several parquet
        // blocks with a block boundary inside the second input.
        let names: Vec<String> = ["f0", "f1", "f2"].iter().map(|s| s.to_string()).collect();
        let a = scratch("hs_a.parquet");
        let b = scratch("hs_b.parquet");
        crafted_competed(&a, 7, 100.0, &names, false);
        crafted_competed(&b, 5, 900.0, &names, true);
        let competed = vec![a, b];
        let n = 12;
        let is_decoy: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let pform = flat(&(0..n).map(|i| format!("PEPTIDEK/{i}")).collect::<Vec<_>>());
        let protein = flat(&(0..n).map(|i| format!("sp|P{i:05}|X")).collect::<Vec<_>>());
        let mz: Vec<f64> = (0..n).map(|i| 400.0 + i as f64 * 1.5).collect();
        let m = load_feature_matrix(&competed, &names, n).unwrap();

        for use_pq in [true, false] {
            let from_matrix = SidecarPaths {
                handoff: scratch(if use_pq { "m.parquet" } else { "m.pin" }),
                out: String::new(),
                foldkeys: String::new(),
                use_pq,
            };
            let from_stream = SidecarPaths {
                handoff: scratch(if use_pq { "s.parquet" } else { "s.pin" }),
                out: String::new(),
                foldkeys: String::new(),
                use_pq,
            };
            let mut w = HandoffWriter::new(&from_matrix, &names, &is_decoy, &pform, &protein, &mz)
                .unwrap()
                .with_block_rows(5);
            for i in 0..n {
                w.push_row(i, m.row(i)).unwrap();
            }
            assert_eq!(w.finish().unwrap(), n as u64);

            let mut w = HandoffWriter::new(&from_stream, &names, &is_decoy, &pform, &protein, &mz)
                .unwrap()
                .with_block_rows(5);
            for_each_feature_row(&competed, &names, |i, v| w.push_row(i, v)).unwrap();
            assert_eq!(w.finish().unwrap(), n as u64);

            let reference = std::fs::read(&from_matrix.handoff).unwrap();
            assert_eq!(
                reference,
                std::fs::read(&from_stream.handoff).unwrap(),
                "handoff differs (parquet = {use_pq})"
            );

            // The batch form the stage uses: whole decoded batches staged column by column,
            // with batches smaller than, straddling and larger than the 5-row blocks, under
            // both readers.
            for scan in [
                mumdia_io::table::ScanOptions::default(),
                mumdia_io::table::ScanOptions::coalesced(),
            ] {
                for batch in [1usize, 3, 16_384] {
                    let from_batches = SidecarPaths {
                        handoff: scratch(if use_pq { "b.parquet" } else { "b.pin" }),
                        out: String::new(),
                        foldkeys: String::new(),
                        use_pq,
                    };
                    let mut w =
                        HandoffWriter::new(&from_batches, &names, &is_decoy, &pform, &protein, &mz)
                            .unwrap()
                            .with_block_rows(5);
                    for_each_feature_batch_with(
                        &competed,
                        &names,
                        &scan,
                        |_| batch,
                        |b| w.push_batch(b),
                    )
                    .unwrap();
                    assert_eq!(w.finish().unwrap(), n as u64);
                    assert_eq!(
                        reference,
                        std::fs::read(&from_batches.handoff).unwrap(),
                        "batch handoff differs (parquet = {use_pq}, {batch}-row batches, {scan:?})"
                    );
                }
            }
        }
        // And the PIN really is the PIN contract: header, then one row per PSM.
        let pin = scratch("m.pin");
        let text = std::fs::read_to_string(&pin).unwrap();
        assert!(text.starts_with(
            "SpecId\tLabel\tScanNr\tExpMass\tCalcMass\tf0\tf1\tf2\tPeptide\tProteins\n"
        ));
        assert_eq!(text.lines().count(), n + 1);
        assert!(text.lines().nth(1).unwrap().starts_with("psm_0\t-1\t0\t"));
    }

    /// The handoff exactly as it was written before the flat/bit metadata columns: the
    /// three metadata columns as `Vec<String>`, `StringArray::from(Vec<String>)` for the
    /// text arrays, `protein[start..end].to_vec()` for `Proteins`, and `label[i] ==
    /// "decoy"` for `Label`. Kept verbatim in the test module so the shipped writer can be
    /// pinned against THIS rather than against another copy of itself.
    fn handoff_from_string_columns(
        paths: &SidecarPaths,
        names: &[String],
        meta: (&[String], &[String], &[String]),
        mz: &[f64],
        rows: &[Vec<f32>],
        block_rows: usize,
    ) {
        use arrow::array::{ArrayRef, Float32Array, Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::io::Write as _;
        use std::sync::Arc;

        let (label, pform, protein) = meta;
        let nf = names.len();
        if !paths.use_pq {
            let mut w = std::io::BufWriter::with_capacity(
                1 << 20,
                std::fs::File::create(&paths.handoff).unwrap(),
            );
            w.write_all(b"SpecId\tLabel\tScanNr\tExpMass\tCalcMass\t")
                .unwrap();
            w.write_all(names.join("\t").as_bytes()).unwrap();
            w.write_all(b"\tPeptide\tProteins\n").unwrap();
            for (i, values) in rows.iter().enumerate() {
                let lab = if label[i] == "decoy" { -1 } else { 1 };
                write!(w, "psm_{}\t{}\t{}\t{:.5}\t{:.5}\t", i, lab, i, mz[i], mz[i]).unwrap();
                for v in values.iter().take(nf) {
                    write!(w, "{:.6}\t", v).unwrap();
                }
                writeln!(w, "-.{}.-\t{}", pform[i], protein[i]).unwrap();
            }
            w.flush().unwrap();
            return;
        }

        let mut fields: Vec<Field> = vec![
            Field::new("SpecId", DataType::Utf8, false),
            Field::new("Label", DataType::Int32, false),
            Field::new("ScanNr", DataType::Int32, false),
            Field::new("ExpMass", DataType::Float64, false),
            Field::new("CalcMass", DataType::Float64, false),
        ];
        for n in names {
            fields.push(Field::new(n, DataType::Float32, false));
        }
        fields.push(Field::new("Peptide", DataType::Utf8, false));
        fields.push(Field::new("Proteins", DataType::Utf8, false));
        let schema = Arc::new(Schema::new(fields));
        let mut w = mumdia_io::table::BatchWriter::with_row_group_rows(
            &paths.handoff,
            schema.clone(),
            HANDOFF_ROW_GROUP_ROWS,
        )
        .unwrap();
        let mut start = 0usize;
        while start < rows.len() {
            let end = (start + block_rows).min(rows.len());
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(nf + 7);
            arrays.push(Arc::new(StringArray::from(
                (start..end).map(|i| format!("psm_{i}")).collect::<Vec<_>>(),
            )));
            arrays.push(Arc::new(Int32Array::from(
                (start..end)
                    .map(|i| if label[i] == "decoy" { -1 } else { 1 })
                    .collect::<Vec<_>>(),
            )));
            arrays.push(Arc::new(Int32Array::from(
                (start..end).map(|i| i as i32).collect::<Vec<_>>(),
            )));
            let mzv: Vec<f64> = mz[start..end].to_vec();
            arrays.push(Arc::new(Float64Array::from(mzv.clone())));
            arrays.push(Arc::new(Float64Array::from(mzv)));
            for fi in 0..nf {
                let col: Vec<f32> = rows[start..end].iter().map(|r| r[fi]).collect();
                arrays.push(Arc::new(Float32Array::from(col)));
            }
            arrays.push(Arc::new(StringArray::from(
                (start..end)
                    .map(|i| format!("-.{}.-", pform[i]))
                    .collect::<Vec<_>>(),
            )));
            arrays.push(Arc::new(StringArray::from(protein[start..end].to_vec())));
            w.write(&RecordBatch::try_new(schema.clone(), arrays).unwrap())
                .unwrap();
            start = end;
        }
        w.close().unwrap();
    }

    #[test]
    fn the_flat_metadata_columns_write_the_same_handoff() {
        // The other half of the flat/bit output-equality claim, and the half the smoke run
        // cannot reach: `ci/smoke.sh` uses `configs/examples/native.json`, whose empty
        // `rescore` block leaves `classifier = native_tda` and `python = None`, so
        // `stream_to_handoff` is false and `run_pin_sidecar` is never entered. Every
        // rewritten read in `HandoffWriter` -- `is_decoy[i]` in the PIN row and in `Label`,
        // `pform.get(i)`/`protein.get(i)` in both encodings, and `from_iter_values` in
        // place of `StringArray::from(protein[start..end].to_vec())` -- is therefore
        // unexercised by the byte-identity evidence from a smoke run, and the streamed
        // -versus -matrix test above compares the new writer against itself.
        //
        // So: both encodings, a block boundary inside the data, decoys, and a protein
        // column with repeats (the `to_vec()` the flat path replaced).
        let names: Vec<String> = ["f0", "f1", "f2"].iter().map(|s| s.to_string()).collect();
        let n = 12usize;
        let label: Vec<String> = (0..n)
            .map(|i| if i % 3 == 0 { "decoy" } else { "target" }.to_string())
            .collect();
        let pform: Vec<String> = (0..n).map(|i| format!("PEPTIDEK[+16]/{i}")).collect();
        let protein: Vec<String> = (0..n)
            .map(|i| format!("sp|P{:05}|PROT_HUMAN;sp|Q{:05}|ALT", i % 5, i % 3))
            .collect();
        let mz: Vec<f64> = (0..n).map(|i| 400.0 + i as f64 * 1.5).collect();
        let values: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..names.len())
                    .map(|j| (i * 7 + j) as f32 * 0.125 - 3.0)
                    .collect()
            })
            .collect();
        let is_decoy: Vec<bool> = label.iter().map(|l| l == "decoy").collect();
        let fpform = flat(&pform);
        let fprotein = flat(&protein);

        for use_pq in [true, false] {
            let ext = if use_pq { "parquet" } else { "pin" };
            let old = SidecarPaths {
                handoff: scratch(&format!("hold.{ext}")),
                out: String::new(),
                foldkeys: String::new(),
                use_pq,
            };
            let new = SidecarPaths {
                handoff: scratch(&format!("hnew.{ext}")),
                out: String::new(),
                foldkeys: String::new(),
                use_pq,
            };
            handoff_from_string_columns(&old, &names, (&label, &pform, &protein), &mz, &values, 5);
            let mut w = HandoffWriter::new(&new, &names, &is_decoy, &fpform, &fprotein, &mz)
                .unwrap()
                .with_block_rows(5);
            for (i, v) in values.iter().enumerate() {
                w.push_row(i, v).unwrap();
            }
            assert_eq!(w.finish().unwrap(), n as u64);
            assert_eq!(
                std::fs::read(&old.handoff).unwrap(),
                std::fs::read(&new.handoff).unwrap(),
                "the flat/bit handoff differs from the Vec<String> handoff (parquet = {use_pq})"
            );
        }
    }

    /// What the handoff block size costs, end to end, at the real column count.
    ///
    /// Ignored: it stages and encodes several hundred MB. Run with
    /// `cargo test -p mumdia --release handoff_block_size -- --ignored --nocapture`, and
    /// run it three times: the numbers on [`HANDOFF_BATCH_ROWS`] are the minimum per arm
    /// over three whole runs, and one run's arms do not separate from each other by more
    /// than one arm varies between runs. It asserts nothing and it is not a regression
    /// test; it is the only thing that can re-derive that constant.
    ///
    /// Both arms pay for everything they use and nothing they do not: each builds its own
    /// `HandoffWriter` through `with_block_size`, so the staging buffer is reserved ONCE
    /// at that arm's size (`with_block_rows` would not do -- `new` reserves at
    /// `HANDOFF_BATCH_ROWS` first, and every arm would pay that 387 MB reservation and
    /// free inside the timer), then pushes every row, transposes, builds every
    /// `RecordBatch` and encodes to a real file. Nothing is hoisted out of the timer,
    /// which is the point: the transpose is only part of the handoff write, so a
    /// per-transpose speedup is not a per-stage speedup, and the second section below
    /// shows exactly how much of the first is transpose.
    #[test]
    #[ignore]
    fn handoff_block_size_end_to_end() {
        let nf = 387usize;
        let rows = 262_144usize;
        let names: Vec<String> = (0..nf).map(|j| format!("f{j}")).collect();
        let is_decoy: Vec<bool> = (0..rows).map(|i| i % 3 == 0).collect();
        let pform = flat(
            &(0..rows)
                .map(|i| format!("PEPTIDEK[+16]{}", i % 100_000))
                .collect::<Vec<_>>(),
        );
        let protein = flat(
            &(0..rows)
                .map(|i| format!("sp|P{:05}|PROT_HUMAN", i % 70_000))
                .collect::<Vec<_>>(),
        );
        let mz: Vec<f64> = (0..rows)
            .map(|i| 400.0 + (i % 9973) as f64 * 0.11)
            .collect();
        // 256 distinct rows, cycled: the values differ row to row, so snappy has real work
        // to do and the encode arm is not a degenerate one. Built once, outside every
        // timer, and read identically by every arm.
        let pool: Vec<Vec<f32>> = (0..256usize)
            .map(|k| {
                (0..nf)
                    .map(|j| {
                        let x = (k * 2_654_435_761 + j * 40_503) as u32;
                        (x as f32 / u32::MAX as f32) * 1000.0 - 500.0
                    })
                    .collect()
            })
            .collect();

        for block in [250_000usize, 131_072, 65_536, 16_384, 4_096, 1_024] {
            let path = scratch(&format!("hbench_{block}.parquet"));
            let mut best = f64::INFINITY;
            let mut bytes = 0u64;
            // Min of three: this is a disk-touching benchmark on a shared machine, and one
            // arm being unlucky is otherwise indistinguishable from an effect.
            for _ in 0..3 {
                let paths = SidecarPaths {
                    handoff: path.clone(),
                    out: String::new(),
                    foldkeys: String::new(),
                    use_pq: true,
                };
                let t0 = std::time::Instant::now();
                let mut w = HandoffWriter::with_block_size(
                    &paths, &names, &is_decoy, &pform, &protein, &mz, block,
                )
                .unwrap();
                for i in 0..rows {
                    w.push_row(i, &pool[i % pool.len()]).unwrap();
                }
                assert_eq!(w.finish().unwrap(), rows as u64);
                best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
                bytes = std::fs::metadata(&path).unwrap().len();
            }
            // `transient` is the staging buffer AND the columnar copy `flush_block`
            // gathers out of it: all `nf` column vectors are live until the
            // `RecordBatch` is built, so the block costs twice what it stages.
            println!(
                "block {block:>7}: {best:8.1} ms  ({:.2} s/Mrow)  transient {:>6.1} MB \
                 (stage {:.1} + columns {:.1})  file {:.1} MB",
                best / 1000.0 * 1e6 / rows as f64,
                (2 * block.min(rows) * nf * 4) as f64 / 1e6,
                (block.min(rows) * nf * 4) as f64 / 1e6,
                (block.min(rows) * nf * 4) as f64 / 1e6,
                bytes as f64 / 1e6
            );
            let _ = std::fs::remove_file(&path);
        }

        // The transpose ALONE, to locate the part the block size actually moves: stage a
        // block row-major, then gather each of the 387 columns out of it at a 1,548-byte
        // stride. Each arm allocates its own staging buffer and its own column vectors
        // inside its own timer, so neither is handed a buffer the other paid for.
        //
        // The gathered columns are COLLECTED AND HELD, exactly as `flush_block` holds them
        // in `arrays` until `RecordBatch::try_new`. An earlier version of this section read
        // only `col[0]` and dropped each column immediately, which is a different loop:
        // 386 of every 387 stores are then dead and formally removable, and the allocator
        // hands back the same hot buffer every iteration instead of `nf` distinct ones.
        println!("-- transpose only (no encode) --");
        for block in [250_000usize, 131_072, 16_384, 4_096, 1_024] {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                let mut stage: Vec<f32> = Vec::with_capacity(block * nf);
                let mut done = 0usize;
                while done < rows {
                    let k = block.min(rows - done);
                    for i in 0..k {
                        stage.extend_from_slice(&pool[(done + i) % pool.len()]);
                    }
                    let mut cols: Vec<Vec<f32>> = Vec::with_capacity(nf);
                    for fi in 0..nf {
                        cols.push((0..k).map(|r| stage[r * nf + fi]).collect());
                    }
                    std::hint::black_box(&cols);
                    drop(cols);
                    stage.clear();
                    done += k;
                }
                best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
            }
            println!(
                "block {block:>7}: {best:8.1} ms  ({:.2} s/Mrow)",
                best / 1000.0 * 1e6 / rows as f64
            );
        }
    }

    /// A minimal competed table: the metadata columns `rescore::run` reads plus two
    /// feature columns. No `.schema.json` companion, so the feature list is reconstructed
    /// from the parquet's own columns (`NON_FEATURE_COLUMNS`).
    fn crafted_competed_table(path: &str, rows: usize) {
        crafted_competed_table_planting(path, rows, None, None);
    }

    /// The same table with one row's `label` replaced by a value that is neither
    /// "target" nor "decoy".
    fn crafted_competed_table_mislabelled(path: &str, rows: usize, row: usize, label: &str) {
        crafted_competed_table_inner(path, rows, None, None, Some((row, label)));
    }

    /// The same table with a NaN planted in the `feat_b` feature column and/or in the
    /// `precursor_mz` metadata column, which is what a malformed competed input looks like
    /// to the two validations (a null f64 cell reads back as NaN, so a null is the same
    /// case).
    fn crafted_competed_table_planting(
        path: &str,
        rows: usize,
        nan_feature_row: Option<usize>,
        nan_mz_row: Option<usize>,
    ) {
        crafted_competed_table_inner(path, rows, nan_feature_row, nan_mz_row, None);
    }

    fn crafted_competed_table_inner(
        path: &str,
        rows: usize,
        nan_feature_row: Option<usize>,
        nan_mz_row: Option<usize>,
        bad_label: Option<(usize, &str)>,
    ) {
        let cols = vec![
            Col::U32("candidate_id".into(), (0..rows as u32).collect()),
            Col::Str(
                "label".into(),
                (0..rows)
                    .map(|i| match bad_label {
                        Some((row, l)) if row == i => l.to_string(),
                        _ => if i % 2 == 0 { "target" } else { "decoy" }.to_string(),
                    })
                    .collect(),
            ),
            Col::U32(
                "base_peptide_id".into(),
                (0..rows as u32).map(|i| i / 2).collect(),
            ),
            Col::Str(
                "peptidoform".into(),
                (0..rows).map(|i| format!("PEPTIDEK{i}")).collect(),
            ),
            Col::Str(
                "protein".into(),
                (0..rows).map(|i| format!("sp|P{i:04}|X")).collect(),
            ),
            Col::F64("charge".into(), (0..rows).map(|_| 2.0).collect()),
            Col::F64(
                "prelim_score".into(),
                (0..rows)
                    .map(|i| if i % 2 == 0 { 9.0 } else { 1.0 } + i as f64 * 0.01)
                    .collect(),
            ),
            Col::F64(
                "precursor_mz".into(),
                (0..rows)
                    .map(|i| {
                        if nan_mz_row == Some(i) {
                            f64::NAN
                        } else {
                            400.0 + i as f64
                        }
                    })
                    .collect(),
            ),
            Col::F64(
                "apex_rt".into(),
                (0..rows).map(|i| 10.0 + i as f64).collect(),
            ),
            Col::F64(
                "elution_lo".into(),
                (0..rows).map(|i| 9.0 + i as f64).collect(),
            ),
            Col::F64(
                "elution_hi".into(),
                (0..rows).map(|i| 11.0 + i as f64).collect(),
            ),
            Col::I32("peak_rank".into(), vec![0; rows]),
            Col::F64(
                "feat_a".into(),
                (0..rows)
                    .map(|i| if i % 2 == 0 { 3.0 } else { 0.5 })
                    .collect(),
            ),
            Col::F64(
                "feat_b".into(),
                (0..rows)
                    .map(|i| {
                        if nan_feature_row == Some(i) {
                            f64::NAN
                        } else {
                            (i % 7) as f64 * 0.25
                        }
                    })
                    .collect(),
            ),
        ];
        write_table(path, cols).unwrap();
    }

    /// Rows `lo..hi` of a larger crafted table, as a table of their own: the rows a band's
    /// competed table holds when the pooled one is their concatenation.
    fn crafted_rows(path: &str, lo: usize, hi: usize) {
        let r = lo..hi;
        let cols = vec![
            Col::U32("candidate_id".into(), r.clone().map(|i| i as u32).collect()),
            Col::Str(
                "label".into(),
                r.clone()
                    .map(|i| if i % 3 == 0 { "decoy" } else { "target" }.to_string())
                    .collect(),
            ),
            Col::U32(
                "base_peptide_id".into(),
                r.clone().map(|i| i as u32 / 2).collect(),
            ),
            Col::Str(
                "peptidoform".into(),
                r.clone().map(|i| format!("PEPTIDEK{}", i % 90)).collect(),
            ),
            Col::Str(
                "protein".into(),
                r.clone().map(|i| format!("sp|P{:04}|X", i % 40)).collect(),
            ),
            Col::F64(
                "charge".into(),
                r.clone().map(|i| 2.0 + (i % 2) as f64).collect(),
            ),
            Col::F64(
                "prelim_score".into(),
                r.clone().map(|i| ((i * 37) % 101) as f64 * 0.1).collect(),
            ),
            Col::F64(
                "precursor_mz".into(),
                r.clone().map(|i| 400.0 + i as f64).collect(),
            ),
            Col::F64(
                "apex_rt".into(),
                r.clone().map(|i| 10.0 + i as f64).collect(),
            ),
            Col::F64(
                "elution_lo".into(),
                r.clone().map(|i| 9.0 + i as f64).collect(),
            ),
            Col::F64(
                "elution_hi".into(),
                r.clone().map(|i| 11.0 + i as f64).collect(),
            ),
            Col::I32("peak_rank".into(), vec![0; hi - lo]),
            Col::F64(
                "feat_a".into(),
                r.clone()
                    .map(|i| {
                        let noise = ((i * 7919) % 97) as f64 / 97.0;
                        if i % 3 == 0 {
                            noise
                        } else {
                            0.6 + noise
                        }
                    })
                    .collect(),
            ),
            Col::F64(
                "feat_b".into(),
                r.clone().map(|i| ((i * 31) % 17) as f64 * 0.25).collect(),
            ),
        ];
        write_table(path, cols).unwrap();
    }

    /// Rescore over a grouped run's band tables with a source map scores exactly what it
    /// scores over the pooled tables they concatenate to (`groups.pool_competed = false`):
    /// the same `psms_scored.parquet` bytes, for one run of three bands and for two runs.
    #[test]
    fn band_tables_with_a_source_map_score_the_pooled_tables_bytes() {
        let bands: Vec<String> = [(0, 110), (110, 260), (260, 420)]
            .iter()
            .enumerate()
            .map(|(k, &(lo, hi))| {
                let path = scratch(&format!("srcmap_band{k}.parquet"));
                crafted_rows(&path, lo, hi);
                path
            })
            .collect();
        let run_scored = |competed: &[String], sources: Option<&[u32]>, name: &str| {
            let out = scratch(&format!("srcmap_{name}_scored.parquet"));
            let cfg = RescoreConfig {
                classifier: RescorerKind::NativeTda,
                ..Default::default()
            };
            run(RescoreParams {
                competed,
                sources,
                out: &out,
                work_dir: &scratch(&format!("srcmap_{name}_work")),
                script_dir: "scripts",
                cfg: &cfg,
                config_hash: "test",
            })
            .unwrap();
            let report: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(format!("{out}.report.json")).unwrap(),
            )
            .unwrap();
            (std::fs::read(&out).unwrap(), report)
        };
        // One run: the pooled table is all 420 rows.
        let pooled = scratch("srcmap_pooled.parquet");
        crafted_rows(&pooled, 0, 420);
        let (want, want_report) = run_scored(std::slice::from_ref(&pooled), None, "one_pooled");
        let (got, got_report) = run_scored(&bands, Some(&[0, 0, 0]), "one_bands");
        assert!(
            got == want,
            "one run: the scored table differs from the pooled one's"
        );
        assert_eq!(want_report["params"]["classifier"], json!("native_tda"));
        assert!(want_report["params"].get("competed_sources").is_none());
        assert_eq!(got_report["params"]["competed_sources"], json!([0, 0, 0]));
        assert_eq!(
            got_report["stats"], want_report["stats"],
            "the counts at 1% are the pooled run's"
        );
        // Two runs: bands 0 and 1 are run 0, band 2 is run 1.
        let run0 = scratch("srcmap_run0.parquet");
        crafted_rows(&run0, 0, 260);
        let run1 = scratch("srcmap_run1.parquet");
        crafted_rows(&run1, 260, 420);
        let (want, _) = run_scored(&[run0, run1], None, "two_pooled");
        let (got, _) = run_scored(&bands, Some(&[0, 0, 1]), "two_bands");
        assert!(
            got == want,
            "two runs: the scored table differs from the pooled one's"
        );
        // A map of the wrong length, or one that goes back, is refused.
        let e = run(RescoreParams {
            competed: &bands,
            sources: Some(&[0, 1]),
            out: &scratch("srcmap_bad_scored.parquet"),
            work_dir: &scratch("srcmap_bad_work"),
            script_dir: "scripts",
            cfg: &RescoreConfig::default(),
            config_hash: "test",
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("2 sources for 3"), "{e}");
        let e = run(RescoreParams {
            competed: &bands,
            sources: Some(&[1, 0, 1]),
            out: &scratch("srcmap_bad2_scored.parquet"),
            work_dir: &scratch("srcmap_bad2_work"),
            script_dir: "scripts",
            cfg: &RescoreConfig::default(),
            config_hash: "test",
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("non-decreasing"), "{e}");
    }

    #[test]
    fn strict_sidecar_streams_the_handoff_without_building_a_matrix() {
        // End-to-end through `run`: under strict with a PIN sidecar the features go from
        // the competed parquet straight into the handoff, and no `FeatureMatrix` is
        // allocated. Driven with an interpreter that cannot be spawned, so the run fails at
        // the child rather than needing a Python environment -- but only AFTER the handoff
        // has been written from the streamed features, which is what is asserted.
        let competed = scratch("run_competed.parquet");
        crafted_competed_table(&competed, 24);
        let work = scratch("run_work");
        let out = scratch("run_scored.parquet");
        let cfg = RescoreConfig {
            classifier: RescorerKind::NnTorch,
            strict: true,
            python: Some("mumdia-no-such-interpreter-for-this-test".to_string()),
            ..Default::default()
        };
        let err = run(RescoreParams {
            competed: &[competed],
            sources: None,
            out: &out,
            work_dir: &work,
            script_dir: "scripts",
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("NnTorch sidecar failed") && err.contains("strict"),
            "{err}"
        );
        // The handoff is a parquet (the default `handoff`), named per invocation, and holds
        // every row exactly once in flat order with the PIN column contract.
        let tag = format!(
            "{}_{}",
            std::path::Path::new(&out)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap(),
            std::process::id()
        );
        let handoff = format!("{work}/rescore_{tag}.features.parquet");
        let t = mumdia_io::table::Table::read(&handoff).unwrap();
        assert_eq!(t.nrows, 24);
        assert_eq!(t.i32("ScanNr").unwrap(), (0..24).collect::<Vec<i32>>());
        assert_eq!(
            t.i32("Label").unwrap(),
            (0..24)
                .map(|i| if i % 2 == 0 { 1 } else { -1 })
                .collect::<Vec<i32>>()
        );
        assert_eq!(
            t.f32("feat_a").unwrap(),
            (0..24)
                .map(|i| if i % 2 == 0 { 3.0f32 } else { 0.5 })
                .collect::<Vec<f32>>()
        );
    }

    /// Drive `run` down the streamed-handoff path (strict + a sidecar classifier + an
    /// interpreter that cannot be spawned) and return the error, the handoff path the
    /// invocation used, and its work directory.
    fn run_streamed(competed: &str, name: &str) -> (String, String, String) {
        let work = scratch(&format!("{name}_work"));
        let out = scratch(&format!("{name}_scored.parquet"));
        let cfg = RescoreConfig {
            classifier: RescorerKind::NnTorch,
            strict: true,
            python: Some("mumdia-no-such-interpreter-for-this-test".to_string()),
            ..Default::default()
        };
        let err = run(RescoreParams {
            competed: &[competed.to_string()],
            sources: None,
            out: &out,
            work_dir: &work,
            script_dir: "scripts",
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap_err()
        .to_string();
        let tag = format!(
            "{}_{}",
            std::path::Path::new(&out)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap(),
            std::process::id()
        );
        (err, format!("{work}/rescore_{tag}.features.parquet"), work)
    }

    /// Nothing that looks like a handoff (published or half-written) is left in `work`.
    fn no_handoff_rubble(work: &str) {
        let left: Vec<String> = std::fs::read_dir(work)
            .map(|d| {
                d.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .filter(|n| n.contains(".features.parquet") || n.ends_with(".pin"))
                    .collect()
            })
            .unwrap_or_default();
        assert!(left.is_empty(), "handoff rubble left behind: {left:?}");
    }

    #[test]
    fn a_non_finite_feature_aborts_the_stream_instead_of_writing_the_whole_handoff() {
        // Before the features were streamed, the non-finite scan ran over the resident
        // matrix and bailed before `run_pin_sidecar` was ever entered, so a malformed
        // competed table cost nothing. Streaming moved the scan into the write and then
        // reported it after `finish()`: the full handoff -- hundreds of GB at experiment
        // scale -- was written and published for a run that was always going to abort.
        // The row and column named must not change, and the file must not be there.
        let competed = scratch("nanfeat_competed.parquet");
        crafted_competed_table_planting(&competed, 24, Some(3), None);
        let (err, handoff, work) = run_streamed(&competed, "nanfeat");
        assert_eq!(
            err, "rescore input contains non-finite feature 'feat_b' at flat row 3: NaN",
            "the message must name the same row and column the serial scan named"
        );
        assert!(
            !std::path::Path::new(&handoff).exists(),
            "the handoff must not survive a stream that aborted: {handoff}"
        );
        no_handoff_rubble(&work);
    }

    #[test]
    fn a_non_finite_precursor_mz_aborts_the_stream_at_its_row() {
        // The scalar scan runs before the stream, so its row is already known; the stream
        // still has to reach that row before the message is decided (an earlier bad feature
        // would win), and must stop there rather than write the remaining rows.
        let competed = scratch("nanmz_competed.parquet");
        crafted_competed_table_planting(&competed, 24, None, Some(2));
        let (err, handoff, work) = run_streamed(&competed, "nanmz");
        assert_eq!(
            err,
            "rescore input contains non-finite prelim_score/precursor_mz at flat row 2"
        );
        assert!(!std::path::Path::new(&handoff).exists(), "{handoff}");
        no_handoff_rubble(&work);
    }

    #[test]
    fn the_first_offending_row_still_decides_the_message_when_both_kinds_are_present() {
        // The serial loop this replaces checked a row's features before its scalars and
        // stopped at the first offending ROW. Aborting the stream early must not change
        // which of the two answers is reported.
        let a = scratch("both_scalar_first.parquet");
        crafted_competed_table_planting(&a, 24, Some(5), Some(2));
        let (err, _, _) = run_streamed(&a, "both_scalar_first");
        assert_eq!(
            err, "rescore input contains non-finite prelim_score/precursor_mz at flat row 2",
            "the scalar row is earlier, so it is the one reported"
        );

        let b = scratch("both_feature_first.parquet");
        crafted_competed_table_planting(&b, 24, Some(1), Some(4));
        let (err, _, _) = run_streamed(&b, "both_feature_first");
        assert_eq!(
            err, "rescore input contains non-finite feature 'feat_b' at flat row 1: NaN",
            "the feature row is earlier, so it is the one reported"
        );
    }

    #[test]
    fn the_earlier_row_wins_and_a_tie_goes_to_the_feature() {
        // The tie-break `bail_non_finite` applies, pinned directly: within one row the old
        // serial loop tested the features first, so an equal row index reports the feature.
        let names: Vec<String> = ["f0", "f1"].iter().map(|s| s.to_string()).collect();
        assert!(bail_non_finite(None, None, &names).is_ok());
        let feature_at = |row: usize| Some((row, 1, f32::NAN));
        for (feature, scalar, expect) in [
            (feature_at(3), None, "feature 'f1' at flat row 3"),
            (None, Some(3), "prelim_score/precursor_mz at flat row 3"),
            (feature_at(3), Some(7), "feature 'f1' at flat row 3"),
            (
                feature_at(7),
                Some(3),
                "prelim_score/precursor_mz at flat row 3",
            ),
            (feature_at(3), Some(3), "feature 'f1' at flat row 3"),
        ] {
            let err = bail_non_finite(feature, scalar, &names)
                .unwrap_err()
                .to_string();
            assert!(err.contains(expect), "{err} should contain {expect}");
        }
    }

    #[test]
    fn an_unknown_label_is_still_refused_by_its_value() {
        // `label` is no longer carried as a `Vec<String>`, so the check that used to be
        // `fdr::validate_labels(&label)` now runs on the flat text during the read. The
        // behaviour it has to keep is the OLD one, in both halves: an unknown value is a
        // hard error, and the message names the value it saw -- a bool could not. It also
        // has to fail at the same point, before a byte of handoff is written.
        let competed = scratch("badlabel_competed.parquet");
        crafted_competed_table_mislabelled(&competed, 24, 7, "TARGET");
        let (err, handoff, work) = run_streamed(&competed, "badlabel");
        assert!(
            err.contains(r#"unknown PSM label "TARGET""#)
                && err.contains(r#"expected "target" or "decoy""#),
            "{err}"
        );
        assert!(!std::path::Path::new(&handoff).exists(), "{handoff}");
        no_handoff_rubble(&work);

        // An empty label is refused the same way, and is not silently a target.
        let competed = scratch("emptylabel_competed.parquet");
        crafted_competed_table_mislabelled(&competed, 24, 0, "");
        let (err, _, _) = run_streamed(&competed, "emptylabel");
        assert!(err.contains(r#"unknown PSM label """#), "{err}");
    }

    #[test]
    fn a_feature_schema_with_no_feature_columns_is_refused() {
        // Reachable only through a hand-written companion: an explicitly empty
        // `rescore.features` list is already refused, and `FeatureSchema::read` refuses to
        // reconstruct an empty list from the parquet. What the old validation did with it
        // was an accident of `chunks_exact(n_features.max(1))` over an empty buffer: the
        // loop ran zero times, which ALSO skipped the prelim_score/precursor_mz check for
        // every row, and the run went on to train on no features at all.
        let competed = scratch("nofeat_competed.parquet");
        crafted_competed_table(&competed, 8);
        std::fs::write(
            format!("{competed}.schema.json"),
            r#"{"feature_columns":[],"schema_id":"empty-for-this-test"}"#,
        )
        .unwrap();
        let (err, _, work) = run_streamed(&competed, "nofeat");
        std::fs::remove_file(format!("{competed}.schema.json")).ok();
        assert!(
            err.contains("declares no feature columns"),
            "an empty feature set must be refused by name: {err}"
        );
        no_handoff_rubble(&work);
    }

    #[test]
    fn a_pass_that_disagrees_about_the_row_count_is_refused_with_both_counts() {
        assert!(refuse_row_disagreement("the streamed handoff", 10, 10).is_ok());
        let err = refuse_row_disagreement("the feature matrix", 9, 10)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("the feature matrix has 9 rows where 10 were expected"),
            "{err}"
        );
        assert!(err.contains("row-misaligns"), "{err}");
    }

    #[test]
    fn a_shared_column_array_writes_the_same_parquet_as_two_copies() {
        use arrow::array::{ArrayRef, Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        // `write_scored_table` hands the same Arrow array to `protein`/`protein_group` and
        // to the three aliases of the pooled q. This asserts the file that produces is
        // byte-for-byte the file `write_table` produced from separate, equal vectors --
        // the output-equality claim of that change, on the mechanism it rests on.
        //
        // The row count is what makes the comparison capable of failing. `write_table`
        // encodes in 65,536-row chunks and `write_scored_table` writes one batch, so below
        // 65,536 rows both write exactly one chunk and no arrangement of the data could
        // tell them apart -- the test asserted the new shape at a size where the two
        // writers cannot diverge. Chunking is not free of that risk in general: mumdia-io's
        // `an_entirely_null_column_keeps_its_rows_but_not_its_page_framing` shows a column
        // type whose page framing DOES move at a chunk boundary. 3 chunks and a short last
        // one is the shape that can catch it here, on the scored table's own duplicate-array
        // arrangement and with a string column in it. The writer's 1,048,576-row row-group
        // seam is the other place the two could part; mumdia-io pins that one directly, at
        // 1,048,583 rows, in `the_row_groups_fall_in_the_same_places_past_the_row_group_maximum`.
        let n = 3 * (1usize << 16) + 7;
        let dir = std::env::temp_dir().join("mumdia_scored_share_test");
        std::fs::create_dir_all(&dir).unwrap();
        let pid = std::process::id();
        let a = dir.join(format!("copies_{pid}.parquet"));
        let b = dir.join(format!("shared_{pid}.parquet"));
        let prot: Vec<String> = (0..n)
            .map(|i| format!("sp|P{:05}|PROT_HUMAN", i % 700))
            .collect();
        let q: Vec<f64> = (0..n).map(|i| (i % 97) as f64 / 97.0).collect();
        let rank: Vec<i32> = (0..n).map(|i| (i % 3) as i32).collect();
        write_table(
            a.to_str().unwrap(),
            vec![
                Col::Str("protein".into(), prot.clone()),
                Col::F64("q_value".into(), q.clone()),
                Col::Str("protein_group".into(), prot.clone()),
                Col::F64("global_q_value".into(), q.clone()),
                Col::I32("selected_peak_rank".into(), rank.clone()),
            ],
        )
        .unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("protein", DataType::Utf8, false),
            Field::new("q_value", DataType::Float64, false),
            Field::new("protein_group", DataType::Utf8, false),
            Field::new("global_q_value", DataType::Float64, false),
            Field::new("selected_peak_rank", DataType::Int32, false),
        ]));
        let protein: ArrayRef = Arc::new(StringArray::from(prot));
        let qq: ArrayRef = Arc::new(Float64Array::from(q));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                protein.clone(),
                qq.clone(),
                protein,
                qq,
                Arc::new(Int32Array::from(rank)),
            ],
        )
        .unwrap();
        let mut w = mumdia_io::table::BatchWriter::new(b.to_str().unwrap(), schema).unwrap();
        w.write(&batch).unwrap();
        assert_eq!(w.close().unwrap(), n as u64);
        assert_eq!(
            std::fs::read(&a).unwrap(),
            std::fs::read(&b).unwrap(),
            "sharing one array across duplicate columns must not change the parquet"
        );
    }

    #[test]
    fn a_one_sided_population_is_refused_with_both_counts_named() {
        assert!(require_both_labels(10, 5).is_ok());
        for (t, d) in [(10usize, 0usize), (0, 10), (0, 0)] {
            let err = require_both_labels(t, d).unwrap_err().to_string();
            // Which side is missing points at a different cause, so both counts
            // have to appear in the message.
            assert!(err.contains(&format!("targets={t}")), "{err}");
            assert!(err.contains(&format!("decoys={d}")), "{err}");
            assert!(err.contains("valid FDR"), "{err}");
        }
    }

    #[test]
    fn competed_feature_schema_must_match_id_and_ordered_columns() {
        let expected = FeatureSchema {
            feature_columns: vec!["a".into(), "b".into()],
            schema_id: "schema-a".into(),
        };
        let same = FeatureSchema {
            feature_columns: vec!["a".into(), "b".into()],
            schema_id: "schema-a".into(),
        };
        assert!(validate_feature_schema(&expected, &same, "same.parquet").is_ok());

        let reordered = FeatureSchema {
            feature_columns: vec!["b".into(), "a".into()],
            schema_id: "schema-a".into(),
        };
        assert!(validate_feature_schema(&expected, &reordered, "reordered.parquet").is_err());

        let different_id = FeatureSchema {
            feature_columns: vec!["a".into(), "b".into()],
            schema_id: "schema-b".into(),
        };
        assert!(validate_feature_schema(&expected, &different_id, "id.parquet").is_err());
    }

    /// `run_psm_q` exactly as `run` computed it before `per_source_q`: one index vector per
    /// source, the pair form of the kernel.
    fn per_source_q_reference(
        source: &[u32],
        scores: &[f64],
        is_decoy: &[bool],
        is_entrapment: &[bool],
        is_real: &[bool],
        qmode: QMode,
        ratio: f64,
    ) -> Vec<f64> {
        let mut by_src: std::collections::BTreeMap<u32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, &s) in source.iter().enumerate() {
            by_src.entry(s).or_default().push(i);
        }
        let mut rq = vec![1.0f64; scores.len()];
        for (_s, idxs) in by_src {
            let q = match qmode {
                QMode::Decoy => {
                    let sd: Vec<(f64, bool)> =
                        idxs.iter().map(|&i| (scores[i], is_decoy[i])).collect();
                    target_decoy_q(&sd)
                }
                QMode::Entrapment => {
                    let sc: Vec<f64> = idxs.iter().map(|&i| scores[i]).collect();
                    let en: Vec<bool> = idxs.iter().map(|&i| is_entrapment[i]).collect();
                    let re: Vec<bool> = idxs.iter().map(|&i| is_real[i]).collect();
                    entrapment_q(&sc, &en, &re, ratio)
                }
            };
            for (k, &i) in idxs.iter().enumerate() {
                rq[i] = q[k];
            }
        }
        rq
    }

    #[test]
    fn per_source_q_reproduces_the_index_vector_form() {
        // Contiguous sources (the stage's own layout), one source, and a shuffled source
        // column that has to take the index path; ties in the scores; both nulls.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let n = 3_000usize;
        let scores: Vec<f64> = (0..n).map(|_| (next() % 400) as f64 * 0.25).collect();
        let is_decoy: Vec<bool> = (0..n).map(|_| next() % 3 == 0).collect();
        let is_ent: Vec<bool> = (0..n).map(|i| !is_decoy[i] && next() % 7 == 0).collect();
        let is_real: Vec<bool> = (0..n).map(|i| !is_decoy[i] && !is_ent[i]).collect();
        let contiguous: Vec<u32> = (0..n).map(|i| (i * 4 / n) as u32).collect();
        let single = vec![0u32; n];
        let shuffled: Vec<u32> = (0..n).map(|_| (next() % 4) as u32).collect();
        for qmode in [QMode::Decoy, QMode::Entrapment] {
            let pooled = match qmode {
                QMode::Decoy => {
                    let sd: Vec<(f64, bool)> = scores
                        .iter()
                        .copied()
                        .zip(is_decoy.iter().copied())
                        .collect();
                    target_decoy_q(&sd)
                }
                QMode::Entrapment => entrapment_q(&scores, &is_ent, &is_real, 1.5),
            };
            // The split form of the pooled q is the pair form.
            if qmode == QMode::Decoy {
                assert_eq!(target_decoy_q_split(&scores, &is_decoy), pooled);
            }
            for source in [&contiguous, &single, &shuffled] {
                let want = per_source_q_reference(
                    source, &scores, &is_decoy, &is_ent, &is_real, qmode, 1.5,
                );
                let got = per_source_q(
                    source, &scores, &is_decoy, &is_ent, &is_real, qmode, 1.5, &pooled,
                );
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn the_sidecar_files_are_removed_unless_kept() {
        let files: Vec<String> = ["h.parquet", "k.parquet", "o.parquet"]
            .iter()
            .map(|n| scratch(&format!("cleanup_{n}")))
            .collect();
        let refs: Vec<&str> = files.iter().map(String::as_str).collect();
        for f in &files {
            std::fs::write(f, b"12345").unwrap();
        }
        // Kept: nothing removed, nothing counted.
        assert_eq!(remove_sidecar_files(&refs, true), 0);
        assert!(files.iter().all(|f| std::path::Path::new(f).exists()));
        // Removed, and a file that is already gone is not an error.
        std::fs::remove_file(&files[1]).unwrap();
        assert_eq!(remove_sidecar_files(&refs, false), 10);
        assert!(files.iter().all(|f| !std::path::Path::new(f).exists()));
        // The flag's spellings.
        for (v, want) in [
            (None, false),
            (Some("1"), true),
            (Some(" TRUE "), true),
            (Some("yes"), true),
            (Some("on"), true),
            (Some("0"), false),
            (Some(""), false),
            (Some("no"), false),
        ] {
            assert_eq!(env_flag(v), want, "{v:?}");
        }
    }

    #[test]
    fn the_space_check_refuses_below_the_floor_and_names_the_way_out() {
        let (floor, estimate) = handoff_space(1_000, 100, HandoffFormat::Parquet);
        assert!(floor < estimate);
        // Half the raw f32 size and the raw size, plus the per-row metadata, keys and
        // output.
        assert_eq!(floor, 1_000 * 100 * 2 + 1_000 * 72);
        assert_eq!(estimate, 1_000 * 100 * 4 + 1_000 * 72);
        let (pin_floor, _) = handoff_space(1_000, 100, HandoffFormat::Pin);
        assert!(
            pin_floor > estimate,
            "a PIN value is never fewer than 9 bytes"
        );
        let e = space_verdict(
            floor - 1,
            floor,
            estimate,
            "D:/x",
            1_000,
            100,
            HandoffFormat::Parquet,
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("MUMDIA_SIDECAR_DIR") && e.contains("--work-dir"),
            "{e}"
        );
        assert!(
            e.contains("D:/x") && e.contains("1000 PSMs x 100 features"),
            "{e}"
        );
        // Between the floor and the estimate it warns and goes on; above it, silently.
        space_verdict(
            floor,
            floor,
            estimate,
            "D:/x",
            1_000,
            100,
            HandoffFormat::Parquet,
        )
        .unwrap();
        space_verdict(
            estimate,
            floor,
            estimate,
            "D:/x",
            1_000,
            100,
            HandoffFormat::Parquet,
        )
        .unwrap();
        // An interpreter that cannot be started skips the check.
        assert_eq!(
            free_bytes("mumdia-no-such-interpreter-for-this-test", "."),
            None
        );
        check_sidecar_space(
            Some("mumdia-no-such-interpreter-for-this-test"),
            ".",
            usize::MAX / 4,
            1_000,
            HandoffFormat::Parquet,
        )
        .unwrap();
    }

    #[test]
    fn the_sidecar_directory_is_named_by_its_handoff() {
        let paths = SidecarPaths {
            handoff: "D:/work/sc/rescore_x_1.features.parquet".to_string(),
            out: "D:/work/sc/rescore_x_1_out.parquet".to_string(),
            foldkeys: "D:/work/sc/rescore_x_1.foldkeys.parquet".to_string(),
            use_pq: true,
        };
        assert_eq!(paths.dir(), "D:/work/sc");
        assert_eq!(paths.format(), HandoffFormat::Parquet);
        assert_eq!(paths.files().len(), 3);
        let bare = SidecarPaths {
            handoff: "rescore.pin".to_string(),
            out: String::new(),
            foldkeys: String::new(),
            use_pq: false,
        };
        assert_eq!(bare.dir(), ".");
        assert_eq!(bare.format(), HandoffFormat::Pin);
    }

    #[test]
    fn the_collapse_is_skipped_exactly_when_no_candidate_repeats() {
        // The map `run` builds for the top-K collapse, as it builds it: the pair count it
        // ends with is `n` exactly when no (source, candidate_id) pair repeats.
        let map_says_unique = |source: &[u32], cid: &[u32]| {
            let pairs: std::collections::HashSet<(u32, u32)> =
                source.iter().copied().zip(cid.iter().copied()).collect();
            pairs.len() == cid.len()
        };
        let cases: Vec<(Vec<u32>, Vec<u32>)> = vec![
            (vec![], vec![]),
            (vec![0], vec![7]),
            (vec![0, 0, 0], vec![3, 1, 2]),
            // The same candidate in two runs is two candidates.
            (vec![0, 0, 1, 1], vec![5, 9, 5, 9]),
            // A repeat inside one run.
            (vec![0, 0, 1], vec![5, 5, 6]),
            (vec![0, 1, 1], vec![5, 6, 6]),
            // Ids on the 64-bit word edges.
            (vec![0, 0, 0], vec![63, 64, 0]),
            (vec![0, 0], vec![64, 64]),
        ];
        for (source, cid) in &cases {
            assert_eq!(
                every_candidate_is_unique(source, cid),
                map_says_unique(source, cid),
                "{source:?} {cid:?}"
            );
        }
        // A source that goes back down cannot be decided by the per-source bitset, even
        // when every pair is unique: the map is asked instead.
        assert!(map_says_unique(&[1, 0], &[4, 4]));
        assert!(!every_candidate_is_unique(&[1, 0], &[4, 4]));
    }

    #[test]
    fn sidecar_scores_require_exact_unique_finite_coverage() {
        assert_eq!(
            align_sidecar_scores(&[1, 0], &[2.0, 1.0], 2, "test").unwrap(),
            vec![1.0, 2.0]
        );
        assert!(align_sidecar_scores(&[0], &[1.0], 2, "test").is_err());
        assert!(align_sidecar_scores(&[0, 0], &[1.0, 2.0], 2, "test").is_err());
        assert!(align_sidecar_scores(&[0, 2], &[1.0, 2.0], 2, "test").is_err());
        assert!(align_sidecar_scores(&[0, 1], &[1.0, f64::NAN], 2, "test").is_err());
    }

    #[test]
    fn picked_peptide_exact_tie_is_won_by_decoy() {
        // The target is deliberately first: row order must not win an exact
        // paired target/decoy tie.
        let mut keys = vec![0u32, 0u32];
        let mut scores = vec![5.0, 5.0];
        let mut decoys = vec![false, true];
        let mut entrapments = vec![false, false];
        let mut real = vec![true, false];
        for key in 1..=100 {
            keys.push(key);
            scores.push(4.0 - key as f64 * 0.01);
            decoys.push(false);
            entrapments.push(false);
            real.push(true);
        }

        let q = grouped_q(
            &keys,
            &scores,
            &decoys,
            &entrapments,
            &real,
            QMode::Decoy,
            1.0,
        );
        assert_eq!(q[0], 1.0, "tied target must be the losing sibling");
        assert!(q[1] < 0.05, "tied decoy should own the picked-group q");
    }

    /// `grouped_q` exactly as it was before the dense rewrite: a `HashMap` keyed on the
    /// group id, read back out through `best.keys()` and a second `HashMap` of q-values.
    /// The tests below pin the rewrite against THIS, not against its own shape.
    fn grouped_q_hashed(
        keys: &[u32],
        scores: &[f64],
        is_decoy: &[bool],
        is_entrapment: &[bool],
        is_real: &[bool],
        qmode: QMode,
        ratio: f64,
    ) -> Vec<f64> {
        let n = scores.len();
        let mut best: HashMap<u32, (f64, bool, bool, bool, usize)> = HashMap::new();
        for i in 0..n {
            if matches!(qmode, QMode::Entrapment) && is_decoy[i] {
                continue;
            }
            let e = best.entry(keys[i]).or_insert((
                f64::NEG_INFINITY,
                is_decoy[i],
                is_entrapment[i],
                is_real[i],
                i,
            ));
            let incoming_null = match qmode {
                QMode::Decoy => is_decoy[i],
                QMode::Entrapment => is_entrapment[i],
            };
            let current_null = match qmode {
                QMode::Decoy => e.1,
                QMode::Entrapment => e.2,
            };
            if scores[i] > e.0 || (scores[i] == e.0 && incoming_null && !current_null) {
                *e = (scores[i], is_decoy[i], is_entrapment[i], is_real[i], i);
            }
        }
        let ks: Vec<u32> = best.keys().cloned().collect();
        let qv = match qmode {
            QMode::Decoy => {
                let sd: Vec<(f64, bool)> = ks.iter().map(|k| (best[k].0, best[k].1)).collect();
                target_decoy_q(&sd)
            }
            QMode::Entrapment => {
                let sc: Vec<f64> = ks.iter().map(|k| best[k].0).collect();
                let e: Vec<bool> = ks.iter().map(|k| best[k].2).collect();
                let r: Vec<bool> = ks.iter().map(|k| best[k].3).collect();
                entrapment_q(&sc, &e, &r, ratio)
            }
        };
        let qmap: HashMap<u32, f64> = ks.into_iter().zip(qv).collect();
        let mut out = vec![1.0f64; n];
        for (key, (_, _, _, _, row)) in best {
            out[row] = qmap[&key];
        }
        out
    }

    /// A deterministic generator: populations with exact score ties (which is what the
    /// tie-break arm exists for), both labels, entrapment rows, and the two degenerate
    /// scores the `or_insert` placeholder interacts with.
    struct GroupedQPopulation {
        keys: Vec<u32>,
        scores: Vec<f64>,
        is_decoy: Vec<bool>,
        is_entrapment: Vec<bool>,
        is_real: Vec<bool>,
    }

    fn grouped_q_population(n: usize, group_stride: u32, seed: u64) -> GroupedQPopulation {
        let mut x = seed | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut keys = Vec::with_capacity(n);
        let mut scores = Vec::with_capacity(n);
        let mut is_decoy = Vec::with_capacity(n);
        let mut is_entrapment = Vec::with_capacity(n);
        let mut is_real = Vec::with_capacity(n);
        for _ in 0..n {
            let r = next();
            // Few groups relative to rows, so most groups really compete.
            keys.push((r % (n as u64 / 3 + 1)) as u32 * group_stride);
            // A coarse score grid forces exact ties.
            scores.push(match r % 41 {
                0 => f64::NEG_INFINITY,
                1 => f64::NAN,
                k => (k % 7) as f64,
            });
            let d = r % 5 == 0;
            let e = !d && r % 11 == 0;
            is_decoy.push(d);
            is_entrapment.push(e);
            is_real.push(!d && !e);
        }
        GroupedQPopulation {
            keys,
            scores,
            is_decoy,
            is_entrapment,
            is_real,
        }
    }

    #[test]
    fn the_dense_group_reduction_reproduces_the_hashed_one() {
        // The output-equality claim of the dense rewrite, against the previous
        // implementation rather than against the new shape, in both q modes and on both
        // arms of the density gate. At 4,000 rows and ~1,334 groups, `group_stride` 1
        // indexes the keys directly (span 1,334) and 100_000 blows the span past the gate
        // and interns. Stride 9 is the case that matters most here: span 12,002 over
        // 4,000 rows is the band a `span <= 4n` gate indexed directly and `span <= n`
        // interns, so this is the population whose ARM the gate change moved, compared
        // against a reference that has no gate at all.
        assert_eq!(
            std::mem::size_of::<Option<GroupBest>>(),
            24,
            "the dense array's per-group cost is the memory claim: a niche in one of the \
             bools must absorb the Option discriminant"
        );
        for stride in [1u32, 9, 100_000] {
            for seed in [1u64, 2, 3, 4] {
                let p = grouped_q_population(4_000, stride, seed);
                let (keys, scores) = (&p.keys, &p.scores);
                let (d, e, r) = (&p.is_decoy, &p.is_entrapment, &p.is_real);
                for qmode in [QMode::Decoy, QMode::Entrapment] {
                    let want = grouped_q_hashed(keys, scores, d, e, r, qmode, 1.5);
                    let got = grouped_q(keys, scores, d, e, r, qmode, 1.5);
                    assert_eq!(
                        want.iter().map(|q| q.to_bits()).collect::<Vec<_>>(),
                        got.iter().map(|q| q.to_bits()).collect::<Vec<_>>(),
                        "stride {stride} seed {seed}"
                    );
                }
            }
        }
        // Empty input, and a population whose every row is skipped in entrapment mode.
        assert!(grouped_q(&[], &[], &[], &[], &[], QMode::Decoy, 1.0).is_empty());
        assert_eq!(
            grouped_q(
                &[3, 3],
                &[1.0, 2.0],
                &[true, true],
                &[false, false],
                &[false, false],
                QMode::Entrapment,
                1.0
            ),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn a_sparse_key_space_does_not_allocate_by_its_span() {
        // The guard the dense path needs: `base_peptide_id` is dense over the LIBRARY, so
        // a competed table of a few rows can carry ids in the hundreds of millions, and an
        // array indexed by them would be gigabytes. The gate must send that to the
        // interning arm, and the interning arm must number the groups 0..k.
        let sparse: Vec<u32> = vec![900_000_000, 12, 900_000_000, 4_000_000];
        let (ids, span) = dense_group_ids(&sparse);
        assert_eq!(span, 3, "one slot per distinct group, not per id value");
        assert_eq!(ids.as_ref(), &[0u32, 1, 0, 2]);

        // Dense enough to index directly: the ids are the keys themselves.
        let dense: Vec<u32> = (0..64u32).collect();
        let (ids, span) = dense_group_ids(&dense);
        assert_eq!(span, 64);
        assert_eq!(ids.as_ref(), dense.as_slice());

        // The band an earlier `span <= 4n` gate admitted, and this one does not. The
        // direct array is 24 bytes per SLOT, so a span of 3n costs 72 bytes per ROW --
        // more than the interning arm (4n for the ids plus 24k for k <= n groups, about
        // 28n) and more than the `HashMap<u32, GroupBest>` the dense rewrite replaced
        // (33 bytes over next_pow2(8n/7) buckets, about 44n). `base_peptide_id` against a
        // library with a few times more base peptides than the pool has rows is exactly
        // this shape, so the gate that was meant to guard it was admitting it.
        let banded: Vec<u32> = (0..2_000u32).map(|i| i * 3).collect();
        let (ids, span) = dense_group_ids(&banded);
        assert_eq!(
            span, 2_000,
            "a span of 5,998 over 2,000 rows must intern rather than index"
        );
        assert_eq!(ids[0], 0);
        assert_eq!(ids[1_999], 1_999);
    }

    #[test]
    fn entrapment_group_competition_excludes_decoys() {
        // A target and its paired decoy SHARE `base_peptide_id`, which is the grouping
        // key. Under entrapment q, `incoming_null` is `is_entrapment`, false for both, so
        // the group winner used to be whichever simply scored higher. When that was the
        // decoy, its tuple counted toward neither the entrapment nor the real population
        // -- the group vanished from the analysis -- and the real target was assigned 1.0.
        // The leak metric and the target count are both computed from this q, so the
        // FDR-validity instrument was measured on a decoy-thinned population.
        //
        // Group 7: a decoy outscoring its real target. Group 8: a real target alone.
        let keys = vec![7u32, 7u32, 8u32];
        let scores = vec![1.0, 9.0, 5.0];
        let is_decoy = vec![false, true, false];
        let is_entrapment = vec![false, false, false];
        let is_real = vec![true, false, true];

        let q = grouped_q(
            &keys,
            &scores,
            &is_decoy,
            &is_entrapment,
            &is_real,
            QMode::Entrapment,
            1.0,
        );
        // The real target of group 7 keeps a meaningful q: the decoy did not delete it.
        assert!(
            q[0] < 1.0,
            "the real target must survive the group, got q = {:?}",
            q
        );
        // The decoy itself gets no entrapment-calibrated group q.
        assert_eq!(q[1], 1.0);
        assert!(q[2] < 1.0);

        // Decoy mode is unchanged: there the decoy IS the null and must compete, and an
        // exact tie goes to it.
        let qd = grouped_q(
            &keys,
            &scores,
            &is_decoy,
            &is_entrapment,
            &is_real,
            QMode::Decoy,
            1.0,
        );
        assert_eq!(
            qd[0], 1.0,
            "picked-TDC: the higher-scoring decoy wins group 7"
        );
        assert!(qd[1] <= 1.0);
    }
}

#[cfg(test)]
mod matrix_ceiling_tests {
    use super::*;

    #[test]
    fn the_ceiling_is_judged_on_the_f32_matrix_before_allocation() {
        // 1,000,000 PSMs x 387 features: 1.548 GB as f32. The old estimate
        // (8 bytes + a 24-byte spine per PSM) was 3.12 GB, so a 2 GiB ceiling used to
        // refuse a matrix that fits with room to spare.
        let bytes = feature_matrix_bytes(1_000_000, 387).unwrap();
        assert_eq!(bytes, 1_548_000_000);
        assert_eq!(matrix_ceiling_exceeded(bytes, 2.0), None);
        assert_eq!(matrix_ceiling_exceeded(bytes, 1.0), Some(1.0));
        // 0 disables the ceiling; an overflowing product is refused rather than wrapped.
        assert_eq!(matrix_ceiling_exceeded(bytes, 0.0), None);
        assert_eq!(feature_matrix_bytes(usize::MAX, 2), None);
    }
}
