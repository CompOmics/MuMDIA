//! Compete step `mumdia compete` (docs/11_compete_rescore_fdr.md): within each
//! competition group keep only the best-scoring candidate before target-decoy
//! counting, so multiple plausible candidates for one elution peak cannot inflate
//! discoveries. MVP groups by base peptide (target + its decoy + charge/mod
//! variants); the grouping is configurable.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use arrow::array::{Array, ArrayRef, Float64Array, Int32Array, StringArray, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use mumdia_core::config::{CompeteConfig, CompeteGroupBy, CompetitionMode};
use mumdia_core::rejection::RejectionReason;
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{require_no_nulls, write_table, BatchWriter, Col, TableFile};
use serde_json::json;
use tracing::{info, warn};

use crate::stages::features::FeatureSchema;

/// Competition group key: `(base-or-peptidoform id, label code, bucket, peak rank)`.
/// Fixed size, so a whole run's keys are one flat buffer rather than a String per PSM.
type GroupKey = (u32, u8, i64, i32);

/// Rows per streamed batch of a single key column.
const KEY_BATCH_ROWS: usize = 1 << 16;

pub struct CompeteParams<'a> {
    pub features: &'a str,
    pub out: &'a str,
    pub cfg: &'a CompeteConfig,
    pub config_hash: &'a str,
}

pub fn run(p: CompeteParams) -> Result<u64> {
    let t0 = Instant::now();
    // `--out` must not be one of this stage's own inputs: every input is read
    // before the output is published, so writing over one replaces it and exits 0
    // (docs/31 F6). The shared guard existed and was wired into two stages.
    mumdia_io::refuse_output_over_input(p.out, &[("--features", p.features)])?;
    // Footer-only open. The key columns below stream one at a time and the feature columns
    // (hundreds of them) are never materialised: the previous path read the whole features
    // table into Arrow and then copied every column into an owned Vec, so compete held two
    // full copies of the widest artifact in the run. Now it holds the key columns plus one
    // batch while the surviving rows are copied through to the output.
    let t = TableFile::open(p.features)?;
    let n = t.nrows;
    let audit = p.cfg.emit_competition_audit;
    // The label is a one-byte code in the key and nowhere else, so it is read as codes.
    // A `Vec<String>` here was one small heap block per PSM for a value only ever compared
    // against two literals; the Strings are read back only for the opt-in audit sidecar,
    // which is the one consumer that prints them.
    let label_code = label_codes(&t)?;
    let prelim = t.f64("prelim_score")?;
    // Top-K peak rank (#7). Part of the competition key so peaks of one candidate
    // compete only within their own rank (a lower-scoring peak of a candidate must
    // not eliminate a sibling's better peak on prelim score before rescore picks).
    // Missing -> 0 (single-apex), so the grouping is unchanged when promotion is off;
    // the output then carries a synthesised all-zero column, as before.
    let peak_rank_col = t.i32("peak_rank").ok();
    let synth_peak_rank = peak_rank_col.is_none();
    let peak_rank: Vec<i32> = peak_rank_col.unwrap_or_else(|| vec![0; n]);
    let schema = FeatureSchema::read(p.features)?;
    let feat_names = &schema.feature_columns;

    let by_pform_charge = matches!(p.cfg.group_by, CompeteGroupBy::PeptidoformCharge);
    // Only read what the grouping (and the optional audit) needs. `base_peptide_id` is
    // dead weight under the default peptidoform-charge grouping, and `candidate_id` is
    // read by the audit sidecar alone; both columns are still required (and type-checked)
    // by the pass-through copy below, which writes every bookkeeping column.
    let base: Vec<u32> = if by_pform_charge {
        Vec::new()
    } else {
        t.u32("base_peptide_id")?
    };
    let apex_rt: Vec<f64> = if matches!(p.cfg.group_by, CompeteGroupBy::Apex) {
        t.f64("apex_rt")?
    } else {
        Vec::new()
    };
    // `charge` is a minimal feature column (present in every set), stored as f64.
    // Only the peptidoform-charge grouping needs it.
    let charge: Option<Vec<f64>> = if by_pform_charge {
        Some(t.f64("charge").map_err(|_| {
            anyhow!("compete group_by=peptidoform_charge requires a 'charge' feature column")
        })?)
    } else {
        None
    };
    // Dense peptidoform id by first appearance (deterministic) so the fixed-size
    // tuple key can separate modforms without allocating a String per PSM. Built
    // only for the peptidoform-charge grouping; empty otherwise. Streamed, so the
    // peptidoform column is never materialised: the map holds one String per DISTINCT
    // peptidoform, not one per row.
    let pform_id: Vec<u32> = if by_pform_charge {
        dense_peptidoform_ids(&t)?
    } else {
        Vec::new()
    };

    // Competition group members. The label is part of the key so a target is NOT
    // competed against its own decoy: the decoy population must survive for the
    // rescorer/FDR to have a valid null (otherwise decoys are depleted and FDR is
    // badly underestimated). Competition only arbitrates redundant charge/mod
    // variants within targets and within decoys.
    // Key is a fixed-size tuple (base-or-pform id, label_code, bucket) instead of a
    // freshly-allocated String per PSM. Precursor grouping uses a constant bucket
    // (0) so its equivalence classes are unchanged.
    //
    // The groups are a SORTED `(key, row)` array, not a `HashMap<key, Vec<row>>`. Under
    // the shipped `group_by = peptidoform_charge` nearly every group is a singleton, so
    // the map was about one small heap block per PSM and it was this stage's dominant
    // memory term; a grouped search that holds several bands at once dies on the kernel's
    // per-process mapping limit long before it runs out of bytes. The resolver walks
    // contiguous runs of equal keys instead, which visits the same groups in the same
    // sorted-key order with members in the same ascending row order.
    let mut entries: Vec<(GroupKey, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let pk = peak_rank[i];
        let key = match p.cfg.group_by {
            CompeteGroupBy::BasePeptide => (base[i], label_code[i], 0i64, pk),
            CompeteGroupBy::Apex => {
                let bucket = (apex_rt[i] / p.cfg.apex_rt_tolerance_s).round() as i64;
                (base[i], label_code[i], bucket, pk)
            }
            CompeteGroupBy::PeptidoformCharge => {
                // pform_id separates modforms; charge in the bucket separates
                // charges -> one group per peptidoform+charge (precursor-level).
                let c = charge.as_ref().unwrap()[i].round() as i64;
                (pform_id[i], label_code[i], c, pk)
            }
        };
        entries.push((key, i));
    }
    entries.sort_unstable();

    // Per-candidate unique-fragment evidence for the `unique_evidence` mode. Prefers
    // an explicit `unique_fragment_count` feature; otherwise approximates it as
    // matched-fragment count discounted by the contested fraction; None if neither
    // is available (mode then falls back to winner-take-all). Read only in that mode:
    // the other modes never consult it.
    let unique_ev: Option<Vec<f64>> = if matches!(p.cfg.mode, CompetitionMode::UniqueEvidence) {
        let unique_ev_src = unique_evidence_with_source(&t);
        // The mode keeps any non-winner whose unique evidence >= unique_evidence_min_fragments.
        // Warn when NOTHING in this run can fall below that threshold, because then the mode is
        // silently identical to CompetitionMode::None.
        //
        // Tested on the VALUES, not on which column the estimate came from. Keying on the
        // column name missed the common case: `peak_contested_frac` is part of the Extended
        // feature set unconditionally, so the estimate always reports itself as
        // "contested-discounted" even when that column is all zeros (competition features off),
        // which discounts nothing and leaves the raw matched count -- exactly the no-op the
        // warning exists to announce.
        if let Some((ev, src)) = unique_ev_src.as_ref() {
            let thr = p.cfg.unique_evidence_min_fragments as f64;
            let below = ev.iter().filter(|v| **v < thr).count();
            if below == 0 {
                warn!(
                    source = src,
                    threshold = thr,
                    candidates = ev.len(),
                    "compete mode=unique_evidence: no candidate's unique evidence falls below \
                     compete.unique_evidence_min_fragments, so NOTHING will be removed and this \
                     run is equivalent to mode=none. Enable the contested/competition features \
                     so the evidence is actually discounted, or raise the threshold."
                );
            }
        }
        if unique_ev_src.is_none() {
            warn!(
                "compete mode=unique_evidence: no unique_fragment_count / \
                 (n_matched_fragments, peak_contested_frac/contested_frac) columns; \
                 falling back to winner-take-all"
            );
        }
        unique_ev_src.map(|(v, _)| v)
    } else {
        None
    };

    // Resolve each group under the configured competition mode.
    let (keep, removed) = resolve_competition(
        &entries,
        &prelim,
        p.cfg.mode,
        p.cfg.margin,
        p.cfg.unique_evidence_min_fragments,
        unique_ev.as_deref(),
    );
    drop(entries);

    let rows = copy_kept_rows(
        &t,
        p.out,
        feat_names,
        synth_peak_rank,
        &keep,
        COMPETED_ROW_GROUP_ROWS,
    )?;
    // Feature schema companion: unchanged feature list, so rescore validates the same schema.
    mumdia_io::json::write_json(&format!("{}.schema.json", p.out), &schema)?;

    // Competition audit sidecar (opt-in): one row per removed PSM with its winner. Lets a
    // post-hoc analysis see what competition removed without re-running the stage. This is
    // the only consumer of the candidate_id / label / peptidoform columns as VALUES, so
    // they are read here rather than held for the whole stage.
    if audit {
        let cid = t.u32("candidate_id")?;
        let label = t.str("label")?;
        let pform = t.str("peptidoform")?;
        let reason_of = |i: usize| {
            if label_code[i] == 1 {
                RejectionReason::OutcompetedByDecoy
            } else {
                RejectionReason::OutcompetedByTarget
            }
        };
        let audit_path = format!("{}.compete_audit.parquet", p.out);
        write_table(
            &audit_path,
            vec![
                Col::U32(
                    "candidate_id".into(),
                    removed.iter().map(|&(m, _)| cid[m]).collect(),
                ),
                Col::Str(
                    "label".into(),
                    removed.iter().map(|&(m, _)| label[m].clone()).collect(),
                ),
                Col::Str(
                    "peptidoform".into(),
                    removed.iter().map(|&(m, _)| pform[m].clone()).collect(),
                ),
                Col::U32(
                    "winner_candidate_id".into(),
                    removed.iter().map(|&(_, w)| cid[w]).collect(),
                ),
                Col::F64(
                    "loser_prelim".into(),
                    removed.iter().map(|&(m, _)| prelim[m]).collect(),
                ),
                Col::F64(
                    "winner_prelim".into(),
                    removed.iter().map(|&(_, w)| prelim[w]).collect(),
                ),
                Col::Str(
                    "rejection_reason".into(),
                    removed
                        .iter()
                        .map(|&(m, _)| reason_of(m).code().to_string())
                        .collect(),
                ),
            ],
        )?;
    }

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("input_rows".to_string(), json!(n));
    stats.insert("kept".to_string(), json!(rows));
    stats.insert("removed".to_string(), json!(removed.len()));
    ArtifactReport {
        logical_name: artifact::PSMS_COMPETED.0.to_string(),
        schema_name: artifact::PSMS_COMPETED.0.to_string(),
        schema_version: artifact::PSMS_COMPETED.1,
        stage: "compete".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({
            "group_by": format!("{:?}", p.cfg.group_by),
            "mode": format!("{:?}", p.cfg.mode),
        }),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;

    info!(
        input = n,
        kept = rows,
        removed = removed.len(),
        mode = ?p.cfg.mode,
        "compete: done"
    );
    Ok(rows)
}

/// The label column as one code per row: `target` -> 0, `decoy` -> 1, anything else -> 2,
/// which is exactly the resolution the competition key applies. Streamed, so the peak is
/// one byte per row plus one decoded batch. Nulls are refused, as `TableFile::str` refuses
/// them for a required column.
fn label_codes(t: &TableFile) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::with_capacity(t.nrows);
    t.for_each_batch(Some(&["label"]), KEY_BATCH_ROWS, |b| {
        let a = b
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("compete: column 'label' is not utf8"))?;
        require_no_nulls(a, "label", t.path(), out.len())?;
        for k in 0..a.len() {
            out.push(match a.value(k) {
                "target" => 0u8,
                "decoy" => 1u8,
                _ => 2u8,
            });
        }
        Ok(())
    })?;
    Ok(out)
}

/// Dense peptidoform ids by first appearance, streamed. Same values as numbering a
/// materialised `Vec<String>` of the column, without the String per row: the map holds one
/// key per DISTINCT peptidoform, and row order (hence the numbering) is the file's.
fn dense_peptidoform_ids(t: &TableFile) -> Result<Vec<u32>> {
    let mut ids: Vec<u32> = Vec::with_capacity(t.nrows);
    let mut seen: HashMap<String, u32> = HashMap::new();
    t.for_each_batch(Some(&["peptidoform"]), KEY_BATCH_ROWS, |b| {
        let a = b
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("compete: column 'peptidoform' is not utf8"))?;
        require_no_nulls(a, "peptidoform", t.path(), ids.len())?;
        for k in 0..a.len() {
            let s = a.value(k);
            match seen.get(s) {
                Some(&id) => ids.push(id),
                None => {
                    let next = seen.len() as u32;
                    seen.insert(s.to_string(), next);
                    ids.push(next);
                }
            }
        }
        Ok(())
    })?;
    Ok(ids)
}

/// The bookkeeping columns every competed table starts with, in order, with the types the
/// features stage writes them in. One place, so the pass-through below cannot drift from
/// the typed schema this stage used to re-declare column by column.
const META_COLUMNS: [(&str, DataType); 11] = [
    ("candidate_id", DataType::UInt32),
    ("peak_rank", DataType::Int32),
    ("label", DataType::Utf8),
    ("base_peptide_id", DataType::UInt32),
    ("peptidoform", DataType::Utf8),
    ("protein", DataType::Utf8),
    ("apex_rt", DataType::Float64),
    ("elution_lo", DataType::Float64),
    ("elution_hi", DataType::Float64),
    ("precursor_mz", DataType::Float64),
    ("prelim_score", DataType::Float64),
];

/// Input rows per streamed batch of the pass-through copy (~50 MB at ~400 f64 columns).
const COPY_BATCH_ROWS: usize = 1 << 14;

/// Row-group cap of the competed table, the same cap the rescore handoff writes with.
///
/// Rescore reads this file back batch by batch, and a parquet reader decodes a WHOLE row
/// group before it slices batches out of it, so on a wide table the row-group size is the
/// reader's working set; it is also what the writer buffers before each flush. At parquet's
/// default 1,048,576 rows and ~387 f64 feature columns that is ~3.2 GB decoded per group.
/// 131,072 rows is ~400 MB. Row-group boundaries are the only thing this moves; values and
/// row order are unchanged.
const COMPETED_ROW_GROUP_ROWS: usize = 131_072;

/// Copy the surviving rows (`keep`, sorted ascending) of the features table into `out`,
/// one input batch at a time, in exactly the column set and order the previous typed
/// rewrite produced: the 11 bookkeeping columns, then the schema's feature columns, all
/// non-nullable. The kept rows of a batch are one contiguous slice of `keep`, and `take`
/// preserves their order, so the output row order is unchanged.
fn copy_kept_rows(
    t: &TableFile,
    out: &str,
    feat_names: &[String],
    synth_peak_rank: bool,
    keep: &[usize],
    row_group_rows: usize,
) -> Result<u64> {
    let mut fields: Vec<Field> = Vec::with_capacity(META_COLUMNS.len() + feat_names.len());
    // Source column per output field; None = synthesised zeros (a pre-v2 features table
    // without `peak_rank`, which the typed path also emitted as zeros).
    let mut source: Vec<Option<String>> = Vec::with_capacity(fields.capacity());
    for (name, dt) in META_COLUMNS.iter() {
        fields.push(Field::new(*name, dt.clone(), false));
        source.push(if *name == "peak_rank" && synth_peak_rank {
            None
        } else {
            Some(name.to_string())
        });
    }
    for name in feat_names {
        fields.push(Field::new(name, DataType::Float64, false));
        source.push(Some(name.clone()));
    }
    for (f, src) in fields.iter().zip(&source) {
        if let Some(s) = src {
            let i = t
                .schema
                .index_of(s)
                .map_err(|_| anyhow!("compete: features table has no column '{s}'"))?;
            let dt = t.schema.field(i).data_type();
            if dt != f.data_type() {
                anyhow::bail!(
                    "compete: column '{s}' is {dt:?} in the features table, expected {:?}",
                    f.data_type()
                );
            }
        }
    }
    let out_schema = Arc::new(Schema::new(fields));
    let proj: Vec<&str> = source.iter().flatten().map(String::as_str).collect();
    let reader = t.batches(Some(&proj), COPY_BATCH_ROWS)?;
    let in_schema = reader.schema();
    let src_idx: Vec<Option<usize>> = source
        .iter()
        .map(|s| {
            s.as_ref()
                .map(|s| in_schema.index_of(s).expect("validated above"))
        })
        .collect();
    let mut w = BatchWriter::with_row_group_rows(out, out_schema.clone(), row_group_rows)?;
    let (mut row0, mut kp) = (0usize, 0usize);
    for b in reader {
        let b = b?;
        let row1 = row0 + b.num_rows();
        let start = kp;
        while kp < keep.len() && keep[kp] < row1 {
            kp += 1;
        }
        if kp > start {
            // `keep` is sorted and unique and every index in `keep[start..kp]` lies in this
            // batch, so when as many rows survive as the batch holds they ARE the batch, in
            // order, and `take` would be an identity copy. Under the shipped grouping that
            // is the normal case, and this batch is the widest artifact of the run: all
            // ~398 columns were being rebuilt to reproduce themselves. Pass the column
            // through (an Arc clone) instead. The synthesised peak_rank column has no
            // source array to pass through and is still built.
            let n_kept = kp - start;
            let idx = if n_kept == b.num_rows() {
                None
            } else {
                Some(UInt32Array::from(
                    keep[start..kp]
                        .iter()
                        .map(|&r| (r - row0) as u32)
                        .collect::<Vec<u32>>(),
                ))
            };
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(src_idx.len());
            for (si, f) in src_idx.iter().zip(out_schema.fields()) {
                arrays.push(match (si, &idx) {
                    (Some(i), None) => densify(b.column(*i).clone(), f)?,
                    (Some(i), Some(ix)) => densify(take(b.column(*i).as_ref(), ix, None)?, f)?,
                    (None, _) => Arc::new(Int32Array::from(vec![0i32; n_kept])),
                });
            }
            w.write(&RecordBatch::try_new(out_schema.clone(), arrays)?)?;
        }
        row0 = row1;
    }
    w.close()
}

/// Apply the typed getters' null policy to a taken column so the output stays non-nullable,
/// exactly as the old typed rewrite made it: f64 null -> NaN, utf8 null -> "", integer null
/// -> the buffer value. Columns without nulls (the normal case) pass through untouched.
fn densify(a: ArrayRef, f: &Field) -> Result<ArrayRef> {
    if a.null_count() == 0 {
        return Ok(a);
    }
    let n = a.len();
    Ok(match f.data_type() {
        DataType::Float64 => {
            let x = a
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("type validated");
            Arc::new(Float64Array::from_iter_values((0..n).map(|k| {
                if x.is_null(k) {
                    f64::NAN
                } else {
                    x.value(k)
                }
            })))
        }
        DataType::Utf8 => {
            let x = a
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("type validated");
            Arc::new(StringArray::from_iter_values((0..n).map(|k| {
                if x.is_null(k) {
                    ""
                } else {
                    x.value(k)
                }
            })))
        }
        DataType::Int32 => {
            let x = a
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("type validated");
            Arc::new(Int32Array::from_iter_values((0..n).map(|k| x.value(k))))
        }
        DataType::UInt32 => {
            let x = a
                .as_any()
                .downcast_ref::<UInt32Array>()
                .expect("type validated");
            Arc::new(UInt32Array::from_iter_values((0..n).map(|k| x.value(k))))
        }
        other => anyhow::bail!("compete: unsupported column type {other:?}"),
    })
}

/// Per-candidate unique-evidence estimate plus the name of the column it came from.
fn unique_evidence_with_source(t: &TableFile) -> Option<(Vec<f64>, &'static str)> {
    if let Some(u) = col_f64(t, "unique_fragment_count") {
        return Some((u, "unique_fragment_count"));
    }
    let nm = col_f64(t, "n_matched_fragments")?;
    let contested = prefer_peak_contested_fraction(
        col_f64(t, "peak_contested_frac"),
        col_f64(t, "contested_frac"),
    );
    match contested {
        Some(cf) => Some((
            nm.iter()
                .zip(cf)
                .map(|(n, c)| n * (1.0 - c).clamp(0.0, 1.0))
                .collect(),
            "contested-discounted n_matched_fragments",
        )),
        None => Some((nm, "n_matched_fragments")),
    }
}

/// The Extended set carries the contested fraction as `peak_contested_frac`; older
/// artifacts as `contested_frac`. Prefer the former.
fn prefer_peak_contested_fraction(
    peak_contested: Option<Vec<f64>>,
    legacy_contested: Option<Vec<f64>>,
) -> Option<Vec<f64>> {
    peak_contested.or(legacy_contested)
}

/// Read a column as f64, accepting an i32 column (widened) as well.
fn col_f64(t: &TableFile, name: &str) -> Option<Vec<f64>> {
    t.f64(name).ok().or_else(|| {
        t.i32(name)
            .ok()
            .map(|v| v.into_iter().map(|x| x as f64).collect())
    })
}

/// Resolve within-group competition per [`CompetitionMode`]. Returns the
/// sorted-unique kept row indices and the `(loser, winner)` removal pairs.
///
/// `entries` is `(key, row)` sorted ascending, so a group is a contiguous run of equal
/// keys and its members are in ascending row order. That is the same visiting order the
/// previous `HashMap<key, Vec<row>>` produced (keys sorted, members pushed in row order),
/// so the winner, the kept set and the removal pairs are unchanged. Deterministic: the
/// winner is the highest `prelim` (ties broken by smallest index).
fn resolve_competition(
    entries: &[(GroupKey, usize)],
    prelim: &[f64],
    mode: CompetitionMode,
    margin: f64,
    unique_min: usize,
    unique_ev: Option<&[f64]>,
) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut keep: Vec<usize> = Vec::new();
    let mut removed: Vec<(usize, usize)> = Vec::new();
    let mut g = 0usize;
    while g < entries.len() {
        let mut h = g + 1;
        while h < entries.len() && entries[h].0 == entries[g].0 {
            h += 1;
        }
        let members = &entries[g..h];
        g = h;
        let win = members
            .iter()
            .map(|&(_, i)| i)
            // `total_cmp`, not `partial_cmp(..).unwrap_or(Equal)`: this picks the
            // single row that survives competition, and treating every NaN
            // prelim_score as equal to every other score made that choice depend on
            // iteration order. `total_cmp` is a genuine total order, so the winner
            // is well defined even then, and the `.then(a.cmp(&b))` index tiebreak
            // keeps it deterministic.
            .min_by(|&a, &b| prelim[b].total_cmp(&prelim[a]).then(a.cmp(&b)))
            .expect("a run of equal keys is never empty");
        match mode {
            CompetitionMode::None | CompetitionMode::FeaturesOnly => {
                keep.extend(members.iter().map(|&(_, i)| i));
            }
            CompetitionMode::WinnerTakeAll => {
                keep.push(win);
                removed.extend(
                    members
                        .iter()
                        .map(|&(_, i)| i)
                        .filter(|&m| m != win)
                        .map(|m| (m, win)),
                );
            }
            CompetitionMode::UniqueEvidence => {
                keep.push(win);
                let thr = unique_min as f64;
                for &(_, m) in members {
                    if m == win {
                        continue;
                    }
                    if unique_ev.map(|u| u[m] >= thr).unwrap_or(false) {
                        keep.push(m);
                    } else {
                        removed.push((m, win));
                    }
                }
            }
            CompetitionMode::MarginGated => {
                keep.push(win);
                for &(_, m) in members {
                    if m == win {
                        continue;
                    }
                    if prelim[win] - prelim[m] >= margin {
                        removed.push((m, win));
                    } else {
                        keep.push(m);
                    }
                }
            }
        }
    }
    keep.sort_unstable();
    keep.dedup();
    (keep, removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_io::table::Col as IoCol;

    #[test]
    fn writing_the_output_over_the_input_is_refused() {
        // docs/31 F6: the features table is opened footer-only and streamed, so the read
        // completes before `AtomicPath::publish` renames over it. Nothing errored: the
        // widest artifact of the run, hundreds of columns and gigabytes on a real library,
        // was replaced by the competed subset at exit 0, recoverable only by re-running
        // extract and features.
        let dir = std::env::temp_dir().join(format!("mumdia_compete_guard_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let features = dir.join("features.parquet");
        mumdia_io::table::write_table(
            features.to_str().unwrap(),
            vec![mumdia_io::table::Col::U32(
                "candidate_id".into(),
                vec![1, 2],
            )],
        )
        .unwrap();
        let cfg = mumdia_core::config::CompeteConfig::default();
        let e = run(CompeteParams {
            features: features.to_str().unwrap(),
            out: features.to_str().unwrap(),
            cfg: &cfg,
            config_hash: "h",
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("its own input"), "{e}");
        assert!(
            mumdia_io::table::TableFile::open(features.to_str().unwrap())
                .unwrap()
                .nrows
                == 2,
            "the input must still be there"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One group of `members`, as the sorted `(key, row)` array the resolver takes.
    fn one_group(members: Vec<usize>) -> Vec<(GroupKey, usize)> {
        let mut e: Vec<(GroupKey, usize)> = members
            .into_iter()
            .map(|i| ((0u32, 0u8, 0i64, 0i32), i))
            .collect();
        e.sort_unstable();
        e
    }

    #[test]
    fn winner_take_all_keeps_only_winner() {
        let g = one_group(vec![0, 1, 2]);
        let prelim = [0.1, 0.9, 0.5];
        let (keep, removed) =
            resolve_competition(&g, &prelim, CompetitionMode::WinnerTakeAll, 0.0, 2, None);
        assert_eq!(keep, vec![1]);
        assert_eq!(removed.len(), 2);
    }

    #[test]
    fn none_and_features_only_keep_all() {
        let g = one_group(vec![0, 1, 2]);
        let prelim = [0.1, 0.9, 0.5];
        for mode in [CompetitionMode::None, CompetitionMode::FeaturesOnly] {
            let (keep, removed) = resolve_competition(&g, &prelim, mode, 0.0, 2, None);
            assert_eq!(keep, vec![0, 1, 2]);
            assert!(removed.is_empty());
        }
    }

    #[test]
    fn margin_gated_keeps_close_losers_removes_distant() {
        // winner idx1 (0.9); idx2 (0.85) within margin 0.1 -> kept; idx0 removed
        let g = one_group(vec![0, 1, 2]);
        let prelim = [0.1, 0.9, 0.85];
        let (keep, removed) =
            resolve_competition(&g, &prelim, CompetitionMode::MarginGated, 0.1, 2, None);
        assert_eq!(keep, vec![1, 2]);
        assert_eq!(removed, vec![(0, 1)]);
    }

    #[test]
    fn unique_evidence_keeps_losers_with_enough_evidence() {
        // winner idx1; idx0 unique 3 (>=2) kept; idx2 unique 1 removed
        let g = one_group(vec![0, 1, 2]);
        let prelim = [0.1, 0.9, 0.5];
        let ev = [3.0, 5.0, 1.0];
        let (keep, removed) = resolve_competition(
            &g,
            &prelim,
            CompetitionMode::UniqueEvidence,
            0.0,
            2,
            Some(&ev),
        );
        assert_eq!(keep, vec![0, 1]);
        assert_eq!(removed, vec![(2, 1)]);
    }

    #[test]
    fn unique_evidence_without_data_falls_back_to_winner_take_all() {
        let g = one_group(vec![0, 1, 2]);
        let prelim = [0.1, 0.9, 0.5];
        let (keep, _) =
            resolve_competition(&g, &prelim, CompetitionMode::UniqueEvidence, 0.0, 2, None);
        assert_eq!(keep, vec![1]);
    }

    #[test]
    fn unique_evidence_prefers_extended_peak_contested_fraction() {
        let selected = prefer_peak_contested_fraction(Some(vec![0.25]), Some(vec![0.75])).unwrap();
        assert_eq!(selected, vec![0.25]);
        let legacy = prefer_peak_contested_fraction(None, Some(vec![0.75])).unwrap();
        assert_eq!(legacy, vec![0.75]);
    }

    #[test]
    fn winner_take_all_is_deterministic_across_groups() {
        let mut g: Vec<(GroupKey, usize)> = vec![
            ((0u32, 0u8, 0i64, 0i32), 0),
            ((0u32, 0u8, 0i64, 0i32), 1),
            ((1u32, 1u8, 0i64, 0i32), 2),
            ((1u32, 1u8, 0i64, 0i32), 3),
        ];
        g.sort_unstable();
        let prelim = [0.2, 0.8, 0.9, 0.3];
        let (keep, _) =
            resolve_competition(&g, &prelim, CompetitionMode::WinnerTakeAll, 0.0, 2, None);
        assert_eq!(keep, vec![1, 2]); // winners of each group, sorted
    }

    #[test]
    fn winner_tie_breaks_to_smallest_index() {
        let g = one_group(vec![0, 1, 2]);
        let prelim = [0.9, 0.9, 0.1]; // tie between idx0 and idx1
        let (keep, _) =
            resolve_competition(&g, &prelim, CompetitionMode::WinnerTakeAll, 0.0, 2, None);
        assert_eq!(keep, vec![0]);
    }

    /// The grouping this stage used until 2026-09-22: one `Vec` per key in a `HashMap`,
    /// keys visited in sorted order, members in ascending row order. Kept here as the
    /// reference the sorted-run resolver has to reproduce exactly.
    fn resolve_via_hashmap(
        entries: &[(GroupKey, usize)],
        prelim: &[f64],
        mode: CompetitionMode,
        margin: f64,
        unique_min: usize,
        unique_ev: Option<&[f64]>,
    ) -> (Vec<usize>, Vec<(usize, usize)>) {
        let mut by_row: Vec<(GroupKey, usize)> = entries.to_vec();
        by_row.sort_unstable_by_key(|&(_, i)| i);
        let mut groups: HashMap<GroupKey, Vec<usize>> = HashMap::new();
        for (k, i) in by_row {
            groups.entry(k).or_default().push(i);
        }
        let mut group_keys: Vec<&GroupKey> = groups.keys().collect();
        group_keys.sort_unstable();
        let mut keep: Vec<usize> = Vec::new();
        let mut removed: Vec<(usize, usize)> = Vec::new();
        for gk in group_keys {
            let members = &groups[gk];
            let win = *members
                .iter()
                .min_by(|&&a, &&b| prelim[b].total_cmp(&prelim[a]).then(a.cmp(&b)))
                .unwrap();
            match mode {
                CompetitionMode::None | CompetitionMode::FeaturesOnly => {
                    keep.extend(members.iter().copied());
                }
                CompetitionMode::WinnerTakeAll => {
                    keep.push(win);
                    removed.extend(
                        members
                            .iter()
                            .copied()
                            .filter(|&m| m != win)
                            .map(|m| (m, win)),
                    );
                }
                CompetitionMode::UniqueEvidence => {
                    keep.push(win);
                    let thr = unique_min as f64;
                    for &m in members {
                        if m == win {
                            continue;
                        }
                        if unique_ev.map(|u| u[m] >= thr).unwrap_or(false) {
                            keep.push(m);
                        } else {
                            removed.push((m, win));
                        }
                    }
                }
                CompetitionMode::MarginGated => {
                    keep.push(win);
                    for &m in members {
                        if m == win {
                            continue;
                        }
                        if prelim[win] - prelim[m] >= margin {
                            removed.push((m, win));
                        } else {
                            keep.push(m);
                        }
                    }
                }
            }
        }
        keep.sort_unstable();
        keep.dedup();
        (keep, removed)
    }

    #[test]
    fn sorted_run_grouping_reproduces_the_hashmap_grouping() {
        // A deterministic population with singletons, multi-member groups, ties, a NaN
        // prelim and a signed-zero pair, over all five modes.
        let n = 600usize;
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut entries: Vec<(GroupKey, usize)> = Vec::with_capacity(n);
        let mut prelim: Vec<f64> = Vec::with_capacity(n);
        let mut ev: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let r = next();
            let key = (
                (r % 97) as u32,
                ((r >> 8) % 3) as u8,
                ((r >> 16) % 5) as i64,
                ((r >> 24) % 3) as i32,
            );
            entries.push((key, i));
            // Coarse quantisation so ties are common.
            prelim.push(match i {
                7 => f64::NAN,
                11 => -0.0,
                12 => 0.0,
                _ => ((r >> 32) % 11) as f64 / 10.0,
            });
            ev.push(((r >> 40) % 5) as f64);
        }
        let reference = entries.clone();
        entries.sort_unstable();
        for mode in [
            CompetitionMode::None,
            CompetitionMode::FeaturesOnly,
            CompetitionMode::WinnerTakeAll,
            CompetitionMode::MarginGated,
            CompetitionMode::UniqueEvidence,
        ] {
            let got = resolve_competition(&entries, &prelim, mode, 0.2, 2, Some(&ev));
            let want = resolve_via_hashmap(&reference, &prelim, mode, 0.2, 2, Some(&ev));
            assert_eq!(got.0, want.0, "kept rows differ for {mode:?}");
            assert_eq!(got.1, want.1, "removal pairs differ for {mode:?}");
        }
    }

    /// A minimal features table: the 11 bookkeeping columns plus two feature columns.
    fn write_features(path: &str, n: usize) {
        let f = |g: fn(usize) -> f64| (0..n).map(g).collect::<Vec<f64>>();
        mumdia_io::table::write_table(
            path,
            vec![
                IoCol::U32("candidate_id".into(), (0..n as u32).collect()),
                IoCol::I32("peak_rank".into(), vec![0; n]),
                IoCol::Str(
                    "label".into(),
                    (0..n)
                        .map(|i| if i % 3 == 0 { "decoy" } else { "target" }.to_string())
                        .collect(),
                ),
                IoCol::U32(
                    "base_peptide_id".into(),
                    (0..n as u32).map(|i| i / 2).collect(),
                ),
                IoCol::Str(
                    "peptidoform".into(),
                    (0..n).map(|i| format!("PEP{}", i / 2)).collect(),
                ),
                IoCol::Str("protein".into(), (0..n).map(|i| format!("P{i}")).collect()),
                IoCol::F64("apex_rt".into(), f(|i| i as f64)),
                IoCol::F64("elution_lo".into(), f(|i| i as f64 - 1.0)),
                IoCol::F64("elution_hi".into(), f(|i| i as f64 + 1.0)),
                IoCol::F64("precursor_mz".into(), f(|i| 400.0 + i as f64)),
                IoCol::F64("prelim_score".into(), f(|i| (i % 7) as f64 / 7.0)),
                IoCol::F64("charge".into(), f(|i| 2.0 + (i % 2) as f64)),
                IoCol::F64("n_matched_fragments".into(), f(|i| (i % 5) as f64)),
            ],
        )
        .unwrap();
    }

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "mumdia_compete_{tag}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn passing_a_whole_batch_through_equals_taking_every_row_of_it() {
        // The pass-through branch must be indistinguishable from the `take` it replaces.
        // Row 9 is dropped in the second copy, so that batch goes through `take`; the rows
        // both copies share have to agree column for column.
        let dir = tmp_dir("passthrough");
        let src_path = dir.join("features.parquet");
        let src = src_path.to_str().unwrap();
        write_features(src, 10);
        let t = TableFile::open(src).unwrap();
        let feats = vec!["charge".to_string(), "n_matched_fragments".to_string()];

        let all_path = dir.join("all.parquet");
        let all = all_path.to_str().unwrap();
        let keep_all: Vec<usize> = (0..10).collect();
        assert_eq!(
            copy_kept_rows(&t, all, &feats, false, &keep_all, 1 << 20).unwrap(),
            10
        );

        let sub_path = dir.join("sub.parquet");
        let sub = sub_path.to_str().unwrap();
        let keep_sub: Vec<usize> = (0..9).collect();
        assert_eq!(
            copy_kept_rows(&t, sub, &feats, false, &keep_sub, 1 << 20).unwrap(),
            9
        );

        let a = TableFile::open(all).unwrap();
        let b = TableFile::open(sub).unwrap();
        assert_eq!(a.column_names(), b.column_names());
        for name in ["candidate_id", "base_peptide_id"] {
            assert_eq!(a.u32(name).unwrap()[..9], b.u32(name).unwrap()[..]);
        }
        assert_eq!(
            a.i32("peak_rank").unwrap()[..9],
            b.i32("peak_rank").unwrap()[..]
        );
        for name in ["label", "peptidoform", "protein"] {
            assert_eq!(a.str(name).unwrap()[..9], b.str(name).unwrap()[..]);
        }
        for name in [
            "apex_rt",
            "elution_lo",
            "elution_hi",
            "precursor_mz",
            "prelim_score",
            "charge",
            "n_matched_fragments",
        ] {
            assert_eq!(a.f64(name).unwrap()[..9], b.f64(name).unwrap()[..]);
        }
        // And the pass-through copy really is the input, values and order.
        assert_eq!(
            a.u32("candidate_id").unwrap(),
            (0..10u32).collect::<Vec<_>>()
        );
        assert_eq!(
            a.f64("precursor_mz").unwrap(),
            t.f64("precursor_mz").unwrap()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_competed_table_is_written_in_capped_row_groups() {
        // Rescore reads this file back a row group at a time, so the cap is its working
        // set. It is the cap the rescore handoff writes with.
        assert_eq!(COMPETED_ROW_GROUP_ROWS, 131_072);
        let dir = tmp_dir("rowgroups");
        let src_path = dir.join("features.parquet");
        let src = src_path.to_str().unwrap();
        write_features(src, 10);
        let t = TableFile::open(src).unwrap();
        let out_path = dir.join("competed.parquet");
        let out = out_path.to_str().unwrap();
        let keep: Vec<usize> = (0..10).collect();
        copy_kept_rows(&t, out, &["charge".to_string()], false, &keep, 4).unwrap();
        let file = std::fs::File::open(out).unwrap();
        let builder =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let meta = builder.metadata();
        let sizes: Vec<i64> = (0..meta.num_row_groups())
            .map(|i| meta.row_group(i).num_rows())
            .collect();
        assert_eq!(sizes, vec![4, 4, 2]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn label_codes_match_the_string_column() {
        let dir = tmp_dir("labels");
        let p_path = dir.join("t.parquet");
        let p = p_path.to_str().unwrap();
        mumdia_io::table::write_table(
            p,
            vec![IoCol::Str(
                "label".into(),
                vec![
                    "target".to_string(),
                    "decoy".to_string(),
                    "target".to_string(),
                    "entrapment".to_string(),
                ],
            )],
        )
        .unwrap();
        let t = TableFile::open(p).unwrap();
        // The pre-2026-09-22 coding, from the materialised column.
        let want: Vec<u8> = t
            .str("label")
            .unwrap()
            .iter()
            .map(|s| match s.as_str() {
                "target" => 0u8,
                "decoy" => 1u8,
                _ => 2u8,
            })
            .collect();
        assert_eq!(label_codes(&t).unwrap(), want);
        assert_eq!(want, vec![0, 1, 0, 2]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn streamed_peptidoform_ids_match_first_appearance_numbering() {
        let dir = tmp_dir("pformids");
        let p_path = dir.join("t.parquet");
        let p = p_path.to_str().unwrap();
        let pforms: Vec<String> = ["B", "A", "B", "C", "A", "C", "D"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        mumdia_io::table::write_table(p, vec![IoCol::Str("peptidoform".into(), pforms.clone())])
            .unwrap();
        let t = TableFile::open(p).unwrap();
        // The pre-2026-09-22 numbering, from the materialised column.
        let mut seen: HashMap<&str, u32> = HashMap::new();
        let want: Vec<u32> = pforms
            .iter()
            .map(|s| {
                let next = seen.len() as u32;
                *seen.entry(s.as_str()).or_insert(next)
            })
            .collect();
        assert_eq!(dense_peptidoform_ids(&t).unwrap(), want);
        assert_eq!(want, vec![0, 1, 0, 2, 1, 2, 3]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
