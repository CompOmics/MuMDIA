//! Compete step `mumdia compete` (PLAN.md Section 4 Stage F, the compete step):
//! within each competition group keep only the best-scoring candidate before
//! target-decoy counting, so multiple plausible candidates for one elution peak
//! cannot inflate discoveries. MVP groups by base peptide (target + its decoy +
//! charge/mod variants); the grouping is configurable.

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
use mumdia_io::table::{write_table, BatchWriter, Col, TableFile};
use serde_json::json;
use tracing::{info, warn};

use crate::stages::features::FeatureSchema;

pub struct CompeteParams<'a> {
    pub features: &'a str,
    pub out: &'a str,
    pub cfg: &'a CompeteConfig,
    pub config_hash: &'a str,
}

pub fn run(p: CompeteParams) -> Result<u64> {
    let t0 = Instant::now();
    // Footer-only open. The key columns below stream one at a time and the feature columns
    // (hundreds of them) are never materialised: the previous path read the whole features
    // table into Arrow and then copied every column into an owned Vec, so compete held two
    // full copies of the widest artifact in the run. Now it holds the key columns plus one
    // batch while the surviving rows are copied through to the output.
    let t = TableFile::open(p.features)?;
    let n = t.nrows;
    let cid = t.u32("candidate_id")?;
    let label = t.str("label")?;
    let base = t.u32("base_peptide_id")?;
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
    // Only read what the grouping (and the optional audit) needs.
    let pform: Vec<String> = if by_pform_charge || p.cfg.emit_competition_audit {
        t.str("peptidoform")?
    } else {
        Vec::new()
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
    // only for the peptidoform-charge grouping; empty otherwise.
    let pform_id: Vec<u32> = if by_pform_charge {
        let mut ids = Vec::with_capacity(n);
        let mut seen: HashMap<&str, u32> = HashMap::new();
        for peptidoform in pform.iter().take(n) {
            let next = seen.len() as u32;
            ids.push(*seen.entry(peptidoform.as_str()).or_insert(next));
        }
        ids
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
    let mut groups: HashMap<(u32, u8, i64, i32), Vec<usize>> = HashMap::new();
    for i in 0..n {
        let label_code = match label[i].as_str() {
            "target" => 0u8,
            "decoy" => 1u8,
            _ => 2u8,
        };
        let pk = peak_rank[i];
        let key = match p.cfg.group_by {
            CompeteGroupBy::Precursor => (base[i], label_code, 0i64, pk),
            CompeteGroupBy::Apex => {
                let bucket = (apex_rt[i] / p.cfg.apex_rt_tolerance_s).round() as i64;
                (base[i], label_code, bucket, pk)
            }
            CompeteGroupBy::PeptidoformCharge => {
                // pform_id separates modforms; charge in the bucket separates
                // charges -> one group per peptidoform+charge (precursor-level).
                let c = charge.as_ref().unwrap()[i].round() as i64;
                (pform_id[i], label_code, c, pk)
            }
        };
        groups.entry(key).or_default().push(i);
    }

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
        &groups,
        &prelim,
        p.cfg.mode,
        p.cfg.margin,
        p.cfg.unique_evidence_min_fragments,
        unique_ev.as_deref(),
    );
    drop(groups);

    let rows = copy_kept_rows(&t, p.out, feat_names, synth_peak_rank, &keep)?;
    // Feature schema companion: unchanged feature list, so rescore validates the same schema.
    mumdia_io::json::write_json(&format!("{}.schema.json", p.out), &schema)?;

    // Competition audit sidecar (opt-in): one row per removed PSM with its winner. Lets a
    // post-hoc analysis see what competition removed without re-running the stage.
    if p.cfg.emit_competition_audit {
        let reason_of = |i: usize| {
            if label[i] == "decoy" {
                RejectionReason::OutcompetedByDecoy
            } else {
                RejectionReason::OutcompetedByTarget
            }
        };
        let audit = format!("{}.compete_audit.parquet", p.out);
        write_table(
            &audit,
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
    let mut w = BatchWriter::new(out, out_schema.clone())?;
    let (mut row0, mut kp) = (0usize, 0usize);
    for b in reader {
        let b = b?;
        let row1 = row0 + b.num_rows();
        let start = kp;
        while kp < keep.len() && keep[kp] < row1 {
            kp += 1;
        }
        if kp > start {
            let idx = UInt32Array::from(
                keep[start..kp]
                    .iter()
                    .map(|&r| (r - row0) as u32)
                    .collect::<Vec<u32>>(),
            );
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(src_idx.len());
            for (si, f) in src_idx.iter().zip(out_schema.fields()) {
                arrays.push(match si {
                    Some(i) => densify(take(b.column(*i).as_ref(), &idx, None)?, f)?,
                    None => Arc::new(Int32Array::from(vec![0i32; idx.len()])),
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
/// Deterministic: groups are visited in sorted key order; the winner is the
/// highest `prelim` (ties broken by smallest index).
fn resolve_competition(
    groups: &HashMap<(u32, u8, i64, i32), Vec<usize>>,
    prelim: &[f64],
    mode: CompetitionMode,
    margin: f64,
    unique_min: usize,
    unique_ev: Option<&[f64]>,
) -> (Vec<usize>, Vec<(usize, usize)>) {
    use std::cmp::Ordering::Equal;
    let mut group_keys: Vec<&(u32, u8, i64, i32)> = groups.keys().collect();
    group_keys.sort_unstable();
    let mut keep: Vec<usize> = Vec::new();
    let mut removed: Vec<(usize, usize)> = Vec::new();
    for gk in group_keys {
        let members = &groups[gk];
        let win = *members
            .iter()
            .min_by(|&&a, &&b| {
                prelim[b]
                    .partial_cmp(&prelim[a])
                    .unwrap_or(Equal)
                    .then(a.cmp(&b))
            })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn one_group(members: Vec<usize>) -> HashMap<(u32, u8, i64, i32), Vec<usize>> {
        let mut g = HashMap::new();
        g.insert((0u32, 0u8, 0i64, 0i32), members);
        g
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
        let mut g = HashMap::new();
        g.insert((0u32, 0u8, 0i64, 0i32), vec![0, 1]);
        g.insert((1u32, 1u8, 0i64, 0i32), vec![2, 3]);
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
}
