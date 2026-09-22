//! Isolation-window groups: the m/z bands a run is searched in, one band's library at a
//! time (`groups.window_groups` in the configuration).
//!
//! A DIA isolation window can only select precursors whose m/z lies inside it, so a group
//! of windows and the library band under it form a closed search: the seed, the RT
//! calibration windows, extract, features and compete of that group need nothing outside
//! the band. `plan` cuts the run's windows, in ascending m/z, into contiguous groups whose
//! bands hold about the same number of library precursors, estimated from the precursor
//! table's row-group statistics (row counts and m/z min/max) so that no table is read to
//! plan. Balancing by precursors rather than by windows matters: on an immunopeptidomics
//! library the per-window precursor counts differed by 3x across the m/z range.
//!
//! Overlapping window schemes are allowed. A band's m/z range is the union of its windows'
//! ranges, so where two adjacent windows overlap across a cut, the precursors in the
//! overlap belong to both bands and are searched twice; the pooling step keeps one row per
//! library-wide candidate. Precursors outside every window (charge states above the
//! acquisition's top window, typically) belong to no band and are never loaded, which is
//! the right outcome for something the run cannot measure; the plan reports how many.

use anyhow::{bail, Context, Result};
use mumdia_io::table::RowGroupStats;
use serde_json::{json, Value};

/// One window group and the library band it selects from.
#[derive(Clone, Debug, PartialEq)]
pub struct Band {
    /// Position in the plan, ascending m/z.
    pub index: usize,
    /// Lowest window lower bound and highest window upper bound of the group.
    pub mz_lo: f64,
    pub mz_hi: f64,
    /// Indices into the plan's window list (sorted by lower bound), contiguous.
    pub windows: Vec<usize>,
    /// Library precursors the band is estimated to hold, from the row-group statistics.
    pub est_precursors: f64,
}

/// The plan for a run: its bands and what the statistics say about coverage.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub bands: Vec<Band>,
    /// Isolation windows as `(lower, upper)`, sorted by lower bound; band window indices
    /// refer to this order.
    pub windows: Vec<(f64, f64)>,
    /// Precursors estimated to lie outside every window, i.e. in no band.
    pub est_unselectable: f64,
    /// Precursors estimated to lie in two bands (window overlap across a cut).
    pub est_duplicated: f64,
}

/// Estimated number of precursors with m/z in `[lo, hi]`, assuming a uniform m/z density
/// inside each row group. Exact at row-group boundaries, which is where cuts land on a
/// sorted table with about 1M rows per group; the estimate only steers the balance.
pub fn est_precursors(stats: &[RowGroupStats], lo: f64, hi: f64) -> f64 {
    let mut n = 0.0;
    for s in stats {
        let (Some(min), Some(max)) = (s.min, s.max) else {
            continue;
        };
        if max < lo || min > hi {
            continue;
        }
        let width = max - min;
        let frac = if width <= 0.0 {
            1.0
        } else {
            ((hi.min(max) - lo.max(min)) / width).clamp(0.0, 1.0)
        };
        n += s.rows as f64 * frac;
    }
    n
}

/// Cut `windows` into `n` contiguous groups balanced by estimated precursor count.
///
/// Windows are sorted by lower bound first. A window that no precursor can fall into
/// still belongs to a group (its scans are read by that group's extract and simply find no
/// candidate). Groups that would select no precursor at all are merged into their neighbour,
/// so every band a caller loads is non-empty; `n` is therefore an upper bound.
pub fn plan(windows: &[(f64, f64)], stats: &[RowGroupStats], n: usize) -> Result<Plan> {
    if windows.is_empty() {
        bail!("no isolation windows to group");
    }
    if n == 0 {
        bail!("groups.window_groups must be >= 1");
    }
    if stats.iter().any(|s| s.min.is_none() || s.max.is_none()) {
        bail!(
            "the precursor table has row groups without precursor_mz statistics; a window-group \
             plan needs them (rewrite the table with the library writers)"
        );
    }
    let mut windows: Vec<(f64, f64)> = windows.to_vec();
    windows.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    if windows
        .iter()
        .any(|w| !(w.0.is_finite() && w.1.is_finite()) || w.1 < w.0)
    {
        bail!("an isolation window has a non-finite or inverted m/z range");
    }
    let total_lo = windows[0].0;
    let total_hi = windows.iter().map(|w| w.1).fold(f64::MIN, f64::max);
    let total = est_precursors(stats, total_lo, total_hi);
    let all: f64 = stats.iter().map(|s| s.rows as f64).sum();
    let est_unselectable = (all - total).max(0.0);

    // Per-window estimate, each window taken on its own range. Overlaps count twice here,
    // which only nudges the balance; the plan's duplicate estimate is computed exactly on
    // the final bands below.
    let per_window: Vec<f64> = windows
        .iter()
        .map(|&(lo, hi)| est_precursors(stats, lo, hi))
        .collect();
    let n = n.min(windows.len()).max(1);
    let target = per_window.iter().sum::<f64>() / n as f64;

    // Cut where the running total crosses each k/n share of the whole. A greedy "close the
    // group once it reaches the target" walk looks equivalent and is not: it closes a group
    // on the window that first reaches the target, so every group overshoots a little, the
    // overshoot compounds, and the windows left at the end are swallowed by the last group.
    // Measured on the 8-12-mer library, whose first 14 windows select nothing because the
    // library starts at m/z 326: 63 groups asked for came back as 36, one of them holding
    // 28 windows and 37.9M precursors against a 3.2M target. Crossing points cannot
    // compound, because each is taken against the running total rather than a reset
    // accumulator.
    let cum: Vec<f64> = per_window
        .iter()
        .scan(0.0, |acc, w| {
            *acc += w;
            Some(*acc)
        })
        .collect();
    let mut cuts: Vec<usize> = vec![0];
    for k in 1..n {
        let want = target * k as f64;
        // First window whose running total reaches this share, and never a cut that would
        // leave fewer windows than groups still to place.
        let at = cum.partition_point(|&c| c < want) + 1;
        let at = at
            .min(windows.len() - (n - k))
            .max(*cuts.last().expect("seeded with 0") + 1);
        if at > *cuts.last().expect("seeded with 0") && at < windows.len() {
            cuts.push(at);
        }
    }
    cuts.push(windows.len());
    let groups: Vec<Vec<usize>> = cuts
        .windows(2)
        .filter(|c| c[1] > c[0])
        .map(|c| (c[0]..c[1]).collect())
        .collect();

    // Bands from the groups, then merge any band that selects nothing into a neighbour.
    let mut bands: Vec<Band> = groups
        .into_iter()
        .map(|ws| {
            let mz_lo = windows[ws[0]].0;
            let mz_hi = ws.iter().map(|&i| windows[i].1).fold(f64::MIN, f64::max);
            Band {
                index: 0,
                mz_lo,
                mz_hi,
                windows: ws,
                est_precursors: est_precursors(stats, mz_lo, mz_hi),
            }
        })
        .collect();
    let mut i = 0;
    while i < bands.len() {
        if bands[i].est_precursors > 0.0 || bands.len() == 1 {
            i += 1;
            continue;
        }
        let empty = bands.remove(i);
        // Prefer the band that follows (a low-m/z empty band at the start of the run is
        // the common case), else the one before.
        let j = if i < bands.len() { i } else { i - 1 };
        let b = &mut bands[j];
        b.windows.extend(empty.windows);
        b.windows.sort_unstable();
        b.mz_lo = b.mz_lo.min(empty.mz_lo);
        b.mz_hi = b.mz_hi.max(empty.mz_hi);
        b.est_precursors = est_precursors(stats, b.mz_lo, b.mz_hi);
    }
    for (k, b) in bands.iter_mut().enumerate() {
        b.index = k;
    }
    let mut est_duplicated = 0.0;
    for pair in bands.windows(2) {
        if pair[0].mz_hi > pair[1].mz_lo {
            est_duplicated += est_precursors(stats, pair[1].mz_lo, pair[0].mz_hi);
        }
    }
    Ok(Plan {
        bands,
        windows,
        est_unselectable,
        est_duplicated,
    })
}

/// Write the precursor rows `[first_row, first_row + n)` of `precursors` to `out` with
/// `candidate_id` renumbered to `0..n`: the band's own precursor table, which the DeepLC
/// sidecars rewrite and every stage of the group loads (`Library::load_with_fragment_offset`
/// with `first_row` as the fragment offset). Streams batch by batch. Returns `n`.
pub fn write_band_slice(precursors: &str, first_row: usize, n: usize, out: &str) -> Result<u64> {
    use arrow::array::{Array, UInt32Array};
    use arrow::record_batch::RecordBatch;
    use mumdia_io::table::{BatchWriter, TableFile};
    use std::sync::Arc;

    let t = TableFile::open_rows(precursors, first_row, n)?;
    let reader = t.batches(None, 1 << 16)?;
    let schema = reader.schema();
    let cid_ix = schema
        .index_of("candidate_id")
        .map_err(|_| anyhow::anyhow!("{precursors} has no candidate_id column"))?;
    let offset = u32::try_from(first_row)
        .map_err(|_| anyhow::anyhow!("band offset {first_row} does not fit u32"))?;
    let mut w = BatchWriter::with_row_group_rows(out, schema.clone(), 1 << 17)?;
    let mut rows = 0u64;
    for b in reader {
        let b = b?;
        let cid = b
            .column(cid_ix)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| anyhow::anyhow!("{precursors}: candidate_id is not u32"))?;
        if cid.null_count() > 0 {
            bail!("{precursors}: candidate_id has nulls");
        }
        let mut local = Vec::with_capacity(cid.len());
        for (k, &c) in cid.values().iter().enumerate() {
            let expect = offset + rows as u32 + k as u32;
            if c != expect {
                bail!(
                    "{precursors}: row {} has candidate_id {c}, expected {expect}; the library \
                     must carry candidate_id as the contiguous row-aligned range",
                    first_row as u64 + rows + k as u64
                );
            }
            local.push(rows as u32 + k as u32);
        }
        let mut cols = b.columns().to_vec();
        cols[cid_ix] = Arc::new(UInt32Array::from(local));
        let batch = RecordBatch::try_new(schema.clone(), cols)?;
        rows += batch.num_rows() as u64;
        w.write(&batch)?;
    }
    w.close()?;
    Ok(rows)
}

/// Write the run-level `cal.json` of a grouped run from the bands' own (`groups/gNN/cal.json`).
///
/// Under global calibration every band fitted the same anchors, so the fit is one and the
/// first band's record is the run's, with a `groups` list documenting each band's copy.
/// Under per-group calibration the bands fitted different anchors; the run-level record then
/// sums `n_train`, takes the median `w_rt`, weights the residual summaries by anchors, and
/// reports a `calibration_status` of `mixed` when the bands disagree. Either way the record
/// keeps the keys an ungrouped `cal.json` has, so tooling reads it unchanged, and adds
/// `window_groups`, `groups_calibration` and `groups`.
pub fn summarise_cal(cals: &[(usize, String)], global: bool, out: &str) -> Result<()> {
    if cals.is_empty() {
        bail!("groups: no band calibration to summarise");
    }
    let mut records: Vec<(usize, Value)> = Vec::with_capacity(cals.len());
    for (index, path) in cals {
        let v: Value = mumdia_io::json::read_json(path)
            .with_context(|| format!("reading the band calibration {path}"))?;
        records.push((*index, v));
    }
    let num = |v: &Value, key: &str| v.get(key).and_then(Value::as_f64);
    let groups: Vec<Value> = records
        .iter()
        .map(|(i, v)| {
            json!({
                "group": i,
                "n_train": v.get("n_train").cloned().unwrap_or(Value::Null),
                "w_rt": v.get("w_rt").cloned().unwrap_or(Value::Null),
                "calibration_status": v.get("calibration_status").cloned().unwrap_or(Value::Null),
                "rt_residual_abs_median_s": v.get("rt_residual_abs_median_s").cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    let mut top = if global {
        records[0].1.clone()
    } else {
        let mut t = records[0].1.clone();
        let n: Vec<f64> = records
            .iter()
            .map(|(_, v)| num(v, "n_train").unwrap_or(0.0))
            .collect();
        let n_sum: f64 = n.iter().sum();
        t["n_train"] = json!(n_sum as u64);
        let mut w: Vec<f64> = records
            .iter()
            .filter_map(|(_, v)| num(v, "w_rt"))
            .filter(|w| w.is_finite())
            .collect();
        w.sort_by(|a, b| a.total_cmp(b));
        t["w_rt"] = if w.is_empty() {
            Value::Null
        } else {
            json!(w[w.len() / 2])
        };
        for key in [
            "rt_residual_median_s",
            "rt_residual_abs_median_s",
            "rt_residual_mad_s",
        ] {
            let mut acc = 0.0;
            let mut wsum = 0.0;
            for ((_, v), ni) in records.iter().zip(&n) {
                if let Some(x) = num(v, key) {
                    if x.is_finite() && *ni > 0.0 {
                        acc += x * ni;
                        wsum += ni;
                    }
                }
            }
            t[key] = if wsum > 0.0 {
                json!(acc / wsum)
            } else {
                Value::Null
            };
        }
        let statuses: Vec<&Value> = records
            .iter()
            .filter_map(|(_, v)| v.get("calibration_status"))
            .collect();
        if statuses.windows(2).any(|p| p[0] != p[1]) {
            t["calibration_status"] = json!("mixed");
        }
        t
    };
    top["window_groups"] = json!(records.len());
    top["groups_calibration"] = json!(if global { "global" } else { "per_group" });
    top["groups"] = Value::Array(groups);
    mumdia_io::json::write_json(out, &top)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_stats(lo: f64, hi: f64, groups: usize, rows_per_group: usize) -> Vec<RowGroupStats> {
        let w = (hi - lo) / groups as f64;
        (0..groups)
            .map(|i| RowGroupStats {
                rows: rows_per_group,
                min: Some(lo + i as f64 * w),
                max: Some(lo + (i + 1) as f64 * w),
            })
            .collect()
    }

    fn contiguous_windows(lo: f64, width: f64, n: usize) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| (lo + i as f64 * width, lo + (i + 1) as f64 * width))
            .collect()
    }

    #[test]
    fn uniform_library_splits_windows_evenly() {
        let windows = contiguous_windows(400.0, 10.0, 10); // 400..500
        let stats = uniform_stats(400.0, 500.0, 20, 1000); // 20k precursors, uniform
        let plan = plan(&windows, &stats, 2).unwrap();
        assert_eq!(plan.bands.len(), 2);
        assert_eq!(plan.bands[0].windows, vec![0, 1, 2, 3, 4]);
        assert_eq!(plan.bands[1].windows, vec![5, 6, 7, 8, 9]);
        assert_eq!((plan.bands[0].mz_lo, plan.bands[0].mz_hi), (400.0, 450.0));
        assert_eq!((plan.bands[1].mz_lo, plan.bands[1].mz_hi), (450.0, 500.0));
        assert!((plan.bands[0].est_precursors - 10_000.0).abs() < 1.0);
        assert_eq!(plan.est_unselectable, 0.0);
        assert_eq!(plan.est_duplicated, 0.0);
        // One group is the whole run.
        let one = plan_one(&windows, &stats);
        assert_eq!(one.bands.len(), 1);
        assert_eq!(one.bands[0].windows.len(), 10);
    }

    fn plan_one(windows: &[(f64, f64)], stats: &[RowGroupStats]) -> Plan {
        plan(windows, stats, 1).unwrap()
    }

    #[test]
    fn skewed_library_moves_the_cut_and_counts_unselectable_precursors() {
        let windows = contiguous_windows(400.0, 10.0, 10);
        // 3x the density in the upper half, plus 5,000 precursors above every window.
        let mut stats = uniform_stats(400.0, 450.0, 5, 1000);
        stats.extend(uniform_stats(450.0, 500.0, 5, 3000));
        stats.push(RowGroupStats {
            rows: 5000,
            min: Some(900.0),
            max: Some(1200.0),
        });
        let plan = plan(&windows, &stats, 2).unwrap();
        // 20,000 selectable precursors, target 10,000: the lower half (5,000) plus the
        // first two upper windows (6,000) makes 11,000, so the cut falls after window 6.
        assert_eq!(plan.bands[0].windows, vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(plan.bands[1].windows, vec![7, 8, 9]);
        assert!((plan.est_unselectable - 5000.0).abs() < 1e-9);
    }

    #[test]
    fn overlapping_windows_give_overlapping_bands_and_a_duplicate_estimate() {
        // 1 m/z overlaps: [400,411], [410,421], ..., 10 windows.
        let windows: Vec<(f64, f64)> = (0..10)
            .map(|i| (400.0 + i as f64 * 10.0, 411.0 + i as f64 * 10.0))
            .collect();
        let stats = uniform_stats(400.0, 501.0, 10, 1000);
        let plan = plan(&windows, &stats, 2).unwrap();
        assert_eq!(plan.bands.len(), 2);
        let (a, b) = (&plan.bands[0], &plan.bands[1]);
        assert!(
            a.mz_hi > b.mz_lo,
            "adjacent bands overlap by the window overlap"
        );
        assert!((a.mz_hi - b.mz_lo - 1.0).abs() < 1e-9);
        // About 1 m/z of 101 m/z at ~99 precursors per m/z.
        assert!(plan.est_duplicated > 90.0 && plan.est_duplicated < 110.0);
    }

    #[test]
    fn empty_bands_are_merged_into_a_neighbour() {
        // The library starts at m/z 440: the first four windows select nothing.
        let windows = contiguous_windows(400.0, 10.0, 10);
        let stats = uniform_stats(440.0, 500.0, 6, 1000);
        let plan = plan(&windows, &stats, 4).unwrap();
        assert!(plan.bands.iter().all(|b| b.est_precursors > 0.0));
        assert!(plan.bands.len() <= 4);
        let covered: Vec<usize> = plan.bands.iter().flat_map(|b| b.windows.clone()).collect();
        assert_eq!(covered, (0..10).collect::<Vec<_>>());
        assert_eq!(plan.bands[0].mz_lo, 400.0);
    }

    #[test]
    fn more_groups_than_windows_is_clamped_and_bad_input_is_refused() {
        let windows = contiguous_windows(400.0, 10.0, 3);
        let stats = uniform_stats(400.0, 430.0, 3, 100);
        assert_eq!(plan(&windows, &stats, 10).unwrap().bands.len(), 3);
        assert!(plan(&[], &stats, 2).is_err());
        assert!(plan(&windows, &stats, 0).is_err());
        assert!(plan(&[(410.0, 400.0)], &stats, 1).is_err());
        let no_stats = vec![RowGroupStats {
            rows: 10,
            min: None,
            max: None,
        }];
        assert!(plan(&windows, &no_stats, 2).is_err());
    }

    /// The cut must not compound: a leading run of windows that select nothing, and then
    /// windows of equal weight, has to come back as the number of bands asked for, evenly
    /// filled. The greedy walk this replaced returned 36 bands for 63 here, one of them
    /// holding a third of the library.
    #[test]
    fn empty_leading_windows_do_not_swallow_the_plan() {
        // 14 windows below the library's first precursor, then 100 windows over it.
        let mut windows: Vec<(f64, f64)> = (0..14)
            .map(|i| (100.0 + i as f64, 101.0 + i as f64))
            .collect();
        windows.extend((0..100).map(|i| (400.0 + i as f64, 401.0 + i as f64)));
        let stats: Vec<RowGroupStats> = (0..100)
            .map(|i| RowGroupStats {
                rows: 1_000_000,
                min: Some(400.0 + i as f64),
                max: Some(401.0 + i as f64),
            })
            .collect();
        let p = plan(&windows, &stats, 25).expect("plans");
        assert_eq!(p.bands.len(), 25, "every band asked for");
        let est: Vec<f64> = p.bands.iter().map(|b| b.est_precursors).collect();
        let (lo, hi) = (
            est.iter().cloned().fold(f64::MAX, f64::min),
            est.iter().cloned().fold(0.0, f64::max),
        );
        assert!(
            hi <= 3.0 * lo.max(1.0),
            "bands within 3x of each other, got {lo} to {hi}"
        );
        assert_eq!(
            p.bands.iter().map(|b| b.windows.len()).sum::<usize>(),
            windows.len(),
            "every window placed exactly once"
        );
    }

    #[test]
    fn run_level_cal_keeps_the_ungrouped_keys_and_summarises_the_bands() {
        let dir = std::env::temp_dir().join(format!("mumdia_groups_cal_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut cals = Vec::new();
        for (i, (n, w, status, resid)) in [
            (100u64, 20.0, "loess", 2.0),
            (300u64, 40.0, "loess", 4.0),
            (0u64, f64::NAN, "unbounded", f64::NAN),
        ]
        .iter()
        .enumerate()
        {
            let p = dir
                .join(format!("cal{i}.json"))
                .to_str()
                .unwrap()
                .to_string();
            let w_val = if w.is_finite() { json!(w) } else { Value::Null };
            let r_val = if resid.is_finite() {
                json!(resid)
            } else {
                Value::Null
            };
            mumdia_io::json::write_json(
                &p,
                &json!({"method": "loess", "n_train": n, "w_rt": w_val,
                        "calibration_status": status, "rt_residual_abs_median_s": r_val,
                        "w_rt_sizing": "in_sample"}),
            )
            .unwrap();
            cals.push((i, p));
        }
        let out = dir.join("cal_pg.json").to_str().unwrap().to_string();
        summarise_cal(&cals, false, &out).unwrap();
        let v: Value = mumdia_io::json::read_json(&out).unwrap();
        assert_eq!(v["n_train"], json!(400));
        assert_eq!(v["w_rt"], json!(40.0), "median of the finite w_rt");
        assert_eq!(v["calibration_status"], json!("mixed"));
        assert!((v["rt_residual_abs_median_s"].as_f64().unwrap() - 3.5).abs() < 1e-9);
        assert_eq!(v["groups_calibration"], json!("per_group"));
        assert_eq!(v["window_groups"], json!(3));
        assert_eq!(v["groups"].as_array().unwrap().len(), 3);
        assert_eq!(v["w_rt_sizing"], json!("in_sample"));

        let out_g = dir.join("cal_g.json").to_str().unwrap().to_string();
        summarise_cal(&cals[..2], true, &out_g).unwrap();
        let v: Value = mumdia_io::json::read_json(&out_g).unwrap();
        assert_eq!(
            v["n_train"],
            json!(100),
            "global: the first band's record is the fit"
        );
        assert_eq!(v["groups_calibration"], json!("global"));
        assert_eq!(v["groups"].as_array().unwrap().len(), 2);
    }
}
