//! Match-between-runs by a second pass (`mbr.strategy = second_pass`, the default of
//! `run-experiment`): the empirical library the second pass searches.
//!
//! The pooled first pass accepts precursors experiment-wide. Each one that it accepts
//! (`precursor_q <= mbr.q_anchor`, target) enters the empirical library with the library decoy
//! paired to it (by rank within `base_peptide_id` and `charge`, `plan_rows`), and both get
//! the precursor's cross-run consensus retention time:
//!
//! 1. every run's confident apexes (`run_psm_q <= q`, target) are mapped onto one reference
//!    run, the run with the most of them, by a monotone binned-median curve over the
//!    precursors the two runs share (`ALIGN_BINS` bins, at least `MIN_SHARED` shared; a run
//!    with fewer keeps its own scale and is counted as unaligned);
//! 2. the consensus is the median of a precursor's mapped apexes; a precursor accepted
//!    experiment-wide but in no single run takes the apex of its winning row.
//!
//! The fragments are the library's own, for the target and the decoy alike, so the only
//! thing the pair does not share with the first pass is its retention time, and that it
//! shares with each other. Every run is then searched again on this library by the ordinary
//! chain (seed, retention-time calibration, extract, features, compete) and rescored in one
//! pool: the second pass controls its FDR with its own decoys, which is why it can report a
//! precursor in a run where the first pass did not, and why it is measured against nulls rather
//! than assumed (docs/12, "Second pass").

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, StringArray, UInt32Array};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use mumdia_io::table::{write_table, Col, TableFile};
use serde::Serialize;
use tracing::{info, warn};

use crate::stages::sub_library;

/// Fewest precursors a run must share with the reference run for its apexes to be aligned
/// onto the reference; below it the run keeps its own retention-time scale.
pub const MIN_SHARED: usize = 200;
/// Bins of the binned-median alignment curve.
pub const ALIGN_BINS: usize = 80;
const BATCH_ROWS: usize = 1 << 16;

pub struct BuildParams<'a> {
    /// The first pass's pooled scored table (`candidate_id`, `source`, `label`, `run_psm_q`,
    /// `precursor_q`, `apex_rt`).
    pub scored: &'a str,
    /// The library the first pass searched; its ids are the scored table's.
    pub lib_precursors: &'a str,
    pub lib_fragments: &'a str,
    pub out_dir: &'a str,
    /// `mbr.q_anchor`.
    pub q: f64,
}

/// What [`build_library`] wrote, also recorded as `empirical_library.json` in its directory.
#[derive(Clone, Debug, Serialize)]
pub struct EmpiricalLibrary {
    pub precursors: String,
    pub fragments: String,
    /// Target precursors accepted experiment-wide in the first pass.
    pub entries: u64,
    pub targets: u64,
    pub decoys: u64,
    pub fragment_rows: u64,
    /// The `source` every run's apexes were mapped onto.
    pub reference_source: Option<u32>,
    /// Runs with at least one confident apex, aligned onto the reference (the reference
    /// included).
    pub runs_aligned: usize,
    /// Runs with confident apexes but fewer than `MIN_SHARED` shared with the reference.
    pub runs_unaligned: usize,
    /// Entries accepted in no single run, which take their winning row's apex.
    pub entries_from_winner: u64,
    /// Entries the library has no paired decoy for; they are searched without one.
    pub entries_without_decoy: u64,
    /// Median number of runs behind an entry's consensus.
    pub median_observations: f64,
    pub q: f64,
}

/// A monotone binned-median curve mapping one run's retention times onto the reference's.
#[derive(Clone, Debug)]
pub(crate) struct Curve {
    cx: Vec<f64>,
    cy: Vec<f64>,
}

fn median_sorted(v: &[f64]) -> f64 {
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

impl Curve {
    /// Fit on `(x, y)` pairs: sorted by x, cut into `bins` equal-count bins, the median x and
    /// the median y of each bin, then y made non-decreasing.
    pub(crate) fn fit(mut pairs: Vec<(f64, f64)>, bins: usize) -> Curve {
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let n = pairs.len();
        let (mut cx, mut cy) = (Vec::with_capacity(bins), Vec::with_capacity(bins));
        for b in 0..bins {
            let (lo, hi) = (b * n / bins, (b + 1) * n / bins);
            if hi <= lo {
                continue;
            }
            let xs: Vec<f64> = pairs[lo..hi].iter().map(|p| p.0).collect();
            let mut ys: Vec<f64> = pairs[lo..hi].iter().map(|p| p.1).collect();
            ys.sort_by(f64::total_cmp);
            cx.push(median_sorted(&xs));
            cy.push(median_sorted(&ys));
        }
        for i in 1..cy.len() {
            if cy[i] < cy[i - 1] {
                cy[i] = cy[i - 1];
            }
        }
        Curve { cx, cy }
    }

    /// Linear interpolation between the bin medians, and linear extrapolation past the outer
    /// ones along the outer two.
    pub(crate) fn map(&self, x: f64) -> f64 {
        let (cx, cy) = (&self.cx, &self.cy);
        let n = cx.len();
        match n {
            0 => return x,
            1 => return cy[0] + (x - cx[0]),
            _ => {}
        }
        let slope = |a: usize, b: usize| (cy[b] - cy[a]) / (cx[b] - cx[a]).max(1e-9);
        if x <= cx[0] {
            return cy[0] + (x - cx[0]) * slope(0, 1);
        }
        if x >= cx[n - 1] {
            return cy[n - 1] + (x - cx[n - 1]) * slope(n - 2, n - 1);
        }
        let k = cx.partition_point(|&c| c <= x);
        let t = (x - cx[k - 1]) / (cx[k] - cx[k - 1]);
        cy[k - 1] + t * (cy[k] - cy[k - 1])
    }
}

/// The consensus retention time of every candidate in `obs` (`(source, candidate_id, apex)`,
/// one per pair), on the reference run's scale, with the number of runs behind it.
pub(crate) struct Consensus {
    pub rt: BTreeMap<u32, (f32, u32)>,
    pub reference: Option<u32>,
    pub aligned: usize,
    pub unaligned: usize,
}

pub(crate) fn consensus(obs: &[(u32, u32, f64)]) -> Consensus {
    let mut by_run: BTreeMap<u32, Vec<(u32, f64)>> = BTreeMap::new();
    for &(s, c, rt) in obs {
        by_run.entry(s).or_default().push((c, rt));
    }
    // The run with the most confident apexes, the lowest source on a tie.
    let mut reference: Option<(u32, usize)> = None;
    for (&s, v) in &by_run {
        if reference.is_none_or(|(_, n)| v.len() > n) {
            reference = Some((s, v.len()));
        }
    }
    let Some((ref_s, _)) = reference else {
        return Consensus {
            rt: BTreeMap::new(),
            reference: None,
            aligned: 0,
            unaligned: 0,
        };
    };
    let ref_rt: HashMap<u32, f64> = by_run[&ref_s].iter().copied().collect();
    let mut mapped: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    let (mut aligned, mut unaligned) = (0usize, 0usize);
    for (&s, v) in &by_run {
        let curve = if s == ref_s {
            None
        } else {
            let shared: Vec<(f64, f64)> = v
                .iter()
                .filter_map(|&(c, rt)| ref_rt.get(&c).map(|&r| (rt, r)))
                .collect();
            if shared.len() >= MIN_SHARED {
                Some(Curve::fit(shared, ALIGN_BINS))
            } else {
                unaligned += 1;
                None
            }
        };
        if s == ref_s || curve.is_some() {
            aligned += 1;
        }
        for &(c, rt) in v {
            let m = curve.as_ref().map_or(rt, |k| k.map(rt));
            mapped.entry(c).or_default().push(m);
        }
    }
    let rt = mapped
        .into_iter()
        .map(|(c, mut v)| {
            v.sort_by(f64::total_cmp);
            (c, (median_sorted(&v) as f32, v.len() as u32))
        })
        .collect();
    Consensus {
        rt,
        reference: Some(ref_s),
        aligned,
        unaligned,
    }
}

fn f64_col(b: &RecordBatch, name: &str, path: &str) -> Result<Float64Array> {
    let ix = b
        .schema()
        .index_of(name)
        .map_err(|_| anyhow!("{path} has no {name} column"))?;
    let a: ArrayRef = cast(b.column(ix), &DataType::Float64)?;
    Ok(a.as_any()
        .downcast_ref::<Float64Array>()
        .expect("cast to f64")
        .clone())
}

fn u32_col(b: &RecordBatch, name: &str, path: &str) -> Result<UInt32Array> {
    let ix = b
        .schema()
        .index_of(name)
        .map_err(|_| anyhow!("{path} has no {name} column"))?;
    b.column(ix)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .cloned()
        .ok_or_else(|| anyhow!("{path}: {name} is not u32"))
}

fn i64_col(b: &RecordBatch, name: &str, path: &str) -> Result<Int64Array> {
    let ix = b
        .schema()
        .index_of(name)
        .map_err(|_| anyhow!("{path} has no {name} column"))?;
    let a: ArrayRef = cast(b.column(ix), &DataType::Int64)?;
    Ok(a.as_any()
        .downcast_ref::<Int64Array>()
        .expect("cast to i64")
        .clone())
}

/// The first pass's entries and observations, from one streaming pass over its scored table:
/// the entries (sorted), every `(source, candidate)` apex at `run_psm_q <= q` (the row with the
/// lowest `run_psm_q` where a pair has several), and each entry's winning row.
struct Scan {
    entries: Vec<u32>,
    obs: BTreeMap<(u32, u32), (f64, f64)>,
    winner: BTreeMap<u32, (f64, f64)>,
}

fn scan_scored(path: &str, q: f64) -> Result<Scan> {
    let t = TableFile::open(path)?;
    let cols = [
        "candidate_id",
        "source",
        "label",
        "run_psm_q",
        "precursor_q",
        "apex_rt",
    ];
    let mut entries: Vec<u32> = Vec::new();
    let mut obs: BTreeMap<(u32, u32), (f64, f64)> = BTreeMap::new();
    let mut winner: BTreeMap<u32, (f64, f64)> = BTreeMap::new();
    for b in t.batches(Some(&cols), BATCH_ROWS)? {
        let b = b?;
        let cid = u32_col(&b, "candidate_id", path)?;
        let src = u32_col(&b, "source", path)?;
        let label_ix = b.schema().index_of("label")?;
        let label: ArrayRef = cast(b.column(label_ix), &DataType::Utf8)?;
        let label = label
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cast to utf8");
        let run_q = f64_col(&b, "run_psm_q", path)?;
        let prec_q = f64_col(&b, "precursor_q", path)?;
        let apex = f64_col(&b, "apex_rt", path)?;
        for i in 0..b.num_rows() {
            if !label.is_valid(i) || label.value(i) != "target" || !apex.is_valid(i) {
                continue;
            }
            let rt = apex.value(i);
            if !rt.is_finite() {
                continue;
            }
            let c = cid.value(i);
            if prec_q.is_valid(i) && prec_q.value(i) <= q {
                entries.push(c);
                let pq = prec_q.value(i);
                if winner.get(&c).is_none_or(|&(w, _)| pq < w) {
                    winner.insert(c, (pq, rt));
                }
            }
            if run_q.is_valid(i) && run_q.value(i) <= q {
                let rq = run_q.value(i);
                let key = (src.value(i), c);
                if obs.get(&key).is_none_or(|&(o, _)| rq < o) {
                    obs.insert(key, (rq, rt));
                }
            }
        }
    }
    entries.sort_unstable();
    entries.dedup();
    Ok(Scan {
        entries,
        obs,
        winner,
    })
}

/// The library rows the second pass keeps and their retention times: every entry and the
/// decoy paired with it, both given the entry's consensus, and the number of entries that have
/// no decoy.
///
/// The pairing is by rank within `(base_peptide_id, charge)`: the i-th target and the i-th
/// decoy, each ordered by `(peptidoform_id, candidate_id)`. The native digest gives a decoy
/// its own `peptidoform_id` and shares the `base_peptide_id`, with the modforms of the target
/// and of the decoy enumerated in the same order; the imported-library decoy builders copy the
/// target row, sharing both ids, and the rank then pairs the rows that share a
/// `peptidoform_id`. Measured on four imported libraries (2M targets each): every target
/// paired, every pair the shared-id pair, and every pair at the same precursor m/z.
fn plan_rows(lib_precursors: &str, irt: &BTreeMap<u32, f32>) -> Result<(HashMap<u32, f32>, u64)> {
    let t = TableFile::open(lib_precursors)?;
    let cols = ["candidate_id", "base_peptide_id", "charge"];
    let mut groups: HashMap<(u32, i64), ()> = HashMap::with_capacity(irt.len());
    for b in t.batches(Some(&cols), BATCH_ROWS)? {
        let b = b?;
        let cid = u32_col(&b, "candidate_id", lib_precursors)?;
        let base = u32_col(&b, "base_peptide_id", lib_precursors)?;
        let ch = i64_col(&b, "charge", lib_precursors)?;
        for i in 0..b.num_rows() {
            if irt.contains_key(&cid.value(i)) {
                groups.insert((base.value(i), ch.value(i)), ());
            }
        }
    }
    // (peptidoform_id, candidate_id) of the targets and of the decoys of every such group.
    type Members = (Vec<(u32, u32)>, Vec<(u32, u32)>);
    let mut members: BTreeMap<(u32, i64), Members> = BTreeMap::new();
    let cols = [
        "candidate_id",
        "peptidoform_id",
        "base_peptide_id",
        "charge",
        "label",
    ];
    for b in t.batches(Some(&cols), BATCH_ROWS)? {
        let b = b?;
        let cid = u32_col(&b, "candidate_id", lib_precursors)?;
        let pf = u32_col(&b, "peptidoform_id", lib_precursors)?;
        let base = u32_col(&b, "base_peptide_id", lib_precursors)?;
        let ch = i64_col(&b, "charge", lib_precursors)?;
        let label: ArrayRef = cast(b.column(b.schema().index_of("label")?), &DataType::Utf8)?;
        let label = label
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cast to utf8");
        for i in 0..b.num_rows() {
            let key = (base.value(i), ch.value(i));
            if !groups.contains_key(&key) {
                continue;
            }
            let m = members.entry(key).or_default();
            let row = (pf.value(i), cid.value(i));
            if label.is_valid(i) && label.value(i) == "decoy" {
                m.1.push(row);
            } else {
                m.0.push(row);
            }
        }
    }
    let mut keep: HashMap<u32, f32> = HashMap::with_capacity(2 * irt.len());
    let mut unpaired = 0u64;
    for (_, (mut targets, mut decoys)) in members {
        targets.sort_unstable();
        decoys.sort_unstable();
        for (rank, &(_, c)) in targets.iter().enumerate() {
            let Some(&v) = irt.get(&c) else { continue };
            keep.insert(c, v);
            match decoys.get(rank) {
                Some(&(_, d)) => {
                    keep.insert(d, v);
                }
                None => unpaired += 1,
            }
        }
    }
    Ok((keep, unpaired))
}

/// Build the empirical library of the second pass into `p.out_dir`
/// (`lib_precursors.parquet`, `lib_fragments.parquet`, `empirical_library.json`). With no
/// entry it writes only the summary and returns `targets == 0`; the caller then keeps the
/// first pass.
pub fn build_library(p: &BuildParams) -> Result<EmpiricalLibrary> {
    let t0 = Instant::now();
    std::fs::create_dir_all(p.out_dir).with_context(|| format!("creating {}", p.out_dir))?;
    let d = |name: &str| format!("{}/{name}", p.out_dir);
    let scan = scan_scored(p.scored, p.q)?;
    let is_entry = |c: u32| scan.entries.binary_search(&c).is_ok();
    let mut obs: Vec<(u32, u32, f64)> = scan
        .obs
        .iter()
        .filter(|((_, c), _)| is_entry(*c))
        .map(|(&(s, c), &(_, rt))| (s, c, rt))
        .collect();
    let cons = consensus(&obs);
    let mut irt: BTreeMap<u32, f32> = cons.rt.iter().map(|(&c, &(v, _))| (c, v)).collect();
    // Entries accepted in no single run take the raw apex of their winning row: on the
    // reference scale when that row is the reference run's, within the alignment's shift of it
    // otherwise. The second pass calibrates every run against this scale anyway.
    let mut from_winner = 0u64;
    for &c in &scan.entries {
        if let (std::collections::btree_map::Entry::Vacant(slot), Some(&(_, rt))) =
            (irt.entry(c), scan.winner.get(&c))
        {
            slot.insert(rt as f32);
            from_winner += 1;
        }
    }
    let mut n_obs: Vec<f64> = cons.rt.values().map(|&(_, n)| f64::from(n)).collect();
    n_obs.sort_by(f64::total_cmp);
    let median_observations = if n_obs.is_empty() {
        0.0
    } else {
        median_sorted(&n_obs)
    };
    obs.clear();

    let mut lib = EmpiricalLibrary {
        precursors: d("lib_precursors.parquet"),
        fragments: d("lib_fragments.parquet"),
        entries: scan.entries.len() as u64,
        targets: 0,
        decoys: 0,
        fragment_rows: 0,
        reference_source: cons.reference,
        runs_aligned: cons.aligned,
        runs_unaligned: cons.unaligned,
        entries_from_winner: from_winner,
        entries_without_decoy: 0,
        median_observations,
        q: p.q,
    };
    if irt.is_empty() {
        warn!(
            q = p.q,
            "second pass: the first pass accepted no target precursor; there is nothing to \
             search again"
        );
        mumdia_io::json::write_json(&d("empirical_library.json"), &lib)?;
        return Ok(lib);
    }
    let (keep, unpaired) = plan_rows(p.lib_precursors, &irt)?;
    lib.entries_without_decoy = unpaired;
    let mut survivors: Vec<u32> = keep.keys().copied().collect();
    survivors.sort_unstable();
    let surv = d("entries.parquet");
    write_table(&surv, vec![Col::U32("candidate_id".into(), survivors)])?;
    let st = sub_library::run(sub_library::SubLibraryParams {
        precursors: p.lib_precursors,
        fragments: p.lib_fragments,
        survivors: &surv,
        out_precursors: &lib.precursors,
        out_fragments: &lib.fragments,
        // Pair-complete already: every entry and its paired decoy.
        pair_linked: false,
        predicted_irt: Some(&keep),
    })?;
    lib.targets = st.targets;
    lib.decoys = st.decoys;
    lib.fragment_rows = st.fragments;
    if unpaired > 0 {
        warn!(
            targets = st.targets,
            decoys = st.decoys,
            entries_without_decoy = unpaired,
            "second pass: some entries have no paired decoy in the library (fewer decoys than \
             targets in their base peptide and charge); they are searched without one"
        );
    }
    if st.targets != lib.entries {
        bail!(
            "second pass: {} entries but {} targets in the empirical library; an entry's \
             candidate_id did not resolve to a library row of the same id",
            lib.entries,
            st.targets
        );
    }
    mumdia_io::json::write_json(&d("empirical_library.json"), &lib)?;
    info!(
        entries = lib.entries,
        targets = lib.targets,
        decoys = lib.decoys,
        reference_source = ?lib.reference_source,
        runs_aligned = lib.runs_aligned,
        runs_unaligned = lib.runs_unaligned,
        entries_from_winner = lib.entries_from_winner,
        median_observations = lib.median_observations,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "second pass: empirical library"
    );
    Ok(lib)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> String {
        let d =
            std::env::temp_dir().join(format!("mumdia_second_pass_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.to_str().unwrap().to_string()
    }

    #[test]
    fn the_curve_maps_a_shifted_run_onto_the_reference_and_extrapolates() {
        let pairs: Vec<(f64, f64)> = (0..1000).map(|i| (i as f64, i as f64 + 30.0)).collect();
        let k = Curve::fit(pairs, ALIGN_BINS);
        for x in [0.0, 1.5, 500.0, 998.0] {
            assert!((k.map(x) - (x + 30.0)).abs() < 1e-6, "{x}");
        }
        assert!(
            (k.map(-100.0) - (-70.0)).abs() < 1e-6,
            "below the first bin"
        );
        assert!((k.map(1500.0) - 1530.0).abs() < 1e-6, "past the last bin");
    }

    #[test]
    fn the_curve_is_monotone_on_noisy_pairs() {
        // A spike in one bin cannot make a later bin map lower.
        let mut pairs: Vec<(f64, f64)> = (0..800).map(|i| (i as f64, i as f64)).collect();
        for p in pairs.iter_mut().skip(100).take(10) {
            p.1 = 5000.0;
        }
        let k = Curve::fit(pairs, 80);
        let ys: Vec<f64> = (0..800).map(|x| k.map(x as f64)).collect();
        assert!(ys.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn consensus_aligns_runs_that_share_enough_and_keeps_the_others_unaligned() {
        let mut obs: Vec<(u32, u32, f64)> = Vec::new();
        // Run 0 (the reference, most apexes): 400 candidates at rt = 10 * c.
        for c in 0..400u32 {
            obs.push((0, c, 10.0 * c as f64));
        }
        // Run 1: 300 of them 20 s late, and candidate 1000 that only run 1 saw.
        for c in 0..300u32 {
            obs.push((1, c, 10.0 * c as f64 + 20.0));
        }
        obs.push((1, 1000, 520.0));
        // Run 2: 50 shared, 40 s late: too few to align, so it keeps its own scale.
        for c in 0..50u32 {
            obs.push((2, c, 10.0 * c as f64 + 40.0));
        }
        let cons = consensus(&obs);
        assert_eq!(cons.reference, Some(0));
        assert_eq!((cons.aligned, cons.unaligned), (2, 1));
        let (rt, n) = cons.rt[&1000];
        assert!(
            (rt - 500.0).abs() < 1e-3,
            "run 1's 520 s is the reference's 500 s: {rt}"
        );
        assert_eq!(n, 1);
        // Candidate 10: 100 (run 0), 120 -> 100 (run 1), 140 unaligned (run 2): median 100.
        let (rt, n) = cons.rt[&10];
        assert!((rt - 100.0).abs() < 1e-3, "{rt}");
        assert_eq!(n, 3);
    }

    /// A library of `n` peptidoforms, each a target (even id) and its decoy (odd id) sharing
    /// `peptidoform_id`, plus a second charge state of peptidoform 0 at the end.
    fn library(dir: &str, n: u32) -> (String, String) {
        let p = format!("{dir}/prec.parquet");
        let f = format!("{dir}/frag.parquet");
        let rows = 2 * n + 2;
        let mut pf: Vec<u32> = (0..2 * n).map(|i| i / 2).collect();
        pf.extend([0, 0]);
        let mut charge = vec![2i32; 2 * n as usize];
        charge.extend([3, 3]);
        let label: Vec<String> = (0..rows)
            .map(|i| if i % 2 == 0 { "target" } else { "decoy" }.to_string())
            .collect();
        write_table(
            &p,
            vec![
                Col::U32("candidate_id".into(), (0..rows).collect()),
                Col::U32("peptidoform_id".into(), pf.clone()),
                Col::U32("base_peptide_id".into(), pf.clone()),
                Col::Str(
                    "peptidoform".into(),
                    pf.iter().map(|x| format!("P{x}")).collect(),
                ),
                Col::I32("charge".into(), charge),
                Col::F64(
                    "precursor_mz".into(),
                    (0..rows).map(|i| 400.0 + i as f64).collect(),
                ),
                Col::F32("predicted_irt".into(), vec![-1.0; rows as usize]),
                Col::Str("label".into(), label),
                Col::Str("protein".into(), vec!["X".to_string(); rows as usize]),
            ],
        )
        .unwrap();
        write_table(
            &f,
            vec![
                Col::U32(
                    "candidate_id".into(),
                    (0..rows).flat_map(|c| [c, c]).collect(),
                ),
                Col::F32(
                    "frag_mz".into(),
                    (0..2 * rows).map(|i| 100.0 + i as f32).collect(),
                ),
            ],
        )
        .unwrap();
        (p, f)
    }

    #[test]
    fn the_empirical_library_holds_each_entry_and_its_decoy_at_the_consensus_rt() {
        let dir = tmp("build");
        let (lp, lf) = library(&dir, 4);
        // Scored rows over two runs. Targets 0 and 2 are accepted experiment-wide (2 in run 1
        // only, by its winner row); target 4 is confident in run 0 but not accepted
        // experiment-wide; decoy 1 would pass every cut and must not enter.
        let scored = format!("{dir}/scored.parquet");
        let rows: Vec<(u32, u32, &str, f64, f64, f64)> = vec![
            // (candidate, source, label, run_psm_q, precursor_q, apex_rt)
            (0, 0, "target", 0.001, 0.001, 100.0),
            (0, 1, "target", 0.002, 1.0, 104.0),
            (2, 1, "target", 0.5, 0.004, 300.0),
            (4, 0, "target", 0.001, 0.2, 500.0),
            (1, 0, "decoy", 0.001, 0.001, 700.0),
        ];
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), rows.iter().map(|r| r.0).collect()),
                Col::U32("source".into(), rows.iter().map(|r| r.1).collect()),
                Col::Str(
                    "label".into(),
                    rows.iter().map(|r| r.2.to_string()).collect(),
                ),
                Col::F64("run_psm_q".into(), rows.iter().map(|r| r.3).collect()),
                Col::F64("precursor_q".into(), rows.iter().map(|r| r.4).collect()),
                Col::F64("apex_rt".into(), rows.iter().map(|r| r.5).collect()),
            ],
        )
        .unwrap();
        let out = format!("{dir}/emp");
        let lib = build_library(&BuildParams {
            scored: &scored,
            lib_precursors: &lp,
            lib_fragments: &lf,
            out_dir: &out,
            q: 0.01,
        })
        .unwrap();
        assert_eq!(lib.entries, 2);
        assert_eq!(
            (lib.targets, lib.decoys),
            (2, 2),
            "each entry with its own decoy"
        );
        assert_eq!(
            lib.entries_from_winner, 1,
            "target 2 was confident in no run"
        );
        let t = TableFile::open(&lib.precursors).unwrap();
        assert_eq!(
            t.u32("candidate_id").unwrap(),
            vec![0, 1, 2, 3],
            "contiguous ids"
        );
        assert_eq!(
            t.f64("precursor_mz").unwrap(),
            vec![400.0, 401.0, 402.0, 403.0],
            "library order kept; the charge-3 rows of peptidoform 0 are not entries"
        );
        // Target 0: run 0 is the reference (one apex each, the lower source wins the tie) and
        // run 1 shares too little to align, so its 104 s stays as it is: median 102 s.
        let irt = t.f32("predicted_irt").unwrap();
        assert_eq!(
            irt,
            vec![102.0, 102.0, 300.0, 300.0],
            "decoys take their target's RT"
        );
        assert_eq!(
            TableFile::open(&lib.fragments)
                .unwrap()
                .u32("candidate_id")
                .unwrap(),
            vec![0, 0, 1, 1, 2, 2, 3, 3]
        );
        assert!(std::path::Path::new(&format!("{out}/empirical_library.json")).exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_native_decoy_with_its_own_peptidoform_id_is_paired_by_rank_in_its_base_peptide() {
        // One base peptide at charge 2 with two modforms: targets at peptidoform ids 10 and
        // 11, their decoys at 20 and 21 (the native digest numbers decoys separately). The
        // entry is the second target, so its decoy is the second decoy.
        let dir = tmp("native");
        let p = format!("{dir}/prec.parquet");
        write_table(
            &p,
            vec![
                Col::U32("candidate_id".into(), vec![0, 1, 2, 3]),
                Col::U32("peptidoform_id".into(), vec![10, 20, 11, 21]),
                Col::U32("base_peptide_id".into(), vec![7; 4]),
                Col::I32("charge".into(), vec![2; 4]),
                Col::Str(
                    "label".into(),
                    ["target", "decoy", "target", "decoy"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                ),
            ],
        )
        .unwrap();
        let irt: BTreeMap<u32, f32> = [(2, 42.0)].into_iter().collect();
        let (keep, unpaired) = plan_rows(&p, &irt).unwrap();
        let mut kept: Vec<(u32, f32)> = keep.into_iter().collect();
        kept.sort_by_key(|k| k.0);
        assert_eq!(kept, vec![(2, 42.0), (3, 42.0)]);
        assert_eq!(unpaired, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_first_pass_without_entries_writes_only_the_summary() {
        let dir = tmp("empty");
        let (lp, lf) = library(&dir, 2);
        let scored = format!("{dir}/scored.parquet");
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![0]),
                Col::U32("source".into(), vec![0]),
                Col::Str("label".into(), vec!["target".into()]),
                Col::F64("run_psm_q".into(), vec![0.5]),
                Col::F64("precursor_q".into(), vec![0.5]),
                Col::F64("apex_rt".into(), vec![10.0]),
            ],
        )
        .unwrap();
        let out = format!("{dir}/emp");
        let lib = build_library(&BuildParams {
            scored: &scored,
            lib_precursors: &lp,
            lib_fragments: &lf,
            out_dir: &out,
            q: 0.01,
        })
        .unwrap();
        assert_eq!((lib.entries, lib.targets), (0, 0));
        assert!(!std::path::Path::new(&lib.precursors).exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
