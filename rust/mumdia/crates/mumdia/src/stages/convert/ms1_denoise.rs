//! The MS1 noise filter of a diaPASEF run (`convert.tdf_ms1_denoise`): the method and the
//! default MS1 settings of dnoise (P. T. Garrett, J. K. Diedrich and J. R. Yates III, "dnoise:
//! Fast Native Data Reduction for Bruker timsTOF", bioRxiv 2026,
//! doi:10.64898/2026.08.27.747603; MIT licence), implemented here on the raw
//! `(TOF index, scan, intensity)` points of one MS1 frame, as convert gathers them before
//! centroiding and as retrace holds them. Integer arithmetic throughout, three steps:
//!
//! 1. **Mobility streaks.** A real ion is a run along the mobility axis. For every TOF index
//!    `c` the intensities of the points within `c +/- 3` are summed per scan; occupied scans
//!    joined across at most 2 empty scans form runs, and the points at `c` are kept when
//!    their run has at least 5 occupied scans (the bridged empty scans do not count). The
//!    pass is applied twice, the second time to the survivors of the first.
//! 2. **Window gate.** An MS1 ion that no diaPASEF window ever isolates is never fragmented.
//!    The survivors are linked into features (within 3 TOF indices on one scan, or up to 3
//!    scans apart), and a feature is kept whole when any of its points lies inside an
//!    isolation window padded by 3 Th and 0.015 1/K0. No windows, no gate.
//! 3. **Halo.** A survivor below 10% of the largest survivor within +/- 2 scans and +/- 80 TOF
//!    indices, its own TOF index excluded, is dropped: the weak m/z flank of a bright ion.
//!
//! On the six ProteoBench HYE diaPASEF runs the tool's MS1 default kept 8.7% of the MS1
//! points, cut the MS1 centroids from 254 million to 8.6 million per run and left the
//! identifications unchanged (docs/35_diapasef.md section 16).

use std::sync::Arc;

use timsrust::QuadrupoleSettings;

/// TOF indices either side of a column whose points form its mobility profile.
const TOF_HALF_WIDTH: u32 = 3;
/// Occupied scans a run needs.
const MIN_OCCUPIED_SCANS: usize = 5;
/// Empty scans a run bridges.
const MAX_GAP_SCANS: usize = 2;
/// Passes of the streak filter.
const PASSES: usize = 2;
/// The halo box, in TOF indices and scans either side, and the fraction of its largest
/// off-column point a point must reach.
const HALO_TOF_HALF_WIDTH: u32 = 80;
const HALO_SCAN_HALF_WIDTH: u32 = 2;
const HALO_FRACTION: f64 = 0.10;
/// The isolation windows' padding in the gate, in Th and 1/K0 either side.
const GATE_MZ_PAD: f64 = 3.0;
const GATE_IM_PAD: f64 = 0.015;

/// One raw point: TOF index, scan, intensity.
pub(crate) type Pt = (u32, u32, u32);

/// The filter of one run: the window gate built from its isolation windows, if any.
#[derive(Debug, Default)]
pub(crate) struct Ms1Denoise {
    /// Per scan, the merged inclusive TOF intervals of the padded isolation windows.
    gate: Option<Vec<Vec<(u32, u32)>>>,
}

impl Ms1Denoise {
    /// The filter of a run with these diaPASEF window groups (`FrameReader::get_dia_windows`)
    /// over `n_scans` mobility scans, with the run's TOF index of an m/z and 1/K0 of a scan.
    /// Without windows only the streaks and the halo apply.
    pub(crate) fn new(
        windows: &[Arc<QuadrupoleSettings>],
        n_scans: usize,
        tof_of: impl Fn(f64) -> f64,
        im_of_scan: impl Fn(f64) -> f64,
    ) -> Ms1Denoise {
        if n_scans == 0 {
            return Ms1Denoise::default();
        }
        let mut per_scan: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_scans];
        let mut any = false;
        for q in windows {
            for k in 0..q.isolation_mz.len() {
                let half = q.isolation_width[k] / 2.0;
                let (mz_lo, mz_hi) = (q.isolation_mz[k] - half, q.isolation_mz[k] + half);
                let t0 = tof_of((mz_lo - GATE_MZ_PAD).max(0.0));
                let t1 = tof_of(mz_hi + GATE_MZ_PAD);
                if !(t0.is_finite() && t1.is_finite()) {
                    continue;
                }
                let tof_lo = t0.min(t1).floor().max(0.0) as u32;
                let tof_hi = t0.max(t1).ceil().max(0.0) as u32;
                let (s_lo, s_hi) =
                    padded_scans(q.scan_starts[k], q.scan_ends[k], n_scans, &im_of_scan);
                for row in &mut per_scan[s_lo..=s_hi] {
                    row.push((tof_lo, tof_hi));
                    any = true;
                }
            }
        }
        for row in &mut per_scan {
            row.sort_unstable();
            let mut merged: Vec<(u32, u32)> = Vec::with_capacity(row.len());
            for &(lo, hi) in row.iter() {
                match merged.last_mut() {
                    Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
                    _ => merged.push((lo, hi)),
                }
            }
            *row = merged;
        }
        Ms1Denoise {
            gate: any.then_some(per_scan),
        }
    }

    /// Whether the run's isolation windows gate the MS1 points.
    pub(crate) fn gated(&self) -> bool {
        self.gate.is_some()
    }

    /// Keep only the points of one MS1 frame the filter keeps, in their input order.
    /// `n_scans` is the frame's scan count.
    pub(crate) fn apply(&self, pts: &mut Vec<Pt>, n_scans: usize) {
        let keep = self.keep(pts, n_scans);
        let mut it = keep.into_iter();
        pts.retain(|_| it.next().unwrap_or(false));
    }

    /// Filter one decoded MS1 frame in place: its scan offsets, TOF indices and
    /// intensities keep the survivors, in their scan order.
    pub(crate) fn apply_to_frame(&self, f: &mut timsrust::Frame) {
        let n_scans = f.scan_offsets.len().saturating_sub(1);
        let mut pts: Vec<Pt> = Vec::with_capacity(f.tof_indices.len());
        for s in 0..n_scans {
            for k in f.scan_offsets[s]..f.scan_offsets[s + 1] {
                pts.push((f.tof_indices[k], s as u32, f.intensities[k]));
            }
        }
        self.apply(&mut pts, n_scans);
        let mut offsets = vec![0usize; n_scans + 1];
        for p in &pts {
            offsets[p.1 as usize + 1] += 1;
        }
        for s in 0..n_scans {
            offsets[s + 1] += offsets[s];
        }
        f.tof_indices = pts.iter().map(|p| p.0).collect();
        f.intensities = pts.iter().map(|p| p.2).collect();
        f.scan_offsets = offsets;
    }

    /// The keep mask of one MS1 frame's points, in their order.
    fn keep(&self, pts: &[Pt], n_scans: usize) -> Vec<bool> {
        let n = pts.len();
        let n_scans = pts
            .iter()
            .map(|p| p.1 as usize + 1)
            .max()
            .unwrap_or(0)
            .max(n_scans);
        let mut keep = vec![true; n];
        if n == 0 {
            return keep;
        }
        for _ in 0..PASSES {
            let active: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
            if active.is_empty() {
                break;
            }
            let sub: Vec<Pt> = active.iter().map(|&i| pts[i]).collect();
            let m = streak_pass(&sub, n_scans);
            let mut next = vec![false; n];
            for (j, &i) in active.iter().enumerate() {
                next[i] = m[j];
            }
            let alive = next.iter().any(|&b| b);
            keep = next;
            if !alive {
                break;
            }
        }
        if let Some(gate) = &self.gate {
            let inside: Vec<bool> = pts
                .iter()
                .map(|&(t, s, _)| {
                    gate.get(s as usize).is_some_and(|row| {
                        let i = row.partition_point(|&(_, hi)| hi < t);
                        i < row.len() && t >= row[i].0
                    })
                })
                .collect();
            let whole = features_touching(pts, &keep, &inside);
            for (k, w) in keep.iter_mut().zip(whole) {
                *k &= w;
            }
        }
        halo(pts, &mut keep, n_scans);
        keep
    }
}

/// The inclusive scans of an isolation window `[start, end)`, padded by [`GATE_IM_PAD`] in
/// 1/K0 and rounded outward: the edges go scan -> 1/K0 -> padded -> fractional scan (the
/// calibration is monotone, so the inverse is found by bisection).
fn padded_scans(
    start: usize,
    end: usize,
    n_scans: usize,
    im_of_scan: &impl Fn(f64) -> f64,
) -> (usize, usize) {
    const EPS: f64 = 1e-3;
    let last = n_scans - 1;
    let first_in = start.min(last);
    let last_in = end.saturating_sub(1).max(start).min(last);
    let (im0, im1) = (im_of_scan(first_in as f64), im_of_scan(last_in as f64));
    let scan_at = |target: f64| -> f64 {
        // 1/K0 falls with the scan index on a TIMS cell; bisect either way.
        let (mut a, mut b) = (0.0f64, last as f64);
        let rising = im_of_scan(b) > im_of_scan(a);
        if (rising && target <= im_of_scan(a)) || (!rising && target >= im_of_scan(a)) {
            return a;
        }
        if (rising && target >= im_of_scan(b)) || (!rising && target <= im_of_scan(b)) {
            return b;
        }
        for _ in 0..60 {
            let m = 0.5 * (a + b);
            let below = if rising {
                im_of_scan(m) < target
            } else {
                im_of_scan(m) > target
            };
            if below {
                a = m;
            } else {
                b = m;
            }
        }
        0.5 * (a + b)
    };
    let s0 = scan_at(im0.max(im1) + GATE_IM_PAD);
    let s1 = scan_at(im0.min(im1) - GATE_IM_PAD);
    let lo = ((s0.min(s1) + EPS).floor().max(0.0) as usize).min(last);
    let hi = ((s0.max(s1) - EPS).ceil().max(0.0) as usize).min(last);
    (lo, hi.max(lo))
}

/// One pass of the streak filter over `pts`: the keep mask, in their order.
fn streak_pass(pts: &[Pt], n_scans: usize) -> Vec<bool> {
    let n = pts.len();
    let mut keep = vec![false; n];
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| pts[i].0);
    let tof: Vec<u32> = order.iter().map(|&i| pts[i].0).collect();
    let mut profile = vec![0u64; n_scans];
    let mut touched: Vec<usize> = Vec::new();
    let mut occupied: Vec<usize> = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut k = 0;
    while k < n {
        let c = tof[k];
        let mut block_hi = k + 1;
        while block_hi < n && tof[block_hi] == c {
            block_hi += 1;
        }
        let w_lo = tof.partition_point(|&t| t < c.saturating_sub(TOF_HALF_WIDTH));
        let w_hi = tof.partition_point(|&t| t <= c.saturating_add(TOF_HALF_WIDTH));
        for &i in &order[w_lo..w_hi] {
            let s = pts[i].1 as usize;
            if profile[s] == 0 {
                touched.push(s);
            }
            profile[s] += pts[i].2 as u64;
        }
        occupied.clear();
        occupied.extend(touched.iter().copied().filter(|&s| profile[s] > 0));
        occupied.sort_unstable();
        spans.clear();
        if let Some(&first) = occupied.first() {
            let (mut start, mut prev, mut count) = (first, first, 1usize);
            for &s in &occupied[1..] {
                if s - prev > MAX_GAP_SCANS + 1 {
                    if count >= MIN_OCCUPIED_SCANS {
                        spans.push((start, prev));
                    }
                    (start, count) = (s, 0);
                }
                prev = s;
                count += 1;
            }
            if count >= MIN_OCCUPIED_SCANS {
                spans.push((start, prev));
            }
        }
        for &i in &order[k..block_hi] {
            let s = pts[i].1 as usize;
            let j = spans.partition_point(|&(_, end)| end < s);
            keep[i] = j < spans.len() && s >= spans[j].0;
        }
        for &s in &touched {
            profile[s] = 0;
        }
        touched.clear();
        k = block_hi;
    }
    keep
}

/// The kept points of every feature with a point `inside`: features link kept points
/// within [`TOF_HALF_WIDTH`] TOF indices on one scan (neighbours in TOF order) or up to
/// `MAX_GAP_SCANS + 1` scans apart. `false` for every point not kept.
fn features_touching(pts: &[Pt], keep: &[bool], inside: &[bool]) -> Vec<bool> {
    let n = pts.len();
    let mut out = vec![false; n];
    let mut idx: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    if idx.is_empty() {
        return out;
    }
    idx.sort_unstable_by_key(|&i| (pts[i].1, pts[i].0));
    let m = idx.len();
    let mut starts: Vec<(u32, usize)> = Vec::new();
    for (k, &i) in idx.iter().enumerate() {
        if starts.last().is_none_or(|&(s, _)| s != pts[i].1) {
            starts.push((pts[i].1, k));
        }
    }
    let rows = |si: usize| (starts[si].1, starts.get(si + 1).map_or(m, |&(_, k)| k));
    let mut parent: Vec<usize> = (0..m).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let union = |p: &mut Vec<usize>, a: usize, b: usize| {
        let (ra, rb) = (find(p, a), find(p, b));
        if ra != rb {
            p[ra] = rb;
        }
    };
    let gap = MAX_GAP_SCANS as u32 + 1;
    for si in 0..starts.len() {
        let (lo, hi) = rows(si);
        let s = starts[si].0;
        for k in lo..hi.saturating_sub(1) {
            if pts[idx[k + 1]].0 - pts[idx[k]].0 <= TOF_HALF_WIDTH {
                union(&mut parent, k, k + 1);
            }
        }
        for (sj, &(s2, _)) in starts.iter().enumerate().skip(si + 1) {
            if s2 - s > gap {
                break;
            }
            let (lo2, hi2) = rows(sj);
            for k in lo..hi {
                let t = pts[idx[k]].0;
                let row = &idx[lo2..hi2];
                let a = row.partition_point(|&i| pts[i].0 < t.saturating_sub(TOF_HALF_WIDTH));
                let b = row.partition_point(|&i| pts[i].0 <= t.saturating_add(TOF_HALF_WIDTH));
                for j in a..b {
                    union(&mut parent, k, lo2 + j);
                }
            }
        }
    }
    let mut hit = vec![false; m];
    for k in 0..m {
        if inside[idx[k]] {
            let r = find(&mut parent, k);
            hit[r] = true;
        }
    }
    for k in 0..m {
        let r = find(&mut parent, k);
        out[idx[k]] = hit[r];
    }
    out
}

/// Drop the kept points below [`HALO_FRACTION`] of the largest kept point in their halo box,
/// their own TOF index excluded.
fn halo(pts: &[Pt], keep: &mut [bool], n_scans: usize) {
    let idx: Vec<usize> = (0..pts.len()).filter(|&i| keep[i]).collect();
    if idx.is_empty() || n_scans == 0 {
        return;
    }
    let mut order = idx.clone();
    order.sort_unstable_by_key(|&i| (pts[i].1, pts[i].0));
    // Per scan: the slice of `order` holding its points, in TOF order.
    let mut block = vec![(0usize, 0usize); n_scans];
    let mut k = 0;
    while k < order.len() {
        let s = pts[order[k]].1 as usize;
        let start = k;
        while k < order.len() && pts[order[k]].1 as usize == s {
            k += 1;
        }
        block[s] = (start, k);
    }
    let mut drop: Vec<usize> = Vec::new();
    for &i in &order {
        let (t, s, inten) = pts[i];
        let lo_s = s.saturating_sub(HALO_SCAN_HALF_WIDTH) as usize;
        let hi_s = ((s + HALO_SCAN_HALF_WIDTH) as usize).min(n_scans - 1);
        let (lo_t, hi_t) = (
            t.saturating_sub(HALO_TOF_HALF_WIDTH),
            t.saturating_add(HALO_TOF_HALF_WIDTH),
        );
        let mut best = 0u32;
        for &(a, b) in &block[lo_s..=hi_s] {
            let row = &order[a..b];
            let j0 = row.partition_point(|&r| pts[r].0 < lo_t);
            for &r in &row[j0..] {
                let (tr, _, ir) = pts[r];
                if tr > hi_t {
                    break;
                }
                if tr != t && ir > best {
                    best = ir;
                }
            }
        }
        if (inten as f64) < HALO_FRACTION * best as f64 {
            drop.push(i);
        }
    }
    for i in drop {
        keep[i] = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vertical streak: TOF `t`, scans `scans`, intensity `w` each.
    fn streak(t: u32, scans: impl IntoIterator<Item = u32>, w: u32) -> Vec<Pt> {
        scans.into_iter().map(|s| (t, s, w)).collect()
    }

    #[test]
    fn a_run_needs_five_occupied_scans_and_bridges_two_empty_ones() {
        // TOF 100: scans 0..5 (five) kept. TOF 200: scans 0..4 (four) dropped. TOF 300:
        // 0, 1, 4, 5, 8 (gaps of 2) is one run of five occupied scans, kept. TOF 400: 0, 1,
        // 5, 6, 7 (a gap of 3) is two runs of two and three, dropped.
        let mut pts = streak(100, 0..5, 10);
        pts.extend(streak(200, 0..4, 10));
        pts.extend(streak(300, [0, 1, 4, 5, 8], 10));
        pts.extend(streak(400, [0, 1, 5, 6, 7], 10));
        let keep = streak_pass(&pts, 16);
        let kept: Vec<u32> = pts
            .iter()
            .zip(&keep)
            .filter(|(_, &k)| k)
            .map(|(p, _)| p.0)
            .collect();
        assert_eq!(
            kept,
            vec![100; 5]
                .into_iter()
                .chain(vec![300; 5])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn neighbouring_tof_indices_share_one_profile() {
        // Five scans spread over TOF 100..=103: each column alone has too few scans, but
        // within +/- 3 TOF indices the profile is a five-scan run, so every point is kept.
        let pts: Vec<Pt> = vec![
            (100, 0, 5),
            (101, 1, 5),
            (102, 2, 5),
            (103, 3, 5),
            (100, 4, 5),
        ];
        assert!(streak_pass(&pts, 8).iter().all(|&k| k));
        // Five TOF indices apart is outside the neighbourhood: nothing is kept.
        let far: Vec<Pt> = (0..5).map(|s| (100 + 5 * s, s, 5)).collect();
        assert!(streak_pass(&far, 8).iter().all(|&k| !k));
    }

    #[test]
    fn the_second_pass_sees_only_the_first_passs_survivors() {
        // TOF 100 has scans 0..3 (three); TOF 102 has scans 3..9, which also completes TOF
        // 100's profile to a run in the first pass. In the second pass TOF 100 is judged on
        // the survivors only, and they include TOF 102, so TOF 100 stays.
        let mut pts = streak(100, 0..3, 5);
        pts.extend(streak(102, 3..9, 5));
        let d = Ms1Denoise::default();
        assert!(d.keep(&pts, 16).iter().all(|&k| k));
        // A lone short streak beside a long one far in TOF is dropped by the first pass.
        let mut pts = streak(100, 0..3, 5);
        pts.extend(streak(150, 0..9, 5));
        let keep = d.keep(&pts, 16);
        assert!(keep[..3].iter().all(|&k| !k) && keep[3..].iter().all(|&k| k));
    }

    #[test]
    fn the_halo_drops_weak_flanks_but_never_the_own_column() {
        // A bright streak at TOF 1000 and a weak one at 1040 (within 80 TOF indices): the
        // weak one is below 10% of the bright one and goes. A weak streak 200 TOF indices
        // away stays, and the bright streak's own weaker scan stays (own column excluded).
        let mut pts = streak(1000, 0..6, 1000);
        pts.push((1000, 6, 20));
        pts.extend(streak(1040, 0..6, 50));
        pts.extend(streak(1200, 0..6, 50));
        let keep = Ms1Denoise::default().keep(&pts, 16);
        let kept_tof: std::collections::BTreeSet<u32> = pts
            .iter()
            .zip(&keep)
            .filter(|(_, &k)| k)
            .map(|(p, _)| p.0)
            .collect();
        assert_eq!(kept_tof, [1000, 1200].into_iter().collect());
        assert!(keep[6], "the own column's weak point is not a halo");
    }

    #[test]
    fn the_gate_keeps_a_feature_that_touches_a_window_and_drops_one_that_does_not() {
        // One window holding TOF 500..=600 at scans 0..=3 (a single slot of an identity
        // calibration, padding turned into a couple of TOF indices). A streak at TOF 550
        // over scans 2..10 touches the window at scans 2 and 3 and is kept whole; a streak at
        // TOF 900 never touches it and goes.
        let q = Arc::new(QuadrupoleSettings {
            index: 1,
            scan_starts: vec![0],
            scan_ends: vec![4],
            isolation_mz: vec![550.0],
            isolation_width: vec![100.0],
            collision_energy: vec![0.0],
        });
        // TOF index = m/z, 1/K0 = 2 - scan / 100 (falling with the scan index).
        let d = Ms1Denoise::new(&[q], 16, |mz| mz, |s| 2.0 - s / 100.0);
        assert!(d.gated());
        let mut pts = streak(550, 2..10, 10);
        pts.extend(streak(900, 2..10, 10));
        let keep = d.keep(&pts, 16);
        assert!(keep[..8].iter().all(|&k| k) && keep[8..].iter().all(|&k| !k));
        // Without windows there is no gate.
        assert!(!Ms1Denoise::new(&[], 16, |mz| mz, |s| 2.0 - s / 100.0).gated());
    }

    #[test]
    fn padded_scans_round_outward_from_the_window() {
        // 1/K0 = 2 - scan / 100: the window [10, 20) spans 1.90..1.81, padded by 0.015 to
        // 1.915..1.795, which is scans 8.5..20.5, rounded outward to 8..=21.
        let (lo, hi) = padded_scans(10, 20, 100, &|s: f64| 2.0 - s / 100.0);
        assert_eq!((lo, hi), (8, 21));
        // Clamped to the frame's scans.
        assert_eq!(padded_scans(0, 3, 4, &|s: f64| 2.0 - s / 100.0), (0, 3));
    }

    #[test]
    fn apply_keeps_the_input_order_of_the_survivors() {
        let mut pts = streak(300, (0..6).rev(), 7);
        pts.insert(2, (900, 1, 7));
        Ms1Denoise::default().apply(&mut pts, 16);
        assert_eq!(pts, streak(300, (0..6).rev(), 7));
    }
}
