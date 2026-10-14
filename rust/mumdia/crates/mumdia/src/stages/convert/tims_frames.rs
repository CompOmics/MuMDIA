//! Raw timsTOF frames held for event-level reads (docs/35_diapasef.md section 6).
//!
//! Convert centroids every frame once and keeps only the centroids. A stage that needs the
//! raw events afterwards (retrace) decodes the frames again through the same
//! [`super::RawTdf`] handle, so the events share convert's m/z and 1/K0 scales, and holds
//! them in this compact form: per frame, its quad settings and its events sorted by TOF and
//! packed into one `u64` each, with a table of the first event of every TOF bucket. A TOF
//! window is then one short binary search inside one bucket, whatever the scan band.
//!
//! The module knows nothing about candidates or chromatograms: it answers "which events of
//! this frame lie in this TOF window, in these scans", and the stage that asks decides what
//! to sum. A later raw-event consumer reuses it.

use std::sync::Arc;

use anyhow::{bail, Result};
use timsrust::{Frame, QuadrupoleSettings};

/// Bits of a packed event: `tof << TOF_SHIFT | scan << SCAN_SHIFT | intensity`.
const TOF_SHIFT: u32 = 44;
const SCAN_SHIFT: u32 = 32;
/// 64 TOF indices per bucket of the lookup table: about 6,000 entries (25 KB) per frame,
/// 0.9 GB for a HYE diaPASEF run.
const LUT_SHIFT: u32 = 6;

/// A held frame: its quad settings and its raw events sorted by TOF, each packed as
/// `tof << 44 | scan << 32 | intensity`, 8 bytes per event as in the decoded frame.
/// `lut[k]` is the index of the first event with TOF index `>= k << 6`, so finding the
/// start of a TOF window ([`TofFrame::first_at`]) searches one bucket instead of the whole
/// frame. The whole-frame search was 70-80% of retrace's busy time on HYE diaPASEF, as
/// cache misses; the table took one retrace call from 2:37 to 1:44 at 0.9 GB more memory.
pub(crate) struct TofFrame {
    quad: Arc<QuadrupoleSettings>,
    ev: Vec<u64>,
    lut: Vec<u32>,
}

impl TofFrame {
    /// Pack and sort the events of a decoded frame.
    pub(crate) fn new(f: Frame) -> Result<TofFrame> {
        let n_scans = f.scan_offsets.len().saturating_sub(1);
        if n_scans >= 1 << (TOF_SHIFT - SCAN_SHIFT) {
            bail!(
                "frame {}: {n_scans} scans do not fit the event packing",
                f.index
            );
        }
        let mut ev = Vec::with_capacity(f.tof_indices.len());
        for sc in 0..n_scans {
            for k in f.scan_offsets[sc]..f.scan_offsets[sc + 1] {
                let t = f.tof_indices[k] as u64;
                if t >= 1 << (64 - TOF_SHIFT) {
                    bail!(
                        "frame {}: TOF index {t} does not fit the event packing",
                        f.index
                    );
                }
                ev.push(t << TOF_SHIFT | (sc as u64) << SCAN_SHIFT | f.intensities[k] as u64);
            }
        }
        ev.sort_unstable();
        let n_buckets = ev
            .last()
            .map_or(0, |&e| (e >> TOF_SHIFT >> LUT_SHIFT) as usize + 1);
        let mut lut = Vec::with_capacity(n_buckets + 1);
        let mut i = 0usize;
        for k in 0..=n_buckets as u64 {
            while i < ev.len() && ev[i] >> TOF_SHIFT >> LUT_SHIFT < k {
                i += 1;
            }
            lut.push(i as u32);
        }
        Ok(TofFrame {
            quad: f.quadrupole_settings,
            ev,
            lut,
        })
    }

    /// The frame's quadrupole settings (its diaPASEF slots).
    pub(crate) fn quad(&self) -> &QuadrupoleSettings {
        &self.quad
    }

    /// Index of the first event with TOF index `>= t`: the same index as a `partition_point`
    /// over the whole frame (pinned by `first_at_equals_the_whole_frame_search`).
    pub(crate) fn first_at(&self, t: u32) -> usize {
        let k = (t >> LUT_SHIFT) as usize;
        match (self.lut.get(k), self.lut.get(k + 1)) {
            (Some(&a), Some(&b)) => {
                let (a, b) = (a as usize, b as usize);
                a + self.ev[a..b].partition_point(|&e| e < (t as u64) << TOF_SHIFT)
            }
            // The closing entry, `ev.len()`: `t` lies past every event.
            (Some(&a), None) => a as usize,
            _ => self.ev.len(),
        }
    }

    /// Call `g(scan, tof, intensity)` for every raw event with TOF index in `[t_lo, t_hi]`
    /// whose scan lies in one of `scans` (half-open ranges) and below `n_scans`, in TOF
    /// order.
    pub(crate) fn visit(
        &self,
        scans: &[(usize, usize)],
        t_lo: u32,
        t_hi: u32,
        n_scans: usize,
        mut g: impl FnMut(usize, u32, f64),
    ) {
        for &e in &self.ev[self.first_at(t_lo)..] {
            let tof = (e >> TOF_SHIFT) as u32;
            if tof > t_hi {
                break;
            }
            let sc = ((e >> SCAN_SHIFT) & ((1 << (TOF_SHIFT - SCAN_SHIFT)) - 1)) as usize;
            if sc < n_scans && scans.iter().any(|&(a, b)| (a..b).contains(&sc)) {
                g(sc, tof, (e & 0xffff_ffff) as f64);
            }
        }
    }
}

/// The raw events of one TOF window in one frame: summed intensity, the intensity-weighted
/// sums of their 1/K0 and of their TOF index, and their count.
#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sum {
    pub(crate) inten: f64,
    pub(crate) imw: f64,
    pub(crate) tofw: f64,
    pub(crate) n: u32,
}

/// Sum the raw events of `f` with TOF index in `[t_lo, t_hi]` whose scan lies in one of
/// `scans` (half-open ranges); `scan_im` is the 1/K0 of every scan of the run.
pub(crate) fn sum_events(
    f: &TofFrame,
    scans: &[(usize, usize)],
    t_lo: u32,
    t_hi: u32,
    scan_im: &[f64],
) -> Sum {
    let mut s = Sum::default();
    f.visit(scans, t_lo, t_hi, scan_im.len(), |sc, tof, w| {
        s.inten += w;
        s.imw += w * scan_im[sc];
        s.tofw += w * tof as f64;
        s.n += 1;
    });
    s
}

/// First scan (inclusive) and last scan (exclusive) with 1/K0 in `[lo, hi]`, given
/// `scan_im` strictly decreasing in the scan index, as TIMS mobility is.
pub(crate) fn scan_band(scan_im: &[f64], lo: f64, hi: f64) -> (usize, usize) {
    let a = scan_im.partition_point(|&v| v > hi);
    let b = scan_im.partition_point(|&v| v >= lo);
    (a, b.max(a))
}

/// The scan ranges of the quad slots whose isolation window holds `pmz`, clipped to the
/// band `(a, b)` and merged, so an event is never counted twice.
pub(crate) fn slot_scans(
    q: &QuadrupoleSettings,
    pmz: f64,
    (a, b): (usize, usize),
) -> Vec<(usize, usize)> {
    let mut r: Vec<(usize, usize)> = (0..q.len())
        .filter(|&k| (pmz - q.isolation_mz[k]).abs() <= q.isolation_width[k] / 2.0)
        .map(|k| (q.scan_starts[k].max(a), q.scan_ends[k].min(b)))
        .filter(|&(x, y)| x < y)
        .collect();
    r.sort_unstable();
    let mut m: Vec<(usize, usize)> = Vec::with_capacity(r.len());
    for (x, y) in r {
        match m.last_mut() {
            Some(last) if x <= last.1 => last.1 = last.1.max(y),
            _ => m.push((x, y)),
        }
    }
    m
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Two slots: scans 0..4 isolate 500 +/- 10, scans 4..8 isolate 600 +/- 10. Scan `s`
    /// holds TOF indices 100, 200 and 300, each with intensity `s + 1`.
    pub(crate) fn frame() -> Frame {
        let mut f = Frame::default();
        let mut off = vec![0usize];
        for s in 0..8u32 {
            for t in [100u32, 200, 300] {
                f.tof_indices.push(t);
                f.intensities.push(s + 1);
            }
            off.push(f.tof_indices.len());
        }
        f.scan_offsets = off;
        f.quadrupole_settings = Arc::new(QuadrupoleSettings {
            index: 1,
            scan_starts: vec![0, 4],
            scan_ends: vec![4, 8],
            isolation_mz: vec![500.0, 600.0],
            isolation_width: vec![20.0, 20.0],
            collision_energy: vec![0.0, 0.0],
        });
        f
    }

    #[test]
    fn first_at_equals_the_whole_frame_search() {
        let mut f = frame();
        // Events across several buckets, one on a bucket edge, and an empty bucket between.
        f.tof_indices[0] = 64;
        f.tof_indices[1] = 1000;
        let fr = TofFrame::new(f).unwrap();
        for t in 0..1200u32 {
            let want = fr.ev.partition_point(|&e| e < (t as u64) << TOF_SHIFT);
            assert_eq!(fr.first_at(t), want, "t = {t}");
        }
        let empty = TofFrame::new(Frame {
            scan_offsets: vec![0],
            ..Frame::default()
        })
        .unwrap();
        assert_eq!(empty.first_at(0), 0);
        assert_eq!(empty.first_at(500), 0);
    }

    #[test]
    fn scan_band_takes_the_scans_inside_the_mobility_interval() {
        let im = [1.4, 1.3, 1.2, 1.1, 1.0, 0.9];
        assert_eq!(scan_band(&im, 1.05, 1.25), (2, 4));
        assert_eq!(scan_band(&im, 1.0, 1.1), (3, 5));
        assert_eq!(scan_band(&im, 2.0, 3.0), (0, 0));
        assert_eq!(scan_band(&im, 0.1, 0.2), (6, 6));
    }

    #[test]
    fn slot_scans_keeps_the_covering_slot_clipped_to_the_band() {
        let f = frame();
        let q = &f.quadrupole_settings;
        assert_eq!(slot_scans(q, 505.0, (0, 8)), vec![(0, 4)]);
        assert_eq!(slot_scans(q, 605.0, (2, 6)), vec![(4, 6)]);
        assert_eq!(slot_scans(q, 700.0, (0, 8)), vec![]);
        assert_eq!(slot_scans(q, 505.0, (5, 8)), vec![]);
    }

    #[test]
    fn slot_scans_merges_overlapping_slots() {
        let q = QuadrupoleSettings {
            index: 1,
            scan_starts: vec![0, 2],
            scan_ends: vec![4, 6],
            isolation_mz: vec![500.0, 505.0],
            isolation_width: vec![20.0, 20.0],
            collision_energy: vec![0.0, 0.0],
        };
        assert_eq!(slot_scans(&q, 502.0, (0, 8)), vec![(0, 6)]);
    }

    #[test]
    fn sum_events_sums_the_tof_window_over_the_scans_with_weighted_mobility() {
        let f = TofFrame::new(frame()).unwrap();
        let im: Vec<f64> = (0..8).map(|s| 1.5 - 0.1 * s as f64).collect();
        // Scans 1..3, TOF 150..=300: TOF 200 and 300 in scans 1 and 2, 2*2 + 2*3 = 10.
        let s = sum_events(&f, &[(1, 3)], 150, 300, &im);
        assert_eq!((s.inten, s.n), (10.0, 4));
        assert!((s.imw / s.inten - (4.0 * 1.4 + 6.0 * 1.3) / 10.0).abs() < 1e-12);
        assert!((s.tofw / s.inten - 250.0).abs() < 1e-12);
        // An empty TOF window and a band past the last scan sum to nothing.
        assert_eq!(sum_events(&f, &[(0, 8)], 101, 199, &im), Sum::default());
        assert_eq!(sum_events(&f, &[(9, 12)], 0, 999, &im), Sum::default());
    }
}
