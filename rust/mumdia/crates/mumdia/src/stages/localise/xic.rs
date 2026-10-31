//! Fragment presence at the identification apex, read back from the MS2 spectra
//! (docs/36_ptm_localisation.md).
//!
//! The chromatograms extract writes hold only a candidate's top library fragments, and only
//! those of the peptidoform that was searched, so most site-determining ions, and every ion
//! of a placement the library does not carry, are not in them. The spectra are therefore
//! read again: the scans of the isolation windows that hold the precursor (and, on
//! diaPASEF, whose 1/K0 range holds its apex 1/K0), the `apex_scans` either side of the
//! scan nearest the apex, and an ion is present when any peak of those scans lies within
//! the fragment tolerance of its calibrated m/z (and, with 1/K0, inside the band around the
//! apex 1/K0).

use std::collections::BTreeMap;

use mumdia_core::types::{Ms2Scan, WindowKey};

use crate::masscal::MassOffset;

/// Slots whose 1/K0 range is widened by this much when deciding whether it holds the apex
/// 1/K0, so a precursor at a slot edge still reads its slot.
const SLOT_IM_SLACK: f64 = 0.02;

/// One isolation window and its scans.
struct Slot {
    lower_mz: f64,
    upper_mz: f64,
    im_lower: Option<f64>,
    im_upper: Option<f64>,
    /// Scan indices, ascending in retention time.
    scans: Vec<usize>,
}

/// The MS2 scans grouped by isolation window, in window-key order.
pub(crate) struct ScanIndex {
    slots: Vec<Slot>,
}

impl ScanIndex {
    /// Group `scans` (RT-sorted, as `spectra::load_ms2` returns them) by window.
    pub(crate) fn new(scans: &[Ms2Scan]) -> ScanIndex {
        let mut by: BTreeMap<WindowKey, Vec<usize>> = BTreeMap::new();
        for (i, s) in scans.iter().enumerate() {
            by.entry(s.window.key()).or_default().push(i);
        }
        let slots = by
            .into_values()
            .map(|v| {
                let w = scans[v[0]].window;
                Slot {
                    lower_mz: w.lower_mz,
                    upper_mz: w.upper_mz,
                    im_lower: w.im_lower.map(f64::from),
                    im_upper: w.im_upper.map(f64::from),
                    scans: v,
                }
            })
            .collect();
        ScanIndex { slots }
    }

    /// The scans the evidence of a precursor at `prec_mz` (apex `apex_rt`, apex 1/K0
    /// `apex_im` where known) is read from: the windows holding it, pooled, in retention-time
    /// order, and the `apex_scans` either side of the one nearest the apex.
    pub(crate) fn apex_scans(
        &self,
        scans: &[Ms2Scan],
        prec_mz: f64,
        apex_rt: f64,
        apex_im: Option<f64>,
        apex_scans: usize,
    ) -> Vec<usize> {
        let reach = apex_scans + 1;
        let mut pool: Vec<usize> = Vec::new();
        for slot in &self.slots {
            if prec_mz < slot.lower_mz || prec_mz > slot.upper_mz {
                continue;
            }
            if let (Some(im), Some(a), Some(b)) = (apex_im, slot.im_lower, slot.im_upper) {
                if im < a - SLOT_IM_SLACK || im > b + SLOT_IM_SLACK {
                    continue;
                }
            }
            let idx = &slot.scans;
            let at = idx.partition_point(|&i| scans[i].rt_seconds < apex_rt);
            let from = at.saturating_sub(reach);
            let to = (at + reach).min(idx.len());
            pool.extend_from_slice(&idx[from..to]);
        }
        if pool.is_empty() {
            return pool;
        }
        // Retention time, then scan position, so slots sharing a frame order deterministically.
        pool.sort_by(|&a, &b| {
            scans[a]
                .rt_seconds
                .total_cmp(&scans[b].rt_seconds)
                .then(a.cmp(&b))
        });
        pool.dedup();
        let mut k0 = 0;
        let mut best = f64::INFINITY;
        for (k, &i) in pool.iter().enumerate() {
            let d = (scans[i].rt_seconds - apex_rt).abs();
            if d < best {
                best = d;
                k0 = k;
            }
        }
        let lo = k0.saturating_sub(apex_scans);
        let hi = (k0 + apex_scans + 1).min(pool.len());
        pool[lo..hi].to_vec()
    }
}

/// Whether any peak of `scans[sel]` lies within `tol_ppm` of `mz` after the mass offset,
/// and inside `im +/- im_half` when a 1/K0 is given and the scan carries per-peak 1/K0.
pub(crate) fn present(
    scans: &[Ms2Scan],
    sel: &[usize],
    mz: f64,
    mass: &MassOffset,
    tol_ppm: f64,
    im: Option<(f64, f64)>,
) -> bool {
    let target = mz * mass.factor_at(mz);
    let (lo, hi) = (
        target * (1.0 - tol_ppm * 1e-6),
        target * (1.0 + tol_ppm * 1e-6),
    );
    for &i in sel {
        let s = &scans[i];
        let mut j = s.peaks.partition_point(|p| (p.mz as f64) < lo);
        while j < s.peaks.len() && (s.peaks[j].mz as f64) <= hi {
            if s.peaks[j].intensity > 0.0 {
                match (im, s.im.as_deref()) {
                    (Some((c, half)), Some(pim)) => {
                        if ((pim[j] as f64) - c).abs() <= half {
                            return true;
                        }
                    }
                    _ => return true,
                }
            }
            j += 1;
        }
    }
    false
}

/// The intensity-weighted median 1/K0 of the peaks matching `mzs` in `scans[sel]`, for a
/// row whose apex 1/K0 is not known. `None` without per-peak 1/K0 or without a match.
pub(crate) fn apex_im_of(
    scans: &[Ms2Scan],
    sel: &[usize],
    mzs: &[f64],
    mass: &MassOffset,
    tol_ppm: f64,
) -> Option<f64> {
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for &mz in mzs {
        let target = mz * mass.factor_at(mz);
        let (lo, hi) = (
            target * (1.0 - tol_ppm * 1e-6),
            target * (1.0 + tol_ppm * 1e-6),
        );
        for &i in sel {
            let s = &scans[i];
            let pim = s.im.as_deref()?;
            let mut j = s.peaks.partition_point(|p| (p.mz as f64) < lo);
            while j < s.peaks.len() && (s.peaks[j].mz as f64) <= hi {
                if s.peaks[j].intensity > 0.0 && pim[j] > 0.0 {
                    pts.push((pim[j] as f64, s.peaks[j].intensity as f64));
                }
                j += 1;
            }
        }
    }
    if pts.is_empty() {
        return None;
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = pts.iter().map(|p| p.1).sum();
    let mut acc = 0.0;
    for (v, w) in &pts {
        acc += w;
        if acc >= total / 2.0 {
            return Some(*v);
        }
    }
    pts.last().map(|p| p.0)
}
