//! Ion-mobility family (`features.im_features`, docs/10_features.md). Part of the
//! `extended` set only, appended after every other family ([`super::families`]) when the
//! switch, resolved once per run or experiment, is on; with it off the feature vector and
//! its schema id are what they were before the family existed.
//!
//! Two groups. The scalars come from psms_extracted v4 (`apex_im`, `apex_im_mad`,
//! `ms1_apex_im`, `im_pred_cal`), carried on the [`Evidence`]; the elution group from the
//! per-point 1/K0 of the 4D chromatograms (`im` in v3, `im_trimmed` in v4), sampled on the
//! elution peak the other families use (`Evidence::im_traces`). A missing input gives 0.0,
//! so a run without 1/K0 yields constant columns. Targets and decoys go through the same
//! code, and nothing here sees a label: a decoy's library 1/K0 follows its target's
//! (docs/06), so the calibrated prediction carries no label either.

use super::Evidence;

pub const NAMES: &[&str] = &[
    "im_error",
    "im_error_abs",
    "im_frag_mad",
    "ms1_im_error_abs",
    "ms1_frag_im_diff",
    "has_ms1_im",
    "im_elution_error_abs",
    "im_elution_sd",
    "im_elution_frag_sd",
];
pub const N_SCALAR: usize = 6;
pub const N_ELUTION: usize = 3;

/// The family's values, in [`NAMES`] order.
pub fn values(e: &Evidence) -> Vec<f64> {
    let mut v = Vec::with_capacity(NAMES.len());
    v.extend(scalars(
        e.apex_im,
        e.im_pred_cal,
        e.apex_im_mad,
        e.ms1_apex_im,
    ));
    v.extend(elution(&e.traces, &e.im_traces, e.im_pred_cal));
    v
}

/// `NAMES[..N_SCALAR]`: the apex 1/K0 against the calibrated prediction, the fragment 1/K0
/// dispersion at the apex, and the MS1 precursor's 1/K0 agreement with both.
pub fn scalars(
    apex_im: Option<f64>,
    im_cal: Option<f64>,
    apex_mad: Option<f64>,
    ms1_im: Option<f64>,
) -> [f64; N_SCALAR] {
    let err = match (apex_im, im_cal) {
        (Some(a), Some(c)) => a - c,
        _ => 0.0,
    };
    let ms1_err = match (ms1_im, im_cal) {
        (Some(m), Some(c)) => (m - c).abs(),
        _ => 0.0,
    };
    let ms1_frag = match (ms1_im, apex_im) {
        (Some(m), Some(a)) => (m - a).abs(),
        _ => 0.0,
    };
    [
        err,
        err.abs(),
        apex_mad.unwrap_or(0.0),
        ms1_err,
        ms1_frag,
        if ms1_im.is_some() { 1.0 } else { 0.0 },
    ]
}

/// `NAMES[N_SCALAR..]` from the fragment traces over the elution peak (`traces`) and their
/// per-point 1/K0 on the same axis (`im_traces`), intensity-weighted, skipping points with
/// zero intensity or no 1/K0: |weighted mean 1/K0 - `im_cal`|, the weighted SD of all
/// points, and the SD of the per-fragment mean 1/K0 weighted by each fragment's summed
/// intensity (0 with fewer than two fragments). All 0 without per-point 1/K0.
pub fn elution(
    traces: &[Vec<f64>],
    im_traces: &[Vec<f64>],
    im_cal: Option<f64>,
) -> [f64; N_ELUTION] {
    let mut pts: Vec<(f64, f64)> = Vec::new(); // (im, w), all fragments
    let mut frags: Vec<(f64, f64)> = Vec::new(); // (mean im, summed w), per fragment
    for (inten, im) in traces.iter().zip(im_traces) {
        let (mut sw, mut swx) = (0.0, 0.0);
        for (&w, &x) in inten.iter().zip(im) {
            if w <= 0.0 || x <= 0.0 {
                continue;
            }
            pts.push((x, w));
            sw += w;
            swx += w * x;
        }
        if sw > 0.0 {
            frags.push((swx / sw, sw));
        }
    }
    let Some((mean, sd)) = crate::stats::weighted_mean_sd(&pts) else {
        return [0.0; N_ELUTION];
    };
    let frag_sd = if frags.len() >= 2 {
        crate::stats::weighted_mean_sd(&frags).map_or(0.0, |m| m.1)
    } else {
        0.0
    };
    [im_cal.map_or(0.0, |c| (mean - c).abs()), sd, frag_sd]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_split_into_the_two_groups() {
        assert_eq!(NAMES.len(), N_SCALAR + N_ELUTION);
        assert_eq!(NAMES.len(), 9);
    }

    #[test]
    fn scalars_measure_against_the_calibrated_prediction_and_are_neutral_without_im() {
        let v = scalars(Some(0.90), Some(0.95), Some(0.01), Some(0.93));
        let want = [-0.05, 0.05, 0.01, 0.02, 0.03, 1.0];
        for (a, b) in v.iter().zip(want) {
            assert!((a - b).abs() < 1e-12, "{v:?}");
        }
        assert_eq!(scalars(None, None, None, None), [0.0; N_SCALAR]);
        // No MS1 peak: the MS1 terms are 0 and flagged absent; the apex terms remain.
        let v = scalars(Some(0.90), Some(0.95), None, None);
        assert_eq!(v[3..], [0.0, 0.0, 0.0]);
        assert!((v[1] - 0.05).abs() < 1e-12);
    }

    /// The IM branch's elution over raw chromatogram rows `(rt, intensity, 1/K0)` restricted
    /// to `[lo, hi]`: the reference the aligned-trace form must equal bit for bit.
    fn raw_rows_elution<'a>(
        rows: impl Iterator<Item = (&'a [f32], &'a [f32], &'a [f32])>,
        lo: f64,
        hi: f64,
        im_cal: Option<f64>,
    ) -> [f64; N_ELUTION] {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        let mut frags: Vec<(f64, f64)> = Vec::new();
        for (rt, inten, im) in rows {
            let (mut sw, mut swx) = (0.0, 0.0);
            for ((&t, &w), &x) in rt.iter().zip(inten).zip(im) {
                let t = t as f64;
                if t < lo || t > hi || w <= 0.0 || x <= 0.0 {
                    continue;
                }
                pts.push((x as f64, w as f64));
                sw += w as f64;
                swx += w as f64 * x as f64;
            }
            if sw > 0.0 {
                frags.push((swx / sw, sw));
            }
        }
        let Some((mean, sd)) = crate::stats::weighted_mean_sd(&pts) else {
            return [0.0; N_ELUTION];
        };
        let frag_sd = if frags.len() >= 2 {
            crate::stats::weighted_mean_sd(&frags).map_or(0.0, |m| m.1)
        } else {
            0.0
        };
        [im_cal.map_or(0.0, |c| (mean - c).abs()), sd, frag_sd]
    }

    /// The traces sliced to the peak `lo..=hi` and widened, as the Evidence holds them.
    fn on_axis(rows: &[&[f32]], lo: usize, hi: usize) -> Vec<Vec<f64>> {
        rows.iter()
            .map(|r| r[lo..=hi].iter().map(|&v| v as f64).collect())
            .collect()
    }

    #[test]
    fn elution_on_the_peak_axis_is_the_raw_row_elution_bit_for_bit() {
        let rt = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        // Fragment a at 0.9, fragment b near 1.0, a third with one weighted point; the
        // points at rt 1 and 5 are outside the peak, and the zero-intensity and no-1/K0
        // points are skipped.
        let ia = [7.0f32, 1.0, 1.0, 0.0, 9.0];
        let xa = [0.3f32, 0.9, 0.9, 0.5, 2.0];
        let ib = [5.0f32, 1.0, 2.5, 1.0, 9.0];
        let xb = [0.1f32, 1.0, 1.01, 0.0, 2.0];
        let ic = [0.0f32, 0.0, 3.25, 0.0, 0.0];
        let xc = [0.0f32, 0.0, 0.97, 0.0, 0.0];
        for im_cal in [Some(1.0), Some(0.951), None] {
            let want = raw_rows_elution(
                [
                    (&rt[..], &ia[..], &xa[..]),
                    (&rt[..], &ib[..], &xb[..]),
                    (&rt[..], &ic[..], &xc[..]),
                ]
                .into_iter(),
                2.0,
                4.0,
                im_cal,
            );
            let got = elution(
                &on_axis(&[&ia, &ib, &ic], 1, 3),
                &on_axis(&[&xa, &xb, &xc], 1, 3),
                im_cal,
            );
            assert_eq!(
                got.map(f64::to_bits),
                want.map(f64::to_bits),
                "{im_cal:?}: {got:?} against {want:?}"
            );
            assert!(got[1] > 0.0 && got[2] > 0.0);
        }
    }

    #[test]
    fn elution_uses_only_weighted_points() {
        // Four points, two at 0.9 and two at 1.0: mean 0.95, SD 0.05; the two fragment
        // means are 0.9 and 1.0 at equal weight, SD 0.05.
        let v = elution(
            &[vec![1.0, 1.0, 0.0], vec![1.0, 1.0, 1.0]],
            &[vec![0.9, 0.9, 0.5], vec![1.0, 1.0, 0.0]],
            Some(1.0),
        );
        for (a, b) in v.iter().zip([0.05, 0.05, 0.05]) {
            assert!((a - b).abs() < 1e-12, "{v:?}");
        }
    }

    #[test]
    fn elution_is_zero_without_per_point_im() {
        let traces = vec![vec![5.0, 5.0]];
        assert_eq!(elution(&traces, &[], Some(1.0)), [0.0; N_ELUTION]);
        // One fragment: the between-fragment SD is 0, the point SD is not.
        let v = elution(&traces, &[vec![0.9, 1.1]], None);
        assert_eq!(v[0], 0.0);
        assert!((v[1] - 0.1).abs() < 1e-12);
        assert_eq!(v[2], 0.0);
    }
}
