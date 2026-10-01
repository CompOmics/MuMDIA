//! Ion-mobility features (TIMS roadmap P5), appended after every other column when
//! `features.im_features` is on. Not an Extended family: `FAMILIES` is always on under
//! Extended, and this block must leave the default feature vector untouched.
//!
//! Two groups. The scalars come from psms_extracted v4 (`apex_im`, `apex_im_mad`,
//! `ms1_apex_im`, `im_pred_cal`); the elution group from the per-point 1/K0 of the 4D
//! chromatograms (`im` in v3, `im_trimmed` in v4), inside the elution peak the other
//! features use. A missing input gives 0.0, so a 3D run yields constant columns. Targets
//! and decoys go through the same code; nothing here sees a label.
//!
//! A third, separately keyed group (`features.im_shape_features`, P7) scores the
//! mobility peak shape at the apex from psms_extracted v5, which extract fills from the
//! per-peak widths of spectra v3 with [`apex_shape`]. Each peak is modelled as a Gaussian
//! in 1/K0 (centroid, width) and agreement is their Bhattacharyya coefficient.

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

/// P7 peak-shape block, appended after [`NAMES`] under `features.im_shape_features`.
pub const SHAPE_NAMES: &[&str] = &[
    "im_frag_width",
    "im_frag_width_mad",
    "im_frag_overlap",
    "ms1_im_width",
    "ms1_frag_overlap",
    "ms1_frag_width_logratio",
];

/// The apex IM-shape values extract writes to psms_extracted v5, in [`SHAPE_NAMES`]
/// order minus the log ratio (derived in [`shape`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ApexShape {
    pub frag_width: Option<f64>,
    pub frag_width_mad: Option<f64>,
    pub frag_overlap: Option<f64>,
    pub ms1_width: Option<f64>,
    pub ms1_overlap: Option<f64>,
}

/// Bhattacharyya coefficient of N(m1, s1^2) and N(m2, s2^2): 1 for identical peaks,
/// towards 0 as centres separate or widths diverge. `None` unless both widths are
/// positive and finite.
pub fn bhattacharyya(m1: f64, s1: f64, m2: f64, s2: f64) -> Option<f64> {
    if !(s1 > 0.0 && s2 > 0.0 && s1.is_finite() && s2.is_finite()) {
        return None;
    }
    let v = s1 * s1 + s2 * s2;
    Some((2.0 * s1 * s2 / v).sqrt() * (-(m1 - m2).powi(2) / (4.0 * v)).exp())
}

/// Apex peak shape from the matched fragment peaks `(1/K0, intensity, width)` and,
/// when found, the MS1 precursor peak `(1/K0, width)`. The fragment consensus is
/// N(`center`, weighted median width), `center` being extract's `apex_im`.
pub fn apex_shape(frags: &[(f64, f64, f64)], center: f64, ms1: Option<(f64, f64)>) -> ApexShape {
    use super::super::search_seed::{weighted_mad, weighted_median};
    let mut widths: Vec<(f64, f64)> = frags.iter().map(|&(_, w, s)| (s, w)).collect();
    let Some(width) = weighted_median(&mut widths) else {
        return ApexShape::default();
    };
    let (mut sw, mut swo) = (0.0, 0.0);
    for &(m, w, s) in frags {
        if let Some(bc) = bhattacharyya(m, s, center, width) {
            sw += w;
            swo += w * bc;
        }
    }
    ApexShape {
        frag_width: Some(width),
        frag_width_mad: weighted_mad(&widths, width),
        frag_overlap: (sw > 0.0).then(|| swo / sw),
        ms1_width: ms1.map(|m| m.1),
        ms1_overlap: ms1.and_then(|(m, s)| bhattacharyya(m, s, center, width)),
    }
}

/// [`SHAPE_NAMES`] values; a missing input gives 0.0 (`has_ms1_im` of the P5 block flags
/// MS1 presence).
pub fn shape(a: &ApexShape) -> [f64; 6] {
    let ratio = match (a.ms1_width, a.frag_width) {
        (Some(m), Some(f)) if m > 0.0 && f > 0.0 => (m / f).ln(),
        _ => 0.0,
    };
    [
        a.frag_width.unwrap_or(0.0),
        a.frag_width_mad.unwrap_or(0.0),
        a.frag_overlap.unwrap_or(0.0),
        a.ms1_width.unwrap_or(0.0),
        a.ms1_overlap.unwrap_or(0.0),
        ratio,
    ]
}

/// `NAMES[..N_SCALAR]`: IM error of the apex against the calibrated prediction, fragment
/// IM dispersion at the apex, and MS1 precursor IM agreement.
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

/// `NAMES[N_SCALAR..]` from fragment traces `(rt, intensity, im)` restricted to
/// `[lo, hi]`, intensity-weighted, skipping points with zero intensity or no 1/K0:
/// |weighted mean 1/K0 - `im_cal`|, the weighted SD of all points, and the SD of the
/// per-fragment mean 1/K0 weighted by each fragment's summed intensity (0 with fewer
/// than two fragments).
pub fn elution<'a>(
    traces: impl Iterator<Item = (&'a [f32], &'a [f32], &'a [f32])>,
    lo: f64,
    hi: f64,
    im_cal: Option<f64>,
) -> [f64; N_ELUTION] {
    let mut pts: Vec<(f64, f64)> = Vec::new(); // (im, w), all fragments
    let mut frags: Vec<(f64, f64)> = Vec::new(); // (mean im, summed w), per fragment
    for (rt, inten, im) in traces {
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
    let Some((mean, sd)) = weighted_mean_sd(&pts) else {
        return [0.0; N_ELUTION];
    };
    let frag_sd = if frags.len() >= 2 {
        weighted_mean_sd(&frags).map_or(0.0, |m| m.1)
    } else {
        0.0
    };
    [im_cal.map_or(0.0, |c| (mean - c).abs()), sd, frag_sd]
}

/// Weighted mean and (population) SD of `(value, weight)` pairs; `None` when weightless.
fn weighted_mean_sd(pts: &[(f64, f64)]) -> Option<(f64, f64)> {
    let sw: f64 = pts.iter().map(|p| p.1).sum();
    if sw <= 0.0 {
        return None;
    }
    let mean = pts.iter().map(|p| p.0 * p.1).sum::<f64>() / sw;
    let var = pts.iter().map(|p| p.1 * (p.0 - mean).powi(2)).sum::<f64>() / sw;
    Some((mean, var.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bhattacharyya_is_one_for_identical_peaks_and_falls_with_offset_and_width() {
        assert!((bhattacharyya(1.0, 0.01, 1.0, 0.01).unwrap() - 1.0).abs() < 1e-12);
        // One SD apart at equal width: exp(-1/8).
        let bc = bhattacharyya(1.0, 0.01, 1.01, 0.01).unwrap();
        assert!((bc - (-0.125f64).exp()).abs() < 1e-12);
        // Width ratio 2 at one centre: sqrt(2*2/5).
        let bc = bhattacharyya(1.0, 0.01, 1.0, 0.02).unwrap();
        assert!((bc - 0.8f64.sqrt()).abs() < 1e-12);
        assert_eq!(bhattacharyya(1.0, 0.0, 1.0, 0.01), None);
    }

    #[test]
    fn apex_shape_scores_fragments_and_ms1_against_the_consensus() {
        // Two fragments on the consensus (width 0.01), one light one offset by 2 SD.
        let frags = [(1.0, 10.0, 0.01), (1.0, 10.0, 0.01), (1.02, 1.0, 0.01)];
        let a = apex_shape(&frags, 1.0, Some((1.0, 0.02)));
        assert_eq!(a.frag_width, Some(0.01));
        assert_eq!(a.frag_width_mad, Some(0.0));
        let want = (20.0 + (-0.5f64).exp()) / 21.0;
        assert!((a.frag_overlap.unwrap() - want).abs() < 1e-12);
        assert_eq!(a.ms1_width, Some(0.02));
        assert!((a.ms1_overlap.unwrap() - 0.8f64.sqrt()).abs() < 1e-12);
        let v = shape(&a);
        assert!((v[5] - 2f64.ln()).abs() < 1e-12);
        // No fragments: everything absent and neutral.
        assert_eq!(
            apex_shape(&[], 1.0, Some((1.0, 0.02))),
            ApexShape::default()
        );
        assert_eq!(shape(&ApexShape::default()), [0.0; 6]);
    }

    #[test]
    fn names_split_into_the_two_groups() {
        assert_eq!(NAMES.len(), N_SCALAR + N_ELUTION);
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

    #[test]
    fn elution_uses_only_weighted_points_inside_the_peak() {
        let rt = [1.0f32, 2.0, 3.0, 4.0];
        // Fragment a sits at 0.9, fragment b at 1.0; the point at rt 4 is outside the
        // peak, and the zero-intensity and no-IM points are skipped.
        let (ia, xa) = ([1.0f32, 1.0, 0.0, 9.0], [0.9f32, 0.9, 0.5, 2.0]);
        let (ib, xb) = ([1.0f32, 1.0, 1.0, 9.0], [1.0f32, 1.0, 0.0, 2.0]);
        let v = elution(
            [(&rt[..], &ia[..], &xa[..]), (&rt[..], &ib[..], &xb[..])].into_iter(),
            1.0,
            3.0,
            Some(1.0),
        );
        // Four points, two at 0.9 and two at 1.0: mean 0.95, SD 0.05; the two fragment
        // means are 0.9 and 1.0 at equal weight, SD 0.05.
        assert!((v[0] - 0.05).abs() < 1e-6, "{v:?}");
        assert!((v[1] - 0.05).abs() < 1e-6, "{v:?}");
        assert!((v[2] - 0.05).abs() < 1e-6, "{v:?}");
    }

    #[test]
    fn elution_is_zero_without_per_point_im() {
        let rt = [1.0f32, 2.0];
        let i = [5.0f32, 5.0];
        let v = elution(
            [(&rt[..], &i[..], &[][..])].into_iter(),
            0.0,
            9.0,
            Some(1.0),
        );
        assert_eq!(v, [0.0; N_ELUTION]);
        // One fragment: the between-fragment SD is 0, the point SD is not.
        let x = [0.9f32, 1.1];
        let v = elution([(&rt[..], &i[..], &x[..])].into_iter(), 0.0, 9.0, None);
        assert_eq!(v[0], 0.0);
        assert!((v[1] - 0.1).abs() < 1e-6);
        assert_eq!(v[2], 0.0);
    }
}
