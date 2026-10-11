//! Extended feature family: retention-time error scaled by the window it was searched in.
//!
//! Under a widen-only local RT window (`rt_im_train.local_window_anchors`) a candidate at a
//! gradient end may be searched several global widths from its predicted RT. The plain
//! `rt_error_*` features cannot tell a large error inside a legitimately widened window
//! from a large error that the global window would have excluded, so these scale the
//! signed error by the window half-width on its own side, and by the global half-width.
//! All 0 when `psms_extracted` carries no window (`extract.emit_rt_window` off).
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items in the same
//! order. Every value is finite.
use super::Evidence;

pub const NAMES: &[&str] = &[
    "rtw_scaled",
    "rtw_scaled_abs",
    "rtw_err_over_global",
    "rtw_excess_global",
    "rtw_widening",
];

pub fn values(e: &Evidence) -> Vec<f64> {
    scaled(e.apex_rt, e.rt_pred_cal, e.rt_lo, e.rt_hi, e.rt_w_global)
}

/// The family's values for an apex at `apex`, predicted RT `cal`, window `[lo, hi]` and
/// global half-width `w`; all 0 when any of them is missing.
fn scaled(apex: f64, cal: f64, lo: f64, hi: f64, w: f64) -> Vec<f64> {
    if ![apex, cal, lo, hi, w].iter().all(|x| x.is_finite()) || w <= 0.0 {
        return vec![0.0; NAMES.len()];
    }
    let err = apex - cal;
    let left = (cal - lo).max(1e-6);
    let right = (hi - cal).max(1e-6);
    let side = if err < 0.0 { left } else { right };
    let v = [
        err / side,
        err.abs() / side,
        err / w,
        (err.abs() - w).max(0.0) / w,
        (left + right) / (2.0 * w),
    ];
    v.iter()
        .map(|x| if x.is_finite() { *x } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_error_is_scaled_by_its_own_side_of_the_window() {
        // predicted 100 s, left side the global 8 s, right side widened to 24 s
        let v = scaled(112.0, 100.0, 92.0, 124.0, 8.0);
        assert_eq!(v, vec![0.5, 0.5, 1.5, 0.5, 2.0]);
        assert_eq!(
            scaled(96.0, 100.0, 92.0, 124.0, 8.0),
            vec![-0.5, 0.5, -0.5, 0.0, 2.0]
        );
        assert_eq!(scaled(96.0, 100.0, f64::NAN, 124.0, 8.0), vec![0.0; 5]);
    }
}
