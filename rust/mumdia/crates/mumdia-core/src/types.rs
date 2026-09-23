//! Core data-model types (docs/02_config_and_data_model.md). Ion mobility is `Option` /
//! nullable on the per-SCAN types, so the same model serves 3D and 4D runs; MVP is 3D so
//! IM is always `None`. It is deliberately absent from [`Peak`]: see that type.

use serde::{Deserialize, Serialize};

/// One observed peak, at the width the spectra artifact stores: two `f32`, 8 bytes.
///
/// This is the single largest resident array in the engine -- one entry per MS2 point for
/// the whole run, shared by every band under `groups.window_groups > 1` -- so its layout
/// is a footprint decision, not a style one. It used to be 24 bytes carrying 8 bytes of
/// information:
///
/// - `ion_mobility: Option<f32>` was 8 of the 24 (an `f32` has no niche, so the `Option`
///   costs a full word) and was written `None` at three sites and read at none. A 4D run
///   wants a per-scan `Vec<f32>` parallel to `peaks` anyway, added when 4D is supported;
///   a per-peak `Option` that is always `None` is not a step towards it.
/// - `mz: f64` was widened at load from the `f32` the artifact stores (`convert` writes
///   the column `f32`), so the f64 held nothing the f32 did not. Consumers widen with
///   `as f64` at the comparison, which is exact and therefore yields the very bit pattern
///   the `f64` field held. `Ms1Scan::mz` in the `mumdia` crate was already `Vec<f32>` on
///   exactly this argument.
///
/// 24 -> 8 bytes is measured, not asserted: see the `size_of` tests below. A decoded HYE
/// Astral run holds ~171.6 M MS2 points, so 3.84 GiB -> 1.28 GiB of resident peaks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    /// Peak m/z, `f32` exactly as the spectra artifact stores it. Widen at the point of
    /// use (`peak.mz as f64`); do not widen it here.
    pub mz: f32,
    pub intensity: f32,
}

/// A DIA isolation window in (m/z, 1/K0). IM bounds are `None` for plain DIA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsolationWindow {
    pub target_mz: f64,
    pub lower_mz: f64,
    pub upper_mz: f64,
    pub im_lower: Option<f32>,
    pub im_upper: Option<f32>,
}

impl IsolationWindow {
    #[inline]
    pub fn covers(&self, mz: f64) -> bool {
        mz >= self.lower_mz && mz <= self.upper_mz
    }
}

/// Whether a record is a target or a decoy, and which strategy made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Label {
    Target,
    Decoy,
}

impl Label {
    /// Percolator label: +1 target, -1 decoy.
    pub fn pin(&self) -> i32 {
        match self {
            Label::Target => 1,
            Label::Decoy => -1,
        }
    }
    pub fn is_decoy(&self) -> bool {
        matches!(self, Label::Decoy)
    }
}

/// Minimal in-memory MS2 scan handed to the seed search and extractor.
///
/// No `id`. The mzML native id is in the spectra artifact's `id` column and stays there;
/// the loader used to decode it into one `String` per scan and no stage read it. Scans are
/// addressed by `scan_index` everywhere, including in the artifacts they key.
#[derive(Clone, Debug)]
pub struct Ms2Scan {
    pub scan_index: u32,
    pub rt_seconds: f64,
    pub window: IsolationWindow,
    /// m/z sorted peaks.
    pub peaks: Vec<Peak>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    /// The peak array is the engine's largest resident buffer, so its per-element width is
    /// a contract rather than an implementation detail. A field added to `Peak` costs
    /// 8 bytes per MS2 point of every run (1.28 GiB on a HYE Astral run at 171.6 M points),
    /// and an `Option<f32>` costs a full word because an `f32` has no niche -- which is how
    /// the dead `ion_mobility` came to be a third of the struct.
    ///
    /// This pins the layout the load path and the ~22 read sites were sized for. If a 4D
    /// run needs ion mobility, give the SCAN a `Vec<f32>` parallel to `peaks` and leave
    /// this at 8.
    #[test]
    fn peak_is_two_f32_and_nothing_else() {
        assert_eq!(size_of::<Peak>(), 8, "Peak must stay two f32 wide");
        assert_eq!(size_of::<Peak>(), 2 * size_of::<f32>());
        // Why the `Option` was 8 and not 4, recorded so the next person does not have to
        // rediscover it before deciding what a field costs.
        assert_eq!(size_of::<Option<f32>>(), 8);
    }

    /// Widening a stored peak m/z to `f64` is exact, so every consumer that says
    /// `peak.mz as f64` sees the bit pattern the old `f64` field held. This is the whole
    /// equality argument for narrowing the field, so it is asserted rather than stated:
    /// round-trip a spread of values through `f32 -> f64 -> f32` and require identity.
    #[test]
    fn f32_mz_widens_to_f64_exactly() {
        for v in [
            0.0f32,
            1.0,
            100.0,
            133.107_1,
            1_999.999_9,
            2_000.000_1,
            f32::MIN_POSITIVE,
            f32::MAX,
            0.1,
            1e-30,
        ] {
            let wide = v as f64;
            assert_eq!(wide as f32, v, "widening {v} to f64 was not exact");
            assert_eq!(f64::from(v).to_bits(), wide.to_bits());
        }
    }
}
