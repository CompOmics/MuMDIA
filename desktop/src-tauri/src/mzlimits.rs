//! The plausibility check of the Search screen's four m/z limits.
//!
//! The limits decide what a library contains: DIA-NN's `--min-pr-mz` / `--max-pr-mz` /
//! `--min-fr-mz` / `--max-fr-mz` for a DIA-NN build, and `peptidoforms.precursor_mz_*` /
//! `predict_frag.fragment_mz_*` for the built-in predictors. A value that is a positive
//! number but not a plausible m/z does not fail there: a highest fragment m/z of 1 made
//! DIA-NN generate 0 precursors, and the build failed two steps later in the importer with
//! a message about fragment rows. So every set limit is checked here before anything
//! starts, against bounds that any real acquisition is inside:
//!
//! - every fragment limit between [`FRAGMENT_LOWEST`] and [`FRAGMENT_HIGHEST`], and the
//!   highest one at least [`FRAGMENT_MAX_FLOOR`] (200, DIA-NN's default lowest fragment
//!   m/z: below it, and without a lower limit of its own, the range is empty);
//! - every precursor limit between [`PRECURSOR_LOWEST`] and [`PRECURSOR_HIGHEST`], and the
//!   highest one at least [`PRECURSOR_MAX_FLOOR`] (300, DIA-NN's default lowest precursor
//!   m/z, for the same reason);
//! - the lower limit of each pair below the upper one.
//!
//! An unset limit (`None`) is no limit and always passes. The interface applies the same
//! rule (`mzProblems` in `ui/app.js`); keep the two together.

use std::collections::BTreeMap;

/// The lowest fragment m/z limit accepted (a y1 ion of glycine is 76).
pub const FRAGMENT_LOWEST: f64 = 50.0;
/// The highest fragment m/z limit accepted.
pub const FRAGMENT_HIGHEST: f64 = 4000.0;
/// The smallest highest fragment m/z accepted.
pub const FRAGMENT_MAX_FLOOR: f64 = 200.0;
/// The lowest precursor m/z limit accepted.
pub const PRECURSOR_LOWEST: f64 = 100.0;
/// The highest precursor m/z limit accepted.
pub const PRECURSOR_HIGHEST: f64 = 6000.0;
/// The smallest highest precursor m/z accepted.
pub const PRECURSOR_MAX_FLOOR: f64 = 300.0;

/// The four limits, each `None` when unset.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MzLimits {
    pub precursor_min: Option<f64>,
    pub precursor_max: Option<f64>,
    pub fragment_min: Option<f64>,
    pub fragment_max: Option<f64>,
}

/// The Search screen's name of each limit, which every refusal uses.
pub const PRECURSOR_MIN_LABEL: &str = "Lowest precursor m/z";
pub const PRECURSOR_MAX_LABEL: &str = "Highest precursor m/z";
pub const FRAGMENT_MIN_LABEL: &str = "Lowest fragment m/z";
pub const FRAGMENT_MAX_LABEL: &str = "Highest fragment m/z";

fn fmt(v: f64) -> String {
    v.to_string()
}

impl MzLimits {
    /// Every problem with the limits, one sentence each naming the field and its value;
    /// empty when they are plausible.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (what, lowest, highest, floor, (lo_label, lo), (hi_label, hi)) in [
            (
                "precursor",
                PRECURSOR_LOWEST,
                PRECURSOR_HIGHEST,
                PRECURSOR_MAX_FLOOR,
                (PRECURSOR_MIN_LABEL, self.precursor_min),
                (PRECURSOR_MAX_LABEL, self.precursor_max),
            ),
            (
                "fragment",
                FRAGMENT_LOWEST,
                FRAGMENT_HIGHEST,
                FRAGMENT_MAX_FLOOR,
                (FRAGMENT_MIN_LABEL, self.fragment_min),
                (FRAGMENT_MAX_LABEL, self.fragment_max),
            ),
        ] {
            let mut pair_ok = true;
            for (label, v, least) in [(lo_label, lo, lowest), (hi_label, hi, floor)] {
                let Some(v) = v else { continue };
                if !v.is_finite() || v < least || v > highest {
                    pair_ok = false;
                    out.push(format!(
                        "{label} is {}, which is not a plausible {what} m/z: it must be \
                         between {} and {}, or empty for no limit.",
                        fmt(v),
                        fmt(least),
                        fmt(highest)
                    ));
                }
            }
            if let (true, Some(a), Some(b)) = (pair_ok, lo, hi) {
                if a >= b {
                    out.push(format!(
                        "{lo_label} ({}) must be below {hi_label} ({}).",
                        fmt(a),
                        fmt(b)
                    ));
                }
            }
        }
        out
    }

    /// `Ok` when the limits are plausible, otherwise every problem in one message.
    pub fn check(&self) -> Result<(), String> {
        let p = self.problems();
        if p.is_empty() {
            Ok(())
        } else {
            Err(p.join(" "))
        }
    }

    /// The limits as the run configuration's keys hold them (`derive_config`'s
    /// overrides): `peptidoforms.precursor_mz_*`, `predict_frag.fragment_mz_*`. A value
    /// that is not a number is reported as such.
    pub fn of_overrides(o: &BTreeMap<String, serde_json::Value>) -> Result<Self, String> {
        let get = |key: &str, label: &str| -> Result<Option<f64>, String> {
            match o.get(key) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(serde_json::Value::Number(n)) => Ok(n.as_f64()),
                Some(other) => Err(format!(
                    "{label} ({key}) is {other}, not a number; leave it empty for no limit."
                )),
            }
        };
        Ok(MzLimits {
            precursor_min: get("peptidoforms.precursor_mz_min", PRECURSOR_MIN_LABEL)?,
            precursor_max: get("peptidoforms.precursor_mz_max", PRECURSOR_MAX_LABEL)?,
            fragment_min: get("predict_frag.fragment_mz_min", FRAGMENT_MIN_LABEL)?,
            fragment_max: get("predict_frag.fragment_mz_max", FRAGMENT_MAX_LABEL)?,
        })
    }

    /// The limits as a setup holds them (`settings.digest.precursor_mz_min` and so on).
    /// `None` for a setup without them; a value that is not a number or null is left to
    /// the interface's type check.
    pub fn of_setup(settings: &serde_json::Value) -> Self {
        let d = settings.get("digest");
        let get = |k: &str| d.and_then(|d| d.get(k)).and_then(|v| v.as_f64());
        MzLimits {
            precursor_min: get("precursor_mz_min"),
            precursor_max: get("precursor_mz_max"),
            fragment_min: get("fragment_mz_min"),
            fragment_max: get("fragment_mz_max"),
        }
    }

    /// The limits for a message, as `precursor m/z 400 to any, fragment m/z any to 1`.
    pub fn describe(&self) -> String {
        let end = |v: Option<f64>| v.map(fmt).unwrap_or_else(|| "any".into());
        format!(
            "precursor m/z {} to {}, fragment m/z {} to {}",
            end(self.precursor_min),
            end(self.precursor_max),
            end(self.fragment_min),
            end(self.fragment_max)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lim(pmin: Option<f64>, pmax: Option<f64>, fmin: Option<f64>, fmax: Option<f64>) -> MzLimits {
        MzLimits {
            precursor_min: pmin,
            precursor_max: pmax,
            fragment_min: fmin,
            fragment_max: fmax,
        }
    }

    /// No limit, and the ranges people search with, pass.
    #[test]
    fn unset_and_ordinary_limits_pass() {
        assert!(MzLimits::default().check().is_ok());
        assert!(lim(Some(380.0), Some(980.5), Some(150.0), Some(1800.0))
            .check()
            .is_ok());
        assert!(lim(None, Some(1200.0), Some(200.0), None).check().is_ok());
        assert!(lim(Some(100.0), Some(6000.0), Some(50.0), Some(4000.0))
            .check()
            .is_ok());
    }

    /// The reported case: a highest fragment m/z of 1 is refused by name, with its value.
    #[test]
    fn a_highest_fragment_mz_of_one_is_refused_by_name() {
        let e = lim(None, None, None, Some(1.0)).check().unwrap_err();
        assert!(e.contains("Highest fragment m/z is 1,"), "{e}");
        assert!(e.contains("between 200 and 4000"), "{e}");
        for (l, field) in [
            (
                lim(Some(1.0), None, None, None),
                "Lowest precursor m/z is 1,",
            ),
            (
                lim(None, Some(299.0), None, None),
                "Highest precursor m/z is 299,",
            ),
            (
                lim(None, Some(7000.0), None, None),
                "Highest precursor m/z is 7000,",
            ),
            (
                lim(None, None, Some(10.0), None),
                "Lowest fragment m/z is 10,",
            ),
            (
                lim(None, None, None, Some(150.0)),
                "Highest fragment m/z is 150,",
            ),
            (
                lim(None, None, None, Some(f64::NAN)),
                "Highest fragment m/z is NaN,",
            ),
            (
                lim(None, None, Some(-1.0), None),
                "Lowest fragment m/z is -1,",
            ),
        ] {
            let e = l.check().unwrap_err();
            assert!(e.contains(field), "{field}: {e}");
        }
    }

    /// A reversed or empty range is refused, naming both fields.
    #[test]
    fn a_lower_limit_at_or_above_the_upper_is_refused() {
        let e = lim(Some(900.0), Some(400.0), None, None)
            .check()
            .unwrap_err();
        assert_eq!(
            e,
            "Lowest precursor m/z (900) must be below Highest precursor m/z (400)."
        );
        let e = lim(None, None, Some(500.0), Some(500.0))
            .check()
            .unwrap_err();
        assert!(e.contains("Lowest fragment m/z (500) must be below"), "{e}");
        // Each problem once: an implausible end is not also reported as a reversed range.
        assert_eq!(lim(None, None, Some(300.0), Some(1.0)).problems().len(), 1);
    }

    #[test]
    fn the_limits_are_read_from_overrides_and_setups() {
        let mut o = BTreeMap::new();
        o.insert(
            "predict_frag.fragment_mz_max".to_string(),
            serde_json::json!(1),
        );
        o.insert("digest.enzyme".to_string(), serde_json::json!("trypsin_p"));
        let l = MzLimits::of_overrides(&o).unwrap();
        assert_eq!(l, lim(None, None, None, Some(1.0)));
        o.insert(
            "peptidoforms.precursor_mz_min".to_string(),
            serde_json::json!("1,800"),
        );
        assert!(MzLimits::of_overrides(&o)
            .unwrap_err()
            .contains("not a number"));

        let s = serde_json::json!({"digest": {"fragment_mz_max": 1, "precursor_mz_min": null}});
        assert_eq!(MzLimits::of_setup(&s), lim(None, None, None, Some(1.0)));
        assert_eq!(
            MzLimits::of_setup(&serde_json::json!({})),
            MzLimits::default()
        );
        assert_eq!(
            lim(Some(400.0), None, None, Some(1.0)).describe(),
            "precursor m/z 400 to any, fragment m/z any to 1"
        );
    }
}
