//! SCIEX ZT Scan DIA files in the console: recognised before a search starts, and the recipe.
//!
//! A ZT Scan DIA mzML written by the SCIEX MS Data Converter names each MS2 spectrum's
//! quadrupole position (the isolation window target) and not the window the quadrupole
//! isolated. The engine refuses such a file unless `convert.isolation_half_width` says how
//! wide that window was: read without it, every spectrum would cover the whole m/z range and
//! the search would grow until the machine runs out of memory (docs/37_ztscan.md). The
//! console asks the engine before a search starts, through the `isolation` block of
//! `mumdia peak-census`, which applies the same rule `convert` refuses by rather than a copy
//! of it here, explains what it found, and fills in the half-width the engine suggests.
//!
//! The recipe (`configs/examples/ztscan.json`) is the imported-library recipe plus the keys
//! measured on ZT Scan; [`recipe_overrides`] is that difference, offered as overrides so it
//! applies on top of whichever preset the search uses.

use std::collections::BTreeMap;

use serde::Serialize;

/// The ZT Scan recipe, compiled in like the other presets (see `main.rs`).
pub const RECIPE: &str = include_str!("../../../configs/examples/ztscan.json");

/// The imported-library recipe the ZT Scan recipe extends.
pub const LIBRARY_RECIPE: &str = include_str!("../../../configs/examples/diann-library.json");

/// Files probed at once. Each probe is an engine process that reads the mzML's index (about
/// 0.7 GB for a 42 GB ZT Scan run) and its first [`PROBE_SPECTRA`] spectra (about 2 s).
const PROBES_AT_ONCE: usize = 4;

/// Spectra the probe reads from the head of each file: the sample `convert` decides its
/// refusal on (10,000 spectra, 15 cycles of the 663-step K562 method), so a method with more
/// steps a cycle is still seen to repeat.
pub const PROBE_SPECTRA: usize = 10_000;

/// What the engine's isolation census says about one selected file.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct Probe {
    pub path: String,
    /// The `isolation` block of `mumdia peak-census` (`convert_refuses`,
    /// `scanning_quadrupole`, `suggested_half_width`, `layout`, ...); `None` for a file that
    /// was not probed (not an mzML) or could not be read.
    pub isolation: Option<serde_json::Value>,
    /// Why the engine could not read the file, when it could not.
    pub error: Option<String>,
}

/// Whether a path is an mzML, the only input probed: a vendor file would be converted by the
/// census first, which takes minutes and writes beside the input.
pub fn is_mzml(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("mzml"))
}

/// Probe one file with `census` (the engine's peak census of its first 2,000 spectra).
fn probe_with(
    path: &str,
    census: &(dyn Fn(&str) -> Result<serde_json::Value, String> + Sync),
) -> Probe {
    if !is_mzml(path) {
        return Probe {
            path: path.to_string(),
            ..Probe::default()
        };
    }
    match census(path) {
        Ok(v) => Probe {
            path: path.to_string(),
            isolation: v.get("isolation").cloned(),
            error: None,
        },
        Err(e) => Probe {
            path: path.to_string(),
            isolation: None,
            error: Some(e),
        },
    }
}

/// Probe every path with `census`, [`PROBES_AT_ONCE`] at a time, in the order given.
pub fn probe_all_with(
    paths: &[String],
    census: &(dyn Fn(&str) -> Result<serde_json::Value, String> + Sync),
) -> Vec<Probe> {
    let mut out = Vec::with_capacity(paths.len());
    for chunk in paths.chunks(PROBES_AT_ONCE) {
        let got: Vec<Probe> = std::thread::scope(|s| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|p| s.spawn(move || probe_with(p, census)))
                .collect();
            handles
                .into_iter()
                .zip(chunk)
                .map(|(h, p)| {
                    h.join().unwrap_or_else(|_| Probe {
                        path: p.clone(),
                        isolation: None,
                        error: Some("the probe of this file failed".into()),
                    })
                })
                .collect()
        });
        out.extend(got);
    }
    out
}

/// Probe every selected mzML through the engine.
pub fn probe_all(paths: &[String]) -> Vec<Probe> {
    probe_all_with(paths, &|p: &str| {
        crate::preflight::peak_census_of(p, PROBE_SPECTRA)
    })
}

/// The ZT Scan recipe as overrides: every setting of [`RECIPE`] that [`LIBRARY_RECIPE`] does
/// not set to the same value. `convert.isolation_half_width` is among them; the interface
/// replaces it with the half-width the user confirmed.
pub fn recipe_overrides() -> Result<BTreeMap<String, serde_json::Value>, String> {
    let parse = |body: &str, name: &str| -> Result<BTreeMap<String, serde_json::Value>, String> {
        let v: serde_json::Value = serde_json::from_str(body)
            .map_err(|e| format!("the bundled {name} does not parse: {e}"))?;
        Ok(crate::settings::flatten(&v))
    };
    let zt = parse(RECIPE, "ztscan.json")?;
    let lib = parse(LIBRARY_RECIPE, "diann-library.json")?;
    Ok(zt
        .into_iter()
        .filter(|(k, v)| lib.get(k) != Some(v))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn zt_census() -> serde_json::Value {
        json!({"ms2_spectra": 1996, "isolation": {
            "ms2_spectra": 1996, "with_offsets": 0, "no_isolation": 0, "target_only": 1996,
            "convert_refuses": true, "scanning_quadrupole": true,
            "suggested_half_width": 1.9, "suggested_half_width_steps": 2.5,
            "layout": {"distinct_targets": 663, "lowest_target": 394.256,
                       "highest_target": 898.271, "median_step": 0.7614,
                       "spectra_per_target": 3.01, "cycle_s": 1.0}}})
    }

    /// The recipe is the imported-library recipe plus exactly the keys measured on ZT Scan.
    #[test]
    fn the_recipe_overrides_are_the_ztscan_keys() {
        let o = recipe_overrides().unwrap();
        let keys: Vec<&str> = o.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "convert.isolation_half_width",
                "extract.q1_fragments",
                "features.q1_consistency",
                "features.q1_deconvolved",
                "features.q1_fragment_features",
                "features.q1_multiscale",
                "features.q1_precursor",
                "quant.q1_cycle_halfwidth",
                "rescore.folds",
                "rescore.seeds",
                "rescore.shadow_q1",
                "rescore.train_margin_frac",
                "retrace.q1_repick",
                "rt_im_train.q1_calibrate",
            ],
            "{o:?}"
        );
        assert_eq!(o["convert.isolation_half_width"], json!(1.9));
        assert_eq!(o["extract.q1_fragments"], json!(true));
    }

    /// Every key the recipe sets is a setting the engine's schema knows, so a configuration
    /// derived with it is one the engine accepts.
    #[test]
    fn every_recipe_key_is_in_the_settings_schema() {
        let schema = crate::settings::load_schema().unwrap();
        for k in recipe_overrides().unwrap().keys() {
            assert!(
                schema.fields.iter().any(|f| &f.path == k),
                "{k} is not in configs/config-schema.json"
            );
        }
    }

    /// An mzML is probed and its isolation block returned; a vendor file is not probed; an
    /// unreadable file carries the engine's error; the order is the order given.
    #[test]
    fn the_probe_reads_the_isolation_block_of_each_mzml() {
        let paths: Vec<String> = [
            "/d/a.mzML",
            "/d/b.raw",
            "/d/c.mzml",
            "/d/bad.mzML",
            "/d/e.d",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let census = |p: &str| -> Result<serde_json::Value, String> {
            if p.contains("bad") {
                Err("the engine could not read this mzML".into())
            } else {
                Ok(zt_census())
            }
        };
        let got = probe_all_with(&paths, &census);
        assert_eq!(
            got.iter().map(|p| p.path.as_str()).collect::<Vec<_>>(),
            paths.iter().map(String::as_str).collect::<Vec<_>>()
        );
        assert_eq!(
            got[0].isolation.as_ref().unwrap()["suggested_half_width"],
            1.9
        );
        assert!(got[1].isolation.is_none() && got[1].error.is_none());
        assert_eq!(
            got[2].isolation.as_ref().unwrap()["scanning_quadrupole"],
            true
        );
        assert!(got[3].isolation.is_none());
        assert!(got[3].error.as_deref().unwrap().contains("could not read"));
        assert_eq!(
            got[4],
            Probe {
                path: "/d/e.d".into(),
                ..Probe::default()
            }
        );
        assert!(is_mzml("x.MZML") && !is_mzml("x.wiff2") && !is_mzml("x"));
    }
}
