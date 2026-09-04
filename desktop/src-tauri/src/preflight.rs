//! Checks run before a search starts, so a problem is explained rather than hit.
//!
//! Everything here answers a question that is cheap now and expensive later. The
//! engine cannot resume: a run that fills the disk at hour three has lost the whole
//! search, and a peak cap carried from another acquisition silently deletes fragment
//! evidence rather than failing.

use std::path::Path;

use serde::Serialize;

/// Rough output size for a single-run search, as a multiple of the mzML.
///
/// From the fixture and the recorded benchmark runs, chromatogram extraction
/// dominates and the whole output lands within a small multiple of the input. This
/// is a guard rail, not a prediction: it exists to catch "you have 4 GB free and a
/// 12 GB input", which is the case that loses a day.
const OUTPUT_SIZE_MULTIPLE: u64 = 5;

#[derive(Serialize, Default, Clone, Debug)]
pub struct Disk {
    pub input_bytes: u64,
    pub estimated_output_bytes: u64,
    pub free_bytes: u64,
    /// False when the estimate does not fit in the free space.
    pub enough: bool,
    /// True when free space could not be determined, in which case `enough` is not
    /// a judgement and the interface should say nothing rather than guess.
    pub unknown: bool,
}

/// Free bytes on the volume holding `path`.
///
/// Shelling out rather than calling the platform API, which would need `unsafe` in a
/// crate that has none. Both commands are present on a stock system and this runs
/// once per search, not in a loop.
fn free_bytes(path: &Path) -> Option<u64> {
    #[cfg(windows)]
    {
        // `wmic` is gone from recent Windows, so use PowerShell's provider, which is
        // present on every supported version.
        let drive = path.components().next().map(|c| {
            c.as_os_str()
                .to_string_lossy()
                .trim_end_matches('\\')
                .to_string()
        })?;
        let drive = drive.trim_end_matches(':').to_string();
        let out = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                &format!("(Get-PSDrive -Name '{drive}').Free"),
            ])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u64>()
            .ok()
    }
    #[cfg(unix)]
    {
        let out = std::process::Command::new("df")
            .args(["-kP", &path.display().to_string()])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        // "Filesystem 1024-blocks Used Available Capacity Mounted"
        let line = text.lines().nth(1)?;
        let available_kb: u64 = line.split_whitespace().nth(3)?.parse().ok()?;
        Some(available_kb * 1024)
    }
}

/// Is there room for this search's output?
pub fn disk(mzml: &str, out_dir: &str) -> Disk {
    let input = std::fs::metadata(mzml).map(|m| m.len()).unwrap_or(0);
    let estimate = input.saturating_mul(OUTPUT_SIZE_MULTIPLE);

    // The output directory may not exist yet, so ask about the nearest ancestor that
    // does; free space is a property of the volume either way.
    let mut probe = Path::new(out_dir).to_path_buf();
    while !probe.exists() {
        match probe.parent() {
            Some(p) if p != probe => probe = p.to_path_buf(),
            _ => break,
        }
    }

    match free_bytes(&probe) {
        Some(free) => Disk {
            input_bytes: input,
            estimated_output_bytes: estimate,
            free_bytes: free,
            enough: free >= estimate,
            unknown: false,
        },
        None => Disk {
            input_bytes: input,
            estimated_output_bytes: estimate,
            free_bytes: 0,
            enough: true,
            unknown: true,
        },
    }
}

/// Peaks per MS2 spectrum for the chosen file, from the engine.
///
/// The interface shows this next to the peak cap. `docs/04_convert.md` is emphatic
/// that a cap is acquisition-specific: on one 50-window Orbitrap DIA run a 300-peak
/// cap discarded 78.6% of MS2 peaks and cost 60% of the peptides. The application
/// has the file, so it can answer the question instead of asking a user to guess.
pub fn peak_census(mzml: &str) -> Result<serde_json::Value, String> {
    let (exe, _) = crate::engine::resolve()?;
    let out = crate::engine::command(&exe)
        .args(["peak-census", "--mzml", mzml, "--max-spectra", "2000"])
        .output()
        .map_err(|e| format!("could not run the engine: {e}"))?;
    serde_json::from_slice(&out.stdout).map_err(|_| {
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("the engine could not read this mzML")
            .trim()
            .to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_space_is_readable_for_a_directory_that_exists() {
        let d = disk(
            "does-not-exist.mzML",
            &std::env::temp_dir().display().to_string(),
        );
        // The input is missing, so the estimate is zero and there is trivially room;
        // what this asserts is that the free-space probe itself works on this
        // platform, because a silent `unknown` would disable the check for everyone.
        assert!(!d.unknown, "free space could not be read on this platform");
        assert!(d.free_bytes > 0, "a real volume reports some free space");
    }

    #[test]
    fn a_missing_output_directory_still_reports_its_volume() {
        // The directory a user picks usually does not exist yet.
        let missing = std::env::temp_dir().join("mumdia_no_such_dir_12345/deeper");
        let d = disk("does-not-exist.mzML", &missing.display().to_string());
        assert!(!d.unknown, "the nearest existing ancestor should answer");
    }

    #[test]
    fn the_estimate_scales_with_the_input() {
        let f = std::env::temp_dir().join("mumdia_preflight_probe.bin");
        std::fs::write(&f, vec![0u8; 1024]).unwrap();
        let d = disk(
            &f.display().to_string(),
            &std::env::temp_dir().display().to_string(),
        );
        assert_eq!(d.input_bytes, 1024);
        assert_eq!(d.estimated_output_bytes, 1024 * OUTPUT_SIZE_MULTIPLE);
        let _ = std::fs::remove_file(&f);
    }
}
