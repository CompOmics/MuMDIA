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
    disk_multi(std::slice::from_ref(&mzml.to_string()), out_dir)
}

/// Bytes an input occupies, whether it is a file or an acquisition directory.
///
/// Recursive with a small depth cap, not one level. One level was wrong and the
/// codebase already knew it: an Agilent `.d` keeps its data under `AcqData/`, and
/// `thermo::label` identifies Agilent precisely by that subdirectory, so a one-level
/// sum measured an Agilent acquisition as 0 bytes and every space check on it was
/// trivially satisfied. The cap keeps a mistaken argument from walking a whole volume.
fn input_size(p: &str) -> u64 {
    fn dir_size(dir: &Path, depth: u32) -> u64 {
        if depth == 0 {
            return 0;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return 0;
        };
        entries
            .flatten()
            .map(|e| {
                let path = e.path();
                match e.metadata() {
                    Ok(m) if m.is_file() => m.len(),
                    Ok(m) if m.is_dir() => dir_size(&path, depth - 1),
                    _ => 0,
                }
            })
            .fold(0, u64::saturating_add)
    }
    match std::fs::metadata(p) {
        Ok(m) if m.is_file() => m.len(),
        Ok(m) if m.is_dir() => dir_size(Path::new(p), 4),
        _ => 0,
    }
}

/// Can this directory be written to? Answered by trying, because a permission bit is
/// not the whole story on a network share.
fn is_writable(dir: &Path) -> bool {
    let probe = dir.join(".mumdia-preflight-probe");
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Free bytes on the volume holding `dir`, walking up to the nearest existing
/// ancestor because the directory may not exist yet.
fn free_space(dir: &Path) -> Option<u64> {
    let mut probe = dir.to_path_buf();
    while !probe.exists() {
        match probe.parent() {
            Some(p) if p != probe => probe = p.to_path_buf(),
            _ => break,
        }
    }
    free_bytes(&probe)
}

/// Room for the CONVERTED mzML, on the volume it is actually written to.
///
/// `raw::ensure_mzml` puts it beside the input when that directory is writable, and
/// only falls back to the output directory otherwise. So a vendor input has a second
/// space requirement on a volume `disk` never looks at. Returns one warning per input
/// that does not obviously fit; an unreadable volume returns nothing rather than a
/// false alarm.
///
/// The multiplier is deliberately modest: an mzML is typically two to four times its
/// vendor file (the AIF `.raw` measured 887 MB against 664 MB from
/// ThermoRawFileParser and 1.10 GB from msconvert), so 4 is the pessimistic end of
/// what was actually observed rather than a guess.
pub fn conversion_space(inputs: &[String]) -> Vec<String> {
    const MZML_SIZE_MULTIPLE: u64 = 4;
    let mut out = Vec::new();
    for input in inputs {
        let p = Path::new(input);
        let ext = p
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        // mzML needs no conversion, so no extra space.
        if !matches!(ext.as_str(), "raw" | "d" | "wiff" | "wiff2") {
            continue;
        }
        let Some(dir) = p.parent() else { continue };
        // Only the beside-the-input case: if that directory is not writable the
        // engine uses the output directory, which `disk` already covers.
        if !is_writable(dir) {
            continue;
        }
        let need = input_size(input).saturating_mul(MZML_SIZE_MULTIPLE);
        let Some(free) = free_space(dir) else {
            continue;
        };
        if need > free {
            let gb = |b: u64| format!("{:.1} GB", b as f64 / 1e9);
            out.push(format!(
                "Converting {} to mzML needs roughly {} beside it, on a drive with {} \
                 free. The conversion writes next to the input, not into the results \
                 folder.",
                p.file_name().unwrap_or_default().to_string_lossy(),
                gb(need),
                gb(free)
            ));
        }
    }
    out
}

/// The same estimate for several inputs, whose intermediates coexist.
///
/// A pooled experiment writes every run's artifacts under one output directory and
/// keeps them, so the input size that matters is the sum. Sizing from one file would
/// understate an eight-run experiment roughly eightfold, and the engine cannot
/// resume: running out of space at hour three loses all of it.
///
/// A directory input (Bruker/Agilent `.d`, Waters `.raw`) has no meaningful length of
/// its own, so its contents are summed one level deep, which is where the acquisition
/// files sit.
pub fn disk_multi(inputs: &[String], out_dir: &str) -> Disk {
    let input: u64 = inputs
        .iter()
        .map(|p| input_size(p))
        .fold(0, u64::saturating_add);
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
    let mut cmd = crate::engine::command(&exe);
    // Every engine invocation gets the same environment; see `components::stamp_env`.
    crate::components::stamp_env(&mut cmd);
    let out = cmd
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
