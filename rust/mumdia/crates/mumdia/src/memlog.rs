//! Payload accounting for the largest owned buffers of each stage.
//!
//! `bench/mem_profile.py` samples process RSS, which says how much a stage holds but not
//! what holds it. These reports name the specific structures the memory audit
//! (`docs/27_memory_footprint_audit.md` section 1) models, so a measured stage peak can be
//! attributed rather than argued from the source.
//!
//! What a report is: the payload bytes of the named buffers, `len * size_of::<T>()`
//! summed over the nested `Vec`s. What it is not: `Vec` headers, spare capacity,
//! allocator slack, Arrow's own buffers, or anything the stage allocates transiently
//! between the reported points. A report is therefore a lower bound on the stage's RSS,
//! and the gap to the sampled peak is itself the interesting quantity: a large gap means
//! the model in section 1 is missing something.

use tracing::info;

const GIB: f64 = (1usize << 30) as f64;

/// Payload bytes of a slice of `T`, for the flat arrays.
#[inline]
pub fn bytes_of<T>(v: &[T]) -> usize {
    std::mem::size_of_val(v)
}

/// Payload bytes of a nested `Vec<Vec<T>>`: the inner values plus the outer spine of
/// `Vec` headers, which is itself gigabytes when there is one allocation per row.
pub fn bytes_of_nested<T>(v: &[Vec<T>]) -> usize {
    let spine = std::mem::size_of_val(v);
    let inner: usize = v.iter().map(|r| std::mem::size_of_val(r.as_slice())).sum();
    spine + inner
}

/// Log one buffer report: the total and each named part, in GiB.
///
/// Emitted at `info` so it lands in the same stream the profiler already parses for
/// stage boundaries; grep for `mem:` to pull the accounting out of a run log.
pub fn report(what: &str, parts: &[(&str, usize)]) {
    let total: usize = parts.iter().map(|(_, b)| *b).sum();
    let detail = parts
        .iter()
        .map(|(name, b)| format!("{name}={:.3}", *b as f64 / GIB))
        .collect::<Vec<_>>()
        .join(" ");
    info!(
        total_gib = total as f64 / GIB,
        parts_gib = %detail,
        "mem: {what}"
    );
}

/// The machine's physical memory in bytes, or the cgroup limit when a container imposes a
/// smaller one; `None` when it cannot be read. The ceiling `extract.max_footprint_gib`
/// falls back to when it is unset.
///
/// The workspace forbids `unsafe` and has no `sysinfo` dependency, so this reads
/// `/proc/meminfo` (and the cgroup v2 `memory.max`) on Linux and asks the platform's own
/// tool elsewhere: `sysctl -n hw.memsize` on macOS, Windows PowerShell's
/// `Win32_ComputerSystem.TotalPhysicalMemory` on Windows, as `storage::free_space` asks for
/// free space. A tool that is missing or refuses gives `None`.
pub fn total_memory() -> Option<u64> {
    if cfg!(target_os = "linux") {
        let total = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|m| parse_meminfo_total(&m));
        let limit = std::fs::read_to_string("/sys/fs/cgroup/memory.max")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
        return match (total, limit) {
            (Some(t), Some(l)) => Some(t.min(l)),
            (t, _) => t,
        };
    }
    let out = if cfg!(windows) {
        std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[uint64](Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory",
            ])
            .output()
            .ok()?
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?
    } else {
        return None;
    };
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|&b| b > 0)
}

/// `MemTotal` of `/proc/meminfo`, in bytes.
fn parse_meminfo_total(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meminfo_total_is_read_in_bytes() {
        assert_eq!(
            parse_meminfo_total("MemTotal:       16318408 kB\nMemFree: 1 kB\n"),
            Some(16318408 * 1024)
        );
        assert_eq!(parse_meminfo_total("MemFree: 1 kB\n"), None);
    }

    #[test]
    fn the_machine_memory_is_readable_here() {
        // Every platform CI and the fleet run on answers; the value only has to be sane.
        let b = total_memory().expect("physical memory is readable");
        assert!(b > (256 << 20), "{b}");
    }

    #[test]
    fn nested_counts_spine_and_values() {
        let v: Vec<Vec<f32>> = vec![vec![0.0; 4], vec![0.0; 6]];
        // 2 x 24-byte Vec header + 10 f32 values
        assert_eq!(
            bytes_of_nested(&v),
            2 * std::mem::size_of::<Vec<f32>>() + 40
        );
    }

    #[test]
    fn flat_counts_values() {
        assert_eq!(bytes_of(&[0u32; 8]), 32);
    }
}
