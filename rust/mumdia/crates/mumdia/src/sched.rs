//! Bounded concurrency for the orchestrators: vendor conversions and per-run chains.
//!
//! Two things run several at once under `run` and `run-experiment` and must keep the
//! serial loop's results: converting vendor inputs to mzML (`convert.parallel_conversions`)
//! and the per-run search chains under `experiment.parallel_runs = "auto"`. Both go
//! through [`map_bounded`], which keeps item order and reports the first failure in item
//! order. [`RunConcurrency`] is the per-run plan: how many chains at once and how wide
//! each chain's own thread pool is.

use std::path::Path;

use anyhow::{bail, Context, Result};
use tracing::{info, warn};

/// The stack reserved for each thread that runs an orchestrator step concurrently: the
/// workers of [`map_bounded`] and of the per-run pools of [`RunConcurrency::map_pooled`].
///
/// The default 2 MiB is what the explicit `parallel_runs` path has always run its chains
/// on (rayon's global pool), but the orchestrators' frames are large without optimisation,
/// which is why `main` runs the CLI on a 256 MiB thread (docs/30 R9). 64 MiB is generous
/// for a debug build and costs nothing in a release one: only touched pages are committed.
pub const WORKER_STACK_BYTES: usize = 64 << 20;

/// Sets `failed` when dropped during a panic, so a panicking item stops the other workers
/// from claiming more items just as a returned error does.
struct SetOnPanic<'a>(&'a std::sync::atomic::AtomicBool);

impl Drop for SetOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

/// Apply `f` to every item on at most `limit` threads, returning the results in item
/// order, or the error of the lowest-index item that failed.
///
/// Once any item has failed or panicked no further item is started; a panic is re-raised
/// after the items already running have finished. With `limit <= 1` this is the plain
/// serial loop, on the calling thread.
pub fn map_bounded<T, R, F>(items: &[T], limit: usize, f: F) -> Result<Vec<R>>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> Result<R> + Sync,
{
    if limit <= 1 || items.len() <= 1 {
        return items.iter().map(&f).collect();
    }
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let slots: Vec<Mutex<Option<Result<R>>>> = items.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for w in 0..limit.min(items.len()) {
            std::thread::Builder::new()
                .name(format!("mumdia-bounded-{w}"))
                .stack_size(WORKER_STACK_BYTES)
                .spawn_scoped(scope, || loop {
                    if failed.load(Ordering::SeqCst) {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(item) = items.get(i) else {
                        return;
                    };
                    let guard = SetOnPanic(&failed);
                    let r = f(item);
                    drop(guard);
                    if r.is_err() {
                        failed.store(true, Ordering::SeqCst);
                    }
                    *slots[i].lock().unwrap_or_else(|p| p.into_inner()) = Some(r);
                })
                // `scope.spawn` panics on this failure too; the named builder only adds the
                // stack size.
                .expect("spawning a worker thread");
        }
    });
    // Items are claimed in index order, so the started items are a prefix of the list
    // and every failure lies inside it: walking the slots in order meets the
    // lowest-index error before it can meet a slot that was never started.
    let mut out = Vec::with_capacity(items.len());
    for (i, slot) in slots.into_iter().enumerate() {
        match slot.into_inner().unwrap_or_else(|p| p.into_inner()) {
            Some(r) => out.push(r?),
            None => bail!("internal: item {i} was never run and no earlier one failed"),
        }
    }
    Ok(out)
}

/// The fewest threads a concurrent per-run chain is given under `parallel_runs = "auto"`.
///
/// One run per this many threads of the budget: a chain's stages are parallel, and a
/// share much narrower than this turns a many-core host's concurrency into long serial
/// stages. The figure is the perf survey's (2026-09-25, O2), not a measured optimum.
pub const AUTO_THREADS_PER_RUN: usize = 16;

/// The share of the memory the process can count on that concurrent chains may fill,
/// under `parallel_runs = "auto"` with a memory reading.
pub const AUTO_MEMORY_FRACTION: f64 = 0.7;

/// How the per-run chains of `run-experiment` share the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunConcurrency {
    /// Chains at once.
    pub par: usize,
    /// `Some(width)` under `"auto"`: each chain runs in a rayon pool of its own, this many
    /// threads wide, and the chains are pulled from a queue with no chunk barrier.
    /// `None` for an explicit count: chunks of `par` on the engine's one pool, as before.
    pub pool_threads: Option<usize>,
}

impl RunConcurrency {
    /// The plan for `setting` (`experiment.parallel_runs`, `0` for `"auto"`), `runs`
    /// chains and a budget of `threads`.
    pub fn resolve(setting: usize, runs: usize, threads: usize) -> Self {
        if setting > 0 {
            return Self {
                par: setting,
                pool_threads: None,
            };
        }
        let threads = threads.max(1);
        let par = (threads / AUTO_THREADS_PER_RUN).clamp(1, runs.max(1));
        Self {
            par,
            pool_threads: Some((threads / par).max(1)),
        }
    }

    /// Whether this is the automatic plan.
    pub fn is_auto(&self) -> bool {
        self.pool_threads.is_some()
    }

    /// Lower an automatic plan to at most `limit` chains at once, widening each chain's
    /// pool to the threads that frees. An explicit plan is never changed.
    pub fn bounded(self, limit: usize, threads: usize) -> Self {
        if !self.is_auto() || limit >= self.par {
            return self;
        }
        let par = limit.max(1);
        Self {
            par,
            pool_threads: Some((threads.max(1) / par).max(1)),
        }
    }

    /// Run `f` for every item under this plan and return the results in item order, or
    /// the first failure in item order.
    ///
    /// For the automatic plan (the explicit plan keeps the orchestrator's own chunk
    /// loops): each item runs inside a rayon pool of `pool_threads`, so
    /// `rayon::current_num_threads()` inside it is the chain's share, and a chain starts
    /// as soon as a slot is free. One chain at a time whose share is the calling pool's
    /// whole width needs no pool of its own: it runs on the calling thread, exactly as the
    /// explicit `parallel_runs = 1` loop does.
    pub fn map_pooled<T, R, F>(&self, items: &[T], f: F) -> Result<Vec<R>>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> Result<R> + Sync,
    {
        let width = self.pool_threads.unwrap_or_else(rayon::current_num_threads);
        if self.par <= 1 && width == rayon::current_num_threads() {
            return map_bounded(items, 1, f);
        }
        map_bounded(items, self.par, |item| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(width)
                .thread_name(|i| format!("mumdia-run-{i}"))
                .stack_size(WORKER_STACK_BYTES)
                .build()
                .context("building a per-run thread pool (experiment.parallel_runs = auto)")?;
            pool.install(|| f(item))
        })
    }
}

/// What the process can count on, read where the platform exposes it without a system
/// call the workspace would need `unsafe` for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryReading {
    /// The process's resident high-water mark (`VmHWM`), since start or since the last
    /// [`PeakWindow::open`] that could reset it.
    pub peak: u64,
    /// The process's resident memory now (`VmRSS`).
    pub resident: u64,
    /// What the system can still give it: `MemAvailable`, lowered to the headroom of the
    /// process's memory cgroup when that is smaller.
    pub available: u64,
    /// Whether a cgroup limit set `available` (a container or a batch job's allocation).
    pub cgroup_limited: bool,
}

/// The process's memory: Linux only, from `/proc/self/status` (`VmHWM`, `VmRSS`),
/// `/proc/meminfo` (`MemAvailable`) and the memory cgroup the process runs in (v2
/// `memory.max`, v1 `memory.limit_in_bytes`). `None` elsewhere, or when a value is
/// missing.
///
/// The cgroup matters because `/proc/meminfo` is not namespaced: inside a Docker container
/// or a SLURM or Kubernetes job with a memory limit, `MemAvailable` is the HOST's free
/// memory, and a 64 GB job on a 1 TB node would otherwise be sized as if it had the node.
pub fn memory_reading() -> Option<MemoryReading> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mem_available = kib_field(&meminfo, "MemAvailable:")?;
    let cgroup = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|text| {
            cgroup_headroom_in(
                &text,
                Path::new("/sys/fs/cgroup"),
                Path::new("/sys/fs/cgroup/memory"),
            )
        });
    let (available, cgroup_limited) = match cgroup {
        Some(h) if h < mem_available => (h, true),
        _ => (mem_available, false),
    };
    Some(MemoryReading {
        peak: kib_field(&status, "VmHWM:")?,
        resident: kib_field(&status, "VmRSS:")?,
        available,
        cgroup_limited,
    })
}

/// A `Name:   123 kB` line of a `/proc` file, in bytes.
fn kib_field(text: &str, name: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with(name))?;
    let kib: u64 = line[name.len()..].split_whitespace().next()?.parse().ok()?;
    Some(kib * 1024)
}

/// A cgroup memory limit file: `Some(None)` for no limit (v2 `max`, or v1's page-counter
/// maximum, which is how v1 spells "unlimited"), `Some(Some(bytes))` for a limit, `None`
/// when the text is not a limit.
fn cgroup_limit(text: &str) -> Option<Option<u64>> {
    let t = text.trim();
    if t == "max" {
        return Some(None);
    }
    let n: u64 = t.parse().ok()?;
    // v1 reports "unlimited" as PAGE_COUNTER_MAX pages, about 2^63 bytes.
    Some(if n >= 1 << 62 { None } else { Some(n) })
}

/// The `name value` line of a cgroup `memory.stat`.
fn stat_field(text: &str, name: &str) -> Option<u64> {
    text.lines().find_map(|l| {
        let mut it = l.split_whitespace();
        if it.next()? != name {
            return None;
        }
        it.next()?.parse().ok()
    })
}

/// The smallest headroom (limit minus working set) of the memory cgroup a process is in and
/// of its ancestors, or `None` when none of them has a limit.
///
/// `proc_cgroup` is the text of `/proc/self/cgroup`; `v2_root` and `v1_root` are where the
/// unified hierarchy and the v1 memory controller are mounted. The working set is the usage
/// less the inactive file cache, which the kernel reclaims before it enforces the limit
/// (the figure `docker stats` and the kubelet use); counting that cache as used would call
/// a container that has read a large file full. Ancestors count because a batch system sets
/// the limit on the job and runs the process in a step below it. When the process's own
/// path is not visible (a cgroup namespace shows it as `/`), the mount root is read, which
/// in a container is the container's own cgroup.
pub fn cgroup_headroom_in(proc_cgroup: &str, v2_root: &Path, v1_root: &Path) -> Option<u64> {
    let mut best: Option<u64> = None;
    for line in proc_cgroup.lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(_id), Some(controllers), Some(rel)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let (root, limit_file, usage_file, inactive_key) = if controllers.is_empty() {
            (v2_root, "memory.max", "memory.current", "inactive_file")
        } else if controllers.split(',').any(|c| c == "memory") {
            (
                v1_root,
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
                "total_inactive_file",
            )
        } else {
            continue;
        };
        let mut dir = root.join(rel.trim_start_matches('/'));
        if !dir.is_dir() {
            dir = root.to_path_buf();
        }
        loop {
            let limit = std::fs::read_to_string(dir.join(limit_file))
                .ok()
                .and_then(|t| cgroup_limit(&t))
                .flatten();
            let usage = std::fs::read_to_string(dir.join(usage_file))
                .ok()
                .and_then(|t| t.trim().parse::<u64>().ok());
            if let (Some(limit), Some(usage)) = (limit, usage) {
                let inactive = std::fs::read_to_string(dir.join("memory.stat"))
                    .ok()
                    .and_then(|t| stat_field(&t, inactive_key))
                    .unwrap_or(0);
                let headroom = limit.saturating_sub(usage.saturating_sub(inactive));
                best = Some(best.map_or(headroom, |b| b.min(headroom)));
            }
            if dir.as_path() == root {
                break;
            }
            match dir.parent() {
                Some(p) if p.starts_with(root) => dir = p.to_path_buf(),
                _ => break,
            }
        }
    }
    best
}

/// How many chains of `peak` bytes fit at once in [`AUTO_MEMORY_FRACTION`] of what the
/// process can count on: what it holds now plus what the system still has available.
/// Never below one.
pub fn chains_that_fit(peak: u64, resident: u64, available: u64) -> usize {
    if peak == 0 {
        return usize::MAX;
    }
    let room = resident.saturating_add(available) as f64 * AUTO_MEMORY_FRACTION;
    ((room / peak as f64).floor() as usize).max(1)
}

/// The start of a step whose peak memory sizes what runs after it.
///
/// On Linux, opening a window resets the process's resident high-water mark to what it
/// holds now (`/proc/self/clear_refs`, value 5, Linux 4.0 and later), so the `VmHWM` read
/// after the step is that step's peak and not the peak of everything before it: the
/// concurrent conversions, or the seed library of an earlier phase. The reset is
/// process-wide and final: under `parallel_runs = "auto"` a lifetime peak read from the
/// process's resource usage (`/usr/bin/time -v`, a parent's `wait4`) covers only the time
/// since the last window opened, so measure an auto experiment's memory with a sampling
/// profiler such as `bench/mem_profile.py`. When the reset is refused the reading includes
/// the earlier phases, which overstates the step and errs towards fewer chains, and the log
/// says so.
#[derive(Clone, Copy, Debug)]
pub struct PeakWindow {
    reset: bool,
}

impl PeakWindow {
    /// Open a window now.
    pub fn open() -> Self {
        #[cfg(target_os = "linux")]
        let reset = std::fs::write("/proc/self/clear_refs", "5").is_ok();
        #[cfg(not(target_os = "linux"))]
        let reset = false;
        Self { reset }
    }
}

/// Bound an automatic plan by the peak memory of `step`, which ran alone since `window`
/// opened, and log the plan. An explicit plan is returned unchanged.
///
/// The peak counts only this process: a sidecar's memory is not in it (see
/// [`one_at_a_time_for_sidecars`]). Without a reading (any platform but Linux) the plan
/// stays the thread-sized one and the log says so.
pub fn bound_by_measured_peak(
    plan: RunConcurrency,
    threads: usize,
    what: &str,
    step: &str,
    window: PeakWindow,
) -> RunConcurrency {
    if !plan.is_auto() {
        return plan;
    }
    match memory_reading() {
        Some(m) => {
            let fit = chains_that_fit(m.peak, m.resident, m.available);
            let bounded = plan.bounded(fit, threads);
            let note = if window.reset {
                ""
            } else {
                " (the high-water mark could not be reset, so the peak includes what ran \
                 before it)"
            };
            info!(
                peak_bytes = m.peak,
                resident_bytes = m.resident,
                available_bytes = m.available,
                cgroup_limited = m.cgroup_limited,
                peak_reset = window.reset,
                chains_that_fit = fit,
                parallel_runs = bounded.par,
                pool_threads = bounded.pool_threads.unwrap_or(0),
                "{what}: parallel_runs = auto, sized from the thread budget and the measured \
                 peak of {step}, which ran alone{note}"
            );
            bounded
        }
        None => {
            info!(
                parallel_runs = plan.par,
                pool_threads = plan.pool_threads.unwrap_or(0),
                "{what}: parallel_runs = auto, sized from the thread budget only (no memory \
                 reading on this platform)"
            );
            plan
        }
    }
}

/// Run `items` under `plan`. When the plan is automatic, has more than one slot, there is
/// more than one item and a memory reading exists, the first item runs alone on the
/// calling thread and the rest run under the plan bounded by its peak
/// ([`bound_by_measured_peak`]); otherwise `rest` runs every item under the plan.
///
/// `one` runs a single item; `rest` runs a slice of items under a plan (the caller chooses
/// plain threads or per-run pools). The results are in item order, and a failure of the
/// first item returns before anything else starts, as the serial loop would.
pub fn run_first_alone<T, R>(
    plan: RunConcurrency,
    threads: usize,
    what: &str,
    step: &str,
    items: &[T],
    one: impl FnOnce(&T) -> Result<R>,
    rest: impl FnOnce(RunConcurrency, &[T]) -> Result<Vec<R>>,
) -> Result<Vec<R>> {
    let measurable = plan.is_auto() && memory_reading().is_some();
    first_alone_then(plan, measurable, items, one, rest, |p, window| {
        bound_by_measured_peak(p, threads, what, step, window)
    })
}

/// [`run_first_alone`] with the reading and the bound passed in, so the ordering is
/// testable on every platform.
fn first_alone_then<T, R>(
    plan: RunConcurrency,
    measurable: bool,
    items: &[T],
    one: impl FnOnce(&T) -> Result<R>,
    rest: impl FnOnce(RunConcurrency, &[T]) -> Result<Vec<R>>,
    bound: impl FnOnce(RunConcurrency, PeakWindow) -> RunConcurrency,
) -> Result<Vec<R>> {
    if !measurable || plan.par <= 1 || items.len() <= 1 {
        return rest(plan, items);
    }
    let window = PeakWindow::open();
    let first = one(&items[0])?;
    let bounded = bound(plan, window);
    let mut out = Vec::with_capacity(items.len());
    out.push(first);
    out.extend(rest(bounded, &items[1..])?);
    Ok(out)
}

/// Run an automatic plan's chains one at a time, because each chain launches its own
/// DeepLC adaptation (`rt_library_scope = per_run`, or a first run that produced no
/// library to share).
///
/// That worker is a child process, and with sharded prediction a tree of them; its memory
/// is in no reading of this process, and the multi-head calibration is where a run's
/// largest process-tree peak sits (13.2 GB on the Astral runs). Sizing concurrent chains
/// from this process's peak would admit several such workers without counting any, so the
/// automatic plan does not try. An explicit `parallel_runs` is the user's own statement of
/// what fits and is returned unchanged.
pub fn one_at_a_time_for_sidecars(
    plan: RunConcurrency,
    threads: usize,
    what: &str,
) -> RunConcurrency {
    if !plan.is_auto() || plan.par <= 1 {
        return plan;
    }
    warn!(
        parallel_runs = 1,
        "{what}: parallel_runs = auto runs the per-run chains one at a time, because each \
         runs its own DeepLC adaptation in a child process whose memory the engine cannot \
         measure; set experiment.parallel_runs to a number to run several at once"
    );
    plan.bounded(1, threads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_map_keeps_item_order_and_the_concurrency_bound() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let items: Vec<usize> = (0..12).collect();
        let live = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let out = map_bounded(&items, 3, |&i| {
            let now = live.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            // Later items finish first, so any reordering would show.
            std::thread::sleep(std::time::Duration::from_millis(30 - 2 * i as u64));
            live.fetch_sub(1, Ordering::SeqCst);
            Ok(i * 10)
        })
        .unwrap();
        assert_eq!(out, (0..12).map(|i| i * 10).collect::<Vec<_>>());
        let peak = peak.load(Ordering::SeqCst);
        assert!(peak <= 3, "at most `limit` at once, saw {peak}");
        assert!(
            peak >= 2,
            "the bound is a bound, not a serial loop: saw {peak}"
        );
    }

    #[test]
    fn bounded_map_reports_the_first_failure_in_item_order_and_stops() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let items: Vec<usize> = (0..40).collect();
        let started = AtomicUsize::new(0);
        let err = map_bounded(&items, 4, |&i| {
            started.fetch_add(1, Ordering::SeqCst);
            // Item 5 fails slowly, item 6 fails at once: the reported error must still be
            // item 5's, as the serial loop would have reported.
            match i {
                5 => {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    bail!("input 5 failed")
                }
                6 => bail!("input 6 failed"),
                // Slow enough that the other workers cannot claim every remaining item
                // before item 6's failure is seen.
                _ => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    Ok(i)
                }
            }
        })
        .unwrap_err();
        assert_eq!(err.to_string(), "input 5 failed");
        assert!(
            started.load(Ordering::SeqCst) < items.len(),
            "no item may start after one has failed"
        );
        // Serial (limit 1) stops at the first failure too.
        let n = AtomicUsize::new(0);
        let err = map_bounded(&items, 1, |&i| {
            n.fetch_add(1, Ordering::SeqCst);
            if i == 2 {
                bail!("two")
            }
            Ok(i)
        })
        .unwrap_err();
        assert_eq!(err.to_string(), "two");
        assert_eq!(n.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn a_panicking_item_stops_the_queue_and_is_re_raised() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let items: Vec<usize> = (0..40).collect();
        let started = AtomicUsize::new(0);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            map_bounded(&items, 4, |&i| {
                started.fetch_add(1, Ordering::SeqCst);
                if i == 2 {
                    panic!("item 2 panicked");
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
                Ok(i)
            })
        }));
        assert!(r.is_err(), "the panic reaches the caller");
        assert!(
            started.load(Ordering::SeqCst) < items.len(),
            "no item may start after one has panicked, saw {}",
            started.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn an_explicit_count_is_kept_and_auto_is_sized_from_the_threads() {
        assert_eq!(
            RunConcurrency::resolve(3, 6, 128),
            RunConcurrency {
                par: 3,
                pool_threads: None
            }
        );
        // One run per 16 threads, never more than the runs.
        let auto = RunConcurrency::resolve(0, 6, 128);
        assert_eq!((auto.par, auto.pool_threads), (6, Some(21)));
        let auto = RunConcurrency::resolve(0, 20, 64);
        assert_eq!((auto.par, auto.pool_threads), (4, Some(16)));
        // A small host runs one at a time on the whole pool.
        let auto = RunConcurrency::resolve(0, 6, 8);
        assert_eq!((auto.par, auto.pool_threads), (1, Some(8)));
        // A bound narrows the count and widens each pool; it never touches an explicit one.
        let b = RunConcurrency::resolve(0, 20, 64).bounded(2, 64);
        assert_eq!((b.par, b.pool_threads), (2, Some(32)));
        assert_eq!(RunConcurrency::resolve(5, 6, 64).bounded(1, 64).par, 5);
        // Chains that launch their own DeepLC worker run one at a time under auto only.
        let one = one_at_a_time_for_sidecars(RunConcurrency::resolve(0, 20, 64), 64, "t");
        assert_eq!((one.par, one.pool_threads), (1, Some(64)));
        assert_eq!(
            one_at_a_time_for_sidecars(RunConcurrency::resolve(3, 20, 64), 64, "t").par,
            3
        );
    }

    #[test]
    fn the_memory_bound_counts_what_the_process_holds_and_what_is_free() {
        const G: u64 = 1 << 30;
        // 10 GB peak, 2 GB held, 98 GB free: 0.7 x 100 / 10 = 7.
        assert_eq!(chains_that_fit(10 * G, 2 * G, 98 * G), 7);
        // Never below one: a run that barely fit alone still runs.
        assert_eq!(chains_that_fit(100 * G, G, G), 1);
        assert_eq!(
            kib_field("Name:\tx\nVmHWM:\t  2048 kB\n", "VmHWM:"),
            Some(2048 * 1024)
        );
        assert_eq!(kib_field("VmRSS: 1 kB\n", "VmHWM:"), None);
    }

    #[test]
    fn cgroup_limits_parse_max_and_the_v1_unlimited_value() {
        assert_eq!(cgroup_limit("max\n"), Some(None));
        assert_eq!(cgroup_limit("68719476736\n"), Some(Some(68_719_476_736)));
        assert_eq!(cgroup_limit("9223372036854771712\n"), Some(None));
        assert_eq!(cgroup_limit("garbage"), None);
        assert_eq!(
            stat_field(
                "anon 5\ninactive_file 300\nactive_file 7\n",
                "inactive_file"
            ),
            Some(300)
        );
        assert_eq!(stat_field("anon 5\n", "inactive_file"), None);
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia_sched_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write(dir: &Path, files: &[(&str, &str)]) {
        std::fs::create_dir_all(dir).unwrap();
        for (n, t) in files {
            std::fs::write(dir.join(n), t).unwrap();
        }
    }

    #[test]
    fn the_cgroup_headroom_is_the_tightest_limit_on_the_path_less_the_working_set() {
        let root = scratch("cg");
        let v2 = root.join("v2");
        let v1 = root.join("v1");
        // A job limit two levels up, the step itself unlimited ("max"): the job's limit
        // counts, less its working set (usage minus the reclaimable inactive file cache).
        write(
            &v2.join("job"),
            &[
                ("memory.max", "1000\n"),
                ("memory.current", "700\n"),
                ("memory.stat", "anon 400\ninactive_file 300\n"),
            ],
        );
        write(
            &v2.join("job/step"),
            &[("memory.max", "max\n"), ("memory.current", "650\n")],
        );
        assert_eq!(
            cgroup_headroom_in("0::/job/step\n", &v2, &v1),
            Some(1000 - (700 - 300))
        );
        // A tighter limit on the step wins.
        write(&v2.join("job/step"), &[("memory.max", "500\n")]);
        assert_eq!(cgroup_headroom_in("0::/job/step\n", &v2, &v1), Some(0));
        // A namespaced container sees "/" and its own cgroup at the mount root.
        write(
            &v2,
            &[("memory.max", "2000\n"), ("memory.current", "500\n")],
        );
        assert_eq!(cgroup_headroom_in("0::/\n", &v2, &v1), Some(1500));
        // A path that is not visible falls back to the mount root.
        assert_eq!(cgroup_headroom_in("0::/elsewhere\n", &v2, &v1), Some(1500));
        // v1: the memory controller's line, with v1's "unlimited" ignored.
        write(
            &v1.join("slurm/job_1"),
            &[
                ("memory.limit_in_bytes", "4096\n"),
                ("memory.usage_in_bytes", "1096\n"),
                ("memory.stat", "total_inactive_file 96\n"),
            ],
        );
        write(&v1, &[("memory.limit_in_bytes", "9223372036854771712\n")]);
        assert_eq!(
            cgroup_headroom_in("5:cpu,cpuacct:/x\n4:memory:/slurm/job_1\n", &v2, &v1),
            Some(4096 - 1000)
        );
        // No limit anywhere: no headroom to apply.
        let empty = root.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert_eq!(cgroup_headroom_in("0::/\n", &empty, &empty), None);
        assert_eq!(cgroup_headroom_in("", &v2, &v1), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_first_item_runs_alone_and_sizes_the_rest() {
        let auto = RunConcurrency::resolve(0, 5, 64);
        assert_eq!(auto.par, 4);
        let order = std::sync::Mutex::new(Vec::new());
        let out = first_alone_then(
            auto,
            true,
            &[10, 11, 12, 13, 14],
            |&x| {
                order.lock().unwrap().push(format!("alone {x}"));
                Ok(x)
            },
            |p, xs| {
                order
                    .lock()
                    .unwrap()
                    .push(format!("rest {} at {}", xs.len(), p.par));
                Ok(xs.to_vec())
            },
            |p, _| p.bounded(2, 64),
        )
        .unwrap();
        assert_eq!(out, vec![10, 11, 12, 13, 14]);
        assert_eq!(*order.lock().unwrap(), ["alone 10", "rest 4 at 2"]);
        // Without a reading, or with one slot, everything goes through `rest` unbounded.
        for (plan, measurable) in [(auto, false), (RunConcurrency::resolve(0, 5, 8), true)] {
            let out = first_alone_then(
                plan,
                measurable,
                &[1, 2, 3],
                |_| -> Result<i32> { panic!("not alone") },
                |p, xs| {
                    assert_eq!(p, plan);
                    Ok(xs.to_vec())
                },
                |_, _| panic!("no bound"),
            )
            .unwrap();
            assert_eq!(out, vec![1, 2, 3]);
        }
        // A failure of the first item starts nothing else.
        let err = first_alone_then(
            auto,
            true,
            &[1, 2, 3],
            |_| -> Result<i32> { bail!("first failed") },
            |_, _| panic!("rest must not run"),
            |_, _| panic!("no bound"),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "first failed");
    }

    #[test]
    fn pooled_chains_see_their_own_pool_width() {
        let plan = RunConcurrency {
            par: 2,
            pool_threads: Some(3),
        };
        let widths = plan
            .map_pooled(&[0usize, 1, 2, 3], |&i| {
                Ok((i, rayon::current_num_threads()))
            })
            .unwrap();
        assert_eq!(widths, vec![(0, 3), (1, 3), (2, 3), (3, 3)]);
        // One chain at a time over the calling pool's whole width runs on the calling
        // thread, as the explicit `parallel_runs = 1` loop does.
        let whole = RunConcurrency {
            par: 1,
            pool_threads: Some(rayon::current_num_threads()),
        };
        let caller = std::thread::current().id();
        let ids = whole
            .map_pooled(&[0usize, 1], |_| Ok(std::thread::current().id()))
            .unwrap();
        assert!(ids.iter().all(|&id| id == caller));
    }
}
