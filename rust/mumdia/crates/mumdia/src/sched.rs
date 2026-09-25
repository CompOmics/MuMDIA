//! Bounded concurrency for the orchestrators: vendor conversions and per-run chains.
//!
//! Two things run several at once under `run` and `run-experiment` and must keep the
//! serial loop's results: converting vendor inputs to mzML (`convert.parallel_conversions`)
//! and the per-run search chains under `experiment.parallel_runs = "auto"`. Both go
//! through [`map_bounded`], which keeps item order and reports the first failure in item
//! order. [`RunConcurrency`] is the per-run plan: how many chains at once and how wide
//! each chain's own thread pool is.

use anyhow::{bail, Context, Result};
use tracing::info;

/// Apply `f` to every item on at most `limit` threads, returning the results in item
/// order, or the error of the lowest-index item that failed.
///
/// Once any item has failed no further item is started. With `limit <= 1` this is the
/// plain serial loop, on the calling thread.
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
        for _ in 0..limit.min(items.len()) {
            scope.spawn(|| loop {
                if failed.load(Ordering::SeqCst) {
                    return;
                }
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some(item) = items.get(i) else {
                    return;
                };
                let r = f(item);
                if r.is_err() {
                    failed.store(true, Ordering::SeqCst);
                }
                *slots[i].lock().unwrap_or_else(|p| p.into_inner()) = Some(r);
            });
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
    /// as soon as a slot is free.
    pub fn map_pooled<T, R, F>(&self, items: &[T], f: F) -> Result<Vec<R>>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> Result<R> + Sync,
    {
        let width = self.pool_threads.unwrap_or_else(rayon::current_num_threads);
        map_bounded(items, self.par, |item| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(width)
                .thread_name(|i| format!("mumdia-run-{i}"))
                .build()
                .context("building a per-run thread pool (experiment.parallel_runs = auto)")?;
            pool.install(|| f(item))
        })
    }
}

/// The process's memory, read where the platform exposes it without a system call the
/// workspace would need `unsafe` for: `(peak resident, resident, available)` in bytes.
///
/// Linux only, from `/proc/self/status` (`VmHWM`, `VmRSS`) and `/proc/meminfo`
/// (`MemAvailable`). `None` elsewhere, or when a value is missing.
pub fn memory_reading() -> Option<(u64, u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    Some((
        kib_field(&status, "VmHWM:")?,
        kib_field(&status, "VmRSS:")?,
        kib_field(&meminfo, "MemAvailable:")?,
    ))
}

/// A `Name:   123 kB` line of a `/proc` file, in bytes.
fn kib_field(text: &str, name: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with(name))?;
    let kib: u64 = line[name.len()..].split_whitespace().next()?.parse().ok()?;
    Some(kib * 1024)
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

/// Bound an automatic plan by the memory the process has used so far, read after a chain
/// that ran alone, and log the plan. An explicit plan is returned unchanged.
///
/// The peak is the process high-water mark, which includes everything the process held
/// before that chain (the conversions, the shared seed library), so it overstates one
/// chain and the bound errs towards fewer chains. Without a reading (any platform but
/// Linux) the plan stays the thread-sized one and the log says so.
pub fn bound_by_measured_peak(plan: RunConcurrency, threads: usize, what: &str) -> RunConcurrency {
    if !plan.is_auto() {
        return plan;
    }
    match memory_reading() {
        Some((peak, resident, available)) => {
            let fit = chains_that_fit(peak, resident, available);
            let bounded = plan.bounded(fit, threads);
            info!(
                peak_bytes = peak,
                resident_bytes = resident,
                available_bytes = available,
                chains_that_fit = fit,
                parallel_runs = bounded.par,
                pool_threads = bounded.pool_threads.unwrap_or(0),
                "{what}: parallel_runs = auto, sized from the thread budget and the measured \
                 peak of the first chain"
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
    }
}
