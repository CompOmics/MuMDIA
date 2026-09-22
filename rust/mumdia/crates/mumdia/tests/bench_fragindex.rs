//! Microbenchmark for the fragment-index probe loop, the hottest loop in the engine.
//! `#[ignore]`d, so `cargo test` never runs it; it is a harness for re-deriving the
//! measurements the comments in `fragindex.rs` and `extract.rs` cite, not an assertion.
//!
//!   cd rust/mumdia
//!   cargo test --release --test bench_fragindex -- --ignored --nocapture --test-threads=1
//!
//! Two parts, because a hoist out of a per-task loop cannot be judged on per-probe cost.
//!
//! Part 1 is one thread's cost per peak, which is what decides whether the per-peak setup
//! is worth taking off the probe's dependency chain. That ratio is set by how many
//! postings the candidate window leaves in the probed bins, so three arms sweep it.
//! EVERY ARM PAYS FOR ITS OWN SETUP INSIDE ITS OWN TIMER. A shared buffer cannot, in a
//! serial measurement, so its fill is timed separately and charged to it explicitly,
//! divided by the number of tasks that read it; reading a buffer somebody else filled is
//! not a measurement of anything.
//!
//! Part 2 is WALL CLOCK over rayon with extract's own task shape (`accumulate_groups`),
//! which is where a hoist that converts parallel work into serial work is decided: the
//! shared buffer is filled once on the calling thread before the pool starts, and the
//! per-task work it removes was spread across the pool.
//!
//! Peaks are m/z sorted within a scan, as production peaks are (`Ms2Scan.peaks` in
//! `mumdia-core/src/types.rs`; the extract loop walks them in that order). An earlier
//! version of this harness drew them uniformly at random in arbitrary order, which is a
//! different cache behaviour in exactly the direction that flatters a precomputed bin.
//!
//! Every arm asserts the same checksum, so an arm that computes something else fails
//! rather than winning.
//!
//! The machine is shared, so each arm is timed several times in a rotating order and the
//! min is the statistic to read; a single round separates arms 5% apart at best.

use mumdia::index::{Candidate, Library};
use mumdia::matchers::fragindex::FragIndex;
use rayon::prelude::*;

const TOL: f64 = 20.0;

/// `extract::MIN_CANDIDATES_PER_TASK`, mirrored so part 2 builds extract's task list.
const MIN_CANDIDATES_PER_TASK: usize = 4096;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f64 {
        self.next_u32() as f64 / 4294967296.0
    }
}

/// Same layout as the engine's MS2 peak (f64 m/z + f32 intensity).
#[derive(Clone, Copy)]
struct Pk {
    mz: f64,
    intensity: f32,
}

/// `extract::MassOffset`, mirrored. The default (`search_seed.mass_cal_loess = false`)
/// is a scalar; with the loess on, `search_seed.rs` builds a ~74-point grid on a fixed
/// 25 Th spacing and every peak pays a binary search plus an interpolation for it.
struct MassOff {
    scalar_ppm: f64,
    grid_mz: Vec<f64>,
    grid_ppm: Vec<f64>,
}
impl MassOff {
    fn scalar(ppm: f64) -> Self {
        MassOff {
            scalar_ppm: ppm,
            grid_mz: Vec::new(),
            grid_ppm: Vec::new(),
        }
    }
    /// The `mass_cal_loess` shape: 25 Th spacing over the fragment range.
    fn grid() -> Self {
        let mut grid_mz = Vec::new();
        let mut grid_ppm = Vec::new();
        let mut mz = 150.0f64;
        let mut i = 0;
        while mz <= 1900.0 {
            grid_mz.push(mz);
            grid_ppm.push(3.7 + 0.3 * ((i % 7) as f64 - 3.0));
            mz += 25.0;
            i += 1;
        }
        MassOff {
            scalar_ppm: 3.7,
            grid_mz,
            grid_ppm,
        }
    }
    #[inline]
    fn factor_at(&self, mz: f64) -> f64 {
        let ppm = if self.grid_mz.len() >= 2 {
            match self.grid_mz.binary_search_by(|g| g.total_cmp(&mz)) {
                Ok(i) => self.grid_ppm[i],
                Err(0) => self.grid_ppm[0],
                Err(i) if i >= self.grid_mz.len() => self.grid_ppm[self.grid_mz.len() - 1],
                Err(i) => {
                    let (x0, x1) = (self.grid_mz[i - 1], self.grid_mz[i]);
                    let (y0, y1) = (self.grid_ppm[i - 1], self.grid_ppm[i]);
                    y0 + (y1 - y0) * (mz - x0) / (x1 - x0)
                }
            }
        } else {
            self.scalar_ppm
        };
        1.0 + ppm * 1e-6
    }
}

fn build_lib(n_cand: usize, n_frag: usize) -> Library {
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let mut frag_mz: Vec<f32> = Vec::with_capacity(n_cand * n_frag);
    let mut frag_int: Vec<f32> = Vec::with_capacity(n_cand * n_frag);
    let mut frag_name_id: Vec<u16> = Vec::with_capacity(n_cand * n_frag);
    let mut prec_mz: Vec<f64> = Vec::with_capacity(n_cand);
    let mut cands: Vec<Candidate> = Vec::with_capacity(n_cand);
    for c in 0..n_cand {
        let p = 400.0 + 800.0 * (c as f64) / (n_cand as f64);
        let start = frag_mz.len();
        for _ in 0..n_frag {
            let mz = 150.0 + 1650.0 * rng.unit();
            frag_mz.push(mz as f32);
            frag_int.push(rng.unit() as f32);
            frag_name_id.push(0);
        }
        cands.push(Candidate {
            candidate_id: c as u32,
            peptidoform_id: c as u32,
            base_peptide_id: c as u32,
            peptidoform: String::new(),
            charge: 2,
            precursor_mz: p,
            predicted_irt: 0.0,
            is_decoy: false,
            protein: String::new(),
            frag_start: start,
            n_frag,
        });
        prec_mz.push(p);
    }
    Library {
        cands,
        frag_mz,
        frag_int,
        frag_name_id,
        frag_name_dict: vec!["f".to_string()],
        idx_mz: Vec::new(),
        idx_cid: Vec::new(),
        idx_int: Vec::new(),
        bucket_min: Vec::new(),
        bucket_size: 1,
        prec_mz,
        global_offset: 0,
    }
}

/// `n_scans` scans of `per_scan` peaks each, concatenated, EACH SCAN SORTED BY m/z --
/// the production stream. Scans are independent draws, so the concatenation is sawtooth
/// in m/z exactly as the engine's scan-major walk is.
fn sorted_scans(n_scans: usize, per_scan: usize, seed: u64) -> Vec<Pk> {
    let mut rng = Lcg(seed);
    let mut out: Vec<Pk> = Vec::with_capacity(n_scans * per_scan);
    for _ in 0..n_scans {
        let mut scan: Vec<Pk> = (0..per_scan)
            .map(|_| Pk {
                mz: 150.0 + 1650.0 * rng.unit(),
                intensity: rng.unit() as f32,
            })
            .collect();
        scan.sort_by(|a, b| a.mz.total_cmp(&b.mz));
        out.extend_from_slice(&scan);
    }
    out
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

const SCAN: usize = 300;

/// Arm A: the pre-commit shipped path. Query m/z and bin computed per peak inside the
/// probe.
fn arm_in_probe(idx: &FragIndex, pk: &[Pk], mo: &MassOff, lo: u32, hi: u32) -> (f64, u64) {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    let t = std::time::Instant::now();
    for scan in pk.chunks(SCAN) {
        for p in scan {
            let inten = p.intensity;
            let q = p.mz / mo.factor_at(p.mz);
            idx.probe_peak_win(&mut nw, q, |cid, pmz, pint, frag| {
                acc = acc
                    .wrapping_add(cid as u64)
                    .wrapping_add(pmz.to_bits())
                    .wrapping_add((pint + inten).to_bits() as u64)
                    .wrapping_add(frag as u64);
            });
        }
    }
    (ms(t.elapsed()), acc)
}

/// Arm F: per-scan scratch of bins, filled inside the timer. No shared buffer, the same
/// number of `ln()` calls as arm A -- only the order changed. The query m/z is
/// recomputed in the peak loop, so `factor_at` runs TWICE per peak.
fn arm_scratch_bin(idx: &FragIndex, pk: &[Pk], mo: &MassOff, lo: u32, hi: u32) -> (f64, u64) {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    let mut scratch: Vec<u32> = Vec::new();
    let t = std::time::Instant::now();
    for scan in pk.chunks(SCAN) {
        scratch.clear();
        scratch.extend(scan.iter().map(|p| idx.bin_of(p.mz / mo.factor_at(p.mz))));
        for (p, &b) in scan.iter().zip(&scratch) {
            let inten = p.intensity;
            let q = p.mz / mo.factor_at(p.mz);
            idx.probe_peak_win_binned(&mut nw, q, b, |cid, pmz, pint, frag| {
                acc = acc
                    .wrapping_add(cid as u64)
                    .wrapping_add(pmz.to_bits())
                    .wrapping_add((pint + inten).to_bits() as u64)
                    .wrapping_add(frag as u64);
            });
        }
    }
    (ms(t.elapsed()), acc)
}

/// Arm F2: per-scan scratch of `(q_mz, bin)`, filled inside the timer. 12 B per peak of
/// ONE scan (a few KB, reused), so unlike a per-window buffer it costs no memory worth
/// naming, and `factor_at` runs once per peak as it did before the split.
fn arm_scratch_qb(idx: &FragIndex, pk: &[Pk], mo: &MassOff, lo: u32, hi: u32) -> (f64, u64) {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    let mut scratch: Vec<(f64, u32)> = Vec::new();
    let t = std::time::Instant::now();
    for scan in pk.chunks(SCAN) {
        scratch.clear();
        scratch.extend(scan.iter().map(|p| {
            let q = p.mz / mo.factor_at(p.mz);
            (q, idx.bin_of(q))
        }));
        for (p, &(q, b)) in scan.iter().zip(&scratch) {
            let inten = p.intensity;
            idx.probe_peak_win_binned(&mut nw, q, b, |cid, pmz, pint, frag| {
                acc = acc
                    .wrapping_add(cid as u64)
                    .wrapping_add(pmz.to_bits())
                    .wrapping_add((pint + inten).to_bits() as u64)
                    .wrapping_add(frag as u64);
            });
        }
    }
    (ms(t.elapsed()), acc)
}

/// Arm G: the probe half of the shared-buffer design -- bins read from a buffer somebody
/// else filled. INCOMPLETE ON ITS OWN. `fill_cost` measures what it did not pay, and
/// `run_arm` charges it back divided by the number of tasks that read one fill.
fn arm_buffer_probe(
    idx: &FragIndex,
    pk: &[Pk],
    bb: &[u32],
    mo: &MassOff,
    lo: u32,
    hi: u32,
) -> (f64, u64) {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    let t = std::time::Instant::now();
    for (scan, bins) in pk.chunks(SCAN).zip(bb.chunks(SCAN)) {
        for (p, &b) in scan.iter().zip(bins) {
            let inten = p.intensity;
            let q = p.mz / mo.factor_at(p.mz);
            idx.probe_peak_win_binned(&mut nw, q, b, |cid, pmz, pint, frag| {
                acc = acc
                    .wrapping_add(cid as u64)
                    .wrapping_add(pmz.to_bits())
                    .wrapping_add((pint + inten).to_bits() as u64)
                    .wrapping_add(frag as u64);
            });
        }
    }
    (ms(t.elapsed()), acc)
}

/// The shared buffer's fill pass, exactly as `accumulate_groups` built it.
fn fill_pass(idx: &FragIndex, pk: &[Pk], mo: &MassOff, out: &mut Vec<u32>) -> f64 {
    let t = std::time::Instant::now();
    out.clear();
    out.extend(pk.iter().map(|p| idx.bin_of(p.mz / mo.factor_at(p.mz))));
    ms(t.elapsed())
}

fn stats(v: &mut [f64]) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2])
}

/// Part 1. `tasks_per_window` is how many tasks read one fill in the shape being
/// modelled; it is what the buffer's fill is amortised over.
fn run_arm(
    idx: &FragIndex,
    pk: &[Pk],
    mo: &MassOff,
    lo: u32,
    hi: u32,
    label: &str,
    tasks_per_window: f64,
) {
    let bb: Vec<u32> = pk
        .iter()
        .map(|p| idx.bin_of(p.mz / mo.factor_at(p.mz)))
        .collect();
    // warm up every path so no arm pays another's cold caches
    let (_, ha) = arm_in_probe(idx, pk, mo, lo, hi);
    let (_, hf) = arm_scratch_bin(idx, pk, mo, lo, hi);
    let (_, hf2) = arm_scratch_qb(idx, pk, mo, lo, hi);
    let (_, hg) = arm_buffer_probe(idx, pk, &bb, mo, lo, hi);
    assert_eq!(ha, hf, "arms A/F disagree ({label})");
    assert_eq!(ha, hf2, "arms A/F2 disagree ({label})");
    assert_eq!(ha, hg, "arms A/G disagree ({label})");
    let reps = 12usize;
    let mut a = Vec::new();
    let mut f = Vec::new();
    let mut f2 = Vec::new();
    let mut g = Vec::new();
    let mut fill = Vec::new();
    let mut scratch_buf: Vec<u32> = Vec::new();
    for r in 0..reps {
        // rotate the order, so warm-cache position cannot favour one arm
        let mut step = [0usize, 1, 2, 3, 4];
        step.rotate_left(r % 5);
        for s in step {
            match s {
                0 => a.push(arm_in_probe(idx, pk, mo, lo, hi).0),
                1 => f.push(arm_scratch_bin(idx, pk, mo, lo, hi).0),
                2 => f2.push(arm_scratch_qb(idx, pk, mo, lo, hi).0),
                3 => g.push(arm_buffer_probe(idx, pk, &bb, mo, lo, hi).0),
                _ => fill.push(fill_pass(idx, pk, mo, &mut scratch_buf)),
            }
        }
    }
    let (amin, amed) = stats(&mut a);
    let (fmin, fmed) = stats(&mut f);
    let (f2min, f2med) = stats(&mut f2);
    let (gmin, gmed) = stats(&mut g);
    let (qmin, qmed) = stats(&mut fill);
    let n = pk.len() as f64;
    // The buffer arm's honest serial cost: its probe plus the share of one fill that a
    // task reading it is responsible for.
    let gtotmin = gmin + qmin / tasks_per_window;
    let gtotmed = gmed + qmed / tasks_per_window;
    let pct = |x: f64, base: f64| 100.0 * (x - base) / base;
    println!(
        "  A in-probe            min {amin:7.2} med {amed:7.2} ms ({:6.1} ns/peak)",
        amin * 1e6 / n
    );
    println!(
        "  F scratch bin         min {fmin:7.2} med {fmed:7.2} ms ({:6.1} ns/peak)  delta min {:+6.1}% med {:+6.1}%",
        fmin * 1e6 / n,
        pct(fmin, amin),
        pct(fmed, amed)
    );
    println!(
        "  F2 scratch (q_mz,bin) min {f2min:7.2} med {f2med:7.2} ms ({:6.1} ns/peak)  delta min {:+6.1}% med {:+6.1}%",
        f2min * 1e6 / n,
        pct(f2min, amin),
        pct(f2med, amed)
    );
    println!(
        "  (buffer fill alone            {qmin:7.2} ms        ({:6.2} ns/peak), amortised over {tasks_per_window} tasks/window)",
        qmin * 1e6 / n
    );
    println!(
        "  G buffer probe only   min {gmin:7.2} med {gmed:7.2} ms ({:6.1} ns/peak)  delta min {:+6.1}% med {:+6.1}%  <- pays no fill, NOT an arm",
        gmin * 1e6 / n,
        pct(gmin, amin),
        pct(gmed, amed)
    );
    println!(
        "  G buffer + its fill   min {gtotmin:7.2} med {gtotmed:7.2} ms ({:6.1} ns/peak)  delta min {:+6.1}% med {:+6.1}%",
        gtotmin * 1e6 / n,
        pct(gtotmin, amin),
        pct(gtotmed, amed)
    );
}

fn setup_cost(idx: &FragIndex, pk: &[Pk], mo: &MassOff, label: &str) {
    let mut div = f64::INFINITY;
    let mut divbin = f64::INFINITY;
    for _ in 0..9 {
        let t = std::time::Instant::now();
        let mut s = 0.0f64;
        for p in pk {
            s += p.mz / mo.factor_at(p.mz);
        }
        std::hint::black_box(s);
        div = div.min(ms(t.elapsed()));

        let t = std::time::Instant::now();
        let mut s = 0u64;
        for p in pk {
            let q = p.mz / mo.factor_at(p.mz);
            s = s.wrapping_add(idx.bin_of(q) as u64);
        }
        std::hint::black_box(s);
        divbin = divbin.min(ms(t.elapsed()));
    }
    let n = pk.len() as f64;
    println!(
        "setup only ({label}): factor_at+divide {:.3} ns/peak, +bin {:.3} ns/peak => bin alone {:.3} ns/peak",
        div * 1e6 / n,
        divbin * 1e6 / n,
        (divbin - div) * 1e6 / n
    );
}

fn count_hits(idx: &FragIndex, pk: &[Pk], mo: &MassOff, lo: u32, hi: u32) -> f64 {
    let mut nw = idx.window_narrow(lo, hi);
    let mut n = 0u64;
    for p in pk {
        idx.probe_peak_win(&mut nw, p.mz / mo.factor_at(p.mz), |_, _, _, _| n += 1);
    }
    n as f64 / pk.len() as f64
}

#[test]
#[ignore]
fn bench_probe() {
    let n_cand = 200_000usize;
    let lib = build_lib(n_cand, 12);
    let t = std::time::Instant::now();
    let idx = FragIndex::build(&lib, TOL);
    println!(
        "index build {:.0} ms, {} postings",
        ms(t.elapsed()),
        n_cand * 12
    );
    let pk = sorted_scans(400_000 / SCAN, SCAN, 0xdead_beef_cafe_1234);
    let scalar = MassOff::scalar(3.7);
    let grid = MassOff::grid();
    setup_cost(&idx, &pk, &scalar, "scalar mass offset");
    setup_cost(&idx, &pk, &grid, "loess mass-offset grid");
    for &(w_lo, w_hi, name) in &[
        (700.0f64, 720.0f64, "narrow (20 m/z DIA window)"),
        (700.0, 800.0, "medium (100 m/z band)"),
        (0.0, 1e9, "wide (all-ion / whole band)"),
    ] {
        let (lo, hi) = idx.candidate_range(w_lo, w_hi);
        println!(
            "--- {name}: {} candidates, {:.2} hits/peak",
            hi - lo,
            count_hits(&idx, &pk, &scalar, lo, hi)
        );
        // 4 tasks per window is what the default shape produces; see `bench_wall`.
        run_arm(&idx, &pk, &scalar, lo, hi, name, 4.0);
    }
    println!("--- medium (100 m/z band), loess mass-offset grid");
    let (lo, hi) = idx.candidate_range(700.0, 800.0);
    run_arm(&idx, &pk, &grid, lo, hi, "medium/grid", 4.0);
}

// ---------------------------------------------------------------------------------
// Part 2: wall clock with extract's task shape.
// ---------------------------------------------------------------------------------

/// One isolation window: its candidate range and its scans (offsets into a flat peak
/// array). Mirrors `extract::WinGroup`.
struct Win {
    lo_cid: u32,
    hi_cid: u32,
    peaks: Vec<Pk>,
}

/// `accumulate_groups`' task list, from its own arithmetic (`extract.rs`): sub-range
/// width from the mean window span over `tasks_per_window`, floored at
/// `MIN_CANDIDATES_PER_TASK`, and one task per (sub-range, window) pair that intersects.
fn task_shape(wins: &[Win], threads: usize) -> (Vec<(usize, u32, u32)>, Vec<usize>) {
    let tasks_per_window = (threads * 2).div_ceil(wins.len()).max(1);
    let g_lo = wins.iter().map(|w| w.lo_cid).min().unwrap();
    let g_hi = wins.iter().map(|w| w.hi_cid).max().unwrap();
    let mean_span = wins
        .iter()
        .map(|w| (w.hi_cid - w.lo_cid) as usize)
        .sum::<usize>()
        / wins.len();
    let width = mean_span
        .div_ceil(tasks_per_window)
        .max(MIN_CANDIDATES_PER_TASK)
        .min(u32::MAX as usize) as u64;
    let n_sub = ((g_hi - g_lo) as u64).div_ceil(width).max(1) as usize;
    let mut tasks: Vec<(usize, u32, u32)> = Vec::new();
    let mut per_window = vec![0usize; wins.len()];
    for k in 0..n_sub {
        let s = (g_lo as u64 + k as u64 * width).min(g_hi as u64) as u32;
        let e = ((s as u64 + width).min(g_hi as u64)) as u32;
        for (gi, w) in wins.iter().enumerate() {
            let (lo, hi) = (w.lo_cid.max(s), w.hi_cid.min(e));
            if hi > lo {
                tasks.push((gi, lo, hi));
                per_window[gi] += 1;
            }
        }
    }
    (tasks, per_window)
}

/// One task's probe over its window's scans, bins computed inside the probe (pre-commit).
fn task_in_probe(idx: &FragIndex, w: &Win, mo: &MassOff, lo: u32, hi: u32) -> u64 {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    for scan in w.peaks.chunks(SCAN) {
        for p in scan {
            let inten = p.intensity;
            let q = p.mz / mo.factor_at(p.mz);
            idx.probe_peak_win(&mut nw, q, |cid, pmz, pint, frag| {
                acc = acc
                    .wrapping_add(cid as u64)
                    .wrapping_add(pmz.to_bits())
                    .wrapping_add((pint + inten).to_bits() as u64)
                    .wrapping_add(frag as u64);
            });
        }
    }
    acc
}

/// One task's probe with a per-scan `(q_mz, bin)` scratch.
fn task_scratch(idx: &FragIndex, w: &Win, mo: &MassOff, lo: u32, hi: u32) -> u64 {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    let mut scratch: Vec<(f64, u32)> = Vec::new();
    for scan in w.peaks.chunks(SCAN) {
        scratch.clear();
        scratch.extend(scan.iter().map(|p| {
            let q = p.mz / mo.factor_at(p.mz);
            (q, idx.bin_of(q))
        }));
        for (p, &(q, b)) in scan.iter().zip(&scratch) {
            let inten = p.intensity;
            idx.probe_peak_win_binned(&mut nw, q, b, |cid, pmz, pint, frag| {
                acc = acc
                    .wrapping_add(cid as u64)
                    .wrapping_add(pmz.to_bits())
                    .wrapping_add((pint + inten).to_bits() as u64)
                    .wrapping_add(frag as u64);
            });
        }
    }
    acc
}

/// One task's probe reading the window's shared bin buffer.
fn task_buffered(idx: &FragIndex, w: &Win, bins: &[u32], mo: &MassOff, lo: u32, hi: u32) -> u64 {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    for (scan, sb) in w.peaks.chunks(SCAN).zip(bins.chunks(SCAN)) {
        for (p, &b) in scan.iter().zip(sb) {
            let inten = p.intensity;
            let q = p.mz / mo.factor_at(p.mz);
            idx.probe_peak_win_binned(&mut nw, q, b, |cid, pmz, pint, frag| {
                acc = acc
                    .wrapping_add(cid as u64)
                    .wrapping_add(pmz.to_bits())
                    .wrapping_add((pint + inten).to_bits() as u64)
                    .wrapping_add(frag as u64);
            });
        }
    }
    acc
}

fn window_bins(idx: &FragIndex, w: &Win, mo: &MassOff) -> Vec<u32> {
    w.peaks
        .iter()
        .map(|p| idx.bin_of(p.mz / mo.factor_at(p.mz)))
        .collect()
}

/// Task checksums summed in a fixed order, so the arms are comparable whatever order
/// the pool finished in.
fn reduce(mut v: Vec<u64>) -> u64 {
    v.sort_unstable();
    v.iter().fold(0u64, |a, &b| a.wrapping_add(b))
}

/// Wall arm 1: pre-commit. Nothing before the pool.
fn wall_in_probe(
    idx: &FragIndex,
    wins: &[Win],
    mo: &MassOff,
    tasks: &[(usize, u32, u32)],
) -> (f64, u64) {
    let t = std::time::Instant::now();
    let out: Vec<u64> = tasks
        .par_iter()
        .map(|&(gi, lo, hi)| task_in_probe(idx, &wins[gi], mo, lo, hi))
        .collect();
    (ms(t.elapsed()), reduce(out))
}

/// Wall arm 2: per-scan scratch. Nothing before the pool.
fn wall_scratch(
    idx: &FragIndex,
    wins: &[Win],
    mo: &MassOff,
    tasks: &[(usize, u32, u32)],
) -> (f64, u64) {
    let t = std::time::Instant::now();
    let out: Vec<u64> = tasks
        .par_iter()
        .map(|&(gi, lo, hi)| task_scratch(idx, &wins[gi], mo, lo, hi))
        .collect();
    (ms(t.elapsed()), reduce(out))
}

/// Wall arm 3: the committed design -- the shared buffer filled SEQUENTIALLY on the
/// calling thread before the pool starts.
fn wall_buffer_serial(
    idx: &FragIndex,
    wins: &[Win],
    mo: &MassOff,
    tasks: &[(usize, u32, u32)],
    per_window: &[usize],
) -> (f64, u64) {
    let t = std::time::Instant::now();
    let bins: Vec<Option<Vec<u32>>> = wins
        .iter()
        .enumerate()
        .map(|(gi, w)| (per_window[gi] > 1).then(|| window_bins(idx, w, mo)))
        .collect();
    let out: Vec<u64> = tasks
        .par_iter()
        .map(|&(gi, lo, hi)| match &bins[gi] {
            Some(b) => task_buffered(idx, &wins[gi], b, mo, lo, hi),
            None => task_scratch(idx, &wins[gi], mo, lo, hi),
        })
        .collect();
    (ms(t.elapsed()), reduce(out))
}

/// Wall arm 4: the same buffer, filled ON THE POOL. The fill is per (window, scan) so it
/// still parallelises when the batch holds one wide window, which is the shape that
/// amortises the buffer best and the shape a per-window fill cannot spread.
fn wall_buffer_parallel(
    idx: &FragIndex,
    wins: &[Win],
    mo: &MassOff,
    tasks: &[(usize, u32, u32)],
    per_window: &[usize],
) -> (f64, u64) {
    let t = std::time::Instant::now();
    let bins: Vec<Option<Vec<u32>>> = wins
        .par_iter()
        .enumerate()
        .map(|(gi, w)| {
            (per_window[gi] > 1).then(|| {
                let mut b = vec![0u32; w.peaks.len()];
                b.par_chunks_mut(SCAN)
                    .zip(w.peaks.par_chunks(SCAN))
                    .for_each(|(out, scan)| {
                        for (o, p) in out.iter_mut().zip(scan) {
                            *o = idx.bin_of(p.mz / mo.factor_at(p.mz));
                        }
                    });
                b
            })
        })
        .collect();
    let out: Vec<u64> = tasks
        .par_iter()
        .map(|&(gi, lo, hi)| match &bins[gi] {
            Some(b) => task_buffered(idx, &wins[gi], b, mo, lo, hi),
            None => task_scratch(idx, &wins[gi], mo, lo, hi),
        })
        .collect();
    (ms(t.elapsed()), reduce(out))
}

fn wall_shape(idx: &FragIndex, n_cand: usize, n_windows: usize, n_scans: usize, label: &str) {
    let mo = MassOff::scalar(3.7);
    let threads = rayon::current_num_threads();
    let span = (n_cand / n_windows) as u32;
    let wins: Vec<Win> = (0..n_windows)
        .map(|i| Win {
            lo_cid: i as u32 * span,
            hi_cid: ((i + 1) as u32 * span).min(n_cand as u32),
            peaks: sorted_scans(n_scans, SCAN, 0x51ed_0000 + i as u64),
        })
        .collect();
    let (tasks, per_window) = task_shape(&wins, threads);
    let n_peaks: usize = wins.iter().map(|w| w.peaks.len()).sum();
    let n_probes: usize = tasks
        .iter()
        .map(|&(gi, _, _)| wins[gi].peaks.len())
        .sum::<usize>();
    println!(
        "--- WALL {label}: {threads} threads, {n_windows} windows x {n_scans} scans x {SCAN} peaks \
         = {} Mpeak; {} tasks, tasks/window {:?}..{:?}, {} Mprobe",
        n_peaks / 1_000_000,
        tasks.len(),
        per_window.iter().min().unwrap(),
        per_window.iter().max().unwrap(),
        n_probes / 1_000_000
    );
    let (_, h1) = wall_in_probe(idx, &wins, &mo, &tasks);
    let (_, h2) = wall_scratch(idx, &wins, &mo, &tasks);
    let (_, h3) = wall_buffer_serial(idx, &wins, &mo, &tasks, &per_window);
    let (_, h4) = wall_buffer_parallel(idx, &wins, &mo, &tasks, &per_window);
    assert_eq!(h1, h2, "wall arms in-probe/scratch disagree ({label})");
    assert_eq!(
        h1, h3,
        "wall arms in-probe/buffer-serial disagree ({label})"
    );
    assert_eq!(h1, h4, "wall arms in-probe/buffer-par disagree ({label})");
    let reps = 9usize;
    let (mut a, mut b, mut c, mut d) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in 0..reps {
        let mut step = [0usize, 1, 2, 3];
        step.rotate_left(r % 4);
        for s in step {
            match s {
                0 => a.push(wall_in_probe(idx, &wins, &mo, &tasks).0),
                1 => b.push(wall_scratch(idx, &wins, &mo, &tasks).0),
                2 => c.push(wall_buffer_serial(idx, &wins, &mo, &tasks, &per_window).0),
                _ => d.push(wall_buffer_parallel(idx, &wins, &mo, &tasks, &per_window).0),
            }
        }
    }
    let (amin, amed) = stats(&mut a);
    let (bmin, bmed) = stats(&mut b);
    let (cmin, cmed) = stats(&mut c);
    let (dmin, dmed) = stats(&mut d);
    let pct = |x: f64, base: f64| 100.0 * (x - base) / base;
    println!("  in-probe (pre-commit)  min {amin:8.1} med {amed:8.1} ms");
    println!(
        "  per-scan scratch       min {bmin:8.1} med {bmed:8.1} ms  delta min {:+6.1}% med {:+6.1}%",
        pct(bmin, amin),
        pct(bmed, amed)
    );
    println!(
        "  shared buffer, serial  min {cmin:8.1} med {cmed:8.1} ms  delta min {:+6.1}% med {:+6.1}%  (vs scratch {:+6.1}%)",
        pct(cmin, amin),
        pct(cmed, amed),
        pct(cmin, bmin)
    );
    println!(
        "  shared buffer, on pool min {dmin:8.1} med {dmed:8.1} ms  delta min {:+6.1}% med {:+6.1}%  (vs scratch {:+6.1}%)",
        pct(dmin, amin),
        pct(dmed, amed),
        pct(dmin, bmin)
    );
}

/// Wall clock of the four designs under extract's own task shape, at the shape the
/// DEFAULT configuration produces (a batch of `windows_in_flight` narrow windows) and at
/// the shape that amortises a shared buffer best (one wide all-ion window per batch).
#[test]
#[ignore]
fn bench_wall() {
    let n_cand = 200_000usize;
    let lib = build_lib(n_cand, 12);
    let idx = FragIndex::build(&lib, TOL);
    // Default: `windows_in_flight` = min(threads, 16) narrow windows in the batch.
    wall_shape(
        &idx,
        n_cand,
        16,
        512,
        "default (16 narrow windows in flight)",
    );
    // A 50-window scheme leaves the same 16 per batch but each window covers less of the
    // candidate axis, so `MIN_CANDIDATES_PER_TASK` splits it into fewer tasks.
    wall_shape(
        &idx,
        50_000,
        16,
        512,
        "narrow windows, small candidate span",
    );
    // All-ion: one window per batch, so `tasks_per_window` is the whole fan-out.
    wall_shape(&idx, n_cand, 1, 1024, "all-ion (one wide window in flight)");
}
