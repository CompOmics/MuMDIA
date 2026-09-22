//! Microbenchmark for the fragment-index probe loop, the hottest loop in the engine.
//! `#[ignore]`d, so `cargo test` never runs it; it is a harness for re-deriving the
//! measurements the comments in `fragindex.rs` and `extract.rs` cite, not an assertion.
//!
//!   cd rust/mumdia
//!   cargo test --release --test bench_fragindex -- --ignored --nocapture --test-threads=1
//!
//! What it separates is the thing that decides whether hoisting the per-peak setup out
//! of the probe pays: how much of a probe is the setup, and how much is the posting
//! verify. That ratio is set by how many postings the candidate window leaves in the
//! probed bins, so the three arms sweep it (1.6, 8.2 and 65 emitted postings per peak).
//! Every arm asserts the same checksum, so an arm that computes something else fails
//! rather than winning.
//!
//! The machine is shared, so each arm is timed 12 times in a rotating order and the
//! min is the statistic to read; a single round separates arms 5% apart at best.

use mumdia::index::{Candidate, Library};
use mumdia::matchers::fragindex::FragIndex;

const TOL: f64 = 20.0;

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

fn peaks(n: usize) -> Vec<Pk> {
    let mut rng = Lcg(0xdead_beef_cafe_1234);
    (0..n)
        .map(|_| Pk {
            mz: 150.0 + 1650.0 * rng.unit(),
            intensity: rng.unit() as f32,
        })
        .collect()
}

#[inline]
fn factor() -> f64 {
    1.0 + 3.7 * 1e-6
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Arm A: the shipped path. Query m/z and bin computed per peak inside the loop.
fn arm_in_probe(idx: &FragIndex, pk: &[Pk], lo: u32, hi: u32) -> (f64, u64) {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    let t = std::time::Instant::now();
    for p in pk {
        let inten = p.intensity;
        let q = p.mz / factor();
        idx.probe_peak_win(&mut nw, q, |cid, pmz, pint, frag| {
            acc = acc
                .wrapping_add(cid as u64)
                .wrapping_add(pmz.to_bits())
                .wrapping_add((pint + inten).to_bits() as u64)
                .wrapping_add(frag as u64);
        });
    }
    (ms(t.elapsed()), acc)
}

/// Arm B: query m/z and bin read from a precomputed per-window buffer. The peak
/// array is still streamed, for the intensity, so the buffer is EXTRA traffic.
fn arm_precomputed(idx: &FragIndex, pk: &[Pk], qb: &[(f64, u32)], lo: u32, hi: u32) -> (f64, u64) {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    let t = std::time::Instant::now();
    for (p, &(q, b)) in pk.iter().zip(qb) {
        let inten = p.intensity;
        idx.probe_peak_win_binned(&mut nw, q, b, |cid, pmz, pint, frag| {
            acc = acc
                .wrapping_add(cid as u64)
                .wrapping_add(pmz.to_bits())
                .wrapping_add((pint + inten).to_bits() as u64)
                .wrapping_add(frag as u64);
        });
    }
    (ms(t.elapsed()), acc)
}

/// Arm C: only the BIN is precomputed (u32, 4 B/peak); the query m/z is recomputed.
fn arm_bin_only(idx: &FragIndex, pk: &[Pk], bb: &[u32], lo: u32, hi: u32) -> (f64, u64) {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    let t = std::time::Instant::now();
    for (p, &b) in pk.iter().zip(bb) {
        let inten = p.intensity;
        let q = p.mz / factor();
        idx.probe_peak_win_binned(&mut nw, q, b, |cid, pmz, pint, frag| {
            acc = acc
                .wrapping_add(cid as u64)
                .wrapping_add(pmz.to_bits())
                .wrapping_add((pint + inten).to_bits() as u64)
                .wrapping_add(frag as u64);
        });
    }
    (ms(t.elapsed()), acc)
}

/// Arm D: what extract actually ships -- bin buffer behind a per-peak `Option` branch
/// (the buffer is absent for a window that was not split into several tasks).
fn arm_bin_branch(idx: &FragIndex, pk: &[Pk], bb: Option<&[u32]>, lo: u32, hi: u32) -> (f64, u64) {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    let t = std::time::Instant::now();
    for (i, p) in pk.iter().enumerate() {
        let inten = p.intensity;
        let q = p.mz / factor();
        let b = match bb {
            Some(s) => s[i],
            None => idx.bin_of(q),
        };
        idx.probe_peak_win_binned(&mut nw, q, b, |cid, pmz, pint, frag| {
            acc = acc
                .wrapping_add(cid as u64)
                .wrapping_add(pmz.to_bits())
                .wrapping_add((pint + inten).to_bits() as u64)
                .wrapping_add(frag as u64);
        });
    }
    (ms(t.elapsed()), acc)
}

/// Arm F: no per-window buffer, but the inner loop still zips a bins SLICE -- the task
/// fills a small reusable scratch once per scan. Models the unsplit window under the
/// branch-free design.
fn arm_scratch(idx: &FragIndex, pk: &[Pk], lo: u32, hi: u32) -> (f64, u64) {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    let mut scratch: Vec<u32> = Vec::new();
    let t = std::time::Instant::now();
    for scan in pk.chunks(300) {
        scratch.clear();
        scratch.extend(scan.iter().map(|p| idx.bin_of(p.mz / factor())));
        for (p, &b) in scan.iter().zip(&scratch) {
            let inten = p.intensity;
            let q = p.mz / factor();
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

/// Arm G: arm C, but walked scan by scan like the engine does.
fn arm_bin_chunked(idx: &FragIndex, pk: &[Pk], bb: &[u32], lo: u32, hi: u32) -> (f64, u64) {
    let mut nw = idx.window_narrow(lo, hi);
    let mut acc = 0u64;
    let t = std::time::Instant::now();
    for (scan, bins) in pk.chunks(300).zip(bb.chunks(300)) {
        for (p, &b) in scan.iter().zip(bins) {
            let inten = p.intensity;
            let q = p.mz / factor();
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

fn stats(v: &mut [f64]) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2])
}

fn run_arm(idx: &FragIndex, pk: &[Pk], lo: u32, hi: u32, label: &str) {
    let qb: Vec<(f64, u32)> = pk
        .iter()
        .map(|p| {
            let q = p.mz / factor();
            (q, idx.bin_of(q))
        })
        .collect();
    let bb: Vec<u32> = qb.iter().map(|&(_, b)| b).collect();
    // warm up every path so no arm pays another's cold caches
    let (_, ha) = arm_in_probe(idx, pk, lo, hi);
    let (_, hb) = arm_precomputed(idx, pk, &qb, lo, hi);
    let (_, hc) = arm_bin_only(idx, pk, &bb, lo, hi);
    let (_, hd) = arm_bin_branch(idx, pk, Some(&bb), lo, hi);
    let (_, hf) = arm_scratch(idx, pk, lo, hi);
    let (_, hg) = arm_bin_chunked(idx, pk, &bb, lo, hi);
    assert_eq!(ha, hb, "arms A/B disagree ({label})");
    assert_eq!(ha, hc, "arms A/C disagree ({label})");
    assert_eq!(ha, hd, "arms A/D disagree ({label})");
    assert_eq!(ha, hf, "arms A/F disagree ({label})");
    assert_eq!(ha, hg, "arms A/G disagree ({label})");
    let reps = 12usize;
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    let mut d = Vec::new();
    let mut e = Vec::new();
    let mut fv = Vec::new();
    let mut g = Vec::new();
    for r in 0..reps {
        // rotate the order, so warm-cache position cannot favour one arm
        let mut step = [0usize, 1, 2, 3, 4, 5, 6];
        step.rotate_left(r % 7);
        for s in step {
            match s {
                0 => a.push(arm_in_probe(idx, pk, lo, hi).0),
                1 => b.push(arm_precomputed(idx, pk, &qb, lo, hi).0),
                2 => c.push(arm_bin_only(idx, pk, &bb, lo, hi).0),
                3 => d.push(arm_bin_branch(idx, pk, Some(&bb), lo, hi).0),
                4 => e.push(arm_bin_branch(idx, pk, None, lo, hi).0),
                5 => fv.push(arm_scratch(idx, pk, lo, hi).0),
                _ => g.push(arm_bin_chunked(idx, pk, &bb, lo, hi).0),
            }
        }
    }
    let (amin, amed) = stats(&mut a);
    let (bmin, bmed) = stats(&mut b);
    let (cmin, cmed) = stats(&mut c);
    let (dmin, dmed) = stats(&mut d);
    let (emin, emed) = stats(&mut e);
    let (fmin, fmed) = stats(&mut fv);
    let (gmin, gmed) = stats(&mut g);
    let n = pk.len() as f64;
    println!(
        "  A in-probe      min {amin:7.2} med {amed:7.2} ms ({:6.1} ns/peak)",
        amed * 1e6 / n
    );
    println!(
        "  B q_mz+bin buf  min {bmin:7.2} med {bmed:7.2} ms ({:6.1} ns/peak)  delta min {:+6.1}% med {:+6.1}%",
        bmed * 1e6 / n,
        100.0 * (bmin - amin) / amin,
        100.0 * (bmed - amed) / amed
    );
    println!(
        "  C bin-only buf  min {cmin:7.2} med {cmed:7.2} ms ({:6.1} ns/peak)  delta min {:+6.1}% med {:+6.1}%",
        cmed * 1e6 / n,
        100.0 * (cmin - amin) / amin,
        100.0 * (cmed - amed) / amed
    );
    println!(
        "  D bin buf+branch min {dmin:6.2} med {dmed:7.2} ms ({:6.1} ns/peak)  delta min {:+6.1}% med {:+6.1}%",
        dmed * 1e6 / n,
        100.0 * (dmin - amin) / amin,
        100.0 * (dmed - amed) / amed
    );
    println!(
        "  E no buf+branch min {emin:6.2} med {emed:7.2} ms ({:6.1} ns/peak)  delta min {:+6.1}% med {:+6.1}%  <- unsplit window, cost of the branch alone",
        emed * 1e6 / n,
        100.0 * (emin - amin) / amin,
        100.0 * (emed - amed) / amed
    );
    println!(
        "  F scratch/scan  min {fmin:7.2} med {fmed:7.2} ms ({:6.1} ns/peak)  delta min {:+6.1}% med {:+6.1}%  <- unsplit window, branch-free design",
        fmed * 1e6 / n,
        100.0 * (fmin - amin) / amin,
        100.0 * (fmed - amed) / amed
    );
    println!(
        "  G bin buf/scan  min {gmin:7.2} med {gmed:7.2} ms ({:6.1} ns/peak)  delta min {:+6.1}% med {:+6.1}%  <- split window, branch-free design",
        gmed * 1e6 / n,
        100.0 * (gmin - amin) / amin,
        100.0 * (gmed - amed) / amed
    );
}

fn setup_cost(idx: &FragIndex, pk: &[Pk]) {
    let mut div = f64::INFINITY;
    let mut divbin = f64::INFINITY;
    for _ in 0..9 {
        let t = std::time::Instant::now();
        let mut s = 0.0f64;
        for p in pk {
            s += p.mz / factor();
        }
        std::hint::black_box(s);
        div = div.min(ms(t.elapsed()));

        let t = std::time::Instant::now();
        let mut s = 0u64;
        for p in pk {
            let q = p.mz / factor();
            s = s.wrapping_add(idx.bin_of(q) as u64);
        }
        std::hint::black_box(s);
        divbin = divbin.min(ms(t.elapsed()));
    }
    let n = pk.len() as f64;
    println!(
        "setup only: divide {:.3} ns/peak, divide+bin {:.3} ns/peak => bin alone {:.3} ns/peak",
        div * 1e6 / n,
        divbin * 1e6 / n,
        (divbin - div) * 1e6 / n
    );
}

fn count_hits(idx: &FragIndex, pk: &[Pk], lo: u32, hi: u32) -> f64 {
    let mut nw = idx.window_narrow(lo, hi);
    let mut n = 0u64;
    for p in pk {
        idx.probe_peak_win(&mut nw, p.mz / factor(), |_, _, _, _| n += 1);
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
    let pk = peaks(400_000);
    setup_cost(&idx, &pk);
    for &(w_lo, w_hi, name) in &[
        (700.0f64, 720.0f64, "narrow (20 m/z DIA window)"),
        (700.0, 800.0, "medium (100 m/z band)"),
        (0.0, 1e9, "wide (all-ion / whole band)"),
    ] {
        let (lo, hi) = idx.candidate_range(w_lo, w_hi);
        println!(
            "--- {name}: {} candidates, {:.2} hits/peak",
            hi - lo,
            count_hits(&idx, &pk, lo, hi)
        );
        run_arm(&idx, &pk, lo, hi, name);
    }
}
